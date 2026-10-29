#pragma once

#include "common.hpp"
#include "draw.hpp"

// Tensor-parallel forward ops: residual add of an fp32 sum, expert-share remap and masks; MTP token probability.

namespace {

__device__ __forceinline__ long long tp_gid() { return static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; }

constexpr int kProbThreads = 256;

}  // namespace

// out = round(float(x) + y): the residual of a rank-summed fp32 share.
extern "C" __global__ void tf_add_wide(const void* x, const float* y, void* out, int kind, long long n) {
    long long i = tp_gid();
    if (i >= n) return;
    if (kind == 1) {
        float v = __half2float(static_cast<const __half*>(x)[i]) + y[i];
        static_cast<__half*>(out)[i] = __float2half_rn(v);
    } else {
        float v = bf16_to_float(static_cast<const uint16_t*>(x)[i]) + y[i];
        static_cast<uint16_t*>(out)[i] = float_to_bf16(v);
    }
}

// out[i] = the rank's local id of expert pick[i], or `skip` (an id past the rank's experts) when another rank holds it.
extern "C" __global__ void tf_moe_localize(const int* pick, const int* remap, int* out, int n, int skip) {
    int i = static_cast<int>(tp_gid());
    if (i >= n) return;
    int local = remap[pick[i]];
    out[i] = local >= 0 ? local : skip;
}

// An item of the skip expert does no work.
extern "C" __global__ void tf_moe_foreign_items(int* items, int count, int skip) {
    int i = static_cast<int>(tp_gid());
    if (i < count && items[i * 3] == skip) items[i * 3 + 2] = 0;
}

// A pair another rank's expert answers has zero rows (selected, not multiplied: the buffer holds anything).
extern "C" __global__ void tf_moe_zero_foreign(float* y, const int* picks, int skip, long long pairs, int d) {
    long long i = tp_gid();
    if (i >= pairs * d) return;
    if (picks[i / d] == skip) y[i] = 0.f;
}

// out[r] = softmax(row r)[id[r]], or of the row's largest value when `id` is null; one block a row, a NaN in gives NaN.
extern "C" __global__ void tf_token_prob(const uint16_t* logits, int kind, int n, const int* id, float* out) {
    const uint16_t* row = logits + static_cast<long long>(blockIdx.x) * n;
    __shared__ float shared[kProbThreads];
    float top = -INFINITY;
    bool nan = false;
    for (int i = threadIdx.x; i < n; i += kProbThreads) {
        float v = widen(row[i], kind);
        nan |= v != v;
        top = fmaxf(top, v);
    }
    shared[threadIdx.x] = nan ? NAN : top;
    __syncthreads();
    for (int stride = kProbThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float a = shared[threadIdx.x], b = shared[threadIdx.x + stride];
            shared[threadIdx.x] = (a != a || b != b) ? NAN : fmaxf(a, b);
        }
        __syncthreads();
    }
    const float peak = shared[0];
    __syncthreads();
    float total = 0.f;
    for (int i = threadIdx.x; i < n; i += kProbThreads) total += expf(widen(row[i], kind) - peak);
    shared[threadIdx.x] = total;
    __syncthreads();
    for (int stride = kProbThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float chosen = id ? expf(widen(row[id[blockIdx.x]], kind) - peak) : 1.f;
        out[blockIdx.x] = chosen / shared[0];
    }
}

namespace {

constexpr int kProbAhead = 16;

// tf_token_prob's second pass for lane t of a row (t, t + 256, ... in its order), kProbAhead loads in flight.
__device__ __forceinline__ float prob_lane_sum(const uint16_t* row, int kind, int n, float peak, int t) {
    float total = 0.f;
    int i = t;
    for (; i + (kProbAhead - 1) * kProbThreads < n; i += kProbAhead * kProbThreads) {
        uint16_t v[kProbAhead];
#pragma unroll
        for (int k = 0; k < kProbAhead; ++k) v[k] = row[i + k * kProbThreads];
#pragma unroll
        for (int k = 0; k < kProbAhead; ++k) total += expf(widen(v[k], kind) - peak);
    }
    for (; i < n; i += kProbThreads) total += expf(widen(row[i], kind) - peak);
    return total;
}

}  // namespace

// tf_token_prob when `id` is the row's tf_argmax: the row's max is row[id] (a NaN anywhere is the argmax, as the max
// pass gave NaN), so only the sum pass runs, in tf_token_prob's lanes, order and tree. One block a row. (W2 H-2 bench)
extern "C" __global__ void __launch_bounds__(kProbThreads)
    tf_token_prob_peak(const uint16_t* logits, int kind, int n, const int* id, float* out) {
    const uint16_t* row = logits + static_cast<long long>(blockIdx.x) * n;
    __shared__ float shared[kProbThreads];
    const float peak = widen(row[id[blockIdx.x]], kind);
    shared[threadIdx.x] = prob_lane_sum(row, kind, n, peak, threadIdx.x);
    __syncthreads();
    for (int stride = kProbThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = expf(widen(row[id[blockIdx.x]], kind) - peak) / shared[0];
}

// tf_token_prob_peak with a row's 256 lanes over 8 one-wave blocks (grid 8 x rows): each lane's sum as there, into
// `parts` (256 a row); the row's last block (`tickets`, zero at rest, put back to zero) runs the same tree. (W2 H-2)
extern "C" __global__ void __launch_bounds__(32) tf_token_prob_lanes(const uint16_t* logits, int kind, int n,
                                                                     const int* id, float* out, float* parts,
                                                                     unsigned* tickets) {
    const int r = blockIdx.y;
    const uint16_t* row = logits + static_cast<long long>(r) * n;
    const float peak = widen(row[id[r]], kind);
    const int t = blockIdx.x * 32 + threadIdx.x;
    parts[r * kProbThreads + t] = prob_lane_sum(row, kind, n, peak, t);
    __threadfence();
    __shared__ unsigned ticket;
    if (threadIdx.x == 0) ticket = atomicAdd(&tickets[r], 1u);
    __syncthreads();
    if (ticket != gridDim.x - 1) return;
    __threadfence();
    __shared__ float shared[kProbThreads];
    for (int k = threadIdx.x; k < kProbThreads; k += 32) shared[k] = __hip_atomic_load(&parts[r * kProbThreads + k], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __syncthreads();
    for (int stride = kProbThreads / 2; stride > 0; stride >>= 1) {
        for (int k = threadIdx.x; k < stride; k += 32) shared[k] += shared[k + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[r] = expf(widen(row[id[r]], kind) - peak) / shared[0];
        tickets[r] = 0;
    }
}
