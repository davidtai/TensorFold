#pragma once

// GLM-5.3-Flash's glue on HIP: mHC (4 residual streams of 4,096, Sinkhorn mixing), the plain residual of the MTP
// layer, RMS norms with bf16 weights, the stream mean and the embedding into streams. The arithmetic is the CUDA
// family's (kernels/cuda/glm_glue_hc.cuh, glm_glue_rows.cuh, glm_glue_norm.cuh): the same formulas and rounding points
// to bf16, in this file's own reduction order (the bits are not Triton's).

#include "ops/common.hpp"

namespace tf {
namespace rocm {
namespace glm {

constexpr int kD = 4096;         // hidden
constexpr int kS = 4;            // hc_mult: residual streams
constexpr int kWide = kS * kD;   // a row of the streams
constexpr int kMix = 24;         // 4 pre, 4 post, 16 comb
constexpr int kBlocks = 16;      // K blocks of the mixing dots (as glue.HC_BLOCKS)
constexpr int kKb = kWide / kBlocks;
constexpr int kPart = 32;        // a block's partials: 24 dots, the sum of squares at 24
constexpr int kThreads = 256;
constexpr int kWaves = kThreads / 32;
constexpr int kMixRows = 4;      // rows a mixing block

__device__ __forceinline__ float bf(uint16_t h) { return bf16_to_float(h); }
__device__ __forceinline__ float rbf(float v) { return bf16_to_float(float_to_bf16(v)); }

__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    return v;
}

// The block's sum of one value a thread: each wave's butterfly, then the waves' sums in order.
__device__ __forceinline__ float block_sum(float v, float* scratch) {
    v = wave_sum(v);
    if ((threadIdx.x & 31) == 0) scratch[threadIdx.x >> 5] = v;
    __syncthreads();
    float total = 0.f;
#pragma unroll
    for (int w = 0; w < kWaves; ++w) total += scratch[w];
    __syncthreads();
    return total;
}

__device__ __forceinline__ float sigmoid1(float v) { return 1.f / (1.f + expf(-v)); }

// The kernels below stay in this namespace (extern "C" keeps their names): other lanes' headers share the group.

// hc pre, part 1: block (row group, K block b) of 256 threads, 4 rows. part[(row * 16 + b) * 32 + m] = x[row] . fn[m]
// over the block's 1,024 inputs (m < 24), and at 24 the sum of x^2. A thread takes 4 contiguous inputs.
extern "C" __global__ void __launch_bounds__(kThreads)
    tf_glm_hc_mix(const uint16_t* __restrict__ x, const uint16_t* __restrict__ fn, float* __restrict__ part, int rows) {
    __shared__ float red[kWaves][kMixRows][kMix + 1];
    const int r0 = blockIdx.x * kMixRows, b = blockIdx.y, t = threadIdx.x, lane = t & 31, wave = t >> 5;
    const long long base = static_cast<long long>(b) * kKb + t * 4;
    float xv[kMixRows][4];
    float ss[kMixRows];
#pragma unroll
    for (int q = 0; q < kMixRows; ++q) {
        const int row = r0 + q < rows ? r0 + q : rows - 1;
        const uint2 u = *reinterpret_cast<const uint2*>(x + static_cast<long long>(row) * kWide + base);
        xv[q][0] = bf(u.x & 0xffff), xv[q][1] = bf(u.x >> 16), xv[q][2] = bf(u.y & 0xffff), xv[q][3] = bf(u.y >> 16);
        ss[q] = xv[q][0] * xv[q][0] + xv[q][1] * xv[q][1] + xv[q][2] * xv[q][2] + xv[q][3] * xv[q][3];
        ss[q] = wave_sum(ss[q]);
    }
    for (int m = 0; m < kMix; ++m) {
        const uint2 u = *reinterpret_cast<const uint2*>(fn + static_cast<long long>(m) * kWide + base);
        const float w0 = bf(u.x & 0xffff), w1 = bf(u.x >> 16), w2 = bf(u.y & 0xffff), w3 = bf(u.y >> 16);
#pragma unroll
        for (int q = 0; q < kMixRows; ++q) {
            float d = xv[q][0] * w0;
            d = fmaf(xv[q][1], w1, d);
            d = fmaf(xv[q][2], w2, d);
            d = fmaf(xv[q][3], w3, d);
            d = wave_sum(d);
            if (lane == 0) red[wave][q][m] = d;
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int q = 0; q < kMixRows; ++q) red[wave][q][kMix] = ss[q];
    }
    __syncthreads();
    if (t < kMixRows * (kMix + 1)) {
        const int q = t / (kMix + 1), m = t % (kMix + 1);
        if (r0 + q < rows) {
            float total = 0.f;
#pragma unroll
            for (int w = 0; w < kWaves; ++w) total += red[w][q][m];
            part[(static_cast<long long>(r0 + q) * kBlocks + b) * kPart + m] = total;
        }
    }
}

// hc pre, part 2: a block a row. The 16 blocks' partials summed; mix = dots / rms(x) over all 16,384 inputs;
//   pre[k]  = hc_eps + sigmoid(mix[k] scale[0] + base[k]),        k < 4
//   post[k] = 2 sigmoid(mix[4 + k] scale[1] + base[4 + k])
//   comb    = Sinkhorn(mix[8..24] scale[2] + base[8..24]) as 4 x 4 [i][j]: softmax over j + hc_eps, then columns
//             (over i) normed, then iters - 1 times rows and columns, each sum + hc_eps;
// then the collapse c = bf16(sum_k pre[k] x[k]) and out = bf16(w bf16(c / rms(c))). post [row][4], comb [row][16].
extern "C" __global__ void __launch_bounds__(kThreads)
    tf_glm_hc_finish(const uint16_t* __restrict__ x, const float* __restrict__ part, const float* __restrict__ base,
                     const float* __restrict__ scale, const uint16_t* __restrict__ nw, uint16_t* __restrict__ out,
                     float* __restrict__ post, float* __restrict__ comb, float eps, float hc_eps, int iters) {
    __shared__ float tot[kMix + 1];
    __shared__ float pre[kS];
    __shared__ float scratch[kWaves];
    const long long r = blockIdx.x;
    const int t = threadIdx.x;
    if (t <= kMix) {
        const float* p = part + r * kBlocks * kPart + t;
        float a = 0.f;
#pragma unroll
        for (int b = 0; b < kBlocks; ++b) a += p[b * kPart];
        tot[t] = a;
    }
    __syncthreads();
    if (t == 0) {
        const float rinv = 1.f / sqrtf(eps + tot[kMix] / static_cast<float>(kWide));
        float mix[kMix];
#pragma unroll
        for (int m = 0; m < kMix; ++m) mix[m] = tot[m] * rinv;
#pragma unroll
        for (int k = 0; k < kS; ++k) {
            pre[k] = hc_eps + sigmoid1(fmaf(mix[k], scale[0], base[k]));
            post[r * kS + k] = 2.f * sigmoid1(fmaf(mix[kS + k], scale[1], base[kS + k]));
        }
        float cm[kS][kS];
#pragma unroll
        for (int i = 0; i < kS; ++i) {
            float mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < kS; ++j) {
                cm[i][j] = fmaf(mix[2 * kS + i * kS + j], scale[2], base[2 * kS + i * kS + j]);
                mx = fmaxf(mx, cm[i][j]);
            }
            float sum = 0.f;
#pragma unroll
            for (int j = 0; j < kS; ++j) {
                cm[i][j] = expf(cm[i][j] - mx);
                sum += cm[i][j];
            }
#pragma unroll
            for (int j = 0; j < kS; ++j) cm[i][j] = cm[i][j] / sum + hc_eps;
        }
        for (int it = 0; it < iters; ++it) {
            if (it > 0) {
#pragma unroll
                for (int i = 0; i < kS; ++i) {
                    const float sum = cm[i][0] + cm[i][1] + cm[i][2] + cm[i][3] + hc_eps;
#pragma unroll
                    for (int j = 0; j < kS; ++j) cm[i][j] = cm[i][j] / sum;
                }
            }
#pragma unroll
            for (int j = 0; j < kS; ++j) {
                const float sum = cm[0][j] + cm[1][j] + cm[2][j] + cm[3][j] + hc_eps;
#pragma unroll
                for (int i = 0; i < kS; ++i) cm[i][j] = cm[i][j] / sum;
            }
        }
#pragma unroll
        for (int i = 0; i < kS; ++i) {
#pragma unroll
            for (int j = 0; j < kS; ++j) comb[r * kS * kS + i * kS + j] = cm[i][j];
        }
    }
    __syncthreads();
    const float p0 = pre[0], p1 = pre[1], p2 = pre[2], p3 = pre[3];
    const uint16_t* row = x + r * kWide;
    constexpr int kPer = kD / kThreads;
    float c[kPer];
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
        const int d = t + j * kThreads;
        float m = p0 * bf(row[d]);
        m = fmaf(p1, bf(row[kD + d]), m);
        m = fmaf(p2, bf(row[2 * kD + d]), m);
        m = fmaf(p3, bf(row[3 * kD + d]), m);
        c[j] = rbf(m);
        acc = fmaf(c[j], c[j], acc);
    }
    const float rinv = 1.f / sqrtf(eps + block_sum(acc, scratch) / static_cast<float>(kD));
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
        const int d = t + j * kThreads;
        out[r * kD + d] = float_to_bf16(bf(nw[d]) * rbf(rinv * c[j]));
    }
}

// hc post: branch = bf16(the `parts` fp32 partials summed in order, `stride` apart); stream s of the row becomes
// bf16(post[s] branch + sum_t comb[t][s] x[t]). Block (row, 256 hidden columns); x may be xout.
extern "C" __global__ void __launch_bounds__(kThreads)
    tf_glm_hc_post(const uint16_t* x, uint16_t* xout, const float* __restrict__ g, int parts, long long stride,
                   const float* __restrict__ post, const float* __restrict__ comb) {
    const long long r = blockIdx.x;
    const int d = blockIdx.y * kThreads + threadIdx.x;
    float branch = g[r * kD + d];
    for (int p = 1; p < parts; ++p) branch += g[p * stride + r * kD + d];
    branch = rbf(branch);
    float xv[kS];
#pragma unroll
    for (int s = 0; s < kS; ++s) xv[s] = bf(x[r * kWide + s * kD + d]);
    const float* cm = comb + r * kS * kS;
#pragma unroll
    for (int s = 0; s < kS; ++s) {
        float m = xv[1] * cm[kS + s];
        m = fmaf(xv[0], cm[s], m);
        m = fmaf(xv[2], cm[2 * kS + s], m);
        m = fmaf(xv[3], cm[3 * kS + s], m);
        xout[r * kWide + s * kD + d] = float_to_bf16(fmaf(branch, post[r * kS + s], m));
    }
}

// The final collapse: out [row][4096] = bf16 of the 4 streams' mean.
extern "C" __global__ void __launch_bounds__(kThreads)
    tf_glm_stream_mean(const uint16_t* __restrict__ x, uint16_t* __restrict__ out) {
    const long long r = blockIdx.x;
    const int d = blockIdx.y * kThreads + threadIdx.x;
    const uint16_t* row = x + r * kWide + d;
    const float sum = ((bf(row[0]) + bf(row[kD])) + bf(row[2 * kD])) + bf(row[3 * kD]);
    out[r * kD + d] = float_to_bf16(sum / static_cast<float>(kS));
}

// The MTP layer's plain residual: xout = bf16(x + bf16(the `parts` fp32 partials summed in order)); x may be xout.
extern "C" __global__ void __launch_bounds__(kThreads)
    tf_glm_residual(const uint16_t* x, uint16_t* xout, const float* __restrict__ g, int parts, long long stride, int d) {
    const long long r = blockIdx.x;
    const int c = blockIdx.y * kThreads + threadIdx.x;
    if (c >= d) return;
    const long long i = r * d + c;
    float a = g[i];
    for (int p = 1; p < parts; ++p) a += g[p * stride + i];
    xout[i] = float_to_bf16(bf(x[i]) + rbf(a));
}

// glue.rmsnorm: out = bf16(w bf16(x / sqrt(mean(x^2) + eps))), bf16 weights as stored; rows `x_stride` and
// `o_stride` elements apart (the MTP layer's enorm and hnorm write the halves of one [e | h] row). A block a row.
extern "C" __global__ void __launch_bounds__(kThreads)
    tf_glm_rms(const uint16_t* __restrict__ x, long long x_stride, const uint16_t* __restrict__ w,
               uint16_t* __restrict__ out, long long o_stride, int width, float eps) {
    __shared__ float scratch[kWaves];
    const long long r = blockIdx.x;
    const uint16_t* row = x + r * x_stride;
    float acc = 0.f;
    for (int i = threadIdx.x; i < width; i += kThreads) {
        const float v = bf(row[i]);
        acc = fmaf(v, v, acc);
    }
    const float rinv = 1.f / sqrtf(eps + block_sum(acc, scratch) / static_cast<float>(width));
    for (int i = threadIdx.x; i < width; i += kThreads) out[r * o_stride + i] = float_to_bf16(bf(w[i]) * rbf(rinv * bf(row[i])));
}

// glue.embed of a BF16 table: token ids[row]'s row into `copies` streams of d: out [row][copies][d].
extern "C" __global__ void __launch_bounds__(kThreads)
    tf_glm_embed(const int* __restrict__ ids, const uint16_t* __restrict__ table, uint16_t* __restrict__ out, int d,
                 int copies) {
    const long long r = blockIdx.x;
    const int c = blockIdx.y * kThreads + threadIdx.x;
    if (c >= d) return;
    const uint16_t v = table[static_cast<long long>(ids[r]) * d + c];
    for (int k = 0; k < copies; ++k) out[(r * copies + k) * d + c] = v;
}

}  // namespace glm
}  // namespace rocm
}  // namespace tf
