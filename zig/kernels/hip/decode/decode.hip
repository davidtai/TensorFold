// Decode-step kernels for a few rows: the MoE router, a wave an expert, and residual tails; sums in the fastest order.

// kind: 1 fp16, 2 bf16 (ops/act.hip's numbering).

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

#include "decode/plan.hpp"
#include "recurrence/kda.hpp"

namespace {

__device__ __forceinline__ float bf16_up(uint32_t bits) { return __uint_as_float(bits << 16); }

// Four consecutive activations (8 bytes) widened.
template <int KIND>
__device__ __forceinline__ float4 widen4(uint2 v) {
    if constexpr (KIND == 1) {
        const __half2 a = __builtin_bit_cast(__half2, v.x);
        const __half2 b = __builtin_bit_cast(__half2, v.y);
        return make_float4(__low2float(a), __high2float(a), __low2float(b), __high2float(b));
    } else {
        return make_float4(bf16_up(v.x & 0xffffu), bf16_up(v.x >> 16), bf16_up(v.y & 0xffffu), bf16_up(v.y >> 16));
    }
}

constexpr int kRouterWaves = 4;
constexpr int kRouterSteps = 16;  // float4 loads a lane keeps in flight
constexpr int kRouterRows = 16;

// Four router weights at `at` of row `w`: fp32 rows as stored, or bf16 rows (ROWS16) widened, which is exact.
template <bool ROWS16>
__device__ __forceinline__ float4 router_weights(const void* w, int at) {
    if constexpr (ROWS16) {
        const uint2 v = *reinterpret_cast<const uint2*>(static_cast<const uint16_t*>(w) + at);
        return make_float4(bf16_up(v.x & 0xffffu), bf16_up(v.x >> 16), bf16_up(v.y & 0xffffu), bf16_up(v.y >> 16));
    } else {
        return *reinterpret_cast<const float4*>(static_cast<const float*>(w) + at);
    }
}

// logits[r, e] = x[r] . rows[e], 16 rows a block, a wave an expert; d is a multiple of 4, weight rows 16-byte aligned
// (8-byte for ROWS16: bf16 rows, the same values as the fp32 rows widened from them, so the same logits).
// `staged`: x's rows of this block are already in shared memory at `x` (router_select's few rows), not global memory.
template <int KIND, bool ROWS16 = false>
__device__ void router_decode(const void* x, const void* rows, float* logits, int r, int d, int e, bool staged = false) {
    const int lane = threadIdx.x & 31;
    const int expert = blockIdx.x * kRouterWaves + (threadIdx.x >> 5);
    if (expert >= e) return;
    // blockIdx.y picks a block of 16 rows: a row's sums are the same whichever block it is in
    const int first = blockIdx.y * kRouterRows;
    if (!staged) x = static_cast<const uint16_t*>(x) + static_cast<long long>(first) * d;
    logits += static_cast<long long>(first) * e;
    r = r - first < kRouterRows ? r - first : kRouterRows;
    const void* w = ROWS16 ? static_cast<const void*>(static_cast<const uint16_t*>(rows) + static_cast<long long>(expert) * d)
                           : static_cast<const void*>(static_cast<const float*>(rows) + static_cast<long long>(expert) * d);
    float acc[kRouterRows];
#pragma unroll
    for (int k = 0; k < kRouterRows; ++k) acc[k] = 0.f;
    for (int base = 0; base < d; base += kRouterSteps * 128) {
        float4 wv[kRouterSteps];
#pragma unroll
        for (int s = 0; s < kRouterSteps; ++s) {
            const int at = base + (lane + 32 * s) * 4;
            wv[s] = at < d ? router_weights<ROWS16>(w, at) : make_float4(0.f, 0.f, 0.f, 0.f);
        }
#pragma unroll
        for (int k = 0; k < kRouterRows; ++k) {
            if (k >= r) continue;
            const uint16_t* xr = static_cast<const uint16_t*>(x) + static_cast<long long>(k) * d;
#pragma unroll
            for (int s = 0; s < kRouterSteps; ++s) {
                const int at = base + (lane + 32 * s) * 4;
                if (at >= d) continue;
                const float4 xv = widen4<KIND>(*reinterpret_cast<const uint2*>(xr + at));
                acc[k] = fmaf(xv.x, wv[s].x, acc[k]);
                acc[k] = fmaf(xv.y, wv[s].y, acc[k]);
                acc[k] = fmaf(xv.z, wv[s].z, acc[k]);
                acc[k] = fmaf(xv.w, wv[s].w, acc[k]);
            }
        }
    }
#pragma unroll
    for (int k = 0; k < kRouterRows; ++k) {
        float v = acc[k];
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) v += __shfl_xor(v, mask, 32);
        if (lane == 0 && k < r) logits[static_cast<long long>(k) * e + expert] = v;
    }
}


__device__ __forceinline__ float load_kind(const void* p, int kind, long long i) {
    return kind == 1 ? __half2float(static_cast<const __half*>(p)[i]) : bf16_up(static_cast<const uint16_t*>(p)[i]);
}

// A float rounded to the activation kind (fp16 or bf16, to nearest even) and widened again.
__device__ __forceinline__ float round_kind(float v, int kind) {
    if (kind == 1) return __half2float(__float2half_rn(v));
    if (v != v) return v;
    const uint32_t u = __float_as_uint(v);
    return __uint_as_float(((u + (((u >> 16) & 1u) + 0x7fffu)) >> 16) << 16);
}

__device__ __forceinline__ void store_kind(void* p, int kind, long long i, float v) {
    if (kind == 1) {
        static_cast<__half*>(p)[i] = __float2half_rn(v);
    } else {
        const uint32_t u = __float_as_uint(v);
        static_cast<uint16_t*>(p)[i] = v != v ? 0x7fc0 : static_cast<uint16_t>((u + (((u >> 16) & 1u) + 0x7fffu)) >> 16);
    }
}

constexpr int kTailThreads = 512;
constexpr int kTailWidth = 8192;

// One row a block, one launch: x = round(x + t), t = y or the slot-ordered weighted sum; normed = rms(x) * weight.
__device__ void tail_body(void* x, const void* y, const float* wts, const float* weight, void* normed, int kind,
                          int slots, int width, float eps) {
    __shared__ float row_v[kTailWidth];
    __shared__ float part[kTailThreads / 32];
    const int row = blockIdx.x;
    const long long at = static_cast<long long>(row) * width;
    float sum = 0.f;
    for (int i = threadIdx.x; i < width; i += kTailThreads) {
        float t;
        if (slots > 0) {
            float acc = 0.f;
            for (int s = 0; s < slots; ++s) {
                acc += static_cast<const float*>(y)[(static_cast<long long>(row) * slots + s) * width + i] *
                       wts[row * slots + s];
            }
            t = round_kind(acc, kind);
        } else {
            t = load_kind(y, kind, at + i);
        }
        const float v = round_kind(load_kind(x, kind, at + i) + t, kind);
        store_kind(x, kind, at + i, v);
        row_v[i] = v;
        sum += v * v;
    }
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) sum += __shfl_xor(sum, mask, 32);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sum;
    __syncthreads();
    float total = 0.f;
#pragma unroll
    for (int w = 0; w < kTailThreads / 32; ++w) total += part[w];
    const float inv = rsqrtf(total / static_cast<float>(width) + eps);
    for (int i = threadIdx.x; i < width; i += kTailThreads) {
        float value = row_v[i] * inv;
        if (weight != nullptr) value *= weight[i];
        store_kind(normed, kind, at + i, value);
    }
}

// Conv launch arguments: x (rows, channels), taps, state, snapshots; q, k (rows, kw), v (rows, vw) out; norm, gate.
struct ConvArgs {
    const void* x;
    int kind;
    const float* weight;
    float* state;
    float* states;
    float* qn;
    float* kn;
    float* v;
    int channels, kernel, rows, kw, vw;
    const float* qw;
    const float* kw_w;
    float eps;
    int norm;
    const void* ga;
    const void* gb;
    const float* a_log;
    const float* dt_bias;
    float* gate;
    float* beta;
    int gcount, heads;
    // A round's plan (plan.count null: the launch's own rows and state): a grid row a slot, rows and state from it.
    PlanArgs plan;
    int layer;
    int slots;  // the launcher's grid rows (the grid says it here)
    // with row f >= 0 here, the window starts from the last window's snapshot after its row f (a keep that dropped
    // rows names it instead of copying it back); null or -1: from `state`
    const int* from;
    // 128 (KDA): a decay a key channel (gcount rows x heads x 128, tf_kda::decay with `lower`), beta a head; else the
    // DeltaNet's gate a head
    int gate_dim;
    float lower;
};

// The conv for a window's rows split into q, k, v: taps in order, then silu; q, k heads rms-normed; gate and beta last.
__device__ void conv_split_body(const ConvArgs& launch) {
    __shared__ float sq[128];
    __shared__ float inv_s;
    ConvArgs c = launch;
    float* samples = nullptr;
    const int conv_blocks = (c.channels + 127) / 128;
    if (launch.plan.count != nullptr) {
        const int slot = blockIdx.y;
        // the gates are one pass over every row of the round, from the first slot's blocks
        if (blockIdx.x >= conv_blocks && slot != 0) return;
        if (blockIdx.x < conv_blocks) {
            const int rows = launch.plan.count[slot];
            if (rows == 0) return;
            const long long row0 = launch.plan.first[slot];
            c.rows = rows;
            c.state = reinterpret_cast<float*>(plan_desc(launch.plan, slot)[plan_first(launch.layer)]);
            c.x = static_cast<const char*>(launch.x) + row0 * c.channels * 2;
            c.qn += row0 * c.kw;
            c.kn += row0 * c.kw;
            c.v += row0 * c.vw;
            // a window of rows keeps its input rows and leaves the window as the round found it, for its keep
            const bool replayed = rows > 1 && slot + 1 < static_cast<int>(gridDim.y);
            c.states = nullptr;
            samples = replayed ? reinterpret_cast<float*>(launch.plan.snaps[2 * launch.layer]) + row0 * c.channels : nullptr;
        }
    }
    if (blockIdx.x >= conv_blocks) {
        const int i = (blockIdx.x - conv_blocks) * 128 + threadIdx.x;
        if (c.ga != nullptr && c.gate_dim == tf_kda::kDim) {
            // KDA: thread i the decay of channel i % 128 of (row, head) i / 128; the head's first channel its beta
            if (i >= c.gcount * tf_kda::kDim) return;
            const int rh = i / tf_kda::kDim, hc = i % (c.heads * tf_kda::kDim);
            if (i % tf_kda::kDim == 0) c.beta[rh] = 1.f / (1.f + expf(-load_kind(c.gb, c.kind, rh)));
            c.gate[i] = tf_kda::decay(load_kind(c.ga, c.kind, i), c.dt_bias[hc], c.a_log[hc / tf_kda::kDim], c.lower);
            return;
        }
        if (c.ga == nullptr || i >= c.gcount) return;
        const int h = i % c.heads;
        c.beta[i] = 1.f / (1.f + expf(-load_kind(c.gb, c.kind, i)));
        const float xg = load_kind(c.ga, c.kind, i) + c.dt_bias[h];
        const float soft = xg > 20.f ? xg : log1pf(expf(xg));
        c.gate[i] = expf(-expf(c.a_log[h]) * soft);
        return;
    }
    const int channel = blockIdx.x * 128 + threadIdx.x;
    const bool live = channel < c.channels;
    const bool head = c.norm == 128 && static_cast<int>(blockIdx.x) * 128 < 2 * c.kw;
    const float* taps = c.weight + static_cast<long long>(live ? channel : 0) * c.kernel;
    const int kept = c.kernel - 1;
    float history[7];
    const int f = c.from != nullptr && c.states != nullptr ? *c.from : -1;
    const float* start = f >= 0 ? c.states + static_cast<long long>(f) * kept * c.channels : c.state;
    if (live) {
        for (int tap = 0; tap < kept; ++tap) history[tap] = start[static_cast<long long>(tap) * c.channels + channel];
    }
    __atomic_signal_fence(__ATOMIC_SEQ_CST);  // the start is read before any snapshot store (it may be one of them)
    for (int r = 0; r < c.rows; ++r) {
        float y = 0.f;
        float sample = 0.f;
        if (live) {
            float acc = 0.f;
            for (int tap = 0; tap < kept; ++tap) acc += history[tap] * taps[tap];
            sample = load_kind(c.x, c.kind, static_cast<long long>(r) * c.channels + channel);
            acc += sample * taps[kept];
            y = acc * (1.f / (1.f + expf(-acc)));
        }
        if (head) {
            // the norm of the head's 128 values, then the scale by its weight
            __syncthreads();
            sq[threadIdx.x] = y * y;
            __syncthreads();
            if (threadIdx.x < 32) {
                float sum = 0.f;
                for (int j = 0; j < 4; ++j) sum += sq[threadIdx.x + 32 * j];
#pragma unroll
                for (int mask = 16; mask > 0; mask >>= 1) sum += __shfl_xor(sum, mask, 32);
                if (threadIdx.x == 0) inv_s = rsqrtf(sum / 128.f + c.eps);
            }
            __syncthreads();
            const bool is_q = channel < c.kw;
            y = y * inv_s * (is_q ? c.qw : c.kw_w)[threadIdx.x];
        }
        if (live) {
            if (channel < c.kw) {
                c.qn[static_cast<long long>(r) * c.kw + channel] = y;
            } else if (channel < 2 * c.kw) {
                c.kn[static_cast<long long>(r) * c.kw + channel - c.kw] = y;
            } else {
                c.v[static_cast<long long>(r) * c.vw + channel - 2 * c.kw] = y;
            }
            for (int tap = 0; tap < kept - 1; ++tap) history[tap] = history[tap + 1];
            if (kept > 0) history[kept - 1] = sample;
            if (samples != nullptr) samples[static_cast<long long>(r) * c.channels + channel] = sample;
            if (c.states != nullptr) {
                for (int tap = 0; tap < kept; ++tap) {
                    c.states[(static_cast<long long>(r) * kept + tap) * c.channels + channel] = history[tap];
                }
            }
        }
    }
    if (live && samples == nullptr) {
        for (int tap = 0; tap < kept; ++tap) c.state[static_cast<long long>(tap) * c.channels + channel] = history[tap];
    }
}

// rms over rows of `width` fp32 values a wave a row as rms_rows; blockIdx.y picks the first or second set of rows.
__device__ void
    rms2_body(const float* x0, const float* w0, float* y0, const float* x1, const float* w1, float* y1, int rows,
                int width, float eps) {
    constexpr int kMost = 1024 / 32;
    const float* x = blockIdx.y == 0 ? x0 : x1;
    const float* weight = blockIdx.y == 0 ? w0 : w1;
    float* y = blockIdx.y == 0 ? y0 : y1;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= rows) return;
    const long long at = static_cast<long long>(row) * width;
    float v[kMost];
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < kMost; ++j) {
        const int i = lane + 32 * j;
        v[j] = i < width ? x[at + i] : 0.f;
        if (i < width) sum += v[j] * v[j];
    }
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) sum += __shfl_xor(sum, mask, 32);
    const float inv = rsqrtf(__shfl(sum, 0, 32) / static_cast<float>(width) + eps);
#pragma unroll
    for (int j = 0; j < kMost; ++j) {
        const int i = lane + 32 * j;
        if (i >= width) break;
        float value = v[j] * inv;
        if (weight != nullptr) value *= weight[i];
        y[at + i] = value;
    }
}

// The gated norm out = rms(y) * weight * round(silu(z) or sigmoid(z)) in the kind, a wave a (row, head).
__device__ void
    gnorm_out_body(const float* y, const float* weight, const void* z, void* out, int kind, int rows, int width,
                     float eps, int sigmoid_gate) {
    constexpr int kMost = 1024 / 32;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= rows) return;
    const long long at = static_cast<long long>(row) * width;
    float v[kMost];
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < kMost; ++j) {
        const int i = lane + 32 * j;
        v[j] = i < width ? y[at + i] : 0.f;
        if (i < width) sum += v[j] * v[j];
    }
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) sum += __shfl_xor(sum, mask, 32);
    const float inv = rsqrtf(__shfl(sum, 0, 32) / static_cast<float>(width) + eps);
#pragma unroll
    for (int j = 0; j < kMost; ++j) {
        const int i = lane + 32 * j;
        if (i >= width) break;
        float value = v[j] * inv;
        if (weight != nullptr) value *= weight[i];
        const float zv = load_kind(z, kind, at + i);
        const float gate = sigmoid_gate ? 1.f / (1.f + expf(-zv)) : zv / (1.f + expf(-zv));
        store_kind(out, kind, at + i, value * round_kind(gate, kind));
    }
}

constexpr int kSelectThreads = 512;
constexpr int kSelectMax = 1024;

__device__ __forceinline__ float round_bf16_f(float v) { return round_kind(v, 2); }

// act.hip's moe_select with picks by rank: expert i is in the top k when fewer than k beat it; `items` writes the plan.
__device__ void select_body(const float* logits, int* pick, float* wts, int* items, int* members, int capacity,
                            int experts, int top_k) {
    __shared__ float v[kSelectMax];
    __shared__ float top_v[32];
    __shared__ int top_i[32];
    const int row = blockIdx.x;
    const int slots = top_k + 1;
    const float* l = logits + static_cast<long long>(row) * (experts + 1);
    for (int i = threadIdx.x; i < experts; i += kSelectThreads) v[i] = l[i];
    __syncthreads();
    for (int i = threadIdx.x; i < experts; i += kSelectThreads) {
        const float vi = v[i];
        int rank = 0;
#pragma unroll 16
        for (int j = 0; j < experts; ++j) {
            const float vj = v[j];
            rank += (vj > vi) || (vj == vi && j < i);
        }
        if (rank < top_k) {
            top_v[rank] = vi;
            top_i[rank] = i;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const float top = top_v[0];
        float ex[32];
        float total = 0.f;
        for (int k = 0; k < top_k; ++k) {
            ex[k] = expf(top_v[k] - top);
            total += ex[k];
        }
        for (int k = 0; k < top_k; ++k) {
            pick[row * slots + k] = top_i[k];
            wts[row * slots + k] = round_bf16_f(ex[k] / total);
        }
        const float gate = round_bf16_f(l[experts]);
        pick[row * slots + top_k] = experts;
        wts[row * slots + top_k] = round_bf16_f(1.f / (1.f + expf(-gate)));
    }
    if (items != nullptr) {
        __syncthreads();
        for (int k = threadIdx.x; k < capacity; k += kSelectThreads) {
            const bool real = k < slots;
            items[3 * k] = real ? pick[k] : 0;
            items[3 * k + 1] = real ? k : 0;
            items[3 * k + 2] = real ? 1 : 0;
            if (real) members[k] = k;
        }
    }
}

// The pick of row `row` (select_body's rule and arithmetic) by the router's block of 32 * kRouterWaves threads: the k-th
// pick is the best expert left under select_body's order (value descending, index ascending), which is the expert of
// rank k there, so top_v and top_i are the same. A row with a NaN logit takes select_body's rank loop instead (its
// order is not a total order there). Logits are read with coherent loads: other blocks of this launch wrote them.
__device__ void select_row(const float* logits, int row, int* pick, float* wts, int* items, int* members, int capacity,
                           int experts, int top_k) {
    constexpr int kThreads = 32 * kRouterWaves;
    __shared__ float v[kSelectMax];
    __shared__ unsigned char taken[kSelectMax];
    __shared__ float top_v[32];
    __shared__ int top_i[32];
    __shared__ float wave_v[kRouterWaves];
    __shared__ int wave_i[kRouterWaves];
    __shared__ int has_nan;
    const int slots = top_k + 1;
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    float* l = const_cast<float*>(logits) + static_cast<long long>(row) * (experts + 1);
    if (threadIdx.x == 0) has_nan = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < experts; i += kThreads) {
        const float vi = __hip_atomic_load(l + i, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        v[i] = vi;
        taken[i] = 0;
        if (vi != vi) has_nan = 1;
    }
    __syncthreads();
    if (has_nan) {
        for (int i = threadIdx.x; i < experts; i += kThreads) {
            const float vi = v[i];
            int rank = 0;
            for (int j = 0; j < experts; ++j) {
                const float vj = v[j];
                rank += (vj > vi) || (vj == vi && j < i);
            }
            if (rank < top_k) {
                top_v[rank] = vi;
                top_i[rank] = i;
            }
        }
    } else {
        for (int k = 0; k < top_k; ++k) {
            float bv = 0.f;
            int bi = 0x7fffffff;
            for (int i = threadIdx.x; i < experts; i += kThreads) {
                const float vi = v[i];
                if (!taken[i] && (bi == 0x7fffffff || vi > bv || (vi == bv && i < bi))) {
                    bv = vi;
                    bi = i;
                }
            }
#pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                const float ov = __shfl_xor(bv, mask, 32);
                const int oi = __shfl_xor(bi, mask, 32);
                if (oi != 0x7fffffff && (bi == 0x7fffffff || ov > bv || (ov == bv && oi < bi))) {
                    bv = ov;
                    bi = oi;
                }
            }
            if (lane == 0) {
                wave_v[wave] = bv;
                wave_i[wave] = bi;
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                float cv = wave_v[0];
                int ci = wave_i[0];
                for (int w = 1; w < kRouterWaves; ++w) {
                    const float ov = wave_v[w];
                    const int oi = wave_i[w];
                    if (oi != 0x7fffffff && (ci == 0x7fffffff || ov > cv || (ov == cv && oi < ci))) {
                        cv = ov;
                        ci = oi;
                    }
                }
                top_v[k] = cv;
                top_i[k] = ci;
                taken[ci] = 1;
            }
            __syncthreads();
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const float top = top_v[0];
        float ex[32];
        float total = 0.f;
        for (int k = 0; k < top_k; ++k) {
            ex[k] = expf(top_v[k] - top);
            total += ex[k];
        }
        for (int k = 0; k < top_k; ++k) {
            pick[row * slots + k] = top_i[k];
            wts[row * slots + k] = round_bf16_f(ex[k] / total);
        }
        const float gate = round_bf16_f(__hip_atomic_load(l + experts, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT));
        pick[row * slots + top_k] = experts;
        wts[row * slots + top_k] = round_bf16_f(1.f / (1.f + expf(-gate)));
    }
    if (items != nullptr) {
        __syncthreads();
        for (int k = threadIdx.x; k < capacity; k += kThreads) {
            const bool real = k < slots;
            items[3 * k] = real ? pick[k] : 0;
            items[3 * k + 1] = real ? k : 0;
            items[3 * k + 2] = real ? 1 : 0;
            if (real) members[k] = k;
        }
    }
    __syncthreads();
}

// select_row for one row by LANES lanes of a wave (32: the wave takes one row; 16: each half of the wave takes its own
// row, so 8 rows fit 4 waves in one pass): each row by the same order (value descending, index ascending) and the same
// comparisons, so the same top_v and top_i, and the group's first lane does thread 0's arithmetic in its order. A lane
// scans KPER logits (KPER * LANES >= experts); top_k <= LANES. `live`: the group has a row (the lanes of a group without
// one run along and write nothing). True when the row had a NaN logit and nothing of it was written (select_row's rank
// loop takes the row then).
template <int KPER, int LANES>
__device__ bool select_row_lanes(const float* logits, int row, bool live, int* pick, float* wts, int experts, int top_k) {
    const int lane = threadIdx.x & 31;
    const int hl = lane & (LANES - 1);
    const int base = lane & ~(LANES - 1);
    const unsigned group = LANES == 32 ? 0xffffffffu : ((1u << LANES) - 1u) << base;
    const int slots = top_k + 1;
    const float* l = logits + static_cast<long long>(live ? row : 0) * (experts + 1);
    float v[KPER];
    unsigned taken = 0;
    bool nan = false;
#pragma unroll
    for (int j = 0; j < KPER; ++j) {
        const int i = hl + LANES * j;
        v[j] = i < experts ? __hip_atomic_load(l + i, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) : 0.f;
        nan |= i < experts && v[j] != v[j];
    }
    const bool bad = (static_cast<unsigned>(__ballot(nan)) & group) != 0u;
    // the k-th pick, held by the group's lane k
    float mine_v = 0.f;
    int mine_i = 0;
    for (int k = 0; k < top_k; ++k) {
        float bv = 0.f;
        int bi = 0x7fffffff;
#pragma unroll
        for (int j = 0; j < KPER; ++j) {
            const int i = hl + LANES * j;
            if (i < experts && !((taken >> j) & 1u) && (bi == 0x7fffffff || v[j] > bv || (v[j] == bv && i < bi))) {
                bv = v[j];
                bi = i;
            }
        }
#pragma unroll
        for (int mask = LANES / 2; mask > 0; mask >>= 1) {
            const float ov = __shfl_xor(bv, mask, 32);
            const int oi = __shfl_xor(bi, mask, 32);
            if (oi != 0x7fffffff && (bi == 0x7fffffff || ov > bv || (ov == bv && oi < bi))) {
                bv = ov;
                bi = oi;
            }
        }
        if (bi % LANES == hl) taken |= 1u << (bi / LANES);
        if (hl == k) {
            mine_v = bv;
            mine_i = bi;
        }
    }
    // thread 0's arithmetic in its order; each exp is taken again for its weight (the same value)
    const float top = __shfl(mine_v, base, 32);
    float total = 0.f;
    for (int k = 0; k < top_k; ++k) total += expf(__shfl(mine_v, base + k, 32) - top);
    if (live && !bad) {
        const int at = row * slots;
        if (hl < top_k) {
            pick[at + hl] = mine_i;
            wts[at + hl] = round_bf16_f(expf(mine_v - top) / total);
        }
        if (hl == 0) {
            const float gate = round_bf16_f(__hip_atomic_load(l + experts, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT));
            pick[at + top_k] = experts;
            wts[at + top_k] = round_bf16_f(1.f / (1.f + expf(-gate)));
        }
    }
    return bad;
}

// tf_moe_route's plan of a few rows' pairs (at most kPlanPairs) by the last block: the pairs stable-sorted by expert
// into members, items of at most `tile` pairs cut a capacity slot each; the same members and items as tf_moe_route's
// (an expert's pairs in pair order, the same starts, ends and slot search).
constexpr int kPlanPairs = 32 * kRouterWaves;  // a pair a thread: 8 rows of top 10 + the shared expert (88)
// The shared memory it takes from `pool`, in ints: 3 * experts + 2 * 32 * kRouterWaves + kPlanPairs.
__device__ void route_plan(const int* picks, int pairs, int experts, int tile, int* members, int* items, int capacity,
                           int* pool) {
    constexpr int kThreads = 32 * kRouterWaves;
    int* counts = pool;
    int* starts = counts + experts;
    int* ends = starts + experts;
    int* scan_c = ends + experts;
    int* scan_n = scan_c + kThreads;
    int* pk = scan_n + kThreads;
    const int t = threadIdx.x;
    for (int e = t; e < experts; e += kThreads) counts[e] = 0;
    if (t < pairs) pk[t] = picks[t];
    __syncthreads();
    if (t < pairs) atomicAdd(&counts[pk[t]], 1);
    __syncthreads();
    const int per = (experts + kThreads - 1) / kThreads;
    const int lo = min(experts, t * per), hi = min(experts, lo + per);
    int own_c = 0, own_n = 0;
    for (int e = lo; e < hi; ++e) {
        own_c += counts[e];
        own_n += (counts[e] + tile - 1) / tile;
    }
    scan_c[t] = own_c;
    scan_n[t] = own_n;
    __syncthreads();
    for (int off = 1; off < kThreads; off <<= 1) {
        const int vc = t >= off ? scan_c[t - off] : 0, vn = t >= off ? scan_n[t - off] : 0;
        __syncthreads();
        scan_c[t] += vc;
        scan_n[t] += vn;
        __syncthreads();
    }
    int start = scan_c[t] - own_c, end = scan_n[t] - own_n;
    for (int e = lo; e < hi; ++e) {
        starts[e] = start;
        start += counts[e];
        end += (counts[e] + tile - 1) / tile;
        ends[e] = end;
    }
    __syncthreads();
    if (t < pairs) {
        const int e = pk[t];
        int before = 0;
        for (int q = 0; q < t; ++q) before += pk[q] == e;
        members[starts[e] + before] = t;
    }
    for (int slot = t; slot < capacity; slot += kThreads) {
        int a = 0, b = experts;
        while (a < b) {  // searchsorted(ends, slot, right=True)
            const int mid = (a + b) / 2;
            if (ends[mid] <= slot) a = mid + 1;
            else b = mid;
        }
        const int e = a < experts - 1 ? a : experts - 1;
        const int reps = (counts[e] + tile - 1) / tile;
        const int within = slot - (ends[e] - reps);
        const int count = counts[e] - within * tile;
        items[slot * 3] = e;
        items[slot * 3 + 1] = starts[e] + within * tile;
        items[slot * 3 + 2] = count < 0 ? 0 : (count > tile ? tile : count);
    }
}

// tf_router_decode's logits from bf16 rows, then the pick of every row (tf_select_decode's) by the last block to finish:
// one launch for both. `e` counts the rows with the shared gate's; `done` is zero between launches (the last block
// zeroes it). `items` also gets the plan: one row's own, or for 2 to 8 rows tf_moe_route's at `tile` pairs an item.
// `staged`: the host gave r * d bf16 of dynamic shared memory for x's rows (x 16-byte aligned, d a multiple of 8). A
// 2 to 8 row plan takes the dynamic shared memory after the router: the waves' top lists, then route_plan's pool.
template <int KIND>
__device__ void router_select(const void* x, const void* rows, float* logits, int r, int d, int e, int* pick, float* wts,
                              int* items, int* members, int capacity, int top_k, int tile, int staged, unsigned* done) {
    __shared__ bool last;
    // the dynamic shared memory the host sized (none where it stages nothing and plans nothing; launches/moe.zig)
    extern __shared__ __attribute__((aligned(16))) uint4 router_shared[];
    // a verify window's few rows in shared memory once a block: each wave (an expert) reads them there, not from L2
    const void* xs = router_shared;
    if (staged) {
        const uint4* from = static_cast<const uint4*>(x);
        for (int i = threadIdx.x; i < r * d / 8; i += 32 * kRouterWaves) router_shared[i] = from[i];
        __syncthreads();
    }
    const int expert = blockIdx.x * kRouterWaves + (threadIdx.x >> 5);
    if (expert < e) router_decode<KIND, true>(staged ? xs : x, rows, logits, r, d, e, staged);
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(done, 1u) == gridDim.x - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    if (r == 1) {
        select_row(logits, 0, pick, wts, items, members, capacity, e - 1, top_k);
    } else if (r <= 2 * kRouterWaves && e - 1 <= kSelectMax) {
        // a wave a row, or past kRouterWaves rows half a wave a row (one pass for 8 rows). A row with a NaN logit then
        // takes select_row's rank loop. The plan's arrays are in the dynamic shared memory (the
        // staged rows are free by then)
        __shared__ int fallback[2 * kRouterWaves];
        int* pool = reinterpret_cast<int*>(router_shared) + 2 * 32 * kRouterWaves;
        const int wave = threadIdx.x >> 5;
        __syncthreads();
        const int lane = threadIdx.x & 31;
        if (r > kRouterWaves && e - 1 <= 16 * 32 && top_k <= 16) {
            // 5 to 8 rows: half a wave a row, rows w and w + kRouterWaves in wave w's two halves
            const int row = wave + (lane >= 16 ? kRouterWaves : 0);
            const bool live = row < r;
            const bool bad = select_row_lanes<32, 16>(logits, row, live, pick, wts, e - 1, top_k);
            if (live && (lane & 15) == 0) fallback[row] = bad;
        } else {
            for (int row = wave; row < r; row += kRouterWaves) {
                const bool bad = e - 1 <= 16 * 32 ? select_row_lanes<16, 32>(logits, row, true, pick, wts, e - 1, top_k)
                                                  : select_row_lanes<kSelectMax / 32, 32>(logits, row, true, pick, wts, e - 1, top_k);
                if (lane == 0) fallback[row] = bad;
            }
        }
        __syncthreads();
        for (int row = 0; row < r; ++row)
            if (fallback[row]) select_row(logits, row, pick, wts, nullptr, nullptr, 0, e - 1, top_k);
        __threadfence_block();
        __syncthreads();
        if (items != nullptr) route_plan(pick, r * (top_k + 1), e, tile, members, items, capacity, pool);
    } else {
        for (int row = 0; row < r; ++row) select_row(logits, row, pick, wts, nullptr, nullptr, 0, e - 1, top_k);
    }
    if (threadIdx.x == 0) *done = 0u;
}

}  // namespace

extern "C" __global__ void __launch_bounds__(32 * kRouterWaves)
    tf_router_select_f16(const void* x, const void* rows, float* logits, int r, int d, int e, int* pick, float* wts,
                         int* items, int* members, int capacity, int top_k, int tile, int staged, unsigned* done) {
    router_select<1>(x, rows, logits, r, d, e, pick, wts, items, members, capacity, top_k, tile, staged, done);
}

extern "C" __global__ void __launch_bounds__(32 * kRouterWaves)
    tf_router_select_bf16(const void* x, const void* rows, float* logits, int r, int d, int e, int* pick, float* wts,
                          int* items, int* members, int capacity, int top_k, int tile, int staged, unsigned* done) {
    router_select<2>(x, rows, logits, r, d, e, pick, wts, items, members, capacity, top_k, tile, staged, done);
}

extern "C" __global__ void __launch_bounds__(32 * kRouterWaves)
    tf_router_decode_f16(const void* x, const float* rows, float* logits, int r, int d, int e) {
    router_decode<1>(x, rows, logits, r, d, e);
}

extern "C" __global__ void __launch_bounds__(32 * kRouterWaves)
    tf_router_decode_bf16(const void* x, const float* rows, float* logits, int r, int d, int e) {
    router_decode<2>(x, rows, logits, r, d, e);
}

extern "C" __global__ void __launch_bounds__(kTailThreads)
    tf_tail(void* x, const void* y, const float* wts, const float* weight, void* normed, int kind, int slots, int width,
            float eps) {
    tail_body(x, y, wts, weight, normed, kind, slots, width, eps);
}

extern "C" __global__ void __launch_bounds__(128) tf_conv_split(ConvArgs c) { conv_split_body(c); }

extern "C" __global__ void __launch_bounds__(256)
    tf_rms2(const float* x0, const float* w0, float* y0, const float* x1, const float* w1, float* y1, int rows, int width,
            float eps) {
    rms2_body(x0, w0, y0, x1, w1, y1, rows, width, eps);
}

extern "C" __global__ void __launch_bounds__(256)
    tf_gnorm_out(const float* y, const float* weight, const void* z, void* out, int kind, int rows, int width,
                 float eps, int sigmoid_gate) {
    gnorm_out_body(y, weight, z, out, kind, rows, width, eps, sigmoid_gate);
}

// A prompt's picks (no plan), a wave a row and 4 rows a block: select_row_lanes's order and arithmetic, so each row's
// picks and weights are select_body's bytes; a block's row with a NaN logit takes select_row's rank loop, which is
// select_body's for it.
extern "C" __global__ void __launch_bounds__(32 * kRouterWaves)
    tf_select_rows(const float* logits, int* pick, float* wts, int rows, int experts, int top_k) {
    __shared__ int fallback[kRouterWaves];
    const int wave = threadIdx.x >> 5;
    const int first = blockIdx.x * kRouterWaves;
    const int row = first + wave;
    const bool live = row < rows;
    const bool bad = experts <= 16 * 32 ? select_row_lanes<16, 32>(logits, row, live, pick, wts, experts, top_k)
                                        : select_row_lanes<kSelectMax / 32, 32>(logits, row, live, pick, wts, experts, top_k);
    if ((threadIdx.x & 31) == 0) fallback[wave] = live && bad;
    __syncthreads();
    for (int w = 0; w < kRouterWaves; ++w)
        if (fallback[w]) select_row(logits, first + w, pick, wts, nullptr, nullptr, 0, experts, top_k);
}

extern "C" __global__ void __launch_bounds__(kSelectThreads)
    tf_select_decode(const float* logits, int* pick, float* wts, int* items, int* members, int capacity, int experts,
                     int top_k) {
    select_body(logits, pick, wts, items, members, capacity, experts, top_k);
}
