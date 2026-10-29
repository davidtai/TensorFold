#pragma once

#include "../ops/common.hpp"
#include "common/wmma.hpp"

// Attention over a row's listed keys (the indexer's), or its first n keys, in parts merged in part order.

namespace {

constexpr int kSparseWaves = 8;
constexpr int kSparseMostDims = 256;
constexpr int kSparsePer = kSparseMostDims / 32;

// Keys a block stages at a time, and the padded row of the staged keys (rows read by distinct lanes, no conflicts).
constexpr int kSparseTile = 32;
constexpr int kSparseRow = kSparseMostDims + 1;

// Block (g, r, p): kv head g's query heads (G at most), row r, part p of its keys, 32 keys staged at a time.
template <int kSparseGroup>
__device__ void sparse_parts(const float* q, const void* k, const void* v, int cache_kind, const int* ids,
                             long long id_stride, const int* count, const int* sparse, float* parts_o, float* parts_ms,
                             int heads, int kv_heads, int dims, long long cap, float scale, int parts) {
    __shared__ float qs[kSparseGroup][kSparseMostDims];
    __shared__ float kt[kSparseTile][kSparseRow];
    __shared__ float ps[kSparseGroup][kSparseTile];
    __shared__ float fs[kSparseGroup], m[kSparseGroup], l[kSparseGroup];
    __shared__ long long keys[kSparseTile];
    const int g = blockIdx.x, r = blockIdx.y, part = blockIdx.z;
    const int group = heads / kv_heads;
    const int t = threadIdx.x, lane = t & 31, wave = t >> 5;
    const int n = count[r];
    const bool listed = ids != nullptr && sparse != nullptr && sparse[r] != 0;
    const int lo = static_cast<int>(static_cast<long long>(part) * n / parts);
    const int hi = static_cast<int>(static_cast<long long>(part + 1) * n / parts);
    const long long head_base = static_cast<long long>(g) * cap * dims;
    for (int i = t; i < group * dims; i += kSparseWaves * 32) {
        const int h = i / dims, d = i % dims;
        qs[h][d] = q[(static_cast<long long>(r) * heads + g * group + h) * dims + d] * scale;
    }
    if (t < kSparseGroup) {
        m[t] = -INFINITY;
        l[t] = 0.f;
    }
    float o[kSparseGroup];
#pragma unroll
    for (int h = 0; h < kSparseGroup; ++h) o[h] = 0.f;
    for (int t0 = lo; t0 < hi; t0 += kSparseTile) {
        const int tn = min(kSparseTile, hi - t0);
        __syncthreads();
        if (t < tn) keys[t] = listed ? ids[r * id_stride + t0 + t] : t0 + t;
        __syncthreads();
        for (int i = t; i < tn * dims; i += kSparseWaves * 32) {
            const int j = i / dims, d = i % dims;
            kt[j][d] = load(k, cache_kind, head_base + keys[j] * dims + d);
        }
        __syncthreads();
        // scores: a lane a key, a wave a head; then that head's softmax over the tile, folded into its running max
        for (int pair = t; pair < kSparseTile * group; pair += kSparseWaves * 32) {
            const int j = pair % kSparseTile, h = pair / kSparseTile;
            float sc = 0.f;
            for (int d = 0; d < dims; ++d) sc += qs[h][d] * kt[j][d];
            const bool live = j < tn;
            float top = live ? sc : -INFINITY;
#pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) top = fmaxf(top, __shfl_xor(top, mask, 32));
            top = fmaxf(top, m[h]);
            const float e = live ? expf(sc - top) : 0.f;
            float sum = e;
#pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) sum += __shfl_xor(sum, mask, 32);
            ps[h][j] = e;
            if (lane == 0) {
                const float f = expf(m[h] - top);
                fs[h] = f;
                l[h] = l[h] * f + sum;
                m[h] = top;
            }
        }
        __syncthreads();
        // values: a thread a dim, every head
        for (int d = t; d < dims; d += kSparseWaves * 32) {
#pragma unroll
            for (int h = 0; h < kSparseGroup; ++h) o[h] *= h < group ? fs[h] : 1.f;
            for (int j = 0; j < tn; ++j) {
                const float vd = load(v, cache_kind, head_base + keys[j] * dims + d);
#pragma unroll
                for (int h = 0; h < kSparseGroup; ++h) {
                    if (h < group) o[h] += ps[h][j] * vd;
                }
            }
        }
    }
    __syncthreads();
    for (int h = 0; h < group; ++h) {
        const long long out = (static_cast<long long>(r) * heads + g * group + h) * parts + part;
        for (int d = t; d < dims; d += kSparseWaves * 32) parts_o[out * dims + d] = o[h];
        if (t == 0) {
            parts_ms[out * 2] = m[h];
            parts_ms[out * 2 + 1] = l[h];
        }
    }
}

// The parts kernel for groups of up to 6 query heads (a tp-4 rank's) and of up to 12 (the whole model's).
extern "C" __global__ void __launch_bounds__(kSparseWaves * 32)
tf_fn_sparse_parts6(const float* q, const void* k, const void* v, int cache_kind, const int* ids, long long id_stride,
                    const int* count, const int* sparse, float* parts_o, float* parts_ms, int heads, int kv_heads,
                    int dims, long long cap, float scale, int parts) {
    sparse_parts<6>(q, k, v, cache_kind, ids, id_stride, count, sparse, parts_o, parts_ms, heads, kv_heads, dims, cap,
                    scale, parts);
}

extern "C" __global__ void __launch_bounds__(kSparseWaves * 32)
tf_fn_sparse_parts12(const float* q, const void* k, const void* v, int cache_kind, const int* ids, long long id_stride,
                     const int* count, const int* sparse, float* parts_o, float* parts_ms, int heads, int kv_heads,
                     int dims, long long cap, float scale, int parts) {
    sparse_parts<12>(q, k, v, cache_kind, ids, id_stride, count, sparse, parts_o, parts_ms, heads, kv_heads, dims, cap,
                     scale, parts);
}

// ---- A prompt's sparse attention on the gfx11 matrix cores (v_wmma_f32_16x16x16_bf16), Qwen3.8 Flash Next only:
// 24 query heads of 256, 2 kv heads, bf16 caches. Block (r, g): row r's 12 heads of kv head g over all its keys in one
// pass, so no parts and no merge; the grid runs kv head 0's rows first, its keys and values (18 MB at 17.7K) stay in
// the MALL. Wave w owns dims 64 w .. 64 w + 63 of q.k and p.v; a 32-key chunk at a time:
//   q.k: A = q (16 rows: the 12 heads, 4 zero), B = keys (16 dims x 16 keys), the four waves' partial scores added in
//        wave order in LDS; q is bf16-exact in the model (tf_qk_rope rounds it, the scale 1/16 is a power of two), so
//        one product is exact; SPLIT carries an fp32 q as bf16 hi + lo (two products) for any other caller.
//   softmax: fp32, online, natural exp as tf_fn_sparse_parts.
//   p.v: A = p as bf16 hi + lo (two products: p keeps about 16 bits), B = values (16 keys x 16 dims) read from a
//        transposed LDS copy of the wave's 64 dims of the chunk's value rows.
// gfx11 fragments (wave32): A lane l holds all 16 k of row l % 16, B lane l all 16 k of column l % 16 (lanes 16-31
// repeat 0-15), D lane l holds D[2 i + l / 16][l % 16]. Other chips build a trap; the launcher never picks it there.
// With qg, the output is the gated bf16 row tf_attn_gate writes; without, the fp32 attention (rows, 24, 256).
// After Strata's gfx11 prompt attention (github.com/Niko1221/Strata, MIT, src/kernels/cuda/qsa_prompt_attn.cu@7dd5cf18,
// prompt_attn_wmma_kernel: waves over 64-dim groups, hi + lo operands) and mlx-serve's qsa_nax.metal (rows = heads).
constexpr int kPromptHeads = 24;
constexpr int kPromptGroup = 12;
constexpr int kPromptDims = 256;
constexpr int kPromptThreads = 128;
constexpr int kPromptChunk = 32;
constexpr int kPromptPRow = kPromptChunk + 4;  // fp32 p row: 16-byte aligned, banks spread
constexpr int kPromptVRow = kPromptChunk + 8;  // bf16 transposed value row (a dim): 80 bytes

struct alignas(16) PromptSmem {
    float part[4][16][kPromptChunk + 1];
    float p[16][kPromptPRow];
    uint16_t vt[4][64][kPromptVRow];
    float m[16], l[16], alpha[16];
};

__device__ __forceinline__ uint32_t prompt_pack(uint16_t lo, uint16_t hi) {
    return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}

// PARTS: block (r, g, p) takes part p of gridDim.z of the row's keys (lo = p n / parts, as tf_fn_sparse_parts) and
// writes its unnormalized o, running max and sum in tf_fn_sparse_parts' layout for tf_fn_sparse_merge(_gate).
// LO: p v also over p's bf16 rounding remainder (p = hi + lo, near fp32's p v); off: p in bf16 once (inexact).
template <bool SPLIT, bool PARTS, bool LO = true>
__device__ void sparse_prompt(const float* __restrict__ q, const uint16_t* __restrict__ k,
                              const uint16_t* __restrict__ v, const int* __restrict__ ids, long long id_stride,
                              const int* __restrict__ count, const int* __restrict__ sparse, const uint16_t* qg,
                              void* out, float* parts_ms, long long cap, float scale) {
#if TF_DEVICE_WMMA_GFX11
    using tf::rocm::bf16x16;
    using tf::rocm::f32x8;
    using tf::rocm::u32x8;
    using Mma = tf::rocm::WmmaBF16;
    __shared__ PromptSmem S;
    const int r = blockIdx.x, g = blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, wave = t >> 5;
    const int col = lane & 15, half = lane >> 4;
    const int dim0 = wave * 64;
    const int n_row = count[r];
    const int part = PARTS ? static_cast<int>(blockIdx.z) : 0, parts = PARTS ? static_cast<int>(gridDim.z) : 1;
    const int lo = static_cast<int>(static_cast<long long>(part) * n_row / parts);
    const int n = static_cast<int>(static_cast<long long>(part + 1) * n_row / parts);
    const bool listed = ids != nullptr && sparse != nullptr && sparse[r] != 0;
    const int* row_ids = ids + r * id_stride;
    const long long head_base = static_cast<long long>(g) * cap * kPromptDims;

    // q fragments of this wave's 64 dims: row col (heads 12..15 zero), k = dim0 + 16 kk + 0..15
    bf16x16 qh[4], ql[4];
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
        u32x8 hw, lw;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            float x0 = 0.f, x1 = 0.f;
            if (col < kPromptGroup) {
                const float* src = q + (static_cast<long long>(r) * kPromptHeads + g * kPromptGroup + col) * kPromptDims +
                                   dim0 + kk * 16 + 2 * i;
                x0 = src[0] * scale;
                x1 = src[1] * scale;
            }
            const uint16_t h0 = float_to_bf16(x0), h1 = float_to_bf16(x1);
            hw[i] = prompt_pack(h0, h1);
            if constexpr (SPLIT) lw[i] = prompt_pack(float_to_bf16(x0 - bf16_to_float(h0)), float_to_bf16(x1 - bf16_to_float(h1)));
        }
        qh[kk] = __builtin_bit_cast(bf16x16, hw);
        if constexpr (SPLIT) ql[kk] = __builtin_bit_cast(bf16x16, lw);
    }
    if (t < 16) {
        S.m[t] = -INFINITY;
        S.l[t] = 0.f;
    }
    f32x8 acc[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[j] = f32x8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int c0 = lo; c0 < n; c0 += kPromptChunk) {
        // this lane's cell (lane = cell): its key, or key 0 past the row's keys (finite values under p = 0)
        const int cell = c0 + lane;
        const bool live = cell < n;
        const int key = live ? (listed ? row_ids[cell] : cell) : 0;
        // every read of the chunk first: the keys of the two 16-cell tiles (this wave's dims), the lane's value row
        uint4 kx[2][4][2];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int kr = __shfl(key, nt * 16 + col, 32);
            const uint4* src = reinterpret_cast<const uint4*>(k + head_base + static_cast<long long>(kr) * kPromptDims + dim0);
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                kx[nt][kk][0] = src[2 * kk];
                kx[nt][kk][1] = src[2 * kk + 1];
            }
        }
        uint4 vx[8];
        {
            const uint4* src = reinterpret_cast<const uint4*>(v + head_base + static_cast<long long>(key) * kPromptDims + dim0);
#pragma unroll
            for (int i = 0; i < 8; ++i) vx[i] = src[i];
        }
        // partial scores of this wave's dims, rows 2 i + half, cell nt * 16 + col
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            f32x8 s = f32x8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                const u32x8 w = {kx[nt][kk][0].x, kx[nt][kk][0].y, kx[nt][kk][0].z, kx[nt][kk][0].w,
                                 kx[nt][kk][1].x, kx[nt][kk][1].y, kx[nt][kk][1].z, kx[nt][kk][1].w};
                const bf16x16 b = __builtin_bit_cast(bf16x16, w);
                s = Mma::mma(qh[kk], b, s);
                if constexpr (SPLIT) s = Mma::mma(ql[kk], b, s);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) S.part[wave][2 * i + half][nt * 16 + col] = s[i];
        }
        // the wave's value rows, transposed: vt[wave][dim][cell]
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t w4[4] = {vx[i].x, vx[i].y, vx[i].z, vx[i].w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                S.vt[wave][i * 8 + 2 * j][lane] = static_cast<uint16_t>(w4[j] & 0xffffu);
                S.vt[wave][i * 8 + 2 * j + 1][lane] = static_cast<uint16_t>(w4[j] >> 16);
            }
        }
        __syncthreads();
        // online softmax: row t / 8, cells 4 (t % 8) ..; the four waves' partials in wave order
        {
            const int row = t >> 3, sub = t & 7;
            float x[4], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = sub * 4 + j;
                const float sc = ((S.part[0][row][c] + S.part[1][row][c]) + S.part[2][row][c]) + S.part[3][row][c];
                x[j] = c0 + c < n ? sc : -INFINITY;
                mx = fmaxf(mx, x[j]);
            }
#pragma unroll
            for (int mask = 1; mask < 8; mask <<= 1) mx = fmaxf(mx, __shfl_xor(mx, mask, 32));
            const float m_old = S.m[row];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float e = x[j] == -INFINITY ? 0.f : expf(x[j] - m_new);
                S.p[row][sub * 4 + j] = e;
                sum += e;
            }
#pragma unroll
            for (int mask = 1; mask < 8; mask <<= 1) sum += __shfl_xor(sum, mask, 32);
            if (sub == 0) {
                const float f = expf(m_old - m_new);
                S.alpha[row] = f;
                S.l[row] = S.l[row] * f + sum;
                S.m[row] = m_new;
            }
        }
        __syncthreads();
        // o = o alpha + p v over this wave's dims: 4 tiles of 16 dims, 2 steps of 16 cells
        float a[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) a[i] = S.alpha[2 * i + half];
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[j][i] *= a[i];
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            u32x8 hw, lw;
            const float* pr = &S.p[col][ks * 16];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float p0 = pr[2 * i], p1 = pr[2 * i + 1];
                const uint16_t h0 = float_to_bf16(p0), h1 = float_to_bf16(p1);
                hw[i] = prompt_pack(h0, h1);
                if constexpr (LO) lw[i] = prompt_pack(float_to_bf16(p0 - bf16_to_float(h0)), float_to_bf16(p1 - bf16_to_float(h1)));
            }
            const bf16x16 ph = __builtin_bit_cast(bf16x16, hw), pl = __builtin_bit_cast(bf16x16, lw);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint4* src = reinterpret_cast<const uint4*>(&S.vt[wave][j * 16 + col][ks * 16]);
                const uint4 lo = src[0], hi = src[1];
                const u32x8 w = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
                const bf16x16 b = __builtin_bit_cast(bf16x16, w);
                acc[j] = Mma::mma(ph, b, acc[j]);
                if constexpr (LO) acc[j] = Mma::mma(pl, b, acc[j]);
            }
        }
        // no barrier: part and vt are rewritten before the next first barrier only by their own wave, p and alpha
        // after it, which every wave reaches once its p.v is done
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int h = 2 * i + half;
        if (h >= kPromptGroup) continue;
        if constexpr (PARTS) {
            const long long row = (static_cast<long long>(r) * kPromptHeads + g * kPromptGroup + h) * parts + part;
#pragma unroll
            for (int j = 0; j < 4; ++j) static_cast<float*>(out)[row * kPromptDims + dim0 + j * 16 + col] = acc[j][i];
            if (wave == 0 && col == 0) {
                parts_ms[row * 2] = S.m[h];
                parts_ms[row * 2 + 1] = S.l[h];
            }
            continue;
        }
        const float total = S.l[h];
        const long long at = (static_cast<long long>(r) * kPromptHeads + g * kPromptGroup + h) * kPromptDims;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int d = dim0 + j * 16 + col;
            const float o = total > 0.f ? acc[j][i] / total : 0.f;
            if (qg == nullptr) {
                static_cast<float*>(out)[at + d] = o;
            } else {
                const float gate = bf16_to_float(qg[at * 2 + kPromptDims + d]);
                static_cast<uint16_t*>(out)[at + d] = float_to_bf16(o * sigmoid(gate));
            }
        }
    }
#else
    __builtin_trap();
#endif
}

extern "C" __global__ void __launch_bounds__(kPromptThreads)
tf_fn_sparse_prompt(const float* q, const void* k, const void* v, const int* ids, long long id_stride,
                    const int* count, const int* sparse, const void* qg, void* out, long long cap, float scale) {
    sparse_prompt<false, false>(q, static_cast<const uint16_t*>(k), static_cast<const uint16_t*>(v), ids, id_stride,
                                count, sparse, static_cast<const uint16_t*>(qg), out, nullptr, cap, scale);
}

// The same with p in bf16 once (no lo pass over its rounding remainder): numerics=inexact's prompt tile.
extern "C" __global__ void __launch_bounds__(kPromptThreads)
tf_fn_sparse_prompt_hi(const float* q, const void* k, const void* v, const int* ids, long long id_stride,
                       const int* count, const int* sparse, const void* qg, void* out, long long cap, float scale) {
    sparse_prompt<false, false, false>(q, static_cast<const uint16_t*>(k), static_cast<const uint16_t*>(v), ids, id_stride,
                                       count, sparse, static_cast<const uint16_t*>(qg), out, nullptr, cap, scale);
}

// The same over an fp32 q that need not be bf16-exact: q as bf16 hi + lo.
extern "C" __global__ void __launch_bounds__(kPromptThreads)
tf_fn_sparse_prompt_split(const float* q, const void* k, const void* v, const int* ids, long long id_stride,
                          const int* count, const int* sparse, const void* qg, void* out, long long cap, float scale) {
    sparse_prompt<true, false>(q, static_cast<const uint16_t*>(k), static_cast<const uint16_t*>(v), ids, id_stride,
                               count, sparse, static_cast<const uint16_t*>(qg), out, nullptr, cap, scale);
}

// The decode and window rows: the same tile over part blockIdx.z of each row's keys, into tf_fn_sparse_parts' parts.
extern "C" __global__ void __launch_bounds__(kPromptThreads)
tf_fn_sparse_tile_parts(const float* q, const void* k, const void* v, const int* ids, long long id_stride,
                        const int* count, const int* sparse, float* parts_o, float* parts_ms, long long cap,
                        float scale) {
    sparse_prompt<false, true>(q, static_cast<const uint16_t*>(k), static_cast<const uint16_t*>(v), ids, id_stride,
                               count, sparse, nullptr, parts_o, parts_ms, cap, scale);
}
extern "C" __global__ void __launch_bounds__(kPromptThreads)
tf_fn_sparse_tile_parts_split(const float* q, const void* k, const void* v, const int* ids, long long id_stride,
                              const int* count, const int* sparse, float* parts_o, float* parts_ms, long long cap,
                              float scale) {
    sparse_prompt<true, true>(q, static_cast<const uint16_t*>(k), static_cast<const uint16_t*>(v), ids, id_stride,
                              count, sparse, nullptr, parts_o, parts_ms, cap, scale);
}

// Block (h, r), a thread a dim: the parts of head h of row r combined in part order, fp32 rows-major out; with qg,
// the row gated as tf_attn_gate gates it (out = round(o sigmoid(gate)) in the activation kind).
__device__ __forceinline__ void sparse_merge(const float* parts_o, const float* parts_ms, const void* qg, int kind,
                                             void* out, int heads, int dims, int parts) {
    const int h = blockIdx.x, r = blockIdx.y;
    const long long at = (static_cast<long long>(r) * heads + h) * parts;
    float top = -INFINITY;
    for (int p = 0; p < parts; ++p) top = fmaxf(top, parts_ms[(at + p) * 2]);
    for (int d = threadIdx.x; d < dims; d += blockDim.x) {
        float total = 0.f, acc = 0.f;
        for (int p = 0; p < parts; ++p) {
            const float mp = parts_ms[(at + p) * 2];
            const float w = mp == -INFINITY ? 0.f : expf(mp - top);
            total += parts_ms[(at + p) * 2 + 1] * w;
            acc += parts_o[(at + p) * dims + d] * w;
        }
        const float o = total > 0.f ? acc / total : 0.f;
        const long long i = (static_cast<long long>(r) * heads + h) * dims + d;
        if (qg == nullptr) static_cast<float*>(out)[i] = o;
        else store(out, kind, i, o * sigmoid(load(qg, kind, (i - d) * 2 + dims + d)));
    }
}

extern "C" __global__ void tf_fn_sparse_merge(const float* parts_o, const float* parts_ms, float* out, int heads,
                                              int dims, int parts) {
    sparse_merge(parts_o, parts_ms, nullptr, 0, out, heads, dims, parts);
}

extern "C" __global__ void tf_fn_sparse_merge_gate(const float* parts_o, const float* parts_ms, const void* qg,
                                                   int kind, void* out, int heads, int dims, int parts) {
    sparse_merge(parts_o, parts_ms, qg, kind, out, heads, dims, parts);
}

// ---- W4 prototypes (w4/proto-ql3, never merged): the inexact prompt tile (sparse_prompt<false, false, false>) with
// fewer 128-byte lines a gather load. Same arithmetic in the same order, so the same bytes.
//   KCO (QL-10): K in coalesced loads, 8 lanes a key's 128-byte line (4 keys a load, 8 loads a chunk, against 16 loads
//        of 16 lines), staged in the wave's own vt area ([32 keys][64 + 8 dims] bf16) and read back as B fragments.
//   VT (QL-3): V in the cache in 4-token blocks of dim pairs: block b = key / 4 of head g holds, for dim pair p, the 4
//        tokens' (2p, 2p + 1) as one dword each: dword ((g cap / 4 + b) 128 + p) 4 + key % 4. A load is one cell's
//        64 dims (32 lanes x 4 bytes over 512 bytes: 4 lines, against 8 loads of 32 lines); each lane then holds dims
//        2 lane, 2 lane + 1 of all 32 cells, written to vt as two rows of 16-byte stores (no 2-byte transpose).
// LDS, launch and outputs are tf_fn_sparse_prompt_hi's.
// W2 Q4 arms (w2/proto-q4, never merged; picks come in aligned 4-token blocks, 100 % of the served lists):
//   V16 (Q4-1): QL-3's V layout read with 16-byte loads: a lane (dim pair) takes a picked block's 4 tokens at once,
//        8 loads a chunk of 4 lines each (QL-3: 32 four-byte loads), then QL-3's 16-byte stores to vt.
//   KBLK (Q4-2): K in the cache in 4-token blocks of 16-dim groups: block b, group dg holds the 4 tokens' 16 dims
//        (128 bytes): element ((g cap / 4 + b) 16 + dg) 64 + (key % 4) 16 + d % 16. The same 16 loads a chunk, a lane
//        a key as today, 4 lines a load against 16.
template <bool KCO, bool VT, bool PARTS = false, bool LO = false, bool V16 = false, bool KBLK = false>
__device__ void w4_sparse_ql(const float* __restrict__ q, const uint16_t* __restrict__ k, const uint16_t* __restrict__ v,
                             const int* __restrict__ ids, long long id_stride, const int* __restrict__ count,
                             const int* __restrict__ sparse, const uint16_t* qg, void* out, float* parts_ms,
                             long long cap, float scale) {
#if TF_DEVICE_WMMA_GFX11
    using tf::rocm::bf16x16;
    using tf::rocm::f32x8;
    using tf::rocm::u32x8;
    using Mma = tf::rocm::WmmaBF16;
    __shared__ PromptSmem S;
    const int r = blockIdx.x, g = blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, wave = t >> 5;
    const int col = lane & 15, half = lane >> 4;
    const int dim0 = wave * 64;
    const int n_row = count[r];
    const int part = PARTS ? static_cast<int>(blockIdx.z) : 0, parts = PARTS ? static_cast<int>(gridDim.z) : 1;
    const int lo_key = static_cast<int>(static_cast<long long>(part) * n_row / parts);
    const int n = static_cast<int>(static_cast<long long>(part + 1) * n_row / parts);
    const bool listed = ids != nullptr && sparse != nullptr && sparse[r] != 0;
    const int* row_ids = ids + r * id_stride;
    const long long head_base = static_cast<long long>(g) * cap * kPromptDims;
    // the compiler's view of the wave's LDS order (one wave's LDS operations run in order)
    auto wave_order = [] {
        __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
        __builtin_amdgcn_wave_barrier();
    };
    constexpr int kKsRow = 9;  // KCO stage row: 64 dims + 8, in 16-byte units
    uint4* ks = reinterpret_cast<uint4*>(&S.vt[wave][0][0]);
    static_assert(32 * kKsRow * 16 <= sizeof(S.vt[0]), "the K stage fits the wave's vt area");

    bf16x16 qh[4];
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
        u32x8 hw;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            float x0 = 0.f, x1 = 0.f;
            if (col < kPromptGroup) {
                const float* src = q + (static_cast<long long>(r) * kPromptHeads + g * kPromptGroup + col) * kPromptDims +
                                   dim0 + kk * 16 + 2 * i;
                x0 = src[0] * scale;
                x1 = src[1] * scale;
            }
            hw[i] = prompt_pack(float_to_bf16(x0), float_to_bf16(x1));
        }
        qh[kk] = __builtin_bit_cast(bf16x16, hw);
    }
    if (t < 16) {
        S.m[t] = -INFINITY;
        S.l[t] = 0.f;
    }
    f32x8 acc[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[j] = f32x8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int c0 = lo_key; c0 < n; c0 += kPromptChunk) {
        const int cell = c0 + lane;
        const bool live = cell < n;
        const int key = live ? (listed ? row_ids[cell] : cell) : 0;
        uint4 kx[2][4][2];
        uint4 kc[8];
        if constexpr (KCO) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int kr = __shfl(key, 4 * i + (lane >> 3), 32);
                kc[i] = *reinterpret_cast<const uint4*>(k + head_base + static_cast<long long>(kr) * kPromptDims + dim0 + 8 * (lane & 7));
            }
        } else if constexpr (KBLK) {
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const int kr = __shfl(key, nt * 16 + col, 32);
                const uint16_t* kb = k + head_base + static_cast<long long>(kr >> 2) * (kPromptDims * 4) + (kr & 3) * 16;
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    const uint4* src = reinterpret_cast<const uint4*>(kb + (dim0 / 16 + kk) * 64);
                    kx[nt][kk][0] = src[0];
                    kx[nt][kk][1] = src[1];
                }
            }
        } else {
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const int kr = __shfl(key, nt * 16 + col, 32);
                const uint4* src = reinterpret_cast<const uint4*>(k + head_base + static_cast<long long>(kr) * kPromptDims + dim0);
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    kx[nt][kk][0] = src[2 * kk];
                    kx[nt][kk][1] = src[2 * kk + 1];
                }
            }
        }
        uint4 vx[8];
        uint32_t vw[32];
        if constexpr (V16) {
            const uint4* v4 = reinterpret_cast<const uint4*>(v) + head_base / 8 + dim0 / 2 + lane;
#pragma unroll
            for (int gi = 0; gi < 8; ++gi) {
                const int kb = __shfl(key, 4 * gi, 32);
                const uint4 w = v4[static_cast<long long>(kb >> 2) * 128];
                vw[4 * gi] = w.x;
                vw[4 * gi + 1] = w.y;
                vw[4 * gi + 2] = w.z;
                vw[4 * gi + 3] = w.w;
            }
        } else if constexpr (VT) {
            const uint32_t* v4 = reinterpret_cast<const uint32_t*>(v) + head_base / 2 + (dim0 / 2 + lane) * 4;
#pragma unroll
            for (int c = 0; c < 32; ++c) {
                const int kc_ = __shfl(key, c, 32);
                vw[c] = v4[static_cast<long long>(kc_ >> 2) * 512 + (kc_ & 3)];
            }
        } else {
            const uint4* src = reinterpret_cast<const uint4*>(v + head_base + static_cast<long long>(key) * kPromptDims + dim0);
#pragma unroll
            for (int i = 0; i < 8; ++i) vx[i] = src[i];
        }
        if constexpr (KCO) {
            wave_order();  // the previous chunk's p.v reads of vt
#pragma unroll
            for (int i = 0; i < 8; ++i) ks[(4 * i + (lane >> 3)) * kKsRow + (lane & 7)] = kc[i];
            wave_order();
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    kx[nt][kk][0] = ks[(nt * 16 + col) * kKsRow + 2 * kk];
                    kx[nt][kk][1] = ks[(nt * 16 + col) * kKsRow + 2 * kk + 1];
                }
        }
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            f32x8 s = f32x8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                const u32x8 w = {kx[nt][kk][0].x, kx[nt][kk][0].y, kx[nt][kk][0].z, kx[nt][kk][0].w,
                                 kx[nt][kk][1].x, kx[nt][kk][1].y, kx[nt][kk][1].z, kx[nt][kk][1].w};
                s = Mma::mma(qh[kk], __builtin_bit_cast(bf16x16, w), s);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) S.part[wave][2 * i + half][nt * 16 + col] = s[i];
        }
        if constexpr (KCO) wave_order();  // the stage's reads before vt is rewritten
        if constexpr (VT || V16) {
            uint32_t ra[16], rb[16];
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                ra[i] = (vw[2 * i] & 0xffffu) | (vw[2 * i + 1] << 16);
                rb[i] = (vw[2 * i] >> 16) | (vw[2 * i + 1] & 0xffff0000u);
            }
            uint4* da = reinterpret_cast<uint4*>(&S.vt[wave][2 * lane][0]);
            uint4* db = reinterpret_cast<uint4*>(&S.vt[wave][2 * lane + 1][0]);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                da[i] = uint4{ra[4 * i], ra[4 * i + 1], ra[4 * i + 2], ra[4 * i + 3]};
                db[i] = uint4{rb[4 * i], rb[4 * i + 1], rb[4 * i + 2], rb[4 * i + 3]};
            }
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const uint32_t w4[4] = {vx[i].x, vx[i].y, vx[i].z, vx[i].w};
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    S.vt[wave][i * 8 + 2 * j][lane] = static_cast<uint16_t>(w4[j] & 0xffffu);
                    S.vt[wave][i * 8 + 2 * j + 1][lane] = static_cast<uint16_t>(w4[j] >> 16);
                }
            }
        }
        __syncthreads();
        {
            const int row = t >> 3, sub = t & 7;
            float x[4], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = sub * 4 + j;
                const float sc = ((S.part[0][row][c] + S.part[1][row][c]) + S.part[2][row][c]) + S.part[3][row][c];
                x[j] = c0 + c < n ? sc : -INFINITY;
                mx = fmaxf(mx, x[j]);
            }
#pragma unroll
            for (int mask = 1; mask < 8; mask <<= 1) mx = fmaxf(mx, __shfl_xor(mx, mask, 32));
            const float m_old = S.m[row];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float e = x[j] == -INFINITY ? 0.f : expf(x[j] - m_new);
                S.p[row][sub * 4 + j] = e;
                sum += e;
            }
#pragma unroll
            for (int mask = 1; mask < 8; mask <<= 1) sum += __shfl_xor(sum, mask, 32);
            if (sub == 0) {
                const float f = expf(m_old - m_new);
                S.alpha[row] = f;
                S.l[row] = S.l[row] * f + sum;
                S.m[row] = m_new;
            }
        }
        __syncthreads();
        float a[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) a[i] = S.alpha[2 * i + half];
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[j][i] *= a[i];
#pragma unroll
        for (int ks_ = 0; ks_ < 2; ++ks_) {
            u32x8 hw, lw;
            const float* pr = &S.p[col][ks_ * 16];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float p0 = pr[2 * i], p1 = pr[2 * i + 1];
                const uint16_t h0 = float_to_bf16(p0), h1 = float_to_bf16(p1);
                hw[i] = prompt_pack(h0, h1);
                if constexpr (LO) lw[i] = prompt_pack(float_to_bf16(p0 - bf16_to_float(h0)), float_to_bf16(p1 - bf16_to_float(h1)));
            }
            const bf16x16 ph = __builtin_bit_cast(bf16x16, hw), pl = __builtin_bit_cast(bf16x16, lw);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint4* src = reinterpret_cast<const uint4*>(&S.vt[wave][j * 16 + col][ks_ * 16]);
                const uint4 lo = src[0], hi = src[1];
                const u32x8 w = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
                const bf16x16 b = __builtin_bit_cast(bf16x16, w);
                acc[j] = Mma::mma(ph, b, acc[j]);
                if constexpr (LO) acc[j] = Mma::mma(pl, b, acc[j]);
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int h = 2 * i + half;
        if (h >= kPromptGroup) continue;
        if constexpr (PARTS) {
            const long long row = (static_cast<long long>(r) * kPromptHeads + g * kPromptGroup + h) * parts + part;
#pragma unroll
            for (int j = 0; j < 4; ++j) static_cast<float*>(out)[row * kPromptDims + dim0 + j * 16 + col] = acc[j][i];
            if (wave == 0 && col == 0) {
                parts_ms[row * 2] = S.m[h];
                parts_ms[row * 2 + 1] = S.l[h];
            }
            continue;
        }
        const float total = S.l[h];
        const long long at = (static_cast<long long>(r) * kPromptHeads + g * kPromptGroup + h) * kPromptDims;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int d = dim0 + j * 16 + col;
            const float o = total > 0.f ? acc[j][i] / total : 0.f;
            if (qg == nullptr) {
                static_cast<float*>(out)[at + d] = o;
            } else {
                const float gate = bf16_to_float(qg[at * 2 + kPromptDims + d]);
                static_cast<uint16_t*>(out)[at + d] = float_to_bf16(o * sigmoid(gate));
            }
        }
    }
#else
    __builtin_trap();
#endif
}

#define W4_QL_KERNEL(NAME, KCO, VT)                                                                                     \
    extern "C" __global__ void __launch_bounds__(kPromptThreads)                                                        \
    NAME(const float* q, const void* k, const void* v, const int* ids, long long id_stride, const int* count,           \
         const int* sparse, const void* qg, void* out, long long cap, float scale) {                                    \
        w4_sparse_ql<KCO, VT>(q, static_cast<const uint16_t*>(k), static_cast<const uint16_t*>(v), ids, id_stride,      \
                              count, sparse, static_cast<const uint16_t*>(qg), out, nullptr, cap, scale);               \
    }
W4_QL_KERNEL(w4_sparse_ql_ref, false, false)
W4_QL_KERNEL(w4_sparse_ql10, true, false)
W4_QL_KERNEL(w4_sparse_ql3, false, true)
W4_QL_KERNEL(w4_sparse_ql3k, true, true)
#undef W4_QL_KERNEL
#define W2_Q4_KERNEL(NAME, V16, KBLK)                                                                                   \
    extern "C" __global__ void __launch_bounds__(kPromptThreads)                                                        \
    NAME(const float* q, const void* k, const void* v, const int* ids, long long id_stride, const int* count,           \
         const int* sparse, const void* qg, void* out, long long cap, float scale) {                                    \
        w4_sparse_ql<false, false, false, false, V16, KBLK>(q, static_cast<const uint16_t*>(k),                         \
            static_cast<const uint16_t*>(v), ids, id_stride, count, sparse, static_cast<const uint16_t*>(qg), out,      \
            nullptr, cap, scale);                                                                                       \
    }
W2_Q4_KERNEL(w2_sparse_q41, true, false)
W2_Q4_KERNEL(w2_sparse_q42, false, true)
W2_Q4_KERNEL(w2_sparse_q412, true, true)
#undef W2_Q4_KERNEL

// The decode tile (tf_fn_sparse_tile_parts: parts, p v over hi + lo) on today's V layout and on QL-3's.
#define W4_QL_PARTS(NAME, VT)                                                                                           \
    extern "C" __global__ void __launch_bounds__(kPromptThreads)                                                        \
    NAME(const float* q, const void* k, const void* v, const int* ids, long long id_stride, const int* count,           \
         const int* sparse, float* parts_o, float* parts_ms, long long cap, float scale) {                              \
        w4_sparse_ql<false, VT, true, true>(q, static_cast<const uint16_t*>(k), static_cast<const uint16_t*>(v), ids,   \
                                            id_stride, count, sparse, nullptr, parts_o, parts_ms, cap, scale);          \
    }
W4_QL_PARTS(w4_sparse_tile_parts_ref, false)
W4_QL_PARTS(w4_sparse_tile_parts_ql3, true)
#undef W4_QL_PARTS
#define W2_Q4_PARTS(NAME, V16, KBLK)                                                                                    \
    extern "C" __global__ void __launch_bounds__(kPromptThreads)                                                        \
    NAME(const float* q, const void* k, const void* v, const int* ids, long long id_stride, const int* count,           \
         const int* sparse, float* parts_o, float* parts_ms, long long cap, float scale) {                              \
        w4_sparse_ql<false, false, true, true, V16, KBLK>(q, static_cast<const uint16_t*>(k),                           \
            static_cast<const uint16_t*>(v), ids, id_stride, count, sparse, nullptr, parts_o, parts_ms, cap, scale);     \
    }
W2_Q4_PARTS(w2_sparse_tile_parts_q41, true, false)
W2_Q4_PARTS(w2_sparse_tile_parts_q42, false, true)
W2_Q4_PARTS(w2_sparse_tile_parts_q412, true, true)
#undef W2_Q4_PARTS

// kv_write (tf_kv_write) into QL-3's V layout: element (r, h, j) of the (len, kv_heads, 256) bf16 rows at slot pos0 + r
extern "C" __global__ void w4_kv_write_ql3(const uint16_t* src, uint16_t* cache, int len, int kv_heads, int total, int pos0) {
    const long long i = gid();
    if (i >= static_cast<long long>(len) * kv_heads * kPromptDims) return;
    const int j = static_cast<int>(i % kPromptDims);
    const int h = static_cast<int>((i / kPromptDims) % kv_heads);
    const int r = static_cast<int>(i / (static_cast<long long>(kPromptDims) * kv_heads));
    const long long slot = pos0 + r;
    const long long to = (((static_cast<long long>(h) * (total / 4) + slot / 4) * (kPromptDims / 2) + j / 2) * 4 + slot % 4) * 2 + j % 2;
    cache[to] = src[i];
}

// kv_write into Q4-2's K layout: element (r, h, j) of the (len, kv_heads, 256) bf16 rows at slot pos0 + r
extern "C" __global__ void w2_kv_write_kblk(const uint16_t* src, uint16_t* cache, int len, int kv_heads, int total, int pos0) {
    const long long i = gid();
    if (i >= static_cast<long long>(len) * kv_heads * kPromptDims) return;
    const int j = static_cast<int>(i % kPromptDims);
    const int h = static_cast<int>((i / kPromptDims) % kv_heads);
    const int r = static_cast<int>(i / (static_cast<long long>(kPromptDims) * kv_heads));
    const long long slot = pos0 + r;
    const long long to = ((static_cast<long long>(h) * (total / 4) + slot / 4) * 16 + j / 16) * 64 + (slot % 4) * 16 + j % 16;
    cache[to] = src[i];
}

}  // namespace
