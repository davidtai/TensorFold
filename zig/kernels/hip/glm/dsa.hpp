#pragma once

// GLM-5.3-Flash's DSA layers on gfx11 (64 heads, NoPE MLA on a 512-wide latent, qk 256, v 256; indexer 32 x 128, pools
// of 4): the latent cache write, the absorb and expand by kv_b, the index key LayerNorm and pool compression, the pool
// scores, and the latent attention (dense or over a row's listed tokens) in 512-key parts and their merge.
// The CUDA counterparts: zig/kernels/cuda/glm_glue_latent.cuh, glm_glue_index.cuh, glm_glue_attn.cuh, glm_ops.cu.
// The pools' top 512 and the token lists are Flash Next's tf_fn_idx_select (flashnext/indexer.hpp): the same rule
// (ties to the lower pool, ascending order, the incomplete pool's tail after them) over the scores written here.
// Each output's arithmetic depends on its row and head only: a row alone gives the bits it gets in any window.

#include "ops/common.hpp"
#include "common/wmma.hpp"

namespace {

constexpr int kLw = 512;    // latent width (kv_lora_rank)
constexpr int kQk = 256;    // a head's q / k width
constexpr int kDv = 256;    // a head's v width
constexpr int kIdxD = 128;  // index head width
constexpr int kIdxHeads = 32;
constexpr int kPool = 4;

// ---- The latent cache: bf16 [capacity][512], position p's row at cache + p rows.

// Row r of `lat` (stride `lat_stride` elements) into slot pos[0] + row0 + r; block r, 64 threads of 16 bytes.
extern "C" __global__ void __launch_bounds__(64)
tf_glm_lat_write(const uint16_t* __restrict__ lat, long long lat_stride, uint16_t* __restrict__ cache,
                 const int* __restrict__ pos, int row0) {
    const long long r = blockIdx.x, at = static_cast<long long>(pos[0]) + row0 + r;
    const uint4* src = reinterpret_cast<const uint4*>(lat + r * lat_stride);
    reinterpret_cast<uint4*>(cache + at * kLw)[threadIdx.x] = src[threadIdx.x];
}

// ---- The absorb: qa [rows][heads][512] bf16 = q [rows][heads][256] . wk [heads][256][512], fp32 fma in k order.
// Block (head, 256 columns, RB rows), a thread a column: wk's rows read once a block, the rows' q in LDS.
constexpr int kAbsorbRows = 16;

extern "C" __global__ void __launch_bounds__(256)
tf_glm_absorb_q(const uint16_t* __restrict__ q, const uint16_t* __restrict__ wk, uint16_t* __restrict__ qa, int rows,
                int heads) {
    __shared__ float qs[kAbsorbRows][kQk];
    const int h = blockIdx.x, n = blockIdx.y * 256 + threadIdx.x, r0 = blockIdx.z * kAbsorbRows;
    const int nr = min(kAbsorbRows, rows - r0);
    for (int i = threadIdx.x; i < nr * kQk; i += 256) {
        const int j = i / kQk, k = i % kQk;
        qs[j][k] = bf16_to_float(q[(static_cast<long long>(r0 + j) * heads + h) * kQk + k]);
    }
    __syncthreads();
    float acc[kAbsorbRows];
#pragma unroll
    for (int j = 0; j < kAbsorbRows; ++j) acc[j] = 0.f;
    const uint16_t* w = wk + static_cast<long long>(h) * kQk * kLw + n;
    for (int k = 0; k < kQk; ++k) {
        const float wv = bf16_to_float(w[static_cast<long long>(k) * kLw]);
#pragma unroll
        for (int j = 0; j < kAbsorbRows; ++j) acc[j] = fmaf(qs[j][k], wv, acc[j]);
    }
    for (int j = 0; j < nr; ++j) qa[(static_cast<long long>(r0 + j) * heads + h) * kLw + n] = float_to_bf16(acc[j]);
}

// ---- The expand: out [rows][heads][256] bf16 = o_lat [rows][heads][512] . wv [heads][256][512]^T.
// Block (head, 32 outputs, RB rows), 8 waves of 4 outputs: lane l holds latent columns 16 l .. 16 l + 15 of the
// output's wv row, adds its 16 products in order, then the lanes by xor 16, 8, 4, 2, 1.
constexpr int kExpandRows = 8;

extern "C" __global__ void __launch_bounds__(256)
tf_glm_expand_v(const uint16_t* __restrict__ ol, const uint16_t* __restrict__ wv, uint16_t* __restrict__ out, int rows,
                int heads) {
    __shared__ float os[kExpandRows][kLw];
    const int h = blockIdx.x, r0 = blockIdx.z * kExpandRows;
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int nr = min(kExpandRows, rows - r0);
    for (int i = threadIdx.x; i < nr * kLw; i += 256) {
        const int j = i / kLw, k = i % kLw;
        os[j][k] = bf16_to_float(ol[(static_cast<long long>(r0 + j) * heads + h) * kLw + k]);
    }
    __syncthreads();
    for (int o = 0; o < 4; ++o) {
        const int jo = blockIdx.y * 32 + wave * 4 + o;
        const uint16_t* wrow = wv + (static_cast<long long>(h) * kDv + jo) * kLw + lane * 16;
        float w[16];
        const uint4 a = reinterpret_cast<const uint4*>(wrow)[0], b = reinterpret_cast<const uint4*>(wrow)[1];
        const uint32_t ws[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            w[2 * i] = bf16_to_float(static_cast<uint16_t>(ws[i] & 0xffffu));
            w[2 * i + 1] = bf16_to_float(static_cast<uint16_t>(ws[i] >> 16));
        }
        for (int j = 0; j < nr; ++j) {
            float s = 0.f;
#pragma unroll
            for (int i = 0; i < 16; ++i) s = fmaf(os[j][lane * 16 + i], w[i], s);
#pragma unroll
            for (int m = 16; m > 0; m >>= 1) s += __shfl_xor(s, m, 32);
            if (lane == 0) out[(static_cast<long long>(r0 + j) * heads + h) * kDv + jo] = float_to_bf16(s);
        }
    }
}

// ---- The indexer's keys: sparse._index_write and sparse._pool_keys.

// A wave a row: the LayerNorm (with bias) of k_raw's 128 values into ik at pos[0] + row0 + r, the gate row into ig.
extern "C" __global__ void __launch_bounds__(32)
tf_glm_index_write(const uint16_t* __restrict__ kr, long long k_stride, const float* __restrict__ gr,
                   const uint16_t* __restrict__ lnw, const uint16_t* __restrict__ lnb, uint16_t* ik, uint16_t* ig,
                   const int* __restrict__ pos, int row0, float eps) {
    const long long r = blockIdx.x, at = static_cast<long long>(pos[0]) + row0 + r;
    const int d0 = threadIdx.x * 4;
    float x[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) x[j] = bf16_to_float(kr[r * k_stride + d0 + j]);
    float s = ((x[0] + x[1]) + x[2]) + x[3];
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) s += __shfl_xor(s, m, 32);
    const float mean = s / static_cast<float>(kIdxD);
    float xc[4], v = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        xc[j] = x[j] - mean;
        v = fmaf(xc[j], xc[j], v);
    }
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor(v, m, 32);
    const float sd = sqrtf(v / static_cast<float>(kIdxD) + eps);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int d = d0 + j;
        ik[at * kIdxD + d] = float_to_bf16(fmaf(xc[j] / sd, bf16_to_float(lnw[d]), bf16_to_float(lnb[d])));
        ig[at * kIdxD + d] = float_to_bf16(gr[r * kIdxD + d]);
    }
}

// Block x: pool P / 4 + x if a window row (pos[0] + row0 .. + rows - 1) completes it; a wave, 4 dims a lane. A dim's
// 4 gate logits (+ ape) by softmax, each weight rounded to bf16, times the bf16 key, rounded, summed in order.
extern "C" __global__ void __launch_bounds__(32)
tf_glm_pool_keys(const uint16_t* ik, const uint16_t* ig, const uint16_t* __restrict__ ape, uint16_t* __restrict__ pk,
                 const int* __restrict__ pos, int row0, int rows) {
    const long long first = static_cast<long long>(pos[0]) + row0, p = first / kPool + blockIdx.x;
    if (kPool * p + kPool - 1 > first + rows - 1) return;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int d = threadIdx.x * 4 + j;
        float l[kPool], e[kPool];
#pragma unroll
        for (int q = 0; q < kPool; ++q) l[q] = bf16_to_float(ig[(kPool * p + q) * kIdxD + d]) + bf16_to_float(ape[q * kIdxD + d]);
        const float m = fmaxf(fmaxf(l[0], l[1]), fmaxf(l[2], l[3]));
#pragma unroll
        for (int q = 0; q < kPool; ++q) e[q] = expf(l[q] - m);
        const float s = ((e[0] + e[1]) + e[2]) + e[3];
        float t = 0.f;
#pragma unroll
        for (int q = 0; q < kPool; ++q) {
            const float wq = bf16_to_float(float_to_bf16(e[q] / s));
            const float tq = bf16_to_float(float_to_bf16(wq * bf16_to_float(ik[(kPool * p + q) * kIdxD + d])));
            t = q == 0 ? tq : t + tq;
        }
        pk[p * kIdxD + d] = float_to_bf16(t);
    }
}

// ---- The pool scores: score[r][p] = sum_h (w_h wscale) relu(scale q_h . pk_p) over the 32 index heads, heads in order,
// for the row's complete pools (p < (P + 1) / 4, P = pos[0] + row0 + r); -inf from there up to the launch's columns.
// Rows with at most `top` complete pools attend densely: none is scored. Block (row, 64 pools), 4 waves of 16 pools,
// lane l holds dims 4 l .. 4 l + 3. The block of pools 0 also writes the row's complete (pools) and end (P + 1) for
// tf_fn_idx_select. A score is written + 0 (so -0 is +0: the two are one order key, as sparse._order_key).
extern "C" __global__ void __launch_bounds__(128)
tf_glm_scores(const uint16_t* __restrict__ qi, const uint16_t* __restrict__ wts, long long w_stride,
              const uint16_t* __restrict__ pk, float* __restrict__ out, long long np, int* __restrict__ complete,
              int* __restrict__ ends, const int* __restrict__ pos, int row0, int cols, float scale, float wscale,
              int top) {
    __shared__ float qs[kIdxHeads][kIdxD];
    __shared__ float ws[kIdxHeads];
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const long long r = blockIdx.x;
    const int at = pos[0] + row0 + static_cast<int>(r);
    const int npool = (at + 1) / kPool;
    if (blockIdx.y == 0 && threadIdx.x == 0) {
        complete[r] = npool;
        ends[r] = at + 1;
    }
    if (npool <= top) return;
    for (int i = threadIdx.x; i < kIdxHeads * kIdxD; i += 128)
        qs[i / kIdxD][i % kIdxD] = bf16_to_float(qi[r * kIdxHeads * kIdxD + i]);
    if (threadIdx.x < kIdxHeads) ws[threadIdx.x] = bf16_to_float(wts[r * w_stride + threadIdx.x]) * wscale;
    __syncthreads();
    const int lim = min(cols, static_cast<int>(np));
    for (int i = 0; i < 16; ++i) {
        const int p = blockIdx.y * 64 + wave * 16 + i;
        if (p >= lim) break;
        if (p >= npool) {
            if (lane == 0) out[r * np + p] = -INFINITY;
            continue;
        }
        const uint2 kv = *reinterpret_cast<const uint2*>(pk + static_cast<long long>(p) * kIdxD + lane * 4);
        const float k0 = bf16_to_float(static_cast<uint16_t>(kv.x & 0xffffu)), k1 = bf16_to_float(static_cast<uint16_t>(kv.x >> 16));
        const float k2 = bf16_to_float(static_cast<uint16_t>(kv.y & 0xffffu)), k3 = bf16_to_float(static_cast<uint16_t>(kv.y >> 16));
        float total = 0.f;
        for (int h = 0; h < kIdxHeads; ++h) {
            const float* qh = &qs[h][lane * 4];
            float d = qh[0] * k0;
            d = fmaf(qh[1], k1, d);
            d = fmaf(qh[2], k2, d);
            d = fmaf(qh[3], k3, d);
#pragma unroll
            for (int m = 16; m > 0; m >>= 1) d += __shfl_xor(d, m, 32);
            total = fmaf(ws[h], fmaxf(d * scale, 0.f), total);
        }
        if (lane == 0) out[r * np + p] = total + 0.f;
    }
}

// ---- The latent attention on the gfx11 matrix cores (v_wmma_f32_16x16x16_bf16).
// Row r's keys: with `ids` null, positions 0 .. pos[0] + row0 + r (dense, causal); else its first count[r] entries of
// ids (sparse[r] set) or positions 0 .. count[r] - 1 (sparse[r] 0): tf_fn_idx_select's lists. Block (row, HB heads,
// part c): the row's key entries c chunk .. (c + 1) chunk - 1, 16 keys a step, its unnormalized o, max and sum into
// parts (layout [(r heads + h) parts + c]); tf_glm_merge combines the parts in order.
// The key rows are the values too (MLA: v = the latent). 8 waves; wave w owns latent dims 64 w .. 64 w + 63:
//   q.k: A = qa (16 heads x 16 dims, bf16 as stored), B = keys (16 dims x 16 keys), straight from the cache; the waves'
//        partial scores added in wave order in LDS, then x scale. qa and the cache are bf16, so the products are exact.
//   softmax: fp32, online, natural exp.
//   p.v: A = p as bf16 hi + lo (p keeps about 16 bits), B = the wave's 64 dims of the step's 16 key rows, from a
//        transposed wave-private LDS copy of what the wave read for q.k.
// gfx11 fragments (wave32): A lane l holds all 16 k of row l % 16, B lane l all 16 k of column l % 16 (lanes 16-31
// repeat 0-15), D lane l element i holds D[2 i + l / 16][l % 16].
constexpr int kAttnThreads = 256;
constexpr int kAttnWaves = 8;
constexpr int kAttnStep = 16;
constexpr int kAttnPRow = kAttnStep + 4;
constexpr int kAttnVRow = kAttnStep + 8;  // a transposed dim's 16 keys, 48 bytes: 16-byte aligned rows

template <int HB>
struct alignas(16) AttnSmem {
    float part[kAttnWaves][HB][kAttnStep + 1];
    float p[HB][kAttnPRow];
    uint16_t vt[kAttnWaves][64][kAttnVRow];
    float m[HB], l[HB], alpha[HB];
};

__device__ __forceinline__ uint32_t attn_pack(uint16_t lo, uint16_t hi) {
    return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}

template <int HB>
__device__ void latent_parts(const uint16_t* __restrict__ qa, const uint16_t* __restrict__ cache,
                             const int* __restrict__ pos, int row0, const int* __restrict__ ids, long long id_stride,
                             const int* __restrict__ count, const int* __restrict__ sparse, float* __restrict__ po,
                             float* __restrict__ pms, int heads, int chunk, float scale) {
#if TF_DEVICE_WMMA_GFX11
    using tf::rocm::bf16x16;
    using tf::rocm::f32x8;
    using tf::rocm::u32x8;
    using Mma = tf::rocm::WmmaBF16;
    constexpr int MT = HB / 16;
    // softmax: 8 threads a head row, 2 cells a thread, for both head blocks (one sum order: the same bits)
    constexpr int CPT = 2;
    constexpr int TPR = kAttnStep / CPT;
    __shared__ AttnSmem<HB> S;
    const int r = blockIdx.x, h0 = blockIdx.y * HB, c = blockIdx.z, parts = gridDim.z;
    const int t = threadIdx.x, lane = t & 31, wave = t >> 5, col = lane & 15, half = lane >> 4;
    const int dim0 = wave * 64;
    const bool dense = ids == nullptr;
    const int n = dense ? pos[0] + row0 + r + 1 : count[r];
    const bool listed = !dense && sparse[r] != 0;
    const int* row_ids = dense ? nullptr : ids + r * id_stride;
    const int lo = c * chunk, hi = min(n, lo + chunk);

    // q fragments: head h0 + 16 mt + col, dims dim0 + 16 kk + 0..15
    bf16x16 qf[MT][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            const uint4* src = reinterpret_cast<const uint4*>(
                qa + (static_cast<long long>(r) * heads + h0 + 16 * mt + col) * kLw + dim0 + 16 * kk);
            const uint4 a = src[0], b = src[1];
            const u32x8 w = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
            qf[mt][kk] = __builtin_bit_cast(bf16x16, w);
        }
    if (t < HB) {
        S.m[t] = -INFINITY;
        S.l[t] = 0.f;
    }
    f32x8 acc[MT][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[mt][j] = f32x8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int c0 = lo; c0 < hi; c0 += kAttnStep) {
        // this lane's key: cell c0 + col, or the step's first key past the row's keys (finite values under p = 0)
        const int cell = c0 + col;
        const int e = cell < hi ? cell : c0;
        const int key = listed ? row_ids[e] : e;
        uint4 kx[8];
        {
            const uint4* src = reinterpret_cast<const uint4*>(cache + static_cast<long long>(key) * kLw + dim0);
#pragma unroll
            for (int i = 0; i < 8; ++i) kx[i] = src[i];
        }
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            f32x8 s = f32x8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                const u32x8 w = {kx[2 * kk].x, kx[2 * kk].y, kx[2 * kk].z, kx[2 * kk].w,
                                 kx[2 * kk + 1].x, kx[2 * kk + 1].y, kx[2 * kk + 1].z, kx[2 * kk + 1].w};
                s = Mma::mma(qf[mt][kk], __builtin_bit_cast(bf16x16, w), s);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) S.part[wave][16 * mt + 2 * i + half][col] = s[i];
        }
        // the wave's 64 dims of the step's keys, transposed: vt[wave][dim][key]; lanes 0-15 dims 0-31, 16-31 dims 32-63
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint4 v = kx[4 * half + i];
            const uint32_t w4[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int d = 32 * half + 8 * i + 2 * j;
                S.vt[wave][d][col] = static_cast<uint16_t>(w4[j] & 0xffffu);
                S.vt[wave][d + 1][col] = static_cast<uint16_t>(w4[j] >> 16);
            }
        }
        __syncthreads();
        // online softmax: head row t / TPR, cells CPT (t % TPR) ..; the waves' partials in wave order
        if (t < HB * TPR) {
            const int row = t / TPR, sub = t % TPR;
            float x[CPT], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < CPT; ++j) {
                const int k = sub * CPT + j;
                float sc = S.part[0][row][k];
#pragma unroll
                for (int w = 1; w < kAttnWaves; ++w) sc += S.part[w][row][k];
                x[j] = c0 + k < hi ? sc * scale : -INFINITY;
                mx = fmaxf(mx, x[j]);
            }
#pragma unroll
            for (int mask = 1; mask < TPR; mask <<= 1) mx = fmaxf(mx, __shfl_xor(mx, mask, 32));
            const float m_old = S.m[row];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.f;
#pragma unroll
            for (int j = 0; j < CPT; ++j) {
                const float pe = x[j] == -INFINITY ? 0.f : expf(x[j] - m_new);
                S.p[row][sub * CPT + j] = pe;
                sum += pe;
            }
#pragma unroll
            for (int mask = 1; mask < TPR; mask <<= 1) sum += __shfl_xor(sum, mask, 32);
            if (sub == 0) {
                const float f = m_old == -INFINITY ? 0.f : expf(m_old - m_new);
                S.alpha[row] = f;
                S.l[row] = S.l[row] * f + sum;
                S.m[row] = m_new;
            }
        }
        __syncthreads();
        // o = o alpha + p v over this wave's dims: 4 tiles of 16 dims, p as hi + lo
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            float a[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) a[i] = S.alpha[16 * mt + 2 * i + half];
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[mt][j][i] *= a[i];
            u32x8 hw, lw;
            const float* pr = &S.p[16 * mt + col][0];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float p0 = pr[2 * i], p1 = pr[2 * i + 1];
                const uint16_t g0 = float_to_bf16(p0), g1 = float_to_bf16(p1);
                hw[i] = attn_pack(g0, g1);
                lw[i] = attn_pack(float_to_bf16(p0 - bf16_to_float(g0)), float_to_bf16(p1 - bf16_to_float(g1)));
            }
            const bf16x16 ph = __builtin_bit_cast(bf16x16, hw), pl = __builtin_bit_cast(bf16x16, lw);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint4* src = reinterpret_cast<const uint4*>(&S.vt[wave][16 * j + col][0]);
                const uint4 v0 = src[0], v1 = src[1];
                const u32x8 w = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
                const bf16x16 b = __builtin_bit_cast(bf16x16, w);
                acc[mt][j] = Mma::mma(ph, b, acc[mt][j]);
                acc[mt][j] = Mma::mma(pl, b, acc[mt][j]);
            }
        }
        // no barrier: part and vt are rewritten before the next first barrier only by their own wave (part is read by
        // the others before the second barrier), p and alpha after it, which every wave reaches once its p.v is done
    }
    __syncthreads();
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int hl = 16 * mt + 2 * i + half;
            const long long row = (static_cast<long long>(r) * heads + h0 + hl) * parts + c;
#pragma unroll
            for (int j = 0; j < 4; ++j) po[row * kLw + dim0 + 16 * j + col] = acc[mt][j][i];
            if (wave == 0 && col == 0) {
                pms[row * 2] = S.m[hl];
                pms[row * 2 + 1] = S.l[hl];
            }
        }
#else
    __builtin_trap();
#endif
}

// The 16-head form (decode windows: 4 blocks a row a part) and the 32-head form (prompt chunks); the same bits.
extern "C" __global__ void __launch_bounds__(kAttnThreads)
tf_glm_latent_parts16(const uint16_t* qa, const uint16_t* cache, const int* pos, int row0, const int* ids,
                      long long id_stride, const int* count, const int* sparse, float* po, float* pms, int heads,
                      int chunk, float scale) {
    latent_parts<16>(qa, cache, pos, row0, ids, id_stride, count, sparse, po, pms, heads, chunk, scale);
}

extern "C" __global__ void __launch_bounds__(kAttnThreads)
tf_glm_latent_parts32(const uint16_t* qa, const uint16_t* cache, const int* pos, int row0, const int* ids,
                      long long id_stride, const int* count, const int* sparse, float* po, float* pms, int heads,
                      int chunk, float scale) {
    latent_parts<32>(qa, cache, pos, row0, ids, id_stride, count, sparse, po, pms, heads, chunk, scale);
}

// Block (head, row), 128 threads of 4 dims: the row's parts merged in part order (parts with no key skipped), o / l
// into out [rows][heads][512] bf16.
extern "C" __global__ void __launch_bounds__(128)
tf_glm_latent_merge(const float* __restrict__ po, const float* __restrict__ pms, uint16_t* __restrict__ out, int heads,
                    int parts) {
    const int h = blockIdx.x, r = blockIdx.y, k0 = threadIdx.x * 4;
    const long long at = (static_cast<long long>(r) * heads + h) * parts;
    float m = -INFINITY;
    for (int c = 0; c < parts; ++c)
        if (pms[(at + c) * 2 + 1] > 0.f) m = fmaxf(m, pms[(at + c) * 2]);
    float l = 0.f, o[4] = {0.f, 0.f, 0.f, 0.f};
    for (int c = 0; c < parts; ++c) {
        const float cl = pms[(at + c) * 2 + 1];
        if (!(cl > 0.f)) continue;
        const float w = expf(pms[(at + c) * 2] - m);
        l = fmaf(cl, w, l);
        const float4 v = *reinterpret_cast<const float4*>(po + (at + c) * kLw + k0);
        o[0] = fmaf(v.x, w, o[0]);
        o[1] = fmaf(v.y, w, o[1]);
        o[2] = fmaf(v.z, w, o[2]);
        o[3] = fmaf(v.w, w, o[3]);
    }
    uint16_t* dst = out + (static_cast<long long>(r) * heads + h) * kLw + k0;
#pragma unroll
    for (int j = 0; j < 4; ++j) dst[j] = float_to_bf16(l > 0.f ? o[j] / l : 0.f);
}

}  // namespace
