// EXL3 routed experts on RDNA (GLM-5.3-Flash's 4-bit mcg trellis experts): the trellis decode, the input rotation,
// the one-row GEMV and the WMMA prompt GEMM over the MoE plan (tf_moe_route's items), and the two epilogues.
//
// W = diag(suh) H W_q H diag(svh) (zig/src/core/exl3_format.zig dequantize), so x W = ((x suh) H) W_q H svh:
//   1. tf_exl3_rotate: xh = fp16(((x * suh) @ H) / sqrt(128)), a row (one rotation for every expert) or a pair;
//   2. tf_exl3_gemv_<MR> / tf_exl3_gemm: z = xh @ W_q in fp32 for each item's member pairs (split-k partials);
//   3. tf_exl3_gateup_epilogue: gate and up through H and svh, the SwiGLU, then * suh_d, H, fp16 (the down inputs);
//   4. the same products over the down matrices, and tf_exl3_down_epilogue: H and svh_d, fp32 [pair][D].
// The arithmetic is zig/kernels/cuda/exl3_experts.cu's (tf_exl3_experts_rot_in, tf_exl3_gateup_epilogue,
// tf_exl3_down_combine): fp16 rotated inputs, fp32 sums, GLM's bf16 roundings in the SwiGLU. The GEMV follows the
// format bench's gemv_kernel (a decoded tile dotted with every row), the GEMM its gemm_kernel (trellis tiles decoded
// into LDS, v_wmma_f32_16x16x16_f16), both on hipfmt/exl3 1e95cb74. Every pair's sums run in one order whatever the
// rows, the item cut or the pass a pair falls in, so a row's bits are the row's alone.
//
// Plan: item z = (expert, first, count) takes pairs members[first .. first + count); an expert id >= `experts` (the
// shared expert) and an empty item are skipped. Pair p = row * slots + slot; a gate/up input row is p / x_div.
// Trellis: per expert a pointer to int16 [K/16][N/16][16 * bits] (zig/src/families/glm/cuda_exl3.zig load).

#include <hip/hip_runtime.h>

#include <cstdint>

#include "common/arch.hpp"
#include "quant/exl3.hpp"

namespace {

using namespace tf::rocm::exl3;

// ---- 1. the trellis decode alone: W_q fp16 [K, N] from a matrix's tiles (the decoder's contract test) -----------

template <int K2>
__device__ void unpack_tile(const uint32_t* tr, int codebook, int tiles_n, uint16_t* out) {
    const int tn = blockIdx.x, tk = blockIdx.y, lane = threadIdx.x;
    const uint32_t* tile = tr + (static_cast<size_t>(tk) * tiles_n + tn) * tile_words<K2>();
    uint32_t w[lane_words<K2>()];
    lane_load<K2>(tile, lane, w);
    uint32_t p[4];
    if (codebook == kMcg) decode_lane<K2, kMcg>(w, lane, p);
    else if (codebook == kMul1) decode_lane<K2, kMul1>(w, lane, p);
    else decode_lane<K2, kInst3>(w, lane, p);
    const int n = tiles_n * 16;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint16_t v = static_cast<uint16_t>(p[j >> 1] >> (16 * (j & 1)));
        out[static_cast<size_t>(tk * 16 + value_row(lane, j)) * n + tn * 16 + value_col(lane, j)] = v;
    }
}

}  // namespace

// Grid (tiles_n, tiles_k), 32 threads. k2: half-bits a value (2 .. 16); codebook: exl3_format.zig's order.
extern "C" __global__ void __launch_bounds__(32) tf_exl3_unpack(const uint32_t* tr, int k2, int codebook, int tiles_n,
                                                                uint16_t* out) {
    switch (k2) {
#define TF_EXL3_UNPACK(K2_) \
    case K2_: unpack_tile<K2_>(tr, codebook, tiles_n, out); break;
        TF_EXL3_UNPACK(2)
        TF_EXL3_UNPACK(3)
        TF_EXL3_UNPACK(4)
        TF_EXL3_UNPACK(5)
        TF_EXL3_UNPACK(6)
        TF_EXL3_UNPACK(7)
        TF_EXL3_UNPACK(8)
        TF_EXL3_UNPACK(9)
        TF_EXL3_UNPACK(10)
        TF_EXL3_UNPACK(11)
        TF_EXL3_UNPACK(12)
        TF_EXL3_UNPACK(13)
        TF_EXL3_UNPACK(14)
        TF_EXL3_UNPACK(15)
        TF_EXL3_UNPACK(16)
#undef TF_EXL3_UNPACK
        default: __builtin_trap();
    }
}

// ---- 2. the input rotation ----------------------------------------------------------------------------------------

// Grid (units, K / 128, mats), 32 threads. per_pair 0: unit = row, one rotation (suh0 = expert 0's, every expert's
// gate and up the same), out0 [rows][K]. per_pair 1: unit = pair p of a routed pick, its expert's suh0 (gate) and suh1
// (up), out0 / out1 [pairs][K]; picks >= experts (the shared expert) are skipped. x: bf16 rows, x_stride apart.
extern "C" __global__ void __launch_bounds__(32) tf_exl3_rotate(const uint16_t* x, int x_stride, const int* pick,
                                                                const uint16_t* suh0, const uint16_t* suh1,
                                                                uint16_t* out0, uint16_t* out1, int k, int slots,
                                                                int experts, int per_pair) {
    const int u = blockIdx.x, blk = blockIdx.y, mat = blockIdx.z, lane = threadIdx.x;
    int row = u, e = 0;
    if (per_pair) {
        e = pick[u];
        if (e < 0 || e >= experts) return;
        row = u / slots;
    }
    const int c = blk * 128 + 4 * lane;
    const uint16_t* suh = (mat ? suh1 : suh0) + static_cast<size_t>(e) * k + c;
    const uint16_t* xr = x + static_cast<size_t>(row) * x_stride + c;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = bf16_value(xr[j]) * f16_value(suh[j]);
    fwht128(v, lane);
    uint16_t* o = (mat ? out1 : out0) + static_cast<size_t>(u) * k + c;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = f16_bits(v[j] * kHad);
}

// ---- 3. the decode GEMV over the plan -----------------------------------------------------------------------------
// Grid (N / 128 * sk, items, mats), 256 threads (8 waves). A block owns one item, one 128-column block (8 tiles) and
// split `sk`'s k range of K / 16 / sk tiles. Wave w: column group w % 4 (2 tiles), k half w / 4. Each pass stages MR
// member rows of the k range into LDS; a decoded tile is dotted with every row. Sums: a lane's dot2 chain over its k
// tiles, the quad of lanes, then the two k halves: one order for each pair. z[((mat sk + split) P + p) N + col].

namespace {

constexpr int kGemvWaves = 8;
constexpr int kGemvGroups = 4;  // column groups of 2 tiles
constexpr int kGemvHalves = kGemvWaves / kGemvGroups;

template <int K2, int CB, int MR>
__device__ void gemv(const uint16_t* X0, const uint16_t* X1, int x_div, const uint64_t* tp0, const uint64_t* tp1,
                     const int* items, const int* members, float* z, int k, int n, int pairs, int sk, int experts) {
    constexpr int TW = tile_words<K2>();
    const int it = blockIdx.y, mat = blockIdx.z;
    const int e = items[3 * it], first = items[3 * it + 1], count = items[3 * it + 2];
    if (count <= 0 || e < 0 || e >= experts) return;
    const int Kt = k >> 4, Nt = n >> 4, NB = n >> 7;
    const int nb = blockIdx.x % NB, split = blockIdx.x / NB;
    const int KPB = Kt / sk, kb0 = split * KPB, KB = KPB * 16;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int cg = warp % kGemvGroups, wk = warp / kGemvGroups;
    const int KPW = KPB / kGemvHalves, kt0 = wk * KPW;
    const int t = lane & 3, g = lane >> 2;
    const uint16_t* X = mat ? X1 : X0;
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? tp1[e] : tp0[e]);

    extern __shared__ __attribute__((aligned(16))) unsigned char smem[];
    _Float16* xs = reinterpret_cast<_Float16*>(smem);                          // [MR][KB]
    float* red = reinterpret_cast<float*>(smem + static_cast<size_t>(MR) * KB * 2);  // [waves][MR][32]

    for (int h0 = 0; h0 < count; h0 += MR) {
        const int rows = count - h0 < MR ? count - h0 : MR;
        // the pass's rows of the k range, 16 bytes a load (rows past the item's: zero)
        const int chunks = KB / 8;
        for (int c = threadIdx.x; c < MR * chunks; c += blockDim.x) {
            const int m = c / chunks, q = c % chunks;
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (m < rows) {
                const int p = members[first + h0 + m];
                v = *reinterpret_cast<const uint4*>(X + static_cast<size_t>(p / x_div) * k + kb0 * 16 + q * 8);
            }
            *reinterpret_cast<uint4*>(xs + static_cast<size_t>(m) * KB + q * 8) = v;
        }
        __syncthreads();

        float acc[MR][2][2];
#pragma unroll
        for (int m = 0; m < MR; ++m)
#pragma unroll
            for (int i = 0; i < 2; ++i) acc[m][i][0] = acc[m][i][1] = 0.f;
        const _Float16* xr = xs + 2 * t;
        for (int kt = 0; kt < KPW; ++kt) {
            const int kl = kt0 + kt;
            const uint32_t* row = T + (static_cast<size_t>(kb0 + kl) * Nt + nb * 8 + cg * 2) * TW;
            h2 xa[MR], xb[MR];
#pragma unroll
            for (int m = 0; m < MR; ++m) {
                xa[m] = *reinterpret_cast<const h2*>(xr + static_cast<size_t>(m) * KB + kl * 16);
                xb[m] = *reinterpret_cast<const h2*>(xr + static_cast<size_t>(m) * KB + kl * 16 + 8);
            }
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                uint32_t w[lane_words<K2>()];
                lane_load<K2>(row + i * TW, lane, w);
                uint32_t p[4];
                decode_lane<K2, CB>(w, lane, p);
#pragma unroll
                for (int m = 0; m < MR; ++m) {
                    acc[m][i][0] = dot2(p[1], xb[m], dot2(p[0], xa[m], acc[m][i][0]));
                    acc[m][i][1] = dot2(p[3], xb[m], dot2(p[2], xa[m], acc[m][i][1]));
                }
            }
        }
#pragma unroll
        for (int m = 0; m < MR; ++m)
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                acc[m][i][0] = quad_sum(acc[m][i][0]);
                acc[m][i][1] = quad_sum(acc[m][i][1]);
            }
        if (t == 0)
#pragma unroll
            for (int m = 0; m < MR; ++m)
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    red[(warp * MR + m) * 32 + i * 16 + g] = acc[m][i][0];
                    red[(warp * MR + m) * 32 + i * 16 + g + 8] = acc[m][i][1];
                }
        __syncthreads();
        for (int idx = threadIdx.x; idx < rows * 128; idx += blockDim.x) {
            const int m = idx >> 7, c = idx & 127, grp = c >> 5, w = c & 31;
            float s = red[(grp * MR + m) * 32 + w];
#pragma unroll
            for (int q = 1; q < kGemvHalves; ++q) s += red[((q * kGemvGroups + grp) * MR + m) * 32 + w];
            const int p = members[first + h0 + m];
            z[((static_cast<size_t>(mat) * sk + split) * pairs + p) * n + nb * 128 + c] = s;
        }
        __syncthreads();
    }
}

}  // namespace

#define TF_EXL3_GEMV(MR_)                                                                                            \
    extern "C" __global__ void __launch_bounds__(256) tf_exl3_gemv_##MR_(                                            \
        const uint16_t* X0, const uint16_t* X1, int x_div, const uint64_t* tp0, const uint64_t* tp1, const int* items, \
        const int* members, float* z, int k, int n, int pairs, int sk, int experts) {                                \
        gemv<8, kMcg, MR_>(X0, X1, x_div, tp0, tp1, items, members, z, k, n, pairs, sk, experts);                    \
    }
TF_EXL3_GEMV(1)
TF_EXL3_GEMV(2)
TF_EXL3_GEMV(4)
TF_EXL3_GEMV(8)
#undef TF_EXL3_GEMV

// ---- 4. the prompt GEMM over the plan (gfx11 WMMA) ----------------------------------------------------------------
// Grid (N / 128, items, mats), 256 threads (8 waves: 2 along m x 4 along n). A block owns one item and one 128-column
// block; passes of 64 member rows. A k step of 32: the step's 2 x 8 trellis tiles decoded into LDS as fp16 [n][k] (the
// B layout: a lane a column, 16 k), the rows' x into LDS [m][k], two v_wmma_f32_16x16x16_f16 a 16 x 16 accumulator;
// the next step's loads go to registers while this step computes. A sum is the WMMA's over k in order: one order for
// each pair. z[(mat P + p) N + col], fp32 before the output rotation.

namespace {

constexpr int kBM = 64, kBK = 32, kLK = kBK + 8;  // rows a pass, k a step, halves an LDS row

typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef float f8 __attribute__((ext_vector_type(8)));

template <int K2, int CB>
__device__ void gemm(const uint16_t* X0, const uint16_t* X1, int x_div, const uint64_t* tp0, const uint64_t* tp1,
                     const int* items, const int* members, float* z, int k, int n, int pairs, int experts) {
#if TF_DEVICE_WMMA_GFX11
    constexpr int TW = tile_words<K2>();
    const int it = blockIdx.y, mat = blockIdx.z, nb = blockIdx.x;
    const int e = items[3 * it], first = items[3 * it + 1], count = items[3 * it + 2];
    if (count <= 0 || e < 0 || e >= experts) return;
    __shared__ __attribute__((aligned(16))) _Float16 As[2][kBM * kLK];
    __shared__ __attribute__((aligned(16))) _Float16 Bs[2][128 * kLK];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int wm = warp & 1, wn = warp >> 1;  // rows 32 wm.., columns 32 wn..
    const int t = lane & 3, g = lane >> 2, l16 = lane & 15;
    const int Nt = n >> 4, iters = k / kBK;
    const uint16_t* X = mat ? X1 : X0;
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? tp1[e] : tp0[e]);
    // a thread's x chunk: row tid / 4, 8 halves at k (tid % 4) * 8; a wave's 2 trellis tiles: q = 2 warp + u
    const int arow = threadIdx.x >> 2, akc = threadIdx.x & 3;

    for (int h0 = 0; h0 < count; h0 += kBM) {
        const int rows = count - h0 < kBM ? count - h0 : kBM;
        const bool alive = arow < rows;
        const uint16_t* xa = X + (alive ? static_cast<size_t>(members[first + h0 + arow] / x_div) * k : 0) + akc * 8;
        uint4 ra;
        uint32_t rb[2][lane_words<K2>()];
        auto gload = [&](int s) {
            ra = alive ? *reinterpret_cast<const uint4*>(xa + s * kBK) : make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                const int q = 2 * warp + u, kk = q >> 3, nt = q & 7;
                lane_load<K2>(T + (static_cast<size_t>(s * 2 + kk) * Nt + nb * 8 + nt) * TW, lane, rb[u]);
            }
        };
        auto sstore = [&](int buf) {
            *reinterpret_cast<uint4*>(&As[buf][arow * kLK + akc * 8]) = ra;
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                const int q = 2 * warp + u, kk = q >> 3, nt = q & 7;
                uint32_t p[4];
                decode_lane<K2, CB>(rb[u], lane, p);
                _Float16* c0 = &Bs[buf][(nt * 16 + g) * kLK + kk * 16 + 2 * t];
                _Float16* c1 = c0 + 8 * kLK;
                *reinterpret_cast<uint32_t*>(c0) = p[0];
                *reinterpret_cast<uint32_t*>(c0 + 8) = p[1];
                *reinterpret_cast<uint32_t*>(c1) = p[2];
                *reinterpret_cast<uint32_t*>(c1 + 8) = p[3];
            }
        };
        f8 acc[2][2];
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int j = 0; j < 2; ++j) acc[i][j] = f8{0, 0, 0, 0, 0, 0, 0, 0};
        gload(0);
        sstore(0);
        __syncthreads();
        for (int s = 0; s < iters; ++s) {
            const int buf = s & 1;
            if (s + 1 < iters) gload(s + 1);
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                h16 af[2], bf[2];
#pragma unroll
                for (int i = 0; i < 2; ++i) af[i] = *reinterpret_cast<const h16*>(&As[buf][(wm * 32 + i * 16 + l16) * kLK + kk * 16]);
#pragma unroll
                for (int j = 0; j < 2; ++j) bf[j] = *reinterpret_cast<const h16*>(&Bs[buf][(wn * 32 + j * 16 + l16) * kLK + kk * 16]);
#pragma unroll
                for (int i = 0; i < 2; ++i)
#pragma unroll
                    for (int j = 0; j < 2; ++j) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af[i], bf[j], acc[i][j]);
            }
            if (s + 1 < iters) sstore(buf ^ 1);
            __syncthreads();
        }
        // accumulator element r of lane l: row 2 r + l / 16, column l % 16 of its 16 x 16 block
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int m = wm * 32 + i * 16 + 2 * r + (lane >> 4);
                if (m >= rows) continue;
                const int p = members[first + h0 + m];
                float* o = z + (static_cast<size_t>(mat) * pairs + p) * n + nb * 128 + wn * 32 + l16;
#pragma unroll
                for (int j = 0; j < 2; ++j) o[j * 16] = acc[i][j][r];
            }
        __syncthreads();
    }
#else
    __builtin_trap();
#endif
}

}  // namespace

extern "C" __global__ void __launch_bounds__(256) tf_exl3_gemm(const uint16_t* X0, const uint16_t* X1, int x_div,
                                                               const uint64_t* tp0, const uint64_t* tp1, const int* items,
                                                               const int* members, float* z, int k, int n, int pairs,
                                                               int experts) {
    gemm<8, kMcg>(X0, X1, x_div, tp0, tp1, items, members, z, k, n, pairs, experts);
}

// ---- 5. the epilogues -----------------------------------------------------------------------------------------------

// Grid (pairs, I / 128), 32 threads: a routed pair's gate and up (splits summed in order) through H and svh, rounded to
// bf16 and limited, silu to bf16, the product to bf16; then * suh_d, H, fp16: xd [pair][I], the down inputs.
extern "C" __global__ void __launch_bounds__(32) tf_exl3_gateup_epilogue(const float* z, const int* pick,
                                                                         const uint16_t* svh_g, const uint16_t* svh_u,
                                                                         const uint16_t* suh_d, uint16_t* xd, int pairs,
                                                                         int n, int sk, int experts, float limit) {
    const int p = blockIdx.x, blk = blockIdx.y, lane = threadIdx.x;
    const int e = pick[p];
    if (e < 0 || e >= experts) return;
    const int c = blk * 128 + 4 * lane;
    float gv[4], uv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float sg = 0.f, su = 0.f;
        for (int s = 0; s < sk; ++s) {
            sg += z[(static_cast<size_t>(s) * pairs + p) * n + c + j];
            su += z[(static_cast<size_t>(sk + s) * pairs + p) * n + c + j];
        }
        gv[j] = sg;
        uv[j] = su;
    }
    fwht128(gv, lane);
    fwht128(uv, lane);
    const size_t at = static_cast<size_t>(e) * n + c;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float gg = bf16r(gv[j] * kHad * f16_value(svh_g[at + j]));
        float uu = bf16r(uv[j] * kHad * f16_value(svh_u[at + j]));
        if (limit > 0) {
            gg = up_to(gg, limit);
            uu = within(uu, limit);
        }
        const float act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
        v[j] = act * f16_value(suh_d[at + j]);
    }
    fwht128(v, lane);
    uint16_t* o = xd + static_cast<size_t>(p) * n + c;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = f16_bits(v[j] * kHad);
}

// Grid (pairs, D / 128), 32 threads: a routed pair's down sums (splits in order) through H and svh_d: y [pair][D] fp32.
// The shared expert's pairs (and any pick >= experts) keep y as it is.
extern "C" __global__ void __launch_bounds__(32) tf_exl3_down_epilogue(const float* z, const int* pick,
                                                                       const uint16_t* svh_d, float* y, int pairs, int d,
                                                                       int sk, int experts) {
    const int p = blockIdx.x, blk = blockIdx.y, lane = threadIdx.x;
    const int e = pick[p];
    if (e < 0 || e >= experts) return;
    const int c = blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float s = 0.f;
        for (int q = 0; q < sk; ++q) s += z[(static_cast<size_t>(q) * pairs + p) * d + c + j];
        v[j] = s;
    }
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) y[static_cast<size_t>(p) * d + c + j] = v[j] * kHad * f16_value(svh_d[static_cast<size_t>(e) * d + c + j]);
}
