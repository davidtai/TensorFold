#pragma once

// A prompt's plain products of 4-bit group-32 MLX weights on the gfx11 matrix cores, K a compile-time constant.
//
// Each output's arithmetic is affine_wmma_gemm's (matrix_gemm.hpp): x as the A operand, two WMMA a 32-code group in
// the same K order (the first from zero), the group folded as fma(dot, scale) then fma(sum x, bias), the sum of x in
// the same order from zero. So the bytes equal the matrix tile's (with out16 set, the matrix tile's then tf_cast's to
// BF16). The block is the same 128 x 128; what changes:
//   - K in 64-wide windows (2 groups a window) in one LDS buffer (40 KB: 3 blocks a WGP against 2), two barriers a
//     window over LDS only (lds_barrier), AHEAD windows of loads held in registers across them;
//   - the group is a K stage at compile time: no select of the first WMMA's addend, no runtime group arithmetic;
//   - a wave owns 64 rows x 32 columns (4 x 2 tiles; 208 VGPRs at AHEAD 1 against 224);
//   - a thread loads one group of one x row (64 bytes) and adds its 32 values itself, and one group of one column's
//     codes;
//   - no branch on the live rows: rows past m read row m - 1 and are not stored.
// gfx1151 at 8,192 rows: 33.3-33.7 TFLOPS on the K 2560 trunk products against 27.6-27.9 (w2/dense_bench.hip).
// After Strata's kernel_bk64 (github.com/Niko1221/Strata, MIT, src/prefill/wmma_gemm.cu@4b9e05a2:396-482: K 64 a tile
// with the next tile in registers, two barriers a tile).

#include <utility>

#include "common/wmma.hpp"
#include "quant/mlx_decoder.hpp"
#include "tiles/epilogue.hpp"

namespace tf {
namespace rocm {

// HC_MIX (hc up, K 320): the 4 streams' columns of 32 neighbouring hidden columns d a block, mixed in the epilogue as
// tf_fn_hc_mix mixes them (x the normed streams [m, 4 dims] bf16, mixed [m, dims] bf16, dims = n / 4).
struct HcMix {
    const uint16_t* x = nullptr;
    uint16_t* mixed = nullptr;
};

namespace dense_tile {
constexpr int kBM = 128;
constexpr int kBN = 128;
constexpr int kKW = 64;  // K a window: 2 groups
constexpr int kThreads = 256;
}  // namespace dense_tile

// K: the product's K (a multiple of 64). AHEAD: windows of loads a thread holds (K / 64 a multiple of it). bf16 x and
// tables, plain products only. Blocks in row-major order of 128 x 128 tiles (Strata's grouped order, 8 row tiles a
// group, measured slower: dn_qkv 14.3 ms against 13.8).
// BF16W: a.words are bf16 weight rows (n, K) row major (GLM's BF16 linears), copied to LDS as stored; a group's sum is
// its two WMMA from zero, folded acc + d on the VALU (IEEE round to nearest) in K order; no scale, bias or sum of x.
// gfx1151's WMMA (and dot2) adds with a truncating ~24-bit aligner, so the long sum over K goes through the VALU, not
// through the instruction's accumulator (lane G4's probe: K 4,096 bf16 flips 0.27 % on one WMMA chain).
template <int K, int AHEAD = 1, bool HC_MIX = false, bool BF16W = false>
__device__ __forceinline__ void wmma_dense_tile(const Affine& a, HcMix mix = {}, const uint16_t* gates = nullptr) {
#if TF_DEVICE_WMMA_GFX11
    using namespace dense_tile;
    using Dec = MlxDecoder<4>;
    using T = WmmaBF16;
    constexpr int kWindows = K / kKW;
    constexpr int kGroups = K / 32;
    constexpr int kLdw = (kKW + 8) / 2;  // a staged row: 72 bf16, so 8 lanes of 16-byte reads hit 32 banks
    constexpr int kWordsRow = BF16W ? K / 2 : K / 8;
    static_assert(K % kKW == 0 && kWindows % AHEAD == 0, "whole windows; a window's register slot is a constant");
    static_assert(!(BF16W && HC_MIX), "the hc mix epilogue reads affine weights");
    __shared__ __attribute__((aligned(16))) uint32_t xs[kBM * kLdw];
    __shared__ __attribute__((aligned(16))) uint32_t ws[kBN * kLdw];
    __shared__ float sx[2][kBM];
    __shared__ float sc[2][kBN];
    __shared__ float bi[2][kBN];

    const int num_n = (a.n + kBN - 1) / kBN;
    const int bm = blockIdx.x / num_n;
    const int bn = blockIdx.x % num_n;
    const int m0 = bm * kBM;
    const int n0 = bn * kBN;
    const int rows_left = a.m - m0;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);

    // a thread's loads: group q of staged row r (codes of column n0 + r, values of x row m0 + r), and the scales
    // (t < 128) or biases of column n0 + t % 128
    const int r = tid >> 1;
    const int q = tid & 1;
    // HC_MIX: local column c is stream c / 32's hidden column bn * 32 + c % 32 (the weights' row of that output)
    const int dims = a.n / 4;
    auto column = [&](int c) { return HC_MIX ? (c / 32) * dims + bn * 32 + c % 32 : n0 + c; };
    const int col = HC_MIX ? column(r) : (n0 + r < a.n ? n0 + r : a.n - 1);
    const int row = r < rows_left ? m0 + r : a.m - 1;
    const uint32_t* wsrc = a.words + static_cast<long long>(col) * kWordsRow + q * (BF16W ? 16 : 4);
    const u32x4* xsrc = reinterpret_cast<const u32x4*>(static_cast<const __bf16*>(a.x) + static_cast<long long>(row) * K) + q * 4;
    const int trow = tid & (kBN - 1);
    const bool thigh = tid >= kBN;
    const int tcol = HC_MIX ? column(trow) : (n0 + trow < a.n ? n0 + trow : a.n - 1);
    const uint16_t* tsrc = static_cast<const uint16_t*>(thigh ? a.bias.p : a.scale.p) + static_cast<long long>(tcol) * kGroups;

    // wave (wm, wn): rows kWR wm.., columns kWC wn.. (kMT x kNT tiles); a lane: column lane % 16, rows 2 i + lane / 16
    constexpr int kWR = 64;
    constexpr int kWC = kBM * kBN / 8 / kWR;
    constexpr int kMT = kWR / 16;
    constexpr int kNT = kWC / 16;
    const int wm = wave % (kBM / kWR);
    const int wn = wave / (kBM / kWR);
    const int rl = lane & 15;
    const int rh = lane >> 4;
    const f32x8 zero = {0, 0, 0, 0, 0, 0, 0, 0};
    f32x8 acc[kMT][kNT];
#pragma unroll
    for (int mt = 0; mt < kMT; ++mt) {
#pragma unroll
        for (int nt = 0; nt < kNT; ++nt) acc[mt][nt] = zero;
    }

    uint4 wreg[AHEAD][BF16W ? 4 : 1];
    u32x4 xreg[AHEAD][4];
    uint16_t treg[AHEAD][2];
    // Every load is unconditional (a window past the last repeats the last): a load behind a branch leaves the compiler
    // no static count, and it waits for vmcnt(0) at the stash.
    auto load = [&]<int R>(std::integral_constant<int, R>, int w) {
        const int wl = w < kWindows ? w : kWindows - 1;
        if constexpr (BF16W) {
            // a group's 32 bf16 of the column: 64 bytes at word 32 wl + 16 q
#pragma unroll
            for (int v = 0; v < 4; ++v) wreg[R][v] = *reinterpret_cast<const uint4*>(wsrc + wl * 32 + v * 4);
        } else {
            wreg[R][0] = *reinterpret_cast<const uint4*>(wsrc + wl * 8);
            treg[R][0] = tsrc[2 * wl];
            treg[R][1] = tsrc[2 * wl + 1];
        }
#pragma unroll
        for (int v = 0; v < 4; ++v) xreg[R][v] = xsrc[wl * 8 + v];
    };
    auto stash = [&]<int R>(std::integral_constant<int, R>) {
        u32x4* wdst = reinterpret_cast<u32x4*>(ws + r * kLdw + q * 16);
        if constexpr (BF16W) {
#pragma unroll
            for (int v = 0; v < 4; ++v) wdst[v] = u32x4{wreg[R][v].x, wreg[R][v].y, wreg[R][v].z, wreg[R][v].w};
            u32x4* xdst = reinterpret_cast<u32x4*>(xs + r * kLdw + q * 16);
#pragma unroll
            for (int v = 0; v < 4; ++v) xdst[v] = xreg[R][v];
            return;
        } else {
        const uint32_t wv[4] = {wreg[R][0].x, wreg[R][0].y, wreg[R][0].z, wreg[R][0].w};
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            uint32_t p[4];
#pragma unroll
            for (int h = 0; h < 4; ++h) p[h] = __builtin_bit_cast(uint32_t, Dec::template pair<T>(wv, v * 8 + h * 2));
            wdst[v] = u32x4{p[0], p[1], p[2], p[3]};
        }
        u32x4* xdst = reinterpret_cast<u32x4*>(xs + r * kLdw + q * 16);
        float run = 0.f;
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            xdst[v] = xreg[R][v];
            const uint32_t u[4] = {xreg[R][v].x, xreg[R][v].y, xreg[R][v].z, xreg[R][v].w};
#pragma unroll
            for (int h = 0; h < 4; ++h) {
                const typename T::pair p = __builtin_bit_cast(typename T::pair, u[h]);
                run += T::lo(p);
                run += T::hi(p);
            }
        }
        sx[q][r] = run;
        float(*tdst)[kBN] = thigh ? bi : sc;
        tdst[0][trow] = __uint_as_float(static_cast<uint32_t>(treg[R][0]) << 16);
        tdst[1][trow] = __uint_as_float(static_cast<uint32_t>(treg[R][1]) << 16);
        }
    };

    [&]<int... R>(std::integer_sequence<int, R...>) {
        (load(std::integral_constant<int, R>{}, R), ...);
    }(std::make_integer_sequence<int, AHEAD>{});
    stash(std::integral_constant<int, 0>{});
    load(std::integral_constant<int, 0>{}, AHEAD);
    lds_barrier();
    // Window w: compute it, stash window w + 1 from its slot, refill the slot with window w + 1 + AHEAD.
    auto window = [&]<int J>(std::integral_constant<int, J>) {
        constexpr int next = (J + 1) % AHEAD;
        {
#pragma unroll
            for (int g = 0; g < 2; ++g) {
                f32x8 d[kMT][kNT];
#pragma unroll
                for (int t = 0; t < 2; ++t) {
                    bf16x16 bf[kNT];
#pragma unroll
                    for (int nt = 0; nt < kNT; ++nt) bf[nt] = frag16(ws + (wn * kWC + nt * 16 + rl) * kLdw + g * 16 + t * 8);
#pragma unroll
                    for (int mt = 0; mt < kMT; ++mt) {
                        const bf16x16 af = frag16(xs + (wm * kWR + mt * 16 + rl) * kLdw + g * 16 + t * 8);
#pragma unroll
                        for (int nt = 0; nt < kNT; ++nt) d[mt][nt] = T::mma(af, bf[nt], t == 0 ? zero : d[mt][nt]);
                    }
                }
                if constexpr (BF16W) {
#pragma unroll
                    for (int mt = 0; mt < kMT; ++mt) {
#pragma unroll
                        for (int nt = 0; nt < kNT; ++nt) {
#pragma unroll
                            for (int i = 0; i < 8; ++i) acc[mt][nt][i] = acc[mt][nt][i] + d[mt][nt][i];
                        }
                    }
                    continue;
                }
#pragma unroll
                for (int mt = 0; mt < kMT; ++mt) {
                    float s_x[8];
#pragma unroll
                    for (int i = 0; i < 8; ++i) s_x[i] = sx[g][wm * kWR + mt * 16 + 2 * i + rh];
#pragma unroll
                    for (int nt = 0; nt < kNT; ++nt) {
                        const float scale = sc[g][wn * kWC + nt * 16 + rl];
                        const float bias = bi[g][wn * kWC + nt * 16 + rl];
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            acc[mt][nt][i] = Dec::fold_scale(acc[mt][nt][i], d[mt][nt][i], scale);
                            acc[mt][nt][i] = Dec::fold_bias(acc[mt][nt][i], s_x[i], bias);
                        }
                    }
                }
            }
        }
        lds_barrier();
        // past the last window the stash rewrites LDS no later window reads
        stash(std::integral_constant<int, next>{});
    };
    for (int w0 = 0; w0 < kWindows; w0 += AHEAD) {
        [&]<int... J>(std::integer_sequence<int, J...>) {
            ((window(std::integral_constant<int, J>{}),
              load(std::integral_constant<int, (J + 1) % AHEAD>{}, w0 + J + 1 + AHEAD), lds_barrier()),
             ...);
        }(std::make_integer_sequence<int, AHEAD>{});
    }

    // fp32 into out, or (out16 set) rounded to BF16 as tf_cast rounds the fp32 sums (nearest even, NaN 0x7fc0)
    auto put = [&](auto store) {
#pragma unroll
        for (int mt = 0; mt < kMT; ++mt) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int orow = m0 + wm * kWR + mt * 16 + 2 * i + rh;
                if (orow >= a.m) continue;
#pragma unroll
                for (int nt = 0; nt < kNT; ++nt) {
                    const int ocol = n0 + wn * kWC + nt * 16 + rl;
                    if (ocol < a.n) store(static_cast<long long>(orow) * a.n + ocol, acc[mt][nt][i]);
                }
            }
        }
    };
    if constexpr (HC_MIX) {
        // wave wn holds stream wn (kWC = 32 columns: the block's d's). The block's x tile [128 rows][4 streams][32 d] goes
        // to LDS in 16-byte loads (rows 0..63 over xs, 64..127 over ws: the stash's, past the last window); each thread
        // turns its x into tf_fn_hc_mix's term in place, its gate read from `gates` (rnd(sigmoid(g)) for every bf16 g:
        // the formula on the critical path cost 0.36 ms of hc up at 2,004 rows); then 16 d a thread sum their 4 terms in
        // stream order.
        auto bits = [](float v) { return __builtin_bit_cast(uint16_t, narrow<T>(v)); };
        auto wide = [](uint16_t h) { return __uint_as_float(static_cast<uint32_t>(h) << 16); };
        auto rnd = [&](float v) { return wide(bits(v)); };
        constexpr int kRow = 4 * 32 + 8;  // a row's 4 x 32 d, padded 16 bytes against bank conflicts
        static_assert(kWR * kRow * 2 <= kBM * kLdw * 4, "64 rows a stash buffer");
        auto slot = [&](int lrow) { return reinterpret_cast<uint16_t*>(lrow < kWR ? xs : ws) + (lrow % kWR) * kRow; };
        constexpr int kLoads = kBM * 4 * 4 / kThreads;  // 16-byte pieces a thread: 4 a (row, stream)
        u32x4 xv[kLoads];
#pragma unroll
        for (int l = 0; l < kLoads; ++l) {
            const int e = tid + l * kThreads, lrow = e / 16, st = (e / 4) % 4, v = e % 4;
            const int orow = m0 + lrow < a.m ? m0 + lrow : a.m - 1;
            xv[l] = *reinterpret_cast<const u32x4*>(mix.x + (static_cast<long long>(orow) * 4 + st) * dims + bn * 32 + v * 8);
        }
        __syncthreads();
#pragma unroll
        for (int l = 0; l < kLoads; ++l) {
            const int e = tid + l * kThreads, lrow = e / 16, st = (e / 4) % 4, v = e % 4;
            *reinterpret_cast<u32x4*>(slot(lrow) + st * 32 + v * 8) = xv[l];
        }
        __syncthreads();
        uint16_t* mine = slot(wm * kWR) + wn * 32;
#pragma unroll
        for (int mt = 0; mt < kMT; ++mt) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                uint16_t* at = mine + (mt * 16 + 2 * i + rh) * kRow;
#pragma unroll
                for (int nt = 0; nt < kNT; ++nt) {
                    const int j = nt * 16 + rl;
                    const float gate = wide(gates[bits(acc[mt][nt][i])]);
                    at[j] = bits(rnd(gate * wide(at[j])));
                }
            }
        }
        __syncthreads();
        static_assert(kBM * 2 == kThreads, "a thread: one row's 16 d");
        const int lrow = tid / 2, d0 = (tid % 2) * 16;
        if (m0 + lrow < a.m) {
            const uint16_t* t = slot(lrow) + d0;
            uint32_t w[4][8];
#pragma unroll
            for (int st = 0; st < 4; ++st) {
                const u32x4 lo = *reinterpret_cast<const u32x4*>(t + st * 32);
                const u32x4 hi = *reinterpret_cast<const u32x4*>(t + st * 32 + 8);
                w[st][0] = lo.x, w[st][1] = lo.y, w[st][2] = lo.z, w[st][3] = lo.w;
                w[st][4] = hi.x, w[st][5] = hi.y, w[st][6] = hi.z, w[st][7] = hi.w;
            }
            uint32_t out[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                uint16_t h[2];
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    float total = 0.f;
#pragma unroll
                    for (int st = 0; st < 4; ++st) total += wide(static_cast<uint16_t>(w[st][k] >> (16 * e)));
                    h[e] = bits(total / 4.f);
                }
                out[k] = static_cast<uint32_t>(h[0]) | (static_cast<uint32_t>(h[1]) << 16);
            }
            u32x4* dst = reinterpret_cast<u32x4*>(mix.mixed + static_cast<long long>(m0 + lrow) * dims + bn * 32 + d0);
            dst[0] = u32x4{out[0], out[1], out[2], out[3]};
            dst[1] = u32x4{out[4], out[5], out[6], out[7]};
        }
    } else if (a.out16 != nullptr) {
        auto* out = reinterpret_cast<typename T::elem*>(a.out16);
        put([&](long long at, float v) { out[at] = narrow<T>(v); });
    } else {
        put([&](long long at, float v) { a.out[at] = v; });
    }
#else
    (void)a;
    (void)mix;
    __builtin_trap();
#endif
}

// A K's windows of loads ahead (gfx1151, 8,192 rows): 1 at K 2560 (dn_qkv 12.9 ms against 13.3 at 2), 6144 (7.90
// against 7.94) and 320 (5 windows); 2 at K 10240 (hc_down 2.21 ms against 2.99 at 1).
template <int K>
struct DenseTune {
    static constexpr int kAhead = K == 10240 ? 2 : 1;
};

}  // namespace rocm
}  // namespace tf
