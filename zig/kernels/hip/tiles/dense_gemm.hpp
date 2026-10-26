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

namespace dense_tile {
constexpr int kBM = 128;
constexpr int kBN = 128;
constexpr int kKW = 64;  // K a window: 2 groups
constexpr int kThreads = 256;
}  // namespace dense_tile

// K: the product's K (a multiple of 64). AHEAD: windows of loads a thread holds (K / 64 a multiple of it). bf16 x and
// tables, plain products only. Blocks in row-major order of 128 x 128 tiles (Strata's grouped order, 8 row tiles a
// group, measured slower: dn_qkv 14.3 ms against 13.8).
template <int K, int AHEAD = 1>
__device__ __forceinline__ void wmma_dense_tile(const Affine& a) {
#if TF_DEVICE_WMMA_GFX11
    using namespace dense_tile;
    using Dec = MlxDecoder<4>;
    using T = WmmaBF16;
    constexpr int kWindows = K / kKW;
    constexpr int kGroups = K / 32;
    constexpr int kLdw = (kKW + 8) / 2;  // a staged row: 72 bf16, so 8 lanes of 16-byte reads hit 32 banks
    constexpr int kWordsRow = K / 8;
    static_assert(K % kKW == 0 && kWindows % AHEAD == 0, "whole windows; a window's register slot is a constant");
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
    const int col = n0 + r < a.n ? n0 + r : a.n - 1;
    const int row = r < rows_left ? m0 + r : a.m - 1;
    const uint32_t* wsrc = a.words + static_cast<long long>(col) * kWordsRow + q * 4;
    const u32x4* xsrc = reinterpret_cast<const u32x4*>(static_cast<const __bf16*>(a.x) + static_cast<long long>(row) * K) + q * 4;
    const int trow = tid & (kBN - 1);
    const bool thigh = tid >= kBN;
    const int tcol = n0 + trow < a.n ? n0 + trow : a.n - 1;
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

    uint4 wreg[AHEAD];
    u32x4 xreg[AHEAD][4];
    uint16_t treg[AHEAD][2];
    // Every load is unconditional (a window past the last repeats the last): a load behind a branch leaves the compiler
    // no static count, and it waits for vmcnt(0) at the stash.
    auto load = [&]<int R>(std::integral_constant<int, R>, int w) {
        const int wl = w < kWindows ? w : kWindows - 1;
        wreg[R] = *reinterpret_cast<const uint4*>(wsrc + wl * 8);
#pragma unroll
        for (int v = 0; v < 4; ++v) xreg[R][v] = xsrc[wl * 8 + v];
        treg[R][0] = tsrc[2 * wl];
        treg[R][1] = tsrc[2 * wl + 1];
    };
    auto stash = [&]<int R>(std::integral_constant<int, R>) {
        const uint32_t wv[4] = {wreg[R].x, wreg[R].y, wreg[R].z, wreg[R].w};
        u32x4* wdst = reinterpret_cast<u32x4*>(ws + r * kLdw + q * 16);
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
    if (a.out16 != nullptr) {
        auto* out = reinterpret_cast<typename T::elem*>(a.out16);
        put([&](long long at, float v) { out[at] = narrow<T>(v); });
    } else {
        put([&](long long at, float v) { a.out[at] = v; });
    }
#else
    (void)a;
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
