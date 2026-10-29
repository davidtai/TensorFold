#pragma once

// Qwen3.8-Flash-Next's routed expert products of a prompt on the gfx11 matrix cores, one shape each as constants:
//   MoeGateUp: the stacked (gate | up) bank, K 2560 -> 2 x 640, silu(gate) * up as the epilogue, bf16 out;
//   MoeDown:   the down bank, K 640 -> 2560, fp32 out (moe_combine sums the slots).
//
// The plan's item z is (expert, first, count); a block takes 128 staged weight rows of one item (gate+up: 64 gate
// columns and the same 64 up columns; down: 128 columns) in 64-row passes. Each output's arithmetic is affine_wmma_gemm's
// (matrix_gemm.hpp): two WMMA a 32-code group in the same K order, the group folded as fma(dot, scale) then
// fma(sum x, bias), the sum of x in the same order; gate+up then applies moe_act_kernel's activation and rounding. So
// the bytes equal the matrix tile (followed by moe_act_kernel for gate+up). What changes:
//   - 64-row passes: an item holds about 40 rows at a 2,048-row chunk, and the 128-row tile left half its waves idle;
//   - K in 64-wide windows: a weight row's 32 bytes come from 2 neighbouring lanes, an x row's 128 bytes from 8;
//   - a row's scales (or biases) for 8 (gate+up) or 4 (down) groups in one load;
//   - the sums of x from the x loads (a quad of lanes adds a group's 32 values in order, lane after lane);
//   - AHEAD windows of loads held in registers across the window barriers, which wait for LDS only (lds_barrier);
//   - gate+up: both halves of a pair in one lane: no fp32 (gate | up) buffer, no second launch.
// Inspired by mlx-serve's moe_prefill_cuda.zig (GU_SOURCE: gate rows 0-63, up rows 64-127 of one tile, SwiGLU in the
// epilogue; register prefetch of K stages ahead).

#include <hip/hip_bfloat16.h>

#include <type_traits>
#include <utility>

#include "common/wmma.hpp"
#include "quant/mlx_decoder.hpp"
#include "tiles/dense_inexact.hpp"
#include "tiles/plan.hpp"

namespace tf {
namespace rocm {

namespace moe_tile {
constexpr int kBM = 64;       // rows a pass
constexpr int kBN = 128;      // staged weight rows a block
constexpr int kThreads = 256;
}  // namespace moe_tile

struct MoeGateUp {
    static constexpr int kK = 2560;  // hidden
    static constexpr int kNI = 640;  // expert width: gate rows [0, 640), up rows [640, 1280) of an expert
    static constexpr int kN = 2 * kNI;
    static constexpr int kTG = 8;    // groups a term load covers
    static constexpr bool kAct = true;
    static constexpr int kBlocks = kNI / 64;
    // windows of loads held in registers (gfx1151, 8,192 rows: 29.1 ms a layer at 4, 30.1 at 2, 33.1 at 8)
    static constexpr int kAhead = 4;
    // staged row `line` of block x: gate column 64 x + line, then the up column 640 + 64 x + line - 64
    __device__ static int col(int bx, int line) { return line < 64 ? 64 * bx + line : kNI + 64 * bx + line - 64; }
    // the same for BN staged rows a block: BN / 2 gate columns, then the same up columns
    template <int BN>
    __device__ static int colw(int bx, int line) {
        constexpr int h = BN / 2;
        return line < h ? h * bx + line : kNI + h * bx + line - h;
    }
};

struct MoeDown {
    static constexpr int kK = 640;   // expert width
    static constexpr int kN = 2560;  // hidden
    static constexpr int kTG = 4;    // 20 groups a row: spans of 4
    static constexpr bool kAct = false;
    static constexpr int kBlocks = kN / 128;
    // (gfx1151, 8,192 rows: 15.8 ms a layer at 1, 16.0 at 2)
    static constexpr int kAhead = 1;
    __device__ static int col(int bx, int line) { return 128 * bx + line; }
    template <int BN>
    __device__ static int colw(int bx, int line) { return BN * bx + line; }
};

// KW: K a window, 64 or 128 (2 or 4 groups); a staged row is KW + 8 bf16, so 8 lanes of 16-byte reads hit 32 banks.
// AHEAD: windows of loads a thread holds in registers. Tables: bf16 planes (kScaleBF16).
template <class S, int BITS, int KW = 64, int AHEAD = 1>
__device__ __forceinline__ void wmma_moe_tile(Affine a, float limit) {
#if TF_DEVICE_WMMA_GFX11
    using namespace moe_tile;
    static_assert(BITS == 4, "a 32-code group is one 16-byte chunk");
    static_assert(KW == 128 || KW == 64, "4 or 2 groups a window");
    constexpr int kK = S::kK;
    constexpr int kGW = KW / 32;               // groups a window
    constexpr int kWindows = kK / KW;
    constexpr int kLdw = (KW + 8) / 2;
    constexpr int kWChunks = kBN * kGW / 256;  // 16-byte weight chunks a thread a window
    constexpr int kXPer = KW / 8;              // 16-byte x pieces a row
    constexpr int kXChunks = kBM * kXPer / 256;
    constexpr int kTG = S::kTG;
    constexpr int kTermEvery = kTG / kGW;      // windows a term load covers
    constexpr int kTW = kTG / 2;               // words a thread's term load
    static_assert(kK % KW == 0 && (kK / 32) % kTG == 0 && kTG % kGW == 0, "whole windows and term spans");
    static_assert(kWindows % AHEAD == 0, "a window's register slot is a compile-time constant");
    using Dec = MlxDecoder<BITS>;
    using T = WmmaBF16;
    using TermLoad = std::conditional_t<kTW == 4, uint4, uint2>;
    if (!Dec::take_item(a, blockIdx.z)) return;
    __shared__ __attribute__((aligned(16))) uint32_t xs[kBM * kLdw];
    __shared__ __attribute__((aligned(16))) uint32_t ws[kBN * kLdw];
    __shared__ float sx[kGW][kBM];
    __shared__ float sc[kTG][kBN];
    __shared__ float bi[kTG][kBN];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int bx = blockIdx.x;
    constexpr int kWordsRow = kK * BITS / 32;
    constexpr int kGroups = kK / 32;

    // weights: chunk i of thread t is row (256 i + t) / kGW, group t % kGW of the window
    const int wq = tid % kGW;
    const uint32_t* wsrc[kWChunks];
#pragma unroll
    for (int i = 0; i < kWChunks; ++i) wsrc[i] = a.words + static_cast<long long>(S::col(bx, (256 * i + tid) / kGW)) * kWordsRow + wq * BITS;
    // terms of row t % 128, kTG groups a load: the scales (t < 128) or the biases
    const int trow = tid & (kBN - 1);
    const bool thigh = tid >= kBN;
    const long long tcol = static_cast<long long>(S::col(bx, trow)) * kGroups;
    const TermLoad* tsrc = reinterpret_cast<const TermLoad*>(static_cast<const uint16_t*>(thigh ? a.bias.p : a.scale.p) + tcol);
    // x: chunk i of thread t is row (256 i + t) / kXPer, 16-byte piece t % kXPer (group piece / 4, quad lane t % 4)
    const int xc = tid % kXPer;

    // wave (wm, wn): rows 32 wm.., staged rows 16 wn.. and 64 + 16 wn..; a lane: column lane % 16, rows 2 i + lane / 16
    const int wm = wave & 1;
    const int wn = wave >> 1;
    const int rl = lane & 15;
    const int rh = lane >> 4;
    const int lines[2] = {wn * 16 + rl, 64 + wn * 16 + rl};
    const f32x8 zero = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int h0 = 0; h0 < a.m; h0 += kBM) {
        const int rows_left = a.m - h0 < kBM ? a.m - h0 : kBM;
        const u32x4* xsrc[kXChunks];
#pragma unroll
        for (int i = 0; i < kXChunks; ++i) {
            const int r = (256 * i + tid) / kXPer;
            const int rs = h0 + (r < rows_left ? r : rows_left - 1);
            xsrc[i] = reinterpret_cast<const u32x4*>(static_cast<const __bf16*>(a.x) + x_row(a, rs) * kK) + xc;
        }
        const bool live0 = wm * 32 < rows_left;
        const bool live1 = wm * 32 + 16 < rows_left;
        u32x4 wreg[AHEAD][kWChunks];
        u32x4 xreg[AHEAD][kXChunks];
        TermLoad treg[AHEAD];
        f32x8 acc[2][2];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
            for (int j = 0; j < 2; ++j) acc[mt][j] = zero;
        }
        // Every load is issued unconditionally (a window past the last repeats it, and the terms are reloaded each
        // window): a load behind a branch leaves the compiler no static count, and it waits for vmcnt(0) at each stash.
        // A window's terms load with it when it starts a span. Whether it does is known at compile time where AHEAD is a
        // multiple of the span's windows (window w + 1 + AHEAD of step J starts a span when J + 1 does); else every
        // window reloads them, which costs an L2 read but keeps the loads out of branches.
        constexpr bool kTermsStatic = AHEAD % kTermEvery == 0;
        auto load = [&]<int R, bool TERMS = true>(std::integral_constant<int, R>, int w, std::bool_constant<TERMS> = {}) {
            const int wl = w < kWindows ? w : kWindows - 1;
#pragma unroll
            for (int i = 0; i < kWChunks; ++i) wreg[R][i] = *reinterpret_cast<const u32x4*>(wsrc[i] + wl * kGW * BITS);
#pragma unroll
            for (int i = 0; i < kXChunks; ++i) xreg[R][i] = xsrc[i][wl * kXPer];
            if constexpr (TERMS) treg[R] = tsrc[wl / kTermEvery];
        };
        auto stash = [&]<int R>(std::integral_constant<int, R>, int w) {
#pragma unroll
            for (int i = 0; i < kWChunks; ++i) {
                const int row = (256 * i + tid) / kGW;
                const uint32_t wv[BITS] = {wreg[R][i].x, wreg[R][i].y, wreg[R][i].z, wreg[R][i].w};
                u32x4* dst = reinterpret_cast<u32x4*>(ws + row * kLdw + wq * 16);
#pragma unroll
                for (int v = 0; v < 4; ++v) {
                    uint32_t q[4];
#pragma unroll
                    for (int h = 0; h < 4; ++h) q[h] = __builtin_bit_cast(uint32_t, Dec::template pair<T>(wv, v * 8 + h * 2));
                    dst[v] = u32x4{q[0], q[1], q[2], q[3]};
                }
            }
#pragma unroll
            for (int i = 0; i < kXChunks; ++i) {
                const int row = (256 * i + tid) / kXPer;
                *reinterpret_cast<u32x4*>(xs + row * kLdw + xc * 4) = xreg[R][i];
                // the group's sum in order: quad lane 0 adds its 8 values, hands the sum to lane 1, ... lane 3 stores it
                const uint32_t q[4] = {xreg[R][i].x, xreg[R][i].y, xreg[R][i].z, xreg[R][i].w};
                float run = 0.f;
#pragma unroll
                for (int hop = 0; hop < 4; ++hop) {
                    const float in = __shfl_up(run, 1, 4);
                    if ((tid & 3) == hop) {
                        run = hop == 0 ? 0.f : in;
#pragma unroll
                        for (int h = 0; h < 4; ++h) {
                            const typename T::pair p = __builtin_bit_cast(typename T::pair, q[h]);
                            run += T::lo(p);
                            run += T::hi(p);
                        }
                    }
                }
                if ((tid & 3) == 3) sx[xc >> 2][row] = run;
            }
            if (w % kTermEvery == 0) {
                const uint32_t* tv = reinterpret_cast<const uint32_t*>(&treg[R]);
                float(*dst)[kBN] = thigh ? bi : sc;
#pragma unroll
                for (int j = 0; j < kTG; ++j) dst[j][trow] = __uint_as_float(((tv[j >> 1] >> (16 * (j & 1))) & 0xffffu) << 16);
            }
        };

        lds_barrier();
        [&]<int... R>(std::integer_sequence<int, R...>) {
            (load(std::integral_constant<int, R>{}, R, std::bool_constant<!kTermsStatic || R % kTermEvery == 0>{}), ...);
        }(std::make_integer_sequence<int, AHEAD>{});
        stash(std::integral_constant<int, 0>{}, 0);
        load(std::integral_constant<int, 0>{}, AHEAD);
        lds_barrier();
        // Window w: compute it, stash window w + 1 from its slot, refill the slot with window w + 1 + AHEAD.
        auto window = [&]<int J>(std::integral_constant<int, J>, int w) {
            constexpr int next = (J + 1) % AHEAD;
            if (live0) {
#pragma unroll
                for (int gi = 0; gi < kGW; ++gi) {
                    f32x8 d[2][2];
#pragma unroll
                    for (int t = 0; t < 2; ++t) {
                        bf16x16 bf[2];
#pragma unroll
                        for (int j = 0; j < 2; ++j) bf[j] = frag16(ws + lines[j] * kLdw + gi * 16 + t * 8);
#pragma unroll
                        for (int mt = 0; mt < 2; ++mt) {
                            if (mt == 1 && !live1) continue;
                            const bf16x16 af = frag16(xs + (wm * 32 + mt * 16 + rl) * kLdw + gi * 16 + t * 8);
#pragma unroll
                            for (int j = 0; j < 2; ++j) d[mt][j] = T::mma(af, bf[j], t == 0 ? zero : d[mt][j]);
                        }
                    }
                    const int gt = (kGW * w + gi) % kTG;
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt) {
                        if (mt == 1 && !live1) continue;
                        float s_x[8];
#pragma unroll
                        for (int i = 0; i < 8; ++i) s_x[i] = sx[gi][wm * 32 + mt * 16 + 2 * i + rh];
#pragma unroll
                        for (int j = 0; j < 2; ++j) {
                            const float scale = sc[gt][lines[j]];
                            const float bias = bi[gt][lines[j]];
#pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                acc[mt][j][i] = Dec::fold_scale(acc[mt][j][i], d[mt][j][i], scale);
                                acc[mt][j][i] = Dec::fold_bias(acc[mt][j][i], s_x[i], bias);
                            }
                        }
                    }
                }
            }
            lds_barrier();
            // past the last window the stash rewrites LDS no later window reads
            stash(std::integral_constant<int, next>{}, w + 1);
            load(std::integral_constant<int, next>{}, w + 1 + AHEAD, std::bool_constant<!kTermsStatic || (J + 1) % kTermEvery == 0>{});
            lds_barrier();
        };
        for (int w0 = 0; w0 < kWindows; w0 += AHEAD) {
            [&]<int... J>(std::integer_sequence<int, J...>) {
                (window(std::integral_constant<int, J>{}, w0 + J), ...);
            }(std::make_integer_sequence<int, AHEAD>{});
        }

#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row = wm * 32 + mt * 16 + 2 * i + rh;
                if (row >= rows_left) continue;
                const long long orow = out_row(a, h0 + row);
                if constexpr (S::kAct) {
                    // moe_act_kernel on the pair's fp32 sums: clamp when limit > 0, g / (1 + exp(-g)) * u, hip_bfloat16
                    float g = acc[mt][0][i];
                    float u = acc[mt][1][i];
                    if (limit > 0.f) {
                        g = fminf(g, limit);
                        u = fminf(fmaxf(u, -limit), limit);
                    }
                    const hip_bfloat16 v(g / (1.f + expf(-g)) * u);
                    reinterpret_cast<uint16_t*>(a.out16)[orow * S::kNI + 64 * bx + wn * 16 + rl] = v.data;
                } else {
#pragma unroll
                    for (int j = 0; j < 2; ++j) a.out[orow * S::kN + S::col(bx, lines[j])] = acc[mt][j][i];
                }
            }
        }
    }
#else
    (void)a;
    (void)limit;
    __builtin_trap();
#endif
}

// numerics=inexact: the same products with the codes dequantized to bf16 in the stash (w = bf16(fma(scale, q, bias)),
// dense_inexact.hpp's rounding) and one WMMA accumulation over K: no per-group d, fold or sums of x, so a wave can hold
// more tiles. BM rows a pass, the same 128 staged weight rows (S::col), waves of WR rows. gfx1151 at 8,192 rows
// (w2/moe_bench.hip, -DTF_BENCH_INEXACT): gate+up at 128 rows, 64 x 32 waves, 22.6 ms against 25.0 for the exact tile;
// down at 64 rows, 32 x 32 waves, 12.2 ms against 13.8. Relative RMS against float64 2.4e-3 and 1.7e-3. Served at
// PC-1's one chunk the 64-row down dequantized each item twice; MG-2 runs it at 128 rows (mlx_tiles.hpp). Down's
// epilogue stages each wave's sums in LDS and stores 16-byte row segments (DL-4d): served at one chunk it was 23 % of a
// block's life, store issue with 64 dword stores a lane (w2/diag-phase).
// BN staged weight rows a block (128; MS-2: 256) and KW K a window (64; 32 keeps BN 256's LDS at 3 blocks a WGP). Each
// output's sum is the same WMMA chain over K in order for any BN and KW: only the block that computes it changes.
template <class S, int AHEAD, int BM, int WR, int BN = moe_tile::kBN, int KW = 64>
__device__ __forceinline__ void wmma_moe_inexact_tile(Affine a, float limit) {
#if TF_DEVICE_WMMA_GFX11
    constexpr int kBN = BN;
    constexpr int kK = S::kK, kGW = KW / 32, kWindows = kK / KW, kLdw = (KW + 8) / 2;
    constexpr int kWChunks = kBN * kGW / 256, kXPer = KW / 8, kXChunks = BM * kXPer / 256;
    constexpr int kWordsRow = kK / 8, kGroups = kK / 32;
    constexpr int kWM = BM / WR, kWC = BM * kBN / 8 / WR, kMT = WR / 16, kNT = kWC / 16;
    static_assert(kWindows % AHEAD == 0 && kXChunks >= 1 && (!S::kAct || kNT % 2 == 0), "shape");
    using Dec = MlxDecoder<4>;
    using T = WmmaBF16;
    if (!Dec::take_item(a, blockIdx.z)) return;
    __shared__ __attribute__((aligned(16))) uint32_t xs[BM * kLdw];
    __shared__ __attribute__((aligned(16))) uint32_t ws[kBN * kLdw];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int bx = blockIdx.x;
    // weights: chunk i of thread t is staged row (256 i + t) / 2, group t % 2 of the window, with its scale and bias
    const int wq = tid % kGW;
    const uint32_t* wsrc[kWChunks];
    const uint16_t* ssrc[kWChunks];
    const uint16_t* bsrc[kWChunks];
#pragma unroll
    for (int i = 0; i < kWChunks; ++i) {
        const long long c = S::template colw<kBN>(bx, (256 * i + tid) / kGW);
        wsrc[i] = a.words + c * kWordsRow + wq * 4;
        ssrc[i] = static_cast<const uint16_t*>(a.scale.p) + c * kGroups + wq;
        bsrc[i] = static_cast<const uint16_t*>(a.bias.p) + c * kGroups + wq;
    }
    const int xc = tid % kXPer;
    // wave (wm, wn): rows WR wm.., kNT staged-row tiles (gate+up: the first half gate, the second the same up columns)
    const int wm = wave % kWM;
    const int wn = wave / kWM;
    const int rl = lane & 15;
    const int rh = lane >> 4;
    int lines[kNT];
#pragma unroll
    for (int j = 0; j < kNT; ++j) {
        if constexpr (S::kAct) lines[j] = (j < kNT / 2 ? 0 : kBN / 2) + wn * (kWC / 2) + (j % (kNT / 2)) * 16 + rl;
        else lines[j] = wn * kWC + j * 16 + rl;
    }
    const f32x8 zero = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int h0 = 0; h0 < a.m; h0 += BM) {
        const int rows_left = a.m - h0 < BM ? a.m - h0 : BM;
        const u32x4* xsrc[kXChunks];
#pragma unroll
        for (int i = 0; i < kXChunks; ++i) {
            const int r = (256 * i + tid) / kXPer;
            const int rs = h0 + (r < rows_left ? r : rows_left - 1);
            xsrc[i] = reinterpret_cast<const u32x4*>(static_cast<const __bf16*>(a.x) + x_row(a, rs) * kK) + xc;
        }
        u32x4 wreg[AHEAD][kWChunks];
        u32x4 xreg[AHEAD][kXChunks];
        uint16_t sreg[AHEAD][kWChunks];
        uint16_t breg[AHEAD][kWChunks];
        f32x8 acc[kMT][kNT];
#pragma unroll
        for (int mt = 0; mt < kMT; ++mt) {
#pragma unroll
            for (int j = 0; j < kNT; ++j) acc[mt][j] = zero;
        }
        // unconditional loads (a window past the last repeats it): no vmcnt(0) at the stash
        auto load = [&]<int R>(std::integral_constant<int, R>, int w) {
            const int wl = w < kWindows ? w : kWindows - 1;
#pragma unroll
            for (int i = 0; i < kWChunks; ++i) {
                wreg[R][i] = *reinterpret_cast<const u32x4*>(wsrc[i] + wl * kGW * 4);
                sreg[R][i] = ssrc[i][wl * kGW];
                breg[R][i] = bsrc[i][wl * kGW];
            }
#pragma unroll
            for (int i = 0; i < kXChunks; ++i) xreg[R][i] = xsrc[i][wl * kXPer];
        };
        auto stash = [&]<int R>(std::integral_constant<int, R>) {
#pragma unroll
            for (int i = 0; i < kWChunks; ++i) {
                const int row = (256 * i + tid) / kGW;
                const float sv = __uint_as_float(static_cast<uint32_t>(sreg[R][i]) << 16);
                const float bv = __uint_as_float(static_cast<uint32_t>(breg[R][i]) << 16);
                const uint32_t wv[4] = {wreg[R][i].x, wreg[R][i].y, wreg[R][i].z, wreg[R][i].w};
                u32x4* dst = reinterpret_cast<u32x4*>(ws + row * kLdw + wq * 16);
#pragma unroll
                for (int v = 0; v < 4; ++v) dst[v] = dense_inexact::dequant8(wv[v], sv, bv);
            }
#pragma unroll
            for (int i = 0; i < kXChunks; ++i) {
                const int row = (256 * i + tid) / kXPer;
                *reinterpret_cast<u32x4*>(xs + row * kLdw + xc * 4) = xreg[R][i];
            }
        };
        lds_barrier();
        [&]<int... R>(std::integer_sequence<int, R...>) {
            (load(std::integral_constant<int, R>{}, R), ...);
        }(std::make_integer_sequence<int, AHEAD>{});
        stash(std::integral_constant<int, 0>{});
        load(std::integral_constant<int, 0>{}, AHEAD);
        lds_barrier();
        auto window = [&]<int J>(std::integral_constant<int, J>, int w) {
            constexpr int next = (J + 1) % AHEAD;
#pragma unroll
            for (int k16 = 0; k16 < KW / 16; ++k16) {
                bf16x16 bf[kNT];
#pragma unroll
                for (int j = 0; j < kNT; ++j) bf[j] = frag16(ws + lines[j] * kLdw + k16 * 8);
#pragma unroll
                for (int mt = 0; mt < kMT; ++mt) {
                    const bf16x16 af = frag16(xs + (wm * WR + mt * 16 + rl) * kLdw + k16 * 8);
#pragma unroll
                    for (int j = 0; j < kNT; ++j) acc[mt][j] = T::mma(af, bf[j], acc[mt][j]);
                }
            }
            lds_barrier();
            // past the last window the stash rewrites LDS no later window reads
            stash(std::integral_constant<int, next>{});
            load(std::integral_constant<int, next>{}, w + 1 + AHEAD);
            lds_barrier();
        };
        for (int w0 = 0; w0 < kWindows; w0 += AHEAD) {
            [&]<int... J>(std::integer_sequence<int, J...>) {
                (window(std::integral_constant<int, J>{}, w0 + J), ...);
            }(std::make_integer_sequence<int, AHEAD>{});
        }
        if constexpr (S::kAct) {
#pragma unroll
            for (int mt = 0; mt < kMT; ++mt) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int row = wm * WR + mt * 16 + 2 * i + rh;
                    if (row >= rows_left) continue;
                    const long long orow = out_row(a, h0 + row);
                    // moe_act_kernel on the pair's fp32 sums: clamp when limit > 0, g / (1 + exp(-g)) * u, hip_bfloat16
#pragma unroll
                    for (int j = 0; j < kNT / 2; ++j) {
                        float g = acc[mt][j][i];
                        float u = acc[mt][j + kNT / 2][i];
                        if (limit > 0.f) {
                            g = fminf(g, limit);
                            u = fminf(fmaxf(u, -limit), limit);
                        }
                        const hip_bfloat16 v(g / (1.f + expf(-g)) * u);
                        reinterpret_cast<uint16_t*>(a.out16)[orow * S::kNI + kBN / 2 * bx + lines[j]] = v.data;
                    }
                }
            }
        } else {
            // down's epilogue through LDS (DL-4d): the loop's last window ended with lds_barrier after its fragment
            // reads and its stash, so no wave reads xs again in this pass, and the next pass starts with lds_barrier
            // before its stash. A wave stages each 16-row tile of its kNT x 16 fp32 sums in its own xs region and
            // stores whole row segments, 16 bytes a lane. Rows past the item are not stored. With BN 256 a wave's row is
            // staged in kParts column parts, in the larger of xs and ws, so the staging fits.
            constexpr int kRegion = (BM >= kBN ? BM : kBN) * kLdw * 4;
            constexpr int kParts = 8 * 16 * (kNT * 16 + 4) * 4 <= kRegion ? 1 : 2;
            constexpr int kPT = kNT / kParts;    // tiles a part
            constexpr int kCols = kPT * 16;      // a wave's sums a row, a part
            constexpr int kChunks = kCols / 4;   // 16-byte chunks a row
            constexpr int kStride = kCols + 4;   // a staged row, in floats
            static_assert(kNT % kParts == 0 && 16 * kChunks % 32 == 0 && 8 * 16 * kStride * 4 <= kRegion, "epilogue staging");
            float* stage = reinterpret_cast<float*>(BM >= kBN ? xs : ws) + wave * 16 * kStride;
            const int col0 = S::template colw<kBN>(bx, lines[0] - rl);  // the wave's first hidden column
#pragma unroll
            for (int mt = 0; mt < kMT; ++mt) {
#pragma unroll
                for (int p = 0; p < kParts; ++p) {
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        float* srow = stage + (2 * i + rh) * kStride + rl;
#pragma unroll
                        for (int j = 0; j < kPT; ++j) srow[16 * j] = acc[mt][p * kPT + j][i];
                    }
                    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
                    __builtin_amdgcn_wave_barrier();
                    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
#pragma unroll
                    for (int c = 0; c < 16 * kChunks / 32; ++c) {
                        const int idx = 32 * c + lane;
                        const int r = idx / kChunks, q = idx % kChunks;
                        const u32x4 v = *reinterpret_cast<const u32x4*>(stage + r * kStride + 4 * q);
                        const int row = wm * WR + mt * 16 + r;
                        if (row < rows_left)
                            *reinterpret_cast<u32x4*>(a.out + out_row(a, h0 + row) * S::kN + col0 + p * kCols + 4 * q) = v;
                    }
                    // the next tile's writes come after these reads in the wave's LDS order
                    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
                    __builtin_amdgcn_wave_barrier();
                    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
                }
            }
        }
    }
#else
    (void)a;
    (void)limit;
    __builtin_trap();
#endif
}

}  // namespace rocm
}  // namespace tf
