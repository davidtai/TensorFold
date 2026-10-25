#pragma once

// Qwen3.8-Flash-Next's routed expert gate and up for a prompt on the gfx11 matrix cores, silu(gate) * up as the epilogue.
//
// The plan's item z is (expert, first, count) over the stacked (gate | up) bank; a block takes 64 gate columns and the
// same 64 up columns of one item, in 64-row passes. Each output's arithmetic is affine_wmma_gemm's (matrix_gemm.hpp):
// the same two WMMA a 32-code group in the same order, the group folded as fma(dot, scale) then fma(sum x, bias), the
// sum of x in the same order, then moe_act_kernel's activation and rounding: the bytes equal the matrix tile followed by
// moe_act_kernel. What changes: 64-row passes (an item holds about 40 rows at a 2,048-row chunk, so the 128-row tile left
// half its waves idle), both halves of a pair in one lane (no fp32 (gate | up) buffer, no second launch), 64 VGPRs of
// accumulators (the 128-row tile keeps 128). Inspired by mlx-serve's moe_prefill_cuda.zig GU_SOURCE (gate rows 0-63,
// up rows 64-127 of one tile, SwiGLU in the epilogue).

#include <hip/hip_bfloat16.h>

#include "common/wmma.hpp"
#include "quant/mlx_decoder.hpp"
#include "tiles/plan.hpp"

namespace tf {
namespace rocm {

namespace moe_gu {
constexpr int kK = 2560;          // hidden
constexpr int kNI = 640;          // expert width: gate rows [0, 640), up rows [640, 1280) of an expert
constexpr int kBM = 64;           // rows a pass
constexpr int kBH = 64;           // columns of each half a block
constexpr int kBN = 2 * kBH;      // staged weight rows: 64 gate, 64 up
constexpr int kStages = kK / 32;  // one 32-code group a stage
constexpr int kLdw = 20;          // words a staged row: 32 bf16 and 8 of padding, as kGemmLd
constexpr int kThreads = 256;
constexpr int kBlocks = kNI / kBH;
}  // namespace moe_gu

// AHEAD: stages of weights and x a thread holds in registers ahead of their use.
template <int BITS, int AHEAD = 4>
__device__ __forceinline__ void wmma_moe_gu_tile(Affine a, float limit) {
#if TF_DEVICE_WMMA_GFX11
    using namespace moe_gu;
    constexpr int kAhead = AHEAD;
    static_assert(kStages % kAhead == 0, "the ring's slot of a stage is a compile-time constant");
    using Dec = MlxDecoder<BITS>;
    using T = WmmaBF16;
    if (!Dec::take_item(a, blockIdx.z)) return;
    __shared__ __attribute__((aligned(16))) uint32_t xs[2][kBM * kLdw];
    __shared__ __attribute__((aligned(16))) uint32_t ws[2][kBN * kLdw];
    __shared__ float sx[2][kBM];
    // The last three groups' scales and biases: the bias lags the scale a stage, as in the 128-row tile.
    __shared__ float sc[3][kBN];
    __shared__ float bi[3][kBN];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int n0 = blockIdx.x * kBH;
    constexpr int kWordsRow = kK * BITS / 32;
    constexpr int kGroups = kK / 32;

    // Staging: thread t decodes half (side) of weight row `line` and moves 16 bytes of x row t / 4.
    const int line = tid & (kBN - 1);
    const int side = __builtin_amdgcn_readfirstlane(tid >> 7);
    const int col = line < kBH ? n0 + line : kNI + n0 + (line - kBH);
    const uint32_t* wsrc = a.words + static_cast<long long>(col) * kWordsRow;
    const long long tbase = static_cast<long long>(col) * kGroups;
    const int xrow = tid >> 2;
    const int xq = tid & 3;
    const bool summer = side == 1 && line < kBM;

    // Wave (wm, wn): rows 32 wm.., gate columns 16 wn.. and the same up columns; a lane: column lane % 16, rows 2 i + lane / 16.
    const int wm = wave & 1;
    const int wn = wave >> 1;
    const int rl = lane & 15;
    const int rh = lane >> 4;
    const int lines[2] = {wn * 16 + rl, kBH + wn * 16 + rl};
    const f32x8 zero = {0, 0, 0, 0, 0, 0, 0, 0};
    uint16_t* out = reinterpret_cast<uint16_t*>(a.out16);

    for (int h0 = 0; h0 < a.m; h0 += kBM) {
        const int rows_left = a.m - h0 < kBM ? a.m - h0 : kBM;
        const int r_s = h0 + (xrow < rows_left ? xrow : rows_left - 1);
        const u32x4* xsrc =
            reinterpret_cast<const u32x4*>(static_cast<const __bf16*>(a.x) + x_row(a, r_s) * kK) + xq;
        const bool live0 = wm * 32 < rows_left;
        const bool live1 = wm * 32 + 16 < rows_left;
        uint32_t wreg[kAhead][BITS];
        u32x4 xreg[kAhead];
        typename Dec::Term term[kAhead];
        float run = 0.f;
        f32x8 acc[2][2];
        f32x8 d[2][2];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                acc[mt][j] = zero;
                d[mt][j] = zero;
            }
        }
        auto fetch = [&](int st, int slot) {
            Dec::template load<true>(wsrc + st * BITS, wreg[slot]);
            xreg[slot] = xsrc[st * 4];
            term[slot] = Dec::term(a, tbase + st);
        };
        auto stash = [&](int st, int slot, int buf) {
            u32x4* dst = reinterpret_cast<u32x4*>(ws[buf] + line * kLdw + side * 8);
#pragma unroll
            for (int v = 0; v < 2; ++v) {
                uint32_t q[4];
#pragma unroll
                for (int h = 0; h < 4; ++h) {
                    q[h] = __builtin_bit_cast(uint32_t, Dec::template pair<T>(wreg[slot], side * 16 + v * 8 + h * 2));
                }
                dst[v] = u32x4{q[0], q[1], q[2], q[3]};
            }
            *reinterpret_cast<u32x4*>(xs[buf] + xrow * kLdw + xq * 4) = xreg[slot];
            if (side == 0) {
                sc[st % 3][line] = Dec::scale_value(a, term[slot]);
                bi[st % 3][line] = Dec::bias_value(a, term[slot]);
            }
        };
        // Row `line`'s 32 values of x in order, as the 128-row tile's sumx.
        auto sumx = [&](int st, int buf) {
            run = 0.f;
            const u32x4* src = reinterpret_cast<const u32x4*>(xs[buf] + line * kLdw);
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                const u32x4 u = src[v];
                const uint32_t q[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
                for (int h = 0; h < 4; ++h) {
                    const typename T::pair p = __builtin_bit_cast(typename T::pair, q[h]);
                    run += T::lo(p);
                    run += T::hi(p);
                }
            }
            sx[st & 1][line] = run;
        };
        auto apply_bias = [&](int g) {
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                if (mt == 1 && !live1) continue;
                float s_x[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) s_x[i] = sx[g & 1][wm * 32 + mt * 16 + 2 * i + rh];
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const float bias = bi[g % 3][lines[j]];
#pragma unroll
                    for (int i = 0; i < 8; ++i) acc[mt][j][i] = Dec::fold_bias(acc[mt][j][i], s_x[i], bias);
                }
            }
        };

        __syncthreads();
#pragma unroll
        for (int j = 0; j < kAhead; ++j) fetch(j, j);
        stash(0, 0, 0);
        fetch(kAhead, 0);
        __syncthreads();
        // Step s computes stage s, stashes stage s + 1 from its slot and refills the slot with stage s + 1 + kAhead.
        auto step = [&]<int J>(std::integral_constant<int, J>, int s) {
            constexpr int next = (J + 1) % kAhead;
            const int cur = s & 1;
            if (live0 && s > 0) apply_bias(s - 1);
            if (live0) {
#pragma unroll
                for (int t = 0; t < 2; ++t) {
                    bf16x16 bf[2];
#pragma unroll
                    for (int j = 0; j < 2; ++j) bf[j] = frag16(ws[cur] + lines[j] * kLdw + t * 8);
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt) {
                        if (mt == 1 && !live1) continue;
                        const bf16x16 af = frag16(xs[cur] + (wm * 32 + mt * 16 + rl) * kLdw + t * 8);
#pragma unroll
                        for (int j = 0; j < 2; ++j) d[mt][j] = T::mma(af, bf[j], t == 0 ? zero : d[mt][j]);
                    }
                }
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    if (mt == 1 && !live1) continue;
#pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        const float scale = sc[s % 3][lines[j]];
#pragma unroll
                        for (int i = 0; i < 8; ++i) acc[mt][j][i] = Dec::fold_scale(acc[mt][j][i], d[mt][j][i], scale);
                    }
                }
            }
            if (summer) sumx(s, cur);
            if (s + 1 < kStages) {
                stash(s + 1, next, cur ^ 1);
                if (s + 1 + kAhead < kStages) fetch(s + 1 + kAhead, next);
            }
            __syncthreads();
        };
        for (int s0 = 0; s0 < kStages; s0 += kAhead) {
            [&]<int... J>(std::integer_sequence<int, J...>) {
                (step(std::integral_constant<int, J>{}, s0 + J), ...);
            }(std::make_integer_sequence<int, kAhead>{});
        }
        if (live0) apply_bias(kStages - 1);

        // moe_act_kernel on the pair's fp32 sums: clamp when limit > 0, g / (1 + exp(-g)) * u, hip_bfloat16's rounding.
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row = wm * 32 + mt * 16 + 2 * i + rh;
                if (row >= rows_left) continue;
                float g = acc[mt][0][i];
                float u = acc[mt][1][i];
                if (limit > 0.f) {
                    g = fminf(g, limit);
                    u = fminf(fmaxf(u, -limit), limit);
                }
                const hip_bfloat16 v(g / (1.f + expf(-g)) * u);
                out[out_row(a, h0 + row) * kNI + n0 + wn * 16 + rl] = v.data;
            }
        }
    }
#else
    (void)a;
    (void)limit;
    __builtin_trap();
#endif
}


// The same products and bytes with K in 128-wide windows (4 groups): a weight row's 64 bytes come from 4 neighbouring
// lanes and an x row's 256 bytes from 16 (one load instruction touches 8 weight rows, not 32), the 8 groups' scales and
// biases of a row come in one 16-byte load each, and the sums of x come from the x loads (a quad of lanes adds a
// group's 32 values in order, lane after lane). Two barriers a window, the next window's loads in flight across it.
// BF16 scale and bias planes (kScaleBF16), or the pairs of both (kScaleBF16Pair): the same values either way.
// KW: K a window, 128 or 64 (4 or 2 groups); a staged row is KW + 8 bf16, so 8 lanes of 16-byte reads hit 32 banks.
template <int BITS, int KW = 128>
__device__ __forceinline__ void wmma_moe_gw_tile(Affine a, float limit) {
#if TF_DEVICE_WMMA_GFX11
    using namespace moe_gu;
    static_assert(BITS == 4, "a 32-code group is one 16-byte chunk");
    static_assert(KW == 128 || KW == 64, "4 or 2 groups a window");
    constexpr int kGW = KW / 32;            // groups a window
    constexpr int kWindows = kK / KW;
    constexpr int kLdw = (KW + 8) / 2;
    constexpr int kWChunks = kBN * kGW / 256;  // 16-byte weight chunks a thread a window
    constexpr int kXPer = KW / 8;              // 16-byte x pieces a row
    constexpr int kXChunks = kBM * kXPer / 256;
    constexpr int kTermEvery = 8 / kGW;        // windows a term load covers (8 groups)
    using Dec = MlxDecoder<BITS>;
    using T = WmmaBF16;
    if (!Dec::take_item(a, blockIdx.z)) return;
    __shared__ __attribute__((aligned(16))) uint32_t xs[kBM * kLdw];
    __shared__ __attribute__((aligned(16))) uint32_t ws[kBN * kLdw];
    __shared__ float sx[kGW][kBM];
    __shared__ float sc[8][kBN];
    __shared__ float bi[8][kBN];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int n0 = blockIdx.x * kBH;
    constexpr int kWordsRow = kK * BITS / 32;
    constexpr int kGroups = kK / 32;
    auto col_of = [&](int line) { return line < kBH ? n0 + line : kNI + n0 + (line - kBH); };

    // weights: chunk i of thread t is row (256 i + t) / kGW, group t % kGW of the window
    const int wq = tid % kGW;
    const uint32_t* wsrc[kWChunks];
#pragma unroll
    for (int i = 0; i < kWChunks; ++i) wsrc[i] = a.words + static_cast<long long>(col_of((256 * i + tid) / kGW)) * kWordsRow + wq * BITS;
    // terms: thread t loads 8 groups of row t % 128, scales (t < 128) or biases; paired tables: groups 0-3 (t < 128)
    // or 4-7 of the 8, scale and bias each
    const int trow = tid & (kBN - 1);
    const bool tbias = tid >= kBN;
    const bool tpair = a.scale.kind == kScaleBF16Pair;
    const uint4* tsrc = tpair ? reinterpret_cast<const uint4*>(static_cast<const uint32_t*>(a.scale.p) +
                                                               static_cast<long long>(col_of(trow)) * kGroups + (tbias ? 4 : 0))
                              : reinterpret_cast<const uint4*>(static_cast<const uint16_t*>(tbias ? a.bias.p : a.scale.p) +
                                                               static_cast<long long>(col_of(trow)) * kGroups);
    const int tstep = tpair ? 2 : 1;  // 16-byte loads from one 8-group term load to the next
    // x: chunk i of thread t is row (256 i + t) / kXPer, 16-byte piece t % kXPer (group piece / 4, quad lane t % 4)
    const int xc = tid % kXPer;

    const int wm = wave & 1;
    const int wn = wave >> 1;
    const int rl = lane & 15;
    const int rh = lane >> 4;
    const int lines[2] = {wn * 16 + rl, kBH + wn * 16 + rl};
    const f32x8 zero = {0, 0, 0, 0, 0, 0, 0, 0};
    uint16_t* out = reinterpret_cast<uint16_t*>(a.out16);

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
        u32x4 wreg[kWChunks];
        u32x4 xreg[kXChunks];
        uint4 treg;
        f32x8 acc[2][2];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
            for (int j = 0; j < 2; ++j) acc[mt][j] = zero;
        }
        auto load = [&](int w) {
#pragma unroll
            for (int i = 0; i < kWChunks; ++i) wreg[i] = *reinterpret_cast<const u32x4*>(wsrc[i] + w * kGW * BITS);
#pragma unroll
            for (int i = 0; i < kXChunks; ++i) xreg[i] = xsrc[i][w * kXPer];
            if (w % kTermEvery == 0) treg = tsrc[tstep * (w / kTermEvery)];
        };
        auto stash = [&](int w) {
#pragma unroll
            for (int i = 0; i < kWChunks; ++i) {
                const int row = (256 * i + tid) / kGW;
                const uint32_t wv[BITS] = {wreg[i].x, wreg[i].y, wreg[i].z, wreg[i].w};
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
                *reinterpret_cast<u32x4*>(xs + row * kLdw + xc * 4) = xreg[i];
                // the group's sum in order: quad lane 0 adds its 8 values, hands the sum to lane 1, ... lane 3 stores it
                const uint32_t q[4] = {xreg[i].x, xreg[i].y, xreg[i].z, xreg[i].w};
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
                const uint32_t tv[4] = {treg.x, treg.y, treg.z, treg.w};
                if (tpair) {
                    const int g0 = tbias ? 4 : 0;
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        sc[g0 + j][trow] = __uint_as_float(tv[j] << 16);
                        bi[g0 + j][trow] = __uint_as_float(tv[j] & 0xffff0000u);
                    }
                } else {
                    float (*dst)[kBN] = tbias ? bi : sc;
#pragma unroll
                    for (int j = 0; j < 8; ++j) dst[j][trow] = __uint_as_float(((tv[j >> 1] >> (16 * (j & 1))) & 0xffffu) << 16);
                }
            }
        };

        __syncthreads();
        load(0);
        stash(0);
        if (kWindows > 1) load(1);
        __syncthreads();
        for (int w = 0; w < kWindows; ++w) {
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
                    const int g8 = (kGW * w + gi) & 7;
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt) {
                        if (mt == 1 && !live1) continue;
                        float s_x[8];
#pragma unroll
                        for (int i = 0; i < 8; ++i) s_x[i] = sx[gi][wm * 32 + mt * 16 + 2 * i + rh];
#pragma unroll
                        for (int j = 0; j < 2; ++j) {
                            const float scale = sc[g8][lines[j]];
                            const float bias = bi[g8][lines[j]];
#pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                acc[mt][j][i] = Dec::fold_scale(acc[mt][j][i], d[mt][j][i], scale);
                                acc[mt][j][i] = Dec::fold_bias(acc[mt][j][i], s_x[i], bias);
                            }
                        }
                    }
                }
            }
            __syncthreads();
            if (w + 1 < kWindows) {
                stash(w + 1);
                if (w + 2 < kWindows) load(w + 2);
            }
            __syncthreads();
        }

#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row = wm * 32 + mt * 16 + 2 * i + rh;
                if (row >= rows_left) continue;
                float g = acc[mt][0][i];
                float u = acc[mt][1][i];
                if (limit > 0.f) {
                    g = fminf(g, limit);
                    u = fminf(fmaxf(u, -limit), limit);
                }
                const hip_bfloat16 v(g / (1.f + expf(-g)) * u);
                out[out_row(a, h0 + row) * kNI + n0 + wn * 16 + rl] = v.data;
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
