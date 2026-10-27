#pragma once

// numerics=inexact: a prompt's plain products of 4-bit group-32 MLX weights with the codes dequantized in the tile.
//
// w = bf16(fma(scale, q, bias)) is staged in LDS, and one WMMA accumulation runs over all K: no per-group dot and
// fold, so a wave holds a 64 x 64 tile (16 accumulators) and reads half a fragment a WMMA where dense_gemm.hpp reads
// three quarters. The sums differ from the exact tiles' by the BF16 rounding of w (relative RMS about 1.7e-3).
// Block 128 rows x 256 columns, 8 waves, K in 64-wide windows in one LDS buffer with the next window in registers:
// Strata's kernel_bk64 layout (github.com/Niko1221/Strata, MIT, src/prefill/wmma_gemm.cu@4b9e05a2:396-482).
// gfx1151 at 8,192 rows: dn_z 6.58 ms (39.2 TFLOPS) against 7.66 exact, out 6.88 against 7.90 (w2/dense_inexact.hip).

#include "common/wmma.hpp"
#include "quant/mlx.hpp"
#include "tiles/epilogue.hpp"

namespace tf {
namespace rocm {

namespace dense_inexact {
constexpr int kBM = 128;
constexpr int kBN = 256;
constexpr int kKW = 64;
constexpr int kLd = kKW + 8;  // bf16 a staged row
constexpr int kThreads = 256;

__device__ __forceinline__ uint32_t round_bf16(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u + (((u >> 16) & 1u) + 0x7fffu)) >> 16;
}

// The 8 codes of a word as 4 BF16 pairs of fma(scale, q, bias), in order.
__device__ __forceinline__ u32x4 dequant8(uint32_t w, float s, float b) {
    uint32_t p[4];
#pragma unroll
    for (int h = 0; h < 4; ++h) {
        const float lo = fmaf(s, static_cast<float>((w >> (8 * h)) & 0xfu), b);
        const float hi = fmaf(s, static_cast<float>((w >> (8 * h + 4)) & 0xfu), b);
        p[h] = round_bf16(lo) | (round_bf16(hi) << 16);
    }
    return u32x4{p[0], p[1], p[2], p[3]};
}
}  // namespace dense_inexact

// K: the product's K (a multiple of 64). bf16 x and tables, plain products; fp32 out, or bf16 (nearest even) with out16.
template <int K>
__device__ __forceinline__ void wmma_dense_inexact_tile(const Affine& a) {
#if TF_DEVICE_WMMA_GFX11
    using namespace dense_inexact;
    using T = WmmaBF16;
    constexpr int kWindows = K / kKW;
    constexpr int kGroups = K / 32;
    constexpr int kWordsRow = K / 8;
    static_assert(K % kKW == 0, "whole windows");
    __shared__ __attribute__((aligned(16))) __bf16 xs[kBM * kLd];
    __shared__ __attribute__((aligned(16))) __bf16 ws[kBN * kLd];
    const int num_n = (a.n + kBN - 1) / kBN;
    const int m0 = blockIdx.x / num_n * kBM;
    const int n0 = blockIdx.x % num_n * kBN;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int rl = lane & 15;
    const int rh = lane >> 4;
    // a thread's loads: group q of x row r (64 bytes); column n0 + tid's two groups of codes (2 x 16 bytes) and terms
    const int xr = tid >> 1;
    const int xq = tid & 1;
    const int xrow = m0 + xr < a.m ? m0 + xr : a.m - 1;
    const u32x4* xsrc = reinterpret_cast<const u32x4*>(static_cast<const __bf16*>(a.x) + static_cast<long long>(xrow) * K) + xq * 4;
    const int col = n0 + tid < a.n ? n0 + tid : a.n - 1;
    const uint4* wsrc = reinterpret_cast<const uint4*>(a.words + static_cast<long long>(col) * kWordsRow);
    const uint16_t* ssrc = static_cast<const uint16_t*>(a.scale.p) + static_cast<long long>(col) * kGroups;
    const uint16_t* bsrc = static_cast<const uint16_t*>(a.bias.p) + static_cast<long long>(col) * kGroups;
    u32x4 xreg[4];
    uint4 wreg[2];
    uint16_t sreg[2];
    uint16_t breg[2];
    // unconditional loads (a window past the last repeats it): no vmcnt(0) at the stash
    auto load = [&](int w) {
        const int wl = w < kWindows ? w : kWindows - 1;
#pragma unroll
        for (int v = 0; v < 4; ++v) xreg[v] = xsrc[wl * 8 + v];
#pragma unroll
        for (int g = 0; g < 2; ++g) {
            wreg[g] = wsrc[2 * wl + g];
            sreg[g] = ssrc[2 * wl + g];
            breg[g] = bsrc[2 * wl + g];
        }
    };
    auto stash = [&]() {
        u32x4* xd = reinterpret_cast<u32x4*>(xs + xr * kLd + xq * 32);
#pragma unroll
        for (int v = 0; v < 4; ++v) xd[v] = xreg[v];
        u32x4* wd = reinterpret_cast<u32x4*>(ws + tid * kLd);
#pragma unroll
        for (int g = 0; g < 2; ++g) {
            const float s = __uint_as_float(static_cast<uint32_t>(sreg[g]) << 16);
            const float b = __uint_as_float(static_cast<uint32_t>(breg[g]) << 16);
            const uint32_t wv[4] = {wreg[g].x, wreg[g].y, wreg[g].z, wreg[g].w};
#pragma unroll
            for (int v = 0; v < 4; ++v) wd[g * 4 + v] = dequant8(wv[v], s, b);
        }
    };
    // wave (wm, wn): rows 64 wm.., columns 64 wn..; a lane: column lane % 16, rows 2 i + lane / 16
    const int wm = wave & 1;
    const int wn = wave >> 1;
    f32x8 acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = f32x8{0, 0, 0, 0, 0, 0, 0, 0};
    }
    load(0);
    stash();
    load(1);
    lds_barrier();
    for (int w = 0; w < kWindows; ++w) {
#pragma unroll
        for (int ks = 0; ks < kKW; ks += 16) {
            bf16x16 af[4];
            bf16x16 bf[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) af[i] = frag16(reinterpret_cast<const uint32_t*>(xs + (wm * 64 + i * 16 + rl) * kLd + ks));
#pragma unroll
            for (int j = 0; j < 4; ++j) bf[j] = frag16(reinterpret_cast<const uint32_t*>(ws + (wn * 64 + j * 16 + rl) * kLd + ks));
#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = T::mma(af[i], bf[j], acc[i][j]);
            }
        }
        lds_barrier();
        // past the last window the stash rewrites LDS no later window reads
        stash();
        load(w + 2);
        lds_barrier();
    }
    auto put = [&](auto store) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int row = m0 + wm * 64 + i * 16 + 2 * e + rh;
                if (row >= a.m) continue;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int c = n0 + wn * 64 + j * 16 + rl;
                    if (c < a.n) store(static_cast<long long>(row) * a.n + c, acc[i][j][e]);
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

}  // namespace rocm
}  // namespace tf
