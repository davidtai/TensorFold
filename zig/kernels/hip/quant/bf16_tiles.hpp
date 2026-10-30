#pragma once

// BF16 weight rows on the shared tiles, named as launches/affine.zig looks them up (the bf16 build only).

#include "quant/bf16.hpp"
#include "quant/mlx_tiles.hpp"

namespace tf {
namespace rocm {

// Decode: the stream tile over BF16 rows, R rows and CB columns a lane; PAIR: (gate | up) rows, silu(gate) * up out.
template <int R, int CB, bool PAIR>
__global__ void __launch_bounds__(32 * kStreamWaves) bf16_stream(Affine a, StreamSides sides, int lpc_log2, int gshift) {
    using Epi = std::conditional_t<PAIR, StreamSwiglu<DotBF16>, StreamRound<DotBF16>>;
    stream_tile<Bf16Decoder, BF16Act, DotBF16, Epi, R, CB>(a, sides, lpc_log2, gshift);
}

// GLM's SwiGLU pair epilogue with CUDA's rounding points (glm/glue.hpp tf_glm_swiglu): the gate and up sums rounded to
// bf16, s = the silu table's entry for the gate's bf16 (its pointer in a.scale.p: BF16 rows carry no scale), u clamped
// to +-limit, out = bf16(u s) (u s exact in fp32).
struct StreamGlmSwiglu {
    static constexpr bool kPair = true;

    template <int CB, class Args>
    __device__ static void put(const Args& a, long long row, int col, const float (&v)[CB], int c, int pair_cols, float limit) {
        constexpr int CH = CB / 2;
        const uint16_t g = __builtin_bit_cast(uint16_t, narrow<DotBF16>(v[c]));
        const uint16_t ub = __builtin_bit_cast(uint16_t, narrow<DotBF16>(v[c + CH]));
        const float s = __uint_as_float(static_cast<uint32_t>(static_cast<const uint16_t*>(a.scale.p)[g]) << 16);
        const float u = fminf(fmaxf(__uint_as_float(static_cast<uint32_t>(ub) << 16), -limit), limit);
        static_cast<__bf16*>(static_cast<void*>(a.out16))[row * pair_cols + col] = narrow<DotBF16>(u * s);
    }
};

// Decode's GLM (gate | up) product: the stream tile over BF16 rows at 2 columns a lane with StreamGlmSwiglu.
template <int R>
__global__ void __launch_bounds__(32 * kStreamWaves) bf16_stream_glu(Affine a, StreamSides sides, int lpc_log2, int gshift) {
    stream_tile<Bf16Decoder, BF16Act, DotBF16, StreamGlmSwiglu, R, 2>(a, sides, lpc_log2, gshift);
}

// A prompt: the dense WMMA tile (tiles/dense_gemm.hpp) with BF16 weight rows copied as stored, at a compiled K: each
// 32-k group's two WMMA from zero, folded on the VALU in K order (no long chain through the WMMA accumulator).
template <int K>
__global__ void __launch_bounds__(dense_tile::kThreads) bf16_wmma(Affine a) {
    wmma_dense_tile<K, DenseTune<K>::kAhead, false, true>(a);
}

}  // namespace rocm
}  // namespace tf
