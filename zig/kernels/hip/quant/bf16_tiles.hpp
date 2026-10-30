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

// A prompt: the dense WMMA tile with its weights copied, not dequantized (one WMMA chain over K), at a compiled K.
template <int K>
__global__ void __launch_bounds__(dense_inexact::kThreads) bf16_wmma(Affine a) {
    wmma_dense_inexact_tile<K, true>(a);
}

}  // namespace rocm
}  // namespace tf
