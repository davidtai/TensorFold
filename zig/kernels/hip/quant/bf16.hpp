#pragma once

// BF16 weight rows on the shared tiles: a WeightDecoder for the decode stream tile (tiles/stream.hpp), and the BF16-row
// form of the inexact dense WMMA tile (tiles/dense_inexact.hpp, BF16W) at GLM-5.3-Flash's K. GLM's non-expert linears
// run BF16 as stored. The weights are (n, k) bf16 row major in Affine::words (bits 16, group 32);
// scale and bias are not read.

#include <type_traits>

#include "common/vec.hpp"
#include "quant/mlx.hpp"

namespace tf {
namespace rocm {

// A 32-element chunk of a column is 16 words of bf16 pairs, already the BF16 Dot's operand: no decode, one term.
struct Bf16Decoder {
    using Args = Affine;
    static constexpr int kWords = 16;

    __device__ static bool take_item(Args& a, int z) { return tf::rocm::take_item(a, z); }

    __device__ static const uint32_t* words(const Args& a) { return a.words; }

    template <typename Idx>
    __device__ static Idx row_words(const Args& a) {
        return static_cast<Idx>(a.k) / 2;
    }

    template <bool WIDE>
    __device__ static void load(const uint32_t* src, uint32_t (&w)[kWords]) {
        if constexpr (WIDE) {
#pragma unroll
            for (int i = 0; i < kWords / 4; ++i) {
                const u32x4 v = reinterpret_cast<const u32x4*>(src)[i];
                w[4 * i] = v.x;
                w[4 * i + 1] = v.y;
                w[4 * i + 2] = v.z;
                w[4 * i + 3] = v.w;
            }
        } else {
#pragma unroll
            for (int i = 0; i < kWords; ++i) w[i] = src[i];
        }
    }

    // The tile's group terms: scale 1 (a lane past the row's last chunk still gets 0 from the tile), no bias.
    struct StreamTerms {
        __device__ explicit StreamTerms(const Args&) {}
        __device__ unsigned short scale_raw(long long) const { return 0x3f80; }
        __device__ unsigned short bias_raw(long long) const { return 0; }
        __device__ float widen(unsigned short v) const { return __uint_as_float(static_cast<uint32_t>(v) << 16); }
    };

    // fma(dot, 1, acc) is acc + dot rounded once; fma(dot, 0, acc) keeps a dead lane's acc.
    __device__ static float fold_scale(float acc, float dot, float scale) { return fmaf(dot, scale, acc); }
    __device__ static float fold_bias(float acc, float, float) { return acc; }

    // A non-finite sum of BF16 operands is that sum: there is no fold to replay.
    template <typename T>
    __device__ static float repair_stream(const Args&, const typename T::elem*, int, float original) {
        return original;
    }

    static constexpr int kWideBytes = 16;

    __device__ static bool wide_aligned(const Args& a) {
        return ((reinterpret_cast<uintptr_t>(a.words) | static_cast<uintptr_t>(a.k) * 2) & 15) == 0;
    }

    template <class Sides>
    __device__ static void bind(Args& a, const Sides& sides, int s) {
        a.words = sides.words[s];
    }

    // Pair I is elements 2 I and 2 I + 1, word I as stored.
    static constexpr bool kGapPairs = false;
    template <int I>
    static constexpr int lo() {
        return 2 * I;
    }
    static constexpr int gap() { return 1; }

    template <typename T, int I>
    __device__ static typename T::pair stream_pair(const uint32_t (&w)[kWords]) {
        static_assert(std::is_same_v<typename T::elem, __bf16>, "BF16 weights pair with BF16 activations");
        return __builtin_bit_cast(typename T::pair, w[I]);
    }
};

}  // namespace rocm
}  // namespace tf
