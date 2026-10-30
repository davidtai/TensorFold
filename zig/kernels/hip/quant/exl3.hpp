#pragma once

// EXL3 trellis tiles on RDNA (wave32): a lane's 8 stream values, the three codebooks, the 128-point Hadamard.
//
// Port of zig/kernels/cuda/exl3_decode.cuh (After ExLlamaV3, MIT, Copyright (c) 2025 Turboderp): lane L decodes
// values 8L .. 8L + 7 of a 16 x 16 tile into the two mma B fragment halves of exl3_format.zig's tilePlacement.
// The decode is exact: each value is one fp16 rounding, as exl3_format.zig's codebookValue. The codebook arithmetic
// follows the format bench's gfx1151 kernels (bench/hip-formats/formats/exl3/exl3_kernels.h on hipfmt/exl3 1e95cb74).

#include <hip/hip_runtime.h>

#include <cstdint>

namespace tf {
namespace rocm {
namespace exl3 {

typedef _Float16 h2 __attribute__((ext_vector_type(2)));
typedef _Float16 h4 __attribute__((ext_vector_type(4)));

// exl3_format.zig Codebook order.
enum Codebook : int { kInst3 = 0, kMcg = 1, kMul1 = 2 };

constexpr float kHad = 0.08838834764831845f;  // 1 / sqrt(128)

// 32-bit words of a tile: 256 values of K2 / 2 bits.
template <int K2>
__host__ __device__ constexpr int tile_words() {
    return 4 * K2;
}

// E(p): one past the last stream bit of value p's 16-bit window (exl3_format.zig streamEnds).
template <int K2>
__host__ __device__ constexpr int stream_end(int p) {
    return (K2 & 1) ? ((p + 1) * K2 - ((p + 1) & 1)) / 2 : (p + 1) * (K2 / 2);
}

// The words a lane reads for its 8 values: 2, or 3 for the widths whose windows span more.
template <int K2>
__host__ __device__ constexpr int lane_words() {
    int most = 0;
    for (int l = 0; l < 32; ++l) {
        const int first = (stream_end<K2>(8 * l) - 16 + 128 * K2) % 32;
        const int need = first + stream_end<K2>(8 * l + 7) - stream_end<K2>(8 * l) + 16;
        most = need > most ? need : most;
    }
    return (most + 31) / 32;
}

// The first word lane `lane` reads and the bit offset of its first window in it.
template <int K2>
__device__ __forceinline__ void lane_start(int lane, int& word, int& offset) {
    const int first = 4 * lane * K2 + K2 / 2 - 16 + 128 * K2;  // E(8 lane) - 16, made non-negative
    word = (first >> 5) % tile_words<K2>();
    offset = first & 31;
}

// This lane's words of a tile (`tile`: the tile's first 32-bit word).
template <int K2>
__device__ __forceinline__ void lane_load(const uint32_t* tile, int lane, uint32_t (&w)[lane_words<K2>()]) {
    int word, offset;
    lane_start<K2>(lane, word, offset);
#pragma unroll
    for (int i = 0; i < lane_words<K2>(); ++i) w[i] = tile[(word + i) % tile_words<K2>()];
}

// The upper 32 bits of (hi:lo) << s, s in [0, 32) (CUDA's __funnelshift_l(lo, hi, s)).
__device__ __forceinline__ uint32_t shl64(uint32_t lo, uint32_t hi, int s) {
    return s ? (hi << s) | (lo >> (32 - s)) : hi;
}

// Bits [d, d + 16) of the 96-bit big-endian stream hi:mid:lo (d a compile-time constant after unrolling).
__device__ __forceinline__ uint32_t window16(uint32_t hi, uint32_t mid, uint32_t lo, int d) {
    if (d <= 16) return (hi >> (16 - d)) & 0xffffu;
    if (d < 32) return shl64(mid, hi, d) >> 16;
    if (d <= 48) return (mid >> (48 - d)) & 0xffffu;
    return shl64(lo, mid, d - 32) >> 16;
}

// The 16-bit states of this lane's 8 values (8 lane + j) from its words.
template <int K2>
__device__ __forceinline__ void lane_states(const uint32_t (&w)[lane_words<K2>()], int lane, uint32_t (&s)[8]) {
    constexpr int LW = lane_words<K2>();
    int word, offset;
    lane_start<K2>(lane, word, offset);
    const uint32_t c = LW > 2 ? w[LW > 2 ? 2 : 0] : 0u;
    const uint32_t hi = shl64(w[1], w[0], offset);  // the stream from the lane's first window on
    const uint32_t mid = shl64(c, w[1], offset);
    const uint32_t lo = c << offset;
#pragma unroll
    for (int j = 0; j < 8; ++j) s[j] = window16(hi, mid, lo, stream_end<K2>(j) - stream_end<K2>(0));
}

// Two codebook values from two 16-bit states, as fp16 pair bits (the first state in the low half).
template <int CB>
__device__ __forceinline__ uint32_t decode2(uint32_t s0, uint32_t s1) {
    if constexpr (CB == kMul1) {
        // fp16 bits of 1024 + the byte sum (exact: unit spacing in [1024, 2048)), then one fp16 fma
        const uint32_t x0 = s0 * 0x83DCD12Du, x1 = s1 * 0x83DCD12Du;
        const uint32_t h0 = 0x6400u + (x0 & 255u) + ((x0 >> 8) & 255u) + ((x0 >> 16) & 255u) + (x0 >> 24);
        const uint32_t h1 = 0x6400u + (x1 & 255u) + ((x1 >> 8) & 255u) + ((x1 >> 16) & 255u) + (x1 >> 24);
        const uint32_t hh = (h0 & 0xffffu) | (h1 << 16);
        const _Float16 sc = __builtin_bit_cast(_Float16, static_cast<unsigned short>(0x1EEE));
        const _Float16 bi = __builtin_bit_cast(_Float16, static_cast<unsigned short>(0xC931));
        const h2 r = __builtin_elementwise_fma(__builtin_bit_cast(h2, hh), h2{sc, sc}, h2{bi, bi});
        return __builtin_bit_cast(uint32_t, r);
    } else {
        uint32_t x0, x1;
        if constexpr (CB == kMcg) {
            x0 = s0 * 0xCBAC1FEDu;
            x1 = s1 * 0xCBAC1FEDu;
        } else {
            x0 = s0 * 89226354u + 64248484u;
            x1 = s1 * 89226354u + 64248484u;
        }
        x0 = (x0 & 0x8FFF8FFFu) ^ 0x3B603B60u;
        x1 = (x1 & 0x8FFF8FFFu) ^ 0x3B603B60u;
        const uint32_t lo = (x0 & 0xffffu) | (x1 << 16);
        const uint32_t hi = (x0 >> 16) | (x1 & 0xffff0000u);
        const h2 r = __builtin_bit_cast(h2, lo) + __builtin_bit_cast(h2, hi);  // one fp16 rounding of an exact sum
        return __builtin_bit_cast(uint32_t, r);
    }
}

// This lane's 8 values as 4 fp16 pairs: p[0] = rows (2t, 2t + 1) and p[1] = rows (2t + 8, 2t + 9) of column g; p[2],
// p[3] the same of column g + 8 (t = lane & 3, g = lane >> 2; exl3_format.zig tilePlacement, row = k, column = n).
template <int K2, int CB>
__device__ __forceinline__ void decode_lane(const uint32_t (&w)[lane_words<K2>()], int lane, uint32_t (&p)[4]) {
    uint32_t s[8];
    lane_states<K2>(w, lane, s);
    p[0] = decode2<CB>(s[0], s[1]);
    p[1] = decode2<CB>(s[2], s[3]);
    p[2] = decode2<CB>(s[4], s[5]);
    p[3] = decode2<CB>(s[6], s[7]);
}

// Row (k) and column (n) in the 16 x 16 tile of this lane's value j.
__host__ __device__ constexpr int value_row(int lane, int j) { return 2 * (lane & 3) + (j & 1) + 8 * ((j >> 1) & 1); }
__host__ __device__ constexpr int value_col(int lane, int j) { return (lane >> 2) + 8 * (j >> 2); }

// Unscaled Walsh-Hadamard transform of 128 fp32 values, 4 a lane (index 4 lane + j), in a fixed butterfly order.
__device__ __forceinline__ void fwht128(float (&v)[4], int lane) {
    const float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c;
    v[1] = b + d;
    v[2] = a - c;
    v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float o = __shfl_xor(v[j], m, 32);
            v[j] = (lane & m) ? o - v[j] : v[j] + o;
        }
    }
}

__device__ __forceinline__ float bf16_value(uint16_t b) { return __builtin_bit_cast(float, static_cast<uint32_t>(b) << 16); }
__device__ __forceinline__ float f16_value(uint16_t b) { return static_cast<float>(__builtin_bit_cast(_Float16, b)); }
__device__ __forceinline__ uint16_t f16_bits(float v) { return __builtin_bit_cast(uint16_t, static_cast<_Float16>(v)); }

// fp32 rounded to bf16 (nearest even), as torch's bf16 cast; a NaN stays a NaN.
__device__ __forceinline__ float bf16r(float x) {
    uint32_t u = __builtin_bit_cast(uint32_t, x);
    if ((u & 0x7fffffffu) > 0x7f800000u) return x;
    u += 0x7fffu + ((u >> 16) & 1u);
    return __builtin_bit_cast(float, u & 0xffff0000u);
}

// The SwiGLU clamps as torch.clamp has them: a NaN stays a NaN, where fminf and fmaxf hand back the limit.
__device__ __forceinline__ float up_to(float x, float l) { return x > l ? l : x; }
__device__ __forceinline__ float within(float x, float l) { return x > l ? l : (x < -l ? -l : x); }

// v_dot2_f32_f16: c + a.x b.x + a.y b.y in fp32.
__device__ __forceinline__ float dot2(uint32_t a, h2 b, float c) {
    return __builtin_amdgcn_fdot2(__builtin_bit_cast(h2, a), b, c, false);
}

// The sum of the 4 lanes of a quad (every lane of the quad gets the same bits).
__device__ __forceinline__ float quad_sum(float v) {
    v += __shfl_xor(v, 1, 32);
    v += __shfl_xor(v, 2, 32);
    return v;
}

}  // namespace exl3
}  // namespace rocm
}  // namespace tf
