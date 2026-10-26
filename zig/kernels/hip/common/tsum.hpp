#pragma once

// Several wave sums at once: CNT values a lane give CNT sums, each bitwise equal to its own xor butterfly
// (`for (o = 16; o > 0; o >>= 1) v += shfl_xor(v, o)`). At offset 16 a lane keeps half of its values and adds the
// partner's copy of that half, at 8 half again, until one value is left, then plain xor steps for the rest. Every
// level adds the same two values as the butterfly does (own + partner), so the bits do not change; a lane only
// carries fewer of them: CNT - 1 + (5 - log2 CNT) moves instead of 5 CNT, by DPP row_xmask and permlanex16.
// Value j's sum ends in the lanes whose bits 4, 3, ... (most significant first) spell j.
//
// Ported from Strata, src/kernels/cuda/s26_tsum.cuh@4b9e05a2, under its license:
//
// MIT License
//
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <hip/hip_runtime.h>

namespace tf {
namespace rocm {

// The value of lane `lane ^ O` (wave32): permlanex16 across the two rows of 16, DPP row_xmask inside a row.
template <int O>
__device__ __forceinline__ float tsum_move(float v) {
    if constexpr (O == 16) {
        return __int_as_float(__builtin_amdgcn_permlanex16(__float_as_int(v), __float_as_int(v), 0x76543210u, 0xfedcba98u,
                                                           false, false));
    } else {
        return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x160 + O, 0xf, 0xf, false));
    }
}

template <int O>
__device__ __forceinline__ float tsum_plain(float x) {
    if constexpr (O >= 1) {
        x += tsum_move<O>(x);
        return tsum_plain<O / 2>(x);
    } else {
        return x;
    }
}

// v[0..CNT), CNT a power of two up to 32: this lane's values; returns the sum of value lane >> (5 - log2 CNT).
template <int CNT, int O = 16>
__device__ __forceinline__ float tsum(float* v, int lane) {
    if constexpr (CNT > 1) {
        constexpr int H = CNT / 2;
        const bool hi = (lane & O) != 0;
#pragma unroll
        for (int j = 0; j < H; ++j) {
            const float send = hi ? v[j] : v[j + H];
            const float keep = hi ? v[j + H] : v[j];
            v[j] = keep + tsum_move<O>(send);
        }
        return tsum<H, O / 2>(v, lane);
    } else {
        return tsum_plain<O>(v[0]);
    }
}

constexpr int tsum_log2(int p) { return p <= 1 ? 0 : 1 + tsum_log2(p / 2); }

}  // namespace rocm
}  // namespace tf
