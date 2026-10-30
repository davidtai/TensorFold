#pragma once

// GLM-5.3-Flash's KDA (Kimi delta attention) around the DeltaNet scans of gated_delta.hip: the CUDA GLM path's prep
// (conv, gates a channel, beta, q and k norms), out (the gated RMSNorm) and conv shift, ported with their math and
// order (cuda/glm_kda.cu prep_kernel_body and out_kernel_body, cuda/glm_glue_rows.cuh tf_glm_conv_shift). Buffers and
// dtypes are the CUDA path's (cuda_state.zig, cuda_kda_block.zig): the stacked bf16 projection rows
// [q | k | v | f_a | g_a | b] (row stride p_stride, beta's logits at b_off), the bf16 decay and gate rows, the bf16 conv
// window [3][3 heads 128], the saves k fp32, v bf16, decays fp32 and beta fp32. The conv taps are bf16 as the
// checkpoint stores them (CUDA widens them at load). One difference, inside the block: q, k and the read-out y stay
// fp32 (CUDA: q and k fp32, y bf16); out reads y fp32.

#include <hip/hip_runtime.h>

#include <cstdint>

namespace tf_kda {

constexpr int kDim = 128;  // KDA's key and value channels a head
constexpr int kTaps = 4;   // short_conv_kernel_size

// v as a scan reads it: fp32 as is (the DeltaNet), bf16 bits widened exactly (KDA's saved v).
__device__ __forceinline__ float widen(float x) { return x; }
__device__ __forceinline__ float widen(uint16_t bits) { return __uint_as_float(static_cast<uint32_t>(bits) << 16); }

// fp32 to bf16 bits, to nearest even (__float2bfloat16_rn); NaN stays NaN.
__device__ __forceinline__ uint16_t bf16_bits(float v) {
    const uint32_t u = __float_as_uint(v);
    if (v != v) return static_cast<uint16_t>((u >> 16) | 0x40u);
    return static_cast<uint16_t>((u + (((u >> 16) & 1u) + 0x7fffu)) >> 16);
}

__device__ __forceinline__ float bf(float x) { return widen(bf16_bits(x)); }

__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// The decay a key channel: exp(lower sigmoid(exp(A_log) (a + dt_bias))), in [exp(lower), 1].
__device__ __forceinline__ float decay(float a, float dt_bias, float a_log, float lower) {
    return expf(lower * sigmoid(expf(a_log) * (a + dt_bias)));
}

// glm_kda.cu's warp_sum on a wave32: the xor 16 ... 1 butterfly, every lane the sum.
__device__ __forceinline__ float wave_sum(float x) {
#pragma unroll
    for (int o = 16; o; o >>= 1) x += __shfl_xor(x, o, 32);
    return x;
}

}  // namespace tf_kda

// Block (row, head), 512 threads: conv + SiLU of the head's 3 x 128 q | k | v channels over [window; P rows], rounded to
// bf16; the decays of its 128 key channels and beta = bf16(sigmoid(b)); then q and k L2-normed (q by 128^-0.5 too).
// Taps cw bf16 [3 heads 128][4] as the checkpoint stores q|k|v_conv1d (widened here, exactly; CUDA widens at load).
// Out: q fp32, k fp32, v bf16, decays fp32 (rows, heads, 128) and beta fp32 (rows, heads). prep_kernel_body's values.
extern "C" __global__ void __launch_bounds__(512)
tf_kda_prep(int H, const uint16_t* __restrict__ P, int p_stride, int b_off, const uint16_t* __restrict__ A,
            int a_stride, const uint16_t* __restrict__ cs, const uint16_t* __restrict__ cw,
            const float* __restrict__ a_log, const float* __restrict__ dt_bias, float lower, float* __restrict__ q_out, float* __restrict__ k_save,
            uint16_t* __restrict__ v_save, float* __restrict__ g_save, float* __restrict__ b_save) {
    using namespace tf_kda;
    constexpr int DK = kDim;
    const int C = 3 * H * DK;
    const int r = blockIdx.x, h = blockIdx.y;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    __shared__ float qs[DK], ks[DK];
    const size_t base = (static_cast<size_t>(r) * H + h) * DK;
    if (t < 3 * DK) {
        const int c = (t / DK) * H * DK + h * DK + (t % DK);
        float acc = 0.0f;
#pragma unroll
        for (int tap = 0; tap < kTaps; ++tap) {
            const int at = r + tap;
            const float x = at < kTaps - 1 ? widen(cs[static_cast<size_t>(at) * C + c])
                                           : widen(P[static_cast<size_t>(at - (kTaps - 1)) * p_stride + c]);
            acc = acc + widen(cw[static_cast<size_t>(c) * kTaps + tap]) * x;
        }
        const float act = bf(acc / (1.0f + expf(-acc)));
        if (t < DK) qs[t] = act;
        else if (t < 2 * DK) ks[t - DK] = act;
        else v_save[base + t - 2 * DK] = bf16_bits(act);
    } else {
        const int i = t - 3 * DK;
        const float a = widen(A[static_cast<size_t>(r) * a_stride + h * DK + i]);
        g_save[base + i] = decay(a, dt_bias[h * DK + i], a_log[h], lower);
        if (i == 0) b_save[r * H + h] = bf(sigmoid(widen(P[static_cast<size_t>(r) * p_stride + b_off + h])));
    }
    __syncthreads();
    if (warp < 2) {
        const float* x = warp == 0 ? qs : ks;
        float v4[4], ss = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            v4[i] = x[lane * 4 + i];
            ss = ss + v4[i] * v4[i];
        }
        ss = wave_sum(ss);
        const float inv = 1.0f / sqrtf(ss + 1e-6f);
        float* dst = warp == 0 ? q_out : k_save;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float y = v4[i] * inv;
            if (warp == 0) y = y * (1.0f / sqrtf(static_cast<float>(DK)));
            dst[base + lane * 4 + i] = y;
        }
    }
}

// Block (row, head), 128 threads: out = bf16(norm_w rms(y) sigmoid(gate)), out_kernel_body's values with y fp32.
extern "C" __global__ void __launch_bounds__(128)
tf_kda_out(int H, const float* __restrict__ y, const uint16_t* __restrict__ G, int g_stride,
           const uint16_t* __restrict__ norm_w, float eps, uint16_t* __restrict__ out) {
    using namespace tf_kda;
    constexpr int DV = kDim;
    const int r = blockIdx.x, h = blockIdx.y;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    __shared__ float rinv;
    const size_t base = (static_cast<size_t>(r) * H + h) * DV;
    if (warp == 0) {
        float ss = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float v = y[base + lane * 4 + i];
            ss = ss + v * v;
        }
        ss = wave_sum(ss);
        if (lane == 0) rinv = 1.0f / sqrtf(ss / static_cast<float>(DV) + eps);
    }
    __syncthreads();
    const float yn = y[base + t] * rinv;
    const float yw = widen(norm_w[t]) * yn;
    const float gate = widen(G[static_cast<size_t>(r) * g_stride + h * DV + t]);
    out[base + t] = bf16_bits(yw * sigmoid(gate));
}

// tf_glm_conv_shift: layer blockIdx.x's 3-row bf16 window takes rows keep .. keep + 2 of [window; projection rows],
// c columns (3 heads 128) a row; window rows conv_layer apart a layer, projection rows proj_row apart, proj_layer a layer.
extern "C" __global__ void __launch_bounds__(256)
tf_kda_conv_shift(uint16_t* conv, const uint16_t* proj, long long keep, long long conv_layer, long long proj_layer,
                  long long proj_row, long long c) {
    const long long k = static_cast<long long>(blockIdx.y) * 256 + threadIdx.x;
    if (k >= c) return;
    const long long l = blockIdx.x;
    uint16_t v[3];
    for (int j = 0; j < 3; ++j) {
        const long long src = keep + j;
        v[j] = src < 3 ? conv[l * conv_layer + src * c + k] : proj[l * proj_layer + (src - 3) * proj_row + k];
    }
    for (int j = 0; j < 3; ++j) conv[l * conv_layer + j * c + k] = v[j];
}
