#pragma once

// GLM-5.3-Flash's KDA (Kimi delta attention) around the DeltaNet scans of gated_delta.hip: the CUDA GLM path's prep
// (conv, gates a channel, beta, q and k norms), out (the gated RMSNorm) and conv shift, ported with their math and
// order (cuda/glm_kda.cu prep_kernel_body and out_kernel_body, cuda/glm_glue_rows.cuh tf_glm_conv_shift). Buffers and
// dtypes are the CUDA path's (cuda_state.zig, cuda_kda_block.zig): the stacked bf16 projection rows
// [q | k | v | f_a | g_a | b] (row stride p_stride, beta's logits at b_off), the bf16 decay and gate rows, the bf16 conv
// window [3][3 heads 128], the saves k fp32, v bf16, decays fp32 and beta fp32. The conv taps are bf16 as the
// checkpoint stores them (CUDA widens them at load). The read-out y is stored fp32 and rounded to bf16 where out reads
// it: CUDA's rounding point (glm_kda.cu: ys = bf(o), or the step kernel's bf16 y_out), the same bits.

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

// Block (row, head), 128 threads: out = bf16(norm_w rms(y) sigmoid(gate)), out_kernel_body's values: y (fp32 as the scan
// writes it) rounded to bf16 first, as CUDA rounds the read-out before the norm.
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
            const float v = bf(y[base + lane * 4 + i]);
            ss = ss + v * v;
        }
        ss = wave_sum(ss);
        if (lane == 0) rinv = 1.0f / sqrtf(ss / static_cast<float>(DV) + eps);
    }
    __syncthreads();
    const float yn = bf(y[base + t]) * rinv;
    const float yw = widen(norm_w[t]) * yn;
    const float gate = widen(G[static_cast<size_t>(r) * g_stride + h * DV + t]);
    out[base + t] = bf16_bits(yw * sigmoid(gate));
}

// The wave forms of prep and out (G2 K-1 / K-2): the block forms' values, a byte for a byte, in the DeltaNet's launch
// shapes. A lane owns channels 4 lane .. 4 lane + 3 of a head, as the block forms' norms read them (lane x 4 + i), so
// every sum keeps its order: the conv's taps in order, a channel at a time; the norms' i = 0..3 then xor 16 ... 1.
constexpr int kKdaSeg = 16;  // rows a prep wave slides its conv window over

// A wave a (16-row segment, head, part), 8 waves a block: part 0 q, 1 k (conv, SiLU, bf16, L2 norm), 2 v (conv, SiLU,
// bf16), 3 the decays of the head's 128 key channels and beta. tf_kda_prep's inputs and outputs. Wave u of the grid:
// part = u % 4, head = (u / 4) % H, segment = u / (4 H).
extern "C" __global__ void __launch_bounds__(256)
tf_kda_prep_wave(int H, const uint16_t* __restrict__ P, int p_stride, int b_off, const uint16_t* __restrict__ A,
                 int a_stride, const uint16_t* __restrict__ cs, const uint16_t* __restrict__ cw,
                 const float* __restrict__ a_log, const float* __restrict__ dt_bias, float lower,
                 float* __restrict__ q_out, float* __restrict__ k_save, uint16_t* __restrict__ v_save,
                 float* __restrict__ g_save, float* __restrict__ b_save, int rows) {
    using namespace tf_kda;
    constexpr int DK = kDim;
    const int C = 3 * H * DK;
    const int lane = threadIdx.x & 31;
    const int u = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (u >= ((rows + kKdaSeg - 1) / kKdaSeg) * H * 4) return;
    const int part = u % 4, h = (u / 4) % H, seg = u / (4 * H);
    const int r0 = seg * kKdaSeg, r1 = min(rows, r0 + kKdaSeg);
    const int ch = h * DK + lane * 4;  // the lane's first channel within a part
    if (part == 3) {
        const float rate = a_log[h];
        float dtb[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) dtb[i] = dt_bias[ch + i];
        for (int r = r0; r < r1; ++r) {
            const uint2 a4 = *reinterpret_cast<const uint2*>(A + static_cast<size_t>(r) * a_stride + ch);
            const uint16_t a[4] = {static_cast<uint16_t>(a4.x & 0xffffu), static_cast<uint16_t>(a4.x >> 16),
                                   static_cast<uint16_t>(a4.y & 0xffffu), static_cast<uint16_t>(a4.y >> 16)};
            float4 g;
            g.x = decay(widen(a[0]), dtb[0], rate, lower);
            g.y = decay(widen(a[1]), dtb[1], rate, lower);
            g.z = decay(widen(a[2]), dtb[2], rate, lower);
            g.w = decay(widen(a[3]), dtb[3], rate, lower);
            *reinterpret_cast<float4*>(g_save + (static_cast<size_t>(r) * H + h) * DK + lane * 4) = g;
            if (lane == 0) b_save[r * H + h] = bf(sigmoid(widen(P[static_cast<size_t>(r) * p_stride + b_off + h])));
        }
        return;
    }
    const int c = part * H * DK + ch;  // the lane's first conv channel
    float w[4][kTaps], win[4][kTaps];
    {
        const uint4 lo = *reinterpret_cast<const uint4*>(cw + static_cast<size_t>(c) * kTaps);
        const uint4 hi = *reinterpret_cast<const uint4*>(cw + static_cast<size_t>(c) * kTaps + 8);
        const uint32_t ws[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int tap = 0; tap < kTaps; ++tap) {
                const uint32_t word = ws[(i * kTaps + tap) >> 1];
                w[i][tap] = widen(static_cast<uint16_t>(((i * kTaps + tap) & 1) ? word >> 16 : word & 0xffffu));
            }
    }
    // input t of the window [conv window; P rows] for the lane's 4 channels
    auto input = [&](int t, float (&x)[4]) {
        const uint16_t* src = t < kTaps - 1 ? cs + static_cast<size_t>(t) * C + c
                                            : P + static_cast<size_t>(t - (kTaps - 1)) * p_stride + c;
        const uint2 v = *reinterpret_cast<const uint2*>(src);
        x[0] = widen(static_cast<uint16_t>(v.x & 0xffffu));
        x[1] = widen(static_cast<uint16_t>(v.x >> 16));
        x[2] = widen(static_cast<uint16_t>(v.y & 0xffffu));
        x[3] = widen(static_cast<uint16_t>(v.y >> 16));
    };
#pragma unroll
    for (int tap = 0; tap < kTaps - 1; ++tap) {
        float x[4];
        input(r0 + tap, x);
#pragma unroll
        for (int i = 0; i < 4; ++i) win[i][tap] = x[i];
    }
    for (int r = r0; r < r1; ++r) {
        float x[4];
        input(r + kTaps - 1, x);
        float act[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            win[i][kTaps - 1] = x[i];
            float acc = 0.0f;
#pragma unroll
            for (int tap = 0; tap < kTaps; ++tap) acc = acc + w[i][tap] * win[i][tap];
            act[i] = bf(acc / (1.0f + expf(-acc)));
#pragma unroll
            for (int tap = 0; tap + 1 < kTaps; ++tap) win[i][tap] = win[i][tap + 1];
        }
        const size_t base = (static_cast<size_t>(r) * H + h) * DK + lane * 4;
        if (part == 2) {
            uint2 o;
            o.x = static_cast<uint32_t>(bf16_bits(act[0])) | (static_cast<uint32_t>(bf16_bits(act[1])) << 16);
            o.y = static_cast<uint32_t>(bf16_bits(act[2])) | (static_cast<uint32_t>(bf16_bits(act[3])) << 16);
            *reinterpret_cast<uint2*>(v_save + base) = o;
            continue;
        }
        float ss = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) ss = ss + act[i] * act[i];
        ss = wave_sum(ss);
        const float inv = 1.0f / sqrtf(ss + 1e-6f);
        float4 y;
        float* yv = reinterpret_cast<float*>(&y);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float e = act[i] * inv;
            if (part == 0) e = e * (1.0f / sqrtf(static_cast<float>(DK)));
            yv[i] = e;
        }
        *reinterpret_cast<float4*>((part == 0 ? q_out : k_save) + base) = y;
    }
}

// A wave a (row, head), 8 waves a block: tf_kda_out's values; the wave's lane x 4 + i sum and butterfly as its wave 0.
extern "C" __global__ void __launch_bounds__(256)
tf_kda_out_wave(int H, const float* __restrict__ y, const uint16_t* __restrict__ G, int g_stride,
                const uint16_t* __restrict__ norm_w, float eps, uint16_t* __restrict__ out, int rows) {
    using namespace tf_kda;
    constexpr int DV = kDim;
    const int lane = threadIdx.x & 31;
    const int u = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (u >= rows * H) return;
    const int r = u / H, h = u % H;
    const size_t base = static_cast<size_t>(u) * DV + lane * 4;
    const float4 y4 = *reinterpret_cast<const float4*>(y + base);
    const float v[4] = {bf(y4.x), bf(y4.y), bf(y4.z), bf(y4.w)};
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) ss = ss + v[i] * v[i];
    ss = wave_sum(ss);
    const float rinv = 1.0f / sqrtf(ss / static_cast<float>(DV) + eps);
    const uint2 g4 = *reinterpret_cast<const uint2*>(G + static_cast<size_t>(r) * g_stride + h * DV + lane * 4);
    const uint2 n4 = *reinterpret_cast<const uint2*>(norm_w + lane * 4);
    const uint32_t gw[2] = {g4.x, g4.y}, nw[2] = {n4.x, n4.y};
    uint16_t o[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint16_t gb = static_cast<uint16_t>((i & 1) ? gw[i >> 1] >> 16 : gw[i >> 1] & 0xffffu);
        const uint16_t nb = static_cast<uint16_t>((i & 1) ? nw[i >> 1] >> 16 : nw[i >> 1] & 0xffffu);
        const float yn = v[i] * rinv;
        const float yw = widen(nb) * yn;
        o[i] = bf16_bits(yw * sigmoid(widen(gb)));
    }
    uint2 w;
    w.x = static_cast<uint32_t>(o[0]) | (static_cast<uint32_t>(o[1]) << 16);
    w.y = static_cast<uint32_t>(o[2]) | (static_cast<uint32_t>(o[3]) << 16);
    *reinterpret_cast<uint2*>(out + base) = w;
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
