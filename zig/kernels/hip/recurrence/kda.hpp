#pragma once

// GLM-5.3-Flash's KDA (Kimi delta attention) gate: a decay a key channel, where the Gated DeltaNet has one a head.
// decay = exp(lower * sigmoid(exp(a_log[head]) * (a + dt_bias[head * 128 + channel]))), in [exp(lower), 1]; beta is
// sigmoid(b) a head, as the DeltaNet's. The CUDA chain's formula (cuda/glm_kda.cu prep_kernel_body), in fp32.

#include <hip/hip_runtime.h>

namespace tf_kda {

constexpr int kDim = 128;  // KDA's key and value channels a head

__device__ __forceinline__ float decay(float a, float dt_bias, float a_log, float lower) {
    const float x = expf(a_log) * (a + dt_bias);
    return expf(lower * (1.0f / (1.0f + expf(-x))));
}

}  // namespace tf_kda
