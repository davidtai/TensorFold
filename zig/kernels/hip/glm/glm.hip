// GLM-5.3-Flash's own kernels as one code object (the build's "glm" group). Each lane topic keeps its kernels in one
// .hpp in this folder and adds one #include line below; the build tracks every file in this folder.

#include <hip/hip_runtime.h>
#include <cstdint>

// Lane topics: one #include "<topic>.hpp" each.

// Rows of bf16 values copied as stored (16-bit words, no conversion): `rows` rows of `cols` values from `src` at a row
// stride of `src_stride` values to `dst` at `dst_stride`. The loader's self-test, and the row placement of its buffers.
extern "C" __global__ void tf_glm_copy_rows_bf16(
    const uint16_t* __restrict__ src,
    uint16_t* __restrict__ dst,
    int rows,
    int cols,
    long long src_stride,
    long long dst_stride) {
    for (long long row = blockIdx.y; row < rows; row += gridDim.y) {
        const uint16_t* s = src + row * src_stride;
        uint16_t* d = dst + row * dst_stride;
        for (long long c = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; c < cols;
             c += static_cast<long long>(gridDim.x) * blockDim.x)
            d[c] = s[c];
    }
}
