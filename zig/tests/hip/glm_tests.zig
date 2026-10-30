//! glm-hip-test: GLM-5.3-Flash's HIP kernels against their host references, and the checkpoint's config checks.
//! Build: zig build glm-hip-test -Dhip=true (zig-out/bin/glm-hip-test); run the binary on the box under the GPU lock.
//! One import line a lane topic; each topic's file imports "hip" and "glm" (zig/src/families/glm/hip.zig).
test {
    _ = @import("glm_config.zig");
    _ = @import("glm_copy.zig");
}
