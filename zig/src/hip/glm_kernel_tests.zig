//! glm-hip-test: GLM-5.3-Flash's HIP kernels against their host references, and the checkpoint's config checks.
//! `zig build glm-hip-test -Dhip=true` installs zig-out/bin/glm-hip-test; run it on the box under the GPU lock
//! (`hip-glm-kernel-test` builds and runs it). hip-kernel-test runs these too (kernel_tests.zig imports this file).
//! A lane adds one import line here for each test file (kernel_tests/glm/<topic>.zig) and touches nothing else.
test {
    _ = @import("kernel_tests/glm/config.zig");
    _ = @import("kernel_tests/glm/copy.zig");
    _ = @import("kernel_tests/kda.zig");
    _ = @import("kernel_tests/exl3.zig");
    _ = @import("kernel_tests/glm/bf16.zig");
    _ = @import("kernel_tests/glm/hc.zig");
    _ = @import("kernel_tests/glm/router.zig");
    _ = @import("kernel_tests/glm/mtp.zig");
}
