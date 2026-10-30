//! GLM-5.3-Flash (model_type glm5_next) for the native HIP engine: admission now; the loader and the forward to come.
//! Kernels: zig/kernels/hip/glm/ (the "glm" group); launchers: zig/src/hip/launches/; tests: zig/src/hip/glm_kernel_tests.zig.
pub const config = @import("config.zig");
pub const Config = config.Config;
pub const exl3_admit = @import("exl3_admit.zig");
/// The registry entry of the native server.
pub const native = @import("hip/native.zig");

test {
    _ = config;
    _ = exl3_admit;
    @import("std").testing.refAllDecls(native);
}
