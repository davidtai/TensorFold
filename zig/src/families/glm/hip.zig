//! GLM-5.3-Flash (model_type glm5_next) for the native HIP engine: admission now; the loader and the forward to come.
//! Kernels: zig/kernels/hip/glm/ (the "glm" group). Launchers: hip/<topic>.zig, one lane a file, re-exported by hip/ops.zig.
pub const config = @import("config.zig");
pub const Config = config.Config;
pub const exl3_admit = @import("exl3_admit.zig");
/// The launchers over the "glm" kernel group.
pub const ops = @import("hip/ops.zig");
/// The registry entry of the native server.
pub const native = @import("hip/native.zig");
/// The config.json checks on a real checkpoint (TF_GLM_CONFIG), for glm-hip-test.
pub const config_checkpoint = @import("config_checkpoint_test.zig");

test {
    _ = config;
    _ = exl3_admit;
    _ = config_checkpoint;
    @import("std").testing.refAllDecls(ops);
    @import("std").testing.refAllDecls(native);
}
