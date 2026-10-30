//! GLM-5.3-Flash (model_type glm5_next) for the native HIP engine: admission now; the loader and the forward to come.
//! Kernels: zig/kernels/hip/glm/ (the "glm" group); launchers: zig/src/hip/launches/; tests: zig/src/hip/glm_kernel_tests.zig.
pub const config = @import("config.zig");
pub const Config = config.Config;
pub const exl3_admit = @import("exl3_admit.zig");
/// The registry entry of the native server.
pub const native = @import("hip/native.zig");
/// The checkpoint onto the GPU: headers, placement, the streaming load, the model's addresses, the sha256 sample,
/// and the TF_GLM_DUMP writer.
pub const hip_load = struct {
    pub const shards = @import("hip/shards.zig");
    pub const layout = @import("hip/layout.zig");
    pub const weights = @import("hip/weights.zig");
    pub const load = @import("hip/load.zig");
    pub const verify = @import("hip/verify.zig");
    pub const dump = @import("hip/dump.zig");
};

test {
    _ = config;
    _ = exl3_admit;
    @import("std").testing.refAllDecls(native);
    _ = hip_load.layout;
    _ = hip_load.load;
    _ = hip_load.dump;
    @import("std").testing.refAllDecls(hip_load.weights);
    @import("std").testing.refAllDecls(hip_load.verify);
    @import("std").testing.refAllDecls(hip_load.shards);
}
