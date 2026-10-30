//! GLM-5.3-Flash (model_type glm5_next) for the native HIP engine: admission, the loader, the forward, the lane backend.
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
    /// A checkpoint directory to `Opened.model` in one call.
    pub const open = @import("hip/open.zig");
};
/// The lane core's backend and engine over the forward (the native server's `open`).
pub const serve = @import("hip/serve.zig");
/// The eager forward (prompt chunks, decode rows, the MTP layer) and the per-stream state and buffers it runs on.
pub const forward = @import("hip/forward.zig");
pub const state = @import("hip/state.zig");

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
    @import("std").testing.refAllDecls(hip_load.open);
    _ = serve;
    @import("std").testing.refAllDecls(serve.Hip);
    @import("std").testing.refAllDecls(serve.Engine);
    _ = forward;
    _ = state;
    _ = @import("hip/fwd_test.zig");
}
