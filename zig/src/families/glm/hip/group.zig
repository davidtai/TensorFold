//! The "glm" kernel group's code object (zig/kernels/hip/glm/glm.hip), loaded on the current device.
const hip = @import("hip");

/// The code object as the build embedded it for -Dhip-arch.
pub fn image() []align(8) const u8 {
    return hip.kernels.images[@backingInt(hip.kernels.Group.glm)];
}

/// Loads the group on the current device; a launcher table takes its kernels from the module (`Module.function`).
pub fn load(r: *const hip.Runtime) hip.Error!hip.Module {
    return hip.Module.load(r, image());
}
