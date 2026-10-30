//! The launchers over the "glm" kernel group (zig/kernels/hip/glm/glm.hip). A lane adds one line here for its topic.
/// The group's code object on this device; each topic resolves its own kernels from it.
pub const group = @import("group.zig");
/// The bf16 row copy: the loader's self-test and row placement.
pub const copy = @import("copy.zig");

test {
    const std = @import("std");
    std.testing.refAllDecls(group);
    std.testing.refAllDecls(copy.Copy);
}
