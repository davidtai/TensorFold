//! The cooperative launch on a real GPU, in its own test binary: on some ROCm stacks (gfx1151, ROCm 10.0) a process that
//! made a cooperative launch hangs in its next hipModuleLoadData, so no other test may run after it in the same process.
//! `hip-gpu-test` runs this binary after runtime_tests.zig's, in a second process.
const std = @import("std");
const Runtime = @import("runtime.zig").Runtime;
const Context = @import("context.zig").Context;
const Stream = @import("stream.zig").Stream;
const DeviceBuffer = @import("memory.zig").DeviceBuffer;
const Module = @import("module.zig").Module;
const launch = @import("launch.zig");
const codeobject = @import("hip_probe");
const probe_image: @import("code_object.zig").Image = .{ .arch = codeobject.arch, .bytes = &codeobject.bytes };

test "a cooperative grid of one block a compute unit, and the device's attributes" {
    var r = try Runtime.open();
    defer r.close();
    var ctx = try Context.init(&r, 0);
    defer ctx.deinit();
    var stream = try Stream.init(&r);
    defer stream.deinit();
    var name_buf: [256]u8 = undefined;
    try std.testing.expect((try ctx.name(&name_buf)).len > 0);
    try std.testing.expectEqual(@as(c_int, 32), try ctx.attribute(.warp_size));
    try std.testing.expect(try ctx.attribute(.cooperative_launch) != 0);
    const cus: u32 = @intCast(try ctx.attribute(.multiprocessor_count));
    var arch_buffer: [256]u8 = undefined;
    var m = try Module.loadForArchitecture(&r, &.{probe_image}, try @import("device_arch.zig").query(&r, 0, &arch_buffer));
    defer m.unload();
    const fill = try m.function("tf_hip_fill_f32");
    try std.testing.expect(try fill.occupancy(256, 0) >= 1);
    try std.testing.expect(try fill.attribute(.max_threads_per_block) >= 256);
    var y = try DeviceBuffer.alloc(&r, cus * 256 * 4);
    defer y.free();
    var args: launch.Args = .{};
    try args.add(y.ptr);
    try args.add(@as(f32, 3));
    try args.add(cus * 256);
    try launch.launch(fill, .{ .grid = .{ .x = cus }, .block = .{ .x = 256 }, .cooperative = true }, stream, &args);
    try stream.synchronize();
    const got = try std.testing.allocator.alloc(f32, cus * 256);
    defer std.testing.allocator.free(got);
    try y.download(0, std.mem.sliceAsBytes(got));
    for (got, 0..) |v, i| try std.testing.expectEqual(3 + @as(f32, @floatFromInt(i)), v);
    try std.testing.expect(r.errorName(1).len > 0);
}
