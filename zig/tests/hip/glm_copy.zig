//! The "glm" group's bf16 row copy on the GPU: every 16-bit word as stored, rows at new strides, the gaps untouched.
const std = @import("std");
const hip = @import("hip");
const glm = @import("glm");

/// Words that a float path would change: NaN payloads, both zeros, subnormals, infinities, the extremes.
const awkward = [_]u16{ 0x7fc1, 0xffff, 0x7f81, 0x8000, 0x0000, 0x0001, 0x807f, 0x7f80, 0xff80, 0x7f7f, 0xff7f };

const Case = struct { rows: usize, cols: usize, src_stride: usize, dst_stride: usize };

test "GLM bf16 row copy: the words as stored at every stride; the gaps between rows keep their bytes" {
    var r = try hip.Runtime.open();
    defer r.close();
    var ctx = try hip.Context.init(&r, 0);
    defer ctx.deinit();
    var m = try glm.ops.group.load(&r);
    defer m.unload();
    const copy = try glm.ops.copy.Copy.load(m);
    var stream = try hip.Stream.init(&r);
    defer stream.deinit();
    const gpa = std.testing.allocator;
    var rng: std.Random.DefaultPrng = .init(0x61AC);
    const cases = [_]Case{
        .{ .rows = 1, .cols = 1, .src_stride = 1, .dst_stride = 1 },
        .{ .rows = 3, .cols = 1000, .src_stride = 1003, .dst_stride = 1024 },
        .{ .rows = 64, .cols = 4096, .src_stride = 4096, .dst_stride = 4096 },
        .{ .rows = 5, .cols = 300_000, .src_stride = 300_001, .dst_stride = 300_032 }, // past the column blocks' reach
        .{ .rows = 70_000, .cols = 8, .src_stride = 9, .dst_stride = 64 }, // past the row blocks' reach
    };
    for (cases) |c| {
        const src_words = (c.rows - 1) * c.src_stride + c.cols;
        const dst_words = (c.rows - 1) * c.dst_stride + c.cols;
        const src = try gpa.alloc(u16, src_words);
        defer gpa.free(src);
        for (src, 0..) |*w, i| w.* = if (i % 13 == 0) awkward[(i / 13) % awkward.len] else rng.random().int(u16);
        var d_src = try hip.DeviceBuffer.fromHost(&r, std.mem.sliceAsBytes(src));
        defer d_src.free();
        var d_dst = try hip.DeviceBuffer.alloc(&r, dst_words * 2);
        defer d_dst.free();
        try d_dst.fill8(0xA5);
        try copy.rows(d_src.base(), d_dst.base(), c.rows, c.cols, c.src_stride, c.dst_stride, stream);
        try stream.synchronize();
        const got = try gpa.alloc(u16, dst_words);
        defer gpa.free(got);
        try d_dst.download(0, std.mem.sliceAsBytes(got));
        const want = try gpa.alloc(u16, dst_words);
        defer gpa.free(want);
        @memset(want, 0xA5A5);
        for (0..c.rows) |row| @memcpy(want[row * c.dst_stride ..][0..c.cols], src[row * c.src_stride ..][0..c.cols]);
        if (!std.mem.eql(u16, want, got)) {
            const at = std.mem.findDiff(u16, want, got).?;
            std.debug.print("copy {any}: word {d} is 0x{x:0>4}, want 0x{x:0>4}\n", .{ c, at, got[at], want[at] });
            return error.TestUnexpectedResult;
        }
    }
    std.debug.print("  glm bf16 row copy: {d} cases, every word as stored, the gaps untouched\n", .{cases.len});
}
