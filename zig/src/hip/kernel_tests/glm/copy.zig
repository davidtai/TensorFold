//! The "glm" group's bf16 row copy (glm/glm.hip) on the GPU: every 16-bit word as stored, rows at new strides, the
//! gaps between rows untouched. No float path: the words a conversion would change are in every case.
const std = @import("std");
const rig = @import("../rig.zig");
const kernels = @import("../../kernels.zig");
const launch = @import("../../launch.zig");

const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;

/// Words that a float path would change: NaN payloads, both zeros, subnormals, infinities, the extremes.
const awkward = [_]u16{ 0x7fc1, 0xffff, 0x7f81, 0x8000, 0x0000, 0x0001, 0x807f, 0x7f80, 0xff80, 0x7f7f, 0xff7f };

const Case = struct { rows: usize, cols: usize, src_stride: usize, dst_stride: usize };

const threads = 256;

test "glm bf16 row copy: the words as stored at every stride; the gaps between rows keep their bytes" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    const f = try t.on.mods[@backingInt(kernels.Group.glm)].function("tf_glm_copy_rows_bf16");
    var rng: rig.Rng = .{ .state = 0x61AC };
    const cases = [_]Case{
        .{ .rows = 1, .cols = 1, .src_stride = 1, .dst_stride = 1 },
        .{ .rows = 3, .cols = 1000, .src_stride = 1003, .dst_stride = 1024 },
        .{ .rows = 64, .cols = 4096, .src_stride = 4096, .dst_stride = 4096 },
        .{ .rows = 5, .cols = 300_000, .src_stride = 300_001, .dst_stride = 300_032 }, // past 1,024 column blocks
        .{ .rows = 70_000, .cols = 8, .src_stride = 9, .dst_stride = 64 }, // past 65,535 row blocks
    };
    for (cases) |c| {
        const src_words = (c.rows - 1) * c.src_stride + c.cols;
        const dst_words = (c.rows - 1) * c.dst_stride + c.cols;
        const src = try gpa.alloc(u16, src_words);
        defer gpa.free(src);
        for (src, 0..) |*w, i| w.* = if (i % 13 == 0) awkward[(i / 13) % awkward.len] else @truncate(rng.next());
        var d_src = try t.upload(src);
        defer d_src.free();
        var d_dst = try t.alloc(dst_words * 2);
        defer d_dst.free();
        try d_dst.fill8(0xA5);
        var args: launch.Args = .{};
        try args.add(at(d_src));
        try args.add(at(d_dst));
        try args.add(@as(c_int, @intCast(c.rows)));
        try args.add(@as(c_int, @intCast(c.cols)));
        try args.add(@as(i64, @intCast(c.src_stride)));
        try args.add(@as(i64, @intCast(c.dst_stride)));
        const grid: launch.Dim3 = .{ .x = @intCast(@min((c.cols + threads - 1) / threads, 1024)), .y = @intCast(@min(c.rows, 65535)) };
        try launch.launch(f, .{ .grid = grid, .block = .{ .x = threads } }, t.stream, &args);
        try t.stream.synchronize();
        const got = try rig.download(u16, d_dst);
        defer gpa.free(got);
        const want = try gpa.alloc(u16, dst_words);
        defer gpa.free(want);
        @memset(want, 0xA5A5);
        for (0..c.rows) |row| @memcpy(want[row * c.dst_stride ..][0..c.cols], src[row * c.src_stride ..][0..c.cols]);
        if (std.mem.findDiff(u16, want, got)) |i| {
            std.debug.print("copy {any}: word {d} is 0x{x:0>4}, want 0x{x:0>4}\n", .{ c, i, got[i], want[i] });
            return error.TestUnexpectedResult;
        }
    }
    std.debug.print("  glm bf16 row copy: {d} cases, every word as stored, the gaps untouched\n", .{cases.len});
}
