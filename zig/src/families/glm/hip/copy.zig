//! The bf16 row copy (tf_glm_copy_rows_bf16): rows placed at a new stride with their 16-bit words as stored.
const std = @import("std");
const hip = @import("hip");

pub const Copy = struct {
    f: hip.Function,

    /// The threads a block and the most blocks a launch takes in each direction (the kernel strides past them).
    const threads = 256;
    const most_cols_blocks = 1024;
    const most_rows_blocks = 65535;

    pub fn load(m: hip.Module) hip.Error!Copy {
        return .{ .f = try m.function("tf_glm_copy_rows_bf16") };
    }

    /// `rows` rows of `cols` bf16 values from device address `src` (a row every `src_stride` values) to `dst` (a row
    /// every `dst_stride` values), on `s`. Values between `cols` and a stride are not read or written.
    pub fn rows(c: Copy, src: u64, dst: u64, n: usize, cols: usize, src_stride: usize, dst_stride: usize, s: hip.Stream) hip.Error!void {
        if (n == 0 or cols == 0) return;
        if (cols > src_stride or cols > dst_stride or n > std.math.maxInt(c_int) or cols > std.math.maxInt(c_int)) return error.Invalid;
        var args: hip.launch.Args = .{};
        try args.add(src);
        try args.add(dst);
        try args.add(@as(c_int, @intCast(n)));
        try args.add(@as(c_int, @intCast(cols)));
        try args.add(@as(i64, @intCast(src_stride)));
        try args.add(@as(i64, @intCast(dst_stride)));
        const grid: hip.launch.Dim3 = .{
            .x = @intCast(@min((cols + threads - 1) / threads, most_cols_blocks)),
            .y = @intCast(@min(n, most_rows_blocks)),
        };
        try hip.launch.launch(c.f, .{ .grid = grid, .block = .{ .x = threads } }, s, &args);
    }
};
