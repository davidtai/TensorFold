//! G4-4c sweep (test-only, not for main): the grouped BF16 prompt tile at load depth 1 and 2 and group sizes of 2 to 64
//! row tiles, at GLM's grouped prompt products, 8,192 and 2,048 rows. Each variant's bits are checked against the
//! row-major tile at a small shape first; then speed facts (no gate), with the served tile (bf16Go) on the same line.
const std = @import("std");
const rig = @import("../rig.zig");
const check = @import("check.zig");
const affine = @import("../../launches/affine.zig");
const DeviceBuffer = @import("../../memory.zig").DeviceBuffer;

const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;
const K = affine.Kernels;
const G = K.sweep_groups.len;

fn skip(t: *Rig) bool {
    return !t.bf16 or t.on.affine.bf16_wmma[0] == null or t.on.affine.bf16_wmma_sweep[0][0][0] == null;
}

fn words(t: *Rig, out: DeviceBuffer, n: usize) ![]u32 {
    try t.stream.synchronize();
    const got = try gpa.alloc(u32, n);
    try out.download(0, std.mem.sliceAsBytes(got));
    return got;
}

test "glm bf16 sweep (G4-4c): every sweep variant gives the row-major tile's bits, 300 x [320, K] at each swept K" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    if (skip(t)) return error.SkipZigTest;
    var rng: rig.Rng = .{ .state = 0x6474_4c01 };
    const m = 300;
    const n = 320;
    var cases: usize = 0;
    for (K.sweep_ks) |kc| {
        const k: usize = @intCast(kc);
        const hx = try check.normals(&rng, m * k, 1.0);
        defer gpa.free(hx);
        const hw = try check.normals(&rng, n * k, 0.05);
        defer gpa.free(hw);
        var x = try t.upload(hx);
        defer x.free();
        var w = try t.upload(hw);
        defer w.free();
        var out = try t.alloc(m * n * 4);
        defer out.free();
        const a = K.bf16Arg(at(x), at(w), at(out), 0, m, n, kc);
        try t.on.affine.bf16Wmma(&t.r, a, false, t.stream.handle);
        const want = try words(t, out, m * n);
        defer gpa.free(want);
        for (1..3) |ah| for (0..G) |g| {
            try out.fill8(0);
            try t.on.affine.bf16Sweep(&t.r, a, ah, g, t.stream.handle);
            const got = try words(t, out, m * n);
            defer gpa.free(got);
            try rig.expect(std.mem.eql(u32, want, got), "glm bf16 sweep K {d} ahead {d} group {d}: bits differ from the row-major tile's", .{ k, ah, K.sweep_groups[g] });
            cases += 1;
        };
    }
    std.debug.print("RESULT glm bf16 sweep: {d} variants bit-equal to the row-major tile\n", .{cases});
}

const Timed = struct {
    t: *Rig,
    a: affine.Arg,
    ahead: usize, // 0: bf16Go (served)
    g: usize,

    fn call(c: Timed) anyerror!void {
        if (c.ahead == 0) return c.t.on.affine.bf16Go(&c.t.r, c.a, true, c.t.stream.handle);
        try c.t.on.affine.bf16Sweep(&c.t.r, c.a, c.ahead, c.g, c.t.stream.handle);
    }
};

test "glm bf16 sweep (G4-4c): speed facts (no gate) of load depth x group size at the grouped prompt products, 8,192 and 2,048 rows" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    if (skip(t)) return error.SkipZigTest;
    const Shape = struct { name: []const u8, n: usize, k: usize };
    const shapes = [_]Shape{
        .{ .name = "MLA o", .n = 4096, .k = 16384 },
        .{ .name = "q_b | indexer wq_b", .n = 20480, .k = 1536 },
        .{ .name = "KDA in", .n = 24896, .k = 4096 },
        .{ .name = "KDA o", .n = 4096, .k = 8192 },
        .{ .name = "shared gate|up", .n = 4096, .k = 4096 },
        .{ .name = "dense gate|up", .n = 24576, .k = 4096 },
        .{ .name = "dense down", .n = 4096, .k = 12288 },
    };
    const max_m = 8192;
    var x = try t.alloc(max_m * 16384 * 2);
    defer x.free();
    try x.fill8(0x3c);
    var w = try t.alloc(24896 * 4096 * 2);
    defer w.free();
    try w.fill8(0x3c);
    var out = try t.alloc(max_m * 24896 * 2);
    defer out.free();
    for ([_]usize{ max_m, 2048 }) |m| for (shapes) |s| {
        const flops: f64 = 2.0 * @as(f64, @floatFromInt(m * s.n * s.k));
        const a = K.bf16Arg(at(x), at(w), 0, at(out), @intCast(m), @intCast(s.n), @intCast(s.k));
        const calls: usize = if (m > 2048) 4 else 10;
        const tf = struct {
            fn of(fl: f64, us: f64) f64 {
                return fl / us / 1e6;
            }
        };
        const served = try check.timeUs(t, Timed{ .t = t, .a = a, .ahead = 0, .g = 0 }, Timed.call, calls);
        var line: [512]u8 = undefined;
        var at_: usize = 0;
        var best: f64 = 0;
        var best_ah: usize = 0;
        var best_g: usize = 0;
        for (1..3) |ah| for (0..G) |g| {
            const us = try check.timeUs(t, Timed{ .t = t, .a = a, .ahead = ah, .g = g }, Timed.call, calls);
            const v = tf.of(flops, us);
            if (v > best) {
                best = v;
                best_ah = ah;
                best_g = g;
            }
            const piece = try std.fmt.bufPrint(line[at_..], " a{d}g{d} {d:.1}", .{ ah, K.sweep_groups[g], v });
            at_ += piece.len;
        };
        std.debug.print("SPEED glm bf16 sweep {s} [{d}, {d}] {d} rows: served {d:.1} TFLOPS; best a{d}g{d} {d:.1};{s}\n", .{ s.name, s.n, s.k, m, tf.of(flops, served), best_ah, K.sweep_groups[best_g], best, line[0..at_] });
    };
}
