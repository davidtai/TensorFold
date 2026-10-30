//! GLM's SwiGLU with CUDA's rounding points (glm_glue_moe.cuh swiglu1): the prompt kernel and the decode pair epilogue,
//! each bit-equal to a host emulation of the sequence: g, u to bf16; g = min(g, 10); u clamped; s = bf16(silu(g));
//! out = bf16(u s).
const std = @import("std");
const rig = @import("../rig.zig");
const check = @import("check.zig");
const affine = @import("../../launches/affine.zig");
const glm = @import("../../launches/glm.zig");
const DeviceBuffer = @import("../../memory.zig").DeviceBuffer;

const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;
const K = affine.Kernels;
const limit = glm.swiglu_limit;

/// bf16 of an fp32 to nearest even, a NaN 0x7fc0 (the kernels' float_to_bf16 and narrow).
fn bf16Of32(f: f32) u16 {
    if (std.math.isNan(f)) return 0x7fc0;
    const u: u32 = @bitCast(f);
    return @intCast((u + (((u >> 16) & 1) + 0x7fff)) >> 16);
}

/// bf16 of a double in one rounding: round to odd onto fp32, then to nearest even (glue.hpp bf16_of).
fn bf16Of64(v: f64) u16 {
    if (std.math.isNan(v)) return 0x7fc0;
    var f: f32 = @floatCast(v);
    if (@as(f64, f) != v) {
        var u: u32 = @bitCast(f);
        if (@abs(@as(f64, f)) > @abs(v)) u -= 1;
        f = @bitCast(u | 1);
    }
    return bf16Of32(f);
}

/// C's fminf / fmaxf: a NaN operand yields the other.
fn fmin(a: f32, b: f32) f32 {
    if (std.math.isNan(a)) return b;
    if (std.math.isNan(b)) return a;
    return @min(a, b);
}
fn fmax(a: f32, b: f32) f32 {
    if (std.math.isNan(a)) return b;
    if (std.math.isNan(b)) return a;
    return @max(a, b);
}

/// bf16(silu(min(g, limit))) for a bf16 g, in double.
fn siluOf(g: u16) u16 {
    const gd: f64 = fmin(@floatCast(check.val(g)), limit);
    return bf16Of64(gd / (1.0 + @exp(-gd)));
}

/// The output of CUDA's sequence for the bf16 gate and up.
fn emulate(g: u16, u: u16) u16 {
    const s: f32 = @floatCast(check.val(siluOf(g)));
    const uc = fmin(fmax(@floatCast(check.val(u)), -limit), limit);
    return bf16Of32(uc * s);
}

fn p(b: DeviceBuffer) ?*anyopaque {
    return @ptrFromInt(at(b));
}

test "glm swiglu (CUDA rounding points), prompt kernel: every bf16 gate against 4 up rows, bit-equal to the host emulation" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x5A16_0001 };
    const w = 1 << 16;
    const rows = 4;
    const hgu = try gpa.alloc(u16, rows * 2 * w);
    defer gpa.free(hgu);
    for (0..rows) |r| for (0..w) |c| {
        hgu[r * 2 * w + c] = @intCast(c); // the gate: every bf16 pattern, NaNs and infinities included
        hgu[r * 2 * w + w + c] = switch (r) {
            0 => 0x3f80, // u = 1: out is the silu table's entry
            1 => @truncate(rng.next()), // any pattern
            2 => check.rnd(rng.normal() * 12), // around the clamp
            else => @intCast((c * 7919 + 13) & 0xffff),
        };
    };
    var gu = try t.upload(hgu);
    defer gu.free();
    var act = try t.alloc(rows * w * 2);
    defer act.free();
    try glm.swiglu(&t.on, p(gu), p(act), rows, w, limit, t.stream.handle);
    try t.stream.synchronize();
    const got = try rig.download(u16, act);
    defer gpa.free(got);
    var bad: usize = 0;
    for (0..rows) |r| for (0..w) |c| {
        const want = emulate(hgu[r * 2 * w + c], hgu[r * 2 * w + w + c]);
        if (got[r * w + c] != want) {
            if (bad < 8) std.debug.print("glm swiglu prompt row {d} g 0x{x:0>4} u 0x{x:0>4}: 0x{x:0>4}, emulation 0x{x:0>4}\n", .{ r, hgu[r * 2 * w + c], hgu[r * 2 * w + w + c], got[r * w + c], want });
            bad += 1;
        }
    };
    std.debug.print("RESULT glm swiglu prompt (CUDA points): {d} of {d} outputs bit-equal to the emulation\n", .{ rows * w - bad, rows * w });
    try rig.expect(bad == 0, "glm swiglu prompt: {d} outputs differ", .{bad});
}

/// x and w on a grid whose products and partial sums are exact in IEEE fp32 (x in {-3..3} / 2, w in {-3..3} / 16: a
/// product is n / 32, |n| <= 9). The tiles' fp32 sums still need not be exact: on gfx1151 an exact zero up-sum of the
/// stream tile came out 2^-23 (run e3c131c6). So the expected bits come from the tile's own fp32 sums of the same
/// product (the plain fp32 tile has the pair's sum order: the row and column tests above), each checked against the
/// exact float64 sum first.
fn grid(rng: *rig.Rng, n: usize, den: f64) ![]u16 {
    const out = try gpa.alloc(u16, n);
    for (out) |*v| v.* = check.rnd(@as(f64, @floatFromInt(@as(i64, @intCast(rng.next() % 7)) - 3)) / den);
    return out;
}

test "glm swiglu (CUDA rounding points), decode pair epilogue and the prompt's WMMA + kernel: bit-equal to the emulation of the tile's own sums" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    if (!t.bf16 or t.on.affine.bf16_glu[0] == null or t.on.affine.bf16_wmma[0] == null) return error.SkipZigTest;
    var rng: rig.Rng = .{ .state = 0x5A16_0002 };
    const k = 4096;
    for ([_]usize{ 2048, 12288 }) |cols| {
        const m = 8;
        const n = 2 * cols;
        const hx = try grid(&rng, m * k, 2);
        defer gpa.free(hx);
        const hw = try grid(&rng, n * k, 16);
        defer gpa.free(hw);
        const exact = try check.Dots.of(hx, hw, m, n, k);
        defer exact.free();
        var x = try t.upload(hx);
        defer x.free();
        var w = try t.upload(hw);
        defer w.free();
        var act = try t.alloc(m * cols * 2);
        defer act.free();
        var gu = try t.alloc(m * n * 2);
        defer gu.free();
        var sums = try t.alloc(m * n * 4);
        defer sums.free();
        var inexact: [2]usize = .{ 0, 0 };
        for ([_]usize{ 1, 2, 3, 4, 5, 8 }) |mm| for ([_]bool{ false, true }) |prompt| {
            const what = if (prompt) "wmma + kernel" else "decode pair";
            // the tile's fp32 sums of (gate | up): the stream tile for decode, the WMMA tile for a prompt
            try t.on.affine.bf16Go(&t.r, K.bf16Arg(at(x), at(w), at(sums), 0, @intCast(mm), @intCast(n), k), prompt, t.stream.handle);
            try act.fill8(0);
            if (prompt) {
                try t.on.affine.bf16Go(&t.r, K.bf16Arg(at(x), at(w), 0, at(gu), @intCast(mm), @intCast(n), k), true, t.stream.handle);
                try glm.swiglu(&t.on, p(gu), p(act), @intCast(mm), @intCast(cols), limit, t.stream.handle);
            } else {
                try t.on.affine.bf16GluGo(&t.r, at(x), at(w), at(act), t.on.glm.silu, @intCast(mm), @intCast(cols), k, limit, t.stream.handle);
            }
            try t.stream.synchronize();
            const fs = try rig.download(f32, sums);
            defer gpa.free(fs);
            const got = try rig.download(u16, act);
            defer gpa.free(got);
            var bad: usize = 0;
            for (0..mm) |r| for (0..cols) |j| {
                const gi = r * n + j;
                const ui = gi + cols;
                for ([_]usize{ gi, ui }) |idx| {
                    try rig.expect(@abs(fs[idx] - exact.want[idx]) <= exact.tol[idx], "glm swiglu {s} [{d}] {d} rows: sum {d} is {e}, exact {e}", .{ what, cols, mm, idx, fs[idx], exact.want[idx] });
                    if (fs[idx] != exact.want[idx]) inexact[@intFromBool(prompt)] += 1;
                }
                const want = emulate(bf16Of32(fs[gi]), bf16Of32(fs[ui]));
                if (got[r * cols + j] != want) {
                    if (bad < 4) std.debug.print("glm swiglu {s} [{d}] {d} rows: ({d}, {d}) is 0x{x:0>4}, emulation 0x{x:0>4} (sums {e} {e}, exact {e} {e})\n", .{ what, cols, mm, r, j, got[r * cols + j], want, fs[gi], fs[ui], exact.want[gi], exact.want[ui] });
                    bad += 1;
                }
            };
            try rig.expect(bad == 0, "glm swiglu {s} [{d}] {d} rows: {d} outputs differ from the emulation", .{ what, cols, mm, bad });
        };
        std.debug.print("RESULT glm swiglu (CUDA points) [2 x {d}, 4096]: decode pair and wmma + kernel at 1-8 rows bit-equal to the emulation of the tiles' sums; fp32 sums not exact: stream {d}, wmma {d}\n", .{ cols, inexact[0], inexact[1] });
    }
}
