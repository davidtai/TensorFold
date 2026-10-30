//! Numerics facts (no gate) of the BF16 tiles on model-like data (x ~ N(0, 1), w ~ N(0, 0.05)): the share of bf16
//! outputs that are not the bf16 of the exact (float64) sum, and the mean |error| of the fp32 sums in fp32 ulps.
const std = @import("std");
const rig = @import("../rig.zig");
const check = @import("check.zig");
const affine = @import("../../launches/affine.zig");

const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;
const K = affine.Kernels;

/// The bf16 nearest a double in one rounding (round to odd onto fp32, then nearest even).
fn rn64(v: f64) u16 {
    var f: f32 = @floatCast(v);
    if (@as(f64, f) != v) {
        var u: u32 = @bitCast(f);
        if (@abs(@as(f64, f)) > @abs(v)) u -= 1;
        f = @bitCast(u | 1);
    }
    return check.rnd(f);
}

fn ulp32(v: f64) f64 {
    const f: f32 = @floatCast(@abs(v));
    if (f == 0) return std.math.pow(f64, 2, -149);
    return @as(f64, std.math.nextAfter(f32, f, std.math.inf(f32))) - f;
}

test "glm bf16 numerics facts: bf16 flips and fp32 ulp error of the stream and WMMA tiles at K 1,536 and 4,096" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    if (!t.bf16 or t.on.affine.bf16_wmma[0] == null) return error.SkipZigTest;
    var rng: rig.Rng = .{ .state = 0xF11B_0001 };
    const n = 1024;
    const m = 256;
    for ([_]usize{ 1536, 4096 }) |k| {
        const hx = try check.normals(&rng, m * k, 1.0);
        defer gpa.free(hx);
        const hw = try check.normals(&rng, n * k, 0.05);
        defer gpa.free(hw);
        const d = try check.Dots.of(hx, hw, m, n, k);
        defer d.free();
        var x = try t.upload(hx);
        defer x.free();
        var w = try t.upload(hw);
        defer w.free();
        var out = try t.alloc(m * n * 4);
        defer out.free();
        for ([_]struct { rows: usize, prompt: bool }{ .{ .rows = 1, .prompt = false }, .{ .rows = 4, .prompt = false }, .{ .rows = m, .prompt = true } }) |c| {
            try t.on.affine.bf16Go(&t.r, K.bf16Arg(at(x), at(w), at(out), 0, @intCast(c.rows), n, @intCast(k)), c.prompt, t.stream.handle);
            try t.stream.synchronize();
            const got = try rig.download(f32, out);
            defer gpa.free(got);
            var flips: usize = 0;
            var ulps: f64 = 0;
            const total = c.rows * n;
            for (0..total) |i| {
                if (check.rnd(got[i]) != rn64(d.want[i])) flips += 1;
                ulps += @abs(@as(f64, got[i]) - d.want[i]) / ulp32(d.want[i]);
            }
            std.debug.print("FACT glm bf16 {s} K {d}, {d} rows x {d}: bf16 flips {d:.3} %, fp32 |err| mean {d:.2} ulp\n", .{ if (c.prompt) "wmma" else "stream", k, c.rows, n, 100.0 * @as(f64, @floatFromInt(flips)) / @as(f64, @floatFromInt(total)), ulps / @as(f64, @floatFromInt(total)) });
        }
    }
}
