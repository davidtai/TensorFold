//! GLM-5.3-Flash's router (BF16 [288, 4096] logits, sigmoid + correction bias, top-8, normed, x 2.5) and the MoE
//! combine with the shared expert, against float64. The pick and combine are Nemotron-H's kernels (ops/nemotron_moe.hpp).
const std = @import("std");
const rig = @import("../rig.zig");
const check = @import("check.zig");
const affine = @import("../../launches/affine.zig");
const nemo = @import("../../launches/nemotron_moe.zig");

const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;
const K = affine.Kernels;

const experts = 288;
const top_k = 8;
const d = 4096;
const scaling: f32 = 2.5;

fn sig(v: f64) f64 {
    return 1 / (1 + @exp(-v));
}

/// Row `logits`' top_k by sigmoid + bias, ties to the lower id, in float64.
fn ranking(logits: []const f64, bias: []const f32, order: []usize) void {
    for (order, 0..) |*o, e| o.* = e;
    const Ctx = struct { l: []const f64, b: []const f32 };
    std.mem.sort(usize, order, Ctx{ .l = logits, .b = bias }, struct {
        fn lt(c: Ctx, a: usize, b: usize) bool {
            const ca = sig(c.l[a]) + c.b[a];
            const cb = sig(c.l[b]) + c.b[b];
            return ca > cb or (ca == cb and a < b);
        }
    }.lt);
}

test "glm router: logits on both tiles within float64's bound; top-8 of 288 by sigmoid + bias as float64 ranks (near ties aside), weights within 2^-16" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    if (!t.bf16 or t.on.affine.bf16_wmma[0] == null) return error.SkipZigTest;
    var rng: rig.Rng = .{ .state = 0x2007_E288 };
    const m = 300;
    const hx = try check.normals(&rng, m * d, 1.0);
    defer gpa.free(hx);
    const hw = try check.normals(&rng, experts * d, 0.02);
    defer gpa.free(hw);
    const bias = try gpa.alloc(f32, experts);
    defer gpa.free(bias);
    for (bias) |*b| b.* = rng.unit() * 0.05;
    const ref = try check.Dots.of(hx, hw, m, experts, d);
    defer ref.free();
    var x = try t.upload(hx);
    defer x.free();
    var w = try t.upload(hw);
    defer w.free();
    var db = try t.upload(bias);
    defer db.free();
    var logits = try t.alloc(m * experts * 4);
    defer logits.free();
    var pick = try t.alloc(m * top_k * 4);
    defer pick.free();
    var wts = try t.alloc(m * top_k * 4);
    defer wts.free();
    const order = try gpa.alloc(usize, experts);
    defer gpa.free(order);
    const lg64 = try gpa.alloc(f64, experts);
    defer gpa.free(lg64);
    for ([_]struct { m: usize, prompt: bool }{ .{ .m = 1, .prompt = false }, .{ .m = 4, .prompt = false }, .{ .m = m, .prompt = true } }) |c| {
        try t.on.affine.bf16Go(&t.r, K.bf16Arg(at(x), at(w), at(logits), 0, @intCast(c.m), experts, d), c.prompt, t.stream.handle);
        try nemo.select(&t.on, @ptrFromInt(at(logits)), @ptrFromInt(at(db)), @ptrFromInt(at(pick)), @ptrFromInt(at(wts)), null, null, 0, @intCast(c.m), experts, top_k, scaling, true, t.stream.handle);
        try t.stream.synchronize();
        const gl = try rig.download(f32, logits);
        defer gpa.free(gl);
        const gp = try rig.download(i32, pick);
        defer gpa.free(gp);
        const gw = try rig.download(f32, wts);
        defer gpa.free(gw);
        var name: [2][80]u8 = undefined;
        var wl: check.Worst = .{ .name = try std.fmt.bufPrint(&name[0], "router logits {d} x [{d}, {d}] {s}", .{ c.m, experts, d, if (c.prompt) "wmma" else "stream" }) };
        var ww: check.Worst = .{ .name = try std.fmt.bufPrint(&name[1], "router weights {d} rows (top 8, normed, x 2.5)", .{c.m}) };
        var swaps: usize = 0;
        for (0..c.m) |row| {
            for (0..experts) |e| wl.add(row * experts + e, gl[row * experts + e], ref.want[row * experts + e], ref.tol[row * experts + e]);
            // the pick against float64's ranking of the float64 logits: a swap only within the logits' bounds
            for (lg64, 0..) |*v, e| v.* = ref.want[row * experts + e];
            ranking(lg64, bias, order);
            var total: f64 = 0;
            for (0..top_k) |k| total += sig(gl[row * experts + @as(usize, @intCast(gp[row * top_k + k]))]);
            for (0..top_k) |k| {
                const id: usize = @intCast(gp[row * top_k + k]);
                if (id != order[k]) {
                    const a = sig(lg64[id]) + bias[id];
                    const b = sig(lg64[order[k]]) + bias[order[k]];
                    const room = 0.25 * (ref.tol[row * experts + id] + ref.tol[row * experts + order[k]]) + std.math.pow(f64, 2, -20);
                    try rig.expect(@abs(a - b) <= room, "router row {d} pick {d}: {d} where float64 ranks {d} (gap {e}, room {e})", .{ row, k, id, order[k], @abs(a - b), room });
                    swaps += 1;
                }
                // the weight from the GPU's own logits: the pick's arithmetic alone
                const want = sig(gl[row * experts + id]) / total * scaling;
                ww.add(row * top_k + k, gw[row * top_k + k], want, @abs(want) * std.math.pow(f64, 2, -16));
            }
        }
        try wl.done();
        try ww.done();
        std.debug.print("RESULT glm router {d} rows: {d} near-tie swaps against float64's ranking\n", .{ c.m, swaps });
    }
}

test "glm moe combine: 8 routed slots by their weights plus the shared expert's row, fp32, within float64's bound" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0xC0B1_0E08 };
    const rows = 5;
    const y = try gpa.alloc(f32, rows * top_k * d);
    defer gpa.free(y);
    for (y) |*v| v.* = @floatCast(rng.normal());
    const sh = try gpa.alloc(f32, rows * d);
    defer gpa.free(sh);
    for (sh) |*v| v.* = @floatCast(rng.normal());
    var hw: [rows * top_k]f32 = undefined;
    for (&hw) |*v| v.* = @floatCast(rng.uniform() * 0.6);
    var dy = try t.upload(y);
    defer dy.free();
    var dsh = try t.upload(sh);
    defer dsh.free();
    var dw = try t.upload(&hw);
    defer dw.free();
    var out = try t.alloc(rows * d * 4);
    defer out.free();
    try nemo.combine(&t.on, @ptrFromInt(at(dy)), @ptrFromInt(at(dw)), @ptrFromInt(at(dsh)), @ptrFromInt(at(out)), 0, rows, top_k, d, t.stream.handle);
    try t.stream.synchronize();
    const got = try rig.download(f32, out);
    defer gpa.free(got);
    var w: check.Worst = .{ .name = "moe combine 5 rows, 8 slots + shared" };
    for (0..rows) |r| for (0..d) |c| {
        var want: f64 = sh[r * d + c];
        var mag: f64 = @abs(want);
        for (0..top_k) |s| {
            const term = @as(f64, y[(r * top_k + s) * d + c]) * hw[r * top_k + s];
            want += term;
            mag += @abs(term);
        }
        w.add(r * d + c, got[r * d + c], want, mag * 12 * std.math.pow(f64, 2, -24));
    };
    try w.done();
}
