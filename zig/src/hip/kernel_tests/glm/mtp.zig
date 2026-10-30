//! The MTP layer's own pieces against float64, stage by stage: enorm(embedding) and hnorm(hidden) into the halves of
//! one [e | h] row, eh_proj [4096, 8192] (BF16, decode and prompt tiles), the plain residual, shared_head.norm, the head.
const std = @import("std");
const rig = @import("../rig.zig");
const check = @import("check.zig");
const affine = @import("../../launches/affine.zig");
const glm = @import("../../launches/glm.zig");

const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;
const K = affine.Kernels;
const D = glm.hidden;
const eps: f32 = 1e-5;

fn p(b: anytype) ?*anyopaque {
    return @ptrFromInt(at(b));
}

/// Row r of x (stride xs) normed by w in float64, checked against row r of `got` (stride gs, from column `off`).
fn normRows(w: *check.Worst, x: []const u16, xs: usize, nw: []const u16, got: []const u16, gs: usize, off: usize, rows: usize) void {
    for (0..rows) |r| {
        var ss: f64 = 0;
        for (x[r * xs ..][0..D]) |v| ss += check.val(v) * check.val(v);
        const rinv = 1 / @sqrt(eps + ss / D);
        for (0..D) |i| {
            const want = check.val(nw[i]) * check.val(x[r * xs + i]) * rinv;
            // two bf16 roundings (x / rms, the output) of u = 2^-8 each
            w.add(r * D + i, check.val(got[r * gs + off + i]), want, @abs(want) * std.math.pow(f64, 2, -7) + 1e-30);
        }
    }
}

test "glm mtp: enorm and hnorm into [e | h], eh_proj on both tiles, residual and shared_head.norm, each stage within float64's bound" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    if (!t.bf16 or t.on.affine.bf16_wmma[0] == null) return error.SkipZigTest;
    var rng: rig.Rng = .{ .state = 0x3779_0045 };
    const vocab = 512;
    const most = 33;
    const htab = try check.normals(&rng, vocab * D, 0.02);
    defer gpa.free(htab);
    const hh = try check.normals(&rng, most * D, 1.0);
    defer gpa.free(hh);
    const hen = try check.normals(&rng, D, 1.0);
    defer gpa.free(hen);
    const hhn = try check.normals(&rng, D, 1.0);
    defer gpa.free(hhn);
    const hpn = try check.normals(&rng, D, 1.0);
    defer gpa.free(hpn);
    const heh = try check.normals(&rng, D * 2 * D, 0.02);
    defer gpa.free(heh);
    const hg = try gpa.alloc(f32, most * D);
    defer gpa.free(hg);
    for (hg) |*v| v.* = @floatCast(rng.normal());
    var hid: [most]i32 = undefined;
    for (&hid) |*v| v.* = @intCast(rng.next() % vocab);
    var tab = try t.upload(htab);
    defer tab.free();
    var ids = try t.upload(&hid);
    defer ids.free();
    var h = try t.upload(hh);
    defer h.free();
    var en = try t.upload(hen);
    defer en.free();
    var hn = try t.upload(hhn);
    defer hn.free();
    var pn = try t.upload(hpn);
    defer pn.free();
    var eh = try t.upload(heh);
    defer eh.free();
    var g = try t.upload(hg);
    defer g.free();
    var me = try t.alloc(most * D * 2);
    defer me.free();
    var mcat = try t.alloc(most * 2 * D * 2);
    defer mcat.free();
    var mx = try t.alloc(most * D * 2);
    defer mx.free();
    var normed = try t.alloc(most * D * 2);
    defer normed.free();
    for ([_]struct { n: usize, prompt: bool }{ .{ .n = 1, .prompt = false }, .{ .n = 4, .prompt = false }, .{ .n = most, .prompt = true } }) |c| {
        const n: c_int = @intCast(c.n);
        try glm.embed(&t.on, p(ids), p(tab), p(me), n, D, 1, t.stream.handle);
        try glm.rms(&t.on, p(me), D, p(en), p(mcat), 2 * D, n, D, eps, t.stream.handle);
        try glm.rms(&t.on, p(h), D, p(hn), @ptrFromInt(at(mcat) + D * 2), 2 * D, n, D, eps, t.stream.handle);
        try t.on.affine.bf16Go(&t.r, K.bf16Arg(at(mcat), at(eh), 0, at(mx), n, D, 2 * D), c.prompt, t.stream.handle);
        try t.stream.synchronize();
        const gcat = try rig.download(u16, mcat);
        defer gpa.free(gcat);
        const gx = try rig.download(u16, mx);
        defer gpa.free(gx);
        // stage 1: the halves of [e | h] from the table's rows and the hidden rows
        const erows = try gpa.alloc(u16, c.n * D);
        defer gpa.free(erows);
        for (0..c.n) |r| @memcpy(erows[r * D ..][0..D], htab[@as(usize, @intCast(hid[r])) * D ..][0..D]);
        var name: [4][80]u8 = undefined;
        var we: check.Worst = .{ .name = try std.fmt.bufPrint(&name[0], "mtp enorm {d} rows into [e | h]", .{c.n}) };
        normRows(&we, erows, D, hen, gcat, 2 * D, 0, c.n);
        try we.done();
        var wh: check.Worst = .{ .name = try std.fmt.bufPrint(&name[1], "mtp hnorm {d} rows into [e | h]", .{c.n}) };
        normRows(&wh, hh, D, hhn, gcat, 2 * D, D, c.n);
        try wh.done();
        // stage 2: eh_proj of the GPU's own [e | h] rows, bf16 out
        const ref = try check.Dots.of(gcat[0 .. c.n * 2 * D], heh, c.n, D, 2 * D);
        defer ref.free();
        var wp: check.Worst = .{ .name = try std.fmt.bufPrint(&name[2], "mtp eh_proj {d} x [4096, 8192] {s}", .{ c.n, if (c.prompt) "wmma" else "stream" }) };
        for (0..c.n * D) |i| wp.add(i, check.val(gx[i]), ref.want[i], ref.tol[i] + @abs(ref.want[i]) * std.math.pow(f64, 2, -8));
        try wp.done();
        // stage 3: the plain residual (the DSA block's output stands in as g) and shared_head.norm of the result
        try glm.residual(&t.on, p(mx), p(mx), p(g), 1, 0, n, D, t.stream.handle);
        try glm.rms(&t.on, p(mx), D, p(pn), p(normed), D, n, D, eps, t.stream.handle);
        try t.stream.synchronize();
        const gr = try rig.download(u16, mx);
        defer gpa.free(gr);
        const gn = try rig.download(u16, normed);
        defer gpa.free(gn);
        var wr: check.Worst = .{ .name = try std.fmt.bufPrint(&name[3], "mtp residual and shared_head.norm {d} rows", .{c.n}) };
        for (0..c.n * D) |i| {
            const want = check.val(gx[i]) + hg[i];
            wr.add(i, check.val(gr[i]), want, (@abs(hg[i]) + @abs(want)) * std.math.pow(f64, 2, -8) + @abs(hg[i]) * std.math.pow(f64, 2, -22));
        }
        normRows(&wr, gr, D, hpn, gn, D, 0, c.n);
        try wr.done();
    }
}
