//! GLM-5.3-Flash's mHC (4 streams of 4,096, Sinkhorn 20), stream mean, residual, norms and embedding against float64.
const std = @import("std");
const rig = @import("../rig.zig");
const check = @import("check.zig");
const glm = @import("../../launches/glm.zig");
const DeviceBuffer = @import("../../memory.zig").DeviceBuffer;

const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;
const D = glm.hidden;
const S = glm.streams;
const W = S * D;
const mixes = 24;
const eps: f32 = 1e-5;
const hc_eps: f32 = 1e-6;
const iters = 20;

fn p(b: DeviceBuffer) ?*anyopaque {
    return @ptrFromInt(at(b));
}

fn sig(v: f64) f64 {
    return 1 / (1 + @exp(-v));
}

/// hc pre of one row in float64 (the model's arithmetic, no rounding but the inputs'), with each output's tolerance.
const Pre = struct {
    pre: [S]f64,
    pre_tol: [S]f64,
    post: [S]f64,
    post_tol: [S]f64,
    comb: [S * S]f64,
    comb_tol: f64,
    out: [D]f64,
    out_tol: [D]f64,

    fn of(x: []const u16, fnw: []const u16, base: []const f32, scale: [3]f32, nw: []const u16) Pre {
        var r: Pre = undefined;
        var ss: f64 = 0;
        for (x) |v| ss += check.val(v) * check.val(v);
        const rinv = 1 / @sqrt(eps + ss / W);
        var mix: [mixes]f64 = undefined;
        var mtol: [mixes]f64 = undefined;
        for (0..mixes) |m| {
            var sum: f64 = 0;
            var abs: f64 = 0;
            for (x, fnw[m * W ..][0..W]) |a, b| {
                const term = check.val(a) * check.val(b);
                sum += term;
                abs += @abs(term);
            }
            mix[m] = sum * rinv;
            // fp32: a thread's 4-term chain, 5 butterfly levels, 8 waves and 16 blocks in order; rinv to 2^-20
            mtol[m] = 64 * std.math.pow(f64, 2, -24) * abs * rinv + @abs(mix[m]) * std.math.pow(f64, 2, -20);
        }
        const tiny = std.math.pow(f64, 2, -20);
        for (0..S) |k| {
            r.pre[k] = hc_eps + sig(mix[k] * scale[0] + base[k]);
            r.pre_tol[k] = 0.25 * @abs(scale[0]) * mtol[k] + tiny;
            r.post[k] = 2 * sig(mix[S + k] * scale[1] + base[S + k]);
            r.post_tol[k] = 0.5 * @abs(scale[1]) * mtol[S + k] + tiny;
        }
        var cm: [S][S]f64 = undefined;
        var ctol: f64 = 0;
        for (0..S) |i| {
            var mx: f64 = -std.math.inf(f64);
            for (0..S) |j| {
                cm[i][j] = mix[2 * S + i * S + j] * scale[2] + base[2 * S + i * S + j];
                ctol = @max(ctol, @abs(scale[2]) * mtol[2 * S + i * S + j]);
                mx = @max(mx, cm[i][j]);
            }
            var sum: f64 = 0;
            for (0..S) |j| {
                cm[i][j] = @exp(cm[i][j] - mx);
                sum += cm[i][j];
            }
            for (0..S) |j| cm[i][j] = cm[i][j] / sum + hc_eps;
        }
        for (0..iters) |it| {
            if (it > 0) for (0..S) |i| {
                const sum = cm[i][0] + cm[i][1] + cm[i][2] + cm[i][3] + hc_eps;
                for (0..S) |j| cm[i][j] /= sum;
            };
            for (0..S) |j| {
                const sum = cm[0][j] + cm[1][j] + cm[2][j] + cm[3][j] + hc_eps;
                for (0..S) |i| cm[i][j] /= sum;
            }
        }
        for (0..S) |i| for (0..S) |j| {
            r.comb[i * S + j] = cm[i][j];
        };
        // a logit's error moves a Sinkhorn entry by at most about its own size (4 x for the normalizations), plus the
        // 20 fp32 iterations' rounding
        r.comb_tol = 4 * ctol + std.math.pow(f64, 2, -16);
        var c: [D]f64 = undefined;
        var cs: f64 = 0;
        for (0..D) |d| {
            c[d] = 0;
            for (0..S) |k| c[d] += r.pre[k] * check.val(x[k * D + d]);
            cs += c[d] * c[d];
        }
        const rinv2 = 1 / @sqrt(eps + cs / D);
        for (0..D) |d| {
            const w = check.val(nw[d]);
            r.out[d] = w * c[d] * rinv2;
            var dp: f64 = 0;
            for (0..S) |k| dp += r.pre_tol[k] * @abs(check.val(x[k * D + d]));
            // three bf16 roundings (c, c / rms, the output), each up to u = 2^-8 relative, and the pre weights' error,
            // twice for the rms it moves
            r.out_tol[d] = @abs(r.out[d]) * 3 * std.math.pow(f64, 2, -8) + 2 * @abs(w) * rinv2 * dp + 1e-30;
        }
        return r;
    }
};

test "glm hc pre: the 24 mixing dots, Sinkhorn (20) and the collapsed RMS-normed row within float64's bounds, 1 to 70 rows" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x6A11_5E0F };
    const rows = 70;
    const hx = try check.normals(&rng, rows * W, 1.0);
    defer gpa.free(hx);
    const hfn = try check.normals(&rng, mixes * W, 0.02);
    defer gpa.free(hfn);
    var base: [mixes]f32 = undefined;
    for (&base) |*b| b.* = rng.unit() * 0.5;
    const scale = [3]f32{ 0.7, 0.9, 1.1 };
    const hnw = try check.normals(&rng, D, 1.0);
    defer gpa.free(hnw);
    var x = try t.upload(hx);
    defer x.free();
    var fnw = try t.upload(hfn);
    defer fnw.free();
    var db = try t.upload(&base);
    defer db.free();
    var dsc = try t.upload(&scale);
    defer dsc.free();
    var nw = try t.upload(hnw);
    defer nw.free();
    var out = try t.alloc(rows * D * 2);
    defer out.free();
    var post = try t.alloc(rows * S * 4);
    defer post.free();
    var comb = try t.alloc(rows * S * S * 4);
    defer comb.free();
    var part = try t.alloc(glm.hcPartBytes(rows));
    defer part.free();
    const refs = try gpa.alloc(Pre, rows);
    defer gpa.free(refs);
    for (refs, 0..) |*r, i| r.* = Pre.of(hx[i * W ..][0..W], hfn, &base, scale, hnw);
    for ([_]usize{ 1, 5, rows }) |m| {
        try glm.hcPre(&t.on, p(x), p(fnw), p(db), p(dsc), p(nw), p(out), p(post), p(comb), p(part), eps, hc_eps, iters, @intCast(m), t.stream.handle);
        try t.stream.synchronize();
        const go = try rig.download(u16, out);
        defer gpa.free(go);
        const gp = try rig.download(f32, post);
        defer gpa.free(gp);
        const gc = try rig.download(f32, comb);
        defer gpa.free(gc);
        var name: [3][64]u8 = undefined;
        var wo: check.Worst = .{ .name = try std.fmt.bufPrint(&name[0], "hc pre {d} rows: normed collapse", .{m}) };
        var wp: check.Worst = .{ .name = try std.fmt.bufPrint(&name[1], "hc pre {d} rows: post", .{m}) };
        var wc: check.Worst = .{ .name = try std.fmt.bufPrint(&name[2], "hc pre {d} rows: Sinkhorn comb", .{m}) };
        for (0..m) |r| {
            const ref = &refs[r];
            for (0..D) |d| wo.add(r * D + d, check.val(go[r * D + d]), ref.out[d], ref.out_tol[d]);
            for (0..S) |k| wp.add(r * S + k, gp[r * S + k], ref.post[k], ref.post_tol[k]);
            for (0..S * S) |k| wc.add(r * S * S + k, gc[r * S * S + k], ref.comb[k], ref.comb_tol);
            // Sinkhorn ends on a column normalization (as glm_glue_hc.cuh's hc_finish): each column (the sum over i of
            // comb[i][j]) is S / (S + hc_eps), 1 within 1e-5. Rows keep the residual of 20 iterations, so each row sum
            // must equal the float64 reference's row sum (same iterations), not 1.
            for (0..S) |i| {
                var rs: f64 = 0;
                var rs_ref: f64 = 0;
                var cs: f64 = 0;
                for (0..S) |j| {
                    rs += gc[r * S * S + i * S + j];
                    rs_ref += ref.comb[i * S + j];
                    cs += gc[r * S * S + j * S + i];
                }
                try rig.expect(@abs(cs - 1) < 1e-5, "hc comb row {d} column {d}: sum {d}, not 1 within 1e-5", .{ r, i, cs });
                try rig.expect(@abs(rs - rs_ref) <= 4 * ref.comb_tol, "hc comb row {d} row {d}: sum {d}, float64 {d}", .{ r, i, rs, rs_ref });
            }
        }
        try wo.done();
        try wp.done();
        try wc.done();
    }
}

test "glm hc pre: a row's normed collapse, post and comb have the same bits at any row count and any place in the chunk" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x6A11_B175 };
    const rows = 70;
    const hx = try check.normals(&rng, rows * W, 1.0);
    defer gpa.free(hx);
    const hfn = try check.normals(&rng, mixes * W, 0.02);
    defer gpa.free(hfn);
    var base: [mixes]f32 = undefined;
    for (&base) |*b| b.* = rng.unit() * 0.5;
    const scale = [3]f32{ 0.7, 0.9, 1.1 };
    const hnw = try check.normals(&rng, D, 1.0);
    defer gpa.free(hnw);
    var x = try t.upload(hx);
    defer x.free();
    var fnw = try t.upload(hfn);
    defer fnw.free();
    var db = try t.upload(&base);
    defer db.free();
    var dsc = try t.upload(&scale);
    defer dsc.free();
    var nw = try t.upload(hnw);
    defer nw.free();
    var out = try t.alloc(rows * D * 2);
    defer out.free();
    var post = try t.alloc(rows * S * 4);
    defer post.free();
    var comb = try t.alloc(rows * S * S * 4);
    defer comb.free();
    var part = try t.alloc(glm.hcPartBytes(rows));
    defer part.free();
    var full: [3][]u8 = undefined;
    const bytes = [3]usize{ D * 2, S * 4, S * S * 4 };
    const bufs = [3]DeviceBuffer{ out, post, comb };
    try glm.hcPre(&t.on, p(x), p(fnw), p(db), p(dsc), p(nw), p(out), p(post), p(comb), p(part), eps, hc_eps, iters, rows, t.stream.handle);
    try t.stream.synchronize();
    for (&full, bufs, bytes) |*f, b, n| {
        f.* = try gpa.alloc(u8, rows * n);
        try b.download(0, f.*);
    }
    defer for (full) |f| gpa.free(f);
    for ([_][2]usize{ .{ 0, 1 }, .{ 0, 5 }, .{ 3, 1 }, .{ 5, 7 }, .{ 63, 7 }, .{ 1, 64 } }) |c| {
        try glm.hcPre(&t.on, @ptrFromInt(at(x) + c[0] * W * 2), p(fnw), p(db), p(dsc), p(nw), p(out), p(post), p(comb), p(part), eps, hc_eps, iters, @intCast(c[1]), t.stream.handle);
        try t.stream.synchronize();
        for (full, bufs, bytes, [_][]const u8{ "normed", "post", "comb" }) |f, b, n, what| {
            const g = try gpa.alloc(u8, c[1] * n);
            defer gpa.free(g);
            try b.download(0, g);
            try rig.expect(std.mem.eql(u8, g, f[c[0] * n ..][0 .. c[1] * n]), "hc pre rows {d}..{d}: {s} differs from the 70-row call", .{ c[0], c[0] + c[1], what });
        }
    }
    std.debug.print("RESULT glm hc pre rows: 6 cuts of 70 rows, normed, post and comb bits equal\n", .{});
}

test "glm hc post: branch (2 partials) into 4 streams through post and comb within float64's bound, in place" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x9057_0001 };
    const rows = 9;
    const hx = try check.normals(&rng, rows * W, 1.0);
    defer gpa.free(hx);
    const hg = try gpa.alloc(f32, 2 * rows * D);
    defer gpa.free(hg);
    for (hg) |*v| v.* = @floatCast(rng.normal() * 0.5);
    var hpost: [rows * S]f32 = undefined;
    for (&hpost) |*v| v.* = @floatCast(2 * sig(rng.normal()));
    var hcomb: [rows * S * S]f32 = undefined;
    for (&hcomb) |*v| v.* = @floatCast(0.25 + rng.unit() * 0.2);
    var x = try t.upload(hx);
    defer x.free();
    var g = try t.upload(hg);
    defer g.free();
    var post = try t.upload(&hpost);
    defer post.free();
    var comb = try t.upload(&hcomb);
    defer comb.free();
    try glm.hcPost(&t.on, p(x), p(x), p(g), 2, rows * D, p(post), p(comb), rows, t.stream.handle);
    try t.stream.synchronize();
    const got = try rig.download(u16, x);
    defer gpa.free(got);
    var w: check.Worst = .{ .name = "hc post 9 rows, in place" };
    for (0..rows) |r| for (0..S) |s| for (0..D) |d| {
        const branch = @as(f64, hg[r * D + d]) + hg[rows * D + r * D + d];
        const pb = hpost[r * S + s] * branch;
        var mixed: f64 = 0;
        var mag: f64 = @abs(pb);
        for (0..S) |tt| {
            const term = hcomb[r * S * S + tt * S + s] * check.val(hx[r * W + tt * D + d]);
            mixed += term;
            mag += @abs(term);
        }
        const want = pb + mixed;
        // two bf16 roundings (the branch, the output) of u = 2^-8 each, and fp32
        const tol = (@abs(pb) + @abs(want)) * std.math.pow(f64, 2, -8) + mag * std.math.pow(f64, 2, -21);
        const i = r * W + s * D + d;
        w.add(i, check.val(got[i]), want, tol);
    };
    try w.done();
}

test "glm stream mean, residual (1 and 2 partials), embedding into streams: within float64's bound, embedding exact" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x5EA4_0002 };
    const rows = 6;
    {
        const hx = try check.normals(&rng, rows * W, 1.0);
        defer gpa.free(hx);
        var x = try t.upload(hx);
        defer x.free();
        var out = try t.alloc(rows * D * 2);
        defer out.free();
        try glm.streamMean(&t.on, p(x), p(out), rows, t.stream.handle);
        try t.stream.synchronize();
        const got = try rig.download(u16, out);
        defer gpa.free(got);
        var w: check.Worst = .{ .name = "stream mean 6 rows" };
        for (0..rows) |r| for (0..D) |d| {
            var sum: f64 = 0;
            var mag: f64 = 0;
            for (0..S) |s| {
                sum += check.val(hx[r * W + s * D + d]);
                mag += @abs(check.val(hx[r * W + s * D + d]));
            }
            w.add(r * D + d, check.val(got[r * D + d]), sum / S, @abs(sum / S) * std.math.pow(f64, 2, -8) + mag * std.math.pow(f64, 2, -22));
        };
        try w.done();
    }
    for ([_]usize{ 4096, 1000 }) |d| for ([_]c_int{ 1, 2 }) |parts| {
        const hx = try check.normals(&rng, rows * d, 1.0);
        defer gpa.free(hx);
        const hg = try gpa.alloc(f32, 2 * rows * d);
        defer gpa.free(hg);
        for (hg) |*v| v.* = @floatCast(rng.normal() * 0.3);
        var x = try t.upload(hx);
        defer x.free();
        var g = try t.upload(hg);
        defer g.free();
        try glm.residual(&t.on, p(x), p(x), p(g), parts, @intCast(rows * d), rows, @intCast(d), t.stream.handle);
        try t.stream.synchronize();
        const got = try rig.download(u16, x);
        defer gpa.free(got);
        var name: [64]u8 = undefined;
        var w: check.Worst = .{ .name = try std.fmt.bufPrint(&name, "residual {d} x {d}, {d} partials, in place", .{ rows, d, parts }) };
        for (0..rows * d) |i| {
            var gs: f64 = hg[i];
            if (parts == 2) gs += hg[rows * d + i];
            const want = check.val(hx[i]) + gs;
            w.add(i, check.val(got[i]), want, (@abs(gs) + @abs(want)) * std.math.pow(f64, 2, -8) + @abs(gs) * std.math.pow(f64, 2, -22));
        }
        try w.done();
    };
    {
        const vocab = 1000;
        const htab = try check.normals(&rng, vocab * D, 1.0);
        defer gpa.free(htab);
        var hid: [rows]i32 = undefined;
        for (&hid) |*v| v.* = @intCast(rng.next() % vocab);
        hid[0] = vocab - 1;
        var tab = try t.upload(htab);
        defer tab.free();
        var ids = try t.upload(&hid);
        defer ids.free();
        var out = try t.alloc(rows * W * 2);
        defer out.free();
        for ([_]c_int{ 4, 1 }) |copies| {
            try glm.embed(&t.on, p(ids), p(tab), p(out), rows, D, copies, t.stream.handle);
            try t.stream.synchronize();
            const got = try rig.download(u16, out);
            defer gpa.free(got);
            const cu: usize = @intCast(copies);
            for (0..rows) |r| for (0..cu) |k| {
                const id: usize = @intCast(hid[r]);
                try rig.expect(std.mem.eql(u16, got[(r * cu + k) * D ..][0..D], htab[id * D ..][0..D]), "embed row {d} copy {d}", .{ r, k });
            };
        }
        std.debug.print("RESULT glm embed: 6 rows into 4 streams and 1, the table's rows exactly\n", .{});
    }
}

test "glm rms: widths 4096, 1536 and 512, bf16 weights, strided rows (the MTP layer's [e | h] halves) within float64's bound" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x4E05_0003 };
    const rows = 5;
    for ([_]usize{ 4096, 1536, 512 }) |width| {
        const xs = width + 64; // the input rows strided too
        const os = 2 * width;
        const hx = try check.normals(&rng, rows * xs, 1.5);
        defer gpa.free(hx);
        const hw = try check.normals(&rng, width, 1.0);
        defer gpa.free(hw);
        var x = try t.upload(hx);
        defer x.free();
        var w = try t.upload(hw);
        defer w.free();
        var out = try t.alloc(rows * os * 2);
        defer out.free();
        try out.fill8(0x7f);
        // the second half of each output row: the first half keeps its fill
        try glm.rms(&t.on, p(x), @intCast(xs), p(w), @ptrFromInt(at(out) + width * 2), @intCast(os), rows, @intCast(width), eps, t.stream.handle);
        try t.stream.synchronize();
        const got = try rig.download(u16, out);
        defer gpa.free(got);
        var name: [64]u8 = undefined;
        var wst: check.Worst = .{ .name = try std.fmt.bufPrint(&name, "rms {d} x {d}, strided", .{ rows, width }) };
        for (0..rows) |r| {
            const row = hx[r * xs ..][0..width];
            var ss: f64 = 0;
            for (row) |v| ss += check.val(v) * check.val(v);
            const rinv = 1 / @sqrt(eps + ss / @as(f64, @floatFromInt(width)));
            for (0..width) |i| {
                const want = check.val(hw[i]) * check.val(row[i]) * rinv;
                // two bf16 roundings (x / rms, the output) of u = 2^-8 each
                wst.add(r * width + i, check.val(got[r * os + width + i]), want, @abs(want) * std.math.pow(f64, 2, -7) + 1e-30);
                try rig.expect(got[r * os + i] == 0x7f7f, "rms {d}: row {d} column {d} outside the output was written", .{ width, r, i });
            }
        }
        try wst.done();
    }
}

const TimedHc = struct {
    t: *Rig,
    b: [9]u64,
    rows: c_int,

    fn call(c: TimedHc) anyerror!void {
        const b = c.b;
        try glm.hcPre(&c.t.on, @ptrFromInt(b[0]), @ptrFromInt(b[1]), @ptrFromInt(b[2]), @ptrFromInt(b[3]), @ptrFromInt(b[4]), @ptrFromInt(b[5]), @ptrFromInt(b[6]), @ptrFromInt(b[7]), @ptrFromInt(b[8]), eps, hc_eps, iters, c.rows, c.t.stream.handle);
    }

    fn post(c: TimedHc) anyerror!void {
        const b = c.b;
        try glm.hcPost(&c.t.on, @ptrFromInt(b[0]), @ptrFromInt(b[0]), @ptrFromInt(b[7]), 1, 0, @ptrFromInt(b[6]), @ptrFromInt(b[7]), c.rows, c.t.stream.handle);
    }
};

test "glm hc: speed facts (no gate): hc pre and hc post at 1, 4 and 2048 rows" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    const most = 2048;
    var bufs: [9]DeviceBuffer = undefined;
    const sizes = [9]usize{ most * W * 2, mixes * W * 2, mixes * 4, 16, D * 2, most * D * 2, most * S * 4, most * D * 4, glm.hcPartBytes(most) };
    for (&bufs, sizes) |*b, n| {
        b.* = try t.alloc(n);
        try b.fill8(0x3c);
    }
    defer for (&bufs) |*b| b.free();
    var addr: [9]u64 = undefined;
    for (bufs, 0..) |b, i| addr[i] = at(b);
    for ([_]c_int{ 1, 4, most }) |rows| {
        const c: TimedHc = .{ .t = t, .b = addr, .rows = rows };
        const pre = try check.timeUs(t, c, TimedHc.call, 100);
        const post = try check.timeUs(t, c, TimedHc.post, 100);
        std.debug.print("SPEED glm hc {d} rows: pre {d:.1} us, post {d:.1} us\n", .{ rows, pre, post });
    }
}
