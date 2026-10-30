//! GLM-5.3-Flash's BF16 linears on the stream tile (decode) and the WMMA tile (prompt) against float64.
const std = @import("std");
const rig = @import("../rig.zig");
const check = @import("check.zig");
const affine = @import("../../launches/affine.zig");
const draw = @import("../../launches/draw.zig");
const moe = @import("../../launches/moe.zig");
const DeviceBuffer = @import("../../memory.zig").DeviceBuffer;

const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;
const K = affine.Kernels;

fn skip(t: *Rig) bool {
    return !t.bf16 or t.on.affine.bf16_stream[0][0] == null;
}

/// One product's device buffers: x [m][k] bf16, w [n][k] bf16, out [m][n] fp32 and bf16.
const Product = struct {
    x: DeviceBuffer,
    w: DeviceBuffer,
    out: DeviceBuffer,
    out16: DeviceBuffer,

    fn init(t: *Rig, hx: []const u16, hw: []const u16, m: usize, n: usize) !Product {
        return .{ .x = try t.upload(hx), .w = try t.upload(hw), .out = try t.alloc(m * n * 4), .out16 = try t.alloc(m * n * 2) };
    }

    fn deinit(p: *Product) void {
        p.x.free();
        p.w.free();
        p.out.free();
        p.out16.free();
    }

    /// Rows 0..m of the product, fp32 (or bf16 with `half`), on the stream tile or with `prompt` the WMMA tile.
    fn run(p: *const Product, t: *Rig, m: usize, n: usize, k: usize, half: bool, prompt: bool) !void {
        const a = K.bf16Arg(at(p.x), at(p.w), if (half) 0 else at(p.out), if (half) at(p.out16) else 0, @intCast(m), @intCast(n), @intCast(k));
        try t.on.affine.bf16Go(&t.r, a, prompt, t.stream.handle);
        try t.stream.synchronize();
    }

    /// Rows 0..m of the last run against `d` (the first m rows of its float64 product).
    fn compare(p: *const Product, name: []const u8, d: check.Dots, m: usize, n: usize, half: bool) !void {
        var w: check.Worst = .{ .name = name };
        if (half) {
            const got = try rig.download(u16, p.out16);
            defer gpa.free(got);
            for (0..m * n) |i| w.add(i, check.val(got[i]), d.want[i], d.tol[i] + @abs(d.want[i]) * std.math.pow(f64, 2, -8));
        } else {
            const got = try rig.download(f32, p.out);
            defer gpa.free(got);
            for (0..m * n) |i| w.add(i, got[i], d.want[i], d.tol[i]);
        }
        try w.done();
    }
};

test "glm bf16: the stream tile at every K of the model, 1 to 8 rows, fp32 and bf16 out, within float64's bound" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    if (skip(t)) return error.SkipZigTest;
    var rng: rig.Rng = .{ .state = 0x6C4D_B16A };
    const rows = [_]usize{ 1, 2, 3, 4, 5, 8 };
    for (K.bf16_ks) |kc| for ([_]usize{ 320, 288 }) |n| {
        const k: usize = @intCast(kc);
        const m = rows[rows.len - 1];
        const hx = try check.normals(&rng, m * k, 1.0);
        defer gpa.free(hx);
        const hw = try check.normals(&rng, n * k, 0.05);
        defer gpa.free(hw);
        const d = try check.Dots.of(hx, hw, m, n, k);
        defer d.free();
        var p = try Product.init(t, hx, hw, m, n);
        defer p.deinit();
        for (rows) |mm| for ([_]bool{ false, true }) |half| {
            try p.run(t, mm, n, k, half, false);
            var name: [96]u8 = undefined;
            try p.compare(try std.fmt.bufPrint(&name, "bf16 stream {d} x [{d}, {d}] {s}", .{ mm, n, k, if (half) "bf16" else "fp32" }), d, mm, n, half);
        };
    };
}

test "glm bf16: the WMMA tile at every K of the model, 300 rows x 320 columns, fp32 and bf16 out, within float64's bound" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    if (skip(t) or t.on.affine.bf16_wmma[0] == null) return error.SkipZigTest;
    var rng: rig.Rng = .{ .state = 0x3A77_B16B };
    const m = 300;
    const n = 320;
    for (K.bf16_ks) |kc| {
        const k: usize = @intCast(kc);
        const hx = try check.normals(&rng, m * k, 1.0);
        defer gpa.free(hx);
        const hw = try check.normals(&rng, n * k, 0.05);
        defer gpa.free(hw);
        const d = try check.Dots.of(hx, hw, m, n, k);
        defer d.free();
        var p = try Product.init(t, hx, hw, m, n);
        defer p.deinit();
        for ([_]usize{ 1, 7, m }) |mm| for ([_]bool{ false, true }) |half| {
            try p.run(t, mm, n, k, half, true);
            var name: [96]u8 = undefined;
            try p.compare(try std.fmt.bufPrint(&name, "bf16 wmma {d} x [{d}, {d}] {s}", .{ mm, n, k, if (half) "bf16" else "fp32" }), d, mm, n, half);
        };
    }
}

test "glm bf16: the head [154880, 4096] on both tiles within float64's bound, and tf_argmax's pick is float64's (near ties aside)" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    if (skip(t) or t.on.affine.bf16_wmma[0] == null) return error.SkipZigTest;
    var rng: rig.Rng = .{ .state = 0x4EAD_0001 };
    const n = 154880;
    const k = 4096;
    const m = 3;
    const hx = try check.normals(&rng, m * k, 1.0);
    defer gpa.free(hx);
    const hw = try gpa.alloc(u16, n * k);
    defer gpa.free(hw);
    for (hw) |*v| v.* = check.rnd(rng.unit() * 0.08);
    const d = try check.Dots.of(hx, hw, m, n, k);
    defer d.free();
    var p = try Product.init(t, hx, hw, m, n);
    defer p.deinit();
    var ids = try t.alloc(m * 4);
    defer ids.free();
    for ([_]struct { m: usize, prompt: bool }{ .{ .m = 1, .prompt = false }, .{ .m = 2, .prompt = false }, .{ .m = 3, .prompt = true } }) |c| {
        var name: [96]u8 = undefined;
        try p.run(t, c.m, n, k, true, c.prompt);
        try p.compare(try std.fmt.bufPrint(&name, "head {d} x [{d}, {d}] bf16 on the {s} tile", .{ c.m, n, k, if (c.prompt) "wmma" else "stream" }), d, c.m, n, true);
        try draw.tf_argmax_rows(&t.on, @ptrFromInt(at(p.out16)), 2, @intCast(c.m), n, @ptrFromInt(at(ids)), t.stream.handle);
        try t.stream.synchronize();
        const got = try rig.download(i32, ids);
        defer gpa.free(got);
        for (0..c.m) |row| {
            const want = d.want[row * n ..][0..n];
            const best = std.mem.indexOfMax(f64, want);
            const pick: usize = @intCast(got[row]);
            // a pick other than float64's is a near tie: within the two values' bounds and bf16 rounding
            const gap = want[best] - want[pick];
            const room = d.tol[row * n + best] + d.tol[row * n + pick] + @abs(want[best]) * std.math.pow(f64, 2, -7);
            try rig.expect(pick == best or gap <= room, "head row {d}: argmax {d} where float64 has {d} (gap {e}, room {e})", .{ row, pick, best, gap, room });
            std.debug.print("RESULT glm head argmax {s} row {d}: {d} (float64 {d})\n", .{ if (c.prompt) "wmma" else "stream", row, pick, best });
        }
    }
}

/// silu(min(g, L)) * clamp(u, -L, L) of float64 sums and its tolerance from theirs, plus the bf16 rounding.
fn swiglu(g: f64, u: f64, tg: f64, tu: f64, limit: f64) [2]f64 {
    const gc = @min(g, limit);
    const uc = @min(@max(u, -limit), limit);
    const s = gc / (1 + @exp(-gc));
    const want = s * uc;
    return .{ want, 1.1 * @abs(uc) * tg + @abs(s) * tu + @abs(want) * std.math.pow(f64, 2, -8) + 1e-30 };
}

test "glm bf16: the shared expert's and the dense MLP's (gate | up) with SwiGLU (limit 10), decode and prompt, within float64's bound" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    if (skip(t) or t.on.affine.bf16_wmma[0] == null) return error.SkipZigTest;
    var rng: rig.Rng = .{ .state = 0x5E1F_6A7E };
    const k = 4096;
    const limit: f32 = 10;
    for ([_]usize{ 2048, 12288 }) |cols| {
        const m: usize = if (cols == 2048) 64 else 16; // the prompt's rows (the host's float64 product bounds them)
        const hx = try check.normals(&rng, m * k, 1.0);
        defer gpa.free(hx);
        // gate rows then up rows, as the checkpoint's gate_proj and up_proj stacked; a large scale so some clamp
        const hw = try gpa.alloc(u16, 2 * cols * k);
        defer gpa.free(hw);
        for (hw) |*v| v.* = check.rnd(rng.unit() * 0.17);
        const d = try check.Dots.of(hx, hw, m, 2 * cols, k);
        defer d.free();
        var x = try t.upload(hx);
        defer x.free();
        var w = try t.upload(hw);
        defer w.free();
        var act = try t.alloc(m * cols * 2);
        defer act.free();
        var both = try t.alloc(m * 2 * cols * 4);
        defer both.free();
        var clamped: usize = 0;
        for ([_]usize{ 1, 2, 3, 4, 7, m }) |mm| {
            if (mm > m) continue;
            const prompt = mm == m;
            if (prompt) {
                // a prompt: (gate | up) fp32 on the WMMA tile, then moe_act_kernel's SwiGLU into bf16
                try t.on.affine.bf16Go(&t.r, K.bf16Arg(at(x), at(w), at(both), 0, @intCast(mm), @intCast(2 * cols), k), true, t.stream.handle);
                try moe.tf_moe_act(&t.on, @ptrFromInt(at(both)), @ptrFromInt(at(act)), 2, @intCast(mm), @intCast(cols), limit, t.stream.handle);
            } else {
                try t.on.affine.bf16PairGo(&t.r, at(x), at(w), at(act), @intCast(mm), @intCast(cols), k, limit, t.stream.handle);
            }
            try t.stream.synchronize();
            const got = try rig.download(u16, act);
            defer gpa.free(got);
            var name: [96]u8 = undefined;
            var wst: check.Worst = .{ .name = try std.fmt.bufPrint(&name, "swiglu {d} x [2 x {d}, {d}] {s}", .{ mm, cols, k, if (prompt) "wmma + moe_act" else "stream pair" }) };
            for (0..mm) |row| for (0..cols) |j| {
                const gi = row * 2 * cols + j;
                const ui = gi + cols;
                if (d.want[gi] > limit or @abs(d.want[ui]) > limit) clamped += 1;
                const r = swiglu(d.want[gi], d.want[ui], d.tol[gi], d.tol[ui], limit);
                wst.add(row * cols + j, check.val(got[row * cols + j]), r[0], r[1]);
            };
            try wst.done();
        }
        std.debug.print("  swiglu [{d}]: {d} clamped (row, column) cases across the checks\n", .{ cols, clamped });
    }
}

/// Rows `lo`..`lo + m` of x (row r at x + r k) through the product of `tile`, out (m, n) as raw 32-bit or 16-bit words.
fn rowsOf(t: *Rig, x: DeviceBuffer, w: DeviceBuffer, out: DeviceBuffer, lo: usize, m: usize, n: usize, k: usize, half: bool, prompt: bool, pair: usize) ![]u32 {
    const xa = at(x) + lo * k * 2;
    if (pair > 0) {
        try t.on.affine.bf16PairGo(&t.r, xa, at(w), at(out), @intCast(m), @intCast(pair), @intCast(k), 10, t.stream.handle);
    } else {
        const a = K.bf16Arg(xa, at(w), if (half) 0 else at(out), if (half) at(out) else 0, @intCast(m), @intCast(n), @intCast(k));
        try t.on.affine.bf16Go(&t.r, a, prompt, t.stream.handle);
    }
    try t.stream.synchronize();
    const cols = if (pair > 0) pair else n;
    const words = m * cols;
    const got = try gpa.alloc(u32, words);
    if (half or pair > 0) {
        const h = try gpa.alloc(u16, words);
        defer gpa.free(h);
        try out.download(0, std.mem.sliceAsBytes(h));
        for (got, h) |*g, v| g.* = v;
    } else {
        try out.download(0, std.mem.sliceAsBytes(got));
    }
    return got;
}

test "glm bf16: the stream tile gives a row the same bits at 1 to 8 rows and at any place in the window (verify windows)" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    if (skip(t)) return error.SkipZigTest;
    var rng: rig.Rng = .{ .state = 0xB175_0001 };
    const most = 8;
    const Shape = struct { n: usize, k: usize, pair: usize = 0 };
    // GLM's decode shapes: KDA q/k/v, KDA o, MLA q_b and o, kv_b, the router, the shared expert (pair and down), the head
    const shapes = [_]Shape{ .{ .n = 8192, .k = 4096 }, .{ .n = 4096, .k = 8192 }, .{ .n = 16384, .k = 1536 }, .{ .n = 4096, .k = 16384 }, .{ .n = 32768, .k = 512 }, .{ .n = 288, .k = 4096 }, .{ .n = 4096, .k = 4096, .pair = 2048 }, .{ .n = 4096, .k = 2048 }, .{ .n = 154880, .k = 4096 }, .{ .n = 4096, .k = 128 } };
    var cases: usize = 0;
    for (shapes) |sh| {
        const hx = try check.normals(&rng, (most + 3) * sh.k, 1.0);
        defer gpa.free(hx);
        const hw = try check.normals(&rng, sh.n * sh.k, 0.05);
        defer gpa.free(hw);
        var x = try t.upload(hx);
        defer x.free();
        var w = try t.upload(hw);
        defer w.free();
        const cols = if (sh.pair > 0) sh.pair else sh.n;
        var out = try t.alloc((most + 3) * cols * 4);
        defer out.free();
        for ([_]bool{ false, true }) |half| {
            if (sh.pair > 0 and !half) continue;
            // the reference: each row alone (1 row)
            const one = try gpa.alloc(u32, (most + 3) * cols);
            defer gpa.free(one);
            for (0..most + 3) |r| {
                const g = try rowsOf(t, x, w, out, r, 1, sh.n, sh.k, half, false, sh.pair);
                defer gpa.free(g);
                @memcpy(one[r * cols ..][0..cols], g);
            }
            for ([_]usize{ 2, 3, 4, 5, 6, 7, 8 }) |m| for ([_]usize{ 0, 3 }) |lo| {
                const g = try rowsOf(t, x, w, out, lo, m, sh.n, sh.k, half, false, sh.pair);
                defer gpa.free(g);
                for (0..m) |r| {
                    const same = std.mem.eql(u32, g[r * cols ..][0..cols], one[(lo + r) * cols ..][0..cols]);
                    try rig.expect(same, "bf16 stream [{d}, {d}] pair {d} {s}: row {d} of {d} (from {d}) differs from the row alone", .{ sh.n, sh.k, sh.pair, if (half) "bf16" else "fp32", r, m, lo });
                }
                cases += 1;
            };
        }
    }
    std.debug.print("RESULT glm bf16 stream rows: {d} windows of 2 to 8 rows, every row's bits equal the row alone\n", .{cases});
}

test "glm bf16: the WMMA tile gives a row the same bits at any m and any place in the chunk (2,048 rows = 4 x 512 = rows alone)" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    if (skip(t) or t.on.affine.bf16_wmma[0] == null) return error.SkipZigTest;
    var rng: rig.Rng = .{ .state = 0xB175_0002 };
    const rows = 2048;
    for (K.bf16_ks) |kc| {
        const k: usize = @intCast(kc);
        const n: usize = 320;
        const hx = try check.normals(&rng, rows * k, 1.0);
        defer gpa.free(hx);
        const hw = try check.normals(&rng, n * k, 0.05);
        defer gpa.free(hw);
        var x = try t.upload(hx);
        defer x.free();
        var w = try t.upload(hw);
        defer w.free();
        var out = try t.alloc(rows * n * 4);
        defer out.free();
        for ([_]bool{ false, true }) |half| {
            const full = try rowsOf(t, x, w, out, 0, rows, n, k, half, true, 0);
            defer gpa.free(full);
            // 512-row chunks, odd cuts that move rows inside the 256-row blocks and 16-row tiles, and rows alone
            const cuts = [_][2]usize{ .{ 0, 512 }, .{ 512, 512 }, .{ 1024, 512 }, .{ 1536, 512 }, .{ 37, 300 }, .{ 1, 255 }, .{ 2000, 48 }, .{ 7, 1 }, .{ 2047, 1 }, .{ 129, 17 } };
            for (cuts) |c| {
                const g = try rowsOf(t, x, w, out, c[0], c[1], n, k, half, true, 0);
                defer gpa.free(g);
                try rig.expect(std.mem.eql(u32, g, full[c[0] * n ..][0 .. c[1] * n]), "bf16 wmma K {d} {s}: rows {d}..{d} alone differ from the 2,048-row call", .{ k, if (half) "bf16" else "fp32", c[0], c[0] + c[1] });
            }
        }
    }
    std.debug.print("RESULT glm bf16 wmma rows: every K, fp32 and bf16, 10 cuts of a 2,048-row chunk equal its rows\n", .{});
}

const Timed = struct {
    t: *Rig,
    a: affine.Arg,
    prompt: bool,
    pair: usize = 0,

    fn call(c: Timed) anyerror!void {
        if (c.pair > 0) return c.t.on.affine.bf16PairGo(&c.t.r, c.a.x, c.a.words, c.a.out16, c.a.m, @intCast(c.pair), c.a.k, 10, c.t.stream.handle);
        try c.t.on.affine.bf16Go(&c.t.r, c.a, c.prompt, c.t.stream.handle);
    }
};

test "glm bf16: speed facts (no gate): the stream tile's GB/s at 1 and 4 rows, the WMMA tile's TFLOPS at 2048 rows" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    if (skip(t) or t.on.affine.bf16_wmma[0] == null) return error.SkipZigTest;
    const Shape = struct { name: []const u8, n: usize, k: usize, pair: usize = 0 };
    const shapes = [_]Shape{
        .{ .name = "KDA q/k/v", .n = 8192, .k = 4096 },
        .{ .name = "KDA o", .n = 4096, .k = 8192 },
        .{ .name = "MLA q_b", .n = 16384, .k = 1536 },
        .{ .name = "MLA o", .n = 4096, .k = 16384 },
        .{ .name = "router", .n = 288, .k = 4096 },
        .{ .name = "shared gate|up", .n = 4096, .k = 4096, .pair = 2048 },
        .{ .name = "shared down", .n = 4096, .k = 2048 },
        .{ .name = "dense gate|up", .n = 24576, .k = 4096, .pair = 12288 },
        .{ .name = "dense down", .n = 4096, .k = 12288 },
        .{ .name = "head", .n = 154880, .k = 4096 },
    };
    const most_rows = 2048;
    var x = try t.alloc(most_rows * 16384 * 2);
    defer x.free();
    try x.fill8(0x3c);
    var w = try t.alloc(154880 * 4096 * 2);
    defer w.free();
    try w.fill8(0x3c);
    var out = try t.alloc(most_rows * 154880 * 4);
    defer out.free();
    for (shapes) |s| {
        const bytes: f64 = @floatFromInt(s.n * s.k * 2);
        for ([_]usize{ 1, 4 }) |m| {
            const a = K.bf16Arg(at(x), at(w), if (s.pair > 0) 0 else at(out), if (s.pair > 0) at(out) else 0, @intCast(m), @intCast(s.n), @intCast(s.k));
            const us = try check.timeUs(t, Timed{ .t = t, .a = a, .prompt = false, .pair = s.pair }, Timed.call, 200);
            std.debug.print("SPEED glm bf16 stream {s} [{d}, {d}] {d} rows: {d:.1} us, {d:.1} GB/s of weights\n", .{ s.name, s.n, s.k, m, us, bytes / us / 1e3 });
        }
        if (s.n == 154880) continue; // the prompt's head is one row
        const a = K.bf16Arg(at(x), at(w), at(out), 0, most_rows, @intCast(s.n), @intCast(s.k));
        const us = try check.timeUs(t, Timed{ .t = t, .a = a, .prompt = true }, Timed.call, 20);
        const flops: f64 = 2.0 * most_rows * @as(f64, @floatFromInt(s.n * s.k));
        std.debug.print("SPEED glm bf16 wmma {s} [{d}, {d}] {d} rows: {d:.1} us, {d:.1} TFLOPS\n", .{ s.name, s.n, s.k, most_rows, us, flops / us / 1e6 });
    }
}
