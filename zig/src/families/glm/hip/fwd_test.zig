//! Lane F's integration tests (`zig build glm-fwd-test -Dhip=true`, filter "glm fwd"): each block of forward.zig on
//! synthetic weights at GLM-5.3-Flash's shapes, against a float64 composition of its ops (the CUDA path's math and
//! rounding points), and a window's rows against the same rows cut into other windows, bit for bit.
//!
//! Tolerance of the float64 compare: Gate 1's bound, a relative Frobenius error under 2e-2 a row. The lanes' contract
//! tests hold each kernel to its elementwise bound; these tests check the composition: a wrong stride, column, offset
//! or buffer gives an error of order 1. The worst error is printed.
//! Skipped in a build without kernels (the host test step's stubs) or without a HIP device.

const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const config = @import("../config.zig");
const weights = @import("weights.zig");
const forward = @import("forward.zig");
const state = @import("state.zig");

const fmt = core.exl3_format;
const State = state.State;
const Buffers = state.Buffers;

const D: usize = 4096;
const n_kda: usize = 8192;
const dk: usize = 128;
const heads: usize = 64;
const bound = 2e-2;

// ---- the device and host helpers ----

const Rig = struct {
    r: hip.Runtime,
    ctx: hip.Context,
    s: hip.Stream,
    l: hip.Launcher,
    model: weights.Model = undefined,
    bufs: std.ArrayList(hip.DeviceBuffer) = .empty,
    arena: std.heap.ArenaAllocator,

    fn open() !*Rig {
        if (hip.kernels.images[0].len == 0) return error.SkipZigTest;
        const t = try std.heap.page_allocator.create(Rig);
        t.arena = .init(std.heap.page_allocator);
        t.bufs = .empty;
        t.r = hip.Runtime.open() catch return error.SkipZigTest;
        t.ctx = try hip.Context.init(&t.r, 0);
        t.s = try hip.Stream.init(&t.r);
        t.l = try hip.Launcher.load(&t.r, .{}, hip.kernels.images);
        return t;
    }

    fn close(t: *Rig) void {
        for (t.bufs.items) |*b| b.free();
        t.l.unload();
        t.s.deinit();
        t.ctx.deinit();
        t.r.close();
        t.arena.deinit();
        std.heap.page_allocator.destroy(t);
    }

    fn a(t: *Rig) std.mem.Allocator {
        return t.arena.allocator();
    }

    fn up(t: *Rig, host: anytype) !u64 {
        const b = try hip.DeviceBuffer.fromHost(&t.r, std.mem.sliceAsBytes(host));
        try t.bufs.append(t.a(), b);
        return b.base();
    }

    fn fwd(t: *Rig, b: *Buffers) !forward.Forward {
        return forward.Forward.init(&t.l, &t.model, .{}, t.s.handle, b);
    }

    fn put(t: *Rig, dst: u64, bytes: []const u8) !void {
        try t.s.synchronize();
        try hip.runtime.check(t.r.api.hipMemcpyHtoD(@ptrFromInt(dst), bytes.ptr, bytes.len));
    }

    fn get(t: *Rig, comptime T: type, src: u64, n: usize) ![]T {
        const out = try t.a().alloc(T, n);
        try t.s.synchronize();
        try hip.runtime.check(t.r.api.hipMemcpyDtoH(@ptrCast(out.ptr), @ptrFromInt(src), n * @sizeOf(T)));
        return out;
    }
};

const Rng = struct {
    state: u64,

    fn next(r: *Rng) u64 {
        r.state ^= r.state >> 12;
        r.state ^= r.state << 25;
        r.state ^= r.state >> 27;
        return r.state *% 0x2545F4914F6CDD1D;
    }

    fn uniform(r: *Rng) f64 {
        return (@as(f64, @floatFromInt(r.next() >> 11)) + 0.5) / 9007199254740992.0;
    }

    fn normal(r: *Rng) f64 {
        return @sqrt(-2 * @log(r.uniform())) * @cos(2 * std.math.pi * r.uniform());
    }
};

/// bf16 bits of v, round to nearest even (as the kernels' float_to_bf16).
fn bfBits(v: f64) u16 {
    const f: f32 = @floatCast(v);
    const u: u32 = @bitCast(f);
    if (std.math.isNan(f)) return @intCast(u >> 16 | 0x40);
    return @intCast((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

fn bfVal(b: u16) f64 {
    return @as(f32, @bitCast(@as(u32, b) << 16));
}

/// v rounded to bf16, as float64.
fn rb(v: f64) f64 {
    return bfVal(bfBits(v));
}

fn sigmoid(v: f64) f64 {
    return 1 / (1 + @exp(-v));
}

/// `n` bf16 values N(0, scale), with their float64 values.
fn bfs(t: *Rig, rng: *Rng, n: usize, scale: f64) !struct { bits: []u16, val: []f64 } {
    const bits = try t.a().alloc(u16, n);
    const val = try t.a().alloc(f64, n);
    for (bits, val) |*b, *v| {
        b.* = bfBits(rng.normal() * scale);
        v.* = bfVal(b.*);
    }
    return .{ .bits = bits, .val = val };
}

/// A BF16 linear [n][k] on the device (N(0, 1/sqrt(k)) unless `scale`) and its float64 values.
const Lin = struct {
    w: weights.Bf16,
    v: []f64,

    fn make(t: *Rig, rng: *Rng, n: usize, k: usize, scale: ?f64) !Lin {
        const x = try bfs(t, rng, n * k, scale orelse 1 / @sqrt(@as(f64, @floatFromInt(k))));
        return .{ .w = .{ .w = try t.up(x.bits), .n = @intCast(n), .k = @intCast(k) }, .v = x.val };
    }
};

/// out [rows][n] = x [rows][k] (row stride xs, from column x0) . w[n0 .. n0 + n][k]^T in float64.
fn mm(a: std.mem.Allocator, x: []const f64, xs: usize, x0: usize, rows: usize, w: []const f64, n0: usize, n: usize, k: usize) ![]f64 {
    const out = try a.alloc(f64, rows * n);
    for (0..rows) |r| {
        const xr = x[r * xs + x0 ..][0..k];
        for (0..n) |j| {
            const wr = w[(n0 + j) * k ..][0..k];
            var s: f64 = 0;
            for (xr, wr) |p, q| s += p * q;
            out[r * n + j] = s;
        }
    }
    return out;
}

/// sum over k of |x[r][k] w[j][k]| for rows 0..rows and outputs 0..n: the BF16 product bound's magnitude.
fn mmAbs(a: std.mem.Allocator, x: []const f64, xs: usize, rows: usize, w: []const f64, n: usize, k: usize) ![]f64 {
    const out = try a.alloc(f64, rows * n);
    for (0..rows) |r| {
        const xr = x[r * xs ..][0..k];
        for (0..n) |j| {
            var s: f64 = 0;
            for (xr, w[j * k ..][0..k]) |p, q| s += @abs(p * q);
            out[r * n + j] = s;
        }
    }
    return out;
}

fn roundAll(v: []f64) void {
    for (v) |*x| x.* = rb(x.*);
}

/// glue.rmsnorm (G4 tf_glm_rms): bf16(w bf16(x / sqrt(mean(x^2) + eps))).
fn rms(a: std.mem.Allocator, x: []const f64, xs: usize, x0: usize, rows: usize, w: []const f64, width: usize, eps: f64) ![]f64 {
    const out = try a.alloc(f64, rows * width);
    for (0..rows) |r| {
        const xr = x[r * xs + x0 ..][0..width];
        var ss: f64 = 0;
        for (xr) |v| ss += v * v;
        const inv = 1 / @sqrt(eps + ss / @as(f64, @floatFromInt(width)));
        for (out[r * width ..][0..width], xr, w) |*o, v, g| o.* = rb(g * rb(v * inv));
    }
    return out;
}

/// The worst relative Frobenius error a row of got (fp32 [rows][n]) against want; fails past `bound`.
fn expectRows(name: []const u8, got: []const f32, want: []const f64, rows: usize, n: usize) !void {
    var worst: f64 = 0;
    for (0..rows) |r| {
        var e: f64 = 0;
        var m: f64 = 0;
        for (got[r * n ..][0..n], want[r * n ..][0..n]) |g, w| {
            e += (g - w) * (g - w);
            m += w * w;
        }
        const rel = @sqrt(e / @max(m, 1e-300));
        if (!(rel < bound)) {
            std.debug.print("FAIL {s}: row {d} relative error {e} (bound {e})\n", .{ name, r, rel, bound });
            return error.TestExpectedApproxEq;
        }
        worst = @max(worst, rel);
    }
    std.debug.print("RESULT {s}: worst relative error a row {e:.3} (bound {e})\n", .{ name, worst, bound });
}

fn expectSame(name: []const u8, x: []const u8, y: []const u8) !void {
    if (!std.mem.eql(u8, x, y)) {
        var first: usize = 0;
        while (x[first] == y[first]) first += 1;
        std.debug.print("FAIL {s}: bytes differ from byte {d} of {d}\n", .{ name, first, x.len });
        return error.TestExpectedEqual;
    }
    std.debug.print("RESULT {s}: bit for bit ({d} bytes)\n", .{ name, x.len });
}

// ---- KDA ----

const KdaW = struct {
    w: weights.Kda,
    proj: []f64,
    fb: []f64,
    gb: []f64,
    o: []f64,
    conv: []f64, // [3 n][4]
    a_log: []f32,
    dt_bias: []f32,
    norm: []f64,

    fn make(t: *Rig, rng: *Rng) !KdaW {
        const W = state.kda_width;
        const proj = try Lin.make(t, rng, W, D, null);
        const fb = try Lin.make(t, rng, n_kda, dk, null);
        const gb = try Lin.make(t, rng, n_kda, dk, null);
        const o = try Lin.make(t, rng, D, n_kda, null);
        const conv = try bfs(t, rng, 3 * n_kda * 4, 0.5);
        const norm = try bfs(t, rng, dk, 1.0);
        const a_log = try t.a().alloc(f32, heads);
        for (a_log) |*v| v.* = @floatCast(rng.uniform() * 2 - 1);
        const dt = try t.a().alloc(f32, n_kda);
        for (dt) |*v| v.* = @floatCast(rng.uniform() - 0.5);
        const p = proj.w;
        return .{
            .w = .{ .proj = p, .q = p, .k = p, .v = p, .f_a = p, .g_a = p, .b = p, .f_b = fb.w, .g_b = gb.w, .conv = try t.up(conv.bits), .a_log = try t.up(a_log), .dt_bias = try t.up(dt), .o_norm = try t.up(norm.bits), .o = o.w },
            .proj = proj.v,
            .fb = fb.v,
            .gb = gb.v,
            .o = o.v,
            .conv = conv.val,
            .a_log = a_log,
            .dt_bias = dt,
            .norm = norm.val,
        };
    }
};

/// The KDA block in float64 from a fresh sequence (zero conv window and state), cuda_kda_block.zig's ops and the
/// chain of zig/tests/cuda/glm_ref_kda.zig reference() (y rounded to bf16 as CUDA; HIP keeps it fp32, G2's known
/// difference 1): part [rows][D]; `s` the state after the rows.
fn kdaRef(a: std.mem.Allocator, kw: *const KdaW, x: []const f64, rows: usize, s: []f64) ![]f64 {
    const W = state.kda_width;
    const n = n_kda;
    const p = try mm(a, x, D, 0, rows, kw.proj, 0, W, D);
    roundAll(p);
    const ka = try mm(a, p, W, 3 * n, rows, kw.fb, 0, n, dk);
    roundAll(ka);
    const kg = try mm(a, p, W, 3 * n + dk, rows, kw.gb, 0, n, dk);
    roundAll(kg);
    const out = try a.alloc(f64, rows * n);
    @memset(s, 0);
    var q: [dk]f64 = undefined;
    var k: [dk]f64 = undefined;
    var v: [dk]f64 = undefined;
    var g: [dk]f64 = undefined;
    var y: [dk]f64 = undefined;
    const lower: f64 = -5;
    for (0..heads) |h| {
        const sh = s[h * dk * dk ..][0 .. dk * dk];
        const rate = @exp(@as(f64, kw.a_log[h]));
        for (0..rows) |row| {
            for (0..3) |part| for (0..dk) |i| {
                const c = part * n + h * dk + i;
                var acc: f64 = 0;
                for (0..4) |tap| {
                    const at = row + tap; // the window's 3 zero rows, then the projection rows
                    const xv = if (at < 3) 0 else p[(at - 3) * W + c];
                    acc += kw.conv[c * 4 + tap] * xv;
                }
                const act = rb(acc * sigmoid(acc));
                switch (part) {
                    0 => q[i] = act,
                    1 => k[i] = act,
                    else => v[i] = act,
                }
            };
            for ([_]*[dk]f64{ &q, &k }, [_]f64{ 1 / @sqrt(@as(f64, dk)), 1 }) |z, extra| {
                var ss: f64 = 0;
                for (z) |e| ss += e * e;
                const inv = 1 / @sqrt(ss + 1e-6) * extra;
                for (z) |*e| e.* *= inv;
            }
            for (&g, 0..) |*e, i| e.* = @exp(lower * sigmoid(rate * (ka[row * n + h * dk + i] + kw.dt_bias[h * dk + i])));
            const beta = rb(sigmoid(p[row * W + 3 * n + 2 * dk + h]));
            for (0..dk) |j| {
                const srow = sh[j * dk ..][0..dk];
                var kv: f64 = 0;
                for (srow, g, k) |*e, gi, ki| {
                    e.* *= gi;
                    kv += e.* * ki;
                }
                const delta = (v[j] - kv) * beta;
                for (srow, k) |*e, ki| e.* += ki * delta;
            }
            var ss: f64 = 0;
            for (&y, 0..) |*e, j| {
                var o: f64 = 0;
                for (sh[j * dk ..][0..dk], q) |e2, qi| o += e2 * qi;
                e.* = rb(o);
                ss += e.* * e.*;
            }
            const rinv = 1 / @sqrt(ss / dk + 1e-5);
            for (y, 0..) |e, tt| out[row * n + h * dk + tt] = rb(kw.norm[tt] * e * rinv * sigmoid(kg[row * n + h * dk + tt]));
        }
    }
    return mm(a, out, n, 0, rows, kw.o, 0, D, n);
}

test "glm fwd: a KDA layer (20 prompt rows, then a decode row) against float64; 300 rows = 128 + 128 + 44, bit for bit" {
    const t = try Rig.open();
    defer t.close();
    const a = t.a();
    var rng: Rng = .{ .state = 0x6b_da_f0_0d };
    const kw = try KdaW.make(t, &rng);
    const cap = 512;
    // 1. float64: 20 prompt rows on the WMMA tiles, the 21st as a decode row on the stream tiles
    {
        const rows = 21;
        const x = try bfs(t, &rng, rows * D, 1.0);
        var st = try State.init(&t.r, cap);
        defer st.deinit();
        var pb = try Buffers.init(&t.r, 20, true, cap);
        defer pb.deinit();
        var db = try Buffers.init(&t.r, 1, false, cap);
        defer db.deinit();
        var fp = try t.fwd(&pb);
        var fd = try t.fwd(&db);
        try t.put(pb.normed, std.mem.sliceAsBytes(x.bits[0 .. 20 * D]));
        try fp.kdaBlock(&st, 0, kw.w, 20);
        const got_p = try t.get(f32, pb.part, 20 * D);
        try t.put(db.normed, std.mem.sliceAsBytes(x.bits[20 * D ..]));
        try fd.kdaBlock(&st, 0, kw.w, 1);
        const got_d = try t.get(f32, db.part, D);
        const s = try a.alloc(f64, heads * dk * dk);
        const want = try kdaRef(a, &kw, x.val, rows, s);
        try expectRows("glm fwd KDA block, 20 prompt rows (WMMA tiles)", got_p, want[0 .. 20 * D], 20, D);
        try expectRows("glm fwd KDA block, the decode row after them (stream tiles)", got_d, want[20 * D ..], 1, D);
        const gs = try t.get(f32, st.recAt(0), heads * dk * dk);
        try expectRows("glm fwd KDA block, the state after 21 rows (64 heads as rows)", gs, s, heads, dk * dk);
        // the conv window: the last 3 rows' q | k | v projections
        const win = try t.get(u16, st.convAt(0), 3 * 3 * n_kda);
        const p = try mm(a, x.val[18 * D ..], D, 0, 3, kw.proj, 0, 3 * n_kda, D);
        const mag = try mmAbs(a, x.val[18 * D ..], D, 3, kw.proj, 3 * n_kda, D);
        // each value within G4's BF16 product bound (kernel_tests/glm/bf16.zig): fp32 accumulation over K in any order,
        // (K + 8) 2^-24 sum |x w|, plus one bf16 rounding, 2^-8 of the value; the values off float64's rounding counted
        var flips: usize = 0;
        for (win, p, mag, 0..) |g, w, ms, i| {
            flips += @intFromBool(g != bfBits(w));
            const tol = (D + 8) * std.math.pow(f64, 2, -24) * ms + @abs(w) * std.math.pow(f64, 2, -8) + 1e-30;
            if (!(@abs(bfVal(g) - w) <= tol)) {
                std.debug.print("FAIL glm fwd KDA block: conv window value {d}: {d} not {d} (tolerance {e})\n", .{ i, bfVal(g), w, tol });
                return error.TestExpectedApproxEq;
            }
        }
        std.debug.print("RESULT glm fwd KDA block: conv window = the last 3 rows' projections within the BF16 product bound; {d} of {d} off float64's rounding\n", .{ flips, win.len });
    }
    // 2. a window's bits do not depend on its cut
    {
        const rows = 300;
        const x = try bfs(t, &rng, rows * D, 1.0);
        var outs: [2][]f32 = undefined;
        var states: [2][]f32 = undefined;
        for (0..2) |cut| {
            var st = try State.init(&t.r, cap);
            defer st.deinit();
            var pb = try Buffers.init(&t.r, rows, true, cap);
            defer pb.deinit();
            var f = try t.fwd(&pb);
            outs[cut] = try a.alloc(f32, rows * D);
            const chunks: []const usize = if (cut == 0) &.{300} else &.{ 128, 128, 44 };
            var at: usize = 0;
            for (chunks) |n| {
                try t.put(pb.normed, std.mem.sliceAsBytes(x.bits[at * D ..][0 .. n * D]));
                try f.kdaBlock(&st, 0, kw.w, n);
                @memcpy(outs[cut][at * D ..][0 .. n * D], try t.get(f32, pb.part, n * D));
                at += n;
            }
            states[cut] = try t.get(f32, st.recAt(0), heads * dk * dk);
        }
        try expectSame("glm fwd KDA block: 300 rows = 128 + 128 + 44, the outputs", std.mem.sliceAsBytes(outs[0]), std.mem.sliceAsBytes(outs[1]));
        try expectSame("glm fwd KDA block: 300 rows = 128 + 128 + 44, the state", std.mem.sliceAsBytes(states[0]), std.mem.sliceAsBytes(states[1]));
    }
}

// ---- DSA ----

const DsaW = struct {
    w: weights.Mla,
    x: []f64,
    qr: []f64,
    qn: []f64,
    kvn: []f64,
    kvb: []f64,
    o: []f64,

    fn make(t: *Rig, rng: *Rng) !DsaW {
        const c: config.Config = .{};
        const x = try Lin.make(t, rng, c.xProj(), D, null);
        const qr = try Lin.make(t, rng, c.qrProj(), c.q_lora, null);
        const kvb = try Lin.make(t, rng, heads * 512, 512, null);
        const o = try Lin.make(t, rng, D, heads * 256, null);
        const gate = try Lin.make(t, rng, 128, D, null);
        const qn = try bfs(t, rng, c.q_lora, 1.0);
        const kvn = try bfs(t, rng, 512, 1.0);
        const lnw = try bfs(t, rng, 128, 1.0);
        const lnb = try bfs(t, rng, 128, 0.1);
        const ape = try bfs(t, rng, 4 * 128, 0.5);
        const xw = x.w;
        const sub = weights.Bf16{ .w = xw.w, .n = 1, .k = xw.k };
        return .{
            .w = .{ .x = xw, .q_a = sub, .kv_a = sub, .qr = qr.w, .q_b = qr.w, .q_norm = try t.up(qn.bits), .kv_norm = try t.up(kvn.bits), .kv_b = kvb.w, .o = o.w, .index = .{ .wk = sub, .weights = sub, .wq_b = qr.w, .ln_w = try t.up(lnw.bits), .ln_b = try t.up(lnb.bits), .gate = gate.w, .ape = try t.up(ape.bits) } },
            .x = x.v,
            .qr = qr.v,
            .qn = qn.val,
            .kvn = kvn.val,
            .kvb = kvb.v,
            .o = o.v,
        };
    }
};

/// The DSA block in float64 from a fresh sequence, every row dense (positions under 2,051): cuda_dsa_block.zig's
/// projections, norms, absorb, causal latent attention over the cache, expand, o_proj (glm_ref_attn.zig's attention).
fn dsaRef(a: std.mem.Allocator, dw: *const DsaW, x: []const f64, rows: usize) ![]f64 {
    const c: config.Config = .{};
    const X: usize = c.xProj();
    const Q: usize = c.qrProj();
    const ql: usize = c.q_lora;
    const dp = try mm(a, x, D, 0, rows, dw.x, 0, X, D);
    roundAll(dp);
    const qr = try rms(a, dp, X, 0, rows, dw.qn, ql, c.eps);
    const lat = try rms(a, dp, X, ql, rows, dw.kvn, 512, c.eps);
    const qq = try mm(a, qr, ql, 0, rows, dw.qr, 0, Q, ql);
    roundAll(qq);
    const vn = try a.alloc(f64, rows * heads * 256);
    const sc = try a.alloc(f64, rows);
    for (0..heads) |h| {
        const wk = dw.kvb[h * 512 * 512 ..][0 .. 256 * 512]; // [256][512]
        const wv = dw.kvb[(h * 512 + 256) * 512 ..][0 .. 256 * 512];
        for (0..rows) |r| {
            var qa: [512]f64 = undefined;
            for (&qa, 0..) |*e, l| {
                var s: f64 = 0;
                for (0..256) |d| s += qq[r * Q + h * 256 + d] * wk[d * 512 + l];
                e.* = rb(s);
            }
            var m: f64 = -std.math.inf(f64);
            for (0..r + 1) |j| {
                var s: f64 = 0;
                for (qa, lat[j * 512 ..][0..512]) |p, q| s += p * q;
                sc[j] = s / 16.0;
                m = @max(m, sc[j]);
            }
            var l: f64 = 0;
            var ol: [512]f64 = @splat(0);
            for (0..r + 1) |j| {
                const e = @exp(sc[j] - m);
                l += e;
                for (&ol, lat[j * 512 ..][0..512]) |*o, v| o.* += e * v;
            }
            for (&ol) |*o| o.* = rb(o.* / l);
            for (0..256) |d| {
                var s: f64 = 0;
                for (ol, wv[d * 512 ..][0..512]) |p, q| s += p * q;
                vn[r * heads * 256 + h * 256 + d] = rb(s);
            }
        }
    }
    return mm(a, vn, heads * 256, 0, rows, dw.o, 0, D, heads * 256);
}

test "glm fwd: a DSA layer (24 dense prompt rows, then a decode row) against float64; 2,200 rows across the 2,051-key limit in two cuts, bit for bit" {
    const t = try Rig.open();
    defer t.close();
    const a = t.a();
    var rng: Rng = .{ .state = 0xd5a_0001 };
    const dw = try DsaW.make(t, &rng);
    {
        const cap = 512;
        const rows = 25;
        const x = try bfs(t, &rng, rows * D, 1.0);
        var st = try State.init(&t.r, cap);
        defer st.deinit();
        var pb = try Buffers.init(&t.r, 24, true, cap);
        defer pb.deinit();
        var db = try Buffers.init(&t.r, 1, false, cap);
        defer db.deinit();
        var fp = try t.fwd(&pb);
        var fd = try t.fwd(&db);
        try st.reset(t.s.handle);
        try t.put(pb.normed, std.mem.sliceAsBytes(x.bits[0 .. 24 * D]));
        try st.ready(t.s.handle, 24);
        try fp.dsaBlock(st.caches[0], st.pos_dev, st.pos, dw.w, 24);
        const got_p = try t.get(f32, pb.part, 24 * D);
        try st.setPos(t.s.handle, 24);
        try t.put(db.normed, std.mem.sliceAsBytes(x.bits[24 * D ..]));
        try fd.dsaBlock(st.caches[0], st.pos_dev, st.pos, dw.w, 1);
        const got_d = try t.get(f32, db.part, D);
        const want = try dsaRef(a, &dw, x.val, rows);
        try expectRows("glm fwd DSA block, 24 dense prompt rows (WMMA tiles)", got_p, want[0 .. 24 * D], 24, D);
        try expectRows("glm fwd DSA block, the decode row at 24 (stream tiles)", got_d, want[24 * D ..], 1, D);
    }
    {
        // cut A: 2,048 (a dense call) + 152 (listed); cut B: 512 x 3 + 504 + 160: rows 2,040-2,050 attend densely
        // inside a listed call, rows 2,051 on select their tokens
        const cap = 2240;
        const rows = 2200;
        const x = try bfs(t, &rng, rows * D, 1.0);
        var outs: [2][]f32 = undefined;
        for (0..2) |cut| {
            var st = try State.init(&t.r, cap);
            defer st.deinit();
            var pb = try Buffers.init(&t.r, 2048, true, cap);
            defer pb.deinit();
            var f = try t.fwd(&pb);
            outs[cut] = try a.alloc(f32, rows * D);
            const chunks: []const usize = if (cut == 0) &.{ 2048, 152 } else &.{ 512, 512, 512, 504, 160 };
            for (chunks) |n| {
                const at = st.pos;
                try t.put(pb.normed, std.mem.sliceAsBytes(x.bits[at * D ..][0 .. n * D]));
                try st.ready(t.s.handle, n);
                try f.dsaBlock(st.caches[0], st.pos_dev, st.pos, dw.w, n);
                @memcpy(outs[cut][at * D ..][0 .. n * D], try t.get(f32, pb.part, n * D));
                try st.setPos(t.s.handle, at + n);
            }
        }
        try expectSame("glm fwd DSA block: 2,200 rows as 2,048 + 152 = 512 x 3 + 504 + 160", std.mem.sliceAsBytes(outs[0]), std.mem.sliceAsBytes(outs[1]));
    }
}

// ---- the dense MLP and the MoE ----

/// The 128-blocks of a row-major [rows][cols] float64 matrix through H / sqrt(128) along `axis` (0: rows, 1: columns).
fn butterfly(x: []f64, rows: usize, cols: usize, axis: usize) void {
    if (axis == 0) {
        var b: usize = 0;
        while (b < rows) : (b += 128) {
            var h: usize = 1;
            while (h < 128) : (h *= 2) {
                var i: usize = 0;
                while (i < 128) : (i += 2 * h) for (i..i + h) |r| {
                    const p = x[(b + r) * cols ..][0..cols];
                    const q = x[(b + r + h) * cols ..][0..cols];
                    for (p, q) |*u, *v| {
                        const s = u.*;
                        u.* = s + v.*;
                        v.* = s - v.*;
                    }
                };
            }
        }
    } else for (0..rows) |r| {
        var b: usize = 0;
        while (b < cols) : (b += 128) {
            const v = x[r * cols + b ..][0..128];
            var h: usize = 1;
            while (h < 128) : (h *= 2) {
                var i: usize = 0;
                while (i < 128) : (i += 2 * h) for (i..i + h) |j| {
                    const s = v[j];
                    v[j] = s + v[j + h];
                    v[j + h] = s - v[j + h];
                };
            }
        }
    }
    const inv = 1.0 / @sqrt(128.0);
    for (x) |*v| v.* *= inv;
}

fn swigluRef(g: f64, u: f64) f64 {
    const gg = @min(g, 10.0);
    const uu = @max(@min(u, 10.0), -10.0);
    return gg / (1 + @exp(-gg)) * uu;
}

/// [gate | up] (w rows 0..cols gate, cols..2 cols up), SwiGLU at CUDA's rounding points (glm_glue_moe.cuh swiglu1: g and
/// u bf16, bf16(silu(min(g, 10))), bf16 of the product), down, in float64.
fn mlpRef(a: std.mem.Allocator, x: []const f64, rows: usize, gu: []const f64, down: []const f64, cols: usize) ![]f64 {
    const h = try mm(a, x, D, 0, rows, gu, 0, 2 * cols, D);
    const act = try a.alloc(f64, rows * cols);
    for (0..rows) |r| for (0..cols) |j| {
        const g = @min(rb(h[r * 2 * cols + j]), 10.0);
        const u = @max(@min(rb(h[r * 2 * cols + cols + j]), 10.0), -10.0);
        act[r * cols + j] = rb(rb(g / (1 + @exp(-g))) * u);
    };
    return mm(a, act, cols, 0, rows, down, 0, D, cols);
}

test "glm fwd: the dense MLP (12,288, CUDA's SwiGLU rounding) against float64 on both tiles; 300 rows = 200 + 100, bit for bit" {
    const t = try Rig.open();
    defer t.close();
    const a = t.a();
    var rng: Rng = .{ .state = 0xde_75e };
    const w: usize = 12288;
    const gu = try Lin.make(t, &rng, 2 * w, D, null);
    const down = try Lin.make(t, &rng, D, w, null);
    const dw: weights.Dense = .{ .gu = gu.w, .down = down.w };
    const x = try bfs(t, &rng, 300 * D, 1.0);
    const want = try mlpRef(a, x.val, 8, gu.v, down.v, w);
    for ([_]bool{ true, false }) |prompt| {
        var b = try Buffers.init(&t.r, 8, prompt, 64);
        defer b.deinit();
        var f = try t.fwd(&b);
        try t.put(b.normed, std.mem.sliceAsBytes(x.bits[0 .. 8 * D]));
        try f.denseBlock(dw, 8);
        try expectRows(if (prompt) "glm fwd dense MLP, 8 rows (WMMA tile, glm.swiglu)" else "glm fwd dense MLP, 8 rows (stream tile, bf16GluGo)", try t.get(f32, b.part, 8 * D), want, 8, D);
    }
    var outs: [2][]f32 = undefined;
    for (0..2) |cut| {
        var b = try Buffers.init(&t.r, 300, true, 64);
        defer b.deinit();
        var f = try t.fwd(&b);
        outs[cut] = try a.alloc(f32, 300 * D);
        const chunks: []const usize = if (cut == 0) &.{300} else &.{ 200, 100 };
        var at: usize = 0;
        for (chunks) |n| {
            try t.put(b.normed, std.mem.sliceAsBytes(x.bits[at * D ..][0 .. n * D]));
            try f.denseBlock(dw, n);
            @memcpy(outs[cut][at * D ..][0 .. n * D], try t.get(f32, b.part, n * D));
            at += n;
        }
    }
    try expectSame("glm fwd dense MLP: 300 rows = 200 + 100", std.mem.sliceAsBytes(outs[0]), std.mem.sliceAsBytes(outs[1]));
}

/// 288 experts over `distinct` synthetic trellis sets (expert e uses set e % distinct): random 4-bit mcg trellis,
/// one input rotation for every gate and up (shared_input), svh +-0.02; the tables, suh and svh stacked as G0 places.
const Experts = struct {
    e: weights.Experts,
    tr: [3][]const []const i16,
    suh: [3][]const f16,
    svh: [3][]const []const f16,
    distinct: usize,

    fn kn(m: usize) [2]usize {
        return if (m < 2) .{ D, 2048 } else .{ 2048, D };
    }

    fn make(t: *Rig, rng: *Rng, distinct: usize) !Experts {
        const a = t.a();
        const E: usize = 288;
        var x: Experts = .{ .e = undefined, .tr = undefined, .suh = undefined, .svh = undefined, .distinct = distinct };
        var tables: [3]u64 = undefined;
        var suh_d: [3]u64 = undefined;
        var svh_d: [3]u64 = undefined;
        const rot = try a.alloc(f16, D);
        for (rot) |*v| v.* = if (rng.next() & 1 == 1) 1.0 else -1.0;
        for (0..3) |m| {
            const k = kn(m)[0];
            const n = kn(m)[1];
            const words = k / 16 * (n / 16) * 64;
            const tr = try a.alloc([]const i16, distinct);
            const sv = try a.alloc([]const f16, distinct);
            const table = try a.alloc(u64, E);
            const su_all = try a.alloc(f16, E * k);
            const sv_all = try a.alloc(f16, E * n);
            const su = if (m < 2) rot else blk: {
                const s = try a.alloc(f16, k);
                for (s) |*v| v.* = if (rng.next() & 1 == 1) 1.0 else -1.0;
                break :blk s;
            };
            const bases = try a.alloc(u64, distinct);
            for (0..distinct) |i| {
                const w = try a.alloc(i16, words);
                for (w) |*v| v.* = @bitCast(@as(u16, @truncate(rng.next())));
                tr[i] = w;
                bases[i] = try t.up(w);
                const v = try a.alloc(f16, n);
                for (v) |*e| e.* = if (rng.next() & 1 == 1) 0.02 else -0.02;
                sv[i] = v;
            }
            for (0..E) |e| {
                table[e] = bases[e % distinct];
                @memcpy(su_all[e * k ..][0..k], su);
                @memcpy(sv_all[e * n ..][0..n], sv[e % distinct]);
            }
            tables[m] = try t.up(table);
            suh_d[m] = try t.up(su_all);
            svh_d[m] = try t.up(sv_all);
            x.tr[m] = tr;
            x.suh[m] = su;
            x.svh[m] = sv;
        }
        x.e = .{ .gate = tables[0], .up = tables[1], .down = tables[2], .suh_g = suh_d[0], .suh_u = suh_d[1], .svh_g = svh_d[0], .svh_u = svh_d[1], .suh_d = suh_d[2], .svh_d = svh_d[2], .count = E, .dims = D, .width = 2048, .shared_input = true, .limit = 10 };
        return x;
    }

    /// Set i's matrix m as float64 [K][N]: diag(suh) H W_q H diag(svh), exl3_format.dequantize's matrix with the
    /// 128-point rotations as butterflies (G1's dequantFast, checked against dequantize to 1e-12 in its tests).
    fn matrix(x: *const Experts, a: std.mem.Allocator, i: usize, m: usize) ![]f64 {
        const k = kn(m)[0];
        const n = kn(m)[1];
        const out = try a.alloc(f64, k * n);
        const wq = try a.alloc(f16, k * n);
        defer a.free(wq);
        fmt.unpack(.{ .halves = 8 }, .mcg, x.tr[m][i], k / 16, n / 16, wq);
        for (out, wq) |*d, v| d.* = @floatCast(v);
        butterfly(out, k, n, 0);
        for (out, 0..) |*v, j| v.* *= @floatCast(x.suh[m][j / n]);
        butterfly(out, k, n, 1);
        for (out, 0..) |*v, j| v.* *= @floatCast(x.svh[m][i][j % n]);
        return out;
    }
};

test "glm fwd: an MoE layer (router, top-8 of 288, EXL3 experts, shared) against float64 on both paths; 300 rows = 100 x 3, bit for bit" {
    const t = try Rig.open();
    defer t.close();
    const a = t.a();
    var rng: Rng = .{ .state = 0x0e_288 };
    const E: usize = 288;
    const distinct = 6;
    const ex = try Experts.make(t, &rng, distinct);
    const router = try Lin.make(t, &rng, E, D, null);
    const bias = try a.alloc(f32, E);
    for (bias) |*v| v.* = @floatCast((rng.uniform() - 0.5) * 0.1);
    const sgu = try Lin.make(t, &rng, 2 * 2048, D, null);
    const sdown = try Lin.make(t, &rng, D, 2048, null);
    const mw: weights.Moe = .{ .router = router.w, .bias = try t.up(bias), .shared_gu = sgu.w, .shared_down = sdown.w, .experts = ex.e, .stacks = undefined };
    const rows = 4;
    const x = try bfs(t, &rng, 300 * D, 1.0);
    // float64: the pick (sigmoid + bias, ties to the lower id), the weights (normed, x 2.5), each expert's SwiGLU
    // (limit 10) and down, the shared expert, the slots' sum in order plus the shared row
    const lg = try mm(a, x.val, D, 0, rows, router.v, 0, E, D);
    var mats: [distinct][3][]f64 = undefined;
    for (0..distinct) |i| for (0..3) |m| {
        mats[i][m] = try ex.matrix(a, i, m);
    };
    const want = try mlpRef(a, x.val, rows, sgu.v, sdown.v, 2048);
    for (0..rows) |r| {
        var order: [288]usize = undefined;
        for (&order, 0..) |*o, e| o.* = e;
        const Ctx = struct { l: []const f64, b: []const f32 };
        std.mem.sort(usize, &order, Ctx{ .l = lg[r * E ..][0..E], .b = bias }, struct {
            fn lt(cx: Ctx, p: usize, q: usize) bool {
                const sp = sigmoid(cx.l[p]) + cx.b[p];
                const sq = sigmoid(cx.l[q]) + cx.b[q];
                return sp > sq or (sp == sq and p < q);
            }
        }.lt);
        var total: f64 = 0;
        for (order[0..8]) |e| total += sigmoid(lg[r * E + e]);
        for (order[0..8]) |e| {
            const wt = sigmoid(lg[r * E + e]) / total * 2.5;
            const i = e % distinct;
            // x [D] . gate and up [D][2048], then act [2048] . down [2048][D], rows of W contiguous
            var g: [2048]f64 = @splat(0);
            var u: [2048]f64 = @splat(0);
            for (0..D) |kk| {
                const xv = x.val[r * D + kk];
                for (&g, mats[i][0][kk * 2048 ..][0..2048]) |*o, wv| o.* += xv * wv;
                for (&u, mats[i][1][kk * 2048 ..][0..2048]) |*o, wv| o.* += xv * wv;
            }
            var y: [D]f64 = @splat(0);
            for (g, u, 0..) |gv, uv, kk| {
                const av = swigluRef(gv, uv);
                for (&y, mats[i][2][kk * D ..][0..D]) |*o, wv| o.* += av * wv;
            }
            for (want[r * D ..][0..D], y) |*o, yv| o.* += wt * yv;
        }
    }
    for ([_]bool{ true, false }) |prompt| {
        var b = try Buffers.init(&t.r, rows, prompt, 64);
        defer b.deinit();
        var f = try t.fwd(&b);
        try t.put(b.normed, std.mem.sliceAsBytes(x.bits[0 .. rows * D]));
        try f.moeBlock(mw, 3, rows);
        try expectRows(if (prompt) "glm fwd MoE block, 4 rows (WMMA tiles, EXL3 GEMM)" else "glm fwd MoE block, 4 rows (stream tiles, EXL3 GEMV)", try t.get(f32, b.part, rows * D), want, rows, D);
    }
    var outs: [2][]f32 = undefined;
    for (0..2) |cut| {
        var b = try Buffers.init(&t.r, 300, true, 64);
        defer b.deinit();
        var f = try t.fwd(&b);
        outs[cut] = try a.alloc(f32, 300 * D);
        const chunks: []const usize = if (cut == 0) &.{300} else &.{ 100, 100, 100 };
        var at: usize = 0;
        for (chunks) |n| {
            try t.put(b.normed, std.mem.sliceAsBytes(x.bits[at * D ..][0 .. n * D]));
            try f.moeBlock(mw, 3, n);
            @memcpy(outs[cut][at * D ..][0 .. n * D], try t.get(f32, b.part, n * D));
            at += n;
        }
    }
    try expectSame("glm fwd MoE block: 300 rows = 100 x 3", std.mem.sliceAsBytes(outs[0]), std.mem.sliceAsBytes(outs[1]));
    // TF_GLM_FORCE_PICKS: forced with the router's own picks and weights, the block gives the free run's bits
    for ([_]bool{ true, false }) |prompt| {
        var b = try Buffers.init(&t.r, rows, prompt, 64);
        defer b.deinit();
        var f = try t.fwd(&b);
        try t.put(b.normed, std.mem.sliceAsBytes(x.bits[0 .. rows * D]));
        try f.moeBlock(mw, 3, rows);
        const free = try t.get(f32, b.part, rows * D);
        const pk = try t.get(i32, f.moe_last.pick, rows * 8);
        const ws = try t.get(f32, f.moe_last.wts, rows * 8);
        const fp = try a.alloc(i16, 46 * rows * 8);
        const fw = try a.alloc(f32, 46 * rows * 8);
        @memset(fp, 0);
        @memset(fw, 0);
        for (pk, ws, 0..) |id, wv, i| {
            fp[3 * rows * 8 + i] = @intCast(id);
            fw[3 * rows * 8 + i] = wv;
        }
        const fc: forward.Forward.Force = .{ .picks = fp, .wts = fw, .rows = rows };
        f.force = &fc;
        try t.put(b.normed, std.mem.sliceAsBytes(x.bits[0 .. rows * D]));
        try f.moeBlock(mw, 3, rows);
        const forced = try t.get(f32, b.part, rows * D);
        try expectSame(if (prompt) "glm fwd MoE block forced with its own picks = free (prompt)" else "glm fwd MoE block forced with its own picks = free (decode)", std.mem.sliceAsBytes(free), std.mem.sliceAsBytes(forced));
        // and forced with other picks it differs (the switch is live)
        for (fp[3 * rows * 8 ..][0 .. rows * 8]) |*v| v.* = @intCast(@mod(@as(i32, v.*) + 1, 288));
        try t.put(b.normed, std.mem.sliceAsBytes(x.bits[0 .. rows * D]));
        try f.moeBlock(mw, 3, rows);
        const other = try t.get(f32, b.part, rows * D);
        if (std.mem.eql(u8, std.mem.sliceAsBytes(other), std.mem.sliceAsBytes(free))) return error.ForceNotLive;
    }
}

// ---- verify windows (the MTP rounds): a window's rows and its keep = the serial decode rows (LANES.md contract 7) ----

/// The windows: (rows, kept) from the first decode row on; each starts at the rows kept so far. A rejected row
/// completes a DSA pool (at 2,056 + 1: rows 2,057-2,060 keep 2), and every keep 1-4 occurs.
const sched = [_][2]usize{ .{ 3, 1 }, .{ 4, 2 }, .{ 2, 2 }, .{ 4, 3 }, .{ 3, 3 }, .{ 2, 1 }, .{ 4, 4 } };
const sched_rows = 16; // the serial rows the windows read: the last starts at 12 with 4 rows

fn sameRows(name: []const u8, at: usize, got: []const f32, want: []const f32) !void {
    const gb: []const u32 = @ptrCast(got);
    const wb: []const u32 = @ptrCast(want);
    if (std.mem.eql(u32, gb, wb)) return;
    var first: usize = 0;
    while (gb[first] == wb[first]) first += 1;
    std.debug.print("FAIL {s}: the window from decode row {d}: row {d} value {d}: {d} not {d}\n", .{ name, at, first / D, first % D, got[first], want[first] });
    return error.TestExpectedEqual;
}

test "glm fwd: verify windows (2-4 decode rows) and keep 1-4 = the serial decode rows, bit for bit: KDA (state, conv window), DSA (dense, listed with a rejected row completing a pool)" {
    const t = try Rig.open();
    defer t.close();
    const a = t.a();
    var rng: Rng = .{ .state = 0x3e_71f7 };
    // KDA: 20 prompt rows, then 16 decode rows serially / in the windows; the state and conv window after each keep
    {
        const kw = try KdaW.make(t, &rng);
        const cap = 512;
        const pre = 20;
        const x = try bfs(t, &rng, (pre + sched_rows) * D, 1.0);
        const S = heads * dk * dk;
        const C = 3 * 3 * n_kda;
        const sp = try a.alloc(f32, sched_rows * D);
        const srec = try a.alloc(f32, sched_rows * S);
        const sconv = try a.alloc(u16, sched_rows * C);
        {
            var st = try State.init(&t.r, cap);
            defer st.deinit();
            var pb = try Buffers.init(&t.r, pre, true, cap);
            defer pb.deinit();
            var db = try Buffers.init(&t.r, 1, false, cap);
            defer db.deinit();
            var fp = try t.fwd(&pb);
            var fd = try t.fwd(&db);
            try t.put(pb.normed, std.mem.sliceAsBytes(x.bits[0 .. pre * D]));
            try fp.kdaBlock(&st, 0, kw.w, pre);
            for (0..sched_rows) |r| {
                try t.put(db.normed, std.mem.sliceAsBytes(x.bits[(pre + r) * D ..][0..D]));
                try fd.kdaBlock(&st, 0, kw.w, 1);
                @memcpy(sp[r * D ..][0..D], try t.get(f32, db.part, D));
                @memcpy(srec[r * S ..][0..S], try t.get(f32, st.recAt(0), S));
                @memcpy(sconv[r * C ..][0..C], try t.get(u16, st.convAt(0), C));
            }
        }
        var st = try State.init(&t.r, cap);
        defer st.deinit();
        var pb = try Buffers.init(&t.r, pre, true, cap);
        defer pb.deinit();
        var wb = try Buffers.init(&t.r, state.window_rows, false, cap);
        defer wb.deinit();
        var fp = try t.fwd(&pb);
        var fw = try t.fwd(&wb);
        try t.put(pb.normed, std.mem.sliceAsBytes(x.bits[0 .. pre * D]));
        try fp.kdaBlock(&st, 0, kw.w, pre);
        var at: usize = 0;
        for (sched) |rk| {
            const R = rk[0];
            const k = rk[1];
            try t.put(wb.normed, std.mem.sliceAsBytes(x.bits[(pre + at) * D ..][0 .. R * D]));
            fw.hold = true;
            try fw.kdaBlock(&st, 0, kw.w, R);
            fw.hold = false;
            st.pending = R;
            try sameRows("glm fwd KDA window rows", at, try t.get(f32, wb.part, R * D), sp[at * D ..][0 .. R * D]);
            try fw.keep(&st, k);
            at += k;
            if (!std.mem.eql(f32, try t.get(f32, st.recAt(0), S), srec[(at - 1) * S ..][0..S])) {
                std.debug.print("FAIL glm fwd KDA keep {d} of {d}: the state differs from the serial state after decode row {d}\n", .{ k, R, at - 1 });
                return error.TestExpectedEqual;
            }
            if (!std.mem.eql(u16, try t.get(u16, st.convAt(0), C), sconv[(at - 1) * C ..][0..C])) {
                std.debug.print("FAIL glm fwd KDA keep {d} of {d}: the conv window differs from the serial one after decode row {d}\n", .{ k, R, at - 1 });
                return error.TestExpectedEqual;
            }
        }
        std.debug.print("RESULT glm fwd KDA: {d} windows of 2-4 rows, keep 1-4: every window row, the state and the conv window after each keep = serial, bit for bit\n", .{sched.len});
    }
    // DSA: the decode rows from 24 (dense) and from 2,056 (listed: past the 2,051-key limit, pools of 4)
    const dw = try DsaW.make(t, &rng);
    for ([_]usize{ 24, 2056 }) |pre| {
        const cap = std.mem.alignForward(usize, pre + sched_rows + 8, 64);
        const x = try bfs(t, &rng, (pre + sched_rows) * D, 1.0);
        const sp = try a.alloc(f32, sched_rows * D);
        for (0..2) |mode| {
            var st = try State.init(&t.r, cap);
            defer st.deinit();
            var pb = try Buffers.init(&t.r, @min(pre, 2048), true, cap);
            defer pb.deinit();
            var db = try Buffers.init(&t.r, if (mode == 0) 1 else state.window_rows, false, cap);
            defer db.deinit();
            var fp = try t.fwd(&pb);
            var fd = try t.fwd(&db);
            try st.reset(t.s.handle);
            while (st.pos < pre) {
                const n = @min(pb.rows, pre - st.pos);
                try t.put(pb.normed, std.mem.sliceAsBytes(x.bits[st.pos * D ..][0 .. n * D]));
                try st.ready(t.s.handle, n);
                try fp.dsaBlock(st.caches[0], st.pos_dev, st.pos, dw.w, n);
                try st.setPos(t.s.handle, st.pos + n);
            }
            if (mode == 0) {
                for (0..sched_rows) |r| {
                    try t.put(db.normed, std.mem.sliceAsBytes(x.bits[(pre + r) * D ..][0..D]));
                    try st.ready(t.s.handle, 1);
                    try fd.dsaBlock(st.caches[0], st.pos_dev, st.pos, dw.w, 1);
                    @memcpy(sp[r * D ..][0..D], try t.get(f32, db.part, D));
                    try st.setPos(t.s.handle, st.pos + 1);
                }
                continue;
            }
            // a window writes its rows' latent and index rows at [pos, pos + R); a keep moves pos; the next window
            // rewrites the rejected rows
            var at: usize = 0;
            for (sched) |rk| {
                const R = rk[0];
                try t.put(db.normed, std.mem.sliceAsBytes(x.bits[(pre + at) * D ..][0 .. R * D]));
                try st.ready(t.s.handle, R);
                try fd.dsaBlock(st.caches[0], st.pos_dev, st.pos, dw.w, R);
                try sameRows(if (pre < 2051) "glm fwd DSA window rows (dense)" else "glm fwd DSA window rows (listed)", at, try t.get(f32, db.part, R * D), sp[at * D ..][0 .. R * D]);
                at += rk[1];
                try st.setPos(t.s.handle, pre + at);
            }
        }
        std.debug.print("RESULT glm fwd DSA from position {d} ({s}): {d} windows of 2-4 rows, keep 1-4: every window row = serial, bit for bit\n", .{ pre, if (pre < 2051) "dense" else "listed", sched.len });
    }
}

test "glm fwd: decode windows of 2, 3, 4 and 8 rows = the same rows one at a time, bit for bit: the dense MLP, the MoE (router, EXL3 GEMV, shared)" {
    const t = try Rig.open();
    defer t.close();
    const a = t.a();
    var rng: Rng = .{ .state = 0x3e_71f8 };
    const rows = state.window_rows;
    const x = try bfs(t, &rng, rows * D, 1.0);
    const w: usize = 12288;
    const gu = try Lin.make(t, &rng, 2 * w, D, null);
    const down = try Lin.make(t, &rng, D, w, null);
    const dense: weights.Dense = .{ .gu = gu.w, .down = down.w };
    const ex = try Experts.make(t, &rng, 6);
    const router = try Lin.make(t, &rng, 288, D, null);
    const bias = try a.alloc(f32, 288);
    for (bias) |*v| v.* = @floatCast((rng.uniform() - 0.5) * 0.1);
    const sgu = try Lin.make(t, &rng, 2 * 2048, D, null);
    const sdown = try Lin.make(t, &rng, D, 2048, null);
    const moe: weights.Moe = .{ .router = router.w, .bias = try t.up(bias), .shared_gu = sgu.w, .shared_down = sdown.w, .experts = ex.e, .stacks = undefined };
    var one = try Buffers.init(&t.r, 1, false, 64);
    defer one.deinit();
    var win = try Buffers.init(&t.r, rows, false, 64);
    defer win.deinit();
    var f1 = try t.fwd(&one);
    var fw = try t.fwd(&win);
    for (0..2) |blk| {
        const name = if (blk == 0) "glm fwd dense MLP decode window" else "glm fwd MoE decode window";
        const serial = try a.alloc(f32, rows * D);
        for (0..rows) |r| {
            try t.put(one.normed, std.mem.sliceAsBytes(x.bits[r * D ..][0..D]));
            if (blk == 0) try f1.denseBlock(dense, 1) else try f1.moeBlock(moe, 3, 1);
            @memcpy(serial[r * D ..][0..D], try t.get(f32, one.part, D));
        }
        for ([_]usize{ 2, 3, 4, 8 }) |R| {
            try t.put(win.normed, std.mem.sliceAsBytes(x.bits[0 .. R * D]));
            if (blk == 0) try fw.denseBlock(dense, R) else try fw.moeBlock(moe, 3, R);
            try sameRows(name, 0, try t.get(f32, win.part, R * D), serial[0 .. R * D]);
        }
        std.debug.print("RESULT {s}: 2, 3, 4 and 8 rows = one at a time, bit for bit\n", .{name});
    }
}

// ---- the DSA block on real weights, op by op (TF_GLM_DIR: the checkpoint; TF_GLM_FWD_DUMP: a TF_GLM_DUMP prompt dir) ----

/// A checkpoint's tensors by name through model.safetensors.index.json (the shards mapped on first use).
const Shards = struct {
    dir: []const u8,
    index: std.json.Parsed(std.json.Value),
    files: std.StringHashMapUnmanaged(core.safetensors.File) = .empty,

    fn open(a: std.mem.Allocator, dir: []const u8) !Shards {
        const text = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" }), a, .limited(1 << 28));
        return .{ .dir = dir, .index = try std.json.parseFromSlice(std.json.Value, a, text, .{}) };
    }

    fn get(s: *Shards, a: std.mem.Allocator, name: []const u8) !core.safetensors.Tensor {
        const shard = (s.index.value.object.get("weight_map").?.object.get(name) orelse return error.NoTensor).string;
        const f = try s.files.getOrPut(a, shard);
        if (!f.found_existing) f.value_ptr.* = try core.safetensors.File.open(a, std.testing.io, try std.fs.path.join(a, &.{ s.dir, shard }));
        return f.value_ptr.get(name) orelse error.NoTensor;
    }

    /// Tensors one after another in one device buffer (G0's pieces: a row stack when they share K).
    fn stack(s: *Shards, t: *Rig, names: []const []const u8) !u64 {
        const a = t.a();
        var bytes: std.ArrayList(u8) = .empty;
        for (names) |n| try bytes.appendSlice(a, (try s.get(a, try std.fmt.allocPrint(a, "model.language_model.{s}", .{n}))).bytes);
        return t.up(bytes.items);
    }
};

fn cat(a: std.mem.Allocator, p: []const u8, s: []const u8) []const u8 {
    return std.fmt.allocPrint(a, "{s}{s}", .{ p, s }) catch unreachable;
}

fn oneTensor(sh: *Shards, t: *Rig, p: []const u8, name: []const u8) !u64 {
    return sh.stack(t, &.{cat(t.a(), p, name)});
}

fn writeNpy(a: std.mem.Allocator, dir: []const u8, name: []const u8, comptime T: type, shape: []const usize, data: []const T) !void {
    const dump = @import("dump.zig");
    // bf16 bit patterns as int16 (the writer has no u16 descr; the reader views them back as uint16)
    const bytes = if (T == u16) try dump.npyBytes(a, i16, shape, std.mem.bytesAsSlice(i16, std.mem.sliceAsBytes(data))) else try dump.npyBytes(a, T, shape, data);
    const path = try std.fs.path.join(a, &.{ dir, name });
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = path, .data = bytes });
}

test "glm fwd: layer 3's DSA block on the checkpoint's weights from a dumped mixer input, every intermediate written for the float64 op check" {
    const dir = std.mem.span(std.c.getenv("TF_GLM_DIR") orelse return error.SkipZigTest);
    const dump_dir = std.mem.span(std.c.getenv("TF_GLM_FWD_DUMP") orelse return error.SkipZigTest);
    const out_dir = std.mem.span(std.c.getenv("TF_GLM_FWD_OUT") orelse return error.SkipZigTest);
    const t = try Rig.open();
    defer t.close();
    const a = t.a();
    const L = 3;
    const R = 64;
    var sh = try Shards.open(a, dir);
    const pre = try std.fmt.allocPrint(a, "layers.{d}.self_attn.", .{L});
    const c: config.Config = .{};
    const xw = try sh.stack(t, &.{ cat(a, pre, "q_a_proj.weight"), cat(a, pre, "kv_a_proj_with_mqa.weight"), cat(a, pre, "indexer.wk.weight"), cat(a, pre, "indexer.weights_proj.weight") });
    const qr = try sh.stack(t, &.{ cat(a, pre, "q_b_proj.weight"), cat(a, pre, "indexer.wq_b.weight") });
    const kvb = try oneTensor(&sh, t, pre, "kv_b_proj.weight");
    const sub_x = weights.Bf16{ .w = xw, .n = 1, .k = D };
    const w: weights.Mla = .{
        .x = .{ .w = xw, .n = c.xProj(), .k = D },
        .q_a = sub_x,
        .kv_a = sub_x,
        .qr = .{ .w = qr, .n = c.qrProj(), .k = c.q_lora },
        .q_b = .{ .w = qr, .n = 1, .k = c.q_lora },
        .q_norm = try oneTensor(&sh, t, pre, "q_a_layernorm.weight"),
        .kv_norm = try oneTensor(&sh, t, pre, "kv_a_layernorm.weight"),
        .kv_b = .{ .w = kvb, .n = 32768, .k = 512 },
        .o = .{ .w = try oneTensor(&sh, t, pre, "o_proj.weight"), .n = D, .k = 16384 },
        .index = .{
            .wk = sub_x,
            .weights = sub_x,
            .wq_b = sub_x,
            .ln_w = try oneTensor(&sh, t, pre, "indexer.k_norm.weight"),
            .ln_b = try oneTensor(&sh, t, pre, "indexer.k_norm.bias"),
            .gate = .{ .w = try oneTensor(&sh, t, pre, "indexer.index_kpool_compress_gate"), .n = 128, .k = D },
            .ape = try oneTensor(&sh, t, pre, "indexer.index_kpool_compress_ape"),
        },
    };
    // the dumped mixer input of layer L (sublayers [L][j][0], j = positions 0..63): bf16 values held as fp32
    const sub_bytes = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, try std.fs.path.join(a, &.{ dump_dir, "sublayers.npy" }), a, .limited(1 << 32));
    const arr = try core.npy.parse(sub_bytes);
    const P = arr.shape[1];
    const vals = std.mem.bytesAsSlice(f32, arr.data);
    const xin = try a.alloc(u16, R * D);
    for (0..R) |j| for (0..D) |k| {
        const v: u32 = @bitCast(vals[((L * P + j) * 4 + 0) * D + k]);
        if (v & 0xffff != 0) return error.NotBf16;
        xin[j * D + k] = @intCast(v >> 16);
    };
    var st = try State.init(&t.r, 128);
    defer st.deinit();
    try st.reset(t.s.handle);
    var b = try Buffers.init(&t.r, R, true, 128);
    defer b.deinit();
    var f = try t.fwd(&b);
    try t.put(b.normed, std.mem.sliceAsBytes(xin));
    try st.ready(t.s.handle, R);
    try f.dsaBlock(st.caches[0], st.pos_dev, 0, w, R);
    const d = f.dsa_last;
    const H: usize = 64;
    try std.Io.Dir.cwd().createDirPath(std.testing.io, out_dir);
    try writeNpy(a, out_dir, "x.npy", u16, &.{ R, D }, xin);
    try writeNpy(a, out_dir, "dp.npy", u16, &.{ R, c.xProj() }, try t.get(u16, d.dp, R * c.xProj()));
    try writeNpy(a, out_dir, "qr.npy", u16, &.{ R, c.q_lora }, try t.get(u16, d.qr, R * c.q_lora));
    try writeNpy(a, out_dir, "lat.npy", u16, &.{ R, 512 }, try t.get(u16, d.lat, R * 512));
    try writeNpy(a, out_dir, "cache.npy", u16, &.{ R, 512 }, try t.get(u16, st.caches[0].lc, R * 512));
    try writeNpy(a, out_dir, "qq.npy", u16, &.{ R, c.qrProj() }, try t.get(u16, d.qq, R * c.qrProj()));
    try writeNpy(a, out_dir, "qa.npy", u16, &.{ R, H, 512 }, try t.get(u16, d.qa, R * H * 512));
    try writeNpy(a, out_dir, "ol.npy", u16, &.{ R, H, 512 }, try t.get(u16, d.ol, R * H * 512));
    try writeNpy(a, out_dir, "vn.npy", u16, &.{ R, H * 256 }, try t.get(u16, d.vn, R * H * 256));
    try writeNpy(a, out_dir, "part.npy", f32, &.{ R, D }, try t.get(f32, b.part, R * D));
    std.debug.print("RESULT glm fwd DSA layer {d} on real weights: {d} rows, intermediates in {s}\n", .{ L, R, out_dir });
}
