//! GLM-5.3-Flash's KDA on HIP against float64 host references: the window conv split and the prompt prep with a decay
//! a key channel, the scan (tile, prompt, hold and keep), and the chain to the gated norm against the float64 chain of
//! tests/cuda/glm_ref_kda.zig (fp32 between kernels, so no bf16 rounding of q, k, v, beta and y). GLM-5.3-Flash on one
//! device: 64 heads x 128, conv 4 taps, gate_lower_bound -5.
const std = @import("std");
const rig = @import("rig.zig");
const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;
const DeviceBuffer = @import("../memory.zig").DeviceBuffer;
const Launcher = @import("../launches.zig").Launcher;

const d = Launcher.kda_dim;
const taps = 4;
const lower: f32 = -5.0;
const glm_heads = 64;

fn sigmoid(x: f64) f64 {
    return 1 / (1 + @exp(-x));
}

/// The largest error over the reference's largest value.
fn relative(got: []const f32, want: []const f64) f64 {
    var scale: f64 = 0;
    var most: f64 = 0;
    for (got, want) |g, w| {
        most = @max(most, @abs(@as(f64, g) - w));
        scale = @max(scale, @abs(w));
    }
    return most / scale;
}

fn widen(xs: []const f32) ![]f64 {
    const out = try gpa.alloc(f64, xs.len);
    for (out, xs) |*o, x| o.* = x;
    return out;
}

/// The L2 norms of q and k as the RMS norm with a weight (flashnext mixers.zig qkConstants): x rsqrt(sum + 1e-6), q
/// also by 128^-0.5. Weights [q x 128 | k x 128].
const qk = struct {
    const inv = 1 / @sqrt(@as(f64, d));
    const q: f32 = @floatCast(inv * inv);
    const k: f32 = @floatCast(inv);
    const eps: f32 = @floatCast(1e-6 * inv * inv);

    fn weights() [2 * d]f32 {
        var w: [2 * d]f32 = undefined;
        @memset(w[0..d], q);
        @memset(w[d..], k);
        return w;
    }
};

/// Device buffers freed together.
const Held = struct {
    list: std.ArrayList(DeviceBuffer) = .empty,

    fn up(h: *Held, t: *Rig, host: anytype) !DeviceBuffer {
        var b = try t.upload(host);
        errdefer b.free();
        try h.list.append(gpa, b);
        return b;
    }

    fn zeros(h: *Held, t: *Rig, bytes: usize) !DeviceBuffer {
        var b = try t.alloc(bytes);
        errdefer b.free();
        try b.fill8(0);
        try h.list.append(gpa, b);
        return b;
    }

    fn deinit(h: *Held) void {
        for (h.list.items) |*b| b.free();
        h.list.deinit(gpa);
    }
};

fn cf(addr: u64) ?[*]const f32 {
    return @ptrFromInt(addr);
}

fn mf(addr: u64) ?[*]f32 {
    return @ptrFromInt(addr);
}

fn cv(addr: u64) ?*const anyopaque {
    return @ptrFromInt(addr);
}

/// One KDA layer's conv and gate inputs: x (rows, q | k | v of heads x 128) and the decay rows a (rows, heads x 128) and
/// beta logits b (rows, heads) in the activation kind; taps fp32 (channels, 4); the conv tail fp32 (3, channels);
/// A_log a head, dt_bias a channel.
const ConvIn = struct {
    heads: usize,
    rows: usize,
    x: []u16,
    a: []u16,
    b: []u16,
    w: []f32,
    tail: []f32,
    a_log: []f32,
    dt_bias: []f32,

    fn make(t: *const Rig, heads: usize, rows: usize, seed: u64) !ConvIn {
        var rng: rig.Rng = .{ .state = seed };
        const n = heads * d;
        const in: ConvIn = .{
            .heads = heads,
            .rows = rows,
            .x = try gpa.alloc(u16, rows * 3 * n),
            .a = try gpa.alloc(u16, rows * n),
            .b = try gpa.alloc(u16, rows * heads),
            .w = try gpa.alloc(f32, 3 * n * taps),
            .tail = try gpa.alloc(f32, (taps - 1) * 3 * n),
            .a_log = try gpa.alloc(f32, heads),
            .dt_bias = try gpa.alloc(f32, n),
        };
        for (in.x) |*v| v.* = t.bits(rng.unit() * 2.0);
        for (in.a) |*v| v.* = t.bits(rng.unit() * 3.0);
        for (in.b) |*v| v.* = t.bits(rng.unit() * 3.0);
        for (in.w) |*v| v.* = rng.unit() / 2.0;
        for (in.tail) |*v| v.* = rng.unit();
        for (in.a_log) |*v| v.* = rng.unit();
        for (in.dt_bias) |*v| v.* = rng.unit() / 2.0;
        return in;
    }

    fn free(in: ConvIn) void {
        gpa.free(in.x);
        gpa.free(in.a);
        gpa.free(in.b);
        gpa.free(in.w);
        gpa.free(in.tail);
        gpa.free(in.a_log);
        gpa.free(in.dt_bias);
    }

    /// Conv input t of channel c: the tail's rows, then x's (no tail: zeros).
    fn window(in: ConvIn, t: *const Rig, with_tail: bool, row: usize, c: usize) f64 {
        const ch = 3 * in.heads * d;
        if (row < taps - 1) return if (with_tail) in.tail[row * ch + c] else 0;
        return t.value(in.x[(row - (taps - 1)) * ch + c]);
    }
};

/// The scan's inputs a row: q and k (heads x 128, q L2 1/sqrt(128), k L2 1), v (heads x 128), decays (heads x 128), beta.
const Prep = struct {
    q: []f64,
    k: []f64,
    v: []f64,
    gate: []f64,
    beta: []f64,

    fn free(p: Prep) void {
        for ([_][]f64{ p.q, p.k, p.v, p.gate, p.beta }) |x| gpa.free(x);
    }
};

/// float64 conv + SiLU, the L2 norms of q and k, the decays exp(lower sigmoid(exp(A_log) (a + dt_bias))), beta.
fn prepRef(t: *const Rig, in: ConvIn, with_tail: bool) !Prep {
    const heads = in.heads;
    const n = heads * d;
    const ch = 3 * n;
    const rows = in.rows;
    const p: Prep = .{
        .q = try gpa.alloc(f64, rows * n),
        .k = try gpa.alloc(f64, rows * n),
        .v = try gpa.alloc(f64, rows * n),
        .gate = try gpa.alloc(f64, rows * n),
        .beta = try gpa.alloc(f64, rows * heads),
    };
    const act = try gpa.alloc(f64, ch);
    defer gpa.free(act);
    for (0..rows) |r| {
        for (act, 0..) |*o, c| {
            var acc: f64 = 0;
            for (0..taps) |tap| acc += @as(f64, in.w[c * taps + tap]) * in.window(t, with_tail, r + tap, c);
            o.* = acc * sigmoid(acc);
        }
        for (0..heads) |h| {
            for ([_][]f64{ p.q, p.k }, 0..) |dst, part| {
                const src = act[part * n + h * d ..][0..d];
                var ss: f64 = 0;
                for (src) |e| ss += e * e;
                const extra: f64 = if (part == 0) 1 / @sqrt(@as(f64, d)) else 1;
                const inv = 1 / @sqrt(ss + 1e-6) * extra;
                for (dst[r * n + h * d ..][0..d], src) |*o, e| o.* = e * inv;
            }
            @memcpy(p.v[r * n + h * d ..][0..d], act[2 * n + h * d ..][0..d]);
            const rate = @exp(@as(f64, in.a_log[h]));
            for (0..d) |i| {
                const c = h * d + i;
                p.gate[r * n + c] = @exp(@as(f64, lower) * sigmoid(rate * (t.value(in.a[r * n + c]) + in.dt_bias[c])));
            }
            p.beta[r * heads + h] = sigmoid(t.value(in.b[r * heads + h]));
        }
    }
    return p;
}

/// float64 delta rule with a decay a key channel: state (heads, 128 v, 128 k) in place, y (rows, heads, 128).
fn scanRef(heads: usize, rows: usize, q: []const f64, k: []const f64, v: []const f64, gate: []const f64, beta: []const f64, state: []f64, y: []f64) void {
    for (0..heads) |h| {
        const s = state[h * d * d ..][0 .. d * d];
        for (0..rows) |t| {
            const at_ = (t * heads + h) * d;
            const g = gate[at_..][0..d];
            const kk = k[at_..][0..d];
            const qq = q[at_..][0..d];
            const b = beta[t * heads + h];
            for (0..d) |r| {
                const row = s[r * d ..][0..d];
                var kv: f64 = 0;
                for (row, g, kk) |*e, gi, ki| {
                    e.* *= gi;
                    kv += e.* * ki;
                }
                const delta = (v[at_ + r] - kv) * b;
                var out: f64 = 0;
                for (row, kk, qq) |*e, ki, qi| {
                    e.* += ki * delta;
                    out += e.* * qi;
                }
                y[at_ + r] = out;
            }
        }
    }
}

/// Weights and rows the conv split and the prompt prep read, on the device.
const ConvDev = struct { x: DeviceBuffer, a: DeviceBuffer, b: DeviceBuffer, w: DeviceBuffer, a_log: DeviceBuffer, dt_bias: DeviceBuffer, norm: DeviceBuffer };

fn convDev(h: *Held, t: *Rig, in: ConvIn) !ConvDev {
    const norm = qk.weights();
    return .{
        .x = try h.up(t, in.x),
        .a = try h.up(t, in.a),
        .b = try h.up(t, in.b),
        .w = try h.up(t, in.w),
        .a_log = try h.up(t, in.a_log),
        .dt_bias = try h.up(t, in.dt_bias),
        .norm = try h.up(t, &norm),
    };
}

/// Where a conv writes: q, k, v, decays (rows, heads x 128) and beta (rows, heads).
const Split = struct { q: u64, k: u64, v: u64, gate: u64, beta: u64 };

/// A window's conv split of in.rows rows with the KDA gate (conv tail `state` in place; `states`: each row's tail).
fn convSplit(t: *Rig, in: ConvIn, c: ConvDev, state: u64, states: u64, out: Split) !void {
    const n = in.heads * d;
    try t.on.tf_conv_split(.{
        .x = at(c.x),
        .kind = t.kind(),
        .weight = at(c.w),
        .state = state,
        .states = states,
        .qn = out.q,
        .kn = out.k,
        .v = out.v,
        .channels = @intCast(3 * n),
        .kernel = taps,
        .rows = @intCast(in.rows),
        .kw = @intCast(n),
        .vw = @intCast(n),
        .qw = at(c.norm),
        .kw_w = at(c.norm) + d * 4,
        .eps = qk.eps,
        .norm = 128,
        .ga = at(c.a),
        .gb = at(c.b),
        .a_log = at(c.a_log),
        .dt_bias = at(c.dt_bias),
        .gate = out.gate,
        .beta = out.beta,
        .gcount = @intCast(in.rows * in.heads),
        .heads = @intCast(in.heads),
        .gate_dim = d,
        .lower = lower,
    }, t.stream.handle);
}

/// A prompt's KDA prework of in.rows rows (conv tail `state`, 0: none; the next tail into `new_state`).
fn promptPrep(t: *Rig, in: ConvIn, c: ConvDev, state: u64, new_state: u64, out: Split) !void {
    try t.on.tf_kda_prompt_prep(cv(at(c.x)), t.kind(), cf(at(c.w)), cf(state), mf(new_state), mf(out.q), mf(out.k), mf(out.v), cf(at(c.norm)), cf(at(c.norm) + d * 4), qk.eps, @intCast(in.rows), @intCast(in.heads), taps, cv(at(c.a)), cv(at(c.b)), cf(at(c.a_log)), cf(at(c.dt_bias)), mf(out.gate), mf(out.beta), lower, t.stream.handle);
}

/// `count` values at a device address inside a held buffer.
fn download(comptime T: type, t: *Rig, addr: u64, count: usize) ![]T {
    const view: DeviceBuffer = .{ .r = &t.r, .ptr = @ptrFromInt(addr), .len = count * @sizeOf(T) };
    return rig.download(T, view);
}

/// q, k, v, decays and beta at `out` against the float64 prework, each within 1e-5 of its largest value.
fn expectPrep(t: *Rig, out: Split, want: Prep, what: []const u8, rows: usize) !void {
    const names = [_][]const u8{ "q", "k", "v", "decay", "beta" };
    for ([_]u64{ out.q, out.k, out.v, out.gate, out.beta }, [_][]f64{ want.q, want.k, want.v, want.gate, want.beta }, names) |addr, ref, name| {
        const got = try download(f32, t, addr, ref.len);
        defer gpa.free(got);
        const e = relative(got, ref);
        std.debug.print("KDA {s} rows={d}: {s} relative error {e:.3}\n", .{ what, rows, name, e });
        try rig.expect(e < 1e-5, "KDA {s} rows={d}: {s} relative error {e}", .{ what, rows, name, e });
    }
}

/// The conv tail after `rows` rows (the last 3 inputs a channel), or after each row r of a window (`snaps`), exact.
fn expectTail(t: *Rig, in: ConvIn, with_tail: bool, addr: u64, after: usize) !void {
    const ch = 3 * in.heads * d;
    const got = try download(f32, t, addr, (taps - 1) * ch);
    defer gpa.free(got);
    for (0..taps - 1) |tap| for (0..ch) |c| {
        const want: f32 = @floatCast(in.window(t, with_tail, after + tap, c));
        try rig.expect(got[tap * ch + c] == want, "KDA conv tail after {d} rows: tap {d} channel {d}: {d} not {d}", .{ after, tap, c, got[tap * ch + c], want });
    };
}

test "KDA: a window's conv split and a prompt's prework write float64's q, k, v, decays a channel and beta" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    const heads = glm_heads;
    const n = heads * d;
    const ch = 3 * n;
    // windows: one row (the plain decode), and verify windows of 3 and 8 rows with each row's conv tail kept aside
    for ([_]usize{ 1, 3, 8 }) |rows| {
        const in = try ConvIn.make(t, heads, rows, 0x4bda + rows);
        defer in.free();
        const want = try prepRef(t, in, true);
        defer want.free();
        var h: Held = .{};
        defer h.deinit();
        const c = try convDev(&h, t, in);
        const tail = try h.up(t, in.tail);
        const snaps = try h.zeros(t, rows * (taps - 1) * ch * 4);
        const out: Split = .{
            .q = at(try h.zeros(t, rows * n * 4)),
            .k = at(try h.zeros(t, rows * n * 4)),
            .v = at(try h.zeros(t, rows * n * 4)),
            .gate = at(try h.zeros(t, rows * n * 4)),
            .beta = at(try h.zeros(t, rows * heads * 4)),
        };
        try convSplit(t, in, c, at(tail), if (rows > 1) at(snaps) else 0, out);
        try t.stream.synchronize();
        try expectPrep(t, out, want, "conv split", rows);
        try expectTail(t, in, true, at(tail), rows);
        if (rows > 1) for (0..rows) |r| try expectTail(t, in, true, at(snaps) + r * (taps - 1) * ch * 4, r + 1);
    }
    // prompts, from a conv tail and from none (a sequence's first chunk)
    for ([_]usize{ 1, 70, 300 }) |rows| for ([_]bool{ true, false }) |with_tail| {
        const in = try ConvIn.make(t, heads, rows, 0x9a7e + rows);
        defer in.free();
        const want = try prepRef(t, in, with_tail);
        defer want.free();
        var h: Held = .{};
        defer h.deinit();
        const c = try convDev(&h, t, in);
        const tail = try h.up(t, in.tail);
        const next = try h.zeros(t, (taps - 1) * ch * 4);
        const out: Split = .{
            .q = at(try h.zeros(t, rows * n * 4)),
            .k = at(try h.zeros(t, rows * n * 4)),
            .v = at(try h.zeros(t, rows * n * 4)),
            .gate = at(try h.zeros(t, rows * n * 4)),
            .beta = at(try h.zeros(t, rows * heads * 4)),
        };
        try promptPrep(t, in, c, if (with_tail) at(tail) else 0, at(next), out);
        try t.stream.synchronize();
        try expectPrep(t, out, want, if (with_tail) "prompt prep" else "prompt prep (no tail)", rows);
        try expectTail(t, in, with_tail, at(next), rows);
    };
}

/// The scan's inputs: q and k L2-normed (q to 128^-0.5), v normal, decays exp(-5 sigmoid(2 z)), beta sigmoid(z).
const ScanIn = struct {
    heads: usize,
    rows: usize,
    q: []f32,
    k: []f32,
    v: []f32,
    gate: []f32,
    beta: []f32,
    state: []f32,

    fn make(heads: usize, rows: usize, seed: u64) !ScanIn {
        var rng: rig.Rng = .{ .state = seed };
        const n = heads * d;
        const in: ScanIn = .{
            .heads = heads,
            .rows = rows,
            .q = try gpa.alloc(f32, rows * n),
            .k = try gpa.alloc(f32, rows * n),
            .v = try gpa.alloc(f32, rows * n),
            .gate = try gpa.alloc(f32, rows * n),
            .beta = try gpa.alloc(f32, rows * heads),
            .state = try gpa.alloc(f32, heads * d * d),
        };
        for ([_][]f32{ in.q, in.k }, [_]f64{ 1 / @sqrt(@as(f64, d)), 1 }) |a, scale| {
            for (0..a.len / d) |row| {
                var ss: f64 = 0;
                for (a[row * d ..][0..d]) |*x| {
                    const z = rng.normal();
                    x.* = @floatCast(z);
                    ss += z * z;
                }
                const inv: f32 = @floatCast(scale / @sqrt(ss));
                for (a[row * d ..][0..d]) |*x| x.* *= inv;
            }
        }
        for (in.v) |*x| x.* = @floatCast(rng.normal());
        for (in.gate) |*g| g.* = @floatCast(@exp(@as(f64, lower) * sigmoid(2 * rng.normal())));
        for (in.beta) |*b| b.* = @floatCast(sigmoid(rng.normal()));
        for (in.state) |*x| x.* = @floatCast(0.1 * rng.normal());
        return in;
    }

    fn free(in: ScanIn) void {
        for ([_][]f32{ in.q, in.k, in.v, in.gate, in.beta, in.state }) |a| gpa.free(a);
    }
};

/// The scan's inputs on the device: q, k, v, gate, beta.
fn scanDev(h: *Held, t: *Rig, in: ScanIn) ![5]DeviceBuffer {
    return .{ try h.up(t, in.q), try h.up(t, in.k), try h.up(t, in.v), try h.up(t, in.gate), try h.up(t, in.beta) };
}

/// tf_kda over rows [from, from + rows) of `in` (state in place; `snaps`: each row's state, 0: none).
fn scan(t: *Rig, heads: usize, in: [5]DeviceBuffer, state: u64, y: u64, from: usize, rows: usize, snaps: u64) !void {
    const n = heads * d;
    const row = from * n * 4;
    try t.on.tf_kda(cf(at(in[0]) + row), cf(at(in[1]) + row), cf(at(in[2]) + row), cf(at(in[3]) + row), cf(at(in[4]) + from * heads * 4), mf(state), mf(y + row), @intCast(rows), @intCast(heads), t.stream.handle, mf(snaps), 0);
}

test "KDA: the scan (tile under 64 rows, prompt blocks from 64) stays within 5e-3 of the float64 recurrence" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    const Shape = struct { rows: usize, heads: usize };
    for ([_]Shape{
        .{ .rows = 1, .heads = glm_heads },
        .{ .rows = 3, .heads = glm_heads },
        .{ .rows = 8, .heads = glm_heads },
        .{ .rows = 63, .heads = glm_heads },
        .{ .rows = 64, .heads = glm_heads },
        .{ .rows = 300, .heads = glm_heads },
        .{ .rows = 1000, .heads = 4 },
        .{ .rows = 4170, .heads = 4 },
    }) |sh| {
        const in = try ScanIn.make(sh.heads, sh.rows, 0x6b6461 + sh.rows);
        defer in.free();
        const n = sh.heads * d;
        const ry = try gpa.alloc(f64, sh.rows * n);
        defer gpa.free(ry);
        const rs = try widen(in.state);
        defer gpa.free(rs);
        {
            const w = [_][]f64{ try widen(in.q), try widen(in.k), try widen(in.v), try widen(in.gate), try widen(in.beta) };
            defer for (w) |x| gpa.free(x);
            scanRef(sh.heads, sh.rows, w[0], w[1], w[2], w[3], w[4], rs, ry);
        }
        var h: Held = .{};
        defer h.deinit();
        const dev = try scanDev(&h, t, in);
        const state = try h.up(t, in.state);
        const y = try h.zeros(t, sh.rows * n * 4);
        try scan(t, sh.heads, dev, at(state), at(y), 0, sh.rows, 0);
        try t.stream.synchronize();
        const gy = try rig.download(f32, y);
        defer gpa.free(gy);
        const gs = try rig.download(f32, state);
        defer gpa.free(gs);
        const ey = relative(gy, ry);
        const es = relative(gs, rs);
        std.debug.print("KDA scan rows={d} heads={d}: y relative error {e:.3}, state {e:.3}\n", .{ sh.rows, sh.heads, ey, es });
        try rig.expect(ey < 5e-3 and es < 5e-3, "KDA scan rows={d}: y {e}, state {e}", .{ sh.rows, ey, es });
    }
}

test "KDA: the prompt blocks give the tile's bits, and a prompt cut anywhere gives one call's bits" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    const heads = glm_heads;
    const rows = 200;
    const in = try ScanIn.make(heads, rows, 0xc07);
    defer in.free();
    var h: Held = .{};
    defer h.deinit();
    const dev = try scanDev(&h, t, in);
    const state = try h.up(t, in.state);
    const y = try h.zeros(t, rows * heads * d * 4);
    // one call: the prompt blocks
    try scan(t, heads, dev, at(state), at(y), 0, rows, 0);
    try t.stream.synchronize();
    const want_y = try rig.download(u8, y);
    defer gpa.free(want_y);
    const want_s = try rig.download(u8, state);
    defer gpa.free(want_s);
    // cuts: [0, c) and [c, 200) (either side of 64 rows), and 50-row calls (the tile alone)
    const Cut = struct { at: []const usize };
    for ([_]Cut{ .{ .at = &.{1} }, .{ .at = &.{63} }, .{ .at = &.{64} }, .{ .at = &.{65} }, .{ .at = &.{136} }, .{ .at = &.{ 50, 100, 150 } } }) |cut| {
        try state.upload(0, std.mem.sliceAsBytes(in.state));
        try y.fill8(0);
        var from: usize = 0;
        for (cut.at) |c| {
            try scan(t, heads, dev, at(state), at(y), from, c - from, 0);
            from = c;
        }
        try scan(t, heads, dev, at(state), at(y), from, rows - from, 0);
        try t.stream.synchronize();
        const gy = try rig.download(u8, y);
        defer gpa.free(gy);
        const gs = try rig.download(u8, state);
        defer gpa.free(gs);
        try rig.expect(std.mem.eql(u8, want_y, gy) and std.mem.eql(u8, want_s, gs), "KDA cut at {any}: y or state differs from one call", .{cut.at});
    }
}

/// Rows a keep's inputs hold a layer: a verify window's most (Flash Next's cache.max_window; GLM drafts at most 3).
const keep_rows = 8;

/// Layer inputs as a window's conv split writes them for its keep: keep_rows rows of k, then of v, of decays and of beta;
/// the rows past `rows` NaN, so a keep that read one would show it.
fn keepInputs(in: ScanIn, rows: usize, out: []f32) void {
    @memset(out, std.math.nan(f32));
    const n = in.heads * d;
    @memcpy(out[0 .. rows * n], in.k[0 .. rows * n]);
    @memcpy(out[keep_rows * n ..][0 .. rows * n], in.v[0 .. rows * n]);
    @memcpy(out[2 * keep_rows * n ..][0 .. rows * n], in.gate[0 .. rows * n]);
    @memcpy(out[3 * keep_rows * n ..][0 .. rows * in.heads], in.beta[0 .. rows * in.heads]);
}

test "KDA: a held window and its keep give the window's outputs, and the state after its row kept - 1" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    const heads = glm_heads;
    const n = heads * d;
    const layers = 2;
    var ins: [layers]ScanIn = undefined;
    for (&ins, 0..) |*in, z| in.* = try ScanIn.make(heads, keep_rows, 0x4ee9 + z);
    defer for (ins) |in| in.free();
    const state_floats = heads * d * d;
    const state_bytes = state_floats * 4;
    const state_stride = state_floats + 64;
    const in_stride = keep_rows * (3 * n + heads);
    for (1..keep_rows + 1) |rows| {
        // the window as a plain scan: its outputs and each row's state
        var want_y: [layers][]u8 = undefined;
        var want_snaps: [layers][]u8 = undefined;
        for (0..layers) |z| {
            var h: Held = .{};
            defer h.deinit();
            const dev = try scanDev(&h, t, ins[z]);
            const state = try h.up(t, ins[z].state);
            const y = try h.zeros(t, rows * n * 4);
            const snaps = try h.zeros(t, rows * state_bytes);
            try scan(t, heads, dev, at(state), at(y), 0, rows, at(snaps));
            try t.stream.synchronize();
            want_y[z] = try rig.download(u8, y);
            want_snaps[z] = try rig.download(u8, snaps);
        }
        defer for (want_y, want_snaps) |a, b| {
            gpa.free(a);
            gpa.free(b);
        };
        // the held window: the same outputs, the state untouched; one row: the plain scan stores the snapshot's bits
        for (0..layers) |z| {
            var h: Held = .{};
            defer h.deinit();
            const dev = try scanDev(&h, t, ins[z]);
            const state = try h.up(t, ins[z].state);
            const y = try h.zeros(t, rows * n * 4);
            try t.on.tf_kda_hold(cf(at(dev[0])), cf(at(dev[1])), cf(at(dev[2])), cf(at(dev[3])), cf(at(dev[4])), mf(at(state)), mf(at(y)), @intCast(rows), heads, t.stream.handle);
            try t.stream.synchronize();
            const got_y = try rig.download(u8, y);
            defer gpa.free(got_y);
            try rig.expect(std.mem.eql(u8, want_y[z], got_y), "KDA hold rows={d} layer={d}: outputs differ", .{ rows, z });
            const held = try rig.download(f32, state);
            defer gpa.free(held);
            try std.testing.expectEqualSlices(u32, @ptrCast(ins[z].state), @ptrCast(held));
            if (rows == 1) {
                try scan(t, heads, dev, at(state), at(y), 0, 1, 0);
                try t.stream.synchronize();
                const stored = try rig.download(u8, state);
                defer gpa.free(stored);
                try rig.expect(std.mem.eql(u8, want_snaps[z][0..state_bytes], stored), "KDA one-row window: state differs from the snapshot", .{});
            }
        }
        // the keep of every count a window of `rows` can keep (1: a full reject, the pending token alone)
        const packed_in = try gpa.alloc(f32, layers * in_stride);
        defer gpa.free(packed_in);
        for (0..layers) |z| keepInputs(ins[z], rows, packed_in[z * in_stride ..][0..in_stride]);
        const states = try gpa.alloc(f32, layers * state_stride);
        defer gpa.free(states);
        for (1..rows + 1) |kept| {
            @memset(states, std.math.nan(f32));
            for (0..layers) |z| @memcpy(states[z * state_stride ..][0..state_floats], ins[z].state);
            var h: Held = .{};
            defer h.deinit();
            const inputs = try h.up(t, packed_in);
            const state = try h.up(t, states);
            try t.on.tf_kda_keep(mf(at(state)), @intCast(state_stride), cf(at(inputs)), @intCast(in_stride), keep_rows, @intCast(kept), layers, heads, t.stream.handle);
            try t.stream.synchronize();
            const got = try rig.download(u8, state);
            defer gpa.free(got);
            for (0..layers) |z| {
                const want = want_snaps[z][(kept - 1) * state_bytes ..][0..state_bytes];
                try rig.expect(std.mem.eql(u8, want, got[z * state_stride * 4 ..][0..state_bytes]), "KDA keep rows={d} kept={d} layer={d}: state differs from the snapshot", .{ rows, kept, z });
                const gap = (z * state_stride + state_floats) * 4;
                const gap_bytes = (state_stride - state_floats) * 4;
                try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(states)[gap..][0..gap_bytes], got[gap..][0..gap_bytes]);
            }
        }
    }
}

/// The chain's output: rms(y) o_norm sigmoid(g) a (row, head) in float64, as glm_ref_kda.zig's reference ends.
fn gnormRef(t: *const Rig, heads: usize, rows: usize, y: []const f64, z: []const u16, norm_w: []const f32, eps: f64, out: []f64) void {
    for (0..rows * heads) |rh| {
        const yy = y[rh * d ..][0..d];
        var ss: f64 = 0;
        for (yy) |e| ss += e * e;
        const rinv = 1 / @sqrt(ss / d + eps);
        for (out[rh * d ..][0..d], yy, 0..) |*o, e, i| o.* = @as(f64, norm_w[i]) * e * rinv * sigmoid(t.value(z[rh * d + i]));
    }
}

/// Out (activation kind) against float64 within its rounding (2^-8 of the value) and 2^-8 of the row's largest value,
/// as glm_ref_kda.zig's tolerance.
fn expectOut(t: *Rig, out: DeviceBuffer, want: []const f64, n: usize, what: []const u8) !void {
    const got = try rig.download(u16, out);
    defer gpa.free(got);
    var worst: f64 = 0;
    for (0..want.len / n) |row| {
        var most: f64 = 0;
        for (want[row * n ..][0..n]) |e| most = @max(most, @abs(e));
        for (got[row * n ..][0..n], want[row * n ..][0..n], 0..) |g, e, c| {
            const tol = @abs(e) * std.math.pow(f64, 2, -8) + most * std.math.pow(f64, 2, -8) + 1e-30;
            const err = @abs(t.value(g) - e);
            worst = @max(worst, err / tol);
            try rig.expect(err <= tol, "KDA chain {s}: row {d} column {d}: {d} not {d} (tolerance {e})", .{ what, row, c, t.value(g), e, tol });
        }
    }
    std.debug.print("KDA chain {s}: worst error {d:.3} of its tolerance\n", .{ what, worst });
}

test "KDA: the window and prompt chains (conv, scan, keep, gated norm) against the float64 chain of glm_ref_kda" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    const heads = glm_heads;
    const n = heads * d;
    const eps: f32 = 1e-5; // GLM-5.3-Flash's rms_norm_eps, o_norm's
    for ([_]usize{ 3, 70 }) |rows| {
        const window = rows < keep_rows;
        const in = try ConvIn.make(t, heads, rows, 0xc4a1 + rows);
        defer in.free();
        var rng: rig.Rng = .{ .state = 0x9a7e5 + rows };
        const z = try gpa.alloc(u16, rows * n);
        defer gpa.free(z);
        for (z) |*v| v.* = t.bits(rng.unit() * 2.0);
        const norm_w = try gpa.alloc(f32, d);
        defer gpa.free(norm_w);
        for (norm_w) |*v| v.* = @floatCast(t.value(t.bits(1.0 + rng.unit() / 4.0)));
        const state0 = try gpa.alloc(f32, heads * d * d);
        defer gpa.free(state0);
        for (state0) |*v| v.* = rng.unit() * 0.05;
        // float64: prework, scan, gated norm
        const want_out = try gpa.alloc(f64, rows * n);
        defer gpa.free(want_out);
        const want_state = try widen(state0);
        defer gpa.free(want_state);
        {
            const p = try prepRef(t, in, true);
            defer p.free();
            const y = try gpa.alloc(f64, rows * n);
            defer gpa.free(y);
            scanRef(heads, rows, p.q, p.k, p.v, p.gate, p.beta, want_state, y);
            gnormRef(t, heads, rows, y, z, norm_w, eps, want_out);
        }
        var h: Held = .{};
        defer h.deinit();
        const c = try convDev(&h, t, in);
        const tail = try h.up(t, in.tail);
        const state = try h.up(t, state0);
        const dz = try h.up(t, z);
        const dnorm = try h.up(t, norm_w);
        const y = try h.zeros(t, rows * n * 4);
        const out = try h.zeros(t, rows * n * 2);
        const q = try h.zeros(t, rows * n * 4);
        // the window's k, v, decays and beta where its keep reads them: `rows` rows of each
        const win = try h.zeros(t, rows * (3 * n + heads) * 4);
        const sp: Split = .{ .q = at(q), .k = at(win), .v = at(win) + rows * n * 4, .gate = at(win) + 2 * rows * n * 4, .beta = at(win) + 3 * rows * n * 4 };
        if (window) {
            const snaps = try h.zeros(t, rows * (taps - 1) * 3 * n * 4);
            try convSplit(t, in, c, at(tail), at(snaps), sp);
            try t.on.tf_kda_hold(cf(sp.q), cf(sp.k), cf(sp.v), cf(sp.gate), cf(sp.beta), mf(at(state)), mf(at(y)), @intCast(rows), heads, t.stream.handle);
        } else {
            const next = try h.zeros(t, (taps - 1) * 3 * n * 4);
            try promptPrep(t, in, c, at(tail), at(next), sp);
            try t.on.tf_kda(cf(sp.q), cf(sp.k), cf(sp.v), cf(sp.gate), cf(sp.beta), mf(at(state)), mf(at(y)), @intCast(rows), heads, t.stream.handle, null, 0);
            try t.stream.synchronize();
            try expectTail(t, in, true, at(next), rows);
        }
        try t.on.tf_gnorm_out(cf(at(y)), cf(at(dnorm)), cv(at(dz)), @ptrFromInt(at(out)), t.kind(), @intCast(rows * heads), d, eps, 1, t.stream.handle);
        try t.stream.synchronize();
        try expectOut(t, out, want_out, n, if (window) "window of 3 rows (conv split, hold, gated norm)" else "prompt of 70 rows (prompt prep, prompt scan, gated norm)");
        if (window) {
            // the keep of all rows: the state a plain scan of the window stores, and float64's
            const plain = try h.up(t, state0);
            const y2 = try h.zeros(t, rows * n * 4);
            try t.on.tf_kda(cf(sp.q), cf(sp.k), cf(sp.v), cf(sp.gate), cf(sp.beta), mf(at(plain)), mf(at(y2)), @intCast(rows), heads, t.stream.handle, null, 0);
            try t.on.tf_kda_keep(mf(at(state)), @intCast(heads * d * d), cf(at(win)), 0, @intCast(rows), @intCast(rows), 1, heads, t.stream.handle);
            try t.stream.synchronize();
            const a = try rig.download(u8, plain);
            defer gpa.free(a);
            const b = try rig.download(u8, state);
            defer gpa.free(b);
            try rig.expect(std.mem.eql(u8, a, b), "KDA window chain: the keep's state differs from the plain scan's", .{});
            const ya = try rig.download(u8, y);
            defer gpa.free(ya);
            const yb = try rig.download(u8, y2);
            defer gpa.free(yb);
            try rig.expect(std.mem.eql(u8, ya, yb), "KDA window chain: the held window's outputs differ from the plain scan's", .{});
        }
        const gs = try rig.download(f32, state);
        defer gpa.free(gs);
        const es = relative(gs, want_state);
        std.debug.print("KDA chain rows={d}: state relative error {e:.3}\n", .{ rows, es });
        try rig.expect(es < 5e-3, "KDA chain rows={d}: state relative error {e}", .{ rows, es });
    }
}
