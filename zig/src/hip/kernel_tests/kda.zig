//! GLM-5.3-Flash's KDA on HIP against float64 host references, on the CUDA path's buffers (cuda_state.zig,
//! cuda_kda_block.zig): the prep (conv, decays a channel, beta, q and k norms), the conv shift, the scan (tile, prompt,
//! hold, keep on the window saves) and the chain to the bf16 out against glm_ref_kda.zig's float64 chain. One device:
//! 64 heads x 128, conv 4 taps, gate_lower_bound -5, o_norm eps 1e-5.
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
/// A decode window's rows (cuda_state.zig window_rows): the saves' rows a layer.
const window_rows = 8;
const eps_out: f32 = 1e-5;

fn sigmoid(x: f64) f64 {
    return 1 / (1 + @exp(-x));
}

const bfOf = rig.bf16Value;

/// bf16 bits of x to nearest even, as the kernels round.
fn bfBits(x: f64) u16 {
    const u: u32 = @bitCast(@as(f32, @floatCast(x)));
    return @intCast((u + (((u >> 16) & 1) + 0x7fff)) >> 16);
}

/// x rounded to bf16, as a float64.
fn rb(x: f64) f64 {
    return bfOf(bfBits(x));
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

fn widenBf(xs: []const u16) ![]f64 {
    const out = try gpa.alloc(f64, xs.len);
    for (out, xs) |*o, x| o.* = bfOf(x);
    return out;
}

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

fn mv(addr: u64) ?*anyopaque {
    return @ptrFromInt(addr);
}

/// `count` values at a device address inside a held buffer.
fn download(comptime T: type, t: *Rig, addr: u64, count: usize) ![]T {
    const view: DeviceBuffer = .{ .r = &t.r, .ptr = @ptrFromInt(addr), .len = count * @sizeOf(T) };
    return rig.download(T, view);
}

/// One KDA layer's block inputs as the CUDA path holds them: the stacked projection p bf16 (rows, kw) = [q | k | v |
/// f_a | g_a | b], the decay rows a and gate rows g bf16 (rows, heads x 128), the conv window cs bf16 [3][3 heads 128],
/// taps bf16 [3 heads 128][4] as the checkpoint stores them, A_log a head, dt_bias a channel, o_norm bf16 [128].
const Block = struct {
    heads: usize,
    rows: usize,
    p: []u16,
    a: []u16,
    g: []u16,
    cs: []u16,
    cw: []u16,
    a_log: []f32,
    dt_bias: []f32,
    norm: []u16,

    fn n(b: Block) usize {
        return b.heads * d;
    }

    /// kda_block.Layer.width: 3 heads 128 + f_a 128 + g_a 128 + b heads.
    fn width(b: Block) usize {
        return 3 * b.n() + 2 * d + b.heads;
    }

    fn bOff(b: Block) usize {
        return 3 * b.n() + 2 * d;
    }

    fn make(heads: usize, rows: usize, seed: u64, zero_window: bool) !Block {
        var rng: rig.Rng = .{ .state = seed };
        const n_ = heads * d;
        const kw = 3 * n_ + 2 * d + heads;
        const b: Block = .{
            .heads = heads,
            .rows = rows,
            .p = try gpa.alloc(u16, rows * kw),
            .a = try gpa.alloc(u16, rows * n_),
            .g = try gpa.alloc(u16, rows * n_),
            .cs = try gpa.alloc(u16, 3 * 3 * n_),
            .cw = try gpa.alloc(u16, 3 * n_ * taps),
            .a_log = try gpa.alloc(f32, heads),
            .dt_bias = try gpa.alloc(f32, n_),
            .norm = try gpa.alloc(u16, d),
        };
        for (b.p) |*v| v.* = bfBits(rng.unit() * 1.0);
        for (b.a) |*v| v.* = bfBits(rng.unit() * 2.0);
        for (b.g) |*v| v.* = bfBits(rng.unit() * 2.0);
        for (b.cs) |*v| v.* = if (zero_window) 0 else bfBits(rng.unit() * 1.0);
        for (b.cw) |*v| v.* = bfBits(rng.unit() * 0.5);
        for (b.a_log) |*v| v.* = rng.unit();
        for (b.dt_bias) |*v| v.* = rng.unit() * 0.5;
        for (b.norm) |*v| v.* = bfBits(1.0 + rng.unit() / 4.0);
        return b;
    }

    fn free(b: Block) void {
        gpa.free(b.p);
        gpa.free(b.a);
        gpa.free(b.g);
        gpa.free(b.cs);
        gpa.free(b.cw);
        gpa.free(b.a_log);
        gpa.free(b.dt_bias);
        gpa.free(b.norm);
    }

    /// Conv input `row` (window rows first) of channel c.
    fn input(b: Block, row: usize, c: usize) f64 {
        if (row < taps - 1) return bfOf(b.cs[row * 3 * b.n() + c]);
        return bfOf(b.p[(row - (taps - 1)) * b.width() + c]);
    }
};

/// The prep's outputs: q, k (L2 of the bf16 acts), v (bf16), decays, beta (bf16); `pre` = conv + SiLU before rounding.
const Prep = struct {
    q: []f64,
    k: []f64,
    v: []f64,
    pre_v: []f64,
    gate: []f64,
    beta: []f64,
    pre_beta: []f64,

    fn free(p: Prep) void {
        for ([_][]f64{ p.q, p.k, p.v, p.pre_v, p.gate, p.beta, p.pre_beta }) |x| gpa.free(x);
    }
};

/// float64 prep_kernel_body: conv + SiLU rounded to bf16, q and k L2-normed (q by 128^-0.5), decays, bf16 beta.
fn prepRef(b: Block) !Prep {
    const heads = b.heads;
    const n = b.n();
    const rows = b.rows;
    const p: Prep = .{
        .q = try gpa.alloc(f64, rows * n),
        .k = try gpa.alloc(f64, rows * n),
        .v = try gpa.alloc(f64, rows * n),
        .pre_v = try gpa.alloc(f64, rows * n),
        .gate = try gpa.alloc(f64, rows * n),
        .beta = try gpa.alloc(f64, rows * heads),
        .pre_beta = try gpa.alloc(f64, rows * heads),
    };
    var act: [3][d]f64 = undefined;
    for (0..rows) |r| for (0..heads) |h| {
        for (0..3) |part| for (0..d) |i| {
            const c = part * n + h * d + i;
            var acc: f64 = 0;
            for (0..taps) |tap| acc += bfOf(b.cw[c * taps + tap]) * b.input(r + tap, c);
            const silu = acc * sigmoid(acc);
            act[part][i] = rb(silu);
            if (part == 2) p.pre_v[r * n + h * d + i] = silu;
        };
        for ([_][]f64{ p.q, p.k }, 0..) |dst, part| {
            var ss: f64 = 0;
            for (act[part]) |e| ss += e * e;
            const extra: f64 = if (part == 0) 1 / @sqrt(@as(f64, d)) else 1;
            const inv = 1 / @sqrt(ss + 1e-6) * extra;
            for (dst[r * n + h * d ..][0..d], act[part]) |*o, e| o.* = e * inv;
        }
        @memcpy(p.v[r * n + h * d ..][0..d], &act[2]);
        const rate = @exp(@as(f64, b.a_log[h]));
        for (0..d) |i| {
            const c = h * d + i;
            p.gate[r * n + c] = @exp(@as(f64, lower) * sigmoid(rate * (bfOf(b.a[r * n + c]) + b.dt_bias[c])));
        }
        const sb = sigmoid(bfOf(b.p[r * b.width() + b.bOff() + h]));
        p.pre_beta[r * heads + h] = sb;
        p.beta[r * heads + h] = rb(sb);
    };
    return p;
}

/// float64 delta rule with a decay a key channel: state (heads, 128 v, 128 k) in place, y (rows, heads, 128).
fn scanRef(heads: usize, rows: usize, q: []const f64, k: []const f64, v: []const f64, gate: []const f64, beta: []const f64, state: []f64, y: []f64) void {
    for (0..heads) |h| {
        const s = state[h * d * d ..][0 .. d * d];
        for (0..rows) |t| {
            const base = (t * heads + h) * d;
            const g = gate[base..][0..d];
            const kk = k[base..][0..d];
            const qq = q[base..][0..d];
            const b = beta[t * heads + h];
            for (0..d) |r| {
                const row = s[r * d ..][0..d];
                var kv: f64 = 0;
                for (row, g, kk) |*e, gi, ki| {
                    e.* *= gi;
                    kv += e.* * ki;
                }
                const delta = (v[base + r] - kv) * b;
                var out: f64 = 0;
                for (row, kk, qq) |*e, ki, qi| {
                    e.* += ki * delta;
                    out += e.* * qi;
                }
                y[base + r] = out;
            }
        }
    }
}

/// A block's inputs on the device.
const BlockDev = struct { p: u64, a: u64, g: u64, cs: u64, cw: u64, a_log: u64, dt_bias: u64, norm: u64 };

fn blockDev(h: *Held, t: *Rig, b: Block) !BlockDev {
    return .{
        .p = at(try h.up(t, b.p)),
        .a = at(try h.up(t, b.a)),
        .g = at(try h.up(t, b.g)),
        .cs = at(try h.up(t, b.cs)),
        .cw = at(try h.up(t, b.cw)),
        .a_log = at(try h.up(t, b.a_log)),
        .dt_bias = at(try h.up(t, b.dt_bias)),
        .norm = at(try h.up(t, b.norm)),
    };
}

/// The scan's inputs as the prep writes them: q fp32, then the saves k fp32, v bf16, decays fp32, beta fp32.
const Saves = struct { q: u64, k: u64, v: u64, g: u64, b: u64 };

fn saves(h: *Held, t: *Rig, rows: usize, heads: usize) !Saves {
    const n = heads * d;
    return .{
        .q = at(try h.zeros(t, rows * n * 4)),
        .k = at(try h.zeros(t, rows * n * 4)),
        .v = at(try h.zeros(t, rows * n * 2)),
        .g = at(try h.zeros(t, rows * n * 4)),
        .b = at(try h.zeros(t, rows * heads * 4)),
    };
}

fn prep(t: *Rig, b: Block, dev: BlockDev, out: Saves) !void {
    try t.on.tf_kda_prep(cv(dev.p), @intCast(b.width()), @intCast(b.bOff()), cv(dev.a), @intCast(b.n()), cv(dev.cs), cv(dev.cw), cf(dev.a_log), cf(dev.dt_bias), lower, mf(out.q), mf(out.k), mv(out.v), mf(out.g), mf(out.b), @intCast(b.rows), @intCast(b.heads), t.stream.handle);
}

/// Bf16 `got` against float64 `pre` rounded: within a bf16 ulp (2^-7 of the value: a rounding boundary that fp32 and
/// float64 sums put on either side), and the count that equals rb(pre) exactly.
fn bfClose(got: []const u16, pre: []const f64, exact: *usize, worst: *f64) bool {
    for (got, pre) |g, w| {
        const e = @abs(bfOf(g) - w);
        const tol = @abs(w) * std.math.pow(f64, 2, -7) + 1e-30;
        worst.* = @max(worst.*, e / tol);
        if (e > tol) return false;
        exact.* += @intFromBool(g == bfBits(w));
    }
    return true;
}

test "KDA: the prep on the CUDA buffers writes float64's q, k, v (bf16), decays a channel and beta" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    const heads = glm_heads;
    var exact: usize = 0;
    var total: usize = 0;
    var worst: f64 = 0;
    const Case = struct { rows: usize, zero: bool };
    for ([_]Case{ .{ .rows = 1, .zero = false }, .{ .rows = 3, .zero = false }, .{ .rows = 8, .zero = false }, .{ .rows = 70, .zero = false }, .{ .rows = 70, .zero = true }, .{ .rows = 300, .zero = false } }) |cs| {
        const b = try Block.make(heads, cs.rows, 0x4bda + cs.rows + @intFromBool(cs.zero), cs.zero);
        defer b.free();
        const want = try prepRef(b);
        defer want.free();
        var h: Held = .{};
        defer h.deinit();
        const dev = try blockDev(&h, t, b);
        const out = try saves(&h, t, cs.rows, heads);
        try prep(t, b, dev, out);
        try t.stream.synchronize();
        const n = cs.rows * b.n();
        // v and beta: bf16, one rounding of float64's value
        const gv = try download(u16, t, out.v, n);
        defer gpa.free(gv);
        const gb = try download(f32, t, out.b, cs.rows * heads);
        defer gpa.free(gb);
        const gb16 = try gpa.alloc(u16, gb.len);
        defer gpa.free(gb16);
        for (gb16, gb) |*o, x| o.* = bfBits(x);
        for (gb16, gb) |o, x| try rig.expect(bfOf(o) == x, "KDA prep: beta {d} is not a bf16 value", .{x});
        try rig.expect(bfClose(gv, want.pre_v, &exact, &worst), "KDA prep rows={d}: v outside a bf16 ulp", .{cs.rows});
        try rig.expect(bfClose(gb16, want.pre_beta, &exact, &worst), "KDA prep rows={d}: beta outside a bf16 ulp", .{cs.rows});
        total += n + gb.len;
        // q and k: L2 of the bf16 acts, within a flipped act's share (2^-7 of the head's largest value) and fp32's 1e-5
        for ([_]u64{ out.q, out.k }, [_][]f64{ want.q, want.k }, [_][]const u8{ "q", "k" }) |addr, ref, name| {
            const got = try download(f32, t, addr, n);
            defer gpa.free(got);
            var e_most: f64 = 0;
            for (0..n / d) |hd| {
                var most: f64 = 0;
                for (ref[hd * d ..][0..d]) |x| most = @max(most, @abs(x));
                for (got[hd * d ..][0..d], ref[hd * d ..][0..d]) |g, w| {
                    const e = @abs(@as(f64, g) - w) / most;
                    e_most = @max(e_most, e);
                }
            }
            std.debug.print("KDA prep rows={d}{s}: {s} worst error {e:.3} of the head's largest value\n", .{ cs.rows, if (cs.zero) " (zero window)" else "", name, e_most });
            try rig.expect(e_most < std.math.pow(f64, 2, -7), "KDA prep rows={d}: {s} error {e}", .{ cs.rows, name, e_most });
        }
        const gg = try download(f32, t, out.g, n);
        defer gpa.free(gg);
        const eg = relative(gg, want.gate);
        std.debug.print("KDA prep rows={d}: decay relative error {e:.3}\n", .{ cs.rows, eg });
        try rig.expect(eg < 1e-5, "KDA prep rows={d}: decay relative error {e}", .{ cs.rows, eg });
    }
    const frac = @as(f64, @floatFromInt(exact)) / @as(f64, @floatFromInt(total));
    std.debug.print("KDA prep: v and beta equal float64's bf16 rounding in {d} of {d} ({d:.5}); worst {d:.3} of a bf16 ulp\n", .{ exact, total, frac, worst });
    try rig.expect(frac > 0.999, "KDA prep: only {d:.5} of v and beta equal float64's rounding", .{frac});
}

test "KDA: the conv shift takes rows keep .. keep + 2 of [window; projection rows], every layer" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    const heads = glm_heads;
    const c = 3 * heads * d;
    const kw = 3 * heads * d + 2 * d + heads;
    const layers = 3;
    var rng: rig.Rng = .{ .state = 0x5a1f7 };
    const conv = try gpa.alloc(u16, layers * 3 * c);
    defer gpa.free(conv);
    for (conv) |*v| v.* = bfBits(rng.unit());
    for ([_]usize{ window_rows, 300 }) |rows| {
        const proj = try gpa.alloc(u16, layers * rows * kw);
        defer gpa.free(proj);
        for (proj) |*v| v.* = bfBits(rng.unit());
        for (0..rows + 1) |keep| {
            if (rows > window_rows and keep != rows) continue; // a prompt chunk commits all its rows
            var h: Held = .{};
            defer h.deinit();
            const dc = try h.up(t, conv);
            const dp = try h.up(t, proj);
            try t.on.tf_kda_conv_shift(mv(at(dc)), cv(at(dp)), @intCast(keep), @intCast(3 * c), @intCast(rows * kw), @intCast(kw), @intCast(c), layers, t.stream.handle);
            try t.stream.synchronize();
            const got = try rig.download(u16, dc);
            defer gpa.free(got);
            for (0..layers) |l| for (0..3) |j| for (0..c) |k| {
                const src = keep + j;
                const want = if (src < 3) conv[l * 3 * c + src * c + k] else proj[l * rows * kw + (src - 3) * kw + k];
                try rig.expect(got[l * 3 * c + j * c + k] == want, "KDA conv shift rows={d} keep={d}: layer {d} row {d} column {d}", .{ rows, keep, l, j, k });
            };
        }
    }
}

/// The scan's inputs: q and k L2-normed (q to 128^-0.5), v bf16, decays exp(-5 sigmoid(2 z)), beta bf16(sigmoid(z)).
const ScanIn = struct {
    heads: usize,
    rows: usize,
    q: []f32,
    k: []f32,
    v: []u16,
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
            .v = try gpa.alloc(u16, rows * n),
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
        for (in.v) |*x| x.* = bfBits(rng.normal());
        for (in.gate) |*g| g.* = @floatCast(@exp(@as(f64, lower) * sigmoid(2 * rng.normal())));
        for (in.beta) |*b| b.* = @floatCast(rb(sigmoid(rng.normal())));
        for (in.state) |*x| x.* = @floatCast(0.1 * rng.normal());
        return in;
    }

    fn free(in: ScanIn) void {
        for ([_][]f32{ in.q, in.k, in.gate, in.beta, in.state }) |a| gpa.free(a);
        gpa.free(in.v);
    }

    fn dev(in: ScanIn, h: *Held, t: *Rig) !Saves {
        return .{ .q = at(try h.up(t, in.q)), .k = at(try h.up(t, in.k)), .v = at(try h.up(t, in.v)), .g = at(try h.up(t, in.gate)), .b = at(try h.up(t, in.beta)) };
    }
};

/// tf_kda over rows [from, from + rows) (state in place; `snaps`: each row's state, 0: none).
fn scan(t: *Rig, heads: usize, in: Saves, state: u64, y: u64, from: usize, rows: usize, snaps: u64) !void {
    const n = heads * d;
    try t.on.tf_kda(cf(in.q + from * n * 4), cf(in.k + from * n * 4), cv(in.v + from * n * 2), cf(in.g + from * n * 4), cf(in.b + from * heads * 4), mf(state), mf(y + from * n * 4), @intCast(rows), @intCast(heads), t.stream.handle, mf(snaps), 0);
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
            const w = [_][]f64{ try widen(in.q), try widen(in.k), try widenBf(in.v), try widen(in.gate), try widen(in.beta) };
            defer for (w) |x| gpa.free(x);
            scanRef(sh.heads, sh.rows, w[0], w[1], w[2], w[3], w[4], rs, ry);
        }
        var h: Held = .{};
        defer h.deinit();
        const dev = try in.dev(&h, t);
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
    const dev = try in.dev(&h, t);
    const state = try h.up(t, in.state);
    const y = try h.zeros(t, rows * heads * d * 4);
    try scan(t, heads, dev, at(state), at(y), 0, rows, 0);
    try t.stream.synchronize();
    const want_y = try rig.download(u8, y);
    defer gpa.free(want_y);
    const want_s = try rig.download(u8, state);
    defer gpa.free(want_s);
    // cuts either side of 64 rows (prompt blocks and tile mixed), and 50-row calls (the tile alone)
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

/// `layers` layers' window saves as cuda_state.zig lays them: k, v (bf16), decays [layers][window_rows][heads][128],
/// beta [layers][window_rows][heads]; the first `rows` rows of layer z from ins[z], the rest NaN.
const WindowSaves = struct {
    k: []f32,
    v: []u16,
    g: []f32,
    b: []f32,

    fn make(ins: []const ScanIn, rows: usize) !WindowSaves {
        const heads = ins[0].heads;
        const kv = window_rows * heads * d;
        const w: WindowSaves = .{
            .k = try gpa.alloc(f32, ins.len * kv),
            .v = try gpa.alloc(u16, ins.len * kv),
            .g = try gpa.alloc(f32, ins.len * kv),
            .b = try gpa.alloc(f32, ins.len * window_rows * heads),
        };
        @memset(w.k, std.math.nan(f32));
        @memset(w.v, 0x7fc0);
        @memset(w.g, std.math.nan(f32));
        @memset(w.b, std.math.nan(f32));
        for (ins, 0..) |in, z| {
            const n = rows * heads * d;
            @memcpy(w.k[z * kv ..][0..n], in.k[0..n]);
            @memcpy(w.v[z * kv ..][0..n], in.v[0..n]);
            @memcpy(w.g[z * kv ..][0..n], in.gate[0..n]);
            @memcpy(w.b[z * window_rows * heads ..][0 .. rows * heads], in.beta[0 .. rows * heads]);
        }
        return w;
    }

    fn free(w: WindowSaves) void {
        gpa.free(w.k);
        gpa.free(w.v);
        gpa.free(w.g);
        gpa.free(w.b);
    }
};

test "KDA: a held window and its keep on the CUDA saves give the window's outputs and its state after row kept - 1" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    const heads = glm_heads;
    const n = heads * d;
    const layers = 2;
    var ins: [layers]ScanIn = undefined;
    for (&ins, 0..) |*in, z| in.* = try ScanIn.make(heads, window_rows, 0x4ee9 + z);
    defer for (ins) |in| in.free();
    const state_floats = heads * d * d;
    const state_bytes = state_floats * 4;
    const state_stride = state_floats + 64;
    for (1..window_rows + 1) |rows| {
        var want_y: [layers][]u8 = undefined;
        var want_snaps: [layers][]u8 = undefined;
        for (0..layers) |z| {
            var h: Held = .{};
            defer h.deinit();
            const dev = try ins[z].dev(&h, t);
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
        for (0..layers) |z| {
            var h: Held = .{};
            defer h.deinit();
            const dev = try ins[z].dev(&h, t);
            const state = try h.up(t, ins[z].state);
            const y = try h.zeros(t, rows * n * 4);
            try t.on.tf_kda_hold(cf(dev.q), cf(dev.k), cv(dev.v), cf(dev.g), cf(dev.b), mf(at(state)), mf(at(y)), @intCast(rows), heads, t.stream.handle);
            try t.stream.synchronize();
            const got_y = try rig.download(u8, y);
            defer gpa.free(got_y);
            try rig.expect(std.mem.eql(u8, want_y[z], got_y), "KDA hold rows={d} layer={d}: outputs differ", .{ rows, z });
            const held = try rig.download(f32, state);
            defer gpa.free(held);
            try std.testing.expectEqualSlices(u32, @ptrCast(ins[z].state), @ptrCast(held));
        }
        const ws = try WindowSaves.make(&ins, rows);
        defer ws.free();
        const states = try gpa.alloc(f32, layers * state_stride);
        defer gpa.free(states);
        for (1..rows + 1) |kept| {
            @memset(states, std.math.nan(f32));
            for (0..layers) |z| @memcpy(states[z * state_stride ..][0..state_floats], ins[z].state);
            var h: Held = .{};
            defer h.deinit();
            const dk = try h.up(t, ws.k);
            const dv = try h.up(t, ws.v);
            const dg = try h.up(t, ws.g);
            const db = try h.up(t, ws.b);
            const state = try h.up(t, states);
            try t.on.tf_kda_keep(mf(at(state)), @intCast(state_stride), cf(at(dk)), cv(at(dv)), cf(at(dg)), cf(at(db)), window_rows * n, window_rows * heads, @intCast(kept), layers, heads, t.stream.handle);
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

test "KDA: windows kept in place, then a prompt chunk, give the bits of one serial scan of the kept rows" {
    // (a) a window whose drafts are all rejected: held, the state is the start's bits; its keep of the pending row
    // alone (kept 0 never happens: the pending row is always kept, cuda_forward.zig commit's BadCommit; tf_kda_keep
    // refuses it);
    // (b) windows of 4 rows keeping 1, 2 and 3; (c) each window after a partial keep; (d) a 70-row prompt chunk after.
    const t = try Rig.open(1 << 20);
    defer t.close();
    const heads = glm_heads;
    const n = heads * d;
    const keeps = [_]usize{ 1, 2, 3 };
    const prompt = 70;
    var wins: [keeps.len]ScanIn = undefined;
    for (&wins, 0..) |*w, i| w.* = try ScanIn.make(heads, 4, 0x5e9 + i);
    defer for (wins) |w| w.free();
    const tail = try ScanIn.make(heads, prompt, 0x5e9f);
    defer tail.free();
    // the serial rows: each window's kept rows, then the prompt's
    var total: usize = prompt;
    for (keeps) |k| total += k;
    const serial: ScanIn = .{ .heads = heads, .rows = total, .q = try gpa.alloc(f32, total * n), .k = try gpa.alloc(f32, total * n), .v = try gpa.alloc(u16, total * n), .gate = try gpa.alloc(f32, total * n), .beta = try gpa.alloc(f32, total * heads), .state = try gpa.dupe(f32, tail.state) };
    defer serial.free();
    {
        var row: usize = 0;
        for (wins ++ [_]ScanIn{tail}, keeps ++ [_]usize{prompt}) |src, k| {
            @memcpy(serial.q[row * n ..][0 .. k * n], src.q[0 .. k * n]);
            @memcpy(serial.k[row * n ..][0 .. k * n], src.k[0 .. k * n]);
            @memcpy(serial.v[row * n ..][0 .. k * n], src.v[0 .. k * n]);
            @memcpy(serial.gate[row * n ..][0 .. k * n], src.gate[0 .. k * n]);
            @memcpy(serial.beta[row * heads ..][0 .. k * heads], src.beta[0 .. k * heads]);
            row += k;
        }
    }
    var h: Held = .{};
    defer h.deinit();
    // the serial scan: one call over every kept row (the prompt blocks; the cut test shows they are the tile's bits)
    const sd = try serial.dev(&h, t);
    const s_state = try h.up(t, serial.state);
    const s_y = try h.zeros(t, total * n * 4);
    try scan(t, heads, sd, at(s_state), at(s_y), 0, total, 0);
    try t.stream.synchronize();
    const want_y = try rig.download(f32, s_y);
    defer gpa.free(want_y);
    const want_s = try rig.download(u8, s_state);
    defer gpa.free(want_s);
    // the engine's order: hold, keep in place, a window at a time, then the prompt chunk
    const state = try h.up(t, serial.state);
    const y = try h.zeros(t, 4 * n * 4);
    const s = t.stream.handle;
    var row: usize = 0;
    for (wins, keeps, 0..) |w, k, i| {
        const wd = try w.dev(&h, t);
        try t.on.tf_kda_hold(cf(wd.q), cf(wd.k), cv(wd.v), cf(wd.g), cf(wd.b), mf(at(state)), mf(at(y)), 4, heads, s);
        if (i == 0) {
            try t.stream.synchronize();
            const held = try rig.download(u32, state);
            defer gpa.free(held);
            try std.testing.expectEqualSlices(u32, @ptrCast(serial.state), held);
        }
        try t.on.tf_kda_keep(mf(at(state)), 0, cf(wd.k), cv(wd.v), cf(wd.g), cf(wd.b), 0, 0, @intCast(k), 1, heads, s);
        try t.stream.synchronize();
        const gy = try rig.download(f32, y);
        defer gpa.free(gy);
        try rig.expect(std.mem.eql(u32, @ptrCast(gy[0 .. k * n]), @ptrCast(want_y[row * n ..][0 .. k * n])), "KDA sequence: window {d} (kept {d}): the kept rows' outputs differ from the serial scan's", .{ i, k });
        row += k;
    }
    const td = try tail.dev(&h, t);
    const py = try h.zeros(t, prompt * n * 4);
    try t.on.tf_kda(cf(td.q), cf(td.k), cv(td.v), cf(td.g), cf(td.b), mf(at(state)), mf(at(py)), prompt, heads, s, null, 0);
    try t.stream.synchronize();
    const gy = try rig.download(f32, py);
    defer gpa.free(gy);
    try rig.expect(std.mem.eql(u32, @ptrCast(gy), @ptrCast(want_y[row * n ..][0 .. prompt * n])), "KDA sequence: the prompt chunk's outputs differ from the serial scan's", .{});
    const gs = try rig.download(u8, state);
    defer gpa.free(gs);
    try rig.expect(std.mem.eql(u8, gs, want_s), "KDA sequence: the state after the prompt chunk differs from the serial scan's", .{});
}

/// glm_ref_kda.zig's float64 chain on a Block: conv + SiLU, L2 q / k, decays, bf16 beta, delta rule, y = bf16(S q),
/// RMS norm, gate. Out [rows][n]; the state after the rows into `state`.
fn chainRef(b: Block, state: []f64, out: []f64) !void {
    const p = try prepRef(b);
    defer p.free();
    const n = b.n();
    const y = try gpa.alloc(f64, b.rows * n);
    defer gpa.free(y);
    scanRef(b.heads, b.rows, p.q, p.k, p.v, p.gate, p.beta, state, y);
    for (0..b.rows * b.heads) |rh| {
        const yy = y[rh * d ..][0..d];
        var ss: f64 = 0;
        for (yy) |*e| {
            e.* = rb(e.*);
            ss += e.* * e.*;
        }
        const rinv = 1 / @sqrt(ss / d + eps_out);
        for (out[rh * d ..][0..d], yy, 0..) |*o, e, i| o.* = bfOf(b.norm[i]) * e * rinv * sigmoid(bfOf(b.g[rh * d + i]));
    }
}

/// Out bf16 against float64 within glm_ref_kda.zig:136's tolerance: 2^-8 of the value (the out's rounding) + 2^-8 of
/// the row's largest value (y's bf16 rounding, which fp32 and float64 sums can put on either side of a boundary). Both
/// round y where CUDA does (glm_kda.cu:109). Prints the share of outs equal to float64's rounding and the mean error.
fn expectOut(got: []const u16, want: []const f64, n: usize, what: []const u8) !void {
    var worst: f64 = 0;
    var sum: f64 = 0;
    var exact: usize = 0;
    for (0..want.len / n) |row| {
        var most: f64 = 0;
        for (want[row * n ..][0..n]) |e| most = @max(most, @abs(e));
        for (got[row * n ..][0..n], want[row * n ..][0..n], 0..) |g, e, c| {
            const tol = @abs(e) * std.math.pow(f64, 2, -8) + most * std.math.pow(f64, 2, -8) + 1e-30;
            const err = @abs(bfOf(g) - e);
            worst = @max(worst, err / tol);
            sum += err / tol;
            exact += @intFromBool(g == bfBits(e));
            try rig.expect(err <= tol, "KDA chain {s}: row {d} column {d}: {d} not {d} (tolerance {e})", .{ what, row, c, bfOf(g), e, tol });
        }
    }
    const count: f64 = @floatFromInt(want.len);
    std.debug.print("KDA chain {s}: worst error {d:.3} of its tolerance, mean {d:.4}; {d:.4} of outs equal float64's rounding\n", .{ what, worst, sum / count, @as(f64, @floatFromInt(exact)) / count });
}

test "KDA: the window and prompt chains on the CUDA buffers against the float64 chain of glm_ref_kda" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    const heads = glm_heads;
    for ([_]usize{ 3, 70 }) |rows| {
        const window = rows <= window_rows;
        const b = try Block.make(heads, rows, 0xc4a1 + rows, false);
        defer b.free();
        const n = b.n();
        var rng: rig.Rng = .{ .state = 0x9a7e5 + rows };
        const state0 = try gpa.alloc(f32, heads * d * d);
        defer gpa.free(state0);
        for (state0) |*v| v.* = rng.unit() * 0.05;
        const want_out = try gpa.alloc(f64, rows * n);
        defer gpa.free(want_out);
        const want_state = try widen(state0);
        defer gpa.free(want_state);
        try chainRef(b, want_state, want_out);
        var h: Held = .{};
        defer h.deinit();
        const dev = try blockDev(&h, t, b);
        const state = try h.up(t, state0);
        const y = try h.zeros(t, rows * n * 4);
        const out = try h.zeros(t, rows * n * 2);
        const sv = try saves(&h, t, rows, heads);
        const s = t.stream.handle;
        try prep(t, b, dev, sv);
        if (window) {
            // the window holds the state; the keep of all its rows commits it, and the conv window shifts by them
            try t.on.tf_kda_hold(cf(sv.q), cf(sv.k), cv(sv.v), cf(sv.g), cf(sv.b), mf(at(state)), mf(at(y)), @intCast(rows), heads, s);
            try t.on.tf_kda_keep(mf(at(state)), 0, cf(sv.k), cv(sv.v), cf(sv.g), cf(sv.b), 0, 0, @intCast(rows), 1, heads, s);
        } else {
            try t.on.tf_kda(cf(sv.q), cf(sv.k), cv(sv.v), cf(sv.g), cf(sv.b), mf(at(state)), mf(at(y)), @intCast(rows), heads, s, null, 0);
        }
        try t.on.tf_kda_out(cf(at(y)), cv(dev.g), @intCast(n), cv(dev.norm), eps_out, mv(at(out)), @intCast(rows), heads, s);
        try t.on.tf_kda_conv_shift(mv(dev.cs), cv(dev.p), @intCast(rows), @intCast(3 * 3 * n), @intCast(rows * b.width()), @intCast(b.width()), @intCast(3 * n), 1, s);
        try t.stream.synchronize();
        const what = if (window) "window of 3 rows (prep, hold, keep, out, conv shift)" else "prompt of 70 rows (prep, prompt scan, out, conv shift)";
        const got = try rig.download(u16, out);
        defer gpa.free(got);
        try expectOut(got, want_out, n, what);
        const gs = try rig.download(f32, state);
        defer gpa.free(gs);
        const es = relative(gs, want_state);
        std.debug.print("KDA chain rows={d}: state relative error {e:.3}\n", .{ rows, es });
        try rig.expect(es < 5e-3, "KDA chain rows={d}: state relative error {e}", .{ rows, es });
        // the conv window after the rows: the last 3 inputs a channel
        const gw = try download(u16, t, dev.cs, 3 * 3 * n);
        defer gpa.free(gw);
        for (0..3) |j| for (0..3 * n) |c| {
            const src = rows + j;
            const want = if (src < 3) b.cs[src * 3 * n + c] else b.p[(src - 3) * b.width() + c];
            try rig.expect(gw[j * 3 * n + c] == want, "KDA chain rows={d}: conv window row {d} column {d}", .{ rows, j, c });
        };
    }
}

/// The median of `runs` launches of `f` in ms, by events on the rig's stream, after two warm-up launches.
fn timeMs(t: *Rig, runs: usize, ctx: anytype, comptime f: fn (@TypeOf(ctx)) anyerror!void) !f64 {
    const Event = @import("../stream.zig").Event;
    var a = try Event.init(&t.r, true);
    defer a.deinit();
    var b = try Event.init(&t.r, true);
    defer b.deinit();
    for (0..2) |_| try f(ctx);
    var ms: [32]f64 = undefined;
    for (ms[0..runs]) |*m| {
        try a.record(t.stream);
        try f(ctx);
        try b.record(t.stream);
        try b.synchronize();
        m.* = try Event.elapsedMs(a, b);
    }
    std.mem.sort(f64, ms[0..runs], {}, std.sort.asc(f64));
    return ms[runs / 2];
}

test "KDA: speed facts (no gate): prep, scans, out and keep at GLM's 64 heads, one layer, beside the DeltaNet's" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    const heads = glm_heads;
    const n = heads * d;
    const rows = 2048;
    const in = try ScanIn.make(heads, rows, 0x5eed);
    defer in.free();
    var h: Held = .{};
    defer h.deinit();
    const dev = try in.dev(&h, t);
    const state = try h.up(t, in.state);
    const y = try h.zeros(t, rows * n * 4);
    // the DeltaNet at the same shape: v fp32, a gate a head
    const v32 = try gpa.alloc(f32, rows * n);
    defer gpa.free(v32);
    for (v32, in.v) |*o, x| o.* = @floatCast(bfOf(x));
    const dv32 = at(try h.up(t, v32));
    const head_gate = at(try h.up(t, in.gate[0 .. rows * heads]));
    const Ctx = struct { t: *Rig, dev: Saves, state: u64, y: u64, v32: u64, g: u64, rows: usize, kda: bool, hold: bool };
    const run = struct {
        fn f(c: Ctx) anyerror!void {
            const l = &c.t.on;
            const s = c.t.stream.handle;
            const r: c_int = @intCast(c.rows);
            const x = c.dev;
            if (c.kda) {
                if (c.hold) return l.tf_kda_hold(cf(x.q), cf(x.k), cv(x.v), cf(x.g), cf(x.b), mf(c.state), mf(c.y), r, glm_heads, s);
                return l.tf_kda(cf(x.q), cf(x.k), cv(x.v), cf(x.g), cf(x.b), mf(c.state), mf(c.y), r, glm_heads, s, null, 0);
            }
            if (c.hold) return l.tf_gdn_hold(cf(x.q), cf(x.k), cf(c.v32), cf(c.g), cf(x.b), mf(c.state), mf(c.y), r, glm_heads, glm_heads, d, d, s);
            return l.tf_gated_delta(cf(x.q), cf(x.k), cf(c.v32), cf(c.g), cf(x.b), mf(c.state), mf(c.y), 1, r, glm_heads, glm_heads, d, d, s, null, 0);
        }
    };
    const Case = struct { name: []const u8, rows: usize, hold: bool };
    for ([_]Case{
        .{ .name = "scan, prompt 2048 rows", .rows = rows, .hold = false },
        .{ .name = "scan, decode 1 row", .rows = 1, .hold = false },
        .{ .name = "scan, verify hold 4 rows", .rows = 4, .hold = true },
    }) |cs| {
        var ms: [2]f64 = undefined;
        for (&ms, [_]bool{ true, false }) |*m, kda| m.* = try timeMs(t, 15, Ctx{ .t = t, .dev = dev, .state = at(state), .y = at(y), .v32 = dv32, .g = head_gate, .rows = cs.rows, .kda = kda, .hold = cs.hold }, run.f);
        std.debug.print("KDA speed {s}, 64 heads: KDA {d:.4} ms, DeltaNet (a gate a head, v fp32) {d:.4} ms, ratio {d:.3}\n", .{ cs.name, ms[0], ms[1], ms[0] / ms[1] });
    }
    // prep and out on the CUDA buffers, 2,048 rows and 1 row
    const b = try Block.make(heads, rows, 0xbeef, false);
    defer b.free();
    const bd = try blockDev(&h, t, b);
    const sv = try saves(&h, t, rows, heads);
    const out = at(try h.zeros(t, rows * n * 2));
    const PO = struct { t: *Rig, b: *const Block, bd: BlockDev, sv: Saves, y: u64, out: u64, rows: usize, which: u1 };
    const po = struct {
        fn f(c: PO) anyerror!void {
            const s = c.t.stream.handle;
            if (c.which == 0) return c.t.on.tf_kda_prep(cv(c.bd.p), @intCast(c.b.width()), @intCast(c.b.bOff()), cv(c.bd.a), @intCast(c.b.n()), cv(c.bd.cs), cv(c.bd.cw), cf(c.bd.a_log), cf(c.bd.dt_bias), lower, mf(c.sv.q), mf(c.sv.k), mv(c.sv.v), mf(c.sv.g), mf(c.sv.b), @intCast(c.rows), glm_heads, s);
            return c.t.on.tf_kda_out(cf(c.y), cv(c.bd.g), @intCast(c.b.n()), cv(c.bd.norm), eps_out, mv(c.out), @intCast(c.rows), glm_heads, s);
        }
    };
    for ([_]usize{ rows, 1 }) |r| for ([_]u1{ 0, 1 }, [_][]const u8{ "prep", "out" }) |which, name| {
        const m = try timeMs(t, 15, PO{ .t = t, .b = &b, .bd = bd, .sv = sv, .y = at(y), .out = out, .rows = r, .which = which }, po.f);
        std.debug.print("KDA speed {s}, {d} rows, 64 heads: {d:.4} ms\n", .{ name, r, m });
    };
    // the keep of 4 rows over GLM's 34 KDA layers on the window saves (one launch), and the conv shift of all 34
    const layers = 34;
    const kv = window_rows * n;
    const sk = at(try h.zeros(t, layers * kv * 4));
    const svv = at(try h.zeros(t, layers * kv * 2));
    const sg = at(try h.zeros(t, layers * kv * 4));
    const sb = at(try h.zeros(t, layers * window_rows * heads * 4));
    const states = at(try h.zeros(t, layers * heads * d * d * 4));
    const conv = at(try h.zeros(t, layers * 3 * 3 * n * 2));
    const proj = at(try h.zeros(t, layers * window_rows * b.width() * 2));
    const Keep = struct { t: *Rig, states: u64, k: u64, v: u64, g: u64, b: u64, conv: u64, proj: u64, width: usize, shift: bool };
    const keep = struct {
        fn f(c: Keep) anyerror!void {
            const s = c.t.stream.handle;
            if (c.shift) return c.t.on.tf_kda_conv_shift(mv(c.conv), cv(c.proj), 4, 3 * 3 * glm_heads * d, window_rows * @as(i64, @intCast(c.width)), @intCast(c.width), 3 * glm_heads * d, 34, s);
            return c.t.on.tf_kda_keep(mf(c.states), glm_heads * d * d, cf(c.k), cv(c.v), cf(c.g), cf(c.b), window_rows * glm_heads * d, window_rows * glm_heads, 4, 34, glm_heads, s);
        }
    };
    for ([_]bool{ false, true }) |shift| {
        const m = try timeMs(t, 15, Keep{ .t = t, .states = states, .k = sk, .v = svv, .g = sg, .b = sb, .conv = conv, .proj = proj, .width = b.width(), .shift = shift }, keep.f);
        if (shift) {
            std.debug.print("KDA speed conv shift of 4 rows, {d} layers: {d:.4} ms\n", .{ layers, m });
        } else {
            std.debug.print("KDA speed keep 4 of 8 rows, {d} layers x 64 heads: {d:.4} ms ({d:.1} GB/s on the states read and written)\n", .{ layers, m, @as(f64, @floatFromInt(2 * layers * heads * d * d * 4)) / m / 1e6 });
        }
    }
}
