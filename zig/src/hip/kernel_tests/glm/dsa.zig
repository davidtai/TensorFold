//! GLM-5.3-Flash's DSA kernels (glm/dsa.hpp) on the GPU against float64 host references (the CUDA tests'
//! zig/tests/cuda/glm_ref_attn.zig and glm_ref_index.zig rules), exact checks where the result is integers or copies,
//! and each launch's bits against the same rows in a wider or a split call. Then the speed facts at the model's shapes.
const std = @import("std");
const rig = @import("../rig.zig");
const DeviceBuffer = @import("../../memory.zig").DeviceBuffer;
const hruntime = @import("../../runtime.zig");
const dsa = @import("../../launches/glm_dsa.zig");

const gpa = std.testing.allocator;
const heads = 64;

test {
    _ = dsa;
}

const Rng = struct {
    state: u64,
    fn next(r: *Rng) u64 {
        r.state ^= r.state >> 12;
        r.state ^= r.state << 25;
        r.state ^= r.state >> 27;
        return r.state *% 0x2545F4914F6CDD1D;
    }
    /// Uniform on [-1, 1).
    fn unit(r: *Rng) f64 {
        return @as(f64, @floatFromInt(r.next() >> 11)) / 4503599627370496.0 - 1.0;
    }
    fn below(r: *Rng, n: usize) usize {
        return @intCast(r.next() % n);
    }
};

fn bfValue(b: u16) f64 {
    return @as(f32, @bitCast(@as(u32, b) << 16));
}

/// Round to nearest even, as float_to_bf16.
fn bfBits(v: f32) u16 {
    const w: u32 = @bitCast(v);
    const bias: u32 = ((w >> 16) & 1) + 0x7fff;
    return @intCast((w +% bias) >> 16);
}

/// The shared kernel-test rig (device, stream, launchers) with the DSA kernels, an arena and owned device buffers.
const Rig = struct {
    base: *rig.Rig,
    k: dsa.Kernels,
    arena: std.heap.ArenaAllocator,
    bufs: std.ArrayList(DeviceBuffer),
    rng: Rng,
    worst: f64 = 0,

    fn open(seed: u64) !*Rig {
        const t = try gpa.create(Rig);
        errdefer gpa.destroy(t);
        t.base = try rig.Rig.open(1 << 20);
        errdefer t.base.close();
        t.k = try dsa.Kernels.load(&t.base.on);
        t.arena = .init(gpa);
        t.bufs = .empty;
        t.rng = .{ .state = seed };
        t.worst = 0;
        return t;
    }

    fn close(t: *Rig) void {
        for (t.bufs.items) |*b| b.free();
        t.bufs.deinit(gpa);
        t.arena.deinit();
        t.base.close();
        gpa.destroy(t);
    }

    fn a(t: *Rig) std.mem.Allocator {
        return t.arena.allocator();
    }

    fn s(t: *Rig) dsa.S {
        return t.base.stream.handle;
    }

    /// Device bytes holding `host`; its address.
    fn dev(t: *Rig, host: anytype) !u64 {
        const b = try DeviceBuffer.fromHost(&t.base.r, std.mem.sliceAsBytes(host));
        try t.bufs.append(gpa, b);
        return b.base();
    }

    /// `n` device bytes, each `fill`.
    fn filled(t: *Rig, n: usize, fill: u8) !u64 {
        const b = try DeviceBuffer.alloc(&t.base.r, n);
        try t.bufs.append(gpa, b);
        try b.fill8(fill);
        return b.base();
    }

    fn zeros(t: *Rig, n: usize) !u64 {
        return t.filled(n, 0);
    }

    /// `n` bytes back from address `p` (inside a buffer of this rig), after the stream's work.
    fn back(t: *Rig, p: u64, n: usize) ![]u8 {
        try t.base.stream.synchronize();
        for (t.bufs.items) |b| {
            const lo = b.base();
            if (p >= lo and p + n <= lo + b.len) {
                const out = try t.a().alloc(u8, n);
                try b.download(p - lo, out);
                return out;
            }
        }
        return error.NotABuffer;
    }

    fn backAs(t: *Rig, comptime T: type, p: u64, n: usize) ![]align(1) const T {
        return std.mem.bytesAsSlice(T, try t.back(p, n * @sizeOf(T)));
    }

    /// `n` bf16 values uniform on [-scale, scale), rounded.
    fn bfs(t: *Rig, n: usize, scale: f64) ![]u16 {
        const out = try t.a().alloc(u16, n);
        for (out) |*v| v.* = bfBits(@floatCast(t.rng.unit() * scale));
        return out;
    }

    fn f64s(t: *Rig, x: []const u16) ![]f64 {
        const out = try t.a().alloc(f64, x.len);
        for (out, x) |*d, v| d.* = bfValue(v);
        return out;
    }
};

fn expect(ok: bool, comptime fmt: []const u8, args: anytype) !void {
    if (ok) return;
    std.debug.print("FAIL " ++ fmt ++ "\n", args);
    return error.TestUnexpectedResult;
}

/// RESULT from each column's worst fraction of its tolerance; all within 1.
fn worstCol(t: *Rig, name: []const u8, col: []const f64) !void {
    var most: f64 = 0;
    var at: usize = 0;
    for (col, 0..) |q, j| if (q > most) {
        most = q;
        at = j;
    };
    t.worst = @max(t.worst, most);
    std.debug.print("RESULT {s}: worst column {d} at {e:.3} of its tolerance\n", .{ name, at, most });
    try expect(most <= 1, "{s} within its float64 tolerance", .{name});
}

/// bf16 output at `p` against want within tol, columns i % n.
fn closeCols(t: *Rig, name: []const u8, p: u64, want: []const f64, tol: []const f64, n: usize) !void {
    const got = try t.backAs(u16, p, want.len);
    const col = try t.a().alloc(f64, n);
    @memset(col, 0);
    for (want, tol, 0..) |w, e, i| {
        const q = @abs(bfValue(got[i]) - w) / e;
        if (!(q <= col[i % n])) col[i % n] = if (std.math.isNan(q)) std.math.inf(f64) else q;
    }
    try worstCol(t, name, col);
}

fn sameBytes(t: *Rig, name: []const u8, a_: u64, b_: u64, n: usize) !void {
    const x = try t.back(a_, n);
    const y = try t.back(b_, n);
    for (x, y, 0..) |u, v, i| try expect(u == v, "{s}: byte {d} differs ({d} vs {d})", .{ name, i, u, v });
    std.debug.print("SAME {s}: {d} bytes\n", .{ name, n });
}

fn posDev(t: *Rig, p: usize) !u64 {
    const v: [1]i32 = .{@intCast(p)};
    return t.dev(&v);
}

const ulp = std.math.pow(f64, 2, -24);
const b8 = std.math.pow(f64, 2, -8);

test "GLM DSA: the latent cache write copies each row into its slot and nothing else" {
    const t = try Rig.open(0x6301);
    defer t.close();
    const cap = 1100;
    const rows = 7;
    const stride = 2048;
    const pos = 1000;
    const row0 = 3;
    const lat = try t.bfs(rows * stride, 1.0);
    const cache = try t.filled(cap * dsa.lw * 2, 0xab);
    try dsa.latentWrite(t.k, t.s(), try t.dev(lat), stride, cache, try posDev(t, pos), row0, rows);
    const got = try t.backAs(u16, cache, cap * dsa.lw);
    for (0..cap) |slot| for (0..dsa.lw) |d| {
        const g = got[slot * dsa.lw + d];
        const at = pos + row0;
        if (slot >= at and slot < at + rows) {
            try expect(g == lat[(slot - at) * stride + d], "latent write slot {d} dim {d}", .{ slot, d });
        } else try expect(g == 0xabab, "latent write: slot {d} left alone", .{slot});
    };
    std.debug.print("PASS latent cache write: {d} rows at {d} exact, other slots untouched\n", .{ rows, pos + row0 });
}

/// float64 y[r][h][n] = sum_k x[r][h k + k] W_h[k][n] over kv_b as stored (absorb: W_h[k][n] = kv_b[512 h + k][n];
/// expand, `values`: W_h[k][n] = kv_b[512 h + 256 + n][k]); x rows `xs` values apart. Bound: bf16 out + (K + 8) 2^-24
/// sum |x w|.
fn headProducts(t: *Rig, x: []const u16, xs: usize, kvb: []const u16, rows: usize, kd: usize, nd: usize, values: bool) !struct { want: []f64, tol: []f64, chain: []f64 } {
    const want = try t.a().alloc(f64, rows * heads * nd);
    const tol = try t.a().alloc(f64, rows * heads * nd);
    const chain = try t.a().alloc(f64, rows * heads * nd);
    const wd = try t.f64s(kvb);
    const xd = try t.f64s(x);
    const per = dsa.qk + dsa.dv;
    for (0..rows) |r| for (0..heads) |h| {
        const xr = xd[r * xs + h * kd ..][0..kd];
        for (0..nd) |n| {
            var sum: f64 = 0;
            var abs: f64 = 0;
            for (0..kd) |k| {
                const wv = if (values) wd[(h * per + dsa.qk + n) * dsa.lw + k] else wd[(h * per + k) * dsa.lw + n];
                sum += xr[k] * wv;
                abs += @abs(xr[k] * wv);
            }
            const i = (r * heads + h) * nd + n;
            want[i] = sum;
            chain[i] = abs * @as(f64, @floatFromInt(kd + 8)) * ulp;
            tol[i] = @abs(sum) * b8 + chain[i] + 1e-30;
        }
    };
    return .{ .want = want, .tol = tol, .chain = chain };
}

/// Each bf16 output is the float64 value rounded to nearest, or (only where the value is within `chain`, the fp32
/// accumulation's bound, of a rounding midpoint) the neighbour on the far side of that midpoint: the output error is
/// one bf16 rounding of an fp32 sum, nothing more.
fn roundedOnce(t: *Rig, name: []const u8, p: u64, want: []const f64, chain: []const f64) !void {
    const got = try t.backAs(u16, p, want.len);
    var nearest: usize = 0;
    var near_mid: usize = 0;
    for (want, chain, 0..) |w, c, i| {
        const lo = bfBits(@floatCast(w - c));
        const hi = bfBits(@floatCast(w + c));
        const g = got[i];
        try expect(g == lo or g == hi, "{s}: output {d} is 0x{x:0>4}; float64 {e} rounds to 0x{x:0>4} (0x{x:0>4} within the fp32 chain)", .{ name, i, g, w, bfBits(@floatCast(w)), if (lo == bfBits(@floatCast(w))) hi else lo });
        if (g == bfBits(@floatCast(w))) nearest += 1 else near_mid += 1;
    }
    std.debug.print("ROUNDED {s}: {d} of {d} outputs are float64 rounded to nearest bf16, {d} the neighbour within the fp32 chain of a midpoint\n", .{ name, nearest, want.len, near_mid });
}

test "GLM DSA: absorb and expand by kv_b as stored against float64; rows alone equal their rows in a wide call" {
    const t = try Rig.open(0x6302);
    defer t.close();
    const rows = 40;
    // kv_b_proj as the loader places it: [64 x 512][512] bf16, per head 256 key rows then 256 value rows
    const kvb = try t.bfs(heads * (dsa.qk + dsa.dv) * dsa.lw, 0.05);
    const dkv = try t.dev(kvb);
    {
        // q in the [q_b | wq_b] rows of 16,384 + 4,096 values (q at column 0)
        const qs = heads * dsa.qk + dsa.index_heads * dsa.index_dims;
        const q = try t.bfs(rows * qs, 1.0);
        const dq = try t.dev(q);
        const out = try t.zeros(rows * heads * dsa.lw * 2);
        try dsa.absorbQ(t.k, t.s(), dq, qs, dsa.kvKeys(dkv), dsa.kv_head, out, rows, heads);
        const ref = try headProducts(t, q, qs, kvb, rows, dsa.qk, dsa.lw, false);
        try closeCols(t, "absorb_q 40 rows x 64 heads, [256] . Wk_h [256, 512] in kv_b as stored", out, ref.want, ref.tol, dsa.lw);
        try roundedOnce(t, "absorb_q", out, ref.want, ref.chain);
        const row_bytes = heads * dsa.lw * 2;
        for ([_][2]usize{ .{ 0, 1 }, .{ 5, 3 }, .{ 17, 4 }, .{ 20, 17 } }) |w| {
            const o = try t.zeros(w[1] * row_bytes);
            try dsa.absorbQ(t.k, t.s(), dq + w[0] * qs * 2, qs, dsa.kvKeys(dkv), dsa.kv_head, o, w[1], heads);
            var name: [96]u8 = undefined;
            try sameBytes(t, try std.fmt.bufPrint(&name, "absorb_q: rows {d}..{d} alone", .{ w[0], w[0] + w[1] - 1 }), out + w[0] * row_bytes, o, w[1] * row_bytes);
        }
    }
    {
        const ol = try t.bfs(rows * heads * dsa.lw, 1.0);
        const dol = try t.dev(ol);
        const out = try t.zeros(rows * heads * dsa.dv * 2);
        try dsa.expandV(t.k, t.s(), dol, dsa.kvValues(dkv), dsa.kv_head, out, rows, heads);
        const ref = try headProducts(t, ol, heads * dsa.lw, kvb, rows, dsa.lw, dsa.dv, true);
        try closeCols(t, "expand_v 40 rows x 64 heads, [512] . Wv_h [256, 512]^T in kv_b as stored", out, ref.want, ref.tol, dsa.dv);
        try roundedOnce(t, "expand_v", out, ref.want, ref.chain);
        const row_bytes = heads * dsa.dv * 2;
        for ([_][2]usize{ .{ 0, 1 }, .{ 5, 3 }, .{ 17, 4 }, .{ 20, 17 } }) |w| {
            const o = try t.zeros(w[1] * row_bytes);
            try dsa.expandV(t.k, t.s(), dol + w[0] * heads * dsa.lw * 2, dsa.kvValues(dkv), dsa.kv_head, o, w[1], heads);
            var name: [96]u8 = undefined;
            try sameBytes(t, try std.fmt.bufPrint(&name, "expand_v: rows {d}..{d} alone", .{ w[0], w[0] + w[1] - 1 }), out + w[0] * row_bytes, o, w[1] * row_bytes);
        }
    }
    std.debug.print("PASS absorb and expand: within float64 tolerance, every row block's bits equal\n", .{});
}

test "GLM DSA: index keys (LayerNorm), gates and the 4-token pools over windows against float64" {
    const t = try Rig.open(0x6303);
    defer t.close();
    const d = dsa.index_dims;
    const total = 17; // positions 0 .. 16: pools 0 .. 3 complete, position 16 in pool 4
    const kstride = d + 32; // [wk | weights_proj] rows
    const kraw = try t.bfs(total * kstride, 2.0);
    for (0..total) |p| for (0..d) |i| { // an offset a row, so the mean matters
        kraw[p * kstride + i] = bfBits(@floatCast(bfValue(kraw[p * kstride + i]) + 0.75));
    };
    const gate = try t.a().alloc(f32, total * d);
    for (gate) |*g| g.* = @floatCast(t.rng.unit() * 3.0);
    const lnw = try t.bfs(d, 1.0);
    const lnb = try t.bfs(d, 0.5);
    const ape = try t.bfs(pool_rows * d, 1.0);
    const dk = try t.dev(kraw);
    const dg = try t.dev(gate);
    const dlnw = try t.dev(lnw);
    const dlnb = try t.dev(lnb);
    const dape = try t.dev(ape);
    const ik = try t.filled(32 * d * 2, 0xff);
    const ig = try t.filled(32 * d * 2, 0xff);
    const pk = try t.filled(8 * d * 2, 0xff);
    // windows (pos, row0, rows): the third window in two calls at row0 0 and 4
    for ([_][3]usize{ .{ 0, 0, 5 }, .{ 5, 0, 1 }, .{ 6, 0, 4 }, .{ 6, 4, 5 }, .{ 15, 0, 2 } }) |w| {
        const first = w[0] + w[1];
        try dsa.indexUpdate(t.k, t.s(), dk + first * kstride * 2, kstride, dg + first * d * 4, dlnw, dlnb, dape, ik, ig, pk, try posDev(t, w[0]), w[1], w[2]);
    }
    const gk = try t.backAs(u16, ik, 32 * d);
    const gg = try t.backAs(u16, ig, 32 * d);
    const col = try t.a().alloc(f64, d);
    @memset(col, 0);
    const norm = try t.a().alloc(f64, total * d); // the float64 LayerNorm, for the pools' bound
    for (0..total) |p| {
        var mean: f64 = 0;
        for (0..d) |i| mean += bfValue(kraw[p * kstride + i]);
        mean /= d;
        var v: f64 = 0;
        for (0..d) |i| v += std.math.pow(f64, bfValue(kraw[p * kstride + i]) - mean, 2);
        const sd = @sqrt(v / d + dsa.index_eps);
        for (0..d) |i| {
            const z = (bfValue(kraw[p * kstride + i]) - mean) / sd;
            const want = z * bfValue(lnw[i]) + bfValue(lnb[i]);
            norm[p * d + i] = want;
            // bf16 out; fp32 mean, variance and division (a few ulp of |z|, and of |mean| / sd through x - mean)
            const tol = @abs(want) * b8 + (@abs(z * bfValue(lnw[i])) + @abs(bfValue(lnb[i])) + @abs(bfValue(lnw[i])) * (1 + @abs(mean) / sd)) * 64 * ulp + 1e-30;
            const q = @abs(bfValue(gk[p * d + i]) - want) / tol;
            if (!(q <= col[i])) col[i] = if (std.math.isNan(q)) std.math.inf(f64) else q;
            try expect(gg[p * d + i] == bfBits(gate[p * d + i]), "index gate row {d} dim {d}: bf16 of the gate", .{ p, i });
        }
    }
    try worstCol(t, "index keys: LayerNorm of 17 rows over 5 windows", col);
    for (total..32) |p| for (0..d) |i| try expect(gk[p * d + i] == 0xffff and gg[p * d + i] == 0xffff, "index rows past the windows untouched ({d})", .{p});
    const gp = try t.backAs(u16, pk, 8 * d);
    @memset(col, 0);
    for (0..4) |p| for (0..d) |i| {
        var l: [4]f64 = undefined;
        var most = -std.math.inf(f64);
        for (0..4) |q| {
            l[q] = bfValue(gg[(4 * p + q) * d + i]) + bfValue(ape[q * d + i]);
            most = @max(most, l[q]);
        }
        var sum: f64 = 0;
        for (&l) |*x| {
            x.* = @exp(x.* - most);
            sum += x.*;
        }
        var want: f64 = 0;
        var abs: f64 = 0;
        for (0..4) |q| {
            const term = l[q] / sum * bfValue(gk[(4 * p + q) * d + i]);
            want += term;
            abs += @abs(term);
        }
        const tol = @abs(want) * b8 + abs * (2 * b8 + 64 * ulp) + 1e-30;
        const q = @abs(bfValue(gp[p * d + i]) - want) / tol;
        if (!(q <= col[i])) col[i] = if (std.math.isNan(q)) std.math.inf(f64) else q;
    };
    try worstCol(t, "pooled keys: pools 0..3 completed over 5 windows", col);
    for (4..8) |p| for (0..d) |i| try expect(gp[p * d + i] == 0xffff, "pool {d} (incomplete) untouched", .{p});
    std.debug.print("PASS index update: LayerNorm keys and pools within float64 tolerance, gates exact, the rest untouched\n", .{});
}

const pool_rows = dsa.pool;

/// sparse._order_key: a score's bits as a u32 whose unsigned order is the scores' (-0 counted as +0).
fn orderKey(sc: f32) u32 {
    const b: i32 = @bitCast(sc + 0.0);
    return @bitCast(b ^ ((b >> 31) | std.math.minInt(i32)));
}

/// The indexer of `rows` rows at `pos` (one call, then the same rows in two calls): scores vs float64, the lists
/// vs the host's rule on the device scores (exact), the split call's lists equal.
fn selection(t: *Rig, pos: usize, rows: usize, zero_pools: [2]usize, tie_pools: [2]usize) !void {
    const hd = dsa.index_heads;
    const d = dsa.index_dims;
    const np = dsa.poolBucket(pos, rows, 1 << 20);
    const cols = dsa.scoreColumns(pos, rows, np);
    // qi at column 16,384 of [q_b | wq_b] rows; the head weights at column 2,176 of [q_a | kv_a | wk | weights_proj] rows
    const qs = heads * dsa.qk + hd * d;
    const qcol = heads * dsa.qk;
    const qrows = try t.bfs(rows * qs, 1.0);
    const qi = try t.a().alloc(u16, rows * hd * d);
    for (0..rows) |r| @memcpy(qi[r * hd * d ..][0 .. hd * d], qrows[r * qs + qcol ..][0 .. hd * d]);
    const wcol = 1536 + 512 + d;
    const wstride = wcol + hd;
    const wrows = try t.bfs(rows * wstride, 1.0);
    const pk = try t.bfs(np * d, 1.0);
    for (zero_pools[0]..zero_pools[1]) |p| @memset(pk[p * d ..][0..d], 0);
    for (tie_pools[0]..tie_pools[1]) |p| @memcpy(pk[p * d ..][0..d], pk[(tie_pools[0] - 1) * d ..][0..d]);
    const dqi = (try t.dev(qrows)) + qcol * 2;
    const dw = try t.dev(wrows);
    const dpk = try t.dev(pk);
    const dpos = try posDev(t, pos);
    const sc: dsa.Scratch = .{ .scores = try t.zeros(rows * np * 4), .complete = try t.zeros(rows * 4), .ends = try t.zeros(rows * 4) };
    const tokens = try t.filled(rows * dsa.width * 4, 0xff);
    const counts = try t.zeros(rows * 4);
    const sparse = try t.zeros(rows * 4);
    try dsa.selectTokens(t.k, t.s(), dqi, qs, dw + wcol * 2, wstride, dpk, dpos, 0, pos, rows, np, sc, tokens, counts, sparse);
    const got = try t.backAs(f32, sc.scores, rows * np);

    // float64: sum_h w_h relu(scale q_h . k_p); bound: (128 + 8) 2^-24 scale sum |q k| a dot, 40 2^-24 sum |w_h relu|
    const scale = 1 / @sqrt(@as(f64, d));
    const wscale: f64 = dsa.head_scale;
    const kd = try t.f64s(pk);
    const col = try t.a().alloc(f64, cols);
    @memset(col, 0);
    var scored: usize = 0;
    for (0..rows) |row| {
        const npool = (pos + row + 1) / 4;
        if (npool <= dsa.topk_pools) continue;
        scored += 1;
        for (0..cols) |p| {
            const g = got[row * np + p];
            if (p >= npool) {
                try expect(g == -std.math.inf(f32), "score row {d} pool {d}: past the visible pools, -inf (got {d})", .{ row, p, g });
                continue;
            }
            var sum: f64 = 0;
            var bound: f64 = 0;
            var spread: f64 = 0;
            for (0..hd) |h| {
                const w = bfValue(wrows[row * wstride + wcol + h]) * wscale;
                var dot: f64 = 0;
                var abs: f64 = 0;
                for (0..d) |i| {
                    const x = bfValue(qi[(row * hd + h) * d + i]) * kd[p * d + i];
                    dot += x;
                    abs += @abs(x);
                }
                sum += w * @max(dot * scale, 0);
                spread += @abs(w) * @max(dot * scale, 0);
                bound += @abs(w) * abs * scale * 136 * ulp;
            }
            const q = @abs(g - sum) / (bound + spread * 40 * ulp + 1e-30);
            if (!(q <= col[p])) col[p] = if (std.math.isNan(q)) std.math.inf(f64) else q;
        }
    }
    var name: [128]u8 = undefined;
    if (scored > 0) try worstCol(t, try std.fmt.bufPrint(&name, "index scores {d} rows at {d} ({d} scored), 32 heads x 128, {d} pools", .{ rows, pos, scored, cols }), col);

    // the host's rule on the device's scores: top 512 of the complete pools by order key, ties to the lower pool,
    // ascending, then the incomplete pool's tail; rows with at most 512 complete pools: dense, count = position + 1
    const gt = try t.backAs(i32, tokens, rows * dsa.width);
    const gc = try t.backAs(i32, counts, rows);
    const gs = try t.backAs(i32, sparse, rows);
    const order = try t.a().alloc(u32, np);
    var ties: usize = 0;
    var zeros_at_cut: usize = 0;
    for (0..rows) |row| {
        const at = pos + row;
        const npool = (at + 1) / 4;
        if (npool <= dsa.topk_pools) {
            try expect(gs[row] == 0 and gc[row] == @as(i32, @intCast(at + 1)), "select row {d} (dense): sparse 0, count {d} (got {d}, {d})", .{ row, at + 1, gs[row], gc[row] });
            continue;
        }
        const keys = try t.a().alloc(u32, npool);
        for (keys, 0..) |*k, i| k.* = orderKey(got[row * np + i]);
        const idx = order[0..npool];
        for (idx, 0..) |*v, i| v.* = @intCast(i);
        std.mem.sort(u32, idx, @as([]const u32, keys), struct {
            fn lt(k: []const u32, x: u32, y: u32) bool {
                return k[x] > k[y] or (k[x] == k[y] and x < y);
            }
        }.lt);
        const cut = keys[idx[dsa.topk_pools - 1]];
        if (keys[idx[dsa.topk_pools]] == cut) ties += 1;
        if (cut == orderKey(0)) zeros_at_cut += 1;
        std.mem.sort(u32, idx[0..dsa.topk_pools], {}, std.sort.asc(u32));
        const tail = at + 1 - 4 * npool;
        try expect(gs[row] == 1 and gc[row] == @as(i32, @intCast(4 * dsa.topk_pools + tail)), "select row {d}: sparse 1, count {d} (got {d}, {d})", .{ row, 4 * dsa.topk_pools + tail, gs[row], gc[row] });
        const list = gt[row * dsa.width ..][0..dsa.width];
        for (idx[0..dsa.topk_pools], 0..) |p, i| for (0..4) |j| {
            try expect(list[4 * i + j] == @as(i32, @intCast(4 * p + j)), "select row {d} token {d}: {d}, the host's rule gives {d}", .{ row, 4 * i + j, list[4 * i + j], 4 * p + j });
        };
        for (0..tail) |j| try expect(list[4 * dsa.topk_pools + j] == @as(i32, @intCast(4 * npool + j)), "select row {d}: tail token {d}", .{ row, j });
    }
    // the same rows in two calls (row0 0 and half): the same lists
    const half = rows / 2;
    const t2 = try t.filled(rows * dsa.width * 4, 0xff);
    const c2 = try t.zeros(rows * 4);
    const s2 = try t.zeros(rows * 4);
    for ([_][2]usize{ .{ 0, half }, .{ half, rows - half } }) |w| {
        try dsa.selectTokens(t.k, t.s(), dqi + w[0] * qs * 2, qs, dw + (w[0] * wstride + wcol) * 2, wstride, dpk, dpos, w[0], pos, w[1], np, sc, t2 + w[0] * dsa.width * 4, c2 + w[0] * 4, s2 + w[0] * 4);
    }
    try sameBytes(t, try std.fmt.bufPrint(&name, "select at {d}: two calls' lists", .{pos}), tokens, t2, rows * dsa.width * 4);
    try sameBytes(t, try std.fmt.bufPrint(&name, "select at {d}: two calls' counts", .{pos}), counts, c2, rows * 4);
    try sameBytes(t, try std.fmt.bufPrint(&name, "select at {d}: two calls' sparse flags", .{pos}), sparse, s2, rows * 4);
    std.debug.print("PASS DSA indexer {d} rows at {d}: lists equal the host's rule exactly ({d} rows tie at the cut, {d} at a zero score)\n", .{ rows, pos, ties, zeros_at_cut });
}

test "GLM DSA: pool scores against float64, top-512 lists equal the host's rule (ties to the lower pool, -0 = +0)" {
    const t = try Rig.open(0x6304);
    defer t.close();
    // 6,000: 1,500 to 1,524 complete pools, 300 zero keys (scores of +-0 around the cut) and 40 exact copies
    try selection(t, 6000, 96, .{ 100, 400 }, .{ 500, 540 });
    // across the dense limit: rows at 2,040 .. 2,059 (510 to 515 complete pools)
    try selection(t, 2040, 20, .{ 10, 30 }, .{ 40, 50 });
}

/// The host rule (zig/tests/cuda/glm_ref_index.zig:77-95): the top 512 of a row's complete pools by order key, ties to
/// the lower pool, in ascending order.
fn hostTop(t: *Rig, row: []align(1) const f32) ![]u32 {
    const keys = try t.a().alloc(u32, row.len);
    for (keys, row) |*k, v| k.* = orderKey(v);
    const idx = try t.a().alloc(u32, row.len);
    for (idx, 0..) |*v, i| v.* = @intCast(i);
    std.mem.sort(u32, idx, @as([]const u32, keys), struct {
        fn lt(k: []const u32, x: u32, y: u32) bool {
            return k[x] > k[y] or (k[x] == k[y] and x < y);
        }
    }.lt);
    std.mem.sort(u32, idx[0..dsa.topk_pools], {}, std.sort.asc(u32));
    return idx[0..dsa.topk_pools];
}

test "GLM DSA: the top-512 select on exact ties equals the reference rule (ties to the lower pool)" {
    const t = try Rig.open(0x6308);
    defer t.close();
    const np = 2048;
    const cases = [_][]const u8{
        "all scores equal (+0)",
        "all scores equal (1.5)",
        "8 distinct values, each repeated",
        "duplicated pairs",
        "the 512th score tied with 300 pools across the cut",
        "the 512th score tied, ties above and below it",
        "integers 0..15 at random, many -inf-free ties",
        "random scores (no ties expected)",
    };
    const rows = cases.len * 2; // each case at two pool counts
    const sc = try t.a().alloc(f32, rows * np);
    @memset(sc, -std.math.inf(f32));
    const complete = try t.a().alloc(i32, rows);
    const ends = try t.a().alloc(i32, rows);
    for (0..rows) |row| {
        const npool: usize = if (row % 2 == 0) 1500 else 2048;
        complete[row] = @intCast(npool);
        ends[row] = @intCast(npool * 4 + row % 4); // tails of 0..3 tokens
        const r = sc[row * np ..][0..npool];
        switch (row / 2) {
            0 => @memset(r, 0.0),
            1 => @memset(r, 1.5),
            2 => for (r) |*v| {
                v.* = @floatFromInt(t.rng.below(8));
            },
            3 => for (r, 0..) |*v, i| {
                v.* = @floatCast(@as(f64, @floatFromInt(i / 2)) * 0.25 - 100);
            },
            4 => for (r, 0..) |*v, i| { // 400 pools above 2.0, 300 at exactly 2.0, the rest below: the cut is in the 2.0s
                v.* = if (i % 7 == 3 and i < 2800) 2.0 else @floatCast(t.rng.unit());
                if (i < 400 * 3 and i % 3 == 0) v.* = @floatCast(3.0 + t.rng.unit() * 0.5);
            },
            5 => for (r, 0..) |*v, i| {
                v.* = @floatFromInt(@as(i32, @intCast((i * 37) % 64)) - 32);
            },
            6 => for (r) |*v| {
                v.* = @floatFromInt(t.rng.below(16));
            },
            else => for (r) |*v| {
                v.* = @floatCast(t.rng.unit());
            },
        }
    }
    const s: dsa.Scratch = .{ .scores = try t.dev(sc), .complete = try t.dev(complete), .ends = try t.dev(ends) };
    const tokens = try t.filled(rows * dsa.width * 4, 0xff);
    const counts = try t.zeros(rows * 4);
    const sparse = try t.zeros(rows * 4);
    try dsa.selectLists(t.k, t.s(), s, np, rows, tokens, counts, sparse);
    const gt = try t.backAs(i32, tokens, rows * dsa.width);
    const gc = try t.backAs(i32, counts, rows);
    const gs = try t.backAs(i32, sparse, rows);
    for (0..rows) |row| {
        const npool: usize = @intCast(complete[row]);
        const r = sc[row * np ..][0..npool];
        const want = try hostTop(t, r);
        const keys = try t.a().alloc(u32, npool);
        for (keys, r) |*k, v| k.* = orderKey(v);
        var at_cut: usize = 0;
        const cut = orderKey(r[(try hostTopCutIndex(t, r))]);
        for (keys) |k| at_cut += @intFromBool(k == cut);
        const tail: usize = @as(usize, @intCast(ends[row])) - 4 * npool;
        try expect(gs[row] == 1 and gc[row] == @as(i32, @intCast(4 * dsa.topk_pools + tail)), "tie case {s} ({d} pools): sparse 1, count {d} (got {d}, {d})", .{ cases[row / 2], npool, 4 * dsa.topk_pools + tail, gs[row], gc[row] });
        const list = gt[row * dsa.width ..][0..dsa.width];
        for (want, 0..) |p, i| for (0..4) |j| {
            try expect(list[4 * i + j] == @as(i32, @intCast(4 * p + j)), "tie case {s} ({d} pools) token {d}: {d}, the reference rule gives {d}", .{ cases[row / 2], npool, 4 * i + j, list[4 * i + j], 4 * p + j });
        };
        for (0..tail) |j| try expect(list[4 * dsa.topk_pools + j] == @as(i32, @intCast(4 * npool + j)), "tie case {s}: tail token {d}", .{ cases[row / 2], j });
        std.debug.print("TIES {s}, {d} pools: the 512 picks equal the reference rule; {d} pools hold the cut's score\n", .{ cases[row / 2], npool, at_cut });
    }
    std.debug.print("PASS top-512 select on exact ties: {d} rows, every pick equals the lower-pool tie rule\n", .{rows});
}

/// The index of the 512th pick of hostTop's ranking (before the ascending sort): the cut's pool.
fn hostTopCutIndex(t: *Rig, row: []align(1) const f32) !usize {
    const keys = try t.a().alloc(u32, row.len);
    for (keys, row) |*k, v| k.* = orderKey(v);
    const idx = try t.a().alloc(u32, row.len);
    for (idx, 0..) |*v, i| v.* = @intCast(i);
    std.mem.sort(u32, idx, @as([]const u32, keys), struct {
        fn lt(k: []const u32, x: u32, y: u32) bool {
            return k[x] > k[y] or (k[x] == k[y] and x < y);
        }
    }.lt);
    return idx[dsa.topk_pools - 1];
}

/// float64 o = sum p c / sum p, p = exp(s - max), s = scale q . c; the CUDA test's bound (bf16 out and p, 2^-8 each,
/// (512 + 8) 2^-24 scale sum |q c|).
fn attnRef(t: *Rig, q: []const u16, cd: []const f64, keys: []const []const u32, scale: f64) !struct { want: []f64, tol: []f64 } {
    const lw = dsa.lw;
    const rows = keys.len;
    const want = try t.a().alloc(f64, rows * heads * lw);
    const tol = try t.a().alloc(f64, rows * heads * lw);
    const qd = try t.a().alloc(f64, lw);
    const o = try t.a().alloc(f64, lw);
    const spread = try t.a().alloc(f64, lw);
    var most_n: usize = 0;
    for (keys) |k| most_n = @max(most_n, k.len);
    const sv = try t.a().alloc(f64, most_n);
    const ds = try t.a().alloc(f64, most_n);
    for (keys, 0..) |set, row| for (0..heads) |h| {
        const base = (row * heads + h) * lw;
        for (qd, q[base..][0..lw]) |*x, v| x.* = bfValue(v);
        var most = -std.math.inf(f64);
        for (set, 0..) |key, i| {
            var dot: f64 = 0;
            var abs: f64 = 0;
            for (qd, cd[key * lw ..][0..lw]) |x, y| {
                dot += x * y;
                abs += @abs(x * y);
            }
            sv[i] = dot * scale;
            ds[i] = abs * scale * 520 * ulp;
            most = @max(most, sv[i]);
        }
        @memset(o, 0);
        @memset(spread, 0);
        var l: f64 = 0;
        for (set, 0..) |key, i| {
            const p = @exp(sv[i] - most);
            l += p;
            for (o, spread, cd[key * lw ..][0..lw]) |*od, *sd, v| {
                od.* += p * v;
                sd.* += p * @abs(v) * (b8 + ds[i]);
            }
        }
        for (0..lw) |dd| {
            want[base + dd] = o[dd] / l;
            tol[base + dd] = @abs(o[dd] / l) * b8 + spread[dd] / l + 1e-30;
        }
    };
    return .{ .want = want, .tol = tol };
}

fn parts(t: *Rig, rows: usize, n: usize) !dsa.Partials {
    return .{ .po = try t.zeros(rows * heads * n * dsa.lw * 4), .pms = try t.zeros(rows * heads * n * 2 * 4) };
}

const scale_qk: f32 = 0.0625; // qk_head_dim 256: 256^-0.5

test "GLM DSA: dense latent attention against float64; head blocks, single rows and split calls give the same bits" {
    const t = try Rig.open(0x6305);
    defer t.close();
    const lw = dsa.lw;
    const pos = 1430;
    const rows = 40;
    const n = dsa.partsFor(pos + rows);
    const q = try t.bfs(rows * heads * lw, 1.0);
    const dq = try t.dev(q);
    const c = try t.bfs((pos + rows) * lw, 1.0);
    const dc = try t.dev(c);
    const dpos = try posDev(t, pos);
    const row_bytes = heads * lw * 2;
    const out = try t.zeros(rows * row_bytes);
    try dsa.attentionTile(t.k, t.s(), dq, dc, dpos, 0, .dense, try parts(t, rows, n), out, rows, heads, n, scale_qk, dsa.heads_tile_wide);
    const keys = try t.a().alloc([]const u32, rows);
    for (keys, 0..) |*k, row| {
        const all = try t.a().alloc(u32, pos + row + 1);
        for (all, 0..) |*v, i| v.* = @intCast(i);
        k.* = all;
    }
    const ref = try attnRef(t, q, try t.f64s(c), keys, scale_qk);
    try closeCols(t, "latent attention dense 40 rows at 1,430, 64 heads, 32-head blocks", out, ref.want, ref.tol, lw);
    const narrow = try t.zeros(rows * row_bytes);
    try dsa.attentionTile(t.k, t.s(), dq, dc, dpos, 0, .dense, try parts(t, rows, n), narrow, rows, heads, n, scale_qk, dsa.heads_tile);
    try sameBytes(t, "dense attention: 16-head blocks equal the 32-head blocks", out, narrow, rows * row_bytes);
    var name: [96]u8 = undefined;
    for ([_][2]usize{ .{ 0, 1 }, .{ 17, 1 }, .{ 39, 1 }, .{ 9, 4 }, .{ 0, 24 }, .{ 24, 16 } }) |w| {
        const o = try t.zeros(w[1] * row_bytes);
        try dsa.attention(t.k, t.s(), dq + w[0] * row_bytes, dc, dpos, w[0], .dense, try parts(t, w[1], n), o, w[1], heads, n, scale_qk);
        try sameBytes(t, try std.fmt.bufPrint(&name, "dense attention: rows {d}..{d} alone", .{ w[0], w[0] + w[1] - 1 }), out + w[0] * row_bytes, o, w[1] * row_bytes);
    }
    std.debug.print("PASS dense latent attention over {d} keys: within float64 tolerance, both head blocks, single rows and split calls equal\n", .{pos + rows});
}

test "GLM DSA: listed latent attention (indexer lists and dense rows) against float64; windows give the same bits" {
    const t = try Rig.open(0x6306);
    defer t.close();
    const lw = dsa.lw;
    const ncache = 6000;
    const rows = 36;
    const w = dsa.width;
    const n = dsa.partsFor(w);
    const q = try t.bfs(rows * heads * lw, 1.0);
    const dq = try t.dev(q);
    const c = try t.bfs(ncache * lw, 1.0);
    const dc = try t.dev(c);
    const tok = try t.a().alloc(i32, rows * w);
    @memset(tok, -1);
    const cnt = try t.a().alloc(i32, rows);
    const sp = try t.a().alloc(i32, rows);
    const sets = try t.a().alloc([]const u32, rows);
    for (sets, 0..) |*set, row| {
        if (row % 6 == 5) { // a dense row: its first count positions
            const m = 1 + t.rng.below(w);
            const all = try t.a().alloc(u32, m);
            for (all, 0..) |*v, i| v.* = @intCast(i);
            set.* = all;
            cnt[row] = @intCast(m);
            sp[row] = 0;
            continue;
        }
        const want = if (row == 0) w else if (row == 1) 1 else 1 + t.rng.below(w);
        const pick = try t.a().alloc(u32, want);
        for (pick) |*v| v.* = @intCast(t.rng.below(ncache));
        std.mem.sort(u32, pick, {}, std.sort.asc(u32));
        var m: usize = 0;
        for (pick) |v| if (m == 0 or pick[m - 1] != v) {
            pick[m] = v;
            m += 1;
        };
        set.* = pick[0..m];
        cnt[row] = @intCast(m);
        sp[row] = 1;
        for (pick[0..m], tok[row * w ..][0..m]) |v, *x| x.* = @intCast(v);
    }
    const dtok = try t.dev(tok);
    const dcnt = try t.dev(cnt);
    const dsp = try t.dev(sp);
    const dpos = try posDev(t, 0);
    const keys: dsa.Keys = .{ .listed = .{ .ids = dtok, .id_stride = w, .counts = dcnt, .sparse = dsp } };
    const row_bytes = heads * lw * 2;
    const out = try t.zeros(rows * row_bytes);
    try dsa.attentionTile(t.k, t.s(), dq, dc, dpos, 0, keys, try parts(t, rows, n), out, rows, heads, n, scale_qk, dsa.heads_tile_wide);
    const ref = try attnRef(t, q, try t.f64s(c), sets, scale_qk);
    try closeCols(t, "latent attention listed 36 rows, up to 2,051 of 6,000 keys (every 6th row dense)", out, ref.want, ref.tol, lw);
    var name: [96]u8 = undefined;
    for ([_][2]usize{ .{ 0, 1 }, .{ 1, 1 }, .{ 5, 1 }, .{ 7, 3 }, .{ 20, 4 } }) |win| {
        const o = try t.zeros(win[1] * row_bytes);
        const kw: dsa.Keys = .{ .listed = .{ .ids = dtok + win[0] * w * 4, .id_stride = w, .counts = dcnt + win[0] * 4, .sparse = dsp + win[0] * 4 } };
        try dsa.attention(t.k, t.s(), dq + win[0] * row_bytes, dc, dpos, win[0], kw, try parts(t, win[1], n), o, win[1], heads, n, scale_qk);
        try sameBytes(t, try std.fmt.bufPrint(&name, "listed attention: rows {d}..{d} as a window", .{ win[0], win[0] + win[1] - 1 }), out + win[0] * row_bytes, o, win[1] * row_bytes);
    }
    std.debug.print("PASS listed latent attention over up to {d} tokens: within float64 tolerance, windows equal\n", .{w});
}

test "GLM DSA: rows under the dense limit get the same bits from a dense call and from a listed call" {
    // lane F's cut across the 2,051-key limit: rows at 2,040 .. 2,050 (510 .. 512 complete pools: dense rows) attend in
    // a dense call in one cut and inside a listed call in the other; there the select gives sparse 0, count P + 1
    const t = try Rig.open(0x6309);
    defer t.close();
    const lw = dsa.lw;
    const pos = 2040;
    const rows = 11;
    const w = dsa.width;
    const n = dsa.partsFor(pos + rows);
    const q = try t.bfs(rows * heads * lw, 1.0);
    const dq = try t.dev(q);
    const dc = try t.dev(try t.bfs((pos + rows) * lw, 1.0));
    const dpos = try posDev(t, pos);
    const row_bytes = heads * lw * 2;
    const dense = try t.zeros(rows * row_bytes);
    try dsa.attention(t.k, t.s(), dq, dc, dpos, 0, .dense, try parts(t, rows, n), dense, rows, heads, n, scale_qk);
    // the select's form for these rows: sparse 0, count = position + 1, the list unused
    const cnt = try t.a().alloc(i32, rows);
    for (cnt, 0..) |*c, r| c.* = @intCast(pos + r + 1);
    const zero = try t.a().alloc(i32, rows);
    @memset(zero, 0);
    const one = try t.a().alloc(i32, rows);
    @memset(one, 1);
    const tok = try t.a().alloc(i32, rows * w);
    @memset(tok, -1);
    for (0..rows) |r| for (0..pos + r + 1) |i| {
        tok[r * w + i] = @intCast(i);
    };
    const dtok = try t.dev(tok);
    const dcnt = try t.dev(cnt);
    var name: [112]u8 = undefined;
    for ([_]struct { sp: []const i32, what: []const u8 }{ .{ .sp = zero, .what = "sparse 0, count P + 1" }, .{ .sp = one, .what = "sparse 1, the list 0 .. P" } }) |c| {
        const keys: dsa.Keys = .{ .listed = .{ .ids = dtok, .id_stride = w, .counts = dcnt, .sparse = try t.dev(c.sp) } };
        for ([_]usize{ dsa.partsFor(w), dsa.partsFor(w) + 1 }) |np| {
            const o = try t.zeros(rows * row_bytes);
            try dsa.attention(t.k, t.s(), dq, dc, dpos, 0, keys, try parts(t, rows, np), o, rows, heads, np, scale_qk);
            try sameBytes(t, try std.fmt.bufPrint(&name, "rows at 2,040 .. 2,050: dense call = listed call ({s}, {d} parts)", .{ c.what, np }), dense, o, rows * row_bytes);
        }
    }
    std.debug.print("PASS dense rows across the limit: a dense call and a listed call (sparse 0 or a full list) give the same bits\n", .{});
}

/// Mean microseconds of `body` over `n` runs after 2 warm-ups (events on the stream).
fn timeUs(t: *Rig, n: usize, ctx: anytype, comptime body: fn (*Rig, @TypeOf(ctx)) anyerror!void) !f64 {
    try body(t, ctx);
    try body(t, ctx);
    const api = &t.base.r.api;
    var e: [2]@import("../../abi.zig").Event = .{ null, null };
    for (&e) |*x| try hruntime.check(api.hipEventCreateWithFlags(x, 0));
    defer for (e) |x| {
        _ = api.hipEventDestroy(x);
    };
    try hruntime.check(api.hipEventRecord(e[0], t.base.stream.handle));
    for (0..n) |_| try body(t, ctx);
    try hruntime.check(api.hipEventRecord(e[1], t.base.stream.handle));
    try hruntime.check(api.hipEventSynchronize(e[1]));
    var ms: f32 = 0;
    try hruntime.check(api.hipEventElapsedTime(&ms, e[0], e[1]));
    return @as(f64, ms) * 1000.0 / @as(f64, @floatFromInt(n));
}

test "GLM DSA: speed facts at the model's shapes (no gate)" {
    const t = try Rig.open(0x6307);
    defer t.close();
    const lw = dsa.lw;
    const ctx = 16384;
    const most = 2048;
    const cache = try t.dev(try t.bfs((ctx + most) * lw, 1.0));
    const qa = try t.dev(try t.bfs(most * heads * lw, 1.0));
    const out = try t.zeros(most * heads * lw * 2);
    const n = dsa.partsFor(dsa.width);
    const p512 = try parts(t, 512, n);
    // full lists: 2,051 ascending tokens a row (strided over the cache)
    const tok = try t.a().alloc(i32, most * dsa.width);
    for (0..most) |r| for (0..dsa.width) |i| {
        tok[r * dsa.width + i] = @intCast((i * 8 + r) % ctx);
    };
    const cnt = try t.a().alloc(i32, most);
    @memset(cnt, dsa.width);
    const sp = try t.a().alloc(i32, most);
    @memset(sp, 1);
    const keys: dsa.Keys = .{ .listed = .{ .ids = try t.dev(tok), .id_stride = dsa.width, .counts = try t.dev(cnt), .sparse = try t.dev(sp) } };
    const C = struct { rows: usize, keys: dsa.Keys, cache: u64, qa: u64, out: u64, p: dsa.Partials, pos: u64, parts: usize };
    const attn = struct {
        fn f(r: *Rig, c: C) anyerror!void {
            try dsa.attention(r.k, r.s(), c.qa, c.cache, c.pos, 0, c.keys, c.p, c.out, c.rows, heads, c.parts, scale_qk);
        }
    }.f;
    const pos16 = try posDev(t, ctx);
    for ([_]usize{ 1, 4, 512 }) |rows| {
        const us = try timeUs(t, 10, C{ .rows = rows, .keys = keys, .cache = cache, .qa = qa, .out = out, .p = p512, .pos = pos16, .parts = n }, attn);
        const flop = 2.0 * 2.0 * @as(f64, @floatFromInt(rows * heads * dsa.width * lw));
        std.debug.print("SPEED sparse latent attention, {d} rows x 64 heads x 2,051 keys: {d:.1} us ({d:.2} TFLOPS)\n", .{ rows, us, flop / us * 1e-6 });
    }
    const pos2k = try posDev(t, 1536);
    for ([_]usize{ 1, 512 }) |rows| {
        const nd = dsa.partsFor(1536 + rows);
        const us = try timeUs(t, 10, C{ .rows = rows, .keys = .dense, .cache = cache, .qa = qa, .out = out, .p = p512, .pos = pos2k, .parts = nd }, attn);
        std.debug.print("SPEED dense latent attention, {d} rows at 1,536 x 64 heads: {d:.1} us\n", .{ rows, us });
    }
    const wk = try t.dev(try t.bfs(heads * (dsa.qk + dsa.dv) * lw, 0.05)); // kv_b as stored
    const q = try t.dev(try t.bfs(most * heads * dsa.qk, 1.0));
    const P = struct { rows: usize, a: u64, w: u64, o: u64 };
    const absorb = struct {
        fn f(r: *Rig, c: P) anyerror!void {
            try dsa.absorbQ(r.k, r.s(), c.a, heads * dsa.qk, dsa.kvKeys(c.w), dsa.kv_head, c.o, c.rows, heads);
        }
    }.f;
    const expand = struct {
        fn f(r: *Rig, c: P) anyerror!void {
            try dsa.expandV(r.k, r.s(), c.a, dsa.kvValues(c.w), dsa.kv_head, c.o, c.rows, heads);
        }
    }.f;
    for ([_]usize{ 1, 4, 512, 2048 }) |rows| {
        const ua = try timeUs(t, 10, P{ .rows = rows, .a = q, .w = wk, .o = out }, absorb);
        const ue = try timeUs(t, 10, P{ .rows = rows, .a = qa, .w = wk, .o = out }, expand);
        std.debug.print("SPEED absorb_q {d} rows: {d:.1} us ({d:.0} GB/s of kv_b halves); expand_v: {d:.1} us\n", .{ rows, ua, 16.777216e6 / ua * 1e-3, ue });
    }
    const np = dsa.poolBucket(ctx, most, 1 << 20);
    const hd = dsa.index_heads;
    const qi = try t.dev(try t.bfs(most * hd * dsa.index_dims, 1.0));
    const wts = try t.dev(try t.bfs(most * hd, 1.0));
    const pk = try t.dev(try t.bfs(np * dsa.index_dims, 1.0));
    const sc: dsa.Scratch = .{ .scores = try t.zeros(512 * np * 4), .complete = try t.zeros(most * 4), .ends = try t.zeros(most * 4) };
    const toks = try t.zeros(most * dsa.width * 4);
    const cts = try t.zeros(most * 4);
    const sps = try t.zeros(most * 4);
    const X = struct { rows: usize, qi: u64, wts: u64, pk: u64, pos: u64, np: usize, sc: dsa.Scratch, toks: u64, cts: u64, sps: u64 };
    const sel = struct {
        fn f(r: *Rig, c: X) anyerror!void {
            try dsa.selectTokens(r.k, r.s(), c.qi, hd * dsa.index_dims, c.wts, hd, c.pk, c.pos, 0, ctx, c.rows, c.np, c.sc, c.toks, c.cts, c.sps);
        }
    }.f;
    for ([_]usize{ 1, 4, 512 }) |rows| {
        const us = try timeUs(t, 10, X{ .rows = rows, .qi = qi, .wts = wts, .pk = pk, .pos = pos16, .np = np, .sc = sc, .toks = toks, .cts = cts, .sps = sps }, sel);
        std.debug.print("SPEED index scores + top-512 select, {d} rows at 16,384 (4,096 pools): {d:.1} us\n", .{ rows, us });
    }
}
