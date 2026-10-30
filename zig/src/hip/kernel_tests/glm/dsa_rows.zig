//! GLM-5.3-Flash's DSA kernels under LANES.md code contract 7 (row invariance): a row's bytes do not depend on the
//! launch's row count, the chunk boundaries or the batch partners, and a prompt chunk matches decode at the seams.
//! A 2,048-row prompt chunk at positions 1,100 .. 3,147 crosses the dense limit (2,051 keys). Rows 937 (position 2,037,
//! dense) and 1,497 (position 2,597, sparse), offsets that are not multiples of 16 or 64, give the same bytes alone (a
//! 1-row call at their position), inside the chunk, and in a 4-row window as its row 2 beside random partner rows.
//! The index keys and pools are written three ways (one chunk call, 1-row decode calls, 4-row windows that straddle
//! the pools) and give the same caches. The attention runs on the select's own lists of the chunk, 64 heads.
const std = @import("std");
const t3 = @import("dsa.zig");
const dsa = @import("../../launches/glm_dsa.zig");

const Rig = t3.Rig;
const heads = t3.heads;
const expect = t3.expect;
const sameBytes = t3.sameBytes;
const posDev = t3.posDev;

const pos0 = 1100;
const chunk = 2048;
const probe = [_]usize{ 937, 1497 };
const win = 4;
const at_in_window = 2;

comptime {
    for (probe) |r| if (r % 16 == 0 or r % 64 == 0 or r < at_in_window or r + win - at_in_window > chunk) @compileError("probe row");
}

/// The 4-row window bytes for row r: random partner rows, row `at_in_window` = row r of `all` (row_elems u16 a row).
fn windowRows(t: *Rig, all: []const u16, row_elems: usize, r: usize, scale: f64) !u64 {
    const w = try t.bfs(win * row_elems, scale);
    @memcpy(w[at_in_window * row_elems ..][0..row_elems], all[r * row_elems ..][0..row_elems]);
    return t.dev(w);
}

/// Row r of a chunk output at `chunk_out` equals the 1-row output and row `at_in_window` of the window output.
fn threeWays(t: *Rig, what: []const u8, chunk_out: u64, alone: u64, window: u64, row_bytes: usize, r: usize) !void {
    var name: [192]u8 = undefined;
    try sameBytes(t, try std.fmt.bufPrint(&name, "{s}: row {d} (position {d}) alone = in the 2,048-row chunk", .{ what, r, pos0 + r }), chunk_out + r * row_bytes, alone, row_bytes);
    try sameBytes(t, try std.fmt.bufPrint(&name, "{s}: row {d} in a 4-row window beside random rows = in the chunk", .{ what, r }), chunk_out + r * row_bytes, window + at_in_window * row_bytes, row_bytes);
}

test "GLM DSA rows (contract 7): latent write, absorb and expand: a row alone, in a 2,048-row chunk and in a window" {
    const t = try Rig.open(0x7301);
    defer t.close();
    const lw = dsa.lw;
    // the latent cache write: one chunk call, then each probe row alone and in a window, into separate caches
    {
        const stride = 2208; // the kv_a columns of [q_a | kv_a | wk | weights_proj] rows (column 1,536)
        const lat = try t.bfs(chunk * stride, 1.0);
        const cap = pos0 + chunk + win;
        const c_chunk = try t.filled(cap * lw * 2, 0xab);
        try dsa.latentWrite(t.k, t.s(), (try t.dev(lat)) + 1536 * 2, stride, c_chunk, try posDev(t, pos0), 0, chunk);
        for (probe) |r| {
            const c1 = try t.filled(cap * lw * 2, 0xab);
            try dsa.latentWrite(t.k, t.s(), (try t.dev(lat[r * stride ..][0..stride])) + 1536 * 2, stride, c1, try posDev(t, pos0 + r), 0, 1);
            const cw = try t.filled(cap * lw * 2, 0xab);
            try dsa.latentWrite(t.k, t.s(), (try windowRows(t, lat, stride, r, 1.0)) + 1536 * 2, stride, cw, try posDev(t, pos0 + r - at_in_window), 0, win);
            const p = (pos0 + r) * lw * 2;
            try sameBytes(t, "latent write: the slot of a row alone = written by the chunk", c_chunk + p, c1 + p, lw * 2);
            try sameBytes(t, "latent write: the slot of a row in a window = written by the chunk", c_chunk + p, cw + p, lw * 2);
        }
    }
    const kvb = try t.dev(try t.bfs(heads * (dsa.qk + dsa.dv) * lw, 0.05));
    {
        const qs = heads * dsa.qk + dsa.index_heads * dsa.index_dims; // [q_b | wq_b] rows
        const q = try t.bfs(chunk * qs, 1.0);
        const row_bytes = heads * lw * 2;
        const out = try t.zeros(chunk * row_bytes);
        try dsa.absorbQ(t.k, t.s(), try t.dev(q), qs, dsa.kvKeys(kvb), dsa.kv_head, out, chunk, heads);
        for (probe) |r| {
            const one = try t.zeros(row_bytes);
            try dsa.absorbQ(t.k, t.s(), try t.dev(q[r * qs ..][0..qs]), qs, dsa.kvKeys(kvb), dsa.kv_head, one, 1, heads);
            const w = try t.zeros(win * row_bytes);
            try dsa.absorbQ(t.k, t.s(), try windowRows(t, q, qs, r, 1.0), qs, dsa.kvKeys(kvb), dsa.kv_head, w, win, heads);
            try threeWays(t, "absorb_q", out, one, w, row_bytes, r);
        }
    }
    {
        const ol = try t.bfs(chunk * heads * lw, 1.0);
        const row_bytes = heads * dsa.dv * 2;
        const out = try t.zeros(chunk * row_bytes);
        try dsa.expandV(t.k, t.s(), try t.dev(ol), dsa.kvValues(kvb), dsa.kv_head, out, chunk, heads);
        for (probe) |r| {
            const one = try t.zeros(row_bytes);
            try dsa.expandV(t.k, t.s(), try t.dev(ol[r * heads * lw ..][0 .. heads * lw]), dsa.kvValues(kvb), dsa.kv_head, one, 1, heads);
            const w = try t.zeros(win * row_bytes);
            try dsa.expandV(t.k, t.s(), try windowRows(t, ol, heads * lw, r, 1.0), dsa.kvValues(kvb), dsa.kv_head, w, win, heads);
            try threeWays(t, "expand_v", out, one, w, row_bytes, r);
        }
    }
    std.debug.print("PASS DSA rows: latent write, absorb_q and expand_v give each probe row the same bytes alone, in the 2,048-row chunk and in a window\n", .{});
}

test "GLM DSA rows (contract 7): index keys and pools from one chunk call = 1-row decode calls = 4-row windows" {
    const t = try Rig.open(0x7302);
    defer t.close();
    const d = dsa.index_dims;
    const kstride = 2208; // the index key at column 2,048 of [q_a | kv_a | wk | weights_proj] rows
    const kraw = try t.dev(try t.bfs(chunk * kstride, 2.0));
    const gate_h = try t.a().alloc(f32, chunk * d);
    for (gate_h) |*g| g.* = @floatCast(t.rng.unit() * 3.0);
    const gate = try t.dev(gate_h);
    const lnw = try t.dev(try t.bfs(d, 1.0));
    const lnb = try t.dev(try t.bfs(d, 0.5));
    const ape = try t.dev(try t.bfs(dsa.pool * d, 1.0));
    const end = pos0 + chunk;
    const npools = end / dsa.pool + 2;
    const Cache = struct { ik: u64, ig: u64, pk: u64 };
    var c: [3]Cache = undefined;
    for (&c) |*x| x.* = .{ .ik = try t.filled(end * d * 2, 0xff), .ig = try t.filled(end * d * 2, 0xff), .pk = try t.filled(npools * d * 2, 0xff) };
    const dpos = try posDev(t, pos0);
    const one = struct {
        fn call(r: *Rig, x: Cache, kr: u64, g: u64, w: u64, b: u64, ap: u64, p: u64, row0: usize, rows: usize) !void {
            try dsa.indexUpdate(r.k, r.s(), kr + row0 * kstride * 2 + 2048 * 2, kstride, g + row0 * d * 4, w, b, ap, x.ik, x.ig, x.pk, p, row0, rows);
        }
    }.call;
    // 1. one call of the whole chunk; 2. one call a row (decode); 3. windows of 4 from row 2 (2 rows first, a tail)
    try one(t, c[0], kraw, gate, lnw, lnb, ape, dpos, 0, chunk);
    for (0..chunk) |r| try one(t, c[1], kraw, gate, lnw, lnb, ape, dpos, r, 1);
    try one(t, c[2], kraw, gate, lnw, lnb, ape, dpos, 0, 2);
    var r0: usize = 2;
    while (r0 < chunk) : (r0 += win) try one(t, c[2], kraw, gate, lnw, lnb, ape, dpos, r0, @min(win, chunk - r0));
    for ([_][]const u8{ "1-row decode calls", "4-row windows across the pools" }, 1..) |how, i| {
        var name: [192]u8 = undefined;
        try sameBytes(t, try std.fmt.bufPrint(&name, "index keys: positions 1,100 .. 3,147 from {s} = from one chunk call", .{how}), c[0].ik + pos0 * d * 2, c[i].ik + pos0 * d * 2, chunk * d * 2);
        try sameBytes(t, try std.fmt.bufPrint(&name, "index gates: from {s} = from one chunk call", .{how}), c[0].ig + pos0 * d * 2, c[i].ig + pos0 * d * 2, chunk * d * 2);
        try sameBytes(t, try std.fmt.bufPrint(&name, "pooled keys: pools 275 .. 786 (and the untouched rest) from {s} = from one chunk call", .{how}), c[0].pk, c[i].pk, npools * d * 2);
    }
    std.debug.print("PASS DSA rows: index keys, gates and pools are the same from a 2,048-row chunk, from decode rows and from windows\n", .{});
}

test "GLM DSA rows (contract 7): scores, top-512 lists and attention: a row alone, in a 2,048-row chunk and in a window" {
    const t = try Rig.open(0x7303);
    defer t.close();
    const lw = dsa.lw;
    const hd = dsa.index_heads;
    const d = dsa.index_dims;
    const w = dsa.width;
    // the indexer: qi at column 16,384 of [q_b | wq_b] rows, the head weights at column 2,176 of xProj rows
    const qs = heads * dsa.qk + hd * d;
    const qcol = heads * dsa.qk;
    const ws = 2208;
    const wcol = 2176;
    const qrows = try t.bfs(chunk * qs, 1.0);
    const wrows = try t.bfs(chunk * ws, 1.0);
    const npools = (pos0 + chunk) / dsa.pool;
    const pk = try t.dev(try t.bfs((npools + 2) * d, 1.0));
    const np = dsa.poolBucket(pos0, chunk, 1 << 20);
    const dpos = try posDev(t, pos0);
    const Lists = struct { tokens: u64, counts: u64, sparse: u64, scores: u64, np: usize };
    const select = struct {
        fn call(r: *Rig, q: u64, wt: u64, pkd: u64, p: u64, row0: usize, host: usize, rows: usize, n: usize) !Lists {
            const sc: dsa.Scratch = .{ .scores = try r.zeros(rows * n * 4), .complete = try r.zeros(rows * 4), .ends = try r.zeros(rows * 4) };
            const l: Lists = .{ .tokens = try r.filled(rows * dsa.width * 4, 0xff), .counts = try r.zeros(rows * 4), .sparse = try r.zeros(rows * 4), .scores = sc.scores, .np = n };
            try dsa.selectTokens(r.k, r.s(), q + qcol * 2, qs, wt + wcol * 2, ws, pkd, p, row0, host, rows, n, sc, l.tokens, l.counts, l.sparse);
            return l;
        }
    }.call;
    const dq = try t.dev(qrows);
    const dw = try t.dev(wrows);
    const all = try select(t, dq, dw, pk, dpos, 0, pos0, chunk, np);
    var name: [192]u8 = undefined;
    // the attention's inputs: a latent cache over every position, qa rows, the chunk's own lists
    const cache = try t.dev(try t.bfs((pos0 + chunk + win) * lw, 1.0));
    const qa = try t.bfs(chunk * heads * lw, 1.0);
    const dqa = try t.dev(qa);
    const row_bytes = heads * lw * 2;
    const n = dsa.partsFor(w); // listed rows: at most 2,051 entries (dense rows below the limit count P + 1 <= 2,051)
    const out = try t.zeros(chunk * row_bytes);
    const part_all = try t3.parts(t, chunk, n);
    const keys_all: dsa.Keys = .{ .listed = .{ .ids = all.tokens, .id_stride = w, .counts = all.counts, .sparse = all.sparse } };
    try dsa.attention(t.k, t.s(), dqa, cache, dpos, 0, keys_all, part_all, out, chunk, heads, n, t3.scale_qk);
    const tok_h = try t.backAs(i32, all.tokens, chunk * w);
    const cnt_h = try t.backAs(i32, all.counts, chunk);
    const sp_h = try t.backAs(i32, all.sparse, chunk);
    for (probe) |r| {
        const at = pos0 + r;
        const npool = (at + 1) / dsa.pool;
        // the select: alone (its own bucket) and in a window at at - 2 beside random rows
        const alone = try select(t, dq + r * qs * 2, dw + r * ws * 2, pk, dpos, r, pos0, 1, dsa.poolBucket(at, 1, 1 << 20));
        const wq = try windowRows(t, qrows, qs, r, 1.0);
        const ww = try windowRows(t, wrows, ws, r, 1.0);
        const wl = try select(t, wq, ww, pk, try posDev(t, at - at_in_window), 0, at - at_in_window, win, dsa.poolBucket(at - at_in_window, win, 1 << 20));
        try sameBytes(t, try std.fmt.bufPrint(&name, "select: row {d} alone, its {d} pool scores = in the chunk", .{ r, npool }), all.scores + r * np * 4, alone.scores, npool * 4);
        try sameBytes(t, try std.fmt.bufPrint(&name, "select: row {d} in a window, its pool scores = in the chunk", .{r}), all.scores + r * np * 4, wl.scores + at_in_window * wl.np * 4, npool * 4);
        for ([_]Lists{ alone, wl }, [_]usize{ 0, at_in_window }, [_][]const u8{ "alone", "in a window" }) |l, i, how| {
            try sameBytes(t, try std.fmt.bufPrint(&name, "select: row {d} {s}, its token list = in the chunk", .{ r, how }), all.tokens + r * w * 4, l.tokens + i * w * 4, w * 4);
            try sameBytes(t, try std.fmt.bufPrint(&name, "select: row {d} {s}, its count and sparse flag = in the chunk", .{ r, how }), all.counts + r * 4, l.counts + i * 4, 4);
            try sameBytes(t, "select: sparse flag", all.sparse + r * 4, l.sparse + i * 4, 4);
        }
        std.debug.print("ROW {d} at {d}: {s}, count {d}\n", .{ r, at, if (sp_h[r] != 0) "sparse list" else "dense", cnt_h[r] });
        // the attention: alone on its own list, and in a window whose partners take other rows' lists
        const one = try t.zeros(row_bytes);
        const k1: dsa.Keys = .{ .listed = .{ .ids = all.tokens + r * w * 4, .id_stride = w, .counts = all.counts + r * 4, .sparse = all.sparse + r * 4 } };
        try dsa.attention(t.k, t.s(), dqa + r * row_bytes, cache, try posDev(t, at), r, k1, try t3.parts(t, 1, n), one, 1, heads, n, t3.scale_qk);
        const wtok = try t.a().alloc(i32, win * w);
        const wcnt = try t.a().alloc(i32, win);
        const wsp = try t.a().alloc(i32, win);
        for (0..win) |k| {
            const src = if (k == at_in_window) r else t.rng.below(chunk);
            @memcpy(wtok[k * w ..][0..w], tok_h[src * w ..][0..w]);
            wcnt[k] = cnt_h[src];
            wsp[k] = sp_h[src];
        }
        const kw: dsa.Keys = .{ .listed = .{ .ids = try t.dev(wtok), .id_stride = w, .counts = try t.dev(wcnt), .sparse = try t.dev(wsp) } };
        const wo = try t.zeros(win * row_bytes);
        try dsa.attention(t.k, t.s(), try windowRows(t, qa, heads * lw, r, 1.0), cache, try posDev(t, at - at_in_window), 0, kw, try t3.parts(t, win, n), wo, win, heads, n, t3.scale_qk);
        try threeWays(t, "listed latent attention", out, one, wo, row_bytes, r);
        if (sp_h[r] == 0) { // a dense row: the dense call alone and in a dense window (partners: random q at P - 2, P - 1, P + 1)
            const nd = dsa.partsFor(at + win);
            const d1 = try t.zeros(row_bytes);
            try dsa.attention(t.k, t.s(), dqa + r * row_bytes, cache, try posDev(t, at), 0, .dense, try t3.parts(t, 1, nd), d1, 1, heads, nd, t3.scale_qk);
            const dwo = try t.zeros(win * row_bytes);
            try dsa.attention(t.k, t.s(), try windowRows(t, qa, heads * lw, r, 1.0), cache, try posDev(t, at - at_in_window), 0, .dense, try t3.parts(t, win, nd), dwo, win, heads, nd, t3.scale_qk);
            try threeWays(t, "dense latent attention (the chunk's row from its listed call)", out, d1, dwo, row_bytes, r);
        }
    }
    std.debug.print("PASS DSA rows: scores, lists and latent attention (64 heads, the chunk's own lists) give each probe row the same bytes alone, in the 2,048-row chunk and in a window\n", .{});
}
