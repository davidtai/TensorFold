//! GLM-5.3-Flash's fast DSA kernels (G3-P3 expand, G3-P4 index scores) against their first forms (the _ref kernels,
//! kept as the byte oracles): the same bytes at 1, 4, 300, 512 and 2,048 rows, at the decode and prompt shapes, with
//! kv_b and the projection rows as the loader places them. The register-tiled absorb (G3-AE) against the row form
//! (tf_glm_absorb_q, which dsa.zig tests against float64), at odd row counts and across tile tails too. Then the speed
//! facts of every form.
const std = @import("std");
const t3 = @import("dsa.zig");
const dsa = @import("../../launches/glm_dsa.zig");

const Rig = t3.Rig;
const heads = t3.heads;
const sameBytes = t3.sameBytes;
const posDev = t3.posDev;

const row_counts = [_]usize{ 1, 4, 300, 512, 2048 };

test "GLM DSA fast (exact): expand_v gives the _ref kernel's bytes at 1, 4, 300, 512 and 2,048 rows" {
    const t = try Rig.open(0x7401);
    defer t.close();
    const lw = dsa.lw;
    const most = row_counts[row_counts.len - 1];
    const kvb = try t.dev(try t.bfs(heads * (dsa.qk + dsa.dv) * lw, 0.05));
    const ol = try t.dev(try t.bfs(most * heads * lw, 1.0));
    var name: [128]u8 = undefined;
    for (row_counts) |rows| {
        const vn = [2]u64{ try t.zeros(rows * heads * dsa.dv * 2), try t.zeros(rows * heads * dsa.dv * 2) };
        for ([_]dsa.Form{ .fast, .ref }, 0..) |form, i| {
            try dsa.expandVForm(t.k, form, t.s(), ol, dsa.kvValues(kvb), dsa.kv_head, vn[i], rows, heads);
        }
        try sameBytes(t, try std.fmt.bufPrint(&name, "expand_v fast = ref, {d} rows", .{rows}), vn[0], vn[1], rows * heads * dsa.dv * 2);
    }
    std.debug.print("PASS DSA fast: expand_v equals its _ref kernel's bytes at every row count\n", .{});
}

/// Row counts of the tile test: 1 and 4 (decode), odd counts and counts on both sides of the 64-row tile (a tile tail
/// of 1, 13 and 63 rows), a prompt window, and the 8,192-row prompt chunk.
const tile_counts = [_]usize{ 1, 4, 7, 63, 64, 65, 127, 333, 512, 2047, 2048, 8192 };

test "GLM DSA tile (exact): the absorb_q tile gives the row form's bytes at odd counts, across tile tails, to 8,192 rows" {
    const t = try Rig.open(0x7405);
    defer t.close();
    const lw = dsa.lw;
    const most = tile_counts[tile_counts.len - 1];
    const kvb = try t.dev(try t.bfs(heads * (dsa.qk + dsa.dv) * lw, 0.05));
    const qs = heads * dsa.qk + dsa.index_heads * dsa.index_dims; // [q_b | wq_b] rows
    const q = try t.dev(try t.bfs(most * qs, 1.0));
    var name: [128]u8 = undefined;
    for (tile_counts) |rows| {
        // one row more than written, the same fill in both: a row the tile misses or writes past the end differs
        var qa: [2]u64 = undefined;
        for ([_]dsa.Tiling{ .tile, .rows }, 0..) |form, i| {
            qa[i] = try t.filled((rows + 1) * heads * lw * 2, 0x5a);
            try dsa.absorbQWith(t.k, form, t.s(), q, qs, dsa.kvKeys(kvb), dsa.kv_head, qa[i], rows, heads);
        }
        try sameBytes(t, try std.fmt.bufPrint(&name, "absorb_q tile = rows form, {d} rows", .{rows}), qa[0], qa[1], (rows + 1) * heads * lw * 2);
    }
    std.debug.print("PASS DSA tile: the absorb_q tile equals the row form's bytes at every row count\n", .{});
}

/// The pool scores of `rows` rows at `pos` in both forms: the score rows, complete and ends equal.
fn scoresBoth(t: *Rig, pos: usize, rows: usize) !void {
    const hd = dsa.index_heads;
    const d = dsa.index_dims;
    const qs = heads * dsa.qk + hd * d;
    const ws = 2208;
    const np = dsa.poolBucket(pos, rows, 1 << 20);
    const npools = (pos + rows) / dsa.pool;
    const q = try t.dev(try t.bfs(rows * qs, 1.0));
    const w = try t.dev(try t.bfs(rows * ws, 1.0));
    const pk = try t.dev(try t.bfs((npools + 2) * d, 1.0));
    const dpos = try posDev(t, pos);
    var sc: [2]dsa.Scratch = undefined;
    for ([_]dsa.Form{ .fast, .ref }, 0..) |form, i| {
        sc[i] = .{ .scores = try t.filled(rows * np * 4, 0x5a), .complete = try t.zeros(rows * 4), .ends = try t.zeros(rows * 4) };
        try dsa.scores(t.k, form, t.s(), q + heads * dsa.qk * 2, qs, w + 2176 * 2, ws, pk, dpos, 0, pos, rows, np, sc[i]);
    }
    var name: [128]u8 = undefined;
    try sameBytes(t, try std.fmt.bufPrint(&name, "index scores fast = ref, {d} rows at {d} ({d} pool columns)", .{ rows, pos, np }), sc[0].scores, sc[1].scores, rows * np * 4);
    try sameBytes(t, "index scores: complete and ends", sc[0].complete, sc[1].complete, rows * 4);
    try sameBytes(t, "index scores: ends", sc[0].ends, sc[1].ends, rows * 4);
}

test "GLM DSA fast (exact): the index scores give the _ref kernel's bytes at 1, 4, 300, 512 and 2,048 rows at 16K and across the dense limit" {
    const t = try Rig.open(0x7402);
    defer t.close();
    for (row_counts) |rows| try scoresBoth(t, 16384, rows);
    try scoresBoth(t, 1100, 2048); // rows at 1,100 .. 3,147: 275 .. 787 complete pools, the first ones dense (not scored)
    try scoresBoth(t, 2040, 16); // rows at 2,040 .. 2,055: 510 .. 514 complete pools, the dense limit at row 2,051 (513)
    try scoresBoth(t, 6001, 7); // complete-pool counts not a multiple of 4 or 16
    std.debug.print("PASS DSA fast: the index scores equal the _ref kernel's bytes at every shape\n", .{});
}

test "GLM DSA fast: speed facts, fast vs ref (no gate)" {
    const t = try Rig.open(0x7403);
    defer t.close();
    const lw = dsa.lw;
    const most = 2048;
    const kvb = try t.dev(try t.bfs(heads * (dsa.qk + dsa.dv) * lw, 0.05));
    const qs = heads * dsa.qk + dsa.index_heads * dsa.index_dims;
    const q = try t.dev(try t.bfs(most * qs, 1.0));
    const ol = try t.dev(try t.bfs(most * heads * lw, 1.0));
    const out = try t.zeros(most * heads * lw * 2);
    // absorb: form 0 the tile, 1 the row form; expand: form 0 tf_glm_expand_v, 1 the _ref
    const P = struct { rows: usize, form: u8, q: u64, ol: u64, kvb: u64, out: u64 };
    const absorb = struct {
        fn f(r: *Rig, c: P) anyerror!void {
            const tl: dsa.Tiling = if (c.form == 0) .tile else .rows;
            try dsa.absorbQWith(r.k, tl, r.s(), c.q, qs, dsa.kvKeys(c.kvb), dsa.kv_head, c.out, c.rows, heads);
        }
    }.f;
    const expand = struct {
        fn f(r: *Rig, c: P) anyerror!void {
            const wv = dsa.kvValues(c.kvb);
            const form: dsa.Form = if (c.form == 0) .fast else .ref;
            try dsa.expandVForm(r.k, form, r.s(), c.ol, wv, dsa.kv_head, c.out, c.rows, heads);
        }
    }.f;
    std.debug.print("\n", .{});
    for ([_]usize{ 1, 4, 512, 2048 }) |rows| {
        var ex: [2]f64 = undefined;
        var ab: [2]f64 = undefined;
        for (0..2) |i| ex[i] = try t3.timeUs(t, 10, P{ .rows = rows, .form = @intCast(i), .q = q, .ol = ol, .kvb = kvb, .out = out }, expand);
        for (0..2) |i| ab[i] = try t3.timeUs(t, 10, P{ .rows = rows, .form = @intCast(i), .q = q, .ol = ol, .kvb = kvb, .out = out }, absorb);
        std.debug.print("SPEED {d} rows: absorb_q tile {d:.1} us (rows form {d:.1}); expand_v fast {d:.1} us (ref {d:.1})\n", .{ rows, ab[0], ab[1], ex[0], ex[1] });
    }
    const d = dsa.index_dims;
    const ws = 2208;
    const ctx = 16384;
    const npmax = dsa.poolBucket(ctx, most, 1 << 20);
    const qi = try t.dev(try t.bfs(most * qs, 1.0));
    const w = try t.dev(try t.bfs(most * ws, 1.0));
    const pk = try t.dev(try t.bfs(((ctx + most) / dsa.pool + 2) * d, 1.0));
    const pos = try posDev(t, ctx);
    const sc: dsa.Scratch = .{ .scores = try t.zeros(most * npmax * 4), .complete = try t.zeros(most * 4), .ends = try t.zeros(most * 4) };
    const toks = try t.zeros(most * dsa.width * 4);
    const cts = try t.zeros(most * 4);
    const sps = try t.zeros(most * 4);
    const X = struct { rows: usize, form: dsa.Form, qi: u64, w: u64, pk: u64, pos: u64, sc: dsa.Scratch, toks: u64, cts: u64, sps: u64 };
    const score = struct {
        fn f(r: *Rig, c: X) anyerror!void {
            const np = dsa.poolBucket(ctx, c.rows, 1 << 20);
            try dsa.scores(r.k, c.form, r.s(), c.qi + heads * dsa.qk * 2, qs, c.w + 2176 * 2, ws, c.pk, c.pos, 0, ctx, c.rows, np, c.sc);
        }
    }.f;
    const sel = struct {
        fn f(r: *Rig, c: X) anyerror!void {
            try dsa.selectLists(r.k, r.s(), c.sc, dsa.poolBucket(ctx, c.rows, 1 << 20), c.rows, c.toks, c.cts, c.sps);
        }
    }.f;
    for ([_]usize{ 1, 4, 512, 2048 }) |rows| {
        var us: [3]f64 = undefined;
        for ([_]dsa.Form{ .fast, .ref }, 0..) |form, i| us[i] = try t3.timeUs(t, 10, X{ .rows = rows, .form = form, .qi = qi, .w = w, .pk = pk, .pos = pos, .sc = sc, .toks = toks, .cts = cts, .sps = sps }, score);
        us[2] = try t3.timeUs(t, 10, X{ .rows = rows, .form = .fast, .qi = qi, .w = w, .pk = pk, .pos = pos, .sc = sc, .toks = toks, .cts = cts, .sps = sps }, sel);
        std.debug.print("SPEED {d} rows at 16,384 (4,096 .. 4,607 pools): index scores fast {d:.1} us (ref {d:.1}); top-512 select alone {d:.1} us\n", .{ rows, us[0], us[1], us[2] });
    }
}
