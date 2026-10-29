//! The routed MoE of MLX affine experts: router, pick rule, plan, gate and up, activation, down, slots summed in order.

const std = @import("std");
const ops = @import("ops/ops.zig");

const Ops = ops.Ops;
const counting = @import("launches/util.zig").counting;
const Tensor = ops.Tensor;
const Kind = ops.Kind;
const Projection = ops.Projection;

/// A layer's experts, shared expert last: gate and up stacked as one (E + 1, 2 NI, ...) weight, down (E + 1, D, ...).
pub const Experts = struct {
    fused: Projection,
    down: Projection,
    count: usize,
    width: usize,
    dims: usize,
    limit: f32 = 0,
};

/// The router's rows in fp32 (E + 1, D), the shared expert's gate last, and the experts.
pub const Routed = struct {
    rows32: u64,
    experts: Experts,
    top_k: usize,
    /// The router's rows, E + 1: the model's expert count, whatever share of the experts this rank holds.
    rows: usize,
    /// A rank's expert ids (int32, E + 1): each id's place among its own, -1 for another rank's; zero on one rank.
    remap: u64 = 0,
    /// Plan the rows' own pairs, not a power-of-two bucket's (no graph replays the launches at other row counts).
    exact: bool = false,
    /// The router's rows in bf16 (E + 1, D), the same values as `rows32`: a decode call's router and pick take one launch
    /// over them (0: the two launches over `rows32`).
    rows16: u64 = 0,
    /// A u32, zero between launches, that the one-launch router and pick counts its finished blocks in.
    done: u64 = 0,
    /// B-5: int32 [8][3] then [8]: for r = 1..8 rows the shared expert's one-item plan (shared id, 0, r) and its members
    /// (row j's shared pair, j * slots + top_k); 0: the shared expert stays in the routed launches.
    shared_plan: u64 = 0,

    /// Routed experts, the shared one not counted.
    pub fn count(r: Routed) usize {
        return r.rows - 1;
    }
};

/// experts.py: the decode tile's most rows, the GEMM tile's rows and where it starts, items' pair cap.
const LANE_ROWS = 8;
const BLOCK_ROWS = 128;
const BLOCK_FROM = 64;
const TILE = 16;

/// MoEBuffers' size for `rows` rows: the next power of two from 16.
pub fn bucket(rows: usize) usize {
    const bits = if (rows <= 1) 0 else std.math.log2_int_ceil(usize, rows);
    return @as(usize, 1) << @intCast(@max(4, bits));
}

/// max_items(pairs, experts, min(TILE, LANE_ROWS)): an item a used expert plus one a tile of pairs past its first.
pub fn capacity(pairs: usize, experts: usize) usize {
    return @min(pairs, experts) + pairs / @min(TILE, LANE_ROWS);
}

/// The live items of a prompt plan of `pairs` pairs over `experts` cut at `tile` pairs an item (MG-1): with a used
/// experts, sum ceil(c / tile) = a + sum floor((c - 1) / tile) <= a + (pairs - a) / tile, which grows with a, so
/// a = min(pairs, experts) bounds it; equal when every used expert has tile k + 1 pairs plus less than a tile spare.
/// tf_moe_route puts the live items first, so a prompt's expert launches take this z and not capacity (a window's).
pub fn planItems(pairs: usize, experts: usize, tile: usize) usize {
    const a = @min(pairs, experts);
    return a + (pairs - a) / tile;
}

/// tile_for: the decode tile below BLOCK_FROM rows, the GEMM tile from there; a lane round's rows keep the decode tile.
pub fn tileFor(o: Ops, rows: usize) usize {
    return if (rows < BLOCK_FROM or o.window) LANE_ROWS else BLOCK_ROWS;
}

/// `x` (rows, D) to (rows, D) in its dtype: the top-k experts plus the shared one; a tp rank returns its fp32 share.
pub fn run(o: Ops, kind: Kind, r: Routed, x: Tensor, rows: usize, prefill: bool) ops.Error!Tensor {
    _ = prefill; // the affine experts' arithmetic does not depend on it (the plan's tile comes from tile_for)
    const ex = r.experts;
    const slots = r.top_k + 1;
    const size = if (r.exact) rows else bucket(rows);
    const pairs = rows * slots;
    const cap = capacity(size * slots, r.rows);
    const logits = try o.arena.of(f32, rows * r.rows);
    const pick = try o.arena.of(i32, size * slots);
    const wts = try o.arena.of(f32, size * slots);
    const items = try o.arena.of(i32, (cap + 1) * 3);
    const members = try o.arena.of(i32, size * slots);
    const count: usize = cap;
    var tile: usize = undefined;
    if (r.remap != 0) {
        try o.moeRouter(x, r.rows32, logits, rows, ex.dims, r.rows);
        // every rank picks the same pairs, runs the ones it holds (the others group under an id past its experts)
        try o.moeSelect(logits, pick, wts, null, 0, rows, r.count(), r.top_k);
        const local = try o.arena.of(i32, size * slots);
        try o.moeLocalize(pick, r.remap, local, pairs, ex.count);
        tile = tileFor(o, rows);
        try o.moeRoute(local, pairs, r.rows, tile, members, items, cap);
        try o.moeForeignItems(items, cap, ex.count);
        const both = try o.projectRouted(x, ex.fused, items, count, members, pairs, slots, @min(tile, rows));
        const act = try o.arena.take(pairs * ex.width * kind.size());
        const act_t: Tensor = .{ .ptr = act, .kind = kind };
        try o.moeAct(both, act_t, pairs, ex.width, ex.limit);
        const y = try o.projectRouted(act_t, ex.down, items, count, members, pairs, 1, @min(tile, rows));
        try o.moeZeroForeign(y, local, ex.count, pairs, ex.dims);
        const share = try o.arena.of(f32, rows * ex.dims);
        try o.moeCombine(y, wts, .{ .ptr = share, .kind = .f32 }, rows, slots, ex.dims);
        return .{ .ptr = share, .kind = .f32 };
    }
    const p = try expertParts(o, kind, r, x, rows, logits, pick, wts, items, members, cap);
    const out = try o.arena.take(rows * ex.dims * kind.size());
    try o.moeCombine(p.y, wts, .{ .ptr = out, .kind = kind }, rows, slots, ex.dims);
    return .{ .ptr = out, .kind = kind };
}

/// The routed slots a row (fp32) and their weights before the sum: a single rank only.
pub const Parts = struct { y: u64, wts: u64, slots: usize };

pub fn parts(o: Ops, kind: Kind, r: Routed, x: Tensor, rows: usize) ops.Error!Parts {
    const slots = r.top_k + 1;
    const size = if (r.exact) rows else bucket(rows);
    const cap = capacity(size * slots, r.rows);
    const logits = try o.arena.of(f32, rows * r.rows);
    const pick = try o.arena.of(i32, size * slots);
    const wts = try o.arena.of(f32, size * slots);
    const items = try o.arena.of(i32, (cap + 1) * 3);
    const members = try o.arena.of(i32, size * slots);
    const p = try expertParts(o, kind, r, x, rows, logits, pick, wts, items, members, cap);
    return .{ .y = p.y, .wts = wts, .slots = slots };
}

/// The shared expert's plan for `rows` rows in `plan` (Routed.shared_plan's layout), written once at load.
pub fn sharedPlanWords(top_k: usize, shared: usize) [8 * 3 + 8]i32 {
    var w: [8 * 3 + 8]i32 = undefined;
    for (0..8) |i| {
        w[i * 3] = @intCast(shared);
        w[i * 3 + 1] = 0;
        w[i * 3 + 2] = @intCast(i + 1);
        w[24 + i] = @intCast(i * (top_k + 1) + top_k);
    }
    return w;
}

/// Select, plan, gate and up, the activation and down: the rows' slots before the combine.
fn expertParts(o: Ops, kind: Kind, r: Routed, x: Tensor, rows: usize, logits: u64, pick: u64, wts: u64, items: u64, members: u64, cap: usize) ops.Error!struct { y: u64 } {
    if (try apartParts(o, kind, r, x, rows, logits, pick, wts, items, members, cap)) |y| return .{ .y = y };
    const ex = r.experts;
    const slots = r.top_k + 1;
    const pairs = rows * slots;
    var count: usize = cap;
    var tile: usize = undefined;
    if (rows == 1) {
        // the router and the pick (and the plan) in one launch where the decode kernels take it
        if (!try o.moeRouterSelect(x, r.rows16, r.done, logits, pick, wts, .{ .items = items, .members = members }, cap, 1, ex.dims, r.rows, r.top_k, 0)) {
            try o.moeRouter(x, r.rows32, logits, rows, ex.dims, r.rows);
            try o.moeSelect(logits, pick, wts, .{ .items = items, .members = members }, cap, 1, r.count(), r.top_k);
        }
        tile = 1;
        count = slots;
    } else {
        tile = tileFor(o, rows);
        // a verify window's 2 to 8 rows: the router, the pick and tf_moe_route's plan in one launch
        const planned = rows <= 8 and try o.moeRouterSelect(x, r.rows16, r.done, logits, pick, wts, .{ .items = items, .members = members }, cap, rows, ex.dims, r.rows, r.top_k, tile);
        if (!planned) {
            if (!try o.moeRouterSelect(x, r.rows16, r.done, logits, pick, wts, null, 0, rows, ex.dims, r.rows, r.top_k, 0)) {
                try o.moeRouterPrompt(x, r.rows32, r.rows16, logits, rows, ex.dims, r.rows);
                try o.moeSelect(logits, pick, wts, null, 0, rows, r.count(), r.top_k);
            }
            try o.moeRoute(pick, pairs, r.rows, tile, members, items, cap);
            // a prompt plan's launches cover its live items only (the slots past them are dead)
            if (tile == BLOCK_ROWS) count = @min(cap, planItems(pairs, r.rows, tile));
        }
    }
    // the gate and up's activation is the launch's epilogue when the tile takes the shape
    const act_t: Tensor = if (try o.projectRoutedAct(x, ex.fused, items, count, members, pairs, slots, @min(tile, rows), ex.limit)) |t| t else blk: {
        const both = try o.projectRouted(x, ex.fused, items, count, members, pairs, slots, @min(tile, rows));
        const act = try o.arena.take(pairs * ex.width * kind.size());
        const t: Tensor = .{ .ptr = act, .kind = kind };
        try o.moeAct(both, t, pairs, ex.width, ex.limit);
        break :blk t;
    };
    const y = try o.projectRouted(act_t, ex.down, items, count, members, pairs, 1, @min(tile, rows));
    return .{ .y = y };
}

test "buffer sizes follow moe.run and max_items" {
    try std.testing.expectEqual(@as(usize, 16), bucket(1));
    try std.testing.expectEqual(@as(usize, 16), bucket(16));
    try std.testing.expectEqual(@as(usize, 32), bucket(17));
    try std.testing.expectEqual(@as(usize, 2048), bucket(2048));
    try std.testing.expectEqual(@as(usize, 129 + 16 * 9 / 8), capacity(16 * 9, 129));
}

test "a prompt plan's live items are at most planItems, equal with every expert at 128 k + 1 pairs (MG-1)" {
    const items = struct {
        fn of(counts: []const usize, tile: usize) usize {
            var n: usize = 0;
            for (counts) |c| n += (c + tile - 1) / tile;
            return n;
        }
    };
    var prng = std.Random.DefaultPrng.init(0x5851F42D4C957F2D);
    const rnd = prng.random();
    var counts: [513]usize = undefined;
    // the coding prompt in 8,192-row chunks and in one chunk, 11 slots a row
    for ([_]usize{ 8192 * 11, 17715 * 11, 20480 * 11 }) |pairs| {
        // the bound's case: 128 k + 1 pairs an expert, the pairs past them on one expert (fewer than a tile)
        const per = (pairs - counts.len) / counts.len / BLOCK_ROWS;
        var used: usize = 0;
        for (&counts) |*c| {
            c.* = BLOCK_ROWS * per + 1;
            used += c.*;
        }
        var spare = pairs - used;
        var e: usize = 0;
        while (spare >= BLOCK_ROWS) : (e += 1) {
            counts[e] += BLOCK_ROWS;
            spare -= BLOCK_ROWS;
        }
        counts[counts.len - 1] += spare;
        try std.testing.expectEqual(planItems(pairs, counts.len, BLOCK_ROWS), items.of(&counts, BLOCK_ROWS));
        // skewed and sparse routings stay under it
        for (0..20) |_| {
            @memset(&counts, 0);
            for (0..pairs) |_| counts[if (rnd.boolean()) rnd.uintLessThan(usize, 8) else rnd.uintLessThan(usize, counts.len)] += 1;
            try std.testing.expect(items.of(&counts, BLOCK_ROWS) <= planItems(pairs, counts.len, BLOCK_ROWS));
        }
        try std.testing.expect(planItems(pairs, counts.len, BLOCK_ROWS) <= capacity(pairs, counts.len));
    }
}

test "the routed products take the decode tile below 64 rows and in a lane round, the GEMM tile from 64 rows" {
    const o: Ops = .{ .l = undefined, .bf16 = true, .stream = undefined, .arena = undefined };
    var round = o;
    round.window = true;
    try std.testing.expectEqual(@as(usize, LANE_ROWS), tileFor(o, 1));
    try std.testing.expectEqual(@as(usize, LANE_ROWS), tileFor(o, BLOCK_FROM - 1));
    try std.testing.expectEqual(@as(usize, BLOCK_ROWS), tileFor(o, BLOCK_FROM));
    try std.testing.expectEqual(@as(usize, BLOCK_ROWS), tileFor(o, 2048));
    try std.testing.expectEqual(@as(usize, LANE_ROWS), tileFor(round, BLOCK_FROM));
    try std.testing.expectEqual(@as(usize, LANE_ROWS), tileFor(round, 2048));
}

/// B-5: a decode window's MoE with the shared expert's gate+up on the side stream (Ops.fork / join), beside the router:
/// its one-item static plan (Routed.shared_plan), the same pair tile at the same rows, into the act buffer the routed
/// items write, so every pair's act rows, the down and the combine are as in expertParts. The routed gate+up leaves the
/// shared item out: at 1 row it is the plan's last item (select_row's k = top_k); at 2 to 8 rows the router's plan puts
/// it in the extra slot `cap` (route_plan's `apart`), and the down takes it there. Null (nothing launched) where the
/// shape keeps expertParts.
fn apartParts(o: Ops, kind: Kind, r: Routed, x: Tensor, rows: usize, logits: u64, pick: u64, wts: u64, items: u64, members: u64, cap: usize) ops.Error!?u64 {
    const ex = r.experts;
    const slots = r.top_k + 1;
    const pairs = rows * slots;
    if (r.shared_plan == 0 or !o.window or o.prefill or !o.fused() or o.side.stream == null or o.side.ring == null) return null;
    if (rows < 1 or rows > LANE_ROWS or kind == .f32 or x.kind == .f32) return null;
    const tile: usize = if (rows == 1) 1 else tileFor(o, rows);
    if (rows > tile or !o.routedActTakes(x, ex.fused, rows)) return null;
    const act = try o.arena.take(pairs * ex.width * kind.size());
    const plan_items = r.shared_plan + (rows - 1) * 3 * 4;
    const plan_members = r.shared_plan + 24 * 4;
    // the side branch starts here (x is written); the router, its pick and plan run on the main stream meanwhile
    try o.fork();
    const planned = try o.moeRouterSelect(x, r.rows16, r.done, logits, pick, wts, .{ .items = items, .members = members, .apart = rows > 1 }, cap, rows, ex.dims, r.rows, r.top_k, if (rows == 1) 0 else tile);
    if (!planned) {
        // nothing was launched: an empty branch, and expertParts takes the call
        try o.join();
        return null;
    }
    // the dry run (the counting stream) counts the side launch as one of its own
    var side = o;
    if (o.stream != counting) side.stream = o.side.stream;
    if (!try side.projectRoutedActInto(x, ex.fused, plan_items, 1, plan_members, slots, rows, ex.limit, act)) return error.BadShape;
    // the routed items: at 1 row the plan's first top_k (the shared expert is its last item), at 2-8 rows slots 0..cap
    const routed = if (rows == 1) slots - 1 else cap;
    if (!try o.projectRoutedActInto(x, ex.fused, items, routed, members, slots, rows, ex.limit, act)) return error.BadShape;
    // the down reads the shared expert's act rows: the branch joins here
    try o.join();
    const all = if (rows == 1) slots else cap + 1;
    return try o.projectRouted(.{ .ptr = act, .kind = kind }, ex.down, items, all, members, pairs, 1, rows);
}
