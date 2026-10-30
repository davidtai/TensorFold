//! GLM-5.3-Flash's DSA layers on HIP (gfx11): latent cache write, absorb and expand, indexer keys, pool scores and the
//! top-512 select (Flash Next's tf_fn_idx_select), dense and sparse latent attention. Kernels: zig/kernels/hip/glm/dsa.hpp
//! in the "glm" group.
//! The CUDA counterparts: cuda_latent.zig, cuda_sparse.zig, cuda_dsa_block.zig.

const std = @import("std");
const runtime = @import("../runtime.zig");
const launch = @import("../launch.zig");
const kernels = @import("../kernels.zig");
const util = @import("util.zig");
const Launcher = @import("../launches.zig").Launcher;
const Stream = @import("../stream.zig").Stream;

const Function = @import("../module.zig").Function;
const Args = launch.Args;
const Dim3 = launch.Dim3;
const Error = runtime.Error;
pub const S = util.S;

/// The shape the kernels compile: latent 512, qk 256, v 256; indexer 32 heads x 128, pools of 4, top 512 pools.
pub const lw = 512;
pub const qk = 256;
pub const dv = 256;
pub const index_heads = 32;
pub const index_dims = 128;
pub const pool = 4;
pub const topk_pools = 512;
/// A row's token list: the 512 pools' tokens, then the incomplete pool's up to 3 (tf_fn_idx_select's key_stride).
pub const width = topk_pools * pool + pool - 1;
/// Key entries one attention part takes (latent.CHUNK), and the 16- and 32-head blocks.
pub const chunk = 512;
pub const heads_tile = 16;
pub const heads_tile_wide = 32;
/// Rows at which a window takes the 32-head block (a prompt chunk); fewer take 16 (more blocks a row). Same bits.
pub const wide_rows = 64;
/// sparse._index_write's LayerNorm epsilon.
pub const index_eps: f32 = 1e-6;

pub const Kernels = struct {
    r: *const runtime.Runtime,
    lat_write: Function,
    absorb_q: Function,
    expand_v: Function,
    index_write: Function,
    pool_keys: Function,
    scores: Function,
    parts: [2]Function, // 16-head, 32-head blocks
    merge: Function,
    /// Flash Next's radix select and token lists (the flashnext group's tf_fn_idx_select).
    select: Function,

    /// The functions in a launcher's glm group (glm/dsa.hpp) and its flashnext group (the select).
    pub fn load(l: *const Launcher) Error!Kernels {
        const m = l.mods[@backingInt(kernels.Group.glm)];
        const fnx = l.mods[@backingInt(kernels.Group.flashnext)];
        return .{
            .r = l.r,
            .lat_write = try m.function("tf_glm_lat_write"),
            .absorb_q = try m.function("tf_glm_absorb_q"),
            .expand_v = try m.function("tf_glm_expand_v"),
            .index_write = try m.function("tf_glm_index_write"),
            .pool_keys = try m.function("tf_glm_pool_keys"),
            .scores = try m.function("tf_glm_scores"),
            .parts = .{ try m.function("tf_glm_latent_parts16"), try m.function("tf_glm_latent_parts32") },
            .merge = try m.function("tf_glm_latent_merge"),
            .select = try fnx.function("tf_fn_idx_select"),
        };
    }

    fn go(k: Kernels, f: Function, grid: Dim3, block: Dim3, s: S, a: *Args) Error!void {
        if (s == util.counting) return;
        try launch.launch(f, .{ .grid = grid, .block = block }, Stream{ .r = k.r, .handle = s }, a);
    }
};

fn dim(x: anytype, y: anytype, z: anytype) Dim3 {
    return .{ .x = @intCast(x), .y = @intCast(y), .z = @intCast(z) };
}

fn cdiv(n: usize, by: usize) usize {
    return (n + by - 1) / by;
}

fn shape(ok: bool) Error!void {
    if (!ok) return error.Invalid;
}

/// latent.latent_write: `rows` rows of `lat` (bf16, stride `lat_stride` elements, a multiple of 8) into the cache
/// (bf16 [capacity][512]) at slots pos[0] + row0 + r (pos an i32 on the device).
pub fn latentWrite(k: Kernels, s: S, lat: u64, lat_stride: usize, cache: u64, pos: u64, row0: usize, rows: usize) Error!void {
    if (rows == 0) return;
    try shape(lat_stride % 8 == 0 and lat % 16 == 0 and cache % 16 == 0);
    var a: Args = .{};
    inline for (.{ lat, @as(i64, @intCast(lat_stride)), cache, pos, @as(c_int, @intCast(row0)) }) |v| try a.add(v);
    try k.go(k.lat_write, dim(rows, 1, 1), dim(64, 1, 1), s, &a);
}

/// kv_b_proj as stored ([heads x 512][512] bf16: per head its 256 key rows, then its 256 value rows): Wk_h at
/// kvKeys(kv_b) + h kv_head, Wv_h at kvValues(kv_b) + h kv_head (elements). cuda_weights.zig:186-192 splits the same rows.
pub const kv_head: usize = (qk + dv) * lw;
pub fn kvKeys(kv_b: u64) u64 {
    return kv_b;
}
pub fn kvValues(kv_b: u64) u64 {
    return kv_b + qk * lw * 2;
}

/// latent.absorb_q: qa [rows][heads][512] bf16 = q_h . Wk_h; q rows `q_stride` values apart (head h at column 256 h),
/// Wk_h [256][512] at wk + h w_head elements.
pub fn absorbQ(k: Kernels, s: S, q: u64, q_stride: usize, wk: u64, w_head: usize, qa: u64, rows: usize, heads: usize) Error!void {
    if (rows == 0) return;
    var a: Args = .{};
    inline for (.{ q, @as(i64, @intCast(q_stride)), wk, @as(i64, @intCast(w_head)), qa, @as(c_int, @intCast(rows)), @as(c_int, @intCast(heads)) }) |v| try a.add(v);
    try k.go(k.absorb_q, dim(heads, lw / 256, cdiv(rows, 16)), dim(256, 1, 1), s, &a);
}

/// latent.expand_v: out [rows][heads][256] bf16 = o_lat [rows][heads][512] . Wv_h^T, Wv_h [256][512] at wv + h w_head.
pub fn expandV(k: Kernels, s: S, ol: u64, wv: u64, w_head: usize, out: u64, rows: usize, heads: usize) Error!void {
    if (rows == 0) return;
    try shape(wv % 16 == 0 and w_head % 8 == 0);
    var a: Args = .{};
    inline for (.{ ol, wv, @as(i64, @intCast(w_head)), out, @as(c_int, @intCast(rows)), @as(c_int, @intCast(heads)) }) |v| try a.add(v);
    try k.go(k.expand_v, dim(heads, dv / 32, cdiv(rows, 8)), dim(256, 1, 1), s, &a);
}

/// sparse.index_update: rows pos[0] + row0 .. of the LayerNormed index keys (k_raw bf16, stride `k_stride`) and the
/// gate rows (fp32 [rows][128]) into ik and ig (bf16, position p's row at ik + p rows: a ring's pointer may be rebased),
/// then the pools those rows complete into pk (bf16 [pools][128]); ape bf16 [4][128].
pub fn indexUpdate(k: Kernels, s: S, k_raw: u64, k_stride: usize, gate: u64, ln_w: u64, ln_b: u64, ape: u64, ik: u64, ig: u64, pk: u64, pos: u64, row0: usize, rows: usize) Error!void {
    if (rows == 0) return;
    var a: Args = .{};
    inline for (.{ k_raw, @as(i64, @intCast(k_stride)), gate, ln_w, ln_b, ik, ig, pos, @as(c_int, @intCast(row0)), index_eps }) |v| try a.add(v);
    try k.go(k.index_write, dim(rows, 1, 1), dim(32, 1, 1), s, &a);
    var b: Args = .{};
    inline for (.{ ik, ig, ape, pk, pos, @as(c_int, @intCast(row0)), @as(c_int, @intCast(rows)) }) |v| try b.add(v);
    try k.go(k.pool_keys, dim(rows / pool + 2, 1, 1), dim(32, 1, 1), s, &b);
}

/// sparse.pool_bucket: the score row width for rows at host_pos .. + rows - 1, a power of two (at least 1024), at most
/// `np_max`.
pub fn poolBucket(host_pos: usize, rows: usize, np_max: usize) usize {
    const visible = (host_pos + rows) / pool + 1;
    return @min(np_max, @max(1024, std.math.ceilPowerOfTwoAssert(usize, visible)));
}

/// sparse.score_columns: the pool columns selection reads for those rows (the last row's complete pools, at least 512).
pub fn scoreColumns(host_pos: usize, rows: usize, np: usize) usize {
    return @min(np, @max(topk_pools, (host_pos + rows) / pool));
}

/// select_tokens' 1/sqrt(heads) head-weight scale, as the Python family writes it for 32 heads.
pub const head_scale: f32 = @floatCast(1.0 / 5.656854249492381);

/// One select's scratch for `rows` rows: scores fp32 [rows][np]; complete and ends i32 [rows].
pub const Scratch = struct { scores: u64, complete: u64, ends: u64 };

/// The bytes Scratch takes for `rows` rows of `np` pool columns.
pub fn scratchBytes(rows: usize, np: usize) usize {
    return rows * np * 4 + 2 * rows * 4;
}

/// sparse.select_tokens for rows row0 .. row0 + rows - 1 of a window at pos (host_pos on the host): the pool scores,
/// then each row's list (tokens i32 [rows][width], counts and sparse i32 [rows]). A row with more than 512 complete
/// pools: sparse 1, its 512 best pools' tokens ascending (ties to the lower pool) then the incomplete pool's tail,
/// count 4 512 + tail. Else sparse 0 and count = position + 1 (attend densely). qi bf16 rows of 32 x 128, `qi_stride`
/// values apart; wts the head weights (bf16, row stride `w_stride`); pk the pooled keys. `np` = poolBucket(...) is the
/// scores' row stride.
pub fn selectTokens(k: Kernels, s: S, qi: u64, qi_stride: usize, wts: u64, w_stride: usize, pk: u64, pos: u64, row0: usize, host_pos: usize, rows: usize, np: usize, sc: Scratch, tokens: u64, counts: u64, sparse: u64) Error!void {
    if (rows == 0) return;
    const cols = scoreColumns(host_pos + row0, rows, np);
    var a: Args = .{};
    const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, index_dims)));
    inline for (.{ qi, @as(i64, @intCast(qi_stride)), wts, @as(i64, @intCast(w_stride)), pk, sc.scores, @as(i64, @intCast(np)), sc.complete, sc.ends, pos, @as(c_int, @intCast(row0)), @as(c_int, @intCast(cols)), scale, head_scale, @as(c_int, topk_pools) }) |v| try a.add(v);
    try k.go(k.scores, dim(rows, cdiv(cols, 64), 1), dim(128, 1, 1), s, &a);
    try selectLists(k, s, sc, np, rows, tokens, counts, sparse);
}

/// The select alone (tf_fn_idx_select) over scores, complete and ends already in `sc`: each row's list as selectTokens.
pub fn selectLists(k: Kernels, s: S, sc: Scratch, np: usize, rows: usize, tokens: u64, counts: u64, sparse: u64) Error!void {
    if (rows == 0) return;
    var b: Args = .{};
    inline for (.{ sc.scores, @as(i64, @intCast(np)), sc.complete, sc.ends, tokens, @as(i64, width), counts, sparse, @as(c_int, topk_pools), @as(c_int, pool) }) |v| try b.add(v);
    try k.go(k.select, dim(rows, 1, 1), dim(1024, 1, 1), s, &b);
}

/// The parts of one attention call: po fp32 [rows][heads][parts][512], pms fp32 [rows][heads][parts][2] (max, sum).
pub const Partials = struct { po: u64, pms: u64 };

pub fn partialBytes(rows: usize, heads: usize, parts: usize) usize {
    return rows * heads * parts * (lw + 2) * 4;
}

/// latent.chunks_for: parts of `chunk` key entries to cover `length` keys.
pub fn partsFor(length: usize) usize {
    return cdiv(length, chunk);
}

/// The keys a row of an attention call attends: `dense` (positions 0 .. its own, pos[0] + row0 + r) or `listed`
/// (tf_fn_idx_select's lists: ids i32 [rows][id_stride], counts and sparse i32 [rows]).
pub const Keys = union(enum) {
    dense,
    listed: struct { ids: u64, id_stride: usize, counts: u64, sparse: u64 },
};

/// latent.attention / sparse_attention: rows row0 .. row0 + rows - 1 of a window at pos, qa bf16 [rows][heads][512] over
/// the latent cache (bf16 [capacity][512]) into out bf16 [rows][heads][512], in `parts` parts of 512 key entries
/// (enough for the longest row: partsFor(host_pos + row0 + rows) dense, partsFor(width) listed). heads a multiple of 32.
pub fn attention(k: Kernels, s: S, qa: u64, cache: u64, pos: u64, row0: usize, keys: Keys, p: Partials, out: u64, rows: usize, heads: usize, parts: usize, scale: f32) Error!void {
    if (rows == 0) return;
    try shape(heads % heads_tile_wide == 0 and parts > 0 and qa % 16 == 0 and cache % 16 == 0);
    const hb: usize = if (rows >= wide_rows) heads_tile_wide else heads_tile;
    return attentionTile(k, s, qa, cache, pos, row0, keys, p, out, rows, heads, parts, scale, hb);
}

/// attention with the head block `hb` (16 or 32) given: both give the same bits.
pub fn attentionTile(k: Kernels, s: S, qa: u64, cache: u64, pos: u64, row0: usize, keys: Keys, p: Partials, out: u64, rows: usize, heads: usize, parts: usize, scale: f32, hb: usize) Error!void {
    if (rows == 0) return;
    try shape((hb == heads_tile or hb == heads_tile_wide) and heads % hb == 0);
    var ids: u64 = 0;
    var stride: usize = 0;
    var counts: u64 = 0;
    var sparse: u64 = 0;
    switch (keys) {
        .dense => {},
        .listed => |l| {
            ids = l.ids;
            stride = l.id_stride;
            counts = l.counts;
            sparse = l.sparse;
        },
    }
    var a: Args = .{};
    inline for (.{ qa, cache, pos, @as(c_int, @intCast(row0)), ids, @as(i64, @intCast(stride)), counts, sparse, p.po, p.pms, @as(c_int, @intCast(heads)), @as(c_int, chunk), scale }) |v| try a.add(v);
    try k.go(k.parts[if (hb == heads_tile) 0 else 1], dim(rows, heads / hb, parts), dim(256, 1, 1), s, &a);
    var b: Args = .{};
    inline for (.{ p.po, p.pms, out, @as(c_int, @intCast(heads)), @as(c_int, @intCast(parts)) }) |v| try b.add(v);
    try k.go(k.merge, dim(heads, rows, 1), dim(128, 1, 1), s, &b);
}

test "the sizing rules of sparse.py and latent.py" {
    try std.testing.expectEqual(@as(usize, 1024), poolBucket(100, 1, 1 << 20));
    try std.testing.expectEqual(@as(usize, 2048), poolBucket(5000, 1, 1 << 20));
    try std.testing.expectEqual(@as(usize, 512), scoreColumns(100, 1, 1024));
    try std.testing.expectEqual(@as(usize, 1250), scoreColumns(5000, 1, 2048));
    try std.testing.expectEqual(@as(usize, 2051), width);
    try std.testing.expectEqual(@as(usize, 5), partsFor(width));
    try std.testing.expectEqual(@as(usize, 1), partsFor(512));
}
