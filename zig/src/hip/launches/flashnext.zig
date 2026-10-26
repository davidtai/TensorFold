//! Flash Next's launches: stream norms, gates and mixes, the n-gram gate, conv and hashes, indexer, sparse attention.

const Launcher = @import("../launches.zig").Launcher;
const Module = @import("../module.zig").Module;
const Function = @import("../module.zig").Function;
const util = @import("util.zig");
const S = util.S;
const P = util.P;
const C = util.C;
const F = util.F;
const CF = util.CF;
const I = util.I;
const CI = util.CI;
const Args = util.Args;
const Error = util.Error;
const ad = util.ad;
const dim = util.dim;
const cdiv = util.cdiv;
const caps = @import("../caps.zig");
const kernels = @import("../kernels.zig");

pub const Fns = struct {
    hc_norm: Function,
    hc_gates: Function,
    rms_wide: Function,
    hc_mix: Function,
    hc_up_mix: Function,
    ple_gate: Function,
    ple_conv: Function,
    ngram_ids: Function,
    idx_pool: Function,
    idx_scores: Function,
    idx_scores_quad: Function,
    idx_select: Function,
    sparse_parts: [2]Function, // groups of up to 6 and up to 12 query heads
    sparse_merge: Function,
    sparse_prompt: [2]Function, // q bf16-exact, q any fp32 (hi + lo)
    sparse_tile_parts: [2]Function, // the same tile over parts of each row's keys
    sparse_merge_gate: Function,

    pub fn load(m: Module) Error!Fns {
        return .{
            .hc_norm = try m.function("tf_fn_hc_norm"),
            .hc_gates = try m.function("tf_fn_hc_gates"),
            .rms_wide = try m.function("tf_fn_rms_wide"),
            .hc_mix = try m.function("tf_fn_hc_mix"),
            .hc_up_mix = try m.function("tf_fn_hc_up_mix"),
            .ple_gate = try m.function("tf_fn_ple_gate"),
            .ple_conv = try m.function("tf_fn_ple_conv"),
            .ngram_ids = try m.function("tf_fn_ngram_ids"),
            .idx_pool = try m.function("tf_fn_idx_pool"),
            .idx_scores = try m.function("tf_fn_idx_scores"),
            .idx_scores_quad = try m.function("tf_fn_idx_scores_quad"),
            .idx_select = try m.function("tf_fn_idx_select"),
            .sparse_parts = .{ try m.function("tf_fn_sparse_parts6"), try m.function("tf_fn_sparse_parts12") },
            .sparse_merge = try m.function("tf_fn_sparse_merge"),
            .sparse_prompt = .{ try m.function("tf_fn_sparse_prompt"), try m.function("tf_fn_sparse_prompt_split") },
            .sparse_tile_parts = .{ try m.function("tf_fn_sparse_tile_parts"), try m.function("tf_fn_sparse_tile_parts_split") },
            .sparse_merge_gate = try m.function("tf_fn_sparse_merge_gate"),
        };
    }
};

/// The widest stream the norm and gate kernels walk: 16 values a thread of 256.
pub const most_dims = 4096;
const threads = 256;

/// `rows` rows of `streams` streams: h += round(branch * inject) when a branch is given, then x = h rinv scale.
pub fn hcNorm(
    l: *const Launcher,
    h: C,
    branch: C,
    inject: C,
    h_out: P,
    x_out: P,
    scale: CF,
    kind: c_int,
    rows: c_int,
    streams: c_int,
    dims: c_int,
    eps: f32,
    s: S,
) Error!void {
    return hcNormShares(l, h, branch, inject, h_out, x_out, scale, kind, rows, streams, dims, eps, 0, 0, s);
}

/// hcNorm with the branch as `world` fp32 shares `stride` values apart (a tensor-parallel gather), summed in order.
pub fn hcNormShares(
    l: *const Launcher,
    h: C,
    branch: C,
    inject: C,
    h_out: P,
    x_out: P,
    scale: CF,
    kind: c_int,
    rows: c_int,
    streams: c_int,
    dims: c_int,
    eps: f32,
    world: c_int,
    stride: i64,
    s: S,
) Error!void {
    if (dims > most_dims) return util.invalid("hcNorm");
    var a: Args = .{};
    inline for (.{
        ad(h),
        ad(branch),
        ad(inject),
        ad(h_out),
        ad(x_out),
        ad(scale),
        kind,
        streams,
        dims,
        eps,
        world,
        stride,
    }) |v| try a.add(v);
    try l.go(l.fnx.hc_norm, dim(streams, rows, 1), dim(1024, 1, 1), 0, s, &a);
}

/// act = silu(z / streams) and inject = 2 sigmoid(z / streams) from the down projection's fp32 sums (no inject: null).
pub fn hcGates(
    l: *const Launcher,
    z: CF,
    act: P,
    inject: P,
    kind: c_int,
    rows: c_int,
    low: c_int,
    streams: c_int,
    s: S,
) Error!void {
    var a: Args = .{};
    inline for (.{ ad(z), ad(act), ad(inject), kind, rows, low, streams }) |v| try a.add(v);
    try l.flat(l.fnx.hc_gates, @as(i64, rows) * (low + if (inject != null) streams else 0), s, &a);
}

/// Each of `rows` rows normed over its whole `width` (any width) times the fp32 `scale`.
pub fn rmsWide(
    l: *const Launcher,
    x: C,
    scale: CF,
    y: P,
    kind: c_int,
    rows: c_int,
    width: c_int,
    eps: f32,
    s: S,
) Error!void {
    if (rows == 0) return;
    var a: Args = .{};
    inline for (.{ ad(x), ad(scale), ad(y), kind, width, eps }) |v| try a.add(v);
    try l.go(l.fnx.rms_wide, dim(rows, 1, 1), dim(threads, 1, 1), 0, s, &a);
}

/// The block input: the mean over the streams of sigmoid(g) times the normed stream.
pub fn hcMix(
    l: *const Launcher,
    g: C,
    gkind: c_int,
    x: C,
    mixed: P,
    kind: c_int,
    rows: c_int,
    streams: c_int,
    dims: c_int,
    s: S,
) Error!void {
    var a: Args = .{};
    inline for (.{ ad(g), gkind, ad(x), ad(mixed), kind, rows, streams, dims }) |v| try a.add(v);
    try l.flat(l.fnx.hc_mix, @as(i64, rows) * dims, s, &a);
}

/// The shape tf_fn_hc_up_mix is compiled for: 4 streams of 2,560, low rank 320, 4-bit groups of 32, bf16.
pub const hc_up = struct {
    pub const streams = 4;
    pub const hidden = 2560;
    pub const low = 320;
    pub const rows = 4; // rows a block
    pub const cols = 32; // hidden columns a block
    /// Rows the launch takes at most: decode and verify windows (a prompt's rows take the products).
    pub const most_rows = 16;
};

/// The hyper-connection read's gates, up projection and mix in one launch: z [rows, n] the down product's fp32 sums
/// (n = 324 with the inject gates, which block 0 writes, else 320), x the normed streams, mixed [rows, 2560].
pub fn hcUpMix(l: *const Launcher, z: CF, n: c_int, words: C, scales: C, biases: C, x: C, mixed: P, inject: P, rows: c_int, s: S) Error!void {
    if (rows <= 0) return;
    if (rows > hc_up.most_rows or (n != hc_up.low and n != hc_up.low + hc_up.streams)) return util.invalid("hcUpMix");
    var a: Args = .{};
    inline for (.{ ad(z), n, ad(words), ad(scales), ad(biases), ad(x), ad(mixed), ad(inject), rows }) |v| try a.add(v);
    try l.go(l.fnx.hc_up_mix, dim(hc_up.hidden / hc_up.cols, cdiv(rows, hc_up.rows), 1), dim(threads, 1, 1), 0, s, &a);
}

/// The n-gram gate of every stream; `normed` is the conv's input rows.
pub fn pleGate(
    l: *const Launcher,
    kv: C,
    h: C,
    key_scale: CF,
    query_scale: CF,
    conv_scale: CF,
    gated: P,
    normed: P,
    kind: c_int,
    rows: c_int,
    streams: c_int,
    dims: c_int,
    eps: f32,
    s: S,
) Error!void {
    if (dims > most_dims) return util.invalid("pleGate");
    var a: Args = .{};
    inline for (.{
        ad(kv),
        ad(h),
        ad(key_scale),
        ad(query_scale),
        ad(conv_scale),
        ad(gated),
        ad(normed),
        kind,
        streams,
        dims,
        eps,
    }) |v| try a.add(v);
    try l.go(l.fnx.ple_gate, dim(streams, rows, 1), dim(threads, 1, 1), 0, s, &a);
}

/// out = h + (gated + silu(conv)): the dilated causal conv over `cin`, its tail rows first.
pub fn pleConv(
    l: *const Launcher,
    cin: C,
    weight: CF,
    gated: C,
    h: C,
    out: P,
    kind: c_int,
    rows: c_int,
    width: c_int,
    taps: c_int,
    dilation: c_int,
    s: S,
) Error!void {
    var a: Args = .{};
    inline for (.{
        ad(cin),
        ad(weight),
        ad(gated),
        ad(h),
        ad(out),
        kind,
        rows,
        width,
        taps,
        dilation,
    }) |v| try a.add(v);
    try l.flat(l.fnx.ple_conv, @as(i64, rows) * width, s, &a);
}

/// Each row's n-gram table rows; `meta` holds the history, eos, multipliers, sizes and offsets (int64).
pub fn ngramIds(
    l: *const Launcher,
    tokens: CI,
    meta: ?*const anyopaque,
    ids: P,
    rows: c_int,
    heads: c_int,
    per: c_int,
    n: c_int,
    s: S,
) Error!void {
    var a: Args = .{};
    inline for (.{ ad(tokens), ad(meta), ad(ids), rows, heads, per, n }) |v| try a.add(v);
    try l.flat(l.fnx.ngram_ids, @as(i64, rows) * heads, s, &a);
}

/// Blocks first .. first + blocks of the raw indexer keys, pooled, normed and rotated into `pooled`.
pub fn idxPool(
    l: *const Launcher,
    raw: C,
    pooled: P,
    scale: CF,
    kind: c_int,
    first: c_int,
    blocks: c_int,
    dims: c_int,
    ratio: c_int,
    rotary: c_int,
    theta: f32,
    eps: f32,
    s: S,
) Error!void {
    if (blocks == 0) return;
    if (dims > threads) return util.invalid("idxPool");
    var a: Args = .{};
    inline for (.{
        ad(raw),
        ad(pooled),
        ad(scale),
        kind,
        first,
        @as(u64, 0),
        dims,
        ratio,
        rotary,
        theta,
        eps,
    }) |v| try a.add(v);
    try l.go(l.fnx.idx_pool, dim(blocks, 1, 1), dim(threads, 1, 1), 0, s, &a);
}

/// The blocks `rows` rows at device positions `pos` complete, pooled into their slots of `pooled` (all blocks').
pub fn idxPoolAt(
    l: *const Launcher,
    raw: C,
    pooled: P,
    scale: CF,
    kind: c_int,
    pos: CI,
    rows: c_int,
    dims: c_int,
    ratio: c_int,
    rotary: c_int,
    theta: f32,
    eps: f32,
    s: S,
) Error!void {
    if (rows == 0) return;
    if (dims > threads) return util.invalid("idxPoolAt");
    var a: Args = .{};
    inline for (.{
        ad(raw),
        ad(pooled),
        ad(scale),
        kind,
        @as(c_int, 0),
        ad(pos),
        dims,
        ratio,
        rotary,
        theta,
        eps,
    }) |v| try a.add(v);
    try l.go(l.fnx.idx_pool, dim(rows, 1, 1), dim(threads, 1, 1), 0, s, &a);
}

/// Every row's score of each of `blocks` pooled blocks it has complete, for rows past `top` complete blocks.
/// Rows from which the block scores take tf_fn_idx_scores_quad (fewer: a verify window's, a wave a block).
pub const quad_rows = 16;

pub fn idxScores(
    l: *const Launcher,
    q: CF,
    pooled: C,
    complete: CI,
    scores: F,
    kind: c_int,
    rows: c_int,
    heads: c_int,
    dims: c_int,
    blocks: c_int,
    stride: i64,
    top: c_int,
    s: S,
) Error!void {
    if (blocks == 0) return;
    if (dims > threads) return util.invalid("idxScores");
    // a prompt slice of Flash Next's 4 x 128 index heads: 4 lanes a score, the same sums (indexer.hpp)
    if (rows >= quad_rows and heads == 4 and dims == 128) {
        var b: Args = .{};
        inline for (.{ ad(q), ad(pooled), ad(complete), ad(scores), kind, rows, blocks, stride, top }) |v| try b.add(v);
        return l.go(l.fnx.idx_scores_quad, dim(cdiv(blocks, 64), cdiv(rows, 32), 1), dim(128, 1, 1), 0, s, &b);
    }
    var a: Args = .{};
    inline for (.{
        ad(q),
        ad(pooled),
        ad(complete),
        ad(scores),
        kind,
        rows,
        heads,
        dims,
        blocks,
        stride,
        top,
    }) |v| try a.add(v);
    try l.go(l.fnx.idx_scores, dim(cdiv(blocks, 8), cdiv(rows, 8), 1), dim(threads, 1, 1), 0, s, &a);
}

/// Each row's keys: its `top` best blocks and its tail past `top` complete blocks, else its first ends[r] keys.
pub fn idxSelect(
    l: *const Launcher,
    scores: CF,
    stride: i64,
    complete: CI,
    ends: CI,
    keys: I,
    key_stride: i64,
    count: I,
    sparse: I,
    rows: c_int,
    top: c_int,
    ratio: c_int,
    s: S,
) Error!void {
    var a: Args = .{};
    inline for (.{
        ad(scores),
        stride,
        ad(complete),
        ad(ends),
        ad(keys),
        key_stride,
        ad(count),
        ad(sparse),
        top,
        ratio,
    }) |v| try a.add(v);
    try l.go(l.fnx.idx_select, dim(rows, 1, 1), dim(1024, 1, 1), 0, s, &a);
}

/// Attention of each row's heads over its listed keys (or its first count[r] keys) in `parts` parts, then merged.
pub fn sparseAttention(
    l: *const Launcher,
    q: CF,
    k: C,
    v: C,
    cache_kind: c_int,
    ids: CI,
    id_stride: i64,
    count: CI,
    sparse: CI,
    parts_o: F,
    parts_ms: F,
    out: F,
    rows: c_int,
    heads: c_int,
    kv_heads: c_int,
    dims: c_int,
    cap: i64,
    scale: f32,
    parts: c_int,
    s: S,
) Error!void {
    if (dims > 256 or @rem(heads, kv_heads) != 0 or @divExact(heads, kv_heads) > 12)
        return util.invalid("sparseAttention");
    var a: Args = .{};
    inline for (.{
        ad(q),
        ad(k),
        ad(v),
        cache_kind,
        ad(ids),
        id_stride,
        ad(count),
        ad(sparse),
        ad(parts_o),
        ad(parts_ms),
        heads,
        kv_heads,
        dims,
        cap,
        scale,
        parts,
    }) |x| try a.add(x);
    const wide: usize = if (@divExact(heads, kv_heads) <= 6) 0 else 1;
    try l.go(l.fnx.sparse_parts[wide], dim(kv_heads, rows, parts), dim(256, 1, 1), 0, s, &a);
    var b: Args = .{};
    inline for (.{ ad(parts_o), ad(parts_ms), ad(out), heads, dims, parts }) |x| try b.add(x);
    try l.go(l.fnx.sparse_merge, dim(heads, rows, 1), dim(256, 1, 1), 0, s, &b);
}

/// The prompt attention's geometry: Qwen3.8 Flash Next's 24 heads of 256 over 2 kv heads, bf16 caches.
pub const prompt_heads = 24;
pub const prompt_kv_heads = 2;
pub const prompt_dims = 256;

/// Whether this build's kernels run the prompt attention on matrix cores (gfx11 WMMA, bf16 activations).
pub fn promptOnMatrix() bool {
    const c = caps.Caps.of(kernels.arch) orelse return false;
    return c.matrix == .wmma11 and c.act == .bf16;
}

/// A prompt's attention of each row's 12 heads a kv head over its listed keys (or its first count[r] keys) in one pass
/// on the matrix cores; with `qg` the gated bf16 rows (tf_attn_gate's), else fp32 (rows, 24, 256). `exact_q`: q holds
/// bf16 values and `scale` is a power of two (one product), else q is carried as bf16 hi + lo.
pub fn sparsePrompt(
    l: *const Launcher,
    q: CF,
    k: C,
    v: C,
    ids: CI,
    id_stride: i64,
    count: CI,
    sparse: CI,
    qg: C,
    out: P,
    rows: c_int,
    cap: i64,
    scale: f32,
    exact_q: bool,
    s: S,
) Error!void {
    if (!promptOnMatrix()) return util.invalid("sparsePrompt");
    if (rows == 0) return;
    var a: Args = .{};
    inline for (.{ ad(q), ad(k), ad(v), ad(ids), id_stride, ad(count), ad(sparse), ad(qg), ad(out), cap, scale }) |x|
        try a.add(x);
    const f = l.fnx.sparse_prompt[if (exact_q) 0 else 1];
    try l.go(f, dim(rows, prompt_kv_heads, 1), dim(128, 1, 1), 0, s, &a);
}

/// A few rows' attention (decode, a window): sparsePrompt's tile over `parts` parts of each row's keys, then the parts
/// merged in part order and gated as tf_attn_gate gates (`kind` rows of `qg` into `out`).
pub fn sparseDecode(
    l: *const Launcher,
    q: CF,
    k: C,
    v: C,
    ids: CI,
    id_stride: i64,
    count: CI,
    sparse: CI,
    parts_o: F,
    parts_ms: F,
    qg: C,
    out: P,
    kind: c_int,
    rows: c_int,
    cap: i64,
    scale: f32,
    parts: c_int,
    exact_q: bool,
    s: S,
) Error!void {
    if (!promptOnMatrix() or parts < 1) return util.invalid("sparseDecode");
    if (rows == 0) return;
    var a: Args = .{};
    inline for (.{ ad(q), ad(k), ad(v), ad(ids), id_stride, ad(count), ad(sparse), ad(parts_o), ad(parts_ms), cap, scale }) |x|
        try a.add(x);
    try l.go(l.fnx.sparse_tile_parts[if (exact_q) 0 else 1], dim(rows, prompt_kv_heads, parts), dim(128, 1, 1), 0, s, &a);
    var b: Args = .{};
    inline for (.{ ad(parts_o), ad(parts_ms), ad(qg), kind, ad(out), @as(c_int, prompt_heads), @as(c_int, prompt_dims), parts }) |x|
        try b.add(x);
    try l.go(l.fnx.sparse_merge_gate, dim(prompt_heads, rows, 1), dim(256, 1, 1), 0, s, &b);
}
