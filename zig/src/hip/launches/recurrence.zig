//! DeltaNet launches: gate, recurrence and the gated norm.

const Launcher = @import("../launches.zig").Launcher;
const Module = @import("../module.zig").Module;
const Function = @import("../module.zig").Function;
const util = @import("util.zig");
const S = util.S;
const P = util.P;
const C = util.C;
const F = util.F;
const CF = util.CF;
const Args = util.Args;
const Error = util.Error;
const ad = util.ad;
const dim = util.dim;
const cdiv = util.cdiv;
const tri = util.tri;
const invalid = util.invalid;

pub fn tf_gdn_gate(l: *const Launcher, av: C, bv: C, kind: c_int, a_log: CF, dt_bias: CF, gate: F, beta: F, count: c_int, heads: c_int, s: S) Error!void {
    var a: Args = .{};
    try a.add(ad(av));
    try a.add(ad(bv));
    try a.add(ad(a_log));
    try a.add(ad(dt_bias));
    try a.add(ad(gate));
    try a.add(ad(beta));
    try a.add(count);
    try a.add(heads);
    try l.go(l.gdn_gate[tri(kind)], dim(cdiv(count, 256), 1, 1), dim(256, 1, 1), 0, s, &a);
}

/// Rows from which a prompt's recurrence takes tf_gdn_prompt_128 (fewer: a wave a row reads the state faster).
pub const prompt_rows = 64;

/// `from`: a device i32 naming the snapshot row (of `states`) a lone stream's window starts from, -1 for `state`; 0: none.
pub fn tf_gated_delta(l: *const Launcher, q: CF, k: CF, v: CF, gate: CF, beta: CF, state: F, y: F, batch: c_int, length: c_int, key_heads: c_int, value_heads: c_int, dk: c_int, dv: c_int, s: S, states: F, from: u64) Error!void {
    if ((dk != 16 and dk != 128) or dv < 1 or key_heads < 1 or @rem(value_heads, key_heads) != 0 or batch < 1 or length < 1) {
        return invalid("gated delta");
    }
    var a: Args = .{};
    try a.add(ad(q));
    try a.add(ad(k));
    try a.add(ad(v));
    try a.add(ad(gate));
    try a.add(ad(beta));
    try a.add(ad(state));
    try a.add(ad(y));
    try a.add(length);
    try a.add(key_heads);
    try a.add(value_heads);
    try a.add(dv);
    try a.add(ad(states));
    if (from != 0 and batch != 1) return invalid("gated delta start row");
    try a.add(from);
    // a prompt's rows (no snapshots): half a value head a block, 4 lanes a row, the same sums (gated_delta.hip)
    if (dk == 128 and dv == 128 and states == null and length >= prompt_rows) {
        var b: Args = .{};
        try b.add(ad(q));
        try b.add(ad(k));
        try b.add(ad(v));
        try b.add(ad(gate));
        try b.add(ad(beta));
        try b.add(ad(state));
        try b.add(ad(y));
        try b.add(length);
        try b.add(key_heads);
        try b.add(value_heads);
        return l.go(l.gdn_prompt, dim(2, value_heads, batch), dim(32, 4, 1), 0, s, &b);
    }
    const wide: usize = if (dk == 128) 0 else 1;
    // Eight rows share one k/q tile; a short dv stays on the one-row wave, which is the same scan.
    if (@rem(dv, 8) == 0) {
        try l.go(l.gated_delta_tile[wide], dim(@divExact(dv, 8), value_heads, batch), dim(32, 8, 1), 0, s, &a);
    } else {
        try l.go(l.gated_delta_wave[wide], dim(dv, value_heads, batch), dim(32, 1, 1), 0, s, &a);
    }
}

/// A verify window's recurrence that holds the state (no snapshot, no store): the rows' outputs only. The keep that
/// follows (tf_gdn_keep) runs the state through the kept rows. dv a multiple of 8.
pub fn tf_gdn_hold(l: *const Launcher, q: CF, k: CF, v: CF, gate: CF, beta: CF, state: F, y: F, length: c_int, key_heads: c_int, value_heads: c_int, dk: c_int, dv: c_int, s: S) Error!void {
    if ((dk != 16 and dk != 128) or @rem(dv, 8) != 0 or key_heads < 1 or @rem(value_heads, key_heads) != 0 or length < 1 or length > 16) {
        return invalid("gated delta hold");
    }
    var a: Args = .{};
    try a.add(ad(q));
    try a.add(ad(k));
    try a.add(ad(v));
    try a.add(ad(gate));
    try a.add(ad(beta));
    try a.add(ad(state));
    try a.add(ad(y));
    try a.add(length);
    try a.add(key_heads);
    try a.add(value_heads);
    try a.add(dv);
    try l.go(l.gdn_hold[if (dk == 128) 0 else 1], dim(@divExact(dv, 8), value_heads, 1), dim(32, 8, 1), 0, s, &a);
}

/// A keep: `layers` DeltaNet states (layer z at state + z * state_stride floats) each run through the first `kept`
/// rows of its window's inputs (k, v, gate, beta of `rows` rows at in + z * in_stride floats), by the window's scan.
pub fn tf_gdn_keep(l: *const Launcher, state: F, state_stride: i64, in: CF, in_stride: i64, rows: c_int, kept: c_int, layers: c_int, key_heads: c_int, value_heads: c_int, dk: c_int, dv: c_int, s: S) Error!void {
    if ((dk != 16 and dk != 128) or @rem(dv, 8) != 0 or kept < 1 or kept > rows or layers < 1) return invalid("gated delta keep");
    var a: Args = .{};
    try a.add(ad(state));
    try a.add(state_stride);
    try a.add(ad(in));
    try a.add(in_stride);
    try a.add(rows);
    try a.add(kept);
    try a.add(key_heads);
    try a.add(value_heads);
    try a.add(dv);
    try l.go(l.gdn_keep[if (dk == 128) 0 else 1], dim(@divExact(dv, 8), value_heads, layers), dim(32, 8, 1), 0, s, &a);
}

/// GLM-5.3-Flash's KDA (gated_delta.hip tf_kda_*_128 on the DeltaNet bodies; recurrence/kda.hpp's ports of the CUDA
/// prep, out and conv shift). Buffers as the CUDA path's (cuda_state.zig, cuda_kda_block.zig): heads x 128 channels,
/// key heads = value heads; decays fp32 (rows, heads, 128); saves k fp32, v bf16, decays fp32, beta fp32 (rows, heads).
pub const KdaFns = struct {
    prep: Function,
    tile: Function,
    hold: Function,
    keep: Function,
    prompt: Function,
    out: Function,
    conv_shift: Function,
    prep_wave: Function,
    out_wave: Function,
    /// The prompt scan staging 8 steps (K-5's other width), and the scans as before K-5 (16 steps): the tests' oracles.
    prompt_t8: Function,
    prompt_t16: Function,
    tile_t16: Function,
    hold_t16: Function,

    pub fn load(m: Module) Error!KdaFns {
        return .{
            .prep = try m.function("tf_kda_prep"),
            .tile = try m.function("tf_kda_tile_128"),
            .hold = try m.function("tf_kda_hold_128"),
            .keep = try m.function("tf_kda_keep_128"),
            .prompt = try m.function("tf_kda_prompt_128"),
            .out = try m.function("tf_kda_out"),
            .conv_shift = try m.function("tf_kda_conv_shift"),
            .prep_wave = try m.function("tf_kda_prep_wave"),
            .out_wave = try m.function("tf_kda_out_wave"),
            .prompt_t8 = try m.function("tf_kda_prompt_128_t8"),
            .prompt_t16 = try m.function("tf_kda_prompt_128_t16"),
            .tile_t16 = try m.function("tf_kda_tile_128_t16"),
            .hold_t16 = try m.function("tf_kda_hold_128_t16"),
        };
    }
};

/// KDA's key and value channels a head.
pub const kda_dim = 128;

/// The two launch forms of tf_kda_prep and tf_kda_out (kda.hpp). They give every row the same bytes (LANES.md code
/// contract 7; kernel_tests/kda.zig R-1): `block` a block a (row, head), CUDA's shape; `wave` the DeltaNet's shapes, a
/// wave a (16-row segment, head, part) for prep and a wave a (row, head) for out.
pub const KdaForm = enum { block, wave };

/// Rows from which tf_kda_prep and tf_kda_out take the wave form (fewer: the block form's more blocks).
pub const kda_wave_rows = 16;

fn aligned(addr: anytype, to: usize) bool {
    return ad(addr) % to == 0;
}

/// kda.chain's prep (glm_kda.cu tf_glm_kda_prep): `rows` rows of the stacked bf16 projection p (row stride p_stride,
/// q | k | v from column 0, beta's logits at b_off), the bf16 decay rows a (stride a_stride) and the bf16 conv window cs
/// [3][3 heads 128] (read, not written); taps cw bf16 [3 heads 128][4] as stored. Out: q fp32, k fp32, v bf16, decays
/// fp32 and beta fp32, row-major (rows, heads, 128) and (rows, heads). The wave form from kda_wave_rows rows where the
/// buffers suit its vector loads; the block form otherwise: the same bytes.
pub fn tf_kda_prep(l: *const Launcher, p: C, p_stride: c_int, b_off: c_int, a: C, a_stride: c_int, cs: C, cw: C, a_log: CF, dt_bias: CF, lower: f32, q: F, k: F, v: P, g: F, b: F, rows: c_int, heads: c_int, s: S) Error!void {
    const vec = @rem(p_stride, 4) == 0 and @rem(a_stride, 4) == 0 and aligned(p, 8) and aligned(a, 8) and aligned(cs, 8) and
        aligned(cw, 16) and aligned(q, 16) and aligned(k, 16) and aligned(v, 8) and aligned(g, 16);
    const form: KdaForm = if (rows >= kda_wave_rows and vec) .wave else .block;
    return tf_kda_prep_as(l, form, p, p_stride, b_off, a, a_stride, cs, cw, a_log, dt_bias, lower, q, k, v, g, b, rows, heads, s);
}

/// tf_kda_prep in a named form (the tests', and the timings').
pub fn tf_kda_prep_as(l: *const Launcher, form: KdaForm, p: C, p_stride: c_int, b_off: c_int, a: C, a_stride: c_int, cs: C, cw: C, a_log: CF, dt_bias: CF, lower: f32, q: F, k: F, v: P, g: F, b: F, rows: c_int, heads: c_int, s: S) Error!void {
    if (heads < 1 or rows < 1 or p_stride < 3 * heads * kda_dim or b_off + heads > p_stride) return invalid("kda prep");
    var x: Args = .{};
    inline for (.{ heads, ad(p), p_stride, b_off, ad(a), a_stride, ad(cs), ad(cw), ad(a_log), ad(dt_bias), lower, ad(q), ad(k), ad(v), ad(g), ad(b) }) |e| try x.add(e);
    if (form == .block) return l.go(l.kda.prep, dim(rows, heads, 1), dim(512, 1, 1), 0, s, &x);
    try x.add(rows);
    const waves = cdiv(rows, 16) * @as(usize, @intCast(heads)) * 4;
    try l.go(l.kda.prep_wave, dim(cdiv(waves, 8), 1, 1), dim(256, 1, 1), 0, s, &x);
}

/// KDA's delta rule over `length` rows from `state` (fp32 (heads, 128 v, 128 k), stored back): q, k fp32, v bf16, decays
/// and beta as tf_kda_prep writes them; y fp32 (rows, heads, 128). A prompt's rows (no snapshots, at least prompt_rows)
/// take the half-head blocks, others the 8-row tile: the same sums. `states`: each row's state; `from`: tf_gated_delta's.
pub fn tf_kda(l: *const Launcher, q: CF, k: CF, v: C, gate: CF, beta: CF, state: F, y: F, length: c_int, heads: c_int, s: S, states: F, from: u64) Error!void {
    return tf_kda_as(l, .t12, q, k, v, gate, beta, state, y, length, heads, s, states, from);
}

/// The scans' LDS staging widths (gated_delta.hip): t12 the engine's prompt scan (tiles of 8 steps), t8 the prompt
/// scan staging 8, t16 the scans as before K-5 (16). A step's arithmetic does not depend on the width: the same bytes.
pub const KdaSteps = enum { t12, t8, t16 };

/// tf_kda at a named staging width (the tests' and the timings').
pub fn tf_kda_as(l: *const Launcher, steps: KdaSteps, q: CF, k: CF, v: C, gate: CF, beta: CF, state: F, y: F, length: c_int, heads: c_int, s: S, states: F, from: u64) Error!void {
    if (heads < 1 or length < 1) return invalid("kda");
    if (states == null and length >= prompt_rows) {
        var b: Args = .{};
        inline for (.{ ad(q), ad(k), ad(v), ad(gate), ad(beta), ad(state), ad(y), length, heads }) |x| try b.add(x);
        const f = switch (steps) {
            .t12 => l.kda.prompt,
            .t8 => l.kda.prompt_t8,
            .t16 => l.kda.prompt_t16,
        };
        return l.go(f, dim(2, heads, 1), dim(32, 4, 1), 0, s, &b);
    }
    var a: Args = .{};
    inline for (.{ ad(q), ad(k), ad(v), ad(gate), ad(beta), ad(state), ad(y), length, heads, ad(states), from }) |x| try a.add(x);
    try l.go(if (steps == .t16) l.kda.tile_t16 else l.kda.tile, dim(kda_dim / 8, heads, 1), dim(32, 8, 1), 0, s, &a);
}

/// A KDA verify window that holds its state: the rows' outputs only (tf_gdn_hold's). At most 16 rows.
pub fn tf_kda_hold(l: *const Launcher, q: CF, k: CF, v: C, gate: CF, beta: CF, state: F, y: F, length: c_int, heads: c_int, s: S) Error!void {
    return holdWith(l, l.kda.hold, q, k, v, gate, beta, state, y, length, heads, s);
}

/// tf_kda_hold as before K-5 (16 steps staged): the tests' oracle.
pub fn tf_kda_hold_t16(l: *const Launcher, q: CF, k: CF, v: C, gate: CF, beta: CF, state: F, y: F, length: c_int, heads: c_int, s: S) Error!void {
    return holdWith(l, l.kda.hold_t16, q, k, v, gate, beta, state, y, length, heads, s);
}

fn holdWith(l: *const Launcher, f: Function, q: CF, k: CF, v: C, gate: CF, beta: CF, state: F, y: F, length: c_int, heads: c_int, s: S) Error!void {
    if (heads < 1 or length < 1 or length > 16) return invalid("kda hold");
    var a: Args = .{};
    inline for (.{ ad(q), ad(k), ad(v), ad(gate), ad(beta), ad(state), ad(y), length, heads }) |x| try a.add(x);
    try l.go(f, dim(kda_dim / 8, heads, 1), dim(32, 8, 1), 0, s, &a);
}

/// A keep on the CUDA window saves (cuda_forward.zig commit's replay_layers inputs): `layers` states (layer z at
/// state + z state_stride floats) each run in place through its first `kept` saved rows. k, v (bf16) and decays at
/// z kv_stride elements, beta at z b_stride; each layer's rows from row 0. The state after the window's row kept - 1.
pub fn tf_kda_keep(l: *const Launcher, state: F, state_stride: i64, k: CF, v: C, gate: CF, beta: CF, kv_stride: i64, b_stride: i64, kept: c_int, layers: c_int, heads: c_int, s: S) Error!void {
    if (heads < 1 or kept < 1 or layers < 1) return invalid("kda keep");
    var a: Args = .{};
    inline for (.{ ad(state), state_stride, ad(k), ad(v), ad(gate), ad(beta), kv_stride, b_stride, kept, heads }) |x| try a.add(x);
    try l.go(l.kda.keep, dim(kda_dim / 8, heads, layers), dim(32, 8, 1), 0, s, &a);
}

/// kda.chain's out (glm_kda.cu tf_glm_kda_out): bf16 out (rows, heads, 128) = norm_w rms(y) sigmoid(g), y fp32 rounded
/// to bf16 on read (CUDA's rounding point), the gate rows g bf16 (stride g_stride), norm_w bf16 [128]. The wave form
/// from kda_wave_rows rows where the buffers suit its vector loads; the block form otherwise: the same bytes.
pub fn tf_kda_out(l: *const Launcher, y: CF, g: C, g_stride: c_int, norm_w: C, eps: f32, out: P, rows: c_int, heads: c_int, s: S) Error!void {
    const vec = @rem(g_stride, 4) == 0 and aligned(y, 16) and aligned(g, 8) and aligned(norm_w, 8) and aligned(out, 8);
    const form: KdaForm = if (rows >= kda_wave_rows and vec) .wave else .block;
    return tf_kda_out_as(l, form, y, g, g_stride, norm_w, eps, out, rows, heads, s);
}

/// tf_kda_out in a named form (the tests', and the timings').
pub fn tf_kda_out_as(l: *const Launcher, form: KdaForm, y: CF, g: C, g_stride: c_int, norm_w: C, eps: f32, out: P, rows: c_int, heads: c_int, s: S) Error!void {
    if (heads < 1 or rows < 1) return invalid("kda out");
    var a: Args = .{};
    inline for (.{ heads, ad(y), ad(g), g_stride, ad(norm_w), eps, ad(out) }) |x| try a.add(x);
    if (form == .block) return l.go(l.kda.out, dim(rows, heads, 1), dim(kda_dim, 1, 1), 0, s, &a);
    try a.add(rows);
    try l.go(l.kda.out_wave, dim(cdiv(@as(usize, @intCast(rows)) * @as(usize, @intCast(heads)), 8), 1, 1), dim(256, 1, 1), 0, s, &a);
}

/// forward._conv_shift (glue.convShift): each of `layers` bf16 conv windows (3 rows of c columns, conv_layer elements
/// apart) takes rows keep .. keep + 2 of [window; projection rows] (proj_row apart, proj_layer a layer).
pub fn tf_kda_conv_shift(l: *const Launcher, conv: P, proj: C, keep: i64, conv_layer: i64, proj_layer: i64, proj_row: i64, c: i64, layers: c_int, s: S) Error!void {
    if (keep < 0 or c < 1 or layers < 1) return invalid("kda conv shift");
    var a: Args = .{};
    inline for (.{ ad(conv), ad(proj), keep, conv_layer, proj_layer, proj_row, c }) |x| try a.add(x);
    try l.go(l.kda.conv_shift, dim(layers, cdiv(c, 256), 1), dim(256, 1, 1), 0, s, &a);
}

/// Tokens a chunk of tf_gdn_wy and its packed operands (gated_delta_wy.hpp).
const wy_chunk = 32;
const wy_pack = 128 * wy_chunk * 2;
const wy_square = wy_chunk * wy_chunk * 2;

fn wyAlign(n: usize) usize {
    return (n + 255) & ~@as(usize, 255);
}

/// The scratch bytes of tf_gdn_wy over `length` rows: k^T, q (a key head) and v^T (a value head) in f16 a chunk,
/// T diag(beta e^G), T diag(beta) and P in f16, G in fp32.
pub fn wyScratch(length: usize, key_heads: usize, value_heads: usize) usize {
    const chunks = (length + wy_chunk - 1) / wy_chunk;
    return 2 * wyAlign(chunks * key_heads * wy_pack) + wyAlign(chunks * value_heads * wy_pack) +
        3 * wyAlign(chunks * value_heads * wy_square) + wyAlign(chunks * value_heads * wy_chunk * 4);
}

/// Whether this build's kernels have the gfx11 matrix-core path of tf_gdn_wy.
pub fn wyOnMatrix() bool {
    const c = @import("../caps.zig").Caps.of(@import("../kernels.zig").arch) orelse return false;
    return c.matrix == .wmma11;
}

/// A prompt's DeltaNet over `length` rows (dk = dv = 128, batch 1, state in place) in 32-token chunks on the gfx11
/// matrix cores: f16 operands, fp32 accumulators, another order than tf_gated_delta (numerics=inexact only).
/// `scratch`: wyScratch(length, key_heads, value_heads) bytes.
pub fn tf_gdn_wy(l: *const Launcher, q: CF, k: CF, v: CF, gate: CF, beta: CF, state: F, y: F, length: c_int, key_heads: c_int, value_heads: c_int, scratch: u64, s: S) Error!void {
    if (key_heads < 1 or value_heads < 1 or @rem(value_heads, key_heads) != 0 or length < 1) return invalid("gated delta wy");
    const n: usize = @intCast(length);
    const kh: usize = @intCast(key_heads);
    const vh: usize = @intCast(value_heads);
    const chunks = (n + wy_chunk - 1) / wy_chunk;
    const kT = scratch;
    const q16 = kT + wyAlign(chunks * kh * wy_pack);
    const vT = q16 + wyAlign(chunks * kh * wy_pack);
    const t1 = vT + wyAlign(chunks * vh * wy_pack);
    const t2 = t1 + wyAlign(chunks * vh * wy_square);
    const p = t2 + wyAlign(chunks * vh * wy_square);
    const g = p + wyAlign(chunks * vh * wy_square);
    var a: Args = .{};
    try a.add(ad(q));
    try a.add(ad(k));
    try a.add(ad(v));
    try a.add(ad(gate));
    try a.add(ad(beta));
    inline for (.{ kT, q16, vT, t1, t2, p, g }) |x| try a.add(x);
    try a.add(length);
    try a.add(key_heads);
    try a.add(value_heads);
    try l.go(l.gdn_wy[0], dim(vh, chunks, 1), dim(128, 1, 1), 0, s, &a);
    var b: Args = .{};
    inline for (.{ kT, q16, vT, t1, t2, p, g }) |x| try b.add(x);
    try b.add(ad(state));
    try b.add(ad(y));
    try b.add(length);
    try b.add(key_heads);
    try b.add(value_heads);
    try l.go(l.gdn_wy[1], dim(vh, 1, 1), dim(256, 1, 1), 0, s, &b);
}

pub fn tf_gdn_gate_prefill(l: *const Launcher, av: C, bv: C, kind: c_int, a_log: CF, dt_bias: CF, gate: F, beta: F, count: c_int, heads: c_int, s: S) Error!void {
    var a: Args = .{};
    try a.add(ad(av));
    try a.add(ad(bv));
    try a.add(kind);
    try a.add(ad(a_log));
    try a.add(ad(dt_bias));
    try a.add(ad(gate));
    try a.add(ad(beta));
    try a.add(count);
    try a.add(heads);
    try l.flat(l.op.gdn_gate_prefill, @as(i64, count) * heads, s, &a);
}

pub fn tf_gnorm_silu(
    l: *const Launcher,
    y: CF,
    z: C,
    out: P,
    kind: c_int,
    n: c_longlong,
    sigmoid: c_int,
    s: S,
) Error!void {
    var a: Args = .{};
    try a.add(ad(y));
    try a.add(ad(z));
    try a.add(ad(out));
    try a.add(kind);
    try a.add(n);
    try a.add(sigmoid);
    try l.flat(l.op.gnorm_silu, n, s, &a);
}
