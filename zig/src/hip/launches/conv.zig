//! Causal conv launches: decode, window rows, prefill and the split with its norms and gates.

const Launcher = @import("../launches.zig").Launcher;
const util = @import("util.zig");
const S = util.S;
const C = util.C;
const F = util.F;
const CF = util.CF;
const Args = util.Args;
const Error = util.Error;
const ad = util.ad;
const dim = util.dim;
const cdiv = util.cdiv;
const invalid = util.invalid;
const PlanArgs = @import("plan.zig").PlanArgs;

/// decode.hip's ConvArgs: the linear attention's conv launch.
pub const ConvArgs = extern struct {
    x: u64,
    kind: c_int,
    weight: u64,
    state: u64,
    states: u64 = 0,
    qn: u64,
    kn: u64,
    v: u64,
    channels: c_int,
    kernel: c_int,
    rows: c_int,
    kw: c_int,
    vw: c_int,
    qw: u64 = 0,
    kw_w: u64 = 0,
    eps: f32 = 0,
    norm: c_int = 0,
    ga: u64 = 0,
    gb: u64 = 0,
    a_log: u64 = 0,
    dt_bias: u64 = 0,
    gate: u64 = 0,
    beta: u64 = 0,
    gcount: c_int = 0,
    heads: c_int = 0,
    /// A round's plan: a grid row a slot, its rows and state from the plan (no plan: this launch's own).
    plan: PlanArgs = .{},
    layer: c_int = 0,
    slots: c_int = 1,
    /// A device i32: with row f >= 0, a window's start is the last window's snapshot after row f (states must be set).
    from: u64 = 0,
    /// kda_dim (GLM's KDA): a decay a key channel, `gcount` x kda_dim of them, and beta a head; 0: a gate a head.
    gate_dim: c_int = 0,
    /// KDA's gate_lower_bound (GLM-5.3-Flash: -5).
    lower: f32 = 0,
};

/// KDA's key and value channels a head (recurrence/kda.hpp kDim).
pub const kda_dim = 128;

pub fn tf_conv_decode(l: *const Launcher, x: CF, weight: CF, state: F, y: F, batch: c_int, channels: c_int, kernel: c_int, s: S) Error!void {
    var a: Args = .{};
    try a.add(ad(x));
    try a.add(ad(weight));
    try a.add(ad(state));
    try a.add(ad(y));
    try a.add(channels);
    try a.add(kernel);
    try l.go(l.conv_decode, dim(cdiv(channels, 128), batch, 1), dim(128, 1, 1), 0, s, &a);
}

pub fn tf_conv_rows(l: *const Launcher, x: CF, weight: CF, state: F, y: F, states: F, rows: c_int, channels: c_int, kernel: c_int, s: S) Error!void {
    var a: Args = .{};
    try a.add(ad(x));
    try a.add(ad(weight));
    try a.add(ad(state));
    try a.add(ad(y));
    try a.add(ad(states));
    try a.add(channels);
    try a.add(kernel);
    try a.add(rows);
    try l.go(l.conv_rows, dim(cdiv(channels, 128), 1, 1), dim(128, 1, 1), 0, s, &a);
}

/// A window's rows through the linear attention's conv, split into q, k and v, normed and gated when `c` asks.
pub fn tf_conv_split(l: *const Launcher, c: ConvArgs, s: S) Error!void {
    if (c.kernel < 1 or c.kernel > 8 or (c.norm != 0 and (c.norm != 128 or @rem(c.kw, 128) != 0))) return invalid("conv split");
    if (c.gate_dim != 0 and c.gate_dim != kda_dim) return invalid("conv split gate");
    var a: Args = .{};
    try a.add(c);
    const gate_blocks = if (c.ga != 0) cdiv(c.gcount * @max(c.gate_dim, 1), 128) else 0;
    try l.go(l.dec.conv_split, dim(cdiv(c.channels, 128) + gate_blocks, c.slots, 1), dim(128, 1, 1), 0, s, &a);
}

pub fn tf_conv_prefill(l: *const Launcher, x: C, kind: c_int, weight: CF, state: CF, out: F, new_state: F, len: c_int, channels: c_int, kernel: c_int, s: S) Error!void {
    var a: Args = .{};
    try a.add(ad(x));
    try a.add(kind);
    try a.add(ad(weight));
    try a.add(ad(state));
    try a.add(ad(out));
    try a.add(ad(new_state));
    try a.add(len);
    try a.add(channels);
    try a.add(kernel);
    // a thread a channel over 16 rows (conv_prefill_kernel's kConvRows), at most 8 taps
    if (kernel > 8) return invalid("conv_prefill");
    try l.go(l.op.conv_prefill, dim(cdiv(channels, 256), @max(1, cdiv(len, 16)), 1), dim(256, 1, 1), 0, s, &a);
}

/// A DeltaNet layer's prompt prework in one launch (linear.hpp tf_gdn_prompt_prep): the conv of `len` rows of x
/// (channels 2 kwidth + vwidth), q and k RMS-normed a 128-channel head (weights qw, kw_w), v, the conv's new state, and
/// gate and beta from a and b (`heads` a row). The values of tf_conv_prefill, two column copies, two tf_rms and
/// tf_gdn_gate_prefill.
pub fn tf_gdn_prompt_prep(l: *const Launcher, x: C, kind: c_int, weight: CF, state: CF, new_state: F, q: F, k: F, v: F, qw: CF, kw_w: CF, eps: f32, len: c_int, kwidth: c_int, vwidth: c_int, kernel: c_int, a: C, b: C, a_log: CF, dt_bias: CF, gate: F, beta: F, heads: c_int, s: S) Error!void {
    return promptPrep(l, x, kind, weight, state, new_state, q, k, v, qw, kw_w, eps, len, kwidth, vwidth, kernel, a, b, a_log, dt_bias, gate, beta, heads, 0, 0, s);
}

/// GLM's KDA prompt prework: tf_gdn_prompt_prep's conv, split and q and k norms (L2 as the RMS norm of qkConstants),
/// with a decay a key channel (`heads` x kda_dim a row, `lower` the gate's bound, dt_bias a channel) and beta a head.
pub fn tf_kda_prompt_prep(l: *const Launcher, x: C, kind: c_int, weight: CF, state: CF, new_state: F, q: F, k: F, v: F, qw: CF, kw_w: CF, eps: f32, len: c_int, heads: c_int, kernel: c_int, a: C, b: C, a_log: CF, dt_bias: CF, gate: F, beta: F, lower: f32, s: S) Error!void {
    const width = heads * kda_dim;
    return promptPrep(l, x, kind, weight, state, new_state, q, k, v, qw, kw_w, eps, len, width, width, kernel, a, b, a_log, dt_bias, gate, beta, heads, kda_dim, lower, s);
}

fn promptPrep(l: *const Launcher, x: C, kind: c_int, weight: CF, state: CF, new_state: F, q: F, k: F, v: F, qw: CF, kw_w: CF, eps: f32, len: c_int, kwidth: c_int, vwidth: c_int, kernel: c_int, a: C, b: C, a_log: CF, dt_bias: CF, gate: F, beta: F, heads: c_int, gate_dim: c_int, lower: f32, s: S) Error!void {
    if (kernel < 1 or kernel > 8 or @rem(kwidth, 128) != 0 or len < 1) return invalid("gdn prompt prep");
    var args: Args = .{};
    inline for (.{ ad(x), kind, ad(weight), ad(state), ad(new_state), ad(q), ad(k), ad(v), ad(qw), ad(kw_w), eps, len, kwidth, vwidth, kernel, ad(a), ad(b), ad(a_log), ad(dt_bias), ad(gate), ad(beta), heads, gate_dim, lower }) |val| try args.add(val);
    const n: usize = @intCast(len);
    const kwu: usize = @intCast(kwidth);
    const vwu: usize = @intCast(vwidth);
    const segs = (n + 15) / 16;
    const qk_heads = 2 * kwu / 128;
    const per_v = (vwu + 255) / 256;
    const states = @as(usize, @intCast(kernel - 1)) * (2 * kwu + vwu);
    const gates = n * @as(usize, @intCast(heads)) * @as(usize, if (gate_dim == kda_dim) kda_dim else 1);
    const total = (qk_heads * segs + 7) / 8 + per_v * segs + (gates + 255) / 256 + (states + 255) / 256;
    const blocks: c_int = @intCast(total);
    try l.go(l.op.gdn_prompt_prep, dim(blocks, 1, 1), dim(256, 1, 1), 0, s, &args);
}
