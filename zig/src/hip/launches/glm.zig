//! GLM-5.3-Flash's glue launches (kernels/hip/glm/glue.hpp): mHC pre and post, the residual, norms, stream mean, embed.

const std = @import("std");
const Launcher = @import("../launches.zig").Launcher;
const Module = @import("../module.zig").Module;
const Function = @import("../module.zig").Function;
const util = @import("util.zig");
const runtime = @import("../runtime.zig");
const launch = @import("../launch.zig");
const S = util.S;
const P = util.P;
const C = util.C;
const Args = util.Args;
const Error = util.Error;
const ad = util.ad;
const dim = util.dim;
const cdiv = util.cdiv;

/// The shape the mHC kernels compile: 4 streams of 4,096, 24 mixes, the dots in 16 K blocks of 32 partials.
pub const hidden = 4096;
pub const streams = 4;
pub const hc_blocks = 16;
pub const hc_part = 32;
pub const threads = 256;
const mix_rows = 4;

/// The mixing kernels' scratch: rows x 16 K blocks x 32 fp32.
pub fn hcPartBytes(rows: usize) usize {
    return rows * hc_blocks * hc_part * 4;
}

/// GLM-5.3-Flash's swiglu_limit: the silu table holds bf16(silu(min(g, 10))).
pub const swiglu_limit: f32 = 10;

pub const Fns = struct {
    /// tf_glm_silu's device address (filled by `fill`): the decode pair's table (affine bf16GluGo).
    silu: u64,
    silu_fill: Function,
    swiglu: Function,
    hc_mix: Function,
    hc_finish: Function,
    hc_post: Function,
    stream_mean: Function,
    residual: Function,
    rms: Function,
    embed: Function,

    pub fn load(m: Module) Error!Fns {
        var f: Fns = undefined;
        inline for (@typeInfo(Fns).@"struct".field_names) |name| {
            if (comptime std.mem.eql(u8, name, "silu")) continue;
            @field(f, name) = try m.function("tf_glm_" ++ name);
        }
        f.silu = (try m.global("tf_glm_silu")).address;
        return f;
    }

    /// tf_glm_silu at swiglu_limit, once a device (Launcher.load).
    pub fn fill(f: *const Fns, r: *const runtime.Runtime) Error!void {
        var a: Args = .{};
        try a.add(swiglu_limit);
        try launch.launch(f.silu_fill, .{ .grid = .{ .x = (1 << 16) / threads }, .block = .{ .x = threads } }, .{ .r = r, .handle = null }, &a);
        try runtime.check(r.api.hipStreamSynchronize(null));
    }
};

/// glue.hc_pre: streams x [rows][4 x 4096] bf16 -> out [rows][4096] (the collapsed row, RMS-normed by `norm_w` bf16),
/// post [rows][4] and comb [rows][4][4] fp32 (Sinkhorn, `iters`). fn_w [24][16384] bf16, base [24] and scale [3] fp32.
pub fn hcPre(l: *const Launcher, x: C, fn_w: C, base: C, scale: C, norm_w: C, out: P, post: P, comb: P, part: P, eps: f32, hc_eps: f32, iters: c_int, rows: c_int, s: S) Error!void {
    if (rows < 1) return;
    var a: Args = .{};
    inline for (.{ ad(x), ad(fn_w), ad(part), rows }) |v| try a.add(v);
    try l.go(l.glm.hc_mix, dim(cdiv(rows, mix_rows), hc_blocks, 1), dim(threads, 1, 1), 0, s, &a);
    var b: Args = .{};
    inline for (.{ ad(x), ad(part), ad(base), ad(scale), ad(norm_w), ad(out), ad(post), ad(comb), eps, hc_eps, iters }) |v| try b.add(v);
    try l.go(l.glm.hc_finish, dim(rows, 1, 1), dim(threads, 1, 1), 0, s, &b);
}

/// glue.hc_post: branch = bf16(sum of `parts` fp32 [rows][4096] partials, `stride` floats apart); each stream s becomes
/// bf16(post[s] branch + sum_t comb[t][s] x[t]). `x` may be `xout`.
pub fn hcPost(l: *const Launcher, x: C, xout: P, g: C, parts: c_int, stride: i64, post: C, comb: C, rows: c_int, s: S) Error!void {
    if (rows < 1) return;
    var a: Args = .{};
    inline for (.{ ad(x), ad(xout), ad(g), parts, stride, ad(post), ad(comb) }) |v| try a.add(v);
    try l.go(l.glm.hc_post, dim(rows, hidden / threads, 1), dim(threads, 1, 1), 0, s, &a);
}

/// glue.stream_mean: out [rows][4096] = bf16 of the 4 streams' mean (before the final norm).
pub fn streamMean(l: *const Launcher, x: C, out: P, rows: c_int, s: S) Error!void {
    if (rows < 1) return;
    var a: Args = .{};
    inline for (.{ ad(x), ad(out) }) |v| try a.add(v);
    try l.go(l.glm.stream_mean, dim(rows, hidden / threads, 1), dim(threads, 1, 1), 0, s, &a);
}

/// glue.residual_add (the MTP layer): xout [rows][d] = bf16(x + bf16(sum of `parts` fp32 partials)); x may be xout.
pub fn residual(l: *const Launcher, x: C, xout: P, g: C, parts: c_int, stride: i64, rows: c_int, d: c_int, s: S) Error!void {
    if (rows < 1) return;
    var a: Args = .{};
    inline for (.{ ad(x), ad(xout), ad(g), parts, stride, d }) |v| try a.add(v);
    try l.go(l.glm.residual, dim(rows, cdiv(d, threads), 1), dim(threads, 1, 1), 0, s, &a);
}

/// glue.rmsnorm with the bf16 weight as stored: out = bf16(w bf16(x / rms(x))), rows strided (elements).
pub fn rms(l: *const Launcher, x: C, x_stride: i64, w: C, out: P, o_stride: i64, rows: c_int, width: c_int, eps: f32, s: S) Error!void {
    if (rows < 1) return;
    var a: Args = .{};
    inline for (.{ ad(x), x_stride, ad(w), ad(out), o_stride, width, eps }) |v| try a.add(v);
    try l.go(l.glm.rms, dim(rows, 1, 1), dim(threads, 1, 1), 0, s, &a);
}

/// glue.embed of a BF16 table: each row's token as `copies` streams of d values, out [rows][copies][d].
pub fn embed(l: *const Launcher, ids: C, table: C, out: P, rows: c_int, d: c_int, copies: c_int, s: S) Error!void {
    if (rows < 1) return;
    var a: Args = .{};
    inline for (.{ ad(ids), ad(table), ad(out), d, copies }) |v| try a.add(v);
    try l.go(l.glm.embed, dim(rows, cdiv(d, threads), 1), dim(threads, 1, 1), 0, s, &a);
}

/// GLM's SwiGLU of a prompt with CUDA's rounding points: gu [rows][gate w | up w] bf16 -> act [rows][w] bf16 (s from the
/// silu table, so the limit on g is swiglu_limit; `limit` clamps u and must equal it).
pub fn swiglu(l: *const Launcher, gu: C, act: P, rows: c_int, w: c_int, limit: f32, s: S) Error!void {
    if (limit != swiglu_limit) return util.invalid("glm swiglu limit");
    const total = @as(i64, rows) * w;
    if (total == 0) return;
    var a: Args = .{};
    inline for (.{ ad(gu), ad(act), w, limit, total }) |v| try a.add(v);
    try l.go(l.glm.swiglu, dim(cdiv(total, threads), 1, 1), dim(threads, 1, 1), 0, s, &a);
}

test "the mixing scratch: 16 blocks of 32 fp32 a row" {
    try std.testing.expectEqual(@as(usize, 2048), hcPartBytes(1));
}
