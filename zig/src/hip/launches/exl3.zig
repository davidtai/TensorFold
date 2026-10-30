//! The EXL3 routed-expert launches (quant/exl3.hip): the trellis decode, the input rotation, the GEMV and the WMMA
//! GEMM over the MoE plan, and the gate/up and down epilogues.

const std = @import("std");
const Launcher = @import("../launches.zig").Launcher;
const Module = @import("../module.zig").Module;
const Function = @import("../module.zig").Function;
const util = @import("util.zig");
const S = util.S;
const Args = util.Args;
const Error = util.Error;
const dim = util.dim;
const invalid = util.invalid;

/// The GEMV's waves split a block's k range in two halves (exl3.hip kGemvHalves).
pub const gemv_halves = 2;
/// The GEMV's row passes: 1, 2, 4 or 8 member rows a pass (tf_exl3_gemv_<MR>).
pub const gemv_rows = [_]usize{ 1, 2, 4, 8 };
/// The most k tiles a GEMV block stages (8 rows x 2,048 halves of LDS).
pub const gemv_max_tiles = 128;

pub const Fns = struct {
    unpack: Function,
    rotate: Function,
    gemv: [gemv_rows.len]Function,
    gemm: Function,
    /// The GEMM on one WMMA chain over K (tf_exl3_gemm_chain): the numerics reference of the tests, not served.
    gemm_chain: Function,
    /// The down GEMM with the down epilogue fused (tf_exl3_gemm_down): y [pair][D] in one launch, the same bits.
    gemm_down: Function,
    gateup: Function,
    down: Function,

    pub fn load(m: Module) Error!Fns {
        return .{
            .unpack = try m.function("tf_exl3_unpack"),
            .rotate = try m.function("tf_exl3_rotate"),
            .gemv = .{ try m.function("tf_exl3_gemv_1"), try m.function("tf_exl3_gemv_2"), try m.function("tf_exl3_gemv_4"), try m.function("tf_exl3_gemv_8") },
            .gemm = try m.function("tf_exl3_gemm"),
            .gemm_chain = try m.function("tf_exl3_gemm_chain"),
            .gemm_down = try m.function("tf_exl3_gemm_down"),
            .gateup = try m.function("tf_exl3_gateup_epilogue"),
            .down = try m.function("tf_exl3_down_epilogue"),
        };
    }
};

fn int(v: usize) c_int {
    return @intCast(v);
}

/// W_q fp16 [16 tiles_k, 16 tiles_n] of one matrix's trellis (k2: half-bits a value, 2 to 16).
pub fn unpack(l: *const Launcher, trellis: u64, k2: usize, codebook: c_int, tiles_k: usize, tiles_n: usize, out: u64, s: S) Error!void {
    if (k2 < 2 or k2 > 16 or tiles_k == 0 or tiles_n == 0) return invalid("exl3 unpack");
    var a: Args = .{};
    try a.add(trellis);
    try a.add(int(k2));
    try a.add(codebook);
    try a.add(int(tiles_n));
    try a.add(out);
    try l.go(l.exl3.unpack, dim(tiles_n, tiles_k, 1), dim(32, 1, 1), 0, s, &a);
}

/// xh = fp16(((x * suh) @ H) / sqrt(128)). per_pair: `units` pairs of `pick`, gate (suh0 -> out0) and up (suh1 ->
/// out1) each at its expert's scales; else `units` rows, one rotation by expert 0's suh0 into out0.
pub fn rotate(l: *const Launcher, x: u64, x_stride: usize, pick: u64, suh0: u64, suh1: u64, out0: u64, out1: u64, units: usize, k: usize, slots: usize, experts: usize, per_pair: bool, s: S) Error!void {
    if (k % 128 != 0 or units == 0) return invalid("exl3 rotate");
    var a: Args = .{};
    try a.add(x);
    try a.add(int(x_stride));
    try a.add(pick);
    try a.add(suh0);
    try a.add(suh1);
    try a.add(out0);
    try a.add(out1);
    try a.add(int(k));
    try a.add(int(slots));
    try a.add(int(experts));
    try a.add(@as(c_int, @intFromBool(per_pair)));
    try l.go(l.exl3.rotate, dim(units, k / 128, @as(usize, if (per_pair) 2 else 1)), dim(32, 1, 1), 0, s, &a);
}

/// One product over the plan: z = X @ W_q (fp32, before the output rotation) for each live item's member pairs.
pub const Product = struct {
    x0: u64, // fp16 rotated inputs of matrix 0 (gate, or down)
    x1: u64, // of matrix 1 (up); unused with mats 1
    x_div: usize, // a member pair p reads input row p / x_div
    tp0: u64, // u64 [experts]: each expert's trellis of matrix 0
    tp1: u64,
    mats: usize, // 1 or 2
    items: u64,
    count: usize, // item slots to launch over (dead ones have no members)
    members: u64,
    z: u64,
    k: usize,
    n: usize,
    pairs: usize, // P: z's pair stride
    experts: usize, // routed experts; items of a higher id (the shared expert) are skipped
};

/// The GEMV's split of K for a shape alone (CUDA's exl3_experts plan: gate/up 4, down 1, else fewer), so a pair's sums
/// never depend on the rows: whole k halves a block, at most gemv_max_tiles k tiles staged.
pub fn split(k: usize, n: usize, gate_up: bool) Error!usize {
    const cands = [_]usize{ if (gate_up) 4 else 1, 2, 1, 8, 16 };
    if (k % 128 != 0 or n % 128 != 0) return invalid("exl3 split");
    const kt = k / 16;
    for (cands) |sk| {
        if (kt % (sk * gemv_halves) == 0 and kt / sk <= gemv_max_tiles) return sk;
    }
    return invalid("exl3 split");
}

/// The GEMV of up to `rows` member rows an item (passes of 1, 2, 4 or 8 rows; more rows take more passes).
pub fn gemv(l: *const Launcher, p: Product, sk: usize, rows: usize, s: S) Error!void {
    if (p.k % 128 != 0 or p.n % 128 != 0 or (p.k / 16) % (sk * gemv_halves) != 0 or p.k / 16 / sk > gemv_max_tiles) return invalid("exl3 gemv");
    if (p.count == 0) return;
    const which: usize = if (rows <= 1) 0 else if (rows <= 2) 1 else if (rows <= 4) 2 else 3;
    const mr = gemv_rows[which];
    const kb = p.k / sk; // halves a block stages a row
    const shared = mr * kb * 2 + 8 * mr * 32 * 4;
    var a: Args = .{};
    try addProduct(&a, p);
    try a.add(int(sk));
    try a.add(int(p.experts));
    try l.go(l.exl3.gemv[which], dim(p.n / 128 * sk, p.count, p.mats), dim(256, 1, 1), @intCast(shared), s, &a);
}

/// The GEMM's threads a block (exl3.hip kGemmThreads: 16 waves, 4 along m x 4 along n, a pass of 128 rows).
pub const gemm_threads = 512;

/// The WMMA GEMM (gfx11): passes of 128 member rows an item, the whole of K a block (no split), 64-k groups folded on
/// the VALU in k order.
pub fn gemm(l: *const Launcher, p: Product, s: S) Error!void {
    return gemmOn(l, l.exl3.gemm, p, s);
}

/// The same product on one WMMA chain over K (the tests' numerics reference).
pub fn gemmChain(l: *const Launcher, p: Product, s: S) Error!void {
    return gemmOn(l, l.exl3.gemm_chain, p, s);
}

/// The down GEMM (mats 1) and the down epilogue in one launch: y [pairs][n] fp32 = the routed pairs' down outputs, the bits
/// of `gemm` then `down` (sk 1). Other pairs keep y.
pub fn gemmDown(l: *const Launcher, p: Product, svh_d: u64, y: u64, s: S) Error!void {
    if (p.k % 32 != 0 or p.n % 128 != 0 or p.mats != 1) return invalid("exl3 gemm down");
    if (p.count == 0) return;
    var a: Args = .{};
    try a.add(p.x0);
    try a.add(int(p.x_div));
    try a.add(p.tp0);
    try a.add(p.items);
    try a.add(p.members);
    try a.add(svh_d);
    try a.add(y);
    try a.add(int(p.k));
    try a.add(int(p.n));
    try a.add(int(p.pairs));
    try a.add(int(p.experts));
    try l.go(l.exl3.gemm_down, dim(p.n / 128, p.count, 1), dim(gemm_threads, 1, 1), 0, s, &a);
}

fn gemmOn(l: *const Launcher, f: Function, p: Product, s: S) Error!void {
    if (p.k % 32 != 0 or p.n % 128 != 0) return invalid("exl3 gemm");
    if (p.count == 0) return;
    var a: Args = .{};
    try addProduct(&a, p);
    try a.add(int(p.experts));
    try l.go(f, dim(p.n / 128, p.count, p.mats), dim(gemm_threads, 1, 1), 0, s, &a);
}

fn addProduct(a: *Args, p: Product) Error!void {
    if (p.mats != 1 and p.mats != 2) return invalid("exl3 product mats");
    try a.add(p.x0);
    try a.add(if (p.mats == 2) p.x1 else p.x0);
    try a.add(int(p.x_div));
    try a.add(p.tp0);
    try a.add(if (p.mats == 2) p.tp1 else p.tp0);
    try a.add(p.items);
    try a.add(p.members);
    try a.add(p.z);
    try a.add(int(p.k));
    try a.add(int(p.n));
    try a.add(int(p.pairs));
}

/// xd [pairs][n] fp16: each routed pair's gate and up (sk splits each in z) through the SwiGLU, rotated for down.
pub fn gateup(l: *const Launcher, z: u64, pick: u64, svh_g: u64, svh_u: u64, suh_d: u64, xd: u64, pairs: usize, n: usize, sk: usize, experts: usize, limit: f32, s: S) Error!void {
    if (n % 128 != 0 or pairs == 0) return invalid("exl3 gate/up epilogue");
    var a: Args = .{};
    try a.add(z);
    try a.add(pick);
    try a.add(svh_g);
    try a.add(svh_u);
    try a.add(suh_d);
    try a.add(xd);
    try a.add(int(pairs));
    try a.add(int(n));
    try a.add(int(sk));
    try a.add(int(experts));
    try a.add(limit);
    try l.go(l.exl3.gateup, dim(epiBlocks(pairs, n), 1, 1), dim(32 * epi_waves, 1, 1), 0, s, &a);
}

/// y [pairs][d] fp32: each routed pair's down sums (sk splits in z) through H and svh_d; other pairs keep y.
pub fn down(l: *const Launcher, z: u64, pick: u64, svh_d: u64, y: u64, pairs: usize, d: usize, sk: usize, experts: usize, s: S) Error!void {
    if (d % 128 != 0 or pairs == 0) return invalid("exl3 down epilogue");
    var a: Args = .{};
    try a.add(z);
    try a.add(pick);
    try a.add(svh_d);
    try a.add(y);
    try a.add(int(pairs));
    try a.add(int(d));
    try a.add(int(sk));
    try a.add(int(experts));
    try l.go(l.exl3.down, dim(epiBlocks(pairs, d), 1, 1), dim(32 * epi_waves, 1, 1), 0, s, &a);
}

/// The epilogues' waves a block (exl3.hip kEpiWaves): a wave a pair's 128-column block.
pub const epi_waves = 8;

fn epiBlocks(pairs: usize, n: usize) usize {
    return (pairs * (n / 128) + epi_waves - 1) / epi_waves;
}

test "the GEMV split depends on the shape alone: GLM's gate/up 4, down 1" {
    try std.testing.expectEqual(@as(usize, 4), try split(4096, 2048, true));
    try std.testing.expectEqual(@as(usize, 1), try split(2048, 4096, false));
    try std.testing.expectEqual(@as(usize, 4), try split(512, 256, true));
    try std.testing.expectEqual(@as(usize, 1), try split(256, 512, false));
    try std.testing.expectEqual(@as(usize, 2), try split(4096, 4096, false)); // 256 k tiles: two blocks of 128
}
