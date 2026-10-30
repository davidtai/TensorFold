//! The MLX affine tiles of tiles/dot2_tiles.hip and tiles/dot2.hip launched from Zig; the registry picks one.

const std = @import("std");
const abi = @import("../abi.zig");
const runtime = @import("../runtime.zig");
const hl = @import("../launch.zig");
const Module = @import("../module.zig").Module;
const Function = @import("../module.zig").Function;
const Stream = @import("../stream.zig").Stream;
const util = @import("util.zig");

const Caps = @import("../caps.zig").Caps;
const registry = @import("core").registry;
const quant = @import("core").quant;
const tuning = @import("../tuning/tuning.zig");
const mlx_entries = @import("mlx_entries.zig");

const Error = runtime.Error;

/// Matrix cores for products: `off` keeps every product on dot2 tiles; `auto` and `on` use them where a tile exists.
pub const Matrix = enum { auto, on, off };

/// Which product tiles a run takes where two give the same bits; every one defaults on.
/// `inexact` (policy numerics=inexact) also lets the tiles that change a product's bits run.
pub const Choices = struct { matrix: Matrix = .auto, stream: bool = true, gemm: bool = true, inexact: bool = false };

/// quant/mlx.hpp's GroupTable, Routing and Affine as the kernels take them by value.
pub const GroupTable = extern struct { p: u64, kind: c_int };
pub const Routing = extern struct { items: u64 = 0, members: u64 = 0, x_div: c_int = 1, first: c_int = 0 };
pub const Arg = extern struct {
    x: u64,
    words: u64,
    scale: GroupTable,
    bias: GroupTable,
    out: u64,
    m: c_int,
    n: c_int,
    k: c_int,
    bits: c_int,
    group: c_int,
    fp16: c_int,
    route: Routing = .{},
    out16: u64 = 0,
};

/// quant/mlx_tiles.hpp's StreamSides: up to four products that share x, K, width and group in one launch.
pub const StreamSides = extern struct {
    words: [4]u64 = @splat(0),
    scale: [4]u64 = @splat(0),
    bias: [4]u64 = @splat(0),
    out: [4]u64 = @splat(0),
    n: [4]c_int = @splat(0),
    first: [5]c_int = @splat(0),
    count: c_int = 0,
    out_half: c_int = 0,
    pair_cols: c_int = 0,
    limit: f32 = 0,
    /// > 1: the launch's row blocks are the grid's x and its column blocks its y (stream.hpp stream_tile)
    row_blocks: c_int = 0,
};

/// One side of a group: its matrix and where its (rows, n) output goes.
pub const Side = struct { words: u64, scale: u64, bias: u64, n: c_int, out: u64 };

comptime {
    std.debug.assert(@sizeOf(GroupTable) == 16 and @sizeOf(Routing) == 24 and @sizeOf(Arg) == 112 and @sizeOf(StreamSides) == 184);
}

/// Everything a launch needs beside the kernels.
pub const Call = struct {
    r: *const runtime.Runtime,
    s: abi.Stream,
    arg: Arg,
    items: c_int = 1,
    sides: []const Side = &.{},
    limit: f32 = 0,
    out_half: bool = false,
    partial: u64 = 0,
    parts: c_int = 1,
};

/// A kernel of the registry: its launch on this backend's kernels.
pub const Launch = *const fn (*const Kernels, Call) Error!void;
pub const Registry = registry.Registry(Launch);
pub const Entry = Registry.E;

pub const bit_widths = [_]c_int{ 2, 3, 4, 5, 6, 8 };
const row_counts = [_]u8{ 1, 2, 4, 8 };
const piece_counts = [_]u8{ 1, 2, 4 };

pub const lane_rows = 8; // kLaneRows
pub const lane_group_max = 128;
const fast_n = 256;
pub const stream_cbs = [_]c_int{ 1, 2, 4, 8 }; // columns a lane carries in the stream tile (1: plain products only)
pub const stream_rows = [_]u8{ 1, 2, 4 }; // its row counts: columns * rows <= 16
pub const stream_waves = 4; // kStreamWaves
pub const streams_wanted = 3000; // kStreamWavesWanted
pub const stream_code_words = 32; // kStreamCodeWords
/// A SwiGLU pair's lane carries at most this many code words (at least 2 columns): on gfx1151 the 4-bit routed gate+up
/// at 2 columns beats 4 and 8 at 1 to 8 rows (more waves in flight; the column's sums are the same)
pub const stream_pair_code_words = 8;
/// The widest columns a lane of a routed plain product (the experts' down) by row tile (stream_rows): on gfx1151 the
/// 4-bit down at K 640 takes 4 at the 2-row tile (8 lost 8 % at 4-8 rows); 1 and 4 rows keep the waves rule
pub const stream_routed_cb_top = [stream_rows.len]c_int{ 8, 4, 4 };
/// The blocks on z of a routed work list (stream_list): each walks about live items / 16 of the plan's items
pub const stream_list_blocks = 16;
/// The columns below which one row of a plain product takes one column a lane (Flash Next's in-proj 16,480 and
/// attention 13,952 do; lm_head 248,320 and the draft readout 79,591 keep the waves rule)
pub const stream_one_col_n = 20_000;

/// Where an unrouted plain product takes the held-codes tile (D-27; stream bench bench14 on gfx1151, the bytes of
/// today's tile): 4-bit at 3 and 4 rows on the rule's columns (`cb`, the 4-row tile); at 5 to 8 rows in one 8-row pass,
/// at CB 4 from held_wide_n columns (Flash Next's in-proj 16,480 and lm_head) and at CB 2 over (held_narrow_lo,
/// held_narrow_hi] (out-proj and attention out, 2,560). Attention in (13,952) and hc down keep two 4-row passes, and
/// 2 rows keep the 2-row tile (served, the held 2-row tile was 5-7 us slower on in-proj and attention in: spec79).
pub const HeldPick = struct { r: usize, c: usize, rows: c_int };
pub const held_wide_n = 16_000;
pub const held_narrow_lo = 1_000;
pub const held_narrow_hi = 4_096;
pub fn heldPick(bits: c_int, m: c_int, total_n: c_int, cb: c_int) ?HeldPick {
    if (bits != 4 or m < 3 or m > 8) return null;
    const ci = std.mem.indexOfScalar(c_int, &stream_cbs, cb) orelse return null;
    if (m <= 4) return if (cb <= 4) .{ .r = 0, .c = ci, .rows = 4 } else null;
    if (total_n >= held_wide_n) return .{ .r = 1, .c = 2, .rows = 8 };
    if (total_n > held_narrow_lo and total_n <= held_narrow_hi) return .{ .r = 1, .c = 1, .rows = 8 };
    return null;
}

test "heldPick: the held tile at 3 and 4 rows on the rule's columns, at 5 to 8 rows by width" {
    try std.testing.expectEqual(@as(?HeldPick, null), heldPick(4, 1, 16480, 1));
    try std.testing.expectEqual(@as(?HeldPick, null), heldPick(4, 2, 248320, 8));
    try std.testing.expectEqual(@as(?HeldPick, .{ .r = 0, .c = 0, .rows = 4 }), heldPick(4, 3, 2560, 1));
    try std.testing.expectEqual(@as(?HeldPick, .{ .r = 0, .c = 2, .rows = 4 }), heldPick(4, 4, 16480, 4));
    try std.testing.expectEqual(@as(?HeldPick, .{ .r = 1, .c = 2, .rows = 8 }), heldPick(4, 6, 16480, 4));
    try std.testing.expectEqual(@as(?HeldPick, .{ .r = 1, .c = 1, .rows = 8 }), heldPick(4, 8, 2560, 1));
    try std.testing.expectEqual(@as(?HeldPick, null), heldPick(4, 5, 13952, 4));
    try std.testing.expectEqual(@as(?HeldPick, null), heldPick(4, 7, 324, 1));
    try std.testing.expectEqual(@as(?HeldPick, null), heldPick(8, 3, 2560, 1));
}

/// The K-parallel tiles of a short prompt: cb columns a lane set, r rows a pass over up to rb row blocks.
pub const KpTile = struct { cb: c_int, r: c_int, waves: c_int, rb: c_int, loop: bool = true };
pub const kp_tiles = [_]KpTile{
    .{ .cb = 8, .r = 1, .waves = 2, .rb = 2, .loop = false },
    .{ .cb = 4, .r = 4, .waves = 2, .rb = 4 },
    .{ .cb = 4, .r = 8, .waves = 2, .rb = 4, .loop = false },
    .{ .cb = 4, .r = 2, .waves = 2, .rb = 4 },
};

/// The GEMM tiles of the m >= 64 products: `gemm` (the default) or the reference `block`.
pub const Tile = enum { gemm, block };

/// Which GEMM tile of the 128-row family a launch runs: the matrix cores, the dot2 tile, or the reference tile.
pub const BlockKind = enum { matrix, gemm, block };

pub const Kernels = struct {
    /// Whether block shape `tier` can take `m` rows: a tile that does not loop over its rows holds rb * r of them.
    pub fn tierTakes(tier: usize, m: c_int) bool {
        return tier >= kp_tiles.len or kp_tiles[tier].loop or m <= kp_tiles[tier].rb * kp_tiles[tier].r;
    }

    caps: Caps,
    wmma: bool,
    /// gfx11: the matrix-core GEMM tile runs (gfx12's WMMA has other layouts).
    matrix: bool,
    /// `off` runs products on dot2 tiles, `on` and `auto` on matrix cores where a tile exists.
    mode: Matrix,
    /// Which GEMM tile the products take; the policy's reference switch picks the reference one.
    tile: Tile,
    /// The tiles that change a product's bits may run (numerics=inexact).
    inexact: bool,
    lanes: [bit_widths.len][row_counts.len][piece_counts.len]Function,
    row: [bit_widths.len][piece_counts.len]Function,
    /// The stream tile (tiles/stream.hpp) by width, rows and columns a lane; null where columns * rows > 16.
    stream: [bit_widths.len][stream_rows.len][stream_cbs.len]?Function,
    /// The same tile with the (gate | up) activation as its epilogue.
    pairs: [bit_widths.len][stream_rows.len][stream_cbs.len]?Function,
    /// The same tile with Nemotron's relu^2 as its epilogue, and with two such products in one launch.
    relu2: [bit_widths.len][stream_rows.len][stream_cbs.len]?Function,
    relu2_pair: [bit_widths.len][stream_rows.len][stream_cbs.len]?Function,
    /// The 5- to 8-row routed work lists at 4-bit (stream.hpp stream_list): [0] the SwiGLU pair at CB 2, [1] the plain
    /// product at CB 4, both on the 4-row tile.
    stream_list: [2]Function,
    /// Off keeps the decode tiles before the stream tile (the policy's reference switch).
    stream_on: bool,
    /// A test's forced columns a lane (a stream_cbs value; 0: the rule), where the shape takes it.
    stream_cb_force: c_int,
    /// The held-codes plain tile at 4-bit (stream.hpp stream_body's HELD) by rows (4, 8) and columns a lane
    /// (stream_cbs; null where not built): `heldPick` says which products take it.
    held: [2][stream_cbs.len]?Function,
    block: [bit_widths.len]Function,
    gemm: [bit_widths.len]Function,
    wmma_gemm: [bit_widths.len]Function,
    /// Qwen3.8-Flash-Next's routed gate and up of a prompt with the activation as epilogue (tiles/moe_gemm.hpp); gfx11.
    moe_gu: ?Function,
    /// The same tile's routed down of a prompt, fp32 out.
    moe_down: ?Function,
    /// Both with the codes dequantized to bf16 (numerics=inexact).
    moe_gu_inexact: ?Function,
    moe_down_inexact: ?Function,
    /// A prompt's plain 4-bit group-32 products at each of dense_ks (tiles/dense_gemm.hpp); gfx11.
    dense: [dense_ks.len]?Function,
    /// hc up (K 320) on the dense tile with tf_fn_hc_mix in its epilogue (tf_affine_dense_hcmix); gfx11.
    dense_hcmix: ?Function,
    /// Fills the tile's gate table (rnd(sigmoid(g)) for every bf16 g) once a device; gfx11.
    hc_gate_fill: ?Function,
    /// The same at each of inexact_ks with the codes dequantized to bf16 (tiles/dense_inexact.hpp); numerics=inexact.
    dense_inexact: [inexact_ks.len]?Function,
    /// A prompt's MoE router (bf16 rows, K 2560) on the same tile, the weights copied (numerics=inexact).
    router_inexact: ?Function,
    /// BF16 weight rows (quant/bf16_tiles.hpp, bits 16): the stream tile by rows (stream_rows) and columns a lane 1, 2;
    /// its SwiGLU pair at 2 columns; the WMMA tile at each of bf16_ks. Null where the GPU has no BF16 dot2 or matrix.
    bf16_stream: [stream_rows.len][2]?Function,
    bf16_pair: [stream_rows.len]?Function,
    /// The same pair with GLM's SwiGLU rounding points (bf16_tiles.hpp StreamGlmSwiglu); its silu table in scale.p.
    bf16_glu: [stream_rows.len]?Function,
    bf16_wmma: [bf16_ks.len]?Function,
    /// The same sums in other block orders and load depths (bf16_tiles.hpp: grouped row tiles, the same with two
    /// windows ahead, two windows ahead alone), and the pre-F-K5 one-chain tile (other bits, a timing reference):
    /// G4-4's measurement, by Bf16Tile.
    bf16_wmma_alt: [4][bf16_ks.len]?Function,
    kp: [bit_widths.len][kp_tiles.len]Function,
    span: [2]Function,
    fold: Function,
    wide: Function,
    tiled: Function,
    reference: Function,
    fill: Function,
    reg: Registry,

    /// `tiles` holds dot2_tiles.hip's kernels, `dot2` dot2.hip's; bf16 on the WMMA build, fp16 on RDNA2.
    pub fn load(tiles_obj: Module, dot2_obj: Module, caps: Caps, choices: Choices) Error!Kernels {
        var k: Kernels = undefined;
        const wmma = caps.act == .bf16;
        k.caps = caps;
        k.wmma = wmma;
        k.matrix = wmma and caps.matrix == .wmma11;
        k.mode = choices.matrix;
        k.inexact = choices.inexact;
        k.tile = if (choices.gemm) .gemm else .block;
        k.stream_on = choices.stream;
        k.stream_cb_force = 0;
        k.reg = Registry.init(tableFor(caps), &mlx_entries.all);
        k.reg.verify(k.env(), .mlx) catch return error.Invalid;
        // the GEMM tile's rows of x a lane keeps (tiles/gemm.hpp GemmShape): 8 for BF16, 16 for FP16
        if (wmma) try k.resolve("7DotBF16", "Li8E", tiles_obj) else try k.resolve("6DotF16", "Li16E", tiles_obj);
        const r = "_ZN2tf4rocm";
        k.span = .{ try dot2_obj.function(r ++ "16affine_dot2_spanILi1EEEvNS0_6AffineEPfi"), try dot2_obj.function(r ++ "16affine_dot2_spanILi8EEEvNS0_6AffineEPfi") };
        k.fold = try dot2_obj.function(r ++ "16affine_dot2_foldEPKfNS0_10GroupTableES3_Pfiii");
        k.wide = try dot2_obj.function(r ++ "16affine_dot2_wideENS0_6AffineE");
        k.tiled = try dot2_obj.function(r ++ "17affine_dot2_tiledENS0_6AffineE");
        k.reference = try dot2_obj.function(r ++ "18affine_dot2_kernelENS0_6AffineE");
        k.fill = try dot2_obj.function(r ++ "13fill_byte_lutEP6__half");
        k.moe_gu = if (k.matrix) try tiles_obj.function(r ++ "18affine_wmma_moe_guILi4EEEvNS0_6AffineEf") else null;
        k.moe_down = if (k.matrix) try tiles_obj.function(r ++ "20affine_wmma_moe_downILi4EEEvNS0_6AffineE") else null;
        k.moe_gu_inexact = if (k.matrix) try tiles_obj.function(r ++ "26affine_wmma_moe_gu_inexactILi4EEEvNS0_6AffineEf") else null;
        k.moe_down_inexact = if (k.matrix) try tiles_obj.function(r ++ "28affine_wmma_moe_down_inexactILi4EEEvNS0_6AffineE") else null;
        k.dense_hcmix = if (k.matrix) try tiles_obj.function("tf_affine_dense_hcmix") else null;
        k.hc_gate_fill = if (k.matrix) try tiles_obj.function("tf_hc_gate_fill") else null;
        inline for (dense_ks, 0..) |dk, i| {
            k.dense[i] = if (k.matrix) try tiles_obj.function(std.fmt.comptimePrint(r ++ "17affine_wmma_denseILi{d}EEEvNS0_6AffineE", .{dk})) else null;
        }
        inline for (inexact_ks, 0..) |dk, i| {
            k.dense_inexact[i] = if (k.matrix) try tiles_obj.function(std.fmt.comptimePrint(r ++ "25affine_wmma_dense_inexactILi{d}EEEvNS0_6AffineE", .{dk})) else null;
        }
        k.router_inexact = if (k.matrix) try tiles_obj.function(r ++ "18affine_wmma_routerILi2560EEEvNS0_6AffineE") else null;
        inline for (stream_rows, 0..) |rows, i| {
            inline for (.{ 1, 2 }, 0..) |cb, c| {
                k.bf16_stream[i][c] = if (wmma) try tiles_obj.function(std.fmt.comptimePrint(r ++ "11bf16_streamILi{d}ELi{d}ELb0EEEvNS0_6AffineENS0_11StreamSidesEii", .{ rows, cb })) else null;
            }
            k.bf16_pair[i] = if (wmma) try tiles_obj.function(std.fmt.comptimePrint(r ++ "11bf16_streamILi{d}ELi2ELb1EEEvNS0_6AffineENS0_11StreamSidesEii", .{rows})) else null;
            k.bf16_glu[i] = if (wmma) try tiles_obj.function(std.fmt.comptimePrint(r ++ "15bf16_stream_gluILi{d}EEEvNS0_6AffineENS0_11StreamSidesEii", .{rows})) else null;
        }
        inline for (bf16_ks, 0..) |bk, i| {
            k.bf16_wmma[i] = if (k.matrix) try tiles_obj.function(std.fmt.comptimePrint(r ++ "9bf16_wmmaILi{d}EEEvNS0_6AffineE", .{bk})) else null;
            inline for (.{ "13bf16_wmma_ord", "14bf16_wmma_ord2", "12bf16_wmma_a2", "15bf16_wmma_chain" }, 0..) |name, v| {
                k.bf16_wmma_alt[v][i] = if (k.matrix) try tiles_obj.function(std.fmt.comptimePrint(r ++ name ++ "ILi{d}EEEvNS0_6AffineE", .{bk})) else null;
            }
        }
        return k;
    }

    fn resolve(k: *Kernels, comptime dot: []const u8, comptime shape: []const u8, m: Module) Error!void {
        @setEvalBranchQuota(400_000);
        inline for (.{ 4, 8 }, 0..) |rows, r| {
            inline for (stream_cbs, 0..) |cb, c| {
                k.held[r][c] = if ((rows == 4 and cb <= 4) or (rows == 8 and (cb == 2 or cb == 4)))
                    try m.function(std.fmt.comptimePrint("_ZN2tf4rocm23affine_dot2_stream_heldINS0_{s}ELi4ELi{d}ELi{d}EEEvNS0_6AffineENS0_11StreamSidesEii", .{ dot, rows, cb }))
                else
                    null;
            }
        }
        k.stream_list = .{
            try m.function(std.fmt.comptimePrint("_ZN2tf4rocm23affine_dot2_stream_listINS0_{s}ELi4ELi4ELi2ELb1EEEvNS0_6AffineENS0_11StreamSidesEiii", .{dot})),
            try m.function(std.fmt.comptimePrint("_ZN2tf4rocm23affine_dot2_stream_listINS0_{s}ELi4ELi4ELi4ELb0EEEvNS0_6AffineENS0_11StreamSidesEiii", .{dot})),
        };
        inline for (bit_widths, 0..) |bits, b| {
            k.block[b] = try m.function(std.fmt.comptimePrint("_ZN2tf4rocm17affine_dot2_blockINS0_{s}ELi{d}EEEvNS0_6AffineE", .{ dot, bits }));
            if (k.matrix) k.wmma_gemm[b] = try m.function(std.fmt.comptimePrint("_ZN2tf4rocm16affine_wmma_gemmILi{d}EEEvNS0_6AffineE", .{bits}));
            k.gemm[b] = try m.function(std.fmt.comptimePrint("_ZN2tf4rocm17affine_gemm_blockINS0_{s}ELi{d}E{s}EEvNS0_6AffineE", .{ dot, bits, shape }));
            inline for (kp_tiles, 0..) |t, i| {
                k.kp[b][i] = try m.function(std.fmt.comptimePrint("_ZN2tf4rocm14affine_gemm_kpINS0_{s}ELi{d}ELi{d}ELi{d}ELi{d}ELi{d}ELb{d}EEEvNS0_6AffineE", .{ dot, bits, t.cb, t.r, t.waves, t.rb, @intFromBool(t.loop) }));
            }
            inline for (piece_counts, 0..) |pieces, p| {
                k.row[b][p] = try m.function(std.fmt.comptimePrint("_ZN2tf4rocm15affine_dot2_rowINS0_{s}ELi{d}ELi{d}EEEvNS0_6AffineEi", .{ dot, bits, pieces }));
                inline for (row_counts, 0..) |rows, r| {
                    k.lanes[b][r][p] = try m.function(std.fmt.comptimePrint("_ZN2tf4rocm17affine_dot2_lanesINS0_{s}ELi{d}ELi{d}ELi{d}EEEvNS0_6AffineE", .{ dot, bits, rows, pieces }));
                }
            }
            inline for (stream_rows, 0..) |rows, r| {
                inline for (stream_cbs, 0..) |cb, c| {
                    k.stream[b][r][c] = if (cb * rows <= 16)
                        try m.function(std.fmt.comptimePrint("_ZN2tf4rocm18affine_dot2_streamINS0_{s}ELi{d}ELi{d}ELi{d}ELb0EEEvNS0_6AffineENS0_11StreamSidesEii", .{ dot, bits, rows, cb }))
                    else
                        null;
                    k.pairs[b][r][c] = if (cb * rows <= 16 and cb >= 2)
                        try m.function(std.fmt.comptimePrint("_ZN2tf4rocm18affine_dot2_streamINS0_{s}ELi{d}ELi{d}ELi{d}ELb1EEEvNS0_6AffineENS0_11StreamSidesEii", .{ dot, bits, rows, cb }))
                    else
                        null;
                    k.relu2[b][r][c] = if (cb * rows <= 16 and cb >= 2)
                        try m.function(
                            std.fmt.comptimePrint("_ZN2tf4rocm24affine_dot2_stream_relu2INS0_{s}" ++
                                "ELi{d}ELi{d}ELi{d}EEEvNS0_6AffineENS0_11StreamSidesEii", .{ dot, bits, rows, cb }),
                        )
                    else
                        null;
                    k.relu2_pair[b][r][c] = if (cb * rows <= 16 and cb >= 2)
                        try m.function(
                            std.fmt.comptimePrint("_ZN2tf4rocm29affine_dot2_stream_relu2_pairINS0_{s}" ++
                                "ELi{d}ELi{d}ELi{d}EEEvNS0_6AffineES3_iii", .{ dot, bits, rows, cb }),
                        )
                    else
                        null;
                }
            }
        }
    }

    /// fp16(bf16(code)) for every byte into the module's constant table: the RDNA2 fp16 tiles read it. Once a device.
    pub fn fillByteLut(k: *const Kernels, r: *const runtime.Runtime, dot2_obj: Module) Error!void {
        if (k.wmma) return;
        const lut = try dot2_obj.global("_ZN2tf4rocm8kByteF16E");
        var args: hl.Args = .{};
        try args.add(lut.address);
        try hl.launch(k.fill, .{ .grid = .{ .x = 1 }, .block = .{ .x = 256 } }, .{ .r = r, .handle = null }, &args);
        try runtime.check(r.api.hipStreamSynchronize(null));
    }

    /// tf_affine_dense_hcmix's gate table into the tiles module: 65,536 gates, once a device.
    pub fn fillHcGate(k: *const Kernels, r: *const runtime.Runtime) Error!void {
        const f = k.hc_gate_fill orelse return;
        var args: hl.Args = .{};
        try hl.launch(f, .{ .grid = .{ .x = 256 }, .block = .{ .x = 256 } }, .{ .r = r, .handle = null }, &args);
        try runtime.check(r.api.hipStreamSynchronize(null));
    }

    /// Launches on `s`; the counting stream runs nothing.
    fn go(r: *const runtime.Runtime, f: Function, grid: hl.Dim3, block: hl.Dim3, s: abi.Stream, args: *hl.Args) Error!void {
        if (s == util.counting) return;
        try hl.launch(f, .{ .grid = grid, .block = block }, .{ .r = r, .handle = s }, args);
    }

    fn refuse(what: []const u8) Error {
        std.log.err("affine: {s}", .{what});
        return error.Invalid;
    }

    fn bitIndex(bits: c_int) ?usize {
        return std.mem.indexOfScalar(c_int, &bit_widths, bits);
    }

    fn pieceIndex(group: c_int) usize {
        return switch (group) {
            32 => 0,
            64 => 1,
            else => 2,
        };
    }

    fn cdiv(n: c_int, by: c_int) c_uint {
        return @intCast(@divTrunc(n + by - 1, by));
    }

    /// Groups a lane covers: the groups rounded up to a power of two, at most 32; none under four groups.
    pub fn rowLanes(groups: c_int) c_int {
        if (groups < 4) return 0;
        var lpc: c_int = 4;
        while (lpc < groups and lpc < 32) lpc <<= 1;
        return lpc;
    }

    /// The lanes a column takes in the K-parallel tile: one a group, 32 at most (kp_lanes).
    fn kpLanes(groups: c_int) c_int {
        return if (groups > 16) 32 else if (groups > 8) 16 else 8;
    }

    /// The bytes the GEMM tile's piece loads need the words aligned to, by width.
    pub fn gemmAlign(bits: c_int) u64 {
        return switch (bits) {
            4, 8 => 16,
            2, 6 => 8,
            else => 4,
        };
    }

    // ---- what the choice sees of this build and of a call ----

    /// The GPU and the run's switches as the registry reads them.
    pub fn env(k: *const Kernels) registry.Env {
        return .{ .bf16 = k.wmma, .matrix = k.matrix, .matrix_on = k.matrix and k.mode != .off, .stream_on = k.stream_on, .gemm_on = k.tile == .gemm, .inexact = k.inexact };
    }

    /// The MLX entry that takes a product, or null.
    pub fn choose(k: *const Kernels, op: registry.Op, path: registry.Path, shape: registry.Shape) ?*const Entry {
        return k.reg.select(k.env(), .mlx, op, path, shape);
    }

    /// The costs of the GPU's generation: RDNA2's, or the gfx11 ones (RDNA4 takes them for its dot2 tiles).
    fn tableFor(caps: Caps) *const tuning.Table {
        return switch (caps.generation) {
            .rdna2 => &tuning.gfx1030,
            .rdna3, .rdna3_5, .rdna4 => &tuning.gfx1100,
        };
    }

    /// The product of `a` as a shape: `items` is a routed plan's item count.
    pub fn shapeOf(a: Arg, items: c_int) registry.Shape {
        const kind: quant.Tables = switch (a.scale.kind) {
            1 => .bf16,
            2 => .f16,
            else => .f32,
        };
        return .{
            .m = @intCast(@max(a.m, 0)),
            .n = @intCast(@max(a.n, 0)),
            .k = @intCast(@max(a.k, 0)),
            .bits = std.math.cast(u8, a.bits) orelse 0,
            .group = std.math.cast(u16, a.group) orelse 0,
            .fp16 = a.fp16 != 0,
            .tables = kind,
            .tables_alike = a.scale.kind == a.bias.kind,
            .x_aligned = a.x % 16 == 0,
            .words_aligned = a.words % gemmAlign(a.bits) == 0,
            .items = if (a.route.items != 0) @intCast(@max(items, 0)) else 0,
        };
    }

    // ---- the tiles ----

    /// The stream tile on `total_n` columns (a group's sum); `sides` counts them off block by block when it has any.
    pub fn streamGo(k: *const Kernels, r: *const runtime.Runtime, a: Arg, sides_in: StreamSides, total_n: c_int, items: c_int, s: abi.Stream) Error!void {
        return k.streamWith(r, a, sides_in, total_n, items, s, false);
    }

    /// `streamGo` with Nemotron's relu^2 as the epilogue: `a.out16` takes relu(round(sum))^2 in the activation type.
    pub fn relu2Go(k: *const Kernels, r: *const runtime.Runtime, a: Arg, items: c_int, s: abi.Stream) Error!void {
        if (a.out16 == 0 or a.out != 0) return refuse("relu2 output");
        return k.streamWith(r, a, .{}, a.n, items, s, true);
    }

    fn streamWith(
        k: *const Kernels,
        r: *const runtime.Runtime,
        a: Arg,
        sides_in: StreamSides,
        total_n: c_int,
        items: c_int,
        s: abi.Stream,
        relu2: bool,
    ) Error!void {
        const b16 = a.bits == 16; // BF16 rows: the rule below keeps them at 1 or 2 columns a lane (16 words a chunk)
        if (b16 and (relu2 or k.bf16_stream[0][0] == null)) return refuse("bf16 stream");
        const b = if (b16) 0 else bitIndex(a.bits).?;
        var sides = sides_in;
        // a pair's lane carries half as many output columns: a gate column and its up column make one
        const pair_div: c_int = if (sides.pair_cols > 0) 2 else 1;
        var lpc_log2: u5 = 0;
        while ((@as(c_int, 1) << lpc_log2) < (a.k >> 5) and lpc_log2 < 5) lpc_log2 += 1;
        const ri: usize = if (a.m == 1) 0 else if (a.m == 2) 1 else 2;
        // the widest column count that still fills the card with waves, within the registers the lane's code words take
        var pick: usize = 0;
        var c: usize = stream_cbs.len;
        while (c > 0) {
            c -= 1;
            // one column a lane is the plain product's only (a pair's gate and up need two; relu^2 keeps its tiles)
            if (stream_cbs[c] * stream_rows[ri] > 16 or stream_cbs[c] * a.bits > stream_code_words or (stream_cbs[c] < 2 and (pair_div == 2 or relu2))) continue;
            if (pair_div == 2 and stream_cbs[c] > 2 and stream_cbs[c] * a.bits > stream_pair_code_words) continue;
            if (a.route.items != 0 and pair_div == 1 and !relu2 and stream_cbs[c] > stream_routed_cb_top[ri]) continue;
            // one row of a plain product (not routed, not a pair or relu^2) of fewer than stream_one_col_n columns at one
            // column a lane: on gfx1151 it beats 4 and 8 on Flash Next's in-proj and attention and matches the rule where
            // the rule took 1; lm_head and the draft head's readout keep the rule (one column a lane slows the served
            // default-spec prompt pass by about 0.1 s)
            if (ri == 0 and a.route.items == 0 and pair_div == 1 and !relu2 and total_n < stream_one_col_n and stream_cbs[c] > 1) continue;
            pick = c;
            const per_block: c_int = stream_waves * (@as(c_int, 32) >> lpc_log2) * @divExact(stream_cbs[c], pair_div);
            if (@as(u64, cdiv(total_n, per_block)) * stream_waves * @as(u64, @intCast(items)) * cdiv(a.m, stream_rows[ri]) >= streams_wanted) break;
        }
        for (stream_cbs, 0..) |cb, i| {
            if (cb != k.stream_cb_force or cb * stream_rows[ri] > 16 or cb * a.bits > stream_code_words or (cb < 2 and (pair_div == 2 or relu2))) continue;
            pick = i;
        }
        if (a.route.items == 0 and pair_div == 1 and !relu2) {
            if (heldPick(a.bits, a.m, total_n, stream_cbs[pick])) |hp| {
                if (k.held[hp.r][hp.c]) |f| return heldGo(r, f, a, sides_in, total_n, items, s, lpc_log2, hp.rows, stream_cbs[hp.c]);
            }
        }
        const per_block: c_int = stream_waves * (@as(c_int, 32) >> lpc_log2) * @divExact(stream_cbs[pick], pair_div);
        var blocks: c_uint = cdiv(total_n, per_block);
        if (sides.count > 0) {
            blocks = 0;
            for (0..@intCast(sides.count)) |i| {
                sides.first[i] = @intCast(blocks);
                blocks += cdiv(sides.n[i], per_block);
            }
            sides.first[@intCast(sides.count)] = @intCast(blocks);
        }
        // rows past the tile's: the row blocks on x and the column blocks on y, so a column block's passes are
        // dispatched together and read its codes once from DRAM (y holds at most 65,535 blocks, else rows stay on y)
        const row_blocks = cdiv(a.m, stream_rows[ri]);
        sides.row_blocks = if (row_blocks > 1 and blocks <= 65535) @intCast(row_blocks) else 0;
        // 5 to 8 rows of a routed 4-bit plan on the 4-row tile (the gate+up pair at CB 2, the down at CB 4, wide-aligned):
        // the work list, so no block is dispatched for a dead item or a dead row block
        const list_cb: c_int = if (pair_div == 2) 2 else 4;
        if (a.route.items != 0 and !relu2 and sides.count == 0 and a.m > 4 and a.bits == 4 and ri == 2 and
            stream_cbs[pick] == list_cb and ((a.words | @as(u64, @intCast(@divExact(a.k * a.bits, 8)))) & 15) == 0)
        {
            var la: hl.Args = .{};
            try la.add(a);
            try la.add(sides);
            try la.add(@as(c_int, lpc_log2));
            try la.add(@as(c_int, switch (a.group) {
                32 => 0,
                64 => 1,
                else => 2,
            }));
            try la.add(items);
            const lf = k.stream_list[if (pair_div == 2) 0 else 1];
            return go(r, lf, .{ .x = blocks, .y = 1, .z = @intCast(@min(items, stream_list_blocks)) }, .{ .x = 32 * stream_waves }, s, &la);
        }
        var args: hl.Args = .{};
        try args.add(a);
        try args.add(sides);
        try args.add(@as(c_int, lpc_log2));
        try args.add(@as(c_int, switch (a.group) {
            32 => 0,
            64 => 1,
            else => 2,
        }));
        const table = if (sides.pair_cols > 0) &k.pairs else if (relu2) &k.relu2 else &k.stream;
        // a BF16 pair with a table in scale.p (BF16 rows carry no scale) takes GLM's SwiGLU rounding points
        const glu = b16 and sides.pair_cols > 0 and a.scale.p != 0;
        const f = if (b16) (if (glu) k.bf16_glu[ri] else if (sides.pair_cols > 0) k.bf16_pair[ri] else k.bf16_stream[ri][pick]).? else table[b][ri][pick].?;
        const grid: hl.Dim3 = if (sides.row_blocks > 1)
            .{ .x = row_blocks, .y = blocks, .z = @intCast(items) }
        else
            .{ .x = blocks, .y = row_blocks, .z = @intCast(items) };
        try go(r, f, grid, .{ .x = 32 * stream_waves }, s, &args);
    }

    /// A plain product on the held-codes tile: `rows` rows a block (every row of the launch in one row block at 5-8).
    fn heldGo(r: *const runtime.Runtime, f: Function, a: Arg, sides_in: StreamSides, total_n: c_int, items: c_int, s: abi.Stream, lpc_log2: u5, rows: c_int, cb: c_int) Error!void {
        const per_block: c_int = stream_waves * (@as(c_int, 32) >> lpc_log2) * cb;
        var sides = sides_in;
        var blocks: c_uint = cdiv(total_n, per_block);
        if (sides.count > 0) {
            blocks = 0;
            for (0..@intCast(sides.count)) |i| {
                sides.first[i] = @intCast(blocks);
                blocks += cdiv(sides.n[i], per_block);
            }
            sides.first[@intCast(sides.count)] = @intCast(blocks);
        }
        sides.row_blocks = 0;
        var args: hl.Args = .{};
        try args.add(a);
        try args.add(sides);
        try args.add(@as(c_int, lpc_log2));
        try args.add(@as(c_int, switch (a.group) {
            32 => 0,
            64 => 1,
            else => 2,
        }));
        try go(r, f, .{ .x = blocks, .y = cdiv(a.m, rows), .z = @intCast(items) }, .{ .x = 32 * stream_waves }, s, &args);
    }

    /// One row on the row tile: a lane owns a group, a wave reads a column's row in one run.
    pub fn rowGo(k: *const Kernels, r: *const runtime.Runtime, a: Arg, s: abi.Stream, items: c_int) Error!void {
        const b = bitIndex(a.bits) orelse return refuse("bits");
        var args: hl.Args = .{};
        try args.add(a);
        try args.add(rowLanes(@divExact(a.k, a.group)));
        try go(r, k.row[b][pieceIndex(a.group)], .{ .x = cdiv(a.n, 128), .z = @intCast(items) }, .{ .x = 128 }, s, &args);
    }

    /// Up to 8 rows on the column tile.
    pub fn lanesGo(k: *const Kernels, r: *const runtime.Runtime, a: Arg, s: abi.Stream, items: c_int) Error!void {
        const b = bitIndex(a.bits) orelse return refuse("bits");
        var args: hl.Args = .{};
        try args.add(a);
        const ri: usize = if (a.m == 1) 0 else if (a.m == 2) 1 else if (a.m <= 4) 2 else 3;
        try go(r, k.lanes[b][ri][pieceIndex(a.group)], .{ .x = cdiv(a.n, 32), .z = @intCast(items) }, .{ .x = 256 }, s, &args);
    }

    /// The wide tile of the RDNA2 decode products of 9 rows and more.
    pub fn wideGo(k: *const Kernels, r: *const runtime.Runtime, a: Arg, s: abi.Stream) Error!void {
        var args: hl.Args = .{};
        try args.add(a);
        try go(r, k.wide, .{ .x = cdiv(a.n, 32), .y = cdiv(a.m, 128) }, .{ .x = 512 }, s, &args);
    }

    /// Any group width up to 128 that the others do not take, 16 columns by 16 rows a block.
    pub fn tiledGo(k: *const Kernels, r: *const runtime.Runtime, a: Arg, s: abi.Stream) Error!void {
        var args: hl.Args = .{};
        try args.add(a);
        try go(r, k.tiled, .{ .x = cdiv(a.n, 16), .y = cdiv(a.m, 16) }, .{ .x = 256 }, s, &args);
    }

    /// The one-thread tile every product has: the reference of the others.
    pub fn referenceGo(k: *const Kernels, r: *const runtime.Runtime, a: Arg, s: abi.Stream) Error!void {
        var args: hl.Args = .{};
        try args.add(a);
        try go(r, k.reference, .{ .x = cdiv(a.n, 32), .y = @intCast(a.m) }, .{ .x = 32 }, s, &args);
    }

    /// The decode tile over groups [z * per, (z + 1) * per) into `partial`, then the fold in group order.
    pub fn splitGo(k: *const Kernels, r: *const runtime.Runtime, a: Arg, partial: u64, parts: c_int, s: abi.Stream) Error!void {
        const groups = @divTrunc(a.k, a.group);
        var args: hl.Args = .{};
        try args.add(a);
        try args.add(partial);
        try args.add(@divExact(groups, parts));
        try go(r, k.span[if (a.m == 1) 0 else 1], .{ .x = cdiv(a.n, fast_n), .y = cdiv(a.m, 8), .z = @intCast(parts) }, .{ .x = fast_n }, s, &args);
        var fold: hl.Args = .{};
        try fold.add(partial);
        try fold.add(a.scale);
        try fold.add(a.bias);
        try fold.add(a.out);
        try fold.add(a.m);
        try fold.add(a.n);
        try fold.add(groups);
        try go(r, k.fold, .{ .x = cdiv(a.n, fast_n), .y = @intCast(a.m) }, .{ .x = fast_n }, s, &fold);
    }

    /// The 128 x 128 tile of a kind over every item of a plan (or the plain product).
    pub fn blockGo(k: *const Kernels, r: *const runtime.Runtime, a: Arg, s: abi.Stream, items: c_int, kind: BlockKind) Error!void {
        const b = bitIndex(a.bits) orelse return refuse("bits");
        var args: hl.Args = .{};
        try args.add(a);
        const f = switch (kind) {
            .matrix => k.wmma_gemm[b],
            .gemm => k.gemm[b],
            .block => k.block[b],
        };
        try go(r, f, .{ .x = cdiv(a.n, 128), .y = cdiv(a.m, 128), .z = @intCast(items) }, .{ .x = 256 }, s, &args);
    }

    /// The shapes tiles/moe_gemm.hpp compiles (MoeGateUp, MoeDown): Flash Next's expert banks, 4-bit group 32, bf16 x.
    pub const MoeShape = struct { n: c_int, k: c_int, bits: c_int = 4, group: c_int = 32, blocks: c_uint };
    pub const moe_gu_shape: MoeShape = .{ .n = 1280, .k = 2560, .blocks = 10 };
    pub const moe_down_shape: MoeShape = .{ .n = 2560, .k = 640, .blocks = 20 };

    /// Whether moeGateUp takes the plan `a` of `items` items (its output aside).
    pub fn moeGateUpTakes(k: *const Kernels, a: Arg, items: c_int) bool {
        return k.moe_gu != null and k.moeTakes(a, items, moe_gu_shape);
    }

    /// Whether moeDown takes the plan `a` of `items` items (its output aside).
    pub fn moeDownTakes(k: *const Kernels, a: Arg, items: c_int) bool {
        return k.moe_down != null and k.moeTakes(a, items, moe_down_shape);
    }

    fn moeTakes(k: *const Kernels, a: Arg, items: c_int, sh: MoeShape) bool {
        if (k.mode == .off or k.tile != .gemm) return false;
        if (a.fp16 != 0 or a.n != sh.n or a.k != sh.k or a.bits != sh.bits or a.group != sh.group) return false;
        // the tile reads 8 groups' bf16 scales (and biases) of a row in one 16-byte load: the bf16 planes only
        if (a.route.items == 0 or a.route.members == 0 or items < 1 or a.scale.kind != 1 or a.bias.kind != 1) return false;
        return a.x % 16 == 0 and a.words % 16 == 0;
    }

    /// A prompt's routed gate and up of that shape as silu(gate) * up into `a.out16` (bf16, pairs x n / 2): the bytes of
    /// the matrix tile followed by moe_act_kernel. False, nothing launched, for any other shape or with the matrix off.
    pub fn moeGateUp(k: *const Kernels, r: *const runtime.Runtime, a: Arg, items: c_int, limit: f32, s: abi.Stream) Error!bool {
        if (!k.moeGateUpTakes(a, items) or a.out16 == 0) return false;
        var args: hl.Args = .{};
        try args.add(a);
        try args.add(limit);
        // numerics=inexact: the codes dequantized to bf16 in the tile, other bits
        const f = if (k.inexact) k.moe_gu_inexact orelse k.moe_gu.? else k.moe_gu.?;
        try go(r, f, .{ .x = moe_gu_shape.blocks, .z = @intCast(items) }, .{ .x = 256 }, s, &args);
        return true;
    }

    /// A prompt's routed down of that shape into `a.out` (fp32, pairs x n): the bytes of the matrix tile. False, nothing
    /// launched, for any other shape or with the matrix off.
    pub fn moeDown(k: *const Kernels, r: *const runtime.Runtime, a: Arg, items: c_int, s: abi.Stream) Error!bool {
        if (!k.moeDownTakes(a, items) or a.out == 0 or a.out16 != 0) return false;
        var args: hl.Args = .{};
        try args.add(a);
        const f = if (k.inexact) k.moe_down_inexact orelse k.moe_down.? else k.moe_down.?;
        try go(r, f, .{ .x = moe_down_shape.blocks, .z = @intCast(items) }, .{ .x = 256 }, s, &args);
        return true;
    }

    /// The K tiles/dense_gemm.hpp compiles (affine_wmma_dense): Flash Next's trunk, 4-bit group 32, bf16 x and tables.
    pub const dense_ks = [_]c_int{ 320, 2560, 6144, 10240 };

    pub fn denseIndex(k: c_int) ?usize {
        for (dense_ks, 0..) |d, i| if (d == k) return i;
        return null;
    }

    /// A prompt's plain product on the dense tile of its K: the bytes of the matrix tile.
    pub fn denseGo(k: *const Kernels, r: *const runtime.Runtime, a: Arg, s: abi.Stream) Error!void {
        const f = k.dense[denseIndex(a.k) orelse return refuse("dense K")] orelse return refuse("dense tile");
        var args: hl.Args = .{};
        try args.add(a);
        try go(r, f, .{ .x = cdiv(a.n, 128) * cdiv(a.m, 128) }, .{ .x = 256 }, s, &args);
    }

    /// hc up's product (K 320, n = 4 streams of dims) on the dense tile, mixed in its epilogue as tf_fn_hc_mix mixes:
    /// `x` the normed streams [m, n] bf16, `mixed` [m, n / 4] bf16. The bytes of denseGo with out16, then hcMix.
    pub fn denseHcMixGo(k: *const Kernels, r: *const runtime.Runtime, a: Arg, x: u64, mixed: u64, s: abi.Stream) Error!void {
        const f = k.dense_hcmix orelse return refuse("hc mix tile");
        if (a.k != 320 or @rem(a.n, 4 * 32) != 0 or a.fp16 != 0) return refuse("hc mix shape");
        var args: hl.Args = .{};
        try args.add(a);
        try args.add(x);
        try args.add(mixed);
        try go(r, f, .{ .x = cdiv(a.n, 128) * cdiv(a.m, 128) }, .{ .x = 256 }, s, &args);
    }

    /// The K tiles/dense_inexact.hpp compiles (affine_wmma_dense_inexact): 256 x 128 blocks.
    pub const inexact_ks = [_]c_int{ 2560, 6144 };

    pub fn inexactIndex(k: c_int) ?usize {
        for (inexact_ks, 0..) |d, i| if (d == k) return i;
        return null;
    }

    /// numerics=inexact: a prompt's plain product with the codes dequantized to bf16 in the tile.
    /// The K affine_wmma_router compiles.
    pub const router_inexact_k = 2560;

    /// logits (m, n) fp32 = x (m, 2560) bf16 . rows16 (n, 2560) bf16 on the inexact dense tile; false where it cannot.
    pub fn routerInexactGo(k: *const Kernels, r: *const runtime.Runtime, x: u64, rows16: u64, logits: u64, m: c_int, n: c_int, d: c_int, s: abi.Stream) Error!bool {
        const f = k.router_inexact orelse return false;
        if (d != router_inexact_k or x % 16 != 0 or rows16 % 16 != 0) return false;
        const a: Arg = .{ .x = x, .words = rows16, .scale = .{ .p = 0, .kind = 0 }, .bias = .{ .p = 0, .kind = 0 }, .out = logits, .m = m, .n = n, .k = d, .bits = 16, .group = 32, .fp16 = 0 };
        var args: hl.Args = .{};
        try args.add(a);
        try go(r, f, .{ .x = cdiv(n, 128) * cdiv(m, 256) }, .{ .x = 256 }, s, &args);
        return true;
    }

    /// The K the BF16 WMMA tile compiles (bf16_wmma): GLM-5.3-Flash's linears (KDA f_b and g_b 128, kv_b 512, q_b and the
    /// indexer's wq_b 1,536, the shared expert's down 2,048, hidden 4,096, KDA o and eh_proj 8,192, the dense MLP's
    /// down 12,288, MLA o 16,384).
    pub const bf16_ks = [_]c_int{ 128, 512, 1536, 2048, 4096, 8192, 12288, 16384 };

    pub fn bf16Index(k: c_int) ?usize {
        for (bf16_ks, 0..) |d, i| if (d == k) return i;
        return null;
    }

    /// A product of BF16 weight rows: out (m, n) = x (m, k) bf16 . w (n, k) bf16, fp32 into `out` or (with `out16`) bf16.
    pub fn bf16Arg(x: u64, w: u64, out: u64, out16: u64, m: c_int, n: c_int, k: c_int) Arg {
        return .{ .x = x, .words = w, .scale = .{ .p = 0, .kind = 1 }, .bias = .{ .p = 0, .kind = 1 }, .out = out, .m = m, .n = n, .k = k, .bits = 16, .group = 32, .fp16 = 0, .out16 = out16 };
    }

    /// A product of BF16 rows (bf16Arg): with `prompt` on the WMMA tile of its K (32-k groups folded on the VALU), else on the stream
    /// tile (a lane's 32-element chunks, folded by shuffle). x rows k apart, 16-byte aligned; k a multiple of 32 (64 on
    /// the WMMA tile). The two tiles sum in different orders: their bits differ.
    pub fn bf16Go(k: *const Kernels, r: *const runtime.Runtime, a: Arg, prompt: bool, s: abi.Stream) Error!void {
        if (a.bits != 16 or a.m < 1 or a.n < 1 or @rem(a.k, 32) != 0 or a.x % 16 != 0) return refuse("bf16 shape");
        if (!prompt) return k.streamGo(r, a, .{}, a.n, 1, s);
        const f = k.bf16_wmma[bf16Index(a.k) orelse return refuse("bf16 K")] orelse return refuse("bf16 wmma tile");
        if (a.words % 16 != 0) return refuse("bf16 rows unaligned");
        var args: hl.Args = .{};
        try args.add(a);
        try go(r, f, .{ .x = cdiv(a.n, 128) * cdiv(a.m, 128) }, .{ .x = 256 }, s, &args);
    }

    /// The prompt tiles a BF16 product may take (G4-4): `dense` is bf16Go's; `ordered`, `ordered2` and `ahead2` give its
    /// bits in other block orders and load depths; `chain` is the pre-F-K5 one-chain tile (other bits, timing only).
    pub const Bf16Tile = enum { dense, ordered, ordered2, ahead2, chain };

    /// bf16Go's prompt product on tile `t` (the tests' A/B of block shapes).
    pub fn bf16GoTile(k: *const Kernels, r: *const runtime.Runtime, a: Arg, t: Bf16Tile, s: abi.Stream) Error!void {
        if (t == .dense) return k.bf16Go(r, a, true, s);
        if (a.bits != 16 or a.m < 1 or a.n < 1 or @rem(a.k, 64) != 0 or a.x % 16 != 0 or a.words % 16 != 0) return refuse("bf16 shape");
        const i = bf16Index(a.k) orelse return refuse("bf16 K");
        const f = k.bf16_wmma_alt[@backingInt(t) - 1][i] orelse return refuse("bf16 wmma tile");
        const rows: c_int = if (t == .chain) 256 else 128;
        var args: hl.Args = .{};
        try args.add(a);
        try go(r, f, .{ .x = cdiv(a.n, 128) * cdiv(a.m, rows) }, .{ .x = 256 }, s, &args);
    }

    /// Decode's (gate | up) product of BF16 rows on the stream tile: `w` is gate (cols, k) then up (cols, k), act (m, cols)
    /// bf16 = silu(min(g, limit)) * clamp(u, +-limit) (limit 0: no clamp), the product rounded once to bf16.
    pub fn bf16PairGo(k: *const Kernels, r: *const runtime.Runtime, x: u64, w: u64, act: u64, m: c_int, cols: c_int, kk: c_int, limit: f32, s: abi.Stream) Error!void {
        const a = bf16Arg(x, w, 0, act, m, 2 * cols, kk);
        if (m < 1 or cols < 1 or @rem(kk, 32) != 0 or x % 16 != 0) return refuse("bf16 pair shape");
        return k.streamGo(r, a, .{ .pair_cols = cols, .limit = limit }, cols, 1, s);
    }

    /// bf16PairGo with GLM's SwiGLU rounding points (CUDA glm_glue_moe.cuh swiglu1): g and u rounded to bf16, s =
    /// bf16(silu(min(g, limit))) from `silu` (launches/glm.zig Fns.silu, the table of every bf16 g), out = bf16(u s).
    pub fn bf16GluGo(k: *const Kernels, r: *const runtime.Runtime, x: u64, w: u64, act: u64, silu: u64, m: c_int, cols: c_int, kk: c_int, limit: f32, s: abi.Stream) Error!void {
        var a = bf16Arg(x, w, 0, act, m, 2 * cols, kk);
        a.scale.p = silu;
        if (silu == 0 or m < 1 or cols < 1 or @rem(kk, 32) != 0 or x % 16 != 0) return refuse("bf16 glu shape");
        return k.streamGo(r, a, .{ .pair_cols = cols, .limit = limit }, cols, 1, s);
    }

    pub fn denseInexactGo(k: *const Kernels, r: *const runtime.Runtime, a: Arg, s: abi.Stream) Error!void {
        const f = k.dense_inexact[inexactIndex(a.k) orelse return refuse("inexact dense K")] orelse return refuse("inexact dense tile");
        var args: hl.Args = .{};
        try args.add(a);
        try go(r, f, .{ .x = cdiv(a.n, 128) * cdiv(a.m, 256) }, .{ .x = 256 }, s, &args);
    }

    /// K-parallel tile `tier` (an index of kp_tiles) of a short prompt.
    pub fn kpGo(k: *const Kernels, r: *const runtime.Runtime, a: Arg, s: abi.Stream, items: c_int, tier: usize) Error!void {
        const b = bitIndex(a.bits) orelse return refuse("bits");
        var args: hl.Args = .{};
        try args.add(a);
        const t = kp_tiles[tier];
        const sets = @divExact(32, kpLanes(@divExact(a.k, a.group)));
        const blocks = @min(cdiv(a.m, t.r), @as(c_uint, @intCast(t.rb)));
        try go(r, k.kp[b][tier], .{ .x = cdiv(a.n, t.waves * t.cb * sets) * blocks, .z = @intCast(items) }, .{ .x = @intCast(32 * t.waves) }, s, &args);
    }

    // ---- the entry points ----

    /// Prefill's tile, same bits however a prompt is cut: matrix on gfx11 unless off, else dot2 GEMM; few rows K-split.
    pub fn prefillLaunch(k: *const Kernels, r: *const runtime.Runtime, a: Arg, s: abi.Stream, items: c_int) Error!void {
        const op: registry.Op = if (a.route.items != 0) .routed else .project;
        const e = k.choose(op, .prefill, shapeOf(a, items)) orelse return refuse("prefill shape");
        try e.launch(k, .{ .r = r, .s = s, .arg = a, .items = items });
    }

    /// launch_affine on the auto schedule: `schedule` 3 is prefill's, 4 a lane round's; `partial`, `parts` the tests'.
    pub fn run(k: *const Kernels, r: *const runtime.Runtime, arg: Arg, schedule: c_int, s: abi.Stream, partial: u64, parts: c_int, out_half: bool) Error!void {
        var a = arg;
        if (schedule == 1) {
            if (a.fp16 == 0 or k.wmma) return refuse("the reference schedule takes fp16 on RDNA2");
            return k.referenceGo(r, a, s);
        }
        if (schedule == 2 or schedule > 4) return refuse("no such schedule");
        var shape = shapeOf(a, 1);
        shape.round = schedule == 4;
        shape.parts = if (partial != 0 and parts > 1) @intCast(parts) else 1;
        const e = k.choose(.project, if (schedule == 3) .prefill else .decode, shape) orelse return refuse("no tile takes this product");
        if (out_half) {
            // the activation type's output is the decode tile's: fp16 on RDNA2, and either type from the stream tile
            if (schedule == 3 or parts > 1 or !e.roundsAct(shape)) return refuse("fp16 output is the decode tile");
            a.out16 = a.out;
            a.out = 0;
        }
        try e.launch(k, .{ .r = r, .s = s, .arg = a, .partial = partial, .parts = parts, .out_half = out_half });
    }

    /// affine_routed_launch: every item of a plan in one launch; `arg.m` is the most rows an item holds.
    pub fn routed(k: *const Kernels, r: *const runtime.Runtime, arg: Arg, items: c_int, s: abi.Stream) Error!void {
        return k.routedWith(r, arg, items, s, k.tile);
    }

    pub fn routedWith(k: *const Kernels, r: *const runtime.Runtime, arg: Arg, items: c_int, s: abi.Stream, tile: Tile) Error!void {
        // the family's own activation type: bf16 on the WMMA build, fp16 on RDNA2
        if ((arg.fp16 != 0) == k.wmma or arg.route.items == 0 or arg.route.members == 0 or items < 1 or arg.route.x_div < 1) return refuse("routed plan");
        var environment = k.env();
        environment.gemm_on = tile == .gemm;
        const e = k.reg.select(environment, .mlx, .routed, .decode, shapeOf(arg, items)) orelse return refuse("no tile takes this plan");
        try e.launch(k, .{ .r = r, .s = s, .arg = arg, .items = items });
    }

    /// `routed` with relu(round(y))^2 into `arg.out16` as the stream tile's epilogue; false where another tile runs.
    pub fn routedRelu2(k: *const Kernels, r: *const runtime.Runtime, arg: Arg, items: c_int, s: abi.Stream) Error!bool {
        if ((arg.fp16 != 0) == k.wmma or items < 1) return false;
        if (arg.route.items == 0 or arg.route.members == 0 or arg.route.x_div < 1) return false;
        var plain = arg;
        plain.out16 = 0;
        const e = k.reg.select(k.env(), .mlx, .routed, .decode, shapeOf(plain, items)) orelse return false;
        if (e.family != .stream) return false;
        try k.relu2Go(r, arg, items, s);
        return true;
    }

    /// The stacked (gate | up) product as silu(gate) * up into `out16`, clamped by `limit` above 0; false if not taken.
    pub fn pairRun(k: *const Kernels, r: *const runtime.Runtime, arg: Arg, limit: f32, items: c_int, s: abi.Stream) Error!bool {
        var shape = shapeOf(arg, items);
        shape.pairs = arg.out16 != 0;
        const e = k.choose(.routed_act, .decode, shape) orelse return false;
        try e.launch(k, .{ .r = r, .s = s, .arg = arg, .items = items, .limit = limit });
        return true;
    }

    /// Up to four products of `m` rows over one x in one launch; outputs (m, n) fp32, or activation type if `out_half`.
    pub fn groupRun(k: *const Kernels, r: *const runtime.Runtime, arg: Arg, group: []const Side, out_half: bool, s: abi.Stream) Error!void {
        if (group.len == 0 or group.len > 4) return refuse("group shape");
        var a = arg;
        a.n = 0;
        for (group) |side| a.n = @max(a.n, side.n);
        var shape = shapeOf(a, 1);
        shape.round = true;
        const e = k.choose(.group, .decode, shape) orelse return refuse("group shape");
        try e.launch(k, .{ .r = r, .s = s, .arg = a, .sides = group, .out_half = out_half });
    }

    /// The stream tile over a group: the sides' addresses and widths in one launch.
    pub fn groupGo(k: *const Kernels, r: *const runtime.Runtime, arg: Arg, group: []const Side, out_half: bool, s: abi.Stream) Error!void {
        var a = arg;
        var sides: StreamSides = .{ .count = @intCast(group.len), .out_half = @intFromBool(out_half) };
        var total: c_int = 0;
        a.n = 0;
        for (group, 0..) |side, i| {
            sides.words[i] = side.words;
            sides.scale[i] = side.scale;
            sides.bias[i] = side.bias;
            sides.out[i] = side.out;
            sides.n[i] = side.n;
            total += side.n;
            a.n = @max(a.n, side.n);
        }
        a.words = group[0].words;
        a.scale.p = group[0].scale;
        a.bias.p = group[0].bias;
        try k.streamGo(r, a, sides, total, 1, s);
    }

    /// The split count of a decode launch: 1 unless `mode` is 2 (the tests' forced comparison) and the groups divide.
    pub fn splitCount(m: c_int, n: c_int, k: c_int, group: c_int, mode: c_int) c_int {
        if (mode != 2 or m < 1 or n < 512) return 1;
        if (!(group == 32 or group == 64 or group == 128) or k < group or @rem(k, group) != 0) return 1;
        const groups = @divTrunc(k, group);
        if (groups < 2) return 1;
        // A full row tile or a wide column grid already fills the card.
        const blocks = @divTrunc(n + fast_n - 1, fast_n) * @divTrunc(m + 8 - 1, 8);
        if (blocks < 1) return 1;
        var want = @divTrunc(256, blocks);
        if (want < 2) want = 2;
        if (want > groups) want = groups;
        var count = want;
        while (count >= 2) : (count -= 1) {
            if (@rem(groups, count) == 0) return count;
        }
        return 1;
    }
};

test "the split count follows the C rule" {
    try std.testing.expectEqual(@as(c_int, 1), Kernels.splitCount(1, 4096, 4096, 64, 0));
    try std.testing.expectEqual(@as(c_int, 1), Kernels.splitCount(1, 256, 4096, 64, 2));
    try std.testing.expectEqual(@as(c_int, 8), Kernels.splitCount(1, 4096, 512, 64, 2));
}

test "the lane groups of a row tile" {
    try std.testing.expectEqual(@as(c_int, 0), Kernels.rowLanes(3));
    try std.testing.expectEqual(@as(c_int, 4), Kernels.rowLanes(4));
    try std.testing.expectEqual(@as(c_int, 32), Kernels.rowLanes(64));
}

test {
    _ = mlx_entries;
}
