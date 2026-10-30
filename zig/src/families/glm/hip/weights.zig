//! GLM-5.3-Flash's weights on the GPU as the HIP kernels take them: device addresses into the slabs layout.zig plans.
//! Every tensor is as stored (dtype and bytes). BF16 linears are rows [n, k] (out, in) for G4's bf16Arg/bf16Go;
//! fused projections are the row stacks layout.zig places together. Routed experts as G1's hip.exl3.Experts takes them.

const std = @import("std");
const config = @import("../config.zig");
const layout = @import("layout.zig");

const lm = layout.lm;

/// A BF16 linear: rows [n, k] row-major as stored (bf16Arg's `words`, bits 16), 256-byte aligned.
pub const Bf16 = struct { w: u64, n: u32, k: u32 };

/// An mHC site: fn bf16 [24][4 x 4096], base fp32 [24], scale fp32 [3] (pre, post, comb), as stored.
pub const Hc = struct { fn_w: u64, base: u64, scale: u64 };

pub const Kda = struct {
    /// [q | k | v | f_a | g_a | b] rows: 3 x 8,192 + 2 x 128 + 64, K 4,096 (config.kdaProj()).
    proj: Bf16,
    q: Bf16,
    k: Bf16,
    v: Bf16,
    f_a: Bf16,
    g_a: Bf16,
    b: Bf16,
    f_b: Bf16,
    g_b: Bf16,
    /// The q, k, v conv taps: bf16 [3][8,192][4] as stored (q_conv1d, k_conv1d, v_conv1d one after another).
    conv: u64,
    a_log: u64, // fp32 [64]
    dt_bias: u64, // fp32 [8,192]
    o_norm: u64, // bf16 [128]
    o: Bf16,
};

pub const Indexer = struct {
    wk: Bf16, // [128, 4,096]
    weights: Bf16, // weights_proj [32, 4,096]
    wq_b: Bf16, // [4,096, 1,536]
    ln_w: u64, // k_norm weight bf16 [128]
    ln_b: u64, // k_norm bias bf16 [128]
    gate: Bf16, // index_kpool_compress_gate [128, 4,096]
    ape: u64, // index_kpool_compress_ape bf16 [4][128]
};

pub const Mla = struct {
    /// [q_a | kv_a | indexer wk | indexer weights_proj] rows: 1,536 + 512 + 128 + 32, K 4,096 (config.xProj()).
    x: Bf16,
    q_a: Bf16,
    kv_a: Bf16,
    /// [q_b | indexer wq_b] rows: 16,384 + 4,096, K 1,536 (config.qrProj()).
    qr: Bf16,
    q_b: Bf16,
    q_norm: u64, // q_a_layernorm bf16 [1,536]
    kv_norm: u64, // kv_a_layernorm bf16 [512]
    /// As stored, [64 x 512, 512]: head h's 256 key rows at 512 h, then its 256 value rows.
    kv_b: Bf16,
    o: Bf16,
    index: Indexer,
};

pub const Dense = struct {
    /// [gate | up] rows: 2 x 12,288, K 4,096 (bf16PairGo's gate rows then up rows).
    gu: Bf16,
    down: Bf16,
};

/// A layer's EXL3 routed experts, field for field G1's hip.exl3.Experts (zig/src/hip/exl3.zig on glm-halo/g1-exl3):
/// gate, up, down are u64 tables [count] of each expert's trellis address (int16 [K/16][N/16][64]).
pub const Experts = struct {
    gate: u64,
    up: u64,
    down: u64,
    suh_g: u64, // fp16 [count][dims]
    suh_u: u64,
    svh_g: u64, // fp16 [count][width]
    svh_u: u64,
    suh_d: u64, // fp16 [count][width]
    svh_d: u64, // fp16 [count][dims]
    count: usize,
    dims: usize,
    width: usize,
    /// Every expert's gate and up suh are the same bytes (the loader compares them): a row rotates once.
    shared_input: bool,
    limit: f32 = 0,
};

/// The stacked trellis the tables point into: expert e's at base + e x stride.
pub const Stacks = struct { gate: u64, up: u64, down: u64, stride_gu: u64, stride_d: u64 };

pub const Moe = struct {
    router: Bf16, // [288, 4,096]
    bias: u64, // e_score_correction_bias fp32 [288]
    /// [gate | up] rows: 2 x 2,048, K 4,096.
    shared_gu: Bf16,
    shared_down: Bf16,
    experts: Experts,
    stacks: Stacks,
};

pub const Mixer = union(config.Kind) { kda: Kda, mla: Mla };
pub const Ffn = union(enum) { dense: Dense, moe: Moe };

pub const Layer = struct {
    index: u32,
    in_norm: u64, // bf16 [4,096]
    post_norm: u64,
    /// Null on the MTP layer (a plain residual).
    attn_hc: ?Hc,
    ffn_hc: ?Hc,
    mixer: Mixer,
    ffn: Ffn,
};

pub const Mtp = struct {
    enorm: u64,
    hnorm: u64,
    eh: Bf16, // [4,096, 8,192]
    norm: u64, // shared_head.norm
    layer: Layer,
};

pub const Model = struct {
    embed: u64, // bf16 [154,880][4,096]
    norm: u64,
    head: Bf16, // lm_head [154,880, 4,096]
    layers: [config.max_layers - 1]Layer,
    mtp: Mtp,
};

const View = struct {
    l: *const layout.Layout,
    base: []const u64,
    buf: [256]u8 = undefined,

    fn at(v: *View, comptime fmt: []const u8, args: anytype) !u64 {
        const name = try std.fmt.bufPrint(&v.buf, lm ++ fmt, args);
        const p = v.l.find(name) orelse return error.NotPlaced;
        return v.base[p.slab] + p.off;
    }

    fn lin(v: *View, comptime fmt: []const u8, args: anytype) !Bf16 {
        const name = try std.fmt.bufPrint(&v.buf, lm ++ fmt, args);
        return v.linNamed(name);
    }

    fn linNamed(v: *View, name: []const u8) !Bf16 {
        const p = v.l.find(name) orelse return error.NotPlaced;
        if (p.src.rank != 2 or p.src.dtype != .bf16) return error.NotALinear;
        return .{ .w = v.base[p.slab] + p.off, .n = @intCast(p.src.shape[0]), .k = @intCast(p.src.shape[1]) };
    }

    /// The row stack from `first` over `rows` rows of the same K (layout.zig placed them one after another).
    fn stack(first: Bf16, rows: u32) Bf16 {
        return .{ .w = first.w, .n = rows, .k = first.k };
    }

    fn hc(v: *View, i: u32, comptime site: []const u8) !Hc {
        return .{ .fn_w = try v.at("layers.{d}.hc_" ++ site ++ "_fn", .{i}), .base = try v.at("layers.{d}.hc_" ++ site ++ "_base", .{i}), .scale = try v.at("layers.{d}.hc_" ++ site ++ "_scale", .{i}) };
    }

    fn kda(v: *View, i: u32, c: config.Config) !Kda {
        const q = try v.lin("layers.{d}.self_attn.q_proj.weight", .{i});
        return .{
            .proj = stack(q, c.kdaProj()),
            .q = q,
            .k = try v.lin("layers.{d}.self_attn.k_proj.weight", .{i}),
            .v = try v.lin("layers.{d}.self_attn.v_proj.weight", .{i}),
            .f_a = try v.lin("layers.{d}.self_attn.f_a_proj.weight", .{i}),
            .g_a = try v.lin("layers.{d}.self_attn.g_a_proj.weight", .{i}),
            .b = try v.lin("layers.{d}.self_attn.b_proj.weight", .{i}),
            .f_b = try v.lin("layers.{d}.self_attn.f_b_proj.weight", .{i}),
            .g_b = try v.lin("layers.{d}.self_attn.g_b_proj.weight", .{i}),
            .conv = try v.at("layers.{d}.self_attn.q_conv1d.weight", .{i}),
            .a_log = try v.at("layers.{d}.self_attn.A_log", .{i}),
            .dt_bias = try v.at("layers.{d}.self_attn.dt_bias", .{i}),
            .o_norm = try v.at("layers.{d}.self_attn.o_norm.weight", .{i}),
            .o = try v.lin("layers.{d}.self_attn.o_proj.weight", .{i}),
        };
    }

    fn mla(v: *View, i: u32, c: config.Config) !Mla {
        const q_a = try v.lin("layers.{d}.self_attn.q_a_proj.weight", .{i});
        const q_b = try v.lin("layers.{d}.self_attn.q_b_proj.weight", .{i});
        return .{
            .x = stack(q_a, c.xProj()),
            .q_a = q_a,
            .kv_a = try v.lin("layers.{d}.self_attn.kv_a_proj_with_mqa.weight", .{i}),
            .qr = stack(q_b, c.qrProj()),
            .q_b = q_b,
            .q_norm = try v.at("layers.{d}.self_attn.q_a_layernorm.weight", .{i}),
            .kv_norm = try v.at("layers.{d}.self_attn.kv_a_layernorm.weight", .{i}),
            .kv_b = try v.lin("layers.{d}.self_attn.kv_b_proj.weight", .{i}),
            .o = try v.lin("layers.{d}.self_attn.o_proj.weight", .{i}),
            .index = .{
                .wk = try v.lin("layers.{d}.self_attn.indexer.wk.weight", .{i}),
                .weights = try v.lin("layers.{d}.self_attn.indexer.weights_proj.weight", .{i}),
                .wq_b = try v.lin("layers.{d}.self_attn.indexer.wq_b.weight", .{i}),
                .ln_w = try v.at("layers.{d}.self_attn.indexer.k_norm.weight", .{i}),
                .ln_b = try v.at("layers.{d}.self_attn.indexer.k_norm.bias", .{i}),
                .gate = try v.lin("layers.{d}.self_attn.indexer.index_kpool_compress_gate", .{i}),
                .ape = try v.at("layers.{d}.self_attn.indexer.index_kpool_compress_ape", .{i}),
            },
        };
    }

    fn moe(v: *View, i: u32, c: config.Config, shared_input: bool) !Moe {
        var tables: [3]u64 = undefined;
        for (v.l.tables) |t| if (t.layer == i) {
            tables[t.proj] = v.base[t.slab] + t.off;
        };
        const sg = try v.lin("layers.{d}.mlp.shared_experts.gate_proj.weight", .{i});
        return .{
            .router = try v.lin("layers.{d}.mlp.gate.weight", .{i}),
            .bias = try v.at("layers.{d}.mlp.gate.e_score_correction_bias", .{i}),
            .shared_gu = stack(sg, 2 * c.shared_inter),
            .shared_down = try v.lin("layers.{d}.mlp.shared_experts.down_proj.weight", .{i}),
            .experts = .{
                .gate = tables[0],
                .up = tables[1],
                .down = tables[2],
                .suh_g = try v.at("layers.{d}.mlp.experts.0.gate_proj.suh", .{i}),
                .suh_u = try v.at("layers.{d}.mlp.experts.0.up_proj.suh", .{i}),
                .svh_g = try v.at("layers.{d}.mlp.experts.0.gate_proj.svh", .{i}),
                .svh_u = try v.at("layers.{d}.mlp.experts.0.up_proj.svh", .{i}),
                .suh_d = try v.at("layers.{d}.mlp.experts.0.down_proj.suh", .{i}),
                .svh_d = try v.at("layers.{d}.mlp.experts.0.down_proj.svh", .{i}),
                .count = c.experts,
                .dims = c.hidden,
                .width = c.moe_inter,
                .shared_input = shared_input,
                .limit = c.swiglu_limit,
            },
            .stacks = .{
                .gate = try v.at("layers.{d}.mlp.experts.0.gate_proj.trellis", .{i}),
                .up = try v.at("layers.{d}.mlp.experts.0.up_proj.trellis", .{i}),
                .down = try v.at("layers.{d}.mlp.experts.0.down_proj.trellis", .{i}),
                .stride_gu = @as(u64, c.hidden) * c.moe_inter / 2,
                .stride_d = @as(u64, c.hidden) * c.moe_inter / 2,
            },
        };
    }

    fn layer(v: *View, i: u32, c: config.Config, shared_input: bool) !Layer {
        const mtp = i == c.layers;
        return .{
            .index = i,
            .in_norm = try v.at("layers.{d}.input_layernorm.weight", .{i}),
            .post_norm = try v.at("layers.{d}.post_attention_layernorm.weight", .{i}),
            .attn_hc = if (mtp) null else try v.hc(i, "attn"),
            .ffn_hc = if (mtp) null else try v.hc(i, "ffn"),
            .mixer = switch (c.kind(i)) {
                .kda => .{ .kda = try v.kda(i, c) },
                .mla => .{ .mla = try v.mla(i, c) },
            },
            .ffn = if (c.isMoe(i)) .{ .moe = try v.moe(i, c, shared_input) } else .{ .dense = .{
                .gu = stack(try v.lin("layers.{d}.mlp.gate_proj.weight", .{i}), 2 * c.dense_inter),
                .down = try v.lin("layers.{d}.mlp.down_proj.weight", .{i}),
            } },
        };
    }
};

/// The model's addresses from `l` and each slab's device base (`base[slab]`); `shared_input[i]`: layer i's gate and
/// up suh are equal for every expert (the loader's comparison; false for the dense layers).
pub fn model(l: *const layout.Layout, base: []const u64, shared_input: []const bool) !Model {
    const c = l.config;
    if (c.layers != config.max_layers - 1 or base.len != l.slabs.len or shared_input.len != c.layers + 1) return error.BadShape;
    var v: View = .{ .l = l, .base = base };
    var m: Model = undefined;
    m.embed = try v.at("embed_tokens.weight", .{});
    m.norm = try v.at("norm.weight", .{});
    m.head = try v.linNamed("lm_head.weight");
    for (&m.layers, 0..) |*ly, i| ly.* = try v.layer(@intCast(i), c, shared_input[i]);
    const t = c.layers;
    m.mtp = .{
        .enorm = try v.at("layers.{d}.enorm.weight", .{t}),
        .hnorm = try v.at("layers.{d}.hnorm.weight", .{t}),
        .eh = try v.lin("layers.{d}.eh_proj.weight", .{t}),
        .norm = try v.at("layers.{d}.shared_head.norm.weight", .{t}),
        .layer = try v.layer(t, c, shared_input[t]),
    };
    return m;
}
