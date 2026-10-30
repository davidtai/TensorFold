//! Where each of GLM-5.3-Flash's text tensors goes on the GPU: its bytes as stored (no cast, no reorder), in one slab a
//! layer, one for the embedding and the final norm, one for the head. Pieces start 256-byte aligned; the tensors of a
//! piece follow each other, so a piece of same-K matrices is also their row-stacked matrix (the CUDA path's fused
//! projections). Each MoE layer's 288 routed experts are stacked by id: one buffer a projection for the trellis, the
//! suh and the svh, and a table of the experts' trellis addresses. The vision tower is not placed. The mcg markers are
//! read and checked on the host, not placed.
//!
//! Host only (headers in, offsets out): `plan` runs on any machine; hip/load.zig moves the bytes.

const std = @import("std");
const config = @import("../config.zig");
const shards = @import("shards.zig");

pub const Src = shards.Src;
pub const DType = shards.DType;
pub const lm = "model.language_model.";
/// A piece's start in its slab.
pub const piece_align = 256;
/// format.py's mcg marker: an int32 scalar a linear whose value names the codebook.
pub const mcg_magic: u32 = 0xCBAC1FED;
pub const projections = [_][]const u8{ "gate_proj", "up_proj", "down_proj" };

/// The byte classes of work/glm53-halo/scripts/tensor_classes.py, in its names.
pub const Class = enum {
    routed_trellis,
    routed_suh,
    routed_svh,
    routed_mcg,
    mtp_experts,
    kda,
    dsa_mla,
    dsa_indexer,
    shared_expert,
    dense_mlp,
    mtp_other,
    lm_head,
    embed,
    router,
    mhc,
    norms,
    vision,

    pub fn label(c: Class) []const u8 {
        return switch (c) {
            .routed_trellis => "routed_experts.trellis",
            .routed_suh => "routed_experts.suh",
            .routed_svh => "routed_experts.svh",
            .routed_mcg => "routed_experts.mcg",
            .mtp_experts => "mtp.routed_experts",
            .kda => "kda",
            .dsa_mla => "dsa.mla",
            .dsa_indexer => "dsa.indexer",
            .shared_expert => "shared_expert",
            .dense_mlp => "dense_mlp(L0-2)",
            .mtp_other => "mtp.other",
            .lm_head => "lm_head",
            .embed => "embed",
            .router => "router",
            .mhc => "mhc",
            .norms => "norms",
            .vision => "vision",
        };
    }
};

pub const class_count = @typeInfo(Class).@"enum".field_names.len;

fn layerOf(name: []const u8) ?u32 {
    const at = std.mem.indexOf(u8, name, "layers.") orelse return null;
    const rest = name[at + "layers.".len ..];
    const end = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    return std.fmt.parseInt(u32, rest[0..end], 10) catch null;
}

/// tensor_classes.py's cls(), rule for rule; `mtp_layer` is its 45. Null: a name it would call "other".
pub fn classify(name: []const u8, mtp_layer: u32) ?Class {
    const has = struct {
        fn f(s: []const u8, part: []const u8) bool {
            return std.mem.indexOf(u8, s, part) != null;
        }
    }.f;
    const l = layerOf(name);
    if (std.mem.startsWith(u8, name, "model.visual")) return .vision;
    if (has(name, "embed_tokens")) return .embed;
    if (std.mem.startsWith(u8, name, "lm_head")) return .lm_head;
    if (l != null and l.? == mtp_layer) return if (has(name, ".experts.")) .mtp_experts else .mtp_other;
    if (has(name, ".experts.")) {
        const tail = name[(std.mem.lastIndexOfScalar(u8, name, '.') orelse return null) + 1 ..];
        if (std.mem.eql(u8, tail, "trellis")) return .routed_trellis;
        if (std.mem.eql(u8, tail, "suh")) return .routed_suh;
        if (std.mem.eql(u8, tail, "svh")) return .routed_svh;
        if (std.mem.eql(u8, tail, "mcg")) return .routed_mcg;
        return null;
    }
    if (has(name, "shared_experts")) return .shared_expert;
    if (has(name, ".mlp.gate.")) return .router;
    if (has(name, ".mlp.")) return .dense_mlp;
    if (has(name, ".indexer.")) return .dsa_indexer;
    if (has(name, ".hc_")) return .mhc;
    if (has(name, "layernorm") or std.mem.endsWith(u8, name, "norm.weight")) return .norms;
    if (has(name, "self_attn")) {
        if (l != null and l.? % 4 == 3) return .dsa_mla;
        return .kda;
    }
    return null;
}

/// One tensor placed: its source and where its bytes start in their slab.
pub const Placed = struct {
    name: []const u8,
    src: Src,
    slab: u16,
    off: u64,
    class: Class,
};

/// A routed layer's table of its experts' trellis addresses (u64 [experts]), filled after the slabs exist.
pub const Table = struct {
    layer: u32,
    proj: u2, // gate, up, down
    slab: u16,
    off: u64,
    /// The stacked trellis it points into: the experts' bytes one after another from `trellis_off`.
    trellis_off: u64,
    stride: u64,
};

/// A routed linear's mcg marker: read from the file, checked, not placed.
pub const Marker = struct { name: []const u8, src: Src };

/// The slabs: a layer each (0 .. layers, the MTP layer last), then the embedding with the final norm, then the head.
pub fn globalSlab(c: config.Config) u16 {
    return @intCast(c.layers + c.mtp);
}
pub fn headSlab(c: config.Config) u16 {
    return @intCast(c.layers + c.mtp + 1);
}

pub const Layout = struct {
    config: config.Config,
    /// Bytes of each slab.
    slabs: []u64,
    placed: []Placed,
    /// `placed` by name.
    by_name: std.StringHashMapUnmanaged(u32),
    tables: []Table,
    markers: []Marker,
    /// Every checkpoint tensor's bytes by class, and the part placed on the GPU.
    stored: [class_count]u64 = @splat(0),
    stored_count: [class_count]u64 = @splat(0),
    on_gpu: [class_count]u64 = @splat(0),
    total_stored: u64 = 0,

    pub fn find(l: *const Layout, name: []const u8) ?Placed {
        const i = l.by_name.get(name) orelse return null;
        return l.placed[i];
    }

    /// Bytes on the GPU over every slab (the alignment gaps and the tables included).
    pub fn slabBytes(l: *const Layout) u64 {
        var n: u64 = 0;
        for (l.slabs) |s| n += s;
        return n;
    }
};

const Builder = struct {
    a: std.mem.Allocator,
    index: *const shards.Index,
    c: config.Config,
    slab: u16 = 0,
    at: u64 = 0,
    slabs: std.ArrayList(u64) = .empty,
    placed: std.ArrayList(Placed) = .empty,
    tables: std.ArrayList(Table) = .empty,
    markers: std.ArrayList(Marker) = .empty,
    used: std.StringHashMapUnmanaged(void) = .empty,
    name_buf: [256]u8 = undefined,
    /// The name of the tensor a refusal is about.
    why: *[]const u8,

    fn begin(b: *Builder, slab: u16) !void {
        if (slab != b.slabs.items.len) return error.SlabOrder;
        try b.slabs.append(b.a, 0);
        b.slab = slab;
        b.at = 0;
    }

    fn end(b: *Builder) void {
        b.slabs.items[b.slab] = std.mem.alignForward(u64, b.at, piece_align);
    }

    fn piece(b: *Builder) void {
        b.at = std.mem.alignForward(u64, b.at, piece_align);
    }

    fn source(b: *Builder, name: []const u8, dtype: DType, shape: []const usize) !Src {
        const s = b.index.get(name) orelse {
            b.why.* = try b.a.dupe(u8, name);
            return error.MissingTensor;
        };
        if (s.dtype != dtype or !std.mem.eql(usize, s.dims(), shape)) {
            b.why.* = try std.fmt.allocPrint(b.a, "{s}: {t} {any}, the HIP family reads {t} {any} as stored", .{ name, s.dtype, s.dims(), dtype, shape });
            return error.UnexpectedTensor;
        }
        // the index's own copy of the name: `name` may live in the builder's buffer
        const got = try b.used.getOrPut(b.a, b.index.tensors.getKey(name).?);
        if (got.found_existing) return error.PlacedTwice;
        return s;
    }

    /// The tensor `fmt` (under `lm` unless `raw`) at the current offset, checked to be `dtype` of `shape`.
    fn putName(b: *Builder, name: []const u8, dtype: DType, shape: []const usize) !void {
        const s = try b.source(name, dtype, shape);
        const owned = try b.a.dupe(u8, name);
        try b.placed.append(b.a, .{ .name = owned, .src = s, .slab = b.slab, .off = b.at, .class = classify(owned, b.c.layers) orelse return error.UnclassifiedTensor });
        b.at += s.len;
    }

    fn put(b: *Builder, comptime fmt: []const u8, args: anytype, dtype: DType, shape: []const usize) !void {
        return b.putName(try std.fmt.bufPrint(&b.name_buf, lm ++ fmt, args), dtype, shape);
    }

    /// One tensor as a piece of its own.
    fn alone(b: *Builder, comptime fmt: []const u8, args: anytype, dtype: DType, shape: []const usize) !void {
        b.piece();
        try b.put(fmt, args, dtype, shape);
    }

    fn marker(b: *Builder, comptime fmt: []const u8, args: anytype) !void {
        const name = try std.fmt.bufPrint(&b.name_buf, lm ++ fmt, args);
        const s = try b.source(name, .i32, &.{1});
        try b.markers.append(b.a, .{ .name = try b.a.dupe(u8, name), .src = s });
    }

    fn hc(b: *Builder, i: u32, comptime site: []const u8) !void {
        const c = b.c;
        const mix = c.hc * (c.hc + 2);
        try b.alone("layers.{d}.hc_" ++ site ++ "_fn", .{i}, .bf16, &.{ mix, c.hc * c.hidden });
        try b.alone("layers.{d}.hc_" ++ site ++ "_base", .{i}, .f32, &.{mix});
        try b.alone("layers.{d}.hc_" ++ site ++ "_scale", .{i}, .f32, &.{3});
    }

    fn kda(b: *Builder, i: u32) !void {
        const c = b.c;
        const d: usize = c.hidden;
        const w: usize = c.kdaWidth();
        const dk: usize = c.kda_dim;
        // [q | k | v | f_a | g_a | b]: kdaProj() rows of K = hidden
        b.piece();
        inline for (.{ "q", "k", "v" }) |x| try b.put("layers.{d}.self_attn." ++ x ++ "_proj.weight", .{i}, .bf16, &.{ w, d });
        inline for (.{ "f_a", "g_a" }) |x| try b.put("layers.{d}.self_attn." ++ x ++ "_proj.weight", .{i}, .bf16, &.{ dk, d });
        try b.put("layers.{d}.self_attn.b_proj.weight", .{i}, .bf16, &.{ c.kda_heads, d });
        try b.alone("layers.{d}.self_attn.f_b_proj.weight", .{i}, .bf16, &.{ w, dk });
        try b.alone("layers.{d}.self_attn.g_b_proj.weight", .{i}, .bf16, &.{ w, dk });
        // the q, k and v conv taps: bf16 [3][width][1][conv] as stored
        b.piece();
        inline for (.{ "q", "k", "v" }) |x| try b.put("layers.{d}.self_attn." ++ x ++ "_conv1d.weight", .{i}, .bf16, &.{ w, 1, c.conv });
        try b.alone("layers.{d}.self_attn.A_log", .{i}, .f32, &.{c.kda_heads});
        try b.alone("layers.{d}.self_attn.dt_bias", .{i}, .f32, &.{w});
        try b.alone("layers.{d}.self_attn.o_norm.weight", .{i}, .bf16, &.{dk});
        try b.alone("layers.{d}.self_attn.o_proj.weight", .{i}, .bf16, &.{ d, w });
    }

    fn mla(b: *Builder, i: u32) !void {
        const c = b.c;
        const d: usize = c.hidden;
        // [q_a | kv_a | indexer wk | indexer weights_proj]: xProj() rows of K = hidden
        b.piece();
        try b.put("layers.{d}.self_attn.q_a_proj.weight", .{i}, .bf16, &.{ c.q_lora, d });
        try b.put("layers.{d}.self_attn.kv_a_proj_with_mqa.weight", .{i}, .bf16, &.{ c.kv_lora, d });
        try b.put("layers.{d}.self_attn.indexer.wk.weight", .{i}, .bf16, &.{ c.i_dim, d });
        try b.put("layers.{d}.self_attn.indexer.weights_proj.weight", .{i}, .bf16, &.{ c.i_heads, d });
        // [q_b | indexer wq_b]: qrProj() rows of K = q_lora
        b.piece();
        try b.put("layers.{d}.self_attn.q_b_proj.weight", .{i}, .bf16, &.{ c.mla_heads * c.nope, c.q_lora });
        try b.put("layers.{d}.self_attn.indexer.wq_b.weight", .{i}, .bf16, &.{ c.i_heads * c.i_dim, c.q_lora });
        try b.alone("layers.{d}.self_attn.q_a_layernorm.weight", .{i}, .bf16, &.{c.q_lora});
        try b.alone("layers.{d}.self_attn.kv_a_layernorm.weight", .{i}, .bf16, &.{c.kv_lora});
        // as stored: head h's key rows [h (nope + v), h (nope + v) + nope), then its value rows
        try b.alone("layers.{d}.self_attn.kv_b_proj.weight", .{i}, .bf16, &.{ c.mla_heads * (c.nope + c.v_dim), c.kv_lora });
        try b.alone("layers.{d}.self_attn.o_proj.weight", .{i}, .bf16, &.{ d, c.mla_heads * c.v_dim });
        try b.alone("layers.{d}.self_attn.indexer.k_norm.weight", .{i}, .bf16, &.{c.i_dim});
        try b.alone("layers.{d}.self_attn.indexer.k_norm.bias", .{i}, .bf16, &.{c.i_dim});
        try b.alone("layers.{d}.self_attn.indexer.index_kpool_compress_gate", .{i}, .bf16, &.{ c.i_dim, d });
        try b.alone("layers.{d}.self_attn.indexer.index_kpool_compress_ape", .{i}, .bf16, &.{ c.kpool, c.i_dim });
    }

    fn dense(b: *Builder, i: u32) !void {
        const c = b.c;
        b.piece(); // [gate | up]: the SwiGLU pair's rows
        inline for (.{ "gate_proj", "up_proj" }) |x| try b.put("layers.{d}.mlp." ++ x ++ ".weight", .{i}, .bf16, &.{ c.dense_inter, c.hidden });
        try b.alone("layers.{d}.mlp.down_proj.weight", .{i}, .bf16, &.{ c.hidden, c.dense_inter });
    }

    fn moe(b: *Builder, i: u32) !void {
        const c = b.c;
        const d: usize = c.hidden;
        const n: usize = c.moe_inter;
        try b.alone("layers.{d}.mlp.gate.weight", .{i}, .bf16, &.{ c.experts, d });
        try b.alone("layers.{d}.mlp.gate.e_score_correction_bias", .{i}, .f32, &.{c.experts});
        b.piece(); // [gate | up]
        inline for (.{ "gate_proj", "up_proj" }) |x| try b.put("layers.{d}.mlp.shared_experts." ++ x ++ ".weight", .{i}, .bf16, &.{ c.shared_inter, d });
        try b.alone("layers.{d}.mlp.shared_experts.down_proj.weight", .{i}, .bf16, &.{ d, c.shared_inter });
        // EXL3 4 bpw: trellis int16 [K/16][N/16][16 bits]; suh fp16 [K], svh fp16 [N]
        const kn = [_][2]usize{ .{ d, n }, .{ d, n }, .{ n, d } };
        var trellis: [3]u64 = undefined;
        inline for (projections, 0..) |p, m| {
            b.piece();
            trellis[m] = b.at;
            for (0..c.experts) |e| try b.put("layers.{d}.mlp.experts.{d}." ++ p ++ ".trellis", .{ i, e }, .i16, &.{ kn[m][0] / 16, kn[m][1] / 16, 16 * 4 });
        }
        inline for (projections, 0..) |p, m| {
            b.piece();
            for (0..c.experts) |e| try b.put("layers.{d}.mlp.experts.{d}." ++ p ++ ".suh", .{ i, e }, .f16, &.{kn[m][0]});
            b.piece();
            for (0..c.experts) |e| try b.put("layers.{d}.mlp.experts.{d}." ++ p ++ ".svh", .{ i, e }, .f16, &.{kn[m][1]});
        }
        for (0..c.experts) |e| inline for (projections) |p| try b.marker("layers.{d}.mlp.experts.{d}." ++ p ++ ".mcg", .{ i, e });
        for (0..3) |m| {
            b.piece();
            try b.tables.append(b.a, .{ .layer = i, .proj = @intCast(m), .slab = b.slab, .off = b.at, .trellis_off = trellis[m], .stride = kn[m][0] / 16 * kn[m][1] / 16 * 16 * 4 * 2 });
            b.at += 8 * c.experts;
        }
    }

    fn layer(b: *Builder, i: u32) !void {
        const c = b.c;
        const d: usize = c.hidden;
        try b.begin(@intCast(i));
        try b.alone("layers.{d}.input_layernorm.weight", .{i}, .bf16, &.{d});
        try b.alone("layers.{d}.post_attention_layernorm.weight", .{i}, .bf16, &.{d});
        const mtp = i == c.layers;
        if (!mtp) {
            try b.hc(i, "attn");
            try b.hc(i, "ffn");
        }
        switch (c.kind(i)) {
            .kda => try b.kda(i),
            .mla => try b.mla(i),
        }
        if (c.isMoe(i)) try b.moe(i) else try b.dense(i);
        if (mtp) {
            try b.alone("layers.{d}.enorm.weight", .{i}, .bf16, &.{d});
            try b.alone("layers.{d}.hnorm.weight", .{i}, .bf16, &.{d});
            try b.alone("layers.{d}.eh_proj.weight", .{i}, .bf16, &.{ d, 2 * d });
            try b.alone("layers.{d}.shared_head.norm.weight", .{i}, .bf16, &.{d});
        }
        b.end();
    }
};

/// The placement of every text tensor of `index` for config `c` (the whole GLM-5.3-Flash shape with its MTP layer).
/// A missing tensor, another dtype or shape, or a tensor the plan does not name (but the vision tower's) is refused;
/// `why` names it.
pub fn plan(a: std.mem.Allocator, index: *const shards.Index, c: config.Config, why: *[]const u8) !Layout {
    if (c.mtp != 1 or !c.exl3) return error.NotGlm53FlashExl3;
    var b: Builder = .{ .a = a, .index = index, .c = c, .why = why };
    for (0..c.layers + c.mtp) |i| try b.layer(@intCast(i));
    try b.begin(globalSlab(c));
    try b.alone("embed_tokens.weight", .{}, .bf16, &.{ c.vocab, c.hidden });
    try b.alone("norm.weight", .{}, .bf16, &.{c.hidden});
    b.end();
    try b.begin(headSlab(c));
    try b.putName("lm_head.weight", .bf16, &.{ c.vocab, c.hidden });
    b.end();
    var l: Layout = .{ .config = c, .slabs = b.slabs.items, .placed = b.placed.items, .by_name = .empty, .tables = b.tables.items, .markers = b.markers.items };
    for (l.placed, 0..) |p, i| {
        try l.by_name.put(a, p.name, @intCast(i));
        l.on_gpu[@backingInt(p.class)] += p.src.len;
    }
    for (index.tensors.keys(), index.tensors.values()) |name, s| {
        const class = classify(name, c.layers) orelse {
            why.* = try a.dupe(u8, name);
            return error.UnclassifiedTensor;
        };
        l.stored[@backingInt(class)] += s.len;
        l.stored_count[@backingInt(class)] += 1;
        l.total_stored += s.len;
        if (class != .vision and !b.used.contains(name)) {
            why.* = try a.dupe(u8, name);
            return error.UnplacedTensor;
        }
    }
    return l;
}

/// PLAN.md section 1's rows over the classes: label and members.
pub const plan_rows = [_]struct { label: []const u8, classes: []const Class }{
    .{ .label = "Routed expert trellis, layers 3-44", .classes = &.{.routed_trellis} },
    .{ .label = "Routed expert suh + svh, layers 3-44", .classes = &.{ .routed_suh, .routed_svh, .routed_mcg } },
    .{ .label = "MTP routed experts", .classes = &.{.mtp_experts} },
    .{ .label = "KDA (34 layers)", .classes = &.{.kda} },
    .{ .label = "DSA MLA (11 layers)", .classes = &.{.dsa_mla} },
    .{ .label = "DSA indexer (11 layers)", .classes = &.{.dsa_indexer} },
    .{ .label = "Shared experts (42 layers)", .classes = &.{.shared_expert} },
    .{ .label = "Dense MLP (layers 0-2)", .classes = &.{.dense_mlp} },
    .{ .label = "MTP non-expert", .classes = &.{.mtp_other} },
    .{ .label = "lm_head", .classes = &.{.lm_head} },
    .{ .label = "embed", .classes = &.{.embed} },
    .{ .label = "Router, mHC, norms", .classes = &.{ .router, .mhc, .norms } },
    .{ .label = "Vision tower", .classes = &.{.vision} },
};

// ---- tests: the plan over a synthetic index of the real names, dtypes and shapes (no bytes) ----

const Fake = struct {
    a: std.mem.Allocator,
    x: shards.Index,
    at: u64 = 0,

    fn add(f: *Fake, name: []const u8, dtype: DType, shape: []const usize) !void {
        var s: Src = .{ .shard = 0, .off = f.at, .len = dtype.size(), .dtype = dtype, .rank = @intCast(shape.len), .shape = @splat(1) };
        for (shape, 0..) |d, i| {
            s.shape[i] = d;
            s.len *= d;
        }
        f.at += s.len;
        try f.x.tensors.put(f.a, try f.a.dupe(u8, name), s);
    }

    fn lmAdd(f: *Fake, comptime fmt: []const u8, args: anytype, dtype: DType, shape: []const usize) !void {
        var buf: [256]u8 = undefined;
        try f.add(try std.fmt.bufPrint(&buf, lm ++ fmt, args), dtype, shape);
    }
};

/// The checkpoint's text tensors at GLM-5.3-Flash's shape (Mia-AiLab's names, dtypes and shapes) and a vision tensor.
fn fakeIndex(a: std.mem.Allocator, c: config.Config) !shards.Index {
    var f: Fake = .{ .a = a, .x = .{ .files = &.{}, .sizes = &.{}, .tensors = .empty } };
    const d: usize = c.hidden;
    for (0..c.layers + 1) |li| {
        const i: u32 = @intCast(li);
        try f.lmAdd("layers.{d}.input_layernorm.weight", .{i}, .bf16, &.{d});
        try f.lmAdd("layers.{d}.post_attention_layernorm.weight", .{i}, .bf16, &.{d});
        if (i < c.layers) inline for (.{ "attn", "ffn" }) |s| {
            try f.lmAdd("layers.{d}.hc_" ++ s ++ "_fn", .{i}, .bf16, &.{ 24, 16384 });
            try f.lmAdd("layers.{d}.hc_" ++ s ++ "_base", .{i}, .f32, &.{24});
            try f.lmAdd("layers.{d}.hc_" ++ s ++ "_scale", .{i}, .f32, &.{3});
        };
        if (c.kind(i) == .kda) {
            inline for (.{ "q", "k", "v" }) |x| {
                try f.lmAdd("layers.{d}.self_attn." ++ x ++ "_proj.weight", .{i}, .bf16, &.{ 8192, 4096 });
                try f.lmAdd("layers.{d}.self_attn." ++ x ++ "_conv1d.weight", .{i}, .bf16, &.{ 8192, 1, 4 });
            }
            inline for (.{ "f_a", "g_a" }) |x| try f.lmAdd("layers.{d}.self_attn." ++ x ++ "_proj.weight", .{i}, .bf16, &.{ 128, 4096 });
            inline for (.{ "f_b", "g_b" }) |x| try f.lmAdd("layers.{d}.self_attn." ++ x ++ "_proj.weight", .{i}, .bf16, &.{ 8192, 128 });
            try f.lmAdd("layers.{d}.self_attn.b_proj.weight", .{i}, .bf16, &.{ 64, 4096 });
            try f.lmAdd("layers.{d}.self_attn.A_log", .{i}, .f32, &.{64});
            try f.lmAdd("layers.{d}.self_attn.dt_bias", .{i}, .f32, &.{8192});
            try f.lmAdd("layers.{d}.self_attn.o_norm.weight", .{i}, .bf16, &.{128});
            try f.lmAdd("layers.{d}.self_attn.o_proj.weight", .{i}, .bf16, &.{ 4096, 8192 });
        } else {
            try f.lmAdd("layers.{d}.self_attn.q_a_proj.weight", .{i}, .bf16, &.{ 1536, 4096 });
            try f.lmAdd("layers.{d}.self_attn.kv_a_proj_with_mqa.weight", .{i}, .bf16, &.{ 512, 4096 });
            try f.lmAdd("layers.{d}.self_attn.q_b_proj.weight", .{i}, .bf16, &.{ 16384, 1536 });
            try f.lmAdd("layers.{d}.self_attn.kv_b_proj.weight", .{i}, .bf16, &.{ 32768, 512 });
            try f.lmAdd("layers.{d}.self_attn.o_proj.weight", .{i}, .bf16, &.{ 4096, 16384 });
            try f.lmAdd("layers.{d}.self_attn.q_a_layernorm.weight", .{i}, .bf16, &.{1536});
            try f.lmAdd("layers.{d}.self_attn.kv_a_layernorm.weight", .{i}, .bf16, &.{512});
            try f.lmAdd("layers.{d}.self_attn.indexer.wk.weight", .{i}, .bf16, &.{ 128, 4096 });
            try f.lmAdd("layers.{d}.self_attn.indexer.weights_proj.weight", .{i}, .bf16, &.{ 32, 4096 });
            try f.lmAdd("layers.{d}.self_attn.indexer.wq_b.weight", .{i}, .bf16, &.{ 4096, 1536 });
            try f.lmAdd("layers.{d}.self_attn.indexer.k_norm.weight", .{i}, .bf16, &.{128});
            try f.lmAdd("layers.{d}.self_attn.indexer.k_norm.bias", .{i}, .bf16, &.{128});
            try f.lmAdd("layers.{d}.self_attn.indexer.index_kpool_compress_gate", .{i}, .bf16, &.{ 128, 4096 });
            try f.lmAdd("layers.{d}.self_attn.indexer.index_kpool_compress_ape", .{i}, .bf16, &.{ 4, 128 });
        }
        if (i < c.dense_layers) {
            inline for (.{ "gate_proj", "up_proj" }) |x| try f.lmAdd("layers.{d}.mlp." ++ x ++ ".weight", .{i}, .bf16, &.{ 12288, 4096 });
            try f.lmAdd("layers.{d}.mlp.down_proj.weight", .{i}, .bf16, &.{ 4096, 12288 });
        } else {
            try f.lmAdd("layers.{d}.mlp.gate.weight", .{i}, .bf16, &.{ c.experts, 4096 });
            try f.lmAdd("layers.{d}.mlp.gate.e_score_correction_bias", .{i}, .f32, &.{c.experts});
            inline for (.{ "gate_proj", "up_proj" }) |x| try f.lmAdd("layers.{d}.mlp.shared_experts." ++ x ++ ".weight", .{i}, .bf16, &.{ 2048, 4096 });
            try f.lmAdd("layers.{d}.mlp.shared_experts.down_proj.weight", .{i}, .bf16, &.{ 4096, 2048 });
            for (0..c.experts) |e| {
                inline for (.{ "gate_proj", "up_proj" }) |x| {
                    try f.lmAdd("layers.{d}.mlp.experts.{d}." ++ x ++ ".trellis", .{ i, e }, .i16, &.{ 256, 128, 64 });
                    try f.lmAdd("layers.{d}.mlp.experts.{d}." ++ x ++ ".suh", .{ i, e }, .f16, &.{4096});
                    try f.lmAdd("layers.{d}.mlp.experts.{d}." ++ x ++ ".svh", .{ i, e }, .f16, &.{2048});
                    try f.lmAdd("layers.{d}.mlp.experts.{d}." ++ x ++ ".mcg", .{ i, e }, .i32, &.{1});
                }
                try f.lmAdd("layers.{d}.mlp.experts.{d}.down_proj.trellis", .{ i, e }, .i16, &.{ 128, 256, 64 });
                try f.lmAdd("layers.{d}.mlp.experts.{d}.down_proj.suh", .{ i, e }, .f16, &.{2048});
                try f.lmAdd("layers.{d}.mlp.experts.{d}.down_proj.svh", .{ i, e }, .f16, &.{4096});
                try f.lmAdd("layers.{d}.mlp.experts.{d}.down_proj.mcg", .{ i, e }, .i32, &.{1});
            }
        }
        if (i == c.layers) {
            inline for (.{ "enorm", "hnorm", "shared_head.norm" }) |x| try f.lmAdd("layers.{d}." ++ x ++ ".weight", .{i}, .bf16, &.{4096});
            try f.lmAdd("layers.{d}.eh_proj.weight", .{i}, .bf16, &.{ 4096, 8192 });
        }
    }
    try f.lmAdd("embed_tokens.weight", .{}, .bf16, &.{ c.vocab, 4096 });
    try f.lmAdd("norm.weight", .{}, .bf16, &.{4096});
    try f.add("lm_head.weight", .bf16, &.{ c.vocab, 4096 });
    try f.add("model.visual.blocks.0.attn.qkv.weight", .bf16, &.{ 3456, 1152 });
    return f.x;
}

fn glm53() config.Config {
    var c: config.Config = .{};
    c.exl3 = true;
    c.head_unit = 128;
    return c;
}

test "glm hip layout: the whole GLM-5.3-Flash checkpoint placed as stored, classes as tensor_classes.py sums them" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = glm53();
    const x = try fakeIndex(a, c);
    var why: []const u8 = "";
    const l = try plan(a, &x, c, &why);
    // the classes' bytes, from the shapes (PLAN.md section 1 rounds them to MiB: 141.750 GiB of trellis and so on)
    const routed: u64 = 42 * 288;
    try std.testing.expectEqual(routed * 3 * 4 << 20, l.on_gpu[@backingInt(Class.routed_trellis)]);
    try std.testing.expectEqual(@as(f64, 141.75), @as(f64, @floatFromInt(l.on_gpu[@backingInt(Class.routed_trellis)])) / (1 << 30));
    try std.testing.expectEqual(routed * 2 * (4096 + 4096 + 2048), l.on_gpu[@backingInt(Class.routed_suh)]);
    try std.testing.expectEqual(routed * 3 * 4, l.stored[@backingInt(Class.routed_mcg)]);
    try std.testing.expectEqual(@as(u64, 0), l.on_gpu[@backingInt(Class.routed_mcg)]);
    try std.testing.expectEqual(@as(u64, 0), l.on_gpu[@backingInt(Class.vision)]);
    try std.testing.expect(l.stored[@backingInt(Class.vision)] > 0);
    try std.testing.expectEqual(@as(usize, 3 * 288 * 43), l.markers.len);
    try std.testing.expectEqual(@as(usize, 3 * 43), l.tables.len);
    try std.testing.expectEqual(@as(usize, 48), l.slabs.len);
    // every byte but the vision tower's and the markers' is on the GPU; the slabs hold them, the tables, the gaps
    var placed: u64 = 0;
    for (l.on_gpu) |n| placed += n;
    try std.testing.expectEqual(l.total_stored - l.stored[@backingInt(Class.vision)] - l.stored[@backingInt(Class.routed_mcg)] - l.stored[@backingInt(Class.mtp_experts)] + l.on_gpu[@backingInt(Class.mtp_experts)], placed);
    try std.testing.expect(l.slabBytes() >= placed + 3 * 43 * 288 * 8);
    try std.testing.expect(l.slabBytes() - placed < 1 << 20);
    // pieces: a stacked projection's tensors follow each other; every piece is 256-byte aligned
    const q = l.find(lm ++ "layers.0.self_attn.q_proj.weight").?;
    const b = l.find(lm ++ "layers.0.self_attn.b_proj.weight").?;
    try std.testing.expectEqual(q.off + (3 * 8192 + 2 * 128) * 4096 * 2, b.off);
    try std.testing.expectEqual(@as(u64, 0), q.off % piece_align);
    const e0 = l.find(lm ++ "layers.3.mlp.experts.0.gate_proj.trellis").?;
    const e287 = l.find(lm ++ "layers.3.mlp.experts.287.gate_proj.trellis").?;
    try std.testing.expectEqual(e0.off + 287 * (4 << 20), e287.off);
    const s5 = l.find(lm ++ "layers.3.mlp.experts.5.down_proj.suh").?;
    const s0 = l.find(lm ++ "layers.3.mlp.experts.0.down_proj.suh").?;
    try std.testing.expectEqual(s0.off + 5 * 2048 * 2, s5.off);
    try std.testing.expectEqual(@as(u16, 45), l.find(lm ++ "layers.45.eh_proj.weight").?.slab);
    try std.testing.expectEqual(@as(u16, 47), l.find("lm_head.weight").?.slab);
    for (l.placed) |p| {
        try std.testing.expect(p.off + p.src.len <= l.slabs[p.slab]);
        try std.testing.expectEqual(@as(u64, 0), p.off % 2);
    }
}

test "glm hip layout: a cast tensor, a missing one or an unknown one is refused by name" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = glm53();
    var why: []const u8 = "";
    var x = try fakeIndex(a, c);
    x.tensors.getPtr(lm ++ "layers.0.self_attn.A_log").?.dtype = .bf16;
    try std.testing.expectError(error.UnexpectedTensor, plan(a, &x, c, &why));
    try std.testing.expect(std.mem.indexOf(u8, why, "A_log") != null);
    x = try fakeIndex(a, c);
    _ = x.tensors.orderedRemove(lm ++ "layers.7.mlp.experts.3.up_proj.svh");
    try std.testing.expectError(error.MissingTensor, plan(a, &x, c, &why));
    try std.testing.expectEqualStrings(lm ++ "layers.7.mlp.experts.3.up_proj.svh", why);
    x = try fakeIndex(a, c);
    try x.tensors.put(a, lm ++ "layers.7.self_attn.extra.weight", x.tensors.get(lm ++ "norm.weight").?);
    try std.testing.expectError(error.UnplacedTensor, plan(a, &x, c, &why));
    try std.testing.expectEqualStrings(lm ++ "layers.7.self_attn.extra.weight", why);
}

test "glm hip layout: classify is tensor_classes.py's rule" {
    const cases = [_]struct { []const u8, Class }{
        .{ "model.visual.blocks.0.attn.qkv.weight", .vision },
        .{ lm ++ "embed_tokens.weight", .embed },
        .{ "lm_head.weight", .lm_head },
        .{ lm ++ "layers.45.mlp.experts.3.up_proj.mcg", .mtp_experts },
        .{ lm ++ "layers.45.input_layernorm.weight", .mtp_other },
        .{ lm ++ "layers.3.mlp.experts.3.up_proj.svh", .routed_svh },
        .{ lm ++ "layers.3.mlp.shared_experts.up_proj.weight", .shared_expert },
        .{ lm ++ "layers.3.mlp.gate.e_score_correction_bias", .router },
        .{ lm ++ "layers.1.mlp.down_proj.weight", .dense_mlp },
        .{ lm ++ "layers.3.self_attn.indexer.k_norm.weight", .dsa_indexer },
        .{ lm ++ "layers.3.hc_attn_fn", .mhc },
        .{ lm ++ "layers.3.self_attn.q_a_layernorm.weight", .norms },
        .{ lm ++ "layers.0.self_attn.o_norm.weight", .norms },
        .{ lm ++ "norm.weight", .norms },
        .{ lm ++ "layers.7.self_attn.kv_b_proj.weight", .dsa_mla },
        .{ lm ++ "layers.6.self_attn.dt_bias", .kda },
    };
    for (cases) |cs| try std.testing.expectEqual(cs[1], classify(cs[0], 45).?);
}
