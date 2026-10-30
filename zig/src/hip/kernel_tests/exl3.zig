//! EXL3 on the GPU against core's decoder and float64 (ports of zig/src/core/exl3_format.zig's oracle tests and
//! zig/tests/cuda/glm_exl3_tests.zig): the trellis decode bit for bit, then GLM's routed experts (decode GEMV and prompt
//! GEMM over tf_moe_route's plan) within the CUDA test's float64 tolerance, a row's bits as alone, a NaN kept to its row.

const std = @import("std");
const core = @import("core");
const rig = @import("rig.zig");
const exl3 = @import("../exl3.zig");
const launch = @import("../launches/exl3.zig");
const DeviceBuffer = @import("../memory.zig").DeviceBuffer;
const Launcher = @import("../launches.zig").Launcher;

const fmt = core.exl3_format;
const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;

fn int(v: usize) c_int {
    return @intCast(v);
}

// ---- the trellis decode ------------------------------------------------------------------------------------------

const Oracle = struct {
    b: []const u8,

    fn u16At(o: Oracle, i: usize) u16 {
        return std.mem.readInt(u16, o.b[i..][0..2], .little);
    }
    fn u32At(o: Oracle, i: usize) u32 {
        return std.mem.readInt(u32, o.b[i..][0..4], .little);
    }
    fn u64At(o: Oracle, i: usize) u64 {
        return std.mem.readInt(u64, o.b[i..][0..8], .little);
    }
};

/// W_q's fp16 bits [16 kt][16 nt] from the GPU decoder.
fn unpackGpu(t: *Rig, trellis: []const u8, halves: usize, codebook: fmt.Codebook, kt: usize, nt: usize) ![]u16 {
    var tr = try t.upload(trellis);
    defer tr.free();
    var out = try t.alloc(kt * nt * 256 * 2);
    defer out.free();
    try out.fill8(0xAB);
    try launch.unpack(&t.on, at(tr), halves, @intCast(@backingInt(codebook)), kt, nt, at(out), t.stream.handle);
    try t.stream.synchronize();
    return rig.download(u16, out);
}

test "EXL3 decode: the GPU trellis decode gives the numpy oracle's states and core's codebook values on every width" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    const bytes = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, "zig/tests/fixtures/exl3_oracle.bin", gpa, .limited(1 << 22));
    defer gpa.free(bytes);
    const o: Oracle = .{ .b = bytes };
    try std.testing.expectEqualSlices(u8, "EXL3ORAC", bytes[0..8]);
    var placement: [256]fmt.Placement = undefined;
    fmt.tilePlacement(&placement);
    const count = o.u64At(8);
    var i: usize = 16;
    var tiles: usize = 0;
    for (0..count) |_| {
        const halves: usize = o.u16At(i);
        const codebook: fmt.Codebook = @fromBackingInt(@intCast(bytes[i + 2]));
        const kt: usize = o.u32At(i + 4);
        const nt: usize = o.u32At(i + 8);
        const words = halves * 8;
        const trellis = bytes[i + 12 ..][0 .. kt * nt * words * 2];
        const states = i + 12 + trellis.len;
        const got = try unpackGpu(t, trellis, halves, codebook, kt, nt);
        defer gpa.free(got);
        for (0..kt) |tk| for (0..nt) |tn| {
            const ti = tk * nt + tn;
            for (0..256) |p| {
                const state = o.u32At(states + (ti * 256 + p) * 4);
                const want: u16 = @bitCast(fmt.codebookValue(codebook, @intCast(state)));
                const v = got[(tk * 16 + placement[p].row) * nt * 16 + tn * 16 + placement[p].col];
                try rig.expect(v == want, "EXL3 decode {d} half-bits, {s}: tile {d} value {d}: {x} for {x}", .{ halves, @tagName(codebook), ti, p, v, want });
            }
            tiles += 1;
        };
        i = states + kt * nt * 256 * 4;
    }
    try std.testing.expectEqual(@as(usize, 27 * 6), tiles);

    // the real mcg 8.00 bpw block (exl3_format.zig's second oracle test): W_q's bits
    i += 4 + @as(usize, o.u32At(i)) * 6;
    const halves: usize = o.u16At(i);
    const kt: usize = o.u32At(i + 2);
    const nt: usize = o.u32At(i + 6);
    const trellis_len: usize = o.u32At(i + 10);
    i += 14;
    const trellis = bytes[i..][0..trellis_len];
    i += trellis_len;
    i += 4 + @as(usize, o.u32At(i)); // suh
    i += 4 + @as(usize, o.u32At(i)); // svh
    const wq_len: usize = o.u32At(i);
    const wq = bytes[i + 4 ..][0..wq_len];
    const got = try unpackGpu(t, trellis, halves, .mcg, kt, nt);
    defer gpa.free(got);
    for (got, 0..) |v, j| try rig.expect(v == std.mem.readInt(u16, wq[j * 2 ..][0..2], .little), "EXL3 decode, real mcg block: value {d}", .{j});

    // GLM's width (4 bpw, mcg) over random words: core's unpack bit for bit
    var rng: rig.Rng = .{ .state = 0xE813 };
    const rkt = 32;
    const rnt = 16;
    const words = try gpa.alloc(i16, rkt * rnt * 64);
    defer gpa.free(words);
    for (words) |*w| w.* = @bitCast(@as(u16, @truncate(rng.next())));
    const want = try gpa.alloc(f16, rkt * rnt * 256);
    defer gpa.free(want);
    fmt.unpack(.{ .halves = 8 }, .mcg, words, rkt, rnt, want);
    const g4 = try unpackGpu(t, std.mem.sliceAsBytes(words), 8, .mcg, rkt, rnt);
    defer gpa.free(g4);
    for (g4, want, 0..) |v, w, j| try rig.expect(v == @as(u16, @bitCast(w)), "EXL3 decode, random 4-bit mcg: value {d}", .{j});
    std.debug.print("RESULT EXL3 decode: {d} oracle tiles on 11 widths and 3 codebooks, the real 8-bit mcg block and {d} random 4-bit mcg tiles bit for bit\n", .{ tiles, rkt * rnt });
}

// ---- the routed experts -----------------------------------------------------------------------------------------

const Shape = struct { experts: usize, dims: usize, width: usize };

const limit: f32 = 7.0;
const top_k = 8;
const slots = top_k + 1;

/// The three matrices of an expert: gate and up [D][I], down [I][D] (K = input, N = output).
fn dimsOf(sh: Shape, mat: usize) [2]usize {
    return if (mat < 2) .{ sh.dims, sh.width } else .{ sh.width, sh.dims };
}

/// A layer's experts on the host, one slice an expert: trellis int16 [K/16][N/16][64], suh fp16 [K], svh fp16 [N].
/// Synthetic sets own their data in `arena`; a checkpoint's slices point into its mapped shards.
const Host = struct {
    shape: Shape,
    tr: [3][]const []const i16,
    suh: [3][]const []const f16,
    svh: [3][]const []const f16,
    arena: std.heap.ArenaAllocator,

    fn deinit(h: *Host) void {
        h.arena.deinit();
    }

    /// Every gate and up input rotation equal to the first (cuda_exl3.zig load's check): a row rotates once.
    fn sharedInput(h: *const Host) bool {
        const first = h.suh[0][0];
        for (0..2) |m| for (h.suh[m]) |s| if (!std.mem.eql(u8, std.mem.sliceAsBytes(s), std.mem.sliceAsBytes(first))) return false;
        return true;
    }
};

/// The layer's experts on the device in exl3.Experts' layout (cuda_exl3.zig load's): per matrix one trellis buffer and
/// a pointer table, suh and svh stacked.
const Dev = struct {
    e: exl3.Experts,
    bufs: [12]DeviceBuffer = undefined,
    used: usize = 0,

    fn init(t: *Rig, h: *const Host) !Dev {
        var d: Dev = .{ .e = undefined };
        errdefer d.deinit();
        const E = h.shape.experts;
        var ptrs: [3]u64 = undefined;
        var suhs: [3]u64 = undefined;
        var svhs: [3]u64 = undefined;
        for (0..3) |m| {
            const kn = dimsOf(h.shape, m);
            const words = kn[0] / 16 * (kn[1] / 16) * 64;
            const tr = try d.keep(try t.alloc(E * words * 2));
            const su = try d.keep(try t.alloc(E * kn[0] * 2));
            const sv = try d.keep(try t.alloc(E * kn[1] * 2));
            for (0..E) |e| {
                try d.bufs[d.used - 3].upload(e * words * 2, std.mem.sliceAsBytes(h.tr[m][e]));
                try d.bufs[d.used - 2].upload(e * kn[0] * 2, std.mem.sliceAsBytes(h.suh[m][e]));
                try d.bufs[d.used - 1].upload(e * kn[1] * 2, std.mem.sliceAsBytes(h.svh[m][e]));
            }
            const table = try gpa.alloc(u64, E);
            defer gpa.free(table);
            for (table, 0..) |*p, e| p.* = tr + e * words * 2;
            ptrs[m] = try d.keep(try t.upload(table));
            suhs[m] = su;
            svhs[m] = sv;
        }
        d.e = .{ .gate = ptrs[0], .up = ptrs[1], .down = ptrs[2], .suh_g = suhs[0], .suh_u = suhs[1], .svh_g = svhs[0], .svh_u = svhs[1], .suh_d = suhs[2], .svh_d = svhs[2], .count = E, .dims = h.shape.dims, .width = h.shape.width, .shared_input = h.sharedInput(), .limit = limit };
        return d;
    }

    fn keep(d: *Dev, b: DeviceBuffer) !u64 {
        d.bufs[d.used] = b;
        d.used += 1;
        return at(b);
    }

    fn deinit(d: *Dev) void {
        for (d.bufs[0..d.used]) |*b| b.free();
        d.used = 0;
    }
};

fn signs(rng: *rig.Rng, out: []f16, scale: f16) void {
    for (out) |*v| v.* = if (rng.next() & 1 == 1) scale else -scale;
}

/// Random 4-bit mcg experts: every gate and up share one input rotation when `one`.
fn synthetic(rng: *rig.Rng, sh: Shape, one: bool) !Host {
    var h: Host = .{ .shape = sh, .tr = undefined, .suh = undefined, .svh = undefined, .arena = .init(gpa) };
    errdefer h.deinit();
    const a = h.arena.allocator();
    const rot = try a.alloc(f16, sh.dims);
    signs(rng, rot, 1.0);
    for (0..3) |m| {
        const kn = dimsOf(sh, m);
        const tr = try a.alloc([]const i16, sh.experts);
        const su = try a.alloc([]const f16, sh.experts);
        const sv = try a.alloc([]const f16, sh.experts);
        for (0..sh.experts) |e| {
            const w = try a.alloc(i16, kn[0] / 16 * (kn[1] / 16) * 64);
            for (w) |*v| v.* = @bitCast(@as(u16, @truncate(rng.next())));
            tr[e] = w;
            if (one and m < 2) su[e] = rot else {
                const s = try a.alloc(f16, kn[0]);
                signs(rng, s, 1.0);
                su[e] = s;
            }
            const v = try a.alloc(f16, kn[1]);
            signs(rng, v, 0.02);
            sv[e] = v;
        }
        h.tr[m] = tr;
        h.suh[m] = su;
        h.svh[m] = sv;
    }
    return h;
}

/// A checkpoint's shards, mapped and indexed by tensor name through model.safetensors.index.json.
const Checkpoint = struct {
    dir: []const u8,
    index: std.json.Parsed(std.json.Value),
    files: std.StringHashMapUnmanaged(core.safetensors.File) = .empty,

    fn open(dir: []const u8) !Checkpoint {
        const path = try std.fs.path.join(gpa, &.{ dir, "model.safetensors.index.json" });
        defer gpa.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, gpa, .limited(1 << 28));
        defer gpa.free(text);
        return .{ .dir = dir, .index = try std.json.parseFromSlice(std.json.Value, gpa, text, .{}) };
    }

    fn close(c: *Checkpoint) void {
        var it = c.files.iterator();
        while (it.next()) |kv| {
            kv.value_ptr.release();
            kv.value_ptr.close(std.testing.io);
        }
        c.files.deinit(gpa);
        c.index.deinit();
    }

    fn get(c: *Checkpoint, name: []const u8) !core.safetensors.Tensor {
        const map = c.index.value.object.get("weight_map") orelse return error.NoWeightMap;
        const shard = (map.object.get(name) orelse return error.NoTensor).string;
        const f = try c.files.getOrPut(gpa, shard);
        if (!f.found_existing) {
            const path = try std.fs.path.join(gpa, &.{ c.dir, shard });
            defer gpa.free(path);
            f.value_ptr.* = core.safetensors.File.open(gpa, std.testing.io, path) catch |err| {
                _ = c.files.remove(shard);
                return err;
            };
        }
        return f.value_ptr.get(name) orelse error.NoTensor;
    }
};

/// A tensor's bytes as T, in place when aligned, else copied into `a`.
fn as(comptime T: type, a: std.mem.Allocator, bytes: []const u8) ![]const T {
    if (@intFromPtr(bytes.ptr) % @alignOf(T) == 0) return @alignCast(std.mem.bytesAsSlice(T, bytes));
    const out = try a.alloc(T, bytes.len / @sizeOf(T));
    @memcpy(std.mem.sliceAsBytes(out), bytes);
    return out;
}

/// Layer `layer`'s experts `ids` of a GLM-5.3-Flash EXL3 checkpoint, with cuda_exl3.zig load's checks (shapes, dtypes,
/// the mcg marker): local expert i is checkpoint expert ids[i].
fn real(c: *Checkpoint, layer: usize, ids: []const usize, sh: Shape) !Host {
    var h: Host = .{ .shape = sh, .tr = undefined, .suh = undefined, .svh = undefined, .arena = .init(gpa) };
    errdefer h.deinit();
    const a = h.arena.allocator();
    const names = [3][]const u8{ "gate_proj", "up_proj", "down_proj" };
    for (0..3) |m| {
        const kn = dimsOf(sh, m);
        const tr = try a.alloc([]const i16, ids.len);
        const su = try a.alloc([]const f16, ids.len);
        const sv = try a.alloc([]const f16, ids.len);
        for (ids, 0..) |id, i| {
            var buf: [160]u8 = undefined;
            const base = try std.fmt.bufPrint(&buf, "model.language_model.layers.{d}.mlp.experts.{d}.{s}.", .{ layer, id, names[m] });
            var nb: [192]u8 = undefined;
            const t = try c.get(try std.fmt.bufPrint(&nb, "{s}trellis", .{base}));
            try rig.expect(t.is(.i16, &.{ kn[0] / 16, kn[1] / 16, 64 }), "EXL3 real: {s}trellis is not int16 [{d}, {d}, 64]", .{ base, kn[0] / 16, kn[1] / 16 });
            const mk = try c.get(try std.fmt.bufPrint(&nb, "{s}mcg", .{base}));
            try rig.expect(mk.dtype == .i32 and mk.numel() == 1 and std.mem.readInt(u32, mk.bytes[0..4], .little) == fmt.McgMul, "EXL3 real: {s}mcg is not the mcg marker", .{base});
            const s = try c.get(try std.fmt.bufPrint(&nb, "{s}suh", .{base}));
            try rig.expect(s.is(.f16, &.{kn[0]}), "EXL3 real: {s}suh is not fp16 [{d}] (the input side)", .{ base, kn[0] });
            const v = try c.get(try std.fmt.bufPrint(&nb, "{s}svh", .{base}));
            try rig.expect(v.is(.f16, &.{kn[1]}), "EXL3 real: {s}svh is not fp16 [{d}] (the output side)", .{ base, kn[1] });
            tr[i] = try as(i16, a, t.bytes);
            su[i] = try as(f16, a, s.bytes);
            sv[i] = try as(f16, a, v.bytes);
        }
        h.tr[m] = tr;
        h.suh[m] = su;
        h.svh[m] = sv;
    }
    return h;
}

/// The rows' (columns') 128-blocks of a row-major [rows][cols] float64 matrix through H / sqrt(128): a butterfly in
/// float64, the same matrix as exl3_format.zig's rotate (checked against dequantize by `anchor`).
fn rotateFast(x: []f64, rows: usize, cols: usize, axis: usize) void {
    const inv = 1.0 / @sqrt(128.0);
    if (axis == 0) {
        var b: usize = 0;
        while (b < rows) : (b += 128) {
            var h: usize = 1;
            while (h < 128) : (h *= 2) {
                var i: usize = 0;
                while (i < 128) : (i += 2 * h) for (i..i + h) |r| {
                    const p = x[(b + r) * cols ..][0..cols];
                    const q = x[(b + r + h) * cols ..][0..cols];
                    for (p, q) |*u, *v| {
                        const s = u.*;
                        u.* = s + v.*;
                        v.* = s - v.*;
                    }
                };
            }
        }
    } else for (0..rows) |r| {
        var b: usize = 0;
        while (b < cols) : (b += 128) {
            const v = x[r * cols + b ..][0..128];
            var h: usize = 1;
            while (h < 128) : (h *= 2) {
                var i: usize = 0;
                while (i < 128) : (i += 2 * h) for (i..i + h) |j| {
                    const s = v[j];
                    v[j] = s + v[j + h];
                    v[j + h] = s - v[j + h];
                };
            }
        }
    }
    for (x) |*v| v.* *= inv;
}

/// W = diag(suh) H W_q H diag(svh) [K][N] in float64 with the butterfly rotation (wq: K x N fp16 scratch).
fn dequantFast(trellis: []const i16, k: usize, n: usize, suh: []const f16, svh: []const f16, out: []f64, wq: []f16) void {
    fmt.unpack(.{ .halves = 8 }, .mcg, trellis, k / 16, n / 16, wq[0 .. k * n]);
    for (out[0 .. k * n], wq[0 .. k * n]) |*d, s| d.* = @floatCast(s);
    rotateFast(out[0 .. k * n], k, n, 0);
    for (out[0 .. k * n], 0..) |*v, i| v.* *= @floatCast(suh[i / n]);
    rotateFast(out[0 .. k * n], k, n, 1);
    for (out[0 .. k * n], 0..) |*v, i| v.* *= @floatCast(svh[i % n]);
}

/// The float64 reference's matrices against exl3_format.zig's dequantize (the oracle): expert e's three matrices,
/// whole (`rows` 0) or their first `rows` input rows.
fn anchor(h: *const Host, e: usize, rows: usize) !void {
    for (0..3) |m| {
        const kn = dimsOf(h.shape, m);
        const k = if (rows == 0) kn[0] else rows;
        const n = kn[1];
        const sub = h.tr[m][e][0 .. k / 16 * (n / 16) * 64];
        const want = try gpa.alloc(f64, k * n);
        defer gpa.free(want);
        const scratch = try gpa.alloc(f64, 2 * k * n);
        defer gpa.free(scratch);
        fmt.dequantize(.{ .halves = 8 }, .mcg, sub, k / 16, n / 16, h.suh[m][e][0..k], h.svh[m][e], want, scratch);
        const fast = try gpa.alloc(f64, k * n);
        defer gpa.free(fast);
        const wq = try gpa.alloc(f16, k * n);
        defer gpa.free(wq);
        dequantFast(sub, k, n, h.suh[m][e][0..k], h.svh[m][e], fast, wq);
        var most: f64 = 0;
        var mag: f64 = 0;
        for (fast, want) |f, w| {
            most = @max(most, @abs(f - w));
            mag = @max(mag, @abs(w));
        }
        try rig.expect(most <= 1e-12 * (1 + mag), "EXL3 reference: the butterfly equals dequantize (matrix {d}: {e} of {e})", .{ m, most, mag });
    }
}

/// Each row's 8 distinct routed picks, then the shared expert's id E.
fn picks(rng: *rig.Rng, rows: usize, experts: usize) ![]i32 {
    const out = try gpa.alloc(i32, rows * slots);
    for (0..rows) |row| {
        var seen: [512]bool = @splat(false);
        for (0..top_k) |j| {
            var e: usize = @intCast(rng.next() % experts);
            while (seen[e]) e = @intCast(rng.next() % experts);
            seen[e] = true;
            out[row * slots + j] = @intCast(e);
        }
        out[row * slots + top_k] = @intCast(experts);
    }
    return out;
}

fn inputs(rng: *rig.Rng, n: usize) ![]u16 {
    const x = try gpa.alloc(u16, n);
    for (x) |*v| v.* = rig.bf16Bits(rng.unit());
    return x;
}

/// The routed experts on `x` (bf16 [rows][D]) and `pk` through tf_moe_route's plan at `tile` pairs an item:
/// fp32 [rows][slots][D], the shared slot as filled (0xff bytes). The caller frees.
fn run(t: *Rig, d: *const Dev, x: []const u16, pk: []const i32, rows: usize, tile: usize, prompt: bool) ![]f32 {
    const D = d.e.dims;
    const E = d.e.count;
    const pairs = rows * slots;
    var o = t.ops(&t.on);
    o.prefill = prompt;
    o.arena.reset();
    var xb = try t.upload(x);
    defer xb.free();
    var pb = try t.upload(pk);
    defer pb.free();
    var y = try t.alloc(pairs * D * 4);
    defer y.free();
    try y.fill8(0xff);
    const cap = @min(pairs, E + 1) + pairs / tile;
    const members = try o.arena.of(i32, pairs);
    const items = try o.arena.of(i32, 3 * cap);
    const scratch: u64 = if (pairs >= 4 * Launcher.route_seg) try o.arena.take(Launcher.routeScratch(pairs, E + 1)) else 0;
    try t.on.tf_moe_route_wide(@ptrFromInt(at(pb)), int(pairs), int(E + 1), int(tile), @ptrFromInt(members), @ptrFromInt(items), int(cap), scratch, t.stream.handle);
    try exl3.run(o, d.e, at(xb), D, .{ .pick = at(pb), .items = items, .count = cap, .members = members, .tile = tile }, rows, slots, prompt, at(y));
    try t.stream.synchronize();
    return rig.download(f32, y);
}

fn bf16Round(v: f64) f64 {
    const f: f32 = @floatCast(v);
    if (std.math.isNan(f)) return f;
    const u: u32 = @bitCast(f);
    const r: u32 = (u +% 0x7fff +% ((u >> 16) & 1)) & 0xffff0000;
    return @as(f32, @bitCast(r));
}

/// The kernels' SwiGLU: gate and up rounded to bf16 and limited, silu to bf16, the product to bf16.
fn act(g: f64, u: f64) f64 {
    const gg = @min(bf16Round(g), limit);
    const uu = @max(@min(bf16Round(u), limit), -limit);
    return bf16Round(bf16Round(gg / (1 + @exp(-gg))) * uu);
}

/// The listed rows' routed slots within the CUDA test's tolerance (|a w| sums / 64 + 1e-6: fp16 rotated inputs, bf16
/// activations, fp16 down inputs), their shared slot untouched; returns the worst difference over its tolerance. The
/// float64 matrices come an expert at a time, for the experts the rows pick.
fn expect(name: []const u8, h: *const Host, x: []const u16, pk: []const i32, which: []const usize, got: []const f32) !f64 {
    const sh = h.shape;
    const D = sh.dims;
    const N = sh.width;
    var w: [3][]f64 = undefined;
    for (&w) |*m| m.* = try gpa.alloc(f64, D * N);
    defer for (w) |m| gpa.free(m);
    const wq = try gpa.alloc(f16, D * N);
    defer gpa.free(wq);
    const g = try gpa.alloc(f64, 2 * N);
    defer gpa.free(g);
    const a = try gpa.alloc(f64, N);
    defer gpa.free(a);
    const y = try gpa.alloc(f64, 2 * D);
    defer gpa.free(y);
    const xv = try gpa.alloc(f64, D);
    defer gpa.free(xv);
    var most: f64 = 0;
    var where: usize = 0;
    for (which) |row| for (got[(row * slots + top_k) * D ..][0..D]) |v|
        try rig.expect(@as(u32, @bitCast(v)) == 0xffffffff, "{s}: the shared slot left alone", .{name});
    for (0..sh.experts) |e| {
        var used = false;
        for (which) |row| for (0..top_k) |j| {
            used = used or pk[row * slots + j] == @as(i32, @intCast(e));
        };
        if (!used) continue;
        for (0..3) |m| {
            const kn = dimsOf(sh, m);
            dequantFast(h.tr[m][e], kn[0], kn[1], h.suh[m][e], h.svh[m][e], w[m], wq);
        }
        for (which) |row| for (0..top_k) |j| {
            if (pk[row * slots + j] != @as(i32, @intCast(e))) continue;
            for (xv, 0..) |*v, kk| v.* = rig.bf16Value(x[row * D + kk]);
            @memset(g, 0);
            for (0..D) |kk| {
                for (g[0..N], g[N..], w[0][kk * N ..][0..N], w[1][kk * N ..][0..N]) |*gg, *uu, w0, w1| {
                    gg.* += xv[kk] * w0;
                    uu.* += xv[kk] * w1;
                }
            }
            for (a, 0..) |*v, n| v.* = act(g[n], g[N + n]);
            @memset(y, 0);
            for (0..N) |kk| {
                for (y[0..D], y[D..], w[2][kk * D ..][0..D]) |*yy, *mag, wd| {
                    yy.* += a[kk] * wd;
                    mag.* += @abs(a[kk] * wd);
                }
            }
            const base = (row * slots + j) * D;
            for (0..D) |c| {
                const tol = y[D + c] / 64.0 + 1e-6;
                const q = @abs(@as(f64, got[base + c]) - y[c]) / tol;
                if (!(q <= most)) {
                    most = q;
                    where = base + c;
                }
            }
        };
    }
    std.debug.print("RESULT {s} ({d} rows checked): largest difference {d:.3} of its tolerance (element {d})\n", .{ name, which.len, most, where });
    try rig.expect(most <= 1, "{s} within its float64 tolerance", .{name});
    return most;
}

/// Each listed row run alone (its picks only, tile `tile`) keeps its routed slots' bits.
fn alone(t: *Rig, name: []const u8, d: *const Dev, x: []const u16, pk: []const i32, all: []const f32, which: []const usize, tile: usize, prompt: bool) !void {
    const D = d.e.dims;
    for (which) |row| {
        const mine = try run(t, d, x[row * D ..][0..D], pk[row * slots ..][0..slots], 1, tile, prompt);
        defer gpa.free(mine);
        const want = std.mem.sliceAsBytes(all[row * slots * D ..][0 .. top_k * D]);
        try rig.expect(std.mem.eql(u8, std.mem.sliceAsBytes(mine[0 .. top_k * D]), want), "{s}: row {d} equals the row alone", .{ name, row });
    }
}

/// The plan's tile tf_moe_route takes for these rows (moe.zig tileFor): 8 below 64 rows, 128 from 64.
fn tileFor(rows: usize) usize {
    return if (rows < 64) 8 else 128;
}

fn every(buf: []usize, rows: usize) []const usize {
    for (buf[0..rows], 0..) |*r, i| r.* = i;
    return buf[0..rows];
}

/// One case: rows routed at `tile`, checked against float64 on `which` (every row when null), those rows as alone.
fn case(t: *Rig, rng: *rig.Rng, name: []const u8, h: *const Host, d: *const Dev, rows: usize, tile: usize, prompt: bool, which: ?[]const usize) !f64 {
    const x = try inputs(rng, rows * h.shape.dims);
    defer gpa.free(x);
    const pk = try picks(rng, rows, h.shape.experts);
    defer gpa.free(pk);
    const got = try run(t, d, x, pk, rows, tile, prompt);
    defer gpa.free(got);
    for (got, 0..) |v, i| if (i / h.shape.dims % slots != top_k) try rig.expect(std.math.isFinite(v), "{s}: element {d} finite", .{ name, i });
    var buf: [64]usize = undefined;
    const rows_checked = which orelse every(&buf, rows);
    const worst = try expect(name, h, x, pk, rows_checked, got);
    if (rows > 1) try alone(t, name, d, x, pk, got, rows_checked, 8, prompt);
    return worst;
}

const small: Shape = .{ .experts = 12, .dims = 512, .width = 256 };
const glm_shape: Shape = .{ .experts = 288, .dims = 4096, .width = 2048 };

test "EXL3 routed experts: decode GEMV and prompt GEMM over the plan within float64 tolerance, rows as alone" {
    const t = try Rig.open(64 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 7 };
    var worst: f64 = 0;
    for ([_]bool{ true, false }) |one| {
        var h = try synthetic(&rng, small, one);
        defer h.deinit();
        try anchor(&h, 0, 128);
        var d = try Dev.init(t, &h);
        defer d.deinit();
        try rig.expect(d.e.shared_input == one, "EXL3: shared_input as built", .{});
        const how = if (one) "one rotation" else "a rotation an expert";
        var name_buf: [96]u8 = undefined;
        // decode windows (the GEMV), the one-row plan's tile 1 too; 40 rows at tile 128 take several row passes
        for ([_][2]usize{ .{ 1, 1 }, .{ 1, 8 }, .{ 3, 8 }, .{ 8, 8 }, .{ 40, 128 } }) |c| {
            const name = try std.fmt.bufPrint(&name_buf, "EXL3 decode, {s}, tile {d}", .{ how, c[1] });
            worst = @max(worst, try case(t, &rng, name, &h, &d, c[0], c[1], false, null));
        }
        // prompt rows (the WMMA GEMM on gfx11), the last past tf_moe_route_wide's 2,048 pairs
        for ([_]usize{ 1, 40, 200, 520 }) |rows| {
            const name = try std.fmt.bufPrint(&name_buf, "EXL3 prompt, {s}", .{how});
            const which = [_]usize{ 0, rows / 2, rows - 1 };
            worst = @max(worst, try case(t, &rng, name, &h, &d, rows, tileFor(rows), true, if (rows > 40) &which else null));
        }
    }
    std.debug.print("RESULT EXL3 routed experts: decode and prompt slot rows within {d:.3} of their float64 tolerance, rows as alone\n", .{worst});
}

test "EXL3 routed experts: a NaN in one row's input stays in that row's routed slots" {
    const t = try Rig.open(16 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x4A4E };
    var h = try synthetic(&rng, small, true);
    defer h.deinit();
    var d = try Dev.init(t, &h);
    defer d.deinit();
    const rows = 4;
    const x = try inputs(&rng, rows * small.dims);
    defer gpa.free(x);
    x[2 * small.dims + 17] = 0x7fc0;
    const pk = try picks(&rng, rows, small.experts);
    defer gpa.free(pk);
    for ([_]bool{ false, true }) |prompt| {
        const got = try run(t, &d, x, pk, rows, 8, prompt);
        defer gpa.free(got);
        for (0..rows) |row| for (0..top_k) |j| for (0..small.dims) |c| {
            const v = got[(row * slots + j) * small.dims + c];
            try rig.expect(if (row == 2) std.math.isNan(v) else std.math.isFinite(v), "EXL3 {s}: a NaN stays in its row ({d})", .{ if (prompt) "prompt" else "decode", row });
        };
    }
}

test "EXL3 routed experts at GLM-5.3-Flash's shape (D 4096, I 2048): decode and prompt within float64 tolerance" {
    if (heavySkipped()) return error.SkipZigTest;
    const t = try Rig.open(64 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x6C4D };
    const sh: Shape = .{ .experts = 9, .dims = 4096, .width = 2048 };
    var worst: f64 = 0;
    for ([_]bool{ true, false }) |one| {
        var h = try synthetic(&rng, sh, one);
        defer h.deinit();
        var d = try Dev.init(t, &h);
        defer d.deinit();
        const how = if (one) "one rotation" else "a rotation an expert";
        var name_buf: [96]u8 = undefined;
        for ([_][3]usize{ .{ 1, 1, 0 }, .{ 4, 8, 0 }, .{ 24, 8, 1 } }) |c| {
            const name = try std.fmt.bufPrint(&name_buf, "EXL3 GLM shape, {s}, {s}", .{ if (c[2] == 1) "prompt" else "decode", how });
            const which = [_]usize{ 0, c[0] - 1 };
            worst = @max(worst, try case(t, &rng, name, &h, &d, c[0], c[1], c[2] == 1, if (c[0] > 4) &which else null));
        }
    }
    std.debug.print("RESULT EXL3 GLM shape: within {d:.3} of the float64 tolerance\n", .{worst});
}

/// TF_GLM_DIR: a GLM-5.3-Flash EXL3 4 bpw checkpoint (Mia-AiLab's); the real-weight tests are skipped when unset, and in
/// a Debug build (hip-kernel-test), where their float64 references take many minutes (hip-glm-kernel-test is ReleaseSafe).
fn checkpointDir() ?[]const u8 {
    if (heavySkipped()) return null;
    const p = std.c.getenv("TF_GLM_DIR") orelse return null;
    return std.mem.span(p);
}

/// The GLM-shape cases' float64 references (D 4,096 x I 2,048) run only in an optimized build.
fn heavySkipped() bool {
    return @import("builtin").mode == .debug;
}

test "EXL3 real weights: 9 of layer 3's experts, the oracle's dequantization, decode GEMV and prompt GEMM" {
    const dir = checkpointDir() orelse return error.SkipZigTest;
    const t = try Rig.open(256 << 20);
    defer t.close();
    var c = try Checkpoint.open(dir);
    defer c.close();
    const ids = [_]usize{ 0, 1, 37, 100, 143, 200, 251, 286, 287 };
    var h = try real(&c, 3, &ids, .{ .experts = ids.len, .dims = 4096, .width = 2048 });
    defer h.deinit();
    // the float64 matrices against exl3_format.zig's dequantize, whole, on two experts
    try anchor(&h, 0, 0);
    try anchor(&h, ids.len - 1, 0);
    var d = try Dev.init(t, &h);
    defer d.deinit();
    std.debug.print("RESULT EXL3 real layer 3: trellis gate/up [256, 128, 64], down [128, 256, 64], suh on the input side, mcg markers; one input rotation for every gate and up: {}\n", .{d.e.shared_input});
    var rng: rig.Rng = .{ .state = 0x3EA1 };
    var worst: f64 = 0;
    for ([_][3]usize{ .{ 1, 1, 0 }, .{ 4, 8, 0 }, .{ 8, 8, 0 }, .{ 24, 8, 1 }, .{ 64, 128, 1 } }) |cs| {
        var name_buf: [96]u8 = undefined;
        const name = try std.fmt.bufPrint(&name_buf, "EXL3 real layer 3 (9 experts), {s}", .{if (cs[2] == 1) "prompt" else "decode"});
        const which = [_]usize{ 0, 1, cs[0] / 2, cs[0] - 1 };
        worst = @max(worst, try case(t, &rng, name, &h, &d, cs[0], cs[1], cs[2] == 1, if (cs[0] > 8) &which else null));
    }
    std.debug.print("RESULT EXL3 real layer 3 (9 experts): within {d:.3} of the float64 tolerance\n", .{worst});
}

test "EXL3 real weights: all 288 of layer 3's experts, top-8 + shared, 1 / 4 / 8 rows and a 2,048-row prompt chunk" {
    const dir = checkpointDir() orelse return error.SkipZigTest;
    const t = try Rig.open(1 << 30);
    defer t.close();
    var c = try Checkpoint.open(dir);
    defer c.close();
    var ids: [288]usize = undefined;
    for (&ids, 0..) |*v, i| v.* = i;
    var h = try real(&c, 3, &ids, glm_shape);
    defer h.deinit();
    var d = try Dev.init(t, &h);
    defer d.deinit();
    var rng: rig.Rng = .{ .state = 0x288 };
    var worst: f64 = 0;
    var name_buf: [96]u8 = undefined;
    for ([_][2]usize{ .{ 1, 1 }, .{ 4, 8 }, .{ 8, 8 } }) |cs| {
        const name = try std.fmt.bufPrint(&name_buf, "EXL3 real layer 3 (288 experts), decode tile {d}", .{cs[1]});
        worst = @max(worst, try case(t, &rng, name, &h, &d, cs[0], cs[1], false, null));
    }
    // a prompt chunk: 18,432 pairs over the three-launch plan, 16 rows checked (the item and pass edges among them)
    var which = [_]usize{ 0, 1, 63, 64, 127, 128, 1023, 1024, 2046, 2047, 0, 0, 0, 0, 0, 0 };
    for (which[10..]) |*r| r.* = 2 + @as(usize, @intCast(rng.next() % 2040));
    const name = "EXL3 real layer 3 (288 experts), prompt 2,048 rows";
    worst = @max(worst, try case(t, &rng, name, &h, &d, 2048, tileFor(2048), true, &which));
    std.debug.print("RESULT EXL3 real layer 3 (288 experts): within {d:.3} of the float64 tolerance\n", .{worst});
}
