//! G4-4: the BF16 prompt tile on other block shapes. Each shape's bits equal bf16Go's tile (F-K5's 32-k groups, the
//! K-order fold) at every K of the model; the grouped tile keeps a row's bits under row cuts (contract 7); and speed
//! facts (no gate) of every tile at GLM's prompt products, 2,048 rows, with the pre-F-K5 one-chain tile as reference.
const std = @import("std");
const rig = @import("../rig.zig");
const check = @import("check.zig");
const affine = @import("../../launches/affine.zig");
const DeviceBuffer = @import("../../memory.zig").DeviceBuffer;

const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;
const K = affine.Kernels;
const Tile = K.Bf16Tile;

fn skip(t: *Rig) bool {
    return !t.bf16 or t.on.affine.bf16_wmma[0] == null or t.on.affine.bf16_wmma_alt[0][0] == null;
}

/// Rows lo..lo + m of the product on tile `tile`: fp32 or (half) bf16 words, as u32.
fn product(t: *Rig, x: DeviceBuffer, w: DeviceBuffer, out: DeviceBuffer, lo: usize, m: usize, n: usize, k: usize, half: bool, tile: Tile) ![]u32 {
    const xa = at(x) + lo * k * 2;
    const a = K.bf16Arg(xa, at(w), if (half) 0 else at(out), if (half) at(out) else 0, @intCast(m), @intCast(n), @intCast(k));
    try t.on.affine.bf16GoTile(&t.r, a, tile, t.stream.handle);
    try t.stream.synchronize();
    const got = try gpa.alloc(u32, m * n);
    if (half) {
        const h = try gpa.alloc(u16, m * n);
        defer gpa.free(h);
        try out.download(0, std.mem.sliceAsBytes(h));
        for (got, h) |*g, v| g.* = v;
    } else {
        try out.download(0, std.mem.sliceAsBytes(got));
    }
    return got;
}

test "glm bf16 tile (G4-4): the grouped and two-ahead tiles give bf16Go's bits at every K of the model, fp32 and bf16 out" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    if (skip(t)) return error.SkipZigTest;
    var rng: rig.Rng = .{ .state = 0x6474_0001 };
    var cases: usize = 0;
    const Shape = struct { m: usize, n: usize, k: usize };
    var shapes: [K.bf16_ks.len + 1]Shape = undefined;
    for (K.bf16_ks, 0..) |kc, i| shapes[i] = .{ .m = 300, .n = 320, .k = @intCast(kc) };
    shapes[K.bf16_ks.len] = .{ .m = 2048, .n = 24896, .k = 4096 }; // KDA's fused input product at a full chunk
    for (shapes) |sh| {
        const hx = try check.normals(&rng, sh.m * sh.k, 1.0);
        defer gpa.free(hx);
        const hw = try check.normals(&rng, sh.n * sh.k, 0.05);
        defer gpa.free(hw);
        var x = try t.upload(hx);
        defer x.free();
        var w = try t.upload(hw);
        defer w.free();
        var out = try t.alloc(sh.m * sh.n * 4);
        defer out.free();
        for ([_]bool{ false, true }) |half| {
            const want = try product(t, x, w, out, 0, sh.m, sh.n, sh.k, half, .dense);
            defer gpa.free(want);
            for ([_]Tile{ .grouped, .ahead2 }) |tile| {
                const got = try product(t, x, w, out, 0, sh.m, sh.n, sh.k, half, tile);
                defer gpa.free(got);
                const diff = std.mem.findDiff(u32, want, got);
                if (diff) |i| std.debug.print("glm bf16 tile {t} {d} x [{d}, {d}] {s}: output {d} is 0x{x}, bf16Go's 0x{x}\n", .{ tile, sh.m, sh.n, sh.k, if (half) "bf16" else "fp32", i, got[i], want[i] });
                try rig.expect(diff == null, "glm bf16 tile {t} [{d}, {d}]: bits differ from bf16Go's", .{ tile, sh.n, sh.k });
                cases += 1;
            }
        }
    }
    std.debug.print("RESULT glm bf16 tile: {d} cases (every K, a 2,048 x [24896, 4096] chunk), grouped and two-ahead bits equal bf16Go's\n", .{cases});
}

test "glm bf16 tile (G4-4, contract 7): the grouped tile gives a row the same bits at any m and any place in the chunk" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    if (skip(t)) return error.SkipZigTest;
    var rng: rig.Rng = .{ .state = 0x6474_0002 };
    const rows = 2048;
    const n = 320;
    for ([_]usize{ 1536, 4096 }) |k| {
        const hx = try check.normals(&rng, rows * k, 1.0);
        defer gpa.free(hx);
        const hw = try check.normals(&rng, n * k, 0.05);
        defer gpa.free(hw);
        var x = try t.upload(hx);
        defer x.free();
        var w = try t.upload(hw);
        defer w.free();
        var out = try t.alloc(rows * n * 4);
        defer out.free();
        const full = try product(t, x, w, out, 0, rows, n, k, false, .grouped);
        defer gpa.free(full);
        const cuts = [_][2]usize{ .{ 0, 512 }, .{ 512, 512 }, .{ 1536, 512 }, .{ 37, 300 }, .{ 1, 255 }, .{ 255, 2 }, .{ 2000, 48 }, .{ 7, 1 }, .{ 2047, 1 }, .{ 129, 17 } };
        for (cuts) |c| {
            const g = try product(t, x, w, out, c[0], c[1], n, k, false, .grouped);
            defer gpa.free(g);
            try rig.expect(std.mem.eql(u32, g, full[c[0] * n ..][0 .. c[1] * n]), "glm bf16 grouped K {d}: rows {d}..{d} differ from the 2,048-row call", .{ k, c[0], c[0] + c[1] });
        }
    }
    std.debug.print("RESULT glm bf16 grouped tile rows: K 1,536 and 4,096, 10 cuts of a 2,048-row chunk equal its rows\n", .{});
}

const Timed = struct {
    t: *Rig,
    a: affine.Arg,
    tile: Tile,

    fn call(c: Timed) anyerror!void {
        try c.t.on.affine.bf16GoTile(&c.t.r, c.a, c.tile, c.t.stream.handle);
    }
};

test "glm bf16 tile (G4-4, G4-T1): speed facts (no gate) of every prompt tile at GLM's products, 2,048 rows" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    if (skip(t)) return error.SkipZigTest;
    const Shape = struct { name: []const u8, n: usize, k: usize };
    // the forward's prompt products (lane F's profile at 2,048 rows: blocks = n / 128 x 16)
    const shapes = [_]Shape{
        .{ .name = "KDA in (q|k|v|f_a|g_a|b)", .n = 24896, .k = 4096 },
        .{ .name = "KDA o", .n = 4096, .k = 8192 },
        .{ .name = "MLA o", .n = 4096, .k = 16384 },
        .{ .name = "q_b | indexer wq_b", .n = 20480, .k = 1536 },
        .{ .name = "MLA a-side", .n = 2304, .k = 4096 },
        .{ .name = "shared gate|up", .n = 4096, .k = 4096 },
        .{ .name = "shared down", .n = 4096, .k = 2048 },
        .{ .name = "dense gate|up", .n = 24576, .k = 4096 },
        .{ .name = "dense down", .n = 4096, .k = 12288 },
        .{ .name = "KDA f_b | g_b", .n = 8192, .k = 128 },
        .{ .name = "kv_b", .n = 32768, .k = 512 },
    };
    const m = 2048;
    var x = try t.alloc(m * 16384 * 2);
    defer x.free();
    try x.fill8(0x3c);
    var w = try t.alloc(32768 * 4096 * 2);
    defer w.free();
    try w.fill8(0x3c);
    var out = try t.alloc(m * 32768 * 2);
    defer out.free();
    for (shapes) |s| {
        const flops: f64 = 2.0 * m * @as(f64, @floatFromInt(s.n * s.k));
        const a = K.bf16Arg(at(x), at(w), 0, at(out), m, @intCast(s.n), @intCast(s.k));
        var us: [4]f64 = undefined;
        for ([_]Tile{ .dense, .grouped, .ahead2, .chain }, 0..) |tile, i| us[i] = try check.timeUs(t, Timed{ .t = t, .a = a, .tile = tile }, Timed.call, 10);
        std.debug.print("SPEED glm bf16 prompt {s} [{d}, {d}] 2048 rows: dense {d:.0} us ({d:.1} TFLOPS), grouped {d:.0} ({d:.1}), ahead2 {d:.0} ({d:.1}), one-chain ref {d:.0} ({d:.1})\n", .{ s.name, s.n, s.k, us[0], flops / us[0] / 1e6, us[1], flops / us[1] / 1e6, us[2], flops / us[2] / 1e6, us[3], flops / us[3] / 1e6 });
    }
}
