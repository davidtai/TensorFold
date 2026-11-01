//! G4-4: the BF16 prompt tile with row tiles grouped (bf16_wmma_ord), bf16Go's tile for weights of 32 MiB and up. Its
//! bits equal the row-major tile's (F-K5's 32-k groups, the K-order fold) at every K of the model; it keeps a row's bits
//! under row cuts (contract 7); and speed facts (no gate) of both orders at GLM's prompt products, 2,048 and 8,192 rows.
const std = @import("std");
const rig = @import("../rig.zig");
const check = @import("check.zig");
const affine = @import("../../launches/affine.zig");
const DeviceBuffer = @import("../../memory.zig").DeviceBuffer;

const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;
const K = affine.Kernels;

fn skip(t: *Rig) bool {
    return !t.bf16 or t.on.affine.bf16_wmma[0] == null or t.on.affine.bf16_wmma_ord[0] == null;
}

/// Rows lo..lo + m of the product, row tiles grouped or not: fp32 or (half) bf16 words, as u32.
fn product(t: *Rig, x: DeviceBuffer, w: DeviceBuffer, out: DeviceBuffer, lo: usize, m: usize, n: usize, k: usize, half: bool, grouped: bool) ![]u32 {
    const xa = at(x) + lo * k * 2;
    const a = K.bf16Arg(xa, at(w), if (half) 0 else at(out), if (half) at(out) else 0, @intCast(m), @intCast(n), @intCast(k));
    try t.on.affine.bf16Wmma(&t.r, a, grouped, t.stream.handle);
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

test "glm bf16 tile (G4-4): the grouped tile gives the row-major tile's bits at every K of the model, fp32 and bf16 out" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    if (skip(t)) return error.SkipZigTest;
    var rng: rig.Rng = .{ .state = 0x6474_0001 };
    var cases: usize = 0;
    const Shape = struct { m: usize, n: usize, k: usize };
    var shapes: [K.bf16_ks.len + 2]Shape = undefined;
    for (K.bf16_ks, 0..) |kc, i| shapes[i] = .{ .m = 300, .n = 320, .k = @intCast(kc) };
    shapes[K.bf16_ks.len] = .{ .m = 2048, .n = 24896, .k = 4096 }; // KDA's fused input product at a 2,048-row chunk
    shapes[K.bf16_ks.len + 1] = .{ .m = 8192, .n = 4096, .k = 8192 }; // KDA o at the served 8,192-row chunk (8 groups)
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
            const want = try product(t, x, w, out, 0, sh.m, sh.n, sh.k, half, false);
            defer gpa.free(want);
            const got = try product(t, x, w, out, 0, sh.m, sh.n, sh.k, half, true);
            defer gpa.free(got);
            const diff = std.mem.findDiff(u32, want, got);
            if (diff) |i| std.debug.print("glm bf16 grouped tile {d} x [{d}, {d}] {s}: output {d} is 0x{x}, the row-major tile's 0x{x}\n", .{ sh.m, sh.n, sh.k, if (half) "bf16" else "fp32", i, got[i], want[i] });
            try rig.expect(diff == null, "glm bf16 grouped tile [{d}, {d}]: bits differ from the row-major tile's", .{ sh.n, sh.k });
            cases += 1;
        }
    }
    std.debug.print("RESULT glm bf16 tile: {d} cases (every K; 2,048 x [24896, 4096]; 8,192 x [4096, 8192]), grouped bits equal row-major\n", .{cases});
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
        const full = try product(t, x, w, out, 0, rows, n, k, false, true);
        defer gpa.free(full);
        const cuts = [_][2]usize{ .{ 0, 512 }, .{ 512, 512 }, .{ 1536, 512 }, .{ 37, 300 }, .{ 1, 255 }, .{ 255, 2 }, .{ 2000, 48 }, .{ 7, 1 }, .{ 2047, 1 }, .{ 129, 17 } };
        for (cuts) |c| {
            const g = try product(t, x, w, out, c[0], c[1], n, k, false, true);
            defer gpa.free(g);
            try rig.expect(std.mem.eql(u32, g, full[c[0] * n ..][0 .. c[1] * n]), "glm bf16 grouped K {d}: rows {d}..{d} differ from the 2,048-row call", .{ k, c[0], c[0] + c[1] });
        }
    }
    std.debug.print("RESULT glm bf16 grouped tile rows: K 1,536 and 4,096, 10 cuts of a 2,048-row chunk equal its rows\n", .{});
}

const Timed = struct {
    t: *Rig,
    a: affine.Arg,
    grouped: bool,

    fn call(c: Timed) anyerror!void {
        try c.t.on.affine.bf16Wmma(&c.t.r, c.a, c.grouped, c.t.stream.handle);
    }
};

test "glm bf16 tile (G4-4): speed facts (no gate) of both block orders at GLM's prompt products, 2,048 and 8,192 rows" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    if (skip(t)) return error.SkipZigTest;
    const Shape = struct { name: []const u8, n: usize, k: usize };
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
    const max_m = 8192;
    var x = try t.alloc(max_m * 16384 * 2);
    defer x.free();
    try x.fill8(0x3c);
    var w = try t.alloc(32768 * 4096 * 2);
    defer w.free();
    try w.fill8(0x3c);
    var out = try t.alloc(max_m * 32768 * 2);
    defer out.free();
    for ([_]usize{ 2048, max_m }) |m| for (shapes) |s| {
        const flops: f64 = 2.0 * @as(f64, @floatFromInt(m * s.n * s.k));
        const a = K.bf16Arg(at(x), at(w), 0, at(out), @intCast(m), @intCast(s.n), @intCast(s.k));
        const calls: usize = if (m > 2048) 4 else 10;
        const row = try check.timeUs(t, Timed{ .t = t, .a = a, .grouped = false }, Timed.call, calls);
        const grp = try check.timeUs(t, Timed{ .t = t, .a = a, .grouped = true }, Timed.call, calls);
        const served = K.bf16Grouped(@intCast(s.n), @intCast(s.k));
        std.debug.print("SPEED glm bf16 prompt {s} [{d}, {d}] {d} rows: row-major {d:.0} us ({d:.1} TFLOPS), grouped {d:.0} ({d:.1}); bf16Go takes {s}\n", .{ s.name, s.n, s.k, m, row, flops / row / 1e6, grp, flops / grp / 1e6, if (served) "grouped" else "row-major" });
    };
}
