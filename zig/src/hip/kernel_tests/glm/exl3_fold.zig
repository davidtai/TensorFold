//! F-K6: the EXL3 prompt GEMM's 64-k groups folded on the VALU, at GLM's gate/up (K 4,096, N 2,048) and down (K 2,048,
//! N 4,096) shapes: within float64's bound, a row's bits as alone, and (facts) bf16 flips against the one-chain GEMM.
const std = @import("std");
const core = @import("core");
const rig = @import("../rig.zig");
const check = @import("check.zig");
const launch = @import("../../launches/exl3.zig");
const DeviceBuffer = @import("../../memory.zig").DeviceBuffer;

const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;
const fmt = core.exl3_format;

/// The bf16 nearest a double in one rounding (round to odd onto fp32, then nearest even).
fn rn64(v: f64) u16 {
    var f: f32 = @floatCast(v);
    if (@as(f64, f) != v) {
        var u: u32 = @bitCast(f);
        if (@abs(@as(f64, f)) > @abs(v)) u -= 1;
        f = @bitCast(u | 1);
    }
    return check.rnd(f);
}

fn ulp32(v: f64) f64 {
    const f: f32 = @floatCast(@abs(v));
    if (f == 0) return std.math.pow(f64, 2, -149);
    return @as(f64, std.math.nextAfter(f32, f, std.math.inf(f32))) - f;
}

/// One expert's matrix on the device and its plan buffers: X fp16 [M][K], trellis tiles, W_q fp16 [K][N] (decoded on the
/// GPU, the decoder's bits checked by kernel_tests/exl3.zig), z fp32 [M][N].
const Case = struct {
    m: usize,
    k: usize,
    n: usize,
    hx: []f16,
    wq: []u16,
    x: DeviceBuffer,
    tr: DeviceBuffer,
    table: DeviceBuffer,
    items: DeviceBuffer,
    members: DeviceBuffer,
    z: DeviceBuffer,

    fn init(t: *Rig, rng: *rig.Rng, m: usize, k: usize, n: usize) !Case {
        var c: Case = undefined;
        c.m = m;
        c.k = k;
        c.n = n;
        c.hx = try gpa.alloc(f16, m * k);
        for (c.hx) |*v| v.* = @floatCast(rng.normal());
        const words = try gpa.alloc(u16, k / 16 * (n / 16) * 64);
        defer gpa.free(words);
        for (words) |*w| w.* = @truncate(rng.next());
        c.x = try t.upload(c.hx);
        c.tr = try t.upload(words);
        const tp = [1]u64{at(c.tr)};
        c.table = try t.upload(&tp);
        var dq = try t.alloc(k * n * 2);
        defer dq.free();
        try launch.unpack(&t.on, at(c.tr), 8, @intCast(@backingInt(fmt.Codebook.mcg)), k / 16, n / 16, at(dq), t.stream.handle);
        try t.stream.synchronize();
        c.wq = try rig.download(u16, dq);
        c.items = try t.alloc(3 * 4);
        const ids = try gpa.alloc(i32, m);
        defer gpa.free(ids);
        for (ids, 0..) |*v, i| v.* = @intCast(i);
        c.members = try t.upload(ids);
        c.z = try t.alloc(m * n * 4);
        return c;
    }

    fn deinit(c: *Case) void {
        gpa.free(c.hx);
        gpa.free(c.wq);
        for ([_]*DeviceBuffer{ &c.x, &c.tr, &c.table, &c.items, &c.members, &c.z }) |b| b.free();
    }

    /// The item (expert 0, first, count) through the folded GEMM, or with `chain` the one-chain reference: z fp32.
    fn run(c: *Case, t: *Rig, first: usize, count: usize, chain: bool) ![]f32 {
        const it = [3]i32{ 0, @intCast(first), @intCast(count) };
        try c.items.upload(0, std.mem.asBytes(&it));
        try c.z.fill8(0xff);
        const p: launch.Product = .{ .x0 = at(c.x), .x1 = at(c.x), .x_div = 1, .tp0 = at(c.table), .tp1 = at(c.table), .mats = 1, .items = at(c.items), .count = 1, .members = at(c.members), .z = at(c.z), .k = c.k, .n = c.n, .pairs = c.m, .experts = 1 };
        if (chain) try launch.gemmChain(&t.on, p, t.stream.handle) else try launch.gemm(&t.on, p, t.stream.handle);
        try t.stream.synchronize();
        return rig.download(f32, c.z);
    }
};

test "glm exl3 fold (F-K6): the prompt GEMM's 64-k groups within float64's bound at the gate/up and down shapes, rows as alone; flips against one chain (facts)" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    if (!t.bf16 or @import("../../exl3.zig").matrix() == false) return error.SkipZigTest;
    var rng: rig.Rng = .{ .state = 0xF0D6_0001 };
    const m = 128; // two passes of 64 rows
    for ([_][2]usize{ .{ 4096, 2048 }, .{ 2048, 4096 } }) |kn| {
        const k = kn[0];
        const n = kn[1];
        var c = try Case.init(t, &rng, m, k, n);
        defer c.deinit();
        // float64: z = X W_q, and the fp32 bound (K + 8) 2^-24 of sum |x w| a value
        const want = try gpa.alloc(f64, m * n);
        defer gpa.free(want);
        const tol = try gpa.alloc(f64, m * n);
        defer gpa.free(tol);
        const wd = try gpa.alloc(f64, k * n);
        defer gpa.free(wd);
        for (wd, c.wq) |*d, b| d.* = @as(f16, @bitCast(b));
        const bound = @as(f64, @floatFromInt(k + 8)) * std.math.pow(f64, 2, -24);
        const row = try gpa.alloc(f64, n);
        defer gpa.free(row);
        const mag = try gpa.alloc(f64, n);
        defer gpa.free(mag);
        for (0..m) |r| {
            @memset(row, 0);
            @memset(mag, 0);
            for (0..k) |kk| {
                const xv: f64 = c.hx[r * k + kk];
                const wr = wd[kk * n ..][0..n];
                for (row, mag, wr) |*s, *a, w| {
                    s.* += xv * w;
                    a.* += @abs(xv * w);
                }
            }
            for (0..n) |j| {
                want[r * n + j] = row[j];
                tol[r * n + j] = mag[j] * bound + 1e-30;
            }
        }
        var flips: [2]usize = .{ 0, 0 };
        var ulps: [2]f64 = .{ 0, 0 };
        var full: []f32 = undefined;
        for ([_]bool{ false, true }, 0..) |chain, which| {
            const got = try c.run(t, 0, m, chain);
            // the fold's result stays as the cuts' reference (freed after them); the chain's after its checks
            defer if (chain) gpa.free(got);
            if (!chain) full = got;
            var name: [80]u8 = undefined;
            var w: check.Worst = .{ .name = try std.fmt.bufPrint(&name, "exl3 gemm {s} {d} x [K {d}, N {d}]", .{ if (chain) "one chain" else "fold G64", m, k, n }) };
            for (0..m * n) |i| {
                w.add(i, got[i], want[i], tol[i]);
                if (check.rnd(got[i]) != rn64(want[i])) flips[which] += 1;
                ulps[which] += @abs(@as(f64, got[i]) - want[i]) / ulp32(want[i]);
            }
            try w.done();
        }
        defer gpa.free(full);
        const total: f64 = @floatFromInt(m * n);
        std.debug.print("FACT glm exl3 prompt gemm K {d}, N {d}, {d} rows: bf16 flips fold G64 {d:.3} % ({d:.2} ulp), one chain {d:.3} % ({d:.2} ulp)\n", .{ k, n, m, 100.0 * @as(f64, @floatFromInt(flips[0])) / total, ulps[0] / total, 100.0 * @as(f64, @floatFromInt(flips[1])) / total, ulps[1] / total });
        // a row's bits as alone, and a cut that moves rows across the 64-row passes
        for ([_][2]usize{ .{ 0, 1 }, .{ 63, 1 }, .{ 64, 1 }, .{ 127, 1 }, .{ 10, 50 }, .{ 37, 91 } }) |cut| {
            const got = try c.run(t, cut[0], cut[1], false);
            defer gpa.free(got);
            for (cut[0]..cut[0] + cut[1]) |r| {
                const same = std.mem.eql(u32, @ptrCast(got[r * n ..][0..n]), @ptrCast(full[r * n ..][0..n]));
                try rig.expect(same, "exl3 gemm fold K {d}: row {d} of the cut ({d}, {d}) differs from the {d}-row call", .{ k, r, cut[0], cut[1], m });
            }
        }
        std.debug.print("RESULT glm exl3 gemm fold K {d}: 6 cuts (single rows, across the 64-row passes) bit-equal to the full call\n", .{k});
    }
}

const TimedGemm = struct {
    t: *Rig,
    p: launch.Product,
    chain: bool,

    fn call(c: TimedGemm) anyerror!void {
        if (c.chain) try launch.gemmChain(&c.t.on, c.p, c.t.stream.handle) else try launch.gemm(&c.t.on, c.p, c.t.stream.handle);
    }
};

test "glm exl3 fold (G4-T1): speed facts (no gate) of the folded prompt GEMM against one chain, 2,048 rows of one expert" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    if (!t.bf16 or @import("../../exl3.zig").matrix() == false) return error.SkipZigTest;
    var rng: rig.Rng = .{ .state = 0xF0D6_0002 };
    const m = 2048;
    for ([_][2]usize{ .{ 4096, 2048 }, .{ 2048, 4096 } }) |kn| {
        var c = try Case.init(t, &rng, m, kn[0], kn[1]);
        defer c.deinit();
        const it = [3]i32{ 0, 0, m };
        try c.items.upload(0, std.mem.asBytes(&it));
        const p: launch.Product = .{ .x0 = at(c.x), .x1 = at(c.x), .x_div = 1, .tp0 = at(c.table), .tp1 = at(c.table), .mats = 1, .items = at(c.items), .count = 1, .members = at(c.members), .z = at(c.z), .k = c.k, .n = c.n, .pairs = c.m, .experts = 1 };
        const flops: f64 = 2.0 * m * @as(f64, @floatFromInt(kn[0] * kn[1]));
        const fold = try check.timeUs(t, TimedGemm{ .t = t, .p = p, .chain = false }, TimedGemm.call, 10);
        const chain = try check.timeUs(t, TimedGemm{ .t = t, .p = p, .chain = true }, TimedGemm.call, 10);
        std.debug.print("SPEED glm exl3 prompt gemm K {d}, N {d}, 2048 rows: fold G64 {d:.0} us ({d:.1} TFLOPS), one chain {d:.0} us ({d:.1}), fold / chain {d:.3}\n", .{ kn[0], kn[1], fold, flops / fold / 1e6, chain, flops / chain / 1e6, fold / chain });
    }
}
