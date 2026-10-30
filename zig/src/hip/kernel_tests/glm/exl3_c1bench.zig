//! C-1 measurement (code contract 7): the EXL3 WMMA prompt GEMM at decode rows. At GLM's gate/up (K 4,096, N 2,048,
//! two matrices) and down (K 2,048, N 4,096) shapes, eight experts an item each with 1, 2, 4, 8 or 16 member rows: the
//! GEMM's and the decode GEMV's time and effective GB/s (facts), a row's GEMM bits at those rows equal to its bits in a
//! 2,048-row chunk (checked), and how many values the GEMV and the GEMM give differently (facts). A measurement only:
//! no forward change. Runs when TF_GLM_C1BENCH is set (skipped otherwise).
const std = @import("std");
const rig = @import("../rig.zig");
const check = @import("check.zig");
const launch = @import("../../launches/exl3.zig");
const DeviceBuffer = @import("../../memory.zig").DeviceBuffer;

const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;

const experts = 8;
const chunk = 2048; // pairs in the chunk call: 256 an expert, four 64-row passes each
const per_expert = chunk / experts;
const first_off = 37; // the small calls' rows sit inside the chunk's first pass of each expert, not at its start

/// One projection kind's buffers: X fp16 [chunk][k] (random, as the rotated input), each expert's trellis (8 bpw half
/// words: 4 bpw), the u64 tables, the members (identity) and the z buffers of the GEMM (mats planes) and the GEMV
/// (mats x sk planes).
const Shape = struct {
    k: usize,
    n: usize,
    mats: usize,
    sk: usize,
    x: DeviceBuffer,
    tr: [2][experts]DeviceBuffer,
    tp: [2]DeviceBuffer,
    members: DeviceBuffer,
    items: DeviceBuffer,
    zm: DeviceBuffer,
    zv: DeviceBuffer,

    fn init(t: *Rig, rng: *rig.Rng, k: usize, n: usize, mats: usize, gate_up: bool) !Shape {
        var s: Shape = undefined;
        s.k = k;
        s.n = n;
        s.mats = mats;
        s.sk = try launch.split(k, n, gate_up);
        const hx = try gpa.alloc(u16, chunk * k);
        defer gpa.free(hx);
        for (hx) |*v| v.* = rig.f16Bits(@floatCast(rng.normal()));
        s.x = try t.upload(hx);
        const words = try gpa.alloc(u16, k / 16 * (n / 16) * 64);
        defer gpa.free(words);
        for (0..2) |m| {
            var tp: [experts]u64 = undefined;
            for (0..experts) |e| {
                if (m < mats) {
                    for (words) |*w| w.* = @truncate(rng.next());
                    s.tr[m][e] = try t.upload(words);
                } else s.tr[m][e] = s.tr[0][e];
                tp[e] = at(s.tr[m][e]);
            }
            s.tp[m] = try t.upload(&tp);
        }
        const ids = try gpa.alloc(i32, chunk);
        defer gpa.free(ids);
        for (ids, 0..) |*v, i| v.* = @intCast(i);
        s.members = try t.upload(ids);
        s.items = try t.alloc(3 * 4 * experts);
        s.zm = try t.alloc(mats * chunk * n * 4);
        s.zv = try t.alloc(mats * s.sk * chunk * n * 4);
        return s;
    }

    fn deinit(s: *Shape) void {
        for (0..s.mats) |m| for (&s.tr[m]) |*b| b.free();
        for ([_]*DeviceBuffer{ &s.x, &s.tp[0], &s.tp[1], &s.members, &s.items, &s.zm, &s.zv }) |b| b.free();
    }

    /// Items: expert e's `rows` pairs from e * 256 + off.
    fn setItems(s: *Shape, rows: usize, off: usize) !void {
        var it: [3 * experts]i32 = undefined;
        for (0..experts) |e| {
            it[3 * e] = @intCast(e);
            it[3 * e + 1] = @intCast(e * per_expert + off);
            it[3 * e + 2] = @intCast(rows);
        }
        try s.items.upload(0, std.mem.asBytes(&it));
    }

    fn product(s: *const Shape, z: DeviceBuffer) launch.Product {
        return .{ .x0 = at(s.x), .x1 = at(s.x), .x_div = 1, .tp0 = at(s.tp[0]), .tp1 = at(s.tp[1]), .mats = s.mats, .items = at(s.items), .count = experts, .members = at(s.members), .z = at(z), .k = s.k, .n = s.n, .pairs = chunk, .experts = experts };
    }
};

const Call = struct { t: *Rig, s: *const Shape, rows: usize, gemm: bool };

fn go(c: Call) anyerror!void {
    if (c.gemm) {
        try launch.gemm(&c.t.on, c.s.product(c.s.zm), c.t.stream.handle);
    } else {
        try launch.gemv(&c.t.on, c.s.product(c.s.zv), c.s.sk, c.rows, c.t.stream.handle);
    }
}

/// The bf16 of an f32 (nearest even), for the flip count.
fn bf16(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @intCast((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

test "glm exl3 c1 (contract 7): the WMMA GEMM at 1-16 decode rows against the GEMV, GLM gate/up and down; a row's GEMM bits as in a 2,048-row chunk" {
    if (std.c.getenv("TF_GLM_C1BENCH") == null) return error.SkipZigTest;
    const t = try Rig.open(1 << 20);
    defer t.close();
    if (!t.bf16 or @import("../../exl3.zig").matrix() == false) return error.SkipZigTest;
    var rng: rig.Rng = .{ .state = 0xC1BE_0001 };
    const Kind = struct { name: []const u8, k: usize, n: usize, mats: usize, gate_up: bool };
    for ([_]Kind{ .{ .name = "gate/up", .k = 4096, .n = 2048, .mats = 2, .gate_up = true }, .{ .name = "down", .k = 2048, .n = 4096, .mats = 1, .gate_up = false } }) |kd| {
        var s = try Shape.init(t, &rng, kd.k, kd.n, kd.mats, kd.gate_up);
        defer s.deinit();
        const n = kd.n;
        // the chunk: every pair, through the GEMM (the reference bits for each row)
        try s.setItems(per_expert, 0);
        try s.zm.fill8(0xff);
        try go(.{ .t = t, .s = &s, .rows = per_expert, .gemm = true });
        try t.stream.synchronize();
        const full = try rig.download(f32, s.zm);
        defer gpa.free(full);
        const bytes: f64 = @floatFromInt(experts * kd.mats * (kd.k / 16) * (n / 16) * 128);
        for ([_]usize{ 1, 2, 4, 8, 16 }) |rows| {
            try s.setItems(rows, first_off);
            // bits: each small call's rows equal the chunk's rows, in every matrix
            try s.zm.fill8(0xff);
            try go(.{ .t = t, .s = &s, .rows = rows, .gemm = true });
            try t.stream.synchronize();
            const got = try rig.download(f32, s.zm);
            defer gpa.free(got);
            for (0..kd.mats) |m| for (0..experts) |e| for (0..rows) |r| {
                const p = e * per_expert + first_off + r;
                const o = (m * chunk + p) * n;
                const same = std.mem.eql(u32, @ptrCast(got[o..][0..n]), @ptrCast(full[o..][0..n]));
                try rig.expect(same, "exl3 c1 {s}: mat {d} expert {d} row {d} of a {d}-row item differs from the 2,048-row chunk", .{ kd.name, m, e, r, rows });
            };
            // the GEMV at the same rows: its split planes summed in order (the epilogues' order), against the GEMM
            try s.zv.fill8(0xff);
            try go(.{ .t = t, .s = &s, .rows = rows, .gemm = false });
            try t.stream.synchronize();
            const gv = try rig.download(f32, s.zv);
            defer gpa.free(gv);
            var diff: usize = 0;
            var flips: usize = 0;
            var total: usize = 0;
            for (0..kd.mats) |m| for (0..experts) |e| for (0..rows) |r| {
                const p = e * per_expert + first_off + r;
                for (0..n) |c| {
                    var v: f32 = 0;
                    for (0..s.sk) |q| v += gv[((m * s.sk + q) * chunk + p) * n + c];
                    const w = got[(m * chunk + p) * n + c];
                    if (@as(u32, @bitCast(v)) != @as(u32, @bitCast(w))) diff += 1;
                    if (bf16(v) != bf16(w)) flips += 1;
                    total += 1;
                }
            };
            const us_m = try check.timeUs(t, Call{ .t = t, .s = &s, .rows = rows, .gemm = true }, go, 50);
            const us_v = try check.timeUs(t, Call{ .t = t, .s = &s, .rows = rows, .gemm = false }, go, 50);
            const tf: f64 = @floatFromInt(total);
            std.debug.print("FACT glm exl3 c1 {s} K {d} N {d}, {d} experts x {d} rows: GEMM {d:.1} us ({d:.1} GB/s), GEMV sk {d} {d:.1} us ({d:.1} GB/s), GEMM / GEMV {d:.3}; GEMV vs GEMM values: fp32 differ {d:.2} %, bf16 differ {d:.3} %\n", .{ kd.name, kd.k, n, experts, rows, us_m, bytes / us_m / 1e3, s.sk, us_v, bytes / us_v / 1e3, us_m / us_v, 100.0 * @as(f64, @floatFromInt(diff)) / tf, 100.0 * @as(f64, @floatFromInt(flips)) / tf });
        }
        std.debug.print("RESULT glm exl3 c1 {s}: a row's GEMM bits at 1, 2, 4, 8 and 16 rows an item equal its bits in the 2,048-row chunk\n", .{kd.name});
    }
}
