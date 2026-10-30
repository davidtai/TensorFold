//! The EXL3 prompt epilogues (lane G1, Phase 3): the down GEMM with its epilogue fused (tf_exl3_gemm_down) gives the
//! bits of the GEMM then the down epilogue, at GLM's down shape (K 2,048, N 4,096), over items that cross the 64-row
//! passes, with a skipped pair (the shared slot) left as it was.
const std = @import("std");
const rig = @import("../rig.zig");
const launch = @import("../../launches/exl3.zig");
const DeviceBuffer = @import("../../memory.zig").DeviceBuffer;

const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;

test "glm exl3 epilogue: the down GEMM with its epilogue fused gives the bits of the GEMM and the down epilogue" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    if (!t.bf16 or @import("../../exl3.zig").matrix() == false) return error.SkipZigTest;
    var rng: rig.Rng = .{ .state = 0xE913_0001 };
    const k = 2048;
    const n = 4096;
    const experts = 3;
    const pairs = 300; // 3 experts: items of 130, 100 and 69 pairs (more than one 64-row pass each), pair 299 skipped
    const hx = try gpa.alloc(u16, pairs * k);
    defer gpa.free(hx);
    for (hx) |*v| v.* = rig.f16Bits(@floatCast(rng.normal()));
    var x = try t.upload(hx);
    defer x.free();
    var tr: [experts]DeviceBuffer = undefined;
    var tp: [experts]u64 = undefined;
    const words = try gpa.alloc(u16, k / 16 * (n / 16) * 64);
    defer gpa.free(words);
    for (0..experts) |e| {
        for (words) |*w| w.* = @truncate(rng.next());
        tr[e] = try t.upload(words);
        tp[e] = at(tr[e]);
    }
    defer for (&tr) |*b| b.free();
    var table = try t.upload(&tp);
    defer table.free();
    const sv = try gpa.alloc(u16, experts * n);
    defer gpa.free(sv);
    for (sv) |*v| v.* = rig.f16Bits(@floatCast((0.5 + rng.uniform()) / 64.0));
    var svh = try t.upload(sv);
    defer svh.free();
    // pick: pairs 0-129 expert 0, 130-229 expert 1, 230-298 expert 2, 299 the shared slot (id = experts)
    var pk: [pairs]i32 = undefined;
    for (&pk, 0..) |*v, i| v.* = if (i < 130) 0 else if (i < 230) 1 else if (i < 299) 2 else experts;
    var pick = try t.upload(&pk);
    defer pick.free();
    // members: each expert's pairs in a shuffled order (the plan's order is a stable sort; any order is a valid plan)
    var mem: [pairs]i32 = undefined;
    for (&mem, 0..) |*v, i| v.* = @intCast(i);
    for ([_][2]usize{ .{ 0, 130 }, .{ 130, 230 }, .{ 230, 299 } }) |r| {
        var i = r[1] - 1;
        while (i > r[0]) : (i -= 1) {
            const j = r[0] + @as(usize, @intCast(rng.next() % (i - r[0] + 1)));
            std.mem.swap(i32, &mem[i], &mem[j]);
        }
    }
    var members = try t.upload(&mem);
    defer members.free();
    const it = [_]i32{ 0, 0, 130, 1, 130, 100, 2, 230, 69 };
    var items = try t.upload(&it);
    defer items.free();
    var z = try t.alloc(pairs * n * 4);
    defer z.free();
    var y0 = try t.alloc(pairs * n * 4);
    defer y0.free();
    var y1 = try t.alloc(pairs * n * 4);
    defer y1.free();
    try y0.fill8(0xa5);
    try y1.fill8(0xa5);
    const p: launch.Product = .{ .x0 = at(x), .x1 = at(x), .x_div = 1, .tp0 = at(table), .tp1 = at(table), .mats = 1, .items = at(items), .count = 3, .members = at(members), .z = at(z), .k = k, .n = n, .pairs = pairs, .experts = experts };
    try launch.gemm(&t.on, p, t.stream.handle);
    try launch.down(&t.on, at(z), at(pick), at(svh), at(y0), pairs, n, 1, experts, t.stream.handle);
    try launch.gemmDown(&t.on, p, at(svh), at(y1), t.stream.handle);
    try t.stream.synchronize();
    const a = try rig.download(u32, y0);
    defer gpa.free(a);
    const b = try rig.download(u32, y1);
    defer gpa.free(b);
    for (0..pairs) |q| {
        const same = std.mem.eql(u32, a[q * n ..][0..n], b[q * n ..][0..n]);
        try rig.expect(same, "exl3 gemm down: pair {d}'s y differs from the GEMM + down epilogue", .{q});
    }
    for (b[299 * n ..][0..n]) |v| try rig.expect(v == 0xa5a5a5a5, "exl3 gemm down: the skipped pair's y left as it was", .{});
    std.debug.print("RESULT glm exl3 gemm down: {d} pairs' y bit-equal to the GEMM + down epilogue (3 items over the 64-row passes), the skipped pair untouched\n", .{pairs});
}
