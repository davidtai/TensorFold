//! The merged decode-step launches against the launches they replace, each against a float64 reference.
const std = @import("std");
const rig = @import("rig.zig");
const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;
const Launcher = @import("../launches.zig").Launcher;

fn p(b: anytype) ?*anyopaque {
    return @ptrFromInt(at(b));
}

/// The router's logits for `rows` rows over (experts, d) fp32 weights: the merged launch no further from float64.
fn router(t: *Rig, rng: *rig.Rng, rows: usize, d: usize, experts: usize) !void {
    const hx = try gpa.alloc(u16, rows * d);
    defer gpa.free(hx);
    for (hx) |*v| v.* = t.bits(rng.unit());
    const hw = try gpa.alloc(f32, experts * d);
    defer gpa.free(hw);
    for (hw) |*v| v.* = rng.unit() / 8.0;
    var x = try t.upload(hx);
    defer x.free();
    var w = try t.upload(hw);
    defer w.free();
    var max_err: [2]f64 = .{ 0, 0 };
    for ([2]*const Launcher{ &t.off, &t.on }, 0..) |l, v| {
        var out = try t.alloc(rows * experts * 4);
        defer out.free();
        try l.tf_moe_router(p(x), t.kind(), @ptrFromInt(at(w)), @ptrFromInt(at(out)), @intCast(rows), @intCast(d), @intCast(experts), t.stream.handle);
        try t.stream.synchronize();
        const got = try rig.download(f32, out);
        defer gpa.free(got);
        for (0..rows) |r| for (0..experts) |e| {
            var y: f64 = 0;
            var norm: f64 = 0;
            for (0..d) |i| {
                const term = t.value(hx[r * d + i]) * @as(f64, hw[e * d + i]);
                y += term;
                norm += @abs(term);
            }
            max_err[v] = @max(max_err[v], @abs(@as(f64, got[r * experts + e]) - y) / norm);
        };
    }
    try std.testing.expect(max_err[1] <= 2 * max_err[0] + 1e-9);
}

/// x = x + y (or the slots' weighted sum) and the next norm: one merged launch against combine, add and norm.
fn tails(t: *Rig, rng: *rig.Rng, rows: usize, width: usize, slots: usize) !void {
    const n = rows * width;
    const eps: f32 = 1e-6;
    const hx = try gpa.alloc(u16, n);
    defer gpa.free(hx);
    for (hx) |*v| v.* = t.bits(rng.unit());
    const hy16 = try gpa.alloc(u16, if (slots > 0) 0 else n);
    defer gpa.free(hy16);
    const hy32 = try gpa.alloc(f32, if (slots > 0) rows * slots * width else 0);
    defer gpa.free(hy32);
    for (hy16) |*v| v.* = t.bits(rng.unit() / 8.0);
    for (hy32) |*v| v.* = rng.unit() / 8.0;
    const hw = try gpa.alloc(f32, rows * @max(slots, 1));
    defer gpa.free(hw);
    for (hw) |*v| v.* = (rng.unit() + 1.0) / @as(f32, @floatFromInt(@max(slots, 1)));
    const hn = try gpa.alloc(f32, width);
    defer gpa.free(hn);
    for (hn) |*v| v.* = 1.0 + rng.unit() / 4.0;
    var y = if (slots > 0) try t.upload(hy32) else try t.upload(hy16);
    defer y.free();
    var wts = try t.upload(hw);
    defer wts.free();
    var weight = try t.upload(hn);
    defer weight.free();
    var tmp = try t.alloc(n * 2);
    defer tmp.free();
    var errs: [2][2]f64 = .{ .{ 0, 0 }, .{ 0, 0 } };
    for (0..2) |v| {
        var x = try t.upload(hx);
        defer x.free();
        var normed = try t.alloc(n * 2);
        defer normed.free();
        const s = t.stream.handle;
        if (v == 0) {
            const sum = if (slots > 0) p(tmp) else p(y);
            if (slots > 0) try t.off.tf_moe_combine(@ptrFromInt(at(y)), @ptrFromInt(at(wts)), sum, t.kind(), @intCast(rows), @intCast(slots), @intCast(width), s);
            try t.off.tf_add(p(x), sum, p(x), t.kind(), @intCast(n), s);
            try t.off.tf_rms(p(x), @ptrFromInt(at(weight)), p(normed), t.kind(), @intCast(rows), @intCast(width), eps, s);
        } else {
            try t.on.tf_tail(p(x), p(y), if (slots > 0) @ptrFromInt(at(wts)) else null, @ptrFromInt(at(weight)), p(normed), t.kind(), @intCast(rows), @intCast(slots), @intCast(width), eps, s);
        }
        try t.stream.synchronize();
        const gx = try rig.download(u16, x);
        defer gpa.free(gx);
        const gn = try rig.download(u16, normed);
        defer gpa.free(gn);
        const want = try gpa.alloc(f64, width);
        defer gpa.free(want);
        for (0..rows) |r| {
            var sq: f64 = 0;
            var top: f64 = 0;
            for (0..width) |c| {
                var add: f64 = 0;
                if (slots > 0) {
                    for (0..slots) |sl| add += @as(f64, hy32[(r * slots + sl) * width + c]) * @as(f64, hw[r * slots + sl]);
                } else add = t.value(hy16[r * width + c]);
                want[c] = t.value(hx[r * width + c]) + add;
                sq += want[c] * want[c];
                top = @max(top, @abs(want[c]));
            }
            const inv = 1.0 / @sqrt(sq / @as(f64, @floatFromInt(width)) + eps);
            for (0..width) |c| {
                errs[v][0] = @max(errs[v][0], @abs(t.value(gx[r * width + c]) - want[c]) / top);
                errs[v][1] = @max(errs[v][1], @abs(t.value(gn[r * width + c]) - want[c] * inv * hn[c]) / (top * inv * 1.25));
            }
        }
    }
    try std.testing.expect(errs[1][0] <= 2 * errs[0][0] + 1e-7);
    try std.testing.expect(errs[1][1] <= 2 * errs[0][1] + 1e-7);
}

/// The router over bf16 rows and the pick in one launch against tf_moe_router over the widened rows and tf_select_decode
/// (and tf_moe_route for 2 to 8 rows): the same logits, picks, weights and plan, byte for byte; `ties` copies expert rows
/// so logits tie. The plan of 2 to 8 rows is the window's (tile 8, the decode tile's rows). B-5: at 2 to 8 rows the one
/// launch with `apart` gives the same logits, picks, weights and members (every pair's index, so the down and the combine
/// read the same rows), the same routed items in the same slots with the shared expert's item out of them, and that item
/// in the extra slot `cap`.
fn routerSelect(t: *Rig, rng: *rig.Rng, rows: usize, d: usize, e: usize, top_k: usize, ties: bool) !void {
    const slots = top_k + 1;
    const cap: usize = 16 * slots + 16 * slots / 8;
    const hx = try gpa.alloc(u16, rows * d);
    defer gpa.free(hx);
    for (hx) |*v| v.* = t.bits(rng.unit());
    const h16 = try gpa.alloc(u16, e * d);
    defer gpa.free(h16);
    for (h16) |*v| v.* = rig.bf16Bits(rng.unit() / 8.0);
    if (ties) for ([_]usize{ 3, 9, 100, 511 }) |dst| @memcpy(h16[dst * d ..][0..d], h16[0..d]);
    const h32 = try gpa.alloc(f32, e * d);
    defer gpa.free(h32);
    for (h16, h32) |b, *v| v.* = @bitCast(@as(u32, b) << 16);
    var x = try t.upload(hx);
    defer x.free();
    var w16 = try t.upload(h16);
    defer w16.free();
    var w32 = try t.upload(h32);
    defer w32.free();
    var done = try t.upload(@as([]const u32, &.{0}));
    defer done.free();
    const sizes = [_]usize{ rows * e * 4, rows * slots * 4, rows * slots * 4, (cap + 1) * 12, rows * slots * 4 };
    var got: [4][5][]u8 = undefined;
    const apart = rows >= 2 and rows <= 8;
    const runs: usize = if (apart) 4 else 3;
    // 0: the two launches; 1 and 2: the one launch twice (its block counter is zero again after each); 3: B-5's apart plan
    for (0..runs) |v| {
        var bufs: [5]@import("../memory.zig").DeviceBuffer = undefined;
        for (&bufs, sizes) |*b, n| {
            b.* = try t.alloc(n);
            const zero = try gpa.alloc(u8, n);
            defer gpa.free(zero);
            @memset(zero, 0);
            try b.upload(0, zero);
        }
        defer for (&bufs) |*b| b.free();
        const s = t.stream.handle;
        const planned = rows <= 8;
        const tile: usize = if (rows == 1) 0 else 8;
        const items: ?[*]i32 = if (planned) @ptrFromInt(at(bufs[3])) else null;
        const members: ?[*]i32 = if (planned) @ptrFromInt(at(bufs[4])) else null;
        if (v == 0) {
            try t.on.tf_moe_router(p(x), t.kind(), @ptrFromInt(at(w32)), @ptrFromInt(at(bufs[0])), @intCast(rows), @intCast(d), @intCast(e), s);
            const one = rows == 1;
            try t.on.tf_moe_select(@ptrFromInt(at(bufs[0])), @ptrFromInt(at(bufs[1])), @ptrFromInt(at(bufs[2])), if (one) items else null, if (one) members else null, @intCast(cap), @intCast(rows), @intCast(e - 1), @intCast(top_k), s);
            if (planned and !one) try t.on.tf_moe_route(@ptrFromInt(at(bufs[1])), @intCast(rows * slots), @intCast(e), @intCast(tile), members, items, @intCast(cap), s);
        } else {
            try std.testing.expect(try t.on.routerSelect(p(x), t.kind(), p(w16), @ptrFromInt(at(bufs[0])), @ptrFromInt(at(bufs[1])), @ptrFromInt(at(bufs[2])), items, members, @intCast(cap), @intCast(rows), @intCast(d), @intCast(e), @intCast(top_k), @intCast(tile), at(done), v == 3, s));
        }
        try t.stream.synchronize();
        for (&got[v], bufs) |*g, b| g.* = try rig.download(u8, b);
    }
    defer for (got[0..runs]) |*g| for (g) |b| gpa.free(b);
    if (apart) {
        // logits, picks, weights and members: the same bytes
        for ([_]usize{ 0, 1, 2, 4 }) |k| if (!std.mem.eql(u8, got[0][k], got[3][k])) {
            std.debug.print("router+pick apart rows={d} ties={}: buffer {d} differs\n", .{ rows, ties, k });
            return error.TestUnexpectedResult;
        };
        const want: []align(1) const i32 = std.mem.bytesAsSlice(i32, got[0][3]);
        const have: []align(1) const i32 = std.mem.bytesAsSlice(i32, got[3][3]);
        const shared: i32 = @intCast(e - 1);
        var n: usize = 0; // have's live routed items so far
        var shared_item: ?[3]i32 = null;
        for (0..cap) |slot| {
            const item = [3]i32{ want[3 * slot], want[3 * slot + 1], want[3 * slot + 2] };
            if (item[2] <= 0) continue;
            if (item[0] == shared) {
                shared_item = item;
                continue;
            }
            try std.testing.expectEqualSlices(i32, &item, have[3 * n ..][0..3]);
            n += 1;
        }
        for (n..cap) |slot| try std.testing.expectEqual(@as(i32, 0), have[3 * slot + 2]);
        try std.testing.expectEqualSlices(i32, &shared_item.?, have[3 * cap ..][0..3]);
    }
    for (1..3) |v| for (0..5) |k| {
        if (k == 3 and v != 0) {
            // the extra slot is B-5's; the plan's capacity slots compare as before
            if (!std.mem.eql(u8, got[0][k][0 .. cap * 12], got[v][k][0 .. cap * 12])) {
                std.debug.print("router+pick rows={d} ties={} run {d}: buffer {d} differs\n", .{ rows, ties, v, k });
                return error.TestUnexpectedResult;
            }
            continue;
        }
        if (!std.mem.eql(u8, got[0][k], got[v][k])) {
            std.debug.print("router+pick rows={d} ties={} run {d}: buffer {d} differs\n", .{ rows, ties, v, k });
            return error.TestUnexpectedResult;
        }
    };
}

/// MG-1: tf_moe_route's prompt plan (items of 128) puts every live item before planItems(pairs, experts, 128), so a
/// prompt's expert launches cover them with that z; `bound` routings (128 k + 1 pairs an expert) fill it exactly.
fn routeLive(t: *Rig, rng: *rig.Rng, rows: usize, bound: bool) !void {
    const moe = @import("../moe.zig");
    const experts = 513;
    const tile = 128;
    const pairs = rows * 11;
    const cap = moe.capacity(pairs, experts);
    const pick = try gpa.alloc(i32, pairs);
    defer gpa.free(pick);
    if (bound) {
        // 128 k + 1 pairs an expert, the pairs past them in whole tiles and one partial spare, in a shuffled order
        const per = (pairs - experts) / experts / tile;
        var n: usize = 0;
        for (0..experts) |e| for (0..tile * per + 1) |_| {
            pick[n] = @intCast(e);
            n += 1;
        };
        var e: usize = 0;
        while (pairs - n >= tile) : (e += 1) {
            for (0..tile) |_| {
                pick[n] = @intCast(e);
                n += 1;
            }
        }
        while (n < pairs) : (n += 1) pick[n] = experts - 1;
        var i: usize = pairs;
        while (i > 1) : (i -= 1) std.mem.swap(i32, &pick[i - 1], &pick[rng.next() % i]);
    } else for (pick) |*e| {
        const r = rng.next();
        e.* = @intCast(if (r % 2 == 0) r % 8 else r % experts);
    }
    var dev_pick = try t.upload(pick);
    defer dev_pick.free();
    var members = try t.alloc(pairs * 4);
    defer members.free();
    var items = try t.alloc(cap * 12);
    defer items.free();
    try items.fill8(0xFF);
    try t.on.tf_moe_route(@ptrFromInt(at(dev_pick)), @intCast(pairs), experts, tile, @ptrFromInt(at(members)), @ptrFromInt(at(items)), @intCast(cap), t.stream.handle);
    try t.stream.synchronize();
    const host = try rig.download(i32, items);
    defer gpa.free(host);
    const z = moe.planItems(pairs, experts, tile);
    var live: usize = 0;
    var past: usize = 0;
    var covered: usize = 0;
    for (0..cap) |slot| if (host[3 * slot + 2] > 0) {
        live += 1;
        covered += @intCast(host[3 * slot + 2]);
        if (slot >= z) past += 1;
    };
    std.debug.print("  prompt plan rows {d}{s}: {d} live items, z {d} (capacity {d})\n", .{ rows, if (bound) " at the bound" else "", live, z, cap });
    try std.testing.expectEqual(@as(usize, 0), past);
    try std.testing.expectEqual(pairs, covered);
    if (bound) try std.testing.expectEqual(z, live);
}

test "a prompt plan's live items come before planItems(pairs, experts, 128), which they fill at 128 k + 1 pairs an expert (MG-1)" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x2545F4914F6CDD1D };
    for ([_]usize{ 8192, 17715, 20480 }) |rows| for ([_]bool{ true, false }) |bound| try routeLive(t, &rng, rows, bound);
}

test "the router and the pick in one launch give the two launches' bytes (Qwen3.8 Flash Next: 513 x 2560, top 10)" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x9E3779B97F4A7C15 };
    for ([_]usize{ 1, 2, 3, 4, 5, 6, 7, 8, 16 }) |rows| for ([_]bool{ false, true }) |ties| try routerSelect(t, &rng, rows, 2560, 513, 10, ties);
}

/// Served plans (coding.json, default spec, halo/main aa79bf10; W7's TF_W7_EXPERTS dump): three MoE layers a verify width
/// of 1 to 8 rows, each i32 layer, rows, slots, capacity, then the picks (rows x slots) and the items (capacity x 3).
const served_plans = @embedFile("served_plans.bin");

/// The router and pick in one launch on a served plan: x row r is one at coordinate r, so expert e's logit for row r is
/// its weight row's entry r; the served picks get descending values in their pick order, the rest small ones. The
/// picks and the plan's items are then the served bytes, and every buffer equals the two (or three) launches'.
fn servedPlan(t: *Rig, rng: *rig.Rng, rows: usize, slots: usize, cap: usize, picks: []const i32, items: []const i32) !void {
    const d: usize = 2560;
    const e: usize = 513;
    const top_k = slots - 1;
    const hx = try gpa.alloc(u16, rows * d);
    defer gpa.free(hx);
    @memset(hx, t.bits(0));
    for (0..rows) |r| hx[r * d + r] = t.bits(1);
    const h16 = try gpa.alloc(u16, e * d);
    defer gpa.free(h16);
    @memset(h16, 0);
    for (0..rows) |r| {
        for (0..e) |ex| h16[ex * d + r] = rig.bf16Bits(rng.unit() / 8.0);
        for (0..top_k) |k| h16[@as(usize, @intCast(picks[r * slots + k])) * d + r] = rig.bf16Bits(4.0 - 0.25 * @as(f32, @floatFromInt(k)));
    }
    const h32 = try gpa.alloc(f32, e * d);
    defer gpa.free(h32);
    for (h16, h32) |b, *v| v.* = @bitCast(@as(u32, b) << 16);
    var x = try t.upload(hx);
    defer x.free();
    var w16 = try t.upload(h16);
    defer w16.free();
    var w32 = try t.upload(h32);
    defer w32.free();
    var done = try t.upload(@as([]const u32, &.{0}));
    defer done.free();
    const sizes = [_]usize{ rows * e * 4, rows * slots * 4, rows * slots * 4, cap * 12, rows * slots * 4 };
    var got: [2][5][]u8 = undefined;
    const tile: usize = if (rows == 1) 0 else 8;
    for (0..2) |v| {
        var bufs: [5]@import("../memory.zig").DeviceBuffer = undefined;
        for (&bufs, sizes) |*b, n| {
            b.* = try t.alloc(n);
            const zero = try gpa.alloc(u8, n);
            defer gpa.free(zero);
            @memset(zero, 0);
            try b.upload(0, zero);
        }
        defer for (&bufs) |*b| b.free();
        const s = t.stream.handle;
        const it: [*]i32 = @ptrFromInt(at(bufs[3]));
        const mem: [*]i32 = @ptrFromInt(at(bufs[4]));
        if (v == 0) {
            try t.on.tf_moe_router(p(x), t.kind(), @ptrFromInt(at(w32)), @ptrFromInt(at(bufs[0])), @intCast(rows), @intCast(d), @intCast(e), s);
            const one = rows == 1;
            try t.on.tf_moe_select(@ptrFromInt(at(bufs[0])), @ptrFromInt(at(bufs[1])), @ptrFromInt(at(bufs[2])), if (one) it else null, if (one) mem else null, @intCast(cap), @intCast(rows), @intCast(e - 1), @intCast(top_k), s);
            if (!one) try t.on.tf_moe_route(@ptrFromInt(at(bufs[1])), @intCast(rows * slots), @intCast(e), @intCast(tile), mem, it, @intCast(cap), s);
        } else {
            try std.testing.expect(try t.on.routerSelect(p(x), t.kind(), p(w16), @ptrFromInt(at(bufs[0])), @ptrFromInt(at(bufs[1])), @ptrFromInt(at(bufs[2])), it, mem, @intCast(cap), @intCast(rows), @intCast(d), @intCast(e), @intCast(top_k), @intCast(tile), at(done), false, s));
        }
        try t.stream.synchronize();
        for (&got[v], bufs) |*g, b| g.* = try rig.download(u8, b);
    }
    defer for (&got) |*g| for (g) |b| gpa.free(b);
    for (0..2) |v| {
        if (!std.mem.eql(u8, got[v][1], std.mem.sliceAsBytes(picks)) or !std.mem.eql(u8, got[v][3], std.mem.sliceAsBytes(items))) {
            std.debug.print("served plan rows={d} run {d}: picks or items differ from the served bytes\n", .{ rows, v });
            return error.TestUnexpectedResult;
        }
    }
    for (0..5) |k| if (!std.mem.eql(u8, got[0][k], got[1][k])) {
        std.debug.print("served plan rows={d}: buffer {d} differs between the launches\n", .{ rows, k });
        return error.TestUnexpectedResult;
    };
}

test "the router and the pick in one launch give a served plan's picks and items, byte for byte (rows 1 to 8)" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x2545F4914F6CDD1D };
    var at_byte: usize = 0;
    var seen: [9]usize = @splat(0);
    while (at_byte + 16 <= served_plans.len) {
        const head = std.mem.bytesAsSlice(i32, served_plans[at_byte..][0..16]);
        const rows: usize = @intCast(head[1]);
        const slots: usize = @intCast(head[2]);
        const cap: usize = @intCast(head[3]);
        at_byte += 16;
        const picks = try gpa.alloc(i32, rows * slots);
        defer gpa.free(picks);
        @memcpy(std.mem.sliceAsBytes(picks), served_plans[at_byte..][0 .. rows * slots * 4]);
        at_byte += rows * slots * 4;
        const items = try gpa.alloc(i32, cap * 3);
        defer gpa.free(items);
        @memcpy(std.mem.sliceAsBytes(items), served_plans[at_byte..][0 .. cap * 12]);
        at_byte += cap * 12;
        try servedPlan(t, &rng, rows, slots, cap, picks, items);
        seen[rows] += 1;
    }
    for (seen[1..]) |n| try std.testing.expect(n == 3);
}

test "a prompt's picks a wave a row (tf_select_rows) are select_body's bytes, ties included" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x5E1EC7 };
    const top_k: usize = 10;
    const slots = top_k + 1;
    for ([_][2]usize{ .{ 4099, 512 }, .{ 37, 512 }, .{ 1029, 1024 }, .{ 6, 64 } }) |shape| {
        const rows = shape[0];
        const experts = shape[1];
        const h = try gpa.alloc(f32, rows * (experts + 1));
        defer gpa.free(h);
        for (h) |*v| v.* = rng.unit() * 4.0;
        // ties: three experts copy expert 0's logit in every row, and in odd rows expert 5 copies expert 1's
        for (0..rows) |r| {
            const row = h[r * (experts + 1) ..][0 .. experts + 1];
            for ([_]usize{ 3, 9, experts / 2 }) |c| row[c] = row[0];
            if (r % 2 == 1) row[5] = row[1];
        }
        var logits = try t.upload(h);
        defer logits.free();
        var got: [2][2][]u8 = undefined;
        for (0..2) |v| {
            var pick = try t.alloc(rows * slots * 4);
            defer pick.free();
            var wts = try t.alloc(rows * slots * 4);
            defer wts.free();
            var items = try t.alloc(slots * 12);
            defer items.free();
            var members = try t.alloc(slots * 4);
            defer members.free();
            // 0: select_body (tf_select_decode, taken where a plan is asked for); 1: tf_select_rows
            const it: ?[*]i32 = if (v == 0) @ptrFromInt(at(items)) else null;
            const mem: ?[*]i32 = if (v == 0) @ptrFromInt(at(members)) else null;
            try t.on.tf_moe_select(@ptrFromInt(at(logits)), @ptrFromInt(at(pick)), @ptrFromInt(at(wts)), it, mem, @intCast(slots), @intCast(rows), @intCast(experts), @intCast(top_k), t.stream.handle);
            try t.stream.synchronize();
            got[v] = .{ try rig.download(u8, pick), try rig.download(u8, wts) };
        }
        defer for (&got) |*g| for (g) |b| gpa.free(b);
        for (0..2) |k| if (!std.mem.eql(u8, got[0][k], got[1][k])) {
            std.debug.print("select rows={d} experts={d}: {s} differ\n", .{ rows, experts, if (k == 0) "picks" else "weights" });
            return error.TestUnexpectedResult;
        };
    }
}

test "the merged router launch is no further from float64 than twice the launch it replaces" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x9E3779B97F4A7C15 };
    for ([_]usize{ 1, 2, 4, 8 }) |rows| try router(t, &rng, rows, 2048, 257);
}

test "the merged residual-and-norm tails are no further from float64 than twice the launches they replace" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x9E3779B97F4A7C15 };
    for ([_]usize{ 1, 4, 16 }) |rows| {
        try tails(t, &rng, rows, 2048, 0);
        try tails(t, &rng, rows, 2048, 9);
    }
    try tails(t, &rng, 1, 4096, 0);
}

const affine = @import("../launches/affine.zig");
const ref = @import("affine/reference.zig");

/// A random packed product on the host and the device: (n, k) at `bits` and `group`, bf16 tables, `experts` stacked.
const Matrix = struct {
    n: usize,
    k: usize,
    bits: usize,
    group: usize,
    words: []u32,
    scale: []u16,
    bias: []u16,
    dev: [3]@import("../memory.zig").DeviceBuffer,

    fn init(t: *Rig, rng: *rig.Rng, n: usize, k: usize, bits: usize, group_size: usize, experts: usize) !Matrix {
        var m: Matrix = .{ .n = n, .k = k, .bits = bits, .group = group_size, .words = undefined, .scale = undefined, .bias = undefined, .dev = undefined };
        m.words = try gpa.alloc(u32, experts * n * k * bits / 32);
        m.scale = try gpa.alloc(u16, experts * n * k / group_size);
        m.bias = try gpa.alloc(u16, m.scale.len);
        for (m.words) |*w| w.* = @truncate(rng.next() >> 16);
        for (m.scale) |*v| v.* = rig.bf16Bits((rng.unit() + 1.0) / 16.0);
        for (m.bias) |*v| v.* = rig.bf16Bits(rng.unit() / 2.0);
        m.dev = .{ try t.upload(m.words), try t.upload(m.scale), try t.upload(m.bias) };
        return m;
    }

    fn deinit(m: *Matrix) void {
        gpa.free(m.words);
        gpa.free(m.scale);
        gpa.free(m.bias);
        for (&m.dev) |*b| b.free();
    }

    fn problem(m: Matrix, t: *const Rig, x: []const u16) ref.Problem {
        return .{ .fp16 = !t.bf16, .n = m.n, .k = m.k, .bits = m.bits, .group = m.group, .x = x, .words = m.words, .scale = m.scale, .bias = m.bias };
    }

    fn arg(m: Matrix, t: *const Rig, x: u64, rows: usize, out: u64) affine.Arg {
        return .{
            .x = x,
            .words = at(m.dev[0]),
            .scale = .{ .p = at(m.dev[1]), .kind = 1 },
            .bias = .{ .p = at(m.dev[2]), .kind = 1 },
            .out = out,
            .m = @intCast(rows),
            .n = @intCast(m.n),
            .k = @intCast(m.k),
            .bits = @intCast(m.bits),
            .group = @intCast(m.group),
            .fp16 = @intFromBool(!t.bf16),
        };
    }
};

/// Up to four products sharing x in one launch against one launch each: no further from float64 than twice that.
fn group(t: *Rig, rng: *rig.Rng, rows: usize, ns: []const usize, k: usize, bits: usize, group_size: usize) !void {
    const hx = try gpa.alloc(u16, rows * k);
    defer gpa.free(hx);
    for (hx) |*v| v.* = t.bits(rng.unit());
    var x = try t.upload(hx);
    defer x.free();
    var ms: [4]Matrix = undefined;
    var outs: [2][4]@import("../memory.zig").DeviceBuffer = undefined;
    for (ns, 0..) |n, i| {
        ms[i] = try Matrix.init(t, rng, n, k, bits, group_size, 1);
        for (&outs) |*o| o[i] = try t.alloc(rows * n * 2);
    }
    defer for (ns, 0..) |_, i| {
        ms[i].deinit();
        for (&outs) |*o| o[i].free();
    };
    const kernels = &t.on.affine;
    for (ms[0..ns.len], 0..) |m, i| try kernels.run(&t.r, m.arg(t, at(x), rows, at(outs[0][i])), 4, t.stream.handle, 0, 1, true);
    var sides: [4]affine.Side = undefined;
    for (ms[0..ns.len], 0..) |m, i| sides[i] = .{ .words = at(m.dev[0]), .scale = at(m.dev[1]), .bias = at(m.dev[2]), .n = @intCast(m.n), .out = at(outs[1][i]) };
    try kernels.groupRun(&t.r, ms[0].arg(t, at(x), rows, 0), sides[0..ns.len], true, t.stream.handle);
    try t.stream.synchronize();
    var errs: [2]f64 = .{ 0, 0 };
    for (0..2) |v| for (ns, 0..) |n, i| {
        const got = try rig.download(u16, outs[v][i]);
        defer gpa.free(got);
        const prob = ms[i].problem(t, hx);
        for (0..rows) |r| for (0..@min(n, 12)) |ci| {
            const col = ci * (n - 1) / @max(@min(n, 12) - 1, 1);
            const want = ref.reference(prob, r, 0, col);
            errs[v] = @max(errs[v], @abs(t.value(got[r * n + col]) - want.y) / want.norm);
        };
    };
    try std.testing.expect(errs[1] <= 2 * errs[0] + 1e-9);
}

/// A token's routed gate and up, then the activation, against one launch with the activation as epilogue.
fn pair(t: *Rig, rng: *rig.Rng, rows: usize, width: usize, k: usize, bits: usize, group_size: usize) !void {
    const experts = 64;
    const slots = 9;
    const pairs = rows * slots;
    var m = try Matrix.init(t, rng, 2 * width, k, bits, group_size, experts);
    defer m.deinit();
    const hx = try gpa.alloc(u16, rows * k);
    defer gpa.free(hx);
    for (hx) |*v| v.* = t.bits(rng.unit());
    var x = try t.upload(hx);
    defer x.free();
    // a token's pairs are its slots; one pair an item
    const pick = try gpa.alloc(u32, pairs);
    defer gpa.free(pick);
    for (pick) |*e| e.* = @intCast(rng.next() % experts);
    const items = try gpa.alloc(i32, pairs * 3);
    defer gpa.free(items);
    const members = try gpa.alloc(i32, pairs);
    defer gpa.free(members);
    for (0..pairs) |i| {
        items[3 * i ..][0..3].* = .{ @intCast(pick[i]), @intCast(i), 1 };
        members[i] = @intCast(i);
    }
    var dev_items = try t.upload(items);
    defer dev_items.free();
    var dev_members = try t.upload(members);
    defer dev_members.free();
    var both = try t.alloc(pairs * 2 * width * 4);
    defer both.free();
    var acts: [2]@import("../memory.zig").DeviceBuffer = .{ try t.alloc(pairs * width * 2), try t.alloc(pairs * width * 2) };
    defer for (&acts) |*a| a.free();
    const kernels = &t.on.affine;
    var a = m.arg(t, at(x), rows, at(both));
    a.route = .{ .items = at(dev_items), .members = at(dev_members), .x_div = slots };
    try kernels.routedWith(&t.r, a, @intCast(pairs), t.stream.handle, .gemm);
    try t.off.tf_moe_act(@ptrFromInt(at(both)), @ptrFromInt(at(acts[0])), t.kind(), @intCast(pairs), @intCast(width), 0, t.stream.handle);
    a.out = 0;
    a.out16 = at(acts[1]);
    try std.testing.expect(try kernels.pairRun(&t.r, a, 0, @intCast(pairs), t.stream.handle));
    try t.stream.synchronize();
    var errs: [2]f64 = .{ 0, 0 };
    const prob = m.problem(t, hx);
    for (0..2) |v| {
        const got = try rig.download(u16, acts[v]);
        defer gpa.free(got);
        for (0..pairs) |pi| for (0..12) |ci| {
            const col = ci * (width - 1) / 11;
            const g = ref.reference(prob, pi / slots, pick[pi], col);
            const u = ref.reference(prob, pi / slots, pick[pi], col + width);
            const want = g.y / (1.0 + @exp(-g.y)) * u.y;
            errs[v] = @max(errs[v], @abs(t.value(got[pi * width + col]) - want) / @max(@abs(want), 0.01));
        };
    }
    try std.testing.expect(errs[1] <= 2 * errs[0] + 1e-9);
}

test "products sharing x in one launch are no further from float64 than twice one launch each" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x9E3779B97F4A7C15 };
    for ([_]usize{ 1, 2, 4, 8, 16 }) |rows| try group(t, &rng, rows, &.{ 8192, 4096, 32, 32 }, 2048, 4, 64);
    try group(t, &rng, 1, &.{ 8192, 512, 512 }, 2048, 4, 64);
    try group(t, &rng, 1, &.{ 12288, 12288 }, 4096, 6, 64);
    try group(t, &rng, 4, &.{ 4096, 4096 }, 4096, 8, 128);
    try group(t, &rng, 2, &.{ 1024, 96, 96 }, 2048, 3, 32);
}

test "the routed gate and up with its activation in one launch is no further from float64 than twice the two launches" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x9E3779B97F4A7C15 };
    for ([_]usize{ 1, 2, 4 }) |rows| try pair(t, &rng, rows, 512, 2048, 4, 64);
    try pair(t, &rng, 1, 768, 2048, 6, 64);
    try pair(t, &rng, 1, 512, 2048, 4, 128);
}

/// A verify window's routed experts at every columns-a-lane count against the launcher's own pick, and against the
/// same pairs with every pair its own item (each pass then on the 1-row tile): the same bytes. Flash Next's gate+up
/// with its activation (`down` false: 640 x 2 columns, K 2560) or its down (`down` true: 2560 columns, K 640, x a row
/// a pair), 4-bit group 32; rows x 11 (row, slot) pairs over 11 + 5.5 (rows - 1) experts, round-robin, or over
/// `unique` experts when given (4 rows an expert at unique 11 x rows / 4); the plan ends in 9 dead items, as tf_moe_route's;
/// and at a.m = the window's rows, the engine's launch (the work list at 5 to 8 rows).
fn windowBytes(t: *Rig, rng: *rig.Rng, down: bool, rows: usize, unique_in: ?usize) !void {
    const n: usize = if (down) 2560 else 2 * 640;
    const k: usize = if (down) 640 else 2560;
    const experts = 64;
    const slots = 11;
    var m = try Matrix.init(t, rng, n, k, 4, 32, experts);
    defer m.deinit();
    const kernels = &t.on.affine;
    defer kernels.stream_cb_force = 0;
    const pairs = rows * slots;
    const unique = unique_in orelse slots + (slots * (rows - 1)) / 2;
    const x_rows = if (down) pairs else rows;
    const hx = try gpa.alloc(u16, x_rows * k);
    defer gpa.free(hx);
    for (hx) |*v| v.* = t.bits(rng.unit());
    var x = try t.upload(hx);
    defer x.free();
    const dead = 9;
    const items = try gpa.alloc(i32, (unique + dead) * 3);
    defer gpa.free(items);
    const members = try gpa.alloc(i32, pairs);
    defer gpa.free(members);
    // the same pairs, every one its own item (members in the same order), then the dead items
    const single = try gpa.alloc(i32, (pairs + dead) * 3);
    defer gpa.free(single);
    var first: usize = 0;
    var most: usize = 0;
    for (0..unique) |i| {
        var count: usize = 0;
        var j = i;
        while (j < pairs) : (j += unique) {
            members[first + count] = @intCast(j);
            single[3 * (first + count) ..][0..3].* = .{ @intCast((i * 7) % experts), @intCast(first + count), 1 };
            count += 1;
        }
        items[3 * i ..][0..3].* = .{ @intCast((i * 7) % experts), @intCast(first), @intCast(count) };
        first += count;
        most = @max(most, count);
    }
    for (0..dead) |d| {
        items[3 * (unique + d) ..][0..3].* = .{ experts - 1, @intCast(pairs), 0 };
        single[3 * (pairs + d) ..][0..3].* = .{ experts - 1, @intCast(pairs), 0 };
    }
    var dev_items = try t.upload(items);
    defer dev_items.free();
    var dev_single = try t.upload(single);
    defer dev_single.free();
    var dev_members = try t.upload(members);
    defer dev_members.free();
    const out_bytes = if (down) pairs * n * 4 else pairs * (n / 2) * 2;
    var want: ?[]u8 = null;
    defer if (want) |w| gpa.free(w);
    // the CBs on the grouped plan, then the rule's CB on the plan of single items (a.m 1: every pass a 1-row tile),
    // then the grouped plan at a.m = the window's rows, as the engine launches it (5 to 8 rows: the work list)
    for ([_]c_int{ 0, 1, 2, 4, 8, -1, -2 }) |cb| {
        var out = try t.alloc(out_bytes);
        defer out.free();
        const split = cb == -1;
        var a = m.arg(t, at(x), if (split) 1 else if (cb == -2) rows else most, if (down) at(out) else 0);
        if (!down) a.out16 = at(out);
        a.route = .{ .items = at(if (split) dev_single else dev_items), .members = at(dev_members), .x_div = if (down) 1 else slots };
        const n_items: c_int = @intCast(if (split) pairs + dead else unique + dead);
        kernels.stream_cb_force = @max(cb, 0);
        if (down) try kernels.routed(&t.r, a, n_items, t.stream.handle) else try std.testing.expect(try kernels.pairRun(&t.r, a, 0, n_items, t.stream.handle));
        try t.stream.synchronize();
        const got = try rig.download(u8, out);
        if (want) |w| {
            defer gpa.free(got);
            try std.testing.expectEqualSlices(u8, w, got);
        } else want = got;
    }
}

test "the routed gate+up and down write the same bytes at every columns a lane and as single-pair items (Flash Next, 1 to 8 rows)" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x9E3779B97F4A7C15 };
    for ([_]bool{ false, true }) |down| {
        for ([_]usize{ 1, 2, 4, 5, 8 }) |rows| try windowBytes(t, &rng, down, rows, null);
        // 4 rows an expert: the 4-row tile
        try windowBytes(t, &rng, down, 4, 11);
        try windowBytes(t, &rng, down, 8, 22);
    }
}

/// A plain product (one matrix, fp32 out) at every columns-a-lane count against the launcher's pick: the same bytes.
fn plainBytes(t: *Rig, rng: *rig.Rng, n: usize, k: usize, rows: usize) !void {
    var m = try Matrix.init(t, rng, n, k, 4, 32, 1);
    defer m.deinit();
    const kernels = &t.on.affine;
    defer kernels.stream_cb_force = 0;
    const hx = try gpa.alloc(u16, rows * k);
    defer gpa.free(hx);
    for (hx) |*v| v.* = t.bits(rng.unit());
    var x = try t.upload(hx);
    defer x.free();
    var want: ?[]u8 = null;
    defer if (want) |w| gpa.free(w);
    for ([_]c_int{ 0, 1, 2, 4, 8 }) |cb| {
        var out = try t.alloc(rows * n * 4);
        defer out.free();
        const a = m.arg(t, at(x), rows, at(out));
        kernels.stream_cb_force = cb;
        try kernels.streamGo(&t.r, a, .{}, a.n, 1, t.stream.handle);
        try t.stream.synchronize();
        const got = try rig.download(u8, out);
        if (want) |w| {
            defer gpa.free(got);
            try std.testing.expectEqualSlices(u8, w, got);
        } else want = got;
    }
}

test "a plain product writes the same bytes at every columns a lane (Flash Next's in-proj K, out-proj, a ragged N; 1 to 4 rows)" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x9E3779B97F4A7C15 };
    for ([_]usize{ 1, 2, 4 }) |rows| {
        try plainBytes(t, &rng, 4096, 2560, rows);
        try plainBytes(t, &rng, 2560, 6144, rows);
        try plainBytes(t, &rng, 4099, 2560, rows);
    }
}

/// Flash Next's prompt expert products (tiles/moe_gemm.hpp) against the 128-row matrix tile: the same bytes. Gate and
/// up (`down` false) is compared after moe_act on both sides; down (`down` true) on the fp32 sums, x a row a pair.
/// `skew` piles the pairs onto a few experts (items of 128 rows and of 1 row); `limit` > 0 clamps before the activation.
/// `inexact`: the numerics=inexact tiles (codes dequantized to bf16) instead, within 2 u (down) or 3 u (gate+up's bf16
/// activation) relative RMS of those bytes, u = 2^-9.
fn promptExperts(t: *Rig, rng: *rig.Rng, down: bool, rows: usize, experts: usize, skew: bool, limit: f32, inexact: bool) !void {
    const kernels = &t.on.affine;
    var chosen = t.on.affine;
    chosen.inexact = inexact;
    const sh = if (down) affine.Kernels.moe_down_shape else affine.Kernels.moe_gu_shape;
    const n: usize = @intCast(sh.n);
    const k: usize = @intCast(sh.k);
    const slots = 11;
    const pairs = rows * slots;
    const width = if (down) n else n / 2;
    var m = try Matrix.init(t, rng, n, k, @intCast(sh.bits), @intCast(sh.group), experts);
    defer m.deinit();
    const x_rows = if (down) pairs else rows;
    const hx = try gpa.alloc(u16, x_rows * k);
    defer gpa.free(hx);
    for (hx) |*v| v.* = t.bits(rng.unit());
    var x = try t.upload(hx);
    defer x.free();
    // each pair draws an expert; sorted by expert (stable), cut into items of at most 128 pairs, as moe_route plans a prompt
    const pick = try gpa.alloc(u32, pairs);
    defer gpa.free(pick);
    for (pick) |*e| {
        const r = rng.next();
        e.* = @intCast(if (skew and r % 4 != 0) r % 3 else r % experts);
    }
    const members = try gpa.alloc(i32, pairs);
    defer gpa.free(members);
    var n_members: usize = 0;
    const items = try gpa.alloc(i32, 3 * (pairs + experts + 8));
    defer gpa.free(items);
    @memset(items, 0);
    var n_items: usize = 0;
    for (0..experts) |e| {
        var run: usize = 0;
        for (0..pairs) |pi| {
            if (pick[pi] != e) continue;
            if (run == 0) {
                items[3 * n_items ..][0..3].* = .{ @intCast(e), @intCast(n_members), 0 };
                n_items += 1;
            }
            members[n_members] = @intCast(pi);
            n_members += 1;
            items[3 * (n_items - 1) + 2] += 1;
            run = if (run + 1 == 128) 0 else run + 1;
        }
    }
    // a plan's capacity holds empty items past the used ones
    const count = n_items + 8;
    var dev_items = try t.upload(items[0 .. 3 * count]);
    defer dev_items.free();
    var dev_members = try t.upload(members);
    defer dev_members.free();
    var sums = [2]@import("../memory.zig").DeviceBuffer{ try t.alloc(pairs * n * 4), try t.alloc(pairs * n * 4) };
    defer for (&sums) |*b| b.free();
    var acts: [2]@import("../memory.zig").DeviceBuffer = .{ try t.alloc(pairs * width * 2), try t.alloc(pairs * width * 2) };
    defer for (&acts) |*b| b.free();
    try sums[1].fill8(0xA5);
    var a = m.arg(t, at(x), 128, at(sums[0]));
    a.route = .{ .items = at(dev_items), .members = at(dev_members), .x_div = if (down) 1 else slots };
    try kernels.blockGo(&t.r, a, t.stream.handle, @intCast(count), .matrix);
    if (down) {
        a.out = at(sums[1]);
        try std.testing.expect(try chosen.moeDown(&t.r, a, @intCast(count), t.stream.handle));
    } else {
        try t.off.tf_moe_act(@ptrFromInt(at(sums[0])), @ptrFromInt(at(acts[0])), t.kind(), @intCast(pairs), @intCast(width), limit, t.stream.handle);
        a.out = 0;
        a.out16 = at(acts[1]);
        try std.testing.expect(try chosen.moeGateUp(&t.r, a, @intCast(count), limit, t.stream.handle));
    }
    try t.stream.synchronize();
    const cmp = if (down) sums else acts;
    const want = try rig.download(u8, cmp[0]);
    defer gpa.free(want);
    const got = try rig.download(u8, cmp[1]);
    defer gpa.free(got);
    const size: usize = if (down) 4 else 2;
    if (inexact) {
        var num: f64 = 0;
        var den: f64 = 0;
        for (0..want.len / size) |e| {
            const w: f64 = if (down) @as(f32, @bitCast(want[e * 4 ..][0..4].*)) else @as(f32, @bitCast(@as(u32, @as(u16, @bitCast(want[e * 2 ..][0..2].*))) << 16));
            const g: f64 = if (down) @as(f32, @bitCast(got[e * 4 ..][0..4].*)) else @as(f32, @bitCast(@as(u32, @as(u16, @bitCast(got[e * 2 ..][0..2].*))) << 16));
            num += (g - w) * (g - w);
            den += w * w;
        }
        const rel = @sqrt(num / den);
        std.debug.print("  prompt experts inexact down {} rows {d}: relative RMS {e:.2} against the exact tiles\n", .{ down, rows, rel });
        // down's fp32 sums differ by the weights' bf16 rounding (2 u, u = 2^-9); gate+up's bf16 outputs carry it and
        // a rounding of the activation on each side (3 u)
        const u: f64 = 1.0 / 512.0;
        try std.testing.expect(rel <= @as(f64, if (down) 2.0 else 3.0) * u);
        return;
    }
    var differ: usize = 0;
    for (0..want.len / size) |e| if (!std.mem.eql(u8, want[e * size ..][0..size], got[e * size ..][0..size])) {
        if (differ < 4) std.debug.print("  prompt experts down {} rows {d} skew {}: pair {d} column {d} differs\n", .{ down, rows, skew, e / width, e % width });
        differ += 1;
    };
    try std.testing.expectEqual(@as(usize, 0), differ);
    // and the matrix tile's sums are float64's within its bound
    const prob = m.problem(t, hx);
    const ref_sums = try rig.download(f32, sums[0]);
    defer gpa.free(ref_sums);
    for (0..8) |s| {
        const pi = s * (pairs - 1) / 7;
        const col = (s * 97) % width;
        const r = ref.reference(prob, if (down) pi else pi / slots, pick[pi], col);
        try std.testing.expect(@abs(@as(f64, ref_sums[pi * n + col]) - r.y) <= 1e-5 * r.norm + 1e-7);
    }
}

test "Flash Next's prompt gate and up with its activation in one launch writes the bytes of the matrix tile and moe_act" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    if (t.on.affine.moe_gu == null) return error.SkipZigTest;
    var rng: rig.Rng = .{ .state = 0x9E3779B97F4A7C15 };
    try promptExperts(t, &rng, false, 300, 24, false, 0, false);
    try promptExperts(t, &rng, false, 2048, 24, true, 0, false);
    try promptExperts(t, &rng, false, 64, 24, false, 0, false);
    try promptExperts(t, &rng, false, 512, 24, false, 7.0, false);
}

test "Flash Next's prompt expert down writes the bytes of the matrix tile" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    if (t.on.affine.moe_down == null) return error.SkipZigTest;
    var rng: rig.Rng = .{ .state = 0x2545F4914F6CDD1D };
    try promptExperts(t, &rng, true, 300, 24, false, 0, false);
    try promptExperts(t, &rng, true, 2048, 24, true, 0, false);
    try promptExperts(t, &rng, true, 64, 24, false, 0, false);
}

/// A prompt's plain product on the dense tile of its K (tiles/dense_gemm.hpp) against the 128-row matrix tile: the same
/// bytes, and the registry's prefill choice for the shape. Rows and columns off the 128 multiples cover the edges. Then
/// the dense tile's BF16 output against the matrix tile's fp32 rounded by tf_cast: the same bytes.
fn promptDense(t: *Rig, rng: *rig.Rng, rows: usize, n: usize, k: usize) !void {
    const kernels = &t.on.affine;
    var m = try Matrix.init(t, rng, n, k, 4, 32, 1);
    defer m.deinit();
    const hx = try gpa.alloc(u16, rows * k);
    defer gpa.free(hx);
    for (hx) |*v| v.* = t.bits(rng.unit());
    var x = try t.upload(hx);
    defer x.free();
    var outs = [2]@import("../memory.zig").DeviceBuffer{ try t.alloc(rows * n * 4), try t.alloc(rows * n * 4) };
    defer for (&outs) |*b| b.free();
    try outs[1].fill8(0xA5);
    var a = m.arg(t, at(x), rows, at(outs[0]));
    try kernels.blockGo(&t.r, a, t.stream.handle, 1, .matrix);
    a.out = at(outs[1]);
    const e = kernels.choose(.project, .prefill, affine.Kernels.shapeOf(a, 1)) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("mlx.project.prefill.dense", e.id);
    try kernels.prefillLaunch(&t.r, a, t.stream.handle, 1);
    try t.stream.synchronize();
    const want = try rig.download(u8, outs[0]);
    defer gpa.free(want);
    const got = try rig.download(u8, outs[1]);
    defer gpa.free(got);
    var differ: usize = 0;
    for (0..want.len / 4) |i| if (!std.mem.eql(u8, want[i * 4 ..][0..4], got[i * 4 ..][0..4])) {
        if (differ < 4) std.debug.print("  prompt dense {d}x{d} rows {d}: row {d} column {d} differs\n", .{ n, k, rows, i / n, i % n });
        differ += 1;
    };
    try std.testing.expectEqual(@as(usize, 0), differ);
    // the BF16 output: the cast of the fp32 sums
    var halves = [2]@import("../memory.zig").DeviceBuffer{ try t.alloc(rows * n * 2), try t.alloc(rows * n * 2) };
    defer for (&halves) |*b| b.free();
    try halves[1].fill8(0xA5);
    try t.on.tf_cast(@ptrFromInt(at(outs[0])), 0, @ptrFromInt(at(halves[0])), t.kind(), @intCast(rows * n), t.stream.handle);
    a.out = 0;
    a.out16 = at(halves[1]);
    try kernels.prefillLaunch(&t.r, a, t.stream.handle, 1);
    try t.stream.synchronize();
    const want16 = try rig.download(u8, halves[0]);
    defer gpa.free(want16);
    const got16 = try rig.download(u8, halves[1]);
    defer gpa.free(got16);
    for (0..want16.len / 2) |i| if (!std.mem.eql(u8, want16[i * 2 ..][0..2], got16[i * 2 ..][0..2])) {
        if (differ < 4) std.debug.print("  prompt dense bf16 {d}x{d} rows {d}: row {d} column {d} differs\n", .{ n, k, rows, i / n, i % n });
        differ += 1;
    };
    try std.testing.expectEqual(@as(usize, 0), differ);
    const prob = m.problem(t, hx);
    const sums = try rig.download(f32, outs[1]);
    defer gpa.free(sums);
    for (0..8) |s| {
        const r = s * (rows - 1) / 7;
        const col = (s * 97) % n;
        const y = ref.reference(prob, r, 0, col);
        try std.testing.expect(@abs(@as(f64, sums[r * n + col]) - y.y) <= 1e-5 * y.norm + 1e-7);
    }
}

test "a prompt's dense products at Flash Next's K write the bytes of the matrix tile, and of tf_cast's rounding in bf16" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    if (t.on.affine.dense[0] == null) return error.SkipZigTest;
    var rng: rig.Rng = .{ .state = 0x6A09E667F3BCC908 };
    try promptDense(t, &rng, 300, 640, 2560);
    try promptDense(t, &rng, 130, 200, 2560);
    try promptDense(t, &rng, 257, 2560, 6144);
    try promptDense(t, &rng, 200, 324, 10240);
    try promptDense(t, &rng, 383, 10240, 320);
    try promptDense(t, &rng, 1024, 1024, 2560);
}

/// numerics=inexact: the inexact dense tile (tiles/dense_inexact.hpp) against the matrix tile's fp32 sums, which differ
/// by the bf16 rounding of the dequantized weights: relative RMS at most 4e-3 (2^-8). Its bf16 output is the rounding
/// of its own fp32 sums, and the registry takes it only under numerics=inexact.
fn promptDenseInexact(t: *Rig, rng: *rig.Rng, rows: usize, n: usize, k: usize) !void {
    const kernels = &t.on.affine;
    var m = try Matrix.init(t, rng, n, k, 4, 32, 1);
    defer m.deinit();
    const hx = try gpa.alloc(u16, rows * k);
    defer gpa.free(hx);
    for (hx) |*v| v.* = t.bits(rng.unit());
    var x = try t.upload(hx);
    defer x.free();
    var outs = [2]@import("../memory.zig").DeviceBuffer{ try t.alloc(rows * n * 4), try t.alloc(rows * n * 4) };
    defer for (&outs) |*b| b.free();
    var halves = [2]@import("../memory.zig").DeviceBuffer{ try t.alloc(rows * n * 2), try t.alloc(rows * n * 2) };
    defer for (&halves) |*b| b.free();
    try outs[1].fill8(0xA5);
    try halves[1].fill8(0xA5);
    var a = m.arg(t, at(x), rows, at(outs[0]));
    try kernels.blockGo(&t.r, a, t.stream.handle, 1, .matrix);
    try std.testing.expect(!std.mem.endsWith(u8, kernels.choose(.project, .prefill, affine.Kernels.shapeOf(a, 1)).?.id, "inexact"));
    a.out = at(outs[1]);
    try kernels.denseInexactGo(&t.r, a, t.stream.handle);
    try t.on.tf_cast(@ptrFromInt(at(outs[1])), 0, @ptrFromInt(at(halves[0])), t.kind(), @intCast(rows * n), t.stream.handle);
    a.out = 0;
    a.out16 = at(halves[1]);
    try kernels.denseInexactGo(&t.r, a, t.stream.handle);
    try t.stream.synchronize();
    const want = try rig.download(f32, outs[0]);
    defer gpa.free(want);
    const got = try rig.download(f32, outs[1]);
    defer gpa.free(got);
    var num: f64 = 0;
    var den: f64 = 0;
    for (want, got) |w, g| {
        num += (@as(f64, g) - w) * (@as(f64, g) - w);
        den += @as(f64, w) * w;
    }
    const rel = @sqrt(num / den);
    std.debug.print("  prompt dense inexact {d}x{d} rows {d}: relative RMS {e:.2} against the exact tile\n", .{ n, k, rows, rel });
    try std.testing.expect(rel <= 4e-3);
    const w16 = try rig.download(u16, halves[0]);
    defer gpa.free(w16);
    const g16 = try rig.download(u16, halves[1]);
    defer gpa.free(g16);
    try std.testing.expectEqualSlices(u16, w16, g16);
}

test "numerics=inexact: the dense tile with bf16-dequantized weights is within 4e-3 of the exact sums" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    if (t.on.affine.dense_inexact[0] == null) return error.SkipZigTest;
    var rng: rig.Rng = .{ .state = 0xBB67AE8584CAA73B };
    try promptDenseInexact(t, &rng, 300, 640, 2560);
    try promptDenseInexact(t, &rng, 257, 2560, 6144);
    try promptDenseInexact(t, &rng, 130, 200, 2560);
    try promptDenseInexact(t, &rng, 1024, 1024, 2560);
}

test "numerics=inexact: a prompt's router logits on the matrix cores sit within fp32 sum rounding of float64 (513 x 2560)" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    if (t.on.affine.router_inexact == null or !t.bf16) return error.SkipZigTest;
    var rng: rig.Rng = .{ .state = 0x510E527FADE682D1 };
    const d: usize = 2560;
    const e: usize = 513;
    for ([_]usize{ 300, 64, 257 }) |rows| {
        const hx = try gpa.alloc(u16, rows * d);
        defer gpa.free(hx);
        for (hx) |*v| v.* = rig.bf16Bits(rng.unit() * 2.0);
        const hw = try gpa.alloc(u16, e * d);
        defer gpa.free(hw);
        for (hw) |*v| v.* = rig.bf16Bits(rng.unit() / 8.0);
        var x = try t.upload(hx);
        defer x.free();
        var w = try t.upload(hw);
        defer w.free();
        var out = try t.alloc(rows * e * 4);
        defer out.free();
        try std.testing.expect(try t.on.affine.routerInexactGo(&t.r, at(x), at(w), at(out), @intCast(rows), @intCast(e), @intCast(d), t.stream.handle));
        try t.stream.synchronize();
        const got = try rig.download(f32, out);
        defer gpa.free(got);
        // every expert of every 7th row and the last: |got - sum| within 2^-14 of sum |x w| (2,560 fp32 adds in a chain
        // of 16-wide WMMA steps), and an exact product a term (bf16 x bf16 fits fp32)
        const wide = struct {
            fn of(b: u16) f64 {
                return @as(f32, @bitCast(@as(u32, b) << 16));
            }
        }.of;
        for (0..rows) |r| if (r % 7 == 0 or r + 1 == rows) for (0..e) |c| {
            var want: f64 = 0;
            var mag: f64 = 0;
            for (0..d) |k| {
                const term = wide(hx[r * d + k]) * wide(hw[c * d + k]);
                want += term;
                mag += @abs(term);
            }
            const err = @abs(@as(f64, got[r * e + c]) - want);
            if (err > mag * std.math.pow(f64, 2, -14) + 1e-30) {
                std.debug.print("router inexact rows={d} ({d}, {d}): {e} where float64 gives {e} (sum |terms| {e})\n", .{ rows, r, c, got[r * e + c], want, mag });
                return error.TestUnexpectedResult;
            }
        };
    }
}

test "numerics=inexact: Flash Next's prompt expert tiles with bf16-dequantized weights are within 2-3 bf16 steps of the exact bytes" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    if (t.on.affine.moe_gu_inexact == null) return error.SkipZigTest;
    var rng: rig.Rng = .{ .state = 0x3C6EF372FE94F82B };
    try promptExperts(t, &rng, false, 300, 24, false, 0, true);
    try promptExperts(t, &rng, false, 2048, 24, true, 0, true);
    try promptExperts(t, &rng, false, 512, 24, false, 7.0, true);
    try promptExperts(t, &rng, true, 300, 24, false, 0, true);
    try promptExperts(t, &rng, true, 2048, 24, true, 0, true);
}
