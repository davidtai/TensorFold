//! G4-1 and G4-5: the mHC kernels' bits do not depend on how the work is cut. The mix with its 24 mixes split over 2
//! and 3 blocks gives the 1-block partials; hc pre as served (the split by row count) gives the 1-block mix and the
//! finish; the post on row cuts gives the post on all rows. Inputs mix large and small magnitudes (2^-12 .. 2^12) so a
//! changed order of any sum shows as a rounding difference. The values against float64 are hc.zig's tests. Then speed
//! facts (no gate) of each kernel at 4 and 2,048 rows.
const std = @import("std");
const rig = @import("../rig.zig");
const check = @import("check.zig");
const glm = @import("../../launches/glm.zig");
const DeviceBuffer = @import("../../memory.zig").DeviceBuffer;

const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;
const D = glm.hidden;
const S = glm.streams;
const W = S * D;
const mixes = 24;
const eps: f32 = 1e-5;
const hc_eps: f32 = 1e-6;
const iters = 20;

fn p(b: DeviceBuffer) ?*anyopaque {
    return @ptrFromInt(at(b));
}

/// n bf16 values, each a normal draw times 2^e, e uniform in [-12, 12]: wide magnitudes in one sum.
fn wide(rng: *rig.Rng, n: usize, scale: f64) ![]u16 {
    const out = try gpa.alloc(u16, n);
    for (out) |*v| {
        const e: i32 = @as(i32, @intCast(rng.next() % 25)) - 12;
        v.* = check.rnd(rng.normal() * scale * std.math.pow(f64, 2, @floatFromInt(e)));
    }
    return out;
}

fn same(name: []const u8, a: DeviceBuffer, b: DeviceBuffer, bytes: usize) !void {
    const x = try gpa.alloc(u8, bytes);
    defer gpa.free(x);
    const y = try gpa.alloc(u8, bytes);
    defer gpa.free(y);
    try a.download(0, x);
    try b.download(0, y);
    if (std.mem.findDiff(u8, x, y)) |i| {
        std.debug.print("glm mhc {s}: byte {d} differs (0x{x} against 0x{x})\n", .{ name, i, x[i], y[i] });
        return error.TestUnexpectedResult;
    }
}

/// One hc pre case: streams, mixing weights, base, scale, the norm weight; outputs twice (two ways to the same bits).
const Pre = struct {
    rows: usize,
    x: DeviceBuffer,
    fnw: DeviceBuffer,
    base: DeviceBuffer,
    scale: DeviceBuffer,
    nw: DeviceBuffer,
    part: [2]DeviceBuffer,
    out: [2]DeviceBuffer,
    post: [2]DeviceBuffer,
    comb: [2]DeviceBuffer,

    fn init(t: *Rig, rng: *rig.Rng, rows: usize, sd: f64) !Pre {
        var c: Pre = undefined;
        c.rows = rows;
        const hx = try wide(rng, rows * W, 1.0);
        defer gpa.free(hx);
        const hfn = try wide(rng, mixes * W, 0.02);
        defer gpa.free(hfn);
        var base: [mixes]f32 = undefined;
        for (&base) |*b| b.* = @floatCast(rng.normal() * sd);
        const scale = [3]f32{ @floatCast(0.5 + rng.uniform()), @floatCast(0.5 + rng.uniform()), @floatCast(sd * (0.5 + rng.uniform())) };
        const hnw = try check.normals(rng, D, 1.0);
        defer gpa.free(hnw);
        c.x = try t.upload(hx);
        c.fnw = try t.upload(hfn);
        c.base = try t.upload(&base);
        c.scale = try t.upload(&scale);
        c.nw = try t.upload(hnw);
        for (0..2) |i| {
            c.part[i] = try t.alloc(glm.hcPartBytes(rows));
            try c.part[i].fill8(0x5a);
            c.out[i] = try t.alloc(rows * D * 2);
            c.post[i] = try t.alloc(rows * S * 4);
            c.comb[i] = try t.alloc(rows * S * S * 4);
        }
        return c;
    }

    fn deinit(c: *Pre) void {
        for ([_]*DeviceBuffer{ &c.x, &c.fnw, &c.base, &c.scale, &c.nw }) |b| b.free();
        for (0..2) |i| for ([_]*DeviceBuffer{ &c.part[i], &c.out[i], &c.post[i], &c.comb[i] }) |b| b.free();
    }
};

test "glm mhc (G4-5): the mix split over 2 and 3 blocks gives the 1-block partials bit for bit; hc pre as served gives the 1-block mix and the finish" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x4C5A_0001 };
    var cases: usize = 0;
    for ([_]usize{ 1, 2, 3, 4, 5, 70, 2048 }) |rows| for ([_]f64{ 0.5, 6.0 }) |sd| {
        var c = try Pre.init(t, &rng, rows, sd);
        defer c.deinit();
        const r: c_int = @intCast(rows);
        try glm.hcMix(&t.on, p(c.x), p(c.fnw), p(c.part[0]), r, 1, t.stream.handle);
        for ([_]c_int{ 2, 3 }) |parts| {
            try glm.hcMix(&t.on, p(c.x), p(c.fnw), p(c.part[1]), r, parts, t.stream.handle);
            try t.stream.synchronize();
            var name: [64]u8 = undefined;
            try same(try std.fmt.bufPrint(&name, "mix {d} rows, {d} parts", .{ rows, parts }), c.part[0], c.part[1], glm.hcPartBytes(rows));
        }
        // the finish on the 1-block partials into [1]; hcPre as served into [0]
        try glm.hcFinish(&t.on, p(c.x), p(c.part[0]), p(c.base), p(c.scale), p(c.nw), p(c.out[1]), p(c.post[1]), p(c.comb[1]), eps, hc_eps, iters, r, t.stream.handle);
        try glm.hcPre(&t.on, p(c.x), p(c.fnw), p(c.base), p(c.scale), p(c.nw), p(c.out[0]), p(c.post[0]), p(c.comb[0]), p(c.part[1]), eps, hc_eps, iters, r, t.stream.handle);
        try t.stream.synchronize();
        var name: [64]u8 = undefined;
        try same(try std.fmt.bufPrint(&name, "hc pre {d} rows: normed", .{rows}), c.out[0], c.out[1], rows * D * 2);
        try same(try std.fmt.bufPrint(&name, "hc pre {d} rows: post", .{rows}), c.post[0], c.post[1], rows * S * 4);
        try same(try std.fmt.bufPrint(&name, "hc pre {d} rows: comb", .{rows}), c.comb[0], c.comb[1], rows * S * S * 4);
        cases += 1;
    };
    std.debug.print("RESULT glm mhc pre: {d} cases (rows 1-5, 70, 2,048; two logit scales), split mixes and hc pre as served bit-equal to the 1-block mix\n", .{cases});
}

test "glm mhc (G4-1): the post on row cuts gives the post on all rows bit for bit, 1 and 2 partials, in place" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x4C5A_0002 };
    var cases: usize = 0;
    for ([_]usize{ 4, 70, 2048 }) |rows| for ([_]c_int{ 1, 2 }) |parts| {
        const hx = try wide(&rng, rows * W, 1.0);
        defer gpa.free(hx);
        const hg = try gpa.alloc(f32, 2 * rows * D);
        defer gpa.free(hg);
        for (hg) |*v| v.* = @floatCast(rng.normal() * std.math.pow(f64, 2, @floatFromInt(@as(i32, @intCast(rng.next() % 25)) - 12)));
        const hp = try gpa.alloc(f32, rows * S);
        defer gpa.free(hp);
        for (hp) |*v| v.* = @floatCast(2 / (1 + @exp(-rng.normal())));
        const hc = try gpa.alloc(f32, rows * S * S);
        defer gpa.free(hc);
        for (hc) |*v| v.* = @floatCast(rng.uniform());
        var xs: [2]DeviceBuffer = .{ try t.upload(hx), try t.upload(hx) };
        defer for (&xs) |*b| b.free();
        var g = try t.upload(hg);
        defer g.free();
        var post = try t.upload(hp);
        defer post.free();
        var comb = try t.upload(hc);
        defer comb.free();
        const stride: i64 = @intCast(rows * D);
        try glm.hcPost(&t.on, p(xs[0]), p(xs[0]), p(g), parts, stride, p(post), p(comb), @intCast(rows), t.stream.handle);
        // cuts of 1, 3 and the rest
        const cuts = [_]usize{ 0, 1, 4, rows };
        for (cuts[0 .. cuts.len - 1], cuts[1..]) |lo, hi| {
            if (lo >= hi) continue;
            const xc: ?*anyopaque = @ptrFromInt(at(xs[1]) + lo * W * 2);
            try glm.hcPost(&t.on, xc, xc, @ptrFromInt(at(g) + lo * D * 4), parts, stride, @ptrFromInt(at(post) + lo * S * 4), @ptrFromInt(at(comb) + lo * S * S * 4), @intCast(hi - lo), t.stream.handle);
        }
        try t.stream.synchronize();
        var name: [64]u8 = undefined;
        try same(try std.fmt.bufPrint(&name, "post {d} rows, {d} partials", .{ rows, parts }), xs[0], xs[1], rows * W * 2);
        cases += 1;
    };
    std.debug.print("RESULT glm mhc post: {d} cases (rows 4, 70, 2,048; 1 and 2 partials, in place), row cuts bit-equal to all rows\n", .{cases});
}

const Which = enum { pre, mix, mix1, finish, post };

const Timed = struct {
    t: *Rig,
    c: *const Pre,
    which: Which,
    g: u64,

    fn call(s: Timed) anyerror!void {
        const t = s.t;
        const c = s.c;
        const r: c_int = @intCast(c.rows);
        const h = t.stream.handle;
        switch (s.which) {
            .pre => try glm.hcPre(&t.on, p(c.x), p(c.fnw), p(c.base), p(c.scale), p(c.nw), p(c.out[0]), p(c.post[0]), p(c.comb[0]), p(c.part[0]), eps, hc_eps, iters, r, h),
            .mix => try glm.hcMix(&t.on, p(c.x), p(c.fnw), p(c.part[0]), r, glm.mixParts(r), h),
            .mix1 => try glm.hcMix(&t.on, p(c.x), p(c.fnw), p(c.part[0]), r, 1, h),
            .finish => try glm.hcFinish(&t.on, p(c.x), p(c.part[0]), p(c.base), p(c.scale), p(c.nw), p(c.out[0]), p(c.post[0]), p(c.comb[0]), eps, hc_eps, iters, r, h),
            .post => try glm.hcPost(&t.on, p(c.x), p(c.x), @ptrFromInt(s.g), 1, 0, p(c.post[0]), p(c.comb[0]), r, h),
        }
    }
};

test "glm mhc (G4-7): the mix's DPP butterfly gives the __shfl_xor butterfly's partials bit for bit at 4, 8, 33 and 2,048 rows" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x4C5A_0007 };
    for ([_]usize{ 4, 8, 33, 2048 }) |rows| {
        var c = try Pre.init(t, &rng, rows, 0.5);
        defer c.deinit();
        const r: c_int = @intCast(rows);
        try glm.hcMix(&t.on, p(c.x), p(c.fnw), p(c.part[0]), r, glm.mixParts(r), t.stream.handle);
        try glm.hcMixShfl(&t.on, p(c.x), p(c.fnw), p(c.part[1]), r, t.stream.handle);
        try t.stream.synchronize();
        try same("mix, DPP against __shfl_xor butterfly", c.part[0], c.part[1], glm.hcPartBytes(rows));
    }
    std.debug.print("RESULT glm mhc mix: the DPP butterfly's partials equal the __shfl_xor butterfly's at 4, 8, 33 and 2,048 rows\n", .{});
}

test "glm mhc (G4-1, G4-5, G4-3): speed facts (no gate) of post, pre, mix and finish at 4 and 2,048 rows" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x4C5A_0003 };
    for ([_]usize{ 4, 2048 }) |rows| {
        var c = try Pre.init(t, &rng, rows, 0.5);
        defer c.deinit();
        var g = try t.alloc(rows * D * 4);
        defer g.free();
        try g.fill8(0x3c);
        // a first pre so post and comb hold real values
        try glm.hcPre(&t.on, p(c.x), p(c.fnw), p(c.base), p(c.scale), p(c.nw), p(c.out[0]), p(c.post[0]), p(c.comb[0]), p(c.part[0]), eps, hc_eps, iters, @intCast(rows), t.stream.handle);
        const calls: usize = if (rows > 64) 20 else 200;
        var us: [5]f64 = undefined;
        for ([_]Which{ .pre, .mix, .mix1, .finish, .post }, 0..) |w, i| {
            us[i] = try check.timeUs(t, Timed{ .t = t, .c = &c, .which = w, .g = at(g) }, Timed.call, calls);
        }
        std.debug.print("SPEED glm mhc {d} rows: pre {d:.1} us (mix {d:.1} at {d} parts, 1 part {d:.1}; finish {d:.1}); post {d:.1} us\n", .{ rows, us[0], us[1], glm.mixParts(@intCast(rows)), us[2], us[3], us[4] });
    }
}
