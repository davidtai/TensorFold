//! What GLM's float64 contract tests share: bf16 values, the float64 product and its bound, the worst-fraction check.
const std = @import("std");
const rig = @import("../rig.zig");

pub const gpa = rig.gpa;

/// A bf16 bit pattern's value.
pub fn val(b: u16) f64 {
    return @as(f32, @bitCast(@as(u32, b) << 16));
}

/// The nearest bf16 of `v` (ties to even).
pub fn rnd(v: f64) u16 {
    const f: f32 = @floatCast(v);
    const u: u32 = @bitCast(f);
    return @intCast((u + (((u >> 16) & 1) + 0x7fff)) >> 16);
}

/// n bf16 values of `scale` times a standard normal draw.
pub fn normals(rng: *rig.Rng, n: usize, scale: f64) ![]u16 {
    const out = try gpa.alloc(u16, n);
    for (out) |*v| v.* = rnd(rng.normal() * scale);
    return out;
}

/// The worst |got - want| / tol of a check, named; RESULT line printed, the check fails past 1.
pub const Worst = struct {
    name: []const u8,
    most: f64 = 0,
    at: usize = 0,
    count: usize = 0,

    pub fn add(w: *Worst, i: usize, got: f64, want: f64, tol: f64) void {
        const q = @abs(got - want) / (tol + 1e-300);
        w.count += 1;
        if (!(q <= w.most)) { // NaN counts as the worst
            w.most = if (std.math.isNan(q)) std.math.inf(f64) else q;
            w.at = i;
        }
    }

    pub fn done(w: *const Worst) !void {
        std.debug.print("RESULT glm {s}: {d} values, worst {e:.3} of tolerance at {d}\n", .{ w.name, w.count, w.most, w.at });
        try rig.expect(w.most <= 1, "glm {s}: worst {e} of tolerance at {d}", .{ w.name, w.most, w.at });
    }
};

/// y = x w^T in float64 over bf16 rows (x [m][k], w [n][k]) with its bound: fp32 (k + 8) 2^-24 of sum |x w|.
pub const Dots = struct {
    want: []f64,
    tol: []f64,

    pub fn of(x: []const u16, w: []const u16, m: usize, n: usize, k: usize) !Dots {
        const want = try gpa.alloc(f64, m * n);
        const tol = try gpa.alloc(f64, m * n);
        const xd = try gpa.alloc(f64, k);
        defer gpa.free(xd);
        const wd = try gpa.alloc(f64, k);
        defer gpa.free(wd);
        const bound = @as(f64, @floatFromInt(k + 8)) * std.math.pow(f64, 2, -24);
        for (0..n) |j| {
            for (wd, w[j * k ..][0..k]) |*d, v| d.* = val(v);
            for (0..m) |row| {
                for (xd, x[row * k ..][0..k]) |*d, v| d.* = val(v);
                var sum: f64 = 0;
                var abs: f64 = 0;
                for (xd, wd) |a, b| {
                    sum += a * b;
                    abs += @abs(a * b);
                }
                want[row * n + j] = sum;
                tol[row * n + j] = abs * bound;
            }
        }
        return .{ .want = want, .tol = tol };
    }

    pub fn free(d: Dots) void {
        gpa.free(d.want);
        gpa.free(d.tol);
    }
};

pub fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

/// Microseconds a call of `f(ctx)` back to back on the rig's stream (10 warm-up calls), as a speed fact only.
pub fn timeUs(t: *rig.Rig, ctx: anytype, comptime f: fn (@TypeOf(ctx)) anyerror!void, calls: usize) !f64 {
    for (0..10) |_| try f(ctx);
    try t.stream.synchronize();
    const t0 = nowNs();
    for (0..calls) |_| try f(ctx);
    try t.stream.synchronize();
    return @as(f64, @floatFromInt(nowNs() - t0)) / @as(f64, @floatFromInt(calls)) / 1e3;
}
