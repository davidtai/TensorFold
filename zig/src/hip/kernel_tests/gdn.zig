//! The DeltaNet recurrence against a float64 recurrence on random inputs, and the same bits however a prompt is cut.
const std = @import("std");
const rig = @import("rig.zig");
const Rig = rig.Rig;
const gpa = rig.gpa;
const DeviceBuffer = @import("../memory.zig").DeviceBuffer;

const d = 128;
const Shape = struct { length: usize, key_heads: usize, value_heads: usize };

const Inputs = struct {
    q: []f32,
    k: []f32,
    v: []f32,
    gate: []f32,
    beta: []f32,
    state: []f32,

    /// L2-normalized q and k, gates exp(-A softplus(x)) in (0, 1], beta = sigmoid, a state of 0.1 scale.
    fn make(sh: Shape, seed: u64) !Inputs {
        var rng: rig.Rng = .{ .state = seed };
        const qk = sh.length * sh.key_heads * d;
        const vy = sh.length * sh.value_heads * d;
        const gb = sh.length * sh.value_heads;
        const in: Inputs = .{
            .q = try gpa.alloc(f32, qk),
            .k = try gpa.alloc(f32, qk),
            .v = try gpa.alloc(f32, vy),
            .gate = try gpa.alloc(f32, gb),
            .beta = try gpa.alloc(f32, gb),
            .state = try gpa.alloc(f32, sh.value_heads * d * d),
        };
        for ([_][]f32{ in.q, in.k }) |a| {
            for (0..a.len / d) |row| {
                var ss: f64 = 0;
                for (a[row * d ..][0..d]) |*x| {
                    const z = rng.normal();
                    x.* = @floatCast(z);
                    ss += z * z;
                }
                const inv: f32 = @floatCast(1 / @sqrt(ss));
                for (a[row * d ..][0..d]) |*x| x.* *= inv;
            }
        }
        for (in.v) |*x| x.* = @floatCast(rng.normal());
        for (in.state) |*x| x.* = @floatCast(0.1 * rng.normal());
        for (in.gate, in.beta, 0..) |*g, *b, i| {
            const h = i % sh.value_heads;
            const a_log = 0.5 + 7.5 * @as(f64, @floatFromInt(h)) / @as(f64, @floatFromInt(sh.value_heads));
            const x = rng.normal() - 2;
            const sp = if (x > 20) x else @log(1 + @exp(x));
            g.* = @floatCast(@exp(-a_log * sp));
            b.* = @floatCast(1 / (1 + @exp(-rng.normal())));
        }
        return in;
    }

    fn free(in: Inputs) void {
        for ([_][]f32{ in.q, in.k, in.v, in.gate, in.beta, in.state }) |a| gpa.free(a);
    }
};

/// The recurrence in float64: y (L, Hv, d) and the final state.
fn reference(sh: Shape, in: Inputs, y: []f64, state: []f64) !void {
    const s = try gpa.alloc(f64, d * d);
    defer gpa.free(s);
    const group = sh.value_heads / sh.key_heads;
    for (0..sh.value_heads) |h| {
        for (s, in.state[h * d * d ..][0 .. d * d]) |*a, b| a.* = b;
        const kh = h / group;
        for (0..sh.length) |t| {
            const gb = t * sh.value_heads + h;
            const decay: f64 = in.gate[gb];
            const beta: f64 = in.beta[gb];
            const k = in.k[(t * sh.key_heads + kh) * d ..][0..d];
            const q = in.q[(t * sh.key_heads + kh) * d ..][0..d];
            const v = in.v[gb * d ..][0..d];
            for (0..d) |r| {
                const row = s[r * d ..][0..d];
                var kv: f64 = 0;
                for (row, k) |*a, kk| {
                    a.* *= decay;
                    kv += a.* * kk;
                }
                const delta = (v[r] - kv) * beta;
                var out: f64 = 0;
                for (row, k, q) |*a, kk, qq| {
                    a.* += kk * delta;
                    out += a.* * qq;
                }
                y[gb * d + r] = out;
            }
        }
        @memcpy(state[h * d * d ..][0 .. d * d], s);
    }
}

/// The largest error over the reference's largest value.
fn relative(got: []const f32, want: []const f64) f64 {
    var scale: f64 = 0;
    var max: f64 = 0;
    for (got, want) |g, w| {
        max = @max(max, @abs(@as(f64, g) - w));
        scale = @max(scale, @abs(w));
    }
    return max / scale;
}

/// One call of `length` rows from row `from`, state in place; q, k, v, gate and beta are offset to that row.
fn run(
    t: *Rig,
    sh: Shape,
    q: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    gate: DeviceBuffer,
    beta: DeviceBuffer,
    state: DeviceBuffer,
    y: DeviceBuffer,
    from: usize,
    length: usize,
) !void {
    const qk = from * sh.key_heads * d * 4;
    const vy = from * sh.value_heads * d * 4;
    const gb = from * sh.value_heads * 4;
    try t.ops(&t.on).gatedDelta(
        rig.at(q) + qk,
        rig.at(k) + qk,
        rig.at(v) + vy,
        rig.at(gate) + gb,
        rig.at(beta) + gb,
        rig.at(state),
        rig.at(y) + vy,
        length,
        sh.key_heads,
        sh.value_heads,
        d,
        d,
        null,
        0,
    );
}

test "the DeltaNet stays within 5e-3 of the float64 recurrence at four lengths" {
    const t = try Rig.open(1 << 30);
    defer t.close();
    for ([_]Shape{
        .{ .length = 64, .key_heads = 2, .value_heads = 4 },
        .{ .length = 200, .key_heads = 2, .value_heads = 4 },
        .{ .length = 1000, .key_heads = 2, .value_heads = 4 },
        .{ .length = 4170, .key_heads = 2, .value_heads = 4 },
    }) |sh| {
        const in = try Inputs.make(sh, 0x9e3779b97f4a7c15 + sh.length);
        defer in.free();
        const ry = try gpa.alloc(f64, sh.length * sh.value_heads * d);
        defer gpa.free(ry);
        const rs = try gpa.alloc(f64, sh.value_heads * d * d);
        defer gpa.free(rs);
        try reference(sh, in, ry, rs);
        var q = try t.upload(in.q);
        defer q.free();
        var k = try t.upload(in.k);
        defer k.free();
        var v = try t.upload(in.v);
        defer v.free();
        var gate = try t.upload(in.gate);
        defer gate.free();
        var beta = try t.upload(in.beta);
        defer beta.free();
        var y = try t.alloc(ry.len * 4);
        defer y.free();
        var state = try t.upload(in.state);
        defer state.free();
        try run(t, sh, q, k, v, gate, beta, state, y, 0, sh.length);
        try t.stream.synchronize();
        const gy = try rig.download(f32, y);
        defer gpa.free(gy);
        const gs = try rig.download(f32, state);
        defer gpa.free(gs);
        try std.testing.expect(relative(gy, ry) < 5e-3);
        try std.testing.expect(relative(gs, rs) < 5e-3);
    }
}

test "a prompt cut on either side of 64 rows gives the same rows and state as one call" {
    const t = try Rig.open(1 << 30);
    defer t.close();
    const sh: Shape = .{ .length = 200, .key_heads = 2, .value_heads = 4 };
    const in = try Inputs.make(sh, 0x51ed);
    defer in.free();
    var q = try t.upload(in.q);
    defer q.free();
    var k = try t.upload(in.k);
    defer k.free();
    var v = try t.upload(in.v);
    defer v.free();
    var gate = try t.upload(in.gate);
    defer gate.free();
    var beta = try t.upload(in.beta);
    defer beta.free();
    var y = try t.alloc(sh.length * sh.value_heads * d * 4);
    defer y.free();
    var state = try t.upload(in.state);
    defer state.free();
    try run(t, sh, q, k, v, gate, beta, state, y, 0, sh.length);
    try t.stream.synchronize();
    const wy = try rig.download(f32, y);
    defer gpa.free(wy);
    const ws = try rig.download(f32, state);
    defer gpa.free(ws);
    for ([_]usize{ 1, 63, 64, 65, 136 }) |c| {
        try state.upload(0, std.mem.sliceAsBytes(in.state));
        try y.fill8(0);
        try run(t, sh, q, k, v, gate, beta, state, y, 0, c);
        try run(t, sh, q, k, v, gate, beta, state, y, c, sh.length - c);
        try t.stream.synchronize();
        const gy = try rig.download(f32, y);
        defer gpa.free(gy);
        const gs = try rig.download(f32, state);
        defer gpa.free(gs);
        try std.testing.expectEqualSlices(u32, @ptrCast(wy), @ptrCast(gy));
        try std.testing.expectEqualSlices(u32, @ptrCast(ws), @ptrCast(gs));
    }
}

/// One window of `length` rows from row `at` with each row's state kept in `snaps`, starting where `from` (a device
/// i32, 0: none) says: -1 the state, row f the snapshot after row f of the last window in `snaps`.
fn window(t: *Rig, sh: Shape, in: [5]DeviceBuffer, state: DeviceBuffer, y: DeviceBuffer, snaps: DeviceBuffer, from: u64, at: usize, length: usize) !void {
    const q, const k, const v, const gate, const beta = in;
    const qk = at * sh.key_heads * d * 4;
    const vy = at * sh.value_heads * d * 4;
    const gb = at * sh.value_heads * 4;
    try t.ops(&t.on).gatedDelta(rig.at(q) + qk, rig.at(k) + qk, rig.at(v) + vy, rig.at(gate) + gb, rig.at(beta) + gb, rig.at(state), rig.at(y) + vy, length, sh.key_heads, sh.value_heads, d, d, rig.at(snaps), from);
}

test "a window started from the last window's snapshot row gives the bits of that snapshot copied back" {
    const t = try Rig.open(1 << 30);
    defer t.close();
    const sh: Shape = .{ .length = 8, .key_heads = 2, .value_heads = 4 };
    const rows = 4;
    const in = try Inputs.make(sh, 0x5747e);
    defer in.free();
    var bufs: [5]DeviceBuffer = undefined;
    for (&bufs, [_][]f32{ in.q, in.k, in.v, in.gate, in.beta }) |*b, a| b.* = try t.upload(a);
    defer for (&bufs) |*b| b.free();
    const state_bytes = sh.value_heads * d * d * 4;
    var y = try t.alloc(sh.length * sh.value_heads * d * 4);
    defer y.free();
    for ([_]i32{ -1, 0, 1, 2, 3 }) |f| {
        // the first window: rows 0-3 from the input state, each row's state into `snaps`
        var state = try t.upload(in.state);
        defer state.free();
        var snaps = try t.alloc(rows * state_bytes);
        defer snaps.free();
        var none = try t.upload(&[_]i32{-1});
        defer none.free();
        try window(t, sh, bufs, state, y, snaps, rig.at(none), 0, rows);
        try t.stream.synchronize();
        // the copy: the start's bytes in a state of their own, then rows 4-7 from it
        const first = try rig.download(u8, snaps);
        defer gpa.free(first);
        const after = try rig.download(u8, state);
        defer gpa.free(after);
        const start = if (f < 0) after else first[@as(usize, @intCast(f)) * state_bytes ..][0..state_bytes];
        var copied = try t.upload(start);
        defer copied.free();
        var copied_snaps = try t.alloc(rows * state_bytes);
        defer copied_snaps.free();
        try y.fill8(0);
        try window(t, sh, bufs, copied, y, copied_snaps, rig.at(none), rows, rows);
        try t.stream.synchronize();
        const want_y = try rig.download(u8, y);
        defer gpa.free(want_y);
        // the start row: rows 4-7 from the state the first window left, `from` naming row f of its snapshots in place
        var word = try t.upload(&[_]i32{f});
        defer word.free();
        try y.fill8(0);
        try window(t, sh, bufs, state, y, snaps, rig.at(word), rows, rows);
        try t.stream.synchronize();
        const got_y = try rig.download(u8, y);
        defer gpa.free(got_y);
        try std.testing.expectEqualSlices(u8, want_y, got_y);
        for ([_][2]DeviceBuffer{ .{ copied, state }, .{ copied_snaps, snaps } }) |pair| {
            const want = try rig.download(u8, pair[0]);
            defer gpa.free(want);
            const got = try rig.download(u8, pair[1]);
            defer gpa.free(got);
            try std.testing.expectEqualSlices(u8, want, got);
        }
    }
}

test "a prompt's recurrence a block a value head gives the bits of a wave a row (Flash Next: 16 key, 48 value heads)" {
    const t = try Rig.open(1 << 30);
    defer t.close();
    for ([_]Shape{
        .{ .length = 64, .key_heads = 2, .value_heads = 4 },
        .{ .length = 333, .key_heads = 2, .value_heads = 4 },
        .{ .length = 300, .key_heads = 16, .value_heads = 48 },
    }) |sh| {
        const in = try Inputs.make(sh, 0xB10C + sh.length);
        defer in.free();
        var bufs: [5]DeviceBuffer = undefined;
        for (&bufs, [_][]f32{ in.q, in.k, in.v, in.gate, in.beta }) |*b, a| b.* = try t.upload(a);
        defer for (&bufs) |*b| b.free();
        const state_bytes = sh.value_heads * d * d * 4;
        var got: [2][2][]u8 = undefined;
        for (0..2) |w| {
            var state = try t.upload(in.state);
            defer state.free();
            var y = try t.alloc(sh.length * sh.value_heads * d * 4);
            defer y.free();
            try y.fill8(0xff);
            // a wave a row: the same launch with snapshots asked for; a block a value head: none
            var snaps = try t.alloc(if (w == 0) sh.length * state_bytes else 4);
            defer snaps.free();
            const F = ?[*]f32;
            try t.on.tf_gated_delta(@ptrFromInt(rig.at(bufs[0])), @ptrFromInt(rig.at(bufs[1])), @ptrFromInt(rig.at(bufs[2])), @ptrFromInt(rig.at(bufs[3])), @ptrFromInt(rig.at(bufs[4])), @ptrFromInt(rig.at(state)), @ptrFromInt(rig.at(y)), 1, @intCast(sh.length), @intCast(sh.key_heads), @intCast(sh.value_heads), d, d, t.stream.handle, if (w == 0) @as(F, @ptrFromInt(rig.at(snaps))) else null, 0);
            try t.stream.synchronize();
            got[w][0] = try rig.download(u8, y);
            got[w][1] = try rig.download(u8, state);
        }
        defer for (&got) |*g| for (g) |b| gpa.free(b);
        for (0..2) |k| if (!std.mem.eql(u8, got[0][k], got[1][k])) {
            std.debug.print("gdn prompt length={d} heads={d}: buffer {d} differs\n", .{ sh.length, sh.value_heads, k });
            return error.TestUnexpectedResult;
        };
    }
}
