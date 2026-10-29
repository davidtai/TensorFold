//! A second HIP stream beside a step's own: `fork` gives it the step's work so far, `join` gives its work back.
//! Work on it runs beside the step's launches (a captured graph keeps the branch); one in-order stream serves every user.
const std = @import("std");
const abi = @import("abi.zig");
const runtime = @import("runtime.zig");
const util = @import("launches/util.zig");

/// Ordering events a step's forks and joins take in turn: each record in a captured step names its own event.
pub const ring_len = 128;

pub const Ring = struct {
    r: *const runtime.Runtime,
    events: [ring_len]abi.Event = @splat(null),
    next: usize = 0,

    pub fn init(r: *const runtime.Runtime) runtime.Error!Ring {
        var ring: Ring = .{ .r = r };
        errdefer ring.deinit();
        for (&ring.events) |*e| {
            try runtime.check(r.api.hipEventCreateWithFlags(e, abi.event_disable_timing));
            if (e.* == null) return error.Invalid;
        }
        return ring;
    }

    pub fn deinit(ring: *Ring) void {
        for (ring.events) |e| if (e != null) {
            _ = ring.r.api.hipEventDestroy(e);
        };
        ring.* = undefined;
    }

    fn take(ring: *Ring) abi.Event {
        const e = ring.events[ring.next];
        ring.next = (ring.next + 1) % ring_len;
        return e;
    }
};

/// The side stream and its events, as `Ops` carries them (both null: no side stream, `fork` and `join` do nothing).
pub const Side = struct {
    stream: abi.Stream = null,
    ring: ?*Ring = null,

    fn on(s: Side, main: abi.Stream) bool {
        return s.stream != null and s.ring != null and main != util.counting;
    }

    /// Later work on the side stream starts after `main`'s work so far.
    pub fn fork(s: Side, main: abi.Stream) runtime.Error!void {
        if (!s.on(main)) return;
        const ring = s.ring.?;
        const e = ring.take();
        try runtime.check(ring.r.api.hipEventRecord(e, main));
        try runtime.check(ring.r.api.hipStreamWaitEvent(s.stream, e, 0));
    }

    /// Later work on `main` starts after the side stream's work so far.
    pub fn join(s: Side, main: abi.Stream) runtime.Error!void {
        if (!s.on(main)) return;
        const ring = s.ring.?;
        const e = ring.take();
        try runtime.check(ring.r.api.hipEventRecord(e, s.stream));
        try runtime.check(ring.r.api.hipStreamWaitEvent(main, e, 0));
    }
};

test "fork records on the main stream and the side waits; join the other way; off without a side or when counting" {
    const Mock = struct {
        var log: [8]struct { what: u8, stream: usize, event: usize } = undefined;
        var n: usize = 0;
        fn record(e: abi.Event, s: abi.Stream) callconv(.c) c_int {
            log[n] = .{ .what = 'r', .stream = @intFromPtr(s), .event = @intFromPtr(e) };
            n += 1;
            return 0;
        }
        fn wait(s: abi.Stream, e: abi.Event, _: c_uint) callconv(.c) c_int {
            log[n] = .{ .what = 'w', .stream = @intFromPtr(s), .event = @intFromPtr(e) };
            n += 1;
            return 0;
        }
    };
    var r: runtime.Runtime = undefined;
    r.api.hipEventRecord = Mock.record;
    r.api.hipStreamWaitEvent = Mock.wait;
    var ring: Ring = .{ .r = &r };
    for (&ring.events, 0..) |*e, i| e.* = @ptrFromInt(0x1000 + i);
    const main: abi.Stream = @ptrFromInt(0x10);
    const side: Side = .{ .stream = @ptrFromInt(0x20), .ring = &ring };
    try side.fork(main);
    try side.join(main);
    try std.testing.expectEqual(@as(usize, 4), Mock.n);
    try std.testing.expectEqual(@as(u8, 'r'), Mock.log[0].what);
    try std.testing.expectEqual(@as(usize, 0x10), Mock.log[0].stream);
    try std.testing.expectEqual(@as(usize, 0x20), Mock.log[1].stream);
    try std.testing.expectEqual(Mock.log[0].event, Mock.log[1].event);
    try std.testing.expectEqual(@as(usize, 0x20), Mock.log[2].stream);
    try std.testing.expectEqual(@as(usize, 0x10), Mock.log[3].stream);
    try std.testing.expect(Mock.log[2].event != Mock.log[0].event);
    try (Side{}).fork(main);
    try side.join(util.counting);
    try std.testing.expectEqual(@as(usize, 4), Mock.n);
}
