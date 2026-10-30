//! A GLM-5.3-Flash checkpoint directory to weights on the device, in two calls: `plan` (config.json admitted, the
//! headers read, the placement; no GPU) and `open` (plan, then the streaming load). The forward takes `Opened.model`.

const std = @import("std");
const hip = @import("hip");
const config = @import("../config.zig");
const native = @import("native.zig");
const shards = @import("shards.zig");
const layout = @import("layout.zig");
const load = @import("load.zig");
const weights = @import("weights.zig");

/// The checkpoint as the placement reads it: config, headers, layout (all in `arena`).
pub const Checkpoint = struct {
    arena: std.heap.ArenaAllocator,
    config: config.Config,
    index: shards.Index,
    layout: layout.Layout,

    pub fn deinit(c: *Checkpoint) void {
        const gpa = c.arena.child_allocator;
        c.arena.deinit();
        gpa.destroy(c);
    }
};

/// `dir`'s config.json admitted (EXL3 4-bit mcg, the whole shape), its headers, the placement; `problem` (in `a`)
/// says why not.
pub fn plan(gpa: std.mem.Allocator, a: std.mem.Allocator, io: std.Io, dir: []const u8, problem: *[]const u8) !*Checkpoint {
    const c = try gpa.create(Checkpoint);
    errdefer gpa.destroy(c);
    c.arena = std.heap.ArenaAllocator.init(gpa);
    errdefer c.arena.deinit();
    const ca = c.arena.allocator();
    const path = try std.fs.path.join(ca, &.{ dir, "config.json" });
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, ca, .limited(1 << 24)) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "cannot read {s} ({t})", .{ path, err });
        return error.Refused;
    };
    c.config = try native.admit(a, gpa, dir, text, problem);
    c.index = shards.read(ca, io, dir) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "cannot read {s}'s safetensors headers ({t})", .{ dir, err });
        return error.Refused;
    };
    var why: []const u8 = "";
    c.layout = layout.plan(ca, &c.index, c.config, &why) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "{s}: the HIP placement refuses {s} ({t})", .{ dir, why, err });
        return error.Refused;
    };
    return c;
}

/// The checkpoint and its weights on the current device.
pub const Opened = struct {
    ck: *Checkpoint,
    w: load.Loaded,

    pub fn model(o: *const Opened) *const weights.Model {
        return &o.w.model;
    }

    pub fn deinit(o: *Opened) void {
        o.w.deinit();
        o.ck.deinit();
        o.* = undefined;
    }
};

/// `plan`, then the load onto the device current on this thread (`r`'s).
pub fn open(gpa: std.mem.Allocator, a: std.mem.Allocator, io: std.Io, r: *const hip.Runtime, dir: []const u8, o: load.Options, problem: *[]const u8) !Opened {
    const ck = try plan(gpa, a, io, dir, problem);
    errdefer ck.deinit();
    const w = load.load(gpa, r, &ck.index, &ck.layout, o) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "loading {s} onto the GPU failed ({t})", .{ dir, err });
        return err;
    };
    return .{ .ck = ck, .w = w };
}
