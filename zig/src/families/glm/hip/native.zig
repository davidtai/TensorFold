//! GLM-5.3-Flash for the native server on HIP: the checkpoint admitted (EXL3, the whole shape); no forward yet.
const std = @import("std");
const hip = @import("hip");
const lanes = @import("lanes");
const modelopt = @import("core").modelopt;
const config = @import("../config.zig");
const exl3_admit = @import("../exl3_admit.zig");

pub const model_type = "glm5_next";
/// Mia-AiLab's EXL3 checkpoint: 4-bit mcg routed experts, every other weight bf16 (exl3_admit.zig).
pub const formats: []const []const u8 = &.{"exl3"};
/// Positions a stream holds when neither --context nor the checkpoint's config names a window.
pub const default_context: i64 = 32768;
/// The engine cuts prompts itself.
pub const prefill_step: u32 = 0;

/// What the native server drives, as the other HIP families hand it over.
pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    streams: usize,
    startup: []const u8,
};

pub const Refused = error{Refused};

/// Why the HIP family stops after admission until the loader and the forward land.
pub const not_wired = "GLM-5.3-Flash on HIP: the forward is not wired yet";

fn refuse(a: std.mem.Allocator, problem: *[]const u8, comptime fmt: []const u8, args: anytype) Refused {
    problem.* = std.fmt.allocPrint(a, fmt, args) catch fmt;
    return error.Refused;
}

/// The checkpoint's config.json (`text`, read from `dir`) as the HIP family takes it: EXL3 4-bit mcg, the whole shape.
pub fn admit(a: std.mem.Allocator, gpa: std.mem.Allocator, dir: []const u8, text: []const u8, problem: *[]const u8) Refused!config.Config {
    if (!exl3_admit.isExl3(gpa, text)) return refuse(
        a,
        problem,
        "GLM-5.3-Flash on HIP reads EXL3 checkpoints (4-bit mcg routed experts, the rest bf16); {s}/config.json names no EXL3 quantization",
        .{dir},
    );
    var why: modelopt.Why = .{};
    exl3_admit.admit("HIP", gpa, text, &why) catch |err| return refuse(a, problem, "{s}/config.json: {s} ({t})", .{ dir, why.text(), err });
    const c = config.parse(gpa, text) catch |err| return refuse(a, problem, "{s} is not a GLM-5.3-Flash checkpoint ({t})", .{ dir, err });
    if (!config.whole(c)) return refuse(
        a,
        problem,
        "{s} has {d} layers, {d} experts and a {d}-token vocabulary: the HIP kernels are built for the whole GLM-5.3-Flash",
        .{ dir, c.layers, c.experts, c.vocab },
    );
    return c;
}

/// Admission from `dir`, before anything loads.
fn check(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, dir: []const u8, problem: *[]const u8) Refused!config.Config {
    const path = std.fs.path.join(a, &.{ dir, "config.json" }) catch return refuse(a, problem, "out of memory", .{});
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 24)) catch |err| return refuse(a, problem, "cannot read {s} ({t})", .{ path, err });
    return admit(a, gpa, dir, text, problem);
}

pub fn open(
    a: std.mem.Allocator,
    gpa: std.mem.Allocator,
    io: std.Io,
    dev: hip.Device,
    dir: []const u8,
    o: anytype,
    problem: *[]const u8,
) anyerror!Loaded {
    _ = dev;
    _ = o;
    _ = try check(a, gpa, io, dir, problem);
    return refuse(a, problem, not_wired, .{});
}

pub fn follow(
    a: std.mem.Allocator,
    gpa: std.mem.Allocator,
    io: std.Io,
    dev: hip.Device,
    dir: []const u8,
    o: anytype,
    problem: *[]const u8,
) anyerror!void {
    _ = dev;
    _ = o;
    _ = try check(a, gpa, io, dir, problem);
    return refuse(a, problem, not_wired, .{});
}

test "GLM on HIP admits EXL3 4-bit mcg at the whole shape and refuses the rest by name" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    try std.testing.expectError(error.Refused, admit(a, std.testing.allocator, "d", "{\"quantization_config\": {\"quant_method\": \"modelopt\"}}", &problem));
    try std.testing.expect(std.mem.indexOf(u8, problem, "reads EXL3 checkpoints") != null);
    try std.testing.expectError(error.Refused, admit(a, std.testing.allocator, "d", "{\"quantization_config\": {\"quant_method\": \"exl3\", \"bits\": 3, \"codebook\": \"mcg\"}}", &problem));
    try std.testing.expectStringStartsWith(problem, "d/config.json: EXL3 at 3 bits: the HIP family reads 4-bit EXL3 experts");
    try std.testing.expectError(error.Refused, admit(a, std.testing.allocator, "d", "{\"quantization_config\": {\"quant_method\": \"exl3\", \"bits\": 4, \"codebook\": \"mcg\"}}", &problem));
    try std.testing.expectStringStartsWith(problem, "d is not a GLM-5.3-Flash checkpoint");
}
