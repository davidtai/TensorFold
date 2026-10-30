//! GLM-5.3-Flash for the native server on HIP: the checkpoint admitted (EXL3, the whole shape), loaded, served by the
//! lane core over lane F's forward (serve.zig): one device, serial decode.
const std = @import("std");
const hip = @import("hip");
const lanes = @import("lanes");
const modelopt = @import("core").modelopt;
const config = @import("../config.zig");
const exl3_admit = @import("../exl3_admit.zig");
const load = @import("load.zig");
const serve = @import("serve.zig");
const state = @import("state.zig");

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

/// The prompt rows a forward takes at once where the policy names none (the HIP default is 1,024): state.prompt_rows
/// (8,192; +7 % prefill on the 16K coding workload against 2,048, scratch 4.44 GiB). `--policy prefill_step=` takes
/// a smaller chunk. A row's bytes do not depend on the chunk (contract 7).
pub const policy_prefill_step: u32 = state.prompt_rows;
/// Drafts a round holds at most (the MTP layer's chain, lane F's `draft`): --mtp-drafts past it is refused before the
/// load.
pub const max_drafts: u32 = serve.max_drafts;
/// The MTP policy where neither --mtp-drafts, --mtp-confidence nor the mtp_depth key names one (the HIP default is 3
/// drafts under 0.3): 3 drafts, the lane core's depth rule, no confidence floor (the harness sweep's
/// `--mtp-drafts N --mtp-confidence 0`).
pub const policy_mtp: struct { drafts: u8, confidence: f64, depth: hip.policy.MtpDepth } = .{ .drafts = 3, .confidence = 0, .depth = .rule };

/// Why `policy`'s MTP settings cannot serve GLM-5.3-Flash on HIP, or null. The draft head gives no probabilities yet,
/// so a confidence floor (--mtp-confidence past 0, or the HIP default's 0.3 under --mtp-drafts alone) and a gated
/// chain are refused; --mtp-confidence 0 is no early stop.
pub fn mtpRefused(policy: hip.Policy) ?[]const u8 {
    if (policy.mtp.drafts == 0) return null;
    if (policy.mtp.confidence > 0)
        return "--mtp-confidence: GLM-5.3-Flash on HIP drafts a fixed chain (its draft head gives no probabilities " ++
            "yet); serve it with --mtp-confidence 0 (with --mtp-drafts alone the HIP default's 0.3 applies), or " ++
            "with neither flag";
    if (policy.mtp_depth == .gated)
        return "mtp_depth=gated: GLM-5.3-Flash on HIP drafts a fixed chain (its draft head gives no probabilities yet); " ++
            "serve it with the depth rule (no mtp_depth key, or mtp_depth=rule)";
    return null;
}

/// The GLM forward launches the exact kernels only (lane F's launcher, numerics=exact): another policy is refused.
pub const inexact_refusal = "GLM-5.3-Flash on HIP runs numerics=exact only (no inexact path yet): serve it without --policy numerics=inexact";

/// Why `policy` cannot serve GLM-5.3-Flash on HIP, or null.
pub fn numericsRefused(policy: hip.Policy) ?[]const u8 {
    return if (policy.numerics != .exact) inexact_refusal else null;
}

/// MemAvailable the streams leave at least: the load's floor (the harness watchdog stops a server under 6 GiB).
const floor: u64 = (load.Options{}).floor;

fn gib(n: u64) f64 {
    return @as(f64, @floatFromInt(n)) / (1 << 30);
}

/// The engine and the lane backend `open` hands the server, freed together.
const Owned = struct {
    gpa: std.mem.Allocator,
    e: *serve.Engine,
    h: *serve.Hip,
    startup: []const u8,

    fn release(p: *anyopaque) void {
        const o: *Owned = @ptrCast(@alignCast(p));
        o.h.deinit();
        o.e.deinit();
        o.gpa.free(o.startup);
        o.gpa.destroy(o);
    }
};

/// The engine for `dir` on `dev`: admitted, loaded, its streams sized by what MemAvailable leaves over the floor.
pub fn open(
    a: std.mem.Allocator,
    gpa: std.mem.Allocator,
    io: std.Io,
    dev: hip.Device,
    dir: []const u8,
    o: anytype,
    problem: *[]const u8,
) anyerror!Loaded {
    _ = try check(a, gpa, io, dir, problem);
    if (numericsRefused(dev.policy)) |why| return refuse(a, problem, "{s}", .{why});
    if (mtpRefused(dev.policy)) |why| return refuse(a, problem, "{s}", .{why});
    if (o.world > 1) return refuse(a, problem, "GLM-5.3-Flash on HIP runs on one device: serve it with --tp 1", .{});
    if (o.cache_gib) |g| if (g > 0) return refuse(a, problem, "GLM-5.3-Flash on HIP keeps no prompt cache yet: serve it with --prompt-cache-gib 0", .{});
    const capacity = std.mem.alignForward(usize, o.window + 8, 64); // cuda_state.zig's align64(context + 8)
    const chunk: usize = @min(@max(dev.policy.prefill_step, 64), state.prompt_rows);
    problem.* = "";
    const drafts: usize = @min(dev.policy.mtp.drafts, serve.max_drafts);
    const e = serve.Engine.create(gpa, a, io, dev.index, dir, capacity, chunk, drafts, problem) catch |err| {
        if (problem.len == 0) return refuse(a, problem, "the native HIP engine cannot load GLM-5.3-Flash from {s} ({t})", .{ dir, err });
        return error.Refused;
    };
    errdefer e.deinit();
    const weights = e.opened.ck.layout.slabBytes();
    const scratch = e.scratchBytes();
    const per: u64 = state.State.bytes(capacity);
    const before = load.memAvailable() orelse return refuse(a, problem, "cannot read MemAvailable", .{});
    const usable = before -| floor;
    var streams: usize = @max(o.streams, 1);
    while (streams > 1 and streams * per > usable) streams -= 1;
    if (streams * per > usable) return refuse(
        a,
        problem,
        "one GLM-5.3-Flash stream at a {d}-token window takes {d:.2} GiB, and MemAvailable is {d:.2} GiB after the " ++
            "model's {d:.2} GiB and {d:.2} GiB of scratch, under the {d:.0} GiB floor with it: lower --context",
        .{ o.window, gib(per), gib(before), gib(weights), gib(scratch), gib(floor) },
    );
    if (streams < o.streams and o.fixed) return refuse(
        a,
        problem,
        "--parallel {d} needs {d:.2} GiB at a {d}-token window and MemAvailable leaves {d:.2} GiB over the {d:.0} " ++
            "GiB floor: serve --parallel {d}, or lower --context",
        .{ o.streams, gib(o.streams * per), o.window, gib(usable), gib(floor), streams },
    );
    const h = serve.Hip.init(gpa, e, streams) catch |err| return refuse(a, problem, "the native HIP engine cannot make GLM's streams ({t})", .{err});
    errdefer h.deinit();
    const after = load.memAvailable() orelse 0;
    const startup = try std.fmt.allocPrint(gpa, "HIP {s} device {d}: GLM-5.3-Flash weights {d:.3} GiB (loaded in " ++
        "{d:.1} s), scratch {d:.2} GiB (prompt chunks of {d} rows); {d} stream{s} at a {d}-token window, {d:.3} " ++
        "GiB each; MemAvailable {d:.2} GiB left (floor {d:.0} GiB); MTP up to {d} drafts a round for a drafting " ++
        "request (--no-drafts: serial decode), no prompt cache", .{
        hip.kernels.arch,
        dev.index,
        gib(weights),
        e.opened.w.stats.seconds,
        gib(scratch),
        chunk,
        streams,
        if (streams == 1) "" else "s",
        o.window,
        gib(per),
        gib(after),
        gib(floor),
        drafts,
    });
    errdefer gpa.free(startup);
    const own = try gpa.create(Owned);
    own.* = .{ .gpa = gpa, .e = e, .h = h, .startup = startup };
    return .{
        .backend = h.backend(),
        .facts = h.facts(),
        .rows = @intCast(1 + drafts),
        .ctx = own,
        .deinit = Owned.release,
        .streams = streams,
        .startup = startup,
    };
}

/// One device only: no rank follows.
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
    return refuse(a, problem, "GLM-5.3-Flash on HIP runs on one device: serve it with --tp 1", .{});
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

test "GLM on HIP serves numerics=exact only: an inexact policy is refused by name" {
    var p: hip.Policy = .{};
    try std.testing.expect(numericsRefused(p) == null);
    p.numerics = .inexact;
    try std.testing.expectEqualStrings(inexact_refusal, numericsRefused(p).?);
    try std.testing.expect(std.mem.indexOf(u8, numericsRefused(p).?, "numerics=exact only") != null);
}

test "GLM on HIP drafts a fixed chain: 3 drafts by default, --mtp-confidence 0 served, a floor or a gated chain refused" {
    var p: hip.Policy = .{};
    p.mtp = .{ .drafts = policy_mtp.drafts, .confidence = policy_mtp.confidence };
    try std.testing.expect(mtpRefused(p) == null);
    p.mtp = .{ .drafts = 2, .confidence = 0 }; // the sweep's --mtp-drafts 2 --mtp-confidence 0
    try std.testing.expect(mtpRefused(p) == null);
    p.mtp.confidence = 0.3; // --mtp-drafts alone: the HIP default's floor
    try std.testing.expect(std.mem.indexOf(u8, mtpRefused(p).?, "--mtp-confidence 0") != null);
    p.mtp.confidence = 0;
    p.mtp_depth = .gated;
    try std.testing.expect(std.mem.indexOf(u8, mtpRefused(p).?, "mtp_depth=gated") != null);
    p.mtp.drafts = 0;
    try std.testing.expect(mtpRefused(p) == null);
    try std.testing.expectEqual(@as(u32, 3), max_drafts);
}
