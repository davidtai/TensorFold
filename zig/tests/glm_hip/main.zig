//! tf-glm-hip-test: GLM-5.3-Flash's checkpoint on one HIP device, and the TF_GLM_DUMP format.
//!   plan <checkpoint>                 the placement from the headers alone (no GPU): bytes by class, slabs
//!   load <checkpoint> [--every N] [--chunk-mib N] [--floor-gib N]
//!                                     everything onto the GPU, bytes by class, a sha256 sample against the files,
//!                                     load time, MemAvailable's minimum, peak GTT and RSS
//!   dump-echo <reference run> <out>   a reference run's own arrays through the dump writer: the .npy files must be
//!                                     the reference's bytes, and ref/compare.py must pass
const std = @import("std");
const hip = @import("hip");
const glm = @import("glm");

const layout = glm.hip_load.layout;
const load = glm.hip_load.load;
const dump = glm.hip_load.dump;
const npy = @import("core").npy;

const usage =
    \\usage: tf-glm-hip-test plan <checkpoint>
    \\       tf-glm-hip-test load <checkpoint> [--every N] [--chunk-mib N] [--floor-gib N]
    \\       tf-glm-hip-test dump-echo <reference run dir> <out dir>
    \\
;

const gib = load.gib;

fn printPlan(l: *const layout.Layout) void {
    std.debug.print("{s:<26} {s:>12} {s:>12} {s:>8}\n", .{ "class", "stored GiB", "on GPU GiB", "tensors" });
    for (0..layout.class_count) |i| {
        const c: layout.Class = @enumFromInt(i);
        std.debug.print("{s:<26} {d:>12.3} {d:>12.3} {d:>8}\n", .{ c.label(), gib(l.stored[i]), gib(l.on_gpu[i]), l.stored_count[i] });
    }
    var on: u64 = 0;
    var n: u64 = 0;
    for (l.on_gpu, l.stored_count) |x, k| {
        on += x;
        n += k;
    }
    std.debug.print("{s:<26} {d:>12.3} {d:>12.3} {d:>8}  ({d} B stored)\n", .{ "TOTAL", gib(l.total_stored), gib(on), n, l.total_stored });
    std.debug.print("\nPLAN.md section 1 rows (stored GiB; the mcg markers in suh + svh as the plan counts them):\n", .{});
    for (layout.plan_rows) |row| {
        var s: u64 = 0;
        for (row.classes) |c| s += l.stored[@backingInt(c)];
        std.debug.print("  {s:<40} {d:>9.3}\n", .{ row.label, gib(s) });
    }
    std.debug.print("\nslabs: {d}, {d:.3} GiB on the GPU (tensors {d:.3} GiB, expert tables and 256-byte alignment the rest)\n", .{ l.slabs.len, gib(l.slabBytes()), gib(on) });
}

fn planOf(gpa: std.mem.Allocator, a: std.mem.Allocator, io: std.Io, dir: []const u8) !*glm.hip_load.open.Checkpoint {
    var problem: []const u8 = "";
    return glm.hip_load.open.plan(gpa, a, io, dir, &problem) catch |err| {
        std.debug.print("{s}\n", .{problem});
        return err;
    };
}

fn plan(init: std.process.Init, dir: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    const ck = try planOf(init.gpa, arena.allocator(), init.io, dir);
    defer ck.deinit();
    std.debug.print("{s}: {d} shards, {d} tensors\n", .{ dir, ck.index.files.len, ck.index.tensors.count() });
    printPlan(&ck.layout);
    std.debug.print("PLAN OK\n", .{});
}

const LoadArgs = struct { every: usize = 512, chunk: usize = 64 << 20, floor: u64 = 6 << 30 };

fn loadArgs(args: []const [:0]const u8) !LoadArgs {
    var o: LoadArgs = .{};
    var i: usize = 0;
    while (i < args.len) : (i += 2) {
        if (i + 1 >= args.len) return error.MissingArgument;
        const v = try std.fmt.parseInt(usize, args[i + 1], 10);
        if (std.mem.eql(u8, args[i], "--every")) o.every = v else if (std.mem.eql(u8, args[i], "--chunk-mib")) o.chunk = v << 20 else if (std.mem.eql(u8, args[i], "--floor-gib")) o.floor = @as(u64, v) << 30 else return error.UnknownOption;
    }
    return o;
}

fn loadAll(init: std.process.Init, dir: []const u8, o: LoadArgs) !void {
    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const ck = try planOf(init.gpa, a, init.io, dir);
    defer ck.deinit();
    const index = &ck.index;
    const l = &ck.layout;
    printPlan(l);
    var r = try hip.Runtime.open();
    defer r.close();
    var ctx = try hip.Context.init(&r, 0);
    defer ctx.deinit();
    var w = try load.load(init.gpa, &r, index, l, .{ .chunk = o.chunk, .floor = o.floor });
    defer w.deinit();
    const s = w.stats;
    std.debug.print("\nLOAD {d:.1} s: {d:.3} GiB placed ({d:.3} GiB read, {d:.2} GB/s), {d:.3} GiB of slabs\n", .{ s.seconds, gib(s.placed_bytes), gib(s.read_bytes), @as(f64, @floatFromInt(s.read_bytes)) / s.seconds / 1e9, gib(s.slab_bytes) });
    std.debug.print("MemAvailable: {d:.2} GiB at the start, {d:.2} GiB at the least (floor {d:.2} GiB)\n", .{ gib(s.start_available), gib(s.min_available), gib(o.floor) });
    std.debug.print("GTT used: {d:.2} GiB before, {d:.2} GiB at the peak; device bytes peak {d:.2} GiB; RSS peak {d:.2} GiB; buffered shards {d}\n", .{ gib(s.gtt_before orelse 0), gib(s.gtt_peak orelse 0), gib(s.device_peak), gib(s.rss_peak orelse 0), s.buffered });
    var shared: usize = 0;
    var moe: usize = 0;
    for (w.shared_input, 0..) |x, i| if (l.config.isMoe(@intCast(i))) {
        moe += 1;
        shared += @intFromBool(x);
    };
    std.debug.print("experts: {d} MoE layers, mcg markers checked ({d}); gate and up share suh (one rotation a row) in {d} of {d}\n", .{ moe, l.markers.len, shared, moe });
    const m = w.model;
    std.debug.print("model: embed 0x{x}, head [{d}, {d}] at 0x{x}, layer 3 experts gate table 0x{x}, MTP eh [{d}, {d}]\n", .{ m.embed, m.head.n, m.head.k, m.head.w, m.layers[3].ffn.moe.experts.gate, m.mtp.eh.n, m.mtp.eh.k });
    const t0 = std.Io.Timestamp.now(init.io, .awake).nanoseconds;
    const rep = try glm.hip_load.verify.check(init.gpa, index, l, w.slabs, o.every);
    defer if (rep.first_bad) |n| init.gpa.free(n);
    const secs = @as(f64, @floatFromInt(std.Io.Timestamp.now(init.io, .awake).nanoseconds - t0)) / 1e9;
    std.debug.print("VERIFY sha256: {d} tensors ({d:.3} GiB) against the files, {d} differ; {d} expert tables, {d} wrong ({d:.1} s)\n", .{ rep.tensors, gib(rep.bytes), rep.mismatched, rep.tables, rep.bad_tables, secs });
    if (rep.mismatched > 0 or rep.bad_tables > 0) return error.BytesDiffer;
    if (s.min_available < o.floor) return error.LowMemory;
    std.debug.print("LOAD OK\n", .{});
}

/// A reference .npy whole, or null when the run has none.
fn readNpy(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) !?npy.Array {
    const path = try std.fs.path.join(a, &.{ dir, name });
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 32)) catch |err| switch (err) {
        error.FileNotFound => return null,
        else => return err,
    };
    return try npy.parse(bytes);
}

fn as(comptime T: type, arr: npy.Array) []align(1) const T {
    return std.mem.bytesAsSlice(T, arr.data);
}

fn copyF32(a: std.mem.Allocator, src: []align(1) const f32) ![]f32 {
    const out = try a.alloc(f32, src.len);
    for (out, src) |*o, v| o.* = v;
    return out;
}

fn echo(init: std.process.Init, ref: []const u8, out: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const eos = [_]u32{ 154820, 154827, 154829 };
    var d = try dump.Dump.open(a, out, ref, &eos);
    for (d.prompts) |p0| {
        var p = p0;
        const rdir = try std.fs.path.join(a, &.{ ref, p.name });
        // the generated count as the reference ran it (older runs' prompts.json has no gen)
        const g = try std.Io.Dir.cwd().readFileAlloc(init.io, try std.fs.path.join(a, &.{ rdir, "greedy.json" }), a, .limited(1 << 24));
        const gj = try std.json.parseFromSliceLeaky(std.json.Value, a, g, .{});
        p.gen = @intCast(gj.object.get("ids").?.array.items.len);
        const s = try d.begin(p);
        const t = p.ids.len;
        const n = t + p.gen - 1;
        if (try readNpy(a, init.io, rdir, "dump_rows.npy")) |rows| {
            const rr = as(i32, rows);
            if (rr.len != s.rows.len) return error.DumpRowsDiffer;
            for (rr, s.rows) |x, y| if (x != y) return error.DumpRowsDiffer;
        }
        if (try readNpy(a, init.io, rdir, "streams.npy")) |x| {
            const v = try copyF32(a, as(f32, x));
            const rd = s.rows.len;
            for (0..dump.L) |l| for (0..rd) |j| for (0..2) |site| {
                try s.stream(l, @intCast(s.rows[j]), @intCast(site), v[((l * rd + j) * 2 + site) * dump.S * dump.D ..][0 .. dump.S * dump.D]);
            };
        }
        if (try readNpy(a, init.io, rdir, "sublayers.npy")) |x| {
            const v = try copyF32(a, as(f32, x));
            const rd = s.rows.len;
            for (0..dump.L) |l| for (0..rd) |j| for (0..4) |k| {
                try s.sublayer(l, @intCast(s.rows[j]), @intCast(k), v[((l * rd + j) * 4 + k) * dump.D ..][0..dump.D]);
            };
        }
        if (try readNpy(a, init.io, rdir, "mtp_streams.npy")) |x| {
            const v = try copyF32(a, as(f32, x));
            for (0..s.rows.len) |j| for (0..2) |site| try s.mtpStream(@intCast(s.rows[j]), @intCast(site), v[(j * 2 + site) * dump.D ..][0..dump.D]);
        }
        if (try readNpy(a, init.io, rdir, "mtp_sublayers.npy")) |x| {
            const v = try copyF32(a, as(f32, x));
            for (0..s.rows.len) |j| for (0..4) |k| try s.mtpSublayer(@intCast(s.rows[j]), @intCast(k), v[(j * 4 + k) * dump.D ..][0..dump.D]);
        }
        if (try readNpy(a, init.io, rdir, "router_picks.npy")) |x| {
            const pk = as(i16, x);
            const wt = as(f32, (try readNpy(a, init.io, rdir, "router_weights.npy")).?);
            var ids: [8]i32 = undefined;
            var ws: [8]f32 = undefined;
            for (0..dump.router_layers) |l| for (0..n) |row| {
                for (0..8) |k| {
                    ids[k] = pk[(l * n + row) * 8 + k];
                    ws[k] = wt[(l * n + row) * 8 + k];
                }
                try s.router(l, row, &ids, &ws);
            };
        }
        if (try readNpy(a, init.io, rdir, "logits.npy")) |x| {
            const v = try copyF32(a, as(f32, x));
            for (0..t) |row| try s.promptLogits(row, v[row * dump.V ..][0..dump.V]);
        }
        if (try readNpy(a, init.io, rdir, "gen_logits.npy")) |x| {
            const v = try copyF32(a, as(f32, x));
            for (0..x.shape[0]) |k| _ = try s.step(v[k * dump.V ..][0..dump.V]);
        }
        if (try readNpy(a, init.io, rdir, "hidden.npy")) |x| {
            const v = try copyF32(a, as(f32, x));
            for (0..x.shape[0]) |row| try s.hiddenRow(row, v[row * dump.D ..][0..dump.D]);
        }
        try s.close();
        std.debug.print("{s}: {d} rows, {d} generated (reference {d})\n", .{ p.name, t, s.generated().len, p.gen });
    }
    try d.finish("dump-echo: the reference's own arrays through the writer");
    // every .npy written from the reference's arrays must be the reference's bytes
    const mine = try std.Io.Dir.cwd().readFileAlloc(init.io, try std.fs.path.join(a, &.{ out, "hashes.tsv" }), a, .limited(1 << 24));
    const theirs = try std.Io.Dir.cwd().readFileAlloc(init.io, try std.fs.path.join(a, &.{ ref, "hashes.tsv" }), a, .limited(1 << 24));
    var same: usize = 0;
    var differ: usize = 0;
    var lines = std.mem.splitScalar(u8, mine, '\n');
    _ = lines.next();
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        var f = std.mem.splitScalar(u8, line, '\t');
        _ = f.next();
        _ = f.next();
        const file = f.next() orelse continue;
        if (!std.mem.endsWith(u8, file, ".npy") and !std.mem.endsWith(u8, file, ")")) continue;
        if (std.mem.indexOf(u8, theirs, line) != null) {
            same += 1;
            std.debug.print("  SAME {s}\n", .{file});
        } else if (std.mem.indexOf(u8, theirs, file) != null) {
            differ += 1;
            std.debug.print("  DIFF {s}\n", .{file});
        }
    }
    std.debug.print("ECHO {d} files byte-identical to the reference, {d} differ\n", .{ same, differ });
    if (differ > 0) return error.EchoDiffers;
}

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const result = if (args.len == 3 and std.mem.eql(u8, args[1], "plan"))
        plan(init, args[2])
    else if (args.len >= 3 and std.mem.eql(u8, args[1], "load"))
        loadAll(init, args[2], loadArgs(args[3..]) catch {
            std.debug.print(usage, .{});
            return 2;
        })
    else if (args.len == 4 and std.mem.eql(u8, args[1], "dump-echo"))
        echo(init, args[2], args[3])
    else {
        std.debug.print(usage, .{});
        return 2;
    };
    result catch |err| {
        std.debug.print("FAIL {t}\n", .{err});
        return 1;
    };
    return 0;
}
