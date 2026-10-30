//! GLM-5.3-Flash's checkpoint onto one HIP device, shard by shard, as layout.zig places it. Each shard is read in
//! chunks with O_DIRECT into one of two pinned staging buffers and copied from there to the slabs, so no tensor byte
//! stays in the page cache (where a file system refuses O_DIRECT, each chunk is dropped from the cache after its
//! read). The bytes are never changed. MemAvailable is read before the slabs are made and after each chunk: under
//! `floor` the load stops. The mcg markers are checked, the expert tables written, and each layer's gate and up suh
//! compared (one input rotation a row when they are equal).

const std = @import("std");
const builtin = @import("builtin");
const hip = @import("hip");
const core = @import("core");
const config = @import("../config.zig");
const shards = @import("shards.zig");
const layout = @import("layout.zig");
const weights = @import("weights.zig");

const dio = core.direct_io;

pub const Options = struct {
    /// Bytes a read takes at most (a staging buffer's size, two of them, pinned).
    chunk: usize = 64 << 20,
    /// The load stops when MemAvailable falls under this.
    floor: u64 = 6 << 30,
    /// Placed ranges closer than this in a file are read in one pass (the bytes between are read and not used).
    gap: u64 = 1 << 20,
    /// A line a shard on stderr.
    log: bool = true,
};

pub const Stats = struct {
    seconds: f64 = 0,
    /// File bytes read (O_DIRECT spans included).
    read_bytes: u64 = 0,
    /// Tensor bytes copied to the slabs: every placed tensor's, once.
    placed_bytes: u64 = 0,
    slab_bytes: u64 = 0,
    start_available: u64 = 0,
    min_available: u64 = std.math.maxInt(u64),
    gtt_before: ?u64 = null,
    gtt_peak: ?u64 = null,
    rss_peak: ?u64 = null,
    device_peak: u64 = 0,
    /// Shards read through the page cache (their file system refused O_DIRECT); each chunk dropped after its read.
    buffered: u32 = 0,
};

pub const Loaded = struct {
    gpa: std.mem.Allocator,
    slabs: []hip.DeviceBuffer,
    base: []u64,
    shared_input: []bool,
    model: weights.Model,
    stats: Stats,

    pub fn deinit(l: *Loaded) void {
        for (l.slabs) |*s| s.free();
        l.gpa.free(l.slabs);
        l.gpa.free(l.base);
        l.gpa.free(l.shared_input);
        l.* = undefined;
    }
};

// ---- the box's memory as the kernel counts it ----

fn readSmall(path: [*:0]const u8, buf: []u8) ?[]const u8 {
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
    if (fd < 0) return null;
    defer _ = std.c.close(fd);
    var got: usize = 0;
    while (got < buf.len) {
        const n = std.c.read(fd, buf.ptr + got, buf.len - got);
        if (n <= 0) break;
        got += @intCast(n);
    }
    return buf[0..got];
}

/// `key`'s value in kB lines ("MemAvailable:  123 kB"), in bytes.
fn kbField(text: []const u8, key: []const u8) ?u64 {
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, key)) continue;
        var it = std.mem.tokenizeAny(u8, line[key.len..], " \t");
        const v = it.next() orelse return null;
        return (std.fmt.parseInt(u64, v, 10) catch return null) * 1024;
    }
    return null;
}

/// MemAvailable in bytes.
pub fn memAvailable() ?u64 {
    var buf: [8192]u8 = undefined;
    return kbField(readSmall("/proc/meminfo", &buf) orelse return null, "MemAvailable:");
}

/// This process's peak resident set (VmHWM) in bytes.
pub fn rssPeak() ?u64 {
    var buf: [8192]u8 = undefined;
    return kbField(readSmall("/proc/self/status", &buf) orelse return null, "VmHWM:");
}

/// The GPU's GTT bytes in use (amdgpu's mem_info_gtt_used of the first card that has it).
pub fn gttUsed() ?u64 {
    var buf: [64]u8 = undefined;
    var path: [96]u8 = undefined;
    for (0..8) |i| {
        const p = std.fmt.bufPrintSentinel(&path, "/sys/class/drm/card{d}/device/mem_info_gtt_used", .{i}, 0) catch return null;
        const text = readSmall(p, &buf) orelse continue;
        return std.fmt.parseInt(u64, std.mem.trim(u8, text, " \n"), 10) catch null;
    }
    return null;
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

pub fn gib(n: u64) f64 {
    return @as(f64, @floatFromInt(n)) / (1 << 30);
}

// ---- the copy plan of each shard ----

/// One file range and where it goes: a slab offset, or the host bytes of marker `marker`.
const Op = struct {
    off: u64,
    len: u64,
    slab: u16,
    dst: u64,
    marker: ?u32 = null,

    fn end(o: Op) u64 {
        return o.off + o.len;
    }
};

fn byOffset(_: void, a: Op, b: Op) bool {
    return a.off < b.off;
}

fn shardOps(a: std.mem.Allocator, l: *const layout.Layout, n_shards: usize) ![][]Op {
    const lists = try a.alloc(std.ArrayList(Op), n_shards);
    for (lists) |*x| x.* = .empty;
    for (l.placed) |p| try lists[p.src.shard].append(a, .{ .off = p.src.off, .len = p.src.len, .slab = p.slab, .dst = p.off });
    for (l.markers, 0..) |m, i| try lists[m.src.shard].append(a, .{ .off = m.src.off, .len = m.src.len, .slab = 0, .dst = 0, .marker = @intCast(i) });
    const out = try a.alloc([]Op, n_shards);
    for (lists, out) |*x, *o| {
        std.mem.sort(Op, x.items, {}, byOffset);
        if (x.items.len > 1) for (x.items[1..], 0..) |op, i| if (op.off < x.items[i].end()) return error.OverlappingTensors;
        o.* = x.items;
    }
    return out;
}

/// The two staging buffers and the streams their copies run on: a buffer is reused once its stream is done.
const Stage = struct {
    host: [2]hip.HostBuffer,
    streams: [2]hip.Stream,
    busy: [2]bool = .{ false, false },
    next: u1 = 0,
};

/// Reads `index`'s shards into fresh slabs as `l` places them: the device must be current.
pub fn load(gpa: std.mem.Allocator, r: *const hip.Runtime, index: *const shards.Index, l: *const layout.Layout, o: Options) !Loaded {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var stats: Stats = .{ .slab_bytes = l.slabBytes(), .gtt_before = gttUsed() };
    const t0 = nowNs();
    _ = hip.usage(true);
    const staging = o.chunk + 2 * dio.alignment;
    stats.start_available = memAvailable() orelse return error.NoMeminfo;
    stats.min_available = stats.start_available;
    if (stats.start_available < stats.slab_bytes + 2 * staging + o.floor) {
        std.log.err("GLM-5.3-Flash on HIP takes {d:.2} GiB of GTT; MemAvailable is {d:.2} GiB and the load keeps {d:.2} GiB free", .{ gib(stats.slab_bytes), gib(stats.start_available), gib(o.floor) });
        return error.NoRoom;
    }
    const ops = try shardOps(a, l, index.files.len);

    const slabs = try gpa.alloc(hip.DeviceBuffer, l.slabs.len);
    var made: usize = 0;
    errdefer {
        for (slabs[0..made]) |*s| s.free();
        gpa.free(slabs);
    }
    for (l.slabs, slabs) |len, *s| {
        s.* = try hip.DeviceBuffer.alloc(r, len);
        made += 1;
    }
    const base = try gpa.alloc(u64, slabs.len);
    errdefer gpa.free(base);
    for (slabs, base) |s, *b| b.* = s.base();

    var st: Stage = undefined;
    st.busy = .{ false, false };
    st.next = 0;
    st.host[0] = try hip.HostBuffer.alloc(r, staging);
    defer st.host[0].free();
    st.host[1] = try hip.HostBuffer.alloc(r, staging);
    defer st.host[1].free();
    st.streams[0] = try hip.Stream.init(r);
    defer st.streams[0].deinit();
    st.streams[1] = try hip.Stream.init(r);
    defer st.streams[1].deinit();
    // a failed load still waits for the copies in flight before their staging is freed
    defer for (st.streams) |s| s.synchronize() catch {};

    const marks = try a.alloc(u8, 4 * l.markers.len);
    @memset(marks, 0);
    for (index.files, ops, 0..) |path, list, si| {
        if (list.len == 0) continue;
        const ts = nowNs();
        var f = try dio.File.open(path);
        defer f.close();
        if (!f.direct) stats.buffered += 1;
        var first: usize = 0;
        var placed_here: u64 = 0;
        while (first < list.len) {
            // a run: ops whose gaps are under o.gap
            var last = first;
            var hi = list[first].end();
            while (last + 1 < list.len and list[last + 1].off <= hi + o.gap) {
                last += 1;
                hi = @max(hi, list[last].end());
            }
            var pos = list[first].off;
            var j = first;
            while (pos < hi) {
                const len: usize = @intCast(@min(o.chunk, hi - pos));
                const k = st.next;
                st.next ^= 1;
                if (st.busy[k]) try st.streams[k].synchronize();
                const buf: dio.Buffer = @alignCast(st.host[k].bytes);
                const got = try f.read(buf, pos, len);
                stats.read_bytes += dio.span(pos, len);
                if (builtin.os.tag == .linux) _ = std.os.linux.fadvise(f.fd, @intCast(pos), @intCast(len), std.os.linux.POSIX_FADV.DONTNEED);
                const start = @intFromPtr(got.ptr) - @intFromPtr(buf.ptr);
                const end = pos + len;
                while (j <= last and list[j].end() <= pos) j += 1;
                var q = j;
                while (q <= last and list[q].off < end) : (q += 1) {
                    const op = list[q];
                    const lo = @max(op.off, pos);
                    const top = @min(op.end(), end);
                    if (lo >= top) continue;
                    const from = start + @as(usize, @intCast(lo - pos));
                    const n: usize = @intCast(top - lo);
                    if (op.marker) |m| {
                        @memcpy(marks[4 * m + @as(usize, @intCast(lo - op.off)) ..][0..n], st.host[k].bytes[from..][0..n]);
                    } else {
                        try slabs[op.slab].uploadAsync(@intCast(op.dst + (lo - op.off)), st.host[k].view(from, from + n), st.streams[k]);
                        placed_here += n;
                    }
                }
                st.busy[k] = true;
                pos = end;
                const avail = memAvailable() orelse 0;
                stats.min_available = @min(stats.min_available, avail);
                if (gttUsed()) |g| stats.gtt_peak = @max(stats.gtt_peak orelse 0, g);
                if (avail < o.floor) {
                    std.log.err("MemAvailable {d:.2} GiB is under the load's floor of {d:.2} GiB at {s}: the load stops", .{ gib(avail), gib(o.floor), path });
                    return error.LowMemory;
                }
            }
            first = last + 1;
        }
        stats.placed_bytes += placed_here;
        if (o.log) std.debug.print("[glm load] shard {d}/{d} {s}: {d:.2} GiB placed in {d:.1} s{s}; MemAvailable {d:.2} GiB\n", .{ si + 1, index.files.len, std.fs.path.basename(path), gib(placed_here), @as(f64, @floatFromInt(nowNs() - ts)) / 1e9, if (f.direct) "" else " (buffered)", gib(stats.min_available) });
    }
    for (st.streams) |s| try s.synchronize();

    var want: u64 = 0;
    for (l.placed) |p| want += p.src.len;
    if (stats.placed_bytes != want) return error.ShortLoad;
    for (l.markers, 0..) |m, i| {
        const v = std.mem.readInt(u32, marks[4 * i ..][0..4], .little);
        if (v != layout.mcg_magic) {
            std.log.err("{s} is 0x{x:0>8}, not the mcg codebook's 0x{x:0>8}: the HIP kernels decode mcg only", .{ m.name, v, layout.mcg_magic });
            return error.NotMcg;
        }
    }
    // the expert tables: expert e's trellis at the stack's base + e x stride
    const c = l.config;
    const tab = try a.alloc(u64, c.experts);
    for (l.tables) |t| {
        for (tab, 0..) |*p, e| p.* = base[t.slab] + t.trellis_off + e * t.stride;
        try slabs[t.slab].upload(@intCast(t.off), std.mem.sliceAsBytes(tab));
    }
    // one input rotation a row when every expert's gate and up suh are the same bytes
    const shared = try gpa.alloc(bool, c.layers + c.mtp);
    errdefer gpa.free(shared);
    const g = try a.alloc(u8, @as(usize, c.experts) * c.hidden * 2);
    const u = try a.alloc(u8, g.len);
    for (shared, 0..) |*sh, i| {
        sh.* = false;
        if (!c.isMoe(@intCast(i))) continue;
        var nb: [128]u8 = undefined;
        const pg = l.find(try std.fmt.bufPrint(&nb, layout.lm ++ "layers.{d}.mlp.experts.0.gate_proj.suh", .{i})).?;
        const pu = l.find(try std.fmt.bufPrint(&nb, layout.lm ++ "layers.{d}.mlp.experts.0.up_proj.suh", .{i})).?;
        try slabs[pg.slab].download(@intCast(pg.off), g);
        try slabs[pu.slab].download(@intCast(pu.off), u);
        sh.* = std.mem.eql(u8, g, u);
    }
    const m = try weights.model(l, base, shared);
    stats.seconds = @as(f64, @floatFromInt(nowNs() - t0)) / 1e9;
    stats.rss_peak = rssPeak();
    stats.device_peak = hip.usage(false).peak;
    if (gttUsed()) |gt| stats.gtt_peak = @max(stats.gtt_peak orelse 0, gt);
    return .{ .gpa = gpa, .slabs = slabs, .base = base, .shared_input = shared, .model = m, .stats = stats };
}

test "glm hip load: MemAvailable and VmHWM lines read in bytes" {
    const text = "MemTotal:       193548104 kB\nMemFree:          1000 kB\nMemAvailable:   188000000 kB\n";
    try std.testing.expectEqual(@as(?u64, 188000000 * 1024), kbField(text, "MemAvailable:"));
    try std.testing.expectEqual(@as(?u64, null), kbField(text, "VmHWM:"));
}

test "glm hip load: a shard's ops in file order; overlapping ranges are refused" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const s: shards.Src = .{ .shard = 0, .off = 0, .len = 0, .dtype = .bf16, .rank = 1, .shape = @splat(1) };
    var placed = [_]layout.Placed{
        .{ .name = "b", .src = blk: {
            var x = s;
            x.off = 100;
            x.len = 50;
            break :blk x;
        }, .slab = 0, .off = 0, .class = .norms },
        .{ .name = "a", .src = blk: {
            var x = s;
            x.off = 10;
            x.len = 90;
            break :blk x;
        }, .slab = 1, .off = 256, .class = .norms },
    };
    var l: layout.Layout = .{ .config = .{}, .slabs = &.{}, .placed = &placed, .by_name = .empty, .tables = &.{}, .markers = &.{} };
    const ops = try shardOps(a, &l, 1);
    try std.testing.expectEqual(@as(u64, 10), ops[0][0].off);
    try std.testing.expectEqual(@as(u64, 100), ops[0][1].off);
    placed[0].src.off = 99;
    l.placed = &placed;
    try std.testing.expectError(error.OverlappingTensors, shardOps(a, &l, 1));
}
