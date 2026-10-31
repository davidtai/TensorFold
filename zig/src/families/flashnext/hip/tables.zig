//! The n-gram tables in host memory (the checkpoint's mapped shards), or on the SSD through a row cache
//! (rows=cached): a window's rows gathered for the device.
const std = @import("std");
const core = @import("core");
const config = @import("../config.zig");
const tensors = @import("tensors.zig");
const ngram = @import("ngram.zig");
const row_cache = @import("row_cache.zig");

/// One shard's packed rows and their group tables, as mapped.
const Shard = struct { words: []const u8, scales: []const u8, biases: []const u8 };

pub const Tables = struct {
    shards: []Shard,
    /// The first row of each shard, and one past the last.
    starts: []usize,
    dims: usize,
    bits: u8,
    group: usize,
    hasher: ngram.Hasher,
    /// rows=cached: the rows come from the SSD through this cache; the mapping's table pages are never read.
    cache: ?*row_cache.Cache = null,

    /// Bytes of a row's packed words, and of its scales (as of its biases).
    pub fn wordBytes(t: Tables) usize {
        return t.dims * t.bits / 8;
    }

    pub fn tableBytes(t: Tables) usize {
        return t.dims / t.group * 2;
    }

    /// The tables of the PLE layer `layer` from the mapped checkpoint; the shards' rows as the reference splits them.
    pub fn init(
        a: std.mem.Allocator,
        ck: *core.checkpoint.Checkpoint,
        c: *const config.Config,
        layer: usize,
        spelling: []const u8,
    ) !Tables {
        const heads = try c.ngramHeadCount();
        const ref = try config.pleRef(c);
        var total: usize = 0;
        for (ref.sizes) |s| total += @intCast(s);
        const rows = try std.math.divCeil(usize, total, c.ngram_divisor) * c.ngram_divisor;
        const base = rows / c.ngram_shards;
        const extra = rows % c.ngram_shards;
        const shards = try a.alloc(Shard, c.ngram_shards);
        const starts = try a.alloc(usize, c.ngram_shards + 1);
        starts[0] = 0;
        var spec: ?@import("../spec.zig").Spec = null;
        for (shards, 0..) |*s, i| {
            var lb: [128]u8 = undefined;
            var nb: [256]u8 = undefined;
            const stem = try a.dupe(u8, try tensors.ple.table(&nb, try tensors.layer(&lb, layer), spelling, i));
            const n = base + @intFromBool(i < extra);
            starts[i + 1] = starts[i] + n;
            const w = try ck.get(try std.fmt.allocPrint(a, "{s}.weight", .{stem}));
            const sc = try ck.get(try std.fmt.allocPrint(a, "{s}.scales", .{stem}));
            const bi = try ck.get(try std.fmt.allocPrint(a, "{s}.biases", .{stem}));
            if (spec == null)
                spec = (try c.declared(stem, core.quant.keys.canonical)) orelse return error.UnsupportedQuantization;
            if (w.shape[0] != n or sc.dtype != .bf16 or bi.dtype != .bf16) return error.UnexpectedNgramTable;
            s.* = .{ .words = w.bytes, .scales = sc.bytes, .biases = bi.bytes };
        }
        const dims = c.ple_dim / heads;
        if (shards[0].words.len != (starts[1] - starts[0]) * dims * spec.?.bits / 8) return error.UnexpectedNgramTable;
        return .{
            .shards = shards,
            .starts = starts,
            .dims = dims,
            .bits = spec.?.bits,
            .group = spec.?.group,
            .hasher = try ngram.Hasher.init(c),
        };
    }

    /// The rows copied once into anonymous memory from `a` (hip.Policy.rows): a round's gather then never faults, and
    /// reclaim cannot take them without swap. Each table is read sequentially, then its pages leave the page cache.
    pub fn keepResident(t: *Tables, a: std.mem.Allocator, ck: *const core.checkpoint.Checkpoint) !void {
        for (t.shards) |*s| {
            inline for (.{ "words", "scales", "biases" }) |f| {
                const from = @field(s, f);
                advise(from, std.os.linux.MADV.SEQUENTIAL);
                const copy = try a.alloc(u8, from.len);
                @memcpy(copy, from);
                advise(from, std.os.linux.MADV.DONTNEED);
                dropCached(ck, from);
                @field(s, f) = copy;
            }
        }
    }

    /// rows=cached: a cache of `mib` MiB that every gather reads through; the table pages leave the page cache. Refused
    /// below two prompt chunks of rows (`chunk` tokens, 16 rows each): a chunk's read-ahead must not evict the rows the
    /// chunk before it pinned.
    pub fn useCache(t: *Tables, gpa: std.mem.Allocator, io: std.Io, ck: *const core.checkpoint.Checkpoint, mib: usize, chunk: usize) !void {
        const sizes = [3]usize{ t.wordBytes(), t.tableBytes(), t.tableBytes() };
        const row = sizes[0] + sizes[1] + sizes[2];
        const slots = row_cache.Cache.slotsFor(mib << 20, row);
        const floor = 2 * chunk * ngram.heads;
        if (slots < floor) {
            const need = std.math.divCeil(usize, row_cache.Cache.bytesFor(floor, row), 1 << 20) catch unreachable;
            std.log.err("tensorfold: ngram_mib={d} holds {d} n-gram rows; a prompt chunk of {d} rows needs at least {d} (ngram_mib={d})", .{ mib, slots, chunk, floor, need });
            return error.NgramCacheTooSmall;
        }
        var paths: std.ArrayList([:0]const u8) = .empty;
        defer paths.deinit(gpa);
        var files: std.ArrayList(usize) = .empty;
        defer files.deinit(gpa);
        const parts = try gpa.alloc([3]row_cache.Part, t.shards.len);
        defer gpa.free(parts);
        for (t.shards, parts) |sh, *p| {
            inline for (.{ sh.words, sh.scales, sh.biases }, 0..) |b, k| {
                const at = fileOf(ck, b) orelse return error.UnexpectedNgramTable;
                const slot = std.mem.indexOfScalar(usize, files.items, at.file) orelse blk: {
                    try files.append(gpa, at.file);
                    try paths.append(gpa, ck.files.items[at.file].path);
                    break :blk files.items.len - 1;
                };
                p[k] = .{ .file = @intCast(slot), .at = at.off };
                advise(b, std.os.linux.MADV.DONTNEED);
                dropCached(ck, b);
            }
        }
        t.cache = try row_cache.Cache.create(gpa, io, paths.items, parts, t.starts, sizes, slots);
        std.log.info("n-gram rows: rows=cached, {d} MiB ({d} rows of {d} B), {d} read threads, O_DIRECT", .{ t.cache.?.bytes() >> 20, slots, row, row_cache.workers });
    }

    pub fn deinit(t: *Tables) void {
        if (t.cache) |c| c.destroy();
        t.cache = null;
    }

    /// rows=cached: the rows of `tokens` after `history` asked for ahead of the gather that needs them; nothing to do on
    /// the other paths. A wrong guess costs reads, never bytes: the gather looks up the ids it is given.
    pub fn ahead(t: *const Tables, gpa: std.mem.Allocator, history: [ngram.context]i64, tokens: []const u32) void {
        const c = t.cache orelse return;
        if (tokens.len == 0) return;
        const rows = gpa.alloc([ngram.heads]u32, tokens.len) catch return;
        defer gpa.free(rows);
        t.hasher.ids(history, tokens, rows);
        c.ahead(@as([*]const u32, @ptrCast(rows.ptr))[0 .. rows.len * ngram.heads]);
    }

    /// rows=cached: one log line of the cache's counters since the last line, for `what` (a prompt chunk, a decode).
    pub fn report(t: *const Tables, comptime what: []const u8, args: anytype) void {
        const c = t.cache orelse return;
        const n = c.takeCounts();
        if (n.gathers == 0 and n.ahead == 0 and n.reads == 0) return;
        std.log.info("ngram cache: " ++ what ++ ": {d} gathers ({d} missed, {d} waited), {d} lookups, {d} hits, {d} pending, {d} misses; ahead {d} ({d} present, {d} queued, {d} dropped); {d} evictions; {d} rows read, {d} B; wait {d:.3} ms; {d} of {d} slots", args ++ .{
            n.gathers,       n.gathers_missed, n.gathers_waited, n.lookups,      n.hits,      n.pending, n.misses,
            n.ahead,         n.ahead_present,  n.ahead_queued,   n.ahead_dropped, n.evictions, n.reads,   n.read_bytes,
            @as(f64, @floatFromInt(n.wait_ns)) / 1e6, c.used(), c.slots,
        });
    }

    /// The checkpoint file holding `b`, and `b`'s offset in it.
    fn fileOf(ck: *const core.checkpoint.Checkpoint, b: []const u8) ?struct { file: usize, off: u64 } {
        for (ck.files.items, 0..) |*f, i| {
            const m = f.map.memory;
            if (@intFromPtr(b.ptr) < @intFromPtr(m.ptr) or @intFromPtr(b.ptr) + b.len > @intFromPtr(m.ptr) + m.len) continue;
            return .{ .file = i, .off = @intFromPtr(b.ptr) - @intFromPtr(m.ptr) };
        }
        return null;
    }

    fn advise(b: []const u8, advice: u32) void {
        if (@import("builtin").os.tag != .linux) return;
        const page = std.heap.pageSize();
        const from = std.mem.alignBackward(usize, @intFromPtr(b.ptr), page);
        const to = std.mem.alignForward(usize, @intFromPtr(b.ptr) + b.len, page);
        _ = std.os.linux.madvise(@ptrFromInt(from), to - from, advice);
    }

    /// The page cache's copy of `b`'s file range dropped (safetensors.File.release does it for a whole file).
    fn dropCached(ck: *const core.checkpoint.Checkpoint, b: []const u8) void {
        if (@import("builtin").os.tag != .linux) return;
        for (ck.files.items) |*f| {
            const m = f.map.memory;
            if (@intFromPtr(b.ptr) < @intFromPtr(m.ptr) or @intFromPtr(b.ptr) + b.len > @intFromPtr(m.ptr) + m.len) continue;
            const off = @intFromPtr(b.ptr) - @intFromPtr(m.ptr);
            _ = std.os.linux.fadvise(f.file.handle, @intCast(off), @intCast(b.len), std.os.linux.POSIX_FADV.DONTNEED);
            return;
        }
    }

    /// Each id's packed words, scales and biases copied into the three staging slices, in id order.
    pub fn gather(t: *const Tables, ids: []const u32, words: []u8, scales: []u8, biases: []u8) !void {
        const wb = t.wordBytes();
        const tb = t.tableBytes();
        if (words.len < ids.len * wb or scales.len < ids.len * tb or biases.len < ids.len * tb)
            return error.StagingTooSmall;
        if (t.cache) |c| return c.gather(ids, .{ words[0 .. ids.len * wb], scales[0 .. ids.len * tb], biases[0 .. ids.len * tb] });
        for (ids, 0..) |id, i| {
            const s = std.sort.upperBound(usize, t.starts, @as(usize, id), struct {
                fn order(x: usize, y: usize) std.math.Order {
                    return std.math.order(x, y);
                }
            }.order) - 1;
            if (s >= t.shards.len) return error.NgramRowOutOfRange;
            const row = id - t.starts[s];
            const sh = t.shards[s];
            @memcpy(words[i * wb ..][0..wb], sh.words[row * wb ..][0..wb]);
            @memcpy(scales[i * tb ..][0..tb], sh.scales[row * tb ..][0..tb]);
            @memcpy(biases[i * tb ..][0..tb], sh.biases[row * tb ..][0..tb]);
        }
    }
};
