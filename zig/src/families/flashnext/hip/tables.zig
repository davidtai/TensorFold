//! The n-gram tables in host memory (the checkpoint's mapped shards): a window's rows gathered for the device.
const std = @import("std");
const core = @import("core");
const config = @import("../config.zig");
const tensors = @import("tensors.zig");
const ngram = @import("ngram.zig");

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
    /// The rows are read through the checkpoint's mapping (rows=mapped, or a rank of a group), not an anonymous copy:
    /// a gather then asks the kernel for each row's pages before it copies them.
    mapped: bool = true,
    /// Mapped rows: one bit a page of each shard's words, scales and biases, set once a gather asked for the page.
    asked: [][3][]u64 = &.{},

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
        t.mapped = false;
    }

    /// Mapped rows: no readahead on a fault (a row is 100 bytes in three tensors; a fault would read its neighbours'
    /// pages too), and a page map for `prefetch`.
    pub fn randomAccess(t: *Tables, a: std.mem.Allocator) !void {
        const page = std.heap.pageSize();
        t.asked = try a.alloc([3][]u64, t.shards.len);
        for (t.shards, t.asked) |s, *bits| {
            inline for (.{ "words", "scales", "biases" }, 0..) |f, k| {
                const b = @field(s, f);
                advise(b, std.os.linux.MADV.RANDOM);
                const from = std.mem.alignBackward(usize, @intFromPtr(b.ptr), page);
                const pages = (std.mem.alignForward(usize, @intFromPtr(b.ptr) + b.len, page) - from) / page;
                bits[k] = try a.alloc(u64, std.math.divCeil(usize, pages, 64) catch unreachable);
                @memset(bits[k], 0);
            }
        }
    }

    /// Mapped rows: the pages of each id not asked for before, asked for at once (MADV_WILLNEED starts their reads and
    /// does not wait), so the cold pages of a window load in parallel and the copy waits only for the slowest. A page
    /// is asked for once: a warm gather makes no call, and a page the kernel drops later costs one fault of its own.
    fn prefetch(t: *const Tables, ids: []const u32) void {
        if (@import("builtin").os.tag != .linux or t.asked.len == 0) return;
        const page = std.heap.pageSize();
        const wb = t.wordBytes();
        const tb = t.tableBytes();
        for (ids) |id| {
            const s = t.shardOf(id) orelse continue;
            const row = id - t.starts[s];
            const sh = t.shards[s];
            inline for (.{ sh.words, sh.scales, sh.biases }, .{ wb, tb, tb }, 0..) |part, n, k| {
                const base = std.mem.alignBackward(usize, @intFromPtr(part.ptr), page);
                const from = std.mem.alignBackward(usize, @intFromPtr(part.ptr) + row * n, page);
                const to = std.mem.alignForward(usize, @intFromPtr(part.ptr) + row * n + n, page);
                const bits = t.asked[s][k];
                var new = false;
                var p = (from - base) / page;
                while (p < (to - base) / page) : (p += 1) {
                    const bit = @as(u64, 1) << @intCast(p % 64);
                    if (bits[p / 64] & bit == 0) {
                        bits[p / 64] |= bit;
                        new = true;
                    }
                }
                if (new) _ = std.os.linux.madvise(@ptrFromInt(from), to - from, std.os.linux.MADV.WILLNEED);
            }
        }
    }

    /// The shard holding row `id`, or null past the last.
    fn shardOf(t: *const Tables, id: u32) ?usize {
        const s = std.sort.upperBound(usize, t.starts, @as(usize, id), struct {
            fn order(x: usize, y: usize) std.math.Order {
                return std.math.order(x, y);
            }
        }.order) - 1;
        return if (s < t.shards.len) s else null;
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
        if (t.mapped) t.prefetch(ids);
        for (ids, 0..) |id, i| {
            const s = t.shardOf(id) orelse return error.NgramRowOutOfRange;
            const row = id - t.starts[s];
            const sh = t.shards[s];
            @memcpy(words[i * wb ..][0..wb], sh.words[row * wb ..][0..wb]);
            @memcpy(scales[i * tb ..][0..tb], sh.scales[row * tb ..][0..tb]);
            @memcpy(biases[i * tb ..][0..tb], sh.biases[row * tb ..][0..tb]);
        }
    }
};
