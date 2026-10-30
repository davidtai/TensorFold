//! The loaded bytes against the files: sha256 of a placed tensor's slab region (read back from the device) against
//! sha256 of its file range (read with O_DIRECT). A sample: each class's first, middle and last tensor, then every
//! `every`-th placed tensor. The expert tables are checked whole against the addresses they must hold.

const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const shards = @import("shards.zig");
const layout = @import("layout.zig");

const dio = core.direct_io;
const Sha = std.crypto.hash.sha2.Sha256;

pub const Report = struct {
    tensors: usize = 0,
    bytes: u64 = 0,
    mismatched: usize = 0,
    tables: usize = 0,
    bad_tables: usize = 0,
    /// The first mismatch's name.
    first_bad: ?[]const u8 = null,
};

/// Indices into `l.placed` of the sample, ascending, no repeats.
pub fn pick(a: std.mem.Allocator, l: *const layout.Layout, every: usize) ![]u32 {
    var want: std.AutoArrayHashMapUnmanaged(u32, void) = .empty;
    var by_class: [layout.class_count]std.ArrayList(u32) = @splat(.empty);
    for (l.placed, 0..) |p, i| try by_class[@backingInt(p.class)].append(a, @intCast(i));
    for (by_class) |list| if (list.items.len > 0) {
        for ([_]usize{ 0, list.items.len / 2, list.items.len - 1 }) |k| try want.put(a, list.items[k], {});
    };
    if (every > 0) {
        var i: usize = 0;
        while (i < l.placed.len) : (i += every) try want.put(a, @intCast(i), {});
    }
    const out = try a.dupe(u32, want.keys());
    std.mem.sort(u32, out, {}, std.sort.asc(u32));
    return out;
}

const piece = 32 << 20;

fn fileHash(f: dio.File, buf: dio.Buffer, off: u64, len: u64) ![32]u8 {
    var h = Sha.init(.{});
    var at: u64 = 0;
    while (at < len) {
        const n: usize = @intCast(@min(dio.fits(buf.len), len - at));
        h.update(try f.read(buf, off + at, n));
        at += n;
    }
    if (@import("builtin").os.tag == .linux) _ = std.os.linux.fadvise(f.fd, @intCast(off), @intCast(len), std.os.linux.POSIX_FADV.DONTNEED);
    return h.finalResult();
}

fn deviceHash(b: hip.DeviceBuffer, host: []u8, off: u64, len: u64) ![32]u8 {
    var h = Sha.init(.{});
    var at: u64 = 0;
    while (at < len) {
        const n: usize = @intCast(@min(host.len, len - at));
        try b.download(@intCast(off + at), host[0..n]);
        h.update(host[0..n]);
        at += n;
    }
    return h.finalResult();
}

/// The sample's hashes on the device and in the files, and every expert table.
pub fn check(gpa: std.mem.Allocator, index: *const shards.Index, l: *const layout.Layout, slabs: []const hip.DeviceBuffer, every: usize) !Report {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var rep: Report = .{};
    const sample = try pick(a, l, every);
    const buf = try a.alignedAlloc(u8, .fromByteUnits(dio.alignment), piece + 2 * dio.alignment);
    const host = try a.alloc(u8, piece);
    var open_shard: ?u32 = null;
    var f: dio.File = undefined;
    defer if (open_shard != null) f.close();
    for (sample) |i| {
        const p = l.placed[i];
        if (open_shard == null or open_shard.? != p.src.shard) {
            if (open_shard != null) f.close();
            f = try dio.File.open(index.files[p.src.shard]);
            open_shard = p.src.shard;
        }
        const want = try fileHash(f, buf, p.src.off, p.src.len);
        const got = try deviceHash(slabs[p.slab], host, p.off, p.src.len);
        rep.tensors += 1;
        rep.bytes += p.src.len;
        if (!std.mem.eql(u8, &want, &got)) {
            rep.mismatched += 1;
            if (rep.first_bad == null) rep.first_bad = p.name;
            std.debug.print("[glm verify] MISMATCH {s}: file {s} device {s}\n", .{ p.name, &std.fmt.bytesToHex(want, .lower), &std.fmt.bytesToHex(got, .lower) });
        }
    }
    const c = l.config;
    const tab = try a.alloc(u64, c.experts);
    for (l.tables) |t| {
        try slabs[t.slab].download(@intCast(t.off), std.mem.sliceAsBytes(tab));
        rep.tables += 1;
        for (tab, 0..) |v, e| if (v != slabs[t.slab].base() + t.trellis_off + e * t.stride) {
            rep.bad_tables += 1;
            break;
        };
    }
    if (rep.first_bad) |n| rep.first_bad = try gpa.dupe(u8, n);
    return rep;
}
