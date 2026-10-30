//! GLM-5.3-Flash's checkpoint as its headers say: every tensor's shard, byte range in the file, dtype and shape.
//! Read from model.safetensors.index.json and each shard's safetensors header (no tensor bytes), backend-neutral.

const std = @import("std");
const st = @import("core").safetensors;

pub const DType = st.DType;

/// One stored tensor: shard `shard` of `files`, bytes [off, off + len) of that file (8 + header + the entry's range).
pub const Src = struct {
    shard: u32,
    off: u64,
    len: u64,
    dtype: DType,
    rank: u8,
    shape: [st.max_rank]usize,

    pub fn dims(s: *const Src) []const usize {
        return s.shape[0..s.rank];
    }
};

/// The checkpoint's shard paths and every tensor by its stored name, in `arena`.
pub const Index = struct {
    files: []const [:0]const u8,
    /// Every shard's size in bytes.
    sizes: []const u64,
    tensors: std.StringArrayHashMapUnmanaged(Src),

    pub fn get(x: *const Index, name: []const u8) ?Src {
        return x.tensors.get(name);
    }
};

fn readAll(fd: std.c.fd_t, buf: []u8, at: u64) !void {
    var got: usize = 0;
    while (got < buf.len) {
        const n = std.c.pread(fd, buf.ptr + got, buf.len - got, @intCast(at + got));
        if (n < 0) {
            if (std.c.errno(n) == .INTR) continue;
            return error.ReadFailed;
        }
        if (n == 0) return error.EndOfFile;
        got += @intCast(n);
    }
}

/// The index of `dir` and each shard's header (tensor names, dtypes, shapes, byte ranges), in `arena`.
pub fn read(arena: std.mem.Allocator, io: std.Io, dir: []const u8) !Index {
    const index_path = try std.fs.path.join(arena, &.{ dir, "model.safetensors.index.json" });
    const text = try std.Io.Dir.cwd().readFileAlloc(io, index_path, arena, .limited(st.index.max_bytes));
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, arena, text, .{ .allocate = .alloc_always });
    var names: std.StringArrayHashMapUnmanaged(void) = .empty;
    var it = (try st.index.weightMap(doc)).iterator();
    while (it.next()) |kv| {
        if (kv.value_ptr.* != .string) return error.BadIndex;
        try names.put(arena, kv.value_ptr.string, {});
    }
    // shards in name order: model-00001-of-00083 first
    const order = try arena.dupe([]const u8, names.keys());
    std.mem.sort([]const u8, order, {}, struct {
        fn less(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.less);
    var files: std.ArrayList([:0]const u8) = .empty;
    var sizes: std.ArrayList(u64) = .empty;
    var tensors: std.StringArrayHashMapUnmanaged(Src) = .empty;
    for (order, 0..) |name, si| {
        if (!st.index.shardName(name)) return error.BadIndex;
        const full = try std.fmt.allocPrintSentinel(arena, "{s}/{s}", .{ dir, name }, 0);
        try files.append(arena, full);
        const fd = std.c.open(full, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
        if (fd < 0) return error.FileNotFound;
        defer _ = std.c.close(fd);
        const end = std.c.lseek(fd, 0, std.c.SEEK.END);
        if (end < 8) return error.BadSafetensors;
        const size: u64 = @intCast(end);
        try sizes.append(arena, size);
        var head: [8]u8 = undefined;
        try readAll(fd, &head, 0);
        const hlen = try st.headerSize(std.mem.readInt(u64, &head, .little), size);
        const header = try arena.alloc(u8, hlen);
        try readAll(fd, header, 8);
        const data = 8 + hlen;
        const entries = try st.parseHeader(arena, header, size - data);
        for (entries.keys(), entries.values()) |tname, e| {
            const got = try tensors.getOrPut(arena, tname);
            if (got.found_existing) return error.DuplicateTensor;
            got.value_ptr.* = .{ .shard = @intCast(si), .off = data + e.begin, .len = e.end - e.begin, .dtype = e.dtype, .rank = e.rank, .shape = e.shape };
        }
    }
    // every tensor the index names is in the shard it names
    var wm = (try st.index.weightMap(doc)).iterator();
    while (wm.next()) |kv| {
        const s = tensors.get(kv.key_ptr.*) orelse return error.IndexedTensorMissing;
        if (!std.mem.endsWith(u8, files.items[s.shard], kv.value_ptr.string)) return error.IndexedTensorElsewhere;
    }
    return .{ .files = files.items, .sizes = sizes.items, .tensors = tensors };
}
