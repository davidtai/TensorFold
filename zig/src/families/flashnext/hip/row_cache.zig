//! The n-gram rows through a fixed RAM cache of rows (hip.Policy.rows = cached): one slot a row, the least recently used
//! row evicted when a new one comes in. A missing row is read from the checkpoint file with O_DIRECT positional reads
//! (core.direct_io), so the page cache holds no copy and the cache is the only RAM the tables take. Parked workers read
//! the rows: the ones a gather misses, and the ones asked for ahead of use (the next prompt chunk, a drafted token).
//! The bytes are the file's, so a gather copies the bytes the resident and mapped paths copy.
//!
//! One thread (the engine's) owns the index, the LRU list and the reservations. A worker only claims a queued slot
//! (queued -> reading), writes its row and publishes it (ready). A slot that is queued or being read is never evicted,
//! so its row id does not change under a worker. MiaAI-Lab's TensorFold patch 0013 (cuda_ple_reader.zig) is the source
//! of the worker queue (a ring of batches, newest first) and of the row read.
const std = @import("std");
const core = @import("core");
const dio = core.direct_io;

/// Read threads: 4 KiB O_DIRECT random reads on the NVMe drive measured peak at 32 (449 K reads/s, p50 70 us; 64 and 128
/// threads read no faster and wait longer).
pub const workers = 32;
/// Bytes a slot costs beyond its row: id, LRU links, gather pin, state.
const slot_extra = 4 + 4 + 4 + 4 + 1;
const none = std.math.maxInt(u32);
const batch_ring = 16;
/// Slots a worker claims at once from a batch.
const claim_grain = 16;

const State = enum(u8) { free, queued, reading, ready, failed };

/// Where one tensor of the rows sits: the reader's file and the byte offset of row 0.
pub const Part = struct { file: u16, at: u64 };

/// The counters since the last `take`.
pub const Counts = struct {
    /// Gathers (a prompt chunk or a round) and the row ids they looked up.
    gathers: u64 = 0,
    lookups: u64 = 0,
    /// Lookups whose row was ready; whose row was still being read ahead; that missed and were read on demand.
    hits: u64 = 0,
    pending: u64 = 0,
    misses: u64 = 0,
    /// Gathers that read a row on demand; that waited for any row.
    gathers_missed: u64 = 0,
    gathers_waited: u64 = 0,
    /// Rows asked for ahead; of them already in the cache; queued for the workers; not queued (the ring was full).
    ahead: u64 = 0,
    ahead_present: u64 = 0,
    ahead_queued: u64 = 0,
    ahead_dropped: u64 = 0,
    evictions: u64 = 0,
    /// Rows read from the SSD (by the workers and by the gather's thread) and the bytes of their direct reads.
    reads: u64 = 0,
    read_bytes: u64 = 0,
    /// Time the gathers spent on rows not ready at lookup.
    wait_ns: u64 = 0,
};

const Batch = struct {
    slots: []u32 = &.{},
    claimed: std.atomic.Value(usize) = .init(0),
    finished: std.atomic.Value(usize) = .init(0),
    done: bool = true,
};

pub const Cache = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    files: []dio.File,
    /// Per shard: words, scales, biases.
    parts: [][3]Part,
    /// First row of each shard, and one past the last (the tables' own).
    starts: []const usize,
    sizes: [3]usize,
    row: usize,
    slots: u32,
    /// Slots handed out before the first eviction.
    fresh: u32 = 0,
    mem: []align(std.heap.page_size_min) u8,
    data: []u8,
    ids: []u32,
    prev: []u32,
    next: []u32,
    pin: []u32,
    state: []std.atomic.Value(State),
    index: []u32,
    shift: u5,
    head: u32 = none,
    tail: u32 = none,
    /// The current gather's mark: a slot pinned with it is not evicted (0: no gather runs).
    epoch: u32 = 0,
    last_epoch: u32 = 0,
    /// The gather thread's scratch: slots of the ids, slots not ready at lookup, a direct read's buffer.
    of: std.ArrayList(u32) = .empty,
    waiting: std.ArrayList(u32) = .empty,
    buf: []align(dio.alignment) u8,
    counts: Counts = .{},
    read_rows: std.atomic.Value(u64) = .init(0),
    read_span: std.atomic.Value(u64) = .init(0),
    taken_reads: u64 = 0,
    taken_bytes: u64 = 0,
    // the workers' queue
    mutex: std.Io.Mutex = .init,
    batches: [batch_ring]Batch = @splat(.{}),
    first: usize = 0,
    end: usize = 0,
    go: std.atomic.Value(u32) = .init(0),
    stopping: std.atomic.Value(bool) = .init(false),
    threads: [workers]std.Thread = undefined,
    spawned: usize = 0,

    /// Bytes `slots` slots take: rows, per-slot fields and the index (at most two thirds full).
    pub fn bytesFor(slots: usize, row: usize) usize {
        return slots * (row + slot_extra) + indexLen(slots) * 4;
    }

    fn indexLen(slots: usize) usize {
        return std.math.ceilPowerOfTwoAssert(usize, @max(16, slots + slots / 2 + 1));
    }

    /// The most slots of `row` bytes that fit `budget` bytes.
    pub fn slotsFor(budget: usize, row: usize) usize {
        var lo: usize = 0;
        var hi: usize = budget / row + 1;
        while (lo + 1 < hi) {
            const mid = lo + (hi - lo) / 2;
            if (bytesFor(mid, row) <= budget) lo = mid else hi = mid;
        }
        return lo;
    }

    /// A cache of `slots` rows of the tensors `parts` (in `paths`, opened here), rows split as `starts`.
    pub fn create(
        gpa: std.mem.Allocator,
        io: std.Io,
        paths: []const [:0]const u8,
        parts: []const [3]Part,
        starts: []const usize,
        sizes: [3]usize,
        slots: usize,
    ) !*Cache {
        if (slots < 2 or slots >= none or parts.len + 1 != starts.len) return error.BadNgramCache;
        const c = try gpa.create(Cache);
        errdefer gpa.destroy(c);
        const files = try gpa.alloc(dio.File, paths.len);
        errdefer gpa.free(files);
        var opened: usize = 0;
        errdefer for (files[0..opened]) |*f| f.close();
        for (paths, files) |p, *f| {
            f.* = try dio.File.open(p);
            opened += 1;
        }
        const own_parts = try gpa.dupe([3]Part, parts);
        errdefer gpa.free(own_parts);
        const row = sizes[0] + sizes[1] + sizes[2];
        const n = slots;
        const len = indexLen(n);
        // one anonymous mapping: a page is resident once a slot on it is written
        const total = bytesFor(n, row);
        const mem = try std.posix.mmap(null, total, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
        errdefer std.posix.munmap(mem);
        var at: usize = 0;
        const carve = struct {
            fn of(comptime T: type, m: []u8, off: *usize, count: usize) []T {
                off.* = std.mem.alignForward(usize, off.*, @alignOf(T));
                const s = @as([*]T, @ptrCast(@alignCast(m[off.*..].ptr)))[0..count];
                off.* += count * @sizeOf(T);
                return s;
            }
        }.of;
        // index and the 4-byte fields first: no padding before the rows and states
        const index = carve(u32, mem, &at, len);
        const ids = carve(u32, mem, &at, n);
        const prev = carve(u32, mem, &at, n);
        const next = carve(u32, mem, &at, n);
        const pin = carve(u32, mem, &at, n);
        const state = carve(std.atomic.Value(State), mem, &at, n);
        const data = carve(u8, mem, &at, n * row);
        std.debug.assert(at <= total);
        const buf = try gpa.alignedAlloc(u8, .fromByteUnits(dio.alignment), 2 * dio.alignment);
        errdefer gpa.free(buf);
        c.* = .{
            .gpa = gpa,
            .io = io,
            .files = files,
            .parts = own_parts,
            .starts = starts,
            .sizes = sizes,
            .row = row,
            .slots = @intCast(n),
            .mem = mem,
            .data = data,
            .ids = ids,
            .prev = prev,
            .next = next,
            .pin = pin,
            .state = state,
            .index = index,
            .shift = @intCast(32 - std.math.log2_int(usize, len)),
            .buf = buf,
        };
        return c;
    }

    pub fn destroy(c: *Cache) void {
        c.stopping.store(true, .release);
        _ = c.go.fetchAdd(1, .release);
        c.io.futexWake(u32, &c.go.raw, std.math.maxInt(u32));
        for (c.threads[0..c.spawned]) |t| t.join();
        for (&c.batches) |*b| if (!b.done) c.gpa.free(b.slots);
        for (c.files) |*f| f.close();
        c.of.deinit(c.gpa);
        c.waiting.deinit(c.gpa);
        c.gpa.free(c.buf);
        std.posix.munmap(c.mem);
        c.gpa.free(c.files);
        c.gpa.free(c.parts);
        c.gpa.destroy(c);
    }

    /// Slots holding a row.
    pub fn used(c: *const Cache) u32 {
        return c.fresh;
    }

    /// Bytes the cache took at create (the bound): rows, per-slot fields and index.
    pub fn bytes(c: *const Cache) usize {
        return c.mem.len;
    }

    // ---- the index: open addressing, linear probes, backward-shift delete (engine thread only)

    fn home(c: *const Cache, id: u32) usize {
        return @intCast((id *% 0x9E3779B1) >> c.shift);
    }

    fn find(c: *const Cache, id: u32) ?u32 {
        const mask = c.index.len - 1;
        var i = c.home(id);
        while (true) : (i = (i + 1) & mask) {
            const e = c.index[i];
            if (e == 0) return null;
            if (c.ids[e - 1] == id) return e - 1;
        }
    }

    fn insert(c: *Cache, slot: u32) void {
        const mask = c.index.len - 1;
        var i = c.home(c.ids[slot]);
        while (c.index[i] != 0) i = (i + 1) & mask;
        c.index[i] = slot + 1;
    }

    fn remove(c: *Cache, slot: u32) void {
        const mask = c.index.len - 1;
        var i = c.home(c.ids[slot]);
        while (c.index[i] != slot + 1) i = (i + 1) & mask;
        var j = i;
        while (true) {
            j = (j + 1) & mask;
            const e = c.index[j];
            if (e == 0) break;
            const k = c.home(c.ids[e - 1]);
            // the entry at j stays when its home is cyclically in (i, j]
            const stays = if (i <= j) (i < k and k <= j) else (i < k or k <= j);
            if (!stays) {
                c.index[i] = e;
                i = j;
            }
        }
        c.index[i] = 0;
    }

    // ---- the LRU list: head is the most recently used (engine thread only)

    fn unlink(c: *Cache, s: u32) void {
        const p = c.prev[s];
        const n = c.next[s];
        if (p != none) c.next[p] = n else c.head = n;
        if (n != none) c.prev[n] = p else c.tail = p;
    }

    fn pushHead(c: *Cache, s: u32) void {
        c.prev[s] = none;
        c.next[s] = c.head;
        if (c.head != none) c.prev[c.head] = s;
        c.head = s;
        if (c.tail == none) c.tail = s;
    }

    fn touch(c: *Cache, s: u32) void {
        if (c.head == s) return;
        c.unlink(s);
        c.pushHead(s);
    }

    /// A slot for a new row: a fresh one, else the least recently used one that no read holds and the gather has not
    /// pinned (its row swapped out). A queued row nobody has claimed yet (a read-ahead that was not needed) is dropped
    /// too: a worker reads the id a slot holds when it claims it, so a stale batch entry reads the slot's new row. Slots
    /// passed over move to the head.
    fn take(c: *Cache) !u32 {
        if (c.fresh < c.slots) {
            const s = c.fresh;
            c.fresh += 1;
            c.pushHead(s);
            return s;
        }
        var n: u32 = 0;
        while (n < c.slots) : (n += 1) {
            const s = c.tail;
            const st = c.state[s].load(.acquire);
            const free = c.epoch == 0 or c.pin[s] != c.epoch;
            if (free and (st == .ready or st == .failed or (st == .queued and c.state[s].cmpxchgStrong(.queued, .free, .acq_rel, .monotonic) == null))) {
                c.remove(s);
                c.touch(s);
                c.counts.evictions += 1;
                return s;
            }
            c.touch(s);
        }
        return error.NgramCacheTooSmall;
    }

    /// A slot for row `id`, queued for a read: in the index from now on.
    fn reserve(c: *Cache, id: u32) !u32 {
        const s = try c.take();
        c.ids[s] = id;
        c.insert(s);
        c.state[s].store(.queued, .release);
        return s;
    }

    // ---- reads

    /// Row `ids[s]` from the files into slot `s`, with the direct reads' buffer `buf`; the bytes its reads took.
    fn readRow(c: *const Cache, s: u32, buf: []align(dio.alignment) u8) !u64 {
        const id: usize = c.ids[s];
        const sh = std.sort.upperBound(usize, c.starts, id, struct {
            fn order(x: usize, y: usize) std.math.Order {
                return std.math.order(x, y);
            }
        }.order) - 1;
        if (sh >= c.parts.len) return error.NgramRowOutOfRange;
        const local = id - c.starts[sh];
        const out = c.data[@as(usize, s) * c.row ..][0..c.row];
        var off: usize = 0;
        var took: u64 = 0;
        for (c.parts[sh], c.sizes) |p, n| {
            const at = p.at + local * n;
            @memcpy(out[off..][0..n], try c.files[p.file].read(buf, at, n));
            took += dio.span(at, n);
            off += n;
        }
        return took;
    }

    /// Claims a queued (or failed) slot and reads its row; false when another thread has it.
    fn claimRead(c: *Cache, s: u32, buf: []align(dio.alignment) u8, from: State) !bool {
        if (c.state[s].cmpxchgStrong(from, .reading, .acq_rel, .monotonic) != null) return false;
        const took = c.readRow(s, buf) catch |err| {
            c.state[s].store(.failed, .release);
            return err;
        };
        c.state[s].store(.ready, .release);
        _ = c.read_rows.fetchAdd(1, .monotonic);
        _ = c.read_span.fetchAdd(took, .monotonic);
        return true;
    }

    /// Slot `s` ready: read here if it is still queued (or a worker's read failed), else wait for the worker's read.
    fn ready(c: *Cache, s: u32) !void {
        var spins: u32 = 0;
        while (true) {
            switch (c.state[s].load(.acquire)) {
                .ready => return,
                .queued => _ = try c.claimRead(s, c.buf, .queued),
                .failed => _ = try c.claimRead(s, c.buf, .failed),
                .reading => {
                    spins += 1;
                    if (spins % 64 == 0) std.Thread.yield() catch {} else std.atomic.spinLoopHint();
                },
                .free => return error.BadNgramCache,
            }
        }
    }

    // ---- the gather and the read-ahead (engine thread)

    /// Each id's row bytes (`sizes` apart) copied into `outs`, in id order: hits from the cache, misses read now on the
    /// workers and this thread.
    pub fn gather(c: *Cache, ids: []const u32, outs: [3][]u8) !void {
        c.last_epoch +%= 1;
        if (c.last_epoch == 0) c.last_epoch = 1;
        c.epoch = c.last_epoch;
        defer c.epoch = 0;
        c.of.clearRetainingCapacity();
        c.waiting.clearRetainingCapacity();
        try c.of.ensureTotalCapacity(c.gpa, ids.len);
        try c.waiting.ensureTotalCapacity(c.gpa, ids.len);
        var misses: u64 = 0;
        var pending: u64 = 0;
        for (ids) |id| {
            if (c.find(id)) |s| {
                c.touch(s);
                const st = c.state[s].load(.acquire);
                if (st == .ready) {
                    c.counts.hits += 1;
                } else {
                    // a read ahead not done yet, or a row this gather already asked for
                    if (c.pin[s] != c.epoch) c.waiting.appendAssumeCapacity(s);
                    pending += 1;
                }
                c.pin[s] = c.epoch;
                c.of.appendAssumeCapacity(s);
            } else {
                const s = try c.reserve(id);
                c.pin[s] = c.epoch;
                c.waiting.appendAssumeCapacity(s);
                c.of.appendAssumeCapacity(s);
                misses += 1;
            }
        }
        c.counts.gathers += 1;
        c.counts.lookups += ids.len;
        c.counts.misses += misses;
        c.counts.pending += pending;
        if (misses > 0) c.counts.gathers_missed += 1;
        if (c.waiting.items.len > 0) {
            c.counts.gathers_waited += 1;
            const t0 = std.Io.Clock.awake.now(c.io).toNanoseconds();
            if (misses > 0) _ = c.submit(c.waiting.items);
            // this thread reads from the back while the workers claim from the front
            var k = c.waiting.items.len;
            while (k > 0) {
                k -= 1;
                const s = c.waiting.items[k];
                const st = c.state[s].load(.acquire);
                if (st == .queued) _ = try c.claimRead(s, c.buf, .queued);
            }
            for (c.waiting.items) |s| try c.ready(s);
            c.counts.wait_ns += @intCast(std.Io.Clock.awake.now(c.io).toNanoseconds() - t0);
        }
        for (c.of.items, 0..) |s, i| {
            const src = c.data[@as(usize, s) * c.row ..][0..c.row];
            var off: usize = 0;
            inline for (0..3) |k| {
                const n = c.sizes[k];
                @memcpy(outs[k][i * n ..][0..n], src[off..][0..n]);
                off += n;
            }
        }
    }

    /// The rows of `ids` asked for ahead of their gather: queued for the workers, and this returns at once.
    pub fn ahead(c: *Cache, ids: []const u32) void {
        if (ids.len == 0) return;
        c.waiting.clearRetainingCapacity();
        c.waiting.ensureTotalCapacity(c.gpa, ids.len) catch return;
        c.counts.ahead += ids.len;
        for (ids) |id| {
            if (c.find(id)) |s| {
                c.touch(s);
                c.counts.ahead_present += 1;
                continue;
            }
            const s = c.reserve(id) catch break;
            c.waiting.appendAssumeCapacity(s);
        }
        if (!c.submit(c.waiting.items)) {
            // nothing reads them now: the gather that needs them reads them
            c.counts.ahead_dropped += c.waiting.items.len;
            return;
        }
        c.counts.ahead_queued += c.waiting.items.len;
    }

    /// The counters since the last call.
    pub fn takeCounts(c: *Cache) Counts {
        var out = c.counts;
        c.counts = .{};
        const r = c.read_rows.load(.monotonic);
        const b = c.read_span.load(.monotonic);
        out.reads = r - c.taken_reads;
        out.read_bytes = b - c.taken_bytes;
        c.taken_reads = r;
        c.taken_bytes = b;
        return out;
    }

    // ---- the workers (0013's queue: a ring of batches, the newest first)

    /// Queues `slots` for the workers; false when the ring is full (the slots stay queued: a gather reads them).
    fn submit(c: *Cache, slots: []const u32) bool {
        if (slots.len == 0) return true;
        const own = c.gpa.dupe(u32, slots) catch return false;
        c.mutex.lockUncancelable(c.io);
        if (c.end - c.first == batch_ring or c.startWorkers() == 0) {
            c.mutex.unlock(c.io);
            c.gpa.free(own);
            return false;
        }
        const b = &c.batches[c.end % batch_ring];
        b.slots = own;
        b.claimed.store(0, .monotonic);
        b.finished.store(0, .monotonic);
        b.done = false;
        c.end += 1;
        c.mutex.unlock(c.io);
        _ = c.go.fetchAdd(1, .release);
        c.io.futexWake(u32, &c.go.raw, @intCast(@min(c.spawned, (slots.len + claim_grain - 1) / claim_grain)));
        return true;
    }

    fn startWorkers(c: *Cache) usize {
        while (c.spawned < workers) {
            c.threads[c.spawned] = std.Thread.spawn(.{}, work, .{c}) catch break;
            c.spawned += 1;
        }
        return c.spawned;
    }

    fn claim(c: *Cache) ?struct { slots: []const u32, batch: *Batch } {
        c.mutex.lockUncancelable(c.io);
        defer c.mutex.unlock(c.io);
        var i = c.end;
        while (i > c.first) {
            i -= 1;
            const b = &c.batches[i % batch_ring];
            const at = b.claimed.load(.monotonic);
            if (at >= b.slots.len) continue;
            const n = @min(claim_grain, b.slots.len - at);
            b.claimed.store(at + n, .monotonic);
            return .{ .slots = b.slots[at..][0..n], .batch = b };
        }
        return null;
    }

    fn finish(c: *Cache, b: *Batch, n: usize) void {
        const len = b.slots.len;
        if (b.finished.fetchAdd(n, .acq_rel) + n != len) return;
        c.mutex.lockUncancelable(c.io);
        defer c.mutex.unlock(c.io);
        c.gpa.free(b.slots);
        b.slots = &.{};
        b.done = true;
        while (c.first < c.end and c.batches[c.first % batch_ring].done) c.first += 1;
    }

    fn work(c: *Cache) void {
        var buf: [2 * dio.alignment]u8 align(dio.alignment) = undefined;
        while (true) {
            const seen = c.go.load(.acquire);
            if (c.stopping.load(.acquire)) return;
            if (c.claim()) |got| {
                // a failed read leaves the slot failed: the gather that needs it reads it again and reports the error
                for (got.slots) |s| _ = c.claimRead(s, &buf, .queued) catch false;
                c.finish(got.batch, got.slots.len);
                continue;
            }
            c.io.futexWaitUncancelable(u32, &c.go.raw, seen);
        }
    }

    /// Returns once the workers have taken every queued batch (tests).
    pub fn drain(c: *Cache) void {
        while (true) {
            c.mutex.lockUncancelable(c.io);
            const idle = c.first == c.end;
            c.mutex.unlock(c.io);
            if (idle) return;
            std.Thread.yield() catch {};
        }
    }
};

// ---- tests: a table file with known rows, gathered through small caches against a plain copy

const Fixture = struct {
    const shards = [_]usize{ 0, 300, 700, 1000 };
    const sizes = [3]usize{ 80, 10, 10 };

    fn byte(id: usize, part: usize, k: usize) u8 {
        return @truncate(id *% 31 +% part *% 101 +% k *% 7 +% (id >> 8));
    }

    /// The file: some header bytes, then for each shard its words, scales and biases tensors.
    fn file(a: std.mem.Allocator, parts: *[3][3]Part) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        try out.appendNTimes(a, 0x5a, 4321);
        for (0..3) |sh| for (0..3) |part| {
            parts[sh][part] = .{ .file = 0, .at = out.items.len };
            for (shards[sh]..shards[sh + 1]) |id| for (0..sizes[part]) |k| try out.append(a, byte(id, part, k));
            try out.appendNTimes(a, 0xa5, 37);
        };
        return out.toOwnedSlice(a);
    }

    fn want(ids: []const u32, outs: [3][]u8) void {
        for (ids, 0..) |id, i| for (0..3) |part| for (0..sizes[part]) |k| {
            outs[part][i * sizes[part] + k] = byte(id, part, k);
        };
    }
};

fn testCache(a: std.mem.Allocator, io: std.Io, dir: std.testing.TmpDir, slots: usize) !*Cache {
    var parts: [3][3]Part = undefined;
    const data = try Fixture.file(a, &parts);
    defer a.free(data);
    try dir.dir.writeFile(io, .{ .sub_path = "rows", .data = data });
    var pbuf: [512]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&pbuf, ".zig-cache/tmp/{s}/rows", .{dir.sub_path}, 0);
    return Cache.create(a, io, &.{path}, &parts, &Fixture.shards, Fixture.sizes, slots);
}

fn expectGather(c: *Cache, ids: []const u32) !void {
    const a = std.testing.allocator;
    var got: [3][]u8 = undefined;
    var exp: [3][]u8 = undefined;
    for (0..3) |p| {
        got[p] = try a.alloc(u8, ids.len * Fixture.sizes[p]);
        exp[p] = try a.alloc(u8, ids.len * Fixture.sizes[p]);
    }
    defer for (0..3) |p| {
        a.free(got[p]);
        a.free(exp[p]);
    };
    try c.gather(ids, got);
    Fixture.want(ids, exp);
    for (0..3) |p| try std.testing.expectEqualSlices(u8, exp[p], got[p]);
}

test "row cache: gathers equal the file's rows through hits, misses, evictions and read-ahead" {
    if (@import("builtin").os.tag != .linux) return error.SkipZigTest;
    const a = std.testing.allocator;
    const io = std.testing.io;
    var dir = std.testing.tmpDir(.{});
    defer dir.cleanup();
    const c = try testCache(a, io, dir, 64);
    defer c.destroy();
    // shard edges, repeats inside one gather, the last row
    const edges = [_]u32{ 0, 299, 300, 301, 699, 700, 999, 0, 299, 999, 5, 5, 5 };
    try expectGather(c, &edges);
    var n = c.takeCounts();
    try std.testing.expectEqual(@as(u64, 8), n.misses);
    try std.testing.expectEqual(@as(u64, 8), n.reads);
    try expectGather(c, &edges);
    n = c.takeCounts();
    try std.testing.expectEqual(@as(u64, edges.len), n.hits);
    try std.testing.expectEqual(@as(u64, 0), n.reads);
    // more distinct rows than slots over several gathers: rows swap out and back in
    var prng = std.Random.DefaultPrng.init(7);
    var ids: [48]u32 = undefined;
    for (0..40) |_| {
        for (&ids) |*x| x.* = prng.random().uintLessThan(u32, 1000);
        // read ahead of the next gather, sometimes for the same rows, sometimes not
        if (prng.random().boolean()) c.ahead(&ids) else c.ahead(ids[0..8]);
        try expectGather(c, &ids);
    }
    n = c.takeCounts();
    try std.testing.expect(n.evictions > 0);
    try std.testing.expect(c.used() == 64);
    c.drain();
}

test "row cache: a gather with more distinct rows than slots is refused, not wrong" {
    if (@import("builtin").os.tag != .linux) return error.SkipZigTest;
    const a = std.testing.allocator;
    const io = std.testing.io;
    var dir = std.testing.tmpDir(.{});
    defer dir.cleanup();
    const c = try testCache(a, io, dir, 8);
    defer c.destroy();
    var ids: [9]u32 = undefined;
    for (&ids, 0..) |*x, i| x.* = @intCast(i * 100);
    var outs: [3][]u8 = undefined;
    for (0..3) |p| outs[p] = try a.alloc(u8, ids.len * Fixture.sizes[p]);
    defer for (0..3) |p| a.free(outs[p]);
    try std.testing.expectError(error.NgramCacheTooSmall, c.gather(&ids, outs));
    c.drain();
    // the refused gather's slots stay queued: a smaller gather after it reads them and is right
    try expectGather(c, ids[0..8]);
}

test "row cache: the size bound" {
    try std.testing.expect(Cache.bytesFor(Cache.slotsFor(256 << 20, 100), 100) <= 256 << 20);
    try std.testing.expect(Cache.bytesFor(Cache.slotsFor(256 << 20, 100) + 1, 100) > 256 << 20);
    // 256 MiB of 100-byte rows: about 2.15 M slots, more than the 1.66 M rows of a 256K prompt and its warm-up
    try std.testing.expect(Cache.slotsFor(256 << 20, 100) > 2_100_000);
}
