//! TF_GLM_DUMP: the HIP forward's per-layer dumps in exactly the float64 reference's run format
//! (work/glm53-halo/lanes/G5-ref.md section 4, written by ref/run_ref.py), so ref/compare.py reads a HIP run as a
//! candidate. Inert when TF_GLM_DUMP is unset.
//!
//! TF_GLM_DUMP=<out dir> and TF_GLM_DUMP_REF=<a reference run dir>: the prompts are the reference's, their ids read
//! from <ref>/<name>/ids.u32 (checked against prompts.json and <ref>/ids.sha256); nothing is tokenized here.
//! The .npy files are byte for byte what numpy.save writes for the same arrays (numpy 2.2's header, 64-byte aligned).
//!
//! The forward calls a `Seq` (one prompt) at each site: `stream`, `sublayer`, `router`, `promptLogits`, `hidden`,
//! `step`, `mtpStream`, `mtpSublayer`, `mtpLogits`; `Seq.close` writes the files; `Dump.finish` writes prompts.json,
//! hashes.tsv and the manifest.

const std = @import("std");

pub const D = 4096;
pub const V = 154880;
pub const S = 4;
pub const L = 45;
/// Router rows: the 45 layers and the MTP layer.
pub const router_layers = 46;
pub const top_k = 8;
/// run_ref.py's FULL_LOGITS_ROWS, DUMP_HEAD, DUMP_TAIL.
pub const full_logits_rows = 1100;
pub const dump_head = 64;
pub const dump_tail = 8;

// ---- files ----

fn cpath(a: std.mem.Allocator, parts: []const []const u8) ![:0]u8 {
    const p = try std.fs.path.join(a, parts);
    defer a.free(p);
    return a.dupeSentinel(u8, p, 0);
}

fn mkdir(path: [:0]const u8) !void {
    const rc = std.c.mkdir(path, 0o755);
    if (rc == 0 or std.c.errno(rc) == .EXIST) return;
    return error.MkdirFailed;
}

fn createFile(path: [:0]const u8) !std.c.fd_t {
    const fd = std.c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o644));
    if (fd < 0) return error.CreateFailed;
    return fd;
}

fn pwriteAll(fd: std.c.fd_t, bytes: []const u8, at: u64) !void {
    var done: usize = 0;
    while (done < bytes.len) {
        const n = std.c.pwrite(fd, bytes.ptr + done, bytes.len - done, @intCast(at + done));
        if (n < 0) {
            if (std.c.errno(n) == .INTR) continue;
            return error.WriteFailed;
        }
        done += @intCast(n);
    }
}

fn writeFile(path: [:0]const u8, bytes: []const u8) !void {
    const fd = try createFile(path);
    defer _ = std.c.close(fd);
    try pwriteAll(fd, bytes, 0);
}

fn readFile(a: std.mem.Allocator, path: [:0]const u8, limit: usize) ![]u8 {
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
    if (fd < 0) return error.FileNotFound;
    defer _ = std.c.close(fd);
    const end = std.c.lseek(fd, 0, std.c.SEEK.END);
    if (end < 0 or @as(u64, @intCast(end)) > limit) return error.TooBig;
    const out = try a.alloc(u8, @intCast(end));
    var got: usize = 0;
    while (got < out.len) {
        const n = std.c.pread(fd, out.ptr + got, out.len - got, @intCast(got));
        if (n <= 0) return error.ReadFailed;
        got += @intCast(n);
    }
    return out;
}

fn sha256File(path: [:0]const u8) !struct { [32]u8, u64 } {
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
    if (fd < 0) return error.FileNotFound;
    defer _ = std.c.close(fd);
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    var buf: [1 << 16]u8 = undefined;
    var total: u64 = 0;
    while (true) {
        const n = std.c.read(fd, &buf, buf.len);
        if (n < 0) return error.ReadFailed;
        if (n == 0) break;
        h.update(buf[0..@intCast(n)]);
        total += @intCast(n);
    }
    return .{ h.finalResult(), total };
}

// ---- .npy and .npz as numpy writes them ----

/// numpy.lib.format's header for a C-order array of `descr` ("<f4") and `shape`, version 1.0.
pub fn npyHeader(a: std.mem.Allocator, descr: []const u8, shape: []const usize) ![]u8 {
    var dict: std.Io.Writer.Allocating = .init(a);
    defer dict.deinit();
    const w = &dict.writer;
    try w.print("{{'descr': '{s}', 'fortran_order': False, 'shape': (", .{descr});
    for (shape, 0..) |d, i| try w.print("{s}{d}", .{ if (i > 0) ", " else "", d });
    if (shape.len == 1) try w.writeAll(",");
    try w.writeAll("), }");
    if (shape.len > 0) {
        var digits: [24]u8 = undefined;
        const first = try std.fmt.bufPrint(&digits, "{d}", .{shape[0]});
        try w.splatByteAll(' ', 21 - first.len);
    }
    const body = dict.written();
    const hlen = body.len + 1;
    const padlen = 64 - ((8 + 2 + hlen) % 64);
    if (hlen + padlen > std.math.maxInt(u16)) return error.HeaderTooBig;
    const out = try a.alloc(u8, 10 + hlen + padlen);
    @memcpy(out[0..8], "\x93NUMPY\x01\x00");
    std.mem.writeInt(u16, out[8..10], @intCast(hlen + padlen), .little);
    @memcpy(out[10..][0..body.len], body);
    @memset(out[10 + body.len ..][0..padlen], ' ');
    out[out.len - 1] = '\n';
    return out;
}

fn descrOf(comptime T: type) []const u8 {
    return switch (T) {
        f32 => "<f4",
        f64 => "<f8",
        i32 => "<i4",
        i16 => "<i2",
        u32 => "<u4",
        else => @compileError("no npy descr for " ++ @typeName(T)),
    };
}

/// A whole array in memory as .npy bytes.
pub fn npyBytes(a: std.mem.Allocator, comptime T: type, shape: []const usize, data: []const T) ![]u8 {
    var n: usize = 1;
    for (shape) |d| n *= d;
    if (n != data.len) return error.BadShape;
    const head = try npyHeader(a, descrOf(T), shape);
    defer a.free(head);
    const out = try a.alloc(u8, head.len + n * @sizeOf(T));
    @memcpy(out[0..head.len], head);
    @memcpy(out[head.len..], std.mem.sliceAsBytes(data));
    return out;
}

/// A .npy file written in place: the header, then elements at their offsets (unwritten ones read 0).
const NpyFile = struct {
    fd: std.c.fd_t,
    data: u64,
    count: usize,

    fn make(a: std.mem.Allocator, path: [:0]const u8, descr: []const u8, item: usize, shape: []const usize) !NpyFile {
        var n: usize = 1;
        for (shape) |d| n *= d;
        const head = try npyHeader(a, descr, shape);
        defer a.free(head);
        const fd = try createFile(path);
        errdefer _ = std.c.close(fd);
        try pwriteAll(fd, head, 0);
        if (std.c.ftruncate(fd, @intCast(head.len + n * item)) != 0) return error.WriteFailed;
        return .{ .fd = fd, .data = head.len, .count = n };
    }

    fn put(f: NpyFile, comptime T: type, at: usize, values: []const T) !void {
        if (at + values.len > f.count) return error.OutOfRange;
        try pwriteAll(f.fd, std.mem.sliceAsBytes(values), f.data + at * @sizeOf(T));
    }

    fn close(f: NpyFile) void {
        _ = std.c.close(f.fd);
    }
};

/// A zip of stored .npy members as numpy.savez writes one (no compression; CRC-32 and sizes in the headers).
pub fn npz(a: std.mem.Allocator, names: []const []const u8, members: []const []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    var central: std.ArrayList(u8) = .empty;
    defer central.deinit(a);
    const le = struct {
        fn u16_(l: *std.ArrayList(u8), al: std.mem.Allocator, v: u16) !void {
            var b: [2]u8 = undefined;
            std.mem.writeInt(u16, &b, v, .little);
            try l.appendSlice(al, &b);
        }
        fn u32_(l: *std.ArrayList(u8), al: std.mem.Allocator, v: u32) !void {
            var b: [4]u8 = undefined;
            std.mem.writeInt(u32, &b, v, .little);
            try l.appendSlice(al, &b);
        }
    };
    for (names, members) |name, data| {
        const file = try std.fmt.allocPrint(a, "{s}.npy", .{name});
        defer a.free(file);
        if (data.len > std.math.maxInt(u32) or out.items.len > std.math.maxInt(u32)) return error.TooBigForZip;
        const crc = std.hash.Crc32.hash(data);
        const local: u32 = @intCast(out.items.len);
        try le.u32_(&out, a, 0x04034b50);
        for ([_]u16{ 20, 0, 0, 0, 0x21 }) |v| try le.u16_(&out, a, v); // version, flags, stored, time, date 1980-01-01
        try le.u32_(&out, a, crc);
        try le.u32_(&out, a, @intCast(data.len));
        try le.u32_(&out, a, @intCast(data.len));
        try le.u16_(&out, a, @intCast(file.len));
        try le.u16_(&out, a, 0);
        try out.appendSlice(a, file);
        try out.appendSlice(a, data);
        try le.u32_(&central, a, 0x02014b50);
        for ([_]u16{ 20, 20, 0, 0, 0, 0x21 }) |v| try le.u16_(&central, a, v);
        try le.u32_(&central, a, crc);
        try le.u32_(&central, a, @intCast(data.len));
        try le.u32_(&central, a, @intCast(data.len));
        for ([_]u16{ @intCast(file.len), 0, 0, 0, 0 }) |v| try le.u16_(&central, a, v);
        try le.u32_(&central, a, 0o100644 << 16); // external attributes: a regular file
        try le.u32_(&central, a, local);
        try central.appendSlice(a, file);
    }
    const at: u32 = @intCast(out.items.len);
    try out.appendSlice(a, central.items);
    try le.u32_(&out, a, 0x06054b50);
    for ([_]u16{ 0, 0, @intCast(names.len), @intCast(names.len) }) |v| try le.u16_(&out, a, v);
    try le.u32_(&out, a, @intCast(central.items.len));
    try le.u32_(&out, a, at);
    try le.u16_(&out, a, 0);
    return out.toOwnedSlice(a);
}

// ---- the reference run's prompts ----

pub const Prompt = struct {
    name: []const u8,
    greedy: bool,
    ids: []const u32,
    /// Tokens run_ref.py generates: --gen for a greedy prompt, 1 for a teacher-forced one.
    gen: u32,
};

/// The prompts of reference run `ref`: prompts.json's names, kinds and gen, the ids from each <name>/ids.u32, which
/// must equal prompts.json's and match <ref>/ids.sha256.
pub fn prompts(a: std.mem.Allocator, ref: []const u8) ![]Prompt {
    const text = try readFile(a, try cpath(a, &.{ ref, "prompts.json" }), 1 << 28);
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
    if (doc != .array) return error.BadPrompts;
    const sums = readFile(a, try cpath(a, &.{ ref, "ids.sha256" }), 1 << 20) catch return error.NoIdsFiles;
    var out: std.ArrayList(Prompt) = .empty;
    for (doc.array.items) |o| {
        if (o != .object) return error.BadPrompts;
        const name = (o.object.get("name") orelse return error.BadPrompts).string;
        if (name.len == 0 or name[0] == '.' or std.mem.indexOfAny(u8, name, "/\\") != null) return error.BadPrompts;
        const bytes = try readFile(a, try cpath(a, &.{ ref, name, "ids.u32" }), 1 << 26);
        if (bytes.len % 4 != 0 or bytes.len == 0) return error.BadIdsFile;
        const ids = try a.alloc(u32, bytes.len / 4);
        for (ids, 0..) |*id, i| id.* = std.mem.readInt(u32, bytes[4 * i ..][0..4], .little);
        const listed = (o.object.get("ids") orelse return error.BadPrompts).array.items;
        if (listed.len != ids.len) return error.IdsDiffer;
        for (listed, ids) |v, id| if (v != .integer or v.integer != id) return error.IdsDiffer;
        var sum: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(bytes, &sum, .{});
        const line = try std.fmt.allocPrint(a, "{s}  {s}/ids.u32", .{ &std.fmt.bytesToHex(sum, .lower), name });
        if (std.mem.indexOf(u8, sums, line) == null) return error.IdsHashDiffers;
        const kind = if (o.object.get("kind")) |k| k.string else "greedy";
        const gen: u32 = if (o.object.get("gen")) |g| @intCast(g.integer) else 1;
        try out.append(a, .{ .name = name, .greedy = std.mem.eql(u8, kind, "greedy"), .ids = ids, .gen = @max(gen, 1) });
    }
    return out.items;
}

// ---- one prompt's dumps ----

/// A row's top 8 (descending, ties to the lower id) of `lg`, its full logsumexp in float64.
pub fn top8(lg: []const f32) struct { ids: [top_k]i32, vals: [top_k]f64, lse: f64 } {
    var ids: [top_k]i32 = @splat(-1);
    var vals: [top_k]f64 = @splat(-std.math.inf(f64));
    var m: f64 = -std.math.inf(f64);
    for (lg, 0..) |x, i| {
        const v: f64 = x;
        if (v > m) m = v;
        if (!(v > vals[top_k - 1])) continue;
        var k: usize = top_k - 1;
        while (k > 0 and v > vals[k - 1]) : (k -= 1) {
            vals[k] = vals[k - 1];
            ids[k] = ids[k - 1];
        }
        vals[k] = v;
        ids[k] = @intCast(i);
    }
    var s: f64 = 0;
    for (lg) |x| s += @exp(@as(f64, x) - m);
    return .{ .ids = ids, .vals = vals, .lse = m + @log(s) };
}

const Top = struct {
    ids: []i32,
    vals: []f64,
    lse: []f64,
    target: []i32,
    tl: []f64,
    rows: usize = 0,

    fn init(a: std.mem.Allocator, n: usize) !Top {
        const t: Top = .{ .ids = try a.alloc(i32, n * top_k), .vals = try a.alloc(f64, n * top_k), .lse = try a.alloc(f64, n), .target = try a.alloc(i32, n), .tl = try a.alloc(f64, n) };
        @memset(t.ids, -1);
        @memset(t.vals, std.math.nan(f64));
        @memset(t.lse, std.math.nan(f64));
        @memset(t.target, -1);
        @memset(t.tl, std.math.nan(f64));
        return t;
    }

    fn set(t: *Top, row: usize, lg: []const f32, target: i32) void {
        const r = top8(lg);
        @memcpy(t.ids[row * top_k ..][0..top_k], &r.ids);
        @memcpy(t.vals[row * top_k ..][0..top_k], &r.vals);
        t.lse[row] = r.lse;
        t.target[row] = target;
        t.tl[row] = if (target >= 0) lg[@intCast(target)] else std.math.nan(f64);
        t.rows = @max(t.rows, row + 1);
    }

    fn file(t: *const Top, a: std.mem.Allocator, n: usize) ![]u8 {
        const names = [_][]const u8{ "ids", "logits", "lse", "target", "target_logit" };
        const m = [_][]const u8{
            try npyBytes(a, i32, &.{ n, top_k }, t.ids[0 .. n * top_k]),
            try npyBytes(a, f64, &.{ n, top_k }, t.vals[0 .. n * top_k]),
            try npyBytes(a, f64, &.{n}, t.lse[0..n]),
            try npyBytes(a, i32, &.{n}, t.target[0..n]),
            try npyBytes(a, f64, &.{n}, t.tl[0..n]),
        };
        return npz(a, &names, &m);
    }
};

const Step = struct { token: u32, margin: f64, top: [top_k]i32, logprob: [top_k]f64 };

fn bf16Widen(out: []f32, in: []const u16) void {
    for (out, in) |*o, b| o.* = @bitCast(@as(u32, b) << 16);
}

pub const Seq = struct {
    a: std.mem.Allocator,
    dump: *Dump,
    p: Prompt,
    dir: [:0]u8,
    /// Prompt position -> its row in the dumps, or -1.
    slot: []i32,
    rows: []i32,
    rows_max: usize,
    streams: NpyFile,
    sub: NpyFile,
    /// Made at the first MTP row: a run without the MTP pass writes no MTP files (as run_ref.py).
    mtp_x: ?NpyFile = null,
    mtp_io: ?NpyFile = null,
    logits: ?NpyFile,
    picks: []i16,
    wts: []f32,
    prompt_top: Top,
    mtp_top: Top,
    hidden: []f32,
    gen: std.ArrayList(u32) = .empty,
    steps: std.ArrayList(Step) = .empty,
    gen_logits: std.ArrayList(f32) = .empty,
    eos: bool = false,
    files: std.ArrayList([]const u8) = .empty,

    /// The dump row of prompt position `pos`, or null when it is not dumped.
    pub fn slotOf(s: *const Seq, pos: usize) ?usize {
        if (pos >= s.slot.len or s.slot[pos] < 0) return null;
        return @intCast(s.slot[pos]);
    }

    /// The 4 streams [S][D] after layer `l`'s mixer write-back (`site` 0) or its FFN write-back (1), at `pos`.
    pub fn stream(s: *Seq, l: usize, pos: usize, site: u1, x: []const f32) !void {
        const j = s.slotOf(pos) orelse return;
        if (x.len != S * D or l >= L) return error.BadShape;
        try s.streams.put(f32, ((l * s.rows.len + j) * 2 + site) * S * D, x);
    }

    pub fn streamBf16(s: *Seq, l: usize, pos: usize, site: u1, x: []const u16) !void {
        if (s.slotOf(pos) == null) return;
        var w: [S * D]f32 = undefined;
        if (x.len != w.len) return error.BadShape;
        bf16Widen(&w, x);
        return s.stream(l, pos, site, &w);
    }

    /// Layer `l`'s mixer input (0), mixer output (1), FFN input (2), FFN output (3) at `pos`, [D].
    pub fn sublayer(s: *Seq, l: usize, pos: usize, k: u2, x: []const f32) !void {
        const j = s.slotOf(pos) orelse return;
        if (x.len != D or l >= L) return error.BadShape;
        try s.sub.put(f32, ((l * s.rows.len + j) * 4 + k) * D, x);
    }

    pub fn sublayerBf16(s: *Seq, l: usize, pos: usize, k: u2, x: []const u16) !void {
        if (s.slotOf(pos) == null) return;
        var w: [D]f32 = undefined;
        if (x.len != D) return error.BadShape;
        bf16Widen(&w, x);
        return s.sublayer(l, pos, k, &w);
    }

    /// The MTP layer's residual [D] after its attention (0) and its MoE (1) at row `pos` (h of pos, token pos + 1).
    pub fn mtpStream(s: *Seq, pos: usize, site: u1, x: []const f32) !void {
        const j = s.slotOf(pos) orelse return;
        if (x.len != D) return error.BadShape;
        try s.mtpFiles();
        try s.mtp_x.?.put(f32, (j * 2 + site) * D, x);
    }

    pub fn mtpSublayer(s: *Seq, pos: usize, k: u2, x: []const f32) !void {
        const j = s.slotOf(pos) orelse return;
        if (x.len != D) return error.BadShape;
        try s.mtpFiles();
        try s.mtp_io.?.put(f32, (j * 4 + k) * D, x);
    }

    fn mtpFiles(s: *Seq) !void {
        if (s.mtp_x != null) return;
        const rd = s.rows.len;
        s.mtp_x = try NpyFile.make(s.a, try cpath(s.a, &.{ s.dir, "mtp_streams.npy" }), "<f4", 4, &.{ rd, 2, D });
        s.mtp_io = try NpyFile.make(s.a, try cpath(s.a, &.{ s.dir, "mtp_sublayers.npy" }), "<f4", 4, &.{ rd, 4, D });
    }

    /// Layer `layer`'s (45: the MTP layer) 8 routed picks of sequence row `row` in slot order, and their weights.
    pub fn router(s: *Seq, layer: usize, row: usize, ids: []const i32, w: []const f32) !void {
        if (layer >= router_layers or row >= s.rows_max or ids.len != top_k or w.len != top_k) return error.BadShape;
        for (ids, s.picks[(layer * s.rows_max + row) * top_k ..][0..top_k]) |id, *p| p.* = @intCast(id);
        @memcpy(s.wts[(layer * s.rows_max + row) * top_k ..][0..top_k], w);
    }

    /// Prompt row `row`'s logits [V]: its top 8, logsumexp, next id and its logit; the whole row up to 1,100 rows.
    /// The last prompt row's logits are greedy step 0.
    pub fn promptLogits(s: *Seq, row: usize, lg: []const f32) !void {
        const t = s.p.ids.len;
        if (row >= t or lg.len != V) return error.BadShape;
        s.prompt_top.set(row, lg, if (row + 1 < t) @intCast(s.p.ids[row + 1]) else -1);
        if (s.logits) |f| try f.put(f32, row * V, lg);
        if (row == t - 1) _ = try s.note(lg);
    }

    /// Sequence row `row`'s final-normed hidden [D] (the input of lm_head and of the MTP layer).
    pub fn hiddenRow(s: *Seq, row: usize, x: []const f32) !void {
        if (row >= s.rows_max or x.len != D) return error.BadShape;
        @memcpy(s.hidden[row * D ..][0..D], x);
    }

    /// A decode step's logits [V] (steps 1 .. gen - 1; step 0 is the last prompt row's): the next greedy token.
    pub fn step(s: *Seq, lg: []const f32) !u32 {
        if (lg.len != V) return error.BadShape;
        try s.gen_logits.appendSlice(s.a, lg);
        return s.note(lg);
    }

    /// The tokens generated so far.
    pub fn generated(s: *const Seq) []const u32 {
        return s.gen.items;
    }

    /// Whether the greedy run is over: an end id, or gen tokens.
    pub fn done(s: *const Seq) bool {
        return s.eos or s.gen.items.len >= s.p.gen;
    }

    fn note(s: *Seq, lg: []const f32) !u32 {
        const r = top8(lg);
        var st: Step = .{ .token = @intCast(r.ids[0]), .margin = r.vals[0] - r.vals[1], .top = r.ids, .logprob = undefined };
        for (&st.logprob, r.vals) |*p, v| p.* = v - r.lse;
        try s.steps.append(s.a, st);
        try s.gen.append(s.a, st.token);
        for (s.dump.eos) |e| if (e == st.token) {
            s.eos = true;
        };
        return st.token;
    }

    /// The MTP layer's logits [V] at sequence row `row` (it predicts the token at row + 2).
    pub fn mtpLogits(s: *Seq, row: usize, lg: []const f32) !void {
        if (row >= s.rows_max or lg.len != V) return error.BadShape;
        const seq_len = s.p.ids.len + s.gen.items.len;
        const target: i32 = if (row + 2 < seq_len) @intCast(s.tokenAt(row + 2)) else -1;
        s.mtp_top.set(row, lg, target);
    }

    fn tokenAt(s: *const Seq, i: usize) u32 {
        return if (i < s.p.ids.len) s.p.ids[i] else s.gen.items[i - s.p.ids.len];
    }

    fn put(s: *Seq, name: []const u8, bytes: []const u8) !void {
        try writeFile(try cpath(s.a, &.{ s.dir, name }), bytes);
        try s.files.append(s.a, name);
    }

    fn jsonFloat(w: *std.Io.Writer, v: f64) !void {
        if (std.math.isNan(v)) return w.writeAll("NaN");
        if (std.math.isInf(v)) return w.writeAll(if (v > 0) "Infinity" else "-Infinity");
        try w.print("{e}", .{v});
    }

    /// Writes the prompt's files; the open .npy files are closed.
    pub fn close(s: *Seq) !void {
        const a = s.a;
        const t = s.p.ids.len;
        const n = t + s.gen.items.len - 1;
        s.streams.close();
        s.sub.close();
        if (s.logits) |f| f.close();
        try s.files.appendSlice(a, &.{ "streams.npy", "sublayers.npy" });
        if (s.logits != null) try s.files.append(a, "logits.npy");
        if (s.mtp_x) |f| {
            f.close();
            s.mtp_io.?.close();
            try s.files.appendSlice(a, &.{ "mtp_streams.npy", "mtp_sublayers.npy" });
        }
        try s.put("dump_rows.npy", try npyBytes(a, i32, &.{s.rows.len}, s.rows));
        const picks = try a.alloc(i16, router_layers * n * top_k);
        const wts = try a.alloc(f32, router_layers * n * top_k);
        for (0..router_layers) |l| {
            @memcpy(picks[l * n * top_k ..][0 .. n * top_k], s.picks[l * s.rows_max * top_k ..][0 .. n * top_k]);
            @memcpy(wts[l * n * top_k ..][0 .. n * top_k], s.wts[l * s.rows_max * top_k ..][0 .. n * top_k]);
        }
        try s.put("router_picks.npy", try npyBytes(a, i16, &.{ router_layers, n, top_k }, picks));
        try s.put("router_weights.npy", try npyBytes(a, f32, &.{ router_layers, n, top_k }, wts));
        try s.put("logits_top.npz", try s.prompt_top.file(a, t));
        if (s.mtp_top.rows > 0) try s.put("mtp_logits_top.npz", try s.mtp_top.file(a, n));
        try s.put("hidden.npy", try npyBytes(a, f32, &.{ n, D }, s.hidden[0 .. n * D]));
        if (s.gen.items.len > 1) try s.put("gen_logits.npy", try npyBytes(a, f32, &.{ s.gen.items.len - 1, V }, s.gen_logits.items));
        // greedy.json: run_ref.py's keys
        var j: std.Io.Writer.Allocating = .init(a);
        const w = &j.writer;
        try w.print("{{\"prompt_rows\": {d}, \"ids\": [", .{t});
        for (s.gen.items, 0..) |id, i| try w.print("{s}{d}", .{ if (i > 0) ", " else "", id });
        try w.print("], \"text\": null, \"eos\": {any}, \"steps\": [", .{s.eos});
        for (s.steps.items, 0..) |st, i| {
            try w.print("{s}{{\"token\": {d}, \"margin\": ", .{ if (i > 0) ", " else "", st.token });
            try jsonFloat(w, st.margin);
            try w.writeAll(", \"top8\": [");
            for (st.top, 0..) |id, k| try w.print("{s}{d}", .{ if (k > 0) ", " else "", id });
            try w.writeAll("], \"top8_logprob\": [");
            for (st.logprob, 0..) |v, k| {
                if (k > 0) try w.writeAll(", ");
                try jsonFloat(w, v);
            }
            try w.writeAll("]}");
        }
        // teacher-forced over the prompt's rows that have a next id
        var nll: f64 = 0;
        var hit: usize = 0;
        for (0..t - 1) |r| {
            nll += s.prompt_top.lse[r] - s.prompt_top.tl[r];
            hit += @intFromBool(s.prompt_top.ids[r * top_k] == s.prompt_top.target[r]);
        }
        const rows_tf: f64 = @floatFromInt(@max(t - 1, 1));
        try w.print("], \"teacher_forced\": {{\"rows\": {d}, \"nll_mean\": ", .{t - 1});
        try jsonFloat(w, nll / rows_tf);
        try w.writeAll(", \"ppl\": ");
        try jsonFloat(w, @exp(nll / rows_tf));
        try w.writeAll(", \"top1_is_next\": ");
        try jsonFloat(w, @as(f64, @floatFromInt(hit)) / rows_tf);
        try w.writeAll("}");
        if (s.mtp_top.rows > 0) {
            var hits: usize = 0;
            var with: usize = 0;
            for (0..n) |r| if (s.mtp_top.target[r] >= 0) {
                with += 1;
                hits += @intFromBool(s.mtp_top.ids[r * top_k] == s.mtp_top.target[r]);
            };
            try w.print(", \"mtp_teacher_forced\": {{\"rows\": {d}, \"top1_is_next_next\": ", .{n});
            try jsonFloat(w, @as(f64, @floatFromInt(hits)) / @as(f64, @floatFromInt(@max(with, 1))));
            try w.writeAll("}");
        }
        try w.writeAll("}\n");
        try s.put("greedy.json", j.written());
        try s.dump.finished.append(s.dump.a, s);
    }
};

pub const Dump = struct {
    a: std.mem.Allocator,
    out: []const u8,
    ref: []const u8,
    prompts: []Prompt,
    eos: []const u32,
    finished: std.ArrayList(*Seq) = .empty,

    /// TF_GLM_DUMP and TF_GLM_DUMP_REF from the environment; null when TF_GLM_DUMP is unset.
    pub fn fromEnv(a: std.mem.Allocator, eos: []const u32) !?Dump {
        const out = std.c.getenv("TF_GLM_DUMP") orelse return null;
        const ref = std.c.getenv("TF_GLM_DUMP_REF") orelse return error.NoDumpRef;
        return try open(a, std.mem.span(out), std.mem.span(ref), eos);
    }

    /// A dump into `out` of reference run `ref`'s prompts (`a` an arena: everything lives until it goes).
    pub fn open(a: std.mem.Allocator, out: []const u8, ref: []const u8, eos: []const u32) !Dump {
        try mkdir(try a.dupeSentinel(u8, out, 0));
        return .{ .a = a, .out = out, .ref = ref, .prompts = try prompts(a, ref), .eos = eos };
    }

    /// Prompt `p`'s dumps: its dump rows (the first 64 and the last 8 positions), the files made.
    pub fn begin(d: *Dump, p: Prompt) !*Seq {
        const a = d.a;
        const t = p.ids.len;
        const dir = try cpath(a, &.{ d.out, p.name });
        try mkdir(dir);
        const s = try a.create(Seq);
        const slot = try a.alloc(i32, t);
        @memset(slot, -1);
        var rows: std.ArrayList(i32) = .empty;
        for (0..t) |i| if (i < dump_head or i + dump_tail >= t) {
            slot[i] = @intCast(rows.items.len);
            try rows.append(a, @intCast(i));
        };
        const rd = rows.items.len;
        const rows_max = t + p.gen - 1;
        s.* = .{
            .a = a,
            .dump = d,
            .p = p,
            .dir = dir,
            .slot = slot,
            .rows = rows.items,
            .rows_max = rows_max,
            .streams = try NpyFile.make(a, try cpath(a, &.{ dir, "streams.npy" }), "<f4", 4, &.{ L, rd, 2, S, D }),
            .sub = try NpyFile.make(a, try cpath(a, &.{ dir, "sublayers.npy" }), "<f4", 4, &.{ L, rd, 4, D }),
            .logits = if (t <= full_logits_rows) try NpyFile.make(a, try cpath(a, &.{ dir, "logits.npy" }), "<f4", 4, &.{ t, V }) else null,
            .picks = try a.alloc(i16, router_layers * rows_max * top_k),
            .wts = try a.alloc(f32, router_layers * rows_max * top_k),
            .prompt_top = try Top.init(a, t),
            .mtp_top = try Top.init(a, rows_max),
            .hidden = try a.alloc(f32, rows_max * D),
        };
        @memset(s.picks, -1);
        @memset(s.wts, 0);
        @memset(s.hidden, 0);
        return s;
    }

    /// prompts.json (the reference's, as read), hashes.tsv over the prompts written, and manifest-hip.json.
    pub fn finish(d: *Dump, note: []const u8) !void {
        const a = d.a;
        try writeFile(try cpath(a, &.{ d.out, "prompts.json" }), try readFile(a, try cpath(a, &.{ d.ref, "prompts.json" }), 1 << 28));
        var h: std.Io.Writer.Allocating = .init(a);
        try h.writer.writeAll("sha256\tbytes\tfile\n");
        for (d.finished.items) |s| {
            const names = try a.dupe([]const u8, s.files.items);
            std.mem.sort([]const u8, names, {}, struct {
                fn less(_: void, x: []const u8, y: []const u8) bool {
                    return std.mem.lessThan(u8, x, y);
                }
            }.less);
            for (names) |name| {
                const sum, const size = try sha256File(try cpath(a, &.{ s.dir, name }));
                try h.writer.print("{s}\t{d}\t{s}/{s}\n", .{ &std.fmt.bytesToHex(sum, .lower), size, s.p.name, name });
            }
            const ids = try a.alloc(u8, 4 * s.gen.items.len);
            for (s.gen.items, 0..) |id, i| std.mem.writeInt(u32, ids[4 * i ..][0..4], id, .little);
            var sum: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(ids, &sum, .{});
            try h.writer.print("{s}\t{d}\t{s}/greedy-ids(int32 le)\n", .{ &std.fmt.bytesToHex(sum, .lower), ids.len, s.p.name });
        }
        try writeFile(try cpath(a, &.{ d.out, "hashes.tsv" }), h.written());
        var m: std.Io.Writer.Allocating = .init(a);
        try m.writer.print("{{\"engine\": \"tensorfold-hip\", \"ref\": \"{s}\", \"note\": \"{s}\", \"dump_rows\": \"first {d} and last {d} prompt positions\", \"prompts\": [", .{ d.ref, note, dump_head, dump_tail });
        for (d.finished.items, 0..) |s, i| try m.writer.print("{s}{{\"name\": \"{s}\", \"rows\": {d}, \"gen\": {d}, \"eos\": {any}}}", .{ if (i > 0) ", " else "", s.p.name, s.p.ids.len, s.gen.items.len, s.eos });
        try m.writer.writeAll("]}\n");
        try writeFile(try cpath(a, &.{ d.out, "manifest-hip.json" }), m.written());
    }
};

test "glm dump: npy headers are numpy 2.2's bytes" {
    const a = std.testing.allocator;
    const h = try npyHeader(a, "<f4", &.{ 45, 72, 2, 4, 4096 });
    defer a.free(h);
    const want = "\x93NUMPY\x01\x00v\x00{'descr': '<f4', 'fortran_order': False, 'shape': (45, 72, 2, 4, 4096), }";
    try std.testing.expectEqualStrings(want, h[0..want.len]);
    try std.testing.expectEqual(@as(usize, 128), h.len);
    try std.testing.expectEqual(@as(u8, '\n'), h[h.len - 1]);
    const one = try npyHeader(a, "<i4", &.{72});
    defer a.free(one);
    const want1 = "\x93NUMPY\x01\x00v\x00{'descr': '<i4', 'fortran_order': False, 'shape': (72,), }";
    try std.testing.expectEqualStrings(want1, one[0..want1.len]);
    try std.testing.expectEqual(@as(usize, 128), one.len);
    // the core reader takes it back
    const core_npy = @import("core").npy;
    const bytes = try npyBytes(a, i16, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer a.free(bytes);
    const arr = try core_npy.parse(bytes);
    try std.testing.expectEqualStrings("<i2", arr.descr);
    try std.testing.expectEqual(@as(usize, 2), arr.rank);
    try std.testing.expectEqual(@as(usize, 3), arr.shape[1]);
}

test "glm dump: the top 8 descend with ties to the lower id; logsumexp in float64" {
    var lg: [20]f32 = @splat(0);
    lg[3] = 5;
    lg[7] = 5;
    lg[1] = 4;
    lg[19] = 6;
    const r = top8(&lg);
    try std.testing.expectEqualSlices(i32, &.{ 19, 3, 7, 1, 0, 2, 4, 5 }, &r.ids);
    var s: f64 = 0;
    for (lg) |x| s += @exp(@as(f64, x));
    try std.testing.expectApproxEqRel(@log(s), r.lse, 1e-15);
}

test "glm dump: an npz is a stored zip whose members are the arrays" {
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const x = try npyBytes(arena.allocator(), i32, &.{2}, &.{ 7, 8 });
    const z = try npz(arena.allocator(), &.{"ids"}, &.{x});
    try std.testing.expectEqual(@as(u32, 0x04034b50), std.mem.readInt(u32, z[0..4], .little));
    try std.testing.expectEqualStrings("ids.npy", z[30..37]);
    try std.testing.expectEqualSlices(u8, x, z[37..][0..x.len]);
    try std.testing.expectEqual(@as(u32, 0x06054b50), std.mem.readInt(u32, z[z.len - 22 ..][0..4], .little));
}
