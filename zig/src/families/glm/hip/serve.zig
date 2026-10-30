//! GLM-5.3-Flash served by the lane core on HIP: the weights (open.zig), lane F's eager forward (forward.zig) for
//! prompt chunks and decode windows, one state a stream, the draw on the head's fp32 logits.
//! A stream that drafts (MTP, d <= 3) runs lane F's rounds: the prompt's rows j < t - 1 through the MTP layer
//! (`mtp`), a verify window of the pending token and the drafts (`window`, held: nothing kept), the keep of the
//! accepted rows (`keep`), and the next drafts from the kept rows (`draft`). A stream that does not draft runs the
//! serial forward (`run`), as before: its prompt and decode work is unchanged.

const std = @import("std");
const hip = @import("hip");
const lanes = @import("lanes");
const serve = @import("hip_serve");
const open = @import("open.zig");
const load = @import("load.zig");
const forward = @import("forward.zig");
const state = @import("state.zig");

const be = lanes.backend;
const draw = serve.draw;
const glue = serve.backend;

const D: usize = 4096;
const V: usize = 154880;

/// Drafts a round holds at most (the CUDA GLM path's 3, `cuda_engine.zig:25-26`).
pub const max_drafts = 3;
/// Rows a window holds at most: the pending token and the drafts.
pub const max_window = 1 + max_drafts;

/// The device side: the weights, the launcher, the prompt and decode windows of the forward.
pub const Engine = struct {
    gpa: std.mem.Allocator,
    r: hip.Runtime,
    ctx: hip.Context,
    stream: hip.Stream,
    l: hip.Launcher,
    opened: open.Opened,
    pb: state.Buffers,
    db: state.Buffers,
    /// The draft rounds' own decode buffers (lane F: `draft` overwrites its Forward's rows).
    mb: state.Buffers,
    fp: forward.Forward,
    fd: forward.Forward,
    fm: forward.Forward,
    /// A prompt chunk's final-normed rows, the MTP layer's input in the prompt pass (bf16 [chunk][D]).
    phid: hip.DeviceBuffer,
    /// Drafts a round asks for at most (0: no stream drafts; the MTP prompt pass is skipped).
    drafts: usize,
    /// Positions a stream's state holds: align64(window + 8).
    capacity: usize,
    /// Prompt rows a forward takes at once.
    chunk: usize,
    /// One logits row on the host (fp32).
    row: []f32,

    /// The weights of `dir` on device `index`, then the windows; `problem` (in `a`) says why not.
    pub fn create(gpa: std.mem.Allocator, a: std.mem.Allocator, io: std.Io, index: c_int, dir: []const u8, capacity: usize, chunk: usize, drafts: usize, problem: *[]const u8) !*Engine {
        if (drafts > max_drafts) return error.TooManyDrafts;
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.gpa = gpa;
        e.capacity = capacity;
        e.chunk = chunk;
        e.drafts = drafts;
        e.r = try hip.Runtime.open();
        errdefer e.r.close();
        e.ctx = try hip.Context.init(&e.r, index);
        errdefer e.ctx.deinit();
        e.stream = try hip.Stream.init(&e.r);
        errdefer e.stream.deinit();
        e.opened = try open.open(gpa, a, io, &e.r, dir, .{}, problem);
        errdefer e.opened.deinit();
        // numerics=exact: the launcher's defaults, as lane F's CLI loads them
        e.l = try hip.Launcher.load(&e.r, .{}, hip.kernels.images);
        errdefer e.l.unload();
        e.pb = try state.Buffers.init(&e.r, chunk, true, capacity);
        errdefer e.pb.deinit();
        e.db = try state.Buffers.init(&e.r, state.window_rows, false, capacity);
        errdefer e.db.deinit();
        e.mb = try state.Buffers.init(&e.r, state.window_rows, false, capacity);
        errdefer e.mb.deinit();
        e.phid = try hip.DeviceBuffer.alloc(&e.r, if (drafts > 0) chunk * D * 2 else 0);
        errdefer e.phid.free();
        const cfg = e.opened.ck.config;
        e.fp = try forward.Forward.init(&e.l, e.opened.model(), cfg, e.stream.handle, &e.pb);
        e.fd = try forward.Forward.init(&e.l, e.opened.model(), cfg, e.stream.handle, &e.db);
        e.fm = try forward.Forward.init(&e.l, e.opened.model(), cfg, e.stream.handle, &e.mb);
        e.row = try gpa.alloc(f32, V);
        return e;
    }

    pub fn deinit(e: *Engine) void {
        e.stream.synchronize() catch {};
        e.gpa.free(e.row);
        e.phid.free();
        e.mb.deinit();
        e.db.deinit();
        e.pb.deinit();
        e.l.unload();
        e.opened.deinit();
        e.stream.deinit();
        e.ctx.deinit();
        e.r.close();
        e.gpa.destroy(e);
    }

    /// Device bytes of the two windows' buffers and arenas (the forward's scratch).
    pub fn scratchBytes(e: *const Engine) u64 {
        return e.pb.buf.len + e.pb.arena.len + e.db.buf.len + e.db.arena.len + e.mb.buf.len + e.mb.arena.len + e.phid.len;
    }

    pub fn eos(e: *const Engine) []const u32 {
        const c = &e.opened.ck.config;
        return c.eos[0..c.eos_n];
    }
};

pub const Lane = struct {
    st: state.State,
    /// The stream drafts (its request asks for drafts and the engine has them): lane F's MTP rounds.
    mtp: bool = false,
    /// The last window ran held (`window`): its keep goes through lane F's `keep`; else `run` kept every row.
    windowed: bool = false,
    /// Drafts held for the next window.
    held: [max_drafts]u32 = undefined,
    held_n: usize = 0,
    /// The rows the next draft absorbs: the last window's final-normed rows, or the prompt's last row (bf16 [rows][D]).
    hidden: hip.DeviceBuffer,
    /// The prompt pass in progress, a chunk a call (null once it is in).
    fill: ?glue.Fill = null,
    /// The last window's rows until their keep.
    pending: ?glue.Pending = null,
    /// A structured reply's prompt row (bf16), kept for its masked first draw.
    first_row: ?[]u16 = null,
    /// The stream's id (one device: for the shared backend's bookkeeping).
    id: u32 = 0,
};

fn bf16(v: f32) u16 {
    const b: u32 = @bitCast(v);
    if (std.math.isNan(v)) return @intCast((b >> 16) | 0x40);
    return @intCast((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

/// The token of one fp32 logits row per `r`: greedy is the first maximum (ties to the lower id, lane F's rule); a
/// sampled row is drawn among the whole vocabulary in float64; a mask leaves only its allowed ids.
pub fn drawRow(gpa: std.mem.Allocator, row: []const f32, r: draw.Request) !u32 {
    const allowed = struct {
        fn f(mask: []const u32, i: usize) bool {
            return mask.len == 0 or (mask[i / 32] >> @intCast(i % 32)) & 1 != 0;
        }
    }.f;
    const greedy = if (r.sampling) |s| s.temperature <= 0 else true;
    if (greedy) {
        var best: ?usize = null;
        for (row, 0..) |v, i| {
            if (!allowed(r.mask, i) or std.math.isNan(v)) continue;
            if (best == null or v > row[best.?]) best = i;
        }
        return @intCast(best orelse return error.NoAllowedToken);
    }
    var n: usize = 0;
    for (0..row.len) |i| n += @intFromBool(allowed(r.mask, i));
    const values = try gpa.alloc(f64, n);
    defer gpa.free(values);
    const ids = try gpa.alloc(u64, n);
    defer gpa.free(ids);
    var k: usize = 0;
    for (row, 0..) |v, i| if (allowed(r.mask, i)) {
        values[k] = v;
        ids[k] = i;
        k += 1;
    };
    return @intCast(try lanes.sampling.choose(gpa, values, ids, r.position, r.sampling.?));
}

/// TF_GLM_TRACE=1 logs every drawn id with its stream and position (a check against lane F's CLI; never in timed runs).
fn tracing() bool {
    const v = std.c.getenv("TF_GLM_TRACE") orelse return false;
    return v[0] == '1';
}

pub const Hip = struct {
    gpa: std.mem.Allocator,
    e: *Engine,
    streams: glue.Streams(Lane),
    /// The shared backend's drawer: a structured reply's first masked draw from its kept bf16 row.
    sampler: draw.Drawer,
    /// One device: no other rank hears the steps.
    tp: serve.protocol.Sender,
    /// Lanes free for the next request: every stream's state is made at startup.
    spare: std.ArrayList(*Lane) = .empty,
    count: usize,
    trace: bool,

    const Shared = glue.Backend(Hip);
    pub const backend = Shared.backend;

    /// The lane backend over `e` with `count` streams' states made now.
    pub fn init(gpa: std.mem.Allocator, e: *Engine, count: usize) !*Hip {
        const h = try gpa.create(Hip);
        errdefer gpa.destroy(h);
        var sampler = try draw.Drawer.init(gpa, &e.r, .bf16, V, max_window);
        errdefer sampler.deinit();
        h.* = .{ .gpa = gpa, .e = e, .streams = .init(gpa), .sampler = sampler, .tp = .{ .gpa = gpa }, .count = count, .trace = tracing() };
        errdefer h.dropAll();
        for (0..count) |_| {
            const l = try gpa.create(Lane);
            errdefer gpa.destroy(l);
            var st = try state.State.init(&e.r, e.capacity);
            errdefer st.deinit();
            var hidden = try hip.DeviceBuffer.alloc(&e.r, state.window_rows * D * 2);
            errdefer hidden.free();
            l.* = .{ .st = st, .hidden = hidden };
            try h.spare.append(gpa, l);
        }
        return h;
    }

    fn drop(h: *Hip, l: *Lane) void {
        l.hidden.free();
        l.st.deinit();
        h.gpa.destroy(l);
    }

    fn dropAll(h: *Hip) void {
        for (h.spare.items) |l| h.drop(l);
        h.spare.deinit(h.gpa);
    }

    pub fn deinit(h: *Hip) void {
        h.e.stream.synchronize() catch {};
        var it = h.streams.map.valueIterator();
        while (it.next()) |l| h.free(l.*);
        h.dropAll();
        h.streams.deinit();
        h.sampler.deinit();
        h.tp.deinit();
        h.gpa.destroy(h);
    }

    /// The facts the round loop reads: windows of the pending token and up to `drafts` drafts (each row the serial
    /// row's bytes: contract 7), a stream a window, the MTP layer drafting from a verify's kept rows.
    pub fn facts(h: *const Hip) lanes.Model {
        const drafting = h.e.drafts > 0;
        return .{
            .exact_width = @intCast(1 + h.e.drafts),
            .gpu_tokens = false,
            .mtp = drafting,
            .speculate = drafting,
            .speculate_early = false,
            .drafts = @intCast(h.e.drafts),
            .copy_drafts = false,
            .hidden_rows = false,
            .batch_rows = max_window,
            .max_streams = @intCast(h.count),
        };
    }

    pub fn drawer(h: *Hip) *draw.Drawer {
        return &h.sampler;
    }

    /// A lane back to the spare list (its state is reset when a request takes it).
    fn free(h: *Hip, l: *Lane) void {
        if (l.first_row) |row| h.gpa.free(row);
        l.first_row = null;
        l.fill = null;
        l.pending = null;
        h.spare.append(h.gpa, l) catch h.drop(l);
    }

    /// The stream's lane and its prompt pass begun from 0 (no prompt cache).
    pub fn begin(h: *Hip, s: *lanes.Stream) !*Lane {
        const prompt = s.prompt();
        if (prompt.len == 0 or prompt.len + s.max_new + max_window + 1 > h.e.capacity) return error.PromptTooLong;
        const gop = try h.streams.map.getOrPut(h.gpa, s);
        if (gop.found_existing) h.free(gop.value_ptr.*);
        errdefer h.streams.map.removeByPtr(gop.key_ptr);
        const l = h.spare.pop() orelse return error.NoFreeStream;
        errdefer h.free(l);
        try l.st.reset(h.e.stream.handle);
        const st = l.st;
        const hidden = l.hidden;
        l.* = .{ .st = st, .hidden = hidden, .mtp = s.drafts and h.e.drafts > 0 };
        gop.value_ptr.* = l;
        l.id = try h.streams.idOf(s);
        s.cached = 0;
        l.fill = glue.Fill.init(0, &.{});
        return l;
    }

    /// Where the chunk from `at` ends: the engine's chunk rows, or the prompt's end.
    pub fn reach(h: *Hip, _: *Lane, at: usize, limit: usize) !usize {
        return @min(limit, at + h.e.chunk);
    }

    /// The forward checks no cancel inside a chunk: a cancel stops the pass between chunks.
    pub fn cancelOf(_: *Hip, _: *lanes.Stream) ?void {
        return null;
    }

    /// `n` bytes from device address `src` to `dst` on the engine's stream.
    fn copy(e: *Engine, dst: u64, src: u64, n: usize) !void {
        try hip.runtime.check(e.r.api.hipMemcpyDtoDAsync(@ptrFromInt(dst), @ptrFromInt(src), n, e.stream.handle));
    }

    /// A drafting stream's prompt rows [from, from + rows) through the MTP layer, each with the next prompt id
    /// (lane F's `mtp`: row j is h_j and token j + 1). The prompt's last row waits for the first drawn token.
    fn mtpPrompt(h: *Hip, l: *Lane, prompt: []const u32, from: usize, rows: usize) !void {
        if (!l.mtp or rows == 0) return;
        const e = h.e;
        try copy(e, e.phid.base(), e.fp.b.fnormed, rows * D * 2);
        try e.fp.mtp(&l.st, prompt[from + 1 ..][0..rows], e.phid.base());
    }

    pub fn advance(h: *Hip, l: *Lane, prompt: []const u32, from: usize, to: usize, _: ?void) !void {
        try h.e.fp.run(&l.st, prompt[from..to]);
        try h.mtpPrompt(l, prompt, from, to - from);
    }

    /// The prompt's last chunk; its last row's token drawn as `req` says (lm_head on that row alone, as lane F's CLI).
    pub fn finish(h: *Hip, l: *Lane, prompt: []const u32, from: usize, req: draw.Request, _: ?void) !u32 {
        const e = h.e;
        const n = prompt.len - from;
        try e.fp.run(&l.st, prompt[from..]);
        const lg = try e.fp.head(e.fp.b.fnormed + (n - 1) * D * 2, 1);
        try e.fp.download(f32, lg, e.row);
        if (req.keep) |k| for (k, e.row) |*x, v| {
            x.* = bf16(v);
        };
        const token = try drawRow(h.gpa, e.row, req);
        if (l.mtp) {
            // the last row is the first draft's to absorb (with the token just drawn); the rest go through now
            try copy(e, l.hidden.base(), e.fp.b.fnormed + (n - 1) * D * 2, D * 2);
            try h.mtpPrompt(l, prompt, from, n - 1);
        }
        if (h.trace) std.log.info("glm trace {d} first at {d}: {d}", .{ l.id, prompt.len, token });
        return token;
    }

    /// No prompt cache: a pass has no cuts.
    pub fn keepCut(_: *Hip, _: *Lane, _: []const u32, _: usize) !void {}

    /// Each stream's window from its kept length, one forward a stream: the pending token and its drafts, each row
    /// drawn per its request. A drafting stream's window is held (lane F's `window`) until its keep; its final-normed
    /// rows are kept for the next draft. Another stream's window is the serial forward's one row.
    pub fn verify(h: *Hip, ws: []const be.Window, out: []be.Verified) !void {
        const e = h.e;
        for (ws, out) |w, *o| {
            const l = try h.streams.get(w.stream);
            try glue.settle(h, l);
            var toks: [state.window_rows]u32 = undefined;
            const n = try glue.windowTokens(w, l.held[0..l.held_n], l.st.pos, &toks);
            if (n > max_window) return error.WindowTooWide;
            var reqs: [state.window_rows]draw.Request = undefined;
            glue.requests(w, &reqs);
            const pos = l.st.pos;
            if (l.mtp) {
                try e.fd.window(&l.st, toks[0..n]);
                try copy(e, l.hidden.base(), e.fd.b.fnormed, n * D * 2);
            } else {
                if (n > 1) return error.WindowTooWide;
                try e.fd.run(&l.st, toks[0..n]);
            }
            l.windowed = l.mtp;
            const lg = try e.fd.head(e.fd.b.fnormed, n);
            for (0..n) |r| {
                try e.fd.download(f32, lg + r * V * 4, e.row);
                o.sampled[r] = try drawRow(h.gpa, e.row, reqs[r]);
            }
            @memcpy(o.drafts, toks[1..n]);
            l.pending = .{ .rows = n };
            if (h.trace) std.log.info("glm trace {d} at {d}: {any} -> {any}", .{ l.id, pos, toks[0..n], o.sampled[0..n] });
        }
    }

    /// The last window's first `rows` rows kept: a held window through lane F's `keep` (the KDA states replayed
    /// through the kept rows, the conv windows shifted, the position moved); the serial forward kept its one row.
    pub fn keepRows(h: *Hip, l: *Lane, rows: usize) !void {
        const n = (l.pending orelse return).rows;
        if (rows == 0 or rows > n) return error.PositionMismatch;
        if (l.windowed) {
            try h.e.fd.keep(&l.st, rows);
        } else if (rows < n) return error.PositionMismatch;
        if (h.trace and l.windowed) std.log.info("glm trace {d} keep {d} of {d} at {d}", .{ l.id, rows, n, l.st.pos });
        l.pending = null;
    }

    /// The keeps land as they are marked.
    pub fn flush(_: *Hip) !void {}

    /// Each stream's next drafts: its kept rows (the last window's, or the prompt's last row with the first drawn
    /// token) absorbed by the MTP layer, then `depth` greedy drafts chained from the last pair (lane F's `draft`).
    pub fn draft(h: *Hip, reqs: []const be.DraftRequest) !void {
        const e = h.e;
        for (reqs) |r| {
            const l = try h.streams.get(r.stream);
            try glue.keepFor(h, l, r);
            l.held_n = 0;
            if (!l.mtp) continue; // a stream that does not draft holds none (flashnext/hip/lanes.zig draftFn)
            if (r.position != l.st.pos + 1) {
                std.log.err("draft for {s} at {d}, the stream holds {d} positions", .{ r.stream.id, r.position, l.st.pos });
                return error.PositionMismatch;
            }
            var first: [1]u32 = undefined;
            const follow: []const u32 = if (r.rows != null) r.follow else blk: {
                first[0] = try glue.firstToken(&h.streams.drawn, r);
                break :blk &first;
            };
            if (follow.len == 0 or follow.len > state.window_rows) return error.PositionMismatch;
            const depth = @min(r.depth, e.drafts);
            l.held_n = try e.fm.draft(&l.st, l.hidden.base(), follow, depth, l.held[0..depth]);
            if (h.trace) std.log.info("glm trace {d} draft at {d}: pairs {any} -> {any}", .{ l.id, l.st.pos, follow, l.held[0..l.held_n] });
        }
    }

    pub fn release(h: *Hip, l: *Lane) void {
        h.free(l);
    }
};

test "glm serve: the fp32 draw is the first maximum, ties to the lower id, within a mask" {
    const gpa = std.testing.allocator;
    var row = [_]f32{ 1, 5, 3, 5, std.math.nan(f32), 2 };
    try std.testing.expectEqual(@as(u32, 1), try drawRow(gpa, &row, .{ .sampling = null, .position = 0 }));
    const mask = [_]u32{0b101100}; // ids 2, 3, 5
    try std.testing.expectEqual(@as(u32, 3), try drawRow(gpa, &row, .{ .sampling = null, .position = 0, .mask = &mask }));
    const none = [_]u32{0};
    try std.testing.expectError(error.NoAllowedToken, drawRow(gpa, &row, .{ .sampling = null, .position = 0, .mask = &none }));
    row[4] = 0;
    const s: lanes.Sampling = .{ .seed = 1, .temperature = 1.0, .top_k = 1 };
    try std.testing.expectEqual(@as(u32, 1), try drawRow(gpa, &row, .{ .sampling = s, .position = 7 }));
}

test "glm serve: bf16 of a logit rounds to nearest even" {
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16(1.0));
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16(@bitCast(@as(u32, 0x3f808000))));
    try std.testing.expectEqual(@as(u16, 0x3f82), bf16(@bitCast(@as(u32, 0x3f818000))));
}

test {
    _ = load;
}
