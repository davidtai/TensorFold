//! tf-glm-hip-test forward: GLM-5.3-Flash's eager forward (zig/src/families/glm/hip/forward.zig) on a whole checkpoint.
//!   forward <checkpoint> <ids.u32> [--gen N] [--chunk R] [--tf] [--mtp] [--mtp-draft d]
//!           [--decode-from P [--decode-to E [--resume-mark]] [--window W --keep K]]
//!     ids.u32: uint32 little-endian token ids (G5's <prompt>/ids.u32). The prompt in chunks of R rows (default
//!     2,048), then N greedy tokens (default 16; the first from the last prompt row), ties to the lower id.
//!     --tf: lm_head over every prompt row: teacher-forced NLL, perplexity and top-1 (G5's teacher_forced).
//!     --mtp: the MTP layer teacher-forced over prompt + generated rows (row j: h_j, token j + 1; predicts j + 2).
//!     The prompt/decode seam (LANES.md code contract 7), a test of the CLI only; unset, nothing changes:
//!     --decode-from P: the prompt's ids[0 .. P) in prompt chunks, then its rows P .. E - 1 one at a time on the
//!     decode buffers (the given ids, not greedy picks), each row's head as a prompt row's.
//!     --decode-to E (default: the prompt's end): rows E .. on the prompt path again, from the decode cut.
//!     --resume-mark: the state is kept at P (a prompt-path mark); after the decode rows (their heads and dumps
//!     discarded) it is restored and rows P .. on the prompt path again: a next turn re-prefilling the reply.
//!     --window W --keep K: the decode rows from P in verify windows of W rows (Forward.window, at most 8), each
//!     keeping its first K (Forward.keep; the next window starts there and rewrites the rest): the MTP rounds' window
//!     and rollback on the given ids. The kept rows' heads as prompt rows'.
//!     --mtp-draft d: greedy decode in MTP rounds (Forward.draft, window, keep): d drafts (at most 3) from the MTP
//!     layer, the pending token and the drafts in one window, the rows up to the first rejected draft kept. The ids
//!     and the LOGITS sha (the kept rows') equal the plain greedy run's when a window row has a serial row's bits.
//!     Prints the generated ids and sha256 of every logits row computed (the numerics=exact hash; the same for any R).
//!   forward <checkpoint> - [--chunk R]   with TF_GLM_DUMP and TF_GLM_DUMP_REF set: every prompt of the reference
//!     run, as run_ref.py runs it (teacher-forced prompt, greedy gen, the MTP pass), dumped in G5-ref.md section 4's
//!     format.
const std = @import("std");
const hip = @import("hip");
const glm = @import("glm");

const fwd = glm.forward;
const dump = glm.hip_load.dump;
const State = fwd.State;
const Buffers = fwd.Buffers;
const D: usize = dump.D;
const V: usize = dump.V;

const usage =
    \\usage: tf-glm-hip-test forward <checkpoint> <ids.u32 | -> [--gen N] [--chunk R] [--tf] [--mtp]
    \\       [--mtp-draft d] [--decode-from P [--decode-to E [--resume-mark]] [--window W --keep K]]
    \\       (ids "-": the prompts of TF_GLM_DUMP_REF, dumped into TF_GLM_DUMP)
    \\
;

const Opts = struct { gen: usize = 16, chunk: usize = glm.state.prompt_rows, tf: bool = false, mtp: bool = false, seam: Seam = .{}, draft: ?usize = null };

/// The prompt/decode seam test's cuts (all null: the plain run); `window` > 0: the decode rows in verify windows.
const Seam = struct { from: ?usize = null, to: ?usize = null, mark: bool = false, window: usize = 0, keep: usize = 0 };

fn opts(args: []const [:0]const u8) !Opts {
    var o: Opts = .{};
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "--tf")) o.tf = true else if (std.mem.eql(u8, a, "--mtp")) o.mtp = true else if (std.mem.eql(u8, a, "--resume-mark")) o.seam.mark = true else {
            if (i + 1 >= args.len) return error.MissingArgument;
            const v = try std.fmt.parseInt(usize, args[i + 1], 10);
            if (std.mem.eql(u8, a, "--gen")) o.gen = v else if (std.mem.eql(u8, a, "--chunk")) o.chunk = v else if (std.mem.eql(u8, a, "--decode-from")) o.seam.from = v else if (std.mem.eql(u8, a, "--decode-to")) o.seam.to = v else if (std.mem.eql(u8, a, "--window")) o.seam.window = v else if (std.mem.eql(u8, a, "--keep")) o.seam.keep = v else if (std.mem.eql(u8, a, "--mtp-draft")) o.draft = v else return error.UnknownOption;
            i += 1;
        }
    }
    if (o.chunk == 0 or o.chunk > glm.state.prompt_rows) return error.BadChunk;
    if ((o.seam.to != null or o.seam.mark) and o.seam.from == null) return error.DecodeToWithoutDecodeFrom;
    if (o.seam.mark and o.seam.to == null) return error.ResumeMarkWithoutDecodeTo;
    if (o.seam.window > 0 or o.seam.keep > 0) {
        if (o.seam.from == null or o.seam.mark) return error.WindowNeedsDecodeFromWithoutMark;
        if (o.seam.window > glm.state.window_rows or o.seam.keep == 0 or o.seam.keep > o.seam.window) return error.BadWindowKeep;
    }
    if (o.draft) |d| if (d > fwd.Forward.max_drafts or o.mtp or o.seam.from != null) return error.BadMtpDraft;
    return o;
}

/// The device side of a run: one stream's state, the prompt and decode windows, every row's hidden.
const Run = struct {
    r: *const hip.Runtime,
    io: std.Io,
    s: hip.Stream,
    st: State,
    pb: Buffers,
    db: Buffers,
    fp: fwd.Forward,
    fd: fwd.Forward,
    /// bf16 [rows][D]: every row's final-normed hidden (the MTP layer's input).
    hid: hip.DeviceBuffer,
    logits: []f32, // host [head_rows][V]
    hash: std.crypto.hash.sha2.Sha256 = .init(.{}),
    /// --resume-mark: the state's leading part (positions, KDA conv windows and states, the KDA window buffers) at
    /// the mark; the DSA caches past the mark are rewritten before they are read.
    mark: ?hip.DeviceBuffer = null,
    /// --window, --mtp-draft: the verify windows' Forward (decode buffers of state.window_rows rows) and the MTP
    /// drafts' (their own buffers: a draft reads the window's final-normed rows).
    fv: ?fwd.Forward = null,
    fm: ?fwd.Forward = null,
    /// Forward.keep's time by kept rows (host-timed between stream syncs), microseconds.
    keep_us: [glm.state.window_rows + 1]f64 = @splat(0),
    keep_n: [glm.state.window_rows + 1]usize = @splat(0),
    /// TF_GLM_PROFILE: each window's GPU time by op group; null: off.
    prof: ?*Profiler = null,

    fn now(io: std.Io) f64 {
        return @as(f64, @floatFromInt(std.Io.Timestamp.now(io, .awake).nanoseconds)) / 1e9;
    }

    /// lm_head over `n` bf16 rows at `x` (window `f`), `each(ctx, row, logits)` for each; the bytes into the hash.
    fn heads(run: *Run, f: *fwd.Forward, x: u64, n: usize, ctx: anytype, comptime each: fn (@TypeOf(ctx), usize, []const f32) anyerror!void) !void {
        var r0: usize = 0;
        while (r0 < n) : (r0 += glm.state.head_rows) {
            const k: usize = @min(glm.state.head_rows, n - r0);
            const lg = try f.head(x + r0 * D * 2, k);
            if (run.prof != null) try f.sync(); // the download alone in its time
            const d0 = Run.now(run.io);
            try f.download(f32, lg, run.logits[0 .. k * V]);
            if (run.prof) |pr| pr.dl += Run.now(run.io) - d0;
            for (0..k) |j| {
                const row = run.logits[j * V ..][0..V];
                run.hash.update(std.mem.sliceAsBytes(row));
                try each(ctx, r0 + j, row);
            }
        }
    }

    /// Window f's `n` final-normed rows into hid from sequence row `row`.
    fn keep(run: *Run, f: *fwd.Forward, row: usize, n: usize) !void {
        try hip.runtime.check(run.r.api.hipMemcpyDtoDAsync(@ptrFromInt(run.hid.base() + row * D * 2), @ptrFromInt(f.b.fnormed), n * D * 2, run.s.handle));
    }

    /// The bytes of the state the mark keeps: everything before the first DSA cache set.
    fn markBytes(st: *const State) usize {
        return st.caches[0].lc - st.buf.base();
    }

    /// Keep the state at the current position; the host fields come back as the returned copy.
    fn saveMark(run: *Run) !State {
        const m = run.mark orelse return error.NoMarkBuffer;
        try hip.runtime.check(run.r.api.hipMemcpyDtoDAsync(@ptrFromInt(m.base()), @ptrFromInt(run.st.buf.base()), markBytes(&run.st), run.s.handle));
        return run.st;
    }

    /// The state kept by saveMark, if no ring rebase came between (the rings are not kept).
    fn restoreMark(run: *Run, at: State) !void {
        const m = run.mark orelse return error.NoMarkBuffer;
        if (at.base != run.st.base or at.mtp_base != run.st.mtp_base) return error.RebasedSinceMark;
        try hip.runtime.check(run.r.api.hipMemcpyDtoDAsync(@ptrFromInt(run.st.buf.base()), @ptrFromInt(m.base()), markBytes(&run.st), run.s.handle));
        run.st = at;
        try run.st.setPos(run.s.handle, at.pos);
        try run.st.setMtpLen(run.s.handle, at.mtp_len);
    }
};

/// One prompt's teacher-forced and greedy rows (as run_ref.py: the prompt, gen tokens, the MTP pass).
const Seq = struct {
    ids: []const u32,
    gen: std.ArrayList(u32) = .empty,
    max_gen: usize,
    eos: []const u32,
    tf: bool,
    dump: ?*dump.Seq,
    nll: f64 = 0,
    hits: usize = 0,
    mtp_hits: usize = 0,
    mtp_rows: usize = 0,
    last: u32 = 0,

    fn tokenAt(q: *const Seq, i: usize) u32 {
        return if (i < q.ids.len) q.ids[i] else q.gen.items[i - q.ids.len];
    }

    fn promptRow(q: *Seq, row: usize, lg: []const f32) !void {
        const t = q.ids.len;
        if (q.dump) |s| try s.promptLogits(row, lg);
        const top = dump.top8(lg);
        if (row + 1 < t) {
            const next = q.ids[row + 1];
            q.nll += top.lse - lg[next];
            q.hits += @intFromBool(top.ids[0] == @as(i32, @intCast(next)));
        } else q.last = @intCast(top.ids[0]);
    }

    fn mtpRow(q: *Seq, row: usize, lg: []const f32) !void {
        if (q.dump) |s| try s.mtpLogits(row, lg);
        if (row + 2 < q.ids.len + q.gen.items.len) {
            q.mtp_rows += 1;
            q.mtp_hits += @intFromBool(dump.top8(lg).ids[0] == @as(i32, @intCast(q.tokenAt(row + 2))));
        }
    }

    fn isEos(q: *const Seq, id: u32) bool {
        for (q.eos) |e| if (e == id) return true;
        return false;
    }
};

const PromptRows = struct {
    q: *Seq,
    at: usize,
    fn each(c: PromptRows, row: usize, lg: []const f32) anyerror!void {
        try c.q.promptRow(c.at + row, lg);
    }
};

const MtpRows = struct {
    q: *Seq,
    at: usize,
    fn each(c: MtpRows, row: usize, lg: []const f32) anyerror!void {
        try c.q.mtpRow(c.at + row, lg);
    }
};

const Greedy = struct {
    q: *Seq,
    a: std.mem.Allocator,
    fn each(c: Greedy, _: usize, lg: []const f32) anyerror!void {
        const id: u32 = if (c.q.dump) |s| try s.step(lg) else @intCast(dump.top8(lg).ids[0]);
        try c.q.gen.append(c.a, id);
    }
};

/// The prompt's rows [from, to) in chunks of `chunk` on the prompt path (lm_head: every row, or the prompt's last).
fn promptRows(run: *Run, q: *Seq, from: usize, to: usize, chunk: usize) !void {
    const t = q.ids.len;
    var at = from;
    while (at < to) : (at += chunk) {
        const n = @min(chunk, to - at);
        run.fp.seq = q.dump;
        run.fp.row0 = at;
        if (run.prof) |pr| try pr.begin();
        try run.fp.run(&run.st, q.ids[at..][0..n]);
        try run.keep(&run.fp, at, n);
        if (q.tf or q.dump != null) {
            try run.heads(&run.fp, run.fp.b.fnormed, n, PromptRows{ .q = q, .at = at }, PromptRows.each);
        } else if (at + n == t) {
            try run.heads(&run.fp, run.fp.b.fnormed + (n - 1) * D * 2, 1, PromptRows{ .q = q, .at = t - 1 }, PromptRows.each);
        }
        if (run.prof) |pr| try pr.end(.chunk, at, n);
    }
}

/// Forward.keep of `rows`, timed.
fn keepTimed(run: *Run, io: std.Io, rows: usize) !void {
    const f = &run.fv.?;
    try f.sync();
    const k0 = Run.now(io);
    try f.keep(&run.st, rows);
    try f.sync();
    run.keep_us[rows] += (Run.now(io) - k0) * 1e6;
    run.keep_n[rows] += 1;
}

/// The prompt's rows [from, to) in verify windows of `w` rows on the given ids, each keeping its first `k` (the
/// last window what is left): the kept rows' heads as prompt rows'.
fn windowRows(run: *Run, io: std.Io, q: *Seq, from: usize, to: usize, w: usize, k: usize) !void {
    const t = q.ids.len;
    const f = &run.fv.?;
    var at = from;
    while (at < to) {
        const n = @min(w, to - at);
        const kept = @min(k, n);
        f.seq = q.dump;
        f.row0 = at;
        try f.window(&run.st, q.ids[at..][0..n]);
        try run.keep(f, at, kept);
        for (0..kept) |r| if (q.tf or q.dump != null or at + r + 1 == t)
            try run.heads(f, f.b.fnormed + r * D * 2, 1, PromptRows{ .q = q, .at = at + r }, PromptRows.each);
        try keepTimed(run, io, kept);
        at += kept;
    }
}

/// The prompt's rows [from, to) one at a time on the decode path (the given ids); `discard`: no heads, no dumps.
fn decodeRows(run: *Run, q: *Seq, from: usize, to: usize, discard: bool) !void {
    const t = q.ids.len;
    for (from..to) |row| {
        run.fd.seq = if (discard) null else q.dump;
        run.fd.row0 = row;
        try run.fd.run(&run.st, q.ids[row..][0..1]);
        try run.keep(&run.fd, row, 1);
        if (!discard and (q.tf or q.dump != null or row + 1 == t))
            try run.heads(&run.fd, run.fd.b.fnormed, 1, PromptRows{ .q = q, .at = row }, PromptRows.each);
    }
}

/// Greedy decode in MTP rounds after the prompt (q.gen holds the first token): the MTP layer absorbs the prompt's
/// rows j < t - 1 on the prompt path, then each round drafts, verifies the pending token and the drafts in one
/// window, keeps the rows up to the first rejected draft, and hashes the kept rows' logits (the plain run's rows).
fn mtpRounds(a: std.mem.Allocator, io: std.Io, run: *Run, q: *Seq, d: usize) !void {
    const t = q.ids.len;
    const fv = &run.fv.?;
    const fm = &run.fm.?;
    // the prompt's pairs (h_j, token j + 1) for j < t - 1, on the prompt buffers (the server's prefill)
    var j0: usize = 0;
    while (j0 + 1 < t) : (j0 += run.fp.b.rows) {
        const n = @min(run.fp.b.rows, t - 1 - j0);
        run.fp.seq = null;
        try run.fp.mtp(&run.st, q.ids[j0 + 1 ..][0..n], run.hid.base() + j0 * D * 2);
    }
    var drafts: [fwd.Forward.max_drafts]u32 = undefined;
    var ids: [fwd.Forward.max_drafts + 1]u32 = undefined;
    var out: [fwd.Forward.max_drafts + 1]u32 = undefined;
    var hin: u64 = run.hid.base() + (t - 1) * D * 2; // the first draft: the last prompt row and the first token
    var follow: []const u32 = q.gen.items[0..1];
    var rounds: usize = 0;
    var drafted: usize = 0;
    var accepted: usize = 0;
    const r0 = Run.now(io);
    while (q.gen.items.len < q.max_gen and !q.isEos(q.gen.items[q.gen.items.len - 1])) {
        const depth = @min(d, q.max_gen - q.gen.items.len - 1);
        const p0 = t + q.gen.items.len - 1;
        if (run.prof) |pr| try pr.begin();
        const n = try fm.draft(&run.st, hin, follow, depth, &drafts);
        ids[0] = q.gen.items[q.gen.items.len - 1];
        @memcpy(ids[1..][0..n], drafts[0..n]);
        const R = n + 1;
        const row = t + q.gen.items.len - 1;
        fv.seq = null;
        fv.row0 = row;
        try fv.window(&run.st, ids[0..R]);
        try run.keep(fv, row, R);
        const lg = try fv.head(fv.b.fnormed, R);
        try fv.download(f32, lg, run.logits[0 .. R * V]);
        for (0..R) |r| out[r] = @intCast(dump.top8(run.logits[r * V ..][0..V]).ids[0]);
        var kept: usize = 1;
        while (kept < R and out[kept - 1] == drafts[kept - 1]) kept += 1;
        try keepTimed(run, io, kept);
        if (run.prof) |pr| try pr.end(.round, p0, kept);
        for (0..kept) |r| {
            if (q.gen.items.len >= q.max_gen or q.isEos(q.gen.items[q.gen.items.len - 1])) break;
            run.hash.update(std.mem.sliceAsBytes(run.logits[r * V ..][0..V]));
            try q.gen.append(a, out[r]);
        }
        rounds += 1;
        drafted += n;
        accepted += kept - 1;
        hin = fv.b.fnormed;
        follow = out[0..kept];
    }
    const secs = Run.now(io) - r0;
    std.debug.print("MTP-DRAFT d {d}: {d} rounds in {d:.2} s, {d} of {d} drafts accepted ({d:.3}), {d:.2} tokens a round, {d:.2} tok/s\n", .{ d, rounds, secs, accepted, drafted, @as(f64, @floatFromInt(accepted)) / @as(f64, @floatFromInt(@max(drafted, 1))), @as(f64, @floatFromInt(q.gen.items.len - 1)) / @as(f64, @floatFromInt(@max(rounds, 1))), @as(f64, @floatFromInt(q.gen.items.len - 1)) / secs });
}

fn runSeq(a: std.mem.Allocator, io: std.Io, run: *Run, q: *Seq, chunk: usize, mtp: bool, seam: Seam, draft: ?usize) !void {
    const t = q.ids.len;
    try run.st.reset(run.s.handle);
    run.hash = .init(.{});
    // 1. the prompt in chunks; with the seam cuts: [0, P) prompt, [P, E) decode, then [E, t) prompt (or [P, t) from
    //    the mark)
    const t0 = Run.now(io);
    const p = if (seam.from) |x| @min(x, t) else t;
    const e = if (seam.to) |x| @min(@max(x, p), t) else t;
    const mark = seam.mark and e > p;
    try promptRows(run, q, 0, p, chunk);
    const at_mark: ?State = if (mark) try run.saveMark() else null;
    if (seam.window > 0) try windowRows(run, io, q, p, e, seam.window, seam.keep) else try decodeRows(run, q, p, e, mark);
    if (at_mark) |m| try run.restoreMark(m);
    try promptRows(run, q, if (mark) p else e, t, chunk);
    try run.fp.sync();
    try run.fd.sync();
    if (p < t) std.debug.print("SEAM rows {d}-{d} by {s}{s}; prompt rows {d}-{d} after it{s}\n", .{ p, e -| 1, if (seam.window > 0) "verify windows" else "serial decode", if (mark) " (discarded)" else "", if (mark) p else e, t -| 1, if (mark) ", from the state kept at the mark" else ", from the decode cut" });
    if (seam.window > 0) std.debug.print("SEAM windows of {d} rows keeping {d}\n", .{ seam.window, seam.keep });
    const t1 = Run.now(io);
    // 2. greedy: the first token from the last prompt row, then one row a step (or MTP rounds)
    try q.gen.append(a, q.last);
    if (draft) |d| try mtpRounds(a, io, run, q, d);
    while (draft == null and q.gen.items.len < q.max_gen and !q.isEos(q.gen.items[q.gen.items.len - 1])) {
        const row = t + q.gen.items.len - 1;
        run.fd.seq = q.dump;
        run.fd.row0 = row;
        if (run.prof) |pr| try pr.begin();
        try run.fd.run(&run.st, q.gen.items[q.gen.items.len - 1 ..]);
        try run.keep(&run.fd, row, 1);
        try run.heads(&run.fd, run.fd.b.fnormed, 1, Greedy{ .q = q, .a = a }, Greedy.each);
        if (run.prof) |pr| try pr.end(.step, row, 1);
    }
    try run.fd.sync();
    const t2 = Run.now(io);
    const rows = t + q.gen.items.len - 1; // rows the main model ran
    if (q.dump) |s| for (0..rows) |row| {
        var h: [D]f32 = undefined;
        try run.fp.rowF32(run.hid.base(), true, row, &h);
        try s.hiddenRow(row, &h);
    };
    // 3. the MTP layer, teacher-forced over the rows (row j: h_j and token j + 1)
    if (mtp) {
        const next = try a.alloc(u32, rows);
        for (next, 0..) |*x, j| x.* = q.tokenAt(j + 1);
        var j0: usize = 0;
        while (j0 < rows) : (j0 += chunk) {
            const n = @min(chunk, rows - j0);
            run.fp.seq = q.dump;
            run.fp.row0 = j0;
            if (run.prof) |pr| try pr.begin();
            try run.fp.mtp(&run.st, next[j0..][0..n], run.hid.base() + j0 * D * 2);
            try run.heads(&run.fp, run.fp.b.fnormed, n, MtpRows{ .q = q, .at = j0 }, MtpRows.each);
            if (run.prof) |pr| try pr.end(.mtp, j0, n);
        }
    }
    var sum: [32]u8 = undefined;
    run.hash.final(&sum);
    std.debug.print("PROMPT {d} rows in {d:.2} s ({d:.1} tok/s, chunks of {d}); DECODE {d} tokens in {d:.2} s\n", .{ t, t1 - t0, @as(f64, @floatFromInt(t)) / (t1 - t0), chunk, q.gen.items.len - 1, t2 - t1 });
    if (q.tf or q.dump != null) {
        const n: f64 = @floatFromInt(@max(t - 1, 1));
        std.debug.print("TEACHER-FORCED {d} rows: nll {d:.6}, ppl {d:.4}, top-1 {d:.4}\n", .{ t - 1, q.nll / n, @exp(q.nll / n), @as(f64, @floatFromInt(q.hits)) / n });
    }
    if (mtp) std.debug.print("MTP {d} rows: top-1 of token j + 2 {d:.4}\n", .{ q.mtp_rows, @as(f64, @floatFromInt(q.mtp_hits)) / @as(f64, @floatFromInt(@max(q.mtp_rows, 1))) });
    std.debug.print("IDS", .{});
    for (q.gen.items) |id| std.debug.print(" {d}", .{id});
    std.debug.print("\nLOGITS sha256 {s} (every logits row computed, fp32 bytes in order)\n", .{&std.fmt.bytesToHex(sum, .lower)});
}

/// `args`: what follows "forward" (checkpoint, ids file or "-", options).
pub fn command(init: std.process.Init, args: []const [:0]const u8) !void {
    if (args.len < 2) {
        std.debug.print(usage, .{});
        return error.Usage;
    }
    const o = opts(args[2..]) catch |err| {
        std.debug.print(usage, .{});
        return err;
    };
    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const io = init.io;
    const dir = args[0];
    var r = try hip.Runtime.open();
    defer r.close();
    var ctx = try hip.Context.init(&r, 0);
    defer ctx.deinit();
    var problem: []const u8 = "";
    var opened = glm.hip_load.open.open(init.gpa, a, io, &r, dir, .{}, &problem) catch |err| {
        std.debug.print("{s}\n", .{problem});
        return err;
    };
    defer opened.deinit();
    const cfg = opened.ck.config;
    const eos = cfg.eos[0..cfg.eos_n];
    var dumps = try dump.Dump.fromEnv(a, eos);
    // the prompts: the reference's under TF_GLM_DUMP, else the ids file
    var longest: usize = 0;
    var ids: []u32 = &.{};
    if (dumps) |*d| {
        for (d.prompts) |p| longest = @max(longest, p.ids.len + p.gen);
    } else {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, args[1], a, .limited(1 << 28));
        if (bytes.len == 0 or bytes.len % 4 != 0) return error.BadIdsFile;
        ids = try a.alloc(u32, bytes.len / 4);
        for (ids, 0..) |*id, i| id.* = std.mem.readInt(u32, bytes[4 * i ..][0..4], .little);
        longest = ids.len + @max(o.gen, 1);
    }
    const capacity = std.mem.alignForward(usize, longest + 8, 64); // cuda_state.zig's align64(context + 8)
    var l = try hip.Launcher.load(&r, .{}, hip.kernels.images); // numerics=exact
    defer l.unload();
    var rn: Run = .{
        .r = &r,
        .io = io,
        .s = try hip.Stream.init(&r),
        .st = undefined,
        .pb = undefined,
        .db = undefined,
        .fp = undefined,
        .fd = undefined,
        .hid = undefined,
        .logits = undefined,
    };
    defer rn.s.deinit();
    rn.logits = try init.gpa.alloc(f32, glm.state.head_rows * V);
    defer init.gpa.free(rn.logits);
    rn.st = try State.init(&r, capacity);
    defer rn.st.deinit();
    rn.pb = try Buffers.init(&r, o.chunk, true, capacity);
    defer rn.pb.deinit();
    rn.db = try Buffers.init(&r, 1, false, capacity);
    defer rn.db.deinit();
    rn.hid = try hip.DeviceBuffer.alloc(&r, longest * D * 2);
    defer rn.hid.free();
    if (o.seam.mark) rn.mark = try hip.DeviceBuffer.alloc(&r, Run.markBytes(&rn.st));
    defer if (rn.mark) |*m| m.free();
    std.debug.print("FORWARD context {d}: state {d:.3} GiB, prompt buffers {d:.3} GiB\n", .{ capacity, gib(State.bytes(capacity)), gib(rn.pb.arena.len + rn.pb.buf.len) });
    rn.fp = try fwd.Forward.init(&l, opened.model(), cfg, rn.s.handle, &rn.pb);
    rn.fd = try fwd.Forward.init(&l, opened.model(), cfg, rn.s.handle, &rn.db);
    // --window, --mtp-draft: the verify windows' and the MTP drafts' decode buffers
    var wb: ?Buffers = null;
    var mb: ?Buffers = null;
    defer if (wb) |*b| b.deinit();
    defer if (mb) |*b| b.deinit();
    if (o.seam.window > 0 or o.draft != null) {
        wb = try Buffers.init(&r, glm.state.window_rows, false, capacity);
        rn.fv = try fwd.Forward.init(&l, opened.model(), cfg, rn.s.handle, &wb.?);
    }
    if (o.draft != null) {
        if (dumps != null) return error.MtpDraftWithDump;
        mb = try Buffers.init(&r, glm.state.window_rows, false, capacity);
        rn.fm = try fwd.Forward.init(&l, opened.model(), cfg, rn.s.handle, &mb.?);
    }
    // TF_GLM_PROFILE=1: GPU time by layer and op group, each window (no result changes); TSV in TF_GLM_PROFILE_OUT
    var profiler: Profiler = undefined;
    const prof_on = if (std.c.getenv("TF_GLM_PROFILE")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
    if (prof_on) {
        profiler = .{ .p = try glm.profile.Prof.init(init.gpa, &r, rn.s.handle, 4096, 8 << 20), .a = a, .io = io };
        rn.prof = &profiler;
        rn.fp.prof = &profiler.p;
        rn.fd.prof = &profiler.p;
        if (rn.fv) |*f| f.prof = &profiler.p;
        if (rn.fm) |*f| f.prof = &profiler.p;
        std.debug.print("PROFILE on: GPU events between op groups\n", .{});
    }
    defer if (rn.prof) |pr| {
        pr.report(if (std.c.getenv("TF_GLM_PROFILE_OUT")) |path| std.mem.span(path) else "profile.tsv") catch |err| std.debug.print("PROFILE report failed: {t}\n", .{err});
        pr.p.deinit();
    };
    if (dumps) |*d| {
        // TF_GLM_FORCE_PICKS=<run dir> (a diagnostic): the teacher-forced prompts take <dir>/<name>/router_picks.npy and
        // router_weights.npy at every MoE layer (G5's or a HIP dump's); greedy prompts run free
        const force_dir: ?[]const u8 = if (std.c.getenv("TF_GLM_FORCE_PICKS")) |v| std.mem.span(v) else null;
        for (d.prompts) |p| {
            const sq = try d.begin(p);
            var q: Seq = .{ .ids = p.ids, .max_gen = p.gen, .eos = eos, .tf = true, .dump = sq };
            var fc: glm.forward.Forward.Force = undefined;
            const forced = force_dir != null and !p.greedy;
            if (forced) {
                fc = try forceOf(a, io, force_dir.?, p.name, p.ids.len + p.gen - 1);
                rn.fp.force = &fc;
                rn.fd.force = &fc;
            }
            std.debug.print("== {s}: {d} ids, gen {d}{s}\n", .{ p.name, p.ids.len, p.gen, if (forced) ", routing FORCED" else "" });
            if (forced) std.debug.print("FORCED picks and weights from {s}/{s}\n", .{ force_dir.?, p.name });
            try runSeq(a, io, &rn, &q, o.chunk, true, o.seam, null);
            rn.fp.force = null;
            rn.fd.force = null;
            try sq.close();
        }
        try d.finish("tf-glm-hip-test forward: eager HIP forward, numerics=exact");
        std.debug.print("DUMP OK\n", .{});
        return;
    }
    var q: Seq = .{ .ids = ids, .max_gen = @max(o.gen, 1), .eos = eos, .tf = o.tf, .dump = null };
    try runSeq(a, io, &rn, &q, o.chunk, o.mtp, o.seam, o.draft);
    for (rn.keep_n, rn.keep_us, 0..) |n, us, k| if (n > 0) std.debug.print("KEEP {d} rows: {d} keeps, {d:.1} us each (host-timed between syncs)\n", .{ k, n, us / @as(f64, @floatFromInt(n)) });
    std.debug.print("FORWARD OK\n", .{});
}

/// The forced routing of prompt `name` from run dir `dir`: router_picks.npy int16 and router_weights.npy fp32, each
/// [46][N][8] with N at least `rows`.
fn forceOf(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8, rows: usize) !glm.forward.Forward.Force {
    const npy = @import("core").npy;
    const read = struct {
        fn f(al: std.mem.Allocator, i: std.Io, d: []const u8, n: []const u8, file: []const u8) !npy.Array {
            const bytes = try std.Io.Dir.cwd().readFileAlloc(i, try std.fs.path.join(al, &.{ d, n, file }), al, .limited(1 << 30));
            return npy.parse(bytes);
        }
    }.f;
    const pk = try read(a, io, dir, name, "router_picks.npy");
    const wt = try read(a, io, dir, name, "router_weights.npy");
    if (!std.mem.eql(u8, pk.descr, "<i2") or !std.mem.eql(u8, wt.descr, "<f4") or pk.rank != 3 or wt.rank != 3) return error.BadForce;
    if (pk.shape[0] != 46 or pk.shape[2] != 8 or wt.shape[0] != 46 or wt.shape[1] != pk.shape[1] or wt.shape[2] != 8) return error.BadForce;
    if (pk.shape[1] < rows) return error.ForceRows;
    const picks = try a.alloc(i16, pk.count());
    @memcpy(std.mem.sliceAsBytes(picks), pk.data);
    const wts = try a.alloc(f32, wt.count());
    @memcpy(std.mem.sliceAsBytes(wts), wt.data);
    return .{ .picks = picks, .wts = wts, .rows = pk.shape[1] };
}

/// TF_GLM_PROFILE's windows: each prompt chunk, decode step and MTP chunk, its op groups' GPU time (profile.zig), its
/// host wall time and the host's logits download.
const Profiler = struct {
    p: glm.profile.Prof,
    a: std.mem.Allocator,
    io: std.Io,
    t0: f64 = 0,
    dl: f64 = 0,
    rows: std.ArrayList(Line) = .empty,
    wins: std.ArrayList(Win) = .empty,

    /// A prompt chunk, a serial decode step, an MTP-layer chunk, an MTP round (--mtp-draft: draft, verify window,
    /// head, keep; its rows the kept rows).
    const Kind = enum { chunk, step, mtp, round };
    const Line = struct { win: usize, row: glm.profile.Row };
    const Win = struct { kind: Kind, pos: usize, rows: usize, gpu_us: f64, wall_us: f64, dl_us: f64, class_us: [glm.profile.classes.len]f64 };

    fn begin(pr: *Profiler) !void {
        pr.dl = 0;
        pr.t0 = Run.now(pr.io);
        try pr.p.start();
    }

    fn end(pr: *Profiler, kind: Kind, pos: usize, rows: usize) !void {
        const got = try pr.p.collect(pr.a);
        var w: Win = .{ .kind = kind, .pos = pos, .rows = rows, .gpu_us = 0, .wall_us = (Run.now(pr.io) - pr.t0) * 1e6, .dl_us = pr.dl * 1e6, .class_us = @splat(0) };
        for (got) |row| {
            w.gpu_us += row.us;
            for (glm.profile.classes, 0..) |c, i| if (std.mem.eql(u8, c, row.op.class())) {
                w.class_us[i] += row.us;
            };
            try pr.rows.append(pr.a, .{ .win = pr.wins.items.len, .row = row });
        }
        try pr.wins.append(pr.a, w);
    }

    fn summary(pr: *const Profiler, label: []const u8, ws: []const Win) void {
        if (ws.len == 0) return;
        var cls: [glm.profile.classes.len]f64 = @splat(0);
        var gpu: f64 = 0;
        var wall: f64 = 0;
        var dl: f64 = 0;
        for (ws) |w| {
            for (&cls, w.class_us) |*o, v| o.* += v;
            gpu += w.gpu_us;
            wall += w.wall_us;
            dl += w.dl_us;
        }
        const nf: f64 = @floatFromInt(ws.len);
        std.debug.print("PROFILE {s} ({d} window{s}, {d} rows): GPU {d:.1} ms, host wall {d:.1} ms, host logits download {d:.2} ms (means)\n", .{ label, ws.len, if (ws.len == 1) "" else "s", ws[0].rows, gpu / nf / 1e3, wall / nf / 1e3, dl / nf / 1e3 });
        for (glm.profile.classes, cls) |c, v| if (v > 0) std.debug.print("  {s:<16} {d:>10.1} us {d:>6.2} %\n", .{ c, v / nf, 100 * v / gpu });
        _ = pr;
    }

    fn report(pr: *const Profiler, path: []const u8) !void {
        var t: std.Io.Writer.Allocating = .init(pr.a);
        try t.writer.writeAll("window\tpos\twin_rows\tlayer\top\tclass\trows\tus\tbytes\n");
        for (pr.rows.items) |ln| {
            const w = pr.wins.items[ln.win];
            try t.writer.print("{t}\t{d}\t{d}\t{d}\t{t}\t{s}\t{d}\t{d:.2}\t{d}\n", .{ w.kind, w.pos, w.rows, ln.row.layer, ln.row.op, ln.row.op.class(), ln.row.rows, ln.row.us, ln.row.bytes });
        }
        for (pr.wins.items) |w| try t.writer.print("{t}\t{d}\t{d}\t-\twindow\tgpu {d:.1} us; host wall {d:.1} us; host logits download {d:.1} us\t{d}\t{d:.2}\t0\n", .{ w.kind, w.pos, w.rows, w.gpu_us, w.wall_us, w.dl_us, w.rows, w.gpu_us });
        try std.Io.Dir.cwd().writeFile(pr.io, .{ .sub_path = path, .data = t.written() });
        std.debug.print("PROFILE TSV {s}: {d} groups in {d} windows\n", .{ path, pr.rows.items.len, pr.wins.items.len });
        // the cases: the first and the last prompt chunk, the decode steps, the MTP chunks
        var first: ?usize = null;
        var last: ?usize = null;
        var steps: std.ArrayList(Win) = .empty;
        var mtps: std.ArrayList(Win) = .empty;
        var rounds: std.ArrayList(Win) = .empty;
        for (pr.wins.items, 0..) |w, i| switch (w.kind) {
            .chunk => {
                if (first == null) first = i;
                last = i;
            },
            .step => try steps.append(pr.a, w),
            .mtp => try mtps.append(pr.a, w),
            .round => try rounds.append(pr.a, w),
        };
        var buf: [64]u8 = undefined;
        if (first) |i| pr.summary(try std.fmt.bufPrint(&buf, "prompt chunk at {d}", .{pr.wins.items[i].pos}), pr.wins.items[i .. i + 1]);
        if (last) |i| if (i != first.?) pr.summary(try std.fmt.bufPrint(&buf, "prompt chunk at {d}", .{pr.wins.items[i].pos}), pr.wins.items[i .. i + 1]);
        pr.summary("decode step", steps.items);
        pr.summary("MTP chunk", mtps.items);
        pr.summary("MTP round", rounds.items);
        if (rounds.items.len > 0) {
            var kept: usize = 0;
            for (rounds.items) |w| kept += w.rows;
            std.debug.print("PROFILE MTP rounds: {d:.2} kept rows a round\n", .{@as(f64, @floatFromInt(kept)) / @as(f64, @floatFromInt(rounds.items.len))});
        }
    }
};

fn gib(n: usize) f64 {
    return @as(f64, @floatFromInt(n)) / (1 << 30);
}
