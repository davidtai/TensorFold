//! GLM-5.3-Flash's per-stream state and the forward's scratch on one HIP device, in the CUDA path's layout
//! (cuda_state.zig): the KDA conv windows and recurrent states, the 12 DSA cache sets (the 11 DSA layers and the MTP
//! layer: latent rows, index-key and gate rings, pooled keys), and a window's buffers. No TP: one rank holds every head.
//! The KDA keep runs in place (G2), so a layer has one state copy, not CUDA's two.

const std = @import("std");
const hip = @import("hip");
const config = @import("../config.zig");

/// A prompt chunk's rows (cuda_state.zig prefill_rows).
pub const prompt_rows = 2048;
/// A decode window's rows (the verify window of the MTP rounds; the eager forward runs 1).
pub const window_rows = 8;
/// Index-key and gate rows a ring holds: pool_keys reads only from the current pool's first row on.
pub const index_ring = 2 * prompt_rows;
/// The DSA layers (3, 7, ..., 43) and the KDA layers of the 45.
pub const dsa_layers = 11;
pub const kda_layers = 34;
/// Pools of index keys (config kpool) and the incomplete pool's tail of the selected token lists.
pub const pool = 4;

const c0: config.Config = .{};
const D: usize = c0.hidden;
const kda_n: usize = @as(usize, c0.kda_heads) * c0.kda_dim; // 8,192
/// A KDA projection row: [q | k | v | f_a | g_a | b] (config.kdaProj()).
pub const kda_width: usize = 3 * kda_n + 2 * @as(usize, c0.kda_dim) + c0.kda_heads;
const lw: usize = c0.kv_lora;
const i_dim: usize = c0.i_dim;

fn a256(n: usize) usize {
    return std.mem.alignForward(usize, n, 256);
}

/// One device allocation handed out front to back, 256-byte aligned.
const Carve = struct {
    base: u64 = 0,
    used: usize = 0,

    fn take(c: *Carve, bytes: usize) u64 {
        const at = a256(c.used);
        c.used = at + bytes;
        return c.base + at;
    }
};

/// One DSA cache set (cuda_dsa_block.zig Caches): latent bf16 [capacity][512], index keys and gates bf16 rows at
/// position p at ik + p rows (a ring's pointer, rebased), pooled keys bf16 [pk_rows][128].
pub const Caches = struct { lc: u64, ik: u64, ig: u64, pk: u64, pk_rows: usize };

/// A sequence's committed caches (cuda_state.zig State, one device).
pub const State = struct {
    buf: hip.DeviceBuffer,
    capacity: usize,
    pos: usize = 0,
    mtp_len: usize = 0,
    pos_dev: u64, // i32
    mtp_pos_dev: u64, // i32
    conv: u64, // bf16 [kda layers][3][3 x 8,192]
    rec: u64, // fp32 [kda layers][64][128 v][128 k], one copy
    // a verify window's rows until its keep (the MTP rounds' step; unused by the eager forward)
    proj: u64, // bf16 [kda layers][window rows][kda width]
    save_k: u64, // fp32 [kda layers][window rows][64][128]
    save_v: u64, // bf16, as save_k
    save_g: u64, // fp32, as save_k
    save_b: u64, // fp32 [kda layers][window rows][64]
    caches: [dsa_layers]Caches,
    mtp: Caches,
    ring_rows: usize,
    rings: [dsa_layers + 1][2]u64, // each set's ik ring, ig ring; the MTP layer's last
    base: usize = 0, // the first position the main rings hold
    mtp_base: usize = 0,

    pub const conv_layer: usize = 3 * 3 * kda_n; // elements
    pub const rec_layer: usize = @as(usize, c0.kda_heads) * c0.kda_dim * c0.kda_dim; // floats

    fn sizes(capacity: usize) struct { total: usize, lc: usize, ring: usize, pk: usize } {
        const ring_rows = @min(capacity, index_ring);
        const lc = capacity * lw * 2;
        const ring = ring_rows * i_dim * 2;
        const pk = (capacity / pool + 2) * i_dim * 2;
        var t: usize = a256(4) * 2 + a256(kda_layers * conv_layer * 2) + a256(kda_layers * rec_layer * 4);
        t += a256(kda_layers * window_rows * kda_width * 2);
        const kv = kda_layers * window_rows * kda_n;
        t += a256(kv * 4) + a256(kv * 2) + a256(kv * 4) + a256(kda_layers * window_rows * c0.kda_heads * 4);
        t += (dsa_layers + 1) * (a256(lc) + 2 * a256(ring) + a256(pk));
        return .{ .total = t, .lc = lc, .ring = ring, .pk = pk };
    }

    /// The bytes a stream of `capacity` positions takes.
    pub fn bytes(capacity: usize) usize {
        return sizes(capacity).total;
    }

    /// A zeroed state for `capacity` positions (cuda_state.zig capacity: align64(context + 8) is the caller's).
    pub fn init(r: *const hip.Runtime, capacity: usize) !State {
        if (capacity == 0 or capacity % pool != 0) return error.BadCapacity;
        const z = sizes(capacity);
        var buf = try hip.DeviceBuffer.alloc(r, z.total);
        errdefer buf.free();
        try buf.fill8(0);
        var c: Carve = .{ .base = buf.base() };
        var st: State = undefined;
        st.buf = buf;
        st.capacity = capacity;
        st.pos = 0;
        st.mtp_len = 0;
        st.base = 0;
        st.mtp_base = 0;
        st.pos_dev = c.take(4);
        st.mtp_pos_dev = c.take(4);
        st.conv = c.take(kda_layers * conv_layer * 2);
        st.rec = c.take(kda_layers * rec_layer * 4);
        st.proj = c.take(kda_layers * window_rows * kda_width * 2);
        const kv = kda_layers * window_rows * kda_n;
        st.save_k = c.take(kv * 4);
        st.save_v = c.take(kv * 2);
        st.save_g = c.take(kv * 4);
        st.save_b = c.take(kda_layers * window_rows * c0.kda_heads * 4);
        st.ring_rows = @min(capacity, index_ring);
        const pk_rows = capacity / pool + 2;
        for (0..dsa_layers + 1) |j| {
            const set: Caches = .{ .lc = c.take(z.lc), .ik = c.take(z.ring), .ig = c.take(z.ring), .pk = c.take(z.pk), .pk_rows = pk_rows };
            st.rings[j] = .{ set.ik, set.ig };
            if (j < dsa_layers) st.caches[j] = set else st.mtp = set;
        }
        std.debug.assert(c.used <= z.total);
        return st;
    }

    pub fn deinit(st: *State) void {
        st.buf.free();
        st.* = undefined;
    }

    /// A fresh sequence on `s`: conv windows and states zeroed, positions 0 (the caches past them are never read).
    pub fn reset(st: *State, s: hip.abi.Stream) !void {
        if (s == hip.counting) return;
        const api = st.buf.r.api;
        try hip.runtime.check(api.hipMemsetAsync(@ptrFromInt(st.conv), 0, kda_layers * conv_layer * 2, s));
        try hip.runtime.check(api.hipMemsetAsync(@ptrFromInt(st.rec), 0, kda_layers * rec_layer * 4, s));
        st.setBases(0, 0);
        try st.setPos(s, 0);
        try st.setMtpLen(s, 0);
    }

    fn put32(st: *const State, s: hip.abi.Stream, at: u64, v: usize) !void {
        if (s == hip.counting) return;
        const x: i32 = @intCast(v);
        try hip.runtime.check(st.buf.r.api.hipMemsetD32Async(@ptrFromInt(at), @bitCast(x), 1, s));
    }

    pub fn setPos(st: *State, s: hip.abi.Stream, pos: usize) !void {
        st.pos = pos;
        try st.put32(s, st.pos_dev, pos);
    }

    pub fn setMtpLen(st: *State, s: hip.abi.Stream, n: usize) !void {
        st.mtp_len = n;
        try st.put32(s, st.mtp_pos_dev, n);
    }

    /// Before `rows` rows at pos: within capacity, the index rings holding them (rebased if not).
    pub fn ready(st: *State, s: hip.abi.Stream, rows: usize) !void {
        if (st.pos + rows > st.capacity) return error.PastCapacity;
        if (st.pos + rows > st.base + st.ring_rows) try st.rebase(s, false);
    }

    /// Before the MTP layer's `n` rows at mtp_len, likewise for its caches.
    pub fn readyMtp(st: *State, s: hip.abi.Stream, n: usize) !void {
        if (st.mtp_len + n > st.capacity) return error.PastCapacity;
        if (st.mtp_len + n > st.mtp_base + st.ring_rows) try st.rebase(s, true);
    }

    /// cuda_state.zig rebase: the rings restart at the current pool's first row (MTP: a verify window earlier).
    fn rebase(st: *State, s: hip.abi.Stream, mtp: bool) !void {
        const at = if (mtp) st.mtp_len else st.pos;
        const old = if (mtp) st.mtp_base else st.base;
        const floor = if (mtp) at -| window_rows else at;
        const nb = floor / pool * pool;
        const row = i_dim * 2;
        const keep = (at - nb) * row;
        const sets: []const [2]u64 = if (mtp) st.rings[dsa_layers..] else st.rings[0..dsa_layers];
        if (keep > 0 and s != hip.counting) for (sets) |pair| for (pair) |r| {
            try hip.runtime.check(st.buf.r.api.hipMemcpyDtoDAsync(@ptrFromInt(r), @ptrFromInt(r + (nb - old) * row), keep, s));
        };
        if (mtp) st.setBases(st.base, nb) else st.setBases(nb, st.mtp_base);
    }

    /// The sets' ik and ig pointers for ring bases `main` and `mtp` (position p's row at pointer + p rows).
    fn setBases(st: *State, main: usize, mtp: usize) void {
        st.base = main;
        st.mtp_base = mtp;
        const row = i_dim * 2;
        for (&st.caches, 0..) |*c, j| {
            c.ik = st.rings[j][0] -% main * row;
            c.ig = st.rings[j][1] -% main * row;
        }
        st.mtp.ik = st.rings[dsa_layers][0] -% mtp * row;
        st.mtp.ig = st.rings[dsa_layers][1] -% mtp * row;
    }

    /// KDA layer `li`'s (its index among the KDA layers) state and conv window.
    pub fn recAt(st: *const State, li: usize) u64 {
        return st.rec + li * rec_layer * 4;
    }

    pub fn convAt(st: *const State, li: usize) u64 {
        return st.conv + li * conv_layer * 2;
    }
};

/// A window's buffers that live across the layers (cuda_state.zig Buffers' x, normed, post, comb, the partial, the
/// tail and the MTP rows), and the arena the blocks take their scratch from.
pub const Buffers = struct {
    buf: hip.DeviceBuffer,
    arena: hip.Arena,
    rows: usize,
    prompt: bool,
    ids: u64, // i32 [rows]
    x: u64, // bf16 [rows][4][D]: the 4 residual streams
    normed: u64, // bf16 [rows][D]: a block's input (hc_pre's collapsed, normed row)
    post: u64, // fp32 [rows][4]
    comb: u64, // fp32 [rows][4][4]
    hcpart: u64, // fp32 [rows][16][32]
    part: u64, // fp32 [rows][D]: a block's output (the mixer's o_proj, the FFN's sum)
    hidden: u64, // bf16 [rows][D]: the stream mean
    fnormed: u64, // bf16 [rows][D]: the final-normed rows (lm_head's and the MTP layer's input)
    hin: u64, // bf16 [rows][D]: the MTP layer's hidden rows
    me: u64, // bf16 [rows][D]: the next tokens' embedding
    mcat: u64, // bf16 [rows][2 D]: [enorm(me) | hnorm(hin)]
    mx: u64, // bf16 [rows][D]: the MTP layer's residual

    fn fixed(rows: usize) [15]usize {
        return .{ rows * 4, rows * 4 * D * 2, rows * D * 2, rows * 4 * 4, rows * 16 * 4, hip.launches.glm.hcPartBytes(rows), rows * D * 4, rows * D * 2, rows * D * 2, rows * D * 2, rows * D * 2, rows * 2 * D * 2, rows * D * 2, 0, 0 };
    }

    /// Buffers for windows of up to `rows` rows (`prompt`: prompt chunks on the WMMA tiles) of a stream of `capacity`.
    pub fn init(r: *const hip.Runtime, rows: usize, prompt: bool, capacity: usize) !Buffers {
        if (rows == 0 or rows > prompt_rows) return error.BadRows;
        const f = fixed(rows);
        var total: usize = 0;
        for (f) |n| total += a256(n);
        var buf = try hip.DeviceBuffer.alloc(r, total);
        errdefer buf.free();
        try buf.fill8(0);
        var c: Carve = .{ .base = buf.base() };
        var p: [f.len]u64 = undefined;
        for (f, &p) |n, *x| x.* = c.take(n);
        var arena = try hip.Arena.init(r, scratchBytes(rows, prompt, capacity));
        errdefer arena.deinit();
        return .{ .buf = buf, .arena = arena, .rows = rows, .prompt = prompt, .ids = p[0], .x = p[1], .normed = p[2], .post = p[3], .comb = p[4], .hcpart = p[5], .part = p[6], .hidden = p[7], .fnormed = p[8], .hin = p[9], .me = p[10], .mcat = p[11], .mx = p[12] };
    }

    pub fn deinit(b: *Buffers) void {
        b.arena.deinit();
        b.buf.free();
        b.* = undefined;
    }
};

/// The vocabulary rows the head writes a call: logits fp32 [head_rows][V] in the arena.
pub const head_rows = 256;
/// Rows a dense or listed attention call takes, and a pool-score block (cuda_dsa_block.zig prompt_att_rows).
pub const att_rows = 512;
/// Attention parts a row may need: 2,051 keys at most (dense under the limit, or a listed row) in parts of 512.
pub const att_parts = 5;

/// The scratch the largest block of a window of `rows` rows takes from the arena (forward.zig's takes, each
/// 256-byte aligned; 64 KiB spare for the alignment of about 40 takes).
pub fn scratchBytes(rows: usize, prompt: bool, capacity: usize) usize {
    const R = rows;
    const B = @min(R, att_rows);
    const V: usize = c0.vocab;
    const kda = R * kda_width * 2 + 2 * R * 128 * 2 + 2 * R * kda_n * 2 + 2 * R * kda_n * 4 + R * kda_n * 2 +
        R * kda_n * 4 + R * c0.kda_heads * 4 + R * kda_n * 4 + R * kda_n * 2;
    const np_max = capacity / pool;
    const H: usize = c0.mla_heads;
    const dsa = R * c0.xProj() * 2 + R * c0.q_lora * 2 + R * lw * 2 + R * c0.qrProj() * 2 + R * i_dim * 4 +
        2 * R * H * lw * 2 + R * H * c0.v_dim * 2 + R * 2051 * 4 + 2 * R * 4 + B * np_max * 4 + 2 * B * 4 +
        B * H * att_parts * (lw + 2) * 4;
    const dense = (if (prompt) R * 2 * c0.dense_inter * 4 else 0) + R * c0.dense_inter * 2;
    const pairs = R * c0.topk;
    const E: usize = c0.experts;
    const ex: hip.exl3.Experts = .{ .gate = 0, .up = 0, .down = 0, .suh_g = 0, .suh_u = 0, .svh_g = 0, .svh_u = 0, .suh_d = 0, .svh_d = 0, .count = E, .dims = D, .width = c0.moe_inter, .shared_input = false };
    const x3 = hip.exl3.scratchBytes(ex, R, c0.topk, prompt) catch unreachable;
    const moe = R * E * 4 + 2 * pairs * 4 + pairs * 4 + 3 * (@min(pairs, E + 1) + pairs) * 4 +
        hip.Launcher.routeScratch(pairs, E + 1) + pairs * D * 4 + x3 + (if (prompt) R * 2 * c0.shared_inter * 4 else 0) +
        R * c0.shared_inter * 2 + R * D * 4;
    const head = @min(R, head_rows) * V * 4;
    return @max(@max(kda, dsa), @max(@max(dense, moe), head)) + (64 << 10);
}

test "glm fwd state: the CUDA layout's bytes at 16K (one KDA state copy, BF16 latent)" {
    const cap: usize = 16448; // align64(16,384 + 8)
    const b = State.bytes(cap);
    // 34 states 136 MiB + 12 sets of latent (16 MiB), two 1 MiB rings and pooled keys (1 MiB), the windows, conv
    try std.testing.expect(b > 136 << 20 and b < 512 << 20); // about 403 MiB
    try std.testing.expectEqual(@as(usize, 24896), kda_width);
}
