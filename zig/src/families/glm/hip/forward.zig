//! GLM-5.3-Flash's eager forward on one HIP device: a prompt chunk (up to 2,048 rows) or a decode window through the
//! 45 layers, the final norm, lm_head, and the MTP layer. One stream, no graphs, no TP. The op order and the buffer
//! dtypes between blocks are the CUDA path's (cuda_forward.zig whole/finish/mtp, cuda_kda_block.zig,
//! cuda_dsa_block.zig, cuda_moe.zig, cuda_mlp.zig); each op is a lane's launcher:
//!   G4 hip.launches.glm (mHC, norms, residual, stream mean, embed, GLM SwiGLU) and the BF16 tiles (bf16Go, bf16GluGo);
//!   G2 the KDA launchers on the Launcher (tf_kda_prep, tf_kda, tf_kda_out, tf_kda_conv_shift);
//!   G3 hip.launches.glm_dsa (latent cache, absorb/expand, indexer, select, latent attention);
//!   G1 hip.exl3.run (the routed EXL3 experts over tf_moe_route's plan);
//!   Nemotron-H's sigmoid pick and combine (nemotron_moe.hpp), as G4's router test runs them.
//! Every product of a prompt window runs on the WMMA tiles (`Buffers.prompt`), of a decode window on the stream tiles:
//! a row's bits do not depend on the chunk it is in. Under TF_GLM_DUMP `seq` receives the per-layer dumps.
//! The MTP rounds (cuda_engine.zig draft/verify/keep, cuda_forward.zig commit): `draft` chains the MTP layer's greedy
//! drafts, `window` runs the pending token and the drafts as one decode window that holds the KDA states, `keep`
//! commits its first rows. A kept window row has the bits of the same row run serially (LANES.md code contract 7).

const std = @import("std");
const hip = @import("hip");
const config = @import("../config.zig");
const weights = @import("weights.zig");
const dump = @import("dump.zig");
const state = @import("state.zig");

const glue = hip.launches.glm;
const dsa = hip.launches.glm_dsa;
const nemo = hip.launches.nemotron_moe;
const K = hip.launches.affine.Kernels;
const S = hip.abi.Stream;

pub const State = state.State;
pub const Buffers = state.Buffers;

const D: usize = 4096;
const V: usize = 154880;
const streams: usize = 4;
const kda_n: usize = 8192;
const kda_heads: usize = 64;
const mla_heads: usize = 64;
const top_k: usize = 8;
const experts: usize = 288;
/// The latent attention's scale: qk 256 ** -0.5 (cuda_dsa_block.zig attend).
const att_scale: f32 = 1.0 / 16.0;

fn c(a: u64) ?*const anyopaque {
    return @ptrFromInt(a);
}
fn m(a: u64) ?*anyopaque {
    return @ptrFromInt(a);
}
fn cf(a: u64) ?[*]const f32 {
    return @ptrFromInt(a);
}
fn mf(a: u64) ?[*]f32 {
    return @ptrFromInt(a);
}
fn ci(a: u64) ?[*]const i32 {
    return @ptrFromInt(a);
}
fn mi(a: u64) ?[*]i32 {
    return @ptrFromInt(a);
}
fn int(v: usize) c_int {
    return @intCast(v);
}

/// G0's weights.Experts is G1's hip.exl3.Experts field for field (G0-load.md section 2).
fn exl3Experts(e: weights.Experts) hip.exl3.Experts {
    var o: hip.exl3.Experts = undefined;
    inline for (@typeInfo(hip.exl3.Experts).@"struct".field_names) |name| @field(o, name) = @field(e, name);
    return o;
}

pub const Forward = struct {
    l: *const hip.Launcher,
    k: dsa.Kernels,
    w: *const weights.Model,
    cfg: config.Config,
    s: S,
    b: *Buffers,
    /// TF_GLM_DUMP: the prompt's dumps, and the sequence row of this window's first row.
    seq: ?*dump.Seq = null,
    row0: usize = 0,
    /// Host scratch for the dumps' downloads.
    host: std.mem.Allocator = std.heap.page_allocator,
    /// The last DSA block's intermediates in the arena (valid until the next block): an op-by-op check reads them.
    dsa_last: DsaLast = .{},
    /// TF_GLM_FORCE_PICKS (a diagnostic, never served): each MoE layer takes these picks and weights for its rows in
    /// place of its own router's, so a compare measures the numerics with the reference's routing.
    force: ?*const Force = null,
    /// The last MoE block's picks and weights in the arena (valid until the next block).
    moe_last: struct { pick: u64 = 0, wts: u64 = 0 } = .{},
    /// Inside `window`: the KDA layers hold their states and save the rows for `keep`.
    hold: bool = false,

    pub const DsaLast = struct { dp: u64 = 0, qr: u64 = 0, lat: u64 = 0, qq: u64 = 0, qa: u64 = 0, ol: u64 = 0, vn: u64 = 0 };

    /// Picks int16 and weights fp32, [router layers 46][rows][8] as G5's router_picks / router_weights hold
    /// them (layer 45: the MTP layer; rows: sequence rows from the prompt's first).
    pub const Force = struct { picks: []const i16, wts: []const f32, rows: usize };

    pub fn init(l: *const hip.Launcher, w: *const weights.Model, cfg: config.Config, s: S, b: *Buffers) !Forward {
        return .{ .l = l, .k = try dsa.Kernels.load(l), .w = w, .cfg = cfg, .s = s, .b = b };
    }

    fn ops(f: *const Forward) hip.ops.Ops {
        return .{ .l = f.l, .bf16 = true, .stream = f.s, .arena = &f.b.arena, .prefill = f.b.prompt };
    }

    /// A BF16 linear: out (rows, w.n) = x (rows, w.k) . w^T, bf16 with `half`, else fp32; the window's tile.
    fn lin(f: *const Forward, x: u64, w: weights.Bf16, rows: usize, out: u64, half: bool) !void {
        const a = K.bf16Arg(x, w.w, if (half) 0 else out, if (half) out else 0, int(rows), @intCast(w.n), @intCast(w.k));
        try f.l.affine.bf16Go(f.l.r, a, f.b.prompt, f.s);
    }

    /// The shared expert's and the dense MLP's [gate | up] with GLM's SwiGLU at CUDA's rounding points (limit 10;
    /// glm_glue_moe.cuh swiglu1: g and u bf16, bf16(silu(min(g, 10))), bf16 of the product) into act bf16 (rows, cols),
    /// as G4's tests compose it: prompt rows bf16 [gate | up] on the WMMA tile, then glm.swiglu; decode rows bf16GluGo.
    fn swiglu(f: *Forward, x: u64, gu: weights.Bf16, cols: usize, rows: usize, act: u64) !void {
        const limit = f.cfg.swiglu_limit;
        if (f.b.prompt) {
            const both = try f.b.arena.of(u16, rows * 2 * cols);
            try f.lin(x, gu, rows, both, true);
            return glue.swiglu(f.l, c(both), m(act), int(rows), int(cols), limit, f.s);
        }
        try f.l.affine.bf16GluGo(f.l.r, x, gu.w, act, f.l.glm.silu, int(rows), int(cols), @intCast(gu.k), limit, f.s);
    }

    pub fn sync(f: *const Forward) !void {
        try hip.runtime.check(f.l.r.api.hipStreamSynchronize(f.s));
    }

    fn upload(f: *const Forward, dst: u64, bytes: []const u8) !void {
        try f.sync(); // the previous window's kernels are done with dst
        try hip.runtime.check(f.l.r.api.hipMemcpyHtoD(m(dst), bytes.ptr, bytes.len));
    }

    /// `out.len` values of T from device address `src` (after the stream's work).
    pub fn download(f: *const Forward, comptime T: type, src: u64, out: []T) !void {
        try f.sync();
        try hip.runtime.check(f.l.r.api.hipMemcpyDtoH(@ptrCast(out.ptr), m(src), out.len * @sizeOf(T)));
    }

    // ---- the layers ----

    /// forward.forward of `ids` on `st` at st.pos: the final-normed rows in b.fnormed; the KDA layers and the position
    /// committed (every row kept).
    pub fn run(f: *Forward, st: *State, ids: []const u32) !void {
        if (st.pending != 0) return error.WindowNotKept;
        try f.layers(st, ids);
        try st.setPos(f.s, st.pos + ids.len);
    }

    /// A verify window (cuda_engine.zig verify): `ids` (the pending token, then the drafts; at most
    /// state.window_rows) from st.pos on decode buffers, the final-normed rows in b.fnormed (lm_head: `head`). Nothing
    /// is committed: the KDA layers hold their states (tf_kda_hold) and save each row's projection and k, v, decay and
    /// beta in st; the DSA latent, index and pool rows are written at [pos, pos + R), which a later window rewrites;
    /// st.pos stays and st.pending = R until `keep`.
    pub fn window(f: *Forward, st: *State, ids: []const u32) !void {
        if (f.b.prompt or ids.len > state.window_rows) return error.BadWindow;
        if (st.pending != 0) return error.WindowNotKept;
        f.hold = true;
        defer f.hold = false;
        try f.layers(st, ids);
        st.pending = ids.len;
    }

    /// forward.commit (cuda_forward.zig commit) of the last window's first `rows`: each KDA layer's state runs in place
    /// through its first `rows` saved rows (tf_kda_keep: the serial rows' sums), the conv windows take the kept
    /// projections, the position moves by `rows`. The window's later rows are rewritten by the next window.
    pub fn keep(f: *Forward, st: *State, rows: usize) !void {
        if (rows == 0 or rows > st.pending) return error.BadKeep;
        const W = state.kda_width;
        const wr = state.window_rows;
        try f.l.tf_kda_keep(mf(st.rec), @intCast(State.rec_layer), cf(st.save_k), c(st.save_v), cf(st.save_g), cf(st.save_b), @intCast(wr * kda_n), @intCast(wr * kda_heads), int(rows), int(state.kda_layers), int(kda_heads), f.s);
        try f.l.tf_kda_conv_shift(m(st.conv), c(st.proj), @intCast(rows), @intCast(State.conv_layer), @intCast(wr * W), @intCast(W), @intCast(3 * kda_n), int(state.kda_layers), f.s);
        st.pending = 0;
        try st.setPos(f.s, st.pos + rows);
    }

    /// The 45 layers and the final norm of `ids` at st.pos (run's and window's body); the position not moved.
    fn layers(f: *Forward, st: *State, ids: []const u32) !void {
        const R = ids.len;
        const b = f.b;
        const cfg = f.cfg;
        if (R == 0 or R > b.rows) return error.BadWindow;
        try st.ready(f.s, R);
        try f.upload(b.ids, std.mem.sliceAsBytes(ids));
        try glue.embed(f.l, c(b.ids), c(f.w.embed), m(b.x), int(R), int(D), int(streams), f.s);
        var li: usize = 0;
        var di: usize = 0;
        for (f.w.layers, 0..) |ly, j| {
            const hc = ly.attn_hc.?;
            try glue.hcPre(f.l, c(b.x), c(hc.fn_w), c(hc.base), c(hc.scale), c(ly.in_norm), m(b.normed), m(b.post), m(b.comb), m(b.hcpart), cfg.eps, cfg.hc_eps, @intCast(cfg.sinkhorn), int(R), f.s);
            try f.dumpRows(j, 0, b.normed, R);
            switch (ly.mixer) {
                .kda => |w| {
                    try f.kdaBlock(st, li, w, R);
                    li += 1;
                },
                .mla => |w| {
                    try f.dsaBlock(st.caches[di], st.pos_dev, st.pos, w, R);
                    di += 1;
                },
            }
            try f.dumpRows(j, 1, b.part, R);
            try glue.hcPost(f.l, c(b.x), m(b.x), c(b.part), 1, 0, c(b.post), c(b.comb), int(R), f.s);
            try f.dumpStreams(j, 0, R);
            const fh = ly.ffn_hc.?;
            try glue.hcPre(f.l, c(b.x), c(fh.fn_w), c(fh.base), c(fh.scale), c(ly.post_norm), m(b.normed), m(b.post), m(b.comb), m(b.hcpart), cfg.eps, cfg.hc_eps, @intCast(cfg.sinkhorn), int(R), f.s);
            try f.dumpRows(j, 2, b.normed, R);
            switch (ly.ffn) {
                .dense => |w| try f.denseBlock(w, R),
                .moe => |w| try f.moeBlock(w, j, R),
            }
            try f.dumpRows(j, 3, b.part, R);
            try glue.hcPost(f.l, c(b.x), m(b.x), c(b.part), 1, 0, c(b.post), c(b.comb), int(R), f.s);
            try f.dumpStreams(j, 1, R);
        }
        // finish: the stream mean, the final norm (cuda_forward.zig whole's tail and finish)
        try glue.streamMean(f.l, c(b.x), m(b.hidden), int(R), f.s);
        try glue.rms(f.l, c(b.hidden), @intCast(D), c(f.w.norm), m(b.fnormed), @intCast(D), int(R), int(D), cfg.eps, f.s);
    }

    /// cuda_kda_block.zig body + fill: projections, the chain (prep, scan, gated norm), the conv window's commit, o_proj
    /// into b.part fp32. The scan runs layer `li`'s state in place (G2: one state copy). Under `hold` (a verify
    /// window) the scan leaves the state (tf_kda_hold), the projection rows and k, v, decay and beta go to st's window
    /// saves, and the conv window waits for `keep`.
    pub fn kdaBlock(f: *Forward, st: *State, li: usize, w: weights.Kda, R: usize) !void {
        const b = f.b;
        const ar = &b.arena;
        ar.reset();
        const W = state.kda_width;
        const n = kda_n;
        const fa = 3 * n;
        const wr = state.window_rows;
        const p = if (f.hold) st.proj + li * wr * W * 2 else try ar.of(u16, R * W);
        const xa = try ar.of(u16, R * 128);
        const xg = try ar.of(u16, R * 128);
        const ka = try ar.of(u16, R * n);
        const kg = try ar.of(u16, R * n);
        const q = try ar.of(f32, R * n);
        const k = if (f.hold) st.save_k + li * wr * n * 4 else try ar.of(f32, R * n);
        const v = if (f.hold) st.save_v + li * wr * n * 2 else try ar.of(u16, R * n);
        const g = if (f.hold) st.save_g + li * wr * n * 4 else try ar.of(f32, R * n);
        const beta = if (f.hold) st.save_b + li * wr * kda_heads * 4 else try ar.of(f32, R * kda_heads);
        const y = try ar.of(f32, R * n);
        const out = try ar.of(u16, R * n);
        try f.lin(b.normed, w.proj, R, p, true);
        // f_b and g_b read the f_a and g_a columns of the projection rows (cuda_kda_block.zig project's x_stride):
        // bf16Go takes rows k apart, so the columns are copied out first (exact)
        try f.l.tf_copy_cols(c(p), @intCast(W), @intCast(fa), m(xa), 2, int(R), 128, f.s);
        try f.l.tf_copy_cols(c(p), @intCast(W), @intCast(fa + 128), m(xg), 2, int(R), 128, f.s);
        try f.lin(xa, w.f_b, R, ka, true);
        try f.lin(xg, w.g_b, R, kg, true);
        const conv = st.convAt(li);
        try f.l.tf_kda_prep(c(p), @intCast(W), @intCast(fa + 256), c(ka), @intCast(n), c(conv), c(w.conv), cf(w.a_log), cf(w.dt_bias), f.cfg.lower_bound, mf(q), mf(k), m(v), mf(g), mf(beta), int(R), int(kda_heads), f.s);
        if (f.hold) {
            try f.l.tf_kda_hold(cf(q), cf(k), c(v), cf(g), cf(beta), mf(st.recAt(li)), mf(y), int(R), int(kda_heads), f.s);
        } else try f.l.tf_kda(cf(q), cf(k), c(v), cf(g), cf(beta), mf(st.recAt(li)), mf(y), int(R), int(kda_heads), f.s, null, 0);
        try f.l.tf_kda_out(cf(y), c(kg), @intCast(n), c(w.o_norm), f.cfg.eps, m(out), int(R), int(kda_heads), f.s);
        // every row kept: the conv window becomes the last 3 inputs (cuda_kda_block.zig bodyProfile); a held window's
        // waits for keep
        if (!f.hold) try f.l.tf_kda_conv_shift(m(conv), c(p), @intCast(R), @intCast(State.conv_layer), @intCast(R * W), @intCast(W), @intCast(3 * n), 1, f.s);
        try f.lin(out, w.o, R, b.part, false);
    }

    /// cuda_dsa_block.zig body + fill on cache set `cs` at device position `pos` (host_pos on the host): projections,
    /// norms, absorbed query, latent and index caches, dense or listed latent attention in 512-row calls, expand,
    /// o_proj into b.part fp32.
    pub fn dsaBlock(f: *Forward, cs: state.Caches, pos: u64, host_pos: usize, w: weights.Mla, R: usize) !void {
        const b = f.b;
        const ar = &b.arena;
        ar.reset();
        const cfg = f.cfg;
        const X: usize = cfg.xProj(); // [q_a | kv_a | wk | weights_proj]
        const Q: usize = cfg.qrProj(); // [q_b | wq_b]
        const ql: usize = cfg.q_lora;
        const lw: usize = cfg.kv_lora;
        const H = mla_heads;
        const dp = try ar.of(u16, R * X);
        const qr = try ar.of(u16, R * ql);
        const lat = try ar.of(u16, R * lw);
        const qq = try ar.of(u16, R * Q);
        const igr = try ar.of(f32, R * cfg.i_dim);
        const qa = try ar.of(u16, R * H * lw);
        const ol = try ar.of(u16, R * H * lw);
        const vn = try ar.of(u16, R * H * cfg.v_dim);
        try f.lin(b.normed, w.x, R, dp, true);
        try glue.rms(f.l, c(dp), @intCast(X), c(w.q_norm), m(qr), @intCast(ql), int(R), int(ql), cfg.eps, f.s);
        try glue.rms(f.l, c(dp + ql * 2), @intCast(X), c(w.kv_norm), m(lat), @intCast(lw), int(R), int(lw), cfg.eps, f.s);
        try f.lin(qr, w.qr, R, qq, true);
        try f.lin(b.normed, w.index.gate, R, igr, false);
        try dsa.absorbQ(f.k, f.s, qq, Q, dsa.kvKeys(w.kv_b.w), dsa.kv_head, qa, R, H);
        try dsa.latentWrite(f.k, f.s, lat, lw, cs.lc, pos, 0, R);
        const wk_col = ql + lw; // the indexer's key rows, then its head weights, in the xProj rows
        try dsa.indexUpdate(f.k, f.s, dp + wk_col * 2, X, igr, w.index.ln_w, w.index.ln_b, w.index.ape, cs.ik, cs.ig, cs.pk, pos, 0, R);
        // rows past the dense limit (2,051 keys) select their tokens; a window that has none attends densely
        const listed = host_pos + R - 1 >= cfg.keyWidth();
        const B: usize = @min(R, state.att_rows);
        var tokens: u64 = 0;
        var counts: u64 = 0;
        var sparse: u64 = 0;
        if (listed) {
            tokens = try ar.of(i32, R * dsa.width);
            counts = try ar.of(i32, R);
            sparse = try ar.of(i32, R);
            const np_max = cs.pk_rows - 2;
            const sc: dsa.Scratch = .{ .scores = try ar.of(f32, B * np_max), .complete = try ar.of(i32, B), .ends = try ar.of(i32, B) };
            var r0: usize = 0;
            while (r0 < R) : (r0 += B) {
                const n: usize = @min(B, R - r0);
                const np = dsa.poolBucket(host_pos + r0, n, np_max);
                try dsa.selectTokens(f.k, f.s, qq + (r0 * Q + cfg.mla_heads * cfg.nope) * 2, Q, dp + (r0 * X + wk_col + cfg.i_dim) * 2, X, cs.pk, pos, r0, host_pos, n, np, sc, tokens + r0 * dsa.width * 4, counts + r0 * 4, sparse + r0 * 4);
            }
        }
        const pt: dsa.Partials = .{ .po = try ar.of(f32, B * H * state.att_parts * lw), .pms = try ar.of(f32, B * H * state.att_parts * 2) };
        var r0: usize = 0;
        while (r0 < R) : (r0 += B) {
            const n: usize = @min(B, R - r0);
            const keys: dsa.Keys = if (listed) .{ .listed = .{ .ids = tokens + r0 * dsa.width * 4, .id_stride = dsa.width, .counts = counts + r0 * 4, .sparse = sparse + r0 * 4 } } else .dense;
            const parts = if (listed) dsa.partsFor(dsa.width) else dsa.partsFor(host_pos + r0 + n);
            const off = r0 * H * lw * 2;
            try dsa.attention(f.k, f.s, qa + off, cs.lc, pos, r0, keys, pt, ol + off, n, H, parts, att_scale);
        }
        try dsa.expandV(f.k, f.s, ol, dsa.kvValues(w.kv_b.w), dsa.kv_head, vn, R, H);
        f.dsa_last = .{ .dp = dp, .qr = qr, .lat = lat, .qq = qq, .qa = qa, .ol = ol, .vn = vn };
        try f.lin(vn, w.o, R, b.part, false);
    }

    /// cuda_mlp.zig: [gate | up], SwiGLU, down into b.part fp32.
    pub fn denseBlock(f: *Forward, w: weights.Dense, R: usize) !void {
        const b = f.b;
        b.arena.reset();
        const width: usize = w.down.k;
        const act = try b.arena.of(u16, R * width);
        try f.swiglu(b.normed, w.gu, width, R, act);
        try f.lin(act, w.down, R, b.part, false);
    }

    /// cuda_moe.zig body + fill: router logits, top-8 by sigmoid + bias (normed, x 2.5), the plan, the routed EXL3
    /// experts, the shared expert, the weighted sum of the slots plus the shared row into b.part fp32. `layer`: the
    /// dump's router row (45 for the MTP layer).
    pub fn moeBlock(f: *Forward, w: weights.Moe, layer: usize, R: usize) !void {
        const b = f.b;
        const ar = &b.arena;
        ar.reset();
        const E = experts;
        const pairs = R * top_k;
        const mlog = try ar.of(f32, R * E);
        const pick = try ar.of(i32, pairs);
        const wts = try ar.of(f32, pairs);
        try f.lin(b.normed, w.router, R, mlog, false);
        try nemo.select(f.l, cf(mlog), cf(w.bias), mi(pick), mf(wts), null, null, 0, int(R), int(E), int(top_k), f.cfg.routed_scale, f.cfg.norm_topk, f.s);
        if (f.force) |fc| try f.forcePicks(fc, layer, R, pick, wts);
        f.moe_last = .{ .pick = pick, .wts = wts };
        // the plan over the routed pairs (experts + 1 as G1's tests plan them); 128 pairs an item in a prompt
        const tile: usize = if (b.prompt) 128 else 8;
        const cap: usize = @min(pairs, E + 1) + pairs / tile;
        const members = try ar.of(i32, pairs);
        const items = try ar.of(i32, 3 * cap);
        const scratch: u64 = if (pairs >= 4 * hip.Launcher.route_seg) try ar.take(hip.Launcher.routeScratch(pairs, E + 1)) else 0;
        try f.l.tf_moe_route_wide(ci(pick), int(pairs), int(E + 1), int(tile), mi(members), mi(items), int(cap), scratch, f.s);
        const ey = try ar.of(f32, pairs * D);
        try hip.exl3.run(f.ops(), exl3Experts(w.experts), b.normed, D, .{ .pick = pick, .items = items, .count = cap, .members = members, .tile = tile }, R, top_k, b.prompt, ey);
        const sw: usize = w.shared_down.k;
        const sact = try ar.of(u16, R * sw);
        try f.swiglu(b.normed, w.shared_gu, sw, R, sact);
        const sy = try ar.of(f32, R * D);
        try f.lin(sact, w.shared_down, R, sy, false);
        try nemo.combine(f.l, cf(ey), cf(wts), cf(sy), m(b.part), 0, int(R), int(top_k), int(D), f.s);
        if (f.seq) |s| try f.dumpRouter(s, layer, pick, wts, R);
    }

    /// The forced picks and weights of rows row0 .. row0 + R of router layer `layer` over the router's.
    fn forcePicks(f: *Forward, fc: *const Force, layer: usize, R: usize, pick: u64, wts: u64) !void {
        if (f.row0 + R > fc.rows or layer >= dump.router_layers) return error.ForceRows;
        const ids = try f.host.alloc(i32, R * top_k);
        defer f.host.free(ids);
        const ws = try f.host.alloc(f32, R * top_k);
        defer f.host.free(ws);
        const at = (layer * fc.rows + f.row0) * top_k;
        for (ids, fc.picks[at..][0 .. R * top_k]) |*o, v| {
            if (v < 0 or v >= experts) return error.ForcePick;
            o.* = v;
        }
        @memcpy(ws, fc.wts[at..][0 .. R * top_k]);
        try f.upload(pick, std.mem.sliceAsBytes(ids));
        try f.upload(wts, std.mem.sliceAsBytes(ws));
    }

    // ---- the head and the MTP layer ----

    /// lm_head of `rows` (at most state.head_rows) bf16 rows at `x`: logits fp32 [rows][V] in the arena (valid until
    /// the next block).
    pub fn head(f: *Forward, x: u64, rows: usize) !u64 {
        if (rows == 0 or rows > state.head_rows) return error.BadWindow;
        f.b.arena.reset();
        const out = try f.b.arena.of(f32, rows * V);
        try f.lin(x, f.w.head, rows, out, false);
        return out;
    }

    /// mtp.mtp_forward (cuda_forward.zig mtp) of `next.len` rows on st's MTP caches from mtp_len: `hin` bf16 [rows][D]
    /// the main model's final-normed rows, `next` each row's next token. The shared_head-normed rows in b.fnormed (the
    /// head's input: the BF16 lm_head, D-2; no 4-bit draft copy).
    pub fn mtp(f: *Forward, st: *State, next: []const u32, hin: u64) !void {
        const n = next.len;
        const b = f.b;
        const cfg = f.cfg;
        const t = f.w.mtp;
        const ly = t.layer;
        if (n == 0 or n > b.rows) return error.BadWindow;
        try st.readyMtp(f.s, n);
        try f.upload(b.ids, std.mem.sliceAsBytes(next));
        try glue.embed(f.l, c(b.ids), c(f.w.embed), m(b.me), int(n), int(D), 1, f.s);
        if (st.mtp_len == 0) try hip.runtime.check(f.l.r.api.hipMemsetAsync(m(b.me), 0, D * 2, f.s)); // position 0's embedding
        try glue.rms(f.l, c(b.me), @intCast(D), c(t.enorm), m(b.mcat), @intCast(2 * D), int(n), int(D), cfg.eps, f.s);
        try glue.rms(f.l, c(hin), @intCast(D), c(t.hnorm), m(b.mcat + D * 2), @intCast(2 * D), int(n), int(D), cfg.eps, f.s);
        try f.lin(b.mcat, t.eh, n, b.mx, true);
        try glue.rms(f.l, c(b.mx), @intCast(D), c(ly.in_norm), m(b.normed), @intCast(D), int(n), int(D), cfg.eps, f.s);
        try f.dumpMtp(0, b.normed, true, n);
        try f.dsaBlock(st.mtp, st.mtp_pos_dev, st.mtp_len, ly.mixer.mla, n);
        try f.dumpMtp(1, b.part, false, n);
        try glue.residual(f.l, c(b.mx), m(b.mx), c(b.part), 1, 0, int(n), int(D), f.s);
        try f.dumpMtpStream(0, n);
        try glue.rms(f.l, c(b.mx), @intCast(D), c(ly.post_norm), m(b.normed), @intCast(D), int(n), int(D), cfg.eps, f.s);
        try f.dumpMtp(2, b.normed, true, n);
        try f.moeBlock(ly.ffn.moe, dump.router_layers - 1, n);
        try f.dumpMtp(3, b.part, false, n);
        try glue.residual(f.l, c(b.mx), m(b.mx), c(b.part), 1, 0, int(n), int(D), f.s);
        try f.dumpMtpStream(1, n);
        try glue.rms(f.l, c(b.mx), @intCast(D), c(t.norm), m(b.fnormed), @intCast(D), int(n), int(D), cfg.eps, f.s);
        try st.setMtpLen(f.s, st.mtp_len + n);
    }

    /// The most drafts a round chains (cuda_engine.zig max_drafts).
    pub const max_drafts = 3;

    /// decode.draft (cuda_engine.zig draft): the MTP layer absorbs the pairs (row j of `hin`, bf16 [rows][D] the main
    /// model's final-normed rows; follow[j], the token after it) at mtp_len, then chains `depth` greedy drafts from the
    /// last pair into `out` (the BF16 lm_head's top-1, ties to the lower id); returns depth. The draft steps' MTP rows
    /// are not kept: mtp_len ends at the absorbed rows, and the next absorb rewrites them. On buffers of its own
    /// (decode, rows >= follow.len): `hin` is usually the window's b.fnormed. Needs mtp_len + follow.len = st.pos.
    pub fn draft(f: *Forward, st: *State, hin: u64, follow: []const u32, depth: usize, out: []u32) !usize {
        if (f.b.prompt or depth > max_drafts or depth > out.len) return error.BadDraft;
        if (st.mtp_len + follow.len != st.pos) return error.MtpOutOfStep;
        var s0: usize = 0;
        var last: u64 = 0;
        while (s0 < follow.len) : (s0 += f.b.rows) {
            const part = follow[s0..@min(follow.len, s0 + f.b.rows)];
            try f.mtp(st, part, hin + s0 * D * 2);
            last = f.b.fnormed + (part.len - 1) * D * 2;
        }
        if (depth == 0) return 0;
        if (follow.len == 0) return error.BadDraft;
        const kept = st.mtp_len;
        const lg = try f.host.alloc(f32, V);
        defer f.host.free(lg);
        for (0..depth) |j| {
            out[j] = try f.top1(last, lg);
            if (j + 1 < depth) {
                try f.mtp(st, out[j..][0..1], last);
                last = f.b.fnormed;
            }
        }
        try st.setMtpLen(f.s, kept);
        return depth;
    }

    /// The greedy pick of one final-normed bf16 row at `x`: lm_head's fp32 row, its first largest value (ties to the
    /// lower id, as dump.top8's first).
    pub fn top1(f: *Forward, x: u64, lg: []f32) !u32 {
        const at = try f.head(x, 1);
        try f.download(f32, at, lg[0..V]);
        var best: usize = 0;
        for (lg[0..V], 0..) |v, i| if (v > lg[best]) {
            best = i;
        };
        return @intCast(best);
    }

    // ---- TF_GLM_DUMP (inert when `seq` is null) ----

    /// Sublayer `k` (0 mixer in bf16, 1 mixer out fp32, 2 FFN in bf16, 3 FFN out fp32) of layer `l`'s dumped rows.
    fn dumpRows(f: *Forward, l: usize, k: u2, src: u64, R: usize) !void {
        const s = f.seq orelse return;
        for (0..R) |r| {
            const pos = f.row0 + r;
            if (s.slotOf(pos) == null) continue;
            if (k == 0 or k == 2) {
                var row: [D]u16 = undefined;
                try f.download(u16, src + r * D * 2, &row);
                try s.sublayerBf16(l, pos, k, &row);
            } else {
                var row: [D]f32 = undefined;
                try f.download(f32, src + r * D * 4, &row);
                try s.sublayer(l, pos, k, &row);
            }
        }
    }

    fn dumpStreams(f: *Forward, l: usize, site: u1, R: usize) !void {
        const s = f.seq orelse return;
        for (0..R) |r| {
            const pos = f.row0 + r;
            if (s.slotOf(pos) == null) continue;
            const row = try f.host.alloc(u16, streams * D);
            defer f.host.free(row);
            try f.download(u16, f.b.x + r * streams * D * 2, row);
            try s.streamBf16(l, pos, site, row);
        }
    }

    fn dumpRouter(f: *Forward, s: *dump.Seq, layer: usize, pick: u64, wts: u64, R: usize) !void {
        const ids = try f.host.alloc(i32, R * top_k);
        defer f.host.free(ids);
        const ws = try f.host.alloc(f32, R * top_k);
        defer f.host.free(ws);
        try f.download(i32, pick, ids);
        try f.download(f32, wts, ws);
        for (0..R) |r| try s.router(layer, f.row0 + r, ids[r * top_k ..][0..top_k], ws[r * top_k ..][0..top_k]);
    }

    fn dumpMtp(f: *Forward, k: u2, src: u64, half: bool, n: usize) !void {
        const s = f.seq orelse return;
        for (0..n) |r| {
            const pos = f.row0 + r;
            if (s.slotOf(pos) == null) continue;
            var row: [D]f32 = undefined;
            try f.rowF32(src, half, r, &row);
            try s.mtpSublayer(pos, k, &row);
        }
    }

    fn dumpMtpStream(f: *Forward, site: u1, n: usize) !void {
        const s = f.seq orelse return;
        for (0..n) |r| {
            const pos = f.row0 + r;
            if (s.slotOf(pos) == null) continue;
            var row: [D]f32 = undefined;
            try f.rowF32(f.b.mx, true, r, &row);
            try s.mtpStream(pos, site, &row);
        }
    }

    /// Row `r` of a [rows][D] buffer (bf16 with `half`, else fp32) as fp32.
    pub fn rowF32(f: *const Forward, src: u64, half: bool, r: usize, out: *[D]f32) !void {
        if (!half) return f.download(f32, src + r * D * 4, out);
        var h: [D]u16 = undefined;
        try f.download(u16, src + r * D * 2, &h);
        for (out, h) |*o, x| o.* = @bitCast(@as(u32, x) << 16);
    }
};

test "glm fwd: the window sizes the blocks assume" {
    const cfg: config.Config = .{};
    try std.testing.expectEqual(@as(u32, 2208), cfg.xProj());
    try std.testing.expectEqual(@as(u32, 20480), cfg.qrProj());
    try std.testing.expectEqual(@as(u32, 2051), cfg.keyWidth());
    try std.testing.expectEqual(@as(usize, 2051), dsa.width);
    try std.testing.expectEqual(@as(u32, 24896), cfg.kdaProj());
    try std.testing.expectEqual(state.kda_width, @as(usize, cfg.kdaProj()));
}
