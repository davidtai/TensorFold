//! TF_GLM_PROFILE: the forward's GPU time by layer and op group, from HIP timing events recorded on its stream between
//! the groups (off by default; it records events only, so no result changes). A mark closes the group since the previous
//! mark: GPU idle time before a group's first launch (the eager host's launch gaps) counts to that group.
//! The routed experts' picks of each MoE layer are copied aside (a small device copy after the group), so a window's
//! EXL3 bytes are counted from the distinct experts it read.

const std = @import("std");
const hip = @import("hip");

/// The op groups. Classes for the summary: `class`.
pub const Op = enum {
    embed,
    hc_pre,
    hc_post,
    kda_proj,
    kda_prep,
    kda_scan,
    kda_out,
    kda_conv,
    kda_oproj,
    dsa_proj,
    dsa_absorb,
    dsa_cache,
    dsa_select,
    dsa_attn,
    dsa_expand,
    dsa_oproj,
    dense_gu,
    dense_down,
    moe_router,
    moe_plan,
    exl3,
    shared_gu,
    shared_down,
    moe_combine,
    tail,
    head,
    mtp_in,
    mtp_residual,
    /// Forward.keep: the KDA states replayed through the kept rows and the conv windows shifted (34 layers each).
    mtp_keep,
    /// Forward.draft's lm_head rows (the draft picks), apart from the verify window's head.
    draft_head,

    pub fn class(o: Op) []const u8 {
        return switch (o) {
            .embed, .hc_pre, .hc_post, .tail, .mtp_residual => "norms/mHC",
            .kda_proj, .kda_oproj => "KDA linears",
            .kda_prep, .kda_scan, .kda_out, .kda_conv => "KDA chain",
            .dsa_proj, .dsa_oproj, .dsa_absorb, .dsa_expand => "DSA linears",
            .dsa_cache, .dsa_select => "DSA indexer",
            .dsa_attn => "DSA attention",
            .dense_gu, .dense_down => "dense MLP",
            .moe_router, .moe_plan => "MoE route",
            .exl3 => "EXL3 experts",
            .shared_gu, .shared_down => "shared expert",
            .moe_combine => "MoE combine",
            .head => "head",
            .mtp_in => "MTP in",
            .mtp_keep => "MTP keep",
            .draft_head => "draft head",
        };
    }
};

pub const classes = [_][]const u8{ "norms/mHC", "KDA linears", "KDA chain", "DSA linears", "DSA indexer", "DSA attention", "dense MLP", "MoE route", "EXL3 experts", "shared expert", "MoE combine", "head", "MTP in", "MTP keep", "draft head" };

pub const Mark = struct { layer: i16, op: Op, rows: u32, bytes: u64 };

/// A measured group: microseconds between its mark and the previous one.
pub const Row = struct { layer: i16, op: Op, rows: u32, us: f64, bytes: u64 };

/// One routed expert's bytes: gate and up trellis [4096/16][2048/16][64] int16, down [2048/16][4096/16][64], suh/svh.
pub const expert_bytes: u64 = 3 * 4096 * 2048 / 2 + 2 * (4096 + 2048) * 2 + (2048 + 4096) * 2;

pub const Prof = struct {
    r: *const hip.Runtime,
    s: hip.abi.Stream,
    ev: []hip.abi.Event,
    marks: []Mark,
    n: usize = 0,
    /// The picks of each MoE group of the window, copied aside: i32 [groups][rows][8].
    picks: hip.DeviceBuffer,
    pick_at: []u64, // byte offset in `picks` of mark i's copy, or maxInt
    pick_used: usize = 0,
    gpa: std.mem.Allocator,

    pub fn init(gpa: std.mem.Allocator, r: *const hip.Runtime, s: hip.abi.Stream, cap: usize, pick_bytes: usize) !Prof {
        const ev = try gpa.alloc(hip.abi.Event, cap);
        errdefer gpa.free(ev);
        var made: usize = 0;
        errdefer for (ev[0..made]) |e| {
            _ = r.api.hipEventDestroy(e);
        };
        for (ev) |*e| {
            try hip.runtime.check(r.api.hipEventCreateWithFlags(e, 0));
            made += 1;
        }
        const pick_at = try gpa.alloc(u64, cap);
        return .{ .r = r, .s = s, .ev = ev, .marks = try gpa.alloc(Mark, cap), .picks = try hip.DeviceBuffer.alloc(r, pick_bytes), .pick_at = pick_at, .gpa = gpa };
    }

    pub fn deinit(p: *Prof) void {
        for (p.ev) |e| _ = p.r.api.hipEventDestroy(e);
        p.gpa.free(p.ev);
        p.gpa.free(p.marks);
        p.gpa.free(p.pick_at);
        p.picks.free();
        p.* = undefined;
    }

    /// A window's first event.
    pub fn start(p: *Prof) !void {
        p.n = 0;
        p.pick_used = 0;
        try hip.runtime.check(p.r.api.hipEventRecord(p.ev[0], p.s));
        p.n = 1;
    }

    /// Closes the group since the previous mark.
    pub fn mark(p: *Prof, layer: i16, op: Op, rows: usize, bytes: u64) !void {
        if (p.n == 0) return;
        if (p.n >= p.ev.len) return error.ProfileFull;
        try hip.runtime.check(p.r.api.hipEventRecord(p.ev[p.n], p.s));
        p.marks[p.n] = .{ .layer = layer, .op = op, .rows = @intCast(rows), .bytes = bytes };
        p.pick_at[p.n] = std.math.maxInt(u64);
        p.n += 1;
    }

    /// Copies an MoE group's picks (i32 [rows][8] at `pick`) aside for its byte count; call after its mark.
    pub fn keepPicks(p: *Prof, pick: u64, rows: usize) !void {
        if (p.n == 0) return;
        const bytes = rows * 8 * 4;
        if (p.pick_used + bytes > p.picks.len) return error.ProfileFull;
        try hip.runtime.check(p.r.api.hipMemcpyDtoDAsync(@ptrFromInt(p.picks.base() + p.pick_used), @ptrFromInt(pick), bytes, p.s));
        p.pick_at[p.n - 1] = p.pick_used;
        p.pick_used += bytes;
    }

    /// The window's rows, after its last event; EXL3 groups' bytes from their distinct experts.
    pub fn collect(p: *Prof, a: std.mem.Allocator) ![]Row {
        if (p.n < 2) return &.{};
        try hip.runtime.check(p.r.api.hipEventSynchronize(p.ev[p.n - 1]));
        try hip.runtime.check(p.r.api.hipStreamSynchronize(p.s));
        const host = try a.alloc(i32, p.pick_used / 4);
        if (host.len > 0) try p.picks.download(0, std.mem.sliceAsBytes(host));
        const out = try a.alloc(Row, p.n - 1);
        for (out, 1..) |*o, i| {
            var ms: f32 = 0;
            try hip.runtime.check(p.r.api.hipEventElapsedTime(&ms, p.ev[i - 1], p.ev[i]));
            const m = p.marks[i];
            var bytes = m.bytes;
            if (p.pick_at[i] != std.math.maxInt(u64)) {
                var seen: [288]bool = @splat(false);
                var distinct: u64 = 0;
                for (host[p.pick_at[i] / 4 ..][0 .. @as(usize, m.rows) * 8]) |e| {
                    const k: usize = @intCast(e);
                    if (k < seen.len and !seen[k]) {
                        seen[k] = true;
                        distinct += 1;
                    }
                }
                bytes = distinct * expert_bytes;
            }
            o.* = .{ .layer = m.layer, .op = m.op, .rows = m.rows, .us = @as(f64, ms) * 1000.0, .bytes = bytes };
        }
        p.n = 0;
        return out;
    }
};

test "glm fwd profile: every op has a class in the summary's list" {
    inline for (@typeInfo(Op).@"enum".field_names) |name| {
        const c = @field(Op, name).class();
        var found = false;
        for (classes) |x| found = found or std.mem.eql(u8, x, c);
        try std.testing.expect(found);
    }
    try std.testing.expectEqual(@as(u64, 12_619_776), expert_bytes); // PLAN.md: one routed expert is 12.62 MB
}
