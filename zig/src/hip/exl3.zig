//! The EXL3 routed experts over the MoE plan (GLM-5.3-Flash's 4-bit mcg trellis experts): rotate, gate and up, the
//! SwiGLU, down; each routed pair's fp32 output in y [pair][dims], the shared expert's pairs left to their own kernel.
//! The arithmetic is zig/src/families/glm/cuda_exl3.zig's (zig/kernels/cuda/exl3_experts.cu), checked against float64
//! in kernel_tests/exl3.zig.

const std = @import("std");
const ops = @import("ops/ops.zig");
const caps = @import("caps.zig");
const kernels = @import("kernels.zig");
const launch = @import("launches/exl3.zig");

const Ops = ops.Ops;
const Error = ops.Error;

/// One layer's EXL3 experts: per expert a trellis pointer (u64 [count], int16 [K/16][N/16][64]) and fp16 scales.
pub const Experts = struct {
    gate: u64,
    up: u64,
    down: u64,
    suh_g: u64, // fp16 [count][dims]
    suh_u: u64,
    svh_g: u64, // fp16 [count][width]
    svh_u: u64,
    suh_d: u64, // fp16 [count][width]
    svh_d: u64, // fp16 [count][dims]
    count: usize, // routed experts; the shared expert's id is `count`
    dims: usize,
    width: usize,
    /// Every expert's gate and up share one input rotation (the loader checks the bytes): a row rotates once.
    shared_input: bool,
    /// The SwiGLU clamp (0: none).
    limit: f32 = 0,
};

/// The rows' routing: `pick` int32 [rows][slots], and tf_moe_route's plan of it (or a one-row pick's own).
pub const Plan = struct {
    pick: u64,
    items: u64,
    /// Item slots to launch over; dead ones have no members.
    count: usize,
    members: u64,
    /// The most pairs an item holds (the plan's tile, or fewer when the rows are fewer).
    tile: usize,
};

/// Whether this build's GPU takes the WMMA prompt GEMM (gfx11); elsewhere a prompt takes the GEMV in passes.
pub fn matrix() bool {
    const c = caps.Caps.of(kernels.arch) orelse return false;
    return c.matrix == .wmma11;
}

/// The GEMV splits of K (gate/up, down) for a decode; a prompt on the GEMM takes no split.
fn splits(e: Experts, gemm: bool) Error![2]usize {
    if (gemm) return .{ 1, 1 };
    return .{ try launch.split(e.dims, e.width, true), try launch.split(e.width, e.dims, false) };
}

/// The routed slots of `rows` bf16 rows (x, x_stride apart) into y (fp32 [rows * slots][dims]); scratch from o.arena.
pub fn run(o: Ops, e: Experts, x: u64, x_stride: usize, plan: Plan, rows: usize, slots: usize, prompt: bool, y: u64) Error!void {
    if (rows == 0 or slots == 0 or e.dims % 128 != 0 or e.width % 128 != 0 or x_stride < e.dims) return error.BadShape;
    const l = o.l;
    const s = o.stream;
    const gemm = prompt and matrix();
    const sk = try splits(e, gemm);
    const pairs = rows * slots;
    const shx = e.shared_input;
    const xg = try o.arena.of(u16, (if (shx) rows else pairs) * e.dims);
    const xu = if (shx) xg else try o.arena.of(u16, pairs * e.dims);
    const xd = try o.arena.of(u16, pairs * e.width);
    const z = try o.arena.of(f32, @max(2 * sk[0] * e.width, sk[1] * e.dims) * pairs);
    try launch.rotate(l, x, x_stride, plan.pick, e.suh_g, e.suh_u, xg, xu, if (shx) rows else pairs, e.dims, slots, e.count, !shx, s);
    const gu: launch.Product = .{ .x0 = xg, .x1 = xu, .x_div = if (shx) slots else 1, .tp0 = e.gate, .tp1 = e.up, .mats = 2, .items = plan.items, .count = plan.count, .members = plan.members, .z = z, .k = e.dims, .n = e.width, .pairs = pairs, .experts = e.count };
    if (gemm) try launch.gemm(l, gu, s) else try launch.gemv(l, gu, sk[0], @min(plan.tile, pairs), s);
    try launch.gateup(l, z, plan.pick, e.svh_g, e.svh_u, e.suh_d, xd, pairs, e.width, sk[0], e.count, e.limit, s);
    var dn = gu;
    dn.x0 = xd;
    dn.x1 = xd;
    dn.x_div = 1;
    dn.tp0 = e.down;
    dn.tp1 = e.down;
    dn.mats = 1;
    dn.k = e.width;
    dn.n = e.dims;
    if (gemm) try launch.gemm(l, dn, s) else try launch.gemv(l, dn, sk[1], @min(plan.tile, pairs), s);
    try launch.down(l, z, plan.pick, e.svh_d, y, pairs, e.dims, sk[1], e.count, s);
}

/// The scratch `run` takes from the arena, 256-byte aligned pieces.
pub fn scratchBytes(e: Experts, rows: usize, slots: usize, prompt: bool) Error!usize {
    const sk = try splits(e, prompt and matrix());
    const pairs = rows * slots;
    const a = struct {
        fn f(n: usize) usize {
            return std.mem.alignForward(usize, n, 256);
        }
    }.f;
    const xg = a((if (e.shared_input) rows else pairs) * e.dims * 2);
    const xu = if (e.shared_input) 0 else a(pairs * e.dims * 2);
    return xg + xu + a(pairs * e.width * 2) + a(@max(2 * sk[0] * e.width, sk[1] * e.dims) * pairs * 4);
}

test "a GLM layer's decode scratch: one rotation a row, the gate/up splits' partials" {
    const e: Experts = .{ .gate = 0, .up = 0, .down = 0, .suh_g = 0, .suh_u = 0, .svh_g = 0, .svh_u = 0, .suh_d = 0, .svh_d = 0, .count = 288, .dims = 4096, .width = 2048, .shared_input = true };
    // 9 slots: 4096 x 2 (x) + 9 x 2048 x 2 (xd) + 2 x 4 x 2048 x 9 x 4 (z)
    try std.testing.expectEqual(@as(usize, 8192 + 36864 + 589824), try scratchBytes(e, 1, 9, false));
}

test {
    _ = launch;
}
