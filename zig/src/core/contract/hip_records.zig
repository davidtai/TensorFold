//! The HIP launches gate 5 checks: each one's threads and LDS, and what hipcc made of it on every RDNA chip we serve.
const std = @import("std");
const limits = @import("limits.zig");

const U = limits.HipUsage;

/// A HIP launch and what hipcc made of it for gfx1030, gfx1100, gfx1151 and gfx1201 (limits.rdna's order).
pub const Record = struct { launch: limits.HipLaunch, usage: [limits.rdna.len]U };

const rms_ns = "_ZN2tf4rocm10rms_kernelI";

/// The 4-bit stream tile with the relu^2 epilogue, `rows` rows and `cb` columns a lane: DotF16 on RDNA2, else DotBF16.
fn relu2Tile(comptime rows: u32, comptime cb: u32) limits.HipLaunch {
    const fmt = "_ZN2tf4rocm24affine_dot2_stream_relu2INS0_{s}ELi4ELi{d}ELi{d}EEEvNS0_6AffineENS0_11StreamSidesEii";
    const half = std.fmt.comptimePrint(fmt, .{ "6DotF16", rows, cb });
    const brain = std.fmt.comptimePrint(fmt, .{ "7DotBF16", rows, cb });
    return .{
        .name = std.fmt.comptimePrint("relu2 stream tile 4-bit {d}x{d}", .{ rows, cb }),
        .symbol = brain,
        .image = "affine_tiles",
        .threads = 128,
        .symbols = .{ half, brain, brain, brain },
    };
}

/// The 4-bit routed work list of 5 to 8 rows on the 4-row tile (stream_list): the SwiGLU pair at CB 2 or the plain
/// product at CB 4; DotF16 on RDNA2, else DotBF16.
fn listTile(comptime pair: bool) limits.HipLaunch {
    const fmt = "_ZN2tf4rocm23affine_dot2_stream_listINS0_{s}ELi4ELi4ELi{d}ELb{d}EEEvNS0_6AffineENS0_11StreamSidesEiii";
    const cb: u32 = if (pair) 2 else 4;
    const half = std.fmt.comptimePrint(fmt, .{ "6DotF16", cb, @intFromBool(pair) });
    const brain = std.fmt.comptimePrint(fmt, .{ "7DotBF16", cb, @intFromBool(pair) });
    return .{
        .name = std.fmt.comptimePrint("{s} stream work list 4-bit 4x{d}", .{ if (pair) "swiglu" else "plain", cb }),
        .symbol = brain,
        .image = "affine_tiles",
        .threads = 128,
        .symbols = .{ half, brain, brain, brain },
    };
}

/// HIP kernels as launched; tf-hip-kernel-contract limits DIR re-derives the usage records from the code objects.
const own_records = [_]Record{
    .{
        .launch = listTile(true),
        .usage = .{
            .{ .vgprs = 161, .sgprs = 102, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 163, .sgprs = 102, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 163, .sgprs = 102, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 163, .sgprs = 105, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = listTile(false),
        .usage = .{
            .{ .vgprs = 194, .sgprs = 107, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 220, .sgprs = 107, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 220, .sgprs = 107, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 220, .sgprs = 107, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "RMS norm tf_rms_rows", .symbol = "tf_rms_rows", .image = "ops", .threads = 256 },
        .usage = .{
            .{ .vgprs = 58, .sgprs = 45, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 43, .sgprs = 45, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 42, .sgprs = 45, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 42, .sgprs = 45, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "RMS norm rms_kernel<__half>",
            .symbol = rms_ns ++ "6__halfEEvPKT_PKfPS3_if",
            .image = "act",
            .threads = 256,
        },
        .usage = .{
            .{ .vgprs = 11, .sgprs = 18, .lds = 32772, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 11, .sgprs = 18, .lds = 32772, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 10, .sgprs = 18, .lds = 32772, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 10, .sgprs = 18, .lds = 32772, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "RMS norm rms_kernel<hip_bfloat16>",
            .symbol = rms_ns ++ "12hip_bfloat16EEvPKT_PKfPS3_if",
            .image = "act",
            .threads = 256,
        },
        .usage = .{
            .{ .vgprs = 11, .sgprs = 18, .lds = 32772, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 11, .sgprs = 18, .lds = 32772, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 10, .sgprs = 18, .lds = 32772, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 10, .sgprs = 18, .lds = 32772, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "mamba tf_mamba_conv", .symbol = "tf_mamba_conv", .image = "mamba", .threads = 256 },
        .usage = .{
            .{ .vgprs = 27, .sgprs = 26, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 26, .sgprs = 26, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 26, .sgprs = 26, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 26, .sgprs = 26, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "mamba tf_mamba_dt", .symbol = "tf_mamba_dt", .image = "mamba", .threads = 256 },
        .usage = .{
            .{ .vgprs = 16, .sgprs = 25, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 16, .sgprs = 25, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 16, .sgprs = 25, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 16, .sgprs = 27, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "mamba tf_mamba_scan", .symbol = "tf_mamba_scan", .image = "mamba", .threads = 128 },
        .usage = .{
            .{ .vgprs = 127, .sgprs = 41, .lds = 20544, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 126, .sgprs = 41, .lds = 20544, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 125, .sgprs = 42, .lds = 20544, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 121, .sgprs = 42, .lds = 20544, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "mamba tf_mamba_gnorm", .symbol = "tf_mamba_gnorm", .image = "mamba", .threads = 128 },
        .usage = .{
            .{ .vgprs = 23, .sgprs = 23, .lds = 16, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 23, .sgprs = 24, .lds = 16, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 23, .sgprs = 23, .lds = 16, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 23, .sgprs = 24, .lds = 16, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "nemotron tf_nemo_select", .symbol = "tf_nemo_select", .image = "ops", .threads = 32 },
        .usage = .{
            .{ .vgprs = 27, .sgprs = 26, .lds = 8320, .private = 0, .bound = 32, .wave = 32 },
            .{ .vgprs = 27, .sgprs = 26, .lds = 8320, .private = 0, .bound = 32, .wave = 32 },
            .{ .vgprs = 27, .sgprs = 26, .lds = 8320, .private = 0, .bound = 32, .wave = 32 },
            .{ .vgprs = 27, .sgprs = 26, .lds = 8320, .private = 0, .bound = 32, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "nemotron tf_nemo_select_rank",
            .symbol = "tf_nemo_select_rank",
            .image = "ops",
            .threads = 512,
        },
        .usage = .{
            .{ .vgprs = 27, .sgprs = 30, .lds = 8320, .private = 0, .bound = 512, .wave = 32 },
            .{ .vgprs = 27, .sgprs = 30, .lds = 8320, .private = 0, .bound = 512, .wave = 32 },
            .{ .vgprs = 27, .sgprs = 30, .lds = 8320, .private = 0, .bound = 512, .wave = 32 },
            .{ .vgprs = 27, .sgprs = 29, .lds = 8320, .private = 0, .bound = 512, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "nemotron tf_nemo_relu2", .symbol = "tf_nemo_relu2", .image = "ops", .threads = 256 },
        .usage = .{
            .{ .vgprs = 8, .sgprs = 9, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 8, .sgprs = 18, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 8, .sgprs = 10, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 8, .sgprs = 10, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "nemotron tf_nemo_localize",
            .symbol = "tf_nemo_localize",
            .image = "ops",
            .threads = 256,
        },
        .usage = .{
            .{ .vgprs = 4, .sgprs = 10, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 4, .sgprs = 18, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 4, .sgprs = 10, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 4, .sgprs = 9, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "nemotron tf_nemo_combine", .symbol = "tf_nemo_combine", .image = "ops", .threads = 256 },
        .usage = .{
            .{ .vgprs = 9, .sgprs = 20, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 10, .sgprs = 20, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 9, .sgprs = 20, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 9, .sgprs = 20, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "planned walk tf_plan_score_f16",
            .symbol = "tf_plan_score_f16",
            .image = "plan",
            .threads = 128,
            .dynamic = 16 * 128 * 4,
        },
        .usage = .{
            .{ .vgprs = 17, .sgprs = 34, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 17, .sgprs = 33, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 17, .sgprs = 34, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 17, .sgprs = 29, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "planned walk tf_plan_score_bf16",
            .symbol = "tf_plan_score_bf16",
            .image = "plan",
            .threads = 128,
            .dynamic = 16 * 128 * 4,
        },
        .usage = .{
            .{ .vgprs = 17, .sgprs = 34, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 17, .sgprs = 33, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 17, .sgprs = 34, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 17, .sgprs = 29, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "planned walk tf_softmax_stats",
            .symbol = "tf_softmax_stats",
            .image = "attention",
            .threads = 256,
        },
        .usage = .{
            .{ .vgprs = 9, .sgprs = 16, .lds = 1024, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 9, .sgprs = 18, .lds = 1024, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 9, .sgprs = 16, .lds = 1024, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 9, .sgprs = 18, .lds = 1024, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "planned walk tf_plan_apply_f16",
            .symbol = "tf_plan_apply_f16",
            .image = "plan",
            .threads = 256,
        },
        .usage = .{
            .{ .vgprs = 57, .sgprs = 44, .lds = 8192, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 57, .sgprs = 44, .lds = 8192, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 57, .sgprs = 44, .lds = 8192, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 57, .sgprs = 44, .lds = 8192, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "planned walk tf_plan_apply_bf16",
            .symbol = "tf_plan_apply_bf16",
            .image = "plan",
            .threads = 256,
        },
        .usage = .{
            .{ .vgprs = 57, .sgprs = 44, .lds = 8192, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 57, .sgprs = 44, .lds = 8192, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 57, .sgprs = 44, .lds = 8192, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 57, .sgprs = 44, .lds = 8192, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "planned walk tf_sum_partials",
            .symbol = "tf_sum_partials",
            .image = "attention",
            .threads = 256,
        },
        .usage = .{
            .{ .vgprs = 5, .sgprs = 14, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 5, .sgprs = 18, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 5, .sgprs = 14, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 5, .sgprs = 16, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "add rms tf_add_rms_f16", .symbol = "tf_add_rms_f16", .image = "act", .threads = 1024 },
        .usage = .{
            .{ .vgprs = 43, .sgprs = 40, .lds = 32772, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 45, .sgprs = 40, .lds = 32772, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 45, .sgprs = 40, .lds = 32772, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 44, .sgprs = 40, .lds = 32772, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "add rms tf_add_rms_bf16", .symbol = "tf_add_rms_bf16", .image = "act", .threads = 1024 },
        .usage = .{
            .{ .vgprs = 44, .sgprs = 40, .lds = 32772, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 45, .sgprs = 40, .lds = 32772, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 45, .sgprs = 40, .lds = 32772, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 44, .sgprs = 40, .lds = 32772, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "mamba tf_mamba_conv_window",
            .symbol = "tf_mamba_conv_window",
            .image = "mamba",
            .threads = 256,
        },
        .usage = .{
            .{ .vgprs = 32, .sgprs = 42, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 38, .sgprs = 42, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 37, .sgprs = 42, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 37, .sgprs = 44, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "mamba tf_mamba_scan_snap",
            .symbol = "tf_mamba_scan_snap",
            .image = "mamba",
            .threads = 128,
        },
        .usage = .{
            .{ .vgprs = 137, .sgprs = 50, .lds = 20544, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 136, .sgprs = 50, .lds = 20544, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 135, .sgprs = 51, .lds = 20544, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 131, .sgprs = 52, .lds = 20544, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = relu2Tile(1, 2),
        .usage = .{
            .{ .vgprs = 79, .sgprs = 54, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 83, .sgprs = 54, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 83, .sgprs = 54, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 83, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = relu2Tile(1, 4),
        .usage = .{
            .{ .vgprs = 112, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 121, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 121, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 121, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = relu2Tile(1, 8),
        .usage = .{
            .{ .vgprs = 187, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 186, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 186, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 186, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = relu2Tile(2, 2),
        .usage = .{
            .{ .vgprs = 93, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 110, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 110, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 110, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = relu2Tile(2, 4),
        .usage = .{
            .{ .vgprs = 112, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 135, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 135, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 135, .sgprs = 65, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = relu2Tile(2, 8),
        .usage = .{
            .{ .vgprs = 164, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 248, .sgprs = 90, .lds = 0, .private = 304, .bound = 128, .wave = 32 },
            .{ .vgprs = 248, .sgprs = 90, .lds = 0, .private = 304, .bound = 128, .wave = 32 },
            .{ .vgprs = 248, .sgprs = 86, .lds = 0, .private = 304, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = relu2Tile(4, 2),
        .usage = .{
            .{ .vgprs = 146, .sgprs = 67, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 148, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 148, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 148, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = relu2Tile(4, 4),
        .usage = .{
            .{ .vgprs = 176, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 181, .sgprs = 71, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 181, .sgprs = 71, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 181, .sgprs = 73, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "planned walk tf_plan_kv_write_act",
            .symbol = "tf_plan_kv_write_act",
            .image = "plan",
            .threads = 256,
        },
        .usage = .{
            .{ .vgprs = 20, .sgprs = 26, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 20, .sgprs = 26, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 20, .sgprs = 26, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 20, .sgprs = 28, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "planned walk tf_plan_score_act_f16",
            .symbol = "tf_plan_score_act_f16",
            .image = "plan",
            .threads = 128,
            .dynamic = 16 * 128 * 4,
        },
        .usage = .{
            .{ .vgprs = 17, .sgprs = 32, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 17, .sgprs = 32, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 17, .sgprs = 32, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 17, .sgprs = 28, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "planned walk tf_plan_score_act_bf16",
            .symbol = "tf_plan_score_act_bf16",
            .image = "plan",
            .threads = 128,
            .dynamic = 16 * 128 * 4,
        },
        .usage = .{
            .{ .vgprs = 17, .sgprs = 32, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 17, .sgprs = 32, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 17, .sgprs = 32, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 17, .sgprs = 28, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "planned walk tf_sum_partials_act",
            .symbol = "tf_sum_partials_act",
            .image = "plan",
            .threads = 256,
        },
        .usage = .{
            .{ .vgprs = 7, .sgprs = 16, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 6, .sgprs = 18, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 6, .sgprs = 16, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 6, .sgprs = 18, .lds = 0, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "argmax tf_argmax_part", .symbol = "tf_argmax_part", .image = "ops", .threads = 256 },
        .usage = .{
            .{ .vgprs = 8, .sgprs = 17, .lds = 2048, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 8, .sgprs = 18, .lds = 2048, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 8, .sgprs = 17, .lds = 2048, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 8, .sgprs = 16, .lds = 2048, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "top-k tf_topk_part", .symbol = "tf_topk_part", .image = "ops", .threads = 1024 },
        .usage = .{
            .{ .vgprs = 33, .sgprs = 36, .lds = 1296, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 78, .sgprs = 36, .lds = 1296, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 78, .sgprs = 36, .lds = 1296, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 78, .sgprs = 36, .lds = 1296, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "top-k tf_topk_join", .symbol = "tf_topk_join", .image = "ops", .threads = 1024 },
        .usage = .{
            .{ .vgprs = 16, .sgprs = 36, .lds = 33808, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 16, .sgprs = 36, .lds = 33808, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 16, .sgprs = 36, .lds = 33808, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 16, .sgprs = 37, .lds = 33808, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "argmax tf_argmax_join", .symbol = "tf_argmax_join", .image = "ops", .threads = 256 },
        .usage = .{
            .{ .vgprs = 10, .sgprs = 21, .lds = 2048, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 10, .sgprs = 21, .lds = 2048, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 10, .sgprs = 21, .lds = 2048, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 10, .sgprs = 20, .lds = 2048, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "planned walk tf_plan_attend_f16",
            .symbol = "tf_plan_attend_f16",
            .image = "plan",
            .threads = 1024,
        },
        .usage = .{
            .{ .vgprs = 38, .sgprs = 32, .lds = 50688, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 38, .sgprs = 32, .lds = 50688, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 38, .sgprs = 32, .lds = 50688, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 38, .sgprs = 32, .lds = 50688, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "planned walk tf_plan_attend_bf16",
            .symbol = "tf_plan_attend_bf16",
            .image = "plan",
            .threads = 1024,
        },
        .usage = .{
            .{ .vgprs = 36, .sgprs = 32, .lds = 50688, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 37, .sgprs = 32, .lds = 50688, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 36, .sgprs = 32, .lds = 50688, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 36, .sgprs = 32, .lds = 50688, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "MoE router (decode, fp32 rows) tf_router_decode_f16", .symbol = "tf_router_decode_f16", .image = "decode", .threads = 128 },
        .usage = .{
            .{ .vgprs = 106, .sgprs = 75, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 90, .sgprs = 75, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 90, .sgprs = 75, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 90, .sgprs = 76, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "MoE router (decode, fp32 rows) tf_router_decode_bf16", .symbol = "tf_router_decode_bf16", .image = "decode", .threads = 128 },
        .usage = .{
            .{ .vgprs = 106, .sgprs = 75, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 91, .sgprs = 75, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 91, .sgprs = 75, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 91, .sgprs = 76, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "MoE router and pick in one launch (decode, bf16 rows) tf_router_select_f16", .symbol = "tf_router_select_f16", .image = "decode", .threads = 256, .dynamic = 5 * 2560 * 2 },
        .usage = .{
            .{ .vgprs = 135, .sgprs = 86, .lds = 5520, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 135, .sgprs = 82, .lds = 5520, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 135, .sgprs = 82, .lds = 5520, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 135, .sgprs = 83, .lds = 5520, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "MoE router and pick in one launch (decode, bf16 rows) tf_router_select_bf16", .symbol = "tf_router_select_bf16", .image = "decode", .threads = 256, .dynamic = 5 * 2560 * 2 },
        .usage = .{
            .{ .vgprs = 135, .sgprs = 86, .lds = 5520, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 135, .sgprs = 82, .lds = 5520, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 135, .sgprs = 82, .lds = 5520, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 135, .sgprs = 83, .lds = 5520, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "prompt expert gate+up tile 4-bit", .symbol = "_ZN2tf4rocm18affine_wmma_moe_guILi4EEEvNS0_6AffineEf", .image = "affine_tiles", .threads = 256, .served = limits.gfx11_only },
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 226, .sgprs = 34, .lds = 36352, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 225, .sgprs = 36, .lds = 36352, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "prompt expert down tile 4-bit", .symbol = "_ZN2tf4rocm20affine_wmma_moe_downILi4EEEvNS0_6AffineE", .image = "affine_tiles", .threads = 256, .served = limits.gfx11_only },
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 189, .sgprs = 33, .lds = 32256, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 189, .sgprs = 34, .lds = 32256, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "prompt dense tile 4-bit K 320", .symbol = "_ZN2tf4rocm17affine_wmma_denseILi320EEEvNS0_6AffineE", .image = "affine_tiles", .threads = 256, .served = limits.gfx11_only },
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 208, .sgprs = 22, .lds = 39936, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 208, .sgprs = 22, .lds = 39936, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "prompt dense tile 4-bit K 320 with the HC mix", .symbol = "tf_affine_dense_hcmix", .image = "affine_tiles", .threads = 256, .served = limits.gfx11_only },
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 215, .sgprs = 23, .lds = 39936, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 215, .sgprs = 23, .lds = 39936, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "prompt dense tile 4-bit K 2560", .symbol = "_ZN2tf4rocm17affine_wmma_denseILi2560EEEvNS0_6AffineE", .image = "affine_tiles", .threads = 256, .served = limits.gfx11_only },
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 208, .sgprs = 22, .lds = 39936, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 208, .sgprs = 22, .lds = 39936, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "prompt dense tile 4-bit K 6144", .symbol = "_ZN2tf4rocm17affine_wmma_denseILi6144EEEvNS0_6AffineE", .image = "affine_tiles", .threads = 256, .served = limits.gfx11_only },
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 208, .sgprs = 22, .lds = 39936, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 208, .sgprs = 22, .lds = 39936, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "prompt dense tile 4-bit K 10240", .symbol = "_ZN2tf4rocm17affine_wmma_denseILi10240EEEvNS0_6AffineE", .image = "affine_tiles", .threads = 256, .served = limits.gfx11_only },
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 246, .sgprs = 22, .lds = 39936, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 246, .sgprs = 22, .lds = 39936, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "Flash Next QSA decode prep", .symbol = "tf_fn_qsa_prep", .image = "flashnext", .threads = 256 },
        .usage = .{
            .{ .vgprs = 36, .sgprs = 50, .lds = 17440, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 43, .sgprs = 50, .lds = 17440, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 34, .sgprs = 52, .lds = 17440, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 34, .sgprs = 51, .lds = 17440, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "prompt dense inexact tile 4-bit K 2560", .symbol = "_ZN2tf4rocm25affine_wmma_dense_inexactILi2560EEEvNS0_6AffineE", .image = "affine_tiles", .threads = 256, .served = limits.gfx11_only },
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 252, .sgprs = 20, .lds = 55296, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 252, .sgprs = 20, .lds = 55296, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "prompt dense inexact tile 4-bit K 6144", .symbol = "_ZN2tf4rocm25affine_wmma_dense_inexactILi6144EEEvNS0_6AffineE", .image = "affine_tiles", .threads = 256, .served = limits.gfx11_only },
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 252, .sgprs = 20, .lds = 55296, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 252, .sgprs = 20, .lds = 55296, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "prompt expert gate+up inexact tile 4-bit", .symbol = "_ZN2tf4rocm26affine_wmma_moe_gu_inexactILi4EEEvNS0_6AffineEf", .image = "affine_tiles", .threads = 256, .served = limits.gfx11_only },
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 219, .sgprs = 31, .lds = 36864, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 218, .sgprs = 31, .lds = 36864, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "prompt expert down inexact tile 4-bit", .symbol = "_ZN2tf4rocm28affine_wmma_moe_down_inexactILi4EEEvNS0_6AffineE", .image = "affine_tiles", .threads = 256, .served = limits.gfx11_only },
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 205, .sgprs = 31, .lds = 36864, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 204, .sgprs = 31, .lds = 36864, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "Flash Next prompt sparse attention tile", .symbol = "tf_fn_sparse_prompt", .image = "flashnext", .threads = 128, .served = limits.gfx11_only },
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 251, .sgprs = 42, .lds = 31424, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 251, .sgprs = 31, .lds = 31424, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "Flash Next prompt sparse attention tile, p in bf16 once", .symbol = "tf_fn_sparse_prompt_hi", .image = "flashnext", .threads = 128, .served = limits.gfx11_only },
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 204, .sgprs = 42, .lds = 31424, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 204, .sgprs = 31, .lds = 31424, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "Flash Next prompt sparse attention tile, split", .symbol = "tf_fn_sparse_prompt_split", .image = "flashnext", .threads = 128, .served = limits.gfx11_only },
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 248, .sgprs = 42, .lds = 31424, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 248, .sgprs = 33, .lds = 31424, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "Flash Next decode sparse attention tile parts", .symbol = "tf_fn_sparse_tile_parts", .image = "flashnext", .threads = 128, .served = limits.gfx11_only },
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 247, .sgprs = 48, .lds = 31424, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 247, .sgprs = 48, .lds = 31424, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = .{ .name = "Flash Next decode sparse attention tile parts, split", .symbol = "tf_fn_sparse_tile_parts_split", .image = "flashnext", .threads = 128, .served = limits.gfx11_only },
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 246, .sgprs = 48, .lds = 31424, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 246, .sgprs = 48, .lds = 31424, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
} ++ @import("hip_records_pair.zig").records;

pub const records = own_records ++ @import("hip_records_stream.zig").records;

/// Gate 5's planted HIP controls: a launch past its __launch_bounds__, and 72 KiB of LDS where a workgroup has 64.
pub const controls = [_]Record{
    .{
        .launch = .{
            .name = "control: past its bounds",
            .symbol = "tf_ctl_bounded",
            .image = "contract_controls",
            .threads = 512,
        },
        .usage = .{
            .{ .vgprs = 4, .sgprs = 10, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 4, .sgprs = 18, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 4, .sgprs = 10, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 4, .sgprs = 7, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    .{
        .launch = .{
            .name = "control: 72 KiB of LDS",
            .symbol = "tf_ctl_big_lds",
            .image = "contract_controls",
            .threads = 256,
            .dynamic = 32 * 1024,
        },
        .usage = .{
            .{ .vgprs = 6, .sgprs = 9, .lds = 40960, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 6, .sgprs = 8, .lds = 40960, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 6, .sgprs = 8, .lds = 40960, .private = 0, .bound = 1024, .wave = 32 },
            .{ .vgprs = 6, .sgprs = 8, .lds = 40960, .private = 0, .bound = 1024, .wave = 32 },
        },
    },
};

/// Every HIP record's problems on every chip we serve, printed; returns how many (a missing record is one).
pub fn check(a: std.mem.Allocator, list: []const Record, quiet: bool) !usize {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    var n: usize = 0;
    for (list) |r| for (limits.rdna, r.usage, r.launch.served) |t, u, served| {
        if (!served) {
            if (u.vgprs != 0) {
                try out.print(a, "{s} on {s}: usage recorded on a chip that does not launch it\n", .{ r.launch.name, t.name });
                n += 1;
            }
        } else if (u.vgprs == 0) {
            try out.print(a, "{s} on {s}: no compiled usage recorded\n", .{ r.launch.name, t.name });
            n += 1;
        } else n += try limits.hip(r.launch, u, t, &out, a);
    };
    if (!quiet and out.items.len > 0) std.debug.print("{s}", .{out.items});
    return n;
}

/// A planted control needs compiled usage and a real limit violation on every chip; returns missed chips.
pub fn controlProblems(a: std.mem.Allocator, r: Record, quiet: bool) !usize {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    var n: usize = 0;
    for (limits.rdna, r.usage) |chip, usage| {
        if (usage.vgprs == 0) {
            n += 1;
            try out.print(a, "{s} on {s}: no compiled usage for control\n", .{ r.launch.name, chip.name });
        } else if (try limits.hip(r.launch, usage, chip, &out, a) == 0) {
            n += 1;
            try out.print(a, "{s} on {s}: planted control fits\n", .{ r.launch.name, chip.name });
        }
    }
    if (!quiet and n != 0) std.debug.print("{s}", .{out.items});
    return n;
}

test "every registered HIP launch fits every RDNA chip we serve, and each planted control breaks one" {
    const a = std.testing.allocator;
    try std.testing.expectEqual(@as(usize, 0), try check(a, &records, false));
    for (controls) |r| try std.testing.expectEqual(@as(usize, 0), try controlProblems(a, r, true));
    for (controls) |r| {
        var good = r;
        for (&good.usage) |*u| u.* = .{ .vgprs = 16 };
        good.launch.threads = 256;
        good.launch.dynamic = 0;
        try std.testing.expectEqual(@as(usize, 0), try check(a, &.{good}, true)); // red only for what it plants
    }
}

test "a control cannot fit or lack compiled usage on any one chip" {
    const a = std.testing.allocator;
    for (controls) |r| {
        for (limits.rdna, 0..) |_, i| {
            var fits = r;
            fits.usage[i] = .{ .vgprs = 16, .bound = fits.launch.threads };
            try std.testing.expectEqual(@as(usize, 1), try controlProblems(a, fits, true));
            var missing = r;
            missing.usage[i].vgprs = 0;
            try std.testing.expectEqual(@as(usize, 1), try controlProblems(a, missing, true));
        }
    }
}

test "a chip a launch is not served on records no usage, and recorded usage there is a problem" {
    const a = std.testing.allocator;
    var r: Record = .{ .launch = .{ .name = "gfx11 tile", .symbol = "k", .image = "affine_tiles", .threads = 256, .served = limits.gfx11_only }, .usage = undefined };
    for (limits.rdna, &r.usage) |t, *u| u.* = if (std.mem.startsWith(u8, t.name, "gfx11")) .{ .vgprs = 64, .bound = 256 } else .{ .vgprs = 0 };
    try std.testing.expectEqual(@as(usize, 0), try check(a, &.{r}, true));
    r.usage[0] = .{ .vgprs = 64, .bound = 256 };
    try std.testing.expectEqual(@as(usize, 1), try check(a, &.{r}, true));
    // served everywhere: each chip without usage is a problem (gfx1030 and gfx1201 here)
    r.launch.served = @splat(true);
    r.usage[0] = .{ .vgprs = 0 };
    try std.testing.expectEqual(@as(usize, 2), try check(a, &.{r}, true));
}
