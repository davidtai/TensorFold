//! Gate 5's records of the EXL3 routed-expert group (quant/exl3.hip, image "exl3"): threads, the most dynamic LDS each
//! launch asks for (launches/exl3.zig), and what hipcc made of each kernel on every RDNA chip we serve.
const limits = @import("limits.zig");
const Record = @import("hip_records.zig").Record;

/// The GEMV's dynamic LDS for `mr` rows a pass at its largest k range (gemv_max_tiles = 128 k tiles = 2,048 halves):
/// mr x 2,048 x 2 B of staged rows plus 8 waves x mr x 32 fp32 sums.
fn gemvLds(comptime mr: u32) u32 {
    return mr * 2048 * 2 + 8 * mr * 32 * 4;
}

fn gemv(comptime mr: u32) limits.HipLaunch {
    const name = "EXL3 routed GEMV, " ++ (switch (mr) {
        1 => "1 row",
        2 => "2 rows",
        4 => "4 rows",
        else => "8 rows",
    }) ++ " a pass";
    const symbol = switch (mr) {
        1 => "tf_exl3_gemv_1",
        2 => "tf_exl3_gemv_2",
        4 => "tf_exl3_gemv_4",
        else => "tf_exl3_gemv_8",
    };
    return .{ .name = name, .symbol = symbol, .image = "exl3", .threads = 256, .dynamic = gemvLds(mr) };
}

pub const records = [_]Record{
    .{ .launch = .{ .name = "EXL3 trellis decode tf_exl3_unpack", .symbol = "tf_exl3_unpack", .image = "exl3", .threads = 32 }, .usage = .{
        .{ .vgprs = 26, .sgprs = 14, .lds = 0, .private = 0, .bound = 32, .wave = 32 },
        .{ .vgprs = 29, .sgprs = 18, .lds = 0, .private = 0, .bound = 32, .wave = 32 },
        .{ .vgprs = 29, .sgprs = 14, .lds = 0, .private = 0, .bound = 32, .wave = 32 },
        .{ .vgprs = 29, .sgprs = 14, .lds = 0, .private = 0, .bound = 32, .wave = 32 },
    } },
    .{ .launch = .{ .name = "EXL3 input rotation tf_exl3_rotate", .symbol = "tf_exl3_rotate", .image = "exl3", .threads = 32 }, .usage = .{
        .{ .vgprs = 14, .sgprs = 26, .lds = 0, .private = 0, .bound = 32, .wave = 32 },
        .{ .vgprs = 14, .sgprs = 26, .lds = 0, .private = 0, .bound = 32, .wave = 32 },
        .{ .vgprs = 14, .sgprs = 26, .lds = 0, .private = 0, .bound = 32, .wave = 32 },
        .{ .vgprs = 13, .sgprs = 26, .lds = 0, .private = 0, .bound = 32, .wave = 32 },
    } },
    .{ .launch = gemv(1), .usage = .{
        .{ .vgprs = 47, .sgprs = 46, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 48, .sgprs = 46, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 46, .sgprs = 44, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 47, .sgprs = 49, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
    } },
    .{ .launch = gemv(2), .usage = .{
        .{ .vgprs = 49, .sgprs = 43, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 50, .sgprs = 42, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 48, .sgprs = 42, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 49, .sgprs = 49, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
    } },
    .{ .launch = gemv(4), .usage = .{
        .{ .vgprs = 61, .sgprs = 43, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 62, .sgprs = 42, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 60, .sgprs = 43, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 61, .sgprs = 50, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
    } },
    .{ .launch = gemv(8), .usage = .{
        .{ .vgprs = 84, .sgprs = 44, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 84, .sgprs = 44, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 82, .sgprs = 46, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 82, .sgprs = 53, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
    } },
    .{ .launch = .{ .name = "EXL3 routed prompt GEMM (WMMA)", .symbol = "tf_exl3_gemm", .image = "exl3", .threads = 256, .served = limits.gfx11_only }, .usage = .{
        .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 181, .sgprs = 32, .lds = 30720, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 180, .sgprs = 32, .lds = 30720, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
    } },
    .{ .launch = .{ .name = "EXL3 routed prompt down GEMM, epilogue fused (WMMA)", .symbol = "tf_exl3_gemm_down", .image = "exl3", .threads = 256, .served = limits.gfx11_only }, .usage = .{
        .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 181, .sgprs = 38, .lds = 30720, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 180, .sgprs = 38, .lds = 30720, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
    } },
    .{ .launch = .{ .name = "EXL3 gate/up epilogue", .symbol = "tf_exl3_gateup_epilogue", .image = "exl3", .threads = 256 }, .usage = .{
        .{ .vgprs = 45, .sgprs = 28, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 44, .sgprs = 26, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 44, .sgprs = 26, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 44, .sgprs = 30, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
    } },
    .{ .launch = .{ .name = "EXL3 down epilogue", .symbol = "tf_exl3_down_epilogue", .image = "exl3", .threads = 256 }, .usage = .{
        .{ .vgprs = 18, .sgprs = 25, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 18, .sgprs = 25, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 18, .sgprs = 25, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 18, .sgprs = 28, .lds = 0, .private = 0, .bound = 256, .wave = 32 },
    } },
};
