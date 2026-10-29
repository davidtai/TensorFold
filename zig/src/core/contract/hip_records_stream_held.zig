//! Compiler-derived launch usage of the held-codes plain stream tile (D-27) on the four served RDNA targets.
const Record = @import("hip_records.zig").Record;
const held = @import("hip_records_stream.zig").held;

pub const records = [_]Record{
    .{
        .launch = held(4, 1),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = held(4, 2),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = held(4, 4),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = held(8, 2),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = held(8, 4),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
};
