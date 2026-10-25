//! Compiler-derived launch usage for every plain and SwiGLU stream specialization over paired tables (_pk) on the
//! four served RDNA targets.
const Record = @import("hip_records.zig").Record;
const tilePk = @import("hip_records_stream.zig").tilePk;

pub const records = [_]Record{
    .{
        .launch = tilePk(2, 1, 2, false),
        .usage = .{
            .{ .vgprs = 57, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 57, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 57, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 57, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(2, 1, 4, false),
        .usage = .{
            .{ .vgprs = 101, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 94, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 94, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 94, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(2, 1, 8, false),
        .usage = .{
            .{ .vgprs = 169, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 158, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 158, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 158, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(2, 2, 2, false),
        .usage = .{
            .{ .vgprs = 79, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 68, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 68, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 68, .sgprs = 65, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(2, 2, 4, false),
        .usage = .{
            .{ .vgprs = 105, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 106, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 106, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 106, .sgprs = 67, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(2, 2, 8, false),
        .usage = .{
            .{ .vgprs = 157, .sgprs = 67, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 158, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 158, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 158, .sgprs = 71, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(2, 4, 2, false),
        .usage = .{
            .{ .vgprs = 117, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 114, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 114, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 114, .sgprs = 73, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(2, 4, 4, false),
        .usage = .{
            .{ .vgprs = 147, .sgprs = 71, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 150, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 150, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 150, .sgprs = 75, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(3, 1, 2, false),
        .usage = .{
            .{ .vgprs = 128, .sgprs = 50, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 109, .sgprs = 51, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 109, .sgprs = 51, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 109, .sgprs = 51, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(3, 1, 4, false),
        .usage = .{
            .{ .vgprs = 92, .sgprs = 52, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 89, .sgprs = 53, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 89, .sgprs = 53, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 89, .sgprs = 53, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(3, 1, 8, false),
        .usage = .{
            .{ .vgprs = 170, .sgprs = 55, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 153, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 153, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 153, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(3, 2, 2, false),
        .usage = .{
            .{ .vgprs = 99, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 119, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 119, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 119, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(3, 2, 4, false),
        .usage = .{
            .{ .vgprs = 88, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 90, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 90, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 90, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(3, 2, 8, false),
        .usage = .{
            .{ .vgprs = 137, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 138, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 138, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 138, .sgprs = 65, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(3, 4, 2, false),
        .usage = .{
            .{ .vgprs = 89, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 107, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 107, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 107, .sgprs = 67, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(3, 4, 4, false),
        .usage = .{
            .{ .vgprs = 131, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 134, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 134, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 134, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(4, 1, 2, false),
        .usage = .{
            .{ .vgprs = 69, .sgprs = 55, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 65, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 65, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 65, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(4, 1, 4, false),
        .usage = .{
            .{ .vgprs = 109, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 106, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 106, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 106, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(4, 1, 8, false),
        .usage = .{
            .{ .vgprs = 181, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 172, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 172, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 172, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(4, 2, 2, false),
        .usage = .{
            .{ .vgprs = 77, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 72, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 72, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 72, .sgprs = 65, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(4, 2, 4, false),
        .usage = .{
            .{ .vgprs = 101, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 106, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 106, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 106, .sgprs = 67, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(4, 2, 8, false),
        .usage = .{
            .{ .vgprs = 173, .sgprs = 67, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 174, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 174, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 174, .sgprs = 71, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(4, 4, 2, false),
        .usage = .{
            .{ .vgprs = 117, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 118, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 118, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 118, .sgprs = 73, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(4, 4, 4, false),
        .usage = .{
            .{ .vgprs = 155, .sgprs = 71, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 174, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 174, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 174, .sgprs = 75, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(5, 1, 2, false),
        .usage = .{
            .{ .vgprs = 130, .sgprs = 50, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 117, .sgprs = 51, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 117, .sgprs = 51, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 117, .sgprs = 51, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(5, 1, 4, false),
        .usage = .{
            .{ .vgprs = 108, .sgprs = 52, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 105, .sgprs = 53, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 105, .sgprs = 53, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 105, .sgprs = 53, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(5, 1, 8, false),
        .usage = .{
            .{ .vgprs = 201, .sgprs = 55, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 185, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 185, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 185, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(5, 2, 2, false),
        .usage = .{
            .{ .vgprs = 77, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 99, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 99, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 99, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(5, 2, 4, false),
        .usage = .{
            .{ .vgprs = 88, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 98, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 98, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 98, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(5, 2, 8, false),
        .usage = .{
            .{ .vgprs = 153, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 162, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 162, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 162, .sgprs = 65, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(5, 4, 2, false),
        .usage = .{
            .{ .vgprs = 92, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 107, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 107, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 107, .sgprs = 67, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(5, 4, 4, false),
        .usage = .{
            .{ .vgprs = 153, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 142, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 142, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 142, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(6, 1, 2, false),
        .usage = .{
            .{ .vgprs = 79, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 69, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 69, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 69, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(6, 1, 4, false),
        .usage = .{
            .{ .vgprs = 124, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 121, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 121, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 121, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(6, 1, 8, false),
        .usage = .{
            .{ .vgprs = 236, .sgprs = 65, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 217, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 217, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 217, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(6, 2, 2, false),
        .usage = .{
            .{ .vgprs = 75, .sgprs = 65, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 76, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 76, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 76, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(6, 2, 4, false),
        .usage = .{
            .{ .vgprs = 109, .sgprs = 67, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 110, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 110, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 110, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(6, 2, 8, false),
        .usage = .{
            .{ .vgprs = 177, .sgprs = 71, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 191, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 191, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 191, .sgprs = 74, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(6, 4, 2, false),
        .usage = .{
            .{ .vgprs = 110, .sgprs = 73, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 116, .sgprs = 74, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 116, .sgprs = 74, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 116, .sgprs = 76, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(6, 4, 4, false),
        .usage = .{
            .{ .vgprs = 153, .sgprs = 75, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 170, .sgprs = 76, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 170, .sgprs = 76, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 170, .sgprs = 78, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(8, 1, 2, false),
        .usage = .{
            .{ .vgprs = 76, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 77, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 77, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 77, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(8, 1, 4, false),
        .usage = .{
            .{ .vgprs = 141, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 138, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 138, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 138, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(8, 1, 8, false),
        .usage = .{
            .{ .vgprs = 242, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 62, .lds = 0, .private = 156, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 62, .lds = 0, .private = 156, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 64, .lds = 0, .private = 156, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(8, 2, 2, false),
        .usage = .{
            .{ .vgprs = 85, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 79, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 79, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 79, .sgprs = 65, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(8, 2, 4, false),
        .usage = .{
            .{ .vgprs = 125, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 122, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 122, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 122, .sgprs = 67, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(8, 2, 8, false),
        .usage = .{
            .{ .vgprs = 201, .sgprs = 67, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 206, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 206, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 206, .sgprs = 71, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(8, 4, 2, false),
        .usage = .{
            .{ .vgprs = 128, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 129, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 129, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 129, .sgprs = 73, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(8, 4, 4, false),
        .usage = .{
            .{ .vgprs = 171, .sgprs = 71, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 172, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 172, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 172, .sgprs = 75, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(2, 1, 2, true),
        .usage = .{
            .{ .vgprs = 56, .sgprs = 55, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 56, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 56, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 56, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(2, 1, 4, true),
        .usage = .{
            .{ .vgprs = 96, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 92, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 92, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 92, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(2, 1, 8, true),
        .usage = .{
            .{ .vgprs = 173, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 154, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 154, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 154, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(2, 2, 2, true),
        .usage = .{
            .{ .vgprs = 79, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 67, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 67, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 67, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(2, 2, 4, true),
        .usage = .{
            .{ .vgprs = 103, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 104, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 104, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 104, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(2, 2, 8, true),
        .usage = .{
            .{ .vgprs = 153, .sgprs = 65, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 154, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 154, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 154, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(2, 4, 2, true),
        .usage = .{
            .{ .vgprs = 116, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 113, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 113, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 113, .sgprs = 74, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(2, 4, 4, true),
        .usage = .{
            .{ .vgprs = 145, .sgprs = 71, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 148, .sgprs = 74, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 148, .sgprs = 74, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 148, .sgprs = 76, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(3, 1, 2, true),
        .usage = .{
            .{ .vgprs = 127, .sgprs = 49, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 108, .sgprs = 50, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 108, .sgprs = 50, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 108, .sgprs = 50, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(3, 1, 4, true),
        .usage = .{
            .{ .vgprs = 90, .sgprs = 50, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 87, .sgprs = 51, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 87, .sgprs = 51, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 87, .sgprs = 51, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(3, 1, 8, true),
        .usage = .{
            .{ .vgprs = 174, .sgprs = 52, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 149, .sgprs = 53, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 149, .sgprs = 53, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 149, .sgprs = 53, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(3, 2, 2, true),
        .usage = .{
            .{ .vgprs = 99, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 119, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 119, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 119, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(3, 2, 4, true),
        .usage = .{
            .{ .vgprs = 86, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 88, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 88, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 88, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(3, 2, 8, true),
        .usage = .{
            .{ .vgprs = 133, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 134, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 134, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 134, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(3, 4, 2, true),
        .usage = .{
            .{ .vgprs = 89, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 107, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 107, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 107, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(3, 4, 4, true),
        .usage = .{
            .{ .vgprs = 110, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 132, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 132, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 132, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(4, 1, 2, true),
        .usage = .{
            .{ .vgprs = 59, .sgprs = 55, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 60, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 60, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 60, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(4, 1, 4, true),
        .usage = .{
            .{ .vgprs = 107, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 104, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 104, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 104, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(4, 1, 8, true),
        .usage = .{
            .{ .vgprs = 185, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 182, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 182, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 182, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(4, 2, 2, true),
        .usage = .{
            .{ .vgprs = 76, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 71, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 71, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 71, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(4, 2, 4, true),
        .usage = .{
            .{ .vgprs = 99, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 104, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 104, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 104, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(4, 2, 8, true),
        .usage = .{
            .{ .vgprs = 169, .sgprs = 65, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 170, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 170, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 170, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(4, 4, 2, true),
        .usage = .{
            .{ .vgprs = 116, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 117, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 117, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 117, .sgprs = 74, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(4, 4, 4, true),
        .usage = .{
            .{ .vgprs = 153, .sgprs = 71, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 171, .sgprs = 74, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 171, .sgprs = 74, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 171, .sgprs = 76, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(5, 1, 2, true),
        .usage = .{
            .{ .vgprs = 129, .sgprs = 49, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 116, .sgprs = 50, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 116, .sgprs = 50, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 116, .sgprs = 50, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(5, 1, 4, true),
        .usage = .{
            .{ .vgprs = 106, .sgprs = 50, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 103, .sgprs = 51, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 103, .sgprs = 51, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 103, .sgprs = 51, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(5, 1, 8, true),
        .usage = .{
            .{ .vgprs = 207, .sgprs = 52, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 181, .sgprs = 53, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 181, .sgprs = 53, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 181, .sgprs = 53, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(5, 2, 2, true),
        .usage = .{
            .{ .vgprs = 77, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 99, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 99, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 99, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(5, 2, 4, true),
        .usage = .{
            .{ .vgprs = 86, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 122, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 122, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 122, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(5, 2, 8, true),
        .usage = .{
            .{ .vgprs = 149, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 158, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 158, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 158, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(5, 4, 2, true),
        .usage = .{
            .{ .vgprs = 92, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 107, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 107, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 107, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(5, 4, 4, true),
        .usage = .{
            .{ .vgprs = 151, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 140, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 140, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 140, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(6, 1, 2, true),
        .usage = .{
            .{ .vgprs = 67, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 68, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 68, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 68, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(6, 1, 4, true),
        .usage = .{
            .{ .vgprs = 122, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 119, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 119, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 119, .sgprs = 62, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(6, 1, 8, true),
        .usage = .{
            .{ .vgprs = 240, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 213, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 213, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 213, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(6, 2, 2, true),
        .usage = .{
            .{ .vgprs = 74, .sgprs = 65, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 75, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 75, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 75, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(6, 2, 4, true),
        .usage = .{
            .{ .vgprs = 107, .sgprs = 67, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 108, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 108, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 108, .sgprs = 71, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(6, 2, 8, true),
        .usage = .{
            .{ .vgprs = 173, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 187, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 187, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 187, .sgprs = 73, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(6, 4, 2, true),
        .usage = .{
            .{ .vgprs = 109, .sgprs = 73, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 115, .sgprs = 76, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 115, .sgprs = 76, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 115, .sgprs = 77, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(6, 4, 4, true),
        .usage = .{
            .{ .vgprs = 151, .sgprs = 75, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 154, .sgprs = 78, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 154, .sgprs = 78, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 154, .sgprs = 79, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(8, 1, 2, true),
        .usage = .{
            .{ .vgprs = 75, .sgprs = 55, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 76, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 76, .sgprs = 56, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 76, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(8, 1, 4, true),
        .usage = .{
            .{ .vgprs = 139, .sgprs = 57, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 136, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 136, .sgprs = 58, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 136, .sgprs = 60, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(8, 1, 8, true),
        .usage = .{
            .{ .vgprs = 246, .sgprs = 59, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 60, .lds = 0, .private = 140, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 60, .lds = 0, .private = 140, .bound = 128, .wave = 32 },
            .{ .vgprs = 256, .sgprs = 62, .lds = 0, .private = 140, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(8, 2, 2, true),
        .usage = .{
            .{ .vgprs = 84, .sgprs = 61, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 78, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 78, .sgprs = 64, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 78, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(8, 2, 4, true),
        .usage = .{
            .{ .vgprs = 133, .sgprs = 63, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 120, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 120, .sgprs = 66, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 120, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(8, 2, 8, true),
        .usage = .{
            .{ .vgprs = 207, .sgprs = 65, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 202, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 202, .sgprs = 68, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 202, .sgprs = 70, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(8, 4, 2, true),
        .usage = .{
            .{ .vgprs = 128, .sgprs = 69, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 128, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 128, .sgprs = 72, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 128, .sgprs = 74, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
    .{
        .launch = tilePk(8, 4, 4, true),
        .usage = .{
            .{ .vgprs = 169, .sgprs = 71, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 170, .sgprs = 74, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 170, .sgprs = 74, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 170, .sgprs = 76, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        },
    },
};
