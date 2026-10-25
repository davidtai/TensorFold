//! Compiler-derived launch usage for every plain and SwiGLU stream specialization over paired tables (_pk) on the
//! four served RDNA targets.
const Record = @import("hip_records.zig").Record;
const tilePk = @import("hip_records_stream.zig").tilePk;

pub const records = [_]Record{
    .{
        .launch = tilePk(2, 1, 2, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(2, 1, 4, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(2, 1, 8, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(2, 2, 2, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(2, 2, 4, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(2, 2, 8, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(2, 4, 2, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(2, 4, 4, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(3, 1, 2, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(3, 1, 4, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(3, 1, 8, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(3, 2, 2, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(3, 2, 4, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(3, 2, 8, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(3, 4, 2, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(3, 4, 4, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(4, 1, 2, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(4, 1, 4, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(4, 1, 8, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(4, 2, 2, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(4, 2, 4, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(4, 2, 8, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(4, 4, 2, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(4, 4, 4, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(5, 1, 2, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(5, 1, 4, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(5, 1, 8, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(5, 2, 2, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(5, 2, 4, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(5, 2, 8, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(5, 4, 2, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(5, 4, 4, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(6, 1, 2, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(6, 1, 4, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(6, 1, 8, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(6, 2, 2, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(6, 2, 4, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(6, 2, 8, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(6, 4, 2, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(6, 4, 4, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(8, 1, 2, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(8, 1, 4, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(8, 1, 8, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(8, 2, 2, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(8, 2, 4, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(8, 2, 8, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(8, 4, 2, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(8, 4, 4, false),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(2, 1, 2, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(2, 1, 4, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(2, 1, 8, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(2, 2, 2, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(2, 2, 4, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(2, 2, 8, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(2, 4, 2, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(2, 4, 4, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(3, 1, 2, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(3, 1, 4, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(3, 1, 8, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(3, 2, 2, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(3, 2, 4, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(3, 2, 8, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(3, 4, 2, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(3, 4, 4, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(4, 1, 2, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(4, 1, 4, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(4, 1, 8, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(4, 2, 2, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(4, 2, 4, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(4, 2, 8, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(4, 4, 2, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(4, 4, 4, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(5, 1, 2, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(5, 1, 4, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(5, 1, 8, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(5, 2, 2, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(5, 2, 4, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(5, 2, 8, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(5, 4, 2, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(5, 4, 4, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(6, 1, 2, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(6, 1, 4, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(6, 1, 8, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(6, 2, 2, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(6, 2, 4, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(6, 2, 8, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(6, 4, 2, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(6, 4, 4, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(8, 1, 2, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(8, 1, 4, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(8, 1, 8, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(8, 2, 2, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(8, 2, 4, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(8, 2, 8, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(8, 4, 2, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
    .{
        .launch = tilePk(8, 4, 4, true),
        .usage = .{
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
            .{ .vgprs = 0, .sgprs = 0, .lds = 0, .private = 0, .bound = 0, .wave = 0 },
        },
    },
};
