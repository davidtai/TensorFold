//! A real checkpoint's config.json against GLM-5.3-Flash's shape: TF_GLM_CONFIG names the file (skipped when unset).
//! Run on the box: TF_GLM_CONFIG=<checkpoint>/config.json zig test -lc zig/src/families/glm/config_checkpoint_test.zig
//! glm-hip-test runs `check` too (zig/src/hip/kernel_tests/glm/config.zig): a test runs only in its binary's root module.
const std = @import("std");
const config = @import("config.zig");

test "TF_GLM_CONFIG parses to the whole GLM-5.3-Flash shape: EXL3 experts, 34 KDA and 11 MLA layers, one MTP layer" {
    try check();
}

/// The checks on TF_GLM_CONFIG's config.json; error.SkipZigTest when it is unset.
pub fn check() !void {
    const path = std.c.getenv("TF_GLM_CONFIG") orelse return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const text = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, std.mem.span(path), a, .limited(1 << 24));
    const c = try config.parse(std.testing.allocator, text);
    try std.testing.expect(config.whole(c));
    try std.testing.expect(c.exl3);
    try std.testing.expectEqual(@as(u32, 128), c.head_unit);
    try std.testing.expectEqual(@as(u32, 34), c.countKind(.kda));
    try std.testing.expectEqual(@as(u32, 11), c.countKind(.mla));
    try std.testing.expectEqual(config.Kind.mla, c.kind(c.layers)); // the MTP layer
    try std.testing.expectEqual(@as(u32, 1), c.mtp);
    try std.testing.expectEqual(@as(u32, 288), c.experts);
    try std.testing.expectEqual(@as(u32, 8), c.topk);
    try std.testing.expectEqual(@as(u32, 3), c.dense_layers);
    try std.testing.expectEqual(@as(u32, 154880), c.vocab);
    try std.testing.expectEqual(@as(u32, 3), c.eos_n);
    try std.testing.expectEqual(@as(u32, 2051), c.keyWidth());
}
