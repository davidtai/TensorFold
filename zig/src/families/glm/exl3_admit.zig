//! The EXL3 checkpoints a GLM backend reads, from config.json alone: 4-bit routed experts on the "mcg" codebook.
//! Backend-neutral: CUDA (cuda_exl3.zig) and HIP (hip/native.zig) admit by it, each naming itself in the refusal.

const std = @import("std");
const modelopt = @import("core").modelopt;

/// config.json's quantization_config (top level, else text_config's), when it has one.
fn block(root: std.json.Value) ?std.json.ObjectMap {
    if (root != .object) return null;
    if (root.object.get("quantization_config")) |q| if (q == .object) return q.object;
    if (root.object.get("text_config")) |t| if (t == .object) if (t.object.get("quantization_config")) |q| if (q == .object) return q.object;
    return null;
}

/// Whether config.json names EXL3 (quant_method "exl3"): `admit` then says whether this family reads it.
pub fn isExl3(gpa: std.mem.Allocator, config_json: []const u8) bool {
    const parsed = std.json.parseFromSlice(std.json.Value, gpa, config_json, .{}) catch return false;
    defer parsed.deinit();
    const q = block(parsed.value) orelse return false;
    const m = q.get("quant_method") orelse return false;
    return m == .string and std.mem.eql(u8, m.string, "exl3");
}

/// The EXL3 checkpoints the kernels read: 4-bit routed experts on the "mcg" codebook (everything else bf16).
/// `backend` ("CUDA", "HIP") names the family that refuses.
pub fn admit(comptime backend: []const u8, gpa: std.mem.Allocator, config_json: []const u8, why: *modelopt.Why) !void {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, config_json, .{});
    defer parsed.deinit();
    const q = block(parsed.value) orelse {
        why.set("no quantization_config", .{});
        return error.UnsupportedQuant;
    };
    const bits = q.get("bits") orelse std.json.Value{ .null = {} };
    if (bits != .integer or bits.integer != 4) {
        why.set("EXL3 at {f} bits: the " ++ backend ++ " family reads 4-bit EXL3 experts", .{std.json.fmt(bits, .{})});
        return error.UnsupportedQuant;
    }
    const book = q.get("codebook") orelse std.json.Value{ .null = {} };
    if (book != .string or !std.mem.eql(u8, book.string, "mcg")) {
        why.set("EXL3 codebook {f}: the " ++ backend ++ " family reads \"mcg\"", .{std.json.fmt(book, .{})});
        return error.UnsupportedQuant;
    }
}

test "EXL3 admission on HIP: Mia-AiLab's block passes; other widths and codebooks are refused by name" {
    const gpa = std.testing.allocator;
    const mia =
        \\{"quantization_config": {"quant_method": "exl3", "bits": 4, "codebook": "mcg", "head_bits": 16}}
    ;
    var why: modelopt.Why = .{};
    try admit("HIP", gpa, mia, &why);
    try std.testing.expectError(error.UnsupportedQuant, admit("HIP", gpa, "{\"quantization_config\": {\"quant_method\": \"exl3\", \"bits\": 3, \"codebook\": \"mcg\"}}", &why));
    try std.testing.expectEqualStrings("EXL3 at 3 bits: the HIP family reads 4-bit EXL3 experts", why.text());
    try std.testing.expectError(error.UnsupportedQuant, admit("HIP", gpa, "{\"text_config\": {\"quantization_config\": {\"quant_method\": \"exl3\", \"bits\": 4, \"codebook\": \"mul1\"}}}", &why));
    try std.testing.expectEqualStrings("EXL3 codebook \"mul1\": the HIP family reads \"mcg\"", why.text());
    try std.testing.expectError(error.UnsupportedQuant, admit("HIP", gpa, "{}", &why));
    try std.testing.expectEqualStrings("no quantization_config", why.text());
}
