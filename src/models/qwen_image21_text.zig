//! Qwen-Image 2.1 prompt conditioning (`comfy/text_encoders/qwen_image21.py`):
//! Qwen3-VL-8B's last layer, NOT final-normed, over
//!
//!   <|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n
//!   <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
//!
//! with everything before the SECOND `<|im_start|>` dropped afterwards: the system
//! turn steers the encoder and is not conditioning. The prompt is not stripped; an
//! empty one becomes a single space (the node's `prevent_empty_text`), and one that
//! already opens with `<|im_start|>` is taken verbatim.

const std = @import("std");
const tp_core = @import("tp_core");
const tokenizer_mod = tp_core.tokenizer;

const Tokenizer = tokenizer_mod.Tokenizer;

pub const system_turn = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n";
pub const template_prefix = system_turn ++ "<|im_start|>user\n";
pub const template_suffix = "<|im_end|>\n<|im_start|>assistant\n";

pub fn buildIds(tok: *const Tokenizer, gpa: std.mem.Allocator, text: []const u8, out: *std.ArrayList(u32)) !void {
    if (std.mem.startsWith(u8, text, "<|im_start|>")) return tok.encode(gpa, text, out);
    const body = if (text.len == 0) " " else text;
    const full = try std.mem.concat(gpa, u8, &.{ template_prefix, body, template_suffix });
    defer gpa.free(full);
    try tok.encode(gpa, full, out);
}

/// Rows the conditioning drops: everything before the second `<|im_start|>`, or
/// none when there is no second one.
pub fn stripOffset(ids: []const u32) usize {
    var seen: usize = 0;
    for (ids, 0..) |id, i| if (id == tokenizer_mod.im_start) {
        seen += 1;
        if (seen == 2) return i;
    };
    return 0;
}

test "the strip keeps the user turn onward" {
    const gpa = std.testing.allocator;
    var tok = try Tokenizer.init(gpa);
    defer tok.deinit();
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    try buildIds(&tok, gpa, "a cat", &ids);
    const off = stripOffset(ids.items);
    try std.testing.expectEqual(tokenizer_mod.im_start, ids.items[0]);
    try std.testing.expectEqual(tokenizer_mod.im_start, ids.items[off]);
    try std.testing.expect(off > 0);
}

const build_options = @import("build_options");
const safetensors = tp_core.safetensors;
const qwen3 = @import("qwen3.zig");
const ops = @import("tp_ops");
const test_gate = @import("../test_gate.zig");

const ref_path = "src/models/assets/qwen_image21_te_ref.safetensors";
const te_ckpt = build_options.model_root ++ "/text_encoders/qwen3vl_8b_fp8_scaled.safetensors";

fn refIds(gpa: std.mem.Allocator, ref: *const safetensors.SafeTensors, key: []const u8) ![]u32 {
    const raw = std.mem.bytesAsSlice(i32, (try ref.require(key)).bytes);
    const ids = try gpa.alloc(u32, raw.len);
    for (ids, raw) |*d, s| d.* = @intCast(s);
    return ids;
}

fn refPrompts(gpa: std.mem.Allocator, ref: *const safetensors.SafeTensors) !std.json.Parsed([]const []const u8) {
    const m = ref.metadata orelse return error.MissingMetadata;
    const v = m.get("prompts") orelse return error.MissingMetadata;
    if (v != .string) return error.MissingMetadata;
    return std.json.parseFromSlice([]const []const u8, gpa, v.string, .{});
}

fn relL2(want: []const f32, got: []const f32) f64 {
    var num: f64 = 0;
    var den: f64 = 0;
    for (want, got) |w, g| {
        num += @as(f64, w - g) * (w - g);
        den += @as(f64, w) * w;
    }
    return @sqrt(num / den);
}

test "the Qwen-Image 2.1 template tokenizes and strips exactly like ComfyUI" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var ref = try safetensors.SafeTensors.open(gpa, io, ref_path);
    defer ref.deinit();
    var tok = try Tokenizer.init(gpa);
    defer tok.deinit();
    const prompts = try refPrompts(gpa, &ref);
    defer prompts.deinit();
    for (prompts.value, 0..) |text, i| {
        var kb: [24]u8 = undefined;
        const want = try refIds(gpa, &ref, try std.mem.print(&kb, "tokens.{d}", .{i}));
        defer gpa.free(want);
        var got: std.ArrayList(u32) = .empty;
        defer got.deinit(gpa);
        try buildIds(&tok, gpa, text, &got);
        errdefer std.debug.print("prompt {d}:\n want {any}\n got  {any}\n", .{ i, want, got.items });
        try std.testing.expectEqualSlices(u32, want, got.items);
        const rows = (try ref.require(try std.mem.print(&kb, "cond.{d}", .{i}))).info.elemCount() / qwen3.Config.vl_8b.hidden;
        try std.testing.expectEqual(rows, got.items.len - stripOffset(got.items));
    }
}

test "the Qwen-Image 2.1 text encoder matches ComfyUI on the real checkpoint" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    try test_gate.requireIntegration();
    try test_gate.requireModelFile(io, te_ckpt);
    var ref = try safetensors.SafeTensors.open(gpa, io, ref_path);
    defer ref.deinit();
    var ck = try safetensors.SafeTensors.open(gpa, io, te_ckpt);
    defer ck.deinit();
    var enc = try qwen3.TextEncoder.loadVariant(gpa, .{ .safetensors = &ck }, .qwen_image21);
    defer enc.deinit();

    const prev = ops.matmul.exact_activations;
    ops.matmul.exact_activations = true;
    defer ops.matmul.exact_activations = prev;

    const hidden = qwen3.Config.vl_8b.hidden;
    for (0..2) |i| {
        var kb: [24]u8 = undefined;
        const ids = try refIds(gpa, &ref, try std.mem.print(&kb, "tokens.{d}", .{i}));
        defer gpa.free(ids);
        const want = try (try ref.require(try std.mem.print(&kb, "cond.{d}", .{i}))).toF32Alloc(gpa);
        defer gpa.free(want);
        const normed = try (try ref.require(try std.mem.print(&kb, "cond_normed.{d}", .{i}))).toF32Alloc(gpa);
        defer gpa.free(normed);
        const full = try enc.encode(io, gpa, ids, null);
        defer gpa.free(full);
        const got = full[stripOffset(ids) * hidden ..];
        try std.testing.expectEqual(want.len, got.len);
        const rel = relL2(want, got);
        const control = relL2(normed, got);
        errdefer std.debug.print("prompt {d}: rel L2 {e:.3}, against the normed tap {e:.3}\n", .{ i, rel, control });
        try std.testing.expect(rel < 1e-4);
        try std.testing.expect(control > 0.1);
    }
}
