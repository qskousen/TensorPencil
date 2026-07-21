//! Krea 2 prompt conditioning: chat template + prefix strip
//! (comfy/text_encoders/krea2.py).

const std = @import("std");
const tokenizer_mod = @import("tp_core").tokenizer;

const Tokenizer = tokenizer_mod.Tokenizer;

pub const template_prefix = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n";
pub const template_suffix = "<|im_end|>\n<|im_start|>assistant\n";

/// Tokenize a prompt wrapped in the Krea 2 template (thinking=true: no
/// <think> block). Prompts that already start with <|im_start|> skip the
/// template, matching ComfyUI.
pub fn buildIds(tok: *const Tokenizer, gpa: std.mem.Allocator, text: []const u8, out: *std.ArrayList(u32)) !void {
    if (std.mem.startsWith(u8, text, "<|im_start|>")) {
        return tok.encode(gpa, text, out);
    }
    try tok.encode(gpa, template_prefix, out);
    try tok.encode(gpa, text, out);
    try tok.encode(gpa, template_suffix, out);
}

/// Token offset at which conditioning starts: everything up to and including
/// the second <|im_start|> (+ "user" "\n" when present) is dropped.
pub fn stripOffset(ids: []const u32) usize {
    var template_end: usize = 0;
    var count: usize = 0;
    for (ids, 0..) |id, i| {
        if (id == tokenizer_mod.im_start and count < 2) {
            template_end = i;
            count += 1;
        }
    }
    if (ids.len > template_end + 3) {
        if (ids[template_end + 1] == 872 and ids[template_end + 2] == 198) { // "user" "\n"
            template_end += 3;
        }
    }
    return template_end;
}

test "strip offset finds user content" {
    // <|im_start|> a <|im_end|> <|im_start|> user \n content...
    const ids = [_]u32{ 151644, 5, 151645, 151644, 872, 198, 42, 43 };
    try std.testing.expectEqual(@as(usize, 6), stripOffset(&ids));
}

test "strip offset without user marker keeps im_start" {
    const ids = [_]u32{ 151644, 5, 151645, 151644, 900, 198, 42 };
    try std.testing.expectEqual(@as(usize, 3), stripOffset(&ids));
}

test "template structure" {
    const gpa = std.testing.allocator;
    var tok = try Tokenizer.init(gpa);
    defer tok.deinit();
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    try buildIds(&tok, gpa, "a cat", &ids);

    const items = ids.items;
    try std.testing.expectEqual(tokenizer_mod.im_start, items[0]);
    // Strip lands right before the prompt tokens ("a cat" = 64, 8251).
    const off = stripOffset(items);
    try std.testing.expectEqual(@as(u32, 64), items[off]);
    try std.testing.expectEqual(@as(u32, 8251), items[off + 1]);
    // Template ends with <|im_end|>\n<|im_start|>assistant\n.
    try std.testing.expectEqual(@as(u32, 198), items[items.len - 1]);
    try std.testing.expectEqual(@as(u32, 77091), items[items.len - 2]);
}
