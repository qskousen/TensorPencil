//! Qwen3 ChatML templating for tp-llm.
//!
//! Turns are appended to a growing token-id list so multi-turn chat is just
//! more appends:
//!   <|im_start|>role\n ... <|im_end|>\n
//! Generation starts from an open assistant turn and stops at <|im_end|> (or
//! <|endoftext|>). Qwen3-VL-Instruct is a non-thinking model: no <think>
//! block, and no implicit default system prompt.
//!
//! User text is tokenized with the same special-token scanner as everything
//! else, so a prompt containing e.g. "<|im_end|>" maps to the special id —
//! matching transformers' behavior with special tokens left unescaped.

const std = @import("std");
const tokenizer_mod = @import("../tokenizer.zig");

const Tokenizer = tokenizer_mod.Tokenizer;

/// "\n" as a single token, used by the template glue.
pub const newline_id: u32 = 198;

/// Vocab-dependent template/stop ids. Defaults match the embedded Qwen3
/// tokenizer; a GGUF-embedded tokenizer overrides them via applyTokenizer
/// (process-global, like the tokenizer itself in tp-llm).
pub var turn_end: u32 = tokenizer_mod.im_end;
pub var pad: u32 = tokenizer_mod.pad_token;
pub var newline: u32 = newline_id;

/// Point the template glue and stop check at `tok`'s vocab.
pub fn applyTokenizer(tok: *const Tokenizer) void {
    turn_end = tok.turn_end;
    pad = tok.pad;
    newline = tok.newline;
}

pub fn isStop(id: u32) bool {
    return id == turn_end or id == pad;
}

pub fn appendSystem(tok: *const Tokenizer, gpa: std.mem.Allocator, text: []const u8, out: *std.ArrayList(u32)) !void {
    try appendTurn(tok, gpa, "system", text, out);
}

pub fn appendUser(tok: *const Tokenizer, gpa: std.mem.Allocator, text: []const u8, out: *std.ArrayList(u32)) !void {
    try appendTurn(tok, gpa, "user", text, out);
}

/// Start the assistant turn the model completes: <|im_start|>assistant\n
pub fn openAssistant(tok: *const Tokenizer, gpa: std.mem.Allocator, out: *std.ArrayList(u32)) !void {
    try tok.encode(gpa, "<|im_start|>assistant\n", out);
}

/// Close a generated assistant turn so another user turn can follow (the
/// stop token itself is never appended by the engine).
pub fn closeAssistant(gpa: std.mem.Allocator, out: *std.ArrayList(u32)) !void {
    try out.append(gpa, turn_end);
    try out.append(gpa, newline);
}

fn appendTurn(tok: *const Tokenizer, gpa: std.mem.Allocator, role: []const u8, text: []const u8, out: *std.ArrayList(u32)) !void {
    try tok.encode(gpa, "<|im_start|>", out);
    try tok.encode(gpa, role, out);
    try out.append(gpa, newline);
    try tok.encode(gpa, text, out);
    try out.append(gpa, turn_end);
    try out.append(gpa, newline);
}

// --- tests -----------------------------------------------------------------

test "turn building matches whole-template tokenization" {
    const gpa = std.testing.allocator;
    var tok = try Tokenizer.init(gpa);
    defer tok.deinit();

    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    try appendSystem(&tok, gpa, "Describe the image:", &ids);
    try appendUser(&tok, gpa, "a cat", &ids);
    try openAssistant(&tok, gpa, &ids);

    var ref: std.ArrayList(u32) = .empty;
    defer ref.deinit(gpa);
    try tok.encode(
        gpa,
        "<|im_start|>system\nDescribe the image:<|im_end|>\n<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n",
        &ref,
    );
    try std.testing.expectEqualSlices(u32, ref.items, ids.items);
}

test "closeAssistant appends im_end + newline" {
    const gpa = std.testing.allocator;
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    try closeAssistant(gpa, &ids);
    try std.testing.expectEqualSlices(u32, &.{ tokenizer_mod.im_end, newline_id }, ids.items);
}

test "stop tokens" {
    try std.testing.expect(isStop(tokenizer_mod.im_end));
    try std.testing.expect(isStop(tokenizer_mod.pad_token));
    try std.testing.expect(!isStop(newline_id));
}
