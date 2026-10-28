//! What a chat model file says about itself, for either container. A GGUF carries
//! its architecture, tokenizer, chat template and trained context. An HF safetensors
//! export (a ComfyUI text encoder that kept its head, say) carries only weights, so
//! for the two families recognized there, Qwen3 and Qwen3.5, the rest comes from the
//! build: the embedded tokenizers and templates, and each family's published context.
//!
//! A safetensors Qwen3 is recognized only as one of the configurations
//! `qwen3.Config.detect` knows, because its RoPE theta appears in no tensor and
//! differs between releases of the same shape (Qwen3-4B against its 2507 refresh).

const std = @import("std");
const tp_core = @import("tp_core");
const Container = tp_core.container.Container;
const WeightStore = tp_core.weights.WeightStore;
const Tokenizer = tp_core.tokenizer.Tokenizer;
const chat_template = @import("chat_template.zig");
const qwen3 = @import("../models/qwen3.zig");
const qwen35 = @import("../models/qwen35.zig");

/// This file's source hashed, for a cache of what a model file was taken for: the
/// catalog mixes it into its rules fingerprint, so an edit here re-scans.
pub const source_fingerprint: u64 = blk: {
    @setEvalBranchQuota(1 << 24);
    break :blk std.hash.Wyhash.hash(0, @embedFile("model_file.zig"));
};

/// The llama.cpp architecture name of the chat model in `c`, or null when this
/// build cannot chat with it. A GGUF says what it is; whether that architecture
/// runs is `chat.familyForArch`'s question. A safetensors file must also have an
/// output head, since an encoder export that dropped it cannot produce text.
pub fn arch(c: *const Container) ?[]const u8 {
    return switch (c.*) {
        .gguf => |*g| g.getStr("general.architecture"),
        .safetensors => hfArch(c.store()),
    };
}

fn hfArch(store: WeightStore) ?[]const u8 {
    const head = store.get("lm_head.weight") != null;
    if (qwen35.Config.detectHf(store)) |_| {
        return if (head) "qwen35" else null;
    } else |_| {}
    const cfg = qwen3.Config.detect(store) catch return null;
    // Qwen3 ties its head to the embedding below 8B (hidden 4096).
    return if (head or cfg.hidden < 4096) "qwen3" else null;
}

/// A safetensors chat model's width and depth, what a model scan groups files by.
/// Null for a file `arch` would not name.
pub const Dims = struct { width: usize, blocks: usize };

pub fn hfDims(store: WeightStore) ?Dims {
    _ = hfArch(store) orelse return null;
    if (qwen35.Config.detectHf(store)) |cfg| {
        return .{ .width = cfg.hidden, .blocks = cfg.n_layers };
    } else |_| {}
    const cfg = qwen3.Config.detect(store) catch return null;
    return .{ .width = cfg.hidden, .blocks = cfg.n_layers };
}

/// The official size name for a recognized width and depth. Counting a file's
/// tensors does not give it: an export may carry a vision tower and MTP layers,
/// count a tied embedding, or pack two weights to a byte.
pub fn sizeLabel(a: []const u8, d: Dims) ?[]const u8 {
    const Known = struct { arch: []const u8, width: usize, blocks: usize, label: []const u8 };
    const known = [_]Known{
        .{ .arch = "qwen3", .width = 1024, .blocks = 28, .label = "0.6B" },
        .{ .arch = "qwen3", .width = 2048, .blocks = 28, .label = "1.7B" },
        .{ .arch = "qwen3", .width = 2560, .blocks = 36, .label = "4B" },
        .{ .arch = "qwen3", .width = 4096, .blocks = 36, .label = "8B" },
        .{ .arch = "qwen3", .width = 5120, .blocks = 40, .label = "14B" },
        .{ .arch = "qwen3", .width = 5120, .blocks = 64, .label = "32B" },
        .{ .arch = "qwen35", .width = 4096, .blocks = 32, .label = "9B" },
        .{ .arch = "qwen35", .width = 5120, .blocks = 64, .label = "27B" },
    };
    for (known) |k| {
        if (std.mem.eql(u8, k.arch, a) and k.width == d.width and k.blocks == d.blocks) return k.label;
    }
    return null;
}

/// The context length the model was trained for, or null when nothing says.
pub fn trainedContext(c: *const Container) ?usize {
    return switch (c.*) {
        .gguf => |*g| if (g.contextLength()) |n| @intCast(n) else null,
        .safetensors => {
            const a = hfArch(c.store()) orelse return null;
            return if (std.mem.eql(u8, a, "qwen35")) 262144 else 32768;
        },
    };
}

/// The model's tokenizer: the GGUF's own, or the embedded one for its family with the
/// end of a turn as its eos, which is what the instruct models' GGUFs declare.
pub fn tokenizer(gpa: std.mem.Allocator, c: *const Container) !Tokenizer {
    return switch (c.*) {
        .gguf => |*g| Tokenizer.initFromGguf(gpa, g),
        .safetensors => hfTokenizer(gpa, hfArch(c.store()) orelse return error.NotAChatModel),
    };
}

fn hfTokenizer(gpa: std.mem.Allocator, arch_name: []const u8) !Tokenizer {
    var t = if (std.mem.eql(u8, arch_name, "qwen35")) try Tokenizer.initQwen35(gpa) else try Tokenizer.init(gpa);
    t.eos = t.turn_end;
    return t;
}

/// The model's chat template: the GGUF's own (null when it has none), or the
/// embedded one for its family.
pub fn template(gpa: std.mem.Allocator, c: *const Container) !?chat_template.ChatTemplate {
    return switch (c.*) {
        .gguf => |*g| chat_template.ChatTemplate.fromGguf(gpa, g),
        .safetensors => {
            const a = hfArch(c.store()) orelse return error.NotAChatModel;
            return if (std.mem.eql(u8, a, "qwen35"))
                try chat_template.ChatTemplate.qwen35Fixed(gpa)
            else
                try chat_template.ChatTemplate.qwen3Embedded(gpa);
        },
    };
}

/// Encode a chat turn through `got` and through the tokenizer of the GGUF at `path`,
/// which shares its vocabulary, and require the same ids and stop tokens.
fn expectTokenizesLikeGguf(got: *const Tokenizer, path: []const u8) !void {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var g = try tp_core.gguf.Gguf.openHeader(gpa, io, path);
    defer g.deinit();
    var want = try Tokenizer.initFromGguf(gpa, &g);
    defer want.deinit();

    // Every special a chat turn carries, and text that exercises the byte-level split.
    const text = "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi! ¿Qué tal? 你好 🙂 x=3.14e-2 #tag  two  spaces\n" ++
        "<tool_response>\n{\"ok\": true}\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n" ++
        "<tool_call>\n{\"name\": \"f\"}\n</tool_call><|im_end|>\n<|endoftext|>";
    var a: std.ArrayList(u32) = .empty;
    defer a.deinit(gpa);
    var b: std.ArrayList(u32) = .empty;
    defer b.deinit(gpa);
    try want.encode(gpa, text, &a);
    try got.encode(gpa, text, &b);
    try std.testing.expectEqualSlices(u32, a.items, b.items);
    try std.testing.expectEqual(want.turn_end, got.turn_end);
    try std.testing.expectEqual(want.eos, got.eos);
    try std.testing.expectEqual(want.bos, got.bos);
}

test "a safetensors Qwen3.5 tokenizes a chat exactly as its GGUF does" {
    const path = "/mnt/watch/models/llm/Heretical-Qwen3.5-9B.Q6_K.gguf";
    try @import("../test_gate.zig").requireModelFile(std.testing.io, path);
    var got = try hfTokenizer(std.testing.allocator, "qwen35");
    defer got.deinit();
    try expectTokenizesLikeGguf(&got, path);
}

test "a safetensors Qwen3 tokenizes a chat exactly as its GGUF does" {
    const path = "/mnt/watch/models/diffusion/text_encoders/Qwen3-4B-Q4_K_M.gguf";
    try @import("../test_gate.zig").requireModelFile(std.testing.io, path);
    var got = try hfTokenizer(std.testing.allocator, "qwen3");
    defer got.deinit();
    try expectTokenizesLikeGguf(&got, path);
}
