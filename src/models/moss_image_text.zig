//! MOSS-Image prompt conditioning (`comfy/text_encoders/moss_image.py`): stock
//! Qwen3.5-9B, the final-normed last hidden state of EVERY token, template
//! included, over
//!
//!   <|im_start|>user\n{prompt.strip()}<|im_end|>\n<|im_start|>assistant\n<think>\n
//!
//! That is the chat template with thinking enabled: an OPEN think block, which
//! ComfyUI chose on purpose. Nothing is stripped and nothing is padded; the DiT
//! rotates text by its own index, so cond and uncond run at their own lengths.
//! A prompt that already opens with `<|im_start|>` is taken verbatim.

const std = @import("std");
const tp_core = @import("tp_core");
const ops = @import("tp_ops");
const qwen35 = @import("qwen35.zig");

const Tokenizer = tp_core.tokenizer.Tokenizer;
const WeightStore = tp_core.weights.WeightStore;

pub const template_prefix = "<|im_start|>user\n";
pub const template_suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n";

/// Python's `str.strip()` over the whitespace a prompt carries in practice.
const strip_set = " \t\n\r\x0b\x0c";

pub fn buildIds(tok: *const Tokenizer, gpa: std.mem.Allocator, text: []const u8, out: *std.ArrayList(u32)) !void {
    if (std.mem.startsWith(u8, text, "<|im_start|>")) return tok.encode(gpa, text, out);
    const full = try std.mem.concat(gpa, u8, &.{ template_prefix, std.mem.trim(u8, text, strip_set), template_suffix });
    defer gpa.free(full);
    try tok.encode(gpa, full, out);
}

/// The Qwen3.5 tower and its tokenizer, on the host.
pub const Encoder = struct {
    lm: qwen35.Model,
    tok: Tokenizer,

    pub fn load(gpa: std.mem.Allocator, store: WeightStore) !Encoder {
        var lm = try qwen35.Model.loadHf(gpa, store);
        errdefer lm.deinit();
        return .{ .lm = lm, .tok = try Tokenizer.initQwen35(gpa) };
    }

    pub fn deinit(self: *Encoder) void {
        self.lm.deinit();
        self.tok.deinit();
        self.* = undefined;
    }

    pub fn integerKind(self: *const Encoder) ?@import("lin.zig").Kind {
        return self.lm.integerKind();
    }

    pub fn hidden(self: *const Encoder) usize {
        return self.lm.cfg.hidden;
    }

    /// `[ids.len][hidden]`, caller frees.
    pub fn encodeIds(self: *const Encoder, io: std.Io, gpa: std.mem.Allocator, ids: []const u32) ![]f32 {
        const cfg = self.lm.cfg;
        var state = try qwen35.State.init(gpa, cfg, ids.len, .f32);
        defer state.deinit(gpa);
        var freqs = try ops.rope.rotateHalfFreqs(gpa, ids.len, cfg.rope_dim, cfg.rope_theta);
        defer freqs.deinit(gpa);
        const out = try gpa.alloc(f32, ids.len * cfg.hidden);
        errdefer gpa.free(out);
        try self.lm.forwardCached(io, gpa, ids, &state, freqs, out);
        return out;
    }
};

// --- tests -----------------------------------------------------------------

const testing = std.testing;
const build_options = @import("build_options");
const safetensors = tp_core.safetensors;
const test_gate = @import("../test_gate.zig");

const ref_path = "src/models/assets/moss_image_te_ref.safetensors";
const te_ckpt = build_options.model_root ++ "/text_encoders/qwen3.5_9b-INT8_CONVROT.safetensors";

fn refIds(gpa: std.mem.Allocator, ref: *const safetensors.SafeTensors, key: []const u8) ![]u32 {
    const v = try ref.require(key);
    const raw = std.mem.bytesAsSlice(i32, v.bytes);
    const ids = try gpa.alloc(u32, raw.len);
    for (ids, raw) |*d, s| d.* = @intCast(s);
    return ids;
}

fn metaJson(gpa: std.mem.Allocator, ref: *const safetensors.SafeTensors, key: []const u8) !std.json.Parsed([]const []const u8) {
    const m = ref.metadata orelse return error.MissingMetadata;
    const v = m.get(key) orelse return error.MissingMetadata;
    if (v != .string) return error.MissingMetadata;
    return std.json.parseFromSlice([]const []const u8, gpa, v.string, .{});
}

test "the Qwen3.5 tokenizer matches ComfyUI's on a hard corpus" {
    const gpa = testing.allocator;
    const io = testing.io;
    var ref = try safetensors.SafeTensors.open(gpa, io, ref_path);
    defer ref.deinit();
    var tok = try Tokenizer.initQwen35(gpa);
    defer tok.deinit();
    const corpus = try metaJson(gpa, &ref, "corpus");
    defer corpus.deinit();
    for (corpus.value, 0..) |text, i| {
        var kb: [16]u8 = undefined;
        const want = try refIds(gpa, &ref, try std.fmt.bufPrint(&kb, "ids.{d}", .{i}));
        defer gpa.free(want);
        var got: std.ArrayList(u32) = .empty;
        defer got.deinit(gpa);
        try tok.encode(gpa, text, &got);
        errdefer std.debug.print("corpus {d} ({s}):\n want {any}\n got  {any}\n", .{ i, text, want, got.items });
        try testing.expectEqualSlices(u32, want, got.items);
    }
}

test "the MOSS-Image template tokenizes exactly like ComfyUI's" {
    const gpa = testing.allocator;
    const io = testing.io;
    var ref = try safetensors.SafeTensors.open(gpa, io, ref_path);
    defer ref.deinit();
    var tok = try Tokenizer.initQwen35(gpa);
    defer tok.deinit();
    const prompts = try metaJson(gpa, &ref, "prompts");
    defer prompts.deinit();
    for (prompts.value, 0..) |text, i| {
        var kb: [16]u8 = undefined;
        const want = try refIds(gpa, &ref, try std.fmt.bufPrint(&kb, "tokens.{d}", .{i}));
        defer gpa.free(want);
        var got: std.ArrayList(u32) = .empty;
        defer got.deinit(gpa);
        try buildIds(&tok, gpa, text, &got);
        errdefer std.debug.print("prompt {d}:\n want {any}\n got  {any}\n", .{ i, want, got.items });
        try testing.expectEqualSlices(u32, want, got.items);
    }
}

test "the MOSS-Image text encoder matches ComfyUI on the real checkpoint" {
    const gpa = testing.allocator;
    const io = testing.io;
    try test_gate.requireIntegration();
    try test_gate.requireModelFile(io, te_ckpt);
    var ref = try safetensors.SafeTensors.open(gpa, io, ref_path);
    defer ref.deinit();
    var ck = try safetensors.SafeTensors.open(gpa, io, te_ckpt);
    defer ck.deinit();
    var enc = try Encoder.load(gpa, .{ .safetensors = &ck });
    defer enc.deinit();
    try testing.expectEqual(@as(usize, 2), enc.lm.cfg.v_group);

    const prev = ops.matmul.exact_activations;
    ops.matmul.exact_activations = true;
    defer ops.matmul.exact_activations = prev;

    for (0..2) |i| {
        var kb: [16]u8 = undefined;
        const ids = try refIds(gpa, &ref, try std.fmt.bufPrint(&kb, "tokens.{d}", .{i}));
        defer gpa.free(ids);
        const want = try (try ref.require(try std.fmt.bufPrint(&kb, "cond.{d}", .{i}))).toF32Alloc(gpa);
        defer gpa.free(want);
        const got = try enc.encodeIds(io, gpa, ids);
        defer gpa.free(got);
        var num: f64 = 0;
        var den: f64 = 0;
        for (want, got) |w, g| {
            num += @as(f64, w - g) * (w - g);
            den += @as(f64, w) * w;
        }
        const rel = @sqrt(num / den);
        errdefer std.debug.print("prompt {d}: rel L2 {e:.3}\n", .{ i, rel });
        try testing.expect(rel < 1e-4);
    }
}
