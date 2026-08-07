//! Z-Image prompt conditioning: the chat template that wraps a prompt before it
//! reaches Qwen3-4B (`comfy/text_encoders/z_image.py::ZImageTokenizer`).
//!
//! Three differences from krea2's otherwise-similar path, all of them conventions
//! the checkpoint does not record:
//!
//! 1. **No system turn.** krea2 prefixes a "Describe the image by detailing…"
//!    system message; Z-Image's template is just a user turn.
//! 2. **Nothing is stripped.** krea2 drops everything up to and including the
//!    second `<|im_start|>` (+ `user` `\n`) before handing the states to the DiT.
//!    Z-Image conditions on the **whole** token sequence, template included — the
//!    `cap_embedder` and `context_refiner` see the chat markers. Stripping them
//!    would shorten the caption by 9 tokens and shift every RoPE position after it.
//! 3. **No padding.** ComfyUI's `Qwen3Tokenizer` is built with
//!    `pad_to_max_length=False`, `min_length=1` and no start/end token, so the
//!    sequence is exactly the template's tokens. The DiT's own 32-token pad is a
//!    separate thing and uses a *learned* pad embedding, not a pad token id.

const std = @import("std");
const tokenizer_mod = @import("tp_core").tokenizer;

const Tokenizer = tokenizer_mod.Tokenizer;

pub const template_prefix = "<|im_start|>user\n";
pub const template_suffix = "<|im_end|>\n<|im_start|>assistant\n";

/// Tokenize a prompt wrapped in the Z-Image template. A prompt that already opens
/// with `<|im_start|>` is passed through untouched, matching what krea2's path does
/// — it is how a caller supplies its own turn structure.
///
/// The formatted string is built and encoded in **one** call rather than as three
/// (prefix, text, suffix). Splitting would be equivalent only as long as BPE never
/// merges across the boundaries, which happens to hold for this template but is a
/// property of the vocabulary rather than of the code.
pub fn buildIds(tok: *const Tokenizer, gpa: std.mem.Allocator, text: []const u8, out: *std.ArrayList(u32)) !void {
    if (std.mem.startsWith(u8, text, "<|im_start|>")) {
        return tok.encode(gpa, text, out);
    }
    const full = try std.mem.concat(gpa, u8, &.{ template_prefix, text, template_suffix });
    defer gpa.free(full);
    try tok.encode(gpa, full, out);
}

// --- tests ------------------------------------------------------------------

const testing = std.testing;
const test_gate = @import("../test_gate.zig");
const safetensors = @import("tp_core").safetensors;
const qwen3 = @import("qwen3.zig");

const ref_path = "src/models/assets/zimage_ref.safetensors";
const te_ckpt = "/home/qt/genai/comfyui/models/text_encoders/qwen_3_4b.safetensors";

test "the Z-Image template tokenizes exactly like ComfyUI's ZImageTokenizer" {
    // Ungated: needs only the fixture's stored ids, not the 8 GB encoder. Worth
    // separating, because a template that is one token off still encodes and still
    // renders — it just conditions on a different sequence than the reference.
    const gpa = testing.allocator;
    const io = testing.io;
    // Not `test_gate.requireModelFile` — that also gates on `-Dintegration`, and
    // this needs no external checkpoint, only the in-repo fixture.
    std.Io.Dir.cwd().access(io, ref_path, .{}) catch return error.SkipZigTest;

    var ref = try safetensors.SafeTensors.open(gpa, io, ref_path);
    defer ref.deinit();
    var tok = try Tokenizer.init(gpa);
    defer tok.deinit();

    // The prompts the generator used, in order.
    const prompts = [_][]const u8{ "a photograph of an astronaut riding a horse", "" };
    for (prompts, 0..) |text, i| {
        var kb: [32]u8 = undefined;
        const bytes = (try ref.require(try std.fmt.bufPrint(&kb, "te.tokens.{d}", .{i}))).bytes;

        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(gpa);
        try buildIds(&tok, gpa, text, &ids);

        errdefer std.debug.print("prompt {d} ({s}): got {any}\n", .{ i, text, ids.items });
        try testing.expectEqual(bytes.len / 4, ids.items.len);
        for (ids.items, 0..) |got, j| {
            try testing.expectEqual(std.mem.readInt(i32, bytes[j * 4 ..][0..4], .little), @as(i32, @intCast(got)));
        }
    }
}

test "the Z-Image text encoder matches ComfyUI's Qwen3-4B penultimate state" {
    // The three things this pins, none of which fail loudly on their own: the
    // `model.` prefix (krea2's checkpoint nests under `model.language_model.`), the
    // RoPE theta of 1e6 (the VL variant's is 5e6), and taking the state *entering*
    // layer 35 with the final norm skipped.
    const gpa = testing.allocator;
    const io = testing.io;
    try test_gate.requireIntegration();
    try test_gate.requireModelFile(io, te_ckpt);
    try test_gate.requireModelFile(io, ref_path);

    var ref = try safetensors.SafeTensors.open(gpa, io, ref_path);
    defer ref.deinit();
    var st = try safetensors.SafeTensors.open(gpa, io, te_ckpt);
    defer st.deinit();

    var enc = try qwen3.TextEncoder.loadVariant(gpa, .{ .safetensors = &st }, .zimage);
    defer enc.deinit();
    try testing.expectEqual(@as(usize, 1), enc.tapCount());
    // 35 layers, not 36: the tap fires before layer 35, so layer 35 and the final
    // norm are never evaluated and are not loaded.
    try testing.expectEqual(@as(usize, 35), enc.layers.len);

    var tok = try Tokenizer.init(gpa);
    defer tok.deinit();

    const prompts = [_][]const u8{ "a photograph of an astronaut riding a horse", "" };
    for (prompts, 0..) |text, i| {
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(gpa);
        try buildIds(&tok, gpa, text, &ids);

        const got = try enc.encode(io, gpa, ids.items, null);
        defer gpa.free(got);

        var kb: [32]u8 = undefined;
        const want = try (try ref.require(try std.fmt.bufPrint(&kb, "te.cond.{d}", .{i}))).toF32Alloc(gpa);
        defer gpa.free(want);
        try testing.expectEqual(want.len, got.len);

        var l2_ref: f64 = 0;
        var l2_err: f64 = 0;
        for (want, got) |e, a| {
            l2_ref += @as(f64, e) * e;
            l2_err += @as(f64, e - a) * (e - a);
        }
        const rel = @sqrt(l2_err / l2_ref);
        errdefer std.debug.print("prompt {d}: cond rel L2 {e:.4}\n", .{ i, rel });
        // The fixture stores f16 and these hidden states reach ~1.4e4 in magnitude
        // (Qwen3 carries large outliers), so the storage rounding dominates.
        try testing.expect(rel < 2e-3);
    }
}

test "the Vulkan text encoder matches the CPU one on Z-Image's bf16 weights" {
    // ⚠️ **This test exists because its absence cost a blank white render.**
    // `qwen3_gpu.encode`'s GEMM handled fp8 only: krea2's encoder checkpoint is fp8,
    // Z-Image's `qwen_3_4b.safetensors` is **bf16**, and the fallback branch read the
    // bf16 bytes as f32. Every GEMM was garbage, the conditioning came out
    // non-finite, and `planarF32ToRgb8` clamped the whole image to white with no
    // error at any layer.
    //
    // The generalization that introduced it also had to be checked here rather than
    // reasoned about: this file's `wcode` helper does read bf16 natively, but that is
    // the LM's GEMV path, not the encoder's GEMM.
    const gpa = testing.allocator;
    const io = testing.io;
    try test_gate.requireIntegration();
    try test_gate.requireModelFile(io, te_ckpt);
    std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;

    const gpu_mod = @import("tp_gpu").context;
    const qwen3_gpu = @import("qwen3_gpu.zig");
    const ctx = gpu_mod.Context.init(gpa) catch return error.SkipZigTest;
    defer ctx.deinit();

    var st = try safetensors.SafeTensors.open(gpa, io, te_ckpt);
    defer st.deinit();
    var enc = try qwen3.TextEncoder.loadVariant(gpa, .{ .safetensors = &st }, .zimage);
    defer enc.deinit();
    if (!qwen3_gpu.supportsWeights(ctx, &enc)) return error.SkipZigTest;

    var tok = try Tokenizer.init(gpa);
    defer tok.deinit();
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    try buildIds(&tok, gpa, "a photograph of an astronaut riding a horse", &ids);

    const want = try enc.encode(io, gpa, ids.items, null);
    defer gpa.free(want);
    const got = try qwen3_gpu.encode(&enc, ctx, io, gpa, ids.items, false, null);
    defer gpa.free(got);
    try testing.expectEqual(want.len, got.len);

    // Non-finite output is the actual failure mode, so name it separately from a
    // numeric mismatch — "rel L2 = nan" does not say which side blew up.
    for (got) |v| {
        errdefer std.debug.print("gpu encode produced a non-finite value\n", .{});
        try testing.expect(std.math.isFinite(v));
    }

    var l2_ref: f64 = 0;
    var l2_err: f64 = 0;
    for (want, got) |e, a| {
        l2_ref += @as(f64, e) * e;
        l2_err += @as(f64, e - a) * (e - a);
    }
    const rel = @sqrt(l2_err / l2_ref);
    errdefer std.debug.print("vulkan encode rel L2 {e:.4}\n", .{rel});
    try testing.expect(rel < 5e-3);
}
