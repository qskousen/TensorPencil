//! Mage-Flow prompt conditioning: the chat template that wraps a prompt before
//! it reaches Qwen3-VL-4B, and the prefix strip that follows
//! (`comfy/text_encoders/mage_flow.py`).
//!
//! Two templates, picked by whether the request carries reference images:
//!
//!   t2i   the caption-describing system turn. BYTE-IDENTICAL to krea2's, which
//!         is not a coincidence worth relying on: both were trained against the
//!         same Qwen-Image template, and a test pins ours against the reference
//!         rather than against `krea2_text`.
//!   edit  an instruction-following system turn, with the user turn opened by
//!         one `Image N: <|vision_start|><|image_pad|><|vision_end|>` block per
//!         reference image, in order, before the instruction itself.
//!
//! The strip is the same rule for both and is `krea2_text.stripOffset`: the
//! reference's `drop_idx` of 34 (t2i) and 64 (edit) are just where each system
//! turn happens to end, and counting `<|im_start|>` finds either. Mage-Flow
//! conditions on what is LEFT, so an off-by-one here shifts every row of the
//! conditioning the DiT sees without changing its shape.

const std = @import("std");
const tp_core = @import("tp_core");
const tokenizer_mod = tp_core.tokenizer;
const image = tp_core.image;
const krea2_text = @import("krea2_text.zig");
const qwen3 = @import("qwen3.zig");
const vit_mod = @import("minimax_h3_vit.zig");

const Tokenizer = tokenizer_mod.Tokenizer;

/// A reference picture as a caller hands it over: planar `[3][h][w]` in [0, 1],
/// at whatever size the file had. Both resizes happen inside.
pub const RefImage = struct {
    rgb: []const f32,
    h: usize,
    w: usize,
};

/// The long edge a reference is capped to for the VL tower, from ComfyUI's
/// `TextEncodeMageFlowEdit` ("training preprocessing").
pub const vl_long_edge: usize = 384;
/// `process_qwen2vl_images`'s own bounds. The cap above means the maximum never
/// binds; the minimum still can, on a small reference.
pub const vl_min_pixels: usize = 3136;
pub const vl_max_pixels: usize = 12845056;

pub const t2i_prefix = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n";
pub const edit_prefix = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n";
pub const suffix = "<|im_end|>\n<|im_start|>assistant\n";

/// One reference image's placeholder. `<|image_pad|>` is a single token here and
/// is expanded to the tower's merged patch tokens when the images are encoded.
pub const vision_block = "<|vision_start|><|image_pad|><|vision_end|>";

/// Everything up to and including the second `<|im_start|>` (+ `user` `\n`) is
/// dropped from the conditioning. Shared with krea2, which runs the same
/// ComfyUI code path.
pub const stripOffset = krea2_text.stripOffset;

/// Tokenize a prompt in the template `n_images` references call for. A prompt
/// that already opens with `<|im_start|>` is passed through untouched, matching
/// what krea2's and Z-Image's paths do: it is how a caller supplies its own turn
/// structure.
pub fn buildIds(
    tok: *const Tokenizer,
    gpa: std.mem.Allocator,
    text: []const u8,
    n_images: usize,
    out: *std.ArrayList(u32),
) !void {
    if (std.mem.startsWith(u8, text, "<|im_start|>")) {
        return tok.encode(gpa, text, out);
    }
    const body = try userTurn(gpa, text, n_images);
    defer gpa.free(body);
    const full = try std.mem.concat(gpa, u8, &.{
        if (n_images > 0) edit_prefix else t2i_prefix,
        body,
        suffix,
    });
    defer gpa.free(full);
    try tok.encode(gpa, full, out);
}

/// Where one reference image's rows land in the token sequence.
pub const Span = struct { at: usize, len: usize };

/// The token sequence plus the rows each reference occupies in it.
pub const Presentation = struct {
    ids: []u32,
    /// One per image, in order and ascending.
    spans: []Span,

    pub fn deinit(self: *Presentation, gpa: std.mem.Allocator) void {
        gpa.free(self.ids);
        gpa.free(self.spans);
        self.* = undefined;
    }
};

/// Tokenize the edit turn and expand each `<|image_pad|>` into as many rows as
/// its reference contributes.
///
/// The template carries ONE pad token per image; the reference replaces that
/// single entry with the image and only later does `process_tokens` give it the
/// tower's merged token count. Expanding here is that same step, and the ids in a
/// span are placeholders whose embeddings the caller overwrites with the tower's
/// rows -- which is why they stay a real pad id rather than a sentinel: the
/// encoder still has to be able to embed them.
///
/// `merged_tokens[i]` is image `i`'s merged token count, `(gh / 2) * (gw / 2)`.
/// Caller frees the result.
pub fn present(
    tok: *const Tokenizer,
    gpa: std.mem.Allocator,
    text: []const u8,
    merged_tokens: []const usize,
) !Presentation {
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    try buildIds(tok, gpa, text, merged_tokens.len, &ids);
    if (merged_tokens.len == 0) {
        return .{ .ids = try ids.toOwnedSlice(gpa), .spans = try gpa.alloc(Span, 0) };
    }

    const pad = tok.specialId("<|image_pad|>") orelse return error.MissingTensor;
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(gpa);
    const spans = try gpa.alloc(Span, merged_tokens.len);
    errdefer gpa.free(spans);

    var seen: usize = 0;
    for (ids.items) |id| {
        if (id != pad) {
            try out.append(gpa, id);
            continue;
        }
        if (seen >= merged_tokens.len) return error.ShapeMismatch;
        const n = merged_tokens[seen];
        if (n == 0) return error.ShapeMismatch;
        spans[seen] = .{ .at = out.items.len, .len = n };
        try out.appendNTimes(gpa, pad, n);
        seen += 1;
    }
    // A prompt that opened with `<|im_start|>` skipped the template, so it may
    // carry a different number of blocks than the caller has images. Refusing
    // beats splicing the wrong picture into the wrong hole.
    if (seen != merged_tokens.len) return error.ShapeMismatch;
    return .{ .ids = try out.toOwnedSlice(gpa), .spans = spans };
}

/// Everything the encoder needs for one turn: the token ids and, when there are
/// reference pictures, the spliced rows, the multimodal rope positions and the
/// DeepStack features. Owns one arena, so a caller frees it in one call.
pub const Conditioning = struct {
    arena: std.heap.ArenaAllocator,
    ids: []const u32,
    vision: qwen3.TextEncoder.Vision,

    pub fn deinit(self: *Conditioning) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

/// Present a turn to the encoder: the template, the reference pictures through
/// the VL tower, and the splice.
///
/// `vit` may be null only when `images` is empty. `hidden` is the language
/// tower's width (which the vision tower's mergers project to) and `rope_dims`
/// its interleaved-mrope slot counts; both come from the encoder's own config so
/// this cannot disagree with it.
///
/// ⚠️ **The reference is resized TWICE, to different sizes, and only the first
/// happens here.** This is the VL copy: capped to a 384 px long edge (bicubic,
/// the node's own step) and then put on the tower's patch grid. The DiT's copy
/// goes to the RENDER's resolution through the VAE, because Mage's RoPE aligns
/// reference and target content by POSITION. Feeding both one extent, which is
/// what MiniMax H3 deliberately does, misaligns the edit here.
pub fn prepare(
    gpa: std.mem.Allocator,
    io: std.Io,
    tok: *const Tokenizer,
    vit: ?*const vit_mod.Vit,
    text: []const u8,
    images: []const RefImage,
    hidden: usize,
    rope_dims: [3]usize,
) !Conditioning {
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();

    if (images.len == 0) {
        var p = try present(tok, gpa, text, &.{});
        defer p.deinit(gpa);
        return .{ .arena = arena, .ids = try a.dupe(u32, p.ids), .vision = .{} };
    }
    const v = vit orelse return error.ComponentNotInCheckpoint;

    const merged = try a.alloc([]f32, images.len);
    const deep = try a.alloc([][]f32, images.len);
    const spans = try a.alloc(vit_mod.ImageSpan, images.len);
    const counts = try a.alloc(usize, images.len);
    for (images, merged, deep, spans, counts) |img, *mg, *ds, *span, *count| {
        const cap = try capForVl(gpa, img);
        defer if (cap.owned) gpa.free(cap.rgb);
        var prep = try vit_mod.preprocessStill(gpa, v.cfg, cap.rgb, cap.h, cap.w, vl_min_pixels, vl_max_pixels);
        defer prep.deinit(gpa);
        var e = try vit_mod.encode(v, io, gpa, prep.patches, prep.grid_h, prep.grid_w);
        defer e.deinit(gpa);
        // Into the arena, so one teardown frees every image's rows.
        mg.* = try a.dupe(f32, e.merged);
        ds.* = try a.alloc([]f32, e.deepstack.len);
        for (e.deepstack, ds.*) |src, *dst| dst.* = try a.dupe(f32, src);
        count.* = e.tokens;
        span.* = .{ .index = 0, .size = e.tokens, .grid_h = prep.grid_h, .grid_w = prep.grid_w };
    }

    var p = try present(tok, gpa, text, counts);
    defer p.deinit(gpa);
    const ids = try a.dupe(u32, p.ids);
    const seq = ids.len;

    const blocks = try a.alloc(qwen3.TextEncoder.Vision.Block, images.len);
    const inject = try a.alloc(qwen3.TextEncoder.Vision.Span, images.len);
    var total_rows: usize = 0;
    for (p.spans, merged, spans, blocks, inject) |sp, mg, *span, *b, *inj| {
        b.* = .{ .at = sp.at, .rows = mg };
        inj.* = .{ .start = sp.at, .len = sp.len };
        span.index = sp.at;
        total_rows += sp.len;
    }

    // DeepStack entry `i` is every image's `i`th feature concatenated in BLOCK
    // order, which is the order `inject` walks them in. Collected from deep
    // tower blocks and injected into the FIRST few decoder layers.
    const n_ds = v.cfg.n_deepstack;
    const ds_out = try a.alloc([]const f32, n_ds);
    for (0..n_ds) |i| {
        const buf = try a.alloc(f32, total_rows * hidden);
        var at: usize = 0;
        for (deep) |per_image| {
            const src = per_image[i];
            @memcpy(buf[at..][0..src.len], src);
            at += src.len;
        }
        ds_out[i] = buf;
    }

    const pos = try a.alloc(f32, 3 * seq);
    try vit_mod.mropePositions(pos, seq, spans);

    return .{
        .arena = arena,
        .ids = ids,
        .vision = .{
            .blocks = blocks,
            .positions = pos,
            .rope_dims = rope_dims,
            .deepstack = ds_out,
            .inject = inject,
        },
    };
}

/// The node's cap: a reference's long edge is resized to 384 px for the VL
/// tower, bicubic, before the tower's own resize. Returns the original slice
/// untouched when it already fits; `owned` says whether to free the result.
pub fn capForVl(gpa: std.mem.Allocator, img: RefImage) !struct { rgb: []const f32, h: usize, w: usize, owned: bool } {
    const long = @max(img.h, img.w);
    if (long <= vl_long_edge) return .{ .rgb = img.rgb, .h = img.h, .w = img.w, .owned = false };
    const scale = @as(f64, vl_long_edge) / @as(f64, @floatFromInt(long));
    const nw: usize = @max(1, @as(usize, @intFromFloat(vit_mod.pyRound(@as(f64, @floatFromInt(img.w)) * scale))));
    const nh: usize = @max(1, @as(usize, @intFromFloat(vit_mod.pyRound(@as(f64, @floatFromInt(img.h)) * scale))));
    const out = try gpa.alloc(f32, 3 * nh * nw);
    errdefer gpa.free(out);
    image.resizeBicubic(out, img.rgb, img.h, img.w, nh, nw);
    return .{ .rgb = out, .h = nh, .w = nw, .owned = true };
}

/// The user turn's text: `Image 1: <block>Image 2: <block>...` then the
/// instruction, which is the body Mage-Flow-Edit was trained on. Caller frees.
pub fn userTurn(gpa: std.mem.Allocator, text: []const u8, n_images: usize) ![]u8 {
    if (n_images == 0) return gpa.dupe(u8, text);
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(gpa);
    for (0..n_images) |i| {
        try buf.print(gpa, "Image {d}: {s}", .{ i + 1, vision_block });
    }
    try buf.appendSlice(gpa, text);
    return buf.toOwnedSlice(gpa);
}

// --- tests -----------------------------------------------------------------

const testing = std.testing;
const test_gate = @import("../test_gate.zig");
const safetensors = tp_core.safetensors;

const ref_path = "src/models/assets/mageflow_te_ref.safetensors";
const te_ckpt = "/home/qt/genai/comfyui/models/text_encoders/qwen3VLInstruct4bHeretic_v10.safetensors";

fn relL2(want: []const f32, got: []const f32) f64 {
    std.debug.assert(want.len == got.len);
    var l2_ref: f64 = 0;
    var l2_err: f64 = 0;
    for (want, got) |e, a| {
        l2_ref += @as(f64, e) * e;
        l2_err += @as(f64, e - a) * (e - a);
    }
    return if (l2_ref > 0) @sqrt(l2_err / l2_ref) else @sqrt(l2_err);
}

test "the Mage-Flow template tokenizes exactly like ComfyUI's MageFlowTokenizer" {
    // Ungated on the encoder: this needs only the fixture's stored ids, not the
    // 5 GB checkpoint. Worth separating, because a template one token off still
    // encodes and still renders, it just conditions on a different sequence.
    const gpa = testing.allocator;
    const io = testing.io;
    std.Io.Dir.cwd().access(io, ref_path, .{}) catch return error.SkipZigTest;

    var ref = try safetensors.SafeTensors.open(gpa, io, ref_path);
    defer ref.deinit();
    var tok = try Tokenizer.init(gpa);
    defer tok.deinit();

    const prompts = [_][]const u8{ "a red apple on a wooden table", "" };
    for (prompts, 0..) |text, i| {
        var kb: [32]u8 = undefined;
        const v = try ref.require(try std.fmt.bufPrint(&kb, "tokens.{d}", .{i}));
        const want = std.mem.bytesAsSlice(i32, v.bytes);

        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(gpa);
        try buildIds(&tok, gpa, text, 0, &ids);

        errdefer std.debug.print("prompt {d}: {d} tokens, reference has {d}\n", .{ i, ids.items.len, want.len });
        try testing.expectEqual(want.len, ids.items.len);
        for (want, ids.items) |w, g| try testing.expectEqual(@as(u32, @intCast(w)), g);
        // The reference's own `drop_idx` for the t2i template.
        try testing.expectEqual(@as(usize, 34), stripOffset(ids.items));
    }
}

test "the Mage-Flow-Edit turn matches ComfyUI: tower, splice, mrope and deepstack" {
    // Staged, because the four pieces fail for different reasons: the tower is
    // the 4B config and the patch flattening, the splice is the template and the
    // pad expansion, the mrope is the timeline, and the deepstack is which rows
    // it lands on. A single conditioning comparison would say only "wrong".
    const gpa = testing.allocator;
    const io = testing.io;
    try test_gate.requireIntegration();
    try test_gate.requireModelFile(io, te_ckpt);
    try test_gate.requireModelFile(io, ref_path);

    var ref = try safetensors.SafeTensors.open(gpa, io, ref_path);
    defer ref.deinit();
    var st = try safetensors.SafeTensors.open(gpa, io, te_ckpt);
    defer st.deinit();

    var pfx = try tp_core.weights.Prefixed.init(gpa, .{ .safetensors = &st }, "model.visual.");
    defer pfx.deinit(gpa);
    var vit = try vit_mod.Vit.load(gpa, pfx.store(), vit_mod.Config.qwen3vl_4b);
    defer vit.deinit();

    const iv = try ref.require("edit.image");
    const shape = iv.info.shape.slice();
    const img_h = shape[1];
    const img_w = shape[2];
    const rgb = try iv.toF32Alloc(gpa);
    defer gpa.free(rgb);
    const img: RefImage = .{ .rgb = rgb, .h = img_h, .w = img_w };

    // --- the tower alone --------------------------------------------------
    {
        const cap = try capForVl(gpa, img);
        defer if (cap.owned) gpa.free(cap.rgb);
        // The cap has to BIND on this fixture, or it pins nothing about itself.
        try testing.expect(cap.owned and @max(cap.h, cap.w) == vl_long_edge);
        // Staged before the tower, so a resize mistake is separable from it.
        {
            const want = try (try ref.require("edit.capped")).toF32Alloc(gpa);
            defer gpa.free(want);
            try testing.expectEqual(want.len, cap.rgb.len);
            const rel = relL2(want, cap.rgb);
            errdefer std.debug.print("384 cap rel L2 {e:.4} ({d}x{d})\n", .{ rel, cap.h, cap.w });
            try testing.expect(rel < 5e-5); // measured 9.3e-6
            try testing.expect(rel < 1e-5);
        }
        var prep = try vit_mod.preprocessStill(gpa, vit.cfg, cap.rgb, cap.h, cap.w, vl_min_pixels, vl_max_pixels);
        defer prep.deinit(gpa);
        {
            const want = try (try ref.require("edit.patches")).toF32Alloc(gpa);
            defer gpa.free(want);
            try testing.expectEqual(want.len, prep.patches.len);
            const rel = relL2(want, prep.patches);
            errdefer std.debug.print("patch matrix rel L2 {e:.4} (grid {d}x{d})\n", .{ rel, prep.grid_h, prep.grid_w });
            // Measured 2.2e-5. A RANDOM reference through two chained resamplers
            // is the worst case for agreement: there is no low-frequency content
            // for the two to settle on, so this floor is the resize's.
            try testing.expect(rel < 1e-4);
            // A RANDOM reference is the worst case for two chained resamplers:
            // there is no low-frequency content for the two to agree on, so
            // this floor is the resize's, not the tower's.
            try testing.expect(rel < 1e-4);
        }
        // The interpolated position rows on their own. ⚠️ This is the one place
        // ComfyUI leaves a bf16 sum in an otherwise-f32 tower: it looks the
        // table up and weights it at `pos_embed.weight.dtype`. The fixture's
        // reference is taken with that upcast, and `edit.pos_bf16` is the floor
        // it would otherwise impose -- measured at 2.6e-3 on these rows and
        // 1.4e-2 on the tower's output, against our 4.1e-7 here. Comparing
        // against the bf16 rows instead would pin ComfyUI's rounding rather than
        // this interpolation, which is what the second assert says out loud.
        {
            const np = prep.grid_h * prep.grid_w;
            const row = try gpa.alloc(f32, np * vit.cfg.dim);
            defer gpa.free(row);
            vit_mod.interpolatePos(row, vit.pos_embed, vit.cfg.pos_grid, vit.cfg.dim, prep.grid_h, prep.grid_w);
            const pos = try gpa.alloc(f32, np * vit.cfg.dim);
            defer gpa.free(pos);
            vit_mod.mergeOrder(pos, row, vit.cfg.dim, prep.grid_h, prep.grid_w, vit.cfg.merge);

            const want = try (try ref.require("edit.pos")).toF32Alloc(gpa);
            defer gpa.free(want);
            const bf16 = try (try ref.require("edit.pos_bf16")).toF32Alloc(gpa);
            defer gpa.free(bf16);
            const rel = relL2(want, pos);
            const floor = relL2(want, bf16);
            errdefer std.debug.print("position rows rel L2 {e:.4}, reference's own bf16 floor {e:.4}\n", .{ rel, floor });
            errdefer std.debug.print("position rows rel L2 {e:.4}, reference's own bf16 floor {e:.4}\n", .{ rel, floor });
            try testing.expect(rel < 1e-5); // measured 4.1e-7
            try testing.expect(floor > 100 * rel); // measured 2.6e-3, i.e. 6400x
        }

        // The tower on the REFERENCE's own patch matrix, so its disagreement is
        // its own rather than the resize's amplified through 24 blocks.
        {
            const patches = try (try ref.require("edit.patches")).toF32Alloc(gpa);
            defer gpa.free(patches);
            var e = try vit_mod.encode(&vit, io, gpa, patches, prep.grid_h, prep.grid_w);
            defer e.deinit(gpa);
            const want = try (try ref.require("edit.merged")).toF32Alloc(gpa);
            defer gpa.free(want);
            const rel = relL2(want, e.merged);
            errdefer std.debug.print("tower on the reference's patches: rel L2 {e:.4}\n", .{rel});
            try testing.expect(rel < 1e-4); // measured 2.5e-5
        }
        var enc = try vit_mod.encode(&vit, io, gpa, prep.patches, prep.grid_h, prep.grid_w);
        defer enc.deinit(gpa);

        const want = try (try ref.require("edit.merged")).toF32Alloc(gpa);
        defer gpa.free(want);
        try testing.expectEqual(want.len, enc.merged.len);
        const rel = relL2(want, enc.merged);
        errdefer std.debug.print("tower merged rel L2 {e:.4} ({d} tokens, grid {d}x{d})\n", .{ rel, enc.tokens, prep.grid_h, prep.grid_w });
        // Looser than the line above by exactly the resize floor: a RANDOM
        // reference through two chained resamplers is the worst case for it.
        // Measured 7.6e-5 against the 2.5e-5 above: the difference is the resize
        // floor, carried through 24 blocks.
        try testing.expect(rel < 3e-4);

        var kb: [32]u8 = undefined;
        for (enc.deepstack, 0..) |got, i| {
            const w = try (try ref.require(try std.fmt.bufPrint(&kb, "edit.deepstack.{d}", .{i}))).toF32Alloc(gpa);
            defer gpa.free(w);
            const r = relL2(w, got);
            errdefer std.debug.print("deepstack {d} rel L2 {e:.4}\n", .{ i, r });
            try testing.expect(r < 1e-4);
        }
    }

    // --- the splice -------------------------------------------------------
    var enc = try qwen3.TextEncoder.loadVariant(gpa, .{ .safetensors = &st }, .mageflow);
    defer enc.deinit();

    var tok = try Tokenizer.init(gpa);
    defer tok.deinit();
    var cond = try prepare(gpa, io, &tok, &vit, "replace the background with a grassland prairie", &.{img}, enc.cfg.hidden, enc.cfg.rope_dims);
    defer cond.deinit();

    {
        // The fixture stores the template ids BEFORE expansion (one pad per
        // image), so the two agree once our expansion is collapsed back.
        const tv = try ref.require("edit.tokens");
        const want = std.mem.bytesAsSlice(i32, tv.bytes);
        try testing.expectEqual(@as(usize, 1), cond.vision.blocks.len);
        const span = cond.vision.inject[0];
        const collapsed = cond.ids.len - span.len + 1;
        errdefer std.debug.print("edit ids: {d} expanded, {d} collapsed, reference {d}\n", .{ cond.ids.len, collapsed, want.len });
        try testing.expectEqual(want.len, collapsed);
        for (want, 0..) |w, i| {
            const j = if (i <= span.start) i else i + span.len - 1;
            try testing.expectEqual(@as(u32, @intCast(w)), cond.ids[j]);
        }
        // The edit template's own drop_idx.
        try testing.expectEqual(@as(usize, 64), stripOffset(cond.ids));
    }

    // --- the whole conditioning -------------------------------------------
    const full = try enc.encodeVision(io, gpa, cond.ids, cond.vision, null);
    defer gpa.free(full);
    const off = stripOffset(cond.ids);
    const row = enc.cfg.hidden;
    const got = full[off * row ..];

    const want = try (try ref.require("edit.cond")).toF32Alloc(gpa);
    defer gpa.free(want);
    try testing.expectEqual(want.len, got.len);
    const rel = relL2(want, got);
    // The control: the SAME encode with the vision left out entirely, i.e. the
    // pad tokens' own embeddings where the tower's rows should be. Without it a
    // tolerance says nothing about whether the splice happened at all.
    const plain = try enc.encode(io, gpa, cond.ids, null);
    defer gpa.free(plain);
    const rel_nosplice = relL2(want, plain[off * row ..]);
    errdefer std.debug.print("edit conditioning rel L2 {e:.4}, unspliced {e:.4}\n", .{ rel, rel_nosplice });
    // Measured 5.2e-4: the resize floor above, carried through the tower and
    // then diluted by the text rows. The control is what makes that number mean
    // something -- an unspliced encode sits at 1.2, three orders out, so this is
    // a statement about the splice and not about the tolerance.
    try testing.expect(rel < 2e-3);
    try testing.expect(rel_nosplice > 100 * rel);
}

test "Mage-Flow conditions on the NORMED final hidden state" {
    const gpa = testing.allocator;
    const io = testing.io;
    try test_gate.requireIntegration();
    try test_gate.requireModelFile(io, te_ckpt);
    try test_gate.requireModelFile(io, ref_path);

    var ref = try safetensors.SafeTensors.open(gpa, io, ref_path);
    defer ref.deinit();
    var st = try safetensors.SafeTensors.open(gpa, io, te_ckpt);
    defer st.deinit();

    var tok = try Tokenizer.init(gpa);
    defer tok.deinit();
    var enc = try qwen3.TextEncoder.loadVariant(gpa, .{ .safetensors = &st }, .mageflow);
    defer enc.deinit();
    try testing.expectEqual(@as(usize, 1), enc.taps.len);

    const prompts = [_][]const u8{ "a red apple on a wooden table", "" };
    for (prompts, 0..) |text, i| {
        var kb: [32]u8 = undefined;
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(gpa);
        try buildIds(&tok, gpa, text, 0, &ids);

        const full = try enc.encode(io, gpa, ids.items, null);
        defer gpa.free(full);
        const off = stripOffset(ids.items);
        const row = enc.cfg.hidden;
        const got = full[off * row ..];

        const want = try (try ref.require(try std.fmt.bufPrint(&kb, "cond.{d}", .{i}))).toF32Alloc(gpa);
        defer gpa.free(want);
        const nonorm = try (try ref.require(try std.fmt.bufPrint(&kb, "cond_nonorm.{d}", .{i}))).toF32Alloc(gpa);
        defer gpa.free(nonorm);
        try testing.expectEqual(want.len, got.len);

        const rel = relL2(want, got[0..want.len]);
        const rel_wrong = relL2(nonorm, got[0..nonorm.len]);
        errdefer std.debug.print(
            "prompt {d}: vs normed tap {e:.4}, vs unnormed tap {e:.4}\n",
            .{ i, rel, rel_wrong },
        );
        // Measured 8.6e-6. The reference ran at the checkpoint's own dtype
        // rather than fp32, but both sides read the same stored bytes, so what
        // is left is reduction order, not a dtype floor.
        //
        // The second assert is what gives the first teeth: krea2's tap over this
        // same checkpoint (the final norm skipped) sits at 6.7e-1, four orders
        // out, so agreement above is a statement about the tap and not about the
        // tolerance being loose.
        try testing.expect(rel < 1e-4);
        try testing.expect(rel_wrong > 1000 * rel);
    }
}

test "the edit turn opens with one numbered block per reference image" {
    const gpa = testing.allocator;
    const none = try userTurn(gpa, "make it red", 0);
    defer gpa.free(none);
    try testing.expectEqualStrings("make it red", none);

    const two = try userTurn(gpa, "make it red", 2);
    defer gpa.free(two);
    try testing.expectEqualStrings(
        "Image 1: " ++ vision_block ++ "Image 2: " ++ vision_block ++ "make it red",
        two,
    );
}

test "present expands each image pad to its own token count" {
    const gpa = testing.allocator;
    var tok = try Tokenizer.init(gpa);
    defer tok.deinit();
    const pad = tok.specialId("<|image_pad|>").?;

    // Two references of different sizes, so a swapped pair is visible.
    var p = try present(&tok, gpa, "make it red", &.{ 6, 15 });
    defer p.deinit(gpa);
    try testing.expectEqual(@as(usize, 2), p.spans.len);
    try testing.expectEqual(@as(usize, 6), p.spans[0].len);
    try testing.expectEqual(@as(usize, 15), p.spans[1].len);
    try testing.expect(p.spans[0].at + p.spans[0].len <= p.spans[1].at);
    for (p.spans) |sp| {
        for (p.ids[sp.at..][0..sp.len]) |id| try testing.expectEqual(pad, id);
    }
    // Exactly the spans are pads, nothing else in the sequence is.
    var pads: usize = 0;
    for (p.ids) |id| {
        if (id == pad) pads += 1;
    }
    try testing.expectEqual(@as(usize, 21), pads);
    // The strip still lands before the first block.
    try testing.expect(stripOffset(p.ids) <= p.spans[0].at);

    // No images is the plain t2i sequence with no spans.
    var t2i = try present(&tok, gpa, "a cat", &.{});
    defer t2i.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), t2i.spans.len);
    for (t2i.ids) |id| try testing.expect(id != pad);
}

test "both templates strip to the first prompt token" {
    const gpa = testing.allocator;
    var tok = try Tokenizer.init(gpa);
    defer tok.deinit();

    for ([_]usize{ 0, 1 }) |n_images| {
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(gpa);
        try buildIds(&tok, gpa, "a cat", n_images, &ids);
        const items = ids.items;
        try testing.expectEqual(tokenizer_mod.im_start, items[0]);
        // The strip lands on the user turn's first token: the prompt itself for
        // t2i ("a cat" = 64, 8251), the "Image" of the first block for edit.
        const off = stripOffset(items);
        if (n_images == 0) {
            try testing.expectEqual(@as(u32, 64), items[off]);
            try testing.expectEqual(@as(u32, 8251), items[off + 1]);
        } else {
            // Whatever "Image" tokenizes to, the prompt must still be in there
            // and the vision block must survive the strip.
            try testing.expect(off < items.len);
            const vision_start = tok.specialId("<|vision_start|>").?;
            var saw_vision = false;
            for (items[off..]) |id| {
                if (id == vision_start) saw_vision = true;
            }
            try testing.expect(saw_vision);
        }
        // Both end with <|im_start|>assistant\n.
        try testing.expectEqual(@as(u32, 198), items[items.len - 1]);
        try testing.expectEqual(@as(u32, 77091), items[items.len - 2]);
    }
}

