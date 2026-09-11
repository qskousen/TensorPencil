//! SenseNova U1.5 8B MoT, a pixel-space text-to-image model that is not a DiT.
//!
//! One Qwen3-shaped trunk carries TWO weight copies per layer (Mixture of
//! Transformers). Text tokens run the base copy causally and leave a KV cache;
//! image tokens run the `_mot_gen` copy, attending with no mask at all over
//! themselves concatenated with that prefix KV. There is no VAE and no separate
//! text encoder: the "latent" is the image in [-1, 1], one token per 32x32 px,
//! and the trunk's own understanding half is what encodes the prompt.
//!
//! Reference is ComfyUI's `comfy/ldm/sensenova/`; fixtures come from
//! tools/gen_sensenova_fixtures.py. GPU twins are sensenova_gpu.zig (Vulkan) and
//! sensenova_cuda.zig (both CUDA arms).
//!
//! ## Conventions that are silent wrong answers
//!
//! - **The two vision towers take different input normalization.** The
//!   understanding tower (`vision_model`, reference images) takes ImageNet
//!   mean/std over [0, 1]; the generation tower (`fm_modules.vision_model_mot_gen`,
//!   the noised canvas) takes raw [-1, 1]. Their convolutions are the same shape,
//!   so feeding one the other's input renders a plausible wrong image.
//! - **Two RoPE conventions live in this model.** The vision embedder rotates
//!   INTERLEAVED pairs, with the first half of the channels against the patch
//!   column and the second half against the row. The trunk rotates SPLIT-HALF.
//! - **The head dim splits 64 / 32 / 32**: the first 64 dims carry the sequence
//!   position at theta 5e6, then 32 for the token row and 32 for its column, both
//!   at theta 1e4. `q_norm` covers the first 64 and `q_norm_hw` the last 64, both
//!   applied BEFORE the rope and before the hw half splits again.
//! - **Every image token shares one t index**, the prefix length, and differs
//!   only in h/w. Prefix tokens run t = position with h = w = 0.
//! - **The generation stream is unmasked.** Only the prefix pass is causal.
//! - **Padding to 32 px is replicate up to 16, then CIRCULAR**, and the velocity
//!   is cropped back to the requested extent.
//! - **The head predicts x0**, and the model returns `v = (x - x0) / max(sigma, 0.02)`
//!   where sigma is `1 - t`.
//! - **The model's `t` is `1 - sigma`**, not the sigma.

const std = @import("std");
const lora_mod = @import("lora.zig");
const tp_core = @import("tp_core");
const weights_mod = tp_core.weights;
const safetensors = tp_core.safetensors;
const tokenizer_mod = tp_core.tokenizer;
const ops = @import("tp_ops");
const quant_weight = @import("quant_weight.zig");
const lin_mod = @import("lin.zig");
const qwen3 = @import("qwen3.zig");

const WeightStore = weights_mod.WeightStore;
const Weight = ops.matmul.Weight;

pub const Config = struct {
    dim: usize,
    inter: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    vocab: usize,
    /// `patch_embedding`'s output width, before the 2x2 merge into `dim`.
    vis_dim: usize,
    /// Pixels per `patch_embedding` cell.
    patch: usize,
    /// Cells per `dense_embedding` cell, so a token covers `patch * merge` px.
    merge: usize,
    /// Sinusoidal frequency count feeding both scalar embedders.
    t_freq: usize,
    t_max_period: f32,
    norm_eps: f32,
    /// RoPE theta for the sequence axis, and for the row/column axes.
    theta_t: f64,
    theta_hw: f64,
    /// RoPE theta inside the vision patch embedder.
    theta_vis: f64,
    /// Token count the resolution noise scale is measured against, and its cap.
    noise_base_seq: f32,
    noise_max: f32,
    /// Floor on `1 - t` in the x0-to-velocity conversion.
    velocity_eps: f32,

    pub fn qDim(self: Config) usize {
        return self.n_heads * self.head_dim;
    }
    pub fn kvDim(self: Config) usize {
        return self.n_kv_heads * self.head_dim;
    }
    /// Pixels per generated token, and the multiple every canvas is padded to.
    pub fn tokenPx(self: Config) usize {
        return self.patch * self.merge;
    }
    /// The head dim's three RoPE spans: sequence, row, column.
    pub fn spanT(self: Config) usize {
        return self.head_dim / 2;
    }
    pub fn spanHw(self: Config) usize {
        return self.head_dim / 4;
    }
    /// `fm_head`'s two convolution widths, fixed by its pixel shuffles: the head
    /// upscales by 2, 2 and 8, and must land on 3 channels, so the second
    /// convolution always writes 3 * 8 * 8.
    pub fn fmC1(self: Config) usize {
        return self.dim / 4;
    }
    pub fn fmC2(self: Config) usize {
        return self.dim / 16;
    }
};

pub const u15_8b: Config = .{
    .dim = 4096,
    .inter = 12288,
    .n_layers = 42,
    .n_heads = 32,
    .n_kv_heads = 8,
    .head_dim = 128,
    .vocab = 151936,
    .vis_dim = 1024,
    .patch = 16,
    .merge = 2,
    .t_freq = 256,
    .t_max_period = 10000.0,
    .norm_eps = 1e-6,
    .theta_t = 5000000.0,
    .theta_hw = 10000.0,
    .theta_vis = 10000.0,
    .noise_base_seq = 64.0,
    .noise_max = 16.0,
    .velocity_eps = 0.02,
};

/// The generated canvas IS the image: three channels, no spatial compression.
pub const latent_channels = 3;
pub const spatial_scale = 1;

/// ComfyUI's default flow shift for this family (`supported_models.SenseNovaU15`).
pub const default_shift: f32 = 3.0;

// --- prompt -----------------------------------------------------------------

pub const tok_im_start: u32 = 151644;
pub const tok_im_end: u32 = 151645;
pub const tok_think_open: u32 = 151667;
pub const tok_think_close: u32 = 151668;
pub const tok_img_context: u32 = 151669;
pub const tok_img_start: u32 = 151670;
pub const tok_img_end: u32 = 151671;

/// The tokens SenseNova adds past the base Qwen vocabulary. Their ids are
/// positional in the checkpoint's embedding table, so the spellings and the
/// numbers both matter.
pub const extra_specials = [_]tokenizer_mod.Special{
    .{ .text = "<IMG_CONTEXT>", .id = tok_img_context },
    .{ .text = "<img>", .id = tok_img_start },
    .{ .text = "</img>", .id = tok_img_end },
    .{ .text = "<quad>", .id = 151672 },
    .{ .text = "</quad>", .id = 151673 },
    .{ .text = "<ref>", .id = 151674 },
    .{ .text = "</ref>", .id = 151675 },
    .{ .text = "<box>", .id = 151676 },
    .{ .text = "</box>", .id = 151677 },
    .{ .text = "<|action_start|>", .id = 151678 },
    .{ .text = "<|action_end|>", .id = 151679 },
    .{ .text = "<|plugin|>", .id = 151680 },
    .{ .text = "<|interpreter|>", .id = 151681 },
};

/// A tokenizer for this family: the Qwen2.5 merge table plus SenseNova's own
/// specials. Both halves matter — the vocabulary is shared with Qwen3 but the
/// merge table is not (`Tokenizer.initQwen25`), and `<img>` terminates every
/// prompt.
pub fn initTokenizer(gpa: std.mem.Allocator) !tokenizer_mod.Tokenizer {
    var tok = try tokenizer_mod.Tokenizer.initQwen25(gpa);
    errdefer tok.deinit();
    try tok.addSpecials(&extra_specials);
    return tok;
}

/// The system prompt ComfyUI's `SenseNovaTokenizer` wraps every conditional
/// prompt in. It is part of the conditioning, not a nicety: the model was tuned
/// with it, and dropping it changes every render.
pub const system_message =
    "You are an image generation and editing assistant that accurately understands and executes " ++
    "user intent.\n\nYou support two modes:\n\n1. Think Mode:\nIf the task requires reasoning, you " ++
    "MUST start with a <think></think> block. Put all reasoning inside the block using plain text. " ++
    "DO NOT include any image tags. Keep it reasonable and directly useful for producing the final " ++
    "image.\n\n2. Non-Think Mode:\nIf no reasoning is needed, directly produce the final image.\n\n" ++
    "Task Types:\n\nA. Text-to-Image Generation:\n" ++
    "- Generate a high-quality image based on the user's description.\n" ++
    "- Ensure visual clarity, semantic consistency, and completeness.\n" ++
    "- DO NOT introduce elements that contradict or override the user's intent.\n\n" ++
    "B. Image Editing:\n" ++
    "- Use the provided image(s) as input or reference for modification or transformation.\n" ++
    "- The result can be an edited image or a new image based on the reference(s).\n" ++
    "- Preserve all unspecified attributes unless explicitly changed.\n\n" ++
    "General Rules:\n" ++
    "- For any visible text in the image, follow the language specified for the rendered text in " ++
    "the user's description, not the language of the prompt. If no language is specified, use the " ++
    "user's input language.";

/// The prompt string for `text`, or the unconditional one when it is empty.
///
/// The empty case is NOT the same template with an empty user turn: it drops the
/// system message and the primed thought block entirely, so a negative branch is
/// a much shorter prefix than a positive one.
pub fn buildPrompt(gpa: std.mem.Allocator, text: []const u8) ![]u8 {
    if (text.len == 0) {
        return gpa.dupe(u8, "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n<img>");
    }
    return std.fmt.allocPrint(gpa,
        "<|im_start|>system\n{s}<|im_end|>\n" ++
        "<|im_start|>user\n{s}<|im_end|>\n" ++
        "<|im_start|>assistant\n<think>\n\n</think>\n\n<img>", .{ system_message, text });
}

/// `buildPrompt` then tokenize, with the pad token dropped the way
/// `SenseNovaTokenizer.tokenize_with_weights` drops it. Caller frees.
pub fn tokenize(
    gpa: std.mem.Allocator,
    tok: *const tokenizer_mod.Tokenizer,
    text: []const u8,
) ![]u32 {
    const prompt = try buildPrompt(gpa, text);
    defer gpa.free(prompt);
    var ids: std.ArrayList(u32) = .empty;
    errdefer ids.deinit(gpa);
    try tok.encode(gpa, prompt, &ids);
    var n: usize = 0;
    for (ids.items) |id| {
        if (id == tokenizer_mod.pad_token) continue;
        ids.items[n] = id;
        n += 1;
    }
    ids.shrinkRetainingCapacity(n);
    return ids.toOwnedSlice(gpa);
}

/// The understanding tower's input normalization: ImageNet mean/std over [0, 1].
///
/// The GENERATION tower takes raw [-1, 1] instead. The two towers have identical
/// convolution shapes, so swapping the two renders a plausible wrong image rather
/// than failing.
pub const imagenet_mean = [3]f32{ 0.485, 0.456, 0.406 };
pub const imagenet_std = [3]f32{ 0.229, 0.224, 0.225 };

/// A reference picture for image editing: planar `[3][h][w]` in [0, 1].
pub const RefImage = struct {
    rgb: []const f32,
    h: usize,
    w: usize,

    /// Tokens this reference contributes, `(rows, cols)`.
    pub fn grid(self: RefImage, cfg: Config) [2]usize {
        const px = cfg.tokenPx();
        return .{
            @max(1, std.math.divCeil(usize, self.h, px) catch unreachable),
            @max(1, std.math.divCeil(usize, self.w, px) catch unreachable),
        };
    }
};

/// The token ids of one prompt plus the position and mask data they imply.
///
/// One struct rather than four returns because the four are derived together and
/// are wrong apart: the rope indexes are read off the ids, and the mask is read
/// off the indexes.
pub const PromptLayout = struct {
    ids: []u32,
    pos_t: []usize,
    pos_h: []usize,
    pos_w: []usize,
    /// Per-query EXCLUSIVE key bound, or null for a prompt with no references,
    /// which is plain causal.
    kv_end: ?[]u32,
    /// Row of each `<IMG_CONTEXT>` token, in order, i.e. where the reference
    /// tower's output rows go.
    img_slots: []u32,
    /// The rope t index every GENERATED token takes.
    time: u32,

    pub fn deinit(self: *PromptLayout, gpa: std.mem.Allocator) void {
        gpa.free(self.ids);
        gpa.free(self.pos_t);
        gpa.free(self.pos_h);
        gpa.free(self.pos_w);
        if (self.kv_end) |e| gpa.free(e);
        gpa.free(self.img_slots);
        self.* = undefined;
    }
};

/// The layout of a prompt with no reference pictures: t runs 0..n-1, the row and
/// column axes are zero, and the mask is plain causal.
pub fn plainLayout(gpa: std.mem.Allocator, ids: []const u32) !PromptLayout {
    const n = ids.len;
    var out: PromptLayout = .{
        .ids = try gpa.dupe(u32, ids),
        .pos_t = try gpa.alloc(usize, n),
        .pos_h = try gpa.alloc(usize, n),
        .pos_w = try gpa.alloc(usize, n),
        .kv_end = null,
        .img_slots = &.{},
        .time = @intCast(n),
    };
    for (out.pos_t, 0..) |*p, i| p.* = i;
    @memset(out.pos_h, 0);
    @memset(out.pos_w, 0);
    out.img_slots = try gpa.alloc(u32, 0);
    return out;
}

/// The ids `<IMG_CONTEXT>` block for one reference, and the label that precedes
/// it when there is more than one. The label ids are `Image`, `-`, the decimal
/// digits and `:`, spelled by id because that is how the reference spells them.
const tok_image_label: u32 = 1906;
const tok_hyphen: u32 = 12;
const tok_digit_zero: u32 = 15;
const tok_colon: u32 = 25;
const tok_newline: u32 = 198;
const tok_user: u32 = 872;
const tok_assistant: u32 = 77091;

/// Splice reference blocks into a tokenized prompt and derive the positions and
/// the block-causal bound.
///
/// `image_only` is the NEGATIVE branch of an edit: it drops the prompt entirely
/// and presents the pictures alone, so a negative conditioning under editing is
/// not the unconditional text prompt.
pub fn editLayout(
    gpa: std.mem.Allocator,
    base_ids: []const u32,
    grids: []const [2]usize,
    image_only: bool,
) !PromptLayout {
    var ids: std.ArrayList(u32) = .empty;
    errdefer ids.deinit(gpa);

    if (image_only) {
        try ids.appendSlice(gpa, &.{ tok_im_start, tok_user, tok_newline });
        for (grids) |g| try appendBlock(gpa, &ids, g);
        try ids.appendSlice(gpa, &.{ tok_im_end, tok_newline, tok_im_start, tok_assistant, tok_newline, tok_img_start });
    } else {
        // After the SECOND `<|im_start|>` plus its role and newline, i.e. at the
        // start of the user's own content.
        var seen: usize = 0;
        var at: usize = base_ids.len;
        for (base_ids, 0..) |id, i| {
            if (id != tok_im_start) continue;
            seen += 1;
            if (seen == 2) {
                at = i + 3;
                break;
            }
        }
        try ids.appendSlice(gpa, base_ids[0..at]);
        for (grids, 0..) |g, gi| {
            if (grids.len > 1) try appendLabel(gpa, &ids, gi);
            try appendBlock(gpa, &ids, g);
            try ids.append(gpa, tok_newline);
        }
        try ids.appendSlice(gpa, base_ids[at..]);
    }

    const n = ids.items.len;
    var out: PromptLayout = .{
        .ids = try ids.toOwnedSlice(gpa),
        .pos_t = try gpa.alloc(usize, n),
        .pos_h = try gpa.alloc(usize, n),
        .pos_w = try gpa.alloc(usize, n),
        .kv_end = try gpa.alloc(u32, n),
        .img_slots = &.{},
        .time = 0,
    };
    errdefer out.deinit(gpa);

    // `time_indexes = cumsum(after_an_img_start + not_a_context_token) - 1`: a
    // reference's context tokens all share ONE t index, and the `<img>` that opens
    // the block bumps it once more so the block does not share it with the text
    // before it.
    var acc: isize = -1;
    var slots: std.ArrayList(u32) = .empty;
    errdefer slots.deinit(gpa);
    for (out.ids, 0..) |id, i| {
        const shift: isize = if (i > 0 and out.ids[i - 1] == tok_img_start) 1 else 0;
        const not_img: isize = if (id != tok_img_context) 1 else 0;
        acc += shift + not_img;
        out.pos_t[i] = @intCast(@max(acc, 0));
        out.pos_h[i] = 0;
        out.pos_w[i] = 0;
        if (id == tok_img_context) try slots.append(gpa, @intCast(i));
    }
    out.img_slots = try slots.toOwnedSlice(gpa);
    out.time = @intCast(out.pos_t[n - 1] + 1);

    {
        var at: usize = 0;
        for (grids) |g| {
            for (0..g[0] * g[1]) |j| {
                const slot = out.img_slots[at + j];
                out.pos_h[slot] = j / g[1];
                out.pos_w[slot] = j % g[1];
            }
            at += g[0] * g[1];
        }
        if (at != out.img_slots.len) return error.ReferenceGridMismatch;
    }

    // Block-causal: query i sees everything up to its own position AND the whole
    // of its own t block. The t index never decreases, so a block is contiguous
    // and the allowed set stays one range.
    var i: usize = 0;
    while (i < n) {
        var j = i;
        while (j + 1 < n and out.pos_t[j + 1] == out.pos_t[i]) j += 1;
        for (i..j + 1) |q| out.kv_end.?[q] = @intCast(j + 1);
        i = j + 1;
    }
    return out;
}

fn appendBlock(gpa: std.mem.Allocator, ids: *std.ArrayList(u32), g: [2]usize) !void {
    try ids.append(gpa, tok_img_start);
    try ids.appendNTimes(gpa, tok_img_context, g[0] * g[1]);
    try ids.append(gpa, tok_img_end);
}

fn appendLabel(gpa: std.mem.Allocator, ids: *std.ArrayList(u32), index: usize) !void {
    try ids.appendSlice(gpa, &.{ tok_image_label, tok_hyphen });
    var buf: [8]u8 = undefined;
    const digits = std.fmt.bufPrint(&buf, "{d}", .{index + 1}) catch unreachable;
    for (digits) |d| try ids.append(gpa, tok_digit_zero + (d - '0'));
    try ids.append(gpa, tok_colon);
}

// --- schedule and noise -----------------------------------------------------

/// ComfyUI's `time_snr_shift`: `shift * v / (1 + (shift - 1) * v)`.
pub fn timeSnrShift(shift: f32, v: f32) f32 {
    if (shift == 1.0) return v;
    return shift * v / (1.0 + (shift - 1.0) * v);
}

/// The sigma schedule, descending from 1 to 0 over `steps + 1` entries
/// (`sampling.upstream_sigmas`).
pub fn sigmaSchedule(gpa: std.mem.Allocator, steps: usize, shift: f32) ![]f32 {
    const out = try gpa.alloc(f32, steps + 1);
    for (out, 0..) |*s, i| {
        const base = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(steps));
        s.* = timeSnrShift(shift, 1.0 - base);
    }
    return out;
}

/// The standard deviation the initial noise is drawn at:
/// `min(sqrt(tokens / 64), 16)` over the token grid.
///
/// Missing it does not degrade the image, it renders noise. It also feeds the
/// second scalar embedder, divided by the cap.
pub fn resolutionNoiseScale(cfg: Config, h: usize, w: usize) f32 {
    const px = cfg.tokenPx();
    const th = std.math.divCeil(usize, h, px) catch unreachable;
    const tw = std.math.divCeil(usize, w, px) catch unreachable;
    const n: f32 = @floatFromInt(th * tw);
    return @min(@sqrt(n / cfg.noise_base_seq), cfg.noise_max);
}

/// The padded extent of one axis: at least 16 px, then rounded up to the token
/// size. `_pad_to_merged_patch_size` does this with a replicate pad followed by a
/// circular one, so the two stages differ in how they FILL, not in how far.
pub fn paddedExtent(cfg: Config, n: usize) usize {
    const floor_n = @max(n, cfg.patch);
    const px = cfg.tokenPx();
    return (floor_n + px - 1) / px * px;
}

/// Pad a planar `[c][h][w]` canvas to the token grid. Replicate up to 16 px on
/// each axis, then CIRCULAR (wrap) up to the multiple of 32. Caller frees.
pub fn padCanvas(
    gpa: std.mem.Allocator,
    cfg: Config,
    x: []const f32,
    c: usize,
    h: usize,
    w: usize,
) ![]f32 {
    const ph = paddedExtent(cfg, h);
    const pw = paddedExtent(cfg, w);
    std.debug.assert(x.len == c * h * w);
    const out = try gpa.alloc(f32, c * ph * pw);
    errdefer gpa.free(out);
    // The replicate stage clamps to the last real row/column; the circular stage
    // wraps within the replicated extent, which is what torch does when the two
    // pads are applied in sequence.
    const rh = @max(h, cfg.patch);
    const rw = @max(w, cfg.patch);
    for (0..c) |ci| {
        const src = x[ci * h * w ..][0 .. h * w];
        const dst = out[ci * ph * pw ..][0 .. ph * pw];
        for (0..ph) |y| {
            const ry = if (y < rh) y else y % rh;
            const sy = @min(ry, h - 1);
            for (0..pw) |xx| {
                const rx = if (xx < rw) xx else xx % rw;
                const sx = @min(rx, w - 1);
                dst[y * pw + xx] = src[sy * w + sx];
            }
        }
    }
    return out;
}

/// DIAGNOSTIC: stage magnitudes on the CPU path (`TP_SENSENOVA_TRACE`).
///
/// The device twin prints the same lines for the same stages, so a device
/// mismatch is localized by diffing two runs rather than by bisecting kernels: the
/// generation pass is a vision tower, a trunk and a convolutional head, and each
/// fails for a different reason.
var trace_on: ?bool = null;

pub fn traceActs(what: []const u8, i: usize, x: []const f32) void {
    if (trace_on == null) trace_on = std.c.getenv("TP_SENSENOVA_TRACE") != null;
    if (!trace_on.?) return;
    var mx: f32 = 0;
    var sum: f64 = 0;
    for (x) |v| {
        mx = @max(mx, @abs(v));
        sum += @as(f64, v) * @as(f64, v);
    }
    std.debug.print("[sensenova-cpu] {s:<10} {d:>2}  max|x| {d:12.4}  rms {d:10.5}  n {d}\n", .{ what, i, mx, @sqrt(sum / @as(f64, @floatFromInt(x.len))), x.len });
}

/// The canvas in [-1, 1] to RGB8, ComfyUI's `((v + 1) / 2).clamp(0, 1) * 255`.
/// This is the whole of "decode" for this family, and also its live preview,
/// because there is no VAE between the two.
pub fn canvasToRgb(rgb_out: []u8, z: []const f32, plane: usize) void {
    std.debug.assert(rgb_out.len >= plane * 3 and z.len >= latent_channels * plane);
    for (0..plane) |p| {
        inline for (0..latent_channels) |c| {
            const u = std.math.clamp((z[c * plane + p] + 1.0) * 0.5, 0.0, 1.0) * 255.0;
            rgb_out[p * 3 + c] = @intFromFloat(u);
        }
    }
}

// --- weights ----------------------------------------------------------------

const Lin = struct {
    w: Weight,
    b: ?[]const f32 = null,
};

/// One MoT stream's weights for one layer: the base (understanding) copy or the
/// `_mot_gen` (generation) copy. The two are identical in shape and are never
/// mixed within a pass.
pub const Stream = struct {
    in_norm: []const f32,
    q: Weight,
    k: Weight,
    v: Weight,
    o: Weight,
    /// Per-head RMSNorm scales over the head dim's two halves: `q_norm` covers
    /// the sequence span, `q_norm_hw` the row-and-column span.
    q_norm: []const f32,
    q_norm_hw: []const f32,
    k_norm: []const f32,
    k_norm_hw: []const f32,
    /// `q_norm ++ q_norm_hw` and `k_norm ++ k_norm_hw` as one head-wide vector
    /// each. The two spans are contiguous and equally wide, so a device group
    /// RMSNorm over half-heads normalizes both in one launch against this.
    q_norm_pair: []const f32,
    k_norm_pair: []const f32,
    post_norm: []const f32,
    gate: Weight,
    up: Weight,
    down: Weight,
};

const Layer = struct {
    base: Stream,
    gen: Stream,
};

/// Convolution weights arrive in `ops.conv`'s `[co][kh][kw][ci]` patch order and
/// materialized to f32; the four here are 30M parameters between them.
pub const Conv = ops.conv.Conv2d;

const Vision = struct {
    patch: Conv,
    dense: Conv,
};

/// The prefix pass's output: every layer's keys and values for the text tokens,
/// plus the sequence position the image tokens then take.
///
/// Token-major within a layer (`[seq][kv_dim]`), not the reference's head-major
/// layout, because the generation pass concatenates the image's own keys onto
/// the end of it and `ops.attention` reads tokens as rows.
pub const Prefix = struct {
    kv: []f32,
    seq: usize,
    kv_dim: usize,
    n_layers: usize,
    /// The t rope index every image token takes: the number of text tokens.
    time: u32,

    pub fn key(self: Prefix, layer: usize) []const f32 {
        return self.kv[(layer * 2) * self.seq * self.kv_dim ..][0 .. self.seq * self.kv_dim];
    }
    pub fn value(self: Prefix, layer: usize) []const f32 {
        return self.kv[(layer * 2 + 1) * self.seq * self.kv_dim ..][0 .. self.seq * self.kv_dim];
    }

    pub fn deinit(self: *Prefix, gpa: std.mem.Allocator) void {
        gpa.free(self.kv);
        self.* = undefined;
    }
};

pub const Model = struct {
    arena: std.heap.ArenaAllocator,
    cfg: Config,

    embed: Weight,
    /// Loaded but unused for image generation, which is why ComfyUI drops it.
    /// Kept so the understanding half stays a complete language model.
    lm_head: ?Weight,
    layers: []Layer,
    norm: []const f32,
    norm_gen: []const f32,

    /// The understanding tower, for reference images. Null when the checkpoint
    /// ships without it.
    vision: ?Vision,
    vision_gen: Vision,
    fm1: Conv,
    fm2: Conv,
    t_mlp0: Lin,
    t_mlp2: Lin,
    ns_mlp0: Lin,
    ns_mlp2: Lin,

    /// The seven linears per layer the generation forward runs, in execution
    /// order. The one list every support scan and GEMM plan reads.
    device_lins: []const Weight,
    /// The same for the prefix pass's base stream.
    prefix_lins: []const Weight,

    /// The LoRA sidecars, borrowed from the session, or null. Every forward
    /// reads it: the device arms hand it to `lin_cuda.Plan` and the host arm
    /// looks each weight up as it multiplies it. The turbo LoRAs patch only the
    /// `_mot_gen` copy, but nothing here assumes that.
    lora: ?*const lora_mod.Stack = null,

    pub fn load(gpa: std.mem.Allocator, store: WeightStore, cfg: Config) !Model {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();
        const l = Loader{ .store = store, .alloc = alloc, .cfg = cfg };

        const layers = try alloc.alloc(Layer, cfg.n_layers);
        for (layers, 0..) |*layer, i| {
            layer.base = try l.stream(i, "");
            layer.gen = try l.stream(i, "_mot_gen");
        }
        const device_lins = try alloc.alloc(Weight, cfg.n_layers * 7);
        const prefix_lins = try alloc.alloc(Weight, cfg.n_layers * 7);
        for (layers, 0..) |*layer, i| {
            device_lins[i * 7 ..][0..7].* = streamLins(&layer.gen);
            prefix_lins[i * 7 ..][0..7].* = streamLins(&layer.base);
        }

        // Every field into a local first: `.arena = arena` in a struct literal
        // copies the arena's state before the later fields allocate into it, and
        // the nodes added after that copy are then never freed.
        const embed = try l.mat("language_model.model.embed_tokens.weight", .{}, cfg.vocab, cfg.dim);
        // ComfyUI drops the head outright, so a converted checkpoint may not carry
        // one. Absent is not an error and must not log like one.
        const lm_head = if (l.has("language_model.lm_head.weight"))
            try l.mat("language_model.lm_head.weight", .{}, cfg.vocab, cfg.dim)
        else
            null;
        const norm = try l.vec("language_model.model.norm.weight", .{}, cfg.dim);
        const norm_gen = try l.vec("language_model.model.norm_mot_gen.weight", .{}, cfg.dim);
        const vision = if (l.has("vision_model.embeddings.patch_embedding.weight"))
            try l.vision("vision_model")
        else
            null;
        const vision_gen = try l.vision("fm_modules.vision_model_mot_gen");
        const fm1 = try l.conv("fm_modules.fm_head.conv1", cfg.fmC1(), cfg.fmC1(), 3, 1, 1);
        const fm2 = try l.conv("fm_modules.fm_head.conv2", 192, cfg.fmC2(), 3, 1, 1);
        const t_mlp0 = try l.linear("fm_modules.timestep_embedder.mlp.0", cfg.dim, cfg.t_freq);
        const t_mlp2 = try l.linear("fm_modules.timestep_embedder.mlp.2", cfg.dim, cfg.dim);
        const ns_mlp0 = try l.linear("fm_modules.noise_scale_embedder.mlp.0", cfg.dim, cfg.t_freq);
        const ns_mlp2 = try l.linear("fm_modules.noise_scale_embedder.mlp.2", cfg.dim, cfg.dim);

        return .{
            .arena = arena,
            .cfg = cfg,
            .embed = embed,
            .lm_head = lm_head,
            .layers = layers,
            .norm = norm,
            .norm_gen = norm_gen,
            .vision = vision,
            .vision_gen = vision_gen,
            .fm1 = fm1,
            .fm2 = fm2,
            .t_mlp0 = t_mlp0,
            .t_mlp2 = t_mlp2,
            .ns_mlp0 = ns_mlp0,
            .ns_mlp2 = ns_mlp2,
            .device_lins = device_lins,
            .prefix_lins = prefix_lins,
        };
    }

    pub fn deinit(self: *Model) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// Embedding rows for `ids` into `[ids.len][dim]`, dequantizing from whatever
    /// the table's storage is. The device arm needs the same gather.
    pub fn embedInto(self: *const Model, ids: []const u32, x: []f32) !void {
        std.debug.assert(x.len == ids.len * self.cfg.dim);
        return qwen3.embedTokens(self.embed, ids, x);
    }

    // --- the prefix (understanding) pass ------------------------------------

    /// The embedded prompt, with each reference picture's tower output written
    /// over its `<IMG_CONTEXT>` rows. Caller frees.
    pub fn embedPrompt(
        self: *const Model,
        io: std.Io,
        gpa: std.mem.Allocator,
        layout: PromptLayout,
        refs: []const RefImage,
    ) ![]f32 {
        const cfg = self.cfg;
        const x = try gpa.alloc(f32, layout.ids.len * cfg.dim);
        errdefer gpa.free(x);
        try self.embedInto(layout.ids, x);
        var at: usize = 0;
        for (refs) |ref| {
            const tokens = try self.visionUnderstand(io, gpa, ref);
            defer gpa.free(tokens);
            const n = tokens.len / cfg.dim;
            traceActs("ref_tok", at, tokens);
            if (at + n > layout.img_slots.len) return error.ReferenceGridMismatch;
            for (0..n) |j| {
                @memcpy(x[@as(usize, layout.img_slots[at + j]) * cfg.dim ..][0..cfg.dim], tokens[j * cfg.dim ..][0..cfg.dim]);
            }
            at += n;
        }
        if (at != layout.img_slots.len) return error.ReferenceGridMismatch;
        return x;
    }

    /// One reference picture through the UNDERSTANDING tower: ImageNet
    /// normalization, then the pad-to-32, then the same patch-and-merge the
    /// generation tower runs with its own weights. Caller frees.
    pub fn visionUnderstand(
        self: *const Model,
        io: std.Io,
        gpa: std.mem.Allocator,
        ref: RefImage,
    ) ![]f32 {
        const cfg = self.cfg;
        const tower = self.vision orelse return error.NoUnderstandingTower;
        std.debug.assert(ref.rgb.len == latent_channels * ref.h * ref.w);
        const norm = try gpa.alloc(f32, ref.rgb.len);
        defer gpa.free(norm);
        const plane = ref.h * ref.w;
        for (0..latent_channels) |c| {
            const inv = 1.0 / imagenet_std[c];
            for (0..plane) |i| norm[c * plane + i] = (ref.rgb[c * plane + i] - imagenet_mean[c]) * inv;
        }
        const padded = try padCanvas(gpa, cfg, norm, latent_channels, ref.h, ref.w);
        defer gpa.free(padded);
        return self.visionForward(io, gpa, tower, padded, paddedExtent(cfg, ref.h), paddedExtent(cfg, ref.w));
    }

    /// Run the prompt through the base stream and keep every layer's keys and
    /// values. This is the whole of `Session.encode` for this family: there is no
    /// separate text encoder, and the conditioning IS this cache.
    pub fn prefixForward(
        self: *const Model,
        io: std.Io,
        gpa: std.mem.Allocator,
        layout: PromptLayout,
        refs: []const RefImage,
        cancel: ?*std.atomic.Value(bool),
    ) !Prefix {
        const cfg = self.cfg;
        const seq = layout.ids.len;
        std.debug.assert(seq > 0);

        const prev_tok = ops.cancel.token;
        ops.cancel.token = cancel;
        defer ops.cancel.token = prev_tok;

        var pre: Prefix = .{
            .kv = try gpa.alloc(f32, cfg.n_layers * 2 * seq * cfg.kvDim()),
            .seq = seq,
            .kv_dim = cfg.kvDim(),
            .n_layers = cfg.n_layers,
            .time = layout.time,
        };
        errdefer pre.deinit(gpa);

        const x = try self.embedPrompt(io, gpa, layout, refs);
        defer gpa.free(x);
        traceActs("embed", 0, x);

        var rope = try self.ropeTables(gpa, maxOf(layout.pos_t) + 1, maxOf(layout.pos_h) + 1, maxOf(layout.pos_w) + 1);
        defer rope.deinit(gpa);

        var ws = try Workspace.init(gpa, cfg, seq, seq);
        defer ws.deinit(gpa);

        for (self.layers, 0..) |*layer, li| {
            if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
            const k_dst = pre.kv[(li * 2) * seq * cfg.kvDim() ..][0 .. seq * cfg.kvDim()];
            const v_dst = pre.kv[(li * 2 + 1) * seq * cfg.kvDim() ..][0 .. seq * cfg.kvDim()];
            try self.attnBlock(io, gpa, &ws, &layer.base, x, seq, .{
                .pos_t = layout.pos_t,
                .pos_h = layout.pos_h,
                .pos_w = layout.pos_w,
                .rope = rope,
                .causal = true,
                .kv_end = layout.kv_end,
                .prefix_k = null,
                .prefix_v = null,
                .k_out = k_dst,
                .v_out = v_dst,
            });
            try self.ffnBlock(io, gpa, &ws, &layer.base, x, seq);
            traceActs("prefix", li, x);
        }
        return pre;
    }

    // --- the generation pass ------------------------------------------------

    /// Predict the flow-matching velocity for one canvas.
    ///
    /// `x` and `out` are planar `[3][h][w]` in [-1, 1]; `t` is the model's own
    /// timestep, `1 - sigma`.
    pub fn predict(
        self: *const Model,
        io: std.Io,
        gpa: std.mem.Allocator,
        out: []f32,
        x: []const f32,
        h: usize,
        w: usize,
        pre: Prefix,
        t: f32,
        cancel: ?*std.atomic.Value(bool),
    ) !void {
        const cfg = self.cfg;
        std.debug.assert(x.len == latent_channels * h * w);
        std.debug.assert(out.len == x.len);
        std.debug.assert(pre.n_layers == cfg.n_layers);

        const prev_tok = ops.cancel.token;
        ops.cancel.token = cancel;
        defer ops.cancel.token = prev_tok;

        const ph = paddedExtent(cfg, h);
        const pw = paddedExtent(cfg, w);
        const xp = try padCanvas(gpa, cfg, x, latent_channels, h, w);
        defer gpa.free(xp);

        const px = cfg.tokenPx();
        const th = ph / px;
        const tw = pw / px;
        const n_img = th * tw;

        var img = try self.visionTokens(io, gpa, xp, ph, pw);
        defer gpa.free(img);
        traceActs("vision", 0, img);

        // One vector for the whole canvas: the timestep plus the resolution's own
        // noise scale, added to every token row.
        const tvec = try self.timeVector(io, gpa, t, ph, pw);
        defer gpa.free(tvec);
        traceActs("tvec", 0, tvec);
        for (0..n_img) |i| {
            for (img[i * cfg.dim ..][0..cfg.dim], tvec) |*v, tv| v.* += tv;
        }
        traceActs("trunk_in", 0, img);

        // Image tokens: one shared t index past the whole prefix, then the row
        // and column of the token within the canvas.
        const pos_t = try gpa.alloc(usize, n_img);
        defer gpa.free(pos_t);
        @memset(pos_t, pre.time);
        const pos_h = try gpa.alloc(usize, n_img);
        defer gpa.free(pos_h);
        const pos_w = try gpa.alloc(usize, n_img);
        defer gpa.free(pos_w);
        for (0..n_img) |i| {
            pos_h[i] = i / tw;
            pos_w[i] = i % tw;
        }

        var rope = try self.ropeTables(gpa, pre.time + 1, th, tw);
        defer rope.deinit(gpa);

        var ws = try Workspace.init(gpa, cfg, n_img, pre.seq + n_img);
        defer ws.deinit(gpa);

        for (self.layers, 0..) |*layer, li| {
            if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
            try self.attnBlock(io, gpa, &ws, &layer.gen, img, n_img, .{
                .pos_t = pos_t,
                .pos_h = pos_h,
                .pos_w = pos_w,
                .rope = rope,
                .causal = false,
                .prefix_k = pre.key(li),
                .prefix_v = pre.value(li),
                .k_out = null,
                .v_out = null,
            });
            try self.ffnBlock(io, gpa, &ws, &layer.gen, img, n_img);
            traceActs("layer", li, img);
        }
        ops.norm.rmsNorm(img, img, self.norm_gen, cfg.norm_eps);
        traceActs("norm", 0, img);

        const pred = try gpa.alloc(f32, latent_channels * ph * pw);
        defer gpa.free(pred);
        try self.fmHead(io, gpa, pred, img, th, tw);
        traceActs("pred", 0, pred);

        // The head predicts x0; the sampler wants the trajectory derivative.
        const denom = @max(1.0 - t, cfg.velocity_eps);
        for (0..latent_channels) |c| {
            const src = xp[c * ph * pw ..];
            const p = pred[c * ph * pw ..];
            const dst = out[c * h * w ..];
            for (0..h) |y| {
                for (0..w) |xx| {
                    dst[y * w + xx] = (src[y * pw + xx] - p[y * pw + xx]) / denom;
                }
            }
        }
    }

    /// The scalar conditioning: `timestep_embedder(t) + noise_scale_embedder(scale / 16)`.
    /// Caller frees.
    ///
    /// The noise-scale term reads the PADDED extent, while the sampler's own
    /// initial-noise scale reads the unpadded one. They agree on every canvas
    /// that is already a multiple of 32 and differ on the rest; this is the
    /// reference's behaviour, not a rounding choice.
    pub fn timeVector(
        self: *const Model,
        io: std.Io,
        gpa: std.mem.Allocator,
        t: f32,
        padded_h: usize,
        padded_w: usize,
    ) ![]f32 {
        const cfg = self.cfg;
        const out = try gpa.alloc(f32, cfg.dim);
        errdefer gpa.free(out);
        const scratch = try gpa.alloc(f32, cfg.dim);
        defer gpa.free(scratch);
        try self.scalarEmbed(io, gpa, out, t, self.t_mlp0, self.t_mlp2);
        const ns = resolutionNoiseScale(cfg, padded_h, padded_w) / cfg.noise_max;
        try self.scalarEmbed(io, gpa, scratch, ns, self.ns_mlp0, self.ns_mlp2);
        for (out, scratch) |*v, s| v.* += s;
        return out;
    }

    fn scalarEmbed(self: *const Model, io: std.Io, gpa: std.mem.Allocator, out: []f32, v: f32, m0: Lin, m2: Lin) !void {
        const cfg = self.cfg;
        const freq = try gpa.alloc(f32, cfg.t_freq);
        defer gpa.free(freq);
        ops.rope.sinCosEmbedding(freq, v, cfg.t_max_period);
        const hidden = try gpa.alloc(f32, cfg.dim);
        defer gpa.free(hidden);
        try ops.matmul.matmul(io, gpa, hidden, freq, 1, m0.w, m0.b);
        ops.act.silu(hidden);
        try ops.matmul.matmul(io, gpa, out, hidden, 1, m2.w, m2.b);
    }

    /// The generation vision tower: patchify the padded canvas into `[n_img][dim]`
    /// token rows. Caller frees.
    pub fn visionTokens(
        self: *const Model,
        io: std.Io,
        gpa: std.mem.Allocator,
        xp: []const f32,
        ph: usize,
        pw: usize,
    ) ![]f32 {
        return self.visionForward(io, gpa, self.vision_gen, xp, ph, pw);
    }

    fn visionForward(
        self: *const Model,
        io: std.Io,
        gpa: std.mem.Allocator,
        tower: Vision,
        xp: []const f32,
        ph: usize,
        pw: usize,
    ) ![]f32 {
        const cfg = self.cfg;
        std.debug.assert(ph % cfg.tokenPx() == 0 and pw % cfg.tokenPx() == 0);

        const cl = try gpa.alloc(f32, latent_channels * ph * pw);
        defer gpa.free(cl);
        planarToChannelLast(cl, xp, latent_channels, ph * pw);

        const gh = ph / cfg.patch;
        const gw = pw / cfg.patch;
        const cells = try gpa.alloc(f32, gh * gw * cfg.vis_dim);
        defer gpa.free(cells);
        try ops.conv.conv2d(io, gpa, cells, cl, ph, pw, tower.patch);
        ops.act.geluErf(cells);

        // Interleaved 2-D rope: the first half of the channels rotates against
        // the cell's column, the second half against its row.
        const half = cfg.vis_dim / 2;
        const cols = try gpa.alloc(usize, gh * gw);
        defer gpa.free(cols);
        const rows = try gpa.alloc(usize, gh * gw);
        defer gpa.free(rows);
        for (0..gh * gw) |i| {
            cols[i] = i % gw;
            rows[i] = i / gw;
        }
        var fx = try visionFreqs(gpa, gw, half, cfg.theta_vis);
        defer fx.deinit(gpa);
        var fy = try visionFreqs(gpa, gh, half, cfg.theta_vis);
        defer fy.deinit(gpa);
        ops.rope.applyInterleavedPosSpan(cells, fx, cols, cfg.vis_dim, 0, half);
        ops.rope.applyInterleavedPosSpan(cells, fy, rows, cfg.vis_dim, half, half);

        const out = try gpa.alloc(f32, (gh / cfg.merge) * (gw / cfg.merge) * cfg.dim);
        errdefer gpa.free(out);
        try ops.conv.conv2d(io, gpa, out, cells, gh, gw, tower.dense);
        return out;
    }

    /// `fm_head`: pixel shuffle by 2, a 3x3 convolution, GELU, shuffle by 2, a
    /// second 3x3 convolution, shuffle by 8. `img` is `[th*tw][dim]`, which is
    /// already the channel-last plane the shuffles read. `out` is planar
    /// `[3][th*32][tw*32]`.
    pub fn fmHead(
        self: *const Model,
        io: std.Io,
        gpa: std.mem.Allocator,
        out: []f32,
        img: []const f32,
        th: usize,
        tw: usize,
    ) !void {
        const cfg = self.cfg;
        const c1 = cfg.fmC1();
        const c2 = cfg.fmC2();
        std.debug.assert(img.len == th * tw * cfg.dim);

        const s1 = try gpa.alloc(f32, th * 2 * tw * 2 * c1);
        defer gpa.free(s1);
        pixelShuffle(s1, img, th, tw, c1, 2);

        const h1 = try gpa.alloc(f32, th * 2 * tw * 2 * c1);
        defer gpa.free(h1);
        try ops.conv.conv2d(io, gpa, h1, s1, th * 2, tw * 2, self.fm1);
        ops.act.geluErf(h1);

        const s2 = try gpa.alloc(f32, th * 4 * tw * 4 * c2);
        defer gpa.free(s2);
        pixelShuffle(s2, h1, th * 2, tw * 2, c2, 2);

        const h2 = try gpa.alloc(f32, th * 4 * tw * 4 * 192);
        defer gpa.free(h2);
        try ops.conv.conv2d(io, gpa, h2, s2, th * 4, tw * 4, self.fm2);

        const rgb = try gpa.alloc(f32, th * 32 * tw * 32 * latent_channels);
        defer gpa.free(rgb);
        pixelShuffle(rgb, h2, th * 4, tw * 4, latent_channels, 8);
        channelLastToPlanar(out, rgb, latent_channels, th * 32 * tw * 32);
    }

    // --- one layer ----------------------------------------------------------

    const AttnArgs = struct {
        pos_t: []const usize,
        pos_h: []const usize,
        pos_w: []const usize,
        rope: RopeTables,
        causal: bool,
        /// Block-causal bound for a prompt carrying reference pictures; null is
        /// plain `causal`.
        kv_end: ?[]const u32 = null,
        /// The prefix keys and values this stream attends to before its own, or
        /// null for the prefix pass itself.
        prefix_k: ?[]const f32,
        prefix_v: ?[]const f32,
        /// Where to keep this pass's own keys and values, for the prefix pass.
        k_out: ?[]f32,
        v_out: ?[]f32,
    };

    /// `y[m][w.rows] = W x + sidecars`. The one host path through a trunk linear,
    /// so a LoRA cannot be forgotten at a call site: applied at some GEMMs and
    /// not others renders a finite, plausible, wrong image with no error.
    fn linHost(
        self: *const Model,
        io: std.Io,
        gpa: std.mem.Allocator,
        y: []f32,
        x: []const f32,
        m: usize,
        w: Weight,
    ) !void {
        try ops.matmul.matmul(io, gpa, y, x, m, w, null);
        if (self.lora) |stack| try stack.applyHost(io, gpa, y, w.rows, x, m, w);
    }

    fn attnBlock(
        self: *const Model,
        io: std.Io,
        gpa: std.mem.Allocator,
        ws: *Workspace,
        s: *const Stream,
        x: []f32,
        seq: usize,
        a: AttnArgs,
    ) !void {
        const cfg = self.cfg;
        const q_dim = cfg.qDim();
        const kv_dim = cfg.kvDim();
        const normed = ws.normed[0 .. seq * cfg.dim];
        ops.norm.rmsNorm(normed, x[0 .. seq * cfg.dim], s.in_norm, cfg.norm_eps);

        const q = ws.q[0 .. seq * q_dim];
        const k = ws.k[0 .. seq * kv_dim];
        const v = ws.v[0 .. seq * kv_dim];
        try self.linHost(io, gpa, q, normed, seq, s.q);
        try self.linHost(io, gpa, k, normed, seq, s.k);
        try self.linHost(io, gpa, v, normed, seq, s.v);

        // The two per-head norms cover the head dim's halves, not the whole head.
        normSpan(q, s.q_norm, cfg.norm_eps, seq * cfg.n_heads, cfg.head_dim, 0);
        normSpan(q, s.q_norm_hw, cfg.norm_eps, seq * cfg.n_heads, cfg.head_dim, cfg.spanT());
        normSpan(k, s.k_norm, cfg.norm_eps, seq * cfg.n_kv_heads, cfg.head_dim, 0);
        normSpan(k, s.k_norm_hw, cfg.norm_eps, seq * cfg.n_kv_heads, cfg.head_dim, cfg.spanT());

        const t_span = cfg.spanT();
        const hw_span = cfg.spanHw();
        for ([_]struct { x: []f32, heads: usize }{
            .{ .x = q, .heads = cfg.n_heads },
            .{ .x = k, .heads = cfg.n_kv_heads },
        }) |arm| {
            ops.rope.applyRotateHalfPosSpan(arm.x, a.rope.t, a.pos_t, arm.heads, cfg.head_dim, 0, t_span);
            ops.rope.applyRotateHalfPosSpan(arm.x, a.rope.h, a.pos_h, arm.heads, cfg.head_dim, t_span, hw_span);
            ops.rope.applyRotateHalfPosSpan(arm.x, a.rope.w, a.pos_w, arm.heads, cfg.head_dim, t_span + hw_span, hw_span);
        }

        traceActs("q", 0, q);
        traceActs("k", 0, k);
        traceActs("v", 0, v);
        if (a.k_out) |dst| @memcpy(dst, k);
        if (a.v_out) |dst| @memcpy(dst, v);

        // The generation stream attends over the prefix's keys and then its own,
        // with no mask; the prefix pass attends causally over itself alone.
        var kk: []const f32 = k;
        var vv: []const f32 = v;
        var seq_kv = seq;
        if (a.prefix_k) |pk| {
            const pn = pk.len / kv_dim;
            seq_kv = pn + seq;
            const kcat = ws.kcat[0 .. seq_kv * kv_dim];
            const vcat = ws.vcat[0 .. seq_kv * kv_dim];
            @memcpy(kcat[0..pk.len], pk);
            @memcpy(kcat[pk.len..], k);
            @memcpy(vcat[0..pk.len], a.prefix_v.?);
            @memcpy(vcat[pk.len..], v);
            kk = kcat;
            vv = vcat;
        }

        const attn_out = ws.attn_out[0 .. seq * q_dim];
        try ops.attention.attention(io, gpa, attn_out, q, kk, vv, .{
            .seq_q = seq,
            .seq_kv = seq_kv,
            .n_heads = cfg.n_heads,
            .n_kv_heads = cfg.n_kv_heads,
            .head_dim = cfg.head_dim,
            .causal = a.causal,
            .kv_end = a.kv_end,
        });
        traceActs("attn", 0, attn_out);
        const delta = ws.delta[0 .. seq * cfg.dim];
        try self.linHost(io, gpa, delta, attn_out, seq, s.o);
        for (x[0 .. seq * cfg.dim], delta) |*xv, d| xv.* += d;
        traceActs("attn_res", 0, x[0 .. seq * cfg.dim]);
    }

    fn ffnBlock(
        self: *const Model,
        io: std.Io,
        gpa: std.mem.Allocator,
        ws: *Workspace,
        s: *const Stream,
        x: []f32,
        seq: usize,
    ) !void {
        const cfg = self.cfg;
        const normed = ws.normed[0 .. seq * cfg.dim];
        ops.norm.rmsNorm(normed, x[0 .. seq * cfg.dim], s.post_norm, cfg.norm_eps);
        const gate = ws.gate[0 .. seq * cfg.inter];
        const up = ws.up[0 .. seq * cfg.inter];
        try self.linHost(io, gpa, gate, normed, seq, s.gate);
        try self.linHost(io, gpa, up, normed, seq, s.up);
        ops.act.siluMul(gate, up);
        const delta = ws.delta[0 .. seq * cfg.dim];
        try self.linHost(io, gpa, delta, gate, seq, s.down);
        for (x[0 .. seq * cfg.dim], delta) |*xv, d| xv.* += d;
    }

    const RopeTables = struct {
        t: ops.rope.Freqs,
        h: ops.rope.Freqs,
        w: ops.rope.Freqs,

        fn deinit(self: *RopeTables, gpa: std.mem.Allocator) void {
            self.t.deinit(gpa);
            self.h.deinit(gpa);
            self.w.deinit(gpa);
        }
    };

    fn ropeTables(self: *const Model, gpa: std.mem.Allocator, n_t: usize, n_h: usize, n_w: usize) !RopeTables {
        const cfg = self.cfg;
        var t = try ops.rope.rotateHalfFreqs(gpa, n_t, cfg.spanT(), cfg.theta_t);
        errdefer t.deinit(gpa);
        var h = try ops.rope.rotateHalfFreqs(gpa, n_h, cfg.spanHw(), cfg.theta_hw);
        errdefer h.deinit(gpa);
        const w = try ops.rope.rotateHalfFreqs(gpa, n_w, cfg.spanHw(), cfg.theta_hw);
        return .{ .t = t, .h = h, .w = w };
    }
};

/// Per-forward activation buffers, sized once per pass.
const Workspace = struct {
    normed: []f32,
    delta: []f32,
    q: []f32,
    k: []f32,
    v: []f32,
    kcat: []f32,
    vcat: []f32,
    attn_out: []f32,
    gate: []f32,
    up: []f32,

    fn init(gpa: std.mem.Allocator, cfg: Config, seq: usize, seq_kv: usize) !Workspace {
        var w: Workspace = undefined;
        w.normed = try gpa.alloc(f32, seq * cfg.dim);
        errdefer gpa.free(w.normed);
        w.delta = try gpa.alloc(f32, seq * cfg.dim);
        errdefer gpa.free(w.delta);
        w.q = try gpa.alloc(f32, seq * cfg.qDim());
        errdefer gpa.free(w.q);
        w.k = try gpa.alloc(f32, seq * cfg.kvDim());
        errdefer gpa.free(w.k);
        w.v = try gpa.alloc(f32, seq * cfg.kvDim());
        errdefer gpa.free(w.v);
        w.kcat = try gpa.alloc(f32, seq_kv * cfg.kvDim());
        errdefer gpa.free(w.kcat);
        w.vcat = try gpa.alloc(f32, seq_kv * cfg.kvDim());
        errdefer gpa.free(w.vcat);
        w.attn_out = try gpa.alloc(f32, seq * cfg.qDim());
        errdefer gpa.free(w.attn_out);
        w.gate = try gpa.alloc(f32, seq * cfg.inter);
        errdefer gpa.free(w.gate);
        w.up = try gpa.alloc(f32, seq * cfg.inter);
        return w;
    }

    fn deinit(self: *Workspace, gpa: std.mem.Allocator) void {
        gpa.free(self.normed);
        gpa.free(self.delta);
        gpa.free(self.q);
        gpa.free(self.k);
        gpa.free(self.v);
        gpa.free(self.kcat);
        gpa.free(self.vcat);
        gpa.free(self.attn_out);
        gpa.free(self.gate);
        gpa.free(self.up);
        self.* = undefined;
    }
};

// --- free helpers -----------------------------------------------------------

fn maxOf(xs: []const usize) usize {
    var m: usize = 0;
    for (xs) |x| m = @max(m, x);
    return m;
}

fn concat(alloc: std.mem.Allocator, a: []const f32, b: []const f32) ![]const f32 {
    const out = try alloc.alloc(f32, a.len + b.len);
    @memcpy(out[0..a.len], a);
    @memcpy(out[a.len..], b);
    return out;
}

/// The seven linears one stream's forward runs, in execution order.
fn streamLins(s: *const Stream) [7]Weight {
    return .{ s.q, s.k, s.v, s.o, s.gate, s.up, s.down };
}

/// RMSNorm over `[off, off + w.len)` of each `row_dim`-wide head.
fn normSpan(x: []f32, w: []const f32, eps: f32, n_rows: usize, row_dim: usize, off: usize) void {
    std.debug.assert(x.len == n_rows * row_dim);
    std.debug.assert(off + w.len <= row_dim);
    for (0..n_rows) |i| {
        const span = x[i * row_dim + off ..][0..w.len];
        ops.norm.rmsNorm(span, span, w, eps);
    }
}

/// The vision embedder's frequency ladder, `theta^(-2i/dim)` over `dim/2` slots,
/// tabulated for `n` integer positions.
fn visionFreqs(gpa: std.mem.Allocator, n: usize, dim: usize, theta: f64) !ops.rope.Freqs {
    const pos = try gpa.alloc(f32, n);
    defer gpa.free(pos);
    for (pos, 0..) |*p, i| p.* = @floatFromInt(i);
    return ops.rope.fluxFreqs(gpa, pos, &.{dim}, theta);
}

/// `nn.PixelShuffle(r)` on a channel-last plane: `[h][w][c*r*r] -> [h*r][w*r][c]`,
/// with the input channel split as `c * r * r + i * r + j`.
fn pixelShuffle(out: []f32, in: []const f32, h: usize, w: usize, c: usize, r: usize) void {
    std.debug.assert(in.len == h * w * c * r * r);
    std.debug.assert(out.len == in.len);
    const ow = w * r;
    for (0..h) |y| {
        for (0..w) |x| {
            const src = in[(y * w + x) * c * r * r ..];
            for (0..c) |ci| {
                for (0..r) |i| {
                    for (0..r) |j| {
                        out[((y * r + i) * ow + (x * r + j)) * c + ci] = src[ci * r * r + i * r + j];
                    }
                }
            }
        }
    }
}

fn planarToChannelLast(out: []f32, in: []const f32, c: usize, plane: usize) void {
    std.debug.assert(out.len == c * plane and in.len == out.len);
    for (0..c) |ci| {
        for (0..plane) |p| out[p * c + ci] = in[ci * plane + p];
    }
}

fn channelLastToPlanar(out: []f32, in: []const f32, c: usize, plane: usize) void {
    std.debug.assert(out.len == c * plane and in.len == out.len);
    for (0..c) |ci| {
        for (0..plane) |p| out[ci * plane + p] = in[p * c + ci];
    }
}

// --- weight loading ---------------------------------------------------------

fn fmtName(buf: []u8, comptime fmt: []const u8, args: anytype) ![]u8 {
    var fbs = std.Io.Writer.fixed(buf);
    try fbs.print(fmt, args);
    return fbs.buffered();
}

const Loader = struct {
    store: WeightStore,
    alloc: std.mem.Allocator,
    cfg: Config,

    fn has(l: Loader, name: []const u8) bool {
        return l.store.get(name) != null;
    }

    fn mat(l: Loader, comptime fmt: []const u8, args: anytype, rows: usize, cols: usize) !Weight {
        var buf: [192]u8 = undefined;
        const nm = try fmtName(&buf, fmt, args);
        return quant_weight.load(l.alloc, l.store, nm, rows, cols, .{ .who = "sensenova" });
    }

    fn vec(l: Loader, comptime fmt: []const u8, args: anytype, len: usize) ![]f32 {
        var buf: [192]u8 = undefined;
        const nm = try fmtName(&buf, fmt, args);
        const view = l.store.get(nm) orelse {
            std.log.err("sensenova: missing tensor {s}", .{nm});
            return error.MissingTensor;
        };
        if (view.info.elemCount() != len) {
            std.log.err("sensenova: {s} has {d} elements, expected {d}", .{ nm, view.info.elemCount(), len });
            return error.ShapeMismatch;
        }
        return view.toF32Alloc(l.alloc);
    }

    fn linear(l: Loader, comptime prefix: []const u8, rows: usize, cols: usize) !Lin {
        return .{
            .w = try l.mat(prefix ++ ".weight", .{}, rows, cols),
            .b = try l.vec(prefix ++ ".bias", .{}, rows),
        };
    }

    /// A convolution, materialized to f32 in `ops.conv`'s patch order. The four
    /// convolutions here are 30M parameters between them, so the f32 copy is
    /// ~110 MB against the model's 33 GB — not worth a packed path.
    fn conv(
        l: Loader,
        comptime prefix: []const u8,
        co: usize,
        ci: usize,
        k: usize,
        stride: usize,
        pad: usize,
    ) !Conv {
        var buf: [192]u8 = undefined;
        const nm = try fmtName(&buf, prefix ++ ".weight", .{});
        const view = l.store.get(nm) orelse {
            std.log.err("sensenova: missing tensor {s}", .{nm});
            return error.MissingTensor;
        };
        if (view.info.elemCount() != co * ci * k * k) {
            std.log.err("sensenova: {s} has {d} elements, expected {d}", .{ nm, view.info.elemCount(), co * ci * k * k });
            return error.ShapeMismatch;
        }
        const torch_w = try view.toF32Alloc(l.alloc);
        defer l.alloc.free(torch_w);
        return .{
            .w = try ops.conv.packWeight(l.alloc, torch_w, co, ci, k),
            .b = try l.vec(prefix ++ ".bias", .{}, co),
            .co = co,
            .ci = ci,
            .k = k,
            .stride = stride,
            .pad = pad,
            .tag = try l.alloc.dupe(u8, nm),
        };
    }

    fn vision(l: Loader, comptime prefix: []const u8) !Vision {
        const cfg = l.cfg;
        return .{
            .patch = try l.conv(prefix ++ ".embeddings.patch_embedding", cfg.vis_dim, latent_channels, cfg.patch, cfg.patch, 0),
            .dense = try l.conv(prefix ++ ".embeddings.dense_embedding", cfg.dim, cfg.vis_dim, cfg.merge, cfg.merge, 0),
        };
    }

    fn stream(l: Loader, layer: usize, comptime sfx: []const u8) !Stream {
        const cfg = l.cfg;
        const p = "language_model.model.layers.{d}.";
        const q_norm = try l.vec(p ++ "self_attn.q_norm" ++ sfx ++ ".weight", .{layer}, cfg.spanT());
        const q_norm_hw = try l.vec(p ++ "self_attn.q_norm_hw" ++ sfx ++ ".weight", .{layer}, cfg.spanT());
        const k_norm = try l.vec(p ++ "self_attn.k_norm" ++ sfx ++ ".weight", .{layer}, cfg.spanT());
        const k_norm_hw = try l.vec(p ++ "self_attn.k_norm_hw" ++ sfx ++ ".weight", .{layer}, cfg.spanT());
        return .{
            .in_norm = try l.vec(p ++ "input_layernorm" ++ sfx ++ ".weight", .{layer}, cfg.dim),
            .q = try l.mat(p ++ "self_attn.q_proj" ++ sfx ++ ".weight", .{layer}, cfg.qDim(), cfg.dim),
            .k = try l.mat(p ++ "self_attn.k_proj" ++ sfx ++ ".weight", .{layer}, cfg.kvDim(), cfg.dim),
            .v = try l.mat(p ++ "self_attn.v_proj" ++ sfx ++ ".weight", .{layer}, cfg.kvDim(), cfg.dim),
            .o = try l.mat(p ++ "self_attn.o_proj" ++ sfx ++ ".weight", .{layer}, cfg.dim, cfg.qDim()),
            .q_norm = q_norm,
            .q_norm_hw = q_norm_hw,
            .k_norm = k_norm,
            .k_norm_hw = k_norm_hw,
            .q_norm_pair = try concat(l.alloc, q_norm, q_norm_hw),
            .k_norm_pair = try concat(l.alloc, k_norm, k_norm_hw),
            .post_norm = try l.vec(p ++ "post_attention_layernorm" ++ sfx ++ ".weight", .{layer}, cfg.dim),
            .gate = try l.mat(p ++ "mlp" ++ sfx ++ ".gate_proj.weight", .{layer}, cfg.inter, cfg.dim),
            .up = try l.mat(p ++ "mlp" ++ sfx ++ ".up_proj.weight", .{layer}, cfg.inter, cfg.dim),
            .down = try l.mat(p ++ "mlp" ++ sfx ++ ".down_proj.weight", .{layer}, cfg.dim, cfg.inter),
        };
    }
};

// --- tests ------------------------------------------------------------------

const testing = std.testing;
const test_gate = @import("../test_gate.zig");
const SafeTensors = safetensors.SafeTensors;

const ref_fixture = @embedFile("assets/sensenova_ref.safetensors");

/// The narrowed configuration `tools/gen_sensenova_fixtures.py` builds. Same
/// architecture, smaller widths: see that file for why a reduction is the only
/// exact reference available for a 33 GB model.
const tiny: Config = .{
    .dim = 128,
    .inter = 192,
    .n_layers = 2,
    .n_heads = 4,
    .n_kv_heads = 2,
    .head_dim = 16,
    .vocab = 64,
    .vis_dim = 32,
    .patch = 16,
    .merge = 2,
    .t_freq = 256,
    .t_max_period = 10000.0,
    .norm_eps = 1e-6,
    .theta_t = 5000000.0,
    .theta_hw = 10000.0,
    .theta_vis = 10000.0,
    .noise_base_seq = 64.0,
    .noise_max = 16.0,
    .velocity_eps = 0.02,
};

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

fn refIds(gpa: std.mem.Allocator, st: *const SafeTensors, name: []const u8) ![]u32 {
    const v = try st.require(name);
    const n = v.info.elemCount();
    const out = try gpa.alloc(u32, n);
    for (out, 0..) |*id, i| id.* = @intCast(std.mem.readInt(i32, v.bytes[i * 4 ..][0..4], .little));
    return out;
}

/// The reference keeps q/k as `[heads][seq][head_dim]`; every kernel here reads
/// tokens as rows. Transpose into `[seq][heads*head_dim]`.
fn headMajorToTokenMajor(out: []f32, in: []const f32, heads: usize, seq: usize, hd: usize) void {
    std.debug.assert(out.len == heads * seq * hd and in.len == out.len);
    for (0..heads) |h| {
        for (0..seq) |s| {
            @memcpy(out[(s * heads + h) * hd ..][0..hd], in[(h * seq + s) * hd ..][0..hd]);
        }
    }
}

test "the three-way rope splits the head dim 64/32/32 with two thetas" {
    const gpa = testing.allocator;
    var st = try SafeTensors.initFromSlice(gpa, ref_fixture);
    defer st.deinit();
    const cfg = u15_8b;

    const idx_v = try st.require("rope.indexes");
    const seq: usize = idx_v.info.shape.slice()[1];
    const pos = try gpa.alloc(usize, 3 * seq);
    defer gpa.free(pos);
    for (pos, 0..) |*p, i| p.* = @intCast(std.mem.readInt(i32, idx_v.bytes[i * 4 ..][0..4], .little));

    var max_t: usize = 0;
    var max_h: usize = 0;
    var max_w: usize = 0;
    for (0..seq) |i| {
        max_t = @max(max_t, pos[i]);
        max_h = @max(max_h, pos[seq + i]);
        max_w = @max(max_w, pos[2 * seq + i]);
    }
    var ft = try ops.rope.rotateHalfFreqs(gpa, max_t + 1, cfg.spanT(), cfg.theta_t);
    defer ft.deinit(gpa);
    var fh = try ops.rope.rotateHalfFreqs(gpa, max_h + 1, cfg.spanHw(), cfg.theta_hw);
    defer fh.deinit(gpa);
    var fw = try ops.rope.rotateHalfFreqs(gpa, max_w + 1, cfg.spanHw(), cfg.theta_hw);
    defer fw.deinit(gpa);

    for ([_][2][]const u8{
        .{ "rope.q_in", "rope.q_out" },
        .{ "rope.k_in", "rope.k_out" },
    }) |pair| {
        const src = try (try st.require(pair[0])).toF32Alloc(gpa);
        defer gpa.free(src);
        const want_hm = try (try st.require(pair[1])).toF32Alloc(gpa);
        defer gpa.free(want_hm);
        const heads = src.len / (seq * cfg.head_dim);

        const got = try gpa.alloc(f32, src.len);
        defer gpa.free(got);
        headMajorToTokenMajor(got, src, heads, seq, cfg.head_dim);
        const want = try gpa.alloc(f32, src.len);
        defer gpa.free(want);
        headMajorToTokenMajor(want, want_hm, heads, seq, cfg.head_dim);

        const t_span = cfg.spanT();
        const hw = cfg.spanHw();
        ops.rope.applyRotateHalfPosSpan(got, ft, pos[0..seq], heads, cfg.head_dim, 0, t_span);
        ops.rope.applyRotateHalfPosSpan(got, fh, pos[seq..][0..seq], heads, cfg.head_dim, t_span, hw);
        ops.rope.applyRotateHalfPosSpan(got, fw, pos[2 * seq ..][0..seq], heads, cfg.head_dim, t_span + hw, hw);

        const rel = relL2(want, got);
        errdefer std.debug.print("{s}: rel L2 {e:.4}\n", .{ pair[0], rel });
        try testing.expect(rel < 1e-6);
    }
}

test "the vision embedder rotates interleaved pairs, column then row" {
    const gpa = testing.allocator;
    var st = try SafeTensors.initFromSlice(gpa, ref_fixture);
    defer st.deinit();

    const grid_v = try st.require("visrope.grid");
    const gh: usize = @intCast(std.mem.readInt(i32, grid_v.bytes[4..8], .little));
    const gw: usize = @intCast(std.mem.readInt(i32, grid_v.bytes[8..12], .little));

    const got = try (try st.require("visrope.in")).toF32Alloc(gpa);
    defer gpa.free(got);
    const want = try (try st.require("visrope.out")).toF32Alloc(gpa);
    defer gpa.free(want);
    const n = gh * gw;
    const ch = got.len / n;
    const half = ch / 2;

    const cols = try gpa.alloc(usize, n);
    defer gpa.free(cols);
    const rows = try gpa.alloc(usize, n);
    defer gpa.free(rows);
    for (0..n) |i| {
        cols[i] = i % gw;
        rows[i] = i / gw;
    }
    var fx = try visionFreqs(gpa, gw, half, 10000.0);
    defer fx.deinit(gpa);
    var fy = try visionFreqs(gpa, gh, half, 10000.0);
    defer fy.deinit(gpa);
    ops.rope.applyInterleavedPosSpan(got, fx, cols, ch, 0, half);
    ops.rope.applyInterleavedPosSpan(got, fy, rows, ch, half, half);

    const rel = relL2(want, got);
    errdefer std.debug.print("visrope rel L2 {e:.4}\n", .{rel});
    try testing.expect(rel < 1e-6);
}

test "the scalar embedders take a raw t with no 1000x scaling" {
    const gpa = testing.allocator;
    var st = try SafeTensors.initFromSlice(gpa, ref_fixture);
    defer st.deinit();
    const ts = try (try st.require("tsembed.t")).toF32Alloc(gpa);
    defer gpa.free(ts);
    const want = try (try st.require("tsembed.out")).toF32Alloc(gpa);
    defer gpa.free(want);
    const dim = want.len / ts.len;

    const got = try gpa.alloc(f32, want.len);
    defer gpa.free(got);
    for (ts, 0..) |t, i| ops.rope.sinCosEmbedding(got[i * dim ..][0..dim], t, u15_8b.t_max_period);

    const rel = relL2(want, got);
    errdefer std.debug.print("tsembed rel L2 {e:.4}\n", .{rel});
    try testing.expect(rel < 1e-6);
}

test "the resolution noise scale saturates at 16" {
    const gpa = testing.allocator;
    var st = try SafeTensors.initFromSlice(gpa, ref_fixture);
    defer st.deinit();
    const hw_v = try st.require("noisescale.hw");
    const want = try (try st.require("noisescale.scale")).toF32Alloc(gpa);
    defer gpa.free(want);
    for (want, 0..) |w, i| {
        const h: usize = @intCast(std.mem.readInt(i32, hw_v.bytes[(i * 2) * 4 ..][0..4], .little));
        const wd: usize = @intCast(std.mem.readInt(i32, hw_v.bytes[(i * 2 + 1) * 4 ..][0..4], .little));
        const got = resolutionNoiseScale(u15_8b, h, wd);
        errdefer std.debug.print("noise scale {d}x{d}: want {d} got {d}\n", .{ h, wd, w, got });
        try testing.expect(@abs(w - got) < 1e-6);
    }
}

test "the canvas pads replicate to 16 then circular to 32" {
    const gpa = testing.allocator;
    var st = try SafeTensors.initFromSlice(gpa, ref_fixture);
    defer st.deinit();
    for (0..3) |i| {
        var kb: [32]u8 = undefined;
        const in_v = try st.require(try std.fmt.bufPrint(&kb, "pad.{d}.in", .{i}));
        const shape = in_v.info.shape.slice();
        const h = shape[1];
        const w = shape[2];
        const x = try in_v.toF32Alloc(gpa);
        defer gpa.free(x);
        const want = try (try st.require(try std.fmt.bufPrint(&kb, "pad.{d}.out", .{i}))).toF32Alloc(gpa);
        defer gpa.free(want);

        const got = try padCanvas(gpa, u15_8b, x, 3, h, w);
        defer gpa.free(got);
        errdefer std.debug.print("pad case {d} ({d}x{d}): {d} vs {d} elements\n", .{ i, h, w, want.len, got.len });
        try testing.expectEqual(want.len, got.len);
        try testing.expectEqualSlices(f32, want, got);
    }
}

test "the sigma schedule matches the upstream flow shift" {
    const gpa = testing.allocator;
    var st = try SafeTensors.initFromSlice(gpa, ref_fixture);
    defer st.deinit();
    for ([_]f32{ 1.0, 3.0 }) |shift| {
        var kb: [32]u8 = undefined;
        const want = try (try st.require(try std.fmt.bufPrint(&kb, "sigmas.{d}", .{@as(u32, @intFromFloat(shift))}))).toF32Alloc(gpa);
        defer gpa.free(want);
        const got = try sigmaSchedule(gpa, want.len - 1, shift);
        defer gpa.free(got);
        const rel = relL2(want, got);
        errdefer std.debug.print("sigmas shift {d}: rel L2 {e:.4}\n", .{ shift, rel });
        try testing.expect(rel < 1e-6);
    }
}

test "the whole forward matches ComfyUI at reduced width" {
    // Both passes at once, because they only make sense together: the prefix keys
    // and values pin the base stream (embedding, causal mask, the head-dim split),
    // and the velocity pins everything the generation stream adds (the vision
    // tower, the two scalar embedders, the unmasked attention over the prefix, the
    // pixel-shuffle head, the crop, and the x0-to-velocity conversion).
    const gpa = testing.allocator;
    const io = testing.io;
    var st = try SafeTensors.initFromSlice(gpa, ref_fixture);
    defer st.deinit();

    var pfx = try weights_mod.Prefixed.init(gpa, .{ .safetensors = &st }, "w.");
    defer pfx.deinit(gpa);
    var model = try Model.load(gpa, pfx.store(), tiny);
    defer model.deinit();

    for (0..2) |ci| {
        var kb: [48]u8 = undefined;
        const ids = try refIds(gpa, &st, try std.fmt.bufPrint(&kb, "fwd.{d}.ids", .{ci}));
        defer gpa.free(ids);

        var layout = try plainLayout(gpa, ids);
        defer layout.deinit(gpa);
        var pre = try model.prefixForward(io, gpa, layout, &.{}, null);
        defer pre.deinit(gpa);
        try testing.expectEqual(@as(u32, @intCast(ids.len)), pre.time);

        const kv_dim = tiny.kvDim();
        for (0..tiny.n_layers) |li| {
            for ([_][]const u8{ "k", "v" }) |which| {
                const ref_hm = try (try st.require(try std.fmt.bufPrint(&kb, "pre.{d}.{s}.{d}", .{ ci, which, li }))).toF32Alloc(gpa);
                defer gpa.free(ref_hm);
                const want = try gpa.alloc(f32, ref_hm.len);
                defer gpa.free(want);
                headMajorToTokenMajor(want, ref_hm, tiny.n_kv_heads, ids.len, tiny.head_dim);
                const got = if (which[0] == 'k') pre.key(li) else pre.value(li);
                try testing.expectEqual(want.len, ids.len * kv_dim);
                const rel = relL2(want, got);
                errdefer std.debug.print("case {d} prefix {s}[{d}] rel L2 {e:.4}\n", .{ ci, which, li, rel });
                try testing.expect(rel < 2e-5);
            }
        }

        const x_v = try st.require(try std.fmt.bufPrint(&kb, "fwd.{d}.x", .{ci}));
        const shape = x_v.info.shape.slice();
        const h = shape[1];
        const w = shape[2];
        const x = try x_v.toF32Alloc(gpa);
        defer gpa.free(x);
        const t = try (try st.require(try std.fmt.bufPrint(&kb, "fwd.{d}.t", .{ci}))).toF32Alloc(gpa);
        defer gpa.free(t);
        const want = try (try st.require(try std.fmt.bufPrint(&kb, "fwd.{d}.v", .{ci}))).toF32Alloc(gpa);
        defer gpa.free(want);

        const got = try gpa.alloc(f32, want.len);
        defer gpa.free(got);
        try model.predict(io, gpa, got, x, h, w, pre, t[0], null);
        const rel = relL2(want, got);
        errdefer std.debug.print("case {d} velocity ({d}x{d}) rel L2 {e:.4}\n", .{ ci, h, w, rel });
        try testing.expect(rel < 5e-5);
    }
}

test "tokenization matches ComfyUI's own SenseNova tokenizer" {
    // The whole conditioning for this family is these ids, so this is the one
    // test between a working render and a plausible wrong one. It also pins the
    // merge table: the third prompt is `#`-heavy, which is exactly where Qwen3's
    // table and the Qwen2.5 one this model wants disagree.
    const gpa = testing.allocator;
    var st = try SafeTensors.initFromSlice(gpa, ref_fixture);
    defer st.deinit();
    const md = st.metadata orelse return error.MissingFixtureMetadata;
    const meta = (md.get("prompts") orelse return error.MissingFixtureMetadata).string;
    var parsed = try std.json.parseFromSlice([]const []const u8, gpa, meta, .{});
    defer parsed.deinit();

    var tok = try initTokenizer(gpa);
    defer tok.deinit();

    for (parsed.value, 0..) |text, i| {
        var kb: [32]u8 = undefined;
        const want = try refIds(gpa, &st, try std.fmt.bufPrint(&kb, "tok.{d}", .{i}));
        defer gpa.free(want);
        const got = try tokenize(gpa, &tok, text);
        defer gpa.free(got);
        errdefer {
            std.debug.print("prompt {d} ({s}): want {d} ids, got {d}\n", .{ i, text, want.len, got.len });
            const n = @min(want.len, got.len);
            for (0..n) |j| if (want[j] != got[j]) {
                std.debug.print("  first divergence at {d}: want {d} got {d}\n", .{ j, want[j], got[j] });
                const lo = j -| 6;
                std.debug.print("  want[{d}..] {any}\n  got [{d}..] {any}\n", .{ lo, want[lo..@min(want.len, j + 6)], lo, got[lo..@min(got.len, j + 6)] });
                break;
            };
        }
        try testing.expectEqualSlices(u32, want, got);
    }
}

test "reference blocks splice into the user turn with a block-causal mask" {
    // The three pieces at once, because each is a silent wrong answer and they are
    // derived from each other: the spliced ids, the time/row/column indexes read
    // off them, and the mask read off the indexes. The `kv_end` bound this engine
    // attends with is checked AGAINST the reference's own mask matrix rather than
    // re-derived, which is the only way to know the two agree.
    const gpa = testing.allocator;
    var st = try SafeTensors.initFromSlice(gpa, ref_fixture);
    defer st.deinit();
    const md = st.metadata orelse return error.MissingFixtureMetadata;
    const meta = (md.get("edit_cases") orelse return error.MissingFixtureMetadata).string;
    const Case = struct { grids: []const [2]usize, image_only: bool };
    var parsed = try std.json.parseFromSlice([]const Case, gpa, meta, .{});
    defer parsed.deinit();

    const base = try refIds(gpa, &st, "edit.base");
    defer gpa.free(base);

    for (parsed.value, 0..) |c, i| {
        var kb: [32]u8 = undefined;
        const want_ids = try refIds(gpa, &st, try std.fmt.bufPrint(&kb, "edit.{d}.ids", .{i}));
        defer gpa.free(want_ids);
        const want_thw = try refIds(gpa, &st, try std.fmt.bufPrint(&kb, "edit.{d}.thw", .{i}));
        defer gpa.free(want_thw);
        const allow = (try st.require(try std.fmt.bufPrint(&kb, "edit.{d}.allow", .{i}))).bytes;

        var got = try editLayout(gpa, base, c.grids, c.image_only);
        defer got.deinit(gpa);
        errdefer std.debug.print("edit case {d}: want {d} ids, got {d}\n", .{ i, want_ids.len, got.ids.len });
        try testing.expectEqualSlices(u32, want_ids, got.ids);

        const n = got.ids.len;
        for (0..n) |j| {
            errdefer std.debug.print("edit case {d} token {d}: thw want ({d},{d},{d})\n", .{ i, j, want_thw[j], want_thw[n + j], want_thw[2 * n + j] });
            try testing.expectEqual(@as(usize, want_thw[j]), got.pos_t[j]);
            try testing.expectEqual(@as(usize, want_thw[n + j]), got.pos_h[j]);
            try testing.expectEqual(@as(usize, want_thw[2 * n + j]), got.pos_w[j]);
        }
        // The generated tokens sit one past the whole prefix.
        try testing.expectEqual(want_thw[n - 1] + 1, got.time);

        for (0..n) |q| {
            const end = got.kv_end.?[q];
            for (0..n) |j| {
                errdefer std.debug.print("edit case {d}: query {d} key {d}, kv_end {d}\n", .{ i, q, j, end });
                try testing.expectEqual(allow[q * n + j] != 0, j < end);
            }
        }
    }
}

test "the understanding tower normalizes with ImageNet statistics" {
    const gpa = testing.allocator;
    var st = try SafeTensors.initFromSlice(gpa, ref_fixture);
    defer st.deinit();
    const in_v = try st.require("refnorm.in");
    const shape = in_v.info.shape.slice(); // [h][w][3], the layout a node hands over
    const h = shape[0];
    const w = shape[1];
    const hwc = try in_v.toF32Alloc(gpa);
    defer gpa.free(hwc);
    const want = try (try st.require("refnorm.out")).toF32Alloc(gpa);
    defer gpa.free(want);

    const got = try gpa.alloc(f32, want.len);
    defer gpa.free(got);
    for (0..latent_channels) |c| {
        for (0..h * w) |i| {
            got[c * h * w + i] = (hwc[i * 3 + c] - imagenet_mean[c]) / imagenet_std[c];
        }
    }
    const rel = relL2(want, got);
    errdefer std.debug.print("refnorm rel L2 {e:.4}\n", .{rel});
    try testing.expect(rel < 1e-6);
}

test "the unconditional prompt is not the conditional one with an empty turn" {
    const gpa = testing.allocator;
    var tok = try initTokenizer(gpa);
    defer tok.deinit();

    const pos = try tokenize(gpa, &tok, "a red fox in the snow");
    defer gpa.free(pos);
    const neg = try tokenize(gpa, &tok, "");
    defer gpa.free(neg);

    // Both end on `<img>`, which is what the generation stream's t index counts
    // past, and the negative branch is far shorter: it drops the system message
    // and the primed thought block outright.
    try testing.expectEqual(tok_img_start, pos[pos.len - 1]);
    try testing.expectEqual(tok_img_start, neg[neg.len - 1]);
    try testing.expectEqual(tok_im_start, pos[0]);
    try testing.expect(neg.len < 16);
    try testing.expect(pos.len > 200);
    // The primed empty thought block is in the conditional prompt only.
    try testing.expect(std.mem.indexOfScalar(u32, pos, tok_think_open) != null);
    try testing.expect(std.mem.indexOfScalar(u32, neg, tok_think_open) == null);
}

const edit_ref_path = "src/models/assets/sensenova_edit_ref.safetensors";
const u15_ckpt = "/home/qt/genai/comfyui/models/diffusion_models/sensenova/sensenovaU158BMot_sft.safetensors";

test "the edit conditioning matches ComfyUI on a real checkpoint" {
    // Gated and real-weight, because the width-reduced fixture cannot carry this:
    // `_prepare_prefix` selects the reference rows by the literal id 151669, which
    // a narrowed vocabulary never contains. Layer 0 decides every convention
    // editing adds (they are all upstream of the trunk); the last layer is checked
    // too because the block mask applies at every depth. See
    // tools/gen_sensenova_edit_ref.py.
    const gpa = testing.allocator;
    const io = testing.io;
    try test_gate.requireIntegration();
    try test_gate.requireModelFile(io, u15_ckpt);
    try test_gate.requireModelFile(io, edit_ref_path);

    var ref = try SafeTensors.open(gpa, io, edit_ref_path);
    defer ref.deinit();
    var ck = try SafeTensors.open(gpa, io, u15_ckpt);
    defer ck.deinit();

    const md = ref.metadata orelse return error.MissingFixtureMetadata;
    const prompt = (md.get("prompt") orelse return error.MissingFixtureMetadata).string;
    const layers = try std.fmt.parseInt(usize, (md.get("layers") orelse return error.MissingFixtureMetadata).string, 10);

    var cfg = u15_8b;
    cfg.n_layers = layers;
    var model = try Model.load(gpa, .{ .safetensors = &ck }, cfg);
    defer model.deinit();

    // The reference picture, [h][w][3] u8 as a node hands it over.
    const ref_v = try ref.require("ref");
    const shape = ref_v.info.shape.slice();
    const rh = shape[0];
    const rw = shape[1];
    const planar = try gpa.alloc(f32, rh * rw * latent_channels);
    defer gpa.free(planar);
    for (0..latent_channels) |c| {
        for (0..rh * rw) |i| planar[c * rh * rw + i] = @as(f32, @floatFromInt(ref_v.bytes[i * 3 + c])) / 255.0;
    }
    const refs = [_]RefImage{.{ .rgb = planar, .h = rh, .w = rw }};

    var tok = try initTokenizer(gpa);
    defer tok.deinit();
    const base = try tokenize(gpa, &tok, prompt);
    defer gpa.free(base);
    {
        const want_base = try refIds(gpa, &ref, "base_ids");
        defer gpa.free(want_base);
        try testing.expectEqualSlices(u32, want_base, base);
    }

    const grids = [_][2]usize{refs[0].grid(cfg)};
    var layout = try editLayout(gpa, base, &grids, false);
    defer layout.deinit(gpa);
    {
        const want_ids = try refIds(gpa, &ref, "ids");
        defer gpa.free(want_ids);
        try testing.expectEqualSlices(u32, want_ids, layout.ids);
        const want_thw = try refIds(gpa, &ref, "thw");
        defer gpa.free(want_thw);
        const n = layout.ids.len;
        for (0..n) |i| {
            try testing.expectEqual(@as(usize, want_thw[i]), layout.pos_t[i]);
            try testing.expectEqual(@as(usize, want_thw[n + i]), layout.pos_h[i]);
            try testing.expectEqual(@as(usize, want_thw[2 * n + i]), layout.pos_w[i]);
        }
        const want_time = try refIds(gpa, &ref, "time");
        defer gpa.free(want_time);
        try testing.expectEqual(want_time[0], layout.time);
    }

    var pre = try model.prefixForward(io, gpa, layout, &refs, null);
    defer pre.deinit(gpa);

    // The first and the last layer the fixture carries; the mask applies at every
    // depth, so a check at layer 0 alone would not see it drift.
    for ([_]usize{ 0, layers - 1 }) |li| {
        var kb: [16]u8 = undefined;
        for ([_][]const u8{ "k", "v" }) |which| {
            const hm = try (try ref.require(try std.fmt.bufPrint(&kb, "{s}.{d}", .{ which, li }))).toF32Alloc(gpa);
            defer gpa.free(hm);
            const want = try gpa.alloc(f32, hm.len);
            defer gpa.free(want);
            headMajorToTokenMajor(want, hm, cfg.n_kv_heads, pre.seq, cfg.head_dim);
            const got = if (which[0] == 'k') pre.key(li) else pre.value(li);
            const rel = relL2(want, got);
            errdefer std.debug.print("edit prefix {s}[{d}] rel L2 {e:.4}\n", .{ which, li, rel });
            // The fixture stores f16, so this is that storage floor and not ours.
            try testing.expect(rel < 3e-3);
        }
    }

    // And one whole denoiser forward on that prefix, so the generation pass is
    // checked against an EDIT conditioning and not only a text one. Nothing in it
    // is edit-specific, which is exactly why it needs saying: the only thing that
    // changes is the prefix's length and the t index the canvas takes past it.
    {
        const x_v = try ref.require("fwd.x");
        const xs = x_v.info.shape.slice();
        const x = try x_v.toF32Alloc(gpa);
        defer gpa.free(x);
        const t = try (try ref.require("fwd.t")).toF32Alloc(gpa);
        defer gpa.free(t);
        const want = try (try ref.require("fwd.v")).toF32Alloc(gpa);
        defer gpa.free(want);
        const got = try gpa.alloc(f32, want.len);
        defer gpa.free(got);
        try model.predict(io, gpa, got, x, xs[1], xs[2], pre, t[0], null);
        const rel = relL2(want, got);
        errdefer std.debug.print("edit forward rel L2 {e:.4}\n", .{rel});
        try testing.expect(rel < 3e-3);
    }
}

const edit_full_path = "src/models/assets/sensenova_edit_full.safetensors";

test "the edit prefix matches ComfyUI at the full 42 layers" {
    // The four-layer check above cannot see a divergence that ACCUMULATES, and the
    // block mask applies at every depth. This one runs the whole trunk against a
    // bf16 reference whose `_mot_gen` half was never materialized (the prefix pass
    // does not read it), which is what makes a 42-layer reference fit at all.
    //
    // bf16 is why the bound is loose where the fp32 check above is 3e-3: that is
    // the REFERENCE's operand precision, not ours, and it compounds down 42 layers
    // of residual stream. Measured 3.3e-3 / 2.1e-3 at layer 0 and 2.6e-2 / 5.3e-2
    // at layer 41, which is this codebase's usual bf16 floor at that depth. It
    // still has teeth: the failure this exists to exclude is a SCALE error, and
    // that lands near 1.
    const gpa = testing.allocator;
    const io = testing.io;
    try test_gate.requireIntegration();
    try test_gate.requireModelFile(io, u15_ckpt);
    try test_gate.requireModelFile(io, edit_full_path);

    var ref = try SafeTensors.open(gpa, io, edit_full_path);
    defer ref.deinit();
    var ck = try SafeTensors.open(gpa, io, u15_ckpt);
    defer ck.deinit();

    const md = ref.metadata orelse return error.MissingFixtureMetadata;
    const prompt = (md.get("prompt") orelse return error.MissingFixtureMetadata).string;
    const cfg = u15_8b;

    var model = try Model.load(gpa, .{ .safetensors = &ck }, cfg);
    defer model.deinit();

    const ref_v = try ref.require("ref");
    const shape = ref_v.info.shape.slice();
    const rh = shape[0];
    const rw = shape[1];
    const planar = try gpa.alloc(f32, rh * rw * latent_channels);
    defer gpa.free(planar);
    for (0..latent_channels) |c| {
        for (0..rh * rw) |i| planar[c * rh * rw + i] = @as(f32, @floatFromInt(ref_v.bytes[i * 3 + c])) / 255.0;
    }
    const refs = [_]RefImage{.{ .rgb = planar, .h = rh, .w = rw }};

    var tok = try initTokenizer(gpa);
    defer tok.deinit();
    const base = try tokenize(gpa, &tok, prompt);
    defer gpa.free(base);
    const grids = [_][2]usize{refs[0].grid(cfg)};
    var layout = try editLayout(gpa, base, &grids, false);
    defer layout.deinit(gpa);
    {
        const want_time = try refIds(gpa, &ref, "time");
        defer gpa.free(want_time);
        try testing.expectEqual(want_time[0], layout.time);
    }

    var pre = try model.prefixForward(io, gpa, layout, &refs, null);
    defer pre.deinit(gpa);

    for ([_]usize{ 0, cfg.n_layers - 1 }) |li| {
        var kb: [16]u8 = undefined;
        for ([_][]const u8{ "k", "v" }) |which| {
            const hm = try (try ref.require(try std.fmt.bufPrint(&kb, "{s}.{d}", .{ which, li }))).toF32Alloc(gpa);
            defer gpa.free(hm);
            const want = try gpa.alloc(f32, hm.len);
            defer gpa.free(want);
            headMajorToTokenMajor(want, hm, cfg.n_kv_heads, pre.seq, cfg.head_dim);
            const got = if (which[0] == 'k') pre.key(li) else pre.value(li);
            const rel = relL2(want, got);
            errdefer std.debug.print("full-depth prefix {s}[{d}] rel L2 {e:.4}\n", .{ which, li, rel });
            try testing.expect(rel < 8e-2);
        }
    }
}

test "the Qwen2.5 merge table drops Qwen3's hash rules" {
    const gpa = testing.allocator;
    var q3 = try tokenizer_mod.Tokenizer.init(gpa);
    defer q3.deinit();
    var q25 = try tokenizer_mod.Tokenizer.initQwen25(gpa);
    defer q25.deinit();

    // `#include` is one Qwen3 merge and two Qwen2.5 tokens. A prompt with no `#`
    // must tokenize identically under both, or the exclusion list is too wide.
    var a: std.ArrayList(u32) = .empty;
    defer a.deinit(gpa);
    var b: std.ArrayList(u32) = .empty;
    defer b.deinit(gpa);
    try q3.encode(gpa, "#include <stdio.h>", &a);
    try q25.encode(gpa, "#include <stdio.h>", &b);
    try testing.expect(!std.mem.eql(u32, a.items, b.items));

    a.clearRetainingCapacity();
    b.clearRetainingCapacity();
    try q3.encode(gpa, "a photograph of an astronaut riding a horse", &a);
    try q25.encode(gpa, "a photograph of an astronaut riding a horse", &b);
    try testing.expectEqualSlices(u32, a.items, b.items);
}
