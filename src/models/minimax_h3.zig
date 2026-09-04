//! MiniMax H3: a joint audio-video flow-matching DiT. Video (24ch, patch 1x2x2)
//! and stereo audio (32ch, 40 Hz) are denoised in ONE packed token sequence, so
//! the audio rows sit in the same attention pass as the video rows:
//!
//!     [text | ref/keyframe cond rows | audio | video]
//!
//! The target streams are always the last two segments, audio before video.
//!
//! The two streams run on DIFFERENT sigma schedules (shift 12 video, 3 audio).
//! The sampler carries the audio latent scaled onto the video schedule; the
//! forward undoes that scale, so the network only ever sees a stream's own
//! latent. Getting the carry wrong leaves the video looking right and the audio
//! subtly wrong.
//!
//! Conventions that are silent wrong answers when got wrong, beyond the carry:
//!
//! - The forward's output is NEGATED, both streams.
//! - adaLN carries three modality tags (video 0, text 1, audio 2) and its
//!   projection reshapes to [M*modalities, expand*hidden], so a modulation row
//!   is `t_row * 3 + tag`. Interleaving timestep and tag the other way is silent.
//! - The text span is NOT uniform: vision-pad tokens inside it carry tag 0
//!   (video), so the span splits into tag runs.
//! - RoPE is partial split-half: [S,3] -> [S,48] -> cat(half, half) -> [S,96],
//!   and the rotation table takes only the first half. Head dim is 128, so 96
//!   dims rotate and 32 pass through untouched.
//! - The position grid is area-normalized floats, not indices, in f64 until rope
//!   time. See `axisFromSqrtArea`.
//! - The video time axis is non-uniform: token k spans `frame_per_token[k % 5]`
//!   frames, so the first token of every group of five spans 1 and the rest 4.
//! - Audio rows are channel-major, with `w` pinned to the extremes of the video
//!   frame's `w` grid and `h` fixed at 0. See `audioGrid`.
//!
//! Reference is ComfyUI: comfy/ldm/minimax/model.py plus the layout half of
//! comfy_extras/nodes_minimax_h3.py. See VIDEO_PLAN.md for the whole list and
//! for how the rest of the family is staged. The VAEs are minimax_h3_vae.zig
//! (video) and minimax_h3_audio.zig (audio); the GPU twins are
//! minimax_h3_cuda.zig and minimax_h3_gpu.zig.

const std = @import("std");
const tp_core = @import("tp_core");
const weights_mod = tp_core.weights;
const ops = @import("tp_ops");
const lora_mod = @import("lora.zig");
const quant_weight = @import("quant_weight.zig");

const WeightStore = weights_mod.WeightStore;
const Weight = ops.matmul.Weight;
const DType = tp_core.dtype.DType;

/// Video latent channels, and the audio latent's channel count. Both are fixed
/// by the architecture rather than probed, because a different value would
/// change the packed layout rather than just a shape.
pub const latent_channels = 24;
pub const audio_latent_channels = 32;
/// Stereo. The audio stream is packed channel-major, so this is also how many
/// times the audio time axis repeats in the sequence.
pub const audio_channels = 2;

/// Pixel frames per video latent token, cycling. The first token of every group
/// of five covers one frame and the rest cover four, which is what makes the
/// time axis non-uniform.
pub const frame_per_token = [5]usize{ 1, 4, 4, 4, 4 };
/// Pixel frames -> shared time axis. One audio latent frame is 1.0 on the same
/// axis, so a pixel frame is 5/3 of one.
pub const frame_rescale: f64 = 5.0 / 3.0;

/// Frames per second the model was trained at. Sets the video/audio frame ratio
/// together with `audio_latent_fps`.
pub const fps = 24;
pub const audio_latent_fps = 40;
/// Audio samples per second out of the audio VAE, and samples per latent frame.
pub const audio_sample_rate = 32000;
pub const audio_samples_per_latent = audio_sample_rate / audio_latent_fps;

/// Spatial downscale of the video VAE, and the DiT's spatial patch. A pixel
/// dimension therefore maps to the DiT's row grid by `spatial_downscale * patch`.
pub const spatial_downscale = 16;
pub const patch_t = 1;
pub const patch_h = 2;
pub const patch_w = 2;

/// Flow shift for each stream. The video shift drives the sampler's sigma
/// schedule; the audio one is derived from it in closed form (`timeShiftSigma`).
pub const shift_video: f32 = 12.0;
pub const shift_audio: f32 = 3.0;

/// Both stream shifts, together.
///
/// A pair rather than two parameters because they are only ever meaningful as a
/// RATIO: the audio sigma is `timeShiftSigma(sigma_v, video, audio)` and the
/// sampler's carry scale is `video / audio`. Three places need both (the timestep
/// labels, the per-step carry, and the final unscale), and passing them
/// separately is how one of them ends up reading a caller's video shift against
/// the module default audio shift, which is a plausible, wrong audio level.
pub const Shifts = struct {
    video: f32 = shift_video,
    audio: f32 = shift_audio,

    /// The audio stream's sigma at a given video sigma.
    pub fn audioSigma(s: Shifts, sigma_v: f32) f32 {
        return @floatCast(timeShiftSigma(sigma_v, s.video, s.audio));
    }

    /// The scale the sampler carries the audio stream at, i.e. the small-sigma
    /// limit of `sigma_v / sigma_a`.
    pub fn carryScale(s: Shifts) f32 {
        return s.video / s.audio;
    }
};

/// Timestep a fully preserved visual / audio condition row is pinned at. Not
/// 1.0 for video: the reference pins visual conditions a hair below it.
pub const visual_cond_timestep: f32 = 0.999;
pub const audio_cond_timestep: f32 = 1.0;

/// Linear latent->RGB approximation for the live preview, from
/// `comfy/latent_formats.py` (`MiniMaxH3Video`). Its own matrix over its own 24
/// channels: another family's is either an out-of-bounds read or a preview with
/// plausible structure and wrong colours.
pub const latent_rgb_factors = [latent_channels][3]f32{
    .{ -0.018555, 0.024344, -0.017536 },
    .{ 0.150164, 0.137244, 0.129221 },
    .{ 0.027367, -0.050369, -0.208606 },
    .{ -0.000793, -0.164622, -0.323161 },
    .{ -0.048556, 0.013970, -0.074286 },
    .{ 0.011740, 0.014172, -0.006906 },
    .{ 0.061517, 0.061212, 0.110025 },
    .{ 0.035321, 0.086879, 0.110059 },
    .{ -0.017426, 0.002997, 0.035356 },
    .{ 0.531539, 0.548819, 0.624404 },
    .{ -0.024968, -0.040234, -0.034302 },
    .{ -0.032549, -0.029096, -0.017221 },
    .{ 0.022609, 0.020286, 0.050661 },
    .{ -0.084001, -0.038131, -0.020805 },
    .{ -0.018830, 0.010412, 0.061120 },
    .{ 0.020777, 0.011196, -0.030994 },
    .{ -0.008390, -0.012201, -0.025687 },
    .{ -0.013281, -0.002924, 0.006331 },
    .{ 0.000260, 0.001833, -0.011038 },
    .{ 0.105471, 0.100482, 0.132106 },
    .{ 0.016529, 0.015213, 0.009999 },
    .{ -0.014015, -0.017438, -0.019134 },
    .{ -0.033787, -0.009984, -0.019725 },
    .{ 0.004224, 0.017284, 0.027196 },
};
pub const latent_rgb_bias = [3]f32{ 0.057426, -0.022078, -0.071449 };

/// Fill `rgb_out` (`[h*w][3]` RGB8) with the latent2rgb preview of ONE frame of
/// a planar `[24][t][h][w]` latent. `frame` indexes the temporal axis, and the
/// per-channel stride is `t * h * w`, not `h * w`: reading a video latent with an
/// image family's stride previews channel 0 of several frames as if they were
/// several channels of one.
pub fn latentPreviewInto(rgb_out: []u8, z: []const f32, t: usize, h: usize, w: usize, frame: usize) void {
    const plane = h * w;
    std.debug.assert(frame < t);
    std.debug.assert(rgb_out.len >= plane * 3 and z.len >= latent_channels * t * plane);
    for (0..plane) |p| {
        var acc = latent_rgb_bias;
        inline for (0..latent_channels) |c| {
            const v = z[(c * t + frame) * plane + p];
            acc[0] += v * latent_rgb_factors[c][0];
            acc[1] += v * latent_rgb_factors[c][1];
            acc[2] += v * latent_rgb_factors[c][2];
        }
        inline for (0..3) |ch| {
            const u = std.math.clamp((acc[ch] + 1.0) * 0.5, 0.0, 1.0) * 255.0;
            rgb_out[p * 3 + ch] = @intFromFloat(u);
        }
    }
}

/// adaLN modality tag. The value IS the tag index the modulation row is built
/// from, so do not reorder.
pub const Tag = enum(u2) {
    video = 0,
    text = 1,
    audio = 2,
};

pub const modality_count = 3;

/// Invert `sigma = s*b / (1 + (s-1)*b)` to the shared base grid and re-apply the
/// other stream's shift. This is how the audio schedule is derived from the
/// video sigma the sampler hands in.
pub fn timeShiftSigma(sigma: f64, from_shift: f64, to_shift: f64) f64 {
    const base = sigma / (from_shift + sigma * (1.0 - from_shift));
    return to_shift * base / (1.0 + (to_shift - 1.0) * base);
}

// --- frame arithmetic ------------------------------------------------------

/// Snap a frame count up to the model's grid. Valid lengths are `17k + 5`, so
/// 5, 22, 39, ... and the default 124 is ~5s at 24 fps.
/// The canvas a REFERENCE VIDEO is resized to: a 768 px short edge under a
/// 768x1344 area cap, each axis rounded to 32.
///
/// Not the generation's canvas and not the video's own size. Two things about it
/// are easy to get wrong:
///
/// - it is driven by the ASPECT RATIO, not by either dimension, so a wide clip
///   gets `768 * ratio` wide and a tall one `768 / ratio` tall;
/// - a source SMALLER than the result is not upscaled: the node falls back to the
///   video's own dimensions rounded to 32, because spending reference tokens on
///   interpolated detail is worse than spending fewer on real detail.
///
/// Returns `(w, h)`, in that order, matching the reference's own tuple.
pub fn adaptCanvas(width: usize, height: usize) struct { w: usize, h: usize } {
    const base_short_edge: f64 = 768.0;
    const max_pixels: f64 = 768.0 * 1344.0;
    const multiple: usize = 32;

    const ratio = @as(f64, @floatFromInt(width)) / @as(f64, @floatFromInt(height));
    var nw: f64 = undefined;
    var nh: f64 = undefined;
    if (ratio >= 1.0) {
        nw = base_short_edge * ratio;
        nh = base_short_edge;
    } else {
        nw = base_short_edge;
        nh = base_short_edge / ratio;
    }
    if (nw * nh > max_pixels) {
        const sc = @sqrt(max_pixels / (nw * nh));
        nw *= sc;
        nh *= sc;
    }
    const cw = @max(multiple, roundTo(nw, multiple));
    const ch = @max(multiple, roundTo(nh, multiple));
    // Do not upscale: a small clip keeps its own size, rounded.
    if (width * height < cw * ch) {
        return .{
            .w = @max(multiple, roundTo(@floatFromInt(width), multiple)),
            .h = @max(multiple, roundTo(@floatFromInt(height), multiple)),
        };
    }
    return .{ .w = cw, .h = ch };
}

/// Python's `round` (half to EVEN) times `m`, matching the reference's
/// `round(v / m) * m`. Half-up differs at exactly half a multiple.
fn roundTo(v: f64, m: usize) usize {
    const mf: f64 = @floatFromInt(m);
    const q = v / mf;
    const r = @round(q);
    const use = if (@abs(q - @trunc(q)) == 0.5) blk: {
        const t = @trunc(q);
        break :blk if (@mod(t, 2.0) == 0.0) t else t + std.math.sign(q);
    } else r;
    return @intFromFloat(use * mf);
}

/// Frames a reference video contributes after cropping to a legal clip length.
///
/// The model's clip grid is `n % 17 == 5`, and a reference video is cropped DOWN
/// to it (`while n % 17 != 5: n -= 1`), never padded up. Fewer than 5 frames is
/// refused rather than padded: the reference raises there.
pub fn refVideoFrames(available: usize, target_frames: usize) !usize {
    var n = @min(available, target_frames);
    if (n < 5) return error.RefVideoTooShort;
    while (n % 17 != 5) n -= 1;
    return n;
}

/// Frame indices the VISION TOWER sees, at 2 fps rather than the clip's 24.
///
/// The tower gets every `fps / 2`-th frame with a timestamp label, so a 5 s clip
/// costs ten vision blocks rather than 120. Feeding it every frame would be both
/// far slower and a different presentation from the one it was trained on.
pub fn refVideoSampleStride() usize {
    return fps / 2;
}

pub fn alignFrameCount(frames: usize) usize {
    var n = @max(5, frames);
    while (n % 17 != 5) n += 1;
    return n;
}

/// Video latent frames for an aligned pixel frame count. Each group of five
/// tokens covers 17 frames, and the tail is the leading 5.
pub fn videoLatentT(frames: usize) usize {
    if (frames <= 5) return 2;
    return ((frames - 5) / 17) * 5 + 2;
}

/// Pixel frames a run of `n` video latent tokens covers. The inverse of
/// `videoLatentT` on the aligned grid, and what the guide nodes use to place a
/// keyframe, so it is worth having rather than re-deriving.
pub fn framesForLatentT(n: usize) usize {
    var total: usize = 0;
    for (0..n) |k| total += frame_per_token[k % frame_per_token.len];
    return total;
}

/// Audio latent frames for a pixel frame count: `round(frames / 24 * 40)`.
///
/// Integer form of the reference's `round(duration * 40)`. Python's `round` is
/// half-to-even, but `5 * frames / 3` can never land exactly on a half (it would
/// need `4 * frames == 3 (mod 6)`, even against odd), so the tie rule is
/// unreachable and this is exact.
pub fn audioLatentT(frames: usize) usize {
    return (10 * frames + 3) / 6;
}

/// The three counts a render is shaped by, from a requested pixel length.
pub fn temporalShape(length: usize) struct { frames: usize, latent_t: usize, audio_t: usize } {
    const frames = alignFrameCount(length);
    return .{ .frames = frames, .latent_t = videoLatentT(frames), .audio_t = audioLatentT(frames) };
}

// --- config ---------------------------------------------------------------

pub const Config = struct {
    hidden: usize,
    n_layers: usize,
    /// Blocks in the text-side `token_refiner`, which runs once per sampling run
    /// rather than per step.
    refiner_layers: usize,
    n_heads: usize,
    head_dim: usize,
    /// Post-swiglu width, i.e. half of `mlp.fc1`'s output.
    ffn: usize,
    /// Width of the text encoder's hidden state that `condition_proj` consumes.
    text_dim: usize,
    /// Width of the adaLN input. Small (8) on curve-form checkpoints, because
    /// the adaLN linears read interpolated coordinates of a time-embedding curve
    /// rather than a time embedding.
    time_embed_dim: usize,
    /// Rows in `adaln_t_table`. Null on checkpoints that ship a `time_embedder`
    /// instead; the local checkpoint is curve-form.
    adaln_curve_grid: ?usize,
    /// Per-axis rope frequency count. The packed rope width is `6 * this`: three
    /// axes, halves duplicated.
    rope_inv_freq_len: usize,

    norm_eps: f32 = 1e-5,
    qk_norm_eps: f32 = 1e-5,
    final_norm_eps: f32 = 1e-5,

    pub fn videoPatchDim(self: Config) usize {
        _ = self;
        return latent_channels * patch_t * patch_h * patch_w;
    }

    /// Rotation pairs: one per (axis, frequency), so three axes worth.
    pub fn ropePairs(self: Config) usize {
        return self.rope_inv_freq_len * 3;
    }

    /// Head dims the rotation covers, `2 * ropePairs`. The remaining
    /// `head_dim - ropeRotDim` dims pass through unrotated.
    pub fn ropeRotDim(self: Config) usize {
        return self.ropePairs() * 2;
    }

    /// Width of the packed angle buffer the rotation table is built from. The
    /// reference emits `cat(half, half)`, so the two halves are identical and
    /// only the first is read; this equals `ropeRotDim` by construction.
    pub fn ropeAngleWidth(self: Config) usize {
        return self.rope_inv_freq_len * 6;
    }

    pub fn usesAdalnCurve(self: Config) bool {
        return self.adaln_curve_grid != null;
    }

    /// Read the config off the checkpoint's own tensor shapes, the way
    /// `comfy/model_detection.py` does. Nothing here is guessed from a name.
    pub fn detect(store: WeightStore) !Config {
        const video_proj = try dimsOf(store, "video_patch_proj.weight");
        try video_proj.rank(2);
        const hidden = video_proj.d[0];

        const video_out = try dimsOf(store, "final_layer.video_out.weight");
        try video_out.rank(2);
        // patch 1x2x2, so the head emits four spatial sub-positions per channel
        if (video_out.d[0] != latent_channels * patch_t * patch_h * patch_w) return error.ShapeMismatch;

        const audio_out = try dimsOf(store, "final_layer.audio_out.weight");
        try audio_out.rank(2);
        if (audio_out.d[0] != audio_latent_channels) return error.ShapeMismatch;

        const q_norm = try dimsOf(store, "blocks.0.attn.q_norm.weight");
        try q_norm.rank(1);
        const head_dim = q_norm.d[0];

        const qkv = try dimsOf(store, "blocks.0.attn.qkv_proj.weight");
        try qkv.rank(2);
        if (head_dim == 0 or qkv.d[0] % (3 * head_dim) != 0) return error.ShapeMismatch;
        const n_heads = qkv.d[0] / (3 * head_dim);

        const fc1 = try dimsOf(store, "blocks.0.mlp.fc1.weight");
        // fc1 is the fused swiglu gate+value, so its output is twice the ffn width
        try fc1.rank(2);
        if (fc1.d[0] % 2 != 0) return error.ShapeMismatch;

        const cond = try dimsOf(store, "condition_proj.weight");
        try cond.rank(2);
        if (cond.d[0] != hidden) return error.ShapeMismatch;

        const inv_freq = try dimsOf(store, "rope.inv_freq");
        try inv_freq.rank(1);

        var curve_grid: ?usize = null;
        var time_embed_dim: usize = 0;
        if (store.get("adaln_t_table") != null) {
            const table = try dimsOf(store, "adaln_t_table");
            try table.rank(2);
            curve_grid = table.d[0];
            time_embed_dim = table.d[1];
        } else {
            const te = try dimsOf(store, "time_embedder.proj_out.weight");
            try te.rank(2);
            time_embed_dim = te.d[0];
        }

        return .{
            .hidden = hidden,
            .n_layers = countBlocks(store, "blocks."),
            .refiner_layers = countBlocks(store, "token_refiner.blocks."),
            .n_heads = n_heads,
            .head_dim = head_dim,
            .ffn = fc1.d[0] / 2,
            .text_dim = cond.d[1],
            .time_embed_dim = time_embed_dim,
            .adaln_curve_grid = curve_grid,
            .rope_inv_freq_len = inv_freq.d[0],
        };
    }
};

/// A tensor's dims, COPIED out of the view.
///
/// `WeightStore.get` returns the `TensorView` by value and its `Shape.dims` is
/// an inline array, so `view.info.shape.slice()` borrows from a temporary. That
/// is fine where the view stays in scope (see `loader.zig`) and dangling the
/// moment it is returned from a helper, which reads back as `0x5555...` rather
/// than as a crash.
const Dims = struct {
    n: usize,
    d: [8]usize,

    fn rank(self: Dims, want: usize) !void {
        if (self.n != want) return error.ShapeMismatch;
    }
};

fn dimsOf(store: WeightStore, name: []const u8) !Dims {
    const view = store.get(name) orelse return error.MissingTensor;
    const s = view.info.shape.slice();
    var out: Dims = .{ .n = s.len, .d = @splat(0) };
    if (s.len > out.d.len) return error.ShapeMismatch;
    for (s, 0..) |v, i| out.d[i] = v;
    return out;
}

/// Count `prefix ++ "{i}." ++ "norm1.weight"` upward from 0. Both the trunk and
/// the token refiner have a `norm1` in every block.
fn countBlocks(store: WeightStore, comptime prefix: []const u8) usize {
    var n: usize = 0;
    while (true) : (n += 1) {
        var buf: [128]u8 = undefined;
        const name = std.fmt.bufPrint(&buf, prefix ++ "{d}.norm1.weight", .{n}) catch return n;
        if (store.get(name) == null) return n;
    }
}

// --- packed layout --------------------------------------------------------

/// One contiguous run of the packed sequence. The sequence is uniform per
/// segment in (modality tag, timestep class), except the text span, whose tag
/// runs are resolved at forward time from the encoder's per-token tags.
pub const Kind = enum {
    text,
    /// Keyframe video condition rows (fl2va).
    cond,
    /// Keyframe audio condition rows.
    cond_audio,
    /// Reference image or reference video rows (ref2va).
    ref_img,
    /// Reference audio rows, including a reference video's soundtrack.
    ref_audio,
    /// The denoised audio stream.
    audio,
    /// The denoised video stream.
    video,
    /// Text-encoder rows that carry the VIDEO tag: a spliced vision block and
    /// its two flanking markers. They are text ROWS (they arrive through
    /// `condition_proj`, not a patch projection) with a video TAG, which is why
    /// this cannot be either `.text` or `.ref_img`.
    text_vision,

    /// Number of kinds, so the per-kind tables below cannot fall out of step
    /// with this enum.
    pub const count = @typeInfo(Kind).@"enum".fields.len;

    pub fn tag(self: Kind) Tag {
        return switch (self) {
            .text => .text,
            .cond, .ref_img, .video, .text_vision => .video,
            .cond_audio, .ref_audio, .audio => .audio,
        };
    }

    /// Whether rows of this kind come from the video patch projection (as
    /// opposed to the audio one, or the text encoder). `.text_vision` is NOT one:
    /// its rows are text-encoder output that merely shares the video tag.
    pub fn isVideoRow(self: Kind) bool {
        return switch (self) {
            .cond, .ref_img, .video => true,
            else => false,
        };
    }

    pub fn isAudioRow(self: Kind) bool {
        return switch (self) {
            .cond_audio, .ref_audio, .audio => true,
            else => false,
        };
    }
};

pub const Segment = struct {
    start: usize,
    stop: usize,
    kind: Kind,

    pub fn len(self: Segment) usize {
        return self.stop - self.start;
    }
};

/// A half-open row range within the text region that carries the VIDEO tag
/// instead of the text one.
///
/// One per spliced vision block, and it covers the block PLUS its two flanking
/// markers: the reference widens each image's embed span by one on each side, so
/// `<|vision_start|>` and `<|vision_end|>` are tagged with the image rather than
/// with the prose around them.
pub const VisionSpan = struct {
    start: usize,
    len: usize,

    pub fn stop(self: VisionSpan) usize {
        return self.start + self.len;
    }
};

/// Where one spliced vision block sits in the token stream: `index` is the first
/// EMBEDDING row (just past `<|vision_start|>`) and `size` the embedding count.
///
/// This exists because the same block needs TWO different spans and they differ
/// by one row at each end:
///
/// - the modality TAG span is widened by one on each side, so the flanking
///   `<|vision_start|>` / `<|vision_end|>` markers are tagged with the image
///   rather than with the prose (`token_tags_from_embeds_info`);
/// - the DEEPSTACK injection span is NOT widened, covering only the embedding
///   rows (`build_image_inputs`'s `visual_pos_masks`).
///
/// Using one where the other belongs is off by a row at each end: the tag version
/// mistags two prose tokens, and the injection version adds a vision feature to a
/// marker. Both are finite and wrong, so the two spans are derived here rather
/// than at the call sites.
pub const VisionBlock = struct {
    index: usize,
    size: usize,

    /// The tag-0 span: the block plus both markers, clamped at the sequence start.
    pub fn tagSpan(self: VisionBlock) VisionSpan {
        const start = if (self.index == 0) 0 else self.index - 1;
        return .{ .start = start, .len = self.index + self.size + 1 - start };
    }

    /// The deepstack span: the embedding rows only.
    pub fn injectSpan(self: VisionBlock) VisionSpan {
        return .{ .start = self.index, .len = self.size };
    }
};

/// A keyframe anchors an image (and/or a snippet of audio) at a pixel frame of
/// the target video. Shapes only: the latents themselves ride in the payload.
pub const Keyframe = struct {
    /// Pixel frame index the keyframe is anchored at, already resolved against
    /// the video length (negative indices counted from the end by the caller).
    frame_index: usize,
    /// Video latent frames contributed, 0 when this anchors audio only.
    latent_t: usize = 0,
    /// Audio latent frames contributed, 0 when this anchors video only.
    audio_t: usize = 0,
};

/// A reference block. `video` with `audio_t == 0` is the reference's
/// soundtrack-less video case; the reference implementation spells that as two
/// kinds ("video" / "video_audio") but treats them identically, so this does not.
pub const Ref = struct {
    pub const RefKind = enum { image, audio, video };

    kind: RefKind,
    /// Video latent frames. 1 for `image`, unused for `audio`.
    latent_t: usize = 0,
    /// Latent spatial dims of this reference, which need not match the target's.
    latent_h: usize = 0,
    latent_w: usize = 0,
    /// Audio latent frames. Unused for `image`.
    audio_t: usize = 0,

    /// Span this block occupies on the shared time axis, ahead of the targets.
    fn timeSpan(self: Ref) f64 {
        return switch (self.kind) {
            .image => 1.0,
            .audio => @floatFromInt(self.audio_t),
            .video => @max(
                @as(f64, @floatFromInt(self.audio_t)),
                videoSpanTotal(self.latent_t),
            ),
        };
    }
};

/// Per-axis coordinate of one row's patch position, area-normalized. The grid is
/// `linspace((1 - ratio) / 2, (1 + ratio) / 2, dim / patch, endpoint=False) * 32`
/// with `ratio = dim / sqrt(h * w)`, in f64. These are NOT indices: a 48x84
/// latent gives a `h` axis stepping ~1.0 from ~3.9 and a `w` axis stepping ~1.0
/// from ~-5.2, and the absolute values feed rope directly.
pub fn axisFromSqrtArea(alloc: std.mem.Allocator, dim: usize, patch: usize, sqrt_area: f64) ![]f64 {
    const n = dim / patch;
    const ratio = @as(f64, @floatFromInt(dim)) / sqrt_area;
    const out = try alloc.alloc(f64, n);
    const step = ratio / @as(f64, @floatFromInt(n));
    const base = (1.0 - ratio) / 2.0;
    for (out, 0..) |*v, i| v.* = (@as(f64, @floatFromInt(i)) * step + base) * 32.0;
    return out;
}

/// Total time-axis span of `n` video latent tokens.
fn videoSpanTotal(n: usize) f64 {
    var total: f64 = 0;
    for (0..n) |k| total += frame_rescale * @as(f64, @floatFromInt(frame_per_token[k % frame_per_token.len]));
    return total;
}

/// A row's (t, h, w) position, f64 until rope time. The reference keeps this in
/// float64 and casts once; matching that matters because the values are large
/// (up to ~32) and feed a cos/sin.
pub const Pos = [3]f64;

/// The static structure of one packed sequence: which rows are what, and where
/// each row sits on the (t, h, w) grid. Depends only on shapes, so it is built
/// once per sampling run rather than per step.
pub const PackedLayout = struct {
    seq_len: usize,
    /// [seq_len] row positions.
    pos: []Pos,
    /// Contiguous, covering [0, seq_len), in packed order.
    segments: []Segment,
    /// Video rows that are condition rows rather than denoised targets, i.e. how
    /// many rows the caller's condition latents must supply. Every condition
    /// segment precedes the target segment, so the video row stream is simply
    /// `cond rows ++ target rows`; `videoCondRows` is where the split is.
    video_cond_rows: usize,
    audio_cond_rows: usize,

    text_len: usize,
    latent_t: usize,
    latent_h: usize,
    latent_w: usize,
    audio_t: usize,

    arena: std.heap.ArenaAllocator,

    pub const Shape = struct {
        /// Text rows, i.e. the encoder's sequence length after the refiner.
        text_len: usize,
        latent_t: usize,
        /// Latent spatial dims, already padded up to the DiT's 2x2 patch.
        latent_h: usize,
        latent_w: usize,
        audio_t: usize,
    };

    pub fn deinit(self: *PackedLayout) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// Whether a cached layout still describes this shape. Refs and keyframes
    /// are part of the identity too, but they are fixed for a sampling run,
    /// which is the only lifetime a layout has.
    pub fn matches(self: *const PackedLayout, s: Shape) bool {
        return self.text_len == s.text_len and self.latent_t == s.latent_t and
            self.latent_h == s.latent_h and self.latent_w == s.latent_w and
            self.audio_t == s.audio_t;
    }

    pub fn segmentOf(self: *const PackedLayout, kind: Kind) ?Segment {
        for (self.segments) |s| {
            if (s.kind == kind) return s;
        }
        return null;
    }

    /// `vision_spans` must be ascending, non-overlapping, and inside
    /// `[0, text_len)`. They are checked rather than trusted: an overlap would
    /// emit segments that do not tile the text region, and the modulation loop
    /// would then modulate some rows twice and others never.
    pub fn build(
        gpa: std.mem.Allocator,
        s: Shape,
        keyframes: []const Keyframe,
        refs: []const Ref,
        vision_spans: []const VisionSpan,
    ) !PackedLayout {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();

        // Scratch for the axis grids lives in the arena too; it is small and the
        // arena is the layout's own lifetime.
        const target_area = @sqrt(@as(f64, @floatFromInt(s.latent_h * s.latent_w)));
        const h_axis = try axisFromSqrtArea(alloc, s.latent_h, patch_h, target_area);
        const w_axis = try axisFromSqrtArea(alloc, s.latent_w, patch_w, target_area);
        const frame_rows = h_axis.len * w_axis.len;
        // Stereo audio rows pin `w` to the extremes of the target frame's w grid,
        // one channel per extreme, whatever the reference block's own grid is.
        const audio_w_low = w_axis[0];
        const audio_w_high = w_axis[w_axis.len - 1];

        // Refs pack between the text and the targets, so the target timeline
        // starts after their spans. Keyframes are placed against that same
        // post-ref origin, while the refs themselves walk from `text_len`.
        var target_cursor: f64 = @floatFromInt(s.text_len);
        for (refs) |r| target_cursor += r.timeSpan();

        var segments: std.ArrayList(Segment) = .empty;
        var pos: std.ArrayList(Pos) = .empty;

        var row: usize = 0;
        var video_cond_rows: usize = 0;
        var audio_cond_rows: usize = 0;

        // text: t counts rows, h and w stay 0. The POSITIONS are uniform over the
        // whole region whatever the tagging is; only the segmentation splits.
        try pos.ensureUnusedCapacity(alloc, s.text_len);
        for (0..s.text_len) |i| pos.appendAssumeCapacity(.{ @floatFromInt(i), 0, 0 });

        // A spliced vision block carries the VIDEO tag, so the text region is a
        // run of `.text` and `.text_vision` segments rather than one segment. The
        // runs must TILE it exactly: the modulation loop walks segments, so a gap
        // leaves rows unmodulated and an overlap modulates them twice.
        var cursor: usize = 0;
        for (vision_spans) |vs| {
            if (vs.start < cursor or vs.stop() > s.text_len or vs.len == 0) return error.BadVisionSpan;
            if (vs.start > cursor) {
                try segments.append(alloc, .{ .start = row + cursor, .stop = row + vs.start, .kind = .text });
            }
            try segments.append(alloc, .{ .start = row + vs.start, .stop = row + vs.stop(), .kind = .text_vision });
            cursor = vs.stop();
        }
        if (cursor < s.text_len) {
            try segments.append(alloc, .{ .start = row + cursor, .stop = row + s.text_len, .kind = .text });
        }
        row += s.text_len;

        for (keyframes) |kf| {
            // anchors count from the target origin, frame_rescale per pixel frame
            const cond_t = target_cursor + frame_rescale * @as(f64, @floatFromInt(kf.frame_index));
            if (kf.latent_t > 0) {
                const n = kf.latent_t * frame_rows;
                try segments.append(alloc, .{ .start = row, .stop = row + n, .kind = .cond });
                try appendVideoGrid(alloc, &pos, kf.latent_t, h_axis, w_axis, cond_t);
                row += n;
                video_cond_rows += n;
            }
            if (kf.audio_t > 0) {
                const n = kf.audio_t * audio_channels;
                try segments.append(alloc, .{ .start = row, .stop = row + n, .kind = .cond_audio });
                try appendAudioGrid(alloc, &pos, cond_t, kf.audio_t, audio_w_low, audio_w_high);
                row += n;
                audio_cond_rows += n;
            }
        }

        var ref_cursor: f64 = @floatFromInt(s.text_len);
        for (refs) |r| {
            switch (r.kind) {
                .image => {
                    const r_area = @sqrt(@as(f64, @floatFromInt(r.latent_h * r.latent_w)));
                    const rh = try axisFromSqrtArea(alloc, r.latent_h, patch_h, r_area);
                    const rw = try axisFromSqrtArea(alloc, r.latent_w, patch_w, r_area);
                    const n = rh.len * rw.len;
                    try segments.append(alloc, .{ .start = row, .stop = row + n, .kind = .ref_img });
                    // a single frame: every row shares the cursor as its t
                    try pos.ensureUnusedCapacity(alloc, n);
                    for (rh) |hv| {
                        for (rw) |wv| pos.appendAssumeCapacity(.{ ref_cursor, hv, wv });
                    }
                    row += n;
                    video_cond_rows += n;
                },
                .audio => {
                    if (r.audio_t > 0) {
                        const n = r.audio_t * audio_channels;
                        try segments.append(alloc, .{ .start = row, .stop = row + n, .kind = .ref_audio });
                        try appendAudioGrid(alloc, &pos, ref_cursor, r.audio_t, audio_w_low, audio_w_high);
                        row += n;
                        audio_cond_rows += n;
                    }
                },
                .video => {
                    // the block's audio rows pack immediately BEFORE its video
                    // rows, both sharing the cursor as their origin
                    const r_area = @sqrt(@as(f64, @floatFromInt(r.latent_h * r.latent_w)));
                    const rh = try axisFromSqrtArea(alloc, r.latent_h, patch_h, r_area);
                    const rw = try axisFromSqrtArea(alloc, r.latent_w, patch_w, r_area);
                    if (r.audio_t > 0) {
                        const n = r.audio_t * audio_channels;
                        try segments.append(alloc, .{ .start = row, .stop = row + n, .kind = .ref_audio });
                        // a reference video's soundtrack pins to ITS OWN w grid
                        try appendAudioGrid(alloc, &pos, ref_cursor, r.audio_t, rw[0], rw[rw.len - 1]);
                        row += n;
                        audio_cond_rows += n;
                    }
                    const n = r.latent_t * rh.len * rw.len;
                    try segments.append(alloc, .{ .start = row, .stop = row + n, .kind = .ref_img });
                    try appendVideoGrid(alloc, &pos, r.latent_t, rh, rw, ref_cursor);
                    row += n;
                    video_cond_rows += n;
                },
            }
            ref_cursor += r.timeSpan();
        }

        // target audio, then target video: always the last two segments
        const n_audio = s.audio_t * audio_channels;
        try segments.append(alloc, .{ .start = row, .stop = row + n_audio, .kind = .audio });
        try appendAudioGrid(alloc, &pos, target_cursor, s.audio_t, audio_w_low, audio_w_high);
        row += n_audio;

        const n_video = s.latent_t * frame_rows;
        try segments.append(alloc, .{ .start = row, .stop = row + n_video, .kind = .video });
        try appendVideoGrid(alloc, &pos, s.latent_t, h_axis, w_axis, target_cursor);
        row += n_video;

        std.debug.assert(pos.items.len == row);

        return .{
            .seq_len = row,
            .pos = pos.items,
            .segments = segments.items,
            .video_cond_rows = video_cond_rows,
            .audio_cond_rows = audio_cond_rows,
            .text_len = s.text_len,
            .latent_t = s.latent_t,
            .latent_h = s.latent_h,
            .latent_w = s.latent_w,
            .audio_t = s.audio_t,
            .arena = arena,
        };
    }
};

/// Video rows in patchify order: frame index outer, then h, then w. The t
/// coordinate is `origin + exclusive cumsum of the per-token spans`.
fn appendVideoGrid(
    alloc: std.mem.Allocator,
    pos: *std.ArrayList(Pos),
    latent_t: usize,
    h_axis: []const f64,
    w_axis: []const f64,
    origin: f64,
) !void {
    try pos.ensureUnusedCapacity(alloc, latent_t * h_axis.len * w_axis.len);
    var t = origin;
    for (0..latent_t) |k| {
        for (h_axis) |hv| {
            for (w_axis) |wv| pos.appendAssumeCapacity(.{ t, hv, wv });
        }
        // exclusive cumsum: advance AFTER emitting, so token 0 sits at the origin
        t += frame_rescale * @as(f64, @floatFromInt(frame_per_token[k % frame_per_token.len]));
    }
}

/// Stereo audio rows, channel-major: the whole time axis for channel 0, then
/// again for channel 1. `h` stays 0 and `w` is pinned per channel to an extreme
/// of the frame's w grid.
fn appendAudioGrid(
    alloc: std.mem.Allocator,
    pos: *std.ArrayList(Pos),
    cursor: f64,
    audio_t: usize,
    w_low: f64,
    w_high: f64,
) !void {
    try pos.ensureUnusedCapacity(alloc, audio_t * audio_channels);
    for ([2]f64{ w_low, w_high }) |wv| {
        for (0..audio_t) |i| {
            pos.appendAssumeCapacity(.{ cursor + @as(f64, @floatFromInt(i)), 0, wv });
        }
    }
}


// --- weights --------------------------------------------------------------

/// A linear plus its optional LoRA sidecar.
///
/// The pairing is structural on purpose. A sidecar that is applied at some GEMM
/// call sites and not others renders a finite, plausible, wrong image with no
/// error anywhere, which is the hazard CLAUDE.md names for weight storage. The
/// weight is reachable only as `.w`, so every site that consumes one has the
/// sidecar in the same expression, and `matLin` below is the only host path.
pub const Lin = struct {
    w: Weight,
    lora: ?*const lora_mod.Target = null,
};

/// `y = W x (+ bias) + sidecar`. The one host path through a `Lin`, so the
/// sidecar cannot be forgotten at a call site.
fn matLin(
    io: std.Io,
    gpa: std.mem.Allocator,
    y: []f32,
    y_stride: usize,
    x: []const f32,
    m: usize,
    l: Lin,
    bias: ?[]const f32,
) !void {
    try ops.matmul.matmul(io, gpa, y[0 .. m * y_stride], x, m, l.w, bias);
    if (l.lora) |t| try t.applyHost(io, gpa, y, y_stride, x, m);
}

/// One attention, trunk or refiner. `qkv_proj` is fused; `q_norm`/`k_norm` are
/// PER-HEAD (their weight is `head_dim` wide, not `hidden`).
pub const Attn = struct {
    qkv: Lin,
    q_norm: []f32,
    k_norm: []f32,
    out: Lin,
};

/// SwiGLU MLP. `fc1` emits `2 * ffn` and the halves are `[gate; value]` in that
/// order: `silu(gate) * value`. The order is not a guess, ComfyUI's
/// `_swiglu_eager` chunks gate-first, and the turbo LoRA's metadata records the
/// Diffusers-to-ComfyUI remap (`[value;gate] -> [gate;value]`) as already
/// applied. Swapping them is finite and wrong.
pub const Mlp = struct {
    fc1: Lin,
    fc2: Lin,
};

/// A token-refiner block: plain pre-norm attention + MLP, no modulation and NO
/// rope. It runs once per sampling run, not per step.
pub const RefinerBlock = struct {
    norm1: []f32,
    norm2: []f32,
    attn: Attn,
    mlp: Mlp,
};

/// One trunk block. `adaln` projects the time embedding to
/// `6 * hidden * 3` (six modulation slots x three modality tags); see
/// `modRowOffset` for the layout, which is the part that is silent when wrong.
pub const Block = struct {
    norm1: []f32,
    norm2: []f32,
    attn: Attn,
    mlp: Mlp,
    adaln: Weight,
    adaln_bias: []f32,
};

/// The output head. Its adaLN has two slots and ONE modality (so a modulation
/// row here is `t_row`, not `t_row * 3 + tag`), and both projections are the
/// checkpoint's f32 island.
pub const FinalLayer = struct {
    norm: []f32,
    adaln: Weight,
    adaln_bias: []f32,
    video_out: Weight,
    video_bias: []f32,
    audio_out: Weight,
    audio_bias: []f32,
};

pub const DiT = struct {
    arena: std.heap.ArenaAllocator,
    cfg: Config,

    video_patch: Weight,
    video_patch_bias: []f32,
    audio_patch: Weight,
    audio_patch_bias: []f32,
    condition_proj: Weight,
    condition_bias: []f32,

    /// `[adaln_curve_grid][time_embed_dim]` f32. Curve-form checkpoints ship this
    /// instead of a time embedder; `timeEmbed` interpolates it.
    adaln_t_table: []f32,
    /// `[rope_inv_freq_len]` f32, shared by all three position axes.
    rope_inv_freq: []f32,

    refiner: []RefinerBlock,
    refiner_final_norm: []f32,
    blocks: []Block,
    final: FinalLayer,

    pub fn deinit(self: *DiT) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// Point every trunk and refiner linear at its LoRA sidecar, if the file has
    /// one for it. Borrows `side`, which must outlive the DiT.
    ///
    /// Returns how many linears were patched. Zero means the file matched
    /// nothing here, which is a mistake worth reporting rather than a render
    /// that quietly ignores a flag the user passed.
    pub fn attachLora(self: *DiT, side: *const lora_mod.Sidecar) !usize {
        var n: usize = 0;
        for (self.blocks) |*b| {
            n += try attachBlockLora(&b.attn, &b.mlp, side);
        }
        for (self.refiner) |*b| {
            n += try attachBlockLora(&b.attn, &b.mlp, side);
        }
        return n;
    }

    fn attachBlockLora(a: *Attn, m: *Mlp, side: *const lora_mod.Sidecar) !usize {
        var n: usize = 0;
        inline for (.{ &a.qkv, &a.out, &m.fc1, &m.fc2 }) |l| {
            switch (side.forWeight(l.w)) {
                .none => {},
                .ok => |t| {
                    l.lora = t;
                    n += 1;
                },
                // Refuse rather than skip: a trunk with only some of its
                // sidecars applied renders plausibly and wrongly.
                .mismatch => |t| {
                    std.log.err("minimax_h3: LoRA {s} is {d}x{d} but the checkpoint's is {d}x{d}", .{ t.tag, t.out_dim, t.in_dim, l.w.rows, l.w.cols });
                    return error.ShapeMismatch;
                },
            }
        }
        return n;
    }


    pub fn load(gpa: std.mem.Allocator, store: WeightStore) !DiT {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();

        const cfg = try Config.detect(store);
        // Only the curve form is implemented. A `time_embedder` checkpoint is a
        // different (earlier) export of the same architecture; refusing by name
        // beats running with an uninitialized table.
        if (!cfg.usesAdalnCurve()) return error.UnsupportedCheckpoint;

        const l: Loader = .{ .alloc = alloc, .store = store, .cfg = cfg };
        const hidden = cfg.hidden;
        const inner = cfg.n_heads * cfg.head_dim;
        const t_dim = cfg.time_embed_dim;

        const refiner = try alloc.alloc(RefinerBlock, cfg.refiner_layers);
        for (refiner, 0..) |*b, i| b.* = .{
            .norm1 = try l.vec("token_refiner.blocks.{d}.norm1.weight", .{i}, hidden),
            .norm2 = try l.vec("token_refiner.blocks.{d}.norm2.weight", .{i}, hidden),
            .attn = try l.attn("token_refiner.blocks.{d}.attn", .{i}),
            .mlp = try l.mlp("token_refiner.blocks.{d}.mlp", .{i}),
        };

        const blocks = try alloc.alloc(Block, cfg.n_layers);
        for (blocks, 0..) |*b, i| b.* = .{
            .norm1 = try l.vec("blocks.{d}.norm1.weight", .{i}, hidden),
            .norm2 = try l.vec("blocks.{d}.norm2.weight", .{i}, hidden),
            .attn = try l.attn("blocks.{d}.attn", .{i}),
            .mlp = try l.mlp("blocks.{d}.mlp", .{i}),
            .adaln = try l.mat("blocks.{d}.adaln_proj.linear.weight", .{i}, 6 * hidden * modality_count, t_dim),
            .adaln_bias = try l.vec("blocks.{d}.adaln_proj.linear.bias", .{i}, 6 * hidden * modality_count),
        };

        // Every arena allocation happens before `arena` is copied into the result;
        // chunks allocated afterwards would be missed by deinit.
        const video_patch = try l.mat("video_patch_proj.weight", .{}, hidden, cfg.videoPatchDim());
        const video_patch_bias = try l.vec("video_patch_proj.bias", .{}, hidden);
        const audio_patch = try l.mat("audio_patch_proj.weight", .{}, hidden, audio_latent_channels);
        const audio_patch_bias = try l.vec("audio_patch_proj.bias", .{}, hidden);
        const condition_proj = try l.mat("condition_proj.weight", .{}, hidden, cfg.text_dim);
        const condition_bias = try l.vec("condition_proj.bias", .{}, hidden);
        const adaln_t_table = try l.vec("adaln_t_table", .{}, cfg.adaln_curve_grid.? * t_dim);
        const rope_inv_freq = try l.vec("rope.inv_freq", .{}, cfg.rope_inv_freq_len);
        const refiner_final_norm = try l.vec("token_refiner.final_norm.weight", .{}, hidden);
        const final: FinalLayer = .{
            .norm = try l.vec("final_layer.norm.weight", .{}, hidden),
            .adaln = try l.mat("final_layer.adaln_proj.linear.weight", .{}, 2 * hidden, t_dim),
            .adaln_bias = try l.vec("final_layer.adaln_proj.linear.bias", .{}, 2 * hidden),
            .video_out = try l.mat("final_layer.video_out.weight", .{}, cfg.videoPatchDim(), hidden),
            .video_bias = try l.vec("final_layer.video_out.bias", .{}, cfg.videoPatchDim()),
            .audio_out = try l.mat("final_layer.audio_out.weight", .{}, audio_latent_channels, hidden),
            .audio_bias = try l.vec("final_layer.audio_out.bias", .{}, audio_latent_channels),
        };
        _ = inner;

        return .{
            .arena = arena,
            .cfg = cfg,
            .video_patch = video_patch,
            .video_patch_bias = video_patch_bias,
            .audio_patch = audio_patch,
            .audio_patch_bias = audio_patch_bias,
            .condition_proj = condition_proj,
            .condition_bias = condition_bias,
            .adaln_t_table = adaln_t_table,
            .rope_inv_freq = rope_inv_freq,
            .refiner = refiner,
            .refiner_final_norm = refiner_final_norm,
            .blocks = blocks,
            .final = final,
        };
    }
};

const Loader = struct {
    alloc: std.mem.Allocator,
    store: WeightStore,
    cfg: Config,

    fn name(l: Loader, buf: []u8, comptime fmt: []const u8, args: anytype, suffix: []const u8) ![]const u8 {
        _ = l;
        const base = try std.fmt.bufPrint(buf, fmt, args);
        if (suffix.len == 0) return base;
        // The scale sidecar's name is the weight's plus a suffix, so it has to be
        // appended after formatting rather than being a second format string.
        if (base.len + suffix.len > buf.len) return error.NameTooLong;
        @memcpy(buf[base.len..][0..suffix.len], suffix);
        return buf[0 .. base.len + suffix.len];
    }

    /// A 2-D weight in its checkpoint dtype, with the int8/int4 convrot sidecars
    /// wired. Omitting those is a PANIC rather than a wrong answer:
    /// `ops.matmul.matmul` asserts `row_scale != null` for an integer weight.
    fn mat(l: Loader, comptime fmt: []const u8, args: anytype, rows: usize, cols: usize) !Weight {
        var buf: [192]u8 = undefined;
        const nm = try l.name(&buf, fmt, args, "");
        const view = l.store.get(nm) orelse {
            std.log.err("minimax_h3: missing tensor {s}", .{nm});
            return error.MissingTensor;
        };
        const shape = view.info.shape.slice();
        if (shape.len != 2 or shape[0] != rows or shape[1] != cols) {
            std.log.err("minimax_h3: {s} has shape {any}, expected {d}x{d}", .{ nm, shape, rows, cols });
            return error.ShapeMismatch;
        }
        const dt = view.info.dtype;
        if (!ops.matmul.supportsDType(dt)) {
            std.log.err("minimax_h3: {s} has unsupported dtype {t}", .{ nm, dt });
            return error.UnsupportedDType;
        }
        var w = Weight.init(view.bytes, dt, rows, cols);
        w.tag = try l.alloc.dupe(u8, nm);

        if (dt == .i8 or dt == .i4) {
            const meta = try quant_weight.int8ScaleConvrot(l.alloc, l.store, nm, rows, cols, "minimax_h3");
            w.row_scale = meta.row_scale;
            w.convrot = meta.convrot;
        }
        return w;
    }

    fn vec(l: Loader, comptime fmt: []const u8, args: anytype, len: usize) ![]f32 {
        var buf: [192]u8 = undefined;
        const nm = try l.name(&buf, fmt, args, "");
        const view = l.store.get(nm) orelse {
            std.log.err("minimax_h3: missing tensor {s}", .{nm});
            return error.MissingTensor;
        };
        if (view.info.elemCount() != len) {
            std.log.err("minimax_h3: {s} has {d} elements, expected {d}", .{ nm, view.info.elemCount(), len });
            return error.ShapeMismatch;
        }
        return view.toF32Alloc(l.alloc);
    }

    fn lin(l: Loader, comptime fmt: []const u8, args: anytype, rows: usize, cols: usize) !Lin {
        return .{ .w = try l.mat(fmt, args, rows, cols) };
    }

    fn attn(l: Loader, comptime fmt: []const u8, args: anytype) !Attn {
        const cfg = l.cfg;
        const inner = cfg.n_heads * cfg.head_dim;
        return .{
            .qkv = try l.lin(fmt ++ ".qkv_proj.weight", args, 3 * inner, cfg.hidden),
            .q_norm = try l.vec(fmt ++ ".q_norm.weight", args, cfg.head_dim),
            .k_norm = try l.vec(fmt ++ ".k_norm.weight", args, cfg.head_dim),
            .out = try l.lin(fmt ++ ".out_proj.weight", args, cfg.hidden, inner),
        };
    }

    fn mlp(l: Loader, comptime fmt: []const u8, args: anytype) !Mlp {
        const cfg = l.cfg;
        return .{
            .fc1 = try l.lin(fmt ++ ".fc1.weight", args, 2 * cfg.ffn, cfg.hidden),
            .fc2 = try l.lin(fmt ++ ".fc2.weight", args, cfg.hidden, cfg.ffn),
        };
    }
};

// --- stream packing -------------------------------------------------------

/// Patchify a planar `[c][t][h][w]` video latent into `[t*(h/2)*(w/2)][c*4]`
/// rows, the reference's einsum `nctrhpwq->nthwcrpq` with `pt = 1`.
///
/// Layout permutations are rms-preserving: every norm and magnitude matches when
/// this is wrong and only the picture differs, so this direction and
/// `unpatchifyVideo` are pinned by their own test.
pub fn patchifyVideo(rows: []f32, z: []const f32, t: usize, h: usize, w: usize) void {
    const ph = patch_h;
    const pw = patch_w;
    std.debug.assert(h % ph == 0 and w % pw == 0);
    const gh = h / ph;
    const gw = w / pw;
    const cols = latent_channels * ph * pw;
    std.debug.assert(rows.len == t * gh * gw * cols);
    std.debug.assert(z.len == latent_channels * t * h * w);
    for (0..t) |ti| {
        for (0..gh) |hi| {
            for (0..gw) |wi| {
                const row = ((ti * gh + hi) * gw + wi) * cols;
                for (0..latent_channels) |c| {
                    for (0..ph) |p| {
                        for (0..pw) |q| {
                            const src = ((c * t + ti) * h + (hi * ph + p)) * w + (wi * pw + q);
                            rows[row + (c * ph + p) * pw + q] = z[src];
                        }
                    }
                }
            }
        }
    }
}

/// The exact inverse of `patchifyVideo`.
pub fn unpatchifyVideo(z: []f32, rows: []const f32, t: usize, h: usize, w: usize) void {
    const ph = patch_h;
    const pw = patch_w;
    std.debug.assert(h % ph == 0 and w % pw == 0);
    const gh = h / ph;
    const gw = w / pw;
    const cols = latent_channels * ph * pw;
    std.debug.assert(rows.len == t * gh * gw * cols);
    std.debug.assert(z.len == latent_channels * t * h * w);
    for (0..t) |ti| {
        for (0..gh) |hi| {
            for (0..gw) |wi| {
                const row = ((ti * gh + hi) * gw + wi) * cols;
                for (0..latent_channels) |c| {
                    for (0..ph) |p| {
                        for (0..pw) |q| {
                            const dst = ((c * t + ti) * h + (hi * ph + p)) * w + (wi * pw + q);
                            z[dst] = rows[row + (c * ph + p) * pw + q];
                        }
                    }
                }
            }
        }
    }
}

/// Pack a planar `[32][2][at]` audio latent into `[2*at][32]` rows,
/// CHANNEL-MAJOR: the whole time axis for stereo channel 0, then again for
/// channel 1. Interleaving instead is the same byte count and a different model.
pub fn packAudio(rows: []f32, z: []const f32, at: usize) void {
    const c_n = audio_latent_channels;
    std.debug.assert(rows.len == audio_channels * at * c_n);
    std.debug.assert(z.len == c_n * audio_channels * at);
    for (0..audio_channels) |s| {
        for (0..at) |ti| {
            const row = (s * at + ti) * c_n;
            for (0..c_n) |c| rows[row + c] = z[(c * audio_channels + s) * at + ti];
        }
    }
}

/// The exact inverse of `packAudio`.
pub fn unpackAudio(z: []f32, rows: []const f32, at: usize) void {
    const c_n = audio_latent_channels;
    std.debug.assert(rows.len == audio_channels * at * c_n);
    std.debug.assert(z.len == c_n * audio_channels * at);
    for (0..audio_channels) |s| {
        for (0..at) |ti| {
            const row = (s * at + ti) * c_n;
            for (0..c_n) |c| z[(c * audio_channels + s) * at + ti] = rows[row + c];
        }
    }
}

// --- timesteps and modulation --------------------------------------------

/// The distinct timestep values one forward needs, and which row of the time
/// embedding each segment reads.
///
/// The reference builds this by collecting every segment's label into a sorted
/// unique list; the row index is then the position in that list. Doing it the
/// same way matters because the row index feeds `modRowOffset`, so a different
/// ORDER is a different (finite, wrong) modulation.
pub const Timesteps = struct {
    /// Upper bound on distinct labels: one per `Kind`, and the two stream values
    /// are always among them. `Workspace` sizes its time-embedding scratch by
    /// this rather than by the count a particular sigma produces, so one
    /// workspace serves every step.
    ///
    /// Eight labels would cover the `Kind`s (and only four of those are ever
    /// distinct); the rest is headroom for a DENOISE MASK's own levels. A mask with
    /// more distinct row values than fit is refused by name rather than quantized:
    /// the label set sizes the modulation table and every row index points into it,
    /// so silently merging levels would change the picture, not just the table.
    pub const max_labels = 16;

    /// Distinct labels, ascending.
    values: [max_labels]f32 = @splat(0),
    n: usize = 0,
    /// Row index per `Kind`, indexed by `@intFromEnum`.
    row_of_kind: [Kind.count]usize = @splat(0),
    /// Per-row label index inside the `.video` target segment, or empty when every
    /// row shares the segment's own label (which is the unmasked case AND the case
    /// where a mask turned out uniform). OWNED; see `deinit`.
    video_rows: []u8 = &.{},
    /// The same for the `.audio` target segment.
    audio_rows: []u8 = &.{},

    pub fn deinit(self: *Timesteps, gpa: std.mem.Allocator) void {
        if (self.video_rows.len > 0) gpa.free(self.video_rows);
        if (self.audio_rows.len > 0) gpa.free(self.audio_rows);
        self.* = undefined;
    }

    pub fn init(sigma_v: f32, visual_aug: f32, audio_aug: f32, shifts: Shifts) Timesteps {
        var self: Timesteps = .{};
        const pk = perKind(sigma_v, visual_aug, audio_aug, shifts);
        // Four distinct values at most, so the label cap cannot be reached here.
        self.seed(pk, 1.0 - sigma_v, 1.0 - shifts.audioSigma(sigma_v)) catch unreachable;
        return self;
    }

    /// Same, with a denoise mask folded in.
    ///
    /// A row with mask value `m` runs at sigma `m * sigma_stream`, so `m = 1` is the
    /// ordinary stream timestep and `m = 0` pins the row at the condition one --
    /// the model is told those rows are already clean. The mask never gates the
    /// arithmetic; it only relabels rows on the time axis.
    ///
    /// ⚠️ **The two streams use DIFFERENT sigmas**: video the sampler's, audio its
    /// own shifted one. They coincide only at shift parity.
    ///
    /// A mask whose rows all agree COLLAPSES into the segment's own label rather
    /// than becoming a per-row table, which is observable: it changes the set of
    /// distinct labels and hence every row index into the modulation table.
    pub fn initMasked(
        gpa: std.mem.Allocator,
        sigma_v: f32,
        visual_aug: f32,
        audio_aug: f32,
        shifts: Shifts,
        video_mask: []const f32,
        audio_mask: []const f32,
    ) !Timesteps {
        const t_v: f32 = 1.0 - sigma_v;
        const sigma_a = shifts.audioSigma(sigma_v);
        const t_a: f32 = 1.0 - sigma_a;

        var v_t: []f32 = &.{};
        defer if (v_t.len > 0) gpa.free(v_t);
        var a_t: []f32 = &.{};
        defer if (a_t.len > 0) gpa.free(a_t);

        var pk = perKind(sigma_v, visual_aug, audio_aug, shifts);
        if (video_mask.len > 0) {
            v_t = try rowTimesteps(gpa, video_mask, sigma_v, @max(t_v, visual_aug));
            if (allSame(v_t) and !force_row_labels) {
                pk[@intFromEnum(Kind.video)] = v_t[0];
                gpa.free(v_t);
                v_t = &.{};
            }
        }
        if (audio_mask.len > 0) {
            a_t = try rowTimesteps(gpa, audio_mask, sigma_a, @max(t_a, audio_aug));
            if (allSame(a_t) and !force_row_labels) {
                pk[@intFromEnum(Kind.audio)] = a_t[0];
                gpa.free(a_t);
                a_t = &.{};
            }
        }

        var self: Timesteps = .{};
        errdefer self.deinit(gpa);
        try self.seed(pk, t_v, t_a);
        for (v_t) |v| try self.insert(v);
        for (a_t) |v| try self.insert(v);
        // Only now are the indices final: an inserted mask level can sort below a
        // kind's label and shift it.
        for (&self.row_of_kind, pk) |*r, v| r.* = self.indexOf(v);
        if (v_t.len > 0) self.video_rows = try self.indexTable(gpa, v_t);
        if (a_t.len > 0) self.audio_rows = try self.indexTable(gpa, a_t);
        return self;
    }

    fn perKind(sigma_v: f32, visual_aug: f32, audio_aug: f32, shifts: Shifts) [Kind.count]f32 {
        const t_v: f32 = 1.0 - sigma_v;
        const t_a: f32 = 1.0 - shifts.audioSigma(sigma_v);
        var out: [Kind.count]f32 = undefined;
        inline for (@typeInfo(Kind).@"enum".fields) |f| {
            const k: Kind = @enumFromInt(f.value);
            out[f.value] = switch (k) {
                // text follows the VIDEO stream, not a label of its own, and a
                // spliced vision block follows the text it sits in: same
                // TIMESTEP, different modality TAG.
                .text, .video, .text_vision => t_v,
                .audio => t_a,
                // a preserved condition row is pinned at (or above) the cond
                // timestep, which for video is 0.999 rather than 1.0
                .cond, .ref_img => @max(t_v, visual_aug),
                .cond_audio, .ref_audio => @max(t_a, audio_aug),
            };
        }
        return out;
    }

    /// `clamp(1 - m * sigma, max = pin)`, per row.
    fn rowTimesteps(gpa: std.mem.Allocator, mask: []const f32, sigma: f32, pin: f32) ![]f32 {
        const out = try gpa.alloc(f32, mask.len);
        for (out, mask) |*o, m| o.* = @min(1.0 - m * sigma, pin);
        return out;
    }

    fn allSame(xs: []const f32) bool {
        for (xs[1..]) |v| if (v != xs[0]) return false;
        return true;
    }

    /// `t_v` and `t_a` are the two STREAM values, which the reference's label set
    /// always contains (`{t_v, t_a} | {segment labels}`) whether or not a segment
    /// carries them. They are passed in rather than read off `pk`, because a
    /// collapsed denoise mask REPLACES `pk[.video]` / `pk[.audio]` with the mask's
    /// own value and `.audio` is the only kind that would otherwise carry `t_a`.
    ///
    /// Kinds absent from the layout do contribute here and not in the reference;
    /// harmless, because the two extras (the pinned condition labels) always sort
    /// ABOVE every stream label and so never shift a used index.
    fn seed(self: *Timesteps, pk: [Kind.count]f32, t_v: f32, t_a: f32) !void {
        try self.insert(t_v);
        try self.insert(t_a);
        for (pk) |v| try self.insert(v);
        for (&self.row_of_kind, pk) |*r, v| r.* = self.indexOf(v);
    }

    fn insert(self: *Timesteps, v: f32) !void {
        for (self.values[0..self.n]) |x| if (x == v) return;
        // Not logged here: this is per step, and the sentence the user needs
        // ("use fewer mask levels") belongs where the mask was accepted. The
        // pipeline says it once.
        if (self.n == max_labels) return error.TooManyTimestepLabels;
        var i = self.n;
        while (i > 0 and self.values[i - 1] > v) : (i -= 1) self.values[i] = self.values[i - 1];
        self.values[i] = v;
        self.n += 1;
    }

    fn indexOf(self: *const Timesteps, v: f32) usize {
        for (self.values[0..self.n], 0..) |x, i| if (x == v) return i;
            unreachable;
    }

    fn indexTable(self: *const Timesteps, gpa: std.mem.Allocator, rows_t: []const f32) ![]u8 {
        const out = try gpa.alloc(u8, rows_t.len);
        for (out, rows_t) |*o, v| o.* = @intCast(self.indexOf(v));
        return out;
    }

    pub fn labels(self: *const Timesteps) []const f32 {
        return self.values[0..self.n];
    }

    pub fn rowFor(self: *const Timesteps, k: Kind) usize {
        return self.row_of_kind[@intFromEnum(k)];
    }

    /// Per-row label indices for a segment, or empty when the segment is uniform.
    /// Only the two TARGET segments can be masked; every condition row is pinned.
    pub fn rowsFor(self: *const Timesteps, k: Kind) []const u8 {
        return switch (k) {
            .video => self.video_rows,
            .audio => self.audio_rows,
            else => &.{},
        };
    }
};

/// Diagnostic: keep the per-row label table even when every row agrees, so a
/// uniform mask takes the per-ROW modulation path instead of collapsing to the
/// scalar one. The two must then produce the same answer, which is how the device's
/// index-buffer arithmetic is checked without any int8 or CPU noise in the way.
pub var force_row_labels: bool = false;

/// Reduce a `[latent_t][mask_h][mask_w]` denoise mask (1 = generate) to one value
/// per 2x2 patch row, replicate-padded up to the latent grid first.
///
/// ⚠️ **The patch reduction is a MAX, not a mean**: a patch regenerates if ANY of
/// its four latent cells does. A mean turns a half-covered patch into a half-noised
/// one, which is a plausible image and not this model's.
///
/// Returns null when every row fully generates. That is the reference's "no mask"
/// answer and is NOT the same as a table of ones: a null keeps the segment-level
/// label and therefore the label set.
pub fn maskRowValues(
    alloc: std.mem.Allocator,
    mask: []const f32,
    latent_t: usize,
    mask_h: usize,
    mask_w: usize,
    lat_h: usize,
    lat_w: usize,
) !?[]f32 {
    std.debug.assert(mask.len == latent_t * mask_h * mask_w);
    std.debug.assert(mask_h > 0 and mask_w > 0);
    std.debug.assert(mask_h <= lat_h and mask_w <= lat_w);
    const rows_h = lat_h / patch_h;
    const rows_w = lat_w / patch_w;
    const out = try alloc.alloc(f32, latent_t * rows_h * rows_w);
    errdefer alloc.free(out);

    var all_generate = true;
    for (0..latent_t) |t| {
        const plane = mask[t * mask_h * mask_w ..][0 .. mask_h * mask_w];
        for (0..rows_h) |ry| {
            for (0..rows_w) |rx| {
                var m: f32 = 0;
                for (0..patch_h) |dy| {
                    // Replicate padding: past the mask's own extent the edge value
                    // repeats. A mask that stops short would otherwise read as 0,
                    // i.e. "preserve" -- the opposite of what a short mask means.
                    const y = @min(ry * patch_h + dy, mask_h - 1);
                    for (0..patch_w) |dx| {
                        const x = @min(rx * patch_w + dx, mask_w - 1);
                        m = @max(m, plane[y * mask_w + x]);
                    }
                }
                out[(t * rows_h + ry) * rows_w + rx] = m;
                if (m < 1.0 - 1e-3) all_generate = false;
            }
        }
    }
    if (all_generate) {
        alloc.free(out);
        return null;
    }
    return out;
}

/// Offset of one modulation vector inside a trunk block's adaLN output.
///
/// The projection emits `[M][3 modalities][6 slots][hidden]`, so the reference's
/// `view(M*3, 6*hidden).chunk(6, -1)` indexes slot `s` of modality `tag` at
/// timestep row `m` as `(m * 3 + tag) * 6 * hidden + s * hidden`. Interleaving
/// timestep and tag the other way round reads a real vector from the wrong place.
pub fn modRowOffset(t_row: usize, tag: Tag, slot: usize, hidden: usize) usize {
    std.debug.assert(slot < 6);
    return (t_row * modality_count + @intFromEnum(tag)) * 6 * hidden + slot * hidden;
}

/// Interpolate the time-embedding curve at each label.
///
/// `t` clamps to [0, 1], scales by `grid - 1`, and the lower index clamps to
/// `grid - 2` so `t == 1.0` lands on the last interval instead of reading one row
/// past the table.
pub fn timeEmbed(out: []f32, table: []const f32, grid: usize, dim: usize, ts: []const f32) void {
    std.debug.assert(table.len == grid * dim);
    std.debug.assert(out.len == ts.len * dim);
    std.debug.assert(grid >= 2);
    for (ts, 0..) |t_raw, i| {
        const t = std.math.clamp(t_raw, 0.0, 1.0);
        const pos = t * @as(f32, @floatFromInt(grid - 1));
        // max-clamp keeps t == 1.0 on the last interval instead of reading one
        // row past the table
        const lo = @min(@as(usize, @intFromFloat(@floor(pos))), grid - 2);
        const frac = pos - @as(f32, @floatFromInt(lo));
        const a = table[lo * dim ..][0..dim];
        const b = table[(lo + 1) * dim ..][0..dim];
        const o = out[i * dim ..][0..dim];
        for (o, a, b) |*dst, av, bv| dst.* = av + (bv - av) * frac;
    }
}

/// Build the per-row rope angle tables from the packed layout's position grid.
///
/// Partial split-half rope: pair `i` of row `p` rotates dims `(i, i + pairs)` of
/// every head by `pos[p][i / inv_len] * inv_freq[i % inv_len]`, and the dims past
/// `2 * pairs` pass through untouched. The reference reaches the same table via
/// `cat(half, half)` and then reading only the first half, so the duplicate is
/// never materialized here.
pub fn ropeFreqs(gpa: std.mem.Allocator, pos: []const Pos, inv_freq: []const f32) !ops.rope.Freqs {
    const inv_len = inv_freq.len;
    const pairs = inv_len * 3;
    const cos = try gpa.alloc(f32, pos.len * pairs);
    errdefer gpa.free(cos);
    const sin = try gpa.alloc(f32, pos.len * pairs);
    errdefer gpa.free(sin);
    for (pos, 0..) |p, row| {
        for (0..3) |axis| {
            for (0..inv_len) |i| {
                // f64 position (the grid is area-normalized and can reach ~32),
                // f32 frequency: the reference casts the grid to f32 first, so
                // the product is computed at f32 there too.
                const ang = @as(f32, @floatCast(p[axis])) * inv_freq[i];
                const at = row * pairs + axis * inv_len + i;
                cos[at] = @cos(ang);
                sin[at] = @sin(ang);
            }
        }
    }
    return .{ .cos = cos, .sin = sin, .half = pairs };
}

// --- forward --------------------------------------------------------------

/// Per-sequence scratch. Sized once per shape rather than once per step: at the
/// default render `h` alone is 38k x 5376 x 4 B = 817 MB, so churning these
/// would dominate.
pub const Workspace = struct {
    seq: usize,
    /// The packed stream, [seq][hidden].
    h: []f32,
    /// Normalized-and-modulated copy of `h`, same shape.
    hn: []f32,
    /// Fused qkv, [seq][3 * inner].
    qkv: []f32,
    /// One sublayer's output, [seq][hidden]. Hidden rather than inner: both
    /// sublayers end in a projection back to the residual width.
    blk: []f32,
    /// Fused swiglu gate+value, [seq][2 * ffn].
    ff: []f32,
    /// Time embedding, [n_labels][t_dim].
    t_emb: []f32,
    /// One block's modulation vectors, [n_labels * 3 * 6][hidden].
    mod: []f32,
    /// Row buffers for the two streams' patchified latents.
    video_rows: []f32,
    audio_rows: []f32,

    /// Host bytes `init` will allocate for this shape, so a caller can refuse
    /// before trying.
    ///
    /// Worth having rather than discovering: the packed sequence is ~38k rows at
    /// the default render, and `h` alone is then 38k x 5376 x 4 B = 817 MB. The
    /// CPU path is a reference implementation, and asking it for a full-resolution
    /// clip is several GB of activations before any weights.
    pub fn bytesFor(cfg: Config, layout: *const PackedLayout) usize {
        const seq = layout.seq_len;
        const inner = cfg.n_heads * cfg.head_dim;
        const frame_rows = (layout.latent_h / patch_h) * (layout.latent_w / patch_w);
        const elems = seq * cfg.hidden * 2 // h, hn
        + seq * 3 * inner // qkv
        + seq * cfg.hidden // blk
        + seq * 2 * cfg.ffn // ff
        + Timesteps.max_labels * cfg.time_embed_dim // t_emb
        + Timesteps.max_labels * modality_count * 6 * cfg.hidden // mod
        + (layout.video_cond_rows + layout.latent_t * frame_rows) * cfg.videoPatchDim() +
            (layout.audio_cond_rows + layout.audio_t * audio_channels) * audio_latent_channels;
        return elems * @sizeOf(f32);
    }

    pub fn init(gpa: std.mem.Allocator, cfg: Config, layout: *const PackedLayout) !Workspace {
        const seq = layout.seq_len;
        const inner = cfg.n_heads * cfg.head_dim;
        const n_labels = Timesteps.max_labels;
        var w: Workspace = .{
            .seq = seq,
            .h = &.{},
            .hn = &.{},
            .qkv = &.{},
            .blk = &.{},
            .ff = &.{},
            .t_emb = &.{},
            .mod = &.{},
            .video_rows = &.{},
            .audio_rows = &.{},
        };
        errdefer w.deinit(gpa);
        w.h = try gpa.alloc(f32, seq * cfg.hidden);
        w.hn = try gpa.alloc(f32, seq * cfg.hidden);
        w.qkv = try gpa.alloc(f32, seq * 3 * inner);
        w.blk = try gpa.alloc(f32, seq * cfg.hidden);
        w.ff = try gpa.alloc(f32, seq * 2 * cfg.ffn);
        w.t_emb = try gpa.alloc(f32, n_labels * cfg.time_embed_dim);
        w.mod = try gpa.alloc(f32, n_labels * modality_count * 6 * cfg.hidden);
        const frame_rows = (layout.latent_h / patch_h) * (layout.latent_w / patch_w);
        w.video_rows = try gpa.alloc(f32, (layout.video_cond_rows + layout.latent_t * frame_rows) * cfg.videoPatchDim());
        w.audio_rows = try gpa.alloc(f32, (layout.audio_cond_rows + layout.audio_t * audio_channels) * audio_latent_channels);
        return w;
    }

    pub fn deinit(self: *Workspace, gpa: std.mem.Allocator) void {
        inline for (.{ self.h, self.hn, self.qkv, self.blk, self.ff, self.t_emb, self.mod, self.video_rows, self.audio_rows }) |b| {
            if (b.len > 0) gpa.free(b);
        }
        self.* = undefined;
    }
};

/// One denoiser input. The two streams stay separate here rather than arriving as
/// one packed buffer: the sampler's pack is a storage decision, and the forward
/// needs each stream's own latent (the audio carry is undone before this point).
pub const Inputs = struct {
    /// Planar `[24][t][h][w]`.
    video: []const f32,
    /// Planar `[32][2][audio_t]`.
    audio: []const f32,
    /// `[text_len][hidden]`, already through `condition_proj` + the token refiner
    /// (see `refineText`, which runs once per sampling run rather than per step).
    text: []const f32,
    /// The VIDEO sigma. The audio stream's is derived from it and `shifts`.
    sigma: f32,
    /// Per-row denoise mask for the VIDEO target segment (1 = generate), one value
    /// per 2x2 patch row, `latent_t * frame_rows` long. Empty generates everything.
    /// Reduced ONCE PER RENDER by `maskRowValues`, not per step: it depends on the
    /// latent grid, and only the mapping to timestep labels moves with sigma.
    video_mask: []const f32 = &.{},
    /// The same for the AUDIO target segment, `2 * audio_t` long and NOT patched.
    audio_mask: []const f32 = &.{},
    /// Both stream shifts. Defaulted to the family's own, so a caller that does
    /// not care never mentions them; a caller that overrides the video shift MUST
    /// pass its pair here too, or the audio timestep labels come from a schedule
    /// nobody is sampling.
    shifts: Shifts = .{},
    /// Condition rows, already patchified/packed, in packed-layout order. Their
    /// count must equal the layout's `video_cond_rows` / `audio_cond_rows`.
    cond_video: []const f32 = &.{},
    cond_audio: []const f32 = &.{},
    visual_cond_aug: f32 = visual_cond_timestep,
    audio_cond_aug: f32 = audio_cond_timestep,
};

/// `condition_proj` + the token refiner: text encoder states -> the trunk's
/// width. Constant across a sampling run, so the caller holds the result.
pub fn refineText(
    dit: *const DiT,
    io: std.Io,
    gpa: std.mem.Allocator,
    out: []f32,
    states: []const f32,
    text_len: usize,
) !void {
    const cfg = dit.cfg;
    std.debug.assert(states.len == text_len * cfg.text_dim);
    std.debug.assert(out.len == text_len * cfg.hidden);
    try ops.matmul.matmul(io, gpa, out, states, text_len, dit.condition_proj, dit.condition_bias);

    const inner = cfg.n_heads * cfg.head_dim;
    const hn = try gpa.alloc(f32, text_len * cfg.hidden);
    defer gpa.free(hn);
    const qkv = try gpa.alloc(f32, text_len * 3 * inner);
    defer gpa.free(qkv);
    // A block's output is HIDDEN wide, not inner: both sublayers end in a
    // projection back to the residual width.
    const blk = try gpa.alloc(f32, text_len * cfg.hidden);
    defer gpa.free(blk);
    const ff = try gpa.alloc(f32, text_len * 2 * cfg.ffn);
    defer gpa.free(ff);

    for (dit.refiner) |*b| {
        // Plain pre-norm residuals, and NO rope: the refiner has no position
        // information at all.
        ops.norm.rmsNorm(hn, out, b.norm1, cfg.norm_eps);
        try runAttn(dit, io, gpa, blk, hn, text_len, &b.attn, qkv, null);
        for (out, blk) |*o, a| o.* += a;
        ops.norm.rmsNorm(hn, out, b.norm2, cfg.norm_eps);
        try runMlp(dit, io, gpa, blk, hn, text_len, &b.mlp, ff);
        for (out, blk) |*o, a| o.* += a;
    }
    ops.norm.rmsNorm(out, out, dit.refiner_final_norm, cfg.final_norm_eps);
}

/// Fused qkv -> per-head Q/K RMSNorm -> optional partial split-half rope -> full
/// (non-causal) attention -> output projection.
fn runAttn(
    dit: *const DiT,
    io: std.Io,
    gpa: std.mem.Allocator,
    out: []f32,
    x: []const f32,
    seq: usize,
    a: *const Attn,
    qkv: []f32,
    freqs: ?ops.rope.Freqs,
) !void {
    const cfg = dit.cfg;
    const inner = cfg.n_heads * cfg.head_dim;
    try matLin(io, gpa, qkv[0 .. seq * 3 * inner], 3 * inner, x, seq, a.qkv, null);

    // The projection emits [seq][3][inner] per row; attention wants three
    // separate [seq][n_heads][head_dim] planes, so de-interleave in place-ish.
    const q = try gpa.alloc(f32, seq * inner);
    defer gpa.free(q);
    const k = try gpa.alloc(f32, seq * inner);
    defer gpa.free(k);
    const v = try gpa.alloc(f32, seq * inner);
    defer gpa.free(v);
    for (0..seq) |i| {
        const src = qkv[i * 3 * inner ..];
        @memcpy(q[i * inner ..][0..inner], src[0..inner]);
        @memcpy(k[i * inner ..][0..inner], src[inner..][0..inner]);
        @memcpy(v[i * inner ..][0..inner], src[2 * inner ..][0..inner]);
    }

    // Per-head norms: the weight is head_dim wide, so rmsNorm's row loop walks
    // one head at a time over the whole buffer.
    ops.norm.rmsNorm(q, q, a.q_norm, cfg.qk_norm_eps);
    ops.norm.rmsNorm(k, k, a.k_norm, cfg.qk_norm_eps);

    if (freqs) |f| {
        ops.rope.applyRotateHalfPartialAt(q, f, 0, seq, cfg.n_heads, cfg.head_dim, cfg.ropeRotDim());
        ops.rope.applyRotateHalfPartialAt(k, f, 0, seq, cfg.n_heads, cfg.head_dim, cfg.ropeRotDim());
    }

    const att = try gpa.alloc(f32, seq * inner);
    defer gpa.free(att);
    try ops.attention.attention(io, gpa, att, q, k, v, .{
        .seq_q = seq,
        .seq_kv = seq,
        .n_heads = cfg.n_heads,
        .n_kv_heads = cfg.n_heads,
        .head_dim = cfg.head_dim,
        // Full attention over the whole pack: text, conditions and both target
        // streams all see each other. There is no mask anywhere in H3.
        .causal = false,
    });
    try matLin(io, gpa, out[0 .. seq * cfg.hidden], cfg.hidden, att, seq, a.out, null);
}

fn runMlp(
    dit: *const DiT,
    io: std.Io,
    gpa: std.mem.Allocator,
    out: []f32,
    x: []const f32,
    seq: usize,
    m: *const Mlp,
    ff: []f32,
) !void {
    const cfg = dit.cfg;
    const two = 2 * cfg.ffn;
    try matLin(io, gpa, ff[0 .. seq * two], two, x, seq, m.fc1, null);
    // [gate; value] per row, gate first. `siluMul` wants the two halves as
    // separate slices, and they are strided by row here, so walk rows.
    const packed_gate = try gpa.alloc(f32, seq * cfg.ffn);
    defer gpa.free(packed_gate);
    for (0..seq) |i| {
        const row = ff[i * two ..][0..two];
        const g = packed_gate[i * cfg.ffn ..][0..cfg.ffn];
        @memcpy(g, row[0..cfg.ffn]);
        ops.act.siluMul(g, row[cfg.ffn..][0..cfg.ffn]);
    }
    try matLin(io, gpa, out[0 .. seq * cfg.hidden], cfg.hidden, packed_gate, seq, m.fc2, null);
}

/// Patchify/pack both streams, project them, and assemble the packed sequence
/// into `out_h` (`[seq][hidden]`).
///
/// Shared by the CPU forward and the device one: everything here is host work in
/// both (the projections are tiny next to the trunk, and `audio_patch_proj` is
/// 32 columns wide, below every tiled GEMM's floor). Returns the step's
/// `Timesteps`, which the caller needs for the modulation either way.
pub fn embedPacked(
    dit: *const DiT,
    io: std.Io,
    gpa: std.mem.Allocator,
    out_h: []f32,
    layout: *const PackedLayout,
    in: Inputs,
) !Timesteps {
    const cfg = dit.cfg;
    const seq = layout.seq_len;
    const hidden = cfg.hidden;
    std.debug.assert(out_h.len == seq * hidden);
    std.debug.assert(in.text.len == layout.text_len * hidden);

    const t = layout.latent_t;
    const lh = layout.latent_h;
    const lw = layout.latent_w;
    const at = layout.audio_t;
    const patch_dim = cfg.videoPatchDim();
    const frame_rows = (lh / patch_h) * (lw / patch_w);
    const n_video_rows = layout.video_cond_rows + t * frame_rows;
    const n_audio_rows = layout.audio_cond_rows + at * audio_channels;

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    // Condition rows come first in the packed order, so each stream's rows are
    // `cond ++ target` and the split is the layout's cond count.
    std.debug.assert(in.cond_video.len == layout.video_cond_rows * patch_dim);
    std.debug.assert(in.cond_audio.len == layout.audio_cond_rows * audio_latent_channels);
    const video_rows = try a.alloc(f32, n_video_rows * patch_dim);
    @memcpy(video_rows[0..in.cond_video.len], in.cond_video);
    patchifyVideo(video_rows[in.cond_video.len..], in.video, t, lh, lw);
    const audio_rows = try a.alloc(f32, n_audio_rows * audio_latent_channels);
    @memcpy(audio_rows[0..in.cond_audio.len], in.cond_audio);
    packAudio(audio_rows[in.cond_audio.len..], in.audio, at);

    const video_embed = try a.alloc(f32, n_video_rows * hidden);
    const audio_embed = try a.alloc(f32, n_audio_rows * hidden);
    try ops.matmul.matmul(io, gpa, video_embed, video_rows, n_video_rows, dit.video_patch, dit.video_patch_bias);
    try ops.matmul.matmul(io, gpa, audio_embed, audio_rows, n_audio_rows, dit.audio_patch, dit.audio_patch_bias);

    // Segments are contiguous and in packed order, and each stream's embedded
    // rows are consumed in that same order.
    var v_off: usize = 0;
    var a_off: usize = 0;
    for (layout.segments) |sg| {
        const n = sg.len();
        const dst = out_h[sg.start * hidden ..][0 .. n * hidden];
        if (sg.kind == .text) {
            @memcpy(dst, in.text);
        } else if (sg.kind.isVideoRow()) {
            @memcpy(dst, video_embed[v_off * hidden ..][0 .. n * hidden]);
            v_off += n;
        } else {
            @memcpy(dst, audio_embed[a_off * hidden ..][0 .. n * hidden]);
            a_off += n;
        }
    }
    std.debug.assert(v_off == n_video_rows and a_off == n_audio_rows);

    if (in.video_mask.len > 0) std.debug.assert(in.video_mask.len == t * frame_rows);
    if (in.audio_mask.len > 0) std.debug.assert(in.audio_mask.len == at * audio_channels);
    // The result OWNS its per-row index tables when a mask is in play, so the
    // caller deinits it. `gpa` rather than the arena above, which is about to go.
    return Timesteps.initMasked(
        gpa,
        in.sigma,
        in.visual_cond_aug,
        in.audio_cond_aug,
        in.shifts,
        in.video_mask,
        in.audio_mask,
    );
}

/// The output heads: norm + modulate each target segment, project, unpack, negate.
///
/// `trunk` is the packed sequence after the last block, on the host. Shared by
/// both paths: the heads are 96 and 32 rows, below `opI8Gemm`'s 128-row floor, so
/// the device path downloads and finishes here.
pub fn finalHeads(
    dit: *const DiT,
    io: std.Io,
    gpa: std.mem.Allocator,
    layout: *const PackedLayout,
    ts: *const Timesteps,
    t_emb: []const f32,
    trunk: []const f32,
    out_video: []f32,
    out_audio: []f32,
) !void {
    const cfg = dit.cfg;
    const hidden = cfg.hidden;
    const labels = ts.labels();

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    // The final layer's adaLN has ONE modality, so its modulation row is the
    // timestep row itself rather than `t_row * 3 + tag`.
    const fmod = try a.alloc(f32, labels.len * 2 * hidden);
    try ops.matmul.matmul(io, gpa, fmod, t_emb, labels.len, dit.final.adaln, dit.final.adaln_bias);

    const video_seg = layout.segmentOf(.video).?;
    const audio_seg = layout.segmentOf(.audio).?;
    const scratch = try a.alloc(f32, @max(video_seg.len(), audio_seg.len()) * hidden);
    const v_rows = try a.alloc(f32, video_seg.len() * cfg.videoPatchDim());
    const a_rows = try a.alloc(f32, audio_seg.len() * audio_latent_channels);
    try finalHead(dit, io, gpa, v_rows, trunk, scratch, fmod, video_seg, ts.rowFor(.video), ts.rowsFor(.video), dit.final.video_out, dit.final.video_bias);
    try finalHead(dit, io, gpa, a_rows, trunk, scratch, fmod, audio_seg, ts.rowFor(.audio), ts.rowsFor(.audio), dit.final.audio_out, dit.final.audio_bias);

    // Unpack, then NEGATE. Both streams: the reference's last act is
    // `[-video_out, -audio_out]`, and it is invisible in every norm.
    unpatchifyVideo(out_video, v_rows, layout.latent_t, layout.latent_h, layout.latent_w);
    for (out_video) |*x| x.* = -x.*;
    unpackAudio(out_audio, a_rows, layout.audio_t);
    for (out_audio) |*x| x.* = -x.*;
}

/// One denoiser step: the packed forward over both streams.
///
/// `out_video` / `out_audio` receive the VELOCITY for each stream, in the same
/// planar layouts the inputs use.
pub fn forward(
    dit: *const DiT,
    io: std.Io,
    gpa: std.mem.Allocator,
    ws: *Workspace,
    layout: *const PackedLayout,
    out_video: []f32,
    out_audio: []f32,
    in: Inputs,
) !void {
    const cfg = dit.cfg;
    const seq = layout.seq_len;
    const hidden = cfg.hidden;
    std.debug.assert(ws.seq == seq);

    var ts = try embedPacked(dit, io, gpa, ws.h, layout, in);
    defer ts.deinit(gpa);
    const labels = ts.labels();
    const t_dim = cfg.time_embed_dim;
    const t_emb = ws.t_emb[0 .. labels.len * t_dim];
    timeEmbed(t_emb, dit.adaln_t_table, cfg.adaln_curve_grid.?, t_dim, labels);

    var freqs = try ropeFreqs(gpa, layout.pos, dit.rope_inv_freq);
    defer freqs.deinit(gpa);

    const mod_rows = labels.len * modality_count;
    for (dit.blocks) |*b| {
        // adaLN reads the curve coordinates directly: curve-form checkpoints fold
        // the silu the non-curve form applies into the table itself.
        try ops.matmul.matmul(io, gpa, ws.mod[0 .. mod_rows * 6 * hidden], t_emb, labels.len, b.adaln, b.adaln_bias);

        ops.norm.rmsNorm(ws.hn, ws.h, b.norm1, cfg.norm_eps);
        modScaleShift(ws.hn, ws.mod, layout, &ts, hidden, 0, 1);
        try runAttn(dit, io, gpa, ws.blk, ws.hn, seq, &b.attn, ws.qkv, freqs);
        modGate(ws.h, ws.blk, ws.mod, layout, &ts, hidden, 2);

        ops.norm.rmsNorm(ws.hn, ws.h, b.norm2, cfg.norm_eps);
        modScaleShift(ws.hn, ws.mod, layout, &ts, hidden, 3, 4);
        try runMlp(dit, io, gpa, ws.blk, ws.hn, seq, &b.mlp, ws.ff);
        modGate(ws.h, ws.blk, ws.mod, layout, &ts, hidden, 5);
    }

    try finalHeads(dit, io, gpa, layout, &ts, t_emb, ws.h, out_video, out_audio);
}


/// One output head: norm + modulate the stream's segment, then project to
/// `rows_out` ([n][w.rows] patch/channel rows, which the caller unpacks).
fn finalHead(
    dit: *const DiT,
    io: std.Io,
    gpa: std.mem.Allocator,
    rows_out: []f32,
    trunk: []const f32,
    scratch: []f32,
    fmod: []const f32,
    seg: Segment,
    t_row: usize,
    /// Per-row label indices when a denoise mask relabelled this segment, empty
    /// otherwise. ⚠️ **The output head modulates per row too.** Fixing only the
    /// trunk leaves the heads reading the segment's own label, which for a
    /// COLLAPSED mask is the mask's value and for a per-row one is the unmasked
    /// stream's: the same mask then renders two different pictures depending on
    /// how it happened to be expressed.
    rows: []const u8,
    w: Weight,
    bias: []const f32,
) !void {
    const hidden = dit.cfg.hidden;
    const n = seg.len();
    std.debug.assert(rows_out.len == n * w.rows);
    std.debug.assert(rows.len == 0 or rows.len == n);
    const src = trunk[seg.start * hidden ..][0 .. n * hidden];
    const dst = scratch[0 .. n * hidden];
    ops.norm.rmsNorm(dst, src, dit.final.norm, dit.cfg.final_norm_eps);
    // Two slots, one modality: `fmod` is [n_labels][2][hidden], shift then scale.
    // With one modality the row index IS the timestep row, which is what the
    // reference's `rows_to_mod_index(...) // 3` recovers.
    for (0..n) |i| {
        const r = if (rows.len == 0) t_row else rows[i];
        const shift = fmod[r * 2 * hidden ..][0..hidden];
        const scale = fmod[r * 2 * hidden + hidden ..][0..hidden];
        const row = dst[i * hidden ..][0..hidden];
        for (row, scale, shift) |*x, sc, sh| x.* = x.* * (1.0 + sc) + sh;
    }
    try ops.matmul.matmul(io, gpa, rows_out, dst, n, w, bias);
}

/// `h[a:b] = h[a:b] * (1 + scale[row]) + shift[row]`, per segment.
fn modScaleShift(
    h: []f32,
    mod: []const f32,
    layout: *const PackedLayout,
    ts: *const Timesteps,
    hidden: usize,
    shift_slot: usize,
    scale_slot: usize,
) void {
    for (layout.segments) |sg| {
        const tag = sg.kind.tag();
        // A denoise mask relabels rows WITHIN a target segment, so the modulation
        // vector is picked per row there and once per segment everywhere else.
        const rows = ts.rowsFor(sg.kind);
        if (rows.len == 0) {
            const t_row = ts.rowFor(sg.kind);
            const shift = mod[modRowOffset(t_row, tag, shift_slot, hidden)..][0..hidden];
            const scale = mod[modRowOffset(t_row, tag, scale_slot, hidden)..][0..hidden];
            for (sg.start..sg.stop) |i| {
                const row = h[i * hidden ..][0..hidden];
                for (row, scale, shift) |*x, sc, sh| x.* = x.* * (1.0 + sc) + sh;
            }
        } else {
            std.debug.assert(rows.len == sg.len());
            for (sg.start..sg.stop, rows) |i, r| {
                const shift = mod[modRowOffset(r, tag, shift_slot, hidden)..][0..hidden];
                const scale = mod[modRowOffset(r, tag, scale_slot, hidden)..][0..hidden];
                const row = h[i * hidden ..][0..hidden];
                for (row, scale, shift) |*x, sc, sh| x.* = x.* * (1.0 + sc) + sh;
            }
        }
    }
}

/// `x[a:b] += other[a:b] * gate[row]`, per segment.
fn modGate(
    x: []f32,
    other: []const f32,
    mod: []const f32,
    layout: *const PackedLayout,
    ts: *const Timesteps,
    hidden: usize,
    gate_slot: usize,
) void {
    for (layout.segments) |sg| {
        const tag = sg.kind.tag();
        const rows = ts.rowsFor(sg.kind);
        if (rows.len == 0) {
            const t_row = ts.rowFor(sg.kind);
            const gate = mod[modRowOffset(t_row, tag, gate_slot, hidden)..][0..hidden];
            for (sg.start..sg.stop) |i| {
                const dst = x[i * hidden ..][0..hidden];
                const src = other[i * hidden ..][0..hidden];
                for (dst, src, gate) |*d, s, g| d.* += s * g;
            }
        } else {
            std.debug.assert(rows.len == sg.len());
            for (sg.start..sg.stop, rows) |i, r| {
                const gate = mod[modRowOffset(r, tag, gate_slot, hidden)..][0..hidden];
                const dst = x[i * hidden ..][0..hidden];
                const src = other[i * hidden ..][0..hidden];
                for (dst, src, gate) |*d, s, g| d.* += s * g;
            }
        }
    }
}

// --- tests ----------------------------------------------------------------

test "frame counts snap to the 17k+5 grid" {
    try std.testing.expectEqual(@as(usize, 5), alignFrameCount(1));
    try std.testing.expectEqual(@as(usize, 5), alignFrameCount(5));
    try std.testing.expectEqual(@as(usize, 22), alignFrameCount(6));
    try std.testing.expectEqual(@as(usize, 22), alignFrameCount(22));
    try std.testing.expectEqual(@as(usize, 39), alignFrameCount(23));
    try std.testing.expectEqual(@as(usize, 124), alignFrameCount(124));
    try std.testing.expectEqual(@as(usize, 141), alignFrameCount(125));
}

test "video latent tokens and their frame span are inverses on the grid" {
    // the default 124-frame (~5s) render
    try std.testing.expectEqual(@as(usize, 37), videoLatentT(124));
    try std.testing.expectEqual(@as(usize, 2), videoLatentT(5));
    try std.testing.expectEqual(@as(usize, 7), videoLatentT(22));

    // every aligned length round-trips: k tokens cover exactly k's frames
    var frames: usize = 5;
    while (frames <= 400) : (frames += 17) {
        errdefer std.debug.print("frames={d} latent_t={d} span={d}\n", .{ frames, videoLatentT(frames), framesForLatentT(videoLatentT(frames)) });
        try std.testing.expectEqual(frames, framesForLatentT(videoLatentT(frames)));
    }
}

test "audio latent frames match round(frames / 24 * 40)" {
    // the integer form must agree with the float one on the whole grid, and the
    // tie case it cannot represent must genuinely never occur
    var frames: usize = 5;
    while (frames <= 3600) : (frames += 1) {
        const exact = 5.0 * @as(f64, @floatFromInt(frames)) / 3.0;
        errdefer std.debug.print("frames={d} exact={d} got={d}\n", .{ frames, exact, audioLatentT(frames) });
        // the tie the integer form cannot represent is a HALF fraction, which
        // needs 4 * frames == 3 (mod 6): even against odd, so it never occurs
        try std.testing.expect(@abs(exact - @floor(exact) - 0.5) > 1e-9);
        try std.testing.expectEqual(@as(usize, @intFromFloat(@round(exact))), audioLatentT(frames));
    }
    try std.testing.expectEqual(@as(usize, 207), audioLatentT(124));
    try std.testing.expectEqual(@as(usize, 8), audioLatentT(5));
}

test "temporal shape of the default render" {
    const s = temporalShape(124);
    try std.testing.expectEqual(@as(usize, 124), s.frames);
    try std.testing.expectEqual(@as(usize, 37), s.latent_t);
    try std.testing.expectEqual(@as(usize, 207), s.audio_t);
}

test "axis grid is area-normalized, not indices" {
    const gpa = std.testing.allocator;
    // a square latent: ratio is 1, so the axis is linspace(0, 1, n) * 32
    const sq = try axisFromSqrtArea(gpa, 16, 2, 16.0);
    defer gpa.free(sq);
    try std.testing.expectEqual(@as(usize, 8), sq.len);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), sq[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), sq[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 28.0), sq[7], 1e-12);

    // the default 1344x768 canvas: h is the short edge, so its axis is centered
    // inside [0, 32] and w's runs outside it
    const area = @sqrt(@as(f64, 48 * 84));
    const h = try axisFromSqrtArea(gpa, 48, 2, area);
    defer gpa.free(h);
    const w = try axisFromSqrtArea(gpa, 84, 2, area);
    defer gpa.free(w);
    try std.testing.expectEqual(@as(usize, 24), h.len);
    try std.testing.expectEqual(@as(usize, 42), w.len);
    try std.testing.expect(h[0] > 0.0);
    try std.testing.expect(w[0] < 0.0);
    // both axes step the same distance: the normalization is by area, so a
    // patch is square on the grid whatever the aspect ratio
    try std.testing.expectApproxEqAbs(h[1] - h[0], w[1] - w[0], 1e-12);
}

test "t2va layout is text, audio, video with the targets last" {
    const gpa = std.testing.allocator;
    // the development shape: 256x256, length 5
    var l = try PackedLayout.build(gpa, .{
        .text_len = 7,
        .latent_t = 2,
        .latent_h = 16,
        .latent_w = 16,
        .audio_t = 8,
    }, &.{}, &.{}, &.{});
    defer l.deinit();

    try std.testing.expectEqual(@as(usize, 3), l.segments.len);
    try std.testing.expectEqual(Kind.text, l.segments[0].kind);
    try std.testing.expectEqual(Kind.audio, l.segments[1].kind);
    try std.testing.expectEqual(Kind.video, l.segments[2].kind);
    // 8 x 8 patch rows per frame, two frames
    try std.testing.expectEqual(@as(usize, 128), l.segments[2].len());
    try std.testing.expectEqual(@as(usize, 16), l.segments[1].len());
    try std.testing.expectEqual(@as(usize, 7 + 16 + 128), l.seq_len);
    try std.testing.expectEqual(@as(usize, 0), l.video_cond_rows);
    try std.testing.expectEqual(@as(usize, 0), l.audio_cond_rows);

    // segments cover the sequence contiguously
    var expect: usize = 0;
    for (l.segments) |sg| {
        try std.testing.expectEqual(expect, sg.start);
        expect = sg.stop;
    }
    try std.testing.expectEqual(l.seq_len, expect);
    try std.testing.expectEqual(l.seq_len, l.pos.len);
}

test "t2va grid: text counts rows, streams share an origin past the text" {
    const gpa = std.testing.allocator;
    var l = try PackedLayout.build(gpa, .{
        .text_len = 3,
        .latent_t = 2,
        .latent_h = 16,
        .latent_w = 16,
        .audio_t = 4,
    }, &.{}, &.{}, &.{});
    defer l.deinit();

    // text: t is the row index, h and w are zero
    try std.testing.expectEqual(@as(f64, 0), l.pos[0][0]);
    try std.testing.expectEqual(@as(f64, 2), l.pos[2][0]);
    try std.testing.expectEqual(@as(f64, 0), l.pos[2][1]);

    const audio = l.segmentOf(.audio).?;
    const video = l.segmentOf(.video).?;
    // both target streams start at the same origin, just past the text
    try std.testing.expectEqual(@as(f64, 3), l.pos[audio.start][0]);
    try std.testing.expectEqual(@as(f64, 3), l.pos[video.start][0]);

    // audio is channel-major: the time axis runs twice, w pinned per channel
    try std.testing.expectEqual(@as(f64, 4), l.pos[audio.start + 1][0]);
    try std.testing.expectEqual(@as(f64, 3), l.pos[audio.start + 4][0]);
    try std.testing.expect(l.pos[audio.start][2] < l.pos[audio.start + 4][2]);
    // and h is zero throughout
    for (audio.start..audio.stop) |i| try std.testing.expectEqual(@as(f64, 0), l.pos[i][1]);

    // the video time axis is non-uniform: token 0 spans 1 frame, token 1 spans 4
    const rows_per_frame = 64;
    try std.testing.expectApproxEqAbs(
        3.0 + frame_rescale,
        l.pos[video.start + rows_per_frame][0],
        1e-12,
    );
}

test "keyframes and refs push the target timeline out" {
    const gpa = std.testing.allocator;
    const shape: PackedLayout.Shape = .{
        .text_len = 2,
        .latent_t = 2,
        .latent_h = 16,
        .latent_w = 16,
        .audio_t = 4,
    };

    // one reference image spans 1.0 on the time axis
    var with_ref = try PackedLayout.build(gpa, shape, &.{}, &.{
        .{ .kind = .image, .latent_t = 1, .latent_h = 16, .latent_w = 16 },
    }, &.{});
    defer with_ref.deinit();

    try std.testing.expectEqual(@as(usize, 4), with_ref.segments.len);
    try std.testing.expectEqual(Kind.ref_img, with_ref.segments[1].kind);
    try std.testing.expectEqual(@as(usize, 64), with_ref.video_cond_rows);
    // the ref sits at the text's end, the targets one full span past it
    const ref_seg = with_ref.segmentOf(.ref_img).?;
    try std.testing.expectEqual(@as(f64, 2), with_ref.pos[ref_seg.start][0]);
    try std.testing.expectEqual(@as(f64, 3), with_ref.pos[with_ref.segmentOf(.video).?.start][0]);

    // a reference video emits its soundtrack BEFORE its video rows
    var with_vid = try PackedLayout.build(gpa, shape, &.{}, &.{
        .{ .kind = .video, .latent_t = 2, .latent_h = 16, .latent_w = 16, .audio_t = 3 },
    }, &.{});
    defer with_vid.deinit();
    try std.testing.expectEqual(Kind.ref_audio, with_vid.segments[1].kind);
    try std.testing.expectEqual(Kind.ref_img, with_vid.segments[2].kind);
    // span is max(audio 3, video 1 + 4 frames scaled) = max(3, 5/3 * 5)
    try std.testing.expectApproxEqAbs(
        2.0 + @max(3.0, frame_rescale * 5.0),
        with_vid.pos[with_vid.segmentOf(.video).?.start][0],
        1e-12,
    );

    // a keyframe anchored at frame 0 shares the target origin
    var with_kf = try PackedLayout.build(gpa, shape, &.{
        .{ .frame_index = 0, .latent_t = 1 },
    }, &.{}, &.{});
    defer with_kf.deinit();
    try std.testing.expectEqual(Kind.cond, with_kf.segments[1].kind);
    try std.testing.expectEqual(
        with_kf.pos[with_kf.segmentOf(.video).?.start][0],
        with_kf.pos[with_kf.segmentOf(.cond).?.start][0],
    );
}

test "condition rows always precede their target, per modality" {
    const gpa = std.testing.allocator;
    // the row streams are fed as `cond ++ target`, which is only correct if no
    // condition segment ever follows the target segment of the same modality
    var l = try PackedLayout.build(gpa, .{
        .text_len = 2,
        .latent_t = 2,
        .latent_h = 16,
        .latent_w = 16,
        .audio_t = 4,
    }, &.{
        .{ .frame_index = 0, .latent_t = 1, .audio_t = 2 },
    }, &.{
        .{ .kind = .image, .latent_t = 1, .latent_h = 32, .latent_w = 16 },
        .{ .kind = .audio, .audio_t = 3 },
    }, &.{});
    defer l.deinit();

    var seen_video_target = false;
    var seen_audio_target = false;
    var video_cond: usize = 0;
    var audio_cond: usize = 0;
    for (l.segments) |sg| {
        if (sg.kind == .video) {
            seen_video_target = true;
        } else if (sg.kind == .audio) {
            seen_audio_target = true;
        } else if (sg.kind.isVideoRow()) {
            try std.testing.expect(!seen_video_target);
            video_cond += sg.len();
        } else if (sg.kind.isAudioRow()) {
            try std.testing.expect(!seen_audio_target);
            audio_cond += sg.len();
        }
    }
    try std.testing.expectEqual(video_cond, l.video_cond_rows);
    try std.testing.expectEqual(audio_cond, l.audio_cond_rows);
}

const mask_fixture = @embedFile("assets/minimax_h3_mask.safetensors");

test "denoise-mask row values and their timesteps match the reference" {
    // From tools/gen_minimax_h3_mask.py. Six masks covering every branch: inert,
    // binary temporal (video continuation), a half-covered patch (amax vs mean), a
    // graded one that really adds labels, a uniform one that must COLLAPSE, and one
    // short of the latent grid so the replicate padding runs.
    const gpa = std.testing.allocator;
    var st = try tp_core.safetensors.SafeTensors.initFromSlice(gpa, mask_fixture);
    defer st.deinit();

    const latent_t: usize = 3;
    const lat_h: usize = 6;
    const lat_w: usize = 8;
    const sigma_v: f32 = 0.6;
    const shifts: Shifts = .{ .video = 12.0, .audio = 3.0 };

    // name, mask_h, mask_w
    const cases = [_]struct { []const u8, usize, usize }{
        .{ "all_ones", 6, 8 },
        .{ "temporal", 6, 8 },
        .{ "spatial_half_patch", 6, 8 },
        .{ "graded", 6, 8 },
        .{ "uniform_half", 6, 8 },
        .{ "short_replicate", 4, 6 },
    };

    for (cases) |c| {
        const name = c[0];
        var buf: [64]u8 = undefined;
        const mv = try st.require(try std.fmt.bufPrint(&buf, "mask.{s}", .{name}));
        const mask = try mv.toF32Alloc(gpa);
        defer gpa.free(mask);
        errdefer std.debug.print("case {s}\n", .{name});

        const got = try maskRowValues(gpa, mask, latent_t, c[1], c[2], lat_h, lat_w);
        defer if (got) |g| gpa.free(g);

        const wv = try st.require(try std.fmt.bufPrint(&buf, "rows.{s}", .{name}));
        const want = try wv.toF32Alloc(gpa);
        defer gpa.free(want);
        if (want.len == 0) {
            // An all-ones mask reduces to NOTHING, which is not the same as a table
            // of ones: it keeps the segment-level label and the label set with it.
            try std.testing.expect(got == null);
            continue;
        }
        try std.testing.expect(got != null);
        try std.testing.expectEqualSlices(f32, want, got.?);

        // ...and the per-row timesteps the mask turns into.
        var ts = try Timesteps.initMasked(
            gpa,
            sigma_v,
            visual_cond_timestep,
            audio_cond_timestep,
            shifts,
            got.?,
            &.{},
        );
        defer ts.deinit(gpa);

        const tv = try st.require(try std.fmt.bufPrint(&buf, "t.{s}", .{name}));
        const want_t = try tv.toF32Alloc(gpa);
        defer gpa.free(want_t);

        const rows = ts.rowsFor(.video);
        if (rows.len == 0) {
            // Collapsed: every row agrees, so it became the segment's own label.
            for (want_t) |v| try std.testing.expectApproxEqAbs(want_t[0], v, 1e-6);
            try std.testing.expectApproxEqAbs(want_t[0], ts.labels()[ts.rowFor(.video)], 1e-6);
        } else {
            try std.testing.expectEqual(want_t.len, rows.len);
            for (want_t, rows) |w, r| try std.testing.expectApproxEqAbs(w, ts.labels()[r], 1e-6);
        }
    }
}

test "a binary denoise mask needs no timestep label of its own" {
    // The video-continuation case, and the reason it is cheap: m = 1 gives the
    // stream's own label and m = 0 gives exactly the pinned condition one, both
    // already in the set. Only a GRADED mask grows it.
    const gpa = std.testing.allocator;
    const sigma_v: f32 = 0.6;
    const shifts: Shifts = .{ .video = 12.0, .audio = 3.0 };

    const plain = Timesteps.init(sigma_v, visual_cond_timestep, audio_cond_timestep, shifts);

    const binary = [_]f32{ 0, 0, 1, 1, 0, 1 };
    var bin = try Timesteps.initMasked(gpa, sigma_v, visual_cond_timestep, audio_cond_timestep, shifts, &binary, &.{});
    defer bin.deinit(gpa);
    try std.testing.expectEqual(plain.labels().len, bin.labels().len);
    try std.testing.expectEqualSlices(f32, plain.labels(), bin.labels());
    // ...and the rows really do split, or the test would pass on a no-op.
    try std.testing.expectEqual(binary.len, bin.rowsFor(.video).len);
    try std.testing.expect(bin.rowsFor(.video)[0] != bin.rowsFor(.video)[2]);

    const graded = [_]f32{ 0, 0.25, 0.5, 0.75, 1.0, 0.5 };
    var grd = try Timesteps.initMasked(gpa, sigma_v, visual_cond_timestep, audio_cond_timestep, shifts, &graded, &.{});
    defer grd.deinit(gpa);
    try std.testing.expect(grd.labels().len > plain.labels().len);
    // Labels stay sorted and deduped whatever order the mask presents them in.
    for (grd.labels()[1..], grd.labels()[0 .. grd.labels().len - 1]) |b, a| {
        try std.testing.expect(b > a);
    }

    // Too many levels is refused by name rather than quantized: merging levels
    // would change the picture, not just the table.
    var many: [Timesteps.max_labels + 4]f32 = undefined;
    for (&many, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) / @as(f32, many.len);
    try std.testing.expectError(
        error.TooManyTimestepLabels,
        Timesteps.initMasked(gpa, sigma_v, visual_cond_timestep, audio_cond_timestep, shifts, &many, &.{}),
    );
}

test "the audio denoise mask uses the audio stream's own sigma" {
    // The two streams run on different sigmas, so using the video one for both is
    // exactly right at shift parity and wrong at the shipped 12/3.
    const gpa = std.testing.allocator;
    var st = try tp_core.safetensors.SafeTensors.initFromSlice(gpa, mask_fixture);
    defer st.deinit();
    const mv = try st.require("audio.mask");
    const mask = try mv.toF32Alloc(gpa);
    defer gpa.free(mask);
    const tv = try st.require("audio.t");
    const want = try tv.toF32Alloc(gpa);
    defer gpa.free(want);

    const shifts: Shifts = .{ .video = 12.0, .audio = 3.0 };
    var ts = try Timesteps.initMasked(gpa, 0.6, visual_cond_timestep, audio_cond_timestep, shifts, &.{}, mask);
    defer ts.deinit(gpa);
    const rows = ts.rowsFor(.audio);
    try std.testing.expectEqual(want.len, rows.len);
    for (want, rows) |w, r| try std.testing.expectApproxEqAbs(w, ts.labels()[r], 1e-6);
    // The video segment is untouched by an audio mask.
    try std.testing.expectEqual(@as(usize, 0), ts.rowsFor(.video).len);

    // At shift PARITY the two sigmas coincide, so the same mask on either stream
    // gives the same GENERATED-row label -- which is why a parity render cannot
    // test the stream sigmas at all. The PRESERVED rows still differ, because the
    // two streams pin at different condition timesteps (1.0 against 0.999); that
    // is a separate constant, not the sigma.
    const par: Shifts = .{ .video = 5.0, .audio = 5.0 };
    var a_par = try Timesteps.initMasked(gpa, 0.6, visual_cond_timestep, audio_cond_timestep, par, &.{}, mask);
    defer a_par.deinit(gpa);
    var v_par = try Timesteps.initMasked(gpa, 0.6, visual_cond_timestep, audio_cond_timestep, par, mask, &.{});
    defer v_par.deinit(gpa);
    var checked: usize = 0;
    for (mask, a_par.rowsFor(.audio), v_par.rowsFor(.video)) |m, ar, vr| {
        if (m < 1.0) continue;
        try std.testing.expectApproxEqAbs(a_par.labels()[ar], v_par.labels()[vr], 1e-6);
        checked += 1;
    }
    try std.testing.expect(checked > 0);
    // ...and at the shipped 12/3 they do NOT agree, which is the whole point.
    for (mask, ts.rowsFor(.audio), v_par.rowsFor(.video)) |m, ar, vr| {
        if (m < 1.0) continue;
        try std.testing.expect(@abs(ts.labels()[ar] - v_par.labels()[vr]) > 1e-3);
        break;
    }
}

test "a reference video's soundtrack packs before its frames and on its own width axis" {
    const gpa = std.testing.allocator;
    // One reference video carrying audio, one standalone audio reference. Both
    // contribute `ref_audio` rows, and the two take their width coordinates from
    // DIFFERENT places: the video's soundtrack rides the VIDEO's grid bounds, the
    // standalone one the target's. Same segment kind, same row count, different
    // positions -- so a port that used one rule for both is a silent wrong answer.
    const ref_h = 32;
    const ref_w = 16;
    var l = try PackedLayout.build(gpa, .{
        .text_len = 2,
        .latent_t = 2,
        .latent_h = 16,
        .latent_w = 16,
        .audio_t = 4,
    }, &.{}, &.{
        .{ .kind = .video, .latent_t = 2, .latent_h = ref_h, .latent_w = ref_w, .audio_t = 3 },
        .{ .kind = .audio, .audio_t = 2 },
    }, &.{});
    defer l.deinit();

    // Two ref_audio segments and one ref_img, and the FIRST ref_audio comes before
    // the ref_img: a block's audio rows pack immediately ahead of its video rows.
    var first_audio: ?usize = null;
    var second_audio: ?usize = null;
    var img_at: ?usize = null;
    for (l.segments, 0..) |sg, si| {
        if (sg.kind == .ref_audio) {
            if (first_audio == null) first_audio = si else second_audio = si;
        } else if (sg.kind == .ref_img) img_at = si;
    }
    try std.testing.expect(first_audio != null and second_audio != null and img_at != null);
    try std.testing.expect(first_audio.? < img_at.?);
    try std.testing.expect(img_at.? < second_audio.?);

    const vid_audio = l.segments[first_audio.?];
    const solo_audio = l.segments[second_audio.?];
    try std.testing.expectEqual(@as(usize, 3 * audio_channels), vid_audio.len());
    try std.testing.expectEqual(@as(usize, 2 * audio_channels), solo_audio.len());

    // The video's soundtrack shares the video block's cursor origin...
    const img = l.segments[img_at.?];
    try std.testing.expectEqual(l.pos[img.start][0], l.pos[vid_audio.start][0]);
    // ...and takes its width coordinates from the video's own grid, which for a
    // 32x16 reference latent is NOT the target's 16x16 axis.
    const ref_axis = try axisFromSqrtArea(gpa, ref_w, patch_w, @sqrt(@as(f64, ref_h) * ref_w));
    defer gpa.free(ref_axis);
    try std.testing.expectApproxEqAbs(ref_axis[0], l.pos[vid_audio.start][2], 1e-9);
    try std.testing.expect(@abs(l.pos[vid_audio.start][2] - l.pos[solo_audio.start][2]) > 1e-6);

    // Both count toward the audio condition rows, and nothing counts twice.
    try std.testing.expectEqual(
        @as(usize, (3 + 2) * audio_channels),
        l.audio_cond_rows,
    );
}

test "a reference video's canvas is aspect-driven and never upscales" {
    // 768 px short edge under a 768x1344 cap, per axis rounded to 32 -- driven by
    // the ASPECT RATIO, not by either dimension. And a clip smaller than the
    // result keeps its own size: spending reference tokens on interpolated detail
    // is worse than spending fewer on real detail, and those tokens ride through
    // every sampling step.
    // 16:9 landscape: 768 tall, 768 * 16/9 = 1365 wide, capped by area.
    const wide = adaptCanvas(1920, 1080);
    try std.testing.expectEqual(@as(usize, 768), wide.h);
    try std.testing.expect(wide.w > wide.h);
    try std.testing.expect(wide.w * wide.h <= 768 * 1344 + 32 * 768); // within a rounding step of the cap
    try std.testing.expectEqual(@as(usize, 0), wide.w % 32);
    try std.testing.expectEqual(@as(usize, 0), wide.h % 32);

    // Portrait swaps which axis gets the short edge.
    const tall = adaptCanvas(1080, 1920);
    try std.testing.expectEqual(@as(usize, 768), tall.w);
    try std.testing.expect(tall.h > tall.w);

    // Square, and LARGER than the 768 canvas, so the canvas is what it gets.
    const sq = adaptCanvas(1000, 1000);
    try std.testing.expectEqual(@as(usize, 768), sq.w);
    try std.testing.expectEqual(@as(usize, 768), sq.h);

    // A SMALL clip is not upscaled: 500x500 is under 768x768, so it keeps its own
    // rounded size (500 / 32 = 15.625 -> 16 -> 512) rather than being stretched.
    const small_sq = adaptCanvas(500, 500);
    try std.testing.expectEqual(@as(usize, 512), small_sq.w);
    try std.testing.expectEqual(@as(usize, 512), small_sq.h);

    // 320x240 likewise: 240 / 32 is exactly 7.5, and half-to-EVEN takes it to 8.
    const small = adaptCanvas(320, 240);
    try std.testing.expectEqual(@as(usize, 320), small.w);
    try std.testing.expectEqual(@as(usize, 256), small.h);
    try std.testing.expect(small.w < 768);
}

test "a reference video crops DOWN to the clip grid and refuses to be too short" {
    // The model's clip lengths are `n % 17 == 5`, and a reference video is cropped
    // down to one, never padded up. Padding would invent motion.
    try std.testing.expectEqual(@as(usize, 5), try refVideoFrames(5, 999));
    try std.testing.expectEqual(@as(usize, 5), try refVideoFrames(21, 999));
    try std.testing.expectEqual(@as(usize, 22), try refVideoFrames(22, 999));
    try std.testing.expectEqual(@as(usize, 22), try refVideoFrames(38, 999));
    try std.testing.expectEqual(@as(usize, 39), try refVideoFrames(39, 999));
    // It also cannot exceed the TARGET clip's length, and is then cropped to the
    // grid again: 22 is already legal, 30 comes back to 22.
    try std.testing.expectEqual(@as(usize, 22), try refVideoFrames(100, 22));
    try std.testing.expectEqual(@as(usize, 22), try refVideoFrames(100, 30));
    // Under five frames is an error, not a pad.
    try std.testing.expectError(error.RefVideoTooShort, refVideoFrames(4, 999));
    try std.testing.expectError(error.RefVideoTooShort, refVideoFrames(0, 999));

    // The tower samples at 2 fps, so a 22-frame clip is two vision blocks, not 11.
    const stride = refVideoSampleStride();
    try std.testing.expectEqual(@as(usize, 12), stride);
    const sampled = (22 + stride - 1) / stride;
    try std.testing.expectEqual(@as(usize, 2), sampled);
}

test "a vision block's tag span and injection span differ by one row at each end" {
    // The trap this type exists for. The reference widens the TAG span by one on
    // each side (so the flanking markers are tagged with the image) and does NOT
    // widen the DEEPSTACK span (so a marker never receives a vision feature).
    // Using either for the other is finite and wrong.
    const b: VisionBlock = .{ .index = 6, .size = 4 };
    const tag_s = b.tagSpan();
    const inj = b.injectSpan();

    try std.testing.expectEqual(@as(usize, 5), tag_s.start);
    try std.testing.expectEqual(@as(usize, 11), tag_s.stop());
    try std.testing.expectEqual(@as(usize, 6), inj.start);
    try std.testing.expectEqual(@as(usize, 10), inj.stop());
    // Exactly one row wider at each end, never two and never the same.
    try std.testing.expectEqual(inj.start - 1, tag_s.start);
    try std.testing.expectEqual(inj.stop() + 1, tag_s.stop());
    try std.testing.expectEqual(inj.len + 2, tag_s.len);

    // A block at the very start has no room for a leading marker, so the tag span
    // clamps rather than wrapping to a huge length.
    const head: VisionBlock = .{ .index = 0, .size = 3 };
    try std.testing.expectEqual(@as(usize, 0), head.tagSpan().start);
    try std.testing.expectEqual(@as(usize, 4), head.tagSpan().stop());
    try std.testing.expectEqual(@as(usize, 0), head.injectSpan().start);
    try std.testing.expectEqual(@as(usize, 3), head.injectSpan().stop());
}

test "a spliced vision block splits the text region into tag runs" {
    // The prose carries modality tag 1 and a spliced vision block carries tag 0,
    // so the text region is a RUN of segments rather than one. Two things have to
    // hold and neither is visible in a render: the runs must TILE the region (the
    // modulation loop walks segments, so a gap leaves rows unmodulated and an
    // overlap modulates them twice), and a vision run's rows must stay TEXT rows
    // -- they arrive through `condition_proj`, not a patch projection.
    const gpa = std.testing.allocator;
    const shape: PackedLayout.Shape = .{ .text_len = 20, .latent_t = 2, .latent_h = 16, .latent_w = 16, .audio_t = 4 };

    // "<Picture 1>: " <block> " a prompt" -> prose, vision, prose.
    var l = try PackedLayout.build(gpa, shape, &.{}, &.{}, &.{
        .{ .start = 5, .len = 8 },
    });
    defer l.deinit();

    const text_segs = l.segments[0..3];
    try std.testing.expectEqual(Kind.text, text_segs[0].kind);
    try std.testing.expectEqual(Kind.text_vision, text_segs[1].kind);
    try std.testing.expectEqual(Kind.text, text_segs[2].kind);
    try std.testing.expectEqualSlices(usize, &.{ 0, 5, 13 }, &.{ text_segs[0].start, text_segs[1].start, text_segs[2].start });
    try std.testing.expectEqualSlices(usize, &.{ 5, 13, 20 }, &.{ text_segs[0].stop, text_segs[1].stop, text_segs[2].stop });

    // The runs tile [0, text_len) exactly.
    var covered: usize = 0;
    for (text_segs) |sg| covered += sg.len();
    try std.testing.expectEqual(shape.text_len, covered);

    // Tags differ, timesteps do not: a vision block sits in the prompt, so it
    // follows the same label and only reads a different modulation row.
    try std.testing.expectEqual(Tag.text, Kind.text.tag());
    try std.testing.expectEqual(Tag.video, Kind.text_vision.tag());
    const ts = Timesteps.init(0.5, visual_cond_timestep, audio_cond_timestep, .{});
    try std.testing.expectEqual(ts.rowFor(.text), ts.rowFor(.text_vision));

    // ...and they are NOT video rows: nothing patch-projects them.
    try std.testing.expect(!Kind.text_vision.isVideoRow());
    try std.testing.expect(!Kind.text_vision.isAudioRow());
    // The positions stay uniform over the whole region whatever the tagging is.
    for (0..shape.text_len) |i| try std.testing.expectEqual(@as(f64, @floatFromInt(i)), l.pos[i][0]);

    // A block flush against either edge emits no empty prose run.
    var head = try PackedLayout.build(gpa, shape, &.{}, &.{}, &.{.{ .start = 0, .len = 4 }});
    defer head.deinit();
    try std.testing.expectEqual(Kind.text_vision, head.segments[0].kind);
    try std.testing.expectEqual(Kind.text, head.segments[1].kind);

    var tail = try PackedLayout.build(gpa, shape, &.{}, &.{}, &.{.{ .start = 16, .len = 4 }});
    defer tail.deinit();
    try std.testing.expectEqual(Kind.text, tail.segments[0].kind);
    try std.testing.expectEqual(Kind.text_vision, tail.segments[1].kind);
    try std.testing.expectEqual(@as(usize, 20), tail.segments[1].stop);

    // Two blocks, as a two-reference prompt gives.
    var two = try PackedLayout.build(gpa, shape, &.{}, &.{}, &.{
        .{ .start = 2, .len = 4 },
        .{ .start = 10, .len = 5 },
    });
    defer two.deinit();
    try std.testing.expectEqual(Kind.text_vision, two.segments[1].kind);
    try std.testing.expectEqual(Kind.text_vision, two.segments[3].kind);
    covered = 0;
    for (two.segments[0..5]) |sg| covered += sg.len();
    try std.testing.expectEqual(shape.text_len, covered);

    // Malformed spans are refused rather than emitting segments that do not tile.
    for ([_][]const VisionSpan{
        &.{ .{ .start = 5, .len = 8 }, .{ .start = 10, .len = 2 } }, // overlapping
        &.{ .{ .start = 3, .len = 0 } }, // empty
        &.{.{ .start = 18, .len = 5 }}, // past the end
        &.{ .{ .start = 9, .len = 2 }, .{ .start = 4, .len = 2 } }, // descending
    }) |bad| {
        try std.testing.expectError(error.BadVisionSpan, PackedLayout.build(gpa, shape, &.{}, &.{}, bad));
    }
}

test "sigma shift maps between the two schedules and inverts" {
    // the shared base grid is where the two schedules agree
    const sigma_v: f64 = 0.5;
    const sigma_a = timeShiftSigma(sigma_v, shift_video, shift_audio);
    // a smaller shift pulls sigma down at the same base point
    try std.testing.expect(sigma_a < sigma_v);
    // and the mapping is invertible
    try std.testing.expectApproxEqAbs(
        sigma_v,
        timeShiftSigma(sigma_a, shift_audio, shift_video),
        1e-12,
    );
    // the endpoints are fixed points of any shift
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), timeShiftSigma(1.0, shift_video, shift_audio), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), timeShiftSigma(0.0, shift_video, shift_audio), 1e-12);
}

test "carryScale is the small-sigma limit of the per-step carry" {
    // This is the relation that makes the audio carry COMPOSE, and nothing pinned
    // it before. The sampler holds `(sigma_v / sigma_a) * x_audio`, so a step
    // undoes it with the sigma-dependent ratio while `processLatentOut` undoes it
    // once at the end with the constant `carryScale`. Those two agree only because
    // the ratio tends to `video / audio` as sigma goes to zero. If they ever
    // stopped agreeing, the audio would come out at the wrong LEVEL and nothing
    // else would look wrong.
    for ([_]Shifts{
        .{},
        .{ .video = 1.0, .audio = 3.0 },
        .{ .video = 3.0, .audio = 3.0 },
        .{ .video = 7.5, .audio = 0.5 },
    }) |s| {
        try std.testing.expectApproxEqRel(s.video / s.audio, s.carryScale(), 1e-6);
        // Approach zero and watch the ratio converge on it.
        var sigma: f32 = 1e-2;
        var prev_err: f64 = std.math.inf(f64);
        while (sigma > 1e-6) : (sigma /= 10.0) {
            const ratio = sigma / s.audioSigma(sigma);
            const err = @abs(@as(f64, ratio) - @as(f64, s.carryScale()));
            errdefer std.debug.print("shifts {d}/{d} sigma {e}: ratio {e} vs {e}\n", .{ s.video, s.audio, sigma, ratio, s.carryScale() });
            try std.testing.expect(err <= prev_err + 1e-6);
            prev_err = err;
        }
        // Relative, not absolute: the residual is f32 rounding on a ratio whose
        // own magnitude is `carryScale`, so a fixed bound would be 15x tighter
        // for `7.5 / 0.5` than for `12 / 3`.
        errdefer std.debug.print("shifts {d}/{d}: residual {e} on a scale of {d}\n", .{ s.video, s.audio, prev_err, s.carryScale() });
        try std.testing.expect(prev_err / @as(f64, s.carryScale()) < 1e-4);
    }
}

test "equal shifts make the audio carry an exact no-op" {
    // The degenerate case is worth stating: with the two shifts equal the audio
    // rides the video schedule untouched at EVERY sigma, not just in the limit.
    // A carry that was subtly wrong elsewhere would still be right here, which is
    // why this is the one setting a level bug can hide in.
    const s: Shifts = .{ .video = 5.0, .audio = 5.0 };
    try std.testing.expectEqual(@as(f32, 1.0), s.carryScale());
    for ([_]f32{ 1.0, 0.9, 0.5, 0.1, 1e-3 }) |sigma| {
        try std.testing.expectApproxEqRel(sigma, s.audioSigma(sigma), 1e-6);
    }
}

test "the timestep labels follow the caller's shifts, not the module defaults" {
    // `Timesteps.init` used to read `shift_video`/`shift_audio` directly, so a
    // caller that overrode the video shift got audio labels from a schedule
    // nobody was sampling. The audio row is the only one that moves.
    const a = Timesteps.init(0.5, visual_cond_timestep, audio_cond_timestep, .{});
    const b = Timesteps.init(0.5, visual_cond_timestep, audio_cond_timestep, .{ .video = 1.0, .audio = 3.0 });
    const av = a.values[a.rowFor(.video)];
    const bv = b.values[b.rowFor(.video)];
    const aa = a.values[a.rowFor(.audio)];
    const ba = b.values[b.rowFor(.audio)];
    try std.testing.expectApproxEqAbs(av, bv, 1e-6); // video is 1 - sigma either way
    errdefer std.debug.print("audio label {d} vs {d}\n", .{ aa, ba });
    try std.testing.expect(@abs(aa - ba) > 0.1);
    // ...and with equal shifts the audio label collapses onto the video one.
    const same = Timesteps.init(0.5, visual_cond_timestep, audio_cond_timestep, .{ .video = 4.0, .audio = 4.0 });
    try std.testing.expectApproxEqAbs(
        same.values[same.rowFor(.video)],
        same.values[same.rowFor(.audio)],
        1e-6,
    );
}

// --- forward parity against the reference ---------------------------------

const forward_fixture = @embedFile("assets/minimax_h3_forward.safetensors");

/// Relative L2 of `got` against `want`. Reported rather than a per-element
/// tolerance because what matters is whether the whole field matches, and a
/// single-element bound hides a small systematic bias.
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

test "a uniform denoise mask gives the same forward either way it is expressed" {
    // A mask whose rows all agree COLLAPSES to a segment-level label; forcing the
    // per-row table instead selects the same labels for the same rows, so the two
    // must agree to the last bit. This is the isolation for the per-row modulation
    // path -- it removes the mask's effect and leaves only the plumbing, and it is
    // what caught the real defect (the device figure merely tracked the host one).
    const gpa = std.testing.allocator;
    const io = std.testing.io;

    var st = try tp_core.safetensors.SafeTensors.initFromSlice(gpa, forward_fixture);
    defer st.deinit();
    var dit = try DiT.load(gpa, .{ .safetensors = &st });
    defer dit.deinit();

    const text_len: usize = 4;
    const t: usize = 2;
    const lh: usize = 4;
    const lw: usize = 6;
    const at: usize = 3;

    var layout = try PackedLayout.build(gpa, .{
        .text_len = text_len,
        .latent_t = t,
        .latent_h = lh,
        .latent_w = lw,
        .audio_t = at,
    }, &.{}, &.{}, &.{});
    defer layout.deinit();
    var ws = try Workspace.init(gpa, dit.cfg, &layout);
    defer ws.deinit(gpa);

    const text = try (try st.require("in.refined")).toF32Alloc(gpa);
    defer gpa.free(text);
    const video = try (try st.require("in.video")).toF32Alloc(gpa);
    defer gpa.free(video);
    const audio = try (try st.require("in.audio")).toF32Alloc(gpa);
    defer gpa.free(audio);

    const frame_rows = (lh / patch_h) * (lw / patch_w);
    const vmask = try gpa.alloc(f32, t * frame_rows);
    defer gpa.free(vmask);
    @memset(vmask, 0.5);
    const amask = try gpa.alloc(f32, at * audio_channels);
    defer gpa.free(amask);
    @memset(amask, 0.5);

    const in: Inputs = .{
        .video = video,
        .audio = audio,
        .text = text,
        .sigma = 0.7,
        // Not shift parity, so the audio stream's own sigma is in play.
        .shifts = .{ .video = 12.0, .audio = 3.0 },
        .video_mask = vmask,
        .audio_mask = amask,
    };

    const out = try gpa.alloc(f32, 4 * (video.len + audio.len));
    defer gpa.free(out);
    const cs_v = out[0..video.len];
    const cs_a = out[video.len..][0..audio.len];
    const cr_v = out[video.len + audio.len ..][0..video.len];
    const cr_a = out[2 * video.len + audio.len ..][0..audio.len];

    force_row_labels = false;
    try forward(&dit, io, gpa, &ws, &layout, cs_v, cs_a, in);
    force_row_labels = true;
    defer force_row_labels = false;
    try forward(&dit, io, gpa, &ws, &layout, cr_v, cr_a, in);

    const ev = relL2(cs_v, cr_v);
    const ea = relL2(cs_a, cr_a);
    errdefer std.debug.print("collapsed vs per-row: video {e}, audio {e}\n", .{ ev, ea });
    try std.testing.expect(ev < 1e-6);
    try std.testing.expect(ea < 1e-6);

    // ...and the mask is not inert, or the whole thing would pass on a no-op.
    force_row_labels = false;
    var unmasked: Inputs = in;
    unmasked.video_mask = &.{};
    unmasked.audio_mask = &.{};
    const um_v = out[2 * video.len + 2 * audio.len ..][0..video.len];
    const um_a = out[3 * video.len + 2 * audio.len ..][0..audio.len];
    try forward(&dit, io, gpa, &ws, &layout, um_v, um_a, unmasked);
    try std.testing.expect(relL2(um_v, cs_v) > 1e-3);
    try std.testing.expect(relL2(um_a, cs_a) > 1e-3);
}

test "the forward matches the reference at a toy width" {
    // The real checkpoint is 21 GB of int8 and cannot be a unit fixture. What is
    // pinned here is the ARCHITECTURE: ComfyUI's own `MiniMaxH3Model` built at a
    // toy width with seeded f32 weights (tools/gen_minimax_h3_forward.py). Every
    // convention that makes H3 what it is runs at this width exactly as at the
    // real one -- the three-tag adaLN row layout, partial split-half rope over
    // the area-normalized grid, channel-major audio, the curve-interpolated time
    // embedding, the swiglu half order, and the negated output.
    //
    // The toy width is deliberately not square: inner (2 x 32 = 64) differs from
    // hidden (32) as it does in the real model (7168 vs 5376), so a square
    // `out_proj` cannot pass by accident.
    const gpa = std.testing.allocator;
    const io = std.testing.io;

    var st = try tp_core.safetensors.SafeTensors.initFromSlice(gpa, forward_fixture);
    defer st.deinit();
    const store: WeightStore = .{ .safetensors = &st };

    var dit = try DiT.load(gpa, store);
    defer dit.deinit();
    const cfg = dit.cfg;
    // The fixture's own config, read back from the weights rather than restated.
    try std.testing.expectEqual(@as(usize, 32), cfg.hidden);
    try std.testing.expectEqual(@as(usize, 3), cfg.n_layers);
    try std.testing.expectEqual(@as(usize, 2), cfg.refiner_layers);
    try std.testing.expectEqual(@as(usize, 2), cfg.n_heads);
    try std.testing.expectEqual(@as(usize, 32), cfg.head_dim);
    try std.testing.expectEqual(@as(usize, 20), cfg.text_dim);
    // partial rope: 12 pairs rotate 24 of the 32-wide head, 8 pass through
    try std.testing.expectEqual(@as(usize, 24), cfg.ropeRotDim());
    try std.testing.expect(cfg.ropeRotDim() < cfg.head_dim);

    const text_len: usize = 4;
    const t: usize = 2;
    const lh: usize = 4;
    const lw: usize = 6;
    const at: usize = 3;
    const sigma: f32 = 0.7;

    const raw_text = try st.require("in.text");
    const want_refined = try st.require("in.refined");
    const in_video = try st.require("in.video");
    const in_audio = try st.require("in.audio");
    const want_video = try st.require("out.video");
    const want_audio = try st.require("out.audio");

    const text_f32 = try raw_text.toF32Alloc(gpa);
    defer gpa.free(text_f32);
    const refined_ref = try want_refined.toF32Alloc(gpa);
    defer gpa.free(refined_ref);
    const video_f32 = try in_video.toF32Alloc(gpa);
    defer gpa.free(video_f32);
    const audio_f32 = try in_audio.toF32Alloc(gpa);
    defer gpa.free(audio_f32);
    const want_v = try want_video.toF32Alloc(gpa);
    defer gpa.free(want_v);
    const want_a = try want_audio.toF32Alloc(gpa);
    defer gpa.free(want_a);

    // --- the text half, which runs once per sampling run ------------------
    const refined = try gpa.alloc(f32, text_len * cfg.hidden);
    defer gpa.free(refined);
    try refineText(&dit, io, gpa, refined, text_f32, text_len);
    {
        const err = relL2(refined_ref, refined);
        errdefer std.debug.print("refined text rel L2 {e}\n", .{err});
        try std.testing.expect(err < 1e-5);
    }

    // --- the packed forward ----------------------------------------------
    var layout = try PackedLayout.build(gpa, .{
        .text_len = text_len,
        .latent_t = t,
        .latent_h = lh,
        .latent_w = lw,
        .audio_t = at,
    }, &.{}, &.{}, &.{});
    defer layout.deinit();

    var ws = try Workspace.init(gpa, cfg, &layout);
    defer ws.deinit(gpa);

    const got_v = try gpa.alloc(f32, latent_channels * t * lh * lw);
    defer gpa.free(got_v);
    const got_a = try gpa.alloc(f32, audio_latent_channels * audio_channels * at);
    defer gpa.free(got_a);

    try forward(&dit, io, gpa, &ws, &layout, got_v, got_a, .{
        .video = video_f32,
        .audio = audio_f32,
        // the REFINED text, which is what the network consumes
        .text = refined,
        .sigma = sigma,
    });

    const err_v = relL2(want_v, got_v);
    const err_a = relL2(want_a, got_a);
    errdefer std.debug.print("video rel L2 {e}, audio rel L2 {e}\n", .{ err_v, err_a });
    try std.testing.expect(err_v < 1e-5);
    try std.testing.expect(err_a < 1e-5);

    // Both streams are NEGATED by the forward, and a sign error is invisible in
    // every norm and in the relative L2 above if it were applied to both sides.
    // The fixture's output has both signs present (its generator asserts that),
    // so compare a signed element directly.
    try std.testing.expect(want_v[0] * got_v[0] > 0);
    try std.testing.expect(want_a[0] * got_a[0] > 0);
}

// --- config detection against the real checkpoint --------------------------

const test_gate = @import("../test_gate.zig");
const SafeTensors = tp_core.safetensors.SafeTensors;

const h3_dit = "/home/qt/genai/comfyui/models/diffusion_models/h3/10erosMaxInt8Ref2va_v10Beta.safetensors";

test "config is read off the checkpoint's own shapes" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    try test_gate.requireModelFile(io, h3_dit);

    var st = try SafeTensors.open(gpa, io, h3_dit);
    defer st.deinit();
    const cfg = try Config.detect(.{ .safetensors = &st });

    try std.testing.expectEqual(@as(usize, 5376), cfg.hidden);
    try std.testing.expectEqual(@as(usize, 50), cfg.n_layers);
    try std.testing.expectEqual(@as(usize, 2), cfg.refiner_layers);
    try std.testing.expectEqual(@as(usize, 56), cfg.n_heads);
    try std.testing.expectEqual(@as(usize, 128), cfg.head_dim);
    try std.testing.expectEqual(@as(usize, 14336), cfg.ffn);
    try std.testing.expectEqual(@as(usize, 5120), cfg.text_dim);
    try std.testing.expectEqual(@as(usize, 16), cfg.rope_inv_freq_len);

    // this checkpoint is curve-form: no time embedder, and the adaLN input is
    // 8 wide because it is a coordinate on a time-embedding curve
    try std.testing.expect(cfg.usesAdalnCurve());
    try std.testing.expectEqual(@as(usize, 1025), cfg.adaln_curve_grid.?);
    try std.testing.expectEqual(@as(usize, 8), cfg.time_embed_dim);

    // 48 pairs over three axes rotate 96 of the 128-wide head, so 32 dims pass
    // through untouched. The packed angle buffer is 96 wide with both halves
    // identical, which is why the two agree.
    try std.testing.expectEqual(@as(usize, 48), cfg.ropePairs());
    try std.testing.expectEqual(@as(usize, 96), cfg.ropeRotDim());
    try std.testing.expectEqual(cfg.ropeRotDim(), cfg.ropeAngleWidth());
    try std.testing.expectEqual(@as(usize, 32), cfg.head_dim - cfg.ropeRotDim());
    try std.testing.expectEqual(@as(usize, 96), cfg.videoPatchDim());
}

// --- layout parity against the reference -----------------------------------

const layout_fixtures = @embedFile("assets/minimax_h3_layout.json");

const FixKeyframe = struct { frame_index: usize, latent_t: usize, audio_t: usize };
const FixRef = struct {
    kind: []const u8,
    latent_t: usize,
    latent_h: usize,
    latent_w: usize,
    audio_t: usize,
};
const FixInput = struct {
    text_len: usize,
    latent_t: usize,
    latent_h: usize,
    latent_w: usize,
    audio_t: usize,
    keyframes: []const FixKeyframe,
    refs: []const FixRef,
};
const FixSegment = struct { start: usize, stop: usize, kind: []const u8 };
const FixCase = struct {
    name: []const u8,
    input: FixInput,
    seq_len: usize,
    segments: []const FixSegment,
    video_cond_rows: usize,
    audio_cond_rows: usize,
    pos: []const f64,
};
const FixBigCase = struct {
    name: []const u8,
    input: FixInput,
    seq_len: usize,
    segments: []const FixSegment,
    video_cond_rows: usize,
    audio_cond_rows: usize,
    pos_checksum: f64,
};
const Fixtures = struct {
    note: []const u8,
    cases: []const FixCase,
    big_cases: []const FixBigCase,
};

/// Order-sensitive, matching the generator's. A plain sum would not notice two
/// rows swapping, which is exactly what a permuted grid produces.
fn posChecksum(pos: []const Pos) f64 {
    var acc: f64 = 0;
    var i: usize = 0;
    for (pos) |p| {
        for (p) |v| {
            i += 1;
            acc += @as(f64, @floatFromInt(i)) * v;
        }
    }
    return acc;
}

fn buildFromFixture(gpa: std.mem.Allocator, in: FixInput) !PackedLayout {
    var keyframes: std.ArrayList(Keyframe) = .empty;
    defer keyframes.deinit(gpa);
    for (in.keyframes) |k| try keyframes.append(gpa, .{
        .frame_index = k.frame_index,
        .latent_t = k.latent_t,
        .audio_t = k.audio_t,
    });

    var refs: std.ArrayList(Ref) = .empty;
    defer refs.deinit(gpa);
    for (in.refs) |r| try refs.append(gpa, .{
        .kind = std.meta.stringToEnum(Ref.RefKind, r.kind) orelse return error.UnknownRefKind,
        .latent_t = r.latent_t,
        .latent_h = r.latent_h,
        .latent_w = r.latent_w,
        .audio_t = r.audio_t,
    });

    return PackedLayout.build(gpa, .{
        .text_len = in.text_len,
        .latent_t = in.latent_t,
        .latent_h = in.latent_h,
        .latent_w = in.latent_w,
        .audio_t = in.audio_t,
    }, keyframes.items, refs.items, &.{});
}

fn expectSegments(expected: []const FixSegment, got: []const Segment, name: []const u8) !void {
    errdefer std.debug.print("case '{s}': {d} segments, expected {d}\n", .{ name, got.len, expected.len });
    try std.testing.expectEqual(expected.len, got.len);
    for (expected, got) |e, g| {
        const want = std.meta.stringToEnum(Kind, e.kind) orelse return error.UnknownKind;
        errdefer std.debug.print(
            "case '{s}': segment {s} [{d},{d}) got {s} [{d},{d})\n",
            .{ name, e.kind, e.start, e.stop, @tagName(g.kind), g.start, g.stop },
        );
        try std.testing.expectEqual(want, g.kind);
        try std.testing.expectEqual(e.start, g.start);
        try std.testing.expectEqual(e.stop, g.stop);
    }
}

test "packed layout matches the reference position grid" {
    const gpa = std.testing.allocator;
    const parsed = try std.json.parseFromSlice(Fixtures, gpa, layout_fixtures, .{});
    defer parsed.deinit();

    for (parsed.value.cases) |c| {
        var l = try buildFromFixture(gpa, c.input);
        defer l.deinit();

        errdefer std.debug.print("case '{s}'\n", .{c.name});
        try std.testing.expectEqual(c.seq_len, l.seq_len);
        try expectSegments(c.segments, l.segments, c.name);
        try std.testing.expectEqual(c.video_cond_rows, l.video_cond_rows);
        try std.testing.expectEqual(c.audio_cond_rows, l.audio_cond_rows);

        // f64 throughout on both sides, so the grid must agree exactly rather
        // than approximately: these values feed a cos/sin and are up to ~32.
        try std.testing.expectEqual(c.pos.len, l.pos.len * 3);
        for (l.pos, 0..) |p, row| {
            for (p, 0..) |v, axis| {
                const want = c.pos[row * 3 + axis];
                errdefer std.debug.print(
                    "case '{s}': row {d} axis {d}: got {d} want {d}\n",
                    .{ c.name, row, axis, v, want },
                );
                try std.testing.expectEqual(want, v);
            }
        }
    }
    // the corpus is only worth what it covers
    try std.testing.expectEqual(@as(usize, 15), parsed.value.cases.len);
}

test "packed layout matches the reference at render scale" {
    const gpa = std.testing.allocator;
    const parsed = try std.json.parseFromSlice(Fixtures, gpa, layout_fixtures, .{});
    defer parsed.deinit();

    for (parsed.value.big_cases) |c| {
        var l = try buildFromFixture(gpa, c.input);
        defer l.deinit();

        errdefer std.debug.print("big case '{s}'\n", .{c.name});
        try std.testing.expectEqual(c.seq_len, l.seq_len);
        try expectSegments(c.segments, l.segments, c.name);
        try std.testing.expectEqual(c.video_cond_rows, l.video_cond_rows);
        try std.testing.expectEqual(c.audio_cond_rows, l.audio_cond_rows);

        const got = posChecksum(l.pos);
        errdefer std.debug.print("big case '{s}': checksum {d} want {d}\n", .{ c.name, got, c.pos_checksum });
        try std.testing.expectApproxEqRel(c.pos_checksum, got, 1e-12);
    }

    // the default render: ~38k packed rows is the shape every cost estimate in
    // VIDEO_PLAN.md is quoted against, so pin it here rather than in prose
    const big = parsed.value.big_cases[0];
    try std.testing.expectEqual(@as(usize, 64 + 207 * 2 + 37 * 24 * 42), big.seq_len);
}

test "video patchify round-trips, and is not a plain reshape" {
    const gpa = std.testing.allocator;
    const t = 2;
    const h = 4;
    const w = 6;
    const n = latent_channels * t * h * w;
    const z = try gpa.alloc(f32, n);
    defer gpa.free(z);
    for (z, 0..) |*v, i| v.* = @floatFromInt(i);

    const rows = try gpa.alloc(f32, n);
    defer gpa.free(rows);
    patchifyVideo(rows, z, t, h, w);

    const back = try gpa.alloc(f32, n);
    defer gpa.free(back);
    unpatchifyVideo(back, rows, t, h, w);
    try std.testing.expectEqualSlices(f32, z, back);

    // A permutation, not the identity: `patchify` that happened to be a memcpy
    // would round-trip too, and every norm would match while the picture differed.
    try std.testing.expect(!std.mem.eql(u8, std.mem.sliceAsBytes(z), std.mem.sliceAsBytes(rows)));

    // Spot-check the mapping itself. Row (0,0,0) column j is channel j/4 at the
    // 2x2 sub-position ((j%4)/2, j%2) of latent frame 0.
    const cols = latent_channels * patch_h * patch_w;
    for (0..latent_channels) |c| {
        for (0..patch_h) |pp| {
            for (0..patch_w) |q| {
                const want = z[((c * t + 0) * h + pp) * w + q];
                try std.testing.expectEqual(want, rows[(c * patch_h + pp) * patch_w + q]);
            }
        }
    }
    _ = cols;
}

test "audio packs channel-major, and round-trips" {
    const gpa = std.testing.allocator;
    const at = 5;
    const n = audio_latent_channels * audio_channels * at;
    const z = try gpa.alloc(f32, n);
    defer gpa.free(z);
    for (z, 0..) |*v, i| v.* = @floatFromInt(i);

    const rows = try gpa.alloc(f32, n);
    defer gpa.free(rows);
    packAudio(rows, z, at);
    const back = try gpa.alloc(f32, n);
    defer gpa.free(back);
    unpackAudio(back, rows, at);
    try std.testing.expectEqualSlices(f32, z, back);

    // Channel-major: rows 0..at are stereo channel 0's whole time axis, and
    // rows at..2*at are channel 1's. Interleaved is the same byte count and a
    // different model, so check an actual element.
    for (0..at) |ti| {
        for (0..audio_latent_channels) |c| {
            // channel 0
            try std.testing.expectEqual(
                z[(c * audio_channels + 0) * at + ti],
                rows[(0 * at + ti) * audio_latent_channels + c],
            );
            // channel 1
            try std.testing.expectEqual(
                z[(c * audio_channels + 1) * at + ti],
                rows[(1 * at + ti) * audio_latent_channels + c],
            );
        }
    }
}

test "adaLN modulation offset interleaves timestep outside tag" {
    const hidden = 8;
    // Slot s of tag `tag` at timestep row m sits at (m*3 + tag)*6*hidden + s*hidden.
    try std.testing.expectEqual(@as(usize, 0), modRowOffset(0, .video, 0, hidden));
    try std.testing.expectEqual(@as(usize, hidden), modRowOffset(0, .video, 1, hidden));
    // next TAG is one 6-slot group along...
    try std.testing.expectEqual(@as(usize, 6 * hidden), modRowOffset(0, .text, 0, hidden));
    try std.testing.expectEqual(@as(usize, 12 * hidden), modRowOffset(0, .audio, 0, hidden));
    // ...and the next TIMESTEP is three of those.
    try std.testing.expectEqual(@as(usize, 18 * hidden), modRowOffset(1, .video, 0, hidden));

    // The transposition that would be silent: tag-outside-timestep, i.e.
    // (tag * n_labels + m) instead of (m * 3 + tag). Assert the two layouts
    // genuinely disagree, so this test has something to catch.
    // The transposition that would be silent is tag-outside-timestep,
    // `tag * n_labels + m` instead of `m * 3 + tag`. Those two agree at
    // individual points by coincidence (m=1,tag=2,L=2 and m=3,tag=2,L=4 both
    // land on the same group), so sweep a grid and require that they disagree
    // SOMEWHERE rather than trusting one hand-picked index.
    const n_labels = 5;
    var disagreements: usize = 0;
    for (0..n_labels) |m| {
        inline for (@typeInfo(Tag).@"enum".fields) |f| {
            const tag: Tag = @enumFromInt(f.value);
            // the real layout, spelled out independently of the function
            const real = (m * modality_count + f.value) * 6 * hidden;
            try std.testing.expectEqual(real, modRowOffset(m, tag, 0, hidden));
            const swapped = (@as(usize, f.value) * n_labels + m) * 6 * hidden;
            if (swapped != real) disagreements += 1;
        }
    }
    try std.testing.expect(disagreements > 0);
}

test "timesteps: the two streams differ, conditions pin high, rows are sorted" {
    // Mid-schedule: the two shifts put the streams at different labels, which is
    // the whole reason the modulation table has more than one row.
    const ts = Timesteps.init(0.5, visual_cond_timestep, audio_cond_timestep, .{});
    const t_v = ts.labels()[ts.rowFor(.video)];
    const t_a = ts.labels()[ts.rowFor(.audio)];
    try std.testing.expect(t_v != t_a);
    // sigma_a < sigma_v at the same base point, so t_a = 1 - sigma_a is LARGER
    try std.testing.expect(t_a > t_v);
    // text follows the video stream rather than having a label of its own
    try std.testing.expectEqual(ts.rowFor(.video), ts.rowFor(.text));

    // Condition rows pin at their stream's cond timestep, above any mid-schedule
    // label, and video's is 0.999 rather than 1.0.
    try std.testing.expectEqual(@as(f32, visual_cond_timestep), ts.labels()[ts.rowFor(.cond)]);
    try std.testing.expectEqual(@as(f32, audio_cond_timestep), ts.labels()[ts.rowFor(.cond_audio)]);
    try std.testing.expectEqual(ts.rowFor(.cond), ts.rowFor(.ref_img));
    try std.testing.expectEqual(ts.rowFor(.cond_audio), ts.rowFor(.ref_audio));

    // Labels are ascending and distinct: the row index IS the position in this
    // list, so the order is load-bearing.
    for (1..ts.labels().len) |i| try std.testing.expect(ts.labels()[i] > ts.labels()[i - 1]);
    try std.testing.expect(ts.labels().len <= Timesteps.max_labels);

    // At sigma 0 both stream labels are 1.0, and the condition pins are `max`
    // against them, so EVERYTHING collapses to a single row. That is the correct
    // degenerate case, not a bug: a fully denoised frame and a preserved
    // condition row are at the same place on the schedule.
    const end = Timesteps.init(0.0, visual_cond_timestep, audio_cond_timestep, .{});
    try std.testing.expectEqual(@as(usize, 1), end.labels().len);
    try std.testing.expectEqual(@as(f32, 1.0), end.labels()[0]);
    inline for (@typeInfo(Kind).@"enum".fields) |f| {
        try std.testing.expectEqual(@as(usize, 0), end.rowFor(@enumFromInt(f.value)));
    }

    // At the TOP of the schedule the visual pin is genuinely its own row: t_v is
    // 0 there, well below 0.999.
    const start = Timesteps.init(1.0, visual_cond_timestep, audio_cond_timestep, .{});
    try std.testing.expect(start.rowFor(.cond) != start.rowFor(.video));
    try std.testing.expectEqual(@as(f32, visual_cond_timestep), start.labels()[start.rowFor(.cond)]);
}

test "time embedding interpolates the curve and clamps both ends" {
    const gpa = std.testing.allocator;
    const grid = 5;
    const dim = 2;
    // a ramp per column, so the interpolated value is checkable by hand
    var table: [grid * dim]f32 = undefined;
    for (0..grid) |i| {
        table[i * dim] = @floatFromInt(i);
        table[i * dim + 1] = @as(f32, @floatFromInt(i)) * 10.0;
    }

    const ts = [_]f32{ 0.0, 0.25, 0.5, 1.0, -1.0, 2.0 };
    const out = try gpa.alloc(f32, ts.len * dim);
    defer gpa.free(out);
    timeEmbed(out, &table, grid, dim, &ts);

    // t=0 is row 0; t=1 is the LAST row, reached as the far end of the last
    // interval rather than by indexing past the table
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), out[3 * dim], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 40.0), out[3 * dim + 1], 1e-6);
    // t=0.25 over a 4-interval grid is exactly row 1
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[1 * dim], 1e-6);
    // t=0.5 is row 2
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out[2 * dim], 1e-6);
    // out-of-range clamps to the ends rather than extrapolating
    try std.testing.expectApproxEqAbs(out[0], out[4 * dim], 1e-6);
    try std.testing.expectApproxEqAbs(out[3 * dim], out[5 * dim], 1e-6);
}

test "rope pairs each axis with its own frequency, halves not duplicated" {
    const gpa = std.testing.allocator;
    const inv = [_]f32{ 1.0, 0.5 };
    const pos = [_]Pos{ .{ 0.0, 0.0, 0.0 }, .{ 1.0, 2.0, 3.0 } };
    var freqs = try ropeFreqs(gpa, &pos, &inv);
    defer freqs.deinit(gpa);

    // 3 axes x 2 frequencies = 6 pairs, so 12 rotated dims
    try std.testing.expectEqual(@as(usize, 6), freqs.half);
    try std.testing.expectEqual(@as(usize, 12), freqs.cos.len);

    // row 0 is all-zero positions: every angle is 0
    for (freqs.cos[0..6]) |c| try std.testing.expectApproxEqAbs(@as(f32, 1.0), c, 1e-6);
    for (freqs.sin[0..6]) |sn| try std.testing.expectApproxEqAbs(@as(f32, 0.0), sn, 1e-6);

    // row 1: pair (axis, i) has angle pos[axis] * inv[i]
    const want = [6]f32{ 1.0 * 1.0, 1.0 * 0.5, 2.0 * 1.0, 2.0 * 0.5, 3.0 * 1.0, 3.0 * 0.5 };
    for (want, 0..) |ang, i| {
        try std.testing.expectApproxEqAbs(@cos(ang), freqs.cos[6 + i], 1e-6);
        try std.testing.expectApproxEqAbs(@sin(ang), freqs.sin[6 + i], 1e-6);
    }
}

test "workspace memory is reported before it is allocated" {
    const gpa = std.testing.allocator;
    const cfg: Config = .{
        .hidden = 5376,
        .n_layers = 50,
        .refiner_layers = 2,
        .n_heads = 56,
        .head_dim = 128,
        .ffn = 14336,
        .text_dim = 5120,
        .time_embed_dim = 8,
        .adaln_curve_grid = 1025,
        .rope_inv_freq_len = 16,
    };

    // The development shape is small enough to actually run on a CPU.
    var dev = try PackedLayout.build(gpa, .{
        .text_len = 8,
        .latent_t = 2,
        .latent_h = 16,
        .latent_w = 16,
        .audio_t = 8,
    }, &.{}, &.{}, &.{});
    defer dev.deinit();
    const dev_bytes = Workspace.bytesFor(cfg, &dev);
    try std.testing.expect(dev_bytes < 64 << 20);

    // The default render is NOT, and that is the point of reporting it: a caller
    // that allocates this blind takes several GB before touching a weight.
    var full = try PackedLayout.build(gpa, .{
        .text_len = 64,
        .latent_t = 37,
        .latent_h = 48,
        .latent_w = 84,
        .audio_t = 207,
    }, &.{}, &.{}, &.{});
    defer full.deinit();
    try std.testing.expect(Workspace.bytesFor(cfg, &full) > 4 << 30);

    // And it is the figure `init` really uses, on the shape that fits.
    var ws = try Workspace.init(gpa, cfg, &dev);
    defer ws.deinit(gpa);
    const actual = (ws.h.len + ws.hn.len + ws.qkv.len + ws.blk.len + ws.ff.len +
        ws.t_emb.len + ws.mod.len + ws.video_rows.len + ws.audio_rows.len) * @sizeOf(f32);
    try std.testing.expectEqual(dev_bytes, actual);
}

test "modality tags are the adaLN row indices, not names" {
    // a modulation row is t_row * 3 + tag, so the numeric values are load bearing
    try std.testing.expectEqual(@as(u2, 0), @intFromEnum(Kind.video.tag()));
    try std.testing.expectEqual(@as(u2, 1), @intFromEnum(Kind.text.tag()));
    try std.testing.expectEqual(@as(u2, 2), @intFromEnum(Kind.audio.tag()));
    // conditions carry their stream's tag, not a tag of their own
    try std.testing.expectEqual(Tag.video, Kind.cond.tag());
    try std.testing.expectEqual(Tag.video, Kind.ref_img.tag());
    try std.testing.expectEqual(Tag.audio, Kind.cond_audio.tag());
    try std.testing.expectEqual(Tag.audio, Kind.ref_audio.tag());
}
