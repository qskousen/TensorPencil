//! Qwen3-VL-4B language model, text-only path, the Krea 2 text encoder.
//!
//! Reference: comfy/text_encoders/llama.py (`Llama2_`, Qwen3VL_4BConfig) and
//! comfy/text_encoders/krea2.py. Config: 36 layers, hidden 2560, GQA 32/8
//! heads (head_dim 128), SwiGLU 9728, RMSNorm eps 1e-6 (plain weight),
//! per-head QK-norm before RoPE, rotate-half RoPE theta 5e6, causal
//! attention. Text-only inputs never trigger the interleaved-mRoPE branch.
//!
//! Krea 2 conditions on the hidden states *entering* layers [2,5...,35]
//! (hidden_states[k] = output of layer k-1), so layer 35 and the final norm
//! are never evaluated and are not loaded.
//!
//! `CausalLM` is the same stack used as a language model (tp-llm): all 36
//! layers plus the final norm, with the LM head tied to the bf16 embedding
//! matrix (the checkpoint ships no lm_head.weight, per Qwen3-4B tying).
//!
//! Weights stay in checkpoint dtype (fp8-e4m3 + per-tensor f32 scales) and are
//! dequantized inside the GEMM; the safetensors mapping must outlive this.

const std = @import("std");
const safetensors = @import("tp_core").safetensors;
const test_gate = @import("../test_gate.zig");
const gguf_mod = @import("tp_core").gguf;
const weights_mod = @import("tp_core").weights;
const dtypes = @import("tp_core").dtype;
const ops = @import("tp_ops");
const transformer = @import("transformer.zig");
const kv_cache_mod = @import("tp_core").kv_cache;

const SafeTensors = safetensors.SafeTensors;
const WeightStore = weights_mod.WeightStore;
const Weight = ops.matmul.Weight;
const KvCache = kv_cache_mod.KvCache;

pub const hidden = 2560;
pub const n_heads = 32;
pub const n_kv_heads = 8;
pub const head_dim = 128;
pub const intermediate = 9728;
pub const n_layers = 36;
pub const vocab_size = 151936;
pub const rms_eps: f32 = 1e-6;
pub const rope_theta: f64 = 5000000.0;

/// Per-layer dims for the shared transformer body. qwen3 has uniform geometry
/// and a compile-time `head_dim`, so these are constant across layers. The
/// encoder uses module-const dims; `CausalLM` derives them from its runtime
/// Config via `dimsFor` (n_layers/hidden/heads vary across Qwen3 checkpoints).
const encoderDims: transformer.Dims = .{
    .hidden = hidden,
    .n_heads = n_heads,
    .n_kv = n_kv_heads,
    .head_dim = head_dim,
    .q_dim = n_heads * head_dim,
    .kv_dim = n_kv_heads * head_dim,
    .intermediate = intermediate,
};
pub fn dimsFor(cfg: Config) transformer.Dims {
    return .{
        .hidden = cfg.hidden,
        .n_heads = cfg.n_heads,
        .n_kv = cfg.n_kv_heads,
        .head_dim = head_dim,
        .q_dim = cfg.qDim(),
        .kv_dim = cfg.kvDim(),
        .intermediate = cfg.intermediate,
    };
}

/// Runtime model configuration for `CausalLM`, the extension point for
/// other Qwen3-family checkpoints (Qwen3-0.6B serves as the speculative-decoding
/// draft model) AND the plain llama/Mistral family
/// (which is structurally identical once q/k are un-permuted at load, see
/// `qk_norm`/`permute_qk`). All checkpoints share head_dim 128 (the
/// flash-decoding kernels require it) and rms_eps 1e-6; everything else
/// (incl. vocab size and whether QK-norm is present) varies.
pub const Config = struct {
    n_layers: usize,
    hidden: usize,
    n_heads: usize,
    n_kv_heads: usize,
    intermediate: usize,
    rope_theta: f64,
    /// Tensor-name prefix up to "layers.N." / "embed_tokens." / "norm.".
    prefix: []const u8,
    /// LM-head / embedding row count (default: the embedded-Qwen3 vocab).
    /// llama/Mistral checkpoints carry their own (e.g. Mistral-Nemo 131074).
    vocab: usize = vocab_size,
    /// Per-head QK RMS-norm before RoPE (qwen3). false for llama/Mistral.
    qk_norm: bool = true,
    /// Un-permute q/k weight rows at load so the stored ggml "NORM"
    /// (interleaved) RoPE layout matches our rotate-half RoPE. Only the
    /// llama/Mistral family (LLAMA_ROPE_TYPE_NORM) needs it; qwen3 is NEOX
    /// (rotate-half) and stores q/k unpermuted.
    permute_qk: bool = false,
    /// RMSNorm epsilon (qwen3 = 1e-6; Mistral-Nemo = 1e-5). Read from
    /// `<arch>.attention.layer_norm_rms_epsilon` for GGUF checkpoints.
    rms_eps: f32 = rms_eps,

    pub fn qDim(self: Config) usize {
        return self.n_heads * head_dim;
    }
    pub fn kvDim(self: Config) usize {
        return self.n_kv_heads * head_dim;
    }

    /// The Krea 2 text-encoder checkpoint: Qwen3-VL-4B.
    pub const vl_4b: Config = .{
        .n_layers = 36,
        .hidden = 2560,
        .n_heads = 32,
        .n_kv_heads = 8,
        .intermediate = 9728,
        .rope_theta = 5000000.0,
        .prefix = "model.language_model.",
    };

    /// Qwen3-0.6B (base or instruct), the draft model.
    pub const qwen3_0_6b: Config = .{
        .n_layers = 28,
        .hidden = 1024,
        .n_heads = 16,
        .n_kv_heads = 8,
        .intermediate = 3072,
        .rope_theta = 1000000.0,
        .prefix = "model.",
    };

    /// Vanilla Qwen3-4B (text-only): the VL's dims but rope_theta 1e6 and
    /// bare tensor names, the EAGLE-3 head's actual training target.
    pub const qwen3_4b: Config = .{
        .n_layers = 36,
        .hidden = 2560,
        .n_heads = 32,
        .n_kv_heads = 8,
        .intermediate = 9728,
        .rope_theta = 1000000.0,
        .prefix = "model.",
    };

    /// Pick the configuration for a checkpoint. GGUFs with llama.cpp
    /// metadata build it from the `qwen3.*` keys; everything else matches a
    /// preset by its embedding tensor name and shape (rope_theta is not
    /// recoverable from weights, so only known configurations load).
    pub fn detect(store: WeightStore) !Config {
        if (store == .gguf) return detectGguf(store.gguf);
        inline for (.{ vl_4b, qwen3_0_6b, qwen3_4b }) |cfg| {
            var buf: [96]u8 = undefined;
            const name = try std.fmt.bufPrint(&buf, "{s}embed_tokens.weight", .{cfg.prefix});
            if (store.get(name)) |view| {
                const shape = view.info.shape.slice();
                if (shape.len == 2 and shape[0] == vocab_size and shape[1] == cfg.hidden) return cfg;
            }
        }
        return error.UnknownModelConfig;
    }

    fn detectGguf(g: *const gguf_mod.Gguf) !Config {
        const arch = g.getStr("general.architecture") orelse "qwen3";
        // The plain llama/Mistral family (Mistral-Nemo etc.) reuses this whole
        // stack: it is qwen3 minus per-head QK-norm, plus a checkpoint-carried
        // vocab and NORM-rope q/k (un-permuted at load). All its hyperparameters
        // live under the `llama.*` metadata prefix.
        const is_llama = std.mem.eql(u8, arch, "llama");
        if (!is_llama and !std.mem.eql(u8, arch, "qwen3")) return error.UnknownModelConfig;
        const p = if (is_llama) "llama" else "qwen3";

        // Full llama.cpp metadata: build the config from the <arch>.* keys.
        var kbuf: [64]u8 = undefined;
        const key = struct {
            fn f(buf: []u8, pfx: []const u8, suffix: []const u8) []const u8 {
                return std.fmt.bufPrint(buf, "{s}.{s}", .{ pfx, suffix }) catch unreachable;
            }
        }.f;
        if (g.getUint(key(&kbuf, p, "block_count"))) |block_count| {
            if (block_count == 0 or block_count > max_layers) return error.UnknownModelConfig;
            // head_dim is fixed by the kernels, not configurable.
            if ((g.getUint(key(&kbuf, p, "attention.key_length")) orelse head_dim) != head_dim or
                (g.getUint(key(&kbuf, p, "attention.value_length")) orelse head_dim) != head_dim)
                return error.UnknownModelConfig;
            return .{
                .n_layers = @intCast(block_count),
                .hidden = @intCast(g.getUint(key(&kbuf, p, "embedding_length")) orelse return error.UnknownModelConfig),
                .n_heads = @intCast(g.getUint(key(&kbuf, p, "attention.head_count")) orelse return error.UnknownModelConfig),
                .n_kv_heads = @intCast(g.getUint(key(&kbuf, p, "attention.head_count_kv")) orelse return error.UnknownModelConfig),
                .intermediate = @intCast(g.getUint(key(&kbuf, p, "feed_forward_length")) orelse return error.UnknownModelConfig),
                .rope_theta = g.getFloat(key(&kbuf, p, "rope.freq_base")) orelse 1e6,
                .prefix = "",
                .vocab = @intCast(g.getUint(key(&kbuf, p, "vocab_size")) orelse vocab_size),
                .qk_norm = !is_llama,
                .permute_qk = is_llama,
                .rms_eps = if (g.getFloat(key(&kbuf, p, "attention.layer_norm_rms_epsilon"))) |e| @floatCast(e) else rms_eps,
            };
        }
        if (is_llama) return error.UnknownModelConfig; // llama needs full metadata
        // Hyperparameter-less GGUF (ComfyUI-style conversion, bare HF names,
        // at most an architecture tag): match a plain-Qwen3 preset by
        // embedding shape. rope_theta is unrecoverable, so a VL-derived
        // conversion would silently get the plain-Qwen3 theta, hence the
        // warning.
        const view = g.get("embed_tokens.weight") orelse return error.UnknownModelConfig;
        const shape = view.info.shape.slice();
        if (shape.len != 2 or shape[0] != vocab_size) return error.UnknownModelConfig;
        inline for (.{ qwen3_0_6b, qwen3_4b }) |preset| {
            if (shape[1] == preset.hidden) {
                var cfg = preset;
                cfg.prefix = "";
                std.log.warn(
                    "gguf has no hyperparameter metadata; assuming plain Qwen3 (rope_theta {d})",
                    .{cfg.rope_theta},
                );
                return cfg;
            }
        }
        return error.UnknownModelConfig;
    }

    /// Upper bound on n_layers, backend steppers use fixed-size per-layer
    /// arrays. Covers the presets (36) and GGUF checkpoints up to Qwen3-32B
    /// (64 layers).
    pub const max_layers = 64;
};

/// Whether a GGUF actually declares its RoPE base, as opposed to `detectGguf`
/// having guessed one. Checked under both metadata prefixes this stack accepts.
fn statesRopeTheta(g: *const gguf_mod.Gguf) bool {
    return g.getFloat("qwen3.rope.freq_base") != null or g.getFloat("llama.rope.freq_base") != null;
}

/// Tap k is the hidden state before layer k runs.
pub const tap_layers = [_]usize{ 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35 };
pub const tap_count = tap_layers.len;

/// Z-Image conditions on the penultimate hidden state with the final norm
/// skipped (`sd1_clip` `layer_idx = -2`, `layer_norm_hidden_state = False`), which
/// comfy resolves to `intermediate_output = 36 - 2 = 34` and captures *after* layer
/// 34 has run. In this file's convention, tap k is the state entering layer k,
/// that is tap 35, i.e. exactly krea2's last tap. Both variants therefore run
/// the identical 35 layers and differ only in how many states they keep.
pub const zimage_taps = [_]usize{35};

/// Anima conditions on the final hidden state with `model.norm` APPLIED
/// (`sd1_clip` `layer = "last"`, and `Qwen3_06BConfig.final_norm = True`). In this
/// file's convention, tap k is the state entering layer k, that is tap
/// `n_layers`, one past the last layer, so unlike the other two variants every
/// layer runs AND the final norm is evaluated. Derived from the preset rather than
/// written as `28`, so the two cannot disagree.
pub const anima_taps = [_]usize{Config.qwen3_0_6b.n_layers};

/// Which conditioning stack a `TextEncoder` produces.
///
/// krea2 and Z-Image are the same 36-layer, 2560-wide Qwen3-4B body and differ only
/// in the checkpoint's tensor prefix (krea2 ships the VL checkpoint, so its language
/// model is nested under `model.language_model.`), the RoPE theta (5e6 vs 1e6) and
/// the tap list. Anima is a different body, Qwen3-0.6B, 28 layers, 1024 wide,
/// which is why the encoder is written against `cfg` rather than this file's
/// module-level dims.
///
/// The theta is the dangerous one, it is not recoverable from the weights, and
/// the wrong value produces a perfectly finite encode whose long-range positions
/// are simply wrong. All three live in `Config` presets so the choice is made once.
pub const Variant = enum {
    /// Krea 2: Qwen3-VL-4B, the 12-layer tap stack the DiT's `txtfusion` consumes.
    krea2,
    /// Z-Image: plain Qwen3-4B, one hidden state.
    zimage,
    /// Anima: Qwen3-0.6B base, the final hidden state after `model.norm`. Its
    /// states are the `llm_adapter`'s cross-attention source, not the denoiser's
    /// context directly, see `models/anima.zig`.
    anima,

    pub fn config(self: Variant) Config {
        return switch (self) {
            .krea2 => Config.vl_4b,
            .zimage => Config.qwen3_4b,
            .anima => Config.qwen3_0_6b,
        };
    }

    pub fn taps(self: Variant) []const usize {
        return switch (self) {
            .krea2 => &tap_layers,
            .zimage => &zimage_taps,
            .anima => &anima_taps,
        };
    }

    /// Whether `norm.weight` is loaded and applied to the last tap.
    ///
    /// This is NOT the same knob as `sd1_clip`'s `layer_norm_hidden_state`, which
    /// only governs an *intermediate* output. krea2 and Z-Image tap before a layer
    /// that still has to run, so the final norm is genuinely never evaluated for
    /// them and is not even loaded; Anima taps past the end, where it always is.
    pub fn appliesFinalNorm(self: Variant) bool {
        return switch (self) {
            .krea2, .zimage => false,
            .anima => true,
        };
    }
};

pub const q_dim = n_heads * head_dim; // 4096
pub const kv_dim = n_kv_heads * head_dim; // 1024

const Layer = struct {
    input_norm: []const f32,
    q: Weight,
    k: Weight,
    v: Weight,
    o: Weight,
    q_norm: []const f32, // [head_dim]
    k_norm: []const f32,
    post_norm: []const f32,
    gate: Weight,
    up: Weight,
    down: Weight,
};

pub const TextEncoder = struct {
    /// This tower cannot apply per-token prompt weights, and the reason is structural
    /// rather than a missing feature: krea2 conditions on qwen3 tap states with no fixed
    /// token window, no chunking, and no empty-prompt reference of matching shape to
    /// interpolate against. See `clip_text.TextEncoder.supports_prompt_weights` for the
    /// three properties an emphasis-capable encoder needs, and `pipeline`'s
    /// `supportsPromptWeights` for what reads this.
    pub const supports_prompt_weights = false;

    arena: std.heap.ArenaAllocator,
    /// `[vocab, hidden]` view into the mapped file. A `Weight`, not a byte
    /// slice, because the dtype is a property of the CHECKPOINT: a safetensors
    /// encoder ships bf16, a GGUF one ships whatever it was quantized to (q6_k for
    /// Qwen3-4B-Q4_K_M). Every gather goes through `embedTokens`, which dispatches
    /// on it, so no call site may assume bf16 and a `* 2` row stride.
    embed: Weight,
    /// Layers `0..taps[last]`, krea2/Z-Image tap before layer 35, so layer 35 is
    /// never loaded; Anima taps past layer 27, so all 28 are.
    layers: []Layer,
    variant: Variant,
    cfg: Config,
    /// Hidden states to keep, in layer order. `encode` emits them token-major.
    taps: []const usize,
    /// `norm.weight`, or null when this variant never reaches the final norm. See
    /// `Variant.appliesFinalNorm`.
    final_norm: ?[]const f32,

    pub fn tapCount(self: *const TextEncoder) usize {
        return self.taps.len;
    }

    /// Krea 2's encoder. Kept as the unqualified name because it is what every
    /// existing caller means; `loadVariant` is the general form.
    ///
    /// `store` may be a `weights.Prefixed` view of a bundled checkpoint, this loader
    /// always sees its own component at the root. See `weights.Prefixed`.
    pub fn load(gpa: std.mem.Allocator, store: weights_mod.WeightStore) !TextEncoder {
        return loadVariant(gpa, store, .krea2);
    }

    pub fn loadVariant(gpa: std.mem.Allocator, store: weights_mod.WeightStore, variant: Variant) !TextEncoder {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();

        // The checkpoint's own metadata wins for everything the checkpoint
        // states; the Variant supplies only what it cannot know. A GGUF carries
        // its dims, its tensor prefix (bare, no `model.`) and, the one that
        // matters most, its `rope.freq_base`, which for a safetensors file is not
        // recoverable and has to come from the Variant's constant. What stays with
        // the Variant is the TAP LIST, because which hidden states to keep is a
        // property of how the *diffusion* model was trained, not of the encoder.
        const cfg = if (store == .gguf) blk: {
            var c = try Config.detect(store);
            c.vocab = @max(c.vocab, 1);
            // A checkpoint that states nothing must not win. ComfyUI-style
            // GGUFs (city96's converter) carry only `general.architecture`, so
            // `detectGguf` falls back to matching a preset by embedding shape and
            // GUESSES plain-Qwen3's 1e6 theta. That is right for Z-Image and
            // silently wrong for krea2, whose Qwen3-VL encoder is 5e6, a value
            // that is not recoverable from weights and encodes perfectly finite
            // nonsense when wrong. Where the file is silent, the Variant's
            // constant is the better answer, since it is what the *diffusion*
            // model was trained against.
            if (!statesRopeTheta(store.gguf)) c.rope_theta = variant.config().rope_theta;
            break :blk c;
        } else variant.config();
        var nbuf: [96]u8 = undefined;
        const embed_name = try std.fmt.bufPrint(&nbuf, "{s}embed_tokens.weight", .{cfg.prefix});
        const embed_view = try store.require(embed_name);
        if (embed_view.info.elemCount() != cfg.vocab * cfg.hidden) return error.ShapeMismatch;
        const embed = Weight.init(embed_view.bytes, embed_view.info.dtype, cfg.vocab, cfg.hidden);

        // Layers up to the last tap. krea2/Z-Image tap at 35, so layer 35 and the
        // final norm are never evaluated and are not loaded; Anima taps at 28, which
        // is every layer, and then the final norm.
        const taps = variant.taps();
        if (taps[taps.len - 1] > cfg.n_layers) return error.UnsupportedModelConfig;
        const layers = try loadLayersCfg(alloc, store, cfg, taps[taps.len - 1]);

        const final_norm: ?[]const f32 = if (variant.appliesFinalNorm())
            try loadNormNamed(alloc, store, try std.fmt.bufPrint(&nbuf, "{s}norm.weight", .{cfg.prefix}), cfg.hidden)
        else
            null;

        return .{
            .arena = arena,
            .embed = embed,
            .layers = layers,
            .variant = variant,
            .cfg = cfg,
            .taps = taps,
            .final_norm = final_norm,
        };
    }

    pub fn deinit(self: *TextEncoder) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// Encode token ids to this variant's conditioning stack,
    /// `[seq][tapCount()][hidden]` row-major (token-major, matching the krea2 DiT's
    /// unpacked context layout). For `.zimage` there is one tap, so the result is
    /// just `[seq][hidden]`.
    pub fn encode(self: *const TextEncoder, io: std.Io, gpa: std.mem.Allocator, ids: []const u32, cancel: ?*std.atomic.Value(bool)) ![]f32 {
        const seq = ids.len;
        std.debug.assert(seq > 0);
        const n_taps = self.taps.len;
        // Arm fine-grained cancel inside the CPU matmul/attention kernels so a
        // stop lands mid-layer, not just between layers.
        const prev_tok = ops.cancel.token;
        ops.cancel.token = cancel;
        defer ops.cancel.token = prev_tok;

        // `self.cfg.hidden`, not this file's 2560: Anima's body is Qwen3-0.6B at
        // 1024. Reading the module constant here allocated 2.5x the rows needed and
        // strode the wrong distance through them.
        const h = self.cfg.hidden;

        const out = try gpa.alloc(f32, seq * n_taps * h);
        errdefer gpa.free(out);

        const x = try gpa.alloc(f32, seq * h);
        defer gpa.free(x);
        try embedTokens(self.embed, ids, x);

        // From the variant's config, not the module constant: the VL checkpoint's
        // theta is 5e6 and plain Qwen3-4B's is 1e6, and the wrong one encodes
        // perfectly finite nonsense.
        var freqs = try ops.rope.rotateHalfFreqs(gpa, seq, head_dim, self.cfg.rope_theta);
        defer freqs.deinit(gpa);

        var scratch = try Scratch.init(gpa, seq, self.cfg);
        defer scratch.deinit(gpa);

        const dims = dimsFor(self.cfg);
        var tap_idx: usize = 0;
        // `n_layers + 1`, because a tap index may be one PAST the last layer:
        // Anima's tap is `cfg.n_layers`. The `l >= self.layers.len` break below is
        // what actually terminates the loop, so no variant runs a layer it should
        // not, but a range of exactly `n_layers` would never fire that final tap,
        // and the `tap_idx == n_taps` assert is what would catch it.
        for (0..self.cfg.n_layers + 1) |l| {
            // Poll cancel between layers so a stop lands mid-encode (a full CPU
            // encode is 36 layers of full-sequence attention).
            if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
            if (tap_idx < n_taps and self.taps[tap_idx] == l) {
                for (0..seq) |t| {
                    const dst = out[(t * n_taps + tap_idx) * h ..][0..h];
                    const src = x[t * h ..][0..h];
                    // The final norm applies only to a tap past the last layer,
                    // which is the only tap a variant that has one ever takes.
                    if (self.final_norm) |w| {
                        if (l == self.cfg.n_layers) {
                            ops.norm.rmsNorm(dst, src, w, self.cfg.rms_eps);
                            continue;
                        }
                    }
                    @memcpy(dst, src);
                }
                tap_idx += 1;
            }
            if (l >= self.layers.len) break;
            // Encoder: full-sequence, no persistent KV cache.
            try transformer.layerForward(transformer.qwen3_spec, .fresh, io, gpa, self.layers[l], x, seq, dims, freqs, self.cfg.rms_eps, {}, 0, 0, false, &scratch);
        }
        std.debug.assert(tap_idx == n_taps);
        return out;
    }
};

/// A Qwen3-family text stack as a language model: all layers, final norm,
/// and the embedding matrix doubling as the tied LM head. The checkpoint's
/// configuration is auto-detected (Config.detect).
pub const CausalLM = struct {
    arena: std.heap.ArenaAllocator,
    cfg: Config,
    /// [vocab, hidden] embedding table view into the mapped file, in the
    /// checkpoint's storage dtype (bf16 for safetensors; GGUFs quantize it).
    embed: Weight,
    /// LM head ([vocab, hidden]): `embed` when tied (Qwen3-4B ships no
    /// lm_head), a separate tensor when the checkpoint carries one (GGUF
    /// "output.weight").
    head: Weight,
    layers: []Layer,
    final_norm: []const f32,

    pub fn load(gpa: std.mem.Allocator, store: WeightStore) !CausalLM {
        const cfg = try Config.detect(store);
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();

        var buf: [96]u8 = undefined;
        const embed_view = try store.require(try std.fmt.bufPrint(&buf, "{s}embed_tokens.weight", .{cfg.prefix}));
        const eshape = embed_view.info.shape.slice();
        if (eshape.len != 2 or eshape[0] != cfg.vocab or eshape[1] != cfg.hidden)
            return error.ShapeMismatch;
        const embed = Weight.init(embed_view.bytes, embed_view.info.dtype, cfg.vocab, cfg.hidden);

        var head = embed;
        if (store.get(try std.fmt.bufPrint(&buf, "{s}lm_head.weight", .{cfg.prefix}))) |hv| {
            const hshape = hv.info.shape.slice();
            if (hshape.len != 2 or hshape[0] != cfg.vocab or hshape[1] != cfg.hidden)
                return error.ShapeMismatch;
            head = Weight.init(hv.bytes, hv.info.dtype, cfg.vocab, cfg.hidden);
        }

        const layers = try loadLayersCfg(alloc, store, cfg, cfg.n_layers);
        const final_norm = try loadNormNamed(alloc, store, try std.fmt.bufPrint(&buf, "{s}norm.weight", .{cfg.prefix}), cfg.hidden);

        return .{ .arena = arena, .cfg = cfg, .embed = embed, .head = head, .layers = layers, .final_norm = final_norm };
    }

    pub fn deinit(self: *CausalLM) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// LM head: logits = hidden @ head^T, [vocab] per position.
    pub fn lmHead(self: *const CausalLM) Weight {
        return self.head;
    }

    /// True when any weight is a ggml block-quantized dtype, such models
    /// only run on the cpu backend until the GPU kernels land.
    pub fn hasBlockQuantWeights(self: *const CausalLM) bool {
        if (self.embed.dtype.isBlockQuant() or self.head.dtype.isBlockQuant()) return true;
        for (self.layers) |l| {
            inline for (.{ l.q, l.k, l.v, l.o, l.gate, l.up, l.down }) |w| {
                if (w.dtype.isBlockQuant()) return true;
            }
        }
        return false;
    }

    /// Forward `ids` at absolute positions [cache.len, cache.len + ids.len),
    /// appending their K/V to the cache: prefill when the cache is empty,
    /// single-token decode when ids.len == 1. `freqs` must cover the final
    /// position. When `out` ([n * hidden], n <= ids.len) is set, it receives
    /// the final-normed hidden states of the last n new positions, ready for
    /// the LM head (n = 1 for decode, n = draft+1 for speculative verify).
    pub fn forwardCached(
        self: *const CausalLM,
        io: std.Io,
        gpa: std.mem.Allocator,
        ids: []const u32,
        cache: *KvCache,
        freqs: ops.rope.Freqs,
        out: ?[]f32,
    ) !void {
        const cfg = self.cfg;
        const seq = ids.len;
        std.debug.assert(seq > 0 and seq <= cache.remaining());
        std.debug.assert(cache.n_layers == cfg.n_layers and cache.kv_dim == cfg.kvDim());

        const x = try gpa.alloc(f32, seq * cfg.hidden);
        defer gpa.free(x);
        try embedTokens(self.embed, ids, x);

        var scratch = try Scratch.init(gpa, seq, cfg);
        defer scratch.deinit(gpa);

        const dims = dimsFor(cfg);
        const pos0 = cache.len;
        for (self.layers, 0..) |layer, l| {
            if (cfg.qk_norm) {
                try transformer.layerForward(transformer.qwen3_spec, .cached, io, gpa, layer, x, seq, dims, freqs, cfg.rms_eps, cache, l, pos0, false, &scratch);
            } else {
                try transformer.layerForward(transformer.llama_spec, .cached, io, gpa, layer, x, seq, dims, freqs, cfg.rms_eps, cache, l, pos0, false, &scratch);
            }
        }
        cache.commit(seq);
        if (out) |o| {
            std.debug.assert(o.len % cfg.hidden == 0);
            const n = o.len / cfg.hidden;
            std.debug.assert(n >= 1 and n <= seq);
            ops.norm.rmsNorm(o, x[(seq - n) * cfg.hidden ..][0 .. n * cfg.hidden], self.final_norm, cfg.rms_eps);
        }
    }

    /// Tree-verify forward (speculative tree drafting):
    /// `ids.len` tree nodes forwarded against the committed cache WITHOUT
    /// appending to it. Node i (parents[i] < i; node 0 is the root) sits at
    /// absolute position cache.len + depth(i) and attends the committed
    /// prefix plus its own ancestor chain; per-layer K/V rows are retained
    /// in tree_k/tree_v ([n_layers][ids.len][kv_dim] row-major) so the
    /// caller can commit the accepted path afterwards. `out` receives the
    /// final-normed hidden states of ALL nodes ([ids.len][hidden]).
    pub fn forwardTree(
        self: *const CausalLM,
        io: std.Io,
        gpa: std.mem.Allocator,
        ids: []const u32,
        parents: []const u32,
        cache: *KvCache, // non-const: kView expands into the cache's f16 scratch
        freqs: ops.rope.Freqs,
        tree_k: []f32,
        tree_v: []f32,
        out: []f32,
    ) !void {
        const cfg = self.cfg;
        const n = ids.len;
        std.debug.assert(n > 0 and parents.len == n);
        std.debug.assert(tree_k.len >= cfg.n_layers * n * cfg.kvDim() and tree_v.len >= tree_k.len);
        std.debug.assert(out.len == n * cfg.hidden);
        std.debug.assert(cache.n_layers == cfg.n_layers and cache.kv_dim == cfg.kvDim());

        const positions = try gpa.alloc(usize, n);
        defer gpa.free(positions);
        positions[0] = cache.len;
        for (parents[1..], 1..) |p, i| {
            std.debug.assert(p < i);
            positions[i] = positions[p] + 1;
            std.debug.assert(positions[i] < cache.capacity);
        }

        const x = try gpa.alloc(f32, n * cfg.hidden);
        defer gpa.free(x);
        try embedTokens(self.embed, ids, x);

        var s = try Scratch.init(gpa, n, cfg);
        defer s.deinit(gpa);

        const dims = dimsFor(cfg);
        for (self.layers, 0..) |layer, l| {
            if (cfg.qk_norm) {
                try transformer.layerForwardTree(transformer.qwen3_spec, io, gpa, layer, x, n, dims, freqs, positions, parents, cache, l, tree_k, tree_v, cfg.rms_eps, &s);
            } else {
                try transformer.layerForwardTree(transformer.llama_spec, io, gpa, layer, x, n, dims, freqs, positions, parents, cache, l, tree_k, tree_v, cfg.rms_eps, &s);
            }
        }
        ops.norm.rmsNorm(out, x, self.final_norm, cfg.rms_eps);
    }
};

/// Look up embedding rows for `ids` into `x` [ids.len, h] f32, dequantizing
/// from the table's storage dtype (bf16, or a GGUF block-quantized format).
/// Shared with the GPU steppers' host-side prefill gathers.
pub fn embedTokens(embed: Weight, ids: []const u32, x: []f32) !void {
    const h = embed.cols;
    const row_bytes = embed.dtype.storageBytes(h);
    for (ids, 0..) |id, t| {
        if (id >= embed.rows) return error.TokenIdOutOfRange;
        const row = embed.bytes[@as(usize, id) * row_bytes ..][0..row_bytes];
        try safetensors.convertToF32(embed.dtype, row, x[t * h ..][0..h]);
    }
}

/// Per-forward activation buffers, sized for `seq` tokens of `cfg`. Public so
/// the CUDA hybrid split can allocate one for its host-resident layers.
pub const Scratch = struct {
    normed: []f32,
    tmp: []f32,
    q: []f32,
    k: []f32,
    v: []f32,
    attn_out: []f32,
    gate: []f32,
    up: []f32,

    /// A borrowed view of the first `seq` rows of a larger scratch (no alloc).
    /// The CUDA split sizes its scratch to a full chunk once, then views it
    /// down to the actual chunk length each call, `layerForward`'s ops
    /// require exact-length slices, so passing the oversized buffer would trip
    /// a length assert (same fix as gemma3.Scratch.viewSeq). Never deinit a
    /// view, it aliases the parent scratch's memory.
    pub fn viewSeq(self: *const Scratch, seq: usize, cfg: Config) Scratch {
        return .{
            .normed = self.normed[0 .. seq * cfg.hidden],
            .tmp = self.tmp[0 .. seq * cfg.hidden],
            .q = self.q[0 .. seq * cfg.qDim()],
            .k = self.k[0 .. seq * cfg.kvDim()],
            .v = self.v[0 .. seq * cfg.kvDim()],
            .attn_out = self.attn_out[0 .. seq * cfg.qDim()],
            .gate = self.gate[0 .. seq * cfg.intermediate],
            .up = self.up[0 .. seq * cfg.intermediate],
        };
    }

    pub fn init(gpa: std.mem.Allocator, seq: usize, cfg: Config) !Scratch {
        var s: Scratch = undefined;
        s.normed = try gpa.alloc(f32, seq * cfg.hidden);
        errdefer gpa.free(s.normed);
        s.tmp = try gpa.alloc(f32, seq * cfg.hidden);
        errdefer gpa.free(s.tmp);
        s.q = try gpa.alloc(f32, seq * cfg.qDim());
        errdefer gpa.free(s.q);
        s.k = try gpa.alloc(f32, seq * cfg.kvDim());
        errdefer gpa.free(s.k);
        s.v = try gpa.alloc(f32, seq * cfg.kvDim());
        errdefer gpa.free(s.v);
        s.attn_out = try gpa.alloc(f32, seq * cfg.qDim());
        errdefer gpa.free(s.attn_out);
        s.gate = try gpa.alloc(f32, seq * cfg.intermediate);
        errdefer gpa.free(s.gate);
        s.up = try gpa.alloc(f32, seq * cfg.intermediate);
        errdefer gpa.free(s.up);
        return s;
    }

    pub fn deinit(self: *Scratch, gpa: std.mem.Allocator) void {
        gpa.free(self.normed);
        gpa.free(self.tmp);
        gpa.free(self.q);
        gpa.free(self.k);
        gpa.free(self.v);
        gpa.free(self.attn_out);
        gpa.free(self.gate);
        gpa.free(self.up);
        self.* = undefined;
    }
};

fn loadLayers(alloc: std.mem.Allocator, store: WeightStore, count: usize) ![]Layer {
    return loadLayersCfg(alloc, store, Config.vl_4b, count);
}

fn loadLayersCfg(alloc: std.mem.Allocator, store: WeightStore, cfg: Config, count: usize) ![]Layer {
    const layers = try alloc.alloc(Layer, count);
    for (layers, 0..) |*layer, i| {
        layer.* = .{
            .input_norm = try loadNorm(alloc, store, cfg, i, "input_layernorm.weight", cfg.hidden),
            .q = try loadQK(alloc, store, cfg, i, "self_attn.q_proj.weight", cfg.n_heads),
            .k = try loadQK(alloc, store, cfg, i, "self_attn.k_proj.weight", cfg.n_kv_heads),
            .v = try loadWeight(store, cfg, i, "self_attn.v_proj.weight", cfg.kvDim(), cfg.hidden),
            .o = try loadWeight(store, cfg, i, "self_attn.o_proj.weight", cfg.hidden, cfg.qDim()),
            // llama/Mistral has no per-head QK-norm; leave the slices empty
            // (the llama LayerSpec never reads them, see transformer.qkvProject).
            .q_norm = if (cfg.qk_norm) try loadNorm(alloc, store, cfg, i, "self_attn.q_norm.weight", head_dim) else &.{},
            .k_norm = if (cfg.qk_norm) try loadNorm(alloc, store, cfg, i, "self_attn.k_norm.weight", head_dim) else &.{},
            .post_norm = try loadNorm(alloc, store, cfg, i, "post_attention_layernorm.weight", cfg.hidden),
            .gate = try loadWeight(store, cfg, i, "mlp.gate_proj.weight", cfg.intermediate, cfg.hidden),
            .up = try loadWeight(store, cfg, i, "mlp.up_proj.weight", cfg.intermediate, cfg.hidden),
            .down = try loadWeight(store, cfg, i, "mlp.down_proj.weight", cfg.hidden, cfg.intermediate),
        };
    }
    return layers;
}

/// Load a Q or K projection, un-permuting its output rows when `cfg.permute_qk`
/// (llama/Mistral) so the stored ggml "NORM" (interleaved) RoPE layout matches
/// our rotate-half RoPE. llama.cpp's converter permutes each head's head_dim
/// rows as `permuted[2*i+g] = hf[g*(hd/2)+i]`; we invert that. Rows are whole
/// block-quant rows, so the un-permute is a byte-level row shuffle into `alloc`.
fn loadQK(alloc: std.mem.Allocator, store: WeightStore, cfg: Config, layer: usize, comptime suffix: []const u8, n_head_rows: usize) !Weight {
    var w = try loadWeight(store, cfg, layer, suffix, n_head_rows * head_dim, cfg.hidden);
    if (!cfg.permute_qk) return w;
    const rb = w.dtype.storageBytes(cfg.hidden); // bytes per output row
    const dst = try alloc.alloc(u8, n_head_rows * head_dim * rb);
    const half = head_dim / 2;
    for (0..n_head_rows) |h| {
        for (0..head_dim) |d| {
            const g = d / half; // 0 = first half, 1 = second half
            const idx = d % half;
            const src_row = h * head_dim + (2 * idx + g);
            const dst_row = h * head_dim + d;
            @memcpy(dst[dst_row * rb ..][0..rb], w.bytes[src_row * rb ..][0..rb]);
        }
    }
    w.bytes = dst;
    return w;
}

fn loadNorm(alloc: std.mem.Allocator, store: WeightStore, cfg: Config, layer: usize, comptime suffix: []const u8, len: usize) ![]f32 {
    var buf: [96]u8 = undefined;
    const name = try std.fmt.bufPrint(&buf, "{s}layers.{d}." ++ suffix, .{ cfg.prefix, layer });
    return loadNormNamed(alloc, store, name, len);
}

fn loadNormNamed(alloc: std.mem.Allocator, store: WeightStore, name: []const u8, len: usize) ![]f32 {
    const view = store.get(name) orelse return error.MissingTensor;
    if (view.info.elemCount() != len) return error.ShapeMismatch;
    return view.toF32Alloc(alloc);
}

fn loadWeight(store: WeightStore, cfg: Config, layer: usize, comptime suffix: []const u8, rows: usize, cols: usize) !Weight {
    var buf: [96]u8 = undefined;
    const name = try std.fmt.bufPrint(&buf, "{s}layers.{d}." ++ suffix, .{ cfg.prefix, layer });
    const view = store.get(name) orelse return error.MissingTensor;
    const shape = view.info.shape.slice();
    if (shape.len != 2 or shape[0] != rows or shape[1] != cols) return error.ShapeMismatch;
    var w = Weight.init(view.bytes, view.info.dtype, rows, cols);
    var scale_buf: [112]u8 = undefined;
    const scale_name = try std.fmt.bufPrint(&scale_buf, "{s}_scale", .{name});
    if (store.get(scale_name)) |scale_view| {
        w.scale = try scale_view.asScalarF32();
    }
    return w;
}

// --- tests -----------------------------------------------------------------

fn readF32File(gpa: std.mem.Allocator, io: std.Io, path: []const u8, n: usize) ![]f32 {
    const out = try gpa.alloc(f32, n);
    errdefer gpa.free(out);
    const file = try std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_only });
    defer file.close(io);
    const bytes = std.mem.sliceAsBytes(out);
    if (try file.readPositionalAll(io, bytes, 0) != bytes.len) return error.ShortRead;
    return out;
}

// The three conditioning variants differ in body, taps and whether the final norm
// runs. This is a fast structural pin, not a numeric one, the numbers are checked
// against ComfyUI by the krea2 test below and by the Anima fixtures, but it is what
// catches a tap list that stops agreeing with the config it indexes, which produces
// a finite encode of the wrong hidden state.
test "each encoder variant's tap list is consistent with its own body" {
    for ([_]Variant{ .krea2, .zimage, .anima }) |v| {
        const cfg = v.config();
        const taps = v.taps();
        errdefer std.debug.print("variant {t}: n_layers={d} taps={any}\n", .{ v, cfg.n_layers, taps });
        try std.testing.expect(taps.len > 0);
        // A tap may be one PAST the last layer, that is exactly the case that
        // carries the final norm, but never further, or `loadVariant` would be
        // asked for a layer the checkpoint does not have.
        try std.testing.expect(taps[taps.len - 1] <= cfg.n_layers);
        for (taps[1..], taps[0 .. taps.len - 1]) |b, a| try std.testing.expect(b > a);
        // The load-bearing rule: the final norm is applied if and only if
        // the last tap is past the last layer. krea2/Z-Image tap before a layer that
        // still has to run, so their final norm is genuinely never evaluated; Anima
        // taps past the end, where ComfyUI's `layer = "last"` always applies it.
        try std.testing.expectEqual(taps[taps.len - 1] == cfg.n_layers, v.appliesFinalNorm());
    }
    // Anima's Qwen3-0.6B body, since the whole width generalization exists for it.
    try std.testing.expectEqual(@as(usize, 28), Variant.anima.config().n_layers);
    try std.testing.expectEqual(@as(usize, 1024), Variant.anima.config().hidden);
    try std.testing.expectEqual(@as(usize, 28), Variant.anima.taps()[0]);
}

// Config detection + weight wiring against a real llama.cpp GGUF; skipped
// when the checkpoint is absent. Load-only, generation quality is validated
// end-to-end via tp-llm (a Debug 4B forward is too slow for the suite).
test "causal lm loads from real qwen3-4b gguf" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const path = "models/text_encoders/Qwen3-4B-Q4_K_M.gguf";
    std.Io.Dir.cwd().access(io, path, .{}) catch return error.SkipZigTest;

    var g = try gguf_mod.Gguf.open(gpa, io, path);
    defer g.deinit();
    var lm = try CausalLM.load(gpa, .{ .gguf = &g });
    defer lm.deinit();

    try std.testing.expectEqual(@as(usize, 36), lm.cfg.n_layers);
    try std.testing.expectEqual(@as(usize, 2560), lm.cfg.hidden);
    try std.testing.expectEqual(@as(usize, 32), lm.cfg.n_heads);
    try std.testing.expectEqual(@as(usize, 8), lm.cfg.n_kv_heads);
    try std.testing.expectEqual(@as(usize, 9728), lm.cfg.intermediate);
    try std.testing.expectEqual(@as(f64, 1e6), lm.cfg.rope_theta);
    try std.testing.expectEqualStrings("", lm.cfg.prefix);

    try std.testing.expectEqual(dtypes.DType.q6_k, lm.embed.dtype);
    // No output.weight in this file: the head ties to the embedding.
    try std.testing.expectEqual(lm.embed.bytes.ptr, lm.head.bytes.ptr);
    try std.testing.expect(lm.hasBlockQuantWeights());
    try std.testing.expectEqual(dtypes.DType.q4_k, lm.layers[0].q.dtype);
    try std.testing.expectEqual(@as(usize, 2560), lm.layers[0].input_norm.len);
}

// Config detection from synthetic GGUF metadata: a Qwen3-32B-shaped header
// (64 layers, larger than any preset) builds the right Config, and a
// block_count above max_layers is rejected.
test "config detects 32b gguf metadata" {
    const gpa = std.testing.allocator;

    var b = try gguf_mod.TestBuilder.init(gpa, 3, 0, 9);
    defer b.deinit();
    try b.kvStr("general.architecture", "qwen3");
    try b.kvUint("qwen3.block_count", 64);
    try b.kvUint("qwen3.embedding_length", 5120);
    try b.kvUint("qwen3.attention.head_count", 64);
    try b.kvUint("qwen3.attention.head_count_kv", 8);
    try b.kvUint("qwen3.attention.key_length", 128);
    try b.kvUint("qwen3.attention.value_length", 128);
    try b.kvUint("qwen3.feed_forward_length", 25600);
    try b.kvF32("qwen3.rope.freq_base", 1e6);
    const file = try b.finish(&.{});
    defer gpa.free(file);

    var g = try gguf_mod.Gguf.initFromSlice(gpa, file);
    defer g.deinit();

    const cfg = try Config.detect(.{ .gguf = &g });
    try std.testing.expectEqual(@as(usize, 64), cfg.n_layers);
    try std.testing.expectEqual(@as(usize, 5120), cfg.hidden);
    try std.testing.expectEqual(@as(usize, 64), cfg.n_heads);
    try std.testing.expectEqual(@as(usize, 8), cfg.n_kv_heads);
    try std.testing.expectEqual(@as(usize, 25600), cfg.intermediate);
    try std.testing.expectEqual(@as(f64, 1e6), cfg.rope_theta);
    try std.testing.expectEqualStrings("", cfg.prefix);
    // qwen3 defaults: QK-norm on, no q/k permute, embedded-vocab default.
    try std.testing.expect(cfg.qk_norm);
    try std.testing.expect(!cfg.permute_qk);
    try std.testing.expectEqual(vocab_size, cfg.vocab);
    try std.testing.expectEqual(@as(f32, 1e-6), cfg.rms_eps);
}

// The plain llama/Mistral family (Mistral-Nemo shape: 40 layers, hidden 5120,
// GQA 32/8, head_dim 128, vocab 131074) reuses this stack with QK-norm OFF,
// q/k un-permuted at load, its own vocab, and eps 1e-5.
test "config detects llama (mistral-nemo) gguf metadata" {
    const gpa = std.testing.allocator;

    var b = try gguf_mod.TestBuilder.init(gpa, 3, 0, 11);
    defer b.deinit();
    try b.kvStr("general.architecture", "llama");
    try b.kvUint("llama.block_count", 40);
    try b.kvUint("llama.embedding_length", 5120);
    try b.kvUint("llama.attention.head_count", 32);
    try b.kvUint("llama.attention.head_count_kv", 8);
    try b.kvUint("llama.attention.key_length", 128);
    try b.kvUint("llama.attention.value_length", 128);
    try b.kvUint("llama.feed_forward_length", 14336);
    try b.kvUint("llama.vocab_size", 131074);
    try b.kvF32("llama.rope.freq_base", 1e6);
    try b.kvF32("llama.attention.layer_norm_rms_epsilon", 1e-5);
    const file = try b.finish(&.{});
    defer gpa.free(file);

    var g = try gguf_mod.Gguf.initFromSlice(gpa, file);
    defer g.deinit();

    const cfg = try Config.detect(.{ .gguf = &g });
    try std.testing.expectEqual(@as(usize, 40), cfg.n_layers);
    try std.testing.expectEqual(@as(usize, 5120), cfg.hidden);
    try std.testing.expectEqual(@as(usize, 32), cfg.n_heads);
    try std.testing.expectEqual(@as(usize, 8), cfg.n_kv_heads);
    try std.testing.expectEqual(@as(usize, 14336), cfg.intermediate);
    try std.testing.expectEqual(@as(usize, 131074), cfg.vocab);
    try std.testing.expectEqual(@as(usize, 4096), cfg.qDim()); // 32*128 != hidden 5120
    try std.testing.expect(!cfg.qk_norm);
    try std.testing.expect(cfg.permute_qk);
    try std.testing.expectEqual(@as(f32, 1e-5), cfg.rms_eps);
}

test "config rejects block_count above max_layers" {
    const gpa = std.testing.allocator;

    var b = try gguf_mod.TestBuilder.init(gpa, 3, 0, 2);
    defer b.deinit();
    try b.kvStr("general.architecture", "qwen3");
    try b.kvUint("qwen3.block_count", Config.max_layers + 1);
    const file = try b.finish(&.{});
    defer gpa.free(file);

    var g = try gguf_mod.Gguf.initFromSlice(gpa, file);
    defer g.deinit();

    try std.testing.expectError(error.UnknownModelConfig, Config.detect(.{ .gguf = &g }));
}

// Parity against ComfyUI's Krea 2 conditioning (f32), post prefix-strip.
// Fixture from tools/dump_text_fixture.py (prompt "a fluffy orange cat
// sitting on a windowsill"); skipped when model or fixture is absent.
test "krea2 conditioning matches comfyui" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const krea2_text = @import("krea2_text.zig");
    const tokenizer_mod = @import("tp_core").tokenizer;
    const te_path = "models/text_encoders/qwen3VLInstruct4bHeretic_v10.safetensors";
    try test_gate.requireModelFile(io, te_path);
    std.Io.Dir.cwd().access(io, "testdata/text_cond.bin", .{}) catch return error.SkipZigTest;

    var tok = try tokenizer_mod.Tokenizer.init(gpa);
    defer tok.deinit();
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    try krea2_text.buildIds(&tok, gpa, "a fluffy orange cat sitting on a windowsill", &ids);

    // Cross-check tokenization against the ids the fixture was built with.
    {
        const ref_ids_file = try std.Io.Dir.cwd().openFile(io, "testdata/text_ids.bin", .{ .mode = .read_only });
        defer ref_ids_file.close(io);
        const n: usize = @intCast((try ref_ids_file.length(io)) / 4);
        const ref_ids = try gpa.alloc(u32, n);
        defer gpa.free(ref_ids);
        if (try ref_ids_file.readPositionalAll(io, std.mem.sliceAsBytes(ref_ids), 0) != n * 4) return error.ShortRead;
        try std.testing.expectEqualSlices(u32, ref_ids, ids.items);
    }

    var st = try SafeTensors.open(gpa, io, te_path);
    defer st.deinit();
    var enc = try TextEncoder.load(gpa, .{ .safetensors = &st });
    defer enc.deinit();

    const cond = try enc.encode(io, gpa, ids.items, null);
    defer gpa.free(cond);

    const offset = krea2_text.stripOffset(ids.items);
    const kept = ids.items.len - offset;
    const expected = try readF32File(gpa, io, "testdata/text_cond.bin", kept * tap_count * hidden);
    defer gpa.free(expected);

    var max_err: f32 = 0;
    var max_val: f32 = 0;
    var sum_err: f64 = 0;
    const stripped = cond[offset * tap_count * hidden ..];
    for (expected, stripped) |e, a| {
        max_err = @max(max_err, @abs(e - a));
        max_val = @max(max_val, @abs(e));
        sum_err += @abs(e - a);
    }
    const mean_err = sum_err / @as(f64, @floatFromInt(expected.len));
    std.debug.print("text parity: max_err={d:.5} mean_err={d:.6} max_val={d:.1}\n", .{ max_err, mean_err, max_val });
    // Hidden states reach magnitudes of O(100); tolerances are relative to that.
    try std.testing.expect(max_err < 0.05);
    try std.testing.expect(mean_err < 5e-4 * @as(f64, max_val));

    // A pre-set cancel flag aborts mid-encode (polled between layers).
    var canceled = std.atomic.Value(bool).init(true);
    try std.testing.expectError(error.Canceled, enc.encode(io, gpa, ids.items, &canceled));
}
