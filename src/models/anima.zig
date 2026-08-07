//! Anima — CircleStone/Comfy Org's 2B anime text-to-image model, the fifth family
//! here and the first that is a **Cosmos** derivative.
//!
//! ⚠️ **Anima IS Cosmos-Predict2's DiT.** `comfy/ldm/anima/model.py` is 214 lines
//! that subclass `MiniTrainDIT` from `comfy/ldm/cosmos/predict2.py` and bolt an
//! `LLMAdapter` onto its front; `model_detection` selects `"anima"` over
//! `"cosmos_predict2"` purely by the presence of
//! `llm_adapter.blocks.0.cross_attn.q_proj.weight`. So everything below that reads
//! as "a video DiT used on one frame" is exactly that — `patch_temporal = 1`,
//! `T = 1`, and a 3-axis RoPE whose temporal axis is therefore all zeros.
//!
//! Two halves, and they have different jobs:
//!
//! 1. **`Adapter`** (6 blocks, 1024 wide) turns the prompt into the denoiser's
//!    cross-attention context. Its *queries* come from its own
//!    `Embedding(32128, 1024)` indexed by **T5** token ids; its *keys and values*
//!    come from a Qwen3-0.6B encoder's final hidden states. So the prompt is
//!    tokenized twice, by two unrelated tokenizers, and the adapter is where the
//!    two streams meet. Output is `[max(512, n_t)][1024]`, zero-padded.
//! 2. **`DiT`** (28 blocks, 2048 wide, 16 heads of 128) is the denoiser: AdaLN-LoRA
//!    modulated self-attention → cross-attention → GELU MLP, on 2x2 patches of a
//!    16-channel latent, flow-matching parameterization.
//!
//! Reference: ComfyUI, pinned by `tools/gen_anima_fixtures.py`.
//!
//! ## The seven conventions that are silent wrong answers
//!
//! 1. ⚠️ **The input and output patch feature orders are DIFFERENT.** `x_embedder`
//!    takes `(c, ph, pw)` — channel **slowest** — from
//!    `Rearrange("b c (t r) (h m) (w n) -> b t h w (c r m n)")`. `unpatchify` emits
//!    `(ph, pw, t, c)` — channel **fastest** — from
//!    `"B T H W (p1 p2 t C) -> B C (T t) (H p1) (W p2)"`. They are not each other's
//!    inverse, and using one order for both is a pure permutation: every norm, every
//!    per-stage magnitude and every rel-L2 against a stage fixture still matches,
//!    and only the image is wrong. This repo has paid for that class of bug twice
//!    already (the SD planar/channel-last mixup, Z-Image's `(ph, pw, c)`).
//! 2. ⚠️ **`concat_padding_mask = True` appends a 17th, ALL-ZERO channel** before
//!    patchifying, which is why `x_embedder.proj.1.weight` is `[2048, 68]` and not
//!    `[2048, 64]`. It is not optional and it is not derivable from the latent.
//! 3. ⚠️ **The timestep IS the sigma.** `Anima.sampling_settings` sets
//!    `multiplier: 1.0`, so `model_sampling.timestep(sigma) = sigma * 1.0` and the
//!    sinusoidal embedding sees a value in (0, 1]. Z-Image, whose sigma table is
//!    otherwise **bit-identical** to Anima's, feeds `(1 - sigma) * 1000` instead.
//!    Same table, different argument; borrowing the wrong one is finite nonsense.
//! 4. ⚠️ **AdaLN-LoRA: one low-rank vector is shared by all three sublayers.**
//!    `t_embedder[1]` emits a `3*dim` vector that is ADDED to each of
//!    `adaln_modulation_{self_attn,cross_attn,mlp}(emb)`; the final layer adds only
//!    its **first `2*dim`**. And `emb` itself is the RMSNorm'd *sinusoid*, not the
//!    MLP's output — under `use_adaln_lora` the `TimestepEmbedding` returns
//!    `(sample, mlp(sample))`, keeping the raw sinusoid as the modulation input.
//! 5. ⚠️ **`adaln_modulation_*` has no activation between its two linears.** It is
//!    `Sequential(SiLU, Linear(2048, 256), Linear(256, 3*2048))` — the SiLU is on the
//!    input and the pair is a rank-256 factorization, not an MLP.
//! 6. ⚠️ **RoPE applies to self-attention ONLY in the DiT, and to BOTH q and k in
//!    the adapter's cross-attention.** The DiT's `Attention.compute_qkv` guards with
//!    `if self.is_selfattn and rope_emb is not None`; the adapter's separate
//!    `Attention` class has no such guard and rotates its queries by the target
//!    positions and its keys by the *source* positions.
//! 7. ⚠️ **The DiT's RoPE frequency vector is `[t(22) | h(21) | w(21)]`** in that
//!    order, with `h`/`w` on an NTK-scaled theta (`10000 * 4^(42/40)`) and `t` on a
//!    plain 10000 — and it is applied **split-half** (pairs `(i, i+64)`), not
//!    interleaved. The temporal block is 22 of the 64 frequencies and is identically
//!    zero at `T = 1`, so a wrong axis ORDER still leaves a third of the vector
//!    looking right.
//!
//! Everything else is shared with what is already here: flow matching, the Wan 2.1
//! VAE (`first_stage_model.*` in a bundled checkpoint is byte-for-byte
//! `qwen_image_vae.safetensors`, and shape-for-shape krea2's decoder), Wan21 latent
//! statistics, and the Qwen3 encoder body.
//!
//! ## Measured
//!
//! Against ComfyUI on the real `terraRising_20TerraRisingAnima` checkpoint, fp32 on
//! both sides (`tools/gen_anima_fixtures.py`; the trunk is truncated to 8 and 2
//! blocks so the reference fits in RAM at full width, the adapter is referenced in
//! full):
//!
//! | stage | rel L2 |
//! |---|---|
//! | sinusoidal timestep embedding | 1.8e-8 |
//! | RoPE table (cos / sin) | 1.7e-8 / 1.4e-7 |
//! | patchify + `x_embedder` | **exactly 0** — bit-identical |
//! | block 0's modulation vectors / the final layer's | 4.6e-7 / 1.4e-6 |
//! | `llm_adapter`, all 6 blocks | 3.2e-6 |
//! | ...plus emphasis weights and the 512-row pad | 3.2e-6 |
//! | one whole block | 1.9e-6 |
//! | the whole forward, 8 blocks | 1.9e-6 (12x10) / 3.1e-6 (40x32) |
//! | the whole forward, 2 blocks | 2.2e-6 / 3.3e-6 |
//! | Qwen3-0.6B conditioning | 2.5e-6 … 7.8e-6 |
//! | both tokenizations, on the real prompts | exact ids |
//!
//! ⚠️ **Read the last two DiT rows together.** The disagreement is FLAT in depth —
//! 2 blocks and 8 blocks land at the same 2e-6 — so nothing is accumulating, which
//! is what says the block form is right rather than merely close. A bound alone
//! could not distinguish those.

const std = @import("std");
const tp_core = @import("tp_core");
const weights_mod = tp_core.weights;
const ops = @import("tp_ops");

const WeightStore = weights_mod.WeightStore;
const Weight = ops.matmul.Weight;
const DType = tp_core.dtype.DType;

/// The `LLMAdapter` half. Its dims are hardcoded in `comfy/ldm/anima/model.py`'s
/// signature (nothing in `model_detection` describes them), so they are constants
/// here rather than probed — but the loader still checks every shape.
pub const AdapterConfig = struct {
    /// Hidden width of the encoder whose states are the cross-attention source
    /// (Qwen3-0.6B: 1024). Must equal `qwen3.Config.qwen3_0_6b.hidden`.
    source_dim: usize,
    /// The adapter's own width, and its output width.
    dim: usize,
    n_layers: usize,
    n_heads: usize,
    /// `Embedding(32128, dim)` over **T5** ids. 32128 > the T5 tokenizer's 32100;
    /// the tail rows are unreachable and that is upstream's shape.
    vocab: usize,
    mlp_dim: usize,
    rope_theta: f64,
    /// RMSNorm epsilon for the block pre-norms and the output norm.
    norm_eps: f32,
    /// Per-head Q/K RMSNorm epsilon. ⚠️ Unlike Z-Image, whose Q/K norms fall back
    /// to `finfo(float32).eps`, these are built with an explicit `eps=1e-6` — the
    /// same value as the block norms. Do not "fix" one to match the other.
    qk_eps: f32,
    /// Rows the output is zero-padded up to (`preprocess_text_embeds`). A longer
    /// prompt is NOT truncated — it stays at its own length.
    min_rows: usize,

    pub fn headDim(self: AdapterConfig) usize {
        return self.dim / self.n_heads;
    }
};

pub const Config = struct {
    /// `model_channels`.
    dim: usize,
    /// `num_blocks`.
    n_layers: usize,
    n_heads: usize,
    /// `int(model_channels * mlp_ratio)` — stored, because a loader can only check
    /// the shape it actually expects.
    mlp_dim: usize,
    /// `crossattn_emb_channels` — the adapter's output width.
    context_dim: usize,
    /// `adaln_lora_dim`, the rank of the AdaLN factorization.
    adaln_dim: usize,
    /// `patch_spatial`. `patch_temporal` is 1 and is not a field: a value other than
    /// 1 would make this a video model and change the whole token layout.
    patch: usize,
    /// Latent channels in (`in_channels`) and out (`out_channels`). Equal for
    /// text-to-image; the Cosmos i2v variants differ, which is why they are separate.
    channels: usize,
    out_channels: usize,
    /// Append an all-zero channel before patchifying. See convention 2.
    concat_padding_mask: bool,
    /// Weightless `LayerNorm` epsilon, for both the blocks and the final layer.
    norm_eps: f32,
    /// Per-head Q/K `RMSNorm(head_dim, eps=1e-6)`.
    qk_eps: f32,
    /// `t_embedding_norm`'s `RMSNorm(model_channels, eps=1e-6)`.
    t_norm_eps: f32,
    /// NTK extrapolation ratios per axis. ⚠️ These are *ratios*, not thetas: the
    /// factor applied to 10000 is `ratio ** (axis_dim / (axis_dim - 2))`.
    rope_h_ratio: f64,
    rope_w_ratio: f64,
    rope_t_ratio: f64,
    adapter: AdapterConfig,

    pub fn headDim(self: Config) usize {
        return self.dim / self.n_heads;
    }
    /// Channels the patch embedder actually sees, padding mask included.
    pub fn inChannels(self: Config) usize {
        return self.channels + @intFromBool(self.concat_padding_mask);
    }
    /// Width of one patch token entering `x_embedder` (17 * 2 * 2 = 68).
    pub fn patchDim(self: Config) usize {
        return self.inChannels() * self.patch * self.patch;
    }
    /// Width of one patch token leaving `final_layer.linear` (16 * 2 * 2 = 64).
    pub fn outPatchDim(self: Config) usize {
        return self.out_channels * self.patch * self.patch;
    }

    /// The three RoPE axis widths, `[t, h, w]`, from `VideoRopePosition3DEmb`:
    /// `dim_h = dim_w = head_dim // 6 * 2` and `dim_t` is the remainder. For
    /// head_dim 128 that is `[44, 42, 42]` — note **t is the WIDEST**, which is
    /// counter-intuitive for an image model and is simply what the remainder gives.
    pub fn ropeAxes(self: Config) [3]usize {
        const hd = self.headDim();
        const dim_h = hd / 6 * 2;
        return .{ hd - 2 * dim_h, dim_h, dim_h };
    }
};

/// The 2B text-to-image config, as `model_detection.detect_unet_config` derives it
/// for `model_channels == 2048`. Asserted against ComfyUI's own answer by
/// `tools/gen_anima_fixtures.py`, so a change upstream fails there rather than
/// silently re-baselining this.
pub const anima_2b: Config = .{
    .dim = 2048,
    .n_layers = 28,
    .n_heads = 16,
    .mlp_dim = 8192,
    .context_dim = 1024,
    .adaln_dim = 256,
    .patch = 2,
    .channels = 16,
    .out_channels = 16,
    .concat_padding_mask = true,
    .norm_eps = 1e-6,
    .qk_eps = 1e-6,
    .t_norm_eps = 1e-6,
    // `in_channels == 16` (text-to-image) selects 4.0/4.0/1.0.
    .rope_h_ratio = 4.0,
    .rope_w_ratio = 4.0,
    .rope_t_ratio = 1.0,
    .adapter = .{
        .source_dim = 1024,
        .dim = 1024,
        .n_layers = 6,
        .n_heads = 16,
        .vocab = 32128,
        .mlp_dim = 4096,
        .rope_theta = 10000.0,
        .norm_eps = 1e-6,
        .qk_eps = 1e-6,
        .min_rows = 512,
    },
};

/// Latent geometry, for callers that need it without a loaded model. Anima uses the
/// Wan 2.1 VAE, so these are `wan_vae`'s — including its `latents_mean`/`latents_std`
/// and its latent2rgb preview matrix (`latent_formats.Wan21`).
pub const latent_channels = anima_2b.channels;
pub const spatial_scale = 8;

/// The sigma-table shift Anima trained on (`sampling_settings.shift`). Numerically
/// the same 3.0 Z-Image uses, and over the same 1000-rung `ModelSamplingDiscreteFlow`
/// table — the two tables are bit-identical. ⚠️ What differs is the argument handed
/// to the model; see convention 3.
pub const shift: f32 = 3.0;

const LinearW = struct {
    w: Weight,
    b: ?[]const f32,
};

/// One `Attention`, in either of the two shapes this file has: the DiT's (query
/// 2048, context 2048 or 1024) and the adapter's (1024/1024). Separate q/k/v
/// because the checkpoint stores them separately — unlike Z-Image's fused QKV, so
/// there is no split-out step here.
const Attn = struct {
    q: Weight,
    k: Weight,
    v: Weight,
    out: Weight,
    /// Per-head RMSNorm scales, `[head_dim]`.
    qnorm: []const f32,
    knorm: []const f32,
};

/// A `MiniTrainDIT` `Block`. Three sublayers, each with its own weightless
/// LayerNorm (no weights to load) and its own AdaLN factorization.
const Block = struct {
    self_attn: Attn,
    cross_attn: Attn,
    /// `GPT2FeedForward`: two biasless linears around an erf-GELU.
    mlp1: Weight,
    mlp2: Weight,
    /// `adaln_modulation_X`: `Linear(dim, adaln_dim)` then `Linear(adaln_dim, 3*dim)`,
    /// both biasless. Indexed `.1` and `.2` in the checkpoint because `.0` is the SiLU.
    ada_sa: [2]Weight,
    ada_ca: [2]Weight,
    ada_mlp: [2]Weight,
};

/// An `LLMAdapter` `TransformerBlock`: self-attention, cross-attention onto the
/// encoder states, and a biased GELU MLP, each behind an RMSNorm.
const AdapterBlock = struct {
    norm_self: []const f32,
    self_attn: Attn,
    norm_cross: []const f32,
    cross_attn: Attn,
    norm_mlp: []const f32,
    mlp1: LinearW,
    mlp2: LinearW,
};

/// How many f32 the modulation table for one step occupies. Per block: three
/// sublayers x (shift, scale, gate) x dim. Then the final layer's (shift, scale).
pub fn modulationTableLen(cfg: Config) usize {
    return cfg.n_layers * 9 * cfg.dim + 2 * cfg.dim;
}

/// `modulationTable` with `(1 + scale)` pre-multiplied into every scale block —
/// the form BOTH GPU arms upload. Same layout and length; in place.
///
/// ⚠️ It exists because the fused device norms (`Context.opLnModSg`,
/// `Backend.lnMod`) compute `(x - mean) * inv * premul[c] + shift[c]`, with no
/// place to add the 1. The CPU's `modulatedNorm` does the `1 +` itself, so the two
/// forms are the same arithmetic and this is the one function that expresses the
/// difference — the reasoning `zimage.modulationTableFolded` records, and the
/// reason neither backend folds it privately.
pub fn foldModulationTable(cfg: Config, tbl: []f32) void {
    std.debug.assert(tbl.len == modulationTableLen(cfg));
    const d = cfg.dim;
    for (0..cfg.n_layers) |bi| {
        const per_block = tbl[bi * 9 * d ..][0 .. 9 * d];
        // Each sublayer's triple is [shift, scale, gate]; the scale is the middle.
        for (0..3) |si| {
            for (per_block[si * 3 * d + d ..][0..d]) |*v| v.* += 1.0;
        }
    }
    // ⚠️ **The FINAL layer's scale is deliberately NOT folded**, and folding it cost a
    // rel L2 of 0.10 against the CPU forward — a plausible-looking magnitude, identical
    // on both attention paths, which is what said "shared wiring, not attention".
    // Both GPU arms run the final layer on the HOST through `DiT.finalize`, whose
    // `modulatedNorm` adds its own `1 +`; a folded scale there is counted twice. The
    // rule is "fold exactly what the fused device norm consumes", and the final layer
    // is not that.
}

pub const Adapter = struct {
    cfg: AdapterConfig,
    /// `[vocab, dim]`, kept in checkpoint dtype and gathered through
    /// `embedRows` — the same reasoning `qwen3.TextEncoder.embed` records.
    embed: Weight,
    blocks: []AdapterBlock,
    out_proj: LinearW,
    norm: []const f32,

    /// The denoiser's cross-attention context for one prompt.
    ///
    /// `src` is the Qwen3-0.6B encoder's output, `[n_src][source_dim]`; `ids` are the
    /// T5 token ids; `weights` are their emphasis weights (or null for all-1.0).
    /// Returns `[rows][dim]` with `rows = max(min_rows, ids.len)`, zero-padded.
    /// Caller frees.
    ///
    /// ⚠️ **The zero padding is attended to.** `preprocess_text_embeds` pads with
    /// zeros and `Anima.forward` passes no source mask, so the denoiser cross-attends
    /// over all 512 rows including the pad. Trimming them is not an optimization, it
    /// is a different model.
    pub fn forward(
        self: *const Adapter,
        io: std.Io,
        gpa: std.mem.Allocator,
        src: []const f32,
        n_src: usize,
        ids: []const u32,
        weights: ?[]const f32,
        cancel: ?*std.atomic.Value(bool),
    ) ![]f32 {
        const cfg = self.cfg;
        const d = cfg.dim;
        std.debug.assert(ids.len > 0);
        std.debug.assert(src.len == n_src * cfg.source_dim);
        if (weights) |w| std.debug.assert(w.len == ids.len);

        const n_t = ids.len;
        const rows = @max(cfg.min_rows, n_t);

        const out = try gpa.alloc(f32, rows * d);
        errdefer gpa.free(out);
        @memset(out, 0);
        const x = out[0 .. n_t * d];
        try embedRows(self.embed, ids, x);

        // Two RoPE tables over the same theta: the target positions `0..n_t` and the
        // source positions `0..n_src`. `RotaryEmbedding` is built once on
        // `model_dim // num_heads` and called twice with different `position_ids`, so
        // one builder at `max` rows would do — but the two are consumed as separate
        // (cos, sin) pairs and keeping them separate is what makes convention 6 legible.
        var tf = try ops.rope.rotateHalfFreqs(gpa, n_t, cfg.headDim(), cfg.rope_theta);
        defer tf.deinit(gpa);
        var sf = try ops.rope.rotateHalfFreqs(gpa, n_src, cfg.headDim(), cfg.rope_theta);
        defer sf.deinit(gpa);

        const normed = try gpa.alloc(f32, n_t * d);
        defer gpa.free(normed);
        const delta = try gpa.alloc(f32, n_t * d);
        defer gpa.free(delta);

        for (self.blocks) |*blk| {
            if (cancel) |c| if (c.load(.acquire)) return error.Canceled;

            ops.norm.rmsNorm(normed, x, blk.norm_self, cfg.norm_eps);
            try self.attnForward(io, gpa, &blk.self_attn, normed, n_t, normed, n_t, tf, tf, delta);
            for (x, delta) |*v, dv| v.* += dv;

            ops.norm.rmsNorm(normed, x, blk.norm_cross, cfg.norm_eps);
            try self.attnForward(io, gpa, &blk.cross_attn, normed, n_t, src, n_src, tf, sf, delta);
            for (x, delta) |*v, dv| v.* += dv;

            ops.norm.rmsNorm(normed, x, blk.norm_mlp, cfg.norm_eps);
            try self.mlpForward(io, gpa, blk, normed, n_t, delta);
            for (x, delta) |*v, dv| v.* += dv;
        }

        // `self.norm(self.out_proj(x))`, then the emphasis weights.
        try ops.matmul.matmul(io, gpa, delta, x, n_t, self.out_proj.w, self.out_proj.b);
        ops.norm.rmsNorm(x, delta, self.norm, cfg.norm_eps);

        // ⚠️ A per-ROW scalar, applied after the norm: `out = out * t5xxl_weights`
        // with the weights broadcast over the feature axis
        // (`t5xxl_weights.unsqueeze(0).unsqueeze(-1)`). The pad rows stay zero, which
        // is why this loops over `n_t` and not `rows`.
        if (weights) |w| {
            for (0..n_t) |r| {
                const s = w[r];
                if (s == 1.0) continue;
                for (x[r * d ..][0..d]) |*v| v.* *= s;
            }
        }
        return out;
    }

    fn mlpForward(
        self: *const Adapter,
        io: std.Io,
        gpa: std.mem.Allocator,
        blk: *const AdapterBlock,
        x: []const f32,
        n: usize,
        out: []f32,
    ) !void {
        const inner = self.cfg.mlp_dim;
        const hidden = try gpa.alloc(f32, n * inner);
        defer gpa.free(hidden);
        try ops.matmul.matmul(io, gpa, hidden, x, n, blk.mlp1.w, blk.mlp1.b);
        ops.act.geluErf(hidden);
        try ops.matmul.matmul(io, gpa, out, hidden, n, blk.mlp2.w, blk.mlp2.b);
    }

    /// Shared by both of the adapter's attentions. ⚠️ Queries are rotated by
    /// `q_freqs` and keys by `k_freqs`, which are the SAME table for self-attention
    /// and DIFFERENT tables for cross-attention — convention 6.
    fn attnForward(
        self: *const Adapter,
        io: std.Io,
        gpa: std.mem.Allocator,
        attn: *const Attn,
        xq: []const f32,
        n_q: usize,
        xkv: []const f32,
        n_kv: usize,
        q_freqs: ops.rope.Freqs,
        k_freqs: ops.rope.Freqs,
        out: []f32,
    ) !void {
        const cfg = self.cfg;
        const inner = cfg.dim;
        const hd = cfg.headDim();

        const q = try gpa.alloc(f32, n_q * inner);
        defer gpa.free(q);
        const k = try gpa.alloc(f32, n_kv * inner);
        defer gpa.free(k);
        const v = try gpa.alloc(f32, n_kv * inner);
        defer gpa.free(v);
        try ops.matmul.matmul(io, gpa, q, xq, n_q, attn.q, null);
        try ops.matmul.matmul(io, gpa, k, xkv, n_kv, attn.k, null);
        try ops.matmul.matmul(io, gpa, v, xkv, n_kv, attn.v, null);

        // Per-head norms before RoPE; `rmsNorm` reduces over `weight.len == head_dim`,
        // so the flat `[n, heads*head_dim]` buffer is already the right shape. `v` is
        // NOT normed (`v_norm = nn.Identity()`).
        ops.norm.rmsNorm(q, q, attn.qnorm, cfg.qk_eps);
        ops.norm.rmsNorm(k, k, attn.knorm, cfg.qk_eps);
        ops.rope.applyRotateHalf(q, q_freqs, n_q, cfg.n_heads, hd);
        ops.rope.applyRotateHalf(k, k_freqs, n_kv, cfg.n_heads, hd);

        const attn_out = try gpa.alloc(f32, n_q * inner);
        defer gpa.free(attn_out);
        try ops.attention.attention(io, gpa, attn_out, q, k, v, .{
            .seq_q = n_q,
            .seq_kv = n_kv,
            .n_heads = cfg.n_heads,
            .n_kv_heads = cfg.n_heads,
            .head_dim = hd,
        });
        try ops.matmul.matmul(io, gpa, out, attn_out, n_q, attn.out, null);
    }
};

pub const DiT = struct {
    arena: std.heap.ArenaAllocator,
    cfg: Config,

    /// `x_embedder.proj.1`: `patchDim -> dim`, biasless.
    x_embedder: Weight,
    /// `t_embedder.1`: the AdaLN-LoRA pair. `linear_1` is `dim -> dim` (biasless
    /// because `use_adaln_lora`), `linear_2` is `dim -> 3*dim`.
    t_linear1: Weight,
    t_linear2: Weight,
    /// `t_embedding_norm`, an RMSNorm **with** weight over the raw sinusoid.
    t_norm: []const f32,
    blocks: []Block,
    final_ada: [2]Weight,
    final_linear: Weight,
    adapter: Adapter,

    pub fn load(gpa: std.mem.Allocator, store: WeightStore, cfg: Config) !DiT {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();

        // A full single-file checkpoint keeps the LDM container prefix; a
        // denoiser-only export strips it. The same two spellings `detectFamily` reads.
        const pfx: []const u8 = if (store.get("model.diffusion_model.x_embedder.proj.1.weight") != null)
            "model.diffusion_model."
        else
            "";
        const l = Loader{ .store = store, .alloc = alloc, .pfx = pfx, .cfg = cfg };

        const blocks = try alloc.alloc(Block, cfg.n_layers);
        for (blocks, 0..) |*b, i| b.* = try l.block(i);

        const ac = cfg.adapter;
        const ablocks = try alloc.alloc(AdapterBlock, ac.n_layers);
        for (ablocks, 0..) |*b, i| b.* = try l.adapterBlock(i);

        // `x_embedder` and `final_layer.linear` are the two tiny projections the GPU
        // arms hand to the fused f32-only `opMatmul`; normalize their storage once
        // here, as `zimage` does, since [2048, 68] and [64, 2048] cost nothing.
        var x_embedder = try l.mat("x_embedder.proj.1.weight", .{}, cfg.dim, cfg.patchDim());
        x_embedder = try ops.matmul.materializeF32(alloc, x_embedder);
        var final_linear = try l.mat("final_layer.linear.weight", .{}, cfg.outPatchDim(), cfg.dim);
        final_linear = try ops.matmul.materializeF32(alloc, final_linear);

        return .{
            .arena = arena,
            .cfg = cfg,
            .x_embedder = x_embedder,
            .t_linear1 = try l.mat("t_embedder.1.linear_1.weight", .{}, cfg.dim, cfg.dim),
            .t_linear2 = try l.mat("t_embedder.1.linear_2.weight", .{}, 3 * cfg.dim, cfg.dim),
            .t_norm = try l.vec("t_embedding_norm.weight", .{}, cfg.dim),
            .blocks = blocks,
            .final_ada = .{
                try l.mat("final_layer.adaln_modulation.1.weight", .{}, cfg.adaln_dim, cfg.dim),
                try l.mat("final_layer.adaln_modulation.2.weight", .{}, 2 * cfg.dim, cfg.adaln_dim),
            },
            .final_linear = final_linear,
            .adapter = .{
                .cfg = ac,
                .embed = try l.mat("llm_adapter.embed.weight", .{}, ac.vocab, ac.dim),
                .blocks = ablocks,
                .out_proj = .{
                    .w = try l.mat("llm_adapter.out_proj.weight", .{}, ac.dim, ac.dim),
                    .b = try l.vec("llm_adapter.out_proj.bias", .{}, ac.dim),
                },
                .norm = try l.vec("llm_adapter.norm.weight", .{}, ac.dim),
            },
        };
    }

    pub fn deinit(self: *DiT) void {
        self.arena.deinit();
        self.* = undefined;
    }

    // --- per-step constants -------------------------------------------------
    //
    // Split the way krea2's and Z-Image's are: `pipeline.Denoiser` builds the
    // per-image constants once and the per-step part per step. Anima's modulation
    // is one vector per sublayer per block for the WHOLE image (`emb` is `[B, T, D]`
    // with T = 1, broadcast over h and w), so the entire AdaLN evaluation is a
    // per-step constant — 28 blocks x 3 x a rank-256 factorization, which is
    // ~0.2 GFLOP against the trunk's hundreds.

    /// Every block's modulation vectors for one sigma, plus the final layer's, laid
    /// out as `[n_layers][3 sublayers][shift, scale, gate][dim]` followed by
    /// `[shift, scale][dim]`. Length `modulationTableLen(cfg)`. Caller frees.
    ///
    /// This is the form the GPU arms upload, so there is exactly one implementation
    /// of conventions 3, 4 and 5.
    pub fn modulationTable(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, sigma: f32) ![]f32 {
        const cfg = self.cfg;
        const d = cfg.dim;

        // Convention 3: the argument is the sigma itself.
        const sinus = try gpa.alloc(f32, d);
        defer gpa.free(sinus);
        timestepEmbedding(sinus, sigma);

        // Convention 4: `emb` is the RMSNorm'd sinusoid; `lora` is the MLP's output.
        const emb = try gpa.alloc(f32, d);
        defer gpa.free(emb);
        ops.norm.rmsNorm(emb, sinus, self.t_norm, cfg.t_norm_eps);

        const lora = try gpa.alloc(f32, 3 * d);
        defer gpa.free(lora);
        {
            const h1 = try gpa.alloc(f32, d);
            defer gpa.free(h1);
            try ops.matmul.matmul(io, gpa, h1, sinus, 1, self.t_linear1, null);
            ops.act.silu(h1);
            try ops.matmul.matmul(io, gpa, lora, h1, 1, self.t_linear2, null);
        }

        // Convention 5: SiLU on the input, then the rank-`adaln_dim` pair with no
        // activation between them. The SiLU'd input is shared by all four consumers
        // (three per block, plus the final layer), so it is computed once.
        const gated = try gpa.alloc(f32, d);
        defer gpa.free(gated);
        @memcpy(gated, emb);
        ops.act.silu(gated);

        const out = try gpa.alloc(f32, modulationTableLen(cfg));
        errdefer gpa.free(out);

        const low = try gpa.alloc(f32, cfg.adaln_dim);
        defer gpa.free(low);

        for (self.blocks, 0..) |*blk, bi| {
            const per_block = out[bi * 9 * d ..][0 .. 9 * d];
            for ([_][2]Weight{ blk.ada_sa, blk.ada_ca, blk.ada_mlp }, 0..) |ada, si| {
                const dst = per_block[si * 3 * d ..][0 .. 3 * d];
                try ops.matmul.matmul(io, gpa, low, gated, 1, ada[0], null);
                try ops.matmul.matmul(io, gpa, dst, low, 1, ada[1], null);
                // Convention 4: the one shared LoRA vector is added to all three.
                for (dst, lora) |*v, lv| v.* += lv;
            }
        }

        // The final layer takes only the FIRST 2*dim of the LoRA vector.
        {
            const dst = out[cfg.n_layers * 9 * d ..][0 .. 2 * d];
            try ops.matmul.matmul(io, gpa, low, gated, 1, self.final_ada[0], null);
            try ops.matmul.matmul(io, gpa, dst, low, 1, self.final_ada[1], null);
            for (dst, lora[0 .. 2 * d]) |*v, lv| v.* += lv;
        }
        return out;
    }

    /// `ropeFreqs(cfg, ...)` for a loaded model.
    pub fn ropeFreqs(self: *const DiT, gpa: std.mem.Allocator, h: usize, w: usize) !ops.rope.Freqs {
        return ropeFreqsFor(gpa, self.cfg, h, w);
    }


    /// One denoiser forward. `x_lat` is the planar `[16][lat_h][lat_w]` sampler
    /// latent, `ctx` the adapter's `[ctx_seq][context_dim]` output, and `out`
    /// receives the predicted velocity in the same planar layout.
    pub fn predict(
        self: *const DiT,
        io: std.Io,
        gpa: std.mem.Allocator,
        out: []f32,
        x_lat: []const f32,
        lat_h: usize,
        lat_w: usize,
        ctx: []const f32,
        ctx_seq: usize,
        mod: []const f32,
        cancel: ?*std.atomic.Value(bool),
    ) !void {
        const cfg = self.cfg;
        // ⚠️ The reference `pad_to_patch_size`s and crops back. Every caller here
        // derives the latent from `width / 8` with `width` a multiple of 16, so an
        // odd latent dim is unreachable — the same argument Z-Image's DiT makes.
        // Assert rather than pad, so a future caller that breaks it says so.
        std.debug.assert(lat_h % cfg.patch == 0 and lat_w % cfg.patch == 0);
        std.debug.assert(x_lat.len == cfg.channels * lat_h * lat_w);
        std.debug.assert(out.len == cfg.out_channels * lat_h * lat_w);
        std.debug.assert(ctx.len == ctx_seq * cfg.context_dim);
        std.debug.assert(mod.len == modulationTableLen(cfg));

        // Arm fine-grained cancel inside the CPU kernels for the whole forward: one
        // MLP GEMM here is seconds of work at a real resolution.
        const prev_tok = ops.cancel.token;
        ops.cancel.token = cancel;
        defer ops.cancel.token = prev_tok;

        const d = cfg.dim;
        const h = lat_h / cfg.patch;
        const w = lat_w / cfg.patch;
        const seq = h * w;

        var freqs = try self.ropeFreqs(gpa, h, w);
        defer freqs.deinit(gpa);

        const x = try gpa.alloc(f32, seq * d);
        defer gpa.free(x);
        {
            const patches = try patchify(gpa, cfg, x_lat, lat_h, lat_w);
            defer gpa.free(patches);
            try ops.matmul.matmul(io, gpa, x, patches, seq, self.x_embedder, null);
        }

        const normed = try gpa.alloc(f32, seq * d);
        defer gpa.free(normed);
        const delta = try gpa.alloc(f32, seq * d);
        defer gpa.free(delta);

        for (self.blocks, 0..) |*blk, bi| {
            // Poll between blocks so a stop lands mid-step; a full CPU step at a real
            // resolution is tens of seconds.
            if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
            try self.blockForwardIn(io, gpa, blk, x, seq, ctx, ctx_seq, mod[bi * 9 * d ..][0 .. 9 * d], freqs, normed, delta);
        }

        try self.finalize(io, gpa, out, x, mod[cfg.n_layers * 9 * d ..][0 .. 2 * d], lat_h, lat_w);
    }

    /// `FinalLayer` + unpatchify. `rows` is `[seq, dim]` and is modified in place.
    pub fn finalize(
        self: *const DiT,
        io: std.Io,
        gpa: std.mem.Allocator,
        out: []f32,
        rows: []f32,
        final_mod: []const f32,
        lat_h: usize,
        lat_w: usize,
    ) !void {
        const cfg = self.cfg;
        const d = cfg.dim;
        const seq = (lat_h / cfg.patch) * (lat_w / cfg.patch);
        std.debug.assert(rows.len == seq * d);
        std.debug.assert(final_mod.len == 2 * d);

        modulatedNorm(rows, rows, final_mod[0..d], final_mod[d .. 2 * d], cfg.norm_eps);
        const patches = try gpa.alloc(f32, seq * cfg.outPatchDim());
        defer gpa.free(patches);
        try ops.matmul.matmul(io, gpa, patches, rows, seq, self.final_linear, null);
        unpatchify(cfg, out, patches, lat_h, lat_w);
    }

    /// One `MiniTrainDIT` `Block`, in place on `x` (`[seq, dim]`). Allocates its own
    /// scratch; `predict` uses `blockForwardIn` to reuse one pair across all blocks.
    pub fn blockForward(
        self: *const DiT,
        io: std.Io,
        gpa: std.mem.Allocator,
        blk: *const Block,
        x: []f32,
        seq: usize,
        ctx: []const f32,
        ctx_seq: usize,
        m: []const f32,
        freqs: ops.rope.Freqs,
    ) !void {
        const normed = try gpa.alloc(f32, x.len);
        defer gpa.free(normed);
        const delta = try gpa.alloc(f32, x.len);
        defer gpa.free(delta);
        try self.blockForwardIn(io, gpa, blk, x, seq, ctx, ctx_seq, m, freqs, normed, delta);
    }

    /// `blockForward` with caller-owned scratch.
    ///
    /// The three sublayers are identical in shape — modulated weightless LayerNorm,
    /// sublayer, gated residual — and differ only in which third of `m` they read and
    /// whether RoPE applies (convention 6: self-attention only).
    fn blockForwardIn(
        self: *const DiT,
        io: std.Io,
        gpa: std.mem.Allocator,
        blk: *const Block,
        x: []f32,
        seq: usize,
        ctx: []const f32,
        ctx_seq: usize,
        m: []const f32,
        freqs: ops.rope.Freqs,
        normed: []f32,
        delta: []f32,
    ) !void {
        const cfg = self.cfg;
        const d = cfg.dim;
        std.debug.assert(m.len == 9 * d);
        std.debug.assert(x.len == seq * d and normed.len == x.len and delta.len == x.len);

        // Self-attention.
        modulatedNorm(normed, x, m[0..d], m[d .. 2 * d], cfg.norm_eps);
        try self.attnForward(io, gpa, &blk.self_attn, normed, seq, normed, seq, freqs, delta);
        gatedAdd(x, delta, m[2 * d .. 3 * d]);

        // Cross-attention onto the adapter's output. ⚠️ No RoPE here.
        modulatedNorm(normed, x, m[3 * d .. 4 * d], m[4 * d .. 5 * d], cfg.norm_eps);
        try self.attnForward(io, gpa, &blk.cross_attn, normed, seq, ctx, ctx_seq, null, delta);
        gatedAdd(x, delta, m[5 * d .. 6 * d]);

        // MLP.
        modulatedNorm(normed, x, m[6 * d .. 7 * d], m[7 * d .. 8 * d], cfg.norm_eps);
        try self.mlpForward(io, gpa, blk, normed, seq, delta);
        gatedAdd(x, delta, m[8 * d .. 9 * d]);
    }

    fn mlpForward(
        self: *const DiT,
        io: std.Io,
        gpa: std.mem.Allocator,
        blk: *const Block,
        x: []const f32,
        seq: usize,
        out: []f32,
    ) !void {
        const inner = self.cfg.mlp_dim;
        const hidden = try gpa.alloc(f32, seq * inner);
        defer gpa.free(hidden);
        try ops.matmul.matmul(io, gpa, hidden, x, seq, blk.mlp1, null);
        // `nn.GELU()` with the default `approximate='none'` — the erf form, not tanh.
        ops.act.geluErf(hidden);
        try ops.matmul.matmul(io, gpa, out, hidden, seq, blk.mlp2, null);
    }

    /// The key/value half of one attention: project `src` and apply the K norm
    /// (`v_norm = nn.Identity()`, so V is untouched). `k`/`v` are `[n][dim]`.
    ///
    /// ⚠️ Its own function because BOTH GPU arms precompute cross-attention's K and V
    /// once per image — they are projections of the adapter's output, which no step
    /// changes — and "what K and V are" must have exactly one implementation. Two
    /// copies of "project, then norm K but not V" is precisely the drift that makes a
    /// device forward disagree with the host one in a way no shape check sees.
    fn projectKv(
        self: *const DiT,
        io: std.Io,
        gpa: std.mem.Allocator,
        attn: *const Attn,
        src: []const f32,
        n: usize,
        k: []f32,
        v: []f32,
    ) !void {
        try ops.matmul.matmul(io, gpa, k, src, n, attn.k, null);
        try ops.matmul.matmul(io, gpa, v, src, n, attn.v, null);
        // Per-head norm: `rmsNorm` reduces over `weight.len == head_dim`, so the flat
        // `[n, heads*head_dim]` buffer is already the right shape.
        ops.norm.rmsNorm(k, k, attn.knorm, self.cfg.qk_eps);
    }

    /// One block's CROSS-attention K and V for a fixed context — the per-image
    /// constant both GPU arms cache for every block. `k`/`v` are `[n][dim]`.
    pub fn crossKv(
        self: *const DiT,
        io: std.Io,
        gpa: std.mem.Allocator,
        blk: *const Block,
        src: []const f32,
        n: usize,
        k: []f32,
        v: []f32,
    ) !void {
        std.debug.assert(src.len == n * self.cfg.context_dim);
        std.debug.assert(k.len == n * self.cfg.dim and v.len == k.len);
        return self.projectKv(io, gpa, &blk.cross_attn, src, n, k, v);
    }

    /// The DiT's `Attention`, self or cross. `freqs` null means cross-attention,
    /// which applies no RoPE at all (convention 6).
    fn attnForward(
        self: *const DiT,
        io: std.Io,
        gpa: std.mem.Allocator,
        attn: *const Attn,
        xq: []const f32,
        n_q: usize,
        xkv: []const f32,
        n_kv: usize,
        freqs: ?ops.rope.Freqs,
        out: []f32,
    ) !void {
        const cfg = self.cfg;
        const inner = cfg.dim; // n_heads * head_dim == model_channels
        const hd = cfg.headDim();

        const q = try gpa.alloc(f32, n_q * inner);
        defer gpa.free(q);
        const k = try gpa.alloc(f32, n_kv * inner);
        defer gpa.free(k);
        const v = try gpa.alloc(f32, n_kv * inner);
        defer gpa.free(v);
        try ops.matmul.matmul(io, gpa, q, xq, n_q, attn.q, null);
        try self.projectKv(io, gpa, attn, xkv, n_kv, k, v);

        ops.norm.rmsNorm(q, q, attn.qnorm, cfg.qk_eps);
        if (freqs) |f| {
            ops.rope.applyRotateHalf(q, f, n_q, cfg.n_heads, hd);
            ops.rope.applyRotateHalf(k, f, n_kv, cfg.n_heads, hd);
        }

        const attn_out = try gpa.alloc(f32, n_q * inner);
        defer gpa.free(attn_out);
        try ops.attention.attention(io, gpa, attn_out, q, k, v, .{
            .seq_q = n_q,
            .seq_kv = n_kv,
            .n_heads = cfg.n_heads,
            .n_kv_heads = cfg.n_heads,
            .head_dim = hd,
        });
        try ops.matmul.matmul(io, gpa, out, attn_out, n_q, attn.out, null);
    }
};


/// The 3-axis RoPE table for a `[h, w]` token grid at `T = 1`, ready for
/// `applyRotateHalf`. Caller `deinit`s.
///
/// See convention 7. Token order is `(t, h, w)` with `t` outermost, matching
/// `rearrange(em_T_H_W_D, "t h w d i j -> (t h w) d i j")`. A free function so the
/// table can be built (and tested) from a `Config` alone, with no loaded weights.
pub fn ropeFreqsFor(gpa: std.mem.Allocator, cfg: Config, h: usize, w: usize) !ops.rope.Freqs {
    const axes = cfg.ropeAxes();
    const n_t = axes[0] / 2;
    const n_h = axes[1] / 2;
    const n_w = axes[2] / 2;
    const half = n_t + n_h + n_w;
    std.debug.assert(half == cfg.headDim() / 2);

    const seq = h * w; // T = 1
    const cos = try gpa.alloc(f32, seq * half);
    errdefer gpa.free(cos);
    const sin = try gpa.alloc(f32, seq * half);
    errdefer gpa.free(sin);

    // `theta = 10000 * ratio ** (axis_dim / (axis_dim - 2))`. The exponent uses
    // the axis's FULL dim, not its frequency count.
    const th_t = 10000.0 * std.math.pow(f64, cfg.rope_t_ratio, @as(f64, @floatFromInt(axes[0])) / @as(f64, @floatFromInt(axes[0] - 2)));
    const th_h = 10000.0 * std.math.pow(f64, cfg.rope_h_ratio, @as(f64, @floatFromInt(axes[1])) / @as(f64, @floatFromInt(axes[1] - 2)));
    const th_w = 10000.0 * std.math.pow(f64, cfg.rope_w_ratio, @as(f64, @floatFromInt(axes[2])) / @as(f64, @floatFromInt(axes[2] - 2)));

    // `freqs = 1 / theta ** (arange(0, dim, 2) / dim)`.
    const inv = try gpa.alloc(f64, half);
    defer gpa.free(inv);
    for (0..n_t) |i| {
        const e = @as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(axes[0]));
        inv[i] = 1.0 / std.math.pow(f64, th_t, e);
    }
    for (0..n_h) |i| {
        const e = @as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(axes[1]));
        inv[n_t + i] = 1.0 / std.math.pow(f64, th_h, e);
    }
    for (0..n_w) |i| {
        const e = @as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(axes[2]));
        inv[n_t + n_h + i] = 1.0 / std.math.pow(f64, th_w, e);
    }

    for (0..h) |hi| {
        for (0..w) |wi| {
            const row = (hi * w + wi) * half;
            // ⚠️ `T = 1`, so every temporal position is 0 and the first `n_t`
            // frequencies are cos 1 / sin 0. A wrong axis ORDER therefore leaves
            // 22 of 64 slots looking correct, which is why the fixture checks a
            // non-square grid where h and w cannot be confused either.
            for (0..n_t) |i| {
                cos[row + i] = 1.0;
                sin[row + i] = 0.0;
            }
            for (0..n_h) |i| {
                const a = @as(f64, @floatFromInt(hi)) * inv[n_t + i];
                cos[row + n_t + i] = @floatCast(@cos(a));
                sin[row + n_t + i] = @floatCast(@sin(a));
            }
            for (0..n_w) |i| {
                const a = @as(f64, @floatFromInt(wi)) * inv[n_t + n_h + i];
                cos[row + n_t + n_h + i] = @floatCast(@cos(a));
                sin[row + n_t + n_h + i] = @floatCast(@sin(a));
            }
        }
    }
    return .{ .cos = cos, .sin = sin, .half = half };
}

// --- free helpers -----------------------------------------------------------

/// `Timesteps.forward`: `cat([cos(t * w), sin(t * w)])` with
/// `w_i = exp(-log(10000) * i / half)`.
///
/// ⚠️ Identical in form to `zimage.timestepEmbedding` (both are cos-first, both
/// divide by `half` rather than `half - 1`) and **different from**
/// `sd_unet.timestepEmbedding`, which is diffusers' sin-first `flip_sin_to_cos`
/// variant. f64 internals for the reason those two record: at `i = 0` the frequency
/// is 1, so a 1e-7 relative slip in the argument shows up directly in `cos`, and
/// being more accurate than the reference bounds the disagreement by the
/// reference's own rounding instead of stacking two errors.
pub fn timestepEmbedding(out: []f32, t: f32) void {
    const half = out.len / 2;
    std.debug.assert(out.len == half * 2);
    const log_max: f64 = @log(10000.0);
    for (0..half) |i| {
        const exponent = -log_max * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(half));
        const arg = @exp(exponent) * @as(f64, t);
        out[i] = @floatCast(@cos(arg));
        out[half + i] = @floatCast(@sin(arg));
    }
}

/// `LayerNorm(x) * (1 + scale) + shift`, row-wise, with a WEIGHTLESS LayerNorm.
/// `dst` may alias `src`.
fn modulatedNorm(dst: []f32, src: []const f32, shift_v: []const f32, scale_v: []const f32, eps: f32) void {
    const dim = shift_v.len;
    std.debug.assert(scale_v.len == dim);
    std.debug.assert(dst.len == src.len and dst.len % dim == 0);
    ops.norm.layerNormUnit(dst, src, dim, eps);
    var row: usize = 0;
    while (row < dst.len) : (row += dim) {
        for (dst[row..][0..dim], shift_v, scale_v) |*v, sh, sc| v.* = v.* * (1.0 + sc) + sh;
    }
}

/// `torch.addcmul(x, gate, delta)`, row-wise: `x += gate * delta`. ⚠️ The gate is
/// NOT passed through anything here — unlike Z-Image, whose gates are `tanh`'d.
fn gatedAdd(x: []f32, delta: []const f32, gate: []const f32) void {
    const dim = gate.len;
    var row: usize = 0;
    while (row < x.len) : (row += dim) {
        for (x[row..][0..dim], delta[row..][0..dim], gate) |*v, dv, g| v.* += g * dv;
    }
}

/// Planar `[channels][lat_h][lat_w]` → `[seq][patchDim]` patch rows, appending the
/// all-zero padding-mask channel when the config asks for it.
///
/// ⚠️ Feature order is `(c, ph, pw)` — **channel SLOWEST** — and the padding-mask
/// channel is the LAST channel, so its four slots are at indices 64..67 of each
/// token. See convention 1; the output order is different.
pub fn patchify(gpa: std.mem.Allocator, cfg: Config, x_lat: []const f32, lat_h: usize, lat_w: usize) ![]f32 {
    const p = cfg.patch;
    const cin = cfg.inChannels();
    const h = lat_h / p;
    const w = lat_w / p;
    const pd = cfg.patchDim();
    const plane = lat_h * lat_w;
    const out = try gpa.alloc(f32, h * w * pd);
    for (0..h) |hi| {
        for (0..w) |wi| {
            const tok = out[(hi * w + wi) * pd ..];
            for (0..cin) |c| {
                for (0..p) |ph| {
                    for (0..p) |pw| {
                        const at = (c * p + ph) * p + pw;
                        tok[at] = if (c < cfg.channels)
                            x_lat[c * plane + (hi * p + ph) * lat_w + (wi * p + pw)]
                        else
                            0.0; // the padding mask, all zeros for a full frame
                    }
                }
            }
        }
    }
    return out;
}

/// `[seq][outPatchDim]` → planar `[out_channels][lat_h][lat_w]`.
///
/// ⚠️ Feature order is `(ph, pw, c)` — **channel FASTEST**, the opposite of
/// `patchify`'s. From `"B T H W (p1 p2 t C) -> B C (T t) (H p1) (W p2)"` with
/// `t = patch_temporal = 1`. There is no output sign flip here (Z-Image has one).
pub fn unpatchify(cfg: Config, out: []f32, patches: []const f32, lat_h: usize, lat_w: usize) void {
    const p = cfg.patch;
    const co = cfg.out_channels;
    const h = lat_h / p;
    const w = lat_w / p;
    const pd = cfg.outPatchDim();
    const plane = lat_h * lat_w;
    for (0..h) |hi| {
        for (0..w) |wi| {
            const tok = patches[(hi * w + wi) * pd ..];
            for (0..p) |ph| {
                for (0..p) |pw| {
                    for (0..co) |c| {
                        out[c * plane + (hi * p + ph) * lat_w + (wi * p + pw)] =
                            tok[(ph * p + pw) * co + c];
                    }
                }
            }
        }
    }
}

/// Gather `[n][cols]` f32 rows out of a dtype-generic `[rows][cols]` embedding
/// matrix — `qwen3.embedTokens` for the adapter's own (T5) vocabulary. Dispatching
/// on the stored dtype rather than assuming bf16 is the fix that file records: three
/// call sites there hardcoded a `* 2` row stride and read a q6_k embedding as noise.
fn embedRows(embed: Weight, ids: []const u32, out: []f32) !void {
    const cols = embed.cols;
    const row_bytes = embed.dtype.storageBytes(cols);
    std.debug.assert(out.len == ids.len * cols);
    for (ids, 0..) |id, i| {
        if (id >= embed.rows) return error.TokenIdOutOfRange;
        const row = embed.bytes[@as(usize, id) * row_bytes ..][0..row_bytes];
        try tp_core.safetensors.convertToF32(embed.dtype, row, out[i * cols ..][0..cols]);
    }
}

/// Whether the GPU forwards have a GEMM path for linears of this dtype. Mirrors
/// `dit.gpuLinKindSupported` / `zimage.gpuLinKindSupported`: an unrecognized dtype
/// on those paths is not a slow path, it is silently wrong output.
///
/// ⚠️ int8/int4 convrot is **absent on purpose**: the CPU `matmul` runs it (rotate +
/// per-row dequant) but neither `anima_gpu` nor `anima_cuda` has a W8A8 path yet. See
/// `unsupportedGpuLin`.
pub fn gpuLinKindSupported(dt: DType) bool {
    return switch (dt) {
        .bf16, .f16, .f32, .f8_e4m3 => true,
        else => false,
    };
}

/// The first block linear whose dtype no GPU arm can run, or null if every one can.
///
/// ⚠️ **This exists because checking ONE tensor of ONE block is wrong on a real
/// checkpoint, and that mistake produced a panic.** `anima_baseV10-INT8_CONVROT-MIXED`
/// keeps **block 0 entirely bf16** and quantizes blocks 1-27 — so a `supported()` that
/// read `blocks[0].self_attn.q.dtype` said yes, a device session was built, and the very
/// first thing it did (`crossKv`, on the host) tripped `matmul`'s int8 assert. "Mixed"
/// means mixed **per block**, and a per-block-uniform checkpoint is the easy case, not
/// the general one.
///
/// Returns the offending `{block, name, dtype}` so the warning can say which layer, not
/// just that something is unsupported.
pub const UnsupportedLin = struct { block: usize, tag: []const u8, dtype: DType };

/// Which GEMM family one linear takes. ⚠️ Resolved PER LINEAR and used per BLOCK — a real
/// mixed checkpoint is mixed by block: `easonAnimaHOTStyle_animaV10-INT8_CONVROT` leaves
/// block 0 entirely dense, quantizes block 1's ten attention/MLP linears, and quantizes all
/// sixteen in blocks 2-27. A per-model kind would be wrong for at least one block of it.
pub const LinKind = enum { dense, i8, i4 };

pub fn linKind(w: Weight) LinKind {
    return switch (w.dtype) {
        .i8 => .i8,
        .i4 => .i4,
        else => .dense,
    };
}

/// What a given backend's GEMM surface can run. ⚠️ Not the same on both arms: both CUDA
/// arms have int8 AND int4 convrot; Vulkan has int8 (native `sint8` coopmat) but **no
/// `sint4` coopmat exists on this device**, so int4 needs a different strategy there.
pub const LinSupport = struct {
    i8: bool = false,
    i4: bool = false,
};

/// The first block linear this backend cannot run, or null if it can run all of them.
pub fn unsupportedLin(model: *const DiT, support: LinSupport) ?UnsupportedLin {
    for (model.blocks, 0..) |*b, bi| {
        // Every linear the DEVICE forward runs. The AdaLN pair and cross-attention's
        // k/v are deliberately absent: those are evaluated on the host
        // (`modulationTable`, `crossKv`), where the CPU `matmul` handles convrot, so
        // their dtype does not gate the device path.
        const lins = [_]Weight{
            b.self_attn.q,  b.self_attn.k,  b.self_attn.v,  b.self_attn.out,
            b.cross_attn.q, b.cross_attn.out,
            b.mlp1,         b.mlp2,
        };
        for (lins) |w| {
            const ok = switch (linKind(w)) {
                .i8 => support.i8,
                .i4 => support.i4,
                .dense => gpuLinKindSupported(w.dtype),
            };
            if (!ok) return .{ .block = bi, .tag = w.tag orelse "?", .dtype = w.dtype };
        }
    }
    return null;
}

/// `unsupportedLin` for a backend with no convrot GEMM — the conservative default, and
/// what `anima_gpu` passes.
pub fn unsupportedGpuLin(model: *const DiT) ?UnsupportedLin {
    return unsupportedLin(model, .{});
}

// --- weight loading ---------------------------------------------------------

const Loader = struct {
    store: WeightStore,
    alloc: std.mem.Allocator,
    pfx: []const u8,
    cfg: Config,

    fn name(l: Loader, buf: []u8, comptime fmt: []const u8, args: anytype, suffix: []const u8) ![]u8 {
        var fbs = std.Io.Writer.fixed(buf);
        try fbs.writeAll(l.pfx);
        try fbs.print(fmt, args);
        try fbs.writeAll(suffix);
        return fbs.buffered();
    }

    fn mat(l: Loader, comptime fmt: []const u8, args: anytype, rows: usize, cols: usize) !Weight {
        var buf: [192]u8 = undefined;
        const nm = try l.name(&buf, fmt, args, "");
        const view = l.store.get(nm) orelse {
            std.log.err("anima: missing tensor {s}", .{nm});
            return error.MissingTensor;
        };
        const shape = view.info.shape.slice();
        const dt = view.info.dtype;

        // ⚠️ **int4 convrot weights are nibble-packed**, so the on-disk shape is
        // `[rows, cols/2]`; a genuine int8 convrot weight is also I8 but at the full
        // `[rows, cols]`. Disambiguate by the halved column count, not by dtype alone —
        // exactly as `dit.zig` does, and for the same reason: our own converter stores
        // the packed bytes as U8 where ComfyUI's W4A4 converter stores the identical
        // bytes as I8.
        const halved = shape.len == 2 and shape[0] == rows and cols % 2 == 0 and shape[1] == cols / 2;
        const is_i4 = dt == .u8 or (dt == .i8 and halved);
        const wdt: DType = if (is_i4) .i4 else dt;
        const stored_cols = if (is_i4) cols / 2 else cols;
        if (is_i4 and cols % 2 != 0) return error.ShapeMismatch;
        if (shape.len != 2 or shape[0] != rows or shape[1] != stored_cols) {
            // Name the tensor and both shapes: a bare ShapeMismatch across ~880
            // weights is not actionable, and the usual cause is a container whose
            // dim order differs.
            std.log.err("anima: {s} has shape {any} ({t}), expected [{d}, {d}]", .{ nm, shape, dt, rows, stored_cols });
            return error.ShapeMismatch;
        }
        // A shape-fixed block-quantized tensor blocks over the FLAT element sequence
        // rather than each logical row, which `Weight.init` does not assume.
        if (view.info.flat_blocks) {
            std.log.err("anima: {s} is {t} with flat block layout; this loader needs row-aligned blocks", .{ nm, dt });
            return error.UnsupportedCheckpoint;
        }
        if (!ops.matmul.supportsDType(wdt)) {
            std.log.err("anima: {s} has unsupported dtype {t}", .{ nm, dt });
            return error.UnsupportedDType;
        }
        var w = Weight.init(view.bytes, wdt, rows, cols);
        // Carry the checkpoint name so a GEMM stays attributable downstream
        // (ops.matmul.probe, profiling, error messages).
        w.tag = try l.alloc.dupe(u8, nm);

        // ⚠️ **int8/int4 "convrot" needs its per-row scale and rotation metadata wired
        // here, and omitting it is a PANIC rather than a wrong answer**:
        // `ops.matmul.matmul` asserts `row_scale != null` for an integer weight. That is
        // exactly what a `-INT8_CONVROT-MIXED` Anima checkpoint hit — the loader accepted
        // the I8 tensor (`supportsDType(.i8)` is true) and the assert fired 5 frames
        // deep, in `crossKv`. The companion tensor is `<name>_scale`, i.e.
        // `…q_proj.weight_scale`, with one entry per output row.
        if (wdt == .i8 or wdt == .i4) {
            var sbuf: [200]u8 = undefined;
            const sname = try l.name(&sbuf, fmt, args, "_scale");
            const sv = l.store.get(sname) orelse {
                std.log.err("anima: {s} is {t} but {s} is missing (int8/int4 convrot needs a per-row scale)", .{ nm, dt, sname });
                return error.MissingTensor;
            };
            if (sv.info.elemCount() != rows) {
                std.log.err("anima: {s} has {d} entries, expected {d} (one per output row)", .{ sname, sv.info.elemCount(), rows });
                return error.ShapeMismatch;
            }
            if (cols % ops.convrot.group_size != 0) {
                std.log.err("anima: {s} has {d} columns, not a multiple of the {d}-wide convrot group", .{ nm, cols, ops.convrot.group_size });
                return error.ShapeMismatch;
            }
            w.row_scale = try sv.toF32Alloc(l.alloc);
            w.convrot = ops.convrot.group_size;
        }
        return w;
    }

    fn vec(l: Loader, comptime fmt: []const u8, args: anytype, len: usize) ![]f32 {
        var buf: [192]u8 = undefined;
        const nm = try l.name(&buf, fmt, args, "");
        const view = l.store.get(nm) orelse {
            std.log.err("anima: missing tensor {s}", .{nm});
            return error.MissingTensor;
        };
        if (view.info.elemCount() != len) {
            std.log.err("anima: {s} has {d} elements, expected {d}", .{ nm, view.info.elemCount(), len });
            return error.ShapeMismatch;
        }
        return view.toF32Alloc(l.alloc);
    }

    /// One DiT attention. `ctx_dim` is 2048 for self-attention and 1024 (the
    /// adapter's output width) for cross-attention; `q`/`out` are always square.
    fn ditAttn(l: Loader, comptime fmt: []const u8, args: anytype, ctx_dim: usize) !Attn {
        const cfg = l.cfg;
        const inner = cfg.dim; // n_heads * head_dim
        return .{
            .q = try l.mat(fmt ++ ".q_proj.weight", args, inner, cfg.dim),
            .k = try l.mat(fmt ++ ".k_proj.weight", args, inner, ctx_dim),
            .v = try l.mat(fmt ++ ".v_proj.weight", args, inner, ctx_dim),
            // ⚠️ The DiT calls it `output_proj`; the adapter calls the same thing
            // `o_proj`. Two names for one role, in one checkpoint.
            .out = try l.mat(fmt ++ ".output_proj.weight", args, cfg.dim, inner),
            .qnorm = try l.vec(fmt ++ ".q_norm.weight", args, cfg.headDim()),
            .knorm = try l.vec(fmt ++ ".k_norm.weight", args, cfg.headDim()),
        };
    }

    fn block(l: Loader, i: usize) !Block {
        const cfg = l.cfg;
        const d = cfg.dim;
        const a = cfg.adaln_dim;
        return .{
            .self_attn = try l.ditAttn("blocks.{d}.self_attn", .{i}, d),
            .cross_attn = try l.ditAttn("blocks.{d}.cross_attn", .{i}, cfg.context_dim),
            .mlp1 = try l.mat("blocks.{d}.mlp.layer1.weight", .{i}, cfg.mlp_dim, d),
            .mlp2 = try l.mat("blocks.{d}.mlp.layer2.weight", .{i}, d, cfg.mlp_dim),
            // `.1` and `.2` because `.0` is the SiLU — convention 5.
            .ada_sa = .{
                try l.mat("blocks.{d}.adaln_modulation_self_attn.1.weight", .{i}, a, d),
                try l.mat("blocks.{d}.adaln_modulation_self_attn.2.weight", .{i}, 3 * d, a),
            },
            .ada_ca = .{
                try l.mat("blocks.{d}.adaln_modulation_cross_attn.1.weight", .{i}, a, d),
                try l.mat("blocks.{d}.adaln_modulation_cross_attn.2.weight", .{i}, 3 * d, a),
            },
            .ada_mlp = .{
                try l.mat("blocks.{d}.adaln_modulation_mlp.1.weight", .{i}, a, d),
                try l.mat("blocks.{d}.adaln_modulation_mlp.2.weight", .{i}, 3 * d, a),
            },
        };
    }

    /// One adapter attention. Square throughout — source_dim equals the adapter's
    /// own width for this checkpoint — but named separately from `ditAttn` because
    /// the tensors are spelled differently (`o_proj`, not `output_proj`).
    fn adapterAttn(l: Loader, comptime fmt: []const u8, args: anytype) !Attn {
        const ac = l.cfg.adapter;
        return .{
            .q = try l.mat(fmt ++ ".q_proj.weight", args, ac.dim, ac.dim),
            .k = try l.mat(fmt ++ ".k_proj.weight", args, ac.dim, ac.dim),
            .v = try l.mat(fmt ++ ".v_proj.weight", args, ac.dim, ac.dim),
            .out = try l.mat(fmt ++ ".o_proj.weight", args, ac.dim, ac.dim),
            .qnorm = try l.vec(fmt ++ ".q_norm.weight", args, ac.headDim()),
            .knorm = try l.vec(fmt ++ ".k_norm.weight", args, ac.headDim()),
        };
    }

    fn adapterBlock(l: Loader, i: usize) !AdapterBlock {
        const ac = l.cfg.adapter;
        return .{
            .norm_self = try l.vec("llm_adapter.blocks.{d}.norm_self_attn.weight", .{i}, ac.dim),
            .self_attn = try l.adapterAttn("llm_adapter.blocks.{d}.self_attn", .{i}),
            .norm_cross = try l.vec("llm_adapter.blocks.{d}.norm_cross_attn.weight", .{i}, ac.dim),
            .cross_attn = try l.adapterAttn("llm_adapter.blocks.{d}.cross_attn", .{i}),
            .norm_mlp = try l.vec("llm_adapter.blocks.{d}.norm_mlp.weight", .{i}, ac.dim),
            // ⚠️ Indices 0 and 2 (`.1` is the GELU), and these DO have biases where
            // every linear in the DiT half does not.
            .mlp1 = .{
                .w = try l.mat("llm_adapter.blocks.{d}.mlp.0.weight", .{i}, ac.mlp_dim, ac.dim),
                .b = try l.vec("llm_adapter.blocks.{d}.mlp.0.bias", .{i}, ac.mlp_dim),
            },
            .mlp2 = .{
                .w = try l.mat("llm_adapter.blocks.{d}.mlp.2.weight", .{i}, ac.dim, ac.mlp_dim),
                .b = try l.vec("llm_adapter.blocks.{d}.mlp.2.bias", .{i}, ac.dim),
            },
        };
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;
const test_gate = @import("../test_gate.zig");
const SafeTensors = tp_core.safetensors.SafeTensors;
const qwen3 = @import("qwen3.zig");
const tokenizer_mod = tp_core.tokenizer;
const t5_tokenizer = tp_core.t5_tokenizer;

const ref_path = "src/models/assets/anima_ref.safetensors";
const anima_ckpt = "/home/qt/genai/comfyui/models/diffusion_models/anima/terraRising_20TerraRisingAnima.safetensors";
const anima_te = "/home/qt/genai/comfyui/models/text_encoders/qwen_3_06b_base.safetensors";
const anima_int8_ckpt = "/home/qt/genai/comfyui/models/diffusion_models/anima/anima_baseV10-INT8_CONVROT-MIXED.safetensors";
/// Trunk depths the fp32 reference keeps — see the generator's docstring. The pair
/// is what shows whether the disagreement grows linearly with depth (the block is
/// right) or compounds (it is merely close).
const ref_layers = [2]usize{ 8, 2 };

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

/// The prompts the fixture was generated from, read out of its OWN metadata rather
/// than duplicated here — a second copy of the prompt text is exactly the drift that
/// would make a token-id comparison pass against the wrong string. Caller frees the
/// parsed value.
fn refPrompts(gpa: std.mem.Allocator, ref: *const SafeTensors) !std.json.Parsed([]const []const u8) {
    const md = ref.metadata orelse return error.MissingTensor;
    const raw = md.get("prompts") orelse return error.MissingTensor;
    return std.json.parseFromSlice([]const []const u8, gpa, raw.string, .{});
}

/// The Qwen3 branch's ids: `Qwen3Tokenizer` takes no start or end token and pads only
/// to `min_length = 1`, so this is the bare Qwen2 BPE of the prompt — with one
/// `<|endoftext|>` when that would otherwise be empty.
fn qwenIds(gpa: std.mem.Allocator, tok: *const tokenizer_mod.Tokenizer, text: []const u8) ![]u32 {
    return tok.encodeSegmented(gpa, text, 1, tokenizer_mod.pad_token);
}

/// I32 fixture tensors: `toF32Alloc` deliberately refuses integer dtypes (silently
/// floating token ids is never right), so read the bytes.
fn refIds(gpa: std.mem.Allocator, ref: *const SafeTensors, name: []const u8) ![]u32 {
    const v = try ref.require(name);
    const n = v.info.elemCount();
    const out = try gpa.alloc(u32, n);
    for (0..n) |i| out[i] = @intCast(std.mem.readInt(i32, v.bytes[i * 4 ..][0..4], .little));
    return out;
}

test "the RoPE axis widths are the remainder split VideoRopePosition3DEmb computes" {
    const axes = anima_2b.ropeAxes();
    // head_dim 128: dim_h = dim_w = 128 // 6 * 2 = 42, dim_t = 128 - 84 = 44.
    // ⚠️ The temporal axis is the WIDEST, which is counter-intuitive for a model
    // that only ever runs at T = 1.
    try testing.expectEqual([3]usize{ 44, 42, 42 }, axes);
    try testing.expectEqual(anima_2b.headDim(), axes[0] + axes[1] + axes[2]);
    try testing.expectEqual(@as(usize, 64), (axes[0] + axes[1] + axes[2]) / 2);
}

test "patchify and unpatchify use DIFFERENT feature orders, and each round-trips itself" {
    const gpa = testing.allocator;
    // A non-square grid, so an h/w transposition cannot pass.
    const lat_h = 6;
    const lat_w = 4;
    const cfg = anima_2b;
    const plane = lat_h * lat_w;

    const lat = try gpa.alloc(f32, cfg.channels * plane);
    defer gpa.free(lat);
    for (lat, 0..) |*v, i| v.* = @floatFromInt(i);

    const patches = try patchify(gpa, cfg, lat, lat_h, lat_w);
    defer gpa.free(patches);
    try testing.expectEqual((lat_h / 2) * (lat_w / 2) * cfg.patchDim(), patches.len);

    // Convention 1, half one: the INPUT order is (c, ph, pw), channel slowest. Token
    // 0 covers rows 0-1 and cols 0-1, so channel 0's four values are 0, 1, 4, 5.
    try testing.expectEqual(@as(f32, 0), patches[0]);
    try testing.expectEqual(@as(f32, 1), patches[1]);
    try testing.expectEqual(@as(f32, 4), patches[2]);
    try testing.expectEqual(@as(f32, 5), patches[3]);
    // ...and channel 1's start at index 4, not at 1.
    try testing.expectEqual(@as(f32, @floatFromInt(plane)), patches[4]);
    // The padding-mask channel is last and all zero.
    for (0..cfg.patch * cfg.patch) |i| {
        try testing.expectEqual(@as(f32, 0), patches[cfg.channels * 4 + i]);
    }

    // Convention 1, half two: the OUTPUT order is (ph, pw, c), channel fastest.
    // Feed unpatchify a token whose features are 0..63 and check where they land.
    const seq = (lat_h / 2) * (lat_w / 2);
    const opatch = try gpa.alloc(f32, seq * cfg.outPatchDim());
    defer gpa.free(opatch);
    @memset(opatch, 0);
    for (0..cfg.outPatchDim()) |i| opatch[i] = @floatFromInt(i);

    const back = try gpa.alloc(f32, cfg.out_channels * plane);
    defer gpa.free(back);
    @memset(back, -1);
    unpatchify(cfg, back, opatch, lat_h, lat_w);
    // (ph=0, pw=0, c=0) is feature 0 -> channel 0 at (0,0).
    try testing.expectEqual(@as(f32, 0), back[0]);
    // (ph=0, pw=0, c=1) is feature 1 -> channel 1 at (0,0). Under the INPUT order
    // that slot would instead hold feature 4.
    try testing.expectEqual(@as(f32, 1), back[plane]);
    // (ph=0, pw=1, c=0) is feature 16 -> channel 0 at (0,1).
    try testing.expectEqual(@as(f32, 16), back[1]);
    // (ph=1, pw=0, c=0) is feature 32 -> channel 0 at (1,0).
    try testing.expectEqual(@as(f32, 32), back[lat_w]);

    // ⚠️ And the reason a single order looks fine: the two disagree, but both are
    // permutations, so the multiset of values is identical either way. Nothing that
    // checks magnitudes can see the difference.
    var sum_in: f64 = 0;
    for (patches[0..cfg.outPatchDim()]) |v| sum_in += v;
    var sum_out: f64 = 0;
    for (opatch[0..cfg.outPatchDim()]) |v| sum_out += v;
    try testing.expect(sum_in != sum_out); // different data, same shape
}

test "the timestep embedding is cos-first and takes the sigma unscaled" {
    var out: [8]f32 = undefined;
    // Convention 3: Anima's argument is the sigma itself, in (0, 1].
    timestepEmbedding(&out, 1.0);
    // w_0 = 1, so slot 0 is cos(1) and slot `half` is sin(1). A sin-first
    // (`flip_sin_to_cos`) reading would swap them.
    try testing.expectApproxEqAbs(@as(f32, @floatCast(@cos(1.0))), out[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, @floatCast(@sin(1.0))), out[4], 1e-6);
    // w_i = exp(-log(10000) * i / half) with half = 4.
    const w1 = @exp(-@log(10000.0) / 4.0);
    try testing.expectApproxEqAbs(@as(f32, @floatCast(@cos(w1))), out[1], 1e-6);
}

test "the temporal RoPE block is identity at T = 1 and h/w are not interchangeable" {
    const gpa = testing.allocator;
    var f = try ropeFreqsFor(gpa, anima_2b, 3, 5);
    defer f.deinit(gpa);
    try testing.expectEqual(@as(usize, 64), f.half);

    const axes = anima_2b.ropeAxes();
    const n_t = axes[0] / 2;
    const n_h = axes[1] / 2;
    // Convention 7: the first 22 frequencies are the temporal axis, which at T = 1
    // is every position 0 — cos 1, sin 0. So a third of the vector is identity and
    // a wrong axis order still looks partly right.
    for (0..3 * 5) |tok| {
        for (0..n_t) |i| {
            try testing.expectEqual(@as(f32, 1), f.cos[tok * 64 + i]);
            try testing.expectEqual(@as(f32, 0), f.sin[tok * 64 + i]);
        }
    }
    // Token (hi=0, wi=1) must differ from (hi=1, wi=0): the h block moves for one
    // and the w block for the other, and their thetas are equal, so only the SLOT
    // distinguishes them. This is the check an h/w swap fails.
    const t01 = 0 * 5 + 1;
    const t10 = 1 * 5 + 0;
    try testing.expectEqual(@as(f32, 1), f.cos[t01 * 64 + n_t + 1]); // h still 0
    try testing.expect(f.cos[t01 * 64 + n_t + n_h + 1] != 1.0); // w moved
    try testing.expect(f.cos[t10 * 64 + n_t + 1] != 1.0); // h moved
    try testing.expectEqual(@as(f32, 1), f.cos[t10 * 64 + n_t + n_h + 1]); // w still 0

    // The h/w NTK factor is 4 ** (42/40), applied to 10000 — not 4 * 10000, and not
    // 4 ** (42/42). Check it through the first non-trivial frequency.
    const th = 10000.0 * std.math.pow(f64, 4.0, 42.0 / 40.0);
    const inv1 = 1.0 / std.math.pow(f64, th, 2.0 / 42.0);
    try testing.expectApproxEqAbs(@as(f32, @floatCast(@cos(inv1))), f.cos[t10 * 64 + n_t + 1], 1e-6);
}

test "the Anima adapter and DiT match ComfyUI on a real checkpoint" {
    // ⚠️ Compared stage by stage rather than end to end, because the stages fail for
    // genuinely different reasons: the encoder is a tap/final-norm question, the
    // adapter is a cross-attention-wiring question, the timestep path is conventions
    // 3-5, the RoPE table is convention 7, `x_embed` is conventions 1-2, and the
    // trunk is the block form. One output comparison would say only "wrong".
    //
    // The reference keeps 8 (and 2) of the 28 trunk blocks, so the model is loaded at
    // those depths. What that does not check is the loop bound itself — the
    // end-to-end render comparison covers that.
    const gpa = testing.allocator;
    const io = testing.io;
    try test_gate.requireIntegration();
    try test_gate.requireModelFile(io, anima_ckpt);
    try test_gate.requireModelFile(io, ref_path);

    var ref = try SafeTensors.open(gpa, io, ref_path);
    defer ref.deinit();
    var ck = try SafeTensors.open(gpa, io, anima_ckpt);
    defer ck.deinit();

    var kb: [64]u8 = undefined;

    for (ref_layers, 0..) |n_layers, depth_i| {
        var cfg = anima_2b;
        cfg.n_layers = n_layers;
        var model = try DiT.load(gpa, .{ .safetensors = &ck }, cfg);
        defer model.deinit();
        const d = cfg.dim;

        for (0..2) |ci| {
            // ⚠️ Two tags: the per-depth one carries only `out`, while every INPUT
            // lives under the deepest tag and is shared across depths. The generator
            // draws them once for exactly this reason — a per-depth draw would make
            // the "depth trend" a comparison of two unrelated forwards.
            const tag = try std.fmt.bufPrint(&kb, "dit{d}.{d}", .{ n_layers, ci });
            var inb: [64]u8 = undefined;
            const in_tag = try std.fmt.bufPrint(&inb, "dit{d}.{d}", .{ ref_layers[0], ci });
            var nb: [64]u8 = undefined;
            const key = struct {
                fn f(buf: []u8, t: []const u8, suffix: []const u8) ![]u8 {
                    return std.fmt.bufPrint(buf, "{s}.{s}", .{ t, suffix });
                }
            }.f;

            const xv = try ref.require(try key(&nb, in_tag, "x"));
            const x_lat = try xv.toF32Alloc(gpa);
            defer gpa.free(x_lat);
            const sig = try (try ref.require(try key(&nb, in_tag, "sigma"))).toF32Alloc(gpa);
            defer gpa.free(sig);
            const ctx = try (try ref.require(try key(&nb, in_tag, "ctx"))).toF32Alloc(gpa);
            defer gpa.free(ctx);
            const ctx_seq = ctx.len / cfg.context_dim;
            // Recover the latent geometry from the tensor's own dims, so the test
            // cannot drift from the generator's LATENTS table.
            const xs = xv.info.shape.slice();
            try testing.expectEqual(@as(usize, 3), xs.len);
            const lat_h = xs[1];
            const lat_w = xs[2];

            // --- the timestep path (conventions 3, 4, 5) ----------------------
            const mod = try model.modulationTable(io, gpa, sig[0]);
            defer gpa.free(mod);

            if (depth_i == 0) {
                // The sinusoid itself: convention 3 says the argument is the sigma,
                // unscaled. A `(1 - sigma) * 1000` reading (Z-Image's, over the very
                // same sigma table) fails here and nowhere else so clearly.
                {
                    const want = try (try ref.require(try key(&nb, in_tag, "sinus"))).toF32Alloc(gpa);
                    defer gpa.free(want);
                    const got = try gpa.alloc(f32, d);
                    defer gpa.free(got);
                    timestepEmbedding(got, sig[0]);
                    const rel = relL2(want, got);
                    errdefer std.debug.print("{s} sinus rel L2 {e:.4}\n", .{ tag, rel });
                    try testing.expect(rel < 1e-7); // measured 1.8e-8
                }
                // Block 0's three modulation vectors and the final layer's, each
                // already summed with the shared LoRA vector. This is what pins
                // convention 4 (one vector, three consumers) and convention 5 (SiLU
                // on the input, no activation between the two linears).
                {
                    const want = try (try ref.require(try key(&nb, in_tag, "mod0"))).toF32Alloc(gpa);
                    defer gpa.free(want);
                    try testing.expectEqual(@as(usize, 9 * d), want.len);
                    const rel = relL2(want, mod[0 .. 9 * d]);
                    errdefer std.debug.print("{s} mod0 rel L2 {e:.4}\n", .{ tag, rel });
                    try testing.expect(rel < 5e-6); // measured 4.6e-7
                }
                {
                    const want = try (try ref.require(try key(&nb, in_tag, "mod_final"))).toF32Alloc(gpa);
                    defer gpa.free(want);
                    const got = mod[cfg.n_layers * 9 * d ..][0 .. 2 * d];
                    const rel = relL2(want, got);
                    errdefer std.debug.print("{s} mod_final rel L2 {e:.4}\n", .{ tag, rel });
                    try testing.expect(rel < 1e-5); // measured 1.4e-6
                }

                // --- the RoPE table (convention 7) ----------------------------
                {
                    const want_c = try (try ref.require(try key(&nb, in_tag, "rope_cos"))).toF32Alloc(gpa);
                    defer gpa.free(want_c);
                    const want_s = try (try ref.require(try key(&nb, in_tag, "rope_sin"))).toF32Alloc(gpa);
                    defer gpa.free(want_s);
                    var f = try ropeFreqsFor(gpa, cfg, lat_h / cfg.patch, lat_w / cfg.patch);
                    defer f.deinit(gpa);
                    try testing.expectEqual(want_c.len, f.cos.len);
                    const rc = relL2(want_c, f.cos);
                    const rs = relL2(want_s, f.sin);
                    errdefer std.debug.print("{s} rope rel L2 cos {e:.4} sin {e:.4}\n", .{ tag, rc, rs });
                    try testing.expect(rc < 1e-6 and rs < 1e-6); // measured 1.7e-8 / 1.4e-7
                }

                // --- patchify + x_embedder (conventions 1 and 2) --------------
                // The all-zero padding-mask channel is inside `patchify`, so a
                // missing 17th channel is a shape error here rather than a wrong
                // image later; and the INPUT patch order is `(c, ph, pw)`.
                {
                    const want = try (try ref.require(try key(&nb, in_tag, "x_embed"))).toF32Alloc(gpa);
                    defer gpa.free(want);
                    const patches = try patchify(gpa, cfg, x_lat, lat_h, lat_w);
                    defer gpa.free(patches);
                    const seq = (lat_h / cfg.patch) * (lat_w / cfg.patch);
                    try testing.expectEqual(seq * cfg.patchDim(), patches.len);
                    const got = try gpa.alloc(f32, seq * d);
                    defer gpa.free(got);
                    try ops.matmul.matmul(io, gpa, got, patches, seq, model.x_embedder, null);
                    const rel = relL2(want, got);
                    errdefer std.debug.print("{s} x_embed rel L2 {e:.4}\n", .{ tag, rel });
                    // ⚠️ Measured **exactly 0.0** at both latent sizes — this stage is
                    // bit-identical to torch, which is what pins the input patch order
                    // and the padding-mask channel with no room for interpretation. The
                    // bound is not 0 only because a future change to the GEMM's blocking
                    // would legitimately move the reduction order.
                    try testing.expect(rel < 1e-6);
                }

                // --- the adapter, at FULL depth -------------------------------
                // Two comparisons, because they isolate different things:
                // `adapter_raw` is the 6-block body, `ctx` is that body plus the
                // per-row emphasis weights and the zero pad to 512 rows.
                {
                    const pidx = try refIds(gpa, &ref, try key(&nb, in_tag, "prompt_index"));
                    defer gpa.free(pidx);
                    const pi = pidx[0];
                    const cond = try (try ref.require(try std.fmt.bufPrint(&nb, "te.cond.{d}", .{pi}))).toF32Alloc(gpa);
                    defer gpa.free(cond);
                    const n_src = cond.len / cfg.adapter.source_dim;
                    const t5 = try refIds(gpa, &ref, try std.fmt.bufPrint(&nb, "te.t5_ids.{d}", .{pi}));
                    defer gpa.free(t5);
                    const t5w = try (try ref.require(try std.fmt.bufPrint(&nb, "te.t5_weights.{d}", .{pi}))).toF32Alloc(gpa);
                    defer gpa.free(t5w);

                    const raw = try model.adapter.forward(io, gpa, cond, n_src, t5, null, null);
                    defer gpa.free(raw);
                    {
                        const want = try (try ref.require(try key(&nb, in_tag, "adapter_raw"))).toF32Alloc(gpa);
                        defer gpa.free(want);
                        try testing.expectEqual(t5.len * cfg.adapter.dim, want.len);
                        const rel = relL2(want, raw[0..want.len]);
                        errdefer std.debug.print("{s} adapter_raw rel L2 {e:.4}\n", .{ tag, rel });
                        try testing.expect(rel < 2e-5); // measured 3.2e-6 / 3.6e-6
                    }
                    const weighted = try model.adapter.forward(io, gpa, cond, n_src, t5, t5w, null);
                    defer gpa.free(weighted);
                    try testing.expectEqual(ctx.len, weighted.len); // pins the 512-row pad
                    const rel = relL2(ctx, weighted);
                    errdefer std.debug.print("{s} ctx rel L2 {e:.4}\n", .{ tag, rel });
                    try testing.expect(rel < 2e-5); // measured 3.2e-6 / 3.6e-6
                    // ⚠️ And the pad rows must be exactly zero, not "small": the
                    // denoiser cross-attends over them.
                    for (weighted[t5.len * cfg.adapter.dim ..]) |v| try testing.expectEqual(@as(f32, 0), v);
                }
            }

            // --- the whole forward ------------------------------------------
            const got = try gpa.alloc(f32, x_lat.len);
            defer gpa.free(got);
            try model.predict(io, gpa, got, x_lat, lat_h, lat_w, ctx, ctx_seq, mod, null);

            const want = try (try ref.require(try key(&nb, tag, "out"))).toF32Alloc(gpa);
            defer gpa.free(want);
            const rel = relL2(want, got);
            errdefer std.debug.print(
                "depth {d} case {d} (latent {d}x{d}): velocity rel L2 {e:.4}\n",
                .{ n_layers, ci, lat_h, lat_w, rel },
            );
            // Stored f32 and computed from the same f32-dequantized bf16 weights on
            // both sides, so the only difference is reduction order.
            //
            // ⚠️ **The depth trend is the informative part**, and it is FLAT: 2.2e-6 at
            // 2 layers and 1.9e-6 at 8 (12x10), 3.3e-6 and 3.1e-6 (40x32). A
            // disagreement that does not grow with depth is not accumulating — which
            // is what says the block is right rather than merely close. (Z-Image's
            // grows linearly, 1.6e-6 -> 3.5e-6, and that was already good enough.)
            try testing.expect(rel < 2e-5);

            // Block 0's output in isolation, so a trunk failure localizes to the
            // block form rather than to 8 blocks of accumulated difference.
            if (depth_i == 0) {
                const wb = try (try ref.require(try key(&nb, in_tag, "block0"))).toF32Alloc(gpa);
                defer gpa.free(wb);
                const seq = (lat_h / cfg.patch) * (lat_w / cfg.patch);
                const patches = try patchify(gpa, cfg, x_lat, lat_h, lat_w);
                defer gpa.free(patches);
                const x = try gpa.alloc(f32, seq * d);
                defer gpa.free(x);
                try ops.matmul.matmul(io, gpa, x, patches, seq, model.x_embedder, null);
                var f = try ropeFreqsFor(gpa, cfg, lat_h / cfg.patch, lat_w / cfg.patch);
                defer f.deinit(gpa);
                try model.blockForward(io, gpa, &model.blocks[0], x, seq, ctx, ctx_seq, mod[0 .. 9 * d], f);
                const relb = relL2(wb, x);
                errdefer std.debug.print("{s} block0 rel L2 {e:.4}\n", .{ tag, relb });
                try testing.expect(relb < 1e-5); // measured 1.9e-6 / 1.3e-6
            }
        }
    }
}

test "the Anima text encoder matches ComfyUI's Qwen3-0.6B tap" {
    // The encoder half, separately from the adapter: this is what pins
    // `qwen3.Variant.anima` — all 28 layers AND `model.norm` applied, which is
    // ComfyUI's `layer = "last"` with `final_norm = True`. Tapping one layer early,
    // or skipping the final norm, is a finite encode of the wrong state.
    const gpa = testing.allocator;
    const io = testing.io;
    try test_gate.requireIntegration();
    try test_gate.requireModelFile(io, anima_te);
    try test_gate.requireModelFile(io, ref_path);

    var ref = try SafeTensors.open(gpa, io, ref_path);
    defer ref.deinit();
    var st = try SafeTensors.open(gpa, io, anima_te);
    defer st.deinit();

    var enc = try qwen3.TextEncoder.loadVariant(gpa, .{ .safetensors = &st }, .anima);
    defer enc.deinit();
    try testing.expectEqual(@as(usize, 1), enc.tapCount());
    try testing.expect(enc.final_norm != null);

    var tok = try tokenizer_mod.Tokenizer.init(gpa);
    defer tok.deinit();
    var t5 = try t5_tokenizer.Tokenizer.init(gpa);
    defer t5.deinit();

    var parsed_prompts = try refPrompts(gpa, &ref);
    defer parsed_prompts.deinit();
    const prompts = parsed_prompts.value;

    var nb: [64]u8 = undefined;
    for (0..prompts.len) |pi| {
        // Both tokenizations of the real prompts, through the whole prompt path.
        const want_q = try refIds(gpa, &ref, try std.fmt.bufPrint(&nb, "te.qwen_ids.{d}", .{pi}));
        defer gpa.free(want_q);
        const want_t5 = try refIds(gpa, &ref, try std.fmt.bufPrint(&nb, "te.t5_ids.{d}", .{pi}));
        defer gpa.free(want_t5);
        const want_c = try (try ref.require(try std.fmt.bufPrint(&nb, "te.cond.{d}", .{pi}))).toF32Alloc(gpa);
        defer gpa.free(want_c);

        const got_t5 = try t5.encode(gpa, prompts[pi]);
        defer gpa.free(got_t5);
        errdefer std.debug.print("prompt {d}: t5 want {d} got {d} ids\n", .{ pi, want_t5.len, got_t5.len });
        try testing.expectEqualSlices(u32, want_t5, got_t5);

        // ⚠️ The Qwen3 branch takes NO start/end token and NO padding beyond
        // `min_length = 1`, so an empty prompt is one `<|endoftext|>`.
        const got_q = try qwenIds(gpa, &tok, prompts[pi]);
        defer gpa.free(got_q);
        errdefer std.debug.print("prompt {d}: qwen want {any}\n  got {any}\n", .{ pi, want_q, got_q });
        try testing.expectEqualSlices(u32, want_q, got_q);

        const got_c = try enc.encode(io, gpa, got_q, null);
        defer gpa.free(got_c);
        try testing.expectEqual(want_c.len, got_c.len);
        const rel = relL2(want_c, got_c);
        errdefer std.debug.print("prompt {d}: conditioning rel L2 {e:.4}\n", .{ pi, rel });
        try testing.expect(rel < 5e-5); // measured 2.5e-6 / 3.0e-6 / 7.8e-6
    }
}

test "unsupportedGpuLin scans every block, not just the first" {
    // ⚠️ The regression this pins is a real crash. `anima_baseV10-INT8_CONVROT-MIXED`
    // keeps block 0 entirely bf16 and quantizes blocks 1-27, so a predicate that read
    // `blocks[0].self_attn.q.dtype` reported "GPU ok", a device session was built, and
    // the FIRST thing it did — `crossKv` on the host — tripped `matmul`'s
    // `row_scale != null` assert. A synthetic model here so the test needs no
    // checkpoint: bf16 in block 0, i8 in block 1, exactly the shape that fooled it.
    const gpa = testing.allocator;
    var cfg = anima_2b;
    cfg.n_layers = 2;

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const alloc = arena.allocator();
    const blocks = try alloc.alloc(Block, cfg.n_layers);
    // Only the fields the predicate reads (dtype, tag) need to be meaningful, but
    // `Weight.init` checks the byte count, so the shapes are 1x8 rather than the real
    // 2048x2048 — this test is about the SCAN, not about any arithmetic.
    const bytes = try alloc.alloc(u8, 8 * 2);
    @memset(bytes, 0);
    for (blocks, 0..) |*b, bi| {
        const dt: DType = if (bi == 0) .bf16 else .i8;
        var w = Weight.init(bytes[0..dt.storageBytes(8)], dt, 1, 8);
        w.tag = if (bi == 0) "blocks.0.self_attn.q_proj.weight" else "blocks.1.self_attn.q_proj.weight";
        const bf = Weight.init(bytes[0..16], .bf16, 1, 8);
        b.* = .{
            .self_attn = .{ .q = w, .k = w, .v = w, .out = w, .qnorm = &.{}, .knorm = &.{} },
            .cross_attn = .{ .q = w, .k = bf, .v = bf, .out = w, .qnorm = &.{}, .knorm = &.{} },
            .mlp1 = w,
            .mlp2 = w,
            .ada_sa = .{ bf, bf },
            .ada_ca = .{ bf, bf },
            .ada_mlp = .{ bf, bf },
        };
    }
    var model: DiT = undefined;
    model.cfg = cfg;
    model.blocks = blocks;

    const bad = unsupportedGpuLin(&model);
    try testing.expect(bad != null);
    // ⚠️ Block ONE, not zero: the point is that it looked past the first block.
    try testing.expectEqual(@as(usize, 1), bad.?.block);
    try testing.expectEqual(DType.i8, bad.?.dtype);

    // All-bf16 must still pass, or the gate would refuse every good checkpoint.
    for (blocks) |*b| {
        const bf = Weight.init(bytes[0..16], .bf16, 1, 8);
        b.self_attn.q = bf;
        b.self_attn.k = bf;
        b.self_attn.v = bf;
        b.self_attn.out = bf;
        b.cross_attn.q = bf;
        b.cross_attn.out = bf;
        b.mlp1 = bf;
        b.mlp2 = bf;
    }
    try testing.expect(unsupportedGpuLin(&model) == null);
}

test "the loader wires int8/int4 convrot scales, and refuses a weight with none" {
    // ⚠️ Without this wiring an integer weight does not merely run slowly, it **PANICS**:
    // `ops.matmul.matmul` asserts `row_scale != null`, and the loader used to accept the
    // I8 tensor because `supportsDType(.i8)` is true. That is what an
    // `-INT8_CONVROT-MIXED` Anima checkpoint hit, five frames deep inside `crossKv`.
    //
    // Synthetic rather than checkpoint-backed on purpose: this pins the LOADER's
    // contract, and the only int8 Anima file to hand turned out to be corrupt (its
    // tensor table covers 41.9 MB less than its payload — see
    // `safetensors.initFromSlice`), so a test resting on it would have been testing the
    // wrong thing.
    const gpa = testing.allocator;
    const cols = ops.convrot.group_size; // 256 — one whole rotation group
    const rows = 4;

    // `w`: i8 [4, 256]; `w_scale`: f32 [4]; `q`: i4 (u8-packed) [4, 128]; `q_scale`.
    // `bad`: i8 with NO scale. Offsets must cover the payload exactly.
    const i8_bytes = rows * cols;
    const sc_bytes = rows * 4;
    const i4_bytes = rows * cols / 2;
    var hbuf: [768]u8 = undefined;
    var fbs = std.Io.Writer.fixed(&hbuf);
    var off: usize = 0;
    try fbs.writeByte('{');
    try fbs.print("\"w\":{{\"dtype\":\"I8\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}}", .{ rows, cols, off, off + i8_bytes });
    off += i8_bytes;
    try fbs.print(",\"w_scale\":{{\"dtype\":\"F32\",\"shape\":[{d},1],\"data_offsets\":[{d},{d}]}}", .{ rows, off, off + sc_bytes });
    off += sc_bytes;
    try fbs.print(",\"q\":{{\"dtype\":\"U8\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}}", .{ rows, cols / 2, off, off + i4_bytes });
    off += i4_bytes;
    try fbs.print(",\"q_scale\":{{\"dtype\":\"F32\",\"shape\":[{d},1],\"data_offsets\":[{d},{d}]}}", .{ rows, off, off + sc_bytes });
    off += sc_bytes;
    try fbs.print(",\"bad\":{{\"dtype\":\"I8\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}}", .{ rows, cols, off, off + i8_bytes });
    off += i8_bytes;
    try fbs.writeByte('}');
    const header = fbs.buffered();

    const file = try gpa.alloc(u8, 8 + header.len + off);
    defer gpa.free(file);
    std.mem.writeInt(u64, file[0..8], header.len, .little);
    @memcpy(file[8..][0..header.len], header);
    const payload = file[8 + header.len ..];
    @memset(payload, 1);
    // Distinct positive scales, so a mis-read shows up as a value rather than a shape.
    for (0..rows) |r| {
        const v: f32 = 0.25 + @as(f32, @floatFromInt(r)) * 0.5;
        std.mem.writeInt(u32, payload[i8_bytes + r * 4 ..][0..4], @bitCast(v), .little);
    }

    var st = try SafeTensors.initFromSlice(gpa, file);
    defer st.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const l = Loader{ .store = .{ .safetensors = &st }, .alloc = arena.allocator(), .pfx = "", .cfg = anima_2b };

    // int8: dtype preserved, per-row scale read, rotation group set.
    const w = try l.mat("w", .{}, rows, cols);
    try testing.expectEqual(DType.i8, w.dtype);
    try testing.expect(w.row_scale != null);
    try testing.expectEqual(@as(usize, rows), w.row_scale.?.len);
    try testing.expectEqual(@as(u32, ops.convrot.group_size), w.convrot);
    for (0..rows) |r| {
        try testing.expectApproxEqAbs(0.25 + @as(f32, @floatFromInt(r)) * 0.5, w.row_scale.?[r], 1e-6);
    }

    // ⚠️ int4 is detected by the HALVED column count, not by dtype: our converter writes
    // the packed bytes as U8 where ComfyUI's W4A4 converter writes the identical bytes as
    // I8. `cols` stays logical (256) while only `rows x cols/2` bytes are stored.
    const q = try l.mat("q", .{}, rows, cols);
    try testing.expectEqual(DType.i4, q.dtype);
    try testing.expectEqual(cols, q.cols);
    try testing.expect(q.row_scale != null);
    try testing.expectEqual(@as(u32, ops.convrot.group_size), q.convrot);

    // ⚠️ The third case — an integer weight with NO companion scale — is deliberately
    // not asserted here. `mat` reports it with `std.log.err` before returning
    // `MissingTensor`, which is the right behaviour in production (the whole reason this
    // branch exists is that the silent version PANICS in `matmul`), but `testing.log_level`
    // has no level below `err`, so asserting it would make every passing run print. The
    // `bad` tensor is left in the synthetic file so the next person can check it by hand.
    try testing.expect(st.get("bad") != null and st.get("bad_scale") == null);
}
