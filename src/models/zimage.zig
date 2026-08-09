//! Z-Image (`NextDiT`) — Tongyi's 6B text-to-image diffusion transformer, the
//! architecture "zit" checkpoints use.
//!
//! ⚠️ **It is not its own model in ComfyUI**: Z-Image runs through
//! `comfy/ldm/lumina/model.py`, the Lumina-Image-2.0 `NextDiT`, switched into a
//! different shape by `z_image_modulation=True` / `pad_tokens_multiple=32` /
//! `time_scale=1000` / `rope_theta=256`, and selected purely by `dim == 3840`
//! (`supported_models.py::ZImage`). Anything here that reads as "Lumina but
//! different" is that flag.
//!
//! Single-stream, like krea2's DiT: the text half and the 2x2-patchified latent
//! half are refined separately, concatenated into one sequence, and run through 30
//! identical blocks. It shares krea2's RMSNorm/SwiGLU/3-axis-interleaved-RoPE
//! vocabulary, so most of `ops` carries straight over. Six things do **not**, and
//! each one is a silent wrong answer rather than an error:
//!
//! 1. **Sandwich norms.** A block is
//!    `x += tanh(gate) * norm2(sublayer(modulate(norm1(x), scale)))` — there is a
//!    second RMSNorm on the sublayer's *output*, inside the residual. krea2 has
//!    only the pre-norm.
//! 2. **The gate is `tanh`'d and the modulation has NO SHIFT.**
//!    `modulate(x, scale) = x * (1 + scale)`, and the AdaLN linear emits four
//!    chunks (`scale_msa, gate_msa, scale_mlp, gate_mlp`) rather than six.
//! 3. **No SiLU before the block AdaLN linear.** Under `z_image_modulation` the
//!    per-block `adaLN_modulation` is a bare `Linear(256, 4*dim)` — the
//!    `nn.Sequential(nn.SiLU(), Linear(...))` of stock Lumina is dropped. The
//!    *final layer* keeps its SiLU. That asymmetry is why the checkpoint numbers
//!    them `layers.N.adaLN_modulation.0` and `final_layer.adaLN_modulation.1`.
//! 4. **The final norm is a weightless LayerNorm (eps 1e-6)**, not an RMSNorm.
//! 5. **Two learned pad tokens.** Both halves are padded up to a multiple of 32
//!    with `cap_pad_token` / `x_pad_token`, and the two halves get their pad
//!    *positions* differently — see `ropeFreqs`.
//! 6. **The output is NEGATED** (`return -img`). Combined with CONST/flow
//!    parameterization (`denoised = x - v*sigma`) the sign is load-bearing: get it
//!    wrong and the sampler walks away from the data manifold, producing noise
//!    with no error anywhere.
//!
//! ⚠️ **The patch feature order is `(ph, pw, c)` — channel FASTEST — where krea2's
//! is `(c, ph, pw)`.** Both patchify and unpatchify use it (`permute(0,2,4,3,5,1)`
//! and `view(h, w, pH, pW, C).permute(4,0,2,1,3)`). Getting it backwards is a pure
//! permutation: every norm, every magnitude and every per-stage statistic still
//! matches, and only the rendered image is wrong. Same class of bug as the SD
//! planar/channel-last mixup this repo already paid for once.
//!
//! Loads from any `WeightStore` — safetensors or GGUF. Large weights keep their
//! checkpoint dtype and dequantize inside the GEMM; norm scales and biases become
//! f32 at load. **The store's mapping must outlive the model.**
//!
//! Pinned against ComfyUI by `tools/gen_zimage_fixtures.py`.

const std = @import("std");
const tp_core = @import("tp_core");
const safetensors = tp_core.safetensors;
const weights_mod = tp_core.weights;
const ops = @import("tp_ops");
const quant_weight = @import("quant_weight.zig");

const SafeTensors = safetensors.SafeTensors;
const WeightStore = weights_mod.WeightStore;
const Weight = ops.matmul.Weight;
const DType = tp_core.dtype.DType;

pub const Config = struct {
    dim: usize,
    /// Blocks in the joint trunk.
    n_layers: usize,
    /// Blocks in *each* of `context_refiner` and `noise_refiner`.
    n_refiner_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    /// SwiGLU inner width. Lumina derives it from `ffn_dim_multiplier` and
    /// `multiple_of`; for Z-Image that resolves to 10240 and is stored directly
    /// here, since a loader can only check the shape it actually expects.
    mlp_dim: usize,
    /// Text-encoder hidden width the `cap_embedder` consumes (Qwen3-4B: 2560).
    cap_dim: usize,
    patch: usize,
    channels: usize,
    /// AdaLN conditioning width — `min(dim, 256)` under `z_image_modulation`,
    /// and also the `t_embedder`'s output width.
    mod_dim: usize,
    /// `TimestepEmbedder`'s hidden width, `min(dim, 1024)`.
    t_hidden: usize,
    /// Sinusoidal frequency count feeding the timestep MLP.
    t_freq: usize,
    rope_theta: f64,
    rope_axes: [3]usize,
    /// Both halves of the sequence are padded up to a multiple of this with a
    /// learned pad token.
    pad_multiple: usize,
    /// The timestep is scaled by this before the sinusoidal embedding.
    time_scale: f32,
    /// RMSNorm epsilon for the block and `cap_embedder` norms.
    norm_eps: f32,
    /// ⚠️ The per-head Q/K RMSNorms are built as `RMSNorm(head_dim,
    /// elementwise_affine=True)` with **no eps argument**, so torch falls back to
    /// `finfo(float32).eps` — a *different*, far smaller epsilon than the 1e-5 the
    /// block norms use. ComfyUI's own fused path spells this out
    /// (`self.q_norm.eps if ... is not None else torch.finfo(torch.float32).eps`).
    qk_eps: f32,
    /// `final_layer.norm_final` is `LayerNorm(dim, elementwise_affine=False, eps=1e-6)`.
    final_eps: f32,

    pub fn qDim(self: Config) usize {
        return self.n_heads * self.head_dim;
    }
    pub fn kvDim(self: Config) usize {
        return self.n_kv_heads * self.head_dim;
    }
    /// Width of one patch token before `x_embedder` / after `final_layer.linear`.
    pub fn patchDim(self: Config) usize {
        return self.channels * self.patch * self.patch;
    }
    /// Round a token count up to the learned-pad-token multiple.
    pub fn padded(self: Config, n: usize) usize {
        if (self.pad_multiple == 0) return n;
        return (n + self.pad_multiple - 1) / self.pad_multiple * self.pad_multiple;
    }
};

/// The one configuration in the wild, as `comfy/model_detection.py` derives it for
/// `dim == 3840`. Asserted against ComfyUI's own detection by
/// `tools/gen_zimage_fixtures.py`, so this cannot silently drift from the reference.
pub const z_image: Config = .{
    .dim = 3840,
    .n_layers = 30,
    .n_refiner_layers = 2,
    .n_heads = 30,
    .n_kv_heads = 30,
    .head_dim = 128,
    .mlp_dim = 10240,
    .cap_dim = 2560,
    .patch = 2,
    .channels = 16,
    .mod_dim = 256,
    .t_hidden = 1024,
    .t_freq = 256,
    .rope_theta = 256.0,
    .rope_axes = .{ 32, 48, 48 },
    .pad_multiple = 32,
    .time_scale = 1000.0,
    .norm_eps = 1e-5,
    .qk_eps = std.math.floatEps(f32),
    .final_eps = 1e-6,
};

/// Latent geometry, for callers that need it without a loaded model.
pub const latent_channels = z_image.channels;
pub const spatial_scale = 8;

/// ComfyUI `latent_formats.Flux` — Z-Image inherits it through `Lumina2`.
pub const scale_factor: f32 = 0.3611;
pub const shift_factor: f32 = 0.1159;

/// Linear latent→RGB approximation for the live sampling preview
/// (`latent_formats.Flux.latent_rgb_factors`). Distinct from krea2's Wan matrix
/// and from either SD one; the wrong matrix is not a crash, just a preview with
/// plausible structure and wrong colours.
pub const latent_rgb_factors = [latent_channels][3]f32{
    .{ -0.0346, 0.0244, 0.0681 },
    .{ 0.0034, 0.0210, 0.0687 },
    .{ 0.0275, -0.0668, -0.0433 },
    .{ -0.0174, 0.0160, 0.0617 },
    .{ 0.0859, 0.0721, 0.0329 },
    .{ 0.0004, 0.0383, 0.0115 },
    .{ 0.0405, 0.0861, 0.0915 },
    .{ -0.0236, -0.0185, -0.0259 },
    .{ -0.0245, 0.0250, 0.1180 },
    .{ 0.1008, 0.0755, -0.0421 },
    .{ -0.0515, 0.0201, 0.0011 },
    .{ 0.0428, -0.0012, -0.0036 },
    .{ 0.0817, 0.0765, 0.0749 },
    .{ -0.1264, -0.0522, -0.1103 },
    .{ -0.0280, -0.0881, -0.0499 },
    .{ -0.1262, -0.0982, -0.0778 },
};
pub const latent_rgb_bias = [3]f32{ -0.0329, -0.0718, -0.0851 };

/// Fill `rgb_out` (`[zh*zw][3]` RGB8) with the latent2rgb preview of the planar
/// `[16][zh*zw]` sampler latent `z`, using ComfyUI's `((v + 1) / 2).clamp(0, 1) * 255`
/// mapping. Its own function rather than `sd_vae`'s because that one is compiled
/// against a 4-channel matrix — the same reason `wan_vae` has its own.
pub fn latentPreviewInto(rgb_out: []u8, z: []const f32, zh: usize, zw: usize) void {
    const plane = zh * zw;
    std.debug.assert(rgb_out.len >= plane * 3 and z.len >= latent_channels * plane);
    for (0..plane) |p| {
        var acc = latent_rgb_bias;
        inline for (0..latent_channels) |c| {
            const v = z[c * plane + p];
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

const LinearW = struct {
    w: Weight,
    b: ?[]const f32,
};

const Attn = struct {
    /// Fused `[q_dim + 2*kv_dim, dim]`, exactly as stored. One GEMM beats three
    /// (the packed CPU kernel amortizes activation packing over N); the halves are
    /// split out of the *result*, which is a row copy and negligible beside it.
    qkv: Weight,
    out: Weight,
    /// Per-head RMSNorm scales, `[head_dim]`, plain weights (not `1 + scale`).
    qnorm: []const f32,
    knorm: []const f32,
};

const Ffn = struct {
    w1: Weight, // gate
    w3: Weight, // up
    w2: Weight, // down
};

const Block = struct {
    attn: Attn,
    ffn: Ffn,
    attn_norm1: []const f32,
    attn_norm2: []const f32,
    ffn_norm1: []const f32,
    ffn_norm2: []const f32,
    /// `Linear(mod_dim, 4*dim)`. Null for `context_refiner`, whose blocks are
    /// built with `modulation=False` and take no timestep at all.
    ada: ?LinearW,
};

pub const DiT = struct {
    arena: std.heap.ArenaAllocator,
    cfg: Config,

    x_embedder: LinearW, // patchDim -> dim
    x_pad_token: []const f32, // [dim]
    cap_norm: []const f32, // [cap_dim]
    cap_embedder: LinearW, // cap_dim -> dim
    cap_pad_token: []const f32, // [dim]
    t_mlp0: LinearW, // t_freq -> t_hidden
    t_mlp2: LinearW, // t_hidden -> mod_dim
    context_refiner: []Block,
    noise_refiner: []Block,
    layers: []Block,
    final_ada: LinearW, // mod_dim -> dim
    final_linear: LinearW, // dim -> patchDim

    pub fn load(gpa: std.mem.Allocator, store: WeightStore, cfg: Config) !DiT {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();

        // A full single-file checkpoint keeps the LDM container prefix; a
        // denoiser-only export strips it. Same two spellings `detectFamily` reads.
        const pfx: []const u8 = if (store.get("model.diffusion_model.cap_embedder.1.weight") != null)
            "model.diffusion_model."
        else
            "";
        const l = Loader{ .store = store, .alloc = alloc, .pfx = pfx, .cfg = cfg };

        const context_refiner = try alloc.alloc(Block, cfg.n_refiner_layers);
        for (context_refiner, 0..) |*b, i| b.* = try l.block("context_refiner.{d}", .{i}, false);
        const noise_refiner = try alloc.alloc(Block, cfg.n_refiner_layers);
        for (noise_refiner, 0..) |*b, i| b.* = try l.block("noise_refiner.{d}", .{i}, true);
        const layers = try alloc.alloc(Block, cfg.n_layers);
        for (layers, 0..) |*b, i| b.* = try l.block("layers.{d}", .{i}, true);

        // `x_embedder` and `final_layer.linear` are the two projections the GPU
        // arms hand to the fused f32-only `opMatmul`; normalize their storage once
        // here. They are tiny ([dim, 64] / [64, dim]) so the copy is free.
        var x_embedder = try l.linear("x_embedder", cfg.dim, cfg.patchDim(), true);
        x_embedder.w = try ops.matmul.materializeF32(alloc, x_embedder.w);
        var final_linear = try l.linear("final_layer.linear", cfg.patchDim(), cfg.dim, true);
        final_linear.w = try ops.matmul.materializeF32(alloc, final_linear.w);

        return .{
            .arena = arena,
            .cfg = cfg,
            .x_embedder = x_embedder,
            .x_pad_token = try l.vec("x_pad_token", .{}, cfg.dim),
            .cap_norm = try l.vec("cap_embedder.0.weight", .{}, cfg.cap_dim),
            .cap_embedder = try l.linear("cap_embedder.1", cfg.dim, cfg.cap_dim, true),
            .cap_pad_token = try l.vec("cap_pad_token", .{}, cfg.dim),
            .t_mlp0 = try l.linear("t_embedder.mlp.0", cfg.t_hidden, cfg.t_freq, true),
            .t_mlp2 = try l.linear("t_embedder.mlp.2", cfg.mod_dim, cfg.t_hidden, true),
            .context_refiner = context_refiner,
            .noise_refiner = noise_refiner,
            .layers = layers,
            // ⚠️ Index 1, not 0: the final layer keeps stock Lumina's
            // `Sequential(SiLU, Linear)` while the blocks lose the SiLU and so
            // number their linear 0. See the module header.
            .final_ada = try l.linear("final_layer.adaLN_modulation.1", cfg.dim, cfg.mod_dim, true),
            .final_linear = final_linear,
        };
    }

    pub fn deinit(self: *DiT) void {
        self.arena.deinit();
        self.* = undefined;
    }

    // --- stages ------------------------------------------------------------
    //
    // Split the way krea2's DiT is, and for the same reason: `pipeline.Denoiser`
    // builds the per-image constants once and calls only the per-step part.
    // ⚠️ `context_refiner` has NO modulation, so the whole text half is a per-image
    // constant here — unlike krea2, where the text fusion is also t-independent but
    // for a different reason. Recomputing it per step would be pure waste.

    /// The AdaLN conditioning vector: `t_embedder((1 - sigma) * time_scale)`.
    /// Returns `[mod_dim]`; caller frees.
    ///
    /// ⚠️ `t = 1 - sigma` (`NextDiT._forward`), and ComfyUI reaches here with
    /// `timesteps = model_sampling.timestep(sigma) = sigma * multiplier` where
    /// Z-Image's multiplier is **1.0**, so the timestep *is* the sigma. Flux-family
    /// models with multiplier 1000 do not have that identity.
    pub fn adalnInput(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, sigma: f32) ![]f32 {
        const cfg = self.cfg;
        const freq = try gpa.alloc(f32, cfg.t_freq);
        defer gpa.free(freq);
        timestepEmbedding(freq, (1.0 - sigma) * cfg.time_scale);

        const hidden = try gpa.alloc(f32, cfg.t_hidden);
        defer gpa.free(hidden);
        try linear(io, gpa, hidden, freq, 1, self.t_mlp0);
        ops.act.silu(hidden);

        const out = try gpa.alloc(f32, cfg.mod_dim);
        errdefer gpa.free(out);
        try linear(io, gpa, out, hidden, 1, self.t_mlp2);
        return out;
    }

    /// How many blocks take a modulation vector: the `noise_refiner` stack then the
    /// trunk. ⚠️ `context_refiner` is NOT among them (`modulation=False`), which is
    /// also why the caption half is timestep-independent.
    pub fn modulatedBlocks(self: *const DiT) usize {
        return self.noise_refiner.len + self.layers.len;
    }

    /// Every modulated block's AdaLN vector for one timestep, laid out contiguously
    /// as `[modulatedBlocks()][4 * dim]` — `noise_refiner` first, then the trunk, in
    /// execution order — followed by one **zero block** of `dim`. Caller frees.
    ///
    /// This is the form the GPU backends upload: the device `modulate` kernel reads
    /// `(1 + c[scale_off + col]) * x + c[shift_off + col]`, and Z-Image's modulation
    /// has **no shift**, so `shift_off` points at that trailing zero block. Cheaper
    /// than a shift-free kernel variant and exactly equal to it.
    ///
    /// ⚠️ **The gates are `tanh`'d here, on the host, and that is deliberate.** A gate
    /// is one `dim`-wide vector per block per step — 32 x 3840 values against the
    /// millions the block itself touches — so doing it here costs nothing measurable
    /// and saves the device a kernel it does not otherwise need.
    pub fn modulationTable(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, adaln: []const f32) ![]f32 {
        const cfg = self.cfg;
        std.debug.assert(adaln.len == cfg.mod_dim);
        const n = self.modulatedBlocks();
        const out = try gpa.alloc(f32, n * 4 * cfg.dim + cfg.dim);
        errdefer gpa.free(out);
        @memset(out[n * 4 * cfg.dim ..], 0);

        var i: usize = 0;
        for ([_][]Block{ self.noise_refiner, self.layers }) |group| {
            for (group) |*blk| {
                const dst = out[i * 4 * cfg.dim ..][0 .. 4 * cfg.dim];
                try linear(io, gpa, dst, adaln, 1, blk.ada.?);
                // The two gate chunks — `gate_msa` and `gate_mlp` — in the order
                // `scale_msa, gate_msa, scale_mlp, gate_mlp`.
                for (dst[1 * cfg.dim ..][0..cfg.dim]) |*g| g.* = std.math.tanh(g.*);
                for (dst[3 * cfg.dim ..][0..cfg.dim]) |*g| g.* = std.math.tanh(g.*);
                i += 1;
            }
        }
        return out;
    }

    /// `modulationTable` with each pre-norm's WEIGHT FOLDED INTO THE SCALE — the form
    /// a fused rms+modulate kernel wants. `[modulatedBlocks()][4 * dim]` laid out as
    /// `premul_attn, gate_attn, premul_ffn, gate_ffn`, then one zero block of `dim`.
    ///
    /// ⚠️ A second layout exists because the two backends fuse differently, not by
    /// accident: Vulkan has a standalone `modulate` kernel and takes the *unfused*
    /// table, while CUDA has only `rmsMod` (`out = x*inv*premul[col] + shift[col]`),
    /// so the norm weight has to arrive already multiplied in. Both are built here,
    /// from the same AdaLN evaluation, so they cannot drift apart — and the device
    /// tests compare each backend against the same CPU forward.
    ///
    /// `premul = norm_w * (1 + scale)` is exactly `rmsNorm(x, norm_w)` followed by
    /// `modulate(·, scale)`, since both are per-column multiplies.
    pub fn modulationTableFolded(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, adaln: []const f32) ![]f32 {
        const cfg = self.cfg;
        const raw = try self.modulationTable(io, gpa, adaln);
        errdefer gpa.free(raw);
        var i: usize = 0;
        for ([_][]Block{ self.noise_refiner, self.layers }) |group| {
            for (group) |*blk| {
                const dst = raw[i * 4 * cfg.dim ..][0 .. 4 * cfg.dim];
                // slot 0: scale_msa -> attn_norm1 * (1 + scale_msa)
                for (dst[0..cfg.dim], blk.attn_norm1) |*v, nw| v.* = nw * (1.0 + v.*);
                // slot 2: scale_mlp -> ffn_norm1 * (1 + scale_mlp)
                for (dst[2 * cfg.dim ..][0..cfg.dim], blk.ffn_norm1) |*v, nw| v.* = nw * (1.0 + v.*);
                // slots 1 and 3 are the tanh'd gates, already in place.
                i += 1;
            }
        }
        return raw;
    }

    /// Byte offset of the zero block inside a `modulationTable`, i.e. the `shift_off`
    /// the device `modulate` kernel should read.
    pub fn zeroShiftOffset(self: *const DiT) usize {
        return self.modulatedBlocks() * 4 * self.cfg.dim;
    }

    /// The final layer's AdaLN scale for one timestep, `[dim]`. Caller frees.
    /// Keeps the SiLU the blocks drop — see the module header.
    pub fn finalScale(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, adaln: []const f32) ![]f32 {
        const cfg = self.cfg;
        const gated = try gpa.alloc(f32, cfg.mod_dim);
        defer gpa.free(gated);
        @memcpy(gated, adaln);
        ops.act.silu(gated);
        const out = try gpa.alloc(f32, cfg.dim);
        errdefer gpa.free(out);
        try linear(io, gpa, out, gated, 1, self.final_ada);
        return out;
    }

    /// The text half, start to finish: `cap_embedder`, pad to the multiple, then
    /// the `context_refiner` stack. Returns `[padded(seq_txt), dim]`; caller frees.
    ///
    /// Independent of the timestep, hence computed once per conditioning.
    pub fn capTokens(
        self: *const DiT,
        io: std.Io,
        gpa: std.mem.Allocator,
        ctx: []const f32,
        seq_txt: usize,
    ) ![]f32 {
        const cfg = self.cfg;
        std.debug.assert(ctx.len == seq_txt * cfg.cap_dim);
        const n = cfg.padded(seq_txt);

        const out = try gpa.alloc(f32, n * cfg.dim);
        errdefer gpa.free(out);
        {
            const normed = try gpa.alloc(f32, ctx.len);
            defer gpa.free(normed);
            ops.norm.rmsNorm(normed, ctx, self.cap_norm, cfg.norm_eps);
            try linear(io, gpa, out[0 .. seq_txt * cfg.dim], normed, seq_txt, self.cap_embedder);
        }
        for (seq_txt..n) |i| @memcpy(out[i * cfg.dim ..][0..cfg.dim], self.cap_pad_token);

        var freqs = try self.capFreqs(gpa, n);
        defer freqs.deinit(gpa);
        for (self.context_refiner) |*blk| {
            try self.blockForward(io, gpa, blk, out, n, null, freqs);
        }
        return out;
    }

    /// RoPE table for the text half: axis 0 runs `1 .. n`, axes 1 and 2 are zero.
    ///
    /// ⚠️ The `+ 1` is real — `embed_cap` writes `arange(len) + 1.0 + offset`, so
    /// position 0 is never used by a caption token. It exists so the image half can
    /// start at `cap_len + 1` and leave a gap.
    pub fn capFreqs(self: *const DiT, gpa: std.mem.Allocator, cap_padded: usize) !ops.rope.Freqs {
        const pos = try gpa.alloc(f32, cap_padded * 3);
        defer gpa.free(pos);
        for (0..cap_padded) |i| {
            pos[i * 3] = @floatFromInt(i + 1);
            pos[i * 3 + 1] = 0;
            pos[i * 3 + 2] = 0;
        }
        return ops.rope.fluxFreqs(gpa, pos, &self.cfg.rope_axes, self.cfg.rope_theta);
    }

    /// RoPE table for the whole `[cap | image]` sequence.
    ///
    /// ⚠️ **The two halves pad differently, and it is not symmetry you can restore.**
    /// The caption's pad tokens are appended *before* its position ids are built, so
    /// they continue the `1..n` ramp. The image's pad tokens are appended and then
    /// its position tensor is `F.pad`ded with **zeros**, so every image pad token
    /// sits at `(0, 0, 0)` — the same position as each other, and a position no real
    /// token occupies. Making them consistent (either way) changes the attention
    /// pattern of every padded render.
    pub fn ropeFreqs(
        self: *const DiT,
        gpa: std.mem.Allocator,
        cap_padded: usize,
        h: usize,
        w: usize,
    ) !ops.rope.Freqs {
        const cfg = self.cfg;
        const n_img = h * w;
        const img_padded = cfg.padded(n_img);
        const seq = cap_padded + img_padded;

        const pos = try gpa.alloc(f32, seq * 3);
        defer gpa.free(pos);
        for (0..cap_padded) |i| {
            pos[i * 3] = @floatFromInt(i + 1);
            pos[i * 3 + 1] = 0;
            pos[i * 3 + 2] = 0;
        }
        const start_t: f32 = @floatFromInt(cap_padded + 1);
        for (0..h) |hi| {
            for (0..w) |wi| {
                const base = (cap_padded + hi * w + wi) * 3;
                pos[base] = start_t;
                pos[base + 1] = @floatFromInt(hi);
                pos[base + 2] = @floatFromInt(wi);
            }
        }
        @memset(pos[(cap_padded + n_img) * 3 .. seq * 3], 0);
        return ops.rope.fluxFreqs(gpa, pos, &cfg.rope_axes, cfg.rope_theta);
    }

    /// The image half up to the joint trunk: patchify, `x_embedder`, pad, then the
    /// `noise_refiner` stack. Returns `[padded(h*w), dim]`; caller frees.
    ///
    /// `freqs` must be the **image slice** of the full table — the refiner runs on
    /// the image tokens alone, at the positions they will hold in the joint
    /// sequence.
    pub fn noiseTokens(
        self: *const DiT,
        io: std.Io,
        gpa: std.mem.Allocator,
        x_lat: []const f32,
        lat_h: usize,
        lat_w: usize,
        adaln: []const f32,
        img_freqs: ops.rope.Freqs,
    ) ![]f32 {
        const cfg = self.cfg;
        const h = lat_h / cfg.patch;
        const w = lat_w / cfg.patch;
        const n_img = h * w;
        const n = cfg.padded(n_img);

        const out = try gpa.alloc(f32, n * cfg.dim);
        errdefer gpa.free(out);
        {
            const patches = try patchify(gpa, cfg, x_lat, lat_h, lat_w);
            defer gpa.free(patches);
            try linear(io, gpa, out[0 .. n_img * cfg.dim], patches, n_img, self.x_embedder);
        }
        for (n_img..n) |i| @memcpy(out[i * cfg.dim ..][0..cfg.dim], self.x_pad_token);

        for (self.noise_refiner) |*blk| {
            try self.blockForward(io, gpa, blk, out, n, adaln, img_freqs);
        }
        return out;
    }

    /// Predict the flow-matching velocity for one latent.
    ///
    /// `x_lat`/`out` are planar `[channels][lat_h][lat_w]`; `ctx` is the text
    /// encoder's `[seq_txt][cap_dim]` conditioning; `sigma` is the flow timestep.
    pub fn forward(
        self: *const DiT,
        io: std.Io,
        gpa: std.mem.Allocator,
        out: []f32,
        x_lat: []const f32,
        lat_h: usize,
        lat_w: usize,
        sigma: f32,
        ctx: []const f32,
        seq_txt: usize,
        cancel: ?*std.atomic.Value(bool),
    ) !void {
        const cap = try self.capTokens(io, gpa, ctx, seq_txt);
        defer gpa.free(cap);
        const adaln = try self.adalnInput(io, gpa, sigma);
        defer gpa.free(adaln);
        try self.predict(io, gpa, out, x_lat, lat_h, lat_w, cap, self.cfg.padded(seq_txt), adaln, cancel);
    }

    /// The per-step half of `forward`, taking the cached text half and timestep
    /// vector. This is what `pipeline.Denoiser` drives.
    pub fn predict(
        self: *const DiT,
        io: std.Io,
        gpa: std.mem.Allocator,
        out: []f32,
        x_lat: []const f32,
        lat_h: usize,
        lat_w: usize,
        cap: []const f32,
        cap_padded: usize,
        adaln: []const f32,
        cancel: ?*std.atomic.Value(bool),
    ) !void {
        const cfg = self.cfg;
        std.debug.assert(lat_h % cfg.patch == 0 and lat_w % cfg.patch == 0);
        std.debug.assert(x_lat.len == cfg.channels * lat_h * lat_w);
        std.debug.assert(out.len == x_lat.len);
        std.debug.assert(cap.len == cap_padded * cfg.dim);
        std.debug.assert(adaln.len == cfg.mod_dim);

        // Arm fine-grained cancel inside the CPU kernels for the whole forward: a
        // single MLP GEMM here is seconds of work, so per-block polling alone
        // leaves a multi-second stop latency.
        const prev_tok = ops.cancel.token;
        ops.cancel.token = cancel;
        defer ops.cancel.token = prev_tok;

        const h = lat_h / cfg.patch;
        const w = lat_w / cfg.patch;
        const n_img = h * w;
        const img_padded = cfg.padded(n_img);
        const seq = cap_padded + img_padded;

        var freqs = try self.ropeFreqs(gpa, cap_padded, h, w);
        defer freqs.deinit(gpa);
        const img_freqs: ops.rope.Freqs = .{
            .cos = freqs.cos[cap_padded * freqs.half ..],
            .sin = freqs.sin[cap_padded * freqs.half ..],
            .half = freqs.half,
        };

        const img = try self.noiseTokens(io, gpa, x_lat, lat_h, lat_w, adaln, img_freqs);
        defer gpa.free(img);

        // Joint sequence: [cap | image], in that order — `unpatchify` slices the
        // image half back out at `cap_padded`.
        const x = try gpa.alloc(f32, seq * cfg.dim);
        defer gpa.free(x);
        @memcpy(x[0 .. cap_padded * cfg.dim], cap);
        @memcpy(x[cap_padded * cfg.dim ..], img);

        for (self.layers) |*blk| {
            // Poll between blocks so a stop lands mid-step; a full CPU step is tens
            // of seconds.
            if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
            try self.blockForward(io, gpa, blk, x, seq, adaln, freqs);
        }

        // The final layer is row-wise, so running it on the image rows alone is
        // exactly equal to running it on the whole sequence and slicing — which is
        // what the reference does. The caption and pad rows are discarded either way.
        try self.finalize(io, gpa, out, x[cap_padded * cfg.dim ..][0 .. n_img * cfg.dim], adaln, lat_h, lat_w);
    }

    /// `final_layer` + unpatchify + the sign flip. `img_rows` is `[n_img, dim]` and
    /// is modified in place.
    pub fn finalize(
        self: *const DiT,
        io: std.Io,
        gpa: std.mem.Allocator,
        out: []f32,
        img_rows: []f32,
        adaln: []const f32,
        lat_h: usize,
        lat_w: usize,
    ) !void {
        const cfg = self.cfg;
        const n_img = (lat_h / cfg.patch) * (lat_w / cfg.patch);
        std.debug.assert(img_rows.len == n_img * cfg.dim);

        const scale = try gpa.alloc(f32, cfg.dim);
        defer gpa.free(scale);
        {
            const gated = try gpa.alloc(f32, cfg.mod_dim);
            defer gpa.free(gated);
            @memcpy(gated, adaln);
            ops.act.silu(gated); // the final layer keeps Lumina's SiLU; blocks do not
            try linear(io, gpa, scale, gated, 1, self.final_ada);
        }
        ops.norm.layerNormUnit(img_rows, img_rows, cfg.dim, cfg.final_eps);
        modulate(img_rows, scale);

        const patches = try gpa.alloc(f32, n_img * cfg.patchDim());
        defer gpa.free(patches);
        try linear(io, gpa, patches, img_rows, n_img, self.final_linear);
        unpatchify(cfg, out, patches, lat_h, lat_w);
    }

    /// One `JointTransformerBlock`. `adaln` is null exactly for the
    /// `context_refiner` blocks, which are built with `modulation=False`.
    fn blockForward(
        self: *const DiT,
        io: std.Io,
        gpa: std.mem.Allocator,
        blk: *const Block,
        x: []f32,
        seq: usize,
        adaln: ?[]const f32,
        freqs: ops.rope.Freqs,
    ) !void {
        const cfg = self.cfg;
        std.debug.assert(x.len == seq * cfg.dim);
        std.debug.assert((blk.ada != null) == (adaln != null));

        // scale_msa, gate_msa, scale_mlp, gate_mlp — in that order, one chunk each.
        var mv: []f32 = &.{};
        defer if (mv.len != 0) gpa.free(mv);
        if (blk.ada) |ada| {
            mv = try gpa.alloc(f32, 4 * cfg.dim);
            try linear(io, gpa, mv, adaln.?, 1, ada);
            // The gates are tanh'd; do it once here rather than per row.
            for (mv[1 * cfg.dim ..][0..cfg.dim]) |*g| g.* = std.math.tanh(g.*);
            for (mv[3 * cfg.dim ..][0..cfg.dim]) |*g| g.* = std.math.tanh(g.*);
        }

        const normed = try gpa.alloc(f32, seq * cfg.dim);
        defer gpa.free(normed);
        const delta = try gpa.alloc(f32, seq * cfg.dim);
        defer gpa.free(delta);

        // x += tanh(gate_msa) * attn_norm2(attn(modulate(attn_norm1(x), scale_msa)))
        ops.norm.rmsNorm(normed, x, blk.attn_norm1, cfg.norm_eps);
        if (mv.len != 0) modulate(normed, mv[0 .. cfg.dim]);
        try self.attnForward(io, gpa, &blk.attn, normed, seq, freqs, delta);
        ops.norm.rmsNorm(delta, delta, blk.attn_norm2, cfg.norm_eps);
        if (mv.len != 0) gatedAdd(x, delta, mv[cfg.dim ..][0..cfg.dim]) else plainAdd(x, delta);

        // x += tanh(gate_mlp) * ffn_norm2(swiglu(modulate(ffn_norm1(x), scale_mlp)))
        ops.norm.rmsNorm(normed, x, blk.ffn_norm1, cfg.norm_eps);
        if (mv.len != 0) modulate(normed, mv[2 * cfg.dim ..][0..cfg.dim]);
        try self.ffnForward(io, gpa, &blk.ffn, normed, seq, delta);
        ops.norm.rmsNorm(delta, delta, blk.ffn_norm2, cfg.norm_eps);
        if (mv.len != 0) gatedAdd(x, delta, mv[3 * cfg.dim ..][0..cfg.dim]) else plainAdd(x, delta);
    }

    /// `JointAttention`: fused QKV, per-head Q/K RMSNorm, interleaved RoPE, full
    /// (unmasked) bidirectional attention, output projection with no bias.
    fn attnForward(
        self: *const DiT,
        io: std.Io,
        gpa: std.mem.Allocator,
        attn: *const Attn,
        x: []const f32,
        seq: usize,
        freqs: ops.rope.Freqs,
        out: []f32,
    ) !void {
        const cfg = self.cfg;
        const q_dim = cfg.qDim();
        const kv_dim = cfg.kvDim();
        const fused_dim = q_dim + 2 * kv_dim;

        const fused = try gpa.alloc(f32, seq * fused_dim);
        defer gpa.free(fused);
        try ops.matmul.matmul(io, gpa, fused, x, seq, attn.qkv, null);

        const q = try gpa.alloc(f32, seq * q_dim);
        defer gpa.free(q);
        const k = try gpa.alloc(f32, seq * kv_dim);
        defer gpa.free(k);
        const v = try gpa.alloc(f32, seq * kv_dim);
        defer gpa.free(v);
        for (0..seq) |r| {
            const row = fused[r * fused_dim ..];
            @memcpy(q[r * q_dim ..][0..q_dim], row[0..q_dim]);
            @memcpy(k[r * kv_dim ..][0..kv_dim], row[q_dim..][0..kv_dim]);
            @memcpy(v[r * kv_dim ..][0..kv_dim], row[q_dim + kv_dim ..][0..kv_dim]);
        }

        // Per-head norms: `rmsNorm` reduces over `weight.len`, which is head_dim, so
        // the flat [seq, heads*head_dim] buffer is already the right shape.
        ops.norm.rmsNorm(q, q, attn.qnorm, cfg.qk_eps);
        ops.norm.rmsNorm(k, k, attn.knorm, cfg.qk_eps);
        ops.rope.applyInterleaved(q, freqs, seq, cfg.n_heads, cfg.head_dim);
        ops.rope.applyInterleaved(k, freqs, seq, cfg.n_kv_heads, cfg.head_dim);

        const attn_out = try gpa.alloc(f32, seq * q_dim);
        defer gpa.free(attn_out);
        try ops.attention.attention(io, gpa, attn_out, q, k, v, .{
            .seq_q = seq,
            .seq_kv = seq,
            .n_heads = cfg.n_heads,
            .n_kv_heads = cfg.n_kv_heads,
            .head_dim = cfg.head_dim,
        });
        try ops.matmul.matmul(io, gpa, out, attn_out, seq, attn.out, null);
    }

    fn ffnForward(
        self: *const DiT,
        io: std.Io,
        gpa: std.mem.Allocator,
        ffn: *const Ffn,
        x: []const f32,
        seq: usize,
        out: []f32,
    ) !void {
        const inner = self.cfg.mlp_dim;
        const gate = try gpa.alloc(f32, seq * inner);
        defer gpa.free(gate);
        const up = try gpa.alloc(f32, seq * inner);
        defer gpa.free(up);
        try ops.matmul.matmul(io, gpa, gate, x, seq, ffn.w1, null);
        try ops.matmul.matmul(io, gpa, up, x, seq, ffn.w3, null);
        ops.act.siluMul(gate, up);
        try ops.matmul.matmul(io, gpa, out, gate, seq, ffn.w2, null);
    }
};

// --- free helpers -----------------------------------------------------------

/// Sinusoidal timestep embedding, `[cos(t w_i) ... sin(t w_i) ...]` with
/// `w_i = 10000^(-i/half)` — `comfy/ldm/modules/diffusionmodules/util.py`.
///
/// f64 internals on purpose, the same reasoning `sd_unet.timestepEmbedding`
/// records: at `i = 0` the argument is the scaled timestep itself (up to 1000), so
/// a 1e-7 relative slip in `freq` becomes ~1e-4 in `cos`. Computing more accurately
/// than the reference bounds the disagreement by the *reference's* own rounding
/// instead of stacking two errors.
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

/// Row-wise AdaLN with **no shift**: `x = (1 + scale) * x`.
fn modulate(x: []f32, scale: []const f32) void {
    const dim = scale.len;
    var row: usize = 0;
    while (row < x.len) : (row += dim) {
        for (x[row..][0..dim], scale) |*v, sc| v.* = (1.0 + sc) * v.*;
    }
}

/// Row-wise gated residual: `x += gate * delta`. `gate` is already `tanh`'d.
fn gatedAdd(x: []f32, delta: []const f32, gate: []const f32) void {
    const dim = gate.len;
    var row: usize = 0;
    while (row < x.len) : (row += dim) {
        for (x[row..][0..dim], delta[row..][0..dim], gate) |*v, d, g| v.* += g * d;
    }
}

fn plainAdd(x: []f32, delta: []const f32) void {
    for (x, delta) |*v, d| v.* += d;
}

/// Planar `[c][lat_h][lat_w]` → `[n_img, patch*patch*channels]` patch rows.
///
/// ⚠️ The feature order inside a token is `(ph, pw, c)` — **channel fastest** —
/// from `x.view(B, C, H/p, p, W/p, p).permute(0, 2, 4, 3, 5, 1).flatten(3)`. krea2's
/// is `(c, ph, pw)`; swapping them is rms-preserving and invisible to every check
/// except the rendered image.
pub fn patchify(gpa: std.mem.Allocator, cfg: Config, x_lat: []const f32, lat_h: usize, lat_w: usize) ![]f32 {
    const p = cfg.patch;
    const c_n = cfg.channels;
    const h = lat_h / p;
    const w = lat_w / p;
    const pd = cfg.patchDim();
    const out = try gpa.alloc(f32, h * w * pd);
    for (0..h) |hi| {
        for (0..w) |wi| {
            const tok = out[(hi * w + wi) * pd ..];
            for (0..p) |ph| {
                for (0..p) |pw| {
                    for (0..c_n) |c| {
                        tok[(ph * p + pw) * c_n + c] =
                            x_lat[c * lat_h * lat_w + (hi * p + ph) * lat_w + (wi * p + pw)];
                    }
                }
            }
        }
    }
    return out;
}

/// The inverse of `patchify`, **with the output sign flip** (`NextDiT` returns
/// `-img`). Scatters `[n_img, patch*patch*channels]` back into planar `out`.
pub fn unpatchify(cfg: Config, out: []f32, patches: []const f32, lat_h: usize, lat_w: usize) void {
    const p = cfg.patch;
    const c_n = cfg.channels;
    const h = lat_h / p;
    const w = lat_w / p;
    const pd = cfg.patchDim();
    for (0..h) |hi| {
        for (0..w) |wi| {
            const tok = patches[(hi * w + wi) * pd ..];
            for (0..p) |ph| {
                for (0..p) |pw| {
                    for (0..c_n) |c| {
                        out[c * lat_h * lat_w + (hi * p + ph) * lat_w + (wi * p + pw)] =
                            -tok[(ph * p + pw) * c_n + c];
                    }
                }
            }
        }
    }
}

fn linear(io: std.Io, gpa: std.mem.Allocator, out: []f32, x: []const f32, m: usize, lw: LinearW) !void {
    try ops.matmul.matmul(io, gpa, out, x, m, lw.w, lw.b);
}

/// Whether the GPU forwards have a GEMM path for block linears of this dtype.
/// Mirrors `dit.gpuLinKindSupported`: an unrecognized dtype on those paths is not a
/// slow path, it is silently wrong output, so both gate on this before dispatching.
pub fn gpuLinKindSupported(dt: DType) bool {
    return switch (dt) {
        // `.nvfp4` is decoded to f16 inside the GEMM (weight-only, which is what NVFP4 is
        // below Blackwell), so it runs wherever the f16 GEMM does.
        .bf16, .f16, .f32, .f8_e4m3, .nvfp4 => true,
        else => false,
    };
}

/// The first block linear the device forward cannot run, or null if it can run all of them.
///
/// ⚠️ **Scans EVERY layer, not `layers[0].attn.qkv`.** That single-tensor probe was correct
/// only while no mixed Z-Image checkpoint existed — and `anima_baseV10-INT8_CONVROT-MIXED`
/// is the standing proof that "mixed" means mixed PER BLOCK, where a block-0 probe says yes
/// and the forward then panics on the first thing it does. Returns the tensor's name so the
/// refusal can say which layer.
pub fn unsupportedGpuLin(model: *const DiT, extra: fn (DType) bool) ?struct { tag: []const u8, dtype: DType } {
    for (model.layers) |*b| {
        const lins = [_]Weight{ b.attn.qkv, b.attn.out, b.ffn.w1, b.ffn.w3, b.ffn.w2 };
        for (lins) |w| {
            if (!gpuLinKindSupported(w.dtype) or !extra(w.dtype))
                return .{ .tag = w.tag orelse "?", .dtype = w.dtype };
        }
        if (b.ada) |a| if (!gpuLinKindSupported(a.w.dtype) or !extra(a.w.dtype))
            return .{ .tag = a.w.tag orelse "?", .dtype = a.w.dtype };
    }
    return null;
}

/// Largest transient buffer any NVFP4 block linear decodes into, per the backend's own
/// sizing rule, or 0 when the model has none.
pub fn maxNvfp4Scratch(model: *const DiT, comptime bytesFor: fn (rows: usize, cols: usize) usize) usize {
    var max: usize = 0;
    for (model.layers) |*b| {
        const lins = [_]Weight{ b.attn.qkv, b.attn.out, b.ffn.w1, b.ffn.w3, b.ffn.w2 };
        for (lins) |w| {
            if (w.dtype == .nvfp4) max = @max(max, bytesFor(w.rows, w.cols));
        }
        if (b.ada) |a| if (a.w.dtype == .nvfp4) {
            max = @max(max, bytesFor(a.w.rows, a.w.cols));
        };
    }
    return max;
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
            std.log.err("zimage: missing tensor {s}", .{nm});
            return error.MissingTensor;
        };
        const shape = view.info.shape.slice();
        // ⚠️ **The packed ComfyUI 4-bit formats come first**, each detected by its OWN
        // sidecar (`_scale_2` for NVFP4, `_s_rel` for W4A8) rather than by dtype or shape:
        // both store `[rows, cols/2]`, so the plain shape check below would reject them.
        // NVFP4's nibbles are E2M1 floats with a per-16-block fp8 scale and W4A8's are
        // unsigned indices into a non-uniform Lloyd-Max codebook — one implementation for
        // every family that ships them (`quant_weight.zig`).
        //
        // ⚠️ **W4A8 is here because being krea2-only made the FIRST Anima W4A8 checkpoint
        // unloadable**, and nothing about Z-Image would have stopped it arriving here
        // instead. `.w4a8` is absent from `gpuLinKindSupported`, so such a checkpoint runs
        // on the CPU (where `ops.matmul` decodes per k-slice) and every GPU arm declines
        // rather than reading the nibbles as something else.
        if (try quant_weight.nvfp4(l.alloc, l.store, nm, rows, cols)) |nv| {
            var w = nv;
            w.tag = try l.alloc.dupe(u8, nm);
            return w;
        }
        if (try quant_weight.w4a8(l.alloc, l.store, nm, rows, cols)) |q| {
            var w = q;
            w.tag = try l.alloc.dupe(u8, nm);
            return w;
        }
        if (shape.len != 2 or shape[0] != rows or shape[1] != cols) {
            // Name the tensor and both shapes: a bare ShapeMismatch across ~450
            // weights is not actionable, and the usual cause is a container whose
            // dim order differs.
            std.log.err("zimage: {s} has shape {any} ({t}), expected [{d}, {d}]", .{ nm, shape, view.info.dtype, rows, cols });
            return error.ShapeMismatch;
        }
        // A shape-fixed block-quantized tensor blocks over the FLAT element
        // sequence rather than each logical row, which `Weight.init` does not
        // assume. Refuse loudly rather than read a wrong byte count.
        if (view.info.flat_blocks) {
            std.log.err("zimage: {s} is {t} with flat block layout; this loader needs row-aligned blocks", .{ nm, view.info.dtype });
            return error.UnsupportedCheckpoint;
        }
        if (!ops.matmul.supportsDType(view.info.dtype)) {
            std.log.err("zimage: {s} has unsupported dtype {t}", .{ nm, view.info.dtype });
            return error.UnsupportedDType;
        }
        var w = Weight.init(view.bytes, view.info.dtype, rows, cols);
        // Carry the checkpoint name so a GEMM stays attributable to a layer
        // downstream (ops.matmul.probe, profiling, error messages).
        w.tag = try l.alloc.dupe(u8, nm);
        return w;
    }

    fn vec(l: Loader, comptime fmt: []const u8, args: anytype, len: usize) ![]f32 {
        var buf: [192]u8 = undefined;
        const nm = try l.name(&buf, fmt, args, "");
        const view = l.store.get(nm) orelse {
            std.log.err("zimage: missing tensor {s}", .{nm});
            return error.MissingTensor;
        };
        if (view.info.elemCount() != len) {
            std.log.err("zimage: {s} has {d} elements, expected {d}", .{ nm, view.info.elemCount(), len });
            return error.ShapeMismatch;
        }
        return view.toF32Alloc(l.alloc);
    }

    fn linear(l: Loader, comptime prefix: []const u8, rows: usize, cols: usize, bias: bool) !LinearW {
        return .{
            .w = try l.mat(prefix ++ ".weight", .{}, rows, cols),
            .b = if (bias) try l.vec(prefix ++ ".bias", .{}, rows) else null,
        };
    }

    fn block(l: Loader, comptime prefix: []const u8, args: anytype, modulated: bool) !Block {
        const cfg = l.cfg;
        return .{
            .attn = .{
                .qkv = try l.mat(prefix ++ ".attention.qkv.weight", args, cfg.qDim() + 2 * cfg.kvDim(), cfg.dim),
                .out = try l.mat(prefix ++ ".attention.out.weight", args, cfg.dim, cfg.qDim()),
                .qnorm = try l.vec(prefix ++ ".attention.q_norm.weight", args, cfg.head_dim),
                .knorm = try l.vec(prefix ++ ".attention.k_norm.weight", args, cfg.head_dim),
            },
            .ffn = .{
                .w1 = try l.mat(prefix ++ ".feed_forward.w1.weight", args, cfg.mlp_dim, cfg.dim),
                .w3 = try l.mat(prefix ++ ".feed_forward.w3.weight", args, cfg.mlp_dim, cfg.dim),
                .w2 = try l.mat(prefix ++ ".feed_forward.w2.weight", args, cfg.dim, cfg.mlp_dim),
            },
            .attn_norm1 = try l.vec(prefix ++ ".attention_norm1.weight", args, cfg.dim),
            .attn_norm2 = try l.vec(prefix ++ ".attention_norm2.weight", args, cfg.dim),
            .ffn_norm1 = try l.vec(prefix ++ ".ffn_norm1.weight", args, cfg.dim),
            .ffn_norm2 = try l.vec(prefix ++ ".ffn_norm2.weight", args, cfg.dim),
            .ada = if (modulated) try l.adaLinear(prefix ++ ".adaLN_modulation.0", args) else null,
        };
    }

    fn adaLinear(l: Loader, comptime prefix: []const u8, args: anytype) !LinearW {
        const cfg = l.cfg;
        return .{
            .w = try l.mat(prefix ++ ".weight", args, 4 * cfg.dim, cfg.mod_dim),
            .b = try l.vec(prefix ++ ".bias", args, 4 * cfg.dim),
        };
    }
};

// --- tests ------------------------------------------------------------------

const testing = std.testing;
const test_gate = @import("../test_gate.zig");

const ref_path = "src/models/assets/zimage_ref.safetensors";
const zit_ckpt = "/home/qt/genai/comfyui/models/checkpoints/zit/unstableRevolution_V2Fp16.safetensors";
/// How many trunk blocks the fp32 reference keeps — see the generator's docstring.
/// A full 30-layer comparison would need a 24.6 GB fp32 torch model.
const ref_layers = 8;

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

test "patchify and unpatchify round-trip with the (ph, pw, c) feature order" {
    // Two things at once, because the second is the one that bites: the round trip
    // is exact, AND the intermediate really does put the channel fastest. A
    // (c, ph, pw) implementation round-trips just as cleanly, so only the
    // element-order assert below distinguishes them.
    const gpa = std.testing.allocator;
    const cfg: Config = .{ .dim = 8, .n_layers = 1, .n_refiner_layers = 1, .n_heads = 1, .n_kv_heads = 1, .head_dim = 8, .mlp_dim = 8, .cap_dim = 8, .patch = 2, .channels = 3, .mod_dim = 4, .t_hidden = 4, .t_freq = 4, .rope_theta = 256.0, .rope_axes = .{ 2, 4, 2 }, .pad_multiple = 4, .time_scale = 1000.0, .norm_eps = 1e-5, .qk_eps = 1e-7, .final_eps = 1e-6 };

    const lat_h = 4;
    const lat_w = 2;
    var lat: [3 * 4 * 2]f32 = undefined;
    for (&lat, 0..) |*v, i| v.* = @floatFromInt(i);

    const patches = try patchify(gpa, cfg, &lat, lat_h, lat_w);
    defer gpa.free(patches);

    // Token (0,0) covers latent rows 0..1, cols 0..1. Channel c contributes
    // lat[c*8 + row*2 + col]. With (ph, pw, c) ordering the token reads
    // [c0(0,0), c1(0,0), c2(0,0), c0(0,1), ...] — channel-major inside each pixel.
    try std.testing.expectEqualSlices(f32, &.{ 0, 8, 16, 1, 9, 17, 2, 10, 18, 3, 11, 19 }, patches[0..12]);

    var back: [3 * 4 * 2]f32 = undefined;
    unpatchify(cfg, &back, patches, lat_h, lat_w);
    // `unpatchify` also applies NextDiT's output negation, so the round trip is -x.
    for (lat, back) |want, got| try std.testing.expectEqual(-want, got);
}

test "modulate has no shift and gatedAdd broadcasts over rows" {
    var x = [_]f32{ 1, 2, 3, 4 }; // 2 rows, dim 2
    modulate(&x, &.{ 0.5, -1.0 });
    try std.testing.expectEqualSlices(f32, &.{ 1.5, 0, 4.5, 0 }, &x);
    gatedAdd(&x, &.{ 1, 1, 2, 2 }, &.{ 2, 0.5 });
    try std.testing.expectEqualSlices(f32, &.{ 3.5, 0.5, 8.5, 1 }, &x);
}

test "padded rounds up to the pad-token multiple" {
    const cfg = z_image;
    try std.testing.expectEqual(@as(usize, 32), cfg.padded(1));
    try std.testing.expectEqual(@as(usize, 32), cfg.padded(16));
    try std.testing.expectEqual(@as(usize, 32), cfg.padded(32));
    try std.testing.expectEqual(@as(usize, 64), cfg.padded(33));
    try std.testing.expectEqual(@as(usize, 64), cfg.padded(36));
    try std.testing.expectEqual(@as(usize, 256), cfg.padded(256));
}

test "the Z-Image DiT matches ComfyUI's NextDiT on a real checkpoint" {
    // ⚠️ Compared stage by stage rather than end to end, because the four stages
    // fail for genuinely different reasons: the text half is a naming/eps problem,
    // the timestep vector is the `1 - sigma` and `time_scale` conventions, the image
    // half is patch order and pad tokens, and the trunk is the block form. A single
    // output comparison would say only "wrong".
    //
    // The reference keeps `ref_layers` of the 30 trunk blocks (see the generator),
    // so the model is loaded at that depth. What this therefore does not check is
    // the loop bound itself — the end-to-end render comparison covers that.
    const gpa = testing.allocator;
    const io = testing.io;
    try test_gate.requireIntegration();
    try test_gate.requireModelFile(io, zit_ckpt);
    try test_gate.requireModelFile(io, ref_path);

    var ref = try SafeTensors.open(gpa, io, ref_path);
    defer ref.deinit();
    var ck = try SafeTensors.open(gpa, io, zit_ckpt);
    defer ck.deinit();

    var cfg = z_image;
    cfg.n_layers = ref_layers;
    var model = try DiT.load(gpa, .{ .safetensors = &ck }, cfg);
    defer model.deinit();

    for (0..2) |ci| {
        var kb: [64]u8 = undefined;
        // I32 in the fixture; `toF32Alloc` deliberately refuses integer dtypes
        // (silently floating token ids is never right), so read the bytes.
        const idx_v = (try ref.require(try std.fmt.bufPrint(&kb, "dit.{d}.cond_index", .{ci}))).bytes;
        const ctx_i: usize = @intCast(std.mem.readInt(i32, idx_v[0..4], .little));

        const ctx = try (try ref.require(try std.fmt.bufPrint(&kb, "te.cond.{d}", .{ctx_i}))).toF32Alloc(gpa);
        defer gpa.free(ctx);
        const seq_txt = ctx.len / cfg.cap_dim;

        const sig = try (try ref.require(try std.fmt.bufPrint(&kb, "dit.{d}.sigma", .{ci}))).toF32Alloc(gpa);
        defer gpa.free(sig);
        const x_lat = try (try ref.require(try std.fmt.bufPrint(&kb, "dit.{d}.x", .{ci}))).toF32Alloc(gpa);
        defer gpa.free(x_lat);
        const lat = std.math.sqrt(x_lat.len / cfg.channels);
        try testing.expectEqual(x_lat.len, cfg.channels * lat * lat);

        // --- the timestep vector -------------------------------------------
        const adaln = try model.adalnInput(io, gpa, sig[0]);
        defer gpa.free(adaln);
        if (ci == 0) {
            const want = try (try ref.require("dit.0.dit.t_emb")).toF32Alloc(gpa);
            defer gpa.free(want);
            const rel = relL2(want, adaln);
            errdefer std.debug.print("t_emb rel L2 {e:.4}\n", .{rel});
            // Measured 1.5e-4, which IS the f16 storage floor of the fixture — the
            // three stage bounds below all sit on it, so they are checking "exact
            // up to how the fixture is stored", not a real disagreement.
            try testing.expect(rel < 1e-3);
        }

        // --- the text half ---------------------------------------------------
        const cap = try model.capTokens(io, gpa, ctx, seq_txt);
        defer gpa.free(cap);
        const cap_padded = cfg.padded(seq_txt);
        if (ci == 0) {
            const want = try (try ref.require("dit.0.dit.ctx_refiner.1")).toF32Alloc(gpa);
            defer gpa.free(want);
            try testing.expectEqual(want.len, cap.len); // pins the pad-to-32
            const rel = relL2(want, cap);
            errdefer std.debug.print("context_refiner rel L2 {e:.4}\n", .{rel});
            try testing.expect(rel < 1e-3); // measured 2.3e-4
        }

        // --- the image half ---------------------------------------------------
        var freqs = try model.ropeFreqs(gpa, cap_padded, lat / cfg.patch, lat / cfg.patch);
        defer freqs.deinit(gpa);
        const img_freqs: ops.rope.Freqs = .{
            .cos = freqs.cos[cap_padded * freqs.half ..],
            .sin = freqs.sin[cap_padded * freqs.half ..],
            .half = freqs.half,
        };
        const img = try model.noiseTokens(io, gpa, x_lat, lat, lat, adaln, img_freqs);
        defer gpa.free(img);
        if (ci == 0) {
            const want = try (try ref.require("dit.0.dit.noise_refiner.1")).toF32Alloc(gpa);
            defer gpa.free(want);
            try testing.expectEqual(want.len, img.len); // pins the pad-to-32 (36 -> 64)
            const rel = relL2(want, img);
            errdefer std.debug.print("noise_refiner rel L2 {e:.4}\n", .{rel});
            try testing.expect(rel < 1e-3); // measured 2.0e-4
        }

        // --- the whole forward -------------------------------------------------
        const got = try gpa.alloc(f32, x_lat.len);
        defer gpa.free(got);
        try model.predict(io, gpa, got, x_lat, lat, lat, cap, cap_padded, adaln, null);

        const want = try (try ref.require(try std.fmt.bufPrint(&kb, "dit.{d}.out", .{ci}))).toF32Alloc(gpa);
        defer gpa.free(want);
        const rel = relL2(want, got);
        errdefer std.debug.print("case {d} (latent {d}): velocity rel L2 {e:.4}\n", .{ ci, lat, rel });
        // Stored f32 and computed from f32 weights on both sides, so the only
        // difference is reduction order and the fused rms+rope kernel ComfyUI uses:
        // measured **3.5e-6 at 8 layers** (and 1.6e-6 at 2), i.e. the disagreement
        // grows roughly LINEARLY with depth rather than compounding — which is what
        // says the block is right and not merely close. Three orders below the f16
        // stage bounds above.
        try testing.expect(rel < 2e-5);
    }
}

test "the Z-Image DiT loads every weight with a tag and the right shapes" {
    // Cheap structural check: it needs only the checkpoint, and a missing or
    // mis-shaped tensor here is a much clearer failure than the same problem
    // surfacing as a numeric mismatch in the parity test above.
    const gpa = testing.allocator;
    const io = testing.io;
    try test_gate.requireIntegration();
    try test_gate.requireModelFile(io, zit_ckpt);

    var ck = try SafeTensors.open(gpa, io, zit_ckpt);
    defer ck.deinit();
    var cfg = z_image;
    cfg.n_layers = ref_layers;
    var model = try DiT.load(gpa, .{ .safetensors = &ck }, cfg);
    defer model.deinit();

    try testing.expectEqual(@as(usize, ref_layers), model.layers.len);
    try testing.expectEqual(@as(usize, 2), model.context_refiner.len);
    try testing.expectEqual(@as(usize, 2), model.noise_refiner.len);

    // context_refiner blocks are `modulation=False` — no AdaLN linear at all. If
    // one were loaded the block would silently modulate the text half.
    for (model.context_refiner) |b| try testing.expect(b.ada == null);
    for (model.noise_refiner) |b| try testing.expect(b.ada != null);
    for (model.layers) |b| try testing.expect(b.ada != null);

    const b0 = model.layers[0];
    try testing.expectEqualStrings("model.diffusion_model.layers.0.attention.qkv.weight", b0.attn.qkv.tag.?);
    try testing.expectEqual(@as(usize, 3 * 3840), b0.attn.qkv.rows);
    try testing.expectEqual(@as(usize, 4 * 3840), b0.ada.?.w.rows);
    try testing.expectEqual(@as(usize, 256), b0.ada.?.w.cols);
    // Distinct per block, or per-layer attribution would collapse them together.
    try testing.expectEqualStrings("model.diffusion_model.layers.1.feed_forward.w2.weight", model.layers[1].ffn.w2.tag.?);
    // The two projections materialized to f32 at load must keep their tags.
    try testing.expectEqualStrings("model.diffusion_model.x_embedder.weight", model.x_embedder.w.tag.?);
    try testing.expectEqualStrings("model.diffusion_model.final_layer.linear.weight", model.final_linear.w.tag.?);
}
