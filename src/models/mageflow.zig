//! Mage-Flow, Microsoft's native-resolution MMDiT, mirroring
//! comfy/ldm/mage_flow/model.py.
//!
//! DOUBLE-stream, unlike every other family here: text and image tokens keep
//! their own modulation, norms, projections and MLP through all 12 blocks and
//! meet only inside the attention, where they are concatenated as
//! `[text | image]`, attended jointly, and split again. The block is
//! Qwen-Image's; what Mage changes is around it, and each change is a silent
//! wrong answer on its own:
//!
//!   - `patch_size = 1`. A token IS one latent pixel, all 128 channels of it,
//!     so there is no 2x2 packing and no patch feature order to get wrong.
//!   - Text tokens are NOT rotated. Their RoPE ids are all zero, which is the
//!     identity rotation, so only the image half carries position.
//!   - Positions are centered on `[-ceil(n/2), floor(n/2))`, i.e. offset by
//!     `n - n/2`. Qwen-Image offsets by `n/2`; the two agree only for even n.
//!   - The timestep frequency table is rounded to bf16 before it multiplies
//!     the timestep, and so is the timestep. The model was trained in bf16 and
//!     ComfyUI keeps that rounding even on an fp32 device.
//!
//! Reference images (Mage-Flow-Edit) are more image tokens appended to the
//! same stream, told apart only by RoPE axis 0 carrying the image index
//! (0 = the canvas, 1..N = references). Nothing else in the forward knows
//! they exist; the output is simply truncated back to the canvas tokens.
//!
//! Loads from any `WeightStore`, safetensors or GGUF. Large weights keep their
//! checkpoint dtype and dequantize inside the GEMM; norm scales and biases
//! become f32 at load. The store's mapping must outlive the model.

const std = @import("std");
const tp_core = @import("tp_core");
const safetensors = tp_core.safetensors;
const weights_mod = tp_core.weights;
const ops = @import("tp_ops");
const quant_weight = @import("quant_weight.zig");

const SafeTensors = safetensors.SafeTensors;
const WeightStore = weights_mod.WeightStore;
const Weight = ops.matmul.Weight;

pub const channels = 128;
pub const n_heads = 24;
pub const head_dim = 128;
pub const features = n_heads * head_dim; // 3072
/// Blocks the released 4B checkpoint carries. The loader counts them rather than
/// assuming it, both because ComfyUI does and because the parity fixture is a
/// truncated copy of the same file.
pub const n_blocks = 12;
pub const mlp_dim = 4 * features; // 12288
/// Text-encoder width the `txt_in` projection consumes (Qwen3-VL-4B: 2560).
pub const txt_dim = 2560;
/// Sinusoidal frequency count feeding the timestep MLP.
pub const tdim = 256;
pub const rope_theta: f64 = 10000.0;
pub const rope_axes = [3]usize{ 16, 56, 56 };
/// One latent pixel per token: 16 px of image per token, via the VAE's own 16x.
pub const patch = 1;
/// Upper bound on the block count a checkpoint may declare, so `depthIn`'s probe
/// loop terminates on a file that is not this architecture at all.
pub const max_blocks = 64;

/// Every norm in this model, learned or not, uses the same epsilon.
const eps: f32 = 1e-6;

/// The tensor that says "this is a Mage-Flow denoiser". `txt_norm` is a bare
/// RMSNorm over 2560 at the top level, which no other family here carries.
pub const probe = "txt_norm.weight";

pub const LinearW = struct {
    w: Weight,
    b: ?[]const f32,
};

/// Joint attention. The two streams have separate projections and separate
/// QK-norms; only the softmax is shared.
pub const Attn = struct {
    q: LinearW,
    k: LinearW,
    v: LinearW,
    o: LinearW,
    txt_q: LinearW,
    txt_k: LinearW,
    txt_v: LinearW,
    txt_o: LinearW,
    qnorm: []const f32, // [head_dim]
    knorm: []const f32,
    txt_qnorm: []const f32,
    txt_knorm: []const f32,
};

/// GELU(tanh) MLP, 4x inner width.
pub const Mlp = struct {
    in: LinearW, // features -> mlp_dim
    out: LinearW, // mlp_dim -> features
};

pub const Block = struct {
    /// -> 6 * features: [shift1, scale1, gate1, shift2, scale2, gate2]. Takes
    /// `silu(temb)` on a dense checkpoint and the rank-r bottleneck on a
    /// compressed one; see `DiT.mod_down`.
    img_mod: LinearW,
    txt_mod: LinearW,
    attn: Attn,
    img_mlp: Mlp,
    txt_mlp: Mlp,
};

/// One reference image's latent, planar `[channels][h][w]`.
pub const Ref = struct {
    lat: []const f32,
    h: usize,
    w: usize,
};

pub const DiT = struct {
    arena: std.heap.ArenaAllocator,
    /// `channels` -> `features`; a token is one latent pixel.
    img_in: LinearW,
    /// Plain RMSNorm weight over the encoder states, applied before `txt_in`.
    txt_norm: []const f32,
    txt_in: LinearW, // txt_dim -> features
    tmlp0: LinearW, // tdim -> features
    tmlp2: LinearW, // features -> features
    blocks: []Block,
    /// Every block linear the device forwards run, the one list every support
    /// scan and GEMM plan reads (`lin`, `lin_cuda`).
    device_lins: []const Weight,
    /// The shared low-rank modulation down-projection of a COMPRESSED checkpoint
    /// (`features -> rank`), null on a dense one. See `rankIn`.
    mod_down: ?LinearW,
    /// silu(temb) -> 2 * features, in [scale, shift] order (NOT the blocks').
    /// Never compressed: only the `features -> 6 * features` block linears are.
    last_mod: LinearW,
    proj_out: LinearW, // features -> channels

    /// The prefix this store spells the denoiser with, resolved against the file
    /// rather than assumed: a bundled checkpoint nests it, a bare export does not.
    pub fn prefixIn(store: WeightStore) []const u8 {
        return if (store.get("model.diffusion_model." ++ probe) != null) "model.diffusion_model." else "";
    }

    /// Rank of the shared low-rank modulation, 0 on a dense checkpoint.
    ///
    /// A compressed checkpoint ("mageflow-lowrank-modulation-v1") replaces every
    /// block's `[6 * features][features]` modulation linear with a head onto a
    /// rank-r bottleneck that one shared `modulation_down` produces. Nothing else
    /// about the architecture moves, so this is the only thing to detect.
    ///
    /// ⚠️ The SiLU moves with it. The reference replaces each block's
    /// `Sequential(SiLU, Linear)` first element with `Identity` and applies SiLU
    /// ONCE, before the down-projection. Applying it per block as well is a
    /// silent wrong answer.
    pub fn rankIn(store: WeightStore) usize {
        var buf: [96]u8 = undefined;
        const nm = std.fmt.bufPrint(&buf, "{s}modulation_down.weight", .{prefixIn(store)}) catch return 0;
        const view = store.get(nm) orelse return 0;
        return view.info.elemCount() / features;
    }

    /// Blocks this store carries, ComfyUI's `count_blocks`.
    pub fn depthIn(store: WeightStore) usize {
        const pfx = prefixIn(store);
        var n: usize = 0;
        while (n < max_blocks) : (n += 1) {
            var buf: [96]u8 = undefined;
            const nm = std.fmt.bufPrint(&buf, "{s}transformer_blocks.{d}.img_mod.1.weight", .{ pfx, n }) catch return n;
            if (store.get(nm) == null) return n;
        }
        return n;
    }

    pub fn load(gpa: std.mem.Allocator, store: WeightStore) !DiT {
        return loadDepth(gpa, store, depthIn(store));
    }

    /// Load only the first `n_layers` blocks. The parity fixture is a truncated
    /// copy of the real checkpoint (an fp32 12-block reference does not fit in
    /// RAM), so the test loads to the same depth.
    pub fn loadDepth(gpa: std.mem.Allocator, store: WeightStore, n_layers: usize) !DiT {
        if (n_layers == 0 or n_layers > max_blocks) return error.UnknownModelConfig;
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();

        const rank = rankIn(store);
        const l = Loader{ .store = store, .alloc = alloc, .pfx = prefixIn(store), .mod_cols = if (rank == 0) features else rank };

        const blocks = try alloc.alloc(Block, n_layers);
        for (blocks, 0..) |*blk, i| blk.* = try l.loadBlock(i);
        const mod_down: ?LinearW = if (rank == 0) null else try l.linear("modulation_down", rank, features);

        const per_block = blockLins(&blocks[0]).len;
        const device_lins = try alloc.alloc(Weight, blocks.len * per_block);
        for (blocks, 0..) |*b, i| device_lins[i * per_block ..][0..per_block].* = blockLins(b);

        // Every field into a local first: `.arena = arena` in a struct literal
        // copies the arena's state, so anything a later field allocates is lost.
        // `img_in` and `proj_out` are the two weights the device arm feeds to the
        // fused `opMatmul`, which has only an f32 pipeline and is the one GEMM
        // here that folds a bias. Materializing them at load costs 3 MB and
        // keeps every checkpoint dtype uniform; reading bf16 bytes as f32 there
        // is pure noise rather than an error. The CPU path is unaffected: the
        // conversion is exact.
        var img_in = try l.linear("img_in", features, channels);
        img_in.w = try ops.matmul.materializeF32(alloc, img_in.w);
        const txt_norm = try l.vec("txt_norm.weight", .{}, txt_dim);
        const txt_in = try l.linear("txt_in", features, txt_dim);
        const tmlp0 = try l.linear("time_text_embed.timestep_embedder.linear_1", features, tdim);
        const tmlp2 = try l.linear("time_text_embed.timestep_embedder.linear_2", features, features);
        const last_mod = try l.linear("norm_out.linear", 2 * features, features);
        var proj_out = try l.linear("proj_out", channels, features);
        proj_out.w = try ops.matmul.materializeF32(alloc, proj_out.w);

        return .{
            .arena = arena,
            .img_in = img_in,
            .txt_norm = txt_norm,
            .txt_in = txt_in,
            .tmlp0 = tmlp0,
            .tmlp2 = tmlp2,
            .blocks = blocks,
            .device_lins = device_lins,
            .mod_down = mod_down,
            .last_mod = last_mod,
            .proj_out = proj_out,
        };
    }

    pub fn deinit(self: *DiT) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// Predict velocity for one latent. `x_lat`/`out` are planar
    /// `[channels][lat_h][lat_w]`; `ctx` is `[seq_txt][txt_dim]` (post-strip
    /// encoder states); `sigma` is the flow-matching timestep; `refs` are the
    /// edit path's reference latents, empty for text-to-image.
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
        refs: []const Ref,
        cancel: ?*std.atomic.Value(bool),
    ) !void {
        std.debug.assert(x_lat.len == channels * lat_h * lat_w);
        std.debug.assert(out.len == x_lat.len);
        std.debug.assert(ctx.len == seq_txt * txt_dim);

        // Arm fine-grained cancel inside the CPU kernels for the whole forward:
        // one MLP GEMM here is seconds of work, so per-block polling alone
        // leaves a multi-second cancel latency.
        const prev_tok = ops.cancel.token;
        ops.cancel.token = cancel;
        defer ops.cancel.token = prev_tok;

        const n_img = lat_h * lat_w;
        var n_tok = n_img;
        for (refs) |r| n_tok += r.h * r.w;
        const seq = seq_txt + n_tok;

        const temb = try self.timestepVector(io, gpa, sigma);
        defer gpa.free(temb);
        const mod_in = try self.blockModInput(io, gpa, temb);
        defer gpa.free(mod_in);

        const x = try gpa.alloc(f32, seq * features);
        defer gpa.free(x);

        // Text half: RMSNorm with the checkpoint's plain weight, then project.
        {
            const normed = try gpa.alloc(f32, ctx.len);
            defer gpa.free(normed);
            ops.norm.rmsNorm(normed, ctx, self.txt_norm, eps);
            try linear(io, gpa, x[0 .. seq_txt * features], normed, seq_txt, self.txt_in);
        }

        // Image half: the canvas, then each reference, each token one latent
        // pixel with its channels as the feature vector.
        {
            const tokens = try gpa.alloc(f32, n_tok * channels);
            defer gpa.free(tokens);
            interleaveChannels(tokens[0 .. n_img * channels], x_lat, n_img);
            var at = n_img * channels;
            for (refs) |r| {
                const n = r.h * r.w;
                interleaveChannels(tokens[at..][0 .. n * channels], r.lat, n);
                at += n * channels;
            }
            try linear(io, gpa, x[seq_txt * features ..], tokens, n_tok, self.img_in);
        }

        var freqs = try self.ropeFreqs(gpa, lat_h, lat_w, seq_txt, refs);
        defer freqs.deinit(gpa);

        for (self.blocks) |*blk| {
            // Poll between blocks so a stop lands mid-step; a full CPU step at
            // this width is tens of seconds.
            if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
            try blockForward(io, gpa, blk, x, seq_txt, n_tok, mod_in, freqs);
        }

        // Final layer. It is row-wise, so running it on the canvas rows alone
        // is what ComfyUI's "whole stream, then slice to num_embeds" computes.
        const img_rows = x[seq_txt * features ..][0 .. n_img * features];
        {
            const mod = try gpa.alloc(f32, 2 * features);
            defer gpa.free(mod);
            try modLinear(io, gpa, mod, temb, self.last_mod);
            const scale = mod[0..features];
            const shift = mod[features..][0..features];
            ops.norm.layerNormUnit(img_rows, img_rows, features, eps);
            modulate(img_rows, scale, shift);
        }
        const final = try gpa.alloc(f32, n_img * channels);
        defer gpa.free(final);
        try linear(io, gpa, final, img_rows, n_img, self.proj_out);
        deinterleaveChannels(out, final, n_img);
    }

    /// `temb = linear_2(silu(linear_1(sinusoid(sigma))))`. Caller frees.
    pub fn timestepVector(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, sigma: f32) ![]f32 {
        var freq: [tdim]f32 = undefined;
        timestepEmbedding(&freq, sigma);
        const t0 = try gpa.alloc(f32, features);
        defer gpa.free(t0);
        try linear(io, gpa, t0, &freq, 1, self.tmlp0);
        ops.act.silu(t0);
        const t = try gpa.alloc(f32, features);
        errdefer gpa.free(t);
        try linear(io, gpa, t, t0, 1, self.tmlp2);
        return t;
    }

    /// The text half through `txt_norm` and `txt_in`: `[seq_txt][features]`.
    /// Caller frees. One RMSNorm and one GEMM over ~100 rows, so the device arm
    /// runs it here too, once per conditioning rather than once per step.
    pub fn textTokens(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, ctx: []const f32, seq_txt: usize) ![]f32 {
        std.debug.assert(ctx.len == seq_txt * txt_dim);
        const normed = try gpa.alloc(f32, ctx.len);
        defer gpa.free(normed);
        ops.norm.rmsNorm(normed, ctx, self.txt_norm, eps);
        const out = try gpa.alloc(f32, seq_txt * features);
        errdefer gpa.free(out);
        try linear(io, gpa, out, normed, seq_txt, self.txt_in);
        return out;
    }

    /// The whole modulation table for one timestep, laid out for the device
    /// AdaLN kernels. Caller frees.
    ///
    /// `[n_blocks][12][features]` then `[2][features]`: per block the image
    /// stream's `premul1, shift1, gate1, premul2, shift2, gate2` and then the
    /// text stream's six, and at the end the final layer's `premul, shift`.
    ///
    /// `premul` is `1 + scale`, FOLDED, because `lnMod` multiplies by one vector
    /// and adds another and has no place for the `1 +`. The chunk ORDER also
    /// differs from the checkpoint's: a block's linear emits `shift, scale, gate`
    /// and the final layer's emits `scale, shift`, so both are rearranged here,
    /// once, rather than at two call sites that could disagree.
    ///
    /// This runs on the host on purpose. On a dense checkpoint the two modulation
    /// linears are `[6 * features][features]` each, 1.4 G parameters over the
    /// trunk, a THIRD of the model, and they depend only on the timestep: building
    /// the table once per sigma keeps 2.7 GB of weights off the device entirely
    /// and saves 24 m=1 GEMVs a forward, which is the worst shape a GEMM has.
    pub fn modulationTable(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, sigma: f32) ![]f32 {
        const temb = try self.timestepVector(io, gpa, sigma);
        defer gpa.free(temb);
        const act = try gpa.alloc(f32, features);
        defer gpa.free(act);
        @memcpy(act, temb);
        ops.act.silu(act);
        const mod_in = try self.blockModInput(io, gpa, temb);
        defer gpa.free(mod_in);

        const out = try gpa.alloc(f32, self.modStride());
        errdefer gpa.free(out);
        const scratch = try gpa.alloc(f32, 6 * features);
        defer gpa.free(scratch);

        for (self.blocks, 0..) |*blk, i| {
            for ([2]LinearW{ blk.img_mod, blk.txt_mod }, 0..) |lw, s| {
                try linear(io, gpa, scratch, mod_in, 1, lw);
                const dst = out[(i * 12 + s * 6) * features ..];
                for (0..2) |half_i| {
                    const src = scratch[half_i * 3 * features ..];
                    const d = dst[half_i * 3 * features ..];
                    for (0..features) |j| d[0 * features + j] = 1.0 + src[1 * features + j]; // premul
                    @memcpy(d[1 * features ..][0..features], src[0 * features ..][0..features]); // shift
                    @memcpy(d[2 * features ..][0..features], src[2 * features ..][0..features]); // gate
                }
            }
        }
        {
            // The final layer's linear emits SCALE first and shift second, the
            // opposite of a block's; `chunk` holds it while it is rearranged.
            try linear(io, gpa, scratch[0 .. 2 * features], act, 1, self.last_mod);
            const fin = out[self.blocks.len * 12 * features ..][0 .. 2 * features];
            for (0..features) |j| {
                fin[j] = 1.0 + scratch[j];
                fin[features + j] = scratch[features + j];
            }
        }
        return out;
    }

    /// What a BLOCK's modulation linear reads: `silu(temb)` on a dense
    /// checkpoint, and the shared low-rank projection of that on a compressed
    /// one. Caller frees.
    ///
    /// One function because it has two consumers that must not drift: the CPU
    /// forward and `modulationTable`, which is what the device arms upload.
    /// `norm_out` is NOT a consumer, it is never compressed and reads `silu(temb)`.
    pub fn blockModInput(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, temb: []const f32) ![]f32 {
        const act = try gpa.alloc(f32, features);
        errdefer gpa.free(act);
        @memcpy(act, temb);
        ops.act.silu(act);
        const md = self.mod_down orelse return act;
        defer gpa.free(act);
        const down = try gpa.alloc(f32, md.w.rows);
        errdefer gpa.free(down);
        try linear(io, gpa, down, act, 1, md);
        return down;
    }

    /// Floats one timestep's `modulationTable` occupies.
    pub fn modStride(self: *const DiT) usize {
        return self.blocks.len * 12 * features + 2 * features;
    }

    /// Joint RoPE table: text at (0,0,0), canvas at (0, row, col) centered, each
    /// reference at (index, row, col) centered on its own size.
    pub fn ropeFreqs(
        self: *const DiT,
        gpa: std.mem.Allocator,
        lat_h: usize,
        lat_w: usize,
        seq_txt: usize,
        refs: []const Ref,
    ) !ops.rope.Freqs {
        _ = self;
        var n_tok = lat_h * lat_w;
        for (refs) |r| n_tok += r.h * r.w;
        const pos = try gpa.alloc(f32, (seq_txt + n_tok) * 3);
        defer gpa.free(pos);
        @memset(pos[0 .. seq_txt * 3], 0);

        var at = seq_txt;
        writePositions(pos, at, 0, lat_h, lat_w);
        at += lat_h * lat_w;
        for (refs, 0..) |r, i| {
            writePositions(pos, at, i + 1, r.h, r.w);
            at += r.h * r.w;
        }
        return ops.rope.fluxFreqs(gpa, pos, &rope_axes, rope_theta);
    }
};

/// One image's ids into `pos` at token `at`: frame index on axis 0, and row /
/// column centered so the positions run `[-ceil(n/2), floor(n/2))`.
fn writePositions(pos: []f32, at: usize, index: usize, h: usize, w: usize) void {
    const h_off: f32 = @floatFromInt(h - h / 2);
    const w_off: f32 = @floatFromInt(w - w / 2);
    const idx: f32 = @floatFromInt(index);
    for (0..h) |hi| {
        for (0..w) |wi| {
            const base = (at + hi * w + wi) * 3;
            pos[base] = idx;
            pos[base + 1] = @as(f32, @floatFromInt(hi)) - h_off;
            pos[base + 2] = @as(f32, @floatFromInt(wi)) - w_off;
        }
    }
}

/// Mage's timestep frequencies: `exp(-ln(10000) * i / 128)` ROUNDED TO BF16,
/// times a bf16-rounded timestep, times 1000, laid out `[cos.., sin..]`.
///
/// The rounding is not incidental. ComfyUI's `MageFlow.process_timestep` casts
/// the timestep to bf16 on every device, and the table is built at the
/// timestep's dtype, so an f32 run that skips both is conditioning the model on
/// a timestep it never saw in training.
fn timestepEmbedding(out: *[tdim]f32, sigma: f32) void {
    const half = tdim / 2;
    const dtypes = tp_core.dtype;
    const t = dtypes.bf16ToF32(dtypes.f32ToBf16(sigma));
    for (0..half) |i| {
        const exponent = -@log(@as(f64, 10000.0)) * @as(f64, @floatFromInt(i)) / @as(f64, half);
        const w = dtypes.bf16ToF32(dtypes.f32ToBf16(@floatCast(@exp(exponent))));
        const arg = 1000.0 * (t * w);
        out[i] = @cos(arg);
        out[half + i] = @sin(arg);
    }
}

/// Planar `[channels][n]` -> token-major `[n][channels]` (`movedim(1, -1)`).
pub fn interleaveChannels(dst: []f32, src: []const f32, n: usize) void {
    std.debug.assert(dst.len == n * channels and src.len == dst.len);
    for (0..channels) |c| {
        const plane = src[c * n ..][0..n];
        for (plane, 0..) |v, i| dst[i * channels + c] = v;
    }
}

/// Token-major `[n][channels]` -> planar `[channels][n]` (`movedim(-1, 1)`).
pub fn deinterleaveChannels(dst: []f32, src: []const f32, n: usize) void {
    std.debug.assert(dst.len == n * channels and src.len == dst.len);
    for (0..channels) |c| {
        const plane = dst[c * n ..][0..n];
        for (plane, 0..) |*v, i| v.* = src[i * channels + c];
    }
}

/// `x` holds `[text | image]`; the two halves are modulated, normed and
/// projected separately and meet only inside `attnForward`.
fn blockForward(
    io: std.Io,
    gpa: std.mem.Allocator,
    blk: *const Block,
    x: []f32,
    seq_txt: usize,
    n_img: usize,
    /// `DiT.blockModInput`, already silu'd and already through the low-rank
    /// projection if this checkpoint has one.
    mod_in: []const f32,
    freqs: ops.rope.Freqs,
) !void {
    const mod = try gpa.alloc(f32, 12 * features);
    defer gpa.free(mod);
    const img_mod = mod[0 .. 6 * features];
    const txt_mod = mod[6 * features ..];
    try linear(io, gpa, img_mod, mod_in, 1, blk.img_mod);
    try linear(io, gpa, txt_mod, mod_in, 1, blk.txt_mod);

    const txt = x[0 .. seq_txt * features];
    const img = x[seq_txt * features ..][0 .. n_img * features];

    const img_n = try gpa.alloc(f32, img.len);
    defer gpa.free(img_n);
    const txt_n = try gpa.alloc(f32, txt.len);
    defer gpa.free(txt_n);

    // Attention half: norm, modulate, joint attend, gated residual.
    ops.norm.layerNormUnit(img_n, img, features, eps);
    modulate(img_n, chunk(img_mod, 1), chunk(img_mod, 0));
    ops.norm.layerNormUnit(txt_n, txt, features, eps);
    modulate(txt_n, chunk(txt_mod, 1), chunk(txt_mod, 0));

    {
        const img_a = try gpa.alloc(f32, img.len);
        defer gpa.free(img_a);
        const txt_a = try gpa.alloc(f32, txt.len);
        defer gpa.free(txt_a);
        try attnForward(io, gpa, &blk.attn, img_n, txt_n, seq_txt, n_img, freqs, img_a, txt_a);
        gatedAdd(img, img_a, chunk(img_mod, 2));
        gatedAdd(txt, txt_a, chunk(txt_mod, 2));
    }

    // MLP half.
    ops.norm.layerNormUnit(img_n, img, features, eps);
    modulate(img_n, chunk(img_mod, 4), chunk(img_mod, 3));
    ops.norm.layerNormUnit(txt_n, txt, features, eps);
    modulate(txt_n, chunk(txt_mod, 4), chunk(txt_mod, 3));

    {
        const img_m = try gpa.alloc(f32, img.len);
        defer gpa.free(img_m);
        try mlpForward(io, gpa, &blk.img_mlp, img_n, n_img, img_m);
        gatedAdd(img, img_m, chunk(img_mod, 5));
    }
    {
        const txt_m = try gpa.alloc(f32, txt.len);
        defer gpa.free(txt_m);
        try mlpForward(io, gpa, &blk.txt_mlp, txt_n, seq_txt, txt_m);
        gatedAdd(txt, txt_m, chunk(txt_mod, 5));
    }
}

/// One of the six `features`-wide pieces of a modulation vector.
fn chunk(mod: []const f32, i: usize) []const f32 {
    return mod[i * features ..][0..features];
}

/// Joint attention: `[text | image]` concatenated, rotated, attended as one
/// sequence, split back. Both streams have their own projections, so the
/// concatenation is of the PROJECTED q/k/v, not of the inputs.
fn attnForward(
    io: std.Io,
    gpa: std.mem.Allocator,
    attn: *const Attn,
    img: []const f32,
    txt: []const f32,
    seq_txt: usize,
    n_img: usize,
    freqs: ops.rope.Freqs,
    img_out: []f32,
    txt_out: []f32,
) !void {
    const seq = seq_txt + n_img;
    const qkv = try gpa.alloc(f32, 3 * seq * features);
    defer gpa.free(qkv);
    const q = qkv[0 .. seq * features];
    const k = qkv[seq * features ..][0 .. seq * features];
    const v = qkv[2 * seq * features ..];

    const txt_len = seq_txt * features;
    try linear(io, gpa, q[0..txt_len], txt, seq_txt, attn.txt_q);
    try linear(io, gpa, k[0..txt_len], txt, seq_txt, attn.txt_k);
    try linear(io, gpa, v[0..txt_len], txt, seq_txt, attn.txt_v);
    try linear(io, gpa, q[txt_len..], img, n_img, attn.q);
    try linear(io, gpa, k[txt_len..], img, n_img, attn.k);
    try linear(io, gpa, v[txt_len..], img, n_img, attn.v);

    // Per-head QK-norm, each stream with its own weight, before RoPE.
    ops.norm.rmsNorm(q[0..txt_len], q[0..txt_len], attn.txt_qnorm, eps);
    ops.norm.rmsNorm(k[0..txt_len], k[0..txt_len], attn.txt_knorm, eps);
    ops.norm.rmsNorm(q[txt_len..], q[txt_len..], attn.qnorm, eps);
    ops.norm.rmsNorm(k[txt_len..], k[txt_len..], attn.knorm, eps);

    ops.rope.applyInterleaved(q, freqs, seq, n_heads, head_dim);
    ops.rope.applyInterleaved(k, freqs, seq, n_heads, head_dim);

    const joint = try gpa.alloc(f32, seq * features);
    defer gpa.free(joint);
    try ops.attention.attention(io, gpa, joint, q, k, v, .{
        .seq_q = seq,
        .seq_kv = seq,
        .n_heads = n_heads,
        .n_kv_heads = n_heads,
        .head_dim = head_dim,
    });

    try linear(io, gpa, txt_out, joint[0..txt_len], seq_txt, attn.txt_o);
    try linear(io, gpa, img_out, joint[txt_len..], n_img, attn.o);
}

fn mlpForward(io: std.Io, gpa: std.mem.Allocator, mlp: *const Mlp, x: []const f32, rows: usize, out: []f32) !void {
    const inner = try gpa.alloc(f32, rows * mlp_dim);
    defer gpa.free(inner);
    try linear(io, gpa, inner, x, rows, mlp.in);
    ops.act.geluTanh(inner);
    try linear(io, gpa, out, inner, rows, mlp.out);
}

/// The final layer's modulation linear, the one place a `Sequential(SiLU, Linear)`
/// survives compression. A block's went through `DiT.blockModInput` instead.
fn modLinear(io: std.Io, gpa: std.mem.Allocator, out: []f32, temb: []const f32, lw: LinearW) !void {
    const act = try gpa.alloc(f32, temb.len);
    defer gpa.free(act);
    @memcpy(act, temb);
    ops.act.silu(act);
    try linear(io, gpa, out, act, 1, lw);
}

/// Row-wise AdaLN: x = (1 + scale) * x + shift.
fn modulate(x: []f32, scale: []const f32, shift: []const f32) void {
    const dim = scale.len;
    var row: usize = 0;
    while (row < x.len) : (row += dim) {
        for (x[row..][0..dim], scale, shift) |*v, sc, sh| v.* = (1.0 + sc) * v.* + sh;
    }
}

/// Row-wise gated residual: x += gate * delta.
fn gatedAdd(x: []f32, delta: []const f32, gate: []const f32) void {
    const dim = gate.len;
    var row: usize = 0;
    while (row < x.len) : (row += dim) {
        for (x[row..][0..dim], delta[row..][0..dim], gate) |*v, d, g| v.* += g * d;
    }
}

fn linear(io: std.Io, gpa: std.mem.Allocator, out: []f32, x: []const f32, m: usize, lw: LinearW) !void {
    try ops.matmul.matmul(io, gpa, out, x, m, lw.w, lw.b);
}

/// The 12 linears one block's device forward runs.
///
/// The two modulation linears are NOT here, and that is not an oversight: they
/// depend only on the timestep, so `modulationTable` evaluates them on the host
/// once per sigma. On a dense checkpoint they are a third of the parameters, so
/// keeping them off this list keeps 2.7 GB off the device.
pub fn blockLins(b: *const Block) [12]Weight {
    return .{
        b.attn.q.w,     b.attn.k.w,      b.attn.v.w,     b.attn.o.w,
        b.attn.txt_q.w, b.attn.txt_k.w,  b.attn.txt_v.w, b.attn.txt_o.w,
        b.img_mlp.in.w, b.img_mlp.out.w, b.txt_mlp.in.w, b.txt_mlp.out.w,
    };
}

// --- weight loading --------------------------------------------------------

const Loader = struct {
    store: WeightStore,
    alloc: std.mem.Allocator,
    pfx: []const u8,
    /// Input width of a block's modulation linear: `features` dense, the rank
    /// compressed.
    mod_cols: usize,

    fn name(l: Loader, buf: []u8, comptime fmt: []const u8, args: anytype) ![]u8 {
        var fbs = std.Io.Writer.fixed(buf);
        try fbs.writeAll(l.pfx);
        try fbs.print(fmt, args);
        return fbs.buffered();
    }

    fn mat(l: Loader, comptime fmt: []const u8, args: anytype, rows: usize, cols: usize) !Weight {
        var buf: [160]u8 = undefined;
        const nm = try l.name(&buf, fmt, args);
        return quant_weight.load(l.alloc, l.store, nm, rows, cols, .{ .who = "mageflow" });
    }

    fn vec(l: Loader, comptime fmt: []const u8, args: anytype, len: usize) ![]f32 {
        var buf: [160]u8 = undefined;
        const nm = try l.name(&buf, fmt, args);
        const view = l.store.get(nm) orelse return error.MissingTensor;
        if (view.info.elemCount() != len) return error.ShapeMismatch;
        return view.toF32Alloc(l.alloc);
    }

    fn linearAt(l: Loader, comptime fmt: []const u8, args: anytype, rows: usize, cols: usize) !LinearW {
        return .{
            .w = try l.mat(fmt ++ ".weight", args, rows, cols),
            .b = try l.vec(fmt ++ ".bias", args, rows),
        };
    }

    fn linear(l: Loader, comptime prefix: []const u8, rows: usize, cols: usize) !LinearW {
        return l.linearAt(prefix, .{}, rows, cols);
    }

    fn loadBlock(l: Loader, i: usize) !Block {
        return .{
            .img_mod = try l.linearAt("transformer_blocks.{d}.img_mod.1", .{i}, 6 * features, l.mod_cols),
            .txt_mod = try l.linearAt("transformer_blocks.{d}.txt_mod.1", .{i}, 6 * features, l.mod_cols),
            .attn = .{
                .q = try l.linearAt("transformer_blocks.{d}.attn.to_q", .{i}, features, features),
                .k = try l.linearAt("transformer_blocks.{d}.attn.to_k", .{i}, features, features),
                .v = try l.linearAt("transformer_blocks.{d}.attn.to_v", .{i}, features, features),
                .o = try l.linearAt("transformer_blocks.{d}.attn.to_out.0", .{i}, features, features),
                .txt_q = try l.linearAt("transformer_blocks.{d}.attn.add_q_proj", .{i}, features, features),
                .txt_k = try l.linearAt("transformer_blocks.{d}.attn.add_k_proj", .{i}, features, features),
                .txt_v = try l.linearAt("transformer_blocks.{d}.attn.add_v_proj", .{i}, features, features),
                .txt_o = try l.linearAt("transformer_blocks.{d}.attn.to_add_out", .{i}, features, features),
                .qnorm = try l.vec("transformer_blocks.{d}.attn.norm_q.weight", .{i}, head_dim),
                .knorm = try l.vec("transformer_blocks.{d}.attn.norm_k.weight", .{i}, head_dim),
                .txt_qnorm = try l.vec("transformer_blocks.{d}.attn.norm_added_q.weight", .{i}, head_dim),
                .txt_knorm = try l.vec("transformer_blocks.{d}.attn.norm_added_k.weight", .{i}, head_dim),
            },
            .img_mlp = .{
                .in = try l.linearAt("transformer_blocks.{d}.img_mlp.net.0.proj", .{i}, mlp_dim, features),
                .out = try l.linearAt("transformer_blocks.{d}.img_mlp.net.2", .{i}, features, mlp_dim),
            },
            .txt_mlp = .{
                .in = try l.linearAt("transformer_blocks.{d}.txt_mlp.net.0.proj", .{i}, mlp_dim, features),
                .out = try l.linearAt("transformer_blocks.{d}.txt_mlp.net.2", .{i}, features, mlp_dim),
            },
        };
    }
};

// --- tests -----------------------------------------------------------------

const testing = std.testing;

test "positions center on [-ceil(n/2), floor(n/2))" {
    // Odd sizes are where Mage and Qwen-Image disagree: 5 rows run -3..1, not -2..2.
    var pos: [5 * 1 * 3]f32 = undefined;
    writePositions(&pos, 0, 0, 5, 1);
    for (0..5) |i| {
        try testing.expectEqual(@as(f32, 0), pos[i * 3]);
        try testing.expectEqual(@as(f32, @as(f32, @floatFromInt(i)) - 3.0), pos[i * 3 + 1]);
    }
    // Even sizes are symmetric about zero.
    var even: [4 * 1 * 3]f32 = undefined;
    writePositions(&even, 0, 2, 4, 1);
    try testing.expectEqual(@as(f32, 2), even[0]);
    try testing.expectEqual(@as(f32, -2), even[1]);
    try testing.expectEqual(@as(f32, 1), even[3 * 3 + 1]);
}

test "channel interleave round-trips planar latents" {
    const n = 3;
    var planar: [channels * n]f32 = undefined;
    for (&planar, 0..) |*v, i| v.* = @floatFromInt(i);
    var tokens: [channels * n]f32 = undefined;
    interleaveChannels(&tokens, &planar, n);
    // Token 1's channel 2 is plane 2's element 1.
    try testing.expectEqual(planar[2 * n + 1], tokens[1 * channels + 2]);
    var back: [channels * n]f32 = undefined;
    deinterleaveChannels(&back, &tokens, n);
    try testing.expectEqualSlices(f32, &planar, &back);
}

test "the timestep table is bf16-rounded" {
    // Frequency 0 is exactly 1, so the first cos/sin pair is the bf16-rounded
    // timestep itself scaled by 1000: the rounding is observable in the output,
    // which is why it cannot be skipped on an fp32 device.
    var out: [tdim]f32 = undefined;
    const sigma: f32 = 0.3141592;
    timestepEmbedding(&out, sigma);
    const dtypes = tp_core.dtype;
    const t = dtypes.bf16ToF32(dtypes.f32ToBf16(sigma));
    try testing.expectApproxEqAbs(@cos(1000.0 * t), out[0], 1e-6);
    try testing.expectApproxEqAbs(@sin(1000.0 * t), out[tdim / 2], 1e-6);
    try testing.expect(t != sigma);
}

test "modulate and gatedAdd broadcast over rows" {
    var x = [_]f32{ 1, 2, 3, 4 };
    modulate(&x, &.{ 0.5, -1.0 }, &.{ 10, 20 });
    try testing.expectEqualSlices(f32, &.{ 11.5, 20, 14.5, 20 }, &x);
    gatedAdd(&x, &.{ 1, 1, 2, 2 }, &.{ 2, 0.5 });
    try testing.expectEqualSlices(f32, &.{ 13.5, 20.5, 18.5, 21 }, &x);
}

const test_gate = @import("../test_gate.zig");

const ref_path = "src/models/assets/mageflow_ref.safetensors";
const mage_ckpt = "/home/qt/genai/comfyui/models/diffusion_models/mageflow/mageFlow_mageFlow4B.safetensors";
const compressed_ref_path = "src/models/assets/mageflow_compressed_ref.safetensors";
const mage_compressed_ckpt = "/home/qt/genai/comfyui/models/diffusion_models/mageflow-compressed/magetrailMageflow4B_v025.safetensors";
/// Blocks the fp32 reference keeps; see tools/gen_mageflow_fixtures.py.
const ref_layers = 2;

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

/// One fixture's three cases against `ckpt`, at the depth the reference kept.
///
/// `want_rank` is the low-rank modulation rank the checkpoint must report: 0 for
/// a dense one. Asserted rather than inferred, so a compressed file quietly
/// loading through the dense path (which it WILL do, every tensor name matches)
/// fails here by name instead of as a number.
fn checkAgainstRef(
    gpa: std.mem.Allocator,
    io: std.Io,
    ckpt: []const u8,
    fixture: []const u8,
    want_rank: usize,
) !void {
    try test_gate.requireIntegration();
    try test_gate.requireModelFile(io, ckpt);
    try test_gate.requireModelFile(io, fixture);

    var ref = try SafeTensors.open(gpa, io, fixture);
    defer ref.deinit();
    var ck = try SafeTensors.open(gpa, io, ckpt);
    defer ck.deinit();

    const store: WeightStore = .{ .safetensors = &ck };
    try testing.expectEqual(@as(usize, n_blocks), DiT.depthIn(store));
    try testing.expectEqual(want_rank, DiT.rankIn(store));
    var model = try DiT.loadDepth(gpa, store, ref_layers);
    defer model.deinit();

    // `even` is the control (both centering conventions agree on even sizes),
    // `odd` is what pins Mage's, `ref` the edit path's appended tokens.
    for ([_][]const u8{ "even", "odd", "ref" }) |case| {
        var kb: [64]u8 = undefined;
        const key = struct {
            fn f(buf: []u8, c: []const u8, suffix: []const u8) []const u8 {
                return std.fmt.bufPrint(buf, "{s}.{s}", .{ c, suffix }) catch unreachable;
            }
        }.f;

        const x_lat = try (try ref.require(key(&kb, case, "x"))).toF32Alloc(gpa);
        defer gpa.free(x_lat);
        const ctx = try (try ref.require(key(&kb, case, "ctx"))).toF32Alloc(gpa);
        defer gpa.free(ctx);
        const sig = try (try ref.require(key(&kb, case, "sigma"))).toF32Alloc(gpa);
        defer gpa.free(sig);
        const want = try (try ref.require(key(&kb, case, "out"))).toF32Alloc(gpa);
        defer gpa.free(want);

        const shape = (try ref.require(key(&kb, case, "x"))).info.shape.slice();
        try testing.expectEqual(@as(usize, 3), shape.len);
        const lat_h = shape[1];
        const lat_w = shape[2];
        const seq_txt = ctx.len / txt_dim;

        var refs: [1]Ref = undefined;
        var n_refs: usize = 0;
        if (ref.get(key(&kb, case, "ref0"))) |v| {
            const rs = v.info.shape.slice();
            refs[0] = .{ .lat = try v.toF32Alloc(gpa), .h = rs[1], .w = rs[2] };
            n_refs = 1;
        }
        defer for (refs[0..n_refs]) |r| gpa.free(r.lat);

        const got = try gpa.alloc(f32, x_lat.len);
        defer gpa.free(got);
        try model.forward(io, gpa, got, x_lat, lat_h, lat_w, sig[0], ctx, seq_txt, refs[0..n_refs], null);

        const rel = relL2(want, got);
        errdefer std.debug.print("case {s} ({d}x{d}, {d} refs): velocity rel L2 {e:.4}\n", .{ case, lat_h, lat_w, n_refs, rel });
        // f32 on both sides, so the only difference is reduction order.
        try testing.expect(rel < 2e-5);
    }
}

test "the Mage-Flow DiT matches ComfyUI on a real checkpoint" {
    try checkAgainstRef(testing.allocator, testing.io, mage_ckpt, ref_path, 0);
}

// Pins the one thing compression decides: where the SiLU sits relative to the
// shared low-rank projection, and that `norm_out` is not compressed with the
// blocks. It has teeth by construction, since applying SiLU per block as well
// (the dense spelling) still runs: a rank-256 head reads the first 256 of a
// 3072 vector without complaint.
test "the compressed Mage-Flow DiT matches its ComfyUI node" {
    try checkAgainstRef(testing.allocator, testing.io, mage_compressed_ckpt, compressed_ref_path, 256);
}
