//! Krea 2 diffusion transformer (`SingleStreamDiT`), mirroring
//! comfy/ldm/krea2/model.py.
//!
//! Single-stream MMDiT: text tokens (from the txtfusion adapter) and 2x2
//! patchified latent tokens form one sequence through 28 identical blocks
//! with AdaLN-single modulation, GQA (48/12, head_dim 128) with per-head
//! QK-norm and sigmoid-gated output, SwiGLU MLPs, and 3-axis interleaved RoPE
//! (theta 1000, dims 32/48/48; text tokens sit at position (0,0,0)).
//!
//! Loads from any `WeightStore`, safetensors *or* GGUF, so a checkpoint's
//! container is not part of this file's business. Large weights keep their
//! checkpoint dtype (fp8-e4m3, int8/int4 convrot with per-row scales, or the ggml
//! block quants from a GGUF) and dequantize inside the GEMM; small vectors (norm
//! scales, biases, modulation) are dequantized to f32 at load. Norms use the
//! `(1 + scale)` convention with eps 1e-5, folded into the weight at load time.
//! The store's mapping must outlive the model, the `Weight`s are views into
//! it.

const std = @import("std");
const safetensors = @import("tp_core").safetensors;
const weights_mod = @import("tp_core").weights;
const ops = @import("tp_ops");
const quant_weight = @import("quant_weight.zig");

const SafeTensors = safetensors.SafeTensors;
const WeightStore = weights_mod.WeightStore;
const Weight = ops.matmul.Weight;
const DType = @import("tp_core").dtype.DType;

pub const features = 6144;
pub const tdim = 256;
pub const n_heads = 48;
pub const n_kv_heads = 12;
pub const head_dim = 128;
pub const n_blocks = 28;
pub const patch = 2;
pub const channels = 16;
pub const mlp_dim = 16384;
pub const rope_theta: f64 = 1000.0;
pub const rope_axes = [3]usize{ 32, 48, 48 };

pub const txt_dim = 2560;
pub const txt_layers = 12;
pub const txt_heads = 20;
pub const txt_mlp_dim = 6912;

const rms_eps: f32 = 1e-5;

const Attn = struct {
    wq: Weight,
    wk: Weight,
    wv: Weight,
    wo: Weight,
    gate: Weight,
    qnorm: []const f32, // effective (1+scale), [head_dim]
    knorm: []const f32,
    heads: usize,
    kv_heads: usize,
};

const Swiglu = struct {
    gate: Weight,
    up: Weight,
    down: Weight,
};

const Block = struct {
    mod: []const f32, // [6 * features]
    prenorm: []const f32,
    postnorm: []const f32,
    attn: Attn,
    mlp: Swiglu,
};

const TxtBlock = struct {
    prenorm: []const f32,
    postnorm: []const f32,
    attn: Attn,
    mlp: Swiglu,
};

const LinearW = struct {
    w: Weight,
    b: ?[]const f32,
};

pub const DiT = struct {
    arena: std.heap.ArenaAllocator,
    first: LinearW, // 64 -> 6144
    blocks: []Block,
    tmlp0: LinearW, // 256 -> 6144
    tmlp2: LinearW, // 6144 -> 6144
    tproj1: LinearW, // 6144 -> 6*6144
    txt_layerwise: [2]TxtBlock,
    txt_projector: Weight, // [1, 12]
    txt_refiner: [2]TxtBlock,
    txtmlp_norm: []const f32,
    txtmlp1: LinearW, // 2560 -> 6144
    txtmlp3: LinearW, // 6144 -> 6144
    last_norm: []const f32,
    last_mod: []const f32, // [2 * features]
    last_linear: LinearW, // 6144 -> 64

    /// Materialize a projection weight to f32 for the GPU `opMatmul` fused-GEMM
    /// path (`dit_cuda`/`dit_gpu`). Used for the two non-quantized linears fed to
    /// it raw, the `first` patch embed and the `last.linear` output projection,
    /// which need its bias + destination-offset support that the bulk-weight
    /// GEMMs don't. That path only has an f32 pipeline (the CUDA fused kernel has
    /// no fp8 variant, `backend.zig` `opMatmul`), so every dtype is normalized to
    /// f32 here. These two weights are tiny ([F,64] / [64,F]) so the cost is
    /// negligible, and it keeps first/last uniform across all checkpoint formats.
    ///
    /// - `f32`: consumed natively, so pass the mmap bytes through.
    /// - `f8_e4m3`/`bf16`/`f16`: materialize to f32 once into `alloc` (the model
    ///   arena, which outlives the model), folding any per-tensor scale. Without
    ///   this the f32 pipeline reads the packed bytes as f32 and the image is
    ///   pure noise (bf16: ComfyUI-native int8 checkpoints) or the run aborts on
    ///   the fp8 assert (fp8 checkpoints on CUDA).
    /// - anything else (i8/i4/block-quant): these projections are never
    ///   quantized in any known checkpoint, and dequanting them here would be
    ///   wrong (int needs the per-row scale + convrot; block-quant needs ggml).
    ///   Refuse loudly rather than emit silent garbage.
    fn opMatmulF32(alloc: std.mem.Allocator, w: Weight) !Weight {
        return ops.matmul.materializeF32(alloc, w);
    }

    pub fn load(gpa: std.mem.Allocator, store: WeightStore) !DiT {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();

        // The fp8 checkpoint uses bare tensor names (`blocks.0...`); the int8
        // convrot checkpoint nests them under `model.diffusion_model.`.
        const pfx: []const u8 = if (store.get("model.diffusion_model.blocks.0.mod.lin") != null)
            "model.diffusion_model."
        else
            "";
        const l = Loader{ .store = store, .alloc = alloc, .pfx = pfx };

        const blocks = try alloc.alloc(Block, n_blocks);
        for (blocks, 0..) |*blk, i| {
            blk.* = .{
                .mod = try l.vec("blocks.{d}.mod.lin", .{i}, 6 * features),
                .prenorm = try l.normScale("blocks.{d}.prenorm.scale", .{i}, features),
                .postnorm = try l.normScale("blocks.{d}.postnorm.scale", .{i}, features),
                .attn = try l.loadAttn("blocks.{d}", .{i}, features, n_heads, n_kv_heads),
                .mlp = try l.loadSwiglu("blocks.{d}", .{i}, features, mlp_dim),
            };
        }

        var txt_layerwise: [2]TxtBlock = undefined;
        var txt_refiner: [2]TxtBlock = undefined;
        for (0..2) |i| {
            txt_layerwise[i] = try l.loadTxtBlock("txtfusion.layerwise_blocks.{d}", i);
            txt_refiner[i] = try l.loadTxtBlock("txtfusion.refiner_blocks.{d}", i);
        }

        // The `first` patch-embed and `last.linear` output projection are the
        // only non-quantized weights the GPU backends feed straight to the
        // f32/fp8-only `opMatmul` (dit_cuda/dit_gpu); it has no bf16 pipeline and
        // would reinterpret bf16 bytes as f32 -> pure noise. ComfyUI-native int8
        // checkpoints store these two as BF16 (our own converter used F32), so
        // materialize any non-f32/fp8 storage to f32 once at load. They are tiny
        // ([F,64] / [64,F]) so the cost is negligible; fp8/f32 pass through, and
        // the CPU matmul handles bf16 natively regardless.
        var first = try l.loadLinear("first", features, channels * patch * patch, true);
        first.w = try opMatmulF32(alloc, first.w);
        const tmlp0 = try l.loadLinear("tmlp.0", features, tdim, true);
        const tmlp2 = try l.loadLinear("tmlp.2", features, features, true);
        const tproj1 = try l.loadLinear("tproj.1", 6 * features, features, true);
        // Both consumers (CPU and CUDA) read the projector through `convertToF32`,
        // which has no int8 arm. It is 12 values, so normalize it here with the other
        // two rather than teaching each caller the dequant.
        const txt_projector = try opMatmulF32(alloc, try l.mat("txtfusion.projector.weight", .{}, 1, txt_layers));
        const txtmlp_norm = try l.normScale("txtmlp.0.scale", .{}, txt_dim);
        const txtmlp1 = try l.loadLinear("txtmlp.1", features, txt_dim, true);
        const txtmlp3 = try l.loadLinear("txtmlp.3", features, features, true);
        const last_norm = try l.normScale("last.norm.scale", .{}, features);
        const last_mod = try l.vec("last.modulation.lin", .{}, 2 * features);
        var last_linear = try l.loadLinear("last.linear", channels * patch * patch, features, true);
        last_linear.w = try opMatmulF32(alloc, last_linear.w);

        return .{
            .arena = arena,
            .first = first,
            .blocks = blocks,
            .tmlp0 = tmlp0,
            .tmlp2 = tmlp2,
            .tproj1 = tproj1,
            .txt_layerwise = txt_layerwise,
            .txt_projector = txt_projector,
            .txt_refiner = txt_refiner,
            .txtmlp_norm = txtmlp_norm,
            .txtmlp1 = txtmlp1,
            .txtmlp3 = txtmlp3,
            .last_norm = last_norm,
            .last_mod = last_mod,
            .last_linear = last_linear,
        };
    }

    pub fn deinit(self: *DiT) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// Predict velocity for one latent. `x_lat`/`out` are planar
    /// [channels][lat_h][lat_w]; `ctx` is [seq_txt][12][2560] (post-strip
    /// encoder output); `sigma` is the flow-matching timestep.
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
        std.debug.assert(lat_h % patch == 0 and lat_w % patch == 0);
        std.debug.assert(x_lat.len == channels * lat_h * lat_w);
        std.debug.assert(out.len == x_lat.len);
        std.debug.assert(ctx.len == seq_txt * txt_layers * txt_dim);

        // Arm fine-grained cancel inside the CPU matmul/attention kernels for
        // the whole forward: a single MLP GEMM is seconds of CPU work, so
        // per-block polling alone leaves a multi-second cancel latency.
        const prev_tok = ops.cancel.token;
        ops.cancel.token = cancel;
        defer ops.cancel.token = prev_tok;
        const h = lat_h / patch;
        const w = lat_w / patch;
        const n_img = h * w;
        const seq = seq_txt + n_img;

        // Timestep path: t = tmlp(sinusoidal(sigma)); tvec = tproj(gelu(t)).
        var temb: [tdim]f32 = undefined;
        ops.rope.timestepEmbedding(&temb, sigma, 10000.0);
        const t = try gpa.alloc(f32, features);
        defer gpa.free(t);
        {
            const t0 = try gpa.alloc(f32, features);
            defer gpa.free(t0);
            try linear(io, gpa, t0, &temb, 1, self.tmlp0);
            ops.act.geluTanh(t0);
            try linear(io, gpa, t, t0, 1, self.tmlp2);
        }
        const tvec = try gpa.alloc(f32, 6 * features);
        defer gpa.free(tvec);
        {
            const tg = try gpa.alloc(f32, features);
            defer gpa.free(tg);
            @memcpy(tg, t);
            ops.act.geluTanh(tg);
            try linear(io, gpa, tvec, tg, 1, self.tproj1);
        }

        // Text path: txtfusion over the 12-layer stack, then txtmlp to 6144.
        const txt_tokens = try self.txtFusion(io, gpa, ctx, seq_txt);
        defer gpa.free(txt_tokens);

        // Combined sequence: [text | image].
        const x = try gpa.alloc(f32, seq * features);
        defer gpa.free(x);
        {
            ops.norm.rmsNorm(txt_tokens, txt_tokens, self.txtmlp_norm, rms_eps);
            const mid = try gpa.alloc(f32, seq_txt * features);
            defer gpa.free(mid);
            try linear(io, gpa, mid, txt_tokens, seq_txt, self.txtmlp1);
            ops.act.geluTanh(mid);
            try linear(io, gpa, x[0 .. seq_txt * features], mid, seq_txt, self.txtmlp3);
        }
        {
            // Patchify: token (hi, wi), feature (c, ph, pw).
            const img_in = try gpa.alloc(f32, n_img * channels * patch * patch);
            defer gpa.free(img_in);
            for (0..h) |hi| {
                for (0..w) |wi| {
                    const tok = img_in[(hi * w + wi) * channels * patch * patch ..];
                    for (0..channels) |c| {
                        for (0..patch) |ph| {
                            for (0..patch) |pw| {
                                tok[c * patch * patch + ph * patch + pw] =
                                    x_lat[c * lat_h * lat_w + (hi * patch + ph) * lat_w + (wi * patch + pw)];
                            }
                        }
                    }
                }
            }
            try linear(io, gpa, x[seq_txt * features ..], img_in, n_img, self.first);
        }

        // RoPE table: text at (0,0,0), image at (0, row, col).
        var freqs = blk: {
            const pos = try gpa.alloc(f32, seq * 3);
            defer gpa.free(pos);
            @memset(pos[0 .. seq_txt * 3], 0);
            for (0..h) |hi| {
                for (0..w) |wi| {
                    const base = (seq_txt + hi * w + wi) * 3;
                    pos[base] = 0;
                    pos[base + 1] = @floatFromInt(hi);
                    pos[base + 2] = @floatFromInt(wi);
                }
            }
            break :blk try ops.rope.fluxFreqs(gpa, pos, &rope_axes, rope_theta);
        };
        defer freqs.deinit(gpa);

        for (self.blocks) |*blk| {
            // Poll cancel between blocks so a stop lands mid-step (a full CPU
            // step can take 30+ seconds) rather than only at step boundaries.
            if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
            try self.blockForward(io, gpa, blk, x, seq, tvec, freqs);
        }

        // Final layer on image tokens only (row-wise, so slicing first is safe).
        const img_rows = x[seq_txt * features ..];
        {
            const scale = try gpa.alloc(f32, features);
            defer gpa.free(scale);
            const shift = try gpa.alloc(f32, features);
            defer gpa.free(shift);
            for (scale, shift, t, 0..) |*sc, *sh, tv, j| {
                sc.* = tv + self.last_mod[j];
                sh.* = tv + self.last_mod[features + j];
            }
            ops.norm.rmsNorm(img_rows, img_rows, self.last_norm, rms_eps);
            var row: usize = 0;
            while (row < img_rows.len) : (row += features) {
                for (img_rows[row..][0..features], scale, shift) |*v, sc, sh| {
                    v.* = (1.0 + sc) * v.* + sh;
                }
            }
        }
        const final = try gpa.alloc(f32, n_img * channels * patch * patch);
        defer gpa.free(final);
        try linear(io, gpa, final, img_rows, n_img, self.last_linear);

        // Unpatchify.
        for (0..h) |hi| {
            for (0..w) |wi| {
                const tok = final[(hi * w + wi) * channels * patch * patch ..];
                for (0..channels) |c| {
                    for (0..patch) |ph| {
                        for (0..patch) |pw| {
                            out[c * lat_h * lat_w + (hi * patch + ph) * lat_w + (wi * patch + pw)] =
                                tok[c * patch * patch + ph * patch + pw];
                        }
                    }
                }
            }
        }
    }

    /// TextFusionTransformer: 2 blocks attending across the 12-layer axis per
    /// token, a Linear(12->1) collapse, then 2 refiner blocks over the tokens.
    /// Returns [seq_txt, txt_dim].
    /// Timestep path: t = tmlp(sinusoidal(sigma)); tvec = tproj(gelu(t)).
    /// Caller frees both slices.
    pub fn timestepVectors(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, sigma: f32) !struct { t: []f32, tvec: []f32 } {
        var temb: [tdim]f32 = undefined;
        ops.rope.timestepEmbedding(&temb, sigma, 10000.0);
        const t = try gpa.alloc(f32, features);
        errdefer gpa.free(t);
        {
            const t0 = try gpa.alloc(f32, features);
            defer gpa.free(t0);
            try linear(io, gpa, t0, &temb, 1, self.tmlp0);
            ops.act.geluTanh(t0);
            try linear(io, gpa, t, t0, 1, self.tmlp2);
        }
        const tvec = try gpa.alloc(f32, 6 * features);
        errdefer gpa.free(tvec);
        {
            const tg = try gpa.alloc(f32, features);
            defer gpa.free(tg);
            @memcpy(tg, t);
            ops.act.geluTanh(tg);
            try linear(io, gpa, tvec, tg, 1, self.tproj1);
        }
        return .{ .t = t, .tvec = tvec };
    }

    /// Text conditioning to combined-sequence tokens: txtfusion + txtmlp.
    /// Returns [seq_txt, features]; caller frees.
    pub fn textTokens(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, ctx: []const f32, seq_txt: usize) ![]f32 {
        const fused = try self.txtFusion(io, gpa, ctx, seq_txt);
        defer gpa.free(fused);
        ops.norm.rmsNorm(fused, fused, self.txtmlp_norm, rms_eps);
        const mid = try gpa.alloc(f32, seq_txt * features);
        defer gpa.free(mid);
        try linear(io, gpa, mid, fused, seq_txt, self.txtmlp1);
        ops.act.geluTanh(mid);
        const out_tokens = try gpa.alloc(f32, seq_txt * features);
        errdefer gpa.free(out_tokens);
        try linear(io, gpa, out_tokens, mid, seq_txt, self.txtmlp3);
        return out_tokens;
    }

    /// Patchify a planar latent into [n_img, channels*patch^2] rows.
    pub fn patchify(gpa: std.mem.Allocator, x_lat: []const f32, lat_h: usize, lat_w: usize) ![]f32 {
        const h = lat_h / patch;
        const w = lat_w / patch;
        const img_in = try gpa.alloc(f32, h * w * channels * patch * patch);
        for (0..h) |hi| {
            for (0..w) |wi| {
                const tok = img_in[(hi * w + wi) * channels * patch * patch ..];
                for (0..channels) |c| {
                    for (0..patch) |ph| {
                        for (0..patch) |pw| {
                            tok[c * patch * patch + ph * patch + pw] =
                                x_lat[c * lat_h * lat_w + (hi * patch + ph) * lat_w + (wi * patch + pw)];
                        }
                    }
                }
            }
        }
        return img_in;
    }

    /// RoPE frequency table for [text | image] positions.
    pub fn ropeFreqs(gpa: std.mem.Allocator, seq_txt: usize, h: usize, w: usize) !ops.rope.Freqs {
        const seq = seq_txt + h * w;
        const pos = try gpa.alloc(f32, seq * 3);
        defer gpa.free(pos);
        @memset(pos[0 .. seq_txt * 3], 0);
        for (0..h) |hi| {
            for (0..w) |wi| {
                const base = (seq_txt + hi * w + wi) * 3;
                pos[base] = 0;
                pos[base + 1] = @floatFromInt(hi);
                pos[base + 2] = @floatFromInt(wi);
            }
        }
        return ops.rope.fluxFreqs(gpa, pos, &rope_axes, rope_theta);
    }

    /// Final layer + unpatchify: img_rows [n_img, features] (modified in
    /// place) -> planar velocity `out`.
    pub fn finalize(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, out: []f32, img_rows: []f32, t: []const f32, lat_h: usize, lat_w: usize) !void {
        const h = lat_h / patch;
        const w = lat_w / patch;
        const n_img = h * w;
        {
            const scale = try gpa.alloc(f32, features);
            defer gpa.free(scale);
            const shift = try gpa.alloc(f32, features);
            defer gpa.free(shift);
            for (scale, shift, t, 0..) |*sc, *sh, tv, j| {
                sc.* = tv + self.last_mod[j];
                sh.* = tv + self.last_mod[features + j];
            }
            ops.norm.rmsNorm(img_rows, img_rows, self.last_norm, rms_eps);
            var row: usize = 0;
            while (row < img_rows.len) : (row += features) {
                for (img_rows[row..][0..features], scale, shift) |*v, sc, sh| {
                    v.* = (1.0 + sc) * v.* + sh;
                }
            }
        }
        const final = try gpa.alloc(f32, n_img * channels * patch * patch);
        defer gpa.free(final);
        try linear(io, gpa, final, img_rows, n_img, self.last_linear);
        unpatchify(out, final, lat_h, lat_w);
    }

    /// Scatter final-layer patch tokens back into the planar latent.
    pub fn unpatchify(out: []f32, final: []const f32, lat_h: usize, lat_w: usize) void {
        const h = lat_h / patch;
        const w = lat_w / patch;
        for (0..h) |hi| {
            for (0..w) |wi| {
                const tok = final[(hi * w + wi) * channels * patch * patch ..];
                for (0..channels) |c| {
                    for (0..patch) |ph| {
                        for (0..patch) |pw| {
                            out[c * lat_h * lat_w + (hi * patch + ph) * lat_w + (wi * patch + pw)] =
                                tok[c * patch * patch + ph * patch + pw];
                        }
                    }
                }
            }
        }
    }

    pub fn txtFusion(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, ctx: []const f32, seq_txt: usize) ![]f32 {
        const work = try gpa.alloc(f32, ctx.len);
        defer gpa.free(work);
        @memcpy(work, ctx);

        // seq_txt independent sequences of length 12.
        for (&self.txt_layerwise) |*blk| {
            try self.txtBlockForward(io, gpa, blk, work, seq_txt, txt_layers);
        }

        const projected = try gpa.alloc(f32, seq_txt * txt_dim);
        errdefer gpa.free(projected);
        {
            var pw: [txt_layers]f32 = undefined;
            try safetensors.convertToF32(self.txt_projector.dtype, self.txt_projector.bytes, &pw);
            for (0..seq_txt) |tok| {
                const dst = projected[tok * txt_dim ..][0..txt_dim];
                @memset(dst, 0);
                for (0..txt_layers) |l| {
                    const src = work[(tok * txt_layers + l) * txt_dim ..][0..txt_dim];
                    for (dst, src) |*d, s| d.* += pw[l] * s;
                }
            }
        }

        for (&self.txt_refiner) |*blk| {
            try self.txtBlockForward(io, gpa, blk, projected, 1, seq_txt);
        }
        return projected;
    }

    /// TextFusionBlock: x += attn(prenorm(x)); x += mlp(postnorm(x)).
    /// `x` holds n_seqs sequences of seq_len rows of txt_dim, no RoPE/mask.
    fn txtBlockForward(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, blk: *const TxtBlock, x: []f32, n_seqs: usize, seq_len: usize) !void {
        _ = self;
        const rows = n_seqs * seq_len;
        const normed = try gpa.alloc(f32, rows * txt_dim);
        defer gpa.free(normed);
        ops.norm.rmsNorm(normed, x, blk.prenorm, rms_eps);

        const a = try gpa.alloc(f32, rows * txt_dim);
        defer gpa.free(a);
        try attnForward(io, gpa, &blk.attn, normed, n_seqs, seq_len, txt_dim, null, a);
        for (x, a) |*xi, ai| xi.* += ai;

        ops.norm.rmsNorm(normed, x, blk.postnorm, rms_eps);
        try swigluForward(io, gpa, &blk.mlp, normed, rows, txt_dim, txt_mlp_dim, a);
        for (x, a) |*xi, ai| xi.* += ai;
    }

    /// SingleStreamBlock with AdaLN-single modulation.
    fn blockForward(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, blk: *const Block, x: []f32, seq: usize, tvec: []const f32, freqs: ops.rope.Freqs) !void {
        _ = self;
        // Six modulation chunks: tvec + per-block learned offset.
        const mv = try gpa.alloc(f32, 6 * features);
        defer gpa.free(mv);
        for (mv, tvec, blk.mod) |*m, tv, bm| m.* = tv + bm;
        const pre_scale = mv[0 * features ..][0..features];
        const pre_shift = mv[1 * features ..][0..features];
        const pre_gate = mv[2 * features ..][0..features];
        const post_scale = mv[3 * features ..][0..features];
        const post_shift = mv[4 * features ..][0..features];
        const post_gate = mv[5 * features ..][0..features];

        const normed = try gpa.alloc(f32, seq * features);
        defer gpa.free(normed);
        const a = try gpa.alloc(f32, seq * features);
        defer gpa.free(a);

        // x += pre_gate * attn((1+pre_scale) * prenorm(x) + pre_shift)
        ops.norm.rmsNorm(normed, x, blk.prenorm, rms_eps);
        modulate(normed, pre_scale, pre_shift);
        try attnForward(io, gpa, &blk.attn, normed, 1, seq, features, freqs, a);
        gatedAdd(x, a, pre_gate);

        // x += post_gate * mlp((1+post_scale) * postnorm(x) + post_shift)
        ops.norm.rmsNorm(normed, x, blk.postnorm, rms_eps);
        modulate(normed, post_scale, post_shift);
        try swigluForward(io, gpa, &blk.mlp, normed, seq, features, mlp_dim, a);
        gatedAdd(x, a, post_gate);
    }
};

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

/// Krea 2 attention: q/k/v/gate projections, per-head QK RMSNorm, optional
/// RoPE, GQA softmax attention, sigmoid-gated output projection.
fn attnForward(
    io: std.Io,
    gpa: std.mem.Allocator,
    attn: *const Attn,
    x: []const f32,
    n_seqs: usize,
    seq_len: usize,
    dim: usize,
    freqs: ?ops.rope.Freqs,
    out: []f32,
) !void {
    const rows = n_seqs * seq_len;
    const q_dim = attn.heads * head_dim;
    const kv_dim = attn.kv_heads * head_dim;
    std.debug.assert(x.len == rows * dim and out.len == rows * dim);

    const q = try gpa.alloc(f32, rows * q_dim);
    defer gpa.free(q);
    const k = try gpa.alloc(f32, rows * kv_dim);
    defer gpa.free(k);
    const v = try gpa.alloc(f32, rows * kv_dim);
    defer gpa.free(v);
    const g = try gpa.alloc(f32, rows * dim);
    defer gpa.free(g);
    try ops.matmul.matmul(io, gpa, q, x, rows, attn.wq, null);
    try ops.matmul.matmul(io, gpa, k, x, rows, attn.wk, null);
    try ops.matmul.matmul(io, gpa, v, x, rows, attn.wv, null);
    try ops.matmul.matmul(io, gpa, g, x, rows, attn.gate, null);

    ops.norm.rmsNorm(q, q, attn.qnorm, rms_eps);
    ops.norm.rmsNorm(k, k, attn.knorm, rms_eps);
    if (freqs) |f| {
        std.debug.assert(n_seqs == 1);
        ops.rope.applyInterleaved(q, f, seq_len, attn.heads, head_dim);
        ops.rope.applyInterleaved(k, f, seq_len, attn.kv_heads, head_dim);
    }

    const attn_out = try gpa.alloc(f32, rows * q_dim);
    defer gpa.free(attn_out);
    for (0..n_seqs) |s| {
        try ops.attention.attention(
            io,
            gpa,
            attn_out[s * seq_len * q_dim ..][0 .. seq_len * q_dim],
            q[s * seq_len * q_dim ..][0 .. seq_len * q_dim],
            k[s * seq_len * kv_dim ..][0 .. seq_len * kv_dim],
            v[s * seq_len * kv_dim ..][0 .. seq_len * kv_dim],
            .{
                .seq_q = seq_len,
                .seq_kv = seq_len,
                .n_heads = attn.heads,
                .n_kv_heads = attn.kv_heads,
                .head_dim = head_dim,
            },
        );
    }
    ops.act.sigmoidMul(attn_out, g);
    try ops.matmul.matmul(io, gpa, out, attn_out, rows, attn.wo, null);
}

fn swigluForward(io: std.Io, gpa: std.mem.Allocator, mlp: *const Swiglu, x: []const f32, rows: usize, dim: usize, inner: usize, out: []f32) !void {
    std.debug.assert(x.len == rows * dim and out.len == rows * dim);
    const gate = try gpa.alloc(f32, rows * inner);
    defer gpa.free(gate);
    const up = try gpa.alloc(f32, rows * inner);
    defer gpa.free(up);
    try ops.matmul.matmul(io, gpa, gate, x, rows, mlp.gate, null);
    try ops.matmul.matmul(io, gpa, up, x, rows, mlp.up, null);
    ops.act.siluMul(gate, up);
    try ops.matmul.matmul(io, gpa, out, gate, rows, mlp.down, null);
}

fn linear(io: std.Io, gpa: std.mem.Allocator, out: []f32, x: []const f32, m: usize, lw: LinearW) !void {
    try ops.matmul.matmul(io, gpa, out, x, m, lw.w, lw.b);
}

/// Which GPU forward is asking `gpuLinKindSupported`. The two do not accept the same
/// set, so a single answer would either lock CUDA out of a format it has or hand Vulkan
/// one it does not.
pub const GpuArm = enum { vulkan, cuda };

/// Whether that GPU DiT forward (`dit_gpu`, `dit_cuda`) has a GEMM path for block
/// linears of this dtype. They branch on int8/int4 convrot and dense bf16 and treat
/// anything else as raw fp8-e4m3, so an unrecognized dtype is not a slow path, it
/// is silently wrong output. Both gate on this before dispatching.
pub fn gpuLinKindSupported(dt: DType, arm: GpuArm) bool {
    return switch (dt) {
        // `.w4a8` is decoded to int8 inside each backend's GEMM (the packed form stays
        // resident), so it runs wherever int8 does.
        // `.nvfp4` decodes to f16 inside each backend's GEMM (weight-only, which is what
        // NVFP4 is below Blackwell), so it runs wherever the f16 GEMM does, everywhere.
        .i8, .i4, .w4a8, .nvfp4, .bf16, .f8_e4m3 => true,
        // The ggml block quants decode per GEMM instead of expanding in VRAM, either to
        // convrot int8 (`Backend.blockQFormat`, q4_k/q8_0 only) or to f16
        // (`Backend.quantKernelSupported`, everything with a dequant kernel), which is
        // what `dit_cuda.blockQKind` picks between. Only the CUDA arm has either; Vulkan
        // has no block-quant GEMM at all.
        .q4_0, .q8_0, .q2_k, .q4_k, .q5_k, .q6_k, .iq4_nl => arm == .cuda,
        else => false,
    };
}

/// Whether any block linear is stored in ComfyUI's packed `asym_w4a8_int8` form.
///
/// Scans EVERY block's linears rather than reading one tensor: a real ComfyUI mixed
/// checkpoint quantizes per block, often leaving block 0 entirely dense, so a probe of
/// `blocks[0]` answers a different question.
pub fn anyW4A8(model: *const DiT) bool {
    for (model.blocks) |*b| {
        for ([_]Weight{ b.attn.wq, b.attn.wk, b.attn.wv, b.attn.wo, b.attn.gate, b.mlp.gate, b.mlp.up, b.mlp.down }) |w|
            if (w.dtype == .w4a8) return true;
    }
    return false;
}

/// A packed W4A8 block linear whose `group_size` is not a multiple of 8, if any, the
/// CUDA decode kernel reads four packed bytes (8 columns) per thread and so assumes one
/// group scale covers them. Returns the offending tensor's name so the refusal can say
/// which layer, since "unsupported" is unactionable across 224 weights.
///
/// No checkpoint in the wild uses a group size below 16 (ComfyUI's default), but the
/// format permits 4 and 8, so this is a refusal rather than an assert. Vulkan's decode is
/// general over the group size and does not need it.
pub fn w4a8SmallGroup(model: *const DiT) ?[]const u8 {
    for (model.blocks) |*b| {
        for ([_]Weight{ b.attn.wq, b.attn.wk, b.attn.wv, b.attn.wo, b.attn.gate, b.mlp.gate, b.mlp.up, b.mlp.down }) |w| {
            if (w.dtype == .w4a8 and w.w4a8.?.group_size % 8 != 0) return w.tag orelse "<untagged>";
        }
    }
    return null;
}

/// Whether the block linears that share the int8 activation prep carry the convrot
/// rotation, or null when they disagree with each other.
///
/// ComfyUI's `int8_tensorwise` ships both ways: rotated with a scale per output row, or
/// unrotated with one scale for the whole tensor. The rotation cancels across the GEMM
/// only when BOTH sides apply it, and one activation prep serves every GEMM in a block,
/// so a block that mixes the two has no prep that is right for all of them. Answering
/// null lets the caller refuse instead of picking one and computing the rest in a basis
/// their weights were never quantized in.
///
/// ⚠️ Every storage form that runs on that prep has to be counted here, not just the
/// plain int8 one: `.w4a8` decodes to a rotated int8 weight and takes int8's prep, so
/// leaving it out reports "unrotated" for a W4A8 checkpoint and pairs a rotated weight
/// with an unrotated activation. That renders noise, not a slightly worse image.
///
/// Scans every block for the reason `anyW4A8` does: a ComfyUI checkpoint quantizes per
/// layer, so one weight is an answer about one weight.
pub fn i8Convrot(model: *const DiT) ?bool {
    var seen: ?bool = null;
    for (model.blocks) |*b| {
        for ([_]Weight{ b.attn.wq, b.attn.wk, b.attn.wv, b.attn.wo, b.attn.gate, b.mlp.gate, b.mlp.up, b.mlp.down }) |w| {
            if (w.dtype != .i8 and w.dtype != .i4 and w.dtype != .w4a8) continue;
            const rot = w.convrot != 0;
            if (seen) |s| {
                if (s != rot) return null;
            } else seen = rot;
        }
    }
    return seen orelse false;
}

/// Whether any block linear is stored in ComfyUI's packed NVFP4 form. Scans every block
/// for the reason `anyW4A8` does: a real ComfyUI mixed checkpoint quantizes per block.
pub fn anyNvfp4(model: *const DiT) bool {
    for (model.blocks) |*b| {
        for ([_]Weight{ b.attn.wq, b.attn.wk, b.attn.wv, b.attn.wo, b.attn.gate, b.mlp.gate, b.mlp.up, b.mlp.down }) |w| {
            if (w.dtype == .nvfp4) return true;
        }
    }
    return false;
}

/// Largest transient f16 buffer any packed NVFP4 block linear decodes into, per the
/// backend's own sizing rule. A caller pre-sizes with this so the scratch never grows
/// mid-forward (which on Vulkan flushes the recording batch).
pub fn maxNvfp4Scratch(model: *const DiT, comptime bytesFor: fn (rows: usize, cols: usize) usize) usize {
    var max: usize = 0;
    for (model.blocks) |*b| {
        for ([_]Weight{ b.attn.wq, b.attn.wk, b.attn.wv, b.attn.wo, b.attn.gate, b.mlp.gate, b.mlp.up, b.mlp.down }) |w| {
            if (w.dtype == .nvfp4) max = @max(max, bytesFor(w.rows, w.cols));
        }
    }
    return max;
}

/// Largest transient int8 buffer any packed W4A8 block linear decodes into, per the
/// backend's own sizing rule (the two differ: CUDA's GEMM reads the raw `[rows][cols]`
/// while Vulkan's reads a row-padded k-major copy).
///
/// A caller pre-sizes with this so the scratch never has to grow mid-forward, which on
/// Vulkan would flush the recording batch and on CUDA sync the stream.
pub fn maxW4A8Scratch(model: *const DiT, comptime bytesFor: fn (rows: usize, cols: usize) usize) usize {
    var max: usize = 0;
    for (model.blocks) |*b| {
        for ([_]Weight{ b.attn.wq, b.attn.wk, b.attn.wv, b.attn.wo, b.attn.gate, b.mlp.gate, b.mlp.up, b.mlp.down }) |w| {
            if (w.dtype == .w4a8) max = @max(max, bytesFor(w.rows, w.cols));
        }
    }
    return max;
}

// --- weight loading --------------------------------------------------------

/// Resolves checkpoint tensor names (optionally under a runtime prefix) and
/// builds `Weight`s, transparently attaching per-row scale + ConvRot metadata
/// for int8/int4-quantized (`I8`/`I4`) tensors.
const Loader = struct {
    store: WeightStore,
    alloc: std.mem.Allocator,
    pfx: []const u8, // "" (fp8) or "model.diffusion_model." (int8/int4)

    fn name(l: Loader, buf: []u8, comptime fmt: []const u8, args: anytype, suffix: []const u8) ![]u8 {
        var fbs = std.Io.Writer.fixed(buf);
        try fbs.writeAll(l.pfx);
        try fbs.print(fmt, args);
        try fbs.writeAll(suffix);
        return fbs.buffered();
    }

    fn mat(l: Loader, comptime fmt: []const u8, args: anytype, rows: usize, cols: usize) !Weight {
        var buf: [160]u8 = undefined;
        const nm = try l.name(&buf, fmt, args, "");
        const view = l.store.get(nm) orelse return error.MissingTensor;
        const shape = view.info.shape.slice();

        // Both ComfyUI 4-bit formats must be recognized BEFORE the int4 heuristic
        // below, and each by its OWN sidecar rather than by dtype or shape: NVFP4 is
        // stored `U8 [rows, cols/2]` and W4A8 `I8 [rows, cols/2]`, which is exactly the
        // signature that heuristic keys on. NVFP4's nibbles are E2M1 floats with a
        // per-16-block fp8 scale and W4A8's are unsigned indices into a non-uniform
        // Lloyd-Max codebook, so reading either as signed int4 times a per-row scale is
        // finite, plausible and wrong. One implementation for all the families that
        // ship them (`quant_weight.zig`).
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

        // int4 convrot weights are nibble-packed (two values per byte), so the
        // on-disk shape is [rows, cols/2]. Our home-grown converter stores the
        // packed bytes as U8; ComfyUI's official W4A4 converter stores the same
        // bytes as I8 (the raw bits, and thus the nibble decode, are identical).
        // A genuine int8-convrot weight is also I8 but at the full [rows, cols],
        // so disambiguate int4 from int8 by the halved column count, not dtype
        // alone. Everything else (fp8/f32/bf16) is one element per stored slot.
        const dt = view.info.dtype;
        const halved = shape.len == 2 and shape[0] == rows and cols % 2 == 0 and shape[1] == cols / 2;
        const is_i4 = dt == .u8 or (dt == .i8 and halved);
        const wdt = if (is_i4) @as(@TypeOf(dt), .i4) else dt;
        const stored_cols = if (is_i4) cols / 2 else cols;
        if (is_i4 and cols % 2 != 0) return error.ShapeMismatch;
        if (shape.len != 2 or shape[0] != rows or shape[1] != stored_cols) {
            // Say which tensor and what was expected: a bare ShapeMismatch over 230
            // weights is not actionable, and the usual cause is a container whose
            // dim order differs (GGUF stores them reversed and the reader flips
            // them back).
            std.log.err("dit: {s} has shape {any} ({t}), expected [{d}, {d}]", .{ nm, shape, dt, rows, stored_cols });
            return error.ShapeMismatch;
        }

        // A shape-fixed block-quantized tensor (`TensorInfo.flat_blocks`) has its
        // blocks tiling the flat element sequence rather than each logical row, which
        // is not what `Weight.init` assumes. krea2 never hits this, its only
        // shape-fixed tensor, `first.weight`, is `keys_hiprec` and so unquantized,
        // but refusing loudly beats a silently wrong byte count in ReleaseFast.
        if (view.info.flat_blocks) {
            std.log.err("dit: {s} is {t} with flat block layout (shape-fixed); the DiT loader needs row-aligned blocks", .{ nm, dt });
            return error.UnsupportedCheckpoint;
        }
        var w = Weight.init(view.bytes, wdt, rows, cols);
        // Carry the checkpoint name on the Weight so a GEMM can be attributed to a
        // layer downstream (ops.matmul.probe, profiling, error messages). Duped into
        // the model arena because `nm` lives in a stack buffer; ~230 short strings
        // per model.
        w.tag = try l.alloc.dupe(u8, nm);
        if (wdt == .i8 or wdt == .i4) {
            // A per-output-row `weight_scale` with the size-256 group rotation folded
            // out at dequant time, or one scalar scale and no rotation at all.
            const meta = try quant_weight.int8Scale(l.alloc, l.store, nm, rows, cols);
            w.row_scale = meta.row_scale;
            w.convrot = meta.convrot;
        }
        return w;
    }

    fn vec(l: Loader, comptime fmt: []const u8, args: anytype, len: usize) ![]f32 {
        var buf: [160]u8 = undefined;
        const nm = try l.name(&buf, fmt, args, "");
        const view = l.store.get(nm) orelse return error.MissingTensor;
        if (view.info.elemCount() != len) return error.ShapeMismatch;
        return view.toF32Alloc(l.alloc);
    }

    /// Zero-centered norm scale -> effective (1 + scale) weight.
    fn normScale(l: Loader, comptime fmt: []const u8, args: anytype, len: usize) ![]f32 {
        const w = try l.vec(fmt, args, len);
        for (w) |*v| v.* += 1.0;
        return w;
    }

    fn loadLinear(l: Loader, comptime prefix: []const u8, rows: usize, cols: usize, bias: bool) !LinearW {
        return .{
            .w = try l.mat(prefix ++ ".weight", .{}, rows, cols),
            .b = if (bias) try l.vec(prefix ++ ".bias", .{}, rows) else null,
        };
    }

    fn loadAttn(l: Loader, comptime prefix: []const u8, args: anytype, dim: usize, heads: usize, kv_heads: usize) !Attn {
        return .{
            .wq = try l.mat(prefix ++ ".attn.wq.weight", args, heads * head_dim, dim),
            .wk = try l.mat(prefix ++ ".attn.wk.weight", args, kv_heads * head_dim, dim),
            .wv = try l.mat(prefix ++ ".attn.wv.weight", args, kv_heads * head_dim, dim),
            .wo = try l.mat(prefix ++ ".attn.wo.weight", args, dim, heads * head_dim),
            .gate = try l.mat(prefix ++ ".attn.gate.weight", args, dim, dim),
            .qnorm = try l.normScale(prefix ++ ".attn.qknorm.qnorm.scale", args, head_dim),
            .knorm = try l.normScale(prefix ++ ".attn.qknorm.knorm.scale", args, head_dim),
            .heads = heads,
            .kv_heads = kv_heads,
        };
    }

    fn loadSwiglu(l: Loader, comptime prefix: []const u8, args: anytype, dim: usize, inner: usize) !Swiglu {
        return .{
            .gate = try l.mat(prefix ++ ".mlp.gate.weight", args, inner, dim),
            .up = try l.mat(prefix ++ ".mlp.up.weight", args, inner, dim),
            .down = try l.mat(prefix ++ ".mlp.down.weight", args, dim, inner),
        };
    }

    fn loadTxtBlock(l: Loader, comptime prefix: []const u8, i: usize) !TxtBlock {
        return .{
            .prenorm = try l.normScale(prefix ++ ".prenorm.scale", .{i}, txt_dim),
            .postnorm = try l.normScale(prefix ++ ".postnorm.scale", .{i}, txt_dim),
            .attn = try l.loadAttn(prefix, .{i}, txt_dim, txt_heads, txt_heads),
            .mlp = try l.loadSwiglu(prefix, .{i}, txt_dim, txt_mlp_dim),
        };
    }
};

// --- tests -----------------------------------------------------------------

const test_gate = @import("../test_gate.zig");

test "modulate and gatedAdd broadcast over rows" {
    var x = [_]f32{ 1, 2, 3, 4 }; // 2 rows, dim 2
    modulate(&x, &.{ 0.5, -1.0 }, &.{ 10, 20 });
    try std.testing.expectEqualSlices(f32, &.{ 11.5, 20, 14.5, 20 }, &x);
    gatedAdd(&x, &.{ 1, 1, 2, 2 }, &.{ 2, 0.5 });
    try std.testing.expectEqualSlices(f32, &.{ 13.5, 20.5, 18.5, 21 }, &x);
}

test "opMatmulF32 passes f32/fp8 through and materializes bf16 to f32" {
    const dtypes = @import("tp_core").dtype;
    const gpa = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const alloc = arena.allocator();

    // f32 passes through untouched (same bytes, so the GPU f32 pipeline is fed
    // the original mmap; no needless copy).
    const f32w = [_]f32{ 1.0, -2.0, 3.5, 0.25 };
    const wf = try DiT.opMatmulF32(alloc, Weight.fromF32(&f32w, 2, 2));
    try std.testing.expectEqual(@as(@TypeOf(wf.dtype), .f32), wf.dtype);
    try std.testing.expectEqual(@intFromPtr(&f32w), @intFromPtr(wf.bytes.ptr));

    // fp8 is materialized to f32 (the CUDA fused opMatmul has no fp8 pipeline);
    // values match the e4m3 LUT exactly.
    const dt = @import("tp_core").dtype;
    const f8 = [_]u8{ 0x38, 0x40, 0x48, 0x50 }; // arbitrary e4m3 bytes
    const w8 = try DiT.opMatmulF32(alloc, Weight.init(&f8, .f8_e4m3, 2, 2));
    try std.testing.expectEqual(@as(@TypeOf(w8.dtype), .f32), w8.dtype);
    const g8 = std.mem.bytesAsSlice(f32, w8.bytes);
    for (f8, g8) |byte, g| try std.testing.expectEqual(dt.f8e4m3ToF32(byte), g);

    // bf16 is materialized to f32 with the exact bf16-rounded values. This is the
    // ComfyUI-native int8-checkpoint case; reading the packed bytes as f32 renders noise.
    const vals = [_]f32{ 1.0, -2.5, 0.125, 42.0 };
    var bf: [vals.len]u16 = undefined;
    for (&bf, vals) |*b, v| b.* = dtypes.f32ToBf16(v);
    const wb = try DiT.opMatmulF32(alloc, Weight.init(std.mem.sliceAsBytes(&bf), .bf16, 2, 2));
    try std.testing.expectEqual(@as(@TypeOf(wb.dtype), .f32), wb.dtype);
    const got = std.mem.bytesAsSlice(f32, wb.bytes);
    for (vals, got) |want, g| {
        try std.testing.expectEqual(dtypes.bf16ToF32(dtypes.f32ToBf16(want)), g);
    }

    // A quantized dtype for these projections is not a known checkpoint shape
    // and can't be dequanted here (no scale/convrot/ggml), so it aborts loudly
    // rather than silently mis-converting.
    const q = [_]u8{ 0, 1, 2, 3 };
    try std.testing.expectError(error.UnsupportedCheckpoint, DiT.opMatmulF32(alloc, Weight.init(&q, .i8, 2, 2)));
}

test "int8 convrot checkpoint loads with per-row scale + rotation metadata" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const path = "models/diffusion_model/krea2CenterSemiraw_v10Int8.safetensors";
    std.Io.Dir.cwd().access(io, path, .{}) catch return error.SkipZigTest;

    var st = try SafeTensors.open(gpa, io, path);
    defer st.deinit();
    var model = try DiT.load(gpa, .{ .safetensors = &st });
    defer model.deinit();

    // The 8 per-block linears are int8 convrot; scales/rotation must be wired.
    const attn = model.blocks[0].attn;
    for ([_]Weight{ attn.wq, attn.wk, attn.wv, attn.wo, attn.gate }) |w| {
        try std.testing.expect(w.dtype == .i8);
        try std.testing.expectEqual(@as(u32, ops.convrot.group_size), w.convrot);
        try std.testing.expect(w.row_scale != null);
        try std.testing.expectEqual(w.rows, w.row_scale.?.len);
        try std.testing.expectEqual(@as(usize, 0), w.cols % ops.convrot.group_size);
    }
    // Non-quantized tensors stay full precision (F32 here), no per-row scale.
    try std.testing.expect(model.first.w.dtype == .f32);
    try std.testing.expect(model.first.w.row_scale == null);

    try std.testing.expectEqual(@as(?bool, true), i8Convrot(&model));
}

test "an unrotated int8_tensorwise checkpoint loads with a broadcast scale" {
    // The other `int8_tensorwise` variant: one scale for the whole tensor and no
    // rotation, which ComfyUI runs as W8A8 with an unrotated activation. It also
    // quantizes the projections and the text-fusion stack, which the convrot files
    // leave dense, so this pins that those still arrive as f32 for the backends
    // that have no int GEMM on that path.
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const path = "models/diffusion_model/rayArtshoot_krea2NSFWV4.safetensors";
    std.Io.Dir.cwd().access(io, path, .{}) catch return error.SkipZigTest;

    var st = try SafeTensors.open(gpa, io, path);
    defer st.deinit();
    var model = try DiT.load(gpa, .{ .safetensors = &st });
    defer model.deinit();

    const attn = model.blocks[0].attn;
    for ([_]Weight{ attn.wq, attn.wk, attn.wv, attn.wo, attn.gate }) |w| {
        try std.testing.expect(w.dtype == .i8);
        try std.testing.expectEqual(@as(u32, 0), w.convrot);
        // Broadcast per row, so every consumer indexes the scale the same way whether
        // the checkpoint stored one value or `rows` of them.
        try std.testing.expectEqual(w.rows, w.row_scale.?.len);
        for (w.row_scale.?) |s| try std.testing.expectEqual(w.row_scale.?[0], s);
    }
    try std.testing.expectEqual(@as(?bool, false), i8Convrot(&model));

    // `first`, `last.linear` and the projector are int8 in this file and are the three
    // the GPU backends hand to an f32-only path.
    for ([_]Weight{ model.first.w, model.last_linear.w, model.txt_projector }) |w| {
        try std.testing.expect(w.dtype == .f32);
        try std.testing.expect(w.row_scale == null);
    }
}

test "every loaded weight carries its checkpoint tensor name as a tag" {
    // The tag is what makes `ops.matmul.probe` able to attribute a GEMM to a layer
    // (per-layer activation statistics, profiling). A weight that silently loses its
    // name would just vanish from any such report, so assert the names are exact,
    // not merely present.
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const path = "models/diffusion_model/krea2CenterSemiraw_v10Fp8.safetensors";
    std.Io.Dir.cwd().access(io, path, .{}) catch return error.SkipZigTest;

    var st = try SafeTensors.open(gpa, io, path);
    defer st.deinit();
    var model = try DiT.load(gpa, .{ .safetensors = &st });
    defer model.deinit();

    // Block 0's eight linears, with the names this checkpoint uses (bare `blocks.N.`;
    // the int8/int4 checkpoints nest under `model.diffusion_model.`, which the loader
    // prefixes, so this also pins that the tag is the RESOLVED name, not the format).
    const b0 = model.blocks[0];
    const expected = [_]struct { w: Weight, name: []const u8 }{
        .{ .w = b0.attn.wq, .name = "blocks.0.attn.wq.weight" },
        .{ .w = b0.attn.wk, .name = "blocks.0.attn.wk.weight" },
        .{ .w = b0.attn.wv, .name = "blocks.0.attn.wv.weight" },
        .{ .w = b0.attn.wo, .name = "blocks.0.attn.wo.weight" },
        .{ .w = b0.attn.gate, .name = "blocks.0.attn.gate.weight" },
        .{ .w = b0.mlp.gate, .name = "blocks.0.mlp.gate.weight" },
        .{ .w = b0.mlp.up, .name = "blocks.0.mlp.up.weight" },
        .{ .w = b0.mlp.down, .name = "blocks.0.mlp.down.weight" },
    };
    for (expected) |e| {
        errdefer std.debug.print("tag mismatch: want {s}, got {?s}\n", .{ e.name, e.w.tag });
        try std.testing.expectEqualStrings(e.name, e.w.tag orelse return error.MissingTag);
    }

    // Distinct per block, or per-layer attribution would collapse layers together.
    try std.testing.expectEqualStrings("blocks.27.mlp.down.weight", model.blocks[27].mlp.down.tag.?);

    // The two projections that get materialized to f32 at load (`first`, `last.linear`,
    // see opMatmulF32) must keep their tags through that copy.
    try std.testing.expectEqualStrings("first.weight", model.first.w.tag orelse return error.MissingTag);
    try std.testing.expectEqualStrings("last.linear.weight", model.last_linear.w.tag orelse return error.MissingTag);

    // And every remaining matmul weight is tagged, so nothing is unattributable.
    for ([_]Weight{ model.tmlp0.w, model.tmlp2.w, model.tproj1.w, model.txtmlp1.w, model.txtmlp3.w, model.txt_projector }) |w| {
        try std.testing.expect(w.tag != null);
    }
    for (model.txt_layerwise, model.txt_refiner) |lw, rf| {
        try std.testing.expect(lw.attn.wq.tag != null and lw.mlp.down.tag != null);
        try std.testing.expect(rf.attn.wq.tag != null and rf.mlp.down.tag != null);
    }
}

test "a DiT loads through an overlay with exactly one weight substituted" {
    // The composition the per-layer attribution arm rests on: a base checkpoint
    // plus one tensor from memory. Two things have to hold, the patched weight is
    // the caller's buffer at the caller's dtype, and *nothing else moves*. The
    // second is the one that would quietly ruin an attribution measurement, so it
    // is asserted pointer-by-pointer against an unpatched load of the same file.
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const path = "models/diffusion_model/krea2CenterSemiraw_v10Fp8.safetensors";
    std.Io.Dir.cwd().access(io, path, .{}) catch return error.SkipZigTest;

    var st = try SafeTensors.open(gpa, io, path);
    defer st.deinit();
    const base: WeightStore = .{ .safetensors = &st };

    var plain = try DiT.load(gpa, base);
    defer plain.deinit();

    // `first.weight` is [6144, 64], the smallest matmul weight in the model, so
    // the substitute buffer is 1.5 MB rather than 150.
    const n = plain.first.w.rows * plain.first.w.cols;
    const sub = try gpa.alloc(f32, n);
    defer gpa.free(sub);
    for (sub, 0..) |*v, i| v.* = @floatFromInt(i % 7);

    var ov: weights_mod.Overlay = .{ .base = base };
    defer ov.deinit(gpa);
    try ov.put(gpa, "first.weight", .f32, std.mem.sliceAsBytes(sub));

    var patched = try DiT.load(gpa, ov.store());
    defer patched.deinit();

    try std.testing.expectEqual(DType.f32, patched.first.w.dtype);
    try std.testing.expectEqual(std.mem.sliceAsBytes(sub).ptr, patched.first.w.bytes.ptr);
    try std.testing.expectEqual(plain.first.w.rows, patched.first.w.rows);
    try std.testing.expectEqual(plain.first.w.cols, patched.first.w.cols);
    try std.testing.expectEqualStrings("first.weight", patched.first.w.tag.?);

    // Every other matmul weight is the same view into the same mapping.
    for (plain.blocks, patched.blocks) |a, b| {
        for ([_]Weight{ a.attn.wq, a.attn.wk, a.attn.wv, a.attn.wo, a.attn.gate, a.mlp.gate, a.mlp.up, a.mlp.down }, //
            [_]Weight{ b.attn.wq, b.attn.wk, b.attn.wv, b.attn.wo, b.attn.gate, b.mlp.gate, b.mlp.up, b.mlp.down }) |wa, wb|
        {
            try std.testing.expectEqual(wa.bytes.ptr, wb.bytes.ptr);
            try std.testing.expectEqual(wa.dtype, wb.dtype);
        }
    }
    try std.testing.expectEqual(plain.tproj1.w.bytes.ptr, patched.tproj1.w.bytes.ptr);
    // `last.linear` is the other weight materialized to f32 at load, so it lands at
    // a fresh arena address per load and can only be compared by value.
    try std.testing.expectEqualSlices(u8, plain.last_linear.w.bytes, patched.last_linear.w.bytes);

    // And dropping the patch restores the base tensor exactly. Compared by value,
    // not by pointer: `first`/`last.linear` are materialized to f32 into the model
    // arena at load (opMatmulF32), so a second load legitimately lands at a
    // different address, which is also why the patched load above compares equal
    // to the patch buffer (an f32 patch passes through untouched).
    try std.testing.expect(ov.remove("first.weight"));
    var restored = try DiT.load(gpa, ov.store());
    defer restored.deinit();
    try std.testing.expectEqual(plain.first.w.dtype, restored.first.w.dtype);
    try std.testing.expectEqualSlices(u8, plain.first.w.bytes, restored.first.w.bytes);
}

// The int8 convrot weights should reconstruct the same linear map as the fp8
// weights (both quantize the same base checkpoint), so a GEMM through each must
// agree to within quantization noise. This validates the whole int8 path,
// loader, per-row scale, and group un-rotation, against the trusted fp8 path.
test "int8 convrot matmul agrees with fp8 within quant noise" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const i8_path = "models/diffusion_model/krea2CenterSemiraw_v10Int8.safetensors";
    const fp8_path = "models/diffusion_model/krea2CenterSemiraw_v10Fp8.safetensors";
    std.Io.Dir.cwd().access(io, i8_path, .{}) catch return error.SkipZigTest;
    std.Io.Dir.cwd().access(io, fp8_path, .{}) catch return error.SkipZigTest;

    var st_i8 = try SafeTensors.open(gpa, io, i8_path);
    defer st_i8.deinit();
    var m_i8 = try DiT.load(gpa, .{ .safetensors = &st_i8 });
    defer m_i8.deinit();
    var st_fp8 = try SafeTensors.open(gpa, io, fp8_path);
    defer st_fp8.deinit();
    var m_fp8 = try DiT.load(gpa, .{ .safetensors = &st_fp8 });
    defer m_fp8.deinit();

    const w_i8 = m_i8.blocks[0].attn.wq;
    const w_fp8 = m_fp8.blocks[0].attn.wq;
    try std.testing.expectEqual(w_fp8.rows, w_i8.rows);
    try std.testing.expectEqual(w_fp8.cols, w_i8.cols);

    const rows_m = 4;
    const x = try gpa.alloc(f32, rows_m * w_i8.cols);
    defer gpa.free(x);
    var prng = std.Random.DefaultPrng.init(1234);
    for (x) |*v| v.* = prng.random().floatNorm(f32);

    const y_i8 = try gpa.alloc(f32, rows_m * w_i8.rows);
    defer gpa.free(y_i8);
    const y_fp8 = try gpa.alloc(f32, rows_m * w_fp8.rows);
    defer gpa.free(y_fp8);
    try ops.matmul.matmul(io, gpa, y_i8, x, rows_m, w_i8, null);
    try ops.matmul.matmul(io, gpa, y_fp8, x, rows_m, w_fp8, null);

    var num: f64 = 0;
    var den: f64 = 0;
    for (y_fp8, y_i8) |ref, got| {
        num += @as(f64, (ref - got)) * (ref - got);
        den += @as(f64, ref) * ref;
    }
    const rel = @sqrt(num / den);
    // Diagnostic only on failure: stderr from a passing test makes the build
    // runner print a spurious red "failed command:" line.
    errdefer std.debug.print("int8-vs-fp8 wq GEMM relative RMSE: {d:.4}\n", .{rel});
    try std.testing.expect(rel < 0.05);
}

const w4a8_real_layer_json = @embedFile("assets/w4a8_real_layer.json");

fn fnv1a64(bytes: []const u8) u64 {
    var h: u64 = 0xcbf29ce484222325;
    for (bytes) |b| h = (h ^ b) *% 0x100000001b3;
    return h;
}

test "a W4A8 layer of a real checkpoint decodes to comfy_kitchen's int8 weight" {
    // The synthetic tier in `ops/w4a8.zig` pins the arithmetic; this pins the
    // CONTAINER half, which is what the loader adds and what nothing else can see:
    // the [N, K/2] shape doubling, `weight_s_rel` read as fp8 rather than as u8, the
    // codebook tensor, and `group_size` derived from the scale's own shape and
    // cross-checked against `comfy_quant`.
    //
    // Goes through `Loader.mat` on ONE tensor rather than `DiT.load`, deliberately:
    // the whole model decodes to 12.2 GB, which in a Debug test binary is neither
    // affordable nor necessary to check the format.
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const path = "models/diffusion_model/krea2CenterSemiraw_v10Int8-ASYM_W4A8_INT8.safetensors";
    try test_gate.requireModelFile(io, path);

    const Ref = struct {
        layer: []const u8,
        rows: usize,
        cols: usize,
        group_size: usize,
        convrot_groupsize: usize,
        packed_fnv1a64: u64,
        s_rel_fnv1a64: u64,
        s_channel_fnv1a64: u64,
        expect_i8_fnv1a64: u64,
        expect_i8_head: []const i32,
        expect_i8_tail: []const i32,
    };
    var parsed = try std.json.parseFromSlice(Ref, gpa, w4a8_real_layer_json, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const ref = parsed.value;
    // The fixture is generated for one layer; if that constant ever moves, the
    // expectations below are about a different tensor.
    try std.testing.expectEqualStrings("blocks.0.attn.wk", ref.layer);

    var st = try SafeTensors.open(gpa, io, path);
    defer st.deinit();
    const store: WeightStore = .{ .safetensors = &st };

    // Check the INPUTS first. A mismatch here means the checkpoint changed, which is a
    // different fact from "the decode is wrong", and the fixture cannot tell them
    // apart after the fact.
    try std.testing.expectEqual(ref.packed_fnv1a64, fnv1a64((try store.require("blocks.0.attn.wk.weight")).bytes));
    try std.testing.expectEqual(ref.s_rel_fnv1a64, fnv1a64((try store.require("blocks.0.attn.wk.weight_s_rel")).bytes));
    try std.testing.expectEqual(ref.s_channel_fnv1a64, fnv1a64((try store.require("blocks.0.attn.wk.weight_s_channel")).bytes));

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const l = Loader{ .store = store, .alloc = arena.allocator(), .pfx = "" };
    const w = try l.mat("blocks.0.attn.wk.weight", .{}, ref.rows, ref.cols);

    // The weight stays PACKED, half the logical element count in bytes, with the
    // sidecars a decode needs hanging off it. That is what keeps the format at its own
    // bit width instead of int8's.
    try std.testing.expectEqual(@as(DType, .w4a8), w.dtype);
    try std.testing.expectEqual(ref.rows, w.rows);
    try std.testing.expectEqual(ref.cols, w.cols);
    try std.testing.expectEqual(@as(u32, @intCast(ref.convrot_groupsize)), w.convrot);
    try std.testing.expectEqual(ref.rows * ref.cols / 2, w.bytes.len);
    try std.testing.expect(w.row_scale != null);
    try std.testing.expectEqual(ref.rows, w.row_scale.?.len);
    try std.testing.expect(w.w4a8 != null);
    try std.testing.expectEqual(@as(u32, @intCast(ref.group_size)), w.w4a8.?.group_size);
    try std.testing.expectEqual(ref.rows * ref.cols / ref.group_size, w.w4a8.?.s_rel.len);

    // Decode it here, since the loader keeps the weight packed, and check that against
    // the reference. This is what pins the metadata wiring: a `s_rel` slice off by a row
    // or a codebook read from the wrong tensor shows up only in the decoded values.
    const got = try arena.allocator().alloc(i8, ref.rows * ref.cols);
    ops.w4a8.decode(got, w.bytes, w.w4a8.?.s_rel, w.w4a8.?.levels, ref.rows, ref.cols, ref.group_size);
    // Head and tail first: a bare hash mismatch says nothing about where, and these
    // localize it to the first or last row.
    for (ref.expect_i8_head, 0..) |want, i| {
        errdefer std.debug.print("element {d}\n", .{i});
        try std.testing.expectEqual(@as(i8, @intCast(want)), got[i]);
    }
    for (ref.expect_i8_tail, 0..) |want, i| {
        const idx = got.len - ref.expect_i8_tail.len + i;
        errdefer std.debug.print("element {d} (of {d})\n", .{ idx, got.len });
        try std.testing.expectEqual(@as(i8, @intCast(want)), got[idx]);
    }
    try std.testing.expectEqual(ref.expect_i8_fnv1a64, fnv1a64(std.mem.sliceAsBytes(got)));
}

test "int4 convrot checkpoint loads with per-row scale + rotation metadata" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    // ComfyUI's official W4A4 converter (packed nibbles stored as I8). This is a
    // mixed int8/int4 checkpoint: the per-block linears are packed int4, while
    // txtfusion stays bf16, so it also exercises the per-layer int8/int4
    // disambiguation in Loader.mat.
    const path = "models/diffusion_model/krea2CenterSemiraw_v10Int8-INT4_CONVROT_SR.safetensors";
    std.Io.Dir.cwd().access(io, path, .{}) catch return error.SkipZigTest;

    var st = try SafeTensors.open(gpa, io, path);
    defer st.deinit();
    var model = try DiT.load(gpa, .{ .safetensors = &st });
    defer model.deinit();

    // The per-block linears are int4 convrot: stored I8 (nibble-packed, two
    // values per byte) but reinterpreted as .i4 with the logical [rows, cols],
    // scale + rotation wired.
    const attn = model.blocks[0].attn;
    for ([_]Weight{ attn.wq, attn.wk, attn.wv, attn.wo, attn.gate }) |w| {
        try std.testing.expect(w.dtype == .i4);
        try std.testing.expectEqual(@as(u32, ops.convrot.group_size), w.convrot);
        try std.testing.expect(w.row_scale != null);
        try std.testing.expectEqual(w.rows, w.row_scale.?.len);
        try std.testing.expectEqual(@as(usize, 0), w.cols % ops.convrot.group_size);
        // On-disk bytes are exactly half the logical element count (2 per byte).
        try std.testing.expectEqual(w.rows * w.cols / 2, w.bytes.len);
    }
    try std.testing.expect(model.first.w.dtype == .f32);
    try std.testing.expect(model.first.w.row_scale == null);
}

// Like the int8 test: the int4 convrot weights quantize the same base
// checkpoint as fp8, so a GEMM through each must agree, but int4's 16 levels
// give a looser bound than int8's 256. This validates the whole int4 path
// (loader, I8->i4 reinterpret, nibble unpack, per-row scale, group un-rotation).
test "int4 convrot matmul agrees with fp8 within quant noise" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const i4_path = "models/diffusion_model/krea2CenterSemiraw_v10Int8-INT4_CONVROT_SR.safetensors";
    const fp8_path = "models/diffusion_model/krea2CenterSemiraw_v10Fp8.safetensors";
    std.Io.Dir.cwd().access(io, i4_path, .{}) catch return error.SkipZigTest;
    std.Io.Dir.cwd().access(io, fp8_path, .{}) catch return error.SkipZigTest;

    var st_i4 = try SafeTensors.open(gpa, io, i4_path);
    defer st_i4.deinit();
    var m_i4 = try DiT.load(gpa, .{ .safetensors = &st_i4 });
    defer m_i4.deinit();
    var st_fp8 = try SafeTensors.open(gpa, io, fp8_path);
    defer st_fp8.deinit();
    var m_fp8 = try DiT.load(gpa, .{ .safetensors = &st_fp8 });
    defer m_fp8.deinit();

    const w_i4 = m_i4.blocks[0].attn.wq;
    const w_fp8 = m_fp8.blocks[0].attn.wq;
    try std.testing.expectEqual(w_fp8.rows, w_i4.rows);
    try std.testing.expectEqual(w_fp8.cols, w_i4.cols);

    const rows_m = 4;
    const x = try gpa.alloc(f32, rows_m * w_i4.cols);
    defer gpa.free(x);
    var prng = std.Random.DefaultPrng.init(1234);
    for (x) |*v| v.* = prng.random().floatNorm(f32);

    const y_i4 = try gpa.alloc(f32, rows_m * w_i4.rows);
    defer gpa.free(y_i4);
    const y_fp8 = try gpa.alloc(f32, rows_m * w_fp8.rows);
    defer gpa.free(y_fp8);
    try ops.matmul.matmul(io, gpa, y_i4, x, rows_m, w_i4, null);
    try ops.matmul.matmul(io, gpa, y_fp8, x, rows_m, w_fp8, null);

    var num: f64 = 0;
    var den: f64 = 0;
    for (y_fp8, y_i4) |ref, got| {
        num += @as(f64, (ref - got)) * (ref - got);
        den += @as(f64, ref) * ref;
    }
    const rel = @sqrt(num / den);
    errdefer std.debug.print("int4-vs-fp8 wq GEMM relative RMSE: {d:.4}\n", .{rel});
    // int4 (16 levels) is much coarser than int8; convrot keeps it usable but
    // the GEMM-level relative error is naturally several × higher. This is a
    // sanity bound (garbage from a wrong rotation/packing would land near ~1.0,
    // uncorrelated), the tight bit-exact check lives in the convrot fixture
    // test. ComfyUI's official W4A4 file is quantized independently of the fp8
    // reference, so it sits a touch above the old home-grown checkpoint's ~0.25.
    try std.testing.expect(rel < 0.30);
}

fn readF32File(gpa: std.mem.Allocator, io: std.Io, path: []const u8, n: usize) ![]f32 {
    const out = try gpa.alloc(f32, n);
    errdefer gpa.free(out);
    const file = try std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_only });
    defer file.close(io);
    const bytes = std.mem.sliceAsBytes(out);
    if (try file.readPositionalAll(io, bytes, 0) != bytes.len) return error.ShortRead;
    return out;
}

// Full-forward parity against ComfyUI (tools/dump_dit_fixture.py). ~28 GEMM
// blocks of a 12B model in Debug mode is minutes of work, so this only runs
// when the marker file `testdata/slow-tests` exists (touch it to enable),
// in addition to model/fixture presence.
test "dit forward matches comfyui" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    std.Io.Dir.cwd().access(io, "testdata/slow-tests", .{}) catch return error.SkipZigTest;
    const dit_path = "models/diffusion_model/krea2CenterSemiraw_v10Fp8.safetensors";
    std.Io.Dir.cwd().access(io, dit_path, .{}) catch return error.SkipZigTest;
    std.Io.Dir.cwd().access(io, "testdata/dit_out.bin", .{}) catch return error.SkipZigTest;

    const seq_txt = 14;
    const x_lat = try readF32File(gpa, io, "testdata/dit_x.bin", channels * 16 * 16);
    defer gpa.free(x_lat);
    const expected = try readF32File(gpa, io, "testdata/dit_out.bin", channels * 16 * 16);
    defer gpa.free(expected);
    const ctx = try readF32File(gpa, io, "testdata/text_cond.bin", seq_txt * txt_layers * txt_dim);
    defer gpa.free(ctx);

    var st = try SafeTensors.open(gpa, io, dit_path);
    defer st.deinit();
    var model = try DiT.load(gpa, .{ .safetensors = &st });
    defer model.deinit();

    const out = try gpa.alloc(f32, channels * 16 * 16);
    defer gpa.free(out);
    try model.forward(io, gpa, out, x_lat, 16, 16, 0.875, ctx, seq_txt, null);

    var max_err: f32 = 0;
    var max_val: f32 = 0;
    var sum_err: f64 = 0;
    for (expected, out) |e, a| {
        max_err = @max(max_err, @abs(e - a));
        max_val = @max(max_val, @abs(e));
        sum_err += @abs(e - a);
    }
    const mean_err = sum_err / @as(f64, @floatFromInt(out.len));
    errdefer std.debug.print("dit parity: max_err={d:.5} mean_err={d:.6} max_val={d:.2}\n", .{ max_err, mean_err, max_val });
    try std.testing.expect(max_err < 0.01 * @max(1.0, max_val));
    try std.testing.expect(mean_err < 1e-3 * @as(f64, @max(1.0, max_val)));

    // A pre-set cancel flag aborts mid-step (polled between blocks).
    var canceled = std.atomic.Value(bool).init(true);
    try std.testing.expectError(error.Canceled, model.forward(io, gpa, out, x_lat, 16, 16, 0.875, ctx, seq_txt, &canceled));
}

test "a GGUF checkpoint loads, with its block-quant dtypes intact" {
    // The GGUF half of ggufy's output (the ggml block quants) could not be loaded
    // at all before `load` took a `WeightStore`, which meant the whole k-quant
    // path had level-1 evidence and nothing else. This is the receipt that it is
    // reachable: same bare tensor names as the safetensors checkpoints, dims
    // un-reversed by the GGUF reader into [rows, cols], and per-layer mixed types
    // from ggufy's sensitivity routing preserved as-is for the GEMM to dequantize.
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const gguf = @import("tp_core").gguf;
    const path = "models/diffusion_model/anim-q4k-calib.gguf";
    std.Io.Dir.cwd().access(io, path, .{}) catch return error.SkipZigTest;

    var g = try gguf.Gguf.open(gpa, io, path);
    defer g.deinit();
    var model = try DiT.load(gpa, .{ .gguf = &g });
    defer model.deinit();

    try std.testing.expectEqual(@as(usize, n_blocks), model.blocks.len);

    // Every per-block linear must be a ggml block quant with no convrot metadata:
    // the k-quants carry their scales inside the block, so a row_scale here would
    // mean the int8/int4 path had misfired on a GGUF tensor.
    var quantized: usize = 0;
    for (model.blocks) |blk| {
        for ([_]Weight{ blk.attn.wq, blk.attn.wk, blk.attn.wv, blk.attn.wo, blk.mlp.gate, blk.mlp.up, blk.mlp.down }) |w| {
            try std.testing.expect(w.row_scale == null);
            try std.testing.expectEqual(@as(u32, 0), w.convrot);
            switch (w.dtype) {
                .q4_0, .q2_k, .q4_k, .q5_k, .q6_k, .q8_0, .iq4_nl, .f16, .bf16, .f32 => {},
                else => {
                    std.debug.print("unexpected GGUF dtype {t} for {s}\n", .{ w.dtype, w.tag orelse "?" });
                    return error.UnexpectedDtype;
                },
            }
            if (w.dtype != .f32 and w.dtype != .f16 and w.dtype != .bf16) quantized += 1;
        }
    }
    // Mixed precision is the point of the sensitivity routing: most linears are
    // quantized, and the file is useless as a measurement target if none are.
    try std.testing.expect(quantized > n_blocks * 5);
}

test "the GPU DiT paths refuse a block-quant checkpoint instead of misreading it" {
    // Both GPU forwards recognize int8/int4/bf16 and treat everything else as raw
    // fp8 bytes. A GGUF checkpoint is neither, and before `pipeline` could open a
    // GGUF the case was unreachable, so the day it became reachable, Vulkan
    // rendered a blank white image with no error, while CUDA (which already had the
    // gate) refused. This pins both to refusing.
    //
    // Checks the *gate*, not a device: it asserts the dtype classification both
    // forwards do, so it runs on the fast suite with no GPU and no checkpoint.
    const supported = [_]DType{ .i8, .i4, .bf16, .f8_e4m3 };
    // The block quants the CUDA arm decodes per GEMM are checked separately below. These
    // are the ones with no diffusion dequant kernel at all, still fp8-shaped garbage on
    // both arms. `.q2_0_g64`/`.q2_0_g128` are LLM-only formats no diffusion quantizer
    // emits, so they belong here rather than in the CUDA list.
    const block_quants = [_]DType{ .q2_0_g64, .q2_0_g128, .q1_0 };

    for (supported) |dt| {
        try std.testing.expect(gpuLinKindSupported(dt, .vulkan));
        try std.testing.expect(gpuLinKindSupported(dt, .cuda));
    }
    for (block_quants) |dt| {
        for ([_]GpuArm{ .vulkan, .cuda }) |arm| {
            std.testing.expect(!gpuLinKindSupported(dt, arm)) catch |e| {
                std.debug.print("block quant {t} would be misread as fp8 by the {t} DiT forward\n", .{ dt, arm });
                return e;
            };
        }
    }
    // The decodable block quants are CUDA-only. Vulkan has no block-quant GEMM, so
    // accepting one there feeds the packed bytes to the fp8 GEMM: a blank white image
    // with no error, which is what this test exists to catch.
    for ([_]DType{ .q4_0, .q8_0, .q2_k, .q4_k, .q5_k, .q6_k, .iq4_nl }) |dt| {
        try std.testing.expect(gpuLinKindSupported(dt, .cuda));
        try std.testing.expect(!gpuLinKindSupported(dt, .vulkan));
    }
    // f32 block linears are not a thing a checkpoint ships, and the GPU paths have
    // no branch for them either.
    try std.testing.expect(!gpuLinKindSupported(.f32, .vulkan));
    try std.testing.expect(!gpuLinKindSupported(.f32, .cuda));
}
