//! Mage-VAE, the codec Mage-Flow's 128-channel / 16x latents live in, mirroring
//! comfy/ldm/mage_flow/vae.py.
//!
//! Not an `AutoencoderKL` and not a CNN ladder: it is a ONE-STEP DIFFUSION codec,
//! so both directions are a single forward of a denoiser at t = 0 with zero
//! noise, and neither direction changes resolution by convolution. Everything
//! happens at one of two grids, the latent grid and the image grid, joined by a
//! 16x16 patch:
//!
//!   decode  latent -> CoD decoder -> per-patch conditioning -> 21 DiCo blocks
//!           -> a NeRF-style per-pixel MLP inside each 16x16 patch -> RGB
//!   encode  image  -> 16x16 patch embed -> 2 head blocks -> 21 DiCo blocks
//!           -> [mean | logvar], of which only the mean is kept
//!
//! Four things here are silent wrong answers if carried over from any other VAE
//! in this tree:
//!
//!   - The image-side pathway is per PIXEL WITHIN A PATCH, not per pixel of the
//!     image. A patch's 256 positions are rows of one small MLP conditioned on
//!     that patch's 384-wide vector, and the position is supplied by a fixed
//!     cosine (DCT) table, not by a convolution. Flatten the patch in the wrong
//!     order and every 16x16 tile comes out internally scrambled while the image
//!     still looks globally right.
//!   - `DiCoBlock`'s 3x3 is DEPTHWISE (`groups = channels`). Run as a dense
//!     convolution it is a different, much larger operator that still trains-free
//!     runs and produces plausible mush.
//!   - The CoD decoder's attention is WINDOWED at 32x32 with replicate padding,
//!     not global. The window is 32x32 whatever the latent is, so a SMALLER
//!     latent is padded UP and its edge rows attend to their own duplicates;
//!     the window is not merely a memory split that vanishes on small inputs.
//!   - `F.gelu` here is the ERF form, not the tanh approximation the DiT uses.
//!
//! Decode's noise input is zero, so the `s_embedder`'s bias-free image
//! projection contributes nothing and the NeRF input's first 3 features are
//! zero. It is written out anyway rather than folded away: the reference is a
//! denoiser and this keeps the two readable side by side.
//!
//! Activations are channel-last `[h][w][c]` throughout, the layout
//! `ops.conv` works in. Weights are materialized to f32 at load, like
//! `sd_vae`; the store's mapping need not outlive the model.

const std = @import("std");
const tp_core = @import("tp_core");
const safetensors = tp_core.safetensors;
const weights_mod = tp_core.weights;
const ops = @import("tp_ops");

const WeightStore = weights_mod.WeightStore;
const Weight = ops.matmul.Weight;
const Conv2d = ops.conv.Conv2d;

pub const latent_channels = 128;
pub const spatial_scale = 16;
/// Pixels per patch side, which is also `spatial_scale`: one latent cell is one
/// patch.
pub const patch = 16;
pub const patch_area = patch * patch;

/// `DConvDenoiser` / `DConvEncoder` trunk width.
pub const hidden = 384;
/// The NeRF pathway's width, which is also how many of a patch's conditioning
/// channels reach a single pixel.
pub const hidden_x = 32;
pub const ffn = 4 * hidden;
pub const n_blocks = 21;
/// `num_blocks - num_cond_blocks` in the reference: the per-pixel MLP's depth.
pub const n_res_blocks = 3;
pub const max_freqs = 8;
/// Position features per pixel: an 8x8 cosine table.
pub const dct_dim = max_freqs * max_freqs;
/// `x_embedder` input: 3 image channels + `hidden_x` patch channels + the table.
pub const nerf_in = 3 + hidden_x + dct_dim;
pub const t_freq = 256;
/// The CoD decoder attends inside 32x32 windows of the LATENT grid.
pub const attn_window = 32;
pub const norm_groups = 32;

/// Encoder head width, before `proj_down` to `hidden`.
pub const enc_head = 768;
pub const n_head_blocks = 2;
/// `bottleneck_dim`: the image patch embed inside the decoder's `s_embedder`.
pub const bottleneck = 128;

const eps: f32 = 1e-6;

/// The tensor that says "this is a Mage-VAE", ComfyUI's own test in `sd.py`.
pub const probe = "student.dconv_encoder.proj_out.weight";

/// How many image pixels one band of the NeRF pathway may hold. The pathway is
/// per-token independent, so banding it changes only the peak: at a 1024x1024
/// decode the whole thing at once is ~700 MB of intermediates.
const nerf_band_bytes: usize = 64 << 20;

/// `latent_formats.Flux2`'s preview matrix, `[preview_channels][rgb]`. Only 32
/// rows for a 128-channel latent, because the 128 are really 32 features over a
/// 2x2 spatial sub-block: channel `c` is feature `c / 4` at sub-position
/// `(c % 4) / 2, (c % 4) % 2`.
pub const preview_channels = 32;
pub const latent_rgb_factors = [preview_channels][3]f32{
    .{ 0.0058, 0.0113, 0.0073 },
    .{ 0.0495, 0.0443, 0.0836 },
    .{ -0.0099, 0.0096, 0.0644 },
    .{ 0.2144, 0.3009, 0.3652 },
    .{ 0.0166, -0.0039, -0.0054 },
    .{ 0.0157, 0.0103, -0.0160 },
    .{ -0.0398, 0.0902, -0.0235 },
    .{ -0.0052, 0.0095, 0.0109 },
    .{ -0.3527, -0.2712, -0.1666 },
    .{ -0.0301, -0.0356, -0.0180 },
    .{ -0.0107, 0.0078, 0.0013 },
    .{ 0.0746, 0.0090, -0.0941 },
    .{ 0.0156, 0.0169, 0.0070 },
    .{ -0.0034, -0.0040, -0.0114 },
    .{ 0.0032, 0.0181, 0.0080 },
    .{ -0.0939, -0.0008, 0.0186 },
    .{ 0.0018, 0.0043, 0.0104 },
    .{ 0.0284, 0.0056, -0.0127 },
    .{ -0.0024, -0.0022, -0.0030 },
    .{ 0.1207, -0.0026, 0.0065 },
    .{ 0.0128, 0.0101, 0.0142 },
    .{ 0.0137, -0.0072, -0.0007 },
    .{ 0.0095, 0.0092, -0.0059 },
    .{ 0.0000, -0.0077, -0.0049 },
    .{ -0.0465, -0.0204, -0.0312 },
    .{ 0.0095, 0.0012, -0.0066 },
    .{ 0.0290, -0.0034, 0.0025 },
    .{ 0.0220, 0.0169, -0.0048 },
    .{ -0.0332, -0.0457, -0.0468 },
    .{ -0.0085, 0.0389, 0.0609 },
    .{ -0.0076, 0.0003, -0.0043 },
    .{ -0.0111, -0.0460, -0.0614 },
};
pub const latent_rgb_bias = [3]f32{ -0.0329, -0.0718, -0.0851 };

/// Cheap latent preview, `z` planar `[128][zh][zw]` -> `rgb_out` `[zh][zw][3]`.
///
/// ComfyUI draws this at TWICE the latent grid, unpacking the 2x2 sub-block into
/// real pixels. The caller here wants one preview pixel per latent cell, so the
/// four sub-positions are averaged first, which is exactly a box downscale of
/// ComfyUI's picture rather than a different matrix.
pub fn latentPreviewInto(rgb_out: []u8, z: []const f32, zh: usize, zw: usize) void {
    const plane = zh * zw;
    std.debug.assert(rgb_out.len >= plane * 3 and z.len >= latent_channels * plane);
    for (0..plane) |p| {
        var acc = latent_rgb_bias;
        inline for (0..preview_channels) |f| {
            var v: f32 = 0;
            inline for (0..4) |sub| v += z[(f * 4 + sub) * plane + p];
            v *= 0.25;
            acc[0] += v * latent_rgb_factors[f][0];
            acc[1] += v * latent_rgb_factors[f][1];
            acc[2] += v * latent_rgb_factors[f][2];
        }
        inline for (0..3) |ch| {
            const u = std.math.clamp((acc[ch] + 1.0) * 0.5, 0.0, 1.0) * 255.0;
            rgb_out[p * 3 + ch] = @intFromFloat(u);
        }
    }
}

pub const LinearW = struct {
    w: Weight,
    b: ?[]const f32,
};

pub const NormW = struct { w: []const f32, b: []const f32 };

/// `DiCoBlock`: depthwise-conv residual with adaLN modulation and a
/// channel-attention gate.
pub const DiCo = struct {
    /// silu(c) -> 6 * hidden: [shift1, scale1, gate1, shift2, scale2, gate2].
    /// Null for the encoder's head blocks, which have no conditioning and carry
    /// their own affine norms instead.
    adaln: ?LinearW,
    norm1: ?NormW,
    norm2: ?NormW,
    conv1: Conv2d,
    /// `[c][k][k]` depthwise 3x3, pad 1. Kept in torch order, see
    /// `ops.conv.depthwiseConv2d`.
    dw: []const f32,
    dw_b: []const f32,
    conv3: Conv2d,
    /// Channel attention: global mean -> 1x1 -> sigmoid -> scale.
    ca: LinearW,
    conv4: Conv2d,
    conv5: Conv2d,
    ch: usize,
};

/// LDM's autoencoder `ResnetBlock`, width-preserving here.
pub const Resnet = struct {
    norm1: NormW,
    conv1: Conv2d,
    norm2: NormW,
    conv2: Conv2d,
};

pub const WinAttn = struct {
    norm: NormW,
    q: Conv2d,
    k: Conv2d,
    v: Conv2d,
    proj: Conv2d,
};

/// `CoDDecoder`: latent -> the per-patch conditioning the denoiser runs on.
pub const Cod = struct {
    conv_in: Conv2d,
    r0: Resnet,
    a0: WinAttn,
    r1: Resnet,
    a1: WinAttn,
    r2: Resnet,
    norm_out: NormW,
    conv_out: Conv2d,
};

/// `SimpleMLPAdaLN`'s residual block, over `hidden_x`.
pub const MlpRes = struct {
    /// silu(c) -> 3 * hidden_x: [shift, scale, gate].
    adaln: LinearW,
    in_ln: NormW,
    fc1: LinearW,
    fc2: LinearW,
};

pub const TimeMlp = struct {
    fc1: LinearW, // t_freq -> hidden
    fc2: LinearW, // hidden -> hidden
};

pub const MageVae = struct {
    arena: std.heap.ArenaAllocator,

    // --- decode --------------------------------------------------------
    cod: Cod,
    t_embed: TimeMlp,
    /// `s_embedder`: a bias-free 16x16 patch embed of the image, concatenated
    /// with the conditioning and projected back to `hidden`.
    s_proj1: Conv2d,
    s_proj2: Conv2d,
    blocks: []DiCo,
    /// 1x1 `hidden` -> `hidden_x * patch_area`: the patch's per-pixel channels.
    y_embedder_x: LinearW,
    x_embedder: LinearW, // nerf_in -> hidden_x
    cond_embed: LinearW, // hidden -> hidden_x * patch_area
    input_proj: LinearW, // hidden_x -> hidden_x
    res_blocks: []MlpRes,
    final_norm: []const f32, // RMSNorm weight over hidden_x
    final_lin: LinearW, // hidden_x -> 3

    // --- encode --------------------------------------------------------
    /// Null when the checkpoint ships decode only.
    enc: ?Encoder,

    const Encoder = struct {
        patch_cond: Conv2d, // 3 -> enc_head, k16 s16
        head_blocks: []DiCo,
        proj_down: Conv2d, // 1x1 enc_head -> hidden
        z_proj: Conv2d, // 1x1 latent_channels -> hidden
        fuse: Conv2d, // 1x1 2*hidden -> hidden
        t_embed: TimeMlp,
        blocks: []DiCo,
        norm_out: NormW,
        proj_out: Conv2d, // 1x1 hidden -> 2 * latent_channels
    };

    /// Whether this store holds a Mage-VAE at all.
    pub fn fits(store: WeightStore, prefix: []const u8) bool {
        var buf: [200]u8 = undefined;
        const nm = std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, probe }) catch return false;
        return store.get(nm) != null;
    }

    /// `prefix` is where the codec sits: `""` for a bare VAE export, `vae.` or
    /// `first_stage_model.` inside a bundled checkpoint.
    pub fn load(gpa: std.mem.Allocator, store: WeightStore, prefix: []const u8) !MageVae {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();
        const l = Loader{ .store = store, .alloc = alloc, .pfx = prefix };

        // Every field into a local first: `.arena = arena` in a struct literal
        // copies the arena's state, so anything a later field allocates is lost.
        const cod = try l.loadCod("pipeline.y_embedder.decoder");
        const t_embed = try l.timeMlp("pipeline.t_embedder", hidden);
        const s_proj1 = try l.convNoBias("pipeline.s_embedder.proj1", .{}, bottleneck, 3, patch, patch, 0);
        const s_proj2 = try l.conv("pipeline.s_embedder.proj2", .{}, hidden, bottleneck + hidden, 1, 1, 0);

        const blocks = try alloc.alloc(DiCo, n_blocks);
        for (blocks, 0..) |*b, i| b.* = try l.dico("pipeline.blocks.{d}", .{i}, hidden, true);

        const y_embedder_x = try l.linear("pipeline.y_embedder_x", .{}, hidden_x * patch_area, hidden);
        const x_embedder = try l.linear("pipeline.x_embedder.embedder.0", .{}, hidden_x, nerf_in);
        const cond_embed = try l.linear("pipeline.dec_net.cond_embed", .{}, hidden_x * patch_area, hidden);
        const input_proj = try l.linear("pipeline.dec_net.input_proj", .{}, hidden_x, hidden_x);

        const res_blocks = try alloc.alloc(MlpRes, n_res_blocks);
        for (res_blocks, 0..) |*r, i| r.* = .{
            .adaln = try l.linear("pipeline.dec_net.res_blocks.{d}.adaLN_modulation.1", .{i}, 3 * hidden_x, hidden_x),
            .in_ln = try l.norm("pipeline.dec_net.res_blocks.{d}.in_ln", .{i}, hidden_x),
            .fc1 = try l.linear("pipeline.dec_net.res_blocks.{d}.mlp.0", .{i}, hidden_x, hidden_x),
            .fc2 = try l.linear("pipeline.dec_net.res_blocks.{d}.mlp.2", .{i}, hidden_x, hidden_x),
        };

        const final_norm = try l.vec("pipeline.final_layer.norm", .{}, ".weight", hidden_x);
        const final_lin = try l.linear("pipeline.final_layer.linear", .{}, 3, hidden_x);

        const enc: ?Encoder = if (l.has("student.dconv_encoder.proj_out.weight")) blk: {
            const head_blocks = try alloc.alloc(DiCo, n_head_blocks);
            for (head_blocks, 0..) |*b, i| b.* = try l.dico("student.dconv_encoder.head_blocks.{d}", .{i}, enc_head, false);
            const enc_blocks = try alloc.alloc(DiCo, n_blocks);
            for (enc_blocks, 0..) |*b, i| b.* = try l.dico("student.dconv_encoder.blocks.{d}", .{i}, hidden, true);
            break :blk .{
                .patch_cond = try l.conv("student.dconv_encoder.patch_cond_embed", .{}, enc_head, 3, patch, patch, 0),
                .head_blocks = head_blocks,
                .proj_down = try l.conv("student.dconv_encoder.proj_down", .{}, hidden, enc_head, 1, 1, 0),
                .z_proj = try l.conv("student.dconv_encoder.z_proj", .{}, hidden, latent_channels, 1, 1, 0),
                .fuse = try l.conv("student.dconv_encoder.fuse_proj", .{}, hidden, 2 * hidden, 1, 1, 0),
                .t_embed = try l.timeMlp("student.dconv_encoder.t_embedder", hidden),
                .blocks = enc_blocks,
                .norm_out = try l.norm("student.dconv_encoder.norm_out", .{}, hidden),
                .proj_out = try l.conv("student.dconv_encoder.proj_out", .{}, 2 * latent_channels, hidden, 1, 1, 0),
            };
        } else null;

        return .{
            .arena = arena,
            .cod = cod,
            .t_embed = t_embed,
            .s_proj1 = s_proj1,
            .s_proj2 = s_proj2,
            .blocks = blocks,
            .y_embedder_x = y_embedder_x,
            .x_embedder = x_embedder,
            .cond_embed = cond_embed,
            .input_proj = input_proj,
            .res_blocks = res_blocks,
            .final_norm = final_norm,
            .final_lin = final_lin,
            .enc = enc,
        };
    }

    pub fn deinit(self: *MageVae) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// Latent `[lat_h][lat_w][latent_channels]` -> RGB `[h][w][3]` in [-1, 1],
    /// `h = lat_h * 16`. Caller frees.
    pub fn decode(
        self: *const MageVae,
        io: std.Io,
        gpa: std.mem.Allocator,
        z: []const f32,
        lat_h: usize,
        lat_w: usize,
    ) ![]f32 {
        const n = lat_h * lat_w;
        std.debug.assert(z.len == n * latent_channels);
        const h = lat_h * patch;
        const w = lat_w * patch;

        const cond = try self.codForward(io, gpa, z, lat_h, lat_w);
        defer gpa.free(cond);

        const c = try timeVector(io, gpa, self.t_embed, 0.0);
        defer gpa.free(c);

        // `s_embedder(noise, cond)`: the noise is zero and `proj1` is bias-free,
        // so its half of the concatenation is zero.
        const s = try gpa.alloc(f32, n * hidden);
        defer gpa.free(s);
        {
            const cat = try gpa.alloc(f32, n * (bottleneck + hidden));
            defer gpa.free(cat);
            for (0..n) |i| {
                const row = cat[i * (bottleneck + hidden) ..];
                @memset(row[0..bottleneck], 0);
                @memcpy(row[bottleneck..][0..hidden], cond[i * hidden ..][0..hidden]);
            }
            try ops.conv.conv2d(io, gpa, s, cat, lat_h, lat_w, self.s_proj2);
        }

        for (self.blocks) |*b| try dicoForward(io, gpa, b, s, lat_h, lat_w, c);

        const dct = try dctTable(gpa);
        defer gpa.free(dct);

        const rgb = try gpa.alloc(f32, h * w * 3);
        errdefer gpa.free(rgb);

        // The per-pixel pathway is independent per latent cell, so it runs in
        // bands; `nerf_band_bytes` caps the widest intermediate.
        const per_token = patch_area * nerf_in * 4;
        const band: usize = @max(1, nerf_band_bytes / per_token);
        var l0: usize = 0;
        while (l0 < n) : (l0 += band) {
            // `: usize` is load-bearing: without it `@min` narrows every
            // multiply below.
            const nb: usize = @min(band, n - l0);
            try self.nerfBand(io, gpa, rgb, cond[l0 * hidden ..][0 .. nb * hidden], s[l0 * hidden ..][0 .. nb * hidden], dct, l0, nb, lat_w);
        }
        return rgb;
    }

    /// Every DiCo block's modulation, laid out for the device AdaLN kernels:
    /// `[n_blocks][6][hidden]` as `premul1, shift1, gate1, premul2, shift2,
    /// gate2`, `premul` being `1 + scale` folded (the kernels multiply by one
    /// vector and add another and have no place for the `1 +`).
    ///
    /// CONSTANT for the whole model, not per image: the conditioning is
    /// `t_embedder(0)` and the timestep never moves, because this is a one-step
    /// codec. So a device arm builds this once at session setup and the 21
    /// `adaLN_modulation` linears never reach the device at all.
    ///
    /// `blocks` is `self.blocks` or an encoder's; caller frees.
    pub fn dicoModTable(io: std.Io, gpa: std.mem.Allocator, blocks: []const DiCo, t_embed: TimeMlp) ![]f32 {
        const c = try timeVector(io, gpa, t_embed, 0.0);
        defer gpa.free(c);
        const act = try gpa.dupe(f32, c);
        defer gpa.free(act);
        ops.act.silu(act);

        const ch = blocks[0].ch;
        const out = try gpa.alloc(f32, blocks.len * 6 * ch);
        errdefer gpa.free(out);
        const scratch = try gpa.alloc(f32, 6 * ch);
        defer gpa.free(scratch);
        for (blocks, 0..) |*b, i| {
            const ad = b.adaln orelse return error.ShapeMismatch;
            try ops.matmul.matmul(io, gpa, scratch, act, 1, ad.w, ad.b);
            const dst = out[i * 6 * ch ..];
            // The checkpoint emits shift, scale, gate per half; the kernels want
            // premul, shift, gate.
            for (0..2) |half_i| {
                const src = scratch[half_i * 3 * ch ..];
                const d = dst[half_i * 3 * ch ..];
                for (0..ch) |j| d[j] = 1.0 + src[ch + j];
                @memcpy(d[ch..][0..ch], src[0..ch]);
                @memcpy(d[2 * ch ..][0..ch], src[2 * ch ..][0..ch]);
            }
        }
        return out;
    }

    /// RGB `[h][w][3]` in [-1, 1] -> latent `[lat_h][lat_w][latent_channels]`
    /// (the posterior mean). `h` and `w` must be multiples of 16. Caller frees.
    pub fn encode(
        self: *const MageVae,
        io: std.Io,
        gpa: std.mem.Allocator,
        img: []const f32,
        h: usize,
        w: usize,
    ) ![]f32 {
        const e = self.enc orelse return error.ComponentNotInCheckpoint;
        if (h % patch != 0 or w % patch != 0) return error.ShapeMismatch;
        std.debug.assert(img.len == h * w * 3);
        const lat_h = h / patch;
        const lat_w = w / patch;
        const n = lat_h * lat_w;

        const head = try gpa.alloc(f32, n * enc_head);
        defer gpa.free(head);
        try ops.conv.conv2d(io, gpa, head, img, h, w, e.patch_cond);
        for (e.head_blocks) |*b| try dicoForward(io, gpa, b, head, lat_h, lat_w, null);

        const cond = try gpa.alloc(f32, n * hidden);
        defer gpa.free(cond);
        try ops.conv.conv2d(io, gpa, cond, head, lat_h, lat_w, e.proj_down);

        // `z_proj(zeros)` is its bias, broadcast over the grid.
        const s = try gpa.alloc(f32, n * hidden);
        defer gpa.free(s);
        {
            const cat = try gpa.alloc(f32, n * 2 * hidden);
            defer gpa.free(cat);
            const zb = e.z_proj.b.?;
            for (0..n) |i| {
                const row = cat[i * 2 * hidden ..];
                @memcpy(row[0..hidden], cond[i * hidden ..][0..hidden]);
                @memcpy(row[hidden..][0..hidden], zb);
            }
            try ops.conv.conv2d(io, gpa, s, cat, lat_h, lat_w, e.fuse);
        }

        const c = try timeVector(io, gpa, e.t_embed, 0.0);
        defer gpa.free(c);
        for (e.blocks) |*b| try dicoForward(io, gpa, b, s, lat_h, lat_w, c);

        ops.norm.layerNorm(s, s, e.norm_out.w, e.norm_out.b, eps);
        const both = try gpa.alloc(f32, n * 2 * latent_channels);
        defer gpa.free(both);
        try ops.conv.conv2d(io, gpa, both, s, lat_h, lat_w, e.proj_out);

        // `[mean | logvar]`, and `sample_posterior=False` keeps only the mean.
        const out = try gpa.alloc(f32, n * latent_channels);
        errdefer gpa.free(out);
        for (0..n) |i| {
            @memcpy(out[i * latent_channels ..][0..latent_channels], both[i * 2 * latent_channels ..][0..latent_channels]);
        }
        return out;
    }

    /// The CoD decoder: latent -> `[n][hidden]` conditioning. Caller frees.
    pub fn codForward(
        self: *const MageVae,
        io: std.Io,
        gpa: std.mem.Allocator,
        z: []const f32,
        lat_h: usize,
        lat_w: usize,
    ) ![]f32 {
        const n = lat_h * lat_w;
        const x = try gpa.alloc(f32, n * hidden);
        errdefer gpa.free(x);
        try ops.conv.conv2d(io, gpa, x, z, lat_h, lat_w, self.cod.conv_in);

        try resnetForward(io, gpa, self.cod.r0, x, lat_h, lat_w);
        try winAttnForward(io, gpa, self.cod.a0, x, lat_h, lat_w);
        try resnetForward(io, gpa, self.cod.r1, x, lat_h, lat_w);
        try winAttnForward(io, gpa, self.cod.a1, x, lat_h, lat_w);
        try resnetForward(io, gpa, self.cod.r2, x, lat_h, lat_w);

        ops.norm.groupNorm(x, x, hidden, norm_groups, self.cod.norm_out.w, self.cod.norm_out.b, eps);
        ops.act.silu(x);
        const out = try gpa.alloc(f32, n * hidden);
        errdefer gpa.free(out);
        try ops.conv.conv2d(io, gpa, out, x, lat_h, lat_w, self.cod.conv_out);
        gpa.free(x);
        return out;
    }

    /// One band of latent cells through the per-pixel pathway, writing its
    /// 16x16 patch of `rgb` for each.
    fn nerfBand(
        self: *const MageVae,
        io: std.Io,
        gpa: std.mem.Allocator,
        rgb: []f32,
        cond: []const f32,
        s: []const f32,
        dct: []const f32,
        l0: usize,
        nb: usize,
        lat_w: usize,
    ) !void {
        const rows = nb * patch_area;

        // Per-pixel channels of each patch, `[nb][hidden_x * patch_area]`.
        const y = try gpa.alloc(f32, nb * hidden_x * patch_area);
        defer gpa.free(y);
        try ops.matmul.matmul(io, gpa, y, cond, nb, self.y_embedder_x.w, self.y_embedder_x.b);

        // NeRF input: the (zero) image pixel, the patch channels for that
        // position, then the position's cosine features.
        const feat = try gpa.alloc(f32, rows * nerf_in);
        defer gpa.free(feat);
        for (0..nb) |i| {
            const yrow = y[i * hidden_x * patch_area ..];
            for (0..patch_area) |k| {
                const f = feat[(i * patch_area + k) * nerf_in ..][0..nerf_in];
                @memset(f[0..3], 0); // decode's noise input is zero
                for (0..hidden_x) |j| f[3 + j] = yrow[j * patch_area + k];
                @memcpy(f[3 + hidden_x ..][0..dct_dim], dct[k * dct_dim ..][0..dct_dim]);
            }
        }

        var x = try gpa.alloc(f32, rows * hidden_x);
        defer gpa.free(x);
        try ops.matmul.matmul(io, gpa, x, feat, rows, self.x_embedder.w, self.x_embedder.b);
        {
            const tmp = try gpa.alloc(f32, rows * hidden_x);
            defer gpa.free(tmp);
            try ops.matmul.matmul(io, gpa, tmp, x, rows, self.input_proj.w, self.input_proj.b);
            @memcpy(x, tmp);
        }

        // The per-pixel conditioning: one `hidden_x` vector per patch position.
        const cvec = try gpa.alloc(f32, rows * hidden_x);
        defer gpa.free(cvec);
        try ops.matmul.matmul(io, gpa, cvec, s, nb, self.cond_embed.w, self.cond_embed.b);

        const mod = try gpa.alloc(f32, rows * 3 * hidden_x);
        defer gpa.free(mod);
        const act = try gpa.alloc(f32, rows * hidden_x);
        defer gpa.free(act);
        for (self.res_blocks) |*r| {
            @memcpy(act, cvec);
            ops.act.silu(act);
            try ops.matmul.matmul(io, gpa, mod, act, rows, r.adaln.w, r.adaln.b);

            ops.norm.layerNorm(act, x, r.in_ln.w, r.in_ln.b, eps);
            for (0..rows) |i| {
                const m = mod[i * 3 * hidden_x ..];
                const a = act[i * hidden_x ..][0..hidden_x];
                for (a, m[0..hidden_x], m[hidden_x..][0..hidden_x]) |*v, shift, scale| {
                    v.* = v.* * (1.0 + scale) + shift;
                }
            }
            const mid = try gpa.alloc(f32, rows * hidden_x);
            defer gpa.free(mid);
            try ops.matmul.matmul(io, gpa, mid, act, rows, r.fc1.w, r.fc1.b);
            ops.act.silu(mid);
            try ops.matmul.matmul(io, gpa, act, mid, rows, r.fc2.w, r.fc2.b);
            for (0..rows) |i| {
                const gate = mod[i * 3 * hidden_x + 2 * hidden_x ..][0..hidden_x];
                const xr = x[i * hidden_x ..][0..hidden_x];
                for (xr, act[i * hidden_x ..][0..hidden_x], gate) |*v, d, g| v.* += g * d;
            }
        }

        ops.norm.rmsNorm(x, x, self.final_norm, eps);
        const pix = try gpa.alloc(f32, rows * 3);
        defer gpa.free(pix);
        try ops.matmul.matmul(io, gpa, pix, x, rows, self.final_lin.w, self.final_lin.b);

        // Scatter each patch back: patch position k is (k / 16, k % 16).
        const img_w = lat_w * patch;
        for (0..nb) |i| {
            const cell = l0 + i;
            const y0 = (cell / lat_w) * patch;
            const x0 = (cell % lat_w) * patch;
            for (0..patch_area) |k| {
                const py = y0 + k / patch;
                const px = x0 + k % patch;
                @memcpy(rgb[(py * img_w + px) * 3 ..][0..3], pix[(i * patch_area + k) * 3 ..][0..3]);
            }
        }
    }
};

/// `t_embedder`: the DConv sinusoidal embedding (cos first, no 1000x scale)
/// through a two-layer MLP. Caller frees.
pub fn timeVector(io: std.Io, gpa: std.mem.Allocator, mlp: TimeMlp, t: f32) ![]f32 {
    var emb: [t_freq]f32 = undefined;
    ops.rope.sinCosEmbedding(&emb, t, 10000.0);
    const mid = try gpa.alloc(f32, hidden);
    defer gpa.free(mid);
    try ops.matmul.matmul(io, gpa, mid, &emb, 1, mlp.fc1.w, mlp.fc1.b);
    ops.act.silu(mid);
    const out = try gpa.alloc(f32, hidden);
    errdefer gpa.free(out);
    try ops.matmul.matmul(io, gpa, out, mid, 1, mlp.fc2.w, mlp.fc2.b);
    return out;
}

/// `NerfEmbedder.fetch_pos`: `[patch_area][dct_dim]`, a separable cosine basis
/// over the patch's normalized coordinates, damped by `1 / (1 + fx*fy)`.
/// Caller frees.
pub fn dctTable(gpa: std.mem.Allocator) ![]f32 {
    const out = try gpa.alloc(f32, patch_area * dct_dim);
    var freq: [max_freqs]f32 = undefined;
    // `torch.linspace(0, max_freqs, max_freqs)`: 8 points from 0 to 8 inclusive,
    // so the step is 8/7 and the top frequency IS `max_freqs`, not `max_freqs-1`.
    for (&freq, 0..) |*f, i| {
        f.* = @as(f32, @floatFromInt(max_freqs)) * @as(f32, @floatFromInt(i)) / @as(f32, max_freqs - 1);
    }
    for (0..patch) |ky| {
        for (0..patch) |kx| {
            // `meshgrid(..., indexing="ij")`: the first axis is y.
            const py = @as(f32, @floatFromInt(ky)) / @as(f32, patch - 1);
            const px = @as(f32, @floatFromInt(kx)) / @as(f32, patch - 1);
            const row = out[(ky * patch + kx) * dct_dim ..][0..dct_dim];
            for (0..max_freqs) |a| {
                const cx = @cos(px * freq[a] * std.math.pi);
                for (0..max_freqs) |b| {
                    const cy = @cos(py * freq[b] * std.math.pi);
                    row[a * max_freqs + b] = cx * cy / (1.0 + freq[a] * freq[b]);
                }
            }
        }
    }
    return out;
}

/// In place: `x += conv2(silu(norm2(conv1(silu(norm1(x))))))`, width-preserving.
fn resnetForward(io: std.Io, gpa: std.mem.Allocator, r: Resnet, x: []f32, h: usize, w: usize) !void {
    const n = h * w;
    const t = try gpa.alloc(f32, n * hidden);
    defer gpa.free(t);
    ops.norm.groupNorm(t, x, hidden, norm_groups, r.norm1.w, r.norm1.b, eps);
    ops.act.silu(t);
    const mid = try gpa.alloc(f32, n * hidden);
    defer gpa.free(mid);
    try ops.conv.conv2d(io, gpa, mid, t, h, w, r.conv1);
    ops.norm.groupNorm(t, mid, hidden, norm_groups, r.norm2.w, r.norm2.b, eps);
    ops.act.silu(t);
    try ops.conv.conv2d(io, gpa, mid, t, h, w, r.conv2);
    for (x, mid) |*v, d| v.* += d;
}

/// In place: `x += proj(windowed_attention(norm(x)))`.
///
/// The windows are 32x32 cells of the LATENT grid, replicate-padded on the right
/// and bottom to fill the last one, and each attends to itself alone. Both halves
/// of that change the answer and are pinned separately: a latent smaller than 32
/// is padded up, so its edge cells attend to duplicates of themselves, and a
/// latent larger than 32 splits, so cells in different windows never meet.
fn winAttnForward(io: std.Io, gpa: std.mem.Allocator, a: WinAttn, x: []f32, h: usize, w: usize) !void {
    const n = h * w;
    const d = attn_window;
    const tokens = d * d;

    const norm = try gpa.alloc(f32, n * hidden);
    defer gpa.free(norm);
    ops.norm.groupNorm(norm, x, hidden, norm_groups, a.norm.w, a.norm.b, eps);

    const qkv = try gpa.alloc(f32, 3 * n * hidden);
    defer gpa.free(qkv);
    const q = qkv[0 .. n * hidden];
    const k = qkv[n * hidden ..][0 .. n * hidden];
    const v = qkv[2 * n * hidden ..];
    try ops.conv.conv2d(io, gpa, q, norm, h, w, a.q);
    try ops.conv.conv2d(io, gpa, k, norm, h, w, a.k);
    try ops.conv.conv2d(io, gpa, v, norm, h, w, a.v);

    const nph = (h + d - 1) / d;
    const npw = (w + d - 1) / d;
    const win = try gpa.alloc(f32, 3 * tokens * hidden);
    defer gpa.free(win);
    const attn = try gpa.alloc(f32, tokens * hidden);
    defer gpa.free(attn);
    const gathered = try gpa.alloc(f32, n * hidden);
    defer gpa.free(gathered);

    for (0..nph) |wy| {
        for (0..npw) |wx| {
            for (0..3) |which| {
                const src = switch (which) {
                    0 => q,
                    1 => k,
                    else => v,
                };
                const dst = win[which * tokens * hidden ..][0 .. tokens * hidden];
                for (0..d) |ty| {
                    // Replicate padding: the edge row/column repeats.
                    const sy = @min(wy * d + ty, h - 1);
                    for (0..d) |tx| {
                        const sx = @min(wx * d + tx, w - 1);
                        @memcpy(dst[(ty * d + tx) * hidden ..][0..hidden], src[(sy * w + sx) * hidden ..][0..hidden]);
                    }
                }
            }
            // One head over all `hidden` channels, scale 1/sqrt(hidden).
            try ops.attention.attention(io, gpa, attn, win[0 .. tokens * hidden], win[tokens * hidden ..][0 .. tokens * hidden], win[2 * tokens * hidden ..], .{
                .seq_q = tokens,
                .seq_kv = tokens,
                .n_heads = 1,
                .n_kv_heads = 1,
                .head_dim = hidden,
            });
            for (0..d) |ty| {
                const sy = wy * d + ty;
                if (sy >= h) break;
                for (0..d) |tx| {
                    const sx = wx * d + tx;
                    if (sx >= w) break;
                    @memcpy(gathered[(sy * w + sx) * hidden ..][0..hidden], attn[(ty * d + tx) * hidden ..][0..hidden]);
                }
            }
        }
    }

    const projected = try gpa.alloc(f32, n * hidden);
    defer gpa.free(projected);
    try ops.conv.conv2d(io, gpa, projected, gathered, h, w, a.proj);
    for (x, projected) |*o, p| o.* += p;
}

/// One `DiCoBlock` in place. `c` is the conditioning vector, or null for the
/// encoder's head blocks, which have affine norms and no modulation instead.
fn dicoForward(io: std.Io, gpa: std.mem.Allocator, b: *const DiCo, x: []f32, h: usize, w: usize, c: ?[]const f32) !void {
    const n = h * w;
    const ch = b.ch;
    const inner = 4 * ch;
    std.debug.assert(x.len == n * ch);

    var mod: []f32 = &.{};
    defer if (mod.len != 0) gpa.free(mod);
    if (b.adaln) |ad| {
        const act = try gpa.alloc(f32, ch);
        defer gpa.free(act);
        @memcpy(act, c.?);
        ops.act.silu(act);
        mod = try gpa.alloc(f32, 6 * ch);
        try ops.matmul.matmul(io, gpa, mod, act, 1, ad.w, ad.b);
    }

    const t = try gpa.alloc(f32, n * ch);
    defer gpa.free(t);
    const u = try gpa.alloc(f32, n * ch);
    defer gpa.free(u);

    // Attention-position half: norm, modulate, 1x1, depthwise 3x3, gelu,
    // channel gate, 1x1, gated residual.
    normPart(b.norm1, t, x, ch);
    if (mod.len != 0) modulate(t, mod[ch..][0..ch], mod[0..ch]);
    try ops.conv.conv2d(io, gpa, u, t, h, w, b.conv1);
    ops.conv.depthwiseConv2d(t, u, h, w, ch, 3, 1, b.dw, b.dw_b);
    ops.act.geluErf(t);
    try channelGate(io, gpa, b.ca, t, n, ch);
    try ops.conv.conv2d(io, gpa, u, t, h, w, b.conv3);
    if (mod.len != 0) gatedAdd(x, u, mod[2 * ch ..][0..ch]) else addInto(x, u);

    // MLP half.
    normPart(b.norm2, t, x, ch);
    if (mod.len != 0) modulate(t, mod[4 * ch ..][0..ch], mod[3 * ch ..][0..ch]);
    const wide = try gpa.alloc(f32, n * inner);
    defer gpa.free(wide);
    try ops.conv.conv2d(io, gpa, wide, t, h, w, b.conv4);
    ops.act.geluErf(wide);
    try ops.conv.conv2d(io, gpa, u, wide, h, w, b.conv5);
    if (mod.len != 0) gatedAdd(x, u, mod[5 * ch ..][0..ch]) else addInto(x, u);
}

/// `LayerNorm2d` over a channel-last activation: the channel axis is already the
/// row, so an affine norm is a plain row-wise LayerNorm and an affine-free one is
/// `layerNormUnit`.
fn normPart(nw: ?NormW, out: []f32, x: []const f32, ch: usize) void {
    if (nw) |n| ops.norm.layerNorm(out, x, n.w, n.b, eps) else ops.norm.layerNormUnit(out, x, ch, eps);
}

/// `x *= sigmoid(W mean_hw(x) + b)`, broadcast over positions.
fn channelGate(io: std.Io, gpa: std.mem.Allocator, ca: LinearW, x: []f32, n: usize, ch: usize) !void {
    const mean = try gpa.alloc(f32, ch);
    defer gpa.free(mean);
    @memset(mean, 0);
    var row: usize = 0;
    while (row < x.len) : (row += ch) {
        for (mean, x[row..][0..ch]) |*m, v| m.* += v;
    }
    const inv = 1.0 / @as(f32, @floatFromInt(n));
    for (mean) |*m| m.* *= inv;

    const gate = try gpa.alloc(f32, ch);
    defer gpa.free(gate);
    try ops.matmul.matmul(io, gpa, gate, mean, 1, ca.w, ca.b);
    ops.act.sigmoid(gate);

    row = 0;
    while (row < x.len) : (row += ch) {
        for (x[row..][0..ch], gate) |*v, g| v.* *= g;
    }
}

fn modulate(x: []f32, scale: []const f32, shift: []const f32) void {
    const dim = scale.len;
    var row: usize = 0;
    while (row < x.len) : (row += dim) {
        for (x[row..][0..dim], scale, shift) |*v, sc, sh| v.* = (1.0 + sc) * v.* + sh;
    }
}

fn gatedAdd(x: []f32, delta: []const f32, gate: []const f32) void {
    const dim = gate.len;
    var row: usize = 0;
    while (row < x.len) : (row += dim) {
        for (x[row..][0..dim], delta[row..][0..dim], gate) |*v, d, g| v.* += g * d;
    }
}

fn addInto(x: []f32, delta: []const f32) void {
    for (x, delta) |*v, d| v.* += d;
}

// --- weight loading --------------------------------------------------------

const Loader = struct {
    store: WeightStore,
    alloc: std.mem.Allocator,
    pfx: []const u8,

    fn has(l: Loader, rel: []const u8) bool {
        var buf: [200]u8 = undefined;
        const nm = std.fmt.bufPrint(&buf, "{s}{s}", .{ l.pfx, rel }) catch return false;
        return l.store.get(nm) != null;
    }

    fn name(l: Loader, buf: []u8, comptime fmt: []const u8, args: anytype, suffix: []const u8) ![]u8 {
        var fbs = std.Io.Writer.fixed(buf);
        try fbs.writeAll(l.pfx);
        try fbs.print(fmt, args);
        try fbs.writeAll(suffix);
        return fbs.buffered();
    }

    fn view(l: Loader, nm: []const u8) !safetensors.TensorView {
        return l.store.get(nm) orelse {
            std.log.err("mage_vae: missing {s}", .{nm});
            return error.MissingTensor;
        };
    }

    fn vec(l: Loader, comptime fmt: []const u8, args: anytype, suffix: []const u8, len: usize) ![]f32 {
        var buf: [200]u8 = undefined;
        const nm = try l.name(&buf, fmt, args, suffix);
        const v = try l.view(nm);
        if (v.info.elemCount() != len) {
            std.log.err("mage_vae: {s} has {d} elements, expected {d}", .{ nm, v.info.elemCount(), len });
            return error.ShapeMismatch;
        }
        return v.toF32Alloc(l.alloc);
    }

    fn norm(l: Loader, comptime fmt: []const u8, args: anytype, ch: usize) !NormW {
        return .{ .w = try l.vec(fmt, args, ".weight", ch), .b = try l.vec(fmt, args, ".bias", ch) };
    }

    /// A conv weight as f32 in im2col patch order. `[co, ci]` is accepted for a
    /// 1x1 because an `nn.Linear` of the same widths is byte-identical.
    fn convW(l: Loader, comptime fmt: []const u8, args: anytype, co: usize, ci: usize, k: usize) !struct { w: []f32, tag: []u8 } {
        var buf: [200]u8 = undefined;
        const nm = try l.name(&buf, fmt, args, ".weight");
        const v = try l.view(nm);
        const shape = v.info.shape.slice();
        const ok = (shape.len == 4 and shape[0] == co and shape[1] == ci and shape[2] == k and shape[3] == k) or
            (k == 1 and shape.len == 2 and shape[0] == co and shape[1] == ci);
        if (!ok) {
            std.log.err("mage_vae: {s} has shape {any}, expected [{d}, {d}, {d}, {d}]", .{ nm, shape, co, ci, k, k });
            return error.ShapeMismatch;
        }
        const torch_w = try v.toF32Alloc(l.alloc);
        defer l.alloc.free(torch_w);
        return .{ .w = try ops.conv.packWeight(l.alloc, torch_w, co, ci, k), .tag = try l.alloc.dupe(u8, nm) };
    }

    fn conv(l: Loader, comptime fmt: []const u8, args: anytype, co: usize, ci: usize, k: usize, stride: usize, pad: usize) !Conv2d {
        const packed_w = try l.convW(fmt, args, co, ci, k);
        return .{
            .w = packed_w.w,
            .tag = packed_w.tag,
            .b = try l.vec(fmt, args, ".bias", co),
            .co = co,
            .ci = ci,
            .k = k,
            .stride = stride,
            .pad = pad,
        };
    }

    fn convNoBias(l: Loader, comptime fmt: []const u8, args: anytype, co: usize, ci: usize, k: usize, stride: usize, pad: usize) !Conv2d {
        const packed_w = try l.convW(fmt, args, co, ci, k);
        return .{ .w = packed_w.w, .tag = packed_w.tag, .b = null, .co = co, .ci = ci, .k = k, .stride = stride, .pad = pad };
    }

    /// A `[rows, cols]` linear, MATERIALIZED to f32 like every other weight here.
    ///
    /// Not the checkpoint's dtype, and the reason is the device arm rather than
    /// taste: these feed `Backend.opMatmul`, whose only pipeline is f32, so bf16
    /// bytes read as f32 are both wrong and HALF the buffer -- an out-of-bounds
    /// read that faults the context. The whole codec is 345 MB, so the f32 copy
    /// is affordable where a DiT's would not be.
    fn linear(l: Loader, comptime fmt: []const u8, args: anytype, rows: usize, cols: usize) !LinearW {
        var buf: [200]u8 = undefined;
        const nm = try l.name(&buf, fmt, args, ".weight");
        const v = try l.view(nm);
        if (v.info.elemCount() != rows * cols) {
            std.log.err("mage_vae: {s} has {d} elements, expected {d}", .{ nm, v.info.elemCount(), rows * cols });
            return error.ShapeMismatch;
        }
        var w = try ops.matmul.materializeF32(l.alloc, Weight.init(v.bytes, v.info.dtype, rows, cols));
        w.tag = try l.alloc.dupe(u8, nm);
        return .{ .w = w, .b = try l.vec(fmt, args, ".bias", rows) };
    }

    fn timeMlp(l: Loader, comptime prefix: []const u8, dim: usize) !TimeMlp {
        return .{
            .fc1 = try l.linear(prefix ++ ".mlp.0", .{}, dim, t_freq),
            .fc2 = try l.linear(prefix ++ ".mlp.2", .{}, dim, dim),
        };
    }

    fn dico(l: Loader, comptime fmt: []const u8, args: anytype, ch: usize, modulated: bool) !DiCo {
        var buf: [200]u8 = undefined;
        const base = try std.fmt.bufPrint(&buf, fmt, args);
        const inner = 4 * ch;
        return .{
            .adaln = if (modulated) try l.linear("{s}.adaLN_modulation.1", .{base}, 6 * ch, ch) else null,
            .norm1 = if (modulated) null else try l.norm("{s}.norm1", .{base}, ch),
            .norm2 = if (modulated) null else try l.norm("{s}.norm2", .{base}, ch),
            .conv1 = try l.conv("{s}.conv1", .{base}, ch, ch, 1, 1, 0),
            .dw = try l.vec("{s}.conv2", .{base}, ".weight", ch * 9),
            .dw_b = try l.vec("{s}.conv2", .{base}, ".bias", ch),
            .conv3 = try l.conv("{s}.conv3", .{base}, ch, ch, 1, 1, 0),
            .ca = try l.linear("{s}.ca.1", .{base}, ch, ch),
            .conv4 = try l.conv("{s}.conv4", .{base}, inner, ch, 1, 1, 0),
            .conv5 = try l.conv("{s}.conv5", .{base}, ch, inner, 1, 1, 0),
            .ch = ch,
        };
    }

    fn resnet(l: Loader, comptime fmt: []const u8, args: anytype) !Resnet {
        var buf: [200]u8 = undefined;
        const base = try std.fmt.bufPrint(&buf, fmt, args);
        return .{
            .norm1 = try l.norm("{s}.norm1", .{base}, hidden),
            .conv1 = try l.conv("{s}.conv1", .{base}, hidden, hidden, 3, 1, 1),
            .norm2 = try l.norm("{s}.norm2", .{base}, hidden),
            .conv2 = try l.conv("{s}.conv2", .{base}, hidden, hidden, 3, 1, 1),
        };
    }

    fn winAttn(l: Loader, comptime fmt: []const u8, args: anytype) !WinAttn {
        var buf: [200]u8 = undefined;
        const base = try std.fmt.bufPrint(&buf, fmt, args);
        return .{
            .norm = try l.norm("{s}.norm", .{base}, hidden),
            .q = try l.conv("{s}.q", .{base}, hidden, hidden, 1, 1, 0),
            .k = try l.conv("{s}.k", .{base}, hidden, hidden, 1, 1, 0),
            .v = try l.conv("{s}.v", .{base}, hidden, hidden, 1, 1, 0),
            .proj = try l.conv("{s}.proj_out", .{base}, hidden, hidden, 1, 1, 0),
        };
    }

    fn loadCod(l: Loader, comptime prefix: []const u8) !Cod {
        return .{
            .conv_in = try l.conv(prefix ++ ".conv_in", .{}, hidden, latent_channels, 3, 1, 1),
            .r0 = try l.resnet(prefix ++ ".block.0", .{}),
            .a0 = try l.winAttn(prefix ++ ".block.1", .{}),
            .r1 = try l.resnet(prefix ++ ".block.2", .{}),
            .a1 = try l.winAttn(prefix ++ ".block.3", .{}),
            .r2 = try l.resnet(prefix ++ ".block.4", .{}),
            .norm_out = try l.norm(prefix ++ ".norm_out", .{}, hidden),
            .conv_out = try l.conv(prefix ++ ".conv_out", .{}, hidden, hidden, 3, 1, 1),
        };
    }
};

// --- tests -----------------------------------------------------------------

const testing = std.testing;

test "the DCT table is separable, damped, and starts at one" {
    const gpa = testing.allocator;
    const dct = try dctTable(gpa);
    defer gpa.free(dct);

    // Frequency pair (0, 0) is a constant 1 everywhere: cos(0)*cos(0)/(1+0).
    for (0..patch_area) |k| try testing.expectApproxEqAbs(@as(f32, 1.0), dct[k * dct_dim], 1e-6);

    // The top frequency is `max_freqs` itself, so at the patch's far corner
    // (px = 1) the a = 7 column is cos(8*pi) = 1, not cos(7*pi) = -1. Getting
    // linspace's endpoint wrong flips the sign of a quarter of the table.
    const corner = (0 * patch + patch - 1) * dct_dim;
    try testing.expectApproxEqAbs(@as(f32, 1.0), dct[corner + 7 * max_freqs], 1e-5);

    // Separability: row (ky, kx) at (a, b) is the x part at (kx, a) times the y
    // part at (ky, b), so the table is determined by its first row and column.
    const at = (3 * patch + 5) * dct_dim + 2 * max_freqs + 4;
    const x_only = dct[(0 * patch + 5) * dct_dim + 2 * max_freqs + 0]; // cos(px*f2*pi)/(1+0)
    const y_only = dct[(3 * patch + 0) * dct_dim + 0 * max_freqs + 4]; // cos(py*f4*pi)/(1+0)
    const damp = 1.0 + (@as(f32, 8.0) * 2.0 / 7.0) * (@as(f32, 8.0) * 4.0 / 7.0);
    try testing.expectApproxEqAbs(x_only * y_only / damp, dct[at], 1e-5);
}

test "modulate and gatedAdd broadcast over rows" {
    var x = [_]f32{ 1, 2, 3, 4 };
    modulate(&x, &.{ 0.5, -1.0 }, &.{ 10, 20 });
    try testing.expectEqualSlices(f32, &.{ 11.5, 20, 14.5, 20 }, &x);
    gatedAdd(&x, &.{ 1, 1, 2, 2 }, &.{ 2, 0.5 });
    try testing.expectEqualSlices(f32, &.{ 13.5, 20.5, 18.5, 21 }, &x);
}

const test_gate = @import("../test_gate.zig");
const SafeTensors = safetensors.SafeTensors;

const ref_path = "src/models/assets/mage_vae_ref.safetensors";
const vae_ckpt = "/home/qt/genai/comfyui/models/vae/mage_flow_vae_bf16.safetensors";

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

/// Torch stores an image as planar `[c][h][w]`; this file works channel-last.
fn toChannelLast(gpa: std.mem.Allocator, planar: []const f32, c: usize, n: usize) ![]f32 {
    const out = try gpa.alloc(f32, planar.len);
    for (0..c) |ci| {
        for (0..n) |i| out[i * c + ci] = planar[ci * n + i];
    }
    return out;
}

test "Mage-VAE matches ComfyUI on the real codec" {
    const gpa = testing.allocator;
    const io = testing.io;
    try test_gate.requireIntegration();
    try test_gate.requireModelFile(io, vae_ckpt);
    try test_gate.requireModelFile(io, ref_path);

    var ref = try SafeTensors.open(gpa, io, ref_path);
    defer ref.deinit();
    var ck = try SafeTensors.open(gpa, io, vae_ckpt);
    defer ck.deinit();

    var model = try MageVae.load(gpa, .{ .safetensors = &ck }, "");
    defer model.deinit();
    try testing.expect(model.enc != null);

    var kb: [64]u8 = undefined;
    const key = struct {
        fn f(buf: []u8, c: []const u8, suffix: []const u8) []const u8 {
            return std.fmt.bufPrint(buf, "{s}.{s}", .{ c, suffix }) catch unreachable;
        }
    }.f;

    // `wide` is the case that separates windowed from global attention: `small`
    // is a single 32x32 window and passes either way.
    for ([_][]const u8{ "small", "wide" }) |case| {
        const zv = try ref.require(key(&kb, case, "z"));
        const zs = zv.info.shape.slice();
        const lat_h = zs[1];
        const lat_w = zs[2];
        const n = lat_h * lat_w;

        const z_planar = try zv.toF32Alloc(gpa);
        defer gpa.free(z_planar);
        const z = try toChannelLast(gpa, z_planar, latent_channels, n);
        defer gpa.free(z);

        // The CoD conditioning first: a windowing or padding mistake shows here,
        // where the final RGB would only say "wrong".
        {
            const cod_planar = try (try ref.require(key(&kb, case, "cod"))).toF32Alloc(gpa);
            defer gpa.free(cod_planar);
            const want = try toChannelLast(gpa, cod_planar, hidden, n);
            defer gpa.free(want);
            const got = try model.codForward(io, gpa, z, lat_h, lat_w);
            defer gpa.free(got);
            const rel = relL2(want, got);
            errdefer std.debug.print("case {s} ({d}x{d}): cod rel L2 {e:.4}\n", .{ case, lat_h, lat_w, rel });
            try testing.expect(rel < 2e-5);
        }

        const rgb_planar = try (try ref.require(key(&kb, case, "rgb"))).toF32Alloc(gpa);
        defer gpa.free(rgb_planar);
        const pixels = rgb_planar.len / 3;
        const want = try toChannelLast(gpa, rgb_planar, 3, pixels);
        defer gpa.free(want);

        const got = try model.decode(io, gpa, z, lat_h, lat_w);
        defer gpa.free(got);
        try testing.expectEqual(want.len, got.len);
        const rel = relL2(want, got);
        errdefer std.debug.print("case {s} ({d}x{d}): decode rel L2 {e:.4}\n", .{ case, lat_h, lat_w, rel });
        try testing.expect(rel < 2e-5);
    }

    {
        const iv = try ref.require("enc.img");
        const is = iv.info.shape.slice();
        const h = is[1];
        const w = is[2];
        const img_planar = try iv.toF32Alloc(gpa);
        defer gpa.free(img_planar);
        const img = try toChannelLast(gpa, img_planar, 3, h * w);
        defer gpa.free(img);

        const lat_planar = try (try ref.require("enc.latent")).toF32Alloc(gpa);
        defer gpa.free(lat_planar);
        const cells = lat_planar.len / latent_channels;
        const want = try toChannelLast(gpa, lat_planar, latent_channels, cells);
        defer gpa.free(want);

        const got = try model.encode(io, gpa, img, h, w);
        defer gpa.free(got);
        try testing.expectEqual(want.len, got.len);
        const rel = relL2(want, got);
        errdefer std.debug.print("encode ({d}x{d}): rel L2 {e:.4}\n", .{ h, w, rel });
        try testing.expect(rel < 2e-5);
    }
}
