//! The SD-family VAE decoder (`AutoencoderKL`, LDM `first_stage_model`), latent to
//! RGB, mirroring `diffusers.AutoencoderKL.decode`.
//!
//! Structurally simpler than the UNet: no conditioning, no timestep, no skip
//! connections. A 1x1 `post_quant_conv`, a 3x3 stem into 512 channels, a middle block
//! (resnet, self-attention, resnet), then four levels of three resnets each doubling
//! the resolution on the way out, and a 3x3 head to 3 channels.
//!
//! Three things differ from the UNet and are easy to carry over wrongly:
//!
//! 1. GroupNorm epsilon is 1e-6 here, 1e-5 in the UNet. LDM's `Normalize` uses
//!    1e-6 for the autoencoder. The difference is invisible on well-conditioned
//!    activations and shows up as faint banding in flat regions.
//! 2. `decoder.up.N` is indexed from the OUTERMOST level. The forward runs
//!    `up.3, up.2, up.1, up.0`, 512, 512, 256, 128 channels, and `up.0` has no
//!    upsample. Reading the list forwards produces a decoder that runs but inverts
//!    the channel ramp.
//! 3. The middle attention is single-head over pixels, with 1x1 convolutions for
//!    q/k/v (not linear layers over a flattened sequence, though it is the same
//!    arithmetic). It carries biases, unlike the UNet's cross-attention q/k/v.
//!
//! The caller owns the latent scaling: `decode` expects `z` already divided by the
//! checkpoint's `scaling_factor` (0.18215 for SD1.5), exactly as
//! `AutoencoderKL.decode` does, so that a parity test can compare the decoder alone.

const std = @import("std");
const tp_core = @import("tp_core");
const safetensors = tp_core.safetensors;
const weights_mod = tp_core.weights;
const ops = @import("tp_ops");

const WeightStore = weights_mod.WeightStore;
const Weight = ops.matmul.Weight;
const Conv2d = ops.conv.Conv2d;

/// Which spelling a checkpoint uses for the *same* `AutoencoderKL` graph.
///
/// These are two naming schemes over identical arithmetic, and the up-block
/// index runs in opposite directions: LDM numbers levels from the outermost
/// (execution runs `up.3 ... up.0`) while diffusers stores them already reversed
/// (execution runs `up_blocks.0 ... up_blocks.3`). Reading one as the other loads a
/// decoder that runs and inverts the channel ramp.
///
/// Detected from the store, never configured (`Decoder.detectNaming`), because
/// it is a property of the FILE and not of the architecture: the very same 16-channel
/// Z-Image VAE ships both ways, LDM-named in the distribution the ComfyUI template
/// points at and diffusers-named in another. A config field here would be a flag that
/// can disagree with the bytes.
pub const Naming = enum {
    /// LDM / `first_stage_model` (SD1.5, SDXL): `decoder.mid.block_1`,
    /// `decoder.up.3.block.0`, `nin_shortcut`, `decoder.norm_out`, and a middle
    /// attention stored as four 1x1 convolutions (`q`/`k`/`v`/`proj_out`).
    ldm,
    /// diffusers `AutoencoderKL` (the Flux / Z-Image VAE exports):
    /// `decoder.mid_block.resnets.0`, `decoder.up_blocks.0.resnets.0`,
    /// `conv_shortcut`, `decoder.conv_norm_out`, and a middle attention stored as
    /// `nn.Linear` (`to_q`/`to_k`/`to_v`/`to_out.0`) with `group_norm` for its
    /// norm. A `[c, c]` Linear and a `[c, c, 1, 1]` convolution hold the same bytes
    /// in the same order, so only the shape check has to accommodate both.
    diffusers,
};

/// SD1.5, SDXL and the 16-channel Flux/Z-Image VAE share this decoder shape; only
/// the widths, the latent channel count, the naming and the latent scaling differ.
pub const Config = struct {
    /// Latent channels: 4 for SD, 16 for Flux/Z-Image.
    z_channels: usize,
    /// Base width; `block_out_channels` below are multiples of it.
    base_channels: usize,
    /// Per-level widths, outermost first (`{128, 256, 512, 512}` for SD).
    block_out_channels: []const usize,
    /// Resnets per level on the decode path (LDM: `num_res_blocks + 1` = 3).
    layers_per_block: usize,
    norm_groups: usize,
    norm_eps: f32,
    /// What `decode` assumes the caller already divided the latent by.
    scaling_factor: f32,
    /// What `decode` assumes the caller already ADDED, after the divide
    /// (`latent_formats.Flux.process_out` is `latent / scale + shift`). Zero for
    /// SD, whose latent format has no shift term.
    shift_factor: f32 = 0,
    /// Store the decoder's running activations as f16 rather than f32, halving
    /// the two widest buffers, at 1056x1584 those hold 256 channels at full image
    /// resolution (428M elements each), so this is most of a whole-image decode's
    /// VRAM. The arithmetic is unchanged: every GEMM already ran f16 operands with
    /// an f32 accumulator, and the norms still reduce in f32.
    ///
    /// RANGE decides this, not precision, and the answer differs per
    /// architecture. These buffers carry the residual stream, whose measured peak
    /// is ~489 for the Flux/Z-Image VAE and ~7e3 for SD1.5, both far inside f16's
    /// 65504, but 4.2e5 for SDXL's, which is why `sd_vae_cuda.residual_act_div`
    /// exists at all. An f16 buffer cannot hold a value the divisor was invented to
    /// sneak through a cast, so the two are alternatives and this stays off for
    /// SDXL. It is a per-config constant rather than a probe because a decode
    /// cannot discover its own peak before allocating.
    act_f16: bool = false,

    pub fn levels(self: Config) usize {
        return self.block_out_channels.len;
    }

    pub fn innermost(self: Config) usize {
        return self.block_out_channels[self.block_out_channels.len - 1];
    }
};

pub const sd15: Config = .{
    .z_channels = 4,
    .base_channels = 128,
    .block_out_channels = &.{ 128, 256, 512, 512 },
    .layers_per_block = 3,
    .norm_groups = 32,
    .norm_eps = 1e-6,
    .scaling_factor = 0.18215,
    .act_f16 = true,
};

/// SDXL's VAE is architecturally identical to SD1.5's, same widths, same depth, same
/// epsilon, and differs only in weights and in the latent scale it was trained against.
/// That one number is not cosmetic: decoding an SDXL latent with 0.18215 hands the decoder
/// values 1.4x too small and produces a washed, low-contrast image with no error anywhere.
pub const sdxl: Config = .{
    .z_channels = 4,
    .base_channels = 128,
    .block_out_channels = &.{ 128, 256, 512, 512 },
    .layers_per_block = 3,
    .norm_groups = 32,
    .norm_eps = 1e-6,
    .scaling_factor = 0.13025,
};

/// The 16-channel `AutoencoderKL` the Flux lineage uses, which Z-Image inherits
/// through `Lumina2` (`latent_formats.Flux`). Architecturally the SD decoder with
/// four times the latent width; the naming and the absence of `post_quant_conv` are
/// discovered from the file rather than stated here.
pub const flux: Config = .{
    .z_channels = 16,
    .base_channels = 128,
    .block_out_channels = &.{ 128, 256, 512, 512 },
    .layers_per_block = 3,
    .norm_groups = 32,
    .norm_eps = 1e-6,
    .scaling_factor = 0.3611,
    .shift_factor = 0.1159,
    .act_f16 = true,
};

pub const latent_channels = 4;
pub const spatial_scale = 8;

/// Linear latent->RGB approximation for the live sampling preview, from ComfyUI's
/// `latent_formats.SD15` / `.SDXL`. A per-pixel 4x3 matmul on the *scaled* latent
/// (the same space the sampler works in, i.e. what `decode` divides by
/// `scaling_factor`), so it needs no VAE and runs every step.
///
/// The two families' factors are genuinely different matrices and SDXL carries a
/// bias where SD1.5's is zero; using one for the other gives a preview with plausible
/// structure and wrong colours.
pub const latent_rgb_factors_sd15 = [latent_channels][3]f32{
    .{ 0.3512, 0.2297, 0.3227 },
    .{ 0.3250, 0.4974, 0.2350 },
    .{ -0.2829, 0.1762, 0.2721 },
    .{ -0.2120, -0.2616, -0.7177 },
};
pub const latent_rgb_bias_sd15 = [3]f32{ 0, 0, 0 };

pub const latent_rgb_factors_sdxl = [latent_channels][3]f32{
    .{ 0.3651, 0.4232, 0.4341 },
    .{ -0.2533, -0.0042, 0.1068 },
    .{ 0.1076, 0.1111, -0.0362 },
    .{ -0.3165, -0.2492, -0.2188 },
};
pub const latent_rgb_bias_sdxl = [3]f32{ 0.1084, -0.0175, -0.0011 };

/// Fill `rgb_out` (`[zh*zw][3]` RGB8) with the latent2rgb preview of the planar
/// `[4][zh*zw]` sampler latent `z`. Same `(v + 1) / 2` mapping ComfyUI's
/// `Latent2RGBPreviewer` uses.
pub fn latentPreviewInto(
    rgb_out: []u8,
    z: []const f32,
    zh: usize,
    zw: usize,
    factors: *const [latent_channels][3]f32,
    bias: [3]f32,
) void {
    const plane = zh * zw;
    std.debug.assert(rgb_out.len >= plane * 3 and z.len >= latent_channels * plane);
    for (0..plane) |p| {
        var acc = bias;
        inline for (0..latent_channels) |c| {
            const v = z[c * plane + p];
            acc[0] += v * factors[c][0];
            acc[1] += v * factors[c][1];
            acc[2] += v * factors[c][2];
        }
        inline for (0..3) |ch| {
            const u = std.math.clamp((acc[ch] + 1.0) * 0.5, 0.0, 1.0) * 255.0;
            rgb_out[p * 3 + ch] = @intFromFloat(u);
        }
    }
}

pub const GroupNormW = struct { w: []const f32, b: []const f32 };

/// LDM's autoencoder `ResnetBlock`, no timestep embedding, unlike the UNet's.
pub const Resnet = struct {
    norm1: GroupNormW,
    conv1: Conv2d,
    norm2: GroupNormW,
    conv2: Conv2d,
    /// 1x1 projection, present only where the width changes.
    nin: ?Conv2d,
    in_ch: usize,
    out_ch: usize,
};

pub const AttnBlock = struct {
    norm: GroupNormW,
    q: Conv2d,
    k: Conv2d,
    v: Conv2d,
    proj: Conv2d,
    channels: usize,
};

pub const Level = struct {
    blocks: []Resnet,
    /// 3x3 convolution after a nearest-neighbour 2x; absent on the outermost level.
    upsample: ?Conv2d,
};

/// The two activation widths a decode needs, in f32 elements.
///
/// They are NOT the same number, and treating them as one over-allocated a
/// third of every decode. `x` (the running activation) and `t` (a norm's output)
/// both reach the widest tensor in the ladder, for a 16-channel VAE at 1056x1584
/// that is 256 channels at FULL image resolution, 428M floats, but `u` only ever
/// receives a convolution's OUTPUT, so it peaks one level narrower (128 channels at
/// full resolution, 214M). Sizing all three at `wide` cost 856 MB per decode,
/// and made `estimatePeakBytes` demand room the decode never used.
/// Query rows per mid-block attention scores band, for a backend that materializes
/// the plane (Vulkan; both CUDA arms stream instead).
///
/// Shared by the kernel caller AND `estimatePeakBytes` on purpose. The whole
/// `seq²` plane is 2.73 GB at a 1056x1584 render, larger than the activations it
/// serves, so the estimate is only meaningful if it uses the same band the decode
/// will actually allocate. Two copies of this constant is exactly the drift that
/// makes a peak estimate lie.
///
/// A whole multiple of the 8-wide kernel tile, so a band's last tile is never
/// partial, which is what lets `attn_out` skip its row clamp when banded.
pub fn scoresBand(n: usize) usize {
    const cap: usize = (256 << 20) / 4; // f32 entries we are willing to hold
    var qb = @max(@as(usize, 8), (cap / @max(n, 1)) & ~@as(usize, 7));
    if (qb > n) qb = std.mem.alignForward(usize, n, 8);
    return qb;
}

pub const ActWidths = struct { wide: u64, out: u64 };

pub fn activationElems(cfg: Config, zh: usize, zw: usize) ActWidths {
    var lh: u64 = zh;
    var lw: u64 = zw;
    var c: u64 = cfg.innermost();
    var wide: u64 = 0;
    var out: u64 = 0;
    for (0..cfg.levels()) |i| {
        const level_idx = cfg.levels() - 1 - i;
        const out_ch: u64 = cfg.block_out_channels[level_idx];
        wide = @max(wide, lh * lw * @max(c, out_ch));
        out = @max(out, lh * lw * out_ch);
        c = out_ch;
        if (level_idx > 0) {
            lh *= 2;
            lw *= 2;
        }
    }
    wide = @max(wide, lh * lw * @max(c, 3));
    out = @max(out, lh * lw * 3); // conv_out writes 3 channels at full resolution
    return .{ .wide = wide, .out = out };
}

pub const Decoder = struct {
    arena: std.heap.ArenaAllocator,
    cfg: Config,
    /// The spelling this checkpoint turned out to use. See `Naming`.
    naming: Naming,
    /// Null for the Flux-lineage VAE, which has no `post_quant_conv` at all,
    /// BFL dropped both `quant_conv` and `post_quant_conv`, so requiring it would
    /// fail the load of every Flux/Z-Image VAE in existence.
    post_quant: ?Conv2d,
    conv_in: Conv2d,
    mid1: Resnet,
    mid_attn: AttnBlock,
    mid2: Resnet,
    /// Innermost first, i.e. execution order (`up.3` ... `up.0`).
    levels: []Level,
    norm_out: GroupNormW,
    conv_out: Conv2d,

    /// Which naming scheme this store uses, from a tensor only one of them has.
    ///
    /// Probed on the middle block, deliberately: `decoder.conv_in` and
    /// `decoder.conv_out` are spelled identically in both schemes, so the obvious
    /// entry points cannot tell them apart. `decoder.mid_block.resnets.0` exists
    /// only in diffusers exports.
    ///
    /// Defaulting to LDM on an unrecognized store is safe, the very next lookup is
    /// `decoder.mid.block_1.norm1.weight`, which reports the missing tensor by name.
    pub fn detectNaming(store: WeightStore, prefix: []const u8) Naming {
        var buf: [200]u8 = undefined;
        const nm = std.fmt.bufPrint(&buf, "{s}decoder.mid_block.resnets.0.norm1.weight", .{prefix}) catch return .ldm;
        return if (store.get(nm) != null) .diffusers else .ldm;
    }

    /// `prefix` is where the autoencoder sits: `first_stage_model.` in an LDM
    /// single-file checkpoint, `""` for a bare diffusers-style VAE export.
    pub fn load(gpa: std.mem.Allocator, store: WeightStore, cfg: Config, prefix: []const u8) !Decoder {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();
        const naming = detectNaming(store, prefix);
        const l: Loader = .{ .store = store, .alloc = alloc, .pfx = prefix, .cfg = cfg, .naming = naming };

        const inner = cfg.innermost();
        const ldm = naming == .ldm;
        const post_quant = if (l.has("post_quant_conv.weight"))
            try l.conv("post_quant_conv", .{}, cfg.z_channels, cfg.z_channels, 1, 1, 0)
        else
            null;
        const conv_in = try l.conv("decoder.conv_in", .{}, inner, cfg.z_channels, 3, 1, 1);
        const mid1 = if (ldm)
            try l.resnet("decoder.mid.block_1", .{}, inner, inner)
        else
            try l.resnet("decoder.mid_block.resnets.0", .{}, inner, inner);
        const mid_attn = if (ldm)
            try l.attn("decoder.mid.attn_1", .{}, inner)
        else
            try l.attn("decoder.mid_block.attentions.0", .{}, inner);
        const mid2 = if (ldm)
            try l.resnet("decoder.mid.block_2", .{}, inner, inner)
        else
            try l.resnet("decoder.mid_block.resnets.1", .{}, inner, inner);

        // Execution order: innermost level first. LDM numbers that level
        // `levels-1` and diffusers numbers it `0`, the same walk under two indices.
        // `ch` tracks the width coming in, which changes at the first resnet of a
        // level whenever the level is narrower than the one before it.
        const levels = try alloc.alloc(Level, cfg.levels());
        var ch = inner;
        for (0..cfg.levels()) |i| {
            const ldm_idx = cfg.levels() - 1 - i; // 3, 2, 1, 0
            const idx = if (ldm) ldm_idx else i;
            const out_ch = cfg.block_out_channels[ldm_idx];
            const blocks = try alloc.alloc(Resnet, cfg.layers_per_block);
            for (blocks, 0..) |*b, j| {
                b.* = if (ldm)
                    try l.resnet("decoder.up.{d}.block.{d}", .{ idx, j }, ch, out_ch)
                else
                    try l.resnet("decoder.up_blocks.{d}.resnets.{d}", .{ idx, j }, ch, out_ch);
                ch = out_ch;
            }
            // The last level executed (the outermost) has no upsample in either
            // scheme; it is `up.0` for LDM and `up_blocks.levels-1` for diffusers.
            const has_up = i + 1 < cfg.levels();
            levels[i] = .{
                .blocks = blocks,
                .upsample = if (!has_up) null else if (ldm)
                    try l.conv("decoder.up.{d}.upsample.conv", .{idx}, ch, ch, 3, 1, 1)
                else
                    try l.conv("decoder.up_blocks.{d}.upsamplers.0.conv", .{idx}, ch, ch, 3, 1, 1),
            };
        }

        const norm_out = if (ldm)
            try l.groupNorm("decoder.norm_out", .{}, ch)
        else
            try l.groupNorm("decoder.conv_norm_out", .{}, ch);
        const conv_out = try l.conv("decoder.conv_out", .{}, 3, ch, 3, 1, 1);

        return .{
            .arena = arena,
            .cfg = cfg,
            .naming = naming,
            .post_quant = post_quant,
            .conv_in = conv_in,
            .mid1 = mid1,
            .mid_attn = mid_attn,
            .mid2 = mid2,
            .levels = levels,
            .norm_out = norm_out,
            .conv_out = conv_out,
        };
    }

    pub fn deinit(self: *Decoder) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// Upper bound on the device activation bytes a whole-image decode of a
    /// `zh x zw` latent needs, for the VAE-decode reclaim ladder to pre-free
    /// against. Mirrors `sd_vae_cuda` / `sd_vae_gpu`'s own sizing exactly: three
    /// f32 buffers (x / t / u) at the widest (positions x channels) product any
    /// level reaches, plus the latent-resolution attention scratch.
    ///
    /// The width that matters is `max(level input, level output)`: the first resnet of a
    /// level reads the PREVIOUS (wider) level's activation at the NEW (doubled)
    /// resolution, so sizing off `block_out_channels` alone under-estimates by 2x.
    /// `scores_resident` says whether the caller's mid-block attention materializes
    /// the O(seq²) scores plane. It is a PARAMETER because the two GPU backends
    /// genuinely differ, and guessing either way costs something real: `sd_vae_gpu`
    /// materializes an f32 plane, while both CUDA arms stream (cuDNN's fused SDPA
    /// under `.libs`, `be.attn`'s online softmax otherwise) and allocate nothing.
    /// At a 132x198 latent that term is 2.73 GB, so charging it on CUDA makes the ladder
    /// evict GBs of resident DiT weights before a decode that never needed the room, then
    /// re-stream them for the next image.
    /// `act_f16` is the CALLER's storage format, not `cfg.act_f16`: only the CUDA
    /// decoder implements it, and reporting the narrow figure to a backend that still
    /// stores f32 sends the ladder into a whole-image decode that cannot fit, which is
    /// the OOM this estimate exists to prevent.
    pub fn estimatePeakBytes(self: *const Decoder, zh: usize, zw: usize, scores_resident: bool, act_f16: bool) u64 {
        const w = activationElems(self.cfg, zh, zw);
        const max_elems = w.wide;
        // q/k/v/out at latent resolution + the O(seq^2) mid-block scores plane
        // (f16 on the GPU paths, which query-tile it, so this term is generous).
        const seq: u64 = @as(u64, zh) * zw;
        // 4 bytes for the scores plane, not 2: `sd_vae_gpu` materializes it in f32
        // (the Flux VAE's logits overflow f16, see that file's `attn`). Under-reporting
        // it here is what would make the ladder attempt a whole-image decode that cannot
        // fit and discover it as an allocation failure.
        // `scores_resident` is the Vulkan arm, which materializes ONE QUERY BAND
        // of the plane (`sd_vae_gpu.scoresBand`), not the whole `seq²`. Both GPU
        // arms also keep a k-major f32 copy of Q and K.
        const band: u64 = scoresBand(@intCast(seq));
        const attn = 4 * seq * self.cfg.innermost() * 4 +
            if (scores_resident) 2 * seq * self.cfg.innermost() * 4 + band * seq * 4 else 0;
        // The activation buffers narrow with `act_f16`; the attention scratch does
        // not (q/k/v/out stay f32, they are at latent resolution, and both GPU
        // attention paths take f32 buffers).
        const abytes: u64 = if (act_f16) 2 else 4;
        return (2 * max_elems + w.out) * abytes + attn;
    }

    /// `z` is channel-last `[lat_h*lat_w][z_channels]`, already divided by
    /// `scaling_factor`. Returns channel-last `[8*lat_h * 8*lat_w][3]` in the
    /// decoder's own output range (roughly [-1, 1]); the caller converts to bytes.
    pub fn decode(
        self: *const Decoder,
        io: std.Io,
        gpa: std.mem.Allocator,
        z: []const f32,
        lat_h: usize,
        lat_w: usize,
    ) ![]f32 {
        const cfg = self.cfg;
        std.debug.assert(z.len == lat_h * lat_w * cfg.z_channels);

        const pq = try gpa.alloc(f32, z.len);
        defer gpa.free(pq);
        if (self.post_quant) |c| {
            try ops.conv.conv2d(io, gpa, pq, z, lat_h, lat_w, c);
        } else {
            @memcpy(pq, z);
        }

        var h = lat_h;
        var w = lat_w;
        var ch = cfg.innermost();
        var cur = try gpa.alloc(f32, h * w * ch);
        errdefer gpa.free(cur);
        try ops.conv.conv2d(io, gpa, cur, pq, h, w, self.conv_in);

        cur = try applyResnet(io, gpa, self.mid1, cur, h, w, cfg);
        try applyAttn(io, gpa, self.mid_attn, cur, h, w, cfg);
        cur = try applyResnet(io, gpa, self.mid2, cur, h, w, cfg);

        for (self.levels) |level| {
            for (level.blocks) |b| {
                cur = try applyResnet(io, gpa, b, cur, h, w, cfg);
                ch = b.out_ch;
            }
            if (level.upsample) |up| {
                const oh = h * 2;
                const ow = w * 2;
                const expanded = try gpa.alloc(f32, oh * ow * ch);
                defer gpa.free(expanded);
                for (0..oh) |y| {
                    for (0..ow) |x| {
                        @memcpy(
                            expanded[(y * ow + x) * ch ..][0..ch],
                            cur[((y / 2) * w + (x / 2)) * ch ..][0..ch],
                        );
                    }
                }
                const conved = try gpa.alloc(f32, oh * ow * ch);
                errdefer gpa.free(conved);
                try ops.conv.conv2d(io, gpa, conved, expanded, oh, ow, up);
                gpa.free(cur);
                cur = conved;
                h = oh;
                w = ow;
            }
        }

        ops.norm.groupNorm(cur, cur, ch, cfg.norm_groups, self.norm_out.w, self.norm_out.b, cfg.norm_eps);
        ops.act.silu(cur);
        const rgb = try gpa.alloc(f32, h * w * 3);
        errdefer gpa.free(rgb);
        try ops.conv.conv2d(io, gpa, rgb, cur, h, w, self.conv_out);
        gpa.free(cur);
        return rgb;
    }
};

/// Frees `src` and returns the new activation, since a resnet may change the width.
fn applyResnet(
    io: std.Io,
    gpa: std.mem.Allocator,
    r: Resnet,
    src: []f32,
    h: usize,
    w: usize,
    cfg: Config,
) ![]f32 {
    const n = h * w;
    std.debug.assert(src.len == n * r.in_ch);

    const tmp = try gpa.alloc(f32, n * r.in_ch);
    defer gpa.free(tmp);
    ops.norm.groupNorm(tmp, src, r.in_ch, cfg.norm_groups, r.norm1.w, r.norm1.b, cfg.norm_eps);
    ops.act.silu(tmp);

    const mid = try gpa.alloc(f32, n * r.out_ch);
    defer gpa.free(mid);
    try ops.conv.conv2d(io, gpa, mid, tmp, h, w, r.conv1);

    const tmp2 = try gpa.alloc(f32, n * r.out_ch);
    defer gpa.free(tmp2);
    ops.norm.groupNorm(tmp2, mid, r.out_ch, cfg.norm_groups, r.norm2.w, r.norm2.b, cfg.norm_eps);
    ops.act.silu(tmp2);

    const out = try gpa.alloc(f32, n * r.out_ch);
    errdefer gpa.free(out);
    try ops.conv.conv2d(io, gpa, out, tmp2, h, w, r.conv2);

    if (r.nin) |nin| {
        const shortcut = try gpa.alloc(f32, n * r.out_ch);
        defer gpa.free(shortcut);
        try ops.conv.conv2d(io, gpa, shortcut, src, h, w, nin);
        for (out, shortcut) |*o, s| o.* += s;
    } else {
        for (out, src) |*o, s| o.* += s;
    }
    gpa.free(src);
    return out;
}

/// In-place: `x += proj(attention(norm(x)))`.
fn applyAttn(
    io: std.Io,
    gpa: std.mem.Allocator,
    a: AttnBlock,
    x: []f32,
    h: usize,
    w: usize,
    cfg: Config,
) !void {
    const n = h * w;
    const ch = a.channels;
    std.debug.assert(x.len == n * ch);

    const norm = try gpa.alloc(f32, n * ch);
    defer gpa.free(norm);
    ops.norm.groupNorm(norm, x, ch, cfg.norm_groups, a.norm.w, a.norm.b, cfg.norm_eps);

    const q = try gpa.alloc(f32, n * ch);
    defer gpa.free(q);
    const k = try gpa.alloc(f32, n * ch);
    defer gpa.free(k);
    const v = try gpa.alloc(f32, n * ch);
    defer gpa.free(v);
    try ops.conv.conv2d(io, gpa, q, norm, h, w, a.q);
    try ops.conv.conv2d(io, gpa, k, norm, h, w, a.k);
    try ops.conv.conv2d(io, gpa, v, norm, h, w, a.v);

    const attn = try gpa.alloc(f32, n * ch);
    defer gpa.free(attn);
    // One head over all `ch` channels, scale 1/sqrt(ch), LDM's `AttnBlock`.
    try ops.attention.attention(io, gpa, attn, q, k, v, .{
        .seq_q = n,
        .seq_kv = n,
        .n_heads = 1,
        .n_kv_heads = 1,
        .head_dim = ch,
    });

    const projected = try gpa.alloc(f32, n * ch);
    defer gpa.free(projected);
    try ops.conv.conv2d(io, gpa, projected, attn, h, w, a.proj);
    for (x, projected) |*o, p| o.* += p;
}

const Loader = struct {
    store: WeightStore,
    alloc: std.mem.Allocator,
    pfx: []const u8,
    cfg: Config,

    naming: Naming,

    /// Whether a tensor (named relative to the store prefix) is present.
    fn has(l: Loader, rel: []const u8) bool {
        var buf: [200]u8 = undefined;
        const nm = std.fmt.bufPrint(&buf, "{s}{s}", .{ l.pfx, rel }) catch return false;
        return l.store.get(nm) != null;
    }

    fn view(l: Loader, nm: []const u8) !safetensors.TensorView {
        return l.store.get(nm) orelse {
            std.log.err("sd_vae: missing {s}", .{nm});
            return error.MissingTensor;
        };
    }

    fn name(l: Loader, buf: []u8, comptime fmt: []const u8, args: anytype, suffix: []const u8) ![]u8 {
        var fbs = std.Io.Writer.fixed(buf);
        try fbs.writeAll(l.pfx);
        try fbs.print(fmt, args);
        try fbs.writeAll(suffix);
        return fbs.buffered();
    }

    fn vec(l: Loader, comptime fmt: []const u8, args: anytype, suffix: []const u8, len: usize) ![]f32 {
        var buf: [200]u8 = undefined;
        const nm = try l.name(&buf, fmt, args, suffix);
        const v = try l.view(nm);
        if (v.info.elemCount() != len) {
            std.log.err("sd_vae: {s} has {d} elements, expected {d}", .{ nm, v.info.elemCount(), len });
            return error.ShapeMismatch;
        }
        return v.toF32Alloc(l.alloc);
    }

    fn groupNorm(l: Loader, comptime fmt: []const u8, args: anytype, ch: usize) !GroupNormW {
        return .{
            .w = try l.vec(fmt, args, ".weight", ch),
            .b = try l.vec(fmt, args, ".bias", ch),
        };
    }

    fn conv(l: Loader, comptime fmt: []const u8, args: anytype, co: usize, ci: usize, k: usize, stride: usize, pad: usize) !Conv2d {
        var buf: [200]u8 = undefined;
        const nm = try l.name(&buf, fmt, args, ".weight");
        const v = try l.view(nm);
        const shape = v.info.shape.slice();
        // A 1x1 convolution and an `nn.Linear` of the same widths are byte-identical
        // row-major, and diffusers stores the VAE's middle attention as the latter,
        // so `[co, ci]` is accepted wherever `[co, ci, 1, 1]` is.
        const ok = (shape.len == 4 and shape[0] == co and shape[1] == ci and shape[2] == k and shape[3] == k) or
            (k == 1 and shape.len == 2 and shape[0] == co and shape[1] == ci);
        if (!ok) {
            std.log.err("sd_vae: {s} has shape {any}, expected [{d}, {d}, {d}, {d}]", .{ nm, shape, co, ci, k, k });
            return error.ShapeMismatch;
        }
        const torch_w = try v.toF32Alloc(l.alloc);
        defer l.alloc.free(torch_w);
        // The GEMM this convolution becomes carries the checkpoint name, so an
        // activation capture can attribute it (see `ops.conv.Conv2d.tag`).
        const tag = try l.alloc.dupe(u8, nm);
        return .{
            .w = try ops.conv.packWeight(l.alloc, torch_w, co, ci, k),
            .tag = tag,
            .b = try l.vec(fmt, args, ".bias", co),
            .co = co,
            .ci = ci,
            .k = k,
            .stride = stride,
            .pad = pad,
        };
    }

    fn resnet(l: Loader, comptime fmt: []const u8, args: anytype, in_ch: usize, out_ch: usize) !Resnet {
        var buf: [200]u8 = undefined;
        // Relative base: the inner helpers add the store prefix themselves (a
        // pre-prefixed base doubles it, and the symptom is a missing tensor).
        const base = try std.fmt.bufPrint(&buf, fmt, args);
        return .{
            .norm1 = try l.groupNorm("{s}.norm1", .{base}, in_ch),
            .conv1 = try l.conv("{s}.conv1", .{base}, out_ch, in_ch, 3, 1, 1),
            .norm2 = try l.groupNorm("{s}.norm2", .{base}, out_ch),
            .conv2 = try l.conv("{s}.conv2", .{base}, out_ch, out_ch, 3, 1, 1),
            .nin = if (in_ch == out_ch) null else if (l.naming == .ldm)
                try l.conv("{s}.nin_shortcut", .{base}, out_ch, in_ch, 1, 1, 0)
            else
                try l.conv("{s}.conv_shortcut", .{base}, out_ch, in_ch, 1, 1, 0),
            .in_ch = in_ch,
            .out_ch = out_ch,
        };
    }

    fn attn(l: Loader, comptime fmt: []const u8, args: anytype, ch: usize) !AttnBlock {
        var buf: [200]u8 = undefined;
        const base = try std.fmt.bufPrint(&buf, fmt, args);
        // diffusers stores these four as `nn.Linear` rather than 1x1 convolutions.
        // Same bytes, same order; `conv` accepts the 2-D shape when k == 1.
        if (l.naming == .diffusers) return .{
            .norm = try l.groupNorm("{s}.group_norm", .{base}, ch),
            .q = try l.conv("{s}.to_q", .{base}, ch, ch, 1, 1, 0),
            .k = try l.conv("{s}.to_k", .{base}, ch, ch, 1, 1, 0),
            .v = try l.conv("{s}.to_v", .{base}, ch, ch, 1, 1, 0),
            .proj = try l.conv("{s}.to_out.0", .{base}, ch, ch, 1, 1, 0),
            .channels = ch,
        };
        return .{
            .norm = try l.groupNorm("{s}.norm", .{base}, ch),
            .q = try l.conv("{s}.q", .{base}, ch, ch, 1, 1, 0),
            .k = try l.conv("{s}.k", .{base}, ch, ch, 1, 1, 0),
            .v = try l.conv("{s}.v", .{base}, ch, ch, 1, 1, 0),
            .proj = try l.conv("{s}.proj_out", .{base}, ch, ch, 1, 1, 0),
            .channels = ch,
        };
    }
};

// --- tests ------------------------------------------------------------------

const testing = std.testing;
const test_gate = @import("../test_gate.zig");

const ref_path = "src/models/assets/sd15_ref.safetensors";
const sd15_ckpt = "/home/qt/genai/comfyui/models/checkpoints/sd1.5/perfectdeliberate_v20.safetensors";

test "the SD1.5 VAE decoder matches diffusers.AutoencoderKL.decode" {
    const gpa = testing.allocator;
    const io = testing.io;
    try test_gate.requireIntegration();
    try test_gate.requireModelFile(io, sd15_ckpt);
    try test_gate.requireModelFile(io, ref_path);

    var ref = try safetensors.SafeTensors.open(gpa, io, ref_path);
    defer ref.deinit();
    var ck = try safetensors.SafeTensors.open(gpa, io, sd15_ckpt);
    defer ck.deinit();

    var dec = try Decoder.load(gpa, .{ .safetensors = &ck }, sd15, "first_stage_model.");
    defer dec.deinit();

    const lat = 8;
    const z_planar = try ref.get("vae.latent").?.toF32Alloc(gpa); // [1][4][8][8]
    defer gpa.free(z_planar);
    const want_planar = try ref.get("vae.image").?.toF32Alloc(gpa); // [1][3][64][64]
    defer gpa.free(want_planar);

    const z = try gpa.alloc(f32, z_planar.len);
    defer gpa.free(z);
    for (0..lat * lat) |p| {
        for (0..4) |c| z[p * 4 + c] = z_planar[c * lat * lat + p];
    }

    const rgb = try dec.decode(io, gpa, z, lat, lat);
    defer gpa.free(rgb);

    const px = lat * 8 * lat * 8;
    try testing.expectEqual(px * 3, rgb.len);
    var l2_ref: f64 = 0;
    var l2_err: f64 = 0;
    var max_abs: f32 = 0;
    for (0..px) |p| {
        for (0..3) |c| {
            const e = want_planar[c * px + p];
            const a = rgb[p * 3 + c];
            max_abs = @max(max_abs, @abs(e - a));
            l2_ref += @as(f64, e) * e;
            l2_err += @as(f64, e - a) * (e - a);
        }
    }
    const rel = @sqrt(l2_err / l2_ref);
    errdefer std.debug.print("vae decode: rel L2 {d:.6} max_abs {d:.6}\n", .{ rel, max_abs });
    // f32 accumulation order differs from torch's across 13 conv blocks plus an
    // attention; this is numeric agreement, not bit-identity.
    try testing.expect(rel < 2e-3);
}

test "the VAE decoder still matches diffusers at the size images are rendered at" {
    // The 8x8 case leaves every convolution in a single im2col band and the
    // mid-block attention at 64 positions. A 512x512 decode bands up to 512 ways and
    // attends over 4096, a different code path in both ops, and the one every real
    // render takes.
    const gpa = testing.allocator;
    const io = testing.io;
    try test_gate.requireIntegration();
    try test_gate.requireModelFile(io, sd15_ckpt);
    try test_gate.requireModelFile(io, ref_path);

    var ref = try safetensors.SafeTensors.open(gpa, io, ref_path);
    defer ref.deinit();
    var ck = try safetensors.SafeTensors.open(gpa, io, sd15_ckpt);
    defer ck.deinit();

    var dec = try Decoder.load(gpa, .{ .safetensors = &ck }, sd15, "first_stage_model.");
    defer dec.deinit();

    const lat = 64;
    const z_planar = try ref.get("vae_big.latent").?.toF32Alloc(gpa);
    defer gpa.free(z_planar);
    const want_planar = try ref.get("vae_big.image").?.toF32Alloc(gpa); // stored f16
    defer gpa.free(want_planar);

    const z = try gpa.alloc(f32, z_planar.len);
    defer gpa.free(z);
    for (0..lat * lat) |p| {
        for (0..4) |c| z[p * 4 + c] = z_planar[c * lat * lat + p];
    }

    const rgb = try dec.decode(io, gpa, z, lat, lat);
    defer gpa.free(rgb);

    const px = lat * 8 * lat * 8;
    var l2_ref: f64 = 0;
    var l2_err: f64 = 0;
    for (0..px) |p| {
        for (0..3) |c| {
            const e = want_planar[c * px + p];
            const a = rgb[p * 3 + c];
            l2_ref += @as(f64, e) * e;
            l2_err += @as(f64, e - a) * (e - a);
        }
    }
    const rel = @sqrt(l2_err / l2_ref);
    errdefer std.debug.print("vae decode at {d}px: rel L2 {d:.6}\n", .{ lat * 8, rel });
    // Looser than the 8x8 case's 2e-3: the reference is stored f16 here (~5e-4
    // relative on its own), which dominates the budget.
    try testing.expect(rel < 3e-3);
}

const sdxl_ref_path = "src/models/assets/sdxl_ref.safetensors";
const sdxl_ckpt = "/home/qt/genai/comfyui/models/checkpoints/sdxl/blackMAGICXL_v145.safetensors";

test "the SDXL VAE decoder matches ComfyUI's AutoencoderKL at both sizes" {
    // The decoder is architecturally identical to SD1.5's, so this is really a test of
    // two things the `sdxl` config asserts: that the same code path loads SDXL's weights
    // (same names, same widths), and that nothing about the wider UNet leaked into the
    // VAE. The scale factor is NOT exercised here, `decode` takes an already-divided
    // latent, exactly as the reference does, which is what makes this the decoder's test
    // and not the pipeline's.
    const gpa = testing.allocator;
    const io = testing.io;
    try test_gate.requireIntegration();
    try test_gate.requireModelFile(io, sdxl_ckpt);
    try test_gate.requireModelFile(io, sdxl_ref_path);

    var ref = try safetensors.SafeTensors.open(gpa, io, sdxl_ref_path);
    defer ref.deinit();
    var ck = try safetensors.SafeTensors.open(gpa, io, sdxl_ckpt);
    defer ck.deinit();

    var dec = try Decoder.load(gpa, .{ .safetensors = &ck }, sdxl, "first_stage_model.");
    defer dec.deinit();

    // The 8x8 case leaves every convolution in one im2col band and the mid attention at
    // 64 positions; 64x64 bands many ways and attends over 4096. The f16-stored reference
    // for the big case contributes ~5e-4 of its own tolerance.
    const cases = [_]struct { lat: usize, z: []const u8, img: []const u8, tol: f64 }{
        .{ .lat = 8, .z = "vae.latent", .img = "vae.image", .tol = 2e-3 },
        .{ .lat = 64, .z = "vae_big.latent", .img = "vae_big.image", .tol = 3e-3 },
    };
    for (cases) |c| {
        const z_planar = try ref.get(c.z).?.toF32Alloc(gpa);
        defer gpa.free(z_planar);
        const want_planar = try ref.get(c.img).?.toF32Alloc(gpa);
        defer gpa.free(want_planar);

        const z = try gpa.alloc(f32, z_planar.len);
        defer gpa.free(z);
        for (0..c.lat * c.lat) |p| {
            for (0..4) |ci| z[p * 4 + ci] = z_planar[ci * c.lat * c.lat + p];
        }

        const rgb = try dec.decode(io, gpa, z, c.lat, c.lat);
        defer gpa.free(rgb);

        const px = c.lat * 8 * c.lat * 8;
        try testing.expectEqual(px * 3, rgb.len);
        var l2_ref: f64 = 0;
        var l2_err: f64 = 0;
        for (0..px) |p| {
            for (0..3) |ci| {
                const e = want_planar[ci * px + p];
                const a = rgb[p * 3 + ci];
                l2_ref += @as(f64, e) * e;
                l2_err += @as(f64, e - a) * (e - a);
            }
        }
        const rel = @sqrt(l2_err / l2_ref);
        errdefer std.debug.print("sdxl vae decode at {d}px: rel L2 {d:.6}\n", .{ c.lat * 8, rel });
        try testing.expect(rel < c.tol);
    }
}

test "the two SD VAE configs differ only in the latent scale" {
    // The scale factor is the one number that is easy to carry over wrongly and produces
    // no error, a washed, low-contrast image. Asserting the rest of the config is
    // *identical* is what makes "only the scale differs" a checked claim rather than a
    // comment.
    try testing.expectEqual(sd15.z_channels, sdxl.z_channels);
    try testing.expectEqual(sd15.base_channels, sdxl.base_channels);
    try testing.expectEqualSlices(usize, sd15.block_out_channels, sdxl.block_out_channels);
    try testing.expectEqual(sd15.layers_per_block, sdxl.layers_per_block);
    try testing.expectEqual(sd15.norm_groups, sdxl.norm_groups);
    try testing.expectEqual(sd15.norm_eps, sdxl.norm_eps);
    try testing.expect(sd15.scaling_factor != sdxl.scaling_factor);
    // The latent2rgb preview is the one place they must NOT be shared: two different
    // matrices, and a bias on SDXL only.
    try testing.expect(latent_rgb_factors_sd15[0][0] != latent_rgb_factors_sdxl[0][0]);
    try testing.expect(latent_rgb_bias_sd15[0] == 0 and latent_rgb_bias_sdxl[0] != 0);
}

test "the SD latent2rgb preview matches ComfyUI's factors, bias and clamp" {
    // Hand-computed from ComfyUI's latent_formats factors with its
    // `((v + 1) / 2).clamp(0, 1) * 255` mapping, so this pins the constants AND the
    // mapping rather than restating the implementation.
    const plane = 2;
    var rgb: [plane * 3]u8 = undefined;

    // SD1.5, unit weight on channel 0 at pixel 0 -> factors[0]; a large value at
    // pixel 1 -> clamped white on R.
    var z15 = [_]f32{0} ** (latent_channels * plane);
    z15[0] = 1;
    z15[1] = 10;
    latentPreviewInto(&rgb, &z15, 1, plane, &latent_rgb_factors_sd15, latent_rgb_bias_sd15);
    try testing.expectEqualSlices(u8, &.{ 172, 156, 168 }, rgb[0..3]);
    try testing.expectEqual(@as(u8, 255), rgb[3]);

    // SDXL, an all-zero latent -> the bias alone (SD1.5's would give a flat 127).
    const z_xl = [_]f32{0} ** (latent_channels * plane);
    latentPreviewInto(&rgb, &z_xl, 1, plane, &latent_rgb_factors_sdxl, latent_rgb_bias_sdxl);
    try testing.expectEqualSlices(u8, &.{ 141, 125, 127 }, rgb[0..3]);
    latentPreviewInto(&rgb, &z_xl, 1, plane, &latent_rgb_factors_sd15, latent_rgb_bias_sd15);
    try testing.expectEqualSlices(u8, &.{ 127, 127, 127 }, rgb[0..3]);
}

const zimage_ref_path = "src/models/assets/zimage_ref.safetensors";
const zimage_vae_ckpt = "/home/qt/genai/comfyui/models/vae/z-image-turbo.vae.safetensors";

test "the 16-channel Flux/Z-Image VAE decoder matches ComfyUI's AutoencoderKL" {
    // Same decoder graph as SD's, so what this actually exercises is the three
    // things the `flux` config changes, each of which fails differently:
    //   - diffusers naming, whose up-block index runs the OPPOSITE way. Reading
    //     it as LDM would invert the channel ramp, which fails as a shape mismatch
    //     at the first level rather than as a bad image, but only because the
    //     widths happen to differ; a symmetric ramp would decode silently wrong.
    //   - no `post_quant_conv`, which under the old loader was a hard
    //     MissingTensor on every Flux-lineage VAE in existence.
    //   - 16 latent channels, which changes only `conv_in`'s input width.
    const gpa = testing.allocator;
    const io = testing.io;
    try test_gate.requireIntegration();
    try test_gate.requireModelFile(io, zimage_vae_ckpt);
    try test_gate.requireModelFile(io, zimage_ref_path);

    var ref = try safetensors.SafeTensors.open(gpa, io, zimage_ref_path);
    defer ref.deinit();
    var ck = try safetensors.SafeTensors.open(gpa, io, zimage_vae_ckpt);
    defer ck.deinit();

    var dec = try Decoder.load(gpa, .{ .safetensors = &ck }, flux, "");
    defer dec.deinit();
    try testing.expect(dec.post_quant == null);

    // 8x8 keeps every convolution in one im2col band; 32x32 bands many ways and
    // attends over 1024 positions.
    for ([_]usize{ 0, 1 }) |ci| {
        var kb: [32]u8 = undefined;
        const z_planar = try (try ref.require(try std.fmt.bufPrint(&kb, "vae.{d}.z", .{ci}))).toF32Alloc(gpa);
        defer gpa.free(z_planar);
        const want_planar = try (try ref.require(try std.fmt.bufPrint(&kb, "vae.{d}.rgb", .{ci}))).toF32Alloc(gpa);
        defer gpa.free(want_planar);

        const zc = flux.z_channels;
        const lat = std.math.sqrt(z_planar.len / zc);
        const z = try gpa.alloc(f32, z_planar.len);
        defer gpa.free(z);
        for (0..lat * lat) |p| {
            for (0..zc) |c| z[p * zc + c] = z_planar[c * lat * lat + p];
        }

        const rgb = try dec.decode(io, gpa, z, lat, lat);
        defer gpa.free(rgb);

        const px = lat * 8 * lat * 8;
        try testing.expectEqual(px * 3, rgb.len);
        var l2_ref: f64 = 0;
        var l2_err: f64 = 0;
        for (0..px) |p| {
            for (0..3) |c| {
                const e = want_planar[c * px + p];
                const a = rgb[p * 3 + c];
                l2_ref += @as(f64, e) * e;
                l2_err += @as(f64, e - a) * (e - a);
            }
        }
        const rel = @sqrt(l2_err / l2_ref);
        errdefer std.debug.print("flux vae decode at {d}px: rel L2 {d:.6}\n", .{ lat * 8, rel });
        // Measured 2e-6 at 64px against an f32 reference, the decode is exact up to
        // reduction order, so this bound is deliberately far tighter than the SD
        // cases above (whose references are stored f16).
        try testing.expect(rel < 1e-4);
    }
}

test "the Flux VAE config differs from SD's only in the latent width and scaling" {
    // The decoder body is shared, so an accidental edit to one config silently
    // changes the other family's decode. Pin what is shared and what is not.
    try testing.expectEqualSlices(usize, sd15.block_out_channels, flux.block_out_channels);
    try testing.expectEqual(sd15.layers_per_block, flux.layers_per_block);
    try testing.expectEqual(sd15.norm_groups, flux.norm_groups);
    try testing.expectEqual(sd15.norm_eps, flux.norm_eps);

    try testing.expectEqual(@as(usize, 4), sd15.z_channels);
    try testing.expectEqual(@as(usize, 16), flux.z_channels);
    // SD's latent format has no shift term; Flux's does, and dropping it decodes a
    // latent offset by 0.116, a plausible image with a colour cast.
    try testing.expectEqual(@as(f32, 0), sd15.shift_factor);
    try testing.expect(flux.shift_factor != 0);
}

test "VAE naming is detected from the store, and the same VAE ships both ways" {
    // The claim this exists to check is not "detection works" but that BOTH spellings
    // are real: the Z-Image VAE the ComfyUI template points at is LDM-named while
    // another distribution of the same 244-tensor 16-channel decoder is diffusers-named.
    // A config field for the naming would be right for one of them and wrong for the
    // other.
    const gpa = testing.allocator;
    const io = testing.io;
    try test_gate.requireIntegration();

    const cases = [_]struct { path: []const u8, want: Naming }{
        .{ .path = "/home/qt/genai/comfyui/models/vae/ae.safetensors", .want = .ldm },
        .{ .path = zimage_vae_ckpt, .want = .diffusers },
    };
    var seen: usize = 0;
    for (cases) |c| {
        test_gate.requireModelFile(io, c.path) catch continue;
        var st = try safetensors.SafeTensors.open(gpa, io, c.path);
        defer st.deinit();
        const store: WeightStore = .{ .safetensors = &st };
        errdefer std.debug.print("{s}: detected {t}, expected {t}\n", .{ c.path, Decoder.detectNaming(store, ""), c.want });
        try testing.expectEqual(c.want, Decoder.detectNaming(store, ""));

        // And both load into the same decoder shape through that detection.
        var dec = try Decoder.load(gpa, store, flux, "");
        defer dec.deinit();
        try testing.expectEqual(c.want, dec.naming);
        try testing.expect(dec.post_quant == null);
        try testing.expectEqual(@as(usize, 4), dec.levels.len);
        seen += 1;
    }
    if (seen == 0) return error.SkipZigTest;
}
