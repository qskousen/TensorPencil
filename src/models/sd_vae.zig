//! The SD-family VAE decoder (`AutoencoderKL`, LDM `first_stage_model`) — latent to
//! RGB, mirroring `diffusers.AutoencoderKL.decode`.
//!
//! Structurally simpler than the UNet: no conditioning, no timestep, no skip
//! connections. A 1x1 `post_quant_conv`, a 3x3 stem into 512 channels, a middle block
//! (resnet, self-attention, resnet), then four levels of three resnets each doubling
//! the resolution on the way out, and a 3x3 head to 3 channels.
//!
//! Three things differ from the UNet and are easy to carry over wrongly:
//!
//! 1. **GroupNorm epsilon is 1e-6 here, 1e-5 in the UNet.** LDM's `Normalize` uses
//!    1e-6 for the autoencoder. The difference is invisible on well-conditioned
//!    activations and shows up as faint banding in flat regions.
//! 2. **`decoder.up.N` is indexed from the OUTERMOST level.** The forward runs
//!    `up.3, up.2, up.1, up.0` — 512, 512, 256, 128 channels — and `up.0` has no
//!    upsample. Reading the list forwards produces a decoder that runs but inverts
//!    the channel ramp.
//! 3. **The middle attention is single-head over pixels, with 1x1 convolutions for
//!    q/k/v** (not linear layers over a flattened sequence, though it is the same
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

/// SD1.5 and SDXL share this decoder shape; only the weights differ.
pub const Config = struct {
    /// Latent channels (4).
    z_channels: usize,
    /// Base width; `block_out_channels` below are multiples of it.
    base_channels: usize,
    /// Per-level widths, **outermost first** (`{128, 256, 512, 512}` for SD).
    block_out_channels: []const usize,
    /// Resnets per level on the decode path (LDM: `num_res_blocks + 1` = 3).
    layers_per_block: usize,
    norm_groups: usize,
    norm_eps: f32,
    /// What `decode` assumes the caller already divided the latent by.
    scaling_factor: f32,

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
};

/// SDXL's VAE is **architecturally identical** to SD1.5's — same widths, same depth, same
/// epsilon — and differs only in weights and in the latent scale it was trained against.
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

pub const latent_channels = 4;
pub const spatial_scale = 8;

/// Linear latent→RGB approximation for the live sampling preview, from ComfyUI's
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

/// LDM's autoencoder `ResnetBlock` — no timestep embedding, unlike the UNet's.
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

pub const Decoder = struct {
    arena: std.heap.ArenaAllocator,
    cfg: Config,
    post_quant: Conv2d,
    conv_in: Conv2d,
    mid1: Resnet,
    mid_attn: AttnBlock,
    mid2: Resnet,
    /// Innermost first, i.e. execution order (`up.3` … `up.0`).
    levels: []Level,
    norm_out: GroupNormW,
    conv_out: Conv2d,

    /// `prefix` is where the autoencoder sits: `first_stage_model.` in an LDM
    /// single-file checkpoint, `""` for a bare diffusers-style VAE export.
    pub fn load(gpa: std.mem.Allocator, store: WeightStore, cfg: Config, prefix: []const u8) !Decoder {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();
        const l: Loader = .{ .store = store, .alloc = alloc, .pfx = prefix, .cfg = cfg };

        const inner = cfg.innermost();
        const post_quant = try l.conv("post_quant_conv", .{}, cfg.z_channels, cfg.z_channels, 1, 1, 0);
        const conv_in = try l.conv("decoder.conv_in", .{}, inner, cfg.z_channels, 3, 1, 1);
        const mid1 = try l.resnet("decoder.mid.block_1", .{}, inner, inner);
        const mid_attn = try l.attn("decoder.mid.attn_1", .{}, inner);
        const mid2 = try l.resnet("decoder.mid.block_2", .{}, inner, inner);

        // Execution order: the highest `up` index first. `ch` tracks the width coming
        // in, which changes at the first resnet of a level whenever the level is
        // narrower than the one before it.
        const levels = try alloc.alloc(Level, cfg.levels());
        var ch = inner;
        for (0..cfg.levels()) |i| {
            const level_idx = cfg.levels() - 1 - i; // 3, 2, 1, 0
            const out_ch = cfg.block_out_channels[level_idx];
            const blocks = try alloc.alloc(Resnet, cfg.layers_per_block);
            for (blocks, 0..) |*b, j| {
                b.* = try l.resnet("decoder.up.{d}.block.{d}", .{ level_idx, j }, ch, out_ch);
                ch = out_ch;
            }
            levels[i] = .{
                .blocks = blocks,
                .upsample = if (level_idx > 0)
                    try l.conv("decoder.up.{d}.upsample.conv", .{level_idx}, ch, ch, 3, 1, 1)
                else
                    null,
            };
        }

        const norm_out = try l.groupNorm("decoder.norm_out", .{}, ch);
        const conv_out = try l.conv("decoder.conv_out", .{}, 3, ch, 3, 1, 1);

        return .{
            .arena = arena,
            .cfg = cfg,
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
    /// ⚠️ The width that matters is `max(level input, level output)`: the first
    /// resnet of a level reads the PREVIOUS (wider) level's activation at the NEW
    /// (doubled) resolution, so sizing off `block_out_channels` alone
    /// under-estimates by 2x — the same trap that cost a wrong decode in
    /// `sd_vae_gpu`.
    pub fn estimatePeakBytes(self: *const Decoder, zh: usize, zw: usize) u64 {
        const cfg = self.cfg;
        var lh: u64 = zh;
        var lw: u64 = zw;
        var c: u64 = cfg.innermost();
        var max_elems: u64 = 0;
        for (0..cfg.levels()) |i| {
            const level_idx = cfg.levels() - 1 - i;
            const out_ch: u64 = cfg.block_out_channels[level_idx];
            max_elems = @max(max_elems, lh * lw * @max(c, out_ch));
            c = out_ch;
            if (level_idx > 0) {
                lh *= 2;
                lw *= 2;
            }
        }
        max_elems = @max(max_elems, lh * lw * @max(c, 3));
        // q/k/v/out at latent resolution + the O(seq^2) mid-block scores plane
        // (f16 on the GPU paths, which query-tile it, so this term is generous).
        const seq: u64 = @as(u64, zh) * zw;
        const attn = 4 * seq * cfg.innermost() * 4 + seq * seq * 2;
        return 3 * max_elems * 4 + attn;
    }

    /// `z` is channel-last `[lat_h*lat_w][z_channels]`, already divided by
    /// `scaling_factor`. Returns channel-last `[8·lat_h * 8·lat_w][3]` in the
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
        try ops.conv.conv2d(io, gpa, pq, z, lat_h, lat_w, self.post_quant);

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
    // One head over all `ch` channels, scale 1/sqrt(ch) — LDM's `AttnBlock`.
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
        if (shape.len != 4 or shape[0] != co or shape[1] != ci or shape[2] != k or shape[3] != k) {
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
            .nin = if (in_ch != out_ch) try l.conv("{s}.nin_shortcut", .{base}, out_ch, in_ch, 1, 1, 0) else null,
            .in_ch = in_ch,
            .out_ch = out_ch,
        };
    }

    fn attn(l: Loader, comptime fmt: []const u8, args: anytype, ch: usize) !AttnBlock {
        var buf: [200]u8 = undefined;
        const base = try std.fmt.bufPrint(&buf, fmt, args);
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
    // ⚠️ The 8x8 case leaves every convolution in a single im2col band and the
    // mid-block attention at 64 positions. A 512x512 decode bands up to 512 ways and
    // attends over 4096 — a different code path in both ops, and the one every real
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
    // VAE. The scale factor is NOT exercised here — `decode` takes an already-divided
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
    // no error — a washed, low-contrast image. Asserting the rest of the config is
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

    // SD1.5, unit weight on channel 0 at pixel 0 → factors[0]; a large value at
    // pixel 1 → clamped white on R.
    var z15 = [_]f32{0} ** (latent_channels * plane);
    z15[0] = 1;
    z15[1] = 10;
    latentPreviewInto(&rgb, &z15, 1, plane, &latent_rgb_factors_sd15, latent_rgb_bias_sd15);
    try testing.expectEqualSlices(u8, &.{ 172, 156, 168 }, rgb[0..3]);
    try testing.expectEqual(@as(u8, 255), rgb[3]);

    // SDXL, an all-zero latent → the bias alone (SD1.5's would give a flat 127).
    const z_xl = [_]f32{0} ** (latent_channels * plane);
    latentPreviewInto(&rgb, &z_xl, 1, plane, &latent_rgb_factors_sdxl, latent_rgb_bias_sdxl);
    try testing.expectEqualSlices(u8, &.{ 141, 125, 127 }, rgb[0..3]);
    latentPreviewInto(&rgb, &z_xl, 1, plane, &latent_rgb_factors_sd15, latent_rgb_bias_sd15);
    try testing.expectEqualSlices(u8, &.{ 127, 127, 127 }, rgb[0..3]);
}
