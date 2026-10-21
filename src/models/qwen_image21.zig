//! Qwen-Image 2.1, a single-stream BLOCK-CAUSAL DiT, mirroring
//! comfy/ldm/qwen_image21/model.py and diffusers' `QwenImage21Transformer2DModel`.
//!
//! One sequence: the projected text with each reference image's latent tokens
//! spliced in at its slot, then the target image last. A token is one latent pixel
//! of the 64-channel latent (patch 1). Conventions, each a silent wrong answer:
//!
//!   - Block-causal attention: a text row sees every key up to itself, an image
//!     block's rows see every key up to the block's end. The target image, last,
//!     sees everything; nothing before it sees it.
//!   - Everything before the target (the PREFIX) is modulated at t = 0, the target
//!     at the sampled t: one shared modulation evaluated at both. The timestep is
//!     rounded to bf16 twice first (as 1000 t, then t), as the reference pipeline does.
//!   - The modulated norm is an unaffine LayerNorm times (1 + scale), NO shift; the
//!     gates are tanh'd. The final layer reads only the sampled-t row.
//!   - RoPE rotates ADJACENT pairs over three axes (16, 56, 56). Text token p sits at
//!     (p, p, p); an image block at (pos, row, col) with row and column centred on
//!     `[-ceil(n/2), floor(n/2))`, offset half a token where a reference's parity
//!     differs from the target's, and the next block starts `max(h, w)` later.
//!   - The text input norm is ZERO-CENTRED (scale 1 + w); the FFN's fused up
//!     projection is `[gate | up]`, SwiGLU.
//!
//! Loads from any `WeightStore`; the FFN is fused (`gate_up`) or split (`proj` the up
//! half, `gate_layer` the gate).

const std = @import("std");
const tp_core = @import("tp_core");
const weights_mod = tp_core.weights;
const dtypes = tp_core.dtype;
const ops = @import("tp_ops");
const quant_weight = @import("quant_weight.zig");
const lin = @import("lin.zig");
const lora_mod = @import("lora.zig");
const sd_vae = @import("sd_vae.zig");

const WeightStore = weights_mod.WeightStore;
const Weight = ops.matmul.Weight;

pub const tdim = 256;
pub const rope_theta: f64 = 10000.0;
const eps: f32 = 1e-6;
const max_blocks = 128;

/// The tensors ComfyUI's detection requires, all five.
pub const probes = [_][]const u8{ "txt_in.text_norm.weight", "modulation.1.weight", "transformer_blocks.0.attn.norm_q.weight", "img_in.weight", "proj_out.weight" };

/// The released checkpoint's shape, for a catalog with only a side file to read.
pub const features = 4096;
pub const n_blocks = 32;
pub const latent_channels = 64;
/// Qwen3-VL-8B's width.
pub const txt_dim = 4096;
/// ComfyUI's map for a fused checkpoint: a LoRA addresses the FFN's two halves.
pub const lora_keys_fused = [_]lora_mod.Rename{
    .{ .from = ".img_mlp.gate_layer", .to = ".img_mlp.gate_up", .slot = 0, .slots = 2 },
    .{ .from = ".img_mlp.proj", .to = ".img_mlp.gate_up", .slot = 1, .slots = 2 },
};

/// How a 2.1 LoRA names this model's tensors: ComfyUI's QwenImage map, which takes
/// diffusers' `transformer.` and bare spellings beside `diffusion_model.`. Unlike
/// ComfyUI, the FFN halves resolve under all three on a fused checkpoint (ComfyUI drops
/// them under `diffusion_model.`, warning that the keys did not load).
pub fn loraKeys(fused_mlp: bool) lora_mod.KeyMap {
    return .{ .renames = if (fused_mlp) &lora_keys_fused else &.{}, .prefixes = &.{ "transformer.", "" } };
}

/// The VAE's Wan 2.2 nested up stage, which a Wan 2.1 export lacks.
pub const vae_probe = "decoder.upsamples.0.upsamples.0.residual.0.gamma";

// The latent format (`comfy/latent_formats.py`, `QwenImage21`): the sampler works in
// `(z - mean) / std` of the VAE's 64 channels, and these are the preview factors.
pub const latents_mean = [latent_channels]f32{ 0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722, 1.8873, -1.7177, -0.6510, 0.2732, 0.7562, -0.6163, -1.0277, 3.8363, 2.0210, 0.0472, 0.9320, 2.0087, 2.4954, -0.1391, -1.4249, 1.8464, -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955, -0.0342, -1.0265, -0.7636, 3.0555, 0.0746, -3.0751, -0.1076, 1.7376, -1.0914, -1.9435, -0.2784, -1.3680, 0.4809, -0.4433, 0.3764, 0.5729, -2.0595, 1.0960, -1.3260, -2.0211, -5.0179, 0.5275, 4.0162, 1.8505, 0.3026, 1.9373, 1.4937, 0.2632, 0.5547, -1.7121, -0.1562, 0.0304 };
pub const latents_std = [latent_channels]f32{ 3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.7910, 3.0785, 3.6500, 3.9308, 3.0904, 2.8778, 3.7675, 3.7320, 5.0756, 3.2864, 4.0397, 3.1317, 4.0443, 2.9249, 3.9454, 3.0988, 4.2489, 3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.2830, 3.5694, 4.2467, 3.9037, 3.2947, 5.0770, 3.5075, 3.2700, 3.4767, 2.8063, 5.1125, 3.5327, 4.7833, 3.1286, 4.1819, 3.8527, 3.8312, 3.5605, 4.3875, 3.9624, 4.0168, 3.5643, 4.0550, 5.5614, 4.2963, 4.4080, 3.4959, 3.8747, 3.7608, 3.5735, 3.1490, 3.7662, 3.6746, 3.4563, 3.8161 };
pub const latent_rgb_factors = [latent_channels][3]f32{
    .{ -0.0158, -0.0115, -0.0174 },
    .{ 0.0030, 0.0120, 0.0027 },
    .{ 0.0637, 0.0470, -0.0127 },
    .{ 0.0360, 0.0661, -0.0030 },
    .{ 0.0159, 0.0181, 0.0082 },
    .{ 0.0132, 0.0326, 0.0169 },
    .{ 0.0191, 0.0261, 0.0136 },
    .{ -0.0146, -0.0276, -0.0361 },
    .{ 0.0187, -0.0024, -0.0072 },
    .{ -0.1059, -0.0090, 0.0350 },
    .{ -0.0195, -0.0226, -0.0138 },
    .{ -0.0295, 0.0024, -0.0215 },
    .{ 0.0191, -0.0393, -0.0001 },
    .{ -0.0144, -0.0166, -0.0272 },
    .{ 0.0389, 0.0430, 0.0445 },
    .{ -0.0153, -0.0336, 0.0031 },
    .{ 0.0339, 0.0122, 0.0220 },
    .{ -0.0136, -0.0078, -0.0120 },
    .{ -0.0340, -0.0282, -0.0245 },
    .{ -0.0133, -0.0176, -0.0133 },
    .{ 0.0109, -0.0087, 0.0096 },
    .{ -0.0010, 0.0044, 0.0016 },
    .{ 0.0301, 0.0053, 0.0361 },
    .{ -0.0281, -0.0205, -0.0032 },
    .{ -0.0725, 0.0002, 0.0160 },
    .{ -0.0036, 0.0158, 0.0807 },
    .{ 0.0087, 0.0040, -0.0053 },
    .{ -0.0260, 0.0183, -0.0077 },
    .{ -0.0039, -0.0035, -0.0107 },
    .{ -0.0026, 0.0172, 0.0237 },
    .{ 0.0088, 0.0078, 0.0078 },
    .{ -0.0087, -0.0310, -0.0122 },
    .{ -0.0027, 0.0018, 0.0094 },
    .{ -0.0064, 0.0292, -0.0256 },
    .{ 0.0594, 0.1049, 0.1180 },
    .{ 0.0103, -0.0103, -0.0026 },
    .{ -0.0091, 0.0025, -0.0015 },
    .{ 0.0178, 0.0243, 0.0292 },
    .{ -0.0063, -0.0012, 0.0202 },
    .{ 0.0452, 0.0246, 0.0143 },
    .{ 0.0149, 0.0270, 0.0052 },
    .{ 0.1484, 0.0801, 0.0804 },
    .{ -0.0120, 0.0040, 0.0010 },
    .{ 0.0181, 0.0051, -0.0021 },
    .{ 0.0132, 0.0050, 0.0019 },
    .{ 0.0291, 0.0020, 0.0092 },
    .{ 0.0066, -0.0410, -0.1314 },
    .{ -0.1153, -0.0629, -0.0802 },
    .{ 0.0258, 0.0378, 0.0298 },
    .{ 0.0375, 0.1139, 0.0468 },
    .{ -0.0142, -0.0126, -0.0276 },
    .{ 0.0339, 0.0153, 0.0138 },
    .{ 0.0346, 0.0211, 0.0267 },
    .{ 0.0369, -0.0431, -0.0993 },
    .{ -0.0052, -0.0092, 0.0056 },
    .{ -0.0279, 0.0410, -0.0357 },
    .{ 0.0036, 0.0017, -0.0083 },
    .{ -0.0441, -0.0367, -0.0454 },
    .{ -0.0001, -0.0092, -0.0001 },
    .{ -0.0222, -0.0183, -0.0051 },
    .{ 0.0039, 0.0053, -0.0184 },
    .{ -0.0094, -0.0075, -0.0143 },
    .{ -0.0066, -0.0088, -0.0063 },
    .{ 0.0220, 0.0074, 0.0100 },
};
pub const latent_rgb_bias = [3]f32{ -0.1228, -0.1869, -0.3083 };

/// The latent2rgb preview of the planar [64][zh*zw] sampler latent `z` into RGB8.
pub fn latentPreviewInto(rgb_out: []u8, z: []const f32, zh: usize, zw: usize) void {
    sd_vae.linearPreviewInto(latent_channels, rgb_out, z, zh, zw, &latent_rgb_factors, latent_rgb_bias);
}

pub const Config = struct {
    dim: usize,
    heads: usize,
    layers: usize,
    ffn: usize,
    txt_dim: usize,
    in_ch: usize,
    out_ch: usize,
    fused_mlp: bool,
    axes: [3]usize = .{ 16, 56, 56 },

    pub fn headDim(c: Config) usize {
        return c.dim / c.heads;
    }

    /// Read off the shapes, as ComfyUI's `detect_unet_config` does.
    pub fn detect(store: WeightStore) !Config {
        const pfx = prefixIn(store);
        var buf: [160]u8 = undefined;
        const nq = store.get(try name(&buf, pfx, "transformer_blocks.0.attn.norm_q.weight")) orelse return error.UnknownModelConfig;
        const hd = nq.info.shape.slice()[0];
        const img_in = store.get(try name(&buf, pfx, "img_in.weight")) orelse return error.UnknownModelConfig;
        const dim = img_in.info.shape.slice()[0];
        const in_ch = img_in.info.shape.slice()[1];
        const po = store.get(try name(&buf, pfx, "proj_out.weight")) orelse return error.UnknownModelConfig;
        const tn = store.get(try name(&buf, pfx, "txt_in.text_norm.weight")) orelse return error.UnknownModelConfig;
        var fused = true;
        const ffn = if (store.get(try name(&buf, pfx, "transformer_blocks.0.img_mlp.gate_up.weight"))) |v|
            v.info.shape.slice()[0] / 2
        else blk: {
            fused = false;
            const p = store.get(try name(&buf, pfx, "transformer_blocks.0.img_mlp.proj.weight")) orelse return error.UnknownModelConfig;
            break :blk p.info.shape.slice()[0];
        };
        var layers: usize = 0;
        while (layers < max_blocks) : (layers += 1) {
            const nm = std.mem.print(&buf, "{s}transformer_blocks.{d}.attn.norm_q.weight", .{ pfx, layers }) catch break;
            if (store.get(nm) == null) break;
        }
        if (hd == 0 or dim % hd != 0) return error.UnknownModelConfig;
        return .{
            .dim = dim,
            .heads = dim / hd,
            .layers = layers,
            .ffn = ffn,
            .txt_dim = tn.info.shape.slice()[0],
            .in_ch = in_ch,
            .out_ch = po.info.shape.slice()[0],
            .fused_mlp = fused,
        };
    }

    fn name(buf: []u8, pfx: []const u8, n: []const u8) ![]const u8 {
        return std.mem.print(buf, "{s}{s}", .{ pfx, n });
    }
};

pub fn prefixIn(store: WeightStore) []const u8 {
    return if (store.get("model.diffusion_model." ++ probes[3]) != null) "model.diffusion_model." else "";
}

pub const LinearW = struct {
    w: Weight,
    b: ?[]const f32 = null,
};

pub const Block = struct {
    q: LinearW,
    k: LinearW,
    v: LinearW,
    o: LinearW,
    q_norm: []const f32,
    k_norm: []const f32,
    /// `[2 * ffn][dim]`, rows `[gate | up]`, when fused; null split.
    gate_up: ?LinearW,
    gate: LinearW,
    up: LinearW,
    down: LinearW,
    /// q, k, v, o, gate, up, down: what a device forward runs, the fused FFN as row views.
    dev: [7]Weight,
};

/// One reference image: its planar latent and the text position it splices in at.
pub const Ref = struct {
    lat: []const f32,
    h: usize,
    w: usize,
    slot: usize,
};

/// The per-timestep vectors: each of the four `[dim]` rows at the sampled t and at 0.
pub const Mod = struct {
    /// `[2][4][dim]`: row 0 the sampled t, row 1 t = 0; scale1, gate1, scale2, gate2,
    /// the gates already tanh'd.
    rows: []f32,
    /// `[dim]`, the final layer's scale at the sampled t.
    head_scale: []f32,

    pub fn deinit(m: *Mod, gpa: std.mem.Allocator) void {
        gpa.free(m.rows);
        gpa.free(m.head_scale);
        m.* = undefined;
    }

    pub fn get(m: Mod, prefix: bool, i: usize) []const f32 {
        const d = m.head_scale.len;
        return m.rows[((if (prefix) @as(usize, 1) else 0) * 4 + i) * d ..][0..d];
    }
};

pub const DiT = struct {
    arena: std.heap.ArenaAllocator,
    cfg: Config,
    img_in: LinearW,
    txt_norm: []const f32,
    txt_in: LinearW,
    txt_out: LinearW,
    time0: LinearW,
    time2: LinearW,
    modulation: LinearW,
    blocks: []Block,
    head_mod: LinearW,
    proj_out: LinearW,
    device_lins: []const Weight,
    /// `img_in` and `proj_out`, the two linears a device step runs outside the blocks.
    edge_lins: []const Weight,
    lora: ?*const lora_mod.Stack = null,

    pub fn load(gpa: std.mem.Allocator, store: WeightStore) !DiT {
        return loadWith(gpa, store, try Config.detect(store));
    }

    pub fn loadWith(gpa: std.mem.Allocator, store: WeightStore, cfg: Config) !DiT {
        if (cfg.layers == 0 or cfg.dim % cfg.heads != 0) return error.UnknownModelConfig;
        if (cfg.axes[0] + cfg.axes[1] + cfg.axes[2] != cfg.headDim()) return error.UnknownModelConfig;
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();
        const l = Loader{ .store = store, .alloc = alloc, .pfx = prefixIn(store), .cfg = cfg };
        const d = cfg.dim;

        const blocks = try alloc.alloc(Block, cfg.layers);
        for (blocks, 0..) |*b, i| b.* = try l.block(i);
        const lins = try alloc.alloc(Weight, blocks.len * 7);
        for (blocks, 0..) |*b, i| lins[i * 7 ..][0..7].* = b.dev;

        // f32 at load, both edges: a device's dense 16-bit route stages its activation
        // at 16 bits, which at these two would be the walk's whole error.
        var img_in = try l.linear("img_in", .{}, d, cfg.in_ch);
        img_in.w = try ops.matmul.materializeF32(alloc, img_in.w);
        const txt_norm = blk: {
            const w = try l.vec("txt_in.text_norm.weight", .{}, cfg.txt_dim);
            for (w) |*v| v.* += 1;
            break :blk w;
        };
        const txt_in = try l.linear("txt_in.in_layer", .{}, d, cfg.txt_dim);
        const txt_out = try l.linear("txt_in.out_layer", .{}, d, d);
        const time0 = try l.linear("time_text_embed.timestep_embedder.linear_1", .{}, d, tdim);
        const time2 = try l.linear("time_text_embed.timestep_embedder.linear_2", .{}, d, d);
        const modulation = try l.linear("modulation.1", .{}, 4 * d, d);
        const head_mod = try l.linear("norm_out.linear", .{}, d, d);
        var proj_out = try l.linear("proj_out", .{}, cfg.out_ch, d);
        proj_out.w = try ops.matmul.materializeF32(alloc, proj_out.w);
        const edges = try alloc.dupe(Weight, &.{ img_in.w, proj_out.w });

        return .{
            .arena = arena,
            .cfg = cfg,
            .img_in = img_in,
            .txt_norm = txt_norm,
            .txt_in = txt_in,
            .txt_out = txt_out,
            .time0 = time0,
            .time2 = time2,
            .modulation = modulation,
            .blocks = blocks,
            .head_mod = head_mod,
            .proj_out = proj_out,
            .device_lins = lins,
            .edge_lins = edges,
        };
    }

    pub fn deinit(self: *DiT) void {
        self.arena.deinit();
        self.* = undefined;
    }

    pub fn attachLora(self: *DiT, gpa: std.mem.Allocator, stack: *lora_mod.Stack) !usize {
        self.lora = stack;
        stack.beginAttach();
        try stack.resolveAll(self.device_lins);
        for (self.blocks) |*b| if (b.gate_up) |f| {
            _ = try stack.resolve(f.w);
        };
        try stack.resolveAll(&.{
            self.img_in.w, self.txt_in.w,     self.txt_out.w,  self.time0.w,
            self.time2.w,  self.modulation.w, self.head_mod.w, self.proj_out.w,
        });
        return stack.finishAttach(gpa);
    }

    /// The timestep vector at `t` (already rounded), `linear_2(silu(linear_1(sinusoid)))`.
    fn temb(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, t: f32) ![]f32 {
        var freq: [tdim]f32 = undefined;
        ops.rope.timestepEmbedding(&freq, t, 10000.0);
        const d = self.cfg.dim;
        const t0 = try gpa.alloc(f32, d);
        defer gpa.free(t0);
        try linear(io, gpa, self.lora, t0, &freq, 1, self.time0);
        ops.act.silu(t0);
        const out = try gpa.alloc(f32, d);
        errdefer gpa.free(out);
        try linear(io, gpa, self.lora, out, t0, 1, self.time2);
        return out;
    }

    /// The modulation at the sampled sigma and at 0, and the head's scale.
    pub fn modulationAt(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, sigma: f32) !Mod {
        const d = self.cfg.dim;
        const rows = try gpa.alloc(f32, 8 * d);
        errdefer gpa.free(rows);
        const head_scale = try gpa.alloc(f32, d);
        errdefer gpa.free(head_scale);
        for ([2]f32{ roundTimestep(sigma), 0 }, 0..) |t, r| {
            const te = try self.temb(io, gpa, t);
            defer gpa.free(te);
            ops.act.silu(te);
            const m = rows[r * 4 * d ..][0 .. 4 * d];
            try linear(io, gpa, self.lora, m, te, 1, self.modulation);
            for (m[d .. 2 * d]) |*v| v.* = std.math.tanh(v.*);
            for (m[3 * d .. 4 * d]) |*v| v.* = std.math.tanh(v.*);
            if (r == 0) try linear(io, gpa, self.lora, head_scale, te, 1, self.head_mod);
        }
        return .{ .rows = rows, .head_scale = head_scale };
    }

    /// The text through `txt_in`: `[n][dim]`, caller frees.
    pub fn textTokens(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, ctx: []const f32, n: usize) ![]f32 {
        const d = self.cfg.dim;
        const normed = try gpa.alloc(f32, ctx.len);
        defer gpa.free(normed);
        ops.norm.rmsNorm(normed, ctx, self.txt_norm, eps);
        const mid = try gpa.alloc(f32, n * d);
        defer gpa.free(mid);
        try linear(io, gpa, self.lora, mid, normed, n, self.txt_in);
        ops.act.geluTanh(mid);
        const out = try gpa.alloc(f32, n * d);
        errdefer gpa.free(out);
        try linear(io, gpa, self.lora, out, mid, n, self.txt_out);
        return out;
    }

    /// Velocity for one latent: planar `[in_ch][lat_h][lat_w]` in, `[out_ch][..]` out.
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
        n_txt: usize,
        refs: []const Ref,
        cancel: ?*std.atomic.Value(bool),
    ) !void {
        const cfg = self.cfg;
        const d = cfg.dim;
        const n_img = lat_h * lat_w;
        std.debug.assert(x_lat.len == cfg.in_ch * n_img and out.len == cfg.out_ch * n_img);
        std.debug.assert(ctx.len == n_txt * cfg.txt_dim);

        const prev_tok = ops.cancel.token;
        ops.cancel.token = cancel;
        defer ops.cancel.token = prev_tok;

        var lay = try Layout.build(gpa, cfg, lat_h, lat_w, n_txt, refs);
        defer lay.deinit(gpa);
        const seq = lay.seq;
        const prefix = seq - n_img;

        const x = try gpa.alloc(f32, seq * d);
        defer gpa.free(x);
        {
            const txt = try self.textTokens(io, gpa, ctx, n_txt);
            defer gpa.free(txt);
            for (lay.parts) |p| switch (p.kind) {
                .text => @memcpy(x[p.row * d ..][0 .. p.n * d], txt[p.src * d ..][0 .. p.n * d]),
                .image => {
                    const lat = if (p.src < refs.len) refs[p.src].lat else x_lat;
                    const tokens = try gpa.alloc(f32, p.n * cfg.in_ch);
                    defer gpa.free(tokens);
                    interleaveChannels(tokens, lat, p.n, cfg.in_ch);
                    try linear(io, gpa, self.lora, x[p.row * d ..][0 .. p.n * d], tokens, p.n, self.img_in);
                },
            };
        }

        var mod = try self.modulationAt(io, gpa, sigma);
        defer mod.deinit(gpa);
        var freqs = try ops.rope.fluxFreqs(gpa, lay.pos, &cfg.axes, rope_theta);
        defer freqs.deinit(gpa);

        for (self.blocks) |*blk| {
            if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
            try blockForward(io, gpa, self.lora, cfg, blk, x, seq, prefix, mod, freqs, lay.kv_end);
        }

        const img = x[prefix * d ..][0 .. n_img * d];
        ops.norm.layerNormUnit(img, img, d, eps);
        modulate(img, mod.head_scale);
        const fin = try gpa.alloc(f32, n_img * cfg.out_ch);
        defer gpa.free(fin);
        try linear(io, gpa, self.lora, fin, img, n_img, self.proj_out);
        deinterleaveChannels(out, fin, n_img, cfg.out_ch);
    }
};

/// The reference pipeline's timestep: `((t * 1000).to(bf16) / 1000).to(bf16)`.
pub fn roundTimestep(sigma: f32) f32 {
    const bf = struct {
        fn f(v: f32) f32 {
            return dtypes.bf16ToF32(dtypes.f32ToBf16(v));
        }
    }.f;
    return bf(bf(sigma * 1000.0) / 1000.0);
}

/// The joint sequence's blocks, positions and per-row key bounds.
pub const Layout = struct {
    pub const Part = struct { kind: enum { text, image }, row: usize, n: usize, src: usize };

    parts: []Part,
    /// `[seq][3]`.
    pos: []f32,
    /// Per row, the exclusive end of the keys it attends.
    kv_end: []u32,
    seq: usize,

    pub fn build(gpa: std.mem.Allocator, cfg: Config, lat_h: usize, lat_w: usize, n_txt: usize, refs: []const Ref) !Layout {
        _ = cfg;
        var parts: std.ArrayList(Part) = .empty;
        errdefer parts.deinit(gpa);
        var seq: usize = 0;
        var t0: usize = 0;
        for (0..refs.len + 1) |k| {
            const end = if (k < refs.len) @min(refs[k].slot, n_txt) else n_txt;
            const t1 = @max(t0, end);
            if (t1 > t0) {
                try parts.append(gpa, .{ .kind = .text, .row = seq, .n = t1 - t0, .src = t0 });
                seq += t1 - t0;
                t0 = t1;
            }
            const n = if (k < refs.len) refs[k].h * refs[k].w else lat_h * lat_w;
            try parts.append(gpa, .{ .kind = .image, .row = seq, .n = n, .src = k });
            seq += n;
        }

        const pos = try gpa.alloc(f32, seq * 3);
        errdefer gpa.free(pos);
        const kv_end = try gpa.alloc(u32, seq);
        errdefer gpa.free(kv_end);
        var p: f32 = 0;
        for (parts.items) |part| switch (part.kind) {
            .text => for (0..part.n) |i| {
                const r = part.row + i;
                pos[r * 3 ..][0..3].* = @splat(p);
                p += 1;
                kv_end[r] = @intCast(r + 1);
            },
            .image => {
                const h = if (part.src < refs.len) refs[part.src].h else lat_h;
                const w = if (part.src < refs.len) refs[part.src].w else lat_w;
                const oh = centre(h, lat_h);
                const ow = centre(w, lat_w);
                for (0..h) |ri| for (0..w) |ci| {
                    const r = part.row + ri * w + ci;
                    pos[r * 3 ..][0..3].* = .{ p, @as(f32, @floatFromInt(ri)) - oh, @as(f32, @floatFromInt(ci)) - ow };
                    kv_end[r] = @intCast(part.row + part.n);
                };
                p += @floatFromInt(@max(h, w));
            },
        };
        return .{ .parts = try parts.toOwnedSlice(gpa), .pos = pos, .kv_end = kv_end, .seq = seq };
    }

    pub fn deinit(self: *Layout, gpa: std.mem.Allocator) void {
        gpa.free(self.parts);
        gpa.free(self.pos);
        gpa.free(self.kv_end);
        self.* = undefined;
    }
};

/// `n - n/2`, less half a token where `n`'s parity differs from the target's.
fn centre(n: usize, target: usize) f32 {
    return @as(f32, @floatFromInt(n - n / 2)) - 0.5 * (@as(f32, @floatFromInt(n % 2)) - @as(f32, @floatFromInt(target % 2)));
}

fn blockForward(
    io: std.Io,
    gpa: std.mem.Allocator,
    stack: ?*const lora_mod.Stack,
    cfg: Config,
    blk: *const Block,
    x: []f32,
    seq: usize,
    prefix: usize,
    mod: Mod,
    freqs: ops.rope.Freqs,
    kv_end: []const u32,
) !void {
    const d = cfg.dim;
    const hd = cfg.headDim();
    const y = try gpa.alloc(f32, seq * d);
    defer gpa.free(y);
    const q = try gpa.alloc(f32, seq * d);
    defer gpa.free(q);
    const k = try gpa.alloc(f32, seq * d);
    defer gpa.free(k);
    const v = try gpa.alloc(f32, seq * d);
    defer gpa.free(v);

    modNorm(y, x, d, prefix, mod, 0);
    try linear(io, gpa, stack, q, y, seq, blk.q);
    try linear(io, gpa, stack, k, y, seq, blk.k);
    try linear(io, gpa, stack, v, y, seq, blk.v);
    ops.norm.rmsNorm(q, q, blk.q_norm, eps);
    ops.norm.rmsNorm(k, k, blk.k_norm, eps);
    ops.rope.applyInterleaved(q, freqs, seq, cfg.heads, hd);
    ops.rope.applyInterleaved(k, freqs, seq, cfg.heads, hd);
    try ops.attention.attention(io, gpa, y, q, k, v, .{
        .seq_q = seq,
        .seq_kv = seq,
        .n_heads = cfg.heads,
        .n_kv_heads = cfg.heads,
        .head_dim = hd,
        .kv_end = kv_end,
    });
    try linear(io, gpa, stack, q, y, seq, blk.o);
    gatedAdd(x, q, d, prefix, mod, 1);

    modNorm(y, x, d, prefix, mod, 2);
    const gate = try gpa.alloc(f32, seq * cfg.ffn);
    defer gpa.free(gate);
    const up = try gpa.alloc(f32, seq * cfg.ffn);
    defer gpa.free(up);
    if (blk.gate_up) |f| {
        const both = try gpa.alloc(f32, seq * 2 * cfg.ffn);
        defer gpa.free(both);
        try linear(io, gpa, stack, both, y, seq, f);
        for (0..seq) |r| {
            const row = both[r * 2 * cfg.ffn ..][0 .. 2 * cfg.ffn];
            @memcpy(gate[r * cfg.ffn ..][0..cfg.ffn], row[0..cfg.ffn]);
            @memcpy(up[r * cfg.ffn ..][0..cfg.ffn], row[cfg.ffn..]);
        }
    } else {
        try linear(io, gpa, stack, gate, y, seq, blk.gate);
        try linear(io, gpa, stack, up, y, seq, blk.up);
    }
    ops.act.siluMul(gate, up);
    try linear(io, gpa, stack, q, gate, seq, blk.down);
    gatedAdd(x, q, d, prefix, mod, 3);
}

/// `LayerNorm(x) * (1 + scale)`, the prefix rows at t = 0 and the rest at t.
fn modNorm(out: []f32, x: []const f32, d: usize, prefix: usize, mod: Mod, i: usize) void {
    ops.norm.layerNormUnit(out, x, d, eps);
    modulate(out[0 .. prefix * d], mod.get(true, i));
    modulate(out[prefix * d ..], mod.get(false, i));
}

fn modulate(x: []f32, scale: []const f32) void {
    const d = scale.len;
    var r: usize = 0;
    while (r < x.len) : (r += d) {
        for (x[r..][0..d], scale) |*v, s| v.* *= 1.0 + s;
    }
}

fn gatedAdd(x: []f32, delta: []const f32, d: usize, prefix: usize, mod: Mod, i: usize) void {
    for (0..x.len / d) |r| {
        const g = mod.get(r < prefix, i);
        for (x[r * d ..][0..d], delta[r * d ..][0..d], g) |*v, dl, gv| v.* += gv * dl;
    }
}

pub fn interleaveChannels(dst: []f32, src: []const f32, n: usize, c: usize) void {
    for (0..c) |ci| for (src[ci * n ..][0..n], 0..) |v, i| {
        dst[i * c + ci] = v;
    };
}

pub fn deinterleaveChannels(dst: []f32, src: []const f32, n: usize, c: usize) void {
    for (0..c) |ci| for (dst[ci * n ..][0..n], 0..) |*v, i| {
        v.* = src[i * c + ci];
    };
}

fn linear(io: std.Io, gpa: std.mem.Allocator, stack: ?*const lora_mod.Stack, out: []f32, x: []const f32, m: usize, lw: LinearW) !void {
    try ops.matmul.matmul(io, gpa, out, x, m, lw.w, lw.b);
    if (stack) |st| try st.applyHost(io, gpa, out, lw.w.rows, x, m, lw.w, lw.b);
}

const Loader = struct {
    store: WeightStore,
    alloc: std.mem.Allocator,
    pfx: []const u8,
    cfg: Config,

    fn full(l: Loader, buf: []u8, comptime f: []const u8, args: anytype) ![]u8 {
        var w = std.Io.Writer.fixed(buf);
        try w.writeAll(l.pfx);
        try w.print(f, args);
        return w.buffered();
    }

    fn vec(l: Loader, comptime f: []const u8, args: anytype, len: usize) ![]f32 {
        var buf: [192]u8 = undefined;
        const nm = try l.full(&buf, f, args);
        const view = l.store.get(nm) orelse {
            std.log.err("qwen-image-2.1: missing tensor {s}", .{nm});
            return error.MissingTensor;
        };
        if (view.info.elemCount() != len) return error.ShapeMismatch;
        return view.toF32Alloc(l.alloc);
    }

    fn linear(l: Loader, comptime f: []const u8, args: anytype, rows: usize, cols: usize) !LinearW {
        var buf: [192]u8 = undefined;
        const nm = try l.full(&buf, f ++ ".weight", args);
        return .{ .w = try quant_weight.load(l.alloc, l.store, nm, rows, cols, .{ .who = "qwen-image-2.1" }) };
    }

    fn block(l: Loader, i: usize) !Block {
        const d = l.cfg.dim;
        const ffn = l.cfg.ffn;
        const hd = l.cfg.headDim();
        const p = "transformer_blocks.{d}.";
        var b: Block = .{
            .q = try l.linear(p ++ "attn.to_q", .{i}, d, d),
            .k = try l.linear(p ++ "attn.to_k", .{i}, d, d),
            .v = try l.linear(p ++ "attn.to_v", .{i}, d, d),
            .o = try l.linear(p ++ "attn.to_out.0", .{i}, d, d),
            .q_norm = try l.vec(p ++ "attn.norm_q.weight", .{i}, hd),
            .k_norm = try l.vec(p ++ "attn.norm_k.weight", .{i}, hd),
            .gate_up = null,
            .gate = undefined,
            .up = undefined,
            .down = try l.linear(p ++ "img_mlp.out", .{i}, d, ffn),
            .dev = undefined,
        };
        if (l.cfg.fused_mlp) {
            const f = try l.linear(p ++ "img_mlp.gate_up", .{i}, 2 * ffn, d);
            b.gate_up = f;
            b.gate = .{ .w = try lin.rowSlice(l.alloc, f.w, 0, ffn) };
            b.up = .{ .w = try lin.rowSlice(l.alloc, f.w, ffn, ffn) };
        } else {
            b.gate = try l.linear(p ++ "img_mlp.gate_layer", .{i}, ffn, d);
            b.up = try l.linear(p ++ "img_mlp.proj", .{i}, ffn, d);
        }
        b.dev = .{ b.q.w, b.k.w, b.v.w, b.o.w, b.gate.w, b.up.w, b.down.w };
        return b;
    }
};

// --- tests -----------------------------------------------------------------

const testing = std.testing;
const safetensors = tp_core.safetensors;
const test_gate = @import("../test_gate.zig");
const build_options = @import("build_options");
const wan_vae = @import("wan_vae.zig");

const forward_ref = "src/models/assets/qwen_image21_forward.safetensors";
const ckpt = build_options.model_root ++ "/diffusion_models/qwen2.1/qwen_image_2.1_bf16.safetensors";

fn relL2(want: []const f32, got: []const f32) f64 {
    var num: f64 = 0;
    var den: f64 = 0;
    for (want, got) |e, a| {
        num += @as(f64, e - a) * (e - a);
        den += @as(f64, e) * e;
    }
    return @sqrt(num / den);
}

fn checkToy(prefix: []const u8, want_fused: bool) !void {
    const gpa = testing.allocator;
    const io = testing.io;
    var ref = try safetensors.SafeTensors.open(gpa, io, forward_ref);
    defer ref.deinit();
    var pre = try weights_mod.Prefixed.init(gpa, .{ .safetensors = &ref }, prefix);
    defer pre.deinit(gpa);
    var cfg = try Config.detect(pre.store());
    try testing.expectEqual(want_fused, cfg.fused_mlp);
    try testing.expectEqual(@as(usize, 2), cfg.heads);
    cfg.axes = .{ 8, 28, 28 };
    var model = try DiT.loadWith(gpa, pre.store(), cfg);
    defer model.deinit();

    for ([_][]const u8{ "even", "odd", "ref" }) |case| {
        var kb: [32]u8 = undefined;
        const xv = try ref.require(try std.mem.print(&kb, "{s}.x", .{case}));
        const x = try xv.toF32Alloc(gpa);
        defer gpa.free(x);
        const ctx = try (try ref.require(try std.mem.print(&kb, "{s}.ctx", .{case}))).toF32Alloc(gpa);
        defer gpa.free(ctx);
        const sig = try (try ref.require(try std.mem.print(&kb, "{s}.sigma", .{case}))).toF32Alloc(gpa);
        defer gpa.free(sig);
        const want = try (try ref.require(try std.mem.print(&kb, "{s}.out", .{case}))).toF32Alloc(gpa);
        defer gpa.free(want);

        var refs: [1]Ref = undefined;
        var n_refs: usize = 0;
        var ref_lat: ?[]f32 = null;
        defer if (ref_lat) |r| gpa.free(r);
        if (ref.get(try std.mem.print(&kb, "{s}.ref0", .{case}))) |rv| {
            const rs = rv.info.shape.slice();
            ref_lat = try rv.toF32Alloc(gpa);
            const slot = std.mem.bytesAsSlice(i32, (try ref.require(try std.mem.print(&kb, "{s}.slot0", .{case}))).bytes)[0];
            refs[0] = .{ .lat = ref_lat.?, .h = rs[1], .w = rs[2], .slot = @intCast(slot) };
            n_refs = 1;
        }

        const shape = xv.info.shape.slice();
        const got = try gpa.alloc(f32, x.len);
        defer gpa.free(got);
        try model.forward(io, gpa, got, x, shape[1], shape[2], sig[0], ctx, ctx.len / cfg.txt_dim, refs[0..n_refs], null);
        const rel = relL2(want, got);
        errdefer std.debug.print("{s} {s}: rel L2 {e:.3}\n", .{ prefix, case, rel });
        try testing.expect(rel < 1e-5);
    }
}

test "the Qwen-Image 2.1 DiT matches ComfyUI at a toy width, fused FFN" {
    try checkToy("fused.", true);
}

test "the Qwen-Image 2.1 DiT matches ComfyUI at a toy width, split FFN" {
    try checkToy("split.", false);
}

test "the timestep is rounded through bf16 as 1000 t, then as t" {
    // 0.25 survives both roundings; 0.3 does not (300 is not a bf16 value).
    try testing.expectEqual(@as(f32, 0.25), roundTimestep(0.25));
    try testing.expect(roundTimestep(0.3) != 0.3);
    try testing.expectEqual(dtypes.bf16ToF32(dtypes.f32ToBf16(dtypes.bf16ToF32(dtypes.f32ToBf16(300.0)) / 1000.0)), roundTimestep(0.3));
}

test "the real Qwen-Image 2.1 checkpoint's config" {
    const io = testing.io;
    try test_gate.requireModelFile(io, ckpt);
    var ck = try safetensors.SafeTensors.open(testing.allocator, io, ckpt);
    defer ck.deinit();
    const cfg = try Config.detect(.{ .safetensors = &ck });
    try testing.expectEqual(Config{ .dim = 4096, .heads = 32, .layers = 32, .ffn = 12288, .txt_dim = 4096, .in_ch = 64, .out_ch = 64, .fused_mlp = true }, cfg);
}

const vae_ref = "src/models/assets/qwen_image21_vae.safetensors";
const vae_ckpt = build_options.model_root ++ "/vae/qwen_image_2.1_vae_bf16.safetensors";

test "the Qwen-Image 2.1 latent format is ComfyUI's" {
    const gpa = testing.allocator;
    var ref = try safetensors.SafeTensors.open(gpa, testing.io, vae_ref);
    defer ref.deinit();
    const Want = struct { key: []const u8, got: []const f32 };
    for ([_]Want{
        .{ .key = "mean", .got = &latents_mean },
        .{ .key = "std", .got = &latents_std },
        .{ .key = "rgb_factors", .got = @as(*const [latent_channels * 3]f32, @ptrCast(&latent_rgb_factors)) },
        .{ .key = "rgb_bias", .got = &latent_rgb_bias },
    }) |c| {
        const want = try (try ref.require(c.key)).toF32Alloc(gpa);
        defer gpa.free(want);
        errdefer std.debug.print("{s} differs\n", .{c.key});
        try testing.expectEqualSlices(f32, want, c.got);
    }
}

test "the Qwen-Image 2.1 VAE decodes RGBA like ComfyUI on the real checkpoint" {
    const gpa = testing.allocator;
    const io = testing.io;
    try test_gate.requireModelFile(io, vae_ckpt);
    var ref = try safetensors.SafeTensors.open(gpa, io, vae_ref);
    defer ref.deinit();
    var ck = try safetensors.SafeTensors.open(gpa, io, vae_ckpt);
    defer ck.deinit();
    var dec = try wan_vae.Decoder.load(gpa, .{ .safetensors = &ck });
    defer dec.deinit();
    try testing.expectEqual(@as(usize, latent_channels), dec.latentChannels());
    try testing.expectEqual(@as(usize, 4), dec.outChannels());
    try testing.expectEqual(@as(usize, 16), dec.scale());

    const zv = try ref.require("z_norm");
    const zs = zv.info.shape.slice();
    const z = try zv.toF32Alloc(gpa);
    defer gpa.free(z);
    const plane = zs[1] * zs[2];
    for (0..latent_channels) |c| for (z[c * plane ..][0..plane]) |*v| {
        v.* = v.* * latents_std[c] + latents_mean[c];
    };
    const z_want = try (try ref.require("z")).toF32Alloc(gpa);
    defer gpa.free(z_want);
    try testing.expect(relL2(z_want, z) < 1e-6);

    const want = try (try ref.require("out")).toF32Alloc(gpa);
    defer gpa.free(want);
    const got = try dec.decode(io, gpa, z, zs[1], zs[2], null);
    defer gpa.free(got);
    try testing.expectEqual(want.len, got.len);
    const rel = relL2(want, got);
    const alpha = relL2(want[3 * want.len / 4 ..], got[3 * got.len / 4 ..]);
    errdefer std.debug.print("vae: rel L2 {e:.3}, alpha plane {e:.3}\n", .{ rel, alpha });
    try testing.expect(rel < 1e-4);
    try testing.expect(alpha < 1e-4);
}

test "the real Qwen-Image 2.1 DiT matches ComfyUI on one forward" {
    const gpa = testing.allocator;
    const io = testing.io;
    const turbo = build_options.model_root ++ "/diffusion_models/qwen2.1/qwen_image_2.1_turbo_bf16.safetensors";
    try test_gate.requireModelFile(io, turbo);
    var ref = try safetensors.SafeTensors.open(gpa, io, "src/models/assets/qwen_image21_real.safetensors");
    defer ref.deinit();
    var te = try safetensors.SafeTensors.open(gpa, io, "src/models/assets/qwen_image21_te_ref.safetensors");
    defer te.deinit();
    var ck = try safetensors.SafeTensors.open(gpa, io, turbo);
    defer ck.deinit();
    var model = try DiT.load(gpa, .{ .safetensors = &ck });
    defer model.deinit();

    const xv = try ref.require("x");
    const xs = xv.info.shape.slice();
    const x = try xv.toF32Alloc(gpa);
    defer gpa.free(x);
    const ctx = try (try te.require("cond.0")).toF32Alloc(gpa);
    defer gpa.free(ctx);
    const sig = try (try ref.require("sigma")).toF32Alloc(gpa);
    defer gpa.free(sig);
    const want = try (try ref.require("out")).toF32Alloc(gpa);
    defer gpa.free(want);

    const got = try gpa.alloc(f32, x.len);
    defer gpa.free(got);
    try model.forward(io, gpa, got, x, xs[1], xs[2], sig[0], ctx, ctx.len / txt_dim, &.{}, null);
    const rel = relL2(want, got);
    errdefer std.debug.print("real forward: rel L2 {e:.3}\n", .{rel});
    try testing.expect(rel < 1e-4);
}

test "a LoRA on Qwen-Image 2.1 matches ComfyUI's, fused and split, under both diffusers spellings" {
    const gpa = testing.allocator;
    const io = testing.io;
    var ref = try safetensors.SafeTensors.open(gpa, io, "src/models/assets/qwen_image21_lora.safetensors");
    defer ref.deinit();
    const xv = try ref.require("x");
    const xs = xv.info.shape.slice();
    const x = try xv.toF32Alloc(gpa);
    defer gpa.free(x);
    const ctx = try (try ref.require("ctx")).toF32Alloc(gpa);
    defer gpa.free(ctx);
    const sig = try (try ref.require("sigma")).toF32Alloc(gpa);
    defer gpa.free(sig);
    const strength = (try (try ref.require("strength")).toF32Alloc(gpa));
    defer gpa.free(strength);
    const want_base = try (try ref.require("out_base")).toF32Alloc(gpa);
    defer gpa.free(want_base);
    const want_lora = try (try ref.require("out_lora")).toF32Alloc(gpa);
    defer gpa.free(want_lora);
    const got = try gpa.alloc(f32, x.len);
    defer gpa.free(got);

    for ([_][]const u8{ "fused.", "split." }) |mp| for ([_][]const u8{ "lora_bare.", "lora_tf." }) |lp| {
        var pre = try weights_mod.Prefixed.init(gpa, .{ .safetensors = &ref }, mp);
        defer pre.deinit(gpa);
        var model = try DiT.load(gpa, pre.store());
        defer model.deinit();
        try model.forward(io, gpa, got, x, xs[1], xs[2], sig[0], ctx, ctx.len / model.cfg.txt_dim, &.{}, null);
        const base_rel = relL2(want_base, got);
        errdefer std.debug.print("{s}: without the LoRA, rel L2 {e:.3}\n", .{ mp, base_rel });
        try testing.expect(base_rel < 1e-5);

        var lv = try weights_mod.Prefixed.init(gpa, .{ .safetensors = &ref }, lp);
        defer lv.deinit(gpa);
        var stack: lora_mod.Stack = .{};
        defer stack.deinit(gpa);
        _ = try stack.add(gpa, lv.store(), strength[0], "fixture", loraKeys(model.cfg.fused_mlp));
        const applied = try model.attachLora(gpa, &stack);
        errdefer std.debug.print("{s} {s}: {d} targets attached\n", .{ mp, lp, applied });
        try testing.expectEqual(@as(usize, 6), applied);
        try model.forward(io, gpa, got, x, xs[1], xs[2], sig[0], ctx, ctx.len / model.cfg.txt_dim, &.{}, null);
        const rel = relL2(want_lora, got);
        errdefer std.debug.print("{s} {s}: with the LoRA, rel L2 {e:.3}\n", .{ mp, lp, rel });
        try testing.expect(rel < 1e-4);
    };
}
