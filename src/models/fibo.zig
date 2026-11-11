//! Fibo 1.5 (Bria, "FIBO 1.x"), an 8B Flux-style MMDiT on the Wan 2.2 VAE's latent (48
//! channels at 16x, patch 1), conditioned on SmolLM3-3B. Reference: diffusers'
//! `BriaFiboTransformer2DModel`, which the ComfyUI_Fibo-1.5 custom nodes run; ComfyUI
//! itself has no model for it.
//!
//! Eight double-stream blocks then 38 single-stream ones, Flux's shapes throughout:
//! LayerNorm (no affine) with AdaLN-Zero shift/scale/gate, per-head RMS QK norms, tanh
//! GELU, `[text | image]` joint attention with text at rope position zero and the image
//! on (0, h, w), interleaved rope at theta 10000 over axes 16/56/56. What is Fibo's own:
//!
//! - Before EVERY block, double and single, the text stream's second half of channels is
//!   replaced by that block's own projection of one encoder hidden state
//!   (`caption_projection`); the first half carries on.
//! - The text stream starts as `context_embedder` of the encoder's last two hidden states
//!   concatenated.
//! - The output norm's modulation is (scale, shift), the blocks' (shift, scale, gate).
//! - The timestep is `sigma * 1000`, cos first.
//!
//! Loads diffusers' names (ggufy's "fibo" GGUFs keep them). Pinned by
//! tools/gen_fibo_forward.py.

const std = @import("std");
const tp_core = @import("tp_core");
const safetensors = tp_core.safetensors;
const weights_mod = tp_core.weights;
const ops = @import("tp_ops");
const quant_weight = @import("quant_weight.zig");
const lora_mod = @import("lora.zig");

const WeightStore = weights_mod.WeightStore;
pub const Weight = ops.matmul.Weight;

/// Tensors only this architecture has (ggufy's detection set).
pub const probes = [_][]const u8{ "caption_projection.0.linear.weight", "transformer_blocks.0.attn.norm_added_k.weight", "single_transformer_blocks.0.proj_mlp.weight" };

/// The Wan 2.2 VAE's latent: 48 channels at 16x.
pub const channels = 48;
pub const spatial_scale = 16;

/// ComfyUI's `latent_formats.Wan22`, which the VAE's config carries too: the sampler
/// works in `(z - mean) / std`, the decoder takes `z * std + mean`.
pub const latents_mean = [channels]f32{ -0.2289000004529953, -0.005200000014156103, -0.13230000436306, -0.23389999568462372, -0.2799000144004822, 0.017400000244379044, 0.18379999697208405, 0.15569999814033508, -0.13819999992847443, 0.05420000106096268, 0.28130000829696655, 0.08910000324249268, 0.15700000524520874, -0.009800000116229057, 0.03750000149011612, -0.18250000476837158, -0.22460000216960907, -0.12070000171661377, -0.0697999969124794, 0.5109000205993652, 0.26649999618530273, -0.21080000698566437, -0.21580000221729279, 0.2502000033855438, -0.20550000667572021, -0.03220000118017197, 0.11089999973773956, 0.156700000166893, -0.07289999723434448, 0.08990000188350677, -0.2799000144004822, -0.12300000339746475, -0.031300000846385956, -0.164900004863739, 0.011699999682605267, 0.0723000019788742, -0.2838999927043915, -0.20829999446868896, -0.052000001072883606, 0.3747999966144562, 0.015200000256299973, 0.195700004696846, 0.14329999685287476, -0.29440000653266907, 0.3573000133037567, -0.05480000004172325, -0.1680999994277954, -0.06669999659061432 };
pub const latents_std = [channels]f32{ 0.476500004529953, 1.0363999605178833, 0.4514000117778778, 1.1677000522613525, 0.5313000082969666, 0.49900001287460327, 0.48179998993873596, 0.5012999773025513, 0.8158000111579895, 1.0343999862670898, 0.5893999934196472, 1.0901000499725342, 0.6884999871253967, 0.6165000200271606, 0.8453999757766724, 0.49779999256134033, 0.5759000182151794, 0.3522999882698059, 0.7135000228881836, 0.680400013923645, 0.583299994468689, 1.4146000146865845, 0.8985999822616577, 0.5659000277519226, 0.7069000005722046, 0.5338000059127808, 0.48890000581741333, 0.4916999936103821, 0.40689998865127563, 0.4999000132083893, 0.6866000294685364, 0.4092999994754791, 0.570900022983551, 0.6065000295639038, 0.6414999961853027, 0.4943999946117401, 0.5726000070571899, 1.204200029373169, 0.545799970626831, 1.6886999607086182, 0.3971000015735626, 1.059999942779541, 0.39430001378059387, 0.5536999702453613, 0.5443999767303467, 0.4088999927043915, 0.7468000054359436, 0.774399995803833 };

/// `latent_formats.Wan22.latent_rgb_factors`, for the live preview.
pub const latent_rgb_factors = [channels][3]f32{
    .{ 0.0119, 0.0103, 0.0046 },
    .{ -0.1062, -0.0504, 0.0165 },
    .{ 0.014, 0.0409, 0.0491 },
    .{ -0.0813, -0.0677, 0.0607 },
    .{ 0.0656, 0.0851, 0.0808 },
    .{ 0.0264, 0.0463, 0.0912 },
    .{ 0.0295, 0.0326, 0.059 },
    .{ -0.0244, -0.027, 0.0025 },
    .{ 0.0443, -0.0102, 0.0288 },
    .{ -0.0465, -0.009, -0.0205 },
    .{ 0.0359, 0.0236, 0.0082 },
    .{ -0.0776, 0.0854, 0.1048 },
    .{ 0.0564, 0.0264, 0.0561 },
    .{ 0.0006, 0.0594, 0.0418 },
    .{ -0.0319, -0.0542, -0.0637 },
    .{ -0.0268, 0.0024, 0.026 },
    .{ 0.0539, 0.0265, 0.0358 },
    .{ -0.0359, -0.0312, -0.0287 },
    .{ -0.0285, -0.1032, -0.1237 },
    .{ 0.1041, 0.0537, 0.0622 },
    .{ -0.0086, -0.0374, -0.0051 },
    .{ 0.039, 0.067, 0.2863 },
    .{ 0.0069, 0.0144, 0.0082 },
    .{ 0.0006, -0.0167, 0.0079 },
    .{ 0.0313, -0.0574, -0.0232 },
    .{ -0.1454, -0.0902, -0.0481 },
    .{ 0.0714, 0.0827, 0.0447 },
    .{ -0.0304, -0.0574, -0.0196 },
    .{ 0.0401, 0.0384, 0.0204 },
    .{ -0.0758, -0.0297, -0.0014 },
    .{ 0.0568, 0.1307, 0.1372 },
    .{ -0.0055, -0.031, -0.038 },
    .{ 0.0239, -0.0305, 0.0325 },
    .{ -0.0663, -0.0673, -0.014 },
    .{ -0.0416, -0.0047, -0.0023 },
    .{ 0.0166, 0.0112, -0.0093 },
    .{ -0.0211, 0.0011, 0.0331 },
    .{ 0.1833, 0.1466, 0.225 },
    .{ -0.0368, 0.037, 0.0295 },
    .{ -0.3441, -0.3543, -0.2008 },
    .{ -0.0479, -0.0489, -0.042 },
    .{ -0.066, -0.0153, 0.08 },
    .{ -0.0101, 0.0068, 0.0156 },
    .{ -0.069, -0.0452, -0.0927 },
    .{ -0.0145, 0.0041, 0.0015 },
    .{ 0.0421, 0.0451, 0.0373 },
    .{ 0.0504, -0.0483, -0.0356 },
    .{ -0.0837, 0.0168, 0.0055 },
};
pub const latent_rgb_bias = [3]f32{ 0.0317, -0.0878, -0.1388 };

/// The preview of a planar sampler latent `[channels][zh][zw]` into RGB8.
pub fn latentPreviewInto(rgb_out: []u8, z: []const f32, zh: usize, zw: usize) void {
    @import("sd_vae.zig").linearPreviewInto(channels, rgb_out, z, zh, zw, &latent_rgb_factors, latent_rgb_bias);
}

pub const Config = struct {
    dim: usize,
    heads: usize,
    head_dim: usize,
    n_double: usize,
    n_single: usize,
    channels: usize,
    /// `context_embedder`'s input: the encoder's last two states side by side.
    joint_dim: usize,
    /// One encoder hidden state, `caption_projection`'s input.
    te_dim: usize,
    t_freq: usize = 256,
    rope_axes: [3]usize,
    rope_theta: f64 = 10000.0,
    time_scale: f32 = 1000.0,
    eps: f32 = 1e-6,

    pub fn blocks(self: Config) usize {
        return self.n_double + self.n_single;
    }
};

/// The release's `FlowMatchEulerDiscreteScheduler`: diffusers' dynamic shift with mu
/// linear in the latent's token count, 0.5 at 256 tokens to 1.15 at 4096, no terminal.
pub const flow: @import("tp_core").schedule.DiffusersFlow = .{ .max_seq = 4096, .max_shift = 1.15, .terminal = null };

/// Fibo 1.5 as released, for callers with no file open.
pub const release: Config = .{ .dim = 3072, .heads = 24, .head_dim = 128, .n_double = 8, .n_single = 38, .channels = channels, .joint_dim = 4096, .te_dim = 2048, .rope_axes = .{ 16, 56, 56 } };

/// Where each block's modulation sits in a device table, in units of elements. A double
/// block holds the image's then the text's `[shift, 1 + scale, gate]` for attention then
/// MLP; a single block one such triple.
pub const Table = struct {
    dim: usize,
    n_double: usize,
    n_single: usize,

    pub const img = 0;
    pub const txt = 6;
    pub const shift_a = 0;
    pub const premul_a = 1;
    pub const gate_a = 2;
    pub const shift_f = 3;
    pub const premul_f = 4;
    pub const gate_f = 5;

    pub fn double(t: Table, i: usize) usize {
        return i * 12 * t.dim;
    }
    pub fn single(t: Table, i: usize) usize {
        return (t.n_double * 12 + i * 3) * t.dim;
    }
    pub fn len(t: Table) usize {
        return (t.n_double * 12 + t.n_single * 3) * t.dim;
    }
};

pub const LinearW = struct {
    w: Weight,
    b: ?[]const f32,
};

pub const Attn = struct {
    q: LinearW,
    k: LinearW,
    v: LinearW,
    qnorm: []const f32,
    knorm: []const f32,
};

pub const Double = struct {
    mod: LinearW, // dim -> 6 dim
    mod_ctx: LinearW,
    img: Attn,
    txt: Attn,
    out: LinearW,
    out_ctx: LinearW,
    ff_in: LinearW, // dim -> 4 dim
    ff_out: LinearW,
    ffc_in: LinearW,
    ffc_out: LinearW,
};

pub const Single = struct {
    mod: LinearW, // dim -> 3 dim
    attn: Attn,
    mlp: LinearW, // dim -> 4 dim
    out: LinearW, // 5 dim -> dim
};

pub const DiT = struct {
    arena: std.heap.ArenaAllocator,
    cfg: Config,
    x_embedder: LinearW,
    context_embedder: LinearW,
    t_lin1: LinearW,
    t_lin2: LinearW,
    /// One per block, double then single, `[dim / 2][te_dim]`, no bias.
    caption: []Weight,
    doubles: []Double,
    singles: []Single,
    norm_out: LinearW, // dim -> 2 dim: scale, shift
    proj_out: LinearW,
    device_lins: []const Weight,
    lora: ?*const lora_mod.Stack = null,

    pub fn load(gpa: std.mem.Allocator, store: WeightStore) !DiT {
        return loadDepth(gpa, store, null, null);
    }

    /// `load` with only the first `n_double` double and `n_single` single blocks, for a
    /// device check that sweeps depth.
    pub fn loadDepth(gpa: std.mem.Allocator, store: WeightStore, n_double: ?usize, n_single: ?usize) !DiT {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();
        const pfx: []const u8 = for ([_][]const u8{ "model.diffusion_model.", "transformer.", "" }) |p| {
            var b: [128]u8 = undefined;
            const n = std.mem.print(&b, "{s}{s}", .{ p, probes[0] }) catch continue;
            if (store.get(n) != null) break p;
        } else return error.MissingTensor;
        var l: Loader = .{ .store = store, .alloc = alloc, .pfx = pfx, .cfg = undefined };
        var cfg = try l.config();
        if (n_double) |n| cfg.n_double = @min(n, cfg.n_double);
        if (n_single) |n| cfg.n_single = @min(n, cfg.n_single);
        l.cfg = cfg;

        const caption = try alloc.alloc(Weight, cfg.blocks());
        for (caption, 0..) |*w, i| w.* = try l.mat("caption_projection.{d}.linear.weight", .{i}, cfg.dim / 2, cfg.te_dim);
        const doubles = try alloc.alloc(Double, cfg.n_double);
        for (doubles, 0..) |*b, i| b.* = try l.double(i);
        const singles = try alloc.alloc(Single, cfg.n_single);
        for (singles, 0..) |*b, i| b.* = try l.single(i);

        var dl: std.ArrayList(Weight) = .empty;
        for (doubles) |*b| try dl.appendSlice(alloc, &.{ b.img.q.w, b.img.k.w, b.img.v.w, b.txt.q.w, b.txt.k.w, b.txt.v.w, b.out.w, b.out_ctx.w, b.ff_in.w, b.ff_out.w, b.ffc_in.w, b.ffc_out.w });
        for (singles) |*b| try dl.appendSlice(alloc, &.{ b.attn.q.w, b.attn.k.w, b.attn.v.w, b.mlp.w, b.out.w });
        try dl.appendSlice(alloc, caption);

        var x_embedder = try l.linear("x_embedder", .{}, cfg.dim, cfg.channels);
        x_embedder.w = try ops.matmul.materializeF32(alloc, x_embedder.w);
        var proj_out = try l.linear("proj_out", .{}, cfg.channels, cfg.dim);
        proj_out.w = try ops.matmul.materializeF32(alloc, proj_out.w);
        const context_embedder = try l.linear("context_embedder", .{}, cfg.dim, cfg.joint_dim);
        const t_lin1 = try l.linear("time_embed.timestep_embedder.linear_1", .{}, cfg.dim, cfg.t_freq);
        const t_lin2 = try l.linear("time_embed.timestep_embedder.linear_2", .{}, cfg.dim, cfg.dim);
        const norm_out = try l.linear("norm_out.linear", .{}, 2 * cfg.dim, cfg.dim);
        const device_lins = try dl.toOwnedSlice(alloc);
        return .{
            .arena = arena,
            .cfg = cfg,
            .x_embedder = x_embedder,
            .context_embedder = context_embedder,
            .t_lin1 = t_lin1,
            .t_lin2 = t_lin2,
            .caption = caption,
            .doubles = doubles,
            .singles = singles,
            .norm_out = norm_out,
            .proj_out = proj_out,
            .device_lins = device_lins,
        };
    }

    pub fn deinit(self: *DiT) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// Resolve `stack` against every linear this model runs and hold it. Returns the
    /// file keys that apply.
    pub fn attachLora(self: *DiT, gpa: std.mem.Allocator, stack: *lora_mod.Stack) !usize {
        self.lora = stack;
        stack.beginAttach();
        try stack.resolveAll(self.device_lins);
        for (self.doubles) |*b| try stack.resolveAll(&.{ b.mod.w, b.mod_ctx.w });
        for (self.singles) |*b| _ = try stack.resolve(b.mod.w);
        try stack.resolveAll(&.{ self.x_embedder.w, self.context_embedder.w, self.t_lin1.w, self.t_lin2.w, self.norm_out.w, self.proj_out.w });
        return stack.finishAttach(gpa);
    }

    /// `temb`, `[dim]`: the timestep MLP at `sigma * 1000`; caller frees.
    pub fn timeEmbed(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, sigma: f32) ![]f32 {
        const cfg = self.cfg;
        const freq = try gpa.alloc(f32, cfg.t_freq);
        defer gpa.free(freq);
        ops.rope.sinCosEmbedding(freq, sigma * cfg.time_scale, 10000.0);
        const h = try gpa.alloc(f32, cfg.dim);
        defer gpa.free(h);
        try linear(io, gpa, self.lora, h, freq, 1, self.t_lin1);
        ops.act.silu(h);
        const out = try gpa.alloc(f32, cfg.dim);
        errdefer gpa.free(out);
        try linear(io, gpa, self.lora, out, h, 1, self.t_lin2);
        return out;
    }

    /// `linear(silu(temb))` for a modulation of `n` chunks; caller frees.
    pub fn modulation(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, lw: LinearW, temb: []const f32) ![]f32 {
        const s = try gpa.dupe(f32, temb);
        defer gpa.free(s);
        ops.act.silu(s);
        const out = try gpa.alloc(f32, lw.w.rows);
        errdefer gpa.free(out);
        try linear(io, gpa, self.lora, out, s, 1, lw);
        return out;
    }

    /// A device step's modulation table (`Table`), every vector at `temb`, scales
    /// stored as `1 + scale`. Caller frees.
    pub fn modTable(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, temb: []const f32) ![]f32 {
        const dim = self.cfg.dim;
        const t: Table = .{ .dim = dim, .n_double = self.cfg.n_double, .n_single = self.cfg.n_single };
        const out = try gpa.alloc(f32, t.len());
        errdefer gpa.free(out);
        for (self.doubles, 0..) |*b, i| {
            inline for (.{ .{ b.mod, Table.img }, .{ b.mod_ctx, Table.txt } }) |pair| {
                const m = try self.modulation(io, gpa, pair[0], temb);
                defer gpa.free(m);
                const dst = out[t.double(i) + pair[1] * dim ..][0 .. 6 * dim];
                @memcpy(dst, m);
                for (dst[1 * dim ..][0..dim]) |*v| v.* += 1.0;
                for (dst[4 * dim ..][0..dim]) |*v| v.* += 1.0;
            }
        }
        for (self.singles, 0..) |*b, i| {
            const m = try self.modulation(io, gpa, b.mod, temb);
            defer gpa.free(m);
            const dst = out[t.single(i)..][0 .. 3 * dim];
            @memcpy(dst, m);
            for (dst[1 * dim ..][0..dim]) |*v| v.* += 1.0;
        }
        return out;
    }

    /// Rope over `[text | image]`: text at (0, 0, 0), image cell (r, c) at (0, r, c).
    pub fn ropeFreqs(self: *const DiT, gpa: std.mem.Allocator, text_len: usize, h: usize, w: usize) !ops.rope.Freqs {
        const seq = text_len + h * w;
        const pos = try gpa.alloc(f32, seq * 3);
        defer gpa.free(pos);
        @memset(pos, 0);
        for (0..h) |r| for (0..w) |c| {
            const b = (text_len + r * w + c) * 3;
            pos[b + 1] = @floatFromInt(r);
            pos[b + 2] = @floatFromInt(c);
        };
        return ops.rope.fluxFreqs(gpa, pos, &self.cfg.rope_axes, self.cfg.rope_theta);
    }

    /// One forward. `x_lat`/`out` are planar `[channels][lat_h][lat_w]`; `ehs` is
    /// `[text_len][joint_dim]`; `layers` holds one `[text_len][te_dim]` state per block,
    /// double blocks first.
    pub fn forward(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, out: []f32, x_lat: []const f32, lat_h: usize, lat_w: usize, sigma: f32, ehs: []const f32, layers: []const f32, text_len: usize, cancel: ?*std.atomic.Value(bool)) !void {
        const cfg = self.cfg;
        const dim = cfg.dim;
        const n_img = lat_h * lat_w;
        const seq = text_len + n_img;
        std.debug.assert(x_lat.len == cfg.channels * n_img and out.len == x_lat.len);
        std.debug.assert(ehs.len == text_len * cfg.joint_dim and layers.len == cfg.blocks() * text_len * cfg.te_dim);

        const prev_tok = ops.cancel.token;
        ops.cancel.token = cancel;
        defer ops.cancel.token = prev_tok;

        const temb = try self.timeEmbed(io, gpa, sigma);
        defer gpa.free(temb);
        var freqs = try self.ropeFreqs(gpa, text_len, lat_h, lat_w);
        defer freqs.deinit(gpa);

        // The joint stream, `[text | image]`, both halves updated in place.
        const x = try gpa.alloc(f32, seq * dim);
        defer gpa.free(x);
        const txt = x[0 .. text_len * dim];
        const img = x[text_len * dim ..];
        try linear(io, gpa, self.lora, txt, ehs, text_len, self.context_embedder);
        {
            const tok = try gpa.alloc(f32, n_img * cfg.channels);
            defer gpa.free(tok);
            for (0..cfg.channels) |c| for (0..n_img) |r| {
                tok[r * cfg.channels + c] = x_lat[c * n_img + r];
            };
            try linear(io, gpa, self.lora, img, tok, n_img, self.x_embedder);
        }

        for (self.doubles, 0..) |*b, i| {
            if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
            try self.injectCaption(io, gpa, txt, layers, text_len, i);
            try self.doubleForward(io, gpa, b, x, text_len, n_img, temb, freqs);
        }
        for (self.singles, 0..) |*b, i| {
            if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
            try self.injectCaption(io, gpa, txt, layers, text_len, cfg.n_double + i);
            try self.singleForward(io, gpa, b, x, seq, temb, freqs);
        }

        const m = try self.finalMod(io, gpa, temb);
        defer gpa.free(m);
        try self.finalLayer(io, gpa, out, img, m, n_img);
    }

    /// `norm_out`'s modulation at `temb`, `[scale | shift]`; caller frees.
    pub fn finalMod(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, temb: []const f32) ![]f32 {
        return self.modulation(io, gpa, self.norm_out, temb);
    }

    /// `norm_out` then `proj_out` on the image rows `[n_img][dim]` (modified in place),
    /// into planar `out` `[channels][n_img]`.
    pub fn finalLayer(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, out: []f32, img: []f32, mod: []const f32, n_img: usize) !void {
        const cfg = self.cfg;
        const dim = cfg.dim;
        ops.norm.layerNormUnit(img, img, dim, cfg.eps);
        adaLn(img, mod[0..dim], null, mod[dim..][0..dim]);
        const tok = try gpa.alloc(f32, n_img * cfg.channels);
        defer gpa.free(tok);
        try linear(io, gpa, self.lora, tok, img, n_img, self.proj_out);
        for (0..cfg.channels) |c| for (0..n_img) |r| {
            out[c * n_img + r] = tok[r * cfg.channels + c];
        };
    }

    /// The text stream's second half of channels becomes block `i`'s projection of its
    /// hidden state.
    pub fn injectCaption(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, txt: []f32, layers: []const f32, text_len: usize, i: usize) !void {
        const cfg = self.cfg;
        const half = cfg.dim / 2;
        const proj = try gpa.alloc(f32, text_len * half);
        defer gpa.free(proj);
        try mm(io, gpa, self.lora, proj, layers[i * text_len * cfg.te_dim ..][0 .. text_len * cfg.te_dim], text_len, self.caption[i], null);
        for (0..text_len) |r| @memcpy(txt[r * cfg.dim + half ..][0..half], proj[r * half ..][0..half]);
    }

    fn doubleForward(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, b: *const Double, x: []f32, text_len: usize, n_img: usize, temb: []const f32, freqs: ops.rope.Freqs) !void {
        const cfg = self.cfg;
        const dim = cfg.dim;
        const seq = text_len + n_img;
        const txt = x[0 .. text_len * dim];
        const img = x[text_len * dim ..];
        const mi = try self.modulation(io, gpa, b.mod, temb);
        defer gpa.free(mi);
        const mt = try self.modulation(io, gpa, b.mod_ctx, temb);
        defer gpa.free(mt);
        const c = struct {
            fn at(m: []const f32, k: usize, d: usize) []const f32 {
                return m[k * d ..][0..d];
            }
        }.at;

        const nrm = try gpa.alloc(f32, seq * dim);
        defer gpa.free(nrm);
        ops.norm.layerNormUnit(nrm, x, dim, cfg.eps);
        adaLn(nrm[0 .. text_len * dim], c(mt, 1, dim), c(mt, 0, dim), null);
        adaLn(nrm[text_len * dim ..], c(mi, 1, dim), c(mi, 0, dim), null);

        // Joint q/k/v: text rows from the context projections, image rows from the image's.
        const q = try gpa.alloc(f32, seq * dim);
        defer gpa.free(q);
        const k = try gpa.alloc(f32, seq * dim);
        defer gpa.free(k);
        const v = try gpa.alloc(f32, seq * dim);
        defer gpa.free(v);
        inline for (.{ .{ &b.txt, 0, text_len }, .{ &b.img, text_len, n_img } }) |s| {
            const a = s[0];
            const r0: usize = s[1];
            const n: usize = s[2];
            const xs = nrm[r0 * dim ..][0 .. n * dim];
            try linear(io, gpa, self.lora, q[r0 * dim ..][0 .. n * dim], xs, n, a.q);
            try linear(io, gpa, self.lora, k[r0 * dim ..][0 .. n * dim], xs, n, a.k);
            try linear(io, gpa, self.lora, v[r0 * dim ..][0 .. n * dim], xs, n, a.v);
            ops.norm.rmsNorm(q[r0 * dim ..][0 .. n * dim], q[r0 * dim ..][0 .. n * dim], a.qnorm, cfg.eps);
            ops.norm.rmsNorm(k[r0 * dim ..][0 .. n * dim], k[r0 * dim ..][0 .. n * dim], a.knorm, cfg.eps);
        }
        const o = try self.attend(io, gpa, q, k, v, seq, freqs);
        defer gpa.free(o);

        const dlt = try gpa.alloc(f32, seq * dim);
        defer gpa.free(dlt);
        try linear(io, gpa, self.lora, dlt[0 .. text_len * dim], o[0 .. text_len * dim], text_len, b.out_ctx);
        try linear(io, gpa, self.lora, dlt[text_len * dim ..], o[text_len * dim ..], n_img, b.out);
        gatedAdd(txt, dlt[0 .. text_len * dim], c(mt, 2, dim));
        gatedAdd(img, dlt[text_len * dim ..], c(mi, 2, dim));

        ops.norm.layerNormUnit(nrm, x, dim, cfg.eps);
        adaLn(nrm[0 .. text_len * dim], c(mt, 4, dim), c(mt, 3, dim), null);
        adaLn(nrm[text_len * dim ..], c(mi, 4, dim), c(mi, 3, dim), null);
        try self.ff(io, gpa, b.ffc_in, b.ffc_out, nrm[0 .. text_len * dim], text_len, dlt[0 .. text_len * dim]);
        try self.ff(io, gpa, b.ff_in, b.ff_out, nrm[text_len * dim ..], n_img, dlt[text_len * dim ..]);
        gatedAdd(txt, dlt[0 .. text_len * dim], c(mt, 5, dim));
        gatedAdd(img, dlt[text_len * dim ..], c(mi, 5, dim));
    }

    fn singleForward(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, b: *const Single, x: []f32, seq: usize, temb: []const f32, freqs: ops.rope.Freqs) !void {
        const cfg = self.cfg;
        const dim = cfg.dim;
        const m = try self.modulation(io, gpa, b.mod, temb);
        defer gpa.free(m);
        const nrm = try gpa.alloc(f32, seq * dim);
        defer gpa.free(nrm);
        ops.norm.layerNormUnit(nrm, x, dim, cfg.eps);
        adaLn(nrm, m[dim..][0..dim], m[0..dim], null);

        const q = try gpa.alloc(f32, seq * dim);
        defer gpa.free(q);
        const k = try gpa.alloc(f32, seq * dim);
        defer gpa.free(k);
        const v = try gpa.alloc(f32, seq * dim);
        defer gpa.free(v);
        try linear(io, gpa, self.lora, q, nrm, seq, b.attn.q);
        try linear(io, gpa, self.lora, k, nrm, seq, b.attn.k);
        try linear(io, gpa, self.lora, v, nrm, seq, b.attn.v);
        ops.norm.rmsNorm(q, q, b.attn.qnorm, cfg.eps);
        ops.norm.rmsNorm(k, k, b.attn.knorm, cfg.eps);
        const o = try self.attend(io, gpa, q, k, v, seq, freqs);
        defer gpa.free(o);

        // proj_out over [attn | gelu(proj_mlp)], row by row.
        const hid = b.mlp.w.rows;
        const cat = try gpa.alloc(f32, seq * (dim + hid));
        defer gpa.free(cat);
        const mlp = try gpa.alloc(f32, seq * hid);
        defer gpa.free(mlp);
        try linear(io, gpa, self.lora, mlp, nrm, seq, b.mlp);
        ops.act.geluTanh(mlp);
        for (0..seq) |r| {
            @memcpy(cat[r * (dim + hid) ..][0..dim], o[r * dim ..][0..dim]);
            @memcpy(cat[r * (dim + hid) + dim ..][0..hid], mlp[r * hid ..][0..hid]);
        }
        const dlt = try gpa.alloc(f32, seq * dim);
        defer gpa.free(dlt);
        try linear(io, gpa, self.lora, dlt, cat, seq, b.out);
        gatedAdd(x, dlt, m[2 * dim ..][0..dim]);
    }

    /// Rope on q and k, then unmasked attention; returns `[seq][dim]`, caller frees.
    fn attend(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, q: []f32, k: []f32, v: []const f32, seq: usize, freqs: ops.rope.Freqs) ![]f32 {
        const cfg = self.cfg;
        ops.rope.applyInterleaved(q, freqs, seq, cfg.heads, cfg.head_dim);
        ops.rope.applyInterleaved(k, freqs, seq, cfg.heads, cfg.head_dim);
        const o = try gpa.alloc(f32, seq * cfg.dim);
        errdefer gpa.free(o);
        try ops.attention.attention(io, gpa, o, q, k, v, .{ .seq_q = seq, .seq_kv = seq, .n_heads = cfg.heads, .n_kv_heads = cfg.heads, .head_dim = cfg.head_dim });
        return o;
    }

    fn ff(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, in: LinearW, outw: LinearW, x: []const f32, rows: usize, out: []f32) !void {
        const h = try gpa.alloc(f32, rows * in.w.rows);
        defer gpa.free(h);
        try linear(io, gpa, self.lora, h, x, rows, in);
        ops.act.geluTanh(h);
        try linear(io, gpa, self.lora, out, h, rows, outw);
    }
};

/// Row-wise `x * (1 + scale) + shift`.
fn adaLn(x: []f32, scale: []const f32, shift: ?[]const f32, shift2: ?[]const f32) void {
    const dim = scale.len;
    const sh = shift orelse shift2.?;
    var row: usize = 0;
    while (row < x.len) : (row += dim) {
        for (x[row..][0..dim], scale, sh) |*v, s, t| v.* = v.* * (1.0 + s) + t;
    }
}

fn gatedAdd(x: []f32, delta: []const f32, gate: []const f32) void {
    const dim = gate.len;
    var row: usize = 0;
    while (row < x.len) : (row += dim) {
        for (x[row..][0..dim], delta[row..][0..dim], gate) |*v, d, g| v.* += g * d;
    }
}

fn linear(io: std.Io, gpa: std.mem.Allocator, stack: ?*const lora_mod.Stack, out: []f32, x: []const f32, m: usize, lw: LinearW) !void {
    try mm(io, gpa, stack, out, x, m, lw.w, lw.b);
}

fn mm(io: std.Io, gpa: std.mem.Allocator, stack: ?*const lora_mod.Stack, out: []f32, x: []const f32, m: usize, w: Weight, bias: ?[]const f32) !void {
    try ops.matmul.matmul(io, gpa, out, x, m, w, bias);
    if (stack) |s| try s.applyHost(io, gpa, out, w.rows, x, m, w, bias);
}

// --- weight loading ---------------------------------------------------------

const Loader = struct {
    store: WeightStore,
    alloc: std.mem.Allocator,
    pfx: []const u8,
    cfg: Config,

    fn name(l: Loader, buf: []u8, comptime fmt: []const u8, args: anytype) ![]const u8 {
        var w = std.Io.Writer.fixed(buf);
        try w.writeAll(l.pfx);
        try w.print(fmt, args);
        return w.buffered();
    }

    fn has(l: Loader, comptime fmt: []const u8, args: anytype) bool {
        var buf: [192]u8 = undefined;
        return l.store.get(l.name(&buf, fmt, args) catch return false) != null;
    }

    fn extent(l: Loader, comptime fmt: []const u8, args: anytype, axis: usize) !usize {
        var buf: [192]u8 = undefined;
        const nm = try l.name(&buf, fmt, args);
        const v = l.store.get(nm) orelse {
            std.log.err("fibo: missing tensor {s}", .{nm});
            return error.MissingTensor;
        };
        const shp = v.info.shape.slice();
        if (axis >= shp.len) return error.ShapeMismatch;
        return shp[axis];
    }

    fn count(l: Loader, comptime fmt: []const u8) usize {
        var n: usize = 0;
        while (l.has(fmt, .{n})) n += 1;
        return n;
    }

    fn config(l: Loader) !Config {
        const dim = try l.extent("x_embedder.weight", .{}, 0);
        const head_dim = try l.extent("transformer_blocks.0.attn.norm_q.weight", .{}, 0);
        if (head_dim == 0 or dim % head_dim != 0) return error.ShapeMismatch;
        return .{
            .dim = dim,
            .heads = dim / head_dim,
            .head_dim = head_dim,
            .n_double = l.count("transformer_blocks.{d}.norm1.linear.weight"),
            .n_single = l.count("single_transformer_blocks.{d}.norm.linear.weight"),
            .channels = try l.extent("x_embedder.weight", .{}, 1),
            .joint_dim = try l.extent("context_embedder.weight", .{}, 1),
            .te_dim = try l.extent("caption_projection.0.linear.weight", .{}, 1),
            .rope_axes = release.rope_axes,
        };
    }

    fn mat(l: Loader, comptime fmt: []const u8, args: anytype, rows: usize, cols: usize) !Weight {
        var buf: [192]u8 = undefined;
        return quant_weight.load(l.alloc, l.store, try l.name(&buf, fmt, args), rows, cols, .{ .who = "fibo" });
    }

    fn vec(l: Loader, comptime fmt: []const u8, args: anytype, len: usize) ![]f32 {
        var buf: [192]u8 = undefined;
        const nm = try l.name(&buf, fmt, args);
        const view = l.store.get(nm) orelse {
            std.log.err("fibo: missing tensor {s}", .{nm});
            return error.MissingTensor;
        };
        if (view.info.elemCount() != len) {
            std.log.err("fibo: {s} has {d} elements, expected {d}", .{ nm, view.info.elemCount(), len });
            return error.ShapeMismatch;
        }
        return view.toF32Alloc(l.alloc);
    }

    fn linear(l: Loader, comptime p: []const u8, args: anytype, rows: usize, cols: usize) !LinearW {
        return .{ .w = try l.mat(p ++ ".weight", args, rows, cols), .b = try l.vec(p ++ ".bias", args, rows) };
    }

    fn attn(l: Loader, comptime p: []const u8, i: usize, comptime q: []const u8, comptime k: []const u8, comptime v: []const u8, comptime qn: []const u8, comptime kn: []const u8) !Attn {
        const d = l.cfg.dim;
        return .{
            .q = try l.linear(p ++ q, .{i}, d, d),
            .k = try l.linear(p ++ k, .{i}, d, d),
            .v = try l.linear(p ++ v, .{i}, d, d),
            .qnorm = try l.vec(p ++ qn ++ ".weight", .{i}, l.cfg.head_dim),
            .knorm = try l.vec(p ++ kn ++ ".weight", .{i}, l.cfg.head_dim),
        };
    }

    fn double(l: Loader, i: usize) !Double {
        const d = l.cfg.dim;
        const p = "transformer_blocks.{d}.";
        return .{
            .mod = try l.linear(p ++ "norm1.linear", .{i}, 6 * d, d),
            .mod_ctx = try l.linear(p ++ "norm1_context.linear", .{i}, 6 * d, d),
            .img = try l.attn(p, i, "attn.to_q", "attn.to_k", "attn.to_v", "attn.norm_q", "attn.norm_k"),
            .txt = try l.attn(p, i, "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.norm_added_q", "attn.norm_added_k"),
            .out = try l.linear(p ++ "attn.to_out.0", .{i}, d, d),
            .out_ctx = try l.linear(p ++ "attn.to_add_out", .{i}, d, d),
            .ff_in = try l.linear(p ++ "ff.net.0.proj", .{i}, 4 * d, d),
            .ff_out = try l.linear(p ++ "ff.net.2", .{i}, d, 4 * d),
            .ffc_in = try l.linear(p ++ "ff_context.net.0.proj", .{i}, 4 * d, d),
            .ffc_out = try l.linear(p ++ "ff_context.net.2", .{i}, d, 4 * d),
        };
    }

    fn single(l: Loader, i: usize) !Single {
        const d = l.cfg.dim;
        const p = "single_transformer_blocks.{d}.";
        return .{
            .mod = try l.linear(p ++ "norm.linear", .{i}, 3 * d, d),
            .attn = try l.attn(p, i, "attn.to_q", "attn.to_k", "attn.to_v", "attn.norm_q", "attn.norm_k"),
            .mlp = try l.linear(p ++ "proj_mlp", .{i}, 4 * d, d),
            .out = try l.linear(p ++ "proj_out", .{i}, d, 5 * d),
        };
    }
};

// --- tests ------------------------------------------------------------------

const testing = std.testing;
const fixture = @embedFile("assets/fibo_forward.safetensors");

test "the Fibo 1.5 DiT matches diffusers' on the toy fixture" {
    const gpa = testing.allocator;
    const io = testing.io;
    var st = try safetensors.SafeTensors.initFromSlice(gpa, fixture);
    defer st.deinit();
    var model = try DiT.load(gpa, .{ .safetensors = &st });
    defer model.deinit();
    try testing.expectEqual(@as(usize, 2), model.doubles.len);
    try testing.expectEqual(@as(usize, 2), model.singles.len);
    model.cfg.rope_axes = .{ 8, 12, 12 };

    for ([_][]const u8{ "a", "b" }) |cs| {
        var kb: [64]u8 = undefined;
        const get = struct {
            fn f(s: *const safetensors.SafeTensors, b: []u8, c: []const u8, what: []const u8) ![]f32 {
                return (try s.require(try std.mem.print(b, "case.{s}.{s}", .{ c, what }))).toF32Alloc(testing.allocator);
            }
        }.f;
        const x = try get(&st, &kb, cs, "x");
        defer gpa.free(x);
        const ehs = try get(&st, &kb, cs, "ehs");
        defer gpa.free(ehs);
        const layers = try get(&st, &kb, cs, "layers");
        defer gpa.free(layers);
        const sig = try get(&st, &kb, cs, "sigma");
        defer gpa.free(sig);
        const want = try get(&st, &kb, cs, "out");
        defer gpa.free(want);
        const shp = (try st.require(try std.mem.print(&kb, "case.{s}.x", .{cs}))).info.shape.slice();
        const lat_h = shp[1];
        const lat_w = shp[2];
        const text_len = ehs.len / model.cfg.joint_dim;
        const got = try gpa.alloc(f32, x.len);
        defer gpa.free(got);
        try model.forward(io, gpa, got, x, lat_h, lat_w, sig[0], ehs, layers, text_len, null);
        var num: f64 = 0;
        var den: f64 = 0;
        for (want, got) |w, g| {
            num += @as(f64, w - g) * (w - g);
            den += @as(f64, w) * w;
        }
        const rel = @sqrt(num / den);
        errdefer std.debug.print("case {s}: velocity rel L2 {e}\n", .{ cs, rel });
        try testing.expect(rel < 1e-5);
    }
}
