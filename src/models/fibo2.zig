//! Fibo 2.0 (Bria), an 8B single-stream DiT on Flux 2's latent (128 channels at 16x,
//! patch 1), conditioned on Qwen3-VL-4B through two paths at once. Reference:
//! `comfy/ldm/fibo2/model.py`.
//!
//! The block is Z-Image's (`zimage.zig`): sandwich RMSNorms, shiftless four-chunk
//! modulation with tanh'd gates, SwiGLU, 3-axis interleaved RoPE at theta 256. What is
//! Fibo's own, each a silent wrong answer when missed:
//!
//! - The text never enters the sequence. A Perceiver squeezes the first context bundle
//!   into `gistCount(text_len)` gist tokens (256 up to 256 text tokens, rising to 512
//!   at 2048 in steps of 64), which lead the joint sequence; the gists sit on rope
//!   planes 1..n and the image on plane `max_gist + 1` whatever n is.
//! - Five injection blocks also cross-attend (no rope) to their own bundle and add it
//!   through a tanh'd SCALAR gate. The context is `[text][bundles * cap_dim]`, bundle 0
//!   for the Perceiver, bundle k for the k-th injection block.
//! - In the trunk the gist rows are modulated at t = 0 and the image rows at the
//!   sampled t; the noise refiner (image rows only) and the final layer use the sampled t.
//! - The timestep is `sigma * 1000`, not Z-Image's `(1 - sigma) * 1000`, the Q/K norms
//!   take the blocks' 1e-5, not float32's epsilon, and the output is not negated.
//!
//! Everything that depends on the text alone (gists, the injection blocks' K and V) is
//! a per-conditioning `Prefix`. Three spellings load to the same views: ComfyUI's
//! repack (`qkv`, `to_kv`, `w13` with its hidden width zero padded), the same with
//! `w1`/`w3` split, and diffusers' (`to_q`/`to_k`/`to_v`, `to_out.0`, `norm_q`).
//! Pinned by tools/gen_fibo2_forward.py.

const std = @import("std");
const tp_core = @import("tp_core");
const safetensors = tp_core.safetensors;
const weights_mod = tp_core.weights;
const ops = @import("tp_ops");
const quant_weight = @import("quant_weight.zig");
const lin = @import("lin.zig");
const lora_mod = @import("lora.zig");

const WeightStore = weights_mod.WeightStore;
pub const Weight = ops.matmul.Weight;

/// The tensor only this architecture has, ComfyUI's detection probe (with `final_layer.linear.weight`).
pub const probe = "perceiver.latent_queries";

/// Flux 2's latent: the VAE packs 2x2 cells of its 32 channels, so a token is one latent cell.
pub const channels = 128;

pub const Config = struct {
    dim: usize,
    n_heads: usize,
    head_dim: usize = 128,
    n_layers: usize,
    n_refiner_layers: usize,
    /// Bit i set: trunk block i cross-attends to its own bundle.
    inject_mask: u64,
    /// One bundle's width (three tapped Qwen3-VL layers of 2560).
    cap_dim: usize,
    perceiver_layers: usize,
    max_gist: usize,
    /// SwiGLU hidden width as stored; a fused `w13` carries zero rows past the real width.
    ffn_dim: usize,
    channels: usize,
    mod_dim: usize = 256,
    t_hidden: usize = 1024,
    t_freq: usize = 256,
    rope_theta: f64 = 256.0,
    rope_axes: [3]usize = .{ 32, 48, 48 },
    time_scale: f32 = 1000.0,
    norm_eps: f32 = 1e-5,
    final_eps: f32 = 1e-6,

    pub fn injects(self: Config, layer: usize) bool {
        return layer < 64 and (self.inject_mask >> @intCast(layer)) & 1 != 0;
    }
    pub fn nInject(self: Config) usize {
        return @popCount(self.inject_mask);
    }
    /// The context bundle trunk block `layer` reads; 0 is the Perceiver's.
    pub fn bundleOf(self: Config, layer: usize) usize {
        const below = if (layer == 0) 0 else self.inject_mask & ((@as(u64, 1) << @intCast(layer)) - 1);
        return 1 + @popCount(below);
    }
    pub fn ctxDim(self: Config) usize {
        return self.cap_dim * (1 + self.nInject());
    }
    /// `gist_count`: 256 up to 256 text tokens, linear to `max_gist` at 2048, rounded up to 64.
    pub fn gistCount(self: Config, text_len: usize) usize {
        const t: f64 = @floatFromInt(text_len);
        const ratio = @min(1.0, @max(0.0, (t - 256.0) / (2048.0 - 256.0)));
        const mg: f64 = @floatFromInt(self.max_gist);
        const n: usize = @intFromFloat(@ceil((256.0 + ratio * (mg - 256.0)) / 64.0) * 64.0);
        return @max(256, @min(self.max_gist, n));
    }
    /// The rope plane the image sits on.
    pub fn imagePlane(self: Config) usize {
        return self.max_gist + 1;
    }
};

/// Fibo 2.0 dev and edit turbo, as ComfyUI's detection reads them. A loader takes its
/// configuration from the file; this is for callers with no file open.
pub const release: Config = .{
    .dim = 3584,
    .n_heads = 28,
    .n_layers = 44,
    .n_refiner_layers = 2,
    .inject_mask = (1 << 6) | (1 << 14) | (1 << 22) | (1 << 30) | (1 << 38),
    .cap_dim = 7680,
    .perceiver_layers = 2,
    .max_gist = 512,
    .ffn_dim = 9728,
    .channels = channels,
};

pub const LinearW = struct {
    w: Weight,
    b: ?[]const f32,
};

/// The trunk's self-attention: per-head RMSNorm on q and k, then rope.
pub const SelfAttn = struct {
    q: Weight,
    k: Weight,
    v: Weight,
    out: Weight,
    qnorm: []const f32,
    knorm: []const f32,
};

/// `fibo2.Attention`, the cross-attention and the Perceiver's attention: no rope.
pub const XAttn = struct {
    q: Weight,
    k: Weight,
    v: Weight,
    out: Weight,
    qnorm: []const f32,
    knorm: []const f32,
};

pub const Ffn = struct {
    w1: Weight, // gate
    w3: Weight, // up
    w2: Weight, // down
};

pub const Cross = struct {
    norm1: []const f32,
    attn: XAttn,
    norm2: []const f32,
    /// `tanh(cross_attn_gate)`, a scalar.
    gate: f32,
};

pub const Block = struct {
    attn: SelfAttn,
    ffn: Ffn,
    attn_norm1: []const f32,
    attn_norm2: []const f32,
    ffn_norm1: []const f32,
    ffn_norm2: []const f32,
    ada: LinearW, // mod_dim -> 4 * dim
    cross: ?Cross,
};

pub const PerceiverLayer = struct {
    cross_norm: []const f32,
    cross: XAttn,
    self_norm: []const f32,
    self: XAttn,
    ffn_norm: []const f32,
    ffn: Ffn,
};

/// What a conditioning contributes, constant across steps: the gist rows and each
/// injection block's normed keys and values over the text.
pub const Prefix = struct {
    n_gist: usize,
    text_len: usize,
    /// `[n_gist][dim]`.
    gist: []f32,
    /// Per injection block in trunk order, `[text_len][dim]` each.
    k: [][]f32,
    v: [][]f32,

    pub fn deinit(self: *Prefix, gpa: std.mem.Allocator) void {
        gpa.free(self.gist);
        for (self.k) |b| gpa.free(b);
        for (self.v) |b| gpa.free(b);
        gpa.free(self.k);
        gpa.free(self.v);
        self.* = undefined;
    }
};

pub const DiT = struct {
    arena: std.heap.ArenaAllocator,
    cfg: Config,

    x_embedder: LinearW, // channels -> dim
    t_mlp0: LinearW, // t_freq -> t_hidden
    t_mlp2: LinearW, // t_hidden -> mod_dim
    noise_refiner: []Block,
    layers: []Block,
    latent_queries: []const f32, // [max_gist][dim]
    perceiver: []PerceiverLayer,
    perceiver_norm: []const f32,
    final_ada: LinearW, // mod_dim -> dim, after a SiLU
    final_linear: LinearW, // dim -> channels
    /// Every linear the device forwards run per step: the refiner and trunk blocks,
    /// cross-attention q and out included.
    device_lins: []const Weight,
    /// The linears of the per-conditioning prefix: the Perceiver and the injection blocks' k and v.
    prefix_lins: []const Weight,
    lora: ?*const lora_mod.Stack = null,

    pub fn load(gpa: std.mem.Allocator, store: WeightStore) !DiT {
        return loadDepth(gpa, store, null);
    }

    /// `load` with the trunk cut to its first `depth` blocks (and their injections), for
    /// a device check that sweeps depth. The context keeps one bundle per kept injection.
    pub fn loadDepth(gpa: std.mem.Allocator, store: WeightStore, depth: ?usize) !DiT {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();

        const pfx: []const u8 = if (store.get("model.diffusion_model." ++ probe) != null) "model.diffusion_model." else "";
        var l: Loader = .{ .store = store, .alloc = alloc, .pfx = pfx, .cfg = undefined, .spell = undefined };
        l.spell = if (l.has("layers.0.attention.qkv.weight", .{})) .comfy else .diffusers;
        var cfg = try l.config();
        if (depth) |n| if (n > 0 and n < cfg.n_layers) {
            cfg.n_layers = n;
            cfg.inject_mask &= (@as(u64, 1) << @intCast(n)) - 1;
        };
        l.cfg = cfg;

        const noise_refiner = try alloc.alloc(Block, cfg.n_refiner_layers);
        for (noise_refiner, 0..) |*b, i| b.* = try l.block("noise_refiner.{d}", i, false);
        const layers = try alloc.alloc(Block, cfg.n_layers);
        for (layers, 0..) |*b, i| b.* = try l.block("layers.{d}", i, cfg.injects(i));
        const perceiver = try alloc.alloc(PerceiverLayer, cfg.perceiver_layers);
        for (perceiver, 0..) |*p, i| p.* = try l.perceiverLayer(i);

        var dl: std.ArrayList(Weight) = .empty;
        for ([_][]Block{ noise_refiner, layers }) |group| for (group) |*b| {
            try dl.appendSlice(alloc, &blockLins(b));
            if (b.cross) |c| try dl.appendSlice(alloc, &.{ c.attn.q, c.attn.out });
        };
        var pl: std.ArrayList(Weight) = .empty;
        for (perceiver) |*p| try pl.appendSlice(alloc, &.{ p.cross.q, p.cross.k, p.cross.v, p.cross.out, p.self.q, p.self.k, p.self.v, p.self.out, p.ffn.w1, p.ffn.w3, p.ffn.w2 });
        for (layers) |*b| if (b.cross) |c| try pl.appendSlice(alloc, &.{ c.attn.k, c.attn.v });

        // The two projections a device arm runs on its f32-only fused matmul; tiny.
        var x_embedder = try l.linear("x_embedder", cfg.dim, cfg.channels);
        x_embedder.w = try ops.matmul.materializeF32(alloc, x_embedder.w);
        var final_linear = try l.linear("final_layer.linear", cfg.channels, cfg.dim);
        final_linear.w = try ops.matmul.materializeF32(alloc, final_linear.w);
        const t_mlp0 = try l.linear("t_embedder.mlp.0", cfg.t_hidden, cfg.t_freq);
        const t_mlp2 = try l.linear("t_embedder.mlp.2", cfg.mod_dim, cfg.t_hidden);
        const latent_queries = try l.vec("perceiver.latent_queries", .{}, cfg.max_gist * cfg.dim);
        const perceiver_norm = try l.vec("perceiver.norm_out.weight", .{}, cfg.dim);
        const final_ada = try l.linear("final_layer.adaLN_modulation.1", cfg.dim, cfg.mod_dim);
        const device_lins = try dl.toOwnedSlice(alloc);
        const prefix_lins = try pl.toOwnedSlice(alloc);

        return .{
            .arena = arena,
            .cfg = cfg,
            .x_embedder = x_embedder,
            .t_mlp0 = t_mlp0,
            .t_mlp2 = t_mlp2,
            .noise_refiner = noise_refiner,
            .layers = layers,
            .latent_queries = latent_queries,
            .perceiver = perceiver,
            .perceiver_norm = perceiver_norm,
            .final_ada = final_ada,
            .final_linear = final_linear,
            .device_lins = device_lins,
            .prefix_lins = prefix_lins,
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
        try stack.resolveAll(self.prefix_lins);
        for ([_][]Block{ self.noise_refiner, self.layers }) |group| for (group) |*b| {
            _ = try stack.resolve(b.ada.w);
        };
        try stack.resolveAll(&.{ self.x_embedder.w, self.t_mlp0.w, self.t_mlp2.w, self.final_ada.w, self.final_linear.w });
        return stack.finishAttach(gpa);
    }

    /// `t_embedder(t * time_scale)`, `[mod_dim]`; caller frees. The trunk's gist rows
    /// take it at `t = 0`.
    pub fn adalnInput(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, t: f32) ![]f32 {
        const cfg = self.cfg;
        const freq = try gpa.alloc(f32, cfg.t_freq);
        defer gpa.free(freq);
        ops.rope.sinCosEmbedding(freq, t * cfg.time_scale, 10000.0);
        const hidden = try gpa.alloc(f32, cfg.t_hidden);
        defer gpa.free(hidden);
        try linear(io, gpa, self.lora, hidden, freq, 1, self.t_mlp0);
        ops.act.silu(hidden);
        const out = try gpa.alloc(f32, cfg.mod_dim);
        errdefer gpa.free(out);
        try linear(io, gpa, self.lora, out, hidden, 1, self.t_mlp2);
        return out;
    }

    /// One block's `[scale_msa | tanh(gate_msa) | scale_mlp | tanh(gate_mlp)]`, `[4 * dim]`.
    pub fn blockMod(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, out: []f32, blk: *const Block, adaln: []const f32) !void {
        const dim = self.cfg.dim;
        try linear(io, gpa, self.lora, out, adaln, 1, blk.ada);
        for (out[1 * dim ..][0..dim]) |*g| g.* = std.math.tanh(g.*);
        for (out[3 * dim ..][0..dim]) |*g| g.* = std.math.tanh(g.*);
    }

    /// A device modulation table (`fibo2_dev.Tbl`): `.step` is every refiner then trunk
    /// block at `adaln`, `.zero` the trunk blocks at t = 0's `adaln`, each block
    /// `[scale_msa | gate_msa | scale_mlp | gate_mlp]`, then one zero block for the shift
    /// the modulation does not have. With `fold`, the two scales become
    /// `norm_w * (1 + scale)`, the premultiplier a fused rms-modulate kernel reads.
    /// Caller frees.
    pub fn modTable(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, adaln: []const f32, zero: bool, fold: bool) ![]f32 {
        const dim = self.cfg.dim;
        const groups: []const []Block = if (zero) &.{self.layers} else &.{ self.noise_refiner, self.layers };
        var n: usize = 0;
        for (groups) |g| n += g.len;
        const out = try gpa.alloc(f32, n * 4 * dim + dim);
        errdefer gpa.free(out);
        @memset(out[n * 4 * dim ..], 0);
        var i: usize = 0;
        for (groups) |g| for (g) |*blk| {
            const e = out[i * 4 * dim ..][0 .. 4 * dim];
            try self.blockMod(io, gpa, e, blk, adaln);
            if (fold) {
                for (e[0..dim], blk.attn_norm1) |*v, w| v.* = w * (1.0 + v.*);
                for (e[2 * dim ..][0..dim], blk.ffn_norm1) |*v, w| v.* = w * (1.0 + v.*);
            }
            i += 1;
        };
        return out;
    }

    /// The final layer's scale, `final_ada(silu(adaln))`, `[dim]`; caller frees.
    pub fn finalScale(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, adaln: []const f32) ![]f32 {
        const gated = try gpa.dupe(f32, adaln);
        defer gpa.free(gated);
        ops.act.silu(gated);
        const out = try gpa.alloc(f32, self.cfg.dim);
        errdefer gpa.free(out);
        try linear(io, gpa, self.lora, out, gated, 1, self.final_ada);
        return out;
    }

    /// Rope positions for `[gists | image]`: gist i on `(i + 1, 0, 0)`, image cell
    /// `(r, c)` on `(max_gist + 1, r, c)`. Returns the table over all `n_gist + h * w` rows.
    pub fn ropeFreqs(self: *const DiT, gpa: std.mem.Allocator, n_gist: usize, h: usize, w: usize) !ops.rope.Freqs {
        const cfg = self.cfg;
        const seq = n_gist + h * w;
        const pos = try gpa.alloc(f32, seq * 3);
        defer gpa.free(pos);
        for (0..n_gist) |i| {
            pos[i * 3 ..][0..3].* = .{ @floatFromInt(i + 1), 0, 0 };
        }
        const plane: f32 = @floatFromInt(cfg.imagePlane());
        for (0..h) |r| for (0..w) |c| {
            pos[(n_gist + r * w + c) * 3 ..][0..3].* = .{ plane, @floatFromInt(r), @floatFromInt(c) };
        };
        return ops.rope.fluxFreqs(gpa, pos, &cfg.rope_axes, cfg.rope_theta);
    }

    /// The per-conditioning half: the Perceiver over bundle 0 and each injection block's
    /// keys and values over its bundle. `ctx` is `[text_len][ctxDim]`.
    pub fn prefix(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, ctx: []const f32, text_len: usize) !Prefix {
        const cfg = self.cfg;
        const dim = cfg.dim;
        std.debug.assert(ctx.len == text_len * cfg.ctxDim());
        const n_gist = cfg.gistCount(text_len);

        const bundle = try gpa.alloc(f32, text_len * cfg.cap_dim);
        defer gpa.free(bundle);
        bundleOf(cfg, bundle, ctx, text_len, 0);

        const gist = try gpa.alloc(f32, n_gist * dim);
        errdefer gpa.free(gist);
        @memcpy(gist, self.latent_queries[0 .. n_gist * dim]);
        {
            const nrm = try gpa.alloc(f32, n_gist * dim);
            defer gpa.free(nrm);
            const dlt = try gpa.alloc(f32, n_gist * dim);
            defer gpa.free(dlt);
            for (self.perceiver) |*p| {
                ops.norm.rmsNorm(nrm, gist, p.cross_norm, cfg.norm_eps);
                try self.xattn(io, gpa, &p.cross, dlt, nrm, n_gist, bundle, text_len);
                plainAdd(gist, dlt);
                ops.norm.rmsNorm(nrm, gist, p.self_norm, cfg.norm_eps);
                try self.xattn(io, gpa, &p.self, dlt, nrm, n_gist, nrm, n_gist);
                plainAdd(gist, dlt);
                ops.norm.rmsNorm(nrm, gist, p.ffn_norm, cfg.norm_eps);
                try self.ffnForward(io, gpa, &p.ffn, nrm, n_gist, dlt);
                plainAdd(gist, dlt);
            }
            ops.norm.rmsNorm(gist, gist, self.perceiver_norm, cfg.norm_eps);
        }

        const n_inj = cfg.nInject();
        const ks = try gpa.alloc([]f32, n_inj);
        errdefer gpa.free(ks);
        const vs = try gpa.alloc([]f32, n_inj);
        errdefer gpa.free(vs);
        var made: usize = 0;
        errdefer for (0..made) |j| {
            gpa.free(ks[j]);
            gpa.free(vs[j]);
        };
        for (self.layers, 0..) |*b, li| {
            const c = b.cross orelse continue;
            bundleOf(cfg, bundle, ctx, text_len, cfg.bundleOf(li));
            ks[made] = try gpa.alloc(f32, text_len * dim);
            vs[made] = gpa.alloc(f32, text_len * dim) catch |e| {
                gpa.free(ks[made]);
                return e;
            };
            made += 1;
            try self.crossKv(io, gpa, &c.attn, bundle, text_len, ks[made - 1], vs[made - 1]);
        }
        std.debug.assert(made == n_inj);
        return .{ .n_gist = n_gist, .text_len = text_len, .gist = gist, .k = ks, .v = vs };
    }

    /// One injection block's normed keys and its values over a bundle.
    pub fn crossKv(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, a: *const XAttn, bundle: []const f32, text_len: usize, k: []f32, v: []f32) !void {
        try mm(io, gpa, self.lora, k, bundle, text_len, a.k, null);
        try mm(io, gpa, self.lora, v, bundle, text_len, a.v, null);
        ops.norm.rmsNorm(k, k, a.knorm, self.cfg.norm_eps);
    }

    /// One forward from scratch. `x_lat`/`out` are planar `[channels][lat_h][lat_w]`.
    pub fn forward(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, out: []f32, x_lat: []const f32, lat_h: usize, lat_w: usize, sigma: f32, ctx: []const f32, text_len: usize, cancel: ?*std.atomic.Value(bool)) !void {
        var pre = try self.prefix(io, gpa, ctx, text_len);
        defer pre.deinit(gpa);
        try self.predict(io, gpa, out, x_lat, lat_h, lat_w, sigma, &pre, cancel);
    }

    /// The per-step half, against a cached `Prefix`.
    pub fn predict(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, out: []f32, x_lat: []const f32, lat_h: usize, lat_w: usize, sigma: f32, pre: *const Prefix, cancel: ?*std.atomic.Value(bool)) !void {
        const cfg = self.cfg;
        const dim = cfg.dim;
        const n_img = lat_h * lat_w;
        const n_gist = pre.n_gist;
        const seq = n_gist + n_img;
        std.debug.assert(x_lat.len == cfg.channels * n_img and out.len == x_lat.len);

        const prev_tok = ops.cancel.token;
        ops.cancel.token = cancel;
        defer ops.cancel.token = prev_tok;

        const adaln = try self.adalnInput(io, gpa, sigma);
        defer gpa.free(adaln);
        const adaln0 = try self.adalnInput(io, gpa, 0);
        defer gpa.free(adaln0);

        var freqs = try self.ropeFreqs(gpa, n_gist, lat_h, lat_w);
        defer freqs.deinit(gpa);
        const img_freqs: ops.rope.Freqs = .{ .cos = freqs.cos[n_gist * freqs.half ..], .sin = freqs.sin[n_gist * freqs.half ..], .half = freqs.half };

        const x = try gpa.alloc(f32, seq * dim);
        defer gpa.free(x);
        const img = x[n_gist * dim ..];
        {
            const tok = try tokens(gpa, cfg.channels, x_lat, n_img);
            defer gpa.free(tok);
            try linear(io, gpa, self.lora, img, tok, n_img, self.x_embedder);
        }
        const mod = try gpa.alloc(f32, 4 * dim);
        defer gpa.free(mod);
        const mod0 = try gpa.alloc(f32, 4 * dim);
        defer gpa.free(mod0);
        for (self.noise_refiner) |*blk| {
            try self.blockMod(io, gpa, mod, blk, adaln);
            try self.blockForward(io, gpa, blk, img, n_img, 0, mod, mod, img_freqs, null);
        }

        @memcpy(x[0 .. n_gist * dim], pre.gist);
        var inj: usize = 0;
        for (self.layers) |*blk| {
            if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
            try self.blockMod(io, gpa, mod, blk, adaln);
            try self.blockMod(io, gpa, mod0, blk, adaln0);
            const kv: ?KV = if (blk.cross != null) .{ .k = pre.k[inj], .v = pre.v[inj], .len = pre.text_len } else null;
            if (blk.cross != null) inj += 1;
            try self.blockForward(io, gpa, blk, x, seq, n_gist, mod0, mod, freqs, kv);
        }

        try self.finalize(io, gpa, out, img, adaln, n_img);
    }

    /// `final_layer` on the image rows (in place), then back to planar.
    pub fn finalize(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, out: []f32, img_rows: []f32, adaln: []const f32, n_img: usize) !void {
        const cfg = self.cfg;
        const scale = try self.finalScale(io, gpa, adaln);
        defer gpa.free(scale);
        const tok = try gpa.alloc(f32, n_img * cfg.channels);
        defer gpa.free(tok);
        try self.finalLayer(io, gpa, tok, img_rows, scale, n_img);
        untokens(cfg.channels, out, tok, n_img);
    }

    /// Unit LayerNorm, `* (1 + scale)` and the final linear, row-wise: `img_rows` is
    /// modified in place and `out` is `[m][channels]`.
    pub fn finalLayer(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, out: []f32, img_rows: []f32, scale: []const f32, m: usize) !void {
        const dim = self.cfg.dim;
        ops.norm.layerNormUnit(img_rows, img_rows, dim, self.cfg.final_eps);
        modulate(img_rows, scale);
        try linear(io, gpa, self.lora, out, img_rows, m, self.final_linear);
    }

    const KV = struct { k: []const f32, v: []const f32, len: usize };

    /// One block over `x` `[rows][dim]`. Rows `[0, split)` take `mod_lo`, the rest
    /// `mod_hi`; `kv` is set exactly for an injection block.
    fn blockForward(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, blk: *const Block, x: []f32, rows: usize, split: usize, mod_lo: []const f32, mod_hi: []const f32, freqs: ops.rope.Freqs, kv: ?KV) !void {
        const cfg = self.cfg;
        const dim = cfg.dim;
        std.debug.assert(x.len == rows * dim);
        const nrm = try gpa.alloc(f32, rows * dim);
        defer gpa.free(nrm);
        const dlt = try gpa.alloc(f32, rows * dim);
        defer gpa.free(dlt);
        const lo = split * dim;

        // x += tanh(gate_msa) * attn_norm2(attn(modulate(attn_norm1(x), scale_msa)))
        ops.norm.rmsNorm(nrm, x, blk.attn_norm1, cfg.norm_eps);
        modulate(nrm[0..lo], mod_lo[0..dim]);
        modulate(nrm[lo..], mod_hi[0..dim]);
        try self.selfAttn(io, gpa, &blk.attn, nrm, rows, freqs, dlt);
        ops.norm.rmsNorm(dlt, dlt, blk.attn_norm2, cfg.norm_eps);
        gatedAdd(x[0..lo], dlt[0..lo], mod_lo[dim..][0..dim]);
        gatedAdd(x[lo..], dlt[lo..], mod_hi[dim..][0..dim]);

        // x += tanh(gate) * cross_attn_norm2(cross_attention(cross_attn_norm1(x), bundle))
        if (blk.cross) |*c| {
            const t = kv.?;
            ops.norm.rmsNorm(nrm, x, c.norm1, cfg.norm_eps);
            try self.crossAttend(io, gpa, &c.attn, dlt, nrm, rows, t.k, t.v, t.len);
            ops.norm.rmsNorm(dlt, dlt, c.norm2, cfg.norm_eps);
            for (x, dlt) |*v, d| v.* += c.gate * d;
        }

        // x += tanh(gate_mlp) * ffn_norm2(swiglu(modulate(ffn_norm1(x), scale_mlp)))
        ops.norm.rmsNorm(nrm, x, blk.ffn_norm1, cfg.norm_eps);
        modulate(nrm[0..lo], mod_lo[2 * dim ..][0..dim]);
        modulate(nrm[lo..], mod_hi[2 * dim ..][0..dim]);
        try self.ffnForward(io, gpa, &blk.ffn, nrm, rows, dlt);
        ops.norm.rmsNorm(dlt, dlt, blk.ffn_norm2, cfg.norm_eps);
        gatedAdd(x[0..lo], dlt[0..lo], mod_lo[3 * dim ..][0..dim]);
        gatedAdd(x[lo..], dlt[lo..], mod_hi[3 * dim ..][0..dim]);
    }

    fn selfAttn(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, a: *const SelfAttn, x: []const f32, rows: usize, freqs: ops.rope.Freqs, out: []f32) !void {
        const cfg = self.cfg;
        const dim = cfg.dim;
        const q = try gpa.alloc(f32, rows * dim);
        defer gpa.free(q);
        const k = try gpa.alloc(f32, rows * dim);
        defer gpa.free(k);
        const v = try gpa.alloc(f32, rows * dim);
        defer gpa.free(v);
        try mm(io, gpa, self.lora, q, x, rows, a.q, null);
        try mm(io, gpa, self.lora, k, x, rows, a.k, null);
        try mm(io, gpa, self.lora, v, x, rows, a.v, null);
        ops.norm.rmsNorm(q, q, a.qnorm, cfg.norm_eps);
        ops.norm.rmsNorm(k, k, a.knorm, cfg.norm_eps);
        ops.rope.applyInterleaved(q, freqs, rows, cfg.n_heads, cfg.head_dim);
        ops.rope.applyInterleaved(k, freqs, rows, cfg.n_heads, cfg.head_dim);
        try self.attendOut(io, gpa, a.out, out, q, rows, k, v, rows);
    }

    /// `fibo2.Attention` with `context` given: queries from `x`, keys and values from `ctx`.
    fn xattn(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, a: *const XAttn, out: []f32, x: []const f32, rows: usize, ctx: []const f32, ctx_rows: usize) !void {
        const dim = self.cfg.dim;
        const k = try gpa.alloc(f32, ctx_rows * dim);
        defer gpa.free(k);
        const v = try gpa.alloc(f32, ctx_rows * dim);
        defer gpa.free(v);
        try self.crossKv(io, gpa, a, ctx, ctx_rows, k, v);
        try self.crossAttend(io, gpa, a, out, x, rows, k, v, ctx_rows);
    }

    /// The query side of `fibo2.Attention` against keys already normed.
    fn crossAttend(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, a: *const XAttn, out: []f32, x: []const f32, rows: usize, k: []const f32, v: []const f32, kv_rows: usize) !void {
        const cfg = self.cfg;
        const q = try gpa.alloc(f32, rows * cfg.dim);
        defer gpa.free(q);
        try mm(io, gpa, self.lora, q, x, rows, a.q, null);
        ops.norm.rmsNorm(q, q, a.qnorm, cfg.norm_eps);
        try self.attendOut(io, gpa, a.out, out, q, rows, k, v, kv_rows);
    }

    fn attendOut(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, wo: Weight, out: []f32, q: []const f32, rows: usize, k: []const f32, v: []const f32, kv_rows: usize) !void {
        const cfg = self.cfg;
        const o = try gpa.alloc(f32, rows * cfg.dim);
        defer gpa.free(o);
        try ops.attention.attention(io, gpa, o, q, k, v, .{
            .seq_q = rows,
            .seq_kv = kv_rows,
            .n_heads = cfg.n_heads,
            .n_kv_heads = cfg.n_heads,
            .head_dim = cfg.head_dim,
        });
        try mm(io, gpa, self.lora, out, o, rows, wo, null);
    }

    fn ffnForward(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, f: *const Ffn, x: []const f32, rows: usize, out: []f32) !void {
        const hid = self.cfg.ffn_dim;
        const gate = try gpa.alloc(f32, rows * hid);
        defer gpa.free(gate);
        const up = try gpa.alloc(f32, rows * hid);
        defer gpa.free(up);
        try mm(io, gpa, self.lora, gate, x, rows, f.w1, null);
        try mm(io, gpa, self.lora, up, x, rows, f.w3, null);
        ops.act.siluMul(gate, up);
        try mm(io, gpa, self.lora, out, gate, rows, f.w2, null);
    }
};

/// The trunk-block linears a device forward runs, in a fixed order.
pub fn blockLins(b: *const Block) [7]Weight {
    return .{ b.attn.q, b.attn.k, b.attn.v, b.attn.out, b.ffn.w1, b.ffn.w3, b.ffn.w2 };
}

/// Bundle `idx` of `ctx` `[text_len][ctxDim]` into `out` `[text_len][cap_dim]`.
pub fn bundleOf(cfg: Config, out: []f32, ctx: []const f32, text_len: usize, idx: usize) void {
    const cd = cfg.cap_dim;
    const w = cfg.ctxDim();
    for (0..text_len) |r| @memcpy(out[r * cd ..][0..cd], ctx[r * w + idx * cd ..][0..cd]);
}

/// Planar `[c][n]` -> token rows `[n][c]`; caller frees.
pub fn tokens(gpa: std.mem.Allocator, c_n: usize, planar: []const f32, n: usize) ![]f32 {
    const out = try gpa.alloc(f32, n * c_n);
    for (0..c_n) |c| for (0..n) |r| {
        out[r * c_n + c] = planar[c * n + r];
    };
    return out;
}

/// Token rows `[n][c]` -> planar `[c][n]`.
pub fn untokens(c_n: usize, planar: []f32, tok: []const f32, n: usize) void {
    for (0..c_n) |c| for (0..n) |r| {
        planar[c * n + r] = tok[r * c_n + c];
    };
}

fn modulate(x: []f32, scale: []const f32) void {
    const dim = scale.len;
    var row: usize = 0;
    while (row < x.len) : (row += dim) {
        for (x[row..][0..dim], scale) |*v, sc| v.* = (1.0 + sc) * v.*;
    }
}

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

fn linear(io: std.Io, gpa: std.mem.Allocator, stack: ?*const lora_mod.Stack, out: []f32, x: []const f32, m: usize, lw: LinearW) !void {
    try mm(io, gpa, stack, out, x, m, lw.w, lw.b);
}

/// Every host GEMM on a model weight goes through here, so a LoRA cannot be missed.
fn mm(io: std.Io, gpa: std.mem.Allocator, stack: ?*const lora_mod.Stack, out: []f32, x: []const f32, m: usize, w: Weight, bias: ?[]const f32) !void {
    try ops.matmul.matmul(io, gpa, out, x, m, w, bias);
    if (stack) |s| try s.applyHost(io, gpa, out, w.rows, x, m, w, bias);
}

/// How LoRAs trained on diffusers' names reach ComfyUI's fused tensors.
pub const lora_keys_fused = [_]lora_mod.Rename{
    .{ .from = ".attention.to_q", .to = ".attention.qkv", .slot = 0, .slots = 3 },
    .{ .from = ".attention.to_k", .to = ".attention.qkv", .slot = 1, .slots = 3 },
    .{ .from = ".attention.to_v", .to = ".attention.qkv", .slot = 2, .slots = 3 },
    .{ .from = ".attention.to_out.0", .to = ".attention.out" },
    .{ .from = "_attention.to_k", .to = "_attention.to_kv", .slot = 0, .slots = 2 },
    .{ .from = "_attention.to_v", .to = "_attention.to_kv", .slot = 1, .slots = 2 },
    .{ .from = "_attention.to_out.0", .to = "_attention.to_out" },
};

// --- weight loading ---------------------------------------------------------

const Spelling = enum { comfy, diffusers };

const Loader = struct {
    store: WeightStore,
    alloc: std.mem.Allocator,
    pfx: []const u8,
    cfg: Config,
    spell: Spelling,

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

    /// Extent `axis` of a 2-D tensor.
    fn extent(l: Loader, comptime fmt: []const u8, args: anytype, axis: usize) !usize {
        var buf: [192]u8 = undefined;
        const nm = try l.name(&buf, fmt, args);
        const v = l.store.get(nm) orelse {
            std.log.err("fibo2: missing tensor {s}", .{nm});
            return error.MissingTensor;
        };
        const shp = v.info.shape.slice();
        if (shp.len != 2) {
            std.log.err("fibo2: {s} is {d}-D, expected 2-D", .{ nm, shp.len });
            return error.ShapeMismatch;
        }
        return shp[axis];
    }

    fn count(l: Loader, comptime fmt: []const u8) usize {
        var n: usize = 0;
        while (l.has(fmt, .{n})) n += 1;
        return n;
    }

    /// The configuration, read off the file the way `model_detection.py` does.
    fn config(l: Loader) !Config {
        const dim = try l.extent("perceiver.latent_queries", .{}, 1);
        if (dim % 128 != 0) return error.ShapeMismatch;
        const n_layers = l.count("layers.{d}.attention_norm1.weight");
        if (n_layers == 0 or n_layers > 64) return error.ShapeMismatch;
        var mask: u64 = 0;
        for (0..n_layers) |i| {
            if (l.has("layers.{d}.cross_attn_gate", .{i})) mask |= @as(u64, 1) << @intCast(i);
        }
        if (mask == 0) return error.ShapeMismatch;
        const first: usize = @ctz(mask);
        const ffn_dim = if (l.has("layers.0.feed_forward.w13.weight", .{}))
            try l.extent("layers.0.feed_forward.w13.weight", .{}, 0) / 2
        else
            try l.extent("layers.0.feed_forward.w1.weight", .{}, 0);
        const cap_dim = switch (l.spell) {
            .comfy => try l.extent("layers.{d}.cross_attention.to_kv.weight", .{first}, 1),
            .diffusers => try l.extent("layers.{d}.cross_attention.to_k.weight", .{first}, 1),
        };
        return .{
            .dim = dim,
            .n_heads = dim / 128,
            .n_layers = n_layers,
            .n_refiner_layers = l.count("noise_refiner.{d}.attention_norm1.weight"),
            .inject_mask = mask,
            .cap_dim = cap_dim,
            .perceiver_layers = l.count("perceiver.layers.{d}.ffn_norm.weight"),
            .max_gist = try l.extent("perceiver.latent_queries", .{}, 0),
            .ffn_dim = ffn_dim,
            .channels = try l.extent("final_layer.linear.weight", .{}, 0),
        };
    }

    fn mat(l: Loader, comptime fmt: []const u8, args: anytype, rows: usize, cols: usize) !Weight {
        var buf: [192]u8 = undefined;
        return quant_weight.load(l.alloc, l.store, try l.name(&buf, fmt, args), rows, cols, .{ .who = "fibo2" });
    }

    fn vec(l: Loader, comptime fmt: []const u8, args: anytype, len: usize) ![]f32 {
        var buf: [192]u8 = undefined;
        const nm = try l.name(&buf, fmt, args);
        const view = l.store.get(nm) orelse {
            std.log.err("fibo2: missing tensor {s}", .{nm});
            return error.MissingTensor;
        };
        if (view.info.elemCount() != len) {
            std.log.err("fibo2: {s} has {d} elements, expected {d}", .{ nm, view.info.elemCount(), len });
            return error.ShapeMismatch;
        }
        return view.toF32Alloc(l.alloc);
    }

    fn linear(l: Loader, comptime prefix: []const u8, rows: usize, cols: usize) !LinearW {
        return .{ .w = try l.mat(prefix ++ ".weight", .{}, rows, cols), .b = try l.vec(prefix ++ ".bias", .{}, rows) };
    }

    fn selfAttn(l: Loader, comptime p: []const u8, i: usize) !SelfAttn {
        const dim = l.cfg.dim;
        const hd = l.cfg.head_dim;
        return switch (l.spell) {
            .comfy => blk: {
                const qkv = try l.mat(p ++ ".attention.qkv.weight", .{i}, 3 * dim, dim);
                break :blk .{
                    .q = try lin.rowSlice(l.alloc, qkv, 0, dim),
                    .k = try lin.rowSlice(l.alloc, qkv, dim, dim),
                    .v = try lin.rowSlice(l.alloc, qkv, 2 * dim, dim),
                    .out = try l.mat(p ++ ".attention.out.weight", .{i}, dim, dim),
                    .qnorm = try l.vec(p ++ ".attention.q_norm.weight", .{i}, hd),
                    .knorm = try l.vec(p ++ ".attention.k_norm.weight", .{i}, hd),
                };
            },
            .diffusers => .{
                .q = try l.mat(p ++ ".attention.to_q.weight", .{i}, dim, dim),
                .k = try l.mat(p ++ ".attention.to_k.weight", .{i}, dim, dim),
                .v = try l.mat(p ++ ".attention.to_v.weight", .{i}, dim, dim),
                .out = try l.mat(p ++ ".attention.to_out.0.weight", .{i}, dim, dim),
                .qnorm = try l.vec(p ++ ".attention.norm_q.weight", .{i}, hd),
                .knorm = try l.vec(p ++ ".attention.norm_k.weight", .{i}, hd),
            },
        };
    }

    /// `p` names the attention module itself, e.g. `layers.{d}.cross_attention`.
    fn xAttn(l: Loader, comptime p: []const u8, i: usize, kv_dim: usize) !XAttn {
        const dim = l.cfg.dim;
        const hd = l.cfg.head_dim;
        var a: XAttn = .{
            .q = try l.mat(p ++ ".to_q.weight", .{i}, dim, dim),
            .k = undefined,
            .v = undefined,
            .out = undefined,
            .qnorm = try l.vec(p ++ ".norm_q.weight", .{i}, hd),
            .knorm = try l.vec(p ++ ".norm_k.weight", .{i}, hd),
        };
        switch (l.spell) {
            .comfy => {
                const kv = try l.mat(p ++ ".to_kv.weight", .{i}, 2 * dim, kv_dim);
                a.k = try lin.rowSlice(l.alloc, kv, 0, dim);
                a.v = try lin.rowSlice(l.alloc, kv, dim, dim);
                a.out = try l.mat(p ++ ".to_out.weight", .{i}, dim, dim);
            },
            .diffusers => {
                a.k = try l.mat(p ++ ".to_k.weight", .{i}, dim, kv_dim);
                a.v = try l.mat(p ++ ".to_v.weight", .{i}, dim, kv_dim);
                a.out = try l.mat(p ++ ".to_out.0.weight", .{i}, dim, dim);
            },
        }
        return a;
    }

    /// `p` names the FFN module, e.g. `layers.{d}.feed_forward`.
    fn ffn(l: Loader, comptime p: []const u8, i: usize) !Ffn {
        const dim = l.cfg.dim;
        const hid = l.cfg.ffn_dim;
        if (l.has(p ++ ".w13.weight", .{i})) {
            const w13 = try l.mat(p ++ ".w13.weight", .{i}, 2 * hid, dim);
            return .{
                .w1 = try lin.rowSlice(l.alloc, w13, 0, hid),
                .w3 = try lin.rowSlice(l.alloc, w13, hid, hid),
                .w2 = try l.mat(p ++ ".w2.weight", .{i}, dim, hid),
            };
        }
        return .{
            .w1 = try l.mat(p ++ ".w1.weight", .{i}, hid, dim),
            .w3 = try l.mat(p ++ ".w3.weight", .{i}, hid, dim),
            .w2 = try l.mat(p ++ ".w2.weight", .{i}, dim, hid),
        };
    }

    fn block(l: Loader, comptime p: []const u8, i: usize, cross: bool) !Block {
        const cfg = l.cfg;
        return .{
            .attn = try l.selfAttn(p, i),
            .ffn = try l.ffn(p ++ ".feed_forward", i),
            .attn_norm1 = try l.vec(p ++ ".attention_norm1.weight", .{i}, cfg.dim),
            .attn_norm2 = try l.vec(p ++ ".attention_norm2.weight", .{i}, cfg.dim),
            .ffn_norm1 = try l.vec(p ++ ".ffn_norm1.weight", .{i}, cfg.dim),
            .ffn_norm2 = try l.vec(p ++ ".ffn_norm2.weight", .{i}, cfg.dim),
            .ada = .{
                .w = try l.mat(p ++ ".adaLN_modulation.0.weight", .{i}, 4 * cfg.dim, cfg.mod_dim),
                .b = try l.vec(p ++ ".adaLN_modulation.0.bias", .{i}, 4 * cfg.dim),
            },
            .cross = if (!cross) null else .{
                .norm1 = try l.vec(p ++ ".cross_attn_norm1.weight", .{i}, cfg.dim),
                .attn = try l.xAttn(p ++ ".cross_attention", i, cfg.cap_dim),
                .norm2 = try l.vec(p ++ ".cross_attn_norm2.weight", .{i}, cfg.dim),
                .gate = std.math.tanh((try l.vec(p ++ ".cross_attn_gate", .{i}, 1))[0]),
            },
        };
    }

    fn perceiverLayer(l: Loader, i: usize) !PerceiverLayer {
        const cfg = l.cfg;
        const p = "perceiver.layers.{d}";
        return .{
            .cross_norm = try l.vec(p ++ ".cross_attn_norm.weight", .{i}, cfg.dim),
            .cross = try l.xAttn(p ++ ".cross_attention", i, cfg.cap_dim),
            .self_norm = try l.vec(p ++ ".self_attn_norm.weight", .{i}, cfg.dim),
            .self = try l.xAttn(p ++ ".self_attention", i, cfg.dim),
            .ffn_norm = try l.vec(p ++ ".ffn_norm.weight", .{i}, cfg.dim),
            .ffn = try l.ffn(p ++ ".ffn", i),
        };
    }
};

// --- tests ------------------------------------------------------------------

const testing = std.testing;
const fixture = @embedFile("assets/fibo2_forward.safetensors");

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

test "gist count follows the text length in steps of 64" {
    const cfg: Config = .{ .dim = 128, .n_heads = 1, .n_layers = 1, .n_refiner_layers = 0, .inject_mask = 1, .cap_dim = 8, .perceiver_layers = 0, .max_gist = 512, .ffn_dim = 8, .channels = 4 };
    try testing.expectEqual(@as(usize, 256), cfg.gistCount(1));
    try testing.expectEqual(@as(usize, 256), cfg.gistCount(256));
    try testing.expectEqual(@as(usize, 320), cfg.gistCount(257));
    try testing.expectEqual(@as(usize, 320), cfg.gistCount(600));
    try testing.expectEqual(@as(usize, 512), cfg.gistCount(2048));
    try testing.expectEqual(@as(usize, 512), cfg.gistCount(5000));
}

test "bundles: the Perceiver reads 0, each injection block the next" {
    const cfg: Config = .{ .dim = 128, .n_heads = 1, .n_layers = 44, .n_refiner_layers = 2, .inject_mask = (1 << 6) | (1 << 14) | (1 << 22) | (1 << 30) | (1 << 38), .cap_dim = 7680, .perceiver_layers = 2, .max_gist = 512, .ffn_dim = 9728, .channels = 128 };
    try testing.expectEqual(@as(usize, 6 * 7680), cfg.ctxDim());
    try testing.expectEqual(@as(usize, 1), cfg.bundleOf(6));
    try testing.expectEqual(@as(usize, 3), cfg.bundleOf(22));
    try testing.expectEqual(@as(usize, 5), cfg.bundleOf(38));
}

/// The fixture's ComfyUI-fused tensors respelled as diffusers' (split q/k/v and k/v,
/// split w1/w3, `to_out.0`, `norm_q`), into a fresh safetensors blob.
fn diffusersSpelling(gpa: std.mem.Allocator, st: *const safetensors.SafeTensors) ![]u8 {
    const Entry = struct { []const u8, []const u8, []const usize, []const u8 };
    var entries: std.ArrayList(Entry) = .empty;
    defer entries.deinit(gpa);
    var names: std.ArrayList([]u8) = .empty;
    defer {
        for (names.items) |n| gpa.free(n);
        names.deinit(gpa);
    }
    var shapes: std.ArrayList([]usize) = .empty;
    defer {
        for (shapes.items) |s| gpa.free(s);
        shapes.deinit(gpa);
    }
    for (st.names()) |key| {
        if (std.mem.startsWith(u8, key, "case.") or std.mem.startsWith(u8, key, "cfg.")) continue;
        const view = st.get(key).?;
        const shp = view.info.shape.slice();
        const bytes = view.bytes;
        const Part = struct { suffix: []const u8, parts: []const []const u8 };
        const splits = [_]Part{
            .{ .suffix = ".attention.qkv.weight", .parts = &.{ ".attention.to_q.weight", ".attention.to_k.weight", ".attention.to_v.weight" } },
            .{ .suffix = "_attention.to_kv.weight", .parts = &.{ "_attention.to_k.weight", "_attention.to_v.weight" } },
            .{ .suffix = ".w13.weight", .parts = &.{ ".w1.weight", ".w3.weight" } },
        };
        const renames = [_][2][]const u8{
            .{ ".attention.out.weight", ".attention.to_out.0.weight" },
            .{ "_attention.to_out.weight", "_attention.to_out.0.weight" },
            .{ ".attention.q_norm.weight", ".attention.norm_q.weight" },
            .{ ".attention.k_norm.weight", ".attention.norm_k.weight" },
        };
        var done = false;
        for (splits) |sp| {
            if (!std.mem.endsWith(u8, key, sp.suffix)) continue;
            const stem = key[0 .. key.len - sp.suffix.len];
            const rows = shp[0] / sp.parts.len;
            const row_bytes = bytes.len / shp[0];
            for (sp.parts, 0..) |pt, pi| {
                const nm = try std.mem.concat(gpa, u8, &.{ stem, pt });
                try names.append(gpa, nm);
                const s = try gpa.dupe(usize, &.{ rows, shp[1] });
                try shapes.append(gpa, s);
                try entries.append(gpa, .{ nm, "BF16", s, bytes[pi * rows * row_bytes ..][0 .. rows * row_bytes] });
            }
            done = true;
        }
        if (done) continue;
        var nm = try gpa.dupe(u8, key);
        for (renames) |r| if (std.mem.endsWith(u8, key, r[0])) {
            gpa.free(nm);
            nm = try std.mem.concat(gpa, u8, &.{ key[0 .. key.len - r[0].len], r[1] });
        };
        try names.append(gpa, nm);
        const s = try gpa.dupe(usize, shp);
        try shapes.append(gpa, s);
        try entries.append(gpa, .{ nm, "BF16", s, bytes });
    }
    return lora_mod.safetensorsBlob(gpa, entries.items);
}

fn checkCases(model: *const DiT, st: *const safetensors.SafeTensors, stages: bool) !void {
    const gpa = testing.allocator;
    const io = testing.io;
    const cfg = model.cfg;
    for ([_][]const u8{ "a", "b" }) |cs| {
        var kb: [64]u8 = undefined;
        const get = struct {
            fn f(s: *const safetensors.SafeTensors, b: []u8, c: []const u8, what: []const u8) ![]f32 {
                return (try s.require(try std.mem.print(b, "case.{s}.{s}", .{ c, what }))).toF32Alloc(testing.allocator);
            }
        }.f;
        const x = try get(st, &kb, cs, "x");
        defer gpa.free(x);
        const ctx = try get(st, &kb, cs, "ctx");
        defer gpa.free(ctx);
        const sig = try get(st, &kb, cs, "sigma");
        defer gpa.free(sig);
        const want = try get(st, &kb, cs, "out");
        defer gpa.free(want);
        const text_len = ctx.len / cfg.ctxDim();
        const xinfo = (try st.require(try std.mem.print(&kb, "case.{s}.x", .{cs}))).info.shape.slice();
        const lat_h = xinfo[1];
        const lat_w = xinfo[2];

        var pre = try model.prefix(io, gpa, ctx, text_len);
        defer pre.deinit(gpa);
        if (stages) {
            const t_mixed = try get(st, &kb, cs, "t_mixed");
            defer gpa.free(t_mixed);
            const a0 = try model.adalnInput(io, gpa, 0);
            defer gpa.free(a0);
            const a1 = try model.adalnInput(io, gpa, sig[0]);
            defer gpa.free(a1);
            const r0 = relL2(t_mixed[0..cfg.mod_dim], a0);
            const r1 = relL2(t_mixed[cfg.mod_dim..], a1);
            errdefer std.debug.print("case {s}: t_emb rel L2 {e} / {e}\n", .{ cs, r0, r1 });
            try testing.expect(r0 < 1e-5 and r1 < 1e-5);

            const gist = try get(st, &kb, cs, "gist");
            defer gpa.free(gist);
            try testing.expectEqual(gist.len, pre.gist.len);
            const rg = relL2(gist, pre.gist);
            errdefer std.debug.print("case {s}: gist rel L2 {e}\n", .{ cs, rg });
            try testing.expect(rg < 1e-5);
        }

        const got = try gpa.alloc(f32, x.len);
        defer gpa.free(got);
        try model.predict(io, gpa, got, x, lat_h, lat_w, sig[0], &pre, null);
        const rel = relL2(want, got);
        errdefer std.debug.print("case {s} ({d}x{d}, {d} tokens, {d} gists): velocity rel L2 {e}\n", .{ cs, lat_h, lat_w, text_len, pre.n_gist, rel });
        try testing.expect(rel < 1e-5);
    }
}

test "the Fibo 2.0 DiT matches ComfyUI's on the toy fixture, both spellings" {
    const gpa = testing.allocator;
    var st = try safetensors.SafeTensors.initFromSlice(gpa, fixture);
    defer st.deinit();

    var model = try DiT.load(gpa, .{ .safetensors = &st });
    defer model.deinit();
    try testing.expectEqual(@as(usize, 3), model.layers.len);
    try testing.expectEqual(@as(u64, 0b101), model.cfg.inject_mask);
    try testing.expectEqual(@as(usize, 48), model.cfg.cap_dim);
    try testing.expectEqual(@as(usize, 32), model.cfg.channels);
    try checkCases(&model, &st, true);

    const blob = try diffusersSpelling(gpa, &st);
    defer gpa.free(blob);
    var st2 = try safetensors.SafeTensors.initFromSlice(gpa, blob);
    defer st2.deinit();
    var model2 = try DiT.load(gpa, .{ .safetensors = &st2 });
    defer model2.deinit();
    try testing.expect(model2.layers[0].attn.q.row_off == 0);
    try testing.expect(model2.layers[0].ffn.w1.rows == model.cfg.ffn_dim);
    // Cases from the first file: the second has none.
    try checkCasesFrom(&model2, &st);
}

fn checkCasesFrom(model: *const DiT, cases: *const safetensors.SafeTensors) !void {
    try checkCases(model, cases, false);
}

test "the Fibo 2.0 dev files load with the release configuration" {
    const gpa = testing.allocator;
    const io = testing.io;
    const test_gate = @import("../test_gate.zig");
    const root = @import("build_options").model_root;
    try test_gate.requireIntegration();
    for ([_][]const u8{ root ++ "/diffusion_models/fibo/fibo2_dev_bf16.safetensors", root ++ "/diffusion_models/fibo/fibo2_dev_int8_convrot.safetensors" }) |path| {
        try test_gate.requireModelFile(io, path);
        var st = try safetensors.SafeTensors.open(gpa, io, path);
        defer st.deinit();
        var model = try DiT.load(gpa, .{ .safetensors = &st });
        defer model.deinit();
        errdefer std.debug.print("{s}: {any}\n", .{ path, model.cfg });
        try testing.expectEqual(release, model.cfg);
        try testing.expectEqual(@as(usize, 2), model.perceiver.len);
    }
}
