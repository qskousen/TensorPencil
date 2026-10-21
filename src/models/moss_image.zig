//! MOSS-Image, a single-stream DiT, mirroring comfy/ldm/moss_image/model.py.
//!
//! One joint stream of `[image | text]` tokens, IMAGE FIRST, through every block.
//! A token is one latent pixel (patch 1) of Flux 2's 128-channel packed latent.
//! Conventions the checkpoint does not record, each a silent wrong answer:
//!
//!   - ONE modulation for every block (`shared_block_modulation`), read straight
//!     off the timestep MLP's output with no SiLU in front, chunked
//!     `[shift_a, scale_a, gate_a, shift_f, scale_f, gate_f]`. The head's own
//!     modulation is `[shift, scale]`.
//!   - The modulated norm is the block's weighted RMSNorm, then `(1 + scale)`,
//!     then `+ shift`: the norm weight is not folded away.
//!   - RoPE has four axes (r, f, h, w) of `head_dim / 4` each, concatenated, and
//!     rotates SPLIT-HALF (x[i] with x[i + head_dim / 2]), not adjacent pairs.
//!     Text token t sits at (0, t, 0, 0); every image token at f = text_len, with
//!     row and column centered on `(n - 1) / 2`, so even sizes get half-integer
//!     positions.
//!   - The FFN's fused up projection is `[hidden | gate]`, and `situ` soft-caps
//!     both: `4 tanh(g / 4) sigmoid(g) * 25 tanh(h / 25)`.
//!   - The text runs through two unmodulated refiner blocks first, rotated by
//!     its own index alone.
//!
//! Loads from any `WeightStore`. Block q/k/v are fused (`proj.qkv`, repacked
//! files) or split (`proj.q`, `.k`, `.v`); both run the same forward.
//!
//! ComfyUI renders this checkpoint with int8 attention, which alone moves the image
//! far more than we differ from it: compare renders against
//! `tools/gen_moss_image_render.py --no-int8-attention`.

const std = @import("std");
const build_options = @import("build_options");
const tp_core = @import("tp_core");
const weights_mod = tp_core.weights;
const ops = @import("tp_ops");
const quant_weight = @import("quant_weight.zig");
const lin = @import("lin.zig");
const lora_mod = @import("lora.zig");

const WeightStore = weights_mod.WeightStore;
const Weight = ops.matmul.Weight;

/// The released checkpoint's shape, for a catalog that has only a side file to
/// look at. `Config.detect` reads the real one off the DiT.
pub const features = 3072;
pub const n_blocks = 42;
/// Qwen3.5-9B's width.
pub const txt_dim = 4096;
/// Sinusoidal frequency count feeding the timestep MLP.
pub const tdim = 256;
pub const rope_theta: f64 = 10000.0;
/// Every norm in this model uses the same epsilon.
const eps: f32 = 1e-6;
/// Upper bound on the block count, so `Config.detect`'s probe loop terminates on
/// a file that is not this architecture.
const max_blocks = 128;

/// The tensors that say "this is a MOSS-Image denoiser" (ComfyUI's detection).
pub const probe = "shared_block_modulation.weight";
pub const probe2 = "layers.0.proj.ffn_up_gate.weight";

pub const Config = struct {
    dim: usize,
    ffn: usize,
    heads: usize,
    layers: usize,
    txt_dim: usize,
    channels: usize,
    fused_qkv: bool,

    pub fn headDim(c: Config) usize {
        return c.dim / c.heads;
    }

    /// Read off the shapes, as ComfyUI's `detect_unet_config` does: heads are
    /// `dim / 128`.
    pub fn detect(store: WeightStore) !Config {
        const pfx = prefixIn(store);
        var buf: [128]u8 = undefined;
        const mod = store.get(try fmt(&buf, pfx, probe)) orelse return error.UnknownModelConfig;
        const dim = mod.info.shape.slice()[0] / 6;
        const ug = store.get(try fmt(&buf, pfx, probe2)) orelse return error.UnknownModelConfig;
        const proj = store.get(try fmt(&buf, pfx, "text_embedder.proj.weight")) orelse return error.UnknownModelConfig;
        const emb = store.get(try fmt(&buf, pfx, "video_embedder.weight")) orelse return error.UnknownModelConfig;
        if (dim == 0 or dim % 128 != 0) return error.UnknownModelConfig;
        var layers: usize = 0;
        while (layers < max_blocks) : (layers += 1) {
            const nm = std.mem.print(&buf, "{s}layers.{d}.norm1.weight", .{ pfx, layers }) catch break;
            if (store.get(nm) == null) break;
        }
        return .{
            .dim = dim,
            .ffn = ug.info.shape.slice()[0] / 2,
            .heads = dim / 128,
            .layers = layers,
            .txt_dim = proj.info.shape.slice()[1],
            .channels = emb.info.shape.slice()[1],
            .fused_qkv = store.get(try fmt(&buf, pfx, "layers.0.proj.qkv.weight")) != null,
        };
    }

    fn fmt(buf: []u8, pfx: []const u8, name: []const u8) ![]const u8 {
        return std.mem.print(buf, "{s}{s}", .{ pfx, name });
    }
};

/// The prefix this store spells the denoiser with: a bundled checkpoint nests it.
pub fn prefixIn(store: WeightStore) []const u8 {
    return if (store.get("model.diffusion_model." ++ probe) != null) "model.diffusion_model." else "";
}

pub const LinearW = struct {
    w: Weight,
    b: ?[]const f32 = null,
};

pub const Block = struct {
    norm1: []const f32,
    norm2: []const f32,
    q_norm: []const f32,
    k_norm: []const f32,
    /// `[3 * dim][dim]` when fused; null split, where `q`/`k`/`v` hold it.
    qkv: ?LinearW,
    q: ?LinearW,
    k: ?LinearW,
    v: ?LinearW,
    o: LinearW,
    /// `[2 * ffn][dim]`, rows `[hidden | gate]`.
    up_gate: LinearW,
    down: LinearW,
    /// What the device runs, in forward order: q, k, v, o, hidden, gate, down. The
    /// fused tensors appear as row views (`lin.rowSlice`).
    dev: [7]Weight,
};

/// An unmodulated pre-norm block over the text alone, with biases.
pub const TextBlock = struct {
    norm1: []const f32,
    norm2: []const f32,
    q_norm: []const f32,
    k_norm: []const f32,
    qkv: LinearW,
    proj: LinearW,
    gate: LinearW,
    up: LinearW,
    down: LinearW,
};

/// The per-timestep vectors every block and the head read.
pub const Mod = struct {
    /// `[6][dim]`: shift_a, scale_a, gate_a, shift_f, scale_f, gate_f.
    block: []f32,
    /// `[2][dim]`: shift, scale.
    head: []f32,

    pub fn deinit(m: *Mod, gpa: std.mem.Allocator) void {
        gpa.free(m.block);
        gpa.free(m.head);
        m.* = undefined;
    }

    pub fn chunk(m: Mod, i: usize) []const f32 {
        const d = m.block.len / 6;
        return m.block[i * d ..][0..d];
    }
};

pub const DiT = struct {
    arena: std.heap.ArenaAllocator,
    cfg: Config,
    video_embedder: LinearW,
    txt_proj: LinearW,
    txt_blocks: [2]TextBlock,
    time0: LinearW,
    time2: LinearW,
    shared_mod: LinearW,
    head_mod: LinearW,
    blocks: []Block,
    final_norm: []const f32,
    final_linear: LinearW,
    /// Every block linear the device forwards run, the one list every support scan
    /// and GEMM plan reads.
    device_lins: []const Weight,
    /// `video_embedder` and `final_linear_video`, which the device arms also run.
    edge_lins: []const Weight,
    lora: ?*const lora_mod.Stack = null,

    pub fn load(gpa: std.mem.Allocator, store: WeightStore) !DiT {
        return loadWith(gpa, store, try Config.detect(store));
    }

    pub fn loadWith(gpa: std.mem.Allocator, store: WeightStore, cfg: Config) !DiT {
        if (cfg.layers == 0 or cfg.dim % cfg.heads != 0 or cfg.headDim() % 8 != 0) return error.UnknownModelConfig;
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();
        const l = Loader{ .store = store, .alloc = alloc, .pfx = prefixIn(store), .cfg = cfg };

        const blocks = try alloc.alloc(Block, cfg.layers);
        for (blocks, 0..) |*b, i| b.* = try l.block(i);
        const lins = try alloc.alloc(Weight, blocks.len * 7);
        for (blocks, 0..) |*b, i| lins[i * 7 ..][0..7].* = b.dev;

        const d = cfg.dim;
        // f32 at load: the device's patch embed takes the f32 GEMM arm, as Z-Image's does.
        var video_embedder = try l.linear("video_embedder", .{}, d, cfg.channels, true);
        video_embedder.w = try ops.matmul.materializeF32(alloc, video_embedder.w);
        const txt_proj = try l.linear("text_embedder.proj", .{}, d, cfg.txt_dim, true);
        const txt_blocks = [2]TextBlock{ try l.textBlock(0), try l.textBlock(1) };
        const time0 = try l.linear("time_embedding.0", .{}, d, tdim, true);
        const time2 = try l.linear("time_embedding.2", .{}, d, d, true);
        const shared_mod = try l.linear("shared_block_modulation", .{}, 6 * d, d, true);
        const head_mod = try l.linear("video_head_modulation", .{}, 2 * d, d, true);
        const final_norm = try l.vec("final_norm_video.weight", .{}, d);
        var final_linear = try l.linear("final_linear_video", .{}, cfg.channels, d, false);
        final_linear.w = try ops.matmul.materializeF32(alloc, final_linear.w);
        // The device linears outside the blocks, which a LoRA can also target.
        const edge_lins = try alloc.alloc(Weight, 2);
        edge_lins[0..2].* = .{ video_embedder.w, final_linear.w };

        return .{
            .arena = arena,
            .cfg = cfg,
            .video_embedder = video_embedder,
            .txt_proj = txt_proj,
            .txt_blocks = txt_blocks,
            .time0 = time0,
            .time2 = time2,
            .shared_mod = shared_mod,
            .head_mod = head_mod,
            .blocks = blocks,
            .final_norm = final_norm,
            .final_linear = final_linear,
            .device_lins = lins,
            .edge_lins = edge_lins,
        };
    }

    pub fn deinit(self: *DiT) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// Resolve `stack` against every linear this model runs and hold it. Returns
    /// the file keys that apply.
    pub fn attachLora(self: *DiT, gpa: std.mem.Allocator, stack: *lora_mod.Stack) !usize {
        self.lora = stack;
        stack.beginAttach();
        try stack.resolveAll(self.device_lins);
        for (self.blocks) |*b| {
            if (b.qkv) |f| _ = try stack.resolve(f.w);
            _ = try stack.resolve(b.up_gate.w);
        }
        for (&self.txt_blocks) |*b| try stack.resolveAll(&.{ b.qkv.w, b.proj.w, b.gate.w, b.up.w, b.down.w });
        try stack.resolveAll(&.{
            self.video_embedder.w, self.txt_proj.w, self.time0.w,        self.time2.w,
            self.shared_mod.w,     self.head_mod.w, self.final_linear.w,
        });
        return stack.finishAttach(gpa);
    }

    /// Velocity for one latent. `x_lat` / `out` are planar `[channels][lat_h][lat_w]`,
    /// `ctx` is `[seq_txt][txt_dim]`, `sigma` the flow-matching timestep.
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
        const cfg = self.cfg;
        const d = cfg.dim;
        const n_img = lat_h * lat_w;
        std.debug.assert(x_lat.len == cfg.channels * n_img and out.len == x_lat.len);
        std.debug.assert(ctx.len == seq_txt * cfg.txt_dim);

        const prev_tok = ops.cancel.token;
        ops.cancel.token = cancel;
        defer ops.cancel.token = prev_tok;

        var mod = try self.modulation(io, gpa, sigma);
        defer mod.deinit(gpa);

        const seq = n_img + seq_txt;
        const x = try gpa.alloc(f32, seq * d);
        defer gpa.free(x);
        {
            const tokens = try gpa.alloc(f32, n_img * cfg.channels);
            defer gpa.free(tokens);
            interleaveChannels(tokens, x_lat, n_img, cfg.channels);
            try linear(io, gpa, self.lora, x[0 .. n_img * d], tokens, n_img, self.video_embedder);
        }
        {
            const txt = try self.textTokens(io, gpa, ctx, seq_txt);
            defer gpa.free(txt);
            @memcpy(x[n_img * d ..], txt);
        }

        var freqs = try ropeFreqs(gpa, cfg, lat_h, lat_w, seq_txt);
        defer freqs.deinit(gpa);

        for (self.blocks) |*blk| {
            if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
            try blockForward(io, gpa, self.lora, cfg, blk, x, seq, mod, freqs);
        }

        const img = x[0 .. n_img * d];
        modRmsNorm(img, img, self.final_norm, mod.head[d..][0..d], mod.head[0..d]);
        const fin = try gpa.alloc(f32, n_img * cfg.channels);
        defer gpa.free(fin);
        try linear(io, gpa, self.lora, fin, img, n_img, self.final_linear);
        deinterleaveChannels(out, fin, n_img, cfg.channels);
    }

    /// `time_embedding(sinusoid(sigma))` through the shared block modulation and
    /// the head's. The host builds these once per sigma for every backend.
    pub fn modulation(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, sigma: f32) !Mod {
        const d = self.cfg.dim;
        var freq: [tdim]f32 = undefined;
        ops.rope.timestepEmbedding(&freq, sigma, 10000.0);
        const t0 = try gpa.alloc(f32, d);
        defer gpa.free(t0);
        try linear(io, gpa, self.lora, t0, &freq, 1, self.time0);
        ops.act.silu(t0);
        const vec = try gpa.alloc(f32, d);
        defer gpa.free(vec);
        try linear(io, gpa, self.lora, vec, t0, 1, self.time2);

        const block = try gpa.alloc(f32, 6 * d);
        errdefer gpa.free(block);
        try linear(io, gpa, self.lora, block, vec, 1, self.shared_mod);
        const head = try gpa.alloc(f32, 2 * d);
        errdefer gpa.free(head);
        try linear(io, gpa, self.lora, head, vec, 1, self.head_mod);
        return .{ .block = block, .head = head };
    }

    /// `modulation` laid out for the device's `rmsMod` (`x * inv_rms * premul + shift`):
    /// rows of `dim` at the `Table` offsets, each norm weight FOLDED into its premul
    /// as `w * (1 + scale)`. Caller frees.
    pub fn foldedTable(self: *const DiT, gpa: std.mem.Allocator, mod: Mod) ![]f32 {
        const d = self.cfg.dim;
        const out = try gpa.alloc(f32, Table.len(self.blocks.len) * d);
        const row = struct {
            fn f(o: []f32, dd: usize, i: usize) []f32 {
                return o[i * dd ..][0..dd];
            }
        }.f;
        @memcpy(row(out, d, Table.shift_a), mod.chunk(0));
        @memcpy(row(out, d, Table.gate_a), mod.chunk(2));
        @memcpy(row(out, d, Table.shift_f), mod.chunk(3));
        @memcpy(row(out, d, Table.gate_f), mod.chunk(5));
        @memcpy(row(out, d, Table.head_shift), mod.head[0..d]);
        fold(row(out, d, Table.head_premul), self.final_norm, mod.head[d..][0..d]);
        for (self.blocks, 0..) |*b, i| {
            fold(row(out, d, Table.premul1(i)), b.norm1, mod.chunk(1));
            fold(row(out, d, Table.premul2(i)), b.norm2, mod.chunk(4));
        }
        return out;
    }

    /// The text through `text_embedder`: `[seq_txt][dim]`, caller frees. It reads
    /// no timestep, so a sampler runs it once per conditioning.
    pub fn textTokens(self: *const DiT, io: std.Io, gpa: std.mem.Allocator, ctx: []const f32, seq_txt: usize) ![]f32 {
        const cfg = self.cfg;
        const d = cfg.dim;
        const x = try gpa.alloc(f32, seq_txt * d);
        errdefer gpa.free(x);
        try linear(io, gpa, self.lora, x, ctx, seq_txt, self.txt_proj);
        var freqs = try textRopeFreqs(gpa, cfg, seq_txt);
        defer freqs.deinit(gpa);
        for (&self.txt_blocks) |*b| try textBlockForward(io, gpa, self.lora, cfg, b, x, seq_txt, freqs);
        return x;
    }
};

/// Row offsets, in units of `dim`, into `DiT.foldedTable`.
pub const Table = struct {
    pub const shift_a = 0;
    pub const gate_a = 1;
    pub const shift_f = 2;
    pub const gate_f = 3;
    pub const head_shift = 4;
    pub const head_premul = 5;
    pub fn premul1(block: usize) usize {
        return 6 + 2 * block;
    }
    pub fn premul2(block: usize) usize {
        return 7 + 2 * block;
    }
    pub fn len(blocks: usize) usize {
        return 6 + 2 * blocks;
    }
};

fn fold(out: []f32, w: []const f32, scale: []const f32) void {
    for (out, w, scale) |*o, wv, sc| o.* = wv * (1.0 + sc);
}

/// The joint table, `[image | text]` rows. Image token (r, c) sits at
/// (0, seq_txt, r - (h - 1) / 2, c - (w - 1) / 2), text token t at (0, t, 0, 0).
pub fn ropeFreqs(gpa: std.mem.Allocator, cfg: Config, lat_h: usize, lat_w: usize, seq_txt: usize) !ops.rope.Freqs {
    const n_img = lat_h * lat_w;
    const pos = try gpa.alloc(f32, (n_img + seq_txt) * 4);
    defer gpa.free(pos);
    @memset(pos, 0);
    const ch: f32 = @as(f32, @floatFromInt(lat_h - 1)) / 2;
    const cw: f32 = @as(f32, @floatFromInt(lat_w - 1)) / 2;
    const f: f32 = @floatFromInt(seq_txt);
    for (0..lat_h) |r| for (0..lat_w) |c| {
        const p = pos[(r * lat_w + c) * 4 ..][0..4];
        p[1] = f;
        p[2] = @as(f32, @floatFromInt(r)) - ch;
        p[3] = @as(f32, @floatFromInt(c)) - cw;
    };
    for (0..seq_txt) |t| pos[(n_img + t) * 4 + 1] = @floatFromInt(t);
    const a = cfg.headDim() / 4;
    return ops.rope.fluxFreqs(gpa, pos, &.{ a, a, a, a }, rope_theta);
}

/// The text rows of `ropeFreqs`, which do not depend on the image.
pub fn textRopeFreqs(gpa: std.mem.Allocator, cfg: Config, seq_txt: usize) !ops.rope.Freqs {
    const pos = try gpa.alloc(f32, seq_txt * 4);
    defer gpa.free(pos);
    @memset(pos, 0);
    for (0..seq_txt) |t| pos[t * 4 + 1] = @floatFromInt(t);
    const a = cfg.headDim() / 4;
    return ops.rope.fluxFreqs(gpa, pos, &.{ a, a, a, a }, rope_theta);
}

fn blockForward(
    io: std.Io,
    gpa: std.mem.Allocator,
    stack: ?*const lora_mod.Stack,
    cfg: Config,
    blk: *const Block,
    x: []f32,
    seq: usize,
    mod: Mod,
    freqs: ops.rope.Freqs,
) !void {
    const d = cfg.dim;
    const y = try gpa.alloc(f32, seq * d);
    defer gpa.free(y);
    const qkv = try gpa.alloc(f32, 3 * seq * d);
    defer gpa.free(qkv);
    const q = qkv[0 .. seq * d];
    const k = qkv[seq * d ..][0 .. seq * d];
    const v = qkv[2 * seq * d ..];

    modRmsNorm(y, x, blk.norm1, mod.chunk(1), mod.chunk(0));
    if (blk.qkv) |f| {
        const fused = try gpa.alloc(f32, seq * 3 * d);
        defer gpa.free(fused);
        try linear(io, gpa, stack, fused, y, seq, f);
        splitRows(fused, &.{ q, k, v }, seq, d);
    } else {
        try linear(io, gpa, stack, q, y, seq, blk.q.?);
        try linear(io, gpa, stack, k, y, seq, blk.k.?);
        try linear(io, gpa, stack, v, y, seq, blk.v.?);
    }
    const att = try attend(io, gpa, cfg, q, k, v, blk.q_norm, blk.k_norm, seq, freqs);
    defer gpa.free(att);
    try linear(io, gpa, stack, y, att, seq, blk.o);
    gatedAdd(x, y, mod.chunk(2));

    modRmsNorm(y, x, blk.norm2, mod.chunk(4), mod.chunk(3));
    const inner = try gpa.alloc(f32, seq * 2 * cfg.ffn);
    defer gpa.free(inner);
    try linear(io, gpa, stack, inner, y, seq, blk.up_gate);
    const act = try gpa.alloc(f32, seq * cfg.ffn);
    defer gpa.free(act);
    for (0..seq) |r| {
        const row = inner[r * 2 * cfg.ffn ..][0 .. 2 * cfg.ffn];
        situ(act[r * cfg.ffn ..][0..cfg.ffn], row[0..cfg.ffn], row[cfg.ffn..]);
    }
    try linear(io, gpa, stack, y, act, seq, blk.down);
    gatedAdd(x, y, mod.chunk(5));
}

fn textBlockForward(
    io: std.Io,
    gpa: std.mem.Allocator,
    stack: ?*const lora_mod.Stack,
    cfg: Config,
    blk: *const TextBlock,
    x: []f32,
    seq: usize,
    freqs: ops.rope.Freqs,
) !void {
    const d = cfg.dim;
    const y = try gpa.alloc(f32, seq * d);
    defer gpa.free(y);
    const qkv = try gpa.alloc(f32, 3 * seq * d);
    defer gpa.free(qkv);
    const fused = try gpa.alloc(f32, seq * 3 * d);
    defer gpa.free(fused);

    ops.norm.rmsNorm(y, x, blk.norm1, eps);
    try linear(io, gpa, stack, fused, y, seq, blk.qkv);
    const q = qkv[0 .. seq * d];
    const k = qkv[seq * d ..][0 .. seq * d];
    const v = qkv[2 * seq * d ..];
    splitRows(fused, &.{ q, k, v }, seq, d);
    const att = try attend(io, gpa, cfg, q, k, v, blk.q_norm, blk.k_norm, seq, freqs);
    defer gpa.free(att);
    try linear(io, gpa, stack, y, att, seq, blk.proj);
    for (x, y) |*a, b| a.* += b;

    ops.norm.rmsNorm(y, x, blk.norm2, eps);
    const hidden = try gpa.alloc(f32, seq * cfg.ffn);
    defer gpa.free(hidden);
    const gate = try gpa.alloc(f32, seq * cfg.ffn);
    defer gpa.free(gate);
    try linear(io, gpa, stack, hidden, y, seq, blk.up);
    try linear(io, gpa, stack, gate, y, seq, blk.gate);
    situ(hidden, hidden, gate);
    try linear(io, gpa, stack, y, hidden, seq, blk.down);
    for (x, y) |*a, b| a.* += b;
}

/// Per-head QK norm, split-half RoPE, full attention. Caller frees the result.
fn attend(
    io: std.Io,
    gpa: std.mem.Allocator,
    cfg: Config,
    q: []f32,
    k: []f32,
    v: []const f32,
    q_norm: []const f32,
    k_norm: []const f32,
    seq: usize,
    freqs: ops.rope.Freqs,
) ![]f32 {
    const hd = cfg.headDim();
    ops.norm.rmsNorm(q, q, q_norm, eps);
    ops.norm.rmsNorm(k, k, k_norm, eps);
    ops.rope.applyRotateHalf(q, freqs, seq, cfg.heads, hd);
    ops.rope.applyRotateHalf(k, freqs, seq, cfg.heads, hd);
    const out = try gpa.alloc(f32, seq * cfg.dim);
    errdefer gpa.free(out);
    try ops.attention.attention(io, gpa, out, q, k, v, .{
        .seq_q = seq,
        .seq_kv = seq,
        .n_heads = cfg.heads,
        .n_kv_heads = cfg.heads,
        .head_dim = hd,
    });
    return out;
}

/// `[rows][parts.len * d]` -> each part `[rows][d]`.
fn splitRows(src: []const f32, parts: []const []f32, rows: usize, d: usize) void {
    const w = parts.len * d;
    for (0..rows) |r| for (parts, 0..) |p, i| {
        @memcpy(p[r * d ..][0..d], src[r * w + i * d ..][0..d]);
    };
}

/// `4 tanh(g / 4) sigmoid(g) * 25 tanh(h / 25)`. `out` may alias `hidden`.
pub fn situ(out: []f32, hidden: []const f32, gate: []const f32) void {
    for (out, hidden, gate) |*o, h, g| {
        const sg = 1.0 / (1.0 + @exp(-g));
        o.* = (4.0 * std.math.tanh(g / 4.0) * sg) * (25.0 * std.math.tanh(h / 25.0));
    }
}

/// `rmsnorm(x) * w * (1 + scale) + shift`, row-wise. `out` may alias `x`.
fn modRmsNorm(out: []f32, x: []const f32, w: []const f32, scale: []const f32, shift: []const f32) void {
    ops.norm.rmsNorm(out, x, w, eps);
    const d = w.len;
    var row: usize = 0;
    while (row < out.len) : (row += d) {
        for (out[row..][0..d], scale, shift) |*v, sc, sh| v.* = v.* * (1.0 + sc) + sh;
    }
}

/// Row-wise gated residual: x += gate * delta.
fn gatedAdd(x: []f32, delta: []const f32, gate: []const f32) void {
    const d = gate.len;
    var row: usize = 0;
    while (row < x.len) : (row += d) {
        for (x[row..][0..d], delta[row..][0..d], gate) |*v, dl, g| v.* += g * dl;
    }
}

/// Planar `[c][n]` -> token-major `[n][c]`.
pub fn interleaveChannels(dst: []f32, src: []const f32, n: usize, c: usize) void {
    for (0..c) |ci| for (src[ci * n ..][0..n], 0..) |v, i| {
        dst[i * c + ci] = v;
    };
}

/// Token-major `[n][c]` -> planar `[c][n]`.
pub fn deinterleaveChannels(dst: []f32, src: []const f32, n: usize, c: usize) void {
    for (0..c) |ci| for (dst[ci * n ..][0..n], 0..) |*v, i| {
        v.* = src[i * c + ci];
    };
}

/// Every host GEMM on a model weight goes through here, so a LoRA cannot be
/// missed at one call site.
fn linear(io: std.Io, gpa: std.mem.Allocator, stack: ?*const lora_mod.Stack, out: []f32, x: []const f32, m: usize, lw: LinearW) !void {
    try ops.matmul.matmul(io, gpa, out, x, m, lw.w, lw.b);
    if (stack) |st| try st.applyHost(io, gpa, out, lw.w.rows, x, m, lw.w, lw.b);
}

/// LoRAs address the split q/k/v (ComfyUI's MOSS-Image key map); on a fused file
/// each lands on its third of `proj.qkv`.
pub const lora_keys_fused = [_]lora_mod.Rename{
    .{ .from = ".proj.q", .to = ".proj.qkv", .slot = 0, .slots = 3 },
    .{ .from = ".proj.k", .to = ".proj.qkv", .slot = 1, .slots = 3 },
    .{ .from = ".proj.v", .to = ".proj.qkv", .slot = 2, .slots = 3 },
};

const Loader = struct {
    store: WeightStore,
    alloc: std.mem.Allocator,
    pfx: []const u8,
    cfg: Config,

    fn name(l: Loader, buf: []u8, comptime f: []const u8, args: anytype) ![]u8 {
        var w = std.Io.Writer.fixed(buf);
        try w.writeAll(l.pfx);
        try w.print(f, args);
        return w.buffered();
    }

    fn vec(l: Loader, comptime f: []const u8, args: anytype, len: usize) ![]f32 {
        var buf: [160]u8 = undefined;
        const nm = try l.name(&buf, f, args);
        const view = l.store.get(nm) orelse {
            std.log.err("moss-image: missing tensor {s}", .{nm});
            return error.MissingTensor;
        };
        if (view.info.elemCount() != len) return error.ShapeMismatch;
        return view.toF32Alloc(l.alloc);
    }

    fn linear(l: Loader, comptime f: []const u8, args: anytype, rows: usize, cols: usize, bias: bool) !LinearW {
        var buf: [160]u8 = undefined;
        const nm = try l.name(&buf, f ++ ".weight", args);
        return .{
            .w = try quant_weight.load(l.alloc, l.store, nm, rows, cols, .{ .who = "moss-image" }),
            .b = if (bias) try l.vec(f ++ ".bias", args, rows) else null,
        };
    }

    fn block(l: Loader, i: usize) !Block {
        var b = try l.blockWeights(i);
        const d = l.cfg.dim;
        const ffn = l.cfg.ffn;
        const q, const k, const v = if (b.qkv) |f|
            .{ try lin.rowSlice(l.alloc, f.w, 0, d), try lin.rowSlice(l.alloc, f.w, d, d), try lin.rowSlice(l.alloc, f.w, 2 * d, d) }
        else
            .{ b.q.?.w, b.k.?.w, b.v.?.w };
        b.dev = .{
            q,
            k,
            v,
            b.o.w,
            try lin.rowSlice(l.alloc, b.up_gate.w, 0, ffn),
            try lin.rowSlice(l.alloc, b.up_gate.w, ffn, ffn),
            b.down.w,
        };
        return b;
    }

    fn blockWeights(l: Loader, i: usize) !Block {
        const d = l.cfg.dim;
        const hd = l.cfg.headDim();
        const fused = l.cfg.fused_qkv;
        return .{
            .norm1 = try l.vec("layers.{d}.norm1.weight", .{i}, d),
            .norm2 = try l.vec("layers.{d}.norm2.weight", .{i}, d),
            .q_norm = try l.vec("layers.{d}.q_norm.weight", .{i}, hd),
            .k_norm = try l.vec("layers.{d}.k_norm.weight", .{i}, hd),
            .qkv = if (fused) try l.linear("layers.{d}.proj.qkv", .{i}, 3 * d, d, false) else null,
            .q = if (fused) null else try l.linear("layers.{d}.proj.q", .{i}, d, d, false),
            .k = if (fused) null else try l.linear("layers.{d}.proj.k", .{i}, d, d, false),
            .v = if (fused) null else try l.linear("layers.{d}.proj.v", .{i}, d, d, false),
            .o = try l.linear("layers.{d}.proj.o", .{i}, d, d, false),
            .up_gate = try l.linear("layers.{d}.proj.ffn_up_gate", .{i}, 2 * l.cfg.ffn, d, false),
            .down = try l.linear("layers.{d}.proj.ffn_down", .{i}, d, l.cfg.ffn, false),
            .dev = undefined,
        };
    }

    fn textBlock(l: Loader, i: usize) !TextBlock {
        const d = l.cfg.dim;
        const hd = l.cfg.headDim();
        const ffn = l.cfg.ffn;
        return .{
            .norm1 = try l.vec("text_embedder.blocks.{d}.norm1.weight", .{i}, d),
            .norm2 = try l.vec("text_embedder.blocks.{d}.norm2.weight", .{i}, d),
            .q_norm = try l.vec("text_embedder.blocks.{d}.q_norm.weight", .{i}, hd),
            .k_norm = try l.vec("text_embedder.blocks.{d}.k_norm.weight", .{i}, hd),
            .qkv = try l.linear("text_embedder.blocks.{d}.qkv", .{i}, 3 * d, d, true),
            .proj = try l.linear("text_embedder.blocks.{d}.proj", .{i}, d, d, true),
            .gate = try l.linear("text_embedder.blocks.{d}.ffn_gate", .{i}, ffn, d, true),
            .up = try l.linear("text_embedder.blocks.{d}.ffn_up", .{i}, ffn, d, true),
            .down = try l.linear("text_embedder.blocks.{d}.ffn_down", .{i}, d, ffn, true),
        };
    }
};

// --- tests -----------------------------------------------------------------

const testing = std.testing;
const safetensors = tp_core.safetensors;
const test_gate = @import("../test_gate.zig");

const forward_ref = "src/models/assets/moss_image_forward.safetensors";
const moss_ckpt = build_options.model_root ++ "/diffusion_models/moss/moss_image_sft-INT8_CONVROT.safetensors";

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
    const store = pre.store();

    var cfg = try Config.detect(store);
    try testing.expectEqual(want_fused, cfg.fused_qkv);
    try testing.expectEqual(@as(usize, 128), cfg.dim);
    try testing.expectEqual(@as(usize, 2), cfg.layers);
    // ComfyUI's detection fixes heads at dim / 128; the toy has two 64-wide heads.
    cfg.heads = 2;
    var model = try DiT.loadWith(gpa, store, cfg);
    defer model.deinit();

    for ([_][]const u8{ "even", "odd" }) |case| {
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

        const shape = xv.info.shape.slice();
        const got = try gpa.alloc(f32, x.len);
        defer gpa.free(got);
        try model.forward(io, gpa, got, x, shape[1], shape[2], sig[0], ctx, ctx.len / cfg.txt_dim, null);
        const rel = relL2(want, got);
        errdefer std.debug.print("{s} {s}: rel L2 {e:.3}\n", .{ prefix, case, rel });
        try testing.expect(rel < 1e-5);
    }
}

test "the MOSS-Image DiT matches ComfyUI at a toy width, fused qkv" {
    try checkToy("fused.", true);
}

test "the MOSS-Image DiT matches ComfyUI at a toy width, split qkv" {
    try checkToy("split.", false);
}

test "situ caps both halves" {
    var out: [3]f32 = undefined;
    situ(&out, &.{ 0, 1000, 1 }, &.{ 1000, 1, -1000 });
    try testing.expectEqual(@as(f32, 0), out[0]);
    // tanh(h / 25) saturates at 1, so the hidden half caps at 25.
    try testing.expectApproxEqRel(@as(f32, 25.0 * 4.0 * std.math.tanh(@as(f32, 0.25)) / (1.0 + @exp(@as(f32, -1)))), out[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0), out[2], 1e-6);
}

test "the real MOSS-Image checkpoint's config" {
    const io = testing.io;
    try test_gate.requireModelFile(io, moss_ckpt);
    var ck = try safetensors.SafeTensors.open(testing.allocator, io, moss_ckpt);
    defer ck.deinit();
    const cfg = try Config.detect(.{ .safetensors = &ck });
    try testing.expectEqual(Config{ .dim = 3072, .ffn = 8192, .heads = 24, .layers = 42, .txt_dim = 4096, .channels = 128, .fused_qkv = true }, cfg);
}
