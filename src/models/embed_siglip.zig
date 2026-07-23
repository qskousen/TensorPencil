//! SigLIP2 ViT-B-16 (webli) encoders for DiffKeep's image / cross-modal space.
//! Two towers share one open_clip checkpoint (`open_clip_model.safetensors`):
//!   - `TextModel` — text → 768-d (cross-modal query). [this file]
//!   - `VisualModel` — image → 768-d (index-time). [TODO]
//! Both emit L2-normalized 768-d vectors comparable to each other (contrastive).
//!
//! ## Text tower (open_clip CLIP text, SigLIP2 variant)
//! `token_embedding[id] + positional_embedding` (learned, NOT RoPE) → 12
//! **pre-LayerNorm** residual blocks (`x += attn(ln_1(x)); x += mlp(ln_2(x))`),
//! non-causal attention (no mask — SigLIP pads to a fixed `context_length` and
//! attends everything), gelu-tanh plain FFN (`c_fc → gelu → c_proj`) → `ln_final`
//! → **last-token pool** (position `context_length-1`) → `text_projection`
//! (Linear +bias) → L2 normalize.
//!
//! SigLIP text input is a fixed 64-token window, right-padded with token 0 (no
//! attention mask); pooling reads the final position. Callers pass the framed
//! ids `[content… , <eos>=1]` (typically ≤60 tokens, matching DiffKeep's
//! `SIGLIP_MAX_TOKENS-4`); `embed` truncates/pads to `context_length`.

const std = @import("std");
const tp_core = @import("tp_core");
const ops = @import("tp_ops");
const loader = @import("loader.zig");
const qwen3 = @import("qwen3.zig");

const SafeTensors = tp_core.safetensors.SafeTensors;
const WeightStore = tp_core.weights.WeightStore;
const Weight = ops.matmul.Weight;

pub const embed_dim: usize = 768;

/// In-place L2 normalization (shared by the per-item and batched paths).
fn l2normalize(v: []f32) void {
    var ss: f32 = 0;
    for (v) |x| ss += x * x;
    const norm = @sqrt(ss);
    if (norm > 0) {
        const inv = 1.0 / norm;
        for (v) |*x| x.* *= inv;
    }
}

pub const TextConfig = struct {
    context_length: usize = 64,
    vocab: usize = 256000,
    width: usize = 768,
    n_heads: usize = 12,
    head_dim: usize = 64,
    n_layers: usize = 12,
    ln_eps: f32 = 1e-6,
    pad_id: u32 = 0,
};

const TextLayer = struct {
    ln1_w: []const f32,
    ln1_b: []const f32,
    in_proj: Weight, // [3*width, width]
    in_proj_bias: []const f32,
    out_proj: Weight, // [width, width]
    out_proj_bias: []const f32,
    ln2_w: []const f32,
    ln2_b: []const f32,
    c_fc: Weight, // [4*width, width] (3072)
    c_fc_bias: []const f32,
    c_proj: Weight, // [width, 4*width]
    c_proj_bias: []const f32,
};

pub const TextModel = struct {
    arena: std.heap.ArenaAllocator,
    st: SafeTensors,
    cfg: TextConfig,
    token_emb: Weight, // [vocab, width]
    pos_emb: []const f32, // [context_length * width]
    layers: []TextLayer,
    ln_final_w: []const f32,
    ln_final_b: []const f32,
    text_proj: Weight, // [width, width]
    text_proj_bias: []const f32,

    /// Load from a directory with `open_clip_model.safetensors`.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !TextModel {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();
        const cfg: TextConfig = .{};
        const w = cfg.width;

        var pbuf: [1024]u8 = undefined;
        var st = try SafeTensors.open(gpa, io, try std.fmt.bufPrint(&pbuf, "{s}/open_clip_model.safetensors", .{dir}));
        errdefer st.deinit();
        const store: WeightStore = .{ .safetensors = &st };

        const token_emb = try loader.matrix(store, "text.token_embedding.weight", cfg.vocab, w);
        const pos_emb = try loader.vector(a, store, "text.positional_embedding", cfg.context_length * w);
        const ln_final_w = try loader.vector(a, store, "text.ln_final.weight", w);
        const ln_final_b = try loader.vector(a, store, "text.ln_final.bias", w);
        const text_proj = try loader.matrix(store, "text.text_projection.weight", w, w);
        const text_proj_bias = try loader.vector(a, store, "text.text_projection.bias", w);

        const layers = try a.alloc(TextLayer, cfg.n_layers);
        for (layers, 0..) |*layer, l| {
            layer.* = .{
                .ln1_w = try loader.indexedVector(a, store, "text.transformer.resblocks.", l, "ln_1.weight", w),
                .ln1_b = try loader.indexedVector(a, store, "text.transformer.resblocks.", l, "ln_1.bias", w),
                .in_proj = try loader.indexedMatrix(store, "text.transformer.resblocks.", l, "attn.in_proj_weight", 3 * w, w),
                .in_proj_bias = try loader.indexedVector(a, store, "text.transformer.resblocks.", l, "attn.in_proj_bias", 3 * w),
                .out_proj = try loader.indexedMatrix(store, "text.transformer.resblocks.", l, "attn.out_proj.weight", w, w),
                .out_proj_bias = try loader.indexedVector(a, store, "text.transformer.resblocks.", l, "attn.out_proj.bias", w),
                .ln2_w = try loader.indexedVector(a, store, "text.transformer.resblocks.", l, "ln_2.weight", w),
                .ln2_b = try loader.indexedVector(a, store, "text.transformer.resblocks.", l, "ln_2.bias", w),
                .c_fc = try loader.indexedMatrix(store, "text.transformer.resblocks.", l, "mlp.c_fc.weight", 4 * w, w),
                .c_fc_bias = try loader.indexedVector(a, store, "text.transformer.resblocks.", l, "mlp.c_fc.bias", 4 * w),
                .c_proj = try loader.indexedMatrix(store, "text.transformer.resblocks.", l, "mlp.c_proj.weight", w, 4 * w),
                .c_proj_bias = try loader.indexedVector(a, store, "text.transformer.resblocks.", l, "mlp.c_proj.bias", w),
            };
        }

        return .{
            .arena = arena,
            .st = st,
            .cfg = cfg,
            .token_emb = token_emb,
            .pos_emb = pos_emb,
            .layers = layers,
            .ln_final_w = ln_final_w,
            .ln_final_b = ln_final_b,
            .text_proj = text_proj,
            .text_proj_bias = text_proj_bias,
        };
    }

    pub fn deinit(self: *TextModel) void {
        self.st.deinit();
        self.arena.deinit();
        self.* = undefined;
    }

    /// Encode framed ids (`content…, <eos>`) into `out` [embed_dim], L2-normalized.
    /// The sequence is truncated/right-padded (token 0) to `context_length`.
    pub fn embed(self: *const TextModel, io: std.Io, gpa: std.mem.Allocator, ids: []const u32, out: []f32) !void {
        const cfg = self.cfg;
        const n = cfg.context_length;
        const w = cfg.width;
        const inter = 4 * w;
        std.debug.assert(out.len == embed_dim);

        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const a = arena.allocator();

        // Fixed-length padded token window.
        const padded = try a.alloc(u32, n);
        for (padded, 0..) |*p, t| p.* = if (t < ids.len) ids[t] else cfg.pad_id;

        // token embedding + learned positional.
        const x = try a.alloc(f32, n * w);
        try qwen3.embedTokens(self.token_emb, padded, x);
        for (x, self.pos_emb) |*xi, pe| xi.* += pe;

        const normed = try a.alloc(f32, n * w);
        const qkv = try a.alloc(f32, n * 3 * w);
        const q = try a.alloc(f32, n * w);
        const k = try a.alloc(f32, n * w);
        const v = try a.alloc(f32, n * w);
        const attn = try a.alloc(f32, n * w);
        const proj = try a.alloc(f32, n * w);
        const fc = try a.alloc(f32, n * inter);

        for (self.layers) |*layer| {
            // Attention (pre-LN): x += out_proj(attn(ln_1(x))).
            ops.norm.layerNorm(normed, x, layer.ln1_w, layer.ln1_b, cfg.ln_eps);
            try ops.matmul.matmul(io, gpa, qkv, normed, n, layer.in_proj, layer.in_proj_bias);
            for (0..n) |t| {
                const src = qkv[t * 3 * w ..];
                @memcpy(q[t * w ..][0..w], src[0..w]);
                @memcpy(k[t * w ..][0..w], src[w .. 2 * w]);
                @memcpy(v[t * w ..][0..w], src[2 * w .. 3 * w]);
            }
            try ops.attention.attention(io, gpa, attn, q, k, v, .{
                .seq_q = n,
                .seq_kv = n,
                .n_heads = cfg.n_heads,
                .n_kv_heads = cfg.n_heads,
                .head_dim = cfg.head_dim,
                .causal = false,
            });
            try ops.matmul.matmul(io, gpa, proj, attn, n, layer.out_proj, layer.out_proj_bias);
            for (x, proj) |*xi, pi| xi.* += pi;

            // MLP (pre-LN): x += c_proj(gelu(c_fc(ln_2(x)))).
            ops.norm.layerNorm(normed, x, layer.ln2_w, layer.ln2_b, cfg.ln_eps);
            try ops.matmul.matmul(io, gpa, fc, normed, n, layer.c_fc, layer.c_fc_bias);
            ops.act.geluTanh(fc);
            try ops.matmul.matmul(io, gpa, proj, fc, n, layer.c_proj, layer.c_proj_bias);
            for (x, proj) |*xi, pi| xi.* += pi;
        }

        // Final norm, last-token pool (position n-1), projection, normalize.
        ops.norm.layerNorm(x, x, self.ln_final_w, self.ln_final_b, cfg.ln_eps);
        const pooled = x[(n - 1) * w ..][0..w];
        try ops.matmul.matmul(io, gpa, out, pooled, 1, self.text_proj, self.text_proj_bias);
        l2normalize(out);
    }

    /// Batched text encode: `ids_list[i]` → `outs[i]` [embed_dim]. Every item is
    /// truncated/padded to the fixed 64-token window, so the batch is a uniform
    /// [B*context_length, width] activation — all GEMMs / LayerNorms / GeGLU run
    /// once over `B*64` rows; only attention (per item) and the final pool loop
    /// over the batch. Bit-identical to per-item `embed`.
    pub fn embedBatch(self: *const TextModel, io: std.Io, gpa: std.mem.Allocator, ids_list: []const []const u32, outs: [][]f32) !void {
        const cfg = self.cfg;
        const n = cfg.context_length;
        const w = cfg.width;
        const inter = 4 * w;
        const b = ids_list.len;
        std.debug.assert(outs.len == b and b > 0);
        const total = b * n;

        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const a = arena.allocator();

        // Padded token windows (one per item), token + positional embedding.
        const padded = try a.alloc(u32, total);
        for (ids_list, 0..) |ids, i| {
            for (padded[i * n ..][0..n], 0..) |*p, t| p.* = if (t < ids.len) ids[t] else cfg.pad_id;
        }
        const x = try a.alloc(f32, total * w);
        try qwen3.embedTokens(self.token_emb, padded, x);
        for (0..b) |i| {
            for (x[i * n * w ..][0 .. n * w], self.pos_emb) |*xi, pe| xi.* += pe;
        }

        const normed = try a.alloc(f32, total * w);
        const qkv = try a.alloc(f32, total * 3 * w);
        const q = try a.alloc(f32, total * w);
        const k = try a.alloc(f32, total * w);
        const v = try a.alloc(f32, total * w);
        const attn = try a.alloc(f32, total * w);
        const proj = try a.alloc(f32, total * w);
        const fc = try a.alloc(f32, total * inter);

        for (self.layers) |*layer| {
            ops.norm.layerNorm(normed, x, layer.ln1_w, layer.ln1_b, cfg.ln_eps);
            try ops.matmul.matmul(io, gpa, qkv, normed, total, layer.in_proj, layer.in_proj_bias);
            for (0..total) |t| {
                const src = qkv[t * 3 * w ..];
                @memcpy(q[t * w ..][0..w], src[0..w]);
                @memcpy(k[t * w ..][0..w], src[w .. 2 * w]);
                @memcpy(v[t * w ..][0..w], src[2 * w .. 3 * w]);
            }
            for (0..b) |i| {
                const s0 = i * n * w;
                try ops.attention.attention(io, gpa, attn[s0..][0 .. n * w], q[s0..][0 .. n * w], k[s0..][0 .. n * w], v[s0..][0 .. n * w], .{
                    .seq_q = n,
                    .seq_kv = n,
                    .n_heads = cfg.n_heads,
                    .n_kv_heads = cfg.n_heads,
                    .head_dim = cfg.head_dim,
                    .causal = false,
                });
            }
            try ops.matmul.matmul(io, gpa, proj, attn, total, layer.out_proj, layer.out_proj_bias);
            for (x, proj) |*xi, pi| xi.* += pi;

            ops.norm.layerNorm(normed, x, layer.ln2_w, layer.ln2_b, cfg.ln_eps);
            try ops.matmul.matmul(io, gpa, fc, normed, total, layer.c_fc, layer.c_fc_bias);
            ops.act.geluTanh(fc);
            try ops.matmul.matmul(io, gpa, proj, fc, total, layer.c_proj, layer.c_proj_bias);
            for (x, proj) |*xi, pi| xi.* += pi;
        }
        ops.norm.layerNorm(x, x, self.ln_final_w, self.ln_final_b, cfg.ln_eps);

        // Gather each item's last-token row, batch the projection, normalize.
        const pooled = try a.alloc(f32, b * w);
        for (0..b) |i| @memcpy(pooled[i * w ..][0..w], x[(i * n + n - 1) * w ..][0..w]);
        const projd = try a.alloc(f32, b * w);
        try ops.matmul.matmul(io, gpa, projd, pooled, b, self.text_proj, self.text_proj_bias);
        for (0..b) |i| {
            std.debug.assert(outs[i].len == embed_dim);
            @memcpy(outs[i], projd[i * w ..][0..w]);
            l2normalize(outs[i]);
        }
    }
};

// ── Visual tower (timm ViT-B/16 + MAP attention-pool head) ──────────────────

pub const VisualConfig = struct {
    image_size: usize = 224,
    patch: usize = 16,
    width: usize = 768,
    n_heads: usize = 12,
    head_dim: usize = 64,
    n_layers: usize = 12,
    mlp_dim: usize = 3072,
    ln_eps: f32 = 1e-6,

    pub fn gridSide(self: VisualConfig) usize {
        return self.image_size / self.patch;
    }
    pub fn nPatches(self: VisualConfig) usize {
        return self.gridSide() * self.gridSide();
    }
    /// Per-patch input length (C*patch*patch), = the conv weight's input dim.
    pub fn patchIn(self: VisualConfig) usize {
        return 3 * self.patch * self.patch;
    }
};

const VisualLayer = struct {
    norm1_w: []const f32,
    norm1_b: []const f32,
    qkv: Weight, // [3*width, width]
    qkv_b: []const f32,
    attn_proj: Weight, // [width, width]
    attn_proj_b: []const f32,
    norm2_w: []const f32,
    norm2_b: []const f32,
    fc1: Weight, // [mlp_dim, width]
    fc1_b: []const f32,
    fc2: Weight, // [width, mlp_dim]
    fc2_b: []const f32,
};

/// A matrix weight whose tensor may be >2-D (e.g. the conv patch-embed weight
/// [out,3,16,16], which is contiguous == [out, 3*16*16]); validated by element
/// count, not shape rank.
fn flatMatrix(store: WeightStore, name: []const u8, rows: usize, cols: usize) !Weight {
    const view = store.get(name) orelse return error.MissingTensor;
    if (view.info.elemCount() != rows * cols) return error.ShapeMismatch;
    return Weight.init(view.bytes, view.info.dtype, rows, cols);
}

pub const VisualModel = struct {
    arena: std.heap.ArenaAllocator,
    st: SafeTensors,
    cfg: VisualConfig,
    patch_w: Weight, // [width, 3*patch*patch]
    patch_b: []const f32,
    pos_emb: []const f32, // [n_patches * width]
    blocks: []VisualLayer,
    norm_w: []const f32, // trunk.norm
    norm_b: []const f32,
    // MAP attention-pool head (timm AttentionPoolLatent).
    latent: []const f32, // [width]
    q_w: Weight,
    q_b: []const f32,
    kv_w: Weight, // [2*width, width]
    kv_b: []const f32,
    proj_w: Weight,
    proj_b: []const f32,
    head_norm_w: []const f32,
    head_norm_b: []const f32,
    head_fc1: Weight, // [mlp_dim, width]
    head_fc1_b: []const f32,
    head_fc2: Weight, // [width, mlp_dim]
    head_fc2_b: []const f32,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !VisualModel {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();
        const cfg: VisualConfig = .{};
        const w = cfg.width;

        var pbuf: [1024]u8 = undefined;
        var st = try SafeTensors.open(gpa, io, try std.fmt.bufPrint(&pbuf, "{s}/open_clip_model.safetensors", .{dir}));
        errdefer st.deinit();
        const store: WeightStore = .{ .safetensors = &st };

        const patch_w = try flatMatrix(store, "visual.trunk.patch_embed.proj.weight", w, cfg.patchIn());
        const patch_b = try loader.vector(a, store, "visual.trunk.patch_embed.proj.bias", w);
        const pos_emb = try loader.vector(a, store, "visual.trunk.pos_embed", cfg.nPatches() * w);

        const blocks = try a.alloc(VisualLayer, cfg.n_layers);
        for (blocks, 0..) |*b, l| {
            b.* = .{
                .norm1_w = try loader.indexedVector(a, store, "visual.trunk.blocks.", l, "norm1.weight", w),
                .norm1_b = try loader.indexedVector(a, store, "visual.trunk.blocks.", l, "norm1.bias", w),
                .qkv = try loader.indexedMatrix(store, "visual.trunk.blocks.", l, "attn.qkv.weight", 3 * w, w),
                .qkv_b = try loader.indexedVector(a, store, "visual.trunk.blocks.", l, "attn.qkv.bias", 3 * w),
                .attn_proj = try loader.indexedMatrix(store, "visual.trunk.blocks.", l, "attn.proj.weight", w, w),
                .attn_proj_b = try loader.indexedVector(a, store, "visual.trunk.blocks.", l, "attn.proj.bias", w),
                .norm2_w = try loader.indexedVector(a, store, "visual.trunk.blocks.", l, "norm2.weight", w),
                .norm2_b = try loader.indexedVector(a, store, "visual.trunk.blocks.", l, "norm2.bias", w),
                .fc1 = try loader.indexedMatrix(store, "visual.trunk.blocks.", l, "mlp.fc1.weight", cfg.mlp_dim, w),
                .fc1_b = try loader.indexedVector(a, store, "visual.trunk.blocks.", l, "mlp.fc1.bias", cfg.mlp_dim),
                .fc2 = try loader.indexedMatrix(store, "visual.trunk.blocks.", l, "mlp.fc2.weight", w, cfg.mlp_dim),
                .fc2_b = try loader.indexedVector(a, store, "visual.trunk.blocks.", l, "mlp.fc2.bias", w),
            };
        }

        return .{
            .arena = arena,
            .st = st,
            .cfg = cfg,
            .patch_w = patch_w,
            .patch_b = patch_b,
            .pos_emb = pos_emb,
            .blocks = blocks,
            .norm_w = try loader.vector(a, store, "visual.trunk.norm.weight", w),
            .norm_b = try loader.vector(a, store, "visual.trunk.norm.bias", w),
            .latent = try loader.vector(a, store, "visual.trunk.attn_pool.latent", w),
            .q_w = try loader.matrix(store, "visual.trunk.attn_pool.q.weight", w, w),
            .q_b = try loader.vector(a, store, "visual.trunk.attn_pool.q.bias", w),
            .kv_w = try loader.matrix(store, "visual.trunk.attn_pool.kv.weight", 2 * w, w),
            .kv_b = try loader.vector(a, store, "visual.trunk.attn_pool.kv.bias", 2 * w),
            .proj_w = try loader.matrix(store, "visual.trunk.attn_pool.proj.weight", w, w),
            .proj_b = try loader.vector(a, store, "visual.trunk.attn_pool.proj.bias", w),
            .head_norm_w = try loader.vector(a, store, "visual.trunk.attn_pool.norm.weight", w),
            .head_norm_b = try loader.vector(a, store, "visual.trunk.attn_pool.norm.bias", w),
            .head_fc1 = try loader.matrix(store, "visual.trunk.attn_pool.mlp.fc1.weight", cfg.mlp_dim, w),
            .head_fc1_b = try loader.vector(a, store, "visual.trunk.attn_pool.mlp.fc1.bias", cfg.mlp_dim),
            .head_fc2 = try loader.matrix(store, "visual.trunk.attn_pool.mlp.fc2.weight", w, cfg.mlp_dim),
            .head_fc2_b = try loader.vector(a, store, "visual.trunk.attn_pool.mlp.fc2.bias", w),
        };
    }

    pub fn deinit(self: *VisualModel) void {
        self.st.deinit();
        self.arena.deinit();
        self.* = undefined;
    }

    /// Encode a decoded, preprocessed RGB image (`img` = CHW [3*224*224], `/255`
    /// then mean/std 0.5) into `out` [embed_dim], L2-normalized.
    pub fn embed(self: *const VisualModel, io: std.Io, gpa: std.mem.Allocator, img: []const f32, out: []f32) !void {
        const cfg = self.cfg;
        const w = cfg.width;
        const np = cfg.nPatches();
        std.debug.assert(out.len == embed_dim and img.len == 3 * cfg.image_size * cfg.image_size);

        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const a = arena.allocator();

        // Patchify: gather each 16×16×3 patch into (c,ky,kx) order, then the conv
        // reduces to a matmul [np, patchIn] @ patch_w^T.
        const patch_in = try self.patchify(a, img);
        const x = try a.alloc(f32, np * w);
        try ops.matmul.matmul(io, gpa, x, patch_in, np, self.patch_w, self.patch_b);
        for (x, self.pos_emb) |*xi, pe| xi.* += pe;

        // ViT blocks (pre-LN).
        const normed = try a.alloc(f32, np * w);
        const qkv = try a.alloc(f32, np * 3 * w);
        const q = try a.alloc(f32, np * w);
        const k = try a.alloc(f32, np * w);
        const v = try a.alloc(f32, np * w);
        const attn = try a.alloc(f32, np * w);
        const proj = try a.alloc(f32, np * w);
        const fc = try a.alloc(f32, np * cfg.mlp_dim);
        for (self.blocks) |*b| {
            ops.norm.layerNorm(normed, x, b.norm1_w, b.norm1_b, cfg.ln_eps);
            try ops.matmul.matmul(io, gpa, qkv, normed, np, b.qkv, b.qkv_b);
            for (0..np) |t| {
                const src = qkv[t * 3 * w ..];
                @memcpy(q[t * w ..][0..w], src[0..w]);
                @memcpy(k[t * w ..][0..w], src[w .. 2 * w]);
                @memcpy(v[t * w ..][0..w], src[2 * w .. 3 * w]);
            }
            try ops.attention.attention(io, gpa, attn, q, k, v, .{
                .seq_q = np,
                .seq_kv = np,
                .n_heads = cfg.n_heads,
                .n_kv_heads = cfg.n_heads,
                .head_dim = cfg.head_dim,
                .causal = false,
            });
            try ops.matmul.matmul(io, gpa, proj, attn, np, b.attn_proj, b.attn_proj_b);
            for (x, proj) |*xi, pi| xi.* += pi;

            ops.norm.layerNorm(normed, x, b.norm2_w, b.norm2_b, cfg.ln_eps);
            try ops.matmul.matmul(io, gpa, fc, normed, np, b.fc1, b.fc1_b);
            ops.act.geluTanh(fc);
            try ops.matmul.matmul(io, gpa, proj, fc, np, b.fc2, b.fc2_b);
            for (x, proj) |*xi, pi| xi.* += pi;
        }
        ops.norm.layerNorm(x, x, self.norm_w, self.norm_b, cfg.ln_eps);

        // MAP attention-pool head + L2 normalize (shared with the GPU path,
        // which runs the ViT body on-device then calls this on the host).
        try self.mapHead(io, gpa, x, out);
    }

    /// Gather the input image into per-patch (c,ky,kx) rows [nPatches, patchIn]
    /// — the conv patch-embed reduces to a matmul over these. Caller owns.
    pub fn patchify(self: *const VisualModel, a: std.mem.Allocator, img: []const f32) ![]f32 {
        const cfg = self.cfg;
        const ps = cfg.patch;
        const isz = cfg.image_size;
        const side = cfg.gridSide();
        const patch_in = try a.alloc(f32, cfg.nPatches() * cfg.patchIn());
        for (0..side) |py| {
            for (0..side) |px| {
                const t = py * side + px;
                var dst = patch_in[t * cfg.patchIn() ..];
                for (0..3) |c| {
                    for (0..ps) |ky| {
                        const row = py * ps + ky;
                        const base = c * isz * isz + row * isz + px * ps;
                        for (0..ps) |kx| dst[c * ps * ps + ky * ps + kx] = img[base + kx];
                    }
                }
            }
        }
        return patch_in;
    }

    /// MAP attention-pool head over post-`trunk.norm` patch tokens `x_normed`
    /// [nPatches * width]: one latent query cross-attends the tokens → proj →
    /// residual pre-norm MLP → L2-normalized 768-d `out`. Cheap; always host.
    pub fn mapHead(self: *const VisualModel, io: std.Io, gpa: std.mem.Allocator, x_normed: []const f32, out: []f32) !void {
        const cfg = self.cfg;
        const w = cfg.width;
        const np = x_normed.len / w;
        std.debug.assert(out.len == embed_dim);

        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const a = arena.allocator();

        const q1 = try a.alloc(f32, w);
        try ops.matmul.matmul(io, gpa, q1, self.latent, 1, self.q_w, self.q_b);
        const kv = try a.alloc(f32, np * 2 * w);
        try ops.matmul.matmul(io, gpa, kv, x_normed, np, self.kv_w, self.kv_b);
        const kk = try a.alloc(f32, np * w);
        const vv = try a.alloc(f32, np * w);
        for (0..np) |t| {
            @memcpy(kk[t * w ..][0..w], kv[t * 2 * w ..][0..w]);
            @memcpy(vv[t * w ..][0..w], kv[t * 2 * w + w ..][0..w]);
        }
        const pooled = try a.alloc(f32, w);
        try ops.attention.attention(io, gpa, pooled, q1, kk, vv, .{
            .seq_q = 1,
            .seq_kv = np,
            .n_heads = cfg.n_heads,
            .n_kv_heads = cfg.n_heads,
            .head_dim = cfg.head_dim,
            .causal = false,
        });
        const p = try a.alloc(f32, w);
        try ops.matmul.matmul(io, gpa, p, pooled, 1, self.proj_w, self.proj_b);
        const hn = try a.alloc(f32, w);
        ops.norm.layerNorm(hn, p, self.head_norm_w, self.head_norm_b, cfg.ln_eps);
        const hf = try a.alloc(f32, cfg.mlp_dim);
        try ops.matmul.matmul(io, gpa, hf, hn, 1, self.head_fc1, self.head_fc1_b);
        ops.act.geluTanh(hf);
        const hm = try a.alloc(f32, w);
        try ops.matmul.matmul(io, gpa, hm, hf, 1, self.head_fc2, self.head_fc2_b);
        for (p, hm) |*pi, mi| pi.* += mi;

        @memcpy(out, p);
        l2normalize(out);
    }

    /// Batched image encode: `imgs[i]` (CHW [3*224*224]) → `outs[i]` [embed_dim].
    /// The per-image patch count is fixed (196), so the batch is a uniform
    /// [B*nPatches, width] activation — every ViT GEMM / LayerNorm / GeGLU runs
    /// once over `B*196` rows; only the self-attention (per image) and the MAP
    /// pool head loop over the batch. Bit-identical to per-item `embed`.
    pub fn embedBatch(self: *const VisualModel, io: std.Io, gpa: std.mem.Allocator, imgs: []const []const f32, outs: [][]f32) !void {
        const cfg = self.cfg;
        const w = cfg.width;
        const np = cfg.nPatches();
        const b = imgs.len;
        std.debug.assert(outs.len == b and b > 0);
        const total = b * np;

        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const a = arena.allocator();

        // Patchify + patch-embed (matmul) + positional, per image.
        const x = try a.alloc(f32, total * w);
        for (imgs, 0..) |img, i| {
            std.debug.assert(img.len == 3 * cfg.image_size * cfg.image_size);
            const patch_in = try self.patchify(a, img);
            try ops.matmul.matmul(io, gpa, x[i * np * w ..][0 .. np * w], patch_in, np, self.patch_w, self.patch_b);
            for (x[i * np * w ..][0 .. np * w], self.pos_emb) |*xi, pe| xi.* += pe;
        }

        const normed = try a.alloc(f32, total * w);
        const qkv = try a.alloc(f32, total * 3 * w);
        const q = try a.alloc(f32, total * w);
        const k = try a.alloc(f32, total * w);
        const v = try a.alloc(f32, total * w);
        const attn = try a.alloc(f32, total * w);
        const proj = try a.alloc(f32, total * w);
        const fc = try a.alloc(f32, total * cfg.mlp_dim);
        for (self.blocks) |*bl| {
            ops.norm.layerNorm(normed, x, bl.norm1_w, bl.norm1_b, cfg.ln_eps);
            try ops.matmul.matmul(io, gpa, qkv, normed, total, bl.qkv, bl.qkv_b);
            for (0..total) |t| {
                const src = qkv[t * 3 * w ..];
                @memcpy(q[t * w ..][0..w], src[0..w]);
                @memcpy(k[t * w ..][0..w], src[w .. 2 * w]);
                @memcpy(v[t * w ..][0..w], src[2 * w .. 3 * w]);
            }
            for (0..b) |i| {
                const s0 = i * np * w;
                try ops.attention.attention(io, gpa, attn[s0..][0 .. np * w], q[s0..][0 .. np * w], k[s0..][0 .. np * w], v[s0..][0 .. np * w], .{
                    .seq_q = np,
                    .seq_kv = np,
                    .n_heads = cfg.n_heads,
                    .n_kv_heads = cfg.n_heads,
                    .head_dim = cfg.head_dim,
                    .causal = false,
                });
            }
            try ops.matmul.matmul(io, gpa, proj, attn, total, bl.attn_proj, bl.attn_proj_b);
            for (x, proj) |*xi, pi| xi.* += pi;

            ops.norm.layerNorm(normed, x, bl.norm2_w, bl.norm2_b, cfg.ln_eps);
            try ops.matmul.matmul(io, gpa, fc, normed, total, bl.fc1, bl.fc1_b);
            ops.act.geluTanh(fc);
            try ops.matmul.matmul(io, gpa, proj, fc, total, bl.fc2, bl.fc2_b);
            for (x, proj) |*xi, pi| xi.* += pi;
        }
        ops.norm.layerNorm(x, x, self.norm_w, self.norm_b, cfg.ln_eps);

        try self.mapHeadBatch(io, gpa, x, outs);
    }

    /// Batched MAP attention-pool head over `x_normed` [B*nPatches*width]. The
    /// latent query is image-independent (computed once); the KV projection and
    /// the residual MLP batch over B rows, and only the cross-attention loops per
    /// image. Bit-identical to `mapHead` per item.
    pub fn mapHeadBatch(self: *const VisualModel, io: std.Io, gpa: std.mem.Allocator, x_normed: []const f32, outs: [][]f32) !void {
        const cfg = self.cfg;
        const w = cfg.width;
        const b = outs.len;
        const np = x_normed.len / w / b;

        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const a = arena.allocator();

        // Latent query (same for every image).
        const q1 = try a.alloc(f32, w);
        try ops.matmul.matmul(io, gpa, q1, self.latent, 1, self.q_w, self.q_b);

        // KV projection over the whole batch, then split per row.
        const kv = try a.alloc(f32, b * np * 2 * w);
        try ops.matmul.matmul(io, gpa, kv, x_normed, b * np, self.kv_w, self.kv_b);

        const pooled = try a.alloc(f32, b * w);
        const kk = try a.alloc(f32, np * w);
        const vv = try a.alloc(f32, np * w);
        for (0..b) |i| {
            for (0..np) |t| {
                const s = (i * np + t) * 2 * w;
                @memcpy(kk[t * w ..][0..w], kv[s..][0..w]);
                @memcpy(vv[t * w ..][0..w], kv[s + w ..][0..w]);
            }
            try ops.attention.attention(io, gpa, pooled[i * w ..][0..w], q1, kk, vv, .{
                .seq_q = 1,
                .seq_kv = np,
                .n_heads = cfg.n_heads,
                .n_kv_heads = cfg.n_heads,
                .head_dim = cfg.head_dim,
                .causal = false,
            });
        }

        // proj → residual pre-norm MLP, batched over B rows.
        const p = try a.alloc(f32, b * w);
        try ops.matmul.matmul(io, gpa, p, pooled, b, self.proj_w, self.proj_b);
        const hn = try a.alloc(f32, b * w);
        ops.norm.layerNorm(hn, p, self.head_norm_w, self.head_norm_b, cfg.ln_eps);
        const hf = try a.alloc(f32, b * cfg.mlp_dim);
        try ops.matmul.matmul(io, gpa, hf, hn, b, self.head_fc1, self.head_fc1_b);
        ops.act.geluTanh(hf);
        const hm = try a.alloc(f32, b * w);
        try ops.matmul.matmul(io, gpa, hm, hf, b, self.head_fc2, self.head_fc2_b);
        for (p, hm) |*pi, mi| pi.* += mi;

        for (0..b) |i| {
            std.debug.assert(outs[i].len == embed_dim);
            @memcpy(outs[i], p[i * w ..][0..w]);
            l2normalize(outs[i]);
        }
    }
};

// --- tests -----------------------------------------------------------------

fn cosine(x: []const f32, y: []const f32) f32 {
    var dot: f32 = 0;
    var nx: f32 = 0;
    var ny: f32 = 0;
    for (x, y) |xi, yi| {
        dot += xi * yi;
        nx += xi * xi;
        ny += yi * yi;
    }
    return dot / (@sqrt(nx) * @sqrt(ny));
}

// Numeric parity vs the immich SigLIP2 textual ONNX (reference vectors keyed to
// the golden BPE ids, padded to 64). Requires ../DiffKeep; skipped when absent.
test "siglip2 text tower matches ONNX reference" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dir = "../DiffKeep/Models/ViT-B-16-SigLIP2-timm";
    std.Io.Dir.cwd().access(io, dir, .{}) catch return error.SkipZigTest;

    const ref_bytes = std.Io.Dir.cwd().readFileAlloc(io, "testdata/siglip2_text_ref_vectors.json", gpa, .limited(4 * 1024 * 1024)) catch return error.SkipZigTest;
    defer gpa.free(ref_bytes);
    var ref = try std.json.parseFromSlice(std.json.Value, gpa, ref_bytes, .{});
    defer ref.deinit();

    var model = try TextModel.open(gpa, io, dir);
    defer model.deinit();

    const out = try gpa.alloc(f32, embed_dim);
    defer gpa.free(out);
    const ids_buf = try gpa.alloc(u32, 64);
    defer gpa.free(ids_buf);

    var it = ref.value.object.iterator();
    var worst: f32 = 1.0;
    while (it.next()) |entry| {
        const case = entry.value_ptr.*.object;
        const ids_json = case.get("ids").?.array.items;
        const want = case.get("vec").?.array.items;
        const ncopy = @min(ids_json.len, ids_buf.len);
        const ids = ids_buf[0..ncopy];
        for (ids_json[0..ncopy], ids) |val, *d| d.* = @intCast(val.integer);

        try model.embed(io, gpa, ids, out);

        var wbuf: [embed_dim]f32 = undefined;
        for (want, 0..) |val, i| wbuf[i] = @floatCast(val.float);
        const cos = cosine(out, &wbuf);
        worst = @min(worst, cos);
        errdefer std.debug.print("case {s}: cosine {d}\n", .{ entry.key_ptr.*, cos });
        try std.testing.expect(cos >= 0.999);
    }
    errdefer std.debug.print("worst cosine {d}\n", .{worst});
    try std.testing.expect(worst >= 0.999);
}

// Numeric parity vs the immich SigLIP2 visual ONNX on a fixed input tensor
// (testdata/siglip2_visual_input.f32, raw f32 CHW [3,224,224]). Requires
// ../DiffKeep; skipped when absent.
test "siglip2 visual tower matches ONNX reference" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dir = "../DiffKeep/Models/ViT-B-16-SigLIP2-timm";
    std.Io.Dir.cwd().access(io, dir, .{}) catch return error.SkipZigTest;

    const raw = std.Io.Dir.cwd().readFileAlloc(io, "testdata/siglip2_visual_input.f32", gpa, .limited(4 * 1024 * 1024)) catch return error.SkipZigTest;
    defer gpa.free(raw);
    const img = try gpa.alloc(f32, raw.len / @sizeOf(f32));
    defer gpa.free(img);
    @memcpy(std.mem.sliceAsBytes(img), raw);
    const ref_bytes = std.Io.Dir.cwd().readFileAlloc(io, "testdata/siglip2_visual_ref.json", gpa, .limited(1024 * 1024)) catch return error.SkipZigTest;
    defer gpa.free(ref_bytes);
    var ref = try std.json.parseFromSlice(std.json.Value, gpa, ref_bytes, .{});
    defer ref.deinit();

    var model = try VisualModel.open(gpa, io, dir);
    defer model.deinit();

    const out = try gpa.alloc(f32, embed_dim);
    defer gpa.free(out);
    try model.embed(io, gpa, img, out);

    const want = ref.value.object.get("vec").?.array.items;
    var wbuf: [embed_dim]f32 = undefined;
    for (want, 0..) |val, i| wbuf[i] = @floatCast(val.float);
    const cos = cosine(out, &wbuf);
    errdefer std.debug.print("visual cosine {d}\n", .{cos});
    try std.testing.expect(cos >= 0.999);
}

test "siglip2 text embedBatch matches per-item" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dir = "../DiffKeep/Models/ViT-B-16-SigLIP2-timm";
    std.Io.Dir.cwd().access(io, dir, .{}) catch return error.SkipZigTest;

    var model = try TextModel.open(gpa, io, dir);
    defer model.deinit();

    const item0 = [_]u32{ 100, 200, 300, 1 };
    const item1 = [_]u32{ 55, 66, 77, 88, 99, 1 };
    const item2 = [_]u32{ 42, 1 };
    const ids_list = [_][]const u32{ &item0, &item1, &item2 };

    var single: [3][embed_dim]f32 = undefined;
    for (ids_list, 0..) |ids, i| try model.embed(io, gpa, ids, &single[i]);
    var batched: [3][embed_dim]f32 = undefined;
    var outs: [3][]f32 = .{ &batched[0], &batched[1], &batched[2] };
    try model.embedBatch(io, gpa, &ids_list, &outs);

    for (0..3) |i| {
        var maxad: f32 = 0;
        for (single[i], batched[i]) |sv, bv| maxad = @max(maxad, @abs(sv - bv));
        errdefer std.debug.print("text item {d}: max abs diff {d}\n", .{ i, maxad });
        try std.testing.expect(maxad < 1e-5);
    }
}

test "siglip2 visual embedBatch matches per-item" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dir = "../DiffKeep/Models/ViT-B-16-SigLIP2-timm";
    std.Io.Dir.cwd().access(io, dir, .{}) catch return error.SkipZigTest;

    var model = try VisualModel.open(gpa, io, dir);
    defer model.deinit();

    const n = 3 * 224 * 224;
    var prng = std.Random.DefaultPrng.init(0x51A1);
    const r = prng.random();
    const imgs_data = try gpa.alloc(f32, 2 * n);
    defer gpa.free(imgs_data);
    for (imgs_data) |*e| e.* = (r.float(f32) - 0.5) * 2.0;
    const imgs = [_][]const f32{ imgs_data[0..n], imgs_data[n..] };

    var single: [2][embed_dim]f32 = undefined;
    for (imgs, 0..) |img, i| try model.embed(io, gpa, img, &single[i]);
    var batched: [2][embed_dim]f32 = undefined;
    var outs: [2][]f32 = .{ &batched[0], &batched[1] };
    try model.embedBatch(io, gpa, &imgs, &outs);

    for (0..2) |i| {
        var maxad: f32 = 0;
        for (single[i], batched[i]) |sv, bv| maxad = @max(maxad, @abs(sv - bv));
        errdefer std.debug.print("visual item {d}: max abs diff {d}\n", .{ i, maxad });
        try std.testing.expect(maxad < 1e-4);
    }
}
