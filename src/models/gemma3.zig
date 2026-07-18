//! Gemma 3 language model (GGUF arch "gemma3"), text-only path. Ported from
//! llama.cpp (src/models/gemma3.cpp + the GEMMA3 hparams in llama-model.cpp).
//!
//! Differences from the Qwen stack this engine already runs:
//!   - "Sandwich" norms: each layer has FOUR RMSNorms — input (pre-attn),
//!     post-attention (on the attn output, BEFORE the residual add),
//!     pre-feedforward, and post-feedforward (on the MLP output, before its
//!     residual add). RMSNorm weights ship with +1 already folded in by the
//!     GGUF converter, so plain rmsNorm is correct.
//!   - Input embeddings are scaled by sqrt(hidden).
//!   - Per-head QK RMS-norm over head_dim (256), then full rotate-half
//!     (NEOX) RoPE. RoPE base/scale alternate by layer: every 6th layer
//!     (`sliding_window_pattern`) is GLOBAL (theta 1e6, linear scale 1/8 —
//!     position interpolation) with full causal attention; the rest are
//!     LOCAL, using a sliding-window causal mask (window 1024, theta 1e4,
//!     no scaling).
//!   - GeGLU (gelu-tanh) FFN gate; tied LM head (no output.weight);
//!     no attention/final-logit softcapping (Gemma 3 dropped both).
//!
//! head_dim 256 with the 12B's 1/sqrt(256) attention scale is exactly the
//! engine's default 1/sqrt(head_dim), so ops.attention needs no change.
//!
//! Weights stay in checkpoint dtype (GGUF block quants) and dequantize
//! inside the GEMM; the Gguf mapping must outlive the model.

const std = @import("std");
const gguf_mod = @import("../gguf.zig");
const weights_mod = @import("../weights.zig");
const qwen3 = @import("qwen3.zig");
const ops = @import("../ops.zig");
const loader = @import("loader.zig");
const transformer = @import("transformer.zig");
const kv_cache_mod = @import("../llm/kv_cache.zig");

const Gguf = gguf_mod.Gguf;
const WeightStore = weights_mod.WeightStore;
const Weight = ops.matmul.Weight;
const KvCache = kv_cache_mod.KvCache;
const PerLayerKvCache = kv_cache_mod.PerLayerKvCache;

/// CPU prefill batch size — bounds per-forward `seq` so a LOCAL layer's KV ring
/// (`sliding_window + prefill_chunk` rows) can't alias a still-needed key in one
/// batch (TODO lever 1). Also caps the activation scratch height.
const prefill_chunk = 128;

/// Gemma 3 vision is always a 16x16 = 256-token soft-image block. A
/// bidirectional image block is prefilled in ONE pass, so it (not
/// prefill_chunk) sets the largest single batch the LOCAL ring must hold.
const max_image_tokens = 256;

/// Largest single prefill batch: a text chunk (prefill_chunk) or a whole
/// bidirectional image block (max_image_tokens), whichever is larger.
const max_batch = @max(prefill_chunk, max_image_tokens);

/// Kill-switch for the LOCAL-layer sliding-window ring cache (A/B validation).
const enable_local_ring = true;

/// Uniform per-layer KV dims (gemma3 is uniform); caller owns the slice.
fn kvDims(cfg: Config, alloc: std.mem.Allocator) ![]usize {
    const dims = try alloc.alloc(usize, cfg.n_layers);
    for (dims) |*d| d.* = cfg.kvDim();
    return dims;
}

/// Per-layer KV ring rows: LOCAL (sliding-window) layers get a fixed ring;
/// GLOBAL layers get 0 (full context). Caller owns the slice.
fn kvRings(cfg: Config, alloc: std.mem.Allocator) ![]usize {
    const rings = try alloc.alloc(usize, cfg.n_layers);
    for (rings, 0..) |*r, l| {
        r.* = if (enable_local_ring and !cfg.isGlobal(l)) cfg.sliding_window + max_batch else 0;
    }
    return rings;
}

pub const Config = struct {
    n_layers: usize,
    hidden: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    intermediate: usize,
    vocab: usize,
    rms_eps: f32,
    /// Global-layer RoPE (theta, linear position scale = 1/factor).
    rope_theta: f64,
    rope_freq_scale: f64,
    /// Local (sliding-window) RoPE.
    rope_theta_local: f64,
    /// Sliding-window size (local layers); every `swa_pattern`-th layer is
    /// global (full attention).
    sliding_window: usize,
    swa_pattern: usize,

    pub fn qDim(self: Config) usize {
        return self.n_heads * self.head_dim;
    }
    pub fn kvDim(self: Config) usize {
        return self.n_kv_heads * self.head_dim;
    }
    /// Input-embedding normalizer (llama.cpp: sqrtf(n_embd), f32).
    pub fn embedScale(self: Config) f32 {
        return @sqrt(@as(f32, @floatFromInt(self.hidden)));
    }
    /// Global (full-attention) layer? The rest are local sliding-window.
    pub fn isGlobal(self: Config, l: usize) bool {
        return (l % self.swa_pattern) == self.swa_pattern - 1;
    }

    pub fn detect(g: *const Gguf) !Config {
        const arch = g.getStr("general.architecture") orelse return error.UnknownModelConfig;
        if (!std.mem.eql(u8, arch, "gemma3")) return error.UnknownModelConfig;

        const key = struct {
            fn u(gg: *const Gguf, comptime name: []const u8) !usize {
                return @intCast(gg.getUint("gemma3." ++ name) orelse return error.UnknownModelConfig);
            }
        };
        const head_dim = try key.u(g, "attention.key_length");
        if ((g.getUint("gemma3.attention.value_length") orelse head_dim) != head_dim)
            return error.UnknownModelConfig;

        const embed = g.get("embed_tokens.weight") orelse return error.UnknownModelConfig;
        const eshape = embed.info.shape.slice();
        if (eshape.len != 2) return error.UnknownModelConfig;

        // Linear rope scaling (position interpolation) applies to GLOBAL
        // layers only; local layers keep scale 1.0 (llama.cpp gemma3).
        var freq_scale: f64 = 1.0;
        if (g.getStr("gemma3.rope.scaling.type")) |st| {
            if (std.mem.eql(u8, st, "linear")) {
                const factor = g.getFloat("gemma3.rope.scaling.factor") orelse 1.0;
                if (factor != 0) freq_scale = 1.0 / factor;
            }
        }

        return .{
            .n_layers = try key.u(g, "block_count"),
            .hidden = @intCast(eshape[1]),
            .n_heads = try key.u(g, "attention.head_count"),
            .n_kv_heads = try key.u(g, "attention.head_count_kv"),
            .head_dim = head_dim,
            .intermediate = try key.u(g, "feed_forward_length"),
            .vocab = @intCast(eshape[0]),
            .rms_eps = @floatCast(g.getFloat("gemma3.attention.layer_norm_rms_epsilon") orelse 1e-6),
            .rope_theta = g.getFloat("gemma3.rope.freq_base") orelse 1e6,
            .rope_freq_scale = freq_scale,
            // llama.cpp hardcodes rope_freq_base_train_swa = 10000 for gemma3
            // unless the gguf carries a swa base key.
            .rope_theta_local = g.getFloat("gemma3.rope.freq_base_swa") orelse 10000.0,
            .sliding_window = @intCast(g.getUint("gemma3.attention.sliding_window") orelse 0),
            .swa_pattern = @intCast(g.getUint("gemma3.attention.sliding_window_pattern") orelse 6),
        };
    }
};

const Layer = struct {
    input_norm: []const f32,
    q: Weight,
    k: Weight,
    v: Weight,
    o: Weight,
    q_norm: []const f32, // [head_dim]
    k_norm: []const f32,
    post_attn_norm: []const f32,
    pre_ffn_norm: []const f32,
    post_ffn_norm: []const f32,
    gate: Weight,
    up: Weight,
    down: Weight,
};

pub const Model = struct {
    arena: std.heap.ArenaAllocator,
    cfg: Config,
    embed: Weight,
    /// LM head: tied to `embed` (Gemma ships no output.weight).
    head: Weight,
    layers: []Layer,
    final_norm: []const f32,

    pub fn load(gpa: std.mem.Allocator, g: *const Gguf) !Model {
        const cfg = try Config.detect(g);
        const store: WeightStore = .{ .gguf = g };
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();

        const embed = try loader.matrix(store, "embed_tokens.weight", cfg.vocab, cfg.hidden);
        // Tied head, unless an explicit output.weight is present.
        const head = if (store.get("lm_head.weight")) |_|
            try loader.matrix(store, "lm_head.weight", cfg.vocab, cfg.hidden)
        else
            embed;
        const final_norm = try loader.vector(alloc, store, "norm.weight", cfg.hidden);

        const layers = try alloc.alloc(Layer, cfg.n_layers);
        for (layers, 0..) |*layer, l| {
            layer.* = .{
                .input_norm = try loader.indexedVector(alloc, store, "layers.", l, "input_layernorm.weight", cfg.hidden),
                .q = try loader.indexedMatrix(store, "layers.", l, "self_attn.q_proj.weight", cfg.qDim(), cfg.hidden),
                .k = try loader.indexedMatrix(store, "layers.", l, "self_attn.k_proj.weight", cfg.kvDim(), cfg.hidden),
                .v = try loader.indexedMatrix(store, "layers.", l, "self_attn.v_proj.weight", cfg.kvDim(), cfg.hidden),
                .o = try loader.indexedMatrix(store, "layers.", l, "self_attn.o_proj.weight", cfg.hidden, cfg.qDim()),
                .q_norm = try loader.indexedVector(alloc, store, "layers.", l, "self_attn.q_norm.weight", cfg.head_dim),
                .k_norm = try loader.indexedVector(alloc, store, "layers.", l, "self_attn.k_norm.weight", cfg.head_dim),
                .post_attn_norm = try loader.indexedVector(alloc, store, "layers.", l, "post_attention_layernorm.weight", cfg.hidden),
                .pre_ffn_norm = try loader.indexedVector(alloc, store, "layers.", l, "pre_feedforward_layernorm.weight", cfg.hidden),
                .post_ffn_norm = try loader.indexedVector(alloc, store, "layers.", l, "post_feedforward_layernorm.weight", cfg.hidden),
                .gate = try loader.indexedMatrix(store, "layers.", l, "mlp.gate_proj.weight", cfg.intermediate, cfg.hidden),
                .up = try loader.indexedMatrix(store, "layers.", l, "mlp.up_proj.weight", cfg.intermediate, cfg.hidden),
                .down = try loader.indexedMatrix(store, "layers.", l, "mlp.down_proj.weight", cfg.hidden, cfg.intermediate),
            };
        }

        return .{ .arena = arena, .cfg = cfg, .embed = embed, .head = head, .layers = layers, .final_norm = final_norm };
    }

    pub fn deinit(self: *Model) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// Forward `ids` at positions [cache.len, cache.len + ids.len), appending
    /// K/V. `freqs_global`/`freqs_local` must cover the final position. When
    /// `out` ([n * hidden], n <= ids.len) is set it receives the final-normed
    /// hidden states of the last n positions (LM-head ready).
    pub fn forwardCached(
        self: *const Model,
        io: std.Io,
        gpa: std.mem.Allocator,
        ids: []const u32,
        cache: *PerLayerKvCache,
        freqs_global: ops.rope.Freqs,
        freqs_local: ops.rope.Freqs,
        out: ?[]f32,
    ) !void {
        const cfg = self.cfg;
        const seq = ids.len;
        std.debug.assert(seq > 0 and seq <= cache.remaining());
        std.debug.assert(cache.n_layers == cfg.n_layers);

        const x = try gpa.alloc(f32, seq * cfg.hidden);
        defer gpa.free(x);
        try qwen3.embedTokens(self.embed, ids, x);
        const scale = cfg.embedScale();
        for (x) |*v| v.* *= scale;
        try self.forwardHidden(io, gpa, x, cache, freqs_global, freqs_local, out, false);
    }

    /// forwardCached over PRE-EMBEDDED input hidden states `x` ([seq*hidden],
    /// mutated in place). Used for image-token rows (SigLIP projector output,
    /// injected UNSCALED — Gemma multiplies only text embeddings by
    /// sqrt(hidden)). `out` semantics match forwardCached.
    pub fn forwardHidden(
        self: *const Model,
        io: std.Io,
        gpa: std.mem.Allocator,
        x: []f32,
        cache: *PerLayerKvCache,
        freqs_global: ops.rope.Freqs,
        freqs_local: ops.rope.Freqs,
        out: ?[]f32,
        // Bidirectional: the whole `x` is one image-token block that must attend
        // itself in full, so it is prefilled in a SINGLE un-chunked pass (a
        // Gemma image block, <= max_image_tokens, always fits the local ring).
        bidirectional: bool,
    ) !void {
        const cfg = self.cfg;
        const seq = x.len / cfg.hidden;
        std.debug.assert(seq > 0 and seq <= cache.remaining());
        std.debug.assert(cache.n_layers == cfg.n_layers);
        // A bidirectional block cannot be split across chunks (a later chunk's
        // KV is not committed when an earlier chunk runs), so it goes in one
        // pass; the local ring is sized to hold `window + max_image_tokens`.
        std.debug.assert(!bidirectional or seq <= max_image_tokens);

        // Prefill in prefill_chunk-sized batches so a LOCAL layer's ring never
        // holds more than `window + prefill_chunk` live positions at once
        // (chunked prefill is token-identical: attention is causal, so a later
        // chunk only reads earlier chunks' committed KV). A bidirectional image
        // block runs as one chunk of the whole `seq`.
        const chunk = if (bidirectional) seq else prefill_chunk;
        var s = try Scratch.init(gpa, @min(seq, chunk), cfg);
        defer s.deinit(gpa);
        var off: usize = 0;
        while (off < seq) {
            const n = @min(chunk, seq - off);
            const xc = x[off * cfg.hidden ..][0 .. n * cfg.hidden];
            for (self.layers, 0..) |*layer, l| {
                try layerForward(io, gpa, cfg, layer, xc, n, freqs_global, freqs_local, cache, l, bidirectional, &s);
            }
            cache.commit(n);
            off += n;
        }

        if (out) |o| {
            std.debug.assert(o.len % cfg.hidden == 0);
            const n = o.len / cfg.hidden;
            std.debug.assert(n >= 1 and n <= seq);
            ops.norm.rmsNorm(o, x[(seq - n) * cfg.hidden ..][0 .. n * cfg.hidden], self.final_norm, cfg.rms_eps);
        }
    }
};

/// One Gemma layer over `x` [seq, hidden], residuals added in place. `x`
/// holds only the `seq` new tokens (at absolute positions cache.len..).
/// Public so the CUDA hybrid CPU/GPU split (gemma3_cuda) can run host-resident
/// layers through the exact same code path. `cache` is `anytype` so it accepts
/// both the pure-CPU `PerLayerKvCache` (with LOCAL rings) and gemma3_cuda's
/// uniform `KvCache` host shadow (ringOf → 0, so linear/full context).
pub fn layerForward(
    io: std.Io,
    gpa: std.mem.Allocator,
    cfg: Config,
    layer: *const Layer,
    x: []f32,
    seq: usize,
    freqs_global: ops.rope.Freqs,
    freqs_local: ops.rope.Freqs,
    cache: anytype,
    l: usize,
    bidirectional: bool,
    s: *Scratch,
) !void {
    const global = cfg.isGlobal(l);
    const dims: transformer.Dims = .{
        .hidden = cfg.hidden,
        .n_heads = cfg.n_heads,
        .n_kv = cfg.n_kv_heads,
        .head_dim = cfg.head_dim,
        .q_dim = cfg.qDim(),
        .kv_dim = cfg.kvDim(),
        .intermediate = cfg.intermediate,
        // Local layers use a sliding-window causal mask; global attend all.
        .sliding_window = if (global) 0 else cfg.sliding_window,
    };
    const freqs = if (global) freqs_global else freqs_local;
    try transformer.layerForward(transformer.gemma3_spec, .cached, io, gpa, layer, x, seq, dims, freqs, cfg.rms_eps, cache, l, cache.len, bidirectional, s);
}

/// Per-forward activation buffers, sized for `seq` tokens of `cfg`. Public so
/// the CUDA hybrid split can allocate one for its host-resident layers.
pub const Scratch = struct {
    normed: []f32,
    tmp: []f32,
    q: []f32,
    k: []f32,
    v: []f32,
    attn_out: []f32,
    gate: []f32,
    up: []f32,

    /// A borrowed view of the first `seq` rows of a larger scratch (no alloc).
    /// The CUDA split sizes its scratch to a full prefill chunk once, then views
    /// it down to the actual chunk length each call — `layerForward`'s ops
    /// require exact-length slices, so passing the oversized buffer would trip a
    /// length assert (the qwen35 split hit this; same fix here). Never deinit a
    /// view — it aliases the parent scratch's memory.
    pub fn viewSeq(self: *const Scratch, seq: usize, cfg: Config) Scratch {
        return .{
            .normed = self.normed[0 .. seq * cfg.hidden],
            .tmp = self.tmp[0 .. seq * cfg.hidden],
            .q = self.q[0 .. seq * cfg.qDim()],
            .k = self.k[0 .. seq * cfg.kvDim()],
            .v = self.v[0 .. seq * cfg.kvDim()],
            .attn_out = self.attn_out[0 .. seq * cfg.qDim()],
            .gate = self.gate[0 .. seq * cfg.intermediate],
            .up = self.up[0 .. seq * cfg.intermediate],
        };
    }

    pub fn init(gpa: std.mem.Allocator, seq: usize, cfg: Config) !Scratch {
        var s: Scratch = undefined;
        var done: usize = 0;
        errdefer inline for (@typeInfo(Scratch).@"struct".fields, 0..) |f, i| {
            if (i < done) gpa.free(@field(s, f.name));
        };
        const sizes = [_]usize{
            seq * cfg.hidden, // normed
            seq * cfg.hidden, // tmp
            seq * cfg.qDim(), // q
            seq * cfg.kvDim(), // k
            seq * cfg.kvDim(), // v
            seq * cfg.qDim(), // attn_out
            seq * cfg.intermediate, // gate
            seq * cfg.intermediate, // up
        };
        inline for (@typeInfo(Scratch).@"struct".fields, 0..) |f, i| {
            @field(s, f.name) = try gpa.alloc(f32, sizes[i]);
            done = i + 1;
        }
        return s;
    }

    pub fn deinit(self: *Scratch, gpa: std.mem.Allocator) void {
        inline for (@typeInfo(Scratch).@"struct".fields) |f| gpa.free(@field(self, f.name));
        self.* = undefined;
    }
};

/// CPU stepper for the engine loop (mirrors engine.CpuModel). Speculative
/// decoding is unsupported for now (no stepAll/truncate).
pub const CpuModel = struct {
    lm: *const Model,
    gpa: std.mem.Allocator,
    cache: PerLayerKvCache,
    /// Global-layer (index 0) and local/sliding-window-layer (index 1) RoPE
    /// tables, rebuilt together on context growth.
    rope: ops.rope.RopeTables(2),
    last_hidden: []f32,
    max_capacity: usize,
    /// Io for the CPU matmuls; set by step, or by the caller before an image
    /// prefill (which happens before the first step).
    io: std.Io = undefined,

    pub fn init(gpa: std.mem.Allocator, lm: *const Model, cap: kv_cache_mod.Capacity) !CpuModel {
        const cfg = lm.cfg;
        const dims = try kvDims(cfg, gpa);
        defer gpa.free(dims);
        const rings = try kvRings(cfg, gpa);
        defer gpa.free(rings);
        var cache = try PerLayerKvCache.init(gpa, cap.initial, dims, rings, cap.kv_dtype);
        errdefer cache.deinit(gpa);
        var rope = try ops.rope.RopeTables(2).init(gpa, .{
            .{ .head_dim = cfg.head_dim, .theta = cfg.rope_theta, .freq_scale = cfg.rope_freq_scale },
            .{ .head_dim = cfg.head_dim, .theta = cfg.rope_theta_local },
        }, cap.initial);
        errdefer rope.deinit(gpa);
        const last_hidden = try gpa.alloc(f32, cfg.hidden);
        return .{ .lm = lm, .gpa = gpa, .cache = cache, .rope = rope, .last_hidden = last_hidden, .max_capacity = cap.max };
    }

    pub fn capacityMax(self: *const CpuModel) usize {
        return self.max_capacity;
    }

    pub fn ensureCapacity(self: *CpuModel, min_rows: usize) !void {
        const target = (try kv_cache_mod.growPlan(self.cache.capacity, self.max_capacity, min_rows)) orelse return;
        self.cache.grow(self.gpa, target) catch return error.ContextFull;
        self.rope.regrow(self.gpa, target) catch return error.ContextFull;
    }

    pub fn deinit(self: *CpuModel) void {
        self.cache.deinit(self.gpa);
        self.rope.deinit(self.gpa);
        self.gpa.free(self.last_hidden);
        self.* = undefined;
    }

    pub fn cached(self: *const CpuModel) usize {
        return self.cache.len;
    }

    pub fn remaining(self: *const CpuModel) usize {
        return self.cache.remaining();
    }

    pub fn vocab(self: *const CpuModel) usize {
        return self.lm.cfg.vocab;
    }

    pub fn step(self: *CpuModel, io: std.Io, ids_new: []const u32, logits: []f32) !void {
        self.io = io;
        try self.lm.forwardCached(io, self.gpa, ids_new, &self.cache, self.rope.get(0), self.rope.get(1), self.last_hidden);
        try ops.matmul.matmul(io, self.gpa, logits, self.last_hidden, 1, self.lm.head, null);
    }

    /// Prefill text tokens (no logits) — for interleaving with prefillImage.
    /// Uses `self.io` (set by step, or by the caller before an image turn).
    pub fn prefill(self: *CpuModel, ids: []const u32) !void {
        try self.lm.forwardCached(self.io, self.gpa, ids, &self.cache, self.rope.get(0), self.rope.get(1), null);
    }

    /// Prefill one image's projected embeddings ([grid_w*grid_h][hidden],
    /// injected UNSCALED) at the next sequential positions. grid dims are
    /// carried for interface parity (gemma is always 16x16 = 256).
    pub fn prefillImage(self: *CpuModel, embeds: []const f32, grid_w: usize, grid_h: usize) !void {
        _ = grid_w;
        _ = grid_h;
        const x = try self.gpa.dupe(f32, embeds);
        defer self.gpa.free(x);
        // Image tokens attend bidirectionally within the block (llama.cpp marks
        // the image span non-causal); one un-chunked pass, causal to the prefix.
        try self.lm.forwardHidden(self.io, self.gpa, x, &self.cache, self.rope.get(0), self.rope.get(1), null, true);
    }
};

// --- tests -----------------------------------------------------------------

// Config + weight wiring against the real Gemma 3 12B checkpoint; skipped
// when absent. Load-only — generation is validated end-to-end via tp-llm
// against llama.cpp (a Debug 12B forward is too slow for the suite).
test "gemma3 loads from real gemma3-12b gguf" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const path = "/home/qt/genai/lmstudio/models/mradermacher/Gemma-3-Starshine-12B-Alt-GGUF/Gemma-3-Starshine-12B-Alt.Q4_K_M.gguf";
    std.Io.Dir.cwd().access(io, path, .{}) catch return error.SkipZigTest;

    var g = try Gguf.open(gpa, io, path);
    defer g.deinit();
    var lm = try Model.load(gpa, &g);
    defer lm.deinit();

    const cfg = lm.cfg;
    try std.testing.expectEqual(@as(usize, 48), cfg.n_layers);
    try std.testing.expectEqual(@as(usize, 3840), cfg.hidden);
    try std.testing.expectEqual(@as(usize, 16), cfg.n_heads);
    try std.testing.expectEqual(@as(usize, 8), cfg.n_kv_heads);
    try std.testing.expectEqual(@as(usize, 256), cfg.head_dim);
    try std.testing.expectEqual(@as(usize, 15360), cfg.intermediate);
    try std.testing.expectEqual(@as(usize, 262145), cfg.vocab);
    try std.testing.expectEqual(@as(f64, 1e6), cfg.rope_theta);
    try std.testing.expectApproxEqAbs(@as(f64, 0.125), cfg.rope_freq_scale, 1e-9);
    try std.testing.expectEqual(@as(f64, 10000.0), cfg.rope_theta_local);
    try std.testing.expectEqual(@as(usize, 1024), cfg.sliding_window);
    try std.testing.expectEqual(@as(usize, 6), cfg.swa_pattern);
    // Layers 5, 11, ... are global; the rest local.
    try std.testing.expect(!cfg.isGlobal(0) and cfg.isGlobal(5) and !cfg.isGlobal(6) and cfg.isGlobal(47));
    // Tied head.
    try std.testing.expectEqual(lm.embed.bytes.ptr, lm.head.bytes.ptr);
    try std.testing.expectEqual(@as(usize, 3840), lm.layers[0].input_norm.len);
    try std.testing.expectEqual(@as(usize, 256), lm.layers[0].q_norm.len);
}
