//! Gemma 4 language model (GGUF arch "gemma4"), text-only path. Ported from
//! llama.cpp (src/models/gemma4.cpp + the GEMMA4 hparams in llama-model.cpp).
//!
//! Shares Gemma 3's "sandwich" norms (input / post-attention / pre-ffw /
//! post-ffw RMSNorm), sqrt(hidden) embedding scale, per-head QK RMS-norm,
//! GeGLU FFN, tied LM head, and the local/global sliding-window layer split
//! (period 6: every 6th layer is GLOBAL full-attention, the rest LOCAL). What
//! is NEW in Gemma 4 (this is the dense 12B variant — no MoE, no per-layer
//! input embeddings):
//!
//!   - Per-layer attention geometry. LOCAL (sliding-window) layers use
//!     head_dim 256 with 8 KV heads (2048-wide KV); GLOBAL layers use head_dim
//!     512 with a single KV head (MQA, 512-wide KV). Query heads are always 16,
//!     so q/o widths also differ per layer. The KV cache therefore has a
//!     per-layer stride (see PerLayerKvCache).
//!   - Attention score scale is 1.0 (NOT 1/sqrt(head_dim)) — llama.cpp
//!     f_attention_scale; the scale is folded into training via the QK norms.
//!   - V is RMS-normalized per head_dim WITHOUT a learned weight (Q/K keep
//!     their weighted norms); V is not rotated.
//!   - RoPE: LOCAL layers theta 1e4 over head_dim 256; GLOBAL layers theta 1e6
//!     over head_dim 512 with per-dimension frequency factors (rope_freqs.weight
//!     — "proportional"/long-context RoPE), replacing Gemma 3's scalar 1/8.
//!   - A per-layer scalar `out_scale` multiplies the whole layer output.
//!   - Final logits are tanh-softcapped at 30, and `suppress_tokens` are forced
//!     to -inf (the checkpoint's <image>/<audio> placeholder ids).
//!
//! Weights stay in checkpoint dtype (GGUF block quants; the 12B QAT is Q4_0)
//! and dequantize inside the GEMM; the Gguf mapping must outlive the model.

const std = @import("std");
const gguf_mod = @import("../gguf.zig");
const weights_mod = @import("../weights.zig");
const qwen3 = @import("qwen3.zig");
const ops = @import("../ops.zig");
const kv_cache_mod = @import("../llm/kv_cache.zig");

const Gguf = gguf_mod.Gguf;
const WeightStore = weights_mod.WeightStore;
const Weight = ops.matmul.Weight;
const PerLayerKvCache = kv_cache_mod.PerLayerKvCache;

pub const Config = struct {
    n_layers: usize,
    hidden: usize,
    n_heads: usize,
    intermediate: usize,
    vocab: usize,
    rms_eps: f32,
    /// GLOBAL (full-attention) layer geometry.
    head_dim_global: usize,
    n_kv_global: usize,
    /// LOCAL (sliding-window) layer geometry.
    head_dim_local: usize,
    n_kv_local: usize,
    /// Global-layer RoPE base; local-layer RoPE base.
    rope_theta: f64,
    rope_theta_local: f64,
    /// Sliding-window size (local layers); every `swa_pattern`-th layer is
    /// global (full attention).
    sliding_window: usize,
    swa_pattern: usize,
    /// tanh logit softcapping (0 = disabled).
    final_logit_softcap: f32,

    /// Global (full-attention) layer? The rest are local sliding-window.
    pub fn isGlobal(self: Config, l: usize) bool {
        return (l % self.swa_pattern) == self.swa_pattern - 1;
    }
    pub fn headDim(self: Config, l: usize) usize {
        return if (self.isGlobal(l)) self.head_dim_global else self.head_dim_local;
    }
    pub fn nKv(self: Config, l: usize) usize {
        return if (self.isGlobal(l)) self.n_kv_global else self.n_kv_local;
    }
    pub fn qDim(self: Config, l: usize) usize {
        return self.n_heads * self.headDim(l);
    }
    pub fn kvDim(self: Config, l: usize) usize {
        return self.nKv(l) * self.headDim(l);
    }
    /// Largest per-layer q/o and kv widths, for sizing shared scratch.
    pub fn maxQDim(self: Config) usize {
        return self.n_heads * @max(self.head_dim_global, self.head_dim_local);
    }
    pub fn maxKvDim(self: Config) usize {
        return @max(self.n_kv_global * self.head_dim_global, self.n_kv_local * self.head_dim_local);
    }
    /// Input-embedding normalizer (llama.cpp: sqrtf(n_embd), f32).
    pub fn embedScale(self: Config) f32 {
        return @sqrt(@as(f32, @floatFromInt(self.hidden)));
    }
    /// Per-layer KV dims, for building the KV cache (caller owns the slice).
    pub fn kvDims(self: Config, alloc: std.mem.Allocator) ![]usize {
        const dims = try alloc.alloc(usize, self.n_layers);
        for (dims, 0..) |*d, l| d.* = self.kvDim(l);
        return dims;
    }

    pub fn detect(g: *const Gguf) !Config {
        const arch = g.getStr("general.architecture") orelse return error.UnknownModelConfig;
        if (!std.mem.eql(u8, arch, "gemma4")) return error.UnknownModelConfig;

        const key = struct {
            fn u(gg: *const Gguf, comptime name: []const u8) !usize {
                return @intCast(gg.getUint("gemma4." ++ name) orelse return error.UnknownModelConfig);
            }
        };

        const n_layers = try key.u(g, "block_count");
        const head_dim_global = try key.u(g, "attention.key_length");
        const head_dim_local = @as(usize, @intCast(g.getUint("gemma4.attention.key_length_swa") orelse head_dim_global));
        if ((g.getUint("gemma4.attention.value_length") orelse head_dim_global) != head_dim_global)
            return error.UnknownModelConfig;

        const embed = g.get("embed_tokens.weight") orelse return error.UnknownModelConfig;
        const eshape = embed.info.shape.slice();
        if (eshape.len != 2) return error.UnknownModelConfig;

        // Per-layer KV-head counts and the sliding-window mask arrays. Gemma 4
        // ships a regular period-6 pattern (5 local, 1 global); derive the
        // period and the two KV-head counts, asserting the arrays match.
        var kv_buf: [max_layers]usize = undefined;
        var swa_buf: [max_layers]usize = undefined;
        if (n_layers > max_layers) return error.UnknownModelConfig;
        const kv_counts = try readUintArray(g, "gemma4.attention.head_count_kv", kv_buf[0..n_layers]);
        const swa_mask = try readUintArray(g, "gemma4.attention.sliding_window_pattern", swa_buf[0..n_layers]);
        var swa_pattern: usize = 0;
        for (0..n_layers) |l| {
            if (swa_mask[l] == 0) { // first GLOBAL layer -> period = l + 1
                swa_pattern = l + 1;
                break;
            }
        }
        if (swa_pattern == 0) return error.UnknownModelConfig; // no global layer found
        var n_kv_global: usize = 0;
        var n_kv_local: usize = 0;
        for (0..n_layers) |l| {
            const is_global = (l % swa_pattern) == swa_pattern - 1;
            // is_global must correspond to sliding_window_pattern == 0.
            if (is_global != (swa_mask[l] == 0)) return error.UnknownModelConfig;
            if (is_global) {
                if (n_kv_global == 0) n_kv_global = kv_counts[l] else if (kv_counts[l] != n_kv_global) return error.UnknownModelConfig;
            } else {
                if (n_kv_local == 0) n_kv_local = kv_counts[l] else if (kv_counts[l] != n_kv_local) return error.UnknownModelConfig;
            }
        }

        return .{
            .n_layers = n_layers,
            .hidden = @intCast(eshape[1]),
            .n_heads = try key.u(g, "attention.head_count"),
            .intermediate = try key.u(g, "feed_forward_length"),
            .vocab = @intCast(eshape[0]),
            .rms_eps = @floatCast(g.getFloat("gemma4.attention.layer_norm_rms_epsilon") orelse 1e-6),
            .head_dim_global = head_dim_global,
            .n_kv_global = n_kv_global,
            .head_dim_local = head_dim_local,
            .n_kv_local = n_kv_local,
            .rope_theta = g.getFloat("gemma4.rope.freq_base") orelse 1e6,
            .rope_theta_local = g.getFloat("gemma4.rope.freq_base_swa") orelse 10000.0,
            .sliding_window = @intCast(g.getUint("gemma4.attention.sliding_window") orelse 0),
            .swa_pattern = swa_pattern,
            .final_logit_softcap = @floatCast(g.getFloat("gemma4.final_logit_softcapping") orelse 0),
        };
    }
};

/// Upper bound on layer count for the stack-allocated per-layer arrays read in
/// Config.detect (Gemma 4 tops out at 60 layers for the 31B).
const max_layers = 128;

/// Read a fixed-length uint/bool array KV into the caller's `out` buffer.
fn readUintArray(g: *const Gguf, key: []const u8, out: []usize) ![]usize {
    const arr = g.getArr(key) orelse return error.UnknownModelConfig;
    if (arr.len != out.len) return error.UnknownModelConfig;
    var it = arr.iterate();
    var i: usize = 0;
    while (it.next()) |v| : (i += 1) {
        out[i] = switch (v) {
            .uint => |u| @intCast(u),
            .int => |s| @intCast(s),
            .boolean => |b| @intFromBool(b),
            else => return error.UnknownModelConfig,
        };
    }
    return out;
}

const Layer = struct {
    input_norm: []const f32,
    q: Weight,
    k: Weight,
    /// V projection. Absent on GLOBAL layers (llama.cpp: reuse the raw K
    /// projection output as V — see layerForward).
    v: ?Weight,
    o: Weight,
    q_norm: []const f32, // [head_dim]
    k_norm: []const f32,
    post_attn_norm: []const f32,
    pre_ffn_norm: []const f32,
    post_ffn_norm: []const f32,
    gate: Weight,
    up: Weight,
    down: Weight,
    /// Per-layer output scalar (llama.cpp layer_output_scale).
    out_scale: f32,
};

pub const Model = struct {
    arena: std.heap.ArenaAllocator,
    cfg: Config,
    embed: Weight,
    /// LM head: tied to `embed` (Gemma ships no output.weight).
    head: Weight,
    layers: []Layer,
    final_norm: []const f32,
    /// Global-layer RoPE frequency factors (rope_freqs.weight), head_dim_global/2.
    rope_freqs: []const f32,
    /// Token ids forced to -inf in the final logits (checkpoint placeholders).
    suppress_tokens: []const u32,

    pub fn load(gpa: std.mem.Allocator, g: *const Gguf) !Model {
        const cfg = try Config.detect(g);
        const store: WeightStore = .{ .gguf = g };
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();

        const embed = try loadMatrix(store, "embed_tokens.weight", cfg.vocab, cfg.hidden);
        const head = if (store.get("lm_head.weight")) |_|
            try loadMatrix(store, "lm_head.weight", cfg.vocab, cfg.hidden)
        else
            embed;
        const final_norm = try loadVec(alloc, store, "norm.weight", cfg.hidden);
        const rope_freqs = try loadVec(alloc, store, "rope_freqs.weight", cfg.head_dim_global / 2);

        const layers = try alloc.alloc(Layer, cfg.n_layers);
        for (layers, 0..) |*layer, l| {
            const hd = cfg.headDim(l);
            layer.* = .{
                .input_norm = try loadLayerVec(alloc, store, l, "input_layernorm.weight", cfg.hidden),
                .q = try loadLayerMatrix(store, l, "self_attn.q_proj.weight", cfg.qDim(l), cfg.hidden),
                .k = try loadLayerMatrix(store, l, "self_attn.k_proj.weight", cfg.kvDim(l), cfg.hidden),
                .v = loadLayerMatrix(store, l, "self_attn.v_proj.weight", cfg.kvDim(l), cfg.hidden) catch |e| switch (e) {
                    error.MissingTensor => null, // global layers reuse K as V
                    else => return e,
                },
                .o = try loadLayerMatrix(store, l, "self_attn.o_proj.weight", cfg.hidden, cfg.qDim(l)),
                .q_norm = try loadLayerVec(alloc, store, l, "self_attn.q_norm.weight", hd),
                .k_norm = try loadLayerVec(alloc, store, l, "self_attn.k_norm.weight", hd),
                .post_attn_norm = try loadLayerVec(alloc, store, l, "post_attention_layernorm.weight", cfg.hidden),
                .pre_ffn_norm = try loadLayerVec(alloc, store, l, "pre_feedforward_layernorm.weight", cfg.hidden),
                .post_ffn_norm = try loadLayerVec(alloc, store, l, "post_feedforward_layernorm.weight", cfg.hidden),
                .gate = try loadLayerMatrix(store, l, "mlp.gate_proj.weight", cfg.intermediate, cfg.hidden),
                .up = try loadLayerMatrix(store, l, "mlp.up_proj.weight", cfg.intermediate, cfg.hidden),
                .down = try loadLayerMatrix(store, l, "mlp.down_proj.weight", cfg.hidden, cfg.intermediate),
                .out_scale = (try loadLayerVec(alloc, store, l, "out_scale.weight", 1))[0],
            };
        }

        const suppress = try loadSuppressTokens(alloc, g);

        return .{
            .arena = arena,
            .cfg = cfg,
            .embed = embed,
            .head = head,
            .layers = layers,
            .final_norm = final_norm,
            .rope_freqs = rope_freqs,
            .suppress_tokens = suppress,
        };
    }

    pub fn deinit(self: *Model) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// Forward `ids` at positions [cache.len, cache.len + ids.len). When `out`
    /// ([n * hidden], n <= ids.len) is set it receives the final-normed hidden
    /// states of the last n positions (LM-head ready).
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

        const x = try gpa.alloc(f32, seq * cfg.hidden);
        defer gpa.free(x);
        try qwen3.embedTokens(self.embed, ids, x);
        const scale = cfg.embedScale();
        for (x) |*v| v.* *= scale;
        try self.forwardHidden(io, gpa, x, cache, freqs_global, freqs_local, out);
    }

    /// forwardCached over PRE-EMBEDDED input hidden states `x` ([seq*hidden],
    /// mutated in place). Reserved for future image-token rows (injected
    /// unscaled). `out` semantics match forwardCached.
    pub fn forwardHidden(
        self: *const Model,
        io: std.Io,
        gpa: std.mem.Allocator,
        x: []f32,
        cache: *PerLayerKvCache,
        freqs_global: ops.rope.Freqs,
        freqs_local: ops.rope.Freqs,
        out: ?[]f32,
    ) !void {
        const cfg = self.cfg;
        const seq = x.len / cfg.hidden;
        std.debug.assert(seq > 0 and seq <= cache.remaining());

        var s = try Scratch.init(gpa, seq, cfg);
        defer s.deinit(gpa);

        for (self.layers, 0..) |*layer, l| {
            try layerForward(io, gpa, cfg, layer, x, seq, freqs_global, freqs_local, cache, l, &s);
        }
        cache.commit(seq);

        if (out) |o| {
            std.debug.assert(o.len % cfg.hidden == 0);
            const n = o.len / cfg.hidden;
            std.debug.assert(n >= 1 and n <= seq);
            ops.norm.rmsNorm(o, x[(seq - n) * cfg.hidden ..][0 .. n * cfg.hidden], self.final_norm, cfg.rms_eps);
        }
    }

    /// Apply tanh logit softcapping and suppress-token masking to a row of
    /// vocab logits (llama.cpp gemma4 result_output tail).
    pub fn finalizeLogits(self: *const Model, logits: []f32) void {
        if (self.cfg.final_logit_softcap != 0) {
            const c = self.cfg.final_logit_softcap;
            for (logits) |*v| v.* = c * std.math.tanh(v.* / c);
        }
        for (self.suppress_tokens) |id| {
            if (id < logits.len) logits[id] = -std.math.inf(f32);
        }
    }
};

/// One Gemma 4 layer over `x` [seq, hidden], residuals added in place. `x`
/// holds only the `seq` new tokens (at absolute positions cache.len..). The
/// per-layer q/k/v/attn widths are sliced out of the max-sized `Scratch`.
pub fn layerForward(
    io: std.Io,
    gpa: std.mem.Allocator,
    cfg: Config,
    layer: *const Layer,
    x: []f32,
    seq: usize,
    freqs_global: ops.rope.Freqs,
    freqs_local: ops.rope.Freqs,
    cache: *PerLayerKvCache,
    l: usize,
    s: *Scratch,
) !void {
    const pos0 = cache.len;
    const global = cfg.isGlobal(l);
    const hd = cfg.headDim(l);
    const n_kv = cfg.nKv(l);
    const q_dim = cfg.qDim(l);
    const kv_dim = cfg.kvDim(l);
    const freqs = if (global) freqs_global else freqs_local;

    const q = s.q[0 .. seq * q_dim];
    const k = s.k[0 .. seq * kv_dim];
    const v = s.v[0 .. seq * kv_dim];
    const attn_out = s.attn_out[0 .. seq * q_dim];

    // --- Attention ---
    ops.norm.rmsNorm(s.normed, x, layer.input_norm, cfg.rms_eps);
    try ops.matmul.matmul(io, gpa, q, s.normed, seq, layer.q, null);
    try ops.matmul.matmul(io, gpa, k, s.normed, seq, layer.k, null);
    // V: its own projection, or (global layers, no v_proj) the RAW K projection
    // output — copied before k_norm/rope mutate `k`.
    if (layer.v) |vw| {
        try ops.matmul.matmul(io, gpa, v, s.normed, seq, vw, null);
    } else {
        @memcpy(v, k);
    }
    ops.norm.rmsNorm(q, q, layer.q_norm, cfg.rms_eps); // per-head: rows of head_dim
    ops.norm.rmsNorm(k, k, layer.k_norm, cfg.rms_eps);
    ops.norm.rmsNormUnit(v, v, hd, cfg.rms_eps); // V: weightless RMS over head_dim
    ops.rope.applyRotateHalfAt(q, freqs, pos0, seq, cfg.n_heads, hd);
    ops.rope.applyRotateHalfAt(k, freqs, pos0, seq, n_kv, hd);

    cache.write(l, k, v);
    try ops.attention.attention(io, gpa, attn_out, q, cache.kView(l, seq), cache.vView(l, seq), .{
        .seq_q = seq,
        .seq_kv = pos0 + seq,
        .n_heads = cfg.n_heads,
        .n_kv_heads = n_kv,
        .head_dim = hd,
        .causal = true,
        .scale = 1.0, // Gemma 4: no 1/sqrt(head_dim) (QK-norm folds it in)
        .window = if (global) 0 else cfg.sliding_window,
    });
    // o_proj, then post-attention norm on the attn output BEFORE the residual.
    try ops.matmul.matmul(io, gpa, s.tmp, attn_out, seq, layer.o, null);
    ops.norm.rmsNorm(s.tmp, s.tmp, layer.post_attn_norm, cfg.rms_eps);
    for (x, s.tmp) |*xi, ti| xi.* += ti;

    // --- MLP (GeGLU) ---
    ops.norm.rmsNorm(s.normed, x, layer.pre_ffn_norm, cfg.rms_eps);
    try ops.matmul.matmul(io, gpa, s.gate, s.normed, seq, layer.gate, null);
    try ops.matmul.matmul(io, gpa, s.up, s.normed, seq, layer.up, null);
    ops.act.geluTanhMul(s.gate, s.up);
    try ops.matmul.matmul(io, gpa, s.tmp, s.gate, seq, layer.down, null);
    ops.norm.rmsNorm(s.tmp, s.tmp, layer.post_ffn_norm, cfg.rms_eps);
    for (x, s.tmp) |*xi, ti| xi.* += ti;

    // --- Per-layer output scale (whole residual stream) ---
    if (layer.out_scale != 1.0) for (x) |*xi| {
        xi.* *= layer.out_scale;
    };
}

/// Per-forward activation buffers, sized for the LARGEST per-layer widths so
/// one allocation serves every layer (q/o global; kv local). `layerForward`
/// slices down to each layer's exact widths.
pub const Scratch = struct {
    normed: []f32,
    tmp: []f32,
    q: []f32,
    k: []f32,
    v: []f32,
    attn_out: []f32,
    gate: []f32,
    up: []f32,

    pub fn init(gpa: std.mem.Allocator, seq: usize, cfg: Config) !Scratch {
        var s: Scratch = undefined;
        var done: usize = 0;
        errdefer inline for (@typeInfo(Scratch).@"struct".fields, 0..) |f, i| {
            if (i < done) gpa.free(@field(s, f.name));
        };
        const sizes = [_]usize{
            seq * cfg.hidden, // normed
            seq * cfg.hidden, // tmp
            seq * cfg.maxQDim(), // q
            seq * cfg.maxKvDim(), // k
            seq * cfg.maxKvDim(), // v
            seq * cfg.maxQDim(), // attn_out
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

fn loadMatrix(store: WeightStore, name: []const u8, rows: usize, cols: usize) !Weight {
    const view = store.get(name) orelse return error.MissingTensor;
    const shape = view.info.shape.slice();
    if (shape.len != 2 or shape[0] != rows or shape[1] != cols) return error.ShapeMismatch;
    return Weight.init(view.bytes, view.info.dtype, rows, cols);
}

fn loadLayerMatrix(store: WeightStore, l: usize, comptime suffix: []const u8, rows: usize, cols: usize) !Weight {
    var buf: [96]u8 = undefined;
    return loadMatrix(store, try std.fmt.bufPrint(&buf, "layers.{d}." ++ suffix, .{l}), rows, cols);
}

fn loadVec(alloc: std.mem.Allocator, store: WeightStore, name: []const u8, len: usize) ![]f32 {
    const view = store.get(name) orelse return error.MissingTensor;
    if (view.info.elemCount() != len) return error.ShapeMismatch;
    return view.toF32Alloc(alloc);
}

fn loadLayerVec(alloc: std.mem.Allocator, store: WeightStore, l: usize, comptime suffix: []const u8, len: usize) ![]f32 {
    var buf: [96]u8 = undefined;
    return loadVec(alloc, store, try std.fmt.bufPrint(&buf, "layers.{d}." ++ suffix, .{l}), len);
}

/// Read tokenizer.ggml.suppress_tokens ([INT32]) into an owned u32 slice.
fn loadSuppressTokens(alloc: std.mem.Allocator, g: *const Gguf) ![]const u32 {
    const arr = g.getArr("tokenizer.ggml.suppress_tokens") orelse return &.{};
    const out = try alloc.alloc(u32, arr.len);
    var it = arr.iterate();
    var i: usize = 0;
    while (it.next()) |val| : (i += 1) {
        out[i] = switch (val) {
            .uint => |u| @intCast(u),
            .int => |s| @intCast(@max(@as(i64, 0), s)),
            else => 0,
        };
    }
    return out;
}

/// CPU stepper for the engine loop (mirrors engine.CpuModel). Speculative
/// decoding is unsupported (recurrent-free but the per-layer KV cache has no
/// batch-region layout yet).
pub const CpuModel = struct {
    lm: *const Model,
    gpa: std.mem.Allocator,
    cache: PerLayerKvCache,
    freqs_global: ops.rope.Freqs,
    freqs_local: ops.rope.Freqs,
    last_hidden: []f32,
    max_capacity: usize,
    io: std.Io = undefined,

    pub fn init(gpa: std.mem.Allocator, lm: *const Model, cap: kv_cache_mod.Capacity) !CpuModel {
        const cfg = lm.cfg;
        const dims = try cfg.kvDims(gpa);
        defer gpa.free(dims);
        var cache = try PerLayerKvCache.init(gpa, cap.initial, dims);
        errdefer cache.deinit(gpa);
        var fg = try buildGlobalFreqs(gpa, cap.initial, cfg, lm.rope_freqs);
        errdefer fg.deinit(gpa);
        var fl = try ops.rope.rotateHalfFreqsFactored(gpa, cap.initial, cfg.head_dim_local, cfg.rope_theta_local, 1.0, null);
        errdefer fl.deinit(gpa);
        const last_hidden = try gpa.alloc(f32, cfg.hidden);
        return .{ .lm = lm, .gpa = gpa, .cache = cache, .freqs_global = fg, .freqs_local = fl, .last_hidden = last_hidden, .max_capacity = cap.max };
    }

    fn buildGlobalFreqs(gpa: std.mem.Allocator, seq: usize, cfg: Config, rope_freqs: []const f32) !ops.rope.Freqs {
        return ops.rope.rotateHalfFreqsFactored(gpa, seq, cfg.head_dim_global, cfg.rope_theta, 1.0, rope_freqs);
    }

    pub fn capacityMax(self: *const CpuModel) usize {
        return self.max_capacity;
    }

    pub fn ensureCapacity(self: *CpuModel, min_rows: usize) !void {
        if (min_rows <= self.cache.capacity) return;
        if (min_rows > self.max_capacity) return error.ContextFull;
        const cfg = self.lm.cfg;
        const target = kv_cache_mod.growTarget(self.cache.capacity, min_rows, self.max_capacity);
        self.cache.grow(self.gpa, target) catch return error.ContextFull;
        const fg = buildGlobalFreqs(self.gpa, target, cfg, self.lm.rope_freqs) catch return error.ContextFull;
        self.freqs_global.deinit(self.gpa);
        self.freqs_global = fg;
        const fl = ops.rope.rotateHalfFreqsFactored(self.gpa, target, cfg.head_dim_local, cfg.rope_theta_local, 1.0, null) catch return error.ContextFull;
        self.freqs_local.deinit(self.gpa);
        self.freqs_local = fl;
    }

    pub fn deinit(self: *CpuModel) void {
        self.cache.deinit(self.gpa);
        self.freqs_global.deinit(self.gpa);
        self.freqs_local.deinit(self.gpa);
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
        try self.lm.forwardCached(io, self.gpa, ids_new, &self.cache, self.freqs_global, self.freqs_local, self.last_hidden);
        try ops.matmul.matmul(io, self.gpa, logits, self.last_hidden, 1, self.lm.head, null);
        self.lm.finalizeLogits(logits);
    }

    /// Prefill text tokens (no logits) — for interleaving with prefillImage.
    /// Uses `self.io` (set by step, or by the caller before an image turn).
    pub fn prefill(self: *CpuModel, ids: []const u32) !void {
        try self.lm.forwardCached(self.io, self.gpa, ids, &self.cache, self.freqs_global, self.freqs_local, null);
    }

    /// Prefill one image's projected embeddings ([grid_w*grid_h][hidden],
    /// injected UNSCALED — Gemma multiplies only text embeddings by
    /// sqrt(hidden)) at the next sequential positions. grid dims are carried
    /// for interface parity (gemma4 grids are variable, W/48 x H/48).
    pub fn prefillImage(self: *CpuModel, embeds: []const f32, grid_w: usize, grid_h: usize) !void {
        _ = grid_w;
        _ = grid_h;
        const x = try self.gpa.dupe(f32, embeds);
        defer self.gpa.free(x);
        try self.lm.forwardHidden(self.io, self.gpa, x, &self.cache, self.freqs_global, self.freqs_local, null);
    }
};

// --- tests -----------------------------------------------------------------

// Config + weight wiring against the real Gemma 4 12B checkpoint; skipped when
// absent. Load-only — generation is validated end-to-end via tp-llm against
// llama.cpp (a Debug 12B forward is too slow for the suite).
test "gemma4 loads from real gemma4-12b gguf" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const path = "/home/qt/genai/lmstudio/models/gemma-4-12b-it-qat-q4_0.gguf";
    std.Io.Dir.cwd().access(io, path, .{}) catch return error.SkipZigTest;

    var g = try Gguf.open(gpa, io, path);
    defer g.deinit();
    var lm = try Model.load(gpa, &g);
    defer lm.deinit();

    const cfg = lm.cfg;
    try std.testing.expectEqual(@as(usize, 48), cfg.n_layers);
    try std.testing.expectEqual(@as(usize, 3840), cfg.hidden);
    try std.testing.expectEqual(@as(usize, 16), cfg.n_heads);
    try std.testing.expectEqual(@as(usize, 15360), cfg.intermediate);
    try std.testing.expectEqual(@as(usize, 262144), cfg.vocab);
    try std.testing.expectEqual(@as(usize, 512), cfg.head_dim_global);
    try std.testing.expectEqual(@as(usize, 1), cfg.n_kv_global);
    try std.testing.expectEqual(@as(usize, 256), cfg.head_dim_local);
    try std.testing.expectEqual(@as(usize, 8), cfg.n_kv_local);
    try std.testing.expectEqual(@as(usize, 6), cfg.swa_pattern);
    try std.testing.expectEqual(@as(usize, 1024), cfg.sliding_window);
    try std.testing.expectEqual(@as(f64, 1e6), cfg.rope_theta);
    try std.testing.expectEqual(@as(f64, 10000.0), cfg.rope_theta_local);
    try std.testing.expectEqual(@as(f32, 30.0), cfg.final_logit_softcap);
    // Layers 5, 11, ... global; the rest local.
    try std.testing.expect(!cfg.isGlobal(0) and cfg.isGlobal(5) and !cfg.isGlobal(6) and cfg.isGlobal(47));
    // Per-layer widths.
    try std.testing.expectEqual(@as(usize, 4096), cfg.qDim(0)); // local: 16*256
    try std.testing.expectEqual(@as(usize, 2048), cfg.kvDim(0)); // local: 8*256
    try std.testing.expectEqual(@as(usize, 8192), cfg.qDim(5)); // global: 16*512
    try std.testing.expectEqual(@as(usize, 512), cfg.kvDim(5)); // global: 1*512
    // Tied head; rope freq factors present; suppress tokens loaded.
    try std.testing.expectEqual(lm.embed.bytes.ptr, lm.head.bytes.ptr);
    try std.testing.expectEqual(@as(usize, 256), lm.rope_freqs.len); // 512/2
    try std.testing.expect(lm.suppress_tokens.len >= 1);
    try std.testing.expectEqual(@as(usize, 256), lm.layers[0].q_norm.len);
    try std.testing.expectEqual(@as(usize, 512), lm.layers[5].q_norm.len);
}
