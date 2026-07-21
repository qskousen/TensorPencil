//! Qwen3-VL-4B language model, text-only path — the Krea 2 text encoder.
//!
//! Reference: comfy/text_encoders/llama.py (`Llama2_`, Qwen3VL_4BConfig) and
//! comfy/text_encoders/krea2.py. Config: 36 layers, hidden 2560, GQA 32/8
//! heads (head_dim 128), SwiGLU 9728, RMSNorm eps 1e-6 (plain weight),
//! per-head QK-norm before RoPE, rotate-half RoPE theta 5e6, causal
//! attention. Text-only inputs never trigger the interleaved-mRoPE branch.
//!
//! Krea 2 conditions on the hidden states *entering* layers [2,5,...,35]
//! (hidden_states[k] = output of layer k-1), so layer 35 and the final norm
//! are never evaluated and are not loaded.
//!
//! `CausalLM` is the same stack used as a language model (tp-llm): all 36
//! layers plus the final norm, with the LM head tied to the bf16 embedding
//! matrix (the checkpoint ships no lm_head.weight, per Qwen3-4B tying).
//!
//! Weights stay in checkpoint dtype (fp8-e4m3 + per-tensor f32 scales) and are
//! dequantized inside the GEMM; the safetensors mapping must outlive this.

const std = @import("std");
const safetensors = @import("tp_core").safetensors;
const test_gate = @import("../test_gate.zig");
const gguf_mod = @import("tp_core").gguf;
const weights_mod = @import("tp_core").weights;
const dtypes = @import("tp_core").dtype;
const ops = @import("tp_ops");
const transformer = @import("transformer.zig");
const kv_cache_mod = @import("tp_core").kv_cache;

const SafeTensors = safetensors.SafeTensors;
const WeightStore = weights_mod.WeightStore;
const Weight = ops.matmul.Weight;
const KvCache = kv_cache_mod.KvCache;

pub const hidden = 2560;
pub const n_heads = 32;
pub const n_kv_heads = 8;
pub const head_dim = 128;
pub const intermediate = 9728;
pub const n_layers = 36;
pub const vocab_size = 151936;
pub const rms_eps: f32 = 1e-6;
pub const rope_theta: f64 = 5000000.0;

/// Per-layer dims for the shared transformer body. qwen3 has uniform geometry
/// and a compile-time `head_dim`, so these are constant across layers. The
/// encoder uses module-const dims; `CausalLM` derives them from its runtime
/// Config via `dimsFor` (n_layers/hidden/heads vary across Qwen3 checkpoints).
const encoderDims: transformer.Dims = .{
    .hidden = hidden,
    .n_heads = n_heads,
    .n_kv = n_kv_heads,
    .head_dim = head_dim,
    .q_dim = n_heads * head_dim,
    .kv_dim = n_kv_heads * head_dim,
    .intermediate = intermediate,
};
pub fn dimsFor(cfg: Config) transformer.Dims {
    return .{
        .hidden = cfg.hidden,
        .n_heads = cfg.n_heads,
        .n_kv = cfg.n_kv_heads,
        .head_dim = head_dim,
        .q_dim = cfg.qDim(),
        .kv_dim = cfg.kvDim(),
        .intermediate = cfg.intermediate,
    };
}

/// Runtime model configuration for `CausalLM` — the extension point for
/// other Qwen3-family checkpoints (LLM_PLAN.md M5 uses Qwen3-0.6B as the
/// speculative-decoding draft model). All checkpoints share head_dim 128
/// (the flash-decoding kernels require it), rms_eps 1e-6, a tied bf16 LM
/// head, and the 151936-token vocab; everything else varies.
pub const Config = struct {
    n_layers: usize,
    hidden: usize,
    n_heads: usize,
    n_kv_heads: usize,
    intermediate: usize,
    rope_theta: f64,
    /// Tensor-name prefix up to "layers.N." / "embed_tokens." / "norm.".
    prefix: []const u8,

    pub fn qDim(self: Config) usize {
        return self.n_heads * head_dim;
    }
    pub fn kvDim(self: Config) usize {
        return self.n_kv_heads * head_dim;
    }

    /// The Krea 2 text-encoder checkpoint: Qwen3-VL-4B.
    pub const vl_4b: Config = .{
        .n_layers = 36,
        .hidden = 2560,
        .n_heads = 32,
        .n_kv_heads = 8,
        .intermediate = 9728,
        .rope_theta = 5000000.0,
        .prefix = "model.language_model.",
    };

    /// Qwen3-0.6B (base or instruct) — the draft model.
    pub const qwen3_0_6b: Config = .{
        .n_layers = 28,
        .hidden = 1024,
        .n_heads = 16,
        .n_kv_heads = 8,
        .intermediate = 3072,
        .rope_theta = 1000000.0,
        .prefix = "model.",
    };

    /// Vanilla Qwen3-4B (text-only): the VL's dims but rope_theta 1e6 and
    /// bare tensor names — the EAGLE-3 head's actual training target.
    pub const qwen3_4b: Config = .{
        .n_layers = 36,
        .hidden = 2560,
        .n_heads = 32,
        .n_kv_heads = 8,
        .intermediate = 9728,
        .rope_theta = 1000000.0,
        .prefix = "model.",
    };

    /// Pick the configuration for a checkpoint. GGUFs with llama.cpp
    /// metadata build it from the `qwen3.*` keys; everything else matches a
    /// preset by its embedding tensor name and shape (rope_theta is not
    /// recoverable from weights, so only known configurations load).
    pub fn detect(store: WeightStore) !Config {
        if (store == .gguf) return detectGguf(store.gguf);
        inline for (.{ vl_4b, qwen3_0_6b, qwen3_4b }) |cfg| {
            var buf: [96]u8 = undefined;
            const name = try std.fmt.bufPrint(&buf, "{s}embed_tokens.weight", .{cfg.prefix});
            if (store.get(name)) |view| {
                const shape = view.info.shape.slice();
                if (shape.len == 2 and shape[0] == vocab_size and shape[1] == cfg.hidden) return cfg;
            }
        }
        return error.UnknownModelConfig;
    }

    fn detectGguf(g: *const gguf_mod.Gguf) !Config {
        if (g.getStr("general.architecture")) |arch| {
            if (!std.mem.eql(u8, arch, "qwen3")) return error.UnknownModelConfig;
        }
        // Full llama.cpp metadata: build the config from the qwen3.* keys.
        if (g.getUint("qwen3.block_count")) |block_count| {
            if (block_count == 0 or block_count > max_layers) return error.UnknownModelConfig;
            // head_dim is a kernel invariant, not configurable.
            if ((g.getUint("qwen3.attention.key_length") orelse head_dim) != head_dim or
                (g.getUint("qwen3.attention.value_length") orelse head_dim) != head_dim)
                return error.UnknownModelConfig;
            return .{
                .n_layers = @intCast(block_count),
                .hidden = @intCast(g.getUint("qwen3.embedding_length") orelse return error.UnknownModelConfig),
                .n_heads = @intCast(g.getUint("qwen3.attention.head_count") orelse return error.UnknownModelConfig),
                .n_kv_heads = @intCast(g.getUint("qwen3.attention.head_count_kv") orelse return error.UnknownModelConfig),
                .intermediate = @intCast(g.getUint("qwen3.feed_forward_length") orelse return error.UnknownModelConfig),
                .rope_theta = g.getFloat("qwen3.rope.freq_base") orelse 1e6,
                .prefix = "",
            };
        }
        // Hyperparameter-less GGUF (ComfyUI-style conversion, bare HF names,
        // at most an architecture tag): match a plain-Qwen3 preset by
        // embedding shape. rope_theta is unrecoverable, so a VL-derived
        // conversion would silently get the plain-Qwen3 theta — hence the
        // warning.
        const view = g.get("embed_tokens.weight") orelse return error.UnknownModelConfig;
        const shape = view.info.shape.slice();
        if (shape.len != 2 or shape[0] != vocab_size) return error.UnknownModelConfig;
        inline for (.{ qwen3_0_6b, qwen3_4b }) |preset| {
            if (shape[1] == preset.hidden) {
                var cfg = preset;
                cfg.prefix = "";
                std.log.warn(
                    "gguf has no hyperparameter metadata; assuming plain Qwen3 (rope_theta {d})",
                    .{cfg.rope_theta},
                );
                return cfg;
            }
        }
        return error.UnknownModelConfig;
    }

    /// Upper bound on n_layers — backend steppers use fixed-size per-layer
    /// arrays. Covers the presets (36) and GGUF checkpoints up to Qwen3-32B
    /// (64 layers).
    pub const max_layers = 64;
};

/// Tap k is the hidden state before layer k runs.
pub const tap_layers = [_]usize{ 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35 };
pub const tap_count = tap_layers.len;

pub const q_dim = n_heads * head_dim; // 4096
pub const kv_dim = n_kv_heads * head_dim; // 1024

const Layer = struct {
    input_norm: []const f32,
    q: Weight,
    k: Weight,
    v: Weight,
    o: Weight,
    q_norm: []const f32, // [head_dim]
    k_norm: []const f32,
    post_norm: []const f32,
    gate: Weight,
    up: Weight,
    down: Weight,
};

pub const TextEncoder = struct {
    arena: std.heap.ArenaAllocator,
    /// bf16 [vocab, hidden] view into the mapped file.
    embed_bytes: []const u8,
    /// Layers 0..34 — the last tap fires before layer 35.
    layers: []Layer,

    pub fn load(gpa: std.mem.Allocator, st: *const SafeTensors) !TextEncoder {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();

        const embed = try st.require("model.language_model.embed_tokens.weight");
        if (embed.info.dtype != .bf16 or embed.info.elemCount() != vocab_size * hidden)
            return error.ShapeMismatch;

        const layers = try loadLayers(alloc, st, n_layers - 1);

        return .{ .arena = arena, .embed_bytes = embed.bytes, .layers = layers };
    }

    pub fn deinit(self: *TextEncoder) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// Encode token ids to the Krea 2 conditioning stack, [seq][tap_count][hidden]
    /// row-major (token-major, matching the DiT's unpacked context layout).
    pub fn encode(self: *const TextEncoder, io: std.Io, gpa: std.mem.Allocator, ids: []const u32) ![]f32 {
        const seq = ids.len;
        std.debug.assert(seq > 0);

        const out = try gpa.alloc(f32, seq * tap_count * hidden);
        errdefer gpa.free(out);

        const x = try gpa.alloc(f32, seq * hidden);
        defer gpa.free(x);
        try embedTokens(Weight.init(self.embed_bytes, .bf16, vocab_size, hidden), ids, x);

        var freqs = try ops.rope.rotateHalfFreqs(gpa, seq, head_dim, rope_theta);
        defer freqs.deinit(gpa);

        var scratch = try Scratch.init(gpa, seq, Config.vl_4b);
        defer scratch.deinit(gpa);

        const dims = encoderDims;
        var tap_idx: usize = 0;
        for (0..n_layers) |l| {
            if (tap_idx < tap_layers.len and tap_layers[tap_idx] == l) {
                for (0..seq) |t| {
                    @memcpy(out[(t * tap_count + tap_idx) * hidden ..][0..hidden], x[t * hidden ..][0..hidden]);
                }
                tap_idx += 1;
            }
            if (l >= self.layers.len) break;
            // Encoder: full-sequence, no persistent KV cache.
            try transformer.layerForward(transformer.qwen3_spec, .fresh, io, gpa, self.layers[l], x, seq, dims, freqs, rms_eps, {}, 0, 0, false, &scratch);
        }
        std.debug.assert(tap_idx == tap_count);
        return out;
    }
};

/// A Qwen3-family text stack as a language model: all layers, final norm,
/// and the embedding matrix doubling as the tied LM head. The checkpoint's
/// configuration is auto-detected (Config.detect).
pub const CausalLM = struct {
    arena: std.heap.ArenaAllocator,
    cfg: Config,
    /// [vocab, hidden] embedding table view into the mapped file, in the
    /// checkpoint's storage dtype (bf16 for safetensors; GGUFs quantize it).
    embed: Weight,
    /// LM head ([vocab, hidden]): `embed` when tied (Qwen3-4B ships no
    /// lm_head), a separate tensor when the checkpoint carries one (GGUF
    /// "output.weight").
    head: Weight,
    layers: []Layer,
    final_norm: []const f32,

    pub fn load(gpa: std.mem.Allocator, store: WeightStore) !CausalLM {
        const cfg = try Config.detect(store);
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();

        var buf: [96]u8 = undefined;
        const embed_view = try store.require(try std.fmt.bufPrint(&buf, "{s}embed_tokens.weight", .{cfg.prefix}));
        const eshape = embed_view.info.shape.slice();
        if (eshape.len != 2 or eshape[0] != vocab_size or eshape[1] != cfg.hidden)
            return error.ShapeMismatch;
        const embed = Weight.init(embed_view.bytes, embed_view.info.dtype, vocab_size, cfg.hidden);

        var head = embed;
        if (store.get(try std.fmt.bufPrint(&buf, "{s}lm_head.weight", .{cfg.prefix}))) |hv| {
            const hshape = hv.info.shape.slice();
            if (hshape.len != 2 or hshape[0] != vocab_size or hshape[1] != cfg.hidden)
                return error.ShapeMismatch;
            head = Weight.init(hv.bytes, hv.info.dtype, vocab_size, cfg.hidden);
        }

        const layers = try loadLayersCfg(alloc, store, cfg, cfg.n_layers);
        const final_norm = try loadNormNamed(alloc, store, try std.fmt.bufPrint(&buf, "{s}norm.weight", .{cfg.prefix}), cfg.hidden);

        return .{ .arena = arena, .cfg = cfg, .embed = embed, .head = head, .layers = layers, .final_norm = final_norm };
    }

    pub fn deinit(self: *CausalLM) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// LM head: logits = hidden @ head^T, [vocab] per position.
    pub fn lmHead(self: *const CausalLM) Weight {
        return self.head;
    }

    /// True when any weight is a ggml block-quantized dtype — such models
    /// only run on the cpu backend until the GPU kernels land.
    pub fn hasBlockQuantWeights(self: *const CausalLM) bool {
        if (self.embed.dtype.isBlockQuant() or self.head.dtype.isBlockQuant()) return true;
        for (self.layers) |l| {
            inline for (.{ l.q, l.k, l.v, l.o, l.gate, l.up, l.down }) |w| {
                if (w.dtype.isBlockQuant()) return true;
            }
        }
        return false;
    }

    /// Forward `ids` at absolute positions [cache.len, cache.len + ids.len),
    /// appending their K/V to the cache: prefill when the cache is empty,
    /// single-token decode when ids.len == 1. `freqs` must cover the final
    /// position. When `out` ([n * hidden], n <= ids.len) is set, it receives
    /// the final-normed hidden states of the last n new positions, ready for
    /// the LM head (n = 1 for decode, n = draft+1 for speculative verify).
    pub fn forwardCached(
        self: *const CausalLM,
        io: std.Io,
        gpa: std.mem.Allocator,
        ids: []const u32,
        cache: *KvCache,
        freqs: ops.rope.Freqs,
        out: ?[]f32,
    ) !void {
        const cfg = self.cfg;
        const seq = ids.len;
        std.debug.assert(seq > 0 and seq <= cache.remaining());
        std.debug.assert(cache.n_layers == cfg.n_layers and cache.kv_dim == cfg.kvDim());

        const x = try gpa.alloc(f32, seq * cfg.hidden);
        defer gpa.free(x);
        try embedTokens(self.embed, ids, x);

        var scratch = try Scratch.init(gpa, seq, cfg);
        defer scratch.deinit(gpa);

        const dims = dimsFor(cfg);
        const pos0 = cache.len;
        for (self.layers, 0..) |layer, l| {
            try transformer.layerForward(transformer.qwen3_spec, .cached, io, gpa, layer, x, seq, dims, freqs, rms_eps, cache, l, pos0, false, &scratch);
        }
        cache.commit(seq);
        if (out) |o| {
            std.debug.assert(o.len % cfg.hidden == 0);
            const n = o.len / cfg.hidden;
            std.debug.assert(n >= 1 and n <= seq);
            ops.norm.rmsNorm(o, x[(seq - n) * cfg.hidden ..][0 .. n * cfg.hidden], self.final_norm, rms_eps);
        }
    }

    /// Tree-verify forward (speculative tree drafting, LLM_PLAN.md M8):
    /// `ids.len` tree nodes forwarded against the committed cache WITHOUT
    /// appending to it. Node i (parents[i] < i; node 0 is the root) sits at
    /// absolute position cache.len + depth(i) and attends the committed
    /// prefix plus its own ancestor chain; per-layer K/V rows are retained
    /// in tree_k/tree_v ([n_layers][ids.len][kv_dim] row-major) so the
    /// caller can commit the accepted path afterwards. `out` receives the
    /// final-normed hidden states of ALL nodes ([ids.len][hidden]).
    pub fn forwardTree(
        self: *const CausalLM,
        io: std.Io,
        gpa: std.mem.Allocator,
        ids: []const u32,
        parents: []const u32,
        cache: *KvCache, // non-const: kView expands into the cache's f16 scratch
        freqs: ops.rope.Freqs,
        tree_k: []f32,
        tree_v: []f32,
        out: []f32,
    ) !void {
        const cfg = self.cfg;
        const n = ids.len;
        std.debug.assert(n > 0 and parents.len == n);
        std.debug.assert(tree_k.len >= cfg.n_layers * n * cfg.kvDim() and tree_v.len >= tree_k.len);
        std.debug.assert(out.len == n * cfg.hidden);
        std.debug.assert(cache.n_layers == cfg.n_layers and cache.kv_dim == cfg.kvDim());

        const positions = try gpa.alloc(usize, n);
        defer gpa.free(positions);
        positions[0] = cache.len;
        for (parents[1..], 1..) |p, i| {
            std.debug.assert(p < i);
            positions[i] = positions[p] + 1;
            std.debug.assert(positions[i] < cache.capacity);
        }

        const x = try gpa.alloc(f32, n * cfg.hidden);
        defer gpa.free(x);
        try embedTokens(self.embed, ids, x);

        var s = try Scratch.init(gpa, n, cfg);
        defer s.deinit(gpa);

        const dims = dimsFor(cfg);
        for (self.layers, 0..) |layer, l| {
            try transformer.layerForwardTree(transformer.qwen3_spec, io, gpa, layer, x, n, dims, freqs, positions, parents, cache, l, tree_k, tree_v, rms_eps, &s);
        }
        ops.norm.rmsNorm(out, x, self.final_norm, rms_eps);
    }
};

/// Look up embedding rows for `ids` into `x` [ids.len, h] f32, dequantizing
/// from the table's storage dtype (bf16, or a GGUF block-quantized format).
/// Shared with the GPU steppers' host-side prefill gathers.
pub fn embedTokens(embed: Weight, ids: []const u32, x: []f32) !void {
    const h = embed.cols;
    const row_bytes = embed.dtype.storageBytes(h);
    for (ids, 0..) |id, t| {
        if (id >= embed.rows) return error.TokenIdOutOfRange;
        const row = embed.bytes[@as(usize, id) * row_bytes ..][0..row_bytes];
        try safetensors.convertToF32(embed.dtype, row, x[t * h ..][0..h]);
    }
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
    /// The CUDA split sizes its scratch to a full chunk once, then views it
    /// down to the actual chunk length each call — `layerForward`'s ops
    /// require exact-length slices, so passing the oversized buffer would trip
    /// a length assert (same fix as gemma3.Scratch.viewSeq). Never deinit a
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
        s.normed = try gpa.alloc(f32, seq * cfg.hidden);
        errdefer gpa.free(s.normed);
        s.tmp = try gpa.alloc(f32, seq * cfg.hidden);
        errdefer gpa.free(s.tmp);
        s.q = try gpa.alloc(f32, seq * cfg.qDim());
        errdefer gpa.free(s.q);
        s.k = try gpa.alloc(f32, seq * cfg.kvDim());
        errdefer gpa.free(s.k);
        s.v = try gpa.alloc(f32, seq * cfg.kvDim());
        errdefer gpa.free(s.v);
        s.attn_out = try gpa.alloc(f32, seq * cfg.qDim());
        errdefer gpa.free(s.attn_out);
        s.gate = try gpa.alloc(f32, seq * cfg.intermediate);
        errdefer gpa.free(s.gate);
        s.up = try gpa.alloc(f32, seq * cfg.intermediate);
        errdefer gpa.free(s.up);
        return s;
    }

    pub fn deinit(self: *Scratch, gpa: std.mem.Allocator) void {
        gpa.free(self.normed);
        gpa.free(self.tmp);
        gpa.free(self.q);
        gpa.free(self.k);
        gpa.free(self.v);
        gpa.free(self.attn_out);
        gpa.free(self.gate);
        gpa.free(self.up);
        self.* = undefined;
    }
};

fn loadLayers(alloc: std.mem.Allocator, st: *const SafeTensors, count: usize) ![]Layer {
    return loadLayersCfg(alloc, .{ .safetensors = st }, Config.vl_4b, count);
}

fn loadLayersCfg(alloc: std.mem.Allocator, store: WeightStore, cfg: Config, count: usize) ![]Layer {
    const layers = try alloc.alloc(Layer, count);
    for (layers, 0..) |*layer, i| {
        layer.* = .{
            .input_norm = try loadNorm(alloc, store, cfg, i, "input_layernorm.weight", cfg.hidden),
            .q = try loadWeight(store, cfg, i, "self_attn.q_proj.weight", cfg.qDim(), cfg.hidden),
            .k = try loadWeight(store, cfg, i, "self_attn.k_proj.weight", cfg.kvDim(), cfg.hidden),
            .v = try loadWeight(store, cfg, i, "self_attn.v_proj.weight", cfg.kvDim(), cfg.hidden),
            .o = try loadWeight(store, cfg, i, "self_attn.o_proj.weight", cfg.hidden, cfg.qDim()),
            .q_norm = try loadNorm(alloc, store, cfg, i, "self_attn.q_norm.weight", head_dim),
            .k_norm = try loadNorm(alloc, store, cfg, i, "self_attn.k_norm.weight", head_dim),
            .post_norm = try loadNorm(alloc, store, cfg, i, "post_attention_layernorm.weight", cfg.hidden),
            .gate = try loadWeight(store, cfg, i, "mlp.gate_proj.weight", cfg.intermediate, cfg.hidden),
            .up = try loadWeight(store, cfg, i, "mlp.up_proj.weight", cfg.intermediate, cfg.hidden),
            .down = try loadWeight(store, cfg, i, "mlp.down_proj.weight", cfg.hidden, cfg.intermediate),
        };
    }
    return layers;
}

fn loadNorm(alloc: std.mem.Allocator, store: WeightStore, cfg: Config, layer: usize, comptime suffix: []const u8, len: usize) ![]f32 {
    var buf: [96]u8 = undefined;
    const name = try std.fmt.bufPrint(&buf, "{s}layers.{d}." ++ suffix, .{ cfg.prefix, layer });
    return loadNormNamed(alloc, store, name, len);
}

fn loadNormNamed(alloc: std.mem.Allocator, store: WeightStore, name: []const u8, len: usize) ![]f32 {
    const view = store.get(name) orelse return error.MissingTensor;
    if (view.info.elemCount() != len) return error.ShapeMismatch;
    return view.toF32Alloc(alloc);
}

fn loadWeight(store: WeightStore, cfg: Config, layer: usize, comptime suffix: []const u8, rows: usize, cols: usize) !Weight {
    var buf: [96]u8 = undefined;
    const name = try std.fmt.bufPrint(&buf, "{s}layers.{d}." ++ suffix, .{ cfg.prefix, layer });
    const view = store.get(name) orelse return error.MissingTensor;
    const shape = view.info.shape.slice();
    if (shape.len != 2 or shape[0] != rows or shape[1] != cols) return error.ShapeMismatch;
    var w = Weight.init(view.bytes, view.info.dtype, rows, cols);
    var scale_buf: [112]u8 = undefined;
    const scale_name = try std.fmt.bufPrint(&scale_buf, "{s}_scale", .{name});
    if (store.get(scale_name)) |scale_view| {
        w.scale = try scale_view.asScalarF32();
    }
    return w;
}

// --- tests -----------------------------------------------------------------

fn readF32File(gpa: std.mem.Allocator, io: std.Io, path: []const u8, n: usize) ![]f32 {
    const out = try gpa.alloc(f32, n);
    errdefer gpa.free(out);
    const file = try std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_only });
    defer file.close(io);
    const bytes = std.mem.sliceAsBytes(out);
    if (try file.readPositionalAll(io, bytes, 0) != bytes.len) return error.ShortRead;
    return out;
}

// Config detection + weight wiring against a real llama.cpp GGUF; skipped
// when the checkpoint is absent. Load-only — generation quality is validated
// end-to-end via tp-llm (a Debug 4B forward is too slow for the suite).
test "causal lm loads from real qwen3-4b gguf" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const path = "models/text_encoders/Qwen3-4B-Q4_K_M.gguf";
    std.Io.Dir.cwd().access(io, path, .{}) catch return error.SkipZigTest;

    var g = try gguf_mod.Gguf.open(gpa, io, path);
    defer g.deinit();
    var lm = try CausalLM.load(gpa, .{ .gguf = &g });
    defer lm.deinit();

    try std.testing.expectEqual(@as(usize, 36), lm.cfg.n_layers);
    try std.testing.expectEqual(@as(usize, 2560), lm.cfg.hidden);
    try std.testing.expectEqual(@as(usize, 32), lm.cfg.n_heads);
    try std.testing.expectEqual(@as(usize, 8), lm.cfg.n_kv_heads);
    try std.testing.expectEqual(@as(usize, 9728), lm.cfg.intermediate);
    try std.testing.expectEqual(@as(f64, 1e6), lm.cfg.rope_theta);
    try std.testing.expectEqualStrings("", lm.cfg.prefix);

    try std.testing.expectEqual(dtypes.DType.q6_k, lm.embed.dtype);
    // No output.weight in this file: the head ties to the embedding.
    try std.testing.expectEqual(lm.embed.bytes.ptr, lm.head.bytes.ptr);
    try std.testing.expect(lm.hasBlockQuantWeights());
    try std.testing.expectEqual(dtypes.DType.q4_k, lm.layers[0].q.dtype);
    try std.testing.expectEqual(@as(usize, 2560), lm.layers[0].input_norm.len);
}

// Config detection from synthetic GGUF metadata: a Qwen3-32B-shaped header
// (64 layers — larger than any preset) builds the right Config, and a
// block_count above max_layers is rejected.
test "config detects 32b gguf metadata" {
    const gpa = std.testing.allocator;

    var b = try gguf_mod.TestBuilder.init(gpa, 3, 0, 9);
    defer b.deinit();
    try b.kvStr("general.architecture", "qwen3");
    try b.kvUint("qwen3.block_count", 64);
    try b.kvUint("qwen3.embedding_length", 5120);
    try b.kvUint("qwen3.attention.head_count", 64);
    try b.kvUint("qwen3.attention.head_count_kv", 8);
    try b.kvUint("qwen3.attention.key_length", 128);
    try b.kvUint("qwen3.attention.value_length", 128);
    try b.kvUint("qwen3.feed_forward_length", 25600);
    try b.kvF32("qwen3.rope.freq_base", 1e6);
    const file = try b.finish(&.{});
    defer gpa.free(file);

    var g = try gguf_mod.Gguf.initFromSlice(gpa, file);
    defer g.deinit();

    const cfg = try Config.detect(.{ .gguf = &g });
    try std.testing.expectEqual(@as(usize, 64), cfg.n_layers);
    try std.testing.expectEqual(@as(usize, 5120), cfg.hidden);
    try std.testing.expectEqual(@as(usize, 64), cfg.n_heads);
    try std.testing.expectEqual(@as(usize, 8), cfg.n_kv_heads);
    try std.testing.expectEqual(@as(usize, 25600), cfg.intermediate);
    try std.testing.expectEqual(@as(f64, 1e6), cfg.rope_theta);
    try std.testing.expectEqualStrings("", cfg.prefix);
}

test "config rejects block_count above max_layers" {
    const gpa = std.testing.allocator;

    var b = try gguf_mod.TestBuilder.init(gpa, 3, 0, 2);
    defer b.deinit();
    try b.kvStr("general.architecture", "qwen3");
    try b.kvUint("qwen3.block_count", Config.max_layers + 1);
    const file = try b.finish(&.{});
    defer gpa.free(file);

    var g = try gguf_mod.Gguf.initFromSlice(gpa, file);
    defer g.deinit();

    try std.testing.expectError(error.UnknownModelConfig, Config.detect(.{ .gguf = &g }));
}

// Parity against ComfyUI's Krea 2 conditioning (f32), post prefix-strip.
// Fixture from tools/dump_text_fixture.py (prompt "a fluffy orange cat
// sitting on a windowsill"); skipped when model or fixture is absent.
test "krea2 conditioning matches comfyui" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const krea2_text = @import("krea2_text.zig");
    const tokenizer_mod = @import("tp_core").tokenizer;
    const te_path = "models/text_encoders/qwen3VLInstruct4bHeretic_v10.safetensors";
    try test_gate.requireModelFile(io, te_path);
    std.Io.Dir.cwd().access(io, "testdata/text_cond.bin", .{}) catch return error.SkipZigTest;

    var tok = try tokenizer_mod.Tokenizer.init(gpa);
    defer tok.deinit();
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    try krea2_text.buildIds(&tok, gpa, "a fluffy orange cat sitting on a windowsill", &ids);

    // Cross-check tokenization against the ids the fixture was built with.
    {
        const ref_ids_file = try std.Io.Dir.cwd().openFile(io, "testdata/text_ids.bin", .{ .mode = .read_only });
        defer ref_ids_file.close(io);
        const n: usize = @intCast((try ref_ids_file.length(io)) / 4);
        const ref_ids = try gpa.alloc(u32, n);
        defer gpa.free(ref_ids);
        if (try ref_ids_file.readPositionalAll(io, std.mem.sliceAsBytes(ref_ids), 0) != n * 4) return error.ShortRead;
        try std.testing.expectEqualSlices(u32, ref_ids, ids.items);
    }

    var st = try SafeTensors.open(gpa, io, te_path);
    defer st.deinit();
    var enc = try TextEncoder.load(gpa, &st);
    defer enc.deinit();

    const cond = try enc.encode(io, gpa, ids.items);
    defer gpa.free(cond);

    const offset = krea2_text.stripOffset(ids.items);
    const kept = ids.items.len - offset;
    const expected = try readF32File(gpa, io, "testdata/text_cond.bin", kept * tap_count * hidden);
    defer gpa.free(expected);

    var max_err: f32 = 0;
    var max_val: f32 = 0;
    var sum_err: f64 = 0;
    const stripped = cond[offset * tap_count * hidden ..];
    for (expected, stripped) |e, a| {
        max_err = @max(max_err, @abs(e - a));
        max_val = @max(max_val, @abs(e));
        sum_err += @abs(e - a);
    }
    const mean_err = sum_err / @as(f64, @floatFromInt(expected.len));
    std.debug.print("text parity: max_err={d:.5} mean_err={d:.6} max_val={d:.1}\n", .{ max_err, mean_err, max_val });
    // Hidden states reach magnitudes of O(100); tolerances are relative to that.
    try std.testing.expect(max_err < 0.05);
    try std.testing.expect(mean_err < 5e-4 * @as(f64, max_val));
}
