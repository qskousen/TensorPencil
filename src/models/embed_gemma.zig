//! EmbeddingGemma (google/embeddinggemma-300m) — a Gemma-3 text *encoder* that
//! maps text to one L2-normalized 768-d embedding.
//!
//! Architecturally it is the Gemma-3 decoder body (4-norm sandwich, per-head
//! QK-norm, GeGLU/gelu-tanh MLP, dual-RoPE with a global layer every 6th) run
//! BIDIRECTIONALLY (no causal mask, `use_bidirectional_attention`), followed by
//! a sentence-transformers head: mean-pool over all tokens → Dense 768→3072 →
//! Dense 3072→768 (both no-bias, identity activation) → L2 normalize.
//!
//! Reuse: the transformer body runs through the shared `transformer.gemma3_spec`
//! `layerForward(.fresh, bidirectional=true)` — the exact tested gemma3 math.
//! Only the head (pooling / dense / normalize) and the safetensors loader are
//! new here.
//!
//! Two gemma-specific loading details vs the GGUF gemma3 path:
//!   - RMSNorm weights: HF `Gemma3RMSNorm` computes `x * (1 + w)`. The GGUF
//!     converter folds the +1 in; raw HF safetensors do NOT, so we add 1.0 to
//!     every norm vector at load (`normVec`).
//!   - Tensor names are the bare HF names (`embed_tokens.weight`,
//!     `layers.N.self_attn.q_proj.weight`, `norm.weight`) — no `model.` prefix
//!     in this sentence-transformers export — so `loader.*` matches directly.
//!
//! Sliding window: EmbeddingGemma's local layers use a 512 sliding window, but
//! for the common case seq <= 512 a window is indistinguishable from full
//! attention, so we run every layer with the full bidirectional mask (window 0)
//! and only alternate the RoPE base per layer. Long inputs (> 512 tokens) would
//! need a bidirectional sliding window; not yet implemented (see TODO in embed).

const std = @import("std");
const tp_core = @import("tp_core");
const ops = @import("tp_ops");
const loader = @import("loader.zig");
const transformer = @import("transformer.zig");
const qwen3 = @import("qwen3.zig");
const gemma3 = @import("gemma3.zig");

const SafeTensors = tp_core.safetensors.SafeTensors;
const WeightStore = tp_core.weights.WeightStore;
const Weight = ops.matmul.Weight;

/// Gemma-3 body config; EmbeddingGemma reuses gemma3.Config verbatim.
pub const Config = gemma3.Config;

/// EmbeddingGemma-300m fixed hyperparameters (config.json). `rope_scaling` is
/// null (no linear position scaling → freq_scale 1.0) and query_pre_attn_scalar
/// == head_dim (256), so the default 1/sqrt(head_dim) attention scale is exact.
pub const config_300m: Config = .{
    .n_layers = 24,
    .hidden = 768,
    .n_heads = 3,
    .n_kv_heads = 1,
    .head_dim = 256,
    .intermediate = 1152,
    .vocab = 262144,
    .rms_eps = 1e-6,
    .rope_theta = 1_000_000.0,
    .rope_freq_scale = 1.0,
    .rope_theta_local = 10_000.0,
    .sliding_window = 512,
    .swa_pattern = 6,
};

/// Output embedding dimension (post-projection).
pub const embed_dim: usize = 768;
/// Hidden width of the projection head's first dense layer.
pub const proj_hidden: usize = 3072;

/// A gemma3 layer; fields match what `transformer.gemma3_spec` reads (anytype).
const Layer = struct {
    input_norm: []const f32,
    q: Weight,
    k: Weight,
    v: Weight,
    o: Weight,
    q_norm: []const f32,
    k_norm: []const f32,
    post_attn_norm: []const f32,
    pre_ffn_norm: []const f32,
    post_ffn_norm: []const f32,
    gate: Weight,
    up: Weight,
    down: Weight,
};

/// Load a 1-D norm vector and fold Gemma's implicit +1 into it (see file doc).
fn normVec(alloc: std.mem.Allocator, store: WeightStore, name: []const u8, len: usize) ![]const f32 {
    const v = try loader.vector(alloc, store, name, len);
    for (v) |*e| e.* += 1.0;
    return v;
}

fn indexedNormVec(alloc: std.mem.Allocator, store: WeightStore, comptime prefix: []const u8, i: usize, comptime suffix: []const u8, len: usize) ![]const f32 {
    const v = try loader.indexedVector(alloc, store, prefix, i, suffix, len);
    for (v) |*e| e.* += 1.0;
    return v;
}

pub const Model = struct {
    arena: std.heap.ArenaAllocator,
    // The three safetensors stay open for the model's lifetime: every Weight is
    // a view into one of these mmaps.
    body_st: SafeTensors,
    dense1_st: SafeTensors,
    dense2_st: SafeTensors,
    cfg: Config,
    embed_w: Weight,
    layers: []Layer,
    final_norm: []const f32,
    dense1: Weight, // [proj_hidden, hidden], no bias
    dense2: Weight, // [hidden, proj_hidden], no bias

    /// Load from a sentence-transformers EmbeddingGemma directory containing
    /// `model.safetensors`, `2_Dense/model.safetensors`, `3_Dense/model.safetensors`.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Model {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();
        const cfg = config_300m;

        var pbuf: [1024]u8 = undefined;

        var body_st = try SafeTensors.open(gpa, io, try std.fmt.bufPrint(&pbuf, "{s}/model.safetensors", .{dir}));
        errdefer body_st.deinit();
        const store: WeightStore = .{ .safetensors = &body_st };

        const embed_w = try loader.matrix(store, "embed_tokens.weight", cfg.vocab, cfg.hidden);
        const final_norm = try normVec(a, store, "norm.weight", cfg.hidden);

        const layers = try a.alloc(Layer, cfg.n_layers);
        for (layers, 0..) |*layer, l| {
            layer.* = .{
                .input_norm = try indexedNormVec(a, store, "layers.", l, "input_layernorm.weight", cfg.hidden),
                .q = try loader.indexedMatrix(store, "layers.", l, "self_attn.q_proj.weight", cfg.qDim(), cfg.hidden),
                .k = try loader.indexedMatrix(store, "layers.", l, "self_attn.k_proj.weight", cfg.kvDim(), cfg.hidden),
                .v = try loader.indexedMatrix(store, "layers.", l, "self_attn.v_proj.weight", cfg.kvDim(), cfg.hidden),
                .o = try loader.indexedMatrix(store, "layers.", l, "self_attn.o_proj.weight", cfg.hidden, cfg.qDim()),
                .q_norm = try indexedNormVec(a, store, "layers.", l, "self_attn.q_norm.weight", cfg.head_dim),
                .k_norm = try indexedNormVec(a, store, "layers.", l, "self_attn.k_norm.weight", cfg.head_dim),
                .post_attn_norm = try indexedNormVec(a, store, "layers.", l, "post_attention_layernorm.weight", cfg.hidden),
                .pre_ffn_norm = try indexedNormVec(a, store, "layers.", l, "pre_feedforward_layernorm.weight", cfg.hidden),
                .post_ffn_norm = try indexedNormVec(a, store, "layers.", l, "post_feedforward_layernorm.weight", cfg.hidden),
                .gate = try loader.indexedMatrix(store, "layers.", l, "mlp.gate_proj.weight", cfg.intermediate, cfg.hidden),
                .up = try loader.indexedMatrix(store, "layers.", l, "mlp.up_proj.weight", cfg.intermediate, cfg.hidden),
                .down = try loader.indexedMatrix(store, "layers.", l, "mlp.down_proj.weight", cfg.hidden, cfg.intermediate),
            };
        }

        var dense1_st = try SafeTensors.open(gpa, io, try std.fmt.bufPrint(&pbuf, "{s}/2_Dense/model.safetensors", .{dir}));
        errdefer dense1_st.deinit();
        const dense1 = try loader.matrix(.{ .safetensors = &dense1_st }, "linear.weight", proj_hidden, cfg.hidden);

        var dense2_st = try SafeTensors.open(gpa, io, try std.fmt.bufPrint(&pbuf, "{s}/3_Dense/model.safetensors", .{dir}));
        errdefer dense2_st.deinit();
        const dense2 = try loader.matrix(.{ .safetensors = &dense2_st }, "linear.weight", cfg.hidden, proj_hidden);

        return .{
            .arena = arena,
            .body_st = body_st,
            .dense1_st = dense1_st,
            .dense2_st = dense2_st,
            .cfg = cfg,
            .embed_w = embed_w,
            .layers = layers,
            .final_norm = final_norm,
            .dense1 = dense1,
            .dense2 = dense2,
        };
    }

    pub fn deinit(self: *Model) void {
        self.dense2_st.deinit();
        self.dense1_st.deinit();
        self.body_st.deinit();
        self.arena.deinit();
        self.* = undefined;
    }

    /// Run the bidirectional Gemma-3 body over `ids` and write the final-normed
    /// hidden states (`last_hidden_state`) into `out` [seq * hidden]. `ids`
    /// must already carry the model frame (`<bos> … <eos>`).
    pub fn forwardHidden(self: *const Model, io: std.Io, gpa: std.mem.Allocator, ids: []const u32, out: []f32) !void {
        const cfg = self.cfg;
        const seq = ids.len;
        std.debug.assert(seq > 0);
        std.debug.assert(out.len == seq * cfg.hidden);
        // TODO: seq > sliding_window would need a bidirectional sliding window;
        // for now every layer uses the full (window-free) bidirectional mask.
        std.debug.assert(seq <= cfg.sliding_window);

        const x = try gpa.alloc(f32, seq * cfg.hidden);
        defer gpa.free(x);
        try qwen3.embedTokens(self.embed_w, ids, x);
        const scale = cfg.embedScale();
        for (x) |*v| v.* *= scale;

        var fg = try ops.rope.rotateHalfFreqs(gpa, seq, cfg.head_dim, cfg.rope_theta);
        defer fg.deinit(gpa);
        var fl = try ops.rope.rotateHalfFreqs(gpa, seq, cfg.head_dim, cfg.rope_theta_local);
        defer fl.deinit(gpa);

        var s = try gemma3.Scratch.init(gpa, seq, cfg);
        defer s.deinit(gpa);

        for (self.layers, 0..) |*layer, l| {
            const global = cfg.isGlobal(l);
            const dims: transformer.Dims = .{
                .hidden = cfg.hidden,
                .n_heads = cfg.n_heads,
                .n_kv = cfg.n_kv_heads,
                .head_dim = cfg.head_dim,
                .q_dim = cfg.qDim(),
                .kv_dim = cfg.kvDim(),
                .intermediate = cfg.intermediate,
                .sliding_window = 0, // full bidirectional (valid for seq <= window)
            };
            const freqs = if (global) fg else fl;
            try transformer.layerForward(transformer.gemma3_spec, .fresh, io, gpa, layer, x, seq, dims, freqs, cfg.rms_eps, {}, l, 0, true, &s);
        }

        ops.norm.rmsNorm(out, x, self.final_norm, cfg.rms_eps);
    }

    /// Encode `ids` (already framed `<bos> … <eos>`) into `out` [embed_dim], an
    /// L2-normalized 768-d embedding: bidirectional body → mean-pool over all
    /// tokens → Dense 768→3072 → Dense 3072→768 → normalize.
    pub fn embed(self: *const Model, io: std.Io, gpa: std.mem.Allocator, ids: []const u32, out: []f32) !void {
        const cfg = self.cfg;
        const seq = ids.len;
        std.debug.assert(out.len == embed_dim);

        const lhs = try gpa.alloc(f32, seq * cfg.hidden);
        defer gpa.free(lhs);
        try self.forwardHidden(io, gpa, ids, lhs);
        try self.head(io, gpa, lhs, out);
    }

    /// Pooling + projection head over the final hidden states `lhs`
    /// [seq * hidden]: mean-pool over all tokens → Dense 768→3072 → Dense
    /// 3072→768 → L2-normalize. Shared with the GPU path (body on device →
    /// download `lhs` → this on host). Cheap; always host.
    pub fn head(self: *const Model, io: std.Io, gpa: std.mem.Allocator, lhs: []const f32, out: []f32) !void {
        const cfg = self.cfg;
        const seq = lhs.len / cfg.hidden;
        std.debug.assert(out.len == embed_dim);

        const pooled = try gpa.alloc(f32, cfg.hidden);
        defer gpa.free(pooled);
        @memset(pooled, 0);
        for (0..seq) |t| {
            for (pooled, lhs[t * cfg.hidden ..][0..cfg.hidden]) |*p, r| p.* += r;
        }
        const inv: f32 = 1.0 / @as(f32, @floatFromInt(seq));
        for (pooled) |*p| p.* *= inv;

        const mid = try gpa.alloc(f32, proj_hidden);
        defer gpa.free(mid);
        try ops.matmul.matmul(io, gpa, mid, pooled, 1, self.dense1, null);
        try ops.matmul.matmul(io, gpa, out, mid, 1, self.dense2, null);

        var ss: f32 = 0;
        for (out) |v| ss += v * v;
        const norm = @sqrt(ss);
        if (norm > 0) {
            const invn = 1.0 / norm;
            for (out) |*v| v.* *= invn;
        }
    }
};

// --- tests -----------------------------------------------------------------

fn cosine(a: []const f32, b: []const f32) f32 {
    var dot: f32 = 0;
    var na: f32 = 0;
    var nb: f32 = 0;
    for (a, b) |x, y| {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    return dot / (@sqrt(na) * @sqrt(nb));
}

// Numeric parity vs the onnx-community EmbeddingGemma ONNX (reference vectors in
// testdata/, generated by scratch_out/gen_embeddinggemma_ref.py). Requires the
// unsloth safetensors checkout under ../DiffKeep; skipped when absent.
test "embeddinggemma matches ONNX reference" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dir = "../DiffKeep/Models/embeddinggemma-300m";
    std.Io.Dir.cwd().access(io, dir, .{}) catch return error.SkipZigTest;

    const ref_bytes = std.Io.Dir.cwd().readFileAlloc(io, "testdata/embeddinggemma_ref_vectors.json", gpa, .limited(4 * 1024 * 1024)) catch return error.SkipZigTest;
    defer gpa.free(ref_bytes);
    var ref = try std.json.parseFromSlice(std.json.Value, gpa, ref_bytes, .{});
    defer ref.deinit();

    var model = try Model.open(gpa, io, dir);
    defer model.deinit();

    const out = try gpa.alloc(f32, embed_dim);
    defer gpa.free(out);
    const ids_buf = try gpa.alloc(u32, 4096);
    defer gpa.free(ids_buf);

    var it = ref.value.object.iterator();
    var worst: f32 = 1.0;
    while (it.next()) |entry| {
        const case = entry.value_ptr.*.object;
        const ids_json = case.get("ids").?.array.items;
        const want = case.get("vec").?.array.items;
        const ids = ids_buf[0..ids_json.len];
        for (ids_json, ids) |v, *d| d.* = @intCast(v.integer);

        try model.embed(io, gpa, ids, out);

        // Compare against the reference vector via cosine similarity.
        var wbuf: [embed_dim]f32 = undefined;
        for (want, 0..) |v, i| wbuf[i] = @floatCast(v.float);
        const cos = cosine(out, &wbuf);
        worst = @min(worst, cos);
        errdefer std.debug.print("case {s}: cosine {d}\n", .{ entry.key_ptr.*, cos });
        try std.testing.expect(cos >= 0.999);
    }
    errdefer std.debug.print("worst cosine {d}\n", .{worst});
    try std.testing.expect(worst >= 0.999);
}
