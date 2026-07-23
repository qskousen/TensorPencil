//! Snowflake Arctic Embed M v2.0 — a GTE-multilingual-base text encoder that
//! maps text to one L2-normalized 768-d embedding (the `.semantic` text space
//! for DiffKeep, candidate A1).
//!
//! Architecture (HF `GteModel` / alibaba-nlp GTE, from the shipped
//! modeling_hf_alibaba_nlp_gte.py):
//!   - Embeddings: word_embeddings[id] + token_type_embeddings[0] (single type)
//!     → LayerNorm (eps 1e-12). No learned position embeddings — RoPE instead.
//!   - 12 **post-LayerNorm** blocks (BERT-style, NOT pre-norm): packed
//!     qkv_proj (+bias) → split q/k/v → NEOX rotate-half RoPE (θ=160000) on q/k
//!     → non-causal (bidirectional) attention (scale 1/√head_dim) → o_proj
//!     (+bias) → residual add → `attn_ln` (LayerNorm) → GeGLU MLP → residual add
//!     → `mlp_ln`.
//!   - GeGLU: up_gate_proj (no bias) → split up=first `intermediate`, gate=second
//!     → **exact-erf** gelu(gate) * up → down_proj (+bias).
//!   - Pooling: CLS (token 0) of the last hidden state → L2 normalize. No dense
//!     pooler (sentence-transformers modules: Transformer → Pooling(CLS) →
//!     Normalize).
//!
//! Tokenization: `Tokenizer.initUnigramFromTokenizerJson` + the `<s> … </s>`
//! frame; CLS pooling reads position 0 (`<s>`). Query texts get a `"query: "`
//! prefix (documents raw) — applied by the caller/façade, not here.

const std = @import("std");
const tp_core = @import("tp_core");
const ops = @import("tp_ops");
const loader = @import("loader.zig");
const qwen3 = @import("qwen3.zig");

const SafeTensors = tp_core.safetensors.SafeTensors;
const WeightStore = tp_core.weights.WeightStore;
const Weight = ops.matmul.Weight;

pub const Config = struct {
    n_layers: usize,
    hidden: usize,
    n_heads: usize,
    head_dim: usize,
    intermediate: usize,
    vocab: usize,
    ln_eps: f32,
    rope_theta: f64,

    pub fn qDim(self: Config) usize {
        return self.n_heads * self.head_dim;
    }
};

/// Snowflake Arctic Embed M v2.0 (config.json).
pub const config_m_v2: Config = .{
    .n_layers = 12,
    .hidden = 768,
    .n_heads = 12,
    .head_dim = 64,
    .intermediate = 3072,
    .vocab = 250048,
    .ln_eps = 1e-12,
    .rope_theta = 160000.0,
};

pub const embed_dim: usize = 768;

/// In-place L2 normalization (shared by `embed` / `embedBatch`).
fn l2normalize(v: []f32) void {
    var ss: f32 = 0;
    for (v) |x| ss += x * x;
    const norm = @sqrt(ss);
    if (norm > 0) {
        const inv = 1.0 / norm;
        for (v) |*x| x.* *= inv;
    }
}

const Layer = struct {
    qkv: Weight, // [3*hidden, hidden]
    qkv_bias: []const f32, // [3*hidden]
    o: Weight, // [hidden, hidden]
    o_bias: []const f32,
    attn_ln_w: []const f32,
    attn_ln_b: []const f32,
    up_gate: Weight, // [2*intermediate, hidden], no bias
    down: Weight, // [hidden, intermediate]
    down_bias: []const f32,
    mlp_ln_w: []const f32,
    mlp_ln_b: []const f32,
};

pub const Model = struct {
    arena: std.heap.ArenaAllocator,
    st: SafeTensors,
    cfg: Config,
    word_emb: Weight, // [vocab, hidden]
    token_type0: []const f32, // [hidden]
    emb_ln_w: []const f32,
    emb_ln_b: []const f32,
    layers: []Layer,

    /// Load from a directory containing `model.safetensors` (the HF GTE names,
    /// no `model.` prefix).
    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Model {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();
        const cfg = config_m_v2;

        var pbuf: [1024]u8 = undefined;
        var st = try SafeTensors.open(gpa, io, try std.fmt.bufPrint(&pbuf, "{s}/model.safetensors", .{dir}));
        errdefer st.deinit();
        const store: WeightStore = .{ .safetensors = &st };

        const word_emb = try loader.matrix(store, "embeddings.word_embeddings.weight", cfg.vocab, cfg.hidden);
        const token_type0 = try loader.vector(a, store, "embeddings.token_type_embeddings.weight", cfg.hidden);
        const emb_ln_w = try loader.vector(a, store, "embeddings.LayerNorm.weight", cfg.hidden);
        const emb_ln_b = try loader.vector(a, store, "embeddings.LayerNorm.bias", cfg.hidden);

        const layers = try a.alloc(Layer, cfg.n_layers);
        for (layers, 0..) |*layer, l| {
            layer.* = .{
                .qkv = try loader.indexedMatrix(store, "encoder.layer.", l, "attention.qkv_proj.weight", 3 * cfg.hidden, cfg.hidden),
                .qkv_bias = try loader.indexedVector(a, store, "encoder.layer.", l, "attention.qkv_proj.bias", 3 * cfg.hidden),
                .o = try loader.indexedMatrix(store, "encoder.layer.", l, "attention.o_proj.weight", cfg.hidden, cfg.hidden),
                .o_bias = try loader.indexedVector(a, store, "encoder.layer.", l, "attention.o_proj.bias", cfg.hidden),
                .attn_ln_w = try loader.indexedVector(a, store, "encoder.layer.", l, "attn_ln.weight", cfg.hidden),
                .attn_ln_b = try loader.indexedVector(a, store, "encoder.layer.", l, "attn_ln.bias", cfg.hidden),
                .up_gate = try loader.indexedMatrix(store, "encoder.layer.", l, "mlp.up_gate_proj.weight", 2 * cfg.intermediate, cfg.hidden),
                .down = try loader.indexedMatrix(store, "encoder.layer.", l, "mlp.down_proj.weight", cfg.hidden, cfg.intermediate),
                .down_bias = try loader.indexedVector(a, store, "encoder.layer.", l, "mlp.down_proj.bias", cfg.hidden),
                .mlp_ln_w = try loader.indexedVector(a, store, "encoder.layer.", l, "mlp_ln.weight", cfg.hidden),
                .mlp_ln_b = try loader.indexedVector(a, store, "encoder.layer.", l, "mlp_ln.bias", cfg.hidden),
            };
        }

        return .{
            .arena = arena,
            .st = st,
            .cfg = cfg,
            .word_emb = word_emb,
            .token_type0 = token_type0,
            .emb_ln_w = emb_ln_w,
            .emb_ln_b = emb_ln_b,
            .layers = layers,
        };
    }

    pub fn deinit(self: *Model) void {
        self.st.deinit();
        self.arena.deinit();
        self.* = undefined;
    }

    /// Batched encode: `ids_list[i]` (already framed `<s> … </s>`) → `outs[i]`
    /// [embed_dim]. All items are packed into one contiguous [total_rows, hidden]
    /// activation (ragged — no padding), so every GEMM / LayerNorm / GeGLU runs
    /// once over `sum(seq_i)` rows and amortizes fork/join + weight reuse across
    /// the batch. RoPE and attention are the only per-item ops (each item attends
    /// only itself), so they loop over the batch. Bit-identical to calling
    /// `embed` per item (same math, just fused rows).
    pub fn embedBatch(self: *const Model, io: std.Io, gpa: std.mem.Allocator, ids_list: []const []const u32, outs: [][]f32) !void {
        const cfg = self.cfg;
        const h = cfg.hidden;
        const b = ids_list.len;
        std.debug.assert(outs.len == b and b > 0);

        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const a = arena.allocator();

        // Per-item row offsets into the packed activation, and the batch's max
        // sequence length (for the shared RoPE table).
        const off = try a.alloc(usize, b + 1);
        off[0] = 0;
        var max_seq: usize = 0;
        for (ids_list, 0..) |ids, i| {
            std.debug.assert(ids.len > 0);
            off[i + 1] = off[i] + ids.len;
            max_seq = @max(max_seq, ids.len);
        }
        const total = off[b];

        // Embeddings: word + token_type[0], then LayerNorm (per row).
        const x = try a.alloc(f32, total * h);
        for (ids_list, 0..) |ids, i| try qwen3.embedTokens(self.word_emb, ids, x[off[i] * h ..][0 .. ids.len * h]);
        for (0..total) |t| {
            for (x[t * h ..][0..h], self.token_type0) |*xi, tt| xi.* += tt;
        }
        ops.norm.layerNorm(x, x, self.emb_ln_w, self.emb_ln_b, cfg.ln_eps);

        var freqs = try ops.rope.rotateHalfFreqs(gpa, max_seq, cfg.head_dim, cfg.rope_theta);
        defer freqs.deinit(gpa);

        const qkv = try a.alloc(f32, total * 3 * h);
        const q = try a.alloc(f32, total * h);
        const k = try a.alloc(f32, total * h);
        const v = try a.alloc(f32, total * h);
        const attn = try a.alloc(f32, total * h);
        const proj = try a.alloc(f32, total * h);
        const ug = try a.alloc(f32, total * 2 * cfg.intermediate);
        const up = try a.alloc(f32, total * cfg.intermediate);
        const gate = try a.alloc(f32, total * cfg.intermediate);
        const mlp = try a.alloc(f32, total * h);

        for (self.layers) |*layer| {
            // Packed qkv over the whole batch, then split per row.
            try ops.matmul.matmul(io, gpa, qkv, x, total, layer.qkv, layer.qkv_bias);
            for (0..total) |t| {
                const src = qkv[t * 3 * h ..];
                @memcpy(q[t * h ..][0..h], src[0..h]);
                @memcpy(k[t * h ..][0..h], src[h .. 2 * h]);
                @memcpy(v[t * h ..][0..h], src[2 * h .. 3 * h]);
            }
            // RoPE + non-causal attention, per item (each attends only itself).
            for (0..b) |i| {
                const s0 = off[i] * h;
                const L = ids_list[i].len;
                ops.rope.applyRotateHalfAt(q[s0..][0 .. L * h], freqs, 0, L, cfg.n_heads, cfg.head_dim);
                ops.rope.applyRotateHalfAt(k[s0..][0 .. L * h], freqs, 0, L, cfg.n_heads, cfg.head_dim);
                try ops.attention.attention(io, gpa, attn[s0..][0 .. L * h], q[s0..][0 .. L * h], k[s0..][0 .. L * h], v[s0..][0 .. L * h], .{
                    .seq_q = L,
                    .seq_kv = L,
                    .n_heads = cfg.n_heads,
                    .n_kv_heads = cfg.n_heads,
                    .head_dim = cfg.head_dim,
                    .causal = false,
                });
            }
            try ops.matmul.matmul(io, gpa, proj, attn, total, layer.o, layer.o_bias);
            for (x, proj) |*xi, pi| xi.* += pi; // residual
            ops.norm.layerNorm(x, x, layer.attn_ln_w, layer.attn_ln_b, cfg.ln_eps);

            // GeGLU MLP over the whole batch.
            try ops.matmul.matmul(io, gpa, ug, x, total, layer.up_gate, null);
            for (0..total) |t| {
                const src = ug[t * 2 * cfg.intermediate ..];
                @memcpy(up[t * cfg.intermediate ..][0..cfg.intermediate], src[0..cfg.intermediate]);
                @memcpy(gate[t * cfg.intermediate ..][0..cfg.intermediate], src[cfg.intermediate .. 2 * cfg.intermediate]);
            }
            ops.act.geluErfMul(gate, up);
            try ops.matmul.matmul(io, gpa, mlp, gate, total, layer.down, layer.down_bias);
            for (x, mlp) |*xi, mi| xi.* += mi; // residual
            ops.norm.layerNorm(x, x, layer.mlp_ln_w, layer.mlp_ln_b, cfg.ln_eps);
        }

        // CLS pooling (each item's first row) + L2 normalize.
        for (0..b) |i| {
            const out = outs[i];
            std.debug.assert(out.len == embed_dim);
            @memcpy(out, x[off[i] * h ..][0..h]);
            l2normalize(out);
        }
    }

    /// Encode `ids` (already framed `<s> … </s>`) into `out` [embed_dim], an
    /// L2-normalized CLS embedding.
    pub fn embed(self: *const Model, io: std.Io, gpa: std.mem.Allocator, ids: []const u32, out: []f32) !void {
        const cfg = self.cfg;
        const h = cfg.hidden;
        const seq = ids.len;
        std.debug.assert(out.len == embed_dim and seq > 0);

        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const a = arena.allocator();

        // Embeddings: word + token_type[0], then LayerNorm.
        const x = try a.alloc(f32, seq * h);
        try qwen3.embedTokens(self.word_emb, ids, x);
        for (0..seq) |t| {
            for (x[t * h ..][0..h], self.token_type0) |*xi, tt| xi.* += tt;
        }
        ops.norm.layerNorm(x, x, self.emb_ln_w, self.emb_ln_b, cfg.ln_eps);

        var freqs = try ops.rope.rotateHalfFreqs(gpa, seq, cfg.head_dim, cfg.rope_theta);
        defer freqs.deinit(gpa);

        const qkv = try a.alloc(f32, seq * 3 * h);
        const q = try a.alloc(f32, seq * h);
        const k = try a.alloc(f32, seq * h);
        const v = try a.alloc(f32, seq * h);
        const attn = try a.alloc(f32, seq * h);
        const proj = try a.alloc(f32, seq * h);
        const ug = try a.alloc(f32, seq * 2 * cfg.intermediate);
        const up = try a.alloc(f32, seq * cfg.intermediate);
        const gate = try a.alloc(f32, seq * cfg.intermediate);
        const mlp = try a.alloc(f32, seq * h);

        for (self.layers) |*layer| {
            // Attention: packed qkv → split → RoPE(q,k) → non-causal attention.
            try ops.matmul.matmul(io, gpa, qkv, x, seq, layer.qkv, layer.qkv_bias);
            for (0..seq) |t| {
                const src = qkv[t * 3 * h ..];
                @memcpy(q[t * h ..][0..h], src[0..h]);
                @memcpy(k[t * h ..][0..h], src[h .. 2 * h]);
                @memcpy(v[t * h ..][0..h], src[2 * h .. 3 * h]);
            }
            ops.rope.applyRotateHalfAt(q, freqs, 0, seq, cfg.n_heads, cfg.head_dim);
            ops.rope.applyRotateHalfAt(k, freqs, 0, seq, cfg.n_heads, cfg.head_dim);
            try ops.attention.attention(io, gpa, attn, q, k, v, .{
                .seq_q = seq,
                .seq_kv = seq,
                .n_heads = cfg.n_heads,
                .n_kv_heads = cfg.n_heads,
                .head_dim = cfg.head_dim,
                .causal = false,
            });
            try ops.matmul.matmul(io, gpa, proj, attn, seq, layer.o, layer.o_bias);
            for (x, proj) |*xi, pi| xi.* += pi; // residual
            ops.norm.layerNorm(x, x, layer.attn_ln_w, layer.attn_ln_b, cfg.ln_eps);

            // GeGLU MLP: up_gate → split → gelu(gate)*up → down.
            try ops.matmul.matmul(io, gpa, ug, x, seq, layer.up_gate, null);
            for (0..seq) |t| {
                const src = ug[t * 2 * cfg.intermediate ..];
                @memcpy(up[t * cfg.intermediate ..][0..cfg.intermediate], src[0..cfg.intermediate]);
                @memcpy(gate[t * cfg.intermediate ..][0..cfg.intermediate], src[cfg.intermediate .. 2 * cfg.intermediate]);
            }
            ops.act.geluErfMul(gate, up); // gate := gelu(gate) * up
            try ops.matmul.matmul(io, gpa, mlp, gate, seq, layer.down, layer.down_bias);
            for (x, mlp) |*xi, mi| xi.* += mi; // residual
            ops.norm.layerNorm(x, x, layer.mlp_ln_w, layer.mlp_ln_b, cfg.ln_eps);
        }

        // CLS pooling (token 0) + L2 normalize.
        @memcpy(out, x[0..h]);
        l2normalize(out);
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

// Numeric parity vs the Snowflake f32 ONNX (reference vectors generated by the
// venv in scratch_out; keyed to the golden token ids). Requires the checkout
// under ../DiffKeep; skipped when absent.
test "snowflake arctic embed matches ONNX reference" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dir = "../DiffKeep/Models/snowflake-arctic-embed-m-v2.0";
    std.Io.Dir.cwd().access(io, dir, .{}) catch return error.SkipZigTest;

    const ref_bytes = std.Io.Dir.cwd().readFileAlloc(io, "testdata/snowflake_ref_vectors.json", gpa, .limited(4 * 1024 * 1024)) catch return error.SkipZigTest;
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
        for (ids_json, ids) |val, *d| d.* = @intCast(val.integer);

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

// Batched forward must be bit-close to per-item (same math, fused rows). Uses
// synthetic weights so it runs in the fast CPU suite (no checkpoint).
test "snowflake embedBatch matches per-item" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    // hidden must equal embed_dim (CLS pool copies the full hidden row); keep
    // everything else small so the test stays in the fast CPU suite.
    const cfg: Config = .{ .n_layers = 2, .hidden = embed_dim, .n_heads = 12, .head_dim = 64, .intermediate = 128, .vocab = 100, .ln_eps = 1e-12, .rope_theta = 160000.0 };
    var prng = std.Random.DefaultPrng.init(0xABCD);
    const r = prng.random();
    const mat = struct {
        fn f(al: std.mem.Allocator, rnd: std.Random, rows: usize, cols: usize) !Weight {
            const d = try al.alloc(f32, rows * cols);
            for (d) |*e| e.* = (rnd.float(f32) - 0.5) * 0.2;
            return Weight.fromF32(d, rows, cols);
        }
    }.f;
    const vecf = struct {
        fn f(al: std.mem.Allocator, rnd: std.Random, n: usize, center: f32) ![]f32 {
            const d = try al.alloc(f32, n);
            for (d) |*e| e.* = center + (rnd.float(f32) - 0.5) * 0.1;
            return d;
        }
    }.f;

    const layers = try a.alloc(Layer, cfg.n_layers);
    for (layers) |*ly| ly.* = .{
        .qkv = try mat(a, r, 3 * cfg.hidden, cfg.hidden),
        .qkv_bias = try vecf(a, r, 3 * cfg.hidden, 0),
        .o = try mat(a, r, cfg.hidden, cfg.hidden),
        .o_bias = try vecf(a, r, cfg.hidden, 0),
        .attn_ln_w = try vecf(a, r, cfg.hidden, 1),
        .attn_ln_b = try vecf(a, r, cfg.hidden, 0),
        .up_gate = try mat(a, r, 2 * cfg.intermediate, cfg.hidden),
        .down = try mat(a, r, cfg.hidden, cfg.intermediate),
        .down_bias = try vecf(a, r, cfg.hidden, 0),
        .mlp_ln_w = try vecf(a, r, cfg.hidden, 1),
        .mlp_ln_b = try vecf(a, r, cfg.hidden, 0),
    };
    const word_data = try a.alloc(f32, cfg.vocab * cfg.hidden);
    for (word_data) |*e| e.* = (r.float(f32) - 0.5) * 0.2;

    var model: Model = .{
        .arena = undefined,
        .st = undefined,
        .cfg = cfg,
        .word_emb = Weight.fromF32(word_data, cfg.vocab, cfg.hidden),
        .token_type0 = try vecf(a, r, cfg.hidden, 0),
        .emb_ln_w = try vecf(a, r, cfg.hidden, 1),
        .emb_ln_b = try vecf(a, r, cfg.hidden, 0),
        .layers = layers,
    };

    // Ragged batch: three items of different lengths.
    const item0 = [_]u32{ 0, 5, 9, 2 };
    const item1 = [_]u32{ 0, 11, 3, 7, 42, 2 };
    const item2 = [_]u32{ 0, 88, 2 };
    const ids_list = [_][]const u32{ &item0, &item1, &item2 };

    var single: [3][embed_dim]f32 = undefined;
    for (ids_list, 0..) |ids, i| try model.embed(io, gpa, ids, &single[i]);

    var batched: [3][embed_dim]f32 = undefined;
    var outs: [3][]f32 = .{ &batched[0], &batched[1], &batched[2] };
    try model.embedBatch(io, gpa, &ids_list, &outs);

    for (0..3) |i| {
        var maxad: f32 = 0;
        for (single[i], batched[i]) |sv, bv| maxad = @max(maxad, @abs(sv - bv));
        errdefer std.debug.print("item {d}: max abs diff {d}\n", .{ i, maxad });
        try std.testing.expect(maxad < 1e-5);
    }
}
