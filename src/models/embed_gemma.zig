//! EmbeddingGemma (google/embeddinggemma-300m), a Gemma-3 text ENCODER mapping text to
//! one L2-normalized 768-d embedding.
//!
//! Architecturally the Gemma-3 decoder body (4-norm sandwich, per-head QK-norm,
//! GeGLU/gelu-tanh MLP, dual-RoPE with a global layer every 6th) run BIDIRECTIONALLY,
//! then a sentence-transformers head: mean-pool over all tokens, Dense 768->3072, Dense
//! 3072->768 (both no-bias, identity activation), L2 normalize.
//!
//! The body runs through the shared `transformer.gemma3_spec` and
//! `layerForward(.fresh, bidirectional=true)`, i.e. the tested gemma3 math. Only the
//! head and the safetensors loader are new here.
//!
//! Two loading details differ from the GGUF gemma3 path. HF `Gemma3RMSNorm` computes
//! `x * (1 + w)` and the GGUF converter folds the +1 in, while raw HF safetensors do
//! not, so `normVec` adds 1.0 to every norm vector at load. And the tensor names are
//! bare HF names with no `model.` prefix in this sentence-transformers export, so
//! `loader.*` matches directly.
//!
//! The local layers attend bidirectionally to keys within `window_radius` (256) positions
//! either side, not 512: transformers turns a bidirectional config's `sliding_window`
//! of 512 into an exclusive bound of 257 (`tools/gen_embeddinggemma_long.py` checks
//! it). Global layers see the whole input.

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
/// null (no linear position scaling -> freq_scale 1.0) and query_pre_attn_scalar
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

/// Longest framed input (`<bos> ... <eos>`), the model's `max_position_embeddings`.
pub const max_seq: usize = 2048;
/// A local layer's query attends keys with `|q - k| <= window_radius`.
pub const window_radius: usize = config_300m.sliding_window / 2;

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

/// Layer `l`'s dimensions; a bidirectional pass reads `sliding_window` as the radius.
fn dims(l: usize) transformer.Dims {
    const cfg = config_300m;
    return .{
        .hidden = cfg.hidden,
        .n_heads = cfg.n_heads,
        .n_kv = cfg.n_kv_heads,
        .head_dim = cfg.head_dim,
        .q_dim = cfg.qDim(),
        .kv_dim = cfg.kvDim(),
        .intermediate = cfg.intermediate,
        .sliding_window = if (cfg.isGlobal(l)) 0 else window_radius,
    };
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

        var body_st = try SafeTensors.open(gpa, io, try std.mem.print(&pbuf, "{s}/model.safetensors", .{dir}));
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

        var dense1_st = try SafeTensors.open(gpa, io, try std.mem.print(&pbuf, "{s}/2_Dense/model.safetensors", .{dir}));
        errdefer dense1_st.deinit();
        const dense1 = try loader.matrix(.{ .safetensors = &dense1_st }, "linear.weight", proj_hidden, cfg.hidden);

        var dense2_st = try SafeTensors.open(gpa, io, try std.mem.print(&pbuf, "{s}/3_Dense/model.safetensors", .{dir}));
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
    /// must already carry the model frame (`<bos> ... <eos>`).
    pub fn forwardHidden(self: *const Model, io: std.Io, gpa: std.mem.Allocator, ids: []const u32, out: []f32) !void {
        const cfg = self.cfg;
        const seq = ids.len;
        std.debug.assert(seq > 0);
        std.debug.assert(out.len == seq * cfg.hidden);
        if (seq > max_seq) return error.SequenceTooLong;

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
            const freqs = if (cfg.isGlobal(l)) fg else fl;
            try transformer.layerForward(transformer.gemma3_spec, .fresh, io, gpa, layer, x, seq, dims(l), freqs, cfg.rms_eps, {}, l, 0, true, &s);
        }

        ops.norm.rmsNorm(out, x, self.final_norm, cfg.rms_eps);
    }

    /// Batched encode: `ids_list[i]` (framed `<bos> ... <eos>`) -> `outs[i]`
    /// [embed_dim]. All sequences are packed contiguously into one
    /// [total_rows, hidden] activation (ragged, no padding), so the whole
    /// Gemma-3 body runs through `transformer.layerForwardBatchedFresh`: every
    /// projection / norm / GeGLU batches over `sum(seq_i)` rows and only RoPE +
    /// attention loop per item. The head (mean-pool + 2 Dense + normalize) is
    /// then applied per item. Bit-identical to per-item `embed`.
    pub fn embedBatch(self: *const Model, io: std.Io, gpa: std.mem.Allocator, ids_list: []const []const u32, outs: [][]f32) !void {
        const cfg = self.cfg;
        const b = ids_list.len;
        std.debug.assert(outs.len == b and b > 0);

        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const a = arena.allocator();

        const row_off = try a.alloc(usize, b + 1);
        row_off[0] = 0;
        var longest: usize = 0;
        for (ids_list, 0..) |ids, i| {
            std.debug.assert(ids.len > 0);
            if (ids.len > max_seq) return error.SequenceTooLong;
            row_off[i + 1] = row_off[i] + ids.len;
            longest = @max(longest, ids.len);
        }
        const total = row_off[b];

        const x = try a.alloc(f32, total * cfg.hidden);
        for (ids_list, 0..) |ids, i| try qwen3.embedTokens(self.embed_w, ids, x[row_off[i] * cfg.hidden ..][0 .. ids.len * cfg.hidden]);
        const scale = cfg.embedScale();
        for (x) |*v| v.* *= scale;

        var fg = try ops.rope.rotateHalfFreqs(gpa, longest, cfg.head_dim, cfg.rope_theta);
        defer fg.deinit(gpa);
        var fl = try ops.rope.rotateHalfFreqs(gpa, longest, cfg.head_dim, cfg.rope_theta_local);
        defer fl.deinit(gpa);

        var s = try gemma3.Scratch.init(gpa, total, cfg);
        defer s.deinit(gpa);

        for (self.layers, 0..) |*layer, l| {
            const freqs = if (cfg.isGlobal(l)) fg else fl;
            try transformer.layerForwardBatchedFresh(transformer.gemma3_spec, io, gpa, layer, x, row_off, dims(l), freqs, cfg.rms_eps, true, &s);
        }

        // final norm (per row) -> per-item head.
        const lhs = try a.alloc(f32, total * cfg.hidden);
        ops.norm.rmsNorm(lhs, x, self.final_norm, cfg.rms_eps);
        for (0..b) |i| {
            std.debug.assert(outs[i].len == embed_dim);
            try self.head(io, gpa, lhs[row_off[i] * cfg.hidden ..][0 .. (row_off[i + 1] - row_off[i]) * cfg.hidden], outs[i]);
        }
    }

    /// Encode `ids` (already framed `<bos> ... <eos>`) into `out` [embed_dim], an
    /// L2-normalized 768-d embedding: bidirectional body -> mean-pool over all
    /// tokens -> Dense 768->3072 -> Dense 3072->768 -> normalize.
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
    /// [seq * hidden]: mean-pool over all tokens -> Dense 768->3072 -> Dense
    /// 3072->768 -> L2-normalize. Shared with the GPU path (body on device ->
    /// download `lhs` -> this on host). Cheap; always host.
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

pub const test_dir = "../DiffKeep/Models/embeddinggemma-300m";
const test_gate = @import("../test_gate.zig");

pub fn cosine(a: []const f32, b: []const f32) f32 {
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (a, b) |x, y| {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    return @floatCast(dot / @sqrt(na * nb));
}

/// A case from `tools/gen_embeddinggemma_long.py`: `wide` and `nowindow` are what a
/// port with a window one too wide, or none, would produce.
pub const LongCase = struct { name: []const u8, ids: []u32, vec: []f32, wide: []f32, nowindow: []f32 };

pub fn loadLongReference(arena: std.mem.Allocator, io: std.Io) ![]LongCase {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, "testdata/embeddinggemma_long_ref.json", arena, .limited(4 * 1024 * 1024));
    const root = try std.json.parseFromSliceLeaky(std.json.Value, arena, bytes, .{});
    var cases: std.ArrayList(LongCase) = .empty;
    var it = root.object.iterator();
    while (it.next()) |e| {
        const o = e.value_ptr.object;
        const ids_j = o.get("ids").?.array.items;
        const ids = try arena.alloc(u32, ids_j.len);
        for (ids, ids_j) |*d, v| d.* = @intCast(v.integer);
        try cases.append(arena, .{
            .name = e.key_ptr.*,
            .ids = ids,
            .vec = try floats(arena, o.get("vec").?.array.items),
            .wide = try floats(arena, o.get("vec_wide").?.array.items),
            .nowindow = try floats(arena, o.get("vec_nowindow").?.array.items),
        });
    }
    return cases.items;
}

fn floats(arena: std.mem.Allocator, items: []const std.json.Value) ![]f32 {
    const out = try arena.alloc(f32, items.len);
    for (out, items) |*d, v| d.* = @floatCast(v.float);
    return out;
}

/// `got` must be within `min_cos` of the reference and closer to it than to either
/// wrong-window vector.
pub fn expectLong(c: LongCase, got: []const f32, min_cos: f32) !void {
    const cos = cosine(got, c.vec);
    const cos_wide = cosine(got, c.wide);
    const cos_nw = cosine(got, c.nowindow);
    errdefer std.debug.print("{s} ({d} tokens): cosine {d}, to the wide window {d}, to no window {d}\n", .{ c.name, c.ids.len, cos, cos_wide, cos_nw });
    try std.testing.expect(cos >= min_cos);
    try std.testing.expect(cos > cos_wide and cos > cos_nw);
}

test "embeddinggemma matches sentence-transformers past the window" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    try test_gate.requireModelFile(io, test_dir ++ "/model.safetensors");
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const cases = try loadLongReference(arena.allocator(), io);
    var model = try Model.open(gpa, io, test_dir);
    defer model.deinit();

    var out: [embed_dim]f32 = undefined;
    for (cases) |c| {
        try model.embed(io, gpa, c.ids, &out);
        try expectLong(c, &out, 0.999999);
    }

    // A ragged batch of every case and a short item, against the reference.
    const short = [_]u32{ 2, 23391, 1902, 1 };
    var short_vec: [embed_dim]f32 = undefined;
    try model.embed(io, gpa, &short, &short_vec);
    const ids_list = try arena.allocator().alloc([]const u32, cases.len + 1);
    const outs = try arena.allocator().alloc([]f32, cases.len + 1);
    for (cases, ids_list[1..], outs[1..]) |c, *ids, *o| {
        ids.* = c.ids;
        o.* = try arena.allocator().alloc(f32, embed_dim);
    }
    ids_list[0] = &short;
    outs[0] = try arena.allocator().alloc(f32, embed_dim);
    try model.embedBatch(io, gpa, ids_list, outs);
    for (cases, outs[1..]) |c, o| try expectLong(c, o, 0.999999);
    const cos_short = cosine(&short_vec, outs[0]);
    errdefer std.debug.print("short item batched with long ones: cosine {d}\n", .{cos_short});
    try std.testing.expect(cos_short >= 0.999999);

    const over = try gpa.alloc(u32, max_seq + 1);
    defer gpa.free(over);
    @memset(over, 2);
    try std.testing.expectError(error.SequenceTooLong, model.embed(io, gpa, over, &out));
    try std.testing.expectError(error.SequenceTooLong, model.embedBatch(io, gpa, &.{ &short, over }, outs[0..2]));
}

// Numeric parity vs the onnx-community EmbeddingGemma ONNX (reference vectors in
// testdata/, generated by scratch_out/gen_embeddinggemma_ref.py). Requires the
// unsloth safetensors checkout under ../DiffKeep; skipped when absent.
test "embeddinggemma matches ONNX reference" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dir = test_dir;
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

test "embeddinggemma embedBatch matches per-item" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dir = test_dir;
    std.Io.Dir.cwd().access(io, dir, .{}) catch return error.SkipZigTest;

    var model = try Model.open(gpa, io, dir);
    defer model.deinit();

    // ragged batch: framed <bos>=2 ... <eos>=1, different lengths.
    const item0 = [_]u32{ 2, 1000, 2049, 1 };
    const item1 = [_]u32{ 2, 500, 600, 700, 800, 1 };
    const item2 = [_]u32{ 2, 42, 1 };
    const ids_list = [_][]const u32{ &item0, &item1, &item2 };

    var single: [3][embed_dim]f32 = undefined;
    for (ids_list, 0..) |ids, i| try model.embed(io, gpa, ids, &single[i]);
    var batched: [3][embed_dim]f32 = undefined;
    var outs: [3][]f32 = .{ &batched[0], &batched[1], &batched[2] };
    try model.embedBatch(io, gpa, &ids_list, &outs);

    for (0..3) |i| {
        var maxad: f32 = 0;
        for (single[i], batched[i]) |sv, bv| maxad = @max(maxad, @abs(sv - bv));
        errdefer std.debug.print("gemma item {d}: max abs diff {d}\n", .{ i, maxad });
        try std.testing.expect(maxad < 1e-4);
    }
}
