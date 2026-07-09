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
const safetensors = @import("../safetensors.zig");
const dtypes = @import("../dtype.zig");
const ops = @import("../ops.zig");
const kv_cache_mod = @import("../llm/kv_cache.zig");

const SafeTensors = safetensors.SafeTensors;
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
        try embedTokens(self.embed_bytes, ids, x);

        var freqs = try ops.rope.rotateHalfFreqs(gpa, seq, head_dim, rope_theta);
        defer freqs.deinit(gpa);

        var scratch = try Scratch.init(gpa, seq);
        defer scratch.deinit(gpa);

        var tap_idx: usize = 0;
        for (0..n_layers) |l| {
            if (tap_idx < tap_layers.len and tap_layers[tap_idx] == l) {
                for (0..seq) |t| {
                    @memcpy(out[(t * tap_count + tap_idx) * hidden ..][0..hidden], x[t * hidden ..][0..hidden]);
                }
                tap_idx += 1;
            }
            if (l >= self.layers.len) break;
            try layerForward(io, gpa, self.layers[l], x, seq, freqs, &scratch);
        }
        std.debug.assert(tap_idx == tap_count);
        return out;
    }
};

/// The full Qwen3-VL-4B text stack as a language model: all 36 layers, final
/// norm, and the embedding matrix doubling as the tied LM head.
pub const CausalLM = struct {
    arena: std.heap.ArenaAllocator,
    /// bf16 [vocab, hidden] view into the mapped file; also the tied LM head.
    embed_bytes: []const u8,
    layers: []Layer,
    final_norm: []const f32,

    pub fn load(gpa: std.mem.Allocator, st: *const SafeTensors) !CausalLM {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();

        const embed = try st.require("model.language_model.embed_tokens.weight");
        if (embed.info.dtype != .bf16 or embed.info.elemCount() != vocab_size * hidden)
            return error.ShapeMismatch;

        const layers = try loadLayers(alloc, st, n_layers);
        const final_norm = try loadNormNamed(alloc, st, "model.language_model.norm.weight", hidden);

        return .{ .arena = arena, .embed_bytes = embed.bytes, .layers = layers, .final_norm = final_norm };
    }

    pub fn deinit(self: *CausalLM) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// Tied LM head: logits = hidden @ embed^T, [vocab] per position.
    pub fn lmHead(self: *const CausalLM) Weight {
        return Weight.init(self.embed_bytes, .bf16, vocab_size, hidden);
    }

    /// Forward `ids` at absolute positions [cache.len, cache.len + ids.len),
    /// appending their K/V to the cache: prefill when the cache is empty,
    /// single-token decode when ids.len == 1. `freqs` must cover the final
    /// position. When `out` ([hidden]) is set, it receives the final-normed
    /// hidden state of the last new position, ready for the LM head.
    pub fn forwardCached(
        self: *const CausalLM,
        io: std.Io,
        gpa: std.mem.Allocator,
        ids: []const u32,
        cache: *KvCache,
        freqs: ops.rope.Freqs,
        out: ?[]f32,
    ) !void {
        const seq = ids.len;
        std.debug.assert(seq > 0 and seq <= cache.remaining());
        std.debug.assert(cache.n_layers == n_layers and cache.kv_dim == kv_dim);

        const x = try gpa.alloc(f32, seq * hidden);
        defer gpa.free(x);
        try embedTokens(self.embed_bytes, ids, x);

        var scratch = try Scratch.init(gpa, seq);
        defer scratch.deinit(gpa);

        for (self.layers, 0..) |layer, l| {
            try layerForwardCached(io, gpa, layer, x, seq, freqs, cache, l, &scratch);
        }
        cache.commit(seq);
        if (out) |o| {
            std.debug.assert(o.len == hidden);
            ops.norm.rmsNorm(o, x[(seq - 1) * hidden ..][0..hidden], self.final_norm, rms_eps);
        }
    }
};

/// Look up bf16 embedding rows for `ids` into `x` [ids.len, hidden] f32.
fn embedTokens(embed_bytes: []const u8, ids: []const u32, x: []f32) !void {
    for (ids, 0..) |id, t| {
        if (id >= vocab_size) return error.TokenIdOutOfRange;
        const row = embed_bytes[@as(usize, id) * hidden * 2 ..][0 .. hidden * 2];
        try safetensors.convertToF32(.bf16, row, x[t * hidden ..][0..hidden]);
    }
}

/// Per-forward activation buffers, sized for `seq` tokens.
const Scratch = struct {
    normed: []f32,
    tmp: []f32,
    q: []f32,
    k: []f32,
    v: []f32,
    attn_out: []f32,
    gate: []f32,
    up: []f32,

    fn init(gpa: std.mem.Allocator, seq: usize) !Scratch {
        var s: Scratch = undefined;
        s.normed = try gpa.alloc(f32, seq * hidden);
        errdefer gpa.free(s.normed);
        s.tmp = try gpa.alloc(f32, seq * hidden);
        errdefer gpa.free(s.tmp);
        s.q = try gpa.alloc(f32, seq * q_dim);
        errdefer gpa.free(s.q);
        s.k = try gpa.alloc(f32, seq * kv_dim);
        errdefer gpa.free(s.k);
        s.v = try gpa.alloc(f32, seq * kv_dim);
        errdefer gpa.free(s.v);
        s.attn_out = try gpa.alloc(f32, seq * q_dim);
        errdefer gpa.free(s.attn_out);
        s.gate = try gpa.alloc(f32, seq * intermediate);
        errdefer gpa.free(s.gate);
        s.up = try gpa.alloc(f32, seq * intermediate);
        errdefer gpa.free(s.up);
        return s;
    }

    fn deinit(self: *Scratch, gpa: std.mem.Allocator) void {
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

/// One transformer layer over `x` [seq, hidden], residuals added in place.
fn layerForward(io: std.Io, gpa: std.mem.Allocator, layer: Layer, x: []f32, seq: usize, freqs: ops.rope.Freqs, s: *Scratch) !void {
    // Attention.
    ops.norm.rmsNorm(s.normed, x, layer.input_norm, rms_eps);
    try ops.matmul.matmul(io, gpa, s.q, s.normed, seq, layer.q, null);
    try ops.matmul.matmul(io, gpa, s.k, s.normed, seq, layer.k, null);
    try ops.matmul.matmul(io, gpa, s.v, s.normed, seq, layer.v, null);
    ops.norm.rmsNorm(s.q, s.q, layer.q_norm, rms_eps); // per-head: rows of head_dim
    ops.norm.rmsNorm(s.k, s.k, layer.k_norm, rms_eps);
    ops.rope.applyRotateHalf(s.q, freqs, seq, n_heads, head_dim);
    ops.rope.applyRotateHalf(s.k, freqs, seq, n_kv_heads, head_dim);
    try ops.attention.attention(io, gpa, s.attn_out, s.q, s.k, s.v, .{
        .seq_q = seq,
        .seq_kv = seq,
        .n_heads = n_heads,
        .n_kv_heads = n_kv_heads,
        .head_dim = head_dim,
        .causal = true,
    });
    try ops.matmul.matmul(io, gpa, s.tmp, s.attn_out, seq, layer.o, null);
    for (x, s.tmp) |*xi, ti| xi.* += ti;

    // MLP.
    ops.norm.rmsNorm(s.normed, x, layer.post_norm, rms_eps);
    try ops.matmul.matmul(io, gpa, s.gate, s.normed, seq, layer.gate, null);
    try ops.matmul.matmul(io, gpa, s.up, s.normed, seq, layer.up, null);
    ops.act.siluMul(s.gate, s.up);
    try ops.matmul.matmul(io, gpa, s.tmp, s.gate, seq, layer.down, null);
    for (x, s.tmp) |*xi, ti| xi.* += ti;
}

/// layerForward against a KV cache: `x` holds only the `seq` NEW tokens (at
/// absolute positions cache.len..), their K/V are appended to layer `l` of
/// the cache, and attention runs over the whole cached prefix.
fn layerForwardCached(
    io: std.Io,
    gpa: std.mem.Allocator,
    layer: Layer,
    x: []f32,
    seq: usize,
    freqs: ops.rope.Freqs,
    cache: *KvCache,
    l: usize,
    s: *Scratch,
) !void {
    const pos0 = cache.len;

    // Attention.
    ops.norm.rmsNorm(s.normed, x, layer.input_norm, rms_eps);
    try ops.matmul.matmul(io, gpa, s.q, s.normed, seq, layer.q, null);
    try ops.matmul.matmul(io, gpa, s.k, s.normed, seq, layer.k, null);
    try ops.matmul.matmul(io, gpa, s.v, s.normed, seq, layer.v, null);
    ops.norm.rmsNorm(s.q, s.q, layer.q_norm, rms_eps); // per-head: rows of head_dim
    ops.norm.rmsNorm(s.k, s.k, layer.k_norm, rms_eps);
    ops.rope.applyRotateHalfAt(s.q, freqs, pos0, seq, n_heads, head_dim);
    ops.rope.applyRotateHalfAt(s.k, freqs, pos0, seq, n_kv_heads, head_dim);
    cache.write(l, s.k[0 .. seq * kv_dim], s.v[0 .. seq * kv_dim]);
    try ops.attention.attention(io, gpa, s.attn_out, s.q, cache.kView(l, seq), cache.vView(l, seq), .{
        .seq_q = seq,
        .seq_kv = pos0 + seq,
        .n_heads = n_heads,
        .n_kv_heads = n_kv_heads,
        .head_dim = head_dim,
        .causal = true,
    });
    try ops.matmul.matmul(io, gpa, s.tmp, s.attn_out, seq, layer.o, null);
    for (x, s.tmp) |*xi, ti| xi.* += ti;

    // MLP.
    ops.norm.rmsNorm(s.normed, x, layer.post_norm, rms_eps);
    try ops.matmul.matmul(io, gpa, s.gate, s.normed, seq, layer.gate, null);
    try ops.matmul.matmul(io, gpa, s.up, s.normed, seq, layer.up, null);
    ops.act.siluMul(s.gate, s.up);
    try ops.matmul.matmul(io, gpa, s.tmp, s.gate, seq, layer.down, null);
    for (x, s.tmp) |*xi, ti| xi.* += ti;
}

fn loadLayers(alloc: std.mem.Allocator, st: *const SafeTensors, count: usize) ![]Layer {
    const layers = try alloc.alloc(Layer, count);
    for (layers, 0..) |*layer, i| {
        layer.* = .{
            .input_norm = try loadNorm(alloc, st, i, "input_layernorm.weight", hidden),
            .q = try loadWeight(alloc, st, i, "self_attn.q_proj.weight", q_dim, hidden),
            .k = try loadWeight(alloc, st, i, "self_attn.k_proj.weight", kv_dim, hidden),
            .v = try loadWeight(alloc, st, i, "self_attn.v_proj.weight", kv_dim, hidden),
            .o = try loadWeight(alloc, st, i, "self_attn.o_proj.weight", hidden, q_dim),
            .q_norm = try loadNorm(alloc, st, i, "self_attn.q_norm.weight", head_dim),
            .k_norm = try loadNorm(alloc, st, i, "self_attn.k_norm.weight", head_dim),
            .post_norm = try loadNorm(alloc, st, i, "post_attention_layernorm.weight", hidden),
            .gate = try loadWeight(alloc, st, i, "mlp.gate_proj.weight", intermediate, hidden),
            .up = try loadWeight(alloc, st, i, "mlp.up_proj.weight", intermediate, hidden),
            .down = try loadWeight(alloc, st, i, "mlp.down_proj.weight", hidden, intermediate),
        };
    }
    return layers;
}

fn loadNorm(alloc: std.mem.Allocator, st: *const SafeTensors, layer: usize, comptime suffix: []const u8, len: usize) ![]f32 {
    var buf: [96]u8 = undefined;
    const name = try std.fmt.bufPrint(&buf, "model.language_model.layers.{d}." ++ suffix, .{layer});
    return loadNormNamed(alloc, st, name, len);
}

fn loadNormNamed(alloc: std.mem.Allocator, st: *const SafeTensors, name: []const u8, len: usize) ![]f32 {
    const view = st.get(name) orelse return error.MissingTensor;
    if (view.info.elemCount() != len) return error.ShapeMismatch;
    return view.toF32Alloc(alloc);
}

fn loadWeight(alloc: std.mem.Allocator, st: *const SafeTensors, layer: usize, comptime suffix: []const u8, rows: usize, cols: usize) !Weight {
    var buf: [96]u8 = undefined;
    const name = try std.fmt.bufPrint(&buf, "model.language_model.layers.{d}." ++ suffix, .{layer});
    const view = st.get(name) orelse return error.MissingTensor;
    const shape = view.info.shape.slice();
    if (shape.len != 2 or shape[0] != rows or shape[1] != cols) return error.ShapeMismatch;
    _ = alloc;
    var w = Weight.init(view.bytes, view.info.dtype, rows, cols);
    var scale_buf: [112]u8 = undefined;
    const scale_name = try std.fmt.bufPrint(&scale_buf, "{s}_scale", .{name});
    if (st.get(scale_name)) |scale_view| {
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

// Parity against ComfyUI's Krea 2 conditioning (f32), post prefix-strip.
// Fixture from tools/dump_text_fixture.py (prompt "a fluffy orange cat
// sitting on a windowsill"); skipped when model or fixture is absent.
test "krea2 conditioning matches comfyui" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const krea2_text = @import("krea2_text.zig");
    const tokenizer_mod = @import("../tokenizer.zig");
    const te_path = "models/text_encoders/qwen3VLInstruct4bHeretic_v10.safetensors";
    std.Io.Dir.cwd().access(io, te_path, .{}) catch return error.SkipZigTest;
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
