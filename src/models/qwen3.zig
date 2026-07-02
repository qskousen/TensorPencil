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
//! Weights stay in checkpoint dtype (fp8-e4m3 + per-tensor f32 scales) and are
//! dequantized inside the GEMM; the safetensors mapping must outlive this.

const std = @import("std");
const safetensors = @import("../safetensors.zig");
const dtypes = @import("../dtype.zig");
const ops = @import("../ops.zig");

const SafeTensors = safetensors.SafeTensors;
const Weight = ops.matmul.Weight;

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

const q_dim = n_heads * head_dim; // 4096
const kv_dim = n_kv_heads * head_dim; // 1024

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

        const layers = try alloc.alloc(Layer, n_layers - 1);
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
        for (ids, 0..) |id, t| {
            if (id >= vocab_size) return error.TokenIdOutOfRange;
            const row = self.embed_bytes[@as(usize, id) * hidden * 2 ..][0 .. hidden * 2];
            try safetensors.convertToF32(.bf16, row, x[t * hidden ..][0..hidden]);
        }

        var freqs = try ops.rope.rotateHalfFreqs(gpa, seq, head_dim, rope_theta);
        defer freqs.deinit(gpa);

        const normed = try gpa.alloc(f32, seq * hidden);
        defer gpa.free(normed);
        const tmp = try gpa.alloc(f32, seq * hidden);
        defer gpa.free(tmp);
        const q = try gpa.alloc(f32, seq * q_dim);
        defer gpa.free(q);
        const k = try gpa.alloc(f32, seq * kv_dim);
        defer gpa.free(k);
        const v = try gpa.alloc(f32, seq * kv_dim);
        defer gpa.free(v);
        const attn_out = try gpa.alloc(f32, seq * q_dim);
        defer gpa.free(attn_out);
        const gate = try gpa.alloc(f32, seq * intermediate);
        defer gpa.free(gate);
        const up = try gpa.alloc(f32, seq * intermediate);
        defer gpa.free(up);

        var tap_idx: usize = 0;
        for (0..n_layers) |l| {
            if (tap_idx < tap_layers.len and tap_layers[tap_idx] == l) {
                for (0..seq) |t| {
                    @memcpy(out[(t * tap_count + tap_idx) * hidden ..][0..hidden], x[t * hidden ..][0..hidden]);
                }
                tap_idx += 1;
            }
            if (l >= self.layers.len) break;
            const layer = self.layers[l];

            // Attention.
            ops.norm.rmsNorm(normed, x, layer.input_norm, rms_eps);
            try ops.matmul.matmul(io, gpa, q, normed, seq, layer.q, null);
            try ops.matmul.matmul(io, gpa, k, normed, seq, layer.k, null);
            try ops.matmul.matmul(io, gpa, v, normed, seq, layer.v, null);
            ops.norm.rmsNorm(q, q, layer.q_norm, rms_eps); // per-head: rows of head_dim
            ops.norm.rmsNorm(k, k, layer.k_norm, rms_eps);
            ops.rope.applyRotateHalf(q, freqs, seq, n_heads, head_dim);
            ops.rope.applyRotateHalf(k, freqs, seq, n_kv_heads, head_dim);
            try ops.attention.attention(io, gpa, attn_out, q, k, v, .{
                .seq_q = seq,
                .seq_kv = seq,
                .n_heads = n_heads,
                .n_kv_heads = n_kv_heads,
                .head_dim = head_dim,
                .causal = true,
            });
            try ops.matmul.matmul(io, gpa, tmp, attn_out, seq, layer.o, null);
            for (x, tmp) |*xi, ti| xi.* += ti;

            // MLP.
            ops.norm.rmsNorm(normed, x, layer.post_norm, rms_eps);
            try ops.matmul.matmul(io, gpa, gate, normed, seq, layer.gate, null);
            try ops.matmul.matmul(io, gpa, up, normed, seq, layer.up, null);
            ops.act.siluMul(gate, up);
            try ops.matmul.matmul(io, gpa, tmp, gate, seq, layer.down, null);
            for (x, tmp) |*xi, ti| xi.* += ti;
        }
        std.debug.assert(tap_idx == tap_count);
        return out;
    }
};

fn loadNorm(alloc: std.mem.Allocator, st: *const SafeTensors, layer: usize, comptime suffix: []const u8, len: usize) ![]f32 {
    var buf: [96]u8 = undefined;
    const name = try std.fmt.bufPrint(&buf, "model.language_model.layers.{d}." ++ suffix, .{layer});
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
