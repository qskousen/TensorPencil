//! CLIP text encoder — the SD-family conditioner (`cond_stage_model` in an LDM
//! single-file checkpoint), mirroring `transformers.CLIPTextModel`.
//!
//! A plain pre-LN transformer, but four details are load-bearing and each one
//! produces a *plausible* wrong conditioning rather than an obvious failure:
//!
//! 1. **The attention is causal.** A text encoder that reads bidirectionally is a
//!    different model; every prompt still encodes, and every image still renders.
//! 2. **The MLP activation is `quick_gelu`** (`x·σ(1.702x)`), not tanh-gelu and not
//!    erf-gelu. CLIP-L (SD1.5) uses quick_gelu; CLIP-G (SDXL's second tower) uses
//!    the erf form — `Config.act` carries which.
//! 3. **Positions are a learned table**, not RoPE and not sinusoidal, and the table
//!    is exactly `max_positions` long: SD's 77-token window is a property of the
//!    weights, not a convention we choose.
//! 4. **The output SD conditions on is `final_layer_norm(x)`** — the last hidden
//!    state, layer-normed. Skipping that norm leaves activations off by a scale the
//!    UNet's cross-attention silently absorbs into softer images.
//!
//! Loads from any `WeightStore` and keeps large weights in their checkpoint dtype
//! (so a quantized text tower runs, and so `ops.matmul.probe` can attribute its
//! GEMMs — ggufy quantizes this encoder today and could not measure it before).
//! The token embedding is the exception: it is a lookup table, never a GEMM, and is
//! read a row at a time.

const std = @import("std");
const tp_core = @import("tp_core");
const safetensors = tp_core.safetensors;
const weights_mod = tp_core.weights;
const ops = @import("tp_ops");

const DType = tp_core.dtype.DType;
const WeightStore = weights_mod.WeightStore;
const Weight = ops.matmul.Weight;

pub const Config = struct {
    hidden: usize,
    layers: usize,
    heads: usize,
    intermediate: usize,
    max_positions: usize,
    eps: f32,
    act: Act,
    /// Token id whose row becomes the pooled output. For SD's CLIP vocabularies
    /// `<|endoftext|>`; also the padding token, so "first occurrence" is what
    /// selects the real end of the prompt.
    eos_id: u32,

    pub const Act = enum { quick_gelu, gelu_erf };

    pub fn headDim(self: Config) usize {
        return self.hidden / self.heads;
    }
};

/// SD1.5's conditioner: ViT-L/14's text tower.
pub const clip_l: Config = .{
    .hidden = 768,
    .layers = 12,
    .heads = 12,
    .intermediate = 3072,
    .max_positions = 77,
    .eps = 1e-5,
    .act = .quick_gelu,
    .eos_id = 49407,
};

/// SDXL's second tower (OpenCLIP ViT-bigG/14). Present for when SDXL lands; the
/// only differences from `clip_l` are dimensions and the activation.
pub const clip_g: Config = .{
    .hidden = 1280,
    .layers = 32,
    .heads = 20,
    .intermediate = 5120,
    .max_positions = 77,
    .eps = 1e-5,
    .act = .gelu_erf,
    .eos_id = 49407,
};

const Layer = struct {
    ln1_w: []const f32,
    ln1_b: []const f32,
    q: Weight,
    q_b: []const f32,
    k: Weight,
    k_b: []const f32,
    v: Weight,
    v_b: []const f32,
    o: Weight,
    o_b: []const f32,
    ln2_w: []const f32,
    ln2_b: []const f32,
    fc1: Weight,
    fc1_b: []const f32,
    fc2: Weight,
    fc2_b: []const f32,
};

pub const TextEncoder = struct {
    arena: std.heap.ArenaAllocator,
    cfg: Config,
    /// The embedding table, kept as a view: `[vocab][hidden]`, read one row per
    /// token. Materializing it would cost 151 MB of f32 for CLIP-L to serve 77 rows.
    tok_embed: safetensors.TensorView,
    vocab: usize,
    pos_embed: []const f32,
    layers: []Layer,
    final_ln_w: []const f32,
    final_ln_b: []const f32,

    /// `prefix` is where the text model sits in the store. For an LDM single-file
    /// checkpoint that is `cond_stage_model.transformer.text_model.`; for a bare
    /// HF export, `text_model.`.
    pub fn load(gpa: std.mem.Allocator, store: WeightStore, cfg: Config, prefix: []const u8) !TextEncoder {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();

        const l: Loader = .{ .store = store, .alloc = alloc, .pfx = prefix };

        const emb_name = try std.fmt.allocPrint(alloc, "{s}embeddings.token_embedding.weight", .{prefix});
        const tok_view = store.get(emb_name) orelse {
            std.log.err("clip_text: missing {s}", .{emb_name});
            return error.MissingTensor;
        };
        const emb_shape = tok_view.info.shape.slice();
        if (emb_shape.len != 2 or emb_shape[1] != cfg.hidden) {
            std.log.err("clip_text: {s} has shape {any}, expected [vocab, {d}]", .{ emb_name, emb_shape, cfg.hidden });
            return error.ShapeMismatch;
        }

        const layers = try alloc.alloc(Layer, cfg.layers);
        for (layers, 0..) |*layer, i| {
            layer.* = .{
                .ln1_w = try l.vec("encoder.layers.{d}.layer_norm1.weight", .{i}, cfg.hidden),
                .ln1_b = try l.vec("encoder.layers.{d}.layer_norm1.bias", .{i}, cfg.hidden),
                .q = try l.mat("encoder.layers.{d}.self_attn.q_proj.weight", .{i}, cfg.hidden, cfg.hidden),
                .q_b = try l.vec("encoder.layers.{d}.self_attn.q_proj.bias", .{i}, cfg.hidden),
                .k = try l.mat("encoder.layers.{d}.self_attn.k_proj.weight", .{i}, cfg.hidden, cfg.hidden),
                .k_b = try l.vec("encoder.layers.{d}.self_attn.k_proj.bias", .{i}, cfg.hidden),
                .v = try l.mat("encoder.layers.{d}.self_attn.v_proj.weight", .{i}, cfg.hidden, cfg.hidden),
                .v_b = try l.vec("encoder.layers.{d}.self_attn.v_proj.bias", .{i}, cfg.hidden),
                .o = try l.mat("encoder.layers.{d}.self_attn.out_proj.weight", .{i}, cfg.hidden, cfg.hidden),
                .o_b = try l.vec("encoder.layers.{d}.self_attn.out_proj.bias", .{i}, cfg.hidden),
                .ln2_w = try l.vec("encoder.layers.{d}.layer_norm2.weight", .{i}, cfg.hidden),
                .ln2_b = try l.vec("encoder.layers.{d}.layer_norm2.bias", .{i}, cfg.hidden),
                .fc1 = try l.mat("encoder.layers.{d}.mlp.fc1.weight", .{i}, cfg.intermediate, cfg.hidden),
                .fc1_b = try l.vec("encoder.layers.{d}.mlp.fc1.bias", .{i}, cfg.intermediate),
                .fc2 = try l.mat("encoder.layers.{d}.mlp.fc2.weight", .{i}, cfg.hidden, cfg.intermediate),
                .fc2_b = try l.vec("encoder.layers.{d}.mlp.fc2.bias", .{i}, cfg.hidden),
            };
        }

        // ⚠️ Every allocation must happen BEFORE the struct literal. `.arena = arena`
        // copies the arena's state by value, so an allocation made by a *later* field
        // initializer lands in the local arena's chunk list and is invisible to the
        // copy the caller will `deinit()` — a leak the test allocator catches and
        // nothing else would. Same reason `dit.zig` builds its fields up front.
        const pos_embed = try l.vec("embeddings.position_embedding.weight", .{}, cfg.max_positions * cfg.hidden);
        const final_ln_w = try l.vec("final_layer_norm.weight", .{}, cfg.hidden);
        const final_ln_b = try l.vec("final_layer_norm.bias", .{}, cfg.hidden);

        return .{
            .arena = arena,
            .cfg = cfg,
            .tok_embed = tok_view,
            .vocab = emb_shape[0],
            .pos_embed = pos_embed,
            .layers = layers,
            .final_ln_w = final_ln_w,
            .final_ln_b = final_ln_b,
        };
    }

    pub fn deinit(self: *TextEncoder) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// `out` is `[ids.len][hidden]`, the final hidden state after
    /// `final_layer_norm` — what SD's cross-attention consumes.
    ///
    /// `hidden_out`, when given, receives the same-shaped activation at
    /// `capture_layer`: 0 = the embedding output, n = after encoder layer n−1. That
    /// is what makes a mismatch localize to a layer instead of just failing at the
    /// end, and it is also how "CLIP skip" would be implemented.
    pub fn forward(
        self: *const TextEncoder,
        io: std.Io,
        gpa: std.mem.Allocator,
        out: []f32,
        ids: []const u32,
        capture: ?Capture,
    ) !void {
        const cfg = self.cfg;
        const seq = ids.len;
        if (seq == 0 or seq > cfg.max_positions) return error.InvalidSequenceLength;
        std.debug.assert(out.len == seq * cfg.hidden);

        // Embedding lookup + learned positional table, straight into `out`, which is
        // the residual stream for the whole forward.
        const row_bytes = self.tok_embed.info.dtype.storageBytes(cfg.hidden);
        for (ids, 0..) |id, p| {
            if (id >= self.vocab) return error.TokenOutOfRange;
            const src = self.tok_embed.bytes[id * row_bytes ..][0..row_bytes];
            const dst = out[p * cfg.hidden ..][0..cfg.hidden];
            try safetensors.convertToF32(self.tok_embed.info.dtype, src, dst);
            for (dst, self.pos_embed[p * cfg.hidden ..][0..cfg.hidden]) |*o, pe| o.* += pe;
        }
        if (capture) |c| if (c.layer == 0) @memcpy(c.dst, out);

        const hd = cfg.headDim();
        const norm_buf = try gpa.alloc(f32, seq * cfg.hidden);
        defer gpa.free(norm_buf);
        const q = try gpa.alloc(f32, seq * cfg.hidden);
        defer gpa.free(q);
        const k = try gpa.alloc(f32, seq * cfg.hidden);
        defer gpa.free(k);
        const v = try gpa.alloc(f32, seq * cfg.hidden);
        defer gpa.free(v);
        const attn_out = try gpa.alloc(f32, seq * cfg.hidden);
        defer gpa.free(attn_out);
        const ff = try gpa.alloc(f32, seq * cfg.intermediate);
        defer gpa.free(ff);

        for (self.layers, 0..) |layer, li| {
            // --- attention block: x = x + o(attn(ln1(x))) ---
            ops.norm.layerNorm(norm_buf, out, layer.ln1_w, layer.ln1_b, cfg.eps);
            try ops.matmul.matmul(io, gpa, q, norm_buf, seq, layer.q, layer.q_b);
            try ops.matmul.matmul(io, gpa, k, norm_buf, seq, layer.k, layer.k_b);
            try ops.matmul.matmul(io, gpa, v, norm_buf, seq, layer.v, layer.v_b);
            try ops.attention.attention(io, gpa, attn_out, q, k, v, .{
                .seq_q = seq,
                .seq_kv = seq,
                .n_heads = cfg.heads,
                .n_kv_heads = cfg.heads,
                .head_dim = hd,
                // The one non-obvious flag in this file. CLIP's text tower is a
                // causal LM body used as an encoder.
                .causal = true,
            });
            try ops.matmul.matmul(io, gpa, norm_buf, attn_out, seq, layer.o, layer.o_b);
            for (out, norm_buf) |*o, r| o.* += r;

            // --- mlp block: x = x + fc2(act(fc1(ln2(x)))) ---
            ops.norm.layerNorm(norm_buf, out, layer.ln2_w, layer.ln2_b, cfg.eps);
            try ops.matmul.matmul(io, gpa, ff, norm_buf, seq, layer.fc1, layer.fc1_b);
            switch (cfg.act) {
                .quick_gelu => ops.act.geluQuick(ff),
                .gelu_erf => ops.act.geluErf(ff),
            }
            try ops.matmul.matmul(io, gpa, norm_buf, ff, seq, layer.fc2, layer.fc2_b);
            for (out, norm_buf) |*o, r| o.* += r;

            if (capture) |c| if (c.layer == li + 1) @memcpy(c.dst, out);
        }

        ops.norm.layerNorm(out, out, self.final_ln_w, self.final_ln_b, cfg.eps);
    }

    /// Which intermediate activation `forward` should copy out, and where to.
    pub const Capture = struct {
        /// 0 = embedding output, n = after encoder layer n−1.
        layer: usize,
        /// `[seq][hidden]`, caller-owned.
        dst: []f32,
    };

    /// The pooled vector SDXL conditions on: the hidden row at the **first**
    /// `eos_id`. HF selects it with `argmax(input_ids)`, which is equivalent only
    /// because `<|endoftext|>` is the highest id in SD's vocabularies and every
    /// padding slot holds it — so "first eos" is the well-defined form of the same
    /// choice. Returns null when the sequence has no eos (an unterminated prompt).
    pub fn pooled(self: *const TextEncoder, hidden: []const f32, ids: []const u32) ?[]const f32 {
        for (ids, 0..) |id, p| {
            if (id == self.cfg.eos_id) return hidden[p * self.cfg.hidden ..][0..self.cfg.hidden];
        }
        return null;
    }
};

const Loader = struct {
    store: WeightStore,
    alloc: std.mem.Allocator,
    pfx: []const u8,

    fn name(l: Loader, buf: []u8, comptime fmt: []const u8, args: anytype) ![]u8 {
        var fbs = std.Io.Writer.fixed(buf);
        try fbs.writeAll(l.pfx);
        try fbs.print(fmt, args);
        return fbs.buffered();
    }

    fn mat(l: Loader, comptime fmt: []const u8, args: anytype, rows: usize, cols: usize) !Weight {
        var buf: [200]u8 = undefined;
        const nm = try l.name(&buf, fmt, args);
        const view = l.store.get(nm) orelse {
            std.log.err("clip_text: missing {s}", .{nm});
            return error.MissingTensor;
        };
        const shape = view.info.shape.slice();
        if (shape.len != 2 or shape[0] != rows or shape[1] != cols) {
            std.log.err("clip_text: {s} has shape {any} ({t}), expected [{d}, {d}]", .{ nm, shape, view.info.dtype, rows, cols });
            return error.ShapeMismatch;
        }
        // Most checkpoints hand us a dtype the GEMM reads directly (f32/f16/bf16/fp8,
        // or a quantized text tower). The rest are materialized to f32 once, here:
        // this SD1.5 merge stores its CLIP linears as **f64**, which ComfyUI also
        // casts on load, and discovering that from inside the first forward would
        // report the problem as far from its cause as possible.
        var w = if (ops.matmul.supportsDType(view.info.dtype)) blk: {
            break :blk Weight.init(view.bytes, view.info.dtype, rows, cols);
        } else blk: {
            std.log.debug("clip_text: materializing {s} ({t}) to f32", .{ nm, view.info.dtype });
            const f32s = try view.toF32Alloc(l.alloc);
            break :blk Weight.fromF32(f32s, rows, cols);
        };
        // The checkpoint name travels with the weight so `ops.matmul.probe` can
        // attribute a GEMM to a layer — which is what makes the text tower
        // measurable at all.
        w.tag = try l.alloc.dupe(u8, nm);
        return w;
    }

    fn vec(l: Loader, comptime fmt: []const u8, args: anytype, len: usize) ![]f32 {
        var buf: [200]u8 = undefined;
        const nm = try l.name(&buf, fmt, args);
        const view = l.store.get(nm) orelse {
            std.log.err("clip_text: missing {s}", .{nm});
            return error.MissingTensor;
        };
        if (view.info.elemCount() != len) {
            std.log.err("clip_text: {s} has {d} elements, expected {d}", .{ nm, view.info.elemCount(), len });
            return error.ShapeMismatch;
        }
        return view.toF32Alloc(l.alloc);
    }
};

// --- tests -----------------------------------------------------------------

const testing = std.testing;
const test_gate = @import("../test_gate.zig");

/// The reference fixture (`tools/gen_sd15_fixtures.py`): a real SD1.5 checkpoint
/// driven through `transformers.CLIPTextModel`. Gated, because it needs the 5 GB
/// checkpoint the fixture was generated against.
const ref_path = "src/models/assets/sd15_ref.safetensors";
const sd15_ckpt = "/home/qt/genai/comfyui/models/checkpoints/sd1.5/perfectdeliberate_v20.safetensors";

test "CLIP-L matches transformers.CLIPTextModel on a real SD1.5 checkpoint" {
    const gpa = testing.allocator;
    const io = testing.io;
    try test_gate.requireIntegration();
    try test_gate.requireModelFile(io, sd15_ckpt);
    try test_gate.requireModelFile(io, ref_path);

    var ref = try safetensors.SafeTensors.open(gpa, io, ref_path);
    defer ref.deinit();
    var ck = try safetensors.SafeTensors.open(gpa, io, sd15_ckpt);
    defer ck.deinit();

    var enc = try TextEncoder.load(gpa, .{ .safetensors = &ck }, clip_l, "cond_stage_model.transformer.text_model.");
    defer enc.deinit();

    // Token ids for two prompts (one of them empty — the CFG branch), [2][77] i32.
    // Read as raw i32: `convertToF32` deliberately refuses integer dtypes, since
    // silently turning token ids into floats is never what a caller wants.
    const tokens_view = ref.get("clip.tokens").?;
    const tokens = std.mem.bytesAsSlice(i32, tokens_view.bytes);
    const expected = try ref.get("clip.hidden").?.toF32Alloc(gpa);
    defer gpa.free(expected);
    const expected_embed = try ref.get("clip.embed_out").?.toF32Alloc(gpa);
    defer gpa.free(expected_embed);
    const expected_l0 = try ref.get("clip.layer0").?.toF32Alloc(gpa);
    defer gpa.free(expected_l0);
    const expected_pooled = try ref.get("clip.pooled").?.toF32Alloc(gpa);
    defer gpa.free(expected_pooled);

    const seq = clip_l.max_positions;
    const hid = clip_l.hidden;
    const n_prompts = tokens.len / seq;

    const ids = try gpa.alloc(u32, seq);
    defer gpa.free(ids);
    const out = try gpa.alloc(f32, seq * hid);
    defer gpa.free(out);
    const cap = try gpa.alloc(f32, seq * hid);
    defer gpa.free(cap);

    for (0..n_prompts) |p| {
        for (ids, 0..) |*id, i| id.* = @intCast(tokens[p * seq + i]);

        // The embedding output first: a wrong positional table or a wrong lookup
        // shows up here, before any attention can smear it.
        try enc.forward(io, gpa, out, ids, .{ .layer = 0, .dst = cap });
        for (expected_embed[p * seq * hid ..][0 .. seq * hid], cap, 0..) |e, a, i| {
            errdefer std.debug.print("clip embed_out prompt {d} idx {d}: {d:.6} vs {d:.6}\n", .{ p, i, e, a });
            try testing.expectApproxEqAbs(e, a, 2e-4);
        }

        // Then layer 0, which is where a non-causal mask or the wrong activation
        // first becomes visible.
        try enc.forward(io, gpa, out, ids, .{ .layer = 1, .dst = cap });
        for (expected_l0[p * seq * hid ..][0 .. seq * hid], cap, 0..) |e, a, i| {
            errdefer std.debug.print("clip layer0 prompt {d} idx {d}: {d:.6} vs {d:.6}\n", .{ p, i, e, a });
            try testing.expectApproxEqAbs(e, a, 2e-3);
        }

        // And the whole tower. Tolerance is loose in absolute terms because CLIP's
        // final hidden states run to ~30 in magnitude; what matters is that it is
        // ~1e-4 relative across 59k values, not that it is 1e-6.
        try enc.forward(io, gpa, out, ids, null);
        var max_abs: f32 = 0;
        var max_rel: f32 = 0;
        for (expected[p * seq * hid ..][0 .. seq * hid], out) |e, a| {
            max_abs = @max(max_abs, @abs(e - a));
            const scale = @max(@abs(e), 1.0);
            max_rel = @max(max_rel, @abs(e - a) / scale);
        }
        errdefer std.debug.print("clip hidden prompt {d}: max_abs {d:.6} max_rel {d:.6}\n", .{ p, max_abs, max_rel });
        try testing.expect(max_rel < 5e-3);

        // The pooled row SDXL will need, selected by first-eos rather than argmax.
        const pooled = enc.pooled(out, ids).?;
        for (expected_pooled[p * hid ..][0..hid], pooled) |e, a| {
            try testing.expectApproxEqAbs(e, a, 5e-2);
        }
    }
}

test "the causal mask is what makes this an encoder and not a bidirectional one" {
    // A property test that needs no checkpoint: with causal attention, appending a
    // token cannot change the rows before it. If the mask were dropped, every row
    // would move — and the model would still produce plausible conditioning, which is
    // why this is worth pinning separately from the fixture.
    const gpa = testing.allocator;
    const io = testing.io;

    const cfg: Config = .{
        .hidden = 8, .layers = 2, .heads = 2, .intermediate = 16,
        .max_positions = 6, .eps = 1e-5, .act = .quick_gelu, .eos_id = 3,
    };
    var ckpt = try TinyCheckpoint.init(gpa, cfg, 16);
    defer ckpt.deinit(gpa);
    var enc = try TextEncoder.load(gpa, ckpt.store(), cfg, "");
    defer enc.deinit();

    const ids4 = [_]u32{ 1, 5, 9, 2 };
    const ids5 = [_]u32{ 1, 5, 9, 2, 7 };
    const out4 = try gpa.alloc(f32, ids4.len * cfg.hidden);
    defer gpa.free(out4);
    const out5 = try gpa.alloc(f32, ids5.len * cfg.hidden);
    defer gpa.free(out5);
    try enc.forward(io, gpa, out4, &ids4, null);
    try enc.forward(io, gpa, out5, &ids5, null);

    // final_layer_norm is per row, so the shared prefix must match exactly.
    for (out4, out5[0..out4.len], 0..) |a, b, i| {
        errdefer std.debug.print("causal prefix diverged at {d}: {d:.6} vs {d:.6}\n", .{ i, a, b });
        try testing.expectApproxEqAbs(a, b, 1e-5);
    }
}

/// A deterministic in-memory **safetensors** checkpoint for the small property
/// tests: every weight is a fixed pseudo-random f32, serialized into a caller-owned
/// buffer and opened through the real parser. Built as a real container rather than
/// a stub store because `WeightStore` is a closed union over the two formats we
/// actually read — and because a test that bypasses the parser would not catch a
/// loader that mis-reads shapes.
const TinyCheckpoint = struct {
    buf: []u8,
    st: safetensors.SafeTensors,

    fn init(gpa: std.mem.Allocator, cfg: Config, vocab: usize) !TinyCheckpoint {
        var names: std.ArrayList([]const u8) = .empty;
        defer {
            for (names.items) |n| gpa.free(n);
            names.deinit(gpa);
        }
        var shapes: std.ArrayList([2]usize) = .empty; // [rows, cols]; cols 0 = rank-1
        defer shapes.deinit(gpa);

        const add = struct {
            fn f(g: std.mem.Allocator, ns: *std.ArrayList([]const u8), ss: *std.ArrayList([2]usize), comptime fmt: []const u8, args: anytype, rows: usize, cols: usize) !void {
                try ns.append(g, try std.fmt.allocPrint(g, fmt, args));
                try ss.append(g, .{ rows, cols });
            }
        }.f;

        try add(gpa, &names, &shapes, "embeddings.token_embedding.weight", .{}, vocab, cfg.hidden);
        try add(gpa, &names, &shapes, "embeddings.position_embedding.weight", .{}, cfg.max_positions, cfg.hidden);
        for (0..cfg.layers) |i| {
            try add(gpa, &names, &shapes, "encoder.layers.{d}.layer_norm1.weight", .{i}, cfg.hidden, 0);
            try add(gpa, &names, &shapes, "encoder.layers.{d}.layer_norm1.bias", .{i}, cfg.hidden, 0);
            inline for (.{ "q_proj", "k_proj", "v_proj", "out_proj" }) |proj| {
                try add(gpa, &names, &shapes, "encoder.layers.{d}.self_attn." ++ proj ++ ".weight", .{i}, cfg.hidden, cfg.hidden);
                try add(gpa, &names, &shapes, "encoder.layers.{d}.self_attn." ++ proj ++ ".bias", .{i}, cfg.hidden, 0);
            }
            try add(gpa, &names, &shapes, "encoder.layers.{d}.layer_norm2.weight", .{i}, cfg.hidden, 0);
            try add(gpa, &names, &shapes, "encoder.layers.{d}.layer_norm2.bias", .{i}, cfg.hidden, 0);
            try add(gpa, &names, &shapes, "encoder.layers.{d}.mlp.fc1.weight", .{i}, cfg.intermediate, cfg.hidden);
            try add(gpa, &names, &shapes, "encoder.layers.{d}.mlp.fc1.bias", .{i}, cfg.intermediate, 0);
            try add(gpa, &names, &shapes, "encoder.layers.{d}.mlp.fc2.weight", .{i}, cfg.hidden, cfg.intermediate);
            try add(gpa, &names, &shapes, "encoder.layers.{d}.mlp.fc2.bias", .{i}, cfg.hidden, 0);
        }
        try add(gpa, &names, &shapes, "final_layer_norm.weight", .{}, cfg.hidden, 0);
        try add(gpa, &names, &shapes, "final_layer_norm.bias", .{}, cfg.hidden, 0);

        // Header first, so the data offsets are known before anything is written.
        var hdr: std.Io.Writer.Allocating = .init(gpa);
        defer hdr.deinit();
        try hdr.writer.writeByte('{');
        var offset: usize = 0;
        for (names.items, shapes.items, 0..) |nm, sh, idx| {
            const n = sh[0] * (if (sh[1] == 0) 1 else sh[1]);
            const bytes = n * @sizeOf(f32);
            if (idx > 0) try hdr.writer.writeByte(',');
            if (sh[1] == 0) {
                try hdr.writer.print("\"{s}\":{{\"dtype\":\"F32\",\"shape\":[{d}],\"data_offsets\":[{d},{d}]}}", .{ nm, sh[0], offset, offset + bytes });
            } else {
                try hdr.writer.print("\"{s}\":{{\"dtype\":\"F32\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}}", .{ nm, sh[0], sh[1], offset, offset + bytes });
            }
            offset += bytes;
        }
        try hdr.writer.writeByte('}');
        const header = hdr.written();

        const buf = try gpa.alloc(u8, 8 + header.len + offset);
        errdefer gpa.free(buf);
        std.mem.writeInt(u64, buf[0..8], header.len, .little);
        @memcpy(buf[8..][0..header.len], header);

        // Deterministic values: an LCG, so the fixture is the same on every machine.
        var seed: u64 = 0x5EED;
        const data = std.mem.bytesAsSlice(f32, buf[8 + header.len ..]);
        for (data) |*x| {
            seed = seed *% 6364136223846793005 +% 1442695040888963407;
            x.* = (@as(f32, @floatFromInt(@as(u32, @truncate(seed >> 33)))) / 2147483648.0 - 1.0) * 0.2;
        }

        return .{ .buf = buf, .st = try safetensors.SafeTensors.initFromSlice(gpa, buf) };
    }

    fn deinit(self: *TinyCheckpoint, gpa: std.mem.Allocator) void {
        self.st.deinit();
        gpa.free(self.buf);
        self.* = undefined;
    }

    fn store(self: *const TinyCheckpoint) WeightStore {
        return .{ .safetensors = &self.st };
    }
};
