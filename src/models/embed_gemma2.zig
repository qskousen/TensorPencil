//! EmbeddingGemma 2 (google/embeddinggemma-2), text path: text to one
//! L2-normalized 768-d embedding. Ported from transformers'
//! `modeling_embedding_gemma2.py`; `tools/gen_embeddinggemma2.py` runs the
//! sentence-transformers pipeline for the fixture.
//!
//! The body is a Gemma 4 layer run bidirectionally (`transformer.layerForward*`
//! `.fresh`): sandwich norms, per-head QK norms, a weightless V norm, attention
//! scale 1.0, tanh GeGLU. Every 6th layer is global (4 heads of 512 over one KV
//! head, rope theta 1e6); the rest are local (4 heads of 256 over two KV heads,
//! theta 1e4) and see only keys within 512 positions either side. What is new:
//!
//!   - Per-layer inputs (PLE) come from a projection of the input embeddings
//!     alone, `rms_norm(W x / sqrt(hidden))` split into one 512-wide slice per
//!     layer, with no per-layer token table. They are the input embeddings, so an
//!     image's soft tokens feed them too.
//!   - Each layer ends with a third residual block, `x += rms_norm(P (gelu(G x) *
//!     ple[l]))`, and then multiplies the whole stream by `layer_scalar`.
//!   - Norm weights multiply as stored; unlike Gemma 3 there is no implicit +1.
//!
//! The head is the final norm, a 512->768 projection (applied to the mean, which
//! is the same linear map as projecting each row), and an L2 normalize. The
//! caller frames ids `<bos> ... <eos>` and adds the task prompt.
//!
//! An image goes through the Gemma 4 vision tower (`gemma4v_vit`, loaded from the
//! same file) to 512-wide soft tokens, which run through the same text body as
//! `<bos> <|image> soft... <image|> <eos>`. They enter unscaled, where a text
//! token's embedding is scaled. Images take no prompt.
//!
//! The embedding scale is sqrt(hidden) in f32. transformers rounds it to the
//! weight dtype first (22.625 under bf16), so a bf16 reference run differs here.

const std = @import("std");
const tp_core = @import("tp_core");
const ops = @import("tp_ops");
const loader = @import("loader.zig");
const transformer = @import("transformer.zig");
const qwen3 = @import("qwen3.zig");
const gemma4v_vit = @import("gemma4v_vit.zig");

const SafeTensors = tp_core.safetensors.SafeTensors;
const WeightStore = tp_core.weights.WeightStore;
const Weight = ops.matmul.Weight;

pub const n_layers = 24;
pub const hidden = 512;
pub const n_heads = 4;
pub const intermediate = 2048;
pub const ple_dim = 512;
pub const vocab = 262144;
pub const embed_dim = 768;
pub const bos = 2;
pub const eos = 1;
pub const boi = 255999;
pub const eoi = 258882;

/// An interleaved RGB8 image of any size; the tower resizes it to its token budget.
pub const Rgb = struct { pixels: []const u8, width: usize, height: usize };
pub const rms_eps: f32 = 1e-6;
pub const window = 512;

pub const Geometry = struct {
    head_dim: usize,
    n_kv: usize,
    theta: f64,
    window: usize,

    pub fn qDim(g: Geometry) usize {
        return n_heads * g.head_dim;
    }
    pub fn kvDim(g: Geometry) usize {
        return g.n_kv * g.head_dim;
    }
};

pub const local: Geometry = .{ .head_dim = 256, .n_kv = 2, .theta = 10_000.0, .window = window };
pub const global: Geometry = .{ .head_dim = 512, .n_kv = 1, .theta = 1_000_000.0, .window = 0 };
pub const max_q_dim = n_heads * 512;
pub const max_kv_dim = 512;

pub fn isGlobal(l: usize) bool {
    return l % 6 == 5;
}

pub fn geometry(l: usize) Geometry {
    return if (isGlobal(l)) global else local;
}

pub fn dims(l: usize) transformer.Dims {
    const g = geometry(l);
    return .{
        .hidden = hidden,
        .n_heads = n_heads,
        .n_kv = g.n_kv,
        .head_dim = g.head_dim,
        .q_dim = g.qDim(),
        .kv_dim = g.kvDim(),
        .intermediate = intermediate,
        .sliding_window = g.window,
    };
}

pub const embed_scale: f32 = @sqrt(@as(f32, hidden));
const ple_scale: f32 = 1.0 / @sqrt(@as(f32, hidden));

const spec: transformer.LayerSpec = .{
    .activation = .gelu_tanh_mul,
    .sandwich_norms = true,
    .v_norm_unit = true,
    .attn_scale = 1.0,
};

/// Fields match what `spec` reads, plus the PLE block and the layer scalar.
pub const Layer = struct {
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
    ple_gate: Weight, // [ple_dim, hidden]
    ple_proj: Weight, // [hidden, ple_dim]
    ple_norm: []const f32,
    scalar: f32,
};

const Scratch = struct {
    normed: []f32,
    tmp: []f32,
    q: []f32,
    k: []f32,
    v: []f32,
    attn_out: []f32,
    gate: []f32,
    up: []f32,

    fn init(gpa: std.mem.Allocator, rows: usize) !Scratch {
        var s: Scratch = undefined;
        const widths = .{ hidden, hidden, max_q_dim, max_kv_dim, max_kv_dim, max_q_dim, intermediate, intermediate };
        inline for (@typeInfo(Scratch).@"struct".field_names, 0..) |name, i| {
            @field(s, name) = gpa.alloc(f32, rows * widths[i]) catch |e| {
                inline for (@typeInfo(Scratch).@"struct".field_names[0..i]) |prev| gpa.free(@field(s, prev));
                return e;
            };
        }
        return s;
    }

    fn deinit(s: *Scratch, gpa: std.mem.Allocator) void {
        inline for (@typeInfo(Scratch).@"struct".field_names) |name| gpa.free(@field(s, name));
    }
};

pub const Model = struct {
    arena: std.heap.ArenaAllocator,
    st: SafeTensors,
    embed_w: Weight,
    ple_model_proj: Weight, // [n_layers * ple_dim, hidden]
    ple_proj_norm: []const f32,
    layers: []Layer,
    final_norm: []const f32,
    out_proj: Weight, // [embed_dim, hidden]
    vit: ?gemma4v_vit.Vit,

    pub const Options = struct { vision: bool = false };

    /// `dir` holds the checkpoint's `model.safetensors`. The vision tower loads
    /// with `opts.vision`; the audio tower is not read.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, opts: Options) !Model {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();

        var pbuf: [1024]u8 = undefined;
        var st = try SafeTensors.open(gpa, io, try std.mem.print(&pbuf, "{s}/model.safetensors", .{dir}));
        errdefer st.deinit();
        const store: WeightStore = .{ .safetensors = &st };

        const p = "language_model.layers.";
        const layers = try a.alloc(Layer, n_layers);
        for (layers, 0..) |*layer, l| {
            const g = geometry(l);
            layer.* = .{
                .input_norm = try loader.indexedVector(a, store, p, l, "input_layernorm.weight", hidden),
                .q = try loader.indexedMatrix(store, p, l, "self_attn.q_proj.weight", g.qDim(), hidden),
                .k = try loader.indexedMatrix(store, p, l, "self_attn.k_proj.weight", g.kvDim(), hidden),
                .v = try loader.indexedMatrix(store, p, l, "self_attn.v_proj.weight", g.kvDim(), hidden),
                .o = try loader.indexedMatrix(store, p, l, "self_attn.o_proj.weight", hidden, g.qDim()),
                .q_norm = try loader.indexedVector(a, store, p, l, "self_attn.q_norm.weight", g.head_dim),
                .k_norm = try loader.indexedVector(a, store, p, l, "self_attn.k_norm.weight", g.head_dim),
                .post_attn_norm = try loader.indexedVector(a, store, p, l, "post_attention_layernorm.weight", hidden),
                .pre_ffn_norm = try loader.indexedVector(a, store, p, l, "pre_feedforward_layernorm.weight", hidden),
                .post_ffn_norm = try loader.indexedVector(a, store, p, l, "post_feedforward_layernorm.weight", hidden),
                .gate = try loader.indexedMatrix(store, p, l, "mlp.gate_proj.weight", intermediate, hidden),
                .up = try loader.indexedMatrix(store, p, l, "mlp.up_proj.weight", intermediate, hidden),
                .down = try loader.indexedMatrix(store, p, l, "mlp.down_proj.weight", hidden, intermediate),
                .ple_gate = try loader.indexedMatrix(store, p, l, "ple_block.per_layer_input_gate.weight", ple_dim, hidden),
                .ple_proj = try loader.indexedMatrix(store, p, l, "ple_block.per_layer_projection.weight", hidden, ple_dim),
                .ple_norm = try loader.indexedVector(a, store, p, l, "ple_block.post_per_layer_input_norm.weight", hidden),
                .scalar = (try loader.indexedVector(a, store, p, l, "layer_scalar", 1))[0],
            };
        }

        const ple_proj_norm = try loader.vector(a, store, "language_model.ple.per_layer_projection_norm.weight", ple_dim);
        const final_norm = try loader.vector(a, store, "language_model.norm.weight", hidden);
        var vit: ?gemma4v_vit.Vit = null;
        if (opts.vision) {
            vit = try gemma4v_vit.Vit.loadSafetensors(gpa, store, "vision_tower.", "embed_vision.embedding_projection.weight");
            if (vit.?.cfg.proj_dim != hidden) return error.ShapeMismatch;
        }
        errdefer if (vit) |*v| v.deinit();
        return .{
            .arena = arena,
            .st = st,
            .embed_w = try loader.matrix(store, "language_model.embed_tokens.weight", vocab, hidden),
            .ple_model_proj = try loader.matrix(store, "language_model.ple.per_layer_model_projection.weight", n_layers * ple_dim, hidden),
            .ple_proj_norm = ple_proj_norm,
            .layers = layers,
            .final_norm = final_norm,
            .out_proj = try loader.matrix(store, "language_model.embedding_projection.weight", embed_dim, hidden),
            .vit = vit,
        };
    }

    pub fn deinit(self: *Model) void {
        if (self.vit) |*v| v.deinit();
        self.st.deinit();
        self.arena.deinit();
        self.* = undefined;
    }

    /// Scaled token embeddings of `ids` into `x` [ids.len * hidden].
    pub fn embedTokens(self: *const Model, ids: []const u32, x: []f32) !void {
        try qwen3.embedTokens(self.embed_w, ids, x);
        for (x) |*v| v.* *= embed_scale;
    }

    /// The per-layer inputs of the input embeddings `x` [rows * hidden], layer-major
    /// into `out` [n_layers * rows * ple_dim]. `tmp` is [rows * n_layers * ple_dim].
    pub fn perLayerInputs(self: *const Model, io: std.Io, gpa: std.mem.Allocator, x: []const f32, rows: usize, tmp: []f32, out: []f32) !void {
        const w = n_layers * ple_dim;
        try ops.matmul.matmul(io, gpa, tmp, x, rows, self.ple_model_proj, null);
        for (tmp) |*v| v.* *= ple_scale;
        ops.norm.rmsNorm(tmp, tmp, self.ple_proj_norm, rms_eps);
        for (0..rows) |r| for (0..n_layers) |l| {
            @memcpy(out[(l * rows + r) * ple_dim ..][0..ple_dim], tmp[r * w + l * ple_dim ..][0..ple_dim]);
        };
    }

    /// Run the body over B items packed back to back in `x` (item i is rows
    /// `[row_off[i], row_off[i + 1])`), each attending only itself. `x` holds the
    /// input embeddings and is left holding the final-normed hidden states.
    pub fn encodeRows(self: *const Model, io: std.Io, gpa: std.mem.Allocator, x: []f32, row_off: []const usize) !void {
        const b = row_off.len - 1;
        const rows = row_off[b];
        std.debug.assert(x.len == rows * hidden);
        var longest: usize = 0;
        for (0..b) |i| longest = @max(longest, row_off[i + 1] - row_off[i]);

        const ple = try gpa.alloc(f32, n_layers * rows * ple_dim);
        defer gpa.free(ple);
        {
            const tmp = try gpa.alloc(f32, rows * n_layers * ple_dim);
            defer gpa.free(tmp);
            try self.perLayerInputs(io, gpa, x, rows, tmp, ple);
        }

        var fl = try ops.rope.rotateHalfFreqs(gpa, longest, local.head_dim, local.theta);
        defer fl.deinit(gpa);
        var fg = try ops.rope.rotateHalfFreqs(gpa, longest, global.head_dim, global.theta);
        defer fg.deinit(gpa);

        var s = try Scratch.init(gpa, rows);
        defer s.deinit(gpa);
        const g = s.gate[0 .. rows * ple_dim];
        const t = s.tmp[0 .. rows * hidden];

        for (self.layers, 0..) |*layer, l| {
            const freqs = if (isGlobal(l)) fg else fl;
            try transformer.layerForwardBatchedFresh(spec, io, gpa, layer, x, row_off, dims(l), freqs, rms_eps, true, &s);

            try ops.matmul.matmul(io, gpa, g, x, rows, layer.ple_gate, null);
            ops.act.geluTanhMul(g, ple[l * rows * ple_dim ..][0 .. rows * ple_dim]);
            try ops.matmul.matmul(io, gpa, t, g, rows, layer.ple_proj, null);
            ops.norm.rmsNorm(t, t, layer.ple_norm, rms_eps);
            for (x, t) |*xi, ti| xi.* = (xi.* + ti) * layer.scalar;
        }
        ops.norm.rmsNorm(x, x, self.final_norm, rms_eps);
    }

    /// Mean-pool the final-normed hidden states `lhs` [seq * hidden], project to
    /// 768 and L2-normalize into `out`. Shared with the device forwards.
    pub fn head(self: *const Model, io: std.Io, gpa: std.mem.Allocator, lhs: []const f32, out: []f32) !void {
        std.debug.assert(out.len == embed_dim and lhs.len % hidden == 0);
        const seq = lhs.len / hidden;
        var pooled: [hidden]f32 = @splat(0);
        for (0..seq) |r| for (&pooled, lhs[r * hidden ..][0..hidden]) |*p, v| {
            p.* += v;
        };
        const inv: f32 = 1.0 / @as(f32, @floatFromInt(seq));
        for (&pooled) |*p| p.* *= inv;
        try ops.matmul.matmul(io, gpa, out, &pooled, 1, self.out_proj, null);
        normalize(out);
    }

    /// `ids_list[i]` (framed `<bos> ... <eos>`) -> `outs[i]` [embed_dim], one
    /// forward over the whole batch.
    pub fn embedBatch(self: *const Model, io: std.Io, gpa: std.mem.Allocator, ids_list: []const []const u32, outs: [][]f32) !void {
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const rows = try self.tokenRows(arena.allocator(), ids_list);
        try self.embedRows(io, gpa, rows, outs);
    }

    pub fn embed(self: *const Model, io: std.Io, gpa: std.mem.Allocator, ids: []const u32, out: []f32) !void {
        var outs = [_][]f32{out};
        try self.embedBatch(io, gpa, &.{ids}, &outs);
    }

    /// The input rows of each id list, allocated with `a`.
    pub fn tokenRows(self: *const Model, a: std.mem.Allocator, ids_list: []const []const u32) ![]const []const f32 {
        const rows = try a.alloc([]const f32, ids_list.len);
        for (ids_list, rows) |ids, *r| {
            std.debug.assert(ids.len > 0);
            const x = try a.alloc(f32, ids.len * hidden);
            try self.embedTokens(ids, x);
            r.* = x;
        }
        return rows;
    }

    /// Items given as their input rows (`rows_list[i]` [n_i * hidden], from
    /// `tokenRows` or `imageRows`) -> `outs[i]` [embed_dim], one forward over all.
    pub fn embedRows(self: *const Model, io: std.Io, gpa: std.mem.Allocator, rows_list: []const []const f32, outs: [][]f32) !void {
        std.debug.assert(outs.len == rows_list.len and rows_list.len > 0);
        const row_off = try gpa.alloc(usize, rows_list.len + 1);
        defer gpa.free(row_off);
        row_off[0] = 0;
        for (rows_list, 0..) |r, i| row_off[i + 1] = row_off[i] + r.len / hidden;
        const x = try gpa.alloc(f32, row_off[rows_list.len] * hidden);
        defer gpa.free(x);
        for (rows_list, 0..) |r, i| @memcpy(x[row_off[i] * hidden ..][0..r.len], r);
        try self.encodeRows(io, gpa, x, row_off);
        for (outs, 0..) |out, i| try self.head(io, gpa, x[row_off[i] * hidden .. row_off[i + 1] * hidden], out);
    }

    /// The input rows of an image whose projected soft tokens are `soft`
    /// [n * hidden]: the frame's scaled token embeddings around them. Caller frees.
    pub fn imageRows(self: *const Model, gpa: std.mem.Allocator, soft: []const f32) ![]f32 {
        std.debug.assert(soft.len % hidden == 0);
        const n = soft.len / hidden;
        const x = try gpa.alloc(f32, (n + 4) * hidden);
        errdefer gpa.free(x);
        try self.embedTokens(&.{ bos, boi }, x[0 .. 2 * hidden]);
        @memcpy(x[2 * hidden ..][0..soft.len], soft);
        try self.embedTokens(&.{ eoi, eos }, x[(n + 2) * hidden ..]);
        return x;
    }

    /// `images[i]` -> `outs[i]` [embed_dim]: the tower per image, then one text
    /// forward over all of them. Needs the model opened with vision.
    pub fn embedImageBatch(self: *const Model, io: std.Io, gpa: std.mem.Allocator, images: []const Rgb, outs: [][]f32) !void {
        const vit = &(self.vit orelse return error.NoVisionTower);
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const rows = try arena.allocator().alloc([]const f32, images.len);
        for (images, rows) |img, *r| {
            var enc = try vit.encode(io, gpa, img.pixels, img.width, img.height);
            defer enc.deinit(gpa);
            r.* = try self.imageRows(arena.allocator(), enc.embeds);
        }
        try self.embedRows(io, gpa, rows, outs);
    }

    pub fn embedImage(self: *const Model, io: std.Io, gpa: std.mem.Allocator, img: Rgb, out: []f32) !void {
        var outs = [_][]f32{out};
        try self.embedImageBatch(io, gpa, &.{img}, &outs);
    }
};

pub fn normalize(v: []f32) void {
    var ss: f32 = 0;
    for (v) |e| ss += e * e;
    if (ss == 0) return;
    const inv = 1.0 / @sqrt(ss);
    for (v) |*e| e.* *= inv;
}

// --- tests -----------------------------------------------------------------

pub const test_dir = "../DiffKeep/Models/embeddinggemma-2";
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

/// The fixture's cases, parsed into `arena`. `prompt` is the sentence-transformers
/// prompt name (`query`, `document`) or null for raw text.
pub const RefCase = struct { name: []const u8, prompt: ?[]const u8, text: []const u8, ids: []u32, vec: []f32, nowindow: ?[]f32 };

pub fn loadReference(arena: std.mem.Allocator, io: std.Io) ![]RefCase {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, "testdata/embeddinggemma2_ref_vectors.json", arena, .limited(16 * 1024 * 1024));
    const root = try std.json.parseFromSliceLeaky(std.json.Value, arena, bytes, .{});
    var cases: std.ArrayList(RefCase) = .empty;
    var it = root.object.iterator();
    while (it.next()) |e| {
        const o = e.value_ptr.object;
        const ids_j = o.get("ids").?.array.items;
        const ids = try arena.alloc(u32, ids_j.len);
        for (ids, ids_j) |*d, v| d.* = @intCast(v.integer);
        const nw: ?[]f32 = if (o.get("vec_nowindow")) |v| try floats(arena, v.array.items) else null;
        const prompt: ?[]const u8 = switch (o.get("prompt").?) {
            .string => |str| str,
            else => null,
        };
        try cases.append(arena, .{
            .name = e.key_ptr.*,
            .prompt = prompt,
            .text = o.get("text").?.string,
            .ids = ids,
            .vec = try floats(arena, o.get("vec").?.array.items),
            .nowindow = nw,
        });
    }
    return cases.items;
}

fn floats(arena: std.mem.Allocator, items: []const std.json.Value) ![]f32 {
    const out = try arena.alloc(f32, items.len);
    for (out, items) |*d, v| d.* = @floatCast(v.float);
    return out;
}

test "embeddinggemma2 matches sentence-transformers" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    try test_gate.requireModelFile(io, test_dir ++ "/model.safetensors");
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const cases = try loadReference(arena.allocator(), io);

    var model = try Model.open(gpa, io, test_dir, .{});
    defer model.deinit();

    var out: [embed_dim]f32 = undefined;
    var windowed = false;
    for (cases) |c| {
        try model.embed(io, gpa, c.ids, &out);
        const cos = cosine(&out, c.vec);
        errdefer std.debug.print("{s}: cosine {d}\n", .{ c.name, cos });
        try std.testing.expect(cos >= 0.99999);
        if (c.nowindow) |nw| {
            windowed = true;
            const cos_nw = cosine(&out, nw);
            errdefer std.debug.print("{s}: cosine to the unwindowed vector {d}\n", .{ c.name, cos_nw });
            try std.testing.expect(cos_nw < 0.99);
        }
    }
    try std.testing.expect(windowed);
}

test "embeddinggemma2 embedBatch matches per-item" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    try test_gate.requireModelFile(io, test_dir ++ "/model.safetensors");
    var model = try Model.open(gpa, io, test_dir, .{});
    defer model.deinit();

    const ids_list = [_][]const u32{ &.{ 2, 1000, 2049, 1 }, &.{ 2, 500, 600, 700, 800, 1 }, &.{ 2, 42, 1 } };
    var single: [3][embed_dim]f32 = undefined;
    for (ids_list, 0..) |ids, i| try model.embed(io, gpa, ids, &single[i]);
    var batched: [3][embed_dim]f32 = undefined;
    var outs: [3][]f32 = .{ &batched[0], &batched[1], &batched[2] };
    try model.embedBatch(io, gpa, &ids_list, &outs);
    for (0..3) |i| {
        var maxad: f32 = 0;
        for (single[i], batched[i]) |a, b| maxad = @max(maxad, @abs(a - b));
        errdefer std.debug.print("item {d}: max abs diff {d}\n", .{ i, maxad });
        try std.testing.expect(maxad < 1e-5);
    }
}

const image_ref = "testdata/embeddinggemma2_image_ref.safetensors";

// The tower fed transformers' own resized pixels (so the resize is out of it), its
// soft tokens per token, the frame, and the vector.
test "embeddinggemma2 vision tower matches transformers" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    try test_gate.requireModelFile(io, test_dir ++ "/model.safetensors");
    var ref = try SafeTensors.open(gpa, io, image_ref);
    defer ref.deinit();
    const rs: WeightStore = .{ .safetensors = &ref };

    var model = try Model.open(gpa, io, test_dir, .{ .vision = true });
    defer model.deinit();
    const vit = &model.vit.?;
    vit.cfg.max_tokens = 70;

    const px = try ref.require("small.pixels");
    const shape = px.info.shape.slice();
    var enc = try vit.encode(io, gpa, px.bytes, shape[1], shape[0]);
    defer enc.deinit(gpa);
    const n = enc.embeds.len / hidden;
    const want = try loader.vector(gpa, rs, "small.soft", n * hidden);
    defer gpa.free(want);
    var worst: f32 = 1;
    for (0..n) |t| worst = @min(worst, cosine(enc.embeds[t * hidden ..][0..hidden], want[t * hidden ..][0..hidden]));
    errdefer std.debug.print("soft tokens: worst cosine {d}\n", .{worst});
    try std.testing.expect(worst >= 0.9999);

    const ids = std.mem.bytesAsSlice(i32, @as([]align(1) const u8, (try ref.require("small.ids")).bytes));
    try std.testing.expectEqual(n + 4, ids.len);
    try std.testing.expectEqual(@as(i32, bos), ids[0]);
    try std.testing.expectEqual(@as(i32, boi), ids[1]);
    try std.testing.expectEqual(@as(i32, eoi), ids[n + 2]);
    try std.testing.expectEqual(@as(i32, eos), ids[n + 3]);

    const x = try model.imageRows(gpa, enc.embeds);
    defer gpa.free(x);
    try model.encodeRows(io, gpa, x, &.{ 0, n + 4 });
    var out: [embed_dim]f32 = undefined;
    try model.head(io, gpa, x, &out);
    const vec = try loader.vector(gpa, rs, "small.vec", embed_dim);
    defer gpa.free(vec);
    const cos = cosine(&out, vec);
    errdefer std.debug.print("small: vector cosine {d}\n", .{cos});
    try std.testing.expect(cos >= 0.99999);
}

// PNG to vector through `embedImage`, resize included, at the default budget.
test "embeddinggemma2 image vectors match sentence-transformers" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    try test_gate.requireModelFile(io, test_dir ++ "/model.safetensors");
    var ref = try SafeTensors.open(gpa, io, image_ref);
    defer ref.deinit();
    const rs: WeightStore = .{ .safetensors = &ref };
    var model = try Model.open(gpa, io, test_dir, .{ .vision = true });
    defer model.deinit();

    for ([_][]const u8{ "down", "up", "tall" }) |name| {
        var pbuf: [128]u8 = undefined;
        const png = try std.Io.Dir.cwd().readFileAlloc(io, try std.mem.print(&pbuf, "testdata/embeddinggemma2_{s}.png", .{name}), gpa, .limited(8 << 20));
        defer gpa.free(png);
        const img = try tp_core.image.decodePngRgb(gpa, png);
        defer gpa.free(img.pixels);
        var out: [embed_dim]f32 = undefined;
        try model.embedImage(io, gpa, .{ .pixels = img.pixels, .width = img.width, .height = img.height }, &out);
        const want = try loader.vector(gpa, rs, try std.mem.print(&pbuf, "vec.{s}", .{name}), embed_dim);
        defer gpa.free(want);
        const cos = cosine(&out, want);
        errdefer std.debug.print("{s}: cosine {d}\n", .{ name, cos });
        try std.testing.expect(cos >= 0.9999);
    }
}
