//! `tp.embed`, the high-level embedding façade DiffKeep calls. It hides the
//! per-model tokenizer, role prefixes, frame tokens, and pooling behind three
//! types that each return an L2-normalized 768-d vector:
//!   - `TextEncoder`, text -> vec (Snowflake / EmbeddingGemma 1 and 2 / SigLIP2 text).
//!   - `ImageEncoder`, preprocessed RGB -> vec (SigLIP2 visual).
//!   - `MultimodalEncoder`, text or any-size RGB8 -> vec in one space (EmbeddingGemma 2).
//!
//! A GPU encoder keeps its weights resident and drops them in `deinit`; the device
//! caches key on host addresses, which the model's own `deinit` then frees.
//!
//! The underlying model forwards (`models.embed_*`) are validated against their
//! references (ONNX, or sentence-transformers for EmbeddingGemma 2); this layer
//! only adds the text-side glue: Snowflake's `"query: "` asymmetry, the
//! EmbeddingGemma task prefixes, SigLIP2's fixed 64-token frame.

const std = @import("std");
const models = @import("tp_models").models;
const tokenizer = @import("tp_core").tokenizer;
const tp_gpu = @import("tp_gpu");

const Tokenizer = tokenizer.Tokenizer;
const test_gate = @import("tp_models").test_gate;

/// Output dimension of every encoder here (all towers are 768-d).
pub const dim: usize = 768;

/// Compute backend. CPU always works; the GPU variants run the transformer body
/// device-side (see `models.embed_*_gpu`/`_cuda`) and require a live handle. The
/// caller owns the handle and must keep it alive for the encoder's lifetime.
pub const Backend = union(enum) {
    cpu,
    vulkan: *tp_gpu.context.Context,
    cuda: *tp_gpu.cuda.Backend,
};

/// Which text encoder to load. `arctic_embed_m_v2`, `embeddinggemma` and
/// `embeddinggemma2` are the candidates for the prompt/query text space;
/// `siglip2_text` is the cross-modal (image-space) text query encoder.
pub const TextModelKind = enum { arctic_embed_m_v2, embeddinggemma, embeddinggemma2, siglip2_text };

/// Snowflake/EmbeddingGemma are asymmetric (queries and documents get different
/// prefixes); SigLIP2 text is symmetric (role ignored).
pub const Role = enum { query, document };

pub const Options = struct { role: Role = .query };

const TextImpl = union(TextModelKind) {
    arctic_embed_m_v2: models.embed_snowflake.Model,
    embeddinggemma: models.embed_gemma.Model,
    embeddinggemma2: models.embed_gemma2.Model,
    siglip2_text: models.embed_siglip.TextModel,
};

pub const TextEncoder = struct {
    io: std.Io,
    tok: Tokenizer,
    impl: TextImpl,
    backend: Backend,

    /// Load a text encoder from `dir` (must hold the model's safetensors +
    /// `tokenizer.json`). `io`/allocator lifetimes must outlive the encoder.
    /// `backend` selects CPU / Vulkan / CUDA (the GPU handle must outlive it).
    pub fn open(gpa: std.mem.Allocator, io: std.Io, kind: TextModelKind, dir: []const u8, backend: Backend) !TextEncoder {
        return openWith(gpa, io, kind, dir, backend, false);
    }

    fn openWith(gpa: std.mem.Allocator, io: std.Io, kind: TextModelKind, dir: []const u8, backend: Backend, vision: bool) !TextEncoder {
        var pbuf: [1024]u8 = undefined;
        const tok_path = try std.mem.print(&pbuf, "{s}/tokenizer.json", .{dir});
        const tok_bytes = try std.Io.Dir.cwd().readFileAlloc(io, tok_path, gpa, .limited(64 * 1024 * 1024));
        defer gpa.free(tok_bytes);

        var tok: Tokenizer = switch (kind) {
            .arctic_embed_m_v2 => try Tokenizer.initUnigramFromTokenizerJson(gpa, tok_bytes),
            .embeddinggemma, .embeddinggemma2, .siglip2_text => try Tokenizer.initGemma4FromTokenizerJson(gpa, tok_bytes),
        };
        errdefer tok.deinit();

        const impl: TextImpl = switch (kind) {
            .arctic_embed_m_v2 => .{ .arctic_embed_m_v2 = try models.embed_snowflake.Model.open(gpa, io, dir) },
            .embeddinggemma => .{ .embeddinggemma = try models.embed_gemma.Model.open(gpa, io, dir) },
            .embeddinggemma2 => .{ .embeddinggemma2 = try models.embed_gemma2.Model.open(gpa, io, dir, .{ .vision = vision }) },
            .siglip2_text => .{ .siglip2_text = try models.embed_siglip.TextModel.open(gpa, io, dir) },
        };
        return .{ .io = io, .tok = tok, .impl = impl, .backend = backend };
    }

    pub fn deinit(self: *TextEncoder) void {
        switch (self.impl) {
            inline else => |*m| {
                evict(self.backend, m);
                m.deinit();
            },
        }
        self.tok.deinit();
        self.* = undefined;
    }

    /// Encode `text` into a freshly allocated L2-normalized 768-d vector (caller
    /// owns). Applies the model's role prefix, tokenizes, frames, and pools.
    pub fn embedText(self: *const TextEncoder, gpa: std.mem.Allocator, text: []const u8, opts: Options) ![]f32 {
        const out = try gpa.alloc(f32, dim);
        errdefer gpa.free(out);
        try self.embedTextInto(gpa, text, opts, out);
        return out;
    }

    /// Batch variant: one vector per input text (DiffKeep batches ~8 at index
    /// time). Runs a single fused forward over the whole batch on the selected
    /// backend (CPU / Vulkan / CUDA), the passed slice IS the "configurable
    /// amount". Returns caller-owned vectors + slice.
    pub fn embedTextBatch(self: *const TextEncoder, gpa: std.mem.Allocator, texts: []const []const u8, opts: Options) ![][]f32 {
        const vecs = try gpa.alloc([]f32, texts.len);
        errdefer gpa.free(vecs);
        for (vecs) |*v| v.* = &.{};
        errdefer for (vecs) |v| if (v.len != 0) gpa.free(v);
        for (vecs) |*v| v.* = try gpa.alloc(f32, dim);

        // B=1 has no cross-item amortization, take the single path (avoids the
        // batched forward's per-item offset bookkeeping for the query case).
        if (texts.len == 1) {
            try self.embedTextInto(gpa, texts[0], opts, vecs[0]);
            return vecs;
        }

        // Frame every text into its own id list (arena-scoped), then run one
        // fused forward over the whole batch on the selected backend.
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const framed = try a.alloc([]const u32, texts.len);
        for (texts, framed) |t, *f| f.* = try self.frameText(a, t, opts);

        switch (self.impl) {
            .arctic_embed_m_v2 => |*m| switch (self.backend) {
                .cpu => try m.embedBatch(self.io, gpa, framed, vecs),
                .vulkan => |ctx| try models.embed_snowflake_gpu.ModelGpu.init(m).embedBatch(ctx, self.io, gpa, framed, vecs),
                .cuda => |be| try models.embed_snowflake_cuda.ModelCuda.init(m).embedBatch(be, self.io, gpa, framed, vecs),
            },
            .embeddinggemma => |*m| switch (self.backend) {
                .cpu => try m.embedBatch(self.io, gpa, framed, vecs),
                .vulkan => |ctx| try models.embed_gemma_gpu.ModelGpu.init(m).embedBatch(ctx, self.io, gpa, framed, vecs),
                .cuda => |be| try models.embed_gemma_cuda.ModelCuda.init(m).embedBatch(be, self.io, gpa, framed, vecs),
            },
            .embeddinggemma2 => |*m| switch (self.backend) {
                .cpu => try m.embedBatch(self.io, gpa, framed, vecs),
                .vulkan => |ctx| try models.embed_gemma2_gpu.ModelGpu.init(m).embedBatch(ctx, self.io, gpa, framed, vecs),
                .cuda => |be| try models.embed_gemma2_cuda.ModelCuda.init(m).embedBatch(be, self.io, gpa, framed, vecs),
            },
            .siglip2_text => |*m| switch (self.backend) {
                .cpu => try m.embedBatch(self.io, gpa, framed, vecs),
                .vulkan => |ctx| try models.embed_siglip_gpu.TextModelGpu.init(m).embedBatch(ctx, self.io, gpa, framed, vecs),
                .cuda => |be| try models.embed_siglip_cuda.TextModelCuda.init(m).embedBatch(be, self.io, gpa, framed, vecs),
            },
        }
        return vecs;
    }

    /// Role prefix -> tokenize -> model frame, returning the framed id list
    /// (allocated with `a`). Shared by the single and batched paths.
    fn frameText(self: *const TextEncoder, a: std.mem.Allocator, text: []const u8, opts: Options) ![]const u32 {
        const kind: TextModelKind = self.impl;
        const prefix = rolePrefix(kind, opts.role);
        const prefixed = if (prefix.len == 0) text else try a.print("{s}{s}", .{ prefix, text });

        var content: std.ArrayList(u32) = .empty;
        try self.tok.encode(a, prefixed, &content);

        var framed: std.ArrayList(u32) = .empty;
        switch (kind) {
            .arctic_embed_m_v2 => {
                try framed.append(a, 0); // <s>
                try framed.appendSlice(a, content.items);
                try framed.append(a, 2); // </s>
            },
            .embeddinggemma, .embeddinggemma2 => {
                try framed.append(a, 2); // <bos>
                try framed.appendSlice(a, content.items);
                try framed.append(a, 1); // <eos>
            },
            .siglip2_text => {
                // SigLIP: [content...(≤60), <eos>=1]; the model pads to 64.
                const n = @min(content.items.len, 60);
                try framed.appendSlice(a, content.items[0..n]);
                try framed.append(a, 1); // <eos>
            },
        }
        return framed.items;
    }

    fn embedTextInto(self: *const TextEncoder, gpa: std.mem.Allocator, text: []const u8, opts: Options, out: []f32) !void {
        std.debug.assert(out.len == dim);
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const framed = try self.frameText(arena.allocator(), text, opts);
        switch (self.impl) {
            .arctic_embed_m_v2 => |*m| switch (self.backend) {
                .cpu => try m.embed(self.io, gpa, framed, out),
                .vulkan => |ctx| try models.embed_snowflake_gpu.ModelGpu.init(m).embed(ctx, self.io, gpa, framed, out),
                .cuda => |be| try models.embed_snowflake_cuda.ModelCuda.init(m).embed(be, self.io, gpa, framed, out),
            },
            .embeddinggemma => |*m| switch (self.backend) {
                .cpu => try m.embed(self.io, gpa, framed, out),
                .vulkan => |ctx| try models.embed_gemma_gpu.ModelGpu.init(m).embed(ctx, self.io, gpa, framed, out),
                .cuda => |be| try models.embed_gemma_cuda.ModelCuda.init(m).embed(be, self.io, gpa, framed, out),
            },
            .embeddinggemma2 => |*m| switch (self.backend) {
                .cpu => try m.embed(self.io, gpa, framed, out),
                .vulkan => |ctx| try models.embed_gemma2_gpu.ModelGpu.init(m).embed(ctx, self.io, gpa, framed, out),
                .cuda => |be| try models.embed_gemma2_cuda.ModelCuda.init(m).embed(be, self.io, gpa, framed, out),
            },
            .siglip2_text => |*m| switch (self.backend) {
                .cpu => try m.embed(self.io, gpa, framed, out),
                .vulkan => |ctx| try models.embed_siglip_gpu.TextModelGpu.init(m).embed(ctx, self.io, gpa, framed, out),
                .cuda => |be| try models.embed_siglip_cuda.TextModelCuda.init(m).embed(be, self.io, gpa, framed, out),
            },
        }
    }
};

/// SigLIP2 visual encoder: decoded, preprocessed RGB -> 768-d.
pub const ImageEncoder = struct {
    io: std.Io,
    model: models.embed_siglip.VisualModel,
    backend: Backend,

    /// `dir` holds `open_clip_model.safetensors`. `backend` selects CPU/GPU.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, backend: Backend) !ImageEncoder {
        return .{ .io = io, .model = try models.embed_siglip.VisualModel.open(gpa, io, dir), .backend = backend };
    }

    pub fn deinit(self: *ImageEncoder) void {
        evict(self.backend, &self.model);
        self.model.deinit();
        self.* = undefined;
    }

    /// `rgb_chw` is a decoded image preprocessed by the caller (libvips): 224²,
    /// `/255`, mean/std 0.5, CHW layout ([3*224*224]). Returns a caller-owned
    /// L2-normalized 768-d vector.
    pub fn embedImage(self: *const ImageEncoder, gpa: std.mem.Allocator, rgb_chw: []const f32) ![]f32 {
        const out = try gpa.alloc(f32, dim);
        errdefer gpa.free(out);
        try self.embedImageInto(gpa, rgb_chw, out);
        return out;
    }

    /// Batch variant (DiffKeep indexes ~8 images at a time). Runs a single fused
    /// ViT forward over all images on the selected backend. Caller-owned.
    pub fn embedImageBatch(self: *const ImageEncoder, gpa: std.mem.Allocator, images: []const []const f32) ![][]f32 {
        const vecs = try gpa.alloc([]f32, images.len);
        errdefer gpa.free(vecs);
        for (vecs) |*v| v.* = &.{};
        errdefer for (vecs) |v| if (v.len != 0) gpa.free(v);
        for (vecs) |*v| v.* = try gpa.alloc(f32, dim);

        if (images.len == 1) {
            try self.embedImageInto(gpa, images[0], vecs[0]);
            return vecs;
        }
        switch (self.backend) {
            .cpu => try self.model.embedBatch(self.io, gpa, images, vecs),
            .vulkan => |ctx| try models.embed_siglip_gpu.VisualModelGpu.init(&self.model).embedBatch(ctx, self.io, gpa, images, vecs),
            .cuda => |be| try models.embed_siglip_cuda.VisualModelCuda.init(&self.model).embedBatch(be, self.io, gpa, images, vecs),
        }
        return vecs;
    }

    fn embedImageInto(self: *const ImageEncoder, gpa: std.mem.Allocator, rgb_chw: []const f32, out: []f32) !void {
        switch (self.backend) {
            .cpu => try self.model.embed(self.io, gpa, rgb_chw, out),
            .vulkan => |ctx| try models.embed_siglip_gpu.VisualModelGpu.init(&self.model).embed(ctx, self.io, gpa, rgb_chw, out),
            .cuda => |be| try models.embed_siglip_cuda.VisualModelCuda.init(&self.model).embed(be, self.io, gpa, rgb_chw, out),
        }
    }
};

/// An interleaved RGB8 image of any size.
pub const Rgb = models.embed_gemma2.Rgb;

/// EmbeddingGemma 2 with its vision tower: text and images into one 768-d space, so a
/// text query ranks images directly. Text takes the same roles and prompts as
/// `TextEncoder`; images take no prompt and are resized by the encoder.
pub const MultimodalEncoder = struct {
    text: TextEncoder,

    /// `dir` holds `model.safetensors` and `tokenizer.json`.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, backend: Backend) !MultimodalEncoder {
        return .{ .text = try TextEncoder.openWith(gpa, io, .embeddinggemma2, dir, backend, true) };
    }

    pub fn deinit(self: *MultimodalEncoder) void {
        self.text.deinit();
        self.* = undefined;
    }

    /// Soft tokens per image: 70, 140, 280 (the default), 560 or 1120. Compute scales
    /// with it; vectors at different budgets are not interchangeable in one index.
    pub fn setImageTokens(self: *MultimodalEncoder, tokens: usize) !void {
        switch (tokens) {
            70, 140, 280, 560, 1120 => self.text.impl.embeddinggemma2.vit.?.cfg.max_tokens = tokens,
            else => return error.UnsupportedImageTokens,
        }
    }

    pub fn embedText(self: *const MultimodalEncoder, gpa: std.mem.Allocator, text: []const u8, opts: Options) ![]f32 {
        return self.text.embedText(gpa, text, opts);
    }

    pub fn embedTextBatch(self: *const MultimodalEncoder, gpa: std.mem.Allocator, texts: []const []const u8, opts: Options) ![][]f32 {
        return self.text.embedTextBatch(gpa, texts, opts);
    }

    /// Caller-owned unit vector.
    pub fn embedImage(self: *const MultimodalEncoder, gpa: std.mem.Allocator, img: Rgb) ![]f32 {
        const vecs = try self.embedImageBatch(gpa, &.{img});
        defer gpa.free(vecs);
        return vecs[0];
    }

    /// One vector per image: the vision tower per image, then one text-body forward
    /// over the batch. Caller-owned vectors + slice.
    pub fn embedImageBatch(self: *const MultimodalEncoder, gpa: std.mem.Allocator, images: []const Rgb) ![][]f32 {
        const vecs = try gpa.alloc([]f32, images.len);
        errdefer gpa.free(vecs);
        for (vecs) |*v| v.* = &.{};
        errdefer for (vecs) |v| if (v.len != 0) gpa.free(v);
        for (vecs) |*v| v.* = try gpa.alloc(f32, dim);
        const m = &self.text.impl.embeddinggemma2;
        const io = self.text.io;
        switch (self.text.backend) {
            .cpu => try m.embedImageBatch(io, gpa, images, vecs),
            .vulkan => |ctx| try models.embed_gemma2_gpu.ModelGpu.init(m).embedImageBatch(ctx, io, gpa, images, vecs),
            .cuda => |be| try models.embed_gemma2_cuda.ModelCuda.init(m).embedImageBatch(be, io, gpa, images, vecs),
        }
        return vecs;
    }
};

/// Drop the device copies of `model`'s weights; run before the model frees the host
/// buffers they key on.
fn evict(backend: Backend, model: anytype) void {
    switch (backend) {
        .cpu => {},
        .vulkan => |ctx| models.tower_gpu.evictModel(ctx, model),
        .cuda => |be| models.tower_cuda.evictModel(be, model),
    }
}

/// Matryoshka truncation of a unit vector from a model trained for it (EmbeddingGemma
/// 1 and 2 at 512/256/128): the first `n` dims, renormalized in place. Without the
/// renormalization the cosine scale differs per vector.
pub fn truncate(v: []f32, n: usize) []f32 {
    const out = v[0..n];
    models.embed_gemma2.normalize(out);
    return out;
}

/// The prefix prepended to `text` before tokenization, per model + role.
fn rolePrefix(kind: TextModelKind, role: Role) []const u8 {
    return switch (kind) {
        .arctic_embed_m_v2 => switch (role) {
            .query => "query: ",
            .document => "",
        },
        .embeddinggemma, .embeddinggemma2 => switch (role) {
            .query => "task: search result | query: ",
            .document => "title: none | text: ",
        },
        .siglip2_text => "", // symmetric
    };
}

// --- tests -----------------------------------------------------------------

// End-to-end façade check: for Snowflake a `.document` role adds no prefix, so
// the façade output must reproduce the direct-model ONNX reference (validating
// tokenizer load + framing + model wiring through the public surface). Skipped
// when the checkout is absent.
test "embed façade: Snowflake document matches ONNX reference" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dir = "../DiffKeep/Models/snowflake-arctic-embed-m-v2.0";
    std.Io.Dir.cwd().access(io, dir, .{}) catch return error.SkipZigTest;

    const ref_bytes = std.Io.Dir.cwd().readFileAlloc(io, "testdata/snowflake_ref_vectors.json", gpa, .limited(4 * 1024 * 1024)) catch return error.SkipZigTest;
    defer gpa.free(ref_bytes);
    var ref = try std.json.parseFromSlice(std.json.Value, gpa, ref_bytes, .{});
    defer ref.deinit();

    var enc = try TextEncoder.open(gpa, io, .arctic_embed_m_v2, dir, .cpu);
    defer enc.deinit();

    // "hello world" reference vector.
    const want_json = ref.value.object.get("hello world").?.object.get("vec").?.array.items;
    var want: [dim]f32 = undefined;
    for (want_json, 0..) |val, i| want[i] = @floatCast(val.float);

    const got = try enc.embedText(gpa, "hello world", .{ .role = .document });
    defer gpa.free(got);

    var dot: f32 = 0;
    var ss: f32 = 0;
    for (got, want) |g, w| {
        dot += g * w;
        ss += g * g;
    }
    errdefer std.debug.print("cosine {d}, |got| {d}\n", .{ dot, @sqrt(ss) });
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), @sqrt(ss), 1e-4); // unit norm
    try std.testing.expect(dot >= 0.999); // both unit -> cosine

    // Batch variant returns one vector per input.
    const batch = try enc.embedTextBatch(gpa, &.{ "hello world", "a query" }, .{ .role = .query });
    defer {
        for (batch) |v| gpa.free(v);
        gpa.free(batch);
    }
    try std.testing.expectEqual(@as(usize, 2), batch.len);
    try std.testing.expectEqual(@as(usize, dim), batch[0].len);
}

// Façade -> GPU dispatch: opening with `.vulkan` must route through the device
// forward and match the CPU façade. Gated on a Vulkan device + the checkpoint.
test "embed façade Vulkan matches CPU" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dir = "../DiffKeep/Models/snowflake-arctic-embed-m-v2.0";
    std.Io.Dir.cwd().access(io, dir, .{}) catch return error.SkipZigTest;
    const ctx = tp_gpu.context.Context.init(gpa, std.testing.io) catch return error.SkipZigTest;
    defer ctx.deinit();

    var cpu_enc = try TextEncoder.open(gpa, io, .arctic_embed_m_v2, dir, .cpu);
    defer cpu_enc.deinit();
    var gpu_enc = try TextEncoder.open(gpa, io, .arctic_embed_m_v2, dir, .{ .vulkan = ctx });
    defer gpu_enc.deinit();

    const c = try cpu_enc.embedText(gpa, "a red bicycle", .{ .role = .query });
    defer gpa.free(c);
    const g = try gpu_enc.embedText(gpa, "a red bicycle", .{ .role = .query });
    defer gpa.free(g);

    var dot: f32 = 0;
    for (c, g) |a, b| dot += a * b;
    errdefer std.debug.print("façade vulkan cosine {d}\n", .{dot});
    try std.testing.expect(dot >= 0.999);
}

// The façade's prompt, tokenizer and frame against the ids sentence-transformers fed
// the model, then a vector end to end.
test "embed façade: EmbeddingGemma 2 matches sentence-transformers" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const eg2 = models.embed_gemma2;
    try test_gate.requireModelFile(io, eg2.test_dir ++ "/model.safetensors");
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = try eg2.loadReference(a, io);

    var enc = try TextEncoder.open(gpa, io, .embeddinggemma2, eg2.test_dir, .cpu);
    defer enc.deinit();

    var roles: usize = 0;
    for (cases) |c| {
        const ids = if (c.prompt) |p| blk: {
            roles += 1;
            const role: Role = if (std.mem.eql(u8, p, "query")) .query else .document;
            break :blk try enc.frameText(a, c.text, .{ .role = role });
        } else blk: {
            var content: std.ArrayList(u32) = .empty;
            try content.append(a, 2);
            try enc.tok.encode(a, c.text, &content);
            try content.append(a, 1);
            break :blk content.items;
        };
        errdefer std.debug.print("{s}: ids differ\n", .{c.name});
        try std.testing.expectEqualSlices(u32, c.ids, ids);
    }
    try std.testing.expect(roles >= 2);

    const q = for (cases) |c| {
        if (c.prompt != null and std.mem.eql(u8, c.prompt.?, "query")) break c;
    } else unreachable;
    const got = try enc.embedText(gpa, q.text, .{ .role = .query });
    defer gpa.free(got);
    const cos = eg2.cosine(got, q.vec);
    errdefer std.debug.print("{s}: cosine {d}\n", .{ q.name, cos });
    try std.testing.expect(cos >= 0.99999);

    // Truncation keeps a unit vector whose prefix points the same way.
    var copy: [dim]f32 = undefined;
    @memcpy(&copy, got);
    const t = truncate(&copy, 256);
    var ss: f32 = 0;
    for (t) |e| ss += e * e;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), ss, 1e-5);
    try std.testing.expect(eg2.cosine(t, got[0..256]) > 0.99999);
}

fn readTestImage(a: std.mem.Allocator, io: std.Io, name: []const u8) !Rgb {
    const png = try std.Io.Dir.cwd().readFileAlloc(io, try a.print("testdata/embeddinggemma2_{s}.png", .{name}), a, .limited(8 << 20));
    const dec = try @import("tp_core").image.decodePngRgb(a, png);
    return .{ .pixels = dec.pixels, .width = dec.width, .height = dec.height };
}

test "embed façade: MultimodalEncoder matches the EmbeddingGemma 2 references" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const eg2 = models.embed_gemma2;
    try test_gate.requireModelFile(io, eg2.test_dir ++ "/model.safetensors");
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    var enc = try MultimodalEncoder.open(gpa, io, eg2.test_dir, .cpu);
    defer enc.deinit();

    const cases = try eg2.loadReference(a, io);
    const q = for (cases) |c| {
        if (c.prompt != null and std.mem.eql(u8, c.prompt.?, "query")) break c;
    } else unreachable;
    const tv = try enc.embedText(gpa, q.text, .{ .role = .query });
    defer gpa.free(tv);
    try std.testing.expect(eg2.cosine(tv, q.vec) >= 0.99999);

    var ref = try @import("tp_core").safetensors.SafeTensors.open(gpa, io, "testdata/embeddinggemma2_image_ref.safetensors");
    defer ref.deinit();
    const want = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, (try ref.require("vec.down")).bytes));
    const iv = try enc.embedImage(gpa, try readTestImage(a, io, "down"));
    defer gpa.free(iv);
    var w: [dim]f32 = undefined;
    for (&w, want) |*d, s| d.* = s;
    const cos = eg2.cosine(iv, &w);
    errdefer std.debug.print("image cosine {d}\n", .{cos});
    try std.testing.expect(cos >= 0.9999);
}

// A GPU encoder against the CPU one, then closed and reopened: its weights are
// evicted in `deinit`, so the second encoder uploads its own.
fn testMultimodalDevice(backend: Backend) !void {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const eg2 = models.embed_gemma2;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const imgs = [_]Rgb{ try readTestImage(arena.allocator(), io, "up"), try readTestImage(arena.allocator(), io, "tall") };

    var cpu = try MultimodalEncoder.open(gpa, io, eg2.test_dir, .cpu);
    defer cpu.deinit();
    const want = try cpu.embedImageBatch(gpa, &imgs);
    defer {
        for (want) |v| gpa.free(v);
        gpa.free(want);
    }
    const want_t = try cpu.embedText(gpa, "a red fox", .{});
    defer gpa.free(want_t);

    for (0..2) |round| {
        var dev = try MultimodalEncoder.open(gpa, io, eg2.test_dir, backend);
        defer dev.deinit();
        const got = try dev.embedImageBatch(gpa, &imgs);
        defer {
            for (got) |v| gpa.free(v);
            gpa.free(got);
        }
        const got_t = try dev.embedText(gpa, "a red fox", .{});
        defer gpa.free(got_t);
        for (want, got) |wv, gv| {
            const cos = eg2.cosine(wv, gv);
            errdefer std.debug.print("round {d}: image cosine {d}\n", .{ round, cos });
            try std.testing.expect(cos >= 0.9999);
        }
        try std.testing.expect(eg2.cosine(want_t, got_t) >= 0.9999);
    }
}

test "embed façade: MultimodalEncoder Vulkan matches CPU across reopen" {
    try test_gate.requireModelFile(std.testing.io, models.embed_gemma2.test_dir ++ "/model.safetensors");
    const ctx = tp_gpu.context.Context.init(std.testing.allocator, std.testing.io) catch return error.SkipZigTest;
    defer ctx.deinit();
    try testMultimodalDevice(.{ .vulkan = ctx });
}

test "embed façade: MultimodalEncoder CUDA matches CPU across reopen" {
    try test_gate.requireModelFile(std.testing.io, models.embed_gemma2.test_dir ++ "/model.safetensors");
    const be = tp_gpu.cuda.Backend.init(std.testing.allocator) catch return error.SkipZigTest;
    defer be.deinit();
    try testMultimodalDevice(.{ .cuda = be });
}
