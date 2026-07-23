//! `tp.embed` — the high-level embedding façade DiffKeep calls. It hides the
//! per-model tokenizer, role prefixes, frame tokens, and pooling behind two
//! types that each return an L2-normalized 768-d vector:
//!   - `TextEncoder` — text → vec (Snowflake / EmbeddingGemma / SigLIP2 text).
//!   - `ImageEncoder` — decoded RGB → vec (SigLIP2 visual).
//!
//! The underlying model forwards (`models.embed_*`) are validated bit-faithful
//! vs the reference ONNX; this layer only adds the text-side glue that the plan
//! (DIFFKEEP.md) specifies: Snowflake's `"query: "` asymmetry, EmbeddingGemma's
//! task prefixes, SigLIP2's fixed 64-token frame. CPU only for now (a `backend`
//! parameter will arrive with the GPU paths — see DIFFKEEP.md M6).

const std = @import("std");
const models = @import("tp_models").models;
const tokenizer = @import("tp_core").tokenizer;
const tp_gpu = @import("tp_gpu");

const Tokenizer = tokenizer.Tokenizer;

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

/// Which text encoder to load. `arctic_embed_m_v2` and `embeddinggemma` are the
/// two candidates for the prompt/query text space; `siglip2_text` is the
/// cross-modal (image-space) text query encoder.
pub const TextModelKind = enum { arctic_embed_m_v2, embeddinggemma, siglip2_text };

/// Snowflake/EmbeddingGemma are asymmetric (queries and documents get different
/// prefixes); SigLIP2 text is symmetric (role ignored).
pub const Role = enum { query, document };

pub const Options = struct { role: Role = .query };

const TextImpl = union(TextModelKind) {
    arctic_embed_m_v2: models.embed_snowflake.Model,
    embeddinggemma: models.embed_gemma.Model,
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
        var pbuf: [1024]u8 = undefined;
        const tok_path = try std.fmt.bufPrint(&pbuf, "{s}/tokenizer.json", .{dir});
        const tok_bytes = try std.Io.Dir.cwd().readFileAlloc(io, tok_path, gpa, .limited(64 * 1024 * 1024));
        defer gpa.free(tok_bytes);

        var tok: Tokenizer = switch (kind) {
            .arctic_embed_m_v2 => try Tokenizer.initUnigramFromTokenizerJson(gpa, tok_bytes),
            .embeddinggemma, .siglip2_text => try Tokenizer.initGemma4FromTokenizerJson(gpa, tok_bytes),
        };
        errdefer tok.deinit();

        const impl: TextImpl = switch (kind) {
            .arctic_embed_m_v2 => .{ .arctic_embed_m_v2 = try models.embed_snowflake.Model.open(gpa, io, dir) },
            .embeddinggemma => .{ .embeddinggemma = try models.embed_gemma.Model.open(gpa, io, dir) },
            .siglip2_text => .{ .siglip2_text = try models.embed_siglip.TextModel.open(gpa, io, dir) },
        };
        return .{ .io = io, .tok = tok, .impl = impl, .backend = backend };
    }

    pub fn deinit(self: *TextEncoder) void {
        switch (self.impl) {
            inline else => |*m| m.deinit(),
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
    /// backend (CPU / Vulkan / CUDA) — the passed slice IS the "configurable
    /// amount". Returns caller-owned vectors + slice.
    pub fn embedTextBatch(self: *const TextEncoder, gpa: std.mem.Allocator, texts: []const []const u8, opts: Options) ![][]f32 {
        const vecs = try gpa.alloc([]f32, texts.len);
        errdefer gpa.free(vecs);
        for (vecs) |*v| v.* = &.{};
        errdefer for (vecs) |v| if (v.len != 0) gpa.free(v);
        for (vecs) |*v| v.* = try gpa.alloc(f32, dim);

        // B=1 has no cross-item amortization — take the single path (avoids the
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
            .siglip2_text => |*m| switch (self.backend) {
                .cpu => try m.embedBatch(self.io, gpa, framed, vecs),
                .vulkan => |ctx| try models.embed_siglip_gpu.TextModelGpu.init(m).embedBatch(ctx, self.io, gpa, framed, vecs),
                .cuda => |be| try models.embed_siglip_cuda.TextModelCuda.init(m).embedBatch(be, self.io, gpa, framed, vecs),
            },
        }
        return vecs;
    }

    /// Role prefix → tokenize → model frame, returning the framed id list
    /// (allocated with `a`). Shared by the single and batched paths.
    fn frameText(self: *const TextEncoder, a: std.mem.Allocator, text: []const u8, opts: Options) ![]const u32 {
        const kind: TextModelKind = self.impl;
        const prefix = rolePrefix(kind, opts.role);
        const prefixed = if (prefix.len == 0) text else try std.fmt.allocPrint(a, "{s}{s}", .{ prefix, text });

        var content: std.ArrayList(u32) = .empty;
        try self.tok.encode(a, prefixed, &content);

        var framed: std.ArrayList(u32) = .empty;
        switch (kind) {
            .arctic_embed_m_v2 => {
                try framed.append(a, 0); // <s>
                try framed.appendSlice(a, content.items);
                try framed.append(a, 2); // </s>
            },
            .embeddinggemma => {
                try framed.append(a, 2); // <bos>
                try framed.appendSlice(a, content.items);
                try framed.append(a, 1); // <eos>
            },
            .siglip2_text => {
                // SigLIP: [content…(≤60), <eos>=1]; the model pads to 64.
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
            .siglip2_text => |*m| switch (self.backend) {
                .cpu => try m.embed(self.io, gpa, framed, out),
                .vulkan => |ctx| try models.embed_siglip_gpu.TextModelGpu.init(m).embed(ctx, self.io, gpa, framed, out),
                .cuda => |be| try models.embed_siglip_cuda.TextModelCuda.init(m).embed(be, self.io, gpa, framed, out),
            },
        }
    }
};

/// SigLIP2 visual encoder: decoded, preprocessed RGB → 768-d.
pub const ImageEncoder = struct {
    io: std.Io,
    model: models.embed_siglip.VisualModel,
    backend: Backend,

    /// `dir` holds `open_clip_model.safetensors`. `backend` selects CPU/GPU.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, backend: Backend) !ImageEncoder {
        return .{ .io = io, .model = try models.embed_siglip.VisualModel.open(gpa, io, dir), .backend = backend };
    }

    pub fn deinit(self: *ImageEncoder) void {
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

/// The prefix prepended to `text` before tokenization, per model + role.
fn rolePrefix(kind: TextModelKind, role: Role) []const u8 {
    return switch (kind) {
        .arctic_embed_m_v2 => switch (role) {
            .query => "query: ",
            .document => "",
        },
        .embeddinggemma => switch (role) {
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
    try std.testing.expect(dot >= 0.999); // both unit → cosine

    // Batch variant returns one vector per input.
    const batch = try enc.embedTextBatch(gpa, &.{ "hello world", "a query" }, .{ .role = .query });
    defer {
        for (batch) |v| gpa.free(v);
        gpa.free(batch);
    }
    try std.testing.expectEqual(@as(usize, 2), batch.len);
    try std.testing.expectEqual(@as(usize, dim), batch[0].len);
}

// Façade → GPU dispatch: opening with `.vulkan` must route through the device
// forward and match the CPU façade. Gated on a Vulkan device + the checkpoint.
test "embed façade Vulkan matches CPU" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dir = "../DiffKeep/Models/snowflake-arctic-embed-m-v2.0";
    std.Io.Dir.cwd().access(io, dir, .{}) catch return error.SkipZigTest;
    const ctx = tp_gpu.context.Context.init(gpa) catch return error.SkipZigTest;
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
