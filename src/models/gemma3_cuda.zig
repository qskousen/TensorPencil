//! Gemma 3 LM on the CUDA backend (tp-llm --backend zig-cuda / cuda): the
//! text stack runs device-resident. Prefill is batched chunks
//! (opMatmulQuant / grouped dp4a GEMVs); decode is per-op fused GEMVs.
//!
//! Gemma-specific vs qwen3_cuda: FOUR RMSNorms per layer (input, post-attn,
//! pre-FFN, post-FFN — the two "post" norms apply to the sublayer output
//! before its residual add), embeddings scaled by sqrt(hidden) on the host
//! before upload, GeGLU (be.geluMul), a tied LM head, and RoPE whose
//! base/scale alternate by layer: every `swa_pattern`-th layer is GLOBAL
//! (theta 1e6, linear scale 1/8, full causal attention), the rest LOCAL
//! (theta 1e4, sliding-window causal mask via opAttnDecode's window arg).
//! Two RoPE tables (global/local) live on device; each layer picks one.
//!
//! No decode-graph capture (measured +0% on this memory-bound regime — the
//! async queue already hides per-op launches), and speculative decoding is
//! unsupported. All weights are GGUF block quants and dequantize inside the
//! GEMM; the Gguf mapping must outlive the model.

const std = @import("std");
const gemma3 = @import("gemma3.zig");
const qwen3 = @import("qwen3.zig");
const cuda = @import("../gpu/cuda.zig");
const ops = @import("../ops.zig");
const kvmod = @import("../llm/kv_cache.zig");

const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;
const Growable = Backend.GrowableTensor;

/// KV chunks per head in the decode attention split; the smaller prefill
/// count bounds the flash-decode scratch across a whole chunk of queries.
const nsplit = 32;
const nsplit_prefill = 8;
/// Rows per batched-prefill chunk (also the activation-buffer height).
const prefill_chunk = 128;
/// Batches up to this take the grouped dp4a GEMV (weight streamed ceil(n/8)x)
/// instead of opMatmulQuant's dequant-to-f16 GEMM (qwen3_cuda's crossover).
const grouped_gemv_max = 40;

fn nbuf(be: *Backend, weights: []const f32) !Buf {
    return .{ .buf = try be.smallBuffer(std.mem.sliceAsBytes(weights)), .mem = .null_handle, .size = 0 };
}

fn offsetBufSized(b: Buf, off_bytes: usize, size: usize) Buf {
    return .{ .buf = @enumFromInt(@intFromEnum(b.buf) + off_bytes), .mem = b.mem, .size = size };
}

pub const CudaLM = struct {
    lm: *const gemma3.Model,
    be: *Backend,
    gpa: std.mem.Allocator,
    cfg: gemma3.Config,
    capacity: usize,
    max_capacity: usize,
    len: usize,
    /// half = head_dim/2; the sin table starts at cap.max*half in each freqs
    /// buffer (baked capacity-independent, like qwen3_cuda).
    sin_off: usize,
    /// Global-layer (theta 1e6, scale 1/8) and local-layer (theta 1e4) RoPE
    /// tables; [2 * cap.max * half] f32 (cos then sin) each.
    freqs_global: Buf,
    freqs_local: Buf,
    k_cache: []Growable,
    v_cache: []Growable,
    bufs: Bufs,
    arena: std.heap.ArenaAllocator,

    pub fn init(gpa: std.mem.Allocator, be: *Backend, lm: *const gemma3.Model, cap: kvmod.Capacity) !CudaLM {
        const cfg = lm.cfg;
        switch (lm.embed.dtype) {
            .bf16, .q8_0, .q4_k, .q5_k, .q6_k => {},
            else => return error.UnsupportedModelConfig,
        }
        if (cfg.head_dim != 256) return error.UnsupportedModelConfig; // attn_split_h256

        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();

        var self: CudaLM = undefined;
        self.lm = lm;
        self.be = be;
        self.gpa = gpa;
        self.cfg = cfg;
        self.capacity = cap.initial;
        self.max_capacity = cap.max;
        self.len = 0;
        self.sin_off = cap.max * (cfg.head_dim / 2);

        self.freqs_global = try uploadFreqs(be, gpa, cap.max, cfg.head_dim, cfg.rope_theta, cfg.rope_freq_scale);
        errdefer be.tensorDestroy(&self.freqs_global);
        self.freqs_local = try uploadFreqs(be, gpa, cap.max, cfg.head_dim, cfg.rope_theta_local, 1.0);
        errdefer be.tensorDestroy(&self.freqs_local);

        self.bufs = try Bufs.init(be, cfg);

        self.k_cache = try alloc.alloc(Growable, cfg.n_layers);
        self.v_cache = try alloc.alloc(Growable, cfg.n_layers);
        for (self.k_cache, self.v_cache) |*kb, *vb| {
            kb.* = try be.growableCreate(cap.initial * cfg.kvDim() * 4, cap.max * cfg.kvDim() * 4);
            vb.* = try be.growableCreate(cap.initial * cfg.kvDim() * 4, cap.max * cfg.kvDim() * 4);
        }

        self.arena = arena;
        return self;
    }

    fn uploadFreqs(be: *Backend, gpa: std.mem.Allocator, rows: usize, head_dim: usize, theta: f64, freq_scale: f64) !Buf {
        const half = head_dim / 2;
        var freqs = try ops.rope.rotateHalfFreqsScaled(gpa, rows, head_dim, theta, freq_scale);
        defer freqs.deinit(gpa);
        const fp = try gpa.alloc(f32, 2 * rows * half);
        defer gpa.free(fp);
        @memcpy(fp[0 .. rows * half], freqs.cos);
        @memcpy(fp[rows * half ..], freqs.sin);
        var buf = try be.tensorCreate(fp.len * 4);
        errdefer be.tensorDestroy(&buf);
        try be.tensorUpload(buf, std.mem.sliceAsBytes(fp));
        return buf;
    }

    pub fn deinit(self: *CudaLM) void {
        const be = self.be;
        for (self.k_cache) |*b| be.growableDestroy(b);
        for (self.v_cache) |*b| be.growableDestroy(b);
        be.tensorDestroy(&self.freqs_global);
        be.tensorDestroy(&self.freqs_local);
        self.bufs.deinit(be);
        self.arena.deinit();
        self.* = undefined;
    }

    pub fn cached(self: *const CudaLM) usize {
        return self.len;
    }
    pub fn remaining(self: *const CudaLM) usize {
        return self.capacity - self.len;
    }
    pub fn capacityMax(self: *const CudaLM) usize {
        return self.max_capacity;
    }
    pub fn vocab(self: *const CudaLM) usize {
        return self.cfg.vocab;
    }

    pub fn ensureCapacity(self: *CudaLM, min_rows: usize) !void {
        if (min_rows <= self.capacity) return;
        if (min_rows > self.max_capacity) return error.ContextFull;
        const target = kvmod.growTarget(self.capacity, min_rows, self.max_capacity);
        const bytes = target * self.cfg.kvDim() * 4;
        for ([2][]Growable{ self.k_cache, self.v_cache }) |caches| {
            for (caches) |*b| {
                self.be.growableEnsure(b, bytes) catch |err| switch (err) {
                    error.DeviceOutOfMemory, error.OutOfMemory => return error.ContextFull,
                    else => return err,
                };
            }
        }
        self.capacity = target;
    }

    /// Forward `ids` at positions [len, len+ids.len); write last-position
    /// vocab logits. Prefill runs in prefill_chunk-sized batches (only the
    /// final chunk computes the LM head).
    pub fn step(self: *CudaLM, io: std.Io, ids: []const u32, logits: []f32) !void {
        _ = io;
        std.debug.assert(ids.len >= 1 and ids.len <= self.remaining());
        std.debug.assert(logits.len == self.cfg.vocab);
        var off: usize = 0;
        while (off < ids.len) {
            const n: usize = @min(prefill_chunk, ids.len - off);
            const last = (off + n == ids.len);
            try self.embedChunk(ids[off..][0..n], if (last) logits else null);
            off += n;
        }
    }

    /// Prefill text tokens (no logits) — for interleaving with prefillImage.
    pub fn prefill(self: *CudaLM, ids: []const u32) !void {
        var off: usize = 0;
        while (off < ids.len) {
            const n: usize = @min(prefill_chunk, ids.len - off);
            try self.embedChunk(ids[off..][0..n], null);
            off += n;
        }
    }

    /// Prefill one image's projected embeddings ([grid_w*grid_h][hidden],
    /// injected UNSCALED) at the next sequential positions. grid dims are
    /// carried for interface parity (gemma is always 16x16 = 256).
    pub fn prefillImage(self: *CudaLM, embeds: []const f32, grid_w: usize, grid_h: usize) !void {
        _ = grid_w;
        _ = grid_h;
        const cfg = self.cfg;
        const total = embeds.len / cfg.hidden;
        std.debug.assert(embeds.len == total * cfg.hidden);
        var off: usize = 0;
        while (off < total) {
            const n: usize = @min(prefill_chunk, total - off);
            try self.forwardRows(embeds[off * cfg.hidden ..][0 .. n * cfg.hidden], null);
            off += n;
        }
    }

    /// Embed `ids` (gather + sqrt(hidden) scale, host-side) then forward the
    /// resulting rows.
    fn embedChunk(self: *CudaLM, ids: []const u32, logits: ?[]f32) !void {
        const cfg = self.cfg;
        const n = ids.len;
        const x = try self.gpa.alloc(f32, n * cfg.hidden);
        defer self.gpa.free(x);
        try qwen3.embedTokens(self.lm.embed, ids, x);
        const scale = cfg.embedScale();
        for (x) |*v| v.* *= scale;
        try self.forwardRows(x, logits);
    }

    /// One batched forward over `n` pre-embedded input rows `x_host`
    /// ([n*hidden]) at positions [len, len+n). When `logits` is set, the last
    /// row's final-normed hidden feeds the LM head. Advances len by n.
    fn forwardRows(self: *CudaLM, x_host: []const f32, logits: ?[]f32) !void {
        const be = self.be;
        const cfg = self.cfg;
        const b = &self.bufs;
        const n = x_host.len / cfg.hidden;
        const hd = cfg.head_dim;
        const eps = cfg.rms_eps;
        const pos0 = self.len;
        std.debug.assert(n >= 1 and n <= prefill_chunk and n <= self.remaining());

        try be.tensorUpload(offsetBufSized(b.x, 0, n * cfg.hidden * 4), std.mem.sliceAsBytes(x_host));

        try be.beginBatch();
        errdefer if (be.batching()) be.abortBatch();

        const attn_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        for (self.lm.layers, 0..) |*layer, l| {
            const global = cfg.isGlobal(l);
            const freqs = if (global) self.freqs_global else self.freqs_local;
            const window: usize = if (global) 0 else cfg.sliding_window;

            // --- Attention ---
            try be.qkNorm(b.x, b.normed, try nbuf(be, layer.input_norm), n, cfg.hidden, eps);
            try self.linear(b.q, b.normed, layer.q, cfg.qDim(), cfg.hidden, n);
            try self.linear(b.k, b.normed, layer.k, cfg.kvDim(), cfg.hidden, n);
            try self.linear(b.v, b.normed, layer.v, cfg.kvDim(), cfg.hidden, n);
            try be.qkNorm(b.q, b.q, try nbuf(be, layer.q_norm), n * cfg.n_heads, hd, eps);
            try be.qkNorm(b.k, b.k, try nbuf(be, layer.k_norm), n * cfg.n_kv_heads, hd, eps);
            try be.ropeHalf(b.q, freqs, n, cfg.n_heads, hd / 2, self.sin_off, pos0);
            try be.ropeHalf(b.k, freqs, n, cfg.n_kv_heads, hd / 2, self.sin_off, pos0);
            try be.tensorCopy(self.k_cache[l].buf, pos0 * cfg.kvDim() * 4, b.k, 0, n * cfg.kvDim() * 4);
            try be.tensorCopy(self.v_cache[l].buf, pos0 * cfg.kvDim() * 4, b.v, 0, n * cfg.kvDim() * 4);
            const ns: usize = if (n == 1) nsplit else nsplit_prefill;
            try be.opAttnDecode(b.q, self.k_cache[l].buf, self.v_cache[l].buf, b.attn, b.attn_scratch, pos0 + 1, n, cfg.n_heads, cfg.n_kv_heads, hd, ns, attn_scale, window);
            // o_proj, then post-attention norm on the attn output BEFORE the residual.
            try self.linear(b.t, b.attn, layer.o, cfg.hidden, cfg.qDim(), n);
            try be.qkNorm(b.t, b.t, try nbuf(be, layer.post_attn_norm), n, cfg.hidden, eps);
            try be.opAdd(b.x, b.t, n * cfg.hidden);

            // --- MLP (GeGLU) ---
            try be.qkNorm(b.x, b.normed, try nbuf(be, layer.pre_ffn_norm), n, cfg.hidden, eps);
            try self.linear(b.gate, b.normed, layer.gate, cfg.intermediate, cfg.hidden, n);
            try self.linear(b.up, b.normed, layer.up, cfg.intermediate, cfg.hidden, n);
            try be.geluMul(b.gate, b.up, n * cfg.intermediate);
            try self.linear(b.t, b.gate, layer.down, cfg.hidden, cfg.intermediate, n);
            try be.qkNorm(b.t, b.t, try nbuf(be, layer.post_ffn_norm), n, cfg.hidden, eps);
            try be.opAdd(b.x, b.t, n * cfg.hidden);
        }

        if (logits) |out| {
            const h = cfg.hidden;
            try be.qkNorm(offsetBufSized(b.x, (n - 1) * h * 4, h * 4), b.t, try nbuf(be, self.lm.final_norm), 1, h, eps);
            try self.lmHead(b.logits, b.t);
            try be.endBatch();
            self.len += n;
            try be.tensorDownload(offsetBufSized(b.logits, 0, cfg.vocab * 4), std.mem.sliceAsBytes(out));
        } else {
            try be.endBatch();
            self.len += n;
        }
    }

    /// LM head over one normed hidden row: y[vocab] = head @ x (tied,
    /// block-quantized).
    fn lmHead(self: *CudaLM, y: Buf, x: Buf) !void {
        const head = self.lm.head;
        try self.be.opGemvQuant(head.dtype, y, x, head.bytes, head.scale, self.cfg.vocab, self.cfg.hidden);
    }

    /// Dense linear over `seq` rows: int8 dp4a GEMV (decode + small batches,
    /// quantized activation × block-quant weight — faster than the f32
    /// dequant GEMV and numerically matches llama.cpp's mmvq), or the
    /// dequant-to-f16 tensor-core GEMM (large prefills). All Gemma weights are
    /// GGUF block quants; a weight whose shape isn't dp4a-tileable
    /// (cols%256 / rows%8) falls back to the f32 GEMV.
    fn linear(self: *CudaLM, y: Buf, x: Buf, w: ops.matmul.Weight, rows_out: usize, cols: usize, seq: usize) !void {
        const be = self.be;
        std.debug.assert(w.dtype.isBlockQuant());
        const dp4a_ok = cols % 256 == 0 and rows_out % 8 == 0;
        if (seq == 1) {
            if (!dp4a_ok) {
                try be.opGemvQuant(w.dtype, y, x, w.bytes, w.scale, rows_out, cols);
            } else {
                try be.opGemvQuantizeX(x, cols);
                // Dedicated single-input kernels for q5_k/q6_k; the grouped
                // kernel (ng=1) covers q4_k/q8_0 (which have no single twin).
                if (w.dtype == .q5_k or w.dtype == .q6_k) {
                    try be.opGemvQuantQ8(w.dtype, y, w.bytes, w.scale, rows_out, cols);
                } else {
                    try be.opGemvQuantQ8N(w.dtype, y, w.bytes, w.scale, rows_out, cols, 1, 0, 1);
                }
            }
        } else if (seq <= grouped_gemv_max and cols % 256 == 0 and rows_out % 8 == 0) {
            try be.opGemvQuantizeX(x, seq * cols);
            var off: usize = 0;
            while (off < seq) : (off += 8) {
                const ng: usize = @min(8, seq - off);
                try be.opGemvQuantQ8N(w.dtype, offsetBufSized(y, off * rows_out * 4, ng * rows_out * 4), w.bytes, w.scale, rows_out, cols, ng, off, seq);
            }
        } else {
            try be.opMatmulQuant(w.dtype, y, x, seq, w.bytes, rows_out, cols);
        }
    }
};

const Bufs = struct {
    x: Buf,
    normed: Buf,
    q: Buf,
    k: Buf,
    v: Buf,
    attn: Buf,
    gate: Buf,
    up: Buf,
    t: Buf,
    attn_scratch: Buf,
    logits: Buf,

    fn init(be: *Backend, cfg: gemma3.Config) !Bufs {
        const pc = prefill_chunk; // GEMM outputs are 128-row padded; pc == 128
        comptime std.debug.assert(prefill_chunk == 128);
        const hd = cfg.head_dim;
        var self: Bufs = undefined;
        var created: usize = 0;
        errdefer inline for (@typeInfo(Bufs).@"struct".fields, 0..) |f, i| {
            if (i < created) be.tensorDestroy(&@field(self, f.name));
        };
        const sizes = [_]usize{
            pc * cfg.hidden, // x
            pc * cfg.hidden, // normed
            pc * cfg.qDim(), // q
            pc * cfg.kvDim(), // k
            pc * cfg.kvDim(), // v
            pc * cfg.qDim(), // attn
            pc * cfg.intermediate, // gate
            pc * cfg.intermediate, // up
            pc * cfg.hidden, // t
            @max(cfg.n_heads * nsplit, pc * cfg.n_heads * nsplit_prefill) * (hd + 4), // attn_scratch
            cfg.vocab, // logits
        };
        inline for (@typeInfo(Bufs).@"struct".fields, 0..) |f, i| {
            @field(self, f.name) = try be.tensorCreate(sizes[i] * 4);
            created = i + 1;
        }
        return self;
    }

    fn deinit(self: *Bufs, be: *Backend) void {
        inline for (@typeInfo(Bufs).@"struct".fields) |f| be.tensorDestroy(&@field(self, f.name));
        self.* = undefined;
    }
};
