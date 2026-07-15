//! Gemma 4 text stack on the CUDA backends (cuda / zig-cuda), device-resident.
//! Mirrors gemma4.zig's CPU forward op-for-op; the whole 12B (Q4_0, ~7 GB) fits
//! the 3090, so this MVP is device-resident only — no CPU/GPU layer split,
//! dynamic offload, or weight streaming (those are gemma3_cuda extras, a
//! follow-up). Prefill runs in 128-row chunks; decode is per-op (no graph
//! capture). Ported from gemma4.zig + the gemma3_cuda structure.
//!
//! Gemma 4 specifics vs gemma3_cuda:
//!   - Per-layer attention geometry: LOCAL layers head_dim 256 / 8 KV heads,
//!     GLOBAL layers head_dim 512 / 1 KV head (MQA). Per-layer KV strides;
//!     activation buffers sized for the max (global q/o, local kv).
//!   - Attention score scale is 1.0 (folded into the QK norms), not 1/sqrt(hd).
//!   - GLOBAL layers (head_dim 512) use the generic naive `attn` op (arbitrary
//!     head_dim, full causal); LOCAL layers use opAttnDecode's flash-split
//!     h256 kernel with the sliding-window mask.
//!   - V is RMS-normalized per head_dim with NO weight (a shared ones buffer
//!     through qkNorm); GLOBAL layers have no v_proj, so V reuses the RAW K
//!     projection (copied before k_norm/rope).
//!   - GLOBAL-layer RoPE divides by rope_freqs (proportional RoPE), baked into
//!     the device freqs table at upload.
//!   - Per-layer scalar out_scale (opScale) multiplies the whole layer output.
//!   - Final logits tanh-softcapped + suppress_tokens forced -inf (host-side).

const std = @import("std");
const gemma4 = @import("gemma4.zig");
const qwen3 = @import("qwen3.zig");
const cuda = @import("../gpu/cuda.zig");
const ops = @import("../ops.zig");
const kvmod = @import("../llm/kv_cache.zig");
const transformer = @import("transformer.zig");
const transformer_gpu = @import("transformer_gpu.zig");

const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;
const Growable = Backend.GrowableTensor;

/// KV chunks per head in the local-layer decode attention split.
const nsplit = 32;
const nsplit_prefill = 8;
/// Rows per batched-prefill chunk (also the activation-buffer height).
const prefill_chunk = 128;
const grouped_gemv_max = 40;

fn nbuf(be: *Backend, weights: []const f32) !Buf {
    return .{ .buf = try be.smallBuffer(std.mem.sliceAsBytes(weights)), .mem = .null_handle, .size = 0 };
}

fn offsetBufSized(b: Buf, off_bytes: usize, size: usize) Buf {
    return .{ .buf = @enumFromInt(@intFromEnum(b.buf) + off_bytes), .mem = b.mem, .size = size };
}

pub const CudaLM = struct {
    lm: *const gemma4.Model,
    be: *Backend,
    gpa: std.mem.Allocator,
    cfg: gemma4.Config,
    arena: std.heap.ArenaAllocator,
    capacity: usize,
    initial_capacity: usize,
    max_capacity: usize,
    len: usize,
    /// sin-table offsets within each freqs buffer (= cap.max * half).
    sin_off_global: usize,
    sin_off_local: usize,
    /// Global-layer (theta 1e6 + proportional rope_freqs) and local-layer
    /// (theta 1e4) RoPE tables, cos then sin.
    freqs_global: Buf,
    freqs_local: Buf,
    /// Device ones vector (len head_dim_global), for the weightless V RMS-norm.
    ones: Buf,
    bufs: Bufs,
    /// Per-layer K/V caches (variable stride = kvDim(l)).
    k_cache: []Growable,
    v_cache: []Growable,
    io: std.Io = undefined,
    /// Always null — this MVP is fully device-resident (no CPU/GPU split). The
    /// field + the offload stubs below exist only so tp-gui's arch-generic
    /// offload machinery (autoOffload / promoteLayers / split.n_cpu, etc.)
    /// compiles and treats gemma4 as a permanently-resident model.
    split: ?Split = null,

    pub const CpuSplitPolicy = enum { tail, attn };
    pub const Split = struct { n_cpu: usize = 0, budget: u64 = 0 };

    pub fn init(gpa: std.mem.Allocator, be: *Backend, lm: *const gemma4.Model, cap: kvmod.Capacity) !CudaLM {
        const cfg = lm.cfg;
        switch (lm.embed.dtype) {
            .bf16, .q4_0, .q8_0, .q4_k, .q5_k, .q6_k => {},
            else => return error.UnsupportedModelConfig,
        }

        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();

        var self: CudaLM = undefined;
        // `split` is always null for gemma4 (device-resident MVP, no CPU/GPU
        // split); `self = undefined` bypasses the field's `= null` default, so
        // set it explicitly — llmResidency reads it (GUI status bar).
        self.split = null;
        self.lm = lm;
        self.be = be;
        self.gpa = gpa;
        self.cfg = cfg;
        self.capacity = cap.initial;
        self.initial_capacity = cap.initial;
        self.max_capacity = cap.max;
        self.len = 0;
        self.sin_off_global = cap.max * (cfg.head_dim_global / 2);
        self.sin_off_local = cap.max * (cfg.head_dim_local / 2);

        self.freqs_global = try uploadFreqs(be, gpa, cap.max, cfg.head_dim_global, cfg.rope_theta, 1.0, lm.rope_freqs);
        errdefer be.tensorDestroy(&self.freqs_global);
        self.freqs_local = try uploadFreqs(be, gpa, cap.max, cfg.head_dim_local, cfg.rope_theta_local, 1.0, null);
        errdefer be.tensorDestroy(&self.freqs_local);

        const ones_host = try gpa.alloc(f32, cfg.head_dim_global);
        defer gpa.free(ones_host);
        @memset(ones_host, 1.0);
        self.ones = .{ .buf = try be.smallBuffer(std.mem.sliceAsBytes(ones_host)), .mem = .null_handle, .size = 0 };

        self.bufs = try Bufs.init(be, cfg);

        self.k_cache = try alloc.alloc(Growable, cfg.n_layers);
        self.v_cache = try alloc.alloc(Growable, cfg.n_layers);
        for (self.k_cache, self.v_cache, 0..) |*kb, *vb, l| {
            const kvd = cfg.kvDim(l);
            kb.* = try be.growableCreate(cap.initial * kvd * 4, cap.max * kvd * 4);
            vb.* = try be.growableCreate(cap.initial * kvd * 4, cap.max * kvd * 4);
        }

        self.arena = arena;
        return self;
    }

    /// Build a device RoPE table (cos[rows*half] ++ sin[rows*half]) with an
    /// optional per-dim frequency-factor divisor (global layers).
    fn uploadFreqs(be: *Backend, gpa: std.mem.Allocator, rows: usize, head_dim: usize, theta: f64, freq_scale: f64, factors: ?[]const f32) !Buf {
        const half = head_dim / 2;
        var freqs = try ops.rope.rotateHalfFreqsFactored(gpa, rows, head_dim, theta, freq_scale, factors);
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
    pub fn resetCache(self: *CudaLM) !void {
        self.len = 0;
    }
    pub fn vocab(self: *const CudaLM) usize {
        return self.cfg.vocab;
    }

    // --- Offload interface (device-resident MVP: no-ops) ---------------------
    // gemma4 always loads fully; these satisfy tp-gui's arch-generic residency
    // calls without a CPU/GPU split. The 12B Q4_0 (~7 GB) fits the 3090; a model
    // that didn't fit would need the real split machinery (gemma3_cuda), a
    // follow-up.
    pub fn autoOffload(self: *CudaLM, budget: u64) !bool {
        _ = self;
        _ = budget;
        return false;
    }
    pub fn enableCpuSplit(self: *CudaLM, policy: CpuSplitPolicy, budget: u64, dynamic: bool) !void {
        _ = self;
        _ = policy;
        _ = budget;
        _ = dynamic;
    }
    pub fn offloadUntilFree(self: *CudaLM, needed_free: u64) !void {
        _ = self;
        _ = needed_free;
    }
    pub fn offloadToBudget(self: *CudaLM, target: u64) !void {
        _ = self;
        _ = target;
    }
    pub fn promoteLayers(self: *CudaLM, budget: u64) !usize {
        _ = self;
        _ = budget;
        return 0;
    }
    /// New-chat: drop the context (KV rows are overwritten lazily on the next
    /// prefill, like resetCache; nothing recurrent to zero).
    pub fn resetResidency(self: *CudaLM, budget: u64) !void {
        _ = budget;
        self.len = 0;
    }

    pub fn ensureCapacity(self: *CudaLM, min_rows: usize) !void {
        if (min_rows <= self.capacity) return;
        if (min_rows > self.max_capacity) return error.ContextFull;
        const target = kvmod.growTarget(self.capacity, min_rows, self.max_capacity);
        for (0..self.cfg.n_layers) |l| {
            const bytes = target * self.cfg.kvDim(l) * 4;
            for ([2]*Growable{ &self.k_cache[l], &self.v_cache[l] }) |b| {
                self.be.growableEnsure(b, bytes) catch |err| switch (err) {
                    error.DeviceOutOfMemory, error.OutOfMemory => return error.ContextFull,
                    else => return err,
                };
            }
        }
        self.capacity = target;
    }

    pub fn step(self: *CudaLM, io: std.Io, ids: []const u32, logits: []f32) !void {
        self.io = io;
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

    /// Prefill one image's projected embeddings ([n*hidden], injected UNSCALED)
    /// at the next sequential positions. grid dims carried for interface parity.
    pub fn prefillImage(self: *CudaLM, embeds: []const f32, grid_w: usize, grid_h: usize) !void {
        _ = grid_w;
        _ = grid_h;
        const cfg = self.cfg;
        const total = embeds.len / cfg.hidden;
        var off: usize = 0;
        while (off < total) {
            const n: usize = @min(prefill_chunk, total - off);
            try self.forwardRows(embeds[off * cfg.hidden ..][0 .. n * cfg.hidden], null);
            off += n;
        }
    }

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

    /// One batched forward over `n` pre-embedded rows at positions [len, len+n).
    /// When `logits` is set the last row's final-normed hidden feeds the LM head
    /// (then tanh softcap + suppress-token masking, host-side). Advances len.
    // --- transformer_gpu.decoderLayer stepper methods (faithful lift of the
    // former forwardRows loop). gemma4 has PER-LAYER geometry, so the
    // geometry-sensitive ops take `l` and read cfg.headDim(l)/nKv(l)/qDim(l). ---

    pub fn normInput(self: *CudaLM, layer: anytype, seq: usize) !void {
        const cfg = self.cfg;
        try self.be.qkNorm(self.bufs.x, self.bufs.normed, try nbuf(self.be, layer.input_norm), seq, cfg.hidden, cfg.rms_eps);
    }
    pub fn projectQKV(self: *CudaLM, l: usize, layer: anytype, seq: usize) !void {
        const cfg = self.cfg;
        const b = &self.bufs;
        const kv_dim = cfg.kvDim(l);
        try self.linear(b.q, b.normed, layer.q, cfg.qDim(l), cfg.hidden, seq);
        try self.linear(b.k, b.normed, layer.k, kv_dim, cfg.hidden, seq);
        // V: its own projection, or (global layers) the RAW K projection copied
        // BEFORE k_norm/rope mutate k.
        if (layer.v) |vw| {
            try self.linear(b.v, b.normed, vw, kv_dim, cfg.hidden, seq);
        } else {
            try self.be.tensorCopy(b.v, 0, b.k, 0, seq * kv_dim * 4);
        }
    }
    pub fn normQK(self: *CudaLM, l: usize, layer: anytype, seq: usize) !void {
        const cfg = self.cfg;
        const b = &self.bufs;
        try self.be.qkNorm(b.q, b.q, try nbuf(self.be, layer.q_norm), seq * cfg.n_heads, cfg.headDim(l), cfg.rms_eps);
        try self.be.qkNorm(b.k, b.k, try nbuf(self.be, layer.k_norm), seq * cfg.nKv(l), cfg.headDim(l), cfg.rms_eps);
    }
    pub fn normV(self: *CudaLM, l: usize, seq: usize) !void {
        const cfg = self.cfg;
        // Weightless RMS over head_dim (shared `ones` weight buffer).
        try self.be.qkNorm(self.bufs.v, self.bufs.v, self.ones, seq * cfg.nKv(l), cfg.headDim(l), cfg.rms_eps);
    }
    pub fn applyRope(self: *CudaLM, l: usize, seq: usize, pos0: usize) !void {
        const cfg = self.cfg;
        const b = &self.bufs;
        const global = cfg.isGlobal(l);
        const freqs = if (global) self.freqs_global else self.freqs_local;
        const sin_off = if (global) self.sin_off_global else self.sin_off_local;
        const hd = cfg.headDim(l);
        try self.be.ropeHalf(b.q, freqs, seq, cfg.n_heads, hd / 2, sin_off, pos0);
        try self.be.ropeHalf(b.k, freqs, seq, cfg.nKv(l), hd / 2, sin_off, pos0);
    }
    pub fn appendKV(self: *CudaLM, l: usize, seq: usize, pos0: usize) !void {
        const cfg = self.cfg;
        const b = &self.bufs;
        const kv_dim = cfg.kvDim(l);
        try self.be.tensorCopy(self.k_cache[l].buf, pos0 * kv_dim * 4, b.k, 0, seq * kv_dim * 4);
        try self.be.tensorCopy(self.v_cache[l].buf, pos0 * kv_dim * 4, b.v, 0, seq * kv_dim * 4);
    }
    pub fn attention(self: *CudaLM, l: usize, seq: usize, pos0: usize) !void {
        const cfg = self.cfg;
        const b = &self.bufs;
        const ns: usize = if (seq == 1) nsplit else nsplit_prefill;
        const window: usize = if (cfg.isGlobal(l)) 0 else cfg.sliding_window;
        try self.be.opAttnDecode(b.q, self.k_cache[l].buf, self.v_cache[l].buf, b.attn, b.attn_scratch, pos0 + 1, seq, cfg.n_heads, cfg.nKv(l), cfg.headDim(l), ns, 1.0, window);
    }
    pub fn projectO(self: *CudaLM, l: usize, layer: anytype, seq: usize) !void {
        const cfg = self.cfg;
        try self.linear(self.bufs.t, self.bufs.attn, layer.o, cfg.hidden, cfg.qDim(l), seq);
    }
    pub fn postAttnNorm(self: *CudaLM, layer: anytype, seq: usize) !void {
        const cfg = self.cfg;
        try self.be.qkNorm(self.bufs.t, self.bufs.t, try nbuf(self.be, layer.post_attn_norm), seq, cfg.hidden, cfg.rms_eps);
    }
    pub fn addResidual(self: *CudaLM, seq: usize) !void {
        try self.be.opAdd(self.bufs.x, self.bufs.t, seq * self.cfg.hidden);
    }
    pub fn normPreFfn(self: *CudaLM, layer: anytype, seq: usize) !void {
        const cfg = self.cfg;
        try self.be.qkNorm(self.bufs.x, self.bufs.normed, try nbuf(self.be, layer.pre_ffn_norm), seq, cfg.hidden, cfg.rms_eps);
    }
    pub fn projectGateUp(self: *CudaLM, layer: anytype, seq: usize) !void {
        const cfg = self.cfg;
        const b = &self.bufs;
        try self.linear(b.gate, b.normed, layer.gate, cfg.intermediate, cfg.hidden, seq);
        try self.linear(b.up, b.normed, layer.up, cfg.intermediate, cfg.hidden, seq);
    }
    pub fn activate(self: *CudaLM, comptime act: transformer.Activation, seq: usize) !void {
        const n = seq * self.cfg.intermediate;
        switch (act) {
            .silu_mul => try self.be.siluMul(self.bufs.gate, self.bufs.up, n),
            .gelu_tanh_mul => try self.be.geluMul(self.bufs.gate, self.bufs.up, n),
        }
    }
    pub fn projectDown(self: *CudaLM, layer: anytype, seq: usize) !void {
        const cfg = self.cfg;
        try self.linear(self.bufs.t, self.bufs.gate, layer.down, cfg.hidden, cfg.intermediate, seq);
    }
    pub fn postFfnNorm(self: *CudaLM, layer: anytype, seq: usize) !void {
        const cfg = self.cfg;
        try self.be.qkNorm(self.bufs.t, self.bufs.t, try nbuf(self.be, layer.post_ffn_norm), seq, cfg.hidden, cfg.rms_eps);
    }
    pub fn outScale(self: *CudaLM, layer: anytype, seq: usize) !void {
        if (layer.out_scale != 1.0) try self.be.opScale(self.bufs.x, layer.out_scale, seq * self.cfg.hidden);
    }

    fn forwardRows(self: *CudaLM, x_host: []const f32, logits: ?[]f32) !void {
        const be = self.be;
        const cfg = self.cfg;
        const b = &self.bufs;
        const n = x_host.len / cfg.hidden;
        const eps = cfg.rms_eps;
        const pos0 = self.len;
        std.debug.assert(n >= 1 and n <= prefill_chunk and n <= self.remaining());

        try be.tensorUpload(offsetBufSized(b.x, 0, n * cfg.hidden * 4), std.mem.sliceAsBytes(x_host));

        try be.beginBatch();
        errdefer if (be.batching()) be.abortBatch();

        for (self.lm.layers, 0..) |*layer, l| {
            try transformer_gpu.decoderLayer(transformer.gemma4_spec, self, layer, l, n, pos0);
        }

        if (logits) |out| {
            const h = cfg.hidden;
            try be.qkNorm(offsetBufSized(b.x, (n - 1) * h * 4, h * 4), b.t, try nbuf(be, self.lm.final_norm), 1, h, eps);
            try self.lmHead(b.logits, b.t);
            try be.endBatch();
            self.len += n;
            try be.tensorDownload(offsetBufSized(b.logits, 0, cfg.vocab * 4), std.mem.sliceAsBytes(out));
            self.lm.finalizeLogits(out); // tanh softcap + suppress tokens
        } else {
            try be.endBatch();
            self.len += n;
        }
    }

    fn lmHead(self: *CudaLM, y: Buf, x: Buf) !void {
        const head = self.lm.head;
        try self.be.opGemvQuant(head.dtype, y, x, head.bytes, head.scale, self.cfg.vocab, self.cfg.hidden);
    }

    /// Dense linear over `seq` rows (int8 dp4a GEMV / grouped GEMV / dequant
    /// tensor-core GEMM), mirroring gemma3_cuda.linear. All Gemma weights are
    /// GGUF block quants.
    fn linear(self: *CudaLM, y: Buf, x: Buf, w: ops.matmul.Weight, rows_out: usize, cols: usize, seq: usize) !void {
        const be = self.be;
        std.debug.assert(w.dtype.isBlockQuant());
        const dp4a_ok = cols % 256 == 0 and rows_out % 8 == 0;
        // q4_0 (the 12B QAT format): decode uses the dp4a int8-activation GEMV
        // (gemv_q4_0_q8n, quantized activation × nibble weight) when tileable,
        // else the fused f32 weight-read-once GEMV; prefill batches use the
        // dequant-to-f16 tensor-core GEMM.
        if (w.dtype == .q4_0) {
            if (seq == 1 and dp4a_ok) {
                try be.opGemvQuantizeX(x, cols);
                try be.opGemvQuantQ8N(w.dtype, y, w.bytes, w.scale, rows_out, cols, 1, 0, 1);
                return;
            }
            if (seq == 1) return be.opGemvQuant(w.dtype, y, x, w.bytes, w.scale, rows_out, cols);
            return be.opMatmulQuant(w.dtype, y, x, seq, w.bytes, rows_out, cols);
        }
        if (seq == 1) {
            if (!dp4a_ok) {
                try be.opGemvQuant(w.dtype, y, x, w.bytes, w.scale, rows_out, cols);
            } else {
                try be.opGemvQuantizeX(x, cols);
                if (w.dtype == .q5_k or w.dtype == .q6_k) {
                    try be.opGemvQuantQ8(w.dtype, y, w.bytes, w.scale, rows_out, cols);
                } else {
                    try be.opGemvQuantQ8N(w.dtype, y, w.bytes, w.scale, rows_out, cols, 1, 0, 1);
                }
            }
        } else if (seq <= grouped_gemv_max and dp4a_ok) {
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

    fn init(be: *Backend, cfg: gemma4.Config) !Bufs {
        const pc = prefill_chunk;
        comptime std.debug.assert(prefill_chunk == 128);
        // The flash-split scratch row is (hd+4) f32; size it for the larger
        // (global) head_dim so both local and global attention fit.
        const hd_max = @max(cfg.head_dim_global, cfg.head_dim_local);
        var self: Bufs = undefined;
        var created: usize = 0;
        errdefer inline for (@typeInfo(Bufs).@"struct".fields, 0..) |f, i| {
            if (i < created) be.tensorDestroy(&@field(self, f.name));
        };
        const sizes = [_]usize{
            pc * cfg.hidden, // x
            pc * cfg.hidden, // normed
            pc * cfg.maxQDim(), // q
            pc * cfg.maxKvDim(), // k
            pc * cfg.maxKvDim(), // v
            pc * cfg.maxQDim(), // attn
            pc * cfg.intermediate, // gate
            pc * cfg.intermediate, // up
            pc * cfg.hidden, // t
            @max(cfg.n_heads * nsplit, pc * cfg.n_heads * nsplit_prefill) * (hd_max + 4), // attn_scratch
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
