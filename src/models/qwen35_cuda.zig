//! Qwen3.5/3.6 hybrid LM on the CUDA backend (tp-llm --backend zig-cuda /
//! cuda): the 64-layer gated-DeltaNet + gated-attention stack runs
//! device-resident. Decode quantizes each activation to int8 once
//! (opGemvQuantizeX) and runs dp4a GEMVs in the GGUF block-quant dtype
//! (opGemvQuantQ8 for q5_k/q6_k, opGemvQuant otherwise); after the first
//! decode step the whole forward replays as one captured CUDA graph
//! (stepDecodeGraph). Prefill runs batched 128-row chunks (stepBatch).
//! The 27B Q5_K_M fits a 24 GB card resident. Speculative decoding is
//! unsupported (recurrent state cannot roll back).

const std = @import("std");
const qwen35 = @import("qwen35.zig");
const qwen3 = @import("qwen3.zig");
const cuda = @import("../gpu/cuda.zig");
const ops = @import("../ops.zig");

const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;

fn nbuf(be: *Backend, weights: []const f32) !Buf {
    return .{ .buf = try be.smallBuffer(std.mem.sliceAsBytes(weights)), .mem = .null_handle, .size = 0 };
}

fn offsetBufSized(b: Buf, off_bytes: usize, size: usize) Buf {
    return .{ .buf = @enumFromInt(@intFromEnum(b.buf) + off_bytes), .mem = b.mem, .size = size };
}

/// KV chunks per head in the decode attention split pass.
const nsplit = 32;
/// Batched-prefill chunk (rows per stepBatch) and its attention split count
/// (bounds the flash-decode scratch).
const prefill_chunk = 128;
const nsplit_prefill = 8;

pub const CudaLM = struct {
    lm: *const qwen35.Model,
    be: *Backend,
    gpa: std.mem.Allocator,
    cfg: qwen35.Config,
    capacity: usize,
    len: usize,

    bufs: Bufs,
    /// Per-attention-slot KV caches, [capacity][kvDim] f32.
    k_cache: []Buf,
    v_cache: []Buf,
    /// [n_lin][channels][kernel-1] rolling conv tails.
    conv_state: Buf,
    /// [n_lin][heads][d][d] delta-rule states.
    ssm_state: Buf,
    freqs_d: Buf,
    sin_off: usize,
    /// Per-step (t, h, w) M-RoPE position triple, uploaded before each token.
    pos3_d: Buf,
    /// Per-row position triples for batched prefill ([prefill_chunk][3]).
    pos3s_d: Buf,
    /// Next M-RoPE position (equals `len` for text-only sessions; images
    /// advance it by max(grid_w, grid_h) while occupying grid_w*grid_h rows).
    pos_next: usize,
    /// Per-linear-layer [a(heads) | dt(heads)] host constants (arena of the
    /// stepper), uploaded via the small-buffer cache.
    a_dt: [][]f32,
    /// TP_DEBUG_BATCH: per-layer hidden-state dump of the last processed row
    /// ([n_layers][hidden]), filled by stepHidden/stepBatch when set.
    layer_dump: ?[]f32 = null,
    /// TP debug: layer-0 input_layernorm output, one hidden row per
    /// processed token (seq appends at op_dump_row; batch writes all rows).
    op_dump: ?[]f32 = null,
    op_dump_row: usize = 0,
    /// Which activation (and its width) the backend q8 scratch currently
    /// holds — gemv() asserts the dp4a path reads what quantizeX staged.
    q8_for: Buf = .{},
    q8_cols: usize = 0,
    /// Captured decode step (CUDA graph): one launch replays the whole
    /// forward, with {token, len} read from device state and the M-RoPE
    /// triple from pos3_d. Null until the second single-token step.
    graph_exec: cuda.cu.CUgraphExec = null,
    /// First single-token decode ran (capture is safe now).
    decode_warm: bool = false,
    /// Cleared permanently if capture fails or weights evict.
    graph_ok: bool = true,
    arena: std.heap.ArenaAllocator,

    pub fn init(gpa: std.mem.Allocator, be: *Backend, lm: *const qwen35.Model, capacity: usize) !CudaLM {
        const cfg = lm.cfg;
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();

        var self: CudaLM = undefined;
        self.lm = lm;
        self.be = be;
        self.gpa = gpa;
        self.cfg = cfg;
        self.capacity = capacity;
        self.len = 0;
        self.pos_next = 0;
        self.layer_dump = null;
        self.op_dump = null;
        self.op_dump_row = 0;
        self.q8_for = .{};
        self.q8_cols = 0;
        self.graph_exec = null;
        self.decode_warm = false;
        // The graph path needs a device-side embedding gather kernel.
        self.graph_ok = switch (lm.embed.dtype) {
            .bf16, .q8_0, .q4_k, .q5_k, .q6_k => true,
            else => false,
        };

        // Rope table for the rotated span (rope_dim), like qwen3_cuda's.
        var freqs = try ops.rope.rotateHalfFreqs(gpa, capacity, cfg.rope_dim, cfg.rope_theta);
        defer freqs.deinit(gpa);
        const half = cfg.rope_dim / 2;
        const fp = try gpa.alloc(f32, 2 * capacity * half);
        defer gpa.free(fp);
        @memcpy(fp[0 .. capacity * half], freqs.cos);
        @memcpy(fp[capacity * half ..], freqs.sin);
        self.sin_off = capacity * half;
        self.freqs_d = try be.tensorCreate(fp.len * 4);
        try be.tensorUpload(self.freqs_d, std.mem.sliceAsBytes(fp));
        self.pos3_d = try be.tensorCreate(3 * 4);
        self.pos3s_d = try be.tensorCreate(prefill_chunk * 3 * 4);

        self.bufs = try Bufs.init(be, cfg);

        const n_attn = cfg.nAttnLayers();
        self.k_cache = try alloc.alloc(Buf, n_attn);
        self.v_cache = try alloc.alloc(Buf, n_attn);
        for (self.k_cache, self.v_cache) |*kb, *vb| {
            kb.* = try be.tensorCreate(capacity * cfg.kvDim() * 4);
            vb.* = try be.tensorCreate(capacity * cfg.kvDim() * 4);
        }

        const n_lin = cfg.n_layers - n_attn;
        const conv_bytes = n_lin * cfg.convChannels() * (cfg.conv_kernel - 1) * 4;
        const ssm_bytes = n_lin * cfg.lin_v_heads * cfg.lin_head_dim * cfg.lin_head_dim * 4;
        self.conv_state = try be.tensorCreate(conv_bytes);
        self.ssm_state = try be.tensorCreate(ssm_bytes);
        try zeroBuffer(be, gpa, self.conv_state, conv_bytes);
        try zeroBuffer(be, gpa, self.ssm_state, ssm_bytes);

        // Concatenated [a | dt_bias] per linear layer for gdn_gates.
        self.a_dt = try alloc.alloc([]f32, n_lin);
        var lin_idx: usize = 0;
        for (lm.layers) |*layer| {
            switch (layer.*) {
                .linear => |*ll| {
                    const buf = try alloc.alloc(f32, 2 * cfg.lin_v_heads);
                    @memcpy(buf[0..cfg.lin_v_heads], ll.a);
                    @memcpy(buf[cfg.lin_v_heads..], ll.dt_bias);
                    self.a_dt[lin_idx] = buf;
                    lin_idx += 1;
                },
                .attn => {},
            }
        }

        // Assigned last: `alloc` mutates the stack `arena`; copying its state
        // any earlier would snapshot an empty arena and leak the chunks.
        self.arena = arena;
        return self;
    }

    pub fn deinit(self: *CudaLM) void {
        const be = self.be;
        if (self.graph_exec != null) be.graphDestroy(self.graph_exec);
        for (self.k_cache) |*b| be.tensorDestroy(b);
        for (self.v_cache) |*b| be.tensorDestroy(b);
        be.tensorDestroy(&self.conv_state);
        be.tensorDestroy(&self.ssm_state);
        be.tensorDestroy(&self.freqs_d);
        be.tensorDestroy(&self.pos3_d);
        be.tensorDestroy(&self.pos3s_d);
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

    pub fn vocab(self: *const CudaLM) usize {
        return self.cfg.vocab;
    }

    /// Debug: reset the session (cache rows are overwritten lazily; conv and
    /// recurrent states must be re-zeroed).
    pub fn debugReset(self: *CudaLM) !void {
        const cfg = self.cfg;
        const n_lin = cfg.n_layers - cfg.nAttnLayers();
        try zeroBuffer(self.be, self.gpa, self.conv_state, n_lin * cfg.convChannels() * (cfg.conv_kernel - 1) * 4);
        try zeroBuffer(self.be, self.gpa, self.ssm_state, n_lin * cfg.lin_v_heads * cfg.lin_head_dim * cfg.lin_head_dim * 4);
        self.len = 0;
        self.pos_next = 0;
    }

    /// Forward `ids_new` (batched prefill for all but the last token, one
    /// decode step for it); `logits` receives the last position's LM head.
    /// Single-token decode replays a captured CUDA graph (one launch instead
    /// of ~1700) once the first decode step has warmed weight residency;
    /// --profile and capture failures fall back to per-op launches.
    pub fn step(self: *CudaLM, io: std.Io, ids_new: []const u32, logits: []f32) !void {
        _ = io;
        const be = self.be;
        std.debug.assert(ids_new.len >= 1 and ids_new.len <= self.remaining());
        std.debug.assert(logits.len == self.cfg.vocab);
        // Any weight eviction (--vram-budget streaming, or live VRAM
        // pressure) means device weight pointers are not stable, and a
        // captured graph would replay against freed buffers.
        if (be.evictions != 0) self.graph_ok = false;
        if (ids_new.len == 1 and self.graph_ok and !be.profile and self.decode_warm) {
            try self.stepDecodeGraph(ids_new[0]);
        } else {
            if (ids_new.len > 1) try self.prefill(ids_new[0 .. ids_new.len - 1]);
            try self.stepOne(ids_new[ids_new.len - 1], true);
            self.decode_warm = true;
        }
        try be.tensorDownload(offsetBufSized(self.bufs.logits, 0, self.cfg.vocab * 4), std.mem.sliceAsBytes(logits));
    }

    /// Prefill tokens without reading logits, in batched chunks.
    pub fn prefill(self: *CudaLM, ids: []const u32) !void {
        const cfg = self.cfg;
        const gpa = self.gpa;
        const x = try gpa.alloc(f32, @min(ids.len, prefill_chunk) * cfg.hidden);
        defer gpa.free(x);
        var pos3s: [prefill_chunk * 3]u32 = undefined;
        var off: usize = 0;
        while (off < ids.len) {
            // usize annotation matters: @min with a comptime_int bound yields
            // a range-narrowed type (u7 here), and `n * 3` would overflow it.
            // See ZIG.md "@min/@max narrow their result type".
            const n: usize = @min(prefill_chunk, ids.len - off);
            try qwen3.embedTokens(self.lm.embed, ids[off..][0..n], x[0 .. n * cfg.hidden]);
            for (0..n) |t| {
                const p: u32 = @intCast(self.pos_next + t);
                pos3s[t * 3 + 0] = p;
                pos3s[t * 3 + 1] = p;
                pos3s[t * 3 + 2] = p;
            }
            try self.stepBatch(x[0 .. n * cfg.hidden], pos3s[0 .. n * 3]);
            self.pos_next += n;
            off += n;
        }
    }

    /// Prefill one image: `embeds` is the ViT output ([grid_w*grid_h rows of
    /// hidden]), injected in batched chunks with M-RoPE grid positions (t
    /// fixed at the image start, h/w = merged-grid row/col); the next text
    /// position continues at start + max(grid_w, grid_h) (mtmd semantics).
    pub fn prefillImage(self: *CudaLM, embeds: []const f32, grid_w: usize, grid_h: usize) !void {
        const cfg = self.cfg;
        std.debug.assert(embeds.len == grid_w * grid_h * cfg.hidden);
        std.debug.assert(grid_w * grid_h <= self.remaining());
        const pos0: u32 = @intCast(self.pos_next);
        if (debug_seq_image) {
            for (0..grid_w * grid_h) |i| {
                const pos3 = [3]u32{
                    pos0,
                    pos0 + @as(u32, @intCast(i / grid_w)),
                    pos0 + @as(u32, @intCast(i % grid_w)),
                };
                try self.stepHidden(embeds[i * cfg.hidden ..][0..cfg.hidden], pos3, false);
            }
            self.pos_next = pos0 + @max(grid_w, grid_h);
            return;
        }
        var pos3s: [prefill_chunk * 3]u32 = undefined;
        var off: usize = 0;
        const total = grid_w * grid_h;
        while (off < total) {
            // usize annotation: see prefill (@min range-narrowing, ZIG.md).
            const n: usize = @min(@min(debug_image_chunk, prefill_chunk), total - off);
            for (0..n) |t| {
                const i = off + t;
                pos3s[t * 3 + 0] = pos0;
                pos3s[t * 3 + 1] = pos0 + @as(u32, @intCast(i / grid_w)); // h = merged row
                pos3s[t * 3 + 2] = pos0 + @as(u32, @intCast(i % grid_w)); // w = merged col
            }
            try self.stepBatch(embeds[off * cfg.hidden ..][0 .. n * cfg.hidden], pos3s[0 .. n * 3]);
            off += n;
        }
        self.pos_next = pos0 + @max(grid_w, grid_h);
    }

    /// Batched prefill step over `n` embedded rows with per-row positions:
    /// projections and MLPs run as tensor-core GEMMs, attention as a
    /// seq_q=n flash-decode batch; the DeltaNet conv/recurrence stays
    /// sequential per token (it is inherently serial across time).
    fn stepBatch(self: *CudaLM, x_host: []const f32, pos3s: []const u32) !void {
        const be = self.be;
        const cfg = self.cfg;
        const b = &self.bufs;
        const n = pos3s.len / 3;
        const hd = cfg.head_dim;
        const eps = cfg.rms_eps;
        std.debug.assert(n >= 1 and n <= prefill_chunk and x_host.len == n * cfg.hidden);
        std.debug.assert(n <= self.remaining());

        try be.tensorUpload(offsetBufSized(b.x, 0, n * cfg.hidden * 4), std.mem.sliceAsBytes(x_host));
        try be.tensorUpload(offsetBufSized(self.pos3s_d, 0, n * 3 * 4), std.mem.sliceAsBytes(pos3s));

        try be.beginBatch();
        errdefer if (be.batching()) be.abortBatch();

        for (self.lm.layers, 0..) |*layer, l| {
            switch (layer.*) {
                .attn => |*al| {
                    const slot = l / cfg.full_attn_interval;
                    try be.qkNorm(b.x, b.normed, try nbuf(be, al.input_norm), n, cfg.hidden, eps);
                    try self.gemm(b.qg, b.normed, al.qg, n);
                    try self.gemm(b.k, b.normed, al.k, n);
                    try self.gemm(b.v, b.normed, al.v, n);
                    try be.opDeinterleave2(b.qg, b.q, b.gate, n * cfg.qDim(), hd);
                    try be.qkNorm(b.q, b.q, try nbuf(be, al.q_norm), n * cfg.n_heads, hd, eps);
                    try be.qkNorm(b.k, b.k, try nbuf(be, al.k_norm), n * cfg.n_kv_heads, hd, eps);
                    try be.opRopeImropePos(b.q, self.pos3s_d, self.freqs_d, n, cfg.n_heads, cfg.rope_dim / 2, self.sin_off, cfg.rope_sections, hd);
                    try be.opRopeImropePos(b.k, self.pos3s_d, self.freqs_d, n, cfg.n_kv_heads, cfg.rope_dim / 2, self.sin_off, cfg.rope_sections, hd);
                    try be.tensorCopy(self.k_cache[slot], self.len * cfg.kvDim() * 4, b.k, 0, n * cfg.kvDim() * 4);
                    try be.tensorCopy(self.v_cache[slot], self.len * cfg.kvDim() * 4, b.v, 0, n * cfg.kvDim() * 4);
                    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
                    if (debug_seq_attn) {
                        for (0..n) |t| {
                            try be.opAttnDecode(
                                offsetBufSized(b.q, t * cfg.qDim() * 4, cfg.qDim() * 4),
                                self.k_cache[slot],
                                self.v_cache[slot],
                                offsetBufSized(b.attn, t * cfg.qDim() * 4, cfg.qDim() * 4),
                                b.attn_scratch,
                                self.len + 1 + t,
                                1,
                                cfg.n_heads,
                                cfg.n_kv_heads,
                                hd,
                                nsplit,
                                scale,
                            );
                        }
                    } else {
                        try be.opAttnDecode(b.q, self.k_cache[slot], self.v_cache[slot], b.attn, b.attn_scratch, self.len + 1, n, cfg.n_heads, cfg.n_kv_heads, hd, nsplit_prefill, scale);
                    }
                    try be.opMulSigmoid(b.attn, b.gate, n * cfg.qDim());
                    try self.gemm(b.t, b.attn, al.o, n);
                    try be.opAdd(b.x, b.t, n * cfg.hidden);
                },
                .linear => |*ll| {
                    const lin_idx = l - l / cfg.full_attn_interval;
                    const channels = cfg.convChannels();
                    const d = cfg.lin_head_dim;
                    const heads = cfg.lin_v_heads;
                    try be.qkNorm(b.x, b.normed, try nbuf(be, ll.input_norm), n, cfg.hidden, eps);
                    if (self.op_dump) |od| {
                        if (l == 0) try be.tensorDownload(offsetBufSized(b.normed, 0, n * cfg.hidden * 4), std.mem.sliceAsBytes(od[0 .. n * cfg.hidden]));
                    }
                    try self.gemm(b.lin_qkv, b.normed, ll.qkv, n);
                    try self.gemm(b.lin_z, b.normed, ll.z, n);
                    const conv_off = lin_idx * channels * (cfg.conv_kernel - 1) * 4;
                    const conv_state = offsetBufSized(self.conv_state, conv_off, channels * (cfg.conv_kernel - 1) * 4);
                    const ssm_off = lin_idx * heads * d * d * 4;
                    const ssm = offsetBufSized(self.ssm_state, ssm_off, heads * d * d * 4);
                    for (0..n) |t| {
                        const normed_t = offsetBufSized(b.normed, t * cfg.hidden * 4, cfg.hidden * 4);
                        try self.quantizeX(normed_t, cfg.hidden);
                        try self.gemv(offsetBufSized(b.ab, 0, heads * 4), normed_t, ll.alpha);
                        try self.gemv(offsetBufSized(b.ab, heads * 4, heads * 4), normed_t, ll.beta);
                        try be.opGdnGates(b.ab, try nbuf(be, self.a_dt[lin_idx]), b.gates, heads);
                        try be.opGdnConvStep(
                            conv_state,
                            offsetBufSized(b.lin_qkv, t * channels * 4, channels * 4),
                            try nbuf(be, ll.conv_w),
                            b.lin_conv,
                            channels,
                        );
                        try be.opL2NormRows(offsetBufSized(b.lin_conv, 0, 2 * cfg.linQKDim() * 4), 2 * cfg.lin_k_heads, d, eps);
                        try be.opGdnDeltaStep(
                            ssm,
                            b.lin_conv,
                            b.gates,
                            offsetBufSized(b.lin_o, t * cfg.linVDim() * 4, cfg.linVDim() * 4),
                            heads,
                            d,
                            cfg.lin_k_heads,
                            1.0 / @sqrt(@as(f32, @floatFromInt(d))),
                        );
                    }
                    try be.qkNorm(b.lin_o, b.lin_o, try nbuf(be, ll.ssm_norm), n * heads, d, eps);
                    try be.siluMul(b.lin_z, b.lin_o, n * cfg.linVDim());
                    try self.gemm(b.t, b.lin_z, ll.out, n);
                    try be.opAdd(b.x, b.t, n * cfg.hidden);
                },
            }
            const mlp = switch (layer.*) {
                .attn => |*al| &al.mlp,
                .linear => |*ll| &ll.mlp,
            };
            try be.qkNorm(b.x, b.normed, try nbuf(be, mlp.post_norm), n, cfg.hidden, eps);
            try self.gemm(b.mlp_gate, b.normed, mlp.gate, n);
            try self.gemm(b.mlp_up, b.normed, mlp.up, n);
            try be.siluMul(b.mlp_gate, b.mlp_up, n * cfg.intermediate);
            try self.gemm(b.t, b.mlp_gate, mlp.down, n);
            try be.opAdd(b.x, b.t, n * cfg.hidden);
            if (self.layer_dump) |dump| {
                try be.tensorDownload(
                    offsetBufSized(b.x, (n - 1) * cfg.hidden * 4, cfg.hidden * 4),
                    std.mem.sliceAsBytes(dump[l * cfg.hidden ..][0..cfg.hidden]),
                );
            }
        }
        try be.endBatch();
        self.len += n;
    }

    /// Prefill chunks up to this many rows take the grouped dp4a GEMV
    /// (weight streamed ceil(n/8) times) instead of opMatmulQuant's full
    /// dequant-to-f16 GEMM. Measured on the 3090 27B: one grouped pass is
    /// ~165 us vs ~0.92 ms per weight per chunk for dequant+hgemm
    /// (n-independent), so the crossover sits near n = 44 — grouped wins
    /// 3-6x on chat-turn-sized chunks, GEMM wins on full 128-row chunks.
    const grouped_prefill_max = 40;

    /// GEMV for one row, grouped dp4a GEMV for small batches,
    /// dequant-to-f16 tensor-core GEMM beyond.
    fn gemm(self: *CudaLM, y: Buf, x: Buf, w: ops.matmul.Weight, n: usize) !void {
        if (n == 1) {
            try self.quantizeX(x, w.cols);
            return self.gemv(y, x, w);
        }
        if (debug_gemv_prefill) {
            for (0..n) |t| {
                const x_t = offsetBufSized(x, t * w.cols * 4, w.cols * 4);
                try self.quantizeX(x_t, w.cols);
                try self.gemv(offsetBufSized(y, t * w.rows * 4, w.rows * 4), x_t, w);
            }
            return;
        }
        if ((w.dtype == .q5_k or w.dtype == .q6_k) and n <= grouped_prefill_max) {
            try self.quantizeX(x, n * w.cols);
            var off: usize = 0;
            while (off < n) : (off += 8) {
                // usize annotation: @min range-narrows (ZIG.md).
                const ng: usize = @min(8, n - off);
                try self.be.opGemvQuantQ8N(w.dtype, offsetBufSized(y, off * w.rows * 4, ng * w.rows * 4), w.bytes, w.scale, w.rows, w.cols, ng, off, n);
            }
            return;
        }
        try self.be.opMatmulQuant(w.dtype, y, x, n, w.bytes, w.rows, w.cols);
    }

    /// Debug escape hatches for bisecting the batched-prefill path.
    const debug_gemv_prefill = false;
    const debug_seq_attn = false;
    const debug_seq_image = false;
    const debug_image_chunk: usize = prefill_chunk;

    /// Debug: one sequential token step (no logits).
    pub fn debugStepOne(self: *CudaLM, id: u32) !void {
        try self.stepOne(id, false);
    }

    fn stepOne(self: *CudaLM, id: u32, want_logits: bool) !void {
        const cfg = self.cfg;
        var x_host: [8192]f32 = undefined;
        try qwen3.embedTokens(self.lm.embed, &.{id}, x_host[0..cfg.hidden]);
        const p: u32 = @intCast(self.pos_next);
        try self.stepHidden(x_host[0..cfg.hidden], .{ p, p, p }, want_logits);
        self.pos_next += 1;
    }

    /// One decode step over an already-embedded hidden row at the given
    /// (t, h, w) M-RoPE positions.
    fn stepHidden(self: *CudaLM, x_host: []const f32, pos3: [3]u32, want_logits: bool) !void {
        const be = self.be;
        try be.tensorUpload(offsetBufSized(self.bufs.x, 0, self.cfg.hidden * 4), std.mem.sliceAsBytes(x_host));
        try be.tensorUpload(self.pos3_d, std.mem.sliceAsBytes(&pos3));

        try be.beginBatch();
        errdefer if (be.batching()) be.abortBatch();
        try self.decodeBody(false, want_logits);
        try be.endBatch();
        self.len += 1;
    }

    /// The single-token decode forward shared by the per-op path
    /// (stepHidden) and the captured-graph recording: identical kernels and
    /// order, except graph mode appends KV rows and reads the attention
    /// length via g_state[1] (the M-RoPE triple comes from pos3_d either
    /// way), and skips the host-download debug taps. Does not bump len.
    fn decodeBody(self: *CudaLM, comptime graph: bool, want_logits: bool) !void {
        const be = self.be;
        const cfg = self.cfg;
        const b = &self.bufs;
        const hd = cfg.head_dim;
        const eps = cfg.rms_eps;

        for (self.lm.layers, 0..) |*layer, l| {
            switch (layer.*) {
                .attn => |*al| {
                    const slot = l / cfg.full_attn_interval;
                    try be.qkNorm(b.x, b.normed, try nbuf(be, al.input_norm), 1, cfg.hidden, eps);
                    try self.quantizeX(b.normed, cfg.hidden);
                    try self.gemv(b.qg, b.normed, al.qg);
                    try self.gemv(b.k, b.normed, al.k);
                    try self.gemv(b.v, b.normed, al.v);
                    try be.opDeinterleave2(b.qg, b.q, b.gate, cfg.qDim(), hd);
                    try be.qkNorm(b.q, b.q, try nbuf(be, al.q_norm), cfg.n_heads, hd, eps);
                    try be.qkNorm(b.k, b.k, try nbuf(be, al.k_norm), cfg.n_kv_heads, hd, eps);
                    try be.opRopeImrope(b.q, self.pos3_d, self.freqs_d, cfg.n_heads, cfg.rope_dim / 2, self.sin_off, cfg.rope_sections, hd);
                    try be.opRopeImrope(b.k, self.pos3_d, self.freqs_d, cfg.n_kv_heads, cfg.rope_dim / 2, self.sin_off, cfg.rope_sections, hd);
                    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
                    if (graph) {
                        try be.opKvAppendS(self.k_cache[slot], b.k, cfg.kvDim(), cfg.kvDim(), 0);
                        try be.opKvAppendS(self.v_cache[slot], b.v, cfg.kvDim(), cfg.kvDim(), 0);
                        try be.opAttnDecodeSGraph(b.q, self.k_cache[slot], self.v_cache[slot], b.attn, b.attn_scratch, cfg.n_heads, cfg.n_kv_heads, hd, nsplit, scale);
                    } else {
                        try be.tensorCopy(self.k_cache[slot], self.len * cfg.kvDim() * 4, b.k, 0, cfg.kvDim() * 4);
                        try be.tensorCopy(self.v_cache[slot], self.len * cfg.kvDim() * 4, b.v, 0, cfg.kvDim() * 4);
                        try be.opAttnDecode(b.q, self.k_cache[slot], self.v_cache[slot], b.attn, b.attn_scratch, self.len + 1, 1, cfg.n_heads, cfg.n_kv_heads, hd, nsplit, scale);
                    }
                    try be.opMulSigmoid(b.attn, b.gate, cfg.qDim());
                    try self.quantizeX(b.attn, cfg.qDim());
                    try self.gemv(b.t, b.attn, al.o);
                    try be.opAdd(b.x, b.t, cfg.hidden);
                },
                .linear => |*ll| {
                    const lin_idx = l - l / cfg.full_attn_interval;
                    const channels = cfg.convChannels();
                    const d = cfg.lin_head_dim;
                    const heads = cfg.lin_v_heads;
                    try be.qkNorm(b.x, b.normed, try nbuf(be, ll.input_norm), 1, cfg.hidden, eps);
                    if (!graph) {
                        if (self.op_dump) |od| {
                            if (l == 0) {
                                try be.tensorDownload(offsetBufSized(b.normed, 0, cfg.hidden * 4), std.mem.sliceAsBytes(od[self.op_dump_row * cfg.hidden ..][0..cfg.hidden]));
                                self.op_dump_row += 1;
                            }
                        }
                    }
                    try self.quantizeX(b.normed, cfg.hidden);
                    try self.gemv(b.lin_qkv, b.normed, ll.qkv);
                    try self.gemv(b.lin_z, b.normed, ll.z);
                    try self.gemv(offsetBufSized(b.ab, 0, heads * 4), b.normed, ll.alpha);
                    try self.gemv(offsetBufSized(b.ab, heads * 4, heads * 4), b.normed, ll.beta);
                    try be.opGdnGates(b.ab, try nbuf(be, self.a_dt[lin_idx]), b.gates, heads);
                    const conv_off = lin_idx * channels * (cfg.conv_kernel - 1) * 4;
                    try be.opGdnConvStep(
                        offsetBufSized(self.conv_state, conv_off, channels * (cfg.conv_kernel - 1) * 4),
                        b.lin_qkv,
                        try nbuf(be, ll.conv_w),
                        b.lin_conv,
                        channels,
                    );
                    try be.opL2NormRows(offsetBufSized(b.lin_conv, 0, 2 * cfg.linQKDim() * 4), 2 * cfg.lin_k_heads, d, eps);
                    const ssm_off = lin_idx * heads * d * d * 4;
                    try be.opGdnDeltaStep(
                        offsetBufSized(self.ssm_state, ssm_off, heads * d * d * 4),
                        b.lin_conv,
                        b.gates,
                        b.lin_o,
                        heads,
                        d,
                        cfg.lin_k_heads,
                        1.0 / @sqrt(@as(f32, @floatFromInt(d))),
                    );
                    try be.qkNorm(b.lin_o, b.lin_o, try nbuf(be, ll.ssm_norm), heads, d, eps);
                    try be.siluMul(b.lin_z, b.lin_o, cfg.linVDim());
                    try self.quantizeX(b.lin_z, cfg.linVDim());
                    try self.gemv(b.t, b.lin_z, ll.out);
                    try be.opAdd(b.x, b.t, cfg.hidden);
                },
            }
            const mlp = switch (layer.*) {
                .attn => |*al| &al.mlp,
                .linear => |*ll| &ll.mlp,
            };
            try be.qkNorm(b.x, b.normed, try nbuf(be, mlp.post_norm), 1, cfg.hidden, eps);
            try self.quantizeX(b.normed, cfg.hidden);
            try self.gemv(b.mlp_gate, b.normed, mlp.gate);
            try self.gemv(b.mlp_up, b.normed, mlp.up);
            try be.siluMul(b.mlp_gate, b.mlp_up, cfg.intermediate);
            try self.quantizeX(b.mlp_gate, cfg.intermediate);
            try self.gemv(b.t, b.mlp_gate, mlp.down);
            try be.opAdd(b.x, b.t, cfg.hidden);
            if (!graph) {
                if (self.layer_dump) |dump| {
                    try be.tensorDownload(
                        offsetBufSized(b.x, 0, cfg.hidden * 4),
                        std.mem.sliceAsBytes(dump[l * cfg.hidden ..][0..cfg.hidden]),
                    );
                }
            }
        }

        if (want_logits) {
            try be.qkNorm(b.x, b.t, try nbuf(be, self.lm.final_norm), 1, cfg.hidden, eps);
            const head = self.lm.head;
            if (head.dtype == .q5_k or head.dtype == .q6_k) {
                try self.quantizeX(b.t, cfg.hidden);
                try be.opGemvQuantQ8(head.dtype, b.logits, head.bytes, 1.0, cfg.vocab, cfg.hidden);
            } else {
                try be.opGemvQuant(head.dtype, b.logits, b.t, head.bytes, 1.0, cfg.vocab, cfg.hidden);
            }
        }
    }

    /// Single-token decode as one captured-graph replay: {token, len} land
    /// in g_state and the M-RoPE triple in pos3_d before the launch (the
    /// graph's kernels read both). Captured on the second decode step — the
    /// first (per-op) step warms weight residency, JIT, and the activation
    /// scratch sizes. Any weight eviction invalidates baked device pointers,
    /// so capture failures and eviction fall back to per-op decode for good.
    fn stepDecodeGraph(self: *CudaLM, id: u32) !void {
        const be = self.be;
        std.debug.assert(self.remaining() >= 1);
        try be.setDecodeState(id, @intCast(self.len));
        const p: u32 = @intCast(self.pos_next);
        try be.tensorUpload(self.pos3_d, std.mem.sliceAsBytes(&[3]u32{ p, p, p }));
        if (self.graph_exec == null) {
            self.captureDecodeGraph() catch |err| {
                std.debug.print("[decode graph capture failed ({t}); falling back to per-op launches]\n", .{err});
                self.graph_ok = false;
                return self.stepOne(id, true);
            };
        }
        if (be.evictions != 0) {
            // Capture itself ran the cache over budget: the fresh graph may
            // already hold evicted-weight pointers. Decode per-op instead.
            self.graph_ok = false;
            return self.stepOne(id, true);
        }
        try be.graphLaunch(self.graph_exec);
        self.len += 1;
        self.pos_next += 1;
    }

    fn captureDecodeGraph(self: *CudaLM) !void {
        const be = self.be;
        // The embed table is only touched device-side by the graph (warm
        // steps embed on host), so gather once outside capture first: its
        // initial cachedWeight cuMemAlloc + upload are illegal on a
        // capturing stream.
        try self.embedGather();
        try be.graphCaptureBegin();
        errdefer if (be.graphCaptureEnd()) |exec| be.graphDestroy(exec) else |_| {};
        try self.recordDecodeOps();
        self.graph_exec = try be.graphCaptureEnd();
    }

    /// Device-side embedding gather of g_state[0]'s row into bufs.x.
    fn embedGather(self: *CudaLM) !void {
        const cfg = self.cfg;
        const x = offsetBufSized(self.bufs.x, 0, cfg.hidden * 4);
        if (self.lm.embed.dtype == .bf16) {
            try self.be.opEmbedGatherS(x, self.lm.embed.bytes, cfg.hidden);
        } else {
            try self.be.opEmbedGatherQuant(self.lm.embed.dtype, x, self.lm.embed.bytes, cfg.hidden);
        }
    }

    /// The recorded decode step: device-side embedding gather (token id from
    /// g_state[0]) followed by the shared decode body in graph mode. No
    /// uploads, downloads, or batch bookkeeping — those live outside the
    /// replay.
    fn recordDecodeOps(self: *CudaLM) !void {
        try self.embedGather();
        try self.decodeBody(true, true);
    }

    /// Quantize a decode activation to the backend's shared q8 scratch for
    /// the dp4a GEMVs that follow; every gemv() reading `x` must be preceded
    /// by a quantizeX(x) with no other quantizeX in between (asserted).
    fn quantizeX(self: *CudaLM, x: Buf, cols: usize) !void {
        try self.be.opGemvQuantizeX(x, cols);
        self.q8_for = x;
        self.q8_cols = cols;
    }

    /// Fused GEMV in the weight's storage dtype (all qwen35 GGUF linear
    /// weights are block-quantized). q5_k/q6_k take the dp4a path against
    /// the activation staged by quantizeX; other dtypes read x directly.
    fn gemv(self: *CudaLM, y: Buf, x: Buf, w: ops.matmul.Weight) !void {
        const be = self.be;
        if (w.dtype == .q5_k or w.dtype == .q6_k) {
            std.debug.assert(self.q8_for.buf == x.buf and self.q8_cols == w.cols);
            try be.opGemvQuantQ8(w.dtype, y, w.bytes, w.scale, w.rows, w.cols);
        } else if (w.dtype.isBlockQuant()) {
            try be.opGemvQuant(w.dtype, y, x, w.bytes, w.scale, w.rows, w.cols);
        } else if (w.dtype == .bf16) {
            try be.opGemvBf16(y, x, w.bytes, w.scale, w.rows, w.cols);
        } else {
            return error.UnsupportedDType;
        }
    }
};

/// Zero a device buffer via chunked uploads of a host zero slab.
fn zeroBuffer(be: *Backend, gpa: std.mem.Allocator, buf: Buf, bytes: usize) !void {
    const chunk = @min(bytes, 4 << 20);
    const zeros = try gpa.alloc(u8, chunk);
    defer gpa.free(zeros);
    @memset(zeros, 0);
    var off: usize = 0;
    while (off < bytes) : (off += chunk) {
        const n = @min(chunk, bytes - off);
        try be.tensorUpload(offsetBufSized(buf, off, n), zeros[0..n]);
    }
}

const Bufs = struct {
    x: Buf,
    normed: Buf,
    qg: Buf,
    q: Buf,
    gate: Buf,
    k: Buf,
    v: Buf,
    attn: Buf,
    attn_scratch: Buf,
    t: Buf,
    lin_qkv: Buf,
    lin_conv: Buf,
    lin_z: Buf,
    lin_o: Buf,
    ab: Buf,
    gates: Buf,
    mlp_gate: Buf,
    mlp_up: Buf,
    logits: Buf,

    fn init(be: *Backend, cfg: qwen35.Config) !Bufs {
        var s: Bufs = undefined;
        // Activation buffers sized for the GEMM's 128-row-padded output
        // (opMatmulQuant always writes align(m, 128) rows — a chunk-sized
        // buffer would let the pad rows overflow into the next buffer).
        const pc = 128;
        comptime std.debug.assert(prefill_chunk <= pc);
        const sizes = [_]usize{
            pc * cfg.hidden, // x
            pc * cfg.hidden, // normed
            pc * cfg.qDim() * 2, // qg
            pc * cfg.qDim(), // q
            pc * cfg.qDim(), // gate
            pc * cfg.kvDim(), // k
            pc * cfg.kvDim(), // v
            pc * cfg.qDim(), // attn
            @max(cfg.n_heads * nsplit, pc * cfg.n_heads * nsplit_prefill) * (cfg.head_dim + 4), // attn_scratch
            pc * cfg.hidden, // t
            pc * cfg.convChannels(), // lin_qkv
            cfg.convChannels(), // lin_conv
            pc * cfg.linVDim(), // lin_z
            pc * cfg.linVDim(), // lin_o
            2 * cfg.lin_v_heads, // ab
            2 * cfg.lin_v_heads, // gates
            pc * cfg.intermediate, // mlp_gate
            pc * cfg.intermediate, // mlp_up
            cfg.vocab, // logits
        };
        var done: usize = 0;
        errdefer {
            inline for (@typeInfo(Bufs).@"struct".fields, 0..) |f, i| {
                if (i < done) be.tensorDestroy(&@field(s, f.name));
            }
        }
        inline for (@typeInfo(Bufs).@"struct".fields, 0..) |f, i| {
            @field(s, f.name) = try be.tensorCreate(sizes[i] * 4);
            done = i + 1;
        }
        return s;
    }

    fn deinit(s: *Bufs, be: *Backend) void {
        inline for (@typeInfo(Bufs).@"struct".fields) |f| {
            be.tensorDestroy(&@field(s, f.name));
        }
        s.* = undefined;
    }
};
