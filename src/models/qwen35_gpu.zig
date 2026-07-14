//! qwen35 hybrid (gated DeltaNet) on the Vulkan backend. Text-only, fixed KV
//! capacity, one token at a time (no batched prefill, CUDA-graph capture, or
//! CPU split) — a port of qwen35_cuda.zig that reuses the validated
//! block-quant GEMV + GDN + RoPE + attention Context ops.
//!
//! Each token's whole layer stack is recorded into one command buffer
//! (beginBatch/endBatch) and submitted once, so the GPU stays saturated
//! instead of paying a submit+wait per op. The KV append is an in-batch
//! device copy (not tensorCopy, which would flush the recording every layer).
//! Prefill still runs a token at a time; batched prefill is a follow-up.

const std = @import("std");
const qwen3 = @import("qwen3.zig");
const qwen35 = @import("qwen35.zig");
const gpu = @import("../gpu/context.zig");
const ops = @import("../ops.zig");
const kvmod = @import("../llm/kv_cache.zig");

const Buf = gpu.DeviceBuffer;
const Weight = ops.matmul.Weight;

/// Upload a small f32 vector (norm weight) to a pointer-cached device buffer.
fn nbuf(ctx: *gpu.Context, w: []const f32) !Buf {
    return .{ .buf = try ctx.smallBuffer(std.mem.sliceAsBytes(w)), .mem = .null_handle, .size = 0 };
}

pub const VulkanLM = struct {
    ctx: *gpu.Context,
    lm: *const qwen35.Model,
    cfg: qwen35.Config,
    gpa: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    capacity: usize,
    len: usize,

    // Per-token activation buffers.
    x: Buf,
    normed: Buf,
    qg: Buf,
    q: Buf,
    gate: Buf,
    k: Buf,
    v: Buf,
    attn: Buf,
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
    // k-split GEMV partials scratch [nchunk][max_rows]; reduced by gemv_combine.
    partials: Buf,

    // Per-attention-slot KV caches [capacity][kvDim]; recurrent conv/ssm states.
    k_cache: []Buf,
    v_cache: []Buf,
    // Per-linear-layer recurrent state (separate buffers: Vulkan can't offset a
    // buffer handle, so each layer gets its own read-from-0 region).
    conv_state: []Buf,
    ssm_state: []Buf,
    freqs_d: Buf,
    sin_off: usize,
    // Per-linear-layer [a | dt_bias] host constants for gdn_gates.
    a_dt: [][]f32,

    /// Superblock chunks per output row in the transposed k-split GEMV. One
    /// thread per (row, chunk); enough chunks to give the 3090 the warps it
    /// needs to hide memory latency (one-thread-per-row is ~1.4 warps/SM).
    const gemv_nchunk = 16;

    fn zeroBuf(ctx: *gpu.Context, gpa: std.mem.Allocator, buf: Buf, bytes: usize) !void {
        const zeros = try gpa.alloc(u8, bytes);
        defer gpa.free(zeros);
        @memset(zeros, 0);
        try ctx.tensorUpload(buf, zeros);
    }

    pub fn init(gpa: std.mem.Allocator, ctx: *gpu.Context, lm: *const qwen35.Model, cap: kvmod.Capacity) !VulkanLM {
        const cfg = lm.cfg;
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();

        var self: VulkanLM = undefined;
        self.ctx = ctx;
        self.lm = lm;
        self.cfg = cfg;
        self.gpa = gpa;
        self.capacity = cap.max;
        self.len = 0;

        // RoPE table for the rotated span, to max capacity up front.
        var freqs = try ops.rope.rotateHalfFreqs(gpa, cap.max, cfg.rope_dim, cfg.rope_theta);
        defer freqs.deinit(gpa);
        const half = cfg.rope_dim / 2;
        const fp = try gpa.alloc(f32, 2 * cap.max * half);
        defer gpa.free(fp);
        @memcpy(fp[0 .. cap.max * half], freqs.cos);
        @memcpy(fp[cap.max * half ..], freqs.sin);
        self.sin_off = cap.max * half;
        self.freqs_d = try ctx.tensorCreate(fp.len * 4);
        try ctx.tensorUpload(self.freqs_d, std.mem.sliceAsBytes(fp));

        // Activation buffers (one token).
        self.x = try ctx.tensorCreate(cfg.hidden * 4);
        self.normed = try ctx.tensorCreate(cfg.hidden * 4);
        self.qg = try ctx.tensorCreate(cfg.qDim() * 2 * 4);
        self.q = try ctx.tensorCreate(cfg.qDim() * 4);
        self.gate = try ctx.tensorCreate(cfg.qDim() * 4);
        self.k = try ctx.tensorCreate(cfg.kvDim() * 4);
        self.v = try ctx.tensorCreate(cfg.kvDim() * 4);
        self.attn = try ctx.tensorCreate(cfg.qDim() * 4);
        self.t = try ctx.tensorCreate(cfg.hidden * 4);
        self.lin_qkv = try ctx.tensorCreate(cfg.convChannels() * 4);
        self.lin_conv = try ctx.tensorCreate(cfg.convChannels() * 4);
        self.lin_z = try ctx.tensorCreate(cfg.linVDim() * 4);
        self.lin_o = try ctx.tensorCreate(cfg.linVDim() * 4);
        self.ab = try ctx.tensorCreate(2 * cfg.lin_v_heads * 4);
        self.gates = try ctx.tensorCreate(2 * cfg.lin_v_heads * 4);
        self.mlp_gate = try ctx.tensorCreate(cfg.intermediate * 4);
        self.mlp_up = try ctx.tensorCreate(cfg.intermediate * 4);
        self.logits = try ctx.tensorCreate(cfg.vocab * 4);
        // Split-GEMV partials, sized for the largest GEMV output (LM head,
        // rows = vocab) times nchunk.
        const max_rows = @max(cfg.vocab, @max(cfg.convChannels(), @max(cfg.intermediate, cfg.qDim() * 2)));
        self.partials = try ctx.tensorCreate(max_rows * gemv_nchunk * 4);

        const n_attn = cfg.nAttnLayers();
        self.k_cache = try alloc.alloc(Buf, n_attn);
        self.v_cache = try alloc.alloc(Buf, n_attn);
        for (self.k_cache, self.v_cache) |*kb, *vb| {
            kb.* = try ctx.tensorCreate(cap.max * cfg.kvDim() * 4);
            vb.* = try ctx.tensorCreate(cap.max * cfg.kvDim() * 4);
        }

        const n_lin = cfg.n_layers - n_attn;
        const conv_bytes = cfg.convChannels() * (cfg.conv_kernel - 1) * 4;
        const ssm_bytes = cfg.lin_v_heads * cfg.lin_head_dim * cfg.lin_head_dim * 4;
        self.conv_state = try alloc.alloc(Buf, n_lin);
        self.ssm_state = try alloc.alloc(Buf, n_lin);
        for (self.conv_state, self.ssm_state) |*cs, *ss| {
            cs.* = try ctx.tensorCreate(conv_bytes);
            ss.* = try ctx.tensorCreate(ssm_bytes);
            try zeroBuf(ctx, gpa, cs.*, conv_bytes);
            try zeroBuf(ctx, gpa, ss.*, ssm_bytes);
        }

        self.a_dt = try alloc.alloc([]f32, n_lin);
        var lin_idx: usize = 0;
        for (lm.layers) |*layer| switch (layer.*) {
            .linear => |*ll| {
                const buf = try alloc.alloc(f32, 2 * cfg.lin_v_heads);
                @memcpy(buf[0..cfg.lin_v_heads], ll.a);
                @memcpy(buf[cfg.lin_v_heads..], ll.dt_bias);
                self.a_dt[lin_idx] = buf;
                lin_idx += 1;
            },
            .attn => {},
        };

        self.arena = arena;
        return self;
    }

    pub fn deinit(self: *VulkanLM) void {
        const ctx = self.ctx;
        inline for (.{ "x", "normed", "qg", "q", "gate", "k", "v", "attn", "t", "lin_qkv", "lin_conv", "lin_z", "lin_o", "ab", "gates", "mlp_gate", "mlp_up", "logits", "partials", "freqs_d" }) |f| {
            ctx.tensorDestroy(&@field(self, f));
        }
        for (self.k_cache) |*b| ctx.tensorDestroy(b);
        for (self.v_cache) |*b| ctx.tensorDestroy(b);
        for (self.conv_state) |*b| ctx.tensorDestroy(b);
        for (self.ssm_state) |*b| ctx.tensorDestroy(b);
        self.arena.deinit();
    }

    pub fn cached(self: *const VulkanLM) usize {
        return self.len;
    }
    pub fn remaining(self: *const VulkanLM) usize {
        return self.capacity - self.len;
    }
    pub fn vocab(self: *const VulkanLM) usize {
        return self.cfg.vocab;
    }
    pub fn ensureCapacity(self: *VulkanLM, min_rows: usize) !void {
        if (min_rows > self.capacity) return error.ContextFull;
    }

    /// GEMV of a block-quant weight against the (f32) activation `x` into
    /// `y[y_off..]`; dequant-on-the-fly. Block-quant formats read the
    /// 32-row-group transposed layout with a k-split reduction (coalesced warp
    /// loads + enough warps to hide latency); anything else falls back to the
    /// raw row-major kernel.
    fn gemvW(self: *VulkanLM, y: Buf, y_off: usize, x: Buf, w: Weight) !void {
        switch (w.dtype) {
            .q8_0, .q4_k, .q5_k, .q6_k => try self.ctx.opGemvQuantT(w.dtype, y, y_off, x, w.bytes, w.scale, w.rows, w.cols, gemv_nchunk, self.partials),
            else => try self.ctx.opGemvQuant(w.dtype, y, y_off, x, w.bytes, w.scale, w.rows, w.cols),
        }
    }

    fn rms(self: *VulkanLM, in: Buf, out: Buf, weight: []const f32, rows: usize, dim: usize) !void {
        try self.ctx.opElt(.rmsnorm, in, out, try nbuf(self.ctx, weight), null, .{
            .u0 = @intCast(rows),
            .u1 = @intCast(dim),
            .f0 = self.cfg.rms_eps,
        }, rows, 1, 1);
    }
    fn add(self: *VulkanLM, x: Buf, t: Buf, total: usize) !void {
        try self.ctx.opElt(.add, x, t, null, null, .{ .u0 = @intCast(total) }, total, 1, 1);
    }
    fn siluMul(self: *VulkanLM, a: Buf, b: Buf, total: usize) !void {
        try self.ctx.opElt(.silu_mul, a, b, null, null, .{ .u0 = @intCast(total) }, total, 1, 1);
    }

    pub fn step(self: *VulkanLM, io: std.Io, ids_new: []const u32, logits: []f32) !void {
        _ = io;
        const cfg = self.cfg;
        const xh = try self.gpa.alloc(f32, cfg.hidden);
        defer self.gpa.free(xh);
        for (ids_new, 0..) |id, i| {
            try qwen3.embedTokens(self.lm.embed, &.{id}, xh);
            // Upload the embedded token before opening the batch (uploads of
            // batch-visible buffers would otherwise flush the recording).
            try self.ctx.tensorUpload(self.x, std.mem.sliceAsBytes(xh));
            try self.ctx.beginBatch();
            errdefer if (self.ctx.batching) self.ctx.abortBatch();
            try self.decodeBody(i + 1 == ids_new.len);
            try self.ctx.endBatch();
            self.len += 1;
        }
        try self.ctx.tensorDownload(self.logits, std.mem.sliceAsBytes(logits));
    }

    fn decodeBody(self: *VulkanLM, want_logits: bool) !void {
        const ctx = self.ctx;
        const cfg = self.cfg;
        const hd = cfg.head_dim;
        const half = cfg.rope_dim / 2;
        const pos = self.len;
        const kvdim = cfg.kvDim();

        for (self.lm.layers, 0..) |*layer, l| {
            switch (layer.*) {
                .attn => |*al| {
                    const slot = l / cfg.full_attn_interval;
                    try self.rms(self.x, self.normed, al.input_norm, 1, cfg.hidden);
                    try self.gemvW(self.qg, 0, self.normed, al.qg);
                    try self.gemvW(self.k, 0, self.normed, al.k);
                    try self.gemvW(self.v, 0, self.normed, al.v);
                    try ctx.opDeinterleave2(self.qg, self.q, self.gate, cfg.qDim(), hd);
                    try self.rms(self.q, self.q, al.q_norm, cfg.n_heads, hd);
                    try self.rms(self.k, self.k, al.k_norm, cfg.n_kv_heads, hd);
                    try ctx.opRopeQwen35(self.q, self.freqs_d, cfg.n_heads, half, self.sin_off, hd, pos);
                    try ctx.opRopeQwen35(self.k, self.freqs_d, cfg.n_kv_heads, half, self.sin_off, hd, pos);
                    // Append K/V to the cache with in-batch device copies
                    // (copy kernel: dst[u2+i] = src[u3+i]) — tensorCopy would
                    // flush the recording and drain the GPU every layer.
                    try ctx.opElt(.copy, self.k, self.k_cache[slot], null, null, .{ .u0 = @intCast(kvdim), .u2 = @intCast(pos * kvdim) }, kvdim, 1, 1);
                    try ctx.opElt(.copy, self.v, self.v_cache[slot], null, null, .{ .u0 = @intCast(kvdim), .u2 = @intCast(pos * kvdim) }, kvdim, 1, 1);
                    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
                    try ctx.opAttnDecodeQ35(self.q, self.k_cache[slot], self.v_cache[slot], self.attn, cfg.n_heads, cfg.n_kv_heads, hd, pos + 1, scale, 0);
                    try ctx.opElt(.sigmoid_mul, self.attn, self.gate, null, null, .{ .u0 = @intCast(cfg.qDim()) }, cfg.qDim(), 1, 1);
                    try self.gemvW(self.t, 0, self.attn, al.o);
                    try self.add(self.x, self.t, cfg.hidden);
                },
                .linear => |*ll| {
                    const lin_idx = l - l / cfg.full_attn_interval;
                    const channels = cfg.convChannels();
                    const d = cfg.lin_head_dim;
                    const heads = cfg.lin_v_heads;
                    try self.rms(self.x, self.normed, ll.input_norm, 1, cfg.hidden);
                    try self.gemvW(self.lin_qkv, 0, self.normed, ll.qkv);
                    try self.gemvW(self.lin_z, 0, self.normed, ll.z);
                    try self.gemvW(self.ab, 0, self.normed, ll.alpha);
                    try self.gemvW(self.ab, heads, self.normed, ll.beta);
                    try ctx.opGdnGates(self.ab, try nbuf(ctx, self.a_dt[lin_idx]), self.gates, heads);
                    try ctx.opGdnConvStep(self.conv_state[lin_idx], self.lin_qkv, try nbuf(ctx, ll.conv_w), self.lin_conv, channels, cfg.conv_kernel);
                    try ctx.opL2NormRows(self.lin_conv, 2 * cfg.lin_k_heads, d, cfg.rms_eps);
                    try ctx.opGdnDeltaStep(self.ssm_state[lin_idx], self.lin_conv, self.gates, self.lin_o, heads, d, cfg.lin_k_heads, 1.0 / @sqrt(@as(f32, @floatFromInt(d))));
                    try self.rms(self.lin_o, self.lin_o, ll.ssm_norm, heads, d);
                    try self.siluMul(self.lin_z, self.lin_o, cfg.linVDim());
                    try self.gemvW(self.t, 0, self.lin_z, ll.out);
                    try self.add(self.x, self.t, cfg.hidden);
                },
            }
            const mlp = switch (layer.*) {
                .attn => |*al| &al.mlp,
                .linear => |*ll| &ll.mlp,
            };
            try self.rms(self.x, self.normed, mlp.post_norm, 1, cfg.hidden);
            try self.gemvW(self.mlp_gate, 0, self.normed, mlp.gate);
            try self.gemvW(self.mlp_up, 0, self.normed, mlp.up);
            try self.siluMul(self.mlp_gate, self.mlp_up, cfg.intermediate);
            try self.gemvW(self.t, 0, self.mlp_gate, mlp.down);
            try self.add(self.x, self.t, cfg.hidden);
        }

        if (want_logits) {
            try self.rms(self.x, self.t, self.lm.final_norm, 1, cfg.hidden);
            try self.gemvW(self.logits, 0, self.t, self.lm.head);
        }
    }
};
