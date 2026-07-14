//! Gemma 3 on the Vulkan backend (tp-llm --backend vulkan). Text-only, fixed
//! KV capacity, one token at a time — a port of gemma3_cuda.zig that reuses
//! the validated block-quant GEMV + RoPE + attention Context ops, mirroring
//! qwen35_gpu's structure.
//!
//! Gemma specifics vs qwen35_gpu: all layers are attention (no DeltaNet),
//! four "sandwich" RMSNorms per layer (the two post-norms apply to the
//! sublayer output before its residual add), embeddings scaled by
//! sqrt(hidden) on the host, GeGLU (gelu_mul), a tied head, and RoPE whose
//! base/scale alternate by layer: global layers (every 6th) use theta 1e6 +
//! linear scale 1/8 with full causal attention; local layers use theta 1e4
//! with a sliding-window (1024) causal mask. Two RoPE tables (global/local)
//! live on device; each layer picks one. Full rotate-half over head_dim 256
//! reuses opRopeQwen35 with half = head_dim/2.
//!
//! Each token's layer stack is recorded into one command buffer and
//! submitted once. Prefill runs a token at a time (batched prefill is a
//! follow-up, as in qwen35_gpu).

const std = @import("std");
const qwen3 = @import("qwen3.zig");
const gemma3 = @import("gemma3.zig");
const gpu = @import("../gpu/context.zig");
const ops = @import("../ops.zig");
const kvmod = @import("../llm/kv_cache.zig");

const Buf = gpu.DeviceBuffer;
const Weight = ops.matmul.Weight;

fn nbuf(ctx: *gpu.Context, w: []const f32) !Buf {
    return .{ .buf = try ctx.smallBuffer(std.mem.sliceAsBytes(w)), .mem = .null_handle, .size = 0 };
}

pub const VulkanLM = struct {
    ctx: *gpu.Context,
    lm: *const gemma3.Model,
    cfg: gemma3.Config,
    gpa: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    capacity: usize,
    len: usize,

    x: Buf,
    normed: Buf,
    q: Buf,
    k: Buf,
    v: Buf,
    attn: Buf,
    gate: Buf,
    up: Buf,
    t: Buf,
    logits: Buf,
    partials: Buf,

    k_cache: []Buf,
    v_cache: []Buf,
    /// Global (theta 1e6, scale 1/8) and local (theta 1e4) RoPE tables, each
    /// [2 * capacity * half] (cos then sin); sin_off = capacity * half.
    freqs_global: Buf,
    freqs_local: Buf,
    sin_off: usize,

    const gemv_nchunk = 16;

    pub fn init(gpa: std.mem.Allocator, ctx: *gpu.Context, lm: *const gemma3.Model, cap: kvmod.Capacity) !VulkanLM {
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
        self.sin_off = cap.max * (cfg.head_dim / 2);

        self.freqs_global = try uploadFreqs(ctx, gpa, cap.max, cfg.head_dim, cfg.rope_theta, cfg.rope_freq_scale);
        self.freqs_local = try uploadFreqs(ctx, gpa, cap.max, cfg.head_dim, cfg.rope_theta_local, 1.0);

        self.x = try ctx.tensorCreate(cfg.hidden * 4);
        self.normed = try ctx.tensorCreate(cfg.hidden * 4);
        self.q = try ctx.tensorCreate(cfg.qDim() * 4);
        self.k = try ctx.tensorCreate(cfg.kvDim() * 4);
        self.v = try ctx.tensorCreate(cfg.kvDim() * 4);
        self.attn = try ctx.tensorCreate(cfg.qDim() * 4);
        self.gate = try ctx.tensorCreate(cfg.intermediate * 4);
        self.up = try ctx.tensorCreate(cfg.intermediate * 4);
        self.t = try ctx.tensorCreate(cfg.hidden * 4);
        self.logits = try ctx.tensorCreate(cfg.vocab * 4);
        const max_rows = @max(cfg.vocab, @max(cfg.intermediate, cfg.qDim()));
        self.partials = try ctx.tensorCreate(max_rows * gemv_nchunk * 4);

        self.k_cache = try alloc.alloc(Buf, cfg.n_layers);
        self.v_cache = try alloc.alloc(Buf, cfg.n_layers);
        for (self.k_cache, self.v_cache) |*kb, *vb| {
            kb.* = try ctx.tensorCreate(cap.max * cfg.kvDim() * 4);
            vb.* = try ctx.tensorCreate(cap.max * cfg.kvDim() * 4);
        }

        self.arena = arena;
        return self;
    }

    fn uploadFreqs(ctx: *gpu.Context, gpa: std.mem.Allocator, rows: usize, head_dim: usize, theta: f64, freq_scale: f64) !Buf {
        const half = head_dim / 2;
        var freqs = try ops.rope.rotateHalfFreqsScaled(gpa, rows, head_dim, theta, freq_scale);
        defer freqs.deinit(gpa);
        const fp = try gpa.alloc(f32, 2 * rows * half);
        defer gpa.free(fp);
        @memcpy(fp[0 .. rows * half], freqs.cos);
        @memcpy(fp[rows * half ..], freqs.sin);
        const buf = try ctx.tensorCreate(fp.len * 4);
        try ctx.tensorUpload(buf, std.mem.sliceAsBytes(fp));
        return buf;
    }

    pub fn deinit(self: *VulkanLM) void {
        const ctx = self.ctx;
        inline for (.{ "x", "normed", "q", "k", "v", "attn", "gate", "up", "t", "logits", "partials", "freqs_global", "freqs_local" }) |f| {
            ctx.tensorDestroy(&@field(self, f));
        }
        for (self.k_cache) |*b| ctx.tensorDestroy(b);
        for (self.v_cache) |*b| ctx.tensorDestroy(b);
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

    pub fn step(self: *VulkanLM, io: std.Io, ids_new: []const u32, logits: []f32) !void {
        _ = io;
        const cfg = self.cfg;
        const xh = try self.gpa.alloc(f32, cfg.hidden);
        defer self.gpa.free(xh);
        const scale = cfg.embedScale();
        for (ids_new, 0..) |id, i| {
            try qwen3.embedTokens(self.lm.embed, &.{id}, xh);
            for (xh) |*e| e.* *= scale; // Gemma scales token embeddings by sqrt(hidden)
            try self.ctx.tensorUpload(self.x, std.mem.sliceAsBytes(xh));
            try self.ctx.beginBatch();
            errdefer if (self.ctx.batching) self.ctx.abortBatch();
            try self.decodeBody(i + 1 == ids_new.len);
            try self.ctx.endBatch();
            self.len += 1;
        }
        try self.ctx.tensorDownload(self.logits, std.mem.sliceAsBytes(logits));
    }

    /// Prefill text tokens (no logits) — for interleaving with prefillImage.
    pub fn prefill(self: *VulkanLM, ids: []const u32) !void {
        const cfg = self.cfg;
        const xh = try self.gpa.alloc(f32, cfg.hidden);
        defer self.gpa.free(xh);
        const scale = cfg.embedScale();
        for (ids) |id| {
            try qwen3.embedTokens(self.lm.embed, &.{id}, xh);
            for (xh) |*e| e.* *= scale;
            try self.ctx.tensorUpload(self.x, std.mem.sliceAsBytes(xh));
            try self.ctx.beginBatch();
            errdefer if (self.ctx.batching) self.ctx.abortBatch();
            try self.decodeBody(false);
            try self.ctx.endBatch();
            self.len += 1;
        }
    }

    /// Prefill one image's projected embeddings ([grid_w*grid_h][hidden],
    /// injected UNSCALED) at the next sequential positions.
    pub fn prefillImage(self: *VulkanLM, embeds: []const f32, grid_w: usize, grid_h: usize) !void {
        _ = grid_w;
        _ = grid_h;
        const cfg = self.cfg;
        const count = embeds.len / cfg.hidden;
        for (0..count) |i| {
            try self.ctx.tensorUpload(self.x, std.mem.sliceAsBytes(embeds[i * cfg.hidden ..][0..cfg.hidden]));
            try self.ctx.beginBatch();
            errdefer if (self.ctx.batching) self.ctx.abortBatch();
            try self.decodeBody(false);
            try self.ctx.endBatch();
            self.len += 1;
        }
    }

    fn decodeBody(self: *VulkanLM, want_logits: bool) !void {
        const ctx = self.ctx;
        const cfg = self.cfg;
        const hd = cfg.head_dim;
        const half = hd / 2;
        const pos = self.len;
        const kvdim = cfg.kvDim();
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));

        for (self.lm.layers, 0..) |*layer, l| {
            const global = cfg.isGlobal(l);
            const freqs = if (global) self.freqs_global else self.freqs_local;
            const window: usize = if (global) 0 else cfg.sliding_window;

            // --- Attention ---
            try self.rms(self.x, self.normed, layer.input_norm, 1, cfg.hidden);
            try self.gemvW(self.q, 0, self.normed, layer.q);
            try self.gemvW(self.k, 0, self.normed, layer.k);
            try self.gemvW(self.v, 0, self.normed, layer.v);
            try self.rms(self.q, self.q, layer.q_norm, cfg.n_heads, hd);
            try self.rms(self.k, self.k, layer.k_norm, cfg.n_kv_heads, hd);
            try ctx.opRopeQwen35(self.q, freqs, cfg.n_heads, half, self.sin_off, hd, pos);
            try ctx.opRopeQwen35(self.k, freqs, cfg.n_kv_heads, half, self.sin_off, hd, pos);
            // KV append (in-batch device copy: dst[u2+i] = src[i]).
            try ctx.opElt(.copy, self.k, self.k_cache[l], null, null, .{ .u0 = @intCast(kvdim), .u2 = @intCast(pos * kvdim) }, kvdim, 1, 1);
            try ctx.opElt(.copy, self.v, self.v_cache[l], null, null, .{ .u0 = @intCast(kvdim), .u2 = @intCast(pos * kvdim) }, kvdim, 1, 1);
            try ctx.opAttnDecodeQ35(self.q, self.k_cache[l], self.v_cache[l], self.attn, cfg.n_heads, cfg.n_kv_heads, hd, pos + 1, scale, window);
            // o_proj, then post-attention norm on the attn output BEFORE the residual.
            try self.gemvW(self.t, 0, self.attn, layer.o);
            try self.rms(self.t, self.t, layer.post_attn_norm, 1, cfg.hidden);
            try self.add(self.x, self.t, cfg.hidden);

            // --- MLP (GeGLU) ---
            try self.rms(self.x, self.normed, layer.pre_ffn_norm, 1, cfg.hidden);
            try self.gemvW(self.gate, 0, self.normed, layer.gate);
            try self.gemvW(self.up, 0, self.normed, layer.up);
            try ctx.opElt(.gelu_mul, self.gate, self.up, null, null, .{ .u0 = @intCast(cfg.intermediate) }, cfg.intermediate, 1, 1);
            try self.gemvW(self.t, 0, self.gate, layer.down);
            try self.rms(self.t, self.t, layer.post_ffn_norm, 1, cfg.hidden);
            try self.add(self.x, self.t, cfg.hidden);
        }

        if (want_logits) {
            try self.rms(self.x, self.t, self.lm.final_norm, 1, cfg.hidden);
            try self.gemvW(self.logits, 0, self.t, self.lm.head);
        }
    }
};
