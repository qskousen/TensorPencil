//! Gemma 3 on the Vulkan backend (tp-llm --backend vulkan), the `gemma3_cuda` of this
//! backend: `transformer_gpu.decoderLayer` over this stepper's ops, batched at any
//! position, every linear through `lin_gpu`, attention the dual kernel against
//! `kv_gpu.Kv`. Text, and image blocks prefilled bidirectionally in one batch.
//!
//! Gemma specifics: four "sandwich" RMSNorms per layer (the two post-norms apply to the
//! sublayer output before its residual add), token embeddings scaled by sqrt(hidden) on
//! the host (an image's are not), GeGLU, a tied head, and RoPE whose base and scale
//! alternate by layer: global layers theta 1e6 with linear scale 1/8 and full causal
//! attention, local layers theta 1e4 with a sliding window, their KV a ring.

const std = @import("std");
const init_defaults = @import("tp_core").init_defaults;
const qwen3 = @import("qwen3.zig");
const gemma3 = @import("gemma3.zig");
const lin_gpu = @import("lin_gpu.zig");
const kv_gpu = @import("kv_gpu.zig");
const gpu = @import("tp_gpu").context;
const ops = @import("tp_ops");
const kvmod = @import("tp_core").kv_cache;
const transformer = @import("transformer.zig");
const transformer_gpu = @import("transformer_gpu.zig");

const Buf = gpu.DeviceBuffer;

fn nbuf(ctx: *gpu.Context, w: []const f32) !Buf {
    return .{ .buf = try ctx.smallBuffer(std.mem.sliceAsBytes(w)), .mem = .null_handle, .size = 0 };
}

/// Rows per batched forward, and the largest image block (Gemma 3 vision is always a
/// 16x16 = 256-token block, prefilled in one bidirectional batch).
const prefill_chunk = 256;
/// KV chunks per head in the attention split pass: a decode step, then a batch, whose
/// query rows already fill the device.
const nsplit = 32;
const nsplit_prefill = 8;
const gemv_nchunk = 16;

/// Rows a local layer's ring holds: its window plus a whole batch, since a batch stores
/// every row before any of its queries reads, and the first query still needs the
/// window behind it.
fn ringRows(cfg: gemma3.Config) usize {
    return cfg.sliding_window + prefill_chunk;
}

pub const VulkanLM = struct {
    /// The prefill batch this stepper wants the engine to hand it.
    pub const prefill_batch = prefill_chunk;
    /// `tp-llm --bench` alternates the grouped and per-head attention kernels, which
    /// must give the same logits hash.
    pub const bench_toggle = &gpu.Context.attn_rows_grouped;

    ctx: *gpu.Context,
    lm: *const gemma3.Model,
    cfg: gemma3.Config,
    gpa: std.mem.Allocator,
    capacity: usize,
    len: usize,
    kv: kv_gpu.Kv,
    lin: lin_gpu.Lin,
    lin_plan: lin_gpu.Plan,
    /// Global (theta 1e6, scale 1/8) and local (theta 1e4) RoPE tables, each
    /// [2 * capacity * half] (cos then sin); sin_off = capacity * half.
    freqs_global: Buf,
    freqs_local: Buf,
    sin_off: usize,
    /// While an image block prefills, its end position: every query in it attends the
    /// whole block. 0 otherwise.
    bidir_end: usize = 0,
    bufs: Bufs,
    /// `bufs.t` holds a residual not yet added to `bufs.x`: the next hidden norm adds it
    /// in the same dispatch.
    residual_pending: bool = false,

    pub fn init(gpa: std.mem.Allocator, ctx: *gpu.Context, lm: *const gemma3.Model, cap: kvmod.Capacity) !VulkanLM {
        const cfg = lm.cfg;
        var self: VulkanLM = init_defaults.of(VulkanLM);
        self.ctx = ctx;
        self.lm = lm;
        self.cfg = cfg;
        self.gpa = gpa;
        self.capacity = cap.max;
        self.len = 0;
        self.bidir_end = 0;
        self.residual_pending = false;
        self.sin_off = cap.max * (cfg.head_dim / 2);

        const max_out = @max(cfg.vocab, @max(cfg.intermediate, cfg.qDim()));
        self.lin = try lin_gpu.Lin.init(gpa, ctx, max_out, gemv_nchunk, 3);
        errdefer self.lin.deinit();
        {
            var arena = std.heap.ArenaAllocator.init(gpa);
            defer arena.deinit();
            const lins = try lm.deviceLins(arena.allocator());
            self.lin_plan = try self.lin.plan(lins, lin_gpu.Use.llm(prefill_chunk), "gemma3 vulkan");
            // Not for the head: a prefill reads no logits and a step runs it as a GEMV,
            // while its expansion would be vocab x hidden of scratch.
            var batched: std.ArrayList(lin_gpu.Weight) = .empty;
            for (lins) |w| if (w.bytes.ptr != lm.head.bytes.ptr) try batched.append(arena.allocator(), w);
            try lin_gpu.presize(ctx, self.lin_plan, batched.items);
        }

        self.freqs_global = try uploadFreqs(ctx, gpa, cap.max, cfg.head_dim, cfg.rope_theta, cfg.rope_freq_scale);
        errdefer ctx.tensorDestroy(&self.freqs_global);
        self.freqs_local = try uploadFreqs(ctx, gpa, cap.max, cfg.head_dim, cfg.rope_theta_local, 1.0);
        errdefer ctx.tensorDestroy(&self.freqs_local);

        const rings = try gpa.alloc(usize, cfg.n_layers);
        defer gpa.free(rings);
        for (rings, 0..) |*r, l| r.* = if (cfg.isGlobal(l)) 0 else ringRows(cfg);
        self.kv = try kv_gpu.Kv.init(gpa, ctx, cfg.n_layers, cap.max, cfg.kvDim(), cap.kv_dtype, rings);
        errdefer self.kv.deinit(gpa, ctx);
        self.bufs = try Bufs.init(ctx, cfg);
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
        var buf = try ctx.tensorCreate(fp.len * 4);
        errdefer ctx.tensorDestroy(&buf);
        try ctx.tensorUpload(buf, std.mem.sliceAsBytes(fp));
        return buf;
    }

    pub fn deinit(self: *VulkanLM) void {
        const ctx = self.ctx;
        self.kv.deinit(self.gpa, ctx);
        ctx.tensorDestroy(&self.freqs_global);
        ctx.tensorDestroy(&self.freqs_local);
        self.bufs.deinit(ctx);
        self.lin.deinit();
        self.* = undefined;
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
    /// Device VRAM (bytes) this Vulkan context has allocated, for the end-of-response
    /// telemetry.
    pub fn vramUsed(self: *const VulkanLM) u64 {
        return self.ctx.device_used;
    }
    pub fn ensureCapacity(self: *VulkanLM, min_rows: usize) !void {
        if (min_rows > self.capacity) return error.ContextFull;
    }
    /// An empty context: the KV rows are left to be overwritten.
    pub fn resetCache(self: *VulkanLM) !void {
        self.len = 0;
    }

    /// Forward `ids` in chunks, then write the last position's logits.
    pub fn step(self: *VulkanLM, io: std.Io, ids: []const u32, logits: []f32) !void {
        _ = io;
        std.debug.assert(ids.len >= 1);
        try self.prefillIds(ids, true);
        try self.ctx.tensorDownload(self.bufs.logits, std.mem.sliceAsBytes(logits[0..self.cfg.vocab]));
    }

    /// Prefill text tokens (no logits), for interleaving with prefillImage.
    pub fn prefill(self: *VulkanLM, ids: []const u32) !void {
        try self.prefillIds(ids, false);
    }

    /// Prefill one image's projected embeddings ([grid_w*grid_h][hidden], unscaled) at
    /// the next positions, every query seeing the whole block (llama.cpp marks image
    /// spans non-causal): one batch, so the block's K/V is stored before any of it is
    /// read.
    pub fn prefillImage(self: *VulkanLM, embeds: []const f32, grid_w: usize, grid_h: usize) !void {
        _ = grid_w;
        _ = grid_h;
        const n = embeds.len / self.cfg.hidden;
        std.debug.assert(n >= 1 and n <= prefill_chunk and n <= self.remaining());
        self.bidir_end = self.len + n;
        defer self.bidir_end = 0;
        try self.forwardRows(embeds, false);
    }

    fn prefillIds(self: *VulkanLM, ids: []const u32, want_logits: bool) !void {
        const cfg = self.cfg;
        const x = try self.gpa.alloc(f32, @min(ids.len, prefill_chunk) * cfg.hidden);
        defer self.gpa.free(x);
        const scale = cfg.embedScale();
        var off: usize = 0;
        while (off < ids.len) {
            const n: usize = @min(prefill_chunk, ids.len - off);
            const xs = x[0 .. n * cfg.hidden];
            try qwen3.embedTokens(self.lm.embed, ids[off..][0..n], xs);
            for (xs) |*v| v.* *= scale;
            try self.forwardRows(xs, want_logits and off + n == ids.len);
            off += n;
        }
    }

    /// One batch over the embedded rows `x` at positions [len, len+n), the last row's
    /// logits into `bufs.logits` when asked. Polls `ops.cancel` between layers.
    fn forwardRows(self: *VulkanLM, x: []const f32, want_logits: bool) !void {
        const ctx = self.ctx;
        const cfg = self.cfg;
        const b = &self.bufs;
        const n = x.len / cfg.hidden;
        std.debug.assert(n >= 1 and n <= prefill_chunk and n <= self.remaining());
        const pos0 = self.len;
        // Before the batch opens: an upload inside it would flush the recording.
        try ctx.tensorUpload(try ctx.view(b.x, 0, x.len * 4), std.mem.sliceAsBytes(x));
        try ctx.beginBatch();
        errdefer if (ctx.batching) ctx.abortBatch();
        self.residual_pending = false;
        const ctok = ops.cancel.token;
        for (self.lm.layers, 0..) |*layer, l| {
            if (ops.cancel.canceled(ctok)) return error.Canceled;
            try transformer_gpu.decoderLayer(transformer.gemma3_spec, self, layer, l, n, pos0);
        }
        if (want_logits) {
            // Every row is normed (the residual is pending on all of them); the head
            // reads the last.
            try self.hiddenNorm(self.lm.final_norm, n);
            var last = b.normed;
            if (n > 1) {
                try ctx.opElt(.copy, b.normed, b.t, null, null, .{
                    .u0 = @intCast(cfg.hidden),
                    .u2 = 0,
                    .u3 = @intCast((n - 1) * cfg.hidden),
                }, cfg.hidden, 1, 1);
                last = b.t;
            }
            try self.lin.linear(self.lin_plan, b.logits, 0, last, 1, self.lm.head);
        }
        try ctx.endBatch();
        self.len += n;
    }

    /// `normed = rmsnorm(x) * w` over `rows` rows, adding the pending residual into x
    /// first.
    fn hiddenNorm(self: *VulkanLM, w: []const f32, rows: usize) !void {
        const b = &self.bufs;
        try self.ctx.opAddRmsNorm(b.x, if (self.residual_pending) b.t else null, try nbuf(self.ctx, w), b.normed, rows, self.cfg.hidden, self.cfg.rms_eps);
        self.residual_pending = false;
    }

    // `transformer_gpu.decoderLayer` stepper methods.

    pub fn normInput(self: *VulkanLM, layer: anytype, seq: usize) !void {
        try self.hiddenNorm(layer.input_norm, seq);
    }
    pub fn projectQKV(self: *VulkanLM, l: usize, layer: anytype, seq: usize) !void {
        const b = &self.bufs;
        const kv = try self.kvOut(l, seq);
        if (seq == 1) return self.lin.linearGroup(self.lin_plan, &.{ b.q, kv[0], kv[1] }, b.normed, &.{ layer.q, layer.k, layer.v });
        const g = try self.lin.prep(self.lin_plan, b.normed, seq, self.cfg.hidden, &.{ layer.q, layer.k, layer.v });
        try self.lin.gemm(self.lin_plan, g, b.q, layer.q, .{});
        try self.lin.gemm(self.lin_plan, g, kv[0], layer.k, .{});
        try self.lin.gemm(self.lin_plan, g, kv[1], layer.v, .{});
    }

    /// Where layer `l` projects its k and v: straight into the cache rows when it can
    /// (`Kv.rowsAt`, a single row the GEMVs write exactly), else the scratch `appendKV`
    /// stores from.
    fn kvOut(self: *VulkanLM, l: usize, seq: usize) ![2]Buf {
        const layer = self.lm.layers[l];
        if (seq == 1 and self.lin.writesRows(self.lin_plan, layer.k, 1) and self.lin.writesRows(self.lin_plan, layer.v, 1))
            if (try self.kv.rowsAt(self.ctx, l, self.len, 1)) |rows| return rows;
        return .{ self.bufs.k, self.bufs.v };
    }
    /// The rope too, in the same pass: the position is the cache length, which
    /// `forwardRows` passes as pos0.
    pub fn normQK(self: *VulkanLM, l: usize, layer: anytype, seq: usize) !void {
        const cfg = self.cfg;
        const ctx = self.ctx;
        const freqs = if (cfg.isGlobal(l)) self.freqs_global else self.freqs_local;
        if (seq == 1) ctx.independent(2);
        try ctx.opQkNormRope(self.bufs.q, try nbuf(ctx, layer.q_norm), freqs, seq * cfg.n_heads, cfg.head_dim, cfg.n_heads, self.len, self.sin_off, cfg.rms_eps);
        try ctx.opQkNormRope((try self.kvOut(l, seq))[0], try nbuf(ctx, layer.k_norm), freqs, seq * cfg.n_kv_heads, cfg.head_dim, cfg.n_kv_heads, self.len, self.sin_off, cfg.rms_eps);
    }
    pub fn applyRope(self: *VulkanLM, l: usize, seq: usize, pos0: usize) !void {
        _ = l;
        _ = seq;
        std.debug.assert(pos0 == self.len);
    }
    pub fn appendKV(self: *VulkanLM, l: usize, seq: usize, pos0: usize) !void {
        // Projected straight into the cache.
        if ((try self.kvOut(l, seq))[0].buf != self.bufs.k.buf) return;
        try self.kv.store(self.ctx, l, self.bufs.k, self.bufs.v, pos0, seq);
    }
    pub fn attention(self: *VulkanLM, l: usize, seq: usize, pos0: usize) !void {
        const cfg = self.cfg;
        const b = &self.bufs;
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));
        const window: usize = if (cfg.isGlobal(l)) 0 else cfg.sliding_window;
        const ns: usize = if (seq == 1) nsplit else nsplit_prefill;
        try self.kv.attend(self.ctx, l, b.q, b.attn, b.attn_scratch, pos0 + 1, seq, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, ns, scale, window, self.bidir_end);
    }
    pub fn projectO(self: *VulkanLM, l: usize, layer: anytype, seq: usize) !void {
        _ = l;
        try self.lin.linear(self.lin_plan, self.bufs.t, 0, self.bufs.attn, seq, layer.o);
    }
    pub fn postAttnNorm(self: *VulkanLM, layer: anytype, seq: usize) !void {
        const b = &self.bufs;
        try self.ctx.opAddRmsNorm(b.t, null, try nbuf(self.ctx, layer.post_attn_norm), b.t, seq, self.cfg.hidden, self.cfg.rms_eps);
    }
    /// Deferred into the next `hiddenNorm`, which adds it in the same dispatch.
    pub fn addResidual(self: *VulkanLM, seq: usize) !void {
        _ = seq;
        std.debug.assert(!self.residual_pending);
        self.residual_pending = true;
    }
    pub fn normPreFfn(self: *VulkanLM, layer: anytype, seq: usize) !void {
        try self.hiddenNorm(layer.pre_ffn_norm, seq);
    }
    pub fn projectGateUp(self: *VulkanLM, layer: anytype, seq: usize) !void {
        const b = &self.bufs;
        if (seq == 1) return self.lin.linearGroup(self.lin_plan, &.{ b.gate, b.up }, b.normed, &.{ layer.gate, layer.up });
        const g = try self.lin.prep(self.lin_plan, b.normed, seq, self.cfg.hidden, &.{ layer.gate, layer.up });
        try self.lin.gemm(self.lin_plan, g, b.gate, layer.gate, .{});
        try self.lin.gemm(self.lin_plan, g, b.up, layer.up, .{});
    }
    pub fn activate(self: *VulkanLM, comptime act: transformer.Activation, seq: usize) !void {
        const n = seq * self.cfg.intermediate;
        const which: gpu.Elt = switch (act) {
            .silu_mul => .silu_mul,
            .gelu_tanh_mul => .gelu_mul,
        };
        try self.ctx.opElt(which, self.bufs.gate, self.bufs.up, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
    }
    pub fn projectDown(self: *VulkanLM, layer: anytype, seq: usize) !void {
        try self.lin.linear(self.lin_plan, self.bufs.t, 0, self.bufs.gate, seq, layer.down);
    }
    pub fn postFfnNorm(self: *VulkanLM, layer: anytype, seq: usize) !void {
        const b = &self.bufs;
        try self.ctx.opAddRmsNorm(b.t, null, try nbuf(self.ctx, layer.post_ffn_norm), b.t, seq, self.cfg.hidden, self.cfg.rms_eps);
    }
};

const Bufs = struct {
    x: Buf,
    normed: Buf,
    q: Buf,
    k: Buf,
    v: Buf,
    attn: Buf,
    attn_scratch: Buf,
    gate: Buf,
    up: Buf,
    t: Buf,
    logits: Buf,

    /// Every activation holds a whole chunk, a multiple of the coop GEMMs' 128-row tile.
    fn init(ctx: *gpu.Context, cfg: gemma3.Config) !Bufs {
        const r = prefill_chunk;
        const sizes = [_]usize{
            r * cfg.hidden,
            r * cfg.hidden,
            r * cfg.qDim(),
            r * cfg.kvDim(),
            r * cfg.kvDim(),
            r * cfg.qDim(),
            @max(nsplit, r * nsplit_prefill) * cfg.n_heads * (cfg.head_dim + 2),
            r * cfg.intermediate,
            r * cfg.intermediate,
            r * cfg.hidden,
            cfg.vocab,
        };
        var s: Bufs = undefined;
        var done: usize = 0;
        errdefer inline for (@typeInfo(Bufs).@"struct".fields, 0..) |f, i| {
            if (i < done) ctx.tensorDestroy(&@field(s, f.name));
        };
        inline for (@typeInfo(Bufs).@"struct".fields, 0..) |f, i| {
            @field(s, f.name) = try ctx.tensorCreate(sizes[i] * 4);
            done = i + 1;
        }
        return s;
    }

    fn deinit(s: *Bufs, ctx: *gpu.Context) void {
        inline for (@typeInfo(Bufs).@"struct".fields) |f| ctx.tensorDestroy(&@field(s, f.name));
    }
};
