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
//!
//! The stepper answers what `gemma3_cuda`'s does: a growable KV, turn checkpoints that
//! carry the rings, the CPU split (a ring's live window crosses to the host's linear
//! cache through `kvmod.ringSegments`), on-device sampling and weight noise.

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
const sample = @import("tp_core").sample;
const bnd = @import("tp_runtime").boundary;
const seqmod = @import("tp_runtime").seq;
const residency = @import("tp_runtime").residency;
const build_options = @import("build_options");

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
    /// `tp-llm --bench` alternates the `OpSDot` q8 decode GEMVs (toggle on: off) with
    /// the f32 ones.
    pub const bench_toggle = &gpu.Context.sdot_off;

    be: *gpu.Context,
    lm: *const gemma3.Model,
    cfg: gemma3.Config,
    gpa: std.mem.Allocator,
    /// Rows the global layers' KV holds now; `ensureCapacity` grows it toward
    /// `max_capacity` and `resetResidency` shrinks it back. The rings are fixed.
    capacity: usize,
    initial_capacity: usize,
    max_capacity: usize,
    kv_dtype: kvmod.KvDtype = .f32,
    len: usize,
    kv: kv_gpu.Kv = .empty,
    lin: lin_gpu.Lin,
    lin_plan: lin_gpu.Plan,
    /// Global (theta 1e6, scale 1/8) and local (theta 1e4) RoPE tables, each
    /// [2 * max_capacity * half] (cos then sin); sin_off = max_capacity * half.
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
    /// Polled before each prefill chunk: residency, pause and cancel (the GUI).
    boundary: ?bnd.Hook = null,
    /// The Io host layers of a split run on, set by every step.
    io: ?std.Io = null,
    /// Layers placed on the host, null when every layer is on the device.
    split: ?Split = null,

    /// The CPU split (`residency.zig` drives it), as gemma3_cuda's.
    pub const Split = struct {
        on_gpu: []bool,
        n_cpu: usize,
        policy: residency.CpuSplitPolicy,
        /// Host K/V for the host layers, linear (row = position) for rings too, in the
        /// device's byte layout; its length tracks `len`.
        cache: kvmod.KvCache,
        scratch: gemma3.Scratch,
        freqs_global: ops.rope.Freqs,
        freqs_local: ops.rope.Freqs,
        /// The hidden while host layers run, a whole chunk of rows.
        hx: []f32,
        /// Whether the live hidden sits in `hx` rather than `bufs.x`.
        on_host: bool = false,
        /// Migrate more layers to the host as the KV grows.
        dynamic: bool = false,
        /// Migration order; `order[0..next)` are on the host.
        order: []usize = &.{},
        next: usize = 0,
        /// The device usage the dynamic split keeps under.
        budget: u64 = 0,
    };

    /// One sequence's context (see `tp_runtime.seq`). `host` is the split's
    /// `cache` and RoPE tables, present exactly when a split is armed.
    pub const Seq = struct {
        capacity: usize = 0,
        max_capacity: usize = 0,
        len: usize = 0,
        kv: kv_gpu.Kv = .empty,
        host: ?Host = null,

        pub const Host = struct { cache: kvmod.KvCache, freqs_global: ops.rope.Freqs, freqs_local: ops.rope.Freqs };
    };

    /// An empty sequence for the current layer placement.
    fn seqAlloc(self: *const VulkanLM, initial: usize, max: usize) !Seq {
        const cfg = self.cfg;
        var s: Seq = .{ .capacity = initial, .max_capacity = max };
        errdefer self.seqFree(&s);
        s.kv = try self.newKv(initial, self.kv_dtype);
        if (self.split) |*sp| {
            for (sp.on_gpu, 0..) |on, l| if (!on) s.kv.release(self.be, l);
            var cache = try kvmod.KvCache.init(self.gpa, cfg.n_layers, initial, cfg.kvDim(), self.kv_dtype);
            errdefer cache.deinit(self.gpa);
            var fg = try ops.rope.rotateHalfFreqsScaled(self.gpa, initial, cfg.head_dim, cfg.rope_theta, cfg.rope_freq_scale);
            errdefer fg.deinit(self.gpa);
            const fl = try ops.rope.rotateHalfFreqsScaled(self.gpa, initial, cfg.head_dim, cfg.rope_theta_local, 1.0);
            s.host = .{ .cache = cache, .freqs_global = fg, .freqs_local = fl };
        }
        return s;
    }

    fn seqFree(self: *const VulkanLM, s: *Seq) void {
        s.kv.deinit(self.gpa, self.be);
        if (s.host) |*h| {
            h.cache.deinit(self.gpa);
            h.freqs_global.deinit(self.gpa);
            h.freqs_local.deinit(self.gpa);
        }
        s.* = .{};
    }

    /// A second, empty sequence of `rows` rows that never grows (see `gemma3_cuda`'s).
    pub fn seqCreate(self: *VulkanLM, rows: usize) !Seq {
        if (rows > self.max_capacity) return error.ContextFull;
        return self.seqAlloc(rows, rows);
    }

    /// Exchange the live sequence with `s`.
    pub fn seqSwap(self: *VulkanLM, s: *Seq) void {
        seqmod.swapShared(Seq, self, s);
        if (self.split) |*sp| {
            const h = &s.host.?;
            std.mem.swap(kvmod.KvCache, &sp.cache, &h.cache);
            std.mem.swap(ops.rope.Freqs, &sp.freqs_global, &h.freqs_global);
            std.mem.swap(ops.rope.Freqs, &sp.freqs_local, &h.freqs_local);
        } else std.debug.assert(s.host == null);
    }

    /// Free a sequence made by `seqCreate` once the device is done with it.
    pub fn seqDestroy(self: *VulkanLM, s: *Seq) void {
        self.be.waitIdle();
        self.seqFree(s);
    }

    pub fn init(gpa: std.mem.Allocator, ctx: *gpu.Context, lm: *const gemma3.Model, cap: kvmod.Capacity) !VulkanLM {
        const cfg = lm.cfg;
        var self: VulkanLM = init_defaults.of(VulkanLM);
        self.be = ctx;
        self.lm = lm;
        self.cfg = cfg;
        self.gpa = gpa;
        self.initial_capacity = @max(@min(cap.initial, cap.max), 1);
        self.kv_dtype = cap.kv_dtype;
        self.sin_off = cap.max * (cfg.head_dim / 2);
        ctx.weight_noise.setDepth(cfg.n_layers);

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

        var first = try self.seqAlloc(self.initial_capacity, cap.max);
        self.seqSwap(&first);
        errdefer {
            var live: Seq = .{};
            self.seqSwap(&live);
            self.seqFree(&live);
        }
        self.bufs = try Bufs.init(ctx, cfg);
        return self;
    }

    /// A cache with every local layer a ring and the global layers at `capacity` rows.
    fn newKv(self: *const VulkanLM, capacity: usize, dtype: kvmod.KvDtype) !kv_gpu.Kv {
        const cfg = self.cfg;
        const rings = try self.gpa.alloc(usize, cfg.n_layers);
        defer self.gpa.free(rings);
        for (rings, 0..) |*r, l| r.* = if (cfg.isGlobal(l)) 0 else ringRows(cfg);
        return kv_gpu.Kv.init(self.gpa, self.be, cfg.n_layers, capacity, cfg.kvDim(), dtype, rings);
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
        const ctx = self.be;
        self.freeSplit();
        var live: Seq = .{};
        self.seqSwap(&live);
        self.seqFree(&live);
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
    pub fn capacityMax(self: *const VulkanLM) usize {
        return self.max_capacity;
    }
    pub fn vocab(self: *const VulkanLM) usize {
        return self.cfg.vocab;
    }
    /// Device VRAM (bytes) this Vulkan context has allocated, for the end-of-response
    /// telemetry.
    pub fn vramUsed(self: *const VulkanLM) u64 {
        return self.be.device_used;
    }

    /// Grow the global layers' KV to hold `min_rows`, by the CUDA steppers' growth
    /// policy. A dynamic split first moves layers to the host until it can, and a real
    /// out-of-memory moves one more and retries.
    pub fn ensureCapacity(self: *VulkanLM, min_rows: usize) !void {
        if (min_rows <= self.capacity) return;
        if (min_rows > self.max_capacity) return error.ContextFull;
        const to = kvmod.growTarget(self.capacity, min_rows, self.max_capacity);
        const cfg = self.cfg;
        if (self.split) |*sp| if (sp.dynamic) {
            const add = self.kv_dtype.sizeBytes((to - self.capacity) * cfg.kvDim());
            while (true) {
                const need = self.liveGlobalLayers() * 2 * add + (32 << 20);
                const free = @min(sp.budget -| self.be.deviceUsed(), self.be.headroom());
                if (free >= need) break;
                if (!(try residency.migrateNext(self))) break;
            }
        };
        while (true) {
            self.kv.grow(self.be, to, self.len) catch |err| switch (err) {
                error.DeviceOutOfMemory, error.OutOfMemory => {
                    if (self.split != null and try residency.migrateNext(self)) continue;
                    return error.ContextFull;
                },
                else => return err,
            };
            break;
        }
        if (self.split) |*sp| {
            sp.cache.grow(self.gpa, to) catch return error.ContextFull;
            const fg = ops.rope.rotateHalfFreqsScaled(self.gpa, to, cfg.head_dim, cfg.rope_theta, cfg.rope_freq_scale) catch return error.ContextFull;
            sp.freqs_global.deinit(self.gpa);
            sp.freqs_global = fg;
            const fl = ops.rope.rotateHalfFreqsScaled(self.gpa, to, cfg.head_dim, cfg.rope_theta_local, 1.0) catch return error.ContextFull;
            sp.freqs_local.deinit(self.gpa);
            sp.freqs_local = fl;
        }
        self.capacity = to;
    }

    /// Advance the committed length by `n` new rows, the host shadow with it.
    fn advance(self: *VulkanLM, n: usize) void {
        self.len += n;
        if (self.split) |*sp| sp.cache.commit(n);
    }

    /// An empty context: the KV rows are left to be overwritten.
    pub fn resetCache(self: *VulkanLM) !void {
        self.len = 0;
        if (self.split) |*sp| sp.cache.truncate(0);
    }

    /// Re-create the KV at another element type, at the initial capacity, empty.
    pub fn reinitCache(self: *VulkanLM, dtype: kvmod.KvDtype) !void {
        var kv = try self.newKv(self.initial_capacity, dtype);
        errdefer kv.deinit(self.gpa, self.be);
        if (self.split) |*sp| {
            for (sp.on_gpu, 0..) |on, l| if (!on) kv.release(self.be, l);
            const cache = try kvmod.KvCache.init(self.gpa, self.cfg.n_layers, self.initial_capacity, self.cfg.kvDim(), dtype);
            sp.cache.deinit(self.gpa);
            sp.cache = cache;
        }
        self.kv.deinit(self.gpa, self.be);
        self.kv = kv;
        self.kv_dtype = dtype;
        self.capacity = self.initial_capacity;
        try self.resetCache();
    }

    /// A turn checkpoint's fixed size: every local layer's whole ring, K then V. The
    /// global layers are append-only and are truncated instead; a ring overwrites the
    /// rows a rollback needs back.
    pub fn checkpointBytes(self: *const VulkanLM) usize {
        var total: usize = 0;
        for (0..self.cfg.n_layers) |l| {
            if (!self.cfg.isGlobal(l)) total += 2 * self.ringBytes();
        }
        return total;
    }

    fn ringBytes(self: *const VulkanLM) usize {
        return self.kv_dtype.sizeBytes(ringRows(self.cfg) * self.cfg.kvDim());
    }

    /// Snapshot the rings at the current position, each from whichever side holds its
    /// layer, so a layer may move between snapshot and restore.
    pub fn checkpoint(self: *VulkanLM, out: []u8) !void {
        std.debug.assert(out.len == self.checkpointBytes());
        try self.checkpointRings(.save, out);
    }

    /// Roll back to `q` committed tokens with a snapshot taken there.
    pub fn restoreCheckpoint(self: *VulkanLM, snap: []const u8, q: usize) !void {
        std.debug.assert(snap.len == self.checkpointBytes() and q <= self.len);
        // First: a host layer's ring translation maps the window `q` defines.
        self.len = q;
        if (self.split) |*sp| sp.cache.truncate(q);
        try self.checkpointRings(.restore, snap);
    }

    const CheckpointDir = enum { save, restore };

    /// Each local layer's ring as ring rows, K then V: a device ring copies whole, a
    /// host layer's linear rows are translated per live segment.
    fn checkpointRings(self: *VulkanLM, comptime dir: CheckpointDir, buf: if (dir == .save) []u8 else []const u8) !void {
        const cfg = self.cfg;
        const kvd = cfg.kvDim();
        const dt = self.kv_dtype;
        const rb = self.ringBytes();
        var off: usize = 0;
        for (0..cfg.n_layers) |l| {
            if (cfg.isGlobal(l)) continue;
            const bk = buf[off..][0..rb];
            const bv = buf[off + rb ..][0..rb];
            off += 2 * rb;
            if (self.split == null or self.split.?.on_gpu[l]) {
                const k = try self.be.view(self.kv.k[l], 0, rb);
                const v = try self.be.view(self.kv.v[l], 0, rb);
                if (comptime dir == .save) {
                    try self.be.tensorDownload(k, bk);
                    try self.be.tensorDownload(v, bv);
                } else {
                    try self.be.tensorUpload(k, bk);
                    try self.be.tensorUpload(v, bv);
                }
                continue;
            }
            const sp = &self.split.?;
            for (kvmod.ringSegments(self.len, ringRows(cfg))) |seg| {
                if (seg.n == 0) continue;
                const hk = sp.cache.kRowBytes(l, seg.abs, seg.n);
                const hv = sp.cache.vRowBytes(l, seg.abs, seg.n);
                const rk = bk[dt.sizeBytes(seg.dev * kvd)..][0..dt.sizeBytes(seg.n * kvd)];
                const rv = bv[dt.sizeBytes(seg.dev * kvd)..][0..dt.sizeBytes(seg.n * kvd)];
                if (comptime dir == .save) {
                    @memcpy(rk, hk);
                    @memcpy(rv, hv);
                } else {
                    @memcpy(hk, rk);
                    @memcpy(hv, rv);
                }
            }
        }
    }

    /// Forward `ids` in chunks, then write the last position's logits.
    pub fn step(self: *VulkanLM, io: std.Io, ids: []const u32, logits: []f32) !void {
        self.io = io;
        std.debug.assert(ids.len >= 1);
        try self.prefillIds(ids, true);
        try self.be.tensorDownload(self.bufs.logits, std.mem.sliceAsBytes(logits[0..self.cfg.vocab]));
    }

    /// Greedy decode without the vocab download: argmax on the device.
    pub fn stepArgmax(self: *VulkanLM, io: std.Io, ids: []const u32) !u32 {
        return self.stepArgmaxPen(io, ids, &.{}, .{});
    }

    /// `stepArgmax` with the sampling penalties scattered onto the device logits first.
    pub fn stepArgmaxPen(self: *VulkanLM, io: std.Io, ids: []const u32, pen: []const sample.PenaltyEntry, sp: sample.Params) !u32 {
        self.io = io;
        try self.prefillIds(ids, true);
        const b = &self.bufs;
        try self.be.opPenalize(b.logits, pen, sp);
        try self.be.opArgmax(b.logits, self.cfg.vocab, b.argmax_out, &b.argmax_v, &b.argmax_i);
        var id_f: [1]f32 = undefined;
        try self.be.tensorDownload(b.argmax_out, std.mem.sliceAsBytes(&id_f));
        return @intFromFloat(id_f[0]);
    }

    /// Max candidates stepSelect can return (host buffer sizing for the engine).
    pub fn maxSelect(self: *const VulkanLM) usize {
        _ = self;
        return gpu.topk_lanes * gpu.topk_m;
    }

    /// Stochastic decode: the top-k candidates selected on the device and only those
    /// downloaded; the engine's Sampler finishes on the CPU.
    pub fn stepSelect(self: *VulkanLM, io: std.Io, ids: []const u32, out_id: []u32, out_logit: []f32) !usize {
        return self.stepSelectPen(io, ids, &.{}, .{}, out_id, out_logit);
    }

    /// `stepSelect` with the penalties applied before the top-k.
    pub fn stepSelectPen(self: *VulkanLM, io: std.Io, ids: []const u32, pen: []const sample.PenaltyEntry, sp: sample.Params, out_id: []u32, out_logit: []f32) !usize {
        self.io = io;
        try self.prefillIds(ids, true);
        const b = &self.bufs;
        try self.be.opPenalize(b.logits, pen, sp);
        const count = try self.be.opTopK(b.logits, self.cfg.vocab, &b.topk_v, &b.topk_i);
        std.debug.assert(count <= out_id.len and count <= out_logit.len);
        try self.be.tensorDownload(b.topk_v, std.mem.sliceAsBytes(out_logit[0..count]));
        var idx_f: [gpu.topk_lanes * gpu.topk_m]f32 = undefined;
        try self.be.tensorDownload(b.topk_i, std.mem.sliceAsBytes(idx_f[0..count]));
        for (out_id[0..count], idx_f[0..count]) |*o, f| o.* = @intFromFloat(f);
        return count;
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
        try bnd.check(self);
        self.bidir_end = self.len + n;
        defer self.bidir_end = 0;
        try self.forwardRows(embeds, false);
    }

    /// Upload every weight now rather than inside the first turn, by running a
    /// prefill-sized and a decode forward: those are the routes, and so the device
    /// layouts, real turns read. The context is emptied after.
    pub fn warmWeights(self: *VulkanLM) void {
        if (self.len != 0) return;
        const zeros: [32]u32 = @splat(0);
        const n = @min(zeros.len, self.capacity - 1);
        self.prefillIds(zeros[0..n], false) catch |err| std.log.warn("gemma3 vulkan warm-up: {t}", .{err});
        self.prefillIds(zeros[0..1], true) catch |err| std.log.warn("gemma3 vulkan warm-up: {t}", .{err});
        self.resetCache() catch |err| std.log.warn("gemma3 vulkan warm-up: {t}", .{err});
    }

    fn prefillIds(self: *VulkanLM, ids: []const u32, want_logits: bool) !void {
        const cfg = self.cfg;
        std.debug.assert(ids.len <= self.remaining());
        const x = try self.gpa.alloc(f32, @min(ids.len, prefill_chunk) * cfg.hidden);
        defer self.gpa.free(x);
        const scale = cfg.embedScale();
        var off: usize = 0;
        while (off < ids.len) {
            if (!want_logits) try bnd.check(self);
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
        const ctx = self.be;
        const cfg = self.cfg;
        const b = &self.bufs;
        const n = x.len / cfg.hidden;
        std.debug.assert(n >= 1 and n <= prefill_chunk and n <= self.remaining());
        const pos0 = self.len;
        // Before the batch opens: an upload inside it would flush the recording.
        try ctx.tensorUpload(try ctx.view(b.x, 0, x.len * 4), std.mem.sliceAsBytes(x));
        ctx.weight_noise.tick();
        try ctx.beginBatch();
        errdefer if (ctx.batching) ctx.abortBatch();
        self.residual_pending = false;
        if (self.split) |*sp| sp.on_host = false;
        const ctok = ops.cancel.token;
        for (self.lm.layers, 0..) |*layer, l| {
            if (ops.cancel.canceled(ctok)) return error.Canceled;
            // A host layer runs the CPU backend's own layer body; the hidden crosses
            // only where residency changes.
            if (self.split) |*sp| {
                if (!sp.on_gpu[l]) {
                    if (!sp.on_host) {
                        try self.flushResidual(n);
                        try ctx.endBatch();
                        try ctx.tensorDownload(try ctx.view(b.x, 0, n * cfg.hidden * 4), std.mem.sliceAsBytes(sp.hx[0 .. n * cfg.hidden]));
                        sp.on_host = true;
                    }
                    const host_io = self.io orelse return error.SplitIoUnset;
                    var sv = sp.scratch.viewSeq(n, cfg);
                    try gemma3.layerForward(host_io, self.gpa, cfg, layer, sp.hx[0 .. n * cfg.hidden], n, sp.freqs_global, sp.freqs_local, &sp.cache, l, self.bidir_end != 0, &sv);
                    continue;
                }
                if (sp.on_host) try self.hiddenToDevice(n);
            }
            try transformer_gpu.decoderLayer(transformer.gemma3_spec, self, layer, l, n, pos0);
        }
        if (self.split) |*sp| if (sp.on_host) try self.hiddenToDevice(n);
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
            ctx.weight_noise.atHead();
            try self.lin.linear(self.lin_plan, b.logits, 0, last, 1, self.lm.head);
        }
        try ctx.endBatch();
        self.advance(n);
    }

    /// The host hidden back into `bufs.x`, and a batch open again: an upload inside
    /// one would flush its recording.
    fn hiddenToDevice(self: *VulkanLM, seq: usize) !void {
        const sp = &self.split.?;
        const n = seq * self.cfg.hidden;
        try self.be.tensorUpload(try self.be.view(self.bufs.x, 0, n * 4), std.mem.sliceAsBytes(sp.hx[0..n]));
        sp.on_host = false;
        try self.be.beginBatch();
    }

    /// Add a pending residual into `bufs.x`, for a reader other than the next norm.
    fn flushResidual(self: *VulkanLM, seq: usize) !void {
        if (!self.residual_pending) return;
        const n = seq * self.cfg.hidden;
        try self.be.opElt(gpu.eltAt("add", .f32), self.bufs.x, self.bufs.t, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
        self.residual_pending = false;
    }

    /// Which layer the coming launches belong to, for the weight-noise curve.
    pub fn noiseAtLayer(self: *VulkanLM, l: usize) void {
        self.be.weight_noise.atLayer(l);
    }

    // Residency: the hooks `residency.zig` drives, with gemma3_cuda's meaning.

    fn layerDeviceBytes(layer: anytype) usize {
        return layer.q.bytes.len + layer.k.bytes.len + layer.v.bytes.len + layer.o.bytes.len +
            layer.gate.bytes.len + layer.up.bytes.len + layer.down.bytes.len;
    }

    pub fn layerWeightBytes(self: *VulkanLM, l: usize) usize {
        return layerDeviceBytes(&self.lm.layers[l]);
    }

    /// The tied head; the embeddings are looked up on the host.
    pub fn nonLayerDeviceBytes(self: *VulkanLM) u64 {
        return self.lm.head.bytes.len;
    }

    /// One layer's K and V at the current capacity: a ring's fixed rows, a global
    /// layer's capacity.
    fn layerKvBytes(self: *const VulkanLM, l: usize) usize {
        const rows = if (self.cfg.isGlobal(l)) self.capacity else ringRows(self.cfg);
        return 2 * self.kv_dtype.sizeBytes(rows * self.cfg.kvDim());
    }

    /// VRAM a promote of layer `l` needs: its weights, its K/V, and slack.
    pub fn promoteCost(self: *VulkanLM, l: usize) usize {
        return layerDeviceBytes(&self.lm.layers[l]) + self.layerKvBytes(l) + residency.promote_slack;
    }

    fn liveGlobalLayers(self: *const VulkanLM) usize {
        var count: usize = 0;
        for (0..self.cfg.n_layers) |l| {
            if (!self.cfg.isGlobal(l)) continue;
            if (self.split) |sp| if (!sp.on_gpu[l]) continue;
            count += 1;
        }
        return count;
    }

    /// Move layer `l` to the host: its live K/V into the shadow, its device K/V and
    /// weights freed.
    pub fn migrateLayer(self: *VulkanLM, l: usize) !void {
        const sp = &self.split.?;
        try self.kv.dropLayer(self.be, l, self.len, &sp.cache, l);
        const layer = &self.lm.layers[l];
        for ([_][]const u8{ layer.q.bytes, layer.k.bytes, layer.v.bytes, layer.o.bytes, layer.gate.bytes, layer.up.bytes, layer.down.bytes }) |w|
            self.be.evictWeightBytes(w);
        sp.on_gpu[l] = false;
        sp.n_cpu += 1;
        std.log.debug("[offload] layer {d} -> CPU at ctx {d} ({d}/{d} on CPU)", .{ l, self.len, sp.n_cpu, self.cfg.n_layers });
    }

    /// Bring layer `l` back to the device with its K/V; its weights re-upload on the
    /// next forward.
    pub fn promoteLayer(self: *VulkanLM, l: usize) !void {
        const sp = &self.split.?;
        try self.kv.restoreLayer(self.be, l, self.capacity, self.len, &sp.cache, l);
        sp.on_gpu[l] = true;
        sp.n_cpu -= 1;
        std.log.debug("[promote] layer {d} -> GPU at ctx {d} ({d}/{d} on CPU)", .{ l, self.len, sp.n_cpu, self.cfg.n_layers });
    }

    pub fn promoteLayers(self: *VulkanLM, budget: u64) !usize {
        return residency.promoteBack(self, budget);
    }

    pub fn offloadUntilFree(self: *VulkanLM, needed_free: u64) !void {
        return residency.offloadUntilFree(self, needed_free);
    }

    pub fn offloadToBudget(self: *VulkanLM, target: u64) !void {
        return residency.offloadToBudget(self, target);
    }

    /// Always arm the dynamic split (`budget == 0` is no offload).
    pub fn autoOffload(self: *VulkanLM, budget: u64) !bool {
        if (budget == 0) return false;
        try self.enableCpuSplit(.attn, budget, true);
        return true;
    }

    /// Place layers on the host until the device-resident weights fit under `budget`.
    /// The last layer leaves first; gemma3 is all attention, so `.attn` and `.tail`
    /// agree.
    pub fn enableCpuSplit(self: *VulkanLM, policy: residency.CpuSplitPolicy, budget: u64, dynamic: bool) !void {
        std.debug.assert(self.split == null);
        const cfg = self.cfg;
        const n = cfg.n_layers;
        const gpa = self.gpa;
        const per = try gpa.alloc(usize, n);
        defer gpa.free(per);
        for (self.lm.layers, 0..) |*layer, l| per[l] = layerDeviceBytes(layer);
        var kv_bytes: usize = 0;
        for (0..n) |l| kv_bytes += self.layerKvBytes(l);
        const reserve = if (dynamic) self.lm.head.bytes.len else kv_bytes + self.lm.head.bytes.len + (512 << 20);
        const order = try gpa.alloc(usize, n);
        errdefer gpa.free(order);
        for (0..n) |i| order[i] = n - 1 - i;
        const on_gpu = try gpa.alloc(bool, n);
        errdefer gpa.free(on_gpu);
        @memset(on_gpu, true);
        const n_cpu = residency.splitHostCount(self, budget, reserve, dynamic, per, order);
        if (n_cpu == 0 and !dynamic) {
            gpa.free(on_gpu);
            gpa.free(order);
            return;
        }
        var cache = try kvmod.KvCache.init(gpa, n, self.capacity, cfg.kvDim(), self.kv_dtype);
        errdefer cache.deinit(gpa);
        cache.len = self.len;
        var scratch = try gemma3.Scratch.init(gpa, prefill_chunk, cfg);
        errdefer scratch.deinit(gpa);
        var fg = try ops.rope.rotateHalfFreqsScaled(gpa, self.capacity, cfg.head_dim, cfg.rope_theta, cfg.rope_freq_scale);
        errdefer fg.deinit(gpa);
        var fl = try ops.rope.rotateHalfFreqsScaled(gpa, self.capacity, cfg.head_dim, cfg.rope_theta_local, 1.0);
        errdefer fl.deinit(gpa);
        const hx = try gpa.alloc(f32, prefill_chunk * cfg.hidden);
        errdefer gpa.free(hx);
        self.split = .{
            .on_gpu = on_gpu,
            .n_cpu = 0,
            .policy = policy,
            .cache = cache,
            .scratch = scratch,
            .freqs_global = fg,
            .freqs_local = fl,
            .hx = hx,
            .dynamic = dynamic,
            .order = order,
            .next = n_cpu,
            .budget = budget,
        };
        for (order[0..n_cpu]) |l| try self.migrateLayer(l);
    }

    fn freeSplit(self: *VulkanLM) void {
        const sp = &(self.split orelse return);
        sp.cache.deinit(self.gpa);
        sp.scratch.deinit(self.gpa);
        sp.freqs_global.deinit(self.gpa);
        sp.freqs_local.deinit(self.gpa);
        self.gpa.free(sp.hx);
        self.gpa.free(sp.on_gpu);
        self.gpa.free(sp.order);
        self.split = null;
    }

    /// New chat: drop the split, shrink the KV back to its initial capacity, empty it,
    /// and re-arm the dynamic offload.
    pub fn resetResidency(self: *VulkanLM, budget: u64) !void {
        self.freeSplit();
        const kv = try self.newKv(self.initial_capacity, self.kv_dtype);
        self.kv.deinit(self.gpa, self.be);
        self.kv = kv;
        self.capacity = self.initial_capacity;
        try self.resetCache();
        _ = try self.autoOffload(budget);
    }

    /// `normed = rmsnorm(x) * w` over `rows` rows, adding the pending residual into x
    /// first.
    fn hiddenNorm(self: *VulkanLM, w: []const f32, rows: usize) !void {
        const b = &self.bufs;
        try self.be.opAddRmsNorm(b.x, if (self.residual_pending) b.t else null, try nbuf(self.be, w), b.normed, rows, self.cfg.hidden, self.cfg.rms_eps);
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
            if (try self.kv.rowsAt(self.be, l, self.len, 1)) |rows| return rows;
        return .{ self.bufs.k, self.bufs.v };
    }
    /// The rope too, in the same pass: the position is the cache length, which
    /// `forwardRows` passes as pos0.
    pub fn normQK(self: *VulkanLM, l: usize, layer: anytype, seq: usize) !void {
        const cfg = self.cfg;
        const ctx = self.be;
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
        try self.kv.store(self.be, l, self.bufs.k, self.bufs.v, pos0, seq);
    }
    pub fn attention(self: *VulkanLM, l: usize, seq: usize, pos0: usize) !void {
        const cfg = self.cfg;
        const b = &self.bufs;
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));
        const window: usize = if (cfg.isGlobal(l)) 0 else cfg.sliding_window;
        const ns: usize = if (seq == 1) nsplit else nsplit_prefill;
        try self.kv.attend(self.be, l, b.q, b.attn, b.attn_scratch, pos0 + 1, seq, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, ns, scale, window, self.bidir_end);
    }
    pub fn projectO(self: *VulkanLM, l: usize, layer: anytype, seq: usize) !void {
        _ = l;
        try self.lin.linear(self.lin_plan, self.bufs.t, 0, self.bufs.attn, seq, layer.o);
    }
    pub fn postAttnNorm(self: *VulkanLM, layer: anytype, seq: usize) !void {
        const b = &self.bufs;
        try self.be.opAddRmsNorm(b.t, null, try nbuf(self.be, layer.post_attn_norm), b.t, seq, self.cfg.hidden, self.cfg.rms_eps);
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
        try self.be.opElt(which, self.bufs.gate, self.bufs.up, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
    }
    pub fn projectDown(self: *VulkanLM, layer: anytype, seq: usize) !void {
        try self.lin.linear(self.lin_plan, self.bufs.t, 0, self.bufs.gate, seq, layer.down);
    }
    pub fn postFfnNorm(self: *VulkanLM, layer: anytype, seq: usize) !void {
        const b = &self.bufs;
        try self.be.opAddRmsNorm(b.t, null, try nbuf(self.be, layer.post_ffn_norm), b.t, seq, self.cfg.hidden, self.cfg.rms_eps);
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
    // On-device sampling scratch: argmax per-lane max and index and the one id out;
    // top-k per-lane values and indices.
    argmax_v: Buf,
    argmax_i: Buf,
    argmax_out: Buf,
    topk_v: Buf,
    topk_i: Buf,

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
            4096,
            4096,
            1,
            gpu.topk_lanes * gpu.topk_m,
            gpu.topk_lanes * gpu.topk_m,
        };
        var s: Bufs = undefined;
        var done: usize = 0;
        errdefer inline for (@typeInfo(Bufs).@"struct".field_names, 0..) |name, i| {
            if (i < done) ctx.tensorDestroy(&@field(s, name));
        };
        inline for (@typeInfo(Bufs).@"struct".field_names, 0..) |name, i| {
            @field(s, name) = try ctx.tensorCreate(sizes[i] * 4);
            done = i + 1;
        }
        return s;
    }

    fn deinit(s: *Bufs, ctx: *gpu.Context) void {
        inline for (@typeInfo(Bufs).@"struct".field_names) |name| ctx.tensorDestroy(&@field(s, name));
    }
};

const test_path = build_options.model_root ++ "/../llm/lmstudio-community/gemma-3-12b-it-GGUF/gemma-3-12b-it-Q4_K_M.gguf";

/// A device and the test checkpoint, or a skip. Heap-held: the model borrows the
/// container's mapping.
const Fixture = struct {
    ctx: *gpu.Context,
    g: @import("tp_core").gguf.Gguf,
    lm: gemma3.Model,

    fn open(gpa: std.mem.Allocator, io: std.Io) !*Fixture {
        std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
        std.Io.Dir.cwd().access(io, test_path, .{}) catch return error.SkipZigTest;
        const f = try gpa.create(Fixture);
        errdefer gpa.destroy(f);
        f.ctx = gpu.Context.init(gpa, io) catch return error.SkipZigTest;
        errdefer f.ctx.deinit();
        f.g = try @TypeOf(f.g).open(gpa, io, test_path);
        errdefer f.g.deinit();
        f.lm = try gemma3.Model.load(gpa, &f.g);
        return f;
    }

    fn close(f: *Fixture, gpa: std.mem.Allocator) void {
        f.lm.deinit();
        f.g.deinit();
        f.ctx.deinit();
        gpa.destroy(f);
    }
};

fn testPrompt(gpa: std.mem.Allocator, n: usize) ![]u32 {
    const p = try gpa.alloc(u32, n);
    for (p, 0..) |*t, i| t.* = @intCast(1000 + (i * 37) % 50000);
    return p;
}

/// `out.len` greedy tokens from `first`, growing the KV as the engine does.
fn greedy(m: *VulkanLM, io: std.Io, first: u32, out: []u32) !void {
    var cur = first;
    for (out) |*t| {
        try m.ensureCapacity(m.cached() + 1);
        cur = try m.stepArgmax(io, &.{cur});
        t.* = cur;
    }
}

fn relL2(want: []const f32, got: []const f32) f64 {
    var num: f64 = 0;
    var den: f64 = 0;
    for (want, got) |a, b| {
        num += (@as(f64, a) - b) * (@as(f64, a) - b);
        den += @as(f64, a) * a;
    }
    return @sqrt(num / den);
}

// Rollback past the ring slack on the real model: the prompt wraps every ring and the
// reply overwrites live-window rows, so only the checkpoint's ring snapshot brings them
// back and the regenerated reply must be token-identical. Then a mid-conversation split
// (host layers translate the ring to linear rows) must repeat itself, and after every
// layer comes back the reply equals the unsplit one exactly. Those two legs test the
// restore, which precedes generation, so they generate only a short prefix.
fn checkpointRingBody(kv_dtype: kvmod.KvDtype) !void {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const f = try Fixture.open(gpa, io);
    defer f.close(gpa);
    const ring = ringRows(f.lm.cfg);
    const prompt_len = ring + 64;
    const n_gen = (ring - f.lm.cfg.sliding_window) + 64;
    const total = prompt_len + n_gen + 8;
    var m = try VulkanLM.init(gpa, f.ctx, &f.lm, .{ .initial = total, .max = total, .kv_dtype = kv_dtype });
    defer m.deinit();
    const prompt = try testPrompt(gpa, prompt_len);
    defer gpa.free(prompt);
    const last = prompt[prompt.len - 1];

    try m.prefill(prompt[0 .. prompt.len - 1]);
    const q = m.cached();
    const snap = try gpa.alloc(u8, m.checkpointBytes());
    defer gpa.free(snap);
    try m.checkpoint(snap);
    var seqs: [5][]u32 = undefined;
    for (&seqs, [5]usize{ n_gen, n_gen, 16, 16, 16 }) |*sq, len| sq.* = try gpa.alloc(u32, len);
    defer for (seqs) |sq| gpa.free(sq);
    const logits = try gpa.alloc(f32, f.lm.cfg.vocab);
    defer gpa.free(logits);
    const logits2 = try gpa.alloc(f32, f.lm.cfg.vocab);
    defer gpa.free(logits2);
    const snap2 = try gpa.alloc(u8, snap.len);
    defer gpa.free(snap2);

    try greedy(&m, io, last, seqs[0]);
    try m.restoreCheckpoint(snap, q);
    try std.testing.expectEqual(q, m.cached());
    try greedy(&m, io, last, seqs[1]);
    try std.testing.expectEqualSlices(u32, seqs[0], seqs[1]);

    try m.enableCpuSplit(.tail, std.math.maxInt(u64), true);
    try m.offloadToBudget(f.ctx.deviceUsed() * 4 / 5);
    try std.testing.expect(m.split.?.n_cpu > 0);
    // A restore reaches every layer where it now lives: a checkpoint taken right after
    // it reads the same bytes back. Logits too: greedy alone can hide a stale ring row.
    try m.restoreCheckpoint(snap, q);
    try m.checkpoint(snap2);
    try std.testing.expectEqualSlices(u8, snap, snap2);
    try m.step(io, &.{last}, logits);
    try m.restoreCheckpoint(snap, q);
    try greedy(&m, io, last, seqs[2]);
    try std.testing.expectEqual(m.cached(), m.split.?.cache.len);
    try m.restoreCheckpoint(snap, q);
    try m.step(io, &.{last}, logits2);
    try std.testing.expectEqualSlices(f32, logits, logits2);
    try m.restoreCheckpoint(snap, q);
    try greedy(&m, io, last, seqs[3]);
    try std.testing.expectEqualSlices(u32, seqs[2], seqs[3]);

    try std.testing.expect(try m.promoteLayers(std.math.maxInt(u64)) > 0);
    try std.testing.expectEqual(@as(usize, 0), m.split.?.n_cpu);
    try m.restoreCheckpoint(snap, q);
    try greedy(&m, io, last, seqs[4]);
    try std.testing.expectEqualSlices(u32, seqs[0][0..seqs[4].len], seqs[4]);
}

test "vulkan gemma3 checkpoint restores the rings past their slack, split and promoted" {
    try checkpointRingBody(.f32);
}

test "vulkan gemma3 ring checkpoint (f16 kv)" {
    try checkpointRingBody(.f16);
}

test "vulkan gemma3 ring checkpoint (q8_0 kv)" {
    try checkpointRingBody(.q8_0);
}

// The GUI turn shape: warm-up, a reset and replay, the device sampler against the
// downloaded logits of the same step, and a KV dtype rebuild that must match a stepper
// built at that dtype.
test "vulkan gemma3 gui turn lifecycle: reset, sampling, kv rebuild" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const f = try Fixture.open(gpa, io);
    defer f.close(gpa);
    const prompt = try testPrompt(gpa, 40);
    defer gpa.free(prompt);
    const last = prompt[prompt.len - 1];

    var m = try VulkanLM.init(gpa, f.ctx, &f.lm, .{ .initial = 256, .max = 256 });
    defer m.deinit();
    m.warmWeights();
    try std.testing.expectEqual(@as(usize, 0), m.cached());
    try m.prefill(prompt[0 .. prompt.len - 1]);
    const q = m.cached();
    const snap = try gpa.alloc(u8, m.checkpointBytes());
    defer gpa.free(snap);
    try m.checkpoint(snap);
    var seq1: [12]u32 = undefined;
    var seq2: [12]u32 = undefined;
    try greedy(&m, io, last, &seq1);
    try m.resetCache();
    try std.testing.expectEqual(@as(usize, 0), m.cached());
    try m.prefill(prompt[0 .. prompt.len - 1]);
    try greedy(&m, io, last, &seq2);
    try std.testing.expectEqualSlices(u32, &seq1, &seq2);

    const logits = try gpa.alloc(f32, m.vocab());
    defer gpa.free(logits);
    try m.restoreCheckpoint(snap, q);
    try m.step(io, &.{last}, logits);
    try std.testing.expectEqual(seq1[0], @as(u32, @intCast(sample.argmax(logits))));
    try m.restoreCheckpoint(snap, q);
    var ids: [gpu.topk_lanes * gpu.topk_m]u32 = undefined;
    var vals: [gpu.topk_lanes * gpu.topk_m]f32 = undefined;
    try std.testing.expectEqual(ids.len, m.maxSelect());
    const count = try m.stepSelect(io, &.{last}, &ids, &vals);
    try std.testing.expect(count > 0);
    var best: usize = 0;
    for (ids[0..count], vals[0..count], 0..) |id, v, i| {
        try std.testing.expectEqual(logits[id], v);
        if (v > vals[best]) best = i;
    }
    try std.testing.expectEqual(seq1[0], ids[best]);

    try m.reinitCache(.f16);
    try std.testing.expectEqual(kvmod.KvDtype.f16, m.kv_dtype);
    try m.prefill(prompt[0 .. prompt.len - 1]);
    try greedy(&m, io, last, &seq1);
    var fresh = try VulkanLM.init(gpa, f.ctx, &f.lm, .{ .initial = 256, .max = 256, .kv_dtype = .f16 });
    defer fresh.deinit();
    try fresh.prefill(prompt[0 .. prompt.len - 1]);
    try greedy(&fresh, io, last, &seq2);
    try std.testing.expectEqualSlices(u32, &seq1, &seq2);
}

// A cache that starts at 16 rows and grows through a reply decodes exactly what one
// built at full size does.
test "vulkan gemma3 kv growth decodes what a full-size cache does" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const f = try Fixture.open(gpa, io);
    defer f.close(gpa);
    const prompt = try testPrompt(gpa, 24);
    defer gpa.free(prompt);
    var outs: [2][40]u32 = undefined;
    for (&outs, [2]usize{ 16, 256 }) |*out, initial| {
        var m = try VulkanLM.init(gpa, f.ctx, &f.lm, .{ .initial = initial, .max = 256, .kv_dtype = .f16 });
        defer m.deinit();
        try m.ensureCapacity(prompt.len);
        try m.prefill(prompt[0 .. prompt.len - 1]);
        try greedy(&m, io, prompt[prompt.len - 1], out);
        if (initial == 16) try std.testing.expect(m.cached() + m.remaining() > 32);
    }
    errdefer std.debug.print("grown {any}\nfull  {any}\n", .{ outs[0], outs[1] });
    try std.testing.expectEqualSlices(u32, &outs[1], &outs[0]);
}

// Host layers need an Io and track the device; layers moved mid-context, rings wrapped,
// carry their live rows and bring them back byte for byte, so after the promote the
// logits equal an unsplit stepper's.
test "vulkan gemma3 cpu split tracks the device and moves wrapped rings losslessly" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const f = try Fixture.open(gpa, io);
    defer f.close(gpa);
    const ring = ringRows(f.lm.cfg);
    const k = ring + 40;
    const n = k + 24;
    const prompt = try testPrompt(gpa, n);
    defer gpa.free(prompt);
    const vocab = f.lm.cfg.vocab;
    const want = try gpa.alloc(f32, vocab);
    defer gpa.free(want);
    const got = try gpa.alloc(f32, vocab);
    defer gpa.free(got);

    for ([_]kvmod.KvDtype{ .f32, .f16, .q8_0 }) |dt| {
        {
            var ref = try VulkanLM.init(gpa, f.ctx, &f.lm, .{ .initial = n, .max = n, .kv_dtype = dt });
            defer ref.deinit();
            try ref.prefill(prompt[0..k]);
            try ref.prefill(prompt[k .. n - 1]);
            try ref.step(io, prompt[n - 1 ..], want);
        }
        var m = try VulkanLM.init(gpa, f.ctx, &f.lm, .{ .initial = n, .max = n, .kv_dtype = dt });
        defer m.deinit();
        try m.prefill(prompt[0..k]);
        // ~6 GB of q4_k layer weights: 4 GiB puts the tail layers on the host.
        try m.enableCpuSplit(.attn, 4 << 30, true);
        try std.testing.expect(m.split.?.n_cpu > 0 and m.split.?.n_cpu < m.cfg.n_layers);
        try std.testing.expectError(error.SplitIoUnset, m.prefill(prompt[k .. n - 1]));
        m.io = io;
        {
            // A host-split run against the device one: same chunks, different arithmetic.
            const snap_len = m.cached();
            try std.testing.expectEqual(k, snap_len);
            try m.prefill(prompt[k .. n - 1]);
            try m.step(io, prompt[n - 1 ..], got);
            const rel = relL2(want, got);
            errdefer std.debug.print("kv {t}: split vs device logits rel L2 {e:.3}\n", .{ dt, rel });
            try std.testing.expect(rel < 5e-2);
        }
        // Back to `k` rows: the split took them at k, so the rings still hold them.
        m.len = k;
        m.split.?.cache.truncate(k);
        _ = try m.promoteLayers(std.math.maxInt(u64));
        try std.testing.expectEqual(@as(usize, 0), m.split.?.n_cpu);
        try m.prefill(prompt[k .. n - 1]);
        try m.step(io, prompt[n - 1 ..], got);
        errdefer std.debug.print("kv {t}: logits after the round trip differ from the unsplit run\n", .{dt});
        try std.testing.expectEqualSlices(f32, want, got);
    }
}

// The GUI's KV dtype toggle with the split armed, and a residency reset re-arming it.
test "vulkan gemma3 cpu split survives a kv dtype toggle, and a residency reset re-arms it" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const f = try Fixture.open(gpa, io);
    defer f.close(gpa);
    var m = try VulkanLM.init(gpa, f.ctx, &f.lm, .{ .initial = 64, .max = 256 });
    defer m.deinit();
    try m.enableCpuSplit(.attn, 4 << 30, true);
    try std.testing.expect(m.split.?.n_cpu > 0);
    m.io = io;
    const prompt = try testPrompt(gpa, 24);
    defer gpa.free(prompt);
    for ([_]kvmod.KvDtype{ .f16, .q8_0, .f32 }) |dt| {
        const n_cpu_before = m.split.?.n_cpu;
        try m.reinitCache(dt);
        const sp = &m.split.?;
        try std.testing.expectEqual(dt, m.kv_dtype);
        try std.testing.expectEqual(dt, sp.cache.kv_dtype);
        try std.testing.expectEqual(@as(usize, 0), sp.cache.len);
        try std.testing.expectEqual(n_cpu_before, sp.n_cpu);
        for (sp.on_gpu, 0..) |on, l| try std.testing.expectEqual(on, m.kv.k[l].buf != .null_handle);
        try m.prefill(prompt[0 .. prompt.len - 1]);
        try std.testing.expectEqual(prompt.len - 1, sp.cache.len);
        try std.testing.expect(try m.stepArgmax(io, prompt[prompt.len - 1 ..]) < m.cfg.vocab);
    }
    try m.ensureCapacity(200);
    try std.testing.expect(m.cached() + m.remaining() >= 200);
    try m.resetResidency(4 << 30);
    try std.testing.expectEqual(@as(usize, 0), m.cached());
    try std.testing.expectEqual(@as(usize, 64), m.remaining());
    try std.testing.expect(m.split.?.n_cpu > 0);
    try m.prefill(prompt[0 .. prompt.len - 1]);
    _ = try m.stepArgmax(io, prompt[prompt.len - 1 ..]);
}

// Weight noise through the prefill's dequant GEMM and the decode GEMV: off is exact,
// on moves both, the same draw counter reproduces both, and off again restores them.
test "vulkan gemma3 weight noise perturbs prefill and decode, reproducibly, and off is exact" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const f = try Fixture.open(gpa, io);
    defer f.close(gpa);
    var m = try VulkanLM.init(gpa, f.ctx, &f.lm, .{ .initial = 512, .max = 512 });
    defer m.deinit();
    const prompt = try testPrompt(gpa, 300);
    defer gpa.free(prompt);
    var bufs: [6][]f32 = undefined;
    for (&bufs) |*b| b.* = try gpa.alloc(f32, f.lm.cfg.vocab);
    defer for (bufs) |b| gpa.free(b);
    const off_pre, const off_dec, const on_pre, const on_dec, const re_pre, const re_dec = bufs;

    const run = struct {
        fn go(mm: *VulkanLM, io_: std.Io, p: []const u32, pre: []f32, dec: []f32) !void {
            try mm.resetCache();
            mm.be.weight_noise.seq = 0;
            try mm.step(io_, p, pre);
            try mm.step(io_, &.{p[0]}, dec);
        }
    }.go;
    try run(&m, io, prompt, off_pre, off_dec);
    const noise = &f.ctx.weight_noise;
    noise.seed = 7;
    noise.setCurve("0.05");
    try std.testing.expect(noise.on());
    try run(&m, io, prompt, on_pre, on_dec);
    try std.testing.expect(!std.mem.eql(f32, off_pre, on_pre));
    try std.testing.expect(!std.mem.eql(f32, off_dec, on_dec));
    try run(&m, io, prompt, re_pre, re_dec);
    try std.testing.expectEqualSlices(f32, on_pre, re_pre);
    try std.testing.expectEqualSlices(f32, on_dec, re_dec);
    noise.setCurve("");
    try run(&m, io, prompt, re_pre, re_dec);
    try std.testing.expectEqualSlices(f32, off_pre, re_pre);
    try std.testing.expectEqualSlices(f32, off_dec, re_dec);
}
