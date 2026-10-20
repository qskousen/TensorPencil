//! Gemma 4 on the Vulkan backend, the `gemma4_cuda` of this backend:
//! `transformer_gpu.decoderLayer` over this stepper's ops, batched at any position,
//! every linear through `lin_gpu`, attention the dual kernel against `kv_gpu.Kv`. Text,
//! and image blocks prefilled bidirectionally in one batch.
//!
//! Against gemma3_gpu: per-layer geometry (local layers head_dim 256 with 8 KV heads,
//! global ones head_dim 512 with one), so the KV has a per-layer row width; attention
//! scale 1.0; V RMS-normed with no weight, and a global layer's V is its RAW K
//! projection, copied before the K norm; global RoPE divides by `rope_freqs`, baked
//! into its table; a per-layer `out_scale` on the whole layer output; the head's
//! logits softcapped and the suppress tokens masked, on the host.
//!
//! The local rings and the activation buffers start at what text needs and grow once,
//! on the first image, to the vision budget's block, as gemma4_cuda's; the RoPE tables
//! cover the committed capacity and grow with it. The split's host cache keeps the
//! device ring layout (`PerLayerKvCache`), so its copies are raw.

const std = @import("std");
const init_defaults = @import("tp_core").init_defaults;
const qwen3 = @import("qwen3.zig");
const gemma4 = @import("gemma4.zig");
const lin_gpu = @import("lin_gpu.zig");
const kv_gpu = @import("kv_gpu.zig");
const gpu = @import("tp_gpu").context;
const ops = @import("tp_ops");
const kvmod = @import("tp_core").kv_cache;
const sample = @import("tp_core").sample;
const bnd = @import("tp_runtime").boundary;
const residency = @import("tp_runtime").residency;
const transformer = @import("transformer.zig");
const transformer_gpu = @import("transformer_gpu.zig");
const build_options = @import("build_options");

const Buf = gpu.DeviceBuffer;

fn nbuf(ctx: *gpu.Context, w: []const f32) !Buf {
    return .{ .buf = try ctx.smallBuffer(std.mem.sliceAsBytes(w)), .mem = .null_handle, .size = 0 };
}

/// Rows per text prefill batch, as gemma4_cuda's.
const prefill_chunk = 256;
/// KV chunks per head in the attention split pass: a decode step, then a batch, whose
/// query rows already fill the device.
const nsplit = 32;
const nsplit_prefill = 8;
const gemv_nchunk = 16;

/// Activation rows for the largest batch, a text chunk or a whole image block, a
/// multiple of the coop GEMMs' 128-row tile.
fn bufRows(cfg: gemma4.Config) usize {
    return std.mem.alignForward(usize, cfg.maxBatch(), 128);
}

/// What text alone needs; `ensureBufRows` grows to `bufRows` on the first image.
const text_buf_rows = std.mem.alignForward(usize, prefill_chunk, 128);

/// A local ring holds its window plus a whole batch: a batch stores every row before
/// any of its queries reads, and the first query still needs the window behind it.
fn localRingText(cfg: gemma4.Config) usize {
    return cfg.sliding_window + prefill_chunk;
}
fn localRingMax(cfg: gemma4.Config) usize {
    return cfg.sliding_window + cfg.maxBatch();
}

pub const VulkanLM = struct {
    /// The prefill batch this stepper wants the engine to hand it.
    pub const prefill_batch = prefill_chunk;
    /// `tp-llm --bench` alternates the `OpSDot` q8 decode GEMVs (toggle on: off) with
    /// the f32 ones.
    pub const bench_toggle = &gpu.Context.sdot_off;

    be: *gpu.Context,
    lm: *const gemma4.Model,
    cfg: gemma4.Config,
    gpa: std.mem.Allocator,
    /// Rows the global layers' KV holds now; `ensureCapacity` grows it toward
    /// `max_capacity` and `resetResidency` shrinks it back.
    capacity: usize,
    initial_capacity: usize,
    max_capacity: usize,
    kv_dtype: kvmod.KvDtype = .f32,
    len: usize,
    kv: kv_gpu.Kv,
    /// Rows each local ring holds: `localRingText`, then `localRingMax` from the first
    /// image on (`growLocalRing`).
    ring_rows: usize = 0,
    /// Rows the activation buffers hold: `text_buf_rows`, then `bufRows` from the first
    /// image on (`ensureBufRows`).
    buf_rows: usize = 0,
    lin: lin_gpu.Lin,
    lin_plan: lin_gpu.Plan,
    /// Global (theta 1e6 over `rope_freqs`) and local (theta 1e4) RoPE tables, cos then
    /// sin, covering `rope_rows` positions; each sin half starts at its offset.
    freqs_global: Buf,
    freqs_local: Buf,
    rope_rows: usize = 0,
    sin_off_global: usize,
    sin_off_local: usize,
    /// A vector of ones, the weightless V norm's weight.
    ones: Buf,
    /// The suppress tokens as penalty entries: the device mask is the penalize scatter
    /// with an infinite presence penalty.
    suppress_pen: []sample.PenaltyEntry,
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

    /// The CPU split (`residency.zig` drives it), as gemma4_cuda's.
    pub const Split = struct {
        on_gpu: []bool,
        n_cpu: usize,
        policy: residency.CpuSplitPolicy,
        /// Host K/V for the host layers, in the device's byte and ring layout; its
        /// length tracks `len`.
        cache: kvmod.PerLayerKvCache,
        scratch: gemma4.Scratch,
        /// 0: global (with `rope_freqs`), 1: local.
        rope: ops.rope.RopeTables(2),
        /// The hidden while host layers run, a whole batch of rows.
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

    pub fn init(gpa: std.mem.Allocator, ctx: *gpu.Context, lm: *const gemma4.Model, cap: kvmod.Capacity) !VulkanLM {
        const cfg = lm.cfg;
        var self: VulkanLM = init_defaults.of(VulkanLM);
        self.be = ctx;
        self.lm = lm;
        self.cfg = cfg;
        self.gpa = gpa;
        self.capacity = @max(@min(cap.initial, cap.max), 1);
        self.initial_capacity = self.capacity;
        self.max_capacity = cap.max;
        self.kv_dtype = cap.kv_dtype;
        self.len = 0;
        self.bidir_end = 0;
        self.residual_pending = false;
        self.boundary = null;
        self.io = null;
        self.split = null;
        ctx.weight_noise.setDepth(cfg.n_layers);

        const max_out = @max(cfg.vocab, @max(cfg.intermediate, cfg.maxQDim()));
        self.lin = try lin_gpu.Lin.init(gpa, ctx, max_out, gemv_nchunk, 3);
        errdefer self.lin.deinit();
        {
            var arena = std.heap.ArenaAllocator.init(gpa);
            defer arena.deinit();
            const lins = try lm.deviceLins(arena.allocator());
            self.lin_plan = try self.lin.plan(lins, lin_gpu.Use.llm(bufRows(cfg)), "gemma4 vulkan");
            // Not for the head: a prefill reads no logits and a step runs it as a GEMV,
            // while its expansion would be vocab x hidden of scratch.
            var batched: std.ArrayList(lin_gpu.Weight) = .empty;
            for (lins) |w| if (w.bytes.ptr != lm.head.bytes.ptr) try batched.append(arena.allocator(), w);
            try lin_gpu.presize(ctx, self.lin_plan, batched.items);
        }

        self.rope_rows = 0;
        self.freqs_global = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 };
        self.freqs_local = self.freqs_global;
        try self.growRope(self.capacity);
        errdefer {
            ctx.tensorDestroy(&self.freqs_global);
            ctx.tensorDestroy(&self.freqs_local);
        }

        const ones = try gpa.alloc(f32, @max(cfg.head_dim_global, cfg.head_dim_local));
        defer gpa.free(ones);
        @memset(ones, 1.0);
        self.ones = try ctx.tensorCreate(ones.len * 4);
        errdefer ctx.tensorDestroy(&self.ones);
        try ctx.tensorUpload(self.ones, std.mem.sliceAsBytes(ones));

        self.suppress_pen = try gpa.alloc(sample.PenaltyEntry, lm.suppress_tokens.len);
        errdefer gpa.free(self.suppress_pen);
        for (self.suppress_pen, lm.suppress_tokens) |*e, id| e.* = .{ .id = id, .count = 1 };

        self.ring_rows = localRingText(cfg);
        self.kv = try self.newKv(self.capacity, cap.kv_dtype);
        errdefer self.kv.deinit(gpa, ctx);
        self.buf_rows = text_buf_rows;
        self.bufs = try Bufs.init(ctx, cfg, self.buf_rows);
        return self;
    }

    /// A cache with every local layer a ring of `ring_rows` and the global layers at
    /// `capacity` rows, each at its own width.
    fn newKv(self: *const VulkanLM, capacity: usize, dtype: kvmod.KvDtype) !kv_gpu.Kv {
        const cfg = self.cfg;
        const dims = try cfg.kvDims(self.gpa);
        defer self.gpa.free(dims);
        const rings = try self.gpa.alloc(usize, cfg.n_layers);
        defer self.gpa.free(rings);
        for (rings, 0..) |*r, l| r.* = if (cfg.isGlobal(l)) 0 else self.ring_rows;
        return kv_gpu.Kv.initDims(self.gpa, self.be, capacity, dims, dtype, rings);
    }

    /// The split's host cache in the device layout: local layers rings of `ring`.
    fn hostShadowCacheAt(self: *const VulkanLM, ring: usize, dtype: kvmod.KvDtype) !kvmod.PerLayerKvCache {
        const cfg = self.cfg;
        const dims = try cfg.kvDims(self.gpa);
        defer self.gpa.free(dims);
        const rings = try self.gpa.alloc(usize, cfg.n_layers);
        defer self.gpa.free(rings);
        for (rings, 0..) |*r, l| r.* = if (cfg.isGlobal(l)) 0 else ring;
        return kvmod.PerLayerKvCache.init(self.gpa, self.capacity, dims, rings, dtype);
    }

    fn uploadFreqs(ctx: *gpu.Context, gpa: std.mem.Allocator, rows: usize, head_dim: usize, theta: f64, factors: ?[]const f32) !Buf {
        const half = head_dim / 2;
        var freqs = try ops.rope.rotateHalfFreqsFactored(gpa, rows, head_dim, theta, 1.0, factors);
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

    /// Both RoPE tables to cover `rows` positions. A table and its sin offset move
    /// together, or every rotation past the old count reads cos as sin.
    fn growRope(self: *VulkanLM, rows: usize) !void {
        if (rows <= self.rope_rows) return;
        const cfg = self.cfg;
        var g = try uploadFreqs(self.be, self.gpa, rows, cfg.head_dim_global, cfg.rope_theta, self.lm.rope_freqs);
        errdefer self.be.tensorDestroy(&g);
        const l = try uploadFreqs(self.be, self.gpa, rows, cfg.head_dim_local, cfg.rope_theta_local, null);
        self.be.tensorDestroy(&self.freqs_global);
        self.be.tensorDestroy(&self.freqs_local);
        self.freqs_global = g;
        self.freqs_local = l;
        self.rope_rows = rows;
        self.sin_off_global = rows * (cfg.head_dim_global / 2);
        self.sin_off_local = rows * (cfg.head_dim_local / 2);
    }

    pub fn deinit(self: *VulkanLM) void {
        const ctx = self.be;
        self.freeSplit();
        self.kv.deinit(self.gpa, ctx);
        ctx.tensorDestroy(&self.freqs_global);
        ctx.tensorDestroy(&self.freqs_local);
        ctx.tensorDestroy(&self.ones);
        self.gpa.free(self.suppress_pen);
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
    /// Layers on the host now, 0 when every layer is on the device.
    pub fn hostLayers(self: *const VulkanLM) usize {
        return if (self.split) |sp| sp.n_cpu else 0;
    }

    /// Device bytes growing the global layers still on the device to `target` rows
    /// commits; the rings are fixed.
    fn growBytes(self: *const VulkanLM, target: usize) u64 {
        const add = target - self.capacity;
        var need: u64 = 0;
        for (0..self.cfg.n_layers) |l| {
            if (!self.cfg.isGlobal(l)) continue;
            if (self.split) |sp| if (!sp.on_gpu[l]) continue;
            need += 2 * self.kv_dtype.sizeBytes(add * self.cfg.kvDim(l));
        }
        return need;
    }

    /// Grow the global layers' KV to hold `min_rows`, by the CUDA steppers' growth
    /// policy. A dynamic split first moves layers to the host until it can, and a real
    /// out-of-memory moves one more and retries.
    pub fn ensureCapacity(self: *VulkanLM, min_rows: usize) !void {
        const target = (try kvmod.growPlan(self.capacity, self.max_capacity, min_rows)) orelse return;
        if (self.split) |*sp| if (sp.dynamic) {
            while (true) {
                const need = self.growBytes(target) + residency.promoteReserve(self);
                const free = @min(sp.budget -| self.be.deviceUsed(), self.be.headroom());
                if (free >= need) break;
                if (!(try residency.migrateNext(self))) break;
            }
        };
        while (true) {
            self.kv.grow(self.be, target, self.len) catch |err| switch (err) {
                error.DeviceOutOfMemory, error.OutOfMemory => {
                    self.ensureOffloadArmed() catch return error.ContextFull;
                    if (try residency.migrateNext(self)) continue;
                    return error.ContextFull;
                },
                else => return err,
            };
            break;
        }
        try self.growRope(target);
        if (self.split) |*sp| {
            sp.cache.grow(self.gpa, target) catch return error.ContextFull;
            sp.rope.regrow(self.gpa, target) catch return error.ContextFull;
        }
        self.capacity = target;
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
        }
        const saved_cap = self.capacity;
        self.capacity = self.initial_capacity;
        if (self.split) |*sp| {
            const cache = self.hostShadowCacheAt(self.ring_rows, dtype) catch |err| {
                self.capacity = saved_cap;
                return err;
            };
            sp.cache.deinit(self.gpa);
            sp.cache = cache;
        }
        self.kv.deinit(self.gpa, self.be);
        self.kv = kv;
        self.kv_dtype = dtype;
        try self.resetCache();
    }

    /// A turn checkpoint's fixed size: every local layer's whole ring, K then V. The
    /// global layers are append-only and are truncated instead; a ring overwrites the
    /// rows a rollback needs back.
    pub fn checkpointBytes(self: *const VulkanLM) usize {
        var total: usize = 0;
        for (0..self.cfg.n_layers) |l| {
            if (!self.cfg.isGlobal(l)) total += 2 * self.kv_dtype.sizeBytes(self.ring_rows * self.cfg.kvDim(l));
        }
        return total;
    }

    /// Snapshot the rings at the current position, each from whichever side holds its
    /// layer (both in ring layout), so a layer may move between snapshot and restore.
    pub fn checkpoint(self: *VulkanLM, out: []u8) !void {
        std.debug.assert(out.len == self.checkpointBytes());
        try self.checkpointRings(.save, out);
    }

    /// Roll back to `q` committed tokens with a snapshot taken there.
    pub fn restoreCheckpoint(self: *VulkanLM, snap: []const u8, q: usize) !void {
        std.debug.assert(snap.len == self.checkpointBytes() and q <= self.len);
        self.len = q;
        if (self.split) |*sp| sp.cache.truncate(q);
        try self.checkpointRings(.restore, snap);
    }

    const CheckpointDir = enum { save, restore };

    fn checkpointRings(self: *VulkanLM, comptime dir: CheckpointDir, buf: if (dir == .save) []u8 else []const u8) !void {
        const cfg = self.cfg;
        var off: usize = 0;
        for (0..cfg.n_layers) |l| {
            if (cfg.isGlobal(l)) continue;
            const rb = self.kv_dtype.sizeBytes(self.ring_rows * cfg.kvDim(l));
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
            const hk = sp.cache.kRowBytes(l, 0, self.ring_rows);
            const hv = sp.cache.vRowBytes(l, 0, self.ring_rows);
            if (comptime dir == .save) {
                @memcpy(bk, hk);
                @memcpy(bv, hv);
            } else {
                @memcpy(hk, bk);
                @memcpy(hv, bv);
            }
        }
    }

    /// Where a forward leaves the last row's logits: nowhere (a prefill), a host
    /// buffer (downloaded and finalized), or raw on the device for the sampling ops.
    const LogitsOut = union(enum) { none, host: []f32, device };

    /// Forward `ids` in chunks, then write the last position's finalized logits.
    pub fn step(self: *VulkanLM, io: std.Io, ids: []const u32, logits: []f32) !void {
        self.io = io;
        std.debug.assert(ids.len >= 1 and logits.len == self.cfg.vocab);
        try self.prefillIds(ids, .{ .host = logits });
    }

    /// Mask the suppress tokens on the device logits: an infinite presence penalty
    /// with a unit repeat penalty leaves every other logit exact.
    fn suppressLogits(self: *VulkanLM) !void {
        const sp: sample.Params = .{ .repeat_penalty = 1.0, .presence_penalty = std.math.inf(f32) };
        var off: usize = 0;
        while (off < self.suppress_pen.len) {
            const n: usize = @min(sample.max_penalty_window, self.suppress_pen.len - off);
            try self.be.opPenalize(self.bufs.logits, self.suppress_pen[off..][0..n], sp);
            off += n;
        }
    }

    /// Greedy decode without the vocab download. The softcap is monotonic, so the
    /// argmax over the raw masked logits is the argmax over the finalized ones.
    pub fn stepArgmax(self: *VulkanLM, io: std.Io, ids: []const u32) !u32 {
        return self.stepArgmaxPen(io, ids, &.{}, .{});
    }

    /// `stepArgmax` with the sampling penalties. Those are not monotonic over the capped
    /// logits, so with any the winner comes from the finalized candidates instead.
    pub fn stepArgmaxPen(self: *VulkanLM, io: std.Io, ids: []const u32, pen: []const sample.PenaltyEntry, sp: sample.Params) !u32 {
        if (pen.len != 0) {
            var out_id: [gpu.topk_lanes * gpu.topk_m]u32 = undefined;
            var out_logit: [gpu.topk_lanes * gpu.topk_m]f32 = undefined;
            const count = try self.stepSelectPen(io, ids, pen, sp, &out_id, &out_logit);
            var best: usize = 0;
            for (1..count) |i| {
                if (out_logit[i] > out_logit[best] or (out_logit[i] == out_logit[best] and out_id[i] < out_id[best])) best = i;
            }
            return out_id[best];
        }
        self.io = io;
        try self.prefillIds(ids, .device);
        const b = &self.bufs;
        try self.suppressLogits();
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

    /// Stochastic decode: the masked top-k over the raw logits on the device, finalized
    /// on the host (`gemma4.finalizeCandidates`: softcap, mask, penalties).
    pub fn stepSelect(self: *VulkanLM, io: std.Io, ids: []const u32, out_id: []u32, out_logit: []f32) !usize {
        return self.stepSelectPen(io, ids, &.{}, .{}, out_id, out_logit);
    }

    pub fn stepSelectPen(self: *VulkanLM, io: std.Io, ids: []const u32, pen: []const sample.PenaltyEntry, sp: sample.Params, out_id: []u32, out_logit: []f32) !usize {
        self.io = io;
        try self.prefillIds(ids, .device);
        const b = &self.bufs;
        try self.suppressLogits();
        const count = try self.be.opTopK(b.logits, self.cfg.vocab, &b.topk_v, &b.topk_i);
        std.debug.assert(count <= out_id.len and count <= out_logit.len);
        try self.be.tensorDownload(b.topk_v, std.mem.sliceAsBytes(out_logit[0..count]));
        var idx_f: [gpu.topk_lanes * gpu.topk_m]f32 = undefined;
        try self.be.tensorDownload(b.topk_i, std.mem.sliceAsBytes(idx_f[0..count]));
        for (out_id[0..count], idx_f[0..count]) |*o, f| o.* = @intFromFloat(f);
        self.lm.finalizeCandidates(out_id[0..count], out_logit[0..count], pen, sp);
        return count;
    }

    /// Prefill text tokens (no logits), for interleaving with prefillImage.
    pub fn prefill(self: *VulkanLM, ids: []const u32) !void {
        try self.prefillIds(ids, .none);
    }

    /// Prefill one image's projected embeddings ([n][hidden], unscaled) at the next
    /// positions, every query seeing the whole block (llama.cpp marks image spans
    /// non-causal): one batch, so the block's K/V is stored before any of it is read.
    /// The first image grows the rings and the buffers to the vision budget's block.
    pub fn prefillImage(self: *VulkanLM, embeds: []const f32, grid_w: usize, grid_h: usize) !void {
        _ = grid_w;
        _ = grid_h;
        const n = embeds.len / self.cfg.hidden;
        std.debug.assert(n >= 1 and n <= self.cfg.maxBatch() and n <= self.remaining());
        try bnd.check(self);
        try self.growLocalRing(localRingMax(self.cfg));
        try self.ensureBufRows(n);
        self.bidir_end = self.len + n;
        defer self.bidir_end = 0;
        try self.forwardRows(embeds, .none);
    }

    /// Every local ring to `new_ring` rows with its live window, on whichever side
    /// holds the layer. A ring addresses position p at row p % ring, so this is a
    /// remap, not a resize. Once per session.
    fn growLocalRing(self: *VulkanLM, new_ring: usize) !void {
        if (new_ring <= self.ring_rows) return;
        const cfg = self.cfg;
        const old_ring = self.ring_rows;
        var new_cache: ?kvmod.PerLayerKvCache = null;
        errdefer if (new_cache) |*c| c.deinit(self.gpa);
        if (self.split) |*sp| {
            new_cache = try self.hostShadowCacheAt(new_ring, self.kv_dtype);
            new_cache.?.len = self.len;
            for (0..cfg.n_layers) |l| {
                if (sp.on_gpu[l]) continue;
                if (cfg.isGlobal(l)) {
                    if (self.len == 0) continue;
                    @memcpy(new_cache.?.kRowBytes(l, 0, self.len), sp.cache.kRowBytes(l, 0, self.len));
                    @memcpy(new_cache.?.vRowBytes(l, 0, self.len), sp.cache.vRowBytes(l, 0, self.len));
                } else {
                    remapHostRing(&sp.cache, &new_cache.?, l, self.len, old_ring, new_ring);
                }
            }
        }
        for (0..cfg.n_layers) |l| {
            if (cfg.isGlobal(l)) continue;
            try self.kv.regrowRing(self.be, self.gpa, l, new_ring, self.len);
        }
        if (new_cache) |c| {
            const sp = &self.split.?;
            sp.cache.deinit(self.gpa);
            sp.cache = c;
            new_cache = null;
        }
        self.ring_rows = new_ring;
        std.log.info("[vram] local K/V rings grown {d} -> {d} rows for an image block", .{ old_ring, new_ring });
    }

    /// One host ring layer's live window from `from` (ring `old`) into `to` (ring
    /// `new`), position by position.
    fn remapHostRing(from: *kvmod.PerLayerKvCache, to: *kvmod.PerLayerKvCache, l: usize, len: usize, old: usize, new: usize) void {
        for (len - @min(len, old)..len) |p| {
            @memcpy(to.kRowBytes(l, p % new, 1), from.kRowBytes(l, p % old, 1));
            @memcpy(to.vRowBytes(l, p % new, 1), from.vRowBytes(l, p % old, 1));
        }
    }

    /// The activation buffers to hold `rows` in one batch. Between forwards only: they
    /// are per-forward scratch, nothing carries across a call.
    fn ensureBufRows(self: *VulkanLM, rows: usize) !void {
        const want = std.mem.alignForward(usize, rows, 128);
        if (want <= self.buf_rows) return;
        std.debug.assert(!self.be.batching);
        var bufs = try Bufs.init(self.be, self.cfg, want);
        errdefer bufs.deinit(self.be);
        self.bufs.deinit(self.be);
        self.bufs = bufs;
        self.buf_rows = want;
        if (self.split) |*sp| if (sp.hx.len < want * self.cfg.hidden) {
            self.gpa.free(sp.hx);
            sp.hx = try self.gpa.alloc(f32, want * self.cfg.hidden);
        };
        std.log.info("[vram] activation buffers grown to {d} rows for a {d}-token image block", .{ want, rows });
    }

    /// Upload every weight now rather than inside the first turn, by running a
    /// prefill-sized and a decode forward: those are the routes, and so the device
    /// layouts, real turns read. The context is emptied after.
    pub fn warmWeights(self: *VulkanLM) void {
        if (self.len != 0) return;
        const zeros = [_]u32{0} ** 32;
        const n = @min(zeros.len, self.capacity - 1);
        self.prefillIds(zeros[0..n], .none) catch |err| std.log.warn("gemma4 vulkan warm-up: {t}", .{err});
        self.prefillIds(zeros[0..1], .device) catch |err| std.log.warn("gemma4 vulkan warm-up: {t}", .{err});
        self.resetCache() catch |err| std.log.warn("gemma4 vulkan warm-up: {t}", .{err});
    }

    fn prefillIds(self: *VulkanLM, ids: []const u32, out: LogitsOut) !void {
        const cfg = self.cfg;
        std.debug.assert(ids.len <= self.remaining());
        const x = try self.gpa.alloc(f32, @min(ids.len, prefill_chunk) * cfg.hidden);
        defer self.gpa.free(x);
        const scale = cfg.embedScale();
        var off: usize = 0;
        while (off < ids.len) {
            if (out == .none) try bnd.check(self);
            const n: usize = @min(prefill_chunk, ids.len - off);
            const xs = x[0 .. n * cfg.hidden];
            try qwen3.embedTokens(self.lm.embed, ids[off..][0..n], xs);
            for (xs) |*v| v.* *= scale;
            try self.forwardRows(xs, if (off + n == ids.len) out else .none);
            off += n;
        }
    }

    /// One forward, offloading layers to the host and retrying when the device runs out
    /// of memory: a failed forward commits nothing (`len` unchanged), so moving a few
    /// layers and running it again is exact. The batch's weight perturbation is drawn
    /// once, outside the retry.
    fn forwardRows(self: *VulkanLM, x: []const f32, out: LogitsOut) !void {
        self.be.weight_noise.tick();
        while (true) {
            return self.forwardRowsOnce(x, out) catch |err| switch (err) {
                error.DeviceOutOfMemory, error.OutOfMemory => {
                    try self.ensureOffloadArmed();
                    var moved: usize = 0;
                    while (moved < 4) : (moved += 1) {
                        if (!(try residency.migrateNext(self))) break;
                    }
                    if (moved == 0) return err;
                    std.log.info("[vram] forward OOM at ctx {d}: offloaded {d} layer(s) to CPU, retrying", .{ self.len, moved });
                    continue;
                },
                else => return err,
            };
        }
    }

    /// One batch over the embedded rows `x` at positions [len, len+n), the last row's
    /// logits where `out` says. Polls `ops.cancel` between layers.
    fn forwardRowsOnce(self: *VulkanLM, x: []const f32, out: LogitsOut) !void {
        const ctx = self.be;
        const cfg = self.cfg;
        const b = &self.bufs;
        const n = x.len / cfg.hidden;
        std.debug.assert(n >= 1 and n <= self.buf_rows and n <= self.remaining());
        const pos0 = self.len;
        // Before the batch opens: an upload inside it would flush the recording.
        try ctx.tensorUpload(try ctx.view(b.x, 0, x.len * 4), std.mem.sliceAsBytes(x));
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
                    try gemma4.layerForward(host_io, self.gpa, cfg, layer, sp.hx[0 .. n * cfg.hidden], n, sp.rope.get(0), sp.rope.get(1), &sp.cache, l, self.bidir_end != 0, &sp.scratch);
                    continue;
                }
                if (sp.on_host) try self.hiddenToDevice(n);
            }
            try transformer_gpu.decoderLayer(transformer.gemma4_spec, self, layer, l, n, pos0);
        }
        if (self.split) |*sp| if (sp.on_host) try self.hiddenToDevice(n);
        if (out != .none) {
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
        self.len += n;
        if (self.split) |*sp| sp.cache.commit(n);
        if (out == .host) {
            try ctx.tensorDownload(b.logits, std.mem.sliceAsBytes(out.host));
            self.lm.finalizeLogits(out.host);
        }
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

    /// `normed = rmsnorm(x) * w` over `rows` rows, adding the pending residual into x
    /// first.
    fn hiddenNorm(self: *VulkanLM, w: []const f32, rows: usize) !void {
        const b = &self.bufs;
        try self.be.opAddRmsNorm(b.x, if (self.residual_pending) b.t else null, try nbuf(self.be, w), b.normed, rows, self.cfg.hidden, self.cfg.rms_eps);
        self.residual_pending = false;
    }

    /// Which layer the coming launches belong to, for the weight-noise curve.
    pub fn noiseAtLayer(self: *VulkanLM, l: usize) void {
        self.be.weight_noise.atLayer(l);
    }

    // Residency: the hooks `residency.zig` drives, with gemma4_cuda's meaning.

    fn layerDeviceBytes(layer: anytype) usize {
        const v_bytes = if (layer.v) |w| w.bytes.len else 0;
        return layer.q.bytes.len + layer.k.bytes.len + v_bytes + layer.o.bytes.len +
            layer.gate.bytes.len + layer.up.bytes.len + layer.down.bytes.len;
    }

    pub fn layerWeightBytes(self: *VulkanLM, l: usize) usize {
        return layerDeviceBytes(&self.lm.layers[l]);
    }

    /// The tied head; the embeddings are looked up on the host.
    pub fn nonLayerDeviceBytes(self: *VulkanLM) u64 {
        return self.lm.head.bytes.len;
    }

    /// One layer's K and V: a ring's rows, a global layer's capacity.
    fn layerKvBytes(self: *const VulkanLM, l: usize) usize {
        const rows = if (self.cfg.isGlobal(l)) self.capacity else self.ring_rows;
        return 2 * self.kv_dtype.sizeBytes(rows * self.cfg.kvDim(l));
    }

    /// VRAM a promote of layer `l` needs: its weights, its K/V, and slack.
    pub fn promoteCost(self: *VulkanLM, l: usize) usize {
        return layerDeviceBytes(&self.lm.layers[l]) + self.layerKvBytes(l) + residency.promote_slack;
    }

    /// Move layer `l` to the host: its live K/V into the shadow, raw (same layout), its
    /// device K/V and weights freed.
    pub fn migrateLayer(self: *VulkanLM, l: usize) !void {
        const sp = &self.split.?;
        try self.kv.dropLayer(self.be, l, self.len, &sp.cache, l);
        const layer = &self.lm.layers[l];
        for ([_][]const u8{ layer.q.bytes, layer.k.bytes, layer.o.bytes, layer.gate.bytes, layer.up.bytes, layer.down.bytes }) |w|
            self.be.evictWeightBytes(w);
        if (layer.v) |w| self.be.evictWeightBytes(w.bytes);
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

    /// The split's host state for `n_cpu` planned layers, every layer still on the
    /// device.
    fn armSplit(self: *VulkanLM, policy: residency.CpuSplitPolicy, budget: u64, dynamic: bool, order: []usize, next: usize) !void {
        const cfg = self.cfg;
        const gpa = self.gpa;
        const on_gpu = try gpa.alloc(bool, cfg.n_layers);
        errdefer gpa.free(on_gpu);
        @memset(on_gpu, true);
        var cache = try self.hostShadowCacheAt(self.ring_rows, self.kv_dtype);
        errdefer cache.deinit(gpa);
        cache.len = self.len;
        var scratch = try gemma4.Scratch.init(gpa, cfg.maxBatch(), cfg);
        errdefer scratch.deinit(gpa);
        var rope = try ops.rope.RopeTables(2).init(gpa, .{
            .{ .head_dim = cfg.head_dim_global, .theta = cfg.rope_theta, .freq_factors = self.lm.rope_freqs },
            .{ .head_dim = cfg.head_dim_local, .theta = cfg.rope_theta_local },
        }, self.capacity);
        errdefer rope.deinit(gpa);
        const hx = try gpa.alloc(f32, @max(cfg.maxBatch(), self.buf_rows) * cfg.hidden);
        errdefer gpa.free(hx);
        self.split = .{
            .on_gpu = on_gpu,
            .n_cpu = 0,
            .policy = policy,
            .cache = cache,
            .scratch = scratch,
            .rope = rope,
            .hx = hx,
            .dynamic = dynamic,
            .order = order,
            .next = next,
            .budget = budget,
        };
    }

    /// A dynamic split with nothing moved yet, so a later out-of-memory can move layers
    /// one at a time (tp-llm arms none up front). No budget planner: run mid-life with
    /// the weights resident, it would move almost everything at once.
    fn ensureOffloadArmed(self: *VulkanLM) !void {
        if (self.split != null) return;
        const n = self.cfg.n_layers;
        const order = try self.gpa.alloc(usize, n);
        errdefer self.gpa.free(order);
        for (0..n) |i| order[i] = n - 1 - i;
        try self.armSplit(.attn, std.math.maxInt(u64), true, order, 0);
    }

    /// Place layers on the host until the device-resident weights fit under `budget`.
    /// The last layer leaves first; gemma4 is all attention, so `.attn` and `.tail`
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
        const n_cpu = residency.splitHostCount(self, budget, reserve, dynamic, per, order);
        if (n_cpu == 0 and !dynamic) {
            gpa.free(order);
            return;
        }
        try self.armSplit(policy, budget, dynamic, order, n_cpu);
        for (order[0..n_cpu]) |l| try self.migrateLayer(l);
    }

    fn freeSplit(self: *VulkanLM) void {
        const sp = &(self.split orelse return);
        sp.cache.deinit(self.gpa);
        sp.scratch.deinit(self.gpa);
        sp.rope.deinit(self.gpa);
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

    // `transformer_gpu.decoderLayer` stepper methods. Geometry is per layer.

    pub fn normInput(self: *VulkanLM, layer: anytype, seq: usize) !void {
        try self.hiddenNorm(layer.input_norm, seq);
    }

    /// q, k and v, a global layer's v being its raw k: copied before `normQK` rewrites k.
    pub fn projectQKV(self: *VulkanLM, l: usize, layer: anytype, seq: usize) !void {
        const b = &self.bufs;
        const kv = try self.kvOut(l, seq);
        if (layer.v) |vw| {
            if (seq == 1) return self.lin.linearGroup(self.lin_plan, &.{ b.q, kv[0], kv[1] }, b.normed, &.{ layer.q, layer.k, vw });
            const g = try self.lin.prep(self.lin_plan, b.normed, seq, self.cfg.hidden, &.{ layer.q, layer.k, vw });
            try self.lin.gemm(self.lin_plan, g, b.q, layer.q, .{});
            try self.lin.gemm(self.lin_plan, g, kv[0], layer.k, .{});
            try self.lin.gemm(self.lin_plan, g, kv[1], vw, .{});
            return;
        }
        if (seq == 1) {
            try self.lin.linearGroup(self.lin_plan, &.{ b.q, kv[0] }, b.normed, &.{ layer.q, layer.k });
        } else {
            const g = try self.lin.prep(self.lin_plan, b.normed, seq, self.cfg.hidden, &.{ layer.q, layer.k });
            try self.lin.gemm(self.lin_plan, g, b.q, layer.q, .{});
            try self.lin.gemm(self.lin_plan, g, kv[0], layer.k, .{});
        }
        const e = seq * self.cfg.kvDim(l);
        try self.be.opElt(.copy, kv[0], kv[1], null, null, .{ .u0 = @intCast(e), .u2 = 0, .u3 = 0 }, e, 1, 1);
    }

    /// Where layer `l` projects its k and v: straight into the cache rows when it can
    /// (`Kv.rowsAt`, a single row the GEMVs write exactly), else the scratch `appendKV`
    /// stores from.
    fn kvOut(self: *VulkanLM, l: usize, seq: usize) ![2]Buf {
        const layer = self.lm.layers[l];
        const direct = seq == 1 and self.lin.writesRows(self.lin_plan, layer.k, 1) and
            (if (layer.v) |vw| self.lin.writesRows(self.lin_plan, vw, 1) else true);
        if (direct) if (try self.kv.rowsAt(self.be, l, self.len, 1)) |rows| return rows;
        return .{ self.bufs.k, self.bufs.v };
    }

    /// The rope too, in the same pass: the position is the cache length, which
    /// `forwardRowsOnce` passes as pos0.
    pub fn normQK(self: *VulkanLM, l: usize, layer: anytype, seq: usize) !void {
        const cfg = self.cfg;
        const ctx = self.be;
        const global = cfg.isGlobal(l);
        const freqs = if (global) self.freqs_global else self.freqs_local;
        const sin_off = if (global) self.sin_off_global else self.sin_off_local;
        const hd = cfg.headDim(l);
        if (seq == 1) ctx.independent(2);
        try ctx.opQkNormRope(self.bufs.q, try nbuf(ctx, layer.q_norm), freqs, seq * cfg.n_heads, hd, cfg.n_heads, self.len, sin_off, cfg.rms_eps);
        try ctx.opQkNormRope((try self.kvOut(l, seq))[0], try nbuf(ctx, layer.k_norm), freqs, seq * cfg.nKv(l), hd, cfg.nKv(l), self.len, sin_off, cfg.rms_eps);
    }

    /// RMS over each head with no learned weight. V is not rotated.
    pub fn normV(self: *VulkanLM, l: usize, seq: usize) !void {
        const cfg = self.cfg;
        const v = (try self.kvOut(l, seq))[1];
        const rows = seq * cfg.nKv(l);
        try self.be.opElt(.rmsnorm, v, v, self.ones, null, .{ .u0 = @intCast(rows), .u1 = @intCast(cfg.headDim(l)), .f0 = cfg.rms_eps }, rows, 1, 1);
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
        const window: usize = if (cfg.isGlobal(l)) 0 else cfg.sliding_window;
        const ns: usize = if (seq == 1) nsplit else nsplit_prefill;
        try self.kv.attend(self.be, l, b.q, b.attn, b.attn_scratch, pos0 + 1, seq, cfg.n_heads, cfg.nKv(l), cfg.headDim(l), ns, transformer.gemma4_spec.attn_scale.?, window, self.bidir_end);
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

    /// The whole layer output, residual included, times the layer's scalar.
    pub fn outScale(self: *VulkanLM, layer: anytype, seq: usize) !void {
        if (layer.out_scale == 1.0) return;
        try self.flushResidual(seq);
        const n = seq * self.cfg.hidden;
        try self.be.opElt(.scale_f32, self.bufs.x, self.bufs.x, null, null, .{ .u0 = @intCast(n), .f0 = layer.out_scale }, n, 1, 1);
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

    /// Every activation holds `rows`, at the widest layer's widths.
    fn init(ctx: *gpu.Context, cfg: gemma4.Config, rows: usize) !Bufs {
        const r = rows;
        const hd = @max(cfg.head_dim_global, cfg.head_dim_local);
        const sizes = [_]usize{
            r * cfg.hidden,
            r * cfg.hidden,
            r * cfg.maxQDim(),
            r * cfg.maxKvDim(),
            r * cfg.maxKvDim(),
            r * cfg.maxQDim(),
            @max(nsplit, r * nsplit_prefill) * cfg.n_heads * (hd + 2),
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

const test_path = build_options.model_root ++ "/../llm/gemma-4-12b-it-qat-q4_0.gguf";

/// A device and the test checkpoint, or a skip. Heap-held: the model borrows the
/// container's mapping.
const Fixture = struct {
    ctx: *gpu.Context,
    g: @import("tp_core").gguf.Gguf,
    lm: gemma4.Model,

    fn open(gpa: std.mem.Allocator, io: std.Io) !*Fixture {
        std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
        std.Io.Dir.cwd().access(io, test_path, .{}) catch return error.SkipZigTest;
        const f = try gpa.create(Fixture);
        errdefer gpa.destroy(f);
        f.ctx = gpu.Context.init(gpa, io) catch return error.SkipZigTest;
        errdefer f.ctx.deinit();
        f.g = try @TypeOf(f.g).open(gpa, io, test_path);
        errdefer f.g.deinit();
        f.lm = try gemma4.Model.load(gpa, &f.g);
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

/// Relative L2 over the finite entries; the suppressed ids are -inf on both sides.
fn relL2(want: []const f32, got: []const f32) f64 {
    var num: f64 = 0;
    var den: f64 = 0;
    for (want, got) |a, b| {
        if (!std.math.isFinite(a) or !std.math.isFinite(b)) continue;
        num += (@as(f64, a) - b) * (@as(f64, a) - b);
        den += @as(f64, a) * a;
    }
    return @sqrt(num / den);
}

// The device forward against the CPU one on the real model, a prefill and a decode
// step: every gemma4 departure from gemma3 (the per-layer geometry, the raw K as V, the
// weightless V norm, the factored global RoPE, out_scale, the softcap) moves these
// logits far past the tolerance when wrong. That is on exact routes; the q8 GEMVs
// quantize every activation, and a single row then lands anywhere in the format's noise
// (0.09 to 0.31 across prompt lengths, where the CPU's own q8 path lands 0.09 to 0.15).
test "vulkan gemma4 tracks the CPU forward" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const f = try Fixture.open(gpa, io);
    defer f.close(gpa);
    const prev = ops.matmul.exact_activations;
    ops.matmul.exact_activations = true;
    defer ops.matmul.exact_activations = prev;
    const prompt = try testPrompt(gpa, 24);
    defer gpa.free(prompt);
    const vocab = f.lm.cfg.vocab;
    var bufs: [4][]f32 = undefined;
    for (&bufs) |*b| b.* = try gpa.alloc(f32, vocab);
    defer for (bufs) |b| gpa.free(b);

    var cpu = try gemma4.CpuModel.init(gpa, &f.lm, .{ .initial = 64, .max = 64 });
    defer cpu.deinit();
    try cpu.step(io, prompt[0 .. prompt.len - 1], bufs[0]);
    try cpu.step(io, prompt[prompt.len - 1 ..], bufs[1]);
    var m = try VulkanLM.init(gpa, f.ctx, &f.lm, .{ .initial = 64, .max = 64 });
    defer m.deinit();
    const sdot_prev = gpu.Context.sdot_off;
    defer gpu.Context.sdot_off = sdot_prev;
    for ([2]bool{ true, false }) |exact| {
        gpu.Context.sdot_off = exact;
        try m.resetCache();
        try m.step(io, prompt[0 .. prompt.len - 1], bufs[2]);
        try m.step(io, prompt[prompt.len - 1 ..], bufs[3]);
        for ([2]usize{ 0, 1 }) |i| {
            const rel = relL2(bufs[i], bufs[i + 2]);
            errdefer std.debug.print("{s}, {s}: vulkan vs cpu logits rel L2 {e:.3}\n", .{ if (exact) "exact" else "q8", if (i == 0) "prefill" else "decode", rel });
            // Exact, the two agree to ~4e-5; dropping the global RoPE factors alone moves
            // this to 8e-2.
            const bound: f32 = if (exact) 1e-3 else 0.5;
            try std.testing.expect(rel < bound);
            if (exact) try std.testing.expectEqual(sample.argmax(bufs[i]), sample.argmax(bufs[i + 2]));
        }
    }
}

// Past the sliding window, where the rings wrap and the local mask bites, against
// CUDA's forward when a CUDA device is present.
test "vulkan gemma4 tracks CUDA past the window" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const f = try Fixture.open(gpa, io);
    defer f.close(gpa);
    const cuda = @import("tp_gpu").cuda;
    const be = cuda.Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();
    const n = localRingText(f.lm.cfg) + 100;
    const prompt = try testPrompt(gpa, n);
    defer gpa.free(prompt);
    const vocab = f.lm.cfg.vocab;
    const want = try gpa.alloc(f32, vocab);
    defer gpa.free(want);
    const got = try gpa.alloc(f32, vocab);
    defer gpa.free(got);
    {
        var c = try @import("gemma4_cuda.zig").CudaLM.init(gpa, be, &f.lm, .{ .initial = n, .max = n });
        defer c.deinit();
        try c.step(io, prompt, want);
    }
    var m = try VulkanLM.init(gpa, f.ctx, &f.lm, .{ .initial = n, .max = n });
    defer m.deinit();
    try m.step(io, prompt, got);
    const rel = relL2(want, got);
    errdefer std.debug.print("vulkan vs cuda logits at {d} tokens rel L2 {e:.3}\n", .{ n, rel });
    // The backends' own disagreement here is ~7e-3.
    try std.testing.expect(rel < 2e-2);
    try std.testing.expectEqual(sample.argmax(want), sample.argmax(got));
}

// Rollback past the ring slack: the prompt wraps every ring and the reply overwrites
// live-window rows, so only the checkpoint brings them back and the regenerated reply
// must be token-identical. Then a mid-conversation split must restore into the host
// layers (a checkpoint right after reads the same bytes, the logits repeat), and after
// every layer comes back the reply equals the unsplit one. The split legs test the
// restore, which precedes generation, so they generate only a short prefix.
fn checkpointRingBody(kv_dtype: kvmod.KvDtype) !void {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const f = try Fixture.open(gpa, io);
    defer f.close(gpa);
    const ring = localRingText(f.lm.cfg);
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
    const snap2 = try gpa.alloc(u8, snap.len);
    defer gpa.free(snap2);
    try m.checkpoint(snap);
    var seqs: [5][]u32 = undefined;
    for (&seqs, [5]usize{ n_gen, n_gen, 16, 16, 16 }) |*sq, len| sq.* = try gpa.alloc(u32, len);
    defer for (seqs) |sq| gpa.free(sq);
    const logits = try gpa.alloc(f32, f.lm.cfg.vocab);
    defer gpa.free(logits);
    const logits2 = try gpa.alloc(f32, f.lm.cfg.vocab);
    defer gpa.free(logits2);

    try greedy(&m, io, last, seqs[0]);
    try m.restoreCheckpoint(snap, q);
    try m.checkpoint(snap2);
    try std.testing.expectEqualSlices(u8, snap, snap2);
    try greedy(&m, io, last, seqs[1]);
    try std.testing.expectEqualSlices(u32, seqs[0], seqs[1]);

    try m.enableCpuSplit(.tail, std.math.maxInt(u64), true);
    m.io = io;
    try m.offloadToBudget(f.ctx.deviceUsed() * 4 / 5);
    try std.testing.expect(m.split.?.n_cpu > 0);
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

test "vulkan gemma4 checkpoint restores the rings past their slack, split and promoted" {
    try checkpointRingBody(.f32);
}

test "vulkan gemma4 ring checkpoint (f16 kv)" {
    try checkpointRingBody(.f16);
}

test "vulkan gemma4 ring checkpoint (q8_0 kv)" {
    try checkpointRingBody(.q8_0);
}

// The GUI turn shape: warm-up, a reset and replay, the device sampler against the
// finalized logits of the same step, and a KV dtype rebuild that must match a stepper
// built at that dtype.
test "vulkan gemma4 gui turn lifecycle: reset, sampling, kv rebuild" {
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
    for (f.lm.suppress_tokens) |id| try std.testing.expect(logits[id] == -std.math.inf(f32));
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
// built at full size does: growth copies rows and rebuilds the RoPE tables, it changes
// no arithmetic.
test "vulkan gemma4 kv growth decodes what a full-size cache does" {
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
test "vulkan gemma4 cpu split tracks the device and moves wrapped rings losslessly" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const f = try Fixture.open(gpa, io);
    defer f.close(gpa);
    const k = localRingText(f.lm.cfg) + 40;
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
        // ~6 GB of q4_0 layer weights: 4 GiB puts the tail layers on the host.
        try m.enableCpuSplit(.attn, 4 << 30, true);
        try std.testing.expect(m.split.?.n_cpu > 0 and m.split.?.n_cpu < m.cfg.n_layers);
        try std.testing.expectError(error.SplitIoUnset, m.prefill(prompt[k .. n - 1]));
        m.io = io;
        try m.prefill(prompt[k .. n - 1]);
        try m.step(io, prompt[n - 1 ..], got);
        const rel = relL2(want, got);
        errdefer std.debug.print("kv {t}: split vs device logits rel L2 {e:.3}\n", .{ dt, rel });
        try std.testing.expect(rel < 5e-2);
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
test "vulkan gemma4 cpu split survives a kv dtype toggle, and a residency reset re-arms it" {
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
test "vulkan gemma4 weight noise perturbs prefill and decode, reproducibly, and off is exact" {
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

// The first image grows the rings, remapping every live row, and the buffers: after a
// prompt that wrapped the text ring, an image block and a step must give exactly the
// logits of a stepper sized for images from the start, with every layer on the device
// and again with host layers, whose ring remap is separate code.
test "vulkan gemma4 image block grows the rings and buffers without moving a logit" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const f = try Fixture.open(gpa, io);
    defer f.close(gpa);
    const cfg = f.lm.cfg;
    const k = localRingText(cfg) + 40;
    const n_img = cfg.maxBatch();
    const total = k + n_img + 8;
    const prompt = try testPrompt(gpa, k + 1);
    defer gpa.free(prompt);
    const img_ids = try testPrompt(gpa, n_img);
    defer gpa.free(img_ids);
    const rows = try gpa.alloc(f32, n_img * cfg.hidden);
    defer gpa.free(rows);
    try qwen3.embedTokens(f.lm.embed, img_ids, rows);
    const vocab = cfg.vocab;
    var outs: [2][]f32 = undefined;
    for (&outs) |*o| o.* = try gpa.alloc(f32, vocab);
    defer for (outs) |o| gpa.free(o);

    for ([2]bool{ false, true }) |split| {
        for (&outs, [2]bool{ false, true }) |out, presized| {
            var m = try VulkanLM.init(gpa, f.ctx, &f.lm, .{ .initial = total, .max = total });
            defer m.deinit();
            m.io = io;
            if (presized) {
                try m.growLocalRing(localRingMax(cfg));
                try m.ensureBufRows(n_img);
            }
            if (split) try m.enableCpuSplit(.attn, 4 << 30, true);
            try m.prefill(prompt[0..k]);
            try m.prefillImage(rows, 0, 0);
            try std.testing.expectEqual(localRingMax(cfg), m.ring_rows);
            try m.step(io, prompt[k..], out);
        }
        errdefer std.debug.print("split {}: logits after a grown ring differ from a presized one\n", .{split});
        try std.testing.expectEqualSlices(f32, outs[1], outs[0]);
    }
}
