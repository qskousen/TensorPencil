//! Gemma 4 text stack on the CUDA backends (cuda / zig-cuda). Mirrors
//! gemma4.zig's CPU forward op-for-op. Prefill runs in 128-row chunks; decode
//! is per-op (no graph capture). Ported from gemma4.zig + the gemma3_cuda
//! structure, including the hybrid CPU/GPU layer split (host-resident layers
//! run gemma4.layerForward against a PerLayerKvCache shadow; dynamic offload
//! migrates layers as the KV grows). Unlike gemma3's uniform linear shadow,
//! the host shadow here keeps the LOCAL layers' ring layout (PerLayerKvCache
//! speaks rings natively), so migrate/promote/checkpoint copies are raw and
//! wholesale — no ring<->linear segment translation.
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
const cuda = @import("tp_gpu").cuda;
const ops = @import("tp_ops");
const kvmod = @import("tp_core").kv_cache;
const sample = @import("tp_core").sample;
const residency = @import("tp_runtime").residency;
const transformer = @import("transformer.zig");
const transformer_gpu = @import("transformer_gpu.zig");

const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;

/// Map the session KV dtype onto the backend's kernel-format tag.
fn kvFmt(dt: kvmod.KvDtype) cuda.backend.KvFmt {
    return switch (dt) {
        .f32 => .f32,
        .f16 => .f16,
        .q8_0 => .q8_0,
    };
}

const Growable = Backend.GrowableTensor;

/// KV chunks per head in the local-layer decode attention split.
const nsplit = 32;
const nsplit_prefill = 8;
/// Rows per batched-prefill chunk (also the activation-buffer height).
const prefill_chunk = 128;
const grouped_gemv_max = 40;

/// Gemma 4 vision emits up to 280 soft-image tokens (llama.cpp
/// set_limit_image_tokens(40, 280)). A bidirectional image block is prefilled
/// in ONE batch so its tokens see each other, so it (not prefill_chunk) sets
/// the largest single batch: the LOCAL ring slack and (rounded up) the
/// activation-buffer height.
const max_image_tokens = 280;
/// Largest actual batch of rows fed in one forward (text chunk or image block).
const max_batch = @max(prefill_chunk, max_image_tokens);
/// Activation-buffer height. opMatmulQuant pads its output rows up to a
/// multiple of 128, so GEMM-output buffers must reserve the padded height.
const buf_rows = std.mem.alignForward(usize, max_batch, 128);

/// Rows a LOCAL (sliding-window) layer's KV ring holds. The window plus one
/// max-batch of slack: within a single prefill batch the queries span
/// `window + seq - 1` positions, so a ring smaller than this would alias a
/// still-needed key against a freshly-written one. LOCAL caches are fixed at
/// this size and never grow with the conversation (TODO lever 1).
fn localRingRows(cfg: gemma4.Config) usize {
    return cfg.sliding_window + max_batch;
}

/// Kill-switch for the LOCAL-layer sliding-window ring cache (TODO lever 1).
/// When false, every layer reserves full context (pre-ring behaviour) — kept
/// for A/B validation that ring output is token-identical.
const enable_local_ring = true;

/// This layer uses the sliding-window ring: a LOCAL layer, ring enabled.
fn usesRing(cfg: gemma4.Config, l: usize) bool {
    return enable_local_ring and !cfg.isGlobal(l);
}

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
    /// KV-cache element storage type (f32 / f16); selects the attention kernel
    /// variant and the per-element byte stride of k_cache/v_cache.
    kv_dtype: kvmod.KvDtype,
    len: usize,
    /// Set for the duration of a bidirectional image-block prefill: the
    /// `attention` stepper reads it and lets every query in the batch attend
    /// the whole block forward (llama.cpp marks image spans non-causal).
    bidir_prefill: bool = false,
    /// sin-table offsets within each freqs buffer (= cap.max * half).
    sin_off_global: usize,
    sin_off_local: usize,
    /// Global-layer (theta 1e6 + proportional rope_freqs) and local-layer
    /// (theta 1e4) RoPE tables, cos then sin.
    freqs_global: Buf,
    freqs_local: Buf,
    /// Device ones vector (len head_dim_global), for the weightless V RMS-norm.
    ones: Buf,
    /// The suppress list as ready-made penalty entries: the device suppress
    /// mask reuses the penalize scatter with an infinite presence penalty
    /// (see suppressLogits). Arena-owned; empty when the model has none.
    suppress_pen: []const sample.PenaltyEntry,
    bufs: Bufs,
    /// Per-layer K/V caches (variable stride = kvDim(l)).
    k_cache: []Growable,
    v_cache: []Growable,
    /// Io for the host matmuls of a hybrid split's CPU-resident layers; set
    /// by the step entry points, or seeded by the session owner BEFORE the
    /// first forward (tp-gui prefills before any step, and an over-budget
    /// model has host layers from init). null fails the host path closed
    /// (error.SplitIoUnset) instead of undefined-pointer UB.
    io: ?std.Io = null,
    /// Hybrid CPU/GPU split: CPU-resident layers run host matmuls (dynamic
    /// offload migrates more to the host as the KV cache grows). null = fully
    /// device-resident (gemma4 is per-op already, so nothing else changes).
    split: ?Split = null,

    /// Which layers a hybrid CPU/GPU split pushes to the host. Gemma4 is
    /// uniform attention (no recurrent layers), so both policies reduce to a
    /// descending-layer order; kept for interface parity with qwen35_cuda.
    pub const CpuSplitPolicy = enum { tail, attn };

    /// CPU-resident layers of a hybrid split + the host state they need.
    /// Allocated with `gpa`; freed by `freeSplit`. Mirrors gemma3_cuda.Split
    /// with gemma4's per-layer KV geometry: the host shadow is a
    /// PerLayerKvCache whose LOCAL layers keep the SAME fixed ring layout as
    /// the device rings, so migrate/promote/checkpoint are wholesale raw
    /// copies.
    pub const Split = struct {
        /// Per-layer: compute on the device? (false = host).
        on_gpu: []bool,
        n_cpu: usize,
        policy: CpuSplitPolicy,
        /// Host K/V for the CPU-resident layers (full n_layers slots; only the
        /// CPU layers' slots are used). Grows in lockstep with the device
        /// caches; `len` tracks the device `len`.
        cache: kvmod.PerLayerKvCache,
        /// Host activation scratch, sized once for the largest batch
        /// (transformer.layerForward slices it down to each call's seq).
        scratch: gemma4.Scratch,
        /// Host RoPE tables (0 = global with the proportional `rope_freqs`
        /// factors, 1 = local), rebuilt together on capacity growth.
        rope: ops.rope.RopeTables(2),
        /// Host hidden buffer ([max_batch * hidden]).
        hx: []f32,
        /// Does the live hidden currently sit in `hx` (vs device `bufs.x`)?
        on_host: bool = false,
        /// Dynamic offload: migrate more layers GPU->CPU as the KV grows.
        dynamic: bool = false,
        /// Migration priority (layer indices); order[0..next) are already on CPU.
        order: []usize = &.{},
        next: usize = 0,
        /// VRAM ceiling (bytes) the dynamic scheduler keeps device usage under.
        budget: u64 = 0,
    };

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
        // `self = undefined` bypasses the fields' `= null` defaults — set them
        // explicitly (an unset `split` would fault llmResidency; an unset `io`
        // would be undefined-pointer UB in the host layer path).
        self.split = null;
        self.io = null;
        self.lm = lm;
        self.be = be;
        self.gpa = gpa;
        self.cfg = cfg;
        self.capacity = cap.initial;
        self.initial_capacity = cap.initial;
        self.max_capacity = cap.max;
        self.kv_dtype = cap.kv_dtype;
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
        try self.allocKvCaches();

        const spen = try alloc.alloc(sample.PenaltyEntry, lm.suppress_tokens.len);
        for (spen, lm.suppress_tokens) |*e, id| e.* = .{ .id = id, .count = 1 };
        self.suppress_pen = spen;

        self.arena = arena;
        return self;
    }

    /// (Re)create the per-layer K/V device buffers at `self.kv_dtype`, sized from
    /// `self.initial_capacity`/`self.max_capacity`. The `k_cache`/`v_cache`
    /// SLICES must already exist (arena-owned); this fills them with fresh
    /// Growables. Used by `init` and `reinitCache` so the sizing stays in one
    /// place. LOCAL layers are a fixed sliding-window ring (never grow).
    /// Host-resident layers of an armed split are skipped — they keep no
    /// device KV (their shadow lives in `split.cache`).
    fn allocKvCaches(self: *CudaLM) !void {
        const cfg = self.cfg;
        const be = self.be;
        const dt = self.kv_dtype;
        for (self.k_cache, self.v_cache, 0..) |*kb, *vb, l| {
            if (self.split) |*sp| if (!sp.on_gpu[l]) continue;
            const kvd = cfg.kvDim(l);
            if (usesRing(cfg, l)) {
                const bytes = dt.sizeBytes(localRingRows(cfg) * kvd);
                kb.* = try be.growableCreate(bytes, bytes);
                vb.* = try be.growableCreate(bytes, bytes);
            } else {
                kb.* = try be.growableCreate(dt.sizeBytes(self.initial_capacity * kvd), dt.sizeBytes(self.max_capacity * kvd));
                vb.* = try be.growableCreate(dt.sizeBytes(self.initial_capacity * kvd), dt.sizeBytes(self.max_capacity * kvd));
            }
        }
    }

    /// Rebuild the KV cache at a new element dtype (GUI f32<->f16 toggle),
    /// keeping the model WEIGHTS resident: free the K/V buffers, re-create them
    /// at `dtype`, and reset the committed length to 0 so the next forward
    /// re-prefills the whole transcript. gemma4 decodes straight through
    /// `opAttnDecode` (no captured graph to invalidate).
    pub fn reinitCache(self: *CudaLM, dtype: kvmod.KvDtype) !void {
        // Host-resident layers keep no device KV (growableDestroy is
        // idempotent on their already-destroyed buffers; allocKvCaches skips
        // them).
        for (self.k_cache) |*b| self.be.growableDestroy(b);
        for (self.v_cache) |*b| self.be.growableDestroy(b);
        self.kv_dtype = dtype;
        self.capacity = self.initial_capacity;
        self.len = 0;
        try self.allocKvCaches();
        if (self.split) |*sp| {
            // The host shadow stores the device dtype; rebuild it to match.
            const cache = try self.hostShadowCache(dtype);
            sp.cache.deinit(self.gpa);
            sp.cache = cache;
        }
    }

    /// Build the split's host K/V shadow: per-layer dims, LOCAL layers with the
    /// SAME fixed ring layout as the device rings (localRingRows), full-context
    /// linear rows for GLOBAL layers, at the current `self.capacity`.
    fn hostShadowCache(self: *CudaLM, dtype: kvmod.KvDtype) !kvmod.PerLayerKvCache {
        const cfg = self.cfg;
        const dims = try cfg.kvDims(self.gpa);
        defer self.gpa.free(dims);
        const rings = try self.gpa.alloc(usize, cfg.n_layers);
        defer self.gpa.free(rings);
        for (rings, 0..) |*r, l| r.* = if (usesRing(cfg, l)) localRingRows(cfg) else 0;
        return kvmod.PerLayerKvCache.init(self.gpa, self.capacity, dims, rings, dtype);
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
        self.freeSplit();
        self.arena.deinit();
        self.* = undefined;
    }

    /// Free an armed split's host state and disarm it (deinit/resetResidency).
    fn freeSplit(self: *CudaLM) void {
        if (self.split) |*sp| {
            sp.cache.deinit(self.gpa);
            sp.scratch.deinit(self.gpa);
            sp.rope.deinit(self.gpa);
            self.gpa.free(sp.hx);
            self.gpa.free(sp.on_gpu);
            self.gpa.free(sp.order);
            self.split = null;
        }
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
    /// Device VRAM (bytes) currently in use by this backend — for the
    /// end-of-response telemetry (`session.statsOf`).
    pub fn vramUsed(self: *const CudaLM) u64 {
        return self.be.deviceUsed();
    }
    /// Reset the session to an empty context (KV rows are overwritten lazily on
    /// the next prefill; nothing recurrent to zero). A split's host shadow
    /// tracks the device length, so truncate it in lockstep.
    pub fn resetCache(self: *CudaLM) !void {
        self.len = 0;
        if (self.split) |*sp| sp.cache.truncate(0);
    }
    pub fn vocab(self: *const CudaLM) usize {
        return self.cfg.vocab;
    }

    /// Fixed byte size of a turn checkpoint (see `checkpoint`): every LOCAL
    /// (sliding-window) layer's full KV ring (per-layer kvDim). GLOBAL layers
    /// are append-only — truncation alone rolls them back — but a LOCAL ring
    /// OVERWRITES its oldest rows as generation advances, so a rollback
    /// further than the ring slack needs them restored. Context-independent.
    pub fn checkpointBytes(self: *const CudaLM) usize {
        const cfg = self.cfg;
        const dt = self.kv_dtype;
        var total: usize = 0;
        for (0..cfg.n_layers) |l| {
            if (usesRing(cfg, l)) total += 2 * dt.sizeBytes(localRingRows(cfg) * cfg.kvDim(l));
        }
        return total;
    }

    /// Snapshot the non-append-only context state at the current position into
    /// `out` (`checkpointBytes` long): each LOCAL layer's ring, wholesale, read
    /// from whichever side currently owns the layer (the host shadow keeps the
    /// same ring-row layout, so both sides copy raw). Owner-agnostic, so layers
    /// may migrate between snapshot and restore. Pair with
    /// `restoreCheckpoint(out, q)` where `q == cached()` at snapshot time.
    pub fn checkpoint(self: *CudaLM, out: []u8) !void {
        std.debug.assert(out.len == self.checkpointBytes());
        try self.checkpointRings(.save, out);
    }

    /// Roll the context back to `q` committed tokens (a turn boundary) using a
    /// snapshot taken there: GLOBAL layers just truncate (rows past `q` are
    /// overwritten by the next write, on the device growables and the split's
    /// host shadow alike); LOCAL rings are restored wholesale into each
    /// layer's CURRENT owner, bringing back the rows the discarded response
    /// overwrote.
    pub fn restoreCheckpoint(self: *CudaLM, snap: []const u8, q: usize) !void {
        std.debug.assert(snap.len == self.checkpointBytes());
        std.debug.assert(q <= self.len);
        self.len = q;
        if (self.split) |*sp| sp.cache.truncate(q);
        try self.checkpointRings(.restore, snap);
    }

    const CheckpointDir = enum { save, restore };

    /// Shared body of checkpoint/restoreCheckpoint: move every LOCAL layer's
    /// ring between the snapshot buffer (ring-row format, K then V per layer)
    /// and the layer's current owner — a wholesale device copy or, host-side,
    /// a raw memcpy of the shadow's identical ring block.
    fn checkpointRings(self: *CudaLM, comptime dir: CheckpointDir, buf: if (dir == .save) []u8 else []const u8) !void {
        const cfg = self.cfg;
        const ring = localRingRows(cfg);
        const dt = self.kv_dtype;
        var off: usize = 0;
        for (0..cfg.n_layers) |l| {
            if (!usesRing(cfg, l)) continue;
            const ring_bytes = dt.sizeBytes(ring * cfg.kvDim(l));
            const bk = buf[off..][0..ring_bytes];
            const bv = buf[off + ring_bytes ..][0..ring_bytes];
            const on_gpu = if (self.split) |*sp| sp.on_gpu[l] else true;
            if (on_gpu) {
                if (comptime dir == .save) {
                    try self.be.tensorDownload(offsetBufSized(self.k_cache[l].buf, 0, ring_bytes), bk);
                    try self.be.tensorDownload(offsetBufSized(self.v_cache[l].buf, 0, ring_bytes), bv);
                } else {
                    try self.be.tensorUpload(offsetBufSized(self.k_cache[l].buf, 0, ring_bytes), bk);
                    try self.be.tensorUpload(offsetBufSized(self.v_cache[l].buf, 0, ring_bytes), bv);
                }
            } else {
                const sp = &self.split.?;
                const hk = sp.cache.kRowBytes(l, 0, ring);
                const hv = sp.cache.vRowBytes(l, 0, ring);
                if (comptime dir == .save) {
                    @memcpy(bk, hk);
                    @memcpy(bv, hv);
                } else {
                    @memcpy(hk, bk);
                    @memcpy(hv, bv);
                }
            }
            off += 2 * ring_bytes;
        }
    }

    // --- Hybrid CPU/GPU split (residency hooks + scheduling delegates) ------

    /// Total device footprint of one layer's streamable weights (quantized
    /// bytes) — the projection + MLP matrices; norms are negligible. GLOBAL
    /// layers have no v_proj (V reuses the raw K projection). `anytype` avoids
    /// naming gemma4's private `Layer` type.
    fn layerDeviceBytes(layer: anytype) usize {
        const v_bytes = if (layer.v) |vw| vw.bytes.len else 0;
        return layer.q.bytes.len + layer.k.bytes.len + v_bytes + layer.o.bytes.len +
            layer.gate.bytes.len + layer.up.bytes.len + layer.down.bytes.len;
    }

    /// Move layer `l`'s live K/V device->host, free its device K/V + weights,
    /// and mark it CPU-resident (a `residency` hook). The host shadow stores
    /// the same dtype AND (for LOCAL layers) the same ring layout as the
    /// device caches, so every copy is raw — rings move wholesale.
    pub fn migrateLayer(self: *CudaLM, l: usize) !void {
        const cfg = self.cfg;
        const sp = &self.split.?;
        const kvd = cfg.kvDim(l);
        const dt = self.kv_dtype;
        if (self.len > 0) {
            const rows = if (usesRing(cfg, l)) localRingRows(cfg) else self.len;
            try self.be.tensorDownload(offsetBufSized(self.k_cache[l].buf, 0, dt.sizeBytes(rows * kvd)), sp.cache.kRowBytes(l, 0, rows));
            try self.be.tensorDownload(offsetBufSized(self.v_cache[l].buf, 0, dt.sizeBytes(rows * kvd)), sp.cache.vRowBytes(l, 0, rows));
        }
        self.be.growableDestroy(&self.k_cache[l]);
        self.be.growableDestroy(&self.v_cache[l]);
        // Free the migrated layer's device weights (the host path reads them from
        // the GGUF mapping) — the bulk of the reclaimed VRAM.
        const layer = &self.lm.layers[l];
        self.be.evictWeightBytes(layer.q.bytes);
        self.be.evictWeightBytes(layer.k.bytes);
        if (layer.v) |vw| self.be.evictWeightBytes(vw.bytes);
        self.be.evictWeightBytes(layer.o.bytes);
        self.be.evictWeightBytes(layer.gate.bytes);
        self.be.evictWeightBytes(layer.up.bytes);
        self.be.evictWeightBytes(layer.down.bytes);
        sp.on_gpu[l] = false;
        sp.n_cpu += 1;
        std.log.debug("[offload] layer {d} (attn) -> CPU at ctx {d} ({d}/{d} on CPU)", .{ l, self.len, sp.n_cpu, cfg.n_layers });
    }

    /// Migrate layers to the host until `@min(budget - deviceUsed, headroom)`
    /// reaches `needed_free` bytes, or nothing is left. Fixed-target variant used
    /// by the VRAM coordinator (free room for the image model). No-op without a
    /// dynamic split. (ensureCapacity keeps its own loop, whose target shrinks per
    /// iteration as layers migrate — a fixed target here can't express that.)
    pub fn offloadUntilFree(self: *CudaLM, needed_free: u64) !void {
        return residency.offloadUntilFree(self, needed_free);
    }

    /// Migrate layers until the LLM's actual total device usage is ≤ `target`
    /// bytes (balanced mode: settle the LLM to its share only when an image model
    /// loads). Live `deviceUsed()`, one-way + idempotent. See qwen35_cuda.
    pub fn offloadToBudget(self: *CudaLM, target: u64) !void {
        return residency.offloadToBudget(self, target);
    }

    /// `residency.promoteBack` cost hook: VRAM a promote of layer `l` needs — its
    /// streamable weights, the KV it re-commits (fixed ring for LOCAL layers, the
    /// current capacity for GLOBAL), plus slack.
    pub fn promoteCost(self: *CudaLM, l: usize) usize {
        const cfg = self.cfg;
        const rows = if (usesRing(cfg, l)) localRingRows(cfg) else self.capacity;
        const kv = 2 * self.kv_dtype.sizeBytes(rows * cfg.kvDim(l));
        return layerDeviceBytes(&self.lm.layers[l]) + kv + (64 << 20);
    }

    /// Bring layer `l` back onto the GPU, preserving its accumulated K/V:
    /// re-create the device K/V and upload the host rows (wholesale ring for
    /// LOCAL, [0,len) for GLOBAL). Weights re-cache lazily on the next GPU
    /// forward. Reverse of migrateLayer.
    pub fn promoteLayer(self: *CudaLM, l: usize) !void {
        const cfg = self.cfg;
        const sp = &self.split.?;
        const kvd = cfg.kvDim(l);
        const dt = self.kv_dtype;
        if (usesRing(cfg, l)) {
            const bytes = dt.sizeBytes(localRingRows(cfg) * kvd);
            self.k_cache[l] = try self.be.growableCreate(bytes, bytes);
            self.v_cache[l] = try self.be.growableCreate(bytes, bytes);
        } else {
            self.k_cache[l] = try self.be.growableCreate(dt.sizeBytes(self.capacity * kvd), dt.sizeBytes(self.max_capacity * kvd));
            self.v_cache[l] = try self.be.growableCreate(dt.sizeBytes(self.capacity * kvd), dt.sizeBytes(self.max_capacity * kvd));
        }
        if (self.len > 0) {
            const rows = if (usesRing(cfg, l)) localRingRows(cfg) else self.len;
            try self.be.tensorUpload(offsetBufSized(self.k_cache[l].buf, 0, dt.sizeBytes(rows * kvd)), sp.cache.kRowBytes(l, 0, rows));
            try self.be.tensorUpload(offsetBufSized(self.v_cache[l].buf, 0, dt.sizeBytes(rows * kvd)), sp.cache.vRowBytes(l, 0, rows));
        }
        sp.on_gpu[l] = true;
        sp.n_cpu -= 1;
        std.log.debug("[promote] layer {d} -> GPU at ctx {d} ({d}/{d} on CPU)", .{ l, self.len, sp.n_cpu, cfg.n_layers });
    }

    /// Migrate CPU layers back onto the GPU (LIFO by offload order), stopping
    /// before the next one would overflow `budget`. Keeps the split armed
    /// (offload can fire again). Returns the number promoted; 0 without a split.
    pub fn promoteLayers(self: *CudaLM, budget: u64) !usize {
        return residency.promoteBack(self, budget);
    }

    /// New-chat reset: drop the split, shrink every K/V cache back to the initial
    /// capacity (frees the grown VRAM), clear the context, and re-arm dynamic
    /// offload for the fresh small context. KV is discarded, so no host->device
    /// copy is needed (unlike promoteLayers).
    pub fn resetResidency(self: *CudaLM, budget: u64) !void {
        self.freeSplit();
        for (self.k_cache) |*b| self.be.growableDestroy(b);
        for (self.v_cache) |*b| self.be.growableDestroy(b);
        self.capacity = self.initial_capacity;
        try self.allocKvCaches();
        try self.resetCache();
        _ = try self.autoOffload(budget);
    }

    /// Always arm the dynamic split (text sessions; `budget == 0` = no offload).
    /// Free when the model fits (0 layers on CPU, per-op decode), and migrates
    /// layers on demand as the KV cache grows — so over-budget growth degrades
    /// via CPU offload (faster than weight streaming) rather than the streaming
    /// fallback. See qwen35_cuda.autoOffload for the measured rationale.
    pub fn autoOffload(self: *CudaLM, budget: u64) !bool {
        if (budget == 0) return false;
        try self.enableCpuSplit(.attn, budget, true);
        return true;
    }

    /// Place layers on the host until the device-resident weights fit under
    /// `budget` (bytes). `dynamic` packs the GPU now (head-only reserve) and
    /// migrates on demand as the KV grows; static reserves generously. No-op
    /// static split (all fit) leaves `self.split == null`. Mirrors
    /// gemma3_cuda.enableCpuSplit with gemma4's per-layer KV geometry.
    pub fn enableCpuSplit(self: *CudaLM, policy: CpuSplitPolicy, budget: u64, dynamic: bool) !void {
        const cfg = self.cfg;
        const n = cfg.n_layers;
        const gpa = self.gpa;

        const per = try gpa.alloc(usize, n);
        defer gpa.free(per);
        var total_weight: usize = 0;
        for (self.lm.layers, 0..) |*layer, l| {
            per[l] = layerDeviceBytes(layer);
            total_weight += per[l];
        }

        // Device memory that must stay resident: KV + LM head + slack. Per-layer
        // KV widths; LOCAL ring layers hold only localRingRows.
        var kv_bytes: usize = 0;
        for (0..n) |l| {
            const rows = if (usesRing(cfg, l)) localRingRows(cfg) else self.capacity;
            kv_bytes += 2 * self.kv_dtype.sizeBytes(rows * cfg.kvDim(l));
        }
        const reserve = if (dynamic)
            self.lm.head.bytes.len
        else
            kv_bytes + self.lm.head.bytes.len + (512 << 20);
        // The plan must respect the card's LIVE free VRAM, not just the
        // abstract budget: other processes may hold a chunk of the card, and
        // the already-committed KV/activation buffers count against it too. A
        // plan the card can't satisfy places everything resident and then
        // faults at the first prefill (weight uploads + lazy PTX JIT collide
        // at zero free). headroom() already keeps a 10% margin.
        const used = self.be.deviceUsed();
        const avail = @min(budget, used + self.be.headroom()) -| used;
        const gpu_weight_budget: usize = if (avail > reserve) @intCast(avail - reserve) else 0;

        // Eviction order: last layer leaves first (descending). gemma4 is
        // all-attention, so `.attn` and `.tail` coincide.
        const order = try gpa.alloc(usize, n);
        errdefer gpa.free(order);
        for (0..n) |i| order[i] = n - 1 - i;

        const on_gpu = try gpa.alloc(bool, n);
        errdefer gpa.free(on_gpu);
        @memset(on_gpu, true);
        var gpu_weight = total_weight;
        var n_cpu: usize = 0;
        while (gpu_weight > gpu_weight_budget and n_cpu < n) {
            gpu_weight -= per[order[n_cpu]];
            n_cpu += 1;
        }
        if (n_cpu == 0 and !dynamic) {
            gpa.free(on_gpu);
            gpa.free(order);
            return; // everything fits resident — no split needed
        }

        // Host state for the CPU-resident layers (sized to the current KV
        // capacity; grows with the device via ensureCapacity). The host shadow
        // stores the SAME dtype and ring layout as the device caches, so
        // migrate/promote and the ring checkpoints are raw byte copies.
        var cache = try self.hostShadowCache(self.kv_dtype);
        errdefer cache.deinit(gpa);
        // The host shadow tracks the SAME committed length as the device from
        // the moment the split arms (per-step commits keep them in lockstep
        // afterwards). Armed mid-conversation (imageReclaim), starting at 0
        // would make host layers attend over nothing; migrateLayer copies each
        // migrated layer's live rows so declaring them committed is correct.
        cache.len = self.len;
        var scratch = try gemma4.Scratch.init(gpa, max_batch, cfg);
        errdefer scratch.deinit(gpa);
        var rope = try ops.rope.RopeTables(2).init(gpa, .{
            .{ .head_dim = cfg.head_dim_global, .theta = cfg.rope_theta, .freq_factors = self.lm.rope_freqs },
            .{ .head_dim = cfg.head_dim_local, .theta = cfg.rope_theta_local },
        }, self.capacity);
        errdefer rope.deinit(gpa);
        const hx = try gpa.alloc(f32, max_batch * cfg.hidden);
        errdefer gpa.free(hx);

        self.split = .{
            .on_gpu = on_gpu,
            .n_cpu = 0, // the placement loops below mark + count the host layers
            .policy = policy,
            .cache = cache,
            .scratch = scratch,
            .rope = rope,
            .hx = hx,
            .dynamic = dynamic,
            .order = order,
            .next = n_cpu,
            .budget = budget,
        };

        // Place the statically-planned layers on the host. Before any tokens
        // (autoOffload-at-init) there is nothing to copy — mark them and free
        // the device K/V; weights are reclaimed lazily. Armed MID-conversation
        // (imageReclaim), each layer's live rows must move to the host instead
        // — migrateLayer does the copy AND the on_gpu/n_cpu bookkeeping, or
        // the context would be destroyed with the device KV.
        if (self.len == 0) {
            const sp = &self.split.?;
            for (order[0..n_cpu]) |l| {
                sp.on_gpu[l] = false;
                sp.n_cpu += 1;
                self.be.growableDestroy(&self.k_cache[l]);
                self.be.growableDestroy(&self.v_cache[l]);
            }
        } else {
            for (order[0..n_cpu]) |l| try self.migrateLayer(l);
        }
    }

    /// Device bytes a grow to `target` rows would commit across the still
    /// on-GPU, growable (non-ring GLOBAL) layers — per-layer KV widths summed
    /// (gemma3's uniform `liveSlots * add` doesn't apply here).
    fn growBytes(self: *CudaLM, target: usize) u64 {
        const add = target - self.capacity;
        var need: u64 = 0;
        for (0..self.cfg.n_layers) |l| {
            if (usesRing(self.cfg, l)) continue;
            if (self.split) |*sp| if (!sp.on_gpu[l]) continue;
            need += 2 * self.kv_dtype.sizeBytes(add * self.cfg.kvDim(l));
        }
        return need;
    }

    pub fn ensureCapacity(self: *CudaLM, min_rows: usize) !void {
        const target = (try kvmod.growPlan(self.capacity, self.max_capacity, min_rows)) orelse return;

        // Dynamic offload: migrate layers GPU->CPU until there's headroom to grow
        // the device KV, instead of streaming weights (the cliff). Each migrated
        // layer frees its device KV + weight VRAM (and shrinks the next need —
        // recompute per iteration). Mirrors gemma3_cuda.
        if (self.split) |*sp| if (sp.dynamic) {
            while (true) {
                const need = self.growBytes(target) + (32 << 20); // + margin
                const free = @min(sp.budget -| self.be.deviceUsed(), self.be.headroom());
                if (free >= need) break;
                if (!(try residency.migrateNext(self))) break; // nothing left; fall through
            }
        };

        // Grow device KV of the GLOBAL layers still on the GPU (LOCAL ring
        // layers are fixed-size — never grow them). Physical VRAM can be
        // exhausted even when the proactive migration above thought there was
        // room: a resident image model on another CUDA context may grab it
        // between the headroom check and this commit. On a real OOM, offload
        // one more layer to the CPU and retry the whole grow, so a full window
        // only ever fails once nothing is left to migrate. growableEnsure is
        // idempotent, so re-running the loop is cheap.
        grow: while (true) {
            for (0..self.cfg.n_layers) |l| {
                if (usesRing(self.cfg, l)) continue;
                if (self.split) |*sp| if (!sp.on_gpu[l]) continue;
                // Byte size MUST match how the buffers were created (kv_dtype
                // block math), or an f16/q8_0 cache requests f32-sized growth,
                // overshoots its VA reservation, and growableEnsure fails with
                // DeviceOutOfMemory → ContextFull once the window grows past
                // ~max_capacity/2.
                const bytes = self.kv_dtype.sizeBytes(target * self.cfg.kvDim(l));
                for ([2]*Growable{ &self.k_cache[l], &self.v_cache[l] }) |b| {
                    self.be.growableEnsure(b, bytes) catch |err| switch (err) {
                        error.DeviceOutOfMemory, error.OutOfMemory => {
                            if (self.split != null and try residency.migrateNext(self)) continue :grow;
                            return error.ContextFull;
                        },
                        else => return err,
                    };
                }
            }
            break;
        }
        // A hybrid split keeps host KV/RoPE for its CPU layers; grow them in
        // lockstep so host positions stay aligned with the device len.
        if (self.split) |*sp| {
            sp.cache.grow(self.gpa, target) catch return error.ContextFull;
            sp.rope.regrow(self.gpa, target) catch return error.ContextFull;
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
            try self.embedChunk(ids[off..][0..n], if (last) .{ .host = logits } else .none);
            off += n;
        }
    }

    /// Forward `ids`, leaving the last row's RAW logits resident in
    /// bufs.logits for the on-device sampling ops.
    fn forwardDeviceLogits(self: *CudaLM, io: std.Io, ids: []const u32) !void {
        self.io = io;
        std.debug.assert(ids.len >= 1 and ids.len <= self.remaining());
        var off: usize = 0;
        while (off < ids.len) {
            const n: usize = @min(prefill_chunk, ids.len - off);
            const last = (off + n == ids.len);
            try self.embedChunk(ids[off..][0..n], if (last) .device else .none);
            off += n;
        }
    }

    /// Force the suppress_tokens to -inf on the DEVICE logits, mirroring
    /// finalizeLogits's masking: reuses the penalize scatter with an infinite
    /// presence penalty — repeat penalty 1.0 leaves the logit itself untouched
    /// (x/1 is exact) and any finite logit minus +inf is exactly -inf.
    fn suppressLogits(self: *CudaLM, lg: Buf) !void {
        const sp: sample.Params = .{ .repeat_penalty = 1.0, .presence_penalty = std.math.inf(f32) };
        var off: usize = 0;
        while (off < self.suppress_pen.len) {
            const n: usize = @min(sample.max_penalty_window, self.suppress_pen.len - off);
            try self.be.opPenalize(lg, self.suppress_pen[off..][0..n], sp);
            off += n;
        }
    }

    /// Greedy decode without the vocab download. The tanh softcap is strictly
    /// MONOTONIC, so the argmax over the raw (suppressed) device logits is the
    /// argmax over the finalized logits — no device tanh needed. Matches
    /// sample.argmax up to softcap rounding collapsing two distinct raw logits
    /// onto one capped value (needs |logit| far beyond real model output).
    pub fn stepArgmax(self: *CudaLM, io: std.Io, ids: []const u32) !u32 {
        return self.stepArgmaxPen(io, ids, &.{}, .{});
    }

    /// `stepArgmax` with sampling penalties. Penalties are NOT monotonic over
    /// the capped logits, so the winner comes from the finalized candidate set
    /// (stepSelectPen's superset argument) instead of a raw device argmax.
    pub fn stepArgmaxPen(self: *CudaLM, io: std.Io, ids: []const u32, pen: []const sample.PenaltyEntry, sp: sample.Params) !u32 {
        if (pen.len != 0) {
            const cap = self.maxSelect();
            const out_id = try self.gpa.alloc(u32, cap);
            defer self.gpa.free(out_id);
            const out_logit = try self.gpa.alloc(f32, cap);
            defer self.gpa.free(out_logit);
            const count = try self.stepSelectPen(io, ids, pen, sp, out_id, out_logit);
            var best: usize = 0; // highest logit, ties to the lowest id (sample.argmax)
            for (1..count) |i| {
                if (out_logit[i] > out_logit[best] or
                    (out_logit[i] == out_logit[best] and out_id[i] < out_id[best])) best = i;
            }
            return out_id[best];
        }
        try self.forwardDeviceLogits(io, ids);
        const be = self.be;
        const b = &self.bufs;
        const lg = offsetBufSized(b.logits, 0, self.cfg.vocab * 4);
        try self.suppressLogits(lg);
        try be.opArgmax(lg, self.cfg.vocab, b.argmax_out, &b.argmax_v, &b.argmax_i);
        var idf: [1]f32 = undefined;
        try be.tensorDownload(b.argmax_out, std.mem.sliceAsBytes(&idf));
        return @intFromFloat(idf[0]);
    }

    /// Max candidates stepSelect can return (host buffer sizing for the engine).
    pub fn maxSelect(self: *const CudaLM) usize {
        _ = self;
        return cuda.backend.topk_lanes * cuda.backend.topk_m;
    }

    /// Stochastic decode: on-device suppress + top-k over the RAW logits (the
    /// monotonic softcap preserves the selection), then the candidates are
    /// finalized on the host — exact softcap (bit-identical tanh), suppress
    /// mask, penalties (gemma4.finalizeCandidates). Returns the candidate count.
    pub fn stepSelect(self: *CudaLM, io: std.Io, ids: []const u32, out_id: []u32, out_logit: []f32) !usize {
        return self.stepSelectPen(io, ids, &.{}, .{}, out_id, out_logit);
    }

    pub fn stepSelectPen(self: *CudaLM, io: std.Io, ids: []const u32, pen: []const sample.PenaltyEntry, sp: sample.Params, out_id: []u32, out_logit: []f32) !usize {
        try self.forwardDeviceLogits(io, ids);
        const be = self.be;
        const b = &self.bufs;
        const lg = offsetBufSized(b.logits, 0, self.cfg.vocab * 4);
        try self.suppressLogits(lg);
        const count = try be.opTopK(lg, self.cfg.vocab, &b.topk_v, &b.topk_i);
        std.debug.assert(count <= out_id.len and count <= out_logit.len);
        try be.tensorDownload(b.topk_v, std.mem.sliceAsBytes(out_logit[0..count]));
        var idx_f: [cuda.backend.topk_lanes * cuda.backend.topk_m]f32 = undefined;
        try be.tensorDownload(b.topk_i, std.mem.sliceAsBytes(idx_f[0..count]));
        for (out_id[0..count], idx_f[0..count]) |*o, f| o.* = @intFromFloat(f);
        self.lm.finalizeCandidates(out_id[0..count], out_logit[0..count], pen, sp);
        return count;
    }

    /// Prefill text tokens (no logits) — for interleaving with prefillImage.
    pub fn prefill(self: *CudaLM, ids: []const u32) !void {
        var off: usize = 0;
        while (off < ids.len) {
            const n: usize = @min(prefill_chunk, ids.len - off);
            try self.embedChunk(ids[off..][0..n], .none);
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
        // A bidirectional image block must be one batch (a later chunk's KV is
        // not committed when an earlier chunk runs); it fits in max_batch rows.
        std.debug.assert(total <= max_batch);
        self.bidir_prefill = true;
        defer self.bidir_prefill = false;
        try self.forwardRows(embeds, .none);
    }

    /// Where a forward leaves the last row's logits: nowhere (prefill), a host
    /// buffer (download + finalize — the CPU-sampling path), or resident on
    /// the device in bufs.logits, RAW (no softcap/suppress) — for the GPU
    /// sampling path, which suppresses on-device and finalizes the downloaded
    /// candidates host-side (gemma4.finalizeCandidates).
    const LogitsOut = union(enum) { none, host: []f32, device };

    fn embedChunk(self: *CudaLM, ids: []const u32, out: LogitsOut) !void {
        const cfg = self.cfg;
        const n = ids.len;
        const x = try self.gpa.alloc(f32, n * cfg.hidden);
        defer self.gpa.free(x);
        try qwen3.embedTokens(self.lm.embed, ids, x);
        const scale = cfg.embedScale();
        for (x) |*v| v.* *= scale;
        try self.forwardRows(x, out);
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
    /// Store `n` K/V elements from `src` (+`src_off` elems) into cache buffer
    /// `dst` at row-element offset `dst_off`. f32 caches copy raw; f16/q8_0
    /// caches convert the f32 projection on store (opStoreKvF16/Q8). All
    /// offsets and `n` are element counts (dtype-agnostic).
    fn storeKv(self: *CudaLM, dst: Buf, dst_off: usize, src: Buf, src_off: usize, n: usize) !void {
        switch (self.kv_dtype) {
            .f16 => try self.be.opStoreKvF16(dst, dst_off, src, src_off, n),
            .q8_0 => try self.be.opStoreKvQ8(dst, dst_off, src, src_off, n),
            .f32 => try self.be.tensorCopy(dst, dst_off * 4, src, src_off * 4, n * 4),
        }
    }

    pub fn appendKV(self: *CudaLM, l: usize, seq: usize, pos0: usize) !void {
        const cfg = self.cfg;
        const b = &self.bufs;
        const kv_dim = cfg.kvDim(l);
        if (!usesRing(cfg, l)) {
            try self.storeKv(self.k_cache[l].buf, pos0 * kv_dim, b.k, 0, seq * kv_dim);
            try self.storeKv(self.v_cache[l].buf, pos0 * kv_dim, b.v, 0, seq * kv_dim);
            return;
        }
        // LOCAL layer: write into the ring at row pos0%ring, splitting the copy
        // when it wraps the ring boundary (seq <= max_batch < ring, so at most
        // one wrap).
        const ring = localRingRows(cfg);
        const start = pos0 % ring;
        const first = @min(seq, ring - start);
        try self.storeKv(self.k_cache[l].buf, start * kv_dim, b.k, 0, first * kv_dim);
        try self.storeKv(self.v_cache[l].buf, start * kv_dim, b.v, 0, first * kv_dim);
        if (first < seq) {
            const rest = seq - first;
            try self.storeKv(self.k_cache[l].buf, 0, b.k, first * kv_dim, rest * kv_dim);
            try self.storeKv(self.v_cache[l].buf, 0, b.v, first * kv_dim, rest * kv_dim);
        }
    }
    pub fn attention(self: *CudaLM, l: usize, seq: usize, pos0: usize) !void {
        const cfg = self.cfg;
        const b = &self.bufs;
        const ns: usize = if (seq == 1) nsplit else nsplit_prefill;
        const window: usize = if (cfg.isGlobal(l)) 0 else cfg.sliding_window;
        const ring: usize = if (usesRing(cfg, l)) localRingRows(cfg) else 0;
        try self.be.opAttnDecode(b.q, self.k_cache[l].buf, self.v_cache[l].buf, b.attn, b.attn_scratch, pos0 + 1, seq, cfg.n_heads, cfg.nKv(l), cfg.headDim(l), ns, 1.0, window, ring, self.bidir_prefill, kvFmt(self.kv_dtype));
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

    fn forwardRows(self: *CudaLM, x_host: []const f32, out: LogitsOut) !void {
        const be = self.be;
        const cfg = self.cfg;
        const b = &self.bufs;
        const n = x_host.len / cfg.hidden;
        const eps = cfg.rms_eps;
        const pos0 = self.len;
        std.debug.assert(n >= 1 and n <= max_batch and n <= self.remaining());

        try be.tensorUpload(offsetBufSized(b.x, 0, n * cfg.hidden * 4), std.mem.sliceAsBytes(x_host));

        try be.beginBatch();
        errdefer if (be.batching()) be.abortBatch();

        // Hybrid split: each chunk begins with the hidden on the device (bufs.x).
        if (self.split) |*sp| sp.on_host = false;

        for (self.lm.layers, 0..) |*layer, l| {
            // Hybrid CPU/GPU split: run host-resident layers on the CPU via the
            // shared gemma4.layerForward, ferrying the hidden across the
            // device<->host boundary only when residency changes.
            if (self.split) |*sp| {
                if (!sp.on_gpu[l]) {
                    if (!sp.on_host) {
                        try be.tensorDownload(offsetBufSized(b.x, 0, n * cfg.hidden * 4), std.mem.sliceAsBytes(sp.hx[0 .. n * cfg.hidden]));
                        sp.on_host = true;
                    }
                    const host_io = self.io orelse return error.SplitIoUnset;
                    try gemma4.layerForward(host_io, self.gpa, cfg, layer, sp.hx[0 .. n * cfg.hidden], n, sp.rope.get(0), sp.rope.get(1), &sp.cache, l, self.bidir_prefill, &sp.scratch);
                    continue;
                }
                if (sp.on_host) {
                    try be.tensorUpload(offsetBufSized(b.x, 0, n * cfg.hidden * 4), std.mem.sliceAsBytes(sp.hx[0 .. n * cfg.hidden]));
                    sp.on_host = false;
                }
            }
            try transformer_gpu.decoderLayer(transformer.gemma4_spec, self, layer, l, n, pos0);
        }

        // If the last layers ran on the host (the descending order migrates
        // the last layers first, so this is the common case), bring the final
        // hidden back to the device for the LM head.
        if (self.split) |*sp| if (sp.on_host) {
            try be.tensorUpload(offsetBufSized(b.x, 0, n * cfg.hidden * 4), std.mem.sliceAsBytes(sp.hx[0 .. n * cfg.hidden]));
            sp.on_host = false;
        };

        if (out != .none) {
            const h = cfg.hidden;
            try be.qkNorm(offsetBufSized(b.x, (n - 1) * h * 4, h * 4), b.t, try nbuf(be, self.lm.final_norm), 1, h, eps);
            try self.lmHead(b.logits, b.t);
            try be.endBatch();
            self.len += n;
            if (self.split) |*sp| sp.cache.commit(n); // keep host len == device len
            switch (out) {
                .host => |dst| {
                    try be.tensorDownload(offsetBufSized(b.logits, 0, cfg.vocab * 4), std.mem.sliceAsBytes(dst));
                    self.lm.finalizeLogits(dst); // tanh softcap + suppress tokens
                },
                .device => {}, // RAW logits stay resident for the GPU sampling path
                .none => unreachable,
            }
        } else {
            try be.endBatch();
            self.len += n;
            if (self.split) |*sp| sp.cache.commit(n);
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
    argmax_v: Buf,
    argmax_i: Buf,
    argmax_out: Buf,
    topk_v: Buf,
    topk_i: Buf,

    fn init(be: *Backend, cfg: gemma4.Config) !Bufs {
        // Height for the largest single batch (text chunk or whole image block),
        // rounded up to the GEMM's 128-row output padding.
        const pc = buf_rows;
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
            4096, // argmax_v (>= opArgmax lane count)
            4096, // argmax_i
            1, // argmax_out (1 id)
            cuda.backend.topk_lanes * cuda.backend.topk_m, // topk_v
            cuda.backend.topk_lanes * cuda.backend.topk_m, // topk_i
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

// Gated on -Dintegration + a CUDA device + the real 12B checkpoint: greedy
// regeneration after `restoreCheckpoint` must be TOKEN-IDENTICAL. Geometry as
// in the gemma3_cuda test: prompt > ring (wrapped) and generation > ring slack,
// so the discarded response OVERWRITES live-window rows that only the
// checkpoint's ring snapshot can bring back. Also re-runs under a
// mid-conversation CPU split (wholesale ring migrate + owner-aware restore),
// asserting repeatability there and bit-identity again after promoting back
// (hybrid CPU/GPU arithmetic isn't guaranteed bit-identical to all-GPU, so the
// split leg can't compare to seq1 directly — mirrors the gemma3_cuda test).
test "checkpoint restore regenerates token-identical on the real model" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const test_gate = @import("../test_gate.zig");
    const Gguf = @import("tp_core").gguf.Gguf;
    const path = "/home/qt/genai/lmstudio/models/gemma-4-12b-it-qat-q4_0.gguf";
    try test_gate.requireModelFile(io, path);
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    var g = try Gguf.open(gpa, io, path);
    defer g.deinit();
    var lm = try gemma4.Model.load(gpa, &g);
    defer lm.deinit();

    const ring = localRingRows(lm.cfg);
    const prompt_len = ring + 64;
    const n_gen = (ring - lm.cfg.sliding_window) + 64;
    const total = prompt_len + n_gen + 8;
    var model = try CudaLM.init(gpa, be, &lm, .{ .initial = total, .max = total });
    defer model.deinit();

    const prompt = try gpa.alloc(u32, prompt_len);
    defer gpa.free(prompt);
    for (prompt, 0..) |*t, i| t.* = @intCast(1000 + (i * 37) % 50000);
    try model.prefill(prompt[0 .. prompt.len - 1]);
    const q = model.cached();
    const snap = try gpa.alloc(u8, model.checkpointBytes());
    defer gpa.free(snap);
    try model.checkpoint(snap);

    const seq1 = try gpa.alloc(u32, n_gen);
    defer gpa.free(seq1);
    var cur: u32 = prompt[prompt.len - 1];
    for (seq1) |*t| {
        cur = try model.stepArgmax(io, &.{cur});
        t.* = cur;
    }

    try model.restoreCheckpoint(snap, q);
    try std.testing.expectEqual(q, model.cached());
    const seq2 = try gpa.alloc(u32, n_gen);
    defer gpa.free(seq2);
    cur = prompt[prompt.len - 1];
    for (seq2) |*t| {
        cur = try model.stepArgmax(io, &.{cur});
        t.* = cur;
    }
    try std.testing.expectEqualSlices(u32, seq1, seq2);

    // Mid-conversation split: migrate a slice of layers to the host, restore,
    // and assert two restored continuations agree (repeatability).
    try model.enableCpuSplit(.tail, std.math.maxInt(u64), true);
    model.io = io; // the session owner's contract (host layers need it)
    try model.offloadToBudget(be.deviceUsed() * 4 / 5);
    try model.restoreCheckpoint(snap, q);
    const seq3 = try gpa.alloc(u32, n_gen);
    defer gpa.free(seq3);
    cur = prompt[prompt.len - 1];
    for (seq3) |*t| {
        cur = try model.stepArgmax(io, &.{cur});
        t.* = cur;
    }
    try model.restoreCheckpoint(snap, q);
    const seq4 = try gpa.alloc(u32, n_gen);
    defer gpa.free(seq4);
    cur = prompt[prompt.len - 1];
    for (seq4) |*t| {
        cur = try model.stepArgmax(io, &.{cur});
        t.* = cur;
    }
    try std.testing.expectEqualSlices(u32, seq3, seq4);

    // Promote everything back and restore once more: the rings round-trip
    // host→device wholesale, so an owner-misdirected restore would now surface
    // as divergence from the all-GPU baseline.
    _ = try model.promoteLayers(std.math.maxInt(u64));
    try model.restoreCheckpoint(snap, q);
    const seq5 = try gpa.alloc(u32, n_gen);
    defer gpa.free(seq5);
    cur = prompt[prompt.len - 1];
    for (seq5) |*t| {
        cur = try model.stepArgmax(io, &.{cur});
        t.* = cur;
    }
    try std.testing.expectEqualSlices(u32, seq1, seq5);

    // f16 KV variant: rebuild the cache at f16 (context cleared, host shadow
    // rebuilt at f16 too — the split stays armed), re-prefill, and repeat the
    // checkpoint round-trip — the ring snapshot/restore is dtype-agnostic byte
    // copies, but the dtype-scaled byte sizing must agree.
    try model.reinitCache(.f16);
    try model.prefill(prompt[0 .. prompt.len - 1]);
    const q16 = model.cached();
    const snap16 = try gpa.alloc(u8, model.checkpointBytes());
    defer gpa.free(snap16);
    try model.checkpoint(snap16);
    const seq6 = try gpa.alloc(u32, n_gen);
    defer gpa.free(seq6);
    cur = prompt[prompt.len - 1];
    for (seq6) |*t| {
        cur = try model.stepArgmax(io, &.{cur});
        t.* = cur;
    }
    try model.restoreCheckpoint(snap16, q16);
    const seq7 = try gpa.alloc(u32, n_gen);
    defer gpa.free(seq7);
    cur = prompt[prompt.len - 1];
    for (seq7) |*t| {
        cur = try model.stepArgmax(io, &.{cur});
        t.* = cur;
    }
    try std.testing.expectEqualSlices(u32, seq6, seq7);

    // f16 + split: the host shadow stores packed f16 (byte-identical to the
    // device rings), so migrate + owner-aware restore must stay exact within
    // the f16 session (repeatability across two restored continuations).
    try model.offloadToBudget(be.deviceUsed() * 4 / 5);
    try model.restoreCheckpoint(snap16, q16);
    const seq8 = try gpa.alloc(u32, n_gen);
    defer gpa.free(seq8);
    cur = prompt[prompt.len - 1];
    for (seq8) |*t| {
        cur = try model.stepArgmax(io, &.{cur});
        t.* = cur;
    }
    try model.restoreCheckpoint(snap16, q16);
    const seq9 = try gpa.alloc(u32, n_gen);
    defer gpa.free(seq9);
    cur = prompt[prompt.len - 1];
    for (seq9) |*t| {
        cur = try model.stepArgmax(io, &.{cur});
        t.* = cur;
    }
    try std.testing.expectEqualSlices(u32, seq8, seq9);

    // q8_0 KV variant: rebuild the cache at q8_0 (host shadow rebuilt too),
    // re-prefill, and repeat the checkpoint round-trip — the ring snapshots
    // stay raw byte copies of the ggml blocks at the block-aware size.
    _ = try model.promoteLayers(std.math.maxInt(u64));
    try model.reinitCache(.q8_0);
    try model.prefill(prompt[0 .. prompt.len - 1]);
    const q8 = model.cached();
    const snap8 = try gpa.alloc(u8, model.checkpointBytes());
    defer gpa.free(snap8);
    try model.checkpoint(snap8);
    const seq10 = try gpa.alloc(u32, n_gen);
    defer gpa.free(seq10);
    cur = prompt[prompt.len - 1];
    for (seq10) |*t| {
        cur = try model.stepArgmax(io, &.{cur});
        t.* = cur;
    }
    // Offload most layers mid-session, restore, and regenerate twice: the
    // owner-aware ring restore + host-shadow attention must be repeatable.
    try model.offloadToBudget(be.deviceUsed() * 4 / 5);
    try model.restoreCheckpoint(snap8, q8);
    const seq11 = try gpa.alloc(u32, n_gen);
    defer gpa.free(seq11);
    cur = prompt[prompt.len - 1];
    for (seq11) |*t| {
        cur = try model.stepArgmax(io, &.{cur});
        t.* = cur;
    }
    try model.restoreCheckpoint(snap8, q8);
    const seq12 = try gpa.alloc(u32, n_gen);
    defer gpa.free(seq12);
    cur = prompt[prompt.len - 1];
    for (seq12) |*t| {
        cur = try model.stepArgmax(io, &.{cur});
        t.* = cur;
    }
    try std.testing.expectEqualSlices(u32, seq11, seq12);
}

// The initial split plan must respect the card's LIVE free VRAM, not just the
// abstract budget: with most of the card occupied (another process, a resident
// image model), a generous budget must still plan layers onto the host —
// planning them all resident faults at the first prefill instead (weight
// uploads + lazy PTX JIT collide at zero free). Mirrors the gemma3_cuda test.
test "cpu split plan respects live free VRAM" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const test_gate = @import("../test_gate.zig");
    const Gguf = @import("tp_core").gguf.Gguf;
    const path = "/home/qt/genai/lmstudio/models/gemma-4-12b-it-qat-q4_0.gguf";
    try test_gate.requireModelFile(io, path);
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    var g = try Gguf.open(gpa, io, path);
    defer g.deinit();
    var lm = try gemma4.Model.load(gpa, &g);
    defer lm.deinit();
    var model = try CudaLM.init(gpa, be, &lm, .{ .initial = 64, .max = 256 });
    defer model.deinit();

    // Leave ~3 GiB free: far less than the checkpoint's layer weights, so a
    // live-aware plan MUST place some layers on the host.
    var balloon = try test_gate.VramBalloon.inflateToFree(gpa, be, 3 << 30);
    defer balloon.deinit();

    try model.enableCpuSplit(.attn, 1 << 40, true); // budget far beyond the card
    errdefer std.debug.print("n_cpu={d}, free={d} MiB\n", .{
        if (model.split) |sp| sp.n_cpu else 0, be.ctx.memGetInfo().free >> 20,
    });
    try std.testing.expect(model.split != null);
    try std.testing.expect(model.split.?.n_cpu > 0);
}

// A hybrid split's host layers can run BEFORE any step(): the tp-gui turn flow
// prefills first, and an over-budget model has host layers from init. With the
// stepper's `io` unseeded that must fail closed (error.SplitIoUnset), and work
// once the session owner seeds it (gui/chat.zig Session.init does).
test "cpu split prefill before any step needs a seeded io" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const test_gate = @import("../test_gate.zig");
    const Gguf = @import("tp_core").gguf.Gguf;
    const path = "/home/qt/genai/lmstudio/models/gemma-4-12b-it-qat-q4_0.gguf";
    try test_gate.requireModelFile(io, path);
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    var g = try Gguf.open(gpa, io, path);
    defer g.deinit();
    var lm = try gemma4.Model.load(gpa, &g);
    defer lm.deinit();
    var model = try CudaLM.init(gpa, be, &lm, .{ .initial = 128, .max = 256 });
    defer model.deinit();

    // A budget below the weight total statically places tail layers on the
    // host right away (kept small so the Debug host prefill stays quick).
    try model.enableCpuSplit(.tail, 5 << 30, true);
    try std.testing.expect(model.split.?.n_cpu > 0);

    var prompt: [24]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(1000 + (i * 37) % 50000);
    try std.testing.expectError(error.SplitIoUnset, model.prefill(prompt[0 .. prompt.len - 1]));
    try std.testing.expectEqual(@as(usize, 0), model.cached());

    model.io = io; // the session owner's contract (tp-gui seeds this at init)
    try model.prefill(prompt[0 .. prompt.len - 1]);
    try std.testing.expectEqual(prompt.len - 1, model.cached());
    const next = try model.stepArgmax(io, prompt[prompt.len - 1 ..]);
    try std.testing.expect(next < lm.cfg.vocab);
}
