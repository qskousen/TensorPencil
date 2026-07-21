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

/// KV chunks per head in the decode attention split; the smaller prefill
/// count bounds the flash-decode scratch across a whole chunk of queries.
const nsplit = 32;
const nsplit_prefill = 8;
/// Rows per batched-prefill chunk (also the activation-buffer height).
const prefill_chunk = 128;
/// Batches up to this take the grouped dp4a GEMV (weight streamed ceil(n/8)x)
/// instead of opMatmulQuant's dequant-to-f16 GEMM (qwen3_cuda's crossover).
const grouped_gemv_max = 40;

/// Gemma 3 vision is always a 16x16 = 256-token soft-image block. A
/// bidirectional image block is prefilled in ONE batch so its tokens see each
/// other, so it (not prefill_chunk) sets the largest single batch: the LOCAL
/// ring slack and (rounded up) the activation-buffer height.
const max_image_tokens = 256;
/// Largest actual batch of rows fed in one forward (text chunk or image block).
const max_batch = @max(prefill_chunk, max_image_tokens);
/// Activation-buffer height. opMatmulQuant pads its output rows up to a
/// multiple of 128, so GEMM-output buffers must reserve the padded height.
const buf_rows = std.mem.alignForward(usize, max_batch, 128);

/// Rows a LOCAL (sliding-window) layer's KV ring holds: window + one max-batch
/// of slack (see gemma4_cuda for the aliasing rationale). LOCAL caches are
/// fixed at this size and never grow (TODO lever 1).
fn localRingRows(cfg: gemma3.Config) usize {
    return cfg.sliding_window + max_batch;
}

/// Kill-switch for the LOCAL-layer sliding-window ring cache (A/B validation).
const enable_local_ring = true;

/// This layer uses the sliding-window ring: a LOCAL layer, ring enabled.
fn usesRing(cfg: gemma3.Config, l: usize) bool {
    return enable_local_ring and !cfg.isGlobal(l);
}

/// One contiguous copy between a ring buffer (row = pos%ring) and a linear
/// host buffer (row = absolute pos). `abs` is the absolute position of the
/// first row, `dev` the ring row it maps to, `n` the row count.
const RingSeg = struct { abs: usize, dev: usize, n: usize };

/// The (up to two) segments covering the live positions `[max(0,len-ring), len)`
/// of a ring, split where the ring wraps. Used to translate a LOCAL layer's KV
/// between its device ring and the linear host shadow on offload/promote.
fn ringSegments(len: usize, ring: usize) [2]RingSeg {
    const start = if (len > ring) len - ring else 0;
    const total = len - start; // <= ring
    const first_dev = start % ring;
    const n1 = @min(total, ring - first_dev);
    return .{
        .{ .abs = start, .dev = first_dev, .n = n1 },
        .{ .abs = start + n1, .dev = 0, .n = total - n1 },
    };
}

fn nbuf(be: *Backend, weights: []const f32) !Buf {
    return .{ .buf = try be.smallBuffer(std.mem.sliceAsBytes(weights)), .mem = .null_handle, .size = 0 };
}

fn offsetBufSized(b: Buf, off_bytes: usize, size: usize) Buf {
    return .{ .buf = @enumFromInt(@intFromEnum(b.buf) + off_bytes), .mem = b.mem, .size = size };
}

/// Which layers a hybrid CPU/GPU split pushes to the host. Gemma3 is uniform
/// attention, so both policies reduce to a descending-layer order (last layer
/// migrates first); kept for interface parity with qwen35_cuda.
pub const CpuSplitPolicy = enum { tail, attn };

pub const CudaLM = struct {
    lm: *const gemma3.Model,
    be: *Backend,
    gpa: std.mem.Allocator,
    cfg: gemma3.Config,
    capacity: usize,
    max_capacity: usize,
    /// The KV capacity a fresh session starts at; resetResidency shrinks back to
    /// it so a new chat frees the grown KV VRAM.
    initial_capacity: usize,
    /// KV-cache element storage type (f32 / f16); selects the attention kernel
    /// variant and the per-element stride of k_cache/v_cache.
    kv_dtype: kvmod.KvDtype,
    len: usize,
    /// Set for the duration of a bidirectional image-block prefill: the
    /// `attention` stepper (device + host-split) reads it and lets every query
    /// in the batch attend the whole block forward (image spans are non-causal).
    bidir_prefill: bool = false,
    /// Io for the host matmuls of a hybrid split's CPU-resident layers; set
    /// by the step entry points, or seeded by the session owner BEFORE the
    /// first forward (tp-gui prefills before any step, and an over-budget
    /// model has host layers from init). null fails the host path closed
    /// (error.SplitIoUnset) instead of undefined-pointer UB.
    io: ?std.Io = null,
    /// Hybrid CPU/GPU split: CPU-resident layers run host matmuls (dynamic
    /// offload migrates more to the host as the KV cache grows). null = fully
    /// device-resident. Forces the per-op path (already the gemma3 default).
    split: ?Split = null,
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

    /// CPU-resident layers of a hybrid split + the host state they need.
    /// Allocated with `gpa`; freed in `deinit`. Mirrors qwen35_cuda.Split,
    /// simplified: gemma3 is all-attention (no conv/ssm), so the host state is
    /// just a KvCache (K/V for the CPU layers) plus the dual RoPE tables.
    pub const Split = struct {
        /// Per-layer: compute on the device? (false = host).
        on_gpu: []bool,
        n_cpu: usize,
        policy: CpuSplitPolicy,
        /// Host K/V for the CPU-resident layers (full n_layers slots; only the
        /// CPU layers' slots are used). Grows in lockstep with the device caches;
        /// `len` tracks the device `len`.
        cache: kvmod.KvCache,
        /// Host activation scratch, sized for a full prefill chunk (viewed down
        /// to the actual chunk length per call).
        scratch: gemma3.Scratch,
        /// Host RoPE tables (global + local), grown with capacity — text only.
        freqs_global: ops.rope.Freqs,
        freqs_local: ops.rope.Freqs,
        /// Host hidden buffer ([prefill_chunk * hidden]).
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
        self.initial_capacity = cap.initial;
        self.max_capacity = cap.max;
        self.kv_dtype = cap.kv_dtype;
        self.len = 0;
        self.split = null;
        self.io = null; // field defaults do not apply to `undefined`-built structs
        self.sin_off = cap.max * (cfg.head_dim / 2);

        self.freqs_global = try uploadFreqs(be, gpa, cap.max, cfg.head_dim, cfg.rope_theta, cfg.rope_freq_scale);
        errdefer be.tensorDestroy(&self.freqs_global);
        self.freqs_local = try uploadFreqs(be, gpa, cap.max, cfg.head_dim, cfg.rope_theta_local, 1.0);
        errdefer be.tensorDestroy(&self.freqs_local);

        self.bufs = try Bufs.init(be, cfg);

        self.k_cache = try alloc.alloc(Growable, cfg.n_layers);
        self.v_cache = try alloc.alloc(Growable, cfg.n_layers);
        try self.allocKvCaches();

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

    /// (Re)create the per-layer K/V device buffers at `self.kv_dtype`, sized
    /// from `self.initial_capacity`/`self.max_capacity` (LOCAL layers = fixed
    /// sliding-window ring). The `k_cache`/`v_cache` slices must already exist.
    /// Host-resident layers of an armed split are skipped — they keep no
    /// device KV (their shadow lives in `split.cache`).
    fn allocKvCaches(self: *CudaLM) !void {
        const cfg = self.cfg;
        const be = self.be;
        const dt = self.kv_dtype;
        for (self.k_cache, self.v_cache, 0..) |*kb, *vb, l| {
            if (self.split) |*sp| if (!sp.on_gpu[l]) continue;
            if (usesRing(cfg, l)) {
                const bytes = dt.sizeBytes(localRingRows(cfg) * cfg.kvDim());
                kb.* = try be.growableCreate(bytes, bytes);
                vb.* = try be.growableCreate(bytes, bytes);
            } else {
                kb.* = try be.growableCreate(dt.sizeBytes(self.initial_capacity * cfg.kvDim()), dt.sizeBytes(self.max_capacity * cfg.kvDim()));
                vb.* = try be.growableCreate(dt.sizeBytes(self.initial_capacity * cfg.kvDim()), dt.sizeBytes(self.max_capacity * cfg.kvDim()));
            }
        }
    }

    /// Store `n` K/V elements from `src` (+`src_off` elems) into cache buffer
    /// `dst` at element offset `dst_off`. f32 copies raw; f16/q8_0 convert on
    /// store.
    fn storeKv(self: *CudaLM, dst: Buf, dst_off: usize, src: Buf, src_off: usize, n: usize) !void {
        switch (self.kv_dtype) {
            .f16 => try self.be.opStoreKvF16(dst, dst_off, src, src_off, n),
            .q8_0 => try self.be.opStoreKvQ8(dst, dst_off, src, src_off, n),
            .f32 => try self.be.tensorCopy(dst, dst_off * 4, src, src_off * 4, n * 4),
        }
    }

    /// Rebuild the KV cache at a new element dtype (GUI toggle), weights resident.
    /// gemma3 decodes straight through `opAttnDecode` (no captured graph), so this
    /// just frees + re-creates the K/V buffers (and a split's host shadow, which
    /// stores the same dtype as the device caches) and resets the length.
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
            const cache = try kvmod.KvCache.init(self.gpa, self.cfg.n_layers, self.capacity, self.cfg.kvDim(), dtype);
            sp.cache.deinit(self.gpa);
            sp.cache = cache;
        }
    }

    pub fn deinit(self: *CudaLM) void {
        const be = self.be;
        for (self.k_cache) |*b| be.growableDestroy(b);
        for (self.v_cache) |*b| be.growableDestroy(b);
        be.tensorDestroy(&self.freqs_global);
        be.tensorDestroy(&self.freqs_local);
        self.bufs.deinit(be);
        if (self.split) |*sp| {
            sp.cache.deinit(self.gpa);
            sp.scratch.deinit(self.gpa);
            sp.freqs_global.deinit(self.gpa);
            sp.freqs_local.deinit(self.gpa);
            self.gpa.free(sp.hx);
            self.gpa.free(sp.on_gpu);
            self.gpa.free(sp.order);
        }
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
    /// Device VRAM (bytes) currently in use by this backend — for the
    /// end-of-response telemetry (`session.statsOf`).
    pub fn vramUsed(self: *const CudaLM) u64 {
        return self.be.deviceUsed();
    }
    /// Reset the session to an empty context (GUI "new chat"). Gemma3 is a plain
    /// attention model with no recurrent/conv state, so the KV rows (overwritten
    /// lazily on the next prefill) only need the position counter cleared.
    pub fn resetCache(self: *CudaLM) !void {
        self.len = 0;
    }

    /// Fixed byte size of a turn checkpoint (see `checkpoint`): every LOCAL
    /// (sliding-window) layer's full KV ring. GLOBAL layers are append-only, so
    /// truncation alone rolls them back — but a LOCAL ring OVERWRITES its
    /// oldest rows as generation advances, so a rollback further than the ring
    /// slack needs them restored. Context-independent (rings are fixed-size).
    pub fn checkpointBytes(self: *const CudaLM) usize {
        const cfg = self.cfg;
        const ring_bytes = self.kv_dtype.sizeBytes(localRingRows(cfg) * cfg.kvDim());
        var total: usize = 0;
        for (0..cfg.n_layers) |l| {
            if (usesRing(cfg, l)) total += 2 * ring_bytes;
        }
        return total;
    }

    /// Snapshot the non-append-only context state at the current position into
    /// `out` (`checkpointBytes` long): each LOCAL layer's ring, in ring-row
    /// format, read from whichever side currently owns the layer (a linear
    /// host shadow is translated through the same ring mapping migrate/promote
    /// use). Owner-agnostic, so layers may migrate between snapshot and
    /// restore. Pair with `restoreCheckpoint(out, q)` where `q == cached()`.
    pub fn checkpoint(self: *CudaLM, out: []u8) !void {
        std.debug.assert(out.len == self.checkpointBytes());
        try self.checkpointRings(.save, out);
    }

    /// Roll the context back to `q` committed tokens (a turn boundary) using a
    /// snapshot taken there: GLOBAL layers just truncate (rows past `q` are
    /// overwritten by the next write, on the device growables and the split's
    /// linear host shadow alike); LOCAL rings are restored wholesale into each
    /// layer's CURRENT owner, bringing back the rows the discarded response
    /// overwrote.
    pub fn restoreCheckpoint(self: *CudaLM, snap: []const u8, q: usize) !void {
        std.debug.assert(snap.len == self.checkpointBytes());
        std.debug.assert(q <= self.len);
        // Truncate FIRST: the host-linear ring translation below maps the
        // snapshot's live window, which is defined by q.
        self.len = q;
        if (self.split) |*sp| sp.cache.truncate(q);
        try self.checkpointRings(.restore, snap);
    }

    const CheckpointDir = enum { save, restore };

    /// Shared body of checkpoint/restoreCheckpoint: move every LOCAL layer's
    /// ring between the snapshot buffer (ring-row format, K then V per layer)
    /// and the layer's current owner. Device-resident rings copy wholesale;
    /// a host-resident layer's linear shadow is translated per live segment
    /// (`ringSegments(self.len, ring)` — the caller ensures `self.len` is the
    /// snapshot's q in both directions).
    fn checkpointRings(self: *CudaLM, comptime dir: CheckpointDir, buf: if (dir == .save) []u8 else []const u8) !void {
        const cfg = self.cfg;
        const kvd = cfg.kvDim();
        const ring = localRingRows(cfg);
        const dt = self.kv_dtype;
        const ring_bytes = dt.sizeBytes(ring * kvd);
        var off: usize = 0;
        for (0..cfg.n_layers) |l| {
            if (!usesRing(cfg, l)) continue;
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
                // Host-resident: the linear shadow stores the same dtype as
                // the device rings (kRowBytes is the device byte layout), so
                // the snapshot's ring-row format is a raw copy either way.
                const sp = &self.split.?;
                for (ringSegments(self.len, ring)) |seg| {
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
            off += 2 * ring_bytes;
        }
    }
    pub fn vocab(self: *const CudaLM) usize {
        return self.cfg.vocab;
    }

    /// Total device footprint of one layer's streamable weights (quantized
    /// bytes) — the projection + MLP matrices; norms are negligible. `anytype`
    /// avoids naming gemma3's private `Layer` type.
    fn layerDeviceBytes(layer: anytype) usize {
        return layer.q.bytes.len + layer.k.bytes.len + layer.v.bytes.len + layer.o.bytes.len +
            layer.gate.bytes.len + layer.up.bytes.len + layer.down.bytes.len;
    }

    /// Layers whose KV still lives on the device (gemma3 is all-attention, so
    /// this is just the on-GPU count).
    fn liveSlots(self: *CudaLM) usize {
        var count: usize = 0;
        for (0..self.cfg.n_layers) |l| {
            if (self.split) |*sp| if (!sp.on_gpu[l]) continue;
            count += 1;
        }
        return count;
    }

    /// Move layer `l`'s live K/V device->host, free its device K/V + weights,
    /// and mark it CPU-resident (a `residency` hook). Mirrors qwen35_cuda's
    /// migrateLayer (attention case only — gemma3 has no recurrent/conv state).
    pub fn migrateLayer(self: *CudaLM, l: usize) !void {
        const cfg = self.cfg;
        const sp = &self.split.?;
        const kvd = cfg.kvDim();
        // The host shadow stores the same dtype as the device caches
        // (kRowBytes is the device byte layout), so every copy is raw.
        const dt = self.kv_dtype;
        if (self.len > 0) {
            if (usesRing(cfg, l)) {
                // LOCAL ring -> linear host shadow: copy the live window,
                // translating ring rows (pos%ring) to absolute host rows.
                for (ringSegments(self.len, localRingRows(cfg))) |seg| {
                    if (seg.n == 0) continue;
                    try self.be.tensorDownload(offsetBufSized(self.k_cache[l].buf, dt.sizeBytes(seg.dev * kvd), dt.sizeBytes(seg.n * kvd)), sp.cache.kRowBytes(l, seg.abs, seg.n));
                    try self.be.tensorDownload(offsetBufSized(self.v_cache[l].buf, dt.sizeBytes(seg.dev * kvd), dt.sizeBytes(seg.n * kvd)), sp.cache.vRowBytes(l, seg.abs, seg.n));
                }
            } else {
                try self.be.tensorDownload(offsetBufSized(self.k_cache[l].buf, 0, dt.sizeBytes(self.len * kvd)), sp.cache.kRowBytes(l, 0, self.len));
                try self.be.tensorDownload(offsetBufSized(self.v_cache[l].buf, 0, dt.sizeBytes(self.len * kvd)), sp.cache.vRowBytes(l, 0, self.len));
            }
        }
        self.be.growableDestroy(&self.k_cache[l]);
        self.be.growableDestroy(&self.v_cache[l]);
        // Free the migrated layer's device weights (the host path reads them from
        // the GGUF mapping) — the bulk of the reclaimed VRAM.
        const layer = &self.lm.layers[l];
        self.be.evictWeightBytes(layer.q.bytes);
        self.be.evictWeightBytes(layer.k.bytes);
        self.be.evictWeightBytes(layer.v.bytes);
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
    /// iteration as liveSlots drops — a fixed target here can't express that.)
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
    /// streamable weights, the KV it re-commits at the current capacity (all
    /// gemma3 layers are attention), plus slack.
    pub fn promoteCost(self: *CudaLM, l: usize) usize {
        const kv_at_cap = 2 * self.kv_dtype.sizeBytes(self.capacity * self.cfg.kvDim());
        return layerDeviceBytes(&self.lm.layers[l]) + kv_at_cap + (64 << 20);
    }

    /// Bring layer `l` back onto the GPU, preserving its accumulated K/V: re-create
    /// the device K/V at the current capacity and upload the host rows [0,len).
    /// Weights re-cache lazily on the next GPU forward. Reverse of migrateLayer.
    pub fn promoteLayer(self: *CudaLM, l: usize) !void {
        const cfg = self.cfg;
        const sp = &self.split.?;
        const kvd = cfg.kvDim();
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
            if (usesRing(cfg, l)) {
                // Linear host shadow -> LOCAL ring: reverse of migrateLayer.
                for (ringSegments(self.len, localRingRows(cfg))) |seg| {
                    if (seg.n == 0) continue;
                    try self.be.tensorUpload(offsetBufSized(self.k_cache[l].buf, dt.sizeBytes(seg.dev * kvd), dt.sizeBytes(seg.n * kvd)), sp.cache.kRowBytes(l, seg.abs, seg.n));
                    try self.be.tensorUpload(offsetBufSized(self.v_cache[l].buf, dt.sizeBytes(seg.dev * kvd), dt.sizeBytes(seg.n * kvd)), sp.cache.vRowBytes(l, seg.abs, seg.n));
                }
            } else {
                try self.be.tensorUpload(offsetBufSized(self.k_cache[l].buf, 0, dt.sizeBytes(self.len * kvd)), sp.cache.kRowBytes(l, 0, self.len));
                try self.be.tensorUpload(offsetBufSized(self.v_cache[l].buf, 0, dt.sizeBytes(self.len * kvd)), sp.cache.vRowBytes(l, 0, self.len));
            }
        }
        sp.on_gpu[l] = true;
        sp.n_cpu -= 1;
        std.log.debug("[promote] layer {d} -> GPU at ctx {d} ({d}/{d} on CPU)", .{ l, self.len, sp.n_cpu, cfg.n_layers });
    }

    /// Migrate CPU layers back onto the GPU (LIFO by offload order), stopping
    /// before the next one would overflow `budget` — so the caller (VRAM
    /// coordinator, after image generation) reclaims LLM residency while leaving
    /// room for whatever else stays resident. Keeps the split armed (offload can
    /// fire again). Returns the number promoted; 0 without a split.
    pub fn promoteLayers(self: *CudaLM, budget: u64) !usize {
        return residency.promoteBack(self, budget);
    }

    /// New-chat reset: drop the split, shrink every K/V cache back to the initial
    /// capacity (frees the grown VRAM), clear the context, and re-arm dynamic
    /// offload for the fresh small context. KV is discarded, so no host->device
    /// copy is needed (unlike promoteLayers).
    pub fn resetResidency(self: *CudaLM, budget: u64) !void {
        if (self.split) |*sp| {
            sp.cache.deinit(self.gpa);
            sp.scratch.deinit(self.gpa);
            sp.freqs_global.deinit(self.gpa);
            sp.freqs_local.deinit(self.gpa);
            self.gpa.free(sp.hx);
            self.gpa.free(sp.on_gpu);
            self.gpa.free(sp.order);
            self.split = null;
        }
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
    /// `budget` (bytes). Must run right after init (before any tokens) with
    /// `budget != 0`. `dynamic` packs the GPU now (head-only reserve) and
    /// migrates on demand as the KV grows; static reserves generously. Forces
    /// the per-op path (already the gemma3 default). No-op static split (all fit)
    /// leaves `self.split == null`. Mirrors qwen35_cuda.enableCpuSplit.
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

        // Device memory that must stay resident: KV + LM head + slack. LOCAL ring
        // layers hold only localRingRows, not the full capacity.
        var kv_rows: usize = 0;
        for (0..n) |l| kv_rows += if (usesRing(cfg, l)) localRingRows(cfg) else self.capacity;
        const kv_bytes = 2 * self.kv_dtype.sizeBytes(kv_rows * cfg.kvDim());
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

        // Eviction order: last layer leaves first (descending). gemma3 is
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
        // stores the SAME dtype as the device caches, so migrate/promote and
        // the ring checkpoint translation are raw byte copies (kRowBytes).
        var cache = try kvmod.KvCache.init(gpa, n, self.capacity, cfg.kvDim(), self.kv_dtype);
        errdefer cache.deinit(gpa);
        // The host shadow tracks the SAME committed length as the device from
        // the moment the split arms (per-step commits keep them in lockstep
        // afterwards). Armed mid-conversation (imageReclaim), starting at 0
        // would make host layers attend over nothing; migrateLayer copies each
        // migrated layer's live rows so declaring them committed is correct.
        cache.len = self.len;
        var scratch = try gemma3.Scratch.init(gpa, max_batch, cfg);
        errdefer scratch.deinit(gpa);
        var fg = try ops.rope.rotateHalfFreqsScaled(gpa, self.capacity, cfg.head_dim, cfg.rope_theta, cfg.rope_freq_scale);
        errdefer fg.deinit(gpa);
        var fl = try ops.rope.rotateHalfFreqsScaled(gpa, self.capacity, cfg.head_dim, cfg.rope_theta_local, 1.0);
        errdefer fl.deinit(gpa);
        const hx = try gpa.alloc(f32, max_batch * cfg.hidden);
        errdefer gpa.free(hx);

        self.split = .{
            .on_gpu = on_gpu,
            .n_cpu = 0, // the placement loops below mark + count the host layers
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

    pub fn ensureCapacity(self: *CudaLM, min_rows: usize) !void {
        const target = (try kvmod.growPlan(self.capacity, self.max_capacity, min_rows)) orelse return;
        // Byte size MUST match how the buffers were created (kv_dtype block
        // math), or an f16/q8_0 cache requests f32-sized growth, overshoots its
        // VA reservation, and growableEnsure fails with DeviceOutOfMemory →
        // ContextFull once the window grows past ~max_capacity/2.
        const bytes = self.kv_dtype.sizeBytes(target * self.cfg.kvDim());

        // Dynamic offload: migrate layers GPU->CPU until there's headroom to grow
        // the device KV, instead of streaming weights (the cliff). Each migrated
        // layer frees its device KV + weight VRAM. Mirrors qwen35_cuda.
        if (self.split) |*sp| if (sp.dynamic) {
            const add = self.kv_dtype.sizeBytes((target - self.capacity) * self.cfg.kvDim());
            while (true) {
                const need = self.liveSlots() * 2 * add + (32 << 20); // + margin
                const free = @min(sp.budget -| self.be.deviceUsed(), self.be.headroom());
                if (free >= need) break;
                if (!(try residency.migrateNext(self))) break; // nothing left; fall through
            }
        };

        // Grow device KV of the layers still on the GPU (LOCAL ring layers are
        // fixed-size — never grow them). Physical VRAM can be exhausted even when
        // the proactive migration above thought there was room: a resident image
        // model on another CUDA context may grab it between the headroom check and
        // this commit. On a real OOM, offload one more layer to the CPU and retry
        // the whole grow, so a full window only ever fails once nothing is left to
        // migrate — never a premature ContextFull while layers can still move to
        // host. growableEnsure is idempotent, so re-running the loop is cheap.
        grow: while (true) {
            for (0..self.cfg.n_layers) |l| {
                if (usesRing(self.cfg, l)) continue;
                if (self.split) |*sp| if (!sp.on_gpu[l]) continue;
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
            const cfg = self.cfg;
            const fg = ops.rope.rotateHalfFreqsScaled(self.gpa, target, cfg.head_dim, cfg.rope_theta, cfg.rope_freq_scale) catch return error.ContextFull;
            sp.freqs_global.deinit(self.gpa);
            sp.freqs_global = fg;
            const fl = ops.rope.rotateHalfFreqsScaled(self.gpa, target, cfg.head_dim, cfg.rope_theta_local, 1.0) catch return error.ContextFull;
            sp.freqs_local.deinit(self.gpa);
            sp.freqs_local = fl;
        }
        self.capacity = target;
    }

    /// Forward `ids` at positions [len, len+ids.len); write last-position
    /// vocab logits. Prefill runs in prefill_chunk-sized batches (only the
    /// final chunk computes the LM head).
    pub fn step(self: *CudaLM, io: std.Io, ids: []const u32, logits: []f32) !void {
        self.io = io; // the CPU half of a hybrid split runs host matmuls through it
        std.debug.assert(ids.len >= 1 and ids.len <= self.remaining());
        std.debug.assert(logits.len == self.cfg.vocab);
        try self.forwardDecode(io, ids, logits);
    }

    /// The decode/prefill forward in prefill_chunk batches; the final chunk runs
    /// the LM head into bufs.logits, downloading to `dl` when non-null. Shared by
    /// step (downloads), stepArgmax and stepSelect (sample on-device).
    fn forwardDecode(self: *CudaLM, io: std.Io, ids: []const u32, dl: ?[]f32) !void {
        self.io = io; // the CPU half of a hybrid split runs host matmuls through it
        std.debug.assert(ids.len >= 1 and ids.len <= self.remaining());
        var off: usize = 0;
        while (off < ids.len) {
            const n: usize = @min(prefill_chunk, ids.len - off);
            const last = (off + n == ids.len);
            try self.embedChunk(ids[off..][0..n], last, if (last) dl else null);
            off += n;
        }
    }

    /// Greedy decode: forward, then argmax the last logits on-device, returning
    /// just the id. Matches sample.argmax.
    pub fn stepArgmax(self: *CudaLM, io: std.Io, ids: []const u32) !u32 {
        return self.stepArgmaxPen(io, ids, &.{}, .{});
    }

    /// `stepArgmax` with sampling penalties scattered onto the device logits
    /// first (opPenalize; see sample.zig) — keeps penalized greedy decode
    /// on the GPU path instead of the full-vocab download.
    pub fn stepArgmaxPen(self: *CudaLM, io: std.Io, ids: []const u32, pen: []const sample.PenaltyEntry, sp: sample.Params) !u32 {
        try self.forwardDecode(io, ids, null);
        const be = self.be;
        const b = &self.bufs;
        try be.opPenalize(offsetBufSized(b.logits, 0, self.cfg.vocab * 4), pen, sp);
        try be.opArgmax(offsetBufSized(b.logits, 0, self.cfg.vocab * 4), self.cfg.vocab, b.argmax_out, &b.argmax_v, &b.argmax_i);
        var idf: [1]f32 = undefined;
        try be.tensorDownload(b.argmax_out, std.mem.sliceAsBytes(&idf));
        return @intFromFloat(idf[0]);
    }

    /// Max candidates stepSelect can return (host buffer sizing for the engine).
    pub fn maxSelect(self: *const CudaLM) usize {
        _ = self;
        return cuda.backend.topk_lanes * cuda.backend.topk_m;
    }

    /// Stochastic decode: forward, select the top-k on-device, download just
    /// those (id,logit) pairs. Returns the candidate count.
    pub fn stepSelect(self: *CudaLM, io: std.Io, ids: []const u32, out_id: []u32, out_logit: []f32) !usize {
        return self.stepSelectPen(io, ids, &.{}, .{}, out_id, out_logit);
    }

    /// `stepSelect` with sampling penalties scattered onto the device logits
    /// before the top-k (opPenalize) — the selected candidates are the true
    /// post-penalty top set, so penalized stochastic decode stays on the GPU.
    pub fn stepSelectPen(self: *CudaLM, io: std.Io, ids: []const u32, pen: []const sample.PenaltyEntry, sp: sample.Params, out_id: []u32, out_logit: []f32) !usize {
        try self.forwardDecode(io, ids, null);
        const be = self.be;
        const b = &self.bufs;
        try be.opPenalize(offsetBufSized(b.logits, 0, self.cfg.vocab * 4), pen, sp);
        const count = try be.opTopK(offsetBufSized(b.logits, 0, self.cfg.vocab * 4), self.cfg.vocab, &b.topk_v, &b.topk_i);
        std.debug.assert(count <= out_id.len and count <= out_logit.len);
        try be.tensorDownload(b.topk_v, std.mem.sliceAsBytes(out_logit[0..count]));
        var idx_f: [cuda.backend.topk_lanes * cuda.backend.topk_m]f32 = undefined;
        try be.tensorDownload(b.topk_i, std.mem.sliceAsBytes(idx_f[0..count]));
        for (out_id[0..count], idx_f[0..count]) |*o, f| o.* = @intFromFloat(f);
        return count;
    }

    /// Prefill text tokens (no logits) — for interleaving with prefillImage.
    pub fn prefill(self: *CudaLM, ids: []const u32) !void {
        var off: usize = 0;
        while (off < ids.len) {
            const n: usize = @min(prefill_chunk, ids.len - off);
            try self.embedChunk(ids[off..][0..n], false, null);
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
        // A bidirectional image block must be one batch (a later chunk's KV is
        // not committed when an earlier chunk runs); it fits in max_batch rows.
        std.debug.assert(total <= max_batch);
        self.bidir_prefill = true;
        defer self.bidir_prefill = false;
        try self.forwardRows(embeds, false, null);
    }

    /// Embed `ids` (gather + sqrt(hidden) scale, host-side) then forward the
    /// resulting rows.
    fn embedChunk(self: *CudaLM, ids: []const u32, want_head: bool, dl: ?[]f32) !void {
        const cfg = self.cfg;
        const n = ids.len;
        const x = try self.gpa.alloc(f32, n * cfg.hidden);
        defer self.gpa.free(x);
        try qwen3.embedTokens(self.lm.embed, ids, x);
        const scale = cfg.embedScale();
        for (x) |*v| v.* *= scale;
        try self.forwardRows(x, want_head, dl);
    }

    /// One batched forward over `n` pre-embedded input rows `x_host`
    /// ([n*hidden]) at positions [len, len+n). When `logits` is set, the last
    /// row's final-normed hidden feeds the LM head. Advances len by n.
    // --- transformer_gpu.decoderLayer stepper methods (faithful lift of the
    // former forwardRows device path; the hybrid CPU-split stays a loop-top
    // hook). ---

    pub fn normInput(self: *CudaLM, layer: anytype, seq: usize) !void {
        const cfg = self.cfg;
        try self.be.qkNorm(self.bufs.x, self.bufs.normed, try nbuf(self.be, layer.input_norm), seq, cfg.hidden, cfg.rms_eps);
    }
    pub fn projectQKV(self: *CudaLM, l: usize, layer: anytype, seq: usize) !void {
        _ = l; // gemma3: uniform geometry
        const cfg = self.cfg;
        const b = &self.bufs;
        try self.linear(b.q, b.normed, layer.q, cfg.qDim(), cfg.hidden, seq);
        try self.linear(b.k, b.normed, layer.k, cfg.kvDim(), cfg.hidden, seq);
        try self.linear(b.v, b.normed, layer.v, cfg.kvDim(), cfg.hidden, seq);
    }
    pub fn normQK(self: *CudaLM, l: usize, layer: anytype, seq: usize) !void {
        _ = l;
        const cfg = self.cfg;
        const b = &self.bufs;
        try self.be.qkNorm(b.q, b.q, try nbuf(self.be, layer.q_norm), seq * cfg.n_heads, cfg.head_dim, cfg.rms_eps);
        try self.be.qkNorm(b.k, b.k, try nbuf(self.be, layer.k_norm), seq * cfg.n_kv_heads, cfg.head_dim, cfg.rms_eps);
    }
    pub fn applyRope(self: *CudaLM, l: usize, seq: usize, pos0: usize) !void {
        const cfg = self.cfg;
        const b = &self.bufs;
        const freqs = if (cfg.isGlobal(l)) self.freqs_global else self.freqs_local;
        try self.be.ropeHalf(b.q, freqs, seq, cfg.n_heads, cfg.head_dim / 2, self.sin_off, pos0);
        try self.be.ropeHalf(b.k, freqs, seq, cfg.n_kv_heads, cfg.head_dim / 2, self.sin_off, pos0);
    }
    pub fn appendKV(self: *CudaLM, l: usize, seq: usize, pos0: usize) !void {
        const cfg = self.cfg;
        const b = &self.bufs;
        const kvd = cfg.kvDim();
        if (!usesRing(cfg, l)) {
            try self.storeKv(self.k_cache[l].buf, pos0 * kvd, b.k, 0, seq * kvd);
            try self.storeKv(self.v_cache[l].buf, pos0 * kvd, b.v, 0, seq * kvd);
            return;
        }
        // LOCAL layer: write into the ring at row pos0%ring, splitting on wrap
        // (seq <= max_batch < ring, so at most one wrap).
        const ring = localRingRows(cfg);
        const start = pos0 % ring;
        const first = @min(seq, ring - start);
        try self.storeKv(self.k_cache[l].buf, start * kvd, b.k, 0, first * kvd);
        try self.storeKv(self.v_cache[l].buf, start * kvd, b.v, 0, first * kvd);
        if (first < seq) {
            const rest = seq - first;
            try self.storeKv(self.k_cache[l].buf, 0, b.k, first * kvd, rest * kvd);
            try self.storeKv(self.v_cache[l].buf, 0, b.v, first * kvd, rest * kvd);
        }
    }
    pub fn attention(self: *CudaLM, l: usize, seq: usize, pos0: usize) !void {
        const cfg = self.cfg;
        const b = &self.bufs;
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));
        const window: usize = if (cfg.isGlobal(l)) 0 else cfg.sliding_window;
        const ring: usize = if (usesRing(cfg, l)) localRingRows(cfg) else 0;
        const ns: usize = if (seq == 1) nsplit else nsplit_prefill;
        try self.be.opAttnDecode(b.q, self.k_cache[l].buf, self.v_cache[l].buf, b.attn, b.attn_scratch, pos0 + 1, seq, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, ns, scale, window, ring, self.bidir_prefill, kvFmt(self.kv_dtype));
    }
    pub fn projectO(self: *CudaLM, l: usize, layer: anytype, seq: usize) !void {
        _ = l;
        const cfg = self.cfg;
        try self.linear(self.bufs.t, self.bufs.attn, layer.o, cfg.hidden, cfg.qDim(), seq);
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

    fn forwardRows(self: *CudaLM, x_host: []const f32, want_head: bool, dl: ?[]f32) !void {
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
            // shared gemma3.layerForward, ferrying the hidden across the
            // device<->host boundary only when residency changes.
            if (self.split) |*sp| {
                if (!sp.on_gpu[l]) {
                    if (!sp.on_host) {
                        try be.tensorDownload(offsetBufSized(b.x, 0, n * cfg.hidden * 4), std.mem.sliceAsBytes(sp.hx[0 .. n * cfg.hidden]));
                        sp.on_host = true;
                    }
                    const host_io = self.io orelse return error.SplitIoUnset;
                    var sv = sp.scratch.viewSeq(n, cfg);
                    try gemma3.layerForward(host_io, self.gpa, cfg, layer, sp.hx[0 .. n * cfg.hidden], n, sp.freqs_global, sp.freqs_local, &sp.cache, l, self.bidir_prefill, &sv);
                    continue;
                }
                if (sp.on_host) {
                    try be.tensorUpload(offsetBufSized(b.x, 0, n * cfg.hidden * 4), std.mem.sliceAsBytes(sp.hx[0 .. n * cfg.hidden]));
                    sp.on_host = false;
                }
            }
            try transformer_gpu.decoderLayer(transformer.gemma3_spec, self, layer, l, n, pos0);
        }

        // If the last layers ran on the host (the `.attn`/`.tail` order migrates
        // the last layers first, so this is the common case), bring the final
        // hidden back to the device for the LM head.
        if (self.split) |*sp| if (sp.on_host) {
            try be.tensorUpload(offsetBufSized(b.x, 0, n * cfg.hidden * 4), std.mem.sliceAsBytes(sp.hx[0 .. n * cfg.hidden]));
            sp.on_host = false;
        };

        if (want_head) {
            const h = cfg.hidden;
            try be.qkNorm(offsetBufSized(b.x, (n - 1) * h * 4, h * 4), b.t, try nbuf(be, self.lm.final_norm), 1, h, eps);
            try self.lmHead(b.logits, b.t);
            try be.endBatch();
            self.len += n;
            if (self.split) |*sp| sp.cache.commit(n); // keep host len == device len
            // `dl` null leaves logits resident (stepArgmax/stepSelect sample them).
            if (dl) |out| try be.tensorDownload(offsetBufSized(b.logits, 0, cfg.vocab * 4), std.mem.sliceAsBytes(out));
        } else {
            try be.endBatch();
            self.len += n;
            if (self.split) |*sp| sp.cache.commit(n);
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
    argmax_v: Buf,
    argmax_i: Buf,
    argmax_out: Buf,
    topk_v: Buf,
    topk_i: Buf,

    fn init(be: *Backend, cfg: gemma3.Config) !Bufs {
        // Height for the largest single batch (text chunk or whole image block),
        // rounded up to the GEMM's 128-row output padding.
        const pc = buf_rows;
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

test "ringSegments covers the live window contiguously" {
    // For each (len, ring), the segments must tile [max(0,len-ring), len)
    // exactly, mapping each absolute position to ring row pos%ring, with no
    // segment wrapping the ring boundary.
    const cases = [_][2]usize{
        .{ 5, 8 }, .{ 8, 8 }, .{ 9, 8 }, .{ 16, 8 }, .{ 17, 8 }, .{ 20, 8 },
        .{ 1000, 1152 }, .{ 1152, 1152 }, .{ 1153, 1152 }, .{ 3000, 1152 },
    };
    for (cases) |c| {
        const len = c[0];
        const ring = c[1];
        const start = if (len > ring) len - ring else 0;
        var abs = start;
        for (ringSegments(len, ring)) |s| {
            if (s.n == 0) continue;
            try std.testing.expectEqual(abs, s.abs); // contiguous, no gap
            try std.testing.expectEqual(abs % ring, s.dev); // ring row of abs
            try std.testing.expect(s.dev + s.n <= ring); // no wrap within a segment
            abs += s.n;
        }
        try std.testing.expectEqual(len, abs); // covered exactly up to len
    }
}

// Gated on -Dintegration + a CUDA device + the real 12B checkpoint: greedy
// regeneration after `restoreCheckpoint` must be TOKEN-IDENTICAL. The geometry
// deliberately exercises the SWA rings: the prompt exceeds the ring size (so
// the rings have wrapped) and the generation exceeds the ring slack (so it
// OVERWRITES live-window rows the rollback needs) — only the checkpoint's ring
// snapshot can bring those rows back; a bare `len` rollback would replay over
// destroyed keys. Also re-runs under a mid-conversation CPU split (host-linear
// ring translation + owner-aware restore), asserting repeatability there and
// bit-identity again after promoting back (hybrid CPU/GPU arithmetic isn't
// guaranteed bit-identical to all-GPU, so the split leg can't compare to seq1
// directly — mirrors the qwen35_cuda test).
fn checkpointRestoreBody(kv_dtype: kvmod.KvDtype) !void {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const test_gate = @import("../test_gate.zig");
    const Gguf = @import("tp_core").gguf.Gguf;
    const path = "/home/qt/genai/lmstudio/models/mradermacher/Gemma-3-Starshine-12B-Alt-GGUF/Gemma-3-Starshine-12B-Alt.Q4_K_M.gguf";
    try test_gate.requireModelFile(io, path);
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    var g = try Gguf.open(gpa, io, path);
    defer g.deinit();
    var lm = try gemma3.Model.load(gpa, &g);
    defer lm.deinit();

    const ring = localRingRows(lm.cfg);
    const prompt_len = ring + 64; // rings wrapped: every new row overwrites
    const n_gen = (ring - lm.cfg.sliding_window) + 64; // > slack: live rows destroyed
    const total = prompt_len + n_gen + 8;
    var model = try CudaLM.init(gpa, be, &lm, .{ .initial = total, .max = total, .kv_dtype = kv_dtype });
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
    // host→device, so an owner-misdirected restore would now surface as
    // divergence from the all-GPU baseline.
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
}

test "checkpoint restore regenerates token-identical on the real model" {
    try checkpointRestoreBody(.f32);
}

// Same workout on an f16 KV cache: the split's host shadow stores packed f16
// (byte-identical to the device rings), so the mid-conversation migrate, the
// owner-aware ring checkpoints, and the promote round trip must all stay
// exact — token-identical within the f16 session.
test "checkpoint restore regenerates token-identical on the real model (f16 kv)" {
    try checkpointRestoreBody(.f16);
}

// And on a q8_0 KV cache: the shadow stores raw ggml blocks, the ring
// checkpoints stay dtype-agnostic byte copies at the block-aware size.
test "checkpoint restore regenerates token-identical on the real model (q8_0 kv)" {
    try checkpointRestoreBody(.q8_0);
}

// The initial split plan must respect the card's LIVE free VRAM, not just the
// abstract budget: with most of the card occupied (another process, a resident
// image model), a generous budget must still plan layers onto the host —
// planning them all resident faults at the first prefill instead (weight
// uploads + lazy PTX JIT collide at zero free; surfaced in tp-gui as
// "PTX JIT failed: CUDA_ERROR_ILLEGAL_ADDRESS").
test "cpu split plan respects live free VRAM" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const test_gate = @import("../test_gate.zig");
    const Gguf = @import("tp_core").gguf.Gguf;
    const path = "/home/qt/genai/lmstudio/models/mradermacher/Gemma-3-Starshine-12B-Alt-GGUF/Gemma-3-Starshine-12B-Alt.Q4_K_M.gguf";
    try test_gate.requireModelFile(io, path);
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    var g = try Gguf.open(gpa, io, path);
    defer g.deinit();
    var lm = try gemma3.Model.load(gpa, &g);
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
// stepper's `io` unseeded that used to be undefined-pointer UB (GP fault deep
// in the host matmul's groupAsync); it must instead fail closed, and work once
// the session owner seeds it (gui/chat.zig Session.init does).
test "cpu split prefill before any step needs a seeded io" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const test_gate = @import("../test_gate.zig");
    const Gguf = @import("tp_core").gguf.Gguf;
    const path = "/home/qt/genai/lmstudio/models/mradermacher/Gemma-3-Starshine-12B-Alt-GGUF/Gemma-3-Starshine-12B-Alt.Q4_K_M.gguf";
    try test_gate.requireModelFile(io, path);
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    var g = try Gguf.open(gpa, io, path);
    defer g.deinit();
    var lm = try gemma3.Model.load(gpa, &g);
    defer lm.deinit();
    var model = try CudaLM.init(gpa, be, &lm, .{ .initial = 128, .max = 256 });
    defer model.deinit();

    // A budget below the weight total statically places tail layers on the
    // host right away (kept small so the Debug host prefill stays quick).
    try model.enableCpuSplit(.tail, 6 << 30, true);
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
