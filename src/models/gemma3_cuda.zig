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
const residency = @import("residency.zig");
const transformer = @import("transformer.zig");
const transformer_gpu = @import("transformer_gpu.zig");

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
    /// Io for the host matmuls of a hybrid split's CPU-resident layers; set by
    /// `step`. Undefined until the first `step` — the offload host path only
    /// runs after generation starts (text-only sessions), same as qwen35_cuda.
    io: std.Io = undefined,
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
    fn allocKvCaches(self: *CudaLM) !void {
        const cfg = self.cfg;
        const be = self.be;
        const esz = self.kv_dtype.elemBytes();
        for (self.k_cache, self.v_cache, 0..) |*kb, *vb, l| {
            if (usesRing(cfg, l)) {
                const bytes = localRingRows(cfg) * cfg.kvDim() * esz;
                kb.* = try be.growableCreate(bytes, bytes);
                vb.* = try be.growableCreate(bytes, bytes);
            } else {
                kb.* = try be.growableCreate(self.initial_capacity * cfg.kvDim() * esz, self.max_capacity * cfg.kvDim() * esz);
                vb.* = try be.growableCreate(self.initial_capacity * cfg.kvDim() * esz, self.max_capacity * cfg.kvDim() * esz);
            }
        }
    }

    /// Store `n` K/V elements from `src` (+`src_off` elems) into cache buffer
    /// `dst` at element offset `dst_off`. f32 copies raw; f16 converts on store.
    fn storeKv(self: *CudaLM, dst: Buf, dst_off: usize, src: Buf, src_off: usize, n: usize) !void {
        if (self.kv_dtype == .f16) {
            try self.be.opStoreKvF16(dst, dst_off, src, src_off, n);
        } else {
            try self.be.tensorCopy(dst, dst_off * 4, src, src_off * 4, n * 4);
        }
    }

    /// Rebuild the KV cache at a new element dtype (GUI toggle), weights resident.
    /// gemma3 decodes straight through `opAttnDecode` (no captured graph), so this
    /// just frees + re-creates the K/V buffers and resets the length. f16 with an
    /// active CPU-offload split is unsupported (the host shadow cache is f32).
    pub fn reinitCache(self: *CudaLM, dtype: kvmod.KvDtype) !void {
        if (dtype == .f16 and self.split != null) return error.KvDtypeUnsupported;
        for (self.k_cache) |*b| self.be.growableDestroy(b);
        for (self.v_cache) |*b| self.be.growableDestroy(b);
        self.kv_dtype = dtype;
        self.capacity = self.initial_capacity;
        self.len = 0;
        try self.allocKvCaches();
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
        if (self.len > 0) {
            const cap = sp.cache.capacity;
            if (usesRing(cfg, l)) {
                // LOCAL ring -> linear host shadow: copy the live window,
                // translating ring rows (pos%ring) to absolute host rows.
                for (ringSegments(self.len, localRingRows(cfg))) |seg| {
                    if (seg.n == 0) continue;
                    const hk = sp.cache.k[l * cap * kvd + seg.abs * kvd ..][0 .. seg.n * kvd];
                    const hv = sp.cache.v[l * cap * kvd + seg.abs * kvd ..][0 .. seg.n * kvd];
                    try self.be.tensorDownload(offsetBufSized(self.k_cache[l].buf, seg.dev * kvd * 4, seg.n * kvd * 4), std.mem.sliceAsBytes(hk));
                    try self.be.tensorDownload(offsetBufSized(self.v_cache[l].buf, seg.dev * kvd * 4, seg.n * kvd * 4), std.mem.sliceAsBytes(hv));
                }
            } else {
                const hk = sp.cache.k[l * cap * kvd ..][0 .. self.len * kvd];
                const hv = sp.cache.v[l * cap * kvd ..][0 .. self.len * kvd];
                try self.be.tensorDownload(offsetBufSized(self.k_cache[l].buf, 0, self.len * kvd * 4), std.mem.sliceAsBytes(hk));
                try self.be.tensorDownload(offsetBufSized(self.v_cache[l].buf, 0, self.len * kvd * 4), std.mem.sliceAsBytes(hv));
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
        std.log.info("[offload] layer {d} (attn) -> CPU at ctx {d} ({d}/{d} on CPU)", .{ l, self.len, sp.n_cpu, cfg.n_layers });
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
        const kv_at_cap = 2 * self.capacity * self.cfg.kvDim() * 4;
        return layerDeviceBytes(&self.lm.layers[l]) + kv_at_cap + (64 << 20);
    }

    /// Bring layer `l` back onto the GPU, preserving its accumulated K/V: re-create
    /// the device K/V at the current capacity and upload the host rows [0,len).
    /// Weights re-cache lazily on the next GPU forward. Reverse of migrateLayer.
    pub fn promoteLayer(self: *CudaLM, l: usize) !void {
        const cfg = self.cfg;
        const sp = &self.split.?;
        const kvd = cfg.kvDim();
        if (usesRing(cfg, l)) {
            const bytes = localRingRows(cfg) * kvd * 4;
            self.k_cache[l] = try self.be.growableCreate(bytes, bytes);
            self.v_cache[l] = try self.be.growableCreate(bytes, bytes);
        } else {
            self.k_cache[l] = try self.be.growableCreate(self.capacity * kvd * 4, self.max_capacity * kvd * 4);
            self.v_cache[l] = try self.be.growableCreate(self.capacity * kvd * 4, self.max_capacity * kvd * 4);
        }
        if (self.len > 0) {
            const cap = sp.cache.capacity;
            if (usesRing(cfg, l)) {
                // Linear host shadow -> LOCAL ring: reverse of migrateLayer.
                for (ringSegments(self.len, localRingRows(cfg))) |seg| {
                    if (seg.n == 0) continue;
                    const hk = sp.cache.k[l * cap * kvd + seg.abs * kvd ..][0 .. seg.n * kvd];
                    const hv = sp.cache.v[l * cap * kvd + seg.abs * kvd ..][0 .. seg.n * kvd];
                    try self.be.tensorUpload(offsetBufSized(self.k_cache[l].buf, seg.dev * kvd * 4, seg.n * kvd * 4), std.mem.sliceAsBytes(hk));
                    try self.be.tensorUpload(offsetBufSized(self.v_cache[l].buf, seg.dev * kvd * 4, seg.n * kvd * 4), std.mem.sliceAsBytes(hv));
                }
            } else {
                const hk = sp.cache.k[l * cap * kvd ..][0 .. self.len * kvd];
                const hv = sp.cache.v[l * cap * kvd ..][0 .. self.len * kvd];
                try self.be.tensorUpload(offsetBufSized(self.k_cache[l].buf, 0, self.len * kvd * 4), std.mem.sliceAsBytes(hk));
                try self.be.tensorUpload(offsetBufSized(self.v_cache[l].buf, 0, self.len * kvd * 4), std.mem.sliceAsBytes(hv));
            }
        }
        sp.on_gpu[l] = true;
        sp.n_cpu -= 1;
        std.log.info("[promote] layer {d} -> GPU at ctx {d} ({d}/{d} on CPU)", .{ l, self.len, sp.n_cpu, cfg.n_layers });
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
        const cfg = self.cfg;
        const kvd = cfg.kvDim();
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
        for (self.k_cache, self.v_cache, 0..) |*kb, *vb, l| {
            self.be.growableDestroy(kb);
            self.be.growableDestroy(vb);
            if (usesRing(cfg, l)) {
                const bytes = localRingRows(cfg) * kvd * 4;
                kb.* = try self.be.growableCreate(bytes, bytes);
                vb.* = try self.be.growableCreate(bytes, bytes);
            } else {
                kb.* = try self.be.growableCreate(self.initial_capacity * kvd * 4, self.max_capacity * kvd * 4);
                vb.* = try self.be.growableCreate(self.initial_capacity * kvd * 4, self.max_capacity * kvd * 4);
            }
        }
        self.capacity = self.initial_capacity;
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
        // f16 KV stays fully resident (see migrateNextLayer); f16 halves the KV
        // footprint, so the pressure that would trigger auto-offload is lower.
        if (self.kv_dtype == .f16) return false;
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
        // f16 KV + CPU offload is unsupported: the host shadow cache is f32.
        if (self.kv_dtype == .f16) return error.KvDtypeUnsupported;
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
        const kv_bytes = 2 * kv_rows * cfg.kvDim() * 4;
        const reserve = if (dynamic)
            self.lm.head.bytes.len
        else
            kv_bytes + self.lm.head.bytes.len + (512 << 20);
        const gpu_weight_budget: usize = if (budget > reserve) budget - reserve else 0;

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
            const l = order[n_cpu];
            on_gpu[l] = false;
            gpu_weight -= per[l];
            n_cpu += 1;
        }
        if (n_cpu == 0 and !dynamic) {
            gpa.free(on_gpu);
            gpa.free(order);
            return; // everything fits resident — no split needed
        }

        // Host state for the CPU-resident layers (sized to the current KV
        // capacity; grows with the device via ensureCapacity; len starts at 0).
        // Offloaded-layer host shadow is always f32 (f16 disables the split).
        var cache = try kvmod.KvCache.init(gpa, n, self.capacity, cfg.kvDim(), .f32);
        errdefer cache.deinit(gpa);
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
            .n_cpu = n_cpu,
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

        // Free the device K/V of layers placed on the host up front (len is 0
        // here, nothing to copy). Weights are reclaimed lazily by the cache.
        for (order[0..n_cpu]) |l| {
            self.be.growableDestroy(&self.k_cache[l]);
            self.be.growableDestroy(&self.v_cache[l]);
        }
    }

    pub fn ensureCapacity(self: *CudaLM, min_rows: usize) !void {
        const target = (try kvmod.growPlan(self.capacity, self.max_capacity, min_rows)) orelse return;
        // Element stride MUST match how the buffers were created (esz), or an f16
        // cache (esz=2) requests f32-sized growth, overshoots its VA reservation,
        // and growableEnsure fails with DeviceOutOfMemory → ContextFull once the
        // window grows past ~max_capacity/2.
        const bytes = target * self.cfg.kvDim() * self.kv_dtype.elemBytes();

        // Dynamic offload: migrate layers GPU->CPU until there's headroom to grow
        // the device KV, instead of streaming weights (the cliff). Each migrated
        // layer frees its device KV + weight VRAM. Mirrors qwen35_cuda.
        if (self.split) |*sp| if (sp.dynamic) {
            const add = (target - self.capacity) * self.cfg.kvDim() * 4;
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
        try self.forwardDecode(io, ids, null);
        const be = self.be;
        const b = &self.bufs;
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
        try self.forwardDecode(io, ids, null);
        const be = self.be;
        const b = &self.bufs;
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
        try self.be.opAttnDecode(b.q, self.k_cache[l].buf, self.v_cache[l].buf, b.attn, b.attn_scratch, pos0 + 1, seq, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, ns, scale, window, ring, self.bidir_prefill, self.kv_dtype == .f16);
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
                    var sv = sp.scratch.viewSeq(n, cfg);
                    try gemma3.layerForward(self.io, self.gpa, cfg, layer, sp.hx[0 .. n * cfg.hidden], n, sp.freqs_global, sp.freqs_local, &sp.cache, l, self.bidir_prefill, &sv);
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
