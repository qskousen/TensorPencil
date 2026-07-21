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
const cuda = @import("tp_gpu").cuda;
const ops = @import("tp_ops");
const kvmod = @import("tp_core").kv_cache;
const sample = @import("tp_core").sample;
const residency = @import("tp_runtime").residency;

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

fn nbuf(be: *Backend, weights: []const f32) !Buf {
    return .{ .buf = try be.smallBuffer(std.mem.sliceAsBytes(weights)), .mem = .null_handle, .size = 0 };
}

fn offsetBufSized(b: Buf, off_bytes: usize, size: usize) Buf {
    return .{ .buf = @enumFromInt(@intFromEnum(b.buf) + off_bytes), .mem = b.mem, .size = size };
}

/// Whether the captured-decode-graph path is available for this embedding dtype
/// (the graph needs a device-side embedding gather kernel). Used at init and to
/// restore graph_ok after a split is dropped (resetResidency).
fn graphOkFor(dtype: anytype) bool {
    return switch (dtype) {
        .bf16, .q8_0, .q4_k, .q5_k, .q6_k => true,
        else => false,
    };
}

/// KV chunks per head in the decode attention split pass.
const nsplit = 32;
/// Batched-prefill chunk (rows per stepBatch) and its attention split count
/// (bounds the flash-decode scratch).
const prefill_chunk = 128;
const nsplit_prefill = 8;

/// Which layers a hybrid CPU/GPU split pushes to the host, once the count is
/// fixed by the VRAM budget. `tail` keeps a contiguous device prefix (the
/// last N layers go to CPU); `attn` evicts the KV-growing attention layers
/// first (frees the most device memory as context grows, but interleaves).
pub const CpuSplitPolicy = enum { tail, attn };

pub const CudaLM = struct {
    lm: *const qwen35.Model,
    be: *Backend,
    gpa: std.mem.Allocator,
    cfg: qwen35.Config,
    /// Committed KV rows; grows in place toward max_capacity (ensureCapacity).
    capacity: usize,
    /// Growth ceiling — the VA reservation behind each KV cache and the RoPE
    /// table are sized to this, so growth never moves a device pointer.
    max_capacity: usize,
    /// The KV capacity a fresh session starts at; resetResidency shrinks back to
    /// it so a new chat frees the grown KV VRAM.
    initial_capacity: usize,
    /// KV-cache element storage type (f32 / f16); selects the attention/append
    /// kernel variant and the per-element stride of k_cache/v_cache.
    kv_dtype: kvmod.KvDtype,
    len: usize,

    bufs: Bufs,
    /// Per-attention-slot KV caches, [capacity][kvDim] f32 (growable).
    k_cache: []Growable,
    v_cache: []Growable,
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
    /// Io for the CPU half of a hybrid split (set each step); undefined and
    /// unread when no split is active.
    /// null fails the host path closed (error.SplitIoUnset) instead of
    /// undefined-pointer UB when a prefill with host layers runs before any
    /// step; the session owner seeds it (tp-gui at init).
    io: ?std.Io = null,
    /// Hybrid CPU/GPU layer split (null = every layer on device). When set,
    /// the layers with `on_gpu[l] == false` run on the host via
    /// `qwen35.Model.cpuLayer`, keeping their weights off the device entirely
    /// (no per-token PCIe re-stream). Forces the per-op decode path — a
    /// captured graph cannot record host compute.
    split: ?Split = null,
    arena: std.heap.ArenaAllocator,

    /// CPU-resident layers of a hybrid split, with the host-side state they
    /// need. Allocated with `gpa`; freed in `deinit`.
    pub const Split = struct {
        /// Per-layer: compute this layer on the device? (false = host).
        on_gpu: []bool,
        /// How many layers run on the host.
        n_cpu: usize,
        /// The policy that placed them (for reporting).
        policy: CpuSplitPolicy,
        /// Host KV (CPU attention layers) + conv/ssm (CPU linear layers),
        /// grown in lockstep with the device caches.
        state: qwen35.State,
        /// Host activation scratch, sized for a full prefill chunk.
        scratch: qwen35.Scratch,
        /// Host RoPE table, grows with capacity (scalar positions — text only).
        freqs: ops.rope.Freqs,
        /// Host hidden buffer ([prefill_chunk * hidden]).
        hx: []f32,
        /// Does the live hidden currently sit in `hx` (vs device `bufs.x`)?
        on_host: bool = false,
        /// Dynamic offload: migrate more layers GPU->CPU as the KV cache grows,
        /// instead of streaming weights (the cliff). Starts from the static
        /// placement and only ever moves layers to the host (context grows).
        dynamic: bool = false,
        /// Migration priority (layer indices); `next` is the position of the
        /// next layer to move to the host. order[0..next) are already on CPU.
        order: []usize = &.{},
        next: usize = 0,
        /// VRAM ceiling (bytes) the dynamic scheduler keeps device usage under.
        budget: u64 = 0,
    };

    pub fn init(gpa: std.mem.Allocator, be: *Backend, lm: *const qwen35.Model, cap: kvmod.Capacity) !CudaLM {
        const cfg = lm.cfg;
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
        self.pos_next = 0;
        self.layer_dump = null;
        self.op_dump = null;
        self.op_dump_row = 0;
        self.q8_for = .{};
        self.q8_cols = 0;
        self.graph_exec = null;
        self.split = null;
        self.io = null; // field defaults do not apply to `undefined`-built structs
        self.decode_warm = false;
        // The graph path needs a device-side embedding gather kernel.
        self.graph_ok = graphOkFor(lm.embed.dtype);

        // Rope table for the rotated span (rope_dim), like qwen3_cuda's —
        // computed to max_capacity up front so sin_off (baked into captured
        // graphs as a kernel param) never changes when the KV caches grow.
        var freqs = try ops.rope.rotateHalfFreqs(gpa, cap.max, cfg.rope_dim, cfg.rope_theta);
        defer freqs.deinit(gpa);
        const half = cfg.rope_dim / 2;
        const fp = try gpa.alloc(f32, 2 * cap.max * half);
        defer gpa.free(fp);
        @memcpy(fp[0 .. cap.max * half], freqs.cos);
        @memcpy(fp[cap.max * half ..], freqs.sin);
        self.sin_off = cap.max * half;
        self.freqs_d = try be.tensorCreate(fp.len * 4);
        try be.tensorUpload(self.freqs_d, std.mem.sliceAsBytes(fp));
        self.pos3_d = try be.tensorCreate(3 * 4);
        self.pos3s_d = try be.tensorCreate(prefill_chunk * 3 * 4);

        self.bufs = try Bufs.init(be, cfg);

        const n_attn = cfg.nAttnLayers();
        const kdt = cap.kv_dtype;
        self.k_cache = try alloc.alloc(Growable, n_attn);
        self.v_cache = try alloc.alloc(Growable, n_attn);
        for (self.k_cache, self.v_cache) |*kb, *vb| {
            kb.* = try be.growableCreate(kdt.sizeBytes(cap.initial * cfg.kvDim()), kdt.sizeBytes(cap.max * cfg.kvDim()));
            vb.* = try be.growableCreate(kdt.sizeBytes(cap.initial * cfg.kvDim()), kdt.sizeBytes(cap.max * cfg.kvDim()));
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
    /// Drops the captured decode graph so it re-captures with the f16/f32 kernels,
    /// frees + re-creates the per-attention-slot K/V buffers (and a split's host
    /// shadow state, which stores the same dtype as the device), resets the length.
    pub fn reinitCache(self: *CudaLM, dtype: kvmod.KvDtype) !void {
        const be = self.be;
        const cfg = self.cfg;
        if (self.graph_exec != null) {
            be.graphDestroy(self.graph_exec);
            self.graph_exec = null;
        }
        self.decode_warm = false;
        // Host-resident attention layers keep no device KV (their growables
        // were already destroyed by migrateLayer; growableDestroy is
        // idempotent) — only GPU-resident slots are re-created at the new
        // element size.
        for (self.k_cache) |*b| be.growableDestroy(b);
        for (self.v_cache) |*b| be.growableDestroy(b);
        self.kv_dtype = dtype;
        self.capacity = self.initial_capacity;
        for (self.lm.layers, 0..) |_, l| {
            if (cfg.isRecurrent(l)) continue;
            if (self.split) |*sp| if (!sp.on_gpu[l]) continue;
            const s = l / cfg.full_attn_interval;
            self.k_cache[s] = try be.growableCreate(dtype.sizeBytes(self.initial_capacity * cfg.kvDim()), dtype.sizeBytes(self.max_capacity * cfg.kvDim()));
            self.v_cache[s] = try be.growableCreate(dtype.sizeBytes(self.initial_capacity * cfg.kvDim()), dtype.sizeBytes(self.max_capacity * cfg.kvDim()));
        }
        if (self.split) |*sp| {
            const state = try qwen35.State.init(self.gpa, cfg, self.capacity, dtype);
            sp.state.deinit(self.gpa);
            sp.state = state;
        }
        // Full context clear, not just `len = 0`: the caller re-prefills the
        // whole transcript, which must start from ZEROED conv/ssm recurrent
        // state and pos 0 — stale state would double-apply the transcript
        // through the linear-attention layers.
        try self.resetCache();
    }

    pub fn deinit(self: *CudaLM) void {
        const be = self.be;
        if (self.graph_exec != null) be.graphDestroy(self.graph_exec);
        for (self.k_cache) |*b| be.growableDestroy(b);
        for (self.v_cache) |*b| be.growableDestroy(b);
        be.tensorDestroy(&self.conv_state);
        be.tensorDestroy(&self.ssm_state);
        be.tensorDestroy(&self.freqs_d);
        be.tensorDestroy(&self.pos3_d);
        be.tensorDestroy(&self.pos3s_d);
        self.bufs.deinit(be);
        if (self.split) |*sp| {
            sp.state.deinit(self.gpa);
            sp.scratch.deinit(self.gpa);
            sp.freqs.deinit(self.gpa);
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

    /// Commit more KV rows, in place: device pointers (and the captured
    /// decode graph — sin_off and the KV strides are capacity-independent)
    /// stay valid. Under VRAM pressure the commit evicts LRU weights into
    /// the streaming path, which flips the graph off via the evictions guard
    /// in step(). error.ContextFull past the window or when even eviction
    /// can't free enough device memory.
    pub fn ensureCapacity(self: *CudaLM, min_rows: usize) !void {
        const target = (try kvmod.growPlan(self.capacity, self.max_capacity, min_rows)) orelse return;
        // Byte size MUST match how the buffers were created (kv_dtype block
        // math), or an f16/q8_0 cache requests f32-sized growth, overshoots its
        // VA reservation, and growableEnsure fails with DeviceOutOfMemory →
        // ContextFull once the window grows past ~max_capacity/2.
        const bytes = self.kv_dtype.sizeBytes(target * self.cfg.kvDim());

        // Dynamic offload: migrate layers GPU->CPU until there's headroom to
        // grow the device KV — instead of streaming weights (the cliff). Each
        // migrated attention layer frees its device KV immediately; linear
        // layers free weight VRAM via the cache. Migrate attention-first (the
        // policy's order) so headroom recovers per step.
        if (self.split) |*sp| if (sp.dynamic) {
            const add = self.kv_dtype.sizeBytes((target - self.capacity) * self.cfg.kvDim());
            while (true) {
                const need = self.liveAttnSlots() * 2 * add + (32 << 20); // + margin
                const free = @min(sp.budget -| self.be.deviceUsed(), self.be.headroom());
                if (free >= need) break;
                if (!(try residency.migrateNext(self))) break; // nothing left; fall through
            }
        };

        // Grow device KV of the attention layers still on the GPU. Physical VRAM
        // can be exhausted even when the proactive migration above thought there
        // was room: a resident image model on another CUDA context may grab it
        // between the headroom check and this commit. On a real OOM, offload one
        // more layer to the CPU and retry the whole grow, so a full window only
        // ever fails once nothing is left to migrate — never a premature
        // ContextFull while layers can still move to host. growableEnsure is
        // idempotent, so re-running the loop is cheap.
        grow: while (true) {
            for (self.lm.layers, 0..) |*layer, l| {
                if (self.cfg.isRecurrent(l)) continue;
                _ = layer;
                if (self.split) |*sp| if (!sp.on_gpu[l]) continue;
                const s = l / self.cfg.full_attn_interval;
                for ([2]*Growable{ &self.k_cache[s], &self.v_cache[s] }) |b| {
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
            sp.state.kv.grow(self.gpa, target) catch return error.ContextFull;
            sp.state.capacity = target;
            const nf = ops.rope.rotateHalfFreqs(self.gpa, target, self.cfg.rope_dim, self.cfg.rope_theta) catch return error.ContextFull;
            sp.freqs.deinit(self.gpa);
            sp.freqs = nf;
        }
        self.capacity = target;
    }

    /// Attention layers whose KV still lives on the device.
    fn liveAttnSlots(self: *CudaLM) usize {
        var count: usize = 0;
        for (self.lm.layers, 0..) |*layer, l| {
            _ = layer;
            if (self.cfg.isRecurrent(l)) continue;
            if (self.split) |*sp| if (!sp.on_gpu[l]) continue;
            count += 1;
        }
        return count;
    }

    /// Move layer `l`'s live state device->host and mark it CPU-resident (a
    /// `residency` hook).
    pub fn migrateLayer(self: *CudaLM, l: usize) !void {
        const cfg = self.cfg;
        const sp = &self.split.?;
        if (!cfg.isRecurrent(l)) {
            // Attention: copy accumulated KV to the host, free the device KV.
            // The host shadow stores the same dtype as the device cache
            // (kRowBytes is the device byte layout), so this is a raw copy.
            const s = l / cfg.full_attn_interval;
            const kvd = cfg.kvDim();
            if (self.len > 0) {
                const dt = self.kv_dtype;
                try self.be.tensorDownload(offsetBufSized(self.k_cache[s].buf, 0, dt.sizeBytes(self.len * kvd)), sp.state.kv.kRowBytes(s, 0, self.len));
                try self.be.tensorDownload(offsetBufSized(self.v_cache[s].buf, 0, dt.sizeBytes(self.len * kvd)), sp.state.kv.vRowBytes(s, 0, self.len));
            }
            self.be.growableDestroy(&self.k_cache[s]);
            self.be.growableDestroy(&self.v_cache[s]);
        } else {
            // Linear (DeltaNet): copy the fixed-size recurrent conv/ssm state.
            const lin_idx = l - l / cfg.full_attn_interval;
            const conv_n = cfg.convChannels() * (cfg.conv_kernel - 1);
            const ssm_n = cfg.lin_v_heads * cfg.lin_head_dim * cfg.lin_head_dim;
            const hc = sp.state.conv[lin_idx * conv_n ..][0..conv_n];
            const hs = sp.state.ssm[lin_idx * ssm_n ..][0..ssm_n];
            try self.be.tensorDownload(offsetBufSized(self.conv_state, lin_idx * conv_n * 4, conv_n * 4), std.mem.sliceAsBytes(hc));
            try self.be.tensorDownload(offsetBufSized(self.ssm_state, lin_idx * ssm_n * 4, ssm_n * 4), std.mem.sliceAsBytes(hs));
        }
        // Free the migrated layer's device weights (ggml reads them from host
        // now) — this is the bulk of the reclaimed VRAM.
        const layer = &self.lm.layers[l];
        const mlp = switch (layer.*) {
            .attn => |*a| &a.mlp,
            .linear => |*ll| &ll.mlp,
        };
        self.be.evictWeightBytes(mlp.gate.bytes);
        self.be.evictWeightBytes(mlp.up.bytes);
        self.be.evictWeightBytes(mlp.down.bytes);
        switch (layer.*) {
            .attn => |*a| {
                self.be.evictWeightBytes(a.qg.bytes);
                self.be.evictWeightBytes(a.k.bytes);
                self.be.evictWeightBytes(a.v.bytes);
                self.be.evictWeightBytes(a.o.bytes);
            },
            .linear => |*ll| {
                self.be.evictWeightBytes(ll.qkv.bytes);
                self.be.evictWeightBytes(ll.z.bytes);
                self.be.evictWeightBytes(ll.alpha.bytes);
                self.be.evictWeightBytes(ll.beta.bytes);
                self.be.evictWeightBytes(ll.out.bytes);
            },
        }
        sp.on_gpu[l] = false;
        sp.n_cpu += 1;
        std.log.debug("[offload] layer {d} ({s}) -> CPU at ctx {d} ({d}/{d} on CPU)", .{ l, if (cfg.isRecurrent(l)) "lin" else "attn", self.len, sp.n_cpu, cfg.n_layers });
    }

    /// Total device footprint of one layer's streamable weights (quantized
    /// bytes, as uploaded) — the big projection + MLP matrices; norms/conv
    /// are negligible and ignored.
    fn layerDeviceBytes(layer: *const qwen35.Layer) usize {
        const mlp = switch (layer.*) {
            .attn => |*a| &a.mlp,
            .linear => |*l| &l.mlp,
        };
        var n: usize = mlp.gate.bytes.len + mlp.up.bytes.len + mlp.down.bytes.len;
        switch (layer.*) {
            .attn => |*a| n += a.qg.bytes.len + a.k.bytes.len + a.v.bytes.len + a.o.bytes.len,
            .linear => |*l| n += l.qkv.bytes.len + l.z.bytes.len + l.alpha.bytes.len + l.beta.bytes.len + l.out.bytes.len,
        }
        return n;
    }

    /// Place layers on the host until the device-resident weights fit under
    /// `budget` (bytes), by the given policy. Must be called right after
    /// `init` (before any tokens), on a CUDA backend with `budget != 0`.
    /// No-op split (all layers fit) leaves `self.split == null`. Forces the
    /// per-op decode path (a captured graph cannot record host compute).
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

        // Device memory that is not streamable layer weight and must stay
        // resident: KV at the current capacity, the LM head, plus slack for
        // activation scratch, the RoPE table, and CUDA context / fragmentation.
        const kv_bytes = 2 * cfg.nAttnLayers() * self.kv_dtype.sizeBytes(self.capacity * cfg.kvDim());
        // Dynamic mode packs as many layers as possible onto the GPU up front
        // (just the LM head reserved) and migrates the rest on demand as the KV
        // cache grows past the current headroom. Static mode can't adapt later,
        // so it reserves generously (KV at capacity + head + slack).
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

        // Eviction order: which layers leave the device first. Kept in the
        // Split (dynamic migration walks it as the KV cache grows).
        const order = try gpa.alloc(usize, n);
        errdefer gpa.free(order);
        switch (policy) {
            .tail => for (0..n) |i| {
                order[i] = n - 1 - i; // last layer first → contiguous device prefix
            },
            .attn => {
                // Attention layers first (descending), then linear (descending):
                // frees the KV-growing layers before the fixed-state ones.
                var w: usize = 0;
                var l = n;
                while (l > 0) {
                    l -= 1;
                    if (!cfg.isRecurrent(l)) {
                        order[w] = l;
                        w += 1;
                    }
                }
                l = n;
                while (l > 0) {
                    l -= 1;
                    if (cfg.isRecurrent(l)) {
                        order[w] = l;
                        w += 1;
                    }
                }
            },
        }

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
        // KV stores the SAME dtype as the device caches, so migrate/promote
        // are raw byte copies (kRowBytes); conv/ssm recurrent state stays f32.
        var state = try qwen35.State.init(gpa, cfg, self.capacity, self.kv_dtype);
        errdefer state.deinit(gpa);
        // The host shadow tracks the SAME committed length as the device from
        // the moment the split arms (the per-step commits keep them in
        // lockstep afterwards). Armed mid-conversation (imageReclaim, or a
        // checkpoint-restore test), starting it at 0 would make host attention
        // layers attend over nothing and break KV rollback bookkeeping;
        // migrateLayer copies each migrated layer's rows [0, len) so declaring
        // them committed is correct.
        state.kv.len = self.len;
        state.len = self.len;
        var scratch = try qwen35.Scratch.init(gpa, prefill_chunk, cfg);
        errdefer scratch.deinit(gpa);
        var freqs = try ops.rope.rotateHalfFreqs(gpa, self.capacity, cfg.rope_dim, cfg.rope_theta);
        errdefer freqs.deinit(gpa);
        const hx = try gpa.alloc(f32, prefill_chunk * cfg.hidden);
        errdefer gpa.free(hx);

        self.split = .{
            .on_gpu = on_gpu,
            .n_cpu = 0, // the placement loops below mark + count the host layers
            .policy = policy,
            .state = state,
            .scratch = scratch,
            .freqs = freqs,
            .hx = hx,
            .dynamic = dynamic,
            .order = order,
            .next = n_cpu,
            .budget = budget,
        };
        self.graph_ok = false; // per-op path: host layers can't be captured

        // Place the statically-planned layers on the host. Before any tokens
        // (the autoOffload-at-init path) there is nothing to copy — mark them
        // and free the device KV; weights are reclaimed lazily by the cache
        // and conv/ssm start zeroed on both sides. Armed MID-conversation
        // (imageReclaim), each layer's live rows/state must move to the host
        // instead — migrateLayer does the copy AND the on_gpu/n_cpu
        // bookkeeping, or the context would be destroyed with the device KV.
        if (self.len == 0) {
            const sp = &self.split.?;
            for (order[0..n_cpu]) |l| {
                sp.on_gpu[l] = false;
                sp.n_cpu += 1;
                if (!cfg.isRecurrent(l)) {
                    const s = l / cfg.full_attn_interval;
                    self.be.growableDestroy(&self.k_cache[s]);
                    self.be.growableDestroy(&self.v_cache[s]);
                }
            }
        } else {
            for (order[0..n_cpu]) |l| try self.migrateLayer(l);
        }
    }

    pub fn vocab(self: *const CudaLM) usize {
        return self.cfg.vocab;
    }

    /// Arm dynamic CPU offload iff the projected max-context footprint (all layer
    /// weights + KV grown to `max_capacity` + reserve slack) won't fit under
    /// `budget` bytes. When it fits, do nothing — the model stays fully resident
    /// on the fast path. When it doesn't, `enableCpuSplit(.attn, budget, dynamic)`
    /// packs the GPU now and migrates layers to the host as the KV cache grows
    /// (instead of the weight-streaming cliff). Must run before any tokens.
    /// `budget == 0` is treated as "no limit" (never offload). Text sessions
    /// only — the host layer path uses scalar RoPE (see enableCpuSplit / the GUI
    /// guard). Returns whether a split was armed.
    pub fn autoOffload(self: *CudaLM, budget: u64) !bool {
        if (budget == 0) return false; // no budget → fully resident, no offload path
        // Always arm the dynamic split. When the model fits its budget it places
        // ZERO layers on the CPU — measured free (per-op decode ties the captured
        // graph: 76.2 vs 77.2 tok/s on the 9B) — and migrates layers on demand as
        // the KV cache grows. Arming unconditionally means over-budget growth
        // degrades via CPU offload (measured ~2.5x FASTER than weight streaming on
        // this box: 18.3 vs 7.0 tok/s) instead of falling into the slow streaming
        // fallback that a "does it fit?" check would leave it in.
        try self.enableCpuSplit(.attn, budget, true);
        return true;
    }

    /// Migrate layers to the host until `@min(budget - deviceUsed, headroom)`
    /// reaches `needed_free` bytes, or nothing is left. Fixed-target variant used
    /// by the VRAM coordinator (free room for the image model). No-op without a
    /// dynamic split. (ensureCapacity keeps its own loop, whose target shrinks per
    /// iteration as liveAttnSlots drops — a fixed target here can't express that.)
    pub fn offloadUntilFree(self: *CudaLM, needed_free: u64) !void {
        return residency.offloadUntilFree(self, needed_free);
    }

    /// Migrate layers to the host until the LLM's ACTUAL total device usage is at
    /// or under `target` bytes — used by the GUI's `balanced` mode to settle the
    /// LLM to its share (e.g. 75% of the limit) ONLY when an image model loads
    /// (contention); with no image loaded the LLM keeps the whole limit. Uses live
    /// `deviceUsed()` (robust — no size estimation), so it must run after the
    /// weights are on the device. One-way + idempotent: once under `target` it's a
    /// no-op, so repeat calls (per image) don't re-shuffle.
    pub fn offloadToBudget(self: *CudaLM, target: u64) !void {
        return residency.offloadToBudget(self, target);
    }

    /// Bring layer `l` back onto the GPU, preserving its accumulated state
    /// (reverse of migrateLayer): attention layers re-create the device K/V at the
    /// current capacity and upload the host rows [0,len); linear layers re-upload
    /// their conv/ssm state into the shared device buffers. Weights re-cache
    /// lazily on the next GPU forward.
    /// `residency.promoteBack` cost hook: VRAM a promote of layer `l` needs — its
    /// streamable weights, the KV it re-commits at capacity (attention layers
    /// only; recurrent layers hold fixed-size conv/ssm state, no KV), plus slack.
    pub fn promoteCost(self: *CudaLM, l: usize) usize {
        const kv_cost: usize = if (!self.cfg.isRecurrent(l)) 2 * self.kv_dtype.sizeBytes(self.capacity * self.cfg.kvDim()) else 0;
        return layerDeviceBytes(&self.lm.layers[l]) + kv_cost + (64 << 20);
    }

    pub fn promoteLayer(self: *CudaLM, l: usize) !void {
        const cfg = self.cfg;
        const sp = &self.split.?;
        if (!cfg.isRecurrent(l)) {
            const s = l / cfg.full_attn_interval;
            const kvd = cfg.kvDim();
            const dt = self.kv_dtype;
            self.k_cache[s] = try self.be.growableCreate(dt.sizeBytes(self.capacity * kvd), dt.sizeBytes(self.max_capacity * kvd));
            self.v_cache[s] = try self.be.growableCreate(dt.sizeBytes(self.capacity * kvd), dt.sizeBytes(self.max_capacity * kvd));
            if (self.len > 0) {
                try self.be.tensorUpload(offsetBufSized(self.k_cache[s].buf, 0, dt.sizeBytes(self.len * kvd)), sp.state.kv.kRowBytes(s, 0, self.len));
                try self.be.tensorUpload(offsetBufSized(self.v_cache[s].buf, 0, dt.sizeBytes(self.len * kvd)), sp.state.kv.vRowBytes(s, 0, self.len));
            }
        } else {
            const lin_idx = l - l / cfg.full_attn_interval;
            const conv_n = cfg.convChannels() * (cfg.conv_kernel - 1);
            const ssm_n = cfg.lin_v_heads * cfg.lin_head_dim * cfg.lin_head_dim;
            const hc = sp.state.conv[lin_idx * conv_n ..][0..conv_n];
            const hs = sp.state.ssm[lin_idx * ssm_n ..][0..ssm_n];
            try self.be.tensorUpload(offsetBufSized(self.conv_state, lin_idx * conv_n * 4, conv_n * 4), std.mem.sliceAsBytes(hc));
            try self.be.tensorUpload(offsetBufSized(self.ssm_state, lin_idx * ssm_n * 4, ssm_n * 4), std.mem.sliceAsBytes(hs));
        }
        sp.on_gpu[l] = true;
        sp.n_cpu -= 1;
        std.log.debug("[promote] layer {d} ({s}) -> GPU at ctx {d} ({d}/{d} on CPU)", .{ l, if (cfg.isRecurrent(l)) "lin" else "attn", self.len, sp.n_cpu, cfg.n_layers });
    }

    /// Migrate CPU layers back onto the GPU (LIFO by offload order), stopping
    /// before the next would overflow `budget` — so the caller (VRAM coordinator,
    /// after image generation) reclaims LLM residency while leaving room for
    /// whatever else stays resident. Keeps the split armed. Returns the count
    /// promoted; 0 without a split.
    pub fn promoteLayers(self: *CudaLM, budget: u64) !usize {
        return residency.promoteBack(self, budget);
    }

    /// New-chat reset: drop the split (restoring the resident fast path + graph),
    /// shrink every K/V cache back to the initial capacity (frees the grown VRAM),
    /// clear the context, and re-arm dynamic offload for the fresh small context.
    /// KV is discarded, so no host->device copy is needed (unlike promoteLayers).
    pub fn resetResidency(self: *CudaLM, budget: u64) !void {
        const cfg = self.cfg;
        const kvd = cfg.kvDim();
        if (self.split) |*sp| {
            sp.state.deinit(self.gpa);
            sp.scratch.deinit(self.gpa);
            sp.freqs.deinit(self.gpa);
            self.gpa.free(sp.hx);
            self.gpa.free(sp.on_gpu);
            self.gpa.free(sp.order);
            self.split = null;
            self.graph_ok = graphOkFor(self.lm.embed.dtype);
            self.decode_warm = false;
        }
        const dt = self.kv_dtype;
        for (self.k_cache, self.v_cache) |*kb, *vb| {
            self.be.growableDestroy(kb);
            self.be.growableDestroy(vb);
            kb.* = try self.be.growableCreate(dt.sizeBytes(self.initial_capacity * kvd), dt.sizeBytes(self.max_capacity * kvd));
            vb.* = try self.be.growableCreate(dt.sizeBytes(self.initial_capacity * kvd), dt.sizeBytes(self.max_capacity * kvd));
        }
        self.capacity = self.initial_capacity;
        try self.resetCache();
        _ = try self.autoOffload(budget);
    }

    /// Reset the session to an empty context (used by the layer-parity debug
    /// harness and by the GUI's "new chat"): KV rows are overwritten lazily, so
    /// only the position counters and the conv/recurrent states need clearing.
    pub fn resetCache(self: *CudaLM) !void {
        const cfg = self.cfg;
        const n_lin = cfg.n_layers - cfg.nAttnLayers();
        try zeroBuffer(self.be, self.gpa, self.conv_state, n_lin * cfg.convChannels() * (cfg.conv_kernel - 1) * 4);
        try zeroBuffer(self.be, self.gpa, self.ssm_state, n_lin * cfg.lin_v_heads * cfg.lin_head_dim * cfg.lin_head_dim * 4);
        self.len = 0;
        self.pos_next = 0;
    }

    /// Fixed byte size of a turn checkpoint (see `checkpoint`): the M-RoPE
    /// position plus every DeltaNet layer's conv + ssm recurrent state.
    /// Independent of context length — attention KV is append-only, so a
    /// rollback copies none of it (`restoreCheckpoint` just truncates).
    pub fn checkpointBytes(self: *const CudaLM) usize {
        const cfg = self.cfg;
        const n_lin = cfg.n_layers - cfg.nAttnLayers();
        const conv_n = cfg.convChannels() * (cfg.conv_kernel - 1);
        const ssm_n = cfg.lin_v_heads * cfg.lin_head_dim * cfg.lin_head_dim;
        return @sizeOf(u64) + n_lin * (conv_n + ssm_n) * 4;
    }

    /// Snapshot the non-append-only context state at the current position into
    /// `out` (`checkpointBytes` long): `pos_next` plus each DeltaNet layer's
    /// conv/ssm state, read from whichever side (device buffer or split host
    /// shadow) currently owns the layer. The layout is owner-agnostic, so a
    /// layer may migrate between snapshot and restore. Pair with
    /// `restoreCheckpoint(out, q)` where `q == cached()` at snapshot time.
    pub fn checkpoint(self: *CudaLM, out: []u8) !void {
        std.debug.assert(out.len == self.checkpointBytes());
        const pos: u64 = self.pos_next;
        @memcpy(out[0..8], std.mem.asBytes(&pos));
        try self.checkpointState(.save, out[8..]);
    }

    /// Roll the context back to `q` committed tokens (a turn boundary) using a
    /// snapshot taken there. Rows past `q` in the append-only attention KV —
    /// device growables and the split's host shadow alike — are simply
    /// abandoned (the next write overwrites them), and the recurrent conv/ssm
    /// state is written back to each layer's CURRENT owner. The captured
    /// decode graph stays valid: it reads {token, len} and the M-RoPE triple
    /// from device state that is re-uploaded before every replay.
    pub fn restoreCheckpoint(self: *CudaLM, snap: []const u8, q: usize) !void {
        std.debug.assert(snap.len == self.checkpointBytes());
        std.debug.assert(q <= self.len);
        var pos: u64 = undefined;
        @memcpy(std.mem.asBytes(&pos), snap[0..8]);
        try self.checkpointState(.restore, snap[8..]);
        self.len = q;
        self.pos_next = @intCast(pos);
        if (self.split) |*sp| {
            sp.state.kv.truncate(q);
            sp.state.len = q;
        }
    }

    const CheckpointDir = enum { save, restore };

    /// Shared body of checkpoint/restoreCheckpoint: move every DeltaNet
    /// layer's conv+ssm slice between the snapshot buffer (`buf`, laid out as
    /// all conv states then all ssm states, lin-layer major — the same order
    /// as the device buffers and the split host arrays) and the layer's
    /// current owner.
    fn checkpointState(self: *CudaLM, comptime dir: CheckpointDir, buf: if (dir == .save) []u8 else []const u8) !void {
        const cfg = self.cfg;
        const n_lin = cfg.n_layers - cfg.nAttnLayers();
        const conv_n = cfg.convChannels() * (cfg.conv_kernel - 1);
        const ssm_n = cfg.lin_v_heads * cfg.lin_head_dim * cfg.lin_head_dim;
        const conv_all = buf[0 .. n_lin * conv_n * 4];
        const ssm_all = buf[n_lin * conv_n * 4 ..][0 .. n_lin * ssm_n * 4];
        for (0..cfg.n_layers) |l| {
            if (!cfg.isRecurrent(l)) continue;
            const lin_idx = l - l / cfg.full_attn_interval;
            const bc = conv_all[lin_idx * conv_n * 4 ..][0 .. conv_n * 4];
            const bs = ssm_all[lin_idx * ssm_n * 4 ..][0 .. ssm_n * 4];
            const on_gpu = if (self.split) |*sp| sp.on_gpu[l] else true;
            if (on_gpu) {
                const dc = offsetBufSized(self.conv_state, lin_idx * conv_n * 4, conv_n * 4);
                const ds = offsetBufSized(self.ssm_state, lin_idx * ssm_n * 4, ssm_n * 4);
                if (comptime dir == .save) {
                    try self.be.tensorDownload(dc, bc);
                    try self.be.tensorDownload(ds, bs);
                } else {
                    try self.be.tensorUpload(dc, bc);
                    try self.be.tensorUpload(ds, bs);
                }
            } else {
                const sp = &self.split.?;
                const hc = std.mem.sliceAsBytes(sp.state.conv[lin_idx * conv_n ..][0..conv_n]);
                const hs = std.mem.sliceAsBytes(sp.state.ssm[lin_idx * ssm_n ..][0..ssm_n]);
                if (comptime dir == .save) {
                    @memcpy(bc, hc);
                    @memcpy(bs, hs);
                } else {
                    @memcpy(hc, bc);
                    @memcpy(hs, bs);
                }
            }
        }
    }

    /// Forward `ids_new` (batched prefill for all but the last token, one
    /// decode step for it); `logits` receives the last position's LM head.
    /// Single-token decode replays a captured CUDA graph (one launch instead
    /// of ~1700) once the first decode step has warmed weight residency;
    /// --profile and capture failures fall back to per-op launches.
    pub fn step(self: *CudaLM, io: std.Io, ids_new: []const u32, logits: []f32) !void {
        std.debug.assert(logits.len == self.cfg.vocab);
        try self.forwardDecode(io, ids_new);
        try self.be.tensorDownload(offsetBufSized(self.bufs.logits, 0, self.cfg.vocab * 4), std.mem.sliceAsBytes(logits));
    }

    /// The decode forward, leaving the last position's logits resident in
    /// bufs.logits (graph replay when warm, else prefill + one step). Shared by
    /// step (downloads them), stepArgmax and stepSelect (sample on-device).
    fn forwardDecode(self: *CudaLM, io: std.Io, ids_new: []const u32) !void {
        self.io = io; // the CPU half of a hybrid split runs host matmuls through it
        const be = self.be;
        std.debug.assert(ids_new.len >= 1 and ids_new.len <= self.remaining());
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
    }

    /// Greedy decode: forward, then argmax the last logits on-device and return
    /// just the id (download 4 B, not the vocab). Matches sample.argmax.
    pub fn stepArgmax(self: *CudaLM, io: std.Io, ids_new: []const u32) !u32 {
        return self.stepArgmaxPen(io, ids_new, &.{}, .{});
    }

    /// `stepArgmax` with sampling penalties scattered onto the device logits
    /// first (opPenalize; see sample.zig) — keeps penalized greedy decode
    /// on the GPU path instead of the full-vocab download.
    pub fn stepArgmaxPen(self: *CudaLM, io: std.Io, ids_new: []const u32, pen: []const sample.PenaltyEntry, sp: sample.Params) !u32 {
        try self.forwardDecode(io, ids_new);
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
    pub fn stepSelect(self: *CudaLM, io: std.Io, ids_new: []const u32, out_id: []u32, out_logit: []f32) !usize {
        return self.stepSelectPen(io, ids_new, &.{}, .{}, out_id, out_logit);
    }

    /// `stepSelect` with sampling penalties scattered onto the device logits
    /// before the top-k (opPenalize) — the selected candidates are the true
    /// post-penalty top set, so penalized stochastic decode stays on the GPU.
    pub fn stepSelectPen(self: *CudaLM, io: std.Io, ids_new: []const u32, pen: []const sample.PenaltyEntry, sp: sample.Params, out_id: []u32, out_logit: []f32) !usize {
        try self.forwardDecode(io, ids_new);
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

        // Hybrid split: each chunk begins with n rows on the device (bufs.x).
        if (self.split) |*sp| sp.on_host = false;

        for (self.lm.layers, 0..) |*layer, l| {
            if (self.split) |*sp| {
                if (!sp.on_gpu[l]) {
                    if (!sp.on_host) {
                        try be.tensorDownload(offsetBufSized(b.x, 0, n * cfg.hidden * 4), std.mem.sliceAsBytes(sp.hx[0 .. n * cfg.hidden]));
                        sp.on_host = true;
                    }
                    const host_io = self.io orelse return error.SplitIoUnset;
                    try self.lm.cpuLayer(host_io, self.gpa, l, sp.hx[0 .. n * cfg.hidden], n, sp.freqs, &sp.state, &sp.scratch);
                    continue;
                }
                if (sp.on_host) {
                    try be.tensorUpload(offsetBufSized(b.x, 0, n * cfg.hidden * 4), std.mem.sliceAsBytes(sp.hx[0 .. n * cfg.hidden]));
                    sp.on_host = false;
                }
            }
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
                    try self.storeKv(self.k_cache[slot].buf, self.len * cfg.kvDim(), b.k, 0, n * cfg.kvDim());
                    try self.storeKv(self.v_cache[slot].buf, self.len * cfg.kvDim(), b.v, 0, n * cfg.kvDim());
                    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
                    if (debug_seq_attn) {
                        for (0..n) |t| {
                            try be.opAttnDecode(
                                offsetBufSized(b.q, t * cfg.qDim() * 4, cfg.qDim() * 4),
                                self.k_cache[slot].buf,
                                self.v_cache[slot].buf,
                                offsetBufSized(b.attn, t * cfg.qDim() * 4, cfg.qDim() * 4),
                                b.attn_scratch,
                                self.len + 1 + t,
                                1,
                                cfg.n_heads,
                                cfg.n_kv_heads,
                                hd,
                                nsplit,
                                scale,
                                0,
                                0,
                                false,
                                kvFmt(self.kv_dtype),
                            );
                        }
                    } else {
                        try be.opAttnDecode(b.q, self.k_cache[slot].buf, self.v_cache[slot].buf, b.attn, b.attn_scratch, self.len + 1, n, cfg.n_heads, cfg.n_kv_heads, hd, nsplit_prefill, scale, 0, 0, false, kvFmt(self.kv_dtype));
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
        // Split: advance the host KV in lockstep (the residual stream itself is
        // discarded — prefill reads no logits; the next chunk re-embeds anew).
        if (self.split) |*sp| {
            sp.state.kv.commit(n);
            sp.state.len += n;
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

        // Hybrid split: the hidden begins on the device (bufs.x). A CPU-resident
        // layer pulls it to the host once, runs there, and the next device layer
        // (if any) pushes it back — see enableCpuSplit. on_host tracks where the
        // live residual stream currently sits.
        if (self.split) |*sp| sp.on_host = false;

        for (self.lm.layers, 0..) |*layer, l| {
            if (self.split) |*sp| {
                if (!sp.on_gpu[l]) {
                    if (!sp.on_host) {
                        try be.tensorDownload(offsetBufSized(b.x, 0, cfg.hidden * 4), std.mem.sliceAsBytes(sp.hx[0..cfg.hidden]));
                        sp.on_host = true;
                    }
                    const host_io = self.io orelse return error.SplitIoUnset;
                    try self.lm.cpuLayer(host_io, self.gpa, l, sp.hx[0..cfg.hidden], 1, sp.freqs, &sp.state, &sp.scratch);
                    continue;
                }
                if (sp.on_host) {
                    try be.tensorUpload(offsetBufSized(b.x, 0, cfg.hidden * 4), std.mem.sliceAsBytes(sp.hx[0..cfg.hidden]));
                    sp.on_host = false;
                }
            }
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
                        const fmt = kvFmt(self.kv_dtype);
                        try be.opKvAppendS(self.k_cache[slot].buf, b.k, cfg.kvDim(), cfg.kvDim(), 0, fmt);
                        try be.opKvAppendS(self.v_cache[slot].buf, b.v, cfg.kvDim(), cfg.kvDim(), 0, fmt);
                        try be.opAttnDecodeSGraph(b.q, self.k_cache[slot].buf, self.v_cache[slot].buf, b.attn, b.attn_scratch, cfg.n_heads, cfg.n_kv_heads, hd, nsplit, scale, fmt);
                    } else {
                        try self.storeKv(self.k_cache[slot].buf, self.len * cfg.kvDim(), b.k, 0, cfg.kvDim());
                        try self.storeKv(self.v_cache[slot].buf, self.len * cfg.kvDim(), b.v, 0, cfg.kvDim());
                        try be.opAttnDecode(b.q, self.k_cache[slot].buf, self.v_cache[slot].buf, b.attn, b.attn_scratch, self.len + 1, 1, cfg.n_heads, cfg.n_kv_heads, hd, nsplit, scale, 0, 0, false, kvFmt(self.kv_dtype));
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

        // Split: the LM head runs on the device, so bring the hidden back if a
        // CPU tail left it on the host, and advance the host KV in lockstep.
        if (self.split) |*sp| {
            if (sp.on_host) {
                try be.tensorUpload(offsetBufSized(b.x, 0, cfg.hidden * 4), std.mem.sliceAsBytes(sp.hx[0..cfg.hidden]));
                sp.on_host = false;
            }
            sp.state.kv.commit(1);
            sp.state.len += 1;
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
                std.log.warn("decode graph capture failed ({t}); falling back to per-op launches", .{err});
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
        // That upload (~874 MB for the 27B) may itself have evicted LRU
        // weights under VRAM pressure — the capture would then re-upload
        // them mid-capture (cuMemAlloc, illegal while capturing). Bail to
        // per-op decode instead of starting a capture that must fail.
        if (be.evictions != 0) return error.WeightsEvicted;
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
        } else if (w.dtype == .f16) {
            try be.opGemvF16(y, x, w.bytes, w.scale, w.rows, w.cols);
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
    argmax_v: Buf,
    argmax_i: Buf,
    argmax_out: Buf,
    topk_v: Buf,
    topk_i: Buf,

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
            4096, // argmax_v (>= opArgmax lane count)
            4096, // argmax_i
            1, // argmax_out (1 id)
            cuda.backend.topk_lanes * cuda.backend.topk_m, // topk_v
            cuda.backend.topk_lanes * cuda.backend.topk_m, // topk_i
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

// Gated on -Dintegration + a CUDA device + the real 27B checkpoint: a greedy
// continuation after `restoreCheckpoint` must be TOKEN-IDENTICAL to the first
// greedy continuation from the same boundary — including after some layers
// migrate to the CPU between the two runs (the snapshot is owner-agnostic and
// restore writes to each layer's current owner). This is the engine-level
// guarantee behind the GUI's fast "regenerate response" rollback.
fn checkpointRestoreBody(kv_dtype: kvmod.KvDtype) !void {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const test_gate = @import("../test_gate.zig");
    const Gguf = @import("tp_core").gguf.Gguf;
    const path = "/home/qt/genai/lmstudio/models/llmfan46/Qwen3.6-27B-uncensored-heretic-v2-GGUF/Qwen3.6-27B-uncensored-heretic-v2-Q5_K_M.gguf";
    try test_gate.requireModelFile(io, path);
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    var g = try Gguf.open(gpa, io, path);
    defer g.deinit();
    var lm = try qwen35.Model.load(gpa, &g);
    defer lm.deinit();
    var model = try CudaLM.init(gpa, be, &lm, .{ .initial = 256, .max = 512, .kv_dtype = kv_dtype });
    defer model.deinit();

    // A fixed prompt of raw ids (no tokenizer needed — greedy determinism is
    // what's under test, not language). Prefill to the boundary, snapshot.
    var prompt: [48]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(1000 + i * 37);
    try model.prefill(prompt[0 .. prompt.len - 1]);
    const q = model.cached();
    const snap = try gpa.alloc(u8, model.checkpointBytes());
    defer gpa.free(snap);
    try model.checkpoint(snap);

    // Greedy continuation #1 (warms + captures the decode graph).
    const n_gen = 16;
    var seq1: [n_gen]u32 = undefined;
    var cur: u32 = prompt[prompt.len - 1];
    for (&seq1) |*t| {
        cur = try model.stepArgmax(io, &.{cur});
        t.* = cur;
    }

    // Roll back and regenerate: must replay the graph against the truncated
    // len and the restored recurrent state, bit-identically.
    try model.restoreCheckpoint(snap, q);
    try std.testing.expectEqual(q, model.cached());
    var seq2: [n_gen]u32 = undefined;
    cur = prompt[prompt.len - 1];
    for (&seq2) |*t| {
        cur = try model.stepArgmax(io, &.{cur});
        t.* = cur;
    }
    try std.testing.expectEqualSlices(u32, &seq1, &seq2);

    // Arm a CPU split MID-conversation and migrate a slice of layers to the
    // host, then restore the SAME snapshot (taken while fully device-resident):
    // restore must write each layer's state to its NEW owner. Hybrid CPU/GPU
    // decode is not bit-identical to all-GPU decode (host f32 vs device dp4a),
    // so assert REPEATABILITY here — two restored continuations must agree.
    try model.enableCpuSplit(.tail, std.math.maxInt(u64), true);
    try model.offloadToBudget(be.deviceUsed() * 4 / 5);
    try model.restoreCheckpoint(snap, q);
    var seq3: [n_gen]u32 = undefined;
    cur = prompt[prompt.len - 1];
    for (&seq3) |*t| {
        cur = try model.stepArgmax(io, &.{cur});
        t.* = cur;
    }
    try model.restoreCheckpoint(snap, q);
    var seq4: [n_gen]u32 = undefined;
    cur = prompt[prompt.len - 1];
    for (&seq4) |*t| {
        cur = try model.stepArgmax(io, &.{cur});
        t.* = cur;
    }
    try std.testing.expectEqualSlices(u32, &seq3, &seq4);

    // Promote everything back onto the device and restore once more: the
    // recurrent state round-trips host→device, so an owner-misdirected
    // restore would now surface as divergence from the all-GPU baseline —
    // which this continuation must match bit-for-bit again.
    _ = try model.promoteLayers(std.math.maxInt(u64));
    try model.restoreCheckpoint(snap, q);
    var seq5: [n_gen]u32 = undefined;
    cur = prompt[prompt.len - 1];
    for (&seq5) |*t| {
        cur = try model.stepArgmax(io, &.{cur});
        t.* = cur;
    }
    try std.testing.expectEqualSlices(u32, &seq1, &seq5);
}

test "checkpoint restore regenerates token-identical on the real model" {
    try checkpointRestoreBody(.f32);
}

// Same workout on an f16 KV cache: the split's host shadow stores packed f16
// (byte-identical to the device caches, slot-indexed like the device), so the
// mid-conversation migrate and the promote round trip must stay exact —
// token-identical within the f16 session.
test "checkpoint restore regenerates token-identical on the real model (f16 kv)" {
    try checkpointRestoreBody(.f16);
}

// And on a q8_0 KV cache: attention KV quantized to ggml blocks (conv/ssm
// recurrent state stays f32), migrate/promote raw copies stay lossless.
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
    const path = "/home/qt/genai/lmstudio/models/llmfan46/Qwen3.6-27B-uncensored-heretic-v2-GGUF/Qwen3.6-27B-uncensored-heretic-v2-Q5_K_M.gguf";
    try test_gate.requireModelFile(io, path);
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    var g = try Gguf.open(gpa, io, path);
    defer g.deinit();
    var lm = try qwen35.Model.load(gpa, &g);
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
    const path = "/home/qt/genai/lmstudio/models/llmfan46/Qwen3.6-27B-uncensored-heretic-v2-GGUF/Qwen3.6-27B-uncensored-heretic-v2-Q5_K_M.gguf";
    try test_gate.requireModelFile(io, path);
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    var g = try Gguf.open(gpa, io, path);
    defer g.deinit();
    var lm = try qwen35.Model.load(gpa, &g);
    defer lm.deinit();
    var model = try CudaLM.init(gpa, be, &lm, .{ .initial = 128, .max = 256 });
    defer model.deinit();

    // A budget below the weight total statically places tail layers on the
    // host right away (kept small so the Debug host prefill stays quick).
    try model.enableCpuSplit(.tail, 17 << 30, true);
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
