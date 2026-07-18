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
const kvmod = @import("../llm/kv_cache.zig");
const residency = @import("residency.zig");

const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;
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
    io: std.Io = undefined,
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
        const esz = cap.kv_dtype.elemBytes();
        self.k_cache = try alloc.alloc(Growable, n_attn);
        self.v_cache = try alloc.alloc(Growable, n_attn);
        for (self.k_cache, self.v_cache) |*kb, *vb| {
            kb.* = try be.growableCreate(cap.initial * cfg.kvDim() * esz, cap.max * cfg.kvDim() * esz);
            vb.* = try be.growableCreate(cap.initial * cfg.kvDim() * esz, cap.max * cfg.kvDim() * esz);
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
    /// `dst` at element offset `dst_off`. f32 copies raw; f16 converts on store.
    fn storeKv(self: *CudaLM, dst: Buf, dst_off: usize, src: Buf, src_off: usize, n: usize) !void {
        if (self.kv_dtype == .f16) {
            try self.be.opStoreKvF16(dst, dst_off, src, src_off, n);
        } else {
            try self.be.tensorCopy(dst, dst_off * 4, src, src_off * 4, n * 4);
        }
    }

    /// Rebuild the KV cache at a new element dtype (GUI toggle), weights resident.
    /// Drops the captured decode graph so it re-captures with the f16/f32 kernels,
    /// frees + re-creates the per-attention-slot K/V buffers, resets the length.
    /// f16 with an active CPU-offload split is unsupported (host shadow is f32).
    pub fn reinitCache(self: *CudaLM, dtype: kvmod.KvDtype) !void {
        if (dtype == .f16 and self.split != null) return error.KvDtypeUnsupported;
        const be = self.be;
        const cfg = self.cfg;
        if (self.graph_exec != null) {
            be.graphDestroy(self.graph_exec);
            self.graph_exec = null;
        }
        self.decode_warm = false;
        for (self.k_cache) |*b| be.growableDestroy(b);
        for (self.v_cache) |*b| be.growableDestroy(b);
        self.kv_dtype = dtype;
        self.capacity = self.initial_capacity;
        self.len = 0;
        const esz = dtype.elemBytes();
        for (self.k_cache, self.v_cache) |*kb, *vb| {
            kb.* = try be.growableCreate(self.initial_capacity * cfg.kvDim() * esz, self.max_capacity * cfg.kvDim() * esz);
            vb.* = try be.growableCreate(self.initial_capacity * cfg.kvDim() * esz, self.max_capacity * cfg.kvDim() * esz);
        }
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
        // Element stride MUST match how the buffers were created (esz), or an f16
        // cache (esz=2) requests f32-sized growth, overshoots its VA reservation,
        // and growableEnsure fails with DeviceOutOfMemory → ContextFull once the
        // window grows past ~max_capacity/2.
        const bytes = target * self.cfg.kvDim() * self.kv_dtype.elemBytes();

        // Dynamic offload: migrate layers GPU->CPU until there's headroom to
        // grow the device KV — instead of streaming weights (the cliff). Each
        // migrated attention layer frees its device KV immediately; linear
        // layers free weight VRAM via the cache. Migrate attention-first (the
        // policy's order) so headroom recovers per step.
        if (self.split) |*sp| if (sp.dynamic) {
            const add = (target - self.capacity) * self.cfg.kvDim() * 4;
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
            const s = l / cfg.full_attn_interval;
            const kvd = cfg.kvDim();
            if (self.len > 0) {
                const cap = sp.state.kv.capacity;
                const hk = sp.state.kv.k[s * cap * kvd ..][0 .. self.len * kvd];
                const hv = sp.state.kv.v[s * cap * kvd ..][0 .. self.len * kvd];
                try self.be.tensorDownload(offsetBufSized(self.k_cache[s].buf, 0, self.len * kvd * 4), std.mem.sliceAsBytes(hk));
                try self.be.tensorDownload(offsetBufSized(self.v_cache[s].buf, 0, self.len * kvd * 4), std.mem.sliceAsBytes(hv));
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
        std.log.info("[offload] layer {d} ({s}) -> CPU at ctx {d} ({d}/{d} on CPU)", .{ l, if (cfg.isRecurrent(l)) "lin" else "attn", self.len, sp.n_cpu, cfg.n_layers });
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

        // Device memory that is not streamable layer weight and must stay
        // resident: KV at the current capacity, the LM head, plus slack for
        // activation scratch, the RoPE table, and CUDA context / fragmentation.
        const kv_bytes = 2 * cfg.nAttnLayers() * self.capacity * cfg.kvDim() * 4;
        // Dynamic mode packs as many layers as possible onto the GPU up front
        // (just the LM head reserved) and migrates the rest on demand as the KV
        // cache grows past the current headroom. Static mode can't adapt later,
        // so it reserves generously (KV at capacity + head + slack).
        const reserve = if (dynamic)
            self.lm.head.bytes.len
        else
            kv_bytes + self.lm.head.bytes.len + (512 << 20);
        const gpu_weight_budget: usize = if (budget > reserve) budget - reserve else 0;

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
        // capacity; grows with the device via ensureCapacity). len starts at
        // 0 — enableCpuSplit runs before any tokens.
        // Offloaded-layer host state is always f32 (f16 disables the split).
        var state = try qwen35.State.init(gpa, cfg, self.capacity, .f32);
        errdefer state.deinit(gpa);
        var scratch = try qwen35.Scratch.init(gpa, prefill_chunk, cfg);
        errdefer scratch.deinit(gpa);
        var freqs = try ops.rope.rotateHalfFreqs(gpa, self.capacity, cfg.rope_dim, cfg.rope_theta);
        errdefer freqs.deinit(gpa);
        const hx = try gpa.alloc(f32, prefill_chunk * cfg.hidden);
        errdefer gpa.free(hx);

        self.split = .{
            .on_gpu = on_gpu,
            .n_cpu = n_cpu,
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

        // Free the device KV of attention layers placed on the host up front
        // (len is 0 here, so there is nothing to copy). Weights are reclaimed
        // lazily by the cache; conv/ssm start zeroed on both sides.
        for (order[0..n_cpu]) |l| {
            if (!cfg.isRecurrent(l)) {
                const s = l / cfg.full_attn_interval;
                self.be.growableDestroy(&self.k_cache[s]);
                self.be.growableDestroy(&self.v_cache[s]);
            }
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
        // f16 KV stays fully resident (host shadow cache is f32); f16 already
        // halves the KV footprint, lowering the pressure that would offload.
        if (self.kv_dtype == .f16) return false;
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
        const kv_cost: usize = if (!self.cfg.isRecurrent(l)) 2 * self.capacity * self.cfg.kvDim() * 4 else 0;
        return layerDeviceBytes(&self.lm.layers[l]) + kv_cost + (64 << 20);
    }

    pub fn promoteLayer(self: *CudaLM, l: usize) !void {
        const cfg = self.cfg;
        const sp = &self.split.?;
        if (!cfg.isRecurrent(l)) {
            const s = l / cfg.full_attn_interval;
            const kvd = cfg.kvDim();
            self.k_cache[s] = try self.be.growableCreate(self.capacity * kvd * 4, self.max_capacity * kvd * 4);
            self.v_cache[s] = try self.be.growableCreate(self.capacity * kvd * 4, self.max_capacity * kvd * 4);
            if (self.len > 0) {
                const cap = sp.state.kv.capacity;
                const hk = sp.state.kv.k[s * cap * kvd ..][0 .. self.len * kvd];
                const hv = sp.state.kv.v[s * cap * kvd ..][0 .. self.len * kvd];
                try self.be.tensorUpload(offsetBufSized(self.k_cache[s].buf, 0, self.len * kvd * 4), std.mem.sliceAsBytes(hk));
                try self.be.tensorUpload(offsetBufSized(self.v_cache[s].buf, 0, self.len * kvd * 4), std.mem.sliceAsBytes(hv));
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
        std.log.info("[promote] layer {d} ({s}) -> GPU at ctx {d} ({d}/{d} on CPU)", .{ l, if (cfg.isRecurrent(l)) "lin" else "attn", self.len, sp.n_cpu, cfg.n_layers });
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
        for (self.k_cache, self.v_cache) |*kb, *vb| {
            self.be.growableDestroy(kb);
            self.be.growableDestroy(vb);
            kb.* = try self.be.growableCreate(self.initial_capacity * kvd * 4, self.max_capacity * kvd * 4);
            vb.* = try self.be.growableCreate(self.initial_capacity * kvd * 4, self.max_capacity * kvd * 4);
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
        try self.forwardDecode(io, ids_new);
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
    pub fn stepSelect(self: *CudaLM, io: std.Io, ids_new: []const u32, out_id: []u32, out_logit: []f32) !usize {
        try self.forwardDecode(io, ids_new);
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
                    try self.lm.cpuLayer(self.io, self.gpa, l, sp.hx[0 .. n * cfg.hidden], n, sp.freqs, &sp.state, &sp.scratch);
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
                            );
                        }
                    } else {
                        try be.opAttnDecode(b.q, self.k_cache[slot].buf, self.v_cache[slot].buf, b.attn, b.attn_scratch, self.len + 1, n, cfg.n_heads, cfg.n_kv_heads, hd, nsplit_prefill, scale, 0, 0, self.kv_dtype == .f16);
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
                    try self.lm.cpuLayer(self.io, self.gpa, l, sp.hx[0..cfg.hidden], 1, sp.freqs, &sp.state, &sp.scratch);
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
                        const kv_f16 = self.kv_dtype == .f16;
                        try be.opKvAppendS(self.k_cache[slot].buf, b.k, cfg.kvDim(), cfg.kvDim(), 0, kv_f16);
                        try be.opKvAppendS(self.v_cache[slot].buf, b.v, cfg.kvDim(), cfg.kvDim(), 0, kv_f16);
                        try be.opAttnDecodeSGraph(b.q, self.k_cache[slot].buf, self.v_cache[slot].buf, b.attn, b.attn_scratch, cfg.n_heads, cfg.n_kv_heads, hd, nsplit, scale, kv_f16);
                    } else {
                        try self.storeKv(self.k_cache[slot].buf, self.len * cfg.kvDim(), b.k, 0, cfg.kvDim());
                        try self.storeKv(self.v_cache[slot].buf, self.len * cfg.kvDim(), b.v, 0, cfg.kvDim());
                        try be.opAttnDecode(b.q, self.k_cache[slot].buf, self.v_cache[slot].buf, b.attn, b.attn_scratch, self.len + 1, 1, cfg.n_heads, cfg.n_kv_heads, hd, nsplit, scale, 0, 0, self.kv_dtype == .f16);
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
