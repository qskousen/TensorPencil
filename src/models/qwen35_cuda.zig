//! Qwen3.5/3.6 hybrid LM on the CUDA backend (tp-llm --backend zig-cuda /
//! cuda): the 64-layer gated-DeltaNet + gated-attention stack runs
//! device-resident. Every projection goes through lin_cuda, which picks the
//! kernel per weight and row count; after the first decode step the whole
//! forward replays as one captured CUDA graph (stepDecodeGraph). Prefill runs
//! batched chunks (stepBatch).
//! The 27B Q5_K_M fits a 24 GB card resident. Speculative decoding is
//! unsupported (recurrent state cannot roll back).

const std = @import("std");
const init_defaults = @import("tp_core").init_defaults;
extern fn getenv(name: [*:0]const u8) ?[*:0]const u8;
const qwen35 = @import("qwen35.zig");
const qwen35_dev = @import("qwen35_dev.zig");
const lin_cuda = @import("lin_cuda.zig");
const qwen3 = @import("qwen3.zig");
const cuda = @import("tp_gpu").cuda;
const ops = @import("tp_ops");
const kvmod = @import("tp_core").kv_cache;
const sample = @import("tp_core").sample;
const residency = @import("tp_runtime").residency;
const bnd = @import("tp_runtime").boundary;

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

/// Whether the captured-decode-graph path is available for this embedding dtype
/// (the graph needs a device-side embedding gather kernel). Used at init and to
/// restore graph_ok after a split is dropped (resetResidency).
fn graphOkFor(dtype: anytype) bool {
    return switch (dtype) {
        .bf16, .q8_0, .q4_k, .q5_k, .q6_k, .iq4_xs => true,
        else => false,
    };
}

/// KV chunks per head in the decode attention split pass.
const nsplit = 32;
/// Batched-prefill chunk (rows per stepBatch) and its attention split count
/// (bounds the flash-decode scratch).
///
/// 256, not 128: the MMQ prefill kernel tiles 128 tokens x 128 rows per block, so
/// a 128-token chunk launches only `rows/128` blocks, 136 for the widest FFN,
/// under two waves on a 3090's 82 SMs, so a large part of each launch is tail. The
/// same change is worth +46% on gemma4 without MMQ. It costs activation
/// scratch, which `Bufs.pad_rows` sizes from this constant rather than a second
/// hardcoded 128.
const prefill_chunk = 512;
const nsplit_prefill = 8;

/// Rows every per-chunk activation buffer is sized for: `opMatmulQuant` and the
/// MMQ kernels always write `align(m, 128)` rows, so a chunk-sized buffer would
/// let the pad rows overflow into the next one. Also the row stride the batched
/// GDN uses to place `beta` after `alpha` in `Bufs.ab_n`.
const pad_rows = std.mem.alignForward(usize, prefill_chunk, 128);

/// `opGdnDeltaChunk`'s limits: a chunk of more than one row, a head dim it tiles.
fn gdnChunkOk(d: usize, n: usize) bool {
    return n > 1 and d % 32 == 0 and d <= 256;
}

pub const CpuSplitPolicy = residency.CpuSplitPolicy;

/// DIAGNOSTIC: run the recurrence and the non-graph attention through the dual kernels
/// Vulkan runs (`dual/attn.zig`) instead of this backend's own, for the A/B that decides
/// which this arm keeps. `tp-llm --bench` alternates it per round (`bench_toggle`).
pub var dual_kernels = false;

pub const CudaLM = struct {
    pub const bench_toggle = &dual_kernels;

    /// The prefill batch this stepper wants the engine to hand it. Tied to
    /// `prefill_chunk` so the two cannot drift: the engine's old hardcoded 256 capped
    /// this silently, and the mismatch cost a measured 31% on the matmul bucket
    /// (bigger buffers, same 255-token work). See `engine.generate`'s prefill loop.
    pub const prefill_batch = prefill_chunk;

    lm: *const qwen35.Model,
    /// What `lin_cuda.plan` decided for this checkpoint, passed to every linear.
    lin_plan: lin_cuda.Plan,
    be: *Backend,
    gpa: std.mem.Allocator,
    cfg: qwen35.Config,
    /// Committed KV rows; grows in place toward max_capacity (ensureCapacity).
    capacity: usize,
    /// Growth ceiling, the VA reservation behind each KV cache and the RoPE
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
    /// Per linear layer, `alpha` and `beta` stacked into one weight padded to the
    /// MMQ row tile: rows 0..v_heads-1 alpha, v_heads..2*v_heads-1 beta, the rest
    /// zero blocks. Null when the dtype has no MMQ kernel or the shape does not
    /// divide. See `ab_stride` for why this exists.
    ab_cat: []?ops.matmul.Weight,
    /// TP_DEBUG_BATCH: per-layer hidden-state dump of the last processed row
    /// ([n_layers][hidden]), filled by stepHidden/stepBatch when set.
    layer_dump: ?[]f32 = null,
    /// TP debug: layer-0 input_layernorm output, one hidden row per
    /// processed token (seq appends at op_dump_row; batch writes all rows).
    op_dump: ?[]f32 = null,
    op_dump_row: usize = 0,
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
    /// (no per-token PCIe re-stream). Forces the per-op decode path, a
    /// captured graph cannot record host compute.
    split: ?Split = null,
    /// The frontend's turn hook, run between prefill chunks: enact a
    /// coordinator-published residency ceiling, park on a pause, stop on a
    /// cancel. The decode loop reaches the same three through
    /// `engine.checkpoint`; prefill never does, so without this a turn's slow
    /// half runs uncoordinated and unstoppable. null on the CLI.
    boundary: ?bnd.Hook = null,
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
        /// Host RoPE table, grows with capacity (scalar positions, text only).
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
        return initUse(gpa, be, lm, cap, lin_cuda.Use.llm(prefill_chunk));
    }

    /// `init` with the linear routes' policy given, for a caller that is not a chat
    /// session (a text encoder runs the encoder dial's integer arithmetic).
    pub fn initUse(gpa: std.mem.Allocator, be: *Backend, lm: *const qwen35.Model, cap: kvmod.Capacity, use: lin_cuda.Use) !CudaLM {
        const cfg = lm.cfg;
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();
        const lin_plan = try lin_cuda.plan(try lm.deviceLins(alloc), use, "qwen35 cuda");

        var self: CudaLM = init_defaults.of(CudaLM);
        self.lm = lm;
        self.lin_plan = lin_plan;
        self.be = be;
        self.gpa = gpa;
        self.cfg = cfg;
        be.weight_noise.setDepth(cfg.n_layers);
        self.capacity = cap.initial;
        self.initial_capacity = cap.initial;
        self.max_capacity = cap.max;
        self.kv_dtype = cap.kv_dtype;
        self.len = 0;
        self.pos_next = 0;
        self.layer_dump = null;
        self.op_dump = null;
        self.op_dump_row = 0;
        self.graph_exec = null;
        self.split = null;
        self.io = null; // field defaults do not apply to `undefined`-built structs
        self.decode_warm = false;
        // The graph path needs a device-side embedding gather kernel.
        self.graph_ok = graphOkFor(lm.embed.dtype);

        // Rope table for the rotated span (rope_dim), like qwen3_cuda's,
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
        self.ab_cat = try alloc.alloc(?ops.matmul.Weight, n_lin);
        var lin_idx: usize = 0;
        for (lm.layers) |*layer| {
            switch (layer.*) {
                .linear => |*ll| {
                    const buf = try alloc.alloc(f32, 2 * cfg.lin_v_heads);
                    @memcpy(buf[0..cfg.lin_v_heads], ll.a);
                    @memcpy(buf[cfg.lin_v_heads..], ll.dt_bias);
                    self.a_dt[lin_idx] = buf;
                    self.ab_cat[lin_idx] = null;
                    // Stack alpha over beta and pad to the MMQ row tile. At 48 rows
                    // each these are far below one GEMM tile, so the alternative is
                    // `n/8` grouped-GEMV launches of a SIX-block grid, which on an
                    // 82-SM card is all latency and no occupancy: MEASURED 26 us per
                    // launch and 16% of prefill, against ~0 for one launch.
                    // Rows past 2*v_heads are all-zero blocks, which every k-quant
                    // dequantizes to 0, so they contribute nothing.
                    const ab_rows = Backend.mmq_pipe_tile;
                    if (ll.alpha.dtype == ll.beta.dtype and ll.alpha.scale == ll.beta.scale and
                        ll.alpha.row_scale == null and ll.beta.row_scale == null and
                        2 * cfg.lin_v_heads <= ab_rows and
                        Backend.mmqPipeFaster(ll.alpha.dtype, ab_rows, cfg.hidden))
                    {
                        const row_bytes = ll.alpha.dtype.storageBytes(cfg.hidden);
                        const cat = try alloc.alloc(u8, ab_rows * row_bytes);
                        @memset(cat, 0);
                        @memcpy(cat[0 .. cfg.lin_v_heads * row_bytes], ll.alpha.bytes[0 .. cfg.lin_v_heads * row_bytes]);
                        @memcpy(cat[cfg.lin_v_heads * row_bytes ..][0 .. cfg.lin_v_heads * row_bytes], ll.beta.bytes[0 .. cfg.lin_v_heads * row_bytes]);
                        self.ab_cat[lin_idx] = .{ .bytes = cat, .dtype = ll.alpha.dtype, .rows = ab_rows, .cols = cfg.hidden, .scale = ll.alpha.scale };
                    }
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
        // idempotent), only GPU-resident slots are re-created at the new
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
        // state and pos 0, stale state would double-apply the transcript
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
    /// Device VRAM (bytes) currently in use by this backend, for the
    /// end-of-response telemetry (`session.statsOf`).
    pub fn vramUsed(self: *const CudaLM) u64 {
        return self.be.deviceUsed();
    }

    /// Commit more KV rows, in place: device pointers (and the captured
    /// decode graph, sin_off and the KV strides are capacity-independent)
    /// stay valid. Under VRAM pressure the commit evicts LRU weights into
    /// the streaming path, which flips the graph off via the evictions guard
    /// in step(). error.ContextFull past the window or when even eviction
    /// can't free enough device memory.
    pub fn ensureCapacity(self: *CudaLM, min_rows: usize) !void {
        const target = (try kvmod.growPlan(self.capacity, self.max_capacity, min_rows)) orelse return;
        // Byte size MUST match how the buffers were created (kv_dtype block
        // math), or an f16/q8_0 cache requests f32-sized growth, overshoots its
        // VA reservation, and growableEnsure fails with DeviceOutOfMemory ->
        // ContextFull once the window grows past ~max_capacity/2.
        const bytes = self.kv_dtype.sizeBytes(target * self.cfg.kvDim());

        // Dynamic offload: migrate layers GPU->CPU until there's headroom to
        // grow the device KV, instead of streaming weights (the cliff). Each
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
        // ever fails once nothing is left to migrate, never a premature
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
                try self.be.tensorDownload(self.k_cache[s].buf.slice(0, dt.sizeBytes(self.len * kvd)), sp.state.kv.kRowBytes(s, 0, self.len));
                try self.be.tensorDownload(self.v_cache[s].buf.slice(0, dt.sizeBytes(self.len * kvd)), sp.state.kv.vRowBytes(s, 0, self.len));
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
            try self.be.tensorDownload(self.conv_state.slice(lin_idx * conv_n * 4, conv_n * 4), std.mem.sliceAsBytes(hc));
            try self.be.tensorDownload(self.ssm_state.slice(lin_idx * ssm_n * 4, ssm_n * 4), std.mem.sliceAsBytes(hs));
        }
        // Free the migrated layer's device weights (ggml reads them from host
        // now), this is the bulk of the reclaimed VRAM.
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
    /// bytes, as uploaded), the big projection + MLP matrices; norms/conv
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

    /// Upload every device-resident weight NOW, so the one-time host->device copy
    /// is not charged to the first forward. Mirrors `layerDeviceBytes` exactly,
    /// the same tensors it counts are the ones warmed here, so the two cannot
    /// drift. See `Backend.warmWeight` for why this exists.
    ///
    /// Skips layers the CPU split placed on the host (their weights never go to
    /// the device) and is best-effort throughout: anything that will not fit still
    /// works, it just pays the copy on first use as before.
    pub fn warmWeights(self: *CudaLM) void {
        const be = self.be;
        be.warmWeight(self.lm.head.bytes);
        if (self.lm.embed.bytes.ptr != self.lm.head.bytes.ptr) be.warmWeight(self.lm.embed.bytes);
        for (self.lm.layers, 0..) |*layer, l| {
            if (self.split) |*sp| if (!sp.on_gpu[l]) continue;
            const mlp = switch (layer.*) {
                .attn => |*a| &a.mlp,
                .linear => |*ll| &ll.mlp,
            };
            be.warmWeight(mlp.gate.bytes);
            be.warmWeight(mlp.up.bytes);
            be.warmWeight(mlp.down.bytes);
            switch (layer.*) {
                .attn => |*a| {
                    be.warmWeight(a.qg.bytes);
                    be.warmWeight(a.k.bytes);
                    be.warmWeight(a.v.bytes);
                    be.warmWeight(a.o.bytes);
                },
                .linear => |*ll| {
                    be.warmWeight(ll.qkv.bytes);
                    be.warmWeight(ll.z.bytes);
                    be.warmWeight(ll.alpha.bytes);
                    be.warmWeight(ll.beta.bytes);
                    be.warmWeight(ll.out.bytes);
                },
            }
        }
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
        for (self.lm.layers, 0..) |*layer, l| per[l] = layerDeviceBytes(layer);

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
        // Eviction order: which layers leave the device first. Kept in the
        // Split (dynamic migration walks it as the KV cache grows).
        const order = try gpa.alloc(usize, n);
        errdefer gpa.free(order);
        switch (policy) {
            .tail => for (0..n) |i| {
                order[i] = n - 1 - i; // last layer first -> contiguous device prefix
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
        const n_cpu = residency.splitHostCount(self, budget, reserve, dynamic, per, order);
        if (n_cpu == 0 and !dynamic) {
            gpa.free(on_gpu);
            gpa.free(order);
            return; // everything fits resident, no split needed
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

        // Place the statically-planned layers on the host. `migrateLayer` copies
        // any live rows (none before the first token), frees the device K/V AND
        // evicts the layer's device weights, which is the only thing that
        // actually reclaims VRAM here: `warmWeights` uploads and PINS every
        // weight during load, so a layer merely marked host-resident keeps its
        // full footprint on the card while computing on the CPU.
        for (order[0..n_cpu]) |l| try self.migrateLayer(l);
    }

    pub fn vocab(self: *const CudaLM) usize {
        return self.cfg.vocab;
    }

    /// Arm dynamic CPU offload iff the projected max-context footprint (all layer
    /// weights + KV grown to `max_capacity` + reserve slack) won't fit under
    /// `budget` bytes. When it fits, do nothing, the model stays fully resident
    /// on the fast path. When it doesn't, `enableCpuSplit(.attn, budget, dynamic)`
    /// packs the GPU now and migrates layers to the host as the KV cache grows
    /// (instead of the weight-streaming cliff). Must run before any tokens.
    /// `budget == 0` is treated as "no limit" (never offload). Text sessions
    /// only, the host layer path uses scalar RoPE (see enableCpuSplit / the GUI
    /// guard). Returns whether a split was armed.
    pub fn autoOffload(self: *CudaLM, budget: u64) !bool {
        if (budget == 0) return false; // no budget -> fully resident, no offload path
        // Always arm the dynamic split. When the model fits its budget it places
        // ZERO layers on the CPU, measured free (per-op decode ties the captured
        // graph: 76.2 vs 77.2 tok/s on the 9B), and migrates layers on demand as
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
    /// iteration as liveAttnSlots drops, a fixed target here can't express that.)
    pub fn offloadUntilFree(self: *CudaLM, needed_free: u64) !void {
        return residency.offloadUntilFree(self, needed_free);
    }

    /// Migrate layers to the host until the LLM's ACTUAL total device usage is at
    /// or under `target` bytes, used by the GUI's `balanced` mode to settle the
    /// LLM to its share (e.g. 75% of the limit) ONLY when an image model loads
    /// (contention); with no image loaded the LLM keeps the whole limit. Uses live
    /// `deviceUsed()` (robust, no size estimation), so it must run after the
    /// weights are on the device. One-way + idempotent: once under `target` it's a
    /// no-op, so repeat calls (per image) don't re-shuffle.
    pub fn offloadToBudget(self: *CudaLM, target: u64) !void {
        return residency.offloadToBudget(self, target);
    }

    /// `residency.demand` hook: this layer's weight bytes alone (no KV). Lets the
    /// demand estimate separate "weights not uploaded yet", which the backend's
    /// pinned-bytes counter already accounts for globally, from the device KV a
    /// promote has to re-create.
    pub fn layerWeightBytes(self: *CudaLM, l: usize) usize {
        return layerDeviceBytes(&self.lm.layers[l]);
    }

    /// `residency.demand` hook: device-resident weights that are not per-layer.
    /// A tied head/embedding is ONE cached device buffer (the weight cache keys
    /// on the byte slice), so count it once.
    pub fn nonLayerDeviceBytes(self: *CudaLM) u64 {
        const h = self.lm.head.bytes;
        const e = self.lm.embed.bytes;
        return h.len + if (e.ptr == h.ptr) 0 else e.len;
    }

    /// `residency.promoteBack` cost hook: VRAM a promote of layer `l` needs, its
    /// streamable weights, the KV it re-commits at capacity (attention layers
    /// only; recurrent layers hold fixed-size conv/ssm state, no KV), plus slack.
    pub fn promoteCost(self: *CudaLM, l: usize) usize {
        const kv_cost: usize = if (!self.cfg.isRecurrent(l)) 2 * self.kv_dtype.sizeBytes(self.capacity * self.cfg.kvDim()) else 0;
        return layerDeviceBytes(&self.lm.layers[l]) + kv_cost + residency.promote_slack;
    }

    /// Bring layer `l` back onto the GPU, preserving its accumulated state
    /// (reverse of migrateLayer): attention layers re-create the device K/V at the
    /// current capacity and upload the host rows [0,len); linear layers re-upload
    /// their conv/ssm state into the shared device buffers. Weights re-cache
    /// lazily on the next GPU forward.
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
                try self.be.tensorUpload(self.k_cache[s].buf.slice(0, dt.sizeBytes(self.len * kvd)), sp.state.kv.kRowBytes(s, 0, self.len));
                try self.be.tensorUpload(self.v_cache[s].buf.slice(0, dt.sizeBytes(self.len * kvd)), sp.state.kv.vRowBytes(s, 0, self.len));
            }
        } else {
            const lin_idx = l - l / cfg.full_attn_interval;
            const conv_n = cfg.convChannels() * (cfg.conv_kernel - 1);
            const ssm_n = cfg.lin_v_heads * cfg.lin_head_dim * cfg.lin_head_dim;
            const hc = sp.state.conv[lin_idx * conv_n ..][0..conv_n];
            const hs = sp.state.ssm[lin_idx * ssm_n ..][0..ssm_n];
            try self.be.tensorUpload(self.conv_state.slice(lin_idx * conv_n * 4, conv_n * 4), std.mem.sliceAsBytes(hc));
            try self.be.tensorUpload(self.ssm_state.slice(lin_idx * ssm_n * 4, ssm_n * 4), std.mem.sliceAsBytes(hs));
        }
        sp.on_gpu[l] = true;
        sp.n_cpu -= 1;
        std.log.debug("[promote] layer {d} ({s}) -> GPU at ctx {d} ({d}/{d} on CPU)", .{ l, if (cfg.isRecurrent(l)) "lin" else "attn", self.len, sp.n_cpu, cfg.n_layers });
    }

    /// Migrate CPU layers back onto the GPU (LIFO by offload order), stopping
    /// before the next would overflow `budget`, so the caller (VRAM coordinator,
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

    /// Reset the session to an empty context: KV rows are overwritten lazily, so only
    /// the position counters and the conv/recurrent states, on either side of a split,
    /// need clearing.
    pub fn resetCache(self: *CudaLM) !void {
        const cfg = self.cfg;
        const n_lin = cfg.n_layers - cfg.nAttnLayers();
        try zeroBuffer(self.be, self.gpa, self.conv_state, n_lin * cfg.convChannels() * (cfg.conv_kernel - 1) * 4);
        try zeroBuffer(self.be, self.gpa, self.ssm_state, n_lin * cfg.lin_v_heads * cfg.lin_head_dim * cfg.lin_head_dim * 4);
        if (self.split) |*sp| {
            @memset(sp.state.conv, 0);
            @memset(sp.state.ssm, 0);
            sp.state.kv.truncate(0);
            sp.state.len = 0;
        }
        self.len = 0;
        self.pos_next = 0;
    }

    /// Fixed byte size of a turn checkpoint (see `checkpoint`): the M-RoPE
    /// position plus every DeltaNet layer's conv + ssm recurrent state.
    /// Independent of context length, attention KV is append-only, so a
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
    /// snapshot taken there. Rows past `q` in the append-only attention KV,
    /// device growables and the split's host shadow alike, are simply
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
    /// all conv states then all ssm states, lin-layer major, the same order
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
                const dc = self.conv_state.slice(lin_idx * conv_n * 4, conv_n * 4);
                const ds = self.ssm_state.slice(lin_idx * ssm_n * 4, ssm_n * 4);
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
        try self.be.tensorDownload(self.bufs.logits.slice(0, self.cfg.vocab * 4), std.mem.sliceAsBytes(logits));
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
        // A captured graph freezes the noise key and sigma in its kernel parameters,
        // so every replay would draw the same perturbation.
        if (ids_new.len == 1 and self.graph_ok and !be.profile and !be.weight_noise.on() and self.decode_warm) {
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
    /// first (opPenalize; see sample.zig), keeps penalized greedy decode
    /// on the GPU path instead of the full-vocab download.
    pub fn stepArgmaxPen(self: *CudaLM, io: std.Io, ids_new: []const u32, pen: []const sample.PenaltyEntry, sp: sample.Params) !u32 {
        try self.forwardDecode(io, ids_new);
        const be = self.be;
        const b = &self.bufs;
        try be.opPenalize(b.logits.slice(0, self.cfg.vocab * 4), pen, sp);
        try be.opArgmax(b.logits.slice(0, self.cfg.vocab * 4), self.cfg.vocab, b.argmax_out, &b.argmax_v, &b.argmax_i);
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
    /// before the top-k (opPenalize), the selected candidates are the true
    /// post-penalty top set, so penalized stochastic decode stays on the GPU.
    pub fn stepSelectPen(self: *CudaLM, io: std.Io, ids_new: []const u32, pen: []const sample.PenaltyEntry, sp: sample.Params, out_id: []u32, out_logit: []f32) !usize {
        try self.forwardDecode(io, ids_new);
        const be = self.be;
        const b = &self.bufs;
        try be.opPenalize(b.logits.slice(0, self.cfg.vocab * 4), pen, sp);
        const count = try be.opTopK(b.logits.slice(0, self.cfg.vocab * 4), self.cfg.vocab, &b.topk_v, &b.topk_i);
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
            try bnd.check(self);
            // usize annotation matters: @min with a comptime_int bound yields
            // a range-narrowed type (u7 here), and `n * 3` would overflow it.
            //
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

    /// The final-normed hidden state of every row of `ids`, prefilled at the cache's
    /// current length: what a text encoder reads of this model. `out` is
    /// `[ids.len][hidden]`.
    pub fn encode(self: *CudaLM, ids: []const u32, out: []f32) !void {
        const cfg = self.cfg;
        const hidden = cfg.hidden;
        std.debug.assert(out.len == ids.len * hidden);
        const x = try self.gpa.alloc(f32, @min(ids.len, prefill_chunk) * hidden);
        defer self.gpa.free(x);
        var pos3s: [prefill_chunk * 3]u32 = undefined;
        var off: usize = 0;
        while (off < ids.len) {
            try bnd.check(self);
            const n: usize = @min(prefill_chunk, ids.len - off);
            try qwen3.embedTokens(self.lm.embed, ids[off..][0..n], x[0 .. n * hidden]);
            for (0..n) |t| pos3s[t * 3 ..][0..3].* = @splat(@intCast(self.pos_next + t));
            try self.stepBatch(x[0 .. n * hidden], pos3s[0 .. n * 3]);
            self.pos_next += n;
            const rows = out[off * hidden ..][0 .. n * hidden];
            // A split whose last layer ran on the host leaves the stream there.
            if (self.split) |*sp| if (sp.on_host) {
                ops.norm.rmsNorm(rows, sp.hx[0 .. n * hidden], self.lm.final_norm, cfg.rms_eps);
                off += n;
                continue;
            };
            var dev: Dev = .of(self);
            try dev.rms(self.bufs.x, self.bufs.normed, self.lm.final_norm, n, hidden);
            try self.be.downloadF32(self.bufs.normed.slice(0, n * hidden * self.bufs.normed.dt.bytes()), rows);
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
        var pos3s: [prefill_chunk * 3]u32 = undefined;
        var off: usize = 0;
        const total = grid_w * grid_h;
        while (off < total) {
            try bnd.check(self);
            // usize annotation: see prefill (@min narrows its result type).
            const n: usize = @min(prefill_chunk, total - off);
            for (0..n) |t| pos3s[t * 3 ..][0..3].* = qwen35.imagePos3(pos0, grid_w, off + t);
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
        std.debug.assert(n >= 1 and n <= prefill_chunk and x_host.len == n * cfg.hidden);
        std.debug.assert(n <= self.remaining());

        try be.tensorUpload(b.x.slice(0, n * cfg.hidden * 4), std.mem.sliceAsBytes(x_host));
        try be.tensorUpload(self.pos3s_d.slice(0, n * 3 * 4), std.mem.sliceAsBytes(pos3s));

        // One perturbation per forward; no-op while sigma is 0 (cuda/wnoise.zig).
        be.weight_noise.tick();
        try be.beginBatch();
        errdefer if (be.batching()) be.abortBatch();
        // A cancel unwinds between layers: the errdefer aborts the recording and `len`
        // only moves after `endBatch`, so nothing is committed.
        var dev: Dev = .of(self);
        try qwen35_dev.batch(&dev, n);
        try be.endBatch();
        self.len += n;
    }

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
        try be.tensorUpload(self.bufs.x.slice(0, self.cfg.hidden * 4), std.mem.sliceAsBytes(x_host));
        try be.tensorUpload(self.pos3_d, std.mem.sliceAsBytes(&pos3));

        be.weight_noise.tick();
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
        var dev: Dev = .of(self);
        try qwen35_dev.decode(&dev, graph, want_logits);
    }

    /// Which layer the coming launches belong to, for the weight-noise curve. This
    /// stepper walks its own layer loops (the GDN layers are not a
    /// `transformer_gpu.decoderLayer`), so it calls this itself.
    pub fn noiseAtLayer(self: *CudaLM, l: usize) void {
        self.be.weight_noise.atLayer(l);
    }

    /// Single-token decode as one captured-graph replay: {token, len} land
    /// in g_state and the M-RoPE triple in pos3_d before the launch (the
    /// graph's kernels read both). Captured on the second decode step, the
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
        // weights under VRAM pressure, the capture would then re-upload
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
        const x = self.bufs.x.slice(0, cfg.hidden * 4);
        if (self.lm.embed.dtype == .bf16) {
            try self.be.opEmbedGatherS(x, self.lm.embed.bytes, cfg.hidden);
        } else {
            try self.be.opEmbedGatherQuant(self.lm.embed.dtype, x, self.lm.embed.bytes, cfg.hidden);
        }
    }

    /// The recorded decode step: device-side embedding gather (token id from
    /// g_state[0]) followed by the shared decode body in graph mode. No
    /// uploads, downloads, or batch bookkeeping, those live outside the
    /// replay.
    fn recordDecodeOps(self: *CudaLM) !void {
        try self.embedGather();
        try self.decodeBody(true, true);
    }
};

/// `qwen35_dev`'s adapter over a `CudaLM`: each op is the call its forward made before
/// the forward was shared, so the CUDA output is unchanged by the move.
const Dev = struct {
    m: *CudaLM,
    cfg: qwen35.Config,
    lm: *const qwen35.Model,
    bufs: *const Bufs,

    const Weight = lin_cuda.Weight;

    pub fn of(m: *CudaLM) Dev {
        return .{ .m = m, .cfg = m.cfg, .lm = m.lm, .bufs = &m.bufs };
    }

    pub fn view(_: *const Dev, b: Buf, off: usize, bytes: usize) error{}!Buf {
        return b.slice(off, bytes);
    }
    pub fn rms(d: *const Dev, x: Buf, out: Buf, w: []const f32, rows: usize, dim: usize) !void {
        try d.m.be.qkNorm(x, out, try nbuf(d.m.be, w), rows, dim, d.cfg.rms_eps);
    }
    pub fn prep(d: *const Dev, x: Buf, m: usize, cols: usize, ws: []const Weight) !lin_cuda.Group {
        return lin_cuda.prep(d.m.be, d.m.lin_plan, x, m, cols, ws);
    }
    pub fn gemm(d: *const Dev, g: lin_cuda.Group, y: Buf, w: Weight) !void {
        try lin_cuda.gemm(d.m.be, d.m.lin_plan, g, y, w, null);
    }
    pub fn linear(d: *const Dev, y: Buf, x: Buf, m: usize, w: Weight) !void {
        try lin_cuda.linear(d.m.be, d.m.lin_plan, y, x, m, w, null);
    }
    pub fn deinterleave2(d: *const Dev, src: Buf, a: Buf, b: Buf, total: usize, hd: usize) !void {
        try d.m.be.opDeinterleave2(src, a, b, total, hd);
    }
    pub fn mulSigmoid(d: *const Dev, a: Buf, g: Buf, n: usize) !void {
        try d.m.be.opMulSigmoid(a, g, n);
    }
    pub fn add(d: *const Dev, x: Buf, t: Buf, n: usize) !void {
        try d.m.be.opAdd(x, t, n);
    }
    pub fn addRms(d: *const Dev, x: Buf, t: Buf, out: Buf, w: []const f32, rows: usize, dim: usize) !void {
        try d.add(x, t, rows * dim);
        try d.rms(x, out, w, rows, dim);
    }
    pub fn siluMul(d: *const Dev, a: Buf, b: Buf, n: usize) !void {
        try d.m.be.siluMul(a, b, n);
    }
    pub fn l2normRows(d: *const Dev, x: Buf, rows: usize, dim: usize) !void {
        try d.m.be.opL2NormRows(x, rows, dim, d.cfg.rms_eps);
    }
    pub fn l2normRowsGrouped(d: *const Dev, x: Buf, rows: usize, dim: usize, group: usize, stride: usize) !void {
        try d.m.be.opL2NormRowsGrouped(x, rows, dim, group, stride, d.cfg.rms_eps);
    }
    pub fn gdnGates(d: *const Dev, ab: Buf, a_dt: []const f32, gates: Buf, heads: usize) !void {
        try d.m.be.opGdnGates(ab, try nbuf(d.m.be, a_dt), gates, heads);
    }
    pub fn gdnConvStep(d: *const Dev, state: Buf, x: Buf, w: []const f32, out: Buf, channels: usize) !void {
        try d.m.be.opGdnConvStep(state, x, try nbuf(d.m.be, w), out, channels);
    }
    pub fn gdnDeltaStep(d: *const Dev, state: Buf, conv: Buf, gates: Buf, o: Buf, heads: usize, dk: usize, k_heads: usize, scale: f32) !void {
        if (dual_kernels) return d.m.be.opGdnDeltaChunkDual(state, conv, gates, o, heads, dk, k_heads, d.cfg.v_group, 1, d.cfg.convChannels(), scale);
        try d.m.be.opGdnDeltaStep(state, conv, gates, o, heads, dk, k_heads, d.cfg.v_group, scale);
    }
    pub fn gdnGatesBatch(d: *const Dev, alpha: Buf, beta: Buf, a_dt: []const f32, gates: Buf, heads: usize, n: usize, pitch: usize) !void {
        try d.m.be.opGdnGatesBatch(alpha, beta, try nbuf(d.m.be, a_dt), gates, heads, n, pitch);
    }
    pub fn gdnConvBatch(d: *const Dev, state: Buf, x: Buf, w: []const f32, out: Buf, channels: usize, n: usize) !void {
        try d.m.be.opGdnConvBatch(state, x, try nbuf(d.m.be, w), out, channels, n);
    }
    pub fn gdnDeltaChunk(d: *const Dev, state: Buf, conv: Buf, gates: Buf, o: Buf, heads: usize, dk: usize, k_heads: usize, n: usize, channels: usize, scale: f32) !void {
        if (dual_kernels) return d.m.be.opGdnDeltaChunkDual(state, conv, gates, o, heads, dk, k_heads, d.cfg.v_group, n, channels, scale);
        try d.m.be.opGdnDeltaChunk(state, conv, gates, o, heads, dk, k_heads, d.cfg.v_group, n, channels, scale);
    }

    pub fn convState(d: *const Dev, lin_idx: usize) Buf {
        const bytes = d.cfg.convChannels() * (d.cfg.conv_kernel - 1) * 4;
        return d.m.conv_state.slice(lin_idx * bytes, bytes);
    }
    pub fn ssmState(d: *const Dev, lin_idx: usize) Buf {
        const bytes = d.cfg.lin_v_heads * d.cfg.lin_head_dim * d.cfg.lin_head_dim * 4;
        return d.m.ssm_state.slice(lin_idx * bytes, bytes);
    }
    pub fn aDt(d: *const Dev, lin_idx: usize) []const f32 {
        return d.m.a_dt[lin_idx];
    }
    pub fn gdnRoute(d: *const Dev, lin_idx: usize, n: usize) qwen35_dev.GdnRoute(Weight) {
        return .{
            .chunk = gdnChunkOk(d.cfg.lin_head_dim, n),
            .stacked = if (n > 1) d.m.ab_cat[lin_idx] else null,
        };
    }
    /// Where the batched alpha and beta land in `ab_n`: one stacked output read as two
    /// views of its pitch, or two tight planes with beta at `pad_rows`.
    pub fn abViews(d: *const Dev, route: qwen35_dev.GdnRoute(Weight), n: usize) error{}!qwen35_dev.AbViews(Buf) {
        const heads = d.cfg.lin_v_heads;
        const ab_n = d.bufs.ab_n;
        if (route.stacked) |cat| return .{
            .alpha = ab_n.slice(0, n * cat.rows * 4),
            .beta = ab_n.slice(heads * 4, n * cat.rows * 4 - heads * 4),
            .pitch = cat.rows,
        };
        return .{
            .alpha = ab_n.slice(0, n * heads * 4),
            .beta = ab_n.slice(pad_rows * heads * 4, n * heads * 4),
            .pitch = heads,
        };
    }

    pub fn ropeStep(d: *const Dev, qk: Buf, heads: usize) !void {
        const c = d.cfg;
        try d.m.be.opRopeImrope(qk, d.m.pos3_d, d.m.freqs_d, heads, c.rope_dim / 2, d.m.sin_off, c.rope_sections, c.head_dim);
    }
    pub fn ropeRows(d: *const Dev, qk: Buf, heads: usize, n: usize) !void {
        const c = d.cfg;
        try d.m.be.opRopeImropePos(qk, d.m.pos3s_d, d.m.freqs_d, n, heads, c.rope_dim / 2, d.m.sin_off, c.rope_sections, c.head_dim);
    }
    pub fn storeKvStep(d: *const Dev, slot: usize, comptime graph: bool) !void {
        const kv = d.cfg.kvDim();
        if (graph) {
            const fmt = kvFmt(d.m.kv_dtype);
            try d.m.be.opKvAppendS(d.m.k_cache[slot].buf, d.bufs.k, kv, kv, 0, fmt);
            try d.m.be.opKvAppendS(d.m.v_cache[slot].buf, d.bufs.v, kv, kv, 0, fmt);
        } else try d.storeKvRows(slot, 1);
    }
    pub fn storeKvRows(d: *const Dev, slot: usize, n: usize) !void {
        const kv = d.cfg.kvDim();
        try d.m.storeKv(d.m.k_cache[slot].buf, d.m.len * kv, d.bufs.k, 0, n * kv);
        try d.m.storeKv(d.m.v_cache[slot].buf, d.m.len * kv, d.bufs.v, 0, n * kv);
    }
    pub fn attendStep(d: *const Dev, slot: usize, comptime graph: bool) !void {
        const c = d.cfg;
        const b = d.bufs;
        if (graph) {
            try d.m.be.opAttnDecodeSGraph(b.q, d.m.k_cache[slot].buf, d.m.v_cache[slot].buf, b.attn, b.attn_scratch, c.n_heads, c.n_kv_heads, c.head_dim, nsplit, attnScale(c), kvFmt(d.m.kv_dtype));
        } else try d.attend(slot, 1, nsplit);
    }
    pub fn attendRows(d: *const Dev, slot: usize, n: usize) !void {
        try d.attend(slot, n, nsplit_prefill);
    }
    pub fn attend(d: *const Dev, slot: usize, n: usize, splits: usize) !void {
        const c = d.cfg;
        const b = d.bufs;
        if (dual_kernels) return d.m.be.opAttnRowsDual(b.q, d.m.k_cache[slot].buf, d.m.v_cache[slot].buf, b.attn, b.attn_scratch, d.m.len + 1, n, c.n_heads, c.n_kv_heads, c.head_dim, splits, attnScale(c), kvFmt(d.m.kv_dtype));
        try d.m.be.opAttnDecode(b.q, d.m.k_cache[slot].buf, d.m.v_cache[slot].buf, b.attn, b.attn_scratch, d.m.len + 1, n, c.n_heads, c.n_kv_heads, c.head_dim, splits, attnScale(c), 0, 0, false, kvFmt(d.m.kv_dtype));
    }
    pub fn attnScale(c: qwen35.Config) f32 {
        return 1.0 / @sqrt(@as(f32, @floatFromInt(c.head_dim)));
    }

    pub fn forwardBegin(d: *const Dev, _: usize) !void {
        // Hybrid split: the hidden begins on the device (bufs.x); `on_host` tracks
        // where the live residual stream sits as CPU layers pull it over.
        if (d.m.split) |*sp| sp.on_host = false;
    }
    /// Run layer `l` on the host when the split put it there; true when it did.
    pub fn layerBegin(d: *const Dev, l: usize, n: usize) !bool {
        const m = d.m;
        m.noiseAtLayer(l);
        const sp = if (m.split) |*s| s else return false;
        const hidden = d.cfg.hidden;
        const x = d.bufs.x.slice(0, n * hidden * 4);
        if (!sp.on_gpu[l]) {
            if (!sp.on_host) {
                try m.be.tensorDownload(x, std.mem.sliceAsBytes(sp.hx[0 .. n * hidden]));
                sp.on_host = true;
            }
            const host_io = m.io orelse return error.SplitIoUnset;
            try m.lm.cpuLayer(host_io, m.gpa, l, sp.hx[0 .. n * hidden], n, sp.freqs, &sp.state, &sp.scratch);
            return true;
        }
        if (sp.on_host) {
            try m.be.tensorUpload(x, std.mem.sliceAsBytes(sp.hx[0 .. n * hidden]));
            sp.on_host = false;
        }
        return false;
    }
    pub fn tapNormedStep(d: *const Dev, l: usize) !void {
        const od = d.m.op_dump orelse return;
        if (l != 0) return;
        const hidden = d.cfg.hidden;
        try d.m.be.tensorDownload(d.bufs.normed.slice(0, hidden * 4), std.mem.sliceAsBytes(od[d.m.op_dump_row * hidden ..][0..hidden]));
        d.m.op_dump_row += 1;
    }
    pub fn tapNormedRows(d: *const Dev, l: usize, n: usize) !void {
        const od = d.m.op_dump orelse return;
        if (l != 0) return;
        try d.m.be.tensorDownload(d.bufs.normed.slice(0, n * d.cfg.hidden * 4), std.mem.sliceAsBytes(od[0 .. n * d.cfg.hidden]));
    }
    pub fn tapLayer(d: *const Dev, l: usize, n: usize) !void {
        const dump = d.m.layer_dump orelse return;
        const hidden = d.cfg.hidden;
        try d.m.be.tensorDownload(d.bufs.x.slice((n - 1) * hidden * 4, hidden * 4), std.mem.sliceAsBytes(dump[l * hidden ..][0..hidden]));
    }
    /// The split's host KV advances in lockstep. A decode step brings the hidden back
    /// for the device LM head; a prefill reads no logits and the next chunk re-embeds.
    pub fn forwardEnd(d: *const Dev, n: usize, step: bool) !void {
        const sp = if (d.m.split) |*s| s else return;
        if (step and sp.on_host) {
            try d.m.be.tensorUpload(d.bufs.x.slice(0, d.cfg.hidden * 4), std.mem.sliceAsBytes(sp.hx[0..d.cfg.hidden]));
            sp.on_host = false;
        }
        sp.state.kv.commit(n);
        sp.state.len += n;
    }
    pub fn headBegin(d: *const Dev) void {
        d.m.be.weight_noise.atHead();
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
        try be.tensorUpload(buf.slice(off, n), zeros[0..n]);
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
    /// Conv output. One token in the decode path; the whole chunk in batched
    /// prefill (`opGdnConvBatch` writes all `n` at once).
    lin_conv: Buf,
    lin_z: Buf,
    lin_o: Buf,
    ab: Buf,
    gates: Buf,
    /// Batched prefill's alpha/beta. Two layouts: from two GEMVs, `alpha[n][heads]`
    /// at offset 0 and `beta[n][heads]` at `pad_rows*heads`; from one stacked GEMM
    /// (`ab_cat`), `[n][mmq_pipe_tile]` with alpha at column 0 and beta at column
    /// `heads`, which is why it is sized for the wider of the two. Separate from
    /// `ab` because the decode path's single fused GEMV writes `[alpha|beta]`
    /// contiguously per token, which neither batched layout matches.
    ab_n: Buf,
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
        const pc = pad_rows; // see the constant's doc comment
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
            pc * cfg.convChannels(), // lin_conv (batched prefill holds the chunk)
            pc * cfg.linVDim(), // lin_z
            pc * cfg.linVDim(), // lin_o
            2 * cfg.lin_v_heads, // ab (decode: [alpha|beta] from one fused GEMV)
            pc * 2 * cfg.lin_v_heads, // gates ([decay|beta] per token)
            @max(2 * pc * cfg.lin_v_heads, pc * Backend.mmq_pipe_tile), // ab_n, see the field
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
            be.noteBuf(f.name, sizes[i] * 4);
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
// greedy continuation from the same boundary, including after some layers
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

    // A fixed prompt of raw ids (no tokenizer needed, greedy determinism is
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
    // so assert REPEATABILITY here, two restored continuations must agree.
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
    // recurrent state round-trips host->device, so an owner-misdirected
    // restore would now surface as divergence from the all-GPU baseline,
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
// mid-conversation migrate and the promote round trip must stay exact,
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
// image model), a generous budget must still plan layers onto the host,
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
        if (model.split) |sp| sp.n_cpu else 0, be.ctx.freeMiB(),
    });
    try std.testing.expect(model.split != null);
    try std.testing.expect(model.split.?.n_cpu > 0);
}

// A hybrid split's host layers can run BEFORE any step(): the tp-gui turn flow
// prefills first, and an over-budget model has host layers from init. With the
// stepper's `io` unseeded that is undefined-pointer UB (a GP fault deep
// in the host matmul's groupAsync); it must instead fail closed, and work once
// the session owner seeds it (engine/chat.zig Session.init does).
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

test "the GDN chunk recurrence is gated on the head dim and the row count only" {
    // Which kernel alpha/beta take is lin_cuda's decision; the regression this pins is
    // that a dtype with no batched GEMV must STILL take the chunk recurrence.
    const n = 512;
    try std.testing.expect(gdnChunkOk(128, n));
    // Decode, and a head dim the chunk kernel cannot take, do not.
    try std.testing.expect(!gdnChunkOk(128, 1));
    try std.testing.expect(!gdnChunkOk(48, n));
    try std.testing.expect(!gdnChunkOk(512, n));
}
