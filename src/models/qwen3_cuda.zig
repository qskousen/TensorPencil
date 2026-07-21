//! Qwen3-VL-4B text encoder on the hand-PTX CUDA backend.
//!
//! The CUDA analogue of `qwen3_gpu` (Vulkan): the whole 35-layer transformer
//! runs device-resident in one batched submission — one upload of the embedded
//! tokens in, one download of the 12-tap conditioning stack out. GEMMs use the
//! fp8-e4m3 path (`opMatmulFp8`, decode + f16 tensor cores); RMSNorm / per-head
//! QK-norm reuse `qkNorm`; rotate-half RoPE, the SwiGLU gate, residual adds, and
//! the naive causal GQA attention are the CUDA eltwise kernels. Attention stays
//! naive-f32 (parity-first): the encoder sequence is the prompt length (tens to
//! low hundreds of tokens), so the O(seq²) kernel is a sub-second one-time cost.
//! The embedding gather (bf16→f32) and the rope table are CPU-side.
//!
//! Weights upload once through the Backend weight cache and pin resident (LLM
//! weights never stream); a model that outgrows VRAM degrades via the hybrid
//! CPU/GPU layer split (--cpu-layers / --offload-grow).

const std = @import("std");
const qwen3 = @import("qwen3.zig");
const cuda = @import("tp_gpu").cuda;
const safetensors = @import("tp_core").safetensors;
const ops = @import("tp_ops");
const spec = @import("../llm/spec.zig");
const spec_limits = @import("tp_core").spec_limits;
const kvmod = @import("tp_core").kv_cache;
const sample = @import("tp_core").sample;
const transformer = @import("transformer.zig");
const transformer_gpu = @import("transformer_gpu.zig");
const residency = @import("tp_runtime").residency;

const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;
const Growable = Backend.GrowableTensor;

/// Map the session KV dtype onto the backend's kernel-format tag.
fn kvFmt(dt: kvmod.KvDtype) cuda.backend.KvFmt {
    return switch (dt) {
        .f32 => .f32,
        .f16 => .f16,
        .q8_0 => .q8_0,
    };
}


const hidden = qwen3.hidden; // 2560
const n_heads = qwen3.n_heads; // 32
const kv_heads = qwen3.n_kv_heads; // 8
const hd = qwen3.head_dim; // 128
const half = hd / 2; // 64
const q_dim = n_heads * hd; // 4096
const kv_dim = kv_heads * hd; // 1024
const intermediate = qwen3.intermediate; // 9728
const n_layers = qwen3.n_layers; // 36
const tap_count = qwen3.tap_count;
const eps = qwen3.rms_eps;
const attn_scale: f32 = 1.0 / @sqrt(@as(f32, hd));

/// Which layers a hybrid CPU/GPU split pushes to the host. qwen3 is uniform
/// attention, so both policies reduce to a descending-layer order (last layer
/// migrates first); kept for interface parity with qwen35_cuda.
pub const CpuSplitPolicy = enum { tail, attn };

/// Encode token ids to the Krea 2 conditioning stack, [seq][tap_count][hidden]
/// (same token-major layout the CPU `encode` returns). Caller frees the result.
pub fn encode(enc: *const qwen3.TextEncoder, be: *Backend, io: std.Io, gpa: std.mem.Allocator, ids: []const u32) ![]f32 {
    _ = io;
    const seq = ids.len;
    std.debug.assert(seq > 0);
    const seq_pad = std.mem.alignForward(usize, seq, 128);

    // CPU: embedding gather (bf16 -> f32) and the rotate-half rope table.
    const x = try gpa.alloc(f32, seq * hidden);
    defer gpa.free(x);
    for (ids, 0..) |id, t| {
        if (id >= qwen3.vocab_size) return error.TokenIdOutOfRange;
        const row = enc.embed_bytes[@as(usize, id) * hidden * 2 ..][0 .. hidden * 2];
        try safetensors.convertToF32(.bf16, row, x[t * hidden ..][0..hidden]);
    }
    var freqs = try ops.rope.rotateHalfFreqs(gpa, seq, hd, qwen3.rope_theta);
    defer freqs.deinit(gpa);
    const fp = try gpa.alloc(f32, 2 * seq * half);
    defer gpa.free(fp);
    @memcpy(fp[0 .. seq * half], freqs.cos);
    @memcpy(fp[seq * half ..], freqs.sin);
    const sin_off = seq * half;

    var bufs = try Bufs.init(be, seq, seq_pad);
    defer bufs.deinit(be);
    var freqs_d = try be.tensorCreate(fp.len * 4);
    defer be.tensorDestroy(&freqs_d);
    try be.tensorUpload(freqs_d, std.mem.sliceAsBytes(fp));
    try be.tensorUpload(bufs.x, std.mem.sliceAsBytes(x));

    const x_d = bufs.x;
    const nd = bufs.normed;
    const q_d = bufs.q;
    const k_d = bufs.k;
    const v_d = bufs.v;
    const attn_d = bufs.attn;
    const g_d = bufs.gate;
    const u_d = bufs.up;
    const t_d = bufs.t;
    const out_d = bufs.out;

    try be.beginBatch();
    errdefer if (be.batching()) be.abortBatch();

    var tap_idx: usize = 0;
    for (0..n_layers) |l| {
        if (tap_idx < qwen3.tap_layers.len and qwen3.tap_layers[tap_idx] == l) {
            // Snapshot the hidden state entering layer l into the tap-major output.
            try be.tensorCopy(out_d, tap_idx * seq * hidden * 4, x_d, 0, seq * hidden * 4);
            tap_idx += 1;
        }
        if (l >= enc.layers.len) break;
        const layer = enc.layers[l];

        // --- Attention ---
        try be.qkNorm(x_d, nd, try nbuf(be, layer.input_norm), seq, hidden, eps);
        try be.opMatmulFp8(q_d, nd, seq, layer.q.bytes, layer.q.scale, q_dim, hidden);
        try be.opMatmulFp8(k_d, nd, seq, layer.k.bytes, layer.k.scale, kv_dim, hidden);
        try be.opMatmulFp8(v_d, nd, seq, layer.v.bytes, layer.v.scale, kv_dim, hidden);
        try be.qkNorm(q_d, q_d, try nbuf(be, layer.q_norm), seq * n_heads, hd, eps);
        try be.qkNorm(k_d, k_d, try nbuf(be, layer.k_norm), seq * kv_heads, hd, eps);
        try be.ropeHalf(q_d, freqs_d, seq, n_heads, half, sin_off, 0);
        try be.ropeHalf(k_d, freqs_d, seq, kv_heads, half, sin_off, 0);
        try be.attn(q_d, k_d, v_d, attn_d, seq, seq, n_heads, kv_heads, hd, attn_scale, true);
        try be.opMatmulFp8(t_d, attn_d, seq, layer.o.bytes, layer.o.scale, hidden, q_dim);
        try be.opAdd(x_d, t_d, seq * hidden);

        // --- MLP (SwiGLU) ---
        try be.qkNorm(x_d, nd, try nbuf(be, layer.post_norm), seq, hidden, eps);
        try be.opMatmulFp8(g_d, nd, seq, layer.gate.bytes, layer.gate.scale, intermediate, hidden);
        try be.opMatmulFp8(u_d, nd, seq, layer.up.bytes, layer.up.scale, intermediate, hidden);
        try be.siluMul(g_d, u_d, seq * intermediate);
        try be.opMatmulFp8(t_d, g_d, seq, layer.down.bytes, layer.down.scale, hidden, intermediate);
        try be.opAdd(x_d, t_d, seq * hidden);
    }
    std.debug.assert(tap_idx == tap_count);
    try be.endBatch();

    // Download tap-major [tap][seq][hidden]; transpose to token-major.
    const tap_major = try gpa.alloc(f32, tap_count * seq * hidden);
    defer gpa.free(tap_major);
    try be.tensorDownload(out_d, std.mem.sliceAsBytes(tap_major));

    const out = try gpa.alloc(f32, seq * tap_count * hidden);
    errdefer gpa.free(out);
    for (0..tap_count) |tp| {
        for (0..seq) |t| {
            @memcpy(out[(t * tap_count + tp) * hidden ..][0..hidden], tap_major[(tp * seq + t) * hidden ..][0..hidden]);
        }
    }
    return out;
}

/// Wrap a CPU f32 norm-weight slice as a (pointer-cached) small device buffer.
fn nbuf(be: *Backend, weights: []const f32) !Buf {
    return .{ .buf = try be.smallBuffer(std.mem.sliceAsBytes(weights)), .mem = .null_handle, .size = 0 };
}

/// KV-cached causal LM on the CUDA backend (tp-llm --backend zig-cuda /
/// cuda): the full 36-layer stack runs device-resident per step — prefill is
/// one batched submission over the whole prompt (opMatmulFp8 tensor-core
/// GEMMs + the square attn kernel), decode is one over a single token (fused
/// gemv_fp8 dequant-GEMVs + warp flash-decoding attention). K/V live on
/// device, [capacity][kv_dim] f32 per layer; the final norm + tied bf16 LM
/// head (gemv_bf16) run on device too, so only the sampled token's embedding
/// goes up and the vocab logits come down each step. Engine-compatible
/// stepper (see llm/engine.zig generate()).
pub const CudaLM = struct {
    lm: *const qwen3.CausalLM,
    be: *Backend,
    gpa: std.mem.Allocator,
    /// Model shape (mirrors lm.cfg): the 4B target or the 0.6B draft.
    cfg: qwen3.Config,
    /// Committed KV rows; grows in place toward max_capacity (ensureCapacity).
    capacity: usize,
    /// The initial committed row count — what resetResidency shrinks back to.
    initial_capacity: usize,
    /// Growth ceiling — the VA reservation behind each KV cache and the RoPE
    /// table are sized to this, so growth never moves a device pointer.
    max_capacity: usize,
    /// Committed cache length (absolute position of the next token).
    len: usize = 0,
    /// KV-cache element storage type (f32 / f16); selects the attention/append
    /// kernel variant and the per-element stride of k_cache/v_cache.
    kv_dtype: kvmod.KvDtype = .f32,
    /// Activation-buffer row budget: the prompt for prefill, 1 afterwards.
    max_rows: usize,
    sin_off: usize,
    /// Only cfg.n_layers entries are live.
    k_cache: [qwen3.Config.max_layers]Growable,
    v_cache: [qwen3.Config.max_layers]Growable,
    freqs_d: Buf,
    bufs: LmBufs,
    /// Hidden-state taps for the EAGLE-3 drafter (enableTaps): the residual
    /// stream ENTERING each tap layer, device-resident for every committed
    /// position — [3][capacity][hidden] f32. Zero size when disabled.
    tap_layers: [3]usize = .{ 0, 0, 0 },
    tap_d: Buf = .{},
    taps_on: bool = false,
    /// Tree-verify state (enableTree, LLM_PLAN.md M8): batch K/V rows live
    /// at rows [capacity, capacity + spec_limits.max_tree_nodes) of the (enlarged)
    /// per-layer caches; TreeBufs holds positions/meta/logits/tap rows.
    tree: ?TreeBufs = null,
    /// Node count of the last stepAllTree (commitTreePath bound).
    tree_n: usize = 0,
    /// Captured decode step (CUDA graph): one launch replays the whole
    /// forward, with {token, pos0} read from device state. Null until the
    /// second single-token step (the first warms weight residency + JIT).
    graph_exec: cuda.cu.CUgraphExec = null,
    /// First single-token decode ran (capture is safe now).
    decode_warm: bool = false,
    /// Cleared permanently if capture fails — falls back to per-op launches.
    graph_ok: bool = true,
    /// Hybrid CPU/GPU split: CPU-resident layers run host matmuls through the
    /// shared `transformer.layerForward` (dynamic offload migrates more to the
    /// host as the KV cache grows). null = fully device-resident. Any host
    /// layer disables the captured decode graph (migrateLayer drops it; the
    /// step dispatch re-checks per token).
    split: ?Split = null,
    /// Io for the host matmuls of a hybrid split's CPU-resident layers; set
    /// by the step entry points, or seeded by the session owner BEFORE the
    /// first forward (tp-gui prefills before any step). null fails the host
    /// path closed (error.SplitIoUnset) instead of undefined-pointer UB.
    io: ?std.Io = null,

    /// CPU-resident layers of a hybrid split + the host state they need.
    /// Allocated with `gpa`; freed in `deinit`/`resetResidency`. Mirrors
    /// gemma3_cuda.Split minus the ring bookkeeping: qwen3 is uniform
    /// full-attention with a single RoPE table.
    pub const Split = struct {
        /// Per-layer: compute on the device? (false = host).
        on_gpu: []bool,
        n_cpu: usize,
        policy: CpuSplitPolicy,
        /// Host K/V for the CPU-resident layers (full n_layers slots; only the
        /// CPU layers' slots are used). Grows in lockstep with the device
        /// caches; `len` tracks the device `len`.
        cache: kvmod.KvCache,
        /// Host activation scratch, sized for a full max_rows chunk (viewed
        /// down to the actual chunk length per call).
        scratch: qwen3.Scratch,
        /// Host RoPE table, regrown with capacity.
        freqs: ops.rope.Freqs,
        /// Host hidden buffer ([max_rows * hidden]).
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

    pub fn init(gpa: std.mem.Allocator, be: *Backend, lm: *const qwen3.CausalLM, cap: kvmod.Capacity, first_seq: usize) !CudaLM {
        // Embedding/LM-head kernels exist for bf16 and the ggml block-quant
        // formats (tied or untied head); anything else has no gather kernel.
        switch (lm.embed.dtype) {
            .bf16, .q8_0, .q4_k, .q5_k, .q6_k => {},
            else => return error.UnsupportedModelConfig,
        }
        switch (lm.head.dtype) {
            .bf16, .q8_0, .q4_k, .q5_k, .q6_k => {},
            else => return error.UnsupportedModelConfig,
        }
        const c = lm.cfg;
        var self: CudaLM = undefined;
        self.lm = lm;
        self.be = be;
        self.gpa = gpa;
        self.cfg = c;
        self.capacity = cap.initial;
        self.initial_capacity = cap.initial;
        self.max_capacity = cap.max;
        self.split = null;
        self.io = null; // field defaults do not apply to `undefined`-built structs
        self.kv_dtype = cap.kv_dtype;
        self.len = 0;
        // Activation buffers always cover a speculative verify batch; padded
        // GEMM buffers are 128-row anyway, so the floor is nearly free.
        self.max_rows = @max(@max(first_seq, 1), spec_limits.max_draft + 1);
        // RoPE table (and its sin offset, baked into captured graphs as a
        // kernel param) cover max_capacity so KV growth never touches them.
        self.sin_off = cap.max * half;
        self.graph_exec = null;
        self.decode_warm = false;
        self.graph_ok = true;
        self.tap_layers = .{ 0, 0, 0 };
        self.tap_d = .{};
        self.taps_on = false;
        self.tree = null;
        self.tree_n = 0;

        var freqs = try ops.rope.rotateHalfFreqs(gpa, cap.max, hd, c.rope_theta);
        defer freqs.deinit(gpa);
        const fp = try gpa.alloc(f32, 2 * cap.max * half);
        defer gpa.free(fp);
        @memcpy(fp[0 .. cap.max * half], freqs.cos);
        @memcpy(fp[cap.max * half ..], freqs.sin);
        self.freqs_d = try be.tensorCreate(fp.len * 4);
        errdefer be.tensorDestroy(&self.freqs_d);
        try be.tensorUpload(self.freqs_d, std.mem.sliceAsBytes(fp));

        const dt = cap.kv_dtype;
        var created: usize = 0;
        errdefer for (self.k_cache[0..created]) |*b| be.growableDestroy(b);
        for (self.k_cache[0..c.n_layers]) |*b| {
            b.* = try be.growableCreate(dt.sizeBytes(cap.initial * c.kvDim()), dt.sizeBytes(cap.max * c.kvDim()));
            created += 1;
        }
        var vcreated: usize = 0;
        errdefer for (self.v_cache[0..vcreated]) |*b| be.growableDestroy(b);
        for (self.v_cache[0..c.n_layers]) |*b| {
            b.* = try be.growableCreate(dt.sizeBytes(cap.initial * c.kvDim()), dt.sizeBytes(cap.max * c.kvDim()));
            vcreated += 1;
        }

        self.bufs = try LmBufs.init(be, self.max_rows, c);
        return self;
    }

    /// Record the residual stream entering `layers` for every forwarded
    /// position (the EAGLE-3 drafter's fused-feature inputs). Call before
    /// any forward; costs 3 x capacity x hidden f32 of VRAM.
    pub fn enableTaps(self: *CudaLM, layers: [3]usize) !void {
        // EAGLE taps + the tree-verify path use f32-only kernels (opAttnDecodeTree,
        // f32 tap snapshots); f16 KV is greedy/chain-decode only for now.
        if (self.kv_dtype != .f32) return error.KvDtypeUnsupported;
        if (self.split != null) return error.SplitUnsupported; // device-only layout
        std.debug.assert(!self.taps_on);
        for (layers) |l| std.debug.assert(l < self.cfg.n_layers);
        self.tap_d = try self.be.tensorCreate(3 * self.capacity * self.cfg.hidden * 4);
        self.tap_layers = layers;
        self.taps_on = true;
    }

    /// Enable the tree-verify path (spec.generateTree): rebuilds the K/V
    /// caches with spec_limits.max_tree_nodes extra rows (the batch region — tree
    /// nodes cannot append linearly, sibling branches collide at the same
    /// position), grows the activation buffers to cover a full tree batch,
    /// and allocates the tree buffers. Call before any forward.
    pub fn enableTree(self: *CudaLM) !void {
        if (self.kv_dtype != .f32) return error.KvDtypeUnsupported; // tree kernels are f32-only
        if (self.split != null) return error.SplitUnsupported; // device-only layout
        const be = self.be;
        const c = self.cfg;
        std.debug.assert(self.tree == null and self.len == 0);

        // The batch region sits at rows [capacity, capacity + max_tree_nodes)
        // — capacity is baked into the layout, so tree sessions stay fixed.
        std.debug.assert(self.capacity == self.max_capacity);
        const kv_bytes = (self.capacity + spec_limits.max_tree_nodes) * c.kvDim() * 4;
        var nk: [qwen3.Config.max_layers]Growable = undefined;
        var nv: [qwen3.Config.max_layers]Growable = undefined;
        var created: usize = 0;
        errdefer for (0..created) |i| {
            be.growableDestroy(if (i < c.n_layers) &nk[i] else &nv[i - c.n_layers]);
        };
        for (nk[0..c.n_layers]) |*b| {
            b.* = try be.growableCreate(kv_bytes, kv_bytes);
            created += 1;
        }
        for (nv[0..c.n_layers]) |*b| {
            b.* = try be.growableCreate(kv_bytes, kv_bytes);
            created += 1;
        }
        var tb = try TreeBufs.init(be, c);
        errdefer tb.deinit(be);
        if (self.max_rows < spec_limits.max_tree_nodes) {
            const bufs = try LmBufs.init(be, spec_limits.max_tree_nodes, c);
            self.bufs.deinit(be);
            self.bufs = bufs;
            self.max_rows = spec_limits.max_tree_nodes;
        }
        for (self.k_cache[0..c.n_layers]) |*b| be.growableDestroy(b);
        for (self.v_cache[0..c.n_layers]) |*b| be.growableDestroy(b);
        self.k_cache = nk;
        self.v_cache = nv;
        self.tree = tb;
    }

    /// Rebuild the KV cache at a new element dtype (GUI toggle), weights resident.
    /// Drops the captured decode graph so it re-captures with the f16/f32 kernels,
    /// frees + re-creates the K/V buffers (and a split's host shadow, which must
    /// store the same dtype as the device caches), and resets the length. Rejected
    /// while EAGLE taps / tree-verify are active (those are f32-only).
    pub fn reinitCache(self: *CudaLM, dtype: kvmod.KvDtype) !void {
        if (dtype != .f32 and (self.taps_on or self.tree != null)) return error.KvDtypeUnsupported;
        const be = self.be;
        const c = self.cfg;
        if (self.graph_exec != null) {
            be.graphDestroy(self.graph_exec);
            self.graph_exec = null;
        }
        self.decode_warm = false;
        // Host-resident layers keep no device KV (their growables were already
        // destroyed by migrateLayer; growableDestroy is idempotent) — only the
        // GPU layers' buffers are re-created at the new element size.
        for (self.k_cache[0..c.n_layers]) |*b| be.growableDestroy(b);
        for (self.v_cache[0..c.n_layers]) |*b| be.growableDestroy(b);
        self.kv_dtype = dtype;
        self.len = 0;
        for (self.k_cache[0..c.n_layers], self.v_cache[0..c.n_layers], 0..) |*kb, *vb, l| {
            if (self.split) |*sp| if (!sp.on_gpu[l]) continue;
            kb.* = try be.growableCreate(dtype.sizeBytes(self.capacity * c.kvDim()), dtype.sizeBytes(self.max_capacity * c.kvDim()));
            vb.* = try be.growableCreate(dtype.sizeBytes(self.capacity * c.kvDim()), dtype.sizeBytes(self.max_capacity * c.kvDim()));
        }
        if (self.split) |*sp| {
            const cache = try kvmod.KvCache.init(self.gpa, c.n_layers, self.capacity, c.kvDim(), dtype);
            sp.cache.deinit(self.gpa);
            sp.cache = cache;
        }
    }

    pub fn deinit(self: *CudaLM) void {
        if (self.split) |*sp| {
            sp.cache.deinit(self.gpa);
            sp.scratch.deinit(self.gpa);
            sp.freqs.deinit(self.gpa);
            self.gpa.free(sp.hx);
            self.gpa.free(sp.on_gpu);
            self.gpa.free(sp.order);
        }
        if (self.tree) |*tb| tb.deinit(self.be);
        if (self.taps_on) self.be.tensorDestroy(&self.tap_d);
        if (self.graph_exec != null) self.be.graphDestroy(self.graph_exec);
        for (self.k_cache[0..self.cfg.n_layers]) |*b| self.be.growableDestroy(b);
        for (self.v_cache[0..self.cfg.n_layers]) |*b| self.be.growableDestroy(b);
        self.be.tensorDestroy(&self.freqs_d);
        self.bufs.deinit(self.be);
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
    /// stay valid. Under VRAM pressure a dynamic split migrates layers to
    /// the CPU to make room (weights are pinned, never evicted into a
    /// streaming path). error.ContextFull past the window, when even
    /// migration can't free enough device memory, or when the tap/tree
    /// layouts (which stride by capacity) pin the session to a fixed size.
    pub fn ensureCapacity(self: *CudaLM, min_rows: usize) !void {
        if (min_rows <= self.capacity) return;
        if (min_rows > self.max_capacity) return error.ContextFull;
        if (self.taps_on or self.tree != null) return error.ContextFull;
        const target = kvmod.growTarget(self.capacity, min_rows, self.max_capacity);
        const bytes = self.kv_dtype.sizeBytes(target * self.cfg.kvDim());

        // Dynamic offload: migrate layers GPU->CPU until there's headroom to
        // grow the device KV, instead of streaming weights (the cliff). Each
        // migrated layer frees its device KV + weight VRAM. Mirrors
        // gemma3_cuda/qwen35_cuda.
        if (self.split) |*sp| if (sp.dynamic) {
            const add = self.kv_dtype.sizeBytes((target - self.capacity) * self.cfg.kvDim());
            while (true) {
                const need = self.liveSlots() * 2 * add + (32 << 20); // + margin
                const free = @min(sp.budget -| self.be.deviceUsed(), self.be.headroom());
                if (free >= need) break;
                if (!(try residency.migrateNext(self))) break; // nothing left; fall through
            }
        };

        // Grow the device KV of the layers still on the GPU. Physical VRAM can
        // be exhausted even after the proactive migration above (another CUDA
        // context can grab it between the check and this commit): on a real
        // OOM, offload one more layer and retry the whole grow, so a full
        // window only ever fails once nothing is left to migrate.
        grow: while (true) {
            for (0..self.cfg.n_layers) |l| {
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
            const freqs = ops.rope.rotateHalfFreqs(self.gpa, target, hd, self.cfg.rope_theta) catch return error.ContextFull;
            sp.freqs.deinit(self.gpa);
            sp.freqs = freqs;
        }
        self.capacity = target;
    }

    /// Forward `ids` at positions [len, len+ids.len), then write
    /// last-position vocab logits. Multi-turn prefills longer than the
    /// activation buffers run as max_rows-sized chunks (each chunk's LM head
    /// is wasted except the last — negligible next to the layer GEMMs).
    /// Single-token decode replays a captured CUDA graph (one launch instead
    /// of ~700-950) once the first decode step has warmed weight residency;
    /// --profile and capture failures fall back to per-op launches.
    pub fn step(self: *CudaLM, io: std.Io, ids: []const u32, logits: []f32) !void {
        self.io = io; // the CPU half of a hybrid split runs host matmuls through it
        // Any weight eviction (the GUI's reclaim paths under VRAM pressure)
        // means device weight pointers are not stable, and a captured graph
        // would replay against freed buffers.
        if (self.be.evictions != 0) self.graph_ok = false;
        if (ids.len == 1 and self.graphEligible()) {
            if (self.decode_warm) return self.stepDecodeGraph(ids[0], logits);
            self.decode_warm = true;
        }
        var off: usize = 0;
        while (off < ids.len) {
            const n = @min(self.max_rows, ids.len - off);
            try self.stepChunk(ids[off..][0..n], logits);
            off += n;
        }
    }

    /// Forward `ids` without downloading logits (the GUI's turn prefill; the
    /// generation step that follows produces the first sampled token). Runs
    /// as max_rows-sized chunks like `step`; each chunk's LM head is wasted,
    /// which is negligible next to the layer GEMMs.
    pub fn prefill(self: *CudaLM, ids: []const u32) !void {
        var off: usize = 0;
        while (off < ids.len) {
            const n = @min(self.max_rows, ids.len - off);
            try self.stepChunk(ids[off..][0..n], null);
            off += n;
        }
    }

    /// Whether the captured-graph decode path may be used or captured right
    /// now: never with a host-resident layer — the graph records device ops
    /// for every layer (migrateLayer also drops any captured graph).
    fn graphEligible(self: *const CudaLM) bool {
        if (self.split) |*sp| if (sp.n_cpu > 0) return false;
        return self.graph_ok and !self.be.profile;
    }

    fn stepDecodeGraph(self: *CudaLM, id: u32, logits: ?[]f32) !void {
        const be = self.be;
        std.debug.assert(self.remaining() >= 1);
        try be.setDecodeState(id, @intCast(self.len));
        if (self.graph_exec == null) {
            self.captureDecodeGraph() catch |err| {
                // Leave graph mode permanently and decode this token normally.
                std.log.warn("decode graph capture failed ({t}); falling back to per-op launches", .{err});
                self.graph_ok = false;
                return self.stepChunk(&.{id}, logits);
            };
        }
        if (be.evictions != 0) {
            // Capture itself ran the cache over budget: the fresh graph may
            // already hold evicted-weight pointers. Decode per-op instead.
            self.graph_ok = false;
            return self.stepChunk(&.{id}, logits);
        }
        try be.graphLaunch(self.graph_exec);
        self.advance(1);
        // `null` leaves logits resident for the on-device argmax (stepArgmax).
        if (logits) |l| try be.tensorDownload(offsetBufSized(self.bufs.logits, 0, qwen3.vocab_size * 4), std.mem.sliceAsBytes(l[0..qwen3.vocab_size]));
    }

    fn captureDecodeGraph(self: *CudaLM) !void {
        const be = self.be;
        // The embed table is only touched device-side by the graph (per-op
        // steps embed on host), so gather once outside capture first: its
        // initial cachedWeight cuMemAlloc + upload are illegal on a capturing
        // stream. Tied-head models never hit this (the warm LM-head GEMV
        // already cached the table, head == embed); untied ones (8B+) failed
        // capture every time without it. Mirrors qwen35_cuda.
        try self.embedGather();
        // That upload may itself have evicted LRU weights under VRAM
        // pressure — the capture would then re-upload them mid-capture
        // (cuMemAlloc again). Bail to per-op decode instead of starting a
        // capture that must fail.
        if (be.evictions != 0) return error.WeightsEvicted;
        try be.graphCaptureBegin();
        errdefer if (be.graphCaptureEnd()) |exec| be.graphDestroy(exec) else |_| {};
        try self.recordDecodeOps();
        self.graph_exec = try be.graphCaptureEnd();
    }

    /// Device-side embedding gather of g_state[0]'s row into bufs.x.
    fn embedGather(self: *CudaLM) !void {
        const c = self.cfg;
        const x = offsetBufSized(self.bufs.x, 0, c.hidden * 4);
        if (self.lm.embed.dtype == .bf16) {
            try self.be.opEmbedGatherS(x, self.lm.embed.bytes, c.hidden);
        } else {
            try self.be.opEmbedGatherQuant(self.lm.embed.dtype, x, self.lm.embed.bytes, c.hidden);
        }
    }

    /// The full single-token decode forward, recorded for graph capture:
    /// identical kernels and order to stepChunk at seq == 1, except the
    /// embedding gather runs on device and pos0/token come from g_state
    /// (bitwise-identical logits either way). No batch bookkeeping, no
    /// downloads — the caller launches the graph and reads bufs.logits.
    fn recordDecodeOps(self: *CudaLM) !void {
        const be = self.be;
        const c = self.cfg;
        const b = &self.bufs;
        try self.embedGather();
        for (self.lm.layers, 0..) |layer, l| {
            if (self.taps_on) {
                for (self.tap_layers, 0..) |tl, j| {
                    // Taps are f32 hidden-state snapshots (EAGLE), never the KV cache.
                    if (l == tl) try be.opKvAppendS(self.tap_d, b.x, c.hidden, c.hidden, j * self.capacity * c.hidden, .f32);
                }
            }
            try be.qkNorm(b.x, b.normed, try nbuf(be, layer.input_norm), 1, c.hidden, eps);
            try self.linear(b.q, b.normed, layer.q, c.qDim(), c.hidden, 1);
            try self.linear(b.k, b.normed, layer.k, c.kvDim(), c.hidden, 1);
            try self.linear(b.v, b.normed, layer.v, c.kvDim(), c.hidden, 1);
            try be.qkNorm(b.q, b.q, try nbuf(be, layer.q_norm), c.n_heads, hd, eps);
            try be.qkNorm(b.k, b.k, try nbuf(be, layer.k_norm), c.n_kv_heads, hd, eps);
            try be.opRopeHalfS(b.q, self.freqs_d, c.n_heads, half, self.sin_off);
            try be.opRopeHalfS(b.k, self.freqs_d, c.n_kv_heads, half, self.sin_off);
            const fmt = kvFmt(self.kv_dtype);
            try be.opKvAppendS(self.k_cache[l].buf, b.k, c.kvDim(), c.kvDim(), 0, fmt);
            try be.opKvAppendS(self.v_cache[l].buf, b.v, c.kvDim(), c.kvDim(), 0, fmt);
            try be.opAttnDecodeSGraph(b.q, self.k_cache[l].buf, self.v_cache[l].buf, b.attn, b.attn_scratch, c.n_heads, c.n_kv_heads, hd, nsplit, attn_scale, fmt);
            try self.linear(b.t, b.attn, layer.o, c.hidden, c.qDim(), 1);
            try be.opAdd(b.x, b.t, c.hidden);
            try be.qkNorm(b.x, b.normed, try nbuf(be, layer.post_norm), 1, c.hidden, eps);
            try self.linear(b.gate, b.normed, layer.gate, c.intermediate, c.hidden, 1);
            try self.linear(b.up, b.normed, layer.up, c.intermediate, c.hidden, 1);
            try be.siluMul(b.gate, b.up, c.intermediate);
            try self.linear(b.t, b.gate, layer.down, c.hidden, c.intermediate, 1);
            try be.opAdd(b.x, b.t, c.hidden);
        }
        try be.qkNorm(offsetBufSized(b.x, 0, c.hidden * 4), b.t, try nbuf(be, self.lm.final_norm), 1, c.hidden, eps);
        try self.lmHeadGemv(b.logits, b.t);
    }

    /// step, but with vocab logits for every new token ([ids.len, vocab]
    /// row-major) — the speculative-decode verify forward. The batch is
    /// engine-capped at spec_limits.max_draft + 1, which max_rows always covers.
    pub fn stepAll(self: *CudaLM, io: std.Io, ids: []const u32, logits: []f32) !void {
        self.io = io; // the CPU half of a hybrid split runs host matmuls through it
        std.debug.assert(logits.len == ids.len * qwen3.vocab_size);
        try self.forwardAll(ids);
        try self.be.tensorDownload(offsetBufSized(self.bufs.logits, 0, ids.len * qwen3.vocab_size * 4), std.mem.sliceAsBytes(logits));
    }

    /// Batched verify forward that leaves the per-row logits resident in
    /// b.logits ([seq][vocab] on device) and advances the cache. Shared by
    /// stepAll (downloads them) and stepAllArgmax (argmaxes them on-device).
    fn forwardAll(self: *CudaLM, ids: []const u32) !void {
        const be = self.be;
        const seq = ids.len;
        std.debug.assert(seq > 0 and seq <= spec_limits.max_draft + 1 and seq <= self.max_rows);
        const b = &self.bufs;
        try self.layersForward(ids);
        errdefer if (be.batching()) be.abortBatch();
        // Final norm on every new position, then the tied bf16 LM head in
        // 4-input groups (each group reads the vocab x hidden weight once).
        // b.t is 128-row padded, so gemv_bf16n's 4-row reads stay in bounds.
        const h = self.cfg.hidden;
        try be.qkNorm(b.x, b.t, try nbuf(be, self.lm.final_norm), seq, h, eps);
        try self.lmHeadAll(b.logits, b.t, seq);
        try be.endBatch();
        self.advance(seq);
    }

    /// stepAll's greedy twin: the verify forward, then a per-row on-device
    /// argmax into out_ids[0..seq] — downloading `seq` ids, not seq*vocab
    /// (~608 KB/row). For greedy speculative decoding, where acceptance only
    /// compares each draft to the target's argmax.
    pub fn stepAllArgmax(self: *CudaLM, io: std.Io, ids: []const u32, out_ids: []u32) !void {
        self.io = io; // the CPU half of a hybrid split runs host matmuls through it
        std.debug.assert(out_ids.len >= ids.len);
        try self.forwardAll(ids);
        const be = self.be;
        const b = &self.bufs;
        for (0..ids.len) |r| {
            try be.opArgmax(offsetBufSized(b.logits, r * qwen3.vocab_size * 4, qwen3.vocab_size * 4), qwen3.vocab_size, b.argmax_out, &b.argmax_v, &b.argmax_i);
            var idf: [1]f32 = undefined;
            try be.tensorDownload(b.argmax_out, std.mem.sliceAsBytes(&idf));
            out_ids[r] = @intFromFloat(idf[0]);
        }
    }

    /// Roll the KV cache back to `new_len` tokens (speculative-decode
    /// rejection); device rows past `new_len` are overwritten by later steps.
    pub fn truncate(self: *CudaLM, new_len: usize) void {
        std.debug.assert(new_len <= self.len);
        self.len = new_len;
        if (self.split) |*sp| sp.cache.truncate(new_len);
    }

    /// Advance the committed length by `n` freshly appended rows, keeping the
    /// split's host shadow (whose CPU layers appended their own rows inside
    /// layerForward) in lockstep with the device counter.
    fn advance(self: *CudaLM, n: usize) void {
        self.len += n;
        if (self.split) |*sp| sp.cache.commit(n);
    }

    /// Reset the session to an empty context (GUI "new chat"). qwen3 is a
    /// plain append-only attention model — no rings, no recurrent state — so
    /// only the position counters need clearing (rows are overwritten lazily
    /// by the next prefill, on the device caches and the split's host shadow
    /// alike).
    pub fn resetCache(self: *CudaLM) !void {
        self.len = 0;
        if (self.split) |*sp| sp.cache.truncate(0);
    }

    /// Turn-boundary checkpoints (GUI regenerate / variant switch): the KV
    /// cache is append-only, so a boundary needs NO snapshot bytes — restoring
    /// is a pure truncate back to `q` (rows past it are overwritten later),
    /// exactly like the speculative-decode rejection path.
    pub fn checkpointBytes(self: *const CudaLM) usize {
        _ = self;
        return 0;
    }

    pub fn checkpoint(self: *CudaLM, out: []u8) !void {
        _ = self;
        std.debug.assert(out.len == 0);
    }

    pub fn restoreCheckpoint(self: *CudaLM, snap: []const u8, q: usize) !void {
        std.debug.assert(snap.len == 0);
        self.truncate(q);
    }

    /// New-chat residency reset (GUI): drop the split (weights come back to
    /// the GPU lazily), drop the captured decode graph, shrink every K/V
    /// cache back to the initial capacity (frees the grown VRAM), clear the
    /// context, and re-arm dynamic offload for the fresh small context. KV
    /// is discarded, so no host->device copy is needed (unlike promoteLayers).
    pub fn resetResidency(self: *CudaLM, budget: u64) !void {
        const be = self.be;
        const c = self.cfg;
        if (self.split) |*sp| {
            sp.cache.deinit(self.gpa);
            sp.scratch.deinit(self.gpa);
            sp.freqs.deinit(self.gpa);
            self.gpa.free(sp.hx);
            self.gpa.free(sp.on_gpu);
            self.gpa.free(sp.order);
            self.split = null;
        }
        // Tap/tree sessions are pinned to a fixed capacity (their layouts
        // stride by it); the GUI never enables either, but stay safe.
        if (!self.taps_on and self.tree == null) {
            // New growable buffers mean new device pointers: a captured graph
            // would replay against the freed ones. (Migrated layers' growables
            // were already destroyed; growableDestroy is idempotent.)
            if (self.graph_exec != null) {
                be.graphDestroy(self.graph_exec);
                self.graph_exec = null;
            }
            self.decode_warm = false;
            const dt = self.kv_dtype;
            for (self.k_cache[0..c.n_layers]) |*b| {
                be.growableDestroy(b);
                b.* = try be.growableCreate(dt.sizeBytes(self.initial_capacity * c.kvDim()), dt.sizeBytes(self.max_capacity * c.kvDim()));
            }
            for (self.v_cache[0..c.n_layers]) |*b| {
                be.growableDestroy(b);
                b.* = try be.growableCreate(dt.sizeBytes(self.initial_capacity * c.kvDim()), dt.sizeBytes(self.max_capacity * c.kvDim()));
            }
            self.capacity = self.initial_capacity;
        }
        try self.resetCache();
        _ = try self.autoOffload(budget);
    }

    /// Total device footprint of one layer's streamable weights (quantized
    /// bytes) — the projection + MLP matrices; norms are negligible. `anytype`
    /// avoids naming qwen3's private `Layer` type.
    fn layerDeviceBytes(layer: anytype) usize {
        return layer.q.bytes.len + layer.k.bytes.len + layer.v.bytes.len + layer.o.bytes.len +
            layer.gate.bytes.len + layer.up.bytes.len + layer.down.bytes.len;
    }

    /// Layers whose KV still lives on the device.
    fn liveSlots(self: *CudaLM) usize {
        if (self.split) |*sp| return self.cfg.n_layers - sp.n_cpu;
        return self.cfg.n_layers;
    }

    /// Move layer `l`'s live K/V device->host, free its device K/V + weights,
    /// and mark it CPU-resident (a `residency` hook). Also drops the captured
    /// decode graph: it holds this layer's (freed) KV pointers, and the graph
    /// can't represent a host layer anyway — the step dispatch re-captures if
    /// every layer is later promoted back.
    pub fn migrateLayer(self: *CudaLM, l: usize) !void {
        const sp = &self.split.?;
        const kvd = self.cfg.kvDim();
        if (self.graph_exec != null) {
            self.be.graphDestroy(self.graph_exec);
            self.graph_exec = null;
        }
        self.decode_warm = false;
        if (self.len > 0) {
            // The host shadow stores the same dtype as the device cache
            // (kRowBytes is the device byte layout), so this is a raw copy.
            const dt = self.kv_dtype;
            try self.be.tensorDownload(offsetBufSized(self.k_cache[l].buf, 0, dt.sizeBytes(self.len * kvd)), sp.cache.kRowBytes(l, 0, self.len));
            try self.be.tensorDownload(offsetBufSized(self.v_cache[l].buf, 0, dt.sizeBytes(self.len * kvd)), sp.cache.vRowBytes(l, 0, self.len));
        }
        self.be.growableDestroy(&self.k_cache[l]);
        self.be.growableDestroy(&self.v_cache[l]);
        // Free the migrated layer's device weights (the host path reads them
        // from the GGUF mapping) — the bulk of the reclaimed VRAM.
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
        std.log.debug("[offload] layer {d} -> CPU at ctx {d} ({d}/{d} on CPU)", .{ l, self.len, sp.n_cpu, self.cfg.n_layers });
    }

    /// Migrate layers to the host until `@min(budget - deviceUsed, headroom)`
    /// reaches `needed_free` bytes, or nothing is left (a `residency` wrapper;
    /// the VRAM coordinator's fixed-target variant).
    pub fn offloadUntilFree(self: *CudaLM, needed_free: u64) !void {
        return residency.offloadUntilFree(self, needed_free);
    }

    /// Migrate layers until total device usage is <= `target` bytes (balanced
    /// mode: settle the LLM to its share when an image model loads).
    pub fn offloadToBudget(self: *CudaLM, target: u64) !void {
        return residency.offloadToBudget(self, target);
    }

    /// `residency.promoteBack` cost hook: VRAM a promote of layer `l` needs —
    /// its streamable weights, the KV it re-commits at the current capacity,
    /// plus slack.
    pub fn promoteCost(self: *CudaLM, l: usize) usize {
        const kv_at_cap = 2 * self.kv_dtype.sizeBytes(self.capacity * self.cfg.kvDim());
        return layerDeviceBytes(&self.lm.layers[l]) + kv_at_cap + (64 << 20);
    }

    /// Bring layer `l` back onto the GPU, preserving its accumulated K/V:
    /// re-create the device K/V at the current capacity and upload the host
    /// rows [0,len). Weights re-cache lazily on the next GPU forward. Reverse
    /// of migrateLayer.
    pub fn promoteLayer(self: *CudaLM, l: usize) !void {
        const sp = &self.split.?;
        const kvd = self.cfg.kvDim();
        const dt = self.kv_dtype;
        self.k_cache[l] = try self.be.growableCreate(dt.sizeBytes(self.capacity * kvd), dt.sizeBytes(self.max_capacity * kvd));
        self.v_cache[l] = try self.be.growableCreate(dt.sizeBytes(self.capacity * kvd), dt.sizeBytes(self.max_capacity * kvd));
        if (self.len > 0) {
            try self.be.tensorUpload(offsetBufSized(self.k_cache[l].buf, 0, dt.sizeBytes(self.len * kvd)), sp.cache.kRowBytes(l, 0, self.len));
            try self.be.tensorUpload(offsetBufSized(self.v_cache[l].buf, 0, dt.sizeBytes(self.len * kvd)), sp.cache.vRowBytes(l, 0, self.len));
        }
        sp.on_gpu[l] = true;
        sp.n_cpu -= 1;
        std.log.debug("[promote] layer {d} -> GPU at ctx {d} ({d}/{d} on CPU)", .{ l, self.len, sp.n_cpu, self.cfg.n_layers });
    }

    /// Migrate CPU layers back onto the GPU (LIFO by offload order), stopping
    /// before the next one would overflow `budget`. Keeps the split armed.
    /// Returns the number promoted; 0 without a split.
    pub fn promoteLayers(self: *CudaLM, budget: u64) !usize {
        return residency.promoteBack(self, budget);
    }

    /// Always arm the dynamic split (`budget == 0` = no offload). Free when
    /// the model fits (0 layers on CPU), and migrates layers on demand as the
    /// KV cache grows — so over-budget growth degrades via CPU offload
    /// (measured ~2.5x faster than the weight-streaming fallback on this box)
    /// rather than the streaming cliff. See qwen35_cuda.autoOffload.
    pub fn autoOffload(self: *CudaLM, budget: u64) !bool {
        if (budget == 0) return false;
        try self.enableCpuSplit(.attn, budget, true);
        return true;
    }

    /// Place layers on the host until the device-resident weights fit under
    /// `budget` (bytes). `dynamic` packs the GPU now (head-only reserve) and
    /// migrates on demand as the KV grows; static reserves generously. A
    /// static no-op split (all fit) leaves `self.split == null`. Mirrors
    /// gemma3_cuda.enableCpuSplit (minus the ring bookkeeping).
    pub fn enableCpuSplit(self: *CudaLM, policy: CpuSplitPolicy, budget: u64, dynamic: bool) !void {
        // Tap/tree layouts stride by a fixed capacity and are device-only.
        if (self.taps_on or self.tree != null) return error.SplitUnsupported;
        std.debug.assert(self.split == null);
        const c = self.cfg;
        const n = c.n_layers;
        const gpa = self.gpa;

        const per = try gpa.alloc(usize, n);
        defer gpa.free(per);
        var total_weight: usize = 0;
        for (self.lm.layers, 0..) |*layer, l| {
            per[l] = layerDeviceBytes(layer);
            total_weight += per[l];
        }

        // Device memory that must stay resident: KV + LM head + slack.
        const kv_bytes = 2 * n * self.kv_dtype.sizeBytes(self.capacity * c.kvDim());
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

        // Eviction order: last layer leaves first (descending). qwen3 is
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
        // stores the SAME dtype as the device caches, so migrate/promote are
        // raw byte copies (kRowBytes) and f16 keeps its halved footprint on
        // the host too.
        var cache = try kvmod.KvCache.init(gpa, n, self.capacity, c.kvDim(), self.kv_dtype);
        errdefer cache.deinit(gpa);
        // The host shadow tracks the SAME committed length as the device from
        // the moment the split arms. Armed mid-conversation (imageReclaim),
        // migrateLayer copies each migrated layer's live rows, so declaring
        // them committed is correct.
        cache.len = self.len;
        var scratch = try qwen3.Scratch.init(gpa, self.max_rows, c);
        errdefer scratch.deinit(gpa);
        var freqs = try ops.rope.rotateHalfFreqs(gpa, self.capacity, hd, c.rope_theta);
        errdefer freqs.deinit(gpa);
        const hx = try gpa.alloc(f32, self.max_rows * c.hidden);
        errdefer gpa.free(hx);

        self.split = .{
            .on_gpu = on_gpu,
            .n_cpu = 0, // the placement below marks + counts the host layers
            .policy = policy,
            .cache = cache,
            .scratch = scratch,
            .freqs = freqs,
            .hx = hx,
            .dynamic = dynamic,
            .order = order,
            .next = n_cpu,
            .budget = budget,
        };

        // Place the statically-planned layers on the host. Before any tokens
        // there is nothing to copy — mark them and free the device K/V;
        // weights are reclaimed lazily. Armed MID-conversation, each layer's
        // live rows must move to the host — migrateLayer does the copy AND
        // the bookkeeping.
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

    /// Tree-verify forward (spec.generateTree): vocab logits for every tree
    /// node. Node i sits at position len + depth(i) (per-row rope), attends
    /// the committed prefix plus its ancestor chain (attn_split_tree), and
    /// its K/V rows land in the batch region at cache row capacity + i —
    /// the linear cache and `len` are untouched until commitTreePath.
    pub fn stepAllTree(self: *CudaLM, io: std.Io, tokens: []const u32, parents: []const u32, logits: []f32) !void {
        _ = io;
        const be = self.be;
        const c = self.cfg;
        const gpa = self.gpa;
        const n = tokens.len;
        std.debug.assert(self.tree != null);
        std.debug.assert(n >= 1 and n <= spec_limits.max_tree_nodes and n <= self.max_rows);
        std.debug.assert(parents.len == n and logits.len == n * qwen3.vocab_size);
        const tb = &self.tree.?;
        const b = &self.bufs;

        // Host: depth-based positions + the attention meta table (per-query
        // kv_len, then the ancestor NODE indices in depth order — the split
        // kernel adds the batch-region base itself).
        var pos: [spec_limits.max_tree_nodes]u32 = undefined;
        var depth: [spec_limits.max_tree_nodes]u32 = undefined;
        pos[0] = @intCast(self.len);
        depth[0] = 0;
        for (parents[1..], 1..) |p, i| {
            std.debug.assert(p < i);
            depth[i] = depth[p] + 1;
            pos[i] = pos[0] + depth[i];
            std.debug.assert(pos[i] < self.capacity);
        }
        var meta: [spec_limits.max_tree_nodes * (spec_limits.max_tree_nodes + 1)]u32 = undefined;
        for (0..n) |i| {
            meta[i * (n + 1)] = @intCast(self.len + depth[i] + 1);
            var j: u32 = @intCast(i);
            var d = depth[i];
            while (true) {
                meta[i * (n + 1) + 1 + d] = j;
                if (j == 0) break;
                j = parents[j];
                d -= 1;
            }
        }
        const meta_off = n * c.n_heads * nsplit * (hd + 4);
        try be.tensorUpload(offsetBufSized(tb.pos, 0, n * 4), std.mem.sliceAsBytes(pos[0..n]));
        try be.tensorUpload(offsetBufSized(tb.scratch, meta_off * 4, n * (n + 1) * 4), std.mem.sliceAsBytes(meta[0 .. n * (n + 1)]));

        // CPU: embedding gather (storage dtype -> f32), upload.
        const x = try gpa.alloc(f32, n * c.hidden);
        defer gpa.free(x);
        try qwen3.embedTokens(self.lm.embed, tokens, x);
        try be.tensorUpload(offsetBufSized(b.x, 0, n * c.hidden * 4), std.mem.sliceAsBytes(x));

        try be.beginBatch();
        errdefer if (be.batching()) be.abortBatch();
        for (self.lm.layers, 0..) |layer, l| {
            if (self.taps_on) {
                for (self.tap_layers, 0..) |tl, j| {
                    if (l == tl) try be.opCopyOff(tb.taps, j * spec_limits.max_tree_nodes * c.hidden, b.x, 0, n * c.hidden);
                }
            }
            // --- Attention ---
            try be.qkNorm(b.x, b.normed, try nbuf(be, layer.input_norm), n, c.hidden, eps);
            try self.linear(b.q, b.normed, layer.q, c.qDim(), c.hidden, n);
            try self.linear(b.k, b.normed, layer.k, c.kvDim(), c.hidden, n);
            try self.linear(b.v, b.normed, layer.v, c.kvDim(), c.hidden, n);
            try be.qkNorm(b.q, b.q, try nbuf(be, layer.q_norm), n * c.n_heads, hd, eps);
            try be.qkNorm(b.k, b.k, try nbuf(be, layer.k_norm), n * c.n_kv_heads, hd, eps);
            try be.opRopeHalfPos(b.q, tb.pos, self.freqs_d, n, c.n_heads, half, self.sin_off);
            try be.opRopeHalfPos(b.k, tb.pos, self.freqs_d, n, c.n_kv_heads, half, self.sin_off);
            try be.tensorCopy(self.k_cache[l].buf, self.capacity * c.kvDim() * 4, b.k, 0, n * c.kvDim() * 4);
            try be.tensorCopy(self.v_cache[l].buf, self.capacity * c.kvDim() * 4, b.v, 0, n * c.kvDim() * 4);
            try be.opAttnDecodeTree(b.q, self.k_cache[l].buf, self.v_cache[l].buf, b.attn, tb.scratch, self.len, self.capacity, n, c.n_heads, c.n_kv_heads, hd, nsplit, attn_scale);
            try self.linear(b.t, b.attn, layer.o, c.hidden, c.qDim(), n);
            try be.opAdd(b.x, b.t, n * c.hidden);

            // --- MLP (SwiGLU) ---
            try be.qkNorm(b.x, b.normed, try nbuf(be, layer.post_norm), n, c.hidden, eps);
            try self.linear(b.gate, b.normed, layer.gate, c.intermediate, c.hidden, n);
            try self.linear(b.up, b.normed, layer.up, c.intermediate, c.hidden, n);
            try be.siluMul(b.gate, b.up, n * c.intermediate);
            try self.linear(b.t, b.gate, layer.down, c.hidden, c.intermediate, n);
            try be.opAdd(b.x, b.t, n * c.hidden);
        }

        // Final norm on every node, then the tied bf16 LM head in 4-input
        // groups (b.t is 128-row padded, so the 4-row reads stay in bounds).
        const h = c.hidden;
        try be.qkNorm(b.x, b.t, try nbuf(be, self.lm.final_norm), n, h, eps);
        try self.lmHeadAll(tb.logits, b.t, n);
        try be.endBatch();
        self.tree_n = n;

        try be.tensorDownload(offsetBufSized(tb.logits, 0, n * qwen3.vocab_size * 4), std.mem.sliceAsBytes(logits));
    }

    /// Copy the accepted root path's K/V rows (and tap rows, when the
    /// EAGLE-3 taps are on) from the batch region into the linear cache at
    /// positions [len, len + path.len), then advance len. ~2 * n_layers *
    /// depth copy_off launches — small next to a verify forward.
    pub fn commitTreePath(self: *CudaLM, path: []const usize) !void {
        const be = self.be;
        const c = self.cfg;
        std.debug.assert(self.tree != null and path.len >= 1 and path[0] == 0);
        std.debug.assert(self.len + path.len <= self.capacity);
        const kvd = c.kvDim();
        for (0..c.n_layers) |l| {
            for (path, 0..) |idx, j| {
                std.debug.assert(idx < self.tree_n);
                try be.opCopyOff(self.k_cache[l].buf, (self.len + j) * kvd, self.k_cache[l].buf, (self.capacity + idx) * kvd, kvd);
                try be.opCopyOff(self.v_cache[l].buf, (self.len + j) * kvd, self.v_cache[l].buf, (self.capacity + idx) * kvd, kvd);
            }
        }
        if (self.taps_on) {
            const tb = &self.tree.?;
            for (0..3) |t| {
                for (path, 0..) |idx, j| {
                    try be.opCopyOff(self.tap_d, (t * self.capacity + self.len + j) * c.hidden, tb.taps, (t * spec_limits.max_tree_nodes + idx) * c.hidden, c.hidden);
                }
            }
        }
        self.len += path.len;
    }

    fn stepChunk(self: *CudaLM, ids: []const u32, logits: ?[]f32) !void {
        const be = self.be;
        const seq = ids.len;
        const b = &self.bufs;

        try self.layersForward(ids);
        errdefer if (be.batching()) be.abortBatch();

        // Final norm on the last position + tied bf16 LM head, on device.
        const h = self.cfg.hidden;
        try be.qkNorm(offsetBufSized(b.x, (seq - 1) * h * 4, h * 4), b.t, try nbuf(be, self.lm.final_norm), 1, h, eps);
        try self.lmHeadGemv(b.logits, b.t);
        try be.endBatch();
        self.advance(seq);

        // `null` leaves logits resident for the on-device argmax (stepArgmax).
        if (logits) |l| try be.tensorDownload(offsetBufSized(b.logits, 0, qwen3.vocab_size * 4), std.mem.sliceAsBytes(l[0..qwen3.vocab_size]));
    }

    /// Greedy decode without the vocab download: forward `ids`, then argmax the
    /// last position's logits on-device and return just that token id. Mirrors
    /// `step`'s graph/chunk dispatch. Matches sample.argmax (temperature 0).
    pub fn stepArgmax(self: *CudaLM, io: std.Io, ids: []const u32) !u32 {
        return self.stepArgmaxPen(io, ids, &.{}, .{});
    }

    /// `stepArgmax` with sampling penalties scattered onto the device logits
    /// first (opPenalize; see sample.zig) — keeps penalized greedy decode
    /// on the GPU path instead of the full-vocab download.
    pub fn stepArgmaxPen(self: *CudaLM, io: std.Io, ids: []const u32, pen: []const sample.PenaltyEntry, sp: sample.Params) !u32 {
        self.io = io; // the CPU half of a hybrid split runs host matmuls through it
        if (self.be.evictions != 0) self.graph_ok = false;
        if (ids.len == 1 and self.graphEligible()) {
            if (self.decode_warm) {
                try self.stepDecodeGraph(ids[0], null);
                return self.argmaxLogits(pen, sp);
            }
            self.decode_warm = true;
        }
        var off: usize = 0;
        while (off < ids.len) {
            const n = @min(self.max_rows, ids.len - off);
            try self.stepChunk(ids[off..][0..n], null);
            off += n;
        }
        return self.argmaxLogits(pen, sp);
    }

    fn argmaxLogits(self: *CudaLM, pen: []const sample.PenaltyEntry, sp: sample.Params) !u32 {
        const be = self.be;
        const b = &self.bufs;
        try be.opPenalize(offsetBufSized(b.logits, 0, qwen3.vocab_size * 4), pen, sp);
        try be.opArgmax(offsetBufSized(b.logits, 0, qwen3.vocab_size * 4), qwen3.vocab_size, b.argmax_out, &b.argmax_v, &b.argmax_i);
        var id_f: [1]f32 = undefined;
        try be.tensorDownload(b.argmax_out, std.mem.sliceAsBytes(&id_f));
        return @intFromFloat(id_f[0]);
    }

    /// Max candidates stepSelect can return (host buffer sizing for the engine).
    pub fn maxSelect(self: *const CudaLM) usize {
        _ = self;
        return cuda.backend.topk_lanes * cuda.backend.topk_m;
    }

    /// Stochastic decode: forward `ids`, select the top-k on-device, download
    /// just those (id,logit) pairs (a few KB vs the ~608 KB vocab). Returns the
    /// candidate count; the engine's Sampler finishes on the CPU over this set.
    pub fn stepSelect(self: *CudaLM, io: std.Io, ids: []const u32, out_id: []u32, out_logit: []f32) !usize {
        return self.stepSelectPen(io, ids, &.{}, .{}, out_id, out_logit);
    }

    /// `stepSelect` with sampling penalties scattered onto the device logits
    /// before the top-k (opPenalize) — the selected candidates are the true
    /// post-penalty top set, so penalized stochastic decode stays on the GPU.
    pub fn stepSelectPen(self: *CudaLM, io: std.Io, ids: []const u32, pen: []const sample.PenaltyEntry, sp: sample.Params, out_id: []u32, out_logit: []f32) !usize {
        self.io = io; // the CPU half of a hybrid split runs host matmuls through it
        if (self.be.evictions != 0) self.graph_ok = false;
        if (ids.len == 1 and self.graphEligible() and self.decode_warm) {
            try self.stepDecodeGraph(ids[0], null);
        } else {
            if (ids.len == 1 and self.graphEligible()) self.decode_warm = true;
            var off: usize = 0;
            while (off < ids.len) {
                const n = @min(self.max_rows, ids.len - off);
                try self.stepChunk(ids[off..][0..n], null);
                off += n;
            }
        }
        const be = self.be;
        const b = &self.bufs;
        try be.opPenalize(offsetBufSized(b.logits, 0, qwen3.vocab_size * 4), pen, sp);
        const count = try be.opTopK(offsetBufSized(b.logits, 0, qwen3.vocab_size * 4), qwen3.vocab_size, &b.topk_v, &b.topk_i);
        std.debug.assert(count <= out_id.len and count <= out_logit.len);
        try be.tensorDownload(b.topk_v, std.mem.sliceAsBytes(out_logit[0..count]));
        var idx_f: [cuda.backend.topk_lanes * cuda.backend.topk_m]f32 = undefined;
        try be.tensorDownload(b.topk_i, std.mem.sliceAsBytes(idx_f[0..count]));
        for (out_id[0..count], idx_f[0..count]) |*o, f| o.* = @intFromFloat(f);
        return count;
    }

    /// The layer stack over `ids` at positions [len, len+seq): embedding
    /// upload, then the whole transformer inside an open batch. The caller
    /// finishes the batch (LM head variants differ) — on success the batch is
    /// still open, with the final hidden states in bufs.x.
    // --- transformer_gpu.decoderLayer stepper methods (faithful lifts of the
    // former layersForward inline body; taps + weight prefetch stay loop-level
    // hooks around the decoderLayer call). ---

    pub fn normInput(self: *CudaLM, layer: anytype, seq: usize) !void {
        const c = self.cfg;
        try self.be.qkNorm(self.bufs.x, self.bufs.normed, try nbuf(self.be, layer.input_norm), seq, c.hidden, eps);
    }
    pub fn projectQKV(self: *CudaLM, l: usize, layer: anytype, seq: usize) !void {
        _ = l; // qwen3: uniform geometry
        const c = self.cfg;
        const b = &self.bufs;
        try self.linear(b.q, b.normed, layer.q, c.qDim(), c.hidden, seq);
        try self.linear(b.k, b.normed, layer.k, c.kvDim(), c.hidden, seq);
        try self.linear(b.v, b.normed, layer.v, c.kvDim(), c.hidden, seq);
    }
    pub fn normQK(self: *CudaLM, l: usize, layer: anytype, seq: usize) !void {
        _ = l;
        const c = self.cfg;
        const b = &self.bufs;
        try self.be.qkNorm(b.q, b.q, try nbuf(self.be, layer.q_norm), seq * c.n_heads, hd, eps);
        try self.be.qkNorm(b.k, b.k, try nbuf(self.be, layer.k_norm), seq * c.n_kv_heads, hd, eps);
    }
    pub fn applyRope(self: *CudaLM, l: usize, seq: usize, pos0: usize) !void {
        _ = l; // qwen3: single rope table for all layers
        const c = self.cfg;
        const b = &self.bufs;
        try self.be.ropeHalf(b.q, self.freqs_d, seq, c.n_heads, half, self.sin_off, pos0);
        try self.be.ropeHalf(b.k, self.freqs_d, seq, c.n_kv_heads, half, self.sin_off, pos0);
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
    pub fn appendKV(self: *CudaLM, l: usize, seq: usize, pos0: usize) !void {
        const c = self.cfg;
        const b = &self.bufs;
        try self.storeKv(self.k_cache[l].buf, pos0 * c.kvDim(), b.k, 0, seq * c.kvDim());
        try self.storeKv(self.v_cache[l].buf, pos0 * c.kvDim(), b.v, 0, seq * c.kvDim());
    }
    pub fn attention(self: *CudaLM, l: usize, seq: usize, pos0: usize) !void {
        const c = self.cfg;
        const b = &self.bufs;
        // f16/q8_0 K/V is only read by the flash-split kernels (opAttnDecode);
        // the square `be.attn` path is f32-only, so non-f32 dtypes route all
        // prefill through opAttnDecode (correct for any seq_q, just less
        // parallel for big batches).
        if (self.kv_dtype != .f32 or seq <= gemv_batch_max) {
            // Batched flash-decoding: the naive square kernel has too little
            // parallelism for a handful of queries. Query batches BEYOND the
            // spec-verify size (f16/q8_0 prefill chunks, up to max_rows) take the
            // smaller prefill split count — attn_scratch is sized for
            // max_rows x nsplit_prefill, and a full nsplit there would
            // overrun it (the fault behind the GUI's "PTX JIT failed:
            // CUDA_ERROR_ILLEGAL_ADDRESS" on a long f16 first prompt). Small
            // batches keep nsplit so spec-verify logits stay bitwise
            // identical to the decode path.
            const ns: usize = if (seq <= gemv_batch_max) nsplit else nsplit_prefill;
            try self.be.opAttnDecode(b.q, self.k_cache[l].buf, self.v_cache[l].buf, b.attn, b.attn_scratch, pos0 + 1, seq, c.n_heads, c.n_kv_heads, hd, ns, attn_scale, 0, 0, false, kvFmt(self.kv_dtype));
        } else {
            try self.be.attn(b.q, self.k_cache[l].buf, self.v_cache[l].buf, b.attn, seq, pos0 + seq, c.n_heads, c.n_kv_heads, hd, attn_scale, true);
        }
    }
    pub fn projectO(self: *CudaLM, l: usize, layer: anytype, seq: usize) !void {
        _ = l;
        const c = self.cfg;
        try self.linear(self.bufs.t, self.bufs.attn, layer.o, c.hidden, c.qDim(), seq);
    }
    pub fn addResidual(self: *CudaLM, seq: usize) !void {
        try self.be.opAdd(self.bufs.x, self.bufs.t, seq * self.cfg.hidden);
    }
    pub fn normPreFfn(self: *CudaLM, layer: anytype, seq: usize) !void {
        const c = self.cfg;
        try self.be.qkNorm(self.bufs.x, self.bufs.normed, try nbuf(self.be, layer.post_norm), seq, c.hidden, eps);
    }
    pub fn projectGateUp(self: *CudaLM, layer: anytype, seq: usize) !void {
        const c = self.cfg;
        const b = &self.bufs;
        try self.linear(b.gate, b.normed, layer.gate, c.intermediate, c.hidden, seq);
        try self.linear(b.up, b.normed, layer.up, c.intermediate, c.hidden, seq);
    }
    pub fn activate(self: *CudaLM, comptime act: transformer.Activation, seq: usize) !void {
        const n = seq * self.cfg.intermediate;
        switch (act) {
            .silu_mul => try self.be.siluMul(self.bufs.gate, self.bufs.up, n),
            .gelu_tanh_mul => try self.be.geluMul(self.bufs.gate, self.bufs.up, n),
        }
    }
    pub fn projectDown(self: *CudaLM, layer: anytype, seq: usize) !void {
        const c = self.cfg;
        try self.linear(self.bufs.t, self.bufs.gate, layer.down, c.hidden, c.intermediate, seq);
    }

    fn layersForward(self: *CudaLM, ids: []const u32) !void {
        const gpa = self.gpa;
        const be = self.be;
        const c = self.cfg;
        const seq = ids.len;
        std.debug.assert(seq > 0 and seq <= self.remaining() and seq <= self.max_rows);
        const pos0 = self.len;

        // CPU: embedding gather (storage dtype -> f32), upload.
        const x = try gpa.alloc(f32, seq * c.hidden);
        defer gpa.free(x);
        try qwen3.embedTokens(self.lm.embed, ids, x);
        try be.tensorUpload(offsetBufSized(self.bufs.x, 0, seq * c.hidden * 4), std.mem.sliceAsBytes(x));

        const b = &self.bufs;
        try be.beginBatch();
        errdefer if (be.batching()) be.abortBatch();

        // Hybrid split: each chunk begins with the hidden on the device (bufs.x).
        if (self.split) |*sp| sp.on_host = false;

        for (self.lm.layers, 0..) |layer, l| {
            if (self.taps_on) {
                for (self.tap_layers, 0..) |tl, j| {
                    if (l == tl) try be.opCopyOff(self.tap_d, (j * self.capacity + pos0) * c.hidden, b.x, 0, seq * c.hidden);
                }
            }
            // Hybrid CPU/GPU split: run host-resident layers on the CPU via the
            // shared transformer.layerForward (the same body the cpu backend
            // uses), ferrying the hidden across the device<->host boundary only
            // when residency changes.
            if (self.split) |*sp| {
                if (!sp.on_gpu[l]) {
                    if (!sp.on_host) {
                        try be.tensorDownload(offsetBufSized(b.x, 0, seq * c.hidden * 4), std.mem.sliceAsBytes(sp.hx[0 .. seq * c.hidden]));
                        sp.on_host = true;
                    }
                    const host_io = self.io orelse return error.SplitIoUnset;
                    var sv = sp.scratch.viewSeq(seq, c);
                    try transformer.layerForward(transformer.qwen3_spec, .cached, host_io, gpa, layer, sp.hx[0 .. seq * c.hidden], seq, qwen3.dimsFor(c), sp.freqs, eps, &sp.cache, l, pos0, false, &sv);
                    continue;
                }
                if (sp.on_host) {
                    try be.tensorUpload(offsetBufSized(b.x, 0, seq * c.hidden * 4), std.mem.sliceAsBytes(sp.hx[0 .. seq * c.hidden]));
                    sp.on_host = false;
                }
            }
            try transformer_gpu.decoderLayer(transformer.qwen3_spec, self, layer, l, seq, pos0);
        }

        // If the last layers ran on the host (the migration order pushes the
        // last layers first, so this is the common case), bring the final
        // hidden back to the device for the final norm + LM head.
        if (self.split) |*sp| if (sp.on_host) {
            try be.tensorUpload(offsetBufSized(b.x, 0, seq * c.hidden * 4), std.mem.sliceAsBytes(sp.hx[0 .. seq * c.hidden]));
            sp.on_host = false;
        };
    }

    pub fn vocab(self: *const CudaLM) usize {
        _ = self;
        return qwen3.vocab_size;
    }

    /// LM head over one normed hidden row: y[vocab] = head @ x.
    fn lmHeadGemv(self: *CudaLM, y: Buf, x: Buf) !void {
        const head = self.lm.head;
        if (head.dtype.isBlockQuant()) {
            try self.be.opGemvQuant(head.dtype, y, x, head.bytes, 1.0, qwen3.vocab_size, self.cfg.hidden);
        } else {
            try self.be.opGemvBf16(y, x, head.bytes, 1.0, qwen3.vocab_size, self.cfg.hidden);
        }
    }

    /// LM head over `seq` normed rows (x is 4-row-group padded) into
    /// y [seq][vocab]: grouped bf16 GEMVs reading the weight once per 4
    /// inputs, or per-row fused GEMVs for ggml block-quant heads.
    fn lmHeadAll(self: *CudaLM, y: Buf, x: Buf, seq: usize) !void {
        const be = self.be;
        const h = self.cfg.hidden;
        const head = self.lm.head;
        if (head.dtype.isBlockQuant()) {
            for (0..seq) |i| {
                try be.opGemvQuant(
                    head.dtype,
                    offsetBufSized(y, i * qwen3.vocab_size * 4, qwen3.vocab_size * 4),
                    offsetBufSized(x, i * h * 4, h * 4),
                    head.bytes,
                    1.0,
                    qwen3.vocab_size,
                    h,
                );
            }
            return;
        }
        var off: usize = 0;
        while (off < seq) : (off += 4) {
            const n: usize = @min(4, seq - off); // annotated: @min would narrow to u3
            try be.opGemvBf16N(
                offsetBufSized(y, off * qwen3.vocab_size * 4, n * qwen3.vocab_size * 4),
                offsetBufSized(x, off * h * 4, 4 * h * 4),
                head.bytes,
                1.0,
                qwen3.vocab_size,
                h,
                n,
            );
        }
    }

    /// Dense linear over `seq` rows, kernel picked by weight dtype and batch
    /// size. fp8 (the 4B target): fused GEMV (1 row), grouped multi-input
    /// GEMV (small batches — speculative verify and short multi-turn
    /// prefills; each fp8 weight row is read once per 4 inputs), or the f16
    /// tensor-core GEMM (large prefills, where the dequant-to-f16 scratch
    /// round trip amortizes). bf16 (the 0.6B draft model): fused GEMV /
    /// grouped GEMVs for everything — draft prompts are small and the model
    /// tiny, so a dedicated GEMM path isn't worth it.
    fn linear(self: *CudaLM, y: Buf, x: Buf, w: ops.matmul.Weight, rows_out: usize, cols: usize, seq: usize) !void {
        const be = self.be;
        if (w.dtype.isBlockQuant()) {
            // GGUF quants: fused GEMV for decode. For small batches (speculative
            // verify — always <= spec_limits.max_draft+1 = 17 — and short prefills), the
            // grouped dp4a GEMV streams each weight ceil(seq/8)x: measured 5-20x
            // faster than the dequant-to-f16 GEMM below the crossover (qgemv-bench
            // on the 3090, ~n=40). Every block-quant (q4_k/q5_k/q6_k/q8_0) now has
            // a grouped kernel; larger seq amortizes the GEMM's one-shot dequant.
            if (seq == 1) {
                try be.opGemvQuant(w.dtype, y, x, w.bytes, w.scale, rows_out, cols);
            } else if (w.dtype.isBlockQuant() and seq <= grouped_gemv_max and
                cols % 256 == 0 and rows_out % 8 == 0)
            {
                try be.opGemvQuantizeX(x, seq * cols); // one q8 activation for all groups
                var off: usize = 0;
                while (off < seq) : (off += 8) {
                    const ng: usize = @min(8, seq - off); // annotated: @min would narrow to u4
                    try be.opGemvQuantQ8N(
                        w.dtype,
                        offsetBufSized(y, off * rows_out * 4, ng * rows_out * 4),
                        w.bytes,
                        w.scale,
                        rows_out,
                        cols,
                        ng,
                        off,
                        seq,
                    );
                }
            } else {
                try be.opMatmulQuant(w.dtype, y, x, seq, w.bytes, rows_out, cols);
            }
            return;
        }
        if (w.dtype != .f8_e4m3) {
            std.debug.assert(w.dtype == .bf16);
            if (seq == 1) {
                try be.opGemvBf16(y, x, w.bytes, w.scale, rows_out, cols);
            } else {
                var off: usize = 0;
                while (off < seq) : (off += 4) {
                    const n: usize = @min(4, seq - off); // annotated: @min would narrow to u3
                    try be.opGemvBf16N(
                        offsetBufSized(y, off * rows_out * 4, n * rows_out * 4),
                        offsetBufSized(x, off * cols * 4, 4 * cols * 4),
                        w.bytes,
                        w.scale,
                        rows_out,
                        cols,
                        n,
                    );
                }
            }
            return;
        }
        if (seq == 1) {
            try be.opGemvFp8(y, x, w.bytes, w.scale, rows_out, cols);
        } else if (seq <= gemv_batch_max) {
            var off: usize = 0;
            while (off < seq) : (off += 4) {
                const n: usize = @min(4, seq - off); // annotated: @min would narrow to u3
                try be.opGemvFp8N(
                    offsetBufSized(y, off * rows_out * 4, n * rows_out * 4),
                    offsetBufSized(x, off * cols * 4, 4 * cols * 4),
                    w.bytes,
                    w.scale,
                    rows_out,
                    cols,
                    n,
                );
            }
        } else {
            try be.opMatmulFp8(y, x, seq, w.bytes, w.scale, rows_out, cols);
        }
    }

    /// Largest batch that goes through grouped GEMVs + batched flash-decode
    /// instead of the GEMM + square-attention prefill path: covers every
    /// speculative verify batch, and ceil(seq/4) fused weight reads stay
    /// well below the GEMM's ~5x dequant-scratch traffic.
    const gemv_batch_max = spec_limits.max_draft + 1;

    /// q5_k/q6_k batches at or below this take the grouped dp4a GEMV instead of
    /// opMatmulQuant's dequant-to-f16 GEMM. Measured crossover (qgemv-bench,
    /// 3090): ~48 rows q5_k, ~35 q6_k; 40 matches qwen35's grouped_prefill_max
    /// and covers every speculative-verify batch (<= spec_limits.max_draft + 1 = 17).
    const grouped_gemv_max = 40;

    /// KV chunks per head in the decode attention split pass (one warp each:
    /// 32 heads x 32 splits x 32 lanes = 32k threads).
    const nsplit = 32;

    /// Split count for MULTI-QUERY opAttnDecode batches past the spec-verify
    /// size (the f16 prefill route): the query batch already provides
    /// parallelism, so fewer splits per head suffice — and the scratch for
    /// max_rows x nsplit would be huge. Matches gemma3_cuda.
    const nsplit_prefill = 8;
};

/// offsetBuf carrying an explicit size (tensorUpload/Download use db.size).
fn offsetBufSized(b: Buf, off_bytes: usize, size: usize) Buf {
    return .{ .buf = @enumFromInt(@intFromEnum(b.buf) + off_bytes), .mem = b.mem, .size = size };
}

const LmBufs = struct {
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

    fn init(be: *Backend, rows: usize, c: qwen3.Config) !LmBufs {
        const rp = std.mem.alignForward(usize, rows, 128); // GEMM outputs are 128-row padded
        const r4 = std.mem.alignForward(usize, rows, 4); // grouped-GEMV inputs are read 4 rows at a time
        var self: LmBufs = undefined;
        var created: usize = 0;
        errdefer inline for (fields, 0..) |name, i| {
            if (i < created) be.tensorDestroy(&@field(self, name));
        };
        const sizes = [fields.len]usize{
            rows * c.hidden * 4, // x
            r4 * c.hidden * 4, // normed
            rp * c.qDim() * 4, // q
            rp * c.kvDim() * 4, // k
            rp * c.kvDim() * 4, // v
            r4 * c.qDim() * 4, // attn
            rp * c.intermediate * 4, // gate
            rp * c.intermediate * 4, // up
            rp * c.hidden * 4, // t
            @max(CudaLM.gemv_batch_max * CudaLM.nsplit, rows * CudaLM.nsplit_prefill) * c.n_heads * (hd + 4) * 4, // attn_scratch (verify batch at nsplit; f16 prefill chunks at nsplit_prefill)
            (spec_limits.max_draft + 1) * qwen3.vocab_size * 4, // logits (verify writes a row per position)
            4096 * 4, // argmax_v (>= opArgmax lane count)
            4096 * 4, // argmax_i
            4, // argmax_out (1 id)
            cuda.backend.topk_lanes * cuda.backend.topk_m * 4, // topk_v
            cuda.backend.topk_lanes * cuda.backend.topk_m * 4, // topk_i
        };
        inline for (fields, sizes) |name, size| {
            @field(self, name) = try be.tensorCreate(size);
            created += 1;
        }
        return self;
    }

    fn deinit(self: *LmBufs, be: *Backend) void {
        inline for (fields) |name| be.tensorDestroy(&@field(self, name));
        self.* = undefined;
    }

    const fields = [_][]const u8{ "x", "normed", "q", "k", "v", "attn", "gate", "up", "t", "attn_scratch", "logits", "argmax_v", "argmax_i", "argmax_out", "topk_v", "topk_i" };
};

/// Tree-verify buffers (CudaLM.enableTree): node positions, the tree
/// split-kernel scratch (partials + the meta table at its tail), per-node
/// vocab logits, and per-node tap rows for the EAGLE-3 drafter.
const TreeBufs = struct {
    pos: Buf, // [max_tree_nodes] u32 absolute positions
    scratch: Buf, // attn partials [n][heads][nsplit][hd+4] + meta tail
    logits: Buf, // [max_tree_nodes][vocab]
    taps: Buf, // [3][max_tree_nodes][hidden] residual entering each tap layer

    fn init(be: *Backend, c: qwen3.Config) !TreeBufs {
        const m = spec_limits.max_tree_nodes;
        var self: TreeBufs = undefined;
        var created: usize = 0;
        errdefer inline for (fields, 0..) |name, i| {
            if (i < created) be.tensorDestroy(&@field(self, name));
        };
        const sizes = [fields.len]usize{
            m * 4, // pos
            (m * c.n_heads * CudaLM.nsplit * (hd + 4) + m * (m + 1)) * 4, // scratch
            m * qwen3.vocab_size * 4, // logits
            3 * m * c.hidden * 4, // taps
        };
        inline for (fields, sizes) |name, size| {
            @field(self, name) = try be.tensorCreate(size);
            created += 1;
        }
        return self;
    }

    fn deinit(self: *TreeBufs, be: *Backend) void {
        inline for (fields) |name| be.tensorDestroy(&@field(self, name));
        self.* = undefined;
    }

    const fields = [_][]const u8{ "pos", "scratch", "logits", "taps" };
};

// Gated on a CUDA device + the checkpoint: tree-verify greedy output
// through CudaLM (enableTree + stepAllTree + commitTreePath, the tree
// attention/rope kernels) must equal vanilla greedy. The n-gram chain rides
// ChainAsTree so real drafts and rejections flow through the tree path; the
// repetitive prompt guarantees accepted drafts. Kept tiny — Debug forwards
// are slow.
test "cuda tree spec matches vanilla greedy on the real model" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const engine = @import("../llm/engine.zig");
    const chat = @import("../llm/chat.zig");
    const tokenizer_mod = @import("tp_core").tokenizer;
    const te_path = "models/text_encoders/qwen3VLInstruct4bHeretic_v10.safetensors";
    std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    std.Io.Dir.cwd().access(io, te_path, .{}) catch return error.SkipZigTest;
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    var st = try safetensors.SafeTensors.open(gpa, io, te_path);
    defer st.deinit();
    var lm = try qwen3.CausalLM.load(gpa, .{ .safetensors = &st });
    defer lm.deinit();
    var tok = try tokenizer_mod.Tokenizer.init(gpa);
    defer tok.deinit();

    var opts: engine.Options = .{ .max_new_tokens = 4, .sampling = .{ .temperature = 0 } };
    var ids_vanilla: std.ArrayList(u32) = .empty;
    defer ids_vanilla.deinit(gpa);
    try chat.appendUser(&tok, gpa, "Count: one two three one two", &ids_vanilla);
    try chat.openAssistant(&tok, gpa, &ids_vanilla);
    var ids_tree: std.ArrayList(u32) = .empty;
    defer ids_tree.deinit(gpa);
    try ids_tree.appendSlice(gpa, ids_vanilla.items);

    {
        var model = try CudaLM.init(gpa, be, &lm, .fixed(try engine.capacityFor(opts, ids_vanilla.items.len)), ids_vanilla.items.len);
        defer model.deinit();
        _ = try engine.generate(&model, &tok, io, gpa, &ids_vanilla, opts, null);
    }
    {
        opts.tree_nodes = 4;
        var stats: spec.Stats = .{};
        opts.spec_stats = &stats;
        var model = try CudaLM.init(gpa, be, &lm, .fixed(try engine.capacityFor(opts, ids_tree.items.len)), ids_tree.items.len);
        defer model.deinit();
        try model.enableTree();
        var ngram: spec.NgramDrafter = .{};
        var drafter: spec.ChainAsTree(spec.NgramDrafter) = .{ .inner = &ngram };
        _ = try spec.generate(&model, &drafter, &tok, io, gpa, &ids_tree, opts, null);
        try std.testing.expect(stats.forwards > 0);
    }
    try std.testing.expectEqualSlices(u32, ids_vanilla.items, ids_tree.items);
}

// Hybrid CPU/GPU split mechanics on the real model: a tight budget places
// layers on the host, generation runs (host layers through the shared
// transformer.layerForward), the host KV shadow stays in lockstep with the
// device counter through generate / rollback (a rollback + regenerate with
// unchanged residency must be token-identical — a broken host truncate would
// diverge), and promoteLayers restores full device residency.
fn cpuSplitGenerateBody(kv_dtype: kvmod.KvDtype) !void {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const engine = @import("../llm/engine.zig");
    const chat = @import("../llm/chat.zig");
    const tokenizer_mod = @import("tp_core").tokenizer;
    const te_path = "models/text_encoders/qwen3VLInstruct4bHeretic_v10.safetensors";
    std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    std.Io.Dir.cwd().access(io, te_path, .{}) catch return error.SkipZigTest;
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    var st = try safetensors.SafeTensors.open(gpa, io, te_path);
    defer st.deinit();
    var lm = try qwen3.CausalLM.load(gpa, .{ .safetensors = &st });
    defer lm.deinit();
    var tok = try tokenizer_mod.Tokenizer.init(gpa);
    defer tok.deinit();

    const opts: engine.Options = .{ .max_new_tokens = 3, .sampling = .{ .temperature = 0 } };
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    try chat.appendUser(&tok, gpa, "Count: one two three one two", &ids);
    try chat.openAssistant(&tok, gpa, &ids);
    const prompt_len = ids.items.len;

    var model = try CudaLM.init(gpa, be, &lm, .{ .initial = 64, .max = 256, .kv_dtype = kv_dtype }, 512);
    defer model.deinit();

    // ~3.6 GB of fp8 layer weights + a ~0.8 GB bf16 head: a 4 GiB budget
    // plans a handful of tail layers onto the host (Debug host forwards are
    // slow, so keep the CPU share small).
    try model.enableCpuSplit(.attn, 4 << 30, true);
    const sp = &model.split.?;
    errdefer std.debug.print("n_cpu={d}\n", .{sp.n_cpu});
    try std.testing.expect(sp.n_cpu > 0);
    try std.testing.expect(sp.n_cpu < model.cfg.n_layers);
    // The host shadow stores the session dtype (raw-byte migrate/promote).
    try std.testing.expectEqual(kv_dtype, sp.cache.kv_dtype);

    // The host matmuls need a seeded Io before the first forward — prefill
    // takes no io, so unseeded must fail closed (not undefined-pointer UB),
    // and the session owner seeds it (the GUI does at init).
    try std.testing.expectError(error.SplitIoUnset, model.prefill(ids.items[0 .. prompt_len - 1]));
    try std.testing.expectEqual(@as(usize, 0), model.cached());
    model.io = io;

    try model.ensureCapacity(prompt_len);
    try model.prefill(ids.items[0 .. prompt_len - 1]);
    const q = model.cached();
    try std.testing.expectEqual(q, sp.cache.len); // host shadow in lockstep
    var empty: [0]u8 = .{};
    try model.checkpoint(&empty);

    const n1 = try engine.generate(&model, &tok, io, gpa, &ids, opts, null);
    try std.testing.expect(n1 > 0);
    try std.testing.expectEqual(model.cached(), sp.cache.len);
    const take1 = try gpa.dupe(u32, ids.items[prompt_len..]);
    defer gpa.free(take1);

    // Rollback + regenerate with unchanged residency: bitwise-identical run.
    try model.restoreCheckpoint(&.{}, q);
    try std.testing.expectEqual(q, model.cached());
    try std.testing.expectEqual(q, sp.cache.len);
    ids.shrinkRetainingCapacity(prompt_len);
    const n2 = try engine.generate(&model, &tok, io, gpa, &ids, opts, null);
    errdefer std.debug.print("take1 {any} vs regenerated {any}\n", .{ take1, ids.items[prompt_len..] });
    try std.testing.expectEqual(n1, n2);
    try std.testing.expectEqualSlices(u32, take1, ids.items[prompt_len..]);

    // Promote everything back: full device residency, and decode still runs
    // (the captured-graph path becomes eligible again).
    const promoted = try model.promoteLayers(std.math.maxInt(u64));
    try std.testing.expect(promoted > 0);
    try std.testing.expectEqual(@as(usize, 0), model.split.?.n_cpu);
    try model.restoreCheckpoint(&.{}, q);
    ids.shrinkRetainingCapacity(prompt_len);
    const n3 = try engine.generate(&model, &tok, io, gpa, &ids, opts, null);
    try std.testing.expect(n3 > 0);
    try std.testing.expectEqual(model.cached(), model.split.?.cache.len);
}

test "cuda cpu split: host layers generate, rollback stays in sync, promote restores" {
    try cpuSplitGenerateBody(.f32);
}

// f16 KV + CPU offload (TODO "f16 kv cache + cpu offload"): the host shadow
// stores packed f16 — byte-identical to the device cache — so the hybrid
// split arms on f16 configs too, instead of leaving a near-VRAM-size model
// with no offload path.
test "cuda cpu split with f16 kv: host layers generate, rollback stays in sync, promote restores" {
    try cpuSplitGenerateBody(.f16);
}

// q8_0 KV + CPU offload: the host shadow stores raw ggml q8_0 blocks
// (byte-identical to the device cache — host packQ80 and the device cvt.rni
// kernels round identically), so migrate/promote raw copies stay lossless and
// the split generates through host layers on a quantized cache.
test "cuda cpu split with q8_0 kv: host layers generate, rollback stays in sync, promote restores" {
    try cpuSplitGenerateBody(.q8_0);
}

// The GUI dtype toggle (chat.Session.rebuildContext) calls reinitCache with
// the offload split still armed: host-resident layers must keep NO device KV,
// the host shadow must be rebuilt at the new dtype, and a full re-prefill
// (the GUI's context reload) must generate in both directions.
test "cuda cpu split survives a kv dtype toggle (reinitCache)" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const te_path = "models/text_encoders/qwen3VLInstruct4bHeretic_v10.safetensors";
    std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    std.Io.Dir.cwd().access(io, te_path, .{}) catch return error.SkipZigTest;
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    var st = try safetensors.SafeTensors.open(gpa, io, te_path);
    defer st.deinit();
    var lm = try qwen3.CausalLM.load(gpa, .{ .safetensors = &st });
    defer lm.deinit();
    var model = try CudaLM.init(gpa, be, &lm, .{ .initial = 64, .max = 256 }, 512);
    defer model.deinit();

    try model.enableCpuSplit(.attn, 4 << 30, true);
    try std.testing.expect(model.split.?.n_cpu > 0);
    model.io = io;

    var prompt: [24]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(1000 + (i * 37) % 50000);

    for ([_]kvmod.KvDtype{ .f16, .q8_0, .f32 }) |dt| {
        const n_cpu_before = model.split.?.n_cpu;
        try model.reinitCache(dt);
        const sp = &model.split.?;
        try std.testing.expectEqual(dt, model.kv_dtype);
        try std.testing.expectEqual(dt, sp.cache.kv_dtype);
        try std.testing.expectEqual(@as(usize, 0), model.cached());
        try std.testing.expectEqual(@as(usize, 0), sp.cache.len);
        try std.testing.expectEqual(n_cpu_before, sp.n_cpu); // placement survives
        try model.prefill(prompt[0 .. prompt.len - 1]);
        try std.testing.expectEqual(prompt.len - 1, model.cached());
        try std.testing.expectEqual(prompt.len - 1, sp.cache.len);
        const next = try model.stepArgmax(io, prompt[prompt.len - 1 ..]);
        try std.testing.expect(next < qwen3.vocab_size);
    }
}

// The GUI session (gui/chat.zig) drives this stepper through prefill /
// zero-byte turn checkpoints / resetResidency. Exercise a full GUI turn shape
// on the real model: prefill the prompt boundary, greedy-generate, roll back
// to the boundary and regenerate (append-only KV, restore = truncate) —
// tokens must be identical — then a residency reset must yield an empty,
// immediately reusable context that replays the same tokens from scratch.
test "cuda gui turn lifecycle: prefill, checkpoint rollback, residency reset" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const engine = @import("../llm/engine.zig");
    const chat = @import("../llm/chat.zig");
    const tokenizer_mod = @import("tp_core").tokenizer;
    const te_path = "models/text_encoders/qwen3VLInstruct4bHeretic_v10.safetensors";
    std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    std.Io.Dir.cwd().access(io, te_path, .{}) catch return error.SkipZigTest;
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    var st = try safetensors.SafeTensors.open(gpa, io, te_path);
    defer st.deinit();
    var lm = try qwen3.CausalLM.load(gpa, .{ .safetensors = &st });
    defer lm.deinit();
    var tok = try tokenizer_mod.Tokenizer.init(gpa);
    defer tok.deinit();

    const opts: engine.Options = .{ .max_new_tokens = 4, .sampling = .{ .temperature = 0 } };
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    try chat.appendUser(&tok, gpa, "Count: one two three one two", &ids);
    try chat.openAssistant(&tok, gpa, &ids);
    const prompt_len = ids.items.len;

    // Deliberately small initial capacity: the turn grows it (in-place), so
    // resetResidency below exercises the real shrink-back + graph-drop path.
    var model = try CudaLM.init(gpa, be, &lm, .{ .initial = 16, .max = 256 }, 512);
    defer model.deinit();

    // GUI turn shape (gui/chat.zig prepareTurn): grow to the prompt, prefill
    // everything but the last token, snapshot the boundary (zero bytes for
    // append-only qwen3), then generate.
    try model.ensureCapacity(prompt_len);
    try model.prefill(ids.items[0 .. prompt_len - 1]);
    const q = model.cached();
    try std.testing.expectEqual(prompt_len - 1, q);
    try std.testing.expectEqual(@as(usize, 0), model.checkpointBytes());
    var empty: [0]u8 = .{};
    try model.checkpoint(&empty);
    const n1 = try engine.generate(&model, &tok, io, gpa, &ids, opts, null);
    try std.testing.expect(n1 > 0);
    const take1 = try gpa.dupe(u32, ids.items[prompt_len..]);
    defer gpa.free(take1);

    // Regenerate: roll back to the boundary and replay — greedy must repeat.
    try model.restoreCheckpoint(&.{}, q);
    try std.testing.expectEqual(q, model.cached());
    ids.shrinkRetainingCapacity(prompt_len);
    const n2 = try engine.generate(&model, &tok, io, gpa, &ids, opts, null);
    errdefer std.debug.print("take1 {any} vs regenerated {any}\n", .{ take1, ids.items[prompt_len..] });
    try std.testing.expectEqual(n1, n2);
    try std.testing.expectEqualSlices(u32, take1, ids.items[prompt_len..]);

    // New chat: the reset clears the context back to the initial committed
    // capacity (the turn grew it above, so this frees + re-creates the KV
    // buffers and drops the captured graph); a from-scratch replay of the
    // same prompt repeats the tokens.
    try std.testing.expect(model.cached() + model.remaining() > 16); // grew mid-turn
    try model.resetResidency(0);
    try std.testing.expectEqual(@as(usize, 0), model.cached());
    try std.testing.expectEqual(@as(usize, 16), model.cached() + model.remaining());
    ids.shrinkRetainingCapacity(prompt_len);
    try model.ensureCapacity(prompt_len);
    try model.prefill(ids.items[0 .. prompt_len - 1]);
    const n3 = try engine.generate(&model, &tok, io, gpa, &ids, opts, null);
    errdefer std.debug.print("take1 {any} vs post-reset {any}\n", .{ take1, ids.items[prompt_len..] });
    try std.testing.expectEqual(n1, n3);
    try std.testing.expectEqualSlices(u32, take1, ids.items[prompt_len..]);
}

const Bufs = struct {
    x: Buf, // residual stream [seq][hidden]
    normed: Buf,
    q: Buf,
    k: Buf,
    v: Buf,
    attn: Buf,
    gate: Buf,
    up: Buf,
    t: Buf,
    out: Buf, // tap-major [tap][seq][hidden]

    fn init(be: *Backend, seq: usize, seq_pad: usize) !Bufs {
        var self: Bufs = undefined;
        var created: usize = 0;
        errdefer inline for (fields, 0..) |name, i| {
            if (i < created) be.tensorDestroy(&@field(self, name));
        };
        // GEMM outputs (q/k/v/gate/up/t) are 128-row padded (pad rows are zero);
        // x/normed/attn are indexed by real seq.
        const sizes = [fields.len]usize{
            seq * hidden * 4, // x
            seq * hidden * 4, // normed
            seq_pad * q_dim * 4, // q
            seq_pad * kv_dim * 4, // k
            seq_pad * kv_dim * 4, // v
            seq * q_dim * 4, // attn
            seq_pad * intermediate * 4, // gate
            seq_pad * intermediate * 4, // up
            seq_pad * hidden * 4, // t
            tap_count * seq * hidden * 4, // out
        };
        inline for (fields, sizes) |name, size| {
            @field(self, name) = try be.tensorCreate(size);
            created += 1;
        }
        return self;
    }

    fn deinit(self: *Bufs, be: *Backend) void {
        inline for (fields) |name| be.tensorDestroy(&@field(self, name));
        self.* = undefined;
    }

    const fields = [_][]const u8{ "x", "normed", "q", "k", "v", "attn", "gate", "up", "t", "out" };
};

// qwen3 twin of the gemma3/qwen35 live-VRAM plan test: with most of the card
// occupied, a generous budget must still plan layers onto the host (the clamp
// by `min(budget, used + headroom)` in enableCpuSplit).
test "cpu split plan respects live free VRAM" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const test_gate = @import("../test_gate.zig");
    const te_path = "models/text_encoders/qwen3VLInstruct4bHeretic_v10.safetensors";
    std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    std.Io.Dir.cwd().access(io, te_path, .{}) catch return error.SkipZigTest;
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    var st = try safetensors.SafeTensors.open(gpa, io, te_path);
    defer st.deinit();
    var lm = try qwen3.CausalLM.load(gpa, .{ .safetensors = &st });
    defer lm.deinit();
    var model = try CudaLM.init(gpa, be, &lm, .{ .initial = 64, .max = 256 }, 512);
    defer model.deinit();

    // Leave ~2 GiB free: less than the ~3.6 GiB of fp8 layer weights, so a
    // live-aware plan MUST place some layers on the host.
    var balloon = try test_gate.VramBalloon.inflateToFree(gpa, be, 2 << 30);
    defer balloon.deinit();

    try model.enableCpuSplit(.attn, 1 << 40, true); // budget far beyond the card
    errdefer std.debug.print("n_cpu={d}, free={d} MiB\n", .{
        if (model.split) |sp| sp.n_cpu else 0, be.ctx.memGetInfo().free >> 20,
    });
    try std.testing.expect(model.split != null);
    try std.testing.expect(model.split.?.n_cpu > 0);
}

// f16 KV routes ALL prefill through the flash-decode attention kernel; a
// query batch larger than the spec-verify size must take the prefill split
// count — with the full nsplit it overran attn_scratch (an async illegal
// address, surfacing later as "PTX JIT failed: CUDA_ERROR_ILLEGAL_ADDRESS"
// and poisoning the process; hit by tp-gui's multi-thousand-token system
// prompt on an f16-KV config).
test "cuda f16 kv long prefill does not overrun the attention scratch" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const te_path = "models/text_encoders/qwen3VLInstruct4bHeretic_v10.safetensors";
    std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    std.Io.Dir.cwd().access(io, te_path, .{}) catch return error.SkipZigTest;
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    var st = try safetensors.SafeTensors.open(gpa, io, te_path);
    defer st.deinit();
    var lm = try qwen3.CausalLM.load(gpa, .{ .safetensors = &st });
    defer lm.deinit();
    // 512-row activation buffers (the GUI sizing): a 700-token prefill runs
    // as a full 512-row f16 chunk plus a remainder — both far beyond the
    // spec-verify batch the scratch used to be sized for.
    var model = try CudaLM.init(gpa, be, &lm, .{ .initial = 1024, .max = 1024, .kv_dtype = .f16 }, 512);
    defer model.deinit();

    var prompt: [700]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(1000 + (i * 37) % 50000);
    try model.prefill(prompt[0 .. prompt.len - 1]);
    try std.testing.expectEqual(prompt.len - 1, model.cached());
    const next = try model.stepArgmax(io, prompt[prompt.len - 1 ..]);
    try std.testing.expect(next < qwen3.vocab_size);
}
