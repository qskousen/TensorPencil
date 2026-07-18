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
//! fp8 weights stream through the Backend weight cache, so a small --vram-budget
//! degrades to weight streaming here exactly as it does for the DiT.

const std = @import("std");
const qwen3 = @import("qwen3.zig");
const cuda = @import("../gpu/cuda.zig");
const safetensors = @import("../safetensors.zig");
const ops = @import("../ops.zig");
const spec = @import("../llm/spec.zig");
const kvmod = @import("../llm/kv_cache.zig");
const transformer = @import("transformer.zig");
const transformer_gpu = @import("transformer_gpu.zig");

const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;
const Growable = Backend.GrowableTensor;

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
    /// at rows [capacity, capacity + spec.max_tree_nodes) of the (enlarged)
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
        self.max_capacity = cap.max;
        self.kv_dtype = cap.kv_dtype;
        self.len = 0;
        // Activation buffers always cover a speculative verify batch; padded
        // GEMM buffers are 128-row anyway, so the floor is nearly free.
        self.max_rows = @max(@max(first_seq, 1), spec.max_draft + 1);
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

        const esz = cap.kv_dtype.elemBytes();
        var created: usize = 0;
        errdefer for (self.k_cache[0..created]) |*b| be.growableDestroy(b);
        for (self.k_cache[0..c.n_layers]) |*b| {
            b.* = try be.growableCreate(cap.initial * c.kvDim() * esz, cap.max * c.kvDim() * esz);
            created += 1;
        }
        var vcreated: usize = 0;
        errdefer for (self.v_cache[0..vcreated]) |*b| be.growableDestroy(b);
        for (self.v_cache[0..c.n_layers]) |*b| {
            b.* = try be.growableCreate(cap.initial * c.kvDim() * esz, cap.max * c.kvDim() * esz);
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
        if (self.kv_dtype == .f16) return error.KvDtypeUnsupported;
        std.debug.assert(!self.taps_on);
        for (layers) |l| std.debug.assert(l < self.cfg.n_layers);
        self.tap_d = try self.be.tensorCreate(3 * self.capacity * self.cfg.hidden * 4);
        self.tap_layers = layers;
        self.taps_on = true;
    }

    /// Enable the tree-verify path (spec.generateTree): rebuilds the K/V
    /// caches with spec.max_tree_nodes extra rows (the batch region — tree
    /// nodes cannot append linearly, sibling branches collide at the same
    /// position), grows the activation buffers to cover a full tree batch,
    /// and allocates the tree buffers. Call before any forward.
    pub fn enableTree(self: *CudaLM) !void {
        if (self.kv_dtype == .f16) return error.KvDtypeUnsupported; // tree kernels are f32-only
        const be = self.be;
        const c = self.cfg;
        std.debug.assert(self.tree == null and self.len == 0);

        // The batch region sits at rows [capacity, capacity + max_tree_nodes)
        // — capacity is baked into the layout, so tree sessions stay fixed.
        std.debug.assert(self.capacity == self.max_capacity);
        const kv_bytes = (self.capacity + spec.max_tree_nodes) * c.kvDim() * 4;
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
        if (self.max_rows < spec.max_tree_nodes) {
            const bufs = try LmBufs.init(be, spec.max_tree_nodes, c);
            self.bufs.deinit(be);
            self.bufs = bufs;
            self.max_rows = spec.max_tree_nodes;
        }
        for (self.k_cache[0..c.n_layers]) |*b| be.growableDestroy(b);
        for (self.v_cache[0..c.n_layers]) |*b| be.growableDestroy(b);
        self.k_cache = nk;
        self.v_cache = nv;
        self.tree = tb;
    }

    /// Rebuild the KV cache at a new element dtype (GUI toggle), weights resident.
    /// Drops the captured decode graph so it re-captures with the f16/f32 kernels,
    /// frees + re-creates the K/V buffers, and resets the length. Rejected while
    /// EAGLE taps / tree-verify are active (those are f32-only).
    pub fn reinitCache(self: *CudaLM, dtype: kvmod.KvDtype) !void {
        if (dtype == .f16 and (self.taps_on or self.tree != null)) return error.KvDtypeUnsupported;
        const be = self.be;
        const c = self.cfg;
        if (self.graph_exec != null) {
            be.graphDestroy(self.graph_exec);
            self.graph_exec = null;
        }
        self.decode_warm = false;
        for (self.k_cache[0..c.n_layers]) |*b| be.growableDestroy(b);
        for (self.v_cache[0..c.n_layers]) |*b| be.growableDestroy(b);
        self.kv_dtype = dtype;
        self.len = 0;
        const esz = dtype.elemBytes();
        for (self.k_cache[0..c.n_layers]) |*b| b.* = try be.growableCreate(self.capacity * c.kvDim() * esz, self.max_capacity * c.kvDim() * esz);
        for (self.v_cache[0..c.n_layers]) |*b| b.* = try be.growableCreate(self.capacity * c.kvDim() * esz, self.max_capacity * c.kvDim() * esz);
    }

    pub fn deinit(self: *CudaLM) void {
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
    /// stay valid. Under VRAM pressure the commit evicts LRU weights into
    /// the streaming path, which flips the graph off via the evictions guard
    /// in step(). error.ContextFull past the window, when even eviction
    /// can't free enough device memory, or when the tap/tree layouts (which
    /// stride by capacity) pin the session to a fixed size.
    pub fn ensureCapacity(self: *CudaLM, min_rows: usize) !void {
        if (min_rows <= self.capacity) return;
        if (min_rows > self.max_capacity) return error.ContextFull;
        if (self.taps_on or self.tree != null) return error.ContextFull;
        const target = kvmod.growTarget(self.capacity, min_rows, self.max_capacity);
        const bytes = target * self.cfg.kvDim() * self.kv_dtype.elemBytes();
        for ([2][]Growable{ self.k_cache[0..self.cfg.n_layers], self.v_cache[0..self.cfg.n_layers] }) |caches| {
            for (caches) |*b| {
                self.be.growableEnsure(b, bytes) catch |err| switch (err) {
                    error.DeviceOutOfMemory, error.OutOfMemory => return error.ContextFull,
                    else => return err,
                };
            }
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
        // Any weight eviction (--vram-budget streaming, or live VRAM pressure
        // from another process) means device weight pointers are not stable,
        // and a captured graph would replay against freed buffers.
        if (self.be.evictions != 0) self.graph_ok = false;
        if (ids.len == 1 and self.graph_ok and !self.be.profile) {
            if (self.decode_warm) return self.stepDecodeGraph(io, ids[0], logits);
            self.decode_warm = true;
        }
        var off: usize = 0;
        while (off < ids.len) {
            const n = @min(self.max_rows, ids.len - off);
            try self.stepChunk(io, ids[off..][0..n], logits);
            off += n;
        }
    }

    fn stepDecodeGraph(self: *CudaLM, io: std.Io, id: u32, logits: ?[]f32) !void {
        const be = self.be;
        std.debug.assert(self.remaining() >= 1);
        try be.setDecodeState(id, @intCast(self.len));
        if (self.graph_exec == null) {
            self.captureDecodeGraph() catch |err| {
                // Leave graph mode permanently and decode this token normally.
                std.log.warn("decode graph capture failed ({t}); falling back to per-op launches", .{err});
                self.graph_ok = false;
                return self.stepChunk(io, &.{id}, logits);
            };
        }
        if (be.evictions != 0) {
            // Capture itself ran the cache over budget: the fresh graph may
            // already hold evicted-weight pointers. Decode per-op instead.
            self.graph_ok = false;
            return self.stepChunk(io, &.{id}, logits);
        }
        try be.graphLaunch(self.graph_exec);
        self.len += 1;
        // `null` leaves logits resident for the on-device argmax (stepArgmax).
        if (logits) |l| try be.tensorDownload(offsetBufSized(self.bufs.logits, 0, qwen3.vocab_size * 4), std.mem.sliceAsBytes(l[0..qwen3.vocab_size]));
    }

    fn captureDecodeGraph(self: *CudaLM) !void {
        const be = self.be;
        try be.graphCaptureBegin();
        errdefer if (be.graphCaptureEnd()) |exec| be.graphDestroy(exec) else |_| {};
        try self.recordDecodeOps();
        self.graph_exec = try be.graphCaptureEnd();
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
        if (self.lm.embed.dtype == .bf16) {
            try be.opEmbedGatherS(offsetBufSized(b.x, 0, c.hidden * 4), self.lm.embed.bytes, c.hidden);
        } else {
            try be.opEmbedGatherQuant(self.lm.embed.dtype, offsetBufSized(b.x, 0, c.hidden * 4), self.lm.embed.bytes, c.hidden);
        }
        for (self.lm.layers, 0..) |layer, l| {
            if (self.taps_on) {
                for (self.tap_layers, 0..) |tl, j| {
                    // Taps are f32 hidden-state snapshots (EAGLE), never the KV cache.
                    if (l == tl) try be.opKvAppendS(self.tap_d, b.x, c.hidden, c.hidden, j * self.capacity * c.hidden, false);
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
            const kv_f16 = self.kv_dtype == .f16;
            try be.opKvAppendS(self.k_cache[l].buf, b.k, c.kvDim(), c.kvDim(), 0, kv_f16);
            try be.opKvAppendS(self.v_cache[l].buf, b.v, c.kvDim(), c.kvDim(), 0, kv_f16);
            try be.opAttnDecodeSGraph(b.q, self.k_cache[l].buf, self.v_cache[l].buf, b.attn, b.attn_scratch, c.n_heads, c.n_kv_heads, hd, nsplit, attn_scale, kv_f16);
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
    /// engine-capped at spec.max_draft + 1, which max_rows always covers.
    pub fn stepAll(self: *CudaLM, io: std.Io, ids: []const u32, logits: []f32) !void {
        _ = io;
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
        std.debug.assert(seq > 0 and seq <= spec.max_draft + 1 and seq <= self.max_rows);
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
        self.len += seq;
    }

    /// stepAll's greedy twin: the verify forward, then a per-row on-device
    /// argmax into out_ids[0..seq] — downloading `seq` ids, not seq*vocab
    /// (~608 KB/row). For greedy speculative decoding, where acceptance only
    /// compares each draft to the target's argmax.
    pub fn stepAllArgmax(self: *CudaLM, io: std.Io, ids: []const u32, out_ids: []u32) !void {
        _ = io;
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
        std.debug.assert(n >= 1 and n <= spec.max_tree_nodes and n <= self.max_rows);
        std.debug.assert(parents.len == n and logits.len == n * qwen3.vocab_size);
        const tb = &self.tree.?;
        const b = &self.bufs;

        // Host: depth-based positions + the attention meta table (per-query
        // kv_len, then the ancestor NODE indices in depth order — the split
        // kernel adds the batch-region base itself).
        var pos: [spec.max_tree_nodes]u32 = undefined;
        var depth: [spec.max_tree_nodes]u32 = undefined;
        pos[0] = @intCast(self.len);
        depth[0] = 0;
        for (parents[1..], 1..) |p, i| {
            std.debug.assert(p < i);
            depth[i] = depth[p] + 1;
            pos[i] = pos[0] + depth[i];
            std.debug.assert(pos[i] < self.capacity);
        }
        var meta: [spec.max_tree_nodes * (spec.max_tree_nodes + 1)]u32 = undefined;
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
        self.prefetchLayer(0);
        for (self.lm.layers, 0..) |layer, l| {
            self.prefetchLayer(l + 1); // == layers.len prefetches the LM head
            if (self.taps_on) {
                for (self.tap_layers, 0..) |tl, j| {
                    if (l == tl) try be.opCopyOff(tb.taps, j * spec.max_tree_nodes * c.hidden, b.x, 0, n * c.hidden);
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
                    try be.opCopyOff(self.tap_d, (t * self.capacity + self.len + j) * c.hidden, tb.taps, (t * spec.max_tree_nodes + idx) * c.hidden, c.hidden);
                }
            }
        }
        self.len += path.len;
    }

    fn stepChunk(self: *CudaLM, io: std.Io, ids: []const u32, logits: ?[]f32) !void {
        const be = self.be;
        const seq = ids.len;
        const b = &self.bufs;

        try self.layersForward(ids);
        errdefer if (be.batching()) be.abortBatch();

        // Final norm on the last position + tied bf16 LM head, on device.
        const h = self.cfg.hidden;
        try be.qkNorm(offsetBufSized(b.x, (seq - 1) * h * 4, h * 4), b.t, try nbuf(be, self.lm.final_norm), 1, h, eps);
        try self.lmHeadGemv(b.logits, b.t);
        self.prefetchLayer(0); // next token's first layer DMAs behind the LM head
        try be.endBatch();
        self.len += seq;

        // `null` leaves logits resident for the on-device argmax (stepArgmax).
        if (logits) |l| try be.tensorDownload(offsetBufSized(b.logits, 0, qwen3.vocab_size * 4), std.mem.sliceAsBytes(l[0..qwen3.vocab_size]));
        _ = io;
    }

    /// Greedy decode without the vocab download: forward `ids`, then argmax the
    /// last position's logits on-device and return just that token id. Mirrors
    /// `step`'s graph/chunk dispatch. Matches sample.argmax (temperature 0).
    pub fn stepArgmax(self: *CudaLM, io: std.Io, ids: []const u32) !u32 {
        if (self.be.evictions != 0) self.graph_ok = false;
        if (ids.len == 1 and self.graph_ok and !self.be.profile) {
            if (self.decode_warm) {
                try self.stepDecodeGraph(io, ids[0], null);
                return self.argmaxLogits();
            }
            self.decode_warm = true;
        }
        var off: usize = 0;
        while (off < ids.len) {
            const n = @min(self.max_rows, ids.len - off);
            try self.stepChunk(io, ids[off..][0..n], null);
            off += n;
        }
        return self.argmaxLogits();
    }

    fn argmaxLogits(self: *CudaLM) !u32 {
        const be = self.be;
        const b = &self.bufs;
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
        if (self.be.evictions != 0) self.graph_ok = false;
        if (ids.len == 1 and self.graph_ok and !self.be.profile and self.decode_warm) {
            try self.stepDecodeGraph(io, ids[0], null);
        } else {
            if (ids.len == 1 and self.graph_ok and !self.be.profile) self.decode_warm = true;
            var off: usize = 0;
            while (off < ids.len) {
                const n = @min(self.max_rows, ids.len - off);
                try self.stepChunk(io, ids[off..][0..n], null);
                off += n;
            }
        }
        const be = self.be;
        const b = &self.bufs;
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
    /// `dst` at element offset `dst_off`. f32 copies raw; f16 converts on store.
    fn storeKv(self: *CudaLM, dst: Buf, dst_off: usize, src: Buf, src_off: usize, n: usize) !void {
        if (self.kv_dtype == .f16) {
            try self.be.opStoreKvF16(dst, dst_off, src, src_off, n);
        } else {
            try self.be.tensorCopy(dst, dst_off * 4, src, src_off * 4, n * 4);
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
        // f16 K/V is only read by the flash-split kernels (opAttnDecode); the
        // square `be.attn` path is f32-only, so f16 routes all prefill through
        // opAttnDecode (correct for any seq_q, just less parallel for big batches).
        if (self.kv_dtype == .f16 or seq <= gemv_batch_max) {
            // Batched flash-decoding: the naive square kernel has too little
            // parallelism for a handful of queries.
            try self.be.opAttnDecode(b.q, self.k_cache[l].buf, self.v_cache[l].buf, b.attn, b.attn_scratch, pos0 + 1, seq, c.n_heads, c.n_kv_heads, hd, nsplit, attn_scale, 0, 0, false, self.kv_dtype == .f16);
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

        self.prefetchLayer(0);
        for (self.lm.layers, 0..) |layer, l| {
            self.prefetchLayer(l + 1); // == layers.len prefetches the LM head
            if (self.taps_on) {
                for (self.tap_layers, 0..) |tl, j| {
                    if (l == tl) try be.opCopyOff(self.tap_d, (j * self.capacity + pos0) * c.hidden, b.x, 0, seq * c.hidden);
                }
            }
            try transformer_gpu.decoderLayer(transformer.qwen3_spec, self, layer, l, seq, pos0);
        }
    }

    /// Upload every linear weight and the LM head now, claiming pin residency
    /// (first-touch, up to the backend's pin_budget) ahead of any model that
    /// forwards later. Under --vram-budget the CLI calls this on the DRAFT
    /// model before the target's first forward: the draft is read once per
    /// drafted token, so pinning it buys more transfer savings per byte than
    /// pinning the same bytes of the target (which verify reads only once per
    /// ~k accepted tokens).
    pub fn prewarmWeights(self: *CudaLM) !void {
        const be = self.be;
        for (self.lm.layers) |layer| {
            inline for (.{ layer.q, layer.k, layer.v, layer.o, layer.gate, layer.up, layer.down }) |w| {
                try be.warmWeight(w.bytes);
            }
        }
        try be.warmWeight(self.lm.embed.bytes);
        if (self.lm.head.bytes.ptr != self.lm.embed.bytes.ptr) try be.warmWeight(self.lm.head.bytes);
    }

    /// Queue layer `l`'s linear weights (or, past the last layer, the tied
    /// bf16 LM head) for async prefetch — called one layer ahead so streamed
    /// weights DMA from the page-locked mmap while the previous layer
    /// computes (the dit_cuda.prefetchBlock analogue). No-op when direct
    /// streaming is off; a cheap cache re-stamp when the weights are resident.
    fn prefetchLayer(self: *CudaLM, l: usize) void {
        const be = self.be;
        if (!be.async_uploads) return;
        if (l < self.lm.layers.len) {
            const layer = self.lm.layers[l];
            inline for (.{ layer.q, layer.k, layer.v, layer.o, layer.gate, layer.up, layer.down }) |w| {
                be.prefetchWeight(w.bytes);
            }
        } else {
            be.prefetchWeight(self.lm.embed.bytes);
            if (self.lm.head.bytes.ptr != self.lm.embed.bytes.ptr) be.prefetchWeight(self.lm.head.bytes);
        }
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
            // verify — always <= spec.max_draft+1 = 17 — and short prefills), the
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
    const gemv_batch_max = spec.max_draft + 1;

    /// q5_k/q6_k batches at or below this take the grouped dp4a GEMV instead of
    /// opMatmulQuant's dequant-to-f16 GEMM. Measured crossover (qgemv-bench,
    /// 3090): ~48 rows q5_k, ~35 q6_k; 40 matches qwen35's grouped_prefill_max
    /// and covers every speculative-verify batch (<= spec.max_draft + 1 = 17).
    const grouped_gemv_max = 40;

    /// KV chunks per head in the decode attention split pass (one warp each:
    /// 32 heads x 32 splits x 32 lanes = 32k threads).
    const nsplit = 32;
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
            CudaLM.gemv_batch_max * c.n_heads * CudaLM.nsplit * (hd + 4) * 4, // attn_scratch (a row per verify query)
            (spec.max_draft + 1) * qwen3.vocab_size * 4, // logits (verify writes a row per position)
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
        const m = spec.max_tree_nodes;
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
    const tokenizer_mod = @import("../tokenizer.zig");
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

// Gated on a CUDA device + the checkpoint: greedy decode under a --vram-budget
// far below the checkpoint size (a first-touch prefix pinned resident, the
// rest LRU-evicted and re-uploaded every token; the eviction guard keeps
// decode off the captured graph) must equal resident greedy decode exactly.
test "cuda weight streaming matches resident greedy on the real model" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const engine = @import("../llm/engine.zig");
    const chat = @import("../llm/chat.zig");
    const tokenizer_mod = @import("../tokenizer.zig");
    const te_path = "models/text_encoders/qwen3VLInstruct4bHeretic_v10.safetensors";
    std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    std.Io.Dir.cwd().access(io, te_path, .{}) catch return error.SkipZigTest;
    // st before be: Backend.deinit unregisters the page-locked mmap, so the
    // mapping must still exist when it runs (defers are LIFO).
    var st = try safetensors.SafeTensors.open(gpa, io, te_path);
    defer st.deinit();
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    var lm = try qwen3.CausalLM.load(gpa, .{ .safetensors = &st });
    defer lm.deinit();
    var tok = try tokenizer_mod.Tokenizer.init(gpa);
    defer tok.deinit();

    const opts: engine.Options = .{ .max_new_tokens = 4, .sampling = .{ .temperature = 0 } };
    var ids_resident: std.ArrayList(u32) = .empty;
    defer ids_resident.deinit(gpa);
    try chat.appendUser(&tok, gpa, "Count: one two three one two", &ids_resident);
    try chat.openAssistant(&tok, gpa, &ids_resident);
    var ids_streamed: std.ArrayList(u32) = .empty;
    defer ids_streamed.deinit(gpa);
    try ids_streamed.appendSlice(gpa, ids_resident.items);

    {
        var model = try CudaLM.init(gpa, be, &lm, .fixed(try engine.capacityFor(opts, ids_resident.items.len)), ids_resident.items.len);
        defer model.deinit();
        _ = try engine.generate(&model, &tok, io, gpa, &ids_resident, opts, null);
    }
    be.evictWeights();
    be.budget_override = 1 << 30; // ~1/4 of the fp8 checkpoint: forces streaming
    be.pin_budget = 1 << 29; // first ~512 MiB of weights pinned, rest streams
    if (st.mapping) |m| be.enableDirectStreaming(m); // prefetched direct-DMA path
    {
        var model = try CudaLM.init(gpa, be, &lm, .fixed(try engine.capacityFor(opts, ids_streamed.items.len)), ids_streamed.items.len);
        defer model.deinit();
        try model.prewarmWeights(); // the draft-pinning path: first claim on pin_budget
        _ = try engine.generate(&model, &tok, io, gpa, &ids_streamed, opts, null);
    }
    try std.testing.expect(be.pinned_bytes > 0); // a prefix actually pinned
    try std.testing.expect(be.evictions > 0); // and the remainder streamed
    try std.testing.expectEqualSlices(u32, ids_resident.items, ids_streamed.items);
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
