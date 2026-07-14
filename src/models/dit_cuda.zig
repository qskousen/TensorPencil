//! CUDA-backend Krea 2 DiT forward (int8 convrot checkpoint), producing the
//! same latent as the CPU/Vulkan paths so the hand-PTX backend can generate
//! like-for-like images. Follows the fallback op sequence (int8 tensor-core
//! GEMMs via `opI8Prep`/`opI8Gemm`, f32 eltwise norm/rope/gate, one naive
//! online-softmax GQA attention kernel). Small/CPU-cheap paths (text fusion,
//! timestep MLPs, patchify, unpatchify) stay on the CPU.
//!
//! Numerics match `DiT.forward` up to floating-point reordering (int8 quant +
//! ex2.approx softmax) — the same regime the Vulkan int8 path runs in.

const std = @import("std");
const dit = @import("dit.zig");
const cuda = @import("../gpu/cuda.zig");
const safetensors = @import("../safetensors.zig");

const DiT = dit.DiT;
const Backend = cuda.Backend;
const DeviceBuffer = cuda.backend.DeviceBuffer;

const F = dit.features; // 6144
const heads = dit.n_heads; // 48
const kv_heads = dit.n_kv_heads; // 12
const hd = dit.head_dim; // 128
const half = hd / 2; // 64
const mlp_dim = dit.mlp_dim; // 16384
const n_blocks = dit.n_blocks; // 28
const patch = dit.patch; // 2
const channels = dit.channels; // 16
const txt_dim = dit.txt_dim; // 2560
const txt_heads = dit.txt_heads; // 20
const txt_layers = dit.txt_layers; // 12
const txt_mlp_dim = dit.txt_mlp_dim; // 6912
const attn_scale: f32 = 1.0 / 11.313708498984761; // 1/sqrt(128)
const eps: f32 = 1e-5;

/// MLP sequence-tile: the gate/up/down GEMMs run over chunks of this many rows
/// so the mg/mu intermediates are [tile][mlp_dim] instead of [seq][mlp_dim]
/// (512 MiB → 128 MiB each at 1 MP). The MLP is per-row so chunks are independent.
const mlp_tile: usize = 2048;

/// A device-buffer sub-view offset `off_bytes` into `b` (CUDA buffers are raw
/// pointers, so a mid-buffer view is just pointer arithmetic — the eltwise/GEMM
/// kernels index from the given base).
fn offsetBuf(b: DeviceBuffer, off_bytes: usize) DeviceBuffer {
    return .{ .buf = @enumFromInt(@intFromEnum(b.buf) + off_bytes), .mem = .null_handle, .size = b.size - off_bytes };
}

/// Quantized-linear dispatch. A convrot checkpoint is homogeneous (every DiT
/// linear is the same width), so `is_i4` picks the W4A4 prep/GEMM (m16n8k64.s4,
/// activations quantized to s4) vs the W8A8 path at each call site.
fn qPrep(be: *Backend, is_i4: bool, x: DeviceBuffer, m: usize, cols: usize) !void {
    if (is_i4) return be.opI4Prep(x, m, cols);
    return be.opI8Prep(x, m, cols, false);
}
fn qGemm(be: *Backend, is_i4: bool, y: DeviceBuffer, w: anytype) !void {
    if (is_i4) return be.opI4Gemm(y, w.bytes, w.row_scale.?, w.rows);
    return be.opI8Gemm(y, w.bytes, w.row_scale.?, w.rows, false);
}

/// Use the tensor-core GQA attention path (hgemm+softmax_row) instead of the
/// naive one-thread-per-(q,head) kernel. On by default — it is O(seq²) faster on
/// the tensor cores and the naive path is O(seq²) latency-bound. Toggle for A/B.
pub var use_tc_attn: bool = true;

// ==== text fusion (CUDA) =====================================================
//
// The int8/int4 convrot checkpoints leave the text-fusion stack unquantized
// (BF16 attn/mlp, F32 projector/txtmlp). Run on the CPU it is ~2 TFLOP of dense
// GEMM — the entire "loading diffusion model" stall on --backend zig-cuda. This
// mirrors DiT.txtFusion+textTokens on the backend: BF16 weights are dequantized
// to f32 once and fed through the f32 `opMatmul` (no int GEMM applies here); the
// block is the plain (no modulation, no RoPE) variant of the sampling block.

/// f32 weight bytes for `opMatmul`: BF16 weights dequant into `arena` (kept alive
/// for the whole fusion so the backend's pointer-keyed weight cache stays valid);
/// F32 weights (projector, txtmlp) pass through their mmap bytes unchanged.
fn txtF32(arena: std.mem.Allocator, w: anytype) ![]const u8 {
    if (w.dtype == .f32) return w.bytes;
    const out = try arena.alloc(f32, w.rows * w.cols);
    try safetensors.convertToF32(w.dtype, w.bytes, out);
    return std.mem.sliceAsBytes(out);
}

/// Device scratch for the text-fusion blocks, sized for the widest phase (the
/// layerwise blocks over seq_txt·12 rows). Reused by the refiner and txtmlp.
const TxtScratch = struct {
    normed: DeviceBuffer,
    q: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    g: DeviceBuffer,
    attn: DeviceBuffer,
    t1: DeviceBuffer,
    gate: DeviceBuffer, // also holds txtmlp mid (seq_txt·6144 ≤ rows_lw·6912)
    up: DeviceBuffer, // also holds txtmlp out

    fn init(be: *Backend, rows_lw: usize) !TxtScratch {
        var s: TxtScratch = undefined;
        s.normed = try be.tensorCreate(rows_lw * txt_dim * 4);
        s.q = try be.tensorCreate(rows_lw * txt_dim * 4);
        s.k = try be.tensorCreate(rows_lw * txt_dim * 4);
        s.v = try be.tensorCreate(rows_lw * txt_dim * 4);
        s.g = try be.tensorCreate(rows_lw * txt_dim * 4);
        s.attn = try be.tensorCreate(rows_lw * txt_dim * 4);
        s.t1 = try be.tensorCreate(rows_lw * txt_dim * 4);
        s.gate = try be.tensorCreate(rows_lw * txt_mlp_dim * 4);
        s.up = try be.tensorCreate(rows_lw * txt_mlp_dim * 4);
        return s;
    }
    fn deinit(s: *TxtScratch, be: *Backend) void {
        inline for (@typeInfo(TxtScratch).@"struct".fields) |f| be.tensorDestroy(&@field(s, f.name));
    }
};

/// One f16 tensor-core GEMM y[m][out] = x[m][in] @ Wᵀ (+bias), W dequant to f32.
/// The naive f32 opMatmul is ~50× too slow for these widths; opConvF16 runs the
/// validated hgemm on tensor cores. f16 is far finer than the int8/int4 the DiT
/// blocks downstream run in, so text-fusion precision is not the bottleneck.
/// `bias` is a real bias, or a zeros slice (≥ out long) for the no-bias linears.
fn txtGemm(be: *Backend, arena: std.mem.Allocator, y: DeviceBuffer, x: DeviceBuffer, m: usize, w: anytype, out: usize, in: usize, bias: []const f32) !void {
    try be.opConvF16(y, 0, x, m, try txtF32(arena, w), out, in, bias);
}

/// One TextFusionBlock on the backend: x += attn(rmsnorm(x)); x += mlp(rmsnorm(x)).
/// `x_d` holds n_seqs sequences of seq_len rows of txt_dim (no RoPE, no mask, no
/// modulation). qkNorm with hd=txt_dim is a plain per-row RMS-norm×weight.
fn txtBlockCuda(be: *Backend, arena: std.mem.Allocator, blk: anytype, x_d: DeviceBuffer, s: *const TxtScratch, zero: []const f32, n_seqs: usize, seq_len: usize) !void {
    const rows = n_seqs * seq_len;

    // --- attention ---
    try be.qkNorm(x_d, s.normed, try normBuf(be, blk.prenorm), rows, txt_dim, eps);
    try txtGemm(be, arena, s.q, s.normed, rows, blk.attn.wq, txt_dim, txt_dim, zero);
    try txtGemm(be, arena, s.k, s.normed, rows, blk.attn.wk, txt_dim, txt_dim, zero);
    try txtGemm(be, arena, s.v, s.normed, rows, blk.attn.wv, txt_dim, txt_dim, zero);
    try txtGemm(be, arena, s.g, s.normed, rows, blk.attn.gate, txt_dim, txt_dim, zero);
    try be.qkNorm(s.q, s.q, try normBuf(be, blk.attn.qnorm), rows * txt_heads, hd, eps);
    try be.qkNorm(s.k, s.k, try normBuf(be, blk.attn.knorm), rows * txt_heads, hd, eps);
    // Each of the n_seqs sequences attends only within itself (12-long layerwise,
    // or the whole prompt for the refiner) — the naive kernel handles one at a
    // time; seq_len is small so the launch count is cheap and one-time.
    for (0..n_seqs) |i| {
        const off = i * seq_len * txt_dim * 4;
        try be.attn(offsetBuf(s.q, off), offsetBuf(s.k, off), offsetBuf(s.v, off), offsetBuf(s.attn, off), seq_len, seq_len, txt_heads, txt_heads, hd, attn_scale, false);
    }
    try be.sigmoidMul(s.attn, s.g, rows * txt_dim);
    try txtGemm(be, arena, s.t1, s.attn, rows, blk.attn.wo, txt_dim, txt_dim, zero);
    try be.opAdd(x_d, s.t1, rows * txt_dim);

    // --- mlp (swiglu) ---
    try be.qkNorm(x_d, s.normed, try normBuf(be, blk.postnorm), rows, txt_dim, eps);
    try txtGemm(be, arena, s.gate, s.normed, rows, blk.mlp.gate, txt_mlp_dim, txt_dim, zero);
    try txtGemm(be, arena, s.up, s.normed, rows, blk.mlp.up, txt_mlp_dim, txt_dim, zero);
    try be.siluMul(s.gate, s.up, rows * txt_mlp_dim);
    try txtGemm(be, arena, s.t1, s.gate, rows, blk.mlp.down, txt_dim, txt_mlp_dim, zero);
    try be.opAdd(x_d, s.t1, rows * txt_dim);
}

/// CUDA port of `DiT.textTokens`: text conditioning [seq_txt·12·txt_dim] → the
/// combined-sequence tokens [seq_txt·features] the sampler consumes. Runs the
/// whole txtfusion + txtmlp stack on the backend so it no longer stalls the CPU.
/// Transient f32 weights are dropped from the cache before returning.
pub fn textTokensCuda(model: *const DiT, be: *Backend, gpa: std.mem.Allocator, cond: []const f32) ![]f32 {
    const seq_txt = cond.len / (txt_layers * txt_dim);
    std.debug.assert(cond.len == seq_txt * txt_layers * txt_dim);
    const rows_lw = seq_txt * txt_layers;

    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var x_d = try be.tensorCreate(rows_lw * txt_dim * 4);
    defer be.tensorDestroy(&x_d);
    var s = try TxtScratch.init(be, rows_lw);
    defer s.deinit(be);
    // Reclaim the transient f32 text weights (and small norm buffers) from the
    // backend cache; they are unused during sampling and would otherwise pin VRAM.
    // SCOPED (not evictWeights, which nukes ALL cached weights): this runs per
    // image inside a persistent pipeline.Session, and a full evict would drop the
    // resident DiT that the next queued image reuses (GUI_VRAM.md Phase 4). The
    // scope drops exactly the weights cached here — the DiT (cached later, in the
    // sampling loop, and pinned) survives.
    be.weightScopeBegin();
    defer be.weightScopeEnd();

    try be.tensorUpload(x_d, std.mem.sliceAsBytes(cond));

    // Zero bias for the no-bias linears (opConvF16 always adds a bias; it reads
    // only the first `out` entries, so one max-width zeros buffer serves all).
    const zero = try arena.alloc(f32, txt_mlp_dim);
    @memset(zero, 0);

    // Layerwise blocks: seq_txt independent 12-long sequences (across the encoder
    // layer axis per token).
    try be.beginBatch();
    errdefer if (be.batching()) be.abortBatch();
    for (&model.txt_layerwise) |*blk| try txtBlockCuda(be, arena, blk, x_d, &s, zero, seq_txt, txt_layers);
    try be.endBatch();

    // Projector: collapse the 12-layer axis (projected[tok][d] = Σ_l pw[l]·x[tok·12+l][d]).
    // Tiny [1,12] contraction — a host round-trip is simpler than a bespoke kernel.
    {
        const work = try arena.alloc(f32, rows_lw * txt_dim);
        try be.tensorDownload(x_d, std.mem.sliceAsBytes(work));
        var pw: [txt_layers]f32 = undefined;
        try safetensors.convertToF32(model.txt_projector.dtype, model.txt_projector.bytes, &pw);
        const projected = try arena.alloc(f32, seq_txt * txt_dim);
        for (0..seq_txt) |tok| {
            const dst = projected[tok * txt_dim ..][0..txt_dim];
            @memset(dst, 0);
            for (0..txt_layers) |l| {
                const src = work[(tok * txt_layers + l) * txt_dim ..][0..txt_dim];
                for (dst, src) |*d, sv| d.* += pw[l] * sv;
            }
        }
        try be.tensorUpload(x_d, std.mem.sliceAsBytes(projected));
    }

    try be.beginBatch();
    // Refiner blocks: one sequence of length seq_txt.
    for (&model.txt_refiner) |*blk| try txtBlockCuda(be, arena, blk, x_d, &s, zero, 1, seq_txt);

    // txtmlp: rmsnorm → Linear(2560→6144) → geluTanh → Linear(6144→6144).
    try be.qkNorm(x_d, s.normed, try normBuf(be, model.txtmlp_norm), seq_txt, txt_dim, eps);
    try txtGemm(be, arena, s.gate, s.normed, seq_txt, model.txtmlp1.w, F, txt_dim, model.txtmlp1.b.?);
    try be.gelu(s.gate, seq_txt * F);
    try txtGemm(be, arena, s.up, s.gate, seq_txt, model.txtmlp3.w, F, F, model.txtmlp3.b.?);
    try be.endBatch();

    const out = try gpa.alloc(f32, seq_txt * F);
    errdefer gpa.free(out);
    try be.tensorDownload(s.up, std.mem.sliceAsBytes(out));
    return out;
}

/// Per-run constants: text-fusion tokens + rope table, uploaded once.
pub const Session = struct {
    seq_txt: usize,
    lat_h: usize,
    lat_w: usize,
    txt0_d: DeviceBuffer,
    txt_len: usize, // element count (seq_txt * F)
    freqs_d: DeviceBuffer,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, be: *Backend, model: *const DiT, lat_h: usize, lat_w: usize, cond: []const f32, seq_txt: usize) !Session {
        _ = io; // text fusion runs on the backend (textTokensCuda), not the CPU
        const h = lat_h / patch;
        const w = lat_w / patch;
        const seq = seq_txt + h * w;

        const txt_tokens = try textTokensCuda(model, be, gpa, cond);
        defer gpa.free(txt_tokens);
        var txt0_d = try be.tensorCreate(txt_tokens.len * 4);
        errdefer be.tensorDestroy(&txt0_d);
        try be.tensorUpload(txt0_d, std.mem.sliceAsBytes(txt_tokens));

        var freqs = try DiT.ropeFreqs(gpa, seq_txt, h, w);
        defer freqs.deinit(gpa);
        std.debug.assert(freqs.half == half);
        const fp = try gpa.alloc(f32, 2 * seq * half);
        defer gpa.free(fp);
        @memcpy(fp[0 .. seq * half], freqs.cos);
        @memcpy(fp[seq * half ..], freqs.sin);
        var freqs_d = try be.tensorCreate(fp.len * 4);
        errdefer be.tensorDestroy(&freqs_d);
        try be.tensorUpload(freqs_d, std.mem.sliceAsBytes(fp));

        return .{
            .seq_txt = seq_txt,
            .lat_h = lat_h,
            .lat_w = lat_w,
            .txt0_d = txt0_d,
            .txt_len = txt_tokens.len,
            .freqs_d = freqs_d,
        };
    }

    pub fn deinit(self: *Session, be: *Backend) void {
        be.tensorDestroy(&self.txt0_d);
        be.tensorDestroy(&self.freqs_d);
    }
};

fn normBuf(be: *Backend, w: []const f32) !DeviceBuffer {
    return .{ .buf = try be.smallBuffer(std.mem.sliceAsBytes(w)), .mem = .null_handle, .size = w.len * 4 };
}

/// Queue all of a DiT block's streamable weights for async prefetch (called one
/// block ahead so the uploads overlap the previous block's compute). Keys must
/// match the byte slices `forward` later fetches (same host pointers → cache hit).
fn prefetchBlock(be: *Backend, blk: anytype) void {
    const bytes = std.mem.sliceAsBytes;
    inline for (.{ blk.attn.wq, blk.attn.wk, blk.attn.wv, blk.attn.gate, blk.attn.wo }) |w| {
        be.prefetchWeight(w.bytes);
        if (w.row_scale) |rs| be.prefetchWeight(bytes(rs));
    }
    be.prefetchWeight(bytes(blk.attn.qnorm));
    be.prefetchWeight(bytes(blk.attn.knorm));
    inline for (.{ blk.mlp.gate, blk.mlp.up, blk.mlp.down }) |w| {
        be.prefetchWeight(w.bytes);
        if (w.row_scale) |rs| be.prefetchWeight(bytes(rs));
    }
}

/// Per-run device scratch shared across sampler steps (and both CFG sessions):
/// the ~12 activation buffers `forward` used to `cuMemAlloc`/free every call. Size
/// once for the largest sequence (both prompts share n_img; only seq_txt differs).
pub const Workspace = struct {
    x_d: DeviceBuffer = .{},
    imgin_d: DeviceBuffer = .{},
    mv_d: DeviceBuffer = .{},
    fin_d: DeviceBuffer = .{},
    t1_d: DeviceBuffer = .{},
    q_d: DeviceBuffer = .{},
    k_d: DeviceBuffer = .{},
    v_d: DeviceBuffer = .{},
    g_d: DeviceBuffer = .{},
    attn_d: DeviceBuffer = .{},
    mg_d: DeviceBuffer = .{},
    mu_d: DeviceBuffer = .{},

    pub fn init(be: *Backend, lat_h: usize, lat_w: usize, seq_txt_cap: usize) !Workspace {
        const n_img = (lat_h / patch) * (lat_w / patch);
        const mpad = std.mem.alignForward(usize, seq_txt_cap + n_img, 128);
        var ws: Workspace = .{};
        errdefer ws.deinit(be);
        ws.x_d = try be.tensorCreate(mpad * F * 4);
        ws.imgin_d = try be.tensorCreate(n_img * channels * patch * patch * 4);
        ws.mv_d = try be.tensorCreate(n_blocks * 6 * F * 4);
        ws.fin_d = try be.tensorCreate(2 * F * 4);
        ws.t1_d = try be.tensorCreate(mpad * F * 4);
        ws.q_d = try be.tensorCreate(mpad * heads * hd * 4);
        ws.k_d = try be.tensorCreate(mpad * kv_heads * hd * 4);
        ws.v_d = try be.tensorCreate(mpad * kv_heads * hd * 4);
        ws.g_d = try be.tensorCreate(mpad * F * 4);
        ws.attn_d = try be.tensorCreate(mpad * heads * hd * 4);
        // mg/mu hold one MLP tile (see mlp_tile) — not the full padded sequence.
        const mlp_rows = @min(mpad, std.mem.alignForward(usize, mlp_tile, 128));
        ws.mg_d = try be.tensorCreate(mlp_rows * mlp_dim * 4);
        ws.mu_d = try be.tensorCreate(mlp_rows * mlp_dim * 4);
        return ws;
    }

    pub fn deinit(self: *Workspace, be: *Backend) void {
        inline for (@typeInfo(Workspace).@"struct".fields) |f| {
            be.tensorDestroy(&@field(self, f.name));
        }
    }
};

/// One DiT forward evaluation: velocity `out` from latent `x_lat` at `sigma`.
/// `ws` must be sized for a sequence ≥ seq_txt + n_img (see Workspace.init).
pub fn forward(model: *const DiT, be: *Backend, sess: *const Session, ws: *const Workspace, io: std.Io, gpa: std.mem.Allocator, out: []f32, x_lat: []const f32, sigma: f32, cancel: ?*std.atomic.Value(bool)) !void {
    const lat_h = sess.lat_h;
    const lat_w = sess.lat_w;
    const seq_txt = sess.seq_txt;
    const n_img = (lat_h / patch) * (lat_w / patch);
    const seq = seq_txt + n_img;
    const sin_off = seq * half;

    // ---- CPU: modulation vectors, final-layer vector, patch embed ----
    const tvv = try model.timestepVectors(io, gpa, sigma);
    defer gpa.free(tvv.t);
    defer gpa.free(tvv.tvec);

    const mv = try gpa.alloc(f32, n_blocks * 6 * F);
    defer gpa.free(mv);
    for (model.blocks, 0..) |blk, b| {
        const base = b * 6 * F;
        for (0..6 * F) |i| mv[base + i] = tvv.tvec[i] + blk.mod[i];
        for (0..F) |c| {
            mv[base + c] = (1.0 + mv[base + c]) * blk.prenorm[c];
            mv[base + 3 * F + c] = (1.0 + mv[base + 3 * F + c]) * blk.postnorm[c];
        }
    }
    const fin = try gpa.alloc(f32, 2 * F);
    defer gpa.free(fin);
    for (0..F) |c| {
        fin[c] = (1.0 + tvv.t[c] + model.last_mod[c]) * model.last_norm[c];
        fin[F + c] = tvv.t[c] + model.last_mod[F + c];
    }
    const img_in = try DiT.patchify(gpa, x_lat, lat_h, lat_w);
    defer gpa.free(img_in);

    // ---- device buffers (from the per-run Workspace) ----
    const x_d = ws.x_d;
    const imgin_d = ws.imgin_d;
    const mv_d = ws.mv_d;
    const fin_d = ws.fin_d;
    const t1_d = ws.t1_d;
    const q_d = ws.q_d;
    const k_d = ws.k_d;
    const v_d = ws.v_d;
    const g_d = ws.g_d;
    const attn_d = ws.attn_d;
    const mg_d = ws.mg_d;
    const mu_d = ws.mu_d;

    try be.tensorUpload(mv_d, std.mem.sliceAsBytes(mv));
    try be.tensorUpload(fin_d, std.mem.sliceAsBytes(fin));
    try be.tensorCopy(x_d, 0, sess.txt0_d, 0, sess.txt_len * 4);
    try be.tensorUpload(imgin_d, std.mem.sliceAsBytes(img_in));

    try be.beginBatch();
    errdefer if (be.batching()) be.abortBatch();

    // int4 (W4A4) vs int8 (W8A8) convrot: gate the quantized-linear path once.
    // The hand-PTX CUDA DiT only handles convrot checkpoints (per-row scale +
    // packed int weights); an fp8/bf16 DiT has no row_scale, so reject it with a
    // clear error instead of unwrapping a null scale into an illegal GPU access.
    const wqt = model.blocks[0].attn.wq.dtype;
    if (wqt != .i8 and wqt != .i4) return error.UnsupportedCheckpoint;
    const is_i4 = wqt == .i4;
    // f16 activation chain (c16): only on the cuBLASLt/irescale int8 libs path.
    // Halves the mlp gate/up/silu/down-input DRAM traffic (the biggest eltwise
    // category). Hand-PTX (igemm_pipe_fused writes f32) and int4 keep f32.
    const mlp_f16 = (be.kernels == .libs) and !is_i4;

    // patch embed: x[seq_txt..] = img_in @ first^T + bias
    const first_f8 = model.first.w.dtype == .f8_e4m3;
    try be.opMatmul(x_d, seq_txt * F * 4, imgin_d, 0, n_img, model.first.w.bytes, first_f8, F, channels * patch * patch, model.first.w.scale, model.first.b);

    // Prefetch block 0's weights before the loop; each iteration prefetches the
    // NEXT block so its upload overlaps this block's compute (async streaming).
    if (be.async_uploads) prefetchBlock(be, model.blocks[0]);

    for (model.blocks, 0..) |blk, b| {
        // Poll cancel between blocks so a stop lands mid-step (≈1/28 of a step)
        // rather than only at step boundaries — matters most under weight
        // streaming, where each block waits on its uploaded weights. The
        // `errdefer` above aborts the in-flight CUDA batch on the way out.
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        if (be.async_uploads and b + 1 < model.blocks.len) prefetchBlock(be, model.blocks[b + 1]);
        const mb = b * 6 * F;
        // --- attention ---
        try be.rmsMod(x_d, t1_d, mv_d, seq, F, mb + 0 * F, mb + 1 * F, eps);
        try qPrep(be, is_i4, t1_d, seq, F);
        try qGemm(be, is_i4, q_d, blk.attn.wq);
        try qGemm(be, is_i4, k_d, blk.attn.wk);
        try qGemm(be, is_i4, v_d, blk.attn.wv);
        try qGemm(be, is_i4, g_d, blk.attn.gate);
        const qn = try normBuf(be, blk.attn.qnorm);
        const kn = try normBuf(be, blk.attn.knorm);
        try be.qkNorm(q_d, q_d, qn, seq * heads, hd, eps);
        try be.qkNorm(k_d, k_d, kn, seq * kv_heads, hd, eps);
        try be.rope(q_d, sess.freqs_d, seq, heads, half, sin_off);
        try be.rope(k_d, sess.freqs_d, seq, kv_heads, half, sin_off);
        if (use_tc_attn)
            try be.opAttnTC(q_d, k_d, v_d, attn_d, seq, heads, kv_heads, hd, attn_scale)
        else
            try be.attn(q_d, k_d, v_d, attn_d, seq, seq, heads, kv_heads, hd, attn_scale, false);
        try be.sigmoidMul(attn_d, g_d, seq * F);
        try qPrep(be, is_i4, attn_d, seq, blk.attn.wo.cols);
        try qGemm(be, is_i4, t1_d, blk.attn.wo);
        try be.gatedAdd(x_d, t1_d, mv_d, seq * F, F, mb + 2 * F);
        // --- mlp (sequence-tiled: mg/mu are [tile][mlp_dim]; each row-chunk is
        // independent, so we walk seq in mlp_tile-row bands over offset x views) ---
        var c0: usize = 0;
        while (c0 < seq) : (c0 += mlp_tile) {
            const tile: usize = @min(mlp_tile, seq - c0);
            const xo = offsetBuf(x_d, c0 * F * 4);
            try be.rmsMod(xo, t1_d, mv_d, tile, F, mb + 3 * F, mb + 4 * F, eps);
            try qPrep(be, is_i4, t1_d, tile, F);
            if (mlp_f16) {
                // gate/up GEMMs emit f16 (irescale_h16); silu_mul_h16 reads/writes
                // f16; the down prep reads f16 — halving the 16384-dim traffic.
                try be.opI8Gemm(mg_d, blk.mlp.gate.bytes, blk.mlp.gate.row_scale.?, blk.mlp.gate.rows, true);
                try be.opI8Gemm(mu_d, blk.mlp.up.bytes, blk.mlp.up.row_scale.?, blk.mlp.up.rows, true);
                try be.siluMul16(mg_d, mu_d, tile * mlp_dim);
                try be.opI8Prep(mg_d, tile, blk.mlp.down.cols, true);
            } else {
                try qGemm(be, is_i4, mg_d, blk.mlp.gate);
                try qGemm(be, is_i4, mu_d, blk.mlp.up);
                try be.siluMul(mg_d, mu_d, tile * mlp_dim);
                try qPrep(be, is_i4, mg_d, tile, blk.mlp.down.cols);
            }
            try qGemm(be, is_i4, t1_d, blk.mlp.down); // down → f32 t1_d for gatedAdd
            try be.gatedAdd(xo, t1_d, mv_d, tile * F, F, mb + 5 * F);
        }
    }

    // --- final layer ---
    try be.rmsMod(x_d, t1_d, fin_d, seq, F, 0, F, eps);
    const last_f8 = model.last_linear.w.dtype == .f8_e4m3;
    try be.opMatmul(imgin_d, 0, t1_d, seq_txt * F * 4, n_img, model.last_linear.w.bytes, last_f8, channels * patch * patch, F, model.last_linear.w.scale, model.last_linear.b);
    try be.endBatch();

    const final_rows = try gpa.alloc(f32, n_img * channels * patch * patch);
    defer gpa.free(final_rows);
    try be.tensorDownload(imgin_d, std.mem.sliceAsBytes(final_rows));
    DiT.unpatchify(out, final_rows, lat_h, lat_w);
}
