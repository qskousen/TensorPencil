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
const attn_scale: f32 = 1.0 / 11.313708498984761; // 1/sqrt(128)
const eps: f32 = 1e-5;

/// Use the tensor-core GQA attention path (hgemm+softmax_row) instead of the
/// naive one-thread-per-(q,head) kernel. On by default — it is O(seq²) faster on
/// the tensor cores and the naive path is O(seq²) latency-bound. Toggle for A/B.
pub var use_tc_attn: bool = true;

/// Per-run constants: text-fusion tokens + rope table, uploaded once.
pub const Session = struct {
    seq_txt: usize,
    lat_h: usize,
    lat_w: usize,
    txt0_d: DeviceBuffer,
    txt_len: usize, // element count (seq_txt * F)
    freqs_d: DeviceBuffer,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, be: *Backend, model: *const DiT, lat_h: usize, lat_w: usize, cond: []const f32, seq_txt: usize) !Session {
        const h = lat_h / patch;
        const w = lat_w / patch;
        const seq = seq_txt + h * w;

        const txt_tokens = try model.textTokens(io, gpa, cond, seq_txt);
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
        ws.mg_d = try be.tensorCreate(mpad * mlp_dim * 4);
        ws.mu_d = try be.tensorCreate(mpad * mlp_dim * 4);
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
pub fn forward(model: *const DiT, be: *Backend, sess: *const Session, ws: *const Workspace, io: std.Io, gpa: std.mem.Allocator, out: []f32, x_lat: []const f32, sigma: f32) !void {
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

    // patch embed: x[seq_txt..] = img_in @ first^T + bias
    const first_f8 = model.first.w.dtype == .f8_e4m3;
    try be.opMatmul(x_d, seq_txt * F * 4, imgin_d, 0, n_img, model.first.w.bytes, first_f8, F, channels * patch * patch, model.first.w.scale, model.first.b);

    // Prefetch block 0's weights before the loop; each iteration prefetches the
    // NEXT block so its upload overlaps this block's compute (async streaming).
    if (be.async_uploads) prefetchBlock(be, model.blocks[0]);

    for (model.blocks, 0..) |blk, b| {
        if (be.async_uploads and b + 1 < model.blocks.len) prefetchBlock(be, model.blocks[b + 1]);
        const mb = b * 6 * F;
        // --- attention ---
        try be.rmsMod(x_d, t1_d, mv_d, seq, F, mb + 0 * F, mb + 1 * F, eps);
        try be.opI8Prep(t1_d, seq, F);
        try be.opI8Gemm(q_d, blk.attn.wq.bytes, blk.attn.wq.row_scale.?, blk.attn.wq.rows, false);
        try be.opI8Gemm(k_d, blk.attn.wk.bytes, blk.attn.wk.row_scale.?, blk.attn.wk.rows, false);
        try be.opI8Gemm(v_d, blk.attn.wv.bytes, blk.attn.wv.row_scale.?, blk.attn.wv.rows, false);
        try be.opI8Gemm(g_d, blk.attn.gate.bytes, blk.attn.gate.row_scale.?, blk.attn.gate.rows, false);
        const qn = try normBuf(be, blk.attn.qnorm);
        const kn = try normBuf(be, blk.attn.knorm);
        try be.qkNorm(q_d, q_d, qn, seq * heads, hd, eps);
        try be.qkNorm(k_d, k_d, kn, seq * kv_heads, hd, eps);
        try be.rope(q_d, sess.freqs_d, seq, heads, half, sin_off);
        try be.rope(k_d, sess.freqs_d, seq, kv_heads, half, sin_off);
        if (use_tc_attn)
            try be.opAttnTC(q_d, k_d, v_d, attn_d, seq, heads, kv_heads, hd, attn_scale)
        else
            try be.attn(q_d, k_d, v_d, attn_d, seq, heads, kv_heads, hd, attn_scale);
        try be.sigmoidMul(attn_d, g_d, seq * F);
        try be.opI8Prep(attn_d, seq, blk.attn.wo.cols);
        try be.opI8Gemm(t1_d, blk.attn.wo.bytes, blk.attn.wo.row_scale.?, blk.attn.wo.rows, false);
        try be.gatedAdd(x_d, t1_d, mv_d, seq * F, F, mb + 2 * F);
        // --- mlp ---
        try be.rmsMod(x_d, t1_d, mv_d, seq, F, mb + 3 * F, mb + 4 * F, eps);
        try be.opI8Prep(t1_d, seq, F);
        try be.opI8Gemm(mg_d, blk.mlp.gate.bytes, blk.mlp.gate.row_scale.?, blk.mlp.gate.rows, false);
        try be.opI8Gemm(mu_d, blk.mlp.up.bytes, blk.mlp.up.row_scale.?, blk.mlp.up.rows, false);
        try be.siluMul(mg_d, mu_d, seq * mlp_dim);
        try be.opI8Prep(mg_d, seq, blk.mlp.down.cols);
        try be.opI8Gemm(t1_d, blk.mlp.down.bytes, blk.mlp.down.row_scale.?, blk.mlp.down.rows, false);
        try be.gatedAdd(x_d, t1_d, mv_d, seq * F, F, mb + 5 * F);
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
