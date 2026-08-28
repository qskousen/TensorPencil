//! CUDA-backend MiniMax H3 video VAE decode, the device twin of
//! `minimax_h3_vae.decodeVolume`.
//!
//! Once the DiT trunk moved to the device this became the whole render. Measured
//! on a 256x256 / 5-frame clip: **420 s of video decode against 39 s of audio and
//! ~2 s of GPU sampling**, because the temporal chunking pads a 2-frame latent to
//! a SEVEN-token window and then runs 36 transformer blocks over 1792 grid tokens
//! of it.
//!
//! Two of this VAE's conventions cost something the DiT port did not need:
//!
//! 1. **`to_qkv` is fused PER HEAD** (`[h0 q|h0 k|h0 v|h1 q|...]` per token), so
//!    the three planes are not row ranges of the weight and `rowSlice` does not
//!    apply. `opDeinterleave3` exists for exactly this. `ff.w1` IS `[gate; value]`
//!    by rows, so that one does split.
//! 2. **Q/K norms are WEIGHTLESS**, and `qkNorm` always multiplies a weight, so it
//!    is fed a unit vector.
//!
//! The temporal chunking, the window blending and the ImageNet finalize stay on
//! the host: they are per-frame memory traffic rather than arithmetic, and
//! `minimax_h3_vae` already pins them against the reference.

const std = @import("std");
const vae = @import("minimax_h3_vae.zig");
const cuda = @import("tp_gpu").cuda;
const ops = @import("tp_ops");

const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;
const Weight = ops.matmul.Weight;

/// A contiguous row range of a row-major weight, as its own `Weight`. Used for
/// `ff.w1`, whose two halves ARE row ranges; see the header for why `to_qkv` is
/// not.
fn rowSlice(w: Weight, from: usize, n: usize) Weight {
    std.debug.assert(from + n <= w.rows);
    const stride = w.dtype.storageBytes(w.cols);
    var out = w;
    out.bytes = w.bytes[from * stride ..][0 .. n * stride];
    out.rows = n;
    if (w.row_scale) |rs| out.row_scale = rs[from..][0..n];
    return out;
}

pub fn supported(dec: *const vae.VideoDecoder) bool {
    for (dec.blocks) |b| {
        inline for (.{ b.qkv, b.out, b.w1, b.w2 }) |w| {
            switch (w.dtype) {
                .f16, .bf16, .f32 => {},
                else => return false,
            }
        }
    }
    switch (dec.proj_out.dtype) {
        .f16, .bf16, .f32 => {},
        else => return false,
    }
    return true;
}

/// Per-window device state. The chunking decodes every window at the same shape,
/// so this is built once per clip rather than once per window.
pub const Session = struct {
    seq: usize,
    grid: usize,
    pairs: usize,
    freqs_d: Buf = .{},
    /// A unit vector for the WEIGHTLESS q/k norms.
    ones_d: Buf = .{},

    pub fn init(be: *Backend, gpa: std.mem.Allocator, dec: *const vae.VideoDecoder, t: usize, h: usize, w: usize) !Session {
        const cfg = dec.cfg;
        const grid = t * h * w;
        const n_suffix = cfg.n_register + 1;
        var s: Session = .{ .seq = grid + n_suffix, .grid = grid, .pairs = cfg.ropePairs() };
        errdefer s.deinit(be);

        var freqs = try vae.ropeFreqs(gpa, cfg, t, h, w, n_suffix);
        defer freqs.deinit(gpa);
        const host = try gpa.alloc(f32, 2 * s.seq * s.pairs);
        defer gpa.free(host);
        @memcpy(host[0 .. s.seq * s.pairs], freqs.cos);
        @memcpy(host[s.seq * s.pairs ..], freqs.sin);
        s.freqs_d = try be.tensorCreate(host.len * 4);
        try be.tensorUpload(s.freqs_d, std.mem.sliceAsBytes(host));

        const ones = try gpa.alloc(f32, cfg.head_dim);
        defer gpa.free(ones);
        @memset(ones, 1.0);
        s.ones_d = try be.tensorCreate(ones.len * 4);
        try be.tensorUpload(s.ones_d, std.mem.sliceAsBytes(ones));
        return s;
    }

    pub fn sinOff(self: Session) usize {
        return self.seq * self.pairs;
    }

    pub fn deinit(self: *Session, be: *Backend) void {
        be.tensorDestroy(&self.freqs_d);
        be.tensorDestroy(&self.ones_d);
    }
};

pub const Workspace = struct {
    x_d: Buf = .{},
    hn_d: Buf = .{},
    qkv_d: Buf = .{},
    q_d: Buf = .{},
    k_d: Buf = .{},
    v_d: Buf = .{},
    proj_d: Buf = .{},
    gate_d: Buf = .{},
    up_d: Buf = .{},
    scale_d: Buf = .{},
    patch_d: Buf = .{},
    /// Head-PADDED attention operands, allocated only when the backend's P@V
    /// tile is wider than this VAE's 64-dim head. See `attnHd`.
    qp_d: Buf = .{},
    kp_d: Buf = .{},
    vp_d: Buf = .{},
    op_d: Buf = .{},

    pub fn init(be: *Backend, dec: *const vae.VideoDecoder, seq: usize, grid: usize) !Workspace {
        const cfg = dec.cfg;
        var ws: Workspace = .{};
        errdefer ws.deinit(be);
        ws.x_d = try be.tensorCreate(seq * cfg.dim * 4);
        ws.hn_d = try be.tensorCreate(seq * cfg.dim * 4);
        ws.qkv_d = try be.tensorCreate(seq * cfg.dim * 3 * 4);
        ws.q_d = try be.tensorCreate(seq * cfg.dim * 4);
        ws.k_d = try be.tensorCreate(seq * cfg.dim * 4);
        ws.v_d = try be.tensorCreate(seq * cfg.dim * 4);
        ws.proj_d = try be.tensorCreate(seq * cfg.dim * 4);
        ws.gate_d = try be.tensorCreate(seq * cfg.ff * 4);
        ws.up_d = try be.tensorCreate(seq * cfg.ff * 4);
        // Both LayerScale vectors of every block, uploaded once per window.
        ws.scale_d = try be.tensorCreate(dec.blocks.len * 2 * cfg.dim * 4);
        ws.patch_d = try be.tensorCreate(grid * cfg.patchDim() * 4);
        const hp = attnHd(be, cfg.head_dim);
        if (hp != cfg.head_dim) {
            const n = seq * cfg.heads * hp * 4;
            ws.qp_d = try be.tensorCreate(n);
            ws.kp_d = try be.tensorCreate(n);
            ws.vp_d = try be.tensorCreate(n);
            ws.op_d = try be.tensorCreate(n);
        }
        return ws;
    }

    pub fn deinit(self: *Workspace, be: *Backend) void {
        inline for (.{
            &self.x_d,     &self.hn_d,  &self.qkv_d, &self.q_d,
            &self.k_d,     &self.v_d,   &self.proj_d, &self.gate_d,
            &self.up_d,    &self.scale_d, &self.patch_d, &self.qp_d,
            &self.kp_d,    &self.vp_d,  &self.op_d,
        }) |b| be.tensorDestroy(b);
        self.* = undefined;
    }
};

fn gemm(be: *Backend, y: Buf, x: Buf, m: usize, w: Weight, bias: ?[]const f32) !void {
    switch (w.dtype) {
        .f16 => try be.opMatmulF16(y, x, m, w.bytes, w.rows, w.cols, bias, false, false),
        .bf16 => try be.opMatmulBf16(y, x, m, w.bytes, w.rows, w.cols, bias, false, false),
        .f32 => try be.opMatmul(y, 0, x, 0, m, w.bytes, false, w.rows, w.cols, w.scale, bias),
        else => return error.UnsupportedDType,
    }
}

/// Head width the tensor-core attention runs at.
///
/// The hand-PTX P@V GEMM tiles at 128 and launches `grid.x = hd / 128`, so a
/// 64-dim head gives a ZERO-sized grid and a launch error, not a wrong answer.
/// This VAE is 32 heads of 64 where every DiT here is 128, so it is the first
/// diffusion trunk to need the SD family's head padding. `.libs` (cuDNN) attends
/// at the true width and needs none.
fn attnHd(be: *const Backend, hd: usize) usize {
    if (be.kernels == .libs) return hd;
    return std.mem.alignForward(usize, hd, 128);
}

/// Force the f32 online-softmax attention instead of the tensor-core path.
///
/// The tensor-core path stores its scores in **f16** whichever of its two
/// variants runs, and this VAE attends over thousands of keys at a 2048-wide
/// dim, which is exactly the regime CLAUDE.md flags for f16's 65504 ceiling. This
/// is the A/B for that.
pub var force_naive_attn: bool = false;

/// Report the residual stream's magnitude per block (`TP_VAE_MAG`).
pub var dbg_mag: bool = false;

/// Attention, padding the head width when the backend needs it. The padding is
/// exact: the extra columns are zero on both sides, so they contribute nothing to
/// the scores and `opHeadUnpad` drops them again.
fn attention(be: *Backend, ws: *Workspace, seq: usize, heads: usize, hd: usize, scale: f32) !void {
    if (force_naive_attn) {
        return be.attn(ws.q_d, ws.k_d, ws.v_d, ws.hn_d, seq, seq, heads, heads, hd, scale, false);
    }
    const hp = attnHd(be, hd);
    if (hp == hd) return be.opAttnTC(ws.q_d, ws.k_d, ws.v_d, ws.hn_d, seq, heads, heads, hd, scale);
    inline for (.{ .{ "q_d", "qp_d" }, .{ "k_d", "kp_d" }, .{ "v_d", "vp_d" } }) |pair| {
        try be.opHeadPad(@field(ws, pair[1]), @field(ws, pair[0]), seq, heads, hp, hd, heads * hd, 0);
    }
    try be.opAttnTC(ws.qp_d, ws.kp_d, ws.vp_d, ws.op_d, seq, heads, heads, hp, scale);
    try be.opHeadUnpad(ws.op_d, ws.hn_d, seq, heads, hd, hp);
}

fn normBuf(be: *Backend, w: []const f32) !Buf {
    return .{ .buf = try be.smallBuffer(std.mem.sliceAsBytes(w)), .mem = .null_handle, .size = w.len * 4 };
}

/// One whole-window decode, RAW (unclamped, un-finalized) planar
/// `[out_channels][t*patch_t][h*patch][w*patch]`, which is what the chunking
/// blends before finalizing.
pub fn decodeVolume(
    dec: *const vae.VideoDecoder,
    be: *Backend,
    sess: *const Session,
    ws: *Workspace,
    io: std.Io,
    gpa: std.mem.Allocator,
    out: []f32,
    z: []const f32,
    t: usize,
    h: usize,
    w: usize,
) !void {
    const cfg = dec.cfg;
    const dim = cfg.dim;
    const hd = cfg.head_dim;
    const heads = cfg.heads;
    const grid = t * h * w;
    const seq = grid + cfg.n_register + 1;
    const attn_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
    std.debug.assert(sess.seq == seq and sess.grid == grid);

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    // Host: denormalize and transpose planar [c][t][h][w] to token-major [n][c],
    // then the two small projections. 24 channels in, so this is memory traffic
    // rather than arithmetic.
    const c_in = cfg.in_channels;
    const rows = try a.alloc(f32, grid * c_in);
    for (0..grid) |i| {
        for (0..c_in) |c| rows[i * c_in + c] = z[c * grid + i] * dec.latents_std[c] + dec.latents_mean[c];
    }
    const pq = try a.alloc(f32, grid * c_in);
    try ops.matmul.matmul(io, gpa, pq, rows, grid, dec.post_quant, dec.post_quant_bias);
    const emb = try a.alloc(f32, seq * dim);
    try ops.matmul.matmul(io, gpa, emb[0 .. grid * dim], pq, grid, dec.x_embedder, dec.x_embedder_bias);
    // Four register tokens then one ZERO token, dropped after `proj_out`.
    @memcpy(emb[grid * dim ..][0 .. cfg.n_register * dim], dec.register_tokens);
    @memset(emb[(grid + cfg.n_register) * dim ..][0..dim], 0);

    const scales = try a.alloc(f32, dec.blocks.len * 2 * dim);
    for (dec.blocks, 0..) |*b, i| {
        @memcpy(scales[(i * 2) * dim ..][0..dim], b.scale1);
        @memcpy(scales[(i * 2 + 1) * dim ..][0..dim], b.scale2);
    }

    try be.beginBatch();
    errdefer if (be.batching()) be.abortBatch();
    try be.tensorUpload(ws.scale_d, std.mem.sliceAsBytes(scales));
    try be.tensorUpload(ws.x_d, std.mem.sliceAsBytes(emb));

    for (dec.blocks, 0..) |*b, bi| {
        // --- attention half ---
        // `qkNorm` is a weighted RMS over its last dim, so at rows=seq, dim=dim
        // it IS the block's pre-norm.
        try be.qkNorm(ws.x_d, ws.hn_d, try normBuf(be, b.norm1), seq, dim, cfg.eps);
        try gemm(be, ws.qkv_d, ws.hn_d, seq, b.qkv, b.qkv_bias);
        try be.opDeinterleave3(ws.qkv_d, ws.q_d, ws.k_d, ws.v_d, seq * dim, hd);
        // WEIGHTLESS q/k norms: feed `qkNorm` ones.
        try be.qkNorm(ws.q_d, ws.q_d, sess.ones_d, seq * heads, hd, cfg.eps);
        try be.qkNorm(ws.k_d, ws.k_d, sess.ones_d, seq * heads, hd, cfg.eps);
        try be.opRopeHalfPart(ws.q_d, sess.freqs_d, seq, heads, sess.pairs, sess.sinOff(), 0, hd);
        try be.opRopeHalfPart(ws.k_d, sess.freqs_d, seq, heads, sess.pairs, sess.sinOff(), 0, hd);
        try attention(be, ws, seq, heads, hd, attn_scale);
        try gemm(be, ws.proj_d, ws.hn_d, seq, b.out, b.out_bias);
        // LayerScale: `gatedAdd` with the learned scale where a DiT has a gate.
        try be.gatedAdd(ws.x_d, ws.proj_d, ws.scale_d, seq * dim, dim, (bi * 2) * dim);

        // --- feed-forward half ---
        try be.qkNorm(ws.x_d, ws.hn_d, try normBuf(be, b.norm2), seq, dim, cfg.eps);
        // `w1` IS `[gate; value]` by rows, so two GEMMs rather than a strided
        // silu-mul over one fused output.
        try gemm(be, ws.gate_d, ws.hn_d, seq, rowSlice(b.w1, 0, cfg.ff), b.w1_bias[0..cfg.ff]);
        try gemm(be, ws.up_d, ws.hn_d, seq, rowSlice(b.w1, cfg.ff, cfg.ff), b.w1_bias[cfg.ff..]);
        try be.siluMul(ws.gate_d, ws.up_d, seq * cfg.ff);
        try gemm(be, ws.proj_d, ws.gate_d, seq, b.w2, b.w2_bias);
        try be.gatedAdd(ws.x_d, ws.proj_d, ws.scale_d, seq * dim, dim, (bi * 2 + 1) * dim);

        // `TP_VAE_MAG`: the residual stream's magnitude per block. The device
        // GEMMs narrow activations to f16 (`halfActivation`), and f16's 65504
        // ceiling is a limit real VAE checkpoints do reach, so this is the
        // measurement rather than the assumption. Costs a full sync per block.
        if (dbg_mag) {
            try be.endBatch();
            const probe = try a.alloc(f32, seq * dim);
            try be.tensorDownload(ws.x_d, std.mem.sliceAsBytes(probe));
            var mx: f32 = 0;
            var nonfinite: usize = 0;
            for (probe) |v| {
                if (!std.math.isFinite(v)) nonfinite += 1 else mx = @max(mx, @abs(v));
            }
            std.debug.print("vae block {d:>2}: max|x| {d:.1}{s}\n", .{
                bi, mx, if (nonfinite != 0) " NONFINITE" else "",
            });
            try be.beginBatch();
        }
    }

    // Head: a LayerNorm WITH bias, where every other norm here is RMS. Only the
    // GRID tokens are projected; the register/zero suffix is dropped.
    try be.opLayerNorm(ws.x_d, ws.hn_d, dec.norm_out_w, dec.norm_out_b, seq, dim, cfg.eps, false);
    try gemm(be, ws.patch_d, ws.hn_d, grid, dec.proj_out, dec.proj_out_bias);
    try be.endBatch();

    const patches = try a.alloc(f32, grid * cfg.patchDim());
    try be.tensorDownload(ws.patch_d, std.mem.sliceAsBytes(patches));
    vae.unpatchify(cfg, out, patches, t, h, w);
}
