//! Vulkan MiniMax H3 video VAE decode, the twin of `minimax_h3_vae_cuda` and the
//! device form of `minimax_h3_vae.decodeVolume`.
//!
//! That arm's header has the two conventions that cost something (`to_qkv` fused
//! PER HEAD, so the planes are not row ranges; weightless q/k norms). The split is
//! the same: the 36 transformer blocks run here, the temporal chunking, window
//! blending and ImageNet finalize stay on the host.
//!
//! One thing differs, and it is the backend's: the tensor-core attention needs a
//! head width its P@V tile divides, and this VAE is 32 heads of 64. The CUDA arm
//! pads to 128 for its hand-PTX tile; here the coop path has the same floor, so
//! `head_pad` / `head_unpad` run for the same reason. The padding is exact, zero
//! on both sides of the scores.
//!
//! `minimax-h3-vae-vk-test` checks it against the CPU decoder on real weights.

const std = @import("std");
const vae = @import("minimax_h3_vae.zig");
const gpu = @import("tp_gpu").context;
const ops = @import("tp_ops");

const Context = gpu.Context;
const Buf = gpu.DeviceBuffer;
const Weight = ops.matmul.Weight;

const nullBuf: Buf = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 };

/// Force the f32 reference attention instead of the tensor-core pipeline. Same
/// A/B the CUDA arm carries: this VAE attends over thousands of keys at a
/// 2048-wide dim, which is where f16 scores start to matter.
pub var force_naive_attn: bool = false;

/// A contiguous row range of a row-major weight, as its own `Weight`. For `ff.w1`,
/// whose two halves ARE row ranges.
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

pub const Session = struct {
    seq: usize,
    grid: usize,
    pairs: usize,
    freqs_d: Buf = nullBuf,
    /// A unit vector for the WEIGHTLESS q/k norms.
    ones_d: Buf = nullBuf,

    pub fn init(ctx: *Context, gpa: std.mem.Allocator, dec: *const vae.VideoDecoder, t: usize, h: usize, w: usize) !Session {
        const cfg = dec.cfg;
        const grid = t * h * w;
        const n_suffix = cfg.n_register + 1;
        var s: Session = .{ .seq = grid + n_suffix, .grid = grid, .pairs = cfg.ropePairs() };
        errdefer s.deinit(ctx);

        var freqs = try vae.ropeFreqs(gpa, cfg, t, h, w, n_suffix);
        defer freqs.deinit(gpa);
        const host = try gpa.alloc(f32, 2 * s.seq * s.pairs);
        defer gpa.free(host);
        @memcpy(host[0 .. s.seq * s.pairs], freqs.cos);
        @memcpy(host[s.seq * s.pairs ..], freqs.sin);
        s.freqs_d = try ctx.tensorCreate(host.len * 4);
        try ctx.tensorUpload(s.freqs_d, std.mem.sliceAsBytes(host));

        const ones = try gpa.alloc(f32, cfg.head_dim);
        defer gpa.free(ones);
        @memset(ones, 1.0);
        s.ones_d = try ctx.tensorCreate(ones.len * 4);
        try ctx.tensorUpload(s.ones_d, std.mem.sliceAsBytes(ones));
        return s;
    }

    pub fn sinOff(self: Session) usize {
        return self.seq * self.pairs;
    }

    pub fn deinit(self: *Session, ctx: *Context) void {
        ctx.tensorDestroy(&self.freqs_d);
        ctx.tensorDestroy(&self.ones_d);
    }
};

/// Head width the tensor-core attention runs at: the coop P@V tile divides 128,
/// and this VAE is 64, so the operands are padded and the result unpadded.
fn attnHd(hd: usize) usize {
    return std.mem.alignForward(usize, hd, 128);
}

pub const Workspace = struct {
    x_d: Buf = nullBuf,
    hn_d: Buf = nullBuf,
    qkv_d: Buf = nullBuf,
    q_d: Buf = nullBuf,
    k_d: Buf = nullBuf,
    v_d: Buf = nullBuf,
    proj_d: Buf = nullBuf,
    gate_d: Buf = nullBuf,
    up_d: Buf = nullBuf,
    scale_d: Buf = nullBuf,
    patch_d: Buf = nullBuf,
    qp_d: Buf = nullBuf,
    kp_d: Buf = nullBuf,
    vp_d: Buf = nullBuf,
    op_d: Buf = nullBuf,
    /// Tensor-core attention scratch, at the PADDED head width.
    qh_d: Buf = nullBuf,
    kh_d: Buf = nullBuf,
    v16_d: Buf = nullBuf,
    s_d: Buf = nullBuf,
    part_d: Buf = nullBuf,
    md_d: Buf = nullBuf,

    pub const nchunks: usize = 8;

    pub fn init(ctx: *Context, dec: *const vae.VideoDecoder, seq: usize, grid: usize) !Workspace {
        const cfg = dec.cfg;
        var ws: Workspace = .{};
        errdefer ws.deinit(ctx);
        ws.x_d = try ctx.tensorCreate(seq * cfg.dim * 4);
        ws.hn_d = try ctx.tensorCreate(seq * cfg.dim * 4);
        ws.qkv_d = try ctx.tensorCreate(seq * cfg.dim * 3 * 4);
        ws.q_d = try ctx.tensorCreate(seq * cfg.dim * 4);
        ws.k_d = try ctx.tensorCreate(seq * cfg.dim * 4);
        ws.v_d = try ctx.tensorCreate(seq * cfg.dim * 4);
        ws.proj_d = try ctx.tensorCreate(seq * cfg.dim * 4);
        ws.gate_d = try ctx.tensorCreate(seq * cfg.ff * 4);
        ws.up_d = try ctx.tensorCreate(seq * cfg.ff * 4);
        ws.scale_d = try ctx.tensorCreate(dec.blocks.len * 2 * cfg.dim * 4);
        ws.patch_d = try ctx.tensorCreate(grid * cfg.patchDim() * 4);

        const hp = attnHd(cfg.head_dim);
        // ROWS_PAD, not seq: the scores pipeline tiles 128 rows and `attn_out`
        // writes whole tiles, so the padded operands and the result are all
        // sized for the padded row count. Sizing them for `seq` is a write past
        // the end, which on Vulkan surfaces as a device loss, not an error.
        const rows_pad = std.mem.alignForward(usize, seq, 128);
        const pad_w = rows_pad * cfg.heads * hp;
        ws.qp_d = try ctx.tensorCreate(pad_w * 4);
        ws.kp_d = try ctx.tensorCreate(pad_w * 4);
        ws.vp_d = try ctx.tensorCreate(pad_w * 4);
        ws.op_d = try ctx.tensorCreate(pad_w * 4);

        ws.qh_d = try ctx.tensorCreate(rows_pad * cfg.heads * hp * 2);
        ws.kh_d = try ctx.tensorCreate(rows_pad * cfg.heads * hp * 2);
        ws.v16_d = try ctx.tensorCreate(rows_pad * cfg.heads * hp * 2);
        // One head at a time: this VAE's sequence is thousands of tokens, so a
        // whole-plane batch is hundreds of MB and the card is shared.
        ws.s_d = try ctx.tensorCreate(rows_pad * rows_pad * 2);
        ws.part_d = try ctx.tensorCreate(seq * nchunks * 2 * 4);
        ws.md_d = try ctx.tensorCreate(rows_pad * 2 * 4);
        return ws;
    }

    pub fn deinit(self: *Workspace, ctx: *Context) void {
        inline for (@typeInfo(Workspace).@"struct".fields) |f| {
            if (f.type == Buf) ctx.tensorDestroy(&@field(self, f.name));
        }
    }
};

/// Widest zero bias any GEMM here needs. Passed WHOLE, never sliced: `smallBuffer`
/// caches by host POINTER, so a slice would map every width to the first length.
var zero_bias: [8192]f32 = @splat(0);

fn gemm(ctx: *Context, y: Buf, x: Buf, m: usize, w: Weight, bias: ?[]const f32) !void {
    const b: []const f32 = bias orelse zero_bias[0..w.rows];
    std.debug.assert(w.rows <= zero_bias.len);
    switch (w.dtype) {
        .bf16 => {
            if (ctx.pipe_coop_bf16w != .null_handle) {
                try ctx.opMatmulCoopBf16(y, 0, x, m, w.bytes, w.rows, w.cols, b);
            } else {
                try ctx.opMatmulCoopF16Wb(y, 0, x, m, w.bytes, w.rows, w.cols, b);
            }
        },
        .f16 => try ctx.opMatmulCoopF16Wh(y, 0, x, m, w.bytes, w.rows, w.cols, b),
        .f32 => try ctx.opMatmul(y, 0, x, 0, m, w.bytes, false, w.rows, w.cols, w.scale, bias),
        else => return error.UnsupportedDType,
    }
}

/// Weighted RMS over the last dim, `x -> out` (may alias). At rows=seq, dim=dim it
/// IS a block's pre-norm; at rows=seq*heads, dim=head_dim with a unit weight it is
/// the weightless q/k norm.
fn rmsNorm(ctx: *Context, x: Buf, out: Buf, w: Buf, rows: usize, dim: usize, eps: f32) !void {
    try ctx.opElt(.rmsnorm, x, out, w, null, .{
        .u0 = @intCast(rows),
        .u1 = @intCast(dim),
        .f0 = eps,
    }, rows, 1, 1);
}

fn normBuf(ctx: *Context, w: []const f32) !Buf {
    return .{ .buf = try ctx.smallBuffer(std.mem.sliceAsBytes(w)), .mem = .null_handle, .size = w.len * 4 };
}

/// Attention at the padded head width, result into `hn_d`.
fn attention(ctx: *Context, ws: *Workspace, seq: usize, heads: usize, hd: usize, scale: f32) !void {
    if (force_naive_attn or ctx.pipe_scores == .null_handle) {
        return ctx.opElt(.attn_full, ws.q_d, ws.k_d, ws.v_d, ws.hn_d, .{
            .u0 = @intCast(seq),
            .u1 = @intCast(heads),
            .u2 = @intCast(heads),
            .u3 = @intCast(hd),
            .f0 = scale,
        }, seq * heads, 1, 1);
    }
    const hp = attnHd(hd);
    inline for (.{ .{ "q_d", "qp_d" }, .{ "k_d", "kp_d" }, .{ "v_d", "vp_d" } }) |pair| {
        try ctx.opElt(.head_pad, @field(ws, pair[0]), @field(ws, pair[1]), null, null, .{
            .u0 = @intCast(seq * heads * hp),
            .u1 = @intCast(hp),
            .u2 = @intCast(hd),
            .u3 = @intCast(heads * hd),
            .u4 = 0,
            .u5 = @intCast(heads),
        }, seq * heads * hp, 1, 1);
    }
    try tcAttention(ctx, ws, seq, heads, hp, scale);
    // `head_unpad` takes the head count in u4, where `head_pad` takes it in u5.
    try ctx.opElt(.head_unpad, ws.op_d, ws.hn_d, null, null, .{
        .u0 = @intCast(seq * heads * hd),
        .u1 = @intCast(hd),
        .u2 = @intCast(hp),
        .u4 = @intCast(heads),
    }, seq * heads * hd, 1, 1);
}

/// The scores/softmax/PV pipeline over the padded operands, one head at a time.
fn tcAttention(ctx: *Context, ws: *Workspace, seq: usize, heads: usize, hp: usize, scale: f32) !void {
    const rows_pad = std.mem.alignForward(usize, seq, 128);
    const inner = heads * hp;
    const nchunks = Workspace.nchunks;
    ctx.independent(3);
    try ctx.opElt(.f32_to_h16, ws.qp_d, null, null, ws.qh_d, .{
        .u0 = @intCast(rows_pad * inner / 2),
        .u1 = @intCast(seq * inner),
        .f0 = scale,
    }, rows_pad * inner / 2, 1, 1);
    try ctx.opElt(.gather_kmajor_h16, ws.kp_d, null, null, ws.kh_d, .{
        .u0 = @intCast(inner * rows_pad / 2),
        .u1 = @intCast(hp),
        .u2 = @intCast(rows_pad),
        .u3 = @intCast(seq),
        .u4 = @intCast(heads),
    }, inner * rows_pad / 2, 1, 1);
    try ctx.opElt(.f32_to_h16, ws.vp_d, null, null, ws.v16_d, .{
        .u0 = @intCast(rows_pad * inner / 2),
        .u1 = @intCast(seq * inner),
        .f0 = 1.0,
    }, rows_pad * inner / 2, 1, 1);

    const s_stride: u32 = @intCast(rows_pad);
    const s_plane: u32 = @intCast(rows_pad * rows_pad);
    var h0: usize = 0;
    while (h0 < heads) : (h0 += 1) {
        try ctx.opAttnScores(ws.s_d, ws.qh_d, ws.kh_d, .{
            .u0 = @intCast(inner),
            .u1 = s_stride,
            .u2 = @intCast(h0),
            .u3 = 1,
            .u4 = @intCast(hp * rows_pad),
            .u5 = s_plane,
        }, rows_pad / 128, rows_pad / 128, 1);
        try ctx.opElt(.softmax_partial, ws.s_d, null, null, ws.part_d, .{
            .u0 = @intCast(seq * nchunks),
            .u1 = @intCast(nchunks),
            .u2 = @intCast(seq),
            .u3 = s_stride,
            .u5 = s_plane,
        }, seq * nchunks, 1, 1);
        try ctx.opElt(.softmax_combine, ws.part_d, null, null, ws.md_d, .{
            .u0 = @intCast(seq),
            .u1 = @intCast(nchunks),
            .u2 = @intCast(seq),
            .u3 = s_stride,
        }, seq, 1, 1);
        try ctx.opAttnOut(ws.s_d, ws.v16_d, ws.op_d, ws.md_d, .{
            .u0 = s_stride,
            .u1 = s_plane,
            .u2 = @intCast(h0),
            .u3 = 1,
            .u4 = @intCast(inner),
            .u5 = @intCast(inner),
            .f0 = @bitCast(@as(u32, @intCast(seq))),
            .f1 = @bitCast(s_stride),
        }, rows_pad / 128, 1);
    }
}

/// One whole-window decode, RAW (unclamped, un-finalized) planar
/// `[out_channels][t*patch_t][h*patch][w*patch]`.
pub fn decodeVolume(
    dec: *const vae.VideoDecoder,
    ctx: *Context,
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
    // then the two small projections.
    const c_in = cfg.in_channels;
    const rows = try a.alloc(f32, grid * c_in);
    for (0..grid) |i| {
        for (0..c_in) |c| rows[i * c_in + c] = z[c * grid + i] * dec.latents_std[c] + dec.latents_mean[c];
    }
    const pq = try a.alloc(f32, grid * c_in);
    try ops.matmul.matmul(io, gpa, pq, rows, grid, dec.post_quant, dec.post_quant_bias);
    const emb = try a.alloc(f32, seq * dim);
    try ops.matmul.matmul(io, gpa, emb[0 .. grid * dim], pq, grid, dec.x_embedder, dec.x_embedder_bias);
    @memcpy(emb[grid * dim ..][0 .. cfg.n_register * dim], dec.register_tokens);
    @memset(emb[(grid + cfg.n_register) * dim ..][0..dim], 0);

    const scales = try a.alloc(f32, dec.blocks.len * 2 * dim);
    for (dec.blocks, 0..) |*b, i| {
        @memcpy(scales[(i * 2) * dim ..][0..dim], b.scale1);
        @memcpy(scales[(i * 2 + 1) * dim ..][0..dim], b.scale2);
    }

    try ctx.beginBatch();
    errdefer ctx.abortBatch();
    try ctx.tensorUpload(ws.scale_d, std.mem.sliceAsBytes(scales));
    try ctx.tensorUpload(ws.x_d, std.mem.sliceAsBytes(emb));

    for (dec.blocks, 0..) |*b, bi| {
        // --- attention half ---
        try rmsNorm(ctx, ws.x_d, ws.hn_d, try normBuf(ctx, b.norm1), seq, dim, cfg.eps);
        try gemm(ctx, ws.qkv_d, ws.hn_d, seq, b.qkv, b.qkv_bias);
        try ctx.opElt(.deinterleave3, ws.qkv_d, ws.q_d, ws.k_d, ws.v_d, .{
            .u0 = @intCast(seq * dim),
            .u1 = @intCast(hd),
        }, seq * dim, 1, 1);
        // WEIGHTLESS q/k norms: a unit weight vector.
        try rmsNorm(ctx, ws.q_d, ws.q_d, sess.ones_d, seq * heads, hd, cfg.eps);
        try rmsNorm(ctx, ws.k_d, ws.k_d, sess.ones_d, seq * heads, hd, cfg.eps);
        inline for (.{ "q_d", "k_d" }) |f| {
            const total = seq * heads * sess.pairs;
            try ctx.opElt(.rope_half_part, @field(ws, f), null, sess.freqs_d, null, .{
                .u0 = @intCast(total),
                .u1 = @intCast(sess.pairs),
                .u2 = @intCast(sess.sinOff()),
                .u3 = @intCast(heads),
                .u4 = 0,
                .u5 = @intCast(hd),
            }, total, 1, 1);
        }
        try attention(ctx, ws, seq, heads, hd, attn_scale);
        try gemm(ctx, ws.proj_d, ws.hn_d, seq, b.out, b.out_bias);
        // LayerScale: `gated_add` with the learned scale where a DiT has a gate.
        try ctx.opElt(.gated_add, ws.x_d, ws.proj_d, ws.scale_d, null, .{
            .u0 = @intCast(seq * dim),
            .u1 = @intCast(dim),
            .u2 = @intCast((bi * 2) * dim),
        }, seq * dim, 1, 1);

        // --- feed-forward half ---
        try rmsNorm(ctx, ws.x_d, ws.hn_d, try normBuf(ctx, b.norm2), seq, dim, cfg.eps);
        try gemm(ctx, ws.gate_d, ws.hn_d, seq, rowSlice(b.w1, 0, cfg.ff), b.w1_bias[0..cfg.ff]);
        try gemm(ctx, ws.up_d, ws.hn_d, seq, rowSlice(b.w1, cfg.ff, cfg.ff), b.w1_bias[cfg.ff..]);
        try ctx.opElt(.silu_mul, ws.gate_d, ws.up_d, null, null, .{
            .u0 = @intCast(seq * cfg.ff),
        }, seq * cfg.ff, 1, 1);
        try gemm(ctx, ws.proj_d, ws.gate_d, seq, b.w2, b.w2_bias);
        try ctx.opElt(.gated_add, ws.x_d, ws.proj_d, ws.scale_d, null, .{
            .u0 = @intCast(seq * dim),
            .u1 = @intCast(dim),
            .u2 = @intCast((bi * 2 + 1) * dim),
        }, seq * dim, 1, 1);
    }

    // Head: a LayerNorm WITH bias, where every other norm here is RMS. Only the
    // GRID tokens are projected; the register/zero suffix is dropped.
    try ctx.opElt(.layernorm, ws.x_d, ws.hn_d, try normBuf(ctx, dec.norm_out_w), try normBuf(ctx, dec.norm_out_b), .{
        .u0 = @intCast(seq),
        .u1 = @intCast(dim),
        .f0 = cfg.eps,
    }, seq, 1, 1);
    try gemm(ctx, ws.patch_d, ws.hn_d, grid, dec.proj_out, dec.proj_out_bias);
    try ctx.endBatch();

    const patches = try a.alloc(f32, grid * cfg.patchDim());
    try ctx.tensorDownload(ws.patch_d, std.mem.sliceAsBytes(patches));
    vae.unpatchify(cfg, out, patches, t, h, w);
}
