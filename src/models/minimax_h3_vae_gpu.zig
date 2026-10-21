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
const minimax_h3_vae_dev = @import("minimax_h3_vae_dev.zig");
const lin_gpu = @import("lin_gpu.zig");
const attn_gpu = @import("attn_gpu.zig");
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
    /// The linear dispatcher, its zero bias sized for the widest projection.
    lin: lin_gpu.Lin,
    x_d: Buf = nullBuf,
    /// The attention output, f32 (see `minimax_h3_vae_dev`); also the f32 staging an
    /// upload into a 16-bit `x_d` goes through.
    ao_d: Buf = nullBuf,
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

    /// `act` is the dial's width for the buffers `minimax_h3_vae_dev` names.
    pub fn init(ctx: *Context, gpa: std.mem.Allocator, dec: *const vae.VideoDecoder, seq: usize, grid: usize, act: gpu.Dt) !Workspace {
        const cfg = dec.cfg;
        var ws: Workspace = .{ .lin = try lin_gpu.Lin.init(gpa, ctx, @max(3 * cfg.dim, cfg.ff, cfg.patchDim()), 0, 0) };
        errdefer ws.deinit(ctx);
        const ab = act.bytes();
        inline for (.{ .{ "x_d", cfg.dim }, .{ "hn_d", cfg.dim }, .{ "proj_d", cfg.dim }, .{ "gate_d", cfg.ff }, .{ "up_d", cfg.ff } }) |f| {
            @field(ws, f[0]) = try ctx.tensorCreate(seq * f[1] * ab);
            @field(ws, f[0]).dt = act;
        }
        ws.ao_d = try ctx.tensorCreate(seq * cfg.dim * 4);
        ws.qkv_d = try ctx.tensorCreate(seq * cfg.dim * 3 * 4);
        ws.q_d = try ctx.tensorCreate(seq * cfg.dim * 4);
        ws.k_d = try ctx.tensorCreate(seq * cfg.dim * 4);
        ws.v_d = try ctx.tensorCreate(seq * cfg.dim * 4);
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
        inline for (@typeInfo(Workspace).@"struct".field_names, @typeInfo(Workspace).@"struct".field_types) |name, FieldType| {
            if (FieldType == Buf) ctx.tensorDestroy(&@field(self, name));
        }
        self.lin.deinit();
    }
};

/// Weighted RMS over the last dim, `x -> out` (may alias). At rows=seq, dim=dim it
/// IS a block's pre-norm; at rows=seq*heads, dim=head_dim with a unit weight it is
/// the weightless q/k norm.
fn rmsNormAt(ctx: *Context, x: Buf, out: Buf, w: Buf, rows: usize, dim: usize, eps: f32) !void {
    std.debug.assert(x.dt == out.dt);
    try ctx.opElt(gpu.eltAt("rmsnorm", x.dt), x, out, w, null, .{
        .u0 = @intCast(rows),
        .u1 = @intCast(dim),
        .f0 = eps,
    }, rows, 1, 1);
}

fn normBuf(ctx: *Context, w: []const f32) !Buf {
    return .{ .buf = try ctx.smallBuffer(std.mem.sliceAsBytes(w)), .mem = .null_handle, .size = w.len * 4 };
}

/// Attention at the padded head width, result into `ao_d`.
fn attentionAt(ctx: *Context, ws: *Workspace, seq: usize, heads: usize, hd: usize, scale: f32) !void {
    if (force_naive_attn or ctx.pipe_scores == .null_handle) {
        return ctx.opElt(.attn_full, ws.q_d, ws.k_d, ws.v_d, ws.ao_d, .{
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
    try ctx.opElt(.head_unpad, ws.op_d, ws.ao_d, null, null, .{
        .u0 = @intCast(seq * heads * hd),
        .u1 = @intCast(hd),
        .u2 = @intCast(hp),
        .u4 = @intCast(heads),
    }, seq * heads * hd, 1, 1);
}

/// The scores/softmax/PV pipeline over the padded operands, one head at a time.
fn tcAttention(ctx: *Context, ws: *Workspace, seq: usize, heads: usize, hp: usize, scale: f32) !void {
    try attn_gpu.stage(ctx, ws.qp_d, ws.kp_d, ws.vp_d, ws.qh_d, ws.kh_d, ws.v16_d, seq, heads, heads, hp, scale);
    try attn_gpu.core(ctx, .{ .s = ws.s_d, .part = ws.part_d, .md = ws.md_d }, ws.qh_d, ws.kh_d, ws.v16_d, ws.op_d, .{
        .q_rows = seq,
        .kv_rows = seq,
        .heads = heads,
        .kv_heads = heads,
        .hd = hp,
        .max_hb = 1,
        .nchunks = Workspace.nchunks,
    }, null);
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
    const grid = t * h * w;
    const seq = grid + cfg.n_register + 1;
    std.debug.assert(sess.seq == seq and sess.grid == grid);

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const emb = try vae.embedWindow(a, io, gpa, dec, z, grid);
    const scales = try vae.layerScales(a, dec);

    try ctx.beginBatch();
    errdefer if (ctx.batching) ctx.abortBatch();
    try ctx.tensorUpload(ws.scale_d, std.mem.sliceAsBytes(scales));
    if (ws.x_d.dt == .f32) {
        try ctx.tensorUpload(ws.x_d, std.mem.sliceAsBytes(emb));
    } else {
        try ctx.tensorUpload(ws.ao_d, std.mem.sliceAsBytes(emb));
        const n = seq * cfg.dim;
        try ctx.opElt(gpu.eltCopy(.f32, ws.x_d.dt), ws.ao_d, ws.x_d, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
    }
    try minimax_h3_vae_dev.body(Dev{ .ctx = ctx, .sess = sess, .ws = ws, .cfg = cfg }, dec, seq, grid);
    try ctx.endBatch();

    const patches = try a.alloc(f32, grid * cfg.patchDim());
    try ctx.tensorDownload(ws.patch_d, std.mem.sliceAsBytes(patches));
    vae.unpatchify(cfg, out, patches, t, h, w);
}

/// The `minimax_h3_vae_dev` adapter.
const Dev = struct {
    ctx: *Context,
    sess: *const Session,
    ws: *Workspace,
    cfg: vae.Config,

    pub fn rmsNorm(d: Dev, x: Buf, y: Buf, w: []const f32, rows: usize, dim: usize) !void {
        try rmsNormAt(d.ctx, x, y, try normBuf(d.ctx, w), rows, dim, d.cfg.eps);
    }

    pub fn unitNorm(d: Dev, x: Buf, seq: usize) !void {
        try rmsNormAt(d.ctx, x, x, d.sess.ones_d, seq * d.cfg.heads, d.cfg.head_dim, d.cfg.eps);
    }

    pub fn gemm(d: Dev, y: Buf, x: Buf, m: usize, w: Weight, bias: ?[]const f32) !void {
        try lin_gpu.towerLinear(&d.ws.lin, y, x, m, w, bias);
    }

    pub fn deinterleave3(d: Dev, seq: usize) !void {
        const n = seq * d.cfg.dim;
        try d.ctx.opElt(.deinterleave3, d.ws.qkv_d, d.ws.q_d, d.ws.k_d, d.ws.v_d, .{
            .u0 = @intCast(n),
            .u1 = @intCast(d.cfg.head_dim),
        }, n, 1, 1);
    }

    pub fn rope(d: Dev, x: Buf, seq: usize) !void {
        const total = seq * d.cfg.heads * d.sess.pairs;
        try d.ctx.opElt(.rope_half_part, x, null, d.sess.freqs_d, null, .{
            .u0 = @intCast(total),
            .u1 = @intCast(d.sess.pairs),
            .u2 = @intCast(d.sess.sinOff()),
            .u3 = @intCast(d.cfg.heads),
            .u4 = 0,
            .u5 = @intCast(d.cfg.head_dim),
        }, total, 1, 1);
    }

    pub fn attention(d: Dev, seq: usize) !void {
        try attentionAt(d.ctx, d.ws, seq, d.cfg.heads, d.cfg.head_dim, 1.0 / @sqrt(@as(f32, @floatFromInt(d.cfg.head_dim))));
    }

    pub fn layerScale(d: Dev, x: Buf, y: Buf, total: usize, off: usize) !void {
        std.debug.assert(x.dt == y.dt);
        try d.ctx.opElt(gpu.eltAt("gated_add", x.dt), x, y, d.ws.scale_d, null, .{
            .u0 = @intCast(total),
            .u1 = @intCast(d.cfg.dim),
            .u2 = @intCast(off),
        }, total, 1, 1);
    }

    pub fn siluMul(d: Dev, g: Buf, u: Buf, n: usize) !void {
        std.debug.assert(g.dt == u.dt);
        try d.ctx.opElt(gpu.eltAt("silu_mul", g.dt), g, u, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
    }

    pub fn layerNorm(d: Dev, x: Buf, y: Buf, w: []const f32, b: []const f32, rows: usize, dim: usize) !void {
        std.debug.assert(x.dt == y.dt);
        try d.ctx.opElt(gpu.eltAt("layernorm", x.dt), x, y, try normBuf(d.ctx, w), try normBuf(d.ctx, b), .{
            .u0 = @intCast(rows),
            .u1 = @intCast(dim),
            .f0 = d.cfg.eps,
        }, rows, 1, 1);
    }

    pub fn blockDone(_: Dev, _: usize, _: usize) !void {}
};
