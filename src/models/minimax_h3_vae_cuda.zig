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
const minimax_h3_vae_dev = @import("minimax_h3_vae_dev.zig");
const lin_cuda = @import("lin_cuda.zig");
const cuda = @import("tp_gpu").cuda;
const ops = @import("tp_ops");

const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;
const Weight = ops.matmul.Weight;

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
    /// The attention output, f32 (see `minimax_h3_vae_dev`); also the f32 staging an
    /// upload into a 16-bit `x_d` goes through.
    ao_d: Buf = .{},
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

    /// `act` is the dial's width for the buffers `minimax_h3_vae_dev` names.
    pub fn init(be: *Backend, dec: *const vae.VideoDecoder, seq: usize, grid: usize, act: cuda.backend.Dt) !Workspace {
        const cfg = dec.cfg;
        var ws: Workspace = .{};
        errdefer ws.deinit(be);
        const ab = act.bytes();
        inline for (.{ .{ "x_d", cfg.dim }, .{ "hn_d", cfg.dim }, .{ "proj_d", cfg.dim }, .{ "gate_d", cfg.ff }, .{ "up_d", cfg.ff } }) |f| {
            @field(ws, f[0]) = try be.tensorCreate(seq * f[1] * ab);
            @field(ws, f[0]).dt = act;
        }
        ws.ao_d = try be.tensorCreate(seq * cfg.dim * 4);
        ws.qkv_d = try be.tensorCreate(seq * cfg.dim * 3 * 4);
        ws.q_d = try be.tensorCreate(seq * cfg.dim * 4);
        ws.k_d = try be.tensorCreate(seq * cfg.dim * 4);
        ws.v_d = try be.tensorCreate(seq * cfg.dim * 4);
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
        inline for (@typeInfo(Workspace).@"struct".fields) |f| be.tensorDestroy(&@field(self, f.name));
        self.* = undefined;
    }
};

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
/// The tensor-core path stores its scores in f16 whichever of its two variants
/// runs, and this VAE attends over thousands of keys at a 2048-wide dim, which is
/// where f16's 65504 ceiling starts to bite. This is the A/B for that.
pub var force_naive_attn: bool = false;

/// Report the residual stream's and the FFN product's magnitude per block (`TP_VAE_MAG`).
pub var dbg_mag: bool = false;
var dbg_ffn: f32 = 0;

fn maxAbs(be: *Backend, a: std.mem.Allocator, b: Buf, n: usize) !f32 {
    const h = try a.alloc(f32, n);
    defer a.free(h);
    try be.tensorDownload(b, std.mem.sliceAsBytes(h));
    var mx: f32 = 0;
    for (h) |v| mx = if (std.math.isFinite(v)) @max(mx, @abs(v)) else std.math.inf(f32);
    return mx;
}

/// Attention, padding the head width when the backend needs it. The padding is
/// exact: the extra columns are zero on both sides, so they contribute nothing to
/// the scores and `opHeadUnpad` drops them again.
fn attentionAt(be: *Backend, ws: *Workspace, seq: usize, heads: usize, hd: usize, scale: f32) !void {
    if (force_naive_attn) {
        return be.attn(ws.q_d, ws.k_d, ws.v_d, ws.ao_d, seq, seq, heads, heads, hd, scale, false);
    }
    const hp = attnHd(be, hd);
    if (hp == hd) return be.opAttnTC(ws.q_d, ws.k_d, ws.v_d, ws.ao_d, seq, heads, heads, hd, scale);
    inline for (.{ .{ "q_d", "qp_d" }, .{ "k_d", "kp_d" }, .{ "v_d", "vp_d" } }) |pair| {
        try be.opHeadPad(@field(ws, pair[1]), @field(ws, pair[0]), seq, heads, hp, hd, heads * hd, 0);
    }
    try be.opAttnTC(ws.qp_d, ws.kp_d, ws.vp_d, ws.op_d, seq, heads, heads, hp, scale);
    try be.opHeadUnpad(ws.op_d, ws.ao_d, seq, heads, hd, hp);
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
    const grid = t * h * w;
    const seq = grid + cfg.n_register + 1;
    std.debug.assert(sess.seq == seq and sess.grid == grid);

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const emb = try vae.embedWindow(a, io, gpa, dec, z, grid);
    const scales = try vae.layerScales(a, dec);

    try be.beginBatch();
    errdefer if (be.batching()) be.abortBatch();
    try be.tensorUpload(ws.scale_d, std.mem.sliceAsBytes(scales));
    if (ws.x_d.dt == .f32) {
        try be.tensorUpload(ws.x_d, std.mem.sliceAsBytes(emb));
    } else {
        try be.tensorUpload(ws.ao_d, std.mem.sliceAsBytes(emb));
        try be.opCast(ws.ao_d, ws.x_d, seq * cfg.dim, 0);
    }
    try minimax_h3_vae_dev.body(Dev{ .be = be, .sess = sess, .ws = ws, .cfg = cfg, .a = a }, dec, seq, grid);
    try be.endBatch();

    const patches = try a.alloc(f32, grid * cfg.patchDim());
    try be.tensorDownload(ws.patch_d, std.mem.sliceAsBytes(patches));
    vae.unpatchify(cfg, out, patches, t, h, w);
}

/// The `minimax_h3_vae_dev` adapter.
const Dev = struct {
    be: *Backend,
    sess: *const Session,
    ws: *Workspace,
    cfg: vae.Config,
    a: std.mem.Allocator,

    /// `qkNorm` is a weighted RMS over its last dim, so at rows=seq, dim=dim it IS a
    /// block's pre-norm.
    pub fn rmsNorm(d: Dev, x: Buf, y: Buf, w: []const f32, rows: usize, dim: usize) !void {
        try d.be.qkNorm(x, y, try normBuf(d.be, w), rows, dim, d.cfg.eps);
    }

    /// `qkNorm` always multiplies a weight, so the weightless norm is fed ones.
    pub fn unitNorm(d: Dev, x: Buf, seq: usize) !void {
        try d.be.qkNorm(x, x, d.sess.ones_d, seq * d.cfg.heads, d.cfg.head_dim, d.cfg.eps);
    }

    pub fn gemm(d: Dev, y: Buf, x: Buf, m: usize, w: Weight, bias: ?[]const f32) !void {
        try lin_cuda.towerLinear(d.be, y, x, m, w, bias, null);
    }

    pub fn deinterleave3(d: Dev, seq: usize) !void {
        try d.be.opDeinterleave3(d.ws.qkv_d, d.ws.q_d, d.ws.k_d, d.ws.v_d, seq * d.cfg.dim, d.cfg.head_dim);
    }

    pub fn rope(d: Dev, x: Buf, seq: usize) !void {
        try d.be.opRopeHalfPart(x, d.sess.freqs_d, seq, d.cfg.heads, d.sess.pairs, d.sess.sinOff(), 0, d.cfg.head_dim);
    }

    pub fn attention(d: Dev, seq: usize) !void {
        try attentionAt(d.be, d.ws, seq, d.cfg.heads, d.cfg.head_dim, 1.0 / @sqrt(@as(f32, @floatFromInt(d.cfg.head_dim))));
    }

    pub fn layerScale(d: Dev, x: Buf, y: Buf, total: usize, off: usize) !void {
        try d.be.gatedAdd(x, y, d.ws.scale_d, total, d.cfg.dim, off);
    }

    pub fn siluMul(d: Dev, g: Buf, u: Buf, n: usize) !void {
        try d.be.siluMul(g, u, n);
        if (dbg_mag and g.dt == .f32) {
            try d.be.endBatch();
            dbg_ffn = try maxAbs(d.be, d.a, g, n);
            try d.be.beginBatch();
        }
    }

    pub fn layerNorm(d: Dev, x: Buf, y: Buf, w: []const f32, b: []const f32, rows: usize, dim: usize) !void {
        try d.be.opLayerNorm(x, y, w, b, rows, dim, d.cfg.eps);
    }

    /// `TP_VAE_MAG`: the residual stream's magnitude per block. The device GEMMs
    /// narrow activations to f16, and f16's 65504 ceiling is a limit real VAE
    /// checkpoints do reach, so this is the measurement rather than the assumption.
    /// Costs a full sync per block.
    pub fn blockDone(d: Dev, bi: usize, seq: usize) !void {
        if (!dbg_mag) return;
        // `ao_d` is dead between blocks, so it can hold an f32 copy of a 16-bit stream.
        const x32 = if (d.ws.x_d.dt == .f32) d.ws.x_d else blk: {
            try d.be.opCast(d.ws.x_d, d.ws.ao_d, seq * d.cfg.dim, 0);
            break :blk d.ws.ao_d;
        };
        try d.be.endBatch();
        const probe = try d.a.alloc(f32, seq * d.cfg.dim);
        try d.be.tensorDownload(x32.slice(0, probe.len * 4), std.mem.sliceAsBytes(probe));
        var mx: f32 = 0;
        var nonfinite: usize = 0;
        for (probe) |v| {
            if (!std.math.isFinite(v)) nonfinite += 1 else mx = @max(mx, @abs(v));
        }
        std.debug.print("vae block {d:>2}: max|x| {d:.1}  max|ffn| {d:.1}{s}\n", .{ bi, mx, dbg_ffn, if (nonfinite != 0) " NONFINITE" else "" });
        try d.be.beginBatch();
    }
};
