//! GPU-resident Mage-Flow forward on the CUDA backends (`zig-cuda`'s hand-PTX
//! and `cuda`'s vendor libraries), the device twin of `mageflow.DiT.forward`.
//!
//! The double-stream block is what makes this not a copy of any other family's
//! arm. Text and image tokens keep their own modulation, norms, projections and
//! MLP and meet only inside ONE attention, so every stage runs twice over two
//! row ranges of one buffer and only the softmax runs once. Three consequences
//! shape the code:
//!
//! 1. **`x_d` holds the joint sequence `[text | image]`**, and each stream is a
//!    row range of it. The per-stream ops take an offset view; only `opAttnTC`
//!    sees the whole thing.
//! 2. **The text stream's q/k/v go to their own buffers and are COPIED in.** A
//!    quantized GEMM stores whole 128-row tiles, so writing the text half at row
//!    0 of the joint buffer would spill into the image half's first rows and the
//!    correctness would rest on the image GEMM running afterwards. The copy is
//!    ~4 MB a block against a ~100-token text half, which buys that away.
//! 3. **The modulation is computed on the HOST, once per sigma.** Its two
//!    linears are `[6 * features][features]` each, a third of the model's
//!    parameters, and they depend only on the timestep, so `modulationTable`
//!    runs at session setup and 1.4 GB of weights never reach the device. That
//!    is also why they are not in `DiT.device_lins`.
//!
//! Numerics match the CPU forward up to floating-point reordering and whatever
//! the checkpoint's own GEMM route costs, the same regime every other family's
//! CUDA arm runs in. `mageflow-cuda-test` is the gate.

const std = @import("std");
const mageflow = @import("mageflow.zig");
const lin_cuda = @import("lin_cuda.zig");
const cuda = @import("tp_gpu").cuda;
const ops = @import("tp_ops");

const DiT = mageflow.DiT;
const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;

const F = mageflow.features; // 3072
const heads = mageflow.n_heads; // 24
const hd = mageflow.head_dim; // 128
const half = hd / 2;
const mlp_dim = mageflow.mlp_dim; // 12288
const channels = mageflow.channels; // 128
const eps: f32 = 1e-6;
const attn_scale: f32 = 1.0 / 11.313708498984761; // 1/sqrt(128)

/// Force the naive one-thread-per-(query,head) attention instead of `opAttnTC`.
/// For A/B and for reproducing a mismatch; the device test runs both.
pub var force_naive_attn: bool = false;

/// Add each block GEMM's bias in a separate pass instead of folding it into the
/// GEMM epilogue. A/B only: the fold is what runs, and this is how its effect is
/// measured against 3090 clock noise (interleave the two in ONE process).
pub var split_bias: bool = false;

/// One block linear plus its bias, folded into the GEMM or added after it.
fn gemmB(be: *Backend, plan: lin_cuda.Plan, y: Buf, x: Buf, m: usize, lw: anytype) !void {
    if (split_bias) {
        try lin_cuda.gemm(be, plan, y, x, m, lw.w, false);
        return be.opAddBiasRows(y, try normBuf(be, lw.b.?), m, lw.w.rows, 0, false);
    }
    try lin_cuda.gemmBias(be, plan, y, x, m, lw.w, false, lw.b.?);
}

/// Rows a GEMM's destination must hold past its offset. Every quantized route
/// launches over `align(m, 128)` rows and each block stores a whole tile.
fn padRows(n: usize) usize {
    return std.mem.alignForward(usize, n, 128);
}

/// Per-image cache: everything constant across sampling steps.
pub const Session = struct {
    /// What `lin_cuda.plan` decided for this checkpoint's block linears.
    plan: lin_cuda.Plan,
    lat_h: usize,
    lat_w: usize,
    seq_txt: usize,
    /// Canvas tokens, i.e. what the output is truncated back to.
    n_img: usize,
    /// Canvas plus every reference image's tokens.
    n_tok: usize,
    seq: usize,

    /// `[seq_txt][F]`, the conditioning already through `txt_norm` and `txt_in`.
    txt_d: Buf,
    /// `[n_tok][channels]` for the reference tokens, at their place in the
    /// stream. The canvas half is rewritten per step; this half never changes.
    tokens: []f32,
    /// Interleaved-RoPE table for the joint sequence, `cos` then `sin`.
    freqs_d: Buf,

    sigmas: []f32,
    /// `[sigmas.len][modStride]`, host-built; see the header.
    mods: []f32,
    mod_stride: usize,

    pub fn init(
        gpa: std.mem.Allocator,
        io: std.Io,
        be: *Backend,
        model: *const DiT,
        lat_h: usize,
        lat_w: usize,
        ctx: []const f32,
        seq_txt: usize,
        refs: []const mageflow.Ref,
        sigmas: []const f32,
    ) !Session {
        std.debug.assert(ctx.len == seq_txt * mageflow.txt_dim);
        const n_img = lat_h * lat_w;
        var n_tok = n_img;
        for (refs) |r| n_tok += r.h * r.w;

        var self: Session = .{
            .plan = try lin_cuda.plan(model.device_lins, lin_cuda.blockq_gemm, "mageflow cuda"),
            .lat_h = lat_h,
            .lat_w = lat_w,
            .seq_txt = seq_txt,
            .n_img = n_img,
            .n_tok = n_tok,
            .seq = seq_txt + n_tok,
            .txt_d = undefined,
            .tokens = &.{},
            .freqs_d = undefined,
            .sigmas = &.{},
            .mods = &.{},
            .mod_stride = model.modStride(),
        };
        var made: usize = 0;
        errdefer {
            const bufs = [_]*Buf{ &self.txt_d, &self.freqs_d };
            for (bufs[0..made]) |b| be.tensorDestroy(b);
            if (self.tokens.len != 0) gpa.free(self.tokens);
            if (self.sigmas.len != 0) gpa.free(self.sigmas);
            if (self.mods.len != 0) gpa.free(self.mods);
        }

        try lin_cuda.presize(be, self.plan, model.device_lins);

        // The text half, whole, on the host: one RMSNorm and one GEMM over ~100
        // rows, once per conditioning rather than once per step.
        {
            const txt = try model.textTokens(io, gpa, ctx, seq_txt);
            defer gpa.free(txt);
            self.txt_d = try be.tensorCreate(padRows(seq_txt) * F * 4);
            made += 1;
            try be.tensorUpload(self.txt_d, std.mem.sliceAsBytes(txt));
        }

        // The image half's token matrix. Only the canvas rows change per step,
        // so the references are written once here.
        self.tokens = try gpa.alloc(f32, n_tok * channels);
        {
            var at = n_img * channels;
            for (refs) |r| {
                const n = r.h * r.w;
                mageflow.interleaveChannels(self.tokens[at..][0 .. n * channels], r.lat, n);
                at += n * channels;
            }
        }

        {
            var freqs = try model.ropeFreqs(gpa, lat_h, lat_w, seq_txt, refs);
            defer freqs.deinit(gpa);
            const flat = try gpa.alloc(f32, 2 * self.seq * half);
            defer gpa.free(flat);
            @memcpy(flat[0 .. self.seq * half], freqs.cos);
            @memcpy(flat[self.seq * half ..], freqs.sin);
            self.freqs_d = try be.tensorCreate(flat.len * 4);
            made += 1;
            try be.tensorUpload(self.freqs_d, std.mem.sliceAsBytes(flat));
        }

        self.sigmas = try gpa.dupe(f32, sigmas);
        self.mods = try gpa.alloc(f32, sigmas.len * self.mod_stride);
        for (sigmas, 0..) |s, i| {
            const tbl = try model.modulationTable(io, gpa, s);
            defer gpa.free(tbl);
            @memcpy(self.mods[i * self.mod_stride ..][0..self.mod_stride], tbl);
        }
        return self;
    }

    pub fn deinit(self: *Session, gpa: std.mem.Allocator, be: *Backend) void {
        be.tensorDestroy(&self.txt_d);
        be.tensorDestroy(&self.freqs_d);
        gpa.free(self.tokens);
        gpa.free(self.sigmas);
        gpa.free(self.mods);
        self.* = undefined;
    }

    fn modsFor(self: *const Session, sigma: f32) ?[]const f32 {
        for (self.sigmas, 0..) |s, i| {
            if (s == sigma) return self.mods[i * self.mod_stride ..][0..self.mod_stride];
        }
        return null;
    }
};

pub const Workspace = struct {
    x_d: Buf,
    ntxt_d: Buf,
    nimg_d: Buf,
    q_d: Buf,
    k_d: Buf,
    v_d: Buf,
    attn_d: Buf,
    qt_d: Buf,
    kt_d: Buf,
    vt_d: Buf,
    dtxt_d: Buf,
    dimg_d: Buf,
    mt_d: Buf,
    mi_d: Buf,
    mod_d: Buf,
    imgin_d: Buf,
    out_d: Buf,

    const fields = [_][]const u8{
        "x_d",    "ntxt_d", "nimg_d", "q_d",  "k_d",  "v_d",     "attn_d",  "qt_d", "kt_d",
        "vt_d",   "dtxt_d", "dimg_d", "mt_d", "mi_d", "mod_d",   "imgin_d", "out_d",
    };

    pub fn init(be: *Backend, model: *const DiT, seq_txt: usize, n_tok: usize) !Workspace {
        const tp = padRows(seq_txt);
        const ip = padRows(n_tok);
        // The image half's GEMMs write at row `seq_txt` and store whole tiles, so
        // the joint buffers hold `seq_txt + align(n_tok, 128)` rows.
        const joint = seq_txt + ip;
        const sizes = [fields.len]usize{
            joint * F * 4,
            tp * F * 4,
            ip * F * 4,
            joint * F * 4,
            joint * F * 4,
            joint * F * 4,
            joint * F * 4,
            tp * F * 4,
            tp * F * 4,
            tp * F * 4,
            tp * F * 4,
            ip * F * 4,
            tp * mlp_dim * 4,
            ip * mlp_dim * 4,
            model.modStride() * 4,
            n_tok * channels * 4,
            n_tok * channels * 4,
        };
        var self: Workspace = undefined;
        var made: usize = 0;
        errdefer inline for (fields, 0..) |name, i| {
            if (i < made) be.tensorDestroy(&@field(self, name));
        };
        inline for (fields, sizes) |name, size| {
            @field(self, name) = try be.tensorCreate(size);
            made += 1;
        }
        return self;
    }

    pub fn deinit(self: *Workspace, be: *Backend) void {
        inline for (fields) |name| be.tensorDestroy(&@field(self, name));
        self.* = undefined;
    }
};

/// Whether the CUDA arms can run every block linear this model has. The refusal,
/// if any, is logged by name; the caller says the trunk then runs on the CPU.
pub fn supported(model: *const DiT) bool {
    if (model.blocks.len == 0) return false;
    _ = lin_cuda.plan(model.device_lins, lin_cuda.blockq_gemm, "mageflow cuda") catch return false;
    return true;
}

/// Exact power-of-two prescale on V across the attention's f16 cast.
///
/// Q and K are RMS-normed per head on the way in, so they arrive at O(10); V is
/// NOT normed and carries whatever the modulation's `1 + scale` puts on the
/// block's input. MEASURED on the released 4B checkpoint at a real conditioning,
/// peak |V| over all 12 blocks:
///
///     256x256    74056     1024x1024  509538     2048x2048  538648
///
/// all past f16's 65504 ceiling, so `opAttnTC` turned V to inf and
/// `softmax @ V` to NaN. The render was solid white with no error, which is this
/// codebase's fifth f16-range incident on a diffusion trunk.
///
/// ⚠️ Two things about that table are worth keeping. A random N(0, 1)
/// conditioning does NOT reach the ceiling -- the device check read 3.4e-3
/// without this, green -- so the axis it rides is the CONDITIONING's magnitude,
/// and a synthetic check cannot see it. And the peak does NOT grow with
/// resolution past 1024, so the headroom below is a fixed margin rather than
/// something to re-derive per image.
///
/// Attention is exactly linear in V, so scaling V down and the output back up is
/// the same arithmetic, and a power of two shifts the exponent rather than the
/// mantissa, so it costs no relative precision. 2^-7 takes the safe ceiling to
/// 8.4e6, 15x over the measured peak; V would have to reach f16's normal
/// minimum (6.1e-5 / 128 = 7.8e-3) before a value lost mantissa bits, and its
/// typical magnitude is three orders above that.
const v_div_log2: u5 = 7;
const v_div: f32 = 1.0 / @as(f32, 1 << v_div_log2);

/// DIAGNOSTIC: report a device buffer's magnitude and non-finite count
/// (`TP_MAGEFLOW_TRACE`).
///
/// A forward that comes back all-NaN says nothing about where it turned, and the
/// stage boundaries are the only places a host-side check can look without a
/// kernel per op. Costs a sync and a download per call, so it is env-gated and
/// never on in a render.
var trace_on: ?bool = null;

fn trace(be: *Backend, what: []const u8, i: usize, buf: Buf, elems: usize) void {
    if (trace_on == null) trace_on = std.c.getenv("TP_MAGEFLOW_TRACE") != null;
    if (!trace_on.?) return;
    const host = be.gpa.alloc(f32, elems) catch return;
    defer be.gpa.free(host);
    const batching = be.batching();
    if (batching) be.endBatch() catch return;
    be.tensorDownload(buf, std.mem.sliceAsBytes(host)) catch return;
    var mx: f32 = 0;
    var bad: usize = 0;
    for (host) |v| {
        if (!std.math.isFinite(v)) bad += 1 else mx = @max(mx, @abs(v));
    }
    std.debug.print("[mageflow-cuda] {s} {d:>2}  max|x| {d:12.1}  non-finite {d}/{d}\n", .{ what, i, mx, bad, elems });
    if (batching) be.beginBatch() catch return;
}

/// One denoiser forward. `out`/`x_lat` are planar `[channels][lat_h][lat_w]`.
pub fn forward(
    model: *const DiT,
    be: *Backend,
    sess: *Session,
    ws: *Workspace,
    io: std.Io,
    gpa: std.mem.Allocator,
    out: []f32,
    x_lat: []const f32,
    sigma: f32,
    cancel: ?*std.atomic.Value(bool),
) !void {
    std.debug.assert(x_lat.len == channels * sess.lat_h * sess.lat_w);
    std.debug.assert(out.len == x_lat.len);
    const seq_txt = sess.seq_txt;
    const n_tok = sess.n_tok;
    const txt_bytes = seq_txt * F * 4;

    var owned: ?[]f32 = null;
    defer if (owned) |b| gpa.free(b);
    const mods = sess.modsFor(sigma) orelse blk: {
        owned = try model.modulationTable(io, gpa, sigma);
        break :blk owned.?;
    };

    // Only the canvas rows move per step; the references were written at init.
    mageflow.interleaveChannels(sess.tokens[0 .. sess.n_img * channels], x_lat, sess.n_img);

    try be.tensorUpload(ws.mod_d, std.mem.sliceAsBytes(mods));
    try be.tensorUpload(ws.imgin_d, std.mem.sliceAsBytes(sess.tokens));

    try be.beginBatch();
    errdefer if (be.batching()) be.abortBatch();

    // The joint stream: the conditioning at row 0, the image tokens after it.
    try be.tensorCopy(ws.x_d, 0, sess.txt_d, 0, txt_bytes);
    try be.opMatmul(ws.x_d, txt_bytes, ws.imgin_d, 0, n_tok, model.img_in.w.bytes, false, F, channels, model.img_in.w.scale, model.img_in.b);

    trace(be, "txt_in", 0, ws.x_d, seq_txt * F);
    trace(be, "img_in", 0, offsetBuf(ws.x_d, txt_bytes, n_tok * F * 4), n_tok * F);
    if (be.async_uploads and model.blocks.len > 0) prefetchBlock(be, &model.blocks[0]);
    for (model.blocks, 0..) |*blk, i| {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        if (be.async_uploads and i + 1 < model.blocks.len) prefetchBlock(be, &model.blocks[i + 1]);
        try blockForward(be, sess, ws, blk, i);
        trace(be, "block", i, ws.x_d, (seq_txt + n_tok) * F);
    }

    // The final layer, on the canvas rows alone: it is row-wise, so running it
    // on them is what ComfyUI's "whole stream, then slice" computes.
    const fin = model.blocks.len * 12 * F;
    try be.lnMod(offsetBuf(ws.x_d, txt_bytes, sess.n_img * F * 4), offsetBuf(ws.x_d, txt_bytes, sess.n_img * F * 4), ws.mod_d, sess.n_img, F, fin + 0 * F, fin + 1 * F, eps);
    try be.opMatmul(ws.out_d, 0, ws.x_d, txt_bytes, sess.n_img, model.proj_out.w.bytes, false, channels, F, model.proj_out.w.scale, model.proj_out.b);
    try be.endBatch();

    const final = try gpa.alloc(f32, sess.n_img * channels);
    defer gpa.free(final);
    try be.tensorDownload(offsetBuf(ws.out_d, 0, final.len * 4), std.mem.sliceAsBytes(final));
    mageflow.deinterleaveChannels(out, final, sess.n_img);
}

fn blockForward(
    be: *Backend,
    sess: *Session,
    ws: *Workspace,
    blk: *const mageflow.Block,
    block_i: usize,
) !void {
    const seq_txt = sess.seq_txt;
    const n_tok = sess.n_tok;
    const seq = sess.seq;
    const txt_bytes = seq_txt * F * 4;
    const plan = sess.plan;
    // Per block: the image stream's six chunks then the text stream's six.
    const img_m = block_i * 12 * F;
    const txt_m = img_m + 6 * F;

    const txt_x = offsetBuf(ws.x_d, 0, seq_txt * F * 4);
    const img_x = offsetBuf(ws.x_d, txt_bytes, n_tok * F * 4);

    // --- attention half -----------------------------------------------------
    try be.lnMod(txt_x, ws.ntxt_d, ws.mod_d, seq_txt, F, txt_m + 0 * F, txt_m + 1 * F, eps);
    try be.lnMod(img_x, ws.nimg_d, ws.mod_d, n_tok, F, img_m + 0 * F, img_m + 1 * F, eps);

    // Text first into its own buffers, then copied in: see the header's note 2.
    try lin_cuda.prep(be, plan, ws.ntxt_d, seq_txt, F, &.{ blk.attn.txt_q.w, blk.attn.txt_k.w, blk.attn.txt_v.w }, false);
    try gemmB(be, plan, ws.qt_d, ws.ntxt_d, seq_txt, blk.attn.txt_q);
    try gemmB(be, plan, ws.kt_d, ws.ntxt_d, seq_txt, blk.attn.txt_k);
    try gemmB(be, plan, ws.vt_d, ws.ntxt_d, seq_txt, blk.attn.txt_v);
    try be.tensorCopy(ws.q_d, 0, ws.qt_d, 0, txt_bytes);
    try be.tensorCopy(ws.k_d, 0, ws.kt_d, 0, txt_bytes);
    try be.tensorCopy(ws.v_d, 0, ws.vt_d, 0, txt_bytes);

    const qi = offsetBuf(ws.q_d, txt_bytes, n_tok * F * 4);
    const ki = offsetBuf(ws.k_d, txt_bytes, n_tok * F * 4);
    const vi = offsetBuf(ws.v_d, txt_bytes, n_tok * F * 4);
    try lin_cuda.prep(be, plan, ws.nimg_d, n_tok, F, &.{ blk.attn.q.w, blk.attn.k.w, blk.attn.v.w }, false);
    try gemmB(be, plan, qi, ws.nimg_d, n_tok, blk.attn.q);
    try gemmB(be, plan, ki, ws.nimg_d, n_tok, blk.attn.k);
    try gemmB(be, plan, vi, ws.nimg_d, n_tok, blk.attn.v);

    // Per-head QK-norm, each stream with its own weight, then RoPE over the
    // whole joint sequence: the text rows sit at position 0, whose rotation is
    // the identity, so applying it there is the same thing as not.
    try be.qkNorm(ws.q_d, ws.q_d, try normBuf(be, blk.attn.txt_qnorm), seq_txt * heads, hd, eps);
    try be.qkNorm(ws.k_d, ws.k_d, try normBuf(be, blk.attn.txt_knorm), seq_txt * heads, hd, eps);
    try be.qkNorm(qi, qi, try normBuf(be, blk.attn.qnorm), n_tok * heads, hd, eps);
    try be.qkNorm(ki, ki, try normBuf(be, blk.attn.knorm), n_tok * heads, hd, eps);
    try be.rope(ws.q_d, sess.freqs_d, seq, heads, half, seq * half);
    try be.rope(ws.k_d, sess.freqs_d, seq, heads, half, seq * half);

    trace(be, "  q", block_i, ws.q_d, seq * F);
    trace(be, "  v", block_i, ws.v_d, seq * F);
    // See `v_div`: V is unnormed and outgrows f16 on a real conditioning.
    try be.opScale(ws.v_d, v_div, seq * F);
    if (force_naive_attn) {
        try be.attn(ws.q_d, ws.k_d, ws.v_d, ws.attn_d, seq, seq, heads, heads, hd, attn_scale, false);
    } else {
        try be.opAttnTC(ws.q_d, ws.k_d, ws.v_d, ws.attn_d, seq, heads, heads, hd, attn_scale);
    }
    try be.opScale(ws.attn_d, 1.0 / v_div, seq * F);

    const at_txt = offsetBuf(ws.attn_d, 0, seq_txt * F * 4);
    const at_img = offsetBuf(ws.attn_d, txt_bytes, n_tok * F * 4);
    try lin_cuda.prep(be, plan, at_txt, seq_txt, F, &.{blk.attn.txt_o.w}, false);
    try gemmB(be, plan, ws.dtxt_d, at_txt, seq_txt, blk.attn.txt_o);
    try lin_cuda.prep(be, plan, at_img, n_tok, F, &.{blk.attn.o.w}, false);
    try gemmB(be, plan, ws.dimg_d, at_img, n_tok, blk.attn.o);

    trace(be, "  attn", block_i, ws.attn_d, seq * F);
    try be.gatedAdd(txt_x, ws.dtxt_d, ws.mod_d, seq_txt * F, F, txt_m + 2 * F);
    try be.gatedAdd(img_x, ws.dimg_d, ws.mod_d, n_tok * F, F, img_m + 2 * F);

    // --- MLP half -----------------------------------------------------------
    try be.lnMod(txt_x, ws.ntxt_d, ws.mod_d, seq_txt, F, txt_m + 3 * F, txt_m + 4 * F, eps);
    try be.lnMod(img_x, ws.nimg_d, ws.mod_d, n_tok, F, img_m + 3 * F, img_m + 4 * F, eps);

    try lin_cuda.prep(be, plan, ws.ntxt_d, seq_txt, F, &.{blk.txt_mlp.in.w}, false);
    try gemmB(be, plan, ws.mt_d, ws.ntxt_d, seq_txt, blk.txt_mlp.in);
    try be.gelu(ws.mt_d, seq_txt * mlp_dim);
    try lin_cuda.prep(be, plan, ws.mt_d, seq_txt, mlp_dim, &.{blk.txt_mlp.out.w}, false);
    try gemmB(be, plan, ws.dtxt_d, ws.mt_d, seq_txt, blk.txt_mlp.out);

    try lin_cuda.prep(be, plan, ws.nimg_d, n_tok, F, &.{blk.img_mlp.in.w}, false);
    try gemmB(be, plan, ws.mi_d, ws.nimg_d, n_tok, blk.img_mlp.in);
    try be.gelu(ws.mi_d, n_tok * mlp_dim);
    try lin_cuda.prep(be, plan, ws.mi_d, n_tok, mlp_dim, &.{blk.img_mlp.out.w}, false);
    try gemmB(be, plan, ws.dimg_d, ws.mi_d, n_tok, blk.img_mlp.out);

    try be.gatedAdd(txt_x, ws.dtxt_d, ws.mod_d, seq_txt * F, F, txt_m + 5 * F);
    try be.gatedAdd(img_x, ws.dimg_d, ws.mod_d, n_tok * F, F, img_m + 5 * F);
}

/// Queue a block's streamable weights for async prefetch, ONE BLOCK AHEAD so the
/// upload overlaps the previous block's compute. Keys must be the same host
/// pointers `blockForward` later fetches, or the prefetch is a cache miss and
/// pure waste. The modulation linears are deliberately absent: they never reach
/// the device.
fn prefetchBlock(be: *Backend, blk: *const mageflow.Block) void {
    inline for (.{
        blk.attn.q,     blk.attn.k,      blk.attn.v,     blk.attn.o,
        blk.attn.txt_q, blk.attn.txt_k,  blk.attn.txt_v, blk.attn.txt_o,
        blk.img_mlp.in, blk.img_mlp.out, blk.txt_mlp.in, blk.txt_mlp.out,
    }) |lw| be.prefetchWeight(lw.w.bytes);
}

/// A non-owning device-pointer view at a byte offset, sized to what will be read.
fn offsetBuf(b: Buf, off_bytes: usize, size: usize) Buf {
    return .{ .buf = @enumFromInt(@intFromEnum(b.buf) + off_bytes), .mem = .null_handle, .size = size };
}

/// Wrap a CPU f32 vector as a (pointer-cached) small device buffer.
fn normBuf(be: *Backend, weights: []const f32) !Buf {
    const h = try be.smallBuffer(std.mem.sliceAsBytes(weights));
    return .{ .buf = h, .size = weights.len * 4 };
}
