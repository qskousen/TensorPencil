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
const mageflow_dev = @import("mageflow_dev.zig");
const lin_cuda = @import("lin_cuda.zig");
const cuda = @import("tp_gpu").cuda;
const ops = @import("tp_ops");

const DiT = mageflow.DiT;
const File = @This();
const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;
const Weight = @import("tp_ops").matmul.Weight;

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

/// Run integer weights weight-only, so the device check can split a W8A8 forward's
/// activation noise from everything else.
pub var force_weight_only: bool = false;

/// One block linear plus its bias.
fn gemmB(be: *Backend, plan: lin_cuda.Plan, g: lin_cuda.Group, y: Buf, lw: anytype) !void {
    try lin_cuda.gemm(be, plan, g, y, lw.w, lw.b.?);
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
            .plan = blk: {
                const p = try linPlan(model);
                break :blk if (force_weight_only) lin_cuda.withInt8(p, be, .weight_only) else p;
            },
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
        // The patch embed and output projection run on the fused `opMatmul`. Each
        // stream's GEMMs take its own rows.
        try lin_cuda.attachLora(be, &self.plan, model.lora, model.device_lins, @max(seq_txt, n_tok));

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

    pub fn init(be: *Backend, model: *const DiT, seq_txt: usize, n_tok: usize, act: cuda.backend.Dt) !Workspace {
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
        // The attention block (q/k/v, their text copies, `attn_d`) stays f32
        // (`mageflow_dev.streamDtFor`).
        inline for (.{ "x_d", "ntxt_d", "nimg_d", "dtxt_d", "dimg_d", "mt_d", "mi_d" }) |name| @field(self, name).dt = act;
        return self;
    }

    pub fn deinit(self: *Workspace, be: *Backend) void {
        inline for (fields) |name| be.tensorDestroy(&@field(self, name));
        self.* = undefined;
    }
};

/// The block linears' plan, without a LoRA.
pub fn linPlan(model: *const DiT) !lin_cuda.Plan {
    var p = try lin_cuda.plan(model.device_lins, .{ .blockq = lin_cuda.blockq_gemm }, "mageflow cuda");
    // The MLP down projection reads past f16 at sigma 1.
    p.act16 = .bf16;
    return p;
}

/// Whether the CUDA arms can run every block linear this model has. The refusal,
/// if any, is logged by name; the caller says the trunk then runs on the CPU.
pub fn supported(model: *const DiT) bool {
    if (model.blocks.len == 0) return false;
    _ = linPlan(model) catch return false;
    if (!lin_cuda.loraRunnable(model.lora, model.device_lins, "mageflow cuda")) return false;
    return true;
}

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
    be.downloadF32(buf, host) catch return;
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
    try mageflow_dev.body(Dev{ .model = model, .be = be, .sess = sess, .ws = ws }, model, cancel);
    try be.endBatch();
    // A 16-bit stream that overflowed can leave the output finite (`allFinite`).
    if (ws.x_d.dt != .f32 and !try be.allFinite(ws.x_d, sess.seq * F)) return error.NonFiniteActivation;

    const final = try gpa.alloc(f32, sess.n_img * channels);
    defer gpa.free(final);
    try be.tensorDownload(ws.out_d.slice(0, final.len * 4), std.mem.sliceAsBytes(final));
    mageflow.deinterleaveChannels(out, final, sess.n_img);
}

/// `mageflow_dev`'s ops on this backend: ONE joint stream, the conditioning at row 0 and
/// the image tokens after it, so the image projections write straight into the joint
/// q/k/v and only the text ones are copied in (see the header's note 2).
const Dev = struct {
    model: *const DiT,
    be: *Backend,
    sess: *Session,
    ws: *Workspace,

    /// The text rows of `b`, in bytes at its own element width.
    fn txtBytes(d: Dev, b: Buf) usize {
        return d.sess.seq_txt * F * b.dt.bytes();
    }

    /// `n` rows of `b` past its text rows.
    fn rows(d: Dev, b: Buf, n: usize) Buf {
        return b.slice(d.txtBytes(b), n * F * b.dt.bytes());
    }

    pub fn streams(d: Dev) Streams {
        const ws = d.ws;
        const n = d.sess.n_tok;
        return .{
            .seq_txt = d.sess.seq_txt,
            .n_tok = n,
            .txt_x = ws.x_d.slice(0, d.txtBytes(ws.x_d)),
            .img_x = d.rows(ws.x_d, n),
            .tq = ws.qt_d,
            .tk = ws.kt_d,
            .tv = ws.vt_d,
            .iq = d.rows(ws.q_d, n),
            .ik = d.rows(ws.k_d, n),
            .iv = d.rows(ws.v_d, n),
        };
    }
    const Streams = struct { seq_txt: usize, n_tok: usize, txt_x: Buf, img_x: Buf, tq: Buf, tk: Buf, tv: Buf, iq: Buf, ik: Buf, iv: Buf };

    pub fn embed(d: Dev) !void {
        const n = d.sess.n_tok;
        if (d.ws.x_d.dt == .f32)
            try d.be.tensorCopy(d.ws.x_d, 0, d.sess.txt_d, 0, d.txtBytes(d.ws.x_d))
        else
            try d.be.opCast(d.sess.txt_d, d.ws.x_d, d.sess.seq_txt * F, 0);
        try lin_cuda.linear(d.be, d.sess.plan, d.rows(d.ws.x_d, n), d.ws.imgin_d, n, d.model.img_in.w, d.model.img_in.b);
        File.trace(d.be, "txt_in", 0, d.ws.x_d, d.sess.seq_txt * F);
        File.trace(d.be, "img_in", 0, d.rows(d.ws.x_d, n), n * F);
    }

    /// Prefetch one block ahead, so each upload overlaps the previous block's compute.
    pub fn blockBegin(d: Dev, i: usize) !void {
        if (!d.be.async_uploads) return;
        if (i == 0) prefetchBlock(d.be, &d.model.blocks[0]);
        if (i + 1 < d.model.blocks.len) prefetchBlock(d.be, &d.model.blocks[i + 1]);
    }

    pub fn trace(d: Dev, what: []const u8, i: usize, buf: Buf, elems: usize) void {
        File.trace(d.be, what, i, buf, elems);
    }

    pub fn traceBlock(d: Dev, i: usize) void {
        File.trace(d.be, "block", i, d.ws.x_d, d.sess.seq * F);
    }

    pub fn lnMod(d: Dev, src: Buf, dst: Buf, premul: usize, shift: usize, n: usize) !void {
        try d.be.lnMod(src, dst, d.ws.mod_d, n, F, premul, shift, eps);
    }

    /// Bias-carrying linears sharing one input, under one prep.
    pub fn gemms(d: Dev, _: mageflow_dev.Stream, x: Buf, m: usize, lws: []const mageflow.LinearW, ys: []const Buf) !void {
        var w: [3]Weight = undefined;
        for (lws, 0..) |lw, k| w[k] = lw.w;
        const g = try lin_cuda.prep(d.be, d.sess.plan, x, m, F, w[0..lws.len]);
        for (lws, ys) |lw, y| try gemmB(d.be, d.sess.plan, g, y, lw);
    }

    pub fn linear(d: Dev, _: mageflow_dev.Stream, y: Buf, x: Buf, m: usize, lw: mageflow.LinearW) !void {
        const g = try lin_cuda.prep(d.be, d.sess.plan, x, m, lw.w.cols, &.{lw.w});
        try gemmB(d.be, d.sess.plan, g, y, lw);
    }

    /// The text q/k/v copied into the joint operands, per-head QK-norm with each stream's
    /// own weight, then RoPE over the whole joint sequence: the text rows sit at position
    /// 0, whose rotation is the identity, so rotating them there is the same as not.
    pub fn qkPrepare(d: Dev, blk: *const mageflow.Block) !void {
        const be = d.be;
        const ws = d.ws;
        const n = d.sess.n_tok;
        const seq_txt = d.sess.seq_txt;
        const seq = d.sess.seq;
        try be.tensorCopy(ws.q_d, 0, ws.qt_d, 0, d.txtBytes(ws.q_d));
        try be.tensorCopy(ws.k_d, 0, ws.kt_d, 0, d.txtBytes(ws.k_d));
        try be.tensorCopy(ws.v_d, 0, ws.vt_d, 0, d.txtBytes(ws.v_d));
        const qi = d.rows(ws.q_d, n);
        const ki = d.rows(ws.k_d, n);
        try be.qkNorm(ws.q_d, ws.q_d, try normBuf(be, blk.attn.txt_qnorm), seq_txt * heads, hd, eps);
        try be.qkNorm(ws.k_d, ws.k_d, try normBuf(be, blk.attn.txt_knorm), seq_txt * heads, hd, eps);
        try be.qkNorm(qi, qi, try normBuf(be, blk.attn.qnorm), n * heads, hd, eps);
        try be.qkNorm(ki, ki, try normBuf(be, blk.attn.knorm), n * heads, hd, eps);
        try be.rope(ws.q_d, d.sess.freqs_d, seq, heads, half, seq * half);
        try be.rope(ws.k_d, d.sess.freqs_d, seq, heads, half, seq * half);
    }

    pub fn scale(d: Dev, buf: Buf, n: usize, f: f32) !void {
        try d.be.opScale(buf, f, n);
    }

    pub fn attention(d: Dev, seq: usize) !void {
        const ws = d.ws;
        if (force_naive_attn) return d.be.attn(ws.q_d, ws.k_d, ws.v_d, ws.attn_d, seq, seq, heads, heads, hd, attn_scale, false);
        try d.be.opAttnTC(ws.q_d, ws.k_d, ws.v_d, ws.attn_d, seq, heads, heads, hd, attn_scale);
    }

    pub fn attnSplit(d: Dev) !struct { txt: Buf, img: Buf } {
        return .{ .txt = d.ws.attn_d.slice(0, d.txtBytes(d.ws.attn_d)), .img = d.rows(d.ws.attn_d, d.sess.n_tok) };
    }

    pub fn gatedAdd(d: Dev, x: Buf, t: Buf, n: usize, off: usize) !void {
        try d.be.gatedAdd(x, t, d.ws.mod_d, n * F, F, off);
    }

    pub fn gelu(d: Dev, buf: Buf, n: usize) !void {
        try d.be.gelu(buf, n);
    }

    pub fn final(d: Dev, fin: usize) !void {
        const canvas = d.rows(d.ws.x_d, d.sess.n_img);
        try d.be.lnMod(canvas, canvas, d.ws.mod_d, d.sess.n_img, F, fin + 0 * F, fin + 1 * F, eps);
        try lin_cuda.linear(d.be, d.sess.plan, d.ws.out_d, canvas, d.sess.n_img, d.model.proj_out.w, d.model.proj_out.b);
    }
};

/// Queue a block's streamable weights for async prefetch, ONE BLOCK AHEAD so the
/// upload overlaps the previous block's compute. Keys must be the same host
/// pointers the forward later fetches, or the prefetch is a cache miss and
/// pure waste. The modulation linears are deliberately absent: they never reach
/// the device.
fn prefetchBlock(be: *Backend, blk: *const mageflow.Block) void {
    inline for (.{
        blk.attn.q,     blk.attn.k,      blk.attn.v,     blk.attn.o,
        blk.attn.txt_q, blk.attn.txt_k,  blk.attn.txt_v, blk.attn.txt_o,
        blk.img_mlp.in, blk.img_mlp.out, blk.txt_mlp.in, blk.txt_mlp.out,
    }) |lw| be.prefetchWeight(lw.w.bytes);
}

/// Wrap a CPU f32 vector as a (pointer-cached) small device buffer.
fn normBuf(be: *Backend, weights: []const f32) !Buf {
    const h = try be.smallBuffer(std.mem.sliceAsBytes(weights));
    return .{ .buf = h, .size = weights.len * 4 };
}
