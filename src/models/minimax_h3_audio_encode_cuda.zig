//! CUDA-backend MiniMax H3 audio VAE ENCODE, the device twin of
//! `minimax_h3_audio_encode.encode`.
//!
//! The host encoder's first residual stage runs 64 channels over the FULL sample
//! count, which is why it needs a banded im2col and per-stage arenas; on the
//! device the same work is a handful of GEMMs and the whole thing is transient.
//!
//! It follows `minimax_h3_audio_cuda`'s conventions rather than inventing its own:
//!
//! 1. **Signals are CHANNEL-LAST `[len][ch]`**, so a warp covers consecutive
//!    channels at one time step and a GEMM's output is already the next op's input.
//! 2. **Conv weights are permuted at session build**, `[out_ch][in_ch][k]` ->
//!    `[out_ch][k][in_ch]`, matching the `(tap, in_ch)` column order channel-last
//!    im2col produces. Reading them unpermuted pairs a tap with the wrong channel.
//! 3. **The GEMMs run in f32**, for the reason the decode side measured: audio is
//!    unforgiving of f16 and this stage is not the bottleneck.
//!
//! What differs from the decode side, and each is a place to get it wrong:
//!
//! - ⚠️ **These convs are STRIDED.** The vocoder's are all stride 1 and
//!   same-padded, so its `conv` helper hardcodes `out_len = in_len`; here the
//!   downsampling conv per stage shortens by its stride and the output length has
//!   to come from the shape.
//! - ⚠️ **`Snake1d`, not `SnakeBeta`**: alpha is stored LINEAR and does double duty
//!   as beta, so it is passed through as stored rather than exponentiated.
//! - The posterior head is a transformer block, not a conv: LayerNorms, three
//!   row-sliced qkv GEMMs, CAUSAL attention, then the mean over heads pooled along
//!   the FEATURE axis. `opMeanHeadsPool` is that last pair fused.
//! - ⚠️ **The fused qkv is three separate GEMMs**, not one plus a de-interleave:
//!   `attn` wants each of q/k/v contiguous with a `heads * hd` row stride, and one
//!   GEMM would give a `3 * heads * hd` stride. The k third's bias is zeros.

const std = @import("std");
const enc_mod = @import("minimax_h3_audio_encode.zig");
const audio = @import("minimax_h3_audio.zig");
const cuda = @import("tp_gpu").cuda;

const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;
const AudioEncoder = enc_mod.AudioEncoder;
const Conv1d = audio.Conv1d;

/// LayerNorm epsilon, matching the host encoder's.
const ln_eps: f32 = 1e-5;

/// The im2col band budget in f32 elements, 64 MB. The first residual stage's patch
/// matrix is `out_len * k * in_ch`, i.e. 57 MB per SECOND of audio at the real
/// widths, so an unbanded buffer grows without bound with the reference's length.
pub var patch_band: usize = 1 << 24;

/// Whether this encoder's shapes are ones the kernels here cover.
///
/// Refused by name rather than at a launch: a mismatch several stages deep surfaces
/// as a bad device read or as quiet noise in the latent.
pub fn supported(enc: *const AudioEncoder) bool {
    if (enc.stages.len == 0) return false;
    if (enc.conv_in.in_ch != 1) return false;
    if (enc.head.in_dim % enc.head.heads != 0) return false;
    // The pooling only reduces, never expands.
    if (enc.head.headDim() < enc.head.out_dim) return false;
    for (enc.stages) |st| {
        if (st.units.len == 0) return false;
        if (st.down.k != 2 * st.down.stride) return false;
        for (st.units) |u| {
            if (u.c1.in_ch != u.c1.out_ch or u.c2.in_ch != u.c2.out_ch) return false;
            if (u.c2.k != 1) return false;
        }
    }
    return true;
}

// --- session --------------------------------------------------------------

/// A conv ready for the device. Shares `minimax_h3_audio_cuda`'s layout choice
/// but keeps its own f32-only weight: nothing here runs the f16 arm.
const DevConv = struct {
    /// `[out_ch][k * in_ch]` f32.
    w: []const f32,
    bias: ?[]const f32,
    out_ch: usize,
    in_ch: usize,
    k: usize,
    dilation: usize,
    padding: usize,
    stride: usize,

    fn plen(self: DevConv) usize {
        return self.k * self.in_ch;
    }

    fn outLen(self: DevConv, in_len: usize) usize {
        const eff = self.dilation * (self.k - 1) + 1;
        const padded = in_len + 2 * self.padding;
        if (padded < eff) return 0;
        return (padded - eff) / self.stride + 1;
    }
};

const DevResUnit = struct {
    a1: []const f32,
    c1: DevConv,
    a2: []const f32,
    c2: DevConv,
};

const DevStage = struct {
    units: []DevResUnit,
    act: []const f32,
    down: DevConv,
};

/// A plain `nn.Linear`, `[out][in]` f32 -- already the GEMM's B, so unlike a conv
/// it needs no permutation.
const DevLin = struct {
    w: []const f32,
    bias: ?[]const f32,
    out: usize,
    in: usize,
};

const DevHead = struct {
    heads: usize,
    in_dim: usize,
    out_dim: usize,
    hidden: usize,
    norm1_w: []const f32,
    norm1_b: []const f32,
    norm2_w: []const f32,
    norm2_b: []const f32,
    norm3_w: []const f32,
    norm3_b: []const f32,
    mlp_norm_w: []const f32,
    mlp_norm_b: []const f32,
    proj: DevLin,
    /// The three thirds of the fused qkv, row-sliced at build so the forward does
    /// not re-slice per call. Each carries its own third of the concatenated bias
    /// (the k third being zeros).
    q: DevLin,
    k: DevLin,
    v: DevLin,
    attn_proj: DevLin,
    w0: DevLin,
    w1: DevLin,
    w2: DevLin,
};

pub const Session = struct {
    arena: std.heap.ArenaAllocator,
    conv_in: DevConv,
    stages: []DevStage,
    act_out: []const f32,
    conv_out: DevConv,
    head: DevHead,
    mean_proj: DevConv,

    pub fn deinit(self: *Session) void {
        self.arena.deinit();
        self.* = undefined;
    }

    pub fn init(gpa: std.mem.Allocator, enc: *const AudioEncoder) !Session {
        if (!supported(enc)) return error.UnsupportedCheckpoint;
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();

        const stages = try a.alloc(DevStage, enc.stages.len);
        for (enc.stages, stages) |*src, *dst| {
            dst.units = try a.alloc(DevResUnit, src.units.len);
            for (src.units, dst.units) |*u, *du| {
                du.* = .{
                    .a1 = u.a1,
                    .c1 = try devConv(a, u.c1),
                    .a2 = u.a2,
                    .c2 = try devConv(a, u.c2),
                };
            }
            dst.act = src.act;
            dst.down = try devConv(a, src.down);
        }

        const h = &enc.head;
        const din = h.in_dim;
        return .{
            .arena = arena,
            .conv_in = try devConv(a, enc.conv_in),
            .stages = stages,
            .act_out = enc.act_out,
            .conv_out = try devConv(a, enc.conv_out),
            .head = .{
                .heads = h.heads,
                .in_dim = din,
                .out_dim = h.out_dim,
                .hidden = h.w0.out,
                .norm1_w = h.norm1.w,
                .norm1_b = h.norm1.b,
                .norm2_w = h.norm2.w,
                .norm2_b = h.norm2.b,
                .norm3_w = h.norm3.w,
                .norm3_b = h.norm3.b,
                .mlp_norm_w = h.mlp_norm.w,
                .mlp_norm_b = h.mlp_norm.b,
                .proj = lin(h.proj),
                .q = .{ .w = h.qkv.w[0 .. din * din], .bias = h.qkv_bias[0..din], .out = din, .in = din },
                .k = .{ .w = h.qkv.w[din * din ..][0 .. din * din], .bias = h.qkv_bias[din..][0..din], .out = din, .in = din },
                .v = .{ .w = h.qkv.w[2 * din * din ..][0 .. din * din], .bias = h.qkv_bias[2 * din ..][0..din], .out = din, .in = din },
                .attn_proj = lin(h.attn_proj),
                .w0 = lin(h.w0),
                .w1 = lin(h.w1),
                .w2 = lin(h.w2),
            },
            .mean_proj = try devConv(a, enc.mean_proj),
        };
    }
};

fn lin(l: enc_mod.Lin) DevLin {
    return .{ .w = l.w, .bias = l.b, .out = l.out, .in = l.in };
}

/// `[out_ch][in_ch][k]` -> `[out_ch][k][in_ch]`, the column order channel-last
/// im2col produces.
fn devConv(a: std.mem.Allocator, c: Conv1d) !DevConv {
    std.debug.assert(c.groups == 1);
    const n = c.out_ch * c.in_ch * c.k;
    std.debug.assert(c.w.len == n);
    const w = try a.alloc(f32, n);
    for (0..c.out_ch) |oc| {
        for (0..c.in_ch) |ic| {
            for (0..c.k) |j| {
                w[(oc * c.k + j) * c.in_ch + ic] = c.w[(oc * c.in_ch + ic) * c.k + j];
            }
        }
    }
    return .{
        .w = w,
        .bias = c.b,
        .out_ch = c.out_ch,
        .in_ch = c.in_ch,
        .k = c.k,
        .dilation = c.dilation,
        .padding = c.padding,
        .stride = c.stride,
    };
}

// --- workspace ------------------------------------------------------------

/// The widths and lengths the pipeline actually reaches, walked rather than
/// assumed: the lengths divide by the stage strides and the widths double, so the
/// peak is not at either end.
pub const Shapes = struct {
    /// Largest `ch * len` any signal buffer holds.
    sig: usize,
    /// Largest banded im2col patch, in elements.
    patch: usize,
    /// Latent frames, i.e. the trunk's output length.
    t: usize,
    /// Largest `rows * width` the posterior head needs.
    head: usize,

    pub fn of(enc: *const AudioEncoder, samples: usize) Shapes {
        const t = enc.latentFrames(samples);
        var s: Shapes = .{ .sig = 0, .patch = 0, .t = t, .head = 0 };
        var len = t * enc.hop();
        var ch: usize = 1;
        s.sig = @max(s.sig, ch * len);
        s.patch = @max(s.patch, patchFor(enc.conv_in, len));
        ch = enc.conv_in.out_ch;
        len = enc.conv_in.outLen(len);
        s.sig = @max(s.sig, ch * len);
        for (enc.stages) |st| {
            for (st.units) |u| {
                s.patch = @max(s.patch, patchFor(u.c1, len));
                s.patch = @max(s.patch, patchFor(u.c2, len));
            }
            s.patch = @max(s.patch, patchFor(st.down, len));
            len = st.down.outLen(len);
            ch = st.down.out_ch;
            s.sig = @max(s.sig, ch * len);
        }
        s.patch = @max(s.patch, patchFor(enc.conv_out, len));
        ch = enc.conv_out.out_ch;
        s.sig = @max(s.sig, ch * len);
        // The head: the widest row buffer is the qkv planes and the MLP hidden.
        s.head = @max(t * enc.head.in_dim, t * enc.head.w0.out);
        s.patch = @max(s.patch, patchFor(enc.mean_proj, len));
        return s;
    }

    fn patchFor(c: Conv1d, in_len: usize) usize {
        const cols = c.k * c.in_ch;
        const rows = @min(c.outLen(in_len), @max(1, patch_band / @max(1, cols)));
        return rows * cols;
    }

    /// Device bytes `Workspace.init` will reserve, so a caller can refuse first.
    pub fn deviceBytes(self: Shapes) usize {
        return (4 * self.sig + self.patch + 5 * self.head) * @sizeOf(f32);
    }
};

pub const Workspace = struct {
    /// The signal ping-pong pair, the residual unit's saved input, and its
    /// intermediate. Four rather than three because a BANDED conv cannot have its
    /// destination alias its source (see `conv`), and the residual unit needs a
    /// scratch that is neither of the ping-pong pair.
    s0: Buf = .{},
    s1: Buf = .{},
    res: Buf = .{},
    tmp: Buf = .{},
    patch: Buf = .{},
    /// The head's rows: the normalized input, q/k/v (one buffer each) and a
    /// scratch that serves the attention output, the pooled rows and the MLP.
    hn: Buf = .{},
    hq: Buf = .{},
    hk: Buf = .{},
    hv: Buf = .{},
    ht: Buf = .{},
    /// The accumulating `[t][out_dim]` residual stream of the head.
    hy: Buf = .{},
    shapes: Shapes = undefined,

    pub fn init(be: *Backend, enc: *const AudioEncoder, samples: usize) !Workspace {
        var ws: Workspace = .{ .shapes = Shapes.of(enc, samples) };
        errdefer ws.deinit(be);
        const s = ws.shapes;
        ws.s0 = try be.tensorCreate(s.sig * 4);
        ws.s1 = try be.tensorCreate(s.sig * 4);
        ws.res = try be.tensorCreate(s.sig * 4);
        ws.tmp = try be.tensorCreate(s.sig * 4);
        ws.patch = try be.tensorCreate(s.patch * 4);
        ws.hn = try be.tensorCreate(s.head * 4);
        ws.hq = try be.tensorCreate(s.head * 4);
        ws.hk = try be.tensorCreate(s.head * 4);
        ws.hv = try be.tensorCreate(s.head * 4);
        ws.ht = try be.tensorCreate(s.head * 4);
        ws.hy = try be.tensorCreate(s.t * @max(enc.head.out_dim, 1) * 4);
        return ws;
    }

    pub fn deinit(self: *Workspace, be: *Backend) void {
        inline for (.{ &self.s0, &self.s1, &self.res, &self.tmp, &self.patch, &self.hn, &self.hq, &self.hk, &self.hv, &self.ht, &self.hy }) |b| {
            if (b.size > 0) be.tensorDestroy(b);
        }
        self.* = undefined;
    }
};

// --- forward --------------------------------------------------------------

fn devBuf(be: *Backend, data: []const f32) !Buf {
    return .{ .buf = try be.smallBuffer(std.mem.sliceAsBytes(data)), .mem = .null_handle, .size = data.len * 4 };
}

/// `dst[out_len][out_ch] = conv(src[in_len][in_ch])`, banded im2col + GEMM.
///
/// ⚠️ **`dst` must NOT alias `src`.** The decode side's unbanded conv tolerates it
/// (its im2col copies the whole input into the patch matrix before the GEMM runs),
/// but banding interleaves the two: band 0's GEMM writes `dst` before band 1's
/// im2col reads `src`. Aliasing them is then correct for any clip short enough to
/// fit one band and wrong for every longer one, which is exactly how it got here.
fn conv(be: *Backend, ws: *Workspace, dst: Buf, src: Buf, c: DevConv, in_len: usize) !usize {
    std.debug.assert(dst.buf != src.buf);
    const out_len = c.outLen(in_len);
    const cols = c.plen();
    const band = @max(1, @min(out_len, patch_band / cols));
    std.debug.assert(band * cols <= ws.shapes.patch);
    var t0: usize = 0;
    while (t0 < out_len) : (t0 += band) {
        const nb = @min(band, out_len - t0);
        try be.opIm2col1d(src, ws.patch, t0, nb, c.k, c.in_ch, in_len, c.dilation, c.padding, c.stride);
        const out = offsetBuf(dst, t0 * c.out_ch * 4);
        be.opMatmulF32Lt(out, ws.patch, nb, std.mem.sliceAsBytes(c.w), c.out_ch, cols, c.bias) catch |err| switch (err) {
            // The hand-PTX arm has no tiled f32 GEMM. Its one-thread-per-output
            // fallback is correct at about a third of the speed; falling back to
            // f16 instead would be a silent 50 dB of SNR, which is the measurement
            // the decode side already paid for.
            error.UnsupportedKernelArm => try be.opMatmul(out, 0, ws.patch, 0, nb, std.mem.sliceAsBytes(c.w), false, c.out_ch, cols, 1.0, c.bias),
            else => return err,
        };
    }
    return out_len;
}

/// A byte offset into a device buffer as its own `Buf`. ⚠️ Only valid on CUDA,
/// where `buf` is a device pointer; on Vulkan it is an opaque handle and this does
/// not port.
fn offsetBuf(b: Buf, off_bytes: usize) Buf {
    return .{ .buf = @enumFromInt(@intFromEnum(b.buf) + off_bytes), .mem = .null_handle, .size = b.size - off_bytes };
}

fn gemm(be: *Backend, dst: Buf, src: Buf, rows: usize, l: DevLin) !void {
    be.opMatmulF32Lt(dst, src, rows, std.mem.sliceAsBytes(l.w), l.out, l.in, l.bias) catch |err| switch (err) {
        error.UnsupportedKernelArm => try be.opMatmul(dst, 0, src, 0, rows, std.mem.sliceAsBytes(l.w), false, l.out, l.in, 1.0, l.bias),
        else => return err,
    };
}

/// One `ResidualUnit`, in place on `x`: Snake -> dilated conv -> Snake -> pointwise
/// conv, plus the residual. Uses `ws.res` to hold the unit's input.
fn resUnit(be: *Backend, ws: *Workspace, x: Buf, u: DevResUnit, len: usize) !void {
    const ch = u.c1.in_ch;
    const alpha1 = try devBuf(be, u.a1);
    const alpha2 = try devBuf(be, u.a2);
    try be.opCopyOff(ws.res, 0, x, 0, len * ch, false);
    try be.opSnake1dCa(x, alpha1, len, ch);
    _ = try conv(be, ws, ws.tmp, x, u.c1, len);
    try be.opSnake1dCa(ws.tmp, alpha2, len, ch);
    _ = try conv(be, ws, x, ws.tmp, u.c2, len);
    // The reference center-crops the residual when the convs shortened the signal;
    // with its padding they never do, and `supported` refuses a checkpoint where a
    // unit is not length-preserving.
    try be.opAdd(x, ws.res, len * ch);
}

/// The `AttnProjection` posterior head over `[t][in_dim]` rows in `ws.s0`,
/// leaving `[t][out_dim]` in `ws.hy`.
fn posterior(be: *Backend, ws: *Workspace, h: DevHead, x: Buf, t: usize) !void {
    const din = h.in_dim;
    const dout = h.out_dim;
    const hd = din / h.heads;

    // x = proj(norm3(x)) + attn(norm1(x)); norm3 feeds the PROJECTION.
    try be.opLayerNorm(x, ws.hn, h.norm3_w, h.norm3_b, t, din, ln_eps, false);
    try gemm(be, ws.hy, ws.hn, t, h.proj);

    try be.opLayerNorm(x, ws.hn, h.norm1_w, h.norm1_b, t, din, ln_eps, false);
    try gemm(be, ws.hq, ws.hn, t, h.q);
    try gemm(be, ws.hk, ws.hn, t, h.k);
    try gemm(be, ws.hv, ws.hn, t, h.v);
    // CAUSAL, and MHA: kv_heads == n_heads.
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
    try be.attn(ws.hq, ws.hk, ws.hv, ws.ht, t, t, h.heads, h.heads, hd, scale, true);
    // ...mean over heads, then pooled along the FEATURE axis to the latent width.
    try be.opMeanHeadsPool(ws.ht, ws.hn, t, h.heads, hd, dout);
    try gemm(be, ws.ht, ws.hn, t, h.attn_proj);
    try be.opAdd(ws.hy, ws.ht, t * dout);

    // x += mlp(norm2(x)), and the MLP carries its own LayerNorm inside.
    try be.opLayerNorm(ws.hy, ws.hn, h.norm2_w, h.norm2_b, t, dout, ln_eps, false);
    try be.opLayerNorm(ws.hn, ws.hn, h.mlp_norm_w, h.mlp_norm_b, t, dout, ln_eps, false);
    try gemm(be, ws.hq, ws.hn, t, h.w0);
    try gemm(be, ws.hk, ws.hn, t, h.w1);
    // GeGLU with the TANH gelu, gate then value.
    try be.geluMul(ws.hq, ws.hk, t * h.hidden);
    try gemm(be, ws.ht, ws.hq, t, h.w2);
    try be.opAdd(ws.hy, ws.ht, t * dout);
}

/// Encode an interleaved stereo waveform to normalized latents, planar
/// `[c_lat][2][t]` -- the same contract as the host `encode`.
pub fn encode(
    enc: *const AudioEncoder,
    sess: *const Session,
    be: *Backend,
    ws: *Workspace,
    gpa: std.mem.Allocator,
    out: []f32,
    wav: []const f32,
    len: usize,
    channels: usize,
) !void {
    const c_lat = enc.latentChannels();
    const t = ws.shapes.t;
    std.debug.assert(t == enc.latentFrames(len));
    const padded = t * enc.hop();
    std.debug.assert(channels == 1 or channels == audio.stereo);
    std.debug.assert(wav.len == len * channels);
    std.debug.assert(out.len == c_lat * audio.stereo * t);

    const mono = try gpa.alloc(f32, padded);
    defer gpa.free(mono);
    const rows = try gpa.alloc(f32, t * c_lat);
    defer gpa.free(rows);

    for (0..audio.stereo) |s| {
        // Right-padded with zeros to a multiple of the hop; one channel, so
        // channel-last and planar are the same thing here.
        @memset(mono[len..], 0);
        const src_ch = if (channels == 1) 0 else s;
        for (0..len) |i| mono[i] = wav[i * channels + src_ch];

        try be.beginBatch();
        errdefer if (be.batching()) be.abortBatch();
        try be.tensorUpload(ws.s0, std.mem.sliceAsBytes(mono));

        var cur = ws.s0;
        var other = ws.s1;
        var cur_len = try conv(be, ws, other, cur, sess.conv_in, padded);
        std.mem.swap(Buf, &cur, &other);

        for (sess.stages) |st| {
            const w = st.down.in_ch;
            for (st.units) |u| try resUnit(be, ws, cur, u, cur_len);
            const alpha = try devBuf(be, st.act);
            try be.opSnake1dCa(cur, alpha, cur_len, w);
            // The strided conv writes the OTHER half of the ping-pong pair, so it
            // never aliases its source; `res` and `tmp` belong to `resUnit`.
            cur_len = try conv(be, ws, other, cur, st.down, cur_len);
            std.mem.swap(Buf, &cur, &other);
        }

        const alpha_out = try devBuf(be, sess.act_out);
        try be.opSnake1dCa(cur, alpha_out, cur_len, sess.conv_out.in_ch);
        cur_len = try conv(be, ws, other, cur, sess.conv_out, cur_len);
        std.mem.swap(Buf, &cur, &other);
        std.debug.assert(cur_len == t);

        try posterior(be, ws, sess.head, cur, t);
        // `mean_proj` is a 1x1 conv over `[t][out_dim]`, i.e. a plain GEMM.
        _ = try conv(be, ws, ws.ht, ws.hy, sess.mean_proj, t);
        try be.endBatch();
        try be.tensorDownload(ws.ht, std.mem.sliceAsBytes(rows));

        for (0..c_lat) |c| {
            const inv = 1.0 / enc.latents_std[c];
            const mean = enc.latents_mean[c];
            const dst = out[(c * audio.stereo + s) * t ..][0..t];
            for (dst, 0..) |*o, i| o.* = (rows[i * c_lat + c] - mean) * inv;
        }
    }
}
