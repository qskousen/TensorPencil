//! CUDA-backend MiniMax H3 video VAE ENCODE, the device twin of
//! `minimax_h3_vae_encode.encodeMoments`.
//!
//! This was the last CPU stage on the reference path and the most expensive one on
//! both axes: a 256x256 clip chunk peaks at 4.35 GB of HOST memory and takes tens
//! of seconds, and it is what stopped a 256px continuation fitting a 14 GB bound.
//!
//! **Only `encodeMoments` moves.** The temporal clip chunking and the spatial
//! tiling above it are intricate, already pinned against the reference, and pure
//! bookkeeping; duplicating them per backend is how the two would drift. The CPU
//! module takes a `Moments` hook and this supplies it, so `encode` runs one code
//! path with either compute underneath.
//!
//! Conventions, and where they differ from the audio port next door:
//!
//! 1. **Volumes are CHANNEL-LAST `[t][h][w][ch]`**, not the reference's planar
//!    `[ch][t][h][w]`. Same reason as the audio port -- a warp covers consecutive
//!    channels at one position -- but here it buys something extra: `opGroupNorm`
//!    is already channel-last and already fuses the SiLU that follows every norm
//!    in this encoder, so the norms need no kernel of their own.
//! 2. **Conv weights are permuted at session build**, `[out][in][kt][kh][kw]` ->
//!    `[out][kt][kh][kw][in]`, matching the column order channel-last im2col
//!    produces.
//! 3. ⚠️ **GroupNorm statistics are PER FRAME.** `opGroupNorm` reduces over all the
//!    positions it is given, so it is called once per frame over `h * w` rather
//!    than once over the volume. One call over everything is a plausible image and
//!    the wrong one, and a single-frame encode cannot tell the two apart.
//! 4. `Downsample3D`'s asymmetric `(0, 1, 0, 1)` reflect pre-pad is folded into the
//!    im2col's per-axis low pad rather than materialized.
//!
//! ⚠️ **The GEMMs run in f32.** The vendor-library arm has a tiled f32 GEMM; the
//! hand-PTX arm does not, and falls back to a correct one-thread-per-output kernel
//! that is roughly a third the speed. Dropping to f16 there instead would put
//! noise into a latent that then rides through every sampling step.

const std = @import("std");
const enc_mod = @import("minimax_h3_vae_encode.zig");
const vae = @import("minimax_h3_vae.zig");
const cuda = @import("tp_gpu").cuda;

const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;
const VideoEncoder = enc_mod.VideoEncoder;
const Conv3d = enc_mod.Conv3d;
const Vol = enc_mod.Vol;

/// The im2col band budget in f32 elements, 64 MB. A 3x3x3 conv duplicates each
/// input 27 times, so an unbanded patch matrix for one level-0 convolution at
/// 1344x768 is 4.7 GB whichever side of the bus it lives on.
pub var patch_band: usize = 1 << 24;

/// GroupNorm statistic chunks, matching what the SD VAE port uses.
const gn_chunks: usize = 64;

/// Whether this encoder's shapes are ones the kernels here cover.
pub fn supported(enc: *const VideoEncoder) bool {
    if (enc.cfg.norm_groups == 0) return false;
    if (enc.conv_in.in_ch != 3) return false;
    for (enc.levels) |lv| {
        for (lv.blocks) |rb| {
            // Every norm here is a GroupNorm over the block's own width, so the
            // width has to divide by the group count at every level.
            if (rb.conv1.in_ch % enc.cfg.norm_groups != 0) return false;
            if (rb.conv2.in_ch % enc.cfg.norm_groups != 0) return false;
        }
    }
    if (enc.conv_out.in_ch % enc.cfg.norm_groups != 0) return false;
    return true;
}

// --- session --------------------------------------------------------------

/// A conv ready for the device: `[out_ch][kt * kh * kw * in_ch]` f32.
const DevConv = struct {
    w: []const f32,
    bias: ?[]const f32,
    out_ch: usize,
    in_ch: usize,
    kt: usize,
    kh: usize,
    kw: usize,
    stride_t: usize,
    stride_s: usize,
    pad_t: usize,

    fn cols(self: DevConv) usize {
        return self.kt * self.kh * self.kw * self.in_ch;
    }
};

const DevResBlock = struct {
    norm1: []const f32,
    norm2: []const f32,
    conv1: DevConv,
    conv2: DevConv,
    shortcut: ?DevConv,
};

const DevLevel = struct {
    blocks: []DevResBlock,
    down: ?DevConv,
    space_stride: usize,
};

pub const Session = struct {
    arena: std.heap.ArenaAllocator,
    conv_in: DevConv,
    levels: []DevLevel,
    norm_out: []const f32,
    conv_out: DevConv,
    quant: DevConv,
    groups: usize,
    eps: f32,

    pub fn deinit(self: *Session) void {
        self.arena.deinit();
        self.* = undefined;
    }

    pub fn init(gpa: std.mem.Allocator, enc: *const VideoEncoder) !Session {
        if (!supported(enc)) return error.UnsupportedCheckpoint;
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();
        const cfg = enc.cfg;

        const levels = try a.alloc(DevLevel, enc.levels.len);
        for (enc.levels, levels, 0..) |*src, *dst, i| {
            dst.blocks = try a.alloc(DevResBlock, src.blocks.len);
            for (src.blocks, dst.blocks) |*rb, *drb| {
                drb.* = .{
                    // The checkpoint's weight ++ bias concatenated, which is the
                    // one buffer `gn_apply` reads both out of.
                    .norm1 = try catNorm(a, rb.norm1_w, rb.norm1_b),
                    .norm2 = try catNorm(a, rb.norm2_w, rb.norm2_b),
                    .conv1 = try devConv(a, rb.conv1),
                    .conv2 = try devConv(a, rb.conv2),
                    .shortcut = if (rb.shortcut) |sc| try devConv(a, sc) else null,
                };
            }
            dst.down = if (src.down) |d| try devConv(a, d) else null;
            dst.space_stride = cfg.space_down[i];
        }

        return .{
            .arena = arena,
            .conv_in = try devConv(a, enc.conv_in),
            .levels = levels,
            .norm_out = try catNorm(a, enc.norm_out_w, enc.norm_out_b),
            .conv_out = try devConv(a, enc.conv_out),
            .quant = try devConv(a, enc.quant),
            .groups = cfg.norm_groups,
            .eps = cfg.norm_eps,
        };
    }
};

fn catNorm(a: std.mem.Allocator, w: []const f32, b: []const f32) ![]const f32 {
    const out = try a.alloc(f32, w.len + b.len);
    @memcpy(out[0..w.len], w);
    @memcpy(out[w.len..], b);
    return out;
}

/// `[out][in][kt][kh][kw]` -> `[out][kt][kh][kw][in]`.
fn devConv(a: std.mem.Allocator, c: Conv3d) !DevConv {
    const taps = c.kt * c.kh * c.kw;
    const n = c.out_ch * c.in_ch * taps;
    std.debug.assert(c.w.len == n);
    const w = try a.alloc(f32, n);
    for (0..c.out_ch) |oc| {
        for (0..c.in_ch) |ic| {
            for (0..taps) |j| w[(oc * taps + j) * c.in_ch + ic] = c.w[(oc * c.in_ch + ic) * taps + j];
        }
    }
    return .{
        .w = w,
        .bias = c.b,
        .out_ch = c.out_ch,
        .in_ch = c.in_ch,
        .kt = c.kt,
        .kh = c.kh,
        .kw = c.kw,
        .stride_t = c.stride_t,
        .stride_s = c.stride_s,
        .pad_t = c.pad_t,
    };
}

// --- workspace ------------------------------------------------------------

/// The extents the pipeline actually reaches, walked rather than assumed: the
/// widths double as the extents shrink, so the peak is not at either end.
pub const Shapes = struct {
    /// Largest `t * h * w * ch` any volume holds.
    vol: usize,
    /// Largest banded patch, in elements.
    patch: usize,
    /// Largest `rows * out_ch` a GEMM band writes.
    gemm: usize,

    pub fn of(enc: *const VideoEncoder, t: usize, h: usize, w: usize) Shapes {
        var s: Shapes = .{ .vol = t * h * w * 3, .patch = 0, .gemm = 0 };
        var ct = t;
        var ch = h;
        var cw = w;
        const acc = &s;

        note(acc, enc.conv_in, ct, ch, cw, .{ 1, 1 }, .{ 1, 1 });
        ct = enc_mod.convOutT(enc.conv_in, ct);
        ch = (ch + 2 - enc.conv_in.kh) / enc.conv_in.stride_s + 1;
        cw = (cw + 2 - enc.conv_in.kw) / enc.conv_in.stride_s + 1;
        s.vol = @max(s.vol, ct * ch * cw * enc.conv_in.out_ch);

        for (enc.levels, 0..) |lv, i| {
            for (lv.blocks) |rb| {
                note(acc, rb.conv1, ct, ch, cw, .{ 1, 1 }, .{ 1, 1 });
                note(acc, rb.conv2, ct, ch, cw, .{ 1, 1 }, .{ 1, 1 });
                if (rb.shortcut) |sc| note(acc, sc, ct, ch, cw, .{ 0, 0 }, .{ 0, 0 });
                s.vol = @max(s.vol, ct * ch * cw * rb.conv2.out_ch);
            }
            if (lv.down) |d| {
                const hi: usize = if (enc.cfg.space_down[i] == 2) 1 else 0;
                note(acc, d, ct, ch, cw, .{ 0, hi }, .{ 0, hi });
                ct = enc_mod.convOutT(d, ct);
                ch = (ch + hi - d.kh) / d.stride_s + 1;
                cw = (cw + hi - d.kw) / d.stride_s + 1;
                s.vol = @max(s.vol, ct * ch * cw * d.out_ch);
            }
        }
        note(acc, enc.conv_out, ct, ch, cw, .{ 1, 1 }, .{ 1, 1 });
        s.vol = @max(s.vol, ct * ch * cw * enc.conv_out.out_ch);
        note(acc, enc.quant, ct, ch, cw, .{ 0, 0 }, .{ 0, 0 });
        return s;
    }

    fn note(s: *Shapes, c: Conv3d, t: usize, h: usize, w: usize, pad_h: [2]usize, pad_w: [2]usize) void {
        const ot = enc_mod.convOutT(c, t);
        const oh = (h + pad_h[0] + pad_h[1] - c.kh) / c.stride_s + 1;
        const ow = (w + pad_w[0] + pad_w[1] - c.kw) / c.stride_s + 1;
        const rows = ot * oh * ow;
        const cols = c.kt * c.kh * c.kw * c.in_ch;
        const band = @max(1, @min(rows, patch_band / @max(1, cols)));
        s.patch = @max(s.patch, band * cols);
        s.gemm = @max(s.gemm, band * c.out_ch);
    }

    /// Device bytes `Workspace.init` will reserve, so a caller can refuse first.
    pub fn deviceBytes(self: Shapes) usize {
        return (3 * self.vol + self.patch + self.gemm) * @sizeOf(f32) + 8192;
    }
};

pub const Workspace = struct {
    /// The volume ping-pong pair, plus the resblock's saved input.
    v0: Buf = .{},
    v1: Buf = .{},
    skip: Buf = .{},
    patch: Buf = .{},
    /// The GEMM's `[rows][out_ch]` band output. Separate from the volume buffers
    /// because a banded conv writes it while still reading the source.
    gemm: Buf = .{},
    /// The im2col parameter blocks, one 64-byte slot per band of the current conv.
    prm: Buf = .{},
    /// Host staging for `prm`, grown on demand and alive for the whole batch.
    prm_host: []([16]u32) = &.{},
    prm_slots: usize = 0,
    gpa: std.mem.Allocator = undefined,
    gstat: Buf = .{},
    gmi: Buf = .{},
    shapes: Shapes = undefined,

    pub fn init(gpa: std.mem.Allocator, be: *Backend, enc: *const VideoEncoder, t: usize, h: usize, w: usize) !Workspace {
        return initFrom(gpa, be, enc, Shapes.of(enc, t, h, w));
    }

    /// Sized from shapes already computed, so the hook below can GROW a workspace
    /// rather than sizing one for the untiled extent. The spatial tiling and the
    /// clip chunking both call the compute with extents smaller than the render's,
    /// and at 1344x768 the untiled level-0 volume alone is 9 GB.
    pub fn initFrom(gpa: std.mem.Allocator, be: *Backend, enc: *const VideoEncoder, shapes: Shapes) !Workspace {
        var ws: Workspace = .{ .shapes = shapes, .gpa = gpa };
        errdefer ws.deinit(be);
        const s = ws.shapes;
        ws.v0 = try be.tensorCreate(s.vol * 4);
        ws.v1 = try be.tensorCreate(s.vol * 4);
        ws.skip = try be.tensorCreate(s.vol * 4);
        ws.patch = try be.tensorCreate(s.patch * 4);
        ws.gemm = try be.tensorCreate(s.gemm * 4);
        // Sized on first use, since the band count depends on the conv.
        ws.prm = .{};
        ws.gstat = try be.tensorCreate(enc.cfg.norm_groups * gn_chunks * 3 * 4);
        ws.gmi = try be.tensorCreate(2 * enc.cfg.norm_groups * 4);
        return ws;
    }

    /// The parameter staging for `n` bands, growing both halves together. Slots are
    /// 64 bytes so a band's block never straddles two.
    fn stage(self: *Workspace, be: *Backend, n: usize) ![]([16]u32) {
        if (n > self.prm_slots) {
            if (self.prm.size > 0) be.tensorDestroy(&self.prm);
            if (self.prm_host.len > 0) self.gpa.free(self.prm_host);
            self.prm_host = &.{};
            self.prm_slots = 0;
            self.prm = try be.tensorCreate(n * 64);
            self.prm_host = try self.gpa.alloc([16]u32, n);
            self.prm_slots = n;
        }
        return self.prm_host[0..n];
    }

    pub fn deinit(self: *Workspace, be: *Backend) void {
        inline for (.{ &self.v0, &self.v1, &self.skip, &self.patch, &self.gemm, &self.prm, &self.gstat, &self.gmi }) |b| {
            if (b.size > 0) be.tensorDestroy(b);
        }
        if (self.prm_host.len > 0) self.gpa.free(self.prm_host);
        self.* = undefined;
    }
};

// --- forward --------------------------------------------------------------

/// A channel-last volume on the device.
const DevVol = struct {
    buf: Buf,
    ch: usize,
    t: usize,
    h: usize,
    w: usize,

    fn elems(self: DevVol) usize {
        return self.ch * self.t * self.h * self.w;
    }
    fn positions(self: DevVol) usize {
        return self.t * self.h * self.w;
    }
};

fn offsetBuf(b: Buf, off_bytes: usize) Buf {
    return .{ .buf = @enumFromInt(@intFromEnum(b.buf) + off_bytes), .mem = .null_handle, .size = b.size - off_bytes };
}

fn devBuf(be: *Backend, data: []const f32) !Buf {
    return .{ .buf = try be.smallBuffer(std.mem.sliceAsBytes(data)), .mem = .null_handle, .size = data.len * 4 };
}

/// `dst = conv3d(src)`, banded im2col + GEMM.
///
/// ⚠️ **`dst` must not alias `src`**, for the same reason the audio port's does not:
/// banding interleaves the two, so band 0's GEMM would overwrite input band 1 has
/// not read yet. That is correct for any volume small enough to fit one band and
/// wrong for every larger one.
fn conv(
    be: *Backend,
    ws: *Workspace,
    dst: Buf,
    src: DevVol,
    c: DevConv,
    /// Spatial pads as (low, high) per axis. They are separate because
    /// `Downsample3D`'s pre-pad is asymmetric `(0, 1)`, and because only the LOW
    /// pad shifts the gather -- the high one just extends the output.
    ///
    /// ⚠️ **The high pad must NOT be folded into the input extent.** Telling the
    /// kernel the volume is one row taller so it can reflect into the pad reads a
    /// row the buffer does not have: an illegal access, not a wrong number. The
    /// reflect is always against the REAL extent, which is also what the reference
    /// does (its pre-pad reflects against the unpadded height and the conv that
    /// follows pads not at all).
    pad_h: [2]usize,
    pad_w: [2]usize,
) !DevVol {
    std.debug.assert(dst.buf != src.buf.buf);
    const front = 2 * c.pad_t;
    const t_pad = src.t + front;
    const out_t = if (t_pad >= c.kt) (t_pad - c.kt) / c.stride_t + 1 else 0;
    const out_h = (src.h + pad_h[0] + pad_h[1] - c.kh) / c.stride_s + 1;
    const out_w = (src.w + pad_w[0] + pad_w[1] - c.kw) / c.stride_s + 1;
    const out: DevVol = .{ .buf = dst, .ch = c.out_ch, .t = out_t, .h = out_h, .w = out_w };
    if (out_t == 0) return out;

    const cols = c.cols();
    const rows = out_t * out_h * out_w;
    const band = @max(1, @min(rows, patch_band / cols));
    std.debug.assert(band * cols <= ws.shapes.patch);

    // ⚠️ The staging array must outlive the batch, not the loop iteration: uploads
    // inside a batch are queued on the stream. One slot per band, so every queued
    // copy still has its source when it runs.
    const n_bands = (rows + band - 1) / band;
    const stage = try ws.stage(be, n_bands);
    var t0: usize = 0;
    var bi: usize = 0;
    while (t0 < rows) : ({
        t0 += band;
        bi += 1;
    }) {
        const nb = @min(band, rows - t0);
        stage[bi] = [16]u32{
            @intCast(cols),        @intCast(c.in_ch),   @intCast(c.kt),       @intCast(c.kh),
            @intCast(c.kw),        @intCast(out_h),     @intCast(out_w),      @intCast(src.t),
            @intCast(src.h),       @intCast(src.w),     @intCast(c.stride_t), @intCast(c.stride_s),
            @intCast(front),       @intCast(pad_h[0]),  @intCast(pad_w[0]),   @intCast(t0),
        };
        const pslot = offsetBuf(ws.prm, bi * 64);
        try be.tensorUpload(pslot, std.mem.sliceAsBytes(stage[bi .. bi + 1]));
        try be.opIm2col3d(src.buf, ws.patch, pslot, nb, cols);
        const wb = std.mem.sliceAsBytes(c.w);
        const gout = offsetBuf(dst, t0 * c.out_ch * 4);
        be.opMatmulF32Lt(gout, ws.patch, nb, wb, c.out_ch, cols, c.bias) catch |err| switch (err) {
            error.UnsupportedKernelArm => try be.opMatmul(gout, 0, ws.patch, 0, nb, wb, false, c.out_ch, cols, 1.0, c.bias),
            else => return err,
        };
    }
    return out;
}

/// GroupNorm + the SiLU that follows it, PER FRAME and in place.
///
/// ⚠️ One call over the whole volume would reduce across frames. The reference's
/// statistics are per frame, and a single-frame encode cannot distinguish the two,
/// so this is a convention a fixture has to carry a clip case to pin.
fn normSilu(be: *Backend, ws: *Workspace, x: DevVol, cat: Buf, groups: usize, eps: f32) !void {
    const per_frame = x.h * x.w;
    for (0..x.t) |f| {
        const off = f * per_frame * x.ch * 4;
        const v = offsetBuf(x.buf, off);
        try be.opGroupNorm(v, v, cat, ws.gstat, ws.gmi, per_frame, x.ch, groups, gn_chunks, eps, true, false);
    }
}

/// One `ResnetBlock3D`. `x` is consumed; the result lands in `dst`.
fn resBlock(be: *Backend, ws: *Workspace, dst: Buf, x: DevVol, rb: DevResBlock, groups: usize, eps: f32) !DevVol {
    // The shortcut reads the block's INPUT, so it is taken before the two norms
    // overwrite it in place.
    const skip: DevVol = if (rb.shortcut) |sc|
        try conv(be, ws, ws.skip, x, sc, .{ 0, 0 }, .{ 0, 0 })
    else blk: {
        try be.opCopyOff(ws.skip, 0, x.buf, 0, x.elems(), false);
        break :blk .{ .buf = ws.skip, .ch = x.ch, .t = x.t, .h = x.h, .w = x.w };
    };

    const n1 = try devBuf(be, rb.norm1);
    try normSilu(be, ws, x, n1, groups, eps);
    var h = try conv(be, ws, dst, x, rb.conv1, .{ 1, 1 }, .{ 1, 1 });

    const n2 = try devBuf(be, rb.norm2);
    try normSilu(be, ws, h, n2, groups, eps);
    // conv2 is length- and width-preserving, so it can go back into `x`'s buffer,
    // which is free now that the norm read it.
    h = try conv(be, ws, x.buf, h, rb.conv2, .{ 1, 1 }, .{ 1, 1 });

    std.debug.assert(h.elems() == skip.elems());
    try be.opAdd(h.buf, skip.buf, h.elems());
    return h;
}

/// The `EncoderFCN3D` trunk plus `quant_conv`, on the device: pixels in
/// `[3][t][h][w]` planar (the reference's layout, which is what the tiling above
/// hands over) to moments in the same planar layout.
pub fn encodeMoments(
    sess: *const Session,
    be: *Backend,
    ws: *Workspace,
    gpa: std.mem.Allocator,
    a: std.mem.Allocator,
    x: Vol,
) !Vol {
    std.debug.assert(x.ch == 3);
    const groups = sess.groups;
    const eps = sess.eps;

    // Planar -> channel-last on the way in. Three channels, so this is cheap; the
    // rest of the pipeline never leaves the device.
    {
        const host = try gpa.alloc(f32, x.elems());
        defer gpa.free(host);
        const per = x.t * x.h * x.w;
        for (0..per) |i| {
            for (0..3) |c| host[i * 3 + c] = x.d[c * per + i];
        }
        try be.beginBatch();
        errdefer if (be.batching()) be.abortBatch();
        try be.tensorUpload(ws.v0, std.mem.sliceAsBytes(host));
    }
    errdefer if (be.batching()) be.abortBatch();

    var cur: DevVol = .{ .buf = ws.v0, .ch = 3, .t = x.t, .h = x.h, .w = x.w };
    var other = ws.v1;
    cur = try conv(be, ws, other, cur, sess.conv_in, .{ 1, 1 }, .{ 1, 1 });
    other = ws.v0;

    for (sess.levels) |lv| {
        for (lv.blocks) |rb| {
            const next = try resBlock(be, ws, other, cur, rb, groups, eps);
            // `resBlock` leaves its result in `cur`'s own buffer (conv2 writes
            // there), so the pair does not swap.
            std.debug.assert(next.buf.buf == cur.buf.buf);
            cur = next;
        }
        if (lv.down) |d| {
            // `Downsample3D`'s asymmetric `(0, 1, 0, 1)` reflect pad is a HIGH pad
            // of one on each spatial axis and nothing else: same output extent, and
            // the reflect still against the real one.
            const hi: usize = if (lv.space_stride == 2) 1 else 0;
            cur = try conv(be, ws, other, cur, d, .{ 0, hi }, .{ 0, hi });
            other = if (cur.buf.buf == ws.v0.buf) ws.v1 else ws.v0;
        }
    }

    const nout = try devBuf(be, sess.norm_out);
    try normSilu(be, ws, cur, nout, groups, eps);
    cur = try conv(be, ws, other, cur, sess.conv_out, .{ 1, 1 }, .{ 1, 1 });
    other = if (cur.buf.buf == ws.v0.buf) ws.v1 else ws.v0;
    cur = try conv(be, ws, other, cur, sess.quant, .{ 0, 0 }, .{ 0, 0 });
    try be.endBatch();

    // ...and channel-last -> planar on the way out. 32 channels at the latent
    // resolution, so also cheap.
    const per = cur.t * cur.h * cur.w;
    const host = try gpa.alloc(f32, cur.elems());
    defer gpa.free(host);
    try be.tensorDownload(cur.buf, std.mem.sliceAsBytes(host[0..cur.elems()]));
    const out = try a.alloc(f32, cur.elems());
    for (0..per) |i| {
        for (0..cur.ch) |c| out[c * per + i] = host[i * cur.ch + c];
    }
    return .{ .d = out, .ch = cur.ch, .t = cur.t, .h = cur.h, .w = cur.w };
}

/// The hook `minimax_h3_vae_encode.encode` takes, so the temporal chunking and the
/// spatial tiling run once for both backends.
///
/// It owns a workspace that GROWS: the tiling calls the compute once per tile and
/// the chunking once per clip, all with extents smaller than the render's, and
/// sizing for the render would reserve several times what any one call needs.
pub const Ctx = struct {
    enc: *const VideoEncoder,
    sess: *const Session,
    be: *Backend,
    /// Host memory for the workspace's parameter staging.
    gpa: std.mem.Allocator,
    ws: ?Workspace = null,
    /// Largest shapes reserved so far, for reporting.
    peak: Shapes = .{ .vol = 0, .patch = 0, .gemm = 0 },

    pub fn deinit(self: *Ctx) void {
        if (self.ws) |*w| w.deinit(self.be);
        self.* = undefined;
    }

    pub fn moments(self: *Ctx) enc_mod.Moments {
        return .{ .ctx = self, .run = run };
    }

    fn run(ctx: *anyopaque, gpa: std.mem.Allocator, a: std.mem.Allocator, x: Vol) anyerror!Vol {
        const self: *Ctx = @ptrCast(@alignCast(ctx));
        const need = Shapes.of(self.enc, x.t, x.h, x.w);
        const have = if (self.ws) |w| w.shapes else Shapes{ .vol = 0, .patch = 0, .gemm = 0 };
        if (need.vol > have.vol or need.patch > have.patch or need.gemm > have.gemm) {
            // Grow to the MAX of the two, so a sequence of differently shaped tiles
            // reallocates at most a few times rather than on every change.
            const grown: Shapes = .{
                .vol = @max(need.vol, have.vol),
                .patch = @max(need.patch, have.patch),
                .gemm = @max(need.gemm, have.gemm),
            };
            if (self.ws) |*w| w.deinit(self.be);
            self.ws = null;
            self.ws = try Workspace.initFrom(self.gpa, self.be, self.enc, grown);
            self.peak = grown;
        }
        return encodeMoments(self.sess, self.be, &self.ws.?, gpa, a, x);
    }
};
