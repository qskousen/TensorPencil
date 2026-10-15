//! The CUDA convolutions every model runs, both arms: a 2-D conv (1x1, 3x3 at stride 1
//! or 2, a fused nearest 2x upsample) whose 3x3 gathers its patch straight into the 16-bit
//! GEMM operand, through cuDNN on the library arm where it can, and the 1-D and 3-D convs,
//! each its own gather into the GEMM `lin_cuda` routes for the weight.

const std = @import("std");
const ops = @import("tp_ops");
const cuda = @import("tp_gpu").cuda;
const lin_cuda = @import("lin_cuda.zig");

const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;
const Dt = cuda.backend.Dt;
const Weight = ops.matmul.Weight;
const Conv2d = ops.conv.Conv2d;
const dual_table = @import("tp_gpu").dual_table;

pub const SampleMode = ops.conv.SampleMode;

/// Cap on the im2col patch band (bytes); bands iterate over output rows.
const patch_band_bytes: usize = 256 << 20;
/// Convolutions at least this wide take the f16 tensor-core GEMM; below it the
/// f32 GEMM wins, because padding `co` out to the cooperative tile costs more
/// than the tensor cores return.
pub const coop_min_co: usize = 96;

/// Output positions per im2col band, a multiple of 4.
pub fn convBand(n_out: usize, patch_len: usize) usize {
    return @max(4, @min(n_out, patch_band_bytes / (patch_len * 4)) & ~@as(usize, 3));
}

/// The `[co][k][k][ci]` device weight a conv's 16-bit kernels read, at `op`: converted
/// once and cached for a dense weight, decoded per call for a block quant
/// (`weight16`). Every CUDA conv arm resolves its weight here.
pub fn weights16(be: *Backend, w: ops.conv.Weights, co: usize, ci: usize, k: usize, op: Dt) !Buf {
    const kk = k * k;
    return switch (w) {
        .krsc => |s| be.weight16(std.mem.sliceAsBytes(s), .f32, false, co, ci, kk, 1, op),
        .torch => |t| be.weight16(t.w.bytes, t.w.dtype, true, co, ci, kk, t.kt, op),
    };
}

/// `weights16` at f16.
pub fn weightsF16(be: *Backend, w: ops.conv.Weights, co: usize, ci: usize, k: usize) !Buf {
    return weights16(be, w, co, ci, k, .f16);
}

fn convW16(be: *Backend, cv: Conv2d, op: Dt) !Buf {
    return weights16(be, cv.w, cv.co, cv.ci, cv.k, op);
}

/// One 2-D convolution, `src` read and `dst` written at their own dtypes; `patch` is
/// the f32 im2col band the paths that still build one use, grown as needed. The GEMM is
/// 16-bit operands with an f32 accumulator for a 16-bit chain or `co >= coop_min_co`,
/// the f32 GEMM below that. `act_div` divides the activation before a 16-bit cast and
/// is undone in the epilogue (`Backend.opConvF16Scaled` says why f16's 65504 is
/// reachable); an f16 `src` cannot carry one, a value that needed it could not be in it.
pub fn conv2d(
    be: *Backend,
    patch: *Buf,
    dst: *Buf,
    src: *const Buf,
    h: usize,
    w: usize,
    cv: Conv2d,
    mode: SampleMode,
    /// Output grid override, for the decoder's upsample only: `null` derives it from
    /// `mode`. The encoder halves rounding up, so on an odd dimension the skip this
    /// output feeds is one row short of `2 * h` and the conv has to produce exactly it.
    /// Sampling the source at `dst / 2` stays correct, which is what makes an explicit
    /// size the whole change (`sd_unet` has the arithmetic).
    out_hw: ?[2]usize,
    act_div: f32,
) !void {
    std.debug.assert(src.dt != .f16 or act_div == 1.0);
    const op = Backend.operand16(src.*, dst.*);
    const cb = cv.b orelse return error.MissingConvBias;
    // 16-bit storage FORCES the cooperative path, because the f32 GEMM arm has no
    // 16-bit form. `coop_min_co` is a performance threshold, not a correctness one
    // (padding `co` to the tile costs more than the tensor cores return below it),
    // so widening it here is free of consequence, and not widening it was a real
    // bug: SD1.5's `post_quant_conv` is 4->4, fell to the f32 arm, and wrote f32
    // into an f16 buffer. Every pixel non-finite, caught by `sd-cuda-test`'s sweep.
    // A checkpoint-order weight has only the 16-bit arm: the f32 GEMM reads repacked f32.
    const coop = cv.co >= coop_min_co or src.dt != .f32 or dst.dt != .f32 or cv.w == .torch;
    if (cv.k == 1) {
        std.debug.assert(mode == .stride1);
        const n = h * w;
        // The activation-capture probe reads f32; a 16-bit buffer would be filed as
        // noise, so it is skipped rather than lied to (capture and 16-bit storage are
        // not used together, one is a measurement run, the other a render).
        if (src.dt == .f32) try lin_cuda.probeInput(be, src.*, n, cv.gemmWeight());
        if (coop) return be.opGemm16(dst.*, 0, src.*, n, try convW16(be, cv, op), cv.co, cv.ci, cb, act_div, op);
        return be.opMatmul(dst.*, 0, src.*, 0, n, std.mem.sliceAsBytes(cv.krsc()), false, cv.co, cv.ci, 1.0, cb);
    }
    std.debug.assert(cv.k == 3);

    const oh = if (out_hw) |o| o[0] else switch (mode) {
        .stride1 => h,
        .upsample2x => 2 * h,
        .stride2 => (h + 1) / 2,
    };
    const ow = if (out_hw) |o| o[1] else switch (mode) {
        .stride1 => w,
        .upsample2x => 2 * w,
        .stride2 => (w + 1) / 2,
    };
    const n_out = oh * ow;
    const patch_len = 9 * cv.ci;

    // Library backend: a fused cuDNN NHWC convolution, which does the im2col
    // implicitly in shared memory. The banded path materializes a [n][9*ci] f32
    // patch and then converts it to f16, so it moves nine times the activation
    // twice; cuDNN reads the activation once.
    //
    // Only stride 1: `cudnn.ConvPlan` is a 3x3 pad-1 stride-1 plan, and the
    // upsample convolutions need the resample fused into the gather, which no
    // convolution op can do.
    //
    // An activation capture stays banded too: the GEMM this convolution is
    // equivalent to reads the [n][9*ci] patch, and that matrix is what the probe
    // has to see. cuDNN never builds it, so probing here would file per-column
    // statistics `9*ci` wide over a `ci`-wide activation.
    if (be.kernels == .libs and mode == .stride1 and act_div == 1.0 and ops.matmul.probe == null) {
        return be.opConvCudnnW16(dst.*, 0, src.*, h, w, try convW16(be, cv, op), cv.co, cv.ci, cb);
    }

    const band = convBand(n_out, patch_len);
    // Once for every band: a block quant's decode is a per-call scratch. Held, since
    // every band launches before the next one reads it.
    const w16: ?Buf = if (coop) try convW16(be, cv, op) else null;
    if (w16) |x| be.hold(x);
    defer if (w16 != null) be.unhold();

    const taps: dual_table.ConvTaps = .{
        .ci = cv.ci,
        .src_w = w,
        .extent_h = if (mode == .upsample2x) oh else h,
        .out_w = ow,
        .first = 0,
        .mode = @intFromEnum(mode),
    };
    // The patch gathered straight into the GEMM's 16-bit operand, unless an activation
    // capture needs the f32 patch the GEMM is equivalent to.
    if (coop and taps.fits(be.gatherKPad(patch_len, op, dst.dt)) and ops.matmul.probe == null and !dual_table.force_conv_patch) {
        var p0: usize = 0;
        while (p0 < n_out) : (p0 += band) {
            var t = taps;
            t.first = p0;
            try be.opConvGemm16(dst.*, p0 * cv.co, src.*, @min(band, n_out - p0), w16.?, cv.co, t, cb, act_div, op);
        }
        return;
    }
    try be.ensureDeviceBuffer(patch, band * patch_len * 4);

    var p0: usize = 0;
    while (p0 < n_out) : (p0 += band) {
        const bn = @min(band, n_out - p0);
        // `h` here is the tap EXTENT, not the source height: when upsampling the taps
        // live in the output grid, and that is where the conv pads (see `im2col_sd`).
        const ext_h = if (mode == .upsample2x) oh else h;
        try be.opIm2colSd(src.*, patch.*, bn, patch_len, cv.ci, w, ext_h, p0, ow, @intFromEnum(mode));
        // Per band, not per convolution: the accumulator sums over rows, so N banded
        // calls contribute exactly the rows one unbanded call would.
        try lin_cuda.probeInput(be, patch.*, bn, cv.gemmWeight());
        if (coop) {
            try be.opGemm16(dst.*, p0 * cv.co, patch.*, bn, w16.?, cv.co, patch_len, cb, act_div, op);
        } else {
            try be.opMatmul(dst.*, p0 * cv.co * 4, patch.*, 0, bn, std.mem.sliceAsBytes(cv.krsc()), false, cv.co, patch_len, 1.0, cb);
        }
    }
}

/// The GEMM a conv weight feeds: `dst[rows][co]` = `x[rows][k*k*ci]` @ Wᵀ + bias, `x`
/// an already gathered patch matrix, on the 16-bit tensor cores at the operand width
/// `x` and `dst` call for.
pub fn patchGemm(be: *Backend, dst: Buf, x: Buf, rows: usize, cv: Conv2d) !void {
    const op = Backend.operand16(x, dst);
    const cb = cv.b orelse return error.MissingConvBias;
    try be.opGemm16(dst, 0, x, rows, try weights16(be, cv.w, cv.co, cv.ci, cv.k, op), cv.co, cv.k * cv.k * cv.ci, cb, 1.0, op);
}

/// A non-overlapping k x k stride-k convolution over channel-last `[h][w][ci]` (a patch
/// embedder): the patch matrix gathered into `patch` (f32, grown), then `patchGemm`.
/// `dst` is `[h/k * w/k][co]`.
pub fn convStride(be: *Backend, patch: *Buf, dst: Buf, src: Buf, h: usize, w: usize, cv: Conv2d) !void {
    const n_out = (h / cv.k) * (w / cv.k);
    try be.ensureDeviceBuffer(patch, n_out * cv.k * cv.k * cv.ci * 4);
    try be.opIm2colStride(patch.*, src, h, w, cv.ci, cv.k);
    try patchGemm(be, dst, patch.*, n_out, cv);
}

/// `dst[out_len][out_ch] = conv1d(src[in_len][in_ch])`: the patch gathered into
/// `patch` (f32, `patch_elems` of it) a band of output rows at a time, each band's GEMM
/// through `lin_cuda` under `plan`. Returns `out_len`. A `dst` aliasing `src` must fit
/// one band: banding would overwrite input a later band still reads.
pub fn conv1d(be: *Backend, plan: lin_cuda.Plan, patch: Buf, patch_elems: usize, dst: Buf, src: Buf, c: ops.conv.Conv1dGemm, in_len: usize) !usize {
    const out_len = c.outLen(in_len);
    const cols = c.w.cols;
    std.debug.assert(cols == c.k * c.in_ch and patch_elems >= cols);
    const band = @min(out_len, patch_elems / cols);
    if (dst.buf == src.buf and band < out_len) return error.AliasedConvBands;
    // The band start rides in a float push word.
    if (out_len > 1 << 24) return error.UnsupportedShape;
    var t0: usize = 0;
    while (t0 < out_len) : (t0 += band) {
        const nb = @min(band, out_len - t0);
        try be.opIm2col1d(src, patch, t0, nb, c.k, c.in_ch, in_len, c.dilation, c.padding, c.stride);
        try lin_cuda.linear(be, plan, dst.viewBytes(t0 * c.outCh() * 4), patch, nb, c.w, c.bias);
    }
    return out_len;
}

/// A channel-last volume `[t][h][w][ch]` on the device.
pub const Vol3 = struct {
    buf: Buf,
    ch: usize,
    t: usize,
    h: usize,
    w: usize,

    pub fn elems(self: Vol3) usize {
        return self.ch * self.t * self.h * self.w;
    }
    pub fn positions(self: Vol3) usize {
        return self.t * self.h * self.w;
    }
};

/// A 3-D causal convolution as the GEMM its gather feeds: `w` is
/// `[out_ch][kt * kh * kw * in_ch]`, channels innermost, the column order `im2col3d`
/// writes. Time is front-padded by `2 * pad_t` only.
pub const Conv3dGemm = struct {
    w: Weight,
    bias: ?[]const f32,
    in_ch: usize,
    kt: usize,
    kh: usize,
    kw: usize,
    stride_t: usize,
    stride_s: usize,
    pad_t: usize,

    pub fn outCh(c: Conv3dGemm) usize {
        return c.w.rows;
    }
};

/// `dst = conv3d(src)`: the patch gathered into `patch` (f32, `patch_elems` of it) a
/// band of output positions at a time, each band's GEMM through `lin_cuda` under
/// `plan`. The gather's shape block is staged in `prm_host` and uploaded once to `prm`;
/// `prm_host` must outlive the batch, since an upload inside one is queued on the
/// stream.
///
/// ⚠️ **`dst` must not alias `src`**, for the same reason the audio port's does not:
/// banding interleaves the two, so band 0's GEMM would overwrite input band 1 has
/// not read yet. That is correct for any volume small enough to fit one band and
/// wrong for every larger one.
pub fn conv3d(
    be: *Backend,
    plan: lin_cuda.Plan,
    patch: Buf,
    patch_elems: usize,
    prm: Buf,
    prm_host: *[16]u32,
    dst: Buf,
    src: Vol3,
    c: Conv3dGemm,
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
) !Vol3 {
    std.debug.assert(dst.buf != src.buf.buf);
    const front = 2 * c.pad_t;
    const t_pad = src.t + front;
    const out_t = if (t_pad >= c.kt) (t_pad - c.kt) / c.stride_t + 1 else 0;
    const out_h = (src.h + pad_h[0] + pad_h[1] - c.kh) / c.stride_s + 1;
    const out_w = (src.w + pad_w[0] + pad_w[1] - c.kw) / c.stride_s + 1;
    const out: Vol3 = .{ .buf = dst, .ch = c.outCh(), .t = out_t, .h = out_h, .w = out_w };
    if (out_t == 0) return out;

    const cols = c.w.cols;
    const rows = out_t * out_h * out_w;
    const band = @max(1, @min(rows, patch_elems / cols));
    std.debug.assert(band * cols <= patch_elems);

    // The same shape for every band: uploaded once, only the band's first row moves.
    prm_host.* = [16]u32{
        @intCast(cols),        @intCast(c.in_ch),   @intCast(c.kt),       @intCast(c.kh),
        @intCast(c.kw),        @intCast(out_h),     @intCast(out_w),      @intCast(src.t),
        @intCast(src.h),       @intCast(src.w),     @intCast(c.stride_t), @intCast(c.stride_s),
        @intCast(front),       @intCast(pad_h[0]),  @intCast(pad_w[0]),   0,
    };
    try be.tensorUpload(prm, std.mem.asBytes(prm_host));
    var t0: usize = 0;
    while (t0 < rows) : (t0 += band) {
        const nb = @min(band, rows - t0);
        try be.opIm2col3d(src.buf, patch, prm, t0, nb, cols);
        try lin_cuda.linear(be, plan, dst.viewBytes(t0 * c.outCh() * 4), patch, nb, c.w, c.bias);
    }
    return out;
}
