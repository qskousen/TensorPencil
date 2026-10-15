//! The Vulkan convolutions every model runs: a 2-D conv (1x1, 3x3 at stride 1 or 2, a
//! fused nearest 2x upsample) whose 3x3 gathers its patch straight into the coop GEMM's
//! 16-bit operand, and the 1-D conv, its gather into the GEMM `lin_gpu` routes for the
//! weight. The twin of `conv_cuda.zig`.

const std = @import("std");
const ops = @import("tp_ops");
const gpu_context = @import("tp_gpu").context;
const lin_gpu = @import("lin_gpu.zig");

const Context = gpu_context.Context;
const DeviceBuffer = gpu_context.DeviceBuffer;
const Conv2d = ops.conv.Conv2d;
const dual_table = @import("tp_gpu").dual_table;

pub const SampleMode = ops.conv.SampleMode;

/// Cap on the im2col patch band (bytes); bands iterate over output rows.
const patch_band_bytes: usize = 256 << 20;
/// Convolutions at least this wide go to the f16 tensor-core GEMM; below it the
/// f32 register-tile GEMM wins, because padding `co` out to the 128-wide
/// cooperative tile would waste more than the tensor cores return.
pub const coop_min_co: usize = 96;
/// A slice of a device buffer holding a per-forward convolution bias.
pub const BiasSlice = struct { buf: DeviceBuffer, off: usize };

/// Output positions per im2col band. A multiple of 4, which keeps the GEMM's `y`
/// byte offset 16-aligned for any `co`. Public so a caller can pre-size the patch
/// buffer before opening a command batch.
pub fn convBand(n_out: usize, patch_len: usize) usize {
    return @max(4, @min(n_out, patch_band_bytes / (patch_len * 4)) & ~@as(usize, 3));
}

/// One 2-D convolution, `src` read and `dst` written at their own `dt`; the twin of
/// `conv_cuda.conv2d`. `patch` is the f32 im2col band the paths that still build one
/// use. `bias_dev`, a device bias slice, replaces `cv.b` for a bias that changes per
/// forward (the SD UNet folds its timestep projection into it). `act_div` divides the
/// activation before a 16-bit cast and is undone in the epilogue
/// (`Context.opMatmulCoopF16WScaled` says why); the f32 GEMM arm needs none.
///
/// 16-bit storage FORCES the cooperative path, because the plain f32 GEMM arm has
/// no f16 form. `coop_min_co` is a PERFORMANCE threshold, not a correctness one,
/// and not forcing it is a real bug, not a theoretical one: on CUDA it left SD1.5's
/// 4->4 `post_quant_conv` on the f32 arm writing f32 into an f16 buffer, and every
/// pixel came out non-finite.
pub fn conv2d(
    ctx: *Context,
    patch: *DeviceBuffer,
    dst: *DeviceBuffer,
    src: *const DeviceBuffer,
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
    bias_dev: ?BiasSlice,
    act_div: f32,
) !void {
    std.debug.assert(src.dt != .f16 or act_div == 1.0);
    // A checkpoint-order weight has only the coop arm: the f32 GEMM reads repacked f32.
    const coop = ctx.pipe_coop_f16w != .null_handle and (cv.co >= coop_min_co or src.dt != .f32 or dst.dt != .f32 or cv.w == .torch);
    if (!coop and cv.w == .torch) return error.UnsupportedDType;
    // Every convolution in an LDM UNet carries a bias; `Conv2d` models the
    // general case, so pin the assumption here rather than in five call sites.
    const cb = cv.b orelse return error.MissingConvBias;
    // A device bias slice (the UNet's folded timestep projection) or the conv's own.
    const bias_buf, const bias_off = if (bias_dev) |bd| .{ bd.buf.buf, bd.off } else .{ try ctx.smallBuffer(std.mem.sliceAsBytes(cb)), 0 };
    const div = if (bias_dev != null) 1.0 else act_div;
    if (cv.k == 1) {
        std.debug.assert(mode == .stride1);
        const n = h * w;
        try lin_gpu.probeInput(ctx, src.*, n, cv.gemmWeight());
        std.debug.assert(coop or bias_dev == null);
        if (coop) return ctx.opMatmulCoopConv(dst.*, 0, src.*, src.dt, n, cv, bias_buf, bias_off, div);
        std.debug.assert(src.dt == .f32 and dst.dt == .f32); // unreachable: `coop` covers the rest
        return ctx.opMatmul(dst.*, 0, src.*, 0, n, std.mem.sliceAsBytes(cv.krsc()), false, cv.co, cv.ci, 1.0, cb);
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

    const band = convBand(n_out, patch_len);
    const taps: dual_table.ConvTaps = .{
        .ci = cv.ci,
        .src_w = w,
        // The tap extent, not the source height: when upsampling, the taps live in
        // the OUTPUT grid and that is where the conv pads (see `im2col_sd`).
        .extent_h = if (mode == .upsample2x) oh else h,
        .out_w = ow,
        .first = 0,
        .mode = @intFromEnum(mode),
    };
    // The patch gathered straight into the GEMM's 16-bit operand, unless an activation
    // capture needs the f32 patch the GEMM is equivalent to.
    if (coop and taps.fits(std.mem.alignForward(usize, patch_len, 64)) and ops.matmul.probe == null and !dual_table.force_conv_patch) {
        var p0: usize = 0;
        while (p0 < n_out) : (p0 += band) {
            var t = taps;
            t.first = p0;
            try ctx.opConvCoop(dst.*, p0 * cv.co, src.*, @min(band, n_out - p0), cv, t, bias_buf, bias_off, div);
        }
        return;
    }
    try ctx.ensureDeviceBuffer(patch, band * patch_len * 4);

    var p0: usize = 0;
    while (p0 < n_out) : (p0 += band) {
        const bn = @min(band, n_out - p0);
        try ctx.opElt(gpu_context.eltAt("im2col_sd", src.dt), src.*, null, null, patch.*, .{
            .u0 = @intCast(bn * patch_len),
            .u1 = @intCast(patch_len),
            .u2 = @intCast(cv.ci),
            .u3 = @intCast(w),
            // The tap extent, not the source height: when upsampling, the taps live in
            // the OUTPUT grid and that is where the conv pads (see `im2col_sd`).
            .u4 = @intCast(if (mode == .upsample2x) oh else h),
            .u5 = @intCast(p0),
            .u6 = @intCast(ow),
            .f0 = @floatFromInt(@intFromEnum(mode)),
        }, bn * patch_len, 1, 1);
        // Per band; the accumulator sums over rows (see sd_unet_cuda).
        try lin_gpu.probeInput(ctx, patch.*, bn, cv.gemmWeight());
        if (coop) {
            // The patch is f32, but its values are `src`'s, which pick the operand type.
            try ctx.opMatmulCoopConv(dst.*, p0 * cv.co, patch.*, src.dt, bn, cv, bias_buf, bias_off, div);
        } else {
            std.debug.assert(bias_dev == null);
            std.debug.assert(dst.dt == .f32);
            try ctx.opMatmul(dst.*, p0 * cv.co * 4, patch.*, 0, bn, std.mem.sliceAsBytes(cv.krsc()), false, cv.co, patch_len, 1.0, cb);
        }
    }
}

/// Whether this device needs its convs repacked to f32 before it can run them: only
/// without the f16 coop GEMM, whose absence leaves the f32 GEMM, which reads only
/// repacked f32. With it, a conv runs from the checkpoint's own tensor and no f32 copy
/// is held (`Context.convWeight16`).
pub fn needsRepack(ctx: *const Context) bool {
    return ctx.pipe_coop_f16w == .null_handle;
}

/// The GEMM a conv weight feeds: `dst[rows][co]` = `x[rows][k*k*ci]` @ Wᵀ + bias, `x`
/// an already gathered patch matrix, on the coop GEMM at the operand width `x` and
/// `dst` call for.
pub fn patchGemm(ctx: *Context, dst: DeviceBuffer, x: DeviceBuffer, rows: usize, cv: Conv2d) !void {
    const cb = cv.b orelse return error.MissingConvBias;
    try ctx.opMatmulCoopConv(dst, 0, x, x.dt, rows, cv, try ctx.smallBuffer(std.mem.sliceAsBytes(cb)), 0, 1.0);
}

/// A non-overlapping k x k stride-k convolution over channel-last `[h][w][ci]` (a patch
/// embedder): the patch matrix gathered into `patch` (f32, grown), then `patchGemm`.
/// `dst` is `[h/k * w/k][co]`.
pub fn convStride(ctx: *Context, patch: *DeviceBuffer, dst: DeviceBuffer, src: DeviceBuffer, h: usize, w: usize, cv: Conv2d) !void {
    std.debug.assert(h % cv.k == 0 and w % cv.k == 0);
    const n_out = (h / cv.k) * (w / cv.k);
    const total = n_out * cv.k * cv.k * cv.ci;
    try ctx.ensureDeviceBuffer(patch, total * 4);
    try ctx.opElt(.im2col_stride, src, patch.*, null, null, .{
        .u0 = @intCast(total),
        .u1 = @intCast(cv.ci),
        .u2 = @intCast(cv.k),
        .u3 = @intCast(w),
    }, total, 1, 1);
    try patchGemm(ctx, dst, patch.*, n_out, cv);
}

/// `dst[out_len][out_ch] = conv1d(src[in_len][in_ch])`: the patch gathered into
/// `patch` (f32, `patch_elems` of it) a band of output rows at a time, each band's GEMM
/// through `lin` under `plan`. Returns `out_len`. A `dst` aliasing `src` must fit one
/// band: banding would overwrite input a later band still reads.
pub fn conv1d(lin: *const lin_gpu.Lin, plan: lin_gpu.Plan, patch: DeviceBuffer, patch_elems: usize, dst: DeviceBuffer, src: DeviceBuffer, c: ops.conv.Conv1dGemm, in_len: usize) !usize {
    const ctx = lin.ctx;
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
        try ctx.opElt(.im2col1d, src, patch, null, null, .{
            .u0 = @intCast(nb * cols),
            .u1 = @intCast(cols),
            .u2 = @intCast(c.in_ch),
            .u3 = @intCast(in_len),
            .u4 = @intCast(c.dilation),
            .u5 = @intCast(c.padding),
            .f0 = @floatFromInt(c.stride),
            .f1 = @floatFromInt(t0),
        }, nb * cols, 1, 1);
        try lin.linearAt(plan, dst, t0 * c.outCh(), patch, 0, nb, c.w, c.bias);
    }
    return out_len;
}

// --- tests ------------------------------------------------------------------

const testing = std.testing;
const none: DeviceBuffer = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 };

/// A live Vulkan context, or a skip: needs the `testdata/gpu-tests` marker and
/// `-Dintegration`.
fn gpuCtx(gpa: std.mem.Allocator, io: std.Io) !*Context {
    std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    return Context.init(gpa, std.testing.io) catch error.SkipZigTest;
}

fn upload(ctx: *Context, db: *DeviceBuffer, host: []const f32) !void {
    try ctx.ensureDeviceBuffer(db, host.len * 4);
    try ctx.tensorUpload(db.*, std.mem.sliceAsBytes(host));
}

fn relL2(want: []const f32, got: []const f32) f64 {
    var num: f64 = 0;
    var den: f64 = 0;
    for (want, got) |e, a| {
        const d = @as(f64, e) - @as(f64, a);
        num += d * d;
        den += @as(f64, e) * @as(f64, e);
    }
    return if (den == 0) num else @sqrt(num / den);
}

test "gpu conv matches ops.conv.conv2d at stride 1, stride 2 and fused 2x upsample" {
    const gpa = testing.allocator;
    const io = testing.io;
    const ctx = try gpuCtx(gpa, io);
    defer ctx.deinit();

    var prng = std.Random.DefaultPrng.init(19);
    const rand = prng.random();
    // Odd extents so stride 2's ceil(h/2) output size and the pad edges both
    // matter; co >= 96 puts it on the tensor-core arm, which is what ships.
    const h: usize = 11;
    const w: usize = 7;
    const ci: usize = 32;
    const co: usize = 128;

    const torch_w = try gpa.alloc(f32, co * ci * 9);
    defer gpa.free(torch_w);
    for (torch_w) |*x| x.* = rand.floatNorm(f32) * 0.1;
    const bias = try gpa.alloc(f32, co);
    defer gpa.free(bias);
    for (bias) |*x| x.* = rand.floatNorm(f32);
    const packed_w = try ops.conv.packWeight(gpa, torch_w, co, ci, 3);
    defer gpa.free(packed_w);
    const in = try gpa.alloc(f32, h * w * ci);
    defer gpa.free(in);
    for (in) |*x| x.* = rand.floatNorm(f32);

    var patch: DeviceBuffer = none;
    defer ctx.tensorDestroy(&patch);
    var src: DeviceBuffer = none;
    defer ctx.tensorDestroy(&src);
    var dst: DeviceBuffer = none;
    defer ctx.tensorDestroy(&dst);
    try upload(ctx, &src, in);

    for ([3]SampleMode{ .stride1, .stride2, .upsample2x }) |mode| {
        const oh = switch (mode) {
            .stride1 => h,
            .stride2 => (h + 1) / 2,
            .upsample2x => 2 * h,
        };
        const ow = switch (mode) {
            .stride1 => w,
            .stride2 => (w + 1) / 2,
            .upsample2x => 2 * w,
        };
        const cv: Conv2d = .{
            .w = .{ .krsc = packed_w },
            .b = bias,
            .co = co,
            .ci = ci,
            .k = 3,
            .stride = if (mode == .stride2) @as(usize, 2) else 1,
            .pad = 1,
        };
        try ctx.ensureDeviceBuffer(&dst, oh * ow * co * 4);
        try conv2d(ctx, &patch, &dst, &src, h, w, cv, mode, null, null, 1.0);
        const got = try gpa.alloc(f32, oh * ow * co);
        defer gpa.free(got);
        try ctx.tensorDownload(dst, std.mem.sliceAsBytes(got));

        // The CPU reference has no fused resample, so the upsample case builds
        // the doubled tensor explicitly, which is exactly the equivalence the
        // fused gather claims.
        const want = try gpa.alloc(f32, oh * ow * co);
        defer gpa.free(want);
        if (mode == .upsample2x) {
            const up = try gpa.alloc(f32, oh * ow * ci);
            defer gpa.free(up);
            for (0..oh) |y| {
                for (0..ow) |xx| {
                    const s = in[((y / 2) * w + (xx / 2)) * ci ..][0..ci];
                    @memcpy(up[(y * ow + xx) * ci ..][0..ci], s);
                }
            }
            try ops.conv.conv2d(io, gpa, want, up, oh, ow, cv);
        } else {
            try ops.conv.conv2d(io, gpa, want, in, h, w, cv);
        }
        const err = relL2(want, got);
        errdefer std.debug.print("mode={t}: rel L2 {d:.6}\n", .{ mode, err });
        // f16 tensor cores on a k = 9*ci reduction.
        try testing.expect(err < 2e-3);
    }
}

test "gpu conv gathered into the GEMM operand matches the f32 patch bit for bit" {
    // Both paths hand the GEMM the same 16-bit values, so any differing byte is a
    // gather defect. ci 32 takes the eight-wide path (a word copy from a 16-bit source
    // of the operand's type), ci 12 the per-element one.
    const gpa = testing.allocator;
    const io = testing.io;
    const ctx = try gpuCtx(gpa, io);
    defer ctx.deinit();
    defer dual_table.force_conv_patch = false;

    var prng = std.Random.DefaultPrng.init(23);
    const rand = prng.random();
    const h: usize = 11;
    const w: usize = 7;
    const co: usize = 128;
    var patch: DeviceBuffer = none;
    defer ctx.tensorDestroy(&patch);
    var cases: usize = 0;
    for ([_]usize{ 32, 12 }) |ci| {
        const wt = try gpa.alloc(f32, co * ci * 9);
        defer gpa.free(wt);
        for (wt) |*x| x.* = rand.floatNorm(f32) * 0.1;
        const bias = try gpa.alloc(f32, co);
        defer gpa.free(bias);
        for (bias) |*x| x.* = rand.floatNorm(f32);
        const in = try gpa.alloc(f32, h * w * ci);
        defer gpa.free(in);
        for (in) |*x| x.* = rand.floatNorm(f32) * 40;
        var src32: DeviceBuffer = none;
        defer ctx.tensorDestroy(&src32);
        try upload(ctx, &src32, in);
        const cv: Conv2d = .{ .w = .{ .krsc = wt }, .b = bias, .co = co, .ci = ci, .k = 3, .pad = 1 };
        for ([_]gpu_context.Dt{ .f32, .f16, .bf16 }) |sdt| {
            var src: DeviceBuffer = none;
            defer if (sdt != .f32) ctx.tensorDestroy(&src);
            if (sdt == .f32) src = src32 else {
                try ctx.ensureDeviceBuffer(&src, in.len * 2);
                src.dt = sdt;
                try ctx.opElt(gpu_context.eltCopy(.f32, sdt), src32, src, null, null, .{ .u0 = @intCast(in.len) }, in.len, 1, 1);
            }
            for ([_]gpu_context.Dt{ .f32, .f16, .bf16 }) |ddt| for ([_]f32{ 1, 256 }) |div| for ([3]SampleMode{ .stride1, .stride2, .upsample2x }) |mode| {
                if (sdt == .f16 and div != 1) continue;
                var c = cv;
                if (mode == .stride2) c.stride = 2;
                const oh = switch (mode) {
                    .stride1 => h,
                    .stride2 => (h + 1) / 2,
                    .upsample2x => 2 * h,
                };
                const ow = switch (mode) {
                    .stride1 => w,
                    .stride2 => (w + 1) / 2,
                    .upsample2x => 2 * w,
                };
                const bytes = oh * ow * co * ddt.bytes();
                var outs: [2][]u8 = undefined;
                for (0..2) |path| {
                    dual_table.force_conv_patch = path == 1;
                    var dst: DeviceBuffer = none;
                    defer ctx.tensorDestroy(&dst);
                    try ctx.ensureDeviceBuffer(&dst, bytes);
                    dst.dt = ddt;
                    try conv2d(ctx, &patch, &dst, &src, h, w, c, mode, null, null, div);
                    outs[path] = try gpa.alloc(u8, bytes);
                    try ctx.tensorDownload(dst, outs[path]);
                }
                defer for (outs) |o| gpa.free(o);
                errdefer std.debug.print("ci {d} src {t} dst {t} div {d} mode {t}: the gather and the patch differ\n", .{ ci, sdt, ddt, div, mode });
                try testing.expectEqualSlices(u8, outs[1], outs[0]);
                cases += 1;
            };
        }
    }
    try testing.expectEqual(@as(usize, 2 * (3 * 3 * 2 * 3 - 3 * 3)), cases);
}

test "gpu conv from the checkpoint's own tensor matches its repacked copy bit for bit" {
    // `convWeight16` decodes a checkpoint-order weight to the f32 values a repack holds,
    // so the device weight and every output byte must agree; a wrong tap order or
    // temporal slice does not.
    const gpa = testing.allocator;
    const io = testing.io;
    const ctx = try gpuCtx(gpa, io);
    defer ctx.deinit();
    const dtype = @import("tp_core").dtype;

    var prng = std.Random.DefaultPrng.init(41);
    const rand = prng.random();
    const h: usize = 9;
    const w: usize = 7;
    const co: usize = 128;
    var patch: DeviceBuffer = none;
    defer ctx.tensorDestroy(&patch);
    var cases: usize = 0;
    for ([_]usize{ 32, 12 }) |ci| {
        const tw = try gpa.alloc(f32, co * ci * 9);
        defer gpa.free(tw);
        for (tw) |*v| v.* = rand.floatNorm(f32) * 0.1;
        const bias = try gpa.alloc(f32, co);
        defer gpa.free(bias);
        for (bias) |*v| v.* = rand.floatNorm(f32);
        const in = try gpa.alloc(f32, h * w * ci);
        defer gpa.free(in);
        for (in) |*v| v.* = rand.floatNorm(f32);
        var src: DeviceBuffer = none;
        defer ctx.tensorDestroy(&src);
        try upload(ctx, &src, in);
        for ([_]@import("tp_core").dtype.DType{ .f32, .f16, .bf16 }) |wdt| {
            const bytes = try gpa.alloc(u8, wdt.storageBytes(tw.len));
            defer gpa.free(bytes);
            switch (wdt) {
                .f32 => @memcpy(bytes, std.mem.sliceAsBytes(tw)),
                .f16 => dtype.f32ToF16Row(tw, bytes),
                else => dtype.f32ToBf16Row(tw, bytes),
            }
            const torch: Conv2d = .{ .w = .{ .torch = .{ .w = ops.matmul.Weight.init(bytes, wdt, co, ci * 9) } }, .b = bias, .co = co, .ci = ci, .k = 3, .pad = 1 };
            const krsc = try torch.materialize(gpa);
            defer gpa.free(krsc);
            var repacked = torch;
            repacked.w = .{ .krsc = krsc };
            for ([3]SampleMode{ .stride1, .stride2, .upsample2x }) |mode| {
                var outs: [2][]u8 = undefined;
                for ([2]Conv2d{ torch, repacked }, 0..) |cv0, i| {
                    var cv = cv0;
                    if (mode == .stride2) cv.stride = 2;
                    const oh: usize = switch (mode) {
                        .stride1 => h,
                        .stride2 => (h + 1) / 2,
                        .upsample2x => 2 * h,
                    };
                    const ow: usize = switch (mode) {
                        .stride1 => w,
                        .stride2 => (w + 1) / 2,
                        .upsample2x => 2 * w,
                    };
                    var dst: DeviceBuffer = none;
                    defer ctx.tensorDestroy(&dst);
                    try ctx.ensureDeviceBuffer(&dst, oh * ow * co * 4);
                    try conv2d(ctx, &patch, &dst, &src, h, w, cv, mode, null, null, 1.0);
                    outs[i] = try gpa.alloc(u8, oh * ow * co * 4);
                    try ctx.tensorDownload(dst, outs[i]);
                }
                defer for (outs) |o| gpa.free(o);
                errdefer std.debug.print("ci {d} weight {t} mode {t}: the checkpoint-order conv differs from its repack\n", .{ ci, wdt, mode });
                try testing.expectEqualSlices(u8, outs[1], outs[0]);
                cases += 1;
            }
        }
    }
    try testing.expectEqual(@as(usize, 2 * 3 * 3), cases);
}
