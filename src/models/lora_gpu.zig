//! Vulkan LoRA sidecar apply, the twin of `lora_cuda`:
//! `dst[m][n] += strength * scale * B[row0 .. row0+n] (A x)` as two bf16 cooperative
//! GEMMs, the second accumulating straight into `dst`. The factors live in the
//! Context's weight cache like any weight, so a stack swapped on a live session must
//! evict them by range before they are freed (`Context.evictWeightRange`).

const std = @import("std");
const lora = @import("lora.zig");
const gpu = @import("tp_gpu").context;

const Buf = gpu.DeviceBuffer;
const Target = lora.Target;
const Weight = @import("tp_ops").matmul.Weight;

/// Whether every term of `t` can run here: a factor is f32, f16 or bf16, a linear
/// LoKr's `w2` and a linear diff raw bf16, and a conv's f16 convolutions.
pub fn supported(t: *const Target) bool {
    for (t.factors) |f| if (!denseFactor(f.a.dtype) or !denseFactor(f.b.dtype)) return false;
    for (t.kron) |k| if (k.w2.dtype != .bf16 and k.w2.dtype != .f16) return false;
    for (t.diffs) |d| if (d.w.dtype != .bf16 and d.w.dtype != .f16) return false;
    return true;
}

const none: Buf = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 };

/// The sidecar's scratch, grown to fit: the A GEMM's `[m][rank]` f32 output (`lo`), a
/// LoKr's per-group product or a diff's or DoRA's base product (`out`, f32), and DoRA's
/// rescaled input or LoKr's stacked groups (`in`). Growing inside a recording batch
/// flushes it, so a caller sizes for its widest apply before the first block.
pub const Scratch = struct {
    lo: Buf = none,
    out: Buf = none,
    in: Buf = none,

    pub fn ensure(self: *Scratch, ctx: *gpu.Context, lo_elems: usize) !void {
        try ctx.ensureDeviceBuffer(&self.lo, lo_elems * 4);
    }

    pub fn outPlane(self: *Scratch, ctx: *gpu.Context, elems: usize) !Buf {
        try ctx.ensureDeviceBuffer(&self.out, elems * 4);
        var b = self.out;
        b.dt = .f32;
        return b;
    }

    pub fn inPlane(self: *Scratch, ctx: *gpu.Context, bytes: usize, dt: gpu.Dt) !Buf {
        try ctx.ensureDeviceBuffer(&self.in, bytes);
        var b = self.in;
        b.dt = dt;
        return b;
    }

    pub fn deinit(self: *Scratch, ctx: *gpu.Context) void {
        ctx.tensorDestroy(&self.lo);
        ctx.tensorDestroy(&self.out);
        ctx.tensorDestroy(&self.in);
    }
};

/// `dst[m][n] += strength * delta x` for every term of `t` in output rows
/// `[row0, row0 + n)`, the twin of `lora_cuda.applyTarget`. A DoRA's base share is the
/// caller's.
pub fn applyTarget(ctx: *gpu.Context, s: *Scratch, zb: []const f32, dst: Buf, x: Buf, m: usize, t: *const Target, strength: f32, row0: usize, n: usize) !void {
    if (t.factorAt(row0, n) != null or (t.kron.len == 0 and t.diffs.len == 0)) {
        try apply(ctx, s, zb, dst, x, m, t, strength, row0, n);
    }
    for (t.kron) |k| {
        if (k.out_off < row0 or k.out_off + k.outRows() > row0 + n) continue;
        const ok = k.w2.rows;
        std.debug.assert(ok <= zb.len);
        const h = try s.outPlane(ctx, m * k.im * ok);
        // `x` read as `[m * im][in_n]` is w2's own input.
        try gemm16(ctx, h, x, m * k.im, k.w2.bytes, ok, k.w2.cols, zb);
        try kronMix(ctx, dst, h, try hostVec(ctx, k.w1), m, k.ol, ok, k.im, n, k.out_off - row0, k.im * ok, ok, strength * t.scale * k.gain);
    }
    for (t.diffs) |d| {
        if (d.out_off < row0 or d.out_off + d.w.rows > row0 + n) continue;
        std.debug.assert(d.w.rows <= zb.len);
        const h = try s.outPlane(ctx, m * d.w.rows);
        try gemm16(ctx, h, x, m, d.w.bytes, d.w.rows, d.w.cols, zb);
        try addPlane(ctx, dst, h, m, d.w.rows, n, d.out_off - row0, strength);
    }
}

const one = [_]f32{1};

/// A host f32 vector on the device, cached by pointer (`forEachFactorBytes` evicts it).
pub fn hostVec(ctx: *gpu.Context, v: []const f32) !Buf {
    return .{ .buf = try ctx.smallBuffer(std.mem.sliceAsBytes(v)), .mem = .null_handle, .size = v.len * 4 };
}

/// `dual/elt.zig` `kronMix` onto `y` at its dtype.
pub fn kronMix(ctx: *gpu.Context, y: Buf, h: Buf, w1: Buf, rows: usize, ol: usize, ok: usize, im: usize, y_stride: usize, out_off: usize, h_row: usize, h_group: usize, g: f32) !void {
    const total = rows * ol * ok;
    try ctx.opElt(gpu.eltAt("kron_mix", y.dt), y, h, w1, null, .{
        .u0 = @intCast(total),
        .u1 = @intCast(ol),
        .u2 = @intCast(ok),
        .u3 = @intCast(im),
        .u4 = @intCast(y_stride),
        .u5 = @intCast(out_off),
        .u6 = @intCast(h_group),
        .f0 = g,
        .f1 = @bitCast(@as(u32, @intCast(h_row))),
    }, total, 1, 1);
}

/// `dst[p][off + o] += g * h[p][o]` over `rows` channels of an `[m][stride]` `dst`.
pub fn addPlane(ctx: *gpu.Context, dst: Buf, h: Buf, m: usize, rows: usize, stride: usize, off: usize, g: f32) !void {
    try kronMix(ctx, dst, h, try hostVec(ctx, &one), m, 1, rows, 1, stride, off, rows, 0, g);
}

/// Groups `[j0, j0 + im)` of `src` (`[h][w][ci]`) stacked as `[im][h + 1][w][in_n]` with
/// a zero row closing each (`dual/elt.zig` `kronStack`).
pub fn kronStack(ctx: *gpu.Context, dst: Buf, src: Buf, im: usize, h: usize, w: usize, in_n: usize, ci: usize, j0: usize) !void {
    std.debug.assert(dst.dt == src.dt);
    const total = im * (h + 1) * w * in_n;
    try ctx.opElt(gpu.eltAt("kron_stack", src.dt), dst, src, null, null, .{
        .u0 = @intCast(total),
        .u1 = @intCast(w),
        .u2 = @intCast(h),
        .u3 = @intCast(in_n),
        .u4 = @intCast(ci),
        .u5 = @intCast(im),
        .u6 = @intCast(j0),
    }, total, 1, 1);
}

/// An output-axis DoRA's rescale of the base product in `y`, `bias` what the base added.
pub fn doraRows(ctx: *gpu.Context, y: Buf, ratio: Buf, bias: ?Buf, rows: usize, width: usize, s: f32) !void {
    const total = rows * width;
    try ctx.opElt(gpu.eltAt("dora_rows", y.dt), y, ratio, bias orelse ratio, null, .{
        .u0 = @intCast(total),
        .u1 = @intCast(width),
        .u2 = @intFromBool(bias != null),
        .f0 = s,
    }, total, 1, 1);
}

/// `dst[p][j] = (d[j] - 1) * src[p][j]` into f32.
pub fn doraCols(ctx: *gpu.Context, dst: Buf, src: Buf, d: Buf, rows: usize, width: usize) !void {
    std.debug.assert(dst.dt == .f32);
    const total = rows * width;
    try ctx.opElt(gpu.eltAt("dora_cols", src.dt), dst, src, d, null, .{ .u0 = @intCast(total), .u1 = @intCast(width) }, total, 1, 1);
}

/// `dst[m][n] += strength * t.scale * B[row0 .. row0+n] (A x)`, `x` the activation the
/// base GEMM read and `dst` its output, each at its own dtype. `zb` is a zero bias at
/// least as wide as the rank and `n`, passed WHOLE (`smallBuffer` caches by host
/// pointer).
///
/// A factor keeps its file's dtype and `lora.device_math` picks the arithmetic:
/// `op16` runs the coop GEMM at `lora.operandFor`'s width (`weightBufferCoop16`, free
/// for a file already at it), `f32` the f32 GEMM on an f32 copy of `x`.
pub fn apply(ctx: *gpu.Context, s: *Scratch, zb: []const f32, dst: Buf, x: Buf, m: usize, t: *const Target, strength: f32, row0: usize, n: usize) !void {
    std.debug.assert(row0 + n <= t.out_dim);
    const f = t.factorAt(row0, n) orelse {
        std.log.err("lora: {s} output rows [{d},{d}) straddle a factor boundary", .{ t.tag, row0, row0 + n });
        return error.Unsupported;
    };
    const rank = f.a.rows;
    std.debug.assert(rank <= zb.len and n <= zb.len);
    try s.ensure(ctx, m * rank);
    if (lora.device_math == .f32) {
        const x32 = if (x.dt == .f32) x else blk: {
            const c = try s.inPlane(ctx, m * t.in_dim * 4, .f32);
            try ctx.opElt(gpu.eltCopy(x.dt, .f32), x, c, null, null, .{ .u0 = @intCast(m * t.in_dim) }, m * t.in_dim, 1, 1);
            break :blk c;
        };
        try ctx.opMatmulF32Dense(s.lo, x32, m, f.a.bytes, f.a.dtype, rank, t.in_dim);
    } else {
        const bf = operandBf16(ctx, x.dt);
        try ctx.opMatmulCoopW(s.lo, x, m, try ctx.weightBufferCoop16(f.a.bytes, f.a.dtype, rank, t.in_dim, bf), rank, t.in_dim, zb, bf);
    }
    try accumulate(ctx, s, dst, m, lora.rowSlice(f.b, row0 - f.out_off, n), rank, t.scale * strength * f.gain);
}

/// The B half alone: `dst[m][n] += scale * s.lo[m][rank] @ bᵀ`, `b` the `[n][rank]`
/// rows of B this output takes, for a caller that made `s.lo` itself (a 3x3 conv's A
/// is a convolution, not a GEMM).
pub fn accumulate(ctx: *gpu.Context, s: *Scratch, dst: Buf, m: usize, b: Weight, rank: usize, scale: f32) !void {
    const n = b.rows;
    std.debug.assert(b.cols == rank);
    try s.ensure(ctx, m * rank);
    if (lora.device_math == .f32) {
        const h = try s.outPlane(ctx, m * n);
        var lo = s.lo;
        lo.dt = .f32;
        try ctx.opMatmulF32Dense(h, lo, m, b.bytes, b.dtype, n, rank);
        return ctx.opElt(gpu.eltAt("add_scaled", dst.dt), dst, h, null, null, .{ .u0 = @intCast(m * n), .f0 = scale }, m * n, 1, 1);
    }
    const bf = operandBf16(ctx, dst.dt);
    try ctx.opMatmulCoopAccW(dst, s.lo, m, try ctx.weightBufferCoop16(b.bytes, b.dtype, n, rank, bf), n, rank, bf, scale);
}

/// Whether a coop GEMM whose 16-bit-facing side is `side` runs bf16 operands
/// (`lora.operandFor`). f16 coop is every device's; bf16 is taken where it exists.
fn operandBf16(ctx: *const gpu.Context, side: gpu.Dt) bool {
    return lora.operandFor(side) == .bf16 and ctx.coopBf16();
}

fn denseFactor(dt: @import("tp_core").dtype.DType) bool {
    return dt == .f32 or dt == .f16 or dt == .bf16;
}

/// The coop GEMM over a raw bf16 `[rows][cols]` weight (a LoKr's `w2`, a diff).
fn gemm16(ctx: *gpu.Context, y: Buf, x: Buf, m: usize, w: []const u8, rows: usize, cols: usize, zb: []const f32) !void {
    const bf = ctx.coopBf16();
    try ctx.opMatmulCoopW(y, x, m, try ctx.weightBufferCoop16(w, .bf16, rows, cols, bf), rows, cols, zb, bf);
}
