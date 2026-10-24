//! CUDA-backend LoRA sidecar apply, the device twin of `lora.Target.applyHost`.
//!
//! `dst[m][n] += strength * scale * B[row0 .. row0+n] (A x)`: the A GEMM into a rank
//! scratch, then the B GEMM accumulating straight onto `dst`. Both go through
//! `opGemmBf16`, so the factors live in the same pointer-keyed device weight
//! cache as the trunk's own weights and the VRAM arbiter sees them.
//!
//! ⚠️ **The accumulate is not free and was the larger half of the cost.** Writing
//! the delta to a scratch plane and adding it is three extra passes over an f32
//! `[m][n]` plane, ~70 GB per step at H3's widths, and measured at roughly twice
//! the two GEMMs it served. `opGemm16Acc` folds it into cuBLASLt's epilogue;
//! `Workspace.fused` says whether this backend can, and the scratch plane only
//! exists when it cannot. See that function for the numbers.
//!
//! Two things shape the interface:
//!
//! 1. **The caller asks for an output ROW RANGE, not a whole target.** A device
//!    trunk holds a fused linear's output as separate buffers (H3 splits qkv
//!    into three planes and `fc1` into gate and value), so a `[m][2 * ffn]`
//!    delta would have to be de-interleaved to be usable. Rows of `B` are the
//!    output's columns, so asking for `B`'s rows `[0, ffn)` produces exactly the
//!    gate delta with no shuffle. This is also why `applyRange` does not care
//!    whether the target was split into block-diagonal factors: a range either
//!    is a factor or is a row slice of one.
//! 2. **A factor keeps its file's dtype** (f32, f16 or bf16), and `lora.device_math`
//!    picks the arithmetic. `op16` converts it once on the device (`weight16`, cached)
//!    to the 16-bit operand width, which is free for a file already at that width;
//!    `f32` runs both GEMMs as true f32 on cuBLASLt (`weight32`). The one other route
//!    is `How.a_i8`: an int8 copy of A made at attach, which runs on the int8
//!    activation a base GEMM already staged.

const std = @import("std");
const lora = @import("lora.zig");
const cuda = @import("tp_gpu").cuda;

const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;
const Target = lora.Target;

/// DIAGNOSTIC ONLY, and it produces a WRONG render: run both sidecar GEMMs but
/// skip the accumulate onto `dst`. The GEMMs and the accumulate have very
/// different profiles (compute against four passes over an f32 `[m][n]` plane),
/// so a cost figure for the sidecar as a whole cannot say which to attack. This
/// removes exactly one of them. Only meaningful with `bench_no_fuse`, since the
/// fused path has no separable accumulate.
pub var bench_skip_add: bool = false;

/// A/B the fused accumulate against materialize-then-add on a backend that can do
/// both. Correct either way; only the traffic differs.
pub var bench_no_fuse: bool = false;

/// DIAGNOSTIC, renders garbage: skip the A GEMM everywhere, or only on the SD linears
/// a shared A GEMM would absorb (attn1's k and v, attn2's v, one A left per input).
/// The second is the ceiling on what sharing can save.
pub var bench_skip_a: enum { none, all, shared } = .none;

fn skipA(tag: []const u8) bool {
    return switch (bench_skip_a) {
        .none => false,
        .all => true,
        .shared => for ([_][]const u8{ "attn1.to_k.weight", "attn1.to_v.weight", "attn2.to_v.weight" }) |s| {
            if (std.mem.endsWith(u8, tag, s)) break true;
        } else false,
    };
}

/// Whether every factor of `t` can run here.
///
/// A factor is f32, f16 or bf16 (`lora.loadTarget` widens anything else). Every
/// width must be a multiple of 32: the rank is the A GEMM's output and the B GEMM's
/// contraction, and `Sidecar.load` pads it to `lora.rank_align`, so every file rank
/// clears it; a hand-built target may not. B's output width is the base linear's
/// own, and SD1.5's 320-wide linears belong to the SD sidecar.
pub fn supported(t: *const Target) bool {
    for (t.kron) |k| if (k.w2.dtype != .bf16 and k.w2.dtype != .f16) return false;
    for (t.diffs) |d| if (d.w.dtype != .bf16 and d.w.dtype != .f16) return false;
    for (t.factors) |f| {
        if (!denseFactor(f.a.dtype) or !denseFactor(f.b.dtype)) return false;
        if (f.a.rows % lora.rank_align != 0) return false;
        if (f.a.cols % 32 != 0) return false; // A's contraction
        if (f.b.rows % 32 != 0) return false; // B's output width
    }
    return true;
}

fn denseFactor(dt: @import("tp_core").dtype.DType) bool {
    return dt == .f32 or dt == .f16 or dt == .bf16;
}

/// `lora.device_math` as this backend runs it: true f32 is cuBLASLt's, so the
/// hand-PTX arm keeps 16-bit operands.
fn mathOn(be: *const Backend) lora.Math {
    return if (be.kernels == .libs) lora.device_math else .op16;
}

/// `lo` elements one apply of a rank-`rank` factor over `m` rows needs. The int8 A
/// GEMM stores `align(m, 128)` rows of `align(rank, 128)` columns, its A padded to
/// the tile by `lora.quantizeA`; the bf16 one writes less.
pub fn loElems(m: usize, rank: usize) usize {
    return std.mem.alignForward(usize, m, 128) * std.mem.alignForward(usize, rank, 128);
}

/// Scratch for one apply. Sized from the largest shape a render will ask for,
/// once, rather than per block.
pub const Workspace = struct {
    /// `[m][rank]` f32.
    lo: Buf = .{},
    /// `[m][n]` f32, the delta before it is accumulated. Unallocated when the
    /// accumulate is fused into the B GEMM, which is where it would have been the
    /// biggest buffer here (205 MB at H3's native canvas).
    hi: Buf = .{},
    lo_elems: usize = 0,
    hi_elems: usize = 0,
    /// Whether the B GEMM accumulates onto `dst` itself. Decided once from the
    /// backend's kernel arm, so it cannot change under a render.
    fused: bool = false,
    /// False when `lo`/`hi` are the backend's own shared buffers (`borrowed`),
    /// so `deinit` leaves them alone. A borrowed workspace is a value, not a
    /// resource.
    owned: bool = true,

    pub fn init(be: *Backend, lo_elems: usize, hi_elems: usize) !Workspace {
        const fused = be.kernels == .libs and !bench_no_fuse;
        var ws: Workspace = .{
            .lo_elems = lo_elems,
            .hi_elems = if (fused) 0 else hi_elems,
            .fused = fused,
        };
        errdefer ws.deinit(be);
        ws.lo = try be.tensorCreate(lo_elems * 4);
        if (!fused) ws.hi = try be.tensorCreate(hi_elems * 4);
        return ws;
    }

    pub fn deinit(self: *Workspace, be: *Backend) void {
        if (self.owned) {
            be.tensorDestroy(&self.lo);
            be.tensorDestroy(&self.hi);
        }
        self.* = .{};
    }
};

/// A workspace over the backend's SHARED sidecar scratch, grown to fit.
///
/// For callers that do not know their widest factor before the first block, i.e.
/// every family that goes through `lin_cuda` rather than sizing its own. Growth
/// syncs the stream, so `lin_cuda.presizeLora` reaches the final size up front
/// and this then only ever hands the buffers back.
pub fn borrowed(be: *Backend, lo_elems: usize, hi_elems: usize) !Workspace {
    const fused = be.kernels == .libs and !bench_no_fuse;
    try be.ensureDeviceBuffer(&be.lora_lo, lo_elems * 4);
    if (!fused) try be.ensureDeviceBuffer(&be.lora_hi, hi_elems * 4);
    return .{
        .lo = be.lora_lo,
        .hi = if (fused) .{} else be.lora_hi,
        .lo_elems = lo_elems,
        .hi_elems = if (fused) 0 else hi_elems,
        .fused = fused,
        .owned = false,
    };
}

/// How `apply` reads its input and writes its output.
pub const How = struct {
    /// The factor's A as int8 (`lora.quantizeA`) in the basis of the int8 activation
    /// the backend has staged for `x`. The A GEMM then reads that prep, not `x`.
    a_i8: ?Weight = null,
};

/// `dst[m][n] += strength * t.scale * B[row0 .. row0+n] (A x)`.
///
/// `x` is `[m][t.in_dim]` (the SAME activation the base GEMM reads, before any int8
/// prep: the sidecar is in the unrotated space), and `dst` is `[m][n]` already holding
/// the base GEMM's output, each at its own dtype. A 16-bit `dst` is accumulated in
/// place by the GEMM on `libs`, through the f32 plane and a scaled add elsewhere.
///
/// `[row0, row0 + n)` must sit inside one factor, which `lora.Target.rowsFit`
/// checked at attach: either it is a factor, or a row slice of one.
pub fn applyRange(
    be: *Backend,
    ws: *Workspace,
    dst: Buf,
    x: Buf,
    m: usize,
    t: *const Target,
    strength: f32,
    row0: usize,
    n: usize,
) !void {
    return apply(be, ws, dst, x, m, t, strength, row0, n, .{});
}

pub fn apply(
    be: *Backend,
    ws: *Workspace,
    dst: Buf,
    x: Buf,
    m: usize,
    t: *const Target,
    strength: f32,
    row0: usize,
    n: usize,
    how: How,
) !void {
    std.debug.assert(row0 + n <= t.out_dim);
    const f = t.factorAt(row0, n) orelse {
        std.log.err("lora: {s} output rows [{d},{d}) straddle a factor boundary", .{ t.tag, row0, row0 + n });
        return error.Unsupported;
    };
    const rank = f.a.rows;
    const math = mathOn(be);
    std.debug.assert(m * rank <= ws.lo_elems);

    // A: lo[m][lo_ld] = x[m][in] @ A^T, of which the first `rank` columns are read.
    var lo_ld = rank;
    if (skipA(t.tag)) {} else if (math == .f32) {
        const x32 = if (x.dt == .f32) x else blk: {
            const c = try inPlane(be, m * t.in_dim * 4, .f32);
            try be.opCast(x, c, m * t.in_dim, 0);
            break :blk c;
        };
        try be.opMatmulF32LtDev(ws.lo, x32, m, try be.weight32(f.a.bytes, f.a.dtype, rank, t.in_dim), rank, t.in_dim, null);
    } else if (how.a_i8) |aq| {
        // `quantizeA` pads A to the int8 GEMM's 128-row tile with zero rows, so
        // `lo` comes back that wide and the B GEMM reads it at that stride.
        std.debug.assert(aq.rows == std.mem.alignForward(usize, rank, 128) and aq.cols == t.in_dim);
        std.debug.assert(be.i8_cols == t.in_dim and be.i8_m == m and be.i8_mpad * aq.rows <= ws.lo_elems);
        try be.opI8Gemm(ws.lo, aq.bytes, aq.row_scale.?, aq.rows, null);
        lo_ld = aq.rows;
    } else if (x.dt != .f32 and be.kernels != .libs) {
        try be.opMatmulHalfW(ws.lo, x, m, f.a.bytes, f.a.dtype, rank, t.in_dim, null);
    } else if (lora.operandFor(x.dt) == .f16) {
        try be.opGemmHalfIn(ws.lo, x, m, try be.weight16(f.a.bytes, f.a.dtype, false, rank, t.in_dim, 1, 1, .f16), rank, t.in_dim);
    } else {
        try be.opGemmBf16LdDev(ws.lo, x, t.in_dim, m, try be.weight16(f.a.bytes, f.a.dtype, false, rank, t.in_dim, 1, 1, .bf16), rank, t.in_dim, null);
    }
    // B: dst[m][n] += scale * lo @ B[row0 - f.out_off ..][n]^T.
    const b_rows = lora.rowSlice(f.b, row0 - f.out_off, n);
    return accumulate(be, ws, dst, ws.lo, lo_ld, m, b_rows, rank, t.scale * strength * f.gain);
}

/// `dst[m][n] += strength * delta x` for every term of `t` in output rows
/// `[row0, row0 + n)`: its factor (`apply`), its LoKr terms and its diffs. A DoRA's
/// base share is the caller's, which holds the base weight.
pub fn applyTarget(be: *Backend, ws: *Workspace, dst: Buf, x: Buf, m: usize, t: *const Target, strength: f32, row0: usize, n: usize, how: How) !void {
    if (t.factorAt(row0, n) != null or (t.kron.len == 0 and t.diffs.len == 0)) {
        try apply(be, ws, dst, x, m, t, strength, row0, n, how);
    }
    for (t.kron) |k| {
        if (k.out_off < row0 or k.out_off + k.outRows() > row0 + n) continue;
        const ok = k.w2.rows;
        // `x` read as `[m * im][in_n]` is w2's own input.
        const h = try outPlane(be, m * k.im * ok);
        try be.opMatmulBf16(h, x, m * k.im, k.w2.bytes, ok, k.w2.cols, null);
        try be.opKronMix(dst, h, try hostVec(be, k.w1), m, k.ol, ok, k.im, n, k.out_off - row0, k.im * ok, ok, strength * t.scale * k.gain);
    }
    for (t.diffs) |d| {
        if (d.out_off < row0 or d.out_off + d.w.rows > row0 + n) continue;
        const h = try outPlane(be, m * d.w.rows);
        try be.opMatmulBf16(h, x, m, d.w.bytes, d.w.rows, d.w.cols, null);
        try addPlane(be, dst, h, m, d.w.rows, n, d.out_off - row0, strength);
    }
}

/// `dst[p][off + o] += g * h[p][o]` over `rows` channels of an `[m][stride]` `dst`.
pub fn addPlane(be: *Backend, dst: Buf, h: Buf, m: usize, rows: usize, stride: usize, off: usize, g: f32) !void {
    try be.opKronMix(dst, h, try hostVec(be, &one), m, 1, rows, 1, stride, off, rows, 0, g);
}

const one = [_]f32{1};

/// `be.lora_out` as `elems` f32, grown if a caller did not presize it.
pub fn outPlane(be: *Backend, elems: usize) !Buf {
    try be.ensureDeviceBuffer(&be.lora_out, elems * 4);
    return be.lora_out.as(.f32);
}

/// `be.lora_in` as `bytes`, at dtype `dt`.
pub fn inPlane(be: *Backend, bytes: usize, dt: cuda.backend.Dt) !Buf {
    try be.ensureDeviceBuffer(&be.lora_in, bytes);
    return be.lora_in.as(dt);
}

/// A host f32 vector on the device, cached by pointer (`forEachFactorBytes` evicts it).
pub fn hostVec(be: *Backend, v: []const f32) !Buf {
    return .{ .buf = try be.smallBuffer(std.mem.sliceAsBytes(v)), .size = v.len * 4 };
}

/// The B half alone: `dst[m][n] += scale * lo[m][..rank] @ bᵀ`, `b` the `[n][rank]`
/// rows of B this output takes, `lo` f32 read at stride `lo_ld`, `dst` at its own
/// dtype. For a caller that made `lo` itself, a 3x3 conv's A being a convolution
/// rather than a GEMM.
pub fn accumulate(be: *Backend, ws: *Workspace, dst: Buf, lo: Buf, lo_ld: usize, m: usize, b: Weight, rank: usize, scale: f32) !void {
    const n = b.rows;
    std.debug.assert(b.cols == rank);
    if (mathOn(be) == .f32 and lo_ld == rank) {
        const w = try be.weight32(b.bytes, b.dtype, n, rank);
        if (dst.dt == .f32) return be.opGemmF32Acc(dst, lo, m, w, n, rank, scale, 1.0);
        const h = try outPlane(be, m * n);
        try be.opGemmF32Acc(h, lo, m, w, n, rank, 1.0, 0.0);
        if (bench_skip_add) return;
        return be.opAddScaled(dst, h, m * n, scale);
    }
    // A 16-bit `dst` sets the width: cuBLASLt has no bf16 GEMM writing f16.
    if (ws.fused) return if (dst.dt != .f32)
        be.opGemmHalfAcc(dst, lo, lo_ld, m, try be.weight16(b.bytes, b.dtype, false, n, rank, 1, 1, dst.dt), n, rank, scale)
    else
        be.opGemm16Acc(dst, lo, lo_ld, m, try be.weight16(b.bytes, b.dtype, false, n, rank, 1, 1, lora.operandFor(.f32)), n, rank, scale);
    std.debug.assert(m * n <= ws.hi_elems);
    try be.opGemmBf16LdDev(ws.hi, lo, lo_ld, m, try be.weight16(b.bytes, b.dtype, false, n, rank, 1, 1, .bf16), n, rank, null);
    if (bench_skip_add) return;
    try be.opAddScaled(dst, ws.hi, m * n, scale);
}

// --- tests -----------------------------------------------------------------

const testing = std.testing;
const ops = @import("tp_ops");
const Weight = ops.matmul.Weight;

fn bf16Weight(bytes: []const u8, rows: usize, cols: usize) Weight {
    return Weight.init(bytes, .bf16, rows, cols);
}

test "a range resolves to the factor that contains it, and a straddle does not" {
    // The device asks for a fused linear's pieces; a range that spanned two
    // factors would silently read the wrong B rows if this returned the first
    // match instead of refusing.
    const a_bytes: [(128 * 32 * 2)]u8 = @splat(0);
    const b_bytes: [(64 * 128 * 2)]u8 = @splat(0);
    var factors: [3]lora.Factor = undefined;
    for (&factors, 0..) |*f, i| f.* = .{
        .a = bf16Weight(&a_bytes, 128, 32),
        .b = bf16Weight(&b_bytes, 64, 128),
        .out_off = i * 64,
    };
    const t: Target = .{ .factors = &factors, .in_dim = 32, .out_dim = 192, .scale = 0.0625, .tag = "qkv" };

    try testing.expectEqual(@as(usize, 0), t.factorAt(0, 64).?.out_off);
    try testing.expectEqual(@as(usize, 64), t.factorAt(64, 64).?.out_off);
    try testing.expectEqual(@as(usize, 128), t.factorAt(128, 64).?.out_off);
    // A sub-range of one factor is fine (that is the fc1 gate/value split).
    try testing.expectEqual(@as(usize, 64), t.factorAt(96, 32).?.out_off);
    // Straddling two is not.
    try testing.expectEqual(@as(?lora.Factor, null), t.factorAt(32, 64));
}

test "device support is refused by shape rather than discovered at a launch" {
    // A width off the GEMM's slab makes its tiles read past the factor. Refusing
    // here keeps that a fallback instead of a bad device access.
    const ok_a: [(128 * 64 * 2)]u8 = @splat(0);
    const ok_b: [(256 * 128 * 2)]u8 = @splat(0);
    var f: lora.Factor = .{
        .a = bf16Weight(&ok_a, 128, 64),
        .b = bf16Weight(&ok_b, 256, 128),
        .out_off = 0,
    };
    var t: Target = .{ .factors = &.{f}, .in_dim = 64, .out_dim = 256, .scale = 1.0, .tag = "ok" };
    try testing.expect(supported(&t));

    // Any rank on the 32 slab runs, 64 and 96 included.
    for ([_]usize{ 32, 64, 96 }) |r| {
        f = .{ .a = bf16Weight(ok_a[0 .. r * 64 * 2], r, 64), .b = bf16Weight(ok_b[0 .. 256 * r * 2], 256, r), .out_off = 0 };
        t.factors = &.{f};
        try testing.expect(supported(&t));
    }
    // An unpadded rank 8 does not; `Sidecar.load` never hands one over.
    f = .{ .a = bf16Weight(ok_a[0 .. 8 * 64 * 2], 8, 64), .b = bf16Weight(ok_b[0 .. 256 * 8 * 2], 256, 8), .out_off = 0 };
    t.factors = &.{f};
    try testing.expect(!supported(&t));
    // SD1.5's 320-wide output runs: both B GEMMs take any multiple of 32.
    f = .{ .a = bf16Weight(ok_a[0 .. 32 * 64 * 2], 32, 64), .b = bf16Weight(ok_b[0 .. 320 * 32 * 2], 320, 32), .out_off = 0 };
    t.factors = &.{f};
    try testing.expect(supported(&t));
    // A width off 32 does not.
    f = .{ .a = bf16Weight(ok_a[0 .. 32 * 64 * 2], 32, 64), .b = bf16Weight(ok_b[0 .. 300 * 32 * 2], 300, 32), .out_off = 0 };
    t.factors = &.{f};
    try testing.expect(!supported(&t));

    // An f32 factor runs (`weight16` / `weight32` read it); a raw fp8 one, which
    // `lora.loadTarget` always widens, has no GEMM here.
    const f32_a: [(128 * 64 * 4)]u8 = @splat(0);
    f = .{ .a = Weight.init(&f32_a, .f32, 128, 64), .b = bf16Weight(&ok_b, 256, 128), .out_off = 0 };
    t.factors = &.{f};
    try testing.expect(supported(&t));
    f = .{ .a = Weight.init(ok_a[0 .. 128 * 64], .f8_e4m3, 128, 64), .b = bf16Weight(&ok_b, 256, 128), .out_off = 0 };
    t.factors = &.{f};
    try testing.expect(!supported(&t));
}
