//! CUDA-backend LoRA sidecar apply, the device twin of `lora.Target.applyHost`.
//!
//! `dst[m][n] += scale * B[row0 .. row0+n] (A x)`: the A GEMM into a rank
//! scratch, then the B GEMM accumulating straight onto `dst`. Both go through
//! `opGemmBf16`, so the factors live in the same pointer-keyed device weight
//! cache as the trunk's own weights and the VRAM arbiter sees them.
//!
//! ⚠️ **The accumulate is not free and was the larger half of the cost.** Writing
//! the delta to a scratch plane and adding it is three extra passes over an f32
//! `[m][n]` plane, ~70 GB per step at H3's widths, and measured at roughly twice
//! the two GEMMs it served. `opGemmBf16Acc` folds it into cuBLASLt's epilogue;
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
//! 2. **bf16 factors only.** `opGemmBf16` reads the raw 16-bit weight as its B
//!    operand. `supported` refuses anything else by name rather than at a
//!    launch, and the host path still works, so a non-bf16 LoRA is slow, not
//!    broken.

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

/// Whether every factor of `t` can run here.
///
/// `opGemmBf16` takes the raw weight as its GEMM B operand, which fixes the tile
/// alignment: the output width must be a multiple of 128 and the contracted
/// width a multiple of 32. Every real LoRA shape clears both (rank 128 or 384
/// against widths that are already multiples of 128), but a rank-64 or rank-96
/// factor would not, and it must fall back rather than launch a GEMM whose tiles
/// read past the weight.
pub fn supported(t: *const Target) bool {
    for (t.factors) |f| {
        if (f.a.dtype != .bf16 or f.b.dtype != .bf16) return false;
        const rank = f.a.rows;
        if (rank % 128 != 0) return false; // A's output width, B's contraction
        if (f.a.cols % 32 != 0) return false; // A's contraction
        if (f.b.rows % 128 != 0) return false; // B's output width
    }
    return true;
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
        be.tensorDestroy(&self.lo);
        be.tensorDestroy(&self.hi);
        self.* = .{};
    }
};

/// `dst[m][n] += scale * B[row0 .. row0+n] (A x)`.
///
/// `x` is `[m][t.in_dim]` f32 (the SAME activation the base GEMM reads, before
/// any int8 prep: the sidecar is in the unrotated space). `dst` is `[m][n]` f32,
/// already holding the base GEMM's output.
///
/// `[row0, row0 + n)` must sit inside one factor. It always does in practice:
/// the ranges a trunk asks for are the fused linear's own pieces, and a
/// block-diagonal split cuts on exactly those boundaries.
pub fn applyRange(
    be: *Backend,
    ws: *Workspace,
    dst: Buf,
    x: Buf,
    m: usize,
    t: *const Target,
    row0: usize,
    n: usize,
) !void {
    std.debug.assert(row0 + n <= t.out_dim);
    const f = factorFor(t, row0, n) orelse {
        std.log.err("lora: {s} output rows [{d},{d}) straddle a factor boundary", .{ t.tag, row0, row0 + n });
        return error.Unsupported;
    };
    const rank = f.a.rows;
    // `supported` is the gate; this is the assert that the caller ran it. The
    // byte arithmetic below hardcodes 16-bit elements.
    std.debug.assert(f.a.dtype == .bf16 and f.b.dtype == .bf16);
    std.debug.assert(m * rank <= ws.lo_elems);
    std.debug.assert(ws.fused or m * n <= ws.hi_elems);

    // A: lo[m][rank] = x[m][in] @ A^T.
    try be.opGemmBf16(ws.lo, x, m, f.a.bytes, rank, t.in_dim, null);
    // B: dst[m][n] += scale * lo @ B[row0 - f.out_off ..][n]^T. Rows of a
    // row-major weight are contiguous, so the slice is a view.
    const b_off = (row0 - f.out_off) * rank * 2;
    const b_rows = f.b.bytes[b_off..][0 .. n * rank * 2];
    if (ws.fused) {
        try be.opGemmBf16Acc(dst, ws.lo, m, b_rows, n, rank, f.scale);
        return;
    }
    try be.opGemmBf16(ws.hi, ws.lo, m, b_rows, n, rank, null);
    if (bench_skip_add) return;
    try be.opAddScaled(dst, ws.hi, m * n, f.scale);
}

/// The factor whose output range contains `[row0, row0 + n)`, or null if it
/// spans two.
fn factorFor(t: *const Target, row0: usize, n: usize) ?lora.Factor {
    for (t.factors) |f| {
        if (row0 >= f.out_off and row0 + n <= f.out_off + f.b.rows) return f;
    }
    return null;
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
    const a_bytes = [_]u8{0} ** (128 * 32 * 2);
    const b_bytes = [_]u8{0} ** (64 * 128 * 2);
    var factors: [3]lora.Factor = undefined;
    for (&factors, 0..) |*f, i| f.* = .{
        .a = bf16Weight(&a_bytes, 128, 32),
        .b = bf16Weight(&b_bytes, 64, 128),
        .out_off = i * 64,
        .scale = 0.0625,
    };
    const t: Target = .{ .factors = &factors, .in_dim = 32, .out_dim = 192, .tag = "qkv" };

    try testing.expectEqual(@as(usize, 0), factorFor(&t, 0, 64).?.out_off);
    try testing.expectEqual(@as(usize, 64), factorFor(&t, 64, 64).?.out_off);
    try testing.expectEqual(@as(usize, 128), factorFor(&t, 128, 64).?.out_off);
    // A sub-range of one factor is fine (that is the fc1 gate/value split).
    try testing.expectEqual(@as(usize, 64), factorFor(&t, 96, 32).?.out_off);
    // Straddling two is not.
    try testing.expectEqual(@as(?lora.Factor, null), factorFor(&t, 32, 64));
}

test "device support is refused by shape rather than discovered at a launch" {
    // `opGemmBf16` reads the raw weight as its GEMM B operand, so a rank that is
    // not a multiple of 128 makes its tiles read past the factor. Refusing here
    // keeps that a fallback instead of a bad device access.
    const ok_a = [_]u8{0} ** (128 * 64 * 2);
    const ok_b = [_]u8{0} ** (256 * 128 * 2);
    var f: lora.Factor = .{
        .a = bf16Weight(&ok_a, 128, 64),
        .b = bf16Weight(&ok_b, 256, 128),
        .out_off = 0,
        .scale = 1.0,
    };
    var t: Target = .{ .factors = &.{f}, .in_dim = 64, .out_dim = 256, .tag = "ok" };
    try testing.expect(supported(&t));

    // rank 64: A's output width and B's contraction both miss the tile.
    const r64_a = [_]u8{0} ** (64 * 64 * 2);
    const r64_b = [_]u8{0} ** (256 * 64 * 2);
    f = .{ .a = bf16Weight(&r64_a, 64, 64), .b = bf16Weight(&r64_b, 256, 64), .out_off = 0, .scale = 1.0 };
    t.factors = &.{f};
    try testing.expect(!supported(&t));

    // f32 factors: no bf16 GEMM to read them.
    const f32_a = [_]u8{0} ** (128 * 64 * 4);
    var fw = bf16Weight(&ok_a, 128, 64);
    fw = Weight.init(&f32_a, .f32, 128, 64);
    f = .{ .a = fw, .b = bf16Weight(&ok_b, 256, 128), .out_off = 0, .scale = 1.0 };
    t.factors = &.{f};
    try testing.expect(!supported(&t));
}
