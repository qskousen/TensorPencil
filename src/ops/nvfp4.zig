//! ComfyUI's NVFP4 weight format: 4-bit E2M1 floats with an fp8 per-16-block scale.
//!
//! Reference: `comfy_kitchen/tensor/nvfp4.py` plus
//! `backends/eager/quantization.py::dequantize_nvfp4` and `float_utils.{to,from}_blocked`.
//! A layer ships:
//!
//! | tensor | dtype | |
//! |---|---|---|
//! | `weight` | U8 `[rows, cols/2]` | two E2M1 codes per byte |
//! | `weight_scale` | F8_E4M3 | per-16-element block scale, **swizzled** |
//! | `weight_scale_2` | F32 scalar | per-tensor scale |
//! | `input_scale` | F32 scalar | static ACTIVATION scale — Blackwell-only, unused here |
//!
//! and decodes as
//!
//! ```
//! total[row, blk] = weight_scale_2 * fp8(block_scale[row, blk])
//! value           = E2M1[nibble] * total[row, blk]
//! ```
//!
//! **This is a weight-only format on anything below Blackwell, and that is ComfyUI's own
//! behaviour, not a shortcut here.** NVFP4's tensor cores are sm_100+
//! (`TensorCoreNVFP4Layout.MIN_SM_VERSION = (10, 0)`); on an older card
//! `pick_operations` moves `nvfp4` from "native ops" to what ComfyUI's log calls
//! "emulated ops", which sets `_full_precision_mm` and makes every linear dequantize the
//! weight to the compute dtype per call before a normal GEMM. The weight stays 4-bit
//! resident. This engine does the same thing: `Weight.dtype == .nvfp4` keeps the packed
//! bytes, and each consumer decodes on demand — the CPU GEMM into its f32 panel, the GPU
//! backends into an f16 scratch feeding the existing f16 tensor-core GEMM.
//!
//! ⚠️ **Three conventions differ from every other 4-bit format in this engine**, and each
//! is a silent wrong answer rather than an error:
//!
//! 1. **Element 2k is the HIGH nibble** (`hi_first = True`). `.i4` convrot and `.w4a8`
//!    are both low-nibble-first. Swapping them permutes adjacent weight pairs, which is
//!    rms-preserving — no magnitude check can see it. `nibble` below owns the choice.
//! 2. **The block scales are SWIZZLED** into cuBLAS's tiled layout, and on disk they
//!    still carry the LOGICAL `[rows, cols/16]` shape — so nothing in the header hints at
//!    it. `unswizzleScales` is the inverse (`from_blocked`), applied ONCE at load so
//!    every consumer sees plain row-major.
//! 3. **The multiply association is `E2M1 * (per_tensor * block)`**, not
//!    `(E2M1 * block) * per_tensor`. Only the first is bit-exact against the reference —
//!    and it is what a per-scale-byte table computes naturally, which is the other reason
//!    for the table below.
//!
//! ⚠️ **No ConvRot.** The int8/int4/W4A8 formats here are all Hadamard-rotated and this
//! one is not, so `Weight.convrot` stays 0 and there is no per-output-row scale either
//! (`row_scale` is null). A GEMM path that assumes either is wrong for this format.
//!
//! **The decode goes through a `[256][16]` table** (`Levels`), the same trick
//! `ops/w4a8.zig` uses and for the same reason: the block scale is fp8, so only 256
//! values are possible, and folding the per-tensor scale in makes one table per weight
//! answer every lookup. Held in both f32 (for the CPU panel) and f16 (the GPU GEMM
//! operand) so neither consumer converts per element.
//!
//! ⚠️ **The GPU half of the table is bf16, not f16, and that is a RANGE decision.** The
//! decoded weights are tiny (E2M1 x fp8 x a per-tensor scale), but the GEMM converts the
//! ACTIVATION to the same format, and Z-Image's trunk activations exceed f16's 65504
//! ceiling — an f16 path renders it solid white, non-finite end to end, and identically on
//! all three backends. bf16 has f32's exponent range and its 8-bit mantissa is more than a
//! 4-bit payload can use, which is also the regime these models' own dense bf16 weights
//! already run in. Same class of failure this repo records for the SDXL VAE.

const std = @import("std");
const dtypes = @import("tp_core").dtype;

/// Elements per block scale. Fixed by the format (NVFP4 is a 16-element microscaling
/// format); `weight_scale` therefore has `cols / 16` entries per row.
pub const block_size = 16;

/// The `format` string a layer's `comfy_quant` carries, when it has one.
/// ⚠️ Z-Image's NVFP4 checkpoint ships NO `comfy_quant` at all, so a loader must be able
/// to recognize the format from `weight_scale_2`'s presence alone.
pub const format_name = "nvfp4";

/// E2M1: 1 sign bit, 2 exponent bits, 1 mantissa bit — 16 codes, sign in bit 3.
/// Verbatim `backends/eager/quantization.py::E2M1_LUT`; a fast test pins it.
pub const e2m1: [16]f32 = .{
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
};

/// The E2M1 code of logical element `k` of a packed byte.
///
/// ⚠️ **HIGH nibble first** — `hi_first = True` in the reference, the opposite of `.i4`
/// and `.w4a8` here. Inlined and tiny so the loop stays branch-free after unrolling.
pub inline fn nibble(byte: u8, k: u1) u4 {
    return @truncate(if (k == 0) byte >> 4 else byte);
}

/// Every value a tensor can decode to, indexed by `[block-scale byte][E2M1 code]`.
///
/// Built once per weight. 12 KiB (f32 + f16), so it stays in cache for the whole tensor,
/// and the per-element work becomes one lookup with no arithmetic at all.
pub const Levels = struct {
    f32v: [256][16]f32,
    /// The same values as **bf16** bit patterns, for the GPU decode that feeds the bf16
    /// tensor-core GEMM. See the module header on why bf16 rather than f16.
    bf16v: [256][16]u16,

    /// `global` is `weight_scale_2`.
    ///
    /// ⚠️ A NaN block-scale byte (fp8 `0x7F`/`0xFF`) **propagates** into the table, and
    /// that is deliberate: the reference propagates it too, so matching is correct.
    /// `ops/w4a8.zig` maps NaN to 0 instead — but only because its target is an integer
    /// and `@intFromFloat(nan)` is illegal behaviour, not because 0 is more right. Either
    /// way a real checkpoint has no NaN scale; this is about which wrong answer to give
    /// for a corrupt one, and here it is the reference's.
    pub fn init(global: f32) Levels {
        var lv: Levels = undefined;
        for (0..256) |b| {
            // ⚠️ The association: `per_tensor * block` FIRST, then `* E2M1`. The other
            // order rounds differently and is not the reference's.
            const total = global * dtypes.f8e4m3ToF32(@intCast(b));
            for (0..16) |i| {
                const v = e2m1[i] * total;
                lv.f32v[b][i] = v;
                lv.bf16v[b][i] = dtypes.f32ToBf16(v);
            }
        }
        return lv;
    }
};

/// The sidecars a packed NVFP4 weight needs, carried on `ops.matmul.Weight.nvfp4`.
pub const Meta = struct {
    /// Per-block scale bytes, **already unswizzled** to row-major `[rows][cols/16]`.
    /// Owned by the loader (a copy — the on-disk order is not this one).
    scales: []const u8,
    /// This weight's `[256][16]` decode table. A pointer so a GPU backend can key a
    /// device copy on its address the way it keys weights.
    levels: *const Levels,

    pub fn blocks(cols: usize) usize {
        return cols / block_size;
    }

    /// The metadata for a contiguous ROW RANGE of the weight — what a fused qkv split
    /// into three row-block GEMMs needs.
    ///
    /// ⚠️ **The scales MUST be sliced along with the bytes.** They are `[rows][cols/16]`
    /// row-major, so a row range is a plain slice — but leaving the full array behind a
    /// shortened `Weight` makes rows index from the FUSED tensor's row 0, so the k and v
    /// blocks read q's block scales. Finite, plausible, and wrong. `levels` is per
    /// TENSOR (it folds `weight_scale_2`), so it is shared unchanged.
    pub fn rowSlice(self: Meta, cols: usize, row0: usize, nrows: usize) Meta {
        const nblk = blocks(cols);
        return .{ .scales = self.scales[row0 * nblk ..][0 .. nrows * nblk], .levels = self.levels };
    }
};

/// Reverse cuBLAS's tiled block-scale layout into plain row-major `[rows][nblk]`.
///
/// The exact inverse of `comfy_kitchen.float_utils.to_blocked`, whose forward direction
/// tiles a `(ceil(rows,128) * 128, ceil(nblk,4) * 4)` padded grid as
/// `(-1, 32, 4, 4) -> transpose(1, 2)`. Reading it as the reference's own five reshapes
/// rather than as a closed-form index:
///
///   `blocked.reshape(-1, 32, 16).reshape(-1, 32, 4, 4).transpose(1,2)`
///     `-> (n_row_blk, n_col_blk, 4, 32, 4) -> (n_row_blk, n_col_blk, 128, 4)`
///     `-> permute(0, 2, 1, 3) -> (padded_rows, padded_cols)`
///
/// which composes to: logical `(r, c)` lives at `blocked[rb][cb][r%32][(r/32)%4][c%4]`
/// in the `(n_row_blk, n_col_blk, 32, 4, 4)` view, with `rb = r/128`, `cb = c/4`.
///
/// ⚠️ Applied ONCE at load, into a caller-owned copy. Doing it per access would put a
/// five-way index computation in the inner loop of every GEMM, and doing it on the device
/// would need the swizzle in a kernel; neither is necessary for a `rows*nblk`-byte array
/// that is 1/16 the weight.
pub fn unswizzleScales(dst: []u8, src: []const u8, rows: usize, nblk: usize) void {
    std.debug.assert(dst.len == rows * nblk);
    const n_col_blk = std.math.divCeil(usize, nblk, 4) catch unreachable;
    // Stride of one row-block in the blocked array: n_col_blk * 32 * 4 * 4.
    const row_blk_stride = n_col_blk * 512;
    for (0..rows) |r| {
        const rb = r / 128;
        const within = r % 128;
        const lo = within % 32; // the 32-axis
        const mid = within / 32; // the first 4-axis
        for (0..nblk) |c| {
            const cb = c / 4;
            const idx = rb * row_blk_stride + cb * 512 + lo * 16 + mid * 4 + (c % 4);
            dst[r * nblk + c] = if (idx < src.len) src[idx] else 0;
        }
    }
}

/// Sanity-check a layer's tensor sizes against each other; returns the block count.
pub fn validate(rows: usize, cols: usize, packed_len: usize, scale_len: usize) !usize {
    // The reference pads to a 16x16 grid, so a real tensor's dims are multiples of 16;
    // `cols % 16` is what the block scale requires and `cols % 2` the nibble packing.
    if (cols == 0 or cols % block_size != 0) return error.ShapeMismatch;
    if (packed_len != rows * cols / 2) return error.ShapeMismatch;
    const nblk = cols / block_size;
    // The stored scale array is the SWIZZLED, padded grid, so it can be larger than
    // rows*nblk — but never smaller, or the unswizzle would read past it.
    if (scale_len < rows * nblk) return error.ShapeMismatch;
    return nblk;
}

/// Decode `n` elements of row `row` starting at column `col0` (even) into f32.
///
/// The level table already folds `E2M1 * (per_tensor * block)`, so this is two lookups
/// per element and bit-identical to the reference at f32 output.
pub inline fn decodeSliceF32(
    dst: []f32,
    packed_bytes: []const u8,
    meta: Meta,
    cols: usize,
    row: usize,
    col0: usize,
    n: usize,
) void {
    std.debug.assert(col0 % 2 == 0);
    const nblk = Meta.blocks(cols);
    const srow = meta.scales[row * nblk ..];
    const byte0 = (row * cols + col0) / 2;
    for (0..n) |k| {
        const c = col0 + k;
        const tbl = &meta.levels.f32v[srow[c / block_size]];
        dst[k] = tbl[nibble(packed_bytes[byte0 + k / 2], @intCast(c & 1))];
    }
}

/// Decode a whole weight to f32 `[rows][cols]` (tests and reference paths).
pub fn decode(dst: []f32, packed_bytes: []const u8, meta: Meta, rows: usize, cols: usize) void {
    std.debug.assert(dst.len == rows * cols);
    for (0..rows) |r| decodeSliceF32(dst[r * cols ..][0..cols], packed_bytes, meta, cols, r, 0, cols);
}

// --- tests -----------------------------------------------------------------

const fixtures_json = @embedFile("assets/nvfp4_fixtures.json");

const Case = struct {
    name: []const u8,
    rows: usize,
    cols: usize,
    global_scale: f32,
    packed_hex: []const u8,
    scale_blocked_hex: []const u8,
    scale_logical_hex: []const u8,
    expect_f32: []const f32,
};

const Fixtures = struct { e2m1_lut: []const f32, cases: []const Case };

fn loadFixtures(gpa: std.mem.Allocator) !std.json.Parsed(Fixtures) {
    return std.json.parseFromSlice(Fixtures, gpa, fixtures_json, .{ .ignore_unknown_fields = true });
}

fn unhex(gpa: std.mem.Allocator, hex: []const u8) ![]u8 {
    const out = try gpa.alloc(u8, hex.len / 2);
    _ = try std.fmt.hexToBytes(out, hex);
    return out;
}

test "the E2M1 table matches comfy_kitchen's" {
    const gpa = std.testing.allocator;
    var parsed = try loadFixtures(gpa);
    defer parsed.deinit();
    // -0.0 == 0.0 compares equal, which is what we want here: the sign of zero is
    // preserved by the table and irrelevant to any product.
    try std.testing.expectEqualSlices(f32, parsed.value.e2m1_lut, &e2m1);
}

test "unswizzleScales inverts comfy_kitchen's to_blocked" {
    // Checked against the fixture's own pair of arrays — the swizzled bytes as a
    // checkpoint stores them and the row-major logical scales — rather than against our
    // own re-derivation, which would prove nothing about the layout.
    const gpa = std.testing.allocator;
    var parsed = try loadFixtures(gpa);
    defer parsed.deinit();
    for (parsed.value.cases) |c| {
        const blocked = try unhex(gpa, c.scale_blocked_hex);
        defer gpa.free(blocked);
        const logical = try unhex(gpa, c.scale_logical_hex);
        defer gpa.free(logical);
        const nblk = c.cols / block_size;
        const got = try gpa.alloc(u8, c.rows * nblk);
        defer gpa.free(got);
        unswizzleScales(got, blocked, c.rows, nblk);
        errdefer std.debug.print("case {s} ({d}x{d}, {d} blocks)\n", .{ c.name, c.rows, c.cols, nblk });
        try std.testing.expectEqualSlices(u8, logical, got);
    }
}

test "NVFP4 decode matches comfy_kitchen's reference on every fixture case" {
    const gpa = std.testing.allocator;
    var parsed = try loadFixtures(gpa);
    defer parsed.deinit();

    for (parsed.value.cases) |c| {
        const packed_bytes = try unhex(gpa, c.packed_hex);
        defer gpa.free(packed_bytes);
        const blocked = try unhex(gpa, c.scale_blocked_hex);
        defer gpa.free(blocked);
        const nblk = try validate(c.rows, c.cols, packed_bytes.len, blocked.len);

        const scales = try gpa.alloc(u8, c.rows * nblk);
        defer gpa.free(scales);
        unswizzleScales(scales, blocked, c.rows, nblk);
        const levels = try gpa.create(Levels);
        defer gpa.destroy(levels);
        levels.* = Levels.init(c.global_scale);
        const meta: Meta = .{ .scales = scales, .levels = levels };

        const got = try gpa.alloc(f32, c.rows * c.cols);
        defer gpa.free(got);
        decode(got, packed_bytes, meta, c.rows, c.cols);

        // Exact: the fixture's reference runs at f32 output and this reproduces its
        // multiply association, so there is nothing for a tolerance to absorb.
        for (got, c.expect_f32, 0..) |g, w, i| {
            std.testing.expectEqual(w, g) catch |e| {
                std.debug.print("case {s}: element {d} (row {d}, col {d}) got {d}, want {d}\n", .{
                    c.name, i, i / c.cols, i % c.cols, g, w,
                });
                return e;
            };
        }
    }
}

test "the low-nibble-first reading of an NVFP4 weight disagrees" {
    // The teeth for convention 1. `hi_first` is the one thing here that a port would most
    // naturally get wrong (every other 4-bit format in this engine is low-first) and the
    // one a magnitude check cannot catch, since swapping adjacent elements preserves the
    // row's rms exactly. So assert both that it differs AND that the rms does not.
    const gpa = std.testing.allocator;
    var parsed = try loadFixtures(gpa);
    defer parsed.deinit();
    const c = parsed.value.cases[0];
    const packed_bytes = try unhex(gpa, c.packed_hex);
    defer gpa.free(packed_bytes);
    const blocked = try unhex(gpa, c.scale_blocked_hex);
    defer gpa.free(blocked);
    const nblk = c.cols / block_size;
    const scales = try gpa.alloc(u8, c.rows * nblk);
    defer gpa.free(scales);
    unswizzleScales(scales, blocked, c.rows, nblk);
    const levels = try gpa.create(Levels);
    defer gpa.destroy(levels);
    levels.* = Levels.init(c.global_scale);

    var differ: usize = 0;
    var ss_ok: f64 = 0;
    var ss_swapped: f64 = 0;
    for (0..c.rows) |r| {
        for (0..c.cols) |col| {
            const tbl = &levels.f32v[scales[r * nblk + col / block_size]];
            const byte = packed_bytes[(r * c.cols + col) / 2];
            const k: u1 = @intCast(col & 1);
            const ok = tbl[nibble(byte, k)];
            const swapped = tbl[@as(u4, @truncate(if (k == 0) byte else byte >> 4))]; // low-first
            if (ok != swapped) differ += 1;
            ss_ok += @as(f64, ok) * ok;
            ss_swapped += @as(f64, swapped) * swapped;
        }
    }
    try std.testing.expect(differ > c.rows * c.cols / 2);
    // Same sum of squares to the last bit: the wrong order is a permutation.
    try std.testing.expectEqual(ss_ok, ss_swapped);
}

test "NVFP4 validate refuses shapes the format cannot have" {
    // `cols` not a multiple of the 16-element block, a packed length that is not
    // rows*cols/2, and a scale array too short for the unswizzle to read.
    try std.testing.expectError(error.ShapeMismatch, validate(4, 24, 4 * 12, 4 * 2));
    try std.testing.expectError(error.ShapeMismatch, validate(4, 32, 4 * 15, 4 * 2));
    try std.testing.expectError(error.ShapeMismatch, validate(4, 32, 4 * 16, 4 * 2 - 1));
    try std.testing.expectEqual(@as(usize, 2), try validate(4, 32, 4 * 16, 4 * 2));
    // A larger scale array is fine — the stored grid is padded to 128x4 blocks.
    try std.testing.expectEqual(@as(usize, 2), try validate(4, 32, 4 * 16, 128 * 4));
}

test "the bf16 half of the level table matches the f32 half" {
    // The GPU decode reads `bf16v` and the CPU `f32v`; they must be the same values so a
    // device render differs from a CPU one only by bf16's own rounding.
    const levels = Levels.init(0.03125);
    for (0..256) |b| {
        for (0..16) |i| {
            try std.testing.expectEqual(dtypes.f32ToBf16(levels.f32v[b][i]), levels.bf16v[b][i]);
        }
    }
}
