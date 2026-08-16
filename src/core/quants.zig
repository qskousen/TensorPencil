//! ggml/GGUF block-quantized weight formats: dequantization to f32.
//!
//! Bit-exact ports of the reference decoders in llama.cpp's ggml-quants.c
//! (dequantize_row_q8_0 / q4_K / q5_K / q6_K), validated against golden
//! fixtures generated with the reference implementation (quants_fixtures.zig,
//! tools/gen_quant_fixtures.c). Operation order matches the reference so the
//! output is bitwise identical.
//!
//! Layouts (little-endian; k-quants use 256-element super-blocks):
//! - q8_0 (34 B / 32):  f16 d, 32 x i8;                     v = q * d
//! - q4_k (144 B / 256): f16 d, f16 dmin, 12 B packed 6-bit sub-block
//!   scales/mins (8 sub-blocks of 32), 128 B low nibbles;   v = d*sc*q - dmin*m
//! - q5_k (176 B / 256): q4_k layout + 32 B of per-element 5th bits
//! - q6_k (210 B / 256): 128 B low nibbles, 64 B high 2-bit pairs,
//!   16 x i8 sub-block scales (16 sub-blocks of 16), f16 d; v = d*sc*(q - 32)
//! - q1_0 (18 B / 128): f16 d, 16 B of sign bits (LSB-first);  v = bit ? d : -d
//! - q2_0_g64 (18 B / 64) and q2_0_g128 (34 B / 128): f16 d, then 2-bit codes
//!   (LSB-first, 4 per byte); v = (code - 1) * d, so the set is {-1, 0, +1, +2}
//!   * d, NOT centred on zero. Same arithmetic, two block sizes, one GGUF type
//!   id; g64 is ggml's, g128 is decoded natively. See `dequantQ2_0G128`.

const std = @import("std");
const dtypes = @import("dtype.zig");
const build_options = @import("build_options");

const DType = dtypes.DType;

/// Whether ggml (the GGUF block-quant CPU backend) was linked in this build.
/// Gated by `-Dggml` (default on). When false, the block-quant dequant/GEMV
/// paths are unavailable: `dequantSlice` panics and the matmul dispatch returns
/// `error.QuantBackendUnavailable` (see ops/matmul.zig). All non-block-quant
/// dtypes (f32/f16/bf16/fp8/int8/int4) are unaffected.
pub const have_ggml = build_options.have_ggml;

/// Returned by the block-quant matmul path when built with `-Dggml=false`.
pub const QuantError = error{QuantBackendUnavailable};

// All ggml-typed helpers live behind the build flag. When ggml is absent the
// `@import("ggml")` is never analyzed (so the module need not be supplied to
// the build), and these collapse to inert stubs.
const gg = if (have_ggml) struct {
    const ggml = @import("ggml");

    /// ggml type enum for a block-quant DType (null otherwise).
    pub fn blockType(dt: DType) ?ggml.c.enum_ggml_type {
        return switch (dt) {
            .q4_0 => ggml.c.GGML_TYPE_Q4_0,
            .q8_0 => ggml.c.GGML_TYPE_Q8_0,
            .q2_k => ggml.c.GGML_TYPE_Q2_K,
            .q4_k => ggml.c.GGML_TYPE_Q4_K,
            .q5_k => ggml.c.GGML_TYPE_Q5_K,
            .q6_k => ggml.c.GGML_TYPE_Q6_K,
            .iq4_nl => ggml.c.GGML_TYPE_IQ4_NL,
            .iq4_xs => ggml.c.GGML_TYPE_IQ4_XS,
            .q1_0 => ggml.c.GGML_TYPE_Q1_0,
            // ggml's GGML_TYPE_Q2_0 is the 64-element variant, so ONLY g64 maps
            // here. `.q2_0_g128` is deliberately absent: its blocks are 128
            // elements, so these kernels would walk the byte stream with the
            // wrong stride and return plausible garbage rather than fail. A null
            // here is what keeps every ggml entry point (dequant, vec_dot,
            // quantize) unreachable for it. See `dequantQ2_0G128`.
            .q2_0_g64 => ggml.c.GGML_TYPE_Q2_0,
            else => null,
        };
    }

    var inited = false;
    pub fn ensureInit() void {
        if (!inited) {
            ggml.c.ggml_cpu_init();
            inited = true;
        }
    }

    pub fn dequantSlice(dt: DType, row: []const u8, elem0: usize, n: usize, dst: []f32) void {
        std.debug.assert(dst.len >= n);
        const be = dt.blockElems();
        std.debug.assert(elem0 % be == 0 and n % be == 0);
        const gt = blockType(dt) orelse unreachable; // not a block-quantized dtype
        const x = row.ptr + (elem0 / be) * dt.blockBytes();
        ggml.c.ggml_get_type_traits(gt).*.to_float.?(x, dst.ptr, @intCast(n));
    }
} else struct {
    // Never reached in a `-Dggml=false` build: every caller is gated on
    // `have_ggml` and errors out before these run. Present only so the module
    // compiles without the ggml import.
    pub fn blockType(dt: DType) ?u32 {
        _ = dt;
        unreachable;
    }
    pub fn ensureInit() void {}
    pub fn dequantSlice(dt: DType, row: []const u8, elem0: usize, n: usize, dst: []f32) void {
        _ = .{ dt, row, elem0, n, dst };
        @panic("quants.dequantSlice: TensorPencil built with -Dggml=false; " ++
            "GGUF block-quant (q4_0/q8_0/q2_k/q4_k/q5_k/q6_k/iq4_nl/q1_0) is unavailable");
    }
};

/// ggml type enum for a block-quant DType (null otherwise). Shared with matmul.
/// Only meaningful in a ggml build.
pub const ggmlType = gg.blockType;

/// Fill ggml's fp16 table / CPU dispatch once (the CPU vec_dot kernels return 0
/// without it). Idempotent; the first block-quant matmul runs on the single
/// main thread before any fan-out, so a plain flag is enough. No-op without ggml.
pub const ensureGgmlInit = gg.ensureInit;

/// Whether this dtype's decode goes through ggml. False for non-block dtypes and
/// for `.q2_0_g128`, which is decoded natively, so it needs neither `-Dggml` nor
/// the `ensureGgmlInit` call, and must never reach a ggml entry point. Callers
/// choosing between the ggml `vec_dot` GEMV and the packed path gate on this.
pub fn usesGgml(dt: DType) bool {
    return dt.isBlockQuant() and ggmlType(dt) != null;
}

/// Dequantize elements [elem0, elem0 + n) of a block-quantized `row` into `dst`
/// via ggml's (auto-vectorized) `to_float`, ~4-12x faster than the scalar Zig
/// decode it replaced. `elem0`/`n` must be block-aligned (ggml blocks never span
/// rows; callers slice at block-aligned offsets). Bit-identical to the ggml
/// reference our golden fixtures were generated from. Panics if built without
/// ggml, except for `.q2_0_g128`, which never needs it.
pub fn dequantSlice(dt: DType, row: []const u8, elem0: usize, n: usize, dst: []f32) void {
    if (dt == .q2_0_g128) return dequantQ2_0G128(row, elem0, n, dst);
    gg.dequantSlice(dt, row, elem0, n, dst);
}

/// SIMD width for the fused q2_0 dot. Capped at 16 because the code extraction
/// packs `2 * q2_vl` shift amounts into one integer load, and 2*16 = 32 bits is
/// the widest that fits a u32; every candidate width divides 128.
const q2_vl: usize = @min(16, @max(4, std.simd.suggestVectorLength(f32) orelse 4));
/// Bit offset of each lane's 2-bit code within the loaded word: 0, 2, 4, ...
const q2_shifts: @Vector(q2_vl, u5) = blk: {
    var s: [q2_vl]u5 = undefined;
    for (&s, 0..) |*e, i| e.* = @intCast(2 * i);
    break :blk s;
};
/// The integer holding one vector's worth of codes (u8 / u16 / u32). Read exactly
/// this wide, a u32 load at every step would run past the 34-byte block on the
/// last block of a row.
const Q2Word = std.meta.Int(.unsigned, q2_vl * 2);

/// Fused q2_0 g128 dot: `Σ (code - 1) * d * x`, reading each weight byte ONCE and
/// never materializing the dequantized row.
///
/// This exists because the g128 arm has no ggml `vec_dot` (see `dequantQ2_0G128`),
/// and the packed fallback it would otherwise take expands the weight to f32
/// panels, 4 B per element against 0.266 B stored, ~15x the memory traffic, which
/// on Bonsai-27B measured as 0.2 tok/s against 2.6 for the first version of this.
///
/// Unlike ggml's `vec_dot`, the activation is NOT quantized: this is exact in
/// `x`, so the small-`m` and packed paths agree to summation order rather than to
/// a 5% activation-quantization bound. It is in fact slightly MORE accurate than
/// dequant-then-dot: `(code-1) * x` is exact in f32 (the multipliers are ±1, 0, 2),
/// so scaling by `d` once per block rounds once where the reference rounds every
/// `(code-1)*d*x` product. That exactness is also why the `- 1` stays in the inner
/// loop rather than being hoisted as `Σ code*x - Σx` (which would be one vector op
/// cheaper, and the sum is even the same for every row): `3 * x` is NOT exact, so hoisting
/// would trade the property for a rounding.
///
/// The codes are extracted with shifts, not a lookup table, and that is a
/// measured choice. A `[256][4]f32` coefficient table indexed per qs byte is the
/// obvious form and was the first version; it costs THREE loads (byte, table row,
/// activation) per 4 elements, and profiling put `matmul` at 91.2% of CPU decode
/// while sustaining only 17.8 GB/s of weight traffic, nowhere near this box's
/// DRAM bandwidth, i.e. load-issue bound, not memory bound. Extracting instead
/// pulls one integer load per `q2_vl` codes and leaves the activation load as the
/// only other memory op.
pub fn dotQ2_0G128(row: []const u8, x: []const f32) f32 {
    const be = 128;
    const bb = 34;
    std.debug.assert(x.len % be == 0);
    const V = @Vector(q2_vl, f32);
    const U = @Vector(q2_vl, u32);
    const step = 4 * q2_vl; // four accumulator chains per pass
    var acc: f32 = 0;
    for (0..x.len / be) |bi| {
        const b = row[bi * bb ..][0..bb];
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, b[0..2], .little))));
        const xb = x[bi * be ..][0..be];
        var s: [4]V = @splat(@as(V, @splat(0)));
        var e: usize = 0;
        while (e < be) : (e += step) {
            inline for (0..4) |u| {
                const e0 = e + u * q2_vl;
                // 4 codes per qs byte, so element e0 starts at qs byte e0/4.
                const word = std.mem.readInt(Q2Word, b[2 + e0 / 4 ..][0..@sizeOf(Q2Word)], .little);
                const codes = (@as(U, @splat(word)) >> q2_shifts) & @as(U, @splat(3));
                const c: V = @floatFromInt(codes);
                s[u] += (c - @as(V, @splat(1))) * @as(V, xb[e0..][0..q2_vl].*);
            }
        }
        acc += d * @reduce(.Add, (s[0] + s[1]) + (s[2] + s[3]));
    }
    return acc;
}

/// Native q2_0 g128 decode: `v = (code - 1) * d` over 128-element / 34-byte blocks.
///
/// This does NOT go through ggml, and that is the whole point. GGUF type id
/// 42 is claimed by two shipped formats with identical arithmetic and different
/// block sizes: upstream ggml's `GGML_TYPE_Q2_0` uses `QK2_0 = 64` (18 B blocks,
/// our `.q2_0_g64`, which *does* use ggml), while the PrismML llama.cpp fork's
/// `prism` branch, which every published Bonsai / "Ternary" GGUF is quantized
/// with, advertising itself as "Q2_0 g128", uses `QK2_0 = 128` (34 B). Calling
/// ggml's `to_float` for type 42 on a g128 file walks the byte stream with the
/// wrong stride and returns plausible garbage rather than failing, so
/// `ggmlType(.q2_0_g128)` is null and this is the only decoder for it.
///
/// The 4-entry LUT is bit-exact against the reference's `((int)q - 1) * d`: the
/// three non-zero products are exact in f32 (`-1*d`, `1*d`, `2*d` are sign flips
/// and an exponent bump), and `0 * d` is written as such so a negative `d` yields
/// the reference's -0.0 rather than +0.0.
fn dequantQ2_0G128(row: []const u8, elem0: usize, n: usize, dst: []f32) void {
    const be = 128; // DType.q2_0_g128.blockElems()
    const bb = 34; // DType.q2_0_g128.blockBytes()
    std.debug.assert(dst.len >= n);
    std.debug.assert(elem0 % be == 0 and n % be == 0);
    var blk = elem0 / be;
    var o: usize = 0;
    while (o < n) : (o += be) {
        const b = row[blk * bb ..][0..bb];
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, b[0..2], .little))));
        const lut = [4]f32{ -d, 0 * d, d, 2 * d };
        for (b[2..], 0..) |byte, i| {
            dst[o + i * 4 + 0] = lut[byte & 3];
            dst[o + i * 4 + 1] = lut[(byte >> 2) & 3];
            dst[o + i * 4 + 2] = lut[(byte >> 4) & 3];
            dst[o + i * 4 + 3] = lut[byte >> 6];
        }
        blk += 1;
    }
}

// ---------------------------------------------------------------------------
// Raw ggml type bridge
//
// `DType` above covers the formats TensorPencil can *compute* with. A tool that
// inspects or rewrites arbitrary GGUF files (a converter, a quantizer) has to
// handle every type ggml knows, including ones we have no kernel for and don't
// want to model in `DType`. `raw` is that escape hatch: the same ggml quantize /
// dequantize / layout entry points, keyed on the numeric `enum ggml_type` id
// straight out of a GGUF header, with the unchecked C surface (out-of-range ids,
// null trait function pointers, unaligned lengths) turned into Zig errors.
//
// Prefer `DType` when the type is one we compute with; reach for `raw` when the
// type is data passing through.
// ---------------------------------------------------------------------------

pub const RawError = error{
    /// Not a valid `enum ggml_type` value in the linked ggml.
    UnknownGgmlType,
    /// ggml knows this type but has no kernel for the requested direction
    /// (e.g. one it can read but not produce).
    UnsupportedGgmlType,
    /// Element count is not a whole number of blocks for this type.
    NotBlockAligned,
    /// A source or destination buffer is the wrong size for the request.
    BufferSizeMismatch,
    /// A type that cannot be quantized without an importance matrix was asked
    /// for without one.
    ImatrixRequired,
} || QuantError;

pub const raw = if (have_ggml) struct {
    const ggml = @import("ggml");

    fn checked(id: u32) RawError!ggml.c.enum_ggml_type {
        if (id >= typeCount()) return error.UnknownGgmlType;
        return @intCast(id);
    }

    /// Number of `enum ggml_type` values in the linked ggml; valid ids are
    /// `0..typeCount() - 1`. This grows with ggml versions, a GGUF written by a
    /// newer llama.cpp can carry ids this build does not know.
    pub fn typeCount() u32 {
        return @intCast(ggml.c.GGML_TYPE_COUNT);
    }

    /// ggml's own name for the type ("q4_K", "f16", ...).
    pub fn name(id: u32) RawError![]const u8 {
        return std.mem.span(ggml.c.ggml_type_name(try checked(id)));
    }

    /// Elements per block (1 for non-block types).
    pub fn blockElems(id: u32) RawError!usize {
        return @intCast(ggml.c.ggml_blck_size(try checked(id)));
    }

    /// Bytes per block (for non-block types, bytes per element).
    pub fn blockBytes(id: u32) RawError!usize {
        return ggml.c.ggml_type_size(try checked(id));
    }

    /// Bytes needed to store `elems` elements of this type.
    pub fn rowBytes(id: u32, elems: usize) RawError!usize {
        const t = try checked(id);
        const be: usize = @intCast(ggml.c.ggml_blck_size(t));
        if (elems % be != 0) return error.NotBlockAligned;
        return ggml.c.ggml_row_size(t, @intCast(elems));
    }

    pub fn isQuantized(id: u32) RawError!bool {
        return ggml.c.ggml_is_quantized(try checked(id));
    }

    /// True for the types (the IQ family) that cannot be produced without an
    /// importance matrix.
    pub fn requiresImatrix(id: u32) RawError!bool {
        return ggml.c.ggml_quantize_requires_imatrix(try checked(id));
    }

    /// Pre-build this type's quantization tables. Optional, `quantizeChunk` does
    /// it internally, but worth calling once up front when many threads will
    /// quantize concurrently, so the table build is off the hot path. ggml
    /// documents init/free as thread-safe.
    pub fn ensureQuantizeInit(id: u32) RawError!void {
        ggml.c.ggml_quantize_init(try checked(id));
    }

    /// Release the memory held by `ensureQuantizeInit` / `quantizeChunk` (the IQ
    /// lookup tables). Optional; call at process end to keep leak checkers quiet.
    pub fn quantizeFree() void {
        ggml.c.ggml_quantize_free();
    }

    /// Quantize `src` (`nrows * n_per_row` f32 values, row-major) into `dst` as
    /// type `id`, returning the number of bytes written.
    ///
    /// `imatrix`, when given, is the per-column importance weighting that the
    /// k-quant and IQ scale searches minimize against, one weight per column, so
    /// `n_per_row` of them, shared by every row in the call. This is the
    /// activation-aware hook: pass per-channel activation energy and ggml picks
    /// block scales that minimize weighted error instead of plain squared error.
    ///
    /// Unlike ggml's `ggml_quantize_chunk` there is no `start` offset, slice
    /// `src`/`dst` instead. Safe to call from many threads on disjoint slices.
    pub fn quantizeChunk(
        id: u32,
        src: []const f32,
        dst: []u8,
        nrows: usize,
        n_per_row: usize,
        imatrix: ?[]const f32,
    ) RawError!usize {
        const t = try checked(id);
        if (src.len != nrows * n_per_row) return error.BufferSizeMismatch;
        const row_bytes = try rowBytes(id, n_per_row);
        if (dst.len < nrows * row_bytes) return error.BufferSizeMismatch;
        if (imatrix) |im| {
            if (im.len != n_per_row) return error.BufferSizeMismatch;
        } else if (ggml.c.ggml_quantize_requires_imatrix(t)) {
            return error.ImatrixRequired;
        }
        // ggml asserts internally on a type it cannot produce; check the trait
        // first so that surfaces as an error instead of aborting the process.
        if (ggml.c.ggml_get_type_traits(t).*.from_float_ref == null)
            return error.UnsupportedGgmlType;

        return ggml.c.ggml_quantize_chunk(
            t,
            src.ptr,
            dst.ptr,
            0,
            @intCast(nrows),
            @intCast(n_per_row),
            if (imatrix) |im| im.ptr else null,
        );
    }

    /// Dequantize `elems` elements of type `id` from `src` into `dst`. `elems`
    /// must be a whole number of blocks. Same ggml `to_float` kernel, and so the
    /// same bytes, as `dequantSlice`, but reachable for any ggml type.
    pub fn dequantRow(id: u32, src: []const u8, elems: usize, dst: []f32) RawError!void {
        const t = try checked(id);
        if (src.len < try rowBytes(id, elems)) return error.BufferSizeMismatch;
        if (dst.len < elems) return error.BufferSizeMismatch;
        const to_float = ggml.c.ggml_get_type_traits(t).*.to_float orelse
            return error.UnsupportedGgmlType;
        ensureGgmlInit(); // to_float reads ggml's fp16 table
        to_float(src.ptr, dst.ptr, @intCast(elems));
    }
} else struct {
    // Built with -Dggml=false: no ggml to bridge to. Every entry point reports
    // the missing backend rather than panicking, so a consumer can degrade.
    pub fn typeCount() u32 {
        return 0;
    }
    pub fn name(id: u32) RawError![]const u8 {
        _ = id;
        return error.QuantBackendUnavailable;
    }
    pub fn blockElems(id: u32) RawError!usize {
        _ = id;
        return error.QuantBackendUnavailable;
    }
    pub fn blockBytes(id: u32) RawError!usize {
        _ = id;
        return error.QuantBackendUnavailable;
    }
    pub fn rowBytes(id: u32, elems: usize) RawError!usize {
        _ = .{ id, elems };
        return error.QuantBackendUnavailable;
    }
    pub fn isQuantized(id: u32) RawError!bool {
        _ = id;
        return error.QuantBackendUnavailable;
    }
    pub fn requiresImatrix(id: u32) RawError!bool {
        _ = id;
        return error.QuantBackendUnavailable;
    }
    pub fn ensureQuantizeInit(id: u32) RawError!void {
        _ = id;
        return error.QuantBackendUnavailable;
    }
    pub fn quantizeFree() void {}
    pub fn quantizeChunk(id: u32, src: []const f32, dst: []u8, nrows: usize, n_per_row: usize, imatrix: ?[]const f32) RawError!usize {
        _ = .{ id, src, dst, nrows, n_per_row, imatrix };
        return error.QuantBackendUnavailable;
    }
    pub fn dequantRow(id: u32, src: []const u8, elems: usize, dst: []f32) RawError!void {
        _ = .{ id, src, elems, dst };
        return error.QuantBackendUnavailable;
    }
};

/// Quantize f32 `src` into `dst` as `dt`, the typed convenience over
/// `raw.quantizeChunk` for the block-quant dtypes TP models. See there for the
/// `imatrix` contract and the threading rules.
pub fn quantizeChunk(
    dt: DType,
    src: []const f32,
    dst: []u8,
    nrows: usize,
    n_per_row: usize,
    imatrix: ?[]const f32,
) RawError!usize {
    if (!have_ggml) return error.QuantBackendUnavailable;
    const gt = ggmlType(dt) orelse return error.UnsupportedGgmlType;
    return raw.quantizeChunk(@intCast(gt), src, dst, nrows, n_per_row, imatrix);
}

// --- tests -----------------------------------------------------------------

const fixtures = @import("quants_fixtures.zig");

/// ggml type ids, as they appear in a GGUF header. Stable (append-only) across
/// ggml versions; the tests below assert each id still names what we expect
/// before using it, so an upstream renumbering fails loudly instead of silently
/// testing the wrong format.
const id_q4_0: u32 = 2;
const id_q5_0: u32 = 6;
const id_q8_0: u32 = 8;
const id_q2_k: u32 = 10;
const id_q3_k: u32 = 11;
const id_q4_k: u32 = 12;
const id_q5_k: u32 = 13;
const id_q6_k: u32 = 14;

/// Deterministic pseudo-Gaussian test weights: a fixed seed so every assertion
/// below is reproducible, at a realistic trained-weight scale.
fn fillTestWeights(dst: []f32, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    for (dst) |*v| v.* = rng.floatNorm(f32) * 0.02;
}

test "raw type metadata agrees with DType for the shared block types" {
    if (!have_ggml) return error.SkipZigTest;
    const pairs = [_]struct { id: u32, dt: DType }{
        .{ .id = id_q4_0, .dt = .q4_0 },
        .{ .id = id_q8_0, .dt = .q8_0 },
        .{ .id = id_q4_k, .dt = .q4_k },
        .{ .id = id_q5_k, .dt = .q5_k },
        .{ .id = id_q6_k, .dt = .q6_k },
    };
    for (pairs) |p| {
        errdefer std.debug.print("mismatch for ggml id {d} / {t}\n", .{ p.id, p.dt });
        try std.testing.expectEqual(p.dt.blockElems(), try raw.blockElems(p.id));
        try std.testing.expectEqual(p.dt.blockBytes(), try raw.blockBytes(p.id));
        const n = p.dt.blockElems() * 4;
        try std.testing.expectEqual(p.dt.storageBytes(n), try raw.rowBytes(p.id, n));
        try std.testing.expect(try raw.isQuantized(p.id));
        // ggmlType(dt) and the literal id must be the same number.
        try std.testing.expectEqual(p.id, @as(u32, @intCast(ggmlType(p.dt).?)));
    }
    // Non-quantized types are single-element "blocks".
    try std.testing.expectEqual(@as(usize, 1), try raw.blockElems(0)); // f32
    try std.testing.expectEqual(@as(usize, 4), try raw.blockBytes(0));
    try std.testing.expect(!try raw.isQuantized(0));
}

test "raw rejects unknown ids, unaligned lengths and missing imatrix" {
    if (!have_ggml) return error.SkipZigTest;
    const bogus = raw.typeCount(); // one past the last valid id
    try std.testing.expectError(error.UnknownGgmlType, raw.name(bogus));
    try std.testing.expectError(error.UnknownGgmlType, raw.blockElems(bogus));
    try std.testing.expectError(error.UnknownGgmlType, raw.rowBytes(bogus, 256));

    // q4_k is a 256-element super-block, so 128 elements is not a whole block.
    try std.testing.expectError(error.NotBlockAligned, raw.rowBytes(id_q4_k, 128));

    var src: [256]f32 = undefined;
    fillTestWeights(&src, 1);
    var dst: [512]u8 = undefined;
    // Wrong src length for the declared shape.
    try std.testing.expectError(error.BufferSizeMismatch, raw.quantizeChunk(id_q4_k, src[0..128], &dst, 1, 256, null));
    // Undersized dst.
    try std.testing.expectError(error.BufferSizeMismatch, raw.quantizeChunk(id_q4_k, &src, dst[0..16], 1, 256, null));
    // imatrix must be one weight per column.
    var short_im: [8]f32 = @splat(1.0);
    try std.testing.expectError(error.BufferSizeMismatch, raw.quantizeChunk(id_q4_k, &src, &dst, 1, 256, &short_im));
}

test "raw quantize/dequantize round-trips every block type ggufy emits" {
    if (!have_ggml) return error.SkipZigTest;
    const n = 256; // one q-super-block, eight 32-element blocks
    var src: [n]f32 = undefined;
    fillTestWeights(&src, 0xC0FFEE);

    // (id, expected ggml name, SNR floor in dB). Floors are well under measured
    // values, they catch a broken path, not a rounding change.
    const cases = [_]struct { id: u32, name: []const u8, snr_floor: f64 }{
        .{ .id = id_q8_0, .name = "q8_0", .snr_floor = 30 },
        .{ .id = id_q6_k, .name = "q6_K", .snr_floor = 25 },
        .{ .id = id_q5_k, .name = "q5_K", .snr_floor = 20 },
        .{ .id = id_q4_k, .name = "q4_K", .snr_floor = 15 },
        .{ .id = id_q3_k, .name = "q3_K", .snr_floor = 10 },
        .{ .id = id_q2_k, .name = "q2_K", .snr_floor = 5 },
        .{ .id = id_q5_0, .name = "q5_0", .snr_floor = 18 },
        .{ .id = id_q4_0, .name = "q4_0", .snr_floor = 12 },
    };
    for (cases) |c| {
        // Fail loudly if upstream ever renumbers, rather than testing the wrong type.
        try std.testing.expectEqualStrings(c.name, try raw.name(c.id));

        const row_bytes = try raw.rowBytes(c.id, n);
        var enc: [512]u8 = undefined;
        try std.testing.expect(row_bytes <= enc.len);
        const written = try raw.quantizeChunk(c.id, &src, enc[0..row_bytes], 1, n, null);
        try std.testing.expectEqual(row_bytes, written);

        var back: [n]f32 = undefined;
        try raw.dequantRow(c.id, enc[0..row_bytes], n, &back);

        var sig: f64 = 0;
        var err: f64 = 0;
        for (src, back) |a, b| {
            sig += @as(f64, a) * a;
            const d = @as(f64, a) - b;
            err += d * d;
            try std.testing.expect(std.math.isFinite(b));
        }
        const snr = 10.0 * std.math.log10(sig / err);
        errdefer std.debug.print("{s}: {d} B/row, snr {d:.2} dB (floor {d})\n", .{ c.name, row_bytes, snr, c.snr_floor });
        try std.testing.expect(snr > c.snr_floor);
    }
}

test "raw quantize is deterministic and matches the typed wrapper" {
    if (!have_ggml) return error.SkipZigTest;
    const n = 256;
    var src: [n]f32 = undefined;
    fillTestWeights(&src, 42);

    var a: [144]u8 = undefined; // q4_k row size for 256 elements
    var b: [144]u8 = undefined;
    var c: [144]u8 = undefined;
    _ = try raw.quantizeChunk(id_q4_k, &src, &a, 1, n, null);
    _ = try raw.quantizeChunk(id_q4_k, &src, &b, 1, n, null);
    _ = try quantizeChunk(.q4_k, &src, &c, 1, n, null);
    try std.testing.expectEqualSlices(u8, &a, &b); // same input -> same bytes
    try std.testing.expectEqualSlices(u8, &a, &c); // typed wrapper is the same call
}

test "raw dequantRow agrees with the fixture-validated dequantSlice" {
    if (!have_ggml) return error.SkipZigTest;
    // Ties the new entry point to the golden path: same bytes in, same floats out.
    var via_slice: [32]f32 = undefined;
    var via_raw: [32]f32 = undefined;
    dequantSlice(.q8_0, &fixtures.q8_0_block, 0, 32, &via_slice);
    try raw.dequantRow(id_q8_0, &fixtures.q8_0_block, 32, &via_raw);
    try std.testing.expectEqualSlices(f32, &via_slice, &via_raw);
}

test "the imatrix steers the q4_k scale search toward the weighted columns" {
    if (!have_ggml) return error.SkipZigTest;
    // The activation-aware hook has to actually reach ggml's scale search,
    // otherwise every "activation-aware" number downstream is a placebo.
    //
    // Two properties, checked separately:
    //  1. Passing an imatrix changes the output bytes at all (it is plumbed).
    //  2. WHICH columns it favours changes the fit. Asserted by comparing two
    //     imatrices against each other rather than against the unweighted encode:
    //     ggml takes a different algorithm branch when quant_weights is non-null
    //     (make_qkx3_quants vs make_qkx2_quants), so weighted-vs-unweighted mixes
    //     the weighting effect with an algorithm change. A-vs-B holds the
    //     algorithm fixed and varies only the weights.
    //
    // The weights must also vary WITHIN a 32-element sub-block: q4_k picks one
    // scale per sub-block, and scaling every weight in a sub-block by a constant
    // leaves that scale's optimum unchanged (a constant factor drops out of the
    // argmin). Hence the even/odd interleave, not a first-half/second-half split.
    const n = 256;
    var src: [n]f32 = undefined;
    fillTestWeights(&src, 7);

    var im_even: [n]f32 = undefined;
    var im_odd: [n]f32 = undefined;
    for (0..n) |i| {
        const hot = i % 2 == 0;
        im_even[i] = if (hot) 1000.0 else 0.001;
        im_odd[i] = if (hot) 0.001 else 1000.0;
    }

    var plain: [144]u8 = undefined; // q4_k row size for 256 elements
    var enc_even: [144]u8 = undefined;
    var enc_odd: [144]u8 = undefined;
    _ = try raw.quantizeChunk(id_q4_k, &src, &plain, 1, n, null);
    _ = try raw.quantizeChunk(id_q4_k, &src, &enc_even, 1, n, &im_even);
    _ = try raw.quantizeChunk(id_q4_k, &src, &enc_odd, 1, n, &im_odd);

    // (1) plumbed at all.
    try std.testing.expect(!std.mem.eql(u8, &plain, &enc_even));
    // Weighting opposite column sets cannot produce the same encoding.
    try std.testing.expect(!std.mem.eql(u8, &enc_even, &enc_odd));

    var deq_even: [n]f32 = undefined;
    var deq_odd: [n]f32 = undefined;
    try raw.dequantRow(id_q4_k, &enc_even, n, &deq_even);
    try raw.dequantRow(id_q4_k, &enc_odd, n, &deq_odd);

    // Squared error on the even columns and on the odd columns, under each imatrix.
    var err: [2][2]f64 = @splat(@splat(0));
    for (0..n) |i| {
        const parity: usize = i % 2;
        const de = @as(f64, src[i]) - deq_even[i];
        const do_ = @as(f64, src[i]) - deq_odd[i];
        err[0][parity] += de * de;
        err[1][parity] += do_ * do_;
    }
    errdefer std.debug.print(
        "err[imatrix][columns]: even-weighted {e:.4}/{e:.4}, odd-weighted {e:.4}/{e:.4} (even/odd cols)\n",
        .{ err[0][0], err[0][1], err[1][0], err[1][1] },
    );
    // (2) each imatrix fits its own favoured columns better than the other one does.
    try std.testing.expect(err[0][0] < err[1][0]); // even columns: even-weighted wins
    try std.testing.expect(err[1][1] < err[0][1]); // odd columns: odd-weighted wins
}

fn expectGolden(dt: DType, block: []const u8, expected_bits: []const u32) !void {
    if (!have_ggml) return error.SkipZigTest; // dequant needs the ggml backend
    const n = dt.blockElems();
    var out: [256]f32 = undefined;
    dequantSlice(dt, block, 0, n, out[0..n]);
    for (expected_bits, out[0..n], 0..) |bits, got, i| {
        const want: f32 = @bitCast(bits);
        std.testing.expectEqual(want, got) catch |err| {
            std.debug.print("{t} elem {d}: want {x:0>8} got {x:0>8}\n", .{ dt, i, bits, @as(u32, @bitCast(got)) });
            return err;
        };
    }
}

test "q8_0 dequant matches ggml reference" {
    try expectGolden(.q8_0, &fixtures.q8_0_block, &fixtures.q8_0_expected_bits);
}

test "q4_k dequant matches ggml reference" {
    try expectGolden(.q4_k, &fixtures.q4_k_block, &fixtures.q4_k_expected_bits);
}

test "q5_k dequant matches ggml reference" {
    try expectGolden(.q5_k, &fixtures.q5_k_block, &fixtures.q5_k_expected_bits);
}

test "q6_k dequant matches ggml reference" {
    try expectGolden(.q6_k, &fixtures.q6_k_block, &fixtures.q6_k_expected_bits);
}

test "iq4_nl dequant matches the non-linear LUT" {
    if (!have_ggml) return error.SkipZigTest; // dequant needs the ggml backend
    // IQ4_NL: 32-elem block = f16 d + 16 nibble bytes; low nibble -> y[j],
    // high nibble -> y[j+16], value = d * kvalues_iq4nl[nibble].
    const kv = [16]f32{ -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };
    var block: [18]u8 = undefined;
    std.mem.writeInt(u16, block[0..2], @bitCast(@as(f16, 2.0)), .little); // d = 2.0
    block[2] = 0x21; // qs[0]: low nibble 1, high nibble 2
    @memset(block[3..], 0); // qs[1..15] = 0 -> both nibbles index kv[0]
    var out: [32]f32 = undefined;
    dequantSlice(.iq4_nl, &block, 0, 32, &out);
    try std.testing.expectEqual(2.0 * kv[1], out[0]); // low nibble of qs[0]
    try std.testing.expectEqual(2.0 * kv[2], out[16]); // high nibble of qs[0]
    for (1..16) |j| {
        try std.testing.expectEqual(2.0 * kv[0], out[j]);
        try std.testing.expectEqual(2.0 * kv[0], out[j + 16]);
    }
}

test "iq4_xs dequant matches ggml reference" {
    try expectGolden(.iq4_xs, &fixtures.iq4_xs_block, &fixtures.iq4_xs_expected_bits);
}

test "q1_0 dequant is sign-bit x block scale" {
    if (!have_ggml) return error.SkipZigTest; // dequant needs the ggml backend
    // Q1_0: 128-elem block = f16 d + 16 bytes of sign bits, LSB-first within each
    // byte (bit j of byte j/8 is element j); set = +d, clear = -d. There is no
    // zero, which is what makes it different in kind from every other block quant
    // here, a cleared bit is -d, not 0.
    var block: [18]u8 = undefined;
    std.mem.writeInt(u16, block[0..2], @bitCast(@as(f16, 0.25)), .little);
    @memset(block[2..], 0); // every element = -d
    block[2] = 0b0000_1001; // elements 0 and 3 = +d
    block[3] = 0b1000_0000; // element 15 = +d
    block[17] = 0b0000_0010; // element 121 = +d

    var out: [128]f32 = undefined;
    dequantSlice(.q1_0, &block, 0, 128, &out);
    for (out, 0..) |v, i| {
        const positive = i == 0 or i == 3 or i == 15 or i == 121;
        errdefer std.debug.print("q1_0 elem {d}: got {d}\n", .{ i, v });
        try std.testing.expectEqual(@as(f32, if (positive) 0.25 else -0.25), v);
    }
}

test "q1_0 round-trips through ggml as the sign of the input" {
    if (!have_ggml) return error.SkipZigTest;
    // Pins the format end to end through the *quantizer* too, so the bit order
    // above is the one ggml actually writes and not just the one it reads. The
    // reconstruction is exact in direction and uniform in magnitude
    // (d = mean|x| over the block), which is the whole content of the format.
    const id_q1_0: u32 = 41;
    try std.testing.expectEqualStrings("q1_0", try raw.name(id_q1_0));
    try std.testing.expectEqual(@as(usize, 128), try raw.blockElems(id_q1_0));
    try std.testing.expectEqual(@as(usize, 18), try raw.blockBytes(id_q1_0));
    try std.testing.expectEqual(id_q1_0, @as(u32, @intCast(ggmlType(.q1_0).?)));

    const n = 256; // two blocks
    var src: [n]f32 = undefined;
    fillTestWeights(&src, 0xB05A1);

    var enc: [36]u8 = undefined;
    try std.testing.expectEqual(enc.len, try raw.rowBytes(id_q1_0, n));
    _ = try raw.quantizeChunk(id_q1_0, &src, &enc, 1, n, null);
    var back: [n]f32 = undefined;
    try raw.dequantRow(id_q1_0, &enc, n, &back);

    for (0..2) |b| {
        var mean_abs: f64 = 0;
        for (src[b * 128 ..][0..128]) |v| mean_abs += @abs(v);
        mean_abs /= 128;
        const d: f32 = @floatCast(@as(f16, @floatCast(mean_abs))); // stored as f16
        for (src[b * 128 ..][0..128], back[b * 128 ..][0..128], 0..) |a, got, i| {
            errdefer std.debug.print("block {d} elem {d}: src {d} got {d} d {d}\n", .{ b, i, a, got, d });
            // `>= 0` is set, so +0.0 encodes as +d, matching the reference's
            // comparison rather than a signbit test.
            try std.testing.expectEqual(if (a >= 0) d else -d, got);
        }
    }
    // Also reachable through the typed dtype path, with the same bytes.
    var typed: [36]u8 = undefined;
    _ = try quantizeChunk(.q1_0, &src, &typed, 1, n, null);
    try std.testing.expectEqualSlices(u8, &enc, &typed);
}

test "q2_0 dequant is a 2-bit code offset by one, at both block sizes" {
    // Q2_0: f16 d + 2-bit codes, 4 per byte, LSB-first (element j is bits
    // [2*(j%4) .. +2) of byte j/4). v = (code - 1) * d, so the set is
    // {-d, 0, +d, +2d}. NOT symmetric, +2d is representable and -2d is not,
    // unlike ggml's ternary tq2_0, which this shares no layout with.
    //
    // Both arms of the ambiguous type id 42 are pinned here with the SAME code
    // pattern in the first block, so the test states exactly what differs: the
    // arithmetic is identical and only the block stride moves.
    inline for (.{ DType.q2_0_g64, DType.q2_0_g128 }) |dt| {
        if (dt == .q2_0_g64 and !have_ggml) continue; // g64 decodes via ggml
        const be = comptime dt.blockElems();
        const bb = comptime dt.blockBytes();
        var block: [bb]u8 = undefined;
        std.mem.writeInt(u16, block[0..2], @bitCast(@as(f16, 0.25)), .little);
        @memset(block[2..], 0b0101_0101); // code 1 everywhere => 0.0
        block[2] = 0b1110_0100; // elements 0..3 = codes 0,1,2,3
        block[bb - 1] = 0b0000_0011; // last 4 elements = codes 3,0,0,0

        var out: [be]f32 = undefined;
        dequantSlice(dt, &block, 0, be, &out);
        for (out, 0..) |v, i| {
            const code: f32 = switch (i) {
                0 => 0,
                1 => 1,
                2 => 2,
                3 => 3,
                be - 4 => 3,
                be - 3, be - 2, be - 1 => 0,
                else => 1,
            };
            errdefer std.debug.print("{t} elem {d}: got {d}\n", .{ dt, i, v });
            try std.testing.expectEqual((code - 1) * 0.25, v);
        }
    }
}

test "the two q2_0 variants decode the same bytes differently" {
    if (!have_ggml) return error.SkipZigTest; // g64 goes through ggml
    // The whole reason `gguf.detectQ2_0Variant` exists: one byte stream is valid
    // under both block sizes and means different things. 128 elements of codes is
    // either ONE g128 block (34 B) or TWO g64 blocks (36 B), so laid out as g64,
    // byte 18 is a second scale, while g128 reads that same byte as 4 codes.
    //
    // Pinning the disagreement is what gives the detector's tests teeth: if these
    // ever agreed, mis-detection would be harmless and the detector pointless.
    var buf: [36]u8 = undefined;
    @memset(&buf, 0b1110_0100); // codes 0,1,2,3 repeating
    std.mem.writeInt(u16, buf[0..2], @bitCast(@as(f16, 0.5)), .little);
    std.mem.writeInt(u16, buf[18..20], @bitCast(@as(f16, 0.25)), .little);

    var as_g64: [128]f32 = undefined;
    var as_g128: [128]f32 = undefined;
    dequantSlice(.q2_0_g64, &buf, 0, 128, &as_g64);
    dequantSlice(.q2_0_g128, buf[0..34], 0, 128, &as_g128);
    try std.testing.expect(!std.mem.eql(f32, &as_g64, &as_g128));
    // Concretely: g64's second block re-reads a scale at byte 18 (0.25), so its
    // element 64 is a scaled code; g128 reads byte 18 as codes and its element
    // 64 is (code-1) * 0.5. Both are "plausible", that is the hazard.
    try std.testing.expectEqual(@as(f32, -0.25), as_g64[64]);
    try std.testing.expectEqual(@as(f32, -0.5), as_g128[64]);
}

test "q2_0_g64 round-trips through ggml as round(x / amax)" {
    if (!have_ggml) return error.SkipZigTest;
    // Pins the g64 arm end to end through ggml's own *quantizer*, so the bit
    // order is the one ggml writes and not just the one it reads. There is no
    // equivalent for g128: nothing here encodes it, and the reference encoder
    // lives in a llama.cpp fork we do not build.
    const id_q2_0: u32 = 42;
    try std.testing.expectEqualStrings("q2_0", try raw.name(id_q2_0));
    try std.testing.expectEqual(@as(usize, 64), try raw.blockElems(id_q2_0));
    try std.testing.expectEqual(@as(usize, 18), try raw.blockBytes(id_q2_0));
    try std.testing.expectEqual(id_q2_0, @as(u32, @intCast(ggmlType(.q2_0_g64).?)));
    // And the g128 arm must NOT reach ggml, whose type 42 is the g64 layout.
    try std.testing.expectEqual(@as(?@TypeOf(ggmlType(.q8_0).?), null), ggmlType(.q2_0_g128));

    const n = 256; // four blocks
    var src: [n]f32 = undefined;
    fillTestWeights(&src, 0xB2050);

    var enc: [72]u8 = undefined;
    try std.testing.expectEqual(enc.len, try raw.rowBytes(id_q2_0, n));
    _ = try raw.quantizeChunk(id_q2_0, &src, &enc, 1, n, null);
    var back: [n]f32 = undefined;
    try raw.dequantRow(id_q2_0, &enc, n, &back);

    for (0..4) |b| {
        // The scale is the block's max |x| (NOT its mean, which is q1_0's rule),
        // so every |x/d| <= 1 and the reference's clamp to [0, 3] never fires:
        // a *symmetric* quantizer only ever emits codes 0..2. The asymmetric +2d
        // code is unreachable from this encoder and exists for the imatrix path.
        var amax: f32 = 0;
        for (src[b * 64 ..][0..64]) |v| amax = @max(amax, @abs(v));
        const d: f32 = @floatCast(@as(f16, @floatCast(amax))); // stored as f16
        const id: f32 = if (amax > 0) 1.0 / amax else 0.0; // reference divides by the f32 amax
        for (src[b * 64 ..][0..64], back[b * 64 ..][0..64], 0..) |a, got, i| {
            errdefer std.debug.print("block {d} elem {d}: src {d} got {d} d {d}\n", .{ b, i, a, got, d });
            try std.testing.expectEqual(@round(a * id) * d, got);
        }
    }
    // Also reachable through the typed dtype path, with the same bytes.
    var typed: [72]u8 = undefined;
    _ = try quantizeChunk(.q2_0_g64, &src, &typed, 1, n, null);
    try std.testing.expectEqualSlices(u8, &enc, &typed);
}

test "dequantSlice block-aligned sub-ranges" {
    if (!have_ggml) return error.SkipZigTest; // dequant needs the ggml backend
    // Dequanting a 2-block row in one call or block-by-block must agree.
    var row: [68]u8 = undefined;
    @memcpy(row[0..34], &fixtures.q8_0_block);
    @memcpy(row[34..68], &fixtures.q8_0_block);
    row[36] = 0x7f; // perturb block 1's quants so the halves differ

    var whole: [64]f32 = undefined;
    dequantSlice(.q8_0, &row, 0, 64, &whole);
    var lo: [32]f32 = undefined;
    var hi: [32]f32 = undefined;
    dequantSlice(.q8_0, &row, 0, 32, &lo);
    dequantSlice(.q8_0, &row, 32, 32, &hi);
    try std.testing.expectEqualSlices(f32, whole[0..32], &lo);
    try std.testing.expectEqualSlices(f32, whole[32..64], &hi);
}

test "storage sizes match ggml block layouts" {
    try std.testing.expectEqual(@as(usize, 34), DType.q8_0.storageBytes(32));
    try std.testing.expectEqual(@as(usize, 144), DType.q4_k.storageBytes(256));
    try std.testing.expectEqual(@as(usize, 176), DType.q5_k.storageBytes(256));
    try std.testing.expectEqual(@as(usize, 210), DType.q6_k.storageBytes(256));
    // q1_0 is 18 B per *128* elements, so a 256-element row is two blocks.
    try std.testing.expectEqual(@as(usize, 36), DType.q1_0.storageBytes(256));
    // A Bonsai-27B hidden row: 5120 = 40 q1_0 blocks.
    try std.testing.expectEqual(@as(usize, 720), DType.q1_0.storageBytes(5120));
    // The two q2_0 variants differ by exactly 17/18 on any row, the only signal
    // that tells them apart in a file (see gguf.detectQ2_0Variant). On Bonsai's
    // 5120-wide row that is 1440 vs 1360 bytes.
    try std.testing.expectEqual(@as(usize, 1440), DType.q2_0_g64.storageBytes(5120));
    try std.testing.expectEqual(@as(usize, 1360), DType.q2_0_g128.storageBytes(5120));
    try std.testing.expectEqual(@as(usize, 72), DType.q2_0_g64.storageBytes(256));
    try std.testing.expectEqual(@as(usize, 68), DType.q2_0_g128.storageBytes(256));
    // A Qwen3-4B hidden row: 2560 = 10 super-blocks.
    try std.testing.expectEqual(@as(usize, 1440), DType.q4_k.storageBytes(2560));
    try std.testing.expect(DType.q4_k.isBlockQuant() and !DType.bf16.isBlockQuant());
}
