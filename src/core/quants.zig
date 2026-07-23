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
            .q4_k => ggml.c.GGML_TYPE_Q4_K,
            .q5_k => ggml.c.GGML_TYPE_Q5_K,
            .q6_k => ggml.c.GGML_TYPE_Q6_K,
            .iq4_nl => ggml.c.GGML_TYPE_IQ4_NL,
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
            "GGUF block-quant (q4_0/q8_0/q4_k/q5_k/q6_k/iq4_nl) is unavailable");
    }
};

/// ggml type enum for a block-quant DType (null otherwise). Shared with matmul.
/// Only meaningful in a ggml build.
pub const ggmlType = gg.blockType;

/// Fill ggml's fp16 table / CPU dispatch once (the CPU vec_dot kernels return 0
/// without it). Idempotent; the first block-quant matmul runs on the single
/// main thread before any fan-out, so a plain flag is enough. No-op without ggml.
pub const ensureGgmlInit = gg.ensureInit;

/// Dequantize elements [elem0, elem0 + n) of a block-quantized `row` into `dst`
/// via ggml's (auto-vectorized) `to_float` — ~4-12x faster than the scalar Zig
/// decode it replaced. `elem0`/`n` must be block-aligned (ggml blocks never span
/// rows; callers slice at block-aligned offsets). Bit-identical to the ggml
/// reference our golden fixtures were generated from. Panics if built without ggml.
pub const dequantSlice = gg.dequantSlice;

// --- tests -----------------------------------------------------------------

const fixtures = @import("quants_fixtures.zig");

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
    // A Qwen3-4B hidden row: 2560 = 10 super-blocks.
    try std.testing.expectEqual(@as(usize, 1440), DType.q4_k.storageBytes(2560));
    try std.testing.expect(DType.q4_k.isBlockQuant() and !DType.bf16.isBlockQuant());
}
