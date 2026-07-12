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
const ggml = @import("ggml");

const DType = dtypes.DType;

/// ggml type enum for a block-quant DType (null otherwise). Shared with matmul.
pub fn ggmlType(dt: DType) ?ggml.c.enum_ggml_type {
    return switch (dt) {
        .q8_0 => ggml.c.GGML_TYPE_Q8_0,
        .q4_k => ggml.c.GGML_TYPE_Q4_K,
        .q5_k => ggml.c.GGML_TYPE_Q5_K,
        .q6_k => ggml.c.GGML_TYPE_Q6_K,
        else => null,
    };
}

/// Fill ggml's fp16 table / CPU dispatch once (the CPU vec_dot kernels return 0
/// without it). Idempotent; the first block-quant matmul runs on the single
/// main thread before any fan-out, so a plain flag is enough.
var ggml_inited = false;
pub fn ensureGgmlInit() void {
    if (!ggml_inited) {
        ggml.c.ggml_cpu_init();
        ggml_inited = true;
    }
}

/// Dequantize elements [elem0, elem0 + n) of a block-quantized `row` into `dst`
/// via ggml's (auto-vectorized) `to_float` — ~4-12x faster than the scalar Zig
/// decode it replaced. `elem0`/`n` must be block-aligned (ggml blocks never span
/// rows; callers slice at block-aligned offsets). Bit-identical to the ggml
/// reference our golden fixtures were generated from.
pub fn dequantSlice(dt: DType, row: []const u8, elem0: usize, n: usize, dst: []f32) void {
    std.debug.assert(dst.len >= n);
    const be = dt.blockElems();
    std.debug.assert(elem0 % be == 0 and n % be == 0);
    const gt = ggmlType(dt) orelse unreachable; // not a block-quantized dtype
    const x = row.ptr + (elem0 / be) * dt.blockBytes();
    ggml.c.ggml_get_type_traits(gt).*.to_float.?(x, dst.ptr, @intCast(n));
}

// --- tests -----------------------------------------------------------------

const fixtures = @import("quants_fixtures.zig");

fn expectGolden(dt: DType, block: []const u8, expected_bits: []const u32) !void {
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

test "dequantSlice block-aligned sub-ranges" {
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
