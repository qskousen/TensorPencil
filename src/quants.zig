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

const DType = dtypes.DType;

/// Dequantize elements [elem0, elem0 + n) of a block-quantized `row` into
/// `dst`. `elem0` and `n` must be block-aligned (blocks never span rows in
/// ggml, and callers slice at block-aligned offsets).
pub fn dequantSlice(dt: DType, row: []const u8, elem0: usize, n: usize, dst: []f32) void {
    std.debug.assert(dst.len >= n);
    switch (dt) {
        .q8_0 => dequantSliceTyped(.q8_0, row, elem0, n, dst),
        .q4_k => dequantSliceTyped(.q4_k, row, elem0, n, dst),
        .q5_k => dequantSliceTyped(.q5_k, row, elem0, n, dst),
        .q6_k => dequantSliceTyped(.q6_k, row, elem0, n, dst),
        else => unreachable, // not a block-quantized dtype
    }
}

fn dequantSliceTyped(comptime dt: DType, row: []const u8, elem0: usize, n: usize, dst: []f32) void {
    const elems = comptime dt.blockElems();
    const bytes = comptime dt.blockBytes();
    std.debug.assert(elem0 % elems == 0 and n % elems == 0);
    std.debug.assert(row.len >= (elem0 + n) / elems * bytes);

    var block = elem0 / elems;
    const block_end = (elem0 + n) / elems;
    var out: usize = 0;
    while (block < block_end) : (block += 1) {
        const b = row[block * bytes ..][0..bytes];
        const d = dst[out..][0..elems];
        switch (dt) {
            .q8_0 => dequantBlockQ8_0(b, d),
            .q4_k => dequantBlockQ4K(b, d),
            .q5_k => dequantBlockQ5K(b, d),
            .q6_k => dequantBlockQ6K(b, d),
            else => unreachable,
        }
        out += elems;
    }
}

inline fn f16At(b: []const u8, off: usize) f32 {
    return dtypes.f16ToF32(std.mem.readInt(u16, b[off..][0..2], .little));
}

fn dequantBlockQ8_0(b: *const [34]u8, dst: *[32]f32) void {
    const d = f16At(b, 0);
    for (dst, b[2..34]) |*v, q| v.* = @as(f32, @floatFromInt(@as(i8, @bitCast(q)))) * d;
}

/// 6-bit sub-block scale and min `j` (0..7) from the packed 12-byte table.
inline fn scaleMinK4(j: usize, q: *const [12]u8) struct { sc: f32, m: f32 } {
    if (j < 4) return .{
        .sc = @floatFromInt(q[j] & 63),
        .m = @floatFromInt(q[j + 4] & 63),
    };
    return .{
        .sc = @floatFromInt((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)),
        .m = @floatFromInt((q[j + 4] >> 4) | ((q[j] >> 6) << 4)),
    };
}

fn dequantBlockQ4K(b: *const [144]u8, dst: *[256]f32) void {
    const d = f16At(b, 0);
    const dmin = f16At(b, 2);
    const scales = b[4..16];
    const qs = b[16..144];

    var is: usize = 0;
    var q: usize = 0;
    var y: usize = 0;
    while (y < 256) : (y += 64) {
        const s1 = scaleMinK4(is, scales);
        const d1 = d * s1.sc;
        const m1 = dmin * s1.m;
        const s2 = scaleMinK4(is + 1, scales);
        const d2 = d * s2.sc;
        const m2 = dmin * s2.m;
        for (0..32) |l| dst[y + l] = d1 * @as(f32, @floatFromInt(qs[q + l] & 0xF)) - m1;
        for (0..32) |l| dst[y + 32 + l] = d2 * @as(f32, @floatFromInt(qs[q + l] >> 4)) - m2;
        q += 32;
        is += 2;
    }
}

fn dequantBlockQ5K(b: *const [176]u8, dst: *[256]f32) void {
    const d = f16At(b, 0);
    const dmin = f16At(b, 2);
    const scales = b[4..16];
    const qh = b[16..48];
    const qs = b[48..176];

    var is: usize = 0;
    var q: usize = 0;
    var y: usize = 0;
    // 5th-bit masks; u32 because the final `<<= 2` walks past a u8 (C wraps).
    var mask1: u32 = 1;
    var mask2: u32 = 2;
    while (y < 256) : (y += 64) {
        const s1 = scaleMinK4(is, scales);
        const d1 = d * s1.sc;
        const m1 = dmin * s1.m;
        const s2 = scaleMinK4(is + 1, scales);
        const d2 = d * s2.sc;
        const m2 = dmin * s2.m;
        for (0..32) |l| {
            const hi: u8 = if (qh[l] & mask1 != 0) 16 else 0;
            dst[y + l] = d1 * @as(f32, @floatFromInt((qs[q + l] & 0xF) + hi)) - m1;
        }
        for (0..32) |l| {
            const hi: u8 = if (qh[l] & mask2 != 0) 16 else 0;
            dst[y + 32 + l] = d2 * @as(f32, @floatFromInt((qs[q + l] >> 4) + hi)) - m2;
        }
        q += 32;
        is += 2;
        mask1 <<= 2;
        mask2 <<= 2;
    }
}

fn dequantBlockQ6K(b: *const [210]u8, dst: *[256]f32) void {
    const d = f16At(b, 208);
    // Two independent 128-element halves.
    for (0..2) |half| {
        const ql = b[half * 64 ..][0..64]; // low nibbles for this half
        const qh = b[128 + half * 32 ..][0..32]; // 2-bit highs
        const sc = b[192 + half * 8 ..][0..8]; // i8 sub-block scales
        const y = dst[half * 128 ..][0..128];
        for (0..32) |l| {
            const is = l / 16;
            const q1 = @as(i32, (ql[l] & 0xF) | ((qh[l] >> 0) & 3) << 4) - 32;
            const q2 = @as(i32, (ql[l + 32] & 0xF) | ((qh[l] >> 2) & 3) << 4) - 32;
            const q3 = @as(i32, (ql[l] >> 4) | ((qh[l] >> 4) & 3) << 4) - 32;
            const q4 = @as(i32, (ql[l + 32] >> 4) | ((qh[l] >> 6) & 3) << 4) - 32;
            y[l] = d * @as(f32, @floatFromInt(@as(i8, @bitCast(sc[is])))) * @as(f32, @floatFromInt(q1));
            y[l + 32] = d * @as(f32, @floatFromInt(@as(i8, @bitCast(sc[is + 2])))) * @as(f32, @floatFromInt(q2));
            y[l + 64] = d * @as(f32, @floatFromInt(@as(i8, @bitCast(sc[is + 4])))) * @as(f32, @floatFromInt(q3));
            y[l + 96] = d * @as(f32, @floatFromInt(@as(i8, @bitCast(sc[is + 6])))) * @as(f32, @floatFromInt(q4));
        }
    }
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
