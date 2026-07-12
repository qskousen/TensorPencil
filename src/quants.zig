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

inline fn f16At(b: []const u8, off: usize) f32 {
    return dtypes.f16ToF32(std.mem.readInt(u16, b[off..][0..2], .little));
}

// --- int8 activation · block-quant weight dot (decode GEMV) ----------------
//
// The f32 decode GEMV (matmul.runRangeBlock) spends ~75% of its time
// dequantizing the weight to f32 before an f32 dot. Instead, quantize the
// activation to int8 once (per 256-block scale, "q8_K"-style) and dot it
// directly against the packed weight quants with integer arithmetic — the
// CPU twin of the GPU dp4a path (llama.cpp vec_dot_q6_K_q8_K). No f32 weight
// expansion, no scratch round-trip.

/// Quantize one activation vector to int8 with a per-256-block scale.
/// `xi8` is [cols] i8; `xd` is [cols/256] f32 (block scale = amax/127).
/// `cols` must be a multiple of 256 (matches the q6_k super-block).
pub fn quantizeActQ8K(x: []const f32, xi8: []i8, xd: []f32) void {
    const cols = x.len;
    std.debug.assert(cols % 256 == 0 and xi8.len == cols and xd.len == cols / 256);
    var blk: usize = 0;
    while (blk * 256 < cols) : (blk += 1) {
        const xs = x[blk * 256 ..][0..256];
        var amax: f32 = 0;
        for (xs) |v| amax = @max(amax, @abs(v));
        if (amax == 0) {
            xd[blk] = 0;
            @memset(xi8[blk * 256 ..][0..256], 0);
            continue;
        }
        xd[blk] = amax / 127.0;
        const inv = 127.0 / amax;
        for (xs, 0..) |v, i| {
            const q = @round(v * inv);
            xi8[blk * 256 + i] = @intFromFloat(std.math.clamp(q, -127.0, 127.0));
        }
    }
}

/// Dot a q6_k weight row (`row` = its packed bytes) with an int8-quantized
/// activation (`xi8`/`xd` from quantizeActQ8K). Mirrors dequantBlockQ6K's
/// element/scale layout exactly, so it agrees with the f32 path up to the
/// activation's int8 rounding. `cols` = elements in the row (multiple of 256).
pub fn dotQ6KQ8K(row: []const u8, xi8: []const i8, xd: []const f32, cols: usize) f32 {
    std.debug.assert(cols % 256 == 0);
    const block_bytes = comptime DType.q6_k.blockBytes();
    var acc: f32 = 0;
    var blk: usize = 0;
    while (blk * 256 < cols) : (blk += 1) {
        const b = row[blk * block_bytes ..][0..block_bytes];
        const d = f16At(b, 208);
        const sc = b[192..208]; // 16 i8 sub-block scales
        var isum: [16]i32 = @splat(0); // per sub-block: sum (q-32)*xi8
        for (0..2) |half| {
            const ql = b[half * 64 ..][0..64];
            const qh = b[128 + half * 32 ..][0..32];
            const base = blk * 256 + half * 128;
            const s0 = half * 8;
            for (0..32) |l| {
                const li = l / 16; // 0 or 1 within the 16-element sub-block
                const q1 = @as(i32, (ql[l] & 0xF) | (@as(u8, (qh[l] >> 0) & 3) << 4)) - 32;
                const q2 = @as(i32, (ql[l + 32] & 0xF) | (@as(u8, (qh[l] >> 2) & 3) << 4)) - 32;
                const q3 = @as(i32, (ql[l] >> 4) | (@as(u8, (qh[l] >> 4) & 3) << 4)) - 32;
                const q4 = @as(i32, (ql[l + 32] >> 4) | (@as(u8, (qh[l] >> 6) & 3) << 4)) - 32;
                isum[s0 + 0 + li] += q1 * xi8[base + l];
                isum[s0 + 2 + li] += q2 * xi8[base + l + 32];
                isum[s0 + 4 + li] += q3 * xi8[base + l + 64];
                isum[s0 + 6 + li] += q4 * xi8[base + l + 96];
            }
        }
        var block_i: i64 = 0;
        for (0..16) |si| block_i += @as(i64, @as(i8, @bitCast(sc[si]))) * isum[si];
        acc += d * xd[blk] * @as(f32, @floatFromInt(block_i));
    }
    return acc;
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

test "q6_k int8 dot agrees with f32 dequant-dot" {
    const cols = 256;
    // Reference weights, and the f32 dot for a random activation.
    var wf32: [cols]f32 = undefined;
    dequantSlice(.q6_k, &fixtures.q6_k_block, 0, cols, &wf32);
    var prng = std.Random.DefaultPrng.init(0xC0FFEE);
    const rnd = prng.random();
    var x: [cols]f32 = undefined;
    for (&x) |*v| v.* = (rnd.float(f32) - 0.5) * 3.0;

    var ref: f32 = 0;
    for (wf32, x) |wv, xv| ref += wv * xv;

    var xi8: [cols]i8 = undefined;
    var xd: [cols / 256]f32 = undefined;
    quantizeActQ8K(&x, &xi8, &xd);
    const got = dotQ6KQ8K(&fixtures.q6_k_block, &xi8, &xd, cols);

    const rel = @abs(got - ref) / (@abs(ref) + 1e-3);
    std.testing.expect(rel < 0.02) catch |err| {
        std.debug.print("q6_k int8 dot: ref {d:.5} got {d:.5} rel {d:.5}\n", .{ ref, got, rel });
        return err;
    };
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
