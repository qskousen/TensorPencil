//! Element data types for model weights and activations, plus conversions to f32.
//!
//! The fp8 type used by the Krea 2 checkpoints is e4m3fn (torch.float8_e4m3fn):
//! 1 sign, 4 exponent (bias 7), 3 mantissa bits; no infinities — exponent 0xF
//! with mantissa 0x7 is NaN, everything else is a finite number (max ±448).

const std = @import("std");
const builtin = @import("builtin");

/// Element type as it appears in safetensors headers.
pub const DType = enum {
    f8_e4m3,
    f16,
    bf16,
    f32,
    f64,
    u8,
    i8,
    /// Signed 4-bit, two values packed per byte (element 2k in the low nibble,
    /// 2k+1 in the high nibble). Internal compute dtype only — int4 "convrot"
    /// weights are stored on disk as raw `U8` (shape [rows, cols/2]) and the
    /// DiT loader reinterprets them as `.i4` with the logical [rows, cols].
    /// Sub-byte, so `byteSize` is undefined; use `storageBytes` for lengths.
    i4,
    u16,
    i16,
    u32,
    i32,
    u64,
    i64,
    bool,
    /// ggml/GGUF block-quantized formats: a block of elements shares packed
    /// scale metadata, so the per-element size is fractional — `byteSize` is
    /// undefined; use `storageBytes` (whole blocks only). Never appears in
    /// safetensors headers; produced by the GGUF loader. Layouts and
    /// dequantization live in quants.zig.
    q4_0, // 32 elems / 18 B: f16 d + 16 B nibbles; v = (nibble - 8) * d
    q8_0, // 32 elems / 34 B: f16 scale + 32 x i8
    q4_k, // 256 elems / 144 B: f16 d + f16 dmin + 12 B 6-bit scales/mins + 128 B nibbles
    q5_k, // 256 elems / 176 B: q4_k + 32 B high bits
    q6_k, // 256 elems / 210 B: 128 B low nibbles + 64 B high 2-bits + 16 x i8 scales + f16 d
    iq4_nl, // 32 elems / 18 B: f16 d + 16 B nibbles; v = d * kvalues_iq4nl[nibble] (non-linear LUT)

    const name_table = .{
        .{ "F8_E4M3", DType.f8_e4m3 },
        .{ "F16", DType.f16 },
        .{ "BF16", DType.bf16 },
        .{ "F32", DType.f32 },
        .{ "F64", DType.f64 },
        .{ "U8", DType.u8 },
        .{ "I8", DType.i8 },
        .{ "U16", DType.u16 },
        .{ "I16", DType.i16 },
        .{ "U32", DType.u32 },
        .{ "I32", DType.i32 },
        .{ "U64", DType.u64 },
        .{ "I64", DType.i64 },
        .{ "BOOL", DType.bool },
    };

    /// Parse a safetensors header dtype string (e.g. "F8_E4M3", "BF16").
    pub fn fromString(s: []const u8) ?DType {
        inline for (name_table) |entry| {
            if (std.mem.eql(u8, s, entry[0])) return entry[1];
        }
        return null;
    }

    /// Per-dtype size facts — the single source of truth the size accessors
    /// below read from (one row per dtype in `info`, instead of the same set
    /// of dtypes re-listed across six parallel switches). Adding a dtype is one
    /// `info` arm.
    pub const Info = struct {
        /// Bytes per element for whole-byte scalar types; null for sub-byte
        /// (`.i4`) and block-quantized types (use `storageBytes`).
        byte_size: ?usize,
        /// Bits per element; null for block-quantized types (fractional).
        bit_size: ?usize,
        /// Elements per quantization block (1 for non-block types).
        block_elems: usize,
        /// Bytes per quantization block; null for non-block types.
        block_bytes: ?usize,
    };

    pub fn info(self: DType) Info {
        return switch (self) {
            .f8_e4m3, .u8, .i8, .bool => .{ .byte_size = 1, .bit_size = 8, .block_elems = 1, .block_bytes = null },
            .f16, .bf16, .u16, .i16 => .{ .byte_size = 2, .bit_size = 16, .block_elems = 1, .block_bytes = null },
            .f32, .u32, .i32 => .{ .byte_size = 4, .bit_size = 32, .block_elems = 1, .block_bytes = null },
            .f64, .u64, .i64 => .{ .byte_size = 8, .bit_size = 64, .block_elems = 1, .block_bytes = null },
            .i4 => .{ .byte_size = null, .bit_size = 4, .block_elems = 1, .block_bytes = null },
            .q4_0 => .{ .byte_size = null, .bit_size = null, .block_elems = 32, .block_bytes = 18 },
            .q8_0 => .{ .byte_size = null, .bit_size = null, .block_elems = 32, .block_bytes = 34 },
            .q4_k => .{ .byte_size = null, .bit_size = null, .block_elems = 256, .block_bytes = 144 },
            .q5_k => .{ .byte_size = null, .bit_size = null, .block_elems = 256, .block_bytes = 176 },
            .q6_k => .{ .byte_size = null, .bit_size = null, .block_elems = 256, .block_bytes = 210 },
            .iq4_nl => .{ .byte_size = null, .bit_size = null, .block_elems = 32, .block_bytes = 18 },
        };
    }

    /// Bytes per element. Undefined for sub-byte (`.i4`) and block-quantized
    /// types — those are unreachable here; use `storageBytes` for their
    /// packed on-disk length.
    pub fn byteSize(self: DType) usize {
        return self.info().byte_size orelse unreachable;
    }

    /// Bits per element (4 for `.i4`, else `byteSize * 8`). Undefined for
    /// block-quantized types (fractional).
    pub fn bitSize(self: DType) usize {
        return self.info().bit_size orelse unreachable;
    }

    /// True for the ggml block-quantized formats (GGUF weights).
    pub fn isBlockQuant(self: DType) bool {
        return self.info().block_bytes != null;
    }

    /// Elements per quantization block (1 for scalar dtypes).
    pub fn blockElems(self: DType) usize {
        return self.info().block_elems;
    }

    /// Bytes per quantization block. Undefined for scalar dtypes.
    pub fn blockBytes(self: DType) usize {
        return self.info().block_bytes orelse unreachable;
    }

    /// Packed storage size in bytes for `count` elements. Handles sub-byte
    /// types (`.i4`: two values per byte, rounding up) and block-quantized
    /// types (`count` must be a multiple of the block size — ggml rows are
    /// whole blocks). For whole-byte types this is just `count * byteSize()`.
    pub fn storageBytes(self: DType, count: usize) usize {
        const inf = self.info();
        if (inf.block_bytes) |bb| {
            std.debug.assert(count % inf.block_elems == 0);
            return (count / inf.block_elems) * bb;
        }
        // Scalar and sub-byte: pack `bit_size` bits per element, rounding up to
        // whole bytes (a whole-byte type reduces to count * byte_size).
        return (count * inf.bit_size.? + 7) / 8;
    }

    /// Decode the signed 4-bit value stored in nibble `idx` (0 = low, 1 = high)
    /// of `byte` to its integer value in [-8, 7].
    pub inline fn nibbleI4(byte: u8, idx: u1) i8 {
        const nib: u4 = @truncate(byte >> (@as(u3, idx) * 4));
        return @as(i8, @as(i4, @bitCast(nib)));
    }
};

/// 2^e, exact for the small exponents used by fp8 decoding. Works at comptime.
fn exp2i(e: i32) f32 {
    var r: f32 = 1.0;
    var i: i32 = 0;
    if (e >= 0) {
        while (i < e) : (i += 1) r *= 2.0;
    } else {
        while (i > e) : (i -= 1) r *= 0.5;
    }
    return r;
}

fn decodeE4m3(byte: u8) f32 {
    const exp: u4 = @truncate(byte >> 3);
    const man: u3 = @truncate(byte);
    const magnitude: f32 = if (exp == 0xF and man == 0x7)
        std.math.nan(f32)
    else if (exp == 0)
        @as(f32, @floatFromInt(man)) * 0x1p-9 // subnormal: (man/8) * 2^-6
    else
        (1.0 + @as(f32, @floatFromInt(man)) * 0.125) * exp2i(@as(i32, exp) - 7);
    return if (byte & 0x80 != 0) -magnitude else magnitude;
}

/// Lookup table mapping every fp8-e4m3fn byte to its f32 value.
pub const f8_e4m3_to_f32_table: [256]f32 = blk: {
    @setEvalBranchQuota(8000);
    var t: [256]f32 = undefined;
    for (&t, 0..) |*v, i| v.* = decodeE4m3(@intCast(i));
    break :blk t;
};

pub inline fn f8e4m3ToF32(byte: u8) f32 {
    return f8_e4m3_to_f32_table[byte];
}

pub inline fn bf16ToF32(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

/// Round-to-nearest-even conversion (matches hardware/torch behavior).
pub fn f32ToBf16(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    if (std.math.isNan(v)) return @truncate((bits >> 16) | 0x0040);
    const round: u32 = 0x7fff + ((bits >> 16) & 1);
    return @truncate((bits + round) >> 16);
}

pub inline fn f16ToF32(bits: u16) f32 {
    const h: f16 = @bitCast(bits);
    return @floatCast(h);
}

/// Vectorized f16 (little-endian bytes) -> f32 row, scaled. `@floatCast` on an
/// 8-wide f16 vector emits hardware vcvtph2ps (F16C) — ~10x the per-element
/// scalar loop it replaces. `src` holds dst.len little-endian u16s.
pub fn f16ToF32Row(src: []const u8, dst: []f32, scale: f32) void {
    const V = 8;
    const sv: @Vector(V, f32) = @splat(scale);
    var i: usize = 0;
    while (i + V <= dst.len) : (i += V) {
        const bits: @Vector(V, u16) = @bitCast(src[i * 2 ..][0 .. V * 2].*);
        const h: @Vector(V, f16) = @bitCast(bits);
        dst[i..][0..V].* = @as(@Vector(V, f32), @floatCast(h)) * sv;
    }
    while (i < dst.len) : (i += 1) dst[i] = f16ToF32(std.mem.readInt(u16, src[i * 2 ..][0..2], .little)) * scale;
}

/// Vectorized bf16 (little-endian bytes) -> f32 row, scaled (bf16 is the high
/// 16 bits of f32, so this is a widening shift — no rounding).
pub fn bf16ToF32Row(src: []const u8, dst: []f32, scale: f32) void {
    const V = 8;
    const sv: @Vector(V, f32) = @splat(scale);
    var i: usize = 0;
    while (i + V <= dst.len) : (i += V) {
        const bits: @Vector(V, u16) = @bitCast(src[i * 2 ..][0 .. V * 2].*);
        const wide: @Vector(V, u32) = @as(@Vector(V, u32), bits) << @splat(16);
        dst[i..][0..V].* = @as(@Vector(V, f32), @bitCast(wide)) * sv;
    }
    while (i < dst.len) : (i += 1) dst[i] = bf16ToF32(std.mem.readInt(u16, src[i * 2 ..][0..2], .little)) * scale;
}

test "dtype string round trip and sizes" {
    try std.testing.expectEqual(DType.f8_e4m3, DType.fromString("F8_E4M3").?);
    try std.testing.expectEqual(DType.bf16, DType.fromString("BF16").?);
    try std.testing.expectEqual(DType.f32, DType.fromString("F32").?);
    try std.testing.expectEqual(@as(?DType, null), DType.fromString("F8_E5M2X"));
    try std.testing.expectEqual(@as(usize, 1), DType.f8_e4m3.byteSize());
    try std.testing.expectEqual(@as(usize, 2), DType.bf16.byteSize());
    try std.testing.expectEqual(@as(usize, 8), DType.i64.byteSize());
}

test "block-quant descriptor: elems, block bytes, storage, isBlockQuant" {
    // ggml block layouts (see the enum doc): q4_0/q8_0 are 32-elem blocks,
    // q4_k/q5_k/q6_k are 256-elem. storageBytes = (count/elems) * block_bytes.
    try std.testing.expect(DType.q4_k.isBlockQuant());
    try std.testing.expect(!DType.f32.isBlockQuant());
    try std.testing.expect(!DType.i4.isBlockQuant());

    try std.testing.expectEqual(@as(usize, 32), DType.q8_0.blockElems());
    try std.testing.expectEqual(@as(usize, 256), DType.q6_k.blockElems());
    try std.testing.expectEqual(@as(usize, 34), DType.q8_0.blockBytes());
    try std.testing.expectEqual(@as(usize, 210), DType.q6_k.blockBytes());

    // One q6_k row of 256 elems = 210 bytes; two blocks of q4_0 = 36 bytes.
    try std.testing.expectEqual(@as(usize, 210), DType.q6_k.storageBytes(256));
    try std.testing.expectEqual(@as(usize, 36), DType.q4_0.storageBytes(64));
    // Whole-byte scalar path still reduces to count * byteSize.
    try std.testing.expectEqual(@as(usize, 12), DType.f32.storageBytes(3));
    try std.testing.expectEqual(@as(usize, 6), DType.f16.storageBytes(3));
}

test "i4 packing and nibble decode" {
    // .i4 is an internal compute dtype, never a safetensors header string.
    try std.testing.expectEqual(@as(?DType, null), DType.fromString("I4"));
    try std.testing.expectEqual(@as(usize, 4), DType.i4.bitSize());
    // Two values per byte, rounding up.
    try std.testing.expectEqual(@as(usize, 128), DType.i4.storageBytes(256));
    try std.testing.expectEqual(@as(usize, 1), DType.i4.storageBytes(1));
    try std.testing.expectEqual(@as(usize, 1), DType.i4.storageBytes(2));

    // Low nibble = element 0, high nibble = element 1; signed [-8, 7].
    // 0xF7: low = 0x7 = 7, high = 0xF = -1.
    try std.testing.expectEqual(@as(i8, 7), DType.nibbleI4(0xF7, 0));
    try std.testing.expectEqual(@as(i8, -1), DType.nibbleI4(0xF7, 1));
    // 0x80: low = 0, high = 0x8 = -8.
    try std.testing.expectEqual(@as(i8, 0), DType.nibbleI4(0x80, 0));
    try std.testing.expectEqual(@as(i8, -8), DType.nibbleI4(0x80, 1));
}

test "fp8 e4m3fn table matches torch.float8_e4m3fn" {
    // f32 bit patterns of torch.arange(256, dtype=uint8).view(float8_e4m3fn).float(),
    // generated with torch 2.6. Entries 0x7f and 0xff are NaN (torch uses a
    // non-canonical NaN payload, so those are compared with isNan instead of bits).
    const expected_bits = [256]u32{
        0x00000000, 0x3b000000, 0x3b800000, 0x3bc00000, 0x3c000000, 0x3c200000, 0x3c400000, 0x3c600000,
        0x3c800000, 0x3c900000, 0x3ca00000, 0x3cb00000, 0x3cc00000, 0x3cd00000, 0x3ce00000, 0x3cf00000,
        0x3d000000, 0x3d100000, 0x3d200000, 0x3d300000, 0x3d400000, 0x3d500000, 0x3d600000, 0x3d700000,
        0x3d800000, 0x3d900000, 0x3da00000, 0x3db00000, 0x3dc00000, 0x3dd00000, 0x3de00000, 0x3df00000,
        0x3e000000, 0x3e100000, 0x3e200000, 0x3e300000, 0x3e400000, 0x3e500000, 0x3e600000, 0x3e700000,
        0x3e800000, 0x3e900000, 0x3ea00000, 0x3eb00000, 0x3ec00000, 0x3ed00000, 0x3ee00000, 0x3ef00000,
        0x3f000000, 0x3f100000, 0x3f200000, 0x3f300000, 0x3f400000, 0x3f500000, 0x3f600000, 0x3f700000,
        0x3f800000, 0x3f900000, 0x3fa00000, 0x3fb00000, 0x3fc00000, 0x3fd00000, 0x3fe00000, 0x3ff00000,
        0x40000000, 0x40100000, 0x40200000, 0x40300000, 0x40400000, 0x40500000, 0x40600000, 0x40700000,
        0x40800000, 0x40900000, 0x40a00000, 0x40b00000, 0x40c00000, 0x40d00000, 0x40e00000, 0x40f00000,
        0x41000000, 0x41100000, 0x41200000, 0x41300000, 0x41400000, 0x41500000, 0x41600000, 0x41700000,
        0x41800000, 0x41900000, 0x41a00000, 0x41b00000, 0x41c00000, 0x41d00000, 0x41e00000, 0x41f00000,
        0x42000000, 0x42100000, 0x42200000, 0x42300000, 0x42400000, 0x42500000, 0x42600000, 0x42700000,
        0x42800000, 0x42900000, 0x42a00000, 0x42b00000, 0x42c00000, 0x42d00000, 0x42e00000, 0x42f00000,
        0x43000000, 0x43100000, 0x43200000, 0x43300000, 0x43400000, 0x43500000, 0x43600000, 0x43700000,
        0x43800000, 0x43900000, 0x43a00000, 0x43b00000, 0x43c00000, 0x43d00000, 0x43e00000, 0x7ff00000,
        0x80000000, 0xbb000000, 0xbb800000, 0xbbc00000, 0xbc000000, 0xbc200000, 0xbc400000, 0xbc600000,
        0xbc800000, 0xbc900000, 0xbca00000, 0xbcb00000, 0xbcc00000, 0xbcd00000, 0xbce00000, 0xbcf00000,
        0xbd000000, 0xbd100000, 0xbd200000, 0xbd300000, 0xbd400000, 0xbd500000, 0xbd600000, 0xbd700000,
        0xbd800000, 0xbd900000, 0xbda00000, 0xbdb00000, 0xbdc00000, 0xbdd00000, 0xbde00000, 0xbdf00000,
        0xbe000000, 0xbe100000, 0xbe200000, 0xbe300000, 0xbe400000, 0xbe500000, 0xbe600000, 0xbe700000,
        0xbe800000, 0xbe900000, 0xbea00000, 0xbeb00000, 0xbec00000, 0xbed00000, 0xbee00000, 0xbef00000,
        0xbf000000, 0xbf100000, 0xbf200000, 0xbf300000, 0xbf400000, 0xbf500000, 0xbf600000, 0xbf700000,
        0xbf800000, 0xbf900000, 0xbfa00000, 0xbfb00000, 0xbfc00000, 0xbfd00000, 0xbfe00000, 0xbff00000,
        0xc0000000, 0xc0100000, 0xc0200000, 0xc0300000, 0xc0400000, 0xc0500000, 0xc0600000, 0xc0700000,
        0xc0800000, 0xc0900000, 0xc0a00000, 0xc0b00000, 0xc0c00000, 0xc0d00000, 0xc0e00000, 0xc0f00000,
        0xc1000000, 0xc1100000, 0xc1200000, 0xc1300000, 0xc1400000, 0xc1500000, 0xc1600000, 0xc1700000,
        0xc1800000, 0xc1900000, 0xc1a00000, 0xc1b00000, 0xc1c00000, 0xc1d00000, 0xc1e00000, 0xc1f00000,
        0xc2000000, 0xc2100000, 0xc2200000, 0xc2300000, 0xc2400000, 0xc2500000, 0xc2600000, 0xc2700000,
        0xc2800000, 0xc2900000, 0xc2a00000, 0xc2b00000, 0xc2c00000, 0xc2d00000, 0xc2e00000, 0xc2f00000,
        0xc3000000, 0xc3100000, 0xc3200000, 0xc3300000, 0xc3400000, 0xc3500000, 0xc3600000, 0xc3700000,
        0xc3800000, 0xc3900000, 0xc3a00000, 0xc3b00000, 0xc3c00000, 0xc3d00000, 0xc3e00000, 0xfff00000,
    };
    for (expected_bits, 0..) |bits, i| {
        const expected: f32 = @bitCast(bits);
        const actual = f8e4m3ToF32(@intCast(i));
        if (std.math.isNan(expected)) {
            try std.testing.expect(std.math.isNan(actual));
        } else {
            try std.testing.expectEqual(bits & 0x7fffffff != bits, std.math.signbit(actual));
            try std.testing.expectEqual(expected, actual);
        }
    }
}

test "fp8 e4m3fn spot values" {
    try std.testing.expectEqual(@as(f32, 0.0), f8e4m3ToF32(0x00));
    try std.testing.expectEqual(@as(f32, 0.001953125), f8e4m3ToF32(0x01)); // smallest subnormal
    try std.testing.expectEqual(@as(f32, 1.0), f8e4m3ToF32(0x38));
    try std.testing.expectEqual(@as(f32, 448.0), f8e4m3ToF32(0x7e)); // max finite
    try std.testing.expectEqual(@as(f32, -4.0), f8e4m3ToF32(0xc8));
    try std.testing.expect(std.math.isNan(f8e4m3ToF32(0x7f)));
    try std.testing.expect(std.math.signbit(f8e4m3ToF32(0x80))); // -0.0
}

test "bf16 conversions" {
    try std.testing.expectEqual(@as(f32, 1.0), bf16ToF32(0x3f80));
    try std.testing.expectEqual(@as(f32, -2.0), bf16ToF32(0xc000));
    try std.testing.expectEqual(@as(u16, 0x3f80), f32ToBf16(1.0));
    // Round to nearest even: 1.0 + 2^-9 is exactly between bf16 values 1.0 and 1.0078125.
    try std.testing.expectEqual(@as(u16, 0x3f80), f32ToBf16(@bitCast(@as(u32, 0x3f808000))));
    try std.testing.expectEqual(@as(u16, 0x3f82), f32ToBf16(@bitCast(@as(u32, 0x3f818000))));
    // Values just above a tie round up.
    try std.testing.expectEqual(@as(u16, 0x3f81), f32ToBf16(@bitCast(@as(u32, 0x3f808001))));
    // Infinities survive.
    try std.testing.expectEqual(@as(u16, 0x7f80), f32ToBf16(std.math.inf(f32)));
    try std.testing.expectEqual(@as(u16, 0xff80), f32ToBf16(-std.math.inf(f32)));
    // NaN stays NaN.
    try std.testing.expect(std.math.isNan(bf16ToF32(f32ToBf16(std.math.nan(f32)))));
    // Round trip of exactly-representable values.
    var b: u32 = 0;
    while (b < 0x100) : (b += 1) {
        const bits: u16 = @intCast(0x3f00 + b);
        try std.testing.expectEqual(bits, f32ToBf16(bf16ToF32(bits)));
    }
}

test "f16 conversion" {
    try std.testing.expectEqual(@as(f32, 1.0), f16ToF32(0x3c00));
    try std.testing.expectEqual(@as(f32, -0.5), f16ToF32(0xb800));
    try std.testing.expectEqual(@as(f32, 65504.0), f16ToF32(0x7bff)); // max f16
}
