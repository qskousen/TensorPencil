//! Block-quant dequantizers, one thread per output pair (16-bit outputs) or per
//! element (f32). The weight is the RAW row-major ggml layout read through the u32
//! view of slot a; the result goes to slot b. u0 = element count. Blocks never
//! straddle a row (every consumer keeps cols a multiple of the block), and a pair
//! never straddles a block (block sizes are even), so the pair's two elements share
//! one block decode.
//!
//! Output width is a comptime choice so the three variants of each format stay one
//! body: f16 and bf16 write pairs, f32 writes elements. `dequant_fp8_*` also takes
//! the 256-entry f32 LUT in slot c and the per-tensor scale in f0.

const k = @import("../dual.zig");
const Env = k.Env;

pub const Out = enum { f16, bf16, f32 };

/// The value of element `e` of the weight, per format.
const Fmt = enum { fp8, q8_0, q4_0, q1_0, q2_0_g64, q2_0_g128, iq4_nl, iq4_xs, q4_k, q5_k, q6_k };

const kvalues_iq4nl = [16]i32{ -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };

inline fn iq4Value(nibble: u32) f32 {
    // A switch, since a lookup into a comptime array by a runtime index lands in
    // private memory on both targets.
    return @floatFromInt(switch (nibble) {
        inline 0...15 => |n| kvalues_iq4nl[n],
        else => unreachable,
    });
}

/// ggml get_scale_min_k4: the 6-bit scale and min of sub-block `is` of a q4_k/q5_k
/// super-block whose 12 scale bytes start at byte offset `sb`.
inline fn scaleMinK4(e: Env, sb: u32, is: u32) [2]u32 {
    if (is < 4) {
        return .{ e.byte(.a, sb + is) & 63, e.byte(.a, sb + is + 4) & 63 };
    }
    const b8 = e.byte(.a, sb + is + 4);
    const b0 = e.byte(.a, sb + is - 4);
    const b4 = e.byte(.a, sb + is);
    return .{ (b8 & 15) | ((b0 >> 6) << 4), (b8 >> 4) | ((b4 >> 6) << 4) };
}

inline fn value(e: Env, comptime fmt: Fmt, idx: u32) f32 {
    switch (fmt) {
        .fp8 => return e.ld(.c, e.byte(.a, idx)) * e.f(0),
        .q8_0 => {
            const blk = (idx >> 5) * 34;
            const q: i32 = @as(i8, @bitCast(@as(u8, @truncate(e.byte(.a, blk + 2 + (idx & 31))))));
            return @as(f32, @floatFromInt(q)) * e.f16At(.a, blk);
        },
        .q4_0, .iq4_nl => {
            const blk = (idx >> 5) * 18;
            const j = idx & 31;
            const qs = e.byte(.a, blk + 2 + (j & 15));
            const nib = if (j < 16) qs & 15 else qs >> 4;
            const v: f32 = if (fmt == .q4_0) @floatFromInt(@as(i32, @intCast(nib)) - 8) else iq4Value(nib);
            return v * e.f16At(.a, blk);
        },
        .q1_0 => {
            const blk = (idx >> 7) * 18;
            const j = idx & 127;
            const bit = (e.byte(.a, blk + 2 + (j >> 3)) >> @intCast(j & 7)) & 1;
            const d = e.f16At(.a, blk);
            return if (bit != 0) d else -d;
        },
        .q2_0_g64, .q2_0_g128 => {
            const qk: u32 = if (fmt == .q2_0_g64) 64 else 128;
            const blk = (idx / qk) * (2 + qk / 4);
            const j = idx % qk;
            const code = (e.byte(.a, blk + 2 + (j >> 2)) >> @intCast(2 * (j & 3))) & 3;
            return @as(f32, @floatFromInt(@as(i32, @intCast(code)) - 1)) * e.f16At(.a, blk);
        },
        .iq4_xs => {
            const blk = (idx >> 8) * 136;
            const j = idx & 255;
            const ib = j >> 5;
            const ls_l = (e.byte(.a, blk + 4 + (ib >> 1)) >> @intCast((ib & 1) * 4)) & 15;
            const ls_h = (e.u16At(.a, blk + 2) >> @intCast(2 * ib)) & 3;
            const dl = e.f16At(.a, blk) * @as(f32, @floatFromInt(@as(i32, @intCast(ls_l | (ls_h << 4))) - 32));
            const qs = e.byte(.a, blk + 8 + (ib << 4) + (j & 15));
            const nib = if (j & 16 == 0) qs & 15 else qs >> 4;
            return iq4Value(nib) * dl;
        },
        .q4_k, .q5_k => {
            const bb: u32 = if (fmt == .q4_k) 144 else 176;
            const blk = (idx >> 8) * bb;
            const j = idx & 255;
            const is = j >> 5;
            const sm = scaleMinK4(e, blk + 4, is);
            const d = e.f16At(.a, blk) * @as(f32, @floatFromInt(sm[0]));
            const m = e.f16At(.a, blk + 2) * @as(f32, @floatFromInt(sm[1]));
            const qs_off: u32 = if (fmt == .q4_k) 16 else 48;
            const qs = e.byte(.a, blk + qs_off + ((j >> 6) << 5) + (j & 31));
            var q = (qs >> @intCast(((j >> 5) & 1) * 4)) & 15;
            if (fmt == .q5_k) {
                const qh = e.byte(.a, blk + 16 + (j & 31));
                q += ((qh >> @intCast((j >> 5) & 7)) & 1) << 4;
            }
            return @as(f32, @floatFromInt(q)) * d - m;
        },
        .q6_k => {
            const blk = (idx >> 8) * 210;
            const j = idx & 255;
            const sc: i32 = @as(i8, @bitCast(@as(u8, @truncate(e.byte(.a, blk + 192 + (j >> 4))))));
            const d = e.f16At(.a, blk + 208) * @as(f32, @floatFromInt(sc));
            const l = j & 31;
            const half = j >> 7;
            const ql = e.byte(.a, blk + half * 64 + ((j >> 5) & 1) * 32 + l);
            const qh = e.byte(.a, blk + 128 + half * 32 + l);
            const lo = (ql >> @intCast(((j >> 6) & 1) * 4)) & 15;
            const hi = (qh >> @intCast(((j >> 5) & 3) * 2)) & 3;
            return d * @as(f32, @floatFromInt(@as(i32, @intCast(lo | (hi << 4))) - 32));
        },
    }
}

inline fn dequant(e: Env, comptime fmt: Fmt, comptime out: Out) void {
    if (out == .f32) {
        const i = k.elem(e) orelse return;
        e.st(.b, i, value(e, fmt, i));
        return;
    }
    const w = e.gid();
    const e0 = w * 2;
    if (e0 >= e.u(0)) return;
    const lo = value(e, fmt, e0);
    const hi = value(e, fmt, e0 + 1);
    e.stW(.b, w, if (out == .f16) k.packH16(lo, hi) else @as(u32, k.bf16Bits(lo)) | (@as(u32, k.bf16Bits(hi)) << 16));
}

// One body per (format, width); the table names them dequant_<fmt>_<width>.
pub inline fn fp8F16(e: Env) void {
    dequant(e, .fp8, .f16);
}
pub inline fn fp8Bf16(e: Env) void {
    dequant(e, .fp8, .bf16);
}
pub inline fn fp8F32(e: Env) void {
    dequant(e, .fp8, .f32);
}
pub inline fn q8_0F16(e: Env) void {
    dequant(e, .q8_0, .f16);
}
pub inline fn q8_0Bf16(e: Env) void {
    dequant(e, .q8_0, .bf16);
}
pub inline fn q8_0F32(e: Env) void {
    dequant(e, .q8_0, .f32);
}
pub inline fn q4_0F16(e: Env) void {
    dequant(e, .q4_0, .f16);
}
pub inline fn q4_0Bf16(e: Env) void {
    dequant(e, .q4_0, .bf16);
}
pub inline fn q4_0F32(e: Env) void {
    dequant(e, .q4_0, .f32);
}
pub inline fn q1_0F16(e: Env) void {
    dequant(e, .q1_0, .f16);
}
pub inline fn q1_0Bf16(e: Env) void {
    dequant(e, .q1_0, .bf16);
}
pub inline fn q1_0F32(e: Env) void {
    dequant(e, .q1_0, .f32);
}
pub inline fn q2_0G64F16(e: Env) void {
    dequant(e, .q2_0_g64, .f16);
}
pub inline fn q2_0G64Bf16(e: Env) void {
    dequant(e, .q2_0_g64, .bf16);
}
pub inline fn q2_0G64F32(e: Env) void {
    dequant(e, .q2_0_g64, .f32);
}
pub inline fn q2_0G128F16(e: Env) void {
    dequant(e, .q2_0_g128, .f16);
}
pub inline fn q2_0G128Bf16(e: Env) void {
    dequant(e, .q2_0_g128, .bf16);
}
pub inline fn q2_0G128F32(e: Env) void {
    dequant(e, .q2_0_g128, .f32);
}
pub inline fn iq4NlF16(e: Env) void {
    dequant(e, .iq4_nl, .f16);
}
pub inline fn iq4NlBf16(e: Env) void {
    dequant(e, .iq4_nl, .bf16);
}
pub inline fn iq4NlF32(e: Env) void {
    dequant(e, .iq4_nl, .f32);
}
pub inline fn iq4XsF16(e: Env) void {
    dequant(e, .iq4_xs, .f16);
}
pub inline fn iq4XsBf16(e: Env) void {
    dequant(e, .iq4_xs, .bf16);
}
pub inline fn iq4XsF32(e: Env) void {
    dequant(e, .iq4_xs, .f32);
}
pub inline fn q4_kF16(e: Env) void {
    dequant(e, .q4_k, .f16);
}
pub inline fn q4_kBf16(e: Env) void {
    dequant(e, .q4_k, .bf16);
}
pub inline fn q4_kF32(e: Env) void {
    dequant(e, .q4_k, .f32);
}
pub inline fn q5_kF16(e: Env) void {
    dequant(e, .q5_k, .f16);
}
pub inline fn q5_kBf16(e: Env) void {
    dequant(e, .q5_k, .bf16);
}
pub inline fn q5_kF32(e: Env) void {
    dequant(e, .q5_k, .f32);
}
pub inline fn q6_kF16(e: Env) void {
    dequant(e, .q6_k, .f16);
}
pub inline fn q6_kBf16(e: Env) void {
    dequant(e, .q6_k, .bf16);
}
pub inline fn q6_kF32(e: Env) void {
    dequant(e, .q6_k, .f32);
}
