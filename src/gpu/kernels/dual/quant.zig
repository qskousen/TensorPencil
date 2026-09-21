//! Block-quant weight decode: the per-element `value` every kernel here is built
//! from, the dequantizers, and the fused GEMV.
//!
//! The weight is the RAW row-major ggml layout read through the u32 view of slot a.
//! The dequantizers run one thread per output pair (16-bit outputs) or per element
//! (f32), write slot b, and take u0 = element count; output width is a comptime
//! choice so the three variants of each format stay one body. Blocks never straddle
//! a row (every consumer keeps cols a multiple of the block), and a pair never
//! straddles a block (block sizes are even), so a pair's two elements share one
//! block decode.
//!
//! `value` takes a byte `base` so the GEMV can point it at one row; the
//! dequantizers pass 0 and walk the tensor as one stream.
//!
//! `tbl` is the slot holding whatever fixed table the format reads: the 256-entry
//! f32 LUT for fp8, `iq_grid.blob` for the four codebook formats, unused by the
//! rest. It differs between the dequantizers (slot c) and the GEMV (slot d, since
//! c carries the output there), which is why it is a parameter and not a constant.

const k = @import("../dual.zig");
const grid = @import("../iq_grid.zig");
const Env = k.Env;
const Slot = k.Slot;

pub const Out = enum { f16, bf16, f32 };

/// The value of element `e` of the weight, per format.
const Fmt = enum {
    fp8,
    q8_0,
    q4_0,
    q1_0,
    q2_0_g64,
    q2_0_g128,
    iq4_nl,
    iq4_xs,
    q2_k,
    q3_k,
    q4_k,
    q5_k,
    q6_k,
    iq2_xxs,
    iq2_xs,
    iq3_xxs,
    iq3_s,
};

/// Block bytes of a block-quant format, matching `DType.blockBytes`. `fp8` is not
/// block-quantized and never asks.
inline fn blockBytes(comptime fmt: Fmt) u32 {
    return switch (fmt) {
        .fp8 => unreachable,
        .q8_0 => 34,
        .q4_0, .iq4_nl => 18,
        .q1_0, .q2_0_g64 => 18,
        .q2_0_g128 => 34,
        .iq4_xs => 136,
        .q2_k => 84,
        .q3_k, .iq3_s => 110,
        .q4_k => 144,
        .q5_k => 176,
        .q6_k => 210,
        .iq2_xxs => 66,
        .iq2_xs => 74,
        .iq3_xxs => 98,
    };
}

inline fn blockElems(comptime fmt: Fmt) u32 {
    return switch (fmt) {
        .fp8 => unreachable,
        .q8_0, .q4_0, .iq4_nl => 32,
        .q1_0, .q2_0_g128 => 128,
        .q2_0_g64 => 64,
        else => 256,
    };
}

const kvalues_iq4nl = [16]i32{ -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };

inline fn iq4Code(nibble: u32) i32 {
    // A switch, since a lookup into a comptime array by a runtime index lands in
    // private memory on both targets.
    return switch (nibble) {
        inline 0...15 => |n| kvalues_iq4nl[n],
        else => unreachable,
    };
}

inline fn iq4Value(nibble: u32) f32 {
    return @floatFromInt(iq4Code(nibble));
}

/// The codebook value of each of four nibbles, one per byte of `nibs`, as packed
/// signed bytes, READ FROM THE TABLE rather than switched on.
///
/// Sixteen bytes is one cache line every lane of a subgroup hits, where the
/// switch `iq4Code` compiles to is eight branchy lookups per group: measured,
/// that was 2x on iq4_nl and iq4_xs, the difference between the dual GEMV losing
/// to the int8 repack and beating it.
inline fn iq4Codes(e: Env, comptime tbl: Slot, nibs: u32) u32 {
    var out: u32 = 0;
    inline for (0..4) |b| {
        const v = e.byte(tbl, grid.kvalues_iq4nl_off + ((nibs >> @intCast(b * 8)) & 15));
        out |= v << @intCast(b * 8);
    }
    return out;
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

/// q3_k's 6-bit scale of sub-block `is` (0..15) from the 12 scale bytes at `sb`:
/// four low bits from one of the first eight, two high bits from the last four.
/// ggml shuffles all sixteen into a `uint32_t aux[4]` at once; this is the same
/// packing read one at a time, `is` selecting aux word `is >> 2`, byte `is & 3`.
inline fn scaleK3(e: Env, sb: u32, is: u32) u32 {
    const w = is >> 2;
    const bi = is & 3;
    const lo_byte = e.byte(.a, sb + (w & 1) * 4 + bi);
    const lo = if (w < 2) lo_byte & 15 else lo_byte >> 4;
    return lo | (((e.byte(.a, sb + 8 + bi) >> @intCast(2 * w)) & 3) << 4);
}

/// 32-bit word at a 2-aligned byte offset. The codebook formats pack their scales
/// and sign indices into u32s inside blocks whose stride is merely even, so these
/// reads are not 4-aligned and `Env` has no whole-word accessor for them.
inline fn u32At2(e: Env, off: u32) u32 {
    return e.u16At(.a, off) | (e.u16At(.a, off + 2) << 16);
}

/// Value `sub` (0..7) of codebook entry `gi`, negated per bit `sub` of the sign
/// byte `signs`. `off` is the table's base in `tbl` and `width` its entry size:
/// the iq2 grids hold eight values per entry, the iq3 grids four.
inline fn gridValue(e: Env, comptime tbl: Slot, comptime off: u32, comptime width: u32, gi: u32, sub: u32, signs: u32) f32 {
    const g: f32 = @floatFromInt(e.byte(tbl, off + gi * width + (sub & (width - 1))));
    return if ((signs >> @intCast(sub)) & 1 != 0) -g else g;
}

inline fn ksigns(e: Env, comptime tbl: Slot, idx: u32) u32 {
    return e.byte(tbl, grid.ksigns_iq2xs_off + idx);
}

/// Element `idx` of the weight whose bytes start at `base` in slot a.
inline fn value(e: Env, comptime fmt: Fmt, comptime tbl: Slot, base: u32, idx: u32) f32 {
    switch (fmt) {
        .fp8 => return e.ld(tbl, e.byte(.a, base + idx)) * e.f(0),
        .q8_0 => {
            const blk = base + (idx >> 5) * 34;
            const q: i32 = @as(i8, @bitCast(@as(u8, @truncate(e.byte(.a, blk + 2 + (idx & 31))))));
            return @as(f32, @floatFromInt(q)) * e.f16At(.a, blk);
        },
        .q4_0, .iq4_nl => {
            const blk = base + (idx >> 5) * 18;
            const j = idx & 31;
            const qs = e.byte(.a, blk + 2 + (j & 15));
            const nib = if (j < 16) qs & 15 else qs >> 4;
            const v: f32 = if (fmt == .q4_0) @floatFromInt(@as(i32, @intCast(nib)) - 8) else iq4Value(nib);
            return v * e.f16At(.a, blk);
        },
        .q1_0 => {
            const blk = base + (idx >> 7) * 18;
            const j = idx & 127;
            const bit = (e.byte(.a, blk + 2 + (j >> 3)) >> @intCast(j & 7)) & 1;
            const d = e.f16At(.a, blk);
            return if (bit != 0) d else -d;
        },
        .q2_0_g64, .q2_0_g128 => {
            const qk: u32 = if (fmt == .q2_0_g64) 64 else 128;
            const blk = base + (idx / qk) * (2 + qk / 4);
            const j = idx % qk;
            const code = (e.byte(.a, blk + 2 + (j >> 2)) >> @intCast(2 * (j & 3))) & 3;
            return @as(f32, @floatFromInt(@as(i32, @intCast(code)) - 1)) * e.f16At(.a, blk);
        },
        .iq4_xs => {
            const blk = base + (idx >> 8) * 136;
            const j = idx & 255;
            const ib = j >> 5;
            const ls_l = (e.byte(.a, blk + 4 + (ib >> 1)) >> @intCast((ib & 1) * 4)) & 15;
            const ls_h = (e.u16At(.a, blk + 2) >> @intCast(2 * ib)) & 3;
            const dl = e.f16At(.a, blk) * @as(f32, @floatFromInt(@as(i32, @intCast(ls_l | (ls_h << 4))) - 32));
            const qs = e.byte(.a, blk + 8 + (ib << 4) + (j & 15));
            const nib = if (j & 16 == 0) qs & 15 else qs >> 4;
            return iq4Value(nib) * dl;
        },
        // 16 B of 4-bit scale/min pairs, 64 B of 2-bit quants, f16 d, f16 dmin.
        // Sixteen sub-blocks of 16, indexed exactly as q3_k's.
        .q2_k => {
            const blk = base + (idx >> 8) * 84;
            const j = idx & 255;
            const q = (e.byte(.a, blk + 16 + (j >> 7) * 32 + (j & 31)) >> @intCast(((j >> 5) & 3) * 2)) & 3;
            const sc = e.byte(.a, blk + (j >> 7) * 8 + ((j >> 5) & 3) * 2 + ((j & 31) >> 4));
            const dl = e.f16At(.a, blk + 80) * @as(f32, @floatFromInt(sc & 15));
            const ml = e.f16At(.a, blk + 82) * @as(f32, @floatFromInt(sc >> 4));
            return dl * @as(f32, @floatFromInt(q)) - ml;
        },
        // 32 B hmask, 64 B of 2-bit quants, 12 B scales, f16 d. Sixteen sub-blocks
        // of 16, and the high bit is INVERTED: a clear hmask bit subtracts 4.
        .q3_k => {
            const blk = base + (idx >> 8) * 110;
            const j = idx & 255;
            const half = j >> 7; // which 32-byte half of qs
            const sub = (j >> 5) & 3; // shift within the byte
            const l = j & 31;
            const q = (e.byte(.a, blk + 32 + half * 32 + l) >> @intCast(sub * 2)) & 3;
            // One hmask bit per element, indexed by l with the bit selecting the
            // (half, sub) pair, so it runs 0..7 across the super-block.
            const h = (e.byte(.a, blk + l) >> @intCast(half * 4 + sub)) & 1;
            const sc = scaleK3(e, blk + 96, half * 8 + sub * 2 + (l >> 4));
            const dl = e.f16At(.a, blk + 108) * @as(f32, @floatFromInt(@as(i32, @intCast(sc)) - 32));
            return dl * @as(f32, @floatFromInt(@as(i32, @intCast(q)) - @as(i32, if (h != 0) 0 else 4)));
        },
        // f16 d then eight u32 PAIRS, one pair per 32 elements: the first holds four
        // grid indices, the second four 7-bit sign indices and a 4-bit scale on top.
        .iq2_xxs => {
            const blk = base + (idx >> 8) * 66;
            const j = idx & 255;
            const qb = blk + 2 + (j >> 5) * 8;
            const aux = u32At2(e, qb + 4);
            const db = e.f16At(.a, blk) * (0.5 + @as(f32, @floatFromInt(aux >> 28))) * 0.25;
            const l = (j >> 3) & 3;
            const signs = ksigns(e, tbl, (aux >> @intCast(7 * l)) & 127);
            const gi = e.byte(.a, qb + l);
            return db * gridValue(e, tbl, grid.iq2xxs_grid_off, 8, gi, j & 7, signs);
        },
        // Each u16 carries a 9-bit grid index and a 7-bit sign index; the scales are
        // explicit, one NIBBLE per 16 elements, so a pair of groups shares one.
        .iq2_xs => {
            const blk = base + (idx >> 8) * 74;
            const j = idx & 255;
            const l = (j >> 3) & 3;
            const q = e.u16At(.a, blk + 2 + ((j >> 5) * 4 + l) * 2);
            const nib = (e.byte(.a, blk + 66 + (j >> 5)) >> @intCast((l >> 1) * 4)) & 15;
            const db = e.f16At(.a, blk) * (0.5 + @as(f32, @floatFromInt(nib))) * 0.25;
            return db * gridValue(e, tbl, grid.iq2xs_grid_off, 8, q & 511, j & 7, ksigns(e, tbl, q >> 9));
        },
        // 64 index bytes then 32 B of iq2_xxs-shaped scale/sign words. A grid entry
        // is four values, so a group of 8 takes two indices and one sign byte.
        .iq3_xxs => {
            const blk = base + (idx >> 8) * 98;
            const j = idx & 255;
            const aux = u32At2(e, blk + 66 + (j >> 5) * 4);
            const db = e.f16At(.a, blk) * (0.5 + @as(f32, @floatFromInt(aux >> 28))) * 0.5;
            const l = (j >> 3) & 3;
            const signs = ksigns(e, tbl, (aux >> @intCast(7 * l)) & 127);
            const gi = e.byte(.a, blk + 2 + (j >> 5) * 8 + 2 * l + ((j & 7) >> 2));
            return db * gridValue(e, tbl, grid.iq3xxs_grid_off, 4, gi, j & 7, signs);
        },
        // iq3_xxs's grid widened to 512 by a 9th index bit held in qh, with the
        // signs stored plainly and the scale an odd integer, not a half-step.
        .iq3_s => {
            const blk = base + (idx >> 8) * 110;
            const j = idx & 255;
            const h = j >> 5; // 32-element half-block, 0..7
            const l = (j >> 3) & 3;
            const qi = 2 * l + ((j & 7) >> 2);
            const nib = (e.byte(.a, blk + 106 + (h >> 1)) >> @intCast((h & 1) * 4)) & 15;
            const db = e.f16At(.a, blk) * @as(f32, @floatFromInt(1 + 2 * nib));
            const hi = (e.byte(.a, blk + 66 + h) >> @intCast(qi)) & 1;
            const gi = e.byte(.a, blk + 2 + h * 8 + qi) | (hi << 8);
            const signs = e.byte(.a, blk + 74 + h * 4 + l);
            return db * gridValue(e, tbl, grid.iq3s_grid_off, 4, gi, j & 7, signs);
        },
        .q4_k, .q5_k => {
            const bb: u32 = if (fmt == .q4_k) 144 else 176;
            const blk = base + (idx >> 8) * bb;
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
            const blk = base + (idx >> 8) * 210;
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
        e.st(.b, i, value(e, fmt, .c, 0, i));
        return;
    }
    const w = e.gid();
    const e0 = w * 2;
    if (e0 >= e.u(0)) return;
    const lo = value(e, fmt, .c, 0, e0);
    const hi = value(e, fmt, .c, 0, e0 + 1);
    e.stW(.b, w, if (out == .f16) k.packH16(lo, hi) else @as(u32, k.bf16Bits(lo)) | (@as(u32, k.bf16Bits(hi)) << 16));
}
// ---- fused GEMV -------------------------------------------------------------
//
// `value` re-reads a block's scale, its high bits and the codebook for EVERY
// element, which costs a dequant nothing (it is one pass over the weight, and the
// re-reads hit L1) but costs a GEMV everything. So the GEMV decodes a GROUP at a
// time with those reads hoisted and takes the quants and codebook entries as
// WORDS.
//
// ⚠️ **The group is EIGHT elements, and wider is worse.** These kernels are bound
// by memory TRANSACTIONS, not by bandwidth and not by load instructions: at eight
// a whole 32-lane subgroup sits inside one 256-element super-block and a load
// touches one cache line, where at 32 it straddles four and each load costs four
// transactions. Measured: 32-element groups issue 1.4x FEWER loads and run 1.6x
// slower. Eight is also the natural unit of all four codebook formats, being one
// sign byte's reach.
//
// `group` is the ONE decoder, shared by the f32 and the int8 dot below, so a
// format is written once whichever activation a backend can stage. Its arithmetic
// is `value`'s, and the device tests score both against ggml, which is what keeps
// the two from drifting.

/// Elements one lane decodes in one go. See the transaction note above.
const group_elems: u32 = 8;

/// One group decoded: eight signed int8 codes packed four per word, plus the
/// affine scale they take. `value(i) == d * code(i) - m`. Only q2_k has a nonzero
/// `m`; for every other format the code carries the sign.
const Group = struct {
    w: [2]u32,
    d: f32,
    m: f32,
};

/// Whether a format's value has a subtracted term, so the dots have to carry the
/// activation's own sum. Skipped entirely for the rest.
inline fn hasMin(comptime fmt: Fmt) bool {
    return fmt == .q2_k or fmt == .q4_k or fmt == .q5_k;
}

/// Word `half` of codebook entry `gi`. Every table is 4-aligned and its entries
/// are 4 or 8 bytes, so four values arrive in one load.
inline fn gridWord(e: Env, comptime tbl: Slot, comptime off: u32, comptime words: u32, gi: u32, half: u32) u32 {
    comptime {
        if (off % 4 != 0) @compileError("codebook table must be 4-aligned");
    }
    return e.ldW(tbl, (off >> 2) + gi * words + half);
}

/// Four codebook values with the sign bits `bits` applied, as packed signed bytes.
inline fn applySigns(g: u32, bits: u32) u32 {
    const mask = k.byteMask(bits);
    return k.subBytes(g ^ mask, mask);
}

/// A group's eight signed values: BOTH gathers issued before either one's sign
/// math, so the two scattered table reads overlap. They are the kernel's only
/// divergent loads and its longest latency, so letting the arithmetic sit between
/// them instead costs more than the arithmetic does.
inline fn signedPair(e: Env, comptime tbl: Slot, comptime off: u32, comptime words: u32, g0: u32, g1: u32, h0: u32, h1: u32, signs: u32) [2]u32 {
    const a0 = gridWord(e, tbl, off, words, g0, h0);
    const a1 = gridWord(e, tbl, off, words, g1, h1);
    return .{ applySigns(a0, signs & 15), applySigns(a1, signs >> 4) };
}

/// Group `g` (elements `g * 8 ..`) of the weight whose bytes start at `base`.
inline fn group(e: Env, comptime fmt: Fmt, comptime tbl: Slot, base: u32, g: u32) Group {
    var r: Group = .{ .w = .{ 0, 0 }, .d = 0, .m = 0 };
    // Block index and the group's first element within it. Every block size here
    // is a power of two multiple of the group, so both are shifts.
    const per_block = blockElems(fmt) / group_elems;
    const sb = g / per_block;
    const j = (g % per_block) * group_elems;
    switch (fmt) {
        // Sixteen sub-blocks of 16; a group sits inside one. The 84-byte stride is
        // 4-aligned, so its eight quant bytes are two whole words.
        .q2_k => {
            const blk = base + sb * 84;
            const half = j >> 7;
            const shift: u5 = @intCast(((j >> 5) & 3) * 2);
            const sc = e.byte(.a, blk + half * 8 + ((j >> 5) & 3) * 2 + ((j & 31) >> 4));
            r.d = e.f16At(.a, blk + 80) * @as(f32, @floatFromInt(sc & 15));
            r.m = e.f16At(.a, blk + 82) * @as(f32, @floatFromInt(sc >> 4));
            const qw = (blk + 16 + half * 32 + (j & 31)) >> 2;
            inline for (0..2) |i| r.w[i] = (e.ldW(.a, qw + @as(u32, @intCast(i))) >> shift) & 0x03030303;
        },
        // Eight sub-blocks of 32, each with a 6-bit scale and min packed across
        // twelve bytes. The 144-byte stride is 16-aligned, so a group's eight
        // nibble bytes are two whole words.
        //
        // This format already has a hand-written kernel on both backends; the dual
        // arm exists so the two approaches can be compared on ONE weight
        // (`TP_DUAL_DECODE`, and `vk-gemv-bench`). Nothing routes here by default.
        .q4_k => {
            const blk = base + sb * 144;
            const is = j >> 5;
            const sm = scaleMinK4(e, blk + 4, is);
            r.d = e.f16At(.a, blk) * @as(f32, @floatFromInt(sm[0]));
            r.m = e.f16At(.a, blk + 2) * @as(f32, @floatFromInt(sm[1]));
            const shift: u5 = @intCast((is & 1) * 4);
            const qw = (blk + 16 + ((j >> 6) << 5) + (j & 31)) >> 2;
            inline for (0..2) |i| r.w[i] = (e.ldW(.a, qw + @as(u32, @intCast(i))) >> shift) & 0x0F0F0F0F;
        },
        // Same mapping, but the code is 2 bits MINUS 4 unless the high-bit mask
        // says otherwise, which folds into a signed code and needs no `m`. The
        // 110-byte stride is only 2-aligned, so quants and mask come in pairs.
        .q3_k => {
            const blk = base + sb * 110;
            const half = j >> 7;
            const sub = (j >> 5) & 3;
            const shift: u5 = @intCast(sub * 2);
            const hshift: u5 = @intCast(half * 4 + sub);
            const lo = j & 31;
            const sc = scaleK3(e, blk + 96, half * 8 + sub * 2 + (lo >> 4));
            r.d = e.f16At(.a, blk + 108) * @as(f32, @floatFromInt(@as(i32, @intCast(sc)) - 32));
            inline for (0..2) |i_| {
                const i: u32 = @intCast(i_);
                const q = (u32At2(e, blk + 32 + half * 32 + lo + i * 4) >> shift) & 0x03030303;
                const h = (u32At2(e, blk + lo + i * 4) >> hshift) & 0x01010101;
                r.w[i_] = k.subBytes(q + (h << 2), 0x04040404);
            }
        },
        // One grid entry of eight values, one sign index and one scale out of the
        // u32 that four groups share.
        .iq2_xxs => {
            const blk = base + sb * 66;
            const qb = blk + 2 + (j >> 5) * 8;
            const aux = u32At2(e, qb + 4);
            const l = (j >> 3) & 3;
            r.d = e.f16At(.a, blk) * (0.5 + @as(f32, @floatFromInt(aux >> 28))) * 0.25;
            const gi = e.byte(.a, qb + l);
            const signs = ksigns(e, tbl, (aux >> @intCast(7 * l)) & 127);
            r.w = signedPair(e, tbl, grid.iq2xxs_grid_off, 2, gi, gi, 0, 1, signs);
        },
        // The scale is explicit here, one nibble per 16 elements, and the u16
        // carries a 9-bit grid index and a 7-bit sign index.
        .iq2_xs => {
            const blk = base + sb * 74;
            const l = (j >> 3) & 3;
            const q = e.u16At(.a, blk + 2 + ((j >> 5) * 4 + l) * 2);
            const nib = (e.byte(.a, blk + 66 + (j >> 5)) >> @intCast((l >> 1) * 4)) & 15;
            r.d = e.f16At(.a, blk) * (0.5 + @as(f32, @floatFromInt(nib))) * 0.25;
            const signs = ksigns(e, tbl, q >> 9);
            r.w = signedPair(e, tbl, grid.iq2xs_grid_off, 2, q & 511, q & 511, 0, 1, signs);
        },
        // An iq3 grid entry is FOUR values, so a group takes two indices, which
        // sit side by side and arrive as one u16.
        .iq3_xxs => {
            const blk = base + sb * 98;
            const l = (j >> 3) & 3;
            const aux = u32At2(e, blk + 66 + (j >> 5) * 4);
            r.d = e.f16At(.a, blk) * (0.5 + @as(f32, @floatFromInt(aux >> 28))) * 0.5;
            const gp = e.u16At(.a, blk + 2 + (j >> 5) * 8 + 2 * l);
            const signs = ksigns(e, tbl, (aux >> @intCast(7 * l)) & 127);
            r.w = signedPair(e, tbl, grid.iq3xxs_grid_off, 1, gp & 255, gp >> 8, 0, 0, signs);
        },
        // iq3_xxs's grid widened to 512 by a 9th bit out of qh, with the signs
        // stored plainly and the scale an odd integer rather than a half-step.
        .iq3_s => {
            const blk = base + sb * 110;
            const h = j >> 5;
            const l = (j >> 3) & 3;
            const nib = (e.byte(.a, blk + 106 + (h >> 1)) >> @intCast((h & 1) * 4)) & 15;
            r.d = e.f16At(.a, blk) * @as(f32, @floatFromInt(1 + 2 * nib));
            const qh = e.byte(.a, blk + 66 + h);
            const gp = e.u16At(.a, blk + 2 + h * 8 + 2 * l);
            const signs = e.byte(.a, blk + 74 + h * 4 + l);
            r.w = signedPair(
                e,
                tbl,
                grid.iq3s_grid_off,
                1,
                (gp & 255) | (((qh >> @intCast(2 * l)) & 1) << 8),
                (gp >> 8) | (((qh >> @intCast(2 * l + 1)) & 1) << 8),
                0,
                0,
                signs,
            );
        },
        // --- the formats that ALSO have a hand-written kernel ------------------
        //
        // These exist so the dual body can be compared against, and on Vulkan
        // replace, the per-backend kernels. `vk-gemv-bench` and `TP_DUAL_DECODE`
        // are how; BACKEND.md carries the verdicts.

        // 32 elements, already signed bytes: the only decode is the block scale.
        // The 34-byte stride is 2-aligned, so the codes come as u16 pairs.
        .q8_0 => {
            const blk = base + sb * 34;
            r.d = e.f16At(.a, blk);
            inline for (0..2) |i| r.w[i] = u32At2(e, blk + 2 + j + @as(u32, @intCast(i)) * 4);
        },
        // 32 elements as 16 nibble bytes, the LOW nibbles first: elements 0..15
        // are the low half of bytes 0..15 and 16..31 the high half, which is not
        // the layout q4_k uses. Code is the nibble less 8.
        .q4_0 => {
            const blk = base + sb * 18;
            r.d = e.f16At(.a, blk);
            const shift: u5 = if (j < 16) 0 else 4;
            inline for (0..2) |i| {
                const w = u32At2(e, blk + 2 + (j & 15) + @as(u32, @intCast(i)) * 4);
                r.w[i] = k.subBytes((w >> shift) & 0x0F0F0F0F, 0x08080808);
            }
        },
        // q4_0's layout over a non-linear codebook, so the nibble is an INDEX and
        // reading it as a signed value is finite, plausible and wrong.
        .iq4_nl => {
            const blk = base + sb * 18;
            r.d = e.f16At(.a, blk);
            const shift: u5 = if (j < 16) 0 else 4;
            inline for (0..2) |i| {
                const w = u32At2(e, blk + 2 + (j & 15) + @as(u32, @intCast(i)) * 4);
                r.w[i] = iq4Codes(e, tbl, (w >> shift) & 0x0F0F0F0F);
            }
        },
        // The same codebook over a k-quant super-block, with a 6-bit sub-block
        // scale split across scales_h and scales_l and biased by -32. Its
        // 136-byte stride IS 4-aligned, so the nibbles are whole words.
        .iq4_xs => {
            const blk = base + sb * 136;
            const ib = j >> 5;
            const ls_l = (e.byte(.a, blk + 4 + (ib >> 1)) >> @intCast((ib & 1) * 4)) & 15;
            const ls_h = (e.u16At(.a, blk + 2) >> @intCast(2 * ib)) & 3;
            r.d = e.f16At(.a, blk) * @as(f32, @floatFromInt(@as(i32, @intCast(ls_l | (ls_h << 4))) - 32));
            const shift: u5 = if (j & 16 == 0) 0 else 4;
            const qw = (blk + 8 + (ib << 4) + (j & 15)) >> 2;
            inline for (0..2) |i| r.w[i] = iq4Codes(e, tbl, (e.ldW(.a, qw + @as(u32, @intCast(i))) >> shift) & 0x0F0F0F0F);
        },
        // q4_k plus a 5th bit per element out of qh, so the code is 0..31.
        .q5_k => {
            const blk = base + sb * 176;
            const is = j >> 5;
            const sm = scaleMinK4(e, blk + 4, is);
            r.d = e.f16At(.a, blk) * @as(f32, @floatFromInt(sm[0]));
            r.m = e.f16At(.a, blk + 2) * @as(f32, @floatFromInt(sm[1]));
            const shift: u5 = @intCast((is & 1) * 4);
            const hbit: u5 = @intCast(is & 7);
            const qw = (blk + 48 + ((j >> 6) << 5) + (j & 31)) >> 2;
            const hw = (blk + 16 + (j & 31)) >> 2;
            inline for (0..2) |i| {
                const lo = (e.ldW(.a, qw + @as(u32, @intCast(i))) >> shift) & 0x0F0F0F0F;
                const hi = (e.ldW(.a, hw + @as(u32, @intCast(i))) >> hbit) & 0x01010101;
                r.w[i] = lo | (hi << 4);
            }
        },
        // Sixteen sub-blocks of 16 with a SIGNED 8-bit scale each; the code is a
        // 4-bit low half plus a 2-bit high half, biased by -32. The 210-byte
        // stride is 2-aligned, so both halves come as u16 pairs.
        .q6_k => {
            const blk = base + sb * 210;
            const sc = k.sext8(e.byte(.a, blk + 192 + (j >> 4)));
            r.d = e.f16At(.a, blk + 208) * @as(f32, @floatFromInt(sc));
            const half = j >> 7;
            const lshift: u5 = @intCast(((j >> 6) & 1) * 4);
            const hshift: u5 = @intCast(((j >> 5) & 3) * 2);
            const lbase = blk + half * 64 + ((j >> 5) & 1) * 32 + (j & 31);
            const hbase = blk + 128 + half * 32 + (j & 31);
            inline for (0..2) |i| {
                const o: u32 = @as(u32, @intCast(i)) * 4;
                const lo = (u32At2(e, lbase + o) >> lshift) & 0x0F0F0F0F;
                const hi = (u32At2(e, hbase + o) >> hshift) & 0x03030303;
                r.w[i] = k.subBytes(lo | (hi << 4), 0x20202020);
            }
        },
        // One bit per weight: the code is the SIGN, and a group's eight bits are
        // one byte. Two loads for eight elements, the cheapest decode here.
        .q1_0 => {
            const blk = base + sb * 18;
            r.d = e.f16At(.a, blk);
            const bits = e.byte(.a, blk + 2 + (j >> 3));
            inline for (0..2) |i| {
                const m = k.byteMask((bits >> @intCast(@as(u32, @intCast(i)) * 4)) & 15);
                r.w[i] = k.subBytes((m & 0x01010101) << 1, 0x01010101);
            }
        },
        // 2-bit codes, four per byte, LSB first; the set is {-1, 0, +1, +2} * d,
        // NOT centred on zero. A group is exactly two bytes, and the shifts lay
        // one byte's four codes into four byte lanes.
        .q2_0_g64, .q2_0_g128 => {
            const qk: u32 = if (fmt == .q2_0_g64) 64 else 128;
            const blk = base + sb * (2 + qk / 4);
            r.d = e.f16At(.a, blk);
            inline for (0..2) |i| {
                const b = e.byte(.a, blk + 2 + (j >> 2) + @as(u32, @intCast(i)));
                const codes = (b | (b << 6) | (b << 12) | (b << 18)) & 0x03030303;
                r.w[i] = k.subBytes(codes, 0x01010101);
            }
        },
        else => @compileError("no grouped decode for " ++ @tagName(fmt)),
    }
    return r;
}

/// `dot(group, x[x0 .. x0+8])` against the f32 activation in slot b.
inline fn dotF32(e: Env, comptime fmt: Fmt, gr: Group, x0: u32) f32 {
    var acc: f32 = 0;
    var sx: f32 = 0;
    inline for (0..2) |wi| {
        const xs = e.ld4(.b, x0 + @as(u32, @intCast(wi)) * 4);
        inline for (0..4) |bi| {
            const cv: f32 = @floatFromInt(k.sext8(gr.w[wi] >> @intCast(bi * 8)));
            acc += cv * xs[bi];
            if (comptime hasMin(fmt)) sx += xs[bi];
        }
    }
    return if (comptime hasMin(fmt)) gr.d * acc - gr.m * sx else gr.d * acc;
}

/// The same dot against the q8_1 activation in slot b, four products per
/// instruction. The layout is the one `quantize_q8_1` writes: `d[nblk]` f32, then
/// one int8 per element, then a per-block sum this does not read (it needs the sum
/// over a GROUP, which the codes give directly).
///
/// Four groups share a q8_1 block, hence its scale.
inline fn dotQ8(e: Env, comptime fmt: Fmt, gr: Group, g: u32, nblk: u32) f32 {
    const x0 = nblk + g * 2;
    const xa = e.ldW(.b, x0);
    const xb = e.ldW(.b, x0 + 1);
    var acc = k.dp4a(gr.w[1], xb, k.dp4a(gr.w[0], xa, 0));
    var r = gr.d * @as(f32, @floatFromInt(acc));
    if (comptime hasMin(fmt)) {
        acc = k.dp4a(0x01010101, xb, k.dp4a(0x01010101, xa, 0));
        r -= gr.m * @as(f32, @floatFromInt(acc));
    }
    return r * e.ld(.b, g >> 2);
}

/// Fused block-quant GEMV: `y[u2 + row] = f0 * dot(W[row], x)`, one subgroup per
/// row, lanes strided over the row's groups. Slot a is the packed weight, b the
/// activation, c the f32 output, d the codebook tables. u0 = rows, u1 = cols,
/// u2 = the output's row offset (a stepper packs several linears into one buffer).
///
/// `q8` picks the activation: the f32 one straight out of the stepper, or the
/// q8_1 one a prep pass staged. Every weight byte is read once from DRAM either
/// way; the shared reads a group repeats are served by L1.
///
/// `cols` must be a whole number of super-blocks, and `rows * rowBytes` must fit
/// u32.
inline fn gemvQ(e: Env, comptime fmt: Fmt, comptime q8: bool) void {
    const rows = e.u(0);
    const cols = e.u(1);
    const groups = cols / group_elems;
    const nblk = cols >> 5;
    const row_bytes = (cols / blockElems(fmt)) * blockBytes(fmt);
    const lane = e.sgLane();
    const lanes = e.sgSize();
    var row = k.sgGlobal(e);
    const step = k.sgTotal(e);
    while (row < rows) : (row += step) {
        const base = row * row_bytes;
        var acc: f32 = 0;
        var g = lane;
        while (g < groups) : (g += lanes) {
            const gr = group(e, fmt, .d, base, g);
            acc += if (q8) dotQ8(e, fmt, gr, g, nblk) else dotF32(e, fmt, gr, g * group_elems);
        }
        const total = k.sgSum(acc);
        if (lane == 0) e.st(.c, e.u(2) + row, total * e.f(0));
    }
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
pub inline fn q2_kF16(e: Env) void {
    dequant(e, .q2_k, .f16);
}
pub inline fn q2_kBf16(e: Env) void {
    dequant(e, .q2_k, .bf16);
}
pub inline fn q2_kF32(e: Env) void {
    dequant(e, .q2_k, .f32);
}
pub inline fn q3_kF16(e: Env) void {
    dequant(e, .q3_k, .f16);
}
pub inline fn q3_kBf16(e: Env) void {
    dequant(e, .q3_k, .bf16);
}
pub inline fn q3_kF32(e: Env) void {
    dequant(e, .q3_k, .f32);
}
pub inline fn iq2XxsF16(e: Env) void {
    dequant(e, .iq2_xxs, .f16);
}
pub inline fn iq2XxsBf16(e: Env) void {
    dequant(e, .iq2_xxs, .bf16);
}
pub inline fn iq2XxsF32(e: Env) void {
    dequant(e, .iq2_xxs, .f32);
}
pub inline fn iq2XsF16(e: Env) void {
    dequant(e, .iq2_xs, .f16);
}
pub inline fn iq2XsBf16(e: Env) void {
    dequant(e, .iq2_xs, .bf16);
}
pub inline fn iq2XsF32(e: Env) void {
    dequant(e, .iq2_xs, .f32);
}
pub inline fn iq3XxsF16(e: Env) void {
    dequant(e, .iq3_xxs, .f16);
}
pub inline fn iq3XxsBf16(e: Env) void {
    dequant(e, .iq3_xxs, .bf16);
}
pub inline fn iq3XxsF32(e: Env) void {
    dequant(e, .iq3_xxs, .f32);
}
pub inline fn iq3SF16(e: Env) void {
    dequant(e, .iq3_s, .f16);
}
pub inline fn iq3SBf16(e: Env) void {
    dequant(e, .iq3_s, .bf16);
}
pub inline fn iq3SF32(e: Env) void {
    dequant(e, .iq3_s, .f32);
}

// The fused GEMVs, named gemv_<fmt> over the f32 activation and gemv_<fmt>_q8
// over the q8_1 one. Only the formats with no hand-written per-backend kernel are
// here; the rest reach decode through those.
pub inline fn gemvQ2_k(e: Env) void {
    gemvQ(e, .q2_k, false);
}
pub inline fn gemvQ2_kQ8(e: Env) void {
    gemvQ(e, .q2_k, true);
}
pub inline fn gemvQ8_0(e: Env) void {
    gemvQ(e, .q8_0, false);
}
pub inline fn gemvQ8_0Q8(e: Env) void {
    gemvQ(e, .q8_0, true);
}
pub inline fn gemvQ4_0(e: Env) void {
    gemvQ(e, .q4_0, false);
}
pub inline fn gemvQ4_0Q8(e: Env) void {
    gemvQ(e, .q4_0, true);
}
pub inline fn gemvIq4Nl(e: Env) void {
    gemvQ(e, .iq4_nl, false);
}
pub inline fn gemvIq4NlQ8(e: Env) void {
    gemvQ(e, .iq4_nl, true);
}
pub inline fn gemvIq4Xs(e: Env) void {
    gemvQ(e, .iq4_xs, false);
}
pub inline fn gemvIq4XsQ8(e: Env) void {
    gemvQ(e, .iq4_xs, true);
}
pub inline fn gemvQ5_k(e: Env) void {
    gemvQ(e, .q5_k, false);
}
pub inline fn gemvQ5_kQ8(e: Env) void {
    gemvQ(e, .q5_k, true);
}
pub inline fn gemvQ6_k(e: Env) void {
    gemvQ(e, .q6_k, false);
}
pub inline fn gemvQ6_kQ8(e: Env) void {
    gemvQ(e, .q6_k, true);
}
pub inline fn gemvQ1_0(e: Env) void {
    gemvQ(e, .q1_0, false);
}
pub inline fn gemvQ1_0Q8(e: Env) void {
    gemvQ(e, .q1_0, true);
}
pub inline fn gemvQ2_0G64(e: Env) void {
    gemvQ(e, .q2_0_g64, false);
}
pub inline fn gemvQ2_0G64Q8(e: Env) void {
    gemvQ(e, .q2_0_g64, true);
}
pub inline fn gemvQ2_0G128(e: Env) void {
    gemvQ(e, .q2_0_g128, false);
}
pub inline fn gemvQ2_0G128Q8(e: Env) void {
    gemvQ(e, .q2_0_g128, true);
}
pub inline fn gemvQ4_k(e: Env) void {
    gemvQ(e, .q4_k, false);
}
pub inline fn gemvQ4_kQ8(e: Env) void {
    gemvQ(e, .q4_k, true);
}
pub inline fn gemvQ3_k(e: Env) void {
    gemvQ(e, .q3_k, false);
}
pub inline fn gemvQ3_kQ8(e: Env) void {
    gemvQ(e, .q3_k, true);
}
pub inline fn gemvIq2Xxs(e: Env) void {
    gemvQ(e, .iq2_xxs, false);
}
pub inline fn gemvIq2XxsQ8(e: Env) void {
    gemvQ(e, .iq2_xxs, true);
}
pub inline fn gemvIq2Xs(e: Env) void {
    gemvQ(e, .iq2_xs, false);
}
pub inline fn gemvIq2XsQ8(e: Env) void {
    gemvQ(e, .iq2_xs, true);
}
pub inline fn gemvIq3Xxs(e: Env) void {
    gemvQ(e, .iq3_xxs, false);
}
pub inline fn gemvIq3XxsQ8(e: Env) void {
    gemvQ(e, .iq3_xxs, true);
}
pub inline fn gemvIq3S(e: Env) void {
    gemvQ(e, .iq3_s, false);
}
pub inline fn gemvIq3SQ8(e: Env) void {
    gemvQ(e, .iq3_s, true);
}
