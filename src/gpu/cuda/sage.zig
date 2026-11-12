//! ComfyUI's int8 attention on CUDA: the kernels that reproduce comfy-kitchen's
//! "sage" SDPA at head dim 128 with more than 1024 keys, on the same approximate
//! instructions (`div.approx`, `rcp.approx`, `ex2.approx`) so the bytes match its
//! kernels, not just the math. `ops/sage_attn.zig` is the readable definition and
//! names what the path computes; this file is its device form.
//!
//! Five launches: `sage_anchor` picks each head's subtracted key, `sage_quant_qk`
//! rotates and quantizes Q and K (one warp per scale group), `sage_vmax` and
//! `sage_quant_v` take V's per-channel range and write it transposed in
//! comfy-kitchen's layout, `sage_attn` attends. The operands are head-major and
//! padded (Q to 128 rows, K and V to 128 keys), so the attention kernel never
//! bounds-checks a load.
//!
//! The u8 probabilities of one lane are keys {2t, 2t+1, 2t+8, 2t+9} of each 16-key
//! block, which is the order the m16n8k32 A operand reads them in; V is stored with
//! each 16 keys permuted to match, so P.V needs no shuffle.

const std = @import("std");
const ptx = @import("ptx.zig");
const Dt = @import("../kernels/dual_table.zig").Dt;

pub const hd = 128;
pub const key_tile = 128;
/// Query rows per attention block.
pub const q_tile = 64;

fn hex(v: f32) u32 {
    return @bitCast(v);
}

/// Rows of the padded int8 Q and K buffers.
pub fn qRows(lq: usize) usize {
    return (lq + 127) / 128 * 128;
}

pub fn kRows(lk: usize) usize {
    return (lk + key_tile - 1) / key_tile * key_tile;
}

pub fn qScales(lq: usize) usize {
    return qRows(lq) / 128 * 32;
}

pub fn kScales(lk: usize) usize {
    return kRows(lk) / key_tile * 4;
}

/// Load one input element at `addr` into f32 `dst` (bf16 widens exactly).
fn loadIn(b: *ptx.Builder, in_dt: Dt, dst: []const u8, addr: []const u8, r_tmp: []const u8) !void {
    if (in_dt == .bf16) {
        try b.linef("ld.global.u16 {s}, [{s}];", .{ r_tmp, addr });
        try b.linef("shl.b32 {s}, {s}, 16;", .{ r_tmp, r_tmp });
        try b.linef("mov.b32 {s}, {s};", .{ dst, r_tmp });
    } else if (in_dt == .f16) {
        const h = try b.reg(.b16);
        try b.linef("ld.global.b16 {s}, [{s}];", .{ h, addr });
        try b.linef("cvt.f32.f16 {s}, {s};", .{ dst, h });
    } else {
        try b.linef("ld.global.f32 {s}, [{s}];", .{ dst, addr });
    }
}

/// `sage_anchor`, grid (heads), 128 threads: thread c owns channel c of the nine
/// sampled keys. Params: k, lk, row stride (elements), anchor out (s32 per head).
pub fn buildAnchor(alloc: std.mem.Allocator, in_dt: Dt) ![:0]u8 {
    var b = ptx.Builder.init(alloc);
    defer b.deinit();
    b.tuning = ".maxntid 128, 1, 1";
    const eb: usize = if (in_dt == .f32) 4 else 2;
    const ns = 9;

    const rd_k = try b.reg(.b64);
    const rd_a = try b.reg(.b64);
    const r_lk = try b.reg(.b32);
    const r_ks = try b.reg(.b32);
    const r_t = try b.reg(.b32);
    const r_h = try b.reg(.b32);
    const r_tmp = try b.reg(.b32);
    const r_row = try b.reg(.b32);
    const rd_tmp = try b.reg(.b64);
    const rd_base = try b.reg(.b64);
    try b.linef("ld.param.u64 {s}, [p_k];", .{rd_k});
    try b.linef("cvta.to.global.u64 {s}, {s};", .{ rd_k, rd_k });
    try b.linef("ld.param.u64 {s}, [p_anchor];", .{rd_a});
    try b.linef("cvta.to.global.u64 {s}, {s};", .{ rd_a, rd_a });
    try b.linef("ld.param.u32 {s}, [p_lk];", .{r_lk});
    try b.linef("ld.param.u32 {s}, [p_ks];", .{r_ks});
    try b.linef("mov.u32 {s}, %tid.x;", .{r_t});
    try b.linef("mov.u32 {s}, %ctaid.x;", .{r_h});
    // base = k + (h * 128 + c) * eb
    try b.linef("mad.lo.u32 {s}, {s}, {d}, {s};", .{ r_tmp, r_h, hd, r_t });
    try b.linef("mul.wide.u32 {s}, {s}, {d};", .{ rd_tmp, r_tmp, eb });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_base, rd_k, rd_tmp });

    const xs = try b.regs(.f32, ns);
    const r_lk1 = try b.reg(.b32);
    try b.linef("sub.u32 {s}, {s}, 1;", .{ r_lk1, r_lk });
    for (0..ns) |s| {
        // row_s = s * (lk - 1) / 8
        try b.linef("mul.lo.u32 {s}, {s}, {d};", .{ r_row, r_lk1, s });
        try b.linef("shr.u32 {s}, {s}, 3;", .{ r_row, r_row });
        try b.linef("mul.lo.u32 {s}, {s}, {s};", .{ r_row, r_row, r_ks });
        try b.linef("mul.wide.u32 {s}, {s}, {d};", .{ rd_tmp, r_row, eb });
        try b.linef("add.s64 {s}, {s}, {s};", .{ rd_tmp, rd_tmp, rd_base });
        try loadIn(&b, in_dt, xs[s], rd_tmp, r_tmp);
    }

    // Per-channel energy, range, sum and each candidate's squared distance.
    const f_e = try b.reg(.f32);
    const f_m = try b.reg(.f32);
    const f_sum = try b.reg(.f32);
    const f_x = try b.reg(.f32);
    const cand = try b.regs(.f32, ns);
    try b.linef("mov.f32 {s}, 0f00000000;", .{f_e});
    try b.linef("mov.f32 {s}, 0f00000000;", .{f_m});
    try b.linef("mov.f32 {s}, 0f00000000;", .{f_sum});
    for (xs) |x| {
        try b.linef("fma.rn.ftz.f32 {s}, {s}, {s}, {s};", .{ f_e, x, x, f_e });
        try b.linef("abs.ftz.f32 {s}, {s};", .{ f_x, x });
        try b.linef("max.ftz.f32 {s}, {s}, {s};", .{ f_m, f_m, f_x });
        try b.linef("add.ftz.f32 {s}, {s}, {s};", .{ f_sum, f_sum, x });
    }
    for (xs, cand) |x, c| {
        try b.linef("mul.ftz.f32 {s}, {s}, 0f41100000;", .{ f_x, x });
        try b.linef("sub.ftz.f32 {s}, {s}, {s};", .{ f_x, f_x, f_sum });
        try b.linef("mul.ftz.f32 {s}, {s}, {s};", .{ c, f_x, f_x });
    }

    // Block sums: a shuffle-down tree per warp, then thread 0 adds the four warp
    // sums in order. Slots: 0..8 candidates, 9 energy, 10 range (max), 11 best.
    const r_lane = try b.reg(.b32);
    const r_warp = try b.reg(.b32);
    const r_smem = try b.reg(.b32);
    const p0 = try b.reg(.pred);
    const p_lane0 = try b.reg(.pred);
    try b.linef("and.b32 {s}, {s}, 31;", .{ r_lane, r_t });
    try b.linef("shr.u32 {s}, {s}, 5;", .{ r_warp, r_t });
    try b.linef("setp.eq.u32 {s}, {s}, 0;", .{ p_lane0, r_lane });
    try b.linef("mov.u32 {s}, smem;", .{r_smem});
    const Red = struct {
        fn warp(bb: *ptx.Builder, f: []const u8, is_max: bool, tmp: []const u8, fx: []const u8) !void {
            for ([_]u32{ 16, 8, 4, 2, 1 }) |off| {
                try bb.linef("mov.b32 {s}, {s};", .{ tmp, f });
                if (is_max) {
                    try bb.linef("shfl.sync.bfly.b32 {s}, {s}, {d}, 0x1f, 0xffffffff;", .{ tmp, tmp, off });
                } else {
                    try bb.linef("shfl.sync.down.b32 {s}, {s}, {d}, 0x1f, 0xffffffff;", .{ tmp, tmp, off });
                }
                try bb.linef("mov.b32 {s}, {s};", .{ fx, tmp });
                if (is_max) {
                    try bb.linef("max.ftz.f32 {s}, {s}, {s};", .{ f, f, fx });
                } else {
                    try bb.linef("add.ftz.f32 {s}, {s}, {s};", .{ f, f, fx });
                }
            }
        }
    };
    // smem layout: [slot][warp] f32
    const Slots = 12;
    for (cand, 0..) |c, s| {
        try Red.warp(&b, c, false, r_tmp, f_x);
        try b.linef("mad.lo.u32 {s}, {s}, 4, {s};", .{ r_row, r_warp, r_smem });
        try b.linef("@{s} st.shared.f32 [{s}+{d}], {s};", .{ p_lane0, r_row, s * 16, c });
    }
    try Red.warp(&b, f_e, false, r_tmp, f_x);
    try b.linef("@{s} st.shared.f32 [{s}+{d}], {s};", .{ p_lane0, r_row, 9 * 16, f_e });
    try Red.warp(&b, f_m, true, r_tmp, f_x);
    try b.linef("@{s} st.shared.f32 [{s}+{d}], {s};", .{ p_lane0, r_row, 10 * 16, f_m });
    try b.line("bar.sync 0;");

    // Thread 0: the best candidate (first strict minimum), broadcast through slot 11.
    const f_a = try b.reg(.f32);
    const f_best = try b.reg(.f32);
    const r_best = try b.reg(.b32);
    const p_lt = try b.reg(.pred);
    try b.linef("setp.ne.u32 {s}, {s}, 0;", .{ p0, r_t });
    try b.linef("@{s} bra AN_WAIT1;", .{p0});
    const Sum4 = struct {
        fn emit(bb: *ptx.Builder, dst: []const u8, smem: []const u8, slot: usize, fx: []const u8, is_max: bool) !void {
            try bb.linef("mov.f32 {s}, 0f00000000;", .{dst});
            for (0..4) |w| {
                try bb.linef("ld.shared.f32 {s}, [{s}+{d}];", .{ fx, smem, slot * 16 + w * 4 });
                if (is_max) {
                    try bb.linef("max.ftz.f32 {s}, {s}, {s};", .{ dst, dst, fx });
                } else {
                    try bb.linef("add.ftz.f32 {s}, {s}, {s};", .{ dst, dst, fx });
                }
            }
        }
    };
    try b.linef("mov.f32 {s}, 0f7F7FFFFF;", .{f_best});
    try b.linef("mov.u32 {s}, 0;", .{r_best});
    for (0..ns) |s| {
        try Sum4.emit(&b, f_a, r_smem, s, f_x, false);
        try b.linef("setp.lt.ftz.f32 {s}, {s}, {s};", .{ p_lt, f_a, f_best });
        try b.linef("selp.f32 {s}, {s}, {s}, {s};", .{ f_best, f_a, f_best, p_lt });
        try b.linef("selp.u32 {s}, {d}, {s}, {s};", .{ r_best, s, r_best, p_lt });
    }
    try b.linef("st.shared.u32 [{s}+{d}], {s};", .{ r_smem, 11 * 16, r_best });
    try b.label("AN_WAIT1");
    try b.line("bar.sync 0;");

    // Residual energy and range around the best sample, the same reductions.
    try b.linef("ld.shared.u32 {s}, [{s}+{d}];", .{ r_best, r_smem, 11 * 16 });
    const f_xb = try b.reg(.f32);
    try b.linef("mov.f32 {s}, {s};", .{ f_xb, xs[0] });
    const p_s = try b.reg(.pred);
    for (1..ns) |s| {
        try b.linef("setp.eq.u32 {s}, {s}, {d};", .{ p_s, r_best, s });
        try b.linef("selp.f32 {s}, {s}, {s}, {s};", .{ f_xb, xs[s], f_xb, p_s });
    }
    const f_e2 = try b.reg(.f32);
    const f_m2 = try b.reg(.f32);
    try b.linef("mov.f32 {s}, 0f00000000;", .{f_e2});
    try b.linef("mov.f32 {s}, 0f00000000;", .{f_m2});
    for (xs) |x| {
        try b.linef("sub.ftz.f32 {s}, {s}, {s};", .{ f_x, x, f_xb });
        try b.linef("fma.rn.ftz.f32 {s}, {s}, {s}, {s};", .{ f_e2, f_x, f_x, f_e2 });
        try b.linef("abs.ftz.f32 {s}, {s};", .{ f_x, f_x });
        try b.linef("max.ftz.f32 {s}, {s}, {s};", .{ f_m2, f_m2, f_x });
    }
    try b.line("bar.sync 0;");
    try Red.warp(&b, f_e2, false, r_tmp, f_x);
    try b.linef("@{s} st.shared.f32 [{s}+{d}], {s};", .{ p_lane0, r_row, 0, f_e2 });
    try Red.warp(&b, f_m2, true, r_tmp, f_x);
    try b.linef("@{s} st.shared.f32 [{s}+{d}], {s};", .{ p_lane0, r_row, 16, f_m2 });
    try b.line("bar.sync 0;");
    try b.linef("@{s} bra AN_DONE;", .{p0});
    const f_oe = try b.reg(.f32);
    const f_om = try b.reg(.f32);
    const f_re = try b.reg(.f32);
    const f_rm = try b.reg(.f32);
    try Sum4.emit(&b, f_oe, r_smem, 9, f_x, false);
    try Sum4.emit(&b, f_om, r_smem, 10, f_x, true);
    try Sum4.emit(&b, f_re, r_smem, 0, f_x, false);
    try Sum4.emit(&b, f_rm, r_smem, 1, f_x, true);
    const p_k1 = try b.reg(.pred);
    try b.linef("setp.lt.ftz.f32 {s}, {s}, {s};", .{ p_k1, f_re, f_oe });
    try b.linef("mul.ftz.f32 {s}, {s}, 0f3F900000;", .{ f_om, f_om });
    try b.linef("setp.le.and.ftz.f32 {s}, {s}, {s}, {s};", .{ p_k1, f_rm, f_om, p_k1 });
    try b.linef("mul.lo.u32 {s}, {s}, {s};", .{ r_tmp, r_best, r_lk1 });
    try b.linef("shr.u32 {s}, {s}, 3;", .{ r_tmp, r_tmp });
    try b.linef("selp.u32 {s}, {s}, 0xffffffff, {s};", .{ r_tmp, r_tmp, p_k1 });
    try b.linef("mul.wide.u32 {s}, {s}, 4;", .{ rd_tmp, r_h });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_tmp, rd_a, rd_tmp });
    try b.linef("st.global.u32 [{s}], {s};", .{ rd_tmp, r_tmp });
    try b.label("AN_DONE");

    const shared = try alloc.print(".shared .align 16 .b8 smem[{d}];", .{Slots * 16});
    defer alloc.free(shared);
    return b.build("sage_anchor", "    .param .u64 p_k,\n    .param .u32 p_lk,\n    .param .u32 p_ks,\n    .param .u64 p_anchor", shared);
}

/// The rotation's sign words: channel i is negated when bit i % 32 of word i / 32 is clear.
const sign_words = [4]u32{ 0x1035997b, 0x8087f5ee, 0xee2e4e1a, 0x71132418 };

/// `sage_quant_qk`, 128 threads, one warp per scale group: Q's groups first
/// (heads * qScales), then K's (heads * kScales). Lane L holds channels 4L..4L+3 of
/// a row. Params: q, k (token-major, row stride in elements), lq, lk, heads, stride,
/// anchor (s32 per head), q8, qs, k8, ks.
pub fn buildQuantQK(alloc: std.mem.Allocator, in_dt: Dt) ![:0]u8 {
    var b = ptx.Builder.init(alloc);
    defer b.deinit();
    b.tuning = ".maxntid 128, 1, 1";
    const eb: usize = if (in_dt == .f32) 4 else 2;

    const rd_q = try b.reg(.b64);
    const rd_k = try b.reg(.b64);
    const rd_an = try b.reg(.b64);
    const rd_q8 = try b.reg(.b64);
    const rd_qs = try b.reg(.b64);
    const rd_k8 = try b.reg(.b64);
    const rd_ks = try b.reg(.b64);
    inline for (.{ .{ rd_q, "p_q" }, .{ rd_k, "p_k" }, .{ rd_an, "p_anchor" }, .{ rd_q8, "p_q8" }, .{ rd_qs, "p_qs" }, .{ rd_k8, "p_k8" }, .{ rd_ks, "p_ks" } }) |p| {
        try b.linef("ld.param.u64 {s}, [{s}];", .{ p[0], p[1] });
        try b.linef("cvta.to.global.u64 {s}, {s};", .{ p[0], p[0] });
    }
    const r_lq = try b.reg(.b32);
    const r_lk = try b.reg(.b32);
    const r_heads = try b.reg(.b32);
    const r_stride = try b.reg(.b32);
    inline for (.{ .{ r_lq, "p_lq" }, .{ r_lk, "p_lk" }, .{ r_heads, "p_heads" }, .{ r_stride, "p_stride" } }) |p|
        try b.linef("ld.param.u32 {s}, [{s}];", .{ p[0], p[1] });

    const r_t = try b.reg(.b32);
    const r_lane = try b.reg(.b32);
    const r_g = try b.reg(.b32); // global warp index
    const r_tmp = try b.reg(.b32);
    const r_tmp2 = try b.reg(.b32);
    const rd_tmp = try b.reg(.b64);
    try b.linef("mov.u32 {s}, %tid.x;", .{r_t});
    try b.linef("and.b32 {s}, {s}, 31;", .{ r_lane, r_t });
    try b.linef("mov.u32 {s}, %ctaid.x;", .{r_g});
    try b.linef("shr.u32 {s}, {s}, 5;", .{ r_tmp, r_t });
    try b.linef("mad.lo.u32 {s}, {s}, 4, {s};", .{ r_g, r_g, r_tmp });

    // Lane sign nibble: bits 4(L%8)..+3 of word L/8; a clear bit negates.
    const r_sign = try b.reg(.b32);
    const p_s = try b.reg(.pred);
    try b.linef("shr.u32 {s}, {s}, 3;", .{ r_tmp, r_lane });
    try b.linef("mov.u32 {s}, {d};", .{ r_sign, sign_words[0] });
    for (1..4) |w| {
        try b.linef("setp.eq.u32 {s}, {s}, {d};", .{ p_s, r_tmp, w });
        try b.linef("selp.b32 {s}, {d}, {s}, {s};", .{ r_sign, sign_words[w], r_sign, p_s });
    }
    try b.linef("and.b32 {s}, {s}, 7;", .{ r_tmp, r_lane });
    try b.linef("shl.b32 {s}, {s}, 2;", .{ r_tmp, r_tmp });
    try b.linef("shr.u32 {s}, {s}, {s};", .{ r_sign, r_sign, r_tmp });
    // xor mask per channel: 0x80000000 where the bit is clear
    const r_neg = try b.regs(.b32, 4);
    for (r_neg, 0..) |r, c| {
        try b.linef("bfe.u32 {s}, {s}, {d}, 1;", .{ r, r_sign, c });
        try b.linef("xor.b32 {s}, {s}, 1;", .{ r, r });
        try b.linef("shl.b32 {s}, {s}, 31;", .{ r, r });
    }

    // Which group this warp has: Q (head, index) or K (head, index), or none.
    const r_nq = try b.reg(.b32);
    const r_nk = try b.reg(.b32);
    const r_qsc = try b.reg(.b32);
    const r_ksc = try b.reg(.b32);
    const p_isk = try b.reg(.pred);
    const p_out = try b.reg(.pred);
    // qScales = ceil(lq/128)*32, kScales = ceil(lk/128)*4
    try b.linef("add.u32 {s}, {s}, 127;", .{ r_qsc, r_lq });
    try b.linef("shr.u32 {s}, {s}, 7;", .{ r_qsc, r_qsc });
    try b.linef("shl.b32 {s}, {s}, 5;", .{ r_qsc, r_qsc });
    try b.linef("add.u32 {s}, {s}, 127;", .{ r_ksc, r_lk });
    try b.linef("shr.u32 {s}, {s}, 7;", .{ r_ksc, r_ksc });
    try b.linef("shl.b32 {s}, {s}, 2;", .{ r_ksc, r_ksc });
    try b.linef("mul.lo.u32 {s}, {s}, {s};", .{ r_nq, r_qsc, r_heads });
    try b.linef("mul.lo.u32 {s}, {s}, {s};", .{ r_nk, r_ksc, r_heads });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_tmp, r_nq, r_nk });
    try b.linef("setp.ge.u32 {s}, {s}, {s};", .{ p_out, r_g, r_tmp });
    try b.linef("@{s} bra QK_END;", .{p_out});
    try b.linef("setp.ge.u32 {s}, {s}, {s};", .{ p_isk, r_g, r_nq });

    // Per-role values: source, length, head, first member row, code and scale bases.
    const r_head = try b.reg(.b32);
    const r_gi = try b.reg(.b32);
    const r_len = try b.reg(.b32);
    const r_r0 = try b.reg(.b32); // first member row
    const rd_src = try b.reg(.b64);
    const rd_dst8 = try b.reg(.b64);
    const rd_dsts = try b.reg(.b64);
    const r_rows8 = try b.reg(.b32); // padded rows per head in the code buffer
    const r_anchor = try b.reg(.b32);
    try b.linef("@{s} bra QK_ISK;", .{p_isk});
    // Q: gi = g % qsc, head = g / qsc; rows 32*(gi/8) + gi%8 + 8j, j < 4
    try b.linef("div.u32 {s}, {s}, {s};", .{ r_head, r_g, r_qsc });
    try b.linef("rem.u32 {s}, {s}, {s};", .{ r_gi, r_g, r_qsc });
    try b.linef("shr.u32 {s}, {s}, 3;", .{ r_tmp, r_gi });
    try b.linef("shl.b32 {s}, {s}, 5;", .{ r_r0, r_tmp });
    try b.linef("and.b32 {s}, {s}, 7;", .{ r_tmp, r_gi });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_r0, r_r0, r_tmp });
    try b.linef("mov.u32 {s}, {s};", .{ r_len, r_lq });
    try b.linef("mov.b64 {s}, {s};", .{ rd_src, rd_q });
    try b.linef("mov.b64 {s}, {s};", .{ rd_dst8, rd_q8 });
    try b.linef("mul.wide.u32 {s}, {s}, 4;", .{ rd_tmp, r_g });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_dsts, rd_qs, rd_tmp });
    try b.linef("shl.b32 {s}, {s}, 2;", .{ r_rows8, r_qsc }); // qsc*4 = padded rows
    try b.linef("mov.u32 {s}, 0xffffffff;", .{r_anchor});
    try b.line("bra.uni QK_GO;");
    try b.label("QK_ISK");
    // K: g -= nq; gi = g % ksc, head = g / ksc; keys 128*(gi/4) + 2*(gi%4) + 8j + p
    try b.linef("sub.u32 {s}, {s}, {s};", .{ r_tmp2, r_g, r_nq });
    try b.linef("div.u32 {s}, {s}, {s};", .{ r_head, r_tmp2, r_ksc });
    try b.linef("rem.u32 {s}, {s}, {s};", .{ r_gi, r_tmp2, r_ksc });
    try b.linef("shr.u32 {s}, {s}, 2;", .{ r_tmp, r_gi });
    try b.linef("shl.b32 {s}, {s}, 7;", .{ r_r0, r_tmp });
    try b.linef("and.b32 {s}, {s}, 3;", .{ r_tmp, r_gi });
    try b.linef("shl.b32 {s}, {s}, 1;", .{ r_tmp, r_tmp });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_r0, r_r0, r_tmp });
    try b.linef("mov.u32 {s}, {s};", .{ r_len, r_lk });
    try b.linef("mov.b64 {s}, {s};", .{ rd_src, rd_k });
    try b.linef("mov.b64 {s}, {s};", .{ rd_dst8, rd_k8 });
    try b.linef("mul.wide.u32 {s}, {s}, 4;", .{ rd_tmp, r_tmp2 });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_dsts, rd_ks, rd_tmp });
    try b.linef("shl.b32 {s}, {s}, 5;", .{ r_rows8, r_ksc }); // ksc*32 = padded keys
    try b.linef("mul.wide.u32 {s}, {s}, 4;", .{ rd_tmp, r_head });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_tmp, rd_an, rd_tmp });
    try b.linef("ld.global.u32 {s}, [{s}];", .{ r_anchor, rd_tmp });
    try b.label("QK_GO");

    // This lane's source column offset (head*128 + 4L) * eb and the anchor values.
    const rd_col = try b.reg(.b64);
    try b.linef("mad.lo.u32 {s}, {s}, {d}, {s};", .{ r_tmp, r_head, hd / 4, r_lane });
    try b.linef("mul.wide.u32 {s}, {s}, {d};", .{ rd_col, r_tmp, 4 * eb });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_src, rd_src, rd_col });
    const f_bias = try b.regs(.f32, 4);
    const p_an = try b.reg(.pred);
    const f_p = try b.reg(.f32);
    const f_tmp = try b.regs(.f32, 4);
    try b.linef("setp.ge.s32 {s}, {s}, 0;", .{ p_an, r_anchor });
    for (f_bias) |f| try b.linef("mov.f32 {s}, 0f00000000;", .{f});
    try b.linef("@!{s} bra QK_NOAN;", .{p_an});
    try b.linef("mul.lo.u32 {s}, {s}, {s};", .{ r_tmp, r_anchor, r_stride });
    try b.linef("mul.wide.u32 {s}, {s}, {d};", .{ rd_tmp, r_tmp, eb });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_tmp, rd_src, rd_tmp });
    try loadRow(&b, in_dt, f_bias, rd_tmp, r_tmp);
    try b.label("QK_NOAN");

    const Rot = struct {
        fn emit(bb: *ptx.Builder, x: []const []const u8, neg: []const []const u8, lane: []const u8, ft: []const []const u8, fp: []const u8, rt: []const u8, pb: []const u8) !void {
            for (x, neg) |v, n| {
                try bb.linef("mov.b32 {s}, {s};", .{ rt, v });
                try bb.linef("xor.b32 {s}, {s}, {s};", .{ rt, rt, n });
                try bb.linef("mov.b32 {s}, {s};", .{ v, rt });
            }
            // H4: a0=x0+x1 a1=x0-x1 a2=x2+x3 a3=x2-x3; y=(a0+a2,a1+a3,a0-a2,a1-a3)*0.5
            try bb.linef("add.ftz.f32 {s}, {s}, {s};", .{ ft[0], x[0], x[1] });
            try bb.linef("sub.ftz.f32 {s}, {s}, {s};", .{ ft[1], x[0], x[1] });
            try bb.linef("add.ftz.f32 {s}, {s}, {s};", .{ ft[2], x[2], x[3] });
            try bb.linef("sub.ftz.f32 {s}, {s}, {s};", .{ ft[3], x[2], x[3] });
            try bb.linef("add.ftz.f32 {s}, {s}, {s};", .{ x[0], ft[0], ft[2] });
            try bb.linef("add.ftz.f32 {s}, {s}, {s};", .{ x[1], ft[1], ft[3] });
            try bb.linef("sub.ftz.f32 {s}, {s}, {s};", .{ x[2], ft[0], ft[2] });
            try bb.linef("sub.ftz.f32 {s}, {s}, {s};", .{ x[3], ft[1], ft[3] });
            for (x) |v| try bb.linef("mul.ftz.f32 {s}, {s}, 0f3F000000;", .{ v, v });
            for ([_]u32{ 1, 2, 4, 8, 16 }) |bit| {
                try bb.linef("and.b32 {s}, {s}, {d};", .{ rt, lane, bit });
                try bb.linef("setp.ne.u32 {s}, {s}, 0;", .{ pb, rt });
                for (x) |v| {
                    try bb.linef("mov.b32 {s}, {s};", .{ rt, v });
                    try bb.linef("shfl.sync.bfly.b32 {s}, {s}, {d}, 0x1f, 0xffffffff;", .{ rt, rt, bit });
                    try bb.linef("mov.b32 {s}, {s};", .{ fp, rt });
                    // low lane: lo + hi ; high lane: lo - hi = partner - self
                    try bb.linef("@!{s} add.ftz.f32 {s}, {s}, {s};", .{ pb, v, v, fp });
                    try bb.linef("@{s} sub.ftz.f32 {s}, {s}, {s};", .{ pb, v, fp, v });
                }
            }
            for (x) |v| try bb.linef("mul.ftz.f32 {s}, {s}, 0f{X:0>8};", .{ v, v, hex(0.1767766952966369) });
        }
    };
    const p_hi = try b.reg(.pred);
    const r_rt = try b.reg(.b32);
    const r_row = try b.reg(.b32);
    const p_valid = try b.reg(.pred);
    const f_mx = try b.reg(.f32);

    // One pass: every member row of the group stays in registers (Q 4 rows, K 32),
    // so the input is read once.
    const Group = struct {
        fn emit(bb: *ptx.Builder, comptime offs: []const usize, e: anytype) !void {
            const xs = try bb.regs(.f32, offs.len * 4);
            try bb.linef("mov.f32 {s}, 0f00000000;", .{e.f_mx});
            // every member's load first, predicated on the row existing
            const valid = try bb.regs(.pred, offs.len);
            for (offs, 0..) |off, j| {
                const x = xs[j * 4 ..][0..4];
                const rd = try bb.reg(.b64);
                try bb.linef("add.u32 {s}, {s}, {d};", .{ e.r_row, e.r_r0, off });
                try bb.linef("setp.lt.u32 {s}, {s}, {s};", .{ valid[j], e.r_row, e.r_len });
                try bb.linef("mul.lo.u32 {s}, {s}, {s};", .{ e.r_tmp, e.r_row, e.r_stride });
                try bb.linef("mul.wide.u32 {s}, {s}, {d};", .{ rd, e.r_tmp, e.eb });
                try bb.linef("add.s64 {s}, {s}, {s};", .{ rd, e.rd_src, rd });
                for (x) |f| try bb.linef("mov.f32 {s}, 0f00000000;", .{f});
                try loadRowIf(bb, e.in_dt, x, rd, try bb.reg(.b32), valid[j]);
            }
            for (offs, 0..) |_, j| {
                const x = xs[j * 4 ..][0..4];
                // a row past len stays zero, through the shift and the rotation
                try bb.linef("and.pred {s}, {s}, {s};", .{ valid[j], valid[j], e.p_an });
                for (x, e.f_bias) |v, bias| try bb.linef("@{s} sub.ftz.f32 {s}, {s}, {s};", .{ valid[j], v, v, bias });
                try Rot.emit(bb, x, e.r_neg, e.r_lane, e.f_tmp, e.f_p, e.r_rt, e.p_hi);
                for (x) |v| {
                    try bb.linef("abs.ftz.f32 {s}, {s};", .{ e.f_p, v });
                    try bb.linef("max.ftz.f32 {s}, {s}, {s};", .{ e.f_mx, e.f_mx, e.f_p });
                }
            }
            // warp max, then sc = mx / 127 + 1e-7 and inv = 1 / sc (both approximate)
            for ([_]u32{ 16, 8, 4, 2, 1 }) |o| {
                try bb.linef("mov.b32 {s}, {s};", .{ e.r_rt, e.f_mx });
                try bb.linef("shfl.sync.bfly.b32 {s}, {s}, {d}, 0x1f, 0xffffffff;", .{ e.r_rt, e.r_rt, o });
                try bb.linef("mov.b32 {s}, {s};", .{ e.f_p, e.r_rt });
                try bb.linef("max.ftz.f32 {s}, {s}, {s};", .{ e.f_mx, e.f_mx, e.f_p });
            }
            try bb.linef("div.approx.ftz.f32 {s}, {s}, 0f42FE0000;", .{ e.f_mx, e.f_mx });
            try bb.linef("add.ftz.f32 {s}, {s}, 0f33D6BF95;", .{ e.f_mx, e.f_mx });
            try bb.linef("setp.eq.u32 {s}, {s}, 0;", .{ e.p_valid, e.r_lane });
            try bb.linef("@{s} st.global.f32 [{s}], {s};", .{ e.p_valid, e.rd_dsts, e.f_mx });
            try bb.linef("rcp.approx.ftz.f32 {s}, {s};", .{ e.f_mx, e.f_mx });
            // codes, cvt.rni.sat.s8(x * inv) packed 4 per lane, at
            // dst + (head * rows8 + row) * 128 + 4L; every member row is inside the padding
            try bb.linef("mad.lo.u32 {s}, {s}, {s}, {s};", .{ e.r_tmp, e.r_head, e.r_rows8, e.r_r0 });
            try bb.linef("mul.wide.u32 {s}, {s}, {d};", .{ e.rd_tmp, e.r_tmp, hd });
            try bb.linef("add.s64 {s}, {s}, {s};", .{ e.rd_tmp, e.rd_tmp, e.rd_dst8 });
            try bb.linef("mul.wide.u32 {s}, {s}, 4;", .{ e.rd_col, e.r_lane });
            try bb.linef("add.s64 {s}, {s}, {s};", .{ e.rd_tmp, e.rd_tmp, e.rd_col });
            const rc = try bb.regs(.b32, 4);
            for (offs, 0..) |off, j| {
                for (xs[j * 4 ..][0..4], rc) |v, r| {
                    try bb.linef("mul.ftz.f32 {s}, {s}, {s};", .{ e.f_p, v, e.f_mx });
                    try bb.linef("cvt.rni.sat.s8.f32 {s}, {s};", .{ r, e.f_p });
                }
                try bb.linef("prmt.b32 {s}, {s}, {s}, 0x3340;", .{ rc[0], rc[0], rc[1] });
                try bb.linef("prmt.b32 {s}, {s}, {s}, 0x3340;", .{ rc[2], rc[2], rc[3] });
                try bb.linef("prmt.b32 {s}, {s}, {s}, 0x5410;", .{ rc[0], rc[0], rc[2] });
                try bb.linef("st.global.b32 [{s}+{d}], {s};", .{ e.rd_tmp, off * hd, rc[0] });
            }
        }
    };
    const env = .{
        .r_row = r_row, .r_r0 = r_r0, .p_valid = p_valid, .r_len = r_len, .r_tmp = r_tmp,
        .r_stride = r_stride, .rd_tmp = rd_tmp, .eb = eb, .rd_src = rd_src, .in_dt = in_dt,
        .f_bias = f_bias, .p_an = p_an, .r_neg = r_neg, .r_lane = r_lane, .f_tmp = f_tmp,
        .f_p = f_p, .r_rt = r_rt, .p_hi = p_hi, .f_mx = f_mx, .rd_dsts = rd_dsts,
        .r_head = r_head, .r_rows8 = r_rows8, .rd_dst8 = rd_dst8, .rd_col = rd_col,
    };
    const q_offs = comptime blk: {
        var o: [4]usize = undefined;
        for (&o, 0..) |*v, j| v.* = 8 * j;
        break :blk o;
    };
    const k_offs = comptime blk: {
        var o: [32]usize = undefined;
        for (&o, 0..) |*v, j| v.* = 8 * (j / 2) + j % 2;
        break :blk o;
    };
    try b.linef("@{s} bra QK_GROUPK;", .{p_isk});
    try Group.emit(&b, &q_offs, env);
    try b.line("bra.uni QK_END;");
    try b.label("QK_GROUPK");
    try Group.emit(&b, &k_offs, env);
    try b.label("QK_END");

    return b.build("sage_quant_qk",
        \\    .param .u64 p_q,
        \\    .param .u64 p_k,
        \\    .param .u32 p_lq,
        \\    .param .u32 p_lk,
        \\    .param .u32 p_heads,
        \\    .param .u32 p_stride,
        \\    .param .u64 p_anchor,
        \\    .param .u64 p_q8,
        \\    .param .u64 p_qs,
        \\    .param .u64 p_k8,
        \\    .param .u64 p_ks
    , "");
}

/// Four consecutive input elements at `addr` into f32 `dst`.
fn loadRow(b: *ptx.Builder, in_dt: Dt, dst: []const []const u8, addr: []const u8, r_tmp: []const u8) !void {
    return loadRowIf(b, in_dt, dst, addr, r_tmp, null);
}

/// `loadRow` under predicate `pred`; `dst` keeps its value where it is false.
fn loadRowIf(b: *ptx.Builder, in_dt: Dt, dst: []const []const u8, addr: []const u8, r_tmp: []const u8, pred: ?[]const u8) !void {
    var pbuf: [16]u8 = undefined;
    const pre = if (pred) |p| try std.fmt.bufPrint(&pbuf, "@{s} ", .{p}) else "";
    if (in_dt == .f16) {
        const w1 = try b.reg(.b32);
        const w0 = r_tmp;
        try b.linef("mov.u32 {s}, 0; mov.u32 {s}, 0;", .{ w0, w1 });
        try b.linef("{s}ld.global.v2.u32 {{{s}, {s}}}, [{s}];", .{ pre, w0, w1, addr });
        for ([_][]const u8{ w0, w1 }, 0..) |w, i| {
            const lo = try b.reg(.b16);
            const hi = try b.reg(.b16);
            try b.linef("mov.b32 {{{s}, {s}}}, {s};", .{ lo, hi, w });
            try b.linef("cvt.f32.f16 {s}, {s};", .{ dst[2 * i], lo });
            try b.linef("cvt.f32.f16 {s}, {s};", .{ dst[2 * i + 1], hi });
        }
    } else if (in_dt == .bf16) {
        // channel 4L is the low half of the first word
        const w1 = try b.reg(.b32);
        const w0 = r_tmp;
        try b.linef("mov.u32 {s}, 0; mov.u32 {s}, 0;", .{ w0, w1 });
        try b.linef("{s}ld.global.v2.u32 {{{s}, {s}}}, [{s}];", .{ pre, w0, w1, addr });
        for ([_][]const u8{ w0, w1 }, 0..) |w, i| {
            const lo = try b.reg(.b32);
            try b.linef("shl.b32 {s}, {s}, 16;", .{ lo, w });
            try b.linef("mov.b32 {s}, {s};", .{ dst[2 * i], lo });
            try b.linef("and.b32 {s}, {s}, 0xffff0000;", .{ lo, w });
            try b.linef("mov.b32 {s}, {s};", .{ dst[2 * i + 1], lo });
        }
    } else {
        try b.linef("{s}ld.global.v4.f32 {{{s}, {s}, {s}, {s}}}, [{s}];", .{ pre, dst[0], dst[1], dst[2], dst[3], addr });
    }
}

/// `sage_vmax`, grid (heads, kRows(lk) / 128), 128 threads: thread c folds |v| of
/// its channel over one key tile into `vmax` (u32 bits of a non-negative f32, which
/// order like the floats; zeroed first). Params: v, lk, stride, vmax.
pub fn buildVMax(alloc: std.mem.Allocator, in_dt: Dt) ![:0]u8 {
    var b = ptx.Builder.init(alloc);
    defer b.deinit();
    b.tuning = ".maxntid 128, 1, 1";
    const eb: usize = if (in_dt == .f32) 4 else 2;
    const rd_v = try b.reg(.b64);
    const rd_m = try b.reg(.b64);
    const r_lk = try b.reg(.b32);
    const r_stride = try b.reg(.b32);
    try b.linef("ld.param.u64 {s}, [p_v];", .{rd_v});
    try b.linef("cvta.to.global.u64 {s}, {s};", .{ rd_v, rd_v });
    try b.linef("ld.param.u64 {s}, [p_vmax];", .{rd_m});
    try b.linef("cvta.to.global.u64 {s}, {s};", .{ rd_m, rd_m });
    try b.linef("ld.param.u32 {s}, [p_lk];", .{r_lk});
    try b.linef("ld.param.u32 {s}, [p_stride];", .{r_stride});
    const r_t = try b.reg(.b32);
    const r_h = try b.reg(.b32);
    const r_n0 = try b.reg(.b32);
    const r_n = try b.reg(.b32);
    const r_end = try b.reg(.b32);
    const r_tmp = try b.reg(.b32);
    const rd_tmp = try b.reg(.b64);
    const rd_p = try b.reg(.b64);
    const rd_step = try b.reg(.b64);
    const f_m = try b.reg(.f32);
    const f_x = try b.reg(.f32);
    const p = try b.reg(.pred);
    try b.linef("mov.u32 {s}, %tid.x;", .{r_t});
    try b.linef("mov.u32 {s}, %ctaid.x;", .{r_h});
    try b.linef("mov.u32 {s}, %ctaid.y;", .{r_n0});
    try b.linef("shl.b32 {s}, {s}, 7;", .{ r_n0, r_n0 });
    try b.linef("add.u32 {s}, {s}, {d};", .{ r_end, r_n0, key_tile });
    try b.linef("min.u32 {s}, {s}, {s};", .{ r_end, r_end, r_lk });
    // p = v + (n0 * stride + h*128 + c) * eb
    try b.linef("mad.lo.u32 {s}, {s}, {d}, {s};", .{ r_tmp, r_h, hd, r_t });
    try b.linef("mad.lo.u32 {s}, {s}, {s}, {s};", .{ r_tmp, r_n0, r_stride, r_tmp });
    try b.linef("mul.wide.u32 {s}, {s}, {d};", .{ rd_tmp, r_tmp, eb });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_p, rd_v, rd_tmp });
    try b.linef("mul.wide.u32 {s}, {s}, {d};", .{ rd_step, r_stride, eb });
    try b.linef("mov.f32 {s}, 0f00000000;", .{f_m});
    try b.linef("mov.u32 {s}, {s};", .{ r_n, r_n0 });
    try b.linef("setp.ge.u32 {s}, {s}, {s};", .{ p, r_n, r_end });
    try b.linef("@{s} bra VM_DONE;", .{p});
    try b.label("VM_LOOP");
    try loadIn(&b, in_dt, f_x, rd_p, r_tmp);
    try b.linef("abs.ftz.f32 {s}, {s};", .{ f_x, f_x });
    try b.linef("max.ftz.f32 {s}, {s}, {s};", .{ f_m, f_m, f_x });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_p, rd_p, rd_step });
    try b.linef("add.u32 {s}, {s}, 1;", .{ r_n, r_n });
    try b.linef("setp.lt.u32 {s}, {s}, {s};", .{ p, r_n, r_end });
    try b.linef("@{s} bra VM_LOOP;", .{p});
    try b.label("VM_DONE");
    try b.linef("mad.lo.u32 {s}, {s}, {d}, {s};", .{ r_tmp, r_h, hd, r_t });
    try b.linef("mul.wide.u32 {s}, {s}, 4;", .{ rd_tmp, r_tmp });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_tmp, rd_m, rd_tmp });
    try b.linef("mov.b32 {s}, {s};", .{ r_tmp, f_m });
    try b.linef("red.global.max.u32 [{s}], {s};", .{ rd_tmp, r_tmp });
    return b.build("sage_vmax", "    .param .u64 p_v,\n    .param .u32 p_lk,\n    .param .u32 p_stride,\n    .param .u64 p_vmax", "");
}

/// `sage_quant_v`, grid (heads, kRows(lk) / 128), 128 threads: one key tile of one
/// head, staged through shared memory with coalesced loads (`chunk` channels at a
/// time), then each channel's codes written as row (head*128 + c) of v8
/// [heads*128][kRows], each 16 keys permuted so keys 2t, 2t+1, 2t+8, 2t+9 land at
/// 4t..4t+3. Keys past lk are zero. Block y 0 also writes the scales. Params: v, lk,
/// stride, vmax, v8, vs.
pub fn buildQuantV(alloc: std.mem.Allocator, in_dt: Dt) ![:0]u8 {
    var b = ptx.Builder.init(alloc);
    defer b.deinit();
    b.tuning = ".maxntid 128, 1, 1";
    const eb: usize = if (in_dt == .f32) 4 else 2;
    const SMEM = 32 * 1024;
    const CH = SMEM / (key_tile * eb); // channels staged per pass
    const ROWB = CH * eb;
    const TPC = 128 / CH; // threads per channel, splitting the 8 key blocks
    const rd_v = try b.reg(.b64);
    const rd_m = try b.reg(.b64);
    const rd_v8 = try b.reg(.b64);
    const rd_vs = try b.reg(.b64);
    inline for (.{ .{ rd_v, "p_v" }, .{ rd_m, "p_vmax" }, .{ rd_v8, "p_v8" }, .{ rd_vs, "p_vs" } }) |p| {
        try b.linef("ld.param.u64 {s}, [{s}];", .{ p[0], p[1] });
        try b.linef("cvta.to.global.u64 {s}, {s};", .{ p[0], p[0] });
    }
    const r_lk = try b.reg(.b32);
    const r_stride = try b.reg(.b32);
    try b.linef("ld.param.u32 {s}, [p_lk];", .{r_lk});
    try b.linef("ld.param.u32 {s}, [p_stride];", .{r_stride});
    const r_t = try b.reg(.b32);
    const r_h = try b.reg(.b32);
    const r_ty = try b.reg(.b32);
    const r_n0 = try b.reg(.b32);
    const r_kr = try b.reg(.b32);
    const r_tmp = try b.reg(.b32);
    const r_smem = try b.reg(.b32);
    const r_c = try b.reg(.b32); // channel within the pass
    const r_kb0 = try b.reg(.b32); // first key block this thread writes
    const rd_tmp = try b.reg(.b64);
    const p = try b.reg(.pred);
    const p_y0 = try b.reg(.pred);
    const f_x = try b.reg(.f32);
    const f_inv = try b.reg(.f32);
    try b.linef("mov.u32 {s}, %tid.x;", .{r_t});
    try b.linef("mov.u32 {s}, %ctaid.x;", .{r_h});
    try b.linef("mov.u32 {s}, %ctaid.y;", .{r_ty});
    try b.linef("setp.eq.u32 {s}, {s}, 0;", .{ p_y0, r_ty });
    try b.linef("shl.b32 {s}, {s}, 7;", .{ r_n0, r_ty });
    try b.linef("mov.u32 {s}, %nctaid.y;", .{r_kr});
    try b.linef("shl.b32 {s}, {s}, 7;", .{ r_kr, r_kr });
    try b.linef("mov.u32 {s}, smem;", .{r_smem});
    try b.linef("and.b32 {s}, {s}, {d};", .{ r_c, r_t, CH - 1 });
    try b.linef("shr.u32 {s}, {s}, {d};", .{ r_kb0, r_t, std.math.log2_int(usize, CH) });
    try b.linef("mul.lo.u32 {s}, {s}, {d};", .{ r_kb0, r_kb0, 8 / TPC });

    // loader: 16-byte piece t % (ROWB/16) of rows t / (ROWB/16) + i * rps
    const PPR = ROWB / 16;
    const RPS = 128 / PPR;
    const r_lr = try b.reg(.b32);
    const r_lp = try b.reg(.b32);
    try b.linef("shr.u32 {s}, {s}, {d};", .{ r_lr, r_t, std.math.log2_int(usize, PPR) });
    try b.linef("and.b32 {s}, {s}, {d};", .{ r_lp, r_t, PPR - 1 });
    const w = try b.regs(.b32, 4);
    const code = try b.regs(.b32, 16);
    const words = try b.regs(.b32, 4);
    const r_n = try b.reg(.b32);
    const r_row = try b.reg(.b32);
    const rd_dst = try b.reg(.b64);
    const h16 = try b.reg(.b16);
    for (0..hd / CH) |pass| {
        if (pass > 0) try b.line("bar.sync 0;");
        for (0..key_tile / RPS) |i| {
            // key n0 + lr + i*RPS, channels pass*CH + lp*16/eb ..
            try b.linef("add.u32 {s}, {s}, {d};", .{ r_n, r_lr, i * RPS });
            try b.linef("add.u32 {s}, {s}, {s};", .{ r_tmp, r_n, r_n0 });
            try b.linef("setp.lt.u32 {s}, {s}, {s};", .{ p, r_tmp, r_lk });
            try b.linef("mad.lo.u32 {s}, {s}, {s}, {d};", .{ r_tmp, r_tmp, r_stride, pass * CH });
            try b.linef("mad.lo.u32 {s}, {s}, {d}, {s};", .{ r_tmp, r_h, hd, r_tmp });
            try b.linef("mul.wide.u32 {s}, {s}, {d};", .{ rd_tmp, r_tmp, eb });
            try b.linef("add.s64 {s}, {s}, {s};", .{ rd_tmp, rd_tmp, rd_v });
            try b.linef("mul.wide.u32 {s}, {s}, 16;", .{ rd_dst, r_lp });
            try b.linef("add.s64 {s}, {s}, {s};", .{ rd_tmp, rd_tmp, rd_dst });
            for (w) |r| try b.linef("mov.u32 {s}, 0;", .{r});
            try b.linef("@{s} ld.global.v4.u32 {{{s}, {s}, {s}, {s}}}, [{s}];", .{ p, w[0], w[1], w[2], w[3], rd_tmp });
            try b.linef("mad.lo.u32 {s}, {s}, {d}, {s};", .{ r_tmp, r_n, ROWB, r_smem });
            try b.linef("mad.lo.u32 {s}, {s}, 16, {s};", .{ r_tmp, r_lp, r_tmp });
            try b.linef("st.shared.v4.u32 [{s}], {{{s}, {s}, {s}, {s}}};", .{ r_tmp, w[0], w[1], w[2], w[3] });
        }
        try b.line("bar.sync 0;");

        // this thread's channel: sc = max(amax * (1/127), 1e-12); inv = rcp.approx(sc)
        try b.linef("mad.lo.u32 {s}, {s}, {d}, {s};", .{ r_row, r_h, hd, r_c });
        try b.linef("add.u32 {s}, {s}, {d};", .{ r_row, r_row, pass * CH });
        try b.linef("mul.wide.u32 {s}, {s}, 4;", .{ rd_tmp, r_row });
        try b.linef("add.s64 {s}, {s}, {s};", .{ rd_dst, rd_m, rd_tmp });
        try b.linef("ld.global.f32 {s}, [{s}];", .{ f_x, rd_dst });
        try b.linef("mul.ftz.f32 {s}, {s}, 0f{X:0>8};", .{ f_x, f_x, hex(1.0 / 127.0) });
        try b.linef("max.ftz.f32 {s}, {s}, 0f2B8CBCCC;", .{ f_x, f_x });
        try b.linef("add.s64 {s}, {s}, {s};", .{ rd_dst, rd_vs, rd_tmp });
        // the first thread of each channel writes its scale
        try b.linef("setp.lt.and.u32 {s}, {s}, {d}, {s};", .{ p, r_t, CH, p_y0 });
        try b.linef("@{s} st.global.f32 [{s}], {s};", .{ p, rd_dst, f_x });
        try b.linef("rcp.approx.ftz.f32 {s}, {s};", .{ f_inv, f_x });
        // destination row: v8 + row * kRows + n0
        try b.linef("mul.wide.u32 {s}, {s}, {s};", .{ rd_dst, r_row, r_kr });
        try b.linef("add.s64 {s}, {s}, {s};", .{ rd_dst, rd_dst, rd_v8 });
        try b.linef("cvt.u64.u32 {s}, {s};", .{ rd_tmp, r_n0 });
        try b.linef("add.s64 {s}, {s}, {s};", .{ rd_dst, rd_dst, rd_tmp });
        // smem column of this channel
        const r_col = try b.reg(.b32);
        try b.linef("mad.lo.u32 {s}, {s}, {d}, {s};", .{ r_col, r_c, eb, r_smem });
        try b.linef("mad.lo.u32 {s}, {s}, {d}, {s};", .{ r_col, r_kb0, 16 * ROWB, r_col });
        for (0..8 / TPC) |kb| {
            for (0..16) |k| {
                const off = (kb * 16 + k) * ROWB;
                if (in_dt == .bf16) {
                    try b.linef("ld.shared.u16 {s}, [{s}+{d}];", .{ r_tmp, r_col, off });
                    try b.linef("shl.b32 {s}, {s}, 16;", .{ r_tmp, r_tmp });
                    try b.linef("mov.b32 {s}, {s};", .{ f_x, r_tmp });
                } else if (in_dt == .f16) {
                    try b.linef("ld.shared.b16 {s}, [{s}+{d}];", .{ h16, r_col, off });
                    try b.linef("cvt.f32.f16 {s}, {s};", .{ f_x, h16 });
                } else {
                    try b.linef("ld.shared.f32 {s}, [{s}+{d}];", .{ f_x, r_col, off });
                }
                // a key past lk was staged as zero, and codes to zero
                try b.linef("mul.ftz.f32 {s}, {s}, {s};", .{ f_x, f_x, f_inv });
                try b.linef("cvt.rni.sat.s8.f32 {s}, {s};", .{ code[k], f_x });
            }
            for (0..4) |t| {
                const ks = [4]usize{ 2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9 };
                try b.linef("prmt.b32 {s}, {s}, {s}, 0x3340;", .{ words[t], code[ks[0]], code[ks[1]] });
                try b.linef("prmt.b32 {s}, {s}, {s}, 0x3340;", .{ r_tmp, code[ks[2]], code[ks[3]] });
                try b.linef("prmt.b32 {s}, {s}, {s}, 0x5410;", .{ words[t], words[t], r_tmp });
            }
            // block kb0 + kb of the tile
            try b.linef("mad.lo.u32 {s}, {s}, 16, {d};", .{ r_tmp, r_kb0, kb * 16 });
            try b.linef("cvt.u64.u32 {s}, {s};", .{ rd_tmp, r_tmp });
            try b.linef("add.s64 {s}, {s}, {s};", .{ rd_tmp, rd_tmp, rd_dst });
            try b.linef("st.global.v4.b32 [{s}], {{{s}, {s}, {s}, {s}}};", .{ rd_tmp, words[0], words[1], words[2], words[3] });
        }
    }
    const shared = try alloc.print(".shared .align 16 .b8 smem[{d}];", .{SMEM});
    defer alloc.free(shared);
    return b.build("sage_quant_v", "    .param .u64 p_v,\n    .param .u32 p_lk,\n    .param .u32 p_stride,\n    .param .u64 p_vmax,\n    .param .u64 p_v8,\n    .param .u64 p_vs", shared);
}

/// `sage_attn`, grid (ceil(lq / q_tile), heads), 128 threads: 4 warps of 16 query
/// rows against 128-key tiles. Every tile but the last is comfy-kitchen's fused
/// path (integer tile max, u8 denominator); the last masks keys past lk, floors each
/// lane's max at -50000 and sums the fp32 probabilities. Params: q8, k8, v8 (from
/// the quant kernels), qs, ks, vs, o, lq, lk, qrows and krows (padded rows per
/// head), o row stride (elements), scale * log2(e). The result is rounded as
/// comfy-kitchen stores it, to f16 for f16 operands and to bf16 otherwise, and
/// `out_dt` f32 holds that rounded value.
pub fn buildAttn(alloc: std.mem.Allocator, in_dt: Dt, out_dt: Dt) ![:0]u8 {
    const round: Dt = if (in_dt == .f16) .f16 else .bf16;
    std.debug.assert(out_dt == .f32 or out_dt == round);
    const rname = if (round == .f16) "f16" else "bf16";
    const ROWB = 128;
    const Q_OFF = 0;
    const K_OFF = q_tile * ROWB;
    const V_OFF = K_OFF + key_tile * ROWB;
    const SH = V_OFF + hd * ROWB;
    const KT = ROWB / 32; // k-steps of 32 bytes
    const NS = key_tile / 8; // score n-tiles
    const ND = hd / 8; // output n-tiles
    const oel: usize = out_dt.bytes();
    const off_neg = hex(-7.9943534);

    var b = ptx.Builder.init(alloc);
    defer b.deinit();
    b.tuning = ".maxntid 128, 1, 1";

    const rd_q8 = try b.reg(.b64);
    const rd_k8 = try b.reg(.b64);
    const rd_v8 = try b.reg(.b64);
    const rd_qs = try b.reg(.b64);
    const rd_ks = try b.reg(.b64);
    const rd_vs = try b.reg(.b64);
    const rd_o = try b.reg(.b64);
    inline for (.{ .{ rd_q8, "p_q8" }, .{ rd_k8, "p_k8" }, .{ rd_v8, "p_v8" }, .{ rd_qs, "p_qs" }, .{ rd_ks, "p_ks" }, .{ rd_vs, "p_vs" }, .{ rd_o, "p_o" } }) |p| {
        try b.linef("ld.param.u64 {s}, [{s}];", .{ p[0], p[1] });
        try b.linef("cvta.to.global.u64 {s}, {s};", .{ p[0], p[0] });
    }
    const r_lq = try b.reg(.b32);
    const r_lk = try b.reg(.b32);
    const r_qrows = try b.reg(.b32);
    const r_krows = try b.reg(.b32);
    const r_os = try b.reg(.b32);
    const f_sm2 = try b.reg(.f32);
    inline for (.{ .{ r_lq, "p_lq" }, .{ r_lk, "p_lk" }, .{ r_qrows, "p_qrows" }, .{ r_krows, "p_krows" }, .{ r_os, "p_os" } }) |p|
        try b.linef("ld.param.u32 {s}, [{s}];", .{ p[0], p[1] });
    try b.linef("ld.param.f32 {s}, [p_sm2];", .{f_sm2});

    const r_t = try b.reg(.b32);
    const r_lane = try b.reg(.b32);
    const r_warp = try b.reg(.b32);
    const r_gid = try b.reg(.b32);
    const r_tf = try b.reg(.b32);
    const r_h = try b.reg(.b32);
    const r_q0 = try b.reg(.b32);
    const r_tmp = try b.reg(.b32);
    const r_tmp2 = try b.reg(.b32);
    const rd_tmp = try b.reg(.b64);
    try b.linef("mov.u32 {s}, %tid.x;", .{r_t});
    try b.linef("and.b32 {s}, {s}, 31;", .{ r_lane, r_t });
    try b.linef("shr.u32 {s}, {s}, 5;", .{ r_warp, r_t });
    try b.linef("shr.u32 {s}, {s}, 2;", .{ r_gid, r_lane });
    try b.linef("and.b32 {s}, {s}, 3;", .{ r_tf, r_lane });
    try b.linef("mov.u32 {s}, %ctaid.y;", .{r_h});
    try b.linef("mov.u32 {s}, %ctaid.x;", .{r_q0});
    try b.linef("mul.lo.u32 {s}, {s}, {d};", .{ r_q0, r_q0, q_tile });

    // Loader: thread t copies chunk t % 8 of rows t / 8 + 16 i; the swizzle
    // `chunk ^ (row & 7)` is the same for every i.
    const r_smem = try b.reg(.b32);
    const r_lr = try b.reg(.b32);
    const r_lc = try b.reg(.b32);
    const r_lsw = try b.reg(.b32);
    try b.linef("mov.u32 {s}, smem;", .{r_smem});
    try b.linef("shr.u32 {s}, {s}, 3;", .{ r_lr, r_t });
    try b.linef("and.b32 {s}, {s}, 7;", .{ r_lc, r_t });
    try b.linef("and.b32 {s}, {s}, 7;", .{ r_tmp, r_lr });
    try b.linef("xor.b32 {s}, {s}, {s};", .{ r_tmp, r_tmp, r_lc });
    try b.linef("shl.b32 {s}, {s}, 4;", .{ r_tmp, r_tmp });
    try b.linef("mad.lo.u32 {s}, {s}, {d}, {s};", .{ r_lsw, r_lr, ROWB, r_tmp });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_lsw, r_lsw, r_smem });
    const Copy = struct {
        /// cp.async `rows` rows of 128 bytes, `stride` bytes apart from `src` (this
        /// thread's row and chunk already in), into shared `dst`.
        fn emit(bb: *ptx.Builder, dst: []const u8, dst_off: usize, src: []const u8, stride: []const u8, rows: usize, cur: []const u8) !void {
            try bb.linef("mov.b64 {s}, {s};", .{ cur, src });
            for (0..rows / 16) |i| {
                try bb.linef("cp.async.cg.shared.global [{s}+{d}], [{s}], 16;", .{ dst, dst_off + i * 16 * ROWB, cur });
                if (i + 1 < rows / 16) try bb.linef("add.s64 {s}, {s}, {s};", .{ cur, cur, stride });
            }
        }
    };
    const rd_cur = try b.reg(.b64);
    // Q: q8 + ((h*qrows + q0 + lr) * 128 + lc*16); stride 16 rows
    const rd_qp = try b.reg(.b64);
    const rd_kp = try b.reg(.b64);
    const rd_vp = try b.reg(.b64);
    const rd_st16 = try b.reg(.b64);
    const rd_vst = try b.reg(.b64);
    try b.linef("mov.u64 {s}, {d};", .{ rd_st16, 16 * ROWB });
    try b.linef("mad.lo.u32 {s}, {s}, {s}, {s};", .{ r_tmp, r_h, r_qrows, r_q0 });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_tmp, r_tmp, r_lr });
    try b.linef("mul.wide.u32 {s}, {s}, {d};", .{ rd_qp, r_tmp, ROWB });
    try b.linef("mul.wide.u32 {s}, {s}, 16;", .{ rd_tmp, r_lc });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_qp, rd_qp, rd_tmp });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_qp, rd_qp, rd_q8 });
    // K: k8 + ((h*krows + lr) * 128 + lc*16), advancing 128 rows per tile
    try b.linef("mad.lo.u32 {s}, {s}, {s}, {s};", .{ r_tmp, r_h, r_krows, r_lr });
    try b.linef("mul.wide.u32 {s}, {s}, {d};", .{ rd_kp, r_tmp, ROWB });
    try b.linef("mul.wide.u32 {s}, {s}, 16;", .{ rd_tmp, r_lc });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_kp, rd_kp, rd_tmp });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_kp, rd_kp, rd_k8 });
    // V: v8 + ((h*128 + lr) * krows + lc*16), advancing 128 bytes per tile
    try b.linef("mad.lo.u32 {s}, {s}, {d}, {s};", .{ r_tmp, r_h, hd, r_lr });
    try b.linef("mul.wide.u32 {s}, {s}, {s};", .{ rd_vp, r_tmp, r_krows });
    try b.linef("mul.wide.u32 {s}, {s}, 16;", .{ rd_tmp, r_lc });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_vp, rd_vp, rd_tmp });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_vp, rd_vp, rd_v8 });
    try b.linef("mul.wide.u32 {s}, {s}, 16;", .{ rd_vst, r_krows });

    try Copy.emit(&b, r_lsw, Q_OFF, rd_qp, rd_st16, q_tile, rd_cur);
    try Copy.emit(&b, r_lsw, K_OFF, rd_kp, rd_st16, key_tile, rd_cur);
    try b.line("cp.async.commit_group;");
    try Copy.emit(&b, r_lsw, V_OFF, rd_vp, rd_vst, hd, rd_cur);
    try b.line("cp.async.commit_group;");

    // ldmatrix lane bases (as in flash_attn, with 128-byte rows): Q at row
    // warp*16 + l7 + b3*8, chunk +b4; K and V at row l7 + b4*8, chunk +b3.
    const r_l7 = try b.reg(.b32);
    const r_b3 = try b.reg(.b32);
    const r_b4 = try b.reg(.b32);
    const r_yq = try b.reg(.b32);
    const r_yk = try b.reg(.b32);
    const r_qa = try b.reg(.b32);
    const r_ka = try b.reg(.b32);
    const r_va = try b.reg(.b32);
    try b.linef("and.b32 {s}, {s}, 7;", .{ r_l7, r_lane });
    try b.linef("bfe.u32 {s}, {s}, 3, 1;", .{ r_b3, r_lane });
    try b.linef("bfe.u32 {s}, {s}, 4, 1;", .{ r_b4, r_lane });
    try b.linef("xor.b32 {s}, {s}, {s};", .{ r_yq, r_b4, r_l7 });
    try b.linef("shl.b32 {s}, {s}, 4;", .{ r_yq, r_yq });
    try b.linef("xor.b32 {s}, {s}, {s};", .{ r_yk, r_b3, r_l7 });
    try b.linef("shl.b32 {s}, {s}, 4;", .{ r_yk, r_yk });
    try b.linef("mad.lo.u32 {s}, {s}, 8, {s};", .{ r_tmp, r_b3, r_l7 });
    try b.linef("mad.lo.u32 {s}, {s}, 16, {s};", .{ r_tmp2, r_warp, r_tmp });
    try b.linef("mad.lo.u32 {s}, {s}, {d}, {s};", .{ r_qa, r_tmp2, ROWB, r_smem });
    try b.linef("mad.lo.u32 {s}, {s}, 8, {s};", .{ r_tmp, r_b4, r_l7 });
    try b.linef("mad.lo.u32 {s}, {s}, {d}, {s};", .{ r_ka, r_tmp, ROWB, r_smem });
    try b.linef("add.u32 {s}, {s}, {d};", .{ r_va, r_ka, V_OFF });
    try b.linef("add.u32 {s}, {s}, {d};", .{ r_ka, r_ka, K_OFF });

    const qa = try b.regs(.b32, KT * 4);
    try b.line("cp.async.wait_group 1;");
    try b.line("bar.sync 0;");
    for (0..KT) |kk| {
        try b.linef("xor.b32 {s}, {s}, {d};", .{ r_tmp, r_yq, kk * 32 });
        try b.linef("add.u32 {s}, {s}, {s};", .{ r_tmp, r_tmp, r_qa });
        try b.linef("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{s}, {s}, {s}, {s}}}, [{s}];", .{ qa[kk * 4], qa[kk * 4 + 1], qa[kk * 4 + 2], qa[kk * 4 + 3], r_tmp });
    }

    // Rows A = q0 + warp*16 + gid and B = A + 8 share one Q scale.
    const r_rA = try b.reg(.b32);
    const f_qs = try b.reg(.f32);
    try b.linef("mad.lo.u32 {s}, {s}, 16, {s};", .{ r_rA, r_warp, r_gid });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_rA, r_rA, r_q0 });
    // qs index: h * qrows/4 + (rA/32)*8 + rA%8
    try b.linef("shr.u32 {s}, {s}, 5;", .{ r_tmp, r_rA });
    try b.linef("shl.b32 {s}, {s}, 3;", .{ r_tmp, r_tmp });
    try b.linef("and.b32 {s}, {s}, 7;", .{ r_tmp2, r_rA });
    try b.linef("add.u32 {s}, {s}, {s};", .{ r_tmp, r_tmp, r_tmp2 });
    try b.linef("shr.u32 {s}, {s}, 2;", .{ r_tmp2, r_qrows });
    try b.linef("mad.lo.u32 {s}, {s}, {s}, {s};", .{ r_tmp, r_h, r_tmp2, r_tmp });
    try b.linef("mul.wide.u32 {s}, {s}, 4;", .{ rd_tmp, r_tmp });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_tmp, rd_tmp, rd_qs });
    try b.linef("ld.global.f32 {s}, [{s}];", .{ f_qs, rd_tmp });
    // this lane's K scale pointer: ks + (h * krows/32 + t) * 4, +16 bytes per tile
    const rd_ksp = try b.reg(.b64);
    try b.linef("shr.u32 {s}, {s}, 5;", .{ r_tmp2, r_krows });
    try b.linef("mad.lo.u32 {s}, {s}, {s}, {s};", .{ r_tmp, r_h, r_tmp2, r_tf });
    try b.linef("mul.wide.u32 {s}, {s}, 4;", .{ rd_ksp, r_tmp });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_ksp, rd_ksp, rd_ks });

    const sacc = try b.regs(.b32, NS * 4);
    const oacc = try b.regs(.f32, ND * 4);
    const pw = try b.regs(.b32, KT * 4);
    const vb = try b.regs(.b32, 4);
    const pv = try b.regs(.b32, 8);
    const f_mA = try b.reg(.f32);
    const f_mB = try b.reg(.f32);
    const f_dA = try b.reg(.f32);
    const f_dB = try b.reg(.f32);
    const f_tA = try b.reg(.f32); // tile max, then the P rescale
    const f_tB = try b.reg(.f32);
    const f_sc = try b.reg(.f32);
    const f_x = try b.reg(.f32);
    const f_y = try b.reg(.f32);
    const f_oA = try b.reg(.f32);
    const f_oB = try b.reg(.f32);
    const f_nA = try b.reg(.f32); // -tile max
    const f_nB = try b.reg(.f32);
    const p_c = try b.reg(.pred);
    for (oacc) |r| try b.linef("mov.f32 {s}, 0f00000000;", .{r});
    try b.linef("mov.f32 {s}, 0fC7435000;", .{f_mA});
    try b.linef("mov.f32 {s}, 0fC7435000;", .{f_mB});
    try b.linef("mov.f32 {s}, 0f3F800000;", .{f_dA});
    try b.linef("mov.f32 {s}, 0f3F800000;", .{f_dB});

    const r_ti = try b.reg(.b32);
    const r_nt = try b.reg(.b32);
    const r_n0 = try b.reg(.b32);
    const p_more = try b.reg(.pred);
    try b.linef("shr.u32 {s}, {s}, 7;", .{ r_nt, r_krows });
    try b.linef("mov.u32 {s}, 0;", .{r_ti});
    try b.linef("mov.u32 {s}, 0;", .{r_n0});
    const f_p = try b.regs(.f32, 8);

    // fused tiles loop, then the last tile
    try b.linef("sub.u32 {s}, {s}, 1;", .{ r_tmp, r_nt });
    try b.linef("setp.eq.u32 {s}, {s}, 0;", .{ p_c, r_tmp });
    try b.linef("@{s} bra SG_LAST;", .{p_c});
    for ([_]bool{ false, true }) |last| {
        if (!last) try b.label("SG_LOOP") else try b.label("SG_LAST");
        try b.linef("add.u32 {s}, {s}, 1;", .{ r_tmp, r_ti });
        try b.linef("setp.lt.u32 {s}, {s}, {s};", .{ p_more, r_tmp, r_nt });
        try b.line("cp.async.wait_group 1;");
        try b.line("bar.sync 0;");

        // S = Q K^T, exact s32
        for (sacc) |r| try b.linef("mov.u32 {s}, 0;", .{r});
        for (0..KT) |kk| {
            try b.linef("xor.b32 {s}, {s}, {d};", .{ r_tmp, r_yk, kk * 32 });
            try b.linef("add.u32 {s}, {s}, {s};", .{ r_tmp, r_tmp, r_ka });
            for (0..NS / 2) |np| {
                try b.linef("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{s}, {s}, {s}, {s}}}, [{s}+{d}];", .{ vb[0], vb[1], vb[2], vb[3], r_tmp, np * 16 * ROWB });
                for (0..2) |half| {
                    const a = sacc[(np * 2 + half) * 4 ..][0..4];
                    try b.linef("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {{{s},{s},{s},{s}}}, {{{s},{s},{s},{s}}}, {{{s},{s}}}, {{{s},{s},{s},{s}}};", .{
                        a[0], a[1], a[2], a[3], qa[kk * 4], qa[kk * 4 + 1], qa[kk * 4 + 2], qa[kk * 4 + 3], vb[half * 2], vb[half * 2 + 1], a[0], a[1], a[2], a[3],
                    });
                }
            }
        }
        try b.line("bar.sync 0;");
        try b.linef("@!{s} bra {s};", .{ p_more, if (last) "SG_NOK1" else "SG_NOK0" });
        try b.linef("add.s64 {s}, {s}, {d};", .{ rd_kp, rd_kp, key_tile * ROWB });
        try Copy.emit(&b, r_lsw, K_OFF, rd_kp, rd_st16, key_tile, rd_cur);
        try b.label(if (last) "SG_NOK1" else "SG_NOK0");
        try b.line("cp.async.commit_group;");

        // sc = sm2 * (qs * ks[tile*4 + t])
        try b.linef("ld.global.f32 {s}, [{s}];", .{ f_sc, rd_ksp });
        try b.linef("mul.ftz.f32 {s}, {s}, {s};", .{ f_sc, f_qs, f_sc });
        try b.linef("mul.ftz.f32 {s}, {s}, {s};", .{ f_sc, f_sm2, f_sc });

        if (!last) {
            // integer lane max per row, then mt = fma(max, sc, -off)
            for ([_][]const u8{ f_tA, f_tB }, 0..) |f, row| {
                try b.linef("max.s32 {s}, {s}, {s};", .{ r_tmp, sacc[row * 2], sacc[row * 2 + 1] });
                for (1..NS) |j| {
                    try b.linef("max.s32 {s}, {s}, {s};", .{ r_tmp, r_tmp, sacc[j * 4 + row * 2] });
                    try b.linef("max.s32 {s}, {s}, {s};", .{ r_tmp, r_tmp, sacc[j * 4 + row * 2 + 1] });
                }
                try b.linef("cvt.rz.f32.s32 {s}, {s};", .{ f, r_tmp });
                try b.linef("fma.rn.ftz.f32 {s}, {s}, {s}, 0f{X:0>8};", .{ f, f, f_sc, off_neg });
            }
        } else {
            // scores to f32 in place (as bits), keys past lk at -1e30; the lane max
            // floored at -50000 in score units
            for (0..NS) |j| for (0..4) |e| {
                const s = sacc[j * 4 + e];
                try b.linef("cvt.rn.f32.s32 {s}, {s};", .{ f_x, s });
                // key = n0 + 8j + 2t + (e & 1)
                try b.linef("mad.lo.u32 {s}, {s}, 2, {s};", .{ r_tmp, r_tf, r_n0 });
                try b.linef("add.u32 {s}, {s}, {d};", .{ r_tmp, r_tmp, 8 * j + (e & 1) });
                try b.linef("setp.lt.u32 {s}, {s}, {s};", .{ p_c, r_tmp, r_lk });
                try b.linef("selp.f32 {s}, {s}, 0f{X:0>8}, {s};", .{ f_x, f_x, hex(-1e30), p_c });
                try b.linef("mov.b32 {s}, {s};", .{ s, f_x });
            };
            for ([_][]const u8{ f_tA, f_tB }, 0..) |f, row| {
                try b.linef("mov.f32 {s}, 0fC7435000;", .{f});
                for (0..NS) |j| for (0..2) |e| {
                    try b.linef("mov.b32 {s}, {s};", .{ f_x, sacc[j * 4 + row * 2 + e] });
                    try b.linef("max.ftz.f32 {s}, {s}, {s};", .{ f, f, f_x });
                };
                try b.linef("fma.rn.ftz.f32 {s}, {s}, {s}, 0f{X:0>8};", .{ f, f, f_sc, off_neg });
            }
        }
        // tile max across the row's four lanes
        for ([_]usize{ 1, 2 }) |d| for ([_][]const u8{ f_tA, f_tB }) |f| {
            try b.linef("mov.b32 {s}, {s};", .{ r_tmp, f });
            try b.linef("shfl.sync.bfly.b32 {s}, {s}, {d}, 0x1f, 0xffffffff;", .{ r_tmp, r_tmp, d });
            try b.linef("mov.b32 {s}, {s};", .{ f_x, r_tmp });
            try b.linef("max.ftz.f32 {s}, {s}, {s};", .{ f, f, f_x });
        };
        // s = ex2(-|m - tm|); o = m < tm ? s : 1; ts = tm < m ? s : 1; m = max
        for ([_][5][]const u8{ .{ f_mA, f_tA, f_oA, f_nA, f_dA }, .{ f_mB, f_tB, f_oB, f_nB, f_dB } }) |s| {
            try b.linef("sub.ftz.f32 {s}, {s}, {s};", .{ f_x, s[0], s[1] });
            try b.linef("abs.ftz.f32 {s}, {s};", .{ f_x, f_x });
            try b.linef("neg.ftz.f32 {s}, {s};", .{ f_x, f_x });
            try b.linef("ex2.approx.ftz.f32 {s}, {s};", .{ f_x, f_x });
            try b.linef("setp.lt.ftz.f32 {s}, {s}, {s};", .{ p_c, s[0], s[1] });
            try b.linef("selp.f32 {s}, {s}, 0f3F800000, {s};", .{ s[2], f_x, p_c });
            try b.linef("setp.lt.ftz.f32 {s}, {s}, {s};", .{ p_c, s[1], s[0] });
            try b.linef("neg.ftz.f32 {s}, {s};", .{ s[3], s[1] });
            try b.linef("max.ftz.f32 {s}, {s}, {s};", .{ s[0], s[0], s[1] });
            try b.linef("selp.f32 {s}, {s}, 0f3F800000, {s};", .{ s[1], f_x, p_c }); // tile scale
            try b.linef("mul.ftz.f32 {s}, {s}, {s};", .{ s[4], s[4], s[2] });
        }
        for (0..ND) |dt| for (0..4) |e|
            try b.linef("mul.ftz.f32 {s}, {s}, {s};", .{ oacc[dt * 4 + e], oacc[dt * 4 + e], if (e < 2) f_oA else f_oB });

        // P per 16-key block: u8 words in A-operand order, and the denominator
        for (0..NS / 2) |blk| for (0..2) |row| {
            const vals = [4][]const u8{ sacc[(2 * blk) * 4 + row * 2], sacc[(2 * blk) * 4 + row * 2 + 1], sacc[(2 * blk + 1) * 4 + row * 2], sacc[(2 * blk + 1) * 4 + row * 2 + 1] };
            const nm = if (row == 0) f_nA else f_nB;
            for (vals, 0..) |s, i| {
                if (last) {
                    try b.linef("mov.b32 {s}, {s};", .{ f_x, s });
                } else {
                    try b.linef("cvt.rn.f32.s32 {s}, {s};", .{ f_x, s });
                }
                try b.linef("fma.rn.ftz.f32 {s}, {s}, {s}, {s};", .{ f_x, f_x, f_sc, nm });
                try b.linef("ex2.approx.ftz.f32 {s}, {s};", .{ f_p[i], f_x });
                if (last) {
                    try b.linef("max.ftz.f32 {s}, {s}, 0f00000000;", .{ f_x, f_p[i] });
                    try b.linef("min.ftz.f32 {s}, {s}, 0f437F0000;", .{ f_x, f_x });
                } else {
                    try b.linef("min.ftz.f32 {s}, {s}, 0f437F0000;", .{ f_x, f_p[i] });
                }
                try b.linef("add.ftz.f32 {s}, {s}, 0f4B000000;", .{ f_x, f_x });
                try b.linef("mov.b32 {s}, {s};", .{ s, f_x });
            }
            // word = bytes [v0, v1, v2, v3]; A k-step blk/2, register row + 2*(blk%2)
            const w = pw[(blk / 2) * 4 + row + 2 * (blk % 2)];
            try b.linef("prmt.b32 {s}, {s}, {s}, 0x3340;", .{ w, vals[0], vals[1] });
            try b.linef("prmt.b32 {s}, {s}, {s}, 0x3340;", .{ r_tmp, vals[2], vals[3] });
            try b.linef("prmt.b32 {s}, {s}, {s}, 0x5410;", .{ w, w, r_tmp });
            const d = if (row == 0) f_dA else f_dB;
            const ts = if (row == 0) f_tA else f_tB;
            if (last) {
                try b.linef("add.ftz.f32 {s}, {s}, {s};", .{ f_x, f_p[0], f_p[1] });
                try b.linef("add.ftz.f32 {s}, {s}, {s};", .{ f_y, f_p[2], f_p[3] });
                try b.linef("add.ftz.f32 {s}, {s}, {s};", .{ f_x, f_x, f_y });
            } else {
                try b.linef("dp4a.u32.u32 {s}, {s}, 0x01010101, 0;", .{ r_tmp, w });
                try b.linef("cvt.rn.f32.u32 {s}, {s};", .{ f_x, r_tmp });
            }
            try b.linef("fma.rn.ftz.f32 {s}, {s}, {s}, {s};", .{ d, f_x, ts, d });
        };

        // O += (P V) * ts once V is in
        try b.line("cp.async.wait_group 1;");
        try b.line("bar.sync 0;");
        for (0..ND / 2) |np| {
            for (pv) |r| try b.linef("mov.u32 {s}, 0;", .{r});
            for (0..KT) |kk| {
                try b.linef("xor.b32 {s}, {s}, {d};", .{ r_tmp, r_yk, kk * 32 });
                try b.linef("add.u32 {s}, {s}, {s};", .{ r_tmp, r_tmp, r_va });
                try b.linef("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{s}, {s}, {s}, {s}}}, [{s}+{d}];", .{ vb[0], vb[1], vb[2], vb[3], r_tmp, np * 16 * ROWB });
                for (0..2) |half| {
                    const a = pv[half * 4 ..][0..4];
                    try b.linef("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {{{s},{s},{s},{s}}}, {{{s},{s},{s},{s}}}, {{{s},{s}}}, {{{s},{s},{s},{s}}};", .{
                        a[0], a[1], a[2], a[3], pw[kk * 4], pw[kk * 4 + 1], pw[kk * 4 + 2], pw[kk * 4 + 3], vb[half * 2], vb[half * 2 + 1], a[0], a[1], a[2], a[3],
                    });
                }
            }
            for (0..2) |half| for (0..4) |e| {
                const o = oacc[(np * 2 + half) * 4 + e];
                try b.linef("cvt.rn.f32.s32 {s}, {s};", .{ f_x, pv[half * 4 + e] });
                try b.linef("fma.rn.ftz.f32 {s}, {s}, {s}, {s};", .{ o, f_x, if (e < 2) f_tA else f_tB, o });
            };
        }
        try b.line("bar.sync 0;");
        try b.linef("@!{s} bra {s};", .{ p_more, if (last) "SG_NOV1" else "SG_NOV0" });
        try b.linef("add.s64 {s}, {s}, {d};", .{ rd_vp, rd_vp, key_tile });
        try Copy.emit(&b, r_lsw, V_OFF, rd_vp, rd_vst, hd, rd_cur);
        try b.label(if (last) "SG_NOV1" else "SG_NOV0");
        try b.line("cp.async.commit_group;");
        try b.linef("add.u32 {s}, {s}, {d};", .{ r_n0, r_n0, key_tile });
        try b.linef("add.s64 {s}, {s}, 16;", .{ rd_ksp, rd_ksp });
        try b.linef("add.u32 {s}, {s}, 1;", .{ r_ti, r_ti });
        if (!last) {
            // loop while the next tile is not the last
            try b.linef("add.u32 {s}, {s}, 1;", .{ r_tmp, r_ti });
            try b.linef("setp.lt.u32 {s}, {s}, {s};", .{ p_c, r_tmp, r_nt });
            try b.linef("@{s} bra SG_LOOP;", .{p_c});
        }
    }

    // d = (d0 + d1) + (d2 + d3) across the row's lanes; O = (O * rcp(d)) * vs
    for ([_][]const u8{ f_dA, f_dB }) |f| for ([_]usize{ 1, 2 }) |d| {
        try b.linef("mov.b32 {s}, {s};", .{ r_tmp, f });
        try b.linef("shfl.sync.bfly.b32 {s}, {s}, {d}, 0x1f, 0xffffffff;", .{ r_tmp, r_tmp, d });
        try b.linef("mov.b32 {s}, {s};", .{ f_x, r_tmp });
        try b.linef("add.ftz.f32 {s}, {s}, {s};", .{ f, f, f_x });
    };
    try b.linef("rcp.approx.ftz.f32 {s}, {s};", .{ f_dA, f_dA });
    try b.linef("rcp.approx.ftz.f32 {s}, {s};", .{ f_dB, f_dB });
    // vs + (h*128 + 2t) * 4, o + ((rA * os) + h*128 + 2t) * oel
    const rd_vsp = try b.reg(.b64);
    const rd_oA = try b.reg(.b64);
    const rd_oB = try b.reg(.b64);
    const p_A = try b.reg(.pred);
    const p_B = try b.reg(.pred);
    try b.linef("mad.lo.u32 {s}, {s}, {d}, 0;", .{ r_tmp, r_h, hd });
    try b.linef("mad.lo.u32 {s}, {s}, 2, {s};", .{ r_tmp, r_tf, r_tmp });
    try b.linef("mul.wide.u32 {s}, {s}, 4;", .{ rd_vsp, r_tmp });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_vsp, rd_vsp, rd_vs });
    try b.linef("mad.lo.u32 {s}, {s}, {s}, {s};", .{ r_tmp2, r_rA, r_os, r_tmp });
    try b.linef("mul.wide.u32 {s}, {s}, {d};", .{ rd_oA, r_tmp2, oel });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_oA, rd_oA, rd_o });
    try b.linef("mul.wide.u32 {s}, {s}, {d};", .{ rd_tmp, r_os, 8 * oel });
    try b.linef("add.s64 {s}, {s}, {s};", .{ rd_oB, rd_oA, rd_tmp });
    try b.linef("setp.lt.u32 {s}, {s}, {s};", .{ p_A, r_rA, r_lq });
    try b.linef("add.u32 {s}, {s}, 8;", .{ r_tmp, r_rA });
    try b.linef("setp.lt.u32 {s}, {s}, {s};", .{ p_B, r_tmp, r_lq });
    const f_v0 = try b.reg(.f32);
    const f_v1 = try b.reg(.f32);
    const h_lo = try b.reg(.b16);
    const h_hi = try b.reg(.b16);
    for (0..ND) |dt| {
        try b.linef("ld.global.v2.f32 {{{s}, {s}}}, [{s}+{d}];", .{ f_v0, f_v1, rd_vsp, dt * 8 * 4 });
        for (0..2) |row| {
            const o = oacc[dt * 4 + row * 2 ..][0..2];
            const inv = if (row == 0) f_dA else f_dB;
            const pr = if (row == 0) p_A else p_B;
            const base = if (row == 0) rd_oA else rd_oB;
            try b.linef("mul.ftz.f32 {s}, {s}, {s}; mul.ftz.f32 {s}, {s}, {s};", .{ o[0], o[0], inv, o[1], o[1], inv });
            try b.linef("mul.ftz.f32 {s}, {s}, {s}; mul.ftz.f32 {s}, {s}, {s};", .{ o[0], o[0], f_v0, o[1], o[1], f_v1 });
            try b.linef("cvt.rn.{s}x2.f32 {s}, {s}, {s};", .{ rname, r_tmp, o[1], o[0] });
            if (out_dt != .f32) {
                try b.linef("@{s} st.global.b32 [{s}+{d}], {s};", .{ pr, base, dt * 8 * 2, r_tmp });
            } else {
                if (round == .bf16) {
                    try b.linef("shl.b32 {s}, {s}, 16;", .{ r_tmp2, r_tmp });
                    try b.linef("mov.b32 {s}, {s};", .{ o[0], r_tmp2 });
                    try b.linef("and.b32 {s}, {s}, 0xffff0000;", .{ r_tmp2, r_tmp });
                    try b.linef("mov.b32 {s}, {s};", .{ o[1], r_tmp2 });
                } else {
                    try b.linef("mov.b32 {{{s}, {s}}}, {s};", .{ h_lo, h_hi, r_tmp });
                    try b.linef("cvt.f32.f16 {s}, {s};", .{ o[0], h_lo });
                    try b.linef("cvt.f32.f16 {s}, {s};", .{ o[1], h_hi });
                }
                try b.linef("@{s} st.global.v2.f32 [{s}+{d}], {{{s}, {s}}};", .{ pr, base, dt * 8 * 4, o[0], o[1] });
            }
        }
    }

    const shared_decl = try alloc.print(".shared .align 128 .b8 smem[{d}];", .{SH});
    defer alloc.free(shared_decl);
    return b.build("sage_attn",
        \\    .param .u64 p_q8,
        \\    .param .u64 p_k8,
        \\    .param .u64 p_v8,
        \\    .param .u64 p_qs,
        \\    .param .u64 p_ks,
        \\    .param .u64 p_vs,
        \\    .param .u64 p_o,
        \\    .param .u32 p_lq,
        \\    .param .u32 p_lk,
        \\    .param .u32 p_qrows,
        \\    .param .u32 p_krows,
        \\    .param .u32 p_os,
        \\    .param .f32 p_sm2
    , shared_decl);
}
