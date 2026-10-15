//! The MiniMax H3 trunk and output heads, written once for every GPU backend over an
//! adapter (`minimax_h3_cuda.Dev`, `minimax_h3_gpu.Dev`). `body` runs between the
//! adapter's upload of a `minimax_h3.DeviceStep` (packed sequence into `ws.x_d`, the
//! two modulation tables, the label tables) and its download of the heads' rows,
//! video from `ws.q_d` and audio from `ws.k_d`.
//!
//! Attention runs in bands of query rows. Pass one writes K and V for the whole
//! sequence a band at a time (they must see the block's input, which no residual add
//! has touched yet); pass two takes Q, the attention and the out-projection a band at
//! a time and folds each into the residual. Only Q, the attention output and the
//! projection are band-sized, which is what takes them off the sequence length. The
//! MLP is banded the same way. A band can straddle segments and each segment
//! modulates with its own row, so a band applies the INTERSECTION of itself with each.
//!
//! Adapter: `ws` (`x_d`, `t1_d`, `q_d`, `attn_d`, `ao_d`, `gate_d`, `up_d`, `mod_d`,
//! `fmod_d`), `labelIdx` (a target segment's per-row label buffer, or null), the
//! band ops `rmsMod` / `gatedAdd` (row offsets as arguments), the linears, `qkNorm`,
//! `rope` (positions from `pos0`), `kvAt` / `kvStore` (where a K or V band lands, and
//! storing it there in the attention's own width) and `kvReady` once both are whole.
//!
//! The residual `x_d` and the norm output `t1_d` are stored at the denoiser's activation
//! dtype; both deltas (`ao_d`, which also takes fc2's output) and the MLP planes stay
//! f32, since fc2's output alone reaches ~7e5 and the MLP product ~7e4.

const std = @import("std");
const minimax_h3 = @import("minimax_h3.zig");

const DiT = minimax_h3.DiT;
const Weight = @import("tp_ops").matmul.Weight;
const Act = @import("tp_gpu").context.Dt;

/// The dtype the residual is stored at when `want` is asked for, on either backend.
pub fn streamDtFor(want: Act) Act {
    return @import("lin.zig").streamAct(Act, want);
}

/// The exact power of two an f16 residual is stored divided by: it reaches ~3e6. Every
/// reader is an RMSNorm, which takes `eps / div^2` and so computes the same thing.
pub fn streamDiv(dt: Act) f32 {
    return if (dt == .f16) 256 else 1;
}

/// The packed sequence as `dt` bytes for an upload into a 16-bit `x_d`, divided by
/// `streamDiv(dt)`. Caller frees.
pub fn narrow(gpa: std.mem.Allocator, h: []const f32, dt: Act) ![]u8 {
    const core = @import("tp_core");
    const tmp = try gpa.alloc(f32, h.len);
    defer gpa.free(tmp);
    const inv = 1 / streamDiv(dt);
    for (tmp, h) |*o, v| o.* = v * inv;
    const out = try gpa.alloc(u8, h.len * 2);
    switch (dt) {
        .f16 => core.dtype.f32ToF16Row(tmp, out),
        .bf16 => core.dtype.f32ToBf16Row(tmp, out),
        .f32 => unreachable,
    }
    return out;
}

/// Query rows the attention is processed in. Every band attends to the WHOLE key
/// sequence, so this bounds Q, the attention output and the out-projection: 2.62 GB
/// each at 15 seconds of 768x1152, against 117 MB here.
pub const attn_band: usize = 4096;

/// Rows the MLP is processed in. The gate/up intermediates are `[tile][ffn]`, so this
/// bounds the biggest activation: at the default render an untiled pair is 2.2 GB each.
pub const mlp_tile: usize = 2048;

/// Rows every GEMM output must be sized for: the int8 routes write the row count
/// ROUNDED UP to 128.
pub fn padRows(rows: usize) usize {
    return std.mem.alignForward(usize, rows, 128);
}

/// Element offset of one modulation vector in `mod_d`: `[block][t_row][tag][slot]
/// [hidden]`, the slots `(premul_msa, shift_msa, gate_msa, premul_mlp, shift_mlp,
/// gate_mlp)`. PREMUL, not scale: the norm weight and the `1 +` are folded in on the
/// host (`buildModTable`), because the kernel applies neither.
pub fn modOff(cfg: minimax_h3.Config, n_labels: usize, block: usize, t_row: usize, tag: minimax_h3.Tag, slot: usize) usize {
    std.debug.assert(t_row < n_labels and slot < 6);
    return (((block * n_labels + t_row) * 3 + @intFromEnum(tag)) * 6 + slot) * cfg.hidden;
}

const Ctx = struct {
    dit: *const DiT,
    layout: *const minimax_h3.PackedLayout,
    ts: *const minimax_h3.Timesteps,
    n_labels: usize,
    block: usize,
    /// `streamDiv` of the residual.
    div: f32,
};

pub fn body(d: anytype, dit: *const DiT, layout: *const minimax_h3.PackedLayout, step: *const minimax_h3.DeviceStep, cancel: ?*std.atomic.Value(bool)) !void {
    const cfg = dit.cfg;
    const seq = layout.seq_len;
    const h = cfg.hidden;
    const ws = d.ws;
    const div = streamDiv(ws.x_d.dt);

    for (dit.blocks, 0..) |*b, bi| {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        const cx: Ctx = .{ .dit = dit, .layout = layout, .ts = &step.ts, .n_labels = step.ts.labels().len, .block = bi, .div = div };

        // Pass one: K and V over the whole sequence. The fused qkv is split by rows
        // (`DiT.device_lins`), so these are two GEMMs into separate planes.
        var k0: usize = 0;
        while (k0 < seq) : (k0 += attn_band) {
            const n = @min(attn_band, seq - k0);
            try modBand(d, cx, k0, n, 0, 1);
            const wk = dit.deviceLin(bi, .k);
            const wv = dit.deviceLin(bi, .v);
            const g = try d.prep(ws.t1_d, n, h, &.{ wk, wv });
            const k_at = try d.kvAt(.k, k0);
            try d.gemm(g, k_at, wk);
            try d.qkNorm(k_at, b.attn.k_norm, n);
            try d.rope(k_at, n, k0);
            try d.kvStore(.k, k_at, k0, n);
            const v_at = try d.kvAt(.v, k0);
            try d.gemm(g, v_at, wv);
            try d.kvStore(.v, v_at, k0, n);
        }
        try d.kvReady(seq);

        // Pass two. Re-normalized rather than kept from pass one, which would need a
        // sequence-sized `t1_d` again; the residual adds only touch rows this pass
        // has already read, so a later band still normalizes the block's input.
        // ⚠️ The rope reads GLOBAL positions (`a0`) while the data is band-relative.
        var a0: usize = 0;
        while (a0 < seq) : (a0 += attn_band) {
            const n = @min(attn_band, seq - a0);
            try modBand(d, cx, a0, n, 0, 1);
            try d.linear(ws.q_d, ws.t1_d, n, dit.deviceLin(bi, .q));
            try d.qkNorm(ws.q_d, b.attn.q_norm, n);
            try d.rope(ws.q_d, n, a0);
            try d.attention(n, seq);
            try d.linear(ws.ao_d, ws.attn_d, n, dit.deviceLin(bi, .out));
            try gatedBand(d, cx, ws.ao_d, a0, n, 2);
        }

        var c0: usize = 0;
        while (c0 < seq) : (c0 += mlp_tile) {
            const n = @min(mlp_tile, seq - c0);
            try modBand(d, cx, c0, n, 3, 4);
            // fc1 is the fused swiglu gate+value; rows [0, ffn) are the GATE.
            const wg = dit.deviceLin(bi, .gate);
            const wu = dit.deviceLin(bi, .up);
            const g = try d.prep(ws.t1_d, n, h, &.{ wg, wu });
            try d.gemm(g, ws.gate_d, wg);
            try d.gemm(g, ws.up_d, wu);
            try d.siluMul(ws.gate_d, ws.up_d, n * cfg.ffn);
            // Into `ao_d`, idle here and at least a tile wide: the output is f32.
            try d.linear(ws.ao_d, ws.gate_d, n, dit.deviceLin(bi, .fc2));
            try gatedBand(d, cx, ws.ao_d, c0, n, 5);
        }
    }

    // The heads stay on the device although they are 96 and 32 rows: reaching them
    // from the host means downloading the whole trunk, 925 MB at 15 seconds of
    // 512x768 every step, against 16 MB of results. Their weights are f32, which
    // takes the f32 GEMM and its lack of a 128-row floor.
    std.debug.assert(dit.final.video_out.dtype == .f32 and dit.final.audio_out.dtype == .f32);
    const final_eps = cfg.final_norm_eps / (div * div);
    try finalHead(d, ws.q_d, layout.segmentOf(.video).?, step.ts.rowFor(.video), d.labelIdx(.video), h, dit.final.video_out, dit.final.video_bias, final_eps);
    try finalHead(d, ws.k_d, layout.segmentOf(.audio).?, step.ts.rowFor(.audio), d.labelIdx(.audio), h, dit.final.audio_out, dit.final.audio_bias, final_eps);
}

/// Normalize and modulate rows `[b0, b0 + n)` of `x_d` into row 0 of `t1_d`, each
/// segment's part with its own modulation row.
fn modBand(d: anytype, cx: Ctx, b0: usize, n: usize, premul_slot: usize, shift_slot: usize) !void {
    const ws = d.ws;
    const cfg = cx.dit.cfg;
    const stride = minimax_h3.modality_count * 6 * cfg.hidden;
    for (cx.layout.segments) |sg| {
        const lo = @max(sg.start, b0);
        const hi = @min(sg.stop, b0 + n);
        if (lo >= hi) continue;
        const tag = sg.kind.tag();
        // With an index buffer the scalar offsets are the LABEL-ZERO ones and the
        // kernel adds `idx[row] * stride`; a band starts mid-segment, so does its index.
        const idx = d.labelIdx(sg.kind);
        const base = if (idx == null) cx.ts.rowFor(sg.kind) else 0;
        try d.rmsMod(ws.x_d, lo, ws.t1_d, lo - b0, ws.mod_d, hi - lo, modOff(cfg, cx.n_labels, cx.block, base, tag, premul_slot), modOff(cfg, cx.n_labels, cx.block, base, tag, shift_slot), cfg.norm_eps / (cx.div * cx.div), idx, lo - sg.start, stride);
    }
}

/// `x_d[b0 + r] += gate * src[r]` over `n` rows, each segment's part with its own gate.
fn gatedBand(d: anytype, cx: Ctx, src: anytype, b0: usize, n: usize, slot: usize) !void {
    const ws = d.ws;
    const cfg = cx.dit.cfg;
    const h = cfg.hidden;
    const stride = minimax_h3.modality_count * 6 * h;
    for (cx.layout.segments) |sg| {
        const lo = @max(sg.start, b0);
        const hi = @min(sg.stop, b0 + n);
        if (lo >= hi) continue;
        const idx = d.labelIdx(sg.kind);
        const base = if (idx == null) cx.ts.rowFor(sg.kind) else 0;
        try d.gatedAdd(ws.x_d, lo, src, lo - b0, ws.mod_d, (hi - lo) * h, modOff(cfg, cx.n_labels, cx.block, base, sg.kind.tag(), slot), idx, lo - sg.start, stride, 1 / cx.div);
    }
}

/// One output head: weightless norm and modulate from the folded final table, then
/// the f32 projection, banded because `t1_d` is.
///
/// ⚠️ **The head modulates PER ROW when a denoise mask relabelled the segment**: with
/// an index buffer the offsets are the LABEL-ZERO ones, exactly as in the trunk.
/// Passing the segment's own label instead renders a masked clip differently
/// depending on how the mask happened to be expressed.
fn finalHead(d: anytype, out: anytype, seg: minimax_h3.Segment, t_row: usize, idx: anytype, h: usize, w: Weight, bias: []const f32, norm_eps: f32) !void {
    const ws = d.ws;
    const n = seg.len();
    const base = if (idx == null) t_row else 0;
    var r0: usize = 0;
    while (r0 < n) : (r0 += attn_band) {
        const rn = @min(attn_band, n - r0);
        try d.rmsMod(ws.x_d, seg.start + r0, ws.t1_d, 0, ws.fmod_d, rn, base * 2 * h, base * 2 * h + h, norm_eps, idx, r0, 2 * h);
        try d.linearAt(out, r0 * w.rows, ws.t1_d, rn, w, bias);
    }
}
