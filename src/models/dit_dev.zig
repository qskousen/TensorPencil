//! The krea2 DiT device forward, written once for every GPU backend over an adapter
//! (`dit_cuda.Dev`, `dit_gpu.Dev`). `body` runs between the adapter's upload of the
//! modulation tables, the text rows and the patches, and its download of the image rows.
//!
//! The adapter holds `bufs` (`x` the residual, `t1` the norm and projection scratch,
//! `attn` the attention output, `mv` / `fin` the modulation tables) and supplies the ops
//! below. Per block it returns a plan (`AttnPlan`, `MlpPlan`): which projections share one
//! staged f16 activation (`Stage`, a fused norm or gate kernel writing the fp8 GEMMs'
//! operand), and which buffer, at which dtype, each output lands in. That is where a
//! backend's fast paths live, so each plan reproduces its backend's own branch
//! conditions; the op order here is what keeps an output bit-identical when it moves.
//! Attention proper (q/k norm, rope, scores, softmax, P@V) is one adapter op. An adapter
//! whose type declares `can_stage = false` never stages, and needs no staging ops.

const std = @import("std");
const dit = @import("dit.zig");
const Weight = @import("tp_ops").matmul.Weight;

const F = dit.features;

pub fn Stage(comptime Buf: type) type {
    return struct { buf: Buf, scale: f32 };
}

pub fn AttnPlan(comptime Buf: type) type {
    return struct {
        /// q/k/v/gate read one staged activation; null runs the norm into `t1`.
        stage: ?Stage(Buf),
        q: Buf,
        k: Buf,
        v: Buf,
        g: Buf,
        /// `attn * sigmoid(g)` staged for `wo`; null multiplies in place.
        wo_stage: ?Stage(Buf),
        wo_out: Buf,
    };
}

pub fn MlpPlan(comptime Buf: type) type {
    return struct {
        /// Rows per pass: the gate/up intermediates are `[tile][mlp_dim]`.
        tile: usize,
        stage: ?Stage(Buf),
        mg: Buf,
        mu: Buf,
        down_stage: ?Stage(Buf),
        down_out: Buf,
    };
}

pub fn body(d: anytype, model: *const dit.DiT, seq_txt: usize, n_img: usize, cancel: ?*std.atomic.Value(bool)) !void {
    const b = d.bufs;
    const seq = seq_txt + n_img;
    const can_stage = @TypeOf(d).can_stage;

    try d.patchEmbed(seq_txt, n_img);
    for (model.blocks, 0..) |*blk, bi| {
        // Between blocks, so a stop lands mid-step; the adapter aborts its batch.
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        try d.blockBegin(bi);
        const mb = bi * 6 * F;

        const att_ws = [_]Weight{ blk.attn.wq, blk.attn.wk, blk.attn.wv, blk.attn.gate };
        const ap = try d.attnPlan(blk, seq);
        const ag = if (can_stage and ap.stage != null) blk_: {
            const st = ap.stage.?;
            try d.rmsMod16(b.x, st, b.mv, mb + 0 * F, mb + 1 * F, seq);
            break :blk_ d.prepStaged(st, seq, F, &att_ws);
        } else blk_: {
            try d.rmsMod(b.x, b.t1, b.mv, mb + 0 * F, mb + 1 * F, seq);
            break :blk_ try d.prep(b.t1, seq, F, &att_ws);
        };
        try d.gemms(ag, &.{ ap.q, ap.k, ap.v, ap.g }, &att_ws);
        try d.attention(blk, ap, seq);
        if (can_stage and ap.wo_stage != null) {
            const st = ap.wo_stage.?;
            try d.gateStage(b.attn, ap.g, st, seq);
            try d.gemms(d.prepStaged(st, seq, blk.attn.wo.cols, &.{blk.attn.wo}), &.{ap.wo_out}, &.{blk.attn.wo});
        } else {
            try d.gateMul(b.attn, ap.g, seq);
            try d.linear(ap.wo_out, b.attn, seq, blk.attn.wo);
        }
        try d.gatedAdd(b.x, ap.wo_out, b.mv, mb + 2 * F, seq);

        const mlp_ws = [_]Weight{ blk.mlp.gate, blk.mlp.up };
        const mp = try d.mlpPlan(blk, seq);
        var c0: usize = 0;
        while (c0 < seq) : (c0 += mp.tile) {
            const tile = @min(mp.tile, seq - c0);
            const xo = try d.rowsAt(b.x, c0, F);
            const mg = if (can_stage and mp.stage != null) blk_: {
                const st = mp.stage.?;
                try d.rmsMod16(xo, st, b.mv, mb + 3 * F, mb + 4 * F, tile);
                break :blk_ d.prepStaged(st, tile, F, &mlp_ws);
            } else blk_: {
                try d.rmsMod(xo, b.t1, b.mv, mb + 3 * F, mb + 4 * F, tile);
                break :blk_ try d.prep(b.t1, tile, F, &mlp_ws);
            };
            try d.gemms(mg, &.{ mp.mg, mp.mu }, &mlp_ws);
            if (can_stage and mp.down_stage != null) {
                const st = mp.down_stage.?;
                try d.siluStage(mp.mg, mp.mu, st, tile);
                try d.gemms(d.prepStaged(st, tile, blk.mlp.down.cols, &.{blk.mlp.down}), &.{mp.down_out}, &.{blk.mlp.down});
            } else {
                try d.siluMul(mp.mg, mp.mu, tile);
                try d.linear(mp.down_out, mp.mg, tile, blk.mlp.down);
            }
            try d.gatedAdd(xo, mp.down_out, b.mv, mb + 5 * F, tile);
        }

        if (model.act_steer) |st| {
            if (st.capture != null) try d.steerCapture(st, seq, seq_txt, bi);
            // Image rows only: the text rows are the conditioning the blocks read.
            const sc = st.scale * if (st.curve_at.len > bi) st.curve_at[bi] else 1;
            if (sc != 0) try d.steer(st, seq, seq_txt, bi, sc);
        }
    }

    // The norm runs over every row; the text rows' share is negligible next to
    // downloading the hidden image rows for a host finalize.
    try d.rmsMod(b.x, b.t1, b.fin, 0, F, seq);
    try d.finalLinear(seq_txt, n_img);
}
