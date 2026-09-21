//! Vulkan MiniMax H3 trunk forward, the device twin of `minimax_h3.forward` and
//! the sibling of `minimax_h3_cuda`, whose stage order it follows exactly so the
//! two can be diffed.
//!
//! The host/device split is that arm's: the 50-block trunk runs here and the
//! host-cheap paths (patch projections, the adaLN projection, the token refiner,
//! the output heads) stay on the CPU. Read its header for the three things that
//! make H3's block different from a DiT's.
//!
//! Two things differ here, both forced by the backend rather than by the model:
//!
//! 1. **A row range is a kernel argument, not an offset view.** A `DeviceBuffer`
//!    is an opaque handle with nowhere to put a byte offset, so where the CUDA arm
//!    hands `rms_mod` and `gated_add` a pointer into the middle of `x`, this passes
//!    the row offset in the push constants. Both arms now drive the same launch;
//!    the CUDA one gave up `offsetBuf` for these two when the offsets were added.
//! 2. **No LoRA sidecar.** `lora_cuda` has no Vulkan twin, so `supported` refuses
//!    a DiT carrying one rather than running the base GEMMs alone, which is a
//!    different model with nothing to say so.
//!
//! Numerics match `minimax_h3.forward` up to int8 quantization and the softmax
//! decomposition, the same regime every other Vulkan arm runs in.
//! `minimax-h3-vk-test` checks it against the CPU forward on real weights.

const std = @import("std");
const minimax_h3 = @import("minimax_h3.zig");
const gpu = @import("tp_gpu").context;
const ops = @import("tp_ops");

const DiT = minimax_h3.DiT;
const Context = gpu.Context;
const Buf = gpu.DeviceBuffer;
const Weight = ops.matmul.Weight;

/// An unallocated handle; a Vulkan `DeviceBuffer` has no empty literal.
const nullBuf: Buf = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 };

/// Force the reference attention path (`attn_full`) instead of the tensor-core
/// pipeline, for an A/B on one render.
pub var force_naive_attn: bool = false;

/// A contiguous row range of a row-major weight, as its own `Weight`. Same split
/// `minimax_h3_cuda` does for the fused qkv and fc1: rows are a byte range and the
/// per-row scales slice with them.
fn rowSlice(w: Weight, from: usize, n: usize) Weight {
    std.debug.assert(from + n <= w.rows);
    const stride = w.dtype.storageBytes(w.cols);
    var out = w;
    out.bytes = w.bytes[from * stride ..][0 .. n * stride];
    out.rows = n;
    if (w.row_scale) |rs| out.row_scale = rs[from..][0..n];
    return out;
}

/// Whether this checkpoint's trunk can run here. int8 convrot only, like the CUDA
/// arm, and refused by name rather than met as a bad access several blocks deep.
pub fn supported(dit: *const DiT) bool {
    for (dit.blocks) |b| {
        inline for (.{ b.attn.qkv, b.attn.out, b.mlp.fc1, b.mlp.fc2 }) |l| {
            const w = l.w;
            if (w.dtype != .i8) return false;
            if (w.row_scale == null) return false;
            if (w.rows % 128 != 0) return false;
            // No Vulkan sidecar: a trunk that would silently drop one is refused.
            if (l.lora.len != 0) return false;
        }
    }
    const cfg = dit.cfg;
    if ((cfg.n_heads * cfg.head_dim) % 128 != 0) return false;
    if (cfg.ffn % 128 != 0) return false;
    return true;
}

/// Per-render device state: everything constant across sampling steps.
pub const Session = struct {
    seq: usize,
    /// `[seq * pairs]` cos then `[seq * pairs]` sin, f32. `sinOff` is the split.
    freqs_d: Buf = nullBuf,
    pairs: usize,

    pub fn init(ctx: *Context, gpa: std.mem.Allocator, dit: *const DiT, layout: *const minimax_h3.PackedLayout) !Session {
        const cfg = dit.cfg;
        const pairs = cfg.ropePairs();
        var s: Session = .{ .seq = layout.seq_len, .pairs = pairs };
        errdefer s.deinit(ctx);

        var freqs = try minimax_h3.ropeFreqs(gpa, layout.pos, dit.rope_inv_freq);
        defer freqs.deinit(gpa);
        const host = try gpa.alloc(f32, 2 * layout.seq_len * pairs);
        defer gpa.free(host);
        @memcpy(host[0 .. layout.seq_len * pairs], freqs.cos);
        @memcpy(host[layout.seq_len * pairs ..], freqs.sin);
        s.freqs_d = try ctx.tensorCreate(host.len * 4);
        try ctx.tensorUpload(s.freqs_d, std.mem.sliceAsBytes(host));
        return s;
    }

    pub fn sinOff(self: Session) usize {
        return self.seq * self.pairs;
    }

    pub fn deinit(self: *Session, ctx: *Context) void {
        ctx.tensorDestroy(&self.freqs_d);
    }
};

/// Per-shape device scratch. Sized exactly like the CUDA arm's, including the
/// 128-row padding every int8 GEMM output needs.
pub const Workspace = struct {
    x_d: Buf = nullBuf,
    t1_d: Buf = nullBuf,
    q_d: Buf = nullBuf,
    k_d: Buf = nullBuf,
    v_d: Buf = nullBuf,
    attn_d: Buf = nullBuf,
    gate_d: Buf = nullBuf,
    up_d: Buf = nullBuf,
    mod_d: Buf = nullBuf,
    vmask_d: Buf = nullBuf,
    amask_d: Buf = nullBuf,
    /// Tensor-core attention scratch: half-precision q/k/v planes, the scores
    /// plane and the softmax partials.
    qh_d: Buf = nullBuf,
    kh_d: Buf = nullBuf,
    v16_d: Buf = nullBuf,
    s_d: Buf = nullBuf,
    part_d: Buf = nullBuf,
    md_d: Buf = nullBuf,

    pub const mlp_tile: usize = 2048;

    /// `opI8Gemm` writes the activation row count rounded UP to 128, so every
    /// GEMM output is sized for that and not for the row count itself.
    pub fn padRows(rows: usize) usize {
        return std.mem.alignForward(usize, rows, 128);
    }

    /// Chunks the softmax is decomposed into along the key axis.
    pub const nchunks: usize = 8;

    pub fn init(ctx: *Context, dit: *const DiT, seq: usize) !Workspace {
        const cfg = dit.cfg;
        const inner = cfg.n_heads * cfg.head_dim;
        const mpad = padRows(seq);
        var ws: Workspace = .{};
        errdefer ws.deinit(ctx);
        ws.x_d = try ctx.tensorCreate(mpad * cfg.hidden * 4);
        ws.t1_d = try ctx.tensorCreate(mpad * cfg.hidden * 4);
        ws.q_d = try ctx.tensorCreate(mpad * inner * 4);
        ws.k_d = try ctx.tensorCreate(mpad * inner * 4);
        ws.v_d = try ctx.tensorCreate(mpad * inner * 4);
        ws.attn_d = try ctx.tensorCreate(mpad * inner * 4);
        const tile = padRows(@min(mlp_tile, seq));
        ws.gate_d = try ctx.tensorCreate(tile * cfg.ffn * 4);
        ws.up_d = try ctx.tensorCreate(tile * cfg.ffn * 4);
        ws.mod_d = try ctx.tensorCreate(dit.blocks.len * minimax_h3.Timesteps.max_labels * 3 * 6 * cfg.hidden * 4);
        ws.vmask_d = try ctx.tensorCreate(seq * 4);
        ws.amask_d = try ctx.tensorCreate(seq * 4);

        // The attention planes are padded on BOTH axes: the scores kernel tiles
        // 128x128 and reads whole tiles.
        ws.qh_d = try ctx.tensorCreate(mpad * inner * 2);
        ws.kh_d = try ctx.tensorCreate(mpad * inner * 2);
        ws.v16_d = try ctx.tensorCreate(mpad * inner * 2);
        // One head's scores at a time keeps this bounded; `headsPerBatch` below
        // takes as many as fit.
        ws.s_d = try ctx.tensorCreate(mpad * mpad * 2);
        ws.part_d = try ctx.tensorCreate(seq * nchunks * 2 * 4);
        ws.md_d = try ctx.tensorCreate(mpad * 2 * 4);
        return ws;
    }

    pub fn deinit(self: *Workspace, ctx: *Context) void {
        inline for (@typeInfo(Workspace).@"struct".fields) |f| {
            if (f.type == Buf) ctx.tensorDestroy(&@field(self, f.name));
        }
    }
};

/// Where block `block`'s modulation for (timestep row, modality) sits in `mod_d`.
/// Identical to `minimax_h3_cuda.modOff`; the host builds one table for both arms.
fn modOff(cfg: minimax_h3.Config, n_labels: usize, block: usize, t_row: usize, tag: minimax_h3.Tag, slot: usize) usize {
    std.debug.assert(t_row < n_labels and slot < 6);
    return (((block * n_labels + t_row) * 3 + @intFromEnum(tag)) * 6 + slot) * cfg.hidden;
}

fn linPrep(ctx: *Context, x: Buf, m: usize, cols: usize) !void {
    try ctx.opI8Prep(x, m, cols);
}

fn linGemm(ctx: *Context, y: Buf, w: Weight) !void {
    std.debug.assert(w.dtype == .i8);
    try ctx.opI8Gemm(y, w.bytes, w.row_scale.?, w.rows, false);
}

/// A weight vector as a small device buffer, for the per-head q/k norms.
fn normBuf(ctx: *Context, w: []const f32) !Buf {
    return .{ .buf = try ctx.smallBuffer(std.mem.sliceAsBytes(w)), .mem = .null_handle, .size = w.len * 4 };
}

pub fn forward(
    dit: *const DiT,
    ctx: *Context,
    sess: *const Session,
    ws: *Workspace,
    io: std.Io,
    gpa: std.mem.Allocator,
    layout: *const minimax_h3.PackedLayout,
    out_video: []f32,
    out_audio: []f32,
    in: minimax_h3.Inputs,
    cancel: ?*std.atomic.Value(bool),
) !void {
    const cfg = dit.cfg;
    const seq = layout.seq_len;
    const h = cfg.hidden;
    const inner = cfg.n_heads * cfg.head_dim;
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));
    const eps = cfg.norm_eps;
    std.debug.assert(sess.seq == seq);

    // --- host: embed both streams into the packed sequence -----------------
    const packed_h = try gpa.alloc(f32, seq * h);
    defer gpa.free(packed_h);
    var ts = try minimax_h3.embedPacked(dit, io, gpa, packed_h, layout, in);
    defer ts.deinit(gpa);
    const labels = ts.labels();
    const n_labels = labels.len;

    // --- host: the time embedding and every block's modulation -------------
    const t_emb = try gpa.alloc(f32, n_labels * cfg.time_embed_dim);
    defer gpa.free(t_emb);
    minimax_h3.timeEmbed(t_emb, dit.adaln_t_table, cfg.adaln_curve_grid.?, cfg.time_embed_dim, labels);

    const mod_host = try gpa.alloc(f32, dit.blocks.len * n_labels * 3 * 6 * h);
    defer gpa.free(mod_host);
    try minimax_h3.buildModTable(dit, io, gpa, mod_host, t_emb, n_labels);

    // --- device: the trunk -------------------------------------------------
    try ctx.beginBatch();
    errdefer ctx.abortBatch();
    try ctx.tensorUpload(ws.x_d, std.mem.sliceAsBytes(packed_h));
    try ctx.tensorUpload(ws.mod_d, std.mem.sliceAsBytes(mod_host));

    const label_stride = minimax_h3.modality_count * 6 * h;
    var vmask: ?Buf = null;
    var amask: ?Buf = null;
    var v_stage: []u32 = &.{};
    defer if (v_stage.len > 0) gpa.free(v_stage);
    var a_stage: []u32 = &.{};
    defer if (a_stage.len > 0) gpa.free(a_stage);
    if (ts.rowsFor(.video).len > 0) {
        const src = ts.rowsFor(.video);
        v_stage = try gpa.alloc(u32, src.len);
        for (v_stage, src) |*o, v| o.* = v;
        try ctx.tensorUpload(ws.vmask_d, std.mem.sliceAsBytes(v_stage));
        vmask = ws.vmask_d;
    }
    if (ts.rowsFor(.audio).len > 0) {
        const src = ts.rowsFor(.audio);
        a_stage = try gpa.alloc(u32, src.len);
        for (a_stage, src) |*o, v| o.* = v;
        try ctx.tensorUpload(ws.amask_d, std.mem.sliceAsBytes(a_stage));
        amask = ws.amask_d;
    }
    const segIdx = struct {
        fn go(kind: minimax_h3.Kind, vm: ?Buf, am: ?Buf) ?Buf {
            return switch (kind) {
                .video => vm,
                .audio => am,
                else => null,
            };
        }
    }.go;

    for (dit.blocks, 0..) |*b, bi| {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;

        // Attention half. Each segment normalizes and modulates with its own
        // modulation row; a segment is a row RANGE, which is a push constant here.
        for (layout.segments) |sg| {
            const tag = sg.kind.tag();
            const idx = segIdx(sg.kind, vmask, amask);
            const base = if (idx == null) ts.rowFor(sg.kind) else 0;
            try rmsModRange(ctx, ws.x_d, sg.start, ws.t1_d, sg.start, ws.mod_d, sg.len(), h, modOff(cfg, n_labels, bi, base, tag, 0), modOff(cfg, n_labels, bi, base, tag, 1), eps, idx, 0, label_stride);
        }
        try linPrep(ctx, ws.t1_d, seq, h);
        try linGemm(ctx, ws.q_d, rowSlice(b.attn.qkv.w, 0, inner));
        try linGemm(ctx, ws.k_d, rowSlice(b.attn.qkv.w, inner, inner));
        try linGemm(ctx, ws.v_d, rowSlice(b.attn.qkv.w, 2 * inner, inner));

        const qn = try normBuf(ctx, b.attn.q_norm);
        const kn = try normBuf(ctx, b.attn.k_norm);
        try ctx.opElt(.rmsnorm, ws.q_d, ws.q_d, qn, null, .{
            .u0 = @intCast(seq * cfg.n_heads),
            .u1 = @intCast(cfg.head_dim),
            .f0 = cfg.qk_norm_eps,
        }, seq * cfg.n_heads, 1, 1);
        try ctx.opElt(.rmsnorm, ws.k_d, ws.k_d, kn, null, .{
            .u0 = @intCast(seq * cfg.n_heads),
            .u1 = @intCast(cfg.head_dim),
            .f0 = cfg.qk_norm_eps,
        }, seq * cfg.n_heads, 1, 1);
        // PARTIAL split-half rope: `pairs` pairs of a `head_dim`-wide head, so the
        // tail passes through untouched.
        inline for (.{ ws.q_d, ws.k_d }) |qk| {
            const total = seq * cfg.n_heads * sess.pairs;
            try ctx.opElt(.rope_half_part, qk, null, sess.freqs_d, null, .{
                .u0 = @intCast(total),
                .u1 = @intCast(sess.pairs),
                .u2 = @intCast(sess.sinOff()),
                .u3 = @intCast(cfg.n_heads),
                .u4 = 0,
                .u5 = @intCast(cfg.head_dim),
            }, total, 1, 1);
        }
        try attention(ctx, ws, seq, cfg, scale);

        try linPrep(ctx, ws.attn_d, seq, inner);
        try linGemm(ctx, ws.t1_d, b.attn.out.w);
        for (layout.segments) |sg| {
            const idx = segIdx(sg.kind, vmask, amask);
            const base = if (idx == null) ts.rowFor(sg.kind) else 0;
            try gatedAddRange(ctx, ws.x_d, sg.start, ws.t1_d, sg.start, ws.mod_d, sg.len() * h, h, modOff(cfg, n_labels, bi, base, sg.kind.tag(), 2), idx, 0, label_stride);
        }

        // MLP half, in row bands so the gate/up intermediates stay bounded. A band
        // can straddle segments, so the modulation is applied per intersection.
        var c0: usize = 0;
        while (c0 < seq) : (c0 += Workspace.mlp_tile) {
            const tile = @min(Workspace.mlp_tile, seq - c0);
            for (layout.segments) |sg| {
                const lo = @max(sg.start, c0);
                const hi = @min(sg.stop, c0 + tile);
                if (lo >= hi) continue;
                const tag = sg.kind.tag();
                const idx = segIdx(sg.kind, vmask, amask);
                const base = if (idx == null) ts.rowFor(sg.kind) else 0;
                try rmsModRange(ctx, ws.x_d, lo, ws.t1_d, lo - c0, ws.mod_d, hi - lo, h, modOff(cfg, n_labels, bi, base, tag, 3), modOff(cfg, n_labels, bi, base, tag, 4), eps, idx, lo - sg.start, label_stride);
            }
            try linPrep(ctx, ws.t1_d, tile, h);
            try linGemm(ctx, ws.gate_d, rowSlice(b.mlp.fc1.w, 0, cfg.ffn));
            try linGemm(ctx, ws.up_d, rowSlice(b.mlp.fc1.w, cfg.ffn, cfg.ffn));
            try ctx.opElt(.silu_mul, ws.gate_d, ws.up_d, null, null, .{
                .u0 = @intCast(tile * cfg.ffn),
            }, tile * cfg.ffn, 1, 1);
            try linPrep(ctx, ws.gate_d, tile, cfg.ffn);
            try linGemm(ctx, ws.t1_d, b.mlp.fc2.w);
            for (layout.segments) |sg| {
                const lo = @max(sg.start, c0);
                const hi = @min(sg.stop, c0 + tile);
                if (lo >= hi) continue;
                const idx = segIdx(sg.kind, vmask, amask);
                const base = if (idx == null) ts.rowFor(sg.kind) else 0;
                try gatedAddRange(ctx, ws.x_d, lo, ws.t1_d, lo - c0, ws.mod_d, (hi - lo) * h, h, modOff(cfg, n_labels, bi, base, sg.kind.tag(), 5), idx, lo - sg.start, label_stride);
            }
        }
    }
    try ctx.endBatch();

    // --- host: the output heads --------------------------------------------
    const trunk = try gpa.alloc(f32, seq * h);
    defer gpa.free(trunk);
    try ctx.tensorDownload(ws.x_d, std.mem.sliceAsBytes(trunk));
    try minimax_h3.finalHeads(dit, io, gpa, layout, &ts, t_emb, trunk, out_video, out_audio);
}

/// `rms_mod` over a row range of `x` into a row range of `out`.
fn rmsModRange(ctx: *Context, x: Buf, x_row: usize, out: Buf, out_row: usize, mod: Buf, rows: usize, dim: usize, premul_off: usize, shift_off: usize, eps: f32, idx: ?Buf, idx_row: usize, idx_stride: usize) !void {
    // The kernel reads `idx` exactly when the stride is non-zero; a stride with no
    // buffer bound is a read of whatever is in that descriptor slot.
    const stride: u32 = if (idx == null) 0 else @intCast(idx_stride);
    try ctx.opElt(.rms_mod, x, out, mod, idx, .{
        .u0 = @intCast(rows),
        .u1 = @intCast(dim),
        .u2 = @intCast(premul_off),
        .u3 = @intCast(shift_off),
        .u4 = stride,
        .u5 = @intCast(x_row),
        .u6 = @intCast(out_row),
        .f0 = eps,
        .f1 = @floatFromInt(idx_row),
    }, rows, 1, 1);
}

/// `gated_add` over a row range; `total` is in elements, the offsets in rows.
fn gatedAddRange(ctx: *Context, a: Buf, a_row: usize, b: Buf, b_row: usize, mod: Buf, total: usize, dim: usize, gate_off: usize, idx: ?Buf, idx_row: usize, idx_stride: usize) !void {
    const stride: u32 = if (idx == null) 0 else @intCast(idx_stride);
    try ctx.opElt(.gated_add, a, b, mod, idx, .{
        .u0 = @intCast(total),
        .u1 = @intCast(dim),
        .u2 = @intCast(gate_off),
        .u3 = stride,
        .u4 = @intCast(a_row * dim),
        .u5 = @intCast(b_row * dim),
        .u6 = @intCast(idx_row),
    }, total, 1, 1);
}

/// Full MHA over the whole pack, q/k/v already normed and rope'd, into `attn_d`.
///
/// The tensor-core pipeline is the same shape `mageflow_gpu` and `zimage_gpu` run:
/// half-precision planes, then per head batch a scores tile, a two-pass softmax and
/// the PV GEMM. `attn_full` is the reference arm, and a requirement rather than a
/// speed choice at production sizes: it is one thread per (query, head) looping
/// every key, which trips the GPU watchdog and surfaces as a device loss.
fn attention(ctx: *Context, ws: *Workspace, seq: usize, cfg: minimax_h3.Config, scale: f32) !void {
    const heads = cfg.n_heads;
    const hd = cfg.head_dim;
    const inner = heads * hd;
    if (force_naive_attn or ctx.pipe_scores == .null_handle) {
        return ctx.opElt(.attn_full, ws.q_d, ws.k_d, ws.v_d, ws.attn_d, .{
            .u0 = @intCast(seq),
            .u1 = @intCast(heads),
            .u2 = @intCast(heads),
            .u3 = @intCast(hd),
            .f0 = scale,
        }, seq * heads, 1, 1);
    }

    const rows_pad = Workspace.padRows(seq);
    const nchunks = Workspace.nchunks;
    ctx.independent(3);
    try ctx.opElt(.f32_to_h16, ws.q_d, null, null, ws.qh_d, .{
        .u0 = @intCast(rows_pad * inner / 2),
        .u1 = @intCast(seq * inner),
        .f0 = scale,
    }, rows_pad * inner / 2, 1, 1);
    try ctx.opElt(.gather_kmajor_h16, ws.k_d, null, null, ws.kh_d, .{
        .u0 = @intCast(inner * rows_pad / 2),
        .u1 = @intCast(hd),
        .u2 = @intCast(rows_pad),
        .u3 = @intCast(seq),
        .u4 = @intCast(heads),
    }, inner * rows_pad / 2, 1, 1);
    try ctx.opElt(.f32_to_h16, ws.v_d, null, null, ws.v16_d, .{
        .u0 = @intCast(rows_pad * inner / 2),
        .u1 = @intCast(seq * inner),
        .f0 = 1.0,
    }, rows_pad * inner / 2, 1, 1);

    const s_stride: u32 = @intCast(rows_pad);
    const s_plane: u32 = @intCast(rows_pad * rows_pad);
    var h0: usize = 0;
    while (h0 < heads) : (h0 += 1) {
        try ctx.opAttnScores(ws.s_d, ws.qh_d, ws.kh_d, .{
            .u0 = @intCast(inner),
            .u1 = s_stride,
            .u2 = @intCast(h0),
            .u3 = 1,
            .u4 = @intCast(hd * rows_pad),
            .u5 = s_plane,
        }, rows_pad / 128, rows_pad / 128, 1);
        try ctx.opElt(.softmax_partial, ws.s_d, null, null, ws.part_d, .{
            .u0 = @intCast(seq * nchunks),
            .u1 = @intCast(nchunks),
            .u2 = @intCast(seq),
            .u3 = s_stride,
            .u5 = s_plane,
        }, seq * nchunks, 1, 1);
        try ctx.opElt(.softmax_combine, ws.part_d, null, null, ws.md_d, .{
            .u0 = @intCast(seq),
            .u1 = @intCast(nchunks),
            .u2 = @intCast(seq),
            .u3 = s_stride,
        }, seq, 1, 1);
        try ctx.opAttnOut(ws.s_d, ws.v16_d, ws.attn_d, ws.md_d, .{
            .u0 = s_stride,
            .u1 = s_plane,
            .u2 = @intCast(h0),
            .u3 = 1,
            .u4 = @intCast(inner),
            .u5 = @intCast(inner),
            .f0 = @bitCast(@as(u32, @intCast(seq))),
            .f1 = @bitCast(s_stride),
        }, rows_pad / 128, 1);
    }
}
