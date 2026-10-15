//! Vulkan MiniMax H3 trunk forward, the device twin of `minimax_h3.forward`. The walk
//! itself is `minimax_h3_dev.zig`; this is its adapter (`Dev`) and what the backend
//! owns. `minimax_h3_cuda`'s header has the three things that make H3's block
//! different from a DiT's.
//!
//! Numerics match `minimax_h3.forward` up to int8 quantization and the softmax
//! decomposition, the same regime every other Vulkan arm runs in.
//! `minimax-h3-vk-test` checks it against the CPU forward on real weights.

const std = @import("std");
const minimax_h3 = @import("minimax_h3.zig");
const gpu = @import("tp_gpu").context;
const ops = @import("tp_ops");
const lin_gpu = @import("lin_gpu.zig");
const minimax_h3_dev = @import("minimax_h3_dev.zig");

const DiT = minimax_h3.DiT;
const Context = gpu.Context;
const Buf = gpu.DeviceBuffer;
const Weight = ops.matmul.Weight;

/// An unallocated handle; a Vulkan `DeviceBuffer` has no empty literal.
const nullBuf: Buf = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 };

/// Force the reference attention path (`attn_full`) instead of the tensor-core
/// pipeline, for an A/B on one render.
pub var force_naive_attn: bool = false;

/// Run integer weights weight-only, so a device check can compare activation dtypes
/// without the W8A8 prep's re-roll noise.
pub var force_weight_only: bool = false;

const lin_use: lin_gpu.Use = .{ .blockq = .auto };

/// Whether this context can run the trunk and the LoRA stack on it.
pub fn supported(ctx: *Context, dit: *const DiT) bool {
    return lin_gpu.runs(ctx, dit.device_lins, lin_use, dit.lora, "minimax_h3_gpu");
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

/// Per-shape device scratch, sized like the CUDA arm's: band-sized activations, K and
/// V over the whole sequence, and every GEMM output padded to 128 rows.
pub const Workspace = struct {
    x_d: Buf = nullBuf,
    t1_d: Buf = nullBuf,
    q_d: Buf = nullBuf,
    k_d: Buf = nullBuf,
    v_d: Buf = nullBuf,
    attn_d: Buf = nullBuf,
    /// The out-projection of one query band. Its own buffer, not a view into `t1_d`:
    /// the int8 routes write the row count ROUNDED UP to 128.
    ao_d: Buf = nullBuf,
    gate_d: Buf = nullBuf,
    up_d: Buf = nullBuf,
    mod_d: Buf = nullBuf,
    fmod_d: Buf = nullBuf,
    vmask_d: Buf = nullBuf,
    amask_d: Buf = nullBuf,
    /// Every block linear's dispatch, and what it decided for this checkpoint.
    lin: ?lin_gpu.Lin = null,
    lin_plan: lin_gpu.Plan = undefined,
    /// Tensor-core attention: the band's half Q, the whole sequence's half K (k-major)
    /// and V, one head's scores plane over the band, the softmax partials.
    qh_d: Buf = nullBuf,
    kh_d: Buf = nullBuf,
    v16_d: Buf = nullBuf,
    s_d: Buf = nullBuf,
    part_d: Buf = nullBuf,
    md_d: Buf = nullBuf,

    /// Chunks the softmax is decomposed into along the key axis.
    pub const nchunks: usize = 8;

    const padRows = minimax_h3_dev.padRows;

    pub fn init(ctx: *Context, dit: *const DiT, seq: usize, act: gpu.Dt) !Workspace {
        const cfg = dit.cfg;
        const inner = cfg.n_heads * cfg.head_dim;
        const mpad = padRows(seq);
        const aband = padRows(@min(minimax_h3_dev.attn_band, seq));
        const tband = padRows(@min(@max(minimax_h3_dev.attn_band, minimax_h3_dev.mlp_tile), seq));
        const tile = padRows(@min(minimax_h3_dev.mlp_tile, seq));
        var ws: Workspace = .{};
        errdefer ws.deinit(ctx);
        ws.lin = try lin_gpu.Lin.init(ctx.gpa, ctx, lin_gpu.maxRows(dit.device_lins), 0, 0);
        ws.lin_plan = try lin_gpu.plan(ctx, dit.device_lins, lin_use, "minimax_h3_gpu");
        if (force_weight_only) ws.lin_plan.int8 = .weight_only;
        if (dit.lora) |st| {
            ws.lin_plan.lora = st;
            try ws.lin.?.presizeLora(st, dit.device_lins, tband);
        }
        ws.x_d = try ctx.tensorCreate(mpad * cfg.hidden * act.bytes());
        ws.x_d.dt = act;
        ws.t1_d = try ctx.tensorCreate(tband * cfg.hidden * act.bytes());
        ws.t1_d.dt = act;
        ws.q_d = try ctx.tensorCreate(aband * inner * 4);
        ws.k_d = try ctx.tensorCreate(mpad * inner * 4);
        ws.v_d = try ctx.tensorCreate(mpad * inner * 4);
        ws.attn_d = try ctx.tensorCreate(aband * inner * 4);
        ws.ao_d = try ctx.tensorCreate(aband * cfg.hidden * 4);
        ws.gate_d = try ctx.tensorCreate(tile * cfg.ffn * 4);
        ws.up_d = try ctx.tensorCreate(tile * cfg.ffn * 4);
        ws.mod_d = try ctx.tensorCreate(dit.blocks.len * minimax_h3.Timesteps.max_labels * 3 * 6 * cfg.hidden * 4);
        ws.fmod_d = try ctx.tensorCreate(minimax_h3.Timesteps.max_labels * 2 * cfg.hidden * 4);
        ws.vmask_d = try ctx.tensorCreate(seq * 4);
        ws.amask_d = try ctx.tensorCreate(seq * 4);

        // Padded on BOTH axes: the scores kernel tiles 128x128 and reads whole tiles.
        ws.qh_d = try ctx.tensorCreate(aband * inner * 2);
        ws.kh_d = try ctx.tensorCreate(mpad * inner * 2);
        ws.v16_d = try ctx.tensorCreate(mpad * inner * 2);
        ws.s_d = try ctx.tensorCreate(aband * mpad * 2);
        ws.part_d = try ctx.tensorCreate(aband * nchunks * 2 * 4);
        ws.md_d = try ctx.tensorCreate(aband * 2 * 4);
        return ws;
    }

    pub fn deinit(self: *Workspace, ctx: *Context) void {
        if (self.lin) |*l| l.deinit();
        inline for (@typeInfo(Workspace).@"struct".fields) |f| {
            if (f.type == Buf) ctx.tensorDestroy(&@field(self, f.name));
        }
    }
};

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
    std.debug.assert(sess.seq == layout.seq_len);
    var step = try minimax_h3.DeviceStep.init(dit, io, gpa, layout, in);
    defer step.deinit(gpa);

    try ctx.beginBatch();
    errdefer if (ctx.batching) ctx.abortBatch();
    if (ws.x_d.dt == .f32) {
        try ctx.tensorUpload(ws.x_d, std.mem.sliceAsBytes(step.packed_h));
    } else {
        const bytes = try minimax_h3_dev.narrow(gpa, step.packed_h, ws.x_d.dt);
        defer gpa.free(bytes);
        try ctx.tensorUpload(ws.x_d, bytes);
    }
    try ctx.tensorUpload(ws.mod_d, std.mem.sliceAsBytes(step.mod));
    try ctx.tensorUpload(ws.fmod_d, std.mem.sliceAsBytes(step.fmod));
    var vmask: ?Buf = null;
    var amask: ?Buf = null;
    if (step.video_idx.len > 0) {
        try ctx.tensorUpload(ws.vmask_d, std.mem.sliceAsBytes(step.video_idx));
        vmask = ws.vmask_d;
    }
    if (step.audio_idx.len > 0) {
        try ctx.tensorUpload(ws.amask_d, std.mem.sliceAsBytes(step.audio_idx));
        amask = ws.amask_d;
    }
    try minimax_h3_dev.body(Dev{ .ctx = ctx, .sess = sess, .ws = ws, .cfg = cfg, .vmask = vmask, .amask = amask }, dit, layout, &step, cancel);
    try ctx.endBatch();
    // A 16-bit residual that overflowed can leave the heads finite.
    if (ws.x_d.dt != .f32 and !try ctx.allFinite(ws.x_d, layout.seq_len * cfg.hidden)) return error.NonFiniteActivation;

    const v_rows = try gpa.alloc(f32, layout.segmentOf(.video).?.len() * cfg.videoPatchDim());
    defer gpa.free(v_rows);
    const a_rows = try gpa.alloc(f32, layout.segmentOf(.audio).?.len() * minimax_h3.audio_latent_channels);
    defer gpa.free(a_rows);
    try ctx.tensorDownload(ws.q_d, std.mem.sliceAsBytes(v_rows));
    try ctx.tensorDownload(ws.k_d, std.mem.sliceAsBytes(a_rows));
    minimax_h3.DeviceStep.finish(layout, out_video, out_audio, v_rows, a_rows);
}

/// The `minimax_h3_dev` adapter.
const Dev = struct {
    ctx: *Context,
    sess: *const Session,
    ws: *Workspace,
    cfg: minimax_h3.Config,
    vmask: ?Buf,
    amask: ?Buf,

    fn l(d: Dev) *const lin_gpu.Lin {
        return &d.ws.lin.?;
    }

    pub fn labelIdx(d: Dev, kind: minimax_h3.Kind) ?Buf {
        return switch (kind) {
            .video => d.vmask,
            .audio => d.amask,
            else => null,
        };
    }

    /// The kernel reads `idx` exactly when the stride is non-zero; a stride with no
    /// buffer bound is a read of whatever is in that descriptor slot.
    pub fn rmsMod(d: Dev, src: Buf, src_row: usize, dst: Buf, dst_row: usize, table: Buf, rows: usize, premul_off: usize, shift_off: usize, norm_eps: f32, idx: ?Buf, idx_row: usize, stride: usize) !void {
        std.debug.assert(src.dt == dst.dt);
        try d.ctx.opElt(gpu.eltAt("rms_mod", src.dt), src, dst, table, idx, .{
            .u0 = @intCast(rows),
            .u1 = @intCast(d.cfg.hidden),
            .u2 = @intCast(premul_off),
            .u3 = @intCast(shift_off),
            .u4 = if (idx == null) 0 else @intCast(stride),
            .u5 = @intCast(src_row),
            .u6 = @intCast(dst_row),
            .f0 = norm_eps,
            .f1 = @floatFromInt(idx_row),
        }, rows, 1, 1);
    }

    /// `total` is in elements, the offsets in rows.
    pub fn gatedAdd(d: Dev, dst: Buf, dst_row: usize, src: Buf, src_row: usize, table: Buf, total: usize, gate_off: usize, idx: ?Buf, idx_row: usize, stride: usize, scale: f32) !void {
        const h = d.cfg.hidden;
        const plain = dst.dt == src.dt and scale == 1;
        std.debug.assert(plain or src.dt == .f32);
        try d.ctx.opElt(if (plain) gpu.eltAt("gated_add", dst.dt) else gpu.eltAt("gated_add_scaled", dst.dt), dst, src, table, idx, .{
            .u0 = @intCast(total),
            .u1 = @intCast(h),
            .u2 = @intCast(gate_off),
            .u3 = if (idx == null) 0 else @intCast(stride),
            .u4 = @intCast(dst_row * h),
            .u5 = @intCast(src_row * h),
            .u6 = @intCast(idx_row),
            .f0 = scale,
        }, total, 1, 1);
    }

    pub fn prep(d: Dev, x: Buf, rows: usize, cols: usize, ws: []const Weight) !lin_gpu.Group {
        return d.l().prep(d.ws.lin_plan, x, rows, cols, ws);
    }

    pub fn gemm(d: Dev, g: lin_gpu.Group, y: Buf, w: Weight) !void {
        try d.l().gemm(d.ws.lin_plan, g, y, w, .{});
    }

    pub fn linear(d: Dev, y: Buf, x: Buf, rows: usize, w: Weight) !void {
        try d.l().linear(d.ws.lin_plan, y, 0, x, rows, w);
    }

    pub fn linearAt(d: Dev, y: Buf, y_off: usize, x: Buf, rows: usize, w: Weight, bias: []const f32) !void {
        try d.l().linearAt(d.ws.lin_plan, y, y_off, x, 0, rows, w, bias);
    }

    pub fn qkNorm(d: Dev, x: Buf, w: []const f32, rows: usize) !void {
        const n = rows * d.cfg.n_heads;
        try d.ctx.opElt(.rmsnorm, x, x, try normBuf(d.ctx, w), null, .{
            .u0 = @intCast(n),
            .u1 = @intCast(d.cfg.head_dim),
            .f0 = d.cfg.qk_norm_eps,
        }, n, 1, 1);
    }

    /// PARTIAL split-half rope: `pairs` pairs of a `head_dim`-wide head, so the tail
    /// passes through untouched.
    pub fn rope(d: Dev, x: Buf, rows: usize, pos0: usize) !void {
        const total = rows * d.cfg.n_heads * d.sess.pairs;
        try d.ctx.opElt(.rope_half_part, x, null, d.sess.freqs_d, null, .{
            .u0 = @intCast(total),
            .u1 = @intCast(d.sess.pairs),
            .u2 = @intCast(d.sess.sinOff()),
            .u3 = @intCast(d.cfg.n_heads),
            .u4 = @intCast(pos0),
            .u5 = @intCast(d.cfg.head_dim),
        }, total, 1, 1);
    }

    pub fn kvAt(d: Dev, which: enum { k, v }, k0: usize) !Buf {
        const b = if (which == .k) d.ws.k_d else d.ws.v_d;
        const off = k0 * d.cfg.n_heads * d.cfg.head_dim * 4;
        return d.ctx.view(b, off, b.size - off);
    }

    pub fn kvStore(_: Dev, _: enum { k, v }, _: Buf, _: usize, _: usize) !void {}

    /// The half K (per-head k-major) and V the tensor-core attention reads, once per
    /// block rather than once per query band.
    pub fn kvReady(d: Dev, seq: usize) !void {
        if (naive(d.ctx)) return;
        const ws = d.ws;
        const hd = d.cfg.head_dim;
        const inner = d.cfg.n_heads * hd;
        const kv_pad = minimax_h3_dev.padRows(seq);
        d.ctx.independent(2);
        try d.ctx.opElt(.gather_kmajor_h16, ws.k_d, null, null, ws.kh_d, .{
            .u0 = @intCast(inner * kv_pad / 2),
            .u1 = @intCast(hd),
            .u2 = @intCast(kv_pad),
            .u3 = @intCast(seq),
            .u4 = @intCast(d.cfg.n_heads),
        }, inner * kv_pad / 2, 1, 1);
        try d.ctx.opElt(.f32_to_h16, ws.v_d, null, null, ws.v16_d, .{
            .u0 = @intCast(kv_pad * inner / 2),
            .u1 = @intCast(seq * inner),
            .f0 = 1.0,
        }, kv_pad * inner / 2, 1, 1);
    }

    /// One query band against the whole key sequence into `attn_d`: per head a scores
    /// tile, a two-pass softmax and the PV GEMM. `attn_cross` is the reference arm, and
    /// it is one thread per (query, head) looping every key, which at production sizes
    /// trips the GPU watchdog and surfaces as a device loss.
    pub fn attention(d: Dev, rows: usize, seq: usize) !void {
        const ctx = d.ctx;
        const ws = d.ws;
        const heads = d.cfg.n_heads;
        const hd = d.cfg.head_dim;
        const inner = heads * hd;
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        if (naive(ctx)) {
            return ctx.opElt(.attn_cross, ws.q_d, ws.k_d, ws.v_d, ws.attn_d, .{
                .u0 = @intCast(rows),
                .u1 = @intCast(heads),
                .u2 = @intCast(hd),
                .u3 = @intCast(seq),
                .f0 = scale,
            }, rows * heads, 1, 1);
        }

        const q_pad = minimax_h3_dev.padRows(rows);
        const kv_pad = minimax_h3_dev.padRows(seq);
        const nchunks = Workspace.nchunks;
        try ctx.opElt(.f32_to_h16, ws.q_d, null, null, ws.qh_d, .{
            .u0 = @intCast(q_pad * inner / 2),
            .u1 = @intCast(rows * inner),
            .f0 = scale,
        }, q_pad * inner / 2, 1, 1);

        // The softmax couples its row count to the valid key count only through
        // `q % u2`, which holds for any band no longer than the sequence.
        const s_plane: u32 = @intCast(q_pad * kv_pad);
        var h0: usize = 0;
        while (h0 < heads) : (h0 += 1) {
            try ctx.opAttnScores(ws.s_d, ws.qh_d, ws.kh_d, .{
                .u0 = @intCast(inner),
                .u1 = @intCast(kv_pad),
                .u2 = @intCast(h0),
                .u3 = 1,
                .u4 = @intCast(hd * kv_pad),
                .u5 = s_plane,
            }, kv_pad / 128, q_pad / 128, 1);
            try ctx.opElt(.softmax_partial, ws.s_d, null, null, ws.part_d, .{
                .u0 = @intCast(rows * nchunks),
                .u1 = @intCast(nchunks),
                .u2 = @intCast(seq),
                .u3 = @intCast(kv_pad),
                .u5 = s_plane,
            }, rows * nchunks, 1, 1);
            try ctx.opElt(.softmax_combine, ws.part_d, null, null, ws.md_d, .{
                .u0 = @intCast(rows),
                .u1 = @intCast(nchunks),
                .u2 = @intCast(seq),
                .u3 = @intCast(q_pad),
            }, rows, 1, 1);
            try ctx.opAttnOut(ws.s_d, ws.v16_d, ws.attn_d, ws.md_d, .{
                .u0 = @intCast(kv_pad),
                .u1 = s_plane,
                .u2 = @intCast(h0),
                .u3 = 1,
                .u4 = @intCast(inner),
                .u5 = @intCast(inner),
                .f0 = @bitCast(@as(u32, @intCast(seq))),
                .f1 = @bitCast(@as(u32, @intCast(q_pad))),
            }, q_pad / 128, 1);
        }
    }

    pub fn siluMul(d: Dev, g: Buf, u: Buf, n: usize) !void {
        try d.ctx.opElt(.silu_mul, g, u, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
    }
};

fn naive(ctx: *const Context) bool {
    return force_naive_attn or ctx.pipe_scores == .null_handle;
}

/// A weight vector as a small device buffer, for the per-head q/k norms.
fn normBuf(ctx: *Context, w: []const f32) !Buf {
    return .{ .buf = try ctx.smallBuffer(std.mem.sliceAsBytes(w)), .mem = .null_handle, .size = w.len * 4 };
}
