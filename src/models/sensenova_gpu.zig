//! GPU-resident SenseNova U1.5 on Vulkan, the device twin of `models/sensenova.zig`.
//! The passes themselves are `sensenova_dev.zig`; this is their adapter (`Dev`) and
//! what the backend owns.
//!
//! Both passes run here, and they are different shapes rather than one shape twice:
//!
//! - the PREFIX pass runs the base MoT copy over the prompt, once per conditioning,
//!   and keeps every layer's keys and values. Its per-query key range is a buffer,
//!   not a rule: a plain prompt is `[0, i+1)` and one carrying reference pictures is
//!   block-causal, so `attnBatched` serves both and the bounds say which;
//! - the GENERATION pass runs the `_mot_gen` copy over the canvas tokens every step,
//!   unmasked, over the prefix KV concatenated with its own. That is rectangular GQA
//!   attention (`seq_q` = canvas against `seq_kv` = prefix + canvas), which is what
//!   the two-pass flash pipeline already does.
//!
//! The per-head norms and the RoPE both work on SPANS of the head rather than the
//! whole of it (64 for the sequence position, then 32 + 32 for the token's row and
//! column), which is why `group_rmsnorm` and `rope_half_span_pos` are used where the
//! other families call a plain head norm and `rope_half`.
//!
//! Two things differ from the CUDA arm, both forced by the backend rather than
//! chosen. A device buffer is an opaque HANDLE here, not a pointer, so there is no
//! `offsetBuf`: a partial read is `tensorDownloadAt` and a concatenation is the
//! `copy` kernel with source and destination offsets. And `tensorCopy` flushes an
//! open recording batch, so the per-layer concatenation goes through that kernel
//! instead, which keeps a whole layer in one submission.

const std = @import("std");
const sensenova = @import("sensenova.zig");
const conv_gpu = @import("conv_gpu.zig");
const lin_gpu = @import("lin_gpu.zig");
const sensenova_dev = @import("sensenova_dev.zig");
const sd_unet_gpu = @import("sd_unet_gpu.zig");
const gpu = @import("tp_gpu").context;
const ops = @import("tp_ops");

const Model = sensenova.Model;
const Config = sensenova.Config;
const Prefix = sensenova.Prefix;
const Context = gpu.Context;
const Buf = gpu.DeviceBuffer;
const Dt = gpu.Dt;
const Weight = ops.matmul.Weight;

/// Force the naive one-thread-per-(query, head) attention instead of the flash
/// pipeline. For A/B and for reproducing a mismatch; the device test runs both.
pub var force_naive_attn: bool = false;

/// Run integer weights weight-only, so the device check can split a W8A8 forward's
/// activation noise from everything else. At f16 (`weight_only_f16`): a bf16 expansion
/// rounds every int8-times-scale weight to 8 bits and read 2x the dense floor at depth.
pub var force_weight_only: bool = false;

/// The flash pipeline tiles queries 128 at a time and stages keys the same way.
const flash_tile = 128;

/// Cap on one im2col band, so a 3x3 over a 1024-wide canvas does not materialize a
/// [4096][9216] patch in one allocation.
const patch_band_bytes: usize = 64 << 20;

/// The generation stream runs as a trunk; the prompt pass is a conditioning, so its
/// block quants stay on the f16 expansion and its integer weights take the
/// conditioning policy, both of which keep the activation unquantized.
const gen_use: lin_gpu.Use = .{ .blockq = .auto };
fn prefixUse() lin_gpu.Use {
    return .{ .blockq = .f16, .int8 = lin_gpu.dual_lin.encoderInt8() };
}

/// The generation stream's linear plan, without a LoRA.
pub fn canvasPlan(ctx: *Context, model: *const Model) !lin_gpu.Plan {
    return lin_gpu.plan(ctx, model.device_lins, gen_use, "sensenova_gpu");
}

/// Whether this context can run every linear this model has, in BOTH streams: a
/// checkpoint that quantized the two MoT copies differently would otherwise build a
/// session and fail inside the encode.
pub fn supported(ctx: *Context, model: *const Model) bool {
    if (model.layers.len == 0) return false;
    // The convolutions are f32 weights through the f16-weight coop GEMM, and there is
    // no scalar fallback wide enough to be worth offering.
    if (ctx.pipe_coop_bf16w == .null_handle and ctx.pipe_coop_f16w == .null_handle) {
        std.log.err("sensenova_gpu: this device has no f16-weight GEMM, which the convolutions need", .{});
        return false;
    }
    return lin_gpu.runs(ctx, model.device_lins, gen_use, model.lora, "sensenova_gpu") and
        lin_gpu.runs(ctx, model.prefix_lins, prefixUse(), model.lora, "sensenova_gpu prefix");
}

// --- shared pieces ----------------------------------------------------------

/// Wrap a host f32 slice as a (pointer-cached) small device buffer.
fn smallBuf(ctx: *Context, values: []const f32) !Buf {
    return .{ .buf = try ctx.smallBuffer(std.mem.sliceAsBytes(values)), .mem = .null_handle, .size = 0 };
}

/// A device buffer of its own for a u32 position array, owned by the caller.
///
/// NOT `smallBuffer`: that cache is keyed on the HOST POINTER, so a scratch array
/// freed here whose address is later recycled would be served this upload.
fn uploadU32(ctx: *Context, values: []const u32) !Buf {
    var buf = try ctx.tensorCreate(values.len * 4);
    errdefer ctx.tensorDestroy(&buf);
    try ctx.tensorUpload(buf, std.mem.sliceAsBytes(values));
    return buf;
}

/// A `sensenova.FlatRope` on the device.
const Table = struct {
    buf: Buf,
    half: usize,
    sin_off: usize,

    fn init(ctx: *Context, gpa: std.mem.Allocator, rows: usize, span: usize, theta: f64, interleaved: bool) !Table {
        const f = try sensenova.FlatRope.init(gpa, rows, span, theta, interleaved);
        defer gpa.free(f.data);
        var buf = try ctx.tensorCreate(f.data.len * 4);
        errdefer ctx.tensorDestroy(&buf);
        try ctx.tensorUpload(buf, std.mem.sliceAsBytes(f.data));
        return .{ .buf = buf, .half = f.half, .sin_off = f.sin_off };
    }

    fn deinit(self: *Table, ctx: *Context) void {
        ctx.tensorDestroy(&self.buf);
    }
};

/// Per-head RMSNorm over the head dim's two equally wide spans, in one launch.
///
/// `norm_w` is the layer's `[q_norm ++ q_norm_hw]` pair as one 128-wide vector, so
/// each half of the head is normalized against its own scale: the two spans are
/// contiguous and equally wide, which is exactly the group form.
fn groupNorm(ctx: *Context, x: Buf, w: Buf, rows: usize, dim: usize, groups: usize, eps: f32) !void {
    std.debug.assert(groups > 0 and dim % groups == 0);
    try ctx.opElt(.group_rmsnorm, x, x, w, null, .{
        .u0 = @intCast(rows * groups),
        .u1 = @intCast(dim / groups),
        .u2 = @intCast(groups),
        .f0 = eps,
    }, rows * groups, 1, 1);
}

/// The per-head norms and the three-span rope, applied to a q or k buffer in place.
fn qkPrepareAt(
    ctx: *Context,
    cfg: Config,
    x: Buf,
    rows: usize,
    heads: usize,
    norm_w: Buf,
    pos: [3]Buf,
    tables: [3]Table,
) !void {
    try groupNorm(ctx, x, norm_w, rows * heads, cfg.head_dim, 2, cfg.norm_eps);
    const t_span = cfg.spanT();
    const hw = cfg.spanHw();
    const offs = [3]usize{ 0, t_span, t_span + hw };
    inline for (0..3) |i| {
        const tb = tables[i];
        std.debug.assert(offs[i] + 2 * tb.half <= cfg.head_dim);
        const total = rows * heads * tb.half;
        try ctx.opElt(.rope_half_span_pos, x, pos[i], tb.buf, null, .{
            .u0 = @intCast(total),
            .u1 = @intCast(tb.half),
            .u2 = @intCast(tb.sin_off),
            .u3 = @intCast(heads),
            .u4 = @intCast(cfg.head_dim),
            .u5 = @intCast(offs[i]),
        }, total, 1, 1);
    }
}


/// Concatenate within the recorded batch: `dst[dst_off..][0..n] = src[src_off..]`.
///
/// `Context.tensorCopy` would do the same bytes, but it runs on the immediate command
/// buffer and flushes any open batch, which would cut every layer into three
/// submissions.
/// Casts when the two dtypes differ, which happens only at the trunk's ends.
fn copyInto(ctx: *Context, dst: Buf, dst_off: usize, src: Buf, src_off: usize, n: usize) !void {
    try ctx.opElt(gpu.eltCopy(src.dt, dst.dt), src, dst, null, null, .{
        .u0 = @intCast(n),
        .u2 = @intCast(dst_off),
        .u3 = @intCast(src_off),
    }, n, 1, 1);
}

// --- attention --------------------------------------------------------------

/// Attention from `b.q` against `b.kcat`/`b.vcat` into `b.attn`.
///
/// The two paths compute the same thing and differ in how the scores are produced.
/// The naive one is a thread per (query, head) reading the whole key range, which is
/// what the prefix pass and a mismatch hunt want; the flash one tiles it on the
/// tensor cores, which is what a render at 1024 canvas tokens against a 1289-key
/// context wants.
fn attentionAt(ctx: *Context, cfg: Config, b: LayerBufs, rows: usize, seq_kv: usize, tc: bool) !void {
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));
    const hd = cfg.head_dim;
    const heads = cfg.n_heads;
    const n_kv = cfg.n_kv_heads;
    if (!tc) {
        // The bounds buffer is the mask. `Workspace`/`PrefixWs` fill an unmasked one
        // with `[0, seq_kv)` for every row, so this arm never needs a second kernel.
        return ctx.attnBatchedDispatch(rows, heads, n_kv, hd, scale, 0);
    }
    const q_pad = std.mem.alignForward(usize, rows, flash_tile);
    const kv_pad = std.mem.alignForward(usize, seq_kv, flash_tile);
    // The scale rides into the f16 cast of Q rather than costing a pass of its own.
    try ctx.opElt(.f32_to_h16, b.q, null, null, b.q16, .{
        .u0 = @intCast(q_pad * heads * hd / 2),
        .u1 = @intCast(rows * heads * hd),
        .f0 = scale,
    }, q_pad * heads * hd / 2, 1, 1);
    try ctx.opElt(.gather_kmajor_h16, b.kcat, null, null, b.k16, .{
        .u0 = @intCast(n_kv * hd * kv_pad / 2),
        .u1 = @intCast(hd),
        .u2 = @intCast(kv_pad),
        .u3 = @intCast(seq_kv),
        .u4 = @intCast(n_kv),
    }, n_kv * hd * kv_pad / 2, 1, 1);
    try ctx.opElt(.f32_to_h16, b.vcat, null, null, b.v16, .{
        .u0 = @intCast(kv_pad * n_kv * hd / 2),
        .u1 = @intCast(seq_kv * n_kv * hd),
        .f0 = 1.0,
    }, kv_pad * n_kv * hd / 2, 1, 1);

    // The MD table is one {max, 1/sum} pair per query row per head, and it lives past
    // the attention output in the same buffer.
    const md_off = q_pad * heads * hd;
    const push: gpu.EltPush = .{
        .u0 = @intCast(heads * hd),
        .u1 = @intCast(kv_pad),
        .u2 = 0,
        .u3 = @intCast(heads / n_kv),
        .u4 = @intCast(n_kv * hd),
        .u5 = @intCast(md_off),
        .f0 = @bitCast(@as(u32, @intCast(seq_kv))),
        // The MD table is indexed by QUERY row, so its plane stride is `q_pad`, not
        // `u1`, which is the key padded length. 0 would mean "= u1" and is only right
        // where the two coincide.
        .f1 = if (q_pad == kv_pad) 0 else @bitCast(@as(u32, @intCast(q_pad))),
    };
    try ctx.opFlash(.md, b.q16, b.k16, b.v16, b.attn, push, q_pad / flash_tile, heads);
    try ctx.opFlash(.out, b.q16, b.k16, b.v16, b.attn, push, q_pad / flash_tile, heads);
}

/// The `[starts ++ ends]` table `attnBatched` reads, as the pass's mask.
///
/// `ends` is the caller's when the prompt carries reference pictures (block-causal,
/// so a picture's tokens see each other in full and the range runs past the query's
/// own position), plain causal when it does not, and the whole key range when the
/// pass is unmasked.
fn boundsTable(gpa: std.mem.Allocator, rows: usize, seq_kv: usize, kv_end: ?[]const u32) ![]u32 {
    const pair = try gpa.alloc(u32, 2 * rows);
    @memset(pair[0..rows], 0);
    if (kv_end) |ends| {
        std.debug.assert(ends.len == rows);
        @memcpy(pair[rows..], ends);
    } else if (rows == seq_kv) {
        for (pair[rows..], 0..) |*e, i| e.* = @intCast(i + 1);
    } else {
        @memset(pair[rows..], @intCast(seq_kv));
    }
    return pair;
}

// --- the prefix pass --------------------------------------------------------

/// Device buffers for one prefix encode, sized to the prompt.
const PrefixWs = struct {
    x: Buf,
    nrm: Buf,
    dlt: Buf,
    q: Buf,
    k: Buf,
    v: Buf,
    attn: Buf,
    mg: Buf,
    mu: Buf,
    bounds: Buf,

    const fields = [_][]const u8{ "x", "nrm", "dlt", "q", "k", "v", "attn", "mg", "mu", "bounds" };

    fn init(ctx: *Context, cfg: Config, seq: usize) !PrefixWs {
        // Padded to 128 rows: a quantized route launches over `align(m, 128)` and each
        // block stores a whole tile, so a buffer sized to the exact row count is
        // written off the end.
        const rows = std.mem.alignForward(usize, seq, 128);
        const sizes = [fields.len]usize{
            rows * cfg.dim * 4,
            rows * cfg.dim * 4,
            rows * cfg.dim * 4,
            rows * cfg.qDim() * 4,
            rows * cfg.kvDim() * 4,
            rows * cfg.kvDim() * 4,
            rows * cfg.qDim() * 4,
            rows * cfg.inter * 4,
            rows * cfg.inter * 4,
            2 * seq * 4,
        };
        var self: PrefixWs = undefined;
        var made: usize = 0;
        errdefer inline for (fields, 0..) |name, i| {
            if (i < made) ctx.tensorDestroy(&@field(self, name));
        };
        inline for (fields, sizes) |name, size| {
            @field(self, name) = try ctx.tensorCreate(size);
            made += 1;
        }
        return self;
    }

    fn deinit(self: *PrefixWs, ctx: *Context) void {
        inline for (fields) |name| ctx.tensorDestroy(&@field(self, name));
        self.* = undefined;
    }
};

/// Run the prompt through the base MoT copy and return the prefix KV cache, the
/// conditioning this family's `encode` produces.
pub fn prefixForward(
    model: *const Model,
    ctx: *Context,
    io: std.Io,
    gpa: std.mem.Allocator,
    layout: sensenova.PromptLayout,
    refs: []const sensenova.RefImage,
    act: Dt,
    cancel: ?*std.atomic.Value(bool),
) !Prefix {
    const cfg = model.cfg;
    const seq = layout.ids.len;
    std.debug.assert(seq > 0);

    var pre: Prefix = .{
        .kv = try gpa.alloc(f32, cfg.n_layers * 2 * seq * cfg.kvDim()),
        .seq = seq,
        .kv_dim = cfg.kvDim(),
        .n_layers = cfg.n_layers,
        .time = layout.time,
    };
    errdefer pre.deinit(gpa);

    var ws = try PrefixWs.init(ctx, cfg, seq);
    defer ws.deinit(ctx);
    inline for (.{ "x", "nrm", "mg", "mu" }) |name| @field(ws, name).dt = sensenova_dev.streamDtFor(act);
    var plin = try lin_gpu.Lin.init(gpa, ctx, lin_gpu.maxRows(model.prefix_lins), 0, 0);
    defer plin.deinit();
    var pplan = try lin_gpu.plan(ctx, model.prefix_lins, prefixUse(), "sensenova_gpu prefix");
    try lin_gpu.presize(ctx, pplan, model.prefix_lins);
    if (model.lora) |st| {
        pplan.lora = st;
        try plin.presizeLora(st, model.prefix_lins, std.mem.alignForward(usize, seq, 128));
    }

    var pp = try sensenova.PromptPositions.init(gpa, layout);
    defer gpa.free(pp.pos);
    var tab_t = try Table.init(ctx, gpa, pp.max[0] + 1, cfg.spanT(), cfg.theta_t, false);
    defer tab_t.deinit(ctx);
    var tab_h = try Table.init(ctx, gpa, pp.max[1] + 1, cfg.spanHw(), cfg.theta_hw, false);
    defer tab_h.deinit(ctx);
    var tab_w = try Table.init(ctx, gpa, pp.max[2] + 1, cfg.spanHw(), cfg.theta_hw, false);
    defer tab_w.deinit(ctx);
    var pos_bufs = [3]Buf{ try uploadU32(ctx, pp.axis(0)), try uploadU32(ctx, pp.axis(1)), try uploadU32(ctx, pp.axis(2)) };
    defer for (&pos_bufs) |*p| ctx.tensorDestroy(p);

    {
        const pair = try boundsTable(gpa, seq, seq, layout.kv_end);
        defer gpa.free(pair);
        try ctx.tensorUpload(ws.bounds, std.mem.sliceAsBytes(pair));
    }
    {
        // The understanding tower runs on the host: it is ~30M parameters against the
        // trunk's 8B, and it runs once per conditioning rather than per step.
        const emb = try model.embedPrompt(io, gpa, layout, refs);
        defer gpa.free(emb);
        try ctx.tensorUpload(ws.dlt, std.mem.sliceAsBytes(emb));
    }
    // Bound BEFORE the first batch, and once: the five buffers are the same at every
    // layer, and the descriptor set is persistent, so rewriting it mid-batch would
    // touch a set already referenced by recorded commands.
    ctx.attnBatchedBind(ws.q, ws.k, ws.v, ws.attn, ws.bounds);

    try sensenova_dev.prefix(Dev{
        .ctx = ctx,
        .l = &plin,
        .plan = pplan,
        .cfg = cfg,
        .b = .{ .x = ws.x, .nrm = ws.nrm, .dlt = ws.dlt, .q = ws.q, .k = ws.k, .v = ws.v, .kcat = ws.k, .vcat = ws.v, .attn = ws.attn, .mg = ws.mg, .mu = ws.mu, .q16 = ws.q, .k16 = ws.k, .v16 = ws.v },
        .pos = pos_bufs,
        .tables = .{ tab_t, tab_h, tab_w },
        .tc = false,
        .kv = pre.kv,
    }, model, seq, cancel);
    return pre;
}

// --- the generation pass ----------------------------------------------------

/// Per-image cache: everything constant across sampling steps.
pub const Session = struct {
    cfg: Config,
    /// Canvas extents as the caller asked for them, and as the model pads them.
    lat_h: usize,
    lat_w: usize,
    ph: usize,
    pw: usize,
    /// Patch-embedder cell grid, then the token grid.
    gh: usize,
    gw: usize,
    th: usize,
    tw: usize,
    n_img: usize,
    prefix_seq: usize,
    /// Whether the flash pipeline is in force. Read once, here, not per call: it
    /// decides whether `Workspace` sizes the f16 attention operands at all.
    tc: bool,

    prefix_kv: Buf,
    tab_t: Table,
    tab_h: Table,
    tab_w: Table,
    tab_vx: Table,
    tab_vy: Table,
    pos_t: Buf,
    pos_h: Buf,
    pos_w: Buf,
    vis_x: Buf,
    vis_y: Buf,
    /// The generation stream's dispatch, and what it decided for this checkpoint.
    lin: lin_gpu.Lin,
    lin_plan: lin_gpu.Plan,

    pub fn init(
        gpa: std.mem.Allocator,
        ctx: *Context,
        model: *const Model,
        lat_h: usize,
        lat_w: usize,
        pre: Prefix,
    ) !Session {
        const cfg = model.cfg;
        const ph = sensenova.paddedExtent(cfg, lat_h);
        const pw = sensenova.paddedExtent(cfg, lat_w);
        const gh = ph / cfg.patch;
        const gw = pw / cfg.patch;
        const th = ph / cfg.tokenPx();
        const tw = pw / cfg.tokenPx();
        var l = try lin_gpu.Lin.init(gpa, ctx, lin_gpu.maxRows(model.device_lins), 0, 0);
        errdefer l.deinit();
        var lp = try canvasPlan(ctx, model);
        if (force_weight_only) lp.int8 = .weight_only_f16;
        if (model.lora) |st| {
            lp.lora = st;
            try l.presizeLora(st, model.device_lins, std.mem.alignForward(usize, th * tw, 128));
        }

        var self: Session = .{
            .lin = l,
            .lin_plan = lp,
            .cfg = cfg,
            .lat_h = lat_h,
            .lat_w = lat_w,
            .ph = ph,
            .pw = pw,
            .gh = gh,
            .gw = gw,
            .th = th,
            .tw = tw,
            .n_img = th * tw,
            .prefix_seq = pre.seq,
            .tc = !force_naive_attn and ctx.pipe_flash_md != .null_handle and ctx.pipe_flash_out != .null_handle,
            .prefix_kv = undefined,
            .tab_t = undefined,
            .tab_h = undefined,
            .tab_w = undefined,
            .tab_vx = undefined,
            .tab_vy = undefined,
            .pos_t = undefined,
            .pos_h = undefined,
            .pos_w = undefined,
            .vis_x = undefined,
            .vis_y = undefined,
        };
        var made: usize = 0;
        errdefer {
            if (made > 0) ctx.tensorDestroy(&self.prefix_kv);
            if (made > 1) self.tab_t.deinit(ctx);
            if (made > 2) self.tab_h.deinit(ctx);
            if (made > 3) self.tab_w.deinit(ctx);
            if (made > 4) self.tab_vx.deinit(ctx);
            if (made > 5) self.tab_vy.deinit(ctx);
            const pos = [_]*Buf{ &self.pos_t, &self.pos_h, &self.pos_w, &self.vis_x, &self.vis_y };
            for (pos[0..(made -| 6)]) |b| ctx.tensorDestroy(b);
        }

        self.prefix_kv = try ctx.tensorCreate(pre.kv.len * 4);
        made += 1;
        try ctx.tensorUpload(self.prefix_kv, std.mem.sliceAsBytes(pre.kv));

        // Every image token takes the same t index, one past the whole prefix.
        self.tab_t = try Table.init(ctx, gpa, pre.time + 1, cfg.spanT(), cfg.theta_t, false);
        made += 1;
        self.tab_h = try Table.init(ctx, gpa, th, cfg.spanHw(), cfg.theta_hw, false);
        made += 1;
        self.tab_w = try Table.init(ctx, gpa, tw, cfg.spanHw(), cfg.theta_hw, false);
        made += 1;
        self.tab_vx = try Table.init(ctx, gpa, gw, cfg.vis_dim / 2, cfg.theta_vis, true);
        made += 1;
        self.tab_vy = try Table.init(ctx, gpa, gh, cfg.vis_dim / 2, cfg.theta_vis, true);
        made += 1;

        var cp = try sensenova.CanvasPositions.init(gpa, pre.time, tw, self.n_img, gw, gh * gw);
        defer cp.deinit(gpa);
        inline for (.{ "pos_t", "pos_h", "pos_w", "vis_x", "vis_y" }, .{ .t, .h, .w, .vis_x, .vis_y }) |name, axis| {
            @field(self, name) = try uploadU32(ctx, cp.axis(axis));
            made += 1;
        }
        return self;
    }

    pub fn deinit(self: *Session, ctx: *Context) void {
        ctx.tensorDestroy(&self.prefix_kv);
        self.tab_t.deinit(ctx);
        self.tab_h.deinit(ctx);
        self.tab_w.deinit(ctx);
        self.tab_vx.deinit(ctx);
        self.tab_vy.deinit(ctx);
        inline for (.{ "pos_t", "pos_h", "pos_w", "vis_x", "vis_y" }) |name| ctx.tensorDestroy(&@field(self, name));
        self.lin.deinit();
        self.* = undefined;
    }
};

pub const Workspace = struct {
    patch: Buf,
    cells: Buf,
    img: Buf,
    nrm: Buf,
    dlt: Buf,
    q: Buf,
    k: Buf,
    v: Buf,
    kcat: Buf,
    vcat: Buf,
    attn: Buf,
    mg: Buf,
    mu: Buf,
    fm_a: Buf,
    fm_b: Buf,
    tvec: Buf,
    /// f16 attention operands, sized only under `Session.tc`; the naive path never
    /// reads them.
    q16: Buf,
    k16: Buf,
    v16: Buf,
    /// Per-query key range for the naive path: `[0, seq_kv)` for every canvas row.
    bounds: Buf,
    /// The canvas stream's storage (`img`, `nrm`, `mg`, `mu`); `sensenova_dev`.
    act: Dt,

    const fields = [_][]const u8{ "patch", "cells", "img", "nrm", "dlt", "q", "k", "v", "kcat", "vcat", "attn", "mg", "mu", "fm_a", "fm_b", "tvec", "q16", "k16", "v16", "bounds" };

    /// `prefix_cap` is the LONGEST prefix any branch sharing this workspace has, not
    /// the positive branch's. Under classifier-free guidance the two branches are
    /// different lengths (an edit's negative presents the pictures with no prompt, so
    /// it is shorter; a plain negative is shorter still), and sizing to one of them
    /// writes the other off the end of `kcat`.
    pub fn init(ctx: *Context, model: *const Model, sess: *const Session, prefix_cap: usize, act: Dt) !Workspace {
        const cfg = model.cfg;
        const rows = std.mem.alignForward(usize, sess.n_img, 128);
        const kv_seq = @max(prefix_cap, sess.prefix_seq) + sess.n_img;
        const kv_rows = std.mem.alignForward(usize, kv_seq, 128);
        const cells = sess.gh * sess.gw;
        // The fm_head's two intermediates, at their widest: after the first shuffle it
        // is `[2*th][2*tw][dim/4]`, the same element count as the token plane, and
        // after the second `[4*th][4*tw][192]`.
        const fm_max = @max(cells * cfg.vis_dim, @max(sess.n_img * cfg.dim, sess.th * 4 * sess.tw * 4 * 192));
        // The flash tiles are 128 wide and its MD table is one pair per query row per
        // head, kept past the attention output in the same buffer.
        const q_pad = std.mem.alignForward(usize, sess.n_img, flash_tile);
        const kv_pad = std.mem.alignForward(usize, kv_seq, flash_tile);
        const attn_len = @max(rows * cfg.qDim(), q_pad * cfg.qDim() + cfg.n_heads * q_pad * 2);
        const sizes = [fields.len]usize{
            // Grown on demand by the two convolution helpers.
            4,
            cells * cfg.vis_dim * 4,
            rows * cfg.dim * 4,
            rows * cfg.dim * 4,
            rows * cfg.dim * 4,
            rows * cfg.qDim() * 4,
            rows * cfg.kvDim() * 4,
            rows * cfg.kvDim() * 4,
            kv_rows * cfg.kvDim() * 4,
            kv_rows * cfg.kvDim() * 4,
            attn_len * 4,
            rows * cfg.inter * 4,
            rows * cfg.inter * 4,
            fm_max * 4,
            fm_max * 4,
            cfg.dim * 4,
            if (sess.tc) q_pad * cfg.qDim() * 2 else 4,
            if (sess.tc) kv_pad * cfg.kvDim() * 2 else 4,
            if (sess.tc) kv_pad * cfg.kvDim() * 2 else 4,
            2 * sess.n_img * 4,
        };
        var self: Workspace = undefined;
        var made: usize = 0;
        errdefer inline for (fields, 0..) |name, i| {
            if (i < made) ctx.tensorDestroy(&@field(self, name));
        };
        inline for (fields, sizes) |name, size| {
            @field(self, name) = try ctx.tensorCreate(size);
            made += 1;
        }
        self.act = act;
        inline for (.{ "img", "nrm", "mg", "mu" }) |name| @field(self, name).dt = act;
        return self;
    }

    pub fn deinit(self: *Workspace, ctx: *Context) void {
        inline for (fields) |name| ctx.tensorDestroy(&@field(self, name));
        self.* = undefined;
    }
};

/// One denoiser forward. `out` and `x_lat` are planar `[3][lat_h][lat_w]`; `t` is the
/// model's own timestep, `1 - sigma`.
pub fn forward(
    model: *const Model,
    ctx: *Context,
    sess: *Session,
    ws: *Workspace,
    io: std.Io,
    gpa: std.mem.Allocator,
    out: []f32,
    x_lat: []const f32,
    t: f32,
    cancel: ?*std.atomic.Value(bool),
) !void {
    const cfg = model.cfg;
    const ch = sensenova.latent_channels;
    std.debug.assert(x_lat.len == ch * sess.lat_h * sess.lat_w);
    std.debug.assert(out.len == x_lat.len);

    const xp = try sensenova.padCanvas(gpa, cfg, x_lat, ch, sess.lat_h, sess.lat_w);
    defer gpa.free(xp);

    // The patch rows are gathered on the host: the canvas arrives there anyway, and on
    // the device it would cost an upload of the canvas plus a kernel for the same bytes.
    {
        const patch = try sensenova.patchRows(gpa, cfg, xp, sess.ph, sess.pw);
        defer gpa.free(patch);
        try ctx.ensureDeviceBuffer(&ws.patch, patch.len * 4);
        try ctx.tensorUpload(ws.patch, std.mem.sliceAsBytes(patch));
    }
    // The scalar conditioning is two 256-wide MLPs; the host runs them in microseconds
    // and the device would need an upload either way.
    {
        const tvec = try model.timeVector(io, gpa, t, sess.ph, sess.pw);
        defer gpa.free(tvec);
        try ctx.tensorUpload(ws.tvec, std.mem.sliceAsBytes(tvec));
    }
    if (!sess.tc) {
        const pair = try boundsTable(gpa, sess.n_img, sess.prefix_seq + sess.n_img, null);
        defer gpa.free(pair);
        try ctx.tensorUpload(ws.bounds, std.mem.sliceAsBytes(pair));
        ctx.attnBatchedBind(ws.q, ws.kcat, ws.vcat, ws.attn, ws.bounds);
    }

    try ctx.beginBatch();
    errdefer if (ctx.batching) ctx.abortBatch();
    try sensenova_dev.canvas(Dev{
        .ctx = ctx,
        .l = &sess.lin,
        .plan = sess.lin_plan,
        .cfg = cfg,
        .b = .{ .x = ws.img, .nrm = ws.nrm, .dlt = ws.dlt, .q = ws.q, .k = ws.k, .v = ws.v, .kcat = ws.kcat, .vcat = ws.vcat, .attn = ws.attn, .mg = ws.mg, .mu = ws.mu, .q16 = ws.q16, .k16 = ws.k16, .v16 = ws.v16 },
        .pos = .{ sess.pos_t, sess.pos_h, sess.pos_w },
        .tables = .{ sess.tab_t, sess.tab_h, sess.tab_w },
        .tc = sess.tc,
    }, model, sess, ws, cancel);
    try ctx.endBatch();
    if (ws.act != .f32 and !try ctx.allFinite(ws.img, sess.n_img * cfg.dim)) return error.NonFiniteActivation;

    const rgb = try gpa.alloc(f32, sess.ph * sess.pw * ch);
    defer gpa.free(rgb);
    try ctx.tensorDownloadAt(ws.fm_a, 0, std.mem.sliceAsBytes(rgb));
    sensenova.velocityFromRows(cfg, out, xp, rgb, t, sess.lat_h, sess.lat_w, sess.ph, sess.pw);
}

const LayerBufs = struct {
    x: Buf,
    nrm: Buf,
    dlt: Buf,
    q: Buf,
    k: Buf,
    v: Buf,
    kcat: Buf,
    vcat: Buf,
    attn: Buf,
    mg: Buf,
    mu: Buf,
    q16: Buf,
    k16: Buf,
    v16: Buf,
};

/// The `sensenova_dev` adapter for one pass.
const Dev = struct {
    ctx: *Context,
    l: *const lin_gpu.Lin,
    plan: lin_gpu.Plan,
    cfg: Config,
    b: LayerBufs,
    pos: [3]Buf,
    tables: [3]Table,
    /// The flash pipeline; otherwise `attnBatched` over the bound mask.
    tc: bool,
    /// The prefix pass's host cache, `[layer][k|v][seq][kv_dim]`.
    kv: []f32 = &.{},

    pub fn rmsNorm(d: Dev, x: Buf, y: Buf, w: []const f32, rows: usize, eps: f32) !void {
        std.debug.assert(x.dt == y.dt);
        try d.ctx.opElt(gpu.eltAt("rmsnorm", x.dt), x, y, try smallBuf(d.ctx, w), null, .{
            .u0 = @intCast(rows),
            .u1 = @intCast(d.cfg.dim),
            .f0 = eps,
        }, rows, 1, 1);
    }

    pub fn gemms(d: Dev, x: Buf, rows: usize, cols: usize, ws: []const Weight, ys: []const Buf) !void {
        const g = try d.l.prep(d.plan, x, rows, cols, ws);
        for (ws, ys) |w, y| try d.l.gemm(d.plan, g, y, w, .{});
    }

    pub fn linear(d: Dev, y: Buf, x: Buf, rows: usize, w: Weight) !void {
        try d.l.linear(d.plan, y, 0, x, rows, w);
    }

    pub fn qkPrepare(d: Dev, x: Buf, rows: usize, heads: usize, norm_pair: []const f32) !void {
        try qkPrepareAt(d.ctx, d.cfg, x, rows, heads, try smallBuf(d.ctx, norm_pair), d.pos, d.tables);
    }

    pub fn copy(d: Dev, dst: Buf, dst_off: usize, src: Buf, src_off: usize, n: usize) !void {
        try copyInto(d.ctx, dst, dst_off, src, src_off, n);
    }

    pub fn attention(d: Dev, rows: usize, seq_kv: usize) !void {
        try attentionAt(d.ctx, d.cfg, d.b, rows, seq_kv, d.tc);
    }

    pub fn add(d: Dev, x: Buf, y: Buf, n: usize) !void {
        std.debug.assert(x.dt == y.dt);
        try d.ctx.opElt(gpu.eltAt("add", x.dt), x, y, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
    }

    pub fn siluMul(d: Dev, g: Buf, u: Buf, n: usize) !void {
        std.debug.assert(g.dt == u.dt);
        try d.ctx.opElt(gpu.eltAt("silu_mul", g.dt), g, u, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
    }

    pub fn addScaled(d: Dev, x: Buf, y: Buf, n: usize, s: f32) !void {
        std.debug.assert(y.dt == .f32);
        try d.ctx.opElt(gpu.eltAt("add_scaled", x.dt), x, y, null, null, .{ .u0 = @intCast(n), .f0 = s }, n, 1, 1);
    }

    pub fn scaleBy(d: Dev, x: Buf, s: f32, n: usize) !void {
        try d.ctx.opElt(.scale_f32, x, x, null, null, .{ .u0 = @intCast(n), .f0 = s }, n, 1, 1);
    }

    pub fn gelu(d: Dev, x: Buf, n: usize) !void {
        try d.ctx.opElt(.gelu_erf, x, null, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
    }

    pub fn prefetch(_: Dev, _: *const sensenova.Stream) void {}

    pub fn beginBatch(d: Dev) !void {
        try d.ctx.beginBatch();
    }

    pub fn endBatch(d: Dev) !void {
        try d.ctx.endBatch();
    }

    pub fn abortBatch(d: Dev) void {
        if (d.ctx.batching) d.ctx.abortBatch();
    }

    pub fn keepKv(d: Dev, li: usize) !void {
        const n = d.kv.len / (d.cfg.n_layers * 2);
        try d.ctx.tensorDownloadAt(d.b.k, 0, std.mem.sliceAsBytes(d.kv[(li * 2) * n ..][0..n]));
        try d.ctx.tensorDownloadAt(d.b.v, 0, std.mem.sliceAsBytes(d.kv[(li * 2 + 1) * n ..][0..n]));
    }

    pub fn patchEmbed(d: Dev, dst: Buf, src: Buf, rows: usize, cv: sensenova.Conv) !void {
        try conv_gpu.patchGemm(d.ctx, dst, src, rows, cv);
    }

    pub fn ropeInter(d: Dev, x: Buf, positions: Buf, tb: Table, rows: usize, row_dim: usize, off: usize) !void {
        std.debug.assert(off + 2 * tb.half <= row_dim);
        const total = rows * tb.half;
        try d.ctx.opElt(.rope_inter_span_pos, x, positions, tb.buf, null, .{
            .u0 = @intCast(total),
            .u1 = @intCast(tb.half),
            .u2 = @intCast(tb.sin_off),
            .u3 = @intCast(row_dim),
            .u4 = @intCast(off),
        }, total, 1, 1);
    }

    pub fn convStride(d: Dev, patch: *Buf, dst: Buf, src: Buf, h: usize, w: usize, cv: sensenova.Conv) !void {
        try conv_gpu.convStride(d.ctx, patch, dst, src, h, w, cv);
    }

    pub fn addBiasRows(d: Dev, x: Buf, bias: Buf, rows: usize, cols: usize) !void {
        try d.ctx.opElt(.add_bias_rows, x, bias, null, null, .{
            .u0 = @intCast(rows * cols),
            .u1 = @intCast(cols),
        }, rows * cols, 1, 1);
    }

    pub fn pixelShuffle(d: Dev, dst: Buf, src: Buf, h: usize, w: usize, co: usize, r: usize) !void {
        const total = h * w * co * r * r;
        try d.ctx.opElt(.pixel_shuffle, src, dst, null, null, .{
            .u0 = @intCast(total),
            .u1 = @intCast(co),
            .u2 = @intCast(r),
            .u3 = @intCast(w),
        }, total, 1, 1);
    }

    /// Through the SD family's banded im2col and the same f16-weight GEMM.
    pub fn conv3x3(d: Dev, patch: *Buf, dst: *Buf, src: *Buf, h: usize, w: usize, cv: sensenova.Conv) !void {
        std.debug.assert(cv.k == 3 and cv.stride == 1 and cv.pad == 1);
        try conv_gpu.conv2d(d.ctx, patch, dst, src, h, w, cv, .stride1, null, null, 1.0);
    }

    pub fn trace(d: Dev, what: []const u8, i: usize, buf: Buf, elems: usize) void {
        traceBuf(d.ctx, what, i, buf, elems);
    }
};

/// DIAGNOSTIC: the same stage lines `sensenova.traceActs` prints on the host
/// (`TP_SENSENOVA_TRACE`), so the two runs diff. Costs a sync and a download per call,
/// so it is env-gated and never on in a render.
var trace_on: ?bool = null;

fn traceBuf(ctx: *Context, what: []const u8, i: usize, buf: Buf, elems: usize) void {
    if (trace_on == null) trace_on = std.c.getenv("TP_SENSENOVA_TRACE") != null;
    if (!trace_on.?) return;
    const host = ctx.gpa.alloc(f32, elems) catch return;
    defer ctx.gpa.free(host);
    const batching = ctx.batching;
    if (batching) ctx.endBatch() catch return;
    ctx.downloadF32At(buf, 0, host) catch return;
    var mx: f32 = 0;
    var sum: f64 = 0;
    var bad: usize = 0;
    for (host) |v| {
        if (!std.math.isFinite(v)) bad += 1 else mx = @max(mx, @abs(v));
        sum += @as(f64, v) * @as(f64, v);
    }
    std.debug.print("[sensenova-vk ] {s:<10} {d:>2}  max|x| {d:12.4}  rms {d:10.5}  n {d}  nonfinite {d}\n", .{ what, i, mx, @sqrt(sum / @as(f64, @floatFromInt(elems))), elems, bad });
    if (batching) ctx.beginBatch() catch return;
}
