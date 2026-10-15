//! GPU-resident SenseNova U1.5 on the CUDA backends (`zig-cuda`'s hand-PTX and
//! `cuda`'s vendor libraries), the device twin of `models/sensenova.zig`. The passes
//! themselves are `sensenova_dev.zig`; this is their adapter (`Dev`) and what the
//! backend owns.
//!
//! Both passes run here, and they are different shapes rather than one shape twice:
//!
//! - the PREFIX pass runs the base MoT copy over the prompt, causally, once per
//!   conditioning, and keeps every layer's keys and values. Its weights are a
//!   second full 8B stack that no denoise step touches, so they are streamed
//!   rather than pinned (the H3 text encoder's pattern);
//! - the GENERATION pass runs the `_mot_gen` copy over the canvas tokens every
//!   step, unmasked, over the prefix KV concatenated with its own.
//!
//! The per-head norms and the RoPE both work on SPANS of the head rather than the
//! whole of it (64 for the sequence position, then 32 + 32 for the token's row and
//! column), which is why `groupRmsNorm` and `opRopeHalfSpanPos` are used where the
//! other families call `qkNorm` and `rope`.
//!
//! `fm_head` and the vision patch embedder are convolutions, and the two kinds are
//! handled differently on purpose: the 3x3s go through the SD family's banded
//! `im2col_sd` (or cuDNN), while the patch and merge convolutions are
//! non-overlapping and so are a pure gather (`opIm2colStride`) into one GEMM.

const std = @import("std");
const sensenova = @import("sensenova.zig");
const conv_cuda = @import("conv_cuda.zig");
const sd_unet_cuda = @import("sd_unet_cuda.zig");
const lin_cuda = @import("lin_cuda.zig");
const sensenova_dev = @import("sensenova_dev.zig");
const lora_cuda = @import("lora_cuda.zig");
const cuda = @import("tp_gpu").cuda;
const ops = @import("tp_ops");

const Model = sensenova.Model;
const Config = sensenova.Config;
const Prefix = sensenova.Prefix;
const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;
const Dt = cuda.backend.Dt;
const Weight = ops.matmul.Weight;

/// Force the naive one-thread-per-(query, head) attention instead of the
/// tensor-core path. For A/B and for reproducing a mismatch; the device test runs
/// both.
pub var force_naive_attn: bool = false;

/// Run integer weights weight-only, so the device check can split a W8A8 forward's
/// activation noise from everything else. At f16 (`weight_only_f16`): a bf16 expansion
/// rounds every int8-times-scale weight to 8 bits and read 2x the dense floor at depth.
pub var force_weight_only: bool = false;

/// Whether the CUDA arms can run every linear this model has, in BOTH streams. The
/// refusal is logged by name; the caller says the trunk then runs on the CPU.
///
/// Both lists, not just the generation one: a checkpoint that quantized the two MoT
/// copies differently would otherwise build a session and fail inside the encode.
pub fn supported(model: *const Model) bool {
    if (model.layers.len == 0) return false;
    _ = lin_cuda.plan(model.device_lins, .{ .blockq = lin_cuda.blockq_gemm }, "sensenova cuda") catch return false;
    _ = lin_cuda.plan(model.prefix_lins, .{ .blockq = lin_cuda.encoder_gemm }, "sensenova cuda prefix") catch return false;
    // A sidecar this backend cannot apply makes the whole trunk unsupported, not
    // "supported without the LoRA": running the base GEMMs alone is a different
    // model, silently.
    if (model.lora) |stack| {
        for ([_][]const Weight{ model.device_lins, model.prefix_lins }) |lins| {
            for (lins) |w| for (stack.forWeight(w)) |h| {
                if (!lora_cuda.supported(h.target)) return false;
            };
        }
    }
    return true;
}

/// Build the int8 A of every attached LoRA hit on an int8 linear, over both streams.
/// Call once the stack is resolved against `model`, before the first forward.
pub fn quantizeLora(gpa: std.mem.Allocator, model: *const Model, stack: *@import("lora.zig").Stack) !void {
    try lin_cuda.quantizeLora(gpa, stack, model.device_lins, lin_cuda.blockq_gemm, .w8a8);
    try lin_cuda.quantizeLora(gpa, stack, model.prefix_lins, lin_cuda.encoder_gemm, lin_cuda.dual_lin.encoderInt8());
}

// --- shared pieces ----------------------------------------------------------

/// Wrap a host f32 slice as a (pointer-cached) small device buffer.
fn smallBuf(be: *Backend, values: []const f32) !Buf {
    const h = try be.smallBuffer(std.mem.sliceAsBytes(values));
    return .{ .buf = h, .size = values.len * 4 };
}

/// A device buffer of its own for a u32 position array, owned by the caller.
fn uploadU32(be: *Backend, values: []const u32) !Buf {
    const buf = try be.tensorCreate(values.len * 4);
    errdefer be.tensorDestroy(@constCast(&buf));
    try be.tensorUpload(buf, std.mem.sliceAsBytes(values));
    return buf;
}

/// A `sensenova.FlatRope` on the device.
const Table = struct {
    buf: Buf,
    half: usize,
    sin_off: usize,

    fn init(be: *Backend, gpa: std.mem.Allocator, rows: usize, span: usize, theta: f64, interleaved: bool) !Table {
        const f = try sensenova.FlatRope.init(gpa, rows, span, theta, interleaved);
        defer gpa.free(f.data);
        const buf = try be.tensorCreate(f.data.len * 4);
        errdefer be.tensorDestroy(@constCast(&buf));
        try be.tensorUpload(buf, std.mem.sliceAsBytes(f.data));
        return .{ .buf = buf, .half = f.half, .sin_off = f.sin_off };
    }

    fn deinit(self: *Table, be: *Backend) void {
        be.tensorDestroy(&self.buf);
    }
};


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

    const fields = [_][]const u8{ "x", "nrm", "dlt", "q", "k", "v", "attn", "mg", "mu" };

    fn init(be: *Backend, cfg: Config, seq: usize) !PrefixWs {
        // Padded to 128 rows: a quantized route launches over `align(m, 128)` and
        // each block stores a whole tile, so a buffer sized to the exact row count
        // is written off the end.
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
        };
        var self: PrefixWs = undefined;
        var made: usize = 0;
        errdefer inline for (fields, 0..) |name, i| {
            if (i < made) be.tensorDestroy(&@field(self, name));
        };
        inline for (fields, sizes) |name, size| {
            @field(self, name) = try be.tensorCreate(size);
            made += 1;
        }
        return self;
    }

    fn deinit(self: *PrefixWs, be: *Backend) void {
        inline for (fields) |name| be.tensorDestroy(&@field(self, name));
        self.* = undefined;
    }
};

/// The generation stream's linear plan, without a LoRA.
pub fn canvasPlan(_: *Backend, model: *const Model) !lin_cuda.Plan {
    return lin_cuda.plan(model.device_lins, .{ .blockq = lin_cuda.blockq_gemm }, "sensenova cuda");
}

/// Run the prompt through the base MoT copy and return the prefix KV cache, the
/// conditioning this family's `encode` produces.
///
/// The base stack is a second full 8B of weights that no denoise step reads, so it
/// is prefetched one layer ahead and left to the cache's LRU rather than pinned:
/// keeping it resident would evict the generation stack the render actually runs.
pub fn prefixForward(
    model: *const Model,
    be: *Backend,
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
    // The prompt pass is the conditioning, so it takes the text encoders' policy for
    // both packed kinds: at full depth a q4_k prefix on the int8 decode (which
    // quantizes the activation) is 35% off the exact KV, on f16 0.19%.
    var plan = lin_cuda.withInt8(try lin_cuda.plan(model.prefix_lins, .{ .blockq = lin_cuda.encoder_gemm }, "sensenova cuda prefix"), be, lin_cuda.dual_lin.encoderInt8());
    try lin_cuda.presize(be, plan, model.prefix_lins);
    try lin_cuda.presizeRows(be, plan, model.prefix_lins, seq);
    if (model.lora) |stack| {
        plan.lora = stack;
        try lin_cuda.presizeLora(be, stack, model.prefix_lins, seq);
    }

    var pre: Prefix = .{
        .kv = try gpa.alloc(f32, cfg.n_layers * 2 * seq * cfg.kvDim()),
        .seq = seq,
        .kv_dim = cfg.kvDim(),
        .n_layers = cfg.n_layers,
        .time = layout.time,
    };
    errdefer pre.deinit(gpa);

    var ws = try PrefixWs.init(be, cfg, seq);
    defer ws.deinit(be);
    inline for (.{ "x", "nrm", "mg", "mu" }) |name| @field(ws, name).dt = sensenova_dev.streamDtFor(act);

    var pp = try sensenova.PromptPositions.init(gpa, layout);
    defer gpa.free(pp.pos);
    var tab_t = try Table.init(be, gpa, pp.max[0] + 1, cfg.spanT(), cfg.theta_t, false);
    defer tab_t.deinit(be);
    var tab_h = try Table.init(be, gpa, pp.max[1] + 1, cfg.spanHw(), cfg.theta_hw, false);
    defer tab_h.deinit(be);
    var tab_w = try Table.init(be, gpa, pp.max[2] + 1, cfg.spanHw(), cfg.theta_hw, false);
    defer tab_w.deinit(be);
    var pos_bufs = [3]Buf{ try uploadU32(be, pp.axis(0)), try uploadU32(be, pp.axis(1)), try uploadU32(be, pp.axis(2)) };
    defer for (&pos_bufs) |*p| be.tensorDestroy(p);

    // The block-causal bound, as the batched attention's per-query key range: an
    // image block's tokens see each other in full, so their range runs past their
    // own position to the end of the block.
    var bounds: ?Buf = null;
    defer if (bounds) |*b| be.tensorDestroy(b);
    if (layout.kv_end) |ends| {
        const pair = try gpa.alloc(u32, 2 * seq);
        defer gpa.free(pair);
        @memset(pair[0..seq], 0);
        @memcpy(pair[seq..], ends);
        bounds = try uploadU32(be, pair);
    }

    {
        // The understanding tower runs on the host: it is ~30M parameters against
        // the trunk's 8B, and it runs once per conditioning rather than per step.
        const emb = try model.embedPrompt(io, gpa, layout, refs);
        defer gpa.free(emb);
        try be.tensorUpload(ws.dlt, std.mem.sliceAsBytes(emb));
    }

    try sensenova_dev.prefix(Dev{
        .be = be,
        .plan = plan,
        .cfg = cfg,
        .b = .{ .x = ws.x, .nrm = ws.nrm, .dlt = ws.dlt, .q = ws.q, .k = ws.k, .v = ws.v, .kcat = ws.k, .vcat = ws.v, .attn = ws.attn, .mg = ws.mg, .mu = ws.mu },
        .pos = pos_bufs,
        .tables = .{ tab_t, tab_h, tab_w },
        .causal = true,
        .bounds = bounds,
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
    plan: lin_cuda.Plan,

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

    pub fn init(
        gpa: std.mem.Allocator,
        be: *Backend,
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
        const n_img = th * tw;

        var self: Session = .{
            .cfg = cfg,
            .lat_h = lat_h,
            .lat_w = lat_w,
            .ph = ph,
            .pw = pw,
            .gh = gh,
            .gw = gw,
            .th = th,
            .tw = tw,
            .n_img = n_img,
            .prefix_seq = pre.seq,
            .plan = blk: {
                const p = try canvasPlan(be, model);
                break :blk if (force_weight_only) lin_cuda.withInt8(p, be, .weight_only_f16) else p;
            },
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
            if (made > 0) be.tensorDestroy(&self.prefix_kv);
            if (made > 1) self.tab_t.deinit(be);
            if (made > 2) self.tab_h.deinit(be);
            if (made > 3) self.tab_w.deinit(be);
            if (made > 4) self.tab_vx.deinit(be);
            if (made > 5) self.tab_vy.deinit(be);
            const pos = [_]*Buf{ &self.pos_t, &self.pos_h, &self.pos_w, &self.vis_x, &self.vis_y };
            for (pos[0..(made -| 6)]) |b| be.tensorDestroy(b);
        }

        try lin_cuda.presize(be, self.plan, model.device_lins);
        if (model.lora) |stack| {
            self.plan.lora = stack;
            // The canvas rows, which is what every generation GEMM is called
            // with, so nothing grows the sidecar scratch mid-render.
            try lin_cuda.presizeLora(be, stack, model.device_lins, n_img);
        }
        self.prefix_kv = try be.tensorCreate(pre.kv.len * 4);
        made += 1;
        try be.tensorUpload(self.prefix_kv, std.mem.sliceAsBytes(pre.kv));

        // Every image token takes the same t index, one past the whole prefix.
        self.tab_t = try Table.init(be, gpa, pre.time + 1, cfg.spanT(), cfg.theta_t, false);
        made += 1;
        self.tab_h = try Table.init(be, gpa, th, cfg.spanHw(), cfg.theta_hw, false);
        made += 1;
        self.tab_w = try Table.init(be, gpa, tw, cfg.spanHw(), cfg.theta_hw, false);
        made += 1;
        self.tab_vx = try Table.init(be, gpa, gw, cfg.vis_dim / 2, cfg.theta_vis, true);
        made += 1;
        self.tab_vy = try Table.init(be, gpa, gh, cfg.vis_dim / 2, cfg.theta_vis, true);
        made += 1;

        // Position arrays get device buffers of their own rather than going through
        // `smallBuffer`: that cache is keyed on the HOST POINTER, so a scratch array
        // freed here whose address is later recycled would be served this upload.
        var cp = try sensenova.CanvasPositions.init(gpa, pre.time, tw, n_img, gw, gh * gw);
        defer cp.deinit(gpa);
        inline for (.{ "pos_t", "pos_h", "pos_w", "vis_x", "vis_y" }, .{ .t, .h, .w, .vis_x, .vis_y }) |name, axis| {
            @field(self, name) = try uploadU32(be, cp.axis(axis));
            made += 1;
        }
        return self;
    }

    pub fn deinit(self: *Session, be: *Backend) void {
        be.tensorDestroy(&self.prefix_kv);
        self.tab_t.deinit(be);
        self.tab_h.deinit(be);
        self.tab_w.deinit(be);
        self.tab_vx.deinit(be);
        self.tab_vy.deinit(be);
        inline for (.{ "pos_t", "pos_h", "pos_w", "vis_x", "vis_y" }) |name| be.tensorDestroy(&@field(self, name));
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
    /// The canvas stream's storage (`img`, `nrm`, `mg`, `mu`); `sensenova_dev`.
    act: Dt,

    const fields = [_][]const u8{ "patch", "cells", "img", "nrm", "dlt", "q", "k", "v", "kcat", "vcat", "attn", "mg", "mu", "fm_a", "fm_b", "tvec" };

    /// `prefix_cap` is the LONGEST prefix any branch sharing this workspace has,
    /// not the positive branch's. Under classifier-free guidance the two branches
    /// are different lengths (an edit's negative presents the pictures with no
    /// prompt, so it is shorter; a plain negative is shorter still), and sizing to
    /// one of them writes the other off the end of `kcat`.
    pub fn init(be: *Backend, model: *const Model, sess: *const Session, prefix_cap: usize, act: Dt) !Workspace {
        const cfg = model.cfg;
        const rows = std.mem.alignForward(usize, sess.n_img, 128);
        const kv_rows = std.mem.alignForward(usize, @max(prefix_cap, sess.prefix_seq) + sess.n_img, 128);
        const cells = sess.gh * sess.gw;
        // The fm_head's two intermediates, at their widest: after the first shuffle
        // it is `[2*th][2*tw][dim/4]`, which is the same element count as the token
        // plane, and after the second `[4*th][4*tw][192]`.
        const fm_max = @max(cells * cfg.vis_dim, @max(sess.n_img * cfg.dim, sess.th * 4 * sess.tw * 4 * 192));
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
            rows * cfg.qDim() * 4,
            rows * cfg.inter * 4,
            rows * cfg.inter * 4,
            fm_max * 4,
            fm_max * 4,
            cfg.dim * 4,
        };
        var self: Workspace = undefined;
        var made: usize = 0;
        errdefer inline for (fields, 0..) |name, i| {
            if (i < made) be.tensorDestroy(&@field(self, name));
        };
        inline for (fields, sizes) |name, size| {
            @field(self, name) = try be.tensorCreate(size);
            made += 1;
        }
        self.act = act;
        inline for (.{ "img", "nrm", "mg", "mu" }) |name| @field(self, name).dt = act;
        return self;
    }

    pub fn deinit(self: *Workspace, be: *Backend) void {
        inline for (fields) |name| be.tensorDestroy(&@field(self, name));
        self.* = undefined;
    }
};

/// One denoiser forward. `out` and `x_lat` are planar `[3][lat_h][lat_w]`; `t` is
/// the model's own timestep, `1 - sigma`.
pub fn forward(
    model: *const Model,
    be: *Backend,
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
        try be.ensureDeviceBuffer(&ws.patch, patch.len * 4);
        try be.tensorUpload(ws.patch, std.mem.sliceAsBytes(patch));
    }
    // The scalar conditioning is two 256-wide MLPs; the host runs them in
    // microseconds and the device would need an upload either way.
    {
        const tvec = try model.timeVector(io, gpa, t, sess.ph, sess.pw);
        defer gpa.free(tvec);
        try be.tensorUpload(ws.tvec, std.mem.sliceAsBytes(tvec));
    }

    try be.beginBatch();
    errdefer if (be.batching()) be.abortBatch();
    try sensenova_dev.canvas(Dev{
        .be = be,
        .plan = sess.plan,
        .cfg = cfg,
        .b = .{ .x = ws.img, .nrm = ws.nrm, .dlt = ws.dlt, .q = ws.q, .k = ws.k, .v = ws.v, .kcat = ws.kcat, .vcat = ws.vcat, .attn = ws.attn, .mg = ws.mg, .mu = ws.mu },
        .pos = .{ sess.pos_t, sess.pos_h, sess.pos_w },
        .tables = .{ sess.tab_t, sess.tab_h, sess.tab_w },
        .causal = false,
    }, model, sess, ws, cancel);
    try be.endBatch();
    if (ws.act != .f32 and !try be.allFinite(ws.img, sess.n_img * cfg.dim)) return error.NonFiniteActivation;

    const rgb = try gpa.alloc(f32, sess.ph * sess.pw * ch);
    defer gpa.free(rgb);
    try be.tensorDownload(ws.fm_a.slice(0, rgb.len * 4), std.mem.sliceAsBytes(rgb));
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
};

/// The `sensenova_dev` adapter for one pass.
const Dev = struct {
    be: *Backend,
    plan: lin_cuda.Plan,
    cfg: Config,
    b: LayerBufs,
    pos: [3]Buf,
    tables: [3]Table,
    causal: bool,
    /// The block-causal per-query key range of a prompt carrying reference pictures,
    /// as `opAttnBatched` wants it; null is plain `causal`.
    bounds: ?Buf = null,
    /// The prefix pass's host cache, `[layer][k|v][seq][kv_dim]`.
    kv: []f32 = &.{},

    pub fn rmsNorm(d: Dev, x: Buf, y: Buf, w: []const f32, rows: usize, eps: f32) !void {
        try d.be.qkNorm(x, y, try smallBuf(d.be, w), rows, d.cfg.dim, eps);
    }

    pub fn gemms(d: Dev, x: Buf, rows: usize, cols: usize, ws: []const Weight, ys: []const Buf) !void {
        const g = try lin_cuda.prep(d.be, d.plan, x, rows, cols, ws);
        for (ws, ys) |w, y| try lin_cuda.gemm(d.be, d.plan, g, y, w, null);
    }

    pub fn linear(d: Dev, y: Buf, x: Buf, rows: usize, w: Weight) !void {
        try lin_cuda.linear(d.be, d.plan, y, x, rows, w, null);
    }

    /// `norm_pair` is the layer's `[q_norm ++ q_norm_hw]` as one 128-wide vector, so
    /// `groupRmsNorm` normalizes each half of the head against its own scale in one
    /// launch: the two spans are contiguous and equally wide, which is the group form.
    pub fn qkPrepare(d: Dev, x: Buf, rows: usize, heads: usize, norm_pair: []const f32) !void {
        const cfg = d.cfg;
        try d.be.groupRmsNorm(x, x, try smallBuf(d.be, norm_pair), rows * heads, cfg.head_dim, 2, cfg.norm_eps);
        const offs = [3]usize{ 0, cfg.spanT(), cfg.spanT() + cfg.spanHw() };
        for (d.pos, d.tables, offs) |p, tb, off| {
            try d.be.opRopeHalfSpanPos(x, p, tb.buf, rows, heads, tb.half, tb.sin_off, cfg.head_dim, off);
        }
    }

    /// Casts when the two dtypes differ, which happens only at the trunk's ends.
    pub fn copy(d: Dev, dst: Buf, dst_off: usize, src: Buf, src_off: usize, n: usize) !void {
        if (dst.dt == src.dt) {
            const eb = dst.dt.bytes();
            return d.be.tensorCopy(dst, dst_off * eb, src, src_off * eb, n * eb);
        }
        std.debug.assert(src_off == 0);
        try d.be.opCast(src, dst, n, dst_off);
    }

    pub fn attention(d: Dev, rows: usize, seq_kv: usize) !void {
        const cfg = d.cfg;
        const b = d.b;
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));
        if (d.bounds) |bnd| {
            // No tensor-core form: the block-causal range is per query, and both TC
            // paths take one rule for the whole launch. It runs once per conditioning.
            try d.be.opAttnBatched(b.q, b.kcat, b.vcat, b.attn, bnd, rows, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, scale);
        } else if (force_naive_attn) {
            try d.be.attn(b.q, b.kcat, b.vcat, b.attn, rows, seq_kv, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, scale, d.causal);
        } else if (d.causal) {
            try d.be.opAttnTCCausal(b.q, b.kcat, b.vcat, b.attn, rows, seq_kv, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, scale, 0);
        } else {
            try d.be.opAttnTCRect(b.q, b.kcat, b.vcat, b.attn, rows, seq_kv, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, scale);
        }
    }

    pub fn add(d: Dev, x: Buf, y: Buf, n: usize) !void {
        try d.be.opAdd(x, y, n);
    }

    pub fn siluMul(d: Dev, g: Buf, u: Buf, n: usize) !void {
        try d.be.siluMul(g, u, n);
    }

    pub fn addScaled(d: Dev, x: Buf, y: Buf, n: usize, s: f32) !void {
        try d.be.opAddScaled(x, y, n, s);
    }

    pub fn scaleBy(d: Dev, x: Buf, s: f32, n: usize) !void {
        try d.be.opScale(x, s, n);
    }

    pub fn gelu(d: Dev, x: Buf, n: usize) !void {
        try d.be.geluErf(x, n);
    }

    /// One layer ahead, so the upload overlaps the previous layer's compute. The base
    /// stack is a second full 8B that no denoise step reads, so it is left to the
    /// cache's LRU rather than pinned: keeping it resident would evict the generation
    /// stack the render actually runs.
    pub fn prefetch(d: Dev, s: *const sensenova.Stream) void {
        if (d.be.async_uploads) prefetchStream(d.be, s);
    }

    pub fn beginBatch(d: Dev) !void {
        try d.be.beginBatch();
    }

    pub fn endBatch(d: Dev) !void {
        try d.be.endBatch();
    }

    pub fn abortBatch(d: Dev) void {
        if (d.be.batching()) d.be.abortBatch();
    }

    pub fn keepKv(d: Dev, li: usize) !void {
        const n = d.kv.len / (d.cfg.n_layers * 2);
        try d.be.tensorDownload(d.b.k.slice(0, n * 4), std.mem.sliceAsBytes(d.kv[(li * 2) * n ..][0..n]));
        try d.be.tensorDownload(d.b.v.slice(0, n * 4), std.mem.sliceAsBytes(d.kv[(li * 2 + 1) * n ..][0..n]));
    }

    pub fn patchEmbed(d: Dev, dst: Buf, src: Buf, rows: usize, cv: sensenova.Conv) !void {
        try conv_cuda.patchGemm(d.be, dst, src, rows, cv);
    }

    pub fn ropeInter(d: Dev, x: Buf, positions: Buf, tb: Table, rows: usize, row_dim: usize, off: usize) !void {
        try d.be.opRopeInterSpanPos(x, positions, tb.buf, rows, row_dim, tb.half, tb.sin_off, off);
    }

    pub fn convStride(d: Dev, patch: *Buf, dst: Buf, src: Buf, h: usize, w: usize, cv: sensenova.Conv) !void {
        try conv_cuda.convStride(d.be, patch, dst, src, h, w, cv);
    }

    pub fn addBiasRows(d: Dev, x: Buf, bias: Buf, rows: usize, cols: usize) !void {
        try d.be.opAddBiasRows(x, bias, rows, cols, 0);
    }

    pub fn pixelShuffle(d: Dev, dst: Buf, src: Buf, h: usize, w: usize, co: usize, r: usize) !void {
        try d.be.opPixelShuffle(dst, src, h, w, co, r);
    }

    pub fn conv3x3(d: Dev, patch: *Buf, dst: *Buf, src: *Buf, h: usize, w: usize, cv: sensenova.Conv) !void {
        std.debug.assert(cv.k == 3 and cv.stride == 1 and cv.pad == 1);
        try conv_cuda.conv2d(d.be, patch, dst, src, h, w, cv, .stride1, null, 1.0);
    }

    pub fn trace(d: Dev, what: []const u8, i: usize, buf: Buf, elems: usize) void {
        traceBuf(d.be, what, i, buf, elems);
    }
};

/// Queue one stream's weights for async prefetch, ONE LAYER AHEAD, so the upload
/// overlaps the previous layer's compute. The keys must be the same host pointers
/// the forward later fetches, or the prefetch is a cache miss and pure waste.
fn prefetchStream(be: *Backend, s: *const sensenova.Stream) void {
    inline for (.{ s.q, s.k, s.v, s.o, s.gate, s.up, s.down }) |w| be.prefetchWeight(w.bytes);
}

/// DIAGNOSTIC: the same stage lines `sensenova.traceActs` prints on the host
/// (`TP_SENSENOVA_TRACE`), so the two runs diff. Costs a sync and a download per
/// call, so it is env-gated and never on in a render.
var trace_on: ?bool = null;

fn traceBuf(be: *Backend, what: []const u8, i: usize, buf: Buf, elems: usize) void {
    if (trace_on == null) trace_on = std.c.getenv("TP_SENSENOVA_TRACE") != null;
    if (!trace_on.?) return;
    const host = be.gpa.alloc(f32, elems) catch return;
    defer be.gpa.free(host);
    const batching = be.batching();
    if (batching) be.endBatch() catch return;
    be.downloadF32(buf, host) catch return;
    var mx: f32 = 0;
    var sum: f64 = 0;
    var bad: usize = 0;
    for (host) |v| {
        if (!std.math.isFinite(v)) bad += 1 else mx = @max(mx, @abs(v));
        sum += @as(f64, v) * @as(f64, v);
    }
    std.debug.print("[sensenova-cu ] {s:<10} {d:>2}  max|x| {d:12.4}  rms {d:10.5}  n {d}  nonfinite {d}\n", .{ what, i, mx, @sqrt(sum / @as(f64, @floatFromInt(elems))), elems, bad });
    if (batching) be.beginBatch() catch return;
}

