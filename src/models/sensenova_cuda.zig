//! GPU-resident SenseNova U1.5 on the CUDA backends (`zig-cuda`'s hand-PTX and
//! `cuda`'s vendor libraries), the device twin of `models/sensenova.zig`.
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
const lin_cuda = @import("lin_cuda.zig");
const lora_cuda = @import("lora_cuda.zig");
const cuda = @import("tp_gpu").cuda;
const ops = @import("tp_ops");

const Model = sensenova.Model;
const Config = sensenova.Config;
const Prefix = sensenova.Prefix;
const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;
const Weight = ops.matmul.Weight;

/// Force the naive one-thread-per-(query, head) attention instead of the
/// tensor-core path. For A/B and for reproducing a mismatch; the device test runs
/// both.
pub var force_naive_attn: bool = false;

/// Cap on one im2col band, so a 3x3 over a 1024-wide canvas does not materialize a
/// [4096][9216] patch in one allocation.
const patch_band_bytes: usize = 64 << 20;

/// Whether the CUDA arms can run every linear this model has, in BOTH streams. The
/// refusal is logged by name; the caller says the trunk then runs on the CPU.
///
/// Both lists, not just the generation one: a checkpoint that quantized the two MoT
/// copies differently would otherwise build a session and fail inside the encode.
pub fn supported(model: *const Model) bool {
    if (model.layers.len == 0) return false;
    _ = lin_cuda.plan(model.device_lins, lin_cuda.blockq_gemm, "sensenova cuda") catch return false;
    _ = lin_cuda.plan(model.prefix_lins, lin_cuda.blockq_gemm, "sensenova cuda prefix") catch return false;
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

/// A rotate-half table flattened as `cos` then `sin`, the layout every rope kernel
/// here reads with `sin_off = rows * half`.
const Table = struct {
    buf: Buf,
    half: usize,
    sin_off: usize,

    fn init(be: *Backend, gpa: std.mem.Allocator, rows: usize, span: usize, theta: f64) !Table {
        var f = try ops.rope.rotateHalfFreqs(gpa, rows, span, theta);
        defer f.deinit(gpa);
        return upload(be, gpa, f, rows);
    }

    fn initInterleaved(be: *Backend, gpa: std.mem.Allocator, rows: usize, dim: usize, theta: f64) !Table {
        const pos = try gpa.alloc(f32, rows);
        defer gpa.free(pos);
        for (pos, 0..) |*p, i| p.* = @floatFromInt(i);
        var f = try ops.rope.fluxFreqs(gpa, pos, &.{dim}, theta);
        defer f.deinit(gpa);
        return upload(be, gpa, f, rows);
    }

    fn upload(be: *Backend, gpa: std.mem.Allocator, f: ops.rope.Freqs, rows: usize) !Table {
        const flat = try gpa.alloc(f32, 2 * rows * f.half);
        defer gpa.free(flat);
        @memcpy(flat[0 .. rows * f.half], f.cos[0 .. rows * f.half]);
        @memcpy(flat[rows * f.half ..], f.sin[0 .. rows * f.half]);
        const buf = try be.tensorCreate(flat.len * 4);
        errdefer be.tensorDestroy(@constCast(&buf));
        try be.tensorUpload(buf, std.mem.sliceAsBytes(flat));
        return .{ .buf = buf, .half = f.half, .sin_off = rows * f.half };
    }

    fn deinit(self: *Table, be: *Backend) void {
        be.tensorDestroy(&self.buf);
    }
};

/// The per-head norms and the three-span rope, applied to a q or k buffer in place.
///
/// `norm_w` is the layer's `[q_norm ++ q_norm_hw]` pair as one 128-wide vector, so
/// `groupRmsNorm` normalizes each half of the head against its own scale in one
/// launch: the two spans are contiguous and equally wide, which is exactly the
/// group form.
fn qkPrepare(
    be: *Backend,
    cfg: Config,
    x: Buf,
    rows: usize,
    heads: usize,
    norm_w: Buf,
    pos: [3]Buf,
    tables: [3]Table,
) !void {
    try be.groupRmsNorm(x, x, norm_w, rows * heads, cfg.head_dim, 2, cfg.norm_eps);
    const t_span = cfg.spanT();
    const hw = cfg.spanHw();
    const offs = [3]usize{ 0, t_span, t_span + hw };
    inline for (0..3) |i| {
        try be.opRopeHalfSpanPos(x, pos[i], tables[i].buf, rows, heads, tables[i].half, tables[i].sin_off, cfg.head_dim, offs[i]);
    }
}

/// A non-overlapping k x k stride-k convolution over channel-last `[h][w][ci]`:
/// gather the patch matrix, then one GEMM. `dst` is `[h/k * w/k][co]`.
fn convStride(be: *Backend, patch: *Buf, dst: Buf, src: Buf, h: usize, w: usize, cv: sensenova.Conv) !void {
    const n_out = (h / cv.k) * (w / cv.k);
    const plen = cv.k * cv.k * cv.ci;
    try be.ensureDeviceBuffer(patch, n_out * plen * 4);
    try be.opIm2colStride(patch.*, src, h, w, cv.ci, cv.k);
    try be.opConvF16(dst, 0, patch.*, n_out, std.mem.sliceAsBytes(cv.w), cv.co, plen, cv.b.?);
}

/// A 3x3 pad-1 stride-1 convolution over channel-last `[h][w][ci]`. cuDNN when the
/// vendor libraries are loaded, else the SD family's banded im2col plus the same
/// f16 GEMM.
fn conv3x3(be: *Backend, patch: *Buf, dst: Buf, src: Buf, h: usize, w: usize, cv: sensenova.Conv) !void {
    std.debug.assert(cv.k == 3 and cv.stride == 1 and cv.pad == 1);
    const bias = cv.b.?;
    if (be.kernels == .libs and ops.matmul.probe == null) {
        return be.opConvCudnn(dst, 0, src, h, w, std.mem.sliceAsBytes(cv.w), cv.co, cv.ci, bias, false, false);
    }
    const n_out = h * w;
    const plen = 9 * cv.ci;
    const band = @max(4, @min(n_out, patch_band_bytes / (plen * 4)) & ~@as(usize, 3));
    try be.ensureDeviceBuffer(patch, band * plen * 4);
    var p0: usize = 0;
    while (p0 < n_out) : (p0 += band) {
        const bn = @min(band, n_out - p0);
        try be.opIm2colSd(src, patch.*, bn, plen, cv.ci, w, h, p0, w, 0, false);
        try be.opConvF16(dst, p0 * cv.co, patch.*, bn, std.mem.sliceAsBytes(cv.w), cv.co, plen, bias);
    }
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
    cancel: ?*std.atomic.Value(bool),
) !Prefix {
    const cfg = model.cfg;
    const seq = layout.ids.len;
    std.debug.assert(seq > 0);
    var plan = try lin_cuda.plan(model.prefix_lins, lin_cuda.blockq_gemm, "sensenova cuda prefix");
    try lin_cuda.presize(be, plan, model.prefix_lins);
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

    // The position arrays are `usize` on the host and `u32` on the device; the
    // tables are sized to the largest position each axis actually reaches, which
    // is 1 row for the hw axes of a prompt with no pictures in it.
    const pos32 = try gpa.alloc(u32, 3 * seq);
    defer gpa.free(pos32);
    var maxpos = [3]usize{ 0, 0, 0 };
    for ([3][]const usize{ layout.pos_t, layout.pos_h, layout.pos_w }, 0..) |src, axis| {
        for (src, 0..) |v, i| {
            pos32[axis * seq + i] = @intCast(v);
            maxpos[axis] = @max(maxpos[axis], v);
        }
    }

    var tab_t = try Table.init(be, gpa, maxpos[0] + 1, cfg.spanT(), cfg.theta_t);
    defer tab_t.deinit(be);
    var tab_h = try Table.init(be, gpa, maxpos[1] + 1, cfg.spanHw(), cfg.theta_hw);
    defer tab_h.deinit(be);
    var tab_w = try Table.init(be, gpa, maxpos[2] + 1, cfg.spanHw(), cfg.theta_hw);
    defer tab_w.deinit(be);
    var pos_bufs = [3]Buf{
        try uploadU32(be, pos32[0..seq]),
        try uploadU32(be, pos32[seq..][0..seq]),
        try uploadU32(be, pos32[2 * seq ..][0..seq]),
    };
    defer for (&pos_bufs) |*p| be.tensorDestroy(p);
    const tables = [3]Table{ tab_t, tab_h, tab_w };

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
        try be.tensorUpload(ws.x, std.mem.sliceAsBytes(emb));
    }

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));
    const kv_bytes = seq * cfg.kvDim() * 4;
    for (model.layers, 0..) |*layer, li| {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        if (be.async_uploads and li + 1 < model.layers.len) prefetchStream(be, &model.layers[li + 1].base);
        try be.beginBatch();
        errdefer if (be.batching()) be.abortBatch();
        try layerForward(be, plan, cfg, &layer.base, .{
            .x = ws.x,
            .nrm = ws.nrm,
            .dlt = ws.dlt,
            .q = ws.q,
            .k = ws.k,
            .v = ws.v,
            .kcat = ws.k,
            .vcat = ws.v,
            .attn = ws.attn,
            .mg = ws.mg,
            .mu = ws.mu,
        }, seq, seq, .{
            .pos = pos_bufs,
            .tables = tables,
            .norm_q = try smallBuf(be, layer.base.q_norm_pair),
            .norm_k = try smallBuf(be, layer.base.k_norm_pair),
            .scale = scale,
            .causal = true,
            .bounds = bounds,
            .prefix_rows = 0,
        });
        try be.endBatch();
        // Down per layer rather than at the end: the whole cache is 172 KB per
        // token per layer, and holding 42 layers of it on the device as well as on
        // the host would cost the generation stack the room it needs.
        try be.tensorDownload(offsetBuf(ws.k, 0, kv_bytes), std.mem.sliceAsBytes(pre.kv[(li * 2) * seq * cfg.kvDim() ..][0 .. seq * cfg.kvDim()]));
        try be.tensorDownload(offsetBuf(ws.v, 0, kv_bytes), std.mem.sliceAsBytes(pre.kv[(li * 2 + 1) * seq * cfg.kvDim() ..][0 .. seq * cfg.kvDim()]));
    }
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
            .plan = try lin_cuda.plan(model.device_lins, lin_cuda.blockq_gemm, "sensenova cuda"),
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
        self.tab_t = try Table.init(be, gpa, pre.time + 1, cfg.spanT(), cfg.theta_t);
        made += 1;
        self.tab_h = try Table.init(be, gpa, th, cfg.spanHw(), cfg.theta_hw);
        made += 1;
        self.tab_w = try Table.init(be, gpa, tw, cfg.spanHw(), cfg.theta_hw);
        made += 1;
        self.tab_vx = try Table.initInterleaved(be, gpa, gw, cfg.vis_dim / 2, cfg.theta_vis);
        made += 1;
        self.tab_vy = try Table.initInterleaved(be, gpa, gh, cfg.vis_dim / 2, cfg.theta_vis);
        made += 1;

        // Position arrays get device buffers of their own rather than going through
        // `smallBuffer`: that cache is keyed on the HOST POINTER, so a scratch array
        // freed here whose address is later recycled would be served this upload.
        {
            const t = try gpa.alloc(u32, n_img);
            defer gpa.free(t);
            const h = try gpa.alloc(u32, n_img);
            defer gpa.free(h);
            const w = try gpa.alloc(u32, n_img);
            defer gpa.free(w);
            for (0..n_img) |i| {
                t[i] = pre.time;
                h[i] = @intCast(i / tw);
                w[i] = @intCast(i % tw);
            }
            self.pos_t = try uploadU32(be, t);
            made += 1;
            self.pos_h = try uploadU32(be, h);
            made += 1;
            self.pos_w = try uploadU32(be, w);
            made += 1;
        }
        {
            const x = try gpa.alloc(u32, gh * gw);
            defer gpa.free(x);
            const y = try gpa.alloc(u32, gh * gw);
            defer gpa.free(y);
            for (0..gh * gw) |i| {
                x[i] = @intCast(i % gw);
                y[i] = @intCast(i / gw);
            }
            self.vis_x = try uploadU32(be, x);
            made += 1;
            self.vis_y = try uploadU32(be, y);
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

    const fields = [_][]const u8{ "patch", "cells", "img", "nrm", "dlt", "q", "k", "v", "kcat", "vcat", "attn", "mg", "mu", "fm_a", "fm_b", "tvec" };

    /// `prefix_cap` is the LONGEST prefix any branch sharing this workspace has,
    /// not the positive branch's. Under classifier-free guidance the two branches
    /// are different lengths (an edit's negative presents the pictures with no
    /// prompt, so it is shorter; a plain negative is shorter still), and sizing to
    /// one of them writes the other off the end of `kcat`.
    pub fn init(be: *Backend, model: *const Model, sess: *const Session, prefix_cap: usize) !Workspace {
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

    // The patch matrix of the 16x16 embedder is built on the host: it reads the
    // canvas, which arrives on the host anyway, and it is a pure gather that would
    // otherwise cost an upload of the canvas plus a kernel for the same bytes.
    {
        const plen = cfg.patch * cfg.patch * ch;
        const patch = try gpa.alloc(f32, sess.gh * sess.gw * plen);
        defer gpa.free(patch);
        for (0..sess.gh) |cy| {
            for (0..sess.gw) |cx| {
                const dst = patch[(cy * sess.gw + cx) * plen ..][0..plen];
                for (0..cfg.patch) |dy| {
                    for (0..cfg.patch) |dx| {
                        const sy = cy * cfg.patch + dy;
                        const sx = cx * cfg.patch + dx;
                        for (0..ch) |c| {
                            dst[(dy * cfg.patch + dx) * ch + c] = xp[c * sess.ph * sess.pw + sy * sess.pw + sx];
                        }
                    }
                }
            }
        }
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

    // --- the generation vision tower -----------------------------------------
    const cells = sess.gh * sess.gw;
    try be.opConvF16(ws.cells, 0, ws.patch, cells, std.mem.sliceAsBytes(model.vision_gen.patch.w), cfg.vis_dim, cfg.patch * cfg.patch * ch, model.vision_gen.patch.b.?);
    try be.geluErf(ws.cells, cells * cfg.vis_dim);
    const vhalf = cfg.vis_dim / 2;
    try be.opRopeInterSpanPos(ws.cells, sess.vis_x, sess.tab_vx.buf, cells, cfg.vis_dim, sess.tab_vx.half, sess.tab_vx.sin_off, 0);
    try be.opRopeInterSpanPos(ws.cells, sess.vis_y, sess.tab_vy.buf, cells, cfg.vis_dim, sess.tab_vy.half, sess.tab_vy.sin_off, vhalf);
    trace(be, "cells", 0, ws.cells, cells * cfg.vis_dim);
    try convStride(be, &ws.patch, ws.img, ws.cells, sess.gh, sess.gw, model.vision_gen.dense);
    trace(be, "vision", 0, ws.img, sess.n_img * cfg.dim);
    try be.opAddBiasRows(ws.img, ws.tvec, sess.n_img, cfg.dim, 0, false);
    trace(be, "trunk_in", 0, ws.img, sess.n_img * cfg.dim);

    // --- the trunk ------------------------------------------------------------
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));
    const pos = [3]Buf{ sess.pos_t, sess.pos_h, sess.pos_w };
    const tables = [3]Table{ sess.tab_t, sess.tab_h, sess.tab_w };
    const kv_bytes = sess.prefix_seq * cfg.kvDim() * 4;
    for (model.layers, 0..) |*layer, li| {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        if (be.async_uploads and li + 1 < model.layers.len) prefetchStream(be, &model.layers[li + 1].gen);
        // The prefix half of the concatenated K/V is a device-to-device copy from
        // the cache uploaded once per conditioning.
        try be.tensorCopy(ws.kcat, 0, sess.prefix_kv, (li * 2) * kv_bytes, kv_bytes);
        try be.tensorCopy(ws.vcat, 0, sess.prefix_kv, (li * 2 + 1) * kv_bytes, kv_bytes);
        try layerForward(be, sess.plan, cfg, &layer.gen, .{
            .x = ws.img,
            .nrm = ws.nrm,
            .dlt = ws.dlt,
            .q = ws.q,
            .k = ws.k,
            .v = ws.v,
            .kcat = ws.kcat,
            .vcat = ws.vcat,
            .attn = ws.attn,
            .mg = ws.mg,
            .mu = ws.mu,
        }, sess.n_img, sess.prefix_seq + sess.n_img, .{
            .pos = pos,
            .tables = tables,
            .norm_q = try smallBuf(be, layer.gen.q_norm_pair),
            .norm_k = try smallBuf(be, layer.gen.k_norm_pair),
            .scale = scale,
            .causal = false,
            .prefix_rows = sess.prefix_seq,
        });
        trace(be, "layer", li, ws.img, sess.n_img * cfg.dim);
    }
    try be.qkNorm(ws.img, ws.img, try smallBuf(be, model.norm_gen), sess.n_img, cfg.dim, cfg.norm_eps);
    trace(be, "norm", 0, ws.img, sess.n_img * cfg.dim);

    // --- fm_head ---------------------------------------------------------------
    // The token plane IS the channel-last input the first shuffle reads.
    try be.opPixelShuffle(ws.fm_a, ws.img, sess.th, sess.tw, cfg.fmC1(), 2);
    try conv3x3(be, &ws.patch, ws.fm_b, ws.fm_a, sess.th * 2, sess.tw * 2, model.fm1);
    try be.geluErf(ws.fm_b, sess.th * 2 * sess.tw * 2 * cfg.fmC1());
    try be.opPixelShuffle(ws.fm_a, ws.fm_b, sess.th * 2, sess.tw * 2, cfg.fmC2(), 2);
    try conv3x3(be, &ws.patch, ws.fm_b, ws.fm_a, sess.th * 4, sess.tw * 4, model.fm2);
    try be.opPixelShuffle(ws.fm_a, ws.fm_b, sess.th * 4, sess.tw * 4, ch, 8);
    trace(be, "pred", 0, ws.fm_a, sess.ph * sess.pw * ch);
    try be.endBatch();

    // The x0-to-velocity conversion and the crop are three planes of arithmetic;
    // they ride back with the download rather than costing a kernel.
    const rgb = try gpa.alloc(f32, sess.ph * sess.pw * ch);
    defer gpa.free(rgb);
    try be.tensorDownload(offsetBuf(ws.fm_a, 0, rgb.len * 4), std.mem.sliceAsBytes(rgb));
    const denom = @max(1.0 - t, cfg.velocity_eps);
    for (0..ch) |c| {
        const src = xp[c * sess.ph * sess.pw ..];
        const dst = out[c * sess.lat_h * sess.lat_w ..];
        for (0..sess.lat_h) |y| {
            for (0..sess.lat_w) |x| {
                dst[y * sess.lat_w + x] = (src[y * sess.pw + x] - rgb[(y * sess.pw + x) * ch + c]) / denom;
            }
        }
    }
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

const LayerArgs = struct {
    pos: [3]Buf,
    tables: [3]Table,
    norm_q: Buf,
    norm_k: Buf,
    scale: f32,
    causal: bool,
    /// The block-causal per-query key range of a prompt carrying reference
    /// pictures, as `opAttnBatched` wants it; null is plain `causal`.
    bounds: ?Buf = null,
    /// Rows of `kcat`/`vcat` already filled with the prefix cache; 0 for the
    /// prefix pass, which attends over itself alone.
    prefix_rows: usize,
};

fn layerForward(
    be: *Backend,
    plan: lin_cuda.Plan,
    cfg: Config,
    s: *const sensenova.Stream,
    b: LayerBufs,
    rows: usize,
    seq_kv: usize,
    a: LayerArgs,
) !void {
    const dim = cfg.dim;
    const kv_dim = cfg.kvDim();

    try be.qkNorm(b.x, b.nrm, try smallBuf(be, s.in_norm), rows, dim, cfg.norm_eps);
    try lin_cuda.prep(be, plan, b.nrm, rows, dim, &.{ s.q, s.k, s.v }, false);
    try lin_cuda.gemm(be, plan, b.q, b.nrm, rows, s.q, false);
    try lin_cuda.gemm(be, plan, b.k, b.nrm, rows, s.k, false);
    try lin_cuda.gemm(be, plan, b.v, b.nrm, rows, s.v, false);

    try qkPrepare(be, cfg, b.q, rows, cfg.n_heads, a.norm_q, a.pos, a.tables);
    try qkPrepare(be, cfg, b.k, rows, cfg.n_kv_heads, a.norm_k, a.pos, a.tables);
    trace(be, "q", 0, b.q, rows * cfg.qDim());
    trace(be, "k", 0, b.k, rows * kv_dim);
    trace(be, "v", 0, b.v, rows * kv_dim);

    // The generation stream attends over the prefix's keys and then its own, with
    // no mask; the prefix pass attends causally over itself alone.
    if (a.prefix_rows != 0) {
        try be.tensorCopy(b.kcat, a.prefix_rows * kv_dim * 4, b.k, 0, rows * kv_dim * 4);
        try be.tensorCopy(b.vcat, a.prefix_rows * kv_dim * 4, b.v, 0, rows * kv_dim * 4);
    }
    if (a.bounds) |bnd| {
        // No tensor-core form: the block-causal range is per query, and both TC
        // paths take one rule for the whole launch. It runs once per conditioning.
        try be.opAttnBatched(b.q, b.kcat, b.vcat, b.attn, bnd, rows, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, a.scale);
    } else if (force_naive_attn) {
        try be.attn(b.q, b.kcat, b.vcat, b.attn, rows, seq_kv, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, a.scale, a.causal);
    } else if (a.causal) {
        try be.opAttnTCCausal(b.q, b.kcat, b.vcat, b.attn, rows, seq_kv, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, a.scale, 0);
    } else {
        try be.opAttnTCRect(b.q, b.kcat, b.vcat, b.attn, rows, seq_kv, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, a.scale);
    }

    trace(be, "attn", 0, b.attn, rows * cfg.qDim());
    try lin_cuda.prep(be, plan, b.attn, rows, cfg.qDim(), &.{s.o}, false);
    try lin_cuda.gemm(be, plan, b.dlt, b.attn, rows, s.o, false);
    try be.opAdd(b.x, b.dlt, rows * dim);
    trace(be, "attn_res", 0, b.x, rows * dim);

    try be.qkNorm(b.x, b.nrm, try smallBuf(be, s.post_norm), rows, dim, cfg.norm_eps);
    try lin_cuda.prep(be, plan, b.nrm, rows, dim, &.{ s.gate, s.up }, false);
    try lin_cuda.gemm(be, plan, b.mg, b.nrm, rows, s.gate, false);
    try lin_cuda.gemm(be, plan, b.mu, b.nrm, rows, s.up, false);
    try be.siluMul(b.mg, b.mu, rows * cfg.inter);
    try lin_cuda.prep(be, plan, b.mg, rows, cfg.inter, &.{s.down}, false);
    try lin_cuda.gemm(be, plan, b.dlt, b.mg, rows, s.down, false);
    try be.opAdd(b.x, b.dlt, rows * dim);
}

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

fn trace(be: *Backend, what: []const u8, i: usize, buf: Buf, elems: usize) void {
    if (trace_on == null) trace_on = std.c.getenv("TP_SENSENOVA_TRACE") != null;
    if (!trace_on.?) return;
    const host = be.gpa.alloc(f32, elems) catch return;
    defer be.gpa.free(host);
    const batching = be.batching();
    if (batching) be.endBatch() catch return;
    be.tensorDownload(offsetBuf(buf, 0, elems * 4), std.mem.sliceAsBytes(host)) catch return;
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

/// A non-owning device-pointer view at a byte offset, sized to what will be read.
fn offsetBuf(b: Buf, off_bytes: usize, size: usize) Buf {
    return .{ .buf = @enumFromInt(@intFromEnum(b.buf) + off_bytes), .mem = .null_handle, .size = size };
}
