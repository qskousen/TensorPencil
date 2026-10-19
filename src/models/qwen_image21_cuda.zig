//! GPU-resident Qwen-Image 2.1 forward on the CUDA backends (`zig-cuda`'s hand-PTX and
//! `cuda`'s vendor libraries), over `qwen_image21_dev`.
//!
//! Host: the text rows through `txt_in` (once per conditioning) and the modulation
//! tables (the prefix's at t = 0 once, the target's per sigma). Device: the prefix pass
//! on a session's first forward, which leaves every block's K and V in the session;
//! then per step the latent embed, the blocks against that cache, and the head.
//! Attention is `opAttnTCCausal` per text run and `opAttnTCRect` everywhere else.

const std = @import("std");
const qwen_image21 = @import("qwen_image21.zig");
const dev = @import("qwen_image21_dev.zig");
const lin_cuda = @import("lin_cuda.zig");
const cuda = @import("tp_gpu").cuda;
const ops = @import("tp_ops");

const DiT = qwen_image21.DiT;
const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;
const Dt = cuda.backend.Dt;
const Weight = ops.matmul.Weight;
const T = dev.Table;

const eps: f32 = 1e-6;

/// Run the int8 linears weight-only, which the device check holds to the exact CPU
/// forward; the format's own W8A8 is checked by its noise instead.
pub var force_weight_only: bool = false;

/// Every attention through the f32 naive kernel, so a device check can hold the rest
/// of the walk to the exact forward.
pub var force_attn_f32: bool = false;

fn allLins(gpa: std.mem.Allocator, model: *const DiT) ![]Weight {
    return std.mem.concat(gpa, Weight, &.{ model.device_lins, model.edge_lins });
}

pub fn linPlan(gpa: std.mem.Allocator, model: *const DiT) !lin_cuda.Plan {
    const lins = try allLins(gpa, model);
    defer gpa.free(lins);
    var p = try lin_cuda.plan(lins, .{ .blockq = lin_cuda.blockq_gemm }, "qwen-image-2.1 cuda");
    p.act16 = .bf16;
    return p;
}

/// One conditioning: its prefix (text rows and their positions), the joint RoPE table,
/// the tables per sigma, and once primed, every block's prefix keys and values.
pub const Session = struct {
    lat_h: usize,
    lat_w: usize,
    n_img: usize,
    /// Prefix rows: everything before the target.
    pre: usize,
    seq: usize,
    plan: lin_cuda.Plan,
    /// The layout's prefix parts, for the block-causal attention.
    parts: []qwen_image21.Layout.Part,

    /// `[pre][dim]` f32, the text rows through `txt_in`.
    pre_d: Buf,
    /// `[seq][half]` cos, then sin.
    freqs_d: Buf,
    /// `[layers][pre][dim]` f32 keys, then values, once `primed`.
    kcache_d: Buf,
    vcache_d: Buf,
    primed: bool = false,

    /// The prefix's table (t = 0), then one per sigma of the schedule.
    table0: []f32,
    sigmas: []f32,
    tables: []f32,

    pub fn init(
        gpa: std.mem.Allocator,
        io: std.Io,
        be: *Backend,
        model: *const DiT,
        lat_h: usize,
        lat_w: usize,
        ctx: []const f32,
        seq_txt: usize,
        sigmas: []const f32,
    ) !Session {
        const cfg = model.cfg;
        const d = cfg.dim;
        const half = cfg.headDim() / 2;
        var lay = try qwen_image21.Layout.build(gpa, cfg, lat_h, lat_w, seq_txt, &.{});
        defer lay.deinit(gpa);
        const n_img = lat_h * lat_w;
        const pre = lay.seq - n_img;
        const tl = T.len * d;

        var self: Session = .{
            .lat_h = lat_h,
            .lat_w = lat_w,
            .n_img = n_img,
            .pre = pre,
            .seq = lay.seq,
            .plan = blk: {
                const p = try linPlan(gpa, model);
                break :blk if (force_weight_only) lin_cuda.withInt8(p, be, .weight_only_f32) else p;
            },
            .parts = &.{},
            .pre_d = undefined,
            .freqs_d = undefined,
            .kcache_d = undefined,
            .vcache_d = undefined,
            .table0 = &.{},
            .sigmas = &.{},
            .tables = &.{},
        };
        var made: usize = 0;
        errdefer {
            inline for (.{ "pre_d", "freqs_d", "kcache_d", "vcache_d" }, 0..) |name, i| {
                if (made > i) be.tensorDestroy(&@field(self, name));
            }
            gpa.free(self.parts);
            gpa.free(self.table0);
            gpa.free(self.sigmas);
            gpa.free(self.tables);
        }

        const lins = try allLins(gpa, model);
        defer gpa.free(lins);
        try lin_cuda.presize(be, self.plan, lins);
        try lin_cuda.attachLora(be, &self.plan, model.lora, lins, @max(pre, n_img));

        // The target is the last part; everything before it is the prefix.
        self.parts = try gpa.dupe(qwen_image21.Layout.Part, lay.parts[0 .. lay.parts.len - 1]);
        {
            const txt = try model.textTokens(io, gpa, ctx, seq_txt);
            defer gpa.free(txt);
            self.pre_d = try be.tensorCreate(@max(txt.len, 1) * 4);
            made += 1;
            try be.tensorUpload(self.pre_d, std.mem.sliceAsBytes(txt));
        }
        {
            var f = try ops.rope.fluxFreqs(gpa, lay.pos, &cfg.axes, qwen_image21.rope_theta);
            defer f.deinit(gpa);
            const flat = try gpa.alloc(f32, 2 * lay.seq * half);
            defer gpa.free(flat);
            @memcpy(flat[0 .. lay.seq * half], f.cos);
            @memcpy(flat[lay.seq * half ..], f.sin);
            self.freqs_d = try be.tensorCreate(flat.len * 4);
            made += 1;
            try be.tensorUpload(self.freqs_d, std.mem.sliceAsBytes(flat));
        }
        const cache = @max(cfg.layers * pre * d * 4, 4);
        self.kcache_d = try be.tensorCreate(cache);
        made += 1;
        self.vcache_d = try be.tensorCreate(cache);
        made += 1;

        self.table0 = try gpa.alloc(f32, tl);
        {
            var mod = try model.modulationAt(io, gpa, 1.0);
            defer mod.deinit(gpa);
            T.fill(self.table0, mod, true);
        }
        self.sigmas = try gpa.dupe(f32, sigmas);
        self.tables = try gpa.alloc(f32, sigmas.len * tl);
        for (sigmas, 0..) |s, i| try table(model, io, gpa, s, self.tables[i * tl ..][0..tl]);
        return self;
    }

    pub fn deinit(self: *Session, gpa: std.mem.Allocator, be: *Backend) void {
        inline for (.{ "pre_d", "freqs_d", "kcache_d", "vcache_d" }) |name| be.tensorDestroy(&@field(self, name));
        gpa.free(self.parts);
        gpa.free(self.table0);
        gpa.free(self.sigmas);
        gpa.free(self.tables);
        self.* = undefined;
    }

    fn cached(self: *const Session, sigma: f32, len: usize) ?[]const f32 {
        for (self.sigmas, 0..) |s, i| if (s == sigma) return self.tables[i * len ..][0..len];
        return null;
    }
};

fn table(model: *const DiT, io: std.Io, gpa: std.mem.Allocator, sigma: f32, out: []f32) !void {
    var mod = try model.modulationAt(io, gpa, sigma);
    defer mod.deinit(gpa);
    T.fill(out, mod, false);
}

/// Activation buffers, shared by both CFG branches, so sized for the longer prefix.
pub const Workspace = struct {
    x_d: Buf,
    nrm_d: Buf,
    dlt_d: Buf,
    q_d: Buf,
    kj_d: Buf,
    vj_d: Buf,
    attn_d: Buf,
    mg_d: Buf,
    mu_d: Buf,
    mv_d: Buf,
    lat_d: Buf,
    /// The residual, the norm output, the delta and the FFN halves. The attention
    /// block stays f32 (`dev.v_div`).
    act: Dt,

    const fields = [_][]const u8{ "x_d", "nrm_d", "dlt_d", "q_d", "kj_d", "vj_d", "attn_d", "mg_d", "mu_d", "mv_d", "lat_d" };

    pub fn init(be: *Backend, model: *const DiT, n_img: usize, pre_cap: usize, act: Dt) !Workspace {
        const cfg = model.cfg;
        const d = cfg.dim;
        // A quantized route's GEMM writes whole 128-row tiles.
        const rows = std.mem.alignForward(usize, @max(n_img, pre_cap), 128);
        const joint = pre_cap + rows;
        const sizes = [fields.len]usize{
            rows * d * 4,
            rows * d * 4,
            rows * d * 4,
            rows * d * 4,
            joint * d * 4,
            joint * d * 4,
            rows * d * 4,
            rows * cfg.ffn * 4,
            rows * cfg.ffn * 4,
            T.len * d * 4,
            n_img * @max(cfg.in_ch, cfg.out_ch) * 4,
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
        inline for (.{ "x_d", "nrm_d", "dlt_d", "mg_d", "mu_d" }) |name| @field(self, name).dt = act;
        return self;
    }

    pub fn deinit(self: *Workspace, be: *Backend) void {
        inline for (fields) |name| be.tensorDestroy(&@field(self, name));
        self.* = undefined;
    }
};

/// Whether the CUDA arms can run every linear; a refusal is logged by name.
pub fn supported(gpa: std.mem.Allocator, model: *const DiT) bool {
    _ = linPlan(gpa, model) catch return false;
    return lin_cuda.loraRunnable(model.lora, model.device_lins, "qwen-image-2.1 cuda") and
        lin_cuda.loraRunnable(model.lora, model.edge_lins, "qwen-image-2.1 cuda");
}

/// One denoiser forward. `out` / `x_lat` are planar `[ch][lat_h][lat_w]`.
pub fn forward(
    model: *const DiT,
    be: *Backend,
    sess: *Session,
    ws: *Workspace,
    io: std.Io,
    gpa: std.mem.Allocator,
    out: []f32,
    x_lat: []const f32,
    sigma: f32,
    cancel: ?*std.atomic.Value(bool),
) !void {
    const cfg = model.cfg;
    const d = cfg.dim;
    const n_img = sess.n_img;
    std.debug.assert(x_lat.len == cfg.in_ch * n_img and out.len == cfg.out_ch * n_img);

    if (!sess.primed and sess.pre > 0) {
        try be.tensorUpload(ws.mv_d, std.mem.sliceAsBytes(sess.table0));
        try be.beginBatch();
        errdefer if (be.batching()) be.abortBatch();
        try toAct(be, sess.pre_d, ws.x_d, sess.pre * d, ws.act);
        try dev.prefix(Dev{ .model = model, .be = be, .sess = sess, .ws = ws }, model, cancel);
        try be.endBatch();
    }
    sess.primed = true;

    const tl = T.len * d;
    var owned: ?[]f32 = null;
    defer if (owned) |b| gpa.free(b);
    const tbl = sess.cached(sigma, tl) orelse blk: {
        owned = try gpa.alloc(f32, tl);
        try table(model, io, gpa, sigma, owned.?);
        break :blk owned.?;
    };
    const tokens = try gpa.alloc(f32, n_img * @max(cfg.in_ch, cfg.out_ch));
    defer gpa.free(tokens);
    qwen_image21.interleaveChannels(tokens[0..x_lat.len], x_lat, n_img, cfg.in_ch);
    try be.tensorUpload(ws.mv_d, std.mem.sliceAsBytes(tbl));
    try be.tensorUpload(ws.lat_d, std.mem.sliceAsBytes(tokens[0..x_lat.len]));

    try be.beginBatch();
    errdefer if (be.batching()) be.abortBatch();
    try dev.target(Dev{ .model = model, .be = be, .sess = sess, .ws = ws }, model, cancel);
    try be.endBatch();
    if (ws.act != .f32 and !try be.allFinite(ws.x_d, n_img * d)) return error.NonFiniteActivation;

    try be.downloadF32(ws.lat_d.slice(0, out.len * 4), tokens[0..out.len]);
    qwen_image21.deinterleaveChannels(out, tokens[0..out.len], n_img, cfg.out_ch);
}

fn toAct(be: *Backend, src: Buf, dst: Buf, n: usize, act: Dt) !void {
    if (act == .f32) try be.tensorCopy(dst, 0, src, 0, n * 4) else try be.opCast(src, dst, n, 0);
}

/// `qwen_image21_dev`'s ops on this backend.
const Dev = struct {
    model: *const DiT,
    be: *Backend,
    sess: *Session,
    ws: *Workspace,

    pub fn rows(d: Dev, pass: dev.Pass) usize {
        return if (pass == .prefix) d.sess.pre else d.sess.n_img;
    }

    /// Where a pass's keys or values land in the joint buffer: the prefix at row 0,
    /// the target after it.
    pub fn kvRows(d: Dev, joint: Buf, pass: dev.Pass) !Buf {
        const off = if (pass == .prefix) 0 else d.sess.pre * d.model.cfg.dim * 4;
        return joint.slice(off, joint.size - off);
    }

    pub fn embed(d: Dev) !void {
        try lin_cuda.linear(d.be, d.sess.plan, d.ws.x_d, d.ws.lat_d, d.sess.n_img, d.model.img_in.w, null);
    }

    /// Prefetch one block ahead, so each upload overlaps the previous block's compute.
    pub fn blockBegin(d: Dev, i: usize) !void {
        if (!d.be.async_uploads) return;
        const blocks = d.model.blocks;
        if (i == 0) prefetchBlock(d.be, &blocks[0]);
        if (i + 1 < blocks.len) prefetchBlock(d.be, &blocks[i + 1]);
    }

    pub fn modNorm(d: Dev, x: Buf, nrm: Buf, n: usize, premul: usize) !void {
        const dim = d.model.cfg.dim;
        try d.be.lnMod(x, nrm, d.ws.mv_d, n, dim, premul * dim, T.zero * dim, eps);
    }

    pub fn prep(d: Dev, x: Buf, m: usize, cols: usize, group: []const Weight) !lin_cuda.Group {
        return lin_cuda.prep(d.be, d.sess.plan, x, m, cols, group);
    }

    pub fn gemms(d: Dev, g: lin_cuda.Group, ys: []const Buf, group: []const Weight) !void {
        for (ys, group) |y, w| try lin_cuda.gemm(d.be, d.sess.plan, g, y, w, null);
    }

    pub fn linear(d: Dev, y: Buf, x: Buf, m: usize, w: Weight) !void {
        try lin_cuda.linear(d.be, d.sess.plan, y, x, m, w, null);
    }

    /// The per-head norms, then interleaved RoPE at the pass's own positions.
    pub fn qkNormRope(d: Dev, qnorm: []const f32, knorm: []const f32, k: Buf, n: usize, pass: dev.Pass) !void {
        const cfg = d.model.cfg;
        const hd = cfg.headDim();
        const half = hd / 2;
        try d.be.qkNorm(d.ws.q_d, d.ws.q_d, try normBuf(d.be, qnorm), n * cfg.heads, hd, eps);
        try d.be.qkNorm(k, k, try normBuf(d.be, knorm), n * cfg.heads, hd, eps);
        const row0 = if (pass == .prefix) 0 else d.sess.pre;
        const f = d.sess.freqs_d.slice(row0 * half * 4, d.sess.freqs_d.size - row0 * half * 4);
        try d.be.rope(d.ws.q_d, f, n, cfg.heads, half, d.sess.seq * half);
        try d.be.rope(k, f, n, cfg.heads, half, d.sess.seq * half);
    }

    pub fn scale(d: Dev, buf: Buf, n: usize, f: f32) !void {
        try d.be.opScale(buf, f, n);
    }

    /// Prefix: one call per part, causal over a text run and through its end for an
    /// image block, then the block's keys and values into the cache. Target: the cache
    /// back in front of its own rows, one unmasked call over all of them.
    pub fn attention(d: Dev, i: usize, pass: dev.Pass) !void {
        const cfg = d.model.cfg;
        const hd = cfg.headDim();
        const dim = cfg.dim;
        const sc: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        const be = d.be;
        const ws = d.ws;
        const pre = d.sess.pre;
        const bytes = pre * dim * 4;
        switch (pass) {
            .prefix => {
                for (d.sess.parts) |p| {
                    const q = ws.q_d.slice(p.row * dim * 4, p.n * dim * 4);
                    const o = ws.attn_d.slice(p.row * dim * 4, p.n * dim * 4);
                    const causal = p.kind == .text;
                    if (force_attn_f32)
                        try be.attn(q, ws.kj_d, ws.vj_d, o, p.n, p.row + p.n, cfg.heads, cfg.heads, hd, sc, causal)
                    else if (causal)
                        try be.opAttnTCCausal(q, ws.kj_d, ws.vj_d, o, p.n, p.row + p.n, cfg.heads, cfg.heads, hd, sc, p.row)
                    else
                        try be.opAttnTCRect(q, ws.kj_d, ws.vj_d, o, p.n, p.row + p.n, cfg.heads, cfg.heads, hd, sc);
                }
                try be.tensorCopy(d.sess.kcache_d, i * bytes, ws.kj_d, 0, bytes);
                try be.tensorCopy(d.sess.vcache_d, i * bytes, ws.vj_d, 0, bytes);
            },
            .target => {
                if (pre > 0) {
                    try be.tensorCopy(ws.kj_d, 0, d.sess.kcache_d, i * bytes, bytes);
                    try be.tensorCopy(ws.vj_d, 0, d.sess.vcache_d, i * bytes, bytes);
                }
                const n = d.sess.n_img;
                if (force_attn_f32)
                    try be.attn(ws.q_d, ws.kj_d, ws.vj_d, ws.attn_d, n, pre + n, cfg.heads, cfg.heads, hd, sc, false)
                else
                    try be.opAttnTCRect(ws.q_d, ws.kj_d, ws.vj_d, ws.attn_d, n, pre + n, cfg.heads, cfg.heads, hd, sc);
            },
        }
    }

    pub fn gatedAdd(d: Dev, x: Buf, t: Buf, n: usize, gate: usize) !void {
        const dim = d.model.cfg.dim;
        try d.be.gatedAdd(x, t, d.ws.mv_d, n * dim, dim, gate * dim);
    }

    pub fn siluMul(d: Dev, gate: Buf, up: Buf, n: usize) !void {
        try d.be.siluMul(gate, up, n);
    }

    /// The final norm at the head's scale, then `proj_out` into `lat_d`, whose tokens
    /// the embed has finished with.
    pub fn head(d: Dev) !void {
        const n_img = d.sess.n_img;
        try d.modNorm(d.ws.x_d, d.ws.nrm_d, n_img, T.head_premul);
        try lin_cuda.linear(d.be, d.sess.plan, d.ws.lat_d, d.ws.nrm_d, n_img, d.model.proj_out.w, null);
    }
};

/// Keys must be the same host pointers the forward fetches, hence the row views.
fn prefetchBlock(be: *Backend, blk: *const qwen_image21.Block) void {
    for (blk.dev) |w| be.prefetchWeight(w.bytes);
    be.prefetchWeight(std.mem.sliceAsBytes(blk.q_norm));
    be.prefetchWeight(std.mem.sliceAsBytes(blk.k_norm));
}

fn normBuf(be: *Backend, weights: []const f32) !Buf {
    const h = try be.smallBuffer(std.mem.sliceAsBytes(weights));
    return .{ .buf = h, .size = weights.len * 4 };
}
