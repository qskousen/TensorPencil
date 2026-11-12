//! GPU-resident Fibo 2.0 forward on the CUDA backends (`zig-cuda`'s hand-PTX and
//! `cuda`'s vendor libraries), over `fibo2_dev`.
//!
//! Host: the modulation tables (t = 0's once, the step's per sigma, the norm weights
//! folded in for `rmsMod`), the rope table, the final layer. Device: the prefix pass at
//! `Session.init`, which leaves the gist rows and every injection block's keys and values
//! resident, then per step the embed, refiner and trunk. Self-attention is `opAttnTC`,
//! cross-attention `opAttnTCRect`.

const std = @import("std");
const fibo2 = @import("fibo2.zig");
const dev = @import("fibo2_dev.zig");
const lin_cuda = @import("lin_cuda.zig");
const cuda = @import("tp_gpu").cuda;
const ops = @import("tp_ops");

const DiT = fibo2.DiT;
const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;
const Dt = cuda.backend.Dt;
const Weight = ops.matmul.Weight;

/// Every attention through the naive f32 kernel, so a device check can hold the rest of
/// the walk to the CPU forward.
pub var force_attn_f32: bool = false;
/// Run the integer linears weight-only, which the device check holds to the exact CPU
/// forward; the format's own W8A8 is checked by its noise instead.
pub var force_weight_only: bool = false;

fn allLins(gpa: std.mem.Allocator, model: *const DiT) ![]Weight {
    return std.mem.concat(gpa, Weight, &.{ model.device_lins, model.prefix_lins });
}

pub fn linPlan(gpa: std.mem.Allocator, model: *const DiT) !lin_cuda.Plan {
    const lins = try allLins(gpa, model);
    defer gpa.free(lins);
    var p = try lin_cuda.plan(lins, .{ .blockq = lin_cuda.blockq_gemm }, "fibo2 cuda");
    // Z-Image's block, whose MLP down projection passes f16's range.
    p.act16 = .bf16;
    return p;
}

/// One conditioning at one resolution: its gist rows and cross keys/values, the rope
/// table, and the tables of the schedule.
pub const Session = struct {
    lat_h: usize,
    lat_w: usize,
    n_img: usize,
    n_gist: usize,
    text_len: usize,
    seq: usize,
    plan: lin_cuda.Plan,

    /// `[n_gist][dim]` f32.
    gist_d: Buf,
    /// `[n_inject][text_pad][dim]` f32: keys normed, values divided by `dev.v_div`.
    kc_d: Buf,
    vc_d: Buf,
    text_pad: usize,
    /// `[seq][half]` cos, then sin.
    freqs_d: Buf,
    /// t = 0's table for the trunk, folded.
    mv0_d: Buf,
    zero_len: usize,

    sigmas: []f32,
    /// Per sigma, the step table (folded) and the final scale.
    tables: []f32,
    finals: []f32,
    step_len: usize,

    pub fn init(
        gpa: std.mem.Allocator,
        io: std.Io,
        be: *Backend,
        model: *const DiT,
        lat_h: usize,
        lat_w: usize,
        ctx: []const f32,
        text_len: usize,
        sigmas: []const f32,
        cancel: ?*std.atomic.Value(bool),
    ) !Session {
        const cfg = model.cfg;
        const dim = cfg.dim;
        const half = cfg.head_dim / 2;
        const n_img = lat_h * lat_w;
        const n_gist = cfg.gistCount(text_len);
        const seq = n_gist + n_img;
        const text_pad = std.mem.alignForward(usize, @max(text_len, 1), 128);

        var self: Session = .{
            .lat_h = lat_h,
            .lat_w = lat_w,
            .n_img = n_img,
            .n_gist = n_gist,
            .text_len = text_len,
            .seq = seq,
            .plan = blk: {
                const p = try linPlan(gpa, model);
                break :blk if (force_weight_only) lin_cuda.withInt8(p, be, .weight_only_f32) else p;
            },
            .gist_d = undefined,
            .kc_d = undefined,
            .vc_d = undefined,
            .text_pad = text_pad,
            .freqs_d = undefined,
            .mv0_d = undefined,
            .zero_len = 0,
            .sigmas = &.{},
            .tables = &.{},
            .finals = &.{},
            .step_len = 0,
        };
        const names = .{ "gist_d", "kc_d", "vc_d", "freqs_d", "mv0_d" };
        var made: usize = 0;
        errdefer {
            inline for (names, 0..) |name, i| if (made > i) be.tensorDestroy(&@field(self, name));
            gpa.free(self.sigmas);
            gpa.free(self.tables);
            gpa.free(self.finals);
        }

        const lins = try allLins(gpa, model);
        defer gpa.free(lins);
        try lin_cuda.presize(be, self.plan, lins);
        try lin_cuda.attachLora(be, &self.plan, model.lora, lins, @max(seq, text_pad));

        self.gist_d = try be.tensorCreate(std.mem.alignForward(usize, n_gist, 128) * dim * 4);
        made += 1;
        const cache = cfg.nInject() * text_pad * dim * 4;
        self.kc_d = try be.tensorCreate(cache);
        made += 1;
        self.vc_d = try be.tensorCreate(cache);
        made += 1;
        {
            var f = try model.ropeFreqs(gpa, n_gist, lat_h, lat_w);
            defer f.deinit(gpa);
            const flat = try gpa.alloc(f32, 2 * seq * half);
            defer gpa.free(flat);
            @memcpy(flat[0 .. seq * half], f.cos);
            @memcpy(flat[seq * half ..], f.sin);
            self.freqs_d = try be.tensorCreate(flat.len * 4);
            made += 1;
            try be.tensorUpload(self.freqs_d, std.mem.sliceAsBytes(flat));
        }
        {
            const a0 = try model.adalnInput(io, gpa, 0);
            defer gpa.free(a0);
            const t0 = try model.modTable(io, gpa, a0, true, true);
            defer gpa.free(t0);
            self.zero_len = t0.len;
            self.mv0_d = try be.tensorCreate(t0.len * 4);
            made += 1;
            try be.tensorUpload(self.mv0_d, std.mem.sliceAsBytes(t0));
        }
        self.sigmas = try gpa.dupe(f32, sigmas);
        self.step_len = (model.noise_refiner.len + model.layers.len) * 4 * dim + dim;
        self.tables = try gpa.alloc(f32, sigmas.len * self.step_len);
        self.finals = try gpa.alloc(f32, sigmas.len * dim);
        for (sigmas, 0..) |s, i| try stepTable(model, io, gpa, s, self.tables[i * self.step_len ..][0..self.step_len], self.finals[i * dim ..][0..dim]);

        try self.runPrefix(gpa, be, model, ctx, cancel);
        return self;
    }

    pub fn deinit(self: *Session, gpa: std.mem.Allocator, be: *Backend) void {
        inline for (.{ "gist_d", "kc_d", "vc_d", "freqs_d", "mv0_d" }) |name| be.tensorDestroy(&@field(self, name));
        gpa.free(self.sigmas);
        gpa.free(self.tables);
        gpa.free(self.finals);
        self.* = undefined;
    }

    /// The Perceiver and the cross keys/values, in f32 scratch freed on return.
    fn runPrefix(self: *Session, gpa: std.mem.Allocator, be: *Backend, model: *const DiT, ctx: []const f32, cancel: ?*std.atomic.Value(bool)) !void {
        const cfg = model.cfg;
        const dim = cfg.dim;
        const nt = self.text_len;
        const ng = self.n_gist;
        const n_b = 1 + cfg.nInject();
        const rows = std.mem.alignForward(usize, @max(ng, nt), 128);

        var bundles = try be.tensorCreate(n_b * self.text_pad * cfg.cap_dim * 4);
        defer be.tensorDestroy(&bundles);
        {
            const host = try gpa.alloc(f32, nt * cfg.cap_dim);
            defer gpa.free(host);
            for (0..n_b) |b| {
                fibo2.bundleOf(cfg, host, ctx, nt, b);
                try be.tensorUpload(bundles.slice(b * self.text_pad * cfg.cap_dim * 4, host.len * 4), std.mem.sliceAsBytes(host));
            }
        }
        try be.tensorUpload(self.gist_d.slice(0, ng * dim * 4), std.mem.sliceAsBytes(model.latent_queries[0 .. ng * dim]));

        var scratch: [8]Buf = undefined;
        var made: usize = 0;
        defer for (scratch[0..made]) |*b| be.tensorDestroy(b);
        const ffn_rows = std.mem.alignForward(usize, ng, 128);
        const sizes = [8]usize{ rows * dim, rows * dim, rows * dim, rows * dim, rows * dim, rows * dim, ffn_rows * cfg.ffn_dim, ffn_rows * cfg.ffn_dim };
        for (sizes) |n| {
            scratch[made] = try be.tensorCreate(n * 4);
            made += 1;
        }
        const p: dev.PrefixBufs(Buf) = .{ .g = self.gist_d, .nrm = scratch[0], .dlt = scratch[1], .q = scratch[2], .k = scratch[3], .v = scratch[4], .o = scratch[5], .mg = scratch[6], .mu = scratch[7] };

        try be.beginBatch();
        errdefer if (be.batching()) be.abortBatch();
        // The prefix reads no workspace.
        try dev.prefix(Dev{ .model = model, .be = be, .sess = self, .ws = undefined, .bundles = bundles }, model, p, cancel);
        try be.endBatch();
    }

    fn cached(self: *const Session, sigma: f32) ?Step {
        const dim = self.finals.len / @max(self.sigmas.len, 1);
        for (self.sigmas, 0..) |s, i| if (s == sigma) return .{
            .table = self.tables[i * self.step_len ..][0..self.step_len],
            .final = self.finals[i * dim ..][0..dim],
        };
        return null;
    }
};

/// One step's table and final scale.
const Step = struct { table: []const f32, final: []const f32 };

fn stepTable(model: *const DiT, io: std.Io, gpa: std.mem.Allocator, sigma: f32, table: []f32, final: []f32) !void {
    const a = try model.adalnInput(io, gpa, sigma);
    defer gpa.free(a);
    const t = try model.modTable(io, gpa, a, false, true);
    defer gpa.free(t);
    @memcpy(table, t);
    const f = try model.finalScale(io, gpa, a);
    defer gpa.free(f);
    @memcpy(final, f);
}

/// Activation buffers, shared by both CFG branches, so sized for the most gists.
pub const Workspace = struct {
    x_d: Buf,
    nrm_d: Buf,
    dlt_d: Buf,
    q_d: Buf,
    k_d: Buf,
    v_d: Buf,
    attn_d: Buf,
    mg_d: Buf,
    mu_d: Buf,
    mv_d: Buf,
    imgin_d: Buf,
    /// The residual, norm output, delta and FFN halves; the attention block stays f32.
    act: Dt,

    const fields = [_][]const u8{ "x_d", "nrm_d", "dlt_d", "q_d", "k_d", "v_d", "attn_d", "mg_d", "mu_d", "mv_d", "imgin_d" };

    pub fn init(be: *Backend, model: *const DiT, n_img: usize, act: Dt) !Workspace {
        const cfg = model.cfg;
        const dim = cfg.dim;
        // A quantized route's GEMM writes whole 128-row tiles.
        const rows = std.mem.alignForward(usize, cfg.max_gist + n_img, 128);
        const sizes = [fields.len]usize{
            rows * dim * 4,
            rows * dim * 4,
            rows * dim * 4,
            rows * dim * 4,
            rows * dim * 4,
            rows * dim * 4,
            rows * dim * 4,
            rows * cfg.ffn_dim * 4,
            rows * cfg.ffn_dim * 4,
            ((cfg.n_refiner_layers + cfg.n_layers) * 4 * dim + dim) * 4,
            n_img * cfg.channels * 4,
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
    return lin_cuda.loraRunnable(model.lora, model.device_lins, "fibo2 cuda") and
        lin_cuda.loraRunnable(model.lora, model.prefix_lins, "fibo2 cuda");
}

/// One denoiser forward. `out` / `x_lat` are planar `[channels][lat_h][lat_w]`.
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
    const dim = cfg.dim;
    const n_img = sess.n_img;
    std.debug.assert(x_lat.len == cfg.channels * n_img and out.len == x_lat.len);

    var owned: ?[]f32 = null;
    defer if (owned) |b| gpa.free(b);
    const step = sess.cached(sigma) orelse blk: {
        owned = try gpa.alloc(f32, sess.step_len + dim);
        try stepTable(model, io, gpa, sigma, owned.?[0..sess.step_len], owned.?[sess.step_len..]);
        break :blk Step{ .table = owned.?[0..sess.step_len], .final = owned.?[sess.step_len..] };
    };
    {
        const tok = try fibo2.tokens(gpa, cfg.channels, x_lat, n_img);
        defer gpa.free(tok);
        try be.tensorUpload(ws.imgin_d, std.mem.sliceAsBytes(tok));
    }
    try be.tensorUpload(ws.mv_d, std.mem.sliceAsBytes(step.table));

    try be.beginBatch();
    errdefer if (be.batching()) be.abortBatch();
    try dev.body(Dev{ .model = model, .be = be, .sess = sess, .ws = ws, .bundles = undefined }, model, cancel);
    try be.endBatch();
    if (ws.act != .f32 and !try be.allFinite(ws.x_d, sess.seq * dim)) return error.NonFiniteActivation;

    const img_rows = try gpa.alloc(f32, n_img * dim);
    defer gpa.free(img_rows);
    const ew = ws.act.bytes();
    try be.downloadF32(ws.x_d.slice(sess.n_gist * dim * ew, n_img * dim * ew), img_rows);
    const tok = try gpa.alloc(f32, n_img * cfg.channels);
    defer gpa.free(tok);
    try model.finalLayer(io, gpa, tok, img_rows, step.final, n_img);
    fibo2.untokens(cfg.channels, out, tok, n_img);
}

/// `fibo2_dev`'s ops on this backend.
const Dev = struct {
    model: *const DiT,
    be: *Backend,
    sess: *Session,
    ws: *Workspace,
    /// The prefix pass's `[bundles][text_pad][cap_dim]` upload.
    bundles: Buf,

    pub fn nGist(d: Dev) usize {
        return d.sess.n_gist;
    }
    pub fn nImg(d: Dev) usize {
        return d.sess.n_img;
    }
    pub fn textLen(d: Dev) usize {
        return d.sess.text_len;
    }

    pub fn rows(_: Dev, buf: Buf, row0: usize, n: usize, width: usize) error{}!Buf {
        const ew = buf.dt.bytes();
        return buf.slice(row0 * width * ew, n * width * ew);
    }

    pub fn bundle(d: Dev, b: usize) error{}!Buf {
        const cd = d.model.cfg.cap_dim;
        return d.bundles.slice(b * d.sess.text_pad * cd * 4, d.sess.text_pad * cd * 4);
    }

    pub fn crossK(d: Dev, j: usize) error{}!Buf {
        const n = d.sess.text_pad * d.model.cfg.dim * 4;
        return d.sess.kc_d.slice(j * n, n);
    }

    pub fn crossV(d: Dev, j: usize) error{}!Buf {
        const n = d.sess.text_pad * d.model.cfg.dim * 4;
        return d.sess.vc_d.slice(j * n, n);
    }

    /// `x_embedder`, materialized f32 at load, so it takes the f32 arm.
    pub fn embed(d: Dev, dst: Buf) !void {
        try lin_cuda.linear(d.be, d.sess.plan, dst, d.ws.imgin_d, d.sess.n_img, d.model.x_embedder.w, d.model.x_embedder.b);
    }

    pub fn loadGist(d: Dev, dst: Buf) !void {
        const n = d.sess.n_gist * d.model.cfg.dim;
        if (dst.dt == .f32) try d.be.tensorCopy(dst, 0, d.sess.gist_d, 0, n * 4) else try d.be.opCast(d.sess.gist_d, dst, n, 0);
    }

    /// Prefetch one block ahead; the last refiner primes the trunk's first.
    pub fn blockBegin(d: Dev, stage: enum { refiner, trunk }, i: usize) !void {
        if (!d.be.async_uploads) return;
        const m = d.model;
        switch (stage) {
            .refiner => {
                if (i == 0) prefetchBlock(d.be, &m.noise_refiner[0]);
                if (i + 1 < m.noise_refiner.len) prefetchBlock(d.be, &m.noise_refiner[i + 1]) else if (m.layers.len > 0) prefetchBlock(d.be, &m.layers[0]);
            },
            .trunk => if (i + 1 < m.layers.len) prefetchBlock(d.be, &m.layers[i + 1]),
        }
    }

    pub fn trace(_: Dev, _: []const u8, _: usize, _: Buf, _: usize) void {}

    fn table(d: Dev, t: dev.Tbl) struct { buf: Buf, zero: usize } {
        const dim = d.model.cfg.dim;
        return switch (t) {
            .step => .{ .buf = d.ws.mv_d, .zero = d.sess.step_len - dim },
            .zero => .{ .buf = d.sess.mv0_d, .zero = d.sess.zero_len - dim },
        };
    }

    /// One launch: the norm weight is folded into the table's scale.
    pub fn modNorm(d: Dev, x: Buf, nrm: Buf, n: usize, t: dev.Tbl, off: usize, _: []const f32) !void {
        const cfg = d.model.cfg;
        const tb = d.table(t);
        try d.be.rmsMod(x, nrm, tb.buf, n, cfg.dim, off, tb.zero, cfg.norm_eps);
    }

    pub fn gatedAdd(d: Dev, x: Buf, dlt: Buf, n: usize, t: dev.Tbl, off: usize) !void {
        const dim = d.model.cfg.dim;
        try d.be.gatedAdd(x, dlt, d.table(t).buf, n * dim, dim, off);
    }

    pub fn norm(d: Dev, x: Buf, out: Buf, w: []const f32, n: usize) !void {
        const cfg = d.model.cfg;
        try d.be.qkNorm(x, out, try normBuf(d.be, w), n, cfg.dim, cfg.norm_eps);
    }

    pub fn headNorm(d: Dev, buf: Buf, w: []const f32, n: usize) !void {
        const cfg = d.model.cfg;
        try d.be.qkNorm(buf, buf, try normBuf(d.be, w), n * cfg.n_heads, cfg.head_dim, cfg.norm_eps);
    }

    /// `headNorm` then rope from joint row `row0`, in one pass.
    pub fn headNormRope(d: Dev, buf: Buf, w: []const f32, n: usize, row0: usize) !void {
        const cfg = d.model.cfg;
        const half = cfg.head_dim / 2;
        try d.be.normRope(buf, buf, try normBuf(d.be, w), d.sess.freqs_d, .{ .rows = n * cfg.n_heads, .dim = cfg.head_dim, .heads = cfg.n_heads, .pairs = half, .sin_off = d.sess.seq * half, .eps = cfg.norm_eps, .pos0 = row0, .interleaved = true });
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

    pub fn attention(d: Dev, q: Buf, k: Buf, v: Buf, o: Buf, q_rows: usize, kv_rows: usize) !void {
        const cfg = d.model.cfg;
        const sc: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));
        const h = cfg.n_heads;
        if (force_attn_f32) return d.be.attn(q, k, v, o, q_rows, kv_rows, h, h, cfg.head_dim, sc, false);
        if (q_rows == kv_rows) return d.be.opAttnTC(q, k, v, o, q_rows, h, h, cfg.head_dim, sc);
        try d.be.opAttnTCRect(q, k, v, o, q_rows, kv_rows, h, h, cfg.head_dim, sc);
    }

    pub fn scale(d: Dev, buf: Buf, n: usize, f: f32) !void {
        try d.be.opScale(buf, f, n);
    }

    pub fn add(d: Dev, x: Buf, t: Buf, n: usize) !void {
        try d.be.opAdd(x, t, n);
    }

    pub fn addScaled(d: Dev, x: Buf, t: Buf, n: usize, s: f32) !void {
        try d.be.opAddScaled(x, t, n, s);
    }

    /// `linear` of `silu(g) * u`, the product made inside the activation prep where
    /// nothing else reads it (`lin_cuda.linearSiluMul`).
    pub fn siluMulLinear(d: Dev, y: Buf, g: Buf, u: Buf, m: usize, w: Weight) !void {
        try lin_cuda.linearSiluMul(d.be, d.sess.plan, y, g, u, m, w);
    }
};

/// Keys must be the host pointers the forward fetches, hence the loader's row views.
fn prefetchBlock(be: *Backend, blk: *const fibo2.Block) void {
    for (fibo2.blockLins(blk)) |w| be.prefetchWeight(w.bytes);
    if (blk.cross) |c| {
        be.prefetchWeight(c.attn.q.bytes);
        be.prefetchWeight(c.attn.out.bytes);
    }
}

fn normBuf(be: *Backend, weights: []const f32) !Buf {
    const h = try be.smallBuffer(std.mem.sliceAsBytes(weights));
    return .{ .buf = h, .size = weights.len * 4 };
}
