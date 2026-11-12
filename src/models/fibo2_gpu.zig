//! GPU-resident Fibo 2.0 forward on Vulkan, over `fibo2_dev`; the host/device split is
//! `fibo2_cuda`'s. The tables are the unfolded ones: Vulkan has a standalone `modulate`,
//! so the norm weight is applied by the norm. Attention is the tensor-core scores
//! pipeline, square for self-attention and `stageRect` for the cross-attention, which
//! is why a device without tensor cores is refused rather than run on the per-thread
//! kernel at canvas size.

const std = @import("std");
const fibo2 = @import("fibo2.zig");
const dev = @import("fibo2_dev.zig");
const lin_gpu = @import("lin_gpu.zig");
const attn_gpu = @import("attn_gpu.zig");
const gpu = @import("tp_gpu").context;
const ops = @import("tp_ops");

const DiT = fibo2.DiT;
const Buf = gpu.DeviceBuffer;
const Weight = ops.matmul.Weight;

/// Every attention through the f32 per-thread kernel, the reference a device check
/// holds the tensor-core path to.
pub var force_attn_f32: bool = false;

/// Z-Image's block, whose activations pass f16's range.
const lin_use: lin_gpu.Use = .{ .blockq = .auto, .act16 = .bf16 };

fn allLins(gpa: std.mem.Allocator, model: *const DiT) ![]Weight {
    return std.mem.concat(gpa, Weight, &.{ model.device_lins, model.prefix_lins });
}

pub fn supported(ctx: *gpu.Context, model: *const DiT) bool {
    return attn_gpu.tensorCores(ctx) and
        lin_gpu.runs(ctx, model.device_lins, lin_use, model.lora, "fibo2 vulkan") and
        lin_gpu.runs(ctx, model.prefix_lins, lin_use, model.lora, "fibo2 vulkan");
}

/// One conditioning at one resolution, as `fibo2_cuda.Session`.
pub const Session = struct {
    lat_h: usize,
    lat_w: usize,
    n_img: usize,
    n_gist: usize,
    text_len: usize,
    seq: usize,
    text_pad: usize,
    lin: lin_gpu.Lin,
    lin_plan: lin_gpu.Plan,

    gist_d: Buf,
    kc_d: Buf,
    vc_d: Buf,
    freqs_d: Buf,
    mv0_d: Buf,
    zero_len: usize,

    sigmas: []f32,
    tables: []f32,
    finals: []f32,
    step_len: usize,

    pub fn init(
        gpa: std.mem.Allocator,
        io: std.Io,
        ctx: *gpu.Context,
        model: *const DiT,
        lat_h: usize,
        lat_w: usize,
        cond: []const f32,
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

        const lins = try allLins(gpa, model);
        defer gpa.free(lins);
        var l = try lin_gpu.Lin.init(gpa, ctx, lin_gpu.maxRows(lins), 0, 0);
        errdefer l.deinit();
        var lp = try lin_gpu.plan(ctx, lins, lin_use, "fibo2 vulkan");
        if (model.lora) |st| {
            lp.lora = st;
            try l.presizeLora(st, lins, std.mem.alignForward(usize, @max(seq, text_pad), 128));
        }

        var self: Session = .{
            .lat_h = lat_h,
            .lat_w = lat_w,
            .n_img = n_img,
            .n_gist = n_gist,
            .text_len = text_len,
            .seq = seq,
            .text_pad = text_pad,
            .lin = l,
            .lin_plan = lp,
            .gist_d = undefined,
            .kc_d = undefined,
            .vc_d = undefined,
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
            inline for (names, 0..) |name, i| if (made > i) ctx.tensorDestroy(&@field(self, name));
            gpa.free(self.sigmas);
            gpa.free(self.tables);
            gpa.free(self.finals);
        }

        self.gist_d = try ctx.tensorCreate(std.mem.alignForward(usize, n_gist, 128) * dim * 4);
        made += 1;
        const cache = cfg.nInject() * text_pad * dim * 4;
        self.kc_d = try ctx.tensorCreate(@max(cache, 4));
        made += 1;
        self.vc_d = try ctx.tensorCreate(@max(cache, 4));
        made += 1;
        {
            var f = try model.ropeFreqs(gpa, n_gist, lat_h, lat_w);
            defer f.deinit(gpa);
            const flat = try gpa.alloc(f32, 2 * seq * half);
            defer gpa.free(flat);
            @memcpy(flat[0 .. seq * half], f.cos);
            @memcpy(flat[seq * half ..], f.sin);
            self.freqs_d = try ctx.tensorCreate(flat.len * 4);
            made += 1;
            try ctx.tensorUpload(self.freqs_d, std.mem.sliceAsBytes(flat));
        }
        {
            const a0 = try model.adalnInput(io, gpa, 0);
            defer gpa.free(a0);
            const t0 = try model.modTable(io, gpa, a0, true, false);
            defer gpa.free(t0);
            self.zero_len = t0.len;
            self.mv0_d = try ctx.tensorCreate(t0.len * 4);
            made += 1;
            try ctx.tensorUpload(self.mv0_d, std.mem.sliceAsBytes(t0));
        }
        self.sigmas = try gpa.dupe(f32, sigmas);
        self.step_len = (model.noise_refiner.len + model.layers.len) * 4 * dim + dim;
        self.tables = try gpa.alloc(f32, sigmas.len * self.step_len);
        self.finals = try gpa.alloc(f32, sigmas.len * dim);
        for (sigmas, 0..) |s, i| try stepTable(model, io, gpa, s, self.tables[i * self.step_len ..][0..self.step_len], self.finals[i * dim ..][0..dim]);

        try self.runPrefix(gpa, ctx, model, cond, cancel);
        return self;
    }

    pub fn deinit(self: *Session, gpa: std.mem.Allocator, ctx: *gpu.Context) void {
        inline for (.{ "gist_d", "kc_d", "vc_d", "freqs_d", "mv0_d" }) |name| ctx.tensorDestroy(&@field(self, name));
        gpa.free(self.sigmas);
        gpa.free(self.tables);
        gpa.free(self.finals);
        self.lin.deinit();
        self.* = undefined;
    }

    fn runPrefix(self: *Session, gpa: std.mem.Allocator, ctx: *gpu.Context, model: *const DiT, cond: []const f32, cancel: ?*std.atomic.Value(bool)) !void {
        const cfg = model.cfg;
        const dim = cfg.dim;
        const nt = self.text_len;
        const ng = self.n_gist;
        const n_b = 1 + cfg.nInject();
        const rows = std.mem.alignForward(usize, @max(ng, nt), 128);

        var bundles = try ctx.tensorCreate(n_b * self.text_pad * cfg.cap_dim * 4);
        defer ctx.tensorDestroy(&bundles);
        {
            const host = try gpa.alloc(f32, nt * cfg.cap_dim);
            defer gpa.free(host);
            for (0..n_b) |b| {
                fibo2.bundleOf(cfg, host, cond, nt, b);
                try ctx.tensorUploadAt(bundles, b * self.text_pad * cfg.cap_dim * 4, std.mem.sliceAsBytes(host));
            }
        }
        try ctx.tensorUploadAt(self.gist_d, 0, std.mem.sliceAsBytes(model.latent_queries[0 .. ng * dim]));

        var scratch: [8]Buf = undefined;
        var made: usize = 0;
        defer for (scratch[0..made]) |*b| ctx.tensorDestroy(b);
        const ffn_rows = std.mem.alignForward(usize, ng, 128);
        const sizes = [8]usize{ rows * dim, rows * dim, rows * dim, rows * dim, rows * dim, rows * dim, ffn_rows * cfg.ffn_dim, ffn_rows * cfg.ffn_dim };
        for (sizes) |n| {
            scratch[made] = try ctx.tensorCreate(n * 4);
            made += 1;
        }
        var attn = try attn_gpu.Scratch.initRect(ctx, ng, @max(ng, nt), cfg.n_heads, cfg.n_heads, cfg.head_dim, attn_gpu.s_bytes_cap);
        defer attn.deinit(ctx);
        const p: dev.PrefixBufs(Buf) = .{ .g = self.gist_d, .nrm = scratch[0], .dlt = scratch[1], .q = scratch[2], .k = scratch[3], .v = scratch[4], .o = scratch[5], .mg = scratch[6], .mu = scratch[7] };

        const lins = try allLins(gpa, model);
        defer gpa.free(lins);
        try lin_gpu.presize(ctx, self.lin_plan, lins);
        try ctx.beginBatch();
        errdefer if (ctx.batching) ctx.abortBatch();
        // The prefix reads no workspace.
        try dev.prefix(Dev{ .model = model, .ctx = ctx, .sess = self, .ws = undefined, .attn = &attn, .bundles = bundles }, model, p, cancel);
        try ctx.endBatch();
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

const Step = struct { table: []const f32, final: []const f32 };

fn stepTable(model: *const DiT, io: std.Io, gpa: std.mem.Allocator, sigma: f32, table: []f32, final: []f32) !void {
    const a = try model.adalnInput(io, gpa, sigma);
    defer gpa.free(a);
    const t = try model.modTable(io, gpa, a, false, false);
    defer gpa.free(t);
    @memcpy(table, t);
    const f = try model.finalScale(io, gpa, a);
    defer gpa.free(f);
    @memcpy(final, f);
}

/// Activation buffers, shared by both CFG branches, so sized for the most gists and the
/// longest text either branch carries.
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
    attn: attn_gpu.Scratch,

    const fields = [_][]const u8{ "x_d", "nrm_d", "dlt_d", "q_d", "k_d", "v_d", "attn_d", "mg_d", "mu_d", "mv_d", "imgin_d" };

    pub fn init(ctx: *gpu.Context, model: *const DiT, n_img: usize, text_cap: usize, act: gpu.Dt) !Workspace {
        const cfg = model.cfg;
        const dim = cfg.dim;
        const seq = cfg.max_gist + n_img;
        const rows = std.mem.alignForward(usize, seq, 128);
        const ew = act.bytes();
        const sizes = [fields.len]usize{
            rows * dim * ew,
            rows * dim * ew,
            rows * dim * ew,
            rows * dim * 4,
            rows * dim * 4,
            rows * dim * 4,
            rows * dim * 4,
            rows * cfg.ffn_dim * ew,
            rows * cfg.ffn_dim * ew,
            ((cfg.n_refiner_layers + cfg.n_layers) * 4 * dim + dim) * 4,
            n_img * cfg.channels * 4,
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
        self.attn = try attn_gpu.Scratch.initRect(ctx, seq, @max(seq, text_cap), cfg.n_heads, cfg.n_heads, cfg.head_dim, attn_gpu.s_bytes_cap);
        inline for (.{ "x_d", "nrm_d", "dlt_d", "mg_d", "mu_d" }) |name| @field(self, name).dt = act;
        return self;
    }

    pub fn deinit(self: *Workspace, ctx: *gpu.Context) void {
        inline for (fields) |name| ctx.tensorDestroy(&@field(self, name));
        self.attn.deinit(ctx);
        self.* = undefined;
    }
};

/// One denoiser forward. `out` / `x_lat` are planar `[channels][lat_h][lat_w]`.
pub fn forward(
    model: *const DiT,
    ctx: *gpu.Context,
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
        try ctx.tensorUpload(ws.imgin_d, std.mem.sliceAsBytes(tok));
    }
    try ctx.tensorUpload(ws.mv_d, std.mem.sliceAsBytes(step.table));
    try lin_gpu.presize(ctx, sess.lin_plan, model.device_lins);

    try ctx.beginBatch();
    errdefer if (ctx.batching) ctx.abortBatch();
    try dev.body(Dev{ .model = model, .ctx = ctx, .sess = sess, .ws = ws, .attn = &ws.attn, .bundles = undefined }, model, cancel);
    try ctx.endBatch();
    if (ws.x_d.dt != .f32 and !try ctx.allFinite(ws.x_d, sess.seq * dim)) return error.NonFiniteActivation;

    const img_rows = try gpa.alloc(f32, n_img * dim);
    defer gpa.free(img_rows);
    try ctx.downloadF32At(ws.x_d, sess.n_gist * dim, img_rows);
    const tok = try gpa.alloc(f32, n_img * cfg.channels);
    defer gpa.free(tok);
    try model.finalLayer(io, gpa, tok, img_rows, step.final, n_img);
    fibo2.untokens(cfg.channels, out, tok, n_img);
}

/// `fibo2_dev`'s ops on this backend.
const Dev = struct {
    model: *const DiT,
    ctx: *gpu.Context,
    sess: *Session,
    ws: *Workspace,
    attn: *const attn_gpu.Scratch,
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

    pub fn rows(d: Dev, buf: Buf, row0: usize, n: usize, width: usize) !Buf {
        const ew = buf.dt.bytes();
        return d.ctx.view(buf, row0 * width * ew, n * width * ew);
    }

    pub fn bundle(d: Dev, b: usize) !Buf {
        const n = d.sess.text_pad * d.model.cfg.cap_dim * 4;
        return d.ctx.view(d.bundles, b * n, n);
    }

    pub fn crossK(d: Dev, j: usize) !Buf {
        const n = d.sess.text_pad * d.model.cfg.dim * 4;
        return d.ctx.view(d.sess.kc_d, j * n, n);
    }

    pub fn crossV(d: Dev, j: usize) !Buf {
        const n = d.sess.text_pad * d.model.cfg.dim * 4;
        return d.ctx.view(d.sess.vc_d, j * n, n);
    }

    pub fn embed(d: Dev, dst: Buf) !void {
        try d.sess.lin.linearAt(d.sess.lin_plan, dst, 0, d.ws.imgin_d, 0, d.sess.n_img, d.model.x_embedder.w, d.model.x_embedder.b);
    }

    pub fn loadGist(d: Dev, dst: Buf) !void {
        const n = d.sess.n_gist * d.model.cfg.dim;
        try d.ctx.opElt(gpu.eltCopy(.f32, dst.dt), d.sess.gist_d, dst, null, null, .{ .u0 = @intCast(n), .u2 = 0, .u3 = 0 }, n, 1, 1);
    }

    pub fn blockBegin(_: Dev, _: enum { refiner, trunk }, _: usize) !void {}

    pub fn trace(_: Dev, _: []const u8, _: usize, _: Buf, _: usize) void {}

    fn table(d: Dev, t: dev.Tbl) struct { buf: Buf, zero: usize } {
        const dim = d.model.cfg.dim;
        return switch (t) {
            .step => .{ .buf = d.ws.mv_d, .zero = d.sess.step_len - dim },
            .zero => .{ .buf = d.sess.mv0_d, .zero = d.sess.zero_len - dim },
        };
    }

    /// The norm, then `(1 + scale)` with no shift.
    pub fn modNorm(d: Dev, x: Buf, nrm: Buf, n: usize, t: dev.Tbl, off: usize, norm_w: []const f32) !void {
        const cfg = d.model.cfg;
        const tb = d.table(t);
        try rmsNorm(d.ctx, x, nrm, norm_w, n, cfg.dim, cfg.norm_eps);
        try d.ctx.opElt(gpu.eltAt("modulate", nrm.dt), nrm, null, tb.buf, null, .{
            .u0 = @intCast(n * cfg.dim),
            .u1 = @intCast(cfg.dim),
            .u2 = @intCast(off),
            .u3 = @intCast(tb.zero),
        }, n * cfg.dim, 1, 1);
    }

    pub fn gatedAdd(d: Dev, x: Buf, dlt: Buf, n: usize, t: dev.Tbl, off: usize) !void {
        const dim = d.model.cfg.dim;
        std.debug.assert(x.dt == dlt.dt);
        try d.ctx.opElt(gpu.eltAt("gated_add", x.dt), x, dlt, d.table(t).buf, null, .{ .u0 = @intCast(n * dim), .u1 = @intCast(dim), .u2 = @intCast(off) }, n * dim, 1, 1);
    }

    pub fn norm(d: Dev, x: Buf, out: Buf, w: []const f32, n: usize) !void {
        const cfg = d.model.cfg;
        try rmsNorm(d.ctx, x, out, w, n, cfg.dim, cfg.norm_eps);
    }

    pub fn headNorm(d: Dev, buf: Buf, w: []const f32, n: usize) !void {
        const cfg = d.model.cfg;
        try rmsNorm(d.ctx, buf, buf, w, n * cfg.n_heads, cfg.head_dim, cfg.norm_eps);
    }

    /// `headNorm` then rope from joint row `row0`, in one pass.
    pub fn headNormRope(d: Dev, buf: Buf, w: []const f32, n: usize, row0: usize) !void {
        const cfg = d.model.cfg;
        const half = cfg.head_dim / 2;
        const wb: Buf = .{ .buf = try d.ctx.smallBuffer(std.mem.sliceAsBytes(w)), .mem = .null_handle, .size = 0 };
        try d.ctx.opNormRope(buf, buf, wb, d.sess.freqs_d, .{ .rows = n * cfg.n_heads, .dim = cfg.head_dim, .heads = cfg.n_heads, .pairs = half, .sin_off = d.sess.seq * half, .eps = cfg.norm_eps, .pos0 = row0, .interleaved = true });
    }

    pub fn prep(d: Dev, x: Buf, m: usize, cols: usize, group: []const Weight) !lin_gpu.Group {
        return d.sess.lin.prep(d.sess.lin_plan, x, m, cols, group);
    }

    pub fn gemms(d: Dev, g: lin_gpu.Group, ys: []const Buf, group: []const Weight) !void {
        for (ys, group) |y, w| try d.sess.lin.gemm(d.sess.lin_plan, g, y, w, .{});
    }

    pub fn linear(d: Dev, y: Buf, x: Buf, m: usize, w: Weight) !void {
        try d.sess.lin.linear(d.sess.lin_plan, y, 0, x, m, w);
    }

    pub fn attention(d: Dev, q: Buf, k: Buf, v: Buf, o: Buf, q_rows: usize, kv_rows: usize) !void {
        const cfg = d.model.cfg;
        const hd = cfg.head_dim;
        const h = cfg.n_heads;
        const sc: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        if (force_attn_f32) return d.ctx.opElt(.attn_causal_batched, q, k, v, o, .{
            .u0 = @intCast(q_rows),
            .u1 = @intCast(h),
            .u2 = @intCast(h),
            .u3 = @intCast(hd),
            .u4 = @intCast(q_rows),
            .u5 = 0,
            .u6 = @intCast(kv_rows),
            .f0 = sc,
        }, q_rows * h, 1, 1);
        const s = d.attn;
        try attn_gpu.stageRect(d.ctx, q, k, v, s.qh, s.kh, s.v16, q_rows, kv_rows, h, h, hd, sc);
        try attn_gpu.core(d.ctx, s.bufs(), s.qh, s.kh, s.v16, o, .{ .q_rows = q_rows, .kv_rows = kv_rows, .heads = h, .kv_heads = h, .hd = hd }, null);
    }

    pub fn scale(d: Dev, buf: Buf, n: usize, f: f32) !void {
        try d.ctx.opElt(.scale_f32, buf, buf, null, null, .{ .u0 = @intCast(n), .f0 = f }, n, 1, 1);
    }

    pub fn add(d: Dev, x: Buf, t: Buf, n: usize) !void {
        try d.ctx.opElt(gpu.eltAt("add", x.dt), x, t, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
    }

    pub fn addScaled(d: Dev, x: Buf, t: Buf, n: usize, s: f32) !void {
        try d.ctx.opElt(gpu.eltAt("add_scaled", x.dt), x, t, null, null, .{ .u0 = @intCast(n), .f0 = s }, n, 1, 1);
    }

    pub fn siluMulLinear(d: Dev, y: Buf, g: Buf, u: Buf, m: usize, w: Weight) !void {
        try d.siluMul(g, u, m * w.cols);
        try d.linear(y, g, m, w);
    }

    fn siluMul(d: Dev, g: Buf, u: Buf, n: usize) !void {
        std.debug.assert(g.dt == u.dt);
        try d.ctx.opElt(gpu.eltAt("silu_mul", g.dt), g, u, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
    }
};

/// Weighted RMSNorm over `[rows][dim]`, `x -> out` (may alias), the subgroup kernel
/// where it exists.
fn rmsNorm(ctx: *gpu.Context, x: Buf, out: Buf, weights: []const f32, rows: usize, dim: usize, eps: f32) !void {
    const w: Buf = .{ .buf = try ctx.smallBuffer(std.mem.sliceAsBytes(weights)), .mem = .null_handle, .size = 0 };
    try ctx.opRmsNorm(x, out, w, rows, dim, eps);
}
