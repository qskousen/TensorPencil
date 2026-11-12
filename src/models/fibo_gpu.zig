//! GPU-resident Fibo 1.5 forward on Vulkan, over `fibo_dev`; the host/device split is
//! `fibo_cuda`'s. Attention is the tensor-core scores pipeline, which a device without
//! tensor cores does not have, so such a device is refused rather than run on the
//! per-thread kernel at canvas size.

const std = @import("std");
const fibo = @import("fibo.zig");
const fibo_text = @import("fibo_text.zig");
const dev = @import("fibo_dev.zig");
const lin_gpu = @import("lin_gpu.zig");
const attn_gpu = @import("attn_gpu.zig");
const gpu = @import("tp_gpu").context;
const ops = @import("tp_ops");

const DiT = fibo.DiT;
const Buf = gpu.DeviceBuffer;
const Weight = ops.matmul.Weight;

/// Every attention through the f32 per-thread kernel, the reference a device check
/// holds the tensor-core path to.
pub var force_attn_f32: bool = false;

const lin_use: lin_gpu.Use = .{ .blockq = .auto, .act16 = .bf16 };

fn allLins(gpa: std.mem.Allocator, model: *const DiT) ![]Weight {
    return std.mem.concat(gpa, Weight, &.{ model.device_lins, &.{model.context_embedder.w} });
}

pub fn supported(ctx: *gpu.Context, model: *const DiT) bool {
    return attn_gpu.tensorCores(ctx) and lin_gpu.runs(ctx, model.device_lins, lin_use, model.lora, "fibo vulkan");
}

fn padRows(n: usize) usize {
    return std.mem.alignForward(usize, n, 128);
}

/// One conditioning at one resolution, as `fibo_cuda.Session`.
pub const Session = struct {
    lat_h: usize,
    lat_w: usize,
    n_img: usize,
    text_len: usize,
    seq: usize,
    lin: lin_gpu.Lin,
    lin_plan: lin_gpu.Plan,
    txt_d: Buf,
    cap_d: Buf,
    text_pad: usize,
    freqs_d: Buf,
    sigmas: []f32,
    tables: []f32,
    finals: []f32,
    table_len: usize,

    pub fn init(
        gpa: std.mem.Allocator,
        io: std.Io,
        ctx: *gpu.Context,
        model: *const DiT,
        lat_h: usize,
        lat_w: usize,
        c: fibo_text.Context,
        text_len: usize,
        sigmas: []const f32,
        act: gpu.Dt,
    ) !Session {
        const cfg = model.cfg;
        const dim = cfg.dim;
        const half = cfg.head_dim / 2;
        const n_img = lat_h * lat_w;
        const seq = text_len + n_img;
        const text_pad = padRows(text_len);
        const t: fibo.Table = .{ .dim = dim, .n_double = cfg.n_double, .n_single = cfg.n_single };

        const lins = try allLins(gpa, model);
        defer gpa.free(lins);
        var l = try lin_gpu.Lin.init(gpa, ctx, lin_gpu.maxRows(lins), 0, 0);
        errdefer l.deinit();
        var lp = try lin_gpu.plan(ctx, lins, lin_use, "fibo vulkan");
        if (model.lora) |st| {
            lp.lora = st;
            try l.presizeLora(st, lins, @max(padRows(seq), text_pad));
        }

        var self: Session = .{
            .lat_h = lat_h,
            .lat_w = lat_w,
            .n_img = n_img,
            .text_len = text_len,
            .seq = seq,
            .lin = l,
            .lin_plan = lp,
            .txt_d = undefined,
            .cap_d = undefined,
            .text_pad = text_pad,
            .freqs_d = undefined,
            .sigmas = &.{},
            .tables = &.{},
            .finals = &.{},
            .table_len = t.len(),
        };
        const names = .{ "txt_d", "cap_d", "freqs_d" };
        var made: usize = 0;
        errdefer {
            inline for (names, 0..) |name, i| if (made > i) ctx.tensorDestroy(&@field(self, name));
            gpa.free(self.sigmas);
            gpa.free(self.tables);
            gpa.free(self.finals);
        }

        self.txt_d = try ctx.tensorCreate(text_pad * dim * act.bytes());
        made += 1;
        self.txt_d.dt = act;
        self.cap_d = try ctx.tensorCreate(cfg.blocks() * text_pad * (dim / 2) * act.bytes());
        made += 1;
        self.cap_d.dt = act;
        {
            var f = try model.ropeFreqs(gpa, text_len, lat_h, lat_w);
            defer f.deinit(gpa);
            const flat = try gpa.alloc(f32, 2 * seq * half);
            defer gpa.free(flat);
            @memcpy(flat[0 .. seq * half], f.cos);
            @memcpy(flat[seq * half ..], f.sin);
            self.freqs_d = try ctx.tensorCreate(flat.len * 4);
            made += 1;
            try ctx.tensorUpload(self.freqs_d, std.mem.sliceAsBytes(flat));
        }
        self.sigmas = try gpa.dupe(f32, sigmas);
        self.tables = try gpa.alloc(f32, sigmas.len * self.table_len);
        self.finals = try gpa.alloc(f32, sigmas.len * 2 * dim);
        for (sigmas, 0..) |s, i| try stepTables(model, io, gpa, s, self.tables[i * self.table_len ..][0..self.table_len], self.finals[i * 2 * dim ..][0 .. 2 * dim]);

        try self.runPrefix(gpa, ctx, model, c);
        return self;
    }

    pub fn deinit(self: *Session, gpa: std.mem.Allocator, ctx: *gpu.Context) void {
        inline for (.{ "txt_d", "cap_d", "freqs_d" }) |name| ctx.tensorDestroy(&@field(self, name));
        gpa.free(self.sigmas);
        gpa.free(self.tables);
        gpa.free(self.finals);
        self.lin.deinit();
        self.* = undefined;
    }

    fn runPrefix(self: *Session, gpa: std.mem.Allocator, ctx: *gpu.Context, model: *const DiT, c: fibo_text.Context) !void {
        const cfg = model.cfg;
        const nt = self.text_len;
        const lins = try allLins(gpa, model);
        defer gpa.free(lins);
        try lin_gpu.presize(ctx, self.lin_plan, lins);
        var ehs = try ctx.tensorCreate(self.text_pad * cfg.joint_dim * 4);
        defer ctx.tensorDestroy(&ehs);
        try ctx.tensorUploadAt(ehs, 0, std.mem.sliceAsBytes(c.ehs));
        // Every block's state at once: an upload inside a batch flushes it.
        const per_in = self.text_pad * cfg.te_dim * 4;
        var layers = try ctx.tensorCreate(cfg.blocks() * per_in);
        defer ctx.tensorDestroy(&layers);
        for (0..cfg.blocks()) |i| {
            try ctx.tensorUploadAt(layers, i * per_in, std.mem.sliceAsBytes(c.layers[i * nt * cfg.te_dim ..][0 .. nt * cfg.te_dim]));
        }
        try ctx.beginBatch();
        errdefer if (ctx.batching) ctx.abortBatch();
        try self.lin.linearAt(self.lin_plan, self.txt_d, 0, ehs, 0, nt, model.context_embedder.w, model.context_embedder.b);
        const per_out = self.text_pad * (cfg.dim / 2) * self.cap_d.dt.bytes();
        for (0..cfg.blocks()) |i| {
            const y = try ctx.view(self.cap_d, i * per_out, per_out);
            const x = try ctx.view(layers, i * per_in, per_in);
            try self.lin.linear(self.lin_plan, y, 0, x, nt, model.caption[i]);
        }
        try ctx.endBatch();
    }

    fn cached(self: *const Session, sigma: f32) ?Step {
        const dim2 = self.finals.len / @max(self.sigmas.len, 1);
        for (self.sigmas, 0..) |s, i| if (s == sigma) return .{
            .table = self.tables[i * self.table_len ..][0..self.table_len],
            .final = self.finals[i * dim2 ..][0..dim2],
        };
        return null;
    }
};

const Step = struct { table: []const f32, final: []const f32 };

fn stepTables(model: *const DiT, io: std.Io, gpa: std.mem.Allocator, sigma: f32, table: []f32, final: []f32) !void {
    const temb = try model.timeEmbed(io, gpa, sigma);
    defer gpa.free(temb);
    const t = try model.modTable(io, gpa, temb);
    defer gpa.free(t);
    @memcpy(table, t);
    const f = try model.finalMod(io, gpa, temb);
    defer gpa.free(f);
    @memcpy(final, f);
}

pub const Workspace = struct {
    x_d: Buf,
    nrm_d: Buf,
    dlt_d: Buf,
    q_d: Buf,
    k_d: Buf,
    v_d: Buf,
    attn_d: Buf,
    mg_d: Buf,
    mlp_d: Buf,
    cat_d: Buf,
    mv_d: Buf,
    imgin_d: Buf,
    attn: attn_gpu.Scratch,

    const fields = [_][]const u8{ "x_d", "nrm_d", "dlt_d", "q_d", "k_d", "v_d", "attn_d", "mg_d", "mlp_d", "cat_d", "mv_d", "imgin_d" };

    pub fn init(ctx: *gpu.Context, model: *const DiT, n_img: usize, text_cap: usize, act: gpu.Dt) !Workspace {
        const cfg = model.cfg;
        const dim = cfg.dim;
        const hid = 4 * dim;
        const seq = text_cap + n_img;
        const rows = padRows(seq);
        const ab = act.bytes();
        const t: fibo.Table = .{ .dim = dim, .n_double = cfg.n_double, .n_single = cfg.n_single };
        const sizes = [fields.len]usize{
            rows * dim * ab,
            rows * dim * ab,
            rows * dim * ab,
            rows * dim * 4,
            rows * dim * 4,
            rows * dim * 4,
            rows * dim * 4,
            rows * hid * ab,
            rows * hid * 4,
            rows * (dim + hid) * 4,
            t.len() * 4,
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
        self.attn = try attn_gpu.Scratch.init(ctx, seq, cfg.heads, cfg.heads, cfg.head_dim, attn_gpu.s_bytes_cap);
        inline for (.{ "x_d", "nrm_d", "dlt_d", "mg_d" }) |name| @field(self, name).dt = act;
        return self;
    }

    pub fn deinit(self: *Workspace, ctx: *gpu.Context) void {
        inline for (fields) |name| ctx.tensorDestroy(&@field(self, name));
        self.attn.deinit(ctx);
        self.* = undefined;
    }
};

pub fn forward(model: *const DiT, ctx: *gpu.Context, sess: *Session, ws: *Workspace, io: std.Io, gpa: std.mem.Allocator, out: []f32, x_lat: []const f32, sigma: f32, cancel: ?*std.atomic.Value(bool)) !void {
    const cfg = model.cfg;
    const dim = cfg.dim;
    const n_img = sess.n_img;
    std.debug.assert(x_lat.len == cfg.channels * n_img and out.len == x_lat.len);
    std.debug.assert(ws.x_d.dt == sess.txt_d.dt);

    var owned: ?[]f32 = null;
    defer if (owned) |b| gpa.free(b);
    const step = sess.cached(sigma) orelse blk: {
        owned = try gpa.alloc(f32, sess.table_len + 2 * dim);
        try stepTables(model, io, gpa, sigma, owned.?[0..sess.table_len], owned.?[sess.table_len..]);
        break :blk Step{ .table = owned.?[0..sess.table_len], .final = owned.?[sess.table_len..] };
    };
    {
        const tok = try gpa.alloc(f32, x_lat.len);
        defer gpa.free(tok);
        for (0..cfg.channels) |c| for (0..n_img) |r| {
            tok[r * cfg.channels + c] = x_lat[c * n_img + r];
        };
        try ctx.tensorUploadAt(ws.imgin_d, 0, std.mem.sliceAsBytes(tok));
    }
    try ctx.tensorUpload(ws.mv_d, std.mem.sliceAsBytes(step.table));
    try lin_gpu.presize(ctx, sess.lin_plan, model.device_lins);

    try ctx.beginBatch();
    errdefer if (ctx.batching) ctx.abortBatch();
    try dev.body(Dev{ .model = model, .ctx = ctx, .sess = sess, .ws = ws }, model, cancel);
    try ctx.endBatch();
    if (ws.x_d.dt != .f32 and !try ctx.allFinite(ws.x_d, sess.seq * dim)) return error.NonFiniteActivation;

    const img = try gpa.alloc(f32, n_img * dim);
    defer gpa.free(img);
    try ctx.downloadF32At(ws.x_d, sess.text_len * dim, img);
    try model.finalLayer(io, gpa, out, img, step.final, n_img);
}

const Dev = struct {
    model: *const DiT,
    ctx: *gpu.Context,
    sess: *Session,
    ws: *Workspace,

    pub fn nText(d: Dev) usize {
        return d.sess.text_len;
    }
    pub fn nImg(d: Dev) usize {
        return d.sess.n_img;
    }

    pub fn rows(d: Dev, buf: Buf, row0: usize, n: usize, width: usize) !Buf {
        const ew = buf.dt.bytes();
        return d.ctx.view(buf, row0 * width * ew, n * width * ew);
    }

    pub fn loadText(d: Dev, dst: Buf) !void {
        const n = d.sess.text_len * d.model.cfg.dim;
        try d.ctx.opElt(gpu.eltCopy(d.sess.txt_d.dt, dst.dt), d.sess.txt_d, dst, null, null, .{ .u0 = @intCast(n), .u2 = 0, .u3 = 0 }, n, 1, 1);
    }

    pub fn embed(d: Dev, dst: Buf) !void {
        try d.sess.lin.linearAt(d.sess.lin_plan, dst, 0, d.ws.imgin_d, 0, d.sess.n_img, d.model.x_embedder.w, d.model.x_embedder.b);
    }

    pub fn caption(d: Dev, i: usize, txt: Buf) !void {
        const dim = d.model.cfg.dim;
        const per = d.sess.text_pad * (dim / 2) * d.sess.cap_d.dt.bytes();
        const src = try d.ctx.view(d.sess.cap_d, i * per, per);
        try d.concat(src, txt, d.sess.text_len, dim / 2, dim, dim / 2);
    }

    pub fn blockBegin(_: Dev, _: usize) !void {}

    pub fn lnMod(d: Dev, x: Buf, out: Buf, n: usize, premul: usize, shift: usize) !void {
        const cfg = d.model.cfg;
        try d.ctx.opLnModSg(x, out, d.ws.mv_d, n, cfg.dim, premul, shift, cfg.eps);
    }

    pub fn gemms(d: Dev, x: Buf, m: usize, cols: usize, lws: []const fibo.LinearW, ys: []const Buf) !void {
        var w: [4]Weight = undefined;
        for (lws, 0..) |lw, k| w[k] = lw.w;
        const g = try d.sess.lin.prep(d.sess.lin_plan, x, m, cols, w[0..lws.len]);
        for (lws, ys) |lw, y| try d.sess.lin.gemm(d.sess.lin_plan, g, y, lw.w, .{ .bias = lw.b });
    }

    pub fn linear(d: Dev, y: Buf, x: Buf, m: usize, lw: fibo.LinearW) !void {
        const g = try d.sess.lin.prep(d.sess.lin_plan, x, m, lw.w.cols, &.{lw.w});
        try d.sess.lin.gemm(d.sess.lin_plan, g, y, lw.w, .{ .bias = lw.b });
    }

    pub fn headNorm(d: Dev, buf: Buf, w: []const f32, n: usize) !void {
        const cfg = d.model.cfg;
        const wb: Buf = .{ .buf = try d.ctx.smallBuffer(std.mem.sliceAsBytes(w)), .mem = .null_handle, .size = 0 };
        try d.ctx.opRmsNorm(buf, buf, wb, n * cfg.heads, cfg.head_dim, cfg.eps);
    }

    pub fn rope(d: Dev, buf: Buf, n: usize) !void {
        const cfg = d.model.cfg;
        const half = cfg.head_dim / 2;
        try d.ctx.opElt(.rope_inter, buf, null, d.sess.freqs_d, null, .{
            .u0 = @intCast(n * cfg.heads * half),
            .u1 = @intCast(half),
            .u2 = @intCast(d.sess.seq * half),
            .u3 = @intCast(cfg.heads),
        }, n * cfg.heads * half, 1, 1);
    }

    pub fn scale(d: Dev, buf: Buf, n: usize, f: f32) !void {
        try d.ctx.opElt(.scale_f32, buf, buf, null, null, .{ .u0 = @intCast(n), .f0 = f }, n, 1, 1);
    }

    pub fn attention(d: Dev, n: usize) !void {
        const cfg = d.model.cfg;
        const ws = d.ws;
        const sc: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));
        if (force_attn_f32) return d.ctx.opElt(.attn_causal_batched, ws.q_d, ws.k_d, ws.v_d, ws.attn_d, .{
            .u0 = @intCast(n),
            .u1 = @intCast(cfg.heads),
            .u2 = @intCast(cfg.heads),
            .u3 = @intCast(cfg.head_dim),
            .u4 = @intCast(n),
            .u5 = 0,
            .u6 = @intCast(n),
            .f0 = sc,
        }, n * cfg.heads, 1, 1);
        try attn_gpu.run(d.ctx, &ws.attn, ws.q_d, ws.k_d, ws.v_d, ws.attn_d, .{ .rows = n, .heads = cfg.heads, .kv_heads = cfg.heads, .hd = cfg.head_dim, .scale = sc }, false);
    }

    pub fn gatedAdd(d: Dev, x: Buf, dlt: Buf, n: usize, gate: usize) !void {
        const dim = d.model.cfg.dim;
        std.debug.assert(x.dt == dlt.dt);
        try d.ctx.opElt(gpu.eltAt("gated_add", x.dt), x, dlt, d.ws.mv_d, null, .{ .u0 = @intCast(n * dim), .u1 = @intCast(dim), .u2 = @intCast(gate) }, n * dim, 1, 1);
    }

    pub fn gelu(d: Dev, buf: Buf, n: usize) !void {
        try d.ctx.opElt(gpu.eltAt("gelu", buf.dt), buf, null, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
    }

    pub fn concat(d: Dev, src: Buf, dst: Buf, n: usize, src_ch: usize, dst_ch: usize, off: usize) !void {
        std.debug.assert(src.dt == dst.dt);
        try d.ctx.opElt(gpu.eltAt("concat_ch", dst.dt), src, dst, null, null, .{ .u0 = @intCast(n * src_ch), .u1 = @intCast(src_ch), .u2 = @intCast(dst_ch), .u3 = @intCast(off) }, n * src_ch, 1, 1);
    }
};
