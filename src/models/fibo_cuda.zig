//! GPU-resident Fibo 1.5 forward on the CUDA backends (`zig-cuda`'s hand-PTX and
//! `cuda`'s vendor libraries), over `fibo_dev`.
//!
//! Host: the modulation tables per sigma, the final layer. Device, once per conditioning
//! at `Session.init`: the text rows through `context_embedder` and every block's caption
//! projection, stored at the stream's dtype; then per step the latent embed and every
//! block.

const std = @import("std");
const fibo = @import("fibo.zig");
const fibo_text = @import("fibo_text.zig");
const dev = @import("fibo_dev.zig");
const lin_cuda = @import("lin_cuda.zig");
const cuda = @import("tp_gpu").cuda;
const ops = @import("tp_ops");

const DiT = fibo.DiT;
const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;
const Dt = cuda.backend.Dt;
const Weight = ops.matmul.Weight;

/// Every attention through the naive f32 kernel, so a device check can hold the rest of
/// the walk to the CPU forward.
pub var force_attn_f32: bool = false;
/// Run the integer linears weight-only, for a device check against the exact forward.
pub var force_weight_only: bool = false;

fn allLins(gpa: std.mem.Allocator, model: *const DiT) ![]Weight {
    return std.mem.concat(gpa, Weight, &.{ model.device_lins, &.{model.context_embedder.w} });
}

pub fn linPlan(gpa: std.mem.Allocator, model: *const DiT) !lin_cuda.Plan {
    const lins = try allLins(gpa, model);
    defer gpa.free(lins);
    var p = try lin_cuda.plan(lins, .{ .blockq = lin_cuda.blockq_gemm }, "fibo cuda");
    p.act16 = .bf16;
    return p;
}

pub fn supported(gpa: std.mem.Allocator, model: *const DiT) bool {
    _ = linPlan(gpa, model) catch return false;
    return lin_cuda.loraRunnable(model.lora, model.device_lins, "fibo cuda");
}

fn padRows(n: usize) usize {
    return std.mem.alignForward(usize, n, 128);
}

/// One conditioning at one resolution.
pub const Session = struct {
    lat_h: usize,
    lat_w: usize,
    n_img: usize,
    text_len: usize,
    seq: usize,
    plan: lin_cuda.Plan,
    /// `[text][dim]` after `context_embedder`, at `act`.
    txt_d: Buf,
    /// `[blocks][text_pad][dim / 2]` caption projections, at `act`.
    cap_d: Buf,
    text_pad: usize,
    /// `[seq][half]` cos, then sin.
    freqs_d: Buf,
    sigmas: []f32,
    tables: []f32,
    finals: []f32,
    table_len: usize,

    /// `ehs` `[text][joint_dim]` and `layers` `[blocks][text][te_dim]`, as
    /// `fibo_text.Context` holds them.
    pub fn init(
        gpa: std.mem.Allocator,
        io: std.Io,
        be: *Backend,
        model: *const DiT,
        lat_h: usize,
        lat_w: usize,
        c: fibo_text.Context,
        text_len: usize,
        sigmas: []const f32,
        act: Dt,
    ) !Session {
        const cfg = model.cfg;
        const dim = cfg.dim;
        const half = cfg.head_dim / 2;
        const n_img = lat_h * lat_w;
        const seq = text_len + n_img;
        const text_pad = padRows(text_len);
        const t: fibo.Table = .{ .dim = dim, .n_double = cfg.n_double, .n_single = cfg.n_single };

        var self: Session = .{
            .lat_h = lat_h,
            .lat_w = lat_w,
            .n_img = n_img,
            .text_len = text_len,
            .seq = seq,
            .plan = blk: {
                const p = try linPlan(gpa, model);
                break :blk if (force_weight_only) lin_cuda.withInt8(p, be, .weight_only_f32) else p;
            },
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
            inline for (names, 0..) |name, i| if (made > i) be.tensorDestroy(&@field(self, name));
            gpa.free(self.sigmas);
            gpa.free(self.tables);
            gpa.free(self.finals);
        }

        const lins = try allLins(gpa, model);
        defer gpa.free(lins);
        try lin_cuda.presize(be, self.plan, lins);
        try lin_cuda.attachLora(be, &self.plan, model.lora, lins, @max(padRows(seq), text_pad));

        self.txt_d = try be.tensorCreate(text_pad * dim * act.bytes());
        made += 1;
        self.txt_d.dt = act;
        self.cap_d = try be.tensorCreate(cfg.blocks() * text_pad * (dim / 2) * act.bytes());
        made += 1;
        self.cap_d.dt = act;
        {
            var f = try model.ropeFreqs(gpa, text_len, lat_h, lat_w);
            defer f.deinit(gpa);
            const flat = try gpa.alloc(f32, 2 * seq * half);
            defer gpa.free(flat);
            @memcpy(flat[0 .. seq * half], f.cos);
            @memcpy(flat[seq * half ..], f.sin);
            self.freqs_d = try be.tensorCreate(flat.len * 4);
            made += 1;
            try be.tensorUpload(self.freqs_d, std.mem.sliceAsBytes(flat));
        }
        self.sigmas = try gpa.dupe(f32, sigmas);
        self.tables = try gpa.alloc(f32, sigmas.len * self.table_len);
        self.finals = try gpa.alloc(f32, sigmas.len * 2 * dim);
        for (sigmas, 0..) |s, i| try stepTables(model, io, gpa, s, self.tables[i * self.table_len ..][0..self.table_len], self.finals[i * 2 * dim ..][0 .. 2 * dim]);

        try self.runPrefix(be, model, c);
        return self;
    }

    pub fn deinit(self: *Session, gpa: std.mem.Allocator, be: *Backend) void {
        inline for (.{ "txt_d", "cap_d", "freqs_d" }) |name| be.tensorDestroy(&@field(self, name));
        gpa.free(self.sigmas);
        gpa.free(self.tables);
        gpa.free(self.finals);
        self.* = undefined;
    }

    /// The text rows through `context_embedder` and each block's caption projection.
    fn runPrefix(self: *Session, be: *Backend, model: *const DiT, c: fibo_text.Context) !void {
        const cfg = model.cfg;
        const nt = self.text_len;
        const ehs_bytes = self.text_pad * cfg.joint_dim * 4;
        const layer_bytes = self.text_pad * cfg.te_dim * 4;
        var in = try be.tensorCreate(@max(ehs_bytes, layer_bytes));
        defer be.tensorDestroy(&in);
        try be.tensorUpload(in.slice(0, c.ehs.len * 4), std.mem.sliceAsBytes(c.ehs));
        try lin_cuda.linear(be, self.plan, self.txt_d, in, nt, model.context_embedder.w, model.context_embedder.b);
        const per = self.text_pad * (cfg.dim / 2) * self.cap_d.dt.bytes();
        for (0..cfg.blocks()) |i| {
            const l = c.layers[i * nt * cfg.te_dim ..][0 .. nt * cfg.te_dim];
            try be.tensorUpload(in.slice(0, l.len * 4), std.mem.sliceAsBytes(l));
            try lin_cuda.linear(be, self.plan, self.cap_d.slice(i * per, per), in, nt, model.caption[i], null);
        }
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

/// Activation buffers, shared by both CFG branches, so sized for the longer text.
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
    act: Dt,

    const fields = [_][]const u8{ "x_d", "nrm_d", "dlt_d", "q_d", "k_d", "v_d", "attn_d", "mg_d", "mlp_d", "cat_d", "mv_d", "imgin_d" };

    pub fn init(be: *Backend, model: *const DiT, n_img: usize, text_cap: usize, act: Dt) !Workspace {
        const cfg = model.cfg;
        const dim = cfg.dim;
        const hid = 4 * dim;
        const rows = padRows(text_cap + n_img);
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
            if (i < made) be.tensorDestroy(&@field(self, name));
        };
        inline for (fields, sizes) |name, size| {
            @field(self, name) = try be.tensorCreate(size);
            made += 1;
        }
        self.act = act;
        inline for (.{ "x_d", "nrm_d", "dlt_d", "mg_d" }) |name| @field(self, name).dt = act;
        return self;
    }

    pub fn deinit(self: *Workspace, be: *Backend) void {
        inline for (fields) |name| be.tensorDestroy(&@field(self, name));
        self.* = undefined;
    }
};

/// One denoiser forward. `out` / `x_lat` are planar `[channels][lat_h][lat_w]`.
pub fn forward(model: *const DiT, be: *Backend, sess: *Session, ws: *Workspace, io: std.Io, gpa: std.mem.Allocator, out: []f32, x_lat: []const f32, sigma: f32, cancel: ?*std.atomic.Value(bool)) !void {
    const cfg = model.cfg;
    const dim = cfg.dim;
    const n_img = sess.n_img;
    std.debug.assert(x_lat.len == cfg.channels * n_img and out.len == x_lat.len);
    std.debug.assert(ws.act == sess.txt_d.dt);

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
        try be.tensorUpload(ws.imgin_d.slice(0, tok.len * 4), std.mem.sliceAsBytes(tok));
    }
    try be.tensorUpload(ws.mv_d, std.mem.sliceAsBytes(step.table));

    try be.beginBatch();
    errdefer if (be.batching()) be.abortBatch();
    try dev.body(Dev{ .model = model, .be = be, .sess = sess, .ws = ws }, model, cancel);
    try be.endBatch();
    if (ws.act != .f32 and !try be.allFinite(ws.x_d, sess.seq * dim)) return error.NonFiniteActivation;

    const img = try gpa.alloc(f32, n_img * dim);
    defer gpa.free(img);
    const ew = ws.act.bytes();
    try be.downloadF32(ws.x_d.slice(sess.text_len * dim * ew, n_img * dim * ew), img);
    try model.finalLayer(io, gpa, out, img, step.final, n_img);
}

/// `fibo_dev`'s ops on this backend.
const Dev = struct {
    model: *const DiT,
    be: *Backend,
    sess: *Session,
    ws: *Workspace,

    pub fn nText(d: Dev) usize {
        return d.sess.text_len;
    }
    pub fn nImg(d: Dev) usize {
        return d.sess.n_img;
    }

    pub fn rows(_: Dev, buf: Buf, row0: usize, n: usize, width: usize) error{}!Buf {
        const ew = buf.dt.bytes();
        return buf.slice(row0 * width * ew, n * width * ew);
    }

    pub fn loadText(d: Dev, dst: Buf) !void {
        try d.be.tensorCopy(dst, 0, d.sess.txt_d, 0, d.sess.text_len * d.model.cfg.dim * dst.dt.bytes());
    }

    pub fn embed(d: Dev, dst: Buf) !void {
        try lin_cuda.linear(d.be, d.sess.plan, dst, d.ws.imgin_d, d.sess.n_img, d.model.x_embedder.w, d.model.x_embedder.b);
    }

    /// Block `i`'s caption projection over the second half of each text row.
    pub fn caption(d: Dev, i: usize, txt: Buf) !void {
        const dim = d.model.cfg.dim;
        const per = d.sess.text_pad * (dim / 2) * d.sess.cap_d.dt.bytes();
        try d.be.opConcatCh(d.sess.cap_d.slice(i * per, per), txt, d.sess.text_len, dim / 2, dim, dim / 2);
    }

    pub fn blockBegin(d: Dev, i: usize) !void {
        if (!d.be.async_uploads) return;
        const m = d.model;
        const next = i + 1;
        if (next < m.doubles.len) {
            for (doubleLins(&m.doubles[next])) |w| d.be.prefetchWeight(w.bytes);
        } else if (next - m.doubles.len < m.singles.len) {
            const s = &m.singles[next - m.doubles.len];
            for ([_]Weight{ s.attn.q.w, s.attn.k.w, s.attn.v.w, s.mlp.w, s.out.w }) |w| d.be.prefetchWeight(w.bytes);
        }
    }

    pub fn lnMod(d: Dev, x: Buf, out: Buf, n: usize, premul: usize, shift: usize) !void {
        const cfg = d.model.cfg;
        try d.be.lnMod(x, out, d.ws.mv_d, n, cfg.dim, premul, shift, cfg.eps);
    }

    pub fn gemms(d: Dev, x: Buf, m: usize, cols: usize, lws: []const fibo.LinearW, ys: []const Buf) !void {
        var w: [4]Weight = undefined;
        for (lws, 0..) |lw, k| w[k] = lw.w;
        const g = try lin_cuda.prep(d.be, d.sess.plan, x, m, cols, w[0..lws.len]);
        for (lws, ys) |lw, y| try lin_cuda.gemm(d.be, d.sess.plan, g, y, lw.w, lw.b);
    }

    pub fn linear(d: Dev, y: Buf, x: Buf, m: usize, lw: fibo.LinearW) !void {
        try lin_cuda.linear(d.be, d.sess.plan, y, x, m, lw.w, lw.b);
    }

    pub fn headNorm(d: Dev, buf: Buf, w: []const f32, n: usize) !void {
        const cfg = d.model.cfg;
        try d.be.qkNorm(buf, buf, try normBuf(d.be, w), n * cfg.heads, cfg.head_dim, cfg.eps);
    }

    pub fn rope(d: Dev, buf: Buf, n: usize) !void {
        const cfg = d.model.cfg;
        const half = cfg.head_dim / 2;
        try d.be.rope(buf, d.sess.freqs_d, n, cfg.heads, half, d.sess.seq * half);
    }

    pub fn scale(d: Dev, buf: Buf, n: usize, f: f32) !void {
        try d.be.opScale(buf, f, n);
    }

    pub fn attention(d: Dev, n: usize) !void {
        const cfg = d.model.cfg;
        const ws = d.ws;
        const sc: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));
        if (force_attn_f32) return d.be.attn(ws.q_d, ws.k_d, ws.v_d, ws.attn_d, n, n, cfg.heads, cfg.heads, cfg.head_dim, sc, false);
        try d.be.opAttnTC(ws.q_d, ws.k_d, ws.v_d, ws.attn_d, n, cfg.heads, cfg.heads, cfg.head_dim, sc);
    }

    pub fn gatedAdd(d: Dev, x: Buf, dlt: Buf, n: usize, gate: usize) !void {
        const dim = d.model.cfg.dim;
        try d.be.gatedAdd(x, dlt, d.ws.mv_d, n * dim, dim, gate);
    }

    pub fn gelu(d: Dev, buf: Buf, n: usize) !void {
        try d.be.gelu(buf, n);
    }

    pub fn concat(d: Dev, src: Buf, dst: Buf, n: usize, src_ch: usize, dst_ch: usize, off: usize) !void {
        try d.be.opConcatCh(src, dst, n, src_ch, dst_ch, off);
    }
};

fn doubleLins(b: *const fibo.Double) [12]Weight {
    return .{ b.img.q.w, b.img.k.w, b.img.v.w, b.txt.q.w, b.txt.k.w, b.txt.v.w, b.out.w, b.out_ctx.w, b.ff_in.w, b.ff_out.w, b.ffc_in.w, b.ffc_out.w };
}

fn normBuf(be: *Backend, weights: []const f32) !Buf {
    const h = try be.smallBuffer(std.mem.sliceAsBytes(weights));
    return .{ .buf = h, .size = weights.len * 4 };
}
