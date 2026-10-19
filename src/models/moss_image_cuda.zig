//! GPU-resident MOSS-Image forward on the CUDA backends (`zig-cuda`'s hand-PTX and
//! `cuda`'s vendor libraries), the CUDA twin of `moss_image_gpu`.
//!
//! Host: the text refiner (once per conditioning: it reads no timestep) and the folded
//! modulation table (once per sigma). Device: the latent embed, all 42 blocks over the
//! joint `[image | text]` sequence and the head. Attention is `opAttnTC` at head width 128.

const std = @import("std");
const moss_image = @import("moss_image.zig");
const moss_image_dev = @import("moss_image_dev.zig");
const lin_cuda = @import("lin_cuda.zig");
const cuda = @import("tp_gpu").cuda;
const ops = @import("tp_ops");

const DiT = moss_image.DiT;
const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;
const Dt = cuda.backend.Dt;
const Weight = ops.matmul.Weight;

const eps: f32 = 1e-6;

/// Run the int8 linears weight-only, which the device check holds to the exact CPU
/// forward; the format's own W8A8 is checked by its noise instead.
pub var force_weight_only: bool = false;

pub fn linPlan(model: *const DiT) !lin_cuda.Plan {
    var p = try lin_cuda.plan(model.device_lins, .{ .blockq = lin_cuda.blockq_gemm }, "moss-image cuda");
    p.act16 = .bf16;
    return p;
}

/// One conditioning's per-image constants: its text tokens, the joint RoPE table and
/// the folded table for every sigma of the schedule.
pub const Session = struct {
    lat_h: usize,
    lat_w: usize,
    n_img: usize,
    seq_txt: usize,
    seq: usize,
    plan: lin_cuda.Plan,

    /// `[seq_txt][dim]` f32, the refined text rows.
    txt_d: Buf,
    /// `[seq][head_dim / 2]` cos, then sin.
    freqs_d: Buf,

    sigmas: []f32,
    tables: []f32,
    table_len: usize,

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
        const n_img = lat_h * lat_w;
        const seq = n_img + seq_txt;
        const half = cfg.headDim() / 2;
        var self: Session = .{
            .lat_h = lat_h,
            .lat_w = lat_w,
            .n_img = n_img,
            .seq_txt = seq_txt,
            .seq = seq,
            .plan = blk: {
                const p = try linPlan(model);
                break :blk if (force_weight_only) lin_cuda.withInt8(p, be, .weight_only_f32) else p;
            },
            .txt_d = undefined,
            .freqs_d = undefined,
            .sigmas = &.{},
            .tables = &.{},
            .table_len = moss_image.Table.len(model.blocks.len) * cfg.dim,
        };
        var made: usize = 0;
        errdefer {
            if (made > 0) be.tensorDestroy(&self.txt_d);
            if (made > 1) be.tensorDestroy(&self.freqs_d);
            if (self.sigmas.len != 0) gpa.free(self.sigmas);
            if (self.tables.len != 0) gpa.free(self.tables);
        }

        try lin_cuda.presize(be, self.plan, model.device_lins);
        const lins = try std.mem.concat(gpa, Weight, &.{ model.device_lins, model.edge_lins });
        defer gpa.free(lins);
        try lin_cuda.attachLora(be, &self.plan, model.lora, lins, seq);

        {
            const txt = try model.textTokens(io, gpa, ctx, seq_txt);
            defer gpa.free(txt);
            self.txt_d = try be.tensorCreate(txt.len * 4);
            made += 1;
            try be.tensorUpload(self.txt_d, std.mem.sliceAsBytes(txt));
        }
        {
            var f = try moss_image.ropeFreqs(gpa, cfg, lat_h, lat_w, seq_txt);
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
        for (sigmas, 0..) |s, i| {
            const t = try table(model, io, gpa, s);
            defer gpa.free(t);
            @memcpy(self.tables[i * self.table_len ..][0..self.table_len], t);
        }
        return self;
    }

    pub fn deinit(self: *Session, gpa: std.mem.Allocator, be: *Backend) void {
        be.tensorDestroy(&self.txt_d);
        be.tensorDestroy(&self.freqs_d);
        gpa.free(self.sigmas);
        gpa.free(self.tables);
        self.* = undefined;
    }

    fn cached(self: *const Session, sigma: f32) ?[]const f32 {
        for (self.sigmas, 0..) |s, i| if (s == sigma) return self.tables[i * self.table_len ..][0..self.table_len];
        return null;
    }
};

fn table(model: *const DiT, io: std.Io, gpa: std.mem.Allocator, sigma: f32) ![]f32 {
    var mod = try model.modulation(io, gpa, sigma);
    defer mod.deinit(gpa);
    return model.foldedTable(gpa, mod);
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
    mh_d: Buf,
    mg_d: Buf,
    mv_d: Buf,
    lat_d: Buf,
    /// The residual, the norm output, the delta and the FFN halves. The attention
    /// block stays f32 (`moss_image_dev.v_div`).
    act: Dt,

    const fields = [_][]const u8{ "x_d", "nrm_d", "dlt_d", "q_d", "k_d", "v_d", "attn_d", "mh_d", "mg_d", "mv_d", "lat_d" };

    pub fn init(be: *Backend, model: *const DiT, n_img: usize, seq_txt_cap: usize, act: Dt) !Workspace {
        const cfg = model.cfg;
        // A quantized route's GEMM writes whole 128-row tiles.
        const rows = std.mem.alignForward(usize, n_img + seq_txt_cap, 128);
        const d = cfg.dim;
        const sizes = [fields.len]usize{
            rows * d * 4,
            rows * d * 4,
            rows * d * 4,
            rows * d * 4,
            rows * d * 4,
            rows * d * 4,
            rows * d * 4,
            rows * cfg.ffn * 4,
            rows * cfg.ffn * 4,
            moss_image.Table.len(model.blocks.len) * d * 4,
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
        inline for (.{ "x_d", "nrm_d", "dlt_d", "mh_d", "mg_d" }) |name| @field(self, name).dt = act;
        return self;
    }

    pub fn deinit(self: *Workspace, be: *Backend) void {
        inline for (fields) |name| be.tensorDestroy(&@field(self, name));
        self.* = undefined;
    }
};

/// Whether the CUDA arms can run every block linear; a refusal is logged by name.
pub fn supported(model: *const DiT) bool {
    _ = linPlan(model) catch return false;
    return lin_cuda.loraRunnable(model.lora, model.device_lins, "moss-image cuda") and
        lin_cuda.loraRunnable(model.lora, model.edge_lins, "moss-image cuda");
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
    const d = cfg.dim;
    const n_img = sess.n_img;
    std.debug.assert(x_lat.len == cfg.channels * n_img and out.len == x_lat.len);

    var owned: ?[]f32 = null;
    defer if (owned) |b| gpa.free(b);
    const tbl = sess.cached(sigma) orelse blk: {
        owned = try table(model, io, gpa, sigma);
        break :blk owned.?;
    };
    const tokens = try gpa.alloc(f32, x_lat.len);
    defer gpa.free(tokens);
    moss_image.interleaveChannels(tokens, x_lat, n_img, cfg.channels);
    try be.tensorUpload(ws.mv_d, std.mem.sliceAsBytes(tbl));
    try be.tensorUpload(ws.lat_d, std.mem.sliceAsBytes(tokens));

    try be.beginBatch();
    errdefer if (be.batching()) be.abortBatch();
    const dev: Dev = .{ .model = model, .be = be, .sess = sess, .ws = ws, .seq = sess.seq };
    try moss_image_dev.body(dev, model, cancel);
    try be.endBatch();
    if (ws.act != .f32 and !try be.allFinite(ws.x_d, sess.seq * d)) return error.NonFiniteActivation;

    try be.downloadF32(ws.lat_d.slice(0, tokens.len * 4), tokens);
    moss_image.deinterleaveChannels(out, tokens, n_img, cfg.channels);
}

/// `moss_image_dev`'s ops on this backend.
const Dev = struct {
    model: *const DiT,
    be: *Backend,
    sess: *Session,
    ws: *Workspace,
    seq: usize,

    /// The latent rows through `video_embedder` (f32 at load), then the text rows.
    pub fn embed(d: Dev) !void {
        const dim = d.model.cfg.dim;
        const n_img = d.sess.n_img;
        try lin_cuda.linear(d.be, d.sess.plan, d.ws.x_d, d.ws.lat_d, n_img, d.model.video_embedder.w, d.model.video_embedder.b);
        const n = d.sess.seq_txt * dim;
        if (d.ws.act == .f32)
            try d.be.tensorCopy(d.ws.x_d, n_img * dim * 4, d.sess.txt_d, 0, n * 4)
        else
            try d.be.opCast(d.sess.txt_d, d.ws.x_d, n, n_img * dim);
    }

    /// Prefetch one block ahead, so each upload overlaps the previous block's compute.
    pub fn blockBegin(d: Dev, i: usize) !void {
        if (!d.be.async_uploads) return;
        const blocks = d.model.blocks;
        if (i == 0) prefetchBlock(d.be, &blocks[0]);
        if (i + 1 < blocks.len) prefetchBlock(d.be, &blocks[i + 1]);
    }

    pub fn modNorm(d: Dev, x: Buf, nrm: Buf, rows: usize, premul: usize, shift: usize) !void {
        const dim = d.model.cfg.dim;
        try d.be.rmsMod(x, nrm, d.ws.mv_d, rows, dim, premul * dim, shift * dim, eps);
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

    pub fn qkNormRope(d: Dev, qnorm: []const f32, knorm: []const f32, rows: usize) !void {
        const cfg = d.model.cfg;
        const hd = cfg.headDim();
        const half = hd / 2;
        try d.be.qkNorm(d.ws.q_d, d.ws.q_d, try normBuf(d.be, qnorm), rows * cfg.heads, hd, eps);
        try d.be.qkNorm(d.ws.k_d, d.ws.k_d, try normBuf(d.be, knorm), rows * cfg.heads, hd, eps);
        try d.be.ropeHalf(d.ws.q_d, d.sess.freqs_d, rows, cfg.heads, half, d.sess.seq * half, 0);
        try d.be.ropeHalf(d.ws.k_d, d.sess.freqs_d, rows, cfg.heads, half, d.sess.seq * half, 0);
    }

    pub fn scale(d: Dev, buf: Buf, n: usize, f: f32) !void {
        try d.be.opScale(buf, f, n);
    }

    pub fn attention(d: Dev, rows: usize) !void {
        const cfg = d.model.cfg;
        const hd = cfg.headDim();
        const sc: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        try d.be.opAttnTC(d.ws.q_d, d.ws.k_d, d.ws.v_d, d.ws.attn_d, rows, cfg.heads, cfg.heads, hd, sc);
    }

    pub fn gatedAdd(d: Dev, x: Buf, t: Buf, rows: usize, gate: usize) !void {
        const dim = d.model.cfg.dim;
        try d.be.gatedAdd(x, t, d.ws.mv_d, rows * dim, dim, gate * dim);
    }

    pub fn situ(d: Dev, hidden: Buf, gate: Buf, n: usize) !void {
        try d.be.situ(hidden, gate, n);
    }

    /// The head over the image rows: its modulated norm, then `final_linear_video`
    /// (f32 at load) into `lat_d`, whose tokens the embed has finished with.
    pub fn head(d: Dev) !void {
        const T = moss_image.Table;
        const n_img = d.sess.n_img;
        try d.modNorm(d.ws.x_d, d.ws.nrm_d, n_img, T.head_premul, T.head_shift);
        try lin_cuda.linear(d.be, d.sess.plan, d.ws.lat_d, d.ws.nrm_d, n_img, d.model.final_linear.w, null);
    }
};

/// Keys must be the same host pointers the forward fetches, hence the row views.
fn prefetchBlock(be: *Backend, blk: *const moss_image.Block) void {
    for (blk.dev) |w| be.prefetchWeight(w.bytes);
    be.prefetchWeight(std.mem.sliceAsBytes(blk.q_norm));
    be.prefetchWeight(std.mem.sliceAsBytes(blk.k_norm));
}

fn normBuf(be: *Backend, weights: []const f32) !Buf {
    const h = try be.smallBuffer(std.mem.sliceAsBytes(weights));
    return .{ .buf = h, .size = weights.len * 4 };
}
