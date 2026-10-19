//! GPU-resident MOSS-Image forward on Vulkan, the twin of `moss_image_cuda`: the same
//! host/device split (text refiner per conditioning, folded table per sigma on the
//! host), the same `moss_image_dev.body`.

const std = @import("std");
const moss_image = @import("moss_image.zig");
const moss_image_dev = @import("moss_image_dev.zig");
const lin_gpu = @import("lin_gpu.zig");
const attn_gpu = @import("attn_gpu.zig");
const gpu = @import("tp_gpu").context;
const ops = @import("tp_ops");

const DiT = moss_image.DiT;
const Buf = gpu.DeviceBuffer;
const Weight = ops.matmul.Weight;

const eps: f32 = 1e-6;

/// Run the int8 linears weight-only, for the device check (see `moss_image_cuda`).
pub var force_weight_only: bool = false;
/// The reference attention, for the device check.
pub var force_attn_full: bool = false;

const lin_use: lin_gpu.Use = .{ .blockq = .auto, .act16 = .bf16 };

pub fn linPlan(ctx: *const gpu.Context, model: *const DiT) !lin_gpu.Plan {
    return lin_gpu.plan(ctx, model.device_lins, lin_use, "moss-image vulkan");
}

pub fn supported(ctx: *gpu.Context, model: *const DiT) bool {
    return lin_gpu.runs(ctx, model.device_lins, lin_use, model.lora, "moss-image vulkan") and
        lin_gpu.runs(ctx, model.edge_lins, lin_use, model.lora, "moss-image vulkan");
}

pub const Session = struct {
    lat_h: usize,
    lat_w: usize,
    n_img: usize,
    seq_txt: usize,
    seq: usize,
    lin: lin_gpu.Lin,
    lin_plan: lin_gpu.Plan,
    txt_d: Buf,
    freqs_d: Buf,
    sigmas: []f32,
    tables: []f32,
    table_len: usize,

    pub fn init(
        gpa: std.mem.Allocator,
        io: std.Io,
        ctx: *gpu.Context,
        model: *const DiT,
        lat_h: usize,
        lat_w: usize,
        ctx_txt: []const f32,
        seq_txt: usize,
        sigmas: []const f32,
    ) !Session {
        const cfg = model.cfg;
        const n_img = lat_h * lat_w;
        const seq = n_img + seq_txt;
        const half = cfg.headDim() / 2;
        var l = try lin_gpu.Lin.init(gpa, ctx, lin_gpu.maxRows(model.device_lins), 0, 0);
        errdefer l.deinit();
        var lp = try linPlan(ctx, model);
        if (force_weight_only) lp.int8 = .weight_only_f32;
        if (model.lora) |st| {
            lp.lora = st;
            const lins = try std.mem.concat(gpa, Weight, &.{ model.device_lins, model.edge_lins });
            defer gpa.free(lins);
            try l.presizeLora(st, lins, std.mem.alignForward(usize, seq, 128));
        }
        var self: Session = .{
            .lat_h = lat_h,
            .lat_w = lat_w,
            .n_img = n_img,
            .seq_txt = seq_txt,
            .seq = seq,
            .lin = l,
            .lin_plan = lp,
            .txt_d = undefined,
            .freqs_d = undefined,
            .sigmas = &.{},
            .tables = &.{},
            .table_len = moss_image.Table.len(model.blocks.len) * cfg.dim,
        };
        var made: usize = 0;
        errdefer {
            if (made > 0) ctx.tensorDestroy(&self.txt_d);
            if (made > 1) ctx.tensorDestroy(&self.freqs_d);
            if (self.sigmas.len != 0) gpa.free(self.sigmas);
            if (self.tables.len != 0) gpa.free(self.tables);
        }
        {
            const txt = try model.textTokens(io, gpa, ctx_txt, seq_txt);
            defer gpa.free(txt);
            self.txt_d = try ctx.tensorCreate(txt.len * 4);
            made += 1;
            try ctx.tensorUpload(self.txt_d, std.mem.sliceAsBytes(txt));
        }
        {
            var f = try moss_image.ropeFreqs(gpa, cfg, lat_h, lat_w, seq_txt);
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
        for (sigmas, 0..) |s, i| {
            const t = try table(model, io, gpa, s);
            defer gpa.free(t);
            @memcpy(self.tables[i * self.table_len ..][0..self.table_len], t);
        }
        return self;
    }

    pub fn deinit(self: *Session, gpa: std.mem.Allocator, ctx: *gpu.Context) void {
        ctx.tensorDestroy(&self.txt_d);
        ctx.tensorDestroy(&self.freqs_d);
        gpa.free(self.sigmas);
        gpa.free(self.tables);
        self.lin.deinit();
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
    attn: attn_gpu.Scratch,

    const fields = [_][]const u8{ "x_d", "nrm_d", "dlt_d", "q_d", "k_d", "v_d", "attn_d", "mh_d", "mg_d", "mv_d", "lat_d" };

    pub fn init(ctx: *gpu.Context, model: *const DiT, n_img: usize, seq_txt_cap: usize, act: gpu.Dt) !Workspace {
        const cfg = model.cfg;
        const seq = n_img + seq_txt_cap;
        // GEMM outputs and the attention's f16 staging write whole 128-row tiles.
        const rows = std.mem.alignForward(usize, seq, 128);
        const d = cfg.dim;
        const ew = act.bytes();
        const sizes = [fields.len]usize{
            seq * d * ew,
            seq * d * ew,
            rows * d * ew,
            rows * d * 4,
            rows * d * 4,
            rows * d * 4,
            rows * d * 4,
            rows * cfg.ffn * ew,
            rows * cfg.ffn * ew,
            moss_image.Table.len(model.blocks.len) * d * 4,
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
        self.attn = try attn_gpu.Scratch.init(ctx, seq, cfg.heads, cfg.heads, cfg.headDim(), attn_gpu.s_bytes_cap);
        inline for (.{ "x_d", "nrm_d", "dlt_d", "mh_d", "mg_d" }) |name| @field(self, name).dt = act;
        return self;
    }

    pub fn deinit(self: *Workspace, ctx: *gpu.Context) void {
        inline for (fields) |name| ctx.tensorDestroy(&@field(self, name));
        self.attn.deinit(ctx);
        self.* = undefined;
    }
};

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
    try ctx.tensorUpload(ws.mv_d, std.mem.sliceAsBytes(tbl));
    try ctx.tensorUpload(ws.lat_d, std.mem.sliceAsBytes(tokens));
    try lin_gpu.presize(ctx, sess.lin_plan, model.device_lins);

    try ctx.beginBatch();
    errdefer if (ctx.batching) ctx.abortBatch();
    const dev: Dev = .{ .model = model, .ctx = ctx, .sess = sess, .ws = ws, .seq = sess.seq };
    try moss_image_dev.body(dev, model, cancel);
    try ctx.endBatch();
    if (ws.x_d.dt != .f32 and !try ctx.allFinite(ws.x_d, sess.seq * d)) return error.NonFiniteActivation;

    try ctx.downloadF32At(ws.lat_d, 0, tokens);
    moss_image.deinterleaveChannels(out, tokens, n_img, cfg.channels);
}

const Dev = struct {
    model: *const DiT,
    ctx: *gpu.Context,
    sess: *Session,
    ws: *Workspace,
    seq: usize,

    pub fn embed(d: Dev) !void {
        const dim = d.model.cfg.dim;
        const n_img = d.sess.n_img;
        const ve = d.model.video_embedder;
        try d.sess.lin.linearAt(d.sess.lin_plan, d.ws.x_d, 0, d.ws.lat_d, 0, n_img, ve.w, ve.b);
        const n = d.sess.seq_txt * dim;
        try d.ctx.opElt(gpu.eltCopy(.f32, d.ws.x_d.dt), d.sess.txt_d, d.ws.x_d, null, null, .{ .u0 = @intCast(n), .u2 = @intCast(n_img * dim), .u3 = 0 }, n, 1, 1);
    }

    pub fn blockBegin(_: Dev, _: usize) !void {}

    pub fn modNorm(d: Dev, x: Buf, nrm: Buf, rows: usize, premul: usize, shift: usize) !void {
        const dim = d.model.cfg.dim;
        try d.ctx.opRmsModSg(x, nrm, d.ws.mv_d, rows, dim, premul * dim, shift * dim, eps);
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

    pub fn qkNormRope(d: Dev, qnorm: []const f32, knorm: []const f32, rows: usize) !void {
        const cfg = d.model.cfg;
        const ctx = d.ctx;
        const hd = cfg.headDim();
        const half = hd / 2;
        ctx.independent(2);
        try rmsNorm(ctx, d.ws.q_d, qnorm, rows * cfg.heads, hd);
        try rmsNorm(ctx, d.ws.k_d, knorm, rows * cfg.heads, hd);
        ctx.independent(2);
        inline for (.{ d.ws.q_d, d.ws.k_d }) |buf| {
            try ctx.opElt(.rope_half, buf, null, d.sess.freqs_d, null, .{
                .u0 = @intCast(rows * cfg.heads * half),
                .u1 = @intCast(half),
                .u2 = @intCast(d.sess.seq * half),
                .u3 = @intCast(cfg.heads),
            }, rows * cfg.heads * half, 1, 1);
        }
    }

    pub fn scale(d: Dev, buf: Buf, n: usize, f: f32) !void {
        try d.ctx.opElt(.scale_f32, buf, buf, null, null, .{ .u0 = @intCast(n), .f0 = f }, n, 1, 1);
    }

    pub fn attention(d: Dev, rows: usize) !void {
        const cfg = d.model.cfg;
        const hd = cfg.headDim();
        try attn_gpu.run(d.ctx, &d.ws.attn, d.ws.q_d, d.ws.k_d, d.ws.v_d, d.ws.attn_d, .{
            .rows = rows,
            .heads = cfg.heads,
            .kv_heads = cfg.heads,
            .hd = hd,
            .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd))),
        }, force_attn_full);
    }

    pub fn gatedAdd(d: Dev, x: Buf, t: Buf, rows: usize, gate: usize) !void {
        const dim = d.model.cfg.dim;
        std.debug.assert(x.dt == t.dt);
        try d.ctx.opElt(gpu.eltAt("gated_add", x.dt), x, t, d.ws.mv_d, null, .{ .u0 = @intCast(rows * dim), .u1 = @intCast(dim), .u2 = @intCast(gate * dim) }, rows * dim, 1, 1);
    }

    pub fn situ(d: Dev, hidden: Buf, gate: Buf, n: usize) !void {
        std.debug.assert(hidden.dt == gate.dt);
        try d.ctx.opElt(gpu.eltAt("situ", hidden.dt), hidden, gate, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
    }

    /// The head's modulated norm, then `final_linear_video` (f32 at load) into `lat_d`.
    pub fn head(d: Dev) !void {
        const T = moss_image.Table;
        const n_img = d.sess.n_img;
        try d.modNorm(d.ws.x_d, d.ws.nrm_d, n_img, T.head_premul, T.head_shift);
        const fl = d.model.final_linear;
        try d.sess.lin.linearAt(d.sess.lin_plan, d.ws.lat_d, 0, d.ws.nrm_d, 0, n_img, fl.w, fl.b);
    }
};

/// Weighted per-row RMSNorm in place, the subgroup kernel where it exists.
fn rmsNorm(ctx: *gpu.Context, x: Buf, weights: []const f32, rows: usize, dim: usize) !void {
    const w: Buf = .{ .buf = try ctx.smallBuffer(std.mem.sliceAsBytes(weights)), .mem = .null_handle, .size = 0 };
    if (ctx.hasSubgroupNorm()) return ctx.opRmsNormSg(x, x, w, rows, dim, eps);
    try ctx.opElt(gpu.eltAt("rmsnorm", x.dt), x, x, w, null, .{ .u0 = @intCast(rows), .u1 = @intCast(dim), .f0 = eps }, rows, 1, 1);
}
