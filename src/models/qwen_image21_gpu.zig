//! GPU-resident Qwen-Image 2.1 forward on Vulkan, the twin of `qwen_image21_cuda`:
//! the same host/device split and the same `qwen_image21_dev` passes. A text run's
//! causal attention is `attn_causal_batched` (f32, one thread per query and head; the
//! prefix runs once per conditioning), everything else the tensor-core `attn_gpu`.

const std = @import("std");
const qwen_image21 = @import("qwen_image21.zig");
const dev = @import("qwen_image21_dev.zig");
const lin_gpu = @import("lin_gpu.zig");
const attn_gpu = @import("attn_gpu.zig");
const gpu = @import("tp_gpu").context;
const ops = @import("tp_ops");

const DiT = qwen_image21.DiT;
const Buf = gpu.DeviceBuffer;
const Weight = ops.matmul.Weight;
const T = dev.Table;

const eps: f32 = 1e-6;

/// Run the int8 linears weight-only, for the device check (see `qwen_image21_cuda`).
pub var force_weight_only: bool = false;
/// Every attention through the f32 kernel, for the device check.
pub var force_attn_f32: bool = false;

const lin_use: lin_gpu.Use = .{ .blockq = .auto, .act16 = .bf16 };

fn allLins(gpa: std.mem.Allocator, model: *const DiT) ![]Weight {
    return std.mem.concat(gpa, Weight, &.{ model.device_lins, model.edge_lins });
}

pub fn linPlan(ctx: *const gpu.Context, model: *const DiT) !lin_gpu.Plan {
    return lin_gpu.plan(ctx, model.device_lins, lin_use, "qwen-image-2.1 vulkan");
}

/// The rectangular attention needs tensor cores; without them a render would run
/// the per-thread kernel at canvas size, which trips the watchdog.
pub fn supported(ctx: *gpu.Context, model: *const DiT) bool {
    return attn_gpu.tensorCores(ctx) and
        lin_gpu.runs(ctx, model.device_lins, lin_use, model.lora, "qwen-image-2.1 vulkan") and
        lin_gpu.runs(ctx, model.edge_lins, lin_use, model.lora, "qwen-image-2.1 vulkan");
}

/// One conditioning, as `qwen_image21_cuda.Session`.
pub const Session = struct {
    lat_h: usize,
    lat_w: usize,
    n_img: usize,
    pre: usize,
    seq: usize,
    lin: lin_gpu.Lin,
    lin_plan: lin_gpu.Plan,
    parts: []qwen_image21.Layout.Part,
    pre_d: Buf,
    freqs_d: Buf,
    kcache_d: Buf,
    vcache_d: Buf,
    primed: bool = false,
    table0: []f32,
    sigmas: []f32,
    tables: []f32,

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
        const d = cfg.dim;
        const half = cfg.headDim() / 2;
        var lay = try qwen_image21.Layout.build(gpa, cfg, lat_h, lat_w, seq_txt, &.{});
        defer lay.deinit(gpa);
        const n_img = lat_h * lat_w;
        const pre = lay.seq - n_img;
        const tl = T.len * d;

        var l = try lin_gpu.Lin.init(gpa, ctx, lin_gpu.maxRows(model.device_lins), 0, 0);
        errdefer l.deinit();
        var lp = try linPlan(ctx, model);
        if (force_weight_only) lp.int8 = .weight_only_f32;
        if (model.lora) |st| {
            lp.lora = st;
            const lins = try allLins(gpa, model);
            defer gpa.free(lins);
            try l.presizeLora(st, lins, std.mem.alignForward(usize, @max(pre, n_img), 128));
        }
        var self: Session = .{
            .lat_h = lat_h,
            .lat_w = lat_w,
            .n_img = n_img,
            .pre = pre,
            .seq = lay.seq,
            .lin = l,
            .lin_plan = lp,
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
                if (made > i) ctx.tensorDestroy(&@field(self, name));
            }
            gpa.free(self.parts);
            gpa.free(self.table0);
            gpa.free(self.sigmas);
            gpa.free(self.tables);
        }

        self.parts = try gpa.dupe(qwen_image21.Layout.Part, lay.parts[0 .. lay.parts.len - 1]);
        {
            const txt = try model.textTokens(io, gpa, ctx_txt, seq_txt);
            defer gpa.free(txt);
            self.pre_d = try ctx.tensorCreate(@max(txt.len, 1) * 4);
            made += 1;
            try ctx.tensorUpload(self.pre_d, std.mem.sliceAsBytes(txt));
        }
        {
            var f = try ops.rope.fluxFreqs(gpa, lay.pos, &cfg.axes, qwen_image21.rope_theta);
            defer f.deinit(gpa);
            const flat = try gpa.alloc(f32, 2 * lay.seq * half);
            defer gpa.free(flat);
            @memcpy(flat[0 .. lay.seq * half], f.cos);
            @memcpy(flat[lay.seq * half ..], f.sin);
            self.freqs_d = try ctx.tensorCreate(flat.len * 4);
            made += 1;
            try ctx.tensorUpload(self.freqs_d, std.mem.sliceAsBytes(flat));
        }
        const cache = @max(cfg.layers * pre * d * 4, 4);
        self.kcache_d = try ctx.tensorCreate(cache);
        made += 1;
        self.vcache_d = try ctx.tensorCreate(cache);
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

    pub fn deinit(self: *Session, gpa: std.mem.Allocator, ctx: *gpu.Context) void {
        inline for (.{ "pre_d", "freqs_d", "kcache_d", "vcache_d" }) |name| ctx.tensorDestroy(&@field(self, name));
        gpa.free(self.parts);
        gpa.free(self.table0);
        gpa.free(self.sigmas);
        gpa.free(self.tables);
        self.lin.deinit();
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
    attn: attn_gpu.Scratch,

    const fields = [_][]const u8{ "x_d", "nrm_d", "dlt_d", "q_d", "kj_d", "vj_d", "attn_d", "mg_d", "mu_d", "mv_d", "lat_d" };

    pub fn init(ctx: *gpu.Context, model: *const DiT, n_img: usize, pre_cap: usize, act: gpu.Dt) !Workspace {
        const cfg = model.cfg;
        const d = cfg.dim;
        // GEMM outputs and the attention's f16 staging write whole 128-row tiles.
        const rows = std.mem.alignForward(usize, @max(n_img, pre_cap), 128);
        const joint = pre_cap + rows;
        const ew = act.bytes();
        const sizes = [fields.len]usize{
            rows * d * ew,
            rows * d * ew,
            rows * d * ew,
            rows * d * 4,
            joint * d * 4,
            joint * d * 4,
            rows * d * 4,
            rows * cfg.ffn * ew,
            rows * cfg.ffn * ew,
            T.len * d * 4,
            n_img * @max(cfg.in_ch, cfg.out_ch) * 4,
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
        self.attn = try attn_gpu.Scratch.initRect(ctx, @max(n_img, pre_cap), pre_cap + n_img, cfg.heads, cfg.heads, cfg.headDim(), attn_gpu.s_bytes_cap);
        inline for (.{ "x_d", "nrm_d", "dlt_d", "mg_d", "mu_d" }) |name| @field(self, name).dt = act;
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
    std.debug.assert(x_lat.len == cfg.in_ch * n_img and out.len == cfg.out_ch * n_img);
    const lins = try allLins(gpa, model);
    defer gpa.free(lins);
    try lin_gpu.presize(ctx, sess.lin_plan, lins);

    if (!sess.primed and sess.pre > 0) {
        try ctx.tensorUpload(ws.mv_d, std.mem.sliceAsBytes(sess.table0));
        try ctx.beginBatch();
        errdefer if (ctx.batching) ctx.abortBatch();
        const n = sess.pre * d;
        try ctx.opElt(gpu.eltCopy(.f32, ws.x_d.dt), sess.pre_d, ws.x_d, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
        try dev.prefix(Dev{ .model = model, .ctx = ctx, .sess = sess, .ws = ws }, model, cancel);
        try ctx.endBatch();
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
    try ctx.tensorUpload(ws.mv_d, std.mem.sliceAsBytes(tbl));
    try ctx.tensorUpload(ws.lat_d, std.mem.sliceAsBytes(tokens[0..x_lat.len]));

    try ctx.beginBatch();
    errdefer if (ctx.batching) ctx.abortBatch();
    try dev.target(Dev{ .model = model, .ctx = ctx, .sess = sess, .ws = ws }, model, cancel);
    try ctx.endBatch();
    if (ws.x_d.dt != .f32 and !try ctx.allFinite(ws.x_d, n_img * d)) return error.NonFiniteActivation;

    try ctx.downloadF32At(ws.lat_d, 0, tokens[0..out.len]);
    qwen_image21.deinterleaveChannels(out, tokens[0..out.len], n_img, cfg.out_ch);
}

/// `qwen_image21_dev`'s ops on this backend.
const Dev = struct {
    model: *const DiT,
    ctx: *gpu.Context,
    sess: *Session,
    ws: *Workspace,

    pub fn rows(d: Dev, pass: dev.Pass) usize {
        return if (pass == .prefix) d.sess.pre else d.sess.n_img;
    }

    pub fn kvRows(d: Dev, joint: Buf, pass: dev.Pass) !Buf {
        if (pass == .prefix) return joint;
        const off = d.sess.pre * d.model.cfg.dim * 4;
        return d.ctx.view(joint, off, joint.size - off);
    }

    pub fn embed(d: Dev) !void {
        const w = d.model.img_in;
        try d.sess.lin.linearAt(d.sess.lin_plan, d.ws.x_d, 0, d.ws.lat_d, 0, d.sess.n_img, w.w, null);
    }

    pub fn blockBegin(_: Dev, _: usize) !void {}

    pub fn modNorm(d: Dev, x: Buf, nrm: Buf, n: usize, premul: usize) !void {
        const dim = d.model.cfg.dim;
        try d.ctx.opLnModSg(x, nrm, d.ws.mv_d, n, dim, premul * dim, T.zero * dim, eps);
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

    pub fn qkNormRope(d: Dev, qnorm: []const f32, knorm: []const f32, k: Buf, n: usize, pass: dev.Pass) !void {
        const cfg = d.model.cfg;
        const ctx = d.ctx;
        const hd = cfg.headDim();
        const half = hd / 2;
        ctx.independent(2);
        try rmsNorm(ctx, d.ws.q_d, qnorm, n * cfg.heads, hd);
        try rmsNorm(ctx, k, knorm, n * cfg.heads, hd);
        const row0 = if (pass == .prefix) 0 else d.sess.pre;
        const f = try ctx.view(d.sess.freqs_d, row0 * half * 4, d.sess.freqs_d.size - row0 * half * 4);
        ctx.independent(2);
        inline for (.{ d.ws.q_d, k }) |buf| {
            try ctx.opElt(gpu.eltAt("rope_inter", buf.dt), buf, null, f, null, .{
                .u0 = @intCast(n * cfg.heads * half),
                .u1 = @intCast(half),
                .u2 = @intCast(d.sess.seq * half),
                .u3 = @intCast(cfg.heads),
            }, n * cfg.heads * half, 1, 1);
        }
    }

    pub fn scale(d: Dev, buf: Buf, n: usize, f: f32) !void {
        try d.ctx.opElt(.scale_f32, buf, buf, null, null, .{ .u0 = @intCast(n), .f0 = f }, n, 1, 1);
    }

    /// As `qwen_image21_cuda.Dev.attention`.
    pub fn attention(d: Dev, i: usize, pass: dev.Pass) !void {
        const cfg = d.model.cfg;
        const dim = cfg.dim;
        const ctx = d.ctx;
        const ws = d.ws;
        const pre = d.sess.pre;
        const n = pre * dim;
        switch (pass) {
            .prefix => {
                for (d.sess.parts) |p| {
                    const q = try ctx.view(ws.q_d, p.row * dim * 4, p.n * dim * 4);
                    const o = try ctx.view(ws.attn_d, p.row * dim * 4, ws.attn_d.size - p.row * dim * 4);
                    if (p.kind == .text) try causal(d, q, o, p.n, p.row, 0) else try rect(d, q, o, p.n, p.row + p.n);
                }
                ctx.independent(2);
                inline for (.{ .{ ws.kj_d, d.sess.kcache_d }, .{ ws.vj_d, d.sess.vcache_d } }) |pr| {
                    try ctx.opElt(gpu.eltCopy(.f32, .f32), pr[0], pr[1], null, null, .{ .u0 = @intCast(n), .u2 = @intCast(i * n), .u3 = 0 }, n, 1, 1);
                }
            },
            .target => {
                if (pre > 0) {
                    ctx.independent(2);
                    inline for (.{ .{ d.sess.kcache_d, ws.kj_d }, .{ d.sess.vcache_d, ws.vj_d } }) |pr| {
                        try ctx.opElt(gpu.eltCopy(.f32, .f32), pr[0], pr[1], null, null, .{ .u0 = @intCast(n), .u2 = 0, .u3 = @intCast(i * n) }, n, 1, 1);
                    }
                }
                try rect(d, ws.q_d, ws.attn_d, d.sess.n_img, pre + d.sess.n_img);
            },
        }
    }

    pub fn gatedAdd(d: Dev, x: Buf, t: Buf, n: usize, gate: usize) !void {
        const dim = d.model.cfg.dim;
        std.debug.assert(x.dt == t.dt);
        try d.ctx.opElt(gpu.eltAt("gated_add", x.dt), x, t, d.ws.mv_d, null, .{ .u0 = @intCast(n * dim), .u1 = @intCast(dim), .u2 = @intCast(gate * dim) }, n * dim, 1, 1);
    }

    pub fn siluMul(d: Dev, gate: Buf, up: Buf, n: usize) !void {
        std.debug.assert(gate.dt == up.dt);
        try d.ctx.opElt(gpu.eltAt("silu_mul", gate.dt), gate, up, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
    }

    pub fn head(d: Dev) !void {
        const n_img = d.sess.n_img;
        try d.modNorm(d.ws.x_d, d.ws.nrm_d, n_img, T.head_premul);
        const w = d.model.proj_out;
        try d.sess.lin.linearAt(d.sess.lin_plan, d.ws.lat_d, 0, d.ws.nrm_d, 0, n_img, w.w, null);
    }
};

/// `q_rows` queries at key position `pos0` onward, each seeing every key up to itself
/// (`limit` 0), or keys `[0, limit)` with no mask.
fn causal(d: Dev, q: Buf, out: Buf, q_rows: usize, pos0: usize, limit: usize) !void {
    const cfg = d.model.cfg;
    const hd = cfg.headDim();
    try d.ctx.opElt(.attn_causal_batched, q, d.ws.kj_d, d.ws.vj_d, out, .{
        .u0 = @intCast(q_rows),
        .u1 = @intCast(cfg.heads),
        .u2 = @intCast(cfg.heads),
        .u3 = @intCast(hd),
        .u4 = @intCast(q_rows),
        .u5 = @intCast(pos0),
        .u6 = @intCast(limit),
        .f0 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd))),
    }, q_rows * cfg.heads, 1, 1);
}

/// `q_rows` queries against keys `[0, kv_rows)`, unmasked.
fn rect(d: Dev, q: Buf, out: Buf, q_rows: usize, kv_rows: usize) !void {
    if (force_attn_f32) return causal(d, q, out, q_rows, 0, kv_rows);
    const cfg = d.model.cfg;
    const hd = cfg.headDim();
    const sc = &d.ws.attn;
    try attn_gpu.stageRect(d.ctx, q, d.ws.kj_d, d.ws.vj_d, sc.qh, sc.kh, sc.v16, q_rows, kv_rows, cfg.heads, cfg.heads, hd, 1.0 / @sqrt(@as(f32, @floatFromInt(hd))));
    try attn_gpu.core(d.ctx, sc.bufs(), sc.qh, sc.kh, sc.v16, out, .{
        .q_rows = q_rows,
        .kv_rows = kv_rows,
        .heads = cfg.heads,
        .kv_heads = cfg.heads,
        .hd = hd,
    }, null);
}

/// Weighted per-row RMSNorm in place, the subgroup kernel where it exists.
fn rmsNorm(ctx: *gpu.Context, x: Buf, weights: []const f32, rows: usize, dim: usize) !void {
    const w: Buf = .{ .buf = try ctx.smallBuffer(std.mem.sliceAsBytes(weights)), .mem = .null_handle, .size = 0 };
    if (ctx.hasSubgroupNorm()) return ctx.opRmsNormSg(x, x, w, rows, dim, eps);
    try ctx.opElt(gpu.eltAt("rmsnorm", x.dt), x, x, w, null, .{ .u0 = @intCast(rows), .u1 = @intCast(dim), .f0 = eps }, rows, 1, 1);
}
