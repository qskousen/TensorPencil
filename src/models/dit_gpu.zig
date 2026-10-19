//! GPU-resident Krea 2 DiT forward: the whole 28-block chain runs on the
//! device (GEMMs, norms, modulation, RoPE, attention, gating) with a single
//! upload of the combined sequence going in and a single download of the
//! image tokens coming out. Small paths (timestep MLPs, txtfusion, patchify,
//! final layer) stay on the CPU where they are cheap.
//!
//! Numerics match the CPU forward up to floating-point reordering; parity is
//! tested against `DiT.forward` (gated on testdata/gpu-tests).

const std = @import("std");
const dit = @import("dit.zig");
const dit_dev = @import("dit_dev.zig");
const lin = @import("lin.zig");
const lin_gpu = @import("lin_gpu.zig");
const attn_gpu = @import("attn_gpu.zig");
const Weight = lin_gpu.Weight;
const gpu = @import("tp_gpu").context;
const ops = @import("tp_ops");

const DiT = dit.DiT;

/// When true, each forward prints a per-category time breakdown (sync-per-op
/// submission makes host-side timing exact).
pub var profile: bool = false;

/// When true, force the full-f32 path (opMatmul GEMMs + eltwise f32
/// attention) even where the tensor-core / coop pipelines exist, the
/// hardware-fallback path, exposed for A/B (speed vs f16 rounding). Must be
/// set before Workspace.init, which sizes its buffers from the same choice.
pub var force_f32: bool = false;

const Prof = struct {
    matmul_ns: i96 = 0,
    attn_ns: i96 = 0,
    scores_ns: i96 = 0,
    smax_ns: i96 = 0,
    aout_ns: i96 = 0,
    elt_ns: i96 = 0,
    prep_ns: i96 = 0, // int8 rotate/rowscale/quantize prep (opI8Prep)
    xfer_ns: i96 = 0,
    cpu_ns: i96 = 0,
};
const F = dit.features;
const heads = dit.n_heads;
const kv_heads = dit.n_kv_heads;
const hd = dit.head_dim;
const attn_scale: f32 = 1.0 / 11.313708498984761; // 1/sqrt(128)

/// Flash attention (coopmat.buildFlashAttn) keeps the {m, 1/d} table in the
/// tail of attn_d and never materializes S, but measured SLOWER than the
/// two-pass path at DiT sizes (attn 1.49 -> 2.43 s at 1120x1680): the out
/// pass recomputes S at the global-fragment-load rate (~0.8 s), which costs
/// more than the coalesced S write + reads it eliminates (~0.5 s). It would
/// win only with both operands staged in shared, which doesn't fit the 48 KB
/// workgroup ceiling next to the S tile at this tiling. Kept behind this
/// switch (kernels are unit-tested) for smaller-seq experiments or a future
/// retiling.
const use_flash = false;
/// Two-pass softmax chunk count: 32 interleaved chunks per row so a warp covers a
/// row with coalesced reads.
const nchunks = 32;
/// Cap on the materialized attention-scores buffer; heads batch to fit it.
const s_bytes_cap: usize = 2 << 30;

/// Per-sampling-run cache: everything constant across steps, the text
/// fusion tokens (a full CPU transformer pass), the rope table, and the
/// timestep vectors for the whole schedule, computed once and, where
/// device-resident, uploaded once.
const lin_use: lin_gpu.Use = .{ .blockq = .auto };

/// Whether this context can run every block linear `model` has.
pub fn supported(ctx: *gpu.Context, model: *const DiT) bool {
    return lin_gpu.runs(ctx, model.device_lins, lin_use, model.lora, "dit_gpu");
}

/// Whether every weight of a group takes `want`. The fused paths below read the
/// weight in ONE storage form and would read any other as its bytes, so each asks
/// this positively rather than ruling formats out. They also skip the LoRA sidecar,
/// so a patched weight takes the dispatcher.
fn allRoute(sess: *const Session, ws: []const Weight, m: usize, want: lin_gpu.Route) bool {
    for (ws) |w| if (patched(sess, w) or sess.lin.routeAt(sess.lin_plan, w, m) != want) return false;
    return true;
}

fn patched(sess: *const Session, w: Weight) bool {
    const st = sess.lin_plan.lora orelse return false;
    return st.forWeight(w).len > 0;
}

/// Whether every weight of a group runs on the int8 prep (int8, int4 and W4A8 alike).
fn allI8(sess: *const Session, ws: []const Weight, m: usize) bool {
    for (ws) |w| {
        if (patched(sess, w)) return false;
        const r = sess.lin.routeAt(sess.lin_plan, w, m) orelse return false;
        if (r.prep() != .i8) return false;
    }
    return true;
}

/// One group of linears over `x` through the dispatcher, for a group no fused path
/// below takes.
fn linGroup(sess: *const Session, p: lin_gpu.Plan, ys: []const gpu.DeviceBuffer, x: gpu.DeviceBuffer, m: usize, m_pad: usize, ws: []const Weight) !void {
    const g = try sess.lin.prep(p, x, m, ws[0].cols, ws);
    for (ys, ws) |y, w| try sess.lin.gemm(p, g, y, w, .{ .m_pad = m_pad });
}

pub const Session = struct {
    seq_txt: usize,
    lat_h: usize,
    lat_w: usize,
    /// Text tokens uploaded once; copied device-to-device into x each step.
    txt0_d: gpu.DeviceBuffer,
    txt_len: usize,
    freqs_d: gpu.DeviceBuffer,
    /// Steering directions, uploaded once. Empty when steering is off.
    dirs_d: gpu.DeviceBuffer = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 },
    /// Per schedule entry: t (F) then tvec (6F).
    sigmas: []f32,
    tvs: []f32,
    /// Every block linear's dispatch, and what it decided for this checkpoint.
    lin: lin_gpu.Lin,
    lin_plan: lin_gpu.Plan,

    pub fn init(
        gpa: std.mem.Allocator,
        io: std.Io,
        ctx: *gpu.Context,
        model: *const DiT,
        lat_h: usize,
        lat_w: usize,
        cond: []const f32,
        seq_txt: usize,
        sigmas: []const f32,
    ) !Session {
        const h = lat_h / dit.patch;
        const w = lat_w / dit.patch;
        const seq = seq_txt + h * w;
        const half = hd / 2;
        var l = try lin_gpu.Lin.init(gpa, ctx, lin_gpu.maxRows(model.device_lins), 0, 0);
        errdefer l.deinit();
        var lp = try lin_gpu.plan(ctx, model.device_lins, lin_use, "dit_gpu");
        if (model.lora) |st| {
            lp.lora = st;
            try l.presizeLora(st, model.device_lins, std.mem.alignForward(usize, seq, 128));
        }

        const txt_tokens = try model.textTokens(io, gpa, cond, seq_txt);
        defer gpa.free(txt_tokens);
        var txt0_d = try ctx.tensorCreate(txt_tokens.len * 4);
        errdefer ctx.tensorDestroy(&txt0_d);
        try ctx.tensorUpload(txt0_d, std.mem.sliceAsBytes(txt_tokens));

        var freqs = try DiT.ropeFreqs(gpa, seq_txt, h, w);
        defer freqs.deinit(gpa);
        std.debug.assert(freqs.half == half);
        const fp = try gpa.alloc(f32, 2 * seq * half);
        defer gpa.free(fp);
        @memcpy(fp[0 .. seq * half], freqs.cos);
        @memcpy(fp[seq * half ..], freqs.sin);
        // The model carries the steering set before a session is built, so the
        // directions upload once here rather than per forward.
        var dirs_d: gpu.DeviceBuffer = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 };
        errdefer ctx.tensorDestroy(&dirs_d);
        if (model.act_steer) |st| if (st.dirs.len > 0) {
            dirs_d = try ctx.tensorCreate(st.dirs.len * 4);
            try ctx.tensorUpload(dirs_d, std.mem.sliceAsBytes(st.dirs));
        };

        var freqs_d = try ctx.tensorCreate(fp.len * 4);
        errdefer ctx.tensorDestroy(&freqs_d);
        try ctx.tensorUpload(freqs_d, std.mem.sliceAsBytes(fp));

        const sig = try gpa.dupe(f32, sigmas);
        errdefer gpa.free(sig);
        const tvs = try gpa.alloc(f32, sigmas.len * 7 * F);
        errdefer gpa.free(tvs);
        for (sigmas, 0..) |sg, i| {
            const tvv = try model.timestepVectors(io, gpa, sg);
            defer gpa.free(tvv.t);
            defer gpa.free(tvv.tvec);
            @memcpy(tvs[i * 7 * F ..][0..F], tvv.t);
            @memcpy(tvs[i * 7 * F + F ..][0 .. 6 * F], tvv.tvec);
        }

        return .{
            .seq_txt = seq_txt,
            .lat_h = lat_h,
            .lat_w = lat_w,
            .dirs_d = dirs_d,
            .txt0_d = txt0_d,
            .txt_len = txt_tokens.len,
            .freqs_d = freqs_d,
            .sigmas = sig,
            .tvs = tvs,
            .lin = l,
            .lin_plan = lp,
        };
    }

    pub fn deinit(self: *Session, gpa: std.mem.Allocator, ctx: *gpu.Context) void {
        ctx.tensorDestroy(&self.txt0_d);
        ctx.tensorDestroy(&self.freqs_d);
        ctx.tensorDestroy(&self.dirs_d);
        gpa.free(self.sigmas);
        gpa.free(self.tvs);
        self.lin.deinit();
        self.* = undefined;
    }

    fn tv(self: *const Session, sigma: f32) ?struct { t: []const f32, tvec: []const f32 } {
        for (self.sigmas, 0..) |sg, i| {
            if (sg == sigma) {
                return .{
                    .t = self.tvs[i * 7 * F ..][0..F],
                    .tvec = self.tvs[i * 7 * F + F ..][0 .. 6 * F],
                };
            }
        }
        return null;
    }
};

/// Accumulate the time since `m` into `bucket` and restamp. Only meaningful when the
/// forward is NOT batching (`--profile on`), where each op is submit-and-wait; inside a
/// recording batch it would time recording, not execution.
fn lapNs(io_: std.Io, m: *std.Io.Timestamp, bucket: *i96) void {
    const now = std.Io.Clock.real.now(io_);
    bucket.* += now.nanoseconds - m.nanoseconds;
    m.* = now;
}


/// Per-run device activation buffers, allocated once and reused every step. Allocating
/// per forward means ~20 create/destroy pairs per call, including the ~2 GiB scores
/// buffer, and each `tensorCreate` is a raw `vkAllocateMemory`.
/// Sized for a maximum sequence; one Workspace is shared by the positive and
/// negative CFG sessions (they differ only in text length). Every kernel
/// writes the regions it later reads (including zero padding), so reuse
/// across steps and sessions needs no clearing.
/// The dtype the residual stream and the projection outputs are stored at when `want`
/// is asked for. An activation capture reads what the f32 stream holds.
pub fn streamDtFor(want: gpu.Dt) gpu.Dt {
    if (want == .f32 or force_f32 or ops.matmul.probe != null) return .f32;
    return want;
}

pub const Workspace = struct {
    seq_cap: usize,
    /// The storage of `x_d` and `t1_d`, and of every GEMM output written into them.
    act: gpu.Dt,
    x_d: gpu.DeviceBuffer,
    t1_d: gpu.DeviceBuffer,
    q_d: gpu.DeviceBuffer,
    k_d: gpu.DeviceBuffer,
    v_d: gpu.DeviceBuffer,
    g_d: gpu.DeviceBuffer,
    attn_d: gpu.DeviceBuffer,
    mg_d: gpu.DeviceBuffer,
    mu_d: gpu.DeviceBuffer,
    mv_d: gpu.DeviceBuffer,
    fin_d: gpu.DeviceBuffer,
    imgin_d: gpu.DeviceBuffer,
    qt_d: gpu.DeviceBuffer,
    kt_d: gpu.DeviceBuffer,
    s_d: gpu.DeviceBuffer,
    v16_d: gpu.DeviceBuffer,
    part_d: gpu.DeviceBuffer,
    md_d: gpu.DeviceBuffer,
    h16_d: gpu.DeviceBuffer,

    pub fn init(ctx: *gpu.Context, lat_h: usize, lat_w: usize, seq_txt_cap: usize, act: gpu.Dt) !Workspace {
        const n_img = (lat_h / dit.patch) * (lat_w / dit.patch);
        const seq = seq_txt_cap + n_img;
        const seq_pad = std.mem.alignForward(usize, seq, 128);
        const coop = !force_f32 and ctx.pipe_coop != .null_handle;
        const tc_attn = !force_f32 and ctx.pipe_scores != .null_handle;
        const flash = use_flash and ctx.pipe_flash_md != .null_handle;
        const s_rows = if (tc_attn) seq_pad else seq;
        const s_esize: usize = if (tc_attn) 2 else 4;
        const hpb = headsPerBatch(s_rows, s_esize, scoresCap(ctx.budget_override), null);

        var ws: Workspace = undefined;
        ws.seq_cap = seq;
        ws.act = act;
        var created: usize = 0;
        errdefer inline for (buf_fields, 0..) |name, i| {
            if (i < created) ctx.tensorDestroy(&@field(ws, name));
        };
        const sizes = [buf_fields.len]usize{
            seq_pad * F * 4, // x_d
            seq_pad * F * 4, // t1_d
            seq_pad * heads * hd * 4, // q_d
            seq_pad * kv_heads * hd * 4, // k_d
            seq_pad * kv_heads * hd * 4, // v_d
            seq_pad * F * 4, // g_d
            seq_pad * F * 4 + if (flash) heads * seq_pad * 2 * 4 else 0, // attn_d
            seq_pad * dit.mlp_dim * 4, // mg_d
            seq_pad * dit.mlp_dim * 4, // mu_d
            dit.n_blocks * 6 * F * 4, // mv_d
            2 * F * 4, // fin_d
            n_img * dit.channels * dit.patch * dit.patch * 4, // imgin_d
            if (tc_attn) seq_pad * heads * hd * 2 else seq * heads * hd * 4 + 32, // qt_d
            if (tc_attn) kv_heads * hd * seq_pad * 2 else seq * kv_heads * hd * 4 + 32, // kt_d
            if (flash) 16 else hpb * s_rows * s_rows * s_esize, // s_d
            if (tc_attn) seq_pad * kv_heads * hd * 2 else 16, // v16_d
            if (tc_attn and !flash) hpb * seq * nchunks * 2 * 4 else 16, // part_d
            if (tc_attn and !flash) hpb * seq_pad * 2 * 4 else 16, // md_d
            if (coop) seq_pad * dit.mlp_dim * 2 else 16, // h16_d
        };
        inline for (buf_fields, sizes) |name, size| {
            @field(ws, name) = try ctx.tensorCreate(size);
            created += 1;
        }
        ws.x_d.dt = act;
        ws.t1_d.dt = act;
        return ws;
    }

    pub fn deinit(self: *Workspace, ctx: *gpu.Context) void {
        inline for (buf_fields) |name| ctx.tensorDestroy(&@field(self, name));
        self.* = undefined;
    }

    const buf_fields = [_][]const u8{
        "x_d",  "t1_d",  "q_d",    "k_d",  "v_d",     "g_d",  "attn_d",
        "mg_d", "mu_d",  "mv_d",   "fin_d", "imgin_d", "qt_d", "kt_d",
        "s_d",  "v16_d", "part_d", "md_d", "h16_d",
    };
};

/// How many heads' scores planes fit the byte cap (and, when reusing a
/// Workspace built for a longer sequence, its actual buffer).
fn headsPerBatch(s_rows: usize, s_esize: usize, cap: usize, ws_s_bytes: ?usize) usize {
    const plane = s_rows * s_rows * s_esize;
    var hb = @max(1, @min(heads, cap / plane));
    if (ws_s_bytes) |c| hb = @max(1, @min(hb, c / plane));
    return hb;
}

/// Byte budget for the materialized attention-scores buffer. Defaults to the
/// 2 GiB cap; a `--vram-budget` shrinks it (attention batches over more head
/// groups, trading a few launches for a much smaller `s_d`, the single biggest
/// activation buffer at high resolution). Floored so at least one head fits.
fn scoresCap(budget: u64) usize {
    if (budget == 0) return s_bytes_cap;
    return @min(s_bytes_cap, @max(64 << 20, budget / 4));
}

pub fn forward(
    model: *const DiT,
    ctx: *gpu.Context,
    sess: *const Session,
    ws: *const Workspace,
    io: std.Io,
    gpa: std.mem.Allocator,
    out: []f32,
    x_lat: []const f32,
    sigma: f32,
    cancel: ?*std.atomic.Value(bool),
) !void {
    try lin_gpu.presize(ctx, sess.lin_plan, model.device_lins);

    const lat_h = sess.lat_h;
    const lat_w = sess.lat_w;
    const seq_txt = sess.seq_txt;
    const n_img = (lat_h / dit.patch) * (lat_w / dit.patch);
    const seq = seq_txt + n_img;
    // Cooperative-matrix GEMMs compute a tile-padded m (128-row workgroup tiles);
    // buffers that receive GEMM output are sized for seq_pad (pad rows stay zero).
    const seq_pad = std.mem.alignForward(usize, seq, 128);
    // An activation capture forces the f32 GEMM path, for coverage first and fidelity
    // second: the staged fp8 path hands the dispatcher an f16 activation, which a probe
    // cannot record as the f32 quantity the CPU reference does.
    const probing = ops.matmul.probe != null;
    const coop = !force_f32 and !probing and ctx.pipe_coop != .null_handle;
    var prof: Prof = .{};
    var t_mark = std.Io.Clock.real.now(io);

    // Timestep vectors cached in the session for the schedule's sigmas.
    var tv_owned: bool = false;
    var tv_t: []const f32 = undefined;
    var tv_vec: []const f32 = undefined;
    if (sess.tv(sigma)) |c| {
        tv_t = c.t;
        tv_vec = c.tvec;
    } else {
        const tvv = try model.timestepVectors(io, gpa, sigma);
        tv_owned = true;
        tv_t = tvv.t;
        tv_vec = tvv.tvec;
    }
    defer if (tv_owned) {
        gpa.free(tv_t);
        gpa.free(tv_vec);
    };
    const mv = try gpa.alloc(f32, dit.n_blocks * 6 * F);
    defer gpa.free(mv);
    const fin = try gpa.alloc(f32, 2 * F);
    defer gpa.free(fin);
    model.modulation(tv_t, tv_vec, mv, fin);
    const img_in = try DiT.patchify(gpa, x_lat, lat_h, lat_w);
    defer gpa.free(img_in);
    lapNs(io, &t_mark, &prof.cpu_ns);

    std.debug.assert(seq <= ws.seq_cap);
    // With the tensor-core scores pipeline Q converts to f16 (softmax scale prefolded)
    // and K gathers into a per-head k-major f16 block with seq_pad columns; the f32
    // fallback keeps k-major f32 gathers.
    comptime std.debug.assert(hd == 128); // buildGemmScores unrolls hd/16
    const tc_attn = !force_f32 and ctx.pipe_scores != .null_handle;
    const flash = use_flash and ctx.pipe_flash_md != .null_handle;
    // Scores batched over head groups to bound VRAM and fit the Workspace buffers, which
    // may be sized for a longer sequence.
    const s_rows = if (tc_attn) seq_pad else seq;
    const s_esize: usize = if (tc_attn) 2 else 4; // the coop path stores S f16
    var heads_per_batch = headsPerBatch(s_rows, s_esize, s_bytes_cap, ws.s_d.size);
    if (tc_attn and !flash) {
        heads_per_batch = @max(1, @min(heads_per_batch, ws.part_d.size / (seq * nchunks * 2 * 4)));
        heads_per_batch = @max(1, @min(heads_per_batch, ws.md_d.size / (seq_pad * 2 * 4)));
    }

    try ctx.tensorUpload(ws.mv_d, std.mem.sliceAsBytes(mv));
    try ctx.tensorUpload(ws.fin_d, std.mem.sliceAsBytes(fin));
    if (ws.act == .f32)
        try ctx.tensorCopy(ws.x_d, 0, sess.txt0_d, 0, sess.txt_len * 4)
    else
        try ctx.opElt(gpu.eltCopy(.f32, ws.act), sess.txt0_d, ws.x_d, null, null, .{ .u0 = @intCast(sess.txt_len) }, sess.txt_len, 1, 1);
    try ctx.tensorUpload(ws.imgin_d, std.mem.sliceAsBytes(img_in));
    lapNs(io, &t_mark, &prof.xfer_ns);

    // Without the coop GEMMs (`force_f32`, a capture) fp8 runs against the f32
    // activation, which on this backend is what `bf16` staging resolves to.
    var gp = sess.lin_plan;
    if (!coop) gp.act16 = .bf16;

    // One submission for the whole forward; profiling stays sync-per-op so the
    // per-category host times remain exact.
    const batched = !profile;
    if (batched) try ctx.beginBatch();
    errdefer if (ctx.batching) ctx.abortBatch();
    const d: Dev = .{
        .model = model,
        .ctx = ctx,
        .sess = sess,
        .ws = ws,
        .io = io,
        .gpa = gpa,
        .prof = &prof,
        .t_mark = &t_mark,
        .gp = gp,
        .coop = coop,
        .tc_attn = tc_attn,
        .flash = flash,
        .s_rows = s_rows,
        .heads_per_batch = heads_per_batch,
        .bufs = .{ .x = ws.x_d, .t1 = ws.t1_d, .attn = ws.attn_d, .mv = ws.mv_d, .fin = ws.fin_d },
    };
    try dit_dev.body(d, model, seq_txt, n_img, cancel);
    if (batched) try ctx.endBatch();
    // A 16-bit stream that overflowed can leave the output finite (`allFinite`).
    if (ws.act != .f32 and !try ctx.allFinite(ws.x_d, seq * F)) return error.NonFiniteActivation;

    const final_rows = try gpa.alloc(f32, n_img * dit.channels * dit.patch * dit.patch);
    defer gpa.free(final_rows);
    try ctx.tensorDownload(ws.imgin_d, std.mem.sliceAsBytes(final_rows));
    lapNs(io, &t_mark, &prof.xfer_ns);
    DiT.unpatchify(out, final_rows, lat_h, lat_w);
    lapNs(io, &t_mark, &prof.cpu_ns);

    if (profile) {
        const ms = struct {
            fn go(ns: i96) f64 {
                return @as(f64, @floatFromInt(ns)) / 1e6;
            }
        }.go;
        std.debug.print(
            "dit gpu profile: matmul {d:.0}ms  attn {d:.0}ms (scores {d:.0} smax {d:.0} out {d:.0})  elt {d:.0}ms  prep {d:.0}ms  xfer {d:.0}ms  cpu {d:.0}ms\n",
            .{ ms(prof.matmul_ns), ms(prof.attn_ns + prof.scores_ns + prof.smax_ns + prof.aout_ns), ms(prof.scores_ns), ms(prof.smax_ns), ms(prof.aout_ns), ms(prof.elt_ns), ms(prof.prep_ns), ms(prof.xfer_ns), ms(prof.cpu_ns) },
        );
    }
}

/// The profile's attention split, for `attn_gpu.core`.
const AttnTimer = struct {
    io: std.Io,
    t_mark: *std.Io.Timestamp,
    prof: *Prof,

    fn lap(self: *const AttnTimer) attn_gpu.Lap {
        return .{ .ctx = self, .f = f };
    }

    fn f(p: *const anyopaque, phase: attn_gpu.Phase) void {
        const self: *const AttnTimer = @ptrCast(@alignCast(p));
        lapNs(self.io, self.t_mark, switch (phase) {
            .scores => &self.prof.scores_ns,
            .softmax => &self.prof.smax_ns,
            .out => &self.prof.aout_ns,
        });
    }
};

/// `dit_dev`'s ops on this backend. Its fp8 fast path stages: the modulated norm (or the
/// gate, or silu) writes the cooperative GEMM's f16 operand with the weight scale folded
/// in, shared by every GEMM of the group, and q/k/v (and with them the gate and `wo`)
/// come out f16 where the attention reads f16.
const Dev = struct {
    model: *const DiT,
    ctx: *gpu.Context,
    sess: *const Session,
    ws: *const Workspace,
    io: std.Io,
    gpa: std.mem.Allocator,
    prof: *Prof,
    t_mark: *std.Io.Timestamp,
    /// The generic path's plan.
    gp: lin_gpu.Plan,
    coop: bool,
    tc_attn: bool,
    flash: bool,
    s_rows: usize,
    heads_per_batch: usize,
    bufs: Bufs,

    pub const can_stage = true;
    const Bufs = struct { x: gpu.DeviceBuffer, t1: gpu.DeviceBuffer, attn: gpu.DeviceBuffer, mv: gpu.DeviceBuffer, fin: gpu.DeviceBuffer };
    const Buf = gpu.DeviceBuffer;
    const St = dit_dev.Stage(Buf);

    fn lap(d: Dev, bucket: *i96) void {
        lapNs(d.io, d.t_mark, bucket);
    }

    fn pad(m: usize) usize {
        return std.mem.alignForward(usize, m, 128);
    }

    fn stage(d: Dev, scale: f32) St {
        return .{ .buf = d.ws.h16_d.as(.f16), .scale = scale };
    }

    pub fn patchEmbed(d: Dev, seq_txt: usize, n_img: usize) !void {
        try d.sess.lin.linearAt(d.sess.lin_plan, d.ws.x_d, seq_txt * F, d.ws.imgin_d, 0, n_img, d.model.first.w, d.model.first.b);
        d.lap(&d.prof.matmul_ns);
    }

    pub fn blockBegin(_: Dev, _: usize) !void {}

    /// The q/k/v/gate projections share one staged activation when all four and `wo`
    /// are cooperative fp8 at one scale (raw e4m3 checkpoints store scale 1 everywhere).
    /// q/k/v come out f16 for the fused norm/rope/gather chain when the tensor-core
    /// scores pipeline is there and the GEMMs can store f16: on that shared fp8 path
    /// (then the gate and `wo`'s output are f16 too), or when all four are int8.
    pub fn attnPlan(d: Dev, blk: *const dit.Block, seq: usize) !dit_dev.AttnPlan(Buf) {
        const ws = d.ws;
        const ctx = d.ctx;
        const att_ws = [_]Weight{ blk.attn.wq, blk.attn.wk, blk.attn.wv, blk.attn.gate };
        const is_i8 = allI8(d.sess, &att_ws, seq);
        const qkv_shared = d.coop and allRoute(d.sess, &(att_ws ++ [_]Weight{blk.attn.wo}), seq, .gemm_fp8) and
            blk.attn.wq.scale == blk.attn.wk.scale and
            blk.attn.wq.scale == blk.attn.wv.scale and
            blk.attn.wq.scale == blk.attn.gate.scale;
        const att16 = qkv_shared and d.tc_attn and ctx.pipe_coop_c16 != .null_handle;
        const i8_f16 = is_i8 and d.tc_attn and ctx.pipe_coop_i8_fs16 != .null_handle;
        const act16 = ws.act != .f32;
        const qkv16 = att16 or i8_f16 or (act16 and d.tc_attn);
        return .{
            .stage = if (qkv_shared) d.stage(blk.attn.wq.scale) else null,
            .q = if (qkv16) ws.q_d.as(.f16) else ws.q_d,
            .k = if (qkv16) ws.k_d.as(.f16) else ws.k_d,
            // V lands straight in the P@V operand.
            .v = if (qkv16) ws.v16_d.as(.f16) else ws.v_d,
            .g = if (att16) ws.g_d.as(.f16) else ws.g_d,
            .wo_stage = if (d.coop and allRoute(d.sess, &.{blk.attn.wo}, seq, .gemm_fp8)) d.stage(blk.attn.wo.scale) else null,
            .wo_out = if (act16) ws.t1_d else if (att16) ws.t1_d.as(.f16) else ws.t1_d,
        };
    }

    /// gate and up share one staged activation when both and `down` are cooperative fp8
    /// at one scale; they then come out f16 where the GEMM can store it, and so does
    /// `down`, for an f16 gated add. One pass over every row.
    pub fn mlpPlan(d: Dev, blk: *const dit.Block, seq: usize) !dit_dev.MlpPlan(Buf) {
        const ws = d.ws;
        const shared = d.coop and allRoute(d.sess, &.{ blk.mlp.gate, blk.mlp.up, blk.mlp.down }, seq, .gemm_fp8) and blk.mlp.gate.scale == blk.mlp.up.scale;
        const mlp16 = shared and d.ctx.pipe_coop_c16 != .null_handle;
        const down_stage = d.coop and allRoute(d.sess, &.{blk.mlp.down}, seq, .gemm_fp8);
        // The staged silu reads f16 or f32; the plain one any dtype, so a 16-bit stream
        // keeps gate/up at its own width there.
        const mdt: gpu.Dt = if (mlp16) .f16 else if (ws.act != .f32 and !down_stage) ws.act else .f32;
        return .{
            .tile = seq,
            .stage = if (shared) d.stage(blk.mlp.gate.scale) else null,
            .mg = ws.mg_d.as(mdt),
            .mu = ws.mu_d.as(mdt),
            .down_stage = if (down_stage) d.stage(blk.mlp.down.scale) else null,
            .down_out = if (ws.act != .f32) ws.t1_d else if (mlp16) ws.t1_d.as(.f16) else ws.t1_d,
        };
    }

    /// One subgroup per row: the norm and the modulation are a single launch. Slot
    /// `premul` carries the norm weight prefolded.
    pub fn rmsMod(d: Dev, src: Buf, dst: Buf, mod: Buf, premul: usize, shift: usize, rows: usize) !void {
        try d.ctx.opRmsModSg(src, dst, mod, rows, F, premul, shift, 1e-5);
        d.lap(&d.prof.elt_ns);
    }

    pub fn rmsMod16(d: Dev, src: Buf, st: St, mod: Buf, premul: usize, shift: usize, rows: usize) !void {
        try d.ctx.opRmsModSgH16(src, st.buf, mod, pad(rows), rows, F, premul, shift, 1e-5, st.scale);
        d.lap(&d.prof.elt_ns);
    }

    pub fn prep(d: Dev, x: Buf, m: usize, cols: usize, group: []const Weight) !lin_gpu.Group {
        const g = try d.sess.lin.prep(d.gp, x, m, cols, group);
        d.lap(&d.prof.prep_ns);
        return g;
    }

    pub fn prepStaged(d: Dev, st: St, m: usize, cols: usize, group: []const Weight) lin_gpu.Group {
        return d.sess.lin.prepStaged(d.gp, st.buf, m, cols, group, st.scale);
    }

    /// A staged group's GEMMs read one operand and write disjoint outputs, so they run
    /// with no barriers between them and the narrow grids overlap their neighbours'
    /// tails. (Not the generic path: a coop GEMM there is three dispatches sharing the
    /// Context's scratch.)
    pub fn gemms(d: Dev, g: lin_gpu.Group, ys: []const Buf, group: []const Weight) !void {
        if (g.staged != null and ys.len > 1) d.ctx.independent(ys.len);
        // The f32 stream's f16 outputs (`att16`, `mlp16`) are the f16-accumulating fast
        // path; under a 16-bit stream every output accumulates f32 (f16 accumulators cost
        // fp8 krea2 2.5e-2 velocity against the CPU at 512²).
        for (ys, group) |y, w| try d.sess.lin.gemm(d.gp, g, y, w, .{ .m_pad = pad(g.m), .acc16 = d.ws.act == .f32 });
        d.lap(&d.prof.matmul_ns);
    }

    pub fn linear(d: Dev, y: Buf, x: Buf, m: usize, w: Weight) !void {
        try linGroup(d.sess, d.gp, &.{y}, x, m, pad(m), &.{w});
        d.lap(&d.prof.matmul_ns);
    }

    /// q/k norm, rope and the conversion into the scores layout, then attention into
    /// `attn_d`. Three arms: f16 q/k/v from the GEMMs take one fused in-place pass each;
    /// int8's f32 outputs fuse norm and rope into one f32 pass; everything else norms,
    /// ropes and converts separately.
    pub fn attention(d: Dev, blk: *const dit.Block, ap: dit_dev.AttnPlan(Buf), seq: usize) !void {
        const ctx = d.ctx;
        const ws = d.ws;
        const io = d.io;
        const prof = d.prof;
        var t_mark = d.t_mark.*;
        defer d.t_mark.* = t_mark;
        const mark = lapNs;
        const q_d = ws.q_d;
        const k_d = ws.k_d;
        const v_d = ws.v_d;
        const v16_d = ws.v16_d;
        const qt_d = ws.qt_d;
        const kt_d = ws.kt_d;
        const attn_d = ws.attn_d;
        const s_d = ws.s_d;
        const part_d = ws.part_d;
        const md_d = ws.md_d;
        const freqs_d = d.sess.freqs_d;
        const seq_pad = pad(seq);
        const half = hd / 2;
        const sin_off: u32 = @intCast(seq * half);
        const tc_attn = d.tc_attn;
        const flash = d.flash;
        const md_off = seq_pad * F;
        const s_rows = d.s_rows;
        const heads_per_batch = d.heads_per_batch;
        const attn_f16 = ap.q.dt == .f16;
        const is_i8 = allI8(d.sess, &.{ blk.attn.wq, blk.attn.wk, blk.attn.wv, blk.attn.gate }, seq);
            if (attn_f16) {
                // Fused per-head norm + rope + scale, in place on the f16 GEMM
                // outputs (value-identical to the chain below, see the kernel);
                // K then gathers f16 -> f16 into the scores layout. V needs
                // nothing: its GEMM already wrote the P@V operand.
                ctx.independent(2);
                try ctx.opElt(.qknorm_rope_f16, q_d, try normBuf(ctx, blk.attn.qnorm), freqs_d, null, .{
                    .u0 = @intCast(seq * heads),
                    .u1 = half,
                    .u2 = sin_off,
                    .u3 = heads,
                    .f0 = attn_scale,
                    .f1 = 1e-5,
                }, seq * heads, 1, 1);
                mark(io, &t_mark, &prof.elt_ns);
                try ctx.opElt(.qknorm_rope_f16, k_d, try normBuf(ctx, blk.attn.knorm), freqs_d, null, .{
                    .u0 = @intCast(seq * kv_heads),
                    .u1 = half,
                    .u2 = sin_off,
                    .u3 = kv_heads,
                    .f0 = 1.0,
                    .f1 = 1e-5,
                }, seq * kv_heads, 1, 1);
                mark(io, &t_mark, &prof.elt_ns);
                try ctx.opElt(.gather_kmajor16, k_d, null, null, kt_d, .{
                    .u0 = @intCast(kv_heads * hd * seq_pad / 2),
                    .u1 = hd,
                    .u2 = @intCast(seq_pad),
                    .u3 = @intCast(seq),
                    .u4 = kv_heads,
                }, kv_heads * hd * seq_pad / 2, 1, 1);
                mark(io, &t_mark, &prof.elt_ns);
            } else if (is_i8 and tc_attn) {
                // int8 f32 GEMM outputs, tensor-core attention: fuse rmsnorm+rope
                // into ONE f32 in-place pass for q and k (2 passes each -> 1), then
                // the existing converts (which zero the seq_pad tail correctly).
                // 7 f32 passes -> 5. q's attn_scale is applied at the f32_to_h16.
                ctx.independent(2);
                try ctx.opElt(.qknorm_rope, q_d, try normBuf(ctx, blk.attn.qnorm), freqs_d, null, .{
                    .u0 = @intCast(seq * heads),
                    .u1 = half,
                    .u2 = sin_off,
                    .u3 = heads,
                    .f0 = 1.0,
                    .f1 = 1e-5,
                }, seq * heads, 1, 1);
                try ctx.opElt(.qknorm_rope, k_d, try normBuf(ctx, blk.attn.knorm), freqs_d, null, .{
                    .u0 = @intCast(seq * kv_heads),
                    .u1 = half,
                    .u2 = sin_off,
                    .u3 = kv_heads,
                    .f0 = 1.0,
                    .f1 = 1e-5,
                }, seq * kv_heads, 1, 1);
                mark(io, &t_mark, &prof.elt_ns);
                try attn_gpu.stage(ctx, q_d, k_d, v_d, qt_d, kt_d, v16_d, seq, heads, kv_heads, hd, attn_scale);
                mark(io, &t_mark, &prof.elt_ns);
            } else {
            // The q and k chains (norm -> rope -> convert/gather) touch disjoint
            // buffers stage by stage; each stage pair/triple runs barrier-free.
            // The SUBGROUP norm, because `Elt.rmsnorm` gives each THREAD a whole 128-wide
            // head, a warp's 32 loads land 512 B apart and each is its own sector. Measured
            // at krea2's geometry at 1120x1680 (`vk-norm-bench`, 352800 x 128): 35 GB/s
            // against 558, 16x, worth ~600 ms/step across the two calls x 28 blocks.
            //
            // This `else` arm is reached by every group neither fused path above takes
            // (dense bf16, fp8 off the shared path, NVFP4, a block quant); an int8 one takes
            // `qknorm_rope` and never gets here.
            //
            // Not bit-identical: the row sum becomes a subgroup tree where it was serial (the
            // more accurate of the two). `independent(2)` still applies, one dispatch either way.
            ctx.independent(2);
            try rmsNormQk(ctx, q_d, blk.attn.qnorm, seq * heads, hd);
            mark(io, &t_mark, &prof.elt_ns);
            try rmsNormQk(ctx, k_d, blk.attn.knorm, seq * kv_heads, hd);
            mark(io, &t_mark, &prof.elt_ns);
            ctx.independent(2);
            try ctx.opElt(.rope_inter, q_d, null, freqs_d, null, .{
                .u0 = @intCast(seq * heads * half),
                .u1 = half,
                .u2 = sin_off,
                .u3 = heads,
            }, seq * heads * half, 1, 1);
            mark(io, &t_mark, &prof.elt_ns);
            try ctx.opElt(.rope_inter, k_d, null, freqs_d, null, .{
                .u0 = @intCast(seq * kv_heads * half),
                .u1 = half,
                .u2 = sin_off,
                .u3 = kv_heads,
            }, seq * kv_heads * half, 1, 1);
            mark(io, &t_mark, &prof.elt_ns);
            if (tc_attn) {
                try attn_gpu.stage(ctx, q_d, k_d, v_d, qt_d, kt_d, v16_d, seq, heads, kv_heads, hd, attn_scale);
            } else {
                ctx.independent(2);
                try ctx.opElt(.gather_kmajor, q_d, null, null, qt_d, .{
                    .u0 = @intCast(seq * heads * hd),
                    .u1 = heads,
                    .u2 = hd,
                    .u3 = @intCast(seq),
                }, seq * heads * hd, 1, 1);
                try ctx.opElt(.gather_kmajor, k_d, null, null, kt_d, .{
                    .u0 = @intCast(seq * kv_heads * hd),
                    .u1 = kv_heads,
                    .u2 = hd,
                    .u3 = @intCast(seq),
                }, seq * kv_heads * hd, 1, 1);
            }
            }
            // On the f16-C path Q was normed/roped/scaled in place: it IS the
            // scores operand.
            const q_src = if (attn_f16) q_d else qt_d;
            if (flash) {
                const push = gpu.EltPush{
                    .u0 = heads * hd,
                    .u1 = @intCast(seq_pad),
                    .u2 = 0,
                    .u3 = heads / kv_heads,
                    .u4 = kv_heads * hd,
                    .u5 = @intCast(md_off),
                    .f0 = @bitCast(@as(u32, @intCast(seq))),
                };
                try ctx.opFlash(.md, q_src, kt_d, v16_d, attn_d, push, seq_pad / 128, heads);
                mark(io, &t_mark, &prof.scores_ns);
                mark(io, &t_mark, &prof.smax_ns);
                try ctx.opFlash(.out, q_src, kt_d, v16_d, attn_d, push, seq_pad / 128, heads);
                mark(io, &t_mark, &prof.aout_ns);
            } else if (tc_attn) {
                const timer: AttnTimer = .{ .io = io, .t_mark = &t_mark, .prof = prof };
                try attn_gpu.core(ctx, .{ .s = s_d, .part = part_d, .md = md_d }, q_src, kt_d, v16_d, attn_d, .{
                    .q_rows = seq,
                    .kv_rows = seq,
                    .heads = heads,
                    .kv_heads = kv_heads,
                    .hd = hd,
                    .max_hb = heads_per_batch,
                    .nchunks = nchunks,
                }, timer.lap());
            } else {
                const s_stride: u32 = @intCast(s_rows);
                const s_plane: u32 = @intCast(s_rows * s_rows);
                var h0: usize = 0;
                while (h0 < heads) : (h0 += heads_per_batch) {
                    const hb = @min(heads_per_batch, heads - h0);
                    try ctx.opElt(.attn_scores, qt_d, kt_d, null, s_d, .{
                        .u0 = @intCast(seq),
                        .u1 = heads,
                        .u2 = kv_heads,
                        .u3 = hd,
                        .u4 = @intCast(h0),
                        .f0 = attn_scale,
                    }, std.math.divCeil(usize, seq, 8) catch unreachable, std.math.divCeil(usize, seq, 8) catch unreachable, hb);
                    mark(io, &t_mark, &prof.scores_ns);
                    try ctx.opElt(.attn_out, s_d, null, v_d, attn_d, .{
                        .u0 = @intCast(seq),
                        .u1 = heads,
                        .u2 = kv_heads,
                        .u3 = hd,
                        .u4 = @intCast(h0),
                        .u5 = s_stride,
                        .f0 = @bitCast(s_plane),
                    }, hd / 8, std.math.divCeil(usize, seq, 8) catch unreachable, hb);
                    mark(io, &t_mark, &prof.aout_ns);
                }
            }
            mark(io, &t_mark, &prof.attn_ns);
    }

    /// `attn * sigmoid(g)` as the next fp8 GEMM's staged operand.
    pub fn gateStage(d: Dev, attn: Buf, g: Buf, st: St, seq: usize) !void {
        const push: gpu.EltPush = .{ .u0 = @intCast(pad(seq) * F / 2), .u1 = @intCast(seq * F), .f0 = st.scale };
        try d.ctx.opElt(if (g.dt == .f16) .sigmoid_mul_g16 else .sigmoid_mul_h16, attn, g, null, st.buf, push, pad(seq) * F / 2, 1, 1);
        d.lap(&d.prof.elt_ns);
    }

    pub fn gateMul(d: Dev, attn: Buf, g: Buf, seq: usize) !void {
        std.debug.assert(attn.dt == g.dt);
        try d.ctx.opElt(gpu.eltAt("sigmoid_mul", attn.dt), attn, g, null, null, .{ .u0 = @intCast(seq * F) }, seq * F, 1, 1);
        d.lap(&d.prof.elt_ns);
    }

    pub fn gatedAdd(d: Dev, x: Buf, t: Buf, mod: Buf, off: usize, rows: usize) !void {
        if (x.dt != .f32) {
            std.debug.assert(t.dt == x.dt);
            try d.ctx.opElt(gpu.eltAt("gated_add", x.dt), x, t, mod, null, .{ .u0 = @intCast(rows * F), .u1 = F, .u2 = @intCast(off) }, rows * F, 1, 1);
        } else if (t.dt == .f16) {
            try d.ctx.opElt(.gated_add16, x, t, mod, null, .{ .u0 = @intCast(rows * F / 2), .u1 = F, .u2 = @intCast(off) }, rows * F / 2, 1, 1);
        } else {
            try d.ctx.opElt(.gated_add, x, t, mod, null, .{ .u0 = @intCast(rows * F), .u1 = F, .u2 = @intCast(off) }, rows * F, 1, 1);
        }
        d.lap(&d.prof.elt_ns);
    }

    pub fn rowsAt(_: Dev, x: Buf, r0: usize, _: usize) !Buf {
        std.debug.assert(r0 == 0); // one MLP pass over every row
        return x;
    }

    /// `silu(gate) * up` as the down GEMM's staged operand.
    pub fn siluStage(d: Dev, mg: Buf, mu: Buf, st: St, tile: usize) !void {
        const n = pad(tile) * dit.mlp_dim / 2;
        if (mg.dt == .f16) {
            try d.ctx.opElt(.silu_mul16, mg, mu, null, st.buf, .{ .u0 = @intCast(n), .f0 = st.scale }, n, 1, 1);
        } else {
            try d.ctx.opElt(.silu_mul_h16, mg, mu, null, st.buf, .{ .u0 = @intCast(n), .u1 = @intCast(tile * dit.mlp_dim), .f0 = st.scale }, n, 1, 1);
        }
        d.lap(&d.prof.elt_ns);
    }

    pub fn siluMul(d: Dev, mg: Buf, mu: Buf, tile: usize) !void {
        std.debug.assert(mg.dt == mu.dt);
        try d.ctx.opElt(gpu.eltAt("silu_mul", mg.dt), mg, mu, null, null, .{ .u0 = @intCast(tile * dit.mlp_dim) }, tile * dit.mlp_dim, 1, 1);
        d.lap(&d.prof.elt_ns);
    }

    /// Reads the residual back and reuses the CPU accumulator (see `dit_cuda`).
    pub fn steerCapture(d: Dev, st: dit.ActSteer, seq: usize, seq_txt: usize, b: usize) !void {
        const n = seq * F;
        const host = try d.gpa.alloc(f32, n);
        defer d.gpa.free(host);
        try d.ctx.downloadF32At(d.ws.x_d, 0, host);
        st.run(host, seq, seq_txt, b);
    }

    pub fn steer(d: Dev, st: dit.ActSteer, seq: usize, seq_txt: usize, b: usize, sc: f32) !void {
        if (d.sess.dirs_d.size == 0) return;
        try d.ctx.opActSteer(d.ws.x_d, d.sess.dirs_d, seq - seq_txt, F, b * F, seq_txt, st.op == .add, st.keep_norm, sc);
    }

    /// The image rows only, read at their offset (which is part of the measurement).
    pub fn finalLinear(d: Dev, seq_txt: usize, n_img: usize) !void {
        const w = d.model.last_linear;
        try probeInputAt(d.ctx, d.ws.t1_d, seq_txt * F * 4, n_img, w.w);
        try d.sess.lin.linearAt(d.sess.lin_plan, d.ws.imgin_d, 0, d.ws.t1_d, seq_txt * F, n_img, w.w, w.b);
        d.lap(&d.prof.matmul_ns);
    }
};

/// Feed `ops.matmul.probe` a GEMM input that lives on the device, for an activation
/// capture running on Vulkan. See `dit_cuda.probeInput` for why this is a download
/// into the CPU accumulator rather than an on-device reduction.
///
/// int8/int4 weights are skipped. Their activation has already been rotated
/// and quantized in place, so the buffer holds a W4A4-shaped value and recording it
/// as a weight-only statistic would be a category error (ACTIVATION_AWARE hygiene
/// rule 4). `ggufy calibrate` refuses those checkpoints on a GPU backend outright.
fn probeInput(ctx: *gpu.Context, x: gpu.DeviceBuffer, m: usize, w: anytype) !void {
    return probeInputAt(ctx, x, 0, m, w);
}

fn probeInputAt(ctx: *gpu.Context, x: gpu.DeviceBuffer, off_bytes: u64, m: usize, w: anytype) !void {
    const p = ops.matmul.probe orelse return;
    if (w.dtype == .i8 or w.dtype == .i4) return;
    if (m == 0 or w.cols == 0) return;
    const host = try ctx.gpa.alloc(f32, m * w.cols);
    defer ctx.gpa.free(host);
    try ctx.tensorDownloadAt(x, off_bytes, std.mem.sliceAsBytes(host));
    p.input(p.ctx, w, host, m);
}

/// Per-head weighted RMSNorm in place over `[rows][hd]`, preferring the subgroup kernel.
/// See the call site for the measurement and for which checkpoints reach it.
fn rmsNormQk(ctx: *gpu.Context, x: gpu.DeviceBuffer, weights: []const f32, rows: usize, width: usize) !void {
    const w = try normBuf(ctx, weights);
    if (ctx.hasSubgroupNorm()) return ctx.opRmsNormSg(x, x, w, rows, width, 1e-5);
    try ctx.opElt(.rmsnorm, x, x, w, null, .{
        .u0 = @intCast(rows),
        .u1 = @intCast(width),
        .f0 = 1e-5,
    }, rows, 1, 1);
}

fn normBuf(ctx: *gpu.Context, weights: []const f32) !gpu.DeviceBuffer {
    // smallBuffer caches by pointer; wrap the raw handle for opElt.
    const buf = try ctx.smallBuffer(std.mem.sliceAsBytes(weights));
    return .{ .buf = buf, .mem = .null_handle, .size = 0 };
}

// Parity against the same ComfyUI fixture as the CPU forward test. Needs the
// model, testdata fixtures, and BOTH markers (slow + gpu).
test "gpu-resident forward matches comfyui fixture" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const safetensors = @import("tp_core").safetensors;
    std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    const dit_path = "models/diffusion_model/krea2CenterSemiraw_v10Fp8.safetensors";
    std.Io.Dir.cwd().access(io, dit_path, .{}) catch return error.SkipZigTest;
    std.Io.Dir.cwd().access(io, "testdata/dit_out.bin", .{}) catch return error.SkipZigTest;

    var ctx = gpu.Context.init(gpa, std.testing.io) catch return error.SkipZigTest;
    defer ctx.deinit();

    const readF32 = struct {
        fn go(alloc: std.mem.Allocator, io_: std.Io, path: []const u8, n: usize) ![]f32 {
            const out = try alloc.alloc(f32, n);
            errdefer alloc.free(out);
            const file = try std.Io.Dir.cwd().openFile(io_, path, .{ .mode = .read_only });
            defer file.close(io_);
            const bytes = std.mem.sliceAsBytes(out);
            if (try file.readPositionalAll(io_, bytes, 0) != bytes.len) return error.ShortRead;
            return out;
        }
    }.go;

    const seq_txt = 14;
    const x_lat = try readF32(gpa, io, "testdata/dit_x.bin", dit.channels * 16 * 16);
    defer gpa.free(x_lat);
    const expected = try readF32(gpa, io, "testdata/dit_out.bin", dit.channels * 16 * 16);
    defer gpa.free(expected);
    const cond = try readF32(gpa, io, "testdata/text_cond.bin", seq_txt * dit.txt_layers * dit.txt_dim);
    defer gpa.free(cond);

    var st = try safetensors.SafeTensors.open(gpa, io, dit_path);
    defer st.deinit();
    var model = try DiT.load(gpa, .{ .safetensors = &st });
    defer model.deinit();

    const out = try gpa.alloc(f32, dit.channels * 16 * 16);
    defer gpa.free(out);
    var sess = try Session.init(gpa, io, ctx, &model, 16, 16, cond, seq_txt, &.{0.875});
    defer sess.deinit(gpa, ctx);
    var ws = try Workspace.init(ctx, 16, 16, seq_txt, .f32);
    defer ws.deinit(ctx);
    try forward(&model, ctx, &sess, &ws, io, gpa, out, x_lat, 0.875, null);

    var max_err: f32 = 0;
    var max_val: f32 = 0;
    for (expected, out) |e, a| {
        // NaN-robust: @max drops NaN operands, which would hide a poisoned
        // output entirely.
        max_err = if (std.math.isNan(a)) std.math.inf(f32) else @max(max_err, @abs(e - a));
        max_val = @max(max_val, @abs(e));
    }
    std.debug.print("dit gpu parity: max_err={d:.5} max_val={d:.2}\n", .{ max_err, max_val });
    // 5%: attention Q/K run in f16 on the tensor-core path (same regime as
    // ComfyUI's fp16 SDPA, so the fixture carries equivalent rounding noise
    // of its own); measured max_err is ~3.8% over 28 blocks, and same-seed
    // images against the f32-attention path are visually identical (mean
    // pixel delta ~0.9%). The scores kernel itself is verified to 5e-3
    // against an f16-rounded reference in gpu/context.zig.
    try std.testing.expect(max_err < 0.05 * @max(1.0, max_val));

    // Full-f32 path (tensor-core pipelines forced off, Workspace re-sized to
    // match): reports the f16-vs-f32 accuracy gap against the same fixture.
    {
        force_f32 = true;
        defer force_f32 = false;
        var ws32 = try Workspace.init(ctx, 16, 16, seq_txt, .f32);
        defer ws32.deinit(ctx);
        const out32 = try gpa.alloc(f32, dit.channels * 16 * 16);
        defer gpa.free(out32);
        try forward(&model, ctx, &sess, &ws32, io, gpa, out32, x_lat, 0.875, null);
        var me: f32 = 0;
        for (expected, out32) |e, a| me = if (std.math.isNan(a)) std.math.inf(f32) else @max(me, @abs(e - a));
        std.debug.print("dit gpu parity (f32): max_err={d:.5} (f16 was {d:.5})\n", .{ me, max_err });
        try std.testing.expect(me < 0.05 * @max(1.0, max_val));
    }

    // Weight streaming under memory pressure: force a budget far below the
    // model size so the LRU weight cache evicts and re-uploads every block,
    // then check the forward is BIT-IDENTICAL to the resident run (same
    // weights, same kernels, only residency changed).
    ctx.evictWeights();
    ctx.budget_override = 3 << 30;
    defer ctx.budget_override = 0;
    const out2 = try gpa.alloc(f32, dit.channels * 16 * 16);
    defer gpa.free(out2);
    try forward(&model, ctx, &sess, &ws, io, gpa, out2, x_lat, 0.875, null);
    for (out, out2) |a, b| try std.testing.expectEqual(a, b);

    // A pre-set cancel flag aborts mid-step (polled between blocks); the
    // errdefer in forward aborts the in-flight batch so the context stays
    // usable afterwards.
    var canceled = std.atomic.Value(bool).init(true);
    try std.testing.expectError(error.Canceled, forward(&model, ctx, &sess, &ws, io, gpa, out2, x_lat, 0.875, &canceled));
}

// The W4A8 decode kernel against the CPU decode it must reproduce. Needs a device
// (testdata/gpu-tests) but NO checkpoint, synthetic weights are enough, and that is
// deliberate: the kernel's correctness must be checkable without a multi-GB file on
// disk, which is exactly the situation that leaves a gated test silently self-skipping.
//
// This is also the test that would have caught the layout half. The first version of
// `w4a8_decode_t` read the row-major storage and transposed as it went (correct, and
// 1757 ms/step); the current one reads inputs that `weightBuffer` has already
// byte-transposed. Those two kernels agree on nothing about their input indexing, so a
// render is the only other thing that can tell them apart.
test "the Vulkan W4A8 decode matches ops.w4a8.decode, including the row padding" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const w4a8 = ops.w4a8;
    std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    var ctx = gpu.Context.init(gpa, std.testing.io) catch return error.SkipZigTest;
    defer ctx.deinit();
    if (!ctx.hasW4A8Decode()) return error.SkipZigTest;

    // Two heights: one that needs no row padding and one that does, since the padded
    // slots are read by the GEMM and must decode to zero.
    const cases = [_]struct { rows: usize, cols: usize, gs: usize }{
        .{ .rows = 256, .cols = 512, .gs = 16 },
        .{ .rows = 1536, .cols = 256, .gs = 16 },
        .{ .rows = 192, .cols = 512, .gs = 32 },
        // Group sizes 8 and 4 are legal in the format and this kernel is general over
        // them, unlike the CUDA one, which needs one scale per 8 columns because it
        // decodes four packed bytes per thread. Pinning them here is what makes that a
        // documented backend difference rather than an accident.
        .{ .rows = 256, .cols = 256, .gs = 8 },
        .{ .rows = 128, .cols = 256, .gs = 4 },
    };
    // Every case's buffers stay ALIVE in one arena, and that is load-bearing rather
    // than tidy: the device weight cache keys on the HOST POINTER, so freeing a case's
    // arrays and letting the allocator hand the same address to the next case scores a
    // stale cache hit and the kernel reads the previous case's weights. (A model never
    // hits it: the weights live in the model arena for the model's lifetime.)
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const alloc = arena.allocator();
    for (cases) |c| {
        const stride = gpu.Context.w4a8ScratchBytes(c.rows, c.cols) / c.cols;
        const groups = c.cols / c.gs;

        var rnd = std.Random.DefaultPrng.init(0xA8 + c.rows);
        const r = rnd.random();
        const packed_bytes = try alloc.alloc(u8, c.rows * c.cols / 2);
        r.bytes(packed_bytes);
        const s_rel = try alloc.alloc(u8, c.rows * groups);
        // Every fp8 byte, NaN (0x7F/0xFF) included: the level table maps those to 0 on
        // both sides, so this pins that the two agree there rather than diverging on a
        // value neither reference defines.
        r.bytes(s_rel);
        const levels = try alloc.create(w4a8.Levels);
        levels.* = w4a8.Levels.init(&w4a8.fixed_lut);

        // Expected: the CPU decode, laid out k-major with zeroed row padding.
        const row_major = try alloc.alloc(i8, c.rows * c.cols);
        w4a8.decode(row_major, packed_bytes, s_rel, .{ .w4 = levels }, c.rows, c.cols, c.gs);
        const want = try alloc.alloc(i8, stride * c.cols);
        @memset(want, 0);
        for (0..c.cols) |k| {
            for (0..c.rows) |row| want[k * stride + row] = row_major[row * c.cols + k];
        }

        const buf = try ctx.w4a8Decode(packed_bytes, s_rel, std.mem.asBytes(levels), c.rows, c.cols, c.gs);
        _ = buf;
        const got = try alloc.alloc(i8, stride * c.cols);
        try ctx.tensorDownload(ctx.w4a8_t, std.mem.sliceAsBytes(got));

        for (want, got, 0..) |wv, gv, i| {
            std.testing.expectEqual(wv, gv) catch |e| {
                std.debug.print("rows {d} cols {d} gs {d} stride {d}: element {d} (k {d}, row {d}) got {d}, want {d}\n", .{
                    c.rows, c.cols, c.gs, stride, i, i / stride, i % stride, gv, wv,
                });
                return e;
            };
        }
    }
}

// The Vulkan NVFP4 decode kernel against the CPU decode it must reproduce. Synthetic
// weights, so no checkpoint is needed, the durable form of the check.
//
// Covers the two things a render cannot localize and the CUDA twin does not share: the
// f16 `[k_pad][n_pad]` output layout (a different padding rule from the inputs' own
// stride) and the fact that BOTH paddings must be written, since this scratch is reused
// between weights of different shapes.
test "the Vulkan NVFP4 decode matches ops.nvfp4.decode, including both paddings" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const nvfp4 = ops.nvfp4;
    std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    var ctx = gpu.Context.init(gpa, std.testing.io) catch return error.SkipZigTest;
    defer ctx.deinit();
    if (!ctx.hasNvfp4Decode()) return error.SkipZigTest;

    const cases = [_]struct { rows: usize, cols: usize }{
        .{ .rows = 128, .cols = 256 }, // no padding on either axis
        .{ .rows = 40, .cols = 192 }, // rows pad to 128, cols to 192 (already 64-aligned)
        .{ .rows = 256, .cols = 64 }, // several row blocks, one k block
    };
    // One arena for every case: the weight cache keys on the HOST POINTER, so a freed and
    // re-allocated address would score a stale hit and decode the previous case's bytes.
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const alloc = arena.allocator();

    for (cases) |c| {
        var rnd = std.Random.DefaultPrng.init(0x4F4 + c.rows);
        const r = rnd.random();
        const packed_bytes = try alloc.alloc(u8, c.rows * c.cols / 2);
        r.bytes(packed_bytes);
        const scales = try alloc.alloc(u8, c.rows * c.cols / nvfp4.block_size);
        r.bytes(scales);
        const levels = try alloc.create(nvfp4.Levels);
        levels.* = nvfp4.Levels.init(0.0125);
        const meta: nvfp4.Meta = .{ .scales = scales, .levels = levels };

        const plain = try alloc.alloc(f32, c.rows * c.cols);
        nvfp4.decode(plain, packed_bytes, meta, c.rows, c.cols);

        const n_pad = std.mem.alignForward(usize, c.rows, 128);
        const k_pad = std.mem.alignForward(usize, c.cols, 64);
        const want = try alloc.alloc(f32, k_pad * n_pad);
        @memset(want, 0); // both paddings must come out zero
        for (0..c.cols) |k| {
            for (0..c.rows) |row| want[k * n_pad + row] = plain[row * c.cols + k];
        }

        _ = try ctx.nvfp4Decode(packed_bytes, scales, std.mem.asBytes(&levels.bf16v), c.rows, c.cols);
        const got_bits = try alloc.alloc(u16, k_pad * n_pad);
        try ctx.tensorDownload(ctx.nvfp4_w16, std.mem.sliceAsBytes(got_bits));

        for (want, got_bits, 0..) |wv, gb, i| {
            const gv: f32 = @bitCast(@as(u32, gb) << 16); // bf16 -> f32
            // A NaN block-scale byte propagates on both sides (see ops/nvfp4.zig); bf16
            // rounds the value, 8 mantissa bits = 2^-8 relative, and nothing else may
            // change it. A wrong nibble or block index moves by a whole E2M1 level.
            const both_nan = std.math.isNan(wv) and std.math.isNan(gv);
            const tol = @max(@abs(wv) * 5e-3, 1e-30);
            std.testing.expect(both_nan or @abs(gv - wv) <= tol) catch |e| {
                std.debug.print("{d}x{d} (n_pad {d}, k_pad {d}): word {d} (k {d}, row {d}) got {d}, want {d}\n", .{
                    c.rows, c.cols, n_pad, k_pad, i, i / n_pad, i % n_pad, gv, wv,
                });
                return e;
            };
        }
    }
}
