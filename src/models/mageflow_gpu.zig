//! GPU-resident Mage-Flow forward on Vulkan, the device twin of
//! `mageflow.DiT.forward` and the Vulkan sibling of `mageflow_cuda`.
//!
//! The host/device split and every convention are that arm's; read its header
//! first. Three things differ here, all forced by the op surface rather than by
//! the model:
//!
//! 1. **The block GEMMs fold their bias.** The coop GEMMs take a bias vector, so
//!    the `opAddBiasRows` pass `mageflow_cuda` runs after every GEMM is not
//!    needed: 12 extra full-plane passes a block, which at 1024x1024 is ~10 GB
//!    of traffic a step that this arm simply does not do.
//! 2. **Attention is the tensor-core scores/PV pipeline**, batching heads to fit
//!    a scores budget, with `attn_full` behind `force_attn_full` as the
//!    reference. That is a REQUIREMENT and not an optimization, for `zimage_gpu`'s
//!    reason: `attn_full` is one thread per (query, head) looping every key and
//!    trips the GPU watchdog at production resolutions, which surfaces as
//!    `error_device_lost` rather than as a slow render.
//! 3. **Each stream lives at offset 0 of its OWN buffer**, where the CUDA arm
//!    keeps one joint residual and takes pointer views into it. Vulkan buffers
//!    are opaque HANDLES, not pointers: `DeviceBuffer` carries no offset and the
//!    norm / GEMM / rope kernels all index from the binding's base, so a "row
//!    range of a bigger buffer" is not expressible. Only the attention needs the
//!    two together, so q/k/v are COPIED into a joint buffer for it and the result
//!    is copied back out. At 1024x1024 that is 8 copies a block, ~4.8 GB a step,
//!    which is under 1% of a step BY CALCULATION and not by isolation; it is what
//!    keeps every other kernel reading from row 0.
//!
//!    A consequence worth knowing: the text half is never rope'd at all. Its
//!    positions are zero, whose rotation is the identity, so the image half gets
//!    its own frequency table and the text half simply skips the kernel.
//!
//! ⚠️ `v_div` is carried over from the CUDA arm deliberately. V is the one
//! attention operand with no norm in front of it and outgrows f16 on a REAL
//! conditioning (509538 at 1024x1024 on the released checkpoint) while a random
//! one stays inside it, so a synthetic device check reads green without this and
//! the render comes back solid white.

const std = @import("std");
const mageflow = @import("mageflow.zig");
const mageflow_dev = @import("mageflow_dev.zig");
const lin = @import("lin.zig");
const lin_gpu = @import("lin_gpu.zig");
const gpu = @import("tp_gpu").context;
const ops = @import("tp_ops");

const DiT = mageflow.DiT;
const File = @This();
const Buf = gpu.DeviceBuffer;
const Weight = ops.matmul.Weight;

const F = mageflow.features; // 3072
const heads = mageflow.n_heads; // 24
const hd = mageflow.head_dim; // 128
const half = hd / 2;
const mlp_dim = mageflow.mlp_dim; // 12288
const channels = mageflow.channels; // 128
const eps: f32 = 1e-6;
const attn_scale: f32 = 1.0 / 11.313708498984761; // 1/sqrt(128)

/// Two-pass softmax chunk count, matching `zimage_gpu` and `dit_gpu`.
const nchunks = 32;
/// Cap on the materialized attention-scores buffer; heads batch to fit it.
///
/// A MEMORY bound, not a speed one: at 1024x1024 the whole 24-head plane is
/// 856 MB and grows as seq², and this holds it near 256 MB. Sweeping the head
/// count is worth ~3% at 1024x1024 and about as much the other way at 768x768,
/// i.e. inside this card's own drift, so no batch size here is claimed to be
/// faster than another.
const s_bytes_cap: usize = 256 << 20;

/// Force `attn_full` even where the tensor-core pipeline exists. For A/B and for
/// reproducing a mismatch; the device test runs both.
pub var force_attn_full: bool = false;

/// Run integer weights weight-only, so the device check can split a W8A8 forward's
/// activation noise from everything else.
pub var force_weight_only: bool = false;

/// Record the forward as ONE submission (the default). Off, every op is its own
/// submit-and-wait, which is what this arm did before batching and what the A/B in
/// `mageflow-bench` measures against.
pub var force_unbatched: bool = false;

/// Per-category host timing. Turning it on makes the forward SUBMIT-PER-OP so each
/// mark is exact, which is a different machine from the batched one it measures:
/// the roll-up is an upper bound on the step and the per-category RATIOS are the
/// part to read.
pub var profile: bool = false;
pub var prof: Prof = .{};

pub const Prof = struct {
    /// Split because the text half is ~1.5% of the FLOPs but a THIRD of the GEMM
    /// launches: at 64 rows a tensor-core GEMM is tail- and launch-bound, so the
    /// two are different questions.
    gemm_ns: i128 = 0,
    gemm_txt_ns: i128 = 0,
    attn_ns: i128 = 0,
    norm_ns: i128 = 0,
    elt_ns: i128 = 0,
    copy_ns: i128 = 0,
    xfer_ns: i128 = 0,
    cpu_ns: i128 = 0,

    pub fn reset(self: *Prof) void {
        self.* = .{};
    }

    pub fn totalNs(self: *const Prof) i128 {
        return self.gemm_ns + self.gemm_txt_ns + self.attn_ns + self.norm_ns + self.elt_ns + self.copy_ns + self.xfer_ns + self.cpu_ns;
    }
};

/// Accumulate the time since the last lap into `acc` and restamp. Only meaningful
/// when the forward is NOT batching, where each op is submit-and-wait; inside a
/// recording batch it would time recording rather than execution. The stamp is
/// file-scope because the profiled path is serialized anyway and threading it
/// through every block helper would be noise in the fast path.
var p_last: std.Io.Timestamp = .{ .nanoseconds = 0 };

/// Magnitude and non-finite count of a device buffer's first floats, under
/// TP_MAGEFLOW_VK_TRACE. A forward that comes back uncorrelated says nothing
/// about where it turned; the stage boundaries are where to look.
fn trace(ctx: *gpu.Context, what: []const u8, b: Buf, n: usize) void {
    if (std.c.getenv("TP_MAGEFLOW_VK_TRACE") == null) return;
    const cap = 1 << 21;
    var buf: [cap]f32 = undefined;
    const take = @min(n, buf.len);
    ctx.downloadF32At(b, 0, buf[0..take]) catch return;
    var bad: usize = 0;
    var peak: f32 = 0;
    var sum: f64 = 0;
    for (buf[0..take]) |v| {
        if (!std.math.isFinite(v)) bad += 1 else {
            peak = @max(peak, @abs(v));
            sum += @abs(v);
        }
    }
    std.debug.print("mageflow_gpu {s:<10} nonfinite {d}/{d}  peak {d:.5}  mean|x| {d:.5}\n", .{
        what, bad, take, peak, sum / @as(f64, @floatFromInt(take)),
    });
}

fn lap(io: std.Io, acc: *i128) void {
    if (!profile) return;
    const now = std.Io.Clock.real.now(io);
    acc.* += now.nanoseconds - p_last.nanoseconds;
    p_last = now;
}

/// Heads per scores batch, 0 = as many as `s_bytes_cap` allows. A/B knob, and the
/// answer it gave is that there is no answer: 4-6 heads beat 24 by ~3% at
/// 1024x1024 and lose by about as much at 768x768.
pub var heads_per_batch: usize = 0;

fn headsPerBatch(rows_pad: usize, cap: usize, ws_s_bytes: ?usize) usize {
    const plane = rows_pad * rows_pad * 2; // f16 scores
    var hb = @max(1, @min(heads, cap / @max(plane, 1)));
    if (heads_per_batch != 0) hb = @max(1, @min(hb, heads_per_batch));
    if (ws_s_bytes) |c| hb = @max(1, @min(hb, c / @max(plane, 1)));
    return hb;
}

fn scoresCap(budget: u64) usize {
    if (budget == 0) return s_bytes_cap;
    return @min(s_bytes_cap, @max(64 << 20, budget / 4));
}

fn useTensorCoreAttn(ctx: *gpu.Context) bool {
    return !force_attn_full and ctx.pipe_scores != .null_handle;
}

/// Per-image cache: everything constant across sampling steps.
pub const Session = struct {
    lat_h: usize,
    lat_w: usize,
    seq_txt: usize,
    /// Canvas tokens, i.e. what the output is truncated back to.
    n_img: usize,
    /// Canvas plus every reference image's tokens.
    n_tok: usize,
    seq: usize,

    /// `[seq_txt][F]`, the conditioning already through `txt_norm` and `txt_in`.
    txt_d: Buf,
    /// `[n_tok][channels]`; the reference half never changes, the canvas half is
    /// rewritten per step.
    tokens: []f32,
    /// The IMAGE half's rope table alone, `cos` then `sin`. The text half sits at
    /// position 0, whose rotation is the identity, so it is never rotated and the
    /// joint table is never needed.
    img_freqs_d: Buf,

    sigmas: []f32,
    mods: []f32,
    mod_stride: usize,
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
        refs: []const mageflow.Ref,
        sigmas: []const f32,
    ) !Session {
        std.debug.assert(cond.len == seq_txt * mageflow.txt_dim);
        const n_img = lat_h * lat_w;
        var n_tok = n_img;
        for (refs) |r| n_tok += r.h * r.w;
        var l = try lin_gpu.Lin.init(gpa, ctx, lin_gpu.maxRows(model.device_lins), 0, 0);
        errdefer l.deinit();
        var lp = try linPlan(ctx, model);
        if (force_weight_only) lp.int8 = .weight_only;
        if (model.lora) |st| {
            lp.lora = st;
            try l.presizeLora(st, model.device_lins, std.mem.alignForward(usize, @max(seq_txt, n_tok), 128));
        }

        var self: Session = .{
            .lin = l,
            .lin_plan = lp,
            .lat_h = lat_h,
            .lat_w = lat_w,
            .seq_txt = seq_txt,
            .n_img = n_img,
            .n_tok = n_tok,
            .seq = seq_txt + n_tok,
            .txt_d = undefined,
            .tokens = &.{},
            .img_freqs_d = undefined,
            .sigmas = &.{},
            .mods = &.{},
            .mod_stride = model.modStride(),
        };
        var made: usize = 0;
        errdefer {
            const bufs = [_]*Buf{ &self.txt_d, &self.img_freqs_d };
            for (bufs[0..made]) |b| ctx.tensorDestroy(b);
            if (self.tokens.len != 0) gpa.free(self.tokens);
            if (self.sigmas.len != 0) gpa.free(self.sigmas);
            if (self.mods.len != 0) gpa.free(self.mods);
        }

        {
            const txt = try model.textTokens(io, gpa, cond, seq_txt);
            defer gpa.free(txt);
            self.txt_d = try ctx.tensorCreate(std.mem.alignForward(usize, seq_txt, 128) * F * 4);
            made += 1;
            try ctx.tensorUpload(self.txt_d, std.mem.sliceAsBytes(txt));
        }

        self.tokens = try gpa.alloc(f32, n_tok * channels);
        {
            var at = n_img * channels;
            for (refs) |r| {
                const n = r.h * r.w;
                mageflow.interleaveChannels(self.tokens[at..][0 .. n * channels], r.lat, n);
                at += n * channels;
            }
        }

        {
            var freqs = try model.ropeFreqs(gpa, lat_h, lat_w, seq_txt, refs);
            defer freqs.deinit(gpa);
            // Only the image rows: row 0 of this table is the image half's first
            // position, not the conditioning's.
            const flat = try gpa.alloc(f32, 2 * n_tok * half);
            defer gpa.free(flat);
            @memcpy(flat[0 .. n_tok * half], freqs.cos[seq_txt * half ..]);
            @memcpy(flat[n_tok * half ..], freqs.sin[seq_txt * half ..]);
            self.img_freqs_d = try ctx.tensorCreate(flat.len * 4);
            made += 1;
            try ctx.tensorUpload(self.img_freqs_d, std.mem.sliceAsBytes(flat));
        }

        self.sigmas = try gpa.dupe(f32, sigmas);
        self.mods = try gpa.alloc(f32, sigmas.len * self.mod_stride);
        for (sigmas, 0..) |s, i| {
            const tbl = try model.modulationTable(io, gpa, s);
            defer gpa.free(tbl);
            @memcpy(self.mods[i * self.mod_stride ..][0..self.mod_stride], tbl);
        }
        return self;
    }

    pub fn deinit(self: *Session, gpa: std.mem.Allocator, ctx: *gpu.Context) void {
        ctx.tensorDestroy(&self.txt_d);
        ctx.tensorDestroy(&self.img_freqs_d);
        gpa.free(self.tokens);
        gpa.free(self.sigmas);
        gpa.free(self.mods);
        self.lin.deinit();
        self.* = undefined;
    }

    fn modsFor(self: *const Session, sigma: f32) ?[]const f32 {
        for (self.sigmas, 0..) |s, i| {
            if (s == sigma) return self.mods[i * self.mod_stride ..][0..self.mod_stride];
        }
        return null;
    }
};

pub const Workspace = struct {
    /// The two residual streams. Each kernel below reads from row 0 of one of
    /// them; nothing but the attention ever sees the two together.
    xt_d: Buf,
    xi_d: Buf,
    /// Modulated-norm output, and after the attention the staging the joint
    /// result is split back into. Dead between those two uses.
    ntxt_d: Buf,
    nimg_d: Buf,
    qt_d: Buf,
    kt_d: Buf,
    vt_d: Buf,
    qi_d: Buf,
    ki_d: Buf,
    vi_d: Buf,
    /// The joint operands, text rows then image rows, built by copy.
    q_d: Buf,
    k_d: Buf,
    v_d: Buf,
    attn_d: Buf,
    dtxt_d: Buf,
    dimg_d: Buf,
    mt_d: Buf,
    mi_d: Buf,
    mod_d: Buf,
    imgin_d: Buf,
    out_d: Buf,
    /// Tensor-core attention operands and scratch.
    qh_d: Buf,
    kh_d: Buf,
    v16_d: Buf,
    s_d: Buf,
    part_d: Buf,
    md_d: Buf,

    const fields = [_][]const u8{
        "xt_d",  "xi_d",   "ntxt_d", "nimg_d",  "qt_d",  "kt_d",  "vt_d",
        "qi_d",  "ki_d",   "vi_d",   "q_d",     "k_d",   "v_d",   "attn_d",
        "dtxt_d", "dimg_d", "mt_d",  "mi_d",    "mod_d", "imgin_d", "out_d",
        "qh_d",  "kh_d",   "v16_d",  "s_d",     "part_d", "md_d",
    };

    pub fn init(ctx: *gpu.Context, model: *const DiT, seq_txt: usize, n_tok: usize, act: gpu.Dt) !Workspace {
        const tp = std.mem.alignForward(usize, seq_txt, 128);
        const ip = std.mem.alignForward(usize, n_tok, 128);
        const seq_pad = std.mem.alignForward(usize, seq_txt + n_tok, 128);
        const tc = useTensorCoreAttn(ctx);
        const hpb = if (tc) headsPerBatch(seq_pad, scoresCap(ctx.budget_override), null) else 1;
        const sizes = [fields.len]usize{
            tp * F * 4,
            ip * F * 4,
            tp * F * 4,
            ip * F * 4,
            tp * F * 4,
            tp * F * 4,
            tp * F * 4,
            ip * F * 4,
            ip * F * 4,
            ip * F * 4,
            seq_pad * F * 4,
            seq_pad * F * 4,
            seq_pad * F * 4,
            seq_pad * F * 4,
            tp * F * 4,
            ip * F * 4,
            tp * mlp_dim * 4,
            ip * mlp_dim * 4,
            model.modStride() * 4,
            n_tok * channels * 4,
            n_tok * channels * 4,
            if (tc) seq_pad * F * 2 else 16,
            if (tc) F * seq_pad * 2 else 16,
            if (tc) seq_pad * F * 2 else 16,
            if (tc) hpb * seq_pad * seq_pad * 2 else 16,
            if (tc) hpb * (seq_txt + n_tok) * nchunks * 2 * 4 else 16,
            if (tc) hpb * seq_pad * 2 * 4 else 16,
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
        // The attention block stays f32 (`mageflow_dev.streamDtFor`).
        inline for (.{ "xt_d", "xi_d", "ntxt_d", "nimg_d", "dtxt_d", "dimg_d", "mt_d", "mi_d" }) |name| @field(self, name).dt = act;
        return self;
    }

    pub fn deinit(self: *Workspace, ctx: *gpu.Context) void {
        inline for (fields) |name| ctx.tensorDestroy(&@field(self, name));
        self.* = undefined;
    }
};

// The MLP down projection reads past f16 at sigma 1.
const lin_use: lin_gpu.Use = .{ .blockq = .auto, .act16 = .bf16 };

/// The block linears' plan, without a LoRA.
pub fn linPlan(ctx: *gpu.Context, model: *const DiT) !lin_gpu.Plan {
    return lin_gpu.plan(ctx, model.device_lins, lin_use, "mageflow_gpu");
}

/// Whether this context can run every block linear this checkpoint has.
pub fn supported(ctx: *gpu.Context, model: *const DiT) bool {
    if (model.blocks.len == 0) return false;
    if (!ctx.hasLnModSg()) {
        std.log.warn("mageflow_gpu: this device has no fused LayerNorm+modulate kernel; " ++
            "the trunk runs on the CPU. Expect CPU sampling speed.", .{});
        return false;
    }
    return lin_gpu.runs(ctx, model.device_lins, lin_use, model.lora, "mageflow_gpu");
}

/// One denoiser forward. `out`/`x_lat` are planar `[channels][lat_h][lat_w]`.
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
    std.debug.assert(x_lat.len == channels * sess.lat_h * sess.lat_w);
    std.debug.assert(out.len == x_lat.len);

    var owned: ?[]f32 = null;
    defer if (owned) |b| gpa.free(b);
    const mods = sess.modsFor(sigma) orelse blk: {
        owned = try model.modulationTable(io, gpa, sigma);
        break :blk owned.?;
    };
    mageflow.interleaveChannels(sess.tokens[0 .. sess.n_img * channels], x_lat, sess.n_img);
    if (profile) p_last = std.Io.Clock.real.now(io);
    try ctx.tensorUpload(ws.mod_d, std.mem.sliceAsBytes(mods));
    try ctx.tensorUpload(ws.imgin_d, std.mem.sliceAsBytes(sess.tokens));
    lap(io, &prof.xfer_ns);

    // One submission for the whole forward. Without this every op is its own
    // submit-and-wait, which at ~570 ops a forward is pure queue latency.
    const batched = !profile and !force_unbatched;
    if (batched) try ctx.beginBatch();
    errdefer if (ctx.batching) ctx.abortBatch();
    try mageflow_dev.body(Dev{ .model = model, .ctx = ctx, .io = io, .sess = sess, .ws = ws }, model, cancel);
    if (batched) try ctx.endBatch();
    // A 16-bit stream that overflowed can leave the output finite (`allFinite`).
    if (ws.xt_d.dt != .f32 and !(try ctx.allFinite(ws.xt_d, sess.seq_txt * F) and try ctx.allFinite(ws.xi_d, sess.n_tok * F))) return error.NonFiniteActivation;

    const final = try gpa.alloc(f32, sess.n_img * channels);
    defer gpa.free(final);
    try ctx.tensorDownload(ws.out_d, std.mem.sliceAsBytes(final));
    lap(io, &prof.xfer_ns);
    mageflow.deinterleaveChannels(out, final, sess.n_img);
    lap(io, &prof.cpu_ns);
}

/// `mageflow_dev`'s ops on this backend: a buffer per stream, the joint q/k/v assembled
/// by copies (see the header's note 3 for why a copy and not a view), and the attention
/// output split back the same way.
const Dev = struct {
    model: *const DiT,
    ctx: *gpu.Context,
    io: std.Io,
    sess: *Session,
    ws: *Workspace,

    fn gemmBucket(stream: mageflow_dev.Stream) *i128 {
        return if (stream == .txt) &prof.gemm_txt_ns else &prof.gemm_ns;
    }

    pub fn streams(d: Dev) Streams {
        const ws = d.ws;
        return .{
            .seq_txt = d.sess.seq_txt,
            .n_tok = d.sess.n_tok,
            .txt_x = ws.xt_d,
            .img_x = ws.xi_d,
            .tq = ws.qt_d,
            .tk = ws.kt_d,
            .tv = ws.vt_d,
            .iq = ws.qi_d,
            .ik = ws.ki_d,
            .iv = ws.vi_d,
        };
    }
    const Streams = struct { seq_txt: usize, n_tok: usize, txt_x: Buf, img_x: Buf, tq: Buf, tk: Buf, tv: Buf, iq: Buf, ik: Buf, iv: Buf };

    pub fn embed(d: Dev) !void {
        const ws = d.ws;
        const n = d.sess.n_tok;
        try copyInto(d.ctx, d.sess.txt_d, 0, ws.xt_d, 0, d.sess.seq_txt * F);
        try d.sess.lin.linearAt(d.sess.lin_plan, ws.xi_d, 0, ws.imgin_d, 0, n, d.model.img_in.w, d.model.img_in.b);
        lap(d.io, &prof.gemm_ns);
        File.trace(d.ctx, "img_in", ws.xi_d, n * F);
        File.trace(d.ctx, "txt_in", ws.xt_d, d.sess.seq_txt * F);
    }

    pub fn blockBegin(_: Dev, _: usize) !void {}

    pub fn trace(d: Dev, what: []const u8, _: usize, buf: Buf, elems: usize) void {
        File.trace(d.ctx, what, buf, elems);
    }

    pub fn traceBlock(d: Dev, _: usize) void {
        File.trace(d.ctx, "blk.img", d.ws.xi_d, d.sess.n_tok * F);
        File.trace(d.ctx, "blk.txt", d.ws.xt_d, d.sess.seq_txt * F);
    }

    pub fn lnMod(d: Dev, src: Buf, dst: Buf, premul: usize, shift: usize, n: usize) !void {
        try d.ctx.opLnModSg(src, dst, d.ws.mod_d, n, F, @intCast(premul), @intCast(shift), eps);
        lap(d.io, &prof.norm_ns);
    }

    pub fn gemms(d: Dev, stream: mageflow_dev.Stream, x: Buf, m: usize, lws: []const mageflow.LinearW, ys: []const Buf) !void {
        var w: [3]Weight = undefined;
        for (lws, 0..) |lw, k| w[k] = lw.w;
        const g = try prep(d.sess, x, m, w[0..lws.len]);
        for (lws, ys) |lw, y| try gemm(d.sess, g, y, lw);
        lap(d.io, gemmBucket(stream));
    }

    pub fn linear(d: Dev, stream: mageflow_dev.Stream, y: Buf, x: Buf, m: usize, lw: mageflow.LinearW) !void {
        try File.linear(d.sess, y, x, m, lw);
        lap(d.io, gemmBucket(stream));
    }

    /// Per-head QK-norm with each stream's own weight; RoPE on the image stream only
    /// (the text rows sit at position 0, whose rotation is the identity); then the
    /// joint operands `[text | image]`.
    pub fn qkPrepare(d: Dev, blk: *const mageflow.Block) !void {
        const ctx = d.ctx;
        const ws = d.ws;
        const seq_txt = d.sess.seq_txt;
        const n = d.sess.n_tok;
        try rmsNorm(ctx, ws.qt_d, ws.qt_d, blk.attn.txt_qnorm, seq_txt * heads, hd, eps);
        try rmsNorm(ctx, ws.kt_d, ws.kt_d, blk.attn.txt_knorm, seq_txt * heads, hd, eps);
        try rmsNorm(ctx, ws.qi_d, ws.qi_d, blk.attn.qnorm, n * heads, hd, eps);
        try rmsNorm(ctx, ws.ki_d, ws.ki_d, blk.attn.knorm, n * heads, hd, eps);
        lap(d.io, &prof.norm_ns);
        const rope: gpu.EltPush = .{ .u0 = @intCast(n * heads * half), .u1 = half, .u2 = @intCast(n * half), .u3 = heads };
        try ctx.opElt(.rope_inter, ws.qi_d, null, d.sess.img_freqs_d, null, rope, n * heads * half, 1, 1);
        try ctx.opElt(.rope_inter, ws.ki_d, null, d.sess.img_freqs_d, null, rope, n * heads * half, 1, 1);
        lap(d.io, &prof.elt_ns);
        const img_off = seq_txt * F;
        try copyInto(ctx, ws.qt_d, 0, ws.q_d, 0, seq_txt * F);
        try copyInto(ctx, ws.kt_d, 0, ws.k_d, 0, seq_txt * F);
        try copyInto(ctx, ws.vt_d, 0, ws.v_d, 0, seq_txt * F);
        try copyInto(ctx, ws.qi_d, 0, ws.q_d, img_off, n * F);
        try copyInto(ctx, ws.ki_d, 0, ws.k_d, img_off, n * F);
        try copyInto(ctx, ws.vi_d, 0, ws.v_d, img_off, n * F);
        lap(d.io, &prof.copy_ns);
    }

    pub fn scale(d: Dev, buf: Buf, n: usize, f: f32) !void {
        try d.ctx.opElt(.scale_f32, buf, buf, null, null, .{ .u0 = @intCast(n), .f0 = f }, n, 1, 1);
        lap(d.io, &prof.elt_ns);
    }

    pub fn attention(d: Dev, seq: usize) !void {
        try File.attention(d.ctx, d.ws, seq);
        lap(d.io, &prof.attn_ns);
    }

    pub fn attnSplit(d: Dev) !struct { txt: Buf, img: Buf } {
        const ws = d.ws;
        const seq_txt = d.sess.seq_txt;
        try copyInto(d.ctx, ws.attn_d, 0, ws.ntxt_d, 0, seq_txt * F);
        try copyInto(d.ctx, ws.attn_d, seq_txt * F, ws.nimg_d, 0, d.sess.n_tok * F);
        lap(d.io, &prof.copy_ns);
        return .{ .txt = ws.ntxt_d, .img = ws.nimg_d };
    }

    pub fn gatedAdd(d: Dev, x: Buf, t: Buf, n: usize, off: usize) !void {
        try File.gatedAdd(d.ctx, x, t, d.ws.mod_d, n, @intCast(off));
        lap(d.io, &prof.elt_ns);
    }

    pub fn gelu(d: Dev, buf: Buf, n: usize) !void {
        try d.ctx.opElt(gpu.eltAt("gelu", buf.dt), buf, null, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
        lap(d.io, &prof.elt_ns);
    }

    pub fn final(d: Dev, fin: usize) !void {
        const ws = d.ws;
        const n = d.sess.n_img;
        try d.ctx.opLnModSg(ws.xi_d, ws.xi_d, ws.mod_d, n, F, @intCast(fin), @intCast(fin + F), eps);
        lap(d.io, &prof.norm_ns);
        File.trace(d.ctx, "fin_x", ws.xi_d, n * F);
        try d.sess.lin.linearAt(d.sess.lin_plan, ws.out_d, 0, ws.xi_d, 0, n, d.model.proj_out.w, d.model.proj_out.b);
        lap(d.io, &prof.gemm_ns);
        File.trace(d.ctx, "proj_out", ws.out_d, n * channels);
    }
};

/// Elements `[src_off, src_off + n)` of `src` into `dst` from `dst_off`, each at its
/// own dtype.
fn copyInto(ctx: *gpu.Context, src: Buf, src_off: usize, dst: Buf, dst_off: usize, n: usize) !void {
    try ctx.opElt(gpu.eltCopy(src.dt, dst.dt), src, dst, null, null, .{
        .u0 = @intCast(n),
        .u2 = @intCast(dst_off),
        .u3 = @intCast(src_off),
    }, n, 1, 1);
}


fn gatedAdd(ctx: *gpu.Context, x: Buf, delta: Buf, mod: Buf, rows: usize, gate_off: u32) !void {
    std.debug.assert(x.dt == delta.dt);
    try ctx.opElt(gpu.eltAt("gated_add", x.dt), x, delta, mod, null, .{
        .u0 = @intCast(rows * F),
        .u1 = F,
        .u2 = gate_off,
    }, rows * F, 1, 1);
}

/// Attention over `rows` positions, q/k/v already normed and rope'd, result into
/// `ws.attn_d`. Both paths are non-causal and the same arithmetic; see the header
/// for why the tensor-core one is a requirement rather than a speed choice.
fn attention(ctx: *gpu.Context, ws: *Workspace, rows: usize) !void {
    if (!useTensorCoreAttn(ctx)) {
        return ctx.opElt(.attn_full, ws.q_d, ws.k_d, ws.v_d, ws.attn_d, .{
            .u0 = @intCast(rows),
            .u1 = heads,
            .u2 = heads,
            .u3 = hd,
            .f0 = attn_scale,
        }, rows * heads, 1, 1);
    }

    const rows_pad = std.mem.alignForward(usize, rows, 128);
    ctx.independent(3);
    try ctx.opElt(.f32_to_h16, ws.q_d, null, null, ws.qh_d, .{
        .u0 = @intCast(rows_pad * F / 2),
        .u1 = @intCast(rows * F),
        .f0 = attn_scale,
    }, rows_pad * F / 2, 1, 1);
    try ctx.opElt(.gather_kmajor_h16, ws.k_d, null, null, ws.kh_d, .{
        .u0 = @intCast(F * rows_pad / 2),
        .u1 = hd,
        .u2 = @intCast(rows_pad),
        .u3 = @intCast(rows),
        .u4 = heads,
    }, F * rows_pad / 2, 1, 1);
    try ctx.opElt(.f32_to_h16, ws.v_d, null, null, ws.v16_d, .{
        .u0 = @intCast(rows_pad * F / 2),
        .u1 = @intCast(rows * F),
        .f0 = 1.0,
    }, rows_pad * F / 2, 1, 1);

    const plane = rows_pad * rows_pad * 2;
    var hb_cap = headsPerBatch(rows_pad, scoresCap(ctx.budget_override), ws.s_d.size);
    hb_cap = @max(1, @min(hb_cap, ws.part_d.size / @max(rows * nchunks * 2 * 4, 1)));
    hb_cap = @max(1, @min(hb_cap, ws.md_d.size / @max(rows_pad * 2 * 4, 1)));
    std.debug.assert(hb_cap * plane <= ws.s_d.size);

    const s_stride: u32 = @intCast(rows_pad);
    const s_plane: u32 = @intCast(rows_pad * rows_pad);
    var h0: usize = 0;
    while (h0 < heads) : (h0 += hb_cap) {
        const hb = @min(hb_cap, heads - h0);
        try ctx.opAttnScores(ws.s_d, ws.qh_d, ws.kh_d, .{
            .u0 = F,
            .u1 = s_stride,
            .u2 = @intCast(h0),
            .u3 = 1,
            .u4 = @intCast(hd * rows_pad),
            .u5 = s_plane,
        }, rows_pad / 128, rows_pad / 128, hb);
        try ctx.opElt(.softmax_partial, ws.s_d, null, null, ws.part_d, .{
            .u0 = @intCast(hb * rows * nchunks),
            .u1 = nchunks,
            .u2 = @intCast(rows),
            .u3 = s_stride,
            .u5 = s_plane,
        }, hb * rows * nchunks, 1, 1);
        try ctx.opElt(.softmax_combine, ws.part_d, null, null, ws.md_d, .{
            .u0 = @intCast(hb * rows),
            .u1 = nchunks,
            .u2 = @intCast(rows),
            .u3 = s_stride,
        }, hb * rows, 1, 1);
        try ctx.opAttnOut(ws.s_d, ws.v16_d, ws.attn_d, ws.md_d, .{
            .u0 = s_stride,
            .u1 = s_plane,
            .u2 = @intCast(h0),
            .u3 = 1,
            .u4 = F,
            .u5 = F,
            .f0 = @bitCast(@as(u32, @intCast(rows))),
            .f1 = @bitCast(s_stride),
        }, rows_pad / 128, hb);
    }
}

/// Every linear here carries a bias, folded into the GEMM where its kernel takes one.
fn prep(sess: *const Session, x: Buf, m: usize, group: []const Weight) !lin_gpu.Group {
    return sess.lin.prep(sess.lin_plan, x, m, F, group);
}

fn gemm(sess: *const Session, g: lin_gpu.Group, y: Buf, lw: anytype) !void {
    return sess.lin.gemm(sess.lin_plan, g, y, lw.w, .{ .bias = lw.b });
}

fn linear(sess: *const Session, y: Buf, x: Buf, m: usize, lw: anytype) !void {
    const g = try sess.lin.prep(sess.lin_plan, x, m, lw.w.cols, &.{lw.w});
    return gemm(sess, g, y, lw);
}

/// Weighted RMSNorm over `[rows][dim]`, `x -> out` (may alias). The subgroup
/// kernel wherever it exists, for `zimage_gpu`'s measured reason.
fn rmsNorm(ctx: *gpu.Context, x: Buf, out: Buf, weights: []const f32, rows: usize, dim: usize, e: f32) !void {
    const w = try normBuf(ctx, weights);
    if (ctx.hasSubgroupNorm()) return ctx.opRmsNormSg(x, out, w, rows, dim, e);
    try ctx.opElt(.rmsnorm, x, out, w, null, .{
        .u0 = @intCast(rows),
        .u1 = @intCast(dim),
        .f0 = e,
    }, rows, 1, 1);
}

fn normBuf(ctx: *gpu.Context, weights: []const f32) !Buf {
    const buf = try ctx.smallBuffer(std.mem.sliceAsBytes(weights));
    return .{ .buf = buf, .mem = .null_handle, .size = 0 };
}
