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
const lin = @import("lin.zig");
const gpu = @import("tp_gpu").context;
const ops = @import("tp_ops");

const DiT = mageflow.DiT;
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

/// See the header, and `mageflow_cuda.v_div` for the measurements.
const v_div_log2: u5 = 7;
const v_div: f32 = 1.0 / @as(f32, 1 << v_div_log2);

/// Force `attn_full` even where the tensor-core pipeline exists. For A/B and for
/// reproducing a mismatch; the device test runs both.
pub var force_attn_full: bool = false;

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
    ctx.tensorDownload(b, std.mem.sliceAsBytes(buf[0..take])) catch return;
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

        var self: Session = .{
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

    pub fn init(ctx: *gpu.Context, model: *const DiT, seq_txt: usize, n_tok: usize) !Workspace {
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
        return self;
    }

    pub fn deinit(self: *Workspace, ctx: *gpu.Context) void {
        inline for (fields) |name| ctx.tensorDestroy(&@field(self, name));
        self.* = undefined;
    }
};

/// Whether this context can run every block linear this checkpoint has.
pub fn supported(ctx: *gpu.Context, model: *const DiT) bool {
    if (model.blocks.len == 0) return false;
    if (!ctx.hasLnModSg()) {
        std.log.warn("mageflow_gpu: this device has no fused LayerNorm+modulate kernel — " ++
            "the trunk runs on the CPU. Expect CPU sampling speed.", .{});
        return false;
    }
    const has_f16w = ctx.pipe_coop_bf16w != .null_handle or ctx.pipe_coop_f16w != .null_handle;
    const caps: lin.Caps = .{ .f32 = true, .fp8 = true, .bf16 = has_f16w, .f16 = has_f16w, .nvfp4 = ctx.hasNvfp4Decode() };
    if (lin.unsupported(model.device_lins, caps)) |bad| {
        std.log.warn("mageflow_gpu: {s} is {t}, which this device has no GEMM for — the trunk " ++
            "runs on the CPU. Expect CPU sampling speed.", .{ bad.tag, bad.dtype });
        return false;
    }
    return true;
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
    const seq_txt = sess.seq_txt;
    const n_tok = sess.n_tok;

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

    try ctx.opElt(.copy, sess.txt_d, ws.xt_d, null, null, .{ .u0 = @intCast(seq_txt * F) }, seq_txt * F, 1, 1);
    try ctx.opMatmul(ws.xi_d, 0, ws.imgin_d, 0, n_tok, model.img_in.w.bytes, false, F, channels, model.img_in.w.scale, model.img_in.b);
    lap(io, &prof.gemm_ns);
    trace(ctx, "img_in", ws.xi_d, n_tok * F);
    trace(ctx, "txt_in", ws.xt_d, seq_txt * F);

    for (model.blocks, 0..) |*blk, i| {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        try blockForward(ctx, io, sess, ws, blk, i);
        trace(ctx, "blk.img", ws.xi_d, n_tok * F);
        trace(ctx, "blk.txt", ws.xt_d, seq_txt * F);
    }

    // The final layer, on the canvas rows alone: it is row-wise, so running it on
    // them is what ComfyUI's "whole stream, then slice" computes.
    const fin: u32 = @intCast(model.blocks.len * 12 * F);
    try ctx.opLnModSg(ws.xi_d, ws.xi_d, ws.mod_d, sess.n_img, F, fin, fin + F, eps);
    lap(io, &prof.norm_ns);
    trace(ctx, "fin_x", ws.xi_d, sess.n_img * F);
    try ctx.opMatmul(ws.out_d, 0, ws.xi_d, 0, sess.n_img, model.proj_out.w.bytes, false, channels, F, model.proj_out.w.scale, model.proj_out.b);
    lap(io, &prof.gemm_ns);
    trace(ctx, "proj_out", ws.out_d, sess.n_img * channels);

    if (batched) try ctx.endBatch();

    const final = try gpa.alloc(f32, sess.n_img * channels);
    defer gpa.free(final);
    try ctx.tensorDownload(ws.out_d, std.mem.sliceAsBytes(final));
    lap(io, &prof.xfer_ns);
    mageflow.deinterleaveChannels(out, final, sess.n_img);
    lap(io, &prof.cpu_ns);
}

fn copyInto(ctx: *gpu.Context, src: Buf, src_off: usize, dst: Buf, dst_off: usize, n: usize) !void {
    try ctx.opElt(.copy, src, dst, null, null, .{
        .u0 = @intCast(n),
        .u2 = @intCast(dst_off),
        .u3 = @intCast(src_off),
    }, n, 1, 1);
}

fn blockForward(ctx: *gpu.Context, io: std.Io, sess: *Session, ws: *Workspace, blk: *const mageflow.Block, block_i: usize) !void {
    const seq_txt = sess.seq_txt;
    const n_tok = sess.n_tok;
    const seq = sess.seq;
    const img_m: u32 = @intCast(block_i * 12 * F);
    const txt_m: u32 = img_m + 6 * F;

    // --- attention half -----------------------------------------------------
    try ctx.opLnModSg(ws.xt_d, ws.ntxt_d, ws.mod_d, seq_txt, F, txt_m + 0 * F, txt_m + 1 * F, eps);
    try ctx.opLnModSg(ws.xi_d, ws.nimg_d, ws.mod_d, n_tok, F, img_m + 0 * F, img_m + 1 * F, eps);
    lap(io, &prof.norm_ns);

    try gemm(ctx, ws.qt_d, 0, ws.ntxt_d, seq_txt, blk.attn.txt_q);
    try gemm(ctx, ws.kt_d, 0, ws.ntxt_d, seq_txt, blk.attn.txt_k);
    try gemm(ctx, ws.vt_d, 0, ws.ntxt_d, seq_txt, blk.attn.txt_v);
    lap(io, &prof.gemm_txt_ns);
    try gemm(ctx, ws.qi_d, 0, ws.nimg_d, n_tok, blk.attn.q);
    try gemm(ctx, ws.ki_d, 0, ws.nimg_d, n_tok, blk.attn.k);
    try gemm(ctx, ws.vi_d, 0, ws.nimg_d, n_tok, blk.attn.v);
    lap(io, &prof.gemm_ns);

    // Per-head QK-norm, each stream with its own weight. RoPE is IMAGE-ONLY: the
    // text rows sit at position 0, whose rotation is the identity.
    try rmsNorm(ctx, ws.qt_d, ws.qt_d, blk.attn.txt_qnorm, seq_txt * heads, hd, eps);
    try rmsNorm(ctx, ws.kt_d, ws.kt_d, blk.attn.txt_knorm, seq_txt * heads, hd, eps);
    try rmsNorm(ctx, ws.qi_d, ws.qi_d, blk.attn.qnorm, n_tok * heads, hd, eps);
    try rmsNorm(ctx, ws.ki_d, ws.ki_d, blk.attn.knorm, n_tok * heads, hd, eps);
    lap(io, &prof.norm_ns);

    const rope: gpu.EltPush = .{
        .u0 = @intCast(n_tok * heads * half),
        .u1 = half,
        .u2 = @intCast(n_tok * half),
        .u3 = heads,
    };
    try ctx.opElt(.rope_inter, ws.qi_d, null, sess.img_freqs_d, null, rope, n_tok * heads * half, 1, 1);
    try ctx.opElt(.rope_inter, ws.ki_d, null, sess.img_freqs_d, null, rope, n_tok * heads * half, 1, 1);
    lap(io, &prof.elt_ns);

    // Assemble the joint operands: see the header's note 3 for why this is a copy
    // and not a view.
    const img_off = seq_txt * F;
    try copyInto(ctx, ws.qt_d, 0, ws.q_d, 0, seq_txt * F);
    try copyInto(ctx, ws.kt_d, 0, ws.k_d, 0, seq_txt * F);
    try copyInto(ctx, ws.vt_d, 0, ws.v_d, 0, seq_txt * F);
    try copyInto(ctx, ws.qi_d, 0, ws.q_d, img_off, n_tok * F);
    try copyInto(ctx, ws.ki_d, 0, ws.k_d, img_off, n_tok * F);
    try copyInto(ctx, ws.vi_d, 0, ws.v_d, img_off, n_tok * F);
    lap(io, &prof.copy_ns);

    trace(ctx, "  q", ws.q_d, seq * F);
    trace(ctx, "  v", ws.v_d, seq * F);
    // See `v_div`: V is unnormed and outgrows f16 on a real conditioning.
    try ctx.opElt(.scale_f32, ws.v_d, ws.v_d, null, null, .{ .u0 = @intCast(seq * F), .f0 = v_div }, seq * F, 1, 1);
    lap(io, &prof.elt_ns);
    try attention(ctx, ws, seq);
    lap(io, &prof.attn_ns);
    try ctx.opElt(.scale_f32, ws.attn_d, ws.attn_d, null, null, .{ .u0 = @intCast(seq * F), .f0 = 1.0 / v_div }, seq * F, 1, 1);
    lap(io, &prof.elt_ns);

    trace(ctx, "  attn", ws.attn_d, seq * F);
    try copyInto(ctx, ws.attn_d, 0, ws.ntxt_d, 0, seq_txt * F);
    try copyInto(ctx, ws.attn_d, img_off, ws.nimg_d, 0, n_tok * F);
    lap(io, &prof.copy_ns);
    try gemm(ctx, ws.dtxt_d, 0, ws.ntxt_d, seq_txt, blk.attn.txt_o);
    lap(io, &prof.gemm_txt_ns);
    try gemm(ctx, ws.dimg_d, 0, ws.nimg_d, n_tok, blk.attn.o);
    lap(io, &prof.gemm_ns);
    try gatedAdd(ctx, ws.xt_d, ws.dtxt_d, ws.mod_d, seq_txt, txt_m + 2 * F);
    try gatedAdd(ctx, ws.xi_d, ws.dimg_d, ws.mod_d, n_tok, img_m + 2 * F);
    lap(io, &prof.elt_ns);

    // --- MLP half -----------------------------------------------------------
    try ctx.opLnModSg(ws.xt_d, ws.ntxt_d, ws.mod_d, seq_txt, F, txt_m + 3 * F, txt_m + 4 * F, eps);
    try ctx.opLnModSg(ws.xi_d, ws.nimg_d, ws.mod_d, n_tok, F, img_m + 3 * F, img_m + 4 * F, eps);
    lap(io, &prof.norm_ns);

    try gemm(ctx, ws.mt_d, 0, ws.ntxt_d, seq_txt, blk.txt_mlp.in);
    lap(io, &prof.gemm_txt_ns);
    try ctx.opElt(.gelu, ws.mt_d, null, null, null, .{ .u0 = @intCast(seq_txt * mlp_dim) }, seq_txt * mlp_dim, 1, 1);
    lap(io, &prof.elt_ns);
    try gemm(ctx, ws.dtxt_d, 0, ws.mt_d, seq_txt, blk.txt_mlp.out);
    lap(io, &prof.gemm_txt_ns);

    try gemm(ctx, ws.mi_d, 0, ws.nimg_d, n_tok, blk.img_mlp.in);
    lap(io, &prof.gemm_ns);
    try ctx.opElt(.gelu, ws.mi_d, null, null, null, .{ .u0 = @intCast(n_tok * mlp_dim) }, n_tok * mlp_dim, 1, 1);
    lap(io, &prof.elt_ns);
    try gemm(ctx, ws.dimg_d, 0, ws.mi_d, n_tok, blk.img_mlp.out);
    lap(io, &prof.gemm_ns);

    try gatedAdd(ctx, ws.xt_d, ws.dtxt_d, ws.mod_d, seq_txt, txt_m + 5 * F);
    try gatedAdd(ctx, ws.xi_d, ws.dimg_d, ws.mod_d, n_tok, img_m + 5 * F);
    lap(io, &prof.elt_ns);
}

fn gatedAdd(ctx: *gpu.Context, x: Buf, delta: Buf, mod: Buf, rows: usize, gate_off: u32) !void {
    try ctx.opElt(.gated_add, x, delta, mod, null, .{
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

/// One block linear, dispatched by weight dtype, writing at `y_off` FLOATS into
/// `y`. Unlike krea2's and Z-Image's, every linear here carries a bias, and the
/// coop GEMMs fold one in, so there is no separate bias pass.
fn gemm(ctx: *gpu.Context, y: Buf, y_off: u32, x: Buf, m: usize, lw: anytype) !void {
    const w: Weight = lw.w;
    const bias: []const f32 = lw.b orelse &zero_bias;
    std.debug.assert(w.rows <= zero_bias.len);
    switch (w.dtype) {
        .bf16 => {
            if (ctx.pipe_coop_bf16w != .null_handle) {
                try ctx.opMatmulCoopBf16(y, y_off, x, m, w.bytes, w.rows, w.cols, bias);
            } else {
                try ctx.opMatmulCoopF16Wb(y, y_off, x, m, w.bytes, w.rows, w.cols, bias);
            }
        },
        .f16 => try ctx.opMatmulCoopF16Wh(y, y_off, x, m, w.bytes, w.rows, w.cols, bias),
        .nvfp4 => {
            std.debug.assert(w.rows % 128 == 0 and w.cols % 32 == 0);
            const meta = w.nvfp4.?;
            try ctx.opMatmulNvfp4(y, x, m, w.bytes, meta.scales, std.mem.asBytes(&meta.levels.bf16v), w.rows, w.cols, bias);
        },
        .f32, .f8_e4m3 => try ctx.opMatmul(y, y_off, x, 0, m, w.bytes, w.dtype == .f8_e4m3, w.rows, w.cols, w.scale, lw.b),
        // `supported` gates this before a session is built.
        else => return error.UnsupportedDType,
    }
}

/// Widest zero bias any Mage-Flow GEMM needs, for the dtypes whose kernel always
/// folds one. Passed WHOLE, never sliced: `smallBuffer` caches by host POINTER, so
/// a sliced one would map every width to whichever length was uploaded first.
const zero_bias: [mlp_dim]f32 = @splat(0);

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
