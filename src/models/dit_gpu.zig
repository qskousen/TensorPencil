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
const gpu = @import("../gpu/context.zig");

const DiT = dit.DiT;

/// When true, each forward prints a per-category time breakdown (sync-per-op
/// submission makes host-side timing exact).
pub var profile: bool = false;

const Prof = struct {
    matmul_ns: i96 = 0,
    attn_ns: i96 = 0,
    scores_ns: i96 = 0,
    smax_ns: i96 = 0,
    aout_ns: i96 = 0,
    elt_ns: i96 = 0,
    xfer_ns: i96 = 0,
    cpu_ns: i96 = 0,
};
const F = dit.features;
const heads = dit.n_heads;
const kv_heads = dit.n_kv_heads;
const hd = dit.head_dim;
const attn_scale: f32 = 1.0 / 11.313708498984761; // 1/sqrt(128)

/// Per-sampling-run cache: everything constant across steps — the text
/// fusion tokens (a full CPU transformer pass), the rope table, and the
/// timestep vectors for the whole schedule — computed once and, where
/// device-resident, uploaded once.
pub const Session = struct {
    seq_txt: usize,
    lat_h: usize,
    lat_w: usize,
    /// Text tokens uploaded once; copied device-to-device into x each step.
    txt0_d: gpu.DeviceBuffer,
    txt_len: usize,
    freqs_d: gpu.DeviceBuffer,
    /// Per schedule entry: t (F) then tvec (6F).
    sigmas: []f32,
    tvs: []f32,

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
            .txt0_d = txt0_d,
            .txt_len = txt_tokens.len,
            .freqs_d = freqs_d,
            .sigmas = sig,
            .tvs = tvs,
        };
    }

    pub fn deinit(self: *Session, gpa: std.mem.Allocator, ctx: *gpu.Context) void {
        ctx.tensorDestroy(&self.txt0_d);
        ctx.tensorDestroy(&self.freqs_d);
        gpa.free(self.sigmas);
        gpa.free(self.tvs);
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

pub fn forward(
    model: *const DiT,
    ctx: *gpu.Context,
    sess: *const Session,
    io: std.Io,
    gpa: std.mem.Allocator,
    out: []f32,
    x_lat: []const f32,
    sigma: f32,
) !void {
    const lat_h = sess.lat_h;
    const lat_w = sess.lat_w;
    const seq_txt = sess.seq_txt;
    const h = lat_h / dit.patch;
    const w = lat_w / dit.patch;
    const n_img = h * w;
    const seq = seq_txt + n_img;
    // Cooperative-matrix GEMMs compute a tile-padded m (128-row workgroup
    // tiles); buffers that receive GEMM output are sized for seq_pad (pad
    // rows stay zero).
    const seq_pad = std.mem.alignForward(usize, seq, 128);
    const coop = ctx.pipe_coop != .null_handle;
    var prof: Prof = .{};
    var t_mark = std.Io.Clock.real.now(io);
    const mark = struct {
        fn lap(io_: std.Io, m: *std.Io.Timestamp, bucket: *i96) void {
            const now = std.Io.Clock.real.now(io_);
            bucket.* += now.nanoseconds - m.nanoseconds;
            m.* = now;
        }
    }.lap;

    // CPU: timestep vectors (cached in the session for schedule sigmas) and
    // per-block modulation (tvec + block offset).
    const half = hd / 2;
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
    for (model.blocks, 0..) |*blk, b| {
        for (mv[b * 6 * F ..][0 .. 6 * F], tv_vec, blk.mod) |*m, tvv, bm| m.* = tvv + bm;
        // Slots 0/3 (pre/post modulation scale) carry the rmsnorm weight
        // prefolded: rms_apply_mod computes x*inv_rms*premul + shift.
        const base = b * 6 * F;
        for (0..F) |c| {
            mv[base + c] = (1.0 + mv[base + c]) * blk.prenorm[c];
            mv[base + 3 * F + c] = (1.0 + mv[base + 3 * F + c]) * blk.postnorm[c];
        }
    }
    // Final-layer modulation, same folded form (premul then shift).
    const fin = try gpa.alloc(f32, 2 * F);
    defer gpa.free(fin);
    for (0..F) |c| {
        fin[c] = (1.0 + tv_t[c] + model.last_mod[c]) * model.last_norm[c];
        fin[F + c] = tv_t[c] + model.last_mod[F + c];
    }

    // CPU: image patches (text tokens and rope live in the session).
    const img_in = try DiT.patchify(gpa, x_lat, lat_h, lat_w);
    defer gpa.free(img_in);

    mark(io, &t_mark, &prof.cpu_ns);

    // Device tensors.
    var x_d = try ctx.tensorCreate(seq_pad * F * 4);
    defer ctx.tensorDestroy(&x_d);
    var t1_d = try ctx.tensorCreate(seq_pad * F * 4);
    defer ctx.tensorDestroy(&t1_d);
    var q_d = try ctx.tensorCreate(seq_pad * heads * hd * 4);
    defer ctx.tensorDestroy(&q_d);
    var k_d = try ctx.tensorCreate(seq_pad * kv_heads * hd * 4);
    defer ctx.tensorDestroy(&k_d);
    var v_d = try ctx.tensorCreate(seq_pad * kv_heads * hd * 4);
    defer ctx.tensorDestroy(&v_d);
    var g_d = try ctx.tensorCreate(seq_pad * F * 4);
    defer ctx.tensorDestroy(&g_d);
    // Flash attention (coopmat.buildFlashAttn) keeps the {m, 1/d} table in
    // the tail of attn_d and never materializes S — but measured SLOWER than
    // the two-pass path at DiT sizes (attn 1.49 -> 2.43 s at 1120x1680): the
    // out pass recomputes S at the global-fragment-load rate (~0.8 s), which
    // costs more than the coalesced S write + reads it eliminates (~0.5 s).
    // It would win only with both operands staged in shared, which doesn't
    // fit the 48 KB workgroup ceiling next to the S tile at this tiling.
    // Kept behind this switch (kernels are unit-tested) for smaller-seq
    // experiments or a future retiling.
    const use_flash = false;
    const flash = use_flash and ctx.pipe_flash_md != .null_handle;
    const md_off = seq_pad * F;
    var attn_d = try ctx.tensorCreate(seq_pad * F * 4 + if (flash) heads * seq_pad * 2 * 4 else 0);
    defer ctx.tensorDestroy(&attn_d);
    var mg_d = try ctx.tensorCreate(seq_pad * dit.mlp_dim * 4);
    defer ctx.tensorDestroy(&mg_d);
    var mu_d = try ctx.tensorCreate(seq_pad * dit.mlp_dim * 4);
    defer ctx.tensorDestroy(&mu_d);
    var mv_d = try ctx.tensorCreate(mv.len * 4);
    defer ctx.tensorDestroy(&mv_d);
    var fin_d = try ctx.tensorCreate(fin.len * 4);
    defer ctx.tensorDestroy(&fin_d);
    const freqs_d = sess.freqs_d;
    var imgin_d = try ctx.tensorCreate(img_in.len * 4);
    defer ctx.tensorDestroy(&imgin_d);
    // Attention operands. With the tensor-core scores pipeline, Q converts
    // to f16 (softmax scale prefolded) and K gathers into a per-head k-major
    // f16 block with seq_pad columns (zero padded); the f32 fallback keeps
    // the old k-major gathers (+32 B pad: its edge tiles read a few elements
    // past the last position, masked at store).
    comptime std.debug.assert(hd == 128); // buildGemmScores unrolls hd/16
    const tc_attn = ctx.pipe_scores != .null_handle;
    var qt_d = try ctx.tensorCreate(if (tc_attn) seq_pad * heads * hd * 2 else seq * heads * hd * 4 + 32);
    defer ctx.tensorDestroy(&qt_d);
    var kt_d = try ctx.tensorCreate(if (tc_attn) kv_heads * hd * seq_pad * 2 else seq * kv_heads * hd * 4 + 32);
    defer ctx.tensorDestroy(&kt_d);
    // Attention scores batched over head groups to bound VRAM (~<= 2 GiB).
    // The flash path never materializes S (or the softmax scratch).
    const s_rows = if (tc_attn) seq_pad else seq;
    const s_esize: usize = if (tc_attn) 2 else 4; // coop path stores S f16
    const heads_per_batch: usize = @max(1, @min(heads, (2 << 30) / (s_rows * s_rows * s_esize)));
    var s_d = try ctx.tensorCreate(if (flash) 16 else heads_per_batch * s_rows * s_rows * s_esize);
    defer ctx.tensorDestroy(&s_d);
    // Two-pass softmax state + f16 V for the tensor-core P@V GEMM. 32
    // interleaved chunks per row: a warp covers a row with coalesced reads.
    const nchunks = 32;
    var v16_d = try ctx.tensorCreate(if (tc_attn) seq_pad * kv_heads * hd * 2 else 16);
    defer ctx.tensorDestroy(&v16_d);
    var part_d = try ctx.tensorCreate(if (tc_attn and !flash) heads_per_batch * seq * nchunks * 2 * 4 else 16);
    defer ctx.tensorDestroy(&part_d);
    var md_d = try ctx.tensorCreate(if (tc_attn and !flash) heads_per_batch * seq_pad * 2 * 4 else 16);
    defer ctx.tensorDestroy(&md_d);
    // Parallel rmsnorm state (32 interleaved chunks per row) and the f16
    // scratch the fused gate kernels write for the wo / mlp.down GEMMs.
    const rms_ch = 32;
    var rmsp_d = try ctx.tensorCreate(seq * rms_ch * 4);
    defer ctx.tensorDestroy(&rmsp_d);
    var rmsi_d = try ctx.tensorCreate(seq * 4);
    defer ctx.tensorDestroy(&rmsi_d);
    var h16_d = try ctx.tensorCreate(if (coop) seq_pad * dit.mlp_dim * 2 else 16);
    defer ctx.tensorDestroy(&h16_d);

    try ctx.tensorUpload(mv_d, std.mem.sliceAsBytes(mv));
    try ctx.tensorUpload(fin_d, std.mem.sliceAsBytes(fin));
    try ctx.tensorCopy(x_d, 0, sess.txt0_d, 0, sess.txt_len * 4);
    try ctx.tensorUpload(imgin_d, std.mem.sliceAsBytes(img_in));
    mark(io, &t_mark, &prof.xfer_ns);

    // Record the whole forward into one submission; profiling stays
    // sync-per-op so the per-category host times remain exact.
    const batched = !profile;
    if (batched) try ctx.beginBatch();
    errdefer if (ctx.batching) ctx.abortBatch();
    // first: image patches -> x rows after the text tokens.
    try ctx.opMatmul(x_d, seq_txt * F * 4, imgin_d, 0, n_img, model.first.w.bytes, model.first.w.dtype == .f8_e4m3, F, dit.channels * dit.patch * dit.patch, model.first.w.scale, model.first.b);
        mark(io, &t_mark, &prof.matmul_ns);

    const sin_off: u32 = @intCast(seq * half);
    const Gemm = struct {
        fn go(c: *gpu.Context, use_coop: bool, y: gpu.DeviceBuffer, x: gpu.DeviceBuffer, m: usize, m_pad: usize, w_: anytype) !void {
            if (use_coop) {
                try c.opMatmulCoop(y, x, m, m_pad, w_.bytes, w_.rows, w_.cols, w_.scale);
            } else {
                try c.opMatmul(y, 0, x, 0, m, w_.bytes, true, w_.rows, w_.cols, w_.scale, null);
            }
        }
    };
    for (model.blocks, 0..) |*blk, b| {
        const mv_base: u32 = @intCast(b * 6 * F);

        // t1 = (1+pre_scale) * prenorm(x) + pre_shift — the norm weight is
        // prefolded into mv slot 0; inv-rms comes from the parallel
        // partial/combine pair (a one-thread-per-row loop over dim 6144 is
        // latency-bound).
        try ctx.opElt(.rms_partial, x_d, null, null, rmsp_d, .{
            .u0 = @intCast(seq * rms_ch),
            .u1 = F,
            .u2 = rms_ch,
        }, seq * rms_ch, 1, 1);
        try ctx.opElt(.rms_combine, rmsp_d, null, null, rmsi_d, .{
            .u0 = @intCast(seq),
            .u1 = F,
            .u2 = rms_ch,
            .f0 = 1e-5,
        }, seq, 1, 1);
        // Attention. When every consumer shares one dequant scale (the Krea
        // 2 DiT stores raw e4m3: all scales are 1), the modulated norm
        // converts straight to f16 once and feeds all four GEMMs — the f32
        // intermediate and three redundant conversions disappear.
        const qkv_shared = coop and
            blk.attn.wq.scale == blk.attn.wk.scale and
            blk.attn.wq.scale == blk.attn.wv.scale and
            blk.attn.wq.scale == blk.attn.gate.scale;
        if (qkv_shared) {
            try ctx.opElt(.rms_apply_mod_h16, x_d, h16_d, mv_d, rmsi_d, .{
                .u0 = @intCast(seq_pad * F / 2),
                .u1 = F,
                .u2 = mv_base + 0 * F,
                .u3 = mv_base + 1 * F,
                .u4 = @intCast(seq * F),
                .f0 = blk.attn.wq.scale,
            }, seq_pad * F / 2, 1, 1);
            mark(io, &t_mark, &prof.elt_ns);
            inline for (.{ .{ q_d, "wq" }, .{ k_d, "wk" }, .{ v_d, "wv" }, .{ g_d, "gate" } }) |v| {
                const w_ = @field(blk.attn, v[1]);
                try ctx.opMatmulCoopH16(v[0], h16_d, seq_pad, w_.bytes, w_.rows, w_.cols);
                mark(io, &t_mark, &prof.matmul_ns);
            }
        } else {
            try ctx.opElt(.rms_apply_mod, x_d, t1_d, mv_d, rmsi_d, .{
                .u0 = @intCast(seq * F),
                .u1 = F,
                .u2 = mv_base + 0 * F,
                .u3 = mv_base + 1 * F,
            }, seq * F, 1, 1);
            mark(io, &t_mark, &prof.elt_ns);
            try Gemm.go(ctx, coop, q_d, t1_d, seq, seq_pad, blk.attn.wq);
            mark(io, &t_mark, &prof.matmul_ns);
            try Gemm.go(ctx, coop, k_d, t1_d, seq, seq_pad, blk.attn.wk);
            mark(io, &t_mark, &prof.matmul_ns);
            try Gemm.go(ctx, coop, v_d, t1_d, seq, seq_pad, blk.attn.wv);
            mark(io, &t_mark, &prof.matmul_ns);
            try Gemm.go(ctx, coop, g_d, t1_d, seq, seq_pad, blk.attn.gate);
            mark(io, &t_mark, &prof.matmul_ns);
        }
        try ctx.opElt(.rmsnorm, q_d, q_d, try normBuf(ctx, blk.attn.qnorm), null, .{
            .u0 = @intCast(seq * heads),
            .u1 = hd,
            .f0 = 1e-5,
        }, seq * heads, 1, 1);
        mark(io, &t_mark, &prof.elt_ns);
        try ctx.opElt(.rmsnorm, k_d, k_d, try normBuf(ctx, blk.attn.knorm), null, .{
            .u0 = @intCast(seq * kv_heads),
            .u1 = hd,
            .f0 = 1e-5,
        }, seq * kv_heads, 1, 1);
        mark(io, &t_mark, &prof.elt_ns);
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
            try ctx.opElt(.f32_to_h16, q_d, null, null, qt_d, .{
                .u0 = @intCast(seq_pad * heads * hd / 2),
                .u1 = @intCast(seq * heads * hd),
                .f0 = attn_scale,
            }, seq_pad * heads * hd / 2, 1, 1);
            try ctx.opElt(.gather_kmajor_h16, k_d, null, null, kt_d, .{
                .u0 = @intCast(kv_heads * hd * seq_pad / 2),
                .u1 = hd,
                .u2 = @intCast(seq_pad),
                .u3 = @intCast(seq),
                .u4 = kv_heads,
            }, kv_heads * hd * seq_pad / 2, 1, 1);
            try ctx.opElt(.f32_to_h16, v_d, null, null, v16_d, .{
                .u0 = @intCast(seq_pad * kv_heads * hd / 2),
                .u1 = @intCast(seq * kv_heads * hd),
                .f0 = 1.0,
            }, seq_pad * kv_heads * hd / 2, 1, 1);
        } else {
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
            try ctx.opFlash(.md, qt_d, kt_d, v16_d, attn_d, push, seq_pad / 128, heads);
            mark(io, &t_mark, &prof.scores_ns);
            mark(io, &t_mark, &prof.smax_ns);
            try ctx.opFlash(.out, qt_d, kt_d, v16_d, attn_d, push, seq_pad / 128, heads);
            mark(io, &t_mark, &prof.aout_ns);
        } else {
            const s_stride: u32 = @intCast(s_rows);
            const s_plane: u32 = @intCast(s_rows * s_rows);
            var h0: usize = 0;
            while (h0 < heads) : (h0 += heads_per_batch) {
                const hb = @min(heads_per_batch, heads - h0);
                if (tc_attn) {
                    try ctx.opAttnScores(s_d, qt_d, kt_d, .{
                        .u0 = heads * hd,
                        .u1 = s_stride,
                        .u2 = @intCast(h0),
                        .u3 = heads / kv_heads,
                        .u4 = @intCast(hd * seq_pad),
                        .u5 = s_plane,
                    }, seq_pad / 128, seq_pad / 128, hb);
                } else {
                    try ctx.opElt(.attn_scores, qt_d, kt_d, null, s_d, .{
                        .u0 = @intCast(seq),
                        .u1 = heads,
                        .u2 = kv_heads,
                        .u3 = hd,
                        .u4 = @intCast(h0),
                        .f0 = attn_scale,
                    }, std.math.divCeil(usize, seq, 8) catch unreachable, std.math.divCeil(usize, seq, 8) catch unreachable, hb);
                }
                mark(io, &t_mark, &prof.scores_ns);
                if (tc_attn) {
                    try ctx.opElt(.softmax_partial, s_d, null, null, part_d, .{
                        .u0 = @intCast(hb * seq * nchunks),
                        .u1 = @intCast(nchunks),
                        .u2 = @intCast(seq),
                        .u3 = s_stride,
                        .u5 = s_plane,
                    }, hb * seq * nchunks, 1, 1);
                    try ctx.opElt(.softmax_combine, part_d, null, null, md_d, .{
                        .u0 = @intCast(hb * seq),
                        .u1 = @intCast(nchunks),
                        .u2 = @intCast(seq),
                        .u3 = s_stride,
                    }, hb * seq, 1, 1);
                    mark(io, &t_mark, &prof.smax_ns);
                    try ctx.opAttnOut(s_d, v16_d, attn_d, md_d, .{
                        .u0 = s_stride,
                        .u1 = s_plane,
                        .u2 = @intCast(h0),
                        .u3 = heads / kv_heads,
                        .u4 = kv_heads * hd,
                        .u5 = heads * hd,
                        .f0 = @bitCast(@as(u32, @intCast(seq))),
                        .f1 = @bitCast(s_stride), // MD rows per head plane
                    }, seq_pad / 128, hb);
                } else {
                    try ctx.opElt(.attn_out, s_d, null, v_d, attn_d, .{
                        .u0 = @intCast(seq),
                        .u1 = heads,
                        .u2 = kv_heads,
                        .u3 = hd,
                        .u4 = @intCast(h0),
                        .u5 = s_stride,
                        .f0 = @bitCast(s_plane),
                    }, hd / 8, std.math.divCeil(usize, seq, 8) catch unreachable, hb);
                }
                mark(io, &t_mark, &prof.aout_ns);
            }
        }
        mark(io, &t_mark, &prof.attn_ns);
        if (coop) {
            try ctx.opElt(.sigmoid_mul_h16, attn_d, g_d, null, h16_d, .{
                .u0 = @intCast(seq_pad * F / 2),
                .u1 = @intCast(seq * F),
                .f0 = blk.attn.wo.scale,
            }, seq_pad * F / 2, 1, 1);
            mark(io, &t_mark, &prof.elt_ns);
            try ctx.opMatmulCoopH16(t1_d, h16_d, seq_pad, blk.attn.wo.bytes, blk.attn.wo.rows, blk.attn.wo.cols);
        } else {
            try ctx.opElt(.sigmoid_mul, attn_d, g_d, null, null, .{ .u0 = @intCast(seq * F) }, seq * F, 1, 1);
            mark(io, &t_mark, &prof.elt_ns);
            try Gemm.go(ctx, coop, t1_d, attn_d, seq, seq_pad, blk.attn.wo);
        }
        mark(io, &t_mark, &prof.matmul_ns);
        try ctx.opElt(.gated_add, x_d, t1_d, mv_d, null, .{
            .u0 = @intCast(seq * F),
            .u1 = F,
            .u2 = mv_base + 2 * F,
        }, seq * F, 1, 1);
        mark(io, &t_mark, &prof.elt_ns);

        // MLP.
        try ctx.opElt(.rms_partial, x_d, null, null, rmsp_d, .{
            .u0 = @intCast(seq * rms_ch),
            .u1 = F,
            .u2 = rms_ch,
        }, seq * rms_ch, 1, 1);
        try ctx.opElt(.rms_combine, rmsp_d, null, null, rmsi_d, .{
            .u0 = @intCast(seq),
            .u1 = F,
            .u2 = rms_ch,
            .f0 = 1e-5,
        }, seq, 1, 1);
        const mlp_shared = coop and blk.mlp.gate.scale == blk.mlp.up.scale;
        if (mlp_shared) {
            try ctx.opElt(.rms_apply_mod_h16, x_d, h16_d, mv_d, rmsi_d, .{
                .u0 = @intCast(seq_pad * F / 2),
                .u1 = F,
                .u2 = mv_base + 3 * F,
                .u3 = mv_base + 4 * F,
                .u4 = @intCast(seq * F),
                .f0 = blk.mlp.gate.scale,
            }, seq_pad * F / 2, 1, 1);
            mark(io, &t_mark, &prof.elt_ns);
            try ctx.opMatmulCoopH16(mg_d, h16_d, seq_pad, blk.mlp.gate.bytes, blk.mlp.gate.rows, blk.mlp.gate.cols);
            mark(io, &t_mark, &prof.matmul_ns);
            try ctx.opMatmulCoopH16(mu_d, h16_d, seq_pad, blk.mlp.up.bytes, blk.mlp.up.rows, blk.mlp.up.cols);
            mark(io, &t_mark, &prof.matmul_ns);
        } else {
            try ctx.opElt(.rms_apply_mod, x_d, t1_d, mv_d, rmsi_d, .{
                .u0 = @intCast(seq * F),
                .u1 = F,
                .u2 = mv_base + 3 * F,
                .u3 = mv_base + 4 * F,
            }, seq * F, 1, 1);
            mark(io, &t_mark, &prof.elt_ns);
            try Gemm.go(ctx, coop, mg_d, t1_d, seq, seq_pad, blk.mlp.gate);
            mark(io, &t_mark, &prof.matmul_ns);
            try Gemm.go(ctx, coop, mu_d, t1_d, seq, seq_pad, blk.mlp.up);
            mark(io, &t_mark, &prof.matmul_ns);
        }
        if (coop) {
            try ctx.opElt(.silu_mul_h16, mg_d, mu_d, null, h16_d, .{
                .u0 = @intCast(seq_pad * dit.mlp_dim / 2),
                .u1 = @intCast(seq * dit.mlp_dim),
                .f0 = blk.mlp.down.scale,
            }, seq_pad * dit.mlp_dim / 2, 1, 1);
            mark(io, &t_mark, &prof.elt_ns);
            try ctx.opMatmulCoopH16(t1_d, h16_d, seq_pad, blk.mlp.down.bytes, blk.mlp.down.rows, blk.mlp.down.cols);
        } else {
            try ctx.opElt(.silu_mul, mg_d, mu_d, null, null, .{ .u0 = @intCast(seq * dit.mlp_dim) }, seq * dit.mlp_dim, 1, 1);
            mark(io, &t_mark, &prof.elt_ns);
            try Gemm.go(ctx, coop, t1_d, mg_d, seq, seq_pad, blk.mlp.down);
        }
        mark(io, &t_mark, &prof.matmul_ns);
        try ctx.opElt(.gated_add, x_d, t1_d, mv_d, null, .{
            .u0 = @intCast(seq * F),
            .u1 = F,
            .u2 = mv_base + 5 * F,
        }, seq * F, 1, 1);
        mark(io, &t_mark, &prof.elt_ns);
    }

    // Final layer on device: modulated rmsnorm then the 6144 -> 64 linear
    // (norm runs over all rows — the text-row waste is negligible next to
    // downloading 100 MB of hidden image rows for a CPU finalize).
    try ctx.opElt(.rms_partial, x_d, null, null, rmsp_d, .{
        .u0 = @intCast(seq * rms_ch),
        .u1 = F,
        .u2 = rms_ch,
    }, seq * rms_ch, 1, 1);
    try ctx.opElt(.rms_combine, rmsp_d, null, null, rmsi_d, .{
        .u0 = @intCast(seq),
        .u1 = F,
        .u2 = rms_ch,
        .f0 = 1e-5,
    }, seq, 1, 1);
    try ctx.opElt(.rms_apply_mod, x_d, t1_d, fin_d, rmsi_d, .{
        .u0 = @intCast(seq * F),
        .u1 = F,
        .u2 = 0,
        .u3 = F,
    }, seq * F, 1, 1);
    mark(io, &t_mark, &prof.elt_ns);
    // imgin_d's input role is long done; it is exactly n_img x 64.
    try ctx.opMatmul(imgin_d, 0, t1_d, seq_txt * F * 4, n_img, model.last_linear.w.bytes, model.last_linear.w.dtype == .f8_e4m3, dit.channels * dit.patch * dit.patch, F, model.last_linear.w.scale, model.last_linear.b);
    mark(io, &t_mark, &prof.matmul_ns);

    if (batched) try ctx.endBatch();

    const final_rows = try gpa.alloc(f32, n_img * dit.channels * dit.patch * dit.patch);
    defer gpa.free(final_rows);
    try ctx.tensorDownload(imgin_d, std.mem.sliceAsBytes(final_rows));
    mark(io, &t_mark, &prof.xfer_ns);
    DiT.unpatchify(out, final_rows, lat_h, lat_w);
    mark(io, &t_mark, &prof.cpu_ns);

    if (profile) {
        const ms = struct {
            fn go(ns: i96) f64 {
                return @as(f64, @floatFromInt(ns)) / 1e6;
            }
        }.go;
        std.debug.print(
            "dit gpu profile: matmul {d:.0}ms  attn {d:.0}ms (scores {d:.0} smax {d:.0} out {d:.0})  elt {d:.0}ms  xfer {d:.0}ms  cpu {d:.0}ms\n",
            .{ ms(prof.matmul_ns), ms(prof.attn_ns + prof.scores_ns + prof.smax_ns + prof.aout_ns), ms(prof.scores_ns), ms(prof.smax_ns), ms(prof.aout_ns), ms(prof.elt_ns), ms(prof.xfer_ns), ms(prof.cpu_ns) },
        );
    }
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
    const safetensors = @import("../safetensors.zig");
    std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    const dit_path = "models/diffusion_model/krea2CenterSemiraw_v10Fp8.safetensors";
    std.Io.Dir.cwd().access(io, dit_path, .{}) catch return error.SkipZigTest;
    std.Io.Dir.cwd().access(io, "testdata/dit_out.bin", .{}) catch return error.SkipZigTest;

    var ctx = gpu.Context.init(gpa) catch return error.SkipZigTest;
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
    var model = try DiT.load(gpa, &st);
    defer model.deinit();

    const out = try gpa.alloc(f32, dit.channels * 16 * 16);
    defer gpa.free(out);
    var sess = try Session.init(gpa, io, ctx, &model, 16, 16, cond, seq_txt, &.{0.875});
    defer sess.deinit(gpa, ctx);
    try forward(&model, ctx, &sess, io, gpa, out, x_lat, 0.875);

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
}
