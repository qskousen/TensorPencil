//! GPU-resident SD1.5 / SDXL UNet forward on the CUDA backends.
//!
//! The CUDA analogue of `sd_unet_gpu`, stage for stage, and it covers both
//! `zig-cuda` and `cuda`: they share this one code path and differ only in where
//! the batched GEMM and prefill attention go (hand-PTX mma against cuBLASLt /
//! cuDNN), which `Backend` decides internally.
//!
//! Two places it differs from the Vulkan path, both forced by the backend's shape:
//!
//! 1. `cuda` attends at the true head width; `zig-cuda` pads to 128. cuDNN's
//!    fused SDPA takes any head width, so the library arm runs SD's 40/64/80/160
//!    exactly. The hand-PTX arm cannot: its P@V GEMM tiles the head dimension in
//!    128-wide blocks, so narrower heads are zero-padded up to 128 (and 160 up to
//!    256) exactly as the Vulkan path does. Zero dimensions contribute nothing to
//!    a dot product and V's zero columns give output columns `opHeadUnpad` drops,
//!    so the padding is exact, it just costs arithmetic. This is the one place
//!    the two CUDA arms do not share a shape.
//!
//! 2. The ResBlock timestep projection is added, not folded. Vulkan folds it
//!    into the convolution bias for free; the CUDA GEMM entry points take a host
//!    bias slice, so folding would mean reworking that plumbing for a ~2% pass.
//!    See `opAddBiasRows`.
//!
//! Numerics are f16 tensor cores for the GEMMs and f32 elsewhere, so this is not
//! bit-identical to `sd_unet.forward`; the CPU path remains the reference.

const std = @import("std");
const sd_unet = @import("sd_unet.zig");
const ops = @import("tp_ops");
const cuda = @import("tp_gpu").cuda;

const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;
const Weight = ops.matmul.Weight;
const Conv2d = ops.conv.Conv2d;
const Config = sd_unet.Config;

/// Cap on the im2col patch band (bytes); bands iterate over output rows.
const patch_band_bytes: usize = 256 << 20;

/// Column groups per GroupNorm statistics pass (must match the Vulkan path's, so
/// the two produce the same partition of the reduction).
const gn_chunks: usize = 256;

/// Convolutions at least this wide take the f16 tensor-core GEMM; below it the
/// f32 GEMM wins, because padding `co` out to the cooperative tile costs more
/// than the tensor cores return.
const coop_min_co: usize = 96;

/// The head width the hand-PTX attention tiles its P@V GEMM in.
///
/// `launchHgemmB` launches `grid.x = n / 128` with `n = head_dim`, so a
/// narrower head asks for a zero-sized grid (`CUDA_ERROR_INVALID_VALUE`) and a
/// non-multiple silently computes only part of the head. Every SD head width
/// trips one or the other: 40, 64 and 80 the former, 160 the latter.
const pv_tile: usize = 128;

/// Shared with the Vulkan path so the two `im2col_sd` kernels cannot disagree
/// about the encoding of their sampling mode.
pub const SampleMode = @import("sd_unet_gpu.zig").SampleMode;

/// Per-conditioning state: the uploaded context, and the per-forward ResBlock
/// timestep projections packed into one device buffer.
pub const Session = struct {
    arena: std.heap.ArenaAllocator,
    ctx_d: Buf = .{},
    ctx_seq: usize,
    adm: ?[]const f32,

    /// Every ResBlock's timestep projection, packed end to end so one upload
    /// serves the whole forward.
    proj_host: []f32,
    proj_d: Buf = .{},
    /// Element offset per ResBlock, indexed by its ordinal in the graph walk.
    /// Not by a host pointer: `Session.replaceDenoiser` reloads the UNet into a
    /// fresh arena while this session stays alive, so pointer keys go stale. See
    /// the longer note on `sd_unet_gpu.Session.bias_off`.
    proj_off: []usize,

    emb: []f32,
    mlp_hidden: []f32,
    mlp_scratch: []f32,

    /// GroupNorm weight ++ bias concatenations, cached by weight pointer, plus
    /// their device buffers: `gn_apply` reads both out of one binding.
    norm_d: std.AutoHashMapUnmanaged(usize, Buf) = .empty,

    /// Stable zero vector for the GEMMs whose weights carry no bias.
    zeros: []f32,

    pub fn init(
        gpa: std.mem.Allocator,
        be: *Backend,
        u: *const sd_unet.UNet,
        context: []const f32,
        ctx_seq: usize,
        adm: ?[]const f32,
    ) !Session {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();
        const cfg = u.cfg;

        var offs: std.ArrayList(usize) = .empty;
        var total: usize = 0;
        {
            var it = sd_unet.ResBlockIter.init(u);
            while (it.next()) |rb| {
                try offs.append(alloc, total);
                total += rb.out_ch;
            }
        }

        var max_ch: usize = cfg.model_channels;
        for (cfg.channel_mult) |m| max_ch = @max(max_ch, cfg.model_channels * m);

        // Allocate BEFORE the literal: `.arena = arena` copies the arena's
        // state, so a later field initializer's allocation leaks. See the same
        // note in `sd_unet_gpu.Session.init`.
        const proj_host = try alloc.alloc(f32, total);
        const emb = try alloc.alloc(f32, cfg.time_embed_dim);
        const mlp_hidden = try alloc.alloc(f32, cfg.time_embed_dim);
        const mlp_scratch = try alloc.alloc(f32, cfg.time_embed_dim);
        const zeros = try alloc.alloc(f32, max_ch * 8);
        @memset(zeros, 0);

        var self: Session = .{
            .arena = arena,
            .ctx_seq = ctx_seq,
            .adm = adm,
            .proj_host = proj_host,
            .proj_off = offs.items,
            .emb = emb,
            .mlp_hidden = mlp_hidden,
            .mlp_scratch = mlp_scratch,
            .zeros = zeros,
        };

        try be.ensureDeviceBuffer(&self.ctx_d, context.len * 4);
        try be.tensorUpload(self.ctx_d, std.mem.sliceAsBytes(context));
        try be.ensureDeviceBuffer(&self.proj_d, @max(total, 1) * 4);
        return self;
    }

    pub fn deinit(self: *Session, be: *Backend) void {
        be.tensorDestroy(&self.ctx_d);
        be.tensorDestroy(&self.proj_d);
        var it = self.norm_d.valueIterator();
        while (it.next()) |b| be.tensorDestroy(b);
        self.arena.deinit();
        self.* = undefined;
    }

    /// The weight ++ bias concatenation for a GroupNorm, uploaded once.
    fn normBuf(self: *Session, be: *Backend, nw: sd_unet.GroupNormW) !Buf {
        const key = @intFromPtr(nw.w.ptr);
        if (self.norm_d.get(key)) |b| return b;
        const alloc = self.arena.allocator();
        const cat = try alloc.alloc(f32, nw.w.len + nw.b.len);
        @memcpy(cat[0..nw.w.len], nw.w);
        @memcpy(cat[nw.w.len..], nw.b);
        var b: Buf = .{};
        try be.ensureDeviceBuffer(&b, cat.len * 4);
        try be.tensorUpload(b, std.mem.sliceAsBytes(cat));
        try self.norm_d.put(alloc, key, b);
        return b;
    }
};

/// Device scratch for one image size, shared by both conditioning branches.
pub const Workspace = struct {
    cur: Buf = .{},
    alt: Buf = .{},
    t1: Buf = .{},
    t2: Buf = .{},
    patch: Buf = .{},
    skips: []Buf,
    skip_ch: []usize,
    stream: Buf = .{},
    nb: Buf = .{},
    q: Buf = .{},
    k: Buf = .{},
    v: Buf = .{},
    ao: Buf = .{},
    /// Head-padded attention operands, used only where `hd % 16 != 0`.
    qp: Buf = .{},
    kp: Buf = .{},
    vp: Buf = .{},
    op: Buf = .{},
    ff: Buf = .{},
    gated: Buf = .{},
    gstat: Buf = .{},
    gmi: Buf = .{},
    xin: Buf = .{},
    eps: Buf = .{},
    gpa: std.mem.Allocator,

    pub fn init(
        gpa: std.mem.Allocator,
        be: *Backend,
        u: *const sd_unet.UNet,
        lat_h: usize,
        lat_w: usize,
        ctx_seq: usize,
    ) !Workspace {
        const cfg = u.cfg;
        var self: Workspace = .{
            .gpa = gpa,
            .skips = try gpa.alloc(Buf, u.input_stages.len + 1),
            .skip_ch = try gpa.alloc(usize, u.input_stages.len + 1),
        };
        errdefer {
            gpa.free(self.skips);
            gpa.free(self.skip_ch);
        }
        for (self.skips) |*s| s.* = .{};
        @memset(self.skip_ch, 0);

        var act: usize = 0;
        var attn: usize = 0;
        var ff: usize = 0;
        var gated: usize = 0;
        var padded: usize = 0;
        var patch: usize = 0;
        var h = lat_h;
        var w = lat_w;
        for (cfg.channel_mult, 0..) |mult, level| {
            const ch = cfg.model_channels * mult;
            act = @max(act, h * w * ch * 3);
            const plen = 9 * ch * 3;
            patch = @max(patch, convBand(h * w, plen) * plen * 4);
            if (cfg.attn_levels[level]) sizeAttn(be, cfg, h * w, ch, ctx_seq, &attn, &ff, &gated, &padded);
            if (level + 1 < cfg.levels()) {
                h = (h + 1) / 2;
                w = (w + 1) / 2;
            }
        }
        sizeAttn(be, cfg, h * w, cfg.model_channels * cfg.channel_mult[cfg.levels() - 1], ctx_seq, &attn, &ff, &gated, &padded);

        inline for (.{ "cur", "alt", "t1", "t2" }) |f| try be.ensureDeviceBuffer(&@field(self, f), act * 4);
        inline for (.{ "stream", "nb", "q", "k", "v", "ao" }) |f| try be.ensureDeviceBuffer(&@field(self, f), attn * 4);
        inline for (.{ "qp", "kp", "vp", "op" }) |f| try be.ensureDeviceBuffer(&@field(self, f), padded);
        try be.ensureDeviceBuffer(&self.ff, ff * 4);
        try be.ensureDeviceBuffer(&self.gated, gated * 4);
        try be.ensureDeviceBuffer(&self.patch, patch);
        try be.ensureDeviceBuffer(&self.gstat, cfg.norm_groups * gn_chunks * 3 * 4);
        try be.ensureDeviceBuffer(&self.gmi, cfg.norm_groups * 2 * 4);
        try be.ensureDeviceBuffer(&self.xin, lat_h * lat_w * cfg.channels * 4);
        try be.ensureDeviceBuffer(&self.eps, lat_h * lat_w * cfg.channels * 4);
        return self;
    }

    fn sizeAttn(be: *const Backend, cfg: Config, n: usize, ch: usize, ctx_seq: usize, attn: *usize, ff: *usize, gated: *usize, padded: *usize) void {
        attn.* = @max(attn.*, @max(n, ctx_seq) * ch);
        ff.* = @max(ff.*, n * ch * 8);
        gated.* = @max(gated.*, n * ch * 4);
        const heads = cfg.headsAt(ch);
        const hd = ch / heads;
        const hp = attnHd(be, hd);
        if (hp != hd) padded.* = @max(padded.*, n * heads * hp * 4);
    }

    pub fn deinit(self: *Workspace, be: *Backend) void {
        inline for (@typeInfo(Workspace).@"struct".fields) |f| {
            if (f.type == Buf) be.tensorDestroy(&@field(self, f.name));
        }
        for (self.skips) |*s| be.tensorDestroy(s);
        self.gpa.free(self.skips);
        self.gpa.free(self.skip_ch);
        self.* = undefined;
    }
};

fn convBand(n_out: usize, patch_len: usize) usize {
    return @max(4, @min(n_out, patch_band_bytes / (patch_len * 4)) & ~@as(usize, 3));
}

/// eps = UNet(x, t, context, y). `x` and `out` are channel-last host buffers.
pub fn forward(
    u: *const sd_unet.UNet,
    be: *Backend,
    sess: *Session,
    ws: *Workspace,
    io: std.Io,
    gpa: std.mem.Allocator,
    out: []f32,
    x: []const f32,
    lat_h: usize,
    lat_w: usize,
    timestep: f32,
    cancel: ?*std.atomic.Value(bool),
) !void {
    const cfg = u.cfg;
    std.debug.assert(x.len == lat_h * lat_w * cfg.channels);
    std.debug.assert(out.len == x.len);
    if ((cfg.adm_channels != null) != (sess.adm != null)) return error.MissingAdmConditioning;

    try embedAndProject(u, be, sess, io, gpa, timestep);
    try be.tensorUpload(ws.xin, std.mem.sliceAsBytes(x));

    try be.beginBatch();
    errdefer if (be.batching()) be.abortBatch();

    var h = lat_h;
    var w = lat_w;
    var ch = cfg.model_channels;
    // ResBlocks in `ResBlockIter`'s order, which is the order below.
    var rb_ord: usize = 0;

    try conv(be, ws, &ws.cur, &ws.xin, h, w, u.stem, .stride1);
    try setSkip(be, ws, 0, &ws.cur, h * w * ch, ch);

    for (u.input_stages, 0..) |stage, si| {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        if (stage.res) |rb| {
            try applyRes(be, sess, ws, rb, rb_ord, &ws.cur, &ws.alt, h, w, cfg);
            rb_ord += 1;
            std.mem.swap(Buf, &ws.cur, &ws.alt);
            ch = rb.out_ch;
        }
        if (stage.attn) |st| try applySpatial(be, sess, ws, st, h, w, cfg);
        if (stage.sample_kind == .down) {
            try conv(be, ws, &ws.alt, &ws.cur, h, w, stage.sample.?, .stride2);
            std.mem.swap(Buf, &ws.cur, &ws.alt);
            h = (h + 1) / 2;
            w = (w + 1) / 2;
        }
        try setSkip(be, ws, si + 1, &ws.cur, h * w * ch, ch);
    }

    try applyRes(be, sess, ws, u.mid_res1, rb_ord, &ws.cur, &ws.alt, h, w, cfg);
    rb_ord += 1;
    std.mem.swap(Buf, &ws.cur, &ws.alt);
    ch = u.mid_res1.out_ch;
    try applySpatial(be, sess, ws, u.mid_attn, h, w, cfg);
    try applyRes(be, sess, ws, u.mid_res2, rb_ord, &ws.cur, &ws.alt, h, w, cfg);
    rb_ord += 1;
    std.mem.swap(Buf, &ws.cur, &ws.alt);
    ch = u.mid_res2.out_ch;

    var skip_top = u.input_stages.len + 1;
    for (u.output_stages) |stage| {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        skip_top -= 1;
        const skip_ch = ws.skip_ch[skip_top];
        {
            const total = ch + skip_ch;
            try be.opConcatCh(ws.cur, ws.alt, h * w, ch, total, 0);
            try be.opConcatCh(ws.skips[skip_top], ws.alt, h * w, skip_ch, total, ch);
            std.mem.swap(Buf, &ws.cur, &ws.alt);
            ch = total;
        }
        const rb = stage.res.?;
        try applyRes(be, sess, ws, rb, rb_ord, &ws.cur, &ws.alt, h, w, cfg);
        rb_ord += 1;
        std.mem.swap(Buf, &ws.cur, &ws.alt);
        ch = rb.out_ch;
        if (stage.attn) |st| try applySpatial(be, sess, ws, st, h, w, cfg);
        if (stage.sample_kind == .up) {
            try conv(be, ws, &ws.alt, &ws.cur, h, w, stage.sample.?, .upsample2x);
            std.mem.swap(Buf, &ws.cur, &ws.alt);
            h *= 2;
            w *= 2;
        }
    }

    try groupNorm(be, sess, ws, &ws.t1, &ws.cur, h * w, ch, u.out_norm, cfg, true);
    try conv(be, ws, &ws.eps, &ws.t1, h, w, u.out_conv, .stride1);

    try be.endBatch();
    try be.tensorDownload(ws.eps, std.mem.sliceAsBytes(out));
}

/// The time embedding (plus SDXL's label embedding) and every ResBlock's
/// `emb_layers` projection, on the host: the widest is a 1280x1280 GEMV against
/// the forward's hundreds of GFLOP.
fn embedAndProject(
    u: *const sd_unet.UNet,
    be: *Backend,
    sess: *Session,
    io: std.Io,
    gpa: std.mem.Allocator,
    timestep: f32,
) !void {
    const cfg = u.cfg;
    const sin = try gpa.alloc(f32, cfg.model_channels);
    defer gpa.free(sin);
    sd_unet.timestepEmbedding(sin, timestep);
    try ops.matmul.matmul(io, gpa, sess.mlp_hidden, sin, 1, u.time_1.w, u.time_1.b);
    ops.act.silu(sess.mlp_hidden);
    try ops.matmul.matmul(io, gpa, sess.emb, sess.mlp_hidden, 1, u.time_2.w, u.time_2.b);
    if (sess.adm) |y| {
        try ops.matmul.matmul(io, gpa, sess.mlp_hidden, y, 1, u.label_1.?.w, u.label_1.?.b);
        ops.act.silu(sess.mlp_hidden);
        try ops.matmul.matmul(io, gpa, sess.mlp_scratch, sess.mlp_hidden, 1, u.label_2.?.w, u.label_2.?.b);
        for (sess.emb, sess.mlp_scratch) |*e, v| e.* += v;
    }

    // emb_layers is [SiLU, Linear] over the SHARED embedding, so the silu runs
    // once here rather than per block.
    @memcpy(sess.mlp_hidden, sess.emb);
    ops.act.silu(sess.mlp_hidden);
    var it = sd_unet.ResBlockIter.init(u);
    var bi: usize = 0;
    while (it.next()) |rb| : (bi += 1) {
        const dst = sess.proj_host[sess.proj_off[bi]..][0..rb.out_ch];
        try ops.matmul.matmul(io, gpa, dst, sess.mlp_hidden, 1, rb.emb.w, rb.emb.b);
    }
    if (sess.proj_host.len > 0) {
        try be.tensorUpload(sess.proj_d, std.mem.sliceAsBytes(sess.proj_host));
    }
}

fn setSkip(be: *Backend, ws: *Workspace, i: usize, src: *const Buf, elems: usize, ch: usize) !void {
    try be.ensureDeviceBuffer(&ws.skips[i], elems * 4);
    ws.skip_ch[i] = ch;
    try be.opCopyOff(ws.skips[i], 0, src.*, 0, elems);
}

/// A ResBlock, result guaranteed in `dst`.
fn applyRes(
    be: *Backend,
    sess: *Session,
    ws: *Workspace,
    rb: sd_unet.ResBlock,
    /// The block's ordinal in `sd_unet.ResBlockIter`'s order. See `Session.proj_off`.
    ordinal: usize,
    src: *Buf,
    dst: *Buf,
    h: usize,
    w: usize,
    cfg: Config,
) !void {
    const n = h * w;
    const out_n = n * rb.out_ch;
    const off = sess.proj_off[ordinal];

    try groupNorm(be, sess, ws, &ws.t1, src, n, rb.in_ch, rb.in_norm, cfg, true);
    try conv(be, ws, dst, &ws.t1, h, w, rb.in_conv, .stride1);
    try be.opAddBiasRows(dst.*, sess.proj_d, n, rb.out_ch, off);

    try groupNorm(be, sess, ws, &ws.t1, dst, n, rb.out_ch, rb.out_norm, cfg, true);
    try conv(be, ws, &ws.t2, &ws.t1, h, w, rb.out_conv, .stride1);

    if (rb.skip) |sk| {
        try conv(be, ws, dst, src, h, w, sk, .stride1);
        try be.opAdd(dst.*, ws.t2, out_n);
    } else {
        try be.opAdd(ws.t2, src.*, out_n);
        std.mem.swap(Buf, &ws.t2, dst);
    }
}

/// A SpatialTransformer, in place on `ws.cur`.
fn applySpatial(
    be: *Backend,
    sess: *Session,
    ws: *Workspace,
    st: sd_unet.SpatialTransformer,
    h: usize,
    w: usize,
    cfg: Config,
) !void {
    const n = h * w;
    const ch = st.channels;
    const heads = cfg.headsAt(ch);
    const hd = ch / heads;
    const ctx_seq = sess.ctx_seq;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));

    try groupNorm(be, sess, ws, &ws.nb, &ws.cur, n, ch, st.norm, cfg, false);
    try applyProj(be, ws, sess, &ws.stream, &ws.nb, h, w, st.proj_in);

    for (st.blocks) |b| {
        // attn1: self-attention over pixels.
        try be.opLayerNorm(ws.stream, ws.nb, b.norm1.w, b.norm1.b, n, ch, cfg.norm_eps);
        try gemm(be, sess, &ws.q, &ws.nb, n, b.attn1.q, null);
        try gemm(be, sess, &ws.k, &ws.nb, n, b.attn1.k, null);
        try gemm(be, sess, &ws.v, &ws.nb, n, b.attn1.v, null);
        try selfAttn(be, ws, n, heads, hd, scale);
        try gemm(be, sess, &ws.nb, &ws.ao, n, b.attn1.out.w, b.attn1.out.b);
        try be.opAdd(ws.stream, ws.nb, n * ch);

        // attn2: cross-attention onto the text conditioning, `ctx_seq` keys, not `n`.
        try be.opLayerNorm(ws.stream, ws.nb, b.norm2.w, b.norm2.b, n, ch, cfg.norm_eps);
        try gemm(be, sess, &ws.q, &ws.nb, n, b.attn2.q, null);
        try gemm(be, sess, &ws.k, &sess.ctx_d, ctx_seq, b.attn2.k, null);
        try gemm(be, sess, &ws.v, &sess.ctx_d, ctx_seq, b.attn2.v, null);
        try be.opAttnCross(ws.q, ws.k, ws.v, ws.ao, n, ctx_seq, heads, hd, scale);
        try gemm(be, sess, &ws.nb, &ws.ao, n, b.attn2.out.w, b.attn2.out.b);
        try be.opAdd(ws.stream, ws.nb, n * ch);

        // ff: GEGLU, whose halves are (value, gate) and whose gate is erf-GELU.
        try be.opLayerNorm(ws.stream, ws.nb, b.norm3.w, b.norm3.b, n, ch, cfg.norm_eps);
        try gemm(be, sess, &ws.ff, &ws.nb, n, b.ff_proj.w, b.ff_proj.b);
        try be.opGeglu(ws.ff, ws.gated, n, b.inner);
        try gemm(be, sess, &ws.nb, &ws.gated, n, b.ff_out.w, b.ff_out.b);
        try be.opAdd(ws.stream, ws.nb, n * ch);
    }

    try applyProj(be, ws, sess, &ws.nb, &ws.stream, h, w, st.proj_out);
    try be.opAdd(ws.cur, ws.nb, n * ch);
}

/// The head width the attention actually runs at: the true one under cuDNN, the
/// next multiple of the P@V tile under hand-PTX. See `pv_tile`.
pub fn attnHd(be: *const Backend, hd: usize) usize {
    if (be.kernels == .libs) return hd;
    return std.mem.alignForward(usize, hd, pv_tile);
}

/// Self-attention over the latent positions, padding the head width when the
/// backend's P@V GEMM needs it (`attnHd`).
///
/// `pub` for `sd-cuda-test`: the hand-PTX kernels have no unit test (the test
/// binary brings up no CUDA context), so the validation command drives them
/// directly rather than only through a whole render.
pub fn selfAttn(be: *Backend, ws: *Workspace, n: usize, heads: usize, hd: usize, scale: f32) !void {
    const hp = attnHd(be, hd);
    if (hp == hd) {
        return be.opAttnTC(ws.q, ws.k, ws.v, ws.ao, n, heads, heads, hd, scale);
    }
    const stride = heads * hd;
    inline for (.{ .{ "q", "qp" }, .{ "k", "kp" }, .{ "v", "vp" } }) |pair| {
        try be.opHeadPad(@field(ws, pair[1]), @field(ws, pair[0]), n, heads, hp, hd, stride, 0);
    }
    try be.opAttnTC(ws.qp, ws.kp, ws.vp, ws.op, n, heads, heads, hp, scale);
    try be.opHeadUnpad(ws.op, ws.ao, n, heads, hd, hp);
}

/// A SpatialTransformer's in/out projection: a 1x1 convolution in SD1.5, an
/// `nn.Linear` in SDXL. On channel-last activations both are the same GEMM.
fn applyProj(be: *Backend, ws: *Workspace, sess: *Session, dst: *Buf, src: *const Buf, h: usize, w: usize, p: sd_unet.Proj) !void {
    switch (p) {
        .conv => |cv| try conv(be, ws, dst, src, h, w, cv, .stride1),
        .linear => |lin| try gemm(be, sess, dst, src, h * w, lin.w, lin.b),
    }
}

fn groupNorm(
    be: *Backend,
    sess: *Session,
    ws: *Workspace,
    dst: *Buf,
    src: *const Buf,
    n: usize,
    ch: usize,
    nw: sd_unet.GroupNormW,
    cfg: Config,
    silu: bool,
) !void {
    const cat = try sess.normBuf(be, nw);
    try be.opGroupNorm(src.*, dst.*, cat, ws.gstat, ws.gmi, n, ch, cfg.norm_groups, gn_chunks, cfg.norm_eps, silu, false);
}

/// A GEMM against a checkpoint weight, routed by the dtype it was stored in.

/// Feed `ops.matmul.probe` this GEMM's input, so an activation capture works when
/// the SD UNet runs on CUDA.
///
/// Without these call sites a CUDA capture of an SD model is silently almost
/// empty, and it passes the cache's own sanity gate. This backend owns its GEMM,
/// so `ops.matmul`'s probe point never fires here. Without them a capture records only
/// the `emb_layers` timestep projections, the sole linears that fall back to
/// `ops.matmul`: 23 of ~282 layers, with attention, the feed-forwards and every
/// convolution invisible.
///
/// Mirrors `dit_cuda.probeInput`: the activation is downloaded and handed to the same
/// host accumulator a CPU capture uses, so a GPU-captured cache differs from a CPU one
/// only by the backend's own GEMM arithmetic.
fn probeInput(be: *Backend, x: Buf, m: usize, w: Weight) !void {
    const p = ops.matmul.probe orelse return;
    if (m == 0 or w.cols == 0) return;
    const host = try be.gpa.alloc(f32, m * w.cols);
    defer be.gpa.free(host);
    try be.tensorDownload(x, std.mem.sliceAsBytes(host));
    p.input(p.ctx, w, host, m);
}

/// The `Weight` a convolution's GEMM is equivalent to, matching `ops.conv` exactly:
/// a 1x1 reads `ci` columns and a 3x3 reads the im2col patch (`9*ci`). Getting this
/// wrong would not fail, it would file a cache whose column count disagrees with the
/// checkpoint, which is what the validator catches, or worse, silently mis-shape the
/// per-column statistics an imatrix is built from.
fn convWeight(cv: Conv2d, cols: usize) Weight {
    var w = Weight.fromF32(cv.w, cv.co, cols);
    w.tag = cv.tag;
    return w;
}

fn gemm(be: *Backend, sess: *Session, y: *Buf, x: *const Buf, m: usize, wt: Weight, bias: ?[]const f32) !void {
    const rows = wt.rows;
    const cols = wt.cols;
    // The whole `zeros` array, NOT `zeros[0..rows]`. Both backends cache a bias
    // by POINTER and size the device buffer from the FIRST call's length, so a
    // narrow layer seen first would leave every wider one reading past the end,
    // which robust-buffer-access hides by returning zeros, i.e. exactly the right
    // answer for THIS bias and nothing else. The kernels read only `rows`
    // entries, so handing them a longer slice is free. (Backend.cachedWeight)
    const b = bias orelse sess.zeros;
    // Before the dispatch: `x` still holds the f32 activation here (unlike the
    // int8/int4 DiT paths, which consume it in place, the SD UNet has no such path,
    // its GEMM switch is {f32, f16, bf16}).
    try probeInput(be, x.*, m, wt);
    switch (wt.dtype) {
        .f32 => {
            if (rows >= coop_min_co) return be.opConvF16(y.*, 0, x.*, m, wt.bytes, rows, cols, b);
            return be.opMatmul(y.*, 0, x.*, 0, m, wt.bytes, false, rows, cols, 1.0, b);
        },
        .f16 => return be.opMatmulF16(y.*, x.*, m, wt.bytes, rows, cols, b),
        .bf16 => return be.opMatmulBf16(y.*, x.*, m, wt.bytes, rows, cols, b),
        else => return error.UnsupportedDType,
    }
}

/// A convolution: 1x1 is a plain GEMM over pixels, 3x3 a banded `im2col_sd` plus
/// the same GEMM.
fn conv(
    be: *Backend,
    ws: *Workspace,
    dst: *Buf,
    src: *const Buf,
    h: usize,
    w: usize,
    cv: Conv2d,
    mode: SampleMode,
) !void {
    return convInto(be, &ws.patch, dst, src, h, w, cv, mode);
}

/// `conv` with the im2col band buffer passed in, so `sd_vae_cuda` shares this
/// exact mapping rather than reimplementing it. Grows `dst` as needed, the VAE
/// decoder's activations change width per level.
pub fn convInto(
    be: *Backend,
    patch: *Buf,
    dst: *Buf,
    src: *const Buf,
    h: usize,
    w: usize,
    cv: Conv2d,
    mode: SampleMode,
) !void {
    return convIntoScaled(be, patch, dst, src, h, w, cv, mode, 1.0);
}

/// `convInto` with the activation divided by `act_div` before the f16 cast (and
/// the result multiplied back), see `Backend.opConvF16Scaled` for why f16's 65504
/// ceiling is reachable in practice. `act_div = 1.0` is exactly `convInto`.
///
/// The f32 arm (`co < coop_min_co`) ignores it: it never casts to f16, so it has
/// f32's range already.
pub fn convIntoScaled(
    be: *Backend,
    patch: *Buf,
    dst: *Buf,
    src: *const Buf,
    h: usize,
    w: usize,
    cv: Conv2d,
    mode: SampleMode,
    act_div: f32,
) !void {
    return convIntoPrec(be, patch, dst, src, h, w, cv, mode, act_div, false, false);
}

/// `convIntoScaled` with f16 activation STORAGE selectable on either side. The GEMM
/// is unchanged, it already ran f16 operands with an f32 accumulator; `src_f16` /
/// `dst_f16` only say how the big activation buffers are laid out in memory, which
/// is where a VAE decode's gigabytes live.
///
/// For a 3x3 the patch stays f32 (it is banded, a few MB), so `src_f16` is
/// consumed by the im2col gather and the GEMM's own source is f32 either way.
///
/// `src_f16` and `act_div != 1` are MUTUALLY EXCLUSIVE, and that is a property of
/// the problem rather than a limitation: `act_div` exists because a value too large
/// for f16 has to be scaled down before the cast, and an f16 *buffer* could not have
/// held that value in the first place. `sd_vae.Config.act_f16` is off exactly where
/// `act_div` is load-bearing (SDXL).
pub fn convIntoPrec(
    be: *Backend,
    patch: *Buf,
    dst: *Buf,
    src: *const Buf,
    h: usize,
    w: usize,
    cv: Conv2d,
    mode: SampleMode,
    act_div: f32,
    src_f16: bool,
    dst_f16: bool,
) !void {
    std.debug.assert(!(src_f16 and act_div != 1.0));
    const cb = cv.b orelse return error.MissingConvBias;
    // f16 storage FORCES the cooperative path, because the f32 GEMM arm has no
    // f16 form. `coop_min_co` is a performance threshold, not a correctness one
    // (padding `co` to the tile costs more than the tensor cores return below it),
    // so widening it here is free of consequence, and not widening it was a real
    // bug: SD1.5's `post_quant_conv` is 4->4, fell to the f32 arm, and wrote f32
    // into an f16 buffer. Every pixel non-finite, caught by `sd-cuda-test`'s sweep.
    const coop = cv.co >= coop_min_co or src_f16 or dst_f16;
    if (cv.k == 1) {
        std.debug.assert(mode == .stride1);
        const n = h * w;
        // The activation-capture probe reads f32; an f16 buffer would be filed as
        // noise, so it is skipped rather than lied to (capture and f16 storage are
        // not used together, one is a measurement run, the other a render).
        if (!src_f16) try probeInput(be, src.*, n, convWeight(cv, cv.ci));
        if (coop) return be.opConvF16Prec(dst.*, 0, src.*, n, std.mem.sliceAsBytes(cv.w), cv.co, cv.ci, cb, act_div, src_f16, dst_f16);
        std.debug.assert(!src_f16 and !dst_f16); // unreachable: `coop` covers those
        return be.opMatmul(dst.*, 0, src.*, 0, n, std.mem.sliceAsBytes(cv.w), false, cv.co, cv.ci, 1.0, cb);
    }
    std.debug.assert(cv.k == 3);

    const oh = switch (mode) {
        .stride1 => h,
        .upsample2x => 2 * h,
        .stride2 => (h + 1) / 2,
    };
    const ow = switch (mode) {
        .stride1 => w,
        .upsample2x => 2 * w,
        .stride2 => (w + 1) / 2,
    };
    const n_out = oh * ow;
    const patch_len = 9 * cv.ci;
    const band = convBand(n_out, patch_len);
    try be.ensureDeviceBuffer(patch, band * patch_len * 4);

    var p0: usize = 0;
    while (p0 < n_out) : (p0 += band) {
        const bn = @min(band, n_out - p0);
        try be.opIm2colSd(src.*, patch.*, bn, patch_len, cv.ci, w, h, p0, ow, @intFromEnum(mode), src_f16);
        // Per band, not per convolution: the accumulator sums over rows, so N banded
        // calls contribute exactly the rows one unbanded call would.
        try probeInput(be, patch.*, bn, convWeight(cv, patch_len));
        if (coop) {
            // The patch is f32 whatever `src_f16` said, so only `dst_f16` applies here.
            try be.opConvF16Prec(dst.*, p0 * cv.co, patch.*, bn, std.mem.sliceAsBytes(cv.w), cv.co, patch_len, cb, act_div, false, dst_f16);
        } else {
            std.debug.assert(!dst_f16);
            try be.opMatmul(dst.*, p0 * cv.co * 4, patch.*, 0, bn, std.mem.sliceAsBytes(cv.w), false, cv.co, patch_len, 1.0, cb);
        }
    }
}
