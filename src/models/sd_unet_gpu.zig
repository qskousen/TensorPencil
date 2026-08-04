//! GPU-resident SD1.5 / SDXL UNet forward (Vulkan).
//!
//! Mirrors `sd_unet.forward` with everything from the latent upload to the eps
//! download on the device. The mapping is the same one `vae_gpu` uses for the Wan
//! decoder — activations stay tight channel-last `[h*w][c]` f32, a 3x3
//! convolution is a banded `im2col_sd` followed by a GEMM, a 1x1 convolution and
//! an `nn.Linear` are the same GEMM with no patch step — plus the three things a
//! UNet has that a VAE decoder does not: GroupNorm over 32 channel groups, a
//! SpatialTransformer (self-attention, cross-attention onto the text
//! conditioning, GEGLU feed-forward), and the LIFO skip stack.
//!
//! Weights are read straight from a loaded `sd_unet.UNet`, in whatever dtype the
//! checkpoint stored them (f32 and f16 are both common in the wild, and ggufy
//! emits block-quant GGUF UNets), and the Context's weight cache uploads and
//! transposes each one lazily on first use.
//!
//! Three things here are worth knowing before changing any of it:
//!
//! 1. **Each ResBlock's timestep-embedding projection is folded into its first
//!    convolution's bias**, on the host, rather than added in a pass of its own.
//!    A conv bias is already a per-channel constant over positions, which is
//!    exactly what the projection is, so this is exact — and it removes one
//!    read-modify-write of the full activation per ResBlock. The folded vector
//!    changes every forward, hence `opMatmulCoopF16WDev` and one packed device
//!    buffer instead of the pointer-cached `smallBuffer` path.
//!
//! 2. **Self-attention pads head_dim up to 128.** The cooperative-matrix
//!    attention pipelines are compiled for head_dim 128 (`coopmat.buildFlashAttn`
//!    and `buildGemmAttnOut` both assume it), while this family's heads are 40
//!    (SD1.5's outermost level), 64 (SDXL everywhere) or 80. Zero-padding is
//!    exact — a zero dimension contributes nothing to a dot product and V's zero
//!    columns produce output columns we drop — and it buys the flash kernel,
//!    which never materializes the `n x n` scores plane. It costs arithmetic in
//!    proportion to `128/head_dim`; see the note on `selfAttn`.
//!
//! 3. **head_dim 160 (SD1.5's two innermost levels) takes the scalar kernel
//!    instead**, because 160 does not fit a 128-wide tile at all. That is
//!    affordable only because the wide-head levels are also the small-`n` ones:
//!    at a 512-square render they carry 256 and 64 positions.
//!
//! Numerics are f16 tensor cores for the GEMMs and f32 everywhere else, so this
//! is NOT bit-identical to `sd_unet.forward`; the CPU path remains the reference.

const std = @import("std");
const sd_unet = @import("sd_unet.zig");
const ops = @import("tp_ops");
const gpu_context = @import("tp_gpu").context;

const Context = gpu_context.Context;
const DeviceBuffer = gpu_context.DeviceBuffer;
const Weight = ops.matmul.Weight;
const Conv2d = ops.conv.Conv2d;
const Config = sd_unet.Config;

const none: DeviceBuffer = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 };

/// Cap on the im2col patch band (bytes); bands iterate over output rows.
const patch_band_bytes: usize = 256 << 20;

/// Column groups per GroupNorm statistics pass. The reduction is over
/// `positions * channels/32` values per group, so 256 chunks keeps every thread
/// walking a useful run at a 64-square latent while still filling the GPU at a
/// 128-square one.
pub const gn_chunks: usize = 256;

/// The head width the cooperative-matrix attention pipelines are built for.
const tc_head_dim: usize = 128;

/// Interleaved chunks per row in the two-pass softmax, so a warp covers a row
/// with coalesced reads. Same value the DiT uses.
const nchunks: u32 = 32;

/// Budget for the O(seq^2) scores plane; heads are processed in batches that fit
/// it. At a 512-square SD1.5 render one head's plane is 33 MB, so 8 heads land in
/// a single batch.
const scores_scratch_bytes: usize = 512 << 20;

/// Convolutions at least this wide go to the f16 tensor-core GEMM; below it the
/// f32 register-tile GEMM wins, because padding `co` out to the 128-wide
/// cooperative tile would waste more than the tensor cores return. Same
/// threshold `vae_gpu` measured.
const coop_min_co: usize = 96;

/// How `im2col_sd` samples its source (must match the kernel's `f0` encoding).
pub const SampleMode = enum(u2) { stride1 = 0, upsample2x = 1, stride2 = 2 };

// --- per-conditioning session ----------------------------------------------

/// Everything tied to one conditioning branch: the uploaded context, and the
/// per-forward ResBlock bias vectors with the timestep embedding folded in.
/// Under classifier-free guidance there are two of these against one `Workspace`
/// — the positive and negative branches differ in their context and, for SDXL,
/// in `adm` (whose pooled half is prompt-dependent), so their folded biases
/// differ too.
pub const Session = struct {
    arena: std.heap.ArenaAllocator,
    ctx_d: DeviceBuffer = none,
    ctx_seq: usize,
    /// SDXL's `y` micro-conditioning; borrowed, must outlive the session.
    adm: ?[]const f32,

    /// Every ResBlock's folded bias, packed end to end so one upload serves the
    /// whole forward — a `tensorUpload` per ResBlock would be ~30 staged copies
    /// and submits per step.
    bias_host: []f32,
    bias_d: DeviceBuffer = none,
    /// Element offset into `bias_host` per ResBlock, indexed by the block's
    /// **ordinal in the graph walk** (`sd_unet.ResBlockIter`'s order, which is the
    /// order `forward` visits them in).
    ///
    /// ⚠️ NOT keyed by a host pointer, which is what this was first and is a trap:
    /// `Session.replaceDenoiser` reloads the UNet into a fresh arena — the whole
    /// point of the per-tensor divergence arm — so every weight pointer changes
    /// while this session stays alive, and a pointer-keyed lookup goes from
    /// correct to a null-unwrap panic. An ordinal is a property of the
    /// architecture, which a reload does not change.
    bias_off: []usize,

    /// Time embedding for this forward (host; 1280 wide).
    emb: []f32,
    /// Scratch for the two-layer time/label MLPs.
    mlp_hidden: []f32,
    mlp_scratch: []f32,

    /// GroupNorm weight ++ bias concatenations, cached by weight pointer.
    /// `gn_apply` reads both out of ONE binding (x, out and the statistics take
    /// the other three), and the checkpoint stores them as two tensors.
    norm_cat: std.AutoHashMapUnmanaged(usize, []f32) = .empty,

    /// A zero vector long enough to serve as "no bias" for the attention and
    /// projection GEMMs, whose weights carry none. Stable, so the Context's
    /// small-buffer cache uploads it once.
    zeros: []f32,

    pub fn init(
        gpa: std.mem.Allocator,
        ctx: *Context,
        u: *const sd_unet.UNet,
        context: []const f32,
        ctx_seq: usize,
        adm: ?[]const f32,
    ) !Session {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();
        const cfg = u.cfg;

        // Total folded-bias length and the per-block offsets, in one walk.
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

        // ⚠️ Every allocation happens BEFORE the struct literal. `.arena = arena`
        // copies the arena's state as it stands, so anything a *later* field
        // initializer allocates is leaked — invisible except to the test
        // allocator. `dit.zig` and `clip_text.zig` both learned this.
        const bias_host = try alloc.alloc(f32, total);
        const emb = try alloc.alloc(f32, cfg.time_embed_dim);
        const mlp_hidden = try alloc.alloc(f32, cfg.time_embed_dim);
        const mlp_scratch = try alloc.alloc(f32, cfg.time_embed_dim);
        // Widest "no bias" any GEMM here needs: a feed-forward projection emits
        // 8x the level's channels.
        const zeros = try alloc.alloc(f32, max_ch * 8);
        @memset(zeros, 0);

        var self: Session = .{
            .arena = arena,
            .ctx_seq = ctx_seq,
            .adm = adm,
            .bias_host = bias_host,
            .bias_off = offs.items,
            .emb = emb,
            .mlp_hidden = mlp_hidden,
            .mlp_scratch = mlp_scratch,
            .zeros = zeros,
        };

        try ctx.ensureDeviceBuffer(&self.ctx_d, context.len * 4);
        try ctx.tensorUpload(self.ctx_d, std.mem.sliceAsBytes(context));
        try ctx.ensureDeviceBuffer(&self.bias_d, @max(total, 1) * 4);
        return self;
    }

    pub fn deinit(self: *Session, ctx: *Context) void {
        ctx.tensorDestroy(&self.ctx_d);
        ctx.tensorDestroy(&self.bias_d);
        self.arena.deinit();
        self.* = undefined;
    }

    fn normCat(self: *Session, nw: sd_unet.GroupNormW) ![]const f32 {
        const key = @intFromPtr(nw.w.ptr);
        if (self.norm_cat.get(key)) |v| return v;
        const alloc = self.arena.allocator();
        const cat = try alloc.alloc(f32, nw.w.len + nw.b.len);
        @memcpy(cat[0..nw.w.len], nw.w);
        @memcpy(cat[nw.w.len..], nw.b);
        try self.norm_cat.put(alloc, key, cat);
        return cat;
    }
};


// --- per-resolution workspace ----------------------------------------------

/// Device scratch for one image size, shared by both conditioning branches.
pub const Workspace = struct {
    /// Ping-pong activations plus two convolution/norm scratches. All four are
    /// sized for the widest stage, which is a concatenated output-side ResBlock
    /// input (`ch + skip_ch`).
    cur: DeviceBuffer = none,
    alt: DeviceBuffer = none,
    t1: DeviceBuffer = none,
    t2: DeviceBuffer = none,
    /// im2col patch band.
    patch: DeviceBuffer = none,
    /// Skip stack: `input_stages.len + 1` activations, and their widths.
    skips: []DeviceBuffer,
    skip_ch: []usize,
    /// SpatialTransformer residual stream and a general `[n][ch]`-ish scratch.
    stream: DeviceBuffer = none,
    nb: DeviceBuffer = none,
    /// Attention q/k/v and the un-padded output.
    q: DeviceBuffer = none,
    k: DeviceBuffer = none,
    v: DeviceBuffer = none,
    ao: DeviceBuffer = none,
    /// f16 head-padded attention operands and the f32 head-padded output.
    qh: DeviceBuffer = none,
    kh: DeviceBuffer = none,
    vh: DeviceBuffer = none,
    apad: DeviceBuffer = none,
    /// Scores plane (f16), its per-chunk softmax partials, and the merged
    /// per-row {max, 1/denom}. Grown on demand, since the batch size depends on
    /// the level's resolution.
    s: DeviceBuffer = none,
    part: DeviceBuffer = none,
    md: DeviceBuffer = none,
    /// Feed-forward: `[n][2*inner]` then the gated `[n][inner]`.
    ff: DeviceBuffer = none,
    gated: DeviceBuffer = none,
    /// GroupNorm chunk statistics and the merged per-group {mean, inv}.
    gstat: DeviceBuffer = none,
    gmi: DeviceBuffer = none,
    /// Latent in, eps out.
    xin: DeviceBuffer = none,
    eps: DeviceBuffer = none,

    gpa: std.mem.Allocator,

    pub fn init(
        gpa: std.mem.Allocator,
        ctx: *Context,
        u: *const sd_unet.UNet,
        lat_h: usize,
        lat_w: usize,
        ctx_seq: usize,
    ) !Workspace {
        const cfg = u.cfg;
        var self: Workspace = .{
            .gpa = gpa,
            .skips = try gpa.alloc(DeviceBuffer, u.input_stages.len + 1),
            .skip_ch = try gpa.alloc(usize, u.input_stages.len + 1),
        };
        errdefer {
            gpa.free(self.skips);
            gpa.free(self.skip_ch);
        }
        for (self.skips) |*s| s.* = none;
        @memset(self.skip_ch, 0);

        // Same sizing walk as `sd_unet.Workspace.init`: the attention and
        // feed-forward scratches are sized per ATTENDING level at its own
        // resolution and width, not by the outermost resolution times the
        // innermost width — a product no stage ever has (SDXL's outermost level
        // does not attend at all).
        var act: usize = 0;
        var attn: usize = 0;
        var ff: usize = 0;
        var gated: usize = 0;
        var pad_f32: usize = 0;
        var pad_f16: usize = 0;
        var scores: usize = 0;
        var part: usize = 0;
        var md: usize = 0;
        var patch: usize = 0;
        var h = lat_h;
        var w = lat_w;
        for (cfg.channel_mult, 0..) |mult, level| {
            const ch = cfg.model_channels * mult;
            act = @max(act, h * w * ch * 3);
            // The widest patch row at this level: a 3x3 convolution over the
            // concatenated output-side input.
            patch = @max(patch, sizePatch(h * w, ch * 3));
            if (cfg.attn_levels[level]) sizeAttn(cfg, h * w, ch, ctx_seq, &attn, &ff, &gated, &pad_f32, &pad_f16, &scores, &part, &md);
            if (level + 1 < cfg.levels()) {
                h = (h + 1) / 2;
                w = (w + 1) / 2;
            }
        }
        // The middle block always attends, at the innermost resolution — which
        // the loop skips when that level carries no SpatialTransformer of its own
        // (SD1.5's fourth level).
        sizeAttn(cfg, h * w, cfg.model_channels * cfg.channel_mult[cfg.levels() - 1], ctx_seq, &attn, &ff, &gated, &pad_f32, &pad_f16, &scores, &part, &md);

        inline for (.{ "cur", "alt", "t1", "t2" }) |f| {
            try ctx.ensureDeviceBuffer(&@field(self, f), act * 4);
        }
        try ctx.ensureDeviceBuffer(&self.stream, attn * 4);
        try ctx.ensureDeviceBuffer(&self.nb, attn * 4);
        inline for (.{ "q", "k", "v", "ao" }) |f| {
            try ctx.ensureDeviceBuffer(&@field(self, f), attn * 4);
        }
        inline for (.{ "qh", "kh", "vh" }) |f| {
            try ctx.ensureDeviceBuffer(&@field(self, f), pad_f16);
        }
        try ctx.ensureDeviceBuffer(&self.apad, pad_f32);
        try ctx.ensureDeviceBuffer(&self.ff, ff * 4);
        try ctx.ensureDeviceBuffer(&self.gated, gated * 4);
        try ctx.ensureDeviceBuffer(&self.s, scores);
        try ctx.ensureDeviceBuffer(&self.part, part);
        try ctx.ensureDeviceBuffer(&self.md, md);
        try ctx.ensureDeviceBuffer(&self.patch, patch);
        try ctx.ensureDeviceBuffer(&self.gstat, cfg.norm_groups * gn_chunks * 3 * 4);
        try ctx.ensureDeviceBuffer(&self.gmi, cfg.norm_groups * 2 * 4);
        try ctx.ensureDeviceBuffer(&self.xin, lat_h * lat_w * cfg.channels * 4);
        try ctx.ensureDeviceBuffer(&self.eps, lat_h * lat_w * cfg.channels * 4);
        return self;
    }

    fn sizeAttn(
        cfg: Config,
        n: usize,
        ch: usize,
        ctx_seq: usize,
        attn: *usize,
        ff: *usize,
        gated: *usize,
        pad_f32: *usize,
        pad_f16: *usize,
        scores: *usize,
        part: *usize,
        md: *usize,
    ) void {
        // Cross-attention writes `ctx_seq` rows of keys and values, which at a
        // small latent EXCEEDS the position count (a 16-square latent gives 64
        // positions against 77 context rows).
        attn.* = @max(attn.*, @max(n, ctx_seq) * ch);
        ff.* = @max(ff.*, n * ch * 8);
        gated.* = @max(gated.*, n * ch * 4);
        const heads = cfg.headsAt(ch);
        if (ch / heads <= tc_head_dim) {
            const seq_pad = std.mem.alignForward(usize, n, tc_head_dim);
            const rows = seq_pad * heads * tc_head_dim;
            pad_f32.* = @max(pad_f32.*, rows * 4);
            pad_f16.* = @max(pad_f16.*, rows * 2);
            const hpb = headsPerBatch(seq_pad, heads);
            scores.* = @max(scores.*, hpb * seq_pad * seq_pad * 2);
            part.* = @max(part.*, hpb * n * nchunks * 2 * 4);
            md.* = @max(md.*, hpb * seq_pad * 2 * 4);
        }
    }

    /// Bytes the im2col band needs for a 3x3 convolution of this shape. A pure
    /// function of it, so `init` and `convInto` cannot disagree about the size —
    /// which matters because growing the band inside a recording batch would free
    /// memory that already recorded dispatches reference.
    fn sizePatch(n_out: usize, ci: usize) usize {
        const patch_len = 9 * ci;
        return convBand(n_out, patch_len) * patch_len * 4;
    }

    pub fn deinit(self: *Workspace, ctx: *Context) void {
        inline for (@typeInfo(Workspace).@"struct".fields) |f| {
            if (f.type == DeviceBuffer) ctx.tensorDestroy(&@field(self, f.name));
        }
        for (self.skips) |*s| ctx.tensorDestroy(s);
        self.gpa.free(self.skips);
        self.gpa.free(self.skip_ch);
        self.* = undefined;
    }
};

// --- forward ----------------------------------------------------------------

/// eps = UNet(x, t, context, y), all on the device. `x` and `out` are
/// channel-last `[lat_h*lat_w][channels]` host buffers, the layout
/// `Denoiser.predictSd` transposes into.
pub fn forward(
    u: *const sd_unet.UNet,
    ctx: *Context,
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

    try embedAndFoldBiases(u, ctx, sess, io, gpa, timestep);
    try ctx.tensorUpload(ws.xin, std.mem.sliceAsBytes(x));

    try ctx.beginBatch();
    var batched = true;
    errdefer if (batched) ctx.endBatch() catch {};

    var h = lat_h;
    var w = lat_w;
    var ch = cfg.model_channels;
    // Counts ResBlocks in the order `ResBlockIter` yields them, which is the order
    // below: the input stages, then the two middle blocks, then the output stages.
    var rb_ord: usize = 0;

    // --- stem ---
    try conv(ctx, ws, &ws.cur, &ws.xin, h, w, u.stem, .stride1, null);
    try setSkip(ctx, ws, 0, &ws.cur, h * w * ch, ch);

    // --- input side ---
    for (u.input_stages, 0..) |stage, si| {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        if (stage.res) |rb| {
            try applyRes(ctx, sess, ws, rb, rb_ord, &ws.cur, &ws.alt, h, w, cfg);
            rb_ord += 1;
            std.mem.swap(DeviceBuffer, &ws.cur, &ws.alt);
            ch = rb.out_ch;
        }
        if (stage.attn) |st| {
            try applySpatial(ctx, sess, ws, st, h, w, cfg);
        }
        if (stage.sample_kind == .down) {
            const oh = (h + 1) / 2;
            const ow = (w + 1) / 2;
            try conv(ctx, ws, &ws.alt, &ws.cur, h, w, stage.sample.?, .stride2, null);
            std.mem.swap(DeviceBuffer, &ws.cur, &ws.alt);
            h = oh;
            w = ow;
        }
        try setSkip(ctx, ws, si + 1, &ws.cur, h * w * ch, ch);
    }

    // --- middle ---
    try applyRes(ctx, sess, ws, u.mid_res1, rb_ord, &ws.cur, &ws.alt, h, w, cfg);
    rb_ord += 1;
    std.mem.swap(DeviceBuffer, &ws.cur, &ws.alt);
    ch = u.mid_res1.out_ch;
    try applySpatial(ctx, sess, ws, u.mid_attn, h, w, cfg);
    try applyRes(ctx, sess, ws, u.mid_res2, rb_ord, &ws.cur, &ws.alt, h, w, cfg);
    rb_ord += 1;
    std.mem.swap(DeviceBuffer, &ws.cur, &ws.alt);
    ch = u.mid_res2.out_ch;

    // --- output side ---
    var skip_top = u.input_stages.len + 1;
    for (u.output_stages) |stage| {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        skip_top -= 1;
        const skip_ch = ws.skip_ch[skip_top];
        {
            // Concatenate along channels into `alt`: current first, then the
            // popped skip. Two strided copies rather than one in-place widening
            // — the source and destination are distinct buffers here, so unlike
            // the CPU path there is nothing to overwrite.
            const total = ch + skip_ch;
            ctx.independent(2);
            try ctx.opElt(.concat_ch, ws.cur, ws.alt, null, null, .{
                .u0 = @intCast(h * w * ch),
                .u1 = @intCast(ch),
                .u2 = @intCast(total),
                .u3 = 0,
            }, h * w * ch, 1, 1);
            try ctx.opElt(.concat_ch, ws.skips[skip_top], ws.alt, null, null, .{
                .u0 = @intCast(h * w * skip_ch),
                .u1 = @intCast(skip_ch),
                .u2 = @intCast(total),
                .u3 = @intCast(ch),
            }, h * w * skip_ch, 1, 1);
            std.mem.swap(DeviceBuffer, &ws.cur, &ws.alt);
            ch = total;
        }
        const rb = stage.res.?;
        try applyRes(ctx, sess, ws, rb, rb_ord, &ws.cur, &ws.alt, h, w, cfg);
        rb_ord += 1;
        std.mem.swap(DeviceBuffer, &ws.cur, &ws.alt);
        ch = rb.out_ch;
        if (stage.attn) |st| {
            try applySpatial(ctx, sess, ws, st, h, w, cfg);
        }
        if (stage.sample_kind == .up) {
            // LDM's `Upsample`: nearest 2x then a 3x3 convolution. The resample
            // is fused into the patch gather, so the doubled tensor never exists.
            try conv(ctx, ws, &ws.alt, &ws.cur, h, w, stage.sample.?, .upsample2x, null);
            std.mem.swap(DeviceBuffer, &ws.cur, &ws.alt);
            h *= 2;
            w *= 2;
        }
    }

    // --- head ---
    try groupNorm(ctx, ws, &ws.t1, &ws.cur, h * w, ch, try sess.normCat(u.out_norm), cfg.norm_groups, cfg.norm_eps, true);
    try conv(ctx, ws, &ws.eps, &ws.t1, h, w, u.out_conv, .stride1, null);

    batched = false;
    try ctx.endBatch();
    try ctx.tensorDownload(ws.eps, std.mem.sliceAsBytes(out));
}

/// The time embedding (plus SDXL's label embedding), and every ResBlock's
/// `emb_layers` projection folded into its first convolution's bias. All on the
/// host: the widest of these is a 1280 x 1280 GEMV, and the whole set is a few
/// tens of MFLOP against the forward's hundreds of GFLOP.
fn embedAndFoldBiases(
    u: *const sd_unet.UNet,
    ctx: *Context,
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

    // `emb = time_embed(t) + label_emb(y)` — a sum, not a concatenation, and the
    // same shape either way, so a missing term would be invisible downstream.
    if (sess.adm) |y| {
        try ops.matmul.matmul(io, gpa, sess.mlp_hidden, y, 1, u.label_1.?.w, u.label_1.?.b);
        ops.act.silu(sess.mlp_hidden);
        try ops.matmul.matmul(io, gpa, sess.mlp_scratch, sess.mlp_hidden, 1, u.label_2.?.w, u.label_2.?.b);
        for (sess.emb, sess.mlp_scratch) |*e, v| e.* += v;
    }

    // emb_layers is [SiLU, Linear] over the SHARED embedding, so the silu is
    // computed once here rather than per block.
    @memcpy(sess.mlp_hidden, sess.emb);
    ops.act.silu(sess.mlp_hidden);
    var it = sd_unet.ResBlockIter.init(u);
    var bi: usize = 0;
    while (it.next()) |rb| : (bi += 1) {
        const dst = sess.bias_host[sess.bias_off[bi]..][0..rb.out_ch];
        try ops.matmul.matmul(io, gpa, dst, sess.mlp_hidden, 1, rb.emb.w, rb.emb.b);
        for (dst, rb.in_conv.b.?) |*d, cb| d.* += cb;
    }
    if (sess.bias_host.len > 0) {
        try ctx.tensorUpload(sess.bias_d, std.mem.sliceAsBytes(sess.bias_host));
    }
}

fn setSkip(ctx: *Context, ws: *Workspace, i: usize, src: *const DeviceBuffer, elems: usize, ch: usize) !void {
    try ctx.ensureDeviceBuffer(&ws.skips[i], elems * 4);
    ws.skip_ch[i] = ch;
    try ctx.opElt(.copy, src.*, ws.skips[i], null, null, .{ .u0 = @intCast(elems) }, elems, 1, 1);
}

/// A ResBlock, result guaranteed in `dst` (which the helper may swap with its
/// own scratch to get there — cheaper than a copy, and `t2` is scratch either
/// way). `src` must survive to the residual add, so the first norm is
/// out-of-place.
fn applyRes(
    ctx: *Context,
    sess: *Session,
    ws: *Workspace,
    rb: sd_unet.ResBlock,
    /// The block's ordinal in `sd_unet.ResBlockIter`'s order, which `forward`
    /// visits them in. See `Session.bias_off`.
    ordinal: usize,
    src: *DeviceBuffer,
    dst: *DeviceBuffer,
    h: usize,
    w: usize,
    cfg: Config,
) !void {
    const n = h * w;
    const bias_off = sess.bias_off[ordinal];

    // in_layers: GroupNorm -> SiLU -> conv3x3, with the timestep projection
    // riding in on the convolution's bias.
    try groupNorm(ctx, ws, &ws.t1, src, n, rb.in_ch, try sess.normCat(rb.in_norm), cfg.norm_groups, cfg.norm_eps, true);
    try conv(ctx, ws, dst, &ws.t1, h, w, rb.in_conv, .stride1, .{ .buf = sess.bias_d, .off = bias_off });

    // out_layers: GroupNorm -> SiLU -> conv3x3.
    try groupNorm(ctx, ws, &ws.t1, dst, n, rb.out_ch, try sess.normCat(rb.out_norm), cfg.norm_groups, cfg.norm_eps, true);
    try conv(ctx, ws, &ws.t2, &ws.t1, h, w, rb.out_conv, .stride1, null);

    const out_n = n * rb.out_ch;
    if (rb.skip) |sk| {
        // `dst` held conv1's output, which out_norm has already consumed.
        try conv(ctx, ws, dst, src, h, w, sk, .stride1, null);
        try ctx.opElt(.add, dst.*, ws.t2, null, null, .{ .u0 = @intCast(out_n) }, out_n, 1, 1);
    } else {
        try ctx.opElt(.add, ws.t2, src.*, null, null, .{ .u0 = @intCast(out_n) }, out_n, 1, 1);
        std.mem.swap(DeviceBuffer, &ws.t2, dst);
    }
}

/// A SpatialTransformer, in place on `ws.cur` (it ends in an outer residual add).
fn applySpatial(
    ctx: *Context,
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

    // GroupNorm -> proj_in, straight into the transformer's residual stream.
    try groupNorm(ctx, ws, &ws.nb, &ws.cur, n, ch, try sess.normCat(st.norm), cfg.norm_groups, cfg.norm_eps, false);
    try applyProj(ctx, ws, sess, &ws.stream, &ws.nb, h, w, st.proj_in);

    for (st.blocks) |b| {
        // attn1: self-attention over pixels, no mask.
        try layerNorm(ctx, &ws.nb, &ws.stream, n, ch, b.norm1, cfg.norm_eps);
        // ⚠️ NOT `ctx.independent(3)`. These three GEMMs look independent — same
        // input, three disjoint outputs — but a coop GEMM is three dispatches
        // sharing the Context's `x_h16` / `y_pad` scratch, so removing the
        // barriers would both let them clobber each other's scratch AND (since
        // the group counts dispatches, not calls) drop the barriers *inside* the
        // first one.
        try gemm(ctx, sess, &ws.q, 0, &ws.nb, n, b.attn1.q, null);
        try gemm(ctx, sess, &ws.k, 0, &ws.nb, n, b.attn1.k, null);
        try gemm(ctx, sess, &ws.v, 0, &ws.nb, n, b.attn1.v, null);
        try selfAttn(ctx, ws, n, heads, hd);
        try gemm(ctx, sess, &ws.nb, 0, &ws.ao, n, b.attn1.out.w, b.attn1.out.b);
        try ctx.opElt(.add, ws.stream, ws.nb, null, null, .{ .u0 = @intCast(n * ch) }, n * ch, 1, 1);

        // attn2: cross-attention onto the text conditioning. The keys and values
        // come from the context, so they are `ctx_seq` rows, not `n`.
        try layerNorm(ctx, &ws.nb, &ws.stream, n, ch, b.norm2, cfg.norm_eps);
        try gemm(ctx, sess, &ws.q, 0, &ws.nb, n, b.attn2.q, null);
        try gemm(ctx, sess, &ws.k, 0, &sess.ctx_d, ctx_seq, b.attn2.k, null);
        try gemm(ctx, sess, &ws.v, 0, &sess.ctx_d, ctx_seq, b.attn2.v, null);
        try ctx.opElt(.attn_cross, ws.q, ws.k, ws.v, ws.ao, .{
            .u0 = @intCast(n),
            .u1 = @intCast(heads),
            .u2 = @intCast(hd),
            .u3 = @intCast(ctx_seq),
            .f0 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd))),
        }, n * heads, 1, 1);
        try gemm(ctx, sess, &ws.nb, 0, &ws.ao, n, b.attn2.out.w, b.attn2.out.b);
        try ctx.opElt(.add, ws.stream, ws.nb, null, null, .{ .u0 = @intCast(n * ch) }, n * ch, 1, 1);

        // ff: GEGLU. `proj` emits [value | gate] per row, in that order, and the
        // gate takes the ERF gelu (see the `geglu` kernel).
        try layerNorm(ctx, &ws.nb, &ws.stream, n, ch, b.norm3, cfg.norm_eps);
        try gemm(ctx, sess, &ws.ff, 0, &ws.nb, n, b.ff_proj.w, b.ff_proj.b);
        try ctx.opElt(.geglu, ws.ff, ws.gated, null, null, .{
            .u0 = @intCast(n * b.inner),
            .u1 = @intCast(b.inner),
        }, n * b.inner, 1, 1);
        try gemm(ctx, sess, &ws.nb, 0, &ws.gated, n, b.ff_out.w, b.ff_out.b);
        try ctx.opElt(.add, ws.stream, ws.nb, null, null, .{ .u0 = @intCast(n * ch) }, n * ch, 1, 1);
    }

    try applyProj(ctx, ws, sess, &ws.nb, &ws.stream, h, w, st.proj_out);
    try ctx.opElt(.add, ws.cur, ws.nb, null, null, .{ .u0 = @intCast(n * ch) }, n * ch, 1, 1);
}

/// Self-attention over the latent positions.
///
/// ⚠️ The cooperative-matrix path is built for head_dim 128, so narrower heads
/// are zero-padded up to it: exact (a zero dimension contributes nothing to a
/// dot product, and V's zero columns give output columns `head_unpad` drops) but
/// it does `128/head_dim` times the arithmetic of the true shape. SD1.5's
/// outermost level pays 3.2x and SDXL pays 2x. The fix is a head_dim-parameterized
/// `buildGemmAttnOut` / `buildFlashAttn`, not a change here.
///
/// head_dim 160 — SD1.5's two innermost levels — does not fit a 128-wide tile at
/// all and takes the general scalar kernel. That is affordable only because the
/// wide-head levels are the small-`n` ones.
fn selfAttn(ctx: *Context, ws: *Workspace, n: usize, heads: usize, hd: usize) !void {
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
    if (hd > tc_head_dim or ctx.pipe_scores == .null_handle or ctx.pipe_attn_out == .null_handle) {
        return ctx.opElt(.attn_full, ws.q, ws.k, ws.v, ws.ao, .{
            .u0 = @intCast(n),
            .u1 = @intCast(heads),
            .u2 = @intCast(heads),
            .u3 = @intCast(hd),
            .f0 = scale,
        }, n * heads, 1, 1);
    }

    const seq_pad = std.mem.alignForward(usize, n, tc_head_dim);
    const rows = seq_pad * heads * tc_head_dim;

    ctx.independent(3);
    // Q: [seq_pad][heads*128] f16 with the softmax scale prefolded.
    try ctx.opElt(.head_pad_h16, ws.q, null, null, ws.qh, .{
        .u0 = @intCast(rows / 2),
        .u1 = @intCast(hd),
        .u2 = tc_head_dim,
        .u3 = @intCast(n),
        .u4 = @intCast(heads),
        .f0 = scale,
    }, rows / 2, 1, 1);
    // K: per-head k-major [heads][128][seq_pad] f16.
    try ctx.opElt(.gather_kmajor_h16, ws.k, null, null, ws.kh, .{
        .u0 = @intCast(rows / 2),
        .u1 = tc_head_dim,
        .u2 = @intCast(seq_pad),
        .u3 = @intCast(n),
        .u4 = @intCast(heads),
        .u5 = @intCast(hd),
    }, rows / 2, 1, 1);
    try ctx.opElt(.head_pad_h16, ws.v, null, null, ws.vh, .{
        .u0 = @intCast(rows / 2),
        .u1 = @intCast(hd),
        .u2 = tc_head_dim,
        .u3 = @intCast(n),
        .u4 = @intCast(heads),
        .f0 = 1.0,
    }, rows / 2, 1, 1);

    // Scores -> two-pass softmax -> P@V, in batches of heads so the O(seq^2)
    // scores plane stays bounded. This is the path the DiT runs (`dit_gpu`'s
    // flash switch is off: the recompute pass costs more than the coalesced S
    // write it saves at these tilings), and the one its kernels are exercised on.
    const s_stride: u32 = @intCast(seq_pad);
    const s_plane_elems = seq_pad * seq_pad;
    const s_plane: u32 = @intCast(s_plane_elems);
    // Pre-sized by `Workspace.init` from the same function, deliberately:
    // growing these inside the recording batch would free memory that already
    // recorded dispatches reference.
    const hpb = headsPerBatch(seq_pad, heads);

    var h0: usize = 0;
    while (h0 < heads) : (h0 += hpb) {
        const hb = @min(hpb, heads - h0);
        try ctx.opAttnScores(ws.s, ws.qh, ws.kh, .{
            .u0 = @intCast(heads * tc_head_dim),
            .u1 = s_stride,
            .u2 = @intCast(h0),
            .u3 = 1, // no GQA: one query head per kv head
            .u4 = @intCast(tc_head_dim * seq_pad),
            .u5 = s_plane,
        }, seq_pad / 128, seq_pad / 128, hb);
        try ctx.opElt(.softmax_partial, ws.s, null, null, ws.part, .{
            .u0 = @intCast(hb * n * nchunks),
            .u1 = nchunks,
            .u2 = @intCast(n),
            .u3 = s_stride,
            .u5 = s_plane,
        }, hb * n * nchunks, 1, 1);
        try ctx.opElt(.softmax_combine, ws.part, null, null, ws.md, .{
            .u0 = @intCast(hb * n),
            .u1 = nchunks,
            .u2 = @intCast(n),
            .u3 = s_stride,
        }, hb * n, 1, 1);
        try ctx.opAttnOut(ws.s, ws.vh, ws.apad, ws.md, .{
            .u0 = s_stride,
            .u1 = s_plane,
            .u2 = @intCast(h0),
            .u3 = 1,
            .u4 = @intCast(heads * tc_head_dim),
            .u5 = @intCast(heads * tc_head_dim),
            .f0 = @bitCast(@as(u32, @intCast(n))),
            .f1 = @bitCast(s_stride), // MD rows per head plane
        }, seq_pad / 128, hb);
    }

    try ctx.opElt(.head_unpad, ws.apad, ws.ao, null, null, .{
        .u0 = @intCast(n * heads * hd),
        .u1 = @intCast(hd),
        .u2 = tc_head_dim,
        .u4 = @intCast(heads),
    }, n * heads * hd, 1, 1);
}

/// A SpatialTransformer's in/out projection, whichever rank the checkpoint stored
/// it at — SD1.5 a 1x1 convolution, SDXL an `nn.Linear`. On channel-last
/// activations both are the same GEMM over pixels.
fn applyProj(
    ctx: *Context,
    ws: *Workspace,
    sess: *Session,
    dst: *DeviceBuffer,
    src: *const DeviceBuffer,
    h: usize,
    w: usize,
    p: sd_unet.Proj,
) !void {
    switch (p) {
        .conv => |cv| try conv(ctx, ws, dst, src, h, w, cv, .stride1, null),
        .linear => |lin| try gemm(ctx, sess, dst, 0, src, h * w, lin.w, lin.b),
    }
}

fn layerNorm(
    ctx: *Context,
    dst: *DeviceBuffer,
    src: *const DeviceBuffer,
    n: usize,
    ch: usize,
    nw: sd_unet.LayerNormW,
    eps: f32,
) !void {
    const wbuf: DeviceBuffer = .{ .buf = try ctx.smallBuffer(std.mem.sliceAsBytes(nw.w)), .mem = .null_handle, .size = 0 };
    const bbuf: DeviceBuffer = .{ .buf = try ctx.smallBuffer(std.mem.sliceAsBytes(nw.b)), .mem = .null_handle, .size = 0 };
    try ctx.opElt(.layernorm, src.*, dst.*, wbuf, bbuf, .{
        .u0 = @intCast(n),
        .u1 = @intCast(ch),
        .f0 = eps,
    }, n, 1, 1);
}

/// GroupNorm (+ the SiLU that follows it everywhere but a SpatialTransformer's
/// input norm): chunk statistics, merge per group, apply per element.
fn groupNorm(
    ctx: *Context,
    ws: *Workspace,
    dst: *DeviceBuffer,
    src: *const DeviceBuffer,
    n: usize,
    ch: usize,
    /// weight ++ bias, from `Session.normCat`.
    cat: []const f32,
    groups: usize,
    eps: f32,
    silu: bool,
) !void {
    return groupNormInto(ctx, ws.gstat, ws.gmi, dst, src, n, ch, cat, groups, eps, silu);
}

/// GroupNorm with the two statistics buffers passed in, so the SD VAE decoder
/// (`sd_vae_gpu`) runs the identical three dispatches without a UNet workspace.
/// `gstat` must hold `groups * gn_chunks * 3` f32 and `gmi` `2 * groups`.
pub fn groupNormInto(
    ctx: *Context,
    gstat: DeviceBuffer,
    gmi: DeviceBuffer,
    dst: *DeviceBuffer,
    src: *const DeviceBuffer,
    n: usize,
    ch: usize,
    cat: []const f32,
    groups: usize,
    eps: f32,
    silu: bool,
) !void {
    std.debug.assert(cat.len == 2 * ch);
    const per_group = ch / groups;
    const cbuf: DeviceBuffer = .{ .buf = try ctx.smallBuffer(std.mem.sliceAsBytes(cat)), .mem = .null_handle, .size = 0 };

    try ctx.opElt(.gn_stats, src.*, null, null, gstat, .{
        .u0 = @intCast(groups * gn_chunks),
        .u1 = @intCast(ch),
        .u2 = gn_chunks,
        .u3 = @intCast(per_group),
        .u4 = @intCast(n),
    }, groups * gn_chunks, 1, 1);
    try ctx.opElt(.gn_combine, gstat, null, null, gmi, .{
        .u0 = @intCast(groups),
        .u2 = gn_chunks,
        .f0 = eps,
    }, groups, 1, 1);
    try ctx.opElt(.gn_apply, src.*, dst.*, cbuf, gmi, .{
        .u0 = @intCast(n * ch),
        .u1 = @intCast(ch),
        .u2 = @intCast(per_group),
        .u3 = @intCast(groups),
        .u4 = @intCast(ch),
        .u5 = @intFromBool(silu),
    }, n * ch, 1, 1);
}

/// Heads per scores batch: as many as fit `scores_scratch_bytes`, at least one.
/// A pure function of the shape, so `Workspace.init` and `selfAttn` cannot
/// disagree about how much scratch the batch needs.
fn headsPerBatch(seq_pad: usize, heads: usize) usize {
    return @max(1, @min(heads, scores_scratch_bytes / (seq_pad * seq_pad * 2)));
}

/// A GEMM against a checkpoint weight, routed by the dtype it was stored in.
/// `bias` null means the weight carries none (attention q/k/v), which the
/// bias-adding kernels still need a vector for — hence the session's zeros.
fn gemm(
    ctx: *Context,
    sess: *Session,
    y: *DeviceBuffer,
    y_off_elems: usize,
    x: *const DeviceBuffer,
    m: usize,
    wt: Weight,
    bias: ?[]const f32,
) !void {
    const rows = wt.rows;
    const cols = wt.cols;
    // ⚠️ The whole `zeros` array, NOT `zeros[0..rows]`. Both backends cache a bias
    // by POINTER and size the device buffer from the FIRST call's length, so a
    // narrow layer seen first would leave every wider one reading past the end —
    // which robust-buffer-access hides by returning zeros, i.e. exactly the right
    // answer for THIS bias and nothing else. The kernels read only `rows`
    // entries, so handing them a longer slice is free. (Context.smallBuffer)
    const b = bias orelse sess.zeros;
    const coop = ctx.pipe_coop_f16w != .null_handle and rows >= coop_min_co;
    switch (wt.dtype) {
        .f32 => {
            // A safetensors payload is 8-byte aligned as a whole, so an individual
            // f32 tensor's offset is 4-aligned in practice but not by guarantee.
            // The coop entry point wants a real `[]f32`; an unaligned weight falls
            // back to the raw-bytes f32 GEMM, which is correct either way. (Every
            // weight takes the same branch every time, so the pointer-keyed weight
            // cache never sees one tensor in two layouts.)
            if (coop and std.mem.isAligned(@intFromPtr(wt.bytes.ptr), @alignOf(f32))) {
                const wf: []const f32 = @as([*]const f32, @ptrCast(@alignCast(wt.bytes.ptr)))[0 .. wt.bytes.len / 4];
                return ctx.opMatmulCoopF16W(y.*, y_off_elems, x.*, m, wf, rows, cols, b);
            }
            return ctx.opMatmul(y.*, y_off_elems * 4, x.*, 0, m, wt.bytes, false, rows, cols, 1.0, b);
        },
        .f16 => return ctx.opMatmulCoopF16Wh(y.*, y_off_elems, x.*, m, wt.bytes, rows, cols, b),
        .bf16 => {
            if (ctx.pipe_coop_bf16w != .null_handle) {
                return ctx.opMatmulCoopBf16(y.*, y_off_elems, x.*, m, wt.bytes, rows, cols, b);
            }
            return ctx.opMatmulCoopF16Wb(y.*, y_off_elems, x.*, m, wt.bytes, rows, cols, b);
        },
        .q8_0, .q4_k, .q5_k, .q6_k, .iq4_nl => {
            // ggufy's quantized SD UNets — the whole reason the loader keeps
            // weights in their checkpoint dtype.
            if (!ctx.hasQuantPrefillGemm()) return error.UnsupportedDType;
            return ctx.opMatmulCoopQuant(wt.dtype, y.*, y_off_elems, x.*, m, wt.bytes, rows, cols, 1.0, b, false);
        },
        else => return error.UnsupportedDType,
    }
}

/// A convolution. 1x1 is a plain GEMM over pixels; 3x3 is a banded `im2col_sd`
/// patch gather followed by the same GEMM, with the source sampling mode
/// covering stride 1, LDM's stride-2 downsample, and the fused nearest-2x
/// upsample. `bias_dev` overrides the checkpoint bias with a slice of a device
/// buffer (the ResBlocks' folded timestep projection).
fn conv(
    ctx: *Context,
    ws: *Workspace,
    dst: *DeviceBuffer,
    src: *const DeviceBuffer,
    h: usize,
    w: usize,
    cv: Conv2d,
    mode: SampleMode,
    bias_dev: ?BiasSlice,
) !void {
    return convInto(ctx, &ws.patch, dst, src, h, w, cv, mode, bias_dev);
}

/// A slice of a device buffer holding a per-forward convolution bias.
pub const BiasSlice = struct { buf: DeviceBuffer, off: usize };

/// Output positions per im2col band. A multiple of 4, which keeps the GEMM's `y`
/// byte offset 16-aligned for any `co`. Public so a caller can pre-size the patch
/// buffer before opening a command batch.
pub fn convBand(n_out: usize, patch_len: usize) usize {
    return @max(4, @min(n_out, patch_band_bytes / (patch_len * 4)) & ~@as(usize, 3));
}

/// `conv` with the im2col band buffer passed in (grown as needed), so the SD VAE
/// decoder shares this exact convolution mapping rather than reimplementing it.
pub fn convInto(
    ctx: *Context,
    patch: *DeviceBuffer,
    dst: *DeviceBuffer,
    src: *const DeviceBuffer,
    h: usize,
    w: usize,
    cv: Conv2d,
    mode: SampleMode,
    bias_dev: ?BiasSlice,
) !void {
    return convIntoScaled(ctx, patch, dst, src, h, w, cv, mode, bias_dev, 1.0);
}

/// `convInto` with the activation divided by `act_div` before the f16 cast (and
/// the result multiplied back) — see `Context.opMatmulCoopF16WScaled` for why f16's
/// 65504 ceiling is reachable in practice. `act_div = 1.0` is exactly `convInto`.
///
/// The non-coop arm ignores it: `opMatmul` never casts to f16, so it has f32's
/// range already. The device-bias arm (`bias_dev`) does not take it either — that
/// is the SD UNet's folded timestep projection, whose activations are GroupNorm
/// outputs, and the VAE decoder (the stage that overflows) never uses it.
pub fn convIntoScaled(
    ctx: *Context,
    patch: *DeviceBuffer,
    dst: *DeviceBuffer,
    src: *const DeviceBuffer,
    h: usize,
    w: usize,
    cv: Conv2d,
    mode: SampleMode,
    bias_dev: ?BiasSlice,
    act_div: f32,
) !void {
    const coop = ctx.pipe_coop_f16w != .null_handle and cv.co >= coop_min_co;
    // Every convolution in an LDM UNet carries a bias; `Conv2d` models the
    // general case, so pin the assumption here rather than in five call sites.
    const cb = cv.b orelse return error.MissingConvBias;
    if (cv.k == 1) {
        std.debug.assert(mode == .stride1);
        const n = h * w;
        if (bias_dev) |bd| {
            std.debug.assert(coop);
            return ctx.opMatmulCoopF16WDev(dst.*, 0, src.*, n, cv.w, cv.co, cv.ci, bd.buf, bd.off);
        }
        if (coop) return ctx.opMatmulCoopF16WScaled(dst.*, 0, src.*, n, cv.w, cv.co, cv.ci, cb, act_div);
        return ctx.opMatmul(dst.*, 0, src.*, 0, n, std.mem.sliceAsBytes(cv.w), false, cv.co, cv.ci, 1.0, cb);
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
    try ctx.ensureDeviceBuffer(patch, band * patch_len * 4);

    var p0: usize = 0;
    while (p0 < n_out) : (p0 += band) {
        const bn = @min(band, n_out - p0);
        try ctx.opElt(.im2col_sd, src.*, null, null, patch.*, .{
            .u0 = @intCast(bn * patch_len),
            .u1 = @intCast(patch_len),
            .u2 = @intCast(cv.ci),
            .u3 = @intCast(w),
            .u4 = @intCast(h),
            .u5 = @intCast(p0),
            .u6 = @intCast(ow),
            .f0 = @floatFromInt(@intFromEnum(mode)),
        }, bn * patch_len, 1, 1);
        if (bias_dev) |bd| {
            std.debug.assert(coop);
            try ctx.opMatmulCoopF16WDev(dst.*, p0 * cv.co, patch.*, bn, cv.w, cv.co, patch_len, bd.buf, bd.off);
        } else if (coop) {
            try ctx.opMatmulCoopF16WScaled(dst.*, p0 * cv.co, patch.*, bn, cv.w, cv.co, patch_len, cb, act_div);
        } else {
            try ctx.opMatmul(dst.*, p0 * cv.co * 4, patch.*, 0, bn, std.mem.sliceAsBytes(cv.w), false, cv.co, patch_len, 1.0, cb);
        }
    }
}

// --- tests ------------------------------------------------------------------
//
// Each kernel this file added is checked against the CPU op it is meant to
// reproduce, on random data, so a mismatch localizes to one kernel instead of
// surfacing as a bad image at the end of a 30-second render. Gated on the GPU
// marker like the other device tests (see context.zig).

const testing = std.testing;

/// A live Vulkan context, or a skip. Needs BOTH the `testdata/gpu-tests` marker
/// and `-Dintegration` (`Context.init` refuses under test without it).
fn gpuCtx(gpa: std.mem.Allocator, io: std.Io) !*Context {
    std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    return Context.init(gpa) catch error.SkipZigTest;
}

fn upload(ctx: *Context, db: *DeviceBuffer, host: []const f32) !void {
    try ctx.ensureDeviceBuffer(db, host.len * 4);
    try ctx.tensorUpload(db.*, std.mem.sliceAsBytes(host));
}

/// Relative L2 of `got` against `want`.
fn relL2(want: []const f32, got: []const f32) f64 {
    var num: f64 = 0;
    var den: f64 = 0;
    for (want, got) |e, a| {
        const d = @as(f64, e) - @as(f64, a);
        num += d * d;
        den += @as(f64, e) * @as(f64, e);
    }
    if (den == 0) return num;
    return @sqrt(num / den);
}

test "gpu group norm matches ops.norm.groupNorm" {
    const gpa = testing.allocator;
    const io = testing.io;
    const ctx = try gpuCtx(gpa, io);
    defer ctx.deinit();

    var prng = std.Random.DefaultPrng.init(7);
    const rand = prng.random();
    const groups = 32;
    // 320 is the outermost SD width; 130 positions is deliberately not a
    // multiple of the 256 statistics chunks, so the short-chunk path runs.
    const n = 130;
    const ch = 320;

    const x = try gpa.alloc(f32, n * ch);
    defer gpa.free(x);
    const cat = try gpa.alloc(f32, 2 * ch);
    defer gpa.free(cat);
    for (cat) |*v| v.* = rand.floatNorm(f32);

    var ws: Workspace = .{
        .gpa = gpa,
        .skips = try gpa.alloc(DeviceBuffer, 0),
        .skip_ch = try gpa.alloc(usize, 0),
    };
    defer ws.deinit(ctx);
    try ctx.ensureDeviceBuffer(&ws.gstat, groups * gn_chunks * 3 * 4);
    try ctx.ensureDeviceBuffer(&ws.gmi, groups * 2 * 4);
    var src: DeviceBuffer = none;
    defer ctx.tensorDestroy(&src);
    var dst: DeviceBuffer = none;
    defer ctx.tensorDestroy(&dst);
    try ctx.ensureDeviceBuffer(&dst, n * ch * 4);

    // Two regimes, because they are bounded by different things.
    //
    // `mean 0` is what a UNet activation actually looks like, and there the
    // device result tracks the f64 CPU reference to f32 rounding.
    //
    // `mean 400` is the regime the CPU implementation's comment warns about (a
    // late VAE decoder block), and its error floor is NOT the statistics
    // algorithm: `x - mean` cancels ~400 down to ~1, so f32's 6e-8 relative
    // representation of `x` itself becomes ~2e-5 absolute in a quantity of size
    // 1. Welford is still what makes even this much work — the shifted
    // `E[x^2] - E[x]^2` form has to resolve 160001 - 160000 in f32 and lands
    // ~1% out on the variance, two orders worse.
    const cases = [_]struct { mean: f32, tol: f64 }{
        .{ .mean = 0, .tol = 2e-6 },
        .{ .mean = 400, .tol = 2e-4 },
    };
    for (cases) |c| {
        for (x) |*v| v.* = c.mean + rand.floatNorm(f32);
        try ctx.tensorUpload(src_or(&src, ctx, x), std.mem.sliceAsBytes(x));
        for ([2]bool{ false, true }) |silu| {
            try groupNorm(ctx, &ws, &dst, &src, n, ch, cat, groups, 1e-5, silu);
            const got = try gpa.alloc(f32, n * ch);
            defer gpa.free(got);
            try ctx.tensorDownload(dst, std.mem.sliceAsBytes(got));

            const want = try gpa.alloc(f32, n * ch);
            defer gpa.free(want);
            ops.norm.groupNorm(want, x, ch, groups, cat[0..ch], cat[ch..], 1e-5);
            if (silu) ops.act.silu(want);
            const err = relL2(want, got);
            errdefer std.debug.print("mean={d} silu={}: rel L2 {d:.8}\n", .{ c.mean, silu, err });
            try testing.expect(err < c.tol);
        }
    }
}

/// Ensure-and-return, so an upload can be inlined into an expression.
fn src_or(db: *DeviceBuffer, ctx: *Context, host: []const f32) DeviceBuffer {
    ctx.ensureDeviceBuffer(db, host.len * 4) catch {};
    return db.*;
}

test "gpu geglu matches the erf-gelu gate" {
    const gpa = testing.allocator;
    const io = testing.io;
    const ctx = try gpuCtx(gpa, io);
    defer ctx.deinit();

    var prng = std.Random.DefaultPrng.init(11);
    const rand = prng.random();
    const n = 37;
    const inner = 96;
    const src_h = try gpa.alloc(f32, n * 2 * inner);
    defer gpa.free(src_h);
    for (src_h) |*v| v.* = rand.floatNorm(f32) * 3;

    var src: DeviceBuffer = none;
    defer ctx.tensorDestroy(&src);
    var dst: DeviceBuffer = none;
    defer ctx.tensorDestroy(&dst);
    try upload(ctx, &src, src_h);
    try ctx.ensureDeviceBuffer(&dst, n * inner * 4);
    try ctx.opElt(.geglu, src, dst, null, null, .{
        .u0 = @intCast(n * inner),
        .u1 = @intCast(inner),
    }, n * inner, 1, 1);

    const got = try gpa.alloc(f32, n * inner);
    defer gpa.free(got);
    try ctx.tensorDownload(dst, std.mem.sliceAsBytes(got));

    // The halves are (value, gate) in that order and the gate takes the ERF
    // gelu; swapping either is a silent quality loss, so pin both.
    for (0..n) |p| {
        const row = src_h[p * 2 * inner ..][0 .. 2 * inner];
        for (0..inner) |j| {
            const want = row[j] * ops.act.geluErfScalar(row[inner + j]);
            try testing.expectApproxEqAbs(want, got[p * inner + j], 1e-5);
        }
    }
}

test "gpu cross attention matches ops.attention with unequal q/kv lengths" {
    const gpa = testing.allocator;
    const io = testing.io;
    const ctx = try gpuCtx(gpa, io);
    defer ctx.deinit();

    var prng = std.Random.DefaultPrng.init(13);
    const rand = prng.random();
    const n = 100;
    const ctx_seq = 77; // the SD family's conditioning length
    const heads = 8;
    const hd = 40; // SD1.5's outermost level
    const dim = heads * hd;

    const q = try gpa.alloc(f32, n * dim);
    defer gpa.free(q);
    const k = try gpa.alloc(f32, ctx_seq * dim);
    defer gpa.free(k);
    const v = try gpa.alloc(f32, ctx_seq * dim);
    defer gpa.free(v);
    for (q) |*x| x.* = rand.floatNorm(f32);
    for (k) |*x| x.* = rand.floatNorm(f32);
    for (v) |*x| x.* = rand.floatNorm(f32);

    var qd: DeviceBuffer = none;
    var kd: DeviceBuffer = none;
    var vd: DeviceBuffer = none;
    var od: DeviceBuffer = none;
    defer inline for (.{ &qd, &kd, &vd, &od }) |b| ctx.tensorDestroy(b);
    try upload(ctx, &qd, q);
    try upload(ctx, &kd, k);
    try upload(ctx, &vd, v);
    try ctx.ensureDeviceBuffer(&od, n * dim * 4);
    try ctx.opElt(.attn_cross, qd, kd, vd, od, .{
        .u0 = @intCast(n),
        .u1 = @intCast(heads),
        .u2 = @intCast(hd),
        .u3 = @intCast(ctx_seq),
        .f0 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd))),
    }, n * heads, 1, 1);

    const got = try gpa.alloc(f32, n * dim);
    defer gpa.free(got);
    try ctx.tensorDownload(od, std.mem.sliceAsBytes(got));

    const want = try gpa.alloc(f32, n * dim);
    defer gpa.free(want);
    try ops.attention.attention(io, gpa, want, q, k, v, .{
        .seq_q = n,
        .seq_kv = ctx_seq,
        .n_heads = heads,
        .n_kv_heads = heads,
        .head_dim = hd,
    });
    try testing.expect(relL2(want, got) < 1e-5);
}

test "gpu self attention matches ops.attention at every SD head width" {
    const gpa = testing.allocator;
    const io = testing.io;
    const ctx = try gpuCtx(gpa, io);
    defer ctx.deinit();

    var prng = std.Random.DefaultPrng.init(17);
    const rand = prng.random();
    // 40/80/160 are SD1.5's three attending levels, 64 is SDXL's. 40, 64 and 80
    // go through the head-padded tensor-core flash path; 160 exceeds the 128-wide
    // tile and takes the scalar kernel.
    const cases = [_]struct { hd: usize, heads: usize }{
        .{ .hd = 40, .heads = 8 },
        .{ .hd = 64, .heads = 10 },
        .{ .hd = 80, .heads = 8 },
        .{ .hd = 160, .heads = 8 },
    };
    // Not a multiple of 128, so the flash path's row padding is exercised.
    const n = 300;

    for (cases) |c| {
        const dim = c.heads * c.hd;
        const q = try gpa.alloc(f32, n * dim);
        defer gpa.free(q);
        const k = try gpa.alloc(f32, n * dim);
        defer gpa.free(k);
        const v = try gpa.alloc(f32, n * dim);
        defer gpa.free(v);
        for (q) |*x| x.* = rand.floatNorm(f32);
        for (k) |*x| x.* = rand.floatNorm(f32);
        for (v) |*x| x.* = rand.floatNorm(f32);

        var ws: Workspace = .{
            .gpa = gpa,
            .skips = try gpa.alloc(DeviceBuffer, 0),
            .skip_ch = try gpa.alloc(usize, 0),
        };
        defer ws.deinit(ctx);
        try upload(ctx, &ws.q, q);
        try upload(ctx, &ws.k, k);
        try upload(ctx, &ws.v, v);
        try ctx.ensureDeviceBuffer(&ws.ao, n * dim * 4);
        const seq_pad = std.mem.alignForward(usize, n, tc_head_dim);
        const rows = seq_pad * c.heads * tc_head_dim;
        inline for (.{ "qh", "kh", "vh" }) |f| {
            try ctx.ensureDeviceBuffer(&@field(ws, f), rows * 2);
        }
        try ctx.ensureDeviceBuffer(&ws.apad, rows * 4);
        // `selfAttn` no longer grows these — `Workspace.init` pre-sizes them so
        // nothing reallocates inside a recording batch — so the test has to.
        const hpb = headsPerBatch(seq_pad, c.heads);
        try ctx.ensureDeviceBuffer(&ws.s, hpb * seq_pad * seq_pad * 2);
        try ctx.ensureDeviceBuffer(&ws.part, hpb * n * nchunks * 2 * 4);
        try ctx.ensureDeviceBuffer(&ws.md, hpb * seq_pad * 2 * 4);

        try selfAttn(ctx, &ws, n, c.heads, c.hd);
        const got = try gpa.alloc(f32, n * dim);
        defer gpa.free(got);
        try ctx.tensorDownload(ws.ao, std.mem.sliceAsBytes(got));

        const want = try gpa.alloc(f32, n * dim);
        defer gpa.free(want);
        try ops.attention.attention(io, gpa, want, q, k, v, .{
            .seq_q = n,
            .seq_kv = n,
            .n_heads = c.heads,
            .n_kv_heads = c.heads,
            .head_dim = c.hd,
        });
        // The flash path runs its GEMMs on f16 tensor cores, so the tolerance is
        // f16-scale rather than f32-scale; the scalar arm (hd 160) is much tighter
        // but one bound covers both.
        const err = relL2(want, got);
        errdefer std.debug.print("hd={d} heads={d}: rel L2 {d:.6}\n", .{ c.hd, c.heads, err });
        try testing.expect(err < 3e-3);
    }
}

test "gpu conv matches ops.conv.conv2d at stride 1, stride 2 and fused 2x upsample" {
    const gpa = testing.allocator;
    const io = testing.io;
    const ctx = try gpuCtx(gpa, io);
    defer ctx.deinit();

    var prng = std.Random.DefaultPrng.init(19);
    const rand = prng.random();
    // Odd extents so stride 2's ceil(h/2) output size and the pad edges both
    // matter; co >= 96 puts it on the tensor-core arm, which is what ships.
    const h: usize = 11;
    const w: usize = 7;
    const ci: usize = 32;
    const co: usize = 128;

    const torch_w = try gpa.alloc(f32, co * ci * 9);
    defer gpa.free(torch_w);
    for (torch_w) |*x| x.* = rand.floatNorm(f32) * 0.1;
    const bias = try gpa.alloc(f32, co);
    defer gpa.free(bias);
    for (bias) |*x| x.* = rand.floatNorm(f32);
    const packed_w = try ops.conv.packWeight(gpa, torch_w, co, ci, 3);
    defer gpa.free(packed_w);
    const in = try gpa.alloc(f32, h * w * ci);
    defer gpa.free(in);
    for (in) |*x| x.* = rand.floatNorm(f32);

    var ws: Workspace = .{
        .gpa = gpa,
        .skips = try gpa.alloc(DeviceBuffer, 0),
        .skip_ch = try gpa.alloc(usize, 0),
    };
    defer ws.deinit(ctx);
    var src: DeviceBuffer = none;
    defer ctx.tensorDestroy(&src);
    var dst: DeviceBuffer = none;
    defer ctx.tensorDestroy(&dst);
    try upload(ctx, &src, in);

    for ([3]SampleMode{ .stride1, .stride2, .upsample2x }) |mode| {
        const oh = switch (mode) {
            .stride1 => h,
            .stride2 => (h + 1) / 2,
            .upsample2x => 2 * h,
        };
        const ow = switch (mode) {
            .stride1 => w,
            .stride2 => (w + 1) / 2,
            .upsample2x => 2 * w,
        };
        const cv: Conv2d = .{
            .w = packed_w,
            .b = bias,
            .co = co,
            .ci = ci,
            .k = 3,
            .stride = if (mode == .stride2) @as(usize, 2) else 1,
            .pad = 1,
        };
        try ctx.ensureDeviceBuffer(&dst, oh * ow * co * 4);
        try conv(ctx, &ws, &dst, &src, h, w, cv, mode, null);
        const got = try gpa.alloc(f32, oh * ow * co);
        defer gpa.free(got);
        try ctx.tensorDownload(dst, std.mem.sliceAsBytes(got));

        // The CPU reference has no fused resample, so the upsample case builds
        // the doubled tensor explicitly — which is exactly the equivalence the
        // fused gather claims.
        const want = try gpa.alloc(f32, oh * ow * co);
        defer gpa.free(want);
        if (mode == .upsample2x) {
            const up = try gpa.alloc(f32, oh * ow * ci);
            defer gpa.free(up);
            for (0..oh) |y| {
                for (0..ow) |xx| {
                    const s = in[((y / 2) * w + (xx / 2)) * ci ..][0..ci];
                    @memcpy(up[(y * ow + xx) * ci ..][0..ci], s);
                }
            }
            try ops.conv.conv2d(io, gpa, want, up, oh, ow, cv);
        } else {
            try ops.conv.conv2d(io, gpa, want, in, h, w, cv);
        }
        const err = relL2(want, got);
        errdefer std.debug.print("mode={t}: rel L2 {d:.6}\n", .{ mode, err });
        // f16 tensor cores on a k = 9*ci reduction.
        try testing.expect(err < 2e-3);
    }
}

test "gpu channel concat interleaves two activations by channel" {
    const gpa = testing.allocator;
    const io = testing.io;
    const ctx = try gpuCtx(gpa, io);
    defer ctx.deinit();

    const n = 17;
    const ch_a = 12;
    const ch_b = 20;
    const total = ch_a + ch_b;
    const av = try gpa.alloc(f32, n * ch_a);
    defer gpa.free(av);
    const bv = try gpa.alloc(f32, n * ch_b);
    defer gpa.free(bv);
    for (av, 0..) |*x, i| x.* = @floatFromInt(i);
    for (bv, 0..) |*x, i| x.* = @floatFromInt(1000 + i);

    var ad: DeviceBuffer = none;
    var bd: DeviceBuffer = none;
    var od: DeviceBuffer = none;
    defer inline for (.{ &ad, &bd, &od }) |x| ctx.tensorDestroy(x);
    try upload(ctx, &ad, av);
    try upload(ctx, &bd, bv);
    try ctx.ensureDeviceBuffer(&od, n * total * 4);
    try ctx.opElt(.concat_ch, ad, od, null, null, .{
        .u0 = @intCast(n * ch_a),
        .u1 = @intCast(ch_a),
        .u2 = @intCast(total),
        .u3 = 0,
    }, n * ch_a, 1, 1);
    try ctx.opElt(.concat_ch, bd, od, null, null, .{
        .u0 = @intCast(n * ch_b),
        .u1 = @intCast(ch_b),
        .u2 = @intCast(total),
        .u3 = @intCast(ch_a),
    }, n * ch_b, 1, 1);

    const got = try gpa.alloc(f32, n * total);
    defer gpa.free(got);
    try ctx.tensorDownload(od, std.mem.sliceAsBytes(got));
    for (0..n) |p| {
        for (0..ch_a) |j| try testing.expectEqual(av[p * ch_a + j], got[p * total + j]);
        for (0..ch_b) |j| try testing.expectEqual(bv[p * ch_b + j], got[p * total + ch_a + j]);
    }
}

test "gpu head pad and unpad round-trip a narrow-head activation" {
    const gpa = testing.allocator;
    const io = testing.io;
    const ctx = try gpuCtx(gpa, io);
    defer ctx.deinit();

    var prng = std.Random.DefaultPrng.init(23);
    const rand = prng.random();
    const n = 300;
    const heads = 8;
    const hd = 40;
    const dim = heads * hd;
    const seq_pad = std.mem.alignForward(usize, n, tc_head_dim);
    const rows = seq_pad * heads * tc_head_dim;

    const src_h = try gpa.alloc(f32, n * dim);
    defer gpa.free(src_h);
    for (src_h) |*x| x.* = rand.floatNorm(f32);

    var src: DeviceBuffer = none;
    var pad16: DeviceBuffer = none;
    var pad32: DeviceBuffer = none;
    var back: DeviceBuffer = none;
    defer inline for (.{ &src, &pad16, &pad32, &back }) |b| ctx.tensorDestroy(b);
    try upload(ctx, &src, src_h);
    try ctx.ensureDeviceBuffer(&pad16, rows * 2);
    try ctx.ensureDeviceBuffer(&pad32, rows * 4);
    try ctx.ensureDeviceBuffer(&back, n * dim * 4);

    try ctx.opElt(.head_pad_h16, src, null, null, pad16, .{
        .u0 = @intCast(rows / 2),
        .u1 = @intCast(hd),
        .u2 = tc_head_dim,
        .u3 = @intCast(n),
        .u4 = @intCast(heads),
        .f0 = 1.0,
    }, rows / 2, 1, 1);

    // f16 pairs back to f32 so `head_unpad` (an f32 kernel) can be checked
    // against the input: the two together are the layout contract the
    // tensor-core attention sits between.
    const words = try gpa.alloc(u32, rows / 2);
    defer gpa.free(words);
    try ctx.tensorDownload(pad16, std.mem.sliceAsBytes(words));
    const widened = try gpa.alloc(f32, rows);
    defer gpa.free(widened);
    for (words, 0..) |wd, i| {
        widened[i * 2] = @floatCast(@as(f16, @bitCast(@as(u16, @truncate(wd)))));
        widened[i * 2 + 1] = @floatCast(@as(f16, @bitCast(@as(u16, @truncate(wd >> 16)))));
    }
    // Every padding lane must be a hard zero, or the scores GEMM picks up junk
    // from whatever the buffer held before.
    for (0..seq_pad) |r| {
        for (0..heads) |h| {
            for (hd..tc_head_dim) |t| {
                try testing.expectEqual(@as(f32, 0), widened[r * heads * tc_head_dim + h * tc_head_dim + t]);
            }
        }
    }
    try upload(ctx, &pad32, widened);
    try ctx.opElt(.head_unpad, pad32, back, null, null, .{
        .u0 = @intCast(n * dim),
        .u1 = @intCast(hd),
        .u2 = tc_head_dim,
        .u4 = @intCast(heads),
    }, n * dim, 1, 1);

    const got = try gpa.alloc(f32, n * dim);
    defer gpa.free(got);
    try ctx.tensorDownload(back, std.mem.sliceAsBytes(got));
    // f16 storage, so exact equality is not on offer; the layout is what matters.
    try testing.expect(relL2(src_h, got) < 1e-3);
}
