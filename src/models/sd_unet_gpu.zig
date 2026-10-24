//! GPU-resident SD1.5 / SDXL UNet forward (Vulkan), mirroring `sd_unet.forward` with
//! everything from the latent upload to the eps download on the device.
//!
//! The VAE decoders share this mapping (`sd_vae_gpu`): activations stay tight channel-last
//! `[h*w][c]`, a 3x3 convolution is a banded `im2col_sd` plus a GEMM, and a 1x1
//! convolution or `nn.Linear` is that GEMM with no patch step. What a UNet adds over a
//! VAE decoder is GroupNorm over 32 channel groups, the SpatialTransformer, and the LIFO
//! skip stack.
//!
//! Weights are read straight from a loaded `sd_unet.UNet` in whatever dtype the
//! checkpoint stored them, and the Context's weight cache uploads and transposes each
//! one lazily on first use.
//!
//! Each ResBlock's timestep-embedding projection is folded into its first convolution's
//! bias on the host rather than added in a pass of its own. A conv bias is already a
//! per-channel constant over positions, which is exactly what the projection is, so this
//! is exact and it removes one read-modify-write of the full activation per ResBlock.
//! The folded vector changes every forward, hence `conv_gpu.conv2d`'s `bias_dev` and one packed
//! device buffer instead of the pointer-cached `smallBuffer` path.
//!
//! Attention head widths are handled in `selfAttn`. Numerics are f16 tensor cores for
//! the GEMMs and f32 everywhere else, so this is NOT bit-identical to
//! `sd_unet.forward`; the CPU path remains the reference.
//!
//! The convolution and transformer activations, skips included, are STORED at
//! `Workspace.act` (`streamDt`); the attention operands stay f32 whatever it is.

const std = @import("std");
const sd_unet = @import("sd_unet.zig");
const conv_gpu = @import("conv_gpu.zig");
const dev = @import("sd_unet_dev.zig");
const lin_gpu = @import("lin_gpu.zig");
const attn_gpu = @import("attn_gpu.zig");
const lora_gpu = @import("lora_gpu.zig");
const lora_mod = @import("lora.zig");
const ops = @import("tp_ops");
const gpu_context = @import("tp_gpu").context;

const File = @This();
const Context = gpu_context.Context;
const dual_table = @import("tp_gpu").dual_table;
const DeviceBuffer = gpu_context.DeviceBuffer;
const Weight = ops.matmul.Weight;
const Conv2d = ops.conv.Conv2d;
const Config = sd_unet.Config;
const Dt = gpu_context.Dt;

const none: DeviceBuffer = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 };


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


pub const SampleMode = @import("sd_unet_dev.zig").SampleMode;

/// The dtype this UNet's activations are stored at on this device: `UNet.act`, else
/// f16 where `Config.act_f16` says, else f32.
pub fn streamDt(ctx: *const Context, u: *const sd_unet.UNet) Dt {
    return streamDtFor(ctx, if (u.act) |a| switch (a) {
        .f16 => .f16,
        .bf16 => .bf16,
        else => .f32,
    } else if (u.cfg.act_f16) .f16 else .f32);
}

/// What the stream would run as if `want` were asked for: a 16-bit one needs the
/// cooperative GEMM of that width, which every 16-bit convolution takes, and
/// `conv_gpu.conv2d` asks for the f16 one either way. An activation capture runs f32, as
/// on cuda (`sd_unet_cuda.streamDt` says why).
pub fn streamDtFor(ctx: *const Context, want: Dt) Dt {
    if (ctx.pipe_coop_f16w == .null_handle or ops.matmul.probe != null) return .f32;
    return switch (want) {
        .f32, .f16 => want,
        .bf16 => if (ctx.pipe_coop_bf16w != .null_handle) .bf16 else .f32,
    };
}

// --- per-conditioning session ----------------------------------------------

/// Everything tied to one conditioning branch: the uploaded context, and the
/// per-forward ResBlock bias vectors with the timestep embedding folded in.
/// Under classifier-free guidance there are two of these against one `Workspace`
/// the positive and negative branches differ in their context and, for SDXL,
/// in `adm` (whose pooled half is prompt-dependent), so their folded biases
/// differ too.
pub const Session = struct {
    arena: std.heap.ArenaAllocator,
    ctx_d: DeviceBuffer = none,
    ctx_seq: usize,
    /// SDXL's `y` micro-conditioning; borrowed, must outlive the session.
    adm: ?[]const f32,

    /// Every ResBlock's folded bias, packed end to end so one upload serves the
    /// whole forward, a `tensorUpload` per ResBlock would be ~30 staged copies
    /// and submits per step.
    bias_host: []f32,
    bias_d: DeviceBuffer = none,
    /// Element offset into `bias_host` per ResBlock, indexed by the block's
    /// ordinal in the graph walk (`sd_unet.ResBlockIter`'s order, which is the
    /// order `forward` visits them in).
    ///
    /// NOT keyed by a host pointer, which is what this was first and is a trap:
    /// `Session.replaceDenoiser` reloads the UNet into a fresh arena, the whole
    /// point of the per-tensor divergence arm, so every weight pointer changes
    /// while this session stays alive, and a pointer-keyed lookup goes from
    /// correct to a null-unwrap panic. An ordinal is a property of the
    /// architecture, which a reload does not change.
    bias_off: []usize,

    /// Time embedding for this forward (host; 1280 wide).
    emb: []f32,
    /// Scratch for the two-layer time/label MLPs.
    mlp_hidden: []f32,
    mlp_scratch: []f32,

    norms: GnCats = .{},

    /// Every linear's dispatch, and what it decided for this checkpoint.
    lin: lin_gpu.Lin,
    lin_plan: lin_gpu.Plan,

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


        // Every allocation happens BEFORE the struct literal. `.arena = arena`
        // copies the arena's state as it stands, so anything a *later* field
        // initializer allocates is leaked, invisible except to the test
        // allocator. `dit.zig` and `clip_text.zig` both learned this.
        const bias_host = try alloc.alloc(f32, total);
        const emb = try alloc.alloc(f32, cfg.time_embed_dim);
        const mlp_hidden = try alloc.alloc(f32, cfg.time_embed_dim);
        const mlp_scratch = try alloc.alloc(f32, cfg.time_embed_dim);
        var l = try lin_gpu.Lin.init(gpa, ctx, lin_gpu.maxRows(u.device_lins), 0, 0);
        errdefer l.deinit();
        var lp = try lin_gpu.plan(ctx, u.device_lins, .{ .blockq = .auto }, "sd_unet_gpu");
        // ComfyUI's fp16 UNet: every dense weight on the f16 tensor cores.
        lp.dense = .f16;
        lp.lora = u.lora;

        var self: Session = .{
            .arena = arena,
            .ctx_seq = ctx_seq,
            .adm = adm,
            .bias_host = bias_host,
            .bias_off = offs.items,
            .emb = emb,
            .mlp_hidden = mlp_hidden,
            .mlp_scratch = mlp_scratch,
            .lin = l,
            .lin_plan = lp,
        };

        try ctx.ensureDeviceBuffer(&self.ctx_d, context.len * 4);
        try ctx.tensorUpload(self.ctx_d, std.mem.sliceAsBytes(context));
        try ctx.ensureDeviceBuffer(&self.bias_d, @max(total, 1) * 4);
        return self;
    }

    pub fn deinit(self: *Session, ctx: *Context) void {
        ctx.tensorDestroy(&self.ctx_d);
        ctx.tensorDestroy(&self.bias_d);
        self.norms.deinit(ctx, self.arena.allocator());
        self.lin.deinit();
        self.arena.deinit();
        self.* = undefined;
    }

};

/// GroupNorm weight ++ bias on the device, keyed by weight pointer and owned here:
/// `gn_apply` reads both from one binding and the checkpoint stores two tensors. Not
/// `Context.smallBuffer`, which would key on the host concatenation and serve its copy to
/// whatever the allocator puts at that address after it is freed.
pub const GnCats = struct {
    map: std.AutoHashMapUnmanaged(usize, DeviceBuffer) = .empty,

    pub fn get(self: *GnCats, ctx: *Context, gpa: std.mem.Allocator, w: []const f32, b: []const f32) !DeviceBuffer {
        const key = @intFromPtr(w.ptr);
        if (self.map.get(key)) |db| return db;
        const cat = try gpa.alloc(f32, w.len + b.len);
        defer gpa.free(cat);
        @memcpy(cat[0..w.len], w);
        @memcpy(cat[w.len..], b);
        var db = try ctx.tensorCreate(cat.len * 4);
        errdefer ctx.tensorDestroy(&db);
        try ctx.uploadRecorded(db, 0, std.mem.sliceAsBytes(cat));
        try self.map.put(gpa, key, db);
        return db;
    }

    /// After the last batch that reads them has been waited on.
    pub fn deinit(self: *GnCats, ctx: *Context, gpa: std.mem.Allocator) void {
        var it = self.map.valueIterator();
        while (it.next()) |db| ctx.tensorDestroy(db);
        self.map.deinit(gpa);
        self.* = undefined;
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
    /// The (h, w) each skip was stored at. The decoder cannot derive it: its own grid
    /// is one row or column larger wherever a downsample rounded up.
    skip_hw: [][2]usize,
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
    /// The storage of `cur`, `alt`, `t1`, `t2`, `stream`, `nb`, `ff`, `gated` and the skips.
    act: Dt = .f32,

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
            .act = streamDt(ctx, u),
            .skips = try gpa.alloc(DeviceBuffer, u.input_stages.len + 1),
            .skip_ch = try gpa.alloc(usize, u.input_stages.len + 1),
            .skip_hw = try gpa.alloc([2]usize, u.input_stages.len + 1),
        };
        errdefer {
            gpa.free(self.skips);
            gpa.free(self.skip_ch);
            gpa.free(self.skip_hw);
        }
        for (self.skips) |*s| s.* = none;
        @memset(self.skip_ch, 0);

        // Same sizing walk as `sd_unet.Workspace.init`: the attention and
        // feed-forward scratches are sized per ATTENDING level at its own
        // resolution and width, not by the outermost resolution times the
        // innermost width, a product no stage ever has (SDXL's outermost level
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
        // The middle block always attends, at the innermost resolution, which
        // the loop skips when that level carries no SpatialTransformer of its own
        // (SD1.5's fourth level).
        sizeAttn(cfg, h * w, cfg.model_channels * cfg.channel_mult[cfg.levels() - 1], ctx_seq, &attn, &ff, &gated, &pad_f32, &pad_f16, &scores, &part, &md);

        const ew = self.act.bytes();
        inline for (.{ "cur", "alt", "t1", "t2" }) |f| {
            try ctx.ensureDeviceBuffer(&@field(self, f), act * ew);
        }
        try ctx.ensureDeviceBuffer(&self.stream, attn * ew);
        try ctx.ensureDeviceBuffer(&self.nb, attn * ew);
        inline for (.{ "q", "k", "v", "ao" }) |f| {
            try ctx.ensureDeviceBuffer(&@field(self, f), attn * 4);
        }
        inline for (.{ "qh", "kh", "vh" }) |f| {
            try ctx.ensureDeviceBuffer(&@field(self, f), pad_f16);
        }
        try ctx.ensureDeviceBuffer(&self.apad, pad_f32);
        try ctx.ensureDeviceBuffer(&self.ff, ff * ew);
        try ctx.ensureDeviceBuffer(&self.gated, gated * ew);
        inline for (.{ "cur", "alt", "t1", "t2", "stream", "nb", "ff", "gated" }) |f| @field(self, f).dt = self.act;
        for (self.skips) |*sk| sk.dt = self.act;
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
    /// function of it, so `init` and `conv_gpu.conv2d` cannot disagree about the size,
    /// which matters because growing the band inside a recording batch would free
    /// memory that already recorded dispatches reference.
    fn sizePatch(n_out: usize, ci: usize) usize {
        const patch_len = 9 * ci;
        return conv_gpu.convBand(n_out, patch_len) * patch_len * 4;
    }

    pub fn deinit(self: *Workspace, ctx: *Context) void {
        inline for (@typeInfo(Workspace).@"struct".field_names, @typeInfo(Workspace).@"struct".field_types) |name, FieldType| {
            if (FieldType == DeviceBuffer) ctx.tensorDestroy(&@field(self, name));
        }
        for (self.skips) |*s| ctx.tensorDestroy(s);
        self.gpa.free(self.skips);
        self.gpa.free(self.skip_ch);
        self.gpa.free(self.skip_hw);
        self.* = undefined;
    }
};

// --- forward ----------------------------------------------------------------

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

    // Each ResBlock's timestep projection is folded into its first convolution's bias.
    try sd_unet.timeProjections(u, io, gpa, timestep, sess.adm, sess.bias_host, sess.bias_off, sess.emb, sess.mlp_hidden, sess.mlp_scratch);
    var it = sd_unet.ResBlockIter.init(u);
    var bi: usize = 0;
    while (it.next()) |rb| : (bi += 1) {
        for (sess.bias_host[sess.bias_off[bi]..][0..rb.out_ch], rb.in_conv.b.?) |*d, cb| d.* += cb;
    }
    if (sess.bias_host.len > 0) try ctx.tensorUpload(sess.bias_d, std.mem.sliceAsBytes(sess.bias_host));
    try ctx.tensorUpload(ws.xin, std.mem.sliceAsBytes(x));

    try ctx.beginBatch();
    var batched = true;
    errdefer if (batched) ctx.endBatch() catch {};
    try dev.body(Dev{ .u = u, .ctx = ctx, .sess = sess, .ws = ws }, lat_h, lat_w, cancel);
    batched = false;
    try ctx.endBatch();
    try ctx.tensorDownload(ws.eps, std.mem.sliceAsBytes(out));
}

/// `sd_unet_dev`'s ops on this backend.
const Dev = struct {
    u: *const sd_unet.UNet,
    ctx: *Context,
    sess: *Session,
    ws: *Workspace,

    pub fn groupNorm(d: Dev, dst: *DeviceBuffer, src: *const DeviceBuffer, n: usize, ch: usize, nw: sd_unet.GroupNormW, silu: bool) !void {
        const cfg = d.u.cfg;
        try groupNormInto(d.ctx, d.ws.gstat, d.ws.gmi, dst, src, n, ch, try d.sess.norms.get(d.ctx, d.sess.arena.allocator(), nw.w, nw.b), cfg.norm_groups, cfg.norm_eps, silu);
    }

    pub fn conv(d: Dev, dst: *DeviceBuffer, src: *const DeviceBuffer, h: usize, w: usize, cv: Conv2d, mode: SampleMode, out_hw: ?[2]usize) !void {
        try conv_gpu.conv2d(d.ctx, &d.ws.patch, dst, src, h, w, cv, mode, out_hw, null, 1.0);
        try convLora(d.ctx, d.sess, d.ws, dst, src, h, w, cv, mode, out_hw);
    }

    /// The timestep projection rides in on the convolution's bias (`forward` folds it).
    pub fn resConv(d: Dev, dst: *DeviceBuffer, src: *const DeviceBuffer, h: usize, w: usize, rb: sd_unet.ResBlock, ordinal: usize) !void {
        try conv_gpu.conv2d(d.ctx, &d.ws.patch, dst, src, h, w, rb.in_conv, .stride1, null, .{ .buf = d.sess.bias_d, .off = d.sess.bias_off[ordinal] }, 1.0);
        try convLora(d.ctx, d.sess, d.ws, dst, src, h, w, rb.in_conv, .stride1, null);
    }

    pub fn add(d: Dev, a: DeviceBuffer, b: DeviceBuffer, n: usize) !void {
        std.debug.assert(a.dt == b.dt);
        try d.ctx.opElt(gpu_context.eltAt("add", a.dt), a, b, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
    }

    pub fn concatSkip(d: Dev, dst: DeviceBuffer, a: DeviceBuffer, a_ch: usize, b: DeviceBuffer, b_ch: usize, n: usize) !void {
        std.debug.assert(a.dt == dst.dt and b.dt == dst.dt);
        const total = a_ch + b_ch;
        const op = gpu_context.eltAt("concat_ch", dst.dt);
        d.ctx.independent(2);
        try d.ctx.opElt(op, a, dst, null, null, .{ .u0 = @intCast(n * a_ch), .u1 = @intCast(a_ch), .u2 = @intCast(total), .u3 = 0 }, n * a_ch, 1, 1);
        try d.ctx.opElt(op, b, dst, null, null, .{ .u0 = @intCast(n * b_ch), .u1 = @intCast(b_ch), .u2 = @intCast(total), .u3 = @intCast(a_ch) }, n * b_ch, 1, 1);
    }

    pub fn setSkip(d: Dev, i: usize, src: *const DeviceBuffer, elems: usize, ch: usize, h: usize, w: usize) !void {
        const ws = d.ws;
        std.debug.assert(src.dt == ws.act and ws.skips[i].dt == ws.act);
        try d.ctx.ensureDeviceBuffer(&ws.skips[i], elems * ws.act.bytes());
        ws.skip_ch[i] = ch;
        ws.skip_hw[i] = .{ h, w };
        try d.ctx.opElt(gpu_context.eltAt("copy", ws.act), src.*, ws.skips[i], null, null, .{ .u0 = @intCast(elems) }, elems, 1, 1);
    }

    pub fn layerNorm(d: Dev, dst: *DeviceBuffer, src: *const DeviceBuffer, n: usize, ch: usize, nw: sd_unet.LayerNormW) !void {
        const ctx = d.ctx;
        std.debug.assert(dst.dt == src.dt);
        const wbuf: DeviceBuffer = .{ .buf = try ctx.smallBuffer(std.mem.sliceAsBytes(nw.w)), .mem = .null_handle, .size = 0 };
        const bbuf: DeviceBuffer = .{ .buf = try ctx.smallBuffer(std.mem.sliceAsBytes(nw.b)), .mem = .null_handle, .size = 0 };
        try ctx.opElt(gpu_context.eltAt("layernorm", src.dt), src.*, dst.*, wbuf, bbuf, .{ .u0 = @intCast(n), .u1 = @intCast(ch), .f0 = d.u.cfg.norm_eps }, n, 1, 1);
    }

    /// One linear through the dispatcher; `bias` null means the weight carries none.
    pub fn gemm(d: Dev, y: *DeviceBuffer, x: *const DeviceBuffer, m: usize, wt: Weight, bias: ?[]const f32) !void {
        const g = try d.sess.lin.prep(d.sess.lin_plan, x.*, m, wt.cols, &.{wt});
        try d.sess.lin.gemm(d.sess.lin_plan, g, y.*, wt, .{ .bias = bias });
    }

    /// One prep per linear, NOT `ctx.independent`: a coop GEMM is three dispatches
    /// sharing the Context's `x_h16` / `y_pad` scratch, and the group counts
    /// dispatches, so it would both clobber that scratch and drop the barriers inside
    /// the first GEMM.
    pub fn gemms(d: Dev, x: *const DeviceBuffer, m: usize, ys: []const *DeviceBuffer, wts: []const Weight) !void {
        std.debug.assert(ys.len == wts.len);
        for (ys, wts) |y, wt| try d.gemm(y, x, m, wt, null);
    }

    pub fn crossKv(d: Dev, wk: Weight, wv: Weight) !void {
        try d.gemms(&d.sess.ctx_d, d.sess.ctx_seq, &.{ &d.ws.k, &d.ws.v }, &.{ wk, wv });
    }

    pub fn selfAttn(d: Dev, n: usize, heads: usize, hd: usize) !void {
        try File.selfAttn(d.ctx, d.ws, n, heads, hd);
    }

    pub fn crossAttn(d: Dev, n: usize, heads: usize, hd: usize) !void {
        const ws = d.ws;
        try d.ctx.opElt(.attn_cross, ws.q, ws.k, ws.v, ws.ao, .{
            .u0 = @intCast(n),
            .u1 = @intCast(heads),
            .u2 = @intCast(hd),
            .u3 = @intCast(d.sess.ctx_seq),
            .f0 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd))),
        }, n * heads, 1, 1);
    }

    pub fn geglu(d: Dev, src: DeviceBuffer, dst: DeviceBuffer, n: usize, inner: usize) !void {
        std.debug.assert(src.dt == dst.dt);
        try d.ctx.opElt(gpu_context.eltAt("geglu", src.dt), src, dst, null, null, .{ .u0 = @intCast(n * inner), .u1 = @intCast(inner) }, n * inner, 1, 1);
    }
};

/// Self-attention over the latent positions.
///
/// The cooperative-matrix path is built for head_dim 128, so narrower heads
/// are zero-padded up to it: exact (a zero dimension contributes nothing to a
/// dot product, and V's zero columns give output columns `head_unpad` drops) but
/// it does `128/head_dim` times the arithmetic of the true shape. SD1.5's
/// outermost level pays 3.2x and SDXL pays 2x. The fix is a head_dim-parameterized
/// `buildGemmAttnOut` / `buildFlashAttn`, not a change here.
///
/// head_dim 160, SD1.5's two innermost levels, does not fit a 128-wide tile at
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

    // Scores -> two-pass softmax -> P@V, in batches of heads so the O(seq^2) scores
    // plane stays bounded. The batch is `Workspace.init`'s, from the same function,
    // deliberately: growing the scratch inside the recording batch would free memory
    // that already recorded dispatches reference.
    try attn_gpu.core(ctx, .{ .s = ws.s, .part = ws.part, .md = ws.md }, ws.qh, ws.kh, ws.vh, ws.apad, .{
        .q_rows = n,
        .kv_rows = n,
        .heads = heads,
        .kv_heads = heads,
        .hd = tc_head_dim,
        .max_hb = headsPerBatch(seq_pad, heads),
        .nchunks = nchunks,
    }, null);

    try ctx.opElt(.head_unpad, ws.apad, ws.ao, null, null, .{
        .u0 = @intCast(n * heads * hd),
        .u1 = @intCast(hd),
        .u2 = tc_head_dim,
        .u4 = @intCast(heads),
    }, n * heads * hd, 1, 1);
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
    /// f32 weight ++ bias, `GnCats.get`.
    cbuf: DeviceBuffer,
    groups: usize,
    eps: f32,
    silu: bool,
) !void {
    // `src` and `dst` at one `dt`; the statistics, weight and bias are f32 whatever it is.
    std.debug.assert(cbuf.size >= 2 * ch * 4 and dst.dt == src.dt);
    const per_group = ch / groups;

    try ctx.opElt(gpu_context.eltAt("gn_stats", src.dt), src.*, null, null, gstat, .{
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
    try ctx.opElt(gpu_context.eltAt("gn_apply", src.dt), src.*, dst.*, cbuf, gmi, .{
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



/// A convolution's LoRA hits, onto its output (`cv`'s bias in it), the twin of
/// `sd_unet_cuda.convLora`. A DoRA's share of the base product goes first; a 1x1 is then
/// the dense sidecar over pixels. A 3x3's factor `A` is itself a convolution to `rank`
/// channels (f16 in the file's order) whose `B` accumulates back out; a LoKr term's
/// `w2` convolves each input channel group (`kronGroups`); a diff is an extra
/// convolution.
fn convLora(ctx: *Context, sess: *const Session, ws: *Workspace, dst: *DeviceBuffer, src: *const DeviceBuffer, h: usize, w: usize, cv: Conv2d, mode: SampleMode, out_hw: ?[2]usize) !void {
    const stack = sess.lin_plan.lora orelse return;
    const cw = cv.gemmWeight();
    const hits = stack.forWeight(cw);
    if (hits.len == 0) return;
    const oh = if (out_hw) |o| o[0] else switch (mode) {
        .stride1 => h,
        .upsample2x => 2 * h,
        .stride2 => (h + 1) / 2,
    };
    const ow = if (out_hw) |o| o[1] else switch (mode) {
        .stride1 => w,
        .upsample2x => 2 * w,
        .stride2 => (w + 1) / 2,
    };
    const n = oh * ow;
    const scratch = sess.lin.lora;
    const zb = sess.lin.zero_bias;
    for (hits) |hit| {
        const strength = stack.files.items[hit.file].strength;
        if (strength == 0) continue;
        const t = hit.target;
        if (t.dora) |d| {
            const ratio = try lora_gpu.hostVec(ctx, d.ratio);
            if (d.on_out) {
                const b: ?DeviceBuffer = if (cv.b) |bb| try lora_gpu.hostVec(ctx, bb) else null;
                try lora_gpu.doraRows(ctx, dst.*, ratio, b, n, cv.co, strength);
            } else {
                // W ((d - 1) x): the base convolution, bias-free, on a rescaled input.
                var xd = try scratch.inPlane(ctx, h * w * cv.ci * 4, .f32);
                try lora_gpu.doraCols(ctx, xd, src.*, ratio, h * w, cv.ci);
                var o = try scratch.outPlane(ctx, n * cv.co);
                var base = cv;
                // WHOLE, never sliced: `smallBuffer` caches by host pointer.
                base.b = zb;
                try conv_gpu.conv2d(ctx, &ws.patch, &o, &xd, h, w, base, mode, out_hw, null, 1.0);
                try lora_gpu.addPlane(ctx, dst.*, o, n, cv.co, cv.co, 0, strength);
            }
        }
        if (cv.k == 1) {
            try lora_gpu.applyTarget(ctx, scratch, zb, dst.*, src.*, n, t, strength, 0, cv.co);
            continue;
        }
        for (t.factors) |f| {
            std.debug.assert(f.a.dtype == .f16 and f.out_off == 0 and f.b.rows == cv.co);
            const r = f.rank();
            const a: Conv2d = .{
                .w = .{ .torch = .{ .w = f.a } },
                .b = zb,
                .co = r,
                .ci = cv.ci,
                .k = cv.k,
                .stride = cv.stride,
                .pad = cv.pad,
            };
            try scratch.ensure(ctx, n * r);
            try conv_gpu.conv2d(ctx, &ws.patch, &scratch.lo, src, h, w, a, mode, out_hw, null, 1.0);
            try lora_gpu.accumulate(ctx, scratch, dst.*, n, f.b, r, t.scale * strength * f.gain);
        }
        for (t.kron) |k| try kronGroups(ctx, sess, ws, dst.*, src, h, w, cv, mode, out_hw, n, k, strength * t.scale * k.gain);
        for (t.diffs) |df| {
            std.debug.assert(df.w.dtype == .f16);
            const dc: Conv2d = .{ .w = .{ .torch = .{ .w = df.w } }, .b = zb, .co = df.w.rows, .ci = cv.ci, .k = cv.k, .stride = cv.stride, .pad = cv.pad };
            var o = try scratch.outPlane(ctx, n * df.w.rows);
            try conv_gpu.conv2d(ctx, &ws.patch, &o, src, h, w, dc, mode, out_hw, null, 1.0);
            try lora_gpu.addPlane(ctx, dst.*, o, n, df.w.rows, cv.co, df.out_off, strength);
        }
    }
}

/// One LoKr term over a 3x3 convolution, as `sd_unet_cuda.kronGroups`: every group in
/// one launch over a zero-separated stack where the conv allows it, else each group
/// into its own slice of one plane, each slice at a storage-aligned offset.
fn kronGroups(ctx: *Context, sess: *const Session, ws: *Workspace, dst: DeviceBuffer, src: *const DeviceBuffer, h: usize, w: usize, cv: Conv2d, mode: SampleMode, out_hw: ?[2]usize, n: usize, k: lora_mod.Kron, g: f32) !void {
    std.debug.assert(k.w2.dtype == .f16 and cv.ci % k.im == 0);
    const scratch = sess.lin.lora;
    const in_n = cv.ci / k.im;
    const ok = k.w2.rows;
    const gc: Conv2d = .{ .w = .{ .torch = .{ .w = k.w2 } }, .b = sess.lin.zero_bias, .co = ok, .ci = in_n, .k = cv.k, .stride = cv.stride, .pad = cv.pad };
    const w1 = try lora_gpu.hostVec(ctx, k.w1);
    const eb = src.dt.bytes();
    if (mode != .stride2 and cv.k == 3 and cv.pad == 1 and out_hw == null) {
        const hs = k.im * (h + 1) - 1;
        var st = try scratch.inPlane(ctx, k.im * (h + 1) * w * in_n * eb, src.dt);
        try lora_gpu.kronStack(ctx, st, src.*, k.im, h, w, in_n, cv.ci, 0);
        const up: usize = if (mode == .upsample2x) 2 else 1;
        var out = try scratch.outPlane(ctx, up * hs * up * w * ok);
        try conv_gpu.conv2d(ctx, &ws.patch, &out, &st, hs, w, gc, mode, null, null, 1.0);
        // Group j's output starts at row up * j * (h + 1) of an up * w wide plane.
        return lora_gpu.kronMix(ctx, dst, out, w1, n, k.ol, ok, k.im, cv.co, k.out_off, ok, up * (h + 1) * up * w * ok, g);
    }
    const gstride = std.mem.alignForward(usize, n * ok * 4, ctx.storage_align) / 4;
    const out = try scratch.outPlane(ctx, k.im * gstride);
    var st = try scratch.inPlane(ctx, (h + 1) * w * in_n * eb, src.dt);
    for (0..k.im) |j| {
        try lora_gpu.kronStack(ctx, st, src.*, 1, h, w, in_n, cv.ci, j);
        var oj = try ctx.view(out, j * gstride * 4, n * ok * 4);
        try conv_gpu.conv2d(ctx, &ws.patch, &oj, &st, h, w, gc, mode, out_hw, null, 1.0);
    }
    try lora_gpu.kronMix(ctx, dst, out, w1, n, k.ol, ok, k.im, cv.co, k.out_off, ok, gstride, g);
}

/// Whether this context can run the UNet `u` and its LoRA stack. A dense factor is
/// bf16, which both sidecar GEMMs read; a 3x3 conv's `A` is f16 (`lora_sd`) and runs
/// as a convolution.
pub fn supported(ctx: *Context, u: *const sd_unet.UNet) bool {
    if (!lin_gpu.runs(ctx, u.device_lins, .{ .blockq = .auto }, u.lora, "sd_unet_gpu")) return false;
    const stack = u.lora orelse return true;
    for (stack.files.items) |f| {
        var it = f.side.index.valueIterator();
        while (it.next()) |t| for (t.factors) |fa| {
            const dense = fa.a.dtype == .bf16;
            if (fa.b.dtype != .bf16 or !(dense or fa.a.dtype == .f16)) {
                std.log.warn("sd_unet_gpu: LoRA {s} has a factor this backend has no GEMM for; the UNet runs on the CPU instead", .{t.tag});
                return false;
            }
        };
    }
    return true;
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
    return Context.init(gpa, std.testing.io) catch error.SkipZigTest;
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
        .skip_hw = try gpa.alloc([2]usize, 0),
    };
    defer ws.deinit(ctx);
    try ctx.ensureDeviceBuffer(&ws.gstat, groups * gn_chunks * 3 * 4);
    try ctx.ensureDeviceBuffer(&ws.gmi, groups * 2 * 4);
    var src: DeviceBuffer = none;
    defer ctx.tensorDestroy(&src);
    var dst: DeviceBuffer = none;
    defer ctx.tensorDestroy(&dst);
    try ctx.ensureDeviceBuffer(&dst, n * ch * 4);
    var cats: GnCats = .{};
    defer cats.deinit(ctx, gpa);
    const cbuf = try cats.get(ctx, gpa, cat[0..ch], cat[ch..]);

    // Two regimes, because they are bounded by different things.
    //
    // `mean 0` is what a UNet activation actually looks like, and there the
    // device result tracks the f64 CPU reference to f32 rounding.
    //
    // `mean 400` is the regime the CPU implementation's comment warns about (a
    // late VAE decoder block), and its error floor is NOT the statistics
    // algorithm: `x - mean` cancels ~400 down to ~1, so f32's 6e-8 relative
    // representation of `x` itself becomes ~2e-5 absolute in a quantity of size
    // 1. Welford is still what makes even this much work, the shifted
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
            try groupNormInto(ctx, ws.gstat, ws.gmi, &dst, &src, n, ch, cbuf, groups, 1e-5, silu);
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

test "a group norm's weights are read again after the GnCats holding them is gone" {
    // One decode after another hands its norms the same freed host addresses, and on a
    // second VAE those hold different weights.
    const gpa = testing.allocator;
    const io = testing.io;
    const ctx = try gpuCtx(gpa, io);
    defer ctx.deinit();

    var prng = std.Random.DefaultPrng.init(11);
    const rand = prng.random();
    const groups = 32;
    const n = 64;
    const ch = 128;
    const x = try gpa.alloc(f32, n * ch);
    defer gpa.free(x);
    for (x) |*v| v.* = rand.floatNorm(f32);
    const wb = try gpa.alloc(f32, 2 * ch);
    defer gpa.free(wb);

    var ws: Workspace = .{
        .gpa = gpa,
        .skips = try gpa.alloc(DeviceBuffer, 0),
        .skip_ch = try gpa.alloc(usize, 0),
        .skip_hw = try gpa.alloc([2]usize, 0),
    };
    defer ws.deinit(ctx);
    try ctx.ensureDeviceBuffer(&ws.gstat, groups * gn_chunks * 3 * 4);
    try ctx.ensureDeviceBuffer(&ws.gmi, groups * 2 * 4);
    var src: DeviceBuffer = none;
    defer ctx.tensorDestroy(&src);
    var dst: DeviceBuffer = none;
    defer ctx.tensorDestroy(&dst);
    try ctx.ensureDeviceBuffer(&dst, n * ch * 4);
    try ctx.tensorUpload(src_or(&src, ctx, x), std.mem.sliceAsBytes(x));
    const got = try gpa.alloc(f32, n * ch);
    defer gpa.free(got);
    const want = try gpa.alloc(f32, n * ch);
    defer gpa.free(want);

    for (0..2) |round| {
        for (wb) |*v| v.* = rand.floatNorm(f32);
        var cats: GnCats = .{};
        defer cats.deinit(ctx, gpa);
        const cbuf = try cats.get(ctx, gpa, wb[0..ch], wb[ch..]);
        try groupNormInto(ctx, ws.gstat, ws.gmi, &dst, &src, n, ch, cbuf, groups, 1e-5, false);
        try ctx.tensorDownload(dst, std.mem.sliceAsBytes(got));
        ops.norm.groupNorm(want, x, ch, groups, wb[0..ch], wb[ch..], 1e-5);
        const err = relL2(want, got);
        errdefer std.debug.print("round {d}: rel L2 {d:.8}\n", .{ round, err });
        try testing.expect(err < 2e-6);
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
        .skip_hw = try gpa.alloc([2]usize, 0),
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
        // `selfAttn` no longer grows these, `Workspace.init` pre-sizes them so
        // nothing reallocates inside a recording batch, so the test has to.
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
