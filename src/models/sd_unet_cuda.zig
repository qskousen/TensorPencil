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
//!    exactly, for self- and cross-attention alike. The hand-PTX arm cannot: its
//!    P@V GEMM tiles the head dimension in 128-wide blocks, so narrower heads are
//!    zero-padded up to 128 (and 160 up to 256) exactly as the Vulkan path does.
//!    Zero dimensions contribute nothing to a dot product and V's zero columns give
//!    output columns `opHeadUnpad` drops, so the padding is exact, it just costs
//!    arithmetic. This is the one place the two CUDA arms do not share a shape.
//!
//! 2. The ResBlock timestep projection is added, not folded. Vulkan folds it
//!    into the convolution bias for free; the CUDA GEMM entry points take a host
//!    bias slice, so folding would mean reworking that plumbing for a ~2% pass.
//!    See `opAddBiasRows`.
//!
//! Numerics are 16-bit tensor cores for the GEMMs and f32 elsewhere, so this is not
//! bit-identical to `sd_unet.forward`; the CPU path remains the reference. Under
//! `cuda` the activations are additionally STORED at 16 bits (`Workspace.act`),
//! which is a change of buffer width, not of arithmetic.

const std = @import("std");
const sd_unet = @import("sd_unet.zig");
const conv_cuda = @import("conv_cuda.zig");
const dev = @import("sd_unet_dev.zig");
const lin_cuda = @import("lin_cuda.zig");
const lora_cuda = @import("lora_cuda.zig");
const lora_mod = @import("lora.zig");
const ops = @import("tp_ops");
const tp_core = @import("tp_core");
const cuda = @import("tp_gpu").cuda;

const File = @This();
const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;
const dual_table = @import("tp_gpu").dual_table;
const Dt = cuda.backend.Dt;
const Weight = ops.matmul.Weight;
const Conv2d = ops.conv.Conv2d;
const Config = sd_unet.Config;


/// Partials per group in the GroupNorm statistics pass, which is the CUDA
/// kernel's parallelism knob: `gn_stats` runs one 256-thread block per partial,
/// so this is `groups * gn_chunks` blocks. Too few starves the card, too many
/// leaves each block reducing almost nothing. The Vulkan path's value is its own;
/// the two kernels partition the reduction differently and always did.
const gn_chunks: usize = 32;


/// The head width the hand-PTX attention tiles its P@V GEMM in.
///
/// `launchHgemmB` launches `grid.x = n / 128` with `n = head_dim`, so a
/// narrower head asks for a zero-sized grid (`CUDA_ERROR_INVALID_VALUE`) and a
/// non-multiple silently computes only part of the head. Every SD head width
/// trips one or the other: 40, 64 and 80 the former, 160 the latter.
const pv_tile: usize = 128;

pub const SampleMode = dev.SampleMode;

/// The dtype this UNet's activations are stored at on this backend: `UNet.act`, else
/// f16 where `Config.act_f16` says, else f32. Session and Workspace must agree: the
/// cross-attention K/V projections read the Session's context at the Workspace's width.
///
/// A 16-bit stream and an activation capture cannot both be on, because `probeInput`
/// reads `x` as f32. A capture is the rarer and the correctness-critical one, so it
/// wins: with a probe installed the whole UNet drops back to f32 rather than skipping
/// the GEMMs it cannot read, which would lose every `transformer_blocks` linear on
/// SDXL (733 of 834 layers) while still reporting a successful capture.
pub fn streamDt(be: *const Backend, u: *const sd_unet.UNet) Dt {
    return streamDtFor(be, u, if (u.act) |a| switch (a) {
        .f16 => .f16,
        .bf16 => .bf16,
        else => .f32,
    } else if (u.cfg.act_f16) .f16 else .f32);
}

/// What the stream would run as if `want` were asked for.
pub fn streamDtFor(be: *const Backend, u: *const sd_unet.UNet, want: Dt) Dt {
    if (want == .f32 or ops.matmul.probe != null) return .f32;
    // A route that reads only f32 activations (fp8, NVFP4, a GGUF format on a
    // dequant route) puts the whole stream at f32.
    const p = planFor(be, u) catch return .f32;
    return if (lin_cuda.takes16(p, u.device_lins)) want else .f32;
}

/// The dispatcher plan for this UNet's linears: dense weights on the f16 tensor
/// cores, as this family has always computed them.
fn planFor(be: *const Backend, u: *const sd_unet.UNet) !lin_cuda.Plan {
    _ = be;
    var p = try lin_cuda.plan(u.device_lins, .{ .blockq = lin_cuda.blockq_gemm }, "sd_unet cuda");
    p.dense = .f16;
    p.lora = u.lora;
    return p;
}

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
    /// The UNet's LoRA sidecars, or null: `run` and `convLora` apply them.
    lora: ?*const lora_mod.Stack = null,
    /// The dispatcher plan every linear runs under.
    plan: lin_cuda.Plan,

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
            .lora = u.lora,
            .plan = try planFor(be, u),
        };

        // The context is a GEMM source like any other, so it is stored at the
        // activation stream's width. Converted once per image on the host: it is
        // 77 rows, against a forward that reads it 32 times.
        const sdt = streamDt(be, u);
        if (sdt != .f32) {
            const half = try alloc.alloc(u16, context.len);
            for (context, half) |v, *o| o.* = if (sdt == .f16) @bitCast(@as(f16, @floatCast(v))) else tp_core.dtype.f32ToBf16(v);
            try be.ensureDeviceBuffer(&self.ctx_d, half.len * 2);
            try be.tensorUpload(self.ctx_d, std.mem.sliceAsBytes(half));
            self.ctx_d.dt = sdt;
        } else {
            try be.ensureDeviceBuffer(&self.ctx_d, context.len * 4);
            try be.tensorUpload(self.ctx_d, std.mem.sliceAsBytes(context));
        }
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
    /// A 3x3 conv sidecar's `[pixels][rank]` plane. `conv_cuda.conv2d` writes its
    /// `dst` without growing it, so `convLora` sizes this first.
    lora_lo: Buf = .{},
    cur: Buf = .{},
    alt: Buf = .{},
    t1: Buf = .{},
    t2: Buf = .{},
    patch: Buf = .{},
    skips: []Buf,
    skip_ch: []usize,
    /// The (h, w) each skip was stored at. The decoder cannot derive it: its own grid
    /// is one row or column larger wherever a downsample rounded up.
    skip_hw: [][2]usize,
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

    /// The activation storage (`streamDt`). Every kernel still computes in f32 and
    /// every per-channel parameter stays f32; a 16-bit stream only narrows the big
    /// buffers, which halves the bandwidth-bound half of the step (the norms, the
    /// adds, the GEGLU) and deletes the per-GEMM and per-attention conversions.
    ///
    /// On the hand-PTX arm Q, K, V and the attention output stay f32 (`attnDt`): its
    /// attention and head padding read f32, and each projection converts on the way.
    ///
    /// ⚠️ f16 tops out at 65504. The SD UNet trunk stays under it (ComfyUI runs these
    /// models in fp16 by default), but that is an empirical fact about these
    /// architectures, which is why f16 is `sd_unet.Config.act_f16` and not a backend
    /// default. `sd-cuda-test`'s whole-forward check is what catches an overflow.
    act: Dt = .f32,

    fn actDt(self: *const Workspace) Dt {
        return self.act;
    }

    fn attnDt(self: *const Workspace, be: *const Backend) Dt {
        return if (be.kernels == .libs) self.act else .f32;
    }

    /// Bytes per activation element.
    fn aw(self: *const Workspace) usize {
        return self.actDt().bytes();
    }

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
            .act = streamDt(be, u),
            .skips = try gpa.alloc(Buf, u.input_stages.len + 1),
            .skip_ch = try gpa.alloc(usize, u.input_stages.len + 1),
            .skip_hw = try gpa.alloc([2]usize, u.input_stages.len + 1),
        };
        errdefer {
            gpa.free(self.skips);
            gpa.free(self.skip_ch);
            gpa.free(self.skip_hw);
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
            act = @max(act, pad128(h * w) * ch * 3);
            const plen = 9 * ch * 3;
            patch = @max(patch, conv_cuda.convBand(h * w, plen) * plen * 4);
            if (cfg.attn_levels[level]) sizeAttn(be, cfg, h * w, ch, ctx_seq, &attn, &ff, &gated, &padded);
            if (level + 1 < cfg.levels()) {
                h = (h + 1) / 2;
                w = (w + 1) / 2;
            }
        }
        sizeAttn(be, cfg, h * w, cfg.model_channels * cfg.channel_mult[cfg.levels() - 1], ctx_seq, &attn, &ff, &gated, &padded);

        const ew = self.aw();
        inline for (.{ "cur", "alt", "t1", "t2" }) |f| try be.ensureDeviceBuffer(&@field(self, f), act * ew);
        inline for (.{ "stream", "nb" }) |f| try be.ensureDeviceBuffer(&@field(self, f), attn * ew);
        inline for (.{ "q", "k", "v", "ao" }) |f| try be.ensureDeviceBuffer(&@field(self, f), attn * self.attnDt(be).bytes());
        inline for (.{ "qp", "kp", "vp", "op" }) |f| try be.ensureDeviceBuffer(&@field(self, f), padded);
        try be.ensureDeviceBuffer(&self.ff, ff * ew);
        try be.ensureDeviceBuffer(&self.gated, gated * ew);
        inline for (.{ "cur", "alt", "t1", "t2", "stream", "nb", "ff", "gated" }) |f| @field(self, f).dt = self.actDt();
        inline for (.{ "q", "k", "v", "ao" }) |f| @field(self, f).dt = self.attnDt(be);
        try be.ensureDeviceBuffer(&self.patch, patch);
        // The block-quant and W4A8 decode scratches at their final size, so no GEMM
        // grows one mid-forward.
        try lin_cuda.presize(be, try planFor(be, u), u.device_lins);
        try be.ensureDeviceBuffer(&self.gstat, cfg.norm_groups * gn_chunks * 3 * 4);
        try be.ensureDeviceBuffer(&self.gmi, cfg.norm_groups * 2 * 4);
        try be.ensureDeviceBuffer(&self.xin, lat_h * lat_w * cfg.channels * 4);
        try be.ensureDeviceBuffer(&self.eps, lat_h * lat_w * cfg.channels * 4);
        return self;
    }

    fn sizeAttn(be: *const Backend, cfg: Config, n: usize, ch: usize, ctx_seq: usize, attn: *usize, ff: *usize, gated: *usize, padded: *usize) void {
        // Rows rounded to 128: the dispatcher's int8 routes store whole tiles.
        const np = pad128(n);
        attn.* = @max(attn.*, @max(np, pad128(ctx_seq)) * ch);
        ff.* = @max(ff.*, np * ch * 8);
        gated.* = @max(gated.*, np * ch * 4);
        const heads = cfg.headsAt(ch);
        const hd = ch / heads;
        const hp = attnHd(be, hd);
        if (hp != hd) padded.* = @max(padded.*, n * heads * hp * 4);
    }

    pub fn deinit(self: *Workspace, be: *Backend) void {
        inline for (@typeInfo(Workspace).@"struct".field_names, @typeInfo(Workspace).@"struct".field_types) |name, FieldType| {
            if (FieldType == Buf) be.tensorDestroy(&@field(self, name));
        }
        for (self.skips) |*s| be.tensorDestroy(s);
        self.gpa.free(self.skips);
        self.gpa.free(self.skip_ch);
        self.gpa.free(self.skip_hw);
        self.* = undefined;
    }
};


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

    try sd_unet.timeProjections(u, io, gpa, timestep, sess.adm, sess.proj_host, sess.proj_off, sess.emb, sess.mlp_hidden, sess.mlp_scratch);
    if (sess.proj_host.len > 0) try be.tensorUpload(sess.proj_d, std.mem.sliceAsBytes(sess.proj_host));
    try be.tensorUpload(ws.xin, std.mem.sliceAsBytes(x));

    try be.beginBatch();
    errdefer if (be.batching()) be.abortBatch();
    try dev.body(Dev{ .u = u, .be = be, .sess = sess, .ws = ws }, lat_h, lat_w, cancel);
    try be.endBatch();
    try be.tensorDownload(ws.eps, std.mem.sliceAsBytes(out));
}

/// `sd_unet_dev`'s ops on this backend.
const Dev = struct {
    u: *const sd_unet.UNet,
    be: *Backend,
    sess: *Session,
    ws: *Workspace,

    pub fn groupNorm(d: Dev, dst: *Buf, src: *const Buf, n: usize, ch: usize, nw: sd_unet.GroupNormW, silu: bool) !void {
        const cfg = d.u.cfg;
        const cat = try d.sess.normBuf(d.be, nw);
        try d.be.opGroupNorm(src.*, dst.*, cat, d.ws.gstat, d.ws.gmi, n, ch, cfg.norm_groups, gn_chunks, cfg.norm_eps, silu);
    }

    pub fn conv(d: Dev, dst: *Buf, src: *const Buf, h: usize, w: usize, cv: Conv2d, mode: SampleMode, out_hw: ?[2]usize) !void {
        try conv_cuda.conv2d(d.be, &d.ws.patch, dst, src, h, w, cv, mode, out_hw, 1.0);
        try convLora(d.be, d.sess, d.ws, dst, src, h, w, cv, mode, out_hw);
    }

    /// The timestep projection is a second pass over the output; folding it into the
    /// conv bias would need a per-forward device bias the GEMM entry points do not take.
    pub fn resConv(d: Dev, dst: *Buf, src: *const Buf, h: usize, w: usize, rb: sd_unet.ResBlock, ordinal: usize) !void {
        try d.conv(dst, src, h, w, rb.in_conv, .stride1, null);
        try d.be.opAddBiasRows(dst.*, d.sess.proj_d, h * w, rb.out_ch, d.sess.proj_off[ordinal]);
    }

    pub fn add(d: Dev, a: Buf, b: Buf, n: usize) !void {
        try d.be.opAdd(a, b, n);
    }

    pub fn concatSkip(d: Dev, dst: Buf, a: Buf, a_ch: usize, b: Buf, b_ch: usize, n: usize) !void {
        const total = a_ch + b_ch;
        try d.be.opConcatCh(a, dst, n, a_ch, total, 0);
        try d.be.opConcatCh(b, dst, n, b_ch, total, a_ch);
    }

    pub fn setSkip(d: Dev, i: usize, src: *const Buf, elems: usize, ch: usize, h: usize, w: usize) !void {
        const ws = d.ws;
        try d.be.ensureDeviceBuffer(&ws.skips[i], elems * ws.aw());
        ws.skips[i].dt = ws.actDt();
        ws.skip_ch[i] = ch;
        ws.skip_hw[i] = .{ h, w };
        try d.be.opCopyOff(ws.skips[i], 0, src.*, 0, elems);
    }

    pub fn layerNorm(d: Dev, dst: *Buf, src: *const Buf, n: usize, ch: usize, nw: sd_unet.LayerNormW) !void {
        try d.be.opLayerNorm(src.*, dst.*, nw.w, nw.b, n, ch, d.u.cfg.norm_eps);
    }

    /// One linear through `lin_cuda`: the route from the weight's storage, the
    /// activation at the stream's width, the LoRA sidecar inside `gemm`.
    pub fn gemm(d: Dev, y: *Buf, x: *const Buf, m: usize, wt: Weight, bias: ?[]const f32) !void {
        const g = try lin_cuda.prep(d.be, d.sess.plan, x.*, m, wt.cols, &.{wt});
        try lin_cuda.gemm(d.be, d.sess.plan, g, y.*, wt, bias);
    }

    /// Under ONE activation prep: an int8 route rotates and quantizes `x` once for the
    /// group instead of once per linear.
    pub fn gemms(d: Dev, x: *const Buf, m: usize, ys: []const *Buf, wts: []const Weight) !void {
        std.debug.assert(ys.len == wts.len);
        const g = try lin_cuda.prep(d.be, d.sess.plan, x.*, m, wts[0].cols, wts);
        for (ys, wts) |y, wt| try lin_cuda.gemm(d.be, d.sess.plan, g, y.*, wt, null);
    }

    pub fn crossKv(d: Dev, wk: Weight, wv: Weight) !void {
        try d.gemms(&d.sess.ctx_d, d.sess.ctx_seq, &.{ &d.ws.k, &d.ws.v }, &.{ wk, wv });
    }

    pub fn selfAttn(d: Dev, n: usize, heads: usize, hd: usize) !void {
        try File.selfAttn(d.be, d.ws, n, heads, hd, attnScale(hd));
    }

    pub fn crossAttn(d: Dev, n: usize, heads: usize, hd: usize) !void {
        const ws = d.ws;
        try d.be.opAttnCross(ws.q, ws.k, ws.v, ws.ao, n, d.sess.ctx_seq, heads, hd, attnScale(hd));
    }

    pub fn geglu(d: Dev, src: Buf, dst: Buf, n: usize, inner: usize) !void {
        try d.be.opGeglu(src, dst, n, inner);
    }
};

fn attnScale(hd: usize) f32 {
    return 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
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






fn pad128(n: usize) usize {
    return std.mem.alignForward(usize, n, 128);
}




/// A convolution's LoRA hits, onto its output (`cv`'s bias in it), the device twin of
/// `lora.Stack.applyHostConv`. A DoRA's share of the base product goes first; a 1x1 is
/// then the dense linear sidecar over pixels. A 3x3's factor `A` is itself a
/// convolution to `rank` channels (`lora_sd` keeps it f16 in the file's order) whose
/// `B` accumulates back out; a LoKr term's `w2` convolves each input channel group,
/// all groups in ONE launch where the conv allows it (`kronGroups`); a diff is an
/// extra convolution.
fn convLora(be: *Backend, sess: *Session, ws: *Workspace, dst: *Buf, src: *const Buf, h: usize, w: usize, cv: Conv2d, mode: SampleMode, out_hw: ?[2]usize) !void {
    const stack = sess.lora orelse return;
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
    for (hits) |hit| {
        const strength = stack.strengthOf(hit.file);
        if (strength == 0) continue;
        const d = hit.target.dora orelse continue;
        const ratio = try lora_cuda.hostVec(be, d.ratio);
        if (d.on_out) {
            const b: ?Buf = if (cv.b) |bb| try lora_cuda.hostVec(be, bb) else null;
            try be.opDoraRows(dst.*, ratio, b, n, cv.co, strength);
            continue;
        }
        // W ((d - 1) x): the base convolution, bias-free, on a rescaled input.
        var xd = try lora_cuda.inPlane(be, h * w * cv.ci * 4, .f32);
        try be.opDoraCols(xd, src.*, ratio, h * w, cv.ci);
        var t = try lora_cuda.outPlane(be, n * cv.co);
        var base = cv;
        base.b = sess.zeros;
        try conv_cuda.conv2d(be, &ws.patch, &t, &xd, h, w, base, mode, out_hw, 1.0);
        try lora_cuda.addPlane(be, dst.*, t, n, cv.co, cv.co, 0, strength);
    }
    if (cv.k == 1) return lin_cuda.sidecarDense(be, stack, dst.*, src.*, h * w, cw);
    for (hits) |hit| {
        const strength = stack.strengthOf(hit.file);
        if (strength == 0) continue;
        const t = hit.target;
        for (t.factors) |f| {
            std.debug.assert(f.a.dtype == .f16 and f.out_off == 0 and f.b.rows == cv.co);
            const r = f.rank();
            const a: Conv2d = .{
                .w = .{ .torch = .{ .w = f.a } },
                .b = sess.zeros,
                .co = r,
                .ci = cv.ci,
                .k = cv.k,
                .stride = cv.stride,
                .pad = cv.pad,
            };
            try be.ensureDeviceBuffer(&ws.lora_lo, n * r * 4);
            try conv_cuda.conv2d(be, &ws.patch, &ws.lora_lo, src, h, w, a, mode, out_hw, 1.0);
            var lws = try lora_cuda.borrowed(be, r, n * cv.co);
            try lora_cuda.accumulate(be, &lws, dst.*, ws.lora_lo, r, n, f.b, r, t.scale * strength * f.gain);
        }
        for (t.kron) |k| try kronGroups(be, sess, ws, dst.*, src, h, w, cv, mode, out_hw, n, k, strength * t.scale * k.gain);
        for (t.diffs) |df| {
            std.debug.assert(df.w.dtype == .f16);
            const dc: Conv2d = .{ .w = .{ .torch = .{ .w = df.w } }, .b = sess.zeros, .co = df.w.rows, .ci = cv.ci, .k = cv.k, .stride = cv.stride, .pad = cv.pad };
            var o = try lora_cuda.outPlane(be, n * df.w.rows);
            try conv_cuda.conv2d(be, &ws.patch, &o, src, h, w, dc, mode, out_hw, 1.0);
            try lora_cuda.addPlane(be, dst.*, o, n, df.w.rows, cv.co, df.out_off, strength);
        }
    }
}

/// One LoKr term over a 3x3 convolution: `w2` over each of the `im` input channel
/// groups, then `w1` across them. Stride 1 and the fused 2x upsample stack the groups
/// as images with a zero row between (`opKronStack`), which a padding-1 3x3 reads as
/// each image's own padding, so one convolution covers every group; a strided or
/// clipped output convolves each group into its own slice of one plane.
fn kronGroups(be: *Backend, sess: *Session, ws: *Workspace, dst: Buf, src: *const Buf, h: usize, w: usize, cv: Conv2d, mode: SampleMode, out_hw: ?[2]usize, n: usize, k: lora_mod.Kron, g: f32) !void {
    std.debug.assert(k.w2.dtype == .f16 and cv.ci % k.im == 0);
    const in_n = cv.ci / k.im;
    const ok = k.w2.rows;
    const gc: Conv2d = .{ .w = .{ .torch = .{ .w = k.w2 } }, .b = sess.zeros, .co = ok, .ci = in_n, .k = cv.k, .stride = cv.stride, .pad = cv.pad };
    const w1 = try lora_cuda.hostVec(be, k.w1);
    const eb = src.dt.bytes();
    if (mode != .stride2 and cv.k == 3 and cv.pad == 1 and out_hw == null) {
        const hs = k.im * (h + 1) - 1;
        var st = try lora_cuda.inPlane(be, k.im * (h + 1) * w * in_n * eb, src.dt);
        try be.opKronStack(st, src.*, k.im, h, w, in_n, cv.ci, 0);
        const up: usize = if (mode == .upsample2x) 2 else 1;
        var out = try lora_cuda.outPlane(be, up * hs * up * w * ok);
        try conv_cuda.conv2d(be, &ws.patch, &out, &st, hs, w, gc, mode, null, 1.0);
        // Group j's output starts at row up * j * (h + 1) of an up * w wide plane.
        return be.opKronMix(dst, out, w1, n, k.ol, ok, k.im, cv.co, k.out_off, ok, up * (h + 1) * up * w * ok, g);
    }
    var out = try lora_cuda.outPlane(be, k.im * n * ok);
    var st = try lora_cuda.inPlane(be, (h + 1) * w * in_n * eb, src.dt);
    for (0..k.im) |j| {
        try be.opKronStack(st, src.*, 1, h, w, in_n, cv.ci, j);
        var oj = out.slice(j * n * ok * 4, n * ok * 4);
        try conv_cuda.conv2d(be, &ws.patch, &oj, &st, h, w, gc, mode, out_hw, 1.0);
    }
    try be.opKronMix(dst, out, w1, n, k.ol, ok, k.im, cv.co, k.out_off, ok, n * ok, g);
}



