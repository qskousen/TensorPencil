//! GPU-resident Mage-VAE decode on Vulkan, the device twin of
//! `mage_vae.MageVae.decode` and the Vulkan sibling of `mage_vae_cuda`.
//!
//! That arm's header carries the structure, the host-side constants and why the
//! nine new kernels exist; they were written in `kernels/dual/` precisely so this
//! file could reach them. Three things differ here:
//!
//! 1. **No buffer views.** Vulkan buffers are opaque handles with no offset, so
//!    the NeRF band addresses its slice of the conditioning through `opMatmul`'s
//!    own `x_off` rather than through a pointer view.
//! 2. **Attention is `attn_scores` + `attn_out`**, the f32 register-tiled pair,
//!    with each 32x32 window as a HEAD and the windows batched to a scores
//!    budget. `attn_out` runs its own online softmax over the raw scores, so
//!    there is no softmax pass between the two; adding one exponentiates twice,
//!    which is finite, plausible and wrong.
//! 3. **The `s_embedder` concatenation is zeroed from the host.** The context has
//!    no device fill, and `scale_f32` by zero would read whatever `tensorCreate`
//!    left behind, which is NaN-by-zero rather than zero.
//!
//! Every weight is f32 after `mage_vae`'s loader, so there is no dtype dispatch
//! and nothing here narrows to f16: this codec's attention logits and residual
//! both reach magnitudes f16 cannot hold, and its head is 384 wide rather than
//! the 128 the tensor-core path tiles for.

const std = @import("std");
const mage_vae = @import("mage_vae.zig");
const conv_gpu = @import("conv_gpu.zig");
const mage_vae_dev = @import("mage_vae_dev.zig");
const lin_gpu = @import("lin_gpu.zig");
const sd_unet_gpu = @import("sd_unet_gpu.zig");
const ops = @import("tp_ops");
const gpu_context = @import("tp_gpu").context;
const dual_table = @import("tp_gpu").dual_table;

const MageVae = mage_vae.MageVae;
const Context = gpu_context.Context;
const Buf = gpu_context.DeviceBuffer;
const Dt = gpu_context.Dt;
const eltAt = gpu_context.eltAt;
const Conv2d = ops.conv.Conv2d;

const H = mage_vae.hidden; // 384
const HX = mage_vae.hidden_x; // 32
const FFN = mage_vae.ffn; // 1536
const PA = mage_vae.patch_area; // 256
const NERF_IN = mage_vae.nerf_in; // 99
const eps: f32 = 1e-6;
/// Chunks per group in the two-pass GroupNorm. It is `sd_unet_gpu`'s, not the
/// CUDA arm's 32: `groupNormInto` uses its own constant and only the caller sizes
/// `gstat`, so a local copy silently under-allocates it.
const gn_chunks = sd_unet_gpu.gn_chunks;

const none: Buf = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 };

/// Cap on the materialized window-attention scores plane. The plane is
/// `windows * 1024 * 1024` floats, so it grows with the latent area and the
/// windows batch to fit this.
const s_bytes_cap: usize = 512 << 20;

const Bufs = struct {
    x: Buf = none,
    t: Buf = none,
    u: Buf = none,
    wide: Buf = none,
    stage: Buf = none,
    rgb: Buf = none,
    patch: Buf = none,
    gstat: Buf = none,
    gmi: Buf = none,
    cat: Buf = none,
    cond: Buf = none,
    /// Windowed attention: the projections, the window-major gather, the k-major
    /// forms the scores kernel reads, and the scores plane.
    aq: Buf = none,
    ak: Buf = none,
    av: Buf = none,
    ao: Buf = none,
    wq: Buf = none,
    wk: Buf = none,
    wv: Buf = none,
    wo: Buf = none,
    qh: Buf = none,
    kh: Buf = none,
    s: Buf = none,
    mods: Buf = none,
    dct: Buf = none,
    gate: Buf = none,
    /// `col_mean`'s partial sums.
    part: Buf = none,
    mean: Buf = none,
    y: Buf = none,
    feat: Buf = none,
    nx: Buf = none,
    nt: Buf = none,
    cvec: Buf = none,
    nmod: Buf = none,
    pix: Buf = none,

    /// What the dial narrows: `mage_vae_cuda.Bufs.typed`.
    const typed = [_][]const u8{ "x", "t", "u", "wide", "cat", "cond", "y", "feat", "nt", "cvec", "nmod" };

    fn setAct(self: *Bufs, act: Dt) void {
        inline for (typed) |f| @field(self, f).dt = act;
    }

    fn deinit(self: *Bufs, ctx: *Context) void {
        inline for (@typeInfo(Bufs).@"struct".field_names) |name| ctx.tensorDestroy(&@field(self, name));
    }
};

/// Whether this context can run the decode at all. Every weight is f32 after the
/// loader, so the only question is that the model loaded.
pub fn supported(ctx: *Context, model: *const MageVae) bool {
    _ = ctx;
    return model.blocks.len != 0;
}

/// Widest im2col row any 3x3 convolution in the decode needs, so the band buffer
/// is sized before a batch opens. `ensureDeviceBuffer` growing it mid-batch would
/// free memory that recorded-but-unsubmitted dispatches still reference.
///
/// Walked rather than hardcoded: a conv that changes width cannot silently
/// outgrow the buffer.
fn maxPatchLen(model: *const MageVae) usize {
    var ci: usize = 0;
    const take = struct {
        fn f(acc: *usize, cv: Conv2d) void {
            if (cv.k == 3) acc.* = @max(acc.*, cv.ci);
        }
    }.f;
    take(&ci, model.cod.conv_in);
    take(&ci, model.cod.conv_out);
    inline for (.{ "r0", "r1", "r2" }) |f| {
        const r = @field(model.cod, f);
        take(&ci, r.conv1);
        take(&ci, r.conv2);
    }
    inline for (.{ "a0", "a1" }) |f| {
        const a = @field(model.cod, f);
        take(&ci, a.q);
        take(&ci, a.k);
        take(&ci, a.v);
        take(&ci, a.proj);
    }
    take(&ci, model.s_proj2);
    for (model.blocks) |*b| {
        take(&ci, b.conv1);
        take(&ci, b.conv3);
        take(&ci, b.conv4);
        take(&ci, b.conv5);
    }
    return 9 * ci;
}

/// Windows attended per batch, so the scores plane stays under the cap.
fn winsPerBatch(tokens: usize, n_win: usize, cap: usize) usize {
    return @max(1, @min(n_win, cap / @max(tokens * tokens * 4, 1)));
}

/// The CoD decoder alone: latent -> `[n][hidden]` conditioning. Caller frees.
///
/// A separate entry point so `mage-vae-vk-test` can compare it against the CPU
/// before the per-pixel pathway runs; the two halves fail for different reasons.
pub fn codForward(model: *const MageVae, ctx: *Context, gpa: std.mem.Allocator, z: []const f32, lat_h: usize, lat_w: usize, act: Dt) ![]f32 {
    const n = lat_h * lat_w;
    var bufs: Bufs = .{};
    defer bufs.deinit(ctx);
    var cats: sd_unet_gpu.GnCats = .{};
    defer cats.deinit(ctx, gpa);
    var lin = try lin_gpu.Lin.init(gpa, ctx, mage_vae.hidden_x * mage_vae.patch_area, 0, 0);
    defer lin.deinit();

    const d = mage_vae.attn_window;
    const nph = (lat_h + d - 1) / d;
    const npw = (lat_w + d - 1) / d;
    const n_win = nph * npw;
    const tokens = d * d;
    const wpb = winsPerBatch(tokens, n_win, s_bytes_cap);

    inline for (.{ "x", "t", "u", "cond", "aq", "ak", "av", "ao" }) |f| {
        @field(bufs, f) = try ctx.tensorCreate(n * H * 4);
    }
    bufs.stage = try ctx.tensorCreate(z.len * 4);
    bufs.gstat = try ctx.tensorCreate(mage_vae.norm_groups * gn_chunks * 3 * 4);
    bufs.gmi = try ctx.tensorCreate(mage_vae.norm_groups * 2 * 4);
    inline for (.{ "wq", "wk", "wv", "wo", "qh", "kh" }) |f| {
        @field(bufs, f) = try ctx.tensorCreate(tokens * n_win * H * 4);
    }
    bufs.s = try ctx.tensorCreate(wpb * tokens * tokens * 4);
    bufs.patch = try ctx.tensorCreate(conv_gpu.convBand(n, maxPatchLen(model)) * maxPatchLen(model) * 4);
    try ctx.tensorUpload(bufs.stage, std.mem.sliceAsBytes(z));
    bufs.setAct(act);

    try ctx.beginBatch();
    errdefer if (ctx.batching) ctx.abortBatch();
    try mage_vae_dev.cod(Dev.init(ctx, gpa, &lin, &bufs, &cats, npw, n_win, act), model, lat_h, lat_w);
    // `x` is dead here and holds `n * H` f32.
    var out32 = bufs.cond;
    if (act != .f32) {
        out32 = bufs.x;
        out32.dt = .f32;
        try ctx.opElt(gpu_context.eltCopy(act, .f32), bufs.cond, out32, null, null, .{ .u0 = @intCast(n * H) }, n * H, 1, 1);
    }
    try ctx.endBatch();

    const out = try gpa.alloc(f32, n * H);
    errdefer gpa.free(out);
    try ctx.tensorDownload(out32, std.mem.sliceAsBytes(out));
    return out;
}

fn traceInto(ctx: *Context, name: []const u8, x: Buf, n: usize) !void {
    const was = ctx.batching;
    if (was) try ctx.endBatch();
    defer if (was) ctx.beginBatch() catch {};
    const raw = try std.heap.smp_allocator.alloc(u8, n * x.dt.bytes());
    defer std.heap.smp_allocator.free(raw);
    try ctx.tensorDownload(x, raw);
    mage_vae_dev.mag.record(name, raw, x.dt);
}

/// Latent `[lat_h][lat_w][latent_channels]` -> RGB `[h][w][3]` in [-1, 1],
/// `h = lat_h * 16`. Caller frees.
pub fn decode(
    model: *const MageVae,
    ctx: *Context,
    io: std.Io,
    gpa: std.mem.Allocator,
    z: []const f32,
    lat_h: usize,
    lat_w: usize,
    act: Dt,
    cancel: ?*std.atomic.Value(bool),
) ![]f32 {
    const n = lat_h * lat_w;
    std.debug.assert(z.len == n * mage_vae.latent_channels);
    const img_h = lat_h * mage_vae.patch;
    const img_w = lat_w * mage_vae.patch;
    const cat_w = mage_vae.bottleneck + H;

    var bufs: Bufs = .{};
    defer bufs.deinit(ctx);
    var cats: sd_unet_gpu.GnCats = .{};
    defer cats.deinit(ctx, gpa);
    var lin = try lin_gpu.Lin.init(gpa, ctx, mage_vae.hidden_x * mage_vae.patch_area, 0, 0);
    defer lin.deinit();

    // `: usize` is load-bearing: without it `@min` narrows every size below.
    const band: usize = @min(mage_vae_dev.nerf_band, n);
    const d = mage_vae.attn_window;
    const nph = (lat_h + d - 1) / d;
    const npw = (lat_w + d - 1) / d;
    const n_win = nph * npw;
    const tokens = d * d;
    const wpb = winsPerBatch(tokens, n_win, s_bytes_cap);

    inline for (.{ "x", "t", "u", "cond", "aq", "ak", "av", "ao" }) |f| {
        @field(bufs, f) = try ctx.tensorCreate(n * H * 4);
    }
    bufs.wide = try ctx.tensorCreate(n * FFN * 4);
    bufs.stage = try ctx.tensorCreate(@max(z.len, n * cat_w) * 4);
    bufs.rgb = try ctx.tensorCreate(img_h * img_w * 3 * 4);
    bufs.cat = try ctx.tensorCreate(n * cat_w * 4);
    bufs.gstat = try ctx.tensorCreate(mage_vae.norm_groups * gn_chunks * 3 * 4);
    bufs.gmi = try ctx.tensorCreate(mage_vae.norm_groups * 2 * 4);
    bufs.gate = try ctx.tensorCreate(H * 4);
    bufs.part = try ctx.tensorCreate((std.math.divCeil(usize, n, dual_table.col_mean_chunk) catch unreachable) * H * 4);
    bufs.mean = try ctx.tensorCreate(H * 4);
    bufs.y = try ctx.tensorCreate(band * HX * PA * 4);
    bufs.feat = try ctx.tensorCreate(band * PA * NERF_IN * 4);
    bufs.nx = try ctx.tensorCreate(band * PA * HX * 4);
    bufs.nt = try ctx.tensorCreate(band * PA * HX * 4);
    bufs.cvec = try ctx.tensorCreate(band * HX * PA * 4);
    bufs.nmod = try ctx.tensorCreate(band * PA * 3 * HX * 4);
    bufs.pix = try ctx.tensorCreate(band * PA * 3 * 4);
    inline for (.{ "wq", "wk", "wv", "wo", "qh", "kh" }) |f| {
        @field(bufs, f) = try ctx.tensorCreate(tokens * n_win * H * 4);
    }
    bufs.s = try ctx.tensorCreate(wpb * tokens * tokens * 4);
    {
        const pl = maxPatchLen(model);
        bufs.patch = try ctx.tensorCreate(conv_gpu.convBand(n, pl) * pl * 4);
    }

    // Host-side constants: the modulation (the timestep is always 0, so this is
    // the same table for every image) and the cosine position table.
    {
        const tbl = try mage_vae.MageVae.dicoModTable(io, gpa, model.blocks, model.t_embed);
        defer gpa.free(tbl);
        bufs.mods = try ctx.tensorCreate(tbl.len * 4);
        try ctx.tensorUpload(bufs.mods, std.mem.sliceAsBytes(tbl));
    }
    {
        const dct = try mage_vae.dctTable(gpa);
        defer gpa.free(dct);
        bufs.dct = try ctx.tensorCreate(dct.len * 4);
        try ctx.tensorUpload(bufs.dct, std.mem.sliceAsBytes(dct));
    }
    try ctx.tensorUpload(bufs.stage, std.mem.sliceAsBytes(z));
    bufs.setAct(act);

    try mage_vae_dev.decode(Dev.init(ctx, gpa, &lin, &bufs, &cats, npw, n_win, act), model, lat_h, lat_w, cancel);

    const rgb = try gpa.alloc(f32, img_h * img_w * 3);
    errdefer gpa.free(rgb);
    try ctx.tensorDownload(bufs.rgb, std.mem.sliceAsBytes(rgb));
    return rgb;
}

/// The `mage_vae_dev` adapter.
const Dev = struct {
    ctx: *Context,
    gpa: std.mem.Allocator,
    lin: *const lin_gpu.Lin,
    bufs: *Bufs,
    cats: *sd_unet_gpu.GnCats,
    npw: usize,
    n_win: usize,
    plan: lin_gpu.Plan,

    /// The codec's linears at the loader's f32, or on the f16 coop GEMM when the
    /// activations are f16.
    fn init(ctx: *Context, gpa: std.mem.Allocator, lin: *const lin_gpu.Lin, bufs: *Bufs, cats: *sd_unet_gpu.GnCats, npw: usize, n_win: usize, act: Dt) Dev {
        var p = lin_gpu.Plan.tower;
        p.dense = if (act == .f16) .f16 else .native;
        return .{ .ctx = ctx, .gpa = gpa, .lin = lin, .bufs = bufs, .cats = cats, .npw = npw, .n_win = n_win, .plan = p };
    }

    pub fn beginBatch(d: Dev) !void {
        try d.ctx.beginBatch();
    }

    pub fn endBatch(d: Dev) !void {
        try d.ctx.endBatch();
    }

    pub fn abortBatch(d: Dev) void {
        if (d.ctx.batching) d.ctx.abortBatch();
    }

    pub fn freeAttn(d: Dev) void {
        inline for (.{ "aq", "ak", "av", "ao", "wq", "wk", "wv", "wo", "qh", "kh", "s" }) |f| d.ctx.tensorDestroy(&@field(d.bufs, f));
    }

    /// From the host: the context has no device fill, and `scale_f32` by zero would
    /// read whatever `tensorCreate` left behind, which is NaN-by-zero rather than zero.
    pub fn zero(d: Dev, x: Buf, n: usize) !void {
        const zeros = try d.gpa.alloc(u8, n * x.dt.bytes());
        defer d.gpa.free(zeros);
        @memset(zeros, 0);
        try d.ctx.tensorUpload(x, zeros);
    }

    /// A 16-bit concatenation moves whole words, so every channel count is even.
    pub fn concatCh(d: Dev, src: Buf, dst: Buf, n: usize, c_src: usize, c_dst: usize, off: usize) !void {
        std.debug.assert(src.dt.bytes() == dst.dt.bytes());
        const sh: u1 = @intFromBool(dst.dt != .f32);
        if (sh == 1) std.debug.assert(c_src % 2 == 0 and c_dst % 2 == 0 and off % 2 == 0);
        try d.ctx.opElt(.concat_ch, src, dst, null, null, .{
            .u0 = @intCast((n * c_src) >> sh),
            .u1 = @intCast(c_src >> sh),
            .u2 = @intCast(c_dst >> sh),
            .u3 = @intCast(off >> sh),
        }, (n * c_src) >> sh, 1, 1);
    }

    pub fn conv(d: Dev, dst: *Buf, src: *const Buf, h: usize, w: usize, cv: Conv2d) !void {
        std.debug.assert(dst.size >= h * w * cv.co * 4);
        try conv_gpu.conv2d(d.ctx, &d.bufs.patch, dst, src, h, w, cv, .stride1, null, null, 1.0);
    }

    pub fn groupNorm(d: Dev, dst: *Buf, src: *const Buf, n: usize, nw: mage_vae.NormW, with_silu: bool) !void {
        try sd_unet_gpu.groupNormInto(d.ctx, d.bufs.gstat, d.bufs.gmi, dst, src, n, H, try d.cats.get(d.ctx, d.gpa, nw.w, nw.b), mage_vae.norm_groups, eps, with_silu);
    }

    pub fn add(d: Dev, x: Buf, y: Buf, n: usize) !void {
        std.debug.assert(x.dt == y.dt);
        try d.ctx.opElt(eltAt("add", x.dt), x, y, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
    }

    /// Each window is a HEAD, and the windows batch to a scores budget.
    pub fn windows(d: Dev, h: usize, w: usize) !void {
        const ctx = d.ctx;
        const b = d.bufs;
        const win = mage_vae.attn_window;
        const tokens = win * win;
        const n_win = d.n_win;
        const wpush: gpu_context.EltPush = .{
            .u0 = @intCast(tokens * n_win * H),
            .u1 = @intCast(h),
            .u2 = @intCast(w),
            .u3 = H,
            .u4 = @intCast(d.npw),
            .u5 = win,
            .u6 = @intCast(n_win),
        };
        try ctx.opElt(.win_gather, b.wq, b.aq, null, null, wpush, wpush.u0, 1, 1);
        try ctx.opElt(.win_gather, b.wk, b.ak, null, null, wpush, wpush.u0, 1, 1);
        try ctx.opElt(.win_gather, b.wv, b.av, null, null, wpush, wpush.u0, 1, 1);

        // Per-head k-major, so the scores kernel loads contiguously. V stays
        // row-major: `attn_out` strides it by the head.
        const gpush: gpu_context.EltPush = .{
            .u0 = @intCast(tokens * n_win * H),
            .u1 = @intCast(n_win),
            .u2 = H,
            .u3 = @intCast(tokens),
        };
        ctx.independent(2);
        try ctx.opElt(.gather_kmajor, b.wq, null, null, b.qh, gpush, gpush.u0, 1, 1);
        try ctx.opElt(.gather_kmajor, b.wk, null, null, b.kh, gpush, gpush.u0, 1, 1);

        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(H)));
        const wpb = winsPerBatch(tokens, n_win, s_bytes_cap);
        const tblk = std.math.divCeil(usize, tokens, 8) catch unreachable;
        var w0: usize = 0;
        while (w0 < n_win) : (w0 += wpb) {
            const wb = @min(wpb, n_win - w0);
            try ctx.opElt(.attn_scores, b.qh, b.kh, null, b.s, .{
                .u0 = @intCast(tokens),
                .u1 = @intCast(n_win),
                .u2 = @intCast(n_win),
                .u3 = H,
                .u4 = @intCast(w0),
                .f0 = scale,
            }, tblk, tblk, wb);
            // No softmax pass: `attn_out` runs its own online softmax over the raw
            // scores, and a pass here would exponentiate twice.
            try ctx.opElt(.attn_out, b.s, null, b.wv, b.wo, .{
                .u0 = @intCast(tokens),
                .u1 = @intCast(n_win),
                .u2 = @intCast(n_win),
                .u3 = H,
                .u4 = @intCast(w0),
                .u5 = @intCast(tokens),
                .f0 = @bitCast(@as(u32, @intCast(tokens * tokens))),
            }, H / 8, tblk, wb);
        }
        try ctx.opElt(.win_scatter, b.ao, b.wo, null, null, wpush, wpush.u0, 1, 1);
    }

    pub fn lnMod(d: Dev, x: Buf, y: Buf, n: usize, shift_off: usize, scale_off: usize) !void {
        try d.ctx.opLnModSg(x, y, d.bufs.mods, n, H, shift_off, scale_off, eps);
    }

    pub fn depthwise3(d: Dev, dst: Buf, src: Buf, k: []const f32, bias: []const f32, h: usize, w: usize) !void {
        std.debug.assert(dst.dt == src.dt);
        try d.ctx.opElt(eltAt("dw_conv3", src.dt), dst, src, try vecBuf(d.ctx, k), try vecBuf(d.ctx, bias), .{
            .u0 = @intCast(h * w * H),
            .u1 = @intCast(h),
            .u2 = @intCast(w),
            .u3 = H,
        }, h * w * H, 1, 1);
    }

    pub fn gelu(d: Dev, x: Buf, n: usize) !void {
        try d.ctx.opElt(eltAt("gelu_erf", x.dt), x, null, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
    }

    pub fn colMean(d: Dev, dst: Buf, src: Buf, n: usize) !void {
        std.debug.assert(dst.dt == .f32);
        const chunk = dual_table.col_mean_chunk;
        const chunks = std.math.divCeil(usize, n, chunk) catch unreachable;
        const inv = 1.0 / @as(f32, @floatFromInt(n));
        const first: Buf = if (chunks == 1) dst else d.bufs.part;
        try d.ctx.opElt(eltAt("col_mean", src.dt), first, src, null, null, .{ .u0 = @intCast(chunks * H), .u1 = @intCast(n), .u2 = chunk, .u3 = H, .f0 = inv }, chunks * H, 1, 1);
        if (chunks == 1) return;
        try d.ctx.opElt(.col_mean, dst, d.bufs.part, null, null, .{ .u0 = H, .u1 = @intCast(chunks), .u2 = @intCast(chunks), .u3 = H, .f0 = 1 }, H, 1, 1);
    }

    /// `x_off` reaches `opMatmul` as a descriptor offset in BYTES, so it must stay
    /// aligned to the device's storage-buffer granularity: a band start is a multiple
    /// of the band, which keeps it a multiple of 16.
    pub fn linear(d: Dev, y: Buf, x: Buf, x_off: usize, rows: usize, l: mage_vae.LinearW, out: usize, in: usize) !void {
        std.debug.assert(l.w.rows == out and l.w.cols == in);
        try d.lin.linearAt(d.plan, y, 0, x, x_off, rows, l.w, l.b);
    }

    pub fn mulColsSigmoid(d: Dev, x: Buf, gate: Buf, n: usize) !void {
        std.debug.assert(gate.dt == .f32);
        try d.ctx.opElt(eltAt("mul_cols_sigmoid", x.dt), x, gate, null, null, .{ .u0 = @intCast(n * H), .u1 = H }, n * H, 1, 1);
    }

    pub fn gatedAdd(d: Dev, x: Buf, delta: Buf, total: usize, gate_off: usize) !void {
        std.debug.assert(x.dt == delta.dt);
        try d.ctx.opElt(eltAt("gated_add", x.dt), x, delta, d.bufs.mods, null, .{
            .u0 = @intCast(total),
            .u1 = H,
            .u2 = @intCast(gate_off),
        }, total, 1, 1);
    }

    pub fn nerfFeat(d: Dev, feat: Buf, y: Buf, nb: usize) !void {
        std.debug.assert(feat.dt == y.dt);
        try d.ctx.opElt(eltAt("nerf_feat", y.dt), feat, y, d.bufs.dct, null, .{
            .u0 = @intCast(nb * PA * NERF_IN),
            .u1 = NERF_IN,
            .u2 = PA,
            .u3 = HX,
            .u4 = mage_vae.dct_dim,
        }, nb * PA * NERF_IN, 1, 1);
    }

    pub fn copy(d: Dev, dst: Buf, src: Buf, n: usize) !void {
        if (dst.dt == src.dt) return d.ctx.tensorCopy(dst, 0, src, 0, n * dst.dt.bytes());
        try d.ctx.opElt(gpu_context.eltCopy(src.dt, dst.dt), src, dst, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
    }

    pub fn silu(d: Dev, x: Buf, n: usize) !void {
        try d.ctx.opElt(eltAt("silu", x.dt), x, null, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
    }

    /// `x` and `y` at one dtype, or `x` f32 and `y` 16-bit.
    pub fn layerNorm(d: Dev, x: Buf, y: Buf, nw: mage_vae.NormW, rows: usize, cols: usize) !void {
        std.debug.assert(x.dt == y.dt or x.dt == .f32);
        const e: gpu_context.Elt = if (x.dt == y.dt) eltAt("layernorm", x.dt) else if (y.dt == .f16) .layernorm_f32_f16 else .layernorm_f32_bf16;
        try d.ctx.opElt(e, x, y, try vecBuf(d.ctx, nw.w), try vecBuf(d.ctx, nw.b), .{
            .u0 = @intCast(rows),
            .u1 = @intCast(cols),
            .f0 = eps,
        }, rows, 1, 1);
    }

    pub fn modulatePerRow(d: Dev, x: Buf, mod: Buf, rows: usize, cols: usize) !void {
        std.debug.assert(x.dt == mod.dt);
        try d.ctx.opElt(eltAt("modulate_pr", x.dt), x, null, mod, null, .{ .u0 = @intCast(rows * cols), .u1 = @intCast(cols) }, rows * cols, 1, 1);
    }

    pub fn gatedAddPerRow(d: Dev, x: Buf, delta: Buf, mod: Buf, rows: usize, cols: usize) !void {
        std.debug.assert(x.dt == .f32 and delta.dt == mod.dt);
        try d.ctx.opElt(eltAt("gated_add_pr", delta.dt), x, delta, mod, null, .{ .u0 = @intCast(rows * cols), .u1 = @intCast(cols) }, rows * cols, 1, 1);
    }

    pub fn rmsNorm(d: Dev, x: Buf, weights: []const f32, rows: usize, dim: usize) !void {
        const w = try vecBuf(d.ctx, weights);
        if (d.ctx.hasSubgroupNorm()) return d.ctx.opRmsNormSg(x, x, w, rows, dim, eps);
        try d.ctx.opElt(.rmsnorm, x, x, w, null, .{
            .u0 = @intCast(rows),
            .u1 = @intCast(dim),
            .f0 = eps,
        }, rows, 1, 1);
    }

    pub fn patchScatter(d: Dev, rgb: Buf, pix: Buf, nb: usize, lat_w: usize, l0: usize) !void {
        try d.ctx.opElt(.patch_scatter, rgb, pix, null, null, .{
            .u0 = @intCast(nb * PA * 3),
            .u1 = @intCast(lat_w),
            .u2 = mage_vae.patch,
            .u3 = @intCast(l0),
        }, nb * PA * 3, 1, 1);
    }

    /// `mage_vae_dev.mag`. Costs a sync per trace.
    pub fn trace(d: Dev, name: []const u8, x: Buf, n: usize) void {
        if (!mage_vae_dev.mag.on) return;
        traceInto(d.ctx, name, x, n) catch |err| std.log.warn("TP_VAE_MAG {s}: {t}", .{ name, err });
    }
};

/// Wrap a CPU f32 vector as a (pointer-cached) small device buffer.
fn vecBuf(ctx: *Context, v: []const f32) !Buf {
    return .{ .buf = try ctx.smallBuffer(std.mem.sliceAsBytes(v)), .mem = .null_handle, .size = v.len * 4 };
}
