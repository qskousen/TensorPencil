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
const sd_unet_gpu = @import("sd_unet_gpu.zig");
const ops = @import("tp_ops");
const gpu_context = @import("tp_gpu").context;

const MageVae = mage_vae.MageVae;
const Context = gpu_context.Context;
const Buf = gpu_context.DeviceBuffer;
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

/// Latent cells one NeRF band covers; see `mage_vae_cuda.nerf_band`.
const nerf_band: usize = 704;

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
    mean: Buf = none,
    y: Buf = none,
    feat: Buf = none,
    nx: Buf = none,
    nt: Buf = none,
    cvec: Buf = none,
    nmod: Buf = none,
    pix: Buf = none,

    fn deinit(self: *Bufs, ctx: *Context) void {
        inline for (@typeInfo(Bufs).@"struct".fields) |f| ctx.tensorDestroy(&@field(self, f.name));
    }
};

/// GroupNorm weight ++ bias concatenations, cached by weight pointer: `gn_apply`
/// reads both from one binding and the checkpoint stores them as two tensors.
/// `Context.smallBuffer` keys on the concatenated slice, so each uploads once.
const NormCats = struct {
    map: std.AutoHashMapUnmanaged(usize, []f32) = .empty,
    alloc: std.mem.Allocator,

    fn get(self: *NormCats, nw: mage_vae.NormW) ![]const f32 {
        const key = @intFromPtr(nw.w.ptr);
        if (self.map.get(key)) |c| return c;
        const c = try self.alloc.alloc(f32, nw.w.len + nw.b.len);
        @memcpy(c[0..nw.w.len], nw.w);
        @memcpy(c[nw.w.len..], nw.b);
        try self.map.put(self.alloc, key, c);
        return c;
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
pub fn codForward(model: *const MageVae, ctx: *Context, gpa: std.mem.Allocator, z: []const f32, lat_h: usize, lat_w: usize) ![]f32 {
    const n = lat_h * lat_w;
    var bufs: Bufs = .{};
    defer bufs.deinit(ctx);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var cats: NormCats = .{ .alloc = arena.allocator() };

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
    bufs.patch = try ctx.tensorCreate(sd_unet_gpu.convBand(n, maxPatchLen(model)) * maxPatchLen(model) * 4);
    try ctx.tensorUpload(bufs.stage, std.mem.sliceAsBytes(z));

    try ctx.beginBatch();
    errdefer if (ctx.batching) ctx.abortBatch();
    try cod(ctx, &bufs, &cats, model, lat_h, lat_w, npw, n_win);
    try ctx.endBatch();

    const out = try gpa.alloc(f32, n * H);
    errdefer gpa.free(out);
    try ctx.tensorDownload(bufs.cond, std.mem.sliceAsBytes(out));
    return out;
}

/// Non-finite count and peak magnitude of a device buffer's first `n` floats,
/// under TP_MAGEVAE_TRACE. The Z-Image pattern: an f16 overflow is invisible in a
/// norm and obvious in a magnitude.
fn trace(ctx: *Context, what: []const u8, b: Buf, n: usize) void {
    if (std.c.getenv("TP_MAGEVAE_TRACE") == null) return;
    var buf: [1 << 16]f32 = undefined;
    const take = @min(n, buf.len);
    ctx.tensorDownload(b, std.mem.sliceAsBytes(buf[0..take])) catch return;
    var bad: usize = 0;
    var peak: f32 = 0;
    for (buf[0..take]) |v| {
        if (!std.math.isFinite(v)) bad += 1 else peak = @max(peak, @abs(v));
    }
    std.debug.print("mage_vae_gpu {s:<12} nonfinite {d}/{d}  peak {d:.4}\n", .{ what, bad, take, peak });
}

/// `conv_in`, three resnets with two windowed attentions between them, and the
/// head, leaving the conditioning in `bufs.cond`.
fn cod(ctx: *Context, bufs: *Bufs, cats: *NormCats, model: *const MageVae, lat_h: usize, lat_w: usize, npw: usize, n_win: usize) !void {
    const n = lat_h * lat_w;
    try conv(ctx, bufs, &bufs.x, &bufs.stage, lat_h, lat_w, model.cod.conv_in);
    trace(ctx, "conv_in", bufs.x, n * H);
    try resnet(ctx, bufs, cats, lat_h, lat_w, model.cod.r0);
    trace(ctx, "r0", bufs.x, n * H);
    try winAttn(ctx, bufs, cats, lat_h, lat_w, model.cod.a0, npw, n_win);
    trace(ctx, "a0", bufs.x, n * H);
    try resnet(ctx, bufs, cats, lat_h, lat_w, model.cod.r1);
    trace(ctx, "r1", bufs.x, n * H);
    try winAttn(ctx, bufs, cats, lat_h, lat_w, model.cod.a1, npw, n_win);
    trace(ctx, "a1", bufs.x, n * H);
    try resnet(ctx, bufs, cats, lat_h, lat_w, model.cod.r2);
    trace(ctx, "r2", bufs.x, n * H);
    try groupNorm(ctx, bufs, cats, &bufs.t, &bufs.x, n, H, model.cod.norm_out, true);
    try conv(ctx, bufs, &bufs.cond, &bufs.t, lat_h, lat_w, model.cod.conv_out);
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
    cancel: ?*std.atomic.Value(bool),
) ![]f32 {
    const n = lat_h * lat_w;
    std.debug.assert(z.len == n * mage_vae.latent_channels);
    const img_h = lat_h * mage_vae.patch;
    const img_w = lat_w * mage_vae.patch;
    const cat_w = mage_vae.bottleneck + H;

    var bufs: Bufs = .{};
    defer bufs.deinit(ctx);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var cats: NormCats = .{ .alloc = arena.allocator() };

    // `: usize` is load-bearing: without it `@min` narrows every size below.
    const band: usize = @min(nerf_band, n);
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
        bufs.patch = try ctx.tensorCreate(sd_unet_gpu.convBand(n, pl) * pl * 4);
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

    try ctx.beginBatch();
    errdefer if (ctx.batching) ctx.abortBatch();
    try cod(ctx, &bufs, &cats, model, lat_h, lat_w, npw, n_win);
    try ctx.endBatch();

    // The attention scratch is dead from here and would otherwise stay resident
    // through the whole per-pixel pathway, which is where the memory goes.
    inline for (.{ "aq", "ak", "av", "ao", "wq", "wk", "wv", "wo", "qh", "kh", "s" }) |f| {
        ctx.tensorDestroy(&@field(bufs, f));
    }

    // --- `s_embedder`: zero image half, then the conditioning -----------------
    // `proj1` is bias-free and decode's noise is zero, so its 128 columns
    // contribute nothing; they are still written (as zeros) rather than folded
    // away, so this and the CPU reference read the same.
    {
        const zeros = try gpa.alloc(f32, n * cat_w);
        defer gpa.free(zeros);
        @memset(zeros, 0);
        try ctx.tensorUpload(bufs.cat, std.mem.sliceAsBytes(zeros));
    }
    try ctx.opElt(.concat_ch, bufs.cond, bufs.cat, null, null, .{
        .u0 = @intCast(n * H),
        .u1 = H,
        .u2 = cat_w,
        .u3 = mage_vae.bottleneck,
    }, n * H, 1, 1);
    try conv(ctx, &bufs, &bufs.x, &bufs.cat, lat_h, lat_w, model.s_proj2);

    // One submission per DiCo block rather than per op; the cancel check wants a
    // boundary anyway, and one batch over all 21 would hold every intermediate.
    for (model.blocks, 0..) |*blk, i| {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        try ctx.beginBatch();
        try dico(ctx, &bufs, lat_h, lat_w, blk, i);
        try ctx.endBatch();
    }

    // --- the per-pixel MLP, in bands over latent cells -------------------------
    var l0: usize = 0;
    while (l0 < n) : (l0 += band) {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        const nb: usize = @min(band, n - l0);
        try ctx.beginBatch();
        try nerfBand(ctx, &bufs, model, l0, nb, lat_w);
        try ctx.endBatch();
    }

    const rgb = try gpa.alloc(f32, img_h * img_w * 3);
    errdefer gpa.free(rgb);
    try ctx.tensorDownload(bufs.rgb, std.mem.sliceAsBytes(rgb));
    return rgb;
}

/// One DiCo block, in place on `bufs.x`.
fn dico(ctx: *Context, bufs: *Bufs, h: usize, w: usize, blk: *const mage_vae.DiCo, i: usize) !void {
    const n = h * w;
    const base = i * 6 * H;

    // x_t = (1 + scale1) * layernorm(x) + shift1
    try ctx.opLnModSg(bufs.x, bufs.t, bufs.mods, n, H, base + 0 * H, base + 1 * H, eps);
    try conv(ctx, bufs, &bufs.u, &bufs.t, h, w, blk.conv1);
    try ctx.opElt(.dw_conv3, bufs.t, bufs.u, try vecBuf(ctx, blk.dw), try vecBuf(ctx, blk.dw_b), .{
        .u0 = @intCast(n * H),
        .u1 = @intCast(h),
        .u2 = @intCast(w),
        .u3 = H,
    }, n * H, 1, 1);
    try ctx.opElt(.gelu_erf, bufs.t, null, null, null, .{ .u0 = @intCast(n * H) }, n * H, 1, 1);

    // Channel attention: the spatial mean, one 1x1, a sigmoid, broadcast back.
    try ctx.opElt(.col_mean, bufs.mean, bufs.t, null, null, .{ .u0 = H, .u1 = @intCast(n) }, H, 1, 1);
    try ctx.opMatmul(bufs.gate, 0, bufs.mean, 0, 1, blk.ca.w.bytes, false, H, H, blk.ca.w.scale, blk.ca.b);
    try ctx.opElt(.mul_cols_sigmoid, bufs.t, bufs.gate, null, null, .{ .u0 = @intCast(n * H), .u1 = H }, n * H, 1, 1);

    try conv(ctx, bufs, &bufs.u, &bufs.t, h, w, blk.conv3);
    try gatedAdd(ctx, bufs.x, bufs.u, bufs.mods, n * H, H, base + 2 * H);

    // x += gate2 * conv5(gelu(conv4((1 + scale2) * layernorm(x) + shift2)))
    try ctx.opLnModSg(bufs.x, bufs.t, bufs.mods, n, H, base + 3 * H, base + 4 * H, eps);
    try conv(ctx, bufs, &bufs.wide, &bufs.t, h, w, blk.conv4);
    try ctx.opElt(.gelu_erf, bufs.wide, null, null, null, .{ .u0 = @intCast(n * FFN) }, n * FFN, 1, 1);
    try conv(ctx, bufs, &bufs.u, &bufs.wide, h, w, blk.conv5);
    try gatedAdd(ctx, bufs.x, bufs.u, bufs.mods, n * H, H, base + 5 * H);
}

/// One band of latent cells through the per-pixel MLP, writing their 16x16 tiles.
fn nerfBand(ctx: *Context, bufs: *Bufs, model: *const MageVae, l0: usize, nb: usize, lat_w: usize) !void {
    const rows = nb * PA;
    // `opMatmul` binds its source at a descriptor offset, so this is BYTES, not
    // floats, and must stay aligned to the device's storage-buffer granularity:
    // `l0` is a multiple of the band, which keeps it a multiple of 16.
    const off: u64 = l0 * H * 4;

    // The patch's per-pixel channels, and the per-pixel conditioning.
    try ctx.opMatmul(bufs.y, 0, bufs.cond, off, nb, model.y_embedder_x.w.bytes, false, HX * PA, H, model.y_embedder_x.w.scale, model.y_embedder_x.b);
    try ctx.opMatmul(bufs.cvec, 0, bufs.x, off, nb, model.cond_embed.w.bytes, false, HX * PA, H, model.cond_embed.w.scale, model.cond_embed.b);

    try ctx.opElt(.nerf_feat, bufs.feat, bufs.y, bufs.dct, null, .{
        .u0 = @intCast(nb * PA * NERF_IN),
        .u1 = NERF_IN,
        .u2 = PA,
        .u3 = HX,
        .u4 = mage_vae.dct_dim,
    }, nb * PA * NERF_IN, 1, 1);
    try ctx.opMatmul(bufs.nx, 0, bufs.feat, 0, rows, model.x_embedder.w.bytes, false, HX, NERF_IN, model.x_embedder.w.scale, model.x_embedder.b);
    try ctx.opMatmul(bufs.nt, 0, bufs.nx, 0, rows, model.input_proj.w.bytes, false, HX, HX, model.input_proj.w.scale, model.input_proj.b);
    try ctx.tensorCopy(bufs.nx, 0, bufs.nt, 0, rows * HX * 4);

    for (model.res_blocks) |*r| {
        // `cvec` IS the conditioning `y`; silu it into the scratch rather than in
        // place, because every block reads the same untouched vector.
        try ctx.tensorCopy(bufs.nt, 0, bufs.cvec, 0, rows * HX * 4);
        try ctx.opElt(.silu, bufs.nt, null, null, null, .{ .u0 = @intCast(rows * HX) }, rows * HX, 1, 1);
        try ctx.opMatmul(bufs.nmod, 0, bufs.nt, 0, rows, r.adaln.w.bytes, false, 3 * HX, HX, r.adaln.w.scale, r.adaln.b);

        try ctx.opElt(.layernorm, bufs.nx, bufs.nt, try vecBuf(ctx, r.in_ln.w), try vecBuf(ctx, r.in_ln.b), .{
            .u0 = @intCast(rows),
            .u1 = HX,
            .f0 = eps,
        }, rows, 1, 1);
        try ctx.opElt(.modulate_pr, bufs.nt, null, bufs.nmod, null, .{ .u0 = @intCast(rows * HX), .u1 = HX }, rows * HX, 1, 1);
        try ctx.opMatmul(bufs.feat, 0, bufs.nt, 0, rows, r.fc1.w.bytes, false, HX, HX, r.fc1.w.scale, r.fc1.b);
        try ctx.opElt(.silu, bufs.feat, null, null, null, .{ .u0 = @intCast(rows * HX) }, rows * HX, 1, 1);
        try ctx.opMatmul(bufs.nt, 0, bufs.feat, 0, rows, r.fc2.w.bytes, false, HX, HX, r.fc2.w.scale, r.fc2.b);
        try ctx.opElt(.gated_add_pr, bufs.nx, bufs.nt, bufs.nmod, null, .{ .u0 = @intCast(rows * HX), .u1 = HX }, rows * HX, 1, 1);
    }

    try rmsNorm(ctx, bufs.nx, bufs.nx, model.final_norm, rows, HX);
    try ctx.opMatmul(bufs.pix, 0, bufs.nx, 0, rows, model.final_lin.w.bytes, false, 3, HX, model.final_lin.w.scale, model.final_lin.b);
    try ctx.opElt(.patch_scatter, bufs.rgb, bufs.pix, null, null, .{
        .u0 = @intCast(nb * PA * 3),
        .u1 = @intCast(lat_w),
        .u2 = mage_vae.patch,
        .u3 = @intCast(l0),
    }, nb * PA * 3, 1, 1);
}

/// `x += conv2(silu(norm2(conv1(silu(norm1(x))))))`, width-preserving.
fn resnet(ctx: *Context, bufs: *Bufs, cats: *NormCats, h: usize, w: usize, r: mage_vae.Resnet) !void {
    const n = h * w;
    try groupNorm(ctx, bufs, cats, &bufs.t, &bufs.x, n, H, r.norm1, true);
    try conv(ctx, bufs, &bufs.u, &bufs.t, h, w, r.conv1);
    try groupNorm(ctx, bufs, cats, &bufs.t, &bufs.u, n, H, r.norm2, true);
    try conv(ctx, bufs, &bufs.u, &bufs.t, h, w, r.conv2);
    try ctx.opElt(.add, bufs.x, bufs.u, null, null, .{ .u0 = @intCast(n * H) }, n * H, 1, 1);
}

/// `x += proj(windowed_attention(norm(x)))`.
///
/// Each 32x32 window attends to itself alone, and the gather lays the windows out
/// as attention HEADS so one pass covers a batch of them. Out-of-range samples
/// REPLICATE the edge, which is the reference's padding and is not the same as
/// attending over a smaller window.
fn winAttn(ctx: *Context, bufs: *Bufs, cats: *NormCats, h: usize, w: usize, a: mage_vae.WinAttn, npw: usize, n_win: usize) !void {
    const n = h * w;
    const d = mage_vae.attn_window;
    const tokens = d * d;
    try groupNorm(ctx, bufs, cats, &bufs.t, &bufs.x, n, H, a.norm, false);
    try conv(ctx, bufs, &bufs.aq, &bufs.t, h, w, a.q);
    try conv(ctx, bufs, &bufs.ak, &bufs.t, h, w, a.k);
    try conv(ctx, bufs, &bufs.av, &bufs.t, h, w, a.v);

    const wpush: gpu_context.EltPush = .{
        .u0 = @intCast(tokens * n_win * H),
        .u1 = @intCast(h),
        .u2 = @intCast(w),
        .u3 = H,
        .u4 = @intCast(npw),
        .u5 = d,
        .u6 = @intCast(n_win),
    };
    try ctx.opElt(.win_gather, bufs.wq, bufs.aq, null, null, wpush, wpush.u0, 1, 1);
    try ctx.opElt(.win_gather, bufs.wk, bufs.ak, null, null, wpush, wpush.u0, 1, 1);
    try ctx.opElt(.win_gather, bufs.wv, bufs.av, null, null, wpush, wpush.u0, 1, 1);

    // Per-head k-major, so the scores kernel loads contiguously. V stays
    // row-major: `attn_out` strides it by the head.
    const gpush: gpu_context.EltPush = .{
        .u0 = @intCast(tokens * n_win * H),
        .u1 = @intCast(n_win),
        .u2 = H,
        .u3 = @intCast(tokens),
    };
    ctx.independent(2);
    try ctx.opElt(.gather_kmajor, bufs.wq, null, null, bufs.qh, gpush, gpush.u0, 1, 1);
    try ctx.opElt(.gather_kmajor, bufs.wk, null, null, bufs.kh, gpush, gpush.u0, 1, 1);

    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(H)));
    const wpb = winsPerBatch(tokens, n_win, s_bytes_cap);
    const tblk = std.math.divCeil(usize, tokens, 8) catch unreachable;
    var w0: usize = 0;
    while (w0 < n_win) : (w0 += wpb) {
        const wb = @min(wpb, n_win - w0);
        try ctx.opElt(.attn_scores, bufs.qh, bufs.kh, null, bufs.s, .{
            .u0 = @intCast(tokens),
            .u1 = @intCast(n_win),
            .u2 = @intCast(n_win),
            .u3 = H,
            .u4 = @intCast(w0),
            .f0 = scale,
        }, tblk, tblk, wb);
        // No softmax pass: `attn_out` runs its own online softmax over the raw
        // scores, and a pass here would exponentiate twice.
        try ctx.opElt(.attn_out, bufs.s, null, bufs.wv, bufs.wo, .{
            .u0 = @intCast(tokens),
            .u1 = @intCast(n_win),
            .u2 = @intCast(n_win),
            .u3 = H,
            .u4 = @intCast(w0),
            .u5 = @intCast(tokens),
            .f0 = @bitCast(@as(u32, @intCast(tokens * tokens))),
        }, H / 8, tblk, wb);
    }

    try ctx.opElt(.win_scatter, bufs.ao, bufs.wo, null, null, wpush, wpush.u0, 1, 1);
    try conv(ctx, bufs, &bufs.t, &bufs.ao, h, w, a.proj);
    try ctx.opElt(.add, bufs.x, bufs.t, null, null, .{ .u0 = @intCast(n * H) }, n * H, 1, 1);
}

fn gatedAdd(ctx: *Context, x: Buf, delta: Buf, mod: Buf, total: usize, dim: usize, gate_off: usize) !void {
    try ctx.opElt(.gated_add, x, delta, mod, null, .{
        .u0 = @intCast(total),
        .u1 = @intCast(dim),
        .u2 = @intCast(gate_off),
    }, total, 1, 1);
}

fn rmsNorm(ctx: *Context, x: Buf, out: Buf, weights: []const f32, rows: usize, dim: usize) !void {
    const w = try vecBuf(ctx, weights);
    if (ctx.hasSubgroupNorm()) return ctx.opRmsNormSg(x, out, w, rows, dim, eps);
    try ctx.opElt(.rmsnorm, x, out, w, null, .{
        .u0 = @intCast(rows),
        .u1 = @intCast(dim),
        .f0 = eps,
    }, rows, 1, 1);
}

fn groupNorm(ctx: *Context, bufs: *Bufs, cats: *NormCats, dst: *Buf, src: *const Buf, n: usize, ch: usize, nw: mage_vae.NormW, silu: bool) !void {
    try sd_unet_gpu.groupNormInto(ctx, bufs.gstat, bufs.gmi, dst, src, n, ch, try cats.get(nw), mage_vae.norm_groups, eps, silu, false);
}

fn conv(ctx: *Context, bufs: *Bufs, dst: *Buf, src: *const Buf, h: usize, w: usize, cv: Conv2d) !void {
    std.debug.assert(dst.size >= h * w * cv.co * 4);
    try sd_unet_gpu.convIntoPrec(ctx, &bufs.patch, dst, src, h, w, cv, .stride1, null, null, 1.0, false, false);
}

/// Wrap a CPU f32 vector as a (pointer-cached) small device buffer.
fn vecBuf(ctx: *Context, v: []const f32) !Buf {
    return .{ .buf = try ctx.smallBuffer(std.mem.sliceAsBytes(v)), .mem = .null_handle, .size = v.len * 4 };
}
