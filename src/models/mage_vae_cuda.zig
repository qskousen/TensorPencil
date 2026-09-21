//! GPU-resident Mage-VAE decode on the CUDA backends, the device twin of
//! `mage_vae.MageVae.decode`.
//!
//! The codec is not a CNN ladder, so this is not a copy of `sd_vae_cuda`: the
//! resolution never changes by convolution, everything happens at the latent
//! grid or inside a 16x16 patch, and the heavy half is a per-PIXEL MLP rather
//! than a stack of upsampling blocks. Four ops had no device kernel before it
//! and are now in `kernels/dual/`, so the Vulkan arm gets them for free:
//!
//!   `dw_conv3`       the DiCo block's depthwise 3x3, which is not an im2col
//!                    GEMM (its patch matrix would be block-diagonal)
//!   `col_mean` +
//!   `mul_cols_sigmoid`  the channel-attention gate
//!   `nerf_feat`      the per-pixel MLP's input rows
//!   `win_gather` /
//!   `win_scatter`    the CoD decoder's 32x32 attention windows, with the
//!                    reference's REPLICATE padding
//!   `patch_scatter`  the fold back to the image
//!   `modulate_pr` /
//!   `gated_add_pr`   per-ROW AdaLN, because the per-pixel MLP conditions each
//!                    pixel on its own patch vector where every transformer here
//!                    conditions a whole row range on one vector
//!
//! Two things are host-side and deliberately so. The DiCo modulation is
//! CONSTANT (the codec is one-step, so its timestep is always 0), and
//! `mage_vae.dicoModTable` builds it once at session setup, which keeps the 21
//! `adaLN_modulation` linears off the device entirely. The DCT position table is
//! likewise fixed and uploaded once.
//!
//! The windowed attention lays each window out as an attention HEAD
//! (`[win_area][n_win][ch]`), so one batched call attends every window against
//! itself alone. It runs `be.attn`, the f32 online softmax, not `opAttnTC`, for
//! `sd_vae_cuda`'s reason: a VAE's attention logits reach magnitudes f16 cannot
//! hold, and this one's head width is 384 rather than the 128 the tensor-core
//! path tiles for.

const std = @import("std");
const mage_vae = @import("mage_vae.zig");
const sd_unet_cuda = @import("sd_unet_cuda.zig");
const ops = @import("tp_ops");
const cuda = @import("tp_gpu").cuda;

const MageVae = mage_vae.MageVae;
const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;
const Conv2d = ops.conv.Conv2d;

const H = mage_vae.hidden; // 384
const HX = mage_vae.hidden_x; // 32
const FFN = mage_vae.ffn; // 1536
const PA = mage_vae.patch_area; // 256
const NERF_IN = mage_vae.nerf_in; // 99
const eps: f32 = 1e-6;
const gn_chunks: usize = 32;

/// Latent cells one NeRF band covers. The pathway is per-cell independent, so
/// this changes only the peak: the widest intermediate is `band * 256 * 99`
/// floats, ~66 MB at 704.
const nerf_band: usize = 704;

/// Force the naive attention instead of `opAttnTC` in the windowed blocks. Kept
/// for symmetry with the other arms' A/B; the default is already `be.attn`.
pub var force_naive_attn: bool = true;

const Bufs = struct {
    /// The running `[n][H]` activation and two scratches of the same shape.
    x: Buf = .{},
    t: Buf = .{},
    u: Buf = .{},
    /// `[n][FFN]`, the DiCo MLP's wide intermediate.
    wide: Buf = .{},
    /// The latent coming in and the RGB going out.
    stage: Buf = .{},
    rgb: Buf = .{},
    patch: Buf = .{},
    gstat: Buf = .{},
    gmi: Buf = .{},
    /// `[n][bottleneck + H]`, the `s_embedder` concatenation.
    cat: Buf = .{},
    /// The CoD conditioning, kept for the NeRF path after the trunk overwrites x.
    cond: Buf = .{},
    /// Windowed attention: projections and the window-major gather.
    aq: Buf = .{},
    ak: Buf = .{},
    av: Buf = .{},
    ao: Buf = .{},
    wq: Buf = .{},
    wk: Buf = .{},
    wv: Buf = .{},
    wo: Buf = .{},
    /// Constants uploaded once per decode.
    mods: Buf = .{},
    dct: Buf = .{},
    gate: Buf = .{},
    mean: Buf = .{},
    /// The NeRF band's buffers.
    y: Buf = .{},
    feat: Buf = .{},
    nx: Buf = .{},
    nt: Buf = .{},
    cvec: Buf = .{},
    nmod: Buf = .{},
    pix: Buf = .{},

    fn deinit(self: *Bufs, be: *Backend) void {
        inline for (@typeInfo(Bufs).@"struct".fields) |f| be.tensorDestroy(&@field(self, f.name));
    }
};

/// GroupNorm weight ++ bias on the device, cached by weight pointer: `gn_apply`
/// reads both from one binding and the checkpoint stores them as two tensors.
const NormBufs = struct {
    map: std.AutoHashMapUnmanaged(usize, Buf) = .empty,
    alloc: std.mem.Allocator,
    be: *Backend,

    fn get(self: *NormBufs, nw: mage_vae.NormW) !Buf {
        const key = @intFromPtr(nw.w.ptr);
        if (self.map.get(key)) |b| return b;
        const cat = try self.alloc.alloc(f32, nw.w.len + nw.b.len);
        @memcpy(cat[0..nw.w.len], nw.w);
        @memcpy(cat[nw.w.len..], nw.b);
        var b: Buf = .{};
        try self.be.ensureDeviceBuffer(&b, cat.len * 4);
        try self.be.tensorUpload(b, std.mem.sliceAsBytes(cat));
        try self.map.put(self.alloc, key, b);
        return b;
    }

    fn deinit(self: *NormBufs) void {
        var it = self.map.valueIterator();
        while (it.next()) |b| self.be.tensorDestroy(b);
        self.map.deinit(self.alloc);
    }
};

/// Whether this backend can run the decode at all. Every weight here is f32
/// after the loader, so the only question is the backend itself.
pub fn supported(model: *const MageVae) bool {
    return model.blocks.len != 0;
}

/// The CoD decoder alone: latent -> `[n][hidden]` conditioning. Caller frees.
///
/// A separate entry point so `mage-vae-cuda-test` can compare it against the CPU
/// before the per-pixel pathway runs: the two halves fail for different reasons,
/// and a whole-decode comparison would say only "wrong".
pub fn codForward(model: *const MageVae, be: *Backend, gpa: std.mem.Allocator, z: []const f32, lat_h: usize, lat_w: usize) ![]f32 {
    const n = lat_h * lat_w;
    var bufs: Bufs = .{};
    defer bufs.deinit(be);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var norms: NormBufs = .{ .alloc = arena.allocator(), .be = be };
    defer norms.deinit();

    const d = mage_vae.attn_window;
    const nph = (lat_h + d - 1) / d;
    const npw = (lat_w + d - 1) / d;
    const n_win = nph * npw;
    inline for (.{ "x", "t", "u" }) |f| try be.ensureDeviceBuffer(&@field(bufs, f), n * H * 4);
    try be.ensureDeviceBuffer(&bufs.stage, z.len * 4);
    try be.ensureDeviceBuffer(&bufs.cond, n * H * 4);
    try be.ensureDeviceBuffer(&bufs.gstat, mage_vae.norm_groups * gn_chunks * 3 * 4);
    try be.ensureDeviceBuffer(&bufs.gmi, mage_vae.norm_groups * 2 * 4);
    inline for (.{ "aq", "ak", "av", "ao" }) |f| try be.ensureDeviceBuffer(&@field(bufs, f), n * H * 4);
    inline for (.{ "wq", "wk", "wv", "wo" }) |f| try be.ensureDeviceBuffer(&@field(bufs, f), d * d * n_win * H * 4);
    try be.tensorUpload(bufs.stage, std.mem.sliceAsBytes(z));

    try be.beginBatch();
    errdefer if (be.batching()) be.abortBatch();
    try cod(be, &bufs, &norms, model, lat_h, lat_w, nph, npw, n_win);
    try be.endBatch();

    const out = try gpa.alloc(f32, n * H);
    errdefer gpa.free(out);
    try be.tensorDownload(bufs.cond, std.mem.sliceAsBytes(out));
    return out;
}

/// `conv_in`, three resnets with two windowed attentions between them, and the
/// head, leaving the conditioning in `bufs.cond`.
fn cod(be: *Backend, bufs: *Bufs, norms: *NormBufs, model: *const MageVae, lat_h: usize, lat_w: usize, nph: usize, npw: usize, n_win: usize) !void {
    const n = lat_h * lat_w;
    try conv(be, bufs, &bufs.x, &bufs.stage, lat_h, lat_w, model.cod.conv_in);
    try resnet(be, bufs, norms, lat_h, lat_w, model.cod.r0);
    try winAttn(be, bufs, norms, lat_h, lat_w, model.cod.a0, nph, npw, n_win);
    try resnet(be, bufs, norms, lat_h, lat_w, model.cod.r1);
    try winAttn(be, bufs, norms, lat_h, lat_w, model.cod.a1, nph, npw, n_win);
    try resnet(be, bufs, norms, lat_h, lat_w, model.cod.r2);
    try groupNorm(be, bufs, norms, &bufs.t, &bufs.x, n, H, model.cod.norm_out, true);
    try conv(be, bufs, &bufs.cond, &bufs.t, lat_h, lat_w, model.cod.conv_out);
}

/// Latent `[lat_h][lat_w][latent_channels]` -> RGB `[h][w][3]` in [-1, 1],
/// `h = lat_h * 16`. Caller frees.
pub fn decode(
    model: *const MageVae,
    be: *Backend,
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

    var bufs: Bufs = .{};
    defer bufs.deinit(be);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var norms: NormBufs = .{ .alloc = arena.allocator(), .be = be };
    defer norms.deinit();

    // `: usize` is load-bearing: without it `@min` narrows every buffer size
    // below, which lands as a null device pointer rather than a compile error.
    const band: usize = @min(nerf_band, n);
    inline for (.{ "x", "t", "u" }) |f| try be.ensureDeviceBuffer(&@field(bufs, f), n * H * 4);
    try be.ensureDeviceBuffer(&bufs.wide, n * FFN * 4);
    try be.ensureDeviceBuffer(&bufs.stage, @max(z.len, n * (mage_vae.bottleneck + H)) * 4);
    try be.ensureDeviceBuffer(&bufs.rgb, img_h * img_w * 3 * 4);
    try be.ensureDeviceBuffer(&bufs.cat, n * (mage_vae.bottleneck + H) * 4);
    try be.ensureDeviceBuffer(&bufs.cond, n * H * 4);
    try be.ensureDeviceBuffer(&bufs.gstat, mage_vae.norm_groups * gn_chunks * 3 * 4);
    try be.ensureDeviceBuffer(&bufs.gmi, mage_vae.norm_groups * 2 * 4);
    inline for (.{ "aq", "ak", "av", "ao" }) |f| try be.ensureDeviceBuffer(&@field(bufs, f), n * H * 4);
    try be.ensureDeviceBuffer(&bufs.gate, H * 4);
    try be.ensureDeviceBuffer(&bufs.mean, H * 4);
    try be.ensureDeviceBuffer(&bufs.y, band * HX * PA * 4);
    try be.ensureDeviceBuffer(&bufs.feat, band * PA * NERF_IN * 4);
    try be.ensureDeviceBuffer(&bufs.nx, band * PA * HX * 4);
    try be.ensureDeviceBuffer(&bufs.nt, band * PA * HX * 4);
    try be.ensureDeviceBuffer(&bufs.cvec, band * HX * PA * 4);
    try be.ensureDeviceBuffer(&bufs.nmod, band * PA * 3 * HX * 4);
    try be.ensureDeviceBuffer(&bufs.pix, band * PA * 3 * 4);

    // The windows, laid out with the window as an attention head.
    const d = mage_vae.attn_window;
    const nph = (lat_h + d - 1) / d;
    const npw = (lat_w + d - 1) / d;
    const n_win = nph * npw;
    inline for (.{ "wq", "wk", "wv", "wo" }) |f| try be.ensureDeviceBuffer(&@field(bufs, f), d * d * n_win * H * 4);

    // Host-side constants: the modulation (the timestep is always 0, so this is
    // the same table for every image) and the cosine position table.
    {
        const tbl = try mage_vae.MageVae.dicoModTable(io, gpa, model.blocks, model.t_embed);
        defer gpa.free(tbl);
        try be.ensureDeviceBuffer(&bufs.mods, tbl.len * 4);
        try be.tensorUpload(bufs.mods, std.mem.sliceAsBytes(tbl));
    }
    {
        const dct = try mage_vae.dctTable(gpa);
        defer gpa.free(dct);
        try be.ensureDeviceBuffer(&bufs.dct, dct.len * 4);
        try be.tensorUpload(bufs.dct, std.mem.sliceAsBytes(dct));
    }
    try be.tensorUpload(bufs.stage, std.mem.sliceAsBytes(z));

    try be.beginBatch();
    errdefer if (be.batching()) be.abortBatch();

    // --- the CoD decoder: latent -> per-patch conditioning --------------------
    try cod(be, &bufs, &norms, model, lat_h, lat_w, nph, npw, n_win);

    // The attention scratch is dead from here and would otherwise stay resident
    // through the whole per-pixel pathway, which is where the memory goes.
    try be.endBatch();
    be.freeAttnScratch();
    inline for (.{ "aq", "ak", "av", "ao", "wq", "wk", "wv", "wo" }) |f| be.tensorDestroy(&@field(bufs, f));
    try be.beginBatch();

    // --- `s_embedder`: zero image half, then the conditioning -----------------
    // `proj1` is bias-free and decode's noise is zero, so its 128 columns
    // contribute nothing; they are still written (as zeros) rather than folded
    // away, so this and the CPU reference read the same.
    try be.tensorZero(bufs.cat, n * (mage_vae.bottleneck + H) * 4);
    try be.opConcatCh(bufs.cond, bufs.cat, n, H, mage_vae.bottleneck + H, mage_vae.bottleneck, false);
    try conv(be, &bufs, &bufs.x, &bufs.cat, lat_h, lat_w, model.s_proj2);

    for (model.blocks, 0..) |*blk, i| {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        try dico(be, &bufs, lat_h, lat_w, blk, i);
    }
    try be.endBatch();

    // --- the per-pixel MLP, in bands over latent cells -------------------------
    var l0: usize = 0;
    while (l0 < n) : (l0 += band) {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        const nb: usize = @min(band, n - l0);
        try be.beginBatch();
        try nerfBand(be, &bufs, model, l0, nb, lat_w);
        try be.endBatch();
    }

    const rgb = try gpa.alloc(f32, img_h * img_w * 3);
    errdefer gpa.free(rgb);
    try be.tensorDownload(bufs.rgb, std.mem.sliceAsBytes(rgb));
    return rgb;
}

/// One DiCo block, in place on `bufs.x`.
fn dico(be: *Backend, bufs: *Bufs, h: usize, w: usize, blk: *const mage_vae.DiCo, i: usize) !void {
    const n = h * w;
    const base = i * 6 * H;

    // x_t = (1 + scale1) * layernorm(x) + shift1
    try be.lnMod(bufs.x, bufs.t, bufs.mods, n, H, base + 0 * H, base + 1 * H, eps);
    try conv(be, bufs, &bufs.u, &bufs.t, h, w, blk.conv1);
    try be.opDepthwise3(bufs.t, bufs.u, try vecBuf(be, blk.dw), try vecBuf(be, blk.dw_b), h, w, H);
    try be.geluErf(bufs.t, n * H);

    // Channel attention: the spatial mean, one 1x1, a sigmoid, broadcast back.
    try be.opColMean(bufs.mean, bufs.t, n, H);
    try be.opMatmul(bufs.gate, 0, bufs.mean, 0, 1, blk.ca.w.bytes, false, H, H, blk.ca.w.scale, blk.ca.b);
    try be.opMulColsSigmoid(bufs.t, bufs.gate, n, H);

    try conv(be, bufs, &bufs.u, &bufs.t, h, w, blk.conv3);
    try be.gatedAdd(bufs.x, bufs.u, bufs.mods, n * H, H, base + 2 * H);

    // x += gate2 * conv5(gelu(conv4((1 + scale2) * layernorm(x) + shift2)))
    try be.lnMod(bufs.x, bufs.t, bufs.mods, n, H, base + 3 * H, base + 4 * H, eps);
    try conv(be, bufs, &bufs.wide, &bufs.t, h, w, blk.conv4);
    try be.geluErf(bufs.wide, n * FFN);
    try conv(be, bufs, &bufs.u, &bufs.wide, h, w, blk.conv5);
    try be.gatedAdd(bufs.x, bufs.u, bufs.mods, n * H, H, base + 5 * H);
}

/// One band of latent cells through the per-pixel MLP, writing their 16x16 tiles.
fn nerfBand(be: *Backend, bufs: *Bufs, model: *const MageVae, l0: usize, nb: usize, lat_w: usize) !void {
    const rows = nb * PA;
    const off = l0 * H * 4;
    const cond_b = view(bufs.cond, off, nb * H * 4);
    const s_b = view(bufs.x, off, nb * H * 4);

    // The patch's per-pixel channels, and the per-pixel conditioning.
    try be.opMatmul(bufs.y, 0, cond_b, 0, nb, model.y_embedder_x.w.bytes, false, HX * PA, H, model.y_embedder_x.w.scale, model.y_embedder_x.b);
    try be.opMatmul(bufs.cvec, 0, s_b, 0, nb, model.cond_embed.w.bytes, false, HX * PA, H, model.cond_embed.w.scale, model.cond_embed.b);

    try be.opNerfFeat(bufs.feat, bufs.y, bufs.dct, nb, PA, HX, mage_vae.dct_dim);
    try be.opMatmul(bufs.nx, 0, bufs.feat, 0, rows, model.x_embedder.w.bytes, false, HX, NERF_IN, model.x_embedder.w.scale, model.x_embedder.b);
    try be.opMatmul(bufs.nt, 0, bufs.nx, 0, rows, model.input_proj.w.bytes, false, HX, HX, model.input_proj.w.scale, model.input_proj.b);
    try be.tensorCopy(bufs.nx, 0, bufs.nt, 0, rows * HX * 4);

    for (model.res_blocks) |*r| {
        // `cvec` IS the conditioning `y`; silu it into the scratch rather than
        // in place, because every block reads the same untouched vector.
        try be.tensorCopy(bufs.nt, 0, bufs.cvec, 0, rows * HX * 4);
        try be.opSilu(bufs.nt, rows * HX);
        try be.opMatmul(bufs.nmod, 0, bufs.nt, 0, rows, r.adaln.w.bytes, false, 3 * HX, HX, r.adaln.w.scale, r.adaln.b);

        try be.opLayerNorm(bufs.nx, bufs.nt, r.in_ln.w, r.in_ln.b, rows, HX, eps, false);
        try be.opModulatePerRow(bufs.nt, bufs.nmod, rows, HX);
        try be.opMatmul(bufs.feat, 0, bufs.nt, 0, rows, r.fc1.w.bytes, false, HX, HX, r.fc1.w.scale, r.fc1.b);
        try be.opSilu(bufs.feat, rows * HX);
        try be.opMatmul(bufs.nt, 0, bufs.feat, 0, rows, r.fc2.w.bytes, false, HX, HX, r.fc2.w.scale, r.fc2.b);
        try be.opGatedAddPerRow(bufs.nx, bufs.nt, bufs.nmod, rows, HX);
    }

    try be.qkNorm(bufs.nx, bufs.nx, try vecBuf(be, model.final_norm), rows, HX, eps);
    try be.opMatmul(bufs.pix, 0, bufs.nx, 0, rows, model.final_lin.w.bytes, false, 3, HX, model.final_lin.w.scale, model.final_lin.b);
    try be.opPatchScatter(bufs.rgb, bufs.pix, nb, lat_w, mage_vae.patch, l0);
}

/// `x += conv2(silu(norm2(conv1(silu(norm1(x))))))`, width-preserving.
fn resnet(be: *Backend, bufs: *Bufs, norms: *NormBufs, h: usize, w: usize, r: mage_vae.Resnet) !void {
    const n = h * w;
    try groupNorm(be, bufs, norms, &bufs.t, &bufs.x, n, H, r.norm1, true);
    try conv(be, bufs, &bufs.u, &bufs.t, h, w, r.conv1);
    try groupNorm(be, bufs, norms, &bufs.t, &bufs.u, n, H, r.norm2, true);
    try conv(be, bufs, &bufs.u, &bufs.t, h, w, r.conv2);
    try be.opAdd(bufs.x, bufs.u, n * H);
}

/// `x += proj(windowed_attention(norm(x)))`.
///
/// Each 32x32 window attends to itself alone, and the gather lays the windows
/// out as attention HEADS so one batched call covers them all. Out-of-range
/// samples REPLICATE the edge, which is the reference's padding and is not the
/// same as attending over a smaller window.
fn winAttn(be: *Backend, bufs: *Bufs, norms: *NormBufs, h: usize, w: usize, a: mage_vae.WinAttn, nph: usize, npw: usize, n_win: usize) !void {
    const n = h * w;
    const d = mage_vae.attn_window;
    const tokens = d * d;
    _ = nph;
    try groupNorm(be, bufs, norms, &bufs.t, &bufs.x, n, H, a.norm, false);
    try conv(be, bufs, &bufs.aq, &bufs.t, h, w, a.q);
    try conv(be, bufs, &bufs.ak, &bufs.t, h, w, a.k);
    try conv(be, bufs, &bufs.av, &bufs.t, h, w, a.v);

    try be.opWinGather(bufs.wq, bufs.aq, h, w, H, npw, d, n_win);
    try be.opWinGather(bufs.wk, bufs.ak, h, w, H, npw, d, n_win);
    try be.opWinGather(bufs.wv, bufs.av, h, w, H, npw, d, n_win);

    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(H)));
    try be.attn(bufs.wq, bufs.wk, bufs.wv, bufs.wo, tokens, tokens, n_win, n_win, H, scale, false);

    try be.opWinScatter(bufs.ao, bufs.wo, h, w, H, npw, d, n_win);
    try conv(be, bufs, &bufs.t, &bufs.ao, h, w, a.proj);
    try be.opAdd(bufs.x, bufs.t, n * H);
}

fn groupNorm(be: *Backend, bufs: *Bufs, norms: *NormBufs, dst: *Buf, src: *const Buf, n: usize, ch: usize, nw: mage_vae.NormW, silu: bool) !void {
    const cat = try norms.get(nw);
    try be.opGroupNorm(src.*, dst.*, cat, bufs.gstat, bufs.gmi, n, ch, mage_vae.norm_groups, gn_chunks, eps, silu, false);
}

fn conv(be: *Backend, bufs: *Bufs, dst: *Buf, src: *const Buf, h: usize, w: usize, cv: Conv2d) !void {
    try sd_unet_cuda.convIntoPrec(be, &bufs.patch, dst, src, h, w, cv, .stride1, null, 1.0, false, false);
}

/// A non-owning device-pointer view at a byte offset, sized to what will be read.
fn view(b: Buf, off_bytes: usize, size: usize) Buf {
    return .{ .buf = @enumFromInt(@intFromEnum(b.buf) + off_bytes), .mem = .null_handle, .size = size };
}

/// Wrap a CPU f32 vector as a (pointer-cached) small device buffer.
fn vecBuf(be: *Backend, v: []const f32) !Buf {
    const handle = try be.smallBuffer(std.mem.sliceAsBytes(v));
    return .{ .buf = handle, .size = v.len * 4 };
}
