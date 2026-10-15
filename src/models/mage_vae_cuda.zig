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
//! (`[win_area][n_win][ch]`) and runs `opAttnTCFlash`, whose scores band is f32, for
//! `sd_vae_cuda`'s reason: a VAE's attention logits can reach magnitudes f16 cannot
//! hold. The naive per-query `be.attn` was 88% of a 1024^2 decode.

const std = @import("std");
const mage_vae = @import("mage_vae.zig");
const conv_cuda = @import("conv_cuda.zig");
const mage_vae_dev = @import("mage_vae_dev.zig");
const lin_cuda = @import("lin_cuda.zig");
const sd_unet_cuda = @import("sd_unet_cuda.zig");
const ops = @import("tp_ops");
const cuda = @import("tp_gpu").cuda;
const dual_table = @import("tp_gpu").dual_table;

const MageVae = mage_vae.MageVae;
const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;
const Dt = cuda.backend.Dt;
const Conv2d = ops.conv.Conv2d;

const H = mage_vae.hidden; // 384
const HX = mage_vae.hidden_x; // 32
const FFN = mage_vae.ffn; // 1536
const PA = mage_vae.patch_area; // 256
const NERF_IN = mage_vae.nerf_in; // 99
const eps: f32 = 1e-6;
const gn_chunks: usize = 32;

/// Force the naive per-query attention in the windowed blocks, for an A/B.
pub var force_naive_attn: bool = false;

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
    /// `col_mean`'s partial sums.
    part: Buf = .{},
    mean: Buf = .{},
    /// The NeRF band's buffers.
    y: Buf = .{},
    feat: Buf = .{},
    nx: Buf = .{},
    nt: Buf = .{},
    cvec: Buf = .{},
    nmod: Buf = .{},
    pix: Buf = .{},

    /// What the dial narrows: every activation but the windowed attention's operands
    /// (`attn` reads f32), the latent, the RGB and the per-pixel MLP's residual `nx`,
    /// which reaches ~3e4 on a real render.
    const typed = [_][]const u8{ "x", "t", "u", "wide", "cat", "cond", "y", "feat", "nt", "cvec", "nmod" };

    fn setAct(self: *Bufs, act: Dt) void {
        inline for (typed) |f| @field(self, f).dt = act;
    }

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
pub fn codForward(model: *const MageVae, be: *Backend, gpa: std.mem.Allocator, z: []const f32, lat_h: usize, lat_w: usize, act: Dt) ![]f32 {
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
    bufs.setAct(act);

    try be.beginBatch();
    errdefer if (be.batching()) be.abortBatch();
    try mage_vae_dev.cod(Dev.init(be, &bufs, &norms, npw, n_win, act), model, lat_h, lat_w);
    // `x` is dead here and holds `n * H` f32.
    const out32 = if (act == .f32) bufs.cond else blk: {
        const x32 = bufs.x.as(.f32);
        try be.opCast(bufs.cond, x32, n * H, 0);
        break :blk x32;
    };
    try be.endBatch();

    const out = try gpa.alloc(f32, n * H);
    errdefer gpa.free(out);
    try be.tensorDownload(out32.slice(0, out.len * 4), std.mem.sliceAsBytes(out));
    return out;
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
    act: Dt,
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
    const band: usize = @min(mage_vae_dev.nerf_band, n);
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
    try be.ensureDeviceBuffer(&bufs.part, (std.math.divCeil(usize, n, dual_table.col_mean_chunk) catch unreachable) * H * 4);
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
    bufs.setAct(act);

    try mage_vae_dev.decode(Dev.init(be, &bufs, &norms, npw, n_win, act), model, lat_h, lat_w, cancel);

    const rgb = try gpa.alloc(f32, img_h * img_w * 3);
    errdefer gpa.free(rgb);
    try be.tensorDownload(bufs.rgb, std.mem.sliceAsBytes(rgb));
    return rgb;
}

/// The `mage_vae_dev` adapter.
const Dev = struct {
    be: *Backend,
    bufs: *Bufs,
    norms: *NormBufs,
    npw: usize,
    n_win: usize,
    plan: lin_cuda.Plan,

    /// The codec's linears at the loader's f32, or on the f16 tensor cores when the
    /// activations are f16.
    fn init(be: *Backend, bufs: *Bufs, norms: *NormBufs, npw: usize, n_win: usize, act: Dt) Dev {
        var p = lin_cuda.Plan.tower;
        p.dense = if (act == .f16) .f16 else .native;
        return .{ .be = be, .bufs = bufs, .norms = norms, .npw = npw, .n_win = n_win, .plan = p };
    }

    pub fn beginBatch(d: Dev) !void {
        try d.be.beginBatch();
    }

    pub fn endBatch(d: Dev) !void {
        try d.be.endBatch();
    }

    pub fn abortBatch(d: Dev) void {
        if (d.be.batching()) d.be.abortBatch();
    }

    pub fn freeAttn(d: Dev) void {
        d.be.freeAttnScratch();
        inline for (.{ "aq", "ak", "av", "ao", "wq", "wk", "wv", "wo" }) |f| d.be.tensorDestroy(&@field(d.bufs, f));
    }

    pub fn zero(d: Dev, x: Buf, n: usize) !void {
        try d.be.tensorZero(x, n * x.dt.bytes());
    }

    pub fn concatCh(d: Dev, src: Buf, dst: Buf, n: usize, c_src: usize, c_dst: usize, off: usize) !void {
        try d.be.opConcatCh(src, dst, n, c_src, c_dst, off);
    }

    pub fn conv(d: Dev, dst: *Buf, src: *const Buf, h: usize, w: usize, cv: Conv2d) !void {
        try conv_cuda.conv2d(d.be, &d.bufs.patch, dst, src, h, w, cv, .stride1, null, 1.0);
    }

    pub fn groupNorm(d: Dev, dst: *Buf, src: *const Buf, n: usize, nw: mage_vae.NormW, with_silu: bool) !void {
        try d.be.opGroupNorm(src.*, dst.*, try d.norms.get(nw), d.bufs.gstat, d.bufs.gmi, n, H, mage_vae.norm_groups, gn_chunks, eps, with_silu);
    }

    pub fn add(d: Dev, x: Buf, y: Buf, n: usize) !void {
        try d.be.opAdd(x, y, n);
    }

    /// The gather lays the windows out as attention HEADS so one batched call covers
    /// them all.
    pub fn windows(d: Dev, h: usize, w: usize) !void {
        const b = d.bufs;
        const win = mage_vae.attn_window;
        const tokens = win * win;
        try d.be.opWinGather(b.wq, b.aq, h, w, H, d.npw, win, d.n_win);
        try d.be.opWinGather(b.wk, b.ak, h, w, H, d.npw, win, d.n_win);
        try d.be.opWinGather(b.wv, b.av, h, w, H, d.npw, win, d.n_win);
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(H)));
        if (force_naive_attn) {
            try d.be.attn(b.wq, b.wk, b.wv, b.wo, tokens, tokens, d.n_win, d.n_win, H, scale, false);
        } else {
            try d.be.opAttnTCFlash(b.wq, b.wk, b.wv, b.wo, tokens, d.n_win, H, scale);
        }
        try d.be.opWinScatter(b.ao, b.wo, h, w, H, d.npw, win, d.n_win);
    }

    pub fn lnMod(d: Dev, x: Buf, y: Buf, n: usize, shift_off: usize, scale_off: usize) !void {
        try d.be.lnMod(x, y, d.bufs.mods, n, H, shift_off, scale_off, eps);
    }

    pub fn depthwise3(d: Dev, dst: Buf, src: Buf, k: []const f32, bias: []const f32, h: usize, w: usize) !void {
        try d.be.opDepthwise3(dst, src, try vecBuf(d.be, k), try vecBuf(d.be, bias), h, w, H);
    }

    pub fn gelu(d: Dev, x: Buf, n: usize) !void {
        try d.be.geluErf(x, n);
    }

    pub fn colMean(d: Dev, dst: Buf, src: Buf, n: usize) !void {
        try d.be.opColMean(dst, src, d.bufs.part, n, H);
    }

    pub fn linear(d: Dev, y: Buf, x: Buf, x_off: usize, rows: usize, l: mage_vae.LinearW, out: usize, in: usize) !void {
        std.debug.assert(l.w.rows == out and l.w.cols == in);
        const eb = x.dt.bytes();
        const xs = if (x_off == 0) x else x.slice(x_off * eb, rows * in * eb);
        try lin_cuda.linear(d.be, d.plan, y, xs, rows, l.w, l.b);
    }

    pub fn mulColsSigmoid(d: Dev, x: Buf, gate: Buf, n: usize) !void {
        try d.be.opMulColsSigmoid(x, gate, n, H);
    }

    pub fn gatedAdd(d: Dev, x: Buf, delta: Buf, total: usize, gate_off: usize) !void {
        try d.be.gatedAdd(x, delta, d.bufs.mods, total, H, gate_off);
    }

    pub fn nerfFeat(d: Dev, feat: Buf, y: Buf, nb: usize) !void {
        try d.be.opNerfFeat(feat, y, d.bufs.dct, nb, PA, HX, mage_vae.dct_dim);
    }

    pub fn copy(d: Dev, dst: Buf, src: Buf, n: usize) !void {
        if (dst.dt != src.dt) return d.be.opCast(src, dst, n, 0);
        try d.be.tensorCopy(dst, 0, src, 0, n * dst.dt.bytes());
    }

    pub fn silu(d: Dev, x: Buf, n: usize) !void {
        try d.be.opSilu(x, n);
    }

    pub fn layerNorm(d: Dev, x: Buf, y: Buf, nw: mage_vae.NormW, rows: usize, cols: usize) !void {
        try d.be.opLayerNorm(x, y, nw.w, nw.b, rows, cols, eps);
    }

    pub fn modulatePerRow(d: Dev, x: Buf, mod: Buf, rows: usize, cols: usize) !void {
        try d.be.opModulatePerRow(x, mod, rows, cols);
    }

    pub fn gatedAddPerRow(d: Dev, x: Buf, delta: Buf, mod: Buf, rows: usize, cols: usize) !void {
        try d.be.opGatedAddPerRow(x, delta, mod, rows, cols);
    }

    pub fn rmsNorm(d: Dev, x: Buf, w: []const f32, rows: usize, cols: usize) !void {
        try d.be.qkNorm(x, x, try vecBuf(d.be, w), rows, cols, eps);
    }

    pub fn patchScatter(d: Dev, rgb: Buf, pix: Buf, nb: usize, lat_w: usize, l0: usize) !void {
        try d.be.opPatchScatter(rgb, pix, nb, lat_w, mage_vae.patch, l0);
    }

    /// `mage_vae_dev.mag`. Costs a sync per trace.
    pub fn trace(d: Dev, name: []const u8, x: Buf, n: usize) void {
        if (!mage_vae_dev.mag.on) return;
        traceInto(d.be, name, x, n) catch |err| std.log.warn("TP_VAE_MAG {s}: {t}", .{ name, err });
    }
};

fn traceInto(be: *Backend, name: []const u8, x: Buf, n: usize) !void {
    const was = be.batching();
    if (was) try be.endBatch();
    defer if (was) be.beginBatch() catch {};
    const raw = try std.heap.smp_allocator.alloc(u8, n * x.dt.bytes());
    defer std.heap.smp_allocator.free(raw);
    try be.tensorDownload(x.slice(0, raw.len), raw);
    mage_vae_dev.mag.record(name, raw, x.dt);
}

/// Wrap a CPU f32 vector as a (pointer-cached) small device buffer.
fn vecBuf(be: *Backend, v: []const f32) !Buf {
    const handle = try be.smallBuffer(std.mem.sliceAsBytes(v));
    return .{ .buf = handle, .size = v.len * 4 };
}
