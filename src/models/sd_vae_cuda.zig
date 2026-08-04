//! GPU-resident SD-family VAE decode on the CUDA backends (AutoencoderKL).
//!
//! The CUDA analogue of `sd_vae_gpu`, and simpler than it in one place: the
//! mid-block's single 512-wide head goes straight through `opAttnTC`, which is
//! head_dim-generic and query-tiles itself when the scores plane would blow the
//! scratch budget (`opAttnTCFlash`, which it selects for exactly this shape —
//! one head over every latent position). The Vulkan path needs a scores shader
//! compiled for 512 and hand-rolled query tiling to reach the same place.
//!
//! Everything else is the same mapping as the UNet's, and the convolution and
//! GroupNorm helpers come from `sd_unet_cuda` rather than being copied.

const std = @import("std");
const sd_vae = @import("sd_vae.zig");
const sd_unet_cuda = @import("sd_unet_cuda.zig");
const ops = @import("tp_ops");
const cuda = @import("tp_gpu").cuda;

const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;
const Conv2d = ops.conv.Conv2d;
const Config = sd_vae.Config;

/// Column groups per GroupNorm statistics pass (matching the UNet's).
const gn_chunks: usize = 256;

const Bufs = struct {
    x: Buf = .{},
    t: Buf = .{},
    u: Buf = .{},
    patch: Buf = .{},
    gstat: Buf = .{},
    gmi: Buf = .{},
    /// Mid-block attention projections and output.
    aq: Buf = .{},
    ak: Buf = .{},
    av: Buf = .{},
    ao: Buf = .{},

    fn deinit(self: *Bufs, be: *Backend) void {
        inline for (@typeInfo(Bufs).@"struct".fields) |f| be.tensorDestroy(&@field(self, f.name));
    }
};

/// GroupNorm weight ++ bias concatenations on the device, cached by weight
/// pointer: `gn_apply` reads both out of one binding and the checkpoint stores
/// them as two tensors.
const NormBufs = struct {
    map: std.AutoHashMapUnmanaged(usize, Buf) = .empty,
    alloc: std.mem.Allocator,
    be: *Backend,

    fn get(self: *NormBufs, nw: sd_vae.GroupNormW) !Buf {
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
    }
};

/// Decode a latent (channel-last `[lat_h*lat_w][4]`, already divided by
/// `scaling_factor`) to channel-last `[8·lat_h * 8·lat_w][3]`. Caller frees.
pub fn decode(
    dec: *const sd_vae.Decoder,
    be: *Backend,
    gpa: std.mem.Allocator,
    z: []const f32,
    lat_h: usize,
    lat_w: usize,
    cancel: ?*std.atomic.Value(bool),
) ![]f32 {
    const cfg = dec.cfg;
    std.debug.assert(z.len == lat_h * lat_w * cfg.z_channels);
    var bufs: Bufs = .{};
    defer bufs.deinit(be);

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var norms: NormBufs = .{ .alloc = arena.allocator(), .be = be };
    defer norms.deinit();

    var h = lat_h;
    var w = lat_w;
    var ch = cfg.innermost();

    // ⚠️ Sizing mirrors the loader's own width bookkeeping: the first resnet of a
    // level reads the PREVIOUS level's (wider) input at the NEW (doubled)
    // resolution, so sizing off `block_out_channels` alone under-allocates by 2x.
    // See the same note in `sd_vae_gpu`, where it cost a wrong decode.
    var max_elems: usize = 0;
    {
        var lh = lat_h;
        var lw = lat_w;
        var c = cfg.innermost();
        for (0..cfg.levels()) |i| {
            const level_idx = cfg.levels() - 1 - i;
            const out_ch = cfg.block_out_channels[level_idx];
            max_elems = @max(max_elems, lh * lw * @max(c, out_ch));
            c = out_ch;
            if (level_idx > 0) {
                lh *= 2;
                lw *= 2;
            }
        }
        max_elems = @max(max_elems, lh * lw * @max(c, 3));
    }

    try be.ensureDeviceBuffer(&bufs.x, @max(z.len, max_elems) * 4);
    try be.tensorUpload(bufs.x, std.mem.sliceAsBytes(z));
    try be.ensureDeviceBuffer(&bufs.t, max_elems * 4);
    try be.ensureDeviceBuffer(&bufs.u, max_elems * 4);
    try be.ensureDeviceBuffer(&bufs.gstat, cfg.norm_groups * gn_chunks * 3 * 4);
    try be.ensureDeviceBuffer(&bufs.gmi, cfg.norm_groups * 2 * 4);
    inline for (.{ "aq", "ak", "av", "ao" }) |f| {
        try be.ensureDeviceBuffer(&@field(bufs, f), lat_h * lat_w * cfg.innermost() * 4);
    }

    try be.beginBatch();
    errdefer if (be.batching()) be.abortBatch();

    try conv(be, &bufs, &bufs.t, &bufs.x, h, w, dec.post_quant, .stride1);
    try conv(be, &bufs, &bufs.x, &bufs.t, h, w, dec.conv_in, .stride1);

    try resnet(be, &bufs, &norms, h, w, dec.mid1, cfg);
    try attn(be, &bufs, &norms, h, w, dec.mid_attn, cfg);
    try resnet(be, &bufs, &norms, h, w, dec.mid2, cfg);

    // The O(seq^2) attention scratch is dead now but sits at the START (the
    // mid-block runs at latent resolution) and would otherwise stay resident
    // through the whole 8x upsampling; flush and free it.
    try be.endBatch();
    be.freeAttnScratch();
    inline for (.{ "aq", "ak", "av", "ao" }) |f| be.tensorDestroy(&@field(bufs, f));
    try be.beginBatch();

    for (dec.levels) |level| {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        for (level.blocks) |b| {
            try resnet(be, &bufs, &norms, h, w, b, cfg);
            ch = b.out_ch;
        }
        if (level.upsample) |up| {
            try convResidual(be, &bufs, &bufs.t, &bufs.x, h, w, up, .upsample2x);
            std.mem.swap(Buf, &bufs.x, &bufs.t);
            h *= 2;
            w *= 2;
        }
    }

    try groupNorm(be, &bufs, &norms, &bufs.t, &bufs.x, h * w, ch, dec.norm_out, cfg, true);
    try conv(be, &bufs, &bufs.u, &bufs.t, h, w, dec.conv_out, .stride1);

    try be.endBatch();

    const rgb = try gpa.alloc(f32, h * w * 3);
    errdefer gpa.free(rgb);
    try be.tensorDownload(bufs.u, std.mem.sliceAsBytes(rgb));
    return rgb;
}

fn resnet(be: *Backend, bufs: *Bufs, norms: *NormBufs, h: usize, w: usize, r: sd_vae.Resnet, cfg: Config) !void {
    const n = h * w;
    const out_n = n * r.out_ch;
    try groupNorm(be, bufs, norms, &bufs.t, &bufs.x, n, r.in_ch, r.norm1, cfg, true);
    try conv(be, bufs, &bufs.u, &bufs.t, h, w, r.conv1, .stride1);
    try groupNorm(be, bufs, norms, &bufs.t, &bufs.u, n, r.out_ch, r.norm2, cfg, true); // consumes u
    try conv(be, bufs, &bufs.u, &bufs.t, h, w, r.conv2, .stride1); // reuses u
    if (r.nin) |nin| {
        try convResidual(be, bufs, &bufs.t, &bufs.x, h, w, nin, .stride1);
        try be.opAdd(bufs.u, bufs.t, out_n);
    } else {
        try be.opAdd(bufs.u, bufs.x, out_n);
    }
    std.mem.swap(Buf, &bufs.x, &bufs.u);
}

/// Mid-block attention: one head over all `ch` channels (512), so `opAttnTC`
/// takes it directly — and selects its own query-tiled path when the scores plane
/// exceeds the scratch budget, which at a 1024-square render it does.
fn attn(be: *Backend, bufs: *Bufs, norms: *NormBufs, h: usize, w: usize, ab: sd_vae.AttnBlock, cfg: Config) !void {
    const n = h * w;
    const ch = ab.channels;
    try groupNorm(be, bufs, norms, &bufs.t, &bufs.x, n, ch, ab.norm, cfg, false);
    try conv(be, bufs, &bufs.aq, &bufs.t, h, w, ab.q, .stride1);
    try conv(be, bufs, &bufs.ak, &bufs.t, h, w, ab.k, .stride1);
    try conv(be, bufs, &bufs.av, &bufs.t, h, w, ab.v, .stride1);
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(ch)));
    try be.opAttnTC(bufs.aq, bufs.ak, bufs.av, bufs.ao, n, 1, 1, ch, scale);
    // Reuse aq for the projection: its query content was consumed by the gather.
    try conv(be, bufs, &bufs.aq, &bufs.ao, h, w, ab.proj, .stride1);
    try be.opAdd(bufs.x, bufs.aq, n * ch);
}

fn groupNorm(
    be: *Backend,
    bufs: *Bufs,
    norms: *NormBufs,
    dst: *Buf,
    src: *const Buf,
    n: usize,
    ch: usize,
    nw: sd_vae.GroupNormW,
    cfg: Config,
    silu: bool,
) !void {
    const cat = try norms.get(nw);
    try be.opGroupNorm(src.*, dst.*, cat, bufs.gstat, bufs.gmi, n, ch, cfg.norm_groups, gn_chunks, cfg.norm_eps, silu);
}

fn conv(
    be: *Backend,
    bufs: *Bufs,
    dst: *Buf,
    src: *const Buf,
    h: usize,
    w: usize,
    cv: Conv2d,
    mode: sd_unet_cuda.SampleMode,
) !void {
    try sd_unet_cuda.convInto(be, &bufs.patch, dst, src, h, w, cv, mode);
}

/// Divisor applied to the activation of a convolution that reads the RESIDUAL
/// STREAM, before it is cast to f16 (`Backend.opConvF16Scaled` undoes it exactly).
///
/// ⚠️ Not defensive — measured. An SDXL VAE decoder's residual reaches **4.2e5**
/// (probed at a 64² latent: the last upsample conv's f32 output is 1.26e5 and the
/// next block's 1x1 shortcut reads it), against f16's ceiling of 65504. Without
/// this, that cast produced `inf`, the following GroupNorm turned it into NaN via
/// its mean, and **every SDXL render at 512² or larger came out solid white with no
/// error** on `cuda`, `zig-cuda` and `vulkan` alike. SD1.5's VAE peaks near 7e3, two
/// orders lower, which is why this was invisible for the family the code was
/// written against — and it is the same reason ComfyUI decodes the SDXL VAE in fp32
/// (or ships the "fp16-fix" weights) by default.
///
/// 256 is a power of two (so the scaling is exact — it only shifts the exponent and
/// f16 keeps all 11 mantissa bits) and leaves 38x headroom over the measured peak.
/// Its cost is that true values below 256·6e-8 = 1.5e-5 underflow to zero, which
/// against a residual whose peak is ~1e5 is 1e-10 relative — hence a modest divisor
/// rather than a blanket huge one.
///
/// ⚠️ Only the residual-reading convolutions need it, and that is a measured claim
/// too: every OTHER convolution here reads a GroupNorm output (peak 67 measured, and
/// bounded by |gamma|·O(1)+|beta| by construction) or the latent itself, so scaling
/// them would only cost precision at the bottom of the range.
const residual_act_div: f32 = 256.0;

/// `conv` for a convolution whose input is the residual stream — the 1x1 shortcut
/// and the level upsamples. See `residual_act_div`.
fn convResidual(
    be: *Backend,
    bufs: *Bufs,
    dst: *Buf,
    src: *const Buf,
    h: usize,
    w: usize,
    cv: Conv2d,
    mode: sd_unet_cuda.SampleMode,
) !void {
    try sd_unet_cuda.convIntoScaled(be, &bufs.patch, dst, src, h, w, cv, mode, residual_act_div);
}
