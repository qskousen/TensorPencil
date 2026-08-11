//! GPU-resident SD-family VAE decode on the CUDA backends (AutoencoderKL).
//!
//! The CUDA analogue of `sd_vae_gpu`. The mid-block attention runs `be.attn`,
//! an f32 online softmax that materializes no scores plane, rather than the
//! tensor-core `opAttnTC`, because that path stores scores in f16 and the
//! Flux/Z-Image VAE's logits overflow it by 152x. See `attn` for the measurement.
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
const gn_chunks: usize = 32;

const Bufs = struct {
    x: Buf = .{},
    t: Buf = .{},
    u: Buf = .{},
    /// f32 on both ends of the decode: the incoming latent and the outgoing RGB.
    /// Separate from `x`/`t` because those are f16 under `Config.act_f16`.
    stage: Buf = .{},
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
/// `scaling_factor`) to channel-last `[8*lat_h * 8*lat_w][3]`. Caller frees.
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

    // Sizing mirrors the loader's own width bookkeeping: the first resnet of a
    // level reads the PREVIOUS level's (wider) input at the NEW (doubled)
    // resolution, so sizing off `block_out_channels` alone under-allocates by 2x.
    // See the same note in `sd_vae_gpu`, where it cost a wrong decode.
    var max_elems: usize = 0;
    // TWO widths, not one. `u` only ever receives a convolution's OUTPUT, so it
    // peaks one level narrower than `x`/`t`, see `sd_vae.activationElems`.
    const widths = sd_vae.activationElems(cfg, lat_h, lat_w);
    max_elems = @intCast(widths.wide);

    // `act_f16` halves the two widest buffers, which is where a whole-image
    // decode's gigabytes are: at 1056x1584 each holds 256 channels at FULL image
    // resolution (428M elements). The latent coming IN and the RGB going OUT are
    // f32 either way, so they get their own small staging buffer rather than
    // borrowing one of these.
    const asz: usize = if (cfg.act_f16) 2 else 4;
    try be.ensureDeviceBuffer(&bufs.x, max_elems * asz);
    try be.ensureDeviceBuffer(&bufs.t, max_elems * asz);
    try be.ensureDeviceBuffer(&bufs.stage, @max(z.len, lat_h * lat_w * 64 * 3) * 4);
    // The upload target depends on whether the checkpoint HAS a
    // `post_quant_conv`: the Flux-lineage 16-channel VAE does not, so the latent is
    // staged straight into `t`, which is where `conv_in` reads from, instead of
    // going through a convolution that does not exist. Allocated before the upload
    // so `ensureDeviceBuffer` cannot move the buffer out from under it.
    try be.tensorUpload(bufs.stage, std.mem.sliceAsBytes(z));
    try be.ensureDeviceBuffer(&bufs.u, @as(usize, @intCast(widths.out)) * asz);
    try be.ensureDeviceBuffer(&bufs.gstat, cfg.norm_groups * gn_chunks * 3 * 4);
    try be.ensureDeviceBuffer(&bufs.gmi, cfg.norm_groups * 2 * 4);
    inline for (.{ "aq", "ak", "av", "ao" }) |f| {
        try be.ensureDeviceBuffer(&@field(bufs, f), lat_h * lat_w * cfg.innermost() * 4);
    }

    try be.beginBatch();
    errdefer if (be.batching()) be.abortBatch();

    // The Flux-lineage 16-channel VAE has NO `post_quant_conv` (BFL dropped it),
    // so `conv_in` reads the staged latent directly; with one it runs first.
    const a16 = cfg.act_f16;
    if (dec.post_quant) |pq| {
        try conv(be, &bufs, &bufs.t, &bufs.stage, h, w, pq, .stride1, false, a16);
        try conv(be, &bufs, &bufs.x, &bufs.t, h, w, dec.conv_in, .stride1, a16, a16);
    } else {
        try conv(be, &bufs, &bufs.x, &bufs.stage, h, w, dec.conv_in, .stride1, false, a16);
    }

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
            try convResidual(be, &bufs, &bufs.t, &bufs.x, h, w, up, .upsample2x, cfg.act_f16);
            std.mem.swap(Buf, &bufs.x, &bufs.t);
            h *= 2;
            w *= 2;
        }
    }

    try groupNorm(be, &bufs, &norms, &bufs.t, &bufs.x, h * w, ch, dec.norm_out, cfg, true);
    // The head writes RGB back into the f32 staging buffer, the host wants f32.
    try conv(be, &bufs, &bufs.stage, &bufs.t, h, w, dec.conv_out, .stride1, a16, false);

    try be.endBatch();

    const rgb = try gpa.alloc(f32, h * w * 3);
    errdefer gpa.free(rgb);
    try be.tensorDownload(bufs.stage, std.mem.sliceAsBytes(rgb));
    return rgb;
}

/// `a += b` over two activation buffers in whichever format they are stored.
fn addAct(be: *Backend, a: Buf, b: Buf, total: usize, act16: bool) !void {
    if (act16) return be.opAddH16(a, b, total);
    return be.opAdd(a, b, total);
}

fn resnet(be: *Backend, bufs: *Bufs, norms: *NormBufs, h: usize, w: usize, r: sd_vae.Resnet, cfg: Config) !void {
    const n = h * w;
    const out_n = n * r.out_ch;
    const a16 = cfg.act_f16;
    try groupNorm(be, bufs, norms, &bufs.t, &bufs.x, n, r.in_ch, r.norm1, cfg, true);
    try conv(be, bufs, &bufs.u, &bufs.t, h, w, r.conv1, .stride1, a16, a16);
    try groupNorm(be, bufs, norms, &bufs.t, &bufs.u, n, r.out_ch, r.norm2, cfg, true); // consumes u
    try conv(be, bufs, &bufs.u, &bufs.t, h, w, r.conv2, .stride1, a16, a16); // reuses u
    if (r.nin) |nin| {
        try convResidual(be, bufs, &bufs.t, &bufs.x, h, w, nin, .stride1, a16);
        try addAct(be, bufs.u, bufs.t, out_n, a16);
    } else {
        try addAct(be, bufs.u, bufs.x, out_n, a16);
    }
    std.mem.swap(Buf, &bufs.x, &bufs.u);
}

/// A/B switch for the mid-block attention (see `attn`). `zimage-cuda-test` runs both.
pub var force_naive_attn: bool = false;

/// Mid-block attention: one head over all `ch` channels (512), so `opAttnTC`
/// takes it directly, and selects its own query-tiled path when the scores plane
/// exceeds the scratch budget, which at a 1024-square render it does.
fn attn(be: *Backend, bufs: *Bufs, norms: *NormBufs, h: usize, w: usize, ab: sd_vae.AttnBlock, cfg: Config) !void {
    const n = h * w;
    const ch = ab.channels;
    const a16 = cfg.act_f16;
    try groupNorm(be, bufs, norms, &bufs.t, &bufs.x, n, ch, ab.norm, cfg, false);
    // q/k/v/out stay f32: they are at latent resolution (tens of MB, not
    // where the memory goes) and both attention paths take f32 buffers.
    try conv(be, bufs, &bufs.aq, &bufs.t, h, w, ab.q, .stride1, a16, false);
    try conv(be, bufs, &bufs.ak, &bufs.t, h, w, ab.k, .stride1, a16, false);
    try conv(be, bufs, &bufs.av, &bufs.t, h, w, ab.v, .stride1, a16, false);
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(ch)));
    // `opAttnTC` on BOTH arms, given the f32 scores band. The constraint it has to
    // respect: a tensor-core path that materializes the scores plane in f16 cannot
    // serve this VAE, because the
    // Flux/Z-Image VAE's attention *logits* reach 9.95e6 on a 12x10 latent where
    // SD1.5's reach 8.3, 152x past f16's 65504 ceiling, and even without
    // overflow f16's quantum up there is ~8000, which destroys a softmax whose
    // differences are O(1). It rendered solid white with no error. Everything
    // upstream is finite and in fact *smaller* than SD's, so only the logits show it.
    //
    // `be.attn` keeps an online softmax in f32 and materializes no scores plane
    // at all, so it cannot overflow and needs no O(seq²) buffer.
    if (force_naive_attn) {
        try be.attn(bufs.aq, bufs.ak, bufs.av, bufs.ao, n, n, 1, 1, ch, scale, false);
    } else {
        try be.opAttnTC(bufs.aq, bufs.ak, bufs.av, bufs.ao, n, 1, 1, ch, scale);
    }
    // The projection lands in `t`, not back in `aq`: the residual add is against
    // `x`, which is f16 under `act_f16`, and `t` is the activation-format scratch.
    // (`t` is free here, the three projections consumed it.)
    try conv(be, bufs, &bufs.t, &bufs.ao, h, w, ab.proj, .stride1, false, a16);
    try addAct(be, bufs.x, bufs.t, n * ch, a16);
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
    try be.opGroupNorm(src.*, dst.*, cat, bufs.gstat, bufs.gmi, n, ch, cfg.norm_groups, gn_chunks, cfg.norm_eps, silu, cfg.act_f16);
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
    /// Storage format of `src` / `dst`. The running activations (`x`/`t`/`u`) are
    /// f16 under `cfg.act_f16`; the attention projections and the host-facing
    /// staging buffer stay f32, so a call has to say which it is touching.
    src16: bool,
    dst16: bool,
) !void {
    try sd_unet_cuda.convIntoPrec(be, &bufs.patch, dst, src, h, w, cv, mode, null, 1.0, src16, dst16);
}

/// Divisor applied to the activation of a convolution that reads the RESIDUAL
/// STREAM, before it is cast to f16 (`Backend.opConvF16Scaled` undoes it exactly).
///
/// Not defensive, measured. An SDXL VAE decoder's residual reaches 4.2e5
/// (probed at a 64² latent: the last upsample conv's f32 output is 1.26e5 and the
/// next block's 1x1 shortcut reads it), against f16's ceiling of 65504. Without
/// this, that cast produced `inf`, the following GroupNorm turned it into NaN via
/// its mean, and every SDXL render at 512² or larger came out solid white with no
/// error on `cuda`, `zig-cuda` and `vulkan` alike. SD1.5's VAE peaks near 7e3, two
/// orders lower, which is why this was invisible for the family the code was
/// written against, and it is the same reason ComfyUI decodes the SDXL VAE in fp32
/// (or ships the "fp16-fix" weights) by default.
///
/// 256 is a power of two (so the scaling is exact, it only shifts the exponent and
/// f16 keeps all 11 mantissa bits) and leaves 38x headroom over the measured peak.
/// Its cost is that true values below 256*6e-8 = 1.5e-5 underflow to zero, which
/// against a residual whose peak is ~1e5 is 1e-10 relative, hence a modest divisor
/// rather than a blanket huge one.
///
/// Only the residual-reading convolutions need it, and that is a measured claim
/// too: every OTHER convolution here reads a GroupNorm output (peak 67 measured, and
/// bounded by |gamma|*O(1)+|beta| by construction) or the latent itself, so scaling
/// them would only cost precision at the bottom of the range.
const residual_act_div: f32 = 256.0;

/// `conv` for a convolution whose input is the residual stream, the 1x1 shortcut
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
    act16: bool,
) !void {
    // Under f16 storage the divisor is 1.0, and that is not a shortcut: `act_div`
    // exists to bring a value too big for f16 *through* the cast, and an f16 buffer
    // could never have held that value. The two answer the same question and
    // `Config.act_f16` is off exactly where the divisor is load-bearing (SDXL).
    const div: f32 = if (act16) 1.0 else residual_act_div;
    try sd_unet_cuda.convIntoPrec(be, &bufs.patch, dst, src, h, w, cv, mode, null, div, act16, act16);
}
