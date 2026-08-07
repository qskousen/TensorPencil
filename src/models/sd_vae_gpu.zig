//! GPU-resident SD-family VAE decode (Vulkan) — the AutoencoderKL decoder.
//!
//! Structurally this is `vae_gpu` (the Wan decoder) with two substitutions: the
//! per-position channel norm becomes **GroupNorm over 32 channel groups**, and the
//! mid-block's single head is 512 wide rather than 384. Everything else is the
//! same mapping, and the convolution and GroupNorm helpers are imported from
//! `sd_unet_gpu` rather than copied, so the two SD stages cannot drift apart.
//!
//! It matters more than a decoder usually would: with the UNet on the device, a
//! CPU decode was measured at **32 of the 47 seconds** of a 1024-square SDXL
//! render — the largest single cost left in the pipeline.
//!
//! ⚠️ **The mid-block attention is O(latent positions squared)**, and the latent
//! is where the decoder starts: 4096 positions at a 512-square render, 16384 at
//! 1024-square, whose full scores plane is 537 MB. It is query-tiled for exactly
//! that reason (the treatment `vae_gpu` gives the Wan mid-block), so the scratch
//! is bounded by the decode ladder's tiling regardless of resolution.
//!
//! Numerics are f16 tensor cores for the GEMMs, so this is not bit-identical to
//! `sd_vae.Decoder.decode`; the CPU path remains the reference.

const std = @import("std");
const sd_vae = @import("sd_vae.zig");
const sd_unet_gpu = @import("sd_unet_gpu.zig");
const ops = @import("tp_ops");
const gpu_context = @import("tp_gpu").context;

const Context = gpu_context.Context;
const DeviceBuffer = gpu_context.DeviceBuffer;
const Conv2d = ops.conv.Conv2d;
const Config = sd_vae.Config;

const none: DeviceBuffer = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 };

/// The head width the mid-block attends at: LDM's `AttnBlock` uses one head over
/// every channel, and the innermost level is 512 wide in both SD1.5 and SDXL.
/// `Context.opAttnScoresSd` is the scores shader compiled for it.
const attn_hd: usize = 512;

/// Scores-plane budget before the query dimension is tiled.

/// Interleaved chunks per row in the two-pass softmax.

const Bufs = struct {
    /// Running activation, plus two scratches (`u` carries both of a resnet's
    /// convolution outputs — conv1's is dead once norm2 has read it).
    x: DeviceBuffer = none,
    t: DeviceBuffer = none,
    u: DeviceBuffer = none,
    patch: DeviceBuffer = none,
    /// GroupNorm chunk statistics and the merged per-group {mean, inv}.
    gstat: DeviceBuffer = none,
    gmi: DeviceBuffer = none,
    /// Mid-block attention. `aq`/`ak`/`av` are the f32 projections, `qh`/`kh`/`vh`
    /// their f16 forms, `s`/`part`/`md` the scores plane and its softmax table,
    /// `ao` the f32 output. `qb`/`ob` are the query and output bands of the tiled
    /// path (see `attn`).
    aq: DeviceBuffer = none,
    ak: DeviceBuffer = none,
    av: DeviceBuffer = none,
    qh: DeviceBuffer = none,
    kh: DeviceBuffer = none,
    s: DeviceBuffer = none,
    ao: DeviceBuffer = none,

    fn deinit(self: *Bufs, ctx: *Context) void {
        inline for (@typeInfo(Bufs).@"struct".fields) |f| {
            ctx.tensorDestroy(&@field(self, f.name));
        }
    }
};

/// GroupNorm weight ++ bias concatenations, cached by weight pointer: `gn_apply`
/// reads both out of one binding, and the checkpoint stores them as two tensors.
/// The concatenated slice is what `Context.smallBuffer` keys on, so each also
/// uploads once.
const NormCats = struct {
    map: std.AutoHashMapUnmanaged(usize, []f32) = .empty,
    alloc: std.mem.Allocator,

    fn get(self: *NormCats, nw: sd_vae.GroupNormW) ![]const f32 {
        const key = @intFromPtr(nw.w.ptr);
        if (self.map.get(key)) |c| return c;
        const c = try self.alloc.alloc(f32, nw.w.len + nw.b.len);
        @memcpy(c[0..nw.w.len], nw.w);
        @memcpy(c[nw.w.len..], nw.b);
        try self.map.put(self.alloc, key, c);
        return c;
    }
};

/// Decode a latent (channel-last `[lat_h*lat_w][4]`, already divided by
/// `scaling_factor`) to channel-last `[8·lat_h * 8·lat_w][3]`. Caller frees.
pub fn decode(
    dec: *const sd_vae.Decoder,
    ctx: *Context,
    gpa: std.mem.Allocator,
    z: []const f32,
    lat_h: usize,
    lat_w: usize,
    cancel: ?*std.atomic.Value(bool),
) ![]f32 {
    const cfg = dec.cfg;
    std.debug.assert(z.len == lat_h * lat_w * cfg.z_channels);
    var bufs: Bufs = .{};
    defer bufs.deinit(ctx);

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var cats: NormCats = .{ .alloc = arena.allocator() };

    var h = lat_h;
    var w = lat_w;
    var ch = cfg.innermost();

    // Activation sizing, and it has to mirror the loader's own width bookkeeping
    // rather than just the per-level output widths.
    //
    // ⚠️ **The first resnet of a level reads the PREVIOUS level's (wider) input at
    // the NEW (doubled) resolution** — 512 channels over 1920 positions where the
    // level's own output is 256 — so sizing off `block_out_channels` alone
    // under-allocates by 2x. That is not a clean failure: `ensureDeviceBuffer`
    // would grow the buffer mid-batch, discarding the activation it holds and
    // freeing memory that recorded-but-unsubmitted dispatches still reference.
    // Measured as a rel L2 of 0.24 against the CPU decoder.
    // ⚠️ TWO widths, not one. `u` only ever receives a convolution's OUTPUT, so it
    // peaks one level narrower than `x`/`t` — see `sd_vae.activationElems`.
    const widths = sd_vae.activationElems(cfg, lat_h, lat_w);
    // The attention pads its rows to 128, so `x`/`t` must clear that too.
    const max_elems = @max(
        std.mem.alignForward(usize, lat_h * lat_w, 128) * cfg.innermost(),
        @as(usize, @intCast(widths.wide)),
    );

    try ctx.ensureDeviceBuffer(&bufs.x, @max(z.len, max_elems) * 4);
    try ctx.ensureDeviceBuffer(&bufs.t, max_elems * 4);
    // ⚠️ The upload target depends on whether the checkpoint HAS a
    // `post_quant_conv`: the Flux-lineage 16-channel VAE does not, so the latent is
    // staged straight into `t` — which is where `conv_in` reads from — instead of
    // going through a convolution that does not exist. Allocated before the upload
    // so `ensureDeviceBuffer` cannot move the buffer out from under it.
    try ctx.tensorUpload(if (dec.post_quant != null) bufs.x else bufs.t, std.mem.sliceAsBytes(z));
    try ctx.ensureDeviceBuffer(&bufs.u, @as(usize, @intCast(widths.out)) * 4);
    try ctx.ensureDeviceBuffer(&bufs.gstat, cfg.norm_groups * sd_unet_gpu.gn_chunks * 3 * 4);
    try ctx.ensureDeviceBuffer(&bufs.gmi, cfg.norm_groups * 2 * 4);

    // The mid-block's attention scratch, sized here rather than inside `attn`:
    // every buffer this decode touches is allocated BEFORE the batch opens, so no
    // `ensureDeviceBuffer` can reallocate a buffer a recorded dispatch references.
    {
        const nseq = lat_h * lat_w;
        inline for (.{ "aq", "ak", "av", "ao" }) |f| {
            try ctx.ensureDeviceBuffer(&@field(bufs, f), nseq * attn_hd * 4);
        }
        // k-major f32 copies of Q and K for the scores kernel.
        inline for (.{ "qh", "kh" }) |f| try ctx.ensureDeviceBuffer(&@field(bufs, f), nseq * attn_hd * 4);
        // ⚠️ The scores plane is **f32** (the Flux VAE's logits reach 9.95e6, 152x
        // past f16 — see `attn`) but only ONE QUERY BAND of it is ever resident:
        // `seq²` at a 1056x1584 render is 2.73 GB, more than the activations it
        // serves. `scoresBand` rows x `nseq` keys instead.
        try ctx.ensureDeviceBuffer(&bufs.s, scoresBand(nseq) * nseq * 4);
        // The im2col band for the widest 3x3 convolution, so `convInto` never has
        // to grow it either. The widest patch row is 9 x the innermost width.
        const patch_len = 9 * cfg.innermost();
        try ctx.ensureDeviceBuffer(&bufs.patch, sd_unet_gpu.convBand(max_elems, patch_len) * patch_len * 4);
    }

    try ctx.beginBatch();
    var batched = true;
    errdefer if (batched) ctx.endBatch() catch {};

    if (dec.post_quant) |pq| try conv(ctx, &bufs, &bufs.t, &bufs.x, h, w, pq, .stride1);
    try conv(ctx, &bufs, &bufs.x, &bufs.t, h, w, dec.conv_in, .stride1);

    try resnet(ctx, &bufs, &cats, h, w, dec.mid1, cfg);
    try attn(ctx, &bufs, &cats, h, w, dec.mid_attn, cfg);
    try resnet(ctx, &bufs, &cats, h, w, dec.mid2, cfg);

    // The attention scratch — above all the O(seq^2) scores plane — is dead for
    // the rest of the decode but sits at the START (the mid-block runs at latent
    // resolution) and would otherwise stay resident through the whole 8x
    // upsampling. `endBatch` submits and waits, so no recorded dispatch still
    // references these buffers when they are destroyed.
    batched = false;
    try ctx.endBatch();
    inline for (.{ "aq", "ak", "av", "qh", "kh", "s", "ao" }) |f| {
        ctx.tensorDestroy(&@field(bufs, f));
    }
    try ctx.beginBatch();
    batched = true;

    for (dec.levels) |level| {
        // Poll cancel between levels so a stop lands mid-decode; the errdefer
        // above flushes the in-flight batch on the way out.
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        for (level.blocks) |b| {
            try resnet(ctx, &bufs, &cats, h, w, b, cfg);
            ch = b.out_ch;
        }
        if (level.upsample) |up| {
            // Nearest 2x then a 3x3 convolution, with the resample fused into the
            // patch gather so the doubled tensor never exists.
            try convResidual(ctx, &bufs, &bufs.t, &bufs.x, h, w, up, .upsample2x);
            std.mem.swap(DeviceBuffer, &bufs.x, &bufs.t);
            h *= 2;
            w *= 2;
        }
    }

    try groupNorm(ctx, &bufs, &bufs.t, &bufs.x, h * w, ch, try cats.get(dec.norm_out), cfg, true);
    try conv(ctx, &bufs, &bufs.u, &bufs.t, h, w, dec.conv_out, .stride1);

    batched = false;
    try ctx.endBatch();

    const rgb = try gpa.alloc(f32, h * w * 3);
    errdefer gpa.free(rgb);
    try ctx.tensorDownload(bufs.u, std.mem.sliceAsBytes(rgb));

    // ⚠️ **A non-finite result is REPORTED, never returned.** `planarF32ToRgb8`
    // clamps NaN to white, so without this check an overflow anywhere above becomes a
    // solid white image with no error — which is exactly how the Flux/Z-Image VAE's
    // f16 limitation below was found, and how the SDXL VAE's was found before it.
    // `pipeline.recoverableDecodeErr` routes this into the decode ladder, whose CPU
    // tier is exact, so the render completes correctly instead of silently blank.
    //
    // ⚠️ **Known cause, and it is a real limitation of this file, not of the caller.**
    // The mid-block scores plane (`bufs.s`) is f16, and the *logits* — not the
    // activations — can exceed 65504. Measured at a 12x10 latent: the SD1.5 VAE's
    // attention q/k peak at 4.9 / 4.6, the Flux VAE's at **1255 / 573**, so its
    // 512-channel dot products overflow the f16 store while SD's sit three orders
    // below it. Everything upstream is finite and small (that VAE's activations are
    // *smaller* than SD's — 489 vs 929 at the last level), so no magnitude check
    // anywhere else can see it. The fix is an f32 scores plane for this one
    // attention, or folding a compensating divisor through `softmax_partial` /
    // `opAttnOut`; until then the Flux-lineage VAE decodes on the CPU.
    // (the `errdefer` above owns `rgb` on this path — do not free it here too)
    for (rgb) |v| {
        if (!std.math.isFinite(v)) return error.GpuDecodeNonFinite;
    }
    return rgb;
}

/// A resnet over `bufs.x` in place (result swapped back into `x`).
fn resnet(ctx: *Context, bufs: *Bufs, cats: *NormCats, h: usize, w: usize, r: sd_vae.Resnet, cfg: Config) !void {
    const n = h * w;
    const out_n = n * r.out_ch;
    try groupNorm(ctx, bufs, &bufs.t, &bufs.x, n, r.in_ch, try cats.get(r.norm1), cfg, true);
    try conv(ctx, bufs, &bufs.u, &bufs.t, h, w, r.conv1, .stride1);
    try groupNorm(ctx, bufs, &bufs.t, &bufs.u, n, r.out_ch, try cats.get(r.norm2), cfg, true); // consumes u
    try conv(ctx, bufs, &bufs.u, &bufs.t, h, w, r.conv2, .stride1); // reuses u
    if (r.nin) |nin| {
        try convResidual(ctx, bufs, &bufs.t, &bufs.x, h, w, nin, .stride1);
        try ctx.opElt(.add, bufs.u, bufs.t, null, null, .{ .u0 = @intCast(out_n) }, out_n, 1, 1);
    } else {
        try ctx.opElt(.add, bufs.u, bufs.x, null, null, .{ .u0 = @intCast(out_n) }, out_n, 1, 1);
    }
    std.mem.swap(DeviceBuffer, &bufs.x, &bufs.u);
}

/// Mid-block attention: one head over all `ch` channels, `x += proj(attn(norm(x)))`.
///
/// ⚠️ **The scores plane is f32, not the f16 the tensor-core path would give**, and
/// that is forced by measurement rather than caution. The Flux/Z-Image VAE's attention
/// logits reach **9.95e6** on a 12x10 latent (SD1.5's reach 8.3) — 152x past f16's
/// 65504 ceiling, and even without overflow f16's quantum up there is ~8000, which
/// would destroy a softmax whose differences are O(1). It produced a solid white image
/// with no error. Everything upstream is finite and small — that VAE's *activations*
/// are in fact smaller than SD's — so nothing but the logits themselves shows it.
///
/// The cost is one f32 O(seq²) plane on one attention at latent resolution;
/// `Decoder.estimatePeakBytes` reports it at 4 bytes so the decode ladder tiles when a
/// whole-image plane will not fit, which is what keeps a 1056x1584 render working.
///
/// ⚠️ **`attn_out` runs its OWN online softmax** over the raw scores, so there is no
/// softmax pass between it and `attn_scores`. Adding one exponentiates twice — finite,
/// plausible, and wrong by rel L2 0.26.
/// Query rows per scores band. A whole multiple of the 8-wide kernel tile (so a
/// band's last tile is never partial, which is what lets `attn_out` skip its row
/// clamp), and chosen so the band plane stays ~256 MB: at seq 26,136 that is
/// 2048 rows and 214 MB, against 2.73 GB for the whole plane.
fn scoresBand(n: usize) usize {
    const cap: usize = (256 << 20) / 4; // f32 entries we are willing to hold
    var qb = @max(@as(usize, 8), (cap / @max(n, 1)) & ~@as(usize, 7));
    if (qb > n) qb = std.mem.alignForward(usize, n, 8);
    return qb;
}

fn attn(ctx: *Context, bufs: *Bufs, cats: *NormCats, h: usize, w: usize, ab: sd_vae.AttnBlock, cfg: Config) !void {
    const n = h * w;
    const ch = ab.channels;
    std.debug.assert(ch == attn_hd);
    try groupNorm(ctx, bufs, &bufs.t, &bufs.x, n, ch, try cats.get(ab.norm), cfg, false);

    // q/k/v are 1x1 convolutions in LDM's AttnBlock, so no patch gather is involved
    // and these three can safely be distinct dedicated buffers.
    try conv(ctx, bufs, &bufs.aq, &bufs.t, h, w, ab.q, .stride1);
    try conv(ctx, bufs, &bufs.ak, &bufs.t, h, w, ab.k, .stride1);
    try conv(ctx, bufs, &bufs.av, &bufs.t, h, w, ab.v, .stride1);

    // Per-head k-major (one head here), so the scores kernel loads contiguously.
    ctx.independent(2);
    try ctx.opElt(.gather_kmajor, bufs.aq, null, null, bufs.qh, .{
        .u0 = @intCast(n * ch),
        .u1 = 1,
        .u2 = @intCast(ch),
        .u3 = @intCast(n),
    }, n * ch, 1, 1);
    try ctx.opElt(.gather_kmajor, bufs.ak, null, null, bufs.kh, .{
        .u0 = @intCast(n * ch),
        .u1 = 1,
        .u2 = @intCast(ch),
        .u3 = @intCast(n),
    }, n * ch, 1, 1);

    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(ch)));
    // ⚠️ **Query-BANDED, because the full plane is the largest thing in the decode.**
    // The mid-block attends over every latent position, so at a 132x198 latent
    // (a 1056x1584 render) seq is 26,136 and an f32 `seq²` plane is **2.73 GB** —
    // more than the activations it exists to serve. `scoresBand` caps it; the
    // arithmetic is unchanged because `attn_out` already runs an ONLINE softmax over
    // whole key rows, so a band of queries is exactly independent work.
    const qb = scoresBand(n);
    const jblk = std.math.divCeil(usize, n, 8) catch unreachable;
    var q0: usize = 0;
    while (q0 < n) : (q0 += qb) {
        const rows = @min(qb, n - q0);
        try ctx.opElt(.attn_scores, bufs.qh, bufs.kh, null, bufs.s, .{
            .u0 = @intCast(n),
            .u1 = 1,
            .u2 = 1,
            .u3 = @intCast(ch),
            .u4 = 0,
            .u5 = @intCast(q0),
            .u6 = @intCast(qb),
            .f0 = scale,
        }, jblk, std.math.divCeil(usize, rows, 8) catch unreachable, 1);
    // ⚠️ **No softmax pass here.** `attn_out` runs its OWN online softmax over the
    // raw scores (it tracks the running max and denominator as it streams j), so a
    // `softmax_rows` in between exponentiates twice — which is finite, plausible, and
    // wrong by rel L2 0.26. `dit_gpu`'s f32 branch has no softmax call for the same
    // reason; that absence reads as a missing step until you check the kernel.
        try ctx.opElt(.attn_out, bufs.s, null, bufs.av, bufs.ao, .{
            .u0 = @intCast(n),
            .u1 = 1,
            .u2 = 1,
            .u3 = @intCast(ch),
            .u4 = 0,
            .u5 = @intCast(n),
            // +1 so that 0 keeps meaning "not banded" — see the kernel.
            .u6 = @intCast(q0 + 1),
            .f0 = @bitCast(@as(u32, @intCast(qb * n))),
        }, ch / 8, std.math.divCeil(usize, rows, 8) catch unreachable, 1);
    }

    // Reuse aq for the projection: its query content was consumed by the gather.
    try conv(ctx, bufs, &bufs.aq, &bufs.ao, h, w, ab.proj, .stride1);
    try ctx.opElt(.add, bufs.x, bufs.aq, null, null, .{ .u0 = @intCast(n * ch) }, n * ch, 1, 1);
}

fn groupNorm(
    ctx: *Context,
    bufs: *Bufs,
    dst: *DeviceBuffer,
    src: *const DeviceBuffer,
    n: usize,
    ch: usize,
    cat: []const f32,
    cfg: Config,
    silu: bool,
) !void {
    try sd_unet_gpu.groupNormInto(ctx, bufs.gstat, bufs.gmi, dst, src, n, ch, cat, cfg.norm_groups, cfg.norm_eps, silu);
}

fn conv(
    ctx: *Context,
    bufs: *Bufs,
    dst: *DeviceBuffer,
    src: *const DeviceBuffer,
    h: usize,
    w: usize,
    cv: Conv2d,
    mode: sd_unet_gpu.SampleMode,
) !void {
    const oh = if (mode == .upsample2x) 2 * h else h;
    const ow = if (mode == .upsample2x) 2 * w else w;
    // Pre-sized in `decode`, deliberately: growing it here would be inside the
    // recording batch. Assert rather than silently reallocate.
    std.debug.assert(dst.size >= oh * ow * cv.co * 4);
    try sd_unet_gpu.convInto(ctx, &bufs.patch, dst, src, h, w, cv, mode, null);
}

/// Divisor applied to the activation of a convolution that reads the RESIDUAL
/// STREAM, before it is cast to f16 (`Context.opMatmulCoopF16WScaled` undoes it
/// exactly). The CUDA twin is `sd_vae_cuda.residual_act_div`, which carries the
/// full measurement.
///
/// ⚠️ Measured, not defensive: an SDXL VAE decoder's residual reaches **4.2e5**
/// against f16's 65504 ceiling, so without this the cast produced `inf`, the next
/// GroupNorm spread it to NaN through its mean, and every SDXL render at 512² or
/// larger came out solid white with no error. SD1.5's VAE peaks near 7e3, which is
/// why the family this code was written against never showed it.
///
/// 256 is a power of two, so the scaling is exact (exponent shift only, all 11
/// mantissa bits kept) with 38x headroom over the measured peak; the cost is that
/// true values below 1.5e-5 underflow to zero, i.e. 1e-10 of that peak. Only the
/// residual-reading convolutions get it — every other one here reads a GroupNorm
/// output (peak 67 measured), where a divisor would only cost low-end precision.
const residual_act_div: f32 = 256.0;

/// `conv` for a convolution whose input is the residual stream — the 1x1 shortcut
/// and the level upsamples. See `residual_act_div`.
fn convResidual(
    ctx: *Context,
    bufs: *Bufs,
    dst: *DeviceBuffer,
    src: *const DeviceBuffer,
    h: usize,
    w: usize,
    cv: Conv2d,
    mode: sd_unet_gpu.SampleMode,
) !void {
    const oh = if (mode == .upsample2x) 2 * h else h;
    const ow = if (mode == .upsample2x) 2 * w else w;
    std.debug.assert(dst.size >= oh * ow * cv.co * 4);
    try sd_unet_gpu.convIntoScaled(ctx, &bufs.patch, dst, src, h, w, cv, mode, null, residual_act_div);
}


// --- tests ------------------------------------------------------------------

test "gpu sd vae decode matches the cpu decoder, at 4 AND 16 latent channels" {
    // ⚠️ **The 16-channel arm is the one this test was missing**, and its absence let
    // a blank white Z-Image render through: the same decoder body serves SD's
    // 4-channel latent and the Flux/Z-Image 16-channel one, which additionally has
    // **no `post_quant_conv`** — so the device path has to stage the latent straight
    // into the `conv_in` input instead of through a convolution that does not exist.
    // A 4-channel-only test cannot see either difference.
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const test_gate = @import("../test_gate.zig");
    try test_gate.requireIntegration();
    std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    const ctx = Context.init(gpa) catch return error.SkipZigTest;
    defer ctx.deinit();

    const Case = struct { path: []const u8, prefix: []const u8, cfg: sd_vae.Config };
    const cases = [_]Case{
        .{
            .path = "/home/qt/genai/comfyui/models/checkpoints/sd1.5/perfectdeliberate_v20.safetensors",
            .prefix = "first_stage_model.",
            .cfg = sd_vae.sd15,
        },
        .{
            .path = "/home/qt/genai/comfyui/models/vae/ae.safetensors",
            .prefix = "",
            .cfg = sd_vae.flux,
        },
    };

    var ran: usize = 0;
    for (cases) |c| {
        test_gate.requireModelFile(io, c.path) catch continue;
        var st = try @import("tp_core").safetensors.SafeTensors.open(gpa, io, c.path);
        defer st.deinit();
        var dec = try sd_vae.Decoder.load(gpa, .{ .safetensors = &st }, c.cfg, c.prefix);
        defer dec.deinit();

        // 12x10, so the levels' extents are not powers of two and the odd-shape paths
        // in the fused upsample run.
        const lat_h = 12;
        const lat_w = 10;
        var prng = std.Random.DefaultPrng.init(29);
        const rand = prng.random();
        const z = try gpa.alloc(f32, lat_h * lat_w * c.cfg.z_channels);
        defer gpa.free(z);
        for (z) |*v| v.* = rand.floatNorm(f32);

        const want = try dec.decode(io, gpa, z, lat_h, lat_w);
        defer gpa.free(want);

        // ⚠️ **The 16-channel arm is EXPECTED to refuse today**, and pinning that is
        // the point: the Flux VAE's mid-block attention logits overflow the f16
        // scores plane (see `decode`'s note — q/k peak at 1255/573 against SD's
        // 4.9/4.6). `decode` reports it instead of returning a white image, and the
        // pipeline's ladder falls back to the exact CPU tier.
        //
        // When the f32-scores fix lands, THIS assertion is what tells you: flip
        // `expect_refusal` to false and the numeric comparison below starts running.
        const expect_refusal = false;
        const got = decode(&dec, ctx, gpa, z, lat_h, lat_w, null) catch |err| {
            errdefer std.debug.print("{d}-channel gpu decode failed with {t}\n", .{ c.cfg.z_channels, err });
            try std.testing.expect(expect_refusal);
            try std.testing.expectEqual(error.GpuDecodeNonFinite, err);
            ran += 1;
            continue;
        };
        defer gpa.free(got);
        errdefer std.debug.print("{d}-channel gpu decode unexpectedly succeeded\n", .{c.cfg.z_channels});
        try std.testing.expect(!expect_refusal);
        try std.testing.expectEqual(want.len, got.len);

        var num: f64 = 0;
        var den: f64 = 0;
        for (want, got) |e, a| {
            const d = @as(f64, e) - @as(f64, a);
            num += d * d;
            den += @as(f64, e) * @as(f64, e);
        }
        const rel = @sqrt(num / den);
        errdefer std.debug.print("sd vae gpu vs cpu ({d} ch) rel L2 {d:.6}\n", .{ c.cfg.z_channels, rel });
        // f16 tensor-core GEMMs through 12 resnets and an 8x upsample.
        try std.testing.expect(rel < 5e-3);
        ran += 1;
    }
    if (ran == 0) return error.SkipZigTest;
}
