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
//! is bounded by `scores_cap` regardless of resolution.
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
const scores_cap: usize = 512 << 20;

/// Interleaved chunks per row in the two-pass softmax.
const nchunks: u32 = 32;

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
    vh: DeviceBuffer = none,
    s: DeviceBuffer = none,
    part: DeviceBuffer = none,
    md: DeviceBuffer = none,
    ao: DeviceBuffer = none,
    qb: DeviceBuffer = none,
    ob: DeviceBuffer = none,

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
    var max_elems = std.mem.alignForward(usize, lat_h * lat_w, 128) * cfg.innermost();
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
        // The head norms at `c` and emits 3 channels, both at the full resolution.
        max_elems = @max(max_elems, lh * lw * @max(c, 3));
    }

    try ctx.ensureDeviceBuffer(&bufs.x, @max(z.len, max_elems) * 4);
    try ctx.tensorUpload(bufs.x, std.mem.sliceAsBytes(z));
    try ctx.ensureDeviceBuffer(&bufs.t, max_elems * 4);
    try ctx.ensureDeviceBuffer(&bufs.u, max_elems * 4);
    try ctx.ensureDeviceBuffer(&bufs.gstat, cfg.norm_groups * sd_unet_gpu.gn_chunks * 3 * 4);
    try ctx.ensureDeviceBuffer(&bufs.gmi, cfg.norm_groups * 2 * 4);

    // The mid-block's attention scratch, sized here rather than inside `attn`:
    // every buffer this decode touches is allocated BEFORE the batch opens, so no
    // `ensureDeviceBuffer` can reallocate a buffer a recorded dispatch references.
    {
        const seq_pad = std.mem.alignForward(usize, lat_h * lat_w, 128);
        const rows = seq_pad * attn_hd;
        const qbn = queryBand(seq_pad);
        inline for (.{ "aq", "ak", "av" }) |f| {
            try ctx.ensureDeviceBuffer(&@field(bufs, f), lat_h * lat_w * attn_hd * 4);
        }
        inline for (.{ "qh", "kh", "vh" }) |f| try ctx.ensureDeviceBuffer(&@field(bufs, f), rows * 2);
        try ctx.ensureDeviceBuffer(&bufs.ao, rows * 4);
        try ctx.ensureDeviceBuffer(&bufs.s, qbn * seq_pad * 2);
        try ctx.ensureDeviceBuffer(&bufs.part, qbn * nchunks * 2 * 4);
        try ctx.ensureDeviceBuffer(&bufs.md, qbn * 2 * 4);
        if (qbn < seq_pad) {
            try ctx.ensureDeviceBuffer(&bufs.qb, qbn * attn_hd * 2);
            try ctx.ensureDeviceBuffer(&bufs.ob, qbn * attn_hd * 4);
        }
        // The im2col band for the widest 3x3 convolution, so `convInto` never has
        // to grow it either. The widest patch row is 9 x the innermost width.
        const patch_len = 9 * cfg.innermost();
        try ctx.ensureDeviceBuffer(&bufs.patch, sd_unet_gpu.convBand(max_elems, patch_len) * patch_len * 4);
    }

    try ctx.beginBatch();
    var batched = true;
    errdefer if (batched) ctx.endBatch() catch {};

    try conv(ctx, &bufs, &bufs.t, &bufs.x, h, w, dec.post_quant, .stride1);
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
    inline for (.{ "aq", "ak", "av", "qh", "kh", "vh", "s", "part", "md", "ao", "qb", "ob" }) |f| {
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
/// The head is 512 wide, which is 4x the 128-column group `buildGemmAttnOut`
/// covers, so P@V runs as four fake heads sharing one scores/MD plane (`u1 = 0`,
/// `f1 = 0` — the trick `vae_gpu` uses for its 384-wide head as three). Queries
/// are tiled whenever the plane exceeds `scores_cap`, which at a 1024-square
/// render it does by half a gigabyte. Bands are copied in and out rather than
/// bound at an offset, because the ops' push constants carry no buffer offset.
fn attn(ctx: *Context, bufs: *Bufs, cats: *NormCats, h: usize, w: usize, ab: sd_vae.AttnBlock, cfg: Config) !void {
    const n = h * w;
    const ch = ab.channels;
    std.debug.assert(ch == attn_hd);
    try groupNorm(ctx, bufs, &bufs.t, &bufs.x, n, ch, try cats.get(ab.norm), cfg, false);

    const seq_pad = std.mem.alignForward(usize, n, 128);
    const rows = seq_pad * ch;
    // q/k/v are 1x1 convolutions in LDM's AttnBlock, so no patch gather is
    // involved and these three can safely be distinct dedicated buffers.
    try conv(ctx, bufs, &bufs.aq, &bufs.t, h, w, ab.q, .stride1);
    try conv(ctx, bufs, &bufs.ak, &bufs.t, h, w, ab.k, .stride1);
    try conv(ctx, bufs, &bufs.av, &bufs.t, h, w, ab.v, .stride1);

    // f16 operands: Q with the softmax scale prefolded and zero pad rows, K
    // k-major with zero pad columns, V with zero pad rows (so padded-j
    // probabilities contribute nothing to P@V).
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(ch)));
    ctx.independent(3);
    try ctx.opElt(.head_pad_h16, bufs.aq, null, null, bufs.qh, .{
        .u0 = @intCast(rows / 2),
        .u1 = @intCast(ch),
        .u2 = @intCast(ch),
        .u3 = @intCast(n),
        .u4 = 1,
        .f0 = scale,
    }, rows / 2, 1, 1);
    try ctx.opElt(.gather_kmajor_h16, bufs.ak, null, null, bufs.kh, .{
        .u0 = @intCast(rows / 2),
        .u1 = @intCast(ch),
        .u2 = @intCast(seq_pad),
        .u3 = @intCast(n),
        .u4 = 1,
    }, rows / 2, 1, 1);
    try ctx.opElt(.head_pad_h16, bufs.av, null, null, bufs.vh, .{
        .u0 = @intCast(rows / 2),
        .u1 = @intCast(ch),
        .u2 = @intCast(ch),
        .u3 = @intCast(n),
        .u4 = 1,
        .f0 = 1.0,
    }, rows / 2, 1, 1);

    const qbn = queryBand(seq_pad);
    const tiled = qbn < seq_pad;

    var q0: usize = 0;
    while (q0 < seq_pad) : (q0 += qbn) {
        const band = @min(qbn, seq_pad - q0);
        const valid = @min(band, n -| q0); // query rows in this band that are real
        const q_src = if (tiled) bufs.qb else bufs.qh;
        const o_dst = if (tiled) bufs.ob else bufs.ao;
        if (tiled) {
            // f16 pairs copied as f32 words; ch is even, so the row offset is exact.
            try ctx.opElt(.copy, bufs.qh, bufs.qb, null, null, .{
                .u0 = @intCast(band * ch / 2),
                .u2 = 0,
                .u3 = @intCast(q0 * ch / 2),
            }, band * ch / 2, 1, 1);
        }
        try ctx.opAttnScoresSd(bufs.s, q_src, bufs.kh, .{
            .u0 = @intCast(ch),
            .u1 = @intCast(seq_pad),
            .u2 = 0,
            .u3 = 1,
            .u4 = @intCast(ch * seq_pad),
            .u5 = @intCast(seq_pad * seq_pad),
        }, seq_pad / 128, band / 128, 1);
        if (valid > 0) {
            try ctx.opElt(.softmax_partial, bufs.s, null, null, bufs.part, .{
                .u0 = @intCast(valid * nchunks),
                .u1 = nchunks,
                .u2 = @intCast(n),
                .u3 = @intCast(seq_pad),
                .u5 = 0,
            }, valid * nchunks, 1, 1);
            try ctx.opElt(.softmax_combine, bufs.part, null, null, bufs.md, .{
                .u0 = @intCast(valid),
                .u1 = nchunks,
                .u2 = @intCast(n),
                .u3 = @intCast(seq_pad),
            }, valid, 1, 1);
        }
        try ctx.opAttnOut(bufs.s, bufs.vh, o_dst, bufs.md, .{
            .u0 = @intCast(seq_pad),
            .u1 = 0,
            .u2 = 0,
            .u3 = 1,
            .u4 = @intCast(ch),
            .u5 = @intCast(ch),
            .f0 = @bitCast(@as(u32, @intCast(n))),
            .f1 = @bitCast(@as(u32, 0)),
        }, band / 128, ch / 128);
        if (tiled) {
            try ctx.opElt(.copy, bufs.ob, bufs.ao, null, null, .{
                .u0 = @intCast(band * ch),
                .u2 = @intCast(q0 * ch),
                .u3 = 0,
            }, band * ch, 1, 1);
        }
    }

    // Reuse aq for the projection: its query content was consumed at the f16 pad.
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

/// Largest 128-multiple query band whose scores plane fits `scores_cap`.
fn queryBand(seq_pad: usize) usize {
    var qbn: usize = (scores_cap / (seq_pad * 2)) & ~@as(usize, 127);
    if (qbn < 128) qbn = 128;
    if (qbn > seq_pad) qbn = seq_pad;
    return qbn;
}

// --- tests ------------------------------------------------------------------

test "gpu sd vae decode matches the cpu decoder" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const test_gate = @import("../test_gate.zig");
    try test_gate.requireIntegration();
    const ckpt = "/home/qt/genai/comfyui/models/checkpoints/sd1.5/perfectdeliberate_v20.safetensors";
    try test_gate.requireModelFile(io, ckpt);
    std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    const ctx = Context.init(gpa) catch return error.SkipZigTest;
    defer ctx.deinit();

    var st = try @import("tp_core").safetensors.SafeTensors.open(gpa, io, ckpt);
    defer st.deinit();
    var dec = try sd_vae.Decoder.load(gpa, .{ .safetensors = &st }, sd_vae.sd15, "first_stage_model.");
    defer dec.deinit();

    // 12x10, so the levels' extents are not powers of two and the odd-shape paths
    // in the fused upsample run.
    const lat_h = 12;
    const lat_w = 10;
    var prng = std.Random.DefaultPrng.init(29);
    const rand = prng.random();
    const z = try gpa.alloc(f32, lat_h * lat_w * 4);
    defer gpa.free(z);
    for (z) |*v| v.* = rand.floatNorm(f32);

    const want = try dec.decode(io, gpa, z, lat_h, lat_w);
    defer gpa.free(want);
    const got = try decode(&dec, ctx, gpa, z, lat_h, lat_w, null);
    defer gpa.free(got);
    try std.testing.expectEqual(want.len, got.len);

    var num: f64 = 0;
    var den: f64 = 0;
    for (want, got) |e, a| {
        const d = @as(f64, e) - @as(f64, a);
        num += d * d;
        den += @as(f64, e) * @as(f64, e);
    }
    const rel = @sqrt(num / den);
    errdefer std.debug.print("sd vae gpu vs cpu rel L2 {d:.6}\n", .{rel});
    // f16 tensor-core GEMMs through 12 resnets and an 8x upsample.
    try std.testing.expect(rel < 5e-3);
}
