//! GPU-resident VAE decode (Vulkan): the AutoencoderKL (`decode`, over `sd_vae_dev`)
//! and the Wan 2.1 decoder (`decodeWan`, over `wan_vae_dev`), one adapter. The two
//! differ in their norm (GroupNorm against a per-position channel L2 norm) and their
//! mid-block head width (512 against 384). The convolution and GroupNorm helpers are
//! `sd_unet_gpu`'s, so the VAE and the UNet cannot drift apart.
//!
//! It matters more than a decoder usually would: with the UNet on the device, a
//! CPU decode was measured at 32 of the 47 seconds of a 1024-square SDXL
//! render, the largest single cost left in the pipeline.
//!
//! The mid-block attention is O(latent positions squared), and the latent
//! is where the decoder starts: 4096 positions at a 512-square render, 16384 at
//! 1024-square, whose full scores plane is 537 MB. It is query-tiled for exactly
//! that reason (the treatment `vae_gpu` gives the Wan mid-block), so the scratch
//! is bounded by the decode ladder's tiling regardless of resolution.
//!
//! Every buffer is sized before the batch opens: growing one mid-batch discards the
//! activation it holds and frees memory recorded dispatches still reference.
//!
//! Numerics are f16 tensor cores for the GEMMs, so this is not bit-identical to
//! `sd_vae.Decoder.decode`; the CPU path remains the reference.

const std = @import("std");
const build_options = @import("build_options");
const sd_vae = @import("sd_vae.zig");
const conv_gpu = @import("conv_gpu.zig");
const sd_unet_gpu = @import("sd_unet_gpu.zig");
const dev = @import("sd_vae_dev.zig");
const wan_vae = @import("wan_vae.zig");
const wan_vae_dev = @import("wan_vae_dev.zig");
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
    /// convolution outputs, conv1's is dead once norm2 has read it).
    x: DeviceBuffer = none,
    /// f32 on both ends of the decode: the incoming latent and the outgoing RGB.
    /// Separate from `x`/`t` because those are at the chain's `dt`.
    stage: DeviceBuffer = none,
    t: DeviceBuffer = none,
    u: DeviceBuffer = none,
    patch: DeviceBuffer = none,
    /// GroupNorm chunk statistics and the merged per-group {mean, inv}.
    gstat: DeviceBuffer = none,
    gmi: DeviceBuffer = none,
    /// Mid-block attention. `aq`/`ak`/`av` are the f32 projections, `ao` the f32
    /// output, `s` one query band of the scores plane. `qh`/`kh` are k-major f32
    /// copies of Q and K on the f32 path, and f16 Q and k-major K on the tensor-core
    /// one, which also keeps f16 V in `vh` and the softmax table in `pt`/`md`.
    aq: DeviceBuffer = none,
    ak: DeviceBuffer = none,
    av: DeviceBuffer = none,
    qh: DeviceBuffer = none,
    kh: DeviceBuffer = none,
    vh: DeviceBuffer = none,
    s: DeviceBuffer = none,
    pt: DeviceBuffer = none,
    md: DeviceBuffer = none,
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
/// `scaling_factor`) to channel-last `[8*lat_h * 8*lat_w][3]`. Caller frees.
pub fn decode(
    dec: *const sd_vae.Decoder,
    ctx: *Context,
    gpa: std.mem.Allocator,
    z: []const f32,
    lat_h: usize,
    lat_w: usize,
    cancel: ?*std.atomic.Value(bool),
    /// The activation chain's element type; null takes the architecture's
    /// (`Config.act_f16`).
    act_override: ?gpu_context.Dt,
) ![]f32 {
    const cfg = dec.cfg;
    std.debug.assert(z.len == lat_h * lat_w * cfg.z_channels);
    var bufs: Bufs = .{};
    defer bufs.deinit(ctx);

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var cats: NormCats = .{ .alloc = arena.allocator() };

    // Activation sizing, and it has to mirror the loader's own width bookkeeping
    // rather than just the per-level output widths.
    //
    // The first resnet of a level reads the PREVIOUS level's (wider) input at
    // the NEW (doubled) resolution, 512 channels over 1920 positions where the
    // level's own output is 256, so sizing off `block_out_channels` alone
    // under-allocates by 2x. That is not a clean failure: `ensureDeviceBuffer`
    // would grow the buffer mid-batch, discarding the activation it holds and
    // freeing memory that recorded-but-unsubmitted dispatches still reference.
    // Measured as a rel L2 of 0.24 against the CPU decoder.
    // TWO widths, not one. `u` only ever receives a convolution's OUTPUT, so it
    // peaks one level narrower than `x`/`t`, see `sd_vae.activationElems`.
    const widths = sd_vae.activationElems(cfg, lat_h, lat_w);
    // The attention pads its rows to 128, so `x`/`t` must clear that too.
    const max_elems = @max(
        std.mem.alignForward(usize, lat_h * lat_w, 128) * cfg.innermost(),
        @as(usize, @intCast(widths.wide)),
    );

    // A 16-bit chain halves the two widest buffers, at 1056x1584 each holds 256
    // channels at FULL image resolution (428M elements), which is where a
    // whole-image decode's gigabytes are. The latent coming IN and the RGB going
    // OUT stay f32, in their own small staging buffer, and so does the attention.
    const act: gpu_context.Dt = act_override orelse if (cfg.act_f16) .f16 else .f32;
    inline for (.{ "x", "t", "u" }) |f| @field(bufs, f).dt = act;
    const asz = act.bytes();
    try ctx.ensureDeviceBuffer(&bufs.x, max_elems * asz);
    try ctx.ensureDeviceBuffer(&bufs.t, max_elems * asz);
    try ctx.ensureDeviceBuffer(&bufs.stage, @max(z.len, lat_h * lat_w * 64 * 3) * 4);
    try ctx.tensorUpload(bufs.stage, std.mem.sliceAsBytes(z));
    try ctx.ensureDeviceBuffer(&bufs.u, @as(usize, @intCast(widths.out)) * asz);
    try ctx.ensureDeviceBuffer(&bufs.gstat, cfg.norm_groups * sd_unet_gpu.gn_chunks * 3 * 4);
    try ctx.ensureDeviceBuffer(&bufs.gmi, cfg.norm_groups * 2 * 4);

    // f32 scores (the Flux VAE's logits reach 9.95e6, 152x past f16), one query band
    // of them: the whole `seq²` at a 1056x1584 render is 2.73 GB.
    try sizeAttn(ctx, &bufs, lat_h * lat_w, attn_hd);
    // The widest patch row is 9 x the innermost width.
    const patch_len = 9 * cfg.innermost();
    try ctx.ensureDeviceBuffer(&bufs.patch, conv_gpu.convBand(max_elems, patch_len) * patch_len * 4);

    try ctx.beginBatch();
    errdefer if (ctx.batching) ctx.endBatch() catch {};
    const ext = try dev.body(Dev{ .ctx = ctx, .bufs = &bufs, .cats = &cats, .groups = cfg.norm_groups, .eps = cfg.norm_eps }, dec, lat_h, lat_w, cancel);
    try ctx.endBatch();

    const rgb = try gpa.alloc(f32, ext[0] * ext[1] * 3);
    errdefer gpa.free(rgb);
    try ctx.tensorDownload(bufs.stage, std.mem.sliceAsBytes(rgb));
    return rgb;
}

/// Decode a Wan 2.1 latent (planar `[16][zh][zw]`, already denormalized) to planar
/// `[3][8*zh][8*zw]` pixels in [-1, 1], the chain at `act`. Reads repacked conv weights
/// (`wan_vae.Decoder.repackConvs`). Caller frees.
pub fn decodeWan(dec: *const wan_vae.Decoder, ctx: *Context, gpa: std.mem.Allocator, z: []const f32, zh: usize, zw: usize, cancel: ?*std.atomic.Value(bool), act: gpu_context.Dt) ![]f32 {
    std.debug.assert(z.len == wan_vae.latent_channels * zh * zw);
    var bufs: Bufs = .{};
    defer bufs.deinit(ctx);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var cats: NormCats = .{ .alloc = arena.allocator() };

    const n0 = zh * zw;
    const c = dec.mid_attn.proj.co;
    const widths = dec.activationElems(zh, zw);
    const wide: usize = @intCast(widths.wide);
    const asz = act.bytes();
    inline for (.{ "x", "t", "u" }) |f| @field(bufs, f).dt = act;
    try ctx.ensureDeviceBuffer(&bufs.x, wide * asz);
    try ctx.ensureDeviceBuffer(&bufs.t, wide * asz);
    try ctx.ensureDeviceBuffer(&bufs.u, @as(usize, @intCast(widths.out)) * asz);
    try ctx.ensureDeviceBuffer(&bufs.stage, @max(z.len, n0 * 64 * 3) * 4);
    {
        const rows = try wan_vae.planarToRows(gpa, z, wan_vae.latent_channels, n0);
        defer gpa.free(rows);
        try ctx.tensorUpload(bufs.stage, std.mem.sliceAsBytes(rows));
    }
    // Tensor-core scores in f16, where the SD decoder needs f32: this decoder's logits
    // were measured below 50 on krea2 and Anima renders, three decades under f16's range.
    const coop = ctx.pipe_scores_vae != .null_handle and ctx.pipe_attn_out != .null_handle and c == 384;
    if (coop) try sizeCoopAttn(ctx, &bufs, n0, c) else try sizeAttn(ctx, &bufs, n0, c);
    var patch_len: usize = 0;
    {
        const convs = try @constCast(dec).convList(gpa);
        defer gpa.free(convs);
        for (convs) |cv| if (cv.k == 3) {
            patch_len = @max(patch_len, 9 * cv.ci);
        };
    }
    try ctx.ensureDeviceBuffer(&bufs.patch, conv_gpu.convBand(wide, patch_len) * patch_len * 4);

    try ctx.beginBatch();
    errdefer if (ctx.batching) ctx.endBatch() catch {};
    const ext = try wan_vae_dev.body(Dev{ .ctx = ctx, .bufs = &bufs, .cats = &cats, .coop_attn = coop }, dec, zh, zw, cancel);
    try ctx.endBatch();

    const n = ext[0] * ext[1];
    const rgb = try gpa.alloc(f32, n * 3);
    defer gpa.free(rgb);
    try ctx.tensorDownload(bufs.stage, std.mem.sliceAsBytes(rgb));
    return wan_vae.rowsToPlanar(gpa, rgb, 3, n);
}

/// The mid-block attention's scratch for `n` positions of a `c`-wide head: f32
/// projections and output, k-major f32 copies of Q and K, and one query band of the f32
/// scores plane (see `Dev.attnCore`).
fn sizeAttn(ctx: *Context, bufs: *Bufs, n: usize, c: usize) !void {
    inline for (.{ "aq", "ak", "av", "ao", "qh", "kh" }) |f| try ctx.ensureDeviceBuffer(&@field(bufs, f), n * c * 4);
    try ctx.ensureDeviceBuffer(&bufs.s, sd_vae.scoresBand(n) * n * 4);
}

/// Query rows per band of the tensor-core path's f16 scores plane, a multiple of 128.
fn coopBand(n: usize) usize {
    const seq_pad = std.mem.alignForward(usize, n, 128);
    const qb = @max(128, ((256 << 20) / (seq_pad * 2)) & ~@as(usize, 127));
    return @min(qb, seq_pad);
}

/// Softmax partials per scores row.
const coop_chunks: usize = 32;

/// The tensor-core attention's scratch (`Dev.attnCoop`) for `n` positions of a
/// `c`-wide head. Rows are padded to the 128-row tile, so `ao` is too.
fn sizeCoopAttn(ctx: *Context, bufs: *Bufs, n: usize, c: usize) !void {
    const seq_pad = std.mem.alignForward(usize, n, 128);
    const qb = coopBand(n);
    inline for (.{ "aq", "ak", "av" }) |f| try ctx.ensureDeviceBuffer(&@field(bufs, f), n * c * 4);
    try ctx.ensureDeviceBuffer(&bufs.ao, seq_pad * c * 4);
    inline for (.{ "qh", "kh", "vh" }) |f| try ctx.ensureDeviceBuffer(&@field(bufs, f), seq_pad * c * 2);
    try ctx.ensureDeviceBuffer(&bufs.s, qb * seq_pad * 2);
    try ctx.ensureDeviceBuffer(&bufs.pt, qb * coop_chunks * 2 * 4);
    try ctx.ensureDeviceBuffer(&bufs.md, qb * 2 * 4);
}

/// `sd_vae_dev`'s and `wan_vae_dev`'s ops on this backend.
const Dev = struct {
    ctx: *Context,
    bufs: *Bufs,
    cats: *NormCats,
    /// GroupNorm's; the Wan decoder has none.
    groups: usize = 0,
    eps: f32 = 0,
    /// The mid-block attention on tensor cores with f16 scores (`attnCoop`).
    coop_attn: bool = false,

    pub fn conv(d: Dev, dst: *DeviceBuffer, src: *const DeviceBuffer, h: usize, w: usize, cv: Conv2d, mode: conv_gpu.SampleMode, act_div: f32) !void {
        const oh = if (mode == .upsample2x) 2 * h else h;
        const ow = if (mode == .upsample2x) 2 * w else w;
        // Pre-sized in `decode`: growing it here would be inside the recording batch.
        std.debug.assert(dst.size >= oh * ow * cv.co * dst.dt.bytes());
        try conv_gpu.conv2d(d.ctx, &d.bufs.patch, dst, src, h, w, cv, mode, null, null, act_div);
    }

    pub fn groupNorm(d: Dev, dst: *DeviceBuffer, src: *const DeviceBuffer, n: usize, ch: usize, nw: sd_vae.GroupNormW, silu: bool) !void {
        try sd_unet_gpu.groupNormInto(d.ctx, d.bufs.gstat, d.bufs.gmi, dst, src, n, ch, try d.cats.get(nw), d.groups, d.eps, silu);
    }

    pub fn channelNorm(d: Dev, dst: *DeviceBuffer, src: *const DeviceBuffer, n: usize, gamma: []const f32, silu: bool) !void {
        std.debug.assert(src.dt == dst.dt);
        const gbuf: DeviceBuffer = .{ .buf = try d.ctx.smallBuffer(std.mem.sliceAsBytes(gamma)), .mem = .null_handle, .size = 0 };
        try d.ctx.opElt(gpu_context.eltAt("vae_norm", src.dt), src.*, dst.*, gbuf, null, .{
            .u0 = @intCast(n),
            .u1 = @intCast(gamma.len),
            .u2 = @intFromBool(silu),
            .f0 = 1e-12,
        }, n, 1, 1);
    }

    pub fn add(d: Dev, a: DeviceBuffer, b: DeviceBuffer, n: usize) !void {
        std.debug.assert(a.dt == b.dt);
        try d.ctx.opElt(gpu_context.eltAt("add", a.dt), a, b, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
    }

    /// The scores plane is f32, not the f16 the tensor-core path would give: the
    /// Flux/Z-Image VAE's attention logits reach 9.95e6 on a 12x10 latent (SD1.5's reach
    /// 8.3), 152x past f16, and it rendered solid white with no error while everything
    /// upstream stayed finite. `Decoder.estimatePeakBytes` charges the plane at 4 bytes so
    /// the decode ladder tiles when a whole one will not fit. `attn_out` runs its OWN
    /// online softmax over the raw scores, so there is no softmax pass between it and
    /// `attn_scores`: one exponentiates twice, finite, plausible, and wrong by rel L2 0.26.
    pub fn attnCore(d: Dev, n: usize, ch: usize) !void {
        if (d.coop_attn) return d.attnCoop(n, ch);
        const b = d.bufs;
        std.debug.assert(ch % 8 == 0);
        // Per-head k-major (one head here), so the scores kernel loads contiguously.
        d.ctx.independent(2);
        try d.ctx.opElt(.gather_kmajor, b.aq, null, null, b.qh, .{
            .u0 = @intCast(n * ch),
            .u1 = 1,
            .u2 = @intCast(ch),
            .u3 = @intCast(n),
        }, n * ch, 1, 1);
        try d.ctx.opElt(.gather_kmajor, b.ak, null, null, b.kh, .{
            .u0 = @intCast(n * ch),
            .u1 = 1,
            .u2 = @intCast(ch),
            .u3 = @intCast(n),
        }, n * ch, 1, 1);

        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(ch)));
        // Query-BANDED, because the full plane is the largest thing in the decode.
        // The mid-block attends over every latent position, so at a 132x198 latent
        // (a 1056x1584 render) seq is 26,136 and an f32 `seq²` plane is 2.73 GB,
        // more than the activations it exists to serve. `scoresBand` caps it; the
        // arithmetic is unchanged because `attn_out` already runs an ONLINE softmax over
        // whole key rows, so a band of queries is exactly independent work.
        const qb = sd_vae.scoresBand(n);
        const jblk = std.math.divCeil(usize, n, 8) catch unreachable;
        var q0: usize = 0;
        while (q0 < n) : (q0 += qb) {
            const rows = @min(qb, n - q0);
            try d.ctx.opElt(.attn_scores, b.qh, b.kh, null, b.s, .{
                .u0 = @intCast(n),
                .u1 = 1,
                .u2 = 1,
                .u3 = @intCast(ch),
                .u4 = 0,
                .u5 = @intCast(q0),
                .u6 = @intCast(qb),
                .f0 = scale,
            }, jblk, std.math.divCeil(usize, rows, 8) catch unreachable, 1);
            try d.ctx.opElt(.attn_out, b.s, null, b.av, b.ao, .{
                .u0 = @intCast(n),
                .u1 = 1,
                .u2 = 1,
                .u3 = @intCast(ch),
                .u4 = 0,
                .u5 = @intCast(n),
                // +1 so that 0 keeps meaning "not banded", see the kernel.
                .u6 = @intCast(q0 + 1),
                .f0 = @bitCast(@as(u32, @intCast(qb * n))),
            }, ch / 8, std.math.divCeil(usize, rows, 8) catch unreachable, 1);
        }
    }

    /// `attnCore` on tensor cores, the scores plane f16 and query-banded: f16 Q with the
    /// softmax scale prefolded, k-major f16 K and f16 V (pad rows zero), then per band
    /// the scores GEMM, a two-pass softmax over the `n` real keys into `md`, and P@V
    /// into that band's rows of `ao`. Only the head width `opAttnScoresVae` is built for.
    fn attnCoop(d: Dev, n: usize, c: usize) !void {
        const b = d.bufs;
        const ctx = d.ctx;
        const seq_pad = std.mem.alignForward(usize, n, 128);
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c)));
        d.ctx.independent(3);
        try ctx.opElt(.f32_to_h16, b.aq, null, null, b.qh, .{ .u0 = @intCast(seq_pad * c / 2), .u1 = @intCast(n * c), .f0 = scale }, seq_pad * c / 2, 1, 1);
        try ctx.opElt(.gather_kmajor_h16, b.ak, null, null, b.kh, .{
            .u0 = @intCast(c * seq_pad / 2),
            .u1 = @intCast(c),
            .u2 = @intCast(seq_pad),
            .u3 = @intCast(n),
            .u4 = 1,
        }, c * seq_pad / 2, 1, 1);
        try ctx.opElt(.f32_to_h16, b.av, null, null, b.vh, .{ .u0 = @intCast(seq_pad * c / 2), .u1 = @intCast(n * c), .f0 = 1.0 }, seq_pad * c / 2, 1, 1);

        const qb = coopBand(n);
        var q0: usize = 0;
        while (q0 < seq_pad) : (q0 += qb) {
            const rows = @min(qb, seq_pad - q0);
            // Pad query rows need no softmax: their output rows are never read.
            const valid = @min(rows, n -| q0);
            const qv = try ctx.view(b.qh, q0 * c * 2, rows * c * 2);
            const ov = try ctx.view(b.ao, q0 * c * 4, rows * c * 4);
            try ctx.opAttnScoresVae(b.s, qv, b.kh, .{
                .u0 = @intCast(c),
                .u1 = @intCast(seq_pad),
                .u2 = 0,
                .u3 = 1,
                .u4 = @intCast(c * seq_pad),
                .u5 = @intCast(rows * seq_pad),
            }, seq_pad / 128, rows / 128, 1);
            if (valid > 0) {
                try ctx.opElt(.softmax_partial, b.s, null, null, b.pt, .{
                    .u0 = @intCast(valid * coop_chunks),
                    .u1 = @intCast(coop_chunks),
                    .u2 = @intCast(n),
                    .u3 = @intCast(seq_pad),
                    .u5 = 0,
                }, valid * coop_chunks, 1, 1);
                try ctx.opElt(.softmax_combine, b.pt, null, null, b.md, .{
                    .u0 = @intCast(valid),
                    .u1 = @intCast(coop_chunks),
                    .u2 = @intCast(n),
                    .u3 = @intCast(seq_pad),
                }, valid, 1, 1);
            }
            // The head as three 128-column groups sharing one scores plane.
            try ctx.opAttnOut(b.s, b.vh, ov, b.md, .{
                .u0 = @intCast(seq_pad),
                .u1 = 0,
                .u2 = 0,
                .u3 = 1,
                .u4 = @intCast(c),
                .u5 = @intCast(c),
                .f0 = @bitCast(@as(u32, @intCast(n))),
                .f1 = @bitCast(@as(u32, 0)),
            }, rows / 128, c / 128);
        }
    }

    /// The attention scratch, above all the O(seq²) scores plane, sits at the START
    /// (the mid-block runs at latent resolution) and would otherwise stay resident
    /// through the whole 8x upsampling. `endBatch` waits, so no recorded dispatch
    /// still references these buffers when they are destroyed.
    pub fn midDone(d: Dev) !void {
        try d.ctx.endBatch();
        inline for (.{ "aq", "ak", "av", "qh", "kh", "vh", "s", "pt", "md", "ao" }) |f| d.ctx.tensorDestroy(&@field(d.bufs, f));
        try d.ctx.beginBatch();
    }
};


// --- tests ------------------------------------------------------------------

test "gpu sd vae decode matches the cpu decoder, at 4 AND 16 latent channels" {
    // The 16-channel arm is the one this test was missing, and its absence let
    // a blank white Z-Image render through: the same decoder body serves SD's
    // 4-channel latent and the Flux/Z-Image 16-channel one, which additionally has
    // no `post_quant_conv`, so the device path has to stage the latent straight
    // into the `conv_in` input instead of through a convolution that does not exist.
    // A 4-channel-only test cannot see either difference.
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const test_gate = @import("../test_gate.zig");
    try test_gate.requireIntegration();
    std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    const ctx = Context.init(gpa, std.testing.io) catch return error.SkipZigTest;
    defer ctx.deinit();

    const Case = struct { path: []const u8, prefix: []const u8, cfg: sd_vae.Config };
    const cases = [_]Case{
        .{
            .path = build_options.model_root ++ "/checkpoints/sd1.5/perfectdeliberate_v20.safetensors",
            .prefix = "first_stage_model.",
            .cfg = sd_vae.sd15,
        },
        .{
            .path = build_options.model_root ++ "/vae/ae.safetensors",
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
        // Vulkan reads repacked f32 conv weights, as the pipeline arranges before a decode.
        try dec.repackConvs(gpa);

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

        const got = try decode(&dec, ctx, gpa, z, lat_h, lat_w, null, null);
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
        errdefer std.debug.print("sd vae gpu vs cpu ({d} ch) rel L2 {d:.6}\n", .{ c.cfg.z_channels, rel });
        // f16 tensor-core GEMMs through 12 resnets and an 8x upsample.
        try std.testing.expect(rel < 5e-3);
        ran += 1;
    }
    if (ran == 0) return error.SkipZigTest;
}

// Full-decoder parity against the same ComfyUI fixture the CPU test uses;
// needs the checkpoint, the fixture, and the GPU marker.
test "gpu wan vae decode matches the comfyui reference" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    const vae_path = build_options.model_root ++ "/vae/krea2RealVae_v10.safetensors";
    std.Io.Dir.cwd().access(io, vae_path, .{}) catch return error.SkipZigTest;
    std.Io.Dir.cwd().access(io, "testdata/vae_z_8x8.bin", .{}) catch return error.SkipZigTest;
    var ctx = Context.init(gpa, std.testing.io) catch return error.SkipZigTest;
    defer ctx.deinit();

    const z = try gpa.alloc(f32, 16 * 8 * 8);
    defer gpa.free(z);
    const expected = try gpa.alloc(f32, 3 * 64 * 64);
    defer gpa.free(expected);
    inline for (.{ .{ "testdata/vae_z_8x8.bin", z }, .{ "testdata/vae_rgb_64.bin", expected } }) |pair| {
        const file = try std.Io.Dir.cwd().openFile(io, pair[0], .{ .mode = .read_only });
        defer file.close(io);
        const bytes = std.mem.sliceAsBytes(pair[1]);
        if (try file.readPositionalAll(io, bytes, 0) != bytes.len) return error.ShortRead;
    }

    var st = try @import("tp_core").safetensors.SafeTensors.open(gpa, io, vae_path);
    defer st.deinit();
    var dec = try wan_vae.Decoder.load(gpa, .{ .safetensors = &st });
    defer dec.deinit();
    try dec.repackConvs(gpa);

    const out = try decodeWan(&dec, ctx, gpa, z, 8, 8, null, .f32);
    defer gpa.free(out);

    var max_err: f32 = 0;
    var sum_err: f64 = 0;
    for (expected, out) |e, a| {
        const err = @abs(e - a);
        max_err = @max(max_err, err);
        sum_err += err;
    }
    const mean_err = sum_err / @as(f64, @floatFromInt(out.len));
    errdefer std.debug.print("vae gpu parity: max_err={d:.6} mean_err={d:.6}\n", .{ max_err, mean_err });
    try std.testing.expect(max_err < 5e-3);
    try std.testing.expect(mean_err < 5e-4);
}
