//! GPU-resident VAE decode on the CUDA backends: the AutoencoderKL (`decode`, over
//! `sd_vae_dev`) and the Wan 2.1 decoder (`decodeWan`, over `wan_vae_dev`), one adapter.
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
const conv_cuda = @import("conv_cuda.zig");
const sd_unet_cuda = @import("sd_unet_cuda.zig");
const dev = @import("sd_vae_dev.zig");
const wan_vae = @import("wan_vae.zig");
const wan_vae_dev = @import("wan_vae_dev.zig");
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
    /// Separate from `x`/`t` because those may be 16-bit.
    stage: Buf = .{},
    patch: Buf = .{},
    gstat: Buf = .{},
    gmi: Buf = .{},
    /// The Wan 2.2 stage input its shortcut adds back (`wan_vae.Layer.save`).
    sv: Buf = .{},
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
    /// Activation storage; null takes `Config.act_f16`.
    act_override: ?cuda.backend.Dt,
) ![]f32 {
    const cfg = dec.cfg;
    const act: cuda.backend.Dt = act_override orelse if (cfg.act_f16) .f16 else .f32;
    std.debug.assert(z.len == lat_h * lat_w * cfg.z_channels);
    var bufs: Bufs = .{};
    defer bufs.deinit(be);

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var norms: NormBufs = .{ .alloc = arena.allocator(), .be = be };
    defer norms.deinit();

    // Sizing mirrors the loader's own width bookkeeping: the first resnet of a
    // level reads the PREVIOUS level's (wider) input at the NEW (doubled)
    // resolution, so sizing off `block_out_channels` alone under-allocates by 2x.
    // TWO widths: `u` only ever receives a convolution's OUTPUT, so it peaks one
    // level narrower than `x`/`t` (`sd_vae.activationElems`).
    const widths = sd_vae.activationElems(cfg, lat_h, lat_w);
    const max_elems: usize = @intCast(widths.wide);

    // 16-bit storage halves the two widest buffers, which is where a whole-image
    // decode's gigabytes are: at 1056x1584 each holds 256 channels at FULL image
    // resolution (428M elements). The latent coming IN and the RGB going OUT are
    // f32 either way, so they get their own small staging buffer rather than
    // borrowing one of these.
    const asz = act.bytes();
    inline for (.{ "x", "t", "u" }) |f| @field(bufs, f).dt = act;
    try be.ensureDeviceBuffer(&bufs.x, max_elems * asz);
    try be.ensureDeviceBuffer(&bufs.t, max_elems * asz);
    try be.ensureDeviceBuffer(&bufs.stage, @max(z.len, lat_h * lat_w * 64 * 3) * 4);
    try be.tensorUpload(bufs.stage, std.mem.sliceAsBytes(z));
    try be.ensureDeviceBuffer(&bufs.u, @as(usize, @intCast(widths.out)) * asz);
    try be.ensureDeviceBuffer(&bufs.gstat, cfg.norm_groups * gn_chunks * 3 * 4);
    try be.ensureDeviceBuffer(&bufs.gmi, cfg.norm_groups * 2 * 4);
    inline for (.{ "aq", "ak", "av", "ao" }) |f| {
        try be.ensureDeviceBuffer(&@field(bufs, f), lat_h * lat_w * cfg.innermost() * 4);
    }

    try be.beginBatch();
    errdefer if (be.batching()) be.abortBatch();
    const ext = try dev.body(Dev{ .be = be, .bufs = &bufs, .norms = &norms, .groups = cfg.norm_groups, .eps = cfg.norm_eps }, dec, lat_h, lat_w, cancel);
    try be.endBatch();

    const rgb = try gpa.alloc(f32, ext[0] * ext[1] * 3);
    errdefer gpa.free(rgb);
    try be.tensorDownload(bufs.stage, std.mem.sliceAsBytes(rgb));
    return rgb;
}

/// Decode a Wan latent (planar `[z][zh][zw]`, already denormalized) to planar
/// `[oc][s*zh][s*zw]` pixels in [-1, 1], the chain at `act`. Caller frees.
pub fn decodeWan(dec: *const wan_vae.Decoder, be: *Backend, gpa: std.mem.Allocator, z: []const f32, zh: usize, zw: usize, cancel: ?*std.atomic.Value(bool), act: cuda.backend.Dt) ![]f32 {
    const zc = dec.latentChannels();
    const oc = dec.outChannels();
    const s = dec.scale();
    std.debug.assert(z.len == zc * zh * zw);
    var bufs: Bufs = .{};
    defer bufs.deinit(be);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var norms: NormBufs = .{ .alloc = arena.allocator(), .be = be };
    defer norms.deinit();

    const n0 = zh * zw;
    const widths = dec.activationElems(zh, zw);
    const asz = act.bytes();
    inline for (.{ "x", "t", "u", "sv" }) |f| @field(bufs, f).dt = act;
    try be.ensureDeviceBuffer(&bufs.x, @as(usize, @intCast(widths.wide)) * asz);
    try be.ensureDeviceBuffer(&bufs.t, @as(usize, @intCast(widths.wide)) * asz);
    try be.ensureDeviceBuffer(&bufs.u, @as(usize, @intCast(widths.out)) * asz);
    if (dec.savedElems(zh, zw) > 0) try be.ensureDeviceBuffer(&bufs.sv, dec.savedElems(zh, zw) * asz);
    try be.ensureDeviceBuffer(&bufs.stage, @max(z.len, n0 * s * s * oc) * 4);
    {
        const rows = try wan_vae.planarToRows(gpa, z, zc, n0);
        defer gpa.free(rows);
        try be.tensorUpload(bufs.stage, std.mem.sliceAsBytes(rows));
    }
    const c = dec.mid_attn.proj.co;
    inline for (.{ "aq", "ak", "av", "ao" }) |f| try be.ensureDeviceBuffer(&@field(bufs, f), n0 * c * 4);

    try be.beginBatch();
    errdefer if (be.batching()) be.abortBatch();
    const ext = try wan_vae_dev.body(Dev{ .be = be, .bufs = &bufs, .norms = &norms }, dec, zh, zw, cancel);
    try be.endBatch();

    const n = ext[0] * ext[1];
    const rgb = try gpa.alloc(f32, n * oc);
    defer gpa.free(rgb);
    try be.tensorDownload(bufs.stage.slice(0, rgb.len * 4), std.mem.sliceAsBytes(rgb));
    return wan_vae.rowsToPlanar(gpa, rgb, oc, n);
}

/// A/B switch for the mid-block attention (see `Dev.attnCore`). `zimage-cuda-test` runs both.
pub var force_naive_attn: bool = false;

/// `sd_vae_dev`'s and `wan_vae_dev`'s ops on this backend.
const Dev = struct {
    be: *Backend,
    bufs: *Bufs,
    norms: *NormBufs,
    /// GroupNorm's; the Wan decoder has none.
    groups: usize = 0,
    eps: f32 = 0,

    pub fn conv(d: Dev, dst: *Buf, src: *const Buf, h: usize, w: usize, cv: Conv2d, mode: conv_cuda.SampleMode, act_div: f32) !void {
        try conv_cuda.conv2d(d.be, &d.bufs.patch, dst, src, h, w, cv, mode, null, act_div);
    }

    pub fn groupNorm(d: Dev, dst: *Buf, src: *const Buf, n: usize, ch: usize, nw: sd_vae.GroupNormW, silu: bool) !void {
        const cat = try d.norms.get(nw);
        try d.be.opGroupNorm(src.*, dst.*, cat, d.bufs.gstat, d.bufs.gmi, n, ch, d.groups, gn_chunks, d.eps, silu);
    }

    pub fn channelNorm(d: Dev, dst: *Buf, src: *const Buf, n: usize, gamma: []const f32, silu: bool) !void {
        std.debug.assert(src.dt == dst.dt);
        const gbuf: Buf = .{ .buf = try d.be.smallBuffer(std.mem.sliceAsBytes(gamma)), .mem = .null_handle, .size = 0 };
        try d.be.opVaeNorm(src.*, dst.*, gbuf, n, gamma.len, silu);
    }

    pub fn add(d: Dev, a: Buf, b: Buf, n: usize) !void {
        try d.be.opAdd(a, b, n);
    }

    pub fn copy(d: Dev, dst: Buf, src: Buf, n: usize) !void {
        std.debug.assert(dst.dt == src.dt);
        try d.be.tensorCopy(dst, 0, src, 0, n * src.dt.bytes());
    }

    pub fn dupUpAdd(d: Dev, x: Buf, sv: Buf, h: usize, w: usize, du: wan_vae.DupUp) !void {
        try d.be.dupUpAdd(x, sv, h, w, du.ci, du.co, du.ft);
    }

    /// `opAttnTC` on both arms, given the f32 scores band. A tensor-core path that
    /// materializes the scores plane in f16 cannot serve this VAE: the Flux/Z-Image VAE's
    /// attention LOGITS reach 9.95e6 on a 12x10 latent where SD1.5's reach 8.3, and it
    /// rendered solid white with no error while everything upstream stayed finite.
    /// `be.attn` keeps an online softmax in f32 and materializes no scores plane.
    pub fn attnCore(d: Dev, n: usize, ch: usize) !void {
        const b = d.bufs;
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(ch)));
        if (force_naive_attn) return d.be.attn(b.aq, b.ak, b.av, b.ao, n, n, 1, 1, ch, scale, false);
        try d.be.opAttnTC(b.aq, b.ak, b.av, b.ao, n, 1, 1, ch, scale);
    }

    /// The O(seq²) attention scratch sits at the START (the mid-block runs at latent
    /// resolution) and would otherwise stay resident through the whole 8x upsampling.
    pub fn midDone(d: Dev) !void {
        try d.be.endBatch();
        d.be.freeAttnScratch();
        inline for (.{ "aq", "ak", "av", "ao" }) |f| d.be.tensorDestroy(&@field(d.bufs, f));
        try d.be.beginBatch();
    }
};
