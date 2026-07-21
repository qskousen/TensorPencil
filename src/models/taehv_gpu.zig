//! Vulkan (generic-GPU) decode for the taew2_1 (TAEHV) approx VAE — the GPU
//! path for the live per-step preview when the diffuser runs on the Vulkan
//! backend. Mirrors `taehv_cuda.decode` on the `Context` abstraction, reusing
//! the same banded im2col + GEMM conv as `vae_gpu` (tensor cores via
//! `opMatmulCoopF16W` for co >= 96, the f32 register-tile GEMM below that).
//!
//! The input Clamp (tanh(x/3)*3) is applied on the host before upload (it's on
//! the small latent). Each stage's spatial Upsample is fused into that stage's
//! 3x3 conv (`up=true`) — valid because it commutes with the per-pixel TGrow
//! 1x1 conv that sits between them. Weights are read straight from the loaded
//! `taehv.Decoder`; the packed [co][kh][kw][ci] layout is exactly the
//! [rows][cols] the GEMM weight cache expects, so they upload/transpose lazily
//! on first use and stay cached across preview steps.
const std = @import("std");
const wan_vae = @import("wan_vae.zig");
const taehv = @import("taehv.zig");
const gpu_context = @import("tp_gpu").context;

const Context = gpu_context.Context;
const DeviceBuffer = gpu_context.DeviceBuffer;

/// Cap on the im2col patch band (bytes); bands iterate over output rows.
const patch_band_bytes: usize = 256 << 20;

const none: DeviceBuffer = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 };

const Bufs = struct {
    x: DeviceBuffer = none,
    t: DeviceBuffer = none,
    u: DeviceBuffer = none,
    patch: DeviceBuffer = none,

    fn deinit(self: *Bufs, ctx: *Context) void {
        inline for (@typeInfo(Bufs).@"struct".fields) |f| ctx.tensorDestroy(&@field(self, f.name));
    }
};

/// Decode planar [16][zh][zw] sampler latent to RGB8 [8*zh][8*zw][3].
pub fn decode(dec: *const taehv.Decoder, ctx: *Context, gpa: std.mem.Allocator, z: []const f32, zh: usize, zw: usize) ![]u8 {
    std.debug.assert(z.len == taehv.latent_channels * zh * zw);
    var bufs: Bufs = .{};
    defer bufs.deinit(ctx);

    var h = zh;
    var w = zw;
    const n0 = zh * zw;
    {
        const rows = try wan_vae.planarToRows(gpa, z, taehv.latent_channels, n0);
        defer gpa.free(rows);
        for (rows) |*v| v.* = std.math.tanh(v.* / 3.0) * 3.0; // Clamp, on host
        try ctx.ensureDeviceBuffer(&bufs.x, n0 * taehv.latent_channels * 4);
        try ctx.tensorUpload(bufs.x, std.mem.sliceAsBytes(rows));
    }

    try ctx.beginBatch();
    var batched = true;
    errdefer if (batched) ctx.endBatch() catch {};

    try conv(ctx, &bufs, &bufs.x, &bufs.t, h, w, dec.conv_in, false); // 16 -> 256
    std.mem.swap(DeviceBuffer, &bufs.x, &bufs.t);
    try relu(ctx, bufs.x, h * w * dec.conv_in.co);

    for (dec.stages) |stage| {
        for (stage.mb) |mb| try memBlock(ctx, &bufs, h, w, mb);
        try conv(ctx, &bufs, &bufs.x, &bufs.t, h, w, stage.tgrow, false); // TGrow 1x1 (first-frame slice)
        std.mem.swap(DeviceBuffer, &bufs.x, &bufs.t);
        try conv(ctx, &bufs, &bufs.x, &bufs.t, h, w, stage.sc, true); // 3x3 with fused 2x upsample
        std.mem.swap(DeviceBuffer, &bufs.x, &bufs.t);
        h *= 2;
        w *= 2;
    }

    try relu(ctx, bufs.x, h * w * 64); // ReLU before head
    try conv(ctx, &bufs, &bufs.x, &bufs.u, h, w, dec.head_conv, false); // 64 -> 3

    batched = false;
    try ctx.endBatch();

    const n = h * w;
    const rgb_rows = try gpa.alloc(f32, n * 3);
    defer gpa.free(rgb_rows);
    try ctx.tensorDownload(bufs.u, std.mem.sliceAsBytes(rgb_rows));
    const rgb = try gpa.alloc(u8, n * 3);
    for (rgb, rgb_rows) |*o, v| o.* = @intFromFloat(std.math.clamp(v, 0.0, 1.0) * 255.0);
    return rgb;
}

/// MemBlock (past=0): out = ReLU(conv4(ReLU(conv2(ReLU(conv0(x))))) + x), in
/// place on bufs.x. The final `+ skip` and its ReLU are one fused `add_relu`.
fn memBlock(ctx: *Context, bufs: *Bufs, h: usize, w: usize, mb: taehv.MemBlock) !void {
    const n = h * w * mb.n;
    try conv(ctx, bufs, &bufs.x, &bufs.t, h, w, mb.conv0, false);
    try relu(ctx, bufs.t, n);
    try conv(ctx, bufs, &bufs.t, &bufs.u, h, w, mb.conv2, false);
    try relu(ctx, bufs.u, n);
    try conv(ctx, bufs, &bufs.u, &bufs.t, h, w, mb.conv4, false);
    try ctx.opElt(.add_relu, bufs.t, bufs.x, null, null, .{ .u0 = @intCast(n) }, n, 1, 1); // + skip, fused ReLU
    std.mem.swap(DeviceBuffer, &bufs.x, &bufs.t);
}

fn relu(ctx: *Context, x: DeviceBuffer, n: usize) !void {
    try ctx.opElt(.relu, x, null, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
}

/// 3x3 (optionally with fused nearest-2x upsample) or 1x1 conv. Mirrors
/// vae_gpu.conv: co >= 96 convs use the f16 tensor-core GEMM (when the coop
/// pipeline exists), smaller ones the f32 GEMM.
fn conv(ctx: *Context, bufs: *Bufs, src: *const DeviceBuffer, dst: *DeviceBuffer, h: usize, w: usize, cv: wan_vae.Conv2d, up: bool) !void {
    const wbytes = std.mem.sliceAsBytes(cv.w);
    const coop = ctx.pipe_coop_f16w != .null_handle and cv.co >= 96;
    if (cv.k == 1) {
        std.debug.assert(!up);
        const n = h * w;
        try ctx.ensureDeviceBuffer(dst, n * cv.co * 4);
        if (coop) return ctx.opMatmulCoopF16W(dst.*, 0, src.*, n, cv.w, cv.co, cv.ci, cv.b);
        return ctx.opMatmul(dst.*, 0, src.*, 0, n, wbytes, false, cv.co, cv.ci, 1.0, cv.b);
    }
    std.debug.assert(cv.k == 3);

    const oh = if (up) 2 * h else h;
    const ow = if (up) 2 * w else w;
    const n_out = oh * ow;
    const patch_len = 9 * cv.ci;
    try ctx.ensureDeviceBuffer(dst, n_out * cv.co * 4);

    // Band positions: multiple of 4 keeps the GEMM's y byte offset 16-aligned
    // for any co.
    const band = @max(4, @min(n_out, patch_band_bytes / (patch_len * 4)) & ~@as(usize, 3));
    try ctx.ensureDeviceBuffer(&bufs.patch, band * patch_len * 4);

    var p0: usize = 0;
    while (p0 < n_out) : (p0 += band) {
        const bn = @min(band, n_out - p0);
        try ctx.opElt(.im2col, src.*, null, null, bufs.patch, .{
            .u0 = @intCast(bn * patch_len),
            .u1 = @intCast(patch_len),
            .u2 = @intCast(cv.ci),
            .u3 = @intCast(w),
            .u4 = @intCast(h),
            .u5 = @intCast(p0),
            .f0 = if (up) 1.0 else 0.0,
        }, bn * patch_len, 1, 1);
        if (coop) {
            try ctx.opMatmulCoopF16W(dst.*, p0 * cv.co, bufs.patch, bn, cv.w, cv.co, patch_len, cv.b);
        } else {
            try ctx.opMatmul(dst.*, p0 * cv.co * 4, bufs.patch, 0, bn, wbytes, false, cv.co, patch_len, 1.0, cv.b);
        }
    }
}

// --- tests -----------------------------------------------------------------

// Full-decoder parity against the CPU taehv reference on random weights and a
// random latent. Gated like the other GPU tests (needs the marker + device).
test "gpu taehv decode matches cpu reference" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    var ctx = Context.init(gpa) catch return error.SkipZigTest;
    defer ctx.deinit();

    var dec = try syntheticDecoder(gpa);
    defer dec.deinit();

    const zh = 6;
    const zw = 5;
    var prng = std.Random.DefaultPrng.init(7);
    const rand = prng.random();
    const z = try gpa.alloc(f32, taehv.latent_channels * zh * zw);
    defer gpa.free(z);
    for (z) |*v| v.* = rand.floatNorm(f32);

    const want = try dec.decode(io, gpa, z, zh, zw);
    defer gpa.free(want);
    const got = try decode(&dec, ctx, gpa, z, zh, zw);
    defer gpa.free(got);

    try std.testing.expectEqual(want.len, got.len);
    var max_diff: u32 = 0;
    for (want, got) |e, a| max_diff = @max(max_diff, @abs(@as(i32, e) - @as(i32, a)));
    std.debug.print("taehv gpu parity: len={d} max_u8_diff={d}\n", .{ got.len, max_diff });
    // f16 tensor-core convs vs f32 CPU: allow a couple of 8-bit levels.
    try std.testing.expect(max_diff <= 2);
}

test "gpu single coop conv float parity" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    var ctx = Context.init(gpa) catch return error.SkipZigTest;
    defer ctx.deinit();
    if (ctx.pipe_coop_f16w == .null_handle) return error.SkipZigTest;

    const h = 6;
    const w = 5;
    const ci = 128;
    const co = 256;
    var prng = std.Random.DefaultPrng.init(3);
    const rand = prng.random();
    const wd = try gpa.alloc(f32, co * 9 * ci);
    defer gpa.free(wd);
    for (wd) |*x| x.* = rand.floatNorm(f32) * 0.05;
    const bd = try gpa.alloc(f32, co);
    defer gpa.free(bd);
    for (bd) |*x| x.* = rand.floatNorm(f32) * 0.05;
    const in = try gpa.alloc(f32, h * w * ci);
    defer gpa.free(in);
    for (in) |*x| x.* = rand.floatNorm(f32);
    const cv: wan_vae.Conv2d = .{ .w = wd, .b = bd, .co = co, .ci = ci, .k = 3 };

    var bufs: Bufs = .{};
    defer bufs.deinit(ctx);
    try ctx.ensureDeviceBuffer(&bufs.x, in.len * 4);
    try ctx.tensorUpload(bufs.x, std.mem.sliceAsBytes(in));
    try ctx.beginBatch();
    try conv(ctx, &bufs, &bufs.x, &bufs.t, h, w, cv, false);
    try ctx.endBatch();
    const got = try gpa.alloc(f32, h * w * co);
    defer gpa.free(got);
    try ctx.tensorDownload(bufs.t, std.mem.sliceAsBytes(got));

    var max_err: f32 = 0;
    for (0..h) |y| for (0..w) |xx| for (0..co) |o| {
        var acc: f64 = bd[o];
        for (0..3) |ky| for (0..3) |kx| {
            const sy = @as(isize, @intCast(y + ky)) - 1;
            const sx = @as(isize, @intCast(xx + kx)) - 1;
            if (sy < 0 or sy >= h or sx < 0 or sx >= w) continue;
            for (0..ci) |cc| acc += @as(f64, wd[((o * 3 + ky) * 3 + kx) * ci + cc]) *
                in[(@as(usize, @intCast(sy)) * w + @as(usize, @intCast(sx))) * ci + cc];
        };
        max_err = @max(max_err, @abs(@as(f32, @floatCast(acc)) - got[(y * w + xx) * co + o]));
    };
    std.debug.print("single coop conv max_err={d:.5}\n", .{max_err});
    try std.testing.expect(max_err < 5e-2);
}

/// Build a TAEHV Decoder with random weights matching the real channel geometry
/// (no checkpoint needed), so the GPU/CPU paths can be compared directly.
fn syntheticDecoder(gpa: std.mem.Allocator) !taehv.Decoder {
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(99);
    const rand = prng.random();

    const mkConv = struct {
        fn f(al: std.mem.Allocator, r: std.Random, ci: usize, co: usize, k: usize) !wan_vae.Conv2d {
            const wd = try al.alloc(f32, co * k * k * ci);
            // He-style init (std = 1/sqrt(fan_in)) so per-layer gain is ~1 and
            // activations stay O(1) through the deep net — the well-conditioned
            // regime real trained weights live in, where f16 error stays bounded.
            // (Larger weights make the net exponentially amplify f16 rounding,
            // which is a property of the weights, not the GEMM.)
            const std_dev = 1.0 / @sqrt(@as(f32, @floatFromInt(k * k * ci)));
            for (wd) |*x| x.* = r.floatNorm(f32) * std_dev;
            const bd = try al.alloc(f32, co);
            for (bd) |*x| x.* = r.floatNorm(f32) * 0.02;
            return .{ .w = wd, .b = bd, .co = co, .ci = ci, .k = k };
        }
    }.f;

    const conv_in = try mkConv(a, rand, 16, 256, 3);
    const specs = [3]struct { n: usize, sc_co: usize, tg_co: usize }{
        .{ .n = 256, .sc_co = 128, .tg_co = 256 },
        .{ .n = 128, .sc_co = 64, .tg_co = 256 },
        .{ .n = 64, .sc_co = 64, .tg_co = 128 },
    };
    var stages: [3]taehv.Stage = undefined;
    for (&stages, specs) |*stage, s| {
        var mb: [3]taehv.MemBlock = undefined;
        for (&mb) |*b| b.* = .{
            .conv0 = try mkConv(a, rand, s.n, s.n, 3),
            .conv2 = try mkConv(a, rand, s.n, s.n, 3),
            .conv4 = try mkConv(a, rand, s.n, s.n, 3),
            .n = s.n,
        };
        var tg = try mkConv(a, rand, s.n, s.tg_co, 1);
        tg.w = tg.w[0 .. s.n * s.n]; // first-frame slice (first n output rows)
        tg.co = s.n;
        tg.b = tg.b[0..s.n];
        stage.* = .{ .mb = mb, .tgrow = tg, .sc = try mkConv(a, rand, s.n, s.sc_co, 3), .n = s.n };
    }
    return .{
        .arena = arena,
        .conv_in = conv_in,
        .stages = stages,
        .head_conv = try mkConv(a, rand, 64, 3, 3),
    };
}
