//! Vulkan decode for the taew2_1 (TAEHV) approx VAE: the adapter for `taehv_dev.zig`.
//! The conv is a banded im2col + GEMM (tensor cores via `opMatmulCoopF16W` for
//! co >= 96, the f32 register-tile GEMM below that). Weights are read straight from the loaded
//! `taehv.Decoder`; the packed [co][kh][kw][ci] layout is exactly the
//! [rows][cols] the GEMM weight cache expects, so they upload/transpose lazily
//! on first use and stay cached across preview steps.
const std = @import("std");
const wan_vae = @import("wan_vae.zig");
const conv_gpu = @import("conv_gpu.zig");
const taehv = @import("taehv.zig");
const taehv_dev = @import("taehv_dev.zig");
const sd_unet_gpu = @import("sd_unet_gpu.zig");
const gpu_context = @import("tp_gpu").context;

const Context = gpu_context.Context;
const DeviceBuffer = gpu_context.DeviceBuffer;

const none: DeviceBuffer = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 };

const Bufs = struct {
    x: DeviceBuffer = none,
    t: DeviceBuffer = none,
    u: DeviceBuffer = none,
    patch: DeviceBuffer = none,

    fn deinit(self: *Bufs, ctx: *Context) void {
        inline for (@typeInfo(Bufs).@"struct".field_names) |name| ctx.tensorDestroy(&@field(self, name));
    }
};

/// `taehv_dev.decode` on this backend. Caller frees.
pub fn decode(dec: *const taehv.Decoder, ctx: *Context, gpa: std.mem.Allocator, z: []const f32, zh: usize, zw: usize) ![]u8 {
    var bufs: Bufs = .{};
    defer bufs.deinit(ctx);
    return taehv_dev.decode(Dev{ .ctx = ctx, .bufs = &bufs }, dec, gpa, z, zh, zw);
}

/// The `taehv_dev` adapter.
const Dev = struct {
    ctx: *Context,
    bufs: *Bufs,

    pub fn upload(d: Dev, b: *DeviceBuffer, data: []const f32) !void {
        try d.ctx.ensureDeviceBuffer(b, data.len * 4);
        try d.ctx.tensorUpload(b.*, std.mem.sliceAsBytes(data));
    }

    pub fn download(d: Dev, b: DeviceBuffer, dst: []f32) !void {
        try d.ctx.tensorDownload(b, std.mem.sliceAsBytes(dst));
    }

    pub fn beginBatch(d: Dev) !void {
        try d.ctx.beginBatch();
    }

    pub fn endBatch(d: Dev) !void {
        try d.ctx.endBatch();
    }

    pub fn abortBatch(d: Dev) void {
        if (d.ctx.batching) d.ctx.endBatch() catch {};
    }

    pub fn conv(d: Dev, src: *const DeviceBuffer, dst: *DeviceBuffer, h: usize, w: usize, cv: wan_vae.Conv2d, up: bool) !void {
        try convAt(d.ctx, d.bufs, src, dst, h, w, cv, up);
    }

    pub fn relu(d: Dev, x: DeviceBuffer, n: usize) !void {
        try d.ctx.opElt(.relu, x, null, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
    }

    pub fn addRelu(d: Dev, x: DeviceBuffer, y: DeviceBuffer, n: usize) !void {
        try d.ctx.opElt(.add_relu, x, y, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
    }
};

/// 3x3 (optionally with a fused nearest-2x upsample) or 1x1 conv, through the shared
/// conv (`conv_gpu.conv2d`).
fn convAt(ctx: *Context, bufs: *Bufs, src: *const DeviceBuffer, dst: *DeviceBuffer, h: usize, w: usize, cv: wan_vae.Conv2d, up: bool) !void {
    const n_out = if (up) 4 * h * w else h * w;
    try ctx.ensureDeviceBuffer(dst, n_out * cv.co * 4);
    try conv_gpu.conv2d(ctx, &bufs.patch, dst, src, h, w, cv.asOps(), if (up) .upsample2x else .stride1, null, null, 1.0);
}

// --- tests -----------------------------------------------------------------

// Full-decoder parity against the CPU taehv reference on random weights and a
// random latent. Gated like the other GPU tests (needs the marker + device).
test "gpu taehv decode matches cpu reference" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    var ctx = Context.init(gpa, std.testing.io) catch return error.SkipZigTest;
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
    errdefer std.debug.print("taehv gpu parity: len={d} max_u8_diff={d}\n", .{ got.len, max_diff });
    // f16 tensor-core convs vs f32 CPU: allow a couple of 8-bit levels.
    try std.testing.expect(max_diff <= 2);
}

test "gpu single coop conv float parity" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    var ctx = Context.init(gpa, std.testing.io) catch return error.SkipZigTest;
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
    const cv: wan_vae.Conv2d = .{ .w = .{ .krsc = wd }, .b = bd, .co = co, .ci = ci, .k = 3 };

    var bufs: Bufs = .{};
    defer bufs.deinit(ctx);
    try ctx.ensureDeviceBuffer(&bufs.x, in.len * 4);
    try ctx.tensorUpload(bufs.x, std.mem.sliceAsBytes(in));
    try ctx.beginBatch();
    try convAt(ctx, &bufs, &bufs.x, &bufs.t, h, w, cv, false);
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
    errdefer std.debug.print("single coop conv max_err={d:.5}\n", .{max_err});
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
            // activations stay O(1) through the deep net, the well-conditioned
            // regime real trained weights live in, where f16 error stays bounded.
            // (Larger weights make the net exponentially amplify f16 rounding,
            // which is a property of the weights, not the GEMM.)
            const std_dev = 1.0 / @sqrt(@as(f32, @floatFromInt(k * k * ci)));
            for (wd) |*x| x.* = r.floatNorm(f32) * std_dev;
            const bd = try al.alloc(f32, co);
            for (bd) |*x| x.* = r.floatNorm(f32) * 0.02;
            return .{ .w = .{ .krsc = wd }, .b = bd, .co = co, .ci = ci, .k = k };
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
        tg.w = .{ .krsc = tg.krsc()[0 .. s.n * s.n] }; // first-frame slice (first n output rows)
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
