//! GPU-resident Wan 2.1 VAE decode.
//!
//! Mirrors `wan_vae.Decoder.decode` with the heavy work on the device:
//! 3x3 convs run as banded im2col (kernels/eltwise.zig `im2col`, which also
//! fuses the nearest-exact 2x upsample by halving source coordinates — the
//! upsampled tensor is never materialized) followed by a GEMM — tensor
//! cores via `opMatmulCoopF16W` for co >= 96, the f32 register-tile GEMM
//! (`opMatmul`, bias included) below that; 1x1 convs are direct GEMMs with
//! the same routing; the per-position
//! channel norms (+ their trailing silus) are the `vae_norm` kernel. Only
//! the mid-block single-head attention core runs on the CPU (one qkv
//! download + one attn-out upload); everything else stays device-resident
//! from the latent upload to the RGB download.
//!
//! Weights are read straight from a loaded `wan_vae.Decoder` — the packed
//! [co][kh][kw][ci] layout is exactly the [rows][cols] the GEMM's weight
//! cache expects, so buffers upload/transpose lazily on first use and stay
//! cached for the run.

const std = @import("std");
const wan_vae = @import("wan_vae.zig");
const ops = @import("../ops.zig");
const gpu_context = @import("../gpu/context.zig");

const Context = gpu_context.Context;
const DeviceBuffer = gpu_context.DeviceBuffer;

/// Cap on the im2col patch band (bytes); bands iterate over output rows.
const patch_band_bytes: usize = 256 << 20;

const none: DeviceBuffer = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 };

const Bufs = struct {
    x: DeviceBuffer = none,
    t: DeviceBuffer = none,
    u: DeviceBuffer = none,
    v: DeviceBuffer = none,
    patch: DeviceBuffer = none,
    // Mid-block attention scratch (tensor-core path).
    qh: DeviceBuffer = none,
    kh: DeviceBuffer = none,
    vh: DeviceBuffer = none,
    s: DeviceBuffer = none,
    md: DeviceBuffer = none,
    pt: DeviceBuffer = none,

    fn deinit(self: *Bufs, ctx: *Context) void {
        inline for (@typeInfo(Bufs).@"struct".fields) |f| {
            ctx.tensorDestroy(&@field(self, f.name));
        }
    }
};

/// Decode a VAE-space latent (planar [16][zh][zw], already denormalized) to
/// planar [3][8*zh][8*zw] pixels in [-1, 1]. Caller frees the result.
pub fn decode(dec: *const wan_vae.Decoder, ctx: *Context, io: std.Io, gpa: std.mem.Allocator, z: []const f32, zh: usize, zw: usize) ![]f32 {
    std.debug.assert(z.len == wan_vae.latent_channels * zh * zw);
    var bufs: Bufs = .{};
    defer bufs.deinit(ctx);

    var h = zh;
    var w = zw;
    const n0 = zh * zw;

    {
        const rows = try wan_vae.planarToRows(gpa, z, wan_vae.latent_channels, n0);
        defer gpa.free(rows);
        try ctx.ensureDeviceBuffer(&bufs.x, n0 * wan_vae.latent_channels * 4);
        try ctx.tensorUpload(bufs.x, std.mem.sliceAsBytes(rows));
    }

    try ctx.beginBatch();
    var batched = true;
    errdefer if (batched) ctx.endBatch() catch {};

    try conv(ctx, &bufs, &bufs.x, &bufs.t, h, w, dec.post_quant, false);
    std.mem.swap(DeviceBuffer, &bufs.x, &bufs.t);
    try conv(ctx, &bufs, &bufs.x, &bufs.t, h, w, dec.conv_in, false);
    std.mem.swap(DeviceBuffer, &bufs.x, &bufs.t);

    try res(ctx, &bufs, h, w, dec.mid_res1);
    try attn(ctx, &bufs, io, gpa, h, w, dec.mid_attn);

    // The attention scratch — above all the ~seq^2 scores plane (bufs.s,
    // ~1.7 GB at 1120x1680) — is dead for the rest of the decode, but it sits
    // at the START (mid-block runs at latent resolution) and would otherwise
    // stay resident through the whole 8x upsampling, doubling peak VRAM. Free
    // it now. endBatch does a submitAndWait, so no recorded dispatch still
    // references these buffers when we destroy them (the Xid 109 lesson).
    batched = false;
    try ctx.endBatch();
    inline for (.{ &bufs.qh, &bufs.kh, &bufs.vh, &bufs.s, &bufs.md, &bufs.pt }) |b| ctx.tensorDestroy(b);
    try ctx.beginBatch();
    batched = true;

    try res(ctx, &bufs, h, w, dec.mid_res2);

    for (dec.ups) |layer| switch (layer) {
        .res => |rb| try res(ctx, &bufs, h, w, rb),
        .up => |cv| {
            try conv(ctx, &bufs, &bufs.x, &bufs.t, h, w, cv, true);
            std.mem.swap(DeviceBuffer, &bufs.x, &bufs.t);
            h *= 2;
            w *= 2;
        },
    };

    // head: norm + silu + conv
    const n = h * w;
    try norm(ctx, &bufs.x, &bufs.t, n, dec.head_norm, true);
    try conv(ctx, &bufs, &bufs.t, &bufs.u, h, w, dec.head_conv, false);

    batched = false;
    try ctx.endBatch();

    const rgb_rows = try gpa.alloc(f32, n * 3);
    defer gpa.free(rgb_rows);
    try ctx.tensorDownload(bufs.u, std.mem.sliceAsBytes(rgb_rows));
    return wan_vae.rowsToPlanar(gpa, rgb_rows, 3, n);
}

/// norm(x) (+ optional fused silu) into `dst`.
fn norm(ctx: *Context, src: *const DeviceBuffer, dst: *DeviceBuffer, n: usize, gamma: []const f32, silu: bool) !void {
    const c = gamma.len;
    try ctx.ensureDeviceBuffer(dst, n * c * 4);
    const gamma_buf: DeviceBuffer = .{
        .buf = try ctx.smallBuffer(std.mem.sliceAsBytes(gamma)),
        .mem = .null_handle,
        .size = 0,
    };
    try ctx.opElt(.vae_norm, src.*, dst.*, gamma_buf, null, .{
        .u0 = @intCast(n),
        .u1 = @intCast(c),
        .u2 = @intFromBool(silu),
    }, n, 1, 1);
}

/// Conv into `dst` ([n_out][co]); with `up`, the source is read through a
/// fused nearest-exact 2x upsample and n_out covers the doubled dims.
/// Convs with co >= 96 run on tensor cores (f16 weights/activations, f32
/// accumulate — opMatmulCoopF16W); smaller ones (post_quant's 16, the
/// 3-channel head) stay on the f32 GEMM, where padding co to the 128-wide
/// coop tile would waste more than the tensor cores return.
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

/// Residual block over bufs.x in place (result swapped back into bufs.x).
/// conv2 reuses `u` (conv1's output is dead once norm2 has read it), so the
/// full-resolution `v` allocation is avoided — `v` then only grows to the
/// latent-resolution mid-attention size. One fewer full-res buffer (~720 MiB@1MP).
fn res(ctx: *Context, bufs: *Bufs, h: usize, w: usize, rb: wan_vae.ResBlock) !void {
    const n = h * w;
    try norm(ctx, &bufs.x, &bufs.t, n, rb.norm1, true);
    try conv(ctx, bufs, &bufs.t, &bufs.u, h, w, rb.conv1, false);
    try norm(ctx, &bufs.u, &bufs.t, n, rb.norm2, true); // consumes u
    try conv(ctx, bufs, &bufs.t, &bufs.u, h, w, rb.conv2, false); // reuses u
    if (rb.shortcut) |sc| {
        try conv(ctx, bufs, &bufs.x, &bufs.t, h, w, sc, false);
        try ctx.opElt(.add, bufs.u, bufs.t, null, null, .{ .u0 = @intCast(n * rb.conv2.co) }, n * rb.conv2.co, 1, 1);
    } else {
        try ctx.opElt(.add, bufs.u, bufs.x, null, null, .{ .u0 = @intCast(n * rb.conv2.co) }, n * rb.conv2.co, 1, 1);
    }
    std.mem.swap(DeviceBuffer, &bufs.x, &bufs.u);
}

/// Mid-block single-head attention, fully on the GPU when the tensor-core
/// attention pipelines exist: q/k/v as three GEMMs over slices of the packed
/// qkv weight, scores via the head_dim-384 coop kernel, two-pass softmax,
/// and P@V with the 384-wide head batched as three fake 128-column heads
/// sharing the single scores/MD plane (u1 = 0, f1 = 0). Falls back to the
/// CPU attention core otherwise.
fn attn(ctx: *Context, bufs: *Bufs, io: std.Io, gpa: std.mem.Allocator, h: usize, w: usize, ab: wan_vae.AttnBlock) !void {
    const n = h * w;
    const c = ab.qkv.ci;
    try norm(ctx, &bufs.x, &bufs.t, n, ab.norm, false);
    if (ctx.pipe_scores_vae == .null_handle or c != 384) {
        return attnCpuCore(ctx, bufs, io, gpa, n, c, ab);
    }

    const seq_pad = std.mem.alignForward(usize, n, 128);
    const nchunks = 32;
    const wq = std.mem.sliceAsBytes(ab.qkv.w[0 .. c * c]);
    const wk = std.mem.sliceAsBytes(ab.qkv.w[c * c .. 2 * c * c]);
    const wv = std.mem.sliceAsBytes(ab.qkv.w[2 * c * c .. 3 * c * c]);
    try ctx.ensureDeviceBuffer(&bufs.u, seq_pad * c * 4); // q, then attn out
    try ctx.ensureDeviceBuffer(&bufs.v, n * c * 4); // k, then proj out
    try ctx.ensureDeviceBuffer(&bufs.patch, n * c * 4); // v
    try ctx.opMatmul(bufs.u, 0, bufs.t, 0, n, wq, false, c, c, 1.0, ab.qkv.b[0..c]);
    try ctx.opMatmul(bufs.v, 0, bufs.t, 0, n, wk, false, c, c, 1.0, ab.qkv.b[c .. 2 * c]);
    try ctx.opMatmul(bufs.patch, 0, bufs.t, 0, n, wv, false, c, c, 1.0, ab.qkv.b[2 * c .. 3 * c]);

    // f16 operands: Q with the softmax scale prefolded (zero pad rows), K
    // per-head k-major (zero pad columns), V zero-padded rows.
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c)));
    try ctx.ensureDeviceBuffer(&bufs.qh, seq_pad * c * 2);
    try ctx.opElt(.f32_to_h16, bufs.u, null, null, bufs.qh, .{
        .u0 = @intCast(seq_pad * c / 2),
        .u1 = @intCast(n * c),
        .f0 = scale,
    }, seq_pad * c / 2, 1, 1);
    try ctx.ensureDeviceBuffer(&bufs.kh, c * seq_pad * 2);
    try ctx.opElt(.gather_kmajor_h16, bufs.v, null, null, bufs.kh, .{
        .u0 = @intCast(c * seq_pad / 2),
        .u1 = @intCast(c),
        .u2 = @intCast(seq_pad),
        .u3 = @intCast(n),
        .u4 = 1,
    }, c * seq_pad / 2, 1, 1);
    try ctx.ensureDeviceBuffer(&bufs.vh, seq_pad * c * 2);
    try ctx.opElt(.f32_to_h16, bufs.patch, null, null, bufs.vh, .{
        .u0 = @intCast(seq_pad * c / 2),
        .u1 = @intCast(n * c),
        .f0 = 1.0,
    }, seq_pad * c / 2, 1, 1);

    // S = Q@K^T, stored f16 [seq_pad][seq_pad].
    try ctx.ensureDeviceBuffer(&bufs.s, seq_pad * seq_pad * 2);
    try ctx.opAttnScoresVae(bufs.s, bufs.qh, bufs.kh, .{
        .u0 = @intCast(c),
        .u1 = @intCast(seq_pad),
        .u2 = 0,
        .u3 = 1,
        .u4 = @intCast(c * seq_pad),
        .u5 = @intCast(seq_pad * seq_pad),
    }, seq_pad / 128, seq_pad / 128, 1);

    // Two-pass softmax -> per-row {max, 1/denom}.
    try ctx.ensureDeviceBuffer(&bufs.pt, n * nchunks * 2 * 4);
    try ctx.opElt(.softmax_partial, bufs.s, null, null, bufs.pt, .{
        .u0 = @intCast(n * nchunks),
        .u1 = nchunks,
        .u2 = @intCast(n),
        .u3 = @intCast(seq_pad),
        .u5 = 0,
    }, n * nchunks, 1, 1);
    try ctx.ensureDeviceBuffer(&bufs.md, seq_pad * 2 * 4);
    try ctx.opElt(.softmax_combine, bufs.pt, null, null, bufs.md, .{
        .u0 = @intCast(n),
        .u1 = nchunks,
        .u2 = @intCast(n),
        .u3 = @intCast(seq_pad),
    }, n, 1, 1);

    // P@V into bufs.u (its q content was consumed by the f16 conversion).
    try ctx.opAttnOut(bufs.s, bufs.vh, bufs.u, bufs.md, .{
        .u0 = @intCast(seq_pad),
        .u1 = 0,
        .u2 = 0,
        .u3 = 1,
        .u4 = @intCast(c),
        .u5 = @intCast(c),
        .f0 = @bitCast(@as(u32, @intCast(n))),
        .f1 = @bitCast(@as(u32, 0)),
    }, seq_pad / 128, 3);

    try ctx.opMatmul(bufs.v, 0, bufs.u, 0, n, std.mem.sliceAsBytes(ab.proj.w), false, c, c, 1.0, ab.proj.b);
    try ctx.opElt(.add, bufs.x, bufs.v, null, null, .{ .u0 = @intCast(n * c) }, n * c, 1, 1);
}

/// CPU fallback for the attention core (one qkv download + one upload).
fn attnCpuCore(ctx: *Context, bufs: *Bufs, io: std.Io, gpa: std.mem.Allocator, n: usize, c: usize, ab: wan_vae.AttnBlock) !void {
    try ctx.ensureDeviceBuffer(&bufs.u, n * 3 * c * 4);
    try ctx.opMatmul(bufs.u, 0, bufs.t, 0, n, std.mem.sliceAsBytes(ab.qkv.w), false, 3 * c, c, 1.0, ab.qkv.b);

    const qkv = try gpa.alloc(f32, n * 3 * c);
    defer gpa.free(qkv);
    try ctx.tensorDownload(bufs.u, std.mem.sliceAsBytes(qkv));

    const q = try gpa.alloc(f32, n * c);
    defer gpa.free(q);
    const k = try gpa.alloc(f32, n * c);
    defer gpa.free(k);
    const v = try gpa.alloc(f32, n * c);
    defer gpa.free(v);
    for (0..n) |i| {
        const row = qkv[i * 3 * c ..];
        @memcpy(q[i * c ..][0..c], row[0..c]);
        @memcpy(k[i * c ..][0..c], row[c .. 2 * c]);
        @memcpy(v[i * c ..][0..c], row[2 * c .. 3 * c]);
    }
    const attn_out = try gpa.alloc(f32, n * c);
    defer gpa.free(attn_out);
    try ops.attention.attention(io, gpa, attn_out, q, k, v, .{
        .seq_q = n,
        .seq_kv = n,
        .n_heads = 1,
        .n_kv_heads = 1,
        .head_dim = c,
    });

    try ctx.ensureDeviceBuffer(&bufs.t, n * c * 4);
    try ctx.tensorUpload(bufs.t, std.mem.sliceAsBytes(attn_out));
    try ctx.ensureDeviceBuffer(&bufs.u, n * c * 4);
    try ctx.opMatmul(bufs.u, 0, bufs.t, 0, n, std.mem.sliceAsBytes(ab.proj.w), false, c, c, 1.0, ab.proj.b);
    try ctx.opElt(.add, bufs.x, bufs.u, null, null, .{ .u0 = @intCast(n * c) }, n * c, 1, 1);
}

// --- tests -----------------------------------------------------------------

// Isolated conv parity: GPU banded im2col + GEMM (both plain and fused-2x)
// vs the CPU conv2d reference, on random data. Gated like the other GPU
// tests (see context.zig).
test "gpu conv matches cpu conv2d" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    var ctx = Context.init(gpa) catch return error.SkipZigTest;
    defer ctx.deinit();

    var prng = std.Random.DefaultPrng.init(31);
    const rand = prng.random();
    const h = 7;
    const w = 5;
    const ci = 6;
    const co = 4;

    const wdata = try gpa.alloc(f32, co * 9 * ci);
    defer gpa.free(wdata);
    for (wdata) |*x| x.* = rand.floatNorm(f32);
    const bias = try gpa.alloc(f32, co);
    defer gpa.free(bias);
    for (bias) |*x| x.* = rand.floatNorm(f32);
    const in = try gpa.alloc(f32, h * w * ci);
    defer gpa.free(in);
    for (in) |*x| x.* = rand.floatNorm(f32);

    const cv: wan_vae.Conv2d = .{ .w = wdata, .b = bias, .co = co, .ci = ci, .k = 3 };

    var bufs: Bufs = .{};
    defer bufs.deinit(ctx);
    try ctx.ensureDeviceBuffer(&bufs.x, in.len * 4);
    try ctx.tensorUpload(bufs.x, std.mem.sliceAsBytes(in));

    // Plain conv.
    {
        try conv(ctx, &bufs, &bufs.x, &bufs.t, h, w, cv, false);
        const got = try gpa.alloc(f32, h * w * co);
        defer gpa.free(got);
        try ctx.tensorDownload(bufs.t, std.mem.sliceAsBytes(got));

        const want = try gpa.alloc(f32, h * w * co);
        defer gpa.free(want);
        try cpuConv(io, gpa, want, in, h, w, cv);
        for (want, got) |e, a| try std.testing.expectApproxEqAbs(e, a, 1e-4);
    }

    // Fused 2x upsample + conv vs explicit CPU nearest2x + conv.
    {
        try conv(ctx, &bufs, &bufs.x, &bufs.t, h, w, cv, true);
        const got = try gpa.alloc(f32, 4 * h * w * co);
        defer gpa.free(got);
        try ctx.tensorDownload(bufs.t, std.mem.sliceAsBytes(got));

        const up = try gpa.alloc(f32, 4 * h * w * ci);
        defer gpa.free(up);
        for (0..2 * h) |y| {
            for (0..2 * w) |x| {
                const src = in[(y / 2 * w + x / 2) * ci ..][0..ci];
                @memcpy(up[(y * 2 * w + x) * ci ..][0..ci], src);
            }
        }
        const want = try gpa.alloc(f32, 4 * h * w * co);
        defer gpa.free(want);
        try cpuConv(io, gpa, want, up, 2 * h, 2 * w, cv);
        for (want, got) |e, a| try std.testing.expectApproxEqAbs(e, a, 1e-4);
    }
}

/// Reference conv for the test (direct, no im2col).
fn cpuConv(io: std.Io, gpa: std.mem.Allocator, out: []f32, in: []const f32, h: usize, w: usize, cv: wan_vae.Conv2d) !void {
    _ = io;
    _ = gpa;
    for (0..h) |y| {
        for (0..w) |x| {
            for (0..cv.co) |o| {
                var acc: f64 = cv.b[o];
                for (0..3) |ky| {
                    for (0..3) |kx| {
                        const sy = @as(isize, @intCast(y + ky)) - 1;
                        const sx = @as(isize, @intCast(x + kx)) - 1;
                        if (sy < 0 or sy >= h or sx < 0 or sx >= w) continue;
                        for (0..cv.ci) |cc| {
                            const wv = cv.w[((o * 3 + ky) * 3 + kx) * cv.ci + cc];
                            acc += @as(f64, wv) * in[(@as(usize, @intCast(sy)) * w + @as(usize, @intCast(sx))) * cv.ci + cc];
                        }
                    }
                }
                out[(y * w + x) * cv.co + o] = @floatCast(acc);
            }
        }
    }
}

// Full-decoder parity against the same ComfyUI fixture the CPU test uses;
// needs the checkpoint, the fixture, and the GPU marker.
test "gpu decode matches comfyui reference" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    const vae_path = "models/vae/krea2RealVae_v10.safetensors";
    std.Io.Dir.cwd().access(io, vae_path, .{}) catch return error.SkipZigTest;
    std.Io.Dir.cwd().access(io, "testdata/vae_z_8x8.bin", .{}) catch return error.SkipZigTest;
    var ctx = Context.init(gpa) catch return error.SkipZigTest;
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

    var st = try @import("../safetensors.zig").SafeTensors.open(gpa, io, vae_path);
    defer st.deinit();
    var dec = try wan_vae.Decoder.load(gpa, &st);
    defer dec.deinit();

    const out = try decode(&dec, ctx, io, gpa, z, 8, 8);
    defer gpa.free(out);

    var max_err: f32 = 0;
    var sum_err: f64 = 0;
    for (expected, out) |e, a| {
        const err = @abs(e - a);
        max_err = @max(max_err, err);
        sum_err += err;
    }
    const mean_err = sum_err / @as(f64, @floatFromInt(out.len));
    std.debug.print("vae gpu parity: max_err={d:.6} mean_err={d:.6}\n", .{ max_err, mean_err });
    try std.testing.expect(max_err < 5e-3);
    try std.testing.expect(mean_err < 5e-4);
}
