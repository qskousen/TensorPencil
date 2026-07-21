//! Device test: the CUDA ViT eltwise kernels (LayerNorm+bias, 2-D vision RoPE,
//! head restride) checked against their CPU references. Lives in the model tier
//! (not the gpu backend) because it depends on both the gpu backend
//! (`tp_gpu.cuda.Backend`) and a model's CPU reference (`vit35.applyVisionRope`),
//! which the gpu module — a lower layer — cannot import. Self-skips without a
//! CUDA device / the integration build.

const std = @import("std");
const cuda = @import("tp_gpu").cuda;
const norm = @import("tp_ops").norm;
const vit35 = @import("vit35.zig");

const Backend = cuda.Backend;

test "vit eltwise kernels match CPU reference" {
    const gpa = std.testing.allocator;
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();
    var prng = std.Random.DefaultPrng.init(77);
    const rand = prng.random();

    // LayerNorm with weight+bias.
    {
        const rows = 5;
        const dim = 384;
        const x = try gpa.alloc(f32, rows * dim);
        defer gpa.free(x);
        for (x) |*v| v.* = rand.floatNorm(f32) * 2.0 + 0.5;
        const w = try gpa.alloc(f32, dim);
        defer gpa.free(w);
        const b = try gpa.alloc(f32, dim);
        defer gpa.free(b);
        for (w) |*v| v.* = rand.floatNorm(f32);
        for (b) |*v| v.* = rand.floatNorm(f32);

        const x_d = try be.tensorCreate(rows * dim * 4);
        const y_d = try be.tensorCreate(rows * dim * 4);
        defer {
            var xd = x_d;
            var yd = y_d;
            be.tensorDestroy(&xd);
            be.tensorDestroy(&yd);
        }
        try be.tensorUpload(x_d, std.mem.sliceAsBytes(x));
        try be.opLayerNorm(x_d, y_d, w, b, rows, dim, 1e-6);

        const got = try gpa.alloc(f32, rows * dim);
        defer gpa.free(got);
        try be.tensorDownload(y_d, std.mem.sliceAsBytes(got));
        const want = try gpa.alloc(f32, rows * dim);
        defer gpa.free(want);
        norm.layerNorm(want, x, w, b, 1e-6);
        for (want, got) |e, a| try std.testing.expectApproxEqAbs(e, a, 1e-3);
    }

    // 2-D vision rope, packed heads (head_dim == 4*half).
    const np = 6;
    const heads = 2;
    const half = 2;
    const hd = 4 * half;
    const max_pos = 4;
    const py = [np]u32{ 0, 0, 1, 1, 2, 3 };
    const px = [np]u32{ 0, 1, 0, 1, 3, 2 };
    var freqs: [2 * max_pos * half]f32 = undefined; // cos then sin
    const sin_off = max_pos * half;
    for (0..max_pos) |p| {
        for (0..half) |i| {
            const theta = @as(f64, @floatFromInt(p)) * std.math.pow(f64, 10000.0, -2.0 * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(2 * half)));
            freqs[p * half + i] = @floatCast(@cos(theta));
            freqs[sin_off + p * half + i] = @floatCast(@sin(theta));
        }
    }
    var q: [np * heads * hd]f32 = undefined;
    for (&q) |*v| v.* = rand.floatNorm(f32);
    var want_q: [q.len]f32 = undefined;
    @memcpy(&want_q, &q);
    vit35.applyVisionRope(&want_q, np, heads, hd, &py, &px, freqs[0..sin_off], freqs[sin_off..], half);

    var pos2: [np * 2]u32 = undefined;
    for (0..np) |t| {
        pos2[t * 2] = py[t];
        pos2[t * 2 + 1] = px[t];
    }
    const q_d = try be.tensorCreate(q.len * 4);
    const pos2_d = try be.tensorCreate(pos2.len * 4);
    const freqs_d = try be.tensorCreate(freqs.len * 4);
    defer {
        var qd = q_d;
        var pd = pos2_d;
        var fd = freqs_d;
        be.tensorDestroy(&qd);
        be.tensorDestroy(&pd);
        be.tensorDestroy(&fd);
    }
    try be.tensorUpload(q_d, std.mem.sliceAsBytes(&q));
    try be.tensorUpload(pos2_d, std.mem.sliceAsBytes(&pos2));
    try be.tensorUpload(freqs_d, std.mem.sliceAsBytes(&freqs));
    try be.opRopeVision(q_d, pos2_d, freqs_d, np, heads, half, sin_off, hd);
    var got_q: [q.len]f32 = undefined;
    try be.tensorDownload(q_d, std.mem.sliceAsBytes(&got_q));
    for (want_q, got_q) |e, a| try std.testing.expectApproxEqAbs(e, a, 1e-5);

    // Head restride: extract the middle third of a qkv-like row, pad
    // 5-dim heads to 8, then compact back to the original.
    {
        const rows = 3;
        const in_hd = 5;
        const out_hd = 8;
        const stride = 3 * heads * in_hd; // [q k v] fused rows
        var src: [rows * stride]f32 = undefined;
        for (&src) |*v| v.* = rand.floatNorm(f32);
        const src_d = try be.tensorCreate(src.len * 4);
        const pad_d = try be.tensorCreate(rows * heads * out_hd * 4);
        const back_d = try be.tensorCreate(rows * heads * in_hd * 4);
        defer {
            var sd = src_d;
            var pd = pad_d;
            var bd = back_d;
            be.tensorDestroy(&sd);
            be.tensorDestroy(&pd);
            be.tensorDestroy(&bd);
        }
        try be.tensorUpload(src_d, std.mem.sliceAsBytes(&src));
        try be.opHeadPad(pad_d, src_d, rows, heads, out_hd, in_hd, stride, heads * in_hd);
        var padded: [rows * heads * out_hd]f32 = undefined;
        try be.tensorDownload(pad_d, std.mem.sliceAsBytes(&padded));
        for (0..rows) |t| {
            for (0..heads) |h| {
                for (0..out_hd) |d| {
                    const e: f32 = if (d < in_hd) src[t * stride + heads * in_hd + h * in_hd + d] else 0;
                    try std.testing.expectEqual(e, padded[(t * heads + h) * out_hd + d]);
                }
            }
        }
        try be.opHeadPad(back_d, pad_d, rows, heads, in_hd, out_hd, heads * out_hd, 0);
        var back: [rows * heads * in_hd]f32 = undefined;
        try be.tensorDownload(back_d, std.mem.sliceAsBytes(&back));
        for (0..rows) |t| {
            for (0..heads) |h| {
                for (0..in_hd) |d| {
                    try std.testing.expectEqual(src[t * stride + heads * in_hd + h * in_hd + d], back[(t * heads + h) * in_hd + d]);
                }
            }
        }
    }
}
