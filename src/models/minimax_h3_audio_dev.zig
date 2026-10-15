//! The MiniMax H3 audio VAE (BigVGAN) decode, written once for every GPU backend over
//! an adapter (`minimax_h3_audio_cuda.Dev`, `minimax_h3_audio_gpu.Dev`). Signals are
//! channel-last `[len][ch]` and the weights are the session's permuted copies
//! (`audio.DevSession`); `minimax_h3_audio_cuda`'s header has why.
//!
//! The adapter holds `ws` (`up`, the `blk0`/`blk1` ping-pong pair, `acc`) and supplies
//! `conv` (im2col and the f32 GEMM; `dst` may alias `src`, the GEMM reads only the
//! patch matrix), `convT`, the anti-aliased activation's two halves `aaUp` / `aaDown`
//! (through the workspace's `2 * len` intermediate), `add`, `copy`, `scale` and the
//! batch and transfer ops.

const std = @import("std");
const audio = @import("minimax_h3_audio.zig");

/// Normalized stereo latents `[32][2][t]` (planar, the sampler's layout) to
/// interleaved samples in [-1, 1], `[len][2]`, one channel per batch.
pub fn decode(d: anytype, dec: *const audio.AudioDecoder, sess: *const audio.DevSession, gpa: std.mem.Allocator, out: []f32, z: []const f32, t: usize, cancel: ?*std.atomic.Value(bool)) !void {
    const ws = d.ws;
    const c_lat = dec.dec_in.in_ch;
    const samples = ws.shapes.samples;
    std.debug.assert(z.len == c_lat * audio.stereo * t);
    std.debug.assert(out.len == samples * audio.stereo);
    std.debug.assert(samples == t * dec.upsampleFactor());

    const lat = try gpa.alloc(f32, t * c_lat);
    defer gpa.free(lat);
    const tail = try gpa.alloc(f32, samples);
    defer gpa.free(tail);

    for (0..audio.stereo) |s| {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        audio.channelLatent(dec, lat, z, t, s);
        try d.beginBatch();
        errdefer d.abortBatch();
        try d.upload(ws.blk0, lat);
        const n = try channel(d, dec, sess, t, cancel);
        std.debug.assert(n == samples);
        try d.endBatch();
        try d.download(ws.blk0, tail);
        audio.storeChannel(out, tail, s);
    }
}

/// One channel from its latent rows in `blk0` to its samples in `blk0`; returns the
/// sample count.
fn channel(d: anytype, dec: *const audio.AudioDecoder, sess: *const audio.DevSession, t: usize, cancel: ?*std.atomic.Value(bool)) !usize {
    const ws = d.ws;
    _ = try d.conv(ws.blk1, ws.blk0, sess.dec_in, t);
    _ = try d.conv(ws.acc, ws.blk1, sess.conv_pre, t);

    var len = t;
    for (sess.ups, 0..) |u, i| {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        const up_len = (len - 1) * u.stride + u.k - 2 * u.padding;
        try d.convT(ws.up, ws.acc, u, len, up_len);
        len = up_len;

        // The stage's resblocks are SUMMED and averaged, not chained, so each reads
        // the SAME `up` and the sum lands in `acc`.
        const n = u.out_ch * len;
        for (0..dec.n_kernels) |j| {
            const r = try ampBlock(d, ws.up, &sess.blocks[i * dec.n_kernels + j], u.out_ch, len);
            if (j == 0) try d.copy(ws.acc, r, n) else try d.add(ws.acc, r, n);
        }
        try d.scale(ws.acc, 1.0 / @as(f32, @floatFromInt(dec.n_kernels)), n);
    }

    try aaSnake(d, ws.acc, sess.act_post, len);
    std.debug.assert(sess.conv_post.outCh() == 1);
    return d.conv(ws.blk0, ws.acc, sess.conv_post, len);
}

/// One `AMPBlock1` over `src`, which is NOT modified. Returns the buffer holding the
/// result (one of the ping-pong pair).
fn ampBlock(d: anytype, src: anytype, blk: *const audio.DevSession.DevBlock, ch: usize, len: usize) !@TypeOf(src) {
    var cur = src;
    var dst = d.ws.blk0;
    var other = d.ws.blk1;
    for (blk.convs1, blk.convs2, 0..) |c1, c2, i| {
        // activations[::2] pairs with convs1 and [1::2] with convs2: six distinct
        // activations per block, not three reused.
        try d.aaUp(cur, blk.acts[2 * i], len);
        try d.aaDown(dst, blk.acts[2 * i], len);
        _ = try d.conv(dst, dst, c1, len);
        try aaSnake(d, dst, blk.acts[2 * i + 1], len);
        _ = try d.conv(dst, dst, c2, len);
        try d.add(dst, cur, ch * len);
        cur = dst;
        // Ping-pong, so the next iteration's residual source stays intact.
        const tmp = dst;
        dst = other;
        other = tmp;
    }
    return cur;
}

/// The anti-aliased activation in place: `x[len][ch]` -> `x[len][ch]`.
fn aaSnake(d: anytype, x: anytype, act: audio.DevAct, len: usize) !void {
    std.debug.assert(act.channels * len <= d.ws.shapes.aa);
    try d.aaUp(x, act, len);
    try d.aaDown(x, act, len);
}
