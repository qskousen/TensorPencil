//! Tiled VAE decode.
//!
//! The VAE decodes the whole image at once, so its peak VRAM scales with image
//! area — and the mid-block self-attention materializes an O(seq²) scores plane
//! (seq = latent H·W), which is *quadratic* in area (20 GB at a 2560² image on
//! a single head). Large images therefore OOM on the GPU and fall back to a slow
//! CPU decode.
//!
//! This decodes the latent in overlapping spatial tiles instead: each tile
//! decodes an at-most `tile×tile` latent region, so both the attention plane and
//! the conv activation buffers stay bounded regardless of the final image size.
//! The 8× pixel overlap between adjacent tiles is feather-blended so the seams
//! are invisible. Because the decoder is convolutional (with zero-padded 3×3
//! convs), a tile decoded in isolation differs from the whole-image decode only
//! near its borders; the overlap + feather hides that. The mid-block attention
//! becomes tile-local, the standard trade-off for tiled VAE decode.
//!
//! Backend-agnostic: the per-tile decode is supplied as `decodeTile`, so the
//! same tiling drives the CPU, CUDA and Vulkan decoders.

const std = @import("std");
const wan_vae = @import("wan_vae.zig");

/// VAE spatial upscale factor (latent → pixels).
const scale = 8;

pub const Params = struct {
    /// Latent tile size (side, in latent pixels). Caps the per-tile attention
    /// plane at (tile²)² · 2 bytes and conv buffers at (8·tile)² · maxC · 4.
    /// 128 → a 512 MiB attention plane, ~1 MP conv tiles.
    tile: usize = 128,
    /// Latent overlap between adjacent tiles (feathered over 8·overlap pixels).
    /// Kept < tile so the step stays positive.
    overlap: usize = 16,
};

/// Decode a denormalized planar [16][zh][zw] latent to planar [3][8·zh][8·zw]
/// pixels in [-1, 1] by decoding overlapping tiles and feather-blending them.
///
/// `decodeTile(ctx, gpa, io, sub, th, tw)` decodes a planar [16][th][tw]
/// sub-latent to planar [3][8·th][8·tw] pixels (it owns the result; this frees
/// it). Caller frees the returned image.
pub fn decode(
    gpa: std.mem.Allocator,
    io: std.Io,
    z: []const f32,
    zh: usize,
    zw: usize,
    params: Params,
    ctx: anytype,
    comptime decodeTile: fn (@TypeOf(ctx), std.mem.Allocator, std.Io, []const f32, usize, usize) anyerror![]f32,
) ![]f32 {
    const cin = wan_vae.latent_channels; // 16
    const cout = 3;
    std.debug.assert(z.len == cin * zh * zw);

    const tile = @max(@as(usize, 1), params.tile);
    const ov = if (params.overlap >= tile) tile / 4 else params.overlap;
    const step = tile - ov; // >= 1 (ov < tile)
    const pf = ov * scale; // pixel feather width

    const height = zh * scale;
    const width = zw * scale;
    const plane = height * width;

    // Weighted accumulation over the full image (host RAM), normalized at the end.
    const accum = try gpa.alloc(f32, cout * plane);
    defer gpa.free(accum);
    @memset(accum, 0);
    const wsum = try gpa.alloc(f32, plane);
    defer gpa.free(wsum);
    @memset(wsum, 0);

    var r0: usize = 0;
    while (true) : (r0 += step) {
        const th = @min(tile, zh - r0);
        var c0: usize = 0;
        while (true) : (c0 += step) {
            const tw = @min(tile, zw - c0);

            // Extract the sub-latent (planar [cin][th][tw]).
            const sub = try gpa.alloc(f32, cin * th * tw);
            defer gpa.free(sub);
            for (0..cin) |ch| {
                for (0..th) |ty| {
                    const src = z[(ch * zh + r0 + ty) * zw + c0 ..][0..tw];
                    @memcpy(sub[(ch * th + ty) * tw ..][0..tw], src);
                }
            }

            const px = try decodeTile(ctx, gpa, io, sub, th, tw); // [cout][ph][pw]
            defer gpa.free(px);

            const ph = th * scale;
            const pw = tw * scale;
            const oy0 = r0 * scale;
            const ox0 = c0 * scale;

            // Only interior edges are feathered; edges on the image border are
            // covered by a single tile, so keep their full weight.
            const top = r0 > 0;
            const bot = r0 + th < zh;
            const left = c0 > 0;
            const right = c0 + tw < zw;

            for (0..ph) |ty| {
                const wy = feather(ty, ph, pf, top, bot);
                const oy = oy0 + ty;
                for (0..pw) |tx| {
                    const w = wy * feather(tx, pw, pf, left, right);
                    const ox = ox0 + tx;
                    const o = oy * width + ox;
                    wsum[o] += w;
                    for (0..cout) |ch| {
                        accum[ch * plane + o] += w * px[(ch * ph + ty) * pw + tx];
                    }
                }
            }

            if (c0 + tw >= zw) break;
        }
        if (r0 + th >= zh) break;
    }

    const out = try gpa.alloc(f32, cout * plane);
    errdefer gpa.free(out);
    for (0..plane) |i| {
        const inv = 1.0 / @max(wsum[i], 1e-8);
        for (0..cout) |ch| out[ch * plane + i] = accum[ch * plane + i] * inv;
    }
    return out;
}

/// Linear feather weight in (0, 1] for pixel `i` of a `len`-pixel tile axis:
/// ramps up over the first `pf` pixels when the near edge is an interior seam,
/// and down over the last `pf` when the far edge is. Always > 0 (so the
/// normalization can never divide by zero where a tile contributes).
fn feather(i: usize, len: usize, pf: usize, near_seam: bool, far_seam: bool) f32 {
    if (pf == 0) return 1.0;
    const pff: f32 = @floatFromInt(pf);
    var w: f32 = 1.0;
    if (near_seam and i < pf) {
        w = @min(w, (@as(f32, @floatFromInt(i)) + 0.5) / pff);
    }
    if (far_seam and i + pf >= len) {
        w = @min(w, (@as(f32, @floatFromInt(len - 1 - i)) + 0.5) / pff);
    }
    return w;
}

// --- tests -----------------------------------------------------------------

const testing = std.testing;

/// A purely-local "decode": nearest upsample of latent channel 0 into all 3
/// output channels. Because every output pixel depends only on the co-located
/// latent value, overlapping tiles agree exactly in their overlap, so the
/// feather-blended tiled result must equal a single whole-image decode. This
/// isolates the tiling/blend math from any real conv/attention behavior.
fn upsampleCh0(_: void, gpa: std.mem.Allocator, io: std.Io, sub: []const f32, th: usize, tw: usize) anyerror![]f32 {
    _ = io;
    const ph = th * scale;
    const pw = tw * scale;
    const out = try gpa.alloc(f32, 3 * ph * pw);
    for (0..ph) |ty| {
        for (0..pw) |tx| {
            const v = sub[(ty / scale) * tw + (tx / scale)]; // channel 0
            for (0..3) |ch| out[(ch * ph + ty) * pw + tx] = v;
        }
    }
    return out;
}

test "tiled decode of a local op equals the whole decode" {
    const gpa = testing.allocator;
    const io = std.testing.io;
    const zh = 40;
    const zw = 56;
    const cin = wan_vae.latent_channels;

    const z = try gpa.alloc(f32, cin * zh * zw);
    defer gpa.free(z);
    // Smooth-ish deterministic latent so borders are non-trivial.
    for (0..cin) |ch| {
        for (0..zh) |y| {
            for (0..zw) |x| {
                const fy: f32 = @floatFromInt(y);
                const fx: f32 = @floatFromInt(x);
                const fc: f32 = @floatFromInt(ch);
                z[(ch * zh + y) * zw + x] = @sin(fy * 0.2 + fc) * @cos(fx * 0.15) + fc * 0.01;
            }
        }
    }

    const whole = try upsampleCh0({}, gpa, io, z, zh, zw);
    defer gpa.free(whole);

    // Several tile/overlap configs, including tile larger than the image.
    const cfgs = [_]Params{
        .{ .tile = 16, .overlap = 4 },
        .{ .tile = 24, .overlap = 8 },
        .{ .tile = 100, .overlap = 16 }, // > zh, single row of tiles
    };
    for (cfgs) |p| {
        const tiled = try decode(gpa, io, z, zh, zw, p, {}, upsampleCh0);
        defer gpa.free(tiled);
        try testing.expectEqual(whole.len, tiled.len);
        for (whole, tiled) |a, b| {
            try testing.expect(!std.math.isNan(b));
            try testing.expectApproxEqAbs(a, b, 1e-4);
        }
    }
}

test "feather ramps up and down over the seam width" {
    // Interior on both sides: 0.5/pf at the edges, 1.0 in the middle.
    const len = 64;
    const pf = 8;
    try testing.expectApproxEqAbs(@as(f32, 0.5 / 8.0), feather(0, len, pf, true, true), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.0), feather(len / 2, len, pf, true, true), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.5 / 8.0), feather(len - 1, len, pf, true, true), 1e-6);
    // Border edges keep full weight.
    try testing.expectApproxEqAbs(@as(f32, 1.0), feather(0, len, pf, false, true), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.0), feather(len - 1, len, pf, true, false), 1e-6);
}
