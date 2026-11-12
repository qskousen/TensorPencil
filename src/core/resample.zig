//! PIL's Lanczos resize, which is what ComfyUI's `common_upscale(..., "lanczos")`
//! runs: the image goes to 8 bits, through Pillow's fixed-point two-pass resampler,
//! and back to float.
//!
//! Each pass rounds to 8 bits, so the order (horizontal first) is part of the
//! answer. The float-to-byte step TRUNCATES
//! `255 * v` computed in f32, so a pixel that came in as `k / 255` can leave as
//! `k - 1`; ComfyUI does this to every picture it resizes. Pinned against Pillow by
//! `tools/gen_resample_fixture.py`.

const std = @import("std");

/// Fractional bits of Pillow's 8-bit coefficients (`32 - 8 - 2`).
const precision_bits = 22;

fn sinc(x: f64) f64 {
    if (x == 0) return 1;
    const px = x * std.math.pi;
    return @sin(px) / px;
}

fn lanczos3(x: f64) f64 {
    if (-3.0 <= x and x < 3.0) return sinc(x) * sinc(x / 3.0);
    return 0;
}

/// One axis's taps: for each output position, the first input index and the
/// integer coefficients over `[xmin, xmin + n)`.
const Coeffs = struct {
    ksize: usize,
    xmin: []usize,
    n: []usize,
    k: []i32,

    /// Pillow's `precompute_coeffs` then `normalize_coeffs_8bpc`, in f64.
    fn init(a: std.mem.Allocator, in_size: usize, out_size: usize) !Coeffs {
        const scale = @as(f64, @floatFromInt(in_size)) / @as(f64, @floatFromInt(out_size));
        const filterscale = @max(scale, 1.0);
        const support = 3.0 * filterscale;
        const ksize: usize = @as(usize, @intFromFloat(@ceil(support))) * 2 + 1;
        const c: Coeffs = .{
            .ksize = ksize,
            .xmin = try a.alloc(usize, out_size),
            .n = try a.alloc(usize, out_size),
            .k = try a.alloc(i32, out_size * ksize),
        };
        const pre = try a.alloc(f64, ksize);
        const ss = 1.0 / filterscale;
        for (0..out_size) |xx| {
            const center = (@as(f64, @floatFromInt(xx)) + 0.5) * scale;
            // `(int)` truncates toward zero, then the bounds clamp
            const lo = @max(truncInt(center - support + 0.5), 0);
            const hi = @min(truncInt(center + support + 0.5), @as(i64, @intCast(in_size)));
            const xmin: usize = @intCast(lo);
            const n: usize = @intCast(@max(hi - lo, 0));
            var ww: f64 = 0;
            for (0..n) |x| {
                const w = lanczos3((@as(f64, @floatFromInt(x + xmin)) - center + 0.5) * ss);
                pre[x] = w;
                ww += w;
            }
            const row = c.k[xx * ksize ..][0..ksize];
            @memset(row, 0);
            for (0..n) |x| {
                const v = if (ww != 0) pre[x] / ww else pre[x];
                const scaled = v * @as(f64, @floatFromInt(@as(i64, 1) << precision_bits));
                row[x] = @intFromFloat(if (v < 0) -0.5 + scaled else 0.5 + scaled);
            }
            c.xmin[xx] = xmin;
            c.n[xx] = n;
        }
        return c;
    }

    fn apply(c: Coeffs, i: usize, src: anytype) u8 {
        var ss: i32 = 1 << (precision_bits - 1);
        for (c.k[i * c.ksize ..][0..c.n[i]], 0..) |w, x| ss += @as(i32, src.at(c.xmin[i] + x)) * w;
        return @intCast(std.math.clamp(ss >> precision_bits, 0, 255));
    }
};

fn truncInt(v: f64) i64 {
    return @intFromFloat(@trunc(v));
}

/// `np.clip(255. * v, 0, 255).astype(np.uint8)` on an f32 array: clamp, then
/// truncate.
pub fn toByte(v: f32) u8 {
    return @intFromFloat(std.math.clamp(255.0 * v, 0, 255));
}

/// Pillow's `Image.resize((out_w, out_h), LANCZOS)` of an 8-bit `[h][w][c]` image
/// into `out`, `[out_h][out_w][c]`.
pub fn lanczosU8(gpa: std.mem.Allocator, out: []u8, src: []const u8, h: usize, w: usize, c: usize, out_h: usize, out_w: usize) !void {
    std.debug.assert(src.len == h * w * c and out.len == out_h * out_w * c);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    // horizontal pass, skipped (no rounding) when the width is kept
    var mid: []const u8 = src;
    if (out_w != w) {
        const cw = try Coeffs.init(a, w, out_w);
        const tmp = try a.alloc(u8, h * out_w * c);
        for (0..h) |y| for (0..out_w) |x| for (0..c) |ch| {
            const Row = struct {
                s: []const u8,
                c: usize,
                ch: usize,
                fn at(r: @This(), i: usize) u8 {
                    return r.s[i * r.c + r.ch];
                }
            };
            tmp[(y * out_w + x) * c + ch] = cw.apply(x, Row{ .s = src[y * w * c ..][0 .. w * c], .c = c, .ch = ch });
        };
        mid = tmp;
    }
    if (out_h == h) {
        @memcpy(out, mid);
        return;
    }
    const ch_ = try Coeffs.init(a, h, out_h);
    const stride = out_w * c;
    for (0..out_h) |y| for (0..stride) |i| {
        const Col = struct {
            s: []const u8,
            stride: usize,
            i: usize,
            fn at(r: @This(), j: usize) u8 {
                return r.s[j * r.stride + r.i];
            }
        };
        out[y * stride + i] = ch_.apply(y, Col{ .s = mid, .stride = stride, .i = i });
    };
}

/// ComfyUI's Lanczos on a planar RGB `[3][h][w]` image in [0, 1], into planar
/// `[3][out_h][out_w]` in [0, 1]: to bytes, `lanczosU8`, and back by `/ 255`.
pub fn lanczosPlanar(gpa: std.mem.Allocator, out: []f32, src: []const f32, h: usize, w: usize, out_h: usize, out_w: usize) !void {
    std.debug.assert(src.len == 3 * h * w and out.len == 3 * out_h * out_w);
    const in_b = try gpa.alloc(u8, h * w * 3);
    defer gpa.free(in_b);
    for (0..3) |ch| for (0..h * w) |i| {
        in_b[i * 3 + ch] = toByte(src[ch * h * w + i]);
    };
    const out_b = try gpa.alloc(u8, out_h * out_w * 3);
    defer gpa.free(out_b);
    try lanczosU8(gpa, out_b, in_b, h, w, 3, out_h, out_w);
    for (0..3) |ch| for (0..out_h * out_w) |i| {
        out[ch * out_h * out_w + i] = @as(f32, @floatFromInt(out_b[i * 3 + ch])) / 255.0;
    };
}

const fixture = @embedFile("assets/resample_lanczos.bin");

test "lanczos matches Pillow byte for byte" {
    // tools/gen_resample_fixture.py: header of u32 (h, w, out_h, out_w) per case,
    // then the source and Pillow's result, RGB bytes.
    const gpa = std.testing.allocator;
    var rest: []const u8 = fixture;
    var cases: usize = 0;
    while (rest.len > 0) {
        const h = std.mem.readInt(u32, rest[0..4], .little);
        const w = std.mem.readInt(u32, rest[4..8], .little);
        const oh = std.mem.readInt(u32, rest[8..12], .little);
        const ow = std.mem.readInt(u32, rest[12..16], .little);
        rest = rest[16..];
        const src = rest[0 .. h * w * 3];
        const want = rest[h * w * 3 ..][0 .. oh * ow * 3];
        rest = rest[h * w * 3 + oh * ow * 3 ..];
        const got = try gpa.alloc(u8, want.len);
        defer gpa.free(got);
        try lanczosU8(gpa, got, src, h, w, 3, oh, ow);
        errdefer std.debug.print("{d}x{d} -> {d}x{d}\n", .{ w, h, ow, oh });
        try std.testing.expectEqualSlices(u8, want, got);
        cases += 1;
    }
    try std.testing.expect(cases >= 4);
}

test "the byte conversion truncates, as numpy's astype does" {
    // 254/255 in f32 times 255 lands just under 254 and truncates to 253.
    const v: f32 = 254.0 / 255.0;
    try std.testing.expectEqual(@as(u8, @intFromFloat(@trunc(255.0 * v))), toByte(v));
    try std.testing.expectEqual(@as(u8, 255), toByte(1.5));
    try std.testing.expectEqual(@as(u8, 0), toByte(-0.2));
}
