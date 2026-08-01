//! Minimal image output: 8-bit RGB PNG.
//!
//! Matches ComfyUI's default PIL encoding: per-scanline adaptive filtering
//! (min-sum-of-absolute-differences heuristic) feeding a zlib deflate stream at
//! compression level 4, split into 64 KiB IDAT chunks.

const std = @import("std");
const flate = std.compress.flate;

const png_signature = [_]u8{ 0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n' };

/// Bytes per pixel for truecolor RGB. The filters predict from the pixel `bpp`
/// bytes back, so this must match the IHDR color type.
const bpp = 3;

/// Largest IDAT payload, matching PIL/ComfyUI's 64 KiB chunking.
const idat_chunk = 65536;

/// A PNG `tEXt` metadata entry (keyword + Latin-1 text). AUTOMATIC1111 stores
/// generation parameters as a single such chunk with keyword "parameters".
pub const TextChunk = struct { keyword: []const u8, text: []const u8 };

/// Encode [h][w][3] u8 RGB pixels as a PNG into `out`.
pub fn encodePngRgb(gpa: std.mem.Allocator, out: *std.ArrayList(u8), pixels: []const u8, width: usize, height: usize) !void {
    return encodePngRgbText(gpa, out, pixels, width, height, &.{});
}

/// Like `encodePngRgb`, but also embeds `texts` as `tEXt` chunks (placed after
/// IHDR, before the image data, per the PNG spec's ancillary-chunk ordering).
pub fn encodePngRgbText(gpa: std.mem.Allocator, out: *std.ArrayList(u8), pixels: []const u8, width: usize, height: usize, texts: []const TextChunk) !void {
    std.debug.assert(pixels.len == width * height * 3);

    try out.appendSlice(gpa, &png_signature);

    // IHDR
    var ihdr: [13]u8 = undefined;
    std.mem.writeInt(u32, ihdr[0..4], @intCast(width), .big);
    std.mem.writeInt(u32, ihdr[4..8], @intCast(height), .big);
    ihdr[8] = 8; // bit depth
    ihdr[9] = 2; // color type: truecolor RGB
    ihdr[10] = 0; // compression
    ihdr[11] = 0; // filter
    ihdr[12] = 0; // interlace
    try writeChunk(gpa, out, "IHDR", &ihdr);

    // tEXt metadata: [keyword]\0[text], one chunk each.
    for (texts) |t| {
        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(gpa);
        try buf.appendSlice(gpa, t.keyword);
        try buf.append(gpa, 0);
        try buf.appendSlice(gpa, t.text);
        try writeChunk(gpa, out, "tEXt", buf.items);
    }

    // Adaptive-filter each scanline: filter byte + filtered row bytes.
    const stride = width * 3;
    const raw = try gpa.alloc(u8, height * (stride + 1));
    defer gpa.free(raw);
    try filterScanlines(gpa, raw, pixels, width, height);

    // zlib stream: 2-byte header (level 4 → FLEVEL "fast", matching ComfyUI),
    // the raw deflate body, then an Adler-32 of the filtered scanlines.
    const deflated = try deflateRaw(gpa, raw);
    defer gpa.free(deflated);

    var zlib: std.ArrayList(u8) = .empty;
    defer zlib.deinit(gpa);
    try zlib.appendSlice(gpa, &.{ 0x78, 0x5e });
    try zlib.appendSlice(gpa, deflated);
    var adler_buf: [4]u8 = undefined;
    std.mem.writeInt(u32, &adler_buf, std.hash.Adler32.hash(raw), .big);
    try zlib.appendSlice(gpa, &adler_buf);

    // Emit the stream as one or more 64 KiB IDAT chunks.
    var off: usize = 0;
    while (off < zlib.items.len) {
        const n = @min(zlib.items.len - off, idat_chunk);
        try writeChunk(gpa, out, "IDAT", zlib.items[off..][0..n]);
        off += n;
    }

    try writeChunk(gpa, out, "IEND", &.{});
}

/// Deflate `data` into a raw (headerless) stream at compression level 4.
fn deflateRaw(gpa: std.mem.Allocator, data: []const u8) ![]u8 {
    // `Compress.init` requires the output writer to have some buffer capacity.
    var aw: std.Io.Writer.Allocating = try .initCapacity(gpa, 4096);
    errdefer aw.deinit();

    const window = try gpa.alloc(u8, flate.max_window_len);
    defer gpa.free(window);

    var comp = try flate.Compress.init(&aw.writer, window, .raw, .level_4);
    try comp.writer.writeAll(data);
    try comp.finish();

    return aw.toOwnedSlice();
}

/// Write, for each scanline, the best of the five PNG filters (chosen by the
/// min-sum-of-absolute-differences heuristic libpng and PIL use) into `raw`,
/// laid out as [filter_byte][filtered_row] per line.
fn filterScanlines(gpa: std.mem.Allocator, raw: []u8, pixels: []const u8, width: usize, height: usize) !void {
    const stride = width * 3;

    const cand = try gpa.alloc(u8, stride);
    defer gpa.free(cand);
    const best = try gpa.alloc(u8, stride);
    defer gpa.free(best);

    var y: usize = 0;
    while (y < height) : (y += 1) {
        const cur = pixels[y * stride ..][0..stride];
        const prev: ?[]const u8 = if (y == 0) null else pixels[(y - 1) * stride ..][0..stride];

        var best_type: u8 = 0;
        var best_cost: u64 = std.math.maxInt(u64);
        var ft: u8 = 0;
        while (ft <= 4) : (ft += 1) {
            var cost: u64 = 0;
            var x: usize = 0;
            while (x < stride) : (x += 1) {
                const a: u8 = if (x >= bpp) cur[x - bpp] else 0; // Raw(x-bpp)
                const b: u8 = if (prev) |p| p[x] else 0; // Prior(x)
                const c: u8 = if (prev != null and x >= bpp) prev.?[x - bpp] else 0; // Prior(x-bpp)
                const v: u8 = switch (ft) {
                    0 => cur[x],
                    1 => cur[x] -% a,
                    2 => cur[x] -% b,
                    3 => cur[x] -% @as(u8, @intCast((@as(u16, a) + b) / 2)),
                    4 => cur[x] -% paeth(a, b, c),
                    else => unreachable,
                };
                cand[x] = v;
                cost += absSigned(v);
            }
            if (cost < best_cost) {
                best_cost = cost;
                best_type = ft;
                @memcpy(best, cand);
            }
        }

        raw[y * (stride + 1)] = best_type;
        @memcpy(raw[y * (stride + 1) + 1 ..][0..stride], best);
    }
}

/// PNG Paeth predictor over the three neighbouring reconstructed bytes.
fn paeth(a: u8, b: u8, c: u8) u8 {
    const p = @as(i32, a) + @as(i32, b) - @as(i32, c);
    const pa = @abs(p - @as(i32, a));
    const pb = @abs(p - @as(i32, b));
    const pc = @abs(p - @as(i32, c));
    if (pa <= pb and pa <= pc) return a;
    if (pb <= pc) return b;
    return c;
}

/// Absolute value of `v` interpreted as a signed byte — the filter cost metric.
fn absSigned(v: u8) u64 {
    return if (v < 128) v else @as(u64, 256) - v;
}

fn writeChunk(gpa: std.mem.Allocator, out: *std.ArrayList(u8), chunk_type: *const [4]u8, data: []const u8) !void {
    var lenbuf: [4]u8 = undefined;
    std.mem.writeInt(u32, &lenbuf, @intCast(data.len), .big);
    try out.appendSlice(gpa, &lenbuf);
    try out.appendSlice(gpa, chunk_type);
    try out.appendSlice(gpa, data);
    var crc = std.hash.Crc32.init();
    crc.update(chunk_type);
    crc.update(data);
    var crcbuf: [4]u8 = undefined;
    std.mem.writeInt(u32, &crcbuf, crc.final(), .big);
    try out.appendSlice(gpa, &crcbuf);
}

pub const DecodedPng = struct {
    /// Interleaved [h][w][3] u8 RGB.
    pixels: []u8,
    width: usize,
    height: usize,
};

/// Decode an 8-bit non-interlaced truecolor PNG (color type 2 RGB or
/// 6 RGBA — alpha is dropped) to interleaved RGB. Covers what image
/// editors and ComfyUI emit; not a general PNG reader (no palette, no
/// 16-bit, no interlacing). Caller frees `pixels`.
pub fn decodePngRgb(gpa: std.mem.Allocator, data: []const u8) !DecodedPng {
    if (data.len < 8 or !std.mem.eql(u8, data[0..8], "\x89PNG\r\n\x1a\n")) return error.InvalidPng;

    var width: usize = 0;
    var height: usize = 0;
    var channels: usize = 0;
    var idat: std.ArrayList(u8) = .empty;
    defer idat.deinit(gpa);

    var i: usize = 8;
    while (i + 12 <= data.len) {
        const len = std.mem.readInt(u32, data[i..][0..4], .big);
        if (i + 12 + len > data.len) return error.InvalidPng;
        const chunk_type = data[i + 4 ..][0..4];
        const payload = data[i + 8 ..][0..len];
        if (std.mem.eql(u8, chunk_type, "IHDR")) {
            if (len != 13) return error.InvalidPng;
            width = std.mem.readInt(u32, payload[0..4], .big);
            height = std.mem.readInt(u32, payload[4..8], .big);
            const bit_depth = payload[8];
            const color_type = payload[9];
            const interlace = payload[12];
            if (bit_depth != 8 or interlace != 0) return error.UnsupportedPng;
            channels = switch (color_type) {
                2 => 3, // truecolor
                6 => 4, // truecolor + alpha
                else => return error.UnsupportedPng,
            };
        } else if (std.mem.eql(u8, chunk_type, "IDAT")) {
            try idat.appendSlice(gpa, payload);
        } else if (std.mem.eql(u8, chunk_type, "IEND")) {
            break;
        }
        i += 12 + len;
    }
    if (width == 0 or height == 0 or channels == 0 or idat.items.len == 0) return error.InvalidPng;

    // Inflate the zlib stream to filtered scanlines.
    var in_reader: std.Io.Reader = .fixed(idat.items);
    const dbuf = try gpa.alloc(u8, flate.max_window_len);
    defer gpa.free(dbuf);
    var dc = flate.Decompress.init(&in_reader, .zlib, dbuf);
    const raw = try dc.reader.allocRemaining(gpa, .unlimited);
    defer gpa.free(raw);

    const stride = width * channels;
    if (raw.len != height * (stride + 1)) return error.InvalidPng;

    // Reverse the per-row filters in place over a reconstruction buffer.
    const recon = try gpa.alloc(u8, height * stride);
    defer gpa.free(recon);
    for (0..height) |y| {
        const ft = raw[y * (stride + 1)];
        const row = raw[y * (stride + 1) + 1 ..][0..stride];
        for (0..stride) |x| {
            const a: u8 = if (x >= channels) recon[y * stride + x - channels] else 0;
            const b: u8 = if (y > 0) recon[(y - 1) * stride + x] else 0;
            const c: u8 = if (y > 0 and x >= channels) recon[(y - 1) * stride + x - channels] else 0;
            const pred: u8 = switch (ft) {
                0 => 0,
                1 => a,
                2 => b,
                3 => @intCast((@as(u16, a) + b) / 2),
                4 => paeth(a, b, c),
                else => return error.InvalidPng,
            };
            recon[y * stride + x] = row[x] +% pred;
        }
    }

    // Drop alpha if present.
    const pixels = try gpa.alloc(u8, width * height * 3);
    errdefer gpa.free(pixels);
    if (channels == 3) {
        @memcpy(pixels, recon);
    } else {
        for (0..width * height) |p| {
            @memcpy(pixels[p * 3 ..][0..3], recon[p * 4 ..][0..3]);
        }
    }
    return .{ .pixels = pixels, .width = width, .height = height };
}

/// Convert decoder output in [-1, 1] (planar [3][h][w], torch layout) to
/// interleaved [h][w][3] u8, matching ComfyUI's (x/2 + 0.5).clamp(0,1) * 255.
pub fn planarF32ToRgb8(gpa: std.mem.Allocator, planar: []const f32, width: usize, height: usize) ![]u8 {
    std.debug.assert(planar.len == 3 * width * height);
    const px = try gpa.alloc(u8, width * height * 3);
    const plane = width * height;
    for (0..plane) |i| {
        for (0..3) |c| {
            const v = std.math.clamp(planar[c * plane + i] * 0.5 + 0.5, 0.0, 1.0);
            px[i * 3 + c] = @intFromFloat(@round(v * 255.0));
        }
    }
    return px;
}

test "png structure is valid" {
    const gpa = std.testing.allocator;
    // 2x2: red, green, blue, white.
    const pixels = [_]u8{ 255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255 };
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try encodePngRgb(gpa, &out, &pixels, 2, 2);

    const bytes = out.items;
    try std.testing.expectEqualSlices(u8, &png_signature, bytes[0..8]);

    // Walk chunks: length + type + data + crc, verifying CRCs. Expect
    // IHDR, one IDAT (tiny image), IEND in order.
    var i: usize = 8;
    var seen_iend = false;
    var n_chunks: usize = 0;
    var idat: std.ArrayList(u8) = .empty;
    defer idat.deinit(gpa);
    while (i < bytes.len) {
        const len = std.mem.readInt(u32, bytes[i..][0..4], .big);
        const chunk_type = bytes[i + 4 ..][0..4];
        const data = bytes[i + 8 ..][0..len];
        const stored_crc = std.mem.readInt(u32, bytes[i + 8 + len ..][0..4], .big);
        var crc = std.hash.Crc32.init();
        crc.update(chunk_type);
        crc.update(data);
        try std.testing.expectEqual(crc.final(), stored_crc);
        if (std.mem.eql(u8, chunk_type, "IDAT")) try idat.appendSlice(gpa, data);
        n_chunks += 1;
        if (std.mem.eql(u8, chunk_type, "IEND")) seen_iend = true;
        i += 12 + len;
    }
    try std.testing.expectEqual(@as(usize, 3), n_chunks);
    try std.testing.expect(seen_iend);
    try std.testing.expectEqual(i, bytes.len);

    // zlib header advertises deflate + FLEVEL "fast" (level 4), like ComfyUI.
    try std.testing.expectEqual(@as(u8, 0x78), idat.items[0]);
    try std.testing.expectEqual(@as(u8, 0x5e), idat.items[1]);
}

test "tEXt metadata chunk is embedded after IHDR" {
    const gpa = std.testing.allocator;
    const pixels = [_]u8{ 255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255 };
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    const params = "a cat\nSteps: 20, Seed: 42";
    try encodePngRgbText(gpa, &out, &pixels, 2, 2, &.{
        .{ .keyword = "parameters", .text = params },
    });

    const bytes = out.items;
    // Walk chunks, find the tEXt chunk, and confirm its order + payload.
    var i: usize = 8;
    var order: std.ArrayList([]const u8) = .empty;
    defer order.deinit(gpa);
    var text_payload: ?[]const u8 = null;
    while (i < bytes.len) {
        const len = std.mem.readInt(u32, bytes[i..][0..4], .big);
        const chunk_type = bytes[i + 4 ..][0..4];
        const data = bytes[i + 8 ..][0..len];
        try order.append(gpa, chunk_type);
        if (std.mem.eql(u8, chunk_type, "tEXt")) text_payload = data;
        i += 12 + len;
    }
    // Expected chunk order: IHDR, tEXt, IDAT…, IEND.
    try std.testing.expectEqualStrings("IHDR", order.items[0]);
    try std.testing.expectEqualStrings("tEXt", order.items[1]);
    try std.testing.expectEqualStrings("IDAT", order.items[2]);

    // Payload is "parameters\0<params>".
    const p = text_payload.?;
    const nul = std.mem.indexOfScalar(u8, p, 0).?;
    try std.testing.expectEqualStrings("parameters", p[0..nul]);
    try std.testing.expectEqualStrings(params, p[nul + 1 ..]);
}

test "png round-trips through deflate and filtering" {
    const gpa = std.testing.allocator;
    const width = 7;
    const height = 5;

    // A gradient so adjacent pixels differ — exercises the filter heuristic.
    var pixels: [width * height * 3]u8 = undefined;
    for (0..height) |y| {
        for (0..width) |x| {
            const i = (y * width + x) * 3;
            pixels[i + 0] = @intCast((x * 30) & 0xff);
            pixels[i + 1] = @intCast((y * 50) & 0xff);
            pixels[i + 2] = @intCast((x * y * 7) & 0xff);
        }
    }

    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try encodePngRgb(gpa, &out, &pixels, width, height);

    // Collect the IDAT payload.
    var idat: std.ArrayList(u8) = .empty;
    defer idat.deinit(gpa);
    var i: usize = 8;
    while (i < out.items.len) {
        const len = std.mem.readInt(u32, out.items[i..][0..4], .big);
        const chunk_type = out.items[i + 4 ..][0..4];
        if (std.mem.eql(u8, chunk_type, "IDAT"))
            try idat.appendSlice(gpa, out.items[i + 8 ..][0..len]);
        i += 12 + len;
    }

    // Inflate the zlib stream back to filtered scanlines.
    var in_reader: std.Io.Reader = .fixed(idat.items);
    const dbuf = try gpa.alloc(u8, flate.max_window_len);
    defer gpa.free(dbuf);
    var dc = flate.Decompress.init(&in_reader, .zlib, dbuf);
    const raw = try dc.reader.allocRemaining(gpa, .unlimited);
    defer gpa.free(raw);

    const stride = width * 3;
    try std.testing.expectEqual(height * (stride + 1), raw.len);

    // Reverse the per-row filters and compare to the original pixels.
    const recon = try gpa.alloc(u8, height * stride);
    defer gpa.free(recon);
    for (0..height) |y| {
        const ft = raw[y * (stride + 1)];
        const row = raw[y * (stride + 1) + 1 ..][0..stride];
        for (0..stride) |x| {
            const a: u8 = if (x >= bpp) recon[y * stride + x - bpp] else 0;
            const b: u8 = if (y > 0) recon[(y - 1) * stride + x] else 0;
            const c: u8 = if (y > 0 and x >= bpp) recon[(y - 1) * stride + x - bpp] else 0;
            const pred: u8 = switch (ft) {
                0 => 0,
                1 => a,
                2 => b,
                3 => @intCast((@as(u16, a) + b) / 2),
                4 => paeth(a, b, c),
                else => unreachable,
            };
            recon[y * stride + x] = row[x] +% pred;
        }
    }
    try std.testing.expectEqualSlices(u8, &pixels, recon);
}

test "png decode round-trips encode" {
    const gpa = std.testing.allocator;
    const width = 13;
    const height = 9;
    var pixels: [width * height * 3]u8 = undefined;
    var prng = std.Random.DefaultPrng.init(5);
    prng.random().bytes(&pixels); // random pixels exercise all filter types

    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try encodePngRgb(gpa, &out, &pixels, width, height);

    const dec = try decodePngRgb(gpa, out.items);
    defer gpa.free(dec.pixels);
    try std.testing.expectEqual(width, dec.width);
    try std.testing.expectEqual(height, dec.height);
    try std.testing.expectEqualSlices(u8, &pixels, dec.pixels);
}

test "png decode rejects garbage" {
    const gpa = std.testing.allocator;
    try std.testing.expectError(error.InvalidPng, decodePngRgb(gpa, "not a png at all"));
}

test "planar conversion clamps and scales" {
    const gpa = std.testing.allocator;
    // 1x2 image, planar [3][1][2].
    const planar = [_]f32{ -1.0, 1.0, 0.0, 2.0, -3.0, 0.5 };
    const px = try planarF32ToRgb8(gpa, &planar, 2, 1);
    defer gpa.free(px);
    try std.testing.expectEqualSlices(u8, &.{ 0, 128, 0, 255, 255, 191 }, px);
}

// ---------------------------------------------------------------------------
// tEXt metadata, reading
// ---------------------------------------------------------------------------

/// Read every `tEXt` chunk out of a PNG — the counterpart to
/// `encodePngRgbText`. AUTOMATIC1111, ComfyUI and this project all record
/// generation parameters this way, so a consumer comparing generated images can
/// recover the seed, sampler and model that produced each one instead of being
/// told them out of band.
///
/// `zTXt`/`iTXt` (compressed and UTF-8 variants) are deliberately not handled:
/// nothing here writes them, and silently returning fewer chunks than a file
/// contains would be worse than a caller knowing only `tEXt` is covered.
/// Keywords and text point into `data`; the slice itself is owned by the caller.
pub fn readTextChunks(gpa: std.mem.Allocator, data: []const u8) ![]TextChunk {
    if (data.len < 8 or !std.mem.eql(u8, data[0..8], &png_signature)) return error.InvalidPng;

    var out: std.ArrayList(TextChunk) = .empty;
    errdefer out.deinit(gpa);

    var pos: usize = 8;
    while (pos + 8 <= data.len) {
        const len = std.mem.readInt(u32, data[pos..][0..4], .big);
        const kind = data[pos + 4 ..][0..4];
        const body_start = pos + 8;
        // +4 for the trailing CRC.
        const next = std.math.add(usize, body_start, @as(usize, len) + 4) catch return error.InvalidPng;
        if (next > data.len) return error.InvalidPng;

        if (std.mem.eql(u8, kind, "tEXt")) {
            const body = data[body_start .. body_start + len];
            // keyword \0 text — a chunk without the separator is malformed.
            const nul = std.mem.indexOfScalar(u8, body, 0) orelse return error.InvalidPng;
            try out.append(gpa, .{ .keyword = body[0..nul], .text = body[nul + 1 ..] });
        } else if (std.mem.eql(u8, kind, "IEND")) {
            break;
        }
        pos = next;
    }
    return out.toOwnedSlice(gpa);
}

/// The text of the first `tEXt` chunk with this keyword, or null.
pub fn findTextChunk(chunks: []const TextChunk, keyword: []const u8) ?[]const u8 {
    for (chunks) |c| if (std.mem.eql(u8, c.keyword, keyword)) return c.text;
    return null;
}

// ---------------------------------------------------------------------------
// Comparison metrics
// ---------------------------------------------------------------------------

/// Mean squared error between two equally sized 8-bit images, in [0, 65025].
pub fn mse(a: []const u8, b: []const u8) !f64 {
    if (a.len != b.len) return error.SizeMismatch;
    if (a.len == 0) return error.EmptyImage;
    var acc: f64 = 0;
    for (a, b) |x, y| {
        const d = @as(f64, @floatFromInt(@as(i32, x) - @as(i32, y)));
        acc += d * d;
    }
    return acc / @as(f64, @floatFromInt(a.len));
}

/// Peak signal-to-noise ratio in dB against an 8-bit peak of 255.
///
/// Unambiguous as metrics go — unlike SSIM (`ssim` below), whose conventions had
/// to be pinned against scikit-image before it could be trusted. Identical
/// images return infinity rather than a large sentinel; a caller reporting a
/// table should special-case it rather than print `inf` and let a reader wonder.
pub fn psnr(a: []const u8, b: []const u8) !f64 {
    const m = try mse(a, b);
    if (m == 0) return std.math.inf(f64);
    return 10.0 * std.math.log10(255.0 * 255.0 / m);
}

/// RMS of the horizontal and vertical first differences of luma — a crude
/// measure of how much fine detail an image carries at all.
///
/// **Not a standard metric, and not a similarity metric.** It exists because
/// PSNR between two generated images is confounded when their composition
/// drifts: a quantization that keeps the subject but loses texture and one that
/// draws a different picture both score badly, and the two failures call for
/// different responses. This number is per-image, so comparing a candidate's
/// against the reference's says whether detail survived independently of whether
/// the image matched.
pub fn detailEnergy(pixels: []const u8, width: usize, height: usize) !f64 {
    if (width == 0 or height == 0) return error.EmptyImage;
    if (pixels.len != width * height * 3) return error.SizeMismatch;

    var acc: f64 = 0;
    var n: usize = 0;
    for (0..height) |y| {
        for (0..width) |x| {
            const i = (y * width + x) * 3;
            const l = luma(pixels[i], pixels[i + 1], pixels[i + 2]);
            if (x + 1 < width) {
                const r = luma(pixels[i + 3], pixels[i + 4], pixels[i + 5]);
                acc += (l - r) * (l - r);
                n += 1;
            }
            if (y + 1 < height) {
                const j = i + width * 3;
                const d = luma(pixels[j], pixels[j + 1], pixels[j + 2]);
                acc += (l - d) * (l - d);
                n += 1;
            }
        }
    }
    return if (n == 0) 0 else @sqrt(acc / @as(f64, @floatFromInt(n)));
}

/// Which window SSIM averages over. The two are different metrics that both go
/// by "SSIM", so the choice is explicit rather than defaulted silently.
pub const SsimWindow = enum {
    /// 7x7 uniform, sample covariance (`N/(N-1)`). scikit-image's default, and
    /// what most reported "SSIM" numbers are.
    uniform7,
    /// 11x11 gaussian, sigma 1.5, population covariance. What Wang et al. (2004)
    /// specify, and what `skimage` gives with
    /// `gaussian_weights=True, sigma=1.5, use_sample_covariance=False`.
    gaussian11,

    pub fn size(self: SsimWindow) usize {
        return switch (self) {
            .uniform7 => 7,
            .gaussian11 => 11,
        };
    }
};

/// Mean structural similarity between two equally sized 8-bit RGB images, in
/// [-1, 1]; 1 for identical input.
///
/// Computed per channel over `window`, then averaged over the three channels —
/// scikit-image's `channel_axis=-1` convention, not luma. Only fully-interior
/// windows contribute (the reference crops the border by the window radius),
/// which is also why no boundary-extension mode appears here: cropped-away
/// pixels are the only ones a mode would have affected.
///
/// Validated against `skimage.metrics.structural_similarity` on the fixtures in
/// `assets/image_metric_fixtures.json`, both windows. `error.ImageTooSmall` when
/// either side is shorter than the window.
pub fn ssim(a: []const u8, b: []const u8, width: usize, height: usize, window: SsimWindow) !f64 {
    if (a.len != b.len) return error.SizeMismatch;
    if (a.len != width * height * 3) return error.SizeMismatch;
    const win = window.size();
    if (width < win or height < win) return error.ImageTooSmall;

    // Separable weights, normalized so they sum to 1 in each axis; the 2D window
    // is their outer product, so a 2D sum decomposes into two 1D passes.
    var w1: [11]f64 = undefined;
    const weights = w1[0..win];
    switch (window) {
        .uniform7 => for (weights) |*v| {
            v.* = 1.0 / @as(f64, @floatFromInt(win));
        },
        .gaussian11 => {
            const sigma = 1.5;
            const radius = @as(f64, @floatFromInt(win / 2));
            var sum: f64 = 0;
            for (weights, 0..) |*v, i| {
                const d = @as(f64, @floatFromInt(i)) - radius;
                v.* = @exp(-(d * d) / (2 * sigma * sigma));
                sum += v.*;
            }
            for (weights) |*v| v.* /= sum;
        },
    }
    // Sample covariance divides by N-1 over the N window taps; the gaussian
    // convention here is the population form, so 1.
    const cov_norm: f64 = switch (window) {
        .uniform7 => blk: {
            const np = @as(f64, @floatFromInt(win * win));
            break :blk np / (np - 1);
        },
        .gaussian11 => 1,
    };

    const data_range: f64 = 255;
    const c1 = (0.01 * data_range) * (0.01 * data_range);
    const c2 = (0.03 * data_range) * (0.03 * data_range);

    const inner_h = height - win + 1;
    const inner_w = width - win + 1;
    var acc: f64 = 0;
    for (0..3) |ch| {
        for (0..inner_h) |y0| {
            for (0..inner_w) |x0| {
                var ux: f64 = 0;
                var uy: f64 = 0;
                var uxx: f64 = 0;
                var uyy: f64 = 0;
                var uxy: f64 = 0;
                for (0..win) |dy| {
                    for (0..win) |dx| {
                        const wgt = weights[dy] * weights[dx];
                        const i = ((y0 + dy) * width + x0 + dx) * 3 + ch;
                        const x: f64 = @floatFromInt(a[i]);
                        const y: f64 = @floatFromInt(b[i]);
                        ux += wgt * x;
                        uy += wgt * y;
                        uxx += wgt * x * x;
                        uyy += wgt * y * y;
                        uxy += wgt * x * y;
                    }
                }
                const vx = cov_norm * (uxx - ux * ux);
                const vy = cov_norm * (uyy - uy * uy);
                const vxy = cov_norm * (uxy - ux * uy);
                const num = (2 * ux * uy + c1) * (2 * vxy + c2);
                const den = (ux * ux + uy * uy + c1) * (vx + vy + c2);
                acc += num / den;
            }
        }
    }
    return acc / @as(f64, @floatFromInt(3 * inner_h * inner_w));
}

/// Rec. 601 luma, the convention PSNR-on-luma implementations use.
fn luma(r: u8, g: u8, b: u8) f64 {
    return 0.299 * @as(f64, @floatFromInt(r)) +
        0.587 * @as(f64, @floatFromInt(g)) +
        0.114 * @as(f64, @floatFromInt(b));
}

test "tEXt chunks read back exactly what the encoder wrote" {
    // The encoder and the reader are counterparts, so round-tripping them is the
    // check that matters: a reader validated only against hand-built bytes could
    // drift from what we actually emit.
    const gpa = std.testing.allocator;
    const width = 4;
    const height = 3;
    var pixels: [width * height * 3]u8 = undefined;
    var prng = std.Random.DefaultPrng.init(11);
    prng.random().bytes(&pixels);

    const texts = [_]TextChunk{
        .{ .keyword = "parameters", .text = "a prompt\nSteps: 16, Seed: 80085, Model: anim-int4-calib" },
        .{ .keyword = "Software", .text = "ggufy" },
    };

    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try encodePngRgbText(gpa, &out, &pixels, width, height, &texts);

    const got = try readTextChunks(gpa, out.items);
    defer gpa.free(got);
    try std.testing.expectEqual(@as(usize, 2), got.len);
    for (got, texts) |g, want| {
        try std.testing.expectEqualStrings(want.keyword, g.keyword);
        try std.testing.expectEqualStrings(want.text, g.text);
    }

    try std.testing.expectEqualStrings(texts[0].text, findTextChunk(got, "parameters").?);
    try std.testing.expect(findTextChunk(got, "absent") == null);

    // A PNG with no text chunks yields an empty slice, not an error.
    var bare: std.ArrayList(u8) = .empty;
    defer bare.deinit(gpa);
    try encodePngRgb(gpa, &bare, &pixels, width, height);
    const none = try readTextChunks(gpa, bare.items);
    defer gpa.free(none);
    try std.testing.expectEqual(@as(usize, 0), none.len);

    try std.testing.expectError(error.InvalidPng, readTextChunks(gpa, "not a png at all"));
}

test "psnr matches its definition on known inputs" {
    const gpa = std.testing.allocator;
    _ = gpa;
    // Identical images: infinite, not a large sentinel a caller might print raw.
    const a = [_]u8{ 10, 20, 30, 40 };
    try std.testing.expect(std.math.isInf(try psnr(&a, &a)));
    try std.testing.expectEqual(@as(f64, 0), try mse(&a, &a));

    // A uniform error of 1 per sample gives MSE 1 and PSNR 20*log10(255).
    const b = [_]u8{ 11, 21, 31, 41 };
    try std.testing.expectApproxEqAbs(@as(f64, 1), try mse(&a, &b), 1e-12);
    try std.testing.expectApproxEqAbs(20.0 * std.math.log10(255.0), try psnr(&a, &b), 1e-9);

    // Maximum possible error: 0 dB, the floor of the scale.
    const lo = [_]u8{ 0, 0 };
    const hi = [_]u8{ 255, 255 };
    try std.testing.expectApproxEqAbs(@as(f64, 0), try psnr(&lo, &hi), 1e-9);

    try std.testing.expectError(error.SizeMismatch, psnr(&a, &lo));
    try std.testing.expectError(error.EmptyImage, psnr(&.{}, &.{}));
}

test "detail energy separates a flat image from a textured one" {
    // The property it exists for: it must respond to fine detail, not to overall
    // brightness or contrast between two images.
    const gpa = std.testing.allocator;
    _ = gpa;
    const w = 8;
    const h = 8;
    var flat: [w * h * 3]u8 = undefined;
    @memset(&flat, 128);
    try std.testing.expectApproxEqAbs(@as(f64, 0), try detailEnergy(&flat, w, h), 1e-12);

    // A one-pixel checkerboard is maximum detail at this resolution.
    var checker: [w * h * 3]u8 = undefined;
    for (0..h) |y| for (0..w) |x| {
        const v: u8 = if ((x + y) % 2 == 0) 0 else 255;
        const i = (y * w + x) * 3;
        checker[i] = v;
        checker[i + 1] = v;
        checker[i + 2] = v;
    };
    const detailed = try detailEnergy(&checker, w, h);
    try std.testing.expect(detailed > 250.0);

    // A smooth ramp carries some detail, but far less than the checkerboard.
    var ramp: [w * h * 3]u8 = undefined;
    for (0..h) |y| for (0..w) |x| {
        const v: u8 = @intCast(x * 30);
        const i = (y * w + x) * 3;
        ramp[i] = v;
        ramp[i + 1] = v;
        ramp[i + 2] = v;
    };
    const mid = try detailEnergy(&ramp, w, h);
    try std.testing.expect(mid > 0 and mid < detailed / 5.0);

    try std.testing.expectError(error.SizeMismatch, detailEnergy(&flat, w, h + 1));
}

const metric_fixtures_json = @embedFile("assets/image_metric_fixtures.json");

test "mse, psnr and ssim match numpy and scikit-image on the reference pairs" {
    const gpa = std.testing.allocator;

    const Pair = struct {
        name: []const u8,
        width: usize,
        height: usize,
        a: []const u8,
        b: []const u8,
        mse: f64,
        /// null means +inf: identical inputs. JSON has no infinity.
        psnr: ?f64,
        ssim: f64,
        ssim_gaussian: f64,
    };
    var parsed = try std.json.parseFromSlice(
        struct { pairs: []const Pair },
        gpa,
        metric_fixtures_json,
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();
    try std.testing.expect(parsed.value.pairs.len >= 5);

    for (parsed.value.pairs) |p| {
        try std.testing.expectEqual(p.width * p.height * 3, p.a.len);

        try std.testing.expectApproxEqAbs(p.mse, try mse(p.a, p.b), 1e-9);
        const got_psnr = try psnr(p.a, p.b);
        if (p.psnr) |want| {
            try std.testing.expectApproxEqAbs(want, got_psnr, 1e-9);
        } else {
            try std.testing.expect(std.math.isInf(got_psnr));
        }

        // 1e-9 rather than a loose tolerance: both sides accumulate in f64 over
        // the same window, so anything larger would mean a convention mismatch
        // (window, covariance normalization, data range) rather than rounding.
        inline for (.{ .{ SsimWindow.uniform7, "ssim" }, .{ SsimWindow.gaussian11, "ssim_gaussian" } }) |case| {
            const want = @field(p, case[1]);
            const got = try ssim(p.a, p.b, p.width, p.height, case[0]);
            std.testing.expectApproxEqAbs(want, got, 1e-9) catch |e| {
                std.debug.print("{s} {s}: skimage {d} got {d}\n", .{ p.name, case[1], want, got });
                return e;
            };
        }
    }
}

test "ssim of an image against itself is exactly 1, and rejects a too-small image" {
    const gpa = std.testing.allocator;
    const w = 12;
    const h = 12;
    const px = try gpa.alloc(u8, w * h * 3);
    defer gpa.free(px);
    var prng = std.Random.DefaultPrng.init(3);
    prng.random().bytes(px);

    try std.testing.expectApproxEqAbs(1.0, try ssim(px, px, w, h, .uniform7), 1e-12);
    try std.testing.expectApproxEqAbs(1.0, try ssim(px, px, w, h, .gaussian11), 1e-12);
    // 11x11 window does not fit in a 10-row image.
    try std.testing.expectError(error.ImageTooSmall, ssim(px[0 .. 10 * w * 3], px[0 .. 10 * w * 3], w, 10, .gaussian11));
}
