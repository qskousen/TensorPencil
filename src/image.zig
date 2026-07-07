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

/// Encode [h][w][3] u8 RGB pixels as a PNG into `out`.
pub fn encodePngRgb(gpa: std.mem.Allocator, out: *std.ArrayList(u8), pixels: []const u8, width: usize, height: usize) !void {
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

test "planar conversion clamps and scales" {
    const gpa = std.testing.allocator;
    // 1x2 image, planar [3][1][2].
    const planar = [_]f32{ -1.0, 1.0, 0.0, 2.0, -3.0, 0.5 };
    const px = try planarF32ToRgb8(gpa, &planar, 2, 1);
    defer gpa.free(px);
    try std.testing.expectEqualSlices(u8, &.{ 0, 128, 0, 255, 255, 191 }, px);
}
