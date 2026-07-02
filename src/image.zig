//! Minimal image output: 8-bit RGB PNG.
//!
//! The zlib stream inside IDAT uses stored (uncompressed) deflate blocks, so
//! no compressor is needed — files are ~3 bytes/pixel but always valid. A real
//! deflate can be swapped in later without changing callers.

const std = @import("std");

const png_signature = [_]u8{ 0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n' };

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

    // Raw scanlines: filter byte 0 + row bytes.
    const stride = width * 3;
    const raw = try gpa.alloc(u8, height * (stride + 1));
    defer gpa.free(raw);
    for (0..height) |y| {
        raw[y * (stride + 1)] = 0;
        @memcpy(raw[y * (stride + 1) + 1 ..][0..stride], pixels[y * stride ..][0..stride]);
    }

    // zlib wrapper with stored deflate blocks.
    var idat: std.ArrayList(u8) = .empty;
    defer idat.deinit(gpa);
    try idat.appendSlice(gpa, &.{ 0x78, 0x01 }); // CMF/FLG (32K window, no preset dict)
    var off: usize = 0;
    while (off < raw.len) {
        const n: u16 = @intCast(@min(raw.len - off, 65535));
        const final: u8 = if (off + n == raw.len) 1 else 0;
        try idat.append(gpa, final); // BFINAL + BTYPE=00 (stored)
        var lenbuf: [4]u8 = undefined;
        std.mem.writeInt(u16, lenbuf[0..2], n, .little);
        std.mem.writeInt(u16, lenbuf[2..4], ~n, .little);
        try idat.appendSlice(gpa, &lenbuf);
        try idat.appendSlice(gpa, raw[off..][0..n]);
        off += n;
    }
    var adler_buf: [4]u8 = undefined;
    std.mem.writeInt(u32, &adler_buf, std.hash.Adler32.hash(raw), .big);
    try idat.appendSlice(gpa, &adler_buf);
    try writeChunk(gpa, out, "IDAT", idat.items);

    try writeChunk(gpa, out, "IEND", &.{});
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

    // Walk chunks: length + type + data + crc, verifying CRCs.
    var i: usize = 8;
    var seen_iend = false;
    var chunk_names: [3][]const u8 = undefined;
    var n_chunks: usize = 0;
    while (i < bytes.len) {
        const len = std.mem.readInt(u32, bytes[i..][0..4], .big);
        const chunk_type = bytes[i + 4 ..][0..4];
        const data = bytes[i + 8 ..][0..len];
        const stored_crc = std.mem.readInt(u32, bytes[i + 8 + len ..][0..4], .big);
        var crc = std.hash.Crc32.init();
        crc.update(chunk_type);
        crc.update(data);
        try std.testing.expectEqual(crc.final(), stored_crc);
        chunk_names[n_chunks] = chunk_type;
        n_chunks += 1;
        if (std.mem.eql(u8, chunk_type, "IEND")) seen_iend = true;
        i += 12 + len;
    }
    try std.testing.expectEqual(@as(usize, 3), n_chunks);
    try std.testing.expect(seen_iend);
    try std.testing.expectEqual(i, bytes.len);

    // Decode the stored-deflate zlib stream back and compare scanlines.
    const ihdr_len = std.mem.readInt(u32, bytes[8..][0..4], .big);
    const idat_start = 8 + 12 + ihdr_len;
    const idat_len = std.mem.readInt(u32, bytes[idat_start..][0..4], .big);
    const z = bytes[idat_start + 8 ..][0 .. idat_len];
    // Skip 2-byte zlib header; single stored block expected for this size.
    try std.testing.expectEqual(@as(u8, 1), z[2]); // BFINAL, stored
    const n = std.mem.readInt(u16, z[3..5], .little);
    try std.testing.expectEqual(@as(u16, 14), n); // 2 rows * (1 + 6)
    const raw = z[7 .. 7 + n];
    try std.testing.expectEqual(@as(u8, 0), raw[0]); // filter byte
    try std.testing.expectEqualSlices(u8, pixels[0..6], raw[1..7]);
    try std.testing.expectEqualSlices(u8, pixels[6..12], raw[8..14]);
}

test "planar conversion clamps and scales" {
    const gpa = std.testing.allocator;
    // 1x2 image, planar [3][1][2].
    const planar = [_]f32{ -1.0, 1.0, 0.0, 2.0, -3.0, 0.5 };
    const px = try planarF32ToRgb8(gpa, &planar, 2, 1);
    defer gpa.free(px);
    try std.testing.expectEqualSlices(u8, &.{ 0, 128, 0, 255, 255, 191 }, px);
}
