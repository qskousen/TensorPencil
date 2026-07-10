//! Thin Zig wrapper over lib/vips/vips_helper.c (system libvips) — image
//! DECODE for tp-llm's --image flag and @path mentions, so chat input takes
//! any common format (jpeg incl. progressive + EXIF rotation, png, webp,
//! gif, tiff, ...) instead of PNG only.
//!
//! Deliberately a module of the tp-llm EXECUTABLE, not of the TensorPencil
//! library: the engine stays pure Zig (its own PNG encode/decode in
//! image.zig is untouched); this is a CLI input-format convenience, the one
//! sanctioned C linkage besides the dlopen'd GPU drivers. Ported from
//! DiffKeep's lib/vips.

const std = @import("std");

const c = @cImport({
    @cInclude("vips_helper.h");
});

var initialized = false;

/// Lazy one-time process init (first decode pays it; text-only sessions
/// never touch libvips).
fn ensureInit() !void {
    if (initialized) return;
    if (c.tp_vips_init("tp-llm") != 0) return error.VipsInitFailed;
    initialized = true;
}

pub const Decoded = struct {
    /// Packed interleaved RGB8, gpa-owned.
    pixels: []u8,
    width: usize,
    height: usize,
};

/// Decode an image file to packed RGB8 at native resolution
/// (EXIF-autorotated, alpha flattened over white).
pub fn loadRgb(gpa: std.mem.Allocator, path: []const u8) !Decoded {
    try ensureInit();
    const pathz = try gpa.dupeZ(u8, path);
    defer gpa.free(pathz);

    var buf: ?*anyopaque = null;
    var len: usize = 0;
    var w: c_int = 0;
    var h: c_int = 0;
    if (c.tp_load_image_rgb(pathz.ptr, &buf, &len, &w, &h) != 0)
        return error.ImageDecodeFailed;
    defer c.tp_vips_free(buf);

    return .{
        .pixels = try gpa.dupe(u8, @as([*]const u8, @ptrCast(buf.?))[0..len]),
        .width = @intCast(w),
        .height = @intCast(h),
    };
}
