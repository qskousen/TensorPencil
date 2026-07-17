//! Copy a generated image to the OS clipboard as a PNG.
//!
//! SDL3's clipboard-set API is pull-based: instead of handing over bytes, you
//! register a provider callback that SDL invokes lazily when another app asks
//! for the data, plus a cleanup callback fired when the clipboard is replaced
//! or the app exits. The encoded PNG therefore has to outlive `copyImage` — we
//! hand SDL a process-allocator-owned buffer and free it in `cleanup`.
const std = @import("std");
const SDLBackend = @import("backend");
const tp = @import("TensorPencil");
const diffuser = @import("diffuser.zig");

const GenImage = diffuser.GenImage;
const SDL = SDLBackend.c;

// SDL owns the clipboard payload until the clipboard changes — well past the
// scope of `copyImage` — so it can't live on a transient/frame allocator.
const gpa = std.heap.smp_allocator;

const png_mime = "image/png";

/// What SDL holds onto between the copy and the eventual paste/cleanup.
const Payload = struct { bytes: []u8 };

/// Encode `gi`'s pixels to PNG and place them on the OS clipboard. No-op if the
/// image hasn't finished rendering (no `rgba` yet). Logs and bails on failure —
/// a failed copy should never take down the viewer.
pub fn copyImage(gi: *GenImage) void {
    const rgba = gi.rgba orelse return;
    const w = gi.width;
    const h = gi.height;
    if (w == 0 or h == 0 or rgba.len < w * h * 4) return;

    // encodePngRgb wants tightly packed RGB; drop the (always-opaque) alpha.
    const rgb = gpa.alloc(u8, w * h * 3) catch return;
    defer gpa.free(rgb);
    var i: usize = 0;
    while (i < w * h) : (i += 1) {
        rgb[i * 3 + 0] = rgba[i * 4 + 0];
        rgb[i * 3 + 1] = rgba[i * 4 + 1];
        rgb[i * 3 + 2] = rgba[i * 4 + 2];
    }

    var png: std.ArrayList(u8) = .empty;
    defer png.deinit(gpa);
    tp.image.encodePngRgb(gpa, &png, rgb, w, h) catch |err| {
        std.log.err("copy image (encode): {t}", .{err});
        return;
    };

    // Give SDL an owned copy it can serve lazily and free on cleanup.
    const payload = gpa.create(Payload) catch return;
    payload.bytes = gpa.dupe(u8, png.items) catch {
        gpa.destroy(payload);
        return;
    };

    var mimes = [_][*c]const u8{png_mime};
    if (!SDL.SDL_SetClipboardData(provide, cleanup, payload, &mimes, mimes.len)) {
        std.log.err("copy image: SDL_SetClipboardData failed: {s}", .{SDL.SDL_GetError()});
        cleanup(payload); // SDL won't call cleanup itself when the set fails.
    }
}

fn provide(userdata: ?*anyopaque, mime_type: [*c]const u8, size: [*c]usize) callconv(.c) ?*const anyopaque {
    _ = mime_type; // only one type was registered
    const payload: *Payload = @ptrCast(@alignCast(userdata.?));
    size.* = payload.bytes.len;
    return payload.bytes.ptr;
}

fn cleanup(userdata: ?*anyopaque) callconv(.c) void {
    const payload: *Payload = @ptrCast(@alignCast(userdata.?));
    gpa.free(payload.bytes);
    gpa.destroy(payload);
}
