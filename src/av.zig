//! Clip muxing via libav*, behind a small C shim (`lib/video/av_helper.c`).
//!
//! Linked into the EXECUTABLES only, never into the `TensorPencil` library
//! module, which stays pure Zig — the same split the libvips image decode uses.
//! The engine hands back a `pipeline.Clip` of plain data and the driver writes
//! the container.
//!
//! The `parameters` metadata tag is not decoration: a reader re-renders from it,
//! exactly as `image.zig` puts the same block in a PNG text chunk. DiffKeep
//! already reads format-level tags out of video files, so there is a consumer.

const std = @import("std");

const c = @cImport({
    @cInclude("av_helper.h");
});

pub const Error = error{ MuxOpenFailed, MuxWriteFailed, MuxFinishFailed };

/// Last error text from the shim, for a diagnostic. Valid until the next call.
pub fn lastError() []const u8 {
    const p = c.tp_mux_last_error() orelse return "unknown";
    return std.mem.span(p);
}

pub const Config = struct {
    width: usize,
    height: usize,
    fps_num: u32 = 24,
    fps_den: u32 = 1,
    /// 1..51, lower is better. 0 takes the encoder default.
    crf: u8 = 18,
    audio_channels: usize = 0,
    audio_sample_rate: u32 = 0,
    /// Format-level metadata. Both must be non-null to be written.
    meta_key: ?[:0]const u8 = null,
    meta_value: ?[:0]const u8 = null,
};

pub const Muxer = struct {
    handle: *c.TpMuxer,

    pub fn open(path: [:0]const u8, cfg: Config) Error!Muxer {
        var raw: c.TpMuxConfig = .{
            .width = @intCast(cfg.width),
            .height = @intCast(cfg.height),
            .fps_num = @intCast(cfg.fps_num),
            .fps_den = @intCast(cfg.fps_den),
            .crf = cfg.crf,
            .audio_channels = @intCast(cfg.audio_channels),
            .audio_sample_rate = @intCast(cfg.audio_sample_rate),
            .meta_key = if (cfg.meta_key) |k| k.ptr else null,
            .meta_value = if (cfg.meta_value) |v| v.ptr else null,
        };
        const h = c.tp_mux_open(path.ptr, &raw) orelse return error.MuxOpenFailed;
        return .{ .handle = h };
    }

    /// One frame of packed RGB8, `width * height * 3` bytes.
    pub fn writeFrame(self: Muxer, rgb: []const u8) Error!void {
        if (c.tp_mux_write_frame(self.handle, rgb.ptr) != 0) return error.MuxWriteFailed;
    }

    /// Interleaved f32 in [-1, 1]; `frames` is the per-channel count. May be
    /// called once with the whole track.
    pub fn writeAudio(self: Muxer, interleaved: []const f32, frames: usize) Error!void {
        if (c.tp_mux_write_audio(self.handle, interleaved.ptr, frames) != 0) return error.MuxWriteFailed;
    }

    /// Flush, write the trailer, close. Consumes the muxer either way.
    pub fn finish(self: Muxer) Error!void {
        if (c.tp_mux_finish(self.handle) != 0) return error.MuxFinishFailed;
    }

    /// Abandon the file without finishing it.
    pub fn abort(self: Muxer) void {
        c.tp_mux_abort(self.handle);
    }
};
