//! Waveform container I/O and rate conversion.
//!
//! The engine reads a reference soundtrack (WAV) and writes a rendered one, and
//! anything it reads has to reach the audio VAE's 32 kHz. Both directions live
//! here so there is one place that knows about interleaving and full scale.
//!
//! Waveforms are INTERLEAVED `[frames][channels]` f32 in [-1, 1], which is what
//! a container gives and what a container wants; the models are planar and
//! convert at their own edge.

const std = @import("std");

/// An interleaved f32 waveform.
pub const Wave = struct {
    /// `[frames][channels]`, interleaved.
    samples: []f32,
    channels: usize,
    sample_rate: u32,

    pub fn frames(self: Wave) usize {
        return if (self.channels == 0) 0 else self.samples.len / self.channels;
    }

    pub fn seconds(self: Wave) f64 {
        return @as(f64, @floatFromInt(self.frames())) / @as(f64, @floatFromInt(self.sample_rate));
    }

    pub fn deinit(self: *Wave, gpa: std.mem.Allocator) void {
        gpa.free(self.samples);
        self.* = undefined;
    }
};

// --- WAV ------------------------------------------------------------------

const fmt_pcm: u16 = 1;
const fmt_float: u16 = 3;
const fmt_extensible: u16 = 0xFFFE;

/// Parse a RIFF/WAVE file: PCM at 8/16/24/32 bits or IEEE float at 32/64,
/// including `WAVE_FORMAT_EXTENSIBLE` (whose real format is the first field of
/// the sub-format GUID). Unknown chunks are skipped, which is the only way to
/// read a file any real encoder produced -- `LIST`, `fact` and `id3 ` chunks are
/// routine and a reader that assumes `fmt ` then `data` fails on half of them.
///
/// 16-bit samples divide by 32768 (full scale), while `encodeWavPcm16` multiplies
/// by 32767 (which is what keeps +1.0 from wrapping). A round trip through both
/// therefore loses one part in 32768, not zero.
pub fn decodeWav(gpa: std.mem.Allocator, bytes: []const u8) !Wave {
    if (bytes.len < 12) return error.InvalidHeader;
    if (!std.mem.eql(u8, bytes[0..4], "RIFF") or !std.mem.eql(u8, bytes[8..12], "WAVE")) {
        return error.InvalidHeader;
    }

    var format: u16 = 0;
    var channels: usize = 0;
    var rate: u32 = 0;
    var bits: u16 = 0;
    var data: ?[]const u8 = null;

    var at: usize = 12;
    while (at + 8 <= bytes.len) {
        const id = bytes[at..][0..4];
        const size = std.mem.readInt(u32, bytes[at + 4 ..][0..4], .little);
        const body_at = at + 8;
        // A truncated final chunk is common enough (a writer that died mid-stream)
        // that it is worth reading what is there rather than refusing the file.
        const body_end = @min(bytes.len, body_at + size);
        const body = bytes[body_at..body_end];

        if (std.mem.eql(u8, id, "fmt ")) {
            if (body.len < 16) return error.InvalidHeader;
            format = std.mem.readInt(u16, body[0..2], .little);
            channels = std.mem.readInt(u16, body[2..4], .little);
            rate = std.mem.readInt(u32, body[4..8], .little);
            bits = std.mem.readInt(u16, body[14..16], .little);
            if (format == fmt_extensible) {
                if (body.len < 26) return error.InvalidHeader;
                format = std.mem.readInt(u16, body[24..26], .little);
            }
        } else if (std.mem.eql(u8, id, "data")) {
            data = body;
        }
        // Chunks are padded to an even length, and the pad byte is not in `size`.
        at = body_at + size + (size & 1);
    }

    const payload = data orelse return error.MissingTensor;
    if (channels == 0 or rate == 0) return error.InvalidHeader;

    const bytes_per: usize = switch (format) {
        fmt_pcm => switch (bits) {
            8, 16, 24, 32 => bits / 8,
            else => return error.UnsupportedDtype,
        },
        fmt_float => switch (bits) {
            32, 64 => bits / 8,
            else => return error.UnsupportedDtype,
        },
        else => return error.UnsupportedDtype,
    };

    const n = payload.len / (bytes_per * channels) * channels;
    const out = try gpa.alloc(f32, n);
    errdefer gpa.free(out);
    for (out, 0..) |*o, i| {
        const p = payload[i * bytes_per ..];
        o.* = switch (format) {
            fmt_pcm => switch (bits) {
                // 8-bit PCM is UNSIGNED, alone among the widths.
                8 => (@as(f32, @floatFromInt(p[0])) - 128.0) / 128.0,
                16 => @as(f32, @floatFromInt(std.mem.readInt(i16, p[0..2], .little))) / 32768.0,
                24 => blk: {
                    const v: i32 = (@as(i32, p[0]) | (@as(i32, p[1]) << 8) |
                        (@as(i32, @as(i8, @bitCast(p[2]))) << 16));
                    break :blk @as(f32, @floatFromInt(v)) / 8388608.0;
                },
                32 => @as(f32, @floatFromInt(std.mem.readInt(i32, p[0..4], .little))) / 2147483648.0,
                else => unreachable,
            },
            fmt_float => switch (bits) {
                32 => @bitCast(std.mem.readInt(u32, p[0..4], .little)),
                64 => @floatCast(@as(f64, @bitCast(std.mem.readInt(u64, p[0..8], .little)))),
                else => unreachable,
            },
            else => unreachable,
        };
    }
    return .{ .samples = out, .channels = channels, .sample_rate = rate };
}

/// Interleaved f32 in [-1, 1] -> a 16-bit PCM WAV. The one container format
/// worth hand-writing: 44 bytes of header and no dependency.
pub fn encodeWavPcm16(gpa: std.mem.Allocator, samples: []const f32, channels: usize, rate: u32) ![]u8 {
    const bytes_per = 2;
    const data_len = samples.len * bytes_per;
    var out = try gpa.alloc(u8, 44 + data_len);
    errdefer gpa.free(out);
    const byte_rate = rate * @as(u32, @intCast(channels)) * bytes_per;

    @memcpy(out[0..4], "RIFF");
    std.mem.writeInt(u32, out[4..8], @intCast(36 + data_len), .little);
    @memcpy(out[8..12], "WAVE");
    @memcpy(out[12..16], "fmt ");
    std.mem.writeInt(u32, out[16..20], 16, .little); // PCM header size
    std.mem.writeInt(u16, out[20..22], fmt_pcm, .little);
    std.mem.writeInt(u16, out[22..24], @intCast(channels), .little);
    std.mem.writeInt(u32, out[24..28], rate, .little);
    std.mem.writeInt(u32, out[28..32], byte_rate, .little);
    std.mem.writeInt(u16, out[32..34], @intCast(channels * bytes_per), .little); // block align
    std.mem.writeInt(u16, out[34..36], 16, .little); // bits per sample
    @memcpy(out[36..40], "data");
    std.mem.writeInt(u32, out[40..44], @intCast(data_len), .little);

    for (samples, 0..) |v, si| {
        // Symmetric scaling by 32767 rather than 32768: the latter clips +1.0 to
        // -32768 on wrap, which is a loud click at exactly full scale.
        const q: i16 = @intFromFloat(@round(std.math.clamp(v, -1.0, 1.0) * 32767.0));
        std.mem.writeInt(i16, out[44 + si * 2 ..][0..2], q, .little);
    }
    return out;
}

// --- resampling -----------------------------------------------------------

/// torchaudio's `functional.resample` defaults, which is the filter every
/// reference implementation of these models feeds its audio VAE through.
pub const ResampleOptions = struct {
    /// Zero crossings of the sinc kept on each side.
    lowpass_filter_width: usize = 6,
    /// Fraction of Nyquist the antialiasing filter passes. It is inside the
    /// window argument, the sinc argument AND the output scale.
    rolloff: f64 = 0.99,
};

/// A polyphase windowed-sinc resampling filter: `phases` FIR kernels of `taps`
/// taps each, one per output phase.
pub const Kernel = struct {
    /// `[phases][taps]`.
    taps: []f32,
    phases: usize,
    taps_per_phase: usize,
    /// Zero samples the signal is padded with on the left (and `width + orig` on
    /// the right).
    width: usize,
    /// The rates AFTER dividing by their GCD. Everything else is derived from
    /// these, so the reduction has to happen first.
    orig: usize,
    new: usize,

    pub fn deinit(self: *Kernel, gpa: std.mem.Allocator) void {
        gpa.free(self.taps);
        self.* = undefined;
    }

    /// Output frames a signal of `len` input frames produces.
    pub fn outLen(self: Kernel, len: usize) usize {
        return (self.new * len + self.orig - 1) / self.orig;
    }
};

/// Build the filter for `orig_rate -> new_rate`.
///
/// Computed in f64 and narrowed once at the end; the reference computes in f32
/// throughout, so individual taps differ in the last bit or two. That is a
/// smaller error than the reference has against its own f64 self, and the
/// convolution averages hundreds of taps.
pub fn sincKernel(gpa: std.mem.Allocator, orig_rate: u32, new_rate: u32, o: ResampleOptions) !Kernel {
    if (orig_rate == 0 or new_rate == 0) return error.InvalidHeader;
    if (o.lowpass_filter_width == 0) return error.InvalidHeader;
    const g = std.math.gcd(@as(usize, orig_rate), @as(usize, new_rate));
    const orig = @as(usize, orig_rate) / g;
    const new = @as(usize, new_rate) / g;

    const lfw: f64 = @floatFromInt(o.lowpass_filter_width);
    const base = @as(f64, @floatFromInt(@min(orig, new))) * o.rolloff;
    const width: usize = @intFromFloat(@ceil(lfw * @as(f64, @floatFromInt(orig)) / base));
    const taps = 2 * width + orig;
    const scale = base / @as(f64, @floatFromInt(orig));

    const out = try gpa.alloc(f32, new * taps);
    errdefer gpa.free(out);
    for (0..new) |p| {
        const phase = -@as(f64, @floatFromInt(p)) / @as(f64, @floatFromInt(new));
        for (0..taps) |i| {
            const idx = (@as(f64, @floatFromInt(i)) - @as(f64, @floatFromInt(width))) /
                @as(f64, @floatFromInt(orig));
            // Clamped BEFORE both the window and the sinc: they are evaluated at
            // the same `t`, not on separate grids.
            const t = std.math.clamp((phase + idx) * base, -lfw, lfw);
            const cw = @cos(t * std.math.pi / lfw / 2.0);
            const w = cw * cw;
            const ts = t * std.math.pi;
            const s = if (t == 0) 1.0 else @sin(ts) / ts;
            out[p * taps + i] = @floatCast(s * w * scale);
        }
    }
    return .{
        .taps = out,
        .phases = new,
        .taps_per_phase = taps,
        .width = width,
        .orig = orig,
        .new = new,
    };
}

/// Apply a kernel to one interleaved channel of `x`, writing `out` frames.
///
/// Output frame `j` takes phase `j % new` of block `j / new`, whose input window
/// starts at `block * orig - width`. Reading it block-major with the phase INSIDE
/// is what the reference's transpose-then-flatten does; the other order
/// interleaves the polyphase branches and shifts the pitch.
fn applyKernel(k: Kernel, out: []f32, x: []const f32, len: usize, channels: usize, ch: usize) void {
    const taps = k.taps_per_phase;
    for (out, 0..) |*o, j| {
        const p = j % k.phases;
        const b = j / k.phases;
        const row = k.taps[p * taps ..][0..taps];
        const base = @as(isize, @intCast(b * k.orig)) - @as(isize, @intCast(k.width));
        // Only the taps that land inside the signal contribute; the rest read the
        // reference's zero padding.
        const lo: usize = @intCast(@max(0, -base));
        const hi = @min(taps, @as(usize, @intCast(@max(0, @as(isize, @intCast(len)) - base))));
        var acc: f32 = 0;
        var i = lo;
        while (i < hi) : (i += 1) {
            const s: usize = @intCast(base + @as(isize, @intCast(i)));
            acc += row[i] * x[s * channels + ch];
        }
        o.* = acc;
    }
}

/// Resample a waveform to `target_rate`. Returns a fresh `Wave`; a matching rate
/// is an exact copy, not a filter pass (the reference's own no-op).
pub fn resample(gpa: std.mem.Allocator, w: Wave, target_rate: u32, o: ResampleOptions) !Wave {
    if (w.sample_rate == target_rate) {
        return .{
            .samples = try gpa.dupe(f32, w.samples),
            .channels = w.channels,
            .sample_rate = target_rate,
        };
    }
    var k = try sincKernel(gpa, w.sample_rate, target_rate, o);
    defer k.deinit(gpa);

    const len = w.frames();
    const out_len = k.outLen(len);
    const out = try gpa.alloc(f32, out_len * w.channels);
    errdefer gpa.free(out);
    const scratch = try gpa.alloc(f32, out_len);
    defer gpa.free(scratch);
    for (0..w.channels) |c| {
        applyKernel(k, scratch, w.samples, len, w.channels, c);
        for (0..out_len) |j| out[j * w.channels + c] = scratch[j];
    }
    return .{ .samples = out, .channels = w.channels, .sample_rate = target_rate };
}

// --- tests ----------------------------------------------------------------

const resample_fixture = @embedFile("assets/resample.safetensors");
const safetensors = @import("safetensors.zig");

fn relL2(want: []const f32, got: []const f32) f64 {
    std.debug.assert(want.len == got.len);
    var l2_ref: f64 = 0;
    var l2_err: f64 = 0;
    for (want, got) |e, a| {
        l2_ref += @as(f64, e) * e;
        l2_err += @as(f64, e - a) * (e - a);
    }
    return if (l2_ref > 0) @sqrt(l2_err / l2_ref) else @sqrt(l2_err);
}

test "a 16-bit WAV round-trips through both directions" {
    const gpa = std.testing.allocator;
    const src = [_]f32{ 0.0, 0.5, -0.5, 1.0, -1.0, 0.25 }; // three stereo frames
    const bytes = try encodeWavPcm16(gpa, &src, 2, 32000);
    defer gpa.free(bytes);
    var w = try decodeWav(gpa, bytes);
    defer w.deinit(gpa);

    try std.testing.expectEqual(@as(usize, 2), w.channels);
    try std.testing.expectEqual(@as(u32, 32000), w.sample_rate);
    try std.testing.expectEqual(@as(usize, 3), w.frames());
    // One part in 32768: the writer scales by 32767 and the reader by 32768.
    for (src, w.samples) |e, a| try std.testing.expectApproxEqAbs(e, a, 1e-4);
}

test "the WAV reader skips chunks it does not know" {
    // A real file has LIST / fact / id3 chunks between `fmt ` and `data`, and an
    // odd-sized chunk carries a pad byte that is NOT counted in its size.
    const gpa = std.testing.allocator;
    const src = [_]f32{ 0.25, -0.25 };
    const plain = try encodeWavPcm16(gpa, &src, 1, 16000);
    defer gpa.free(plain);

    // Splice a 3-byte "LIST" chunk (plus its pad) in front of `data`.
    var spliced: std.ArrayList(u8) = .empty;
    defer spliced.deinit(gpa);
    try spliced.appendSlice(gpa, plain[0..36]);
    try spliced.appendSlice(gpa, "LIST");
    var sz: [4]u8 = undefined;
    std.mem.writeInt(u32, &sz, 3, .little);
    try spliced.appendSlice(gpa, &sz);
    try spliced.appendSlice(gpa, &[_]u8{ 1, 2, 3, 0 }); // 3 bytes + pad
    try spliced.appendSlice(gpa, plain[36..]);
    std.mem.writeInt(u32, spliced.items[4..8], @intCast(spliced.items.len - 8), .little);

    var w = try decodeWav(gpa, spliced.items);
    defer w.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 1), w.channels);
    try std.testing.expectEqual(@as(usize, 2), w.frames());
    for (src, w.samples) |e, a| try std.testing.expectApproxEqAbs(e, a, 1e-4);
}

test "resampling to the same rate is the identity" {
    const gpa = std.testing.allocator;
    const src = [_]f32{ 0.1, -0.2, 0.3, -0.4 };
    var w = try resample(gpa, .{ .samples = @constCast(&src), .channels = 2, .sample_rate = 32000 }, 32000, .{});
    defer w.deinit(gpa);
    try std.testing.expectEqualSlices(f32, &src, w.samples);
}

test "the resampler matches torchaudio at its defaults" {
    // From tools/gen_resample.py. Three rate relations: a large-GCD downsample, the
    // awkward 44.1k -> 32k (441 -> 320, a 320-phase filter), and an upsample.
    const gpa = std.testing.allocator;
    var st = try safetensors.SafeTensors.initFromSlice(gpa, resample_fixture);
    defer st.deinit();

    const sigv = try st.require("in.signal");
    const sig = try sigv.toF32Alloc(gpa);
    defer gpa.free(sig);

    const cases = [_][2]u32{ .{ 48000, 32000 }, .{ 44100, 32000 }, .{ 16000, 32000 } };
    const widths = [_]usize{ 10, 9, 7 };
    const phases = [_]usize{ 2, 320, 2 };
    for (cases, widths, phases) |c, want_width, want_phases| {
        var buf: [64]u8 = undefined;
        const tag = try std.fmt.bufPrint(&buf, "{d}_{d}", .{ c[0], c[1] });

        var k = try sincKernel(gpa, c[0], c[1], .{});
        defer k.deinit(gpa);
        errdefer std.debug.print("case {s}\n", .{tag});
        try std.testing.expectEqual(want_width, k.width);
        try std.testing.expectEqual(want_phases, k.phases);
        try std.testing.expectEqual(2 * k.width + k.orig, k.taps_per_phase);

        // Against the reference's f64 kernel, which is the filter the formula
        // defines. Its own f32 kernel -- the one it actually applies -- sits 1e-5
        // away from that for 441 -> 320, whose 320 phases have near-zero taps with
        // little significance left in f32. Comparing against the f32 kernel would
        // read as a 1e-5 error with nothing wrong, so the fixture carries both and
        // the f32 gap is asserted to be the LOOSER of the two.
        var kbuf: [80]u8 = undefined;
        const kv = try st.require(try std.fmt.bufPrint(&kbuf, "kernel64.{s}", .{tag}));
        const want_k = try kv.toF32Alloc(gpa);
        defer gpa.free(want_k);
        const k_err = relL2(want_k, k.taps);
        errdefer std.debug.print("kernel rel L2 vs f64 {e}\n", .{k_err});
        try std.testing.expect(k_err < 1e-6);

        const kv32 = try st.require(try std.fmt.bufPrint(&kbuf, "kernel.{s}", .{tag}));
        const want_k32 = try kv32.toF32Alloc(gpa);
        defer gpa.free(want_k32);
        try std.testing.expect(relL2(want_k32, k.taps) >= k_err);

        const wantv = try st.require(try std.fmt.bufPrint(&kbuf, "out.{s}", .{tag}));
        const want = try wantv.toF32Alloc(gpa);
        defer gpa.free(want);

        var got = try resample(gpa, .{ .samples = sig, .channels = 1, .sample_rate = c[0] }, c[1], .{});
        defer got.deinit(gpa);
        try std.testing.expectEqual(want.len, got.samples.len);
        // The reference convolves with its f32 kernel, so its output carries the
        // same 1e-5 for the 320-phase case. That is the floor here, not our error.
        const err = relL2(want, got.samples);
        errdefer std.debug.print("resample rel L2 {e}\n", .{err});
        try std.testing.expect(err < 3e-5);
    }
}

test "a resampled stereo signal keeps its channels apart" {
    // The per-channel loop writes back interleaved, which is the one place a
    // stereo pair can get mixed. Two channels that differ only in sign must come
    // out differing only in sign.
    const gpa = std.testing.allocator;
    const n = 300;
    const src = try gpa.alloc(f32, n * 2);
    defer gpa.free(src);
    for (0..n) |i| {
        const v = @sin(@as(f32, @floatFromInt(i)) * 0.11);
        src[i * 2] = v;
        src[i * 2 + 1] = -v;
    }
    var w = try resample(gpa, .{ .samples = src, .channels = 2, .sample_rate = 44100 }, 32000, .{});
    defer w.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 2), w.channels);
    for (0..w.frames()) |i| {
        try std.testing.expectApproxEqAbs(w.samples[i * 2], -w.samples[i * 2 + 1], 1e-6);
    }
    // ...and it really resampled: 44.1k -> 32k shortens by 320/441.
    try std.testing.expectEqual((320 * n + 440) / 441, w.frames());
}
