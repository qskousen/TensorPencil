//! Image framing for the composer's quick settings: pick an aspect ratio and a
//! pixel budget, get concrete dimensions.
//!
//! Ratio + megapixels rather than raw width/height, because those are the two
//! things a person actually decides ("landscape, and not too slow"), while
//! width and height are what the model needs. Held apart here so the mapping is
//! pure and testable; the settings view still edits width/height directly for
//! anyone who wants an exact size.
//!
//! Dimensions are rounded to a multiple of 64. Every architecture here patches
//! or downsamples by 8 at minimum and most by 16 or 32; 64 clears all of them,
//! and being one pixel off is a hard error inside a UNet, not a slightly
//! different picture.
const std = @import("std");

pub const align_to: usize = 64;
pub const min_dim: usize = 128;
pub const max_dim: usize = 4096;

pub const Ratio = enum {
    square,
    landscape_3_2,
    portrait_2_3,
    landscape_4_3,
    portrait_3_4,
    wide_16_9,
    tall_9_16,

    pub fn label(self: Ratio) []const u8 {
        return switch (self) {
            .square => "1:1",
            .landscape_3_2 => "3:2",
            .portrait_2_3 => "2:3",
            .landscape_4_3 => "4:3",
            .portrait_3_4 => "3:4",
            .wide_16_9 => "16:9",
            .tall_9_16 => "9:16",
        };
    }

    /// width:height as a float.
    pub fn value(self: Ratio) f64 {
        return switch (self) {
            .square => 1.0,
            .landscape_3_2 => 3.0 / 2.0,
            .portrait_2_3 => 2.0 / 3.0,
            .landscape_4_3 => 4.0 / 3.0,
            .portrait_3_4 => 3.0 / 4.0,
            .wide_16_9 => 16.0 / 9.0,
            .tall_9_16 => 9.0 / 16.0,
        };
    }
};

/// The pixel budget, in megapixels. These are the sizes worth offering: 1.0 MP
/// is the 1024² most current checkpoints are trained at, and the neighbours are
/// a stop either side.
pub const Size = enum {
    mp_025,
    mp_05,
    mp_1,
    mp_2,
    mp_4,

    pub fn label(self: Size) []const u8 {
        return switch (self) {
            .mp_025 => "0.25 MP",
            .mp_05 => "0.5 MP",
            .mp_1 => "1 MP",
            .mp_2 => "2 MP",
            .mp_4 => "4 MP",
        };
    }

    pub fn megapixels(self: Size) f64 {
        return switch (self) {
            .mp_025 => 0.25,
            .mp_05 => 0.5,
            .mp_1 => 1.0,
            .mp_2 => 2.0,
            .mp_4 => 4.0,
        };
    }
};

pub const Dims = struct { w: usize, h: usize };

/// Bounds on a typed pixel budget. The floor is a thumbnail, the ceiling is
/// past what any checkpoint here was trained for and well past what fits in
/// VRAM; both exist so a stray keystroke cannot ask for a 900 MP render.
pub const mp_min: f64 = 0.05;
pub const mp_max: f64 = 16.0;

fn snap(v: f64) usize {
    const r = @round(v / @as(f64, align_to)) * @as(f64, align_to);
    const i: usize = @intFromFloat(@max(r, @as(f64, align_to)));
    return std.math.clamp(i, min_dim, max_dim);
}

/// Concrete dimensions for a ratio and one of the offered budgets.
pub fn dims(ratio: Ratio, size: Size) Dims {
    return dimsMp(ratio, size.megapixels());
}

/// Concrete dimensions for a ratio and an arbitrary typed budget, so the
/// composer can accept "1.8" as readily as a preset.
pub fn dimsMp(ratio: Ratio, mp: f64) Dims {
    const area = std.math.clamp(mp, mp_min, mp_max) * 1_000_000.0;
    const r = ratio.value();
    // w*h = area and w/h = r  ->  w = sqrt(area*r).
    //
    // Height is derived from the SNAPPED width, not snapped independently.
    // Rounding both to 64 lets the two errors pull the same way, and at a
    // quarter megapixel a 64px step is a big enough fraction of a side to land
    // the result nearer a different ratio than the one asked for.
    const w = snap(@sqrt(area * r));
    return .{ .w = w, .h = snap(@as(f64, @floatFromInt(w)) / r) };
}

/// The closest ratio to a width/height already in the config, so the chip shows
/// what is actually set rather than resetting the user's exact size on sight.
pub fn nearestRatio(w: usize, h: usize) Ratio {
    if (w == 0 or h == 0) return .square;
    const have = @as(f64, @floatFromInt(w)) / @as(f64, @floatFromInt(h));
    var best: Ratio = .square;
    var best_err = std.math.inf(f64);
    for (std.enums.values(Ratio)) |r| {
        // Compared in log space: 16:9 and 9:16 are the same distance from
        // square, which a plain difference does not give.
        const err = @abs(@log(have) - @log(r.value()));
        if (err < best_err) {
            best_err = err;
            best = r;
        }
    }
    return best;
}

/// The pixel budget a width/height already represents.
pub fn megapixelsOf(w: usize, h: usize) f64 {
    return @as(f64, @floatFromInt(w * h)) / 1_000_000.0;
}

/// Format a budget the way it is typed: up to two decimals, no trailing zeros.
pub fn formatMp(buf: []u8, mp: f64) []const u8 {
    var tmp: [32]u8 = undefined;
    const t = std.fmt.bufPrint(&tmp, "{d:.2}", .{mp}) catch return "";
    var end = t.len;
    if (std.mem.indexOfScalar(u8, t, '.') != null) {
        while (end > 0 and t[end - 1] == '0') end -= 1;
        if (end > 0 and t[end - 1] == '.') end -= 1;
    }
    const n = @min(end, buf.len);
    @memcpy(buf[0..n], t[0..n]);
    return buf[0..n];
}

/// The closest offered budget to a width/height already in the config.
pub fn nearestSize(w: usize, h: usize) Size {
    const mp = @as(f64, @floatFromInt(w * h)) / 1_000_000.0;
    var best: Size = .mp_1;
    var best_err = std.math.inf(f64);
    for (std.enums.values(Size)) |s| {
        const err = @abs(@log(@max(mp, 0.001)) - @log(s.megapixels()));
        if (err < best_err) {
            best_err = err;
            best = s;
        }
    }
    return best;
}

test "dimensions hit the pixel budget at the requested ratio" {
    for (std.enums.values(Ratio)) |r| {
        for (std.enums.values(Size)) |s| {
            const d = dims(r, s);
            // Every dimension a model can actually take.
            try std.testing.expectEqual(@as(usize, 0), d.w % align_to);
            try std.testing.expectEqual(@as(usize, 0), d.h % align_to);
            try std.testing.expect(d.w >= min_dim and d.w <= max_dim);
            try std.testing.expect(d.h >= min_dim and d.h <= max_dim);

            // Rounding to 64 moves the area, but not by much: within 25% of the
            // budget, and the aspect within 12% of the request. (The tightest
            // case is 0.25 MP at 16:9, where 64px is a big fraction of a side.)
            const mp = @as(f64, @floatFromInt(d.w * d.h)) / 1_000_000.0;
            const err = @abs(mp - s.megapixels()) / s.megapixels();
            errdefer std.debug.print("{s} {s} -> {d}x{d} = {d:.3} MP\n", .{ r.label(), s.label(), d.w, d.h, mp });
            try std.testing.expect(err < 0.25);
            const got = @as(f64, @floatFromInt(d.w)) / @as(f64, @floatFromInt(d.h));
            try std.testing.expect(@abs(@log(got) - @log(r.value())) < 0.12);
        }
    }
}

test "a typed budget is honoured and clamped" {
    // The whole point of the text field: a value between the presets.
    const d = dimsMp(.landscape_3_2, 1.8);
    try std.testing.expectEqual(@as(usize, 0), d.w % align_to);
    try std.testing.expectEqual(@as(usize, 0), d.h % align_to);
    const mp = megapixelsOf(d.w, d.h);
    try std.testing.expect(@abs(mp - 1.8) / 1.8 < 0.1);
    // Nonsense is clamped, never obeyed.
    const huge = dimsMp(.square, 9000);
    try std.testing.expect(huge.w <= max_dim and huge.h <= max_dim);
    const tiny = dimsMp(.square, 0);
    try std.testing.expect(tiny.w >= min_dim and tiny.h >= min_dim);
}

test "budgets format the way they are typed" {
    var buf: [16]u8 = undefined;
    try std.testing.expectEqualStrings("1", formatMp(&buf, 1.0));
    try std.testing.expectEqualStrings("1.8", formatMp(&buf, 1.8));
    try std.testing.expectEqualStrings("0.25", formatMp(&buf, 0.25));
    try std.testing.expectEqualStrings("2.5", formatMp(&buf, 2.5));
}

test "the 1 MP square is the 1024 every checkpoint is trained at" {
    const d = dims(.square, .mp_1);
    try std.testing.expectEqual(@as(usize, 1024), d.w);
    try std.testing.expectEqual(@as(usize, 1024), d.h);
}

test "an existing size maps back to the chips that describe it" {
    try std.testing.expectEqual(Ratio.square, nearestRatio(1024, 1024));
    try std.testing.expectEqual(Ratio.landscape_3_2, nearestRatio(1216, 832));
    try std.testing.expectEqual(Ratio.tall_9_16, nearestRatio(768, 1344));
    try std.testing.expectEqual(Size.mp_1, nearestSize(1024, 1024));
    try std.testing.expectEqual(Size.mp_025, nearestSize(512, 512));
    // A hand-typed size that matches nothing still picks its nearest neighbour
    // rather than snapping the user's config to a preset.
    try std.testing.expectEqual(Ratio.landscape_4_3, nearestRatio(1000, 760));
}

test "every ratio and size round-trips through its own dims" {
    for (std.enums.values(Ratio)) |r| {
        for (std.enums.values(Size)) |s| {
            const d = dims(r, s);
            errdefer std.debug.print("{s} {s} -> {d}x{d}\n", .{ r.label(), s.label(), d.w, d.h });
            try std.testing.expectEqual(r, nearestRatio(d.w, d.h));
            try std.testing.expectEqual(s, nearestSize(d.w, d.h));
        }
    }
}
