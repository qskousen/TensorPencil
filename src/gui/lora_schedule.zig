//! A LoRA's step schedule as the studio and Settings both edit it: a chip naming
//! the steps it reaches, and an editor (every step, a window of noise levels, or a
//! gain curve) with what it does to each step drawn underneath. The schedule
//! itself is `lora.Schedule`; the steps come from `pipeline.stepLevels`.

const std = @import("std");
const dvui = @import("dvui");
const tp = @import("TensorPencil");
const config = @import("shared").config;
const model_spec = @import("shared").model_spec;
const fonts = @import("fonts.zig");
const style = @import("style.zig");
const bubbles = @import("bubbles.zig");

const C = style.C;
const F = style.F;
pub const Schedule = tp.models.lora.Schedule;
const Curve = tp.models.lora.Curve;

/// The noise level of each step a render at these settings takes; empty when the
/// scheduler cannot be built for this family.
pub fn levelsFor(arena: std.mem.Allocator, fam: model_spec.Family, width: usize, height: usize, steps: usize, sched: ?tp.sampler.Scheduler) []const f32 {
    return tp.pipeline.stepLevels(arena, fam, width, height, steps, sched) catch &.{};
}

/// `s` as the wire and the config carry it: empty for every step.
pub fn text(buf: *[config.max_lora_schedule]u8, s: Schedule) []const u8 {
    if (s == .always) return "";
    // A full curve fits (`max_lora_schedule`); this cannot fail.
    return std.fmt.bufPrint(buf, "{f}", .{s}) catch unreachable;
}

/// What a schedule does at these steps, for the chip: "every step",
/// "steps 1-2 of 8", "steps 1-3, 7 of 8" (1-based, like the progress readout).
pub fn summary(buf: []u8, s: Schedule, levels: []const f32) []const u8 {
    if (s == .always) return "every step";
    if (levels.len == 0) return if (s == .window) "window" else "curve";
    var w: std.Io.Writer = .fixed(buf);
    w.writeAll("steps ") catch return "steps";
    var any = false;
    var i: usize = 0;
    while (i < levels.len) {
        if (s.gain(levels[i]) == 0) {
            i += 1;
            continue;
        }
        var j = i;
        while (j + 1 < levels.len and s.gain(levels[j + 1]) != 0) j += 1;
        if (any) w.writeAll(", ") catch break;
        any = true;
        if (j == i) w.print("{d}", .{i + 1}) catch break else w.print("{d}-{d}", .{ i + 1, j + 1 }) catch break;
        i = j + 1;
    }
    if (!any) return "no steps";
    w.print(" of {d}", .{levels.len}) catch {};
    return w.buffered();
}

/// The chip that opens the editor. Returns true when clicked.
pub fn chip(src: std.lang.SourceLocation, id: usize, s: Schedule, levels: []const f32) bool {
    var buf: [64]u8 = undefined;
    return style.chip(src, summary(&buf, s, levels), .{
        .id_extra = id,
        .margin_x = 6,
        .dropdown = true,
        .text = if (s == .always) C.text_dim else C.text,
    });
}

/// The editor for `s`, laid out under the row that owns it. Returns true when the
/// schedule changed this frame.
pub fn editor(src: std.lang.SourceLocation, id: usize, s: *Schedule, levels: []const f32) bool {
    const before = s.*;
    var box = dvui.box(src, .{ .dir = .vertical }, .{
        .id_extra = id,
        .expand = .horizontal,
        .margin = .{ .x = 28, .h = 8 },
    });
    defer box.deinit();

    const mode: usize = switch (s.*) {
        .always => 0,
        .window => 1,
        .curve => 2,
    };
    {
        // Its own row: `segmented` centers itself vertically in its parent.
        var seg_row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .margin = .{ .h = 4 } });
        defer seg_row.deinit();
        if (style.segmented(@src(), &.{ "Every step", "Window", "Curve" }, mode)) |pick| if (pick != mode) {
            s.* = switch (pick) {
                0 => .always,
                1 => .{ .window = .{ .hi = 1, .lo = defaultWindowLo(levels) } },
                else => .{ .curve = Curve.init(&.{ .{ .level = 1, .gain = 1 }, .{ .level = 0.5, .gain = 0 } }) catch unreachable },
            };
        };
    }
    help("A schedule reads the noise level: 1 is pure noise, where the first step starts, " ++
        "and 0 is the finished image. Keyed to it rather than to step numbers, so it holds at any step count.");

    switch (s.*) {
        .always => {},
        .window => |*win| {
            levelSlider(@src(), 0, "From", &win.hi, 0, 1);
            levelSlider(@src(), 1, "To", &win.lo, 0, win.hi);
        },
        .curve => |*c| curvePoints(c),
    }

    if (levels.len > 0) {
        var gains: [128]f32 = undefined;
        const n = @min(levels.len, gains.len);
        for (gains[0..n], levels[0..n]) |*g, l| g.* = s.gain(l);
        style.sparkline(@src(), gains[0..n], C.blue, 18);
    }
    return !s.eql(before);
}

/// A window on the first two steps, the usual ask: down to halfway between the
/// second step's level and the third's.
fn defaultWindowLo(levels: []const f32) f32 {
    if (levels.len < 3) return 0;
    return (levels[1] + levels[2]) / 2;
}

fn help(t: []const u8) void {
    var tl = dvui.textLayout(@src(), .{}, .{
        .expand = .horizontal,
        .background = false,
        .max_size_content = .width(style.Layout.prose_max),
        .padding = .{ .y = 2, .h = 8 },
    });
    defer tl.deinit();
    fonts.addStyled(tl, t, .{}, .{ .font = F.row, .color_text = .{ .color = C.text_ghost } });
}

fn levelSlider(src: std.lang.SourceLocation, id: usize, label: []const u8, v: *f32, min: f32, max: f32) void {
    var row = dvui.box(src, .{ .dir = .horizontal }, .{ .id_extra = id, .margin = .{ .h = 5 } });
    defer row.deinit();
    dvui.labelNoFmt(@src(), label, .{}, .{
        .font = F.ui,
        .color_text = .{ .color = C.text_dim },
        .padding = .{},
        .gravity_y = 0.5,
        .min_size_content = .{ .w = 60 },
    });
    _ = dvui.sliderEntry(@src(), "{d:0.3}", .{ .value = v, .min = min, .max = max, .interval = 0.005 }, .{
        .gravity_y = 0.5,
        .min_size_content = .{ .w = 160 },
    });
    v.* = std.math.clamp(v.*, min, max);
}

/// Each point's level stays strictly between its neighbours', so the list is
/// always one `Curve.init` would accept without re-sorting rows under the mouse.
fn curvePoints(c: *Curve) void {
    const gap: f32 = 0.001;
    var remove: ?usize = null;
    for (0..c.n) |k| {
        var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .id_extra = k, .margin = .{ .h = 2 } });
        defer row.deinit();
        const p = &c.points[k];
        const hi: f32 = if (k == 0) 1 else c.points[k - 1].level - gap;
        const lo: f32 = if (k + 1 == c.n) 0 else c.points[k + 1].level + gap;
        dvui.labelNoFmt(@src(), "level", .{}, .{ .font = F.ui, .color_text = .{ .color = C.text_dim }, .gravity_y = 0.5 });
        _ = dvui.sliderEntry(@src(), "{d:0.3}", .{ .value = &p.level, .min = lo, .max = hi, .interval = 0.005 }, .{
            .gravity_y = 0.5,
            .min_size_content = .{ .w = 110 },
        });
        p.level = std.math.clamp(p.level, lo, hi);
        dvui.labelNoFmt(@src(), "gain", .{}, .{ .font = F.ui, .color_text = .{ .color = C.text_dim }, .gravity_y = 0.5, .margin = .{ .x = 8 } });
        _ = dvui.sliderEntry(@src(), "{d:0.2}", .{ .value = &p.gain, .min = 0, .max = 1, .interval = 0.05 }, .{
            .gravity_y = 0.5,
            .min_size_content = .{ .w = 90 },
        });
        if (c.n > 1 and bubbles.secondaryButton(@src(), k, "Remove", true)) remove = k;
    }
    if (remove) |k| {
        std.mem.copyForwards(Curve.Point, c.points[k .. c.n - 1], c.points[k + 1 .. c.n]);
        c.n -= 1;
    }
    // A new point goes halfway down the last span, or below the last point.
    const last = c.points[c.n - 1];
    const room = c.n < Curve.max_points and last.level > 2 * gap;
    var add_row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .margin = .{ .h = 4 } });
    defer add_row.deinit();
    if (bubbles.secondaryButton(@src(), 99, "Add point", room)) {
        c.points[c.n] = .{ .level = last.level / 2, .gain = last.gain };
        c.n += 1;
    }
}

test "a schedule's chip says which steps it reaches" {
    var buf: [64]u8 = undefined;
    const levels = [_]f32{ 1.0, 0.9, 0.76, 0.51, 0.3, 0.1 };
    try std.testing.expectEqualStrings("every step", summary(&buf, .always, &levels));
    const first_two: Schedule = .{ .window = .{ .hi = 1, .lo = defaultWindowLo(&levels) } };
    try std.testing.expectEqualStrings("steps 1-2 of 6", summary(&buf, first_two, &levels));
    const one: Schedule = .{ .window = .{ .hi = 0.8, .lo = 0.7 } };
    try std.testing.expectEqualStrings("steps 3 of 6", summary(&buf, one, &levels));
    const none: Schedule = .{ .window = .{ .hi = 0.2, .lo = 0.15 } };
    try std.testing.expectEqualStrings("no steps", summary(&buf, none, &levels));
    // A curve that dips to zero and comes back is two runs.
    const dip: Schedule = .{ .curve = try Curve.init(&.{
        .{ .level = 0.9, .gain = 1 },
        .{ .level = 0.76, .gain = 0 },
        .{ .level = 0.51, .gain = 0 },
        .{ .level = 0.3, .gain = 1 },
    }) };
    try std.testing.expectEqualStrings("steps 1-2, 5-6 of 6", summary(&buf, dip, &levels));
    try std.testing.expectEqualStrings("window", summary(&buf, first_two, &.{}));
}
