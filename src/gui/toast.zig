//! Transient notices over the workspace: something happened that the user did
//! not ask for and cannot read anywhere else.
//!
//! The case this exists for is a render that failed on one host. Its queue row
//! is gone and there is nothing left to click, so without a word on screen the
//! image simply vanished. A notice says what failed, why, and what was done
//! about it.
//!
//! dvui owns the stack, the timer and the removal. This owns the wording, the
//! palette and how long a notice stays: `display` is the whole look.
const std = @import("std");
const dvui = @import("dvui");
const style = @import("style.zig");
const fonts = @import("fonts.zig");

const C = style.C;

pub const Tone = enum(u8) {
    /// Something finished or moved. Nothing is wrong.
    info,
    /// It worked, but not the way it was asked.
    warn,
    /// It did not happen.
    err,

    fn color(self: Tone) dvui.Color {
        return switch (self) {
            .info => C.blue,
            .warn => C.amber,
            .err => C.danger,
        };
    }
};

/// Long enough to read a sentence and look back. This is the only place a
/// failure is stated, so it does not flash past; a notice can also be
/// dismissed, which is what a user who has read it does.
const dwell_us: i32 = 14_000_000;

/// Longest notice kept. Past this the text is cut, which is better than a
/// notice that covers the picture it is about.
pub const max_text = 240;

var next_slot: usize = 0;

/// Notices waiting for a frame. dvui's toast list belongs to the window being
/// drawn, and most of what is worth saying is learned while pumping the hosts,
/// which happens BETWEEN frames: handing it to dvui there panics. So `post`
/// only writes here, and `pump` hands them over once drawing has begun.
const Pending = struct {
    tone: Tone = .info,
    len: usize = 0,
    buf: [max_text]u8 = @splat(0),
};
var queued: [8]Pending = @splat(.{});
var n_queued: usize = 0;

/// Put a notice up. Callable from anywhere on the frame thread, inside a frame
/// or between two.
pub fn post(tone: Tone, comptime fmt: []const u8, args: anytype) void {
    var buf: [max_text]u8 = undefined;
    // A fixed writer, not `bufPrint`: overrunning the buffer leaves what fit
    // (which is the cut this file promises), where `bufPrint`'s error hands
    // back the whole array with the tail still uninitialised.
    var w = std.Io.Writer.fixed(&buf);
    w.print(fmt, args) catch {};
    postText(tone, w.buffered());
}

pub fn postText(tone: Tone, text: []const u8) void {
    // Past the ring the oldest waiting notice goes: a burst is a burst, and
    // the newest is the one still worth reading.
    if (n_queued == queued.len) {
        std.mem.copyForwards(Pending, queued[0 .. queued.len - 1], queued[1..]);
        n_queued -= 1;
    }
    const n = @min(text.len, max_text);
    queued[n_queued] = .{ .tone = tone, .len = n };
    @memcpy(queued[n_queued].buf[0..n], text[0..n]);
    n_queued += 1;
}

/// Hand every waiting notice to dvui. Call once per frame, inside
/// `Window.begin`/`end`.
pub fn pump() void {
    for (queued[0..n_queued]) |*p| {
        // A fresh id per notice, or two in one frame are one notice.
        next_slot +%= 1;
        const im = dvui.toastAdd(null, @src(), next_slot, null, display, dwell_us);
        dvui.dataSetSlice(null, im.id, "_message", p.buf[0..p.len]);
        dvui.dataSet(null, im.id, "_tone", @intFromEnum(p.tone));
        im.mutex.unlock(dvui.io);
    }
    n_queued = 0;
}

fn display(id: dvui.Id) !void {
    const msg = dvui.dataGetSlice(null, id, "_message", []u8) orelse {
        dvui.toastRemove(id);
        return;
    };
    const tone: Tone = @enumFromInt(dvui.dataGet(null, id, "_tone", u8) orelse 0);

    var animator = dvui.animate(@src(), .{ .kind = .alpha, .duration = 300_000 }, .{
        .id_extra = id.asUsize(),
        .gravity_x = 0.5,
    });
    defer animator.deinit();

    var row = dvui.box(@src(), .{ .dir = .horizontal }, .{
        .background = true,
        .color_fill = C.raised,
        .corner_radius = style.R.card,
        .border = dvui.Rect.all(1),
        .color_border = tone.color(),
        .padding = .{ .x = 12, .y = 8, .w = 8, .h = 8 },
        .margin = .{ .x = 8, .y = 4, .w = 8, .h = 4 },
    });
    defer row.deinit();

    // A notice is model- and host-named text, so it goes through the run
    // splitter like everything else a user reads.
    // Bounded, or the label takes the whole row and the dismiss leaves the
    // screen: a text layout given no ceiling expands to whatever it is offered.
    fonts.richLabel(@src(), msg, .{
        .gravity_y = 0.5,
        .color_text = C.text_hi,
        .max_size_content = .width(480),
    });

    // Read and done with. The mark goes through `style.mark` because no face
    // we bundle has the glyph.
    var x: dvui.ButtonWidget = undefined;
    x.init(@src(), .{}, .{
        .gravity_y = 0.5,
        .corner_radius = style.R.chip,
        .padding = dvui.Rect.all(3),
        .margin = .{ .x = 8 },
    });
    x.processEvents();
    x.drawBackground();
    style.mark(@src(), .close, 11, C.text_ghost, .{});
    const dismissed = x.clicked();
    x.deinit();
    if (dismissed) dvui.toastRemove(id);

    if (dvui.timerDone(id)) animator.startEnd();
    if (animator.end()) {
        dvui.toastRemove(id);
        // Claims no space on the frame after it goes.
        animator.data().min_size = .{};
    }
}
