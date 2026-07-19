//! Visual smoke probe for the chat markdown renderer: opens a hidden window,
//! renders a torture document through markdown_view.zig with the bundled
//! fonts, and writes a PNG screenshot. `zig build md-probe -- out.png`.
//!
//! Exists because the renderer's failure modes are visual (a bold face that
//! resolves to regular, a mono face that isn't mono, run splits at wrong
//! boundaries) — unit tests can't see them, and eyeballing via a live chat
//! needs a loaded model. Not part of any default build or test step.
const std = @import("std");
const dvui = @import("dvui");
const Backend = @import("backend");
const fonts = @import("fonts.zig");
const markdown_view = @import("markdown_view.zig");

pub const panic = dvui.App.panic;
pub const std_options: std.Options = .{ .logFn = dvui.App.logFn };

const doc =
    \\# Heading one
    \\## Heading two
    \\### Heading three
    \\
    \\A paragraph with **bold**, *italic*, ***bold italic***, `inline code`,
    \\~~strikethrough~~, a [markdown link](https://ziglang.org), a bare
    \\https://example.com/path, and snake_case_identifiers left alone.
    \\Math stays literal: 3 * 4 * 5.
    \\**Bold with `code inside` it**; an italic [link](https://e.com/it) too.
    \\
    \\CJK with emphasis: 日本語のテキスト、太字は**こう**、コードは`こう`。
    \\Emoji runs: ⚙ ✅ 🦊 mixed into ordinary text.
    \\
    \\- bullet one
    \\- bullet two with `code`
    \\  - nested bullet
    \\      - deep nested (indent ≥ 6 once panicked: u2 @min narrowing)
    \\1. ordered first
    \\2. ordered second
    \\
    \\> A blockquote line,
    \\> and its second line.
    \\
    \\---
    \\
    \\```zig
    \\// full-width card, mono face, copy button
    \\pub fn main(init: std.process.Init) !void {
    \\    std.debug.print("日本語 in code = {d}\n", .{42});
    \\}
    \\```
    \\
    \\After the fence, selection flows through everything above in one layout.
    \\Streaming tail (no closer follows): **unclosed stays literal.
;

fn frame() void {
    var scroll = dvui.scrollArea(@src(), .{}, .{
        .expand = .both,
        .background = true,
        .color_fill = dvui.themeGet().fill,
    });
    defer scroll.deinit();
    markdown_view.render(@src(), doc, .{ .prose = .{ .padding = dvui.Rect.all(8) } });
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const arena = init.arena.allocator();
    Backend.c.SDL_SetMainReady();

    const args = try init.minimal.args.toSlice(arena);
    const out_path: []const u8 = if (args.len > 1) args[1] else "md_probe.png";

    var back = try Backend.initWindow(.{
        .io = init.io,
        .allocator = gpa,
        .size = .{ .w = 720, .h = 1080 },
        .vsync = false,
        .title = "md-probe",
        .hidden = true,
        .environ_map = init.environ_map,
    });
    defer back.deinit();

    var win = try dvui.Window.init(@src(), gpa, back.backend(), .{});
    defer win.deinit();

    try win.begin(win.frame_time_ns);
    fonts.install();
    _ = try win.end(.{});

    // A few settle frames (text layouts report min sizes a frame late), then
    // capture the last one. The synthetic mouse hover sits on the code card's
    // copy button so its tooltip shows in the capture.
    for (0..4) |_| {
        try win.begin(win.frame_time_ns + 16 * std.time.ns_per_ms);
        _ = try win.addEventMouseMotion(.{ .pt = .{ .x = 704, .y = 643 } });
        frame();
        _ = try win.end(.{});
    }

    try win.begin(win.frame_time_ns + 16 * std.time.ns_per_ms);
    _ = try win.addEventMouseMotion(.{ .pt = .{ .x = 704, .y = 643 } });
    var pic = dvui.Picture.start(dvui.windowRectPixels()) orelse return error.CaptureUnsupported;
    frame();
    _ = dvui.currentWindow().endRendering(.{});
    pic.stop();
    var aw: std.Io.Writer.Allocating = try .initCapacity(gpa, 1 << 20);
    defer aw.deinit();
    try pic.png(&aw.writer);
    pic.deinit();
    _ = try win.end(.{});

    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = out_path, .data = aw.writer.buffered() });
    std.debug.print("md-probe: wrote {s}\n", .{out_path});
}
