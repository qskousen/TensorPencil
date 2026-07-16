//! Full-size image viewer — a second OS window with zoom, pan, and keyboard
//! navigation between all images in the conversation. Zoom/pan mechanics are
//! ported from DiffKeep's image_viewer (renderImage at a computed physical
//! rect, wheel-zoom toward the cursor, drag-to-pan via mouse capture).
const std = @import("std");
const dvui = @import("dvui");
const SDLBackend = @import("backend");
const diffuser = @import("diffuser.zig");
const fonts = @import("fonts.zig");

pub const GenImage = diffuser.GenImage;

/// Where the viewer's navigable image list comes from, decoupled from any
/// particular driver: the chat transcript or the image studio's gallery both
/// provide one. `collect` fills `buf` with the done images in display order.
pub const ImageSource = struct {
    ctx: *anyopaque,
    gpa: std.mem.Allocator,
    collect: *const fn (ctx: *anyopaque, buf: *std.ArrayList(*GenImage)) void,
};

pub const Viewer = struct {
    gpa: std.mem.Allocator,
    back: SDLBackend,
    win: dvui.Window,
    win_id: u32,
    src: ImageSource,
    cur: *GenImage,
    open: bool = true,
    shown: bool = false,
    // Fonts/theme are per-window; install the broad-coverage font once so the
    // info bar's arrows/dots don't tofu in the default Latin-only font.
    fonts_ready: bool = false,
    zoom: f32 = 1.0,
    zoom_mode: ZoomMode = .fit,
    pan_x: f32 = 0,
    pan_y: f32 = 0,

    const ZoomMode = enum { fit, pct100, custom };

    /// Heap-allocated and built in place: `dvui.Window` captures a pointer to
    /// the `SDLBackend` (via `back.backend()`), so the backend must live at a
    /// stable address — a by-value return would leave the window pointing at a
    /// dead copy.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, src: ImageSource, cur: *GenImage) !*Viewer {
        const self = try gpa.create(Viewer);
        errdefer gpa.destroy(self);
        self.gpa = gpa;
        self.back = try SDLBackend.initWindow(.{
            .io = io,
            .allocator = gpa,
            .size = .{ .w = 1000, .h = 760 },
            .min_size = .{ .w = 400, .h = 300 },
            .vsync = true,
            .title = "tp-gui - image",
            .hidden = true, // shown after the first frame is presented
        });
        errdefer self.back.deinit();
        self.win = try dvui.Window.init(@src(), gpa, self.back.backend(), .{ .id_extra = 1 });
        self.win_id = SDLBackend.c.SDL_GetWindowID(self.back.window);
        self.src = src;
        self.cur = cur;
        self.open = true;
        self.shown = false;
        self.fonts_ready = false;
        self.zoom = 1.0;
        self.zoom_mode = .fit;
        self.pan_x = 0;
        self.pan_y = 0;
        return self;
    }

    pub fn deinit(self: *Viewer) void {
        const gpa = self.gpa;
        self.win.deinit();
        // Destroy only this window's renderer/window, then suppress the SDL_Quit
        // in back.deinit: SDL3 doesn't refcount SDL_Init/SDL_Quit, so quitting
        // here would tear down the video subsystem out from under the main
        // window. Only the main window's back.deinit SDL_Quits, at exit.
        SDLBackend.c.SDL_DestroyRenderer(self.back.renderer);
        SDLBackend.c.SDL_DestroyWindow(self.back.window);
        self.back.we_own_window = false;
        self.back.deinit();
        gpa.destroy(self);
    }

    /// Point the viewer at a different image and reset the view.
    pub fn setImage(self: *Viewer, gi: *GenImage) void {
        self.cur = gi;
        self.resetView();
    }

    fn resetView(self: *Viewer) void {
        self.zoom_mode = .fit;
        self.pan_x = 0;
        self.pan_y = 0;
    }

    fn nav(self: *Viewer, dir: i64) void {
        var buf: std.ArrayList(*GenImage) = .empty;
        defer buf.deinit(self.src.gpa);
        self.src.collect(self.src.ctx, &buf);
        if (buf.items.len == 0) return;
        var idx: usize = 0;
        for (buf.items, 0..) |gi, i| if (gi == self.cur) {
            idx = i;
            break;
        };
        const ni = @as(i64, @intCast(idx)) + dir;
        if (ni < 0 or ni >= @as(i64, @intCast(buf.items.len))) return;
        self.cur = buf.items[@intCast(ni)];
        self.resetView();
    }

    fn navTo(self: *Viewer, last: bool) void {
        var buf: std.ArrayList(*GenImage) = .empty;
        defer buf.deinit(self.src.gpa);
        self.src.collect(self.src.ctx, &buf);
        if (buf.items.len == 0) return;
        self.cur = buf.items[if (last) buf.items.len - 1 else 0];
        self.resetView();
    }

    /// The viewer's frame content (called between win.begin/win.end). Handles
    /// keyboard nav + zoom, then draws the dark image area with the current
    /// image scaled/panned.
    pub fn render(self: *Viewer) void {
        // This window's own font DB/theme (dvui is per-window); install once so
        // the info bar renders arrows/dots instead of tofu.
        if (!self.fonts_ready) {
            fonts.install() catch |err| std.log.err("viewer font install: {t}", .{err});
            self.fonts_ready = true;
        }
        self.handleKeys();

        // Position/index label (which image, current zoom).
        var buf: std.ArrayList(*GenImage) = .empty;
        defer buf.deinit(self.src.gpa);
        self.src.collect(self.src.ctx, &buf);
        var idx: usize = 0;
        for (buf.items, 0..) |gi, i| if (gi == self.cur) {
            idx = i;
            break;
        };

        var col = dvui.box(@src(), .{ .dir = .vertical }, .{ .expand = .both, .background = true });
        defer col.deinit();

        {
            var bar = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .padding = dvui.Rect.all(6) });
            defer bar.deinit();
            dvui.label(@src(), "image {d}/{d}   {d:.0}%   (← → navigate · scroll zoom · drag pan · 0 fit · 1 100% · Esc close)", .{
                idx + 1, buf.items.len, self.zoom * 100,
            }, .{ .gravity_y = 0.5 });
        }

        self.renderImageArea();
    }

    fn handleKeys(self: *Viewer) void {
        for (dvui.events()) |*e| {
            if (e.handled or e.evt != .key) continue;
            const ke = e.evt.key;
            if (ke.action != .down and ke.action != .repeat) continue;
            switch (ke.code) {
                .left => {
                    e.handled = true;
                    self.nav(-1);
                },
                .right => {
                    e.handled = true;
                    self.nav(1);
                },
                .home => {
                    e.handled = true;
                    self.navTo(false);
                },
                .end => {
                    e.handled = true;
                    self.navTo(true);
                },
                .equal, .kp_add => {
                    e.handled = true;
                    self.zoom = std.math.clamp(self.zoom * 1.25, 0.05, 32.0);
                    self.zoom_mode = .custom;
                },
                .minus, .kp_subtract => {
                    e.handled = true;
                    self.zoom = std.math.clamp(self.zoom * 0.8, 0.05, 32.0);
                    self.zoom_mode = .custom;
                },
                .zero, .kp_0 => {
                    e.handled = true;
                    self.resetView();
                },
                .one, .kp_1 => {
                    e.handled = true;
                    self.zoom = 1.0;
                    self.zoom_mode = .pct100;
                    self.pan_x = 0;
                    self.pan_y = 0;
                },
                .escape => {
                    e.handled = true;
                    self.open = false;
                },
                else => {},
            }
        }
    }

    fn renderImageArea(self: *Viewer) void {
        var area = dvui.box(@src(), .{}, .{
            .expand = .both,
            .background = true,
            .color_fill = .{ .r = 18, .g = 18, .b = 18, .a = 255 },
        });
        defer area.deinit();

        const gi = self.cur;
        const rgba = gi.rgba orelse return;
        const iw = gi.width;
        const ih = gi.height;
        if (iw == 0 or ih == 0) return;

        const wd = area.data();
        const crs = wd.contentRectScale();

        if (self.zoom_mode == .fit) {
            const cw = crs.r.w / crs.s;
            const ch = crs.r.h / crs.s;
            self.zoom = @min(cw / @as(f32, @floatFromInt(iw)), ch / @as(f32, @floatFromInt(ih)));
        }

        self.handleAreaMouse(wd, crs);

        const s = crs.s;
        const img_w_p = @as(f32, @floatFromInt(iw)) * self.zoom * s;
        const img_h_p = @as(f32, @floatFromInt(ih)) * self.zoom * s;
        const cx = crs.r.x + crs.r.w / 2.0 + self.pan_x * self.zoom * s;
        const cy = crs.r.y + crs.r.h / 2.0 + self.pan_y * self.zoom * s;
        const img_rect: dvui.Rect.Physical = .{
            .x = cx - img_w_p / 2.0,
            .y = cy - img_h_p / 2.0,
            .w = img_w_p,
            .h = img_h_p,
        };
        const prev_clip = dvui.clipGet();
        dvui.clipSet(crs.r.intersect(prev_clip));
        dvui.renderImage(.{ .pixels = .{
            .rgba = rgba,
            .width = @intCast(iw),
            .height = @intCast(ih),
            .invalidation = .ptr,
        } }, .{ .r = img_rect, .s = s }, .{}) catch {};
        dvui.clipSet(prev_clip);
    }

    fn handleAreaMouse(self: *Viewer, wd: *dvui.WidgetData, crs: dvui.RectScale) void {
        const id = wd.id;
        for (dvui.events()) |*e| {
            if (e.handled or e.evt != .mouse) continue;
            const me = e.evt.mouse;
            switch (me.action) {
                .wheel_y => |dy| {
                    if (!dvui.eventMatchSimple(e, wd)) continue;
                    e.handled = true;
                    const s = crs.s;
                    const cx = crs.r.x + crs.r.w / 2.0 + self.pan_x * self.zoom * s;
                    const cy = crs.r.y + crs.r.h / 2.0 + self.pan_y * self.zoom * s;
                    const cur_x = (me.p.x - cx) / (self.zoom * s);
                    const cur_y = (me.p.y - cy) / (self.zoom * s);
                    const factor: f32 = if (dy > 0) 1.12 else (1.0 / 1.12);
                    const nz = std.math.clamp(self.zoom * factor, 0.05, 32.0);
                    self.pan_x += cur_x * (self.zoom - nz);
                    self.pan_y += cur_y * (self.zoom - nz);
                    self.zoom = nz;
                    self.zoom_mode = .custom;
                },
                .press => {
                    if (me.button.pointer() and dvui.eventMatchSimple(e, wd)) {
                        e.handled = true;
                        dvui.captureMouse(wd, e.num);
                        dvui.dragPreStart(me.p, .{});
                    }
                },
                .motion => |delta| {
                    if (dvui.captured(id) and dvui.dragging(me.p, null) != null) {
                        e.handled = true;
                        const s = crs.s;
                        self.pan_x += delta.x / (self.zoom * s);
                        self.pan_y += delta.y / (self.zoom * s);
                        self.zoom_mode = .custom;
                    }
                },
                .release => {
                    if (dvui.captured(id)) {
                        e.handled = true;
                        dvui.captureMouse(null, e.num);
                    }
                },
                else => {},
            }
        }
    }
};
