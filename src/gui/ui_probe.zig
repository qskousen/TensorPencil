//! Visual probe for the chat workspace: renders the whole screen from canned
//! data into a PNG, with no model, no GPU and no engine.
//! `zig build ui-probe -- out.png [width height]`.
//!
//! Exists because every failure mode of this screen is visual — a face that
//! resolved to the wrong weight, a band that stole a rail's width, a hairline
//! that vanished over one surface, a tofu box where a mark should be — and
//! none of them raise an error. Driving the real app to see them needs a
//! loaded checkpoint and several gigabytes of VRAM.
//!
//! The canned data is deliberately awkward: a CJK conversation title in a
//! fixed-width rail, a long prose paragraph, a queue mid-generation, an
//! unloaded engine. Those are the cases that break layout, so they are the
//! ones worth having in the picture.
const std = @import("std");
const dvui = @import("dvui");
const Backend = @import("backend");
const style = @import("style.zig");
const fonts = @import("fonts.zig");
const shell = @import("shell.zig");
const bubbles = @import("bubbles.zig");
const queue_rail = @import("queue_rail.zig");
const meter = @import("meter.zig");

pub const panic = dvui.App.panic;
pub const std_options: std.Options = .{ .logFn = dvui.App.logFn };

const C = style.C;
const F = style.F;
const L = style.Layout;

// ------------------------------------------------------------- canned state

var g_tab: shell.Tab = .chat;
var g_rail_tab: queue_rail.Tab = .queue;
var g_expanded: bool = true;
var g_selected_tile: ?usize = null;
var g_input: [256]u8 = [_]u8{0} ** 256;

const conv_today = [_]shell.ConvRow{
    .{ .id = 1, .title = "Lighthouse in fog, 4 looks", .sub = "2 studio edits" },
    .{ .id = 2, .title = "Ceramic mug product shots" },
    // A CJK title in a 206px rail: ideographs are full-width, so this is where
    // a character-count budget silently truncates and a width budget does not.
    .{ .id = 3, .title = "日本語のタイトル、折り返しの確認" },
};
const conv_earlier = [_]shell.ConvRow{
    .{ .id = 4, .title = "Isometric shop fronts" },
};
const groups = [_]shell.ConvGroup{
    .{ .head = "TODAY", .rows = &conv_today },
    .{ .head = "EARLIER", .rows = &conv_earlier },
};

// Mid-handoff: two have landed, two are still in the queue. This is the state
// worth having in the picture, because it is the one where the card and the
// rail have to agree about where each image lives.
const tiles = [_]bubbles.Tile{ .pending, .pending, .pending, .pending };

const jobs = [_]queue_rail.Job{
    .{
        .id = 1,
        .title = "Lighthouse · 4.2 cfg",
        .thumb = .loading,
        .state = .{ .running = .{ .step = 18, .steps = 34, .it_s = 4.2 } },
    },
    .{
        .id = 2,
        .title = "Lighthouse · taller crop",
        .thumb = .dashed,
        .state = .{ .queued = .{ .eta_s = 11 } },
        .from_studio = true,
    },
};

const library = [_]queue_rail.LibraryItem{
    .{ .id = 10 }, .{ .id = 11 }, .{ .id = 12 },
    .{ .id = 13 }, .{ .id = 14 }, .{ .id = 15 },
};

const quick = [_]bubbles.QuickSetting{
    .{ .label = "3:2" },
    .{ .label = "1 MP" },
    .{ .label = "thinking: on" },
};

// ------------------------------------------------------------------ actions

fn noop() void {}
fn noopId(_: u64) void {}
fn noopCtx(_: *anyopaque) void {}
fn noopCtxIdx(_: *anyopaque, _: usize) void {}
fn onTab(t: shell.Tab) void {
    g_tab = t;
}
fn onRailTab(t: queue_rail.Tab) void {
    g_rail_tab = t;
}
fn onToggle(_: *anyopaque) void {
    g_expanded = !g_expanded;
}
fn onSelectTile(_: *anyopaque, i: usize) void {
    g_selected_tile = i;
}
fn onReorder(_: usize, _: usize) void {}

var g_ctx: u8 = 0;

// -------------------------------------------------------------------- frame

fn frame() void {
    var root = dvui.box(@src(), .{ .dir = .vertical }, .{
        .expand = .both,
        .background = true,
        .color_fill = C.canvas,
    });
    defer root.deinit();

    const bands = shell.Bands.from(root);

    shell.titleBar(.{
        .tab = g_tab,
        .diff_model = "sdxl-turbo · fp16",
        .residents = "qwen-7b + sdxl-turbo",
    }, .{ .on_tab = onTab, .on_model_menu = noop });

    {
        var body = dvui.box(@src(), .{ .dir = .horizontal }, .{
            .expand = .horizontal,
            .min_size_content = .{ .h = bands.body },
            .max_size_content = .height(bands.body),
        });
        defer body.deinit();

        shell.sidebar(.{
            .groups = &groups,
            .selected = 1,
            .models_pct = 0.73,
        }, .{
            .on_new_chat = noop,
            .on_select = noopId,
            .on_delete = noopId,
            .on_models = noop,
            .on_settings = noop,
        });

        chatColumn(bands.body);

        queue_rail.render(.{
            .tab = g_rail_tab,
            .jobs = &jobs,
            .library = &library,
        }, .{
            .on_tab = onRailTab,
            .on_pause_all = noop,
            .on_open = noopId,
            .on_cancel = noopId,
            .on_reorder = onReorder,
        });
    }

    statusBar(states[0], 0);
}

fn chatColumn(h: f32) void {
    // Width COMPUTED from the band, not left to the box layout: a child's min
    // size propagates up, so one long unbroken line grows the column and
    // squeezes the rails (spec §11 understates this).
    const band_w = dvui.parentGet().data().contentRect().w;
    const col_w = @max(240, band_w - L.sidebar_w - L.rail_w);
    var col = dvui.box(@src(), .{ .dir = .vertical }, .{
        .expand = .vertical,
        .min_size_content = .{ .w = col_w },
        .max_size_content = .width(col_w),
    });
    defer col.deinit();

    _ = h;
    const inner = col.data().contentRect().h;
    const composer_h: f32 = 118;
    {
        var sc = dvui.scrollArea(@src(), .{ .horizontal = .none }, .{
            .expand = .horizontal,
            .min_size_content = .{ .h = @max(80, inner - composer_h) },
            .max_size_content = .height(@max(80, inner - composer_h)),
            .background = false,
        });
        defer sc.deinit();

        var t = dvui.box(@src(), .{ .dir = .vertical }, .{
            .expand = .horizontal,
            .padding = .{ .x = 30, .y = 24, .w = 30, .h = 24 },
        });
        defer t.deinit();

        bubbles.userBubble(@src(), "A lighthouse in heavy fog at dawn. Quiet, a little melancholy. Four variations.");
        gap(@src(), 0, 18);
        bubbles.agentProse(@src(), "Four takes coming up — desaturated, light source behind the fog so the beam reads as glow.");
        gap(@src(), 1, 18);

        // The section the reply carries at the point the model emitted the call
        // (app.renderReply draws this inline); the card follows it.
        if (dvui.expander(@src(), "Tool call", .{ .default_expanded = true }, .{ .margin = .{ .y = 6 } })) {
            var tl = dvui.textLayout(@src(), .{}, .{
                .expand = .horizontal,
                .background = true,
                .color_fill = C.sunken,
                .color_border = style.tint(C.blue, 120),
                .border = .{ .x = 2 },
                .corner_radius = .{ .x = 0, .y = 6, .w = 6, .h = 0 },
                .margin = .{ .h = 6 },
                .padding = .{ .x = 11, .y = 8, .w = 11, .h = 8 },
                .font = F.mono,
                .color_text = C.text_dim,
            });
            defer tl.deinit();
            fonts.addStyled(tl, "<image width=\"1216\" height=\"832\" steps=\"34\" seed=\"8812\">a lighthouse in heavy fog at dawn, desaturated, backlit beam, melancholy</image>", .{}, .{ .font = F.mono, .color_text = C.text_dim });
        }

        bubbles.toolCard(@src(), .{
            .meta = "×4 · 1216×832 · seed 8812",
            .tiles = &tiles,
            .selected = g_selected_tile,
            .prompt = "a lighthouse in heavy fog at dawn, desaturated, backlit beam, melancholy",
            .expanded = g_expanded,
            .busy = true,
            .status = "rendering 1 of 4 · in queue",
        }, .{
            .ctx = @ptrCast(&g_ctx),
            .on_toggle = onToggle,
            .on_select = onSelectTile,
            .on_open_studio = noopCtx,
        });
        gap(@src(), 2, 18);

        bubbles.agentProse(@src(), "They are rendering now — you will see them fill in here as each one lands.");
    }

    composer(composer_h);
}

fn composer(h: f32) void {
    var box = dvui.box(@src(), .{ .dir = .vertical }, .{
        .expand = .horizontal,
        .min_size_content = .{ .h = h },
        .max_size_content = .height(h),
        .background = true,
        .color_fill = C.canvas,
        .border = style.Edge.top,
        .color_border = style.hairline_soft,
        .padding = .{ .x = 30, .y = 12, .w = 30, .h = 18 },
    });
    defer box.deinit();

    const cm: bubbles.Composer = .{ .quick = &quick };
    const cb: bubbles.ComposerActions = .{
        .ctx = @ptrCast(&g_ctx),
        .on_quick = noopCtxIdx,
        .on_all_settings = noopCtx,
        .on_reference = noopCtx,
    };
    bubbles.quickRow(@src(), cm, cb);

    var f = bubbles.inputBegin(@src(), false);
    var te = dvui.textEntry(@src(), .{
        .text = .{ .buffer = &g_input },
        .multiline = true,
        .placeholder = cm.placeholder,
        .break_lines = true,
        .scroll_horizontal = false,
    }, .{
        .expand = .horizontal,
        .gravity_y = 0.5,
        .background = false,
        .border = .{},
        .padding = .{},
        .font = F.input,
        .color_text = C.text_hi,
        .theme = style.noFocusTheme(),
        .min_size_content = .{ .h = 20 },
        .max_size_content = .size(.{ .w = 160, .h = 90 }),
    });
    te.deinit();
    _ = bubbles.inputEnd(&f, cm, cb);
}

fn gap(src: std.builtin.SourceLocation, id: usize, h: f32) void {
    _ = dvui.spacer(src, .{ .id_extra = id, .min_size_content = .{ .h = h } });
}

// ---------------------------------------------------------------- status bar

/// A canned VRAM state. Byte counts are the mockup's, so a screenshot can be
/// held next to it.
const VramState = struct {
    label: []const u8,
    gpu: f32,
    cpu: f32,
    sys: u64,
    llm_w: u64,
    llm_ctx: u64,
    ctx_tokens: usize,
    ovh: u64,
    te: u64,
    dit: u64,
    vae: u64,
    lat: u64,
    llm_loaded: bool = true,
    limit: f32 = 0.93,
    /// Bytes NOT on the card: LLM layer weights running on the CPU, pipeline
    /// weights streaming from host RAM. 0 = that side is fully resident.
    llm_host: u64 = 0,
    llm_host_layers: usize = 0,
    llm_layers: usize = 0,
    diff_off: u64 = 0,
};

const gb: u64 = 1 << 30;
const gbf = @as(f64, @floatFromInt(gb));

fn g(x: f64) u64 {
    return @intFromFloat(x * gbf);
}

const states = [_]VramState{
    .{
        .label = "CHAT · BOTH RESIDENT",
        .gpu = 0.63,
        .cpu = 0.11,
        .sys = g(2.4),
        .llm_w = g(3.6),
        .llm_ctx = g(0.8),
        .ctx_tokens = 6000,
        .ovh = g(0.4),
        .te = g(3.3),
        .dit = g(7.9),
        .vae = g(0.8),
        .lat = g(0.1),
    },
    .{
        // The unloaded engine states only what is resident: no ghost slot, no
        // dashed placeholder, the gap simply widens.
        .label = "STUDIO · LLM NOT LOADED",
        .gpu = 0.14,
        .cpu = 0.06,
        .sys = g(2.4),
        .llm_w = 0,
        .llm_ctx = 0,
        .ctx_tokens = 0,
        .ovh = 0,
        .te = g(3.3),
        .dit = g(7.9),
        .vae = g(0.8),
        .lat = g(0.1),
        .llm_loaded = false,
    },
    .{
        .label = "CHAT · LONG CONTEXT, LARGE BATCH",
        .gpu = 0.96,
        .cpu = 0.18,
        .sys = g(2.4),
        .llm_w = g(3.6),
        .llm_ctx = g(3.4),
        .ctx_tokens = 28000,
        .ovh = g(0.4),
        .te = g(1.1),
        .dit = g(10.8),
        .vae = g(1.0),
        .lat = g(0.3),
    },
    .{
        // Both sides squeezed: the bar looks the same as a healthy card (it can
        // only ever show what IS resident), and the totals are the only place the
        // difference shows. That is the case this state exists to check.
        .label = "CHAT · BOTH PARTLY OFFLOADED",
        .gpu = 0.71,
        .cpu = 0.64,
        .sys = g(2.4),
        .llm_w = g(6.2),
        .llm_ctx = g(1.4),
        .ctx_tokens = 12000,
        .ovh = g(0.4),
        .te = 0,
        .dit = g(5.9),
        .vae = g(0.8),
        .lat = g(0.2),
        .llm_host = g(2.4),
        .llm_host_layers = 9,
        .llm_layers = 48,
        .diff_off = g(4.7),
    },
};

var g_split_h: f32 = 0.42;
var g_limit_h: f32 = 0.93;

fn meterActions() meter.Actions {
    return .{
        .on_change = noop,
        .on_commit = noop,
        .on_eject_llm = noop,
        .on_eject_diff = noop,
        .on_toggle_pause_llm = noop,
        .on_toggle_pause_diff = noop,
    };
}

/// A deterministic pseudo-history so the sparklines look like real telemetry
/// without a clock. Varies per meter index and settles near `level`.
fn history(out: []f32, level: f32, seed: usize) []const f32 {
    for (out, 0..) |*v, i| {
        const t: f32 = @floatFromInt((i * 37 + seed * 101) % 23);
        v.* = std.math.clamp(level * (0.55 + t / 23.0 * 0.75), 0.02, 1.0);
    }
    return out;
}

fn statusBar(st: VramState, id: usize) void {
    var bar = dvui.box(@src(), .{ .dir = .horizontal }, .{
        .id_extra = id,
        .expand = .horizontal,
        .min_size_content = .{ .h = L.status_h },
        .max_size_content = .height(L.status_h),
        .background = true,
        .color_fill = C.chrome,
        .border = style.Edge.top,
        .color_border = style.hairline,
        .padding = .{ .x = 14, .y = 9, .w = 14, .h = 10 },
    });
    defer bar.deinit();

    var h1: [26]f32 = undefined;
    var h2: [26]f32 = undefined;
    var h3: [26]f32 = undefined;
    const total_b = g(24);
    const used = st.sys + st.llm_w + st.llm_ctx + st.ovh + st.te + st.dit + st.vae + st.lat;
    const vfrac: f32 = @floatCast(@as(f64, @floatFromInt(used)) / @as(f64, @floatFromInt(total_b)));
    const vcol = if (vfrac > 0.95) C.danger else if (vfrac >= 0.80) C.amber else C.meter_vram;

    style.historyMeter(@src(), 0, "GPU", st.gpu, history(&h1, st.gpu, id), C.meter_gpu);
    style.historyMeter(@src(), 1, "CPU", st.cpu, history(&h2, st.cpu, id + 5), C.meter_cpu);
    style.historyMeter(@src(), 2, "VRAM", vfrac, history(&h3, vfrac, id + 9), vcol);
    style.vsep(@src());

    var mm: meter.Model = .{
        .total = total_b,
        .system = st.sys,
        .overhead = st.ovh,
        .llm_w = st.llm_w,
        .llm_ctx = st.llm_ctx,
        .ctx_tokens = st.ctx_tokens,
        .llm_host = st.llm_host,
        .llm_host_layers = st.llm_host_layers,
        .llm_layers = st.llm_layers,
        .diff_off = st.diff_off,
        .te = st.te,
        .dit = st.dit,
        .latent = st.lat,
        .vae = st.vae,
        .split = &g_split_h,
        .limit = &g_limit_h,
        .floor_llm = 0.04,
        .floor_diff = 0.04,
        .llm_loaded = st.llm_loaded,
        .diff_loaded = true,
        .llm_armed = false,
        .diff_armed = false,
        .llm_paused = false,
        .diff_paused = false,
    };
    meter.render(&mm, meterActions());
}

/// The status-bar state sheet: the same bar under three different loads,
/// stacked, so the unloaded case and the pressure thresholds can be compared
/// side by side rather than reasoned about.
fn statesFrame() void {
    var root = dvui.box(@src(), .{ .dir = .vertical }, .{
        .expand = .both,
        .background = true,
        .color_fill = C.canvas,
    });
    defer root.deinit();

    for (states, 0..) |st, i| {
        style.sectionHead(@src(), st.label, .{ .id_extra = i, .padding = .{ .x = 14, .y = 14, .h = 8 } });
        statusBar(st, i);
    }
}

// --------------------------------------------------------------------- main

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const arena = init.arena.allocator();
    Backend.c.SDL_SetMainReady();

    const args = try init.minimal.args.toSlice(arena);
    var out_path: []const u8 = "ui_probe.png";
    var states_mode = false;
    var dims: [2]?u32 = .{ null, null };
    var seen_out = false;
    for (args[1..]) |arg| {
        if (std.mem.eql(u8, arg, "--states")) {
            states_mode = true;
        } else if (std.fmt.parseInt(u32, arg, 10)) |n| {
            if (dims[0] == null) dims[0] = n else dims[1] = n;
        } else |_| {
            if (!seen_out) {
                out_path = arg;
                seen_out = true;
            }
        }
    }
    // The mockups' own canvases, so a screenshot can be held next to them.
    const w: u32 = dims[0] orelse 1200;
    // The state sheet's height follows the number of states (head + bar each).
    const h: u32 = dims[1] orelse (if (states_mode) @as(u32, 126 * states.len) else 705);

    var back = try Backend.initWindow(.{
        .io = init.io,
        .allocator = gpa,
        .size = .{ .w = @floatFromInt(w), .h = @floatFromInt(h) },
        .vsync = false,
        .title = "ui-probe",
        .hidden = true,
        .environ_map = init.environ_map,
    });
    defer back.deinit();

    var win = try dvui.Window.init(@src(), gpa, back.backend(), .{});
    defer win.deinit();

    try win.begin(win.frame_time_ns);
    style.install();
    _ = try win.end(.{});

    // Text layouts report their min size a frame late, so an early capture
    // catches the screen mid-settle.
    // A synthetic pointer parked on the second conversation row, so the hover
    // fill and the delete affordance are in the captured frame. Hover is read
    // from the previous frame, so the event has to be present on every settle
    // pass, not just the last.
    const hover_pt: dvui.Point.Physical = .{ .x = 110, .y = 187 };
    for (0..5) |i| {
        try win.begin(win.frame_time_ns + @as(i128, @intCast(i + 1)) * 16 * std.time.ns_per_ms);
        _ = try win.addEventMouseMotion(.{ .pt = hover_pt });
        style.install();
        if (states_mode) statesFrame() else frame();
        _ = try win.end(.{});
    }

    try win.begin(win.frame_time_ns + 6 * 16 * std.time.ns_per_ms);
    _ = try win.addEventMouseMotion(.{ .pt = hover_pt });
    style.install();
    var pic = dvui.Picture.start(dvui.windowRectPixels()) orelse return error.CaptureUnsupported;
    if (states_mode) statesFrame() else frame();
    _ = dvui.currentWindow().endRendering(.{});
    pic.stop();
    var aw: std.Io.Writer.Allocating = try .initCapacity(gpa, 1 << 20);
    defer aw.deinit();
    try pic.png(&aw.writer);
    pic.deinit();
    _ = try win.end(.{});

    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = out_path, .data = aw.writer.buffered() });
    std.debug.print("ui-probe: wrote {s} ({d}×{d})\n", .{ out_path, w, h });
}
