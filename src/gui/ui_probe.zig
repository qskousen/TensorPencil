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
const model_menu = @import("model_menu.zig");
const catalog = @import("shared").catalog;
const selection = @import("client").selection;
const models = @import("client").models;
const config = @import("shared").config;
const config_view = @import("config_view.zig");
const bubbles = @import("bubbles.zig");
const queue_rail = @import("queue_rail.zig");
const meter = @import("meter.zig");
const status_bar = @import("status_bar.zig");
const mirror = @import("client").mirror;
const wire = @import("serve").wire;
const image_view = @import("image_view.zig");
const toast = @import("toast.zig");

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
var g_noise_sel: usize = 1;
var g_noise_amt: [16]u8 = [_]u8{0} ** 16;
const noise_names = [_][]const u8{ "flat", "front", "front steep", "first 40%", "bell", "late (corrupts)" };
/// The front-loaded curve the field shows, sampled for the preview: high at the
/// first layer, zero at the LM head. Hardcoded rather than evaluated so the probe
/// stays a pure renderer.
const noise_shape = [_]f32{ 1.00, 0.90, 0.81, 0.72, 0.64, 0.56, 0.49, 0.42, 0.36, 0.30, 0.25, 0.20, 0.16, 0.12, 0.09, 0.06, 0.04, 0.02, 0.01, 0.00 };

fn noopPick(_: model_menu.Pick) void {}
fn noHostStatus(name: []const u8, _: *const config.HostEntry) config_view.HostState {
    if (std.mem.eql(u8, name, "lydia")) return .{ .text = "token refused: re-pair this host", .tone = .bad };
    if (std.mem.eql(u8, name, "cellar")) return .{ .text = "reached it, token accepted: Apply & Reload to use it", .tone = .pending };
    return .{ .text = "up", .tone = .ok };
}
fn noHostSync(name: []const u8) config_view.HostSync {
    // One file just landed and two more are still short: the state a host is
    // in halfway through being fed a model, and the one that used to read as
    // "done" with nothing left to do.
    if (std.mem.eql(u8, name, "lydia")) return .{
        .sending = "qwen3_06b_instruct",
        .status = "done",
        .missing = "krea2RealVae_v10",
        .missing_count = 2,
    };
    if (!std.mem.eql(u8, name, "attic")) return .{};
    return .{ .missing = "krea2CenterSemiraw_v10Int8", .missing_count = 1 };
}
fn noSendModel(_: []const u8) void {}
fn noSendPath(_: []const u8, _: []const u8) void {}
fn noPullPath(_: []const u8, _: []const u8, _: []const u8) void {}
fn noTryHost(_: *const config.HostEntry) void {}
fn noReconnectHost(_: []const u8) void {}

// The two title-bar menus, as the catalog would build them: a supported class
// with the current file checked, and a greyed class with its reason.
const probe_llm_items = [_]model_menu.Item{
    .{ .label = "Gemma-4-31B-it-Q4_K_M", .path = "/m/a.gguf", .selected = true, .note = "2 of 3" },
    .{ .label = "Gemma-4-Dark-Thoughts-31B.i1-Q4_K_S", .path = "/m/b.gguf", .note = "lydia" },
};
const probe_llm_grey = [_]model_menu.Item{
    .{ .label = "nomic-embed-text-v1.5.Q8_0", .path = "/m/c.gguf", .greyed = true, .note = "architecture 'nomic-bert' is not supported" },
};
const probe_llm_groups = [_]model_menu.Group{
    .{ .label = "Gemma 4 31B", .items = &probe_llm_items },
    .{ .label = "nomic-bert 137M", .items = &probe_llm_grey, .greyed = true },
};
const probe_llm_menu: model_menu.Menu = .{ .groups = &probe_llm_groups, .none_label = "no chat model" };
const probe_image_items = [_]model_menu.Item{
    .{ .label = "sdxl-turbo-fp16", .path = "/m/x.safetensors", .selected = true, .note = "rtxpro6k" },
    .{ .label = "dreamshaperXL10_alpha2Xl10", .path = "/m/y.safetensors", .note = "attic · missing files" },
    // Only a host that is down has it: still listed, greyed, and named.
    .{ .label = "kWALUAN_v08INT8_fp16", .path = "/m/z.safetensors", .greyed = true, .note = "only on cellar (down)" },
};
const probe_image_groups = [_]model_menu.Group{.{ .label = "SDXL", .items = &probe_image_items }};
const probe_image_menu: model_menu.Menu = .{ .groups = &probe_image_groups, .none_label = "no image model" };

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

// Mid-run: one landed, one rendering with a preview, one rendering with none
// yet, one still waiting for a host. Every tile state in one card, which is the
// picture worth having: the card is where a render is watched now, and the four
// look nothing alike.
var tiles = [_]bubbles.Tile{ .pending, .pending, .pending, .pending };

const jobs = [_]queue_rail.Job{
    .{
        .id = 2,
        .title = "Lighthouse · taller crop",
        .state = .{ .queued = .{ .eta_s = 11 } },
        .from_studio = true,
    },
    // A job that will not render keeps its row and says why; vanishing is what
    // this state exists to stop.
    .{
        .id = 3,
        .title = "Lighthouse · wide",
        .state = .{ .failed = .{ .why = "the checkpoint is missing a component (VAE or text encoder)" } },
        .host = "lydia",
    },
    // Still in the client's own queue, and stuck there: no host names it
    // because no host can run it, which is not the same as waiting a turn.
    .{
        .id = 4,
        .title = "Ceramic mug · turntable",
        .state = .{ .queued = .{ .note = "no host can run this yet" } },
    },
};

var library = [_]queue_rail.LibraryItem{
    .{ .id = 10, .host = "workshop" }, .{ .id = 11, .host = "lydia" }, .{ .id = 12, .host = "workshop" },
    .{ .id = 13, .host = "lydia" },    .{ .id = 14, .host = "workshop" }, .{ .id = 15, .host = "lydia" },
};

/// A gradient `w` x `h`, bright at the bottom where the tile's own strip and the
/// library's host tag sit: neither is worth having if it does not survive that.
fn cannedPixels(gpa: std.mem.Allocator, w: u32, h: u32) ![]u8 {
    const rgba = try gpa.alloc(u8, @as(usize, w) * h * 4);
    for (0..h) |y| for (0..w) |x| {
        const i = (y * w + x) * 4;
        rgba[i + 0] = @intCast(150 + (y * 105) / h);
        rgba[i + 1] = @intCast(120 + (x * 130) / w);
        rgba[i + 2] = @intCast(90 + ((x + y) * 60) / (w + h));
        rgba[i + 3] = 255;
    };
    return rgba;
}

/// The tool card's four tiles: landed, rendering with a preview, rendering with
/// none yet, and waiting. The preview is a fraction of the finished size and a
/// different shape from the tile, which is what the fit has to survive.
fn cannedTiles(gpa: std.mem.Allocator) !void {
    tiles[0] = .{ .rgba = .{ .px = try cannedPixels(gpa, 304, 208), .w = 304, .h = 208 } };
    tiles[1] = .{ .rendering = .{
        .px = .{ .px = try cannedPixels(gpa, 152, 104), .w = 152, .h = 104 },
        .step = 18,
        .steps = 34,
        .label = "step 18 / 34",
    } };
    tiles[2] = .{ .rendering = .{ .step = 0, .steps = 34, .label = "step 0 / 34 · lydia" } };
}

/// The same run after the conversation was reopened and two of the files have
/// gone: nothing is running, the gone ones are ghost slots that say so, and the
/// two that are left can be handed to a model that sees images. Nothing on this
/// screen may read as a failure or offer to render anything.
var g_reopened = false;

fn cannedReopened(gpa: std.mem.Allocator) !void {
    tiles[0] = .{ .rgba = .{ .px = try cannedPixels(gpa, 304, 208), .w = 304, .h = 208 } };
    tiles[1] = .{ .rgba = .{ .px = try cannedPixels(gpa, 304, 208), .w = 304, .h = 208 } };
    tiles[2] = .missing;
    tiles[3] = .missing;
}

/// Real pixels under half the library tiles.
fn cannedLibrary(gpa: std.mem.Allocator) !void {
    const w: u32 = 96;
    const h: u32 = 96;
    for (&library, 0..) |*item, k| {
        if (k % 2 == 1) continue;
        item.thumb = .{ .rgba = .{ .px = try cannedPixels(gpa, w, h), .w = w, .h = h } };
    }
}

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
fn noopCtxId(_: *anyopaque, _: usize) void {}
fn onReorder(_: u64, _: ?u64) void {}

var g_ctx: u8 = 0;

// -------------------------------------------------------------------- frame

fn frame() void {
    toast.pump();
    var root = dvui.box(@src(), .{ .dir = .vertical }, .{
        .expand = .both,
        .background = true,
        .color_fill = C.canvas,
    });
    defer root.deinit();

    const bands = shell.Bands.from(root, status_bar.bar_outer_height);

    shell.titleBar(.{
        .tab = g_tab,
        .llm = .{ .label = "Gemma-4-31B-it-Q4_K_M", .note = "2 of 3", .resident = true },
        .llm_menu = probe_llm_menu,
        .image = .{ .label = "sdxl-turbo-fp16", .note = "rtxpro6k", .warn = true },
        .image_menu = probe_image_menu,
    }, .{ .on_tab = onTab, .on_llm_pick = noopPick, .on_image_pick = noopPick, .on_settings = noop });

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
            .on_open_library = noopId,
            .on_cancel = noopId,
            .on_retry = noopId,
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
            .aspect = 1216.0 / 832.0,
            .selected = g_selected_tile,
            .prompt = "a lighthouse in heavy fog at dawn, desaturated, backlit beam, melancholy",
            .expanded = g_expanded,
            .busy = !g_reopened,
            .status = "rendering 2 of 4",
            .gone = if (g_reopened) 2 else 0,
            .can_send = g_reopened,
        }, .{
            .ctx = @ptrCast(&g_ctx),
            .on_toggle = onToggle,
            .on_select = onSelectTile,
            .on_open_studio = noopCtx,
            .on_send_to_chat = noopCtx,
            .on_cancel = noopCtxId,
        });
        gap(@src(), 2, 18);

        bubbles.agentProse(@src(), if (g_reopened)
            "Two of those are still here. The other two were only ever on disk, and that copy is gone."
        else
            "They are rendering now — you will see them fill in here as each one lands.");
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

    // Noise controls ON, so the probe shows the lit state: the toggle's whole job
    // is to be legible as on-vs-off at chip size, which is only checkable here.
    const cm: bubbles.Composer = .{
        .quick = &quick,
        .noise = .{
            .on = true,
            .shape = &noise_shape,
            .valid = true,
            .names = &noise_names,
            .sel = &g_noise_sel,
            .amount_buf = &g_noise_amt,
            .amount = 0.4,
            .amount_live = true,
        },
    };
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
/// Three hosts' bars, through the REAL `status_bar.render` rather than the
/// canned drawing above: one bar per host, each with its own name, its own
/// sampling history and its own draggable handles. This is the path the app
/// takes, so it is the one worth a picture; what it checks is that two bars do
/// not share a ring, a widget id or a handle.
var host_views: [4]status_bar.View = @splat(.{});
var host_mirrors: [4]mirror.Mirror = undefined;
var host_meters: [4]struct { split: f32, limit: f32 } = .{
    .{ .split = 0.60, .limit = 0.95 },
    .{ .split = 0.35, .limit = 0.80 },
    .{ .split = 0.50, .limit = 0.98 },
    .{ .split = 0.50, .limit = 0.95 },
};
const host_names = [_][]const u8{ "local", "lydia", "basement", "attic" };
/// The fourth bar is a host that is not answering: its row says why, in place
/// of numbers that would be stale.
const host_trouble = [_][]const u8{ "", "", "", "token refused: re-pair this host" };
/// Four reasons a bar is not taking work, one per row: held back on purpose,
/// short of a file, not yet describing itself, and not answering at all.
/// Without the note the first three are indistinguishable from an idle host.
const host_note = [_][]const u8{ "paused", "missing files", "starting up", "" };

fn cannedHosts(gpa: std.mem.Allocator) void {
    // A 24 GB card mid-chat, a small 4 GB card rendering, and one idle.
    const t = [_]wire.Telemetry{
        .{ .have_gpu = true, .gpu_util = 71, .cpu = 22, .vram_total = 24 * gb, .vram_used = 17 * gb, .vram_proc = 15 * gb, .llm_used = 12 * gb, .ctx_kv = 2 * gb, .ctx_tokens = 8192, .layers_gpu = 36, .limit = 22 * gb },
        .{ .have_gpu = true, .gpu_util = 96, .cpu = 8, .vram_total = 4 * gb, .vram_used = 3 * gb, .vram_proc = 3 * gb, .diff_dit = 1800 * 1024 * 1024, .diff_te = 172 * 1024 * 1024, .diff_latent = 120 * 1024 * 1024, .limit = 3 * gb },
        // A host that has not said what card it has: the bar must say so
        // rather than draw an invented one.
        .{ .cpu = 5 },
        .{},
    };
    const st = [_]wire.State{
        .{ .llm_resident = true, .llm_busy = true, .diff_present = true },
        .{ .diff_present = true, .diff_busy = true, .diff_family = "krea2", .pending_images = 2 },
        .{},
        .{},
    };
    for (&host_mirrors, 0..) |*m, i| {
        m.* = mirror.Mirror.init(gpa);
        m.telemetry = t[i];
        m.state = st[i];
        // Two samples so the sparklines have a slope rather than a dot.
        m.telemetry_seq = 1;
    }
}

fn hostsFrame() void {
    var root = dvui.box(@src(), .{ .dir = .vertical }, .{
        .expand = .both,
        .background = true,
        .color_fill = C.canvas,
    });
    defer root.deinit();

    style.sectionHead(@src(), "one bar per engine host", .{ .padding = .{ .x = 14, .y = 14, .h = 8 } });
    for (&host_views, 0..) |*v, i| {
        // Feed each bar a few samples so its history is not one flat dot.
        host_mirrors[i].telemetry_seq = @intCast(i + 1);
        status_bar.render(v, &host_mirrors[i], host_names[i], host_trouble[i], host_note[i], i, &host_meters[i].split, &host_meters[i].limit, .{
            .on_change = noop,
            .on_commit = noop,
            .on_eject_llm = noop,
            .on_eject_diff = noop,
            .on_toggle_pause_llm = noop,
            .on_toggle_pause_diff = noop,
        });
    }
}

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

/// The settings view over a DEFAULT config, so every section renders with the
/// values a fresh install ships. Settings is the one screen with computed
/// readouts in it (the noise curve's peak / head / cutoff line), and those are
/// exactly the kind of thing that reads wrong without being checked.
var probe_cfg: config.Config = blk: {
    var c: config.Config = .{};
    // Noise ENABLED, unlike a fresh install: it puts the curve preview in its lit
    // state, which is the drawing worth watching. Its two other states are one
    // edit away — a malformed expression goes amber, and a gated curve
    // (`max(0, 0.6*(1-t/0.4))`) adds the cutoff clause to the readout.
    c.weight_noise = true;
    // Two hosts, so the Hosts section draws its rows and not just the add form:
    // one healthy and one in trouble, which are the two colors that row has.
    _ = c.addHost("attic", "/run/user/1000/tensorpencil-qt/attic.sock", true);
    _ = c.addHost("lydia", "tp://10.0.0.130:7777/MIIBJDCBzKADAgECAhBuvy7FDePdICB9_bN_VedD#" ++ ("3c" ** 32), false);
    // Just pasted and answered, not applied: the one row whose news is good.
    _ = c.addHost("cellar", "tp://10.0.0.9:7777/MIIBJDCBzKADAgECAhBuvy7FDePdICB9_bN_VedD#" ++ ("7a" ** 32), false);
    break :blk c;
};

/// A catalog the settings form can draw from: one folder, a Gemma 4 class with a
/// matching and a non-matching tower, a Krea2 checkpoint with its side files.
fn cannedCatalog(gpa: std.mem.Allocator) !catalog.Catalog {
    const llm = catalog.Llm{ .arch = "gemma4", .size_label = "31B", .width = 5376, .blocks = 60, .supported = true, .vision = true, .class = "Gemma 4 31B" };
    var te: catalog.Entry = .{ .path = "/models/text_encoders/qwen3VLInstruct4b.safetensors", .size = 1, .mtime_ns = 1 };
    te.side.set(.krea2, .conditioner);
    var vae: catalog.Entry = .{ .path = "/models/vae/krea2RealVae_v10.safetensors", .size = 1, .mtime_ns = 1 };
    vae.side.set(.krea2, .decoder);
    vae.side.set(.anima, .decoder);
    var taew: catalog.Entry = .{ .path = "/models/vae_approx/taew2_1.safetensors", .size = 1, .mtime_ns = 1 };
    taew.preview.fams[@intFromEnum(catalog.Family.krea2)] = true;
    // Two SenseNova LoRAs, so the LoRA section draws with one row on and one
    // still on offer in the add dropdown. Both of its states in one frame.
    var turbo: catalog.Entry = .{ .path = "/models/loras/sensenovaU158BMot_8StepTurboLora.safetensors", .size = 1, .mtime_ns = 1 };
    turbo.lora = .{ .info = .{ .targets = 294, .rank = 128, .depth = 42 } };
    turbo.lora.?.fams[@intFromEnum(catalog.Family.sensenova)] = true;
    var style_lora: catalog.Entry = .{ .path = "/models/loras/sensenovaInkWash_v2.safetensors", .size = 1, .mtime_ns = 1 };
    style_lora.lora = .{ .info = .{ .targets = 168, .rank = 64, .depth = 42 } };
    style_lora.lora.?.fams[@intFromEnum(catalog.Family.sensenova)] = true;
    return catalog.Catalog.fromEntries(gpa, &.{
        .{ .path = "/models/llm/Gemma-4-31B-it-Q4_K_M.gguf", .size = 1, .mtime_ns = 1, .llm = llm },
        .{ .path = "/models/llm/Gemma-4-Dark-Thoughts-31B.i1-Q4_K_S.gguf", .size = 1, .mtime_ns = 1, .llm = llm },
        .{ .path = "/models/llm/mmproj-gemma-4-31b.gguf", .size = 1, .mtime_ns = 1, .tower = .{ .projector = "gemma4v", .arch = "gemma4", .width = 5376 } },
        .{ .path = "/models/llm/mmproj-gemma-4-12b.gguf", .size = 1, .mtime_ns = 1, .tower = .{ .projector = "gemma4uv", .arch = "gemma4", .width = 3840 } },
        .{ .path = "/models/llm/nomic-embed.gguf", .size = 1, .mtime_ns = 1, .llm = .{ .arch = "nomic-bert", .size_label = "137M", .width = 768, .blocks = 12, .supported = false, .vision = false, .class = "nomic-bert 137M" }, .note = "architecture 'nomic-bert' is not supported" },
        .{ .path = "/models/diffusion_models/krea2/krea2CenterSemiraw_v10Int8.safetensors", .size = 1, .mtime_ns = 1, .ckpt = .{ .family = .krea2, .contents = .{ .denoiser = true } } },
        .{ .path = "/models/checkpoints/sdxl/dreamshaperXL.safetensors", .size = 1, .mtime_ns = 1, .ckpt = .{ .family = .sdxl, .contents = .{ .denoiser = true, .conditioner = true, .conditioner2 = true, .decoder = true } } },
        .{ .path = "/models/diffusion_models/sensenova/sensenovaU158BMot_sft.safetensors", .size = 1, .mtime_ns = 1, .ckpt = .{ .family = .sensenova, .contents = .{ .denoiser = true, .conditioner = true } } },
        te,
        vae,
        taew,
        turbo,
        style_lora,
    });
}

/// Swap the settings probe onto SenseNova with one LoRA on (`--settings --lora`).
///
/// Its own mode rather than the default, because the two diffusion states draw
/// DIFFERENT rows and each is worth seeing: krea2 has the side-file and preview
/// rows and no sidecar path, SenseNova has the LoRA list and no side files.
/// Giving the canned krea2 entry a LoRA instead would have the fixture assert a
/// capability the engine does not have.
///
/// The dial is set to something other than 1, since a slider parked at its
/// default says nothing about whether the row renders the value it holds.
fn cannedLoraSelection(cfg: *config.Config) void {
    selection.selectCheckpoint(cfg, &probe_mirror.catalog, "/models/diffusion_models/sensenova/sensenovaU158BMot_sft.safetensors");
    if (cfg.addFamilyLora("sensenova", "/models/loras/sensenovaU158BMot_8StepTurboLora.safetensors")) |l| {
        l.strength = 0.85;
    }
}

// ------------------------------------------------------------------- studio

/// Canned prompt library for the studio's left rail. Deliberately awkward for
/// the same reason the conversation titles are: a long prompt that must
/// ellipsize by WIDTH, and a CJK one where a character budget would cut early.
const prompt_today = [_]shell.ConvRow{
    .{ .id = 1, .title = "a lighthouse in heavy fog, long exposure, muted palette" },
    .{ .id = 2, .title = "ceramic mug on oak, soft window light" },
    .{ .id = 3, .title = "霧の中の灯台、長時間露光" },
};
const prompt_earlier = [_]shell.ConvRow{
    .{ .id = 4, .title = "isometric shop fronts, pastel" },
};
const prompt_groups = [_]shell.ConvGroup{
    .{ .head = "TODAY", .rows = &prompt_today },
    .{ .head = "EARLIER", .rows = &prompt_earlier },
};

/// The studio draws from a mirror, so the probe fills one by hand: no host, no
/// engine, and nothing touches the GPU.
var probe_mirror: mirror.Mirror = undefined;

fn noopPost(_: wire.Request) void {}

/// The canvas source, over the one canned mirror. Two hosts would draw the
/// same, so the probe names one to show the caption carrying it.
fn probeImages() image_view.Images {
    return .{ .live = probeLive, .newest = probeNewest, .hostOf = probeHostOf };
}

fn probeLive(_: *anyopaque, out: []*const mirror.Image) []const *const mirror.Image {
    var n: usize = 0;
    for (probe_mirror.images.items) |*im| {
        if (n >= out.len) break;
        switch (im.status()) {
            .generating, .suspended => {},
            else => continue,
        }
        out[n] = im;
        n += 1;
    }
    return out[0..n];
}

fn probeNewest(_: *anyopaque) ?*const mirror.Image {
    const imgs = probe_mirror.images.items;
    var i = imgs.len;
    while (i > 0) {
        i -= 1;
        if (imgs[i].status() == .done) return &imgs[i];
    }
    return null;
}

fn probeHostOf(_: *anyopaque, _: wire.ImageId) []const u8 {
    return "workshop";
}

/// The catalog a host would have sent.
fn setCannedCatalog(c: catalog.Catalog) void {
    probe_mirror.catalog.deinit();
    probe_mirror.catalog = c;
}

/// The four hosts of `probe_cfg` with a catalog each, merged as the app merges
/// them. Every state the overview grid and the menu markers can draw is in
/// here, because each is a different wrong-looking thing when it breaks:
///
///   - a file on this machine and one other: the grid has two filled dots;
///   - a file two of three hosts can run: the marker reads "2 of 3";
///   - a checkpoint a host holds without the VAE it needs: a faint dot, and
///     the marker says missing files;
///   - a file only a host that is DOWN has: greyed, named, still listed;
///   - a file no host but a remote one has: it is in the menu at all, which is
///     the whole point.
var probe_models: models.Union = undefined;
/// The id text of the chat model only the remote host has, for the selection.
var probe_remote_llm_buf: [catalog.id_text_len]u8 = undefined;
var probe_remote_llm: []const u8 = "";

/// A file only a remote host has, named the way a remote catalog names one: the
/// id text as its path, the stem beside it. The id text is COMPUTED, because a
/// host computes it the same way and a fixture whose two disagree tests nothing
/// that can happen.
fn remoteEntry(buf: *[catalog.id_text_len]u8, name: []const u8, size: u64) catalog.Entry {
    return .{
        .path = catalog.idText(catalog.modelId(name, size, 0), buf),
        .name = name,
        .size = size,
        .mtime_ns = 1,
    };
}

/// One host's catalog: the entries of `full` whose path contains any of `keep`,
/// plus `extra` (files this machine does not have).
fn hostCatalog(gpa: std.mem.Allocator, full: *const catalog.Catalog, keep: []const []const u8, extra: []const catalog.Entry) !catalog.Catalog {
    var list: std.ArrayList(catalog.Entry) = .empty;
    defer list.deinit(gpa);
    for (full.entries) |*e| {
        for (keep) |k| if (std.mem.indexOf(u8, e.path, k) != null) {
            try list.append(gpa, e.*);
            break;
        };
    }
    try list.appendSlice(gpa, extra);
    return catalog.Catalog.fromEntries(gpa, list.items);
}

fn buildProbeModels(gpa: std.mem.Allocator) !void {
    const full = &probe_mirror.catalog;

    // A remote host names its files by id, never by path, and carries the stem
    // beside it. Drawing them any other way would hide the case that matters.
    const zimg: catalog.Entry = .{ .path = "id:00000000000000a1", .name = "zImageTurbo_v10", .size = 41, .mtime_ns = 1, .ckpt = .{ .family = .zimage, .contents = .{ .denoiser = true } } };
    var zimg_te: catalog.Entry = .{ .path = "id:00000000000000a2", .name = "qwen3_4b_instruct", .size = 42, .mtime_ns = 1 };
    zimg_te.side.set(.zimage, .conditioner);
    var zimg_vae: catalog.Entry = .{ .path = "id:00000000000000a3", .name = "fluxVae", .size = 43, .mtime_ns = 1 };
    zimg_vae.side.set(.zimage, .decoder);
    // A chat model only the remote host has, named the way a remote catalog
    // names one. The settings probe SELECTS it: a reference is an id there, and
    // a label built from the reference rather than the catalog reads as
    // "id:7da8e81dfa8c1632" to the user.
    var b6: [catalog.id_text_len]u8 = undefined;
    var remote_llm = remoteEntry(&b6, "Qwen3-4B-Instruct-2507-Q4_K_M", 46);
    remote_llm.llm = .{ .arch = "qwen3", .size_label = "4B", .width = 2560, .blocks = 36, .supported = true, .vision = false, .class = "Qwen 3 4B" };
    probe_remote_llm = catalog.idText(catalog.modelId("Qwen3-4B-Instruct-2507-Q4_K_M", 46, 0), &probe_remote_llm_buf);
    // An Anima checkpoint with nothing on that host to pair it with.
    var b4: [catalog.id_text_len]u8 = undefined;
    var orphan = remoteEntry(&b4, "pastelkaAnima_v7", 44);
    orphan.ckpt = .{ .family = .anima, .contents = .{ .denoiser = true } };
    // A checkpoint only the host that is down has.
    var b5: [catalog.id_text_len]u8 = undefined;
    var stranded = remoteEntry(&b5, "kWALUAN_v08INT8", 45);
    stranded.ckpt = .{ .family = .krea2, .contents = .{ .denoiser = true } };

    var attic = try hostCatalog(gpa, full, &.{ "Gemma-4-31B", "dreamshaperXL" }, &.{orphan});
    defer attic.deinit();
    // Holds the Krea 2 checkpoint and its encoder, but no VAE: it has the file
    // and still cannot render with it.
    var lydia = try hostCatalog(gpa, full, &.{ "krea2CenterSemiraw", "qwen3VLInstruct4b", "Gemma-4-31B" }, &.{ zimg, zimg_te, zimg_vae, remote_llm });
    defer lydia.deinit();
    var cellar = try hostCatalog(gpa, full, &.{}, &.{stranded});
    defer cellar.deinit();

    probe_models.deinit();
    probe_models = try models.build(gpa, &.{
        .{ .name = "local", .cat = full },
        .{ .name = "attic", .cat = &attic },
        .{ .name = "lydia", .cat = &lydia },
        .{ .name = "cellar", .up = false, .cat = &cellar },
    });
}

/// One finished render for the canvas, as a synthetic gradient. Real pixels
/// rather than a placeholder because the canvas's whole job is sizing an image
/// into the column, and a flat block would not show that going wrong.
fn cannedStudio(gpa: std.mem.Allocator) !void {
    probe_mirror.state.diff_present = true;

    const w: u32 = 1216;
    const h: u32 = 832;
    const rgba = try gpa.alloc(u8, @as(usize, w) * h * 4);
    for (0..h) |y| for (0..w) |x| {
        const i = (y * w + x) * 4;
        rgba[i + 0] = @intCast(40 + (x * 120) / w);
        rgba[i + 1] = @intCast(60 + (y * 90) / h);
        rgba[i + 2] = @intCast(90 + ((x + y) * 100) / (w + h));
        rgba[i + 3] = 255;
    };
    _ = probe_mirror.addLocal(.{
        .prompt = "a lighthouse in heavy fog",
        .status = .done,
        .req_width = w,
        .req_height = h,
        .req_steps = 28,
        .req_seed = 8812,
        .width = w,
        .height = h,
        .pixels_rev = 1,
        .from_studio = true,
    }, rgba, null);

    // One still queued, so the rail has a row and the composer shows its count.
    _ = probe_mirror.addLocal(.{
        .prompt = "a lighthouse, taller crop",
        .status = .pending,
        .req_width = 832,
        .req_height = 1216,
        .req_steps = 28,
        .total = 28,
        .from_studio = true,
    }, null, null);

    // Three renders in motion, which is what two or three hosts produce all the
    // time and what the canvas grid exists for. Different aspect ratios and one
    // with no preview yet: the tile sizing and the empty slot are the two things
    // that go wrong here, and neither shows with a single square render.
    try cannedLive(gpa, "ceramic mug on oak, soft window light", 1216, 832, 7, 28, .generating, true);
    try cannedLive(gpa, "isometric shop fronts, pastel", 832, 1216, 19, 28, .generating, true);
    try cannedLive(gpa, "霧の中の灯台、長時間露光", 1024, 1024, 2, 28, .generating, false);
}

/// One render in flight, with a coarse preview standing in for the sampler's.
fn cannedLive(
    gpa: std.mem.Allocator,
    prompt: []const u8,
    w: u32,
    h: u32,
    step: u32,
    total: u32,
    status: wire.ImageStatus,
    with_preview: bool,
) !void {
    const id = probe_mirror.addLocal(.{
        .prompt = prompt,
        .status = status,
        .req_width = w,
        .req_height = h,
        .req_steps = total,
        .req_seed = 4410 + step,
        .step = step,
        .total = total,
        .width = w,
        .height = h,
        .from_studio = true,
    }, null, null);
    if (!with_preview) return;

    // A preview arrives at a fraction of the final size, so the probe's is one
    // too: a tile that sizes off the preview rather than the request is a bug
    // only a non-matching size shows.
    const pw = w / 8;
    const ph = h / 8;
    const px = try gpa.alloc(u8, @as(usize, pw) * ph * 4);
    for (0..ph) |y| for (0..pw) |x| {
        const i = (y * pw + x) * 4;
        px[i + 0] = @intCast(30 + (y * 150) / ph);
        px[i + 1] = @intCast(70 + (x * 80) / pw);
        px[i + 2] = @intCast(120 - (x * 90) / pw);
        px[i + 3] = 255;
    };
    const im = probe_mirror.byId(id) orelse return;
    im.preview = px;
    im.preview_w = pw;
    im.preview_h = ph;
    im.preview_rev = 1;
}

fn studioFrame() void {
    var root = dvui.box(@src(), .{ .dir = .vertical }, .{
        .expand = .both,
        .background = true,
        .color_fill = C.canvas,
    });
    defer root.deinit();

    const bands = shell.Bands.from(root, status_bar.bar_outer_height);

    shell.titleBar(.{
        .tab = .studio,
        .llm = .{ .label = "Gemma-4-31B-it-Q4_K_M", .resident = true },
        .llm_menu = probe_llm_menu,
        .image = .{ .label = "sensenovaU158BMot_sft", .resident = true },
        .image_menu = probe_image_menu,
    }, .{ .on_tab = onTab, .on_llm_pick = noopPick, .on_image_pick = noopPick, .on_settings = noop });

    {
        var body = dvui.box(@src(), .{ .dir = .horizontal }, .{
            .expand = .horizontal,
            .min_size_content = .{ .h = bands.body },
            .max_size_content = .height(bands.body),
        });
        defer body.deinit();

        shell.sidebar(.{
            .groups = &prompt_groups,
            .new_label = "New prompt",
            .empty = "Prompts you generate from are kept here.",
            .footer = false,
        }, .{
            .on_new_chat = noop,
            .on_select = noopId,
            .on_delete = noopId,
            .on_models = noop,
            .on_settings = noop,
        });

        {
            const band_w = dvui.parentGet().data().contentRect().w;
            const col_w = @max(320, band_w - style.Layout.sidebar_w - style.Layout.rail_w);
            var col = dvui.box(@src(), .{ .dir = .vertical }, .{
                .expand = .vertical,
                .min_size_content = .{ .w = col_w },
                .max_size_content = .width(col_w),
            });
            defer col.deinit();
            image_view.render(&probe_cfg, &probe_mirror, &probe_models, probeImages(), noopPost, true, .{ .settings = noop, .save_defaults = noop, .cancel = noopId });
        }

        // The same rail as chat: work in motion is on the canvas here and in the
        // tool card there, so neither lists it.
        queue_rail.render(.{
            .tab = g_rail_tab,
            .jobs = &jobs,
            .library = &library,
        }, .{
            .on_tab = onRailTab,
            .on_pause_all = noop,
            .on_open_library = noopId,
            .on_cancel = noopId,
            .on_retry = noopId,
            .on_reorder = onReorder,
        });
    }

    statusBar(states[0], 0);
}

fn settingsFrame() void {
    // The app pushes this from a live capability probe; the probe has no model, so
    // it asserts the supported case, which is the one with a section to look at.
    config_view.g_noise_supported = true;
    // The add form as it looks having refused: pressing Add with no name is
    // what a pasted pairing string alone does, and it used to say nothing.
    config_view.g_add_host = .no_name;
    // Two library sections open and the rest closed: the grid and the header
    // are different drawings and both are worth seeing in one frame.
    config_view.g_lib_open[0] = true; // chat models
    config_view.g_lib_open[2] = true; // image checkpoints
    config_view.render(&probe_cfg, &probe_mirror, &probe_models, .{
        .apply = noop,
        .cancel = noop,
        .rescan = noop,
        .hostStatus = noHostStatus,
        .hostSync = noHostSync,
        .sendModel = noSendModel,
        .sendPath = noSendPath,
        .pullPath = noPullPath,
        .tryHost = noTryHost,
        .reconnectHost = noReconnectHost,
    });
}

// --------------------------------------------------------------------- main

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const arena = init.arena.allocator();
    Backend.c.SDL_SetMainReady();

    const args = try init.minimal.args.toSlice(arena);
    probe_mirror = mirror.Mirror.init(std.heap.smp_allocator);
    probe_models = models.empty(std.heap.smp_allocator);
    var out_path: []const u8 = "ui_probe.png";
    var states_mode = false;
    var settings_mode = false;
    var studio_mode = false;
    var click_pt: ?dvui.Point.Physical = null;
    var lora_mode = false;
    var hosts_mode = false;
    var dims: [2]?u32 = .{ null, null };
    var seen_out = false;
    for (args[1..]) |arg| {
        if (std.mem.eql(u8, arg, "--states")) {
            states_mode = true;
        } else if (std.mem.eql(u8, arg, "--settings")) {
            settings_mode = true;
            // The form reads the catalog and the config's selection; give it both.
            setCannedCatalog(try cannedCatalog(std.heap.smp_allocator));
            try buildProbeModels(std.heap.smp_allocator);
            _ = probe_cfg.addModelDir("/models");
            // A model only the remote host has, referenced by id: the row must
            // read as its name.
            selection.selectLlm(&probe_cfg, &probe_models.cat, probe_remote_llm);
            selection.selectCheckpoint(&probe_cfg, &probe_mirror.catalog, "/models/diffusion_models/krea2/krea2CenterSemiraw_v10Int8.safetensors");
        } else if (std.mem.startsWith(u8, arg, "--click=")) {
            // `--click=X,Y`, in physical pixels, matching the captured PNG. One
            // argument because this loop walks values, not indexes.
            const v = arg["--click=".len..];
            const comma = std.mem.indexOfScalar(u8, v, ',') orelse return error.InvalidArgs;
            click_pt = .{
                .x = try std.fmt.parseFloat(f32, v[0..comma]),
                .y = try std.fmt.parseFloat(f32, v[comma + 1 ..]),
            };
        } else if (std.mem.eql(u8, arg, "--studio")) {
            studio_mode = true;
            setCannedCatalog(try cannedCatalog(std.heap.smp_allocator));
            try buildProbeModels(std.heap.smp_allocator);
            _ = probe_cfg.addModelDir("/models");
        } else if (std.mem.eql(u8, arg, "--lora")) {
            lora_mode = true;
        } else if (std.mem.eql(u8, arg, "--library")) {
            g_rail_tab = .library;
            try cannedLibrary(std.heap.smp_allocator);
        } else if (std.mem.eql(u8, arg, "--reopened")) {
            g_reopened = true;
        } else if (std.mem.eql(u8, arg, "--hosts")) {
            hosts_mode = true;
            cannedHosts(std.heap.smp_allocator);
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
    // The flag may arrive after `--settings`, so the swap happens once the whole
    // command line has been read.
    if (settings_mode and lora_mode) cannedLoraSelection(&probe_cfg);
    if (!settings_mode and !states_mode and !studio_mode and !hosts_mode) {
        if (g_reopened) try cannedReopened(std.heap.smp_allocator) else try cannedTiles(std.heap.smp_allocator);
    }
    // The studio needs a checkpoint selected for its architecture-dependent
    // rows; `--lora` puts it on SenseNova, which is the only family here with a
    // LoRA section to draw (see Family.supportsLora).
    if (studio_mode) {
        if (lora_mode) {
            cannedLoraSelection(&probe_cfg);
            // The studio reads the family from the checkpoint FILE, and the
            // probe's paths are canned, so it is told directly.
            image_view.forced_family = .sensenova;
            // Every section open: a probe of a collapsible form that shows only
            // the folded heads is a probe of four words.
            probe_cfg.studio_open_loras = true;
            probe_cfg.studio_open_advanced = true;
        } else {
            selection.selectCheckpoint(&probe_cfg, &probe_mirror.catalog, "/models/diffusion_models/krea2/krea2CenterSemiraw_v10Int8.safetensors");
            image_view.forced_family = .krea2;
            // Advanced open with conditioning noise ON, because its rows only
            // exist when the toggle is, and an unopened section probes nothing.
            probe_cfg.studio_open_advanced = true;
            probe_cfg.cond_noise = true;
            probe_cfg.cond_steers.items[0] = .{ .text = .lit("tentacles, suckers"), .scale = 1.2 };
            probe_cfg.cond_steers.items[1] = .{ .text = .lit("anime, cel shading"), .scale = -0.8 };
            probe_cfg.cond_steers.count = 2;
        }
        try cannedStudio(std.heap.smp_allocator);
    }

    // Settings is a tall scrolled form; give it a canvas the whole thing fits on
    // so a capture shows every section rather than whatever the scroll happens to
    // be resting on.
    const h: u32 = dims[1] orelse (if (states_mode) @as(u32, 126 * states.len) else if (settings_mode) @as(u32, 2600) else if (studio_mode) @as(u32, 900) else if (hosts_mode) @as(u32, 60 + 4 * @as(u32, @intFromFloat(status_bar.bar_outer_height))) else 705);

    const win_opts: Backend.InitOptions = .{
        .io = init.io,
        .allocator = gpa,
        .size = .{ .w = @floatFromInt(w), .h = @floatFromInt(h) },
        .vsync = false,
        .title = "ui-probe",
        .hidden = true,
        .environ_map = init.environ_map,
    };
    // The window is never shown, so a display is only worth having for the
    // fidelity of it: SDL picks no driver at all without one, and its dummy
    // driver lands on the software renderer, whose glyph blitting is blurrier.
    // Layout is identical either way, which is what a headless run is asking
    // about. Retry rather than probe for a display, so any reason SDL cannot
    // reach one takes the same path.
    var back = Backend.initWindow(win_opts) catch blk: {
        Backend.c.SDL_Quit();
        std.debug.print("ui-probe: no display, falling back to the SDL dummy video driver\n", .{});
        _ = Backend.c.SDL_SetHint(Backend.c.SDL_HINT_VIDEO_DRIVER, "dummy");
        break :blk try Backend.initWindow(win_opts);
    };
    defer back.deinit();

    var win = try dvui.Window.init(@src(), gpa, back.backend(), .{});
    defer win.deinit();

    try win.begin(win.frame_time_ns);
    style.install();
    _ = try win.end(.{});

    // One notice, posted from OUTSIDE any frame, which is where the app learns
    // of a failed render (pumping the hosts happens between frames). Handing
    // that straight to dvui panics; `toast.pump` inside the frame is why this
    // reaches the screen at all. Reopening a conversation raises none: pictures
    // whose files are gone are drawn, never announced.
    if (!g_reopened)
        toast.post(.warn, "Image failed on lydia: the checkpoint is missing a component (VAE or text encoder). Trying local instead.", .{});

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
        if (states_mode) statesFrame() else if (settings_mode) settingsFrame() else if (studio_mode) studioFrame() else if (hosts_mode) hostsFrame() else frame();
        _ = try win.end(.{});
    }

    // `--click X Y`: press and release at a point, then report what took keyboard
    // focus. A GUI's "clicking does nothing" is otherwise only reproducible by
    // driving someone's actual screen.
    if (click_pt) |cp| {
        for (0..4) |i| {
            try win.begin(win.frame_time_ns + @as(i128, @intCast(6 + i)) * 16 * std.time.ns_per_ms);
            _ = try win.addEventMouseMotion(.{ .pt = cp });
            if (i == 1) _ = try win.addEventMouseButton(.left, .press);
            if (i == 2) _ = try win.addEventMouseButton(.left, .release);
            style.install();
            if (states_mode) statesFrame() else if (settings_mode) settingsFrame() else if (studio_mode) studioFrame() else if (hosts_mode) hostsFrame() else frame();
            const foc = dvui.focusedWidgetId();
            _ = try win.end(.{});
            std.debug.print("ui-probe: after frame {d} at ({d},{d}) focus={?} steer_editing={any}\n", .{
                i, cp.x, cp.y, foc, image_view.steer_editing,
            });
        }
    }

    try win.begin(win.frame_time_ns + 12 * 16 * std.time.ns_per_ms);
    _ = try win.addEventMouseMotion(.{ .pt = hover_pt });
    style.install();
    var pic = dvui.Picture.start(dvui.windowRectPixels()) orelse return error.CaptureUnsupported;
    if (states_mode) statesFrame() else if (settings_mode) settingsFrame() else if (studio_mode) studioFrame() else if (hosts_mode) hostsFrame() else frame();
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
