//! Transcript leaves and the composer, as pure presentation over plain data.
//!
//! The asymmetry between the two message shapes is load-bearing: the user's
//! turn is a card ON the page, the agent's turn IS the page. So a user message
//! gets a bubble, a right edge and a tail corner; an agent message gets no
//! bubble, no avatar and no background at all — only a measure limit.
//!
//! Every generation the model requests appears here as a tool call, with its
//! prompt behind one disclosure: visible depth, collapsed by default.
//!
//! While a call is still rendering its tiles are DASHED EMPTY SLOTS. The pixels
//! are in the queue rail during that window, and they move here as each one
//! lands — so an image is in exactly one place at any moment. The card reserves
//! the shape so the transcript does not jump when they arrive.
const std = @import("std");
const dvui = @import("dvui");
const style = @import("style.zig");
const fonts = @import("fonts.zig");
const markdown_view = @import("markdown_view.zig");

const C = style.C;
/// Widest a tool-call tile is allowed to be. A thumbnail is an index entry;
/// the viewer is where an image is actually looked at.
const tile_max: f32 = 150;
const F = style.F;
const R = style.R;
const L = style.Layout;

/// One image tile in a tool call's grid.
pub const Tile = union(enum) {
    /// A slot the card has reserved for an image still in the queue. Dashed and
    /// empty rather than a grey block, so it reads as "nothing here yet" and
    /// not as a render that came out black.
    pending,
    /// Failed or canceled: the slot stays, so the grid does not reflow.
    failed,
    rgba: struct { px: []const u8, w: u32, h: u32 },
};

pub const ToolCall = struct {
    /// The tool name, shown as an amber tag.
    name: []const u8 = "image.generate",
    /// "1024² · seed 8812"
    meta: []const u8,
    tiles: []const Tile,
    /// Index into `tiles`, outlined in blue: the user's pick.
    selected: ?usize = null,
    /// Behind the card's own disclosure: the prompt, readable. The RAW call is
    /// a separate section above the card (see app.renderReply) — one is what
    /// was asked for, the other is what was literally emitted.
    prompt: []const u8 = "",
    expanded: bool = false,
    /// Still rendering: the action row is disabled and the empty slots say so.
    busy: bool = false,
    /// Shown under the tiles while `busy`, e.g. "rendering 2 of 4".
    status: []const u8 = "",
    /// Distinguishes several cards in one reply, which now happens whenever the
    /// model talks between generations.
    id_extra: usize = 0,
};

pub const ToolActions = struct {
    ctx: *anyopaque,
    on_toggle: *const fn (*anyopaque) void,
    on_select: *const fn (*anyopaque, usize) void,
    on_open_studio: *const fn (*anyopaque) void,
};

/// A right-aligned user message. The 3px bottom-right corner is the tail.
pub fn userBubble(src: std.builtin.SourceLocation, text: []const u8) void {
    var wrap = dvui.box(src, .{ .dir = .horizontal }, .{ .expand = .horizontal });
    defer wrap.deinit();
    var b = dvui.box(@src(), .{ .dir = .vertical }, .{
        .gravity_x = 1.0,
        .max_size_content = .width(L.bubble_max),
        .background = true,
        .color_fill = C.bubble_user,
        .border = style.Edge.all,
        .color_border = style.hairline,
        // tl, tr, br, bl — the tail is the small one.
        .corner_radius = .{ .x = 10, .y = 10, .w = 3, .h = 10 },
        .padding = .{ .x = 14, .y = 11, .w = 14, .h = 11 },
    });
    defer b.deinit();
    var tl = dvui.textLayout(@src(), .{}, .{
        .expand = .horizontal,
        .background = false,
        .padding = .{},
        .font = F.prose,
        .color_text = C.text_hi,
    });
    defer tl.deinit();
    fonts.addStyled(tl, text, .{}, .{ .font = F.prose, .color_text = C.text_hi });
}

/// Agent prose: no bubble, no avatar, just a measure limit. Markdown-rendered,
/// since that is what the model emits.
pub fn agentProse(src: std.builtin.SourceLocation, text: []const u8) void {
    var wrap = dvui.box(src, .{ .dir = .vertical }, .{
        .expand = .horizontal,
        .max_size_content = .width(L.prose_max),
    });
    defer wrap.deinit();
    markdown_view.render(@src(), text, .{ .prose = .{
        .background = false,
        .padding = .{},
        .font = F.prose,
        .color_text = C.text,
    } });
}

/// The tool-call card: header (amber tag, meta, disclosure), a grid of tiles,
/// and the action row.
pub fn toolCard(src: std.builtin.SourceLocation, tc: ToolCall, cb: ToolActions) void {
    var wrap = dvui.box(src, .{ .dir = .vertical }, .{ .expand = .horizontal, .id_extra = tc.id_extra });
    defer wrap.deinit();

    var card = dvui.box(@src(), .{ .dir = .vertical }, .{
        .expand = .horizontal,
        .max_size_content = .width(L.card_max),
        .background = true,
        .color_fill = C.card,
        .border = style.Edge.all,
        .color_border = style.hairline_hi,
        .corner_radius = R.card,
    });
    defer card.deinit();

    // ---- header
    {
        var hdr = dvui.box(@src(), .{ .dir = .horizontal }, .{
            .expand = .horizontal,
            // See app.zig's composer block: a one-edge border makes dvui force
            // a background on, so the card's own surface has to be named or it
            // gets the theme's.
            .background = true,
            .color_fill = C.card,
            .border = style.Edge.bottom,
            .color_border = style.hairline_soft,
            .padding = .{ .x = 12, .y = 10, .w = 12, .h = 10 },
        });
        defer hdr.deinit();

        // The tag is amber because a tool call is the machine acting.
        dvui.labelNoFmt(@src(), tc.name, .{}, .{
            .font = F.mono,
            .color_text = C.amber,
            .background = true,
            .color_fill = style.over(C.card, C.amber, 0.12),
            .corner_radius = dvui.Rect.all(4),
            .padding = .{ .x = 6, .y = 4, .w = 6, .h = 4 },
            .margin = .{ .w = 9 },
            .gravity_y = 0.5,
        });
        dvui.labelNoFmt(@src(), tc.meta, .{}, .{
            .font = F.mono,
            .color_text = C.text_dim,
            .padding = .{},
            .gravity_y = 0.5,
        });
        if (tc.prompt.len > 0) {
            var d: dvui.ButtonWidget = undefined;
            d.init(@src(), .{}, .{
                .gravity_x = 1.0,
                .gravity_y = 0.5,
                .background = false,
                .color_fill_hover = style.hover_wash,
                .color_fill_press = style.hover_wash,
                .corner_radius = R.chip,
                .padding = .{ .x = 5, .y = 3, .w = 5, .h = 3 },
                .margin = .{},
            });
            d.processEvents();
            d.drawBackground();
            {
                var row = dvui.box(@src(), .{ .dir = .horizontal }, .{});
                defer row.deinit();
                style.mark(@src(), if (tc.expanded) .disclosure_open else .disclosure, 9, C.text_dim, .{ .margin = .{ .w = 4 } });
                dvui.labelNoFmt(@src(), "prompt", .{}, .{
                    .font = F.mono,
                    .color_text = C.text_dim,
                    .padding = .{},
                    .gravity_y = 0.5,
                });
            }
            const clicked = d.clicked();
            d.deinit();
            if (clicked) cb.on_toggle(cb.ctx);
        }
    }

    // ---- collapsed depth
    if (tc.expanded) {
        var det = dvui.box(@src(), .{ .dir = .vertical }, .{
            .expand = .horizontal,
            .background = true,
            .color_fill = C.sunken,
            .border = style.Edge.bottom,
            .color_border = style.hairline_soft,
            .padding = .{ .x = 12, .y = 10, .w = 12, .h = 10 },
        });
        defer det.deinit();
        var tl = dvui.textLayout(@src(), .{}, .{
            .expand = .horizontal,
            .background = false,
            .padding = .{},
            .font = F.code,
            .color_text = C.text_dim,
        });
        defer tl.deinit();
        fonts.addStyled(tl, tc.prompt, .{}, .{ .font = F.code, .color_text = C.text_dim });
    }

    // ---- tiles
    //
    // A CAPPED grid, wrapping at 4 across. Dividing the card width by the tile
    // count made a single render 570px wide, which is a poster in the middle of
    // a conversation; the tile is an index entry, and the viewer (click one) is
    // where you actually look at a picture.
    if (tc.tiles.len > 0) {
        const gap: f32 = 8;
        const per_row: usize = 4;

        var grid = dvui.box(@src(), .{ .dir = .vertical }, .{
            .expand = .horizontal,
            .padding = dvui.Rect.all(12),
        });
        defer grid.deinit();

        const avail = grid.data().contentRect().w;
        const across: f32 = @floatFromInt(@min(tc.tiles.len, per_row));
        const cell = @min(tile_max, @max(24, (avail - gap * (across - 1)) / across));

        var row_start: usize = 0;
        while (row_start < tc.tiles.len) : (row_start += per_row) {
            const row_end = @min(row_start + per_row, tc.tiles.len);
            var row = dvui.box(@src(), .{ .dir = .horizontal }, .{
                .id_extra = row_start,
                .expand = .horizontal,
                .margin = .{ .h = if (row_end < tc.tiles.len) gap else 0 },
            });
            defer row.deinit();

            for (tc.tiles[row_start..row_end], row_start..) |t, i| {
                const sel = tc.selected == i;
                var cellbox = dvui.box(@src(), .{}, .{
                    .id_extra = i,
                    .min_size_content = .{ .w = cell, .h = cell },
                    .max_size_content = .size(.{ .w = cell, .h = cell }),
                    .margin = .{ .w = if (i + 1 < row_end) gap else 0 },
                    // No glow and no scale on selection: a 1.5px blue edge is
                    // enough, and anything else makes the grid jump.
                    .border = if (sel) dvui.Rect.all(1.5) else .{},
                    .color_border = C.blue,
                    .corner_radius = R.chip,
                });
                const crs = cellbox.data().contentRectScale();
                switch (t) {
                    .rgba => |px| _ = dvui.image(@src(), .{
                        .source = .{ .pixels = .{ .rgba = px.px, .width = px.w, .height = px.h } },
                        .shrink = .ratio,
                    }, .{
                        .expand = .both,
                        .corner_radius = R.chip,
                    }),
                    .pending => dashedSlot(crs.r, crs.s, style.hairline_hi),
                    .failed => dashedSlot(crs.r, crs.s, style.tint(C.danger, 90)),
                }
                const clicked = dvui.clicked(cellbox.data(), .{});
                cellbox.deinit();
                if (clicked) cb.on_select(cb.ctx, i);
            }
            // Keep a short row left-aligned instead of stretching its tiles.
            var fill = dvui.box(@src(), .{}, .{ .id_extra = row_start, .expand = .horizontal });
            fill.deinit();
        }
    }

    // ---- actions
    {
        var acts = dvui.box(@src(), .{ .dir = .horizontal }, .{
            .expand = .horizontal,
            .padding = .{ .x = 12, .w = 12, .h = 12 },
        });
        defer acts.deinit();

        if (primaryButton(@src(), "Open in Studio", !tc.busy)) cb.on_open_studio(cb.ctx);

        // While rendering, the right-hand caption says where the pixels are:
        // the images are in the QUEUE until they land here (see queue_rail).
        if (tc.busy and tc.status.len > 0) {
            dvui.labelNoFmt(@src(), tc.status, .{}, .{
                .font = F.mono,
                .color_text = C.amber, // the machine is working
                .gravity_x = 1.0,
                .gravity_y = 0.5,
                .padding = .{},
            });
        } else if (tc.selected) |sel| {
            var buf: [32]u8 = undefined;
            dvui.labelNoFmt(@src(), std.fmt.bufPrint(&buf, "render {d:0>2} selected", .{sel + 1}) catch "", .{}, .{
                .font = F.mono,
                .color_text = C.text_ghost,
                .gravity_x = 1.0,
                .gravity_y = 0.5,
                .padding = .{},
            });
        }
    }
}

/// The light-on-dark primary action. Never amber: a button is the user acting.
pub fn primaryButton(src: std.builtin.SourceLocation, label: []const u8, enabled: bool) bool {
    var bw: dvui.ButtonWidget = undefined;
    bw.init(src, .{}, .{
        .background = true,
        .color_fill = if (enabled) C.text_hi else C.raised,
        .color_fill_hover = if (enabled) C.text_hi.lighten(-6) else C.raised,
        .color_fill_press = if (enabled) C.text_hi.lighten(-12) else C.raised,
        .corner_radius = R.button,
        .padding = .{ .x = 12, .y = 7, .w = 12, .h = 7 },
        .margin = .{ .w = 8 },
        .gravity_y = 0.5,
    });
    bw.processEvents();
    bw.drawBackground();
    dvui.labelNoFmt(@src(), label, .{}, .{
        .font = F.ui,
        .color_text = if (enabled) C.text_ink else C.text_ghost,
        .padding = .{},
        .gravity_x = 0.5,
        .gravity_y = 0.5,
    });
    const clicked = bw.clicked();
    bw.drawFocus();
    bw.deinit();
    return clicked and enabled;
}

pub fn secondaryButton(src: std.builtin.SourceLocation, id: usize, label: []const u8, enabled: bool) bool {
    var bw: dvui.ButtonWidget = undefined;
    bw.init(src, .{}, .{
        .id_extra = id,
        .background = true,
        .color_fill = C.raised_dim,
        .color_fill_hover = C.raised_dim.lighten(6),
        .color_fill_press = C.raised_dim.lighten(12),
        .border = style.Edge.all,
        .color_border = style.hairline,
        .corner_radius = R.button,
        .padding = .{ .x = 12, .y = 7, .w = 12, .h = 7 },
        .margin = .{ .w = 8 },
        .gravity_y = 0.5,
    });
    bw.processEvents();
    bw.drawBackground();
    dvui.labelNoFmt(@src(), label, .{}, .{
        .font = F.ui,
        .color_text = if (enabled) C.text else C.text_ghost,
        .padding = .{},
        .gravity_x = 0.5,
        .gravity_y = 0.5,
    });
    const clicked = bw.clicked();
    bw.drawFocus();
    bw.deinit();
    return clicked and enabled;
}

/// A dashed 1px outline: a reserved slot with nothing in it yet.
fn dashedSlot(r: dvui.Rect.Physical, scale: f32, color: dvui.Color) void {
    if (r.w <= 0 or r.h <= 0) return;
    const dash: f32 = 3 * @max(1, scale);
    const gap: f32 = 3 * @max(1, scale);
    const th: f32 = @max(1, scale);
    var x = r.x;
    while (x < r.x + r.w) : (x += dash + gap) {
        const w = @min(dash, r.x + r.w - x);
        (dvui.Rect.Physical{ .x = x, .y = r.y, .w = w, .h = th }).fill(.{}, .{ .color = color });
        (dvui.Rect.Physical{ .x = x, .y = r.y + r.h - th, .w = w, .h = th }).fill(.{}, .{ .color = color });
    }
    var y = r.y;
    while (y < r.y + r.h) : (y += dash + gap) {
        const h = @min(dash, r.y + r.h - y);
        (dvui.Rect.Physical{ .x = r.x, .y = y, .w = th, .h = h }).fill(.{}, .{ .color = color });
        (dvui.Rect.Physical{ .x = r.x + r.w - th, .y = y, .w = th, .h = h }).fill(.{}, .{ .color = color });
    }
}

// ------------------------------------------------------------------ composer

pub const QuickSetting = struct {
    label: []const u8,
    /// Opens a menu when clicked.
    dropdown: bool = true,
};

pub const Composer = struct {
    quick: []const QuickSetting = &.{},
    placeholder: []const u8 = "Describe an image, or ask for changes…",
    busy: bool = false,
    /// Hides the reference affordance when the model cannot see images.
    can_attach: bool = true,
    /// Weight-noise controls, beside the entry. Absent when the loaded model
    /// cannot take it (no CUDA backend), so the affordance never lies about
    /// having an effect.
    noise: ?Noise = null,
};

/// The composer's weight-noise controls: a lit/unlit toggle, a preview of the
/// active curve's shape, and a dropdown to pick one from the saved library.
///
/// Editing an expression lives in Settings, not here. A curve worth using is
/// something like `max(0, 0.6*(1-t/0.4))`, which is not a thing to retype in a
/// composer chip; what belongs here is flipping between the ones you keep, which
/// is a per-message decision.
///
/// `shape` is the active curve sampled first-layer-to-LM-head and normalized to
/// its own peak, so the preview answers "what shape" and the name answers "which
/// one". `names` is the library plus, when the active curve matches none of it, a
/// trailing entry for the typed-in one; `sel` indexes it and is mutated in place.
pub const Noise = struct {
    on: bool,
    shape: []const f32,
    /// The active curve parses. A malformed one is noise OFF, which is invisible
    /// without saying so: a half-typed expression looks exactly like a flat 0.
    valid: bool,
    names: []const []const u8,
    sel: *usize,
    /// The amount field's text, held across frames because it is typed into, and
    /// `amount` is what the config holds, redisplayed whenever it is not being
    /// edited. Shape says where the noise goes, this says how much.
    amount_buf: []u8,
    amount: f32,
    /// The active shape actually reads `a`. A curve with its amplitude baked in
    /// ignores the field, so the field says so by dimming rather than by quietly
    /// doing nothing.
    amount_live: bool,
};

pub const ComposerActions = struct {
    ctx: *anyopaque,
    on_quick: *const fn (*anyopaque, usize) void,
    on_all_settings: *const fn (*anyopaque) void,
    on_reference: *const fn (*anyopaque) void,
    /// The noise toggle was clicked.
    on_noise_toggle: *const fn (*anyopaque) void = noopCtx,
    /// A different saved curve was picked from the dropdown.
    on_noise_pick: *const fn (*anyopaque) void = noopCtx,
    /// The amount field was committed (Enter, or focus left it).
    on_noise_amount: *const fn (*anyopaque) void = noopCtx,
};

fn noopCtx(_: *anyopaque) void {}

/// The quick-settings row. These three are the only knobs a beginner needs;
/// everything else lives in Studio, and the link says so.
pub fn quickRow(src: std.builtin.SourceLocation, m: Composer, cb: ComposerActions) void {
    var row = dvui.box(src, .{ .dir = .horizontal }, .{
        .expand = .horizontal,
        .margin = .{ .h = 10 },
    });
    defer row.deinit();

    for (m.quick, 0..) |q, i| {
        if (style.chip(@src(), q.label, .{
            .dropdown = q.dropdown,
            .id_extra = i,
            .fill = C.chip,
            .font = F.mono,
        })) cb.on_quick(cb.ctx, i);
        _ = dvui.spacer(@src(), .{ .id_extra = i, .min_size_content = .{ .w = 8 } });
    }

    var link: dvui.ButtonWidget = undefined;
    link.init(@src(), .{}, .{
        .gravity_x = 1.0,
        .gravity_y = 0.5,
        .background = false,
        .color_fill_hover = style.hover_wash,
        .color_fill_press = style.hover_wash,
        .corner_radius = R.chip,
        .padding = .{ .x = 5, .y = 3, .w = 5, .h = 3 },
        .margin = .{},
    });
    link.processEvents();
    link.drawBackground();
    {
        var lr = dvui.box(@src(), .{ .dir = .horizontal }, .{});
        defer lr.deinit();
        dvui.labelNoFmt(@src(), "All settings in Studio", .{}, .{
            .font = F.ui_sm,
            .color_text = C.text_ghost,
            .padding = .{},
            .gravity_y = 0.5,
        });
        style.mark(@src(), .external, 9, C.text_ghost, .{ .margin = .{ .x = 4 } });
    }
    const clicked = link.clicked();
    link.deinit();
    if (clicked) cb.on_all_settings(cb.ctx);
}

/// The sunken input frame. Split into begin/end so the caller can put its own
/// text entry (with its own buffer, key handling and attachment state) between
/// the two, while the frame, the reference affordance and Send stay here and
/// stay identical everywhere.
pub const InputFrame = struct {
    box: *dvui.BoxWidget,
};

/// `focused` lights the frame's own edge. The entry inside is stripped of
/// chrome, so its built-in focus ring would otherwise sit a pixel off the text
/// (see style.noFocusTheme, which the caller should hand the entry).
pub fn inputBegin(src: std.builtin.SourceLocation, focused: bool) InputFrame {
    return .{ .box = dvui.box(src, .{ .dir = .horizontal }, .{
        .expand = .horizontal,
        .background = true,
        .color_fill = C.sunken,
        .border = style.Edge.all,
        .color_border = if (focused) C.blue else style.hairline_hi,
        .corner_radius = R.input,
        .padding = .{ .x = 14, .y = 12, .w = 12, .h = 12 },
    }) };
}

/// The weight-noise pair inside the input frame: a toggle that reads as ON by
/// filling (not by a caption change, which is unreadable at chip size), and the
/// strength field beside it, dimmed while off so it is visibly inert rather than
/// missing. Deliberately here and not in the quick-settings row above: this is a
/// knob you reach for mid-conversation, per message, the way you reach for Send.
fn noiseControls(src: std.builtin.SourceLocation, nz: Noise, cb: ComposerActions) void {
    var row = dvui.box(src, .{ .dir = .horizontal }, .{ .gravity_y = 0.5, .margin = .{ .w = 8 } });
    defer row.deinit();

    if (style.chip(@src(), "noise", .{
        .fill = if (nz.on) C.blue else C.chip,
        .text = if (nz.on) C.text_hi else C.text_ghost,
        .border = if (nz.on) C.blue else style.hairline,
        .font = F.mono,
    })) cb.on_noise_toggle(cb.ctx);

    _ = dvui.spacer(@src(), .{ .min_size_content = .{ .w = 4 } });

    // The shape preview goes BEFORE the field, so the curve reads left to right
    // as picture-then-formula and the field stays adjacent to the entry it edits.
    {
        var well = dvui.box(@src(), .{ .dir = .horizontal }, .{
            .min_size_content = .{ .w = 46 },
            .max_size_content = .width(46),
            .gravity_y = 0.5,
            .margin = .{ .w = 4 },
        });
        defer well.deinit();
        style.sparkline(@src(), nz.shape, if (!nz.valid) C.amber else if (nz.on) C.blue else C.text_ghost, 20);
    }

    if (nz.names.len != 0) {
        if (style.chipDropdown(@src(), nz.names, nz.sel, .{
            .text = if (!nz.valid) C.amber else if (nz.on) C.text_hi else C.text_ghost,
            .border = if (!nz.valid) C.amber else style.hairline,
            .font = F.mono,
        })) cb.on_noise_pick(cb.ctx);
    }

    _ = dvui.spacer(@src(), .{ .min_size_content = .{ .w = 4 } });

    // How much, beside where. A single number is the one thing about this worth
    // reaching for mid-conversation, which is why it is a field here and the
    // expression is not.
    const live = nz.on and nz.valid and nz.amount_live;
    const res = style.chipInput(@src(), nz.amount_buf, "", 34, .{
        .text_input = if (live) C.text_hi else C.text_ghost,
        .border = style.hairline,
        .font = F.mono,
    });
    // COMMIT BEFORE SYNC: on the frame focus leaves, the field has just committed
    // and is no longer being edited, so syncing first would overwrite the typed
    // text with the old value and then parse that.
    if (res.committed) cb.on_noise_amount(cb.ctx);
    if (!res.editing) {
        var tmp: [16]u8 = undefined;
        const want = std.fmt.bufPrint(&tmp, "{d}", .{nz.amount}) catch "";
        if (!std.mem.eql(u8, std.mem.sliceTo(nz.amount_buf, 0), want)) {
            @memset(nz.amount_buf, 0);
            @memcpy(nz.amount_buf[0..@min(want.len, nz.amount_buf.len - 1)], want);
        }
    }
}

/// Returns true when Send was pressed. `busy` swaps it for Stop.
pub fn inputEnd(f: *InputFrame, m: Composer, cb: ComposerActions) bool {
    if (m.noise) |nz| noiseControls(@src(), nz, cb);
    if (m.can_attach) {
        var ref: dvui.ButtonWidget = undefined;
        ref.init(@src(), .{}, .{
            .gravity_y = 0.5,
            .background = false,
            .color_fill_hover = style.hover_wash,
            .color_fill_press = style.hover_wash,
            .corner_radius = R.chip,
            .padding = .{ .x = 6, .y = 4, .w = 6, .h = 4 },
            .margin = .{ .w = 8 },
        });
        ref.processEvents();
        ref.drawBackground();
        {
            var rr = dvui.box(@src(), .{ .dir = .horizontal }, .{});
            defer rr.deinit();
            style.mark(@src(), .plus, 10, C.text_ghost, .{ .margin = .{ .w = 3 } });
            dvui.labelNoFmt(@src(), "reference", .{}, .{
                .font = F.mono,
                .color_text = C.text_ghost,
                .padding = .{},
                .gravity_y = 0.5,
            });
        }
        const clicked = ref.clicked();
        ref.deinit();
        if (clicked) cb.on_reference(cb.ctx);
    }
    const send = primaryButton(@src(), if (m.busy) "Stop" else "Send", true);
    f.box.deinit();
    return send;
}
