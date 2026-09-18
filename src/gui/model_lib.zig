//! Title-bar chips and menus built from the catalog and the config, plus the
//! adapter that turns a config into the scan request the host takes. The
//! catalog is every host's, merged (`client/models.zig`); nothing here scans.
const std = @import("std");
const catalog = @import("shared").catalog;
const config = @import("shared").config;
const selection = @import("client").selection;
const mirror = @import("client").mirror;
const models = @import("client").models;
const wire = @import("serve").wire;
const model_menu = @import("model_menu.zig");
const model_spec = @import("shared").model_spec;

const Catalog = catalog.Catalog;

/// Ask the host to scan the config's folders and probe its loose files.
pub fn startScan(cfg: *const config.Config, post: *const fn (wire.Request) void) void {
    var dirs: [config.max_model_dirs][]const u8 = undefined;
    var files: [config.max_model_dirs][]const u8 = undefined;
    post(.{ .scan = .{ .dirs = paths(&cfg.model_dirs, &dirs), .files = paths(&cfg.model_files, &files) } });
}

fn paths(list: *const config.ModelDirList, out: *[config.max_model_dirs][]const u8) []const []const u8 {
    var n: usize = 0;
    for (list.slice()) |*d| {
        out[n] = d.path.opt() orelse continue;
        n += 1;
    }
    return out[0..n];
}

/// A one-line summary for Settings: what the last scan found, or that one runs.
/// The count is every host's files merged, since that is what the pickers
/// offer; `m` is only what says whether a scan has happened at all.
pub fn statusLine(m: *const mirror.Mirror, u: *const models.Union, buf: []u8) []const u8 {
    if (m.scanning) return "scanning…";
    if (!m.scannedOnce() and u.cat.entries.len == 0) return "not scanned yet";
    var unusable: usize = 0;
    for (u.cat.entries) |*e| if (e.unused()) {
        unusable += 1;
    };
    const n_hosts = models.count(u.up);
    return std.fmt.bufPrint(buf, "{d} model files{s}, {d} of no use here{s}", .{
        u.cat.entries.len,
        if (n_hosts > 1) " across the hosts" else "",
        unusable,
        if (m.scan_report.bad_folders > 0) " · a folder could not be read" else "",
    }) catch "";
}

// ── Chips and menus ─────────────────────────────────────────────────────────


pub fn llmChip(arena: std.mem.Allocator, cfg: *const config.Config, u: *const models.Union, resident: bool) model_menu.Chip {
    const cat = &u.cat;
    const p = cfg.llm_model.opt() orelse return .{ .label = "no chat model", .empty = true };
    const known = selection.llm(cfg, cat) != null;
    return .{
        .label = cat.refName(p),
        .note = chipNote(arena, u, cat.resolve(p)),
        .resident = resident,
        .warn = if (cat.resolve(p)) |e| !known or (e.llm != null and !e.llm.?.supported) else false,
    };
}

pub fn imageChip(arena: std.mem.Allocator, cfg: *const config.Config, u: *const models.Union, resident: bool) model_menu.Chip {
    const cat = &u.cat;
    const p = cfg.diffusion_model.opt() orelse return .{ .label = "no image model", .empty = true };
    return .{
        .label = cat.refName(p),
        .note = chipNote(arena, u, cat.resolve(p)),
        .resident = resident,
        .warn = imageIncomplete(cfg, cat),
    };
}

fn chipNote(arena: std.mem.Allocator, u: *const models.Union, e: ?*const catalog.Entry) []const u8 {
    const entry = e orelse return "";
    const r = u.byId(entry.id()) orelse return "";
    return where(arena, u, r).note;
}

/// The configured checkpoint lacks a component nothing supplies, judged from
/// the catalog. False when the catalog does not know the file.
pub fn imageIncomplete(cfg: *const config.Config, cat: *const Catalog) bool {
    const e = selection.checkpoint(cfg, cat) orelse return false;
    const ck = e.ckpt.?;
    return model_spec.missing(.{ .family = ck.family, .contents = ck.contents }, .{
        .conditioner = cfg.text_encoder.opt() != null,
        .conditioner2 = cfg.text_encoder_2.opt() != null,
        .decoder = cfg.vae.opt() != null,
    }).any();
}

// ── Where a file is ─────────────────────────────────────────────────────────

/// What a menu row says about which machines hold a file, and whether it can be
/// picked at all.
pub const Where = struct {
    note: []const u8 = "",
    greyed: bool = false,
};

/// Say where a file is, and say nothing when there is nothing to say: one host,
/// or every host can run it. A marker on every row is a marker nobody reads.
///
/// What is counted is hosts that could RUN the file, not hosts holding it: a
/// machine with the checkpoint and no VAE renders nothing, and a count that
/// included it would be a number the user cannot act on.
pub fn where(arena: std.mem.Allocator, u: *const models.Union, r: *const models.Row) Where {
    const n_up = models.count(u.up);
    if (n_up <= 1) return .{}; // one machine: naming it is noise
    const runs = r.render & u.up;
    const held = r.on & u.up;
    if (models.count(runs) == n_up) return .{}; // everywhere

    if (models.first(runs)) |only| {
        if (models.count(runs) == 1) return .{ .note = u.nameOf(only) };
        return .{ .note = std.fmt.allocPrint(arena, "{d} of {d}", .{ models.count(runs), n_up }) catch "" };
    }
    // Held somewhere up, but no machine has everything it needs to run it.
    if (models.first(held)) |only| {
        const who = if (models.count(held) == 1) u.nameOf(only) else "";
        return .{ .note = std.fmt.allocPrint(arena, "{s}{s}missing files", .{
            who,
            if (who.len > 0) " · " else "",
        }) catch "missing files" };
    }
    // Only a host that is down has it. The file is real; the machine is what
    // is missing, so the row says which one rather than vanishing.
    if (models.first(r.on)) |only| {
        return .{
            .greyed = true,
            .note = std.fmt.allocPrint(arena, "only on {s} (down)", .{u.nameOf(only)}) catch "host down",
        };
    }
    return .{};
}

/// Chat models grouped by class. `arena` is the frame arena; nothing here
/// outlives the frame.
pub fn llmMenu(arena: std.mem.Allocator, cfg: *const config.Config, u: *const models.Union) model_menu.Menu {
    var menu: model_menu.Menu = .{
        .none_label = "no chat model",
        .none_selected = cfg.llm_model.opt() == null,
        .empty_note = "no chat models in the model folders",
    };
    const cat = &u.cat;
    const sel = cat.resolve(cfg.llm_model.slice());
    const classes = cat.llmClasses(arena) catch return menu;
    const groups = arena.alloc(model_menu.Group, classes.len) catch return menu;
    for (classes, groups) |c, *g| {
        const items = arena.alloc(model_menu.Item, c.members.len) catch return menu;
        for (c.members, items) |mi, *it| {
            const e = &cat.entries[mi];
            const w = where(arena, u, u.row(mi));
            it.* = .{
                .label = e.stem(),
                .path = e.path,
                .selected = sel == e,
                .greyed = !c.supported or w.greyed,
                .note = if (c.supported) w.note else e.note,
            };
        }
        g.* = .{ .label = c.label, .items = items, .greyed = !c.supported };
    }
    menu.groups = groups;
    return menu;
}

/// Diffusion checkpoints grouped by family, in enum order.
pub fn imageMenu(arena: std.mem.Allocator, cfg: *const config.Config, u: *const models.Union) model_menu.Menu {
    var menu: model_menu.Menu = .{
        .none_label = "no image model",
        .none_selected = cfg.diffusion_model.opt() == null,
        .empty_note = "no image models in the model folders",
    };
    const cat = &u.cat;
    const sel = cat.resolve(cfg.diffusion_model.slice());
    var groups: std.ArrayList(model_menu.Group) = .empty;
    const fams = cat.families();
    inline for (@typeInfo(catalog.Family).@"enum".fields) |ff| {
        const fam: catalog.Family = @enumFromInt(ff.value);
        if (fams.contains(fam)) {
            const idx = cat.checkpoints(arena, fam) catch return menu;
            const items = arena.alloc(model_menu.Item, idx.len) catch return menu;
            for (idx, items) |ci, *it| {
                const e = &cat.entries[ci];
                const w = where(arena, u, u.row(ci));
                it.* = .{
                    .label = e.stem(),
                    .path = e.path,
                    .selected = sel == e,
                    .greyed = w.greyed,
                    .note = w.note,
                };
            }
            groups.append(arena, .{ .label = model_spec.traits(fam).short, .items = items }) catch return menu;
        }
    }
    menu.groups = groups.items;
    return menu;
}

// ── Tests ────────────────────────────────────────────────────────────────────

const testing = std.testing;

fn testCat(gpa: std.mem.Allocator, files: []const []const u8) !Catalog {
    var list: std.ArrayList(catalog.Entry) = .empty;
    defer list.deinit(gpa);
    for (files, 0..) |p, i| try list.append(gpa, .{ .path = p, .size = i + 1, .mtime_ns = 1 });
    return Catalog.fromEntries(gpa, list.items);
}

fn noteFor(u: *const models.Union, arena: std.mem.Allocator, path: []const u8) Where {
    for (u.cat.entries, 0..) |*e, i| {
        if (std.mem.eql(u8, e.path, path)) return where(arena, u, u.row(i));
    }
    return .{ .note = "<not listed>" };
}

// The rule the whole marker rests on: say nothing when there is nothing to say.
// A badge on every row is a badge nobody reads.
test "a file every host can run is not marked" {
    const gpa = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var a = try testCat(gpa, &.{ "/m/x.safetensors", "/m/y.safetensors" });
    defer a.deinit();

    var both = try models.build(gpa, &.{
        .{ .name = "local", .cat = &a },
        .{ .name = "b", .cat = &a },
    });
    defer both.deinit();
    try testing.expectEqualStrings("", noteFor(&both, arena.allocator(), "/m/x.safetensors").note);

    // And with one host there is nothing to compare against either.
    var one = try models.build(gpa, &.{.{ .name = "local", .cat = &a }});
    defer one.deinit();
    try testing.expectEqualStrings("", noteFor(&one, arena.allocator(), "/m/x.safetensors").note);
}

test "a file on one of several hosts names that host" {
    const gpa = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var a = try testCat(gpa, &.{ "/m/x.safetensors", "/m/y.safetensors" });
    defer a.deinit();
    var b = try testCat(gpa, &.{"/m/y.safetensors"});
    defer b.deinit();
    var c = try testCat(gpa, &.{"/m/y.safetensors"});
    defer c.deinit();

    var u = try models.build(gpa, &.{
        .{ .name = "local", .cat = &a },
        .{ .name = "attic", .cat = &b },
        .{ .name = "lydia", .cat = &c },
    });
    defer u.deinit();
    try testing.expectEqualStrings("local", noteFor(&u, arena.allocator(), "/m/x.safetensors").note);
}

test "a file on some hosts is counted" {
    const gpa = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var a = try testCat(gpa, &.{"/m/x.safetensors"});
    defer a.deinit();
    var none = Catalog.init(gpa);
    defer none.deinit();

    var u = try models.build(gpa, &.{
        .{ .name = "local", .cat = &a },
        .{ .name = "attic", .cat = &a },
        .{ .name = "lydia", .cat = &none },
    });
    defer u.deinit();
    try testing.expectEqualStrings("2 of 3", noteFor(&u, arena.allocator(), "/m/x.safetensors").note);
}

// A checkpoint whose side files are on no machine that has it: the row still
// lists, and says what is wrong rather than counting a host that cannot run it.
test "a checkpoint nobody can complete says so" {
    const gpa = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const ck: catalog.Entry = .{ .path = "/m/krea.safetensors", .size = 9, .mtime_ns = 1, .ckpt = .{ .family = .krea2, .contents = .{ .denoiser = true } } };
    var a = try Catalog.fromEntries(gpa, &.{ck});
    defer a.deinit();
    var none = Catalog.init(gpa);
    defer none.deinit();

    var u = try models.build(gpa, &.{
        .{ .name = "local", .cat = &none },
        .{ .name = "attic", .cat = &a },
    });
    defer u.deinit();
    const w = noteFor(&u, arena.allocator(), "/m/krea.safetensors");
    try testing.expectEqualStrings("attic · missing files", w.note);
    try testing.expect(!w.greyed);
}

// A file only a host that is DOWN has stays in the menu, greyed and named: the
// file is real, the machine is what is missing.
test "a file only a down host has is greyed and named" {
    const gpa = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var a = try testCat(gpa, &.{"/m/x.safetensors"});
    defer a.deinit();
    var b = try testCat(gpa, &.{"/m/gone.safetensors"});
    defer b.deinit();

    var u = try models.build(gpa, &.{
        .{ .name = "local", .cat = &a },
        .{ .name = "attic", .cat = &a },
        .{ .name = "cellar", .up = false, .cat = &b },
    });
    defer u.deinit();
    const w = noteFor(&u, arena.allocator(), "/m/gone.safetensors");
    try testing.expect(w.greyed);
    try testing.expectEqualStrings("only on cellar (down)", w.note);
}
