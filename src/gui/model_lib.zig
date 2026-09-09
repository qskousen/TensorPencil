//! The app's live model catalog: the index loaded at startup, a scan running on
//! a worker thread, and the swap-in of its result on the UI thread. Also builds
//! the title-bar chips and menus from the catalog and the config.
//!
//! `cat` is owned by the UI thread. While a scan runs, the worker reads it as the
//! previous catalog (to reuse unchanged entries) and the UI reads it to draw;
//! nobody writes it until the worker has been joined. A second scan requested
//! mid-scan is remembered and started when the first lands.
const std = @import("std");
const catalog = @import("catalog.zig");
const config = @import("config.zig");
const selection = @import("selection.zig");
const model_menu = @import("model_menu.zig");
const model_spec = @import("model_spec.zig");

pub const Catalog = catalog.Catalog;

pub var cat: Catalog = undefined;
var g_gpa: std.mem.Allocator = undefined;
var g_io: std.Io = undefined;
var g_wakeup: ?*const fn () void = null;
var g_index_path: ?[]u8 = null;
var g_ready = false;

var g_thread: ?std.Thread = null;
var g_done = std.atomic.Value(bool).init(false);
var g_result: ?Catalog = null;
var g_report: catalog.ScanReport = .{};
var g_scanned_once = false;
/// The folder and file lists handed to the running worker, owned here until it
/// is joined.
var g_dirs: std.ArrayList([]u8) = .empty;
var g_files: std.ArrayList([]u8) = .empty;
var g_rescan_wanted = false;
/// Lists for the deferred rescan.
var g_next_dirs: std.ArrayList([]u8) = .empty;
var g_next_files: std.ArrayList([]u8) = .empty;

pub fn init(gpa: std.mem.Allocator, io: std.Io, wakeup: ?*const fn () void, index_path: ?[]const u8) void {
    g_gpa = gpa;
    g_io = io;
    g_wakeup = wakeup;
    g_index_path = if (index_path) |p| gpa.dupe(u8, p) catch null else null;
    cat = if (g_index_path) |p| Catalog.load(gpa, io, p) else Catalog.init(gpa);
    g_ready = true;
}

pub fn deinit() void {
    if (!g_ready) return;
    if (g_thread) |t| t.join();
    g_thread = null;
    if (g_result) |*r| r.deinit();
    g_result = null;
    freeDirs(&g_dirs);
    freeDirs(&g_files);
    freeDirs(&g_next_dirs);
    freeDirs(&g_next_files);
    cat.deinit();
    if (g_index_path) |p| g_gpa.free(p);
    g_index_path = null;
    g_ready = false;
}

/// Replace the catalog outright (probes with canned data).
pub fn setCanned(c: Catalog) void {
    if (g_ready) cat.deinit();
    cat = c;
    g_ready = true;
}

fn freeDirs(list: *std.ArrayList([]u8)) void {
    for (list.items) |d| g_gpa.free(d);
    list.deinit(g_gpa);
}

fn copyList(list: *const config.ModelDirList, into: *std.ArrayList([]u8)) void {
    freeDirs(into);
    into.* = .empty;
    for (list.slice()) |*d| {
        const p = d.path.opt() orelse continue;
        into.append(g_gpa, g_gpa.dupe(u8, p) catch continue) catch {};
    }
}

/// Scan the config's folders on a worker thread. Safe to call while one runs:
/// the request is kept and served when the current scan lands.
pub fn startScan(cfg: *const config.Config) void {
    if (!g_ready) return;
    if (g_thread != null) {
        g_rescan_wanted = true;
        copyList(&cfg.model_dirs, &g_next_dirs);
        copyList(&cfg.model_files, &g_next_files);
        return;
    }
    copyList(&cfg.model_dirs, &g_dirs);
    copyList(&cfg.model_files, &g_files);
    g_done.store(false, .release);
    g_thread = std.Thread.spawn(.{}, worker, .{}) catch |err| {
        std.log.err("catalog scan thread: {t}", .{err});
        return;
    };
}

fn worker() void {
    var rep: catalog.ScanReport = .{};
    const dirs: []const []const u8 = @ptrCast(g_dirs.items);
    const files: []const []const u8 = @ptrCast(g_files.items);
    const res = catalog.scan(g_gpa, g_io, dirs, files, &cat, &rep) catch |err| blk: {
        std.log.err("catalog scan failed: {t}", .{err});
        break :blk null;
    };
    g_result = res;
    g_report = rep;
    g_done.store(true, .release);
    if (g_wakeup) |w| w();
}

pub fn scanning() bool {
    return g_thread != null;
}

/// UI thread, once per frame: adopt a finished scan. True when the catalog
/// changed this frame, so the caller re-resolves its selections.
pub fn poll() bool {
    if (!g_ready) return false;
    const t = g_thread orelse return false;
    if (!g_done.load(.acquire)) return false;
    t.join();
    g_thread = null;
    g_scanned_once = true;
    var changed = false;
    if (g_result) |r| {
        cat.deinit();
        cat = r;
        g_result = null;
        changed = true;
        if (g_index_path) |p| cat.save(g_io, g_gpa, p) catch |err| std.log.warn("catalog index save failed: {t}", .{err});
        std.log.info("[catalog] {d} files, {d} probed, {d} reused, {d} bad folders", .{ g_report.files, g_report.probed, g_report.reused, g_report.bad_folders });
    }
    if (g_rescan_wanted) {
        g_rescan_wanted = false;
        freeDirs(&g_dirs);
        freeDirs(&g_files);
        g_dirs = g_next_dirs;
        g_files = g_next_files;
        g_next_dirs = .empty;
        g_next_files = .empty;
        g_done.store(false, .release);
        g_thread = std.Thread.spawn(.{}, worker, .{}) catch |err| blk: {
            std.log.err("catalog scan thread: {t}", .{err});
            break :blk null;
        };
    }
    return changed;
}

/// A file picked from outside the folders: probe it now, so the pick resolves
/// this frame, and keep it in the config so the next scan sees it too.
pub fn addFile(cfg: *config.Config, path: []const u8) void {
    _ = cfg.addModelFile(path);
    if (g_thread != null) {
        // The running scan will not have it; the deferred one will.
        startScan(cfg);
    }
    cat.append(g_gpa, g_io, path) catch |err| std.log.warn("catalog: {s}: {t}", .{ path, err });
}

/// A one-line summary for Settings: what the last scan found, or that one runs.
pub fn statusLine(buf: []u8) []const u8 {
    if (scanning()) return "scanning…";
    if (!g_scanned_once and cat.entries.len == 0) return "not scanned yet";
    var unusable: usize = 0;
    for (cat.entries) |*e| if (e.unused()) {
        unusable += 1;
    };
    return std.fmt.bufPrint(buf, "{d} model files, {d} of no use here{s}", .{
        cat.entries.len,
        unusable,
        if (g_report.bad_folders > 0) " · a folder could not be read" else "",
    }) catch "";
}

// ── Chips and menus ─────────────────────────────────────────────────────────

fn stemOf(path: []const u8) []const u8 {
    const base = std.fs.path.basename(path);
    return base[0 .. std.mem.lastIndexOfScalar(u8, base, '.') orelse base.len];
}

pub fn llmChip(cfg: *const config.Config, resident: bool) model_menu.Chip {
    const p = cfg.llm_model.opt() orelse return .{ .label = "no chat model", .empty = true };
    const known = selection.llm(cfg, &cat) != null;
    return .{ .label = stemOf(p), .resident = resident, .warn = if (cat.find(p)) |e| !known or (e.llm != null and !e.llm.?.supported) else false };
}

pub fn imageChip(cfg: *const config.Config, resident: bool) model_menu.Chip {
    const p = cfg.diffusion_model.opt() orelse return .{ .label = "no image model", .empty = true };
    return .{ .label = stemOf(p), .resident = resident, .warn = imageIncomplete(cfg) };
}

/// The configured checkpoint lacks a component nothing supplies, judged from
/// the catalog. False when the catalog does not know the file.
pub fn imageIncomplete(cfg: *const config.Config) bool {
    const e = selection.checkpoint(cfg, &cat) orelse return false;
    const ck = e.ckpt.?;
    return model_spec.missing(.{ .family = ck.family, .contents = ck.contents }, .{
        .conditioner = cfg.text_encoder.opt() != null,
        .conditioner2 = cfg.text_encoder_2.opt() != null,
        .decoder = cfg.vae.opt() != null,
    }).any();
}

/// Chat models grouped by class. `arena` is the frame arena; nothing here
/// outlives the frame.
pub fn llmMenu(arena: std.mem.Allocator, cfg: *const config.Config) model_menu.Menu {
    var menu: model_menu.Menu = .{
        .none_label = "no chat model",
        .none_selected = cfg.llm_model.opt() == null,
        .empty_note = "no chat models in the model folders",
    };
    const classes = cat.llmClasses(arena) catch return menu;
    const groups = arena.alloc(model_menu.Group, classes.len) catch return menu;
    for (classes, groups) |c, *g| {
        const items = arena.alloc(model_menu.Item, c.members.len) catch return menu;
        for (c.members, items) |mi, *it| {
            const e = &cat.entries[mi];
            it.* = .{
                .label = e.stem(),
                .path = e.path,
                .selected = std.mem.eql(u8, e.path, cfg.llm_model.slice()),
                .greyed = !c.supported,
                .note = if (c.supported) "" else e.note,
            };
        }
        g.* = .{ .label = c.label, .items = items, .greyed = !c.supported };
    }
    menu.groups = groups;
    return menu;
}

/// Diffusion checkpoints grouped by family, in enum order.
pub fn imageMenu(arena: std.mem.Allocator, cfg: *const config.Config) model_menu.Menu {
    var menu: model_menu.Menu = .{
        .none_label = "no image model",
        .none_selected = cfg.diffusion_model.opt() == null,
        .empty_note = "no image models in the model folders",
    };
    var groups: std.ArrayList(model_menu.Group) = .empty;
    const fams = cat.families();
    inline for (@typeInfo(catalog.Family).@"enum".fields) |ff| {
        const fam: catalog.Family = @enumFromInt(ff.value);
        if (fams.contains(fam)) {
            const idx = cat.checkpoints(arena, fam) catch return menu;
            const items = arena.alloc(model_menu.Item, idx.len) catch return menu;
            for (idx, items) |ci, *it| {
                const e = &cat.entries[ci];
                it.* = .{
                    .label = e.stem(),
                    .path = e.path,
                    .selected = std.mem.eql(u8, e.path, cfg.diffusion_model.slice()),
                };
            }
            groups.append(arena, .{ .label = model_spec.traits(fam).short, .items = items }) catch return menu;
        }
    }
    menu.groups = groups.items;
    return menu;
}
