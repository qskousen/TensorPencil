//! The live model catalog: the index loaded at startup, a scan running on a
//! worker thread, and the swap-in of its result on the owning thread. Knows
//! nothing about the config or the UI; it is handed folder and file lists.
//!
//! `cat` is owned by the polling thread. While a scan runs, the worker reads it
//! as the previous catalog (to reuse unchanged entries) and the owner reads it
//! to draw; nobody writes it until the worker has been joined. A second scan
//! requested mid-scan is remembered and started when the first lands.
const std = @import("std");
const catalog = @import("shared").catalog;

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
    freeList(&g_dirs);
    freeList(&g_files);
    freeList(&g_next_dirs);
    freeList(&g_next_files);
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

fn freeList(list: *std.ArrayList([]u8)) void {
    for (list.items) |d| g_gpa.free(d);
    list.deinit(g_gpa);
    list.* = .empty;
}

fn copyList(from: []const []const u8, into: *std.ArrayList([]u8)) void {
    freeList(into);
    for (from) |p| into.append(g_gpa, g_gpa.dupe(u8, p) catch continue) catch {};
}

/// Scan `dirs` and probe `files` on a worker thread. Safe to call while one
/// runs: the request is kept and served when the current scan lands.
pub fn startScan(dirs: []const []const u8, files: []const []const u8) void {
    if (!g_ready) return;
    if (g_thread != null) {
        g_rescan_wanted = true;
        copyList(dirs, &g_next_dirs);
        copyList(files, &g_next_files);
        return;
    }
    copyList(dirs, &g_dirs);
    copyList(files, &g_files);
    spawnWorker();
}

fn spawnWorker() void {
    g_done.store(false, .release);
    g_thread = std.Thread.spawn(.{}, worker, .{}) catch |err| blk: {
        std.log.err("catalog scan thread: {t}", .{err});
        break :blk null;
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

/// What the last finished scan reported.
pub fn lastReport() catalog.ScanReport {
    return g_report;
}

/// Owning thread, once per frame: adopt a finished scan. True when the catalog
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
        freeList(&g_dirs);
        freeList(&g_files);
        g_dirs = g_next_dirs;
        g_files = g_next_files;
        g_next_dirs = .empty;
        g_next_files = .empty;
        spawnWorker();
    }
    return changed;
}

/// A file picked from outside the folders: probed now so the pick resolves this
/// frame, unless a scan is running. The catalog is the worker's until it is
/// joined, so a scan in flight instead gets the file queued onto the deferred
/// one, with the running lists.
pub fn addFile(path: []const u8) void {
    if (g_thread != null) {
        if (!g_rescan_wanted) {
            g_rescan_wanted = true;
            copyList(@ptrCast(g_dirs.items), &g_next_dirs);
            copyList(@ptrCast(g_files.items), &g_next_files);
        }
        g_next_files.append(g_gpa, g_gpa.dupe(u8, path) catch return) catch {};
        return;
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
