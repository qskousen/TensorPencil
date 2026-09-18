//! `zig build catalog-probe -- <folder>...`: scan model folders exactly as tp-gui
//! does, through the same worker module, and print what each file was taken
//! for. The answer to "why is my model not in the menu" without opening the GUI.
//! This binary links no dvui, which is what keeps `model_scan.zig` free of it.
const std = @import("std");
const catalog = @import("shared").catalog;
const model_scan = @import("engine").scan;

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const io = init.io;
    const gpa = std.heap.smp_allocator;

    const args = try init.minimal.args.toSlice(arena);
    var folders: std.ArrayList([]const u8) = .empty;
    for (args[1..]) |a| try folders.append(arena, a);
    if (folders.items.len == 0) {
        std.debug.print("usage: catalog-probe <folder>...\n", .{});
        return error.NoFolders;
    }

    var buf: [8192]u8 = undefined;
    var w = std.Io.File.Writer.initStreaming(.stdout(), io, &buf);
    const out = &w.interface;

    model_scan.init(gpa, io, null, null);
    defer model_scan.deinit();
    const t0 = std.Io.Clock.awake.now(io).nanoseconds;
    model_scan.startScan(folders.items, &.{});
    while (model_scan.scanning()) {
        if (model_scan.poll()) break;
        std.Io.sleep(io, .{ .nanoseconds = 5 * std.time.ns_per_ms }, .real) catch {};
    }
    const ms = @divTrunc(std.Io.Clock.awake.now(io).nanoseconds - t0, std.time.ns_per_ms);
    const rep = model_scan.lastReport();
    const cat = &model_scan.cat;

    for (cat.entries) |*e| {
        try out.print("{s}\n", .{e.path});
        if (e.llm) |l| try out.print("    llm    {s}  arch={s} width={d} blocks={d}{s}{s}\n", .{
            l.class, l.arch, l.width, l.blocks, if (l.vision) " vision" else "", if (l.supported) "" else " UNSUPPORTED",
        });
        if (e.tower) |t| try out.print("    tower  projector={s} serves={s} width={d}\n", .{ t.projector, t.arch orelse "-", t.width });
        if (e.ckpt) |c| try out.print("    ckpt   {t}  bundles: denoiser{s}{s}{s}{s}\n", .{
            c.family,
            if (c.contents.conditioner) " te" else "",
            if (c.contents.conditioner2) " te2" else "",
            if (c.contents.decoder) " vae" else "",
            if (c.contents.decoder2) " vae2" else "",
        });
        inline for (@typeInfo(catalog.Family).@"enum".fields) |ff| {
            const fam: catalog.Family = @enumFromInt(ff.value);
            inline for ([_]catalog.Component{ .conditioner, .conditioner2, .decoder, .decoder2 }) |comp| {
                if (e.side.has(fam, comp)) try out.print("    side   {t}/{t}\n", .{ fam, comp });
            }
            if (e.preview.has(fam)) try out.print("    preview {t}\n", .{fam});
        }
        if (e.lora) |l| {
            try out.print("    lora   {d} linears, rank {d}, depth {d}\n", .{ l.info.targets, l.info.rank, l.info.depth });
            inline for (@typeInfo(catalog.Family).@"enum".fields) |ff| {
                if (l.has(@enumFromInt(ff.value))) try out.print("    lora   fits {s}\n", .{ff.name});
            }
        }
        if (e.note.len > 0) try out.print("    note   {s}\n", .{e.note});
    }
    try out.print("\n{d} files, {d} probed, {d} bad folders, {d} ms\n", .{ rep.files, rep.probed, rep.bad_folders, ms });
    try out.flush();
}
