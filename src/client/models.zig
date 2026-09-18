//! Every host's catalog as one list: one row per model FILE, however many
//! machines hold it, joined on `catalog.Entry.id()` (stem, size and header
//! hash, which is what makes the same file on two machines one model).
//!
//! A picker reads the merged catalog exactly as it read one host's, and the
//! parallel `Row` says where that file is. Two rows for one model would say
//! the model belongs to a machine, and it does not: a render is placed by the
//! scheduler, not by which menu the model was picked from.
//!
//! Sources are passed in the order the bits are read back in, and the caller
//! puts the LOCAL host first. That decides which entry represents a model, and
//! so what a pick stores: a file on this machine is represented by its real
//! path, a file only a remote host holds by the `id:` text its catalog carries.
//! Both resolve on every host (`engine/host.zig` resolves ids against its own
//! catalog; `hosts.postSettingsTo` rewrites a path to an id for a remote one),
//! so this adds no new rule about references.
//!
//! Build this when a host's catalog moves, not per frame: the join is
//! quadratic in entries, and the menus are drawn from it every frame.
const std = @import("std");
const catalog = @import("shared").catalog;
const model_spec = @import("shared").model_spec;

const Catalog = catalog.Catalog;

/// One bit per source, indexed as the caller passed them.
pub const Set = u16;
pub const max_sources = @bitSizeOf(Set);

pub fn bit(i: usize) Set {
    return @as(Set, 1) << @intCast(i);
}

pub fn has(s: Set, i: usize) bool {
    return i < max_sources and s & bit(i) != 0;
}

pub fn count(s: Set) usize {
    return @popCount(s);
}

/// The lowest source in `s`, null when empty.
pub fn first(s: Set) ?usize {
    if (s == 0) return null;
    return @ctz(s);
}

/// One host's catalog, as the union takes it.
pub const Source = struct {
    name: []const u8,
    /// Answering right now. A file only a host that is down holds is still a
    /// row: the file is real, the machine is what is missing, and that is what
    /// the user needs told.
    up: bool = true,
    /// Its catalog has arrived. Nothing is known about a host that has not
    /// scanned, so it contributes nothing rather than reading as empty.
    scanned: bool = true,
    cat: *const Catalog,
};

/// Where one model file is.
pub const Row = struct {
    id: catalog.ModelId,
    /// Hosts holding the file.
    on: Set = 0,
    /// Of those, the ones that could also RUN it: a checkpoint needs a file for
    /// every component it does not bundle, and the side file has to be on the
    /// SAME machine, so a host holding the checkpoint alone loads nothing.
    /// Equal to `on` for anything that is not a checkpoint.
    render: Set = 0,
};

/// The merged catalog plus, per entry, where that file is.
pub const Union = struct {
    gpa: std.mem.Allocator,
    cat: Catalog,
    /// Parallel to `cat.entries`.
    rows: []Row = &.{},
    /// Source names, owned: a slot can be dropped while this stands.
    names: [][]u8 = &.{},
    /// Sources that were up, and that had scanned, when this was built.
    up: Set = 0,
    scanned: Set = 0,

    pub fn deinit(self: *Union) void {
        for (self.names) |n| self.gpa.free(n);
        self.gpa.free(self.names);
        self.gpa.free(self.rows);
        self.cat.deinit();
        self.* = undefined;
    }

    pub fn nameOf(self: *const Union, i: usize) []const u8 {
        return if (i < self.names.len) self.names[i] else "";
    }

    /// The row for `cat.entries[i]`.
    pub fn row(self: *const Union, i: usize) *const Row {
        return &self.rows[i];
    }

    pub fn byId(self: *const Union, id: catalog.ModelId) ?*const Row {
        for (self.rows) |*r| if (r.id == id) return r;
        return null;
    }

    /// The row for whatever a settings field holds, id text or path.
    pub fn resolve(self: *const Union, ref: []const u8) ?*const Row {
        for (self.cat.entries, self.rows) |*e, *r| {
            if (std.mem.eql(u8, e.path, ref)) return r;
        }
        const id = catalog.parseId(ref) orelse return null;
        return self.byId(id);
    }

    /// Hosts up and holding every file needed to run `ref`, as a count.
    pub fn runnableOn(self: *const Union, ref: []const u8) usize {
        const r = self.resolve(ref) orelse return 0;
        return count(r.render & self.up);
    }
};

/// An empty union, so a caller has something valid before the first build.
pub fn empty(gpa: std.mem.Allocator) Union {
    return .{ .gpa = gpa, .cat = Catalog.init(gpa) };
}

pub fn build(gpa: std.mem.Allocator, sources: []const Source) !Union {
    var uniq: std.ArrayList(catalog.Entry) = .empty;
    defer uniq.deinit(gpa);
    var seen: std.AutoHashMapUnmanaged(catalog.ModelId, void) = .empty;
    defer seen.deinit(gpa);

    var up: Set = 0;
    var scanned: Set = 0;
    const n = @min(sources.len, max_sources);
    for (sources[0..n], 0..) |src, si| {
        if (src.up) up |= bit(si);
        if (!src.scanned) continue;
        scanned |= bit(si);
        for (src.cat.entries) |*e| {
            const gop = try seen.getOrPut(gpa, e.id());
            if (gop.found_existing) continue;
            try uniq.append(gpa, e.*);
        }
    }

    var u: Union = .{
        .gpa = gpa,
        .cat = try Catalog.fromEntries(gpa, uniq.items),
        .up = up,
        .scanned = scanned,
    };
    errdefer u.deinit();

    u.names = try gpa.alloc([]u8, n);
    @memset(u.names, &.{});
    for (sources[0..n], u.names) |src, *slot| slot.* = try gpa.dupe(u8, src.name);

    u.rows = try gpa.alloc(Row, u.cat.entries.len);
    for (u.cat.entries, u.rows) |*e, *r| {
        const id = e.id();
        r.* = .{ .id = id };
        for (sources[0..n], 0..) |src, si| {
            if (!src.scanned) continue;
            const he = src.cat.byId(id) orelse continue;
            r.on |= bit(si);
            if (runsAlone(src.cat, he)) r.render |= bit(si);
        }
    }
    return u;
}

/// Can this host load the file with what else it has: a checkpoint needs
/// something for every component it does not carry, and that side file has to
/// be on the same machine. Anything that is not a checkpoint answers yes, since
/// nothing else is loaded on its own.
///
/// This asks what the host COULD supply, not what the settings currently name:
/// the question a menu row answers is whether picking this model would give the
/// machine a render it can run at all.
fn runsAlone(cat: *const Catalog, e: *const catalog.Entry) bool {
    const c = e.ckpt orelse return true;
    return !model_spec.missing(
        .{ .family = c.family, .contents = c.contents },
        .{
            .conditioner = cat.firstSide(c.family, .conditioner) != null,
            .conditioner2 = cat.firstSide(c.family, .conditioner2) != null,
            .decoder = cat.firstSide(c.family, .decoder) != null,
            .decoder2 = cat.firstSide(c.family, .decoder2) != null,
        },
    ).any();
}

// ── Tests ────────────────────────────────────────────────────────────────────

const testing = std.testing;

fn entry(path: []const u8, size: u64) catalog.Entry {
    return .{ .path = path, .size = size, .mtime_ns = 0 };
}

fn ckptEntry(path: []const u8, size: u64, fam: catalog.Family, contents: model_spec.Contents) catalog.Entry {
    var e = entry(path, size);
    e.ckpt = .{ .family = fam, .contents = contents };
    return e;
}

fn sideEntry(path: []const u8, size: u64, fam: catalog.Family, comp: catalog.Component) catalog.Entry {
    var e = entry(path, size);
    e.side.set(fam, comp);
    return e;
}

// The same file on two machines is one row, and the local path represents it.
test "a file on two hosts is one row" {
    const gpa = testing.allocator;
    var a = try Catalog.fromEntries(gpa, &.{entry("/models/foo.safetensors", 100)});
    defer a.deinit();
    // What a remote host sends: an id text, with the stem carried separately.
    var remote_e = entry("", 100);
    remote_e.name = "foo";
    var id_buf: [catalog.id_text_len]u8 = undefined;
    remote_e.path = catalog.idText(catalog.modelId("foo", 100, 0), &id_buf);
    var b = try Catalog.fromEntries(gpa, &.{remote_e});
    defer b.deinit();

    var u = try build(gpa, &.{
        .{ .name = "local", .cat = &a },
        .{ .name = "rtxpro6k", .cat = &b },
    });
    defer u.deinit();

    try testing.expectEqual(@as(usize, 1), u.cat.entries.len);
    try testing.expectEqualStrings("/models/foo.safetensors", u.cat.entries[0].path);
    try testing.expectEqual(@as(usize, 2), count(u.row(0).on));
}

// A host that holds the checkpoint but no VAE cannot render it, and that is
// not the same answer as not holding it.
test "holding a checkpoint is not being able to run it" {
    const gpa = testing.allocator;
    const ck = ckptEntry("/m/krea.safetensors", 100, .krea2, .{ .denoiser = true });
    const te = sideEntry("/m/qwen.safetensors", 200, .krea2, .conditioner);
    const vae = sideEntry("/m/wan.safetensors", 300, .krea2, .decoder);

    var whole = try Catalog.fromEntries(gpa, &.{ ck, te, vae });
    defer whole.deinit();
    var partial = try Catalog.fromEntries(gpa, &.{ ck, te });
    defer partial.deinit();

    var u = try build(gpa, &.{
        .{ .name = "local", .cat = &whole },
        .{ .name = "b", .cat = &partial },
    });
    defer u.deinit();

    const r = u.byId(ck.id()).?;
    try testing.expectEqual(@as(usize, 2), count(r.on));
    try testing.expectEqual(@as(usize, 1), count(r.render));
    try testing.expect(has(r.render, 0));
    try testing.expect(!has(r.render, 1));
}

// A host that has not scanned contributes nothing, rather than reading as a
// machine that holds none of the files.
test "an unscanned host is not an empty one" {
    const gpa = testing.allocator;
    var a = try Catalog.fromEntries(gpa, &.{entry("/m/foo.safetensors", 100)});
    defer a.deinit();
    var none = Catalog.init(gpa);
    defer none.deinit();

    var u = try build(gpa, &.{
        .{ .name = "local", .cat = &a },
        .{ .name = "b", .scanned = false, .cat = &none },
    });
    defer u.deinit();

    try testing.expectEqual(@as(Set, 0b01), u.scanned);
    try testing.expectEqual(@as(usize, 1), count(u.row(0).on));
}

// A file only a host that is down holds is still listed: `on` names it, and
// nothing in `up` does.
test "a down host still accounts for its files" {
    const gpa = testing.allocator;
    var none = Catalog.init(gpa);
    defer none.deinit();
    var b = try Catalog.fromEntries(gpa, &.{entry("/m/foo.safetensors", 100)});
    defer b.deinit();

    var u = try build(gpa, &.{
        .{ .name = "local", .cat = &none },
        .{ .name = "b", .up = false, .cat = &b },
    });
    defer u.deinit();

    try testing.expectEqual(@as(usize, 1), u.cat.entries.len);
    try testing.expectEqual(@as(Set, 0b01), u.up);
    try testing.expectEqual(@as(usize, 0), count(u.row(0).on & u.up));
}
