//! The model catalog: every checkpoint under the user's model folders, read from its
//! header alone and sorted into what it can be, so the chips and the settings screen
//! offer files by architecture instead of asking for paths.
//!
//! One file can be several things at once. A Qwen3-4B GGUF is a chat model AND
//! Z-Image's text encoder; a bundled SDXL checkpoint is a denoiser AND a VAE another
//! SDXL file can borrow. So an `Entry` carries every role independently rather than
//! one tag, and the queries below pick the role they need.
//!
//! What a file IS comes from the engine, never from its name or folder: an LLM from
//! its GGUF metadata (`chat.familyForArch` says whether this build runs it), a
//! diffusion checkpoint from `pipeline.detectFamily`, a side file from
//! `model_spec.storeFits`, which is the pipeline's own probe table plus the width
//! check that tells two encoders with the same tensor names apart. Nothing here
//! enumerates tensor names.
//!
//! Scanning opens headers only (`Container.openHeader`). A folder of forty 15 GB
//! files must not be mapped, let alone prefetched, to draw a menu. Results are kept
//! in a JSON index keyed on path, size and mtime, so a rescan re-reads only what
//! changed. `scan` is synchronous; the app runs it on a worker thread.
const std = @import("std");
const tp = @import("TensorPencil");
const model_spec = @import("model_spec.zig");

const pipeline = tp.pipeline;
const chat = tp.llm.chat;

pub const Family = model_spec.Family;
pub const Component = model_spec.Component;

const family_count = @typeInfo(Family).@"enum".fields.len;
const component_count = @typeInfo(Component).@"enum".fields.len;

/// A chat model. `class` is the menu group it belongs to ("Gemma 4 31B"): the
/// architecture's display name plus the size, because that is how people name
/// models, while `width` is what a vision tower actually has to match.
pub const Llm = struct {
    arch: []const u8,
    size_label: []const u8,
    /// `<arch>.embedding_length`; a vision tower's projection must equal it.
    width: u64,
    blocks: u64,
    /// This build has a stepper for the architecture.
    supported: bool,
    /// The architecture takes a vision tower at all.
    vision: bool,
    class: []const u8,
};

/// A vision tower (mmproj). `arch` is the LLM architecture its projector serves,
/// null when no tower loader here accepts the projector type.
pub const Tower = struct {
    projector: []const u8,
    arch: ?[]const u8,
    /// `clip.vision.projection_dim`: the LLM width it projects into.
    width: u64,
};

/// A diffusion checkpoint: its architecture and which components it bundles.
pub const Ckpt = struct {
    family: Family,
    contents: model_spec.Contents,
};

/// Which (family, component) slots a file can fill as a SIDE file: it carries the
/// component and it is the right width for that family.
pub const Slots = struct {
    bits: [family_count][component_count]bool = @splat(@splat(false)),

    pub fn has(self: Slots, fam: Family, comp: Component) bool {
        return self.bits[@intFromEnum(fam)][@intFromEnum(comp)];
    }
    pub fn set(self: *Slots, fam: Family, comp: Component) void {
        self.bits[@intFromEnum(fam)][@intFromEnum(comp)] = true;
    }
    pub fn any(self: Slots) bool {
        for (self.bits) |row| for (row) |b| if (b) return true;
        return false;
    }
};

/// Which families a preview (approx-VAE) file can decode for.
pub const Preview = struct {
    fams: [family_count]bool = @splat(false),

    pub fn has(self: Preview, fam: Family) bool {
        return self.fams[@intFromEnum(fam)];
    }
    pub fn any(self: Preview) bool {
        for (self.fams) |b| if (b) return true;
        return false;
    }
};

pub const Entry = struct {
    path: []const u8,
    size: u64,
    mtime_ns: i64,
    llm: ?Llm = null,
    tower: ?Tower = null,
    ckpt: ?Ckpt = null,
    side: Slots = .{},
    preview: Preview = .{},
    /// Why the file is offered nowhere, empty when it has a role. Shown greyed so
    /// the user knows the file was seen and why it is not a choice.
    note: []const u8 = "",

    /// The file name without its extension, what a menu shows.
    pub fn stem(self: *const Entry) []const u8 {
        const base = std.fs.path.basename(self.path);
        return base[0 .. std.mem.lastIndexOfScalar(u8, base, '.') orelse base.len];
    }

    /// Nothing here can use this file.
    pub fn unused(self: *const Entry) bool {
        return self.llm == null and self.tower == null and self.ckpt == null and
            !self.side.any() and !self.preview.any();
    }

    /// Offered as a chat model: known architecture, so it will load.
    pub fn isChatModel(self: *const Entry) bool {
        return if (self.llm) |l| l.supported else false;
    }

    fn dupe(self: Entry, a: std.mem.Allocator) !Entry {
        var out = self;
        out.path = try a.dupe(u8, self.path);
        out.note = try a.dupe(u8, self.note);
        if (self.llm) |l| out.llm = .{
            .arch = try a.dupe(u8, l.arch),
            .size_label = try a.dupe(u8, l.size_label),
            .width = l.width,
            .blocks = l.blocks,
            .supported = l.supported,
            .vision = l.vision,
            .class = try a.dupe(u8, l.class),
        };
        if (self.tower) |t| out.tower = .{
            .projector = try a.dupe(u8, t.projector),
            .arch = if (t.arch) |x| try a.dupe(u8, x) else null,
            .width = t.width,
        };
        return out;
    }
};

pub const ScanReport = struct {
    /// Model files seen under the folders.
    files: usize = 0,
    /// Taken from the previous catalog unchanged (same size and mtime).
    reused: usize = 0,
    /// Headers actually opened.
    probed: usize = 0,
    /// Folders that could not be opened or walked.
    bad_folders: usize = 0,
};

pub const Catalog = struct {
    arena: std.heap.ArenaAllocator,
    /// Sorted by path.
    entries: []Entry = &.{},

    pub fn init(gpa: std.mem.Allocator) Catalog {
        return .{ .arena = std.heap.ArenaAllocator.init(gpa) };
    }

    pub fn deinit(self: *Catalog) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// A catalog over ready-made entries (probes, tests): copies them into the
    /// catalog's own arena.
    pub fn fromEntries(gpa: std.mem.Allocator, entries: []const Entry) !Catalog {
        var cat = Catalog.init(gpa);
        errdefer cat.deinit();
        const a = cat.arena.allocator();
        const out = try a.alloc(Entry, entries.len);
        for (entries, out) |e, *o| o.* = try e.dupe(a);
        std.mem.sort(Entry, out, {}, pathLessThan);
        cat.entries = out;
        return cat;
    }

    /// Probe one more file and add it, keeping path order. A path already
    /// present is re-probed in place. For a file picked from outside the folders.
    pub fn append(self: *Catalog, gpa: std.mem.Allocator, io: std.Io, path: []const u8) !void {
        const a = self.arena.allocator();
        const st = std.Io.Dir.cwd().statFile(io, path, .{}) catch |err| return err;
        const e = try probe(a, gpa, io, try a.dupe(u8, path), st.size, clampNs(st.mtime.nanoseconds));
        if (self.indexOf(path)) |i| {
            self.entries[i] = e;
            return;
        }
        const out = try a.alloc(Entry, self.entries.len + 1);
        @memcpy(out[0..self.entries.len], self.entries);
        out[self.entries.len] = e;
        std.mem.sort(Entry, out, {}, pathLessThan);
        self.entries = out;
    }

    pub fn find(self: *const Catalog, path: []const u8) ?*const Entry {
        for (self.entries) |*e| if (std.mem.eql(u8, e.path, path)) return e;
        return null;
    }

    pub fn indexOf(self: *const Catalog, path: []const u8) ?usize {
        for (self.entries, 0..) |*e, i| if (std.mem.eql(u8, e.path, path)) return i;
        return null;
    }

    // ── Queries ─────────────────────────────────────────────────────────────

    /// One menu group of chat models. Supported classes first, then by label, so
    /// what cannot load sits greyed at the bottom rather than mixed in.
    pub const Class = struct {
        label: []const u8,
        arch: []const u8,
        width: u64,
        supported: bool,
        /// Indices into `entries`, in path order.
        members: []const usize,

        fn lessThan(_: void, a: Class, b: Class) bool {
            if (a.supported != b.supported) return a.supported;
            return std.mem.lessThan(u8, a.label, b.label);
        }
    };

    pub fn llmClasses(self: *const Catalog, gpa: std.mem.Allocator) ![]Class {
        var classes: std.ArrayList(Class) = .empty;
        errdefer freeClasses(gpa, classes.items);
        var members: std.ArrayList(std.ArrayList(usize)) = .empty;
        defer {
            for (members.items) |*m| m.deinit(gpa);
            members.deinit(gpa);
        }
        for (self.entries, 0..) |*e, i| {
            const l = e.llm orelse continue;
            const slot: usize = blk: {
                for (classes.items, 0..) |c, ci| {
                    if (std.mem.eql(u8, c.arch, l.arch) and std.mem.eql(u8, c.label, l.class) and c.width == l.width) break :blk ci;
                }
                try classes.append(gpa, .{ .label = l.class, .arch = l.arch, .width = l.width, .supported = l.supported, .members = &.{} });
                try members.append(gpa, .empty);
                break :blk classes.items.len - 1;
            };
            try members.items[slot].append(gpa, i);
        }
        for (classes.items, members.items) |*c, *m| c.members = try m.toOwnedSlice(gpa);
        std.mem.sort(Class, classes.items, {}, Class.lessThan);
        return classes.toOwnedSlice(gpa);
    }

    pub fn freeClasses(gpa: std.mem.Allocator, classes: []Class) void {
        for (classes) |c| gpa.free(c.members);
        gpa.free(classes);
    }

    /// Vision towers that fit chat model `llm_idx`: a projector for its architecture
    /// AND a projection of its width. The width is the part a name cannot tell you:
    /// a 12B Gemma 4 tower projects into 3840 and a 31B model reads 5376.
    pub fn towersFor(self: *const Catalog, gpa: std.mem.Allocator, llm_idx: usize) ![]usize {
        const l = self.entries[llm_idx].llm orelse return gpa.alloc(usize, 0);
        var out: std.ArrayList(usize) = .empty;
        errdefer out.deinit(gpa);
        if (!l.vision) return out.toOwnedSlice(gpa);
        for (self.entries, 0..) |*e, i| {
            const t = e.tower orelse continue;
            const arch = t.arch orelse continue;
            if (std.mem.eql(u8, arch, l.arch) and t.width == l.width) try out.append(gpa, i);
        }
        return out.toOwnedSlice(gpa);
    }

    /// Diffusion checkpoints of one family, in path order.
    pub fn checkpoints(self: *const Catalog, gpa: std.mem.Allocator, fam: Family) ![]usize {
        var out: std.ArrayList(usize) = .empty;
        errdefer out.deinit(gpa);
        for (self.entries, 0..) |*e, i| {
            const c = e.ckpt orelse continue;
            if (c.family == fam) try out.append(gpa, i);
        }
        return out.toOwnedSlice(gpa);
    }

    /// Files that can supply `comp` to a `fam` checkpoint: standalone side files
    /// first, then bundled checkpoints of the same family that carry it (ggufy's
    /// UNet-only GGUF takes its CLIP and VAE from the original checkpoint).
    pub fn sidesFor(self: *const Catalog, gpa: std.mem.Allocator, fam: Family, comp: Component) ![]usize {
        var out: std.ArrayList(usize) = .empty;
        errdefer out.deinit(gpa);
        for (self.entries, 0..) |*e, i| {
            if (e.ckpt == null and e.side.has(fam, comp)) try out.append(gpa, i);
        }
        for (self.entries, 0..) |*e, i| {
            const c = e.ckpt orelse continue;
            if (c.family == fam and c.contents.has(comp)) try out.append(gpa, i);
        }
        return out.toOwnedSlice(gpa);
    }

    /// The first entry `sidesFor` would list, without allocating: what a slot
    /// defaults to when nothing is remembered for the family.
    pub fn firstSide(self: *const Catalog, fam: Family, comp: Component) ?usize {
        for (self.entries, 0..) |*e, i| if (e.ckpt == null and e.side.has(fam, comp)) return i;
        for (self.entries, 0..) |*e, i| if (e.ckpt) |c| if (c.family == fam and c.contents.has(comp)) return i;
        return null;
    }

    pub fn firstCheckpoint(self: *const Catalog, fam: Family) ?usize {
        for (self.entries, 0..) |*e, i| if (e.ckpt) |c| if (c.family == fam) return i;
        return null;
    }

    pub fn firstPreview(self: *const Catalog, fam: Family) ?usize {
        for (self.entries, 0..) |*e, i| if (e.preview.has(fam)) return i;
        return null;
    }

    /// The first entry `towersFor` would list.
    pub fn firstTower(self: *const Catalog, llm_idx: usize) ?usize {
        const l = self.entries[llm_idx].llm orelse return null;
        if (!l.vision) return null;
        for (self.entries, 0..) |*e, i| {
            const t = e.tower orelse continue;
            const arch = t.arch orelse continue;
            if (std.mem.eql(u8, arch, l.arch) and t.width == l.width) return i;
        }
        return null;
    }

    pub fn previewsFor(self: *const Catalog, gpa: std.mem.Allocator, fam: Family) ![]usize {
        var out: std.ArrayList(usize) = .empty;
        errdefer out.deinit(gpa);
        for (self.entries, 0..) |*e, i| if (e.preview.has(fam)) try out.append(gpa, i);
        return out.toOwnedSlice(gpa);
    }

    /// Families with at least one checkpoint, in enum order.
    pub fn families(self: *const Catalog) std.EnumSet(Family) {
        var set = std.EnumSet(Family).initEmpty();
        for (self.entries) |*e| if (e.ckpt) |c| set.insert(c.family);
        return set;
    }

    // ── Index file ──────────────────────────────────────────────────────────

    pub fn save(self: *const Catalog, io: std.Io, gpa: std.mem.Allocator, path: []const u8) !void {
        const json = try std.json.Stringify.valueAlloc(gpa, self.entries, .{ .whitespace = .indent_2 });
        defer gpa.free(json);
        if (std.fs.path.dirname(path)) |dir| {
            std.Io.Dir.cwd().createDirPath(io, dir) catch |err| switch (err) {
                error.PathAlreadyExists => {},
                else => return err,
            };
        }
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = json });
    }

    /// A missing or unreadable index is an empty catalog, not an error: the next
    /// scan rebuilds it.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, path: []const u8) Catalog {
        var cat = Catalog.init(gpa);
        const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(64 << 20)) catch return cat;
        defer gpa.free(bytes);
        const a = cat.arena.allocator();
        // `alloc_always`: by default a string that needs no unescaping is a slice
        // of the input, which is freed on return.
        cat.entries = std.json.parseFromSliceLeaky([]Entry, a, bytes, .{
            .ignore_unknown_fields = true,
            .allocate = .alloc_always,
        }) catch return cat;
        return cat;
    }
};

// ── Scanning ────────────────────────────────────────────────────────────────

fn isModelFile(name: []const u8) bool {
    const ext = std.fs.path.extension(name);
    return std.ascii.eqlIgnoreCase(ext, ".gguf") or std.ascii.eqlIgnoreCase(ext, ".safetensors");
}

fn clampNs(ns: i96) i64 {
    if (ns > std.math.maxInt(i64)) return std.math.maxInt(i64);
    if (ns < std.math.minInt(i64)) return std.math.minInt(i64);
    return @intCast(ns);
}

fn pathLessThan(_: void, a: Entry, b: Entry) bool {
    return std.mem.lessThan(u8, a.path, b.path);
}

/// Walk `folders` (recursively) and classify every model file, plus the single
/// `files` named outright (a file picked from outside the folders), taking an
/// entry from `prev` unchanged when its size and mtime still match. A file reached
/// twice, through a folder and its parent, is one entry. `gpa` is scratch for
/// the header parses; the result owns its own arena.
pub fn scan(
    gpa: std.mem.Allocator,
    io: std.Io,
    folders: []const []const u8,
    files: []const []const u8,
    prev: ?*const Catalog,
    report: ?*ScanReport,
) !Catalog {
    var cat = Catalog.init(gpa);
    errdefer cat.deinit();
    const a = cat.arena.allocator();
    var rep: ScanReport = .{};
    defer if (report) |r| {
        r.* = rep;
    };

    var seen: std.StringHashMapUnmanaged(void) = .empty;
    defer seen.deinit(gpa);
    var list: std.ArrayList(Entry) = .empty;
    for (files) |f| {
        const st = std.Io.Dir.cwd().statFile(io, f, .{}) catch continue;
        if (st.kind != .file) continue;
        const path = try a.dupe(u8, f);
        if ((try seen.getOrPut(gpa, path)).found_existing) continue;
        try classifyOrReuse(a, gpa, io, &list, prev, path, st, &rep);
    }
    for (folders) |folder| {
        // Through cwd so a relative folder works too (tests); an absolute one is
        // opened as itself.
        var dir = std.Io.Dir.cwd().openDir(io, folder, .{ .iterate = true }) catch {
            rep.bad_folders += 1;
            continue;
        };
        defer dir.close(io);
        var walker = try dir.walk(gpa);
        defer walker.deinit();
        while (walker.next(io) catch {
            rep.bad_folders += 1;
            break;
        }) |ent| {
            if (ent.kind != .file and ent.kind != .sym_link) continue;
            if (!isModelFile(ent.basename)) continue;
            const st = ent.dir.statFile(io, ent.basename, .{}) catch continue;
            if (st.kind != .file) continue;
            const path = try std.fs.path.join(a, &.{ folder, ent.path });
            if ((try seen.getOrPut(gpa, path)).found_existing) continue;
            try classifyOrReuse(a, gpa, io, &list, prev, path, st, &rep);
        }
    }
    std.mem.sort(Entry, list.items, {}, pathLessThan);
    cat.entries = list.items;
    return cat;
}

fn classifyOrReuse(
    a: std.mem.Allocator,
    gpa: std.mem.Allocator,
    io: std.Io,
    list: *std.ArrayList(Entry),
    prev: ?*const Catalog,
    path: []const u8,
    st: std.Io.File.Stat,
    rep: *ScanReport,
) !void {
    rep.files += 1;
    const mtime = clampNs(st.mtime.nanoseconds);
    if (prev) |p| if (p.find(path)) |old| if (old.size == st.size and old.mtime_ns == mtime) {
        var e = try old.dupe(a);
        e.path = path;
        try list.append(a, e);
        rep.reused += 1;
        return;
    };
    rep.probed += 1;
    try list.append(a, try probe(a, gpa, io, path, st.size, mtime));
}

/// Classify one file from its header. Never fails: a file that cannot be read
/// becomes an entry with a note, because "this was seen and refused" is what
/// the user needs to know about it.
pub fn probe(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, path: []const u8, size: u64, mtime_ns: i64) !Entry {
    var e: Entry = .{ .path = path, .size = size, .mtime_ns = mtime_ns };
    var c = pipeline.Container.openHeader(gpa, io, path) catch |err| {
        e.note = try std.fmt.allocPrint(a, "unreadable: {t}", .{err});
        return e;
    };
    defer c.deinit();
    const store = c.store();

    var unsupported_arch: ?[]const u8 = null;
    var kbuf: [96]u8 = undefined;
    if (c == .gguf) {
        const g = &c.gguf;
        if (g.getStr("general.architecture")) |arch| {
            if (g.isVisionProjector()) {
                const proj = g.getStr("clip.vision.projector_type") orelse g.getStr("clip.projector_type") orelse "";
                e.tower = .{
                    .projector = try a.dupe(u8, proj),
                    .arch = if (chat.archForProjector(proj)) |x| try a.dupe(u8, x) else null,
                    .width = g.getUint("clip.vision.projection_dim") orelse g.getUint("clip.vision.embedding_length") orelse 0,
                };
            } else if (!std.mem.eql(u8, g.getStr("general.type") orelse "model", "model")) {
                // An adapter, a control vector: metadata says it is not a model.
            } else if (g.getUint(archKey(&kbuf, arch, "embedding_length")) != null) {
                // A language model has a width. A diffusion GGUF (wan, flux, a
                // krea2 DiT) carries an architecture string and nothing else, and
                // is classified by its tensors below.
                e.llm = try llmInfo(a, g, store, arch);
                if (!e.llm.?.supported) unsupported_arch = e.llm.?.arch;
            } else if (chat.archLabel(arch) == null) {
                unsupported_arch = try a.dupe(u8, arch);
            }
        }
    }

    if (pipeline.detectFamily(store)) |fam| {
        e.ckpt = .{ .family = fam, .contents = model_spec.scan(store, fam) };
    } else |_| {}

    inline for (@typeInfo(Family).@"enum".fields) |ff| {
        const fam: Family = @enumFromInt(ff.value);
        inline for ([_]Component{ .conditioner, .conditioner2, .decoder, .decoder2 }) |comp| {
            if (model_spec.storeFits(store, fam, comp)) e.side.set(fam, comp);
        }
        if (model_spec.previewFits(store, fam)) e.preview.fams[ff.value] = true;
    }

    if (e.unused()) {
        if (e.tower) |t| {
            if (t.arch == null) e.note = try std.fmt.allocPrint(a, "vision projector '{s}' has no tower here", .{t.projector});
        } else if (unsupported_arch) |arch| {
            e.note = try std.fmt.allocPrint(a, "architecture '{s}' is not supported", .{arch});
        } else {
            e.note = "not a model this build knows";
        }
    }
    // A tower is listed under its LLM, so on its own it is not "unused"; only an
    // unsupported projector leaves it with nowhere to go.
    if (e.tower) |t| if (t.arch == null and e.note.len == 0) {
        e.note = try std.fmt.allocPrint(a, "vision projector '{s}' has no tower here", .{t.projector});
    };
    // Likewise an LLM this build cannot run keeps its class (so it is shown greyed
    // in its group) and gets the note.
    if (e.llm) |l| if (!l.supported and e.note.len == 0) {
        e.note = try std.fmt.allocPrint(a, "architecture '{s}' is not supported", .{l.arch});
    };
    return e;
}

fn archKey(buf: []u8, arch: []const u8, key: []const u8) []const u8 {
    return std.fmt.bufPrint(buf, "{s}.{s}", .{ arch, key }) catch "";
}

fn llmInfo(a: std.mem.Allocator, g: *const tp.gguf.Gguf, store: tp.weights.WeightStore, arch: []const u8) !Llm {
    var kbuf: [96]u8 = undefined;
    const width = g.getUint(archKey(&kbuf, arch, "embedding_length")) orelse 0;
    const blocks = g.getUint(archKey(&kbuf, arch, "block_count")) orelse 0;
    const size_label: []const u8 = if (g.getStr("general.size_label")) |s| try sizeLabel(a, s) else try paramLabel(a, store);
    const label = chat.archLabel(arch);
    return .{
        .arch = try a.dupe(u8, arch),
        .size_label = size_label,
        .width = width,
        .blocks = blocks,
        .supported = label != null,
        .vision = chat.archHasVision(arch),
        .class = try std.fmt.allocPrint(a, "{s} {s}", .{ label orelse arch, size_label }),
    };
}

/// Quantizers spell the same model "4B" and "4.0B"; the zero fraction is noise
/// that would split one class in two. Owned by `a`.
fn sizeLabel(a: std.mem.Allocator, s: []const u8) ![]const u8 {
    if (std.mem.indexOf(u8, s, ".0")) |dot| if (dot + 3 == s.len and std.ascii.isUpper(s[dot + 2])) {
        return std.fmt.allocPrint(a, "{s}{c}", .{ s[0..dot], s[dot + 2] });
    };
    return a.dupe(u8, s);
}

/// "31B" from the tensor shapes, for a GGUF whose converter wrote no size label.
fn paramLabel(a: std.mem.Allocator, store: tp.weights.WeightStore) ![]const u8 {
    var params: u64 = 0;
    for (store.names()) |n| {
        const v = store.get(n) orelse continue;
        params += v.info.shape.count();
    }
    if (params >= 1_000_000_000) return std.fmt.allocPrint(a, "{d}B", .{(params + 500_000_000) / 1_000_000_000});
    return std.fmt.allocPrint(a, "{d}M", .{(params + 500_000) / 1_000_000});
}

// ── Tests ──────────────────────────────────────────────────────────────────

const testing = std.testing;

/// A safetensors file whose tensors are all zero: header plus a payload of the
/// right length, which is all a header-only probe ever checks.
fn stFile(gpa: std.mem.Allocator, specs: []const struct { name: []const u8, dims: []const usize }) ![]u8 {
    var header: std.ArrayList(u8) = .empty;
    defer header.deinit(gpa);
    var off: usize = 0;
    try header.append(gpa, '{');
    for (specs, 0..) |sp, i| {
        var n: usize = 4;
        for (sp.dims) |d| n *= d;
        if (i > 0) try header.append(gpa, ',');
        try header.print(gpa, "\"{s}\":{{\"dtype\":\"F32\",\"shape\":[", .{sp.name});
        for (sp.dims, 0..) |d, j| try header.print(gpa, "{s}{d}", .{ if (j > 0) "," else "", d });
        try header.print(gpa, "],\"data_offsets\":[{d},{d}]}}", .{ off, off + n });
        off += n;
    }
    try header.append(gpa, '}');
    const out = try gpa.alloc(u8, 8 + header.items.len + off);
    std.mem.writeInt(u64, out[0..8], header.items.len, .little);
    @memcpy(out[8 .. 8 + header.items.len], header.items);
    @memset(out[8 + header.items.len ..], 0);
    return out;
}

const TensorSpec = struct { name: []const u8, ne: []const u64 };

/// A GGUF of f32 tensors with the given (ggml, reversed) dims and the kv
/// strings/uints given. `token_embd.weight` canonicalizes to `embed_tokens.weight`,
/// `blk.N.attn_norm.weight` to `layers.N.input_layernorm.weight`.
fn ggufFile(
    gpa: std.mem.Allocator,
    strs: []const [2][]const u8,
    uints: []const struct { []const u8, u32 },
    tensors: []const TensorSpec,
) ![]u8 {
    var b = try tp.gguf.TestBuilder.init(gpa, 3, tensors.len, strs.len + uints.len);
    defer b.deinit();
    for (strs) |kv| try b.kvStr(kv[0], kv[1]);
    for (uints) |kv| try b.kvUint(kv[0], kv[1]);
    var off: usize = 0;
    for (tensors) |t| {
        try b.tensor(t.name, t.ne, 0, off);
        var bytes: usize = 4;
        for (t.ne) |d| bytes *= @intCast(d);
        off = std.mem.alignForward(usize, off + bytes, 32);
    }
    const data = try gpa.alloc(u8, off);
    defer gpa.free(data);
    @memset(data, 0);
    return b.finish(data);
}

const Tree = struct {
    tmp: testing.TmpDir,
    /// cwd-relative path of the tmp dir, which `scan` opens through cwd.
    root: []u8,

    fn init(gpa: std.mem.Allocator) !Tree {
        var tmp = testing.tmpDir(.{});
        errdefer tmp.cleanup();
        const root = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}", .{tmp.sub_path});
        return .{ .tmp = tmp, .root = root };
    }

    fn put(self: *Tree, io: std.Io, sub: []const u8, data: []const u8) !void {
        if (std.fs.path.dirname(sub)) |d| self.tmp.dir.createDirPath(io, d) catch {};
        try self.tmp.dir.writeFile(io, .{ .sub_path = sub, .data = data });
    }

    fn deinit(self: *Tree, gpa: std.mem.Allocator) void {
        gpa.free(self.root);
        self.tmp.cleanup();
    }
};

fn fixtureTree(gpa: std.mem.Allocator, io: std.Io) !Tree {
    var t = try Tree.init(gpa);
    errdefer t.deinit(gpa);

    // Qwen3-4B: a chat model AND Z-Image's / Krea2-class encoder by width 2560.
    const q4b = try ggufFile(gpa, &.{ .{ "general.architecture", "qwen3" }, .{ "general.size_label", "4.0B" } }, &.{ .{ "qwen3.embedding_length", 2560 }, .{ "qwen3.block_count", 36 } }, &.{
        .{ .name = "token_embd.weight", .ne = &.{ 2560, 8 } },
        .{ .name = "blk.35.attn_norm.weight", .ne = &.{2560} },
    });
    defer gpa.free(q4b);
    try t.put(io, "llm/sub/Qwen3-4B-Q4_K_M.gguf", q4b);

    // Qwen3-0.6B: Anima's encoder, not Z-Image's.
    const q06 = try ggufFile(gpa, &.{ .{ "general.architecture", "qwen3" }, .{ "general.size_label", "0.6B" } }, &.{ .{ "qwen3.embedding_length", 1024 }, .{ "qwen3.block_count", 28 } }, &.{
        .{ .name = "token_embd.weight", .ne = &.{ 1024, 8 } },
        .{ .name = "blk.27.attn_norm.weight", .ne = &.{1024} },
    });
    defer gpa.free(q06);
    try t.put(io, "te/qwen3_06b.gguf", q06);

    // Gemma 4 31B, no size label: computed from the one tensor (8 x 5376 = 43k -> "0M").
    const g31 = try ggufFile(gpa, &.{.{ "general.architecture", "gemma4" }}, &.{ .{ "gemma4.embedding_length", 5376 }, .{ "gemma4.block_count", 60 } }, &.{.{ .name = "token_embd.weight", .ne = &.{ 5376, 8 } }});
    defer gpa.free(g31);
    try t.put(io, "llm/Gemma-4-31B.gguf", g31);

    // A 5120-wide, 64-layer chat model: H3's encoder width, not its depth.
    const q33 = try ggufFile(gpa, &.{ .{ "general.architecture", "qwen3" }, .{ "general.size_label", "33B" } }, &.{ .{ "qwen3.embedding_length", 5120 }, .{ "qwen3.block_count", 64 } }, &.{
        .{ .name = "token_embd.weight", .ne = &.{ 5120, 8 } },
        .{ .name = "blk.49.attn_norm.weight", .ne = &.{5120} },
        .{ .name = "blk.50.attn_norm.weight", .ne = &.{5120} },
        .{ .name = "blk.63.attn_norm.weight", .ne = &.{5120} },
    });
    defer gpa.free(q33);
    try t.put(io, "llm/Qwen3-33B.gguf", q33);

    // A Krea2 GGUF denoiser: an architecture string and no LLM metadata.
    const kdit = try ggufFile(gpa, &.{.{ "general.architecture", "krea2" }}, &.{}, &.{.{ .name = "blocks.0.attn.wq.weight", .ne = &.{ 8, 8 } }});
    defer gpa.free(kdit);
    try t.put(io, "ckpt/krea2/krea2-q8_0.gguf", kdit);

    // Two towers: the 12B one (3840) and a 31B one (5376). Only the second fits.
    const t12 = try ggufFile(gpa, &.{ .{ "general.architecture", "clip" }, .{ "general.type", "mmproj" }, .{ "clip.vision.projector_type", "gemma4uv" } }, &.{.{ "clip.vision.projection_dim", 3840 }}, &.{});
    defer gpa.free(t12);
    try t.put(io, "llm/mmproj-gemma-4-12b.gguf", t12);
    const t31 = try ggufFile(gpa, &.{ .{ "general.architecture", "clip" }, .{ "general.type", "mmproj" }, .{ "clip.vision.projector_type", "gemma4v" } }, &.{.{ "clip.vision.projection_dim", 5376 }}, &.{});
    defer gpa.free(t31);
    try t.put(io, "llm/mmproj-gemma-4-31b.gguf", t31);

    // An architecture this build has no stepper for.
    const bert = try ggufFile(gpa, &.{ .{ "general.architecture", "bert" }, .{ "general.size_label", "137M" } }, &.{.{ "bert.embedding_length", 768 }}, &.{.{ .name = "token_embd.weight", .ne = &.{ 768, 4 } }});
    defer gpa.free(bert);
    try t.put(io, "llm/nomic-embed.gguf", bert);

    // A bundled SDXL checkpoint: UNet + CLIP-L + VAE, no CLIP-G.
    const sdxl = try stFile(gpa, &.{
        .{ .name = "model.diffusion_model.label_emb.0.0.weight", .dims = &.{ 4, 4 } },
        .{ .name = "model.diffusion_model.input_blocks.0.0.weight", .dims = &.{ 4, 4, 3, 3 } },
        .{ .name = "conditioner.embedders.0.transformer.text_model.final_layer_norm.weight", .dims = &.{768} },
        .{ .name = "first_stage_model.decoder.conv_in.weight", .dims = &.{ 8, 4, 3, 3 } },
    });
    defer gpa.free(sdxl);
    try t.put(io, "ckpt/sdxl/dreamshaperXL.safetensors", sdxl);

    // A standalone 4-channel KL VAE (SD family) and a 16-channel one (Z-Image).
    const vae4 = try stFile(gpa, &.{.{ .name = "decoder.conv_in.weight", .dims = &.{ 8, 4, 3, 3 } }});
    defer gpa.free(vae4);
    try t.put(io, "vae/sdxl.vae.safetensors", vae4);
    const vae16 = try stFile(gpa, &.{.{ .name = "decoder.conv_in.weight", .dims = &.{ 8, 16, 3, 3 } }});
    defer gpa.free(vae16);
    try t.put(io, "vae/ae.safetensors", vae16);

    // The Wan approx-VAE preview decoder.
    const taew = try stFile(gpa, &.{.{ .name = "decoder.1.weight", .dims = &.{ 4, 16, 3, 3 } }});
    defer gpa.free(taew);
    try t.put(io, "vae_approx/taew2_1.safetensors", taew);

    // Junk with the right extension, and a file the walker must skip.
    try t.put(io, "vae/broken.safetensors", "not a checkpoint at all");
    try t.put(io, "README.md", "hello");
    return t;
}

fn entryNamed(cat: *const Catalog, stem: []const u8) *const Entry {
    for (cat.entries) |*e| if (std.mem.eql(u8, e.stem(), stem)) return e;
    unreachable;
}

test "scan classifies every fixture by what the engine says it is" {
    const gpa = testing.allocator;
    const io = testing.io;
    var tree = try fixtureTree(gpa, io);
    defer tree.deinit(gpa);

    var rep: ScanReport = .{};
    var cat = try scan(gpa, io, &.{tree.root}, &.{}, null, &rep);
    defer cat.deinit();
    errdefer std.debug.print("report {any}\n", .{rep});
    try testing.expectEqual(@as(usize, 13), rep.files);
    try testing.expectEqual(@as(usize, 13), rep.probed);
    try testing.expectEqual(@as(usize, 0), rep.bad_folders);
    try testing.expectEqual(@as(usize, 13), cat.entries.len);

    const q4b = entryNamed(&cat, "Qwen3-4B-Q4_K_M");
    try testing.expect(q4b.isChatModel());
    try testing.expectEqualStrings("Qwen 3 4B", q4b.llm.?.class);
    try testing.expect(!q4b.llm.?.vision);
    // ...and it is Z-Image's encoder, Krea2's by width too, but never Anima's.
    try testing.expect(q4b.side.has(.zimage, .conditioner));
    try testing.expect(!q4b.side.has(.anima, .conditioner));
    try testing.expect(!q4b.side.has(.sd15, .conditioner));

    const q06 = entryNamed(&cat, "qwen3_06b");
    try testing.expect(q06.side.has(.anima, .conditioner));
    try testing.expect(!q06.side.has(.zimage, .conditioner));

    // Right width for H3, wrong depth: not its encoder. Still a chat model.
    const q33 = entryNamed(&cat, "Qwen3-33B");
    try testing.expect(q33.isChatModel());
    try testing.expect(!q33.side.has(.minimax_h3, .conditioner));
    try testing.expect(!q33.side.any());

    // A diffusion GGUF is a checkpoint, never an "unsupported LLM".
    const kdit = entryNamed(&cat, "krea2-q8_0");
    try testing.expect(kdit.llm == null);
    try testing.expectEqual(Family.krea2, kdit.ckpt.?.family);
    try testing.expectEqual(@as(usize, 0), kdit.note.len);

    const g31 = entryNamed(&cat, "Gemma-4-31B");
    try testing.expect(g31.isChatModel());
    try testing.expect(g31.llm.?.vision);
    try testing.expectEqual(@as(u64, 5376), g31.llm.?.width);
    try testing.expectEqualStrings("Gemma 4 0M", g31.llm.?.class); // 43k params in the fixture

    const bert = entryNamed(&cat, "nomic-embed");
    try testing.expect(!bert.isChatModel());
    try testing.expect(bert.llm != null and !bert.llm.?.supported);
    try testing.expect(std.mem.indexOf(u8, bert.note, "bert") != null);

    const sdxl = entryNamed(&cat, "dreamshaperXL");
    try testing.expectEqual(Family.sdxl, sdxl.ckpt.?.family);
    try testing.expect(sdxl.ckpt.?.contents.conditioner and sdxl.ckpt.?.contents.decoder);
    try testing.expect(!sdxl.ckpt.?.contents.conditioner2);

    const vae4 = entryNamed(&cat, "sdxl.vae");
    try testing.expect(vae4.side.has(.sd15, .decoder) and vae4.side.has(.sdxl, .decoder));
    try testing.expect(!vae4.side.has(.zimage, .decoder));
    const vae16 = entryNamed(&cat, "ae");
    try testing.expect(vae16.side.has(.zimage, .decoder));
    try testing.expect(!vae16.side.has(.sdxl, .decoder));

    const taew = entryNamed(&cat, "taew2_1");
    try testing.expect(taew.preview.has(.krea2) and taew.preview.has(.anima));
    try testing.expect(!taew.preview.has(.zimage));

    const broken = entryNamed(&cat, "broken");
    try testing.expect(broken.unused());
    try testing.expect(std.mem.startsWith(u8, broken.note, "unreadable"));
}

test "queries: classes, towers by width, side donors, previews" {
    const gpa = testing.allocator;
    const io = testing.io;
    var tree = try fixtureTree(gpa, io);
    defer tree.deinit(gpa);
    var cat = try scan(gpa, io, &.{tree.root}, &.{}, null, null);
    defer cat.deinit();

    const classes = try cat.llmClasses(gpa);
    defer Catalog.freeClasses(gpa, classes);
    // Two supported classes sorted by label, the unsupported one last.
    try testing.expectEqual(@as(usize, 5), classes.len);
    try testing.expectEqualStrings("Gemma 4 0M", classes[0].label);
    try testing.expectEqualStrings("Qwen 3 0.6B", classes[1].label);
    try testing.expectEqualStrings("Qwen 3 33B", classes[2].label);
    try testing.expectEqualStrings("Qwen 3 4B", classes[3].label); // "4.0B" normalized
    try testing.expect(!classes[4].supported);
    try testing.expectEqual(@as(usize, 1), classes[3].members.len);

    // Only the 5376-wide tower fits the 31B, though both are gemma4 towers.
    const g31 = cat.indexOf(entryNamed(&cat, "Gemma-4-31B").path).?;
    const towers = try cat.towersFor(gpa, g31);
    defer gpa.free(towers);
    try testing.expectEqual(@as(usize, 1), towers.len);
    try testing.expectEqualStrings("mmproj-gemma-4-31b", cat.entries[towers[0]].stem());
    // A text-only model gets no towers at all.
    const q4b = cat.indexOf(entryNamed(&cat, "Qwen3-4B-Q4_K_M").path).?;
    const none = try cat.towersFor(gpa, q4b);
    defer gpa.free(none);
    try testing.expectEqual(@as(usize, 0), none.len);

    // SDXL VAE candidates: the standalone file first, then the bundled checkpoint.
    const vaes = try cat.sidesFor(gpa, .sdxl, .decoder);
    defer gpa.free(vaes);
    try testing.expectEqual(@as(usize, 2), vaes.len);
    try testing.expectEqualStrings("sdxl.vae", cat.entries[vaes[0]].stem());
    try testing.expectEqualStrings("dreamshaperXL", cat.entries[vaes[1]].stem());
    // Nobody has a CLIP-G, so the SDXL second tower has no candidates.
    const te2 = try cat.sidesFor(gpa, .sdxl, .conditioner2);
    defer gpa.free(te2);
    try testing.expectEqual(@as(usize, 0), te2.len);

    const ck = try cat.checkpoints(gpa, .sdxl);
    defer gpa.free(ck);
    try testing.expectEqual(@as(usize, 1), ck.len);
    try testing.expect(cat.families().contains(.sdxl));
    try testing.expect(cat.families().contains(.krea2));
    try testing.expect(!cat.families().contains(.anima));

    const pv = try cat.previewsFor(gpa, .anima);
    defer gpa.free(pv);
    try testing.expectEqual(@as(usize, 1), pv.len);
}

test "a folder inside another listed folder, or a named file, is one entry" {
    const gpa = testing.allocator;
    const io = testing.io;
    var tree = try fixtureTree(gpa, io);
    defer tree.deinit(gpa);
    const sub = try std.fs.path.join(gpa, &.{ tree.root, "llm" });
    defer gpa.free(sub);
    const one = try std.fs.path.join(gpa, &.{ tree.root, "vae", "ae.safetensors" });
    defer gpa.free(one);

    var rep: ScanReport = .{};
    var cat = try scan(gpa, io, &.{ tree.root, sub }, &.{one}, null, &rep);
    defer cat.deinit();
    try testing.expectEqual(@as(usize, 13), cat.entries.len);
    try testing.expectEqual(@as(usize, 13), rep.files);

    // Only the file: one entry, and `append` re-probes it in place.
    var solo = try scan(gpa, io, &.{}, &.{one}, null, null);
    defer solo.deinit();
    try testing.expectEqual(@as(usize, 1), solo.entries.len);
    try testing.expect(solo.entries[0].side.has(.zimage, .decoder));
    const two = try std.fs.path.join(gpa, &.{ tree.root, "vae", "sdxl.vae.safetensors" });
    defer gpa.free(two);
    try solo.append(gpa, io, two);
    try solo.append(gpa, io, one);
    try testing.expectEqual(@as(usize, 2), solo.entries.len);
    try testing.expect(solo.find(two).?.side.has(.sdxl, .decoder));
    try testing.expectError(error.FileNotFound, solo.append(gpa, io, "/nonexistent/x.safetensors"));
}

test "index round trip, and a rescan reuses unchanged files" {
    const gpa = testing.allocator;
    const io = testing.io;
    var tree = try fixtureTree(gpa, io);
    defer tree.deinit(gpa);
    var cat = try scan(gpa, io, &.{tree.root}, &.{}, null, null);
    defer cat.deinit();

    const idx = try std.fs.path.join(gpa, &.{ tree.root, "index", "catalog.json" });
    defer gpa.free(idx);
    try cat.save(io, gpa, idx);
    var loaded = Catalog.load(gpa, io, idx);
    defer loaded.deinit();
    try testing.expectEqual(cat.entries.len, loaded.entries.len);
    for (cat.entries, loaded.entries) |a, b| {
        try testing.expectEqualStrings(a.path, b.path);
        try testing.expectEqual(a.size, b.size);
        try testing.expectEqual(a.mtime_ns, b.mtime_ns);
        try testing.expectEqual(a.ckpt, b.ckpt);
        try testing.expectEqual(a.side, b.side);
        try testing.expectEqual(a.preview, b.preview);
        try testing.expectEqualStrings(a.note, b.note);
        try testing.expectEqual(a.llm != null, b.llm != null);
        if (a.llm) |l| try testing.expectEqualStrings(l.class, b.llm.?.class);
    }

    // Everything unchanged: nothing is re-opened. Touch one file and only it is.
    var rep: ScanReport = .{};
    var again = try scan(gpa, io, &.{tree.root}, &.{}, &loaded, &rep);
    defer again.deinit();
    try testing.expectEqual(rep.files, rep.reused);
    try testing.expectEqual(@as(usize, 0), rep.probed);

    try tree.put(io, "vae/broken.safetensors", "still not a checkpoint, but longer");
    var rep2: ScanReport = .{};
    var third = try scan(gpa, io, &.{tree.root}, &.{}, &again, &rep2);
    defer third.deinit();
    try testing.expectEqual(@as(usize, 1), rep2.probed);
    try testing.expectEqual(rep2.files - 1, rep2.reused);

    // A missing index loads as empty, a missing folder is counted, not fatal.
    var empty = Catalog.load(gpa, io, "/nonexistent/tp-gui-catalog.json");
    defer empty.deinit();
    try testing.expectEqual(@as(usize, 0), empty.entries.len);
    var rep3: ScanReport = .{};
    var gone = try scan(gpa, io, &.{"/nonexistent/tp-gui-models"}, &.{}, null, &rep3);
    defer gone.deinit();
    try testing.expectEqual(@as(usize, 1), rep3.bad_folders);
}
