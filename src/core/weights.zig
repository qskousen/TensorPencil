//! WeightStore, uniform read access over checkpoint containers
//! (safetensors, GGUF), so model loaders don't care which format a
//! checkpoint ships in. Both containers resolve HF-style tensor names
//! (GGUF llama.cpp names are canonicalized at parse time; see gguf.zig).
//!
//! Three arms wrap another store: `Prefixed` (one component of a bundled file),
//! `Overlay` (a few tensors substituted from caller-owned memory, so an experiment
//! can change one weight without rewriting the checkpoint) and `Mix` (each layer
//! taken whole from one of two stores).

const std = @import("std");
const safetensors = @import("safetensors.zig");
const gguf = @import("gguf.zig");
const dtypes = @import("dtype.zig");

pub const TensorView = safetensors.TensorView;

pub const WeightStore = union(enum) {
    safetensors: *const safetensors.SafeTensors,
    gguf: *const gguf.Gguf,
    overlay: *const Overlay,
    prefixed: *const Prefixed,
    mix: *const Mix,

    pub fn get(self: WeightStore, name: []const u8) ?TensorView {
        return switch (self) {
            .safetensors => |st| st.get(name),
            .gguf => |g| g.get(name),
            .overlay => |o| o.get(name),
            .prefixed => |pfx| pfx.get(name),
            .mix => |m| m.get(name),
        };
    }

    /// Like `get`, but a missing tensor is an error, for required weights.
    pub fn require(self: WeightStore, name: []const u8) !TensorView {
        return self.get(name) orelse error.MissingTensor;
    }

    pub fn count(self: WeightStore) usize {
        return switch (self) {
            .safetensors => |st| st.count(),
            .gguf => |g| g.count(),
            .overlay => |o| o.base.count(),
            .prefixed => |pfx| pfx.stripped.len,
            .mix => |m| m.name_list.items.len,
        };
    }

    pub fn names(self: WeightStore) []const []const u8 {
        return switch (self) {
            .safetensors => |st| st.names(),
            .gguf => |g| g.names(),
            // Exactly the base's names: a patch may only *substitute* a tensor the
            // base already has (`Overlay.put` enforces it), so the namespace is
            // unchanged and this stays honest without allocating.
            .overlay => |o| o.base.names(),
            // The *stripped* names, built once at construction: a consumer that
            // enumerates a prefixed view and then looks the results up must get the
            // same namespace back, or every lookup misses.
            .prefixed => |pfx| pfx.stripped,
            .mix => |m| m.name_list.items,
        };
    }

    /// The whole-file mmap backing the tensor bytes (for page-locked direct
    /// GPU streaming); null on the buffered-read path.
    pub fn mapping(self: WeightStore) ?[]align(std.heap.page_size_min) const u8 {
        return switch (self) {
            .safetensors => |st| st.mapping,
            .gguf => |g| g.mapping,
            // A prefixed view relabels names only; every tensor's bytes are still the
            // base's, at the same offsets, so direct GPU streaming stays valid.
            .prefixed => |pfx| pfx.base.mapping(),
            // Deliberately null rather than the base's mapping. A patched tensor's
            // bytes are caller-owned and lie OUTSIDE the base mapping, so a
            // consumer that turns a view into a mapping-relative offset (which is
            // what this accessor is for) would compute a garbage offset for exactly
            // the tensors an experiment changed, and read the unpatched bytes.
            // Losing the streaming fast path is a slowdown; that would be a silent
            // wrong answer.
            .overlay => null,
            // Two mappings, so no single one is right for every tensor.
            .mix => null,
        };
    }
};

/// A view of one component inside a larger container: `get("decoder.conv_in.weight")`
/// resolves `prefix ++ "decoder.conv_in.weight"` in the base.
///
/// This is what makes container style orthogonal to architecture. Any model ships either
/// bundled (denoiser, text encoder and VAE in one file under their own prefixes) or as
/// separate files. Without this every loader needs a prefix parameter and every caller
/// needs to know which spelling a file used; with it, a loader is handed a store in which
/// its component sits at the root, exactly as if it came from a dedicated file.
///
/// `names()` returns the STRIPPED names, built once here. A consumer that
/// enumerates the view and then looks those names up (the activation-capture sanity
/// gate does exactly that) would otherwise miss on every one of them.
pub const Prefixed = struct {
    base: WeightStore,
    prefix: []const u8,
    /// Names under `prefix`, with it removed. Owned by this struct.
    stripped: []const []const u8,

    /// Borrows `base` and `prefix`; both must outlive this view (and the models
    /// loaded through it).
    pub fn init(gpa: std.mem.Allocator, base: WeightStore, prefix: []const u8) !Prefixed {
        var list: std.ArrayList([]const u8) = .empty;
        errdefer list.deinit(gpa);
        for (base.names()) |n| {
            if (std.mem.startsWith(u8, n, prefix)) try list.append(gpa, n[prefix.len..]);
        }
        return .{ .base = base, .prefix = prefix, .stripped = try list.toOwnedSlice(gpa) };
    }

    pub fn deinit(self: *Prefixed, gpa: std.mem.Allocator) void {
        gpa.free(self.stripped);
        self.* = undefined;
    }

    pub fn get(self: *const Prefixed, name: []const u8) ?TensorView {
        // Stack-composed rather than allocated: names are short and this is on the
        // load path for every weight.
        var buf: [512]u8 = undefined;
        if (self.prefix.len + name.len > buf.len) return null;
        @memcpy(buf[0..self.prefix.len], self.prefix);
        @memcpy(buf[self.prefix.len..][0..name.len], name);
        return self.base.get(buf[0 .. self.prefix.len + name.len]);
    }

    pub fn store(self: *const Prefixed) WeightStore {
        return .{ .prefixed = self };
    }
};

/// A base store with a few tensors substituted from caller-owned memory.
///
/// Exists for per-layer attribution: quantize exactly ONE tensor and run the whole
/// model. Writing a modified checkpoint per data point costs a full copy of the file
/// (tens of GB); here it costs one tensor's worth of memory. Being a plain read-through
/// map it also serves merged LoRA/delta weights, per-layer dtype experiments and a GUI
/// hot-swap.
///
/// Two lifetime rules, both because model loaders keep VIEWS into the store rather than
/// copies: the patch bytes must outlive the model, and the `Overlay` must stay at a
/// stable address for as long as a `WeightStore` refers to it.
///
/// A patch may change a tensor's dtype (f32 for the base's bf16 is the normal case,
/// since f32 holds a dequantized round-trip exactly) but not its shape, which is
/// checked. A differently-shaped substitution is a bug every time, and catching it here
/// beats a puzzling `ShapeMismatch` from a loader 200 weights later.
pub const Overlay = struct {
    base: WeightStore,
    patch: std.StringHashMapUnmanaged(TensorView) = .empty,

    /// Frees the map's own memory. The patch *bytes* and the key strings are
    /// caller-owned and are not touched.
    pub fn deinit(self: *Overlay, gpa: std.mem.Allocator) void {
        self.patch.deinit(gpa);
        self.* = undefined;
    }

    /// Substitute `name`. `bytes` and `name` are borrowed, not copied.
    ///
    /// The replacement keeps the base tensor's shape and takes the given dtype, so
    /// `bytes` must be exactly that many bytes, the one thing a caller can get
    /// wrong here (an f32 buffer handed in as bf16 reads as noise, and every
    /// downstream number would be about a model nobody meant to run).
    ///
    /// Errors: `MissingTensor` if the base has no such tensor (the overlay may
    /// not extend the namespace, see `WeightStore.names`), `LengthMismatch` if
    /// `bytes` does not match shape × dtype.
    pub fn put(self: *Overlay, gpa: std.mem.Allocator, name: []const u8, dt: dtypes.DType, bytes: []const u8) !void {
        const orig = self.base.get(name) orelse return error.MissingTensor;
        var info = orig.info;
        if (dt.storageBytes(info.elemCount()) != bytes.len) return error.LengthMismatch;
        info.dtype = dt;
        info.name = name;
        // Offsets describe a position in a container's data section, which a patch
        // has none of. Keep them self-consistent (start 0, end = byteLen) so
        // `byteLen()` still answers correctly for anything that asks.
        info.start = 0;
        info.end = bytes.len;
        try self.patch.put(gpa, name, .{ .info = info, .bytes = bytes });
    }

    /// `put`, taking a whole view, for a replacement that already carries its own
    /// metadata (another container's tensor, say). The shape must match the base's.
    pub fn putView(self: *Overlay, gpa: std.mem.Allocator, name: []const u8, view: TensorView) !void {
        const orig = self.base.get(name) orelse return error.MissingTensor;
        if (!view.info.shape.eql(orig.info.shape)) return error.ShapeMismatch;
        try self.patch.put(gpa, name, view);
    }

    /// Drop a substitution, restoring the base's tensor.
    pub fn remove(self: *Overlay, name: []const u8) bool {
        return self.patch.remove(name);
    }

    /// Drop every substitution. Cheap enough to call between data points, which
    /// is how a per-tensor sweep reuses one overlay.
    pub fn clear(self: *Overlay) void {
        self.patch.clearRetainingCapacity();
    }

    pub fn get(self: *const Overlay, name: []const u8) ?TensorView {
        if (self.patch.get(name)) |v| return v;
        return self.base.get(name);
    }

    /// A store view of this overlay. Takes a pointer for rule 2 above.
    pub fn store(self: *const Overlay) WeightStore {
        return .{ .overlay = self };
    }
};

/// Two stores with each layer taken whole from one of them, `a` unless the layer is
/// listed in `from_b`.
///
/// A layer is a tensor name minus its last component, so a quantized layer's weight,
/// scales and `comfy_quant` move together. That is what an `Overlay` cannot do: a
/// dense and a quantized checkpoint differ in shape and in namespace, not just bytes.
///
/// Both stores and every listed layer name are borrowed. The `Mix` must stay at a
/// stable address while a model loaded through it is alive, as with `Overlay`.
pub const Mix = struct {
    a: WeightStore,
    b: WeightStore,
    from_b: std.StringHashMapUnmanaged(void) = .empty,
    name_list: std.ArrayList([]const u8) = .empty,

    /// `a`'s namespace, before any layer is taken from `b`.
    pub fn init(gpa: std.mem.Allocator, a: WeightStore, b: WeightStore) !Mix {
        var m: Mix = .{ .a = a, .b = b };
        try m.rebuild(gpa);
        return m;
    }

    pub fn deinit(self: *Mix, gpa: std.mem.Allocator) void {
        self.from_b.deinit(gpa);
        self.name_list.deinit(gpa);
        self.* = undefined;
    }

    pub fn layerOf(name: []const u8) []const u8 {
        const dot = std.mem.lastIndexOfScalar(u8, name, '.') orelse return name;
        return name[0..dot];
    }

    /// Serve exactly `layers` from `b` and everything else from `a`.
    pub fn select(self: *Mix, gpa: std.mem.Allocator, layers: []const []const u8) !void {
        self.from_b.clearRetainingCapacity();
        for (layers) |l| try self.from_b.put(gpa, l, {});
        try self.rebuild(gpa);
    }

    fn fromB(self: *const Mix, name: []const u8) bool {
        return self.from_b.contains(layerOf(name));
    }

    fn rebuild(self: *Mix, gpa: std.mem.Allocator) !void {
        self.name_list.clearRetainingCapacity();
        for (self.a.names()) |n| if (!self.fromB(n)) try self.name_list.append(gpa, n);
        for (self.b.names()) |n| if (self.fromB(n)) try self.name_list.append(gpa, n);
    }

    pub fn get(self: *const Mix, name: []const u8) ?TensorView {
        return if (self.fromB(name)) self.b.get(name) else self.a.get(name);
    }

    pub fn store(self: *const Mix) WeightStore {
        return .{ .mix = self };
    }
};

test "weight store dispatches to both containers" {
    const gpa = std.testing.allocator;

    // safetensors arm
    const header =
        \\{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}
    ;
    var st_file: [8 + header.len + 8]u8 = undefined;
    std.mem.writeInt(u64, st_file[0..8], header.len, .little);
    @memcpy(st_file[8..][0..header.len], header);
    const vals = [2]f32{ 3.0, -1.0 };
    @memcpy(st_file[8 + header.len ..], std.mem.sliceAsBytes(&vals));
    var st = try safetensors.SafeTensors.initFromSlice(gpa, &st_file);
    defer st.deinit();

    const store: WeightStore = .{ .safetensors = &st };
    try std.testing.expectEqual(@as(usize, 1), store.count());
    const w = try store.require("w");
    try std.testing.expectEqual(@as(usize, 2), w.info.elemCount());
    try std.testing.expect(store.get("nope") == null);
    try std.testing.expectError(error.MissingTensor, store.require("nope"));
    try std.testing.expect(store.mapping() == null);
}

/// Two tensors, `a` (f32[2]) and `b` (f32[2]), in a caller-owned safetensors slice.
fn twoTensorFile(buf: []u8) ![]u8 {
    const header =
        \\{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[2],"data_offsets":[8,16]}}
    ;
    std.mem.writeInt(u64, buf[0..8], header.len, .little);
    @memcpy(buf[8..][0..header.len], header);
    const vals = [4]f32{ 1.0, 2.0, 3.0, 4.0 };
    @memcpy(buf[8 + header.len ..][0..16], std.mem.sliceAsBytes(&vals));
    return buf[0 .. 8 + header.len + 16];
}

test "overlay substitutes one tensor and passes the rest through" {
    const gpa = std.testing.allocator;
    var file: [256]u8 = undefined;
    const slice = try twoTensorFile(&file);
    var st = try safetensors.SafeTensors.initFromSlice(gpa, slice);
    defer st.deinit();

    var ov: Overlay = .{ .base = .{ .safetensors = &st } };
    defer ov.deinit(gpa);
    const store = ov.store();

    // Untouched overlay == base.
    try std.testing.expectEqual(@as(usize, 2), store.count());
    {
        const a0 = try (try store.require("a")).toF32Alloc(gpa);
        defer gpa.free(a0);
        try std.testing.expectEqualSlices(f32, &.{ 1.0, 2.0 }, a0);
    }

    // Substitute `a`, keeping the shape and changing the dtype (f32 -> bf16), the
    // reverse of the normal direction but the same check.
    const patched = [2]u16{ dtypes.f32ToBf16(-7.0), dtypes.f32ToBf16(0.5) };
    try ov.put(gpa, "a", .bf16, std.mem.sliceAsBytes(&patched));

    const a = try (try store.require("a")).toF32Alloc(gpa);
    defer gpa.free(a);
    try std.testing.expectEqualSlices(f32, &.{ -7.0, 0.5 }, a);
    try std.testing.expectEqual(dtypes.DType.bf16, (store.get("a").?).info.dtype);

    // `b` still reads through, and the namespace is unchanged.
    const b = try (try store.require("b")).toF32Alloc(gpa);
    defer gpa.free(b);
    try std.testing.expectEqualSlices(f32, &.{ 3.0, 4.0 }, b);
    try std.testing.expectEqual(@as(usize, 2), store.count());
    try std.testing.expectEqualStrings("a", store.names()[0]);
    try std.testing.expect(store.get("nope") == null);

    // Restoring is exact, this is what lets one overlay serve a whole sweep.
    try std.testing.expect(ov.remove("a"));
    const a2 = try (try store.require("a")).toF32Alloc(gpa);
    defer gpa.free(a2);
    try std.testing.expectEqualSlices(f32, &.{ 1.0, 2.0 }, a2);
}

test "overlay refuses a patch it cannot honestly represent" {
    const gpa = std.testing.allocator;
    var file: [256]u8 = undefined;
    const slice = try twoTensorFile(&file);
    var st = try safetensors.SafeTensors.initFromSlice(gpa, slice);
    defer st.deinit();

    var ov: Overlay = .{ .base = .{ .safetensors = &st } };
    defer ov.deinit(gpa);

    // Not in the base: the overlay may substitute, never extend, because
    // `names()`/`count()` report the base's namespace.
    const bytes = [8]u8{ 0, 0, 0, 0, 0, 0, 0, 0 };
    try std.testing.expectError(error.MissingTensor, ov.put(gpa, "nope", .f32, &bytes));

    // Right dtype, wrong length, the mistake that would otherwise be read as
    // valid weights.
    try std.testing.expectError(error.LengthMismatch, ov.put(gpa, "a", .f32, bytes[0..4]));
    // Right length in bytes, wrong dtype for that length (2 f32s != 8 bf16 slots).
    try std.testing.expectError(error.LengthMismatch, ov.put(gpa, "a", .bf16, &bytes));

    // A whole-view patch is checked on shape rather than length.
    var wrong = st.get("a").?;
    wrong.info.shape = @import("tensor.zig").Shape.init(&.{ 1, 2 });
    try std.testing.expectError(error.ShapeMismatch, ov.putView(gpa, "a", wrong));
    try std.testing.expectEqual(@as(usize, 0), ov.patch.count());
}

test "an overlay withholds the base mapping" {
    // Because a patched tensor's bytes are NOT inside it, and this accessor exists
    // for turning a view into a mapping-relative offset. Needs a real mmap-backed
    // base for the assertion to mean anything.
    const gpa = std.testing.allocator;
    const io = std.testing.io;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var file: [256]u8 = undefined;
    const slice = try twoTensorFile(&file);
    {
        const f = try tmp.dir.createFile(io, "w.safetensors", .{ .truncate = true });
        defer f.close(io);
        var buf: [64]u8 = undefined;
        var w = f.writer(io, &buf);
        try w.interface.writeAll(slice);
        try w.interface.flush();
    }

    var st = try safetensors.SafeTensors.openIn(gpa, io, tmp.dir, "w.safetensors");
    defer st.deinit();
    const base: WeightStore = .{ .safetensors = &st };
    try std.testing.expect(base.mapping() != null);

    var ov: Overlay = .{ .base = base };
    defer ov.deinit(gpa);
    try std.testing.expect(ov.store().mapping() == null);
}

/// A safetensors file of f32 tensors in a caller-owned buffer.
fn f32File(buf: []u8, comptime header: []const u8, vals: []const f32) ![]u8 {
    std.mem.writeInt(u64, buf[0..8], header.len, .little);
    @memcpy(buf[8..][0..header.len], header);
    const data = std.mem.sliceAsBytes(vals);
    @memcpy(buf[8 + header.len ..][0..data.len], data);
    return buf[0 .. 8 + header.len + data.len];
}

test "mix moves a layer whole, sidecars included, and back" {
    const gpa = std.testing.allocator;
    // `a` is dense; `b` stores layer `l` at a different shape plus a scale `a` lacks.
    var fa: [256]u8 = undefined;
    var st_a = try safetensors.SafeTensors.initFromSlice(gpa, try f32File(&fa,
        \\{"k.weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"l.weight":{"dtype":"F32","shape":[2],"data_offsets":[8,16]}}
    , &.{ 1, 2, 3, 4 }));
    defer st_a.deinit();
    var fb: [256]u8 = undefined;
    var st_b = try safetensors.SafeTensors.initFromSlice(gpa, try f32File(&fb,
        \\{"k.weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"l.weight":{"dtype":"F32","shape":[1],"data_offsets":[8,12]},"l.weight_scale":{"dtype":"F32","shape":[1],"data_offsets":[12,16]}}
    , &.{ 9, 9, 5, 6 }));
    defer st_b.deinit();

    var mix = try Mix.init(gpa, .{ .safetensors = &st_a }, .{ .safetensors = &st_b });
    defer mix.deinit(gpa);
    const store = mix.store();
    try std.testing.expectEqual(@as(usize, 2), store.count());
    try std.testing.expect(store.get("l.weight_scale") == null);
    try std.testing.expectEqual(@as(usize, 2), store.get("l.weight").?.info.elemCount());

    try mix.select(gpa, &.{"l"});
    try std.testing.expectEqual(@as(usize, 3), store.count());
    try std.testing.expectEqual(@as(usize, 1), store.get("l.weight").?.info.elemCount());
    const scale = try (try store.require("l.weight_scale")).toF32Alloc(gpa);
    defer gpa.free(scale);
    try std.testing.expectEqualSlices(f32, &.{6}, scale);
    // An unselected layer still reads from `a`, even though `b` has it too.
    const k = try (try store.require("k.weight")).toF32Alloc(gpa);
    defer gpa.free(k);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2 }, k);
    for (store.names()) |n| try std.testing.expect(store.get(n) != null);
    try std.testing.expect(store.mapping() == null);

    try mix.select(gpa, &.{});
    try std.testing.expectEqual(@as(usize, 2), store.count());
    try std.testing.expect(store.get("l.weight_scale") == null);
    const l = try (try store.require("l.weight")).toF32Alloc(gpa);
    defer gpa.free(l);
    try std.testing.expectEqualSlices(f32, &.{ 3, 4 }, l);
}

test "mix layer is the name minus its last component" {
    try std.testing.expectEqualStrings("blocks.3.attn.wq", Mix.layerOf("blocks.3.attn.wq.weight_s_rel"));
    try std.testing.expectEqualStrings("blocks.0.mod", Mix.layerOf("blocks.0.mod.lin"));
    try std.testing.expectEqualStrings("bare", Mix.layerOf("bare"));
}
