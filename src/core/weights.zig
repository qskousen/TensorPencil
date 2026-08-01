//! WeightStore — uniform read access over checkpoint containers
//! (safetensors, GGUF), so model loaders don't care which format a
//! checkpoint ships in. Both containers resolve HF-style tensor names
//! (GGUF llama.cpp names are canonicalized at parse time; see gguf.zig).
//!
//! The third arm is an `Overlay`: a base store with a few tensors substituted
//! from caller-owned memory. It exists so an experiment can change one weight
//! without rewriting the checkpoint — see the doc comment on `Overlay`.

const std = @import("std");
const safetensors = @import("safetensors.zig");
const gguf = @import("gguf.zig");
const dtypes = @import("dtype.zig");

pub const TensorView = safetensors.TensorView;

pub const WeightStore = union(enum) {
    safetensors: *const safetensors.SafeTensors,
    gguf: *const gguf.Gguf,
    overlay: *const Overlay,

    pub fn get(self: WeightStore, name: []const u8) ?TensorView {
        return switch (self) {
            .safetensors => |st| st.get(name),
            .gguf => |g| g.get(name),
            .overlay => |o| o.get(name),
        };
    }

    /// Like `get`, but a missing tensor is an error — for required weights.
    pub fn require(self: WeightStore, name: []const u8) !TensorView {
        return self.get(name) orelse error.MissingTensor;
    }

    pub fn count(self: WeightStore) usize {
        return switch (self) {
            .safetensors => |st| st.count(),
            .gguf => |g| g.count(),
            .overlay => |o| o.base.count(),
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
        };
    }

    /// The whole-file mmap backing the tensor bytes (for page-locked direct
    /// GPU streaming); null on the buffered-read path.
    pub fn mapping(self: WeightStore) ?[]align(std.heap.page_size_min) const u8 {
        return switch (self) {
            .safetensors => |st| st.mapping,
            .gguf => |g| g.mapping,
            // Deliberately null rather than the base's mapping. A patched tensor's
            // bytes are caller-owned and lie OUTSIDE the base mapping, so a
            // consumer that turns a view into a mapping-relative offset (which is
            // what this accessor is for) would compute a garbage offset for exactly
            // the tensors an experiment changed — and read the unpatched bytes.
            // Losing the streaming fast path is a slowdown; that would be a silent
            // wrong answer.
            .overlay => null,
        };
    }
};

/// A base store with a few tensors substituted from caller-owned memory.
///
/// The measurement this exists for: quantize exactly ONE tensor of a checkpoint
/// and run the whole model, to attribute the model's error to that layer. Doing
/// it by writing a modified checkpoint costs a full copy of the file per data
/// point (~26 GB for krea2), which is what made per-layer attribution
/// unaffordable; here it costs one tensor's worth of memory.
///
/// It is a plain read-through map, so it serves anything else shaped the same
/// way — merged LoRA/delta weights, per-layer dtype experiments, a GUI hot-swap.
///
/// Two lifetime rules, both because model loaders keep *views* into the store
/// rather than copies:
///
///   1. the patch bytes must outlive the model loaded from the overlay, and
///   2. the `Overlay` itself must stay at a stable address for as long as a
///      `WeightStore` refers to it (it is held by pointer).
///
/// A patch may change a tensor's **dtype** — substituting f32 for the base's
/// bf16 is the normal case, since the f32 holds a dequantized round-trip
/// exactly — but not its **shape**, which is checked. Substituting a
/// differently-shaped tensor is a bug every time (a transposed buffer, the wrong
/// layer), and the shape check catches it here rather than as a puzzling
/// `ShapeMismatch` from a loader 200 weights later.
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
    /// `bytes` must be exactly that many bytes — the one thing a caller can get
    /// wrong here (an f32 buffer handed in as bf16 reads as noise, and every
    /// downstream number would be about a model nobody meant to run).
    ///
    /// Errors: `MissingTensor` if the base has no such tensor (the overlay may
    /// not extend the namespace — see `WeightStore.names`), `LengthMismatch` if
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

    /// `put`, taking a whole view — for a replacement that already carries its own
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

    // Restoring is exact — this is what lets one overlay serve a whole sweep.
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

    // Right dtype, wrong length — the mistake that would otherwise be read as
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
