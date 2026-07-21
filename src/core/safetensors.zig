//! Read-only safetensors loader.
//!
//! Format: 8-byte little-endian u64 header length, JSON header mapping tensor
//! names to {dtype, shape, data_offsets}, then the raw data section. Files are
//! memory-mapped (they range up to ~12 GiB), so tensor bytes are read lazily
//! by the OS page cache and nothing is copied at open time.

const std = @import("std");
const builtin = @import("builtin");
const dtypes = @import("dtype.zig");
const quants = @import("quants.zig");
const tensors = @import("tensor.zig");

const DType = dtypes.DType;
const TensorInfo = tensors.TensorInfo;

/// Reject absurd header lengths before trusting the u64 prefix.
const max_header_len: u64 = 256 * 1024 * 1024;

pub const ParseError = error{
    FileTooSmall,
    InvalidHeader,
    UnknownDType,
    InvalidShape,
    InvalidOffsets,
    DuplicateTensor,
    OutOfMemory,
};

pub const ConvertError = error{ UnsupportedDType, LengthMismatch };

/// A tensor's metadata plus its raw bytes within the data section.
pub const TensorView = struct {
    info: TensorInfo,
    bytes: []const u8,

    /// Convert/dequantize to a newly-allocated f32 buffer.
    pub fn toF32Alloc(self: TensorView, gpa: std.mem.Allocator) ![]f32 {
        const out = try gpa.alloc(f32, self.info.elemCount());
        errdefer gpa.free(out);
        try convertToF32(self.info.dtype, self.bytes, out);
        return out;
    }

    /// Read a rank-0 (or single-element) tensor as one f32.
    pub fn asScalarF32(self: TensorView) !f32 {
        if (self.info.elemCount() != 1) return error.LengthMismatch;
        var out: [1]f32 = undefined;
        try convertToF32(self.info.dtype, self.bytes, &out);
        return out[0];
    }
};

/// When true (default), `SafeTensors.open` mmaps the checkpoint; when false it
/// reads the whole file into an owned heap buffer via buffered I/O. mmap is the
/// better default (zero-copy, lazy paging, OS page-cache warmth across runs),
/// but its fault path deadlocks on ZFS under memory pressure — set this false
/// (CLI `--mmap off`) for checkpoints on ZFS, or just keep them on ext4/NVMe.
pub var use_mmap: bool = true;

pub const SafeTensors = struct {
    /// Whole-file mapping; null when constructed from a caller-owned slice or
    /// the buffered-read path.
    mapping: ?[]align(std.heap.page_size_min) const u8,
    /// Owned buffer backing `payload` in the buffered-read path (`use_mmap`
    /// false); null for mmap and caller-owned slices. Freed with `gpa`.
    owned: ?[]u8 = null,
    /// Allocator that owns `owned` (and backs `arena`). Only used for freeing.
    gpa: std.mem.Allocator = undefined,
    /// Data section (file bytes after the JSON header).
    payload: []const u8,
    /// Tensor name -> info, in header order. Names point into the header
    /// bytes (or the arena), valid until deinit.
    index: std.StringArrayHashMapUnmanaged(TensorInfo),
    /// Optional __metadata__ string map from the header.
    metadata: ?std.json.ObjectMap,
    arena: std.heap.ArenaAllocator,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !SafeTensors {
        return openIn(gpa, io, std.Io.Dir.cwd(), path);
    }

    pub fn openIn(gpa: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, path: []const u8) !SafeTensors {
        const file = try dir.openFile(io, path, .{ .mode = .read_only });
        defer file.close(io);
        const len = try file.length(io);
        if (len < 8) return error.FileTooSmall;

        if (!use_mmap) {
            // Buffered read into an owned heap buffer: no mmap fault path (ZFS-
            // safe), and the read is a single explicit up-front phase so weight
            // access during compute never touches disk. Fails with a clean OOM
            // under memory pressure instead of a D-state deadlock.
            const buf = try gpa.alloc(u8, @intCast(len));
            errdefer gpa.free(buf);
            if (try file.readPositionalAll(io, buf, 0) != buf.len) return error.ShortRead;
            var st = try initFromSlice(gpa, buf);
            st.owned = buf;
            return st;
        }

        const mapping = try std.posix.mmap(
            null,
            @intCast(len),
            .{ .READ = true },
            .{ .TYPE = .PRIVATE },
            file.handle,
            0,
        );
        errdefer std.posix.munmap(mapping);
        // Kick off an async read-ahead of the whole file. The weights are
        // each touched once (GPU upload / transpose), so a cold file
        // otherwise faults in synchronously on first access — a 12 GB DiT
        // adds tens of seconds to the first sampling step, and the 4.9 GB
        // text encoder inflates the encode. WILLNEED overlaps that disk read
        // with setup/compute instead. Advisory and best-effort.
        std.posix.madvise(@constCast(mapping.ptr), mapping.len, std.posix.MADV.WILLNEED) catch {};
        var st = try initFromSlice(gpa, mapping);
        st.mapping = mapping;
        return st;
    }

    /// Parse from a caller-owned buffer (used by tests). The buffer must
    /// outlive the returned SafeTensors.
    pub fn initFromSlice(gpa: std.mem.Allocator, data: []const u8) ParseError!SafeTensors {
        if (data.len < 8) return error.FileTooSmall;
        const header_len64 = std.mem.readInt(u64, data[0..8], .little);
        if (header_len64 > max_header_len or header_len64 > data.len - 8) return error.InvalidHeader;
        const header_len: usize = @intCast(header_len64);
        const header_bytes = data[8 .. 8 + header_len];
        const payload = data[8 + header_len ..];

        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();

        const root = std.json.parseFromSliceLeaky(std.json.Value, alloc, header_bytes, .{}) catch
            return error.InvalidHeader;
        if (root != .object) return error.InvalidHeader;

        var index: std.StringArrayHashMapUnmanaged(TensorInfo) = .empty;
        var metadata: ?std.json.ObjectMap = null;

        var it = root.object.iterator();
        while (it.next()) |entry| {
            const name = entry.key_ptr.*;
            if (std.mem.eql(u8, name, "__metadata__")) {
                if (entry.value_ptr.* == .object) metadata = entry.value_ptr.object;
                continue;
            }
            const info = try parseTensorEntry(name, entry.value_ptr.*, payload.len);
            const slot = try index.getOrPut(alloc, name);
            if (slot.found_existing) return error.DuplicateTensor;
            slot.value_ptr.* = info;
        }

        return .{
            .mapping = null,
            .payload = payload,
            .index = index,
            .metadata = metadata,
            .arena = arena,
            .gpa = gpa,
        };
    }

    fn parseTensorEntry(name: []const u8, value: std.json.Value, payload_len: usize) ParseError!TensorInfo {
        if (value != .object) return error.InvalidHeader;
        const obj = value.object;

        const dtype_val = obj.get("dtype") orelse return error.InvalidHeader;
        if (dtype_val != .string) return error.InvalidHeader;
        const dt = DType.fromString(dtype_val.string) orelse return error.UnknownDType;

        const shape_val = obj.get("shape") orelse return error.InvalidHeader;
        if (shape_val != .array) return error.InvalidHeader;
        const shape_items = shape_val.array.items;
        if (shape_items.len > tensors.max_rank) return error.InvalidShape;
        var dims: [tensors.max_rank]usize = @splat(0);
        for (shape_items, 0..) |item, i| {
            if (item != .integer or item.integer < 0) return error.InvalidShape;
            dims[i] = @intCast(item.integer);
        }
        const shape: tensors.Shape = .{ .dims = dims, .rank = shape_items.len };

        const off_val = obj.get("data_offsets") orelse return error.InvalidHeader;
        if (off_val != .array or off_val.array.items.len != 2) return error.InvalidOffsets;
        const start_v = off_val.array.items[0];
        const end_v = off_val.array.items[1];
        if (start_v != .integer or end_v != .integer) return error.InvalidOffsets;
        if (start_v.integer < 0 or end_v.integer < start_v.integer) return error.InvalidOffsets;
        const start: usize = @intCast(start_v.integer);
        const end: usize = @intCast(end_v.integer);
        if (end > payload_len) return error.InvalidOffsets;
        if (end - start != dt.storageBytes(shape.count())) return error.InvalidOffsets;

        return .{ .name = name, .dtype = dt, .shape = shape, .start = start, .end = end };
    }

    pub fn deinit(self: *SafeTensors) void {
        if (self.owned) |b| self.gpa.free(b);
        self.arena.deinit();
        if (self.mapping) |m| std.posix.munmap(m);
        self.* = undefined;
    }

    pub fn count(self: *const SafeTensors) usize {
        return self.index.count();
    }

    pub fn get(self: *const SafeTensors, name: []const u8) ?TensorView {
        const info = self.index.get(name) orelse return null;
        return .{ .info = info, .bytes = self.payload[info.start..info.end] };
    }

    /// Like `get`, but a missing tensor is an error — for required weights.
    pub fn require(self: *const SafeTensors, name: []const u8) !TensorView {
        return self.get(name) orelse error.MissingTensor;
    }

    /// Tensor names in header order.
    pub fn names(self: *const SafeTensors) []const []const u8 {
        return self.index.keys();
    }
};

/// Convert raw little-endian tensor bytes to f32. `bytes` may be unaligned
/// (safetensors gives no alignment guarantees).
pub fn convertToF32(dt: DType, bytes: []const u8, out: []f32) ConvertError!void {
    if (bytes.len != dt.storageBytes(out.len)) return error.LengthMismatch;
    switch (dt) {
        .f32 => {
            if (comptime builtin.cpu.arch.endian() == .little) {
                @memcpy(std.mem.sliceAsBytes(out), bytes);
            } else {
                for (out, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little));
            }
        },
        .f8_e4m3 => for (out, 0..) |*v, i| {
            v.* = dtypes.f8e4m3ToF32(bytes[i]);
        },
        .bf16 => dtypes.bf16ToF32Row(bytes, out, 1.0),
        .f16 => dtypes.f16ToF32Row(bytes, out, 1.0),
        // ggml block-quantized GGUF tensors (rows are whole blocks, which the
        // GGUF parser validated, so the length check above already enforced
        // block alignment).
        .q4_0, .q8_0, .q4_k, .q5_k, .q6_k => quants.dequantSlice(dt, bytes, 0, out.len, out),
        else => return error.UnsupportedDType,
    }
}

// --- tests ---------------------------------------------------------------

/// Build a minimal in-memory safetensors file for tests.
fn buildTestFile(gpa: std.mem.Allocator, header_json: []const u8, payload: []const u8) ![]u8 {
    const buf = try gpa.alloc(u8, 8 + header_json.len + payload.len);
    std.mem.writeInt(u64, buf[0..8], header_json.len, .little);
    @memcpy(buf[8..][0..header_json.len], header_json);
    @memcpy(buf[8 + header_json.len ..], payload);
    return buf;
}

test "parse synthetic file" {
    const gpa = std.testing.allocator;
    const header =
        \\{"__metadata__":{"format":"pt"},
        \\ "a.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},
        \\ "b":{"dtype":"F8_E4M3","shape":[4],"data_offsets":[16,20]},
        \\ "c.scale":{"dtype":"F32","shape":[],"data_offsets":[20,24]}}
    ;
    var payload: [24]u8 = undefined;
    const a_vals = [4]f32{ 1.0, -2.5, 0.0, 3.25 };
    @memcpy(payload[0..16], std.mem.sliceAsBytes(&a_vals));
    payload[16..20].* = .{ 0x38, 0x40, 0xc8, 0x00 }; // fp8: 1.0, 2.0, -4.0, 0.0
    const c_val = [1]f32{0.5};
    @memcpy(payload[20..24], std.mem.sliceAsBytes(&c_val));

    const file_bytes = try buildTestFile(gpa, header, &payload);
    defer gpa.free(file_bytes);

    var st = try SafeTensors.initFromSlice(gpa, file_bytes);
    defer st.deinit();

    try std.testing.expectEqual(@as(usize, 3), st.count());
    try std.testing.expectEqualStrings("pt", st.metadata.?.get("format").?.string);

    const a = st.get("a.weight").?;
    try std.testing.expectEqual(DType.f32, a.info.dtype);
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, a.info.shape.slice());
    const a_f32 = try a.toF32Alloc(gpa);
    defer gpa.free(a_f32);
    try std.testing.expectEqualSlices(f32, &a_vals, a_f32);

    const b = st.get("b").?;
    const b_f32 = try b.toF32Alloc(gpa);
    defer gpa.free(b_f32);
    try std.testing.expectEqualSlices(f32, &.{ 1.0, 2.0, -4.0, 0.0 }, b_f32);

    try std.testing.expectEqual(@as(f32, 0.5), try st.get("c.scale").?.asScalarF32());
    try std.testing.expectEqual(@as(?TensorView, null), st.get("missing"));
    try std.testing.expectError(error.MissingTensor, st.require("missing"));
}

test "reject malformed files" {
    const gpa = std.testing.allocator;

    // Too small for the length prefix.
    try std.testing.expectError(error.FileTooSmall, SafeTensors.initFromSlice(gpa, &.{ 1, 2, 3 }));

    // Header length exceeding the file.
    var tiny: [16]u8 = @splat(0);
    std.mem.writeInt(u64, tiny[0..8], 1000, .little);
    try std.testing.expectError(error.InvalidHeader, SafeTensors.initFromSlice(gpa, &tiny));

    // Offsets that disagree with shape * dtype size.
    {
        const header =
            \\{"t":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}
        ;
        const bytes = try buildTestFile(gpa, header, &[_]u8{0} ** 4);
        defer gpa.free(bytes);
        try std.testing.expectError(error.InvalidOffsets, SafeTensors.initFromSlice(gpa, bytes));
    }

    // Offsets past the end of the payload.
    {
        const header =
            \\{"t":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}
        ;
        const bytes = try buildTestFile(gpa, header, &[_]u8{0} ** 4);
        defer gpa.free(bytes);
        try std.testing.expectError(error.InvalidOffsets, SafeTensors.initFromSlice(gpa, bytes));
    }

    // Unknown dtype string.
    {
        const header =
            \\{"t":{"dtype":"Q4_K","shape":[2],"data_offsets":[0,8]}}
        ;
        const bytes = try buildTestFile(gpa, header, &[_]u8{0} ** 8);
        defer gpa.free(bytes);
        try std.testing.expectError(error.UnknownDType, SafeTensors.initFromSlice(gpa, bytes));
    }

    // Negative shape entry.
    {
        const header =
            \\{"t":{"dtype":"F32","shape":[-2],"data_offsets":[0,8]}}
        ;
        const bytes = try buildTestFile(gpa, header, &[_]u8{0} ** 8);
        defer gpa.free(bytes);
        try std.testing.expectError(error.InvalidShape, SafeTensors.initFromSlice(gpa, bytes));
    }

    // Not JSON at all.
    {
        const bytes = try buildTestFile(gpa, "hello", &.{});
        defer gpa.free(bytes);
        try std.testing.expectError(error.InvalidHeader, SafeTensors.initFromSlice(gpa, bytes));
    }
}

test "open via mmap round trip" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const header =
        \\{"w":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}
    ;
    var payload: [4]u8 = undefined;
    std.mem.writeInt(u16, payload[0..2], 0x3f80, .little); // 1.0
    std.mem.writeInt(u16, payload[2..4], 0xc000, .little); // -2.0
    const bytes = try buildTestFile(gpa, header, &payload);
    defer gpa.free(bytes);

    try tmp.dir.writeFile(io, .{ .sub_path = "t.safetensors", .data = bytes });

    var st = try SafeTensors.openIn(gpa, io, tmp.dir, "t.safetensors");
    defer st.deinit();
    const w = try st.require("w");
    const vals = try w.toF32Alloc(gpa);
    defer gpa.free(vals);
    try std.testing.expectEqualSlices(f32, &.{ 1.0, -2.0 }, vals);
}

test "open via buffered read round trip (use_mmap=false)" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const header =
        \\{"w":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}
    ;
    var payload: [4]u8 = undefined;
    std.mem.writeInt(u16, payload[0..2], 0x3f80, .little); // 1.0
    std.mem.writeInt(u16, payload[2..4], 0xc000, .little); // -2.0
    const bytes = try buildTestFile(gpa, header, &payload);
    defer gpa.free(bytes);
    try tmp.dir.writeFile(io, .{ .sub_path = "t.safetensors", .data = bytes });

    const prev = use_mmap;
    use_mmap = false;
    defer use_mmap = prev;

    var st = try SafeTensors.openIn(gpa, io, tmp.dir, "t.safetensors");
    defer st.deinit(); // must free the owned buffer, not munmap
    try std.testing.expect(st.mapping == null and st.owned != null);
    const w = try st.require("w");
    const vals = try w.toF32Alloc(gpa);
    defer gpa.free(vals);
    try std.testing.expectEqualSlices(f32, &.{ 1.0, -2.0 }, vals);
}

// Validation against the real Krea 2 checkpoints; skipped when models/ is not
// present (e.g. CI or a fresh checkout).
test "real model headers" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;

    const vae_path = "models/vae/krea2RealVae_v10.safetensors";
    std.Io.Dir.cwd().access(io, vae_path, .{}) catch return error.SkipZigTest;

    {
        var st = try SafeTensors.open(gpa, io, vae_path);
        defer st.deinit();
        try std.testing.expectEqual(@as(usize, 194), st.count());
        const conv1 = try st.require("decoder.conv1.weight");
        try std.testing.expectEqual(DType.f32, conv1.info.dtype);
        try std.testing.expectEqualSlices(usize, &.{ 384, 16, 3, 3, 3 }, conv1.info.shape.slice());
    }

    {
        var st = try SafeTensors.open(gpa, io, "models/diffusion_model/krea2CenterSemiraw_v10Fp8.safetensors");
        defer st.deinit();
        try std.testing.expectEqual(@as(usize, 431), st.count());
        const wq = try st.require("blocks.0.attn.wq.weight");
        try std.testing.expectEqual(DType.f8_e4m3, wq.info.dtype);
        try std.testing.expectEqualSlices(usize, &.{ 6144, 6144 }, wq.info.shape.slice());
        const wk = try st.require("blocks.27.attn.wk.weight");
        try std.testing.expectEqualSlices(usize, &.{ 1536, 6144 }, wk.info.shape.slice());
        try std.testing.expect(st.get("blocks.28.attn.wq.weight") == null);
        const sigmas = try st.require("model_sampling.sigmas");
        try std.testing.expectEqualSlices(usize, &.{10000}, sigmas.info.shape.slice());
    }

    {
        var st = try SafeTensors.open(gpa, io, "models/text_encoders/qwen3VLInstruct4bHeretic_v10.safetensors");
        defer st.deinit();
        try std.testing.expectEqual(@as(usize, 1217), st.count());
        const emb = try st.require("model.language_model.embed_tokens.weight");
        try std.testing.expectEqual(DType.bf16, emb.info.dtype);
        try std.testing.expectEqualSlices(usize, &.{ 151936, 2560 }, emb.info.shape.slice());
        const q = try st.require("model.language_model.layers.35.self_attn.q_proj.weight");
        try std.testing.expectEqual(DType.f8_e4m3, q.info.dtype);
        try std.testing.expectEqualSlices(usize, &.{ 4096, 2560 }, q.info.shape.slice());
        const scale = try st.require("model.language_model.layers.35.self_attn.q_proj.weight_scale");
        try std.testing.expectEqual(DType.f32, scale.info.dtype);
        _ = try scale.asScalarF32();
    }
}
