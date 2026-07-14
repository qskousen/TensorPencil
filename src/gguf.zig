//! Read-only GGUF (ggml universal file) loader — the llama.cpp checkpoint
//! container. Format: "GGUF" magic, u32 version (2 or 3 supported), u64
//! tensor count, u64 kv count, kv metadata (typed key/value pairs), a tensor
//! table (name, dims, ggml type, data offset), then the tensor data section
//! aligned to `general.alignment` (default 32). Files are memory-mapped like
//! safetensors; tensor bytes are lazy views into the mapping.
//!
//! Two conventions differ from safetensors and are normalized at parse time:
//! - Dims are stored fastest-first (ggml `ne` order); `TensorInfo.shape` is
//!   reversed into torch/safetensors row-major order, so a torch
//!   `[vocab, hidden]` matrix reads back as `[vocab, hidden]`. The byte
//!   layout is identical (row-major, last dim contiguous) — no repacking.
//! - llama.cpp tensor names ("blk.3.attn_q.weight", "token_embd.weight")
//!   are translated to the HF-style names the model loaders use
//!   ("layers.3.self_attn.q_proj.weight", "embed_tokens.weight");
//!   unrecognized names pass through unchanged. ComfyUI-converted GGUFs
//!   already use bare HF names and pass through.

const std = @import("std");
const dtypes = @import("dtype.zig");
const tensors = @import("tensor.zig");
const safetensors = @import("safetensors.zig");

const DType = dtypes.DType;
const TensorInfo = tensors.TensorInfo;
pub const TensorView = safetensors.TensorView;

pub const ParseError = error{
    FileTooSmall,
    InvalidMagic,
    UnsupportedVersion,
    InvalidHeader,
    UnsupportedTensorType,
    InvalidShape,
    InvalidOffsets,
    DuplicateTensor,
    OutOfMemory,
};

/// ggml tensor type ids (ggml.h `enum ggml_type`) we can load.
fn dtypeFromGgml(id: u32) ?DType {
    return switch (id) {
        0 => .f32,
        1 => .f16,
        8 => .q8_0,
        12 => .q4_k,
        13 => .q5_k,
        14 => .q6_k,
        16 => .i8, // GGML_TYPE_I8 (raw, no blocks)
        26 => .i32, // GGML_TYPE_I32
        30 => .bf16,
        else => null,
    };
}

/// A parsed metadata value. Strings and array spans point into the mapped
/// file (or the caller's slice) — valid until deinit.
pub const Value = union(enum) {
    uint: u64,
    int: i64,
    float: f64,
    boolean: bool,
    str: []const u8,
    arr: Array,
};

/// A typed metadata array, kept as its raw byte span (large tokenizer vocab
/// arrays parse lazily). `iterate` walks the elements.
pub const Array = struct {
    elem_type: u32,
    len: usize,
    /// Raw bytes of all elements (strings are length-prefixed inside).
    bytes: []const u8,

    pub fn iterate(self: Array) Iterator {
        return .{ .arr = self, .rest = self.bytes, .remaining = self.len };
    }

    pub const Iterator = struct {
        arr: Array,
        rest: []const u8,
        remaining: usize,

        /// Next element, or null when exhausted. Element parse errors were
        /// ruled out when the array span was validated at file parse time.
        pub fn next(self: *Iterator) ?Value {
            if (self.remaining == 0) return null;
            self.remaining -= 1;
            var r = Reader{ .data = self.rest, .pos = 0 };
            const v = readScalarValue(&r, self.arr.elem_type) catch unreachable;
            self.rest = self.rest[r.pos..];
            return v;
        }
    };
};

const Reader = struct {
    data: []const u8,
    pos: usize,

    fn take(self: *Reader, n: usize) ParseError![]const u8 {
        if (self.data.len - self.pos < n) return error.InvalidHeader;
        defer self.pos += n;
        return self.data[self.pos..][0..n];
    }

    fn int(self: *Reader, comptime T: type) ParseError!T {
        const raw = try self.take(@sizeOf(T));
        return std.mem.readInt(T, raw[0..@sizeOf(T)], .little);
    }

    fn str(self: *Reader) ParseError![]const u8 {
        const len = try self.int(u64);
        if (len > self.data.len - self.pos) return error.InvalidHeader;
        return try self.take(@intCast(len));
    }
};

/// Read one scalar (non-array) value of kv type `t`.
fn readScalarValue(r: *Reader, t: u32) ParseError!Value {
    return switch (t) {
        0 => .{ .uint = try r.int(u8) },
        1 => .{ .int = try r.int(i8) },
        2 => .{ .uint = try r.int(u16) },
        3 => .{ .int = try r.int(i16) },
        4 => .{ .uint = try r.int(u32) },
        5 => .{ .int = try r.int(i32) },
        6 => .{ .float = @as(f32, @bitCast(try r.int(u32))) },
        7 => .{ .boolean = (try r.int(u8)) != 0 },
        8 => .{ .str = try r.str() },
        10 => .{ .uint = try r.int(u64) },
        11 => .{ .int = try r.int(i64) },
        12 => .{ .float = @as(f64, @bitCast(try r.int(u64))) },
        else => error.InvalidHeader,
    };
}

pub const Gguf = struct {
    /// Whole-file mapping; null when constructed from a caller-owned slice
    /// or the buffered-read path (safetensors.use_mmap toggles both loaders).
    mapping: ?[]align(std.heap.page_size_min) const u8,
    /// Owned buffer in the buffered-read path; freed with `gpa`.
    owned: ?[]u8 = null,
    gpa: std.mem.Allocator = undefined,
    /// Tensor data section (file bytes from the aligned data offset).
    payload: []const u8,
    /// Canonical tensor name -> info, in file order.
    index: std.StringArrayHashMapUnmanaged(TensorInfo),
    /// Metadata key -> value, in file order. Keys/strings point into the
    /// file bytes; formatted canonical names live in the arena.
    kv: std.StringArrayHashMapUnmanaged(Value),
    alignment: usize,
    version: u32,
    arena: std.heap.ArenaAllocator,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !Gguf {
        return openIn(gpa, io, std.Io.Dir.cwd(), path);
    }

    pub fn openIn(gpa: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, path: []const u8) !Gguf {
        const file = try dir.openFile(io, path, .{ .mode = .read_only });
        defer file.close(io);
        const len = try file.length(io);
        if (len < 24) return error.FileTooSmall;

        if (!safetensors.use_mmap) {
            const buf = try gpa.alloc(u8, @intCast(len));
            errdefer gpa.free(buf);
            if (try file.readPositionalAll(io, buf, 0) != buf.len) return error.ShortRead;
            var g = try initFromSlice(gpa, buf);
            g.owned = buf;
            return g;
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
        std.posix.madvise(@constCast(mapping.ptr), mapping.len, std.posix.MADV.WILLNEED) catch {};
        var g = try initFromSlice(gpa, mapping);
        g.mapping = mapping;
        return g;
    }

    /// Parse from a caller-owned buffer (tests). Must outlive the Gguf.
    pub fn initFromSlice(gpa: std.mem.Allocator, data: []const u8) ParseError!Gguf {
        if (data.len < 24) return error.FileTooSmall;
        if (!std.mem.eql(u8, data[0..4], "GGUF")) return error.InvalidMagic;

        var r = Reader{ .data = data, .pos = 4 };
        const version = try r.int(u32);
        if (version != 2 and version != 3) return error.UnsupportedVersion;
        const n_tensors = try r.int(u64);
        const n_kv = try r.int(u64);
        // A tensor entry is at least 24 bytes, a kv at least 12: cheap sanity
        // bound before trusting the counts.
        if (n_tensors > data.len / 24 or n_kv > data.len / 12) return error.InvalidHeader;

        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();

        var kv: std.StringArrayHashMapUnmanaged(Value) = .empty;
        for (0..@intCast(n_kv)) |_| {
            const key = try r.str();
            const t = try r.int(u32);
            const value = try readValue(&r, t);
            const slot = try kv.getOrPut(alloc, key);
            slot.value_ptr.* = value; // duplicate keys: last wins, like llama.cpp
        }

        const arch: []const u8 = if (kv.get("general.architecture")) |v|
            (if (v == .str) v.str else "")
        else
            "";

        var alignment: usize = 32;
        if (kv.get("general.alignment")) |v| switch (v) {
            .uint => |a| if (a != 0 and std.math.isPowerOfTwo(a)) {
                alignment = @intCast(a);
            } else return error.InvalidHeader,
            else => return error.InvalidHeader,
        };

        // Tensor table: canonicalize names, reverse dims, validate spans
        // against the data section.
        const Raw = struct { name: []const u8, dt: DType, shape: tensors.Shape, offset: usize };
        const raw_infos = try alloc.alloc(Raw, @intCast(n_tensors));
        for (raw_infos) |*ri| {
            const raw_name = try r.str();
            const n_dims = try r.int(u32);
            if (n_dims > tensors.max_rank) return error.InvalidShape;
            var ne: [tensors.max_rank]u64 = @splat(1);
            for (0..n_dims) |d| ne[d] = try r.int(u64);
            const type_id = try r.int(u32);
            const offset = try r.int(u64);

            var dims: [tensors.max_rank]usize = @splat(0);
            for (0..n_dims) |d| {
                const v = ne[n_dims - 1 - d]; // reverse to row-major order
                if (v == 0 or v > std.math.maxInt(usize)) return error.InvalidShape;
                dims[d] = @intCast(v);
            }
            const dt = dtypeFromGgml(type_id) orelse return error.UnsupportedTensorType;
            // Blocks must tile the contiguous dim exactly (ggml guarantees it).
            if (ne[0] % dt.blockElems() != 0) return error.InvalidShape;
            if (offset % alignment != 0) return error.InvalidOffsets;
            ri.* = .{
                .name = try canonicalName(alloc, raw_name, arch),
                .dt = dt,
                .shape = .{ .dims = dims, .rank = n_dims },
                .offset = @intCast(offset),
            };
        }

        // Data section starts at the next alignment boundary after the table.
        const data_start = std.mem.alignForward(usize, r.pos, alignment);
        if (data_start > data.len) return error.InvalidOffsets;
        const payload = data[data_start..];

        var index: std.StringArrayHashMapUnmanaged(TensorInfo) = .empty;
        try index.ensureTotalCapacity(alloc, raw_infos.len);
        for (raw_infos) |ri| {
            const n_elems = ri.shape.count();
            const nbytes = ri.dt.storageBytes(n_elems);
            if (ri.offset > payload.len or payload.len - ri.offset < nbytes) return error.InvalidOffsets;
            const slot = index.getOrPutAssumeCapacity(ri.name);
            if (slot.found_existing) return error.DuplicateTensor;
            slot.value_ptr.* = .{
                .name = ri.name,
                .dtype = ri.dt,
                .shape = ri.shape,
                .start = ri.offset,
                .end = ri.offset + nbytes,
            };
        }

        return .{
            .mapping = null,
            .payload = payload,
            .index = index,
            .kv = kv,
            .alignment = alignment,
            .version = version,
            .arena = arena,
            .gpa = gpa,
        };
    }

    fn readValue(r: *Reader, t: u32) ParseError!Value {
        if (t != 9) return readScalarValue(r, t);
        const elem_type = try r.int(u32);
        if (elem_type == 9) return error.InvalidHeader; // no nested arrays
        const len64 = try r.int(u64);
        if (len64 > r.data.len - r.pos) return error.InvalidHeader;
        const len: usize = @intCast(len64);
        // Walk the elements once to find (and validate) the span.
        const start = r.pos;
        for (0..len) |_| _ = try readScalarValue(r, elem_type);
        return .{ .arr = .{
            .elem_type = elem_type,
            .len = len,
            .bytes = r.data[start..r.pos],
        } };
    }

    pub fn deinit(self: *Gguf) void {
        if (self.owned) |b| self.gpa.free(b);
        self.arena.deinit();
        if (self.mapping) |m| std.posix.munmap(m);
        self.* = undefined;
    }

    pub fn count(self: *const Gguf) usize {
        return self.index.count();
    }

    pub fn get(self: *const Gguf, name: []const u8) ?TensorView {
        const info = self.index.get(name) orelse return null;
        return .{ .info = info, .bytes = self.payload[info.start..info.end] };
    }

    /// Like `get`, but a missing tensor is an error — for required weights.
    pub fn require(self: *const Gguf, name: []const u8) !TensorView {
        return self.get(name) orelse error.MissingTensor;
    }

    /// Canonical tensor names in file order.
    pub fn names(self: *const Gguf) []const []const u8 {
        return self.index.keys();
    }

    // -- typed metadata accessors ------------------------------------------

    pub fn getUint(self: *const Gguf, key: []const u8) ?u64 {
        const v = self.kv.get(key) orelse return null;
        return switch (v) {
            .uint => |u| u,
            .int => |i| if (i >= 0) @intCast(i) else null,
            else => null,
        };
    }

    pub fn getFloat(self: *const Gguf, key: []const u8) ?f64 {
        const v = self.kv.get(key) orelse return null;
        return switch (v) {
            .float => |f| f,
            .uint => |u| @floatFromInt(u),
            .int => |i| @floatFromInt(i),
            else => null,
        };
    }

    pub fn getStr(self: *const Gguf, key: []const u8) ?[]const u8 {
        const v = self.kv.get(key) orelse return null;
        return if (v == .str) v.str else null;
    }

    pub fn getBool(self: *const Gguf, key: []const u8) ?bool {
        const v = self.kv.get(key) orelse return null;
        return if (v == .boolean) v.boolean else null;
    }

    pub fn getArr(self: *const Gguf, key: []const u8) ?Array {
        const v = self.kv.get(key) orelse return null;
        return if (v == .arr) v.arr else null;
    }
};

/// llama.cpp layer-tensor suffixes -> HF-style suffixes (Qwen3/llama family).
const layer_suffix_map = [_][2][]const u8{
    .{ "attn_norm.weight", "input_layernorm.weight" },
    .{ "post_attention_norm.weight", "post_attention_layernorm.weight" },
    .{ "attn_q.weight", "self_attn.q_proj.weight" },
    .{ "attn_k.weight", "self_attn.k_proj.weight" },
    .{ "attn_v.weight", "self_attn.v_proj.weight" },
    .{ "attn_output.weight", "self_attn.o_proj.weight" },
    .{ "attn_q_norm.weight", "self_attn.q_norm.weight" },
    .{ "attn_k_norm.weight", "self_attn.k_norm.weight" },
    .{ "ffn_norm.weight", "post_attention_layernorm.weight" },
    .{ "ffn_gate.weight", "mlp.gate_proj.weight" },
    .{ "ffn_up.weight", "mlp.up_proj.weight" },
    .{ "ffn_down.weight", "mlp.down_proj.weight" },
};

/// Gemma 3 layer-tensor suffixes -> HF-style suffixes. Gemma's "sandwich"
/// norms mean `ffn_norm` is the PRE-feedforward norm (not the post-attention
/// norm as in the Qwen/llama map above), and it carries two extra norms
/// (`post_attention_norm`, `post_ffw_norm`) that would otherwise collide.
const gemma3_layer_suffix_map = [_][2][]const u8{
    .{ "attn_norm.weight", "input_layernorm.weight" },
    .{ "attn_q.weight", "self_attn.q_proj.weight" },
    .{ "attn_k.weight", "self_attn.k_proj.weight" },
    .{ "attn_v.weight", "self_attn.v_proj.weight" },
    .{ "attn_output.weight", "self_attn.o_proj.weight" },
    .{ "attn_q_norm.weight", "self_attn.q_norm.weight" },
    .{ "attn_k_norm.weight", "self_attn.k_norm.weight" },
    .{ "post_attention_norm.weight", "post_attention_layernorm.weight" },
    .{ "ffn_norm.weight", "pre_feedforward_layernorm.weight" },
    .{ "post_ffw_norm.weight", "post_feedforward_layernorm.weight" },
    .{ "ffn_gate.weight", "mlp.gate_proj.weight" },
    .{ "ffn_up.weight", "mlp.up_proj.weight" },
    .{ "ffn_down.weight", "mlp.down_proj.weight" },
};

/// Translate a llama.cpp tensor name to the HF-style name the model loaders
/// use (prefix-less, e.g. "layers.3.self_attn.q_proj.weight"). The per-layer
/// suffix map is `arch`-dependent (gemma3 differs from the Qwen/llama
/// family — see gemma3_layer_suffix_map). Names that don't match the
/// convention (including ComfyUI-style GGUFs that already carry HF names)
/// pass through unchanged.
pub fn canonicalName(alloc: std.mem.Allocator, raw: []const u8, arch: []const u8) ![]const u8 {
    if (std.mem.eql(u8, raw, "token_embd.weight")) return "embed_tokens.weight";
    if (std.mem.eql(u8, raw, "output_norm.weight")) return "norm.weight";
    if (std.mem.eql(u8, raw, "output.weight")) return "lm_head.weight";
    if (std.mem.startsWith(u8, raw, "blk.")) {
        const rest = raw["blk.".len..];
        const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return raw;
        const layer = rest[0..dot];
        const suffix = rest[dot + 1 ..];
        _ = std.fmt.parseInt(u32, layer, 10) catch return raw;
        const map: []const [2][]const u8 = if (std.mem.eql(u8, arch, "gemma3"))
            &gemma3_layer_suffix_map
        else
            &layer_suffix_map;
        for (map) |entry| {
            if (std.mem.eql(u8, suffix, entry[0])) {
                return std.fmt.allocPrint(alloc, "layers.{s}.{s}", .{ layer, entry[1] });
            }
        }
        // Unmapped per-layer tensors (e.g. qwen35's ssm_* / attn_qkv) keep
        // their llama.cpp suffix under the layers.N. prefix.
        return std.fmt.allocPrint(alloc, "layers.{s}.{s}", .{ layer, suffix });
    }
    return raw;
}

// --- tests -----------------------------------------------------------------

/// Minimal in-memory GGUF builder for tests.
const TestBuilder = struct {
    buf: std.ArrayList(u8) = .empty,
    gpa: std.mem.Allocator,

    fn init(gpa: std.mem.Allocator, version: u32, n_tensors: u64, n_kv: u64) !TestBuilder {
        var b = TestBuilder{ .gpa = gpa };
        try b.buf.appendSlice(gpa, "GGUF");
        try b.int(u32, version);
        try b.int(u64, n_tensors);
        try b.int(u64, n_kv);
        return b;
    }

    fn int(self: *TestBuilder, comptime T: type, v: T) !void {
        var raw: [@sizeOf(T)]u8 = undefined;
        std.mem.writeInt(T, &raw, v, .little);
        try self.buf.appendSlice(self.gpa, &raw);
    }

    fn str(self: *TestBuilder, s: []const u8) !void {
        try self.int(u64, s.len);
        try self.buf.appendSlice(self.gpa, s);
    }

    fn kvUint(self: *TestBuilder, key: []const u8, v: u32) !void {
        try self.str(key);
        try self.int(u32, 4);
        try self.int(u32, v);
    }

    fn kvF32(self: *TestBuilder, key: []const u8, v: f32) !void {
        try self.str(key);
        try self.int(u32, 6);
        try self.int(u32, @bitCast(v));
    }

    fn kvStr(self: *TestBuilder, key: []const u8, v: []const u8) !void {
        try self.str(key);
        try self.int(u32, 8);
        try self.str(v);
    }

    fn tensor(self: *TestBuilder, name: []const u8, ne: []const u64, type_id: u32, offset: u64) !void {
        try self.str(name);
        try self.int(u32, @intCast(ne.len));
        for (ne) |d| try self.int(u64, d);
        try self.int(u32, type_id);
        try self.int(u64, offset);
    }

    /// Pad to the 32-byte data boundary and append the data section.
    fn finish(self: *TestBuilder, data: []const u8) ![]u8 {
        const aligned = std.mem.alignForward(usize, self.buf.items.len, 32);
        try self.buf.appendNTimes(self.gpa, 0, aligned - self.buf.items.len);
        try self.buf.appendSlice(self.gpa, data);
        return self.buf.toOwnedSlice(self.gpa);
    }

    fn deinit(self: *TestBuilder) void {
        self.buf.deinit(self.gpa);
    }
};

test "parse synthetic gguf" {
    const gpa = std.testing.allocator;

    var payload: [64 + 34]u8 = undefined;
    const a_vals = [6]f32{ 1, 2, 3, 4, 5, 6 };
    @memcpy(payload[0..24], std.mem.sliceAsBytes(&a_vals));
    @memset(payload[24..64], 0);
    @memcpy(payload[64..98], &@import("quants_fixtures.zig").q8_0_block);

    var b = try TestBuilder.init(gpa, 3, 2, 4);
    defer b.deinit();
    try b.kvStr("general.architecture", "qwen3");
    try b.kvUint("qwen3.block_count", 36);
    try b.kvF32("qwen3.rope.freq_base", 1e6);
    // An i32 array [7, -3].
    try b.str("test.arr");
    try b.int(u32, 9);
    try b.int(u32, 5);
    try b.int(u64, 2);
    try b.int(u32, 7);
    try b.int(u32, @bitCast(@as(i32, -3)));
    // f32 tensor, ggml ne [3, 2] = torch shape [2, 3].
    try b.tensor("blk.0.attn_q.weight", &.{ 3, 2 }, 0, 0);
    // q8_0 tensor at the next 32-aligned offset (24 -> 64).
    try b.tensor("token_embd.weight", &.{32}, 8, 64);
    const file = try b.finish(&payload);
    defer gpa.free(file);

    var g = try Gguf.initFromSlice(gpa, file);
    defer g.deinit();

    try std.testing.expectEqual(@as(u32, 3), g.version);
    try std.testing.expectEqual(@as(usize, 2), g.count());
    try std.testing.expectEqualStrings("qwen3", g.getStr("general.architecture").?);
    try std.testing.expectEqual(@as(u64, 36), g.getUint("qwen3.block_count").?);
    try std.testing.expectApproxEqAbs(@as(f64, 1e6), g.getFloat("qwen3.rope.freq_base").?, 0.1);

    var it = g.getArr("test.arr").?.iterate();
    try std.testing.expectEqual(@as(i64, 7), it.next().?.int);
    try std.testing.expectEqual(@as(i64, -3), it.next().?.int);
    try std.testing.expectEqual(@as(?Value, null), it.next());

    // Canonical name + reversed dims.
    const q = try g.require("layers.0.self_attn.q_proj.weight");
    try std.testing.expectEqual(DType.f32, q.info.dtype);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, q.info.shape.slice());
    const qf = try q.toF32Alloc(gpa);
    defer gpa.free(qf);
    try std.testing.expectEqualSlices(f32, &a_vals, qf);

    const emb = try g.require("embed_tokens.weight");
    try std.testing.expectEqual(DType.q8_0, emb.info.dtype);
    try std.testing.expectEqual(@as(usize, 34), emb.bytes.len);
    try std.testing.expect(g.get("blk.0.attn_q.weight") == null);
    try std.testing.expectError(error.MissingTensor, g.require("missing"));
}

test "reject malformed gguf" {
    const gpa = std.testing.allocator;

    try std.testing.expectError(error.FileTooSmall, Gguf.initFromSlice(gpa, "GGUF"));
    {
        var bad: [24]u8 = @splat(0);
        @memcpy(bad[0..4], "GGML");
        try std.testing.expectError(error.InvalidMagic, Gguf.initFromSlice(gpa, &bad));
    }
    { // v1 unsupported
        var b = try TestBuilder.init(gpa, 1, 0, 0);
        defer b.deinit();
        const file = try b.finish(&.{});
        defer gpa.free(file);
        try std.testing.expectError(error.UnsupportedVersion, Gguf.initFromSlice(gpa, file));
    }
    { // unknown ggml tensor type
        var b = try TestBuilder.init(gpa, 3, 1, 0);
        defer b.deinit();
        try b.tensor("t", &.{4}, 99, 0);
        const file = try b.finish(&[_]u8{0} ** 64);
        defer gpa.free(file);
        try std.testing.expectError(error.UnsupportedTensorType, Gguf.initFromSlice(gpa, file));
    }
    { // tensor data past the end of the file
        var b = try TestBuilder.init(gpa, 3, 1, 0);
        defer b.deinit();
        try b.tensor("t", &.{64}, 0, 0);
        const file = try b.finish(&[_]u8{0} ** 16);
        defer gpa.free(file);
        try std.testing.expectError(error.InvalidOffsets, Gguf.initFromSlice(gpa, file));
    }
    { // q8_0 row not a whole number of blocks
        var b = try TestBuilder.init(gpa, 3, 1, 0);
        defer b.deinit();
        try b.tensor("t", &.{ 16, 2 }, 8, 0);
        const file = try b.finish(&[_]u8{0} ** 68);
        defer gpa.free(file);
        try std.testing.expectError(error.InvalidShape, Gguf.initFromSlice(gpa, file));
    }
    { // duplicate tensor name
        var b = try TestBuilder.init(gpa, 3, 2, 0);
        defer b.deinit();
        try b.tensor("t", &.{4}, 0, 0);
        try b.tensor("t", &.{4}, 0, 32);
        const file = try b.finish(&[_]u8{0} ** 64);
        defer gpa.free(file);
        try std.testing.expectError(error.DuplicateTensor, Gguf.initFromSlice(gpa, file));
    }
    { // unaligned tensor offset
        var b = try TestBuilder.init(gpa, 3, 1, 0);
        defer b.deinit();
        try b.tensor("t", &.{4}, 0, 8);
        const file = try b.finish(&[_]u8{0} ** 64);
        defer gpa.free(file);
        try std.testing.expectError(error.InvalidOffsets, Gguf.initFromSlice(gpa, file));
    }
    { // truncated kv section
        var b = try TestBuilder.init(gpa, 3, 0, 5);
        defer b.deinit();
        try b.str("only.one");
        const file = try b.finish(&.{});
        defer gpa.free(file);
        try std.testing.expectError(error.InvalidHeader, Gguf.initFromSlice(gpa, file));
    }
}

test "canonical name translation" {
    const gpa = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const alloc = arena.allocator();

    try std.testing.expectEqualStrings("embed_tokens.weight", try canonicalName(alloc, "token_embd.weight", "qwen3"));
    try std.testing.expectEqualStrings("lm_head.weight", try canonicalName(alloc, "output.weight", "qwen3"));
    try std.testing.expectEqualStrings("norm.weight", try canonicalName(alloc, "output_norm.weight", "qwen3"));
    try std.testing.expectEqualStrings(
        "layers.17.mlp.down_proj.weight",
        try canonicalName(alloc, "blk.17.ffn_down.weight", "qwen3"),
    );
    try std.testing.expectEqualStrings(
        "layers.0.self_attn.k_norm.weight",
        try canonicalName(alloc, "blk.0.attn_k_norm.weight", "qwen3"),
    );
    // Unmapped blk suffixes keep their name under the layers.N. prefix;
    // non-blk names pass through.
    try std.testing.expectEqualStrings("layers.0.ssm_conv1d.weight", try canonicalName(alloc, "blk.0.ssm_conv1d.weight", "qwen35"));
    try std.testing.expectEqualStrings("layers.5.ssm_dt.bias", try canonicalName(alloc, "blk.5.ssm_dt.bias", "qwen35"));
    try std.testing.expectEqualStrings(
        "layers.2.post_attention_layernorm.weight",
        try canonicalName(alloc, "blk.2.post_attention_norm.weight", "qwen3"),
    );
    try std.testing.expectEqualStrings(
        "layers.3.self_attn.q_proj.weight",
        try canonicalName(alloc, "layers.3.self_attn.q_proj.weight", "qwen3"),
    );

    // Gemma 3: the two extra norms must NOT collide, and ffn_norm is the
    // PRE-feedforward norm (not post-attention as in the Qwen/llama map).
    try std.testing.expectEqualStrings(
        "layers.0.pre_feedforward_layernorm.weight",
        try canonicalName(alloc, "blk.0.ffn_norm.weight", "gemma3"),
    );
    try std.testing.expectEqualStrings(
        "layers.0.post_attention_layernorm.weight",
        try canonicalName(alloc, "blk.0.post_attention_norm.weight", "gemma3"),
    );
    try std.testing.expectEqualStrings(
        "layers.0.post_feedforward_layernorm.weight",
        try canonicalName(alloc, "blk.0.post_ffw_norm.weight", "gemma3"),
    );
    try std.testing.expectEqualStrings(
        "layers.7.self_attn.o_proj.weight",
        try canonicalName(alloc, "blk.7.attn_output.weight", "gemma3"),
    );
}

// Real-checkpoint smoke test; skipped when the model file is absent.
test "real qwen3-4b gguf headers" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const path = "models/text_encoders/Qwen3-4B-Q4_K_M.gguf";
    std.Io.Dir.cwd().access(io, path, .{}) catch return error.SkipZigTest;

    var g = try Gguf.open(gpa, io, path);
    defer g.deinit();

    try std.testing.expectEqual(@as(usize, 398), g.count());
    try std.testing.expectEqualStrings("qwen3", g.getStr("general.architecture").?);
    try std.testing.expectEqual(@as(u64, 36), g.getUint("qwen3.block_count").?);
    try std.testing.expectEqual(@as(u64, 2560), g.getUint("qwen3.embedding_length").?);

    const emb = try g.require("embed_tokens.weight");
    try std.testing.expectEqual(DType.q6_k, emb.info.dtype);
    try std.testing.expectEqualSlices(usize, &.{ 151936, 2560 }, emb.info.shape.slice());

    const q = try g.require("layers.35.self_attn.q_proj.weight");
    try std.testing.expectEqual(DType.q4_k, q.info.dtype);
    try std.testing.expectEqualSlices(usize, &.{ 4096, 2560 }, q.info.shape.slice());

    const norm = try g.require("layers.0.input_layernorm.weight");
    try std.testing.expectEqual(DType.f32, norm.info.dtype);
    try std.testing.expectEqualSlices(usize, &.{2560}, norm.info.shape.slice());
}
