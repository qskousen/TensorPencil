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
    /// GGUF type id 42 (Q2_0) is claimed by two formats with different block
    /// sizes and the file's geometry does not identify which. See
    /// `detectQ2_0Variant` — guessing here is a silent wrong answer.
    AmbiguousQ2_0Variant,
    DuplicateTensor,
    OutOfMemory,
};

/// ggml tensor type ids (ggml.h `enum ggml_type`) we can load.
///
/// `q2_0` supplies the resolution of the ambiguous id 42 — see
/// `detectQ2_0Variant`. Passing the wrong one is a silent wrong answer, so there
/// is no default; the only caller resolves it from file geometry first.
fn dtypeFromGgml(id: u32, q2_0: DType) ?DType {
    return switch (id) {
        0 => .f32,
        1 => .f16,
        2 => .q4_0,
        8 => .q8_0,
        12 => .q4_k,
        13 => .q5_k,
        14 => .q6_k,
        20 => .iq4_nl, // GGML_TYPE_IQ4_NL (32-elem block, non-linear 4-bit LUT)
        41 => .q1_0, // GGML_TYPE_Q1_0 (128-elem block, 1 sign bit per weight)
        42 => q2_0, // GGML_TYPE_Q2_0 — AMBIGUOUS, resolved by the caller
        24 => .i8, // GGML_TYPE_I8 (raw, no blocks) — NOT 16, which is IQ2_XXS
        26 => .i32, // GGML_TYPE_I32
        30 => .bf16,
        else => null,
    };
}

/// One tensor-table row, reduced to what `detectQ2_0Variant` needs.
const Span = struct { type_id: u32, elems: u64, offset: u64 };

/// Resolve GGUF type id 42 — which two shipped formats claim — from the file's
/// own geometry. Returns `.q2_0_g64` (upstream ggml, 64 elems / 18 B) or
/// `.q2_0_g128` (the PrismML fork's `prism` branch, 128 elems / 34 B).
///
/// ⚠️ **This cannot be skipped or guessed.** The two are indistinguishable from
/// the header: same type id, same `general.file_type`, same arithmetic. Only the
/// on-disk row length differs, by 17/18. And the error is not symmetric — reading
/// a g64 file as g128 computes *smaller* spans than reality, so the bounds check
/// below passes and every tensor view is silently short and misaligned against
/// the real block stream. That is a wrong answer with no diagnostic, which is why
/// this runs before any dtype is assigned.
///
/// The discriminator is the gap to the next tensor in offset order: a candidate
/// is accepted only if `alignForward(size) == gap` exactly. The two candidates
/// differ by ~5.6% (80 B on a 1440 B row) against at most `alignment` (typically
/// 32 B) of padding, so for any real weight matrix there is no ambiguity band —
/// but a gap that fits *both* (tiny tensors) or *neither* (a non-contiguous
/// writer) simply casts no vote rather than guessing. Disagreement between
/// tensors, or no vote at all, is an error: this is exactly the situation where
/// picking a default is worse than refusing.
fn detectQ2_0Variant(alloc: std.mem.Allocator, spans: []const Span, alignment: usize, payload_len: u64) ParseError!DType {
    const order = try alloc.alloc(u32, spans.len);
    defer alloc.free(order);
    for (order, 0..) |*o, i| o.* = @intCast(i);
    std.mem.sort(u32, order, spans, struct {
        fn lt(s: []const Span, a: u32, b: u32) bool {
            return s[a].offset < s[b].offset;
        }
    }.lt);

    var found: ?DType = null;
    for (order, 0..) |idx, i| {
        const s = spans[idx];
        if (s.type_id != 42) continue;
        // The span available to this tensor: up to the next tensor's offset, or
        // the end of the data section for the last one.
        const limit = if (i + 1 < order.len) spans[order[i + 1]].offset else payload_len;
        if (limit <= s.offset) continue; // overlapping/duplicate offsets: no signal
        const gap = limit - s.offset;

        var vote: ?DType = null;
        for ([_]DType{ .q2_0_g64, .q2_0_g128 }) |cand| {
            const be: u64 = cand.blockElems();
            if (s.elems % be != 0) continue;
            const size = s.elems / be * cand.blockBytes();
            if (std.mem.alignForward(u64, size, alignment) != gap) continue;
            if (vote != null) {
                vote = null; // both fit this gap — too small to discriminate
                break;
            }
            vote = cand;
        }
        const v = vote orelse continue;
        if (found) |f| {
            if (f != v) return error.AmbiguousQ2_0Variant; // the file disagrees with itself
        } else found = v;
    }
    return found orelse error.AmbiguousQ2_0Variant;
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
    /// or the buffered-read path (safetensors.read_mode toggles both loaders).
    mapping: ?[]align(std.heap.page_size_min) const u8,
    /// Owned buffer in the buffered-read path; freed with `gpa`.
    owned: ?[]u8 = null,
    gpa: std.mem.Allocator = undefined,
    /// The mapped file, kept OPEN so `readTo` can fetch tensor bytes with
    /// positional reads instead of faulting the mapping. Null for the
    /// buffered-read and caller-slice paths. See `SafeTensors.readTo`.
    file: ?std.Io.File = null,
    /// The `Io` `file` was opened with; needed to read and to close it.
    io: ?std.Io = null,
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
        // Closed on every path EXCEPT a successful mmap, which keeps it for
        // `readTo` (see SafeTensors.openIn — same one-flag arrangement).
        var keep_file = false;
        defer if (!keep_file) file.close(io);
        const len = try file.length(io);
        if (len < 24) return error.FileTooSmall;

        if (!safetensors.useMmap()) {
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
        g.file = file;
        g.io = io;
        keep_file = true;
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

        // Pre-scan the tensor table for GGUF type id 42, whose block size is not
        // determined by the header (see `detectQ2_0Variant`). This walks the table
        // once to collect offsets and element counts, then rewinds; the mapping is
        // already in memory, so the second pass is only string/int decoding. A
        // resolution has to exist before any dtype is assigned, because the size of
        // every 42 tensor depends on it.
        const table_start = r.pos;
        const q2_0_variant: DType = blk: {
            const spans = try alloc.alloc(Span, @intCast(n_tensors));
            defer alloc.free(spans);
            var any_q2_0 = false;
            for (spans) |*sp| {
                _ = try r.str();
                const nd = try r.int(u32);
                if (nd > tensors.max_rank) return error.InvalidShape;
                var elems: u64 = 1;
                for (0..nd) |_| {
                    const v = try r.int(u64);
                    if (v == 0) return error.InvalidShape;
                    elems = std.math.mul(u64, elems, v) catch return error.InvalidShape;
                }
                const tid = try r.int(u32);
                sp.* = .{ .type_id = tid, .elems = elems, .offset = try r.int(u64) };
                if (tid == 42) any_q2_0 = true;
            }
            const ds = std.mem.alignForward(usize, r.pos, alignment);
            if (ds > r.data.len) return error.InvalidOffsets;
            r.pos = table_start; // rewind for the real pass below
            if (!any_q2_0) break :blk .q2_0_g128; // unused; no tensor references it
            break :blk try detectQ2_0Variant(alloc, spans, alignment, r.data.len - ds);
        };

        // Tensor table: canonicalize names, reverse dims, validate spans
        // against the data section.
        const Raw = struct { name: []const u8, dt: DType, shape: tensors.Shape, offset: usize, flat_blocks: bool };
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
            const dt = dtypeFromGgml(type_id, q2_0_variant) orelse return error.UnsupportedTensorType;
            // Blocks must tile the contiguous dim exactly (ggml guarantees it).
            if (ne[0] % dt.blockElems() != 0) return error.InvalidShape;
            if (offset % alignment != 0) return error.InvalidOffsets;

            // ComfyUI-GGUF's shape fix, undone. Its converters (and ggufy's) reshape
            // a tensor whose contiguous dim is not a multiple of 256 to
            // `(n/256, 256)` so ggml's block quants can tile it, recording the true
            // shape in `comfy.gguf.orig_shape.<name>`. Without restoring it here a
            // consumer sees the storage shape — krea2's patch embed arrives as
            // [1536, 256] instead of [6144, 64] — and every shape check downstream
            // fails on a file that is perfectly well formed.
            var shape: tensors.Shape = .{ .dims = dims, .rank = n_dims };
            var flat_blocks = false;
            if (origShape(&kv, alloc, raw_name)) |orig| {
                // ⚠️ **This used to refuse block-quantized tensors**, on the reasoning
                // that "ComfyUI only ever applies the fix to tensors it then leaves
                // unquantized, since the criterion that triggers it — a contiguous dim
                // not divisible by 256 — is the same one that makes k-quantization
                // impossible". That is true of ComfyUI's converter and **false of
                // ggufy's**, which blocks over a tensor's *flat* element count rather
                // than per row: it reshapes first and then happily k-quantizes the
                // result. Measured on a ggufy SD1.5 q4_k file: **166 of 686 tensors,
                // 72.7% of the parameters** — every convolution, whose contiguous dim
                // is `kw` (1 or 3). So the refusal made TensorPencil unable to load any
                // ggufy SD-family GGUF at all (`time_embed.0.weight has shape
                // {1600, 256}, expected [1280, 320]`), i.e. the whole GGUF output path
                // for the SD family was unmeasurable — the same gap that was closed for
                // krea2 and missed here.
                //
                // Restoring the shape is correct for the values: the fix is a pure
                // regrouping, so flat row-major order is preserved, and 256-wide
                // storage rows hold exactly one block each — flat and per-row blocking
                // coincide, with or without an imatrix. What it is *not* is compatible
                // with row-aligned blocking, so `flat_blocks` tells the loaders to
                // materialize instead of pointing a packed `Weight` at the bytes.
                //
                // ⚠️ **Named shortcut:** materializing costs the memory the
                // quantization saved (SD1.5 q4_k: ~0.5 GB on disk, ~3.4 GB f32
                // resident). The robust alternative is a block GEMM that understands
                // flat blocking, on all four backends — a real kernel project, and not
                // one to undertake incidentally. This path is functionally complete
                // (correct values, every consumer works) and purely additive: these
                // files previously hard-errored, so nothing that loads today changes.
                if (orig.count() != shape.count()) {
                    std.log.warn("gguf: ignoring orig_shape for '{s}': {d} elements, stored has {d}", .{
                        raw_name, orig.count(), shape.count(),
                    });
                } else {
                    flat_blocks = dt.blockElems() != 1;
                    shape = orig;
                }
            }

            ri.* = .{
                .name = try canonicalName(alloc, raw_name, arch),
                .dt = dt,
                .shape = shape,
                .offset = @intCast(offset),
                .flat_blocks = flat_blocks,
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
                .flat_blocks = ri.flat_blocks,
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
        if (self.file) |f| f.close(self.io.?);
        self.* = undefined;
    }

    /// Fill `dst` with the bytes `src` (a slice of this file's mapping) points at,
    /// via a positional read. The GGUF twin of `SafeTensors.readTo` — see there for
    /// why reading beats faulting on a cold multi-GB checkpoint. False when this
    /// store has no open file or `src` is not inside the mapping; the caller then
    /// uses `src` directly, which is the pre-existing behaviour.
    pub fn readTo(self: *const Gguf, dst: []u8, src: []const u8) bool {
        if (safetensors.read_mode != .pread) return false;
        const f = self.file orelse return false;
        const io = self.io orelse return false;
        const m = self.mapping orelse return false;
        if (dst.len != src.len) return false;
        const base = @intFromPtr(m.ptr);
        const at = @intFromPtr(src.ptr);
        if (at < base or at + src.len > base + m.len) return false;
        const n = f.readPositionalAll(io, dst, at - base) catch return false;
        return n == dst.len;
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

    /// True if this GGUF is a CLIP/vision projector — an `mmproj-*.gguf` vision
    /// tower (`general.architecture == "clip"`) rather than an LLM. Lets a
    /// vision-tower slot reject a wrong pick (e.g. a second LLM file) with a
    /// clear error instead of dying deep in the tower loader with a cryptic
    /// config error.
    pub fn isVisionProjector(self: *const Gguf) bool {
        const arch = self.getStr("general.architecture") orelse return false;
        return std.mem.eql(u8, arch, "clip");
    }

    /// The model's trained context length (`<arch>.context_length`), when the
    /// container records both the architecture and the key. `null` lets the
    /// caller pick a family default.
    pub fn contextLength(self: *const Gguf) ?u64 {
        const arch = self.getStr("general.architecture") orelse return null;
        var buf: [64]u8 = undefined;
        const key = std.fmt.bufPrint(&buf, "{s}.context_length", .{arch}) catch return null;
        return self.getUint(key);
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

/// Gemma 4 layer-tensor suffixes -> HF-style suffixes. Same "sandwich"-norm
/// layout as Gemma 3, plus `layer_output_scale` (a per-layer scalar the whole
/// layer output is multiplied by). Q/K/V and their per-head norms have
/// layer-dependent dimensions (sliding-window vs global), but the names match.
const gemma4_layer_suffix_map = [_][2][]const u8{
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
    .{ "layer_output_scale.weight", "out_scale.weight" },
};

/// Translate a llama.cpp tensor name to the HF-style name the model loaders
/// use (prefix-less, e.g. "layers.3.self_attn.q_proj.weight"). The per-layer
/// suffix map is `arch`-dependent (gemma3 differs from the Qwen/llama
/// family — see gemma3_layer_suffix_map). Names that don't match the
/// convention (including ComfyUI-style GGUFs that already carry HF names)
/// pass through unchanged.
/// The logical shape ComfyUI-GGUF records for a tensor it reshaped to fit ggml's
/// block quants: `comfy.gguf.orig_shape.<raw name>`, an array of dimensions in
/// row-major (torch) order — already the order this reader uses, so no reversal.
/// Null when absent or malformed.
fn origShape(kv: *const std.StringArrayHashMapUnmanaged(Value), alloc: std.mem.Allocator, raw_name: []const u8) ?tensors.Shape {
    const key = std.fmt.allocPrint(alloc, "comfy.gguf.orig_shape.{s}", .{raw_name}) catch return null;
    const v = kv.get(key) orelse return null;
    const arr = switch (v) {
        .arr => |a| a,
        else => return null,
    };
    if (arr.len == 0 or arr.len > tensors.max_rank) return null;
    var dims: [tensors.max_rank]usize = @splat(0);
    var it = arr.iterate();
    var i: usize = 0;
    while (it.next()) |item| : (i += 1) {
        const n: i128 = switch (item) {
            .uint => |u| @intCast(u),
            .int => |x| x,
            else => return null,
        };
        if (n <= 0 or n > std.math.maxInt(usize)) return null;
        dims[i] = @intCast(n);
    }
    return .{ .dims = dims, .rank = arr.len };
}

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
        else if (std.mem.eql(u8, arch, "gemma4"))
            &gemma4_layer_suffix_map
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

/// Minimal in-memory GGUF builder for tests (also used by model config-detect
/// tests in src/models/).
pub const TestBuilder = struct {
    buf: std.ArrayList(u8) = .empty,
    gpa: std.mem.Allocator,

    pub fn init(gpa: std.mem.Allocator, version: u32, n_tensors: u64, n_kv: u64) !TestBuilder {
        var b = TestBuilder{ .gpa = gpa };
        try b.buf.appendSlice(gpa, "GGUF");
        try b.int(u32, version);
        try b.int(u64, n_tensors);
        try b.int(u64, n_kv);
        return b;
    }

    pub fn int(self: *TestBuilder, comptime T: type, v: T) !void {
        var raw: [@sizeOf(T)]u8 = undefined;
        std.mem.writeInt(T, &raw, v, .little);
        try self.buf.appendSlice(self.gpa, &raw);
    }

    pub fn str(self: *TestBuilder, s: []const u8) !void {
        try self.int(u64, s.len);
        try self.buf.appendSlice(self.gpa, s);
    }

    pub fn kvUint(self: *TestBuilder, key: []const u8, v: u32) !void {
        try self.str(key);
        try self.int(u32, 4);
        try self.int(u32, v);
    }

    pub fn kvF32(self: *TestBuilder, key: []const u8, v: f32) !void {
        try self.str(key);
        try self.int(u32, 6);
        try self.int(u32, @bitCast(v));
    }

    pub fn kvStr(self: *TestBuilder, key: []const u8, v: []const u8) !void {
        try self.str(key);
        try self.int(u32, 8);
        try self.str(v);
    }

    pub fn tensor(self: *TestBuilder, name: []const u8, ne: []const u64, type_id: u32, offset: u64) !void {
        try self.str(name);
        try self.int(u32, @intCast(ne.len));
        for (ne) |d| try self.int(u64, d);
        try self.int(u32, type_id);
        try self.int(u64, offset);
    }

    /// Pad to the 32-byte data boundary and append the data section.
    pub fn finish(self: *TestBuilder, data: []const u8) ![]u8 {
        const aligned = std.mem.alignForward(usize, self.buf.items.len, 32);
        try self.buf.appendNTimes(self.gpa, 0, aligned - self.buf.items.len);
        try self.buf.appendSlice(self.gpa, data);
        return self.buf.toOwnedSlice(self.gpa);
    }

    pub fn deinit(self: *TestBuilder) void {
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

test "the q2_0 variant is detected from file geometry, not the type id" {
    const gpa = std.testing.allocator;
    // GGUF type 42 is claimed by two formats (see `detectQ2_0Variant`). Build the
    // SAME header twice, differing only in how much data each tensor occupies,
    // and require the parser to reach opposite conclusions. A 1024-element row is
    // 288 B as g64 (16 blocks x 18) and 272 B as g128 (8 x 34) — and 272 rounds up
    // to 288 under 32 B alignment, so the FIRST tensor's gap is deliberately
    // ambiguous. Detection must therefore come from the second tensor, which is
    // what pins that a too-small gap casts no vote instead of guessing.
    for ([_]struct { row: usize, want: DType }{
        .{ .row = 288, .want = .q2_0_g64 },
        .{ .row = 272, .want = .q2_0_g128 },
    }) |c| {
        const n_rows = 8;
        const stride = std.mem.alignForward(usize, c.row, 32);
        const payload = try gpa.alloc(u8, stride + c.row * n_rows);
        defer gpa.free(payload);
        @memset(payload, 0x24);

        var b = try TestBuilder.init(gpa, 3, 2, 1);
        defer b.deinit();
        try b.kvStr("general.architecture", "qwen3");
        try b.tensor("blk.0.attn_q.weight", &.{1024}, 42, 0);
        try b.tensor("blk.0.attn_k.weight", &.{ 1024, n_rows }, 42, stride);
        const file = try b.finish(payload);
        defer gpa.free(file);

        var g = try Gguf.initFromSlice(gpa, file);
        defer g.deinit();
        const t = try g.require("layers.0.self_attn.k_proj.weight");
        errdefer std.debug.print("row {d}: got {t}\n", .{ c.row, t.info.dtype });
        try std.testing.expectEqual(c.want, t.info.dtype);
        try std.testing.expectEqual(c.row * n_rows, t.bytes.len);
    }
}

test "a q2_0 file whose geometry fits neither variant is refused, not guessed" {
    const gpa = std.testing.allocator;
    // Sizes that match no candidate must be an error. Guessing would hand the
    // loader silently short, misaligned tensor views — the failure mode the
    // detector exists to prevent, and one no bounds check downstream can see.
    const payload = try gpa.alloc(u8, 4096);
    defer gpa.free(payload);
    @memset(payload, 0);

    var b = try TestBuilder.init(gpa, 3, 2, 1);
    defer b.deinit();
    try b.kvStr("general.architecture", "qwen3");
    try b.tensor("blk.0.attn_q.weight", &.{ 1024, 4 }, 42, 0);
    try b.tensor("blk.0.attn_k.weight", &.{ 1024, 4 }, 42, 1500); // neither 4*288 nor 4*272
    const file = try b.finish(payload);
    defer gpa.free(file);

    try std.testing.expectError(error.AmbiguousQ2_0Variant, Gguf.initFromSlice(gpa, file));
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

// `isVisionProjector` must distinguish a CLIP mmproj (arch "clip") from an LLM
// GGUF, so a vision-tower slot pointed at an LLM is rejected with a clear error
// instead of a cryptic config failure. Header-only (no weights); self-skips
// when the checkpoints are absent.
test "isVisionProjector: mmproj yes, LLM no" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const mmproj = "/home/qt/genai/lmstudio/models/mmproj-gemma-4-12b-it-qat-q4_0.gguf";
    const llm = "/home/qt/genai/lmstudio/models/gemma-4-12b-it-qat-q4_0.gguf";
    std.Io.Dir.cwd().access(io, mmproj, .{}) catch return error.SkipZigTest;
    std.Io.Dir.cwd().access(io, llm, .{}) catch return error.SkipZigTest;

    var mg = try Gguf.open(gpa, io, mmproj);
    defer mg.deinit();
    try std.testing.expect(mg.isVisionProjector());

    var lg = try Gguf.open(gpa, io, llm);
    defer lg.deinit();
    try std.testing.expect(!lg.isVisionProjector()); // an LLM in the mmproj slot
}

test "shape-fixed block-quantized tensors restore their logical shape and flag flat blocks" {
    // ⚠️ The regression this pins, measured 2026-08-02: the parser used to REFUSE to
    // restore `comfy.gguf.orig_shape` for a block-quantized tensor, on the reasoning
    // that the shape fix only ever lands on tensors the converter then leaves
    // unquantized. True of ComfyUI's converter; false of ggufy's, which blocks over a
    // tensor's flat element count and so reshapes *then* k-quantizes. Measured on a
    // ggufy SD1.5 q4_k file: 166 of 686 tensors, 72.7% of the parameters — every
    // convolution. The result was that TensorPencil could not load any ggufy SD-family
    // GGUF at all, so that whole output path was unmeasurable above level 1.
    const gpa = std.testing.allocator;

    // One q4_k super-block is 256 elements in 144 bytes. Stored as [1, 256]; the
    // logical shape is [2, 128] — 256 elements, but a row length that is NOT a
    // multiple of 256, which is exactly the case `Weight.init` cannot take.
    var payload: [144]u8 = @splat(0);
    payload[0] = 0x11; // any non-zero content; the parser does not decode it

    var b = try TestBuilder.init(gpa, 3, 1, 2);
    defer b.deinit();
    try b.kvStr("general.architecture", "sd1.5");
    // comfy.gguf.orig_shape.<name> = [2, 128], an i32 array in row-major order.
    try b.str("comfy.gguf.orig_shape.conv.weight");
    try b.int(u32, 9);
    try b.int(u32, 5);
    try b.int(u64, 2);
    try b.int(u32, 2);
    try b.int(u32, 128);
    // ggml ne [256, 1] = stored torch shape [1, 256]; type 12 = q4_k.
    try b.tensor("conv.weight", &.{ 256, 1 }, 12, 0);
    const file = try b.finish(&payload);
    defer gpa.free(file);

    var g = try Gguf.initFromSlice(gpa, file);
    defer g.deinit();

    const v = g.get("conv.weight").?;
    // The logical shape is restored — a consumer must see [2, 128], not [1, 256].
    try std.testing.expectEqualSlices(usize, &.{ 2, 128 }, v.info.shape.slice());
    // ...and it is flagged, because 128 is not a whole number of q4_k blocks, so the
    // bytes can only be read as one flat 256-element sequence.
    try std.testing.expect(v.info.flat_blocks);
    // The byte range still follows the element count, which the reshape preserves.
    try std.testing.expectEqual(@as(usize, 144), v.bytes.len);
}

test "an unquantized shape fix does not claim flat blocks" {
    // The krea2 case, and the reason `flat_blocks` is not simply "has orig_shape":
    // an f32 tensor has no blocks at all, so nothing downstream needs to materialize
    // it and the fast path must stay available.
    const gpa = std.testing.allocator;

    var payload: [24]u8 = @splat(0);
    var b = try TestBuilder.init(gpa, 3, 1, 2);
    defer b.deinit();
    try b.kvStr("general.architecture", "sd1.5");
    try b.str("comfy.gguf.orig_shape.embed.weight");
    try b.int(u32, 9);
    try b.int(u32, 5);
    try b.int(u64, 2);
    try b.int(u32, 3);
    try b.int(u32, 2);
    // ggml ne [2, 3] = stored [3, 2] f32; orig_shape says [3, 2] as well here, so the
    // only thing under test is that the flag stays false.
    try b.tensor("embed.weight", &.{ 2, 3 }, 0, 0);
    const file = try b.finish(&payload);
    defer gpa.free(file);

    var g = try Gguf.initFromSlice(gpa, file);
    defer g.deinit();
    const v = g.get("embed.weight").?;
    try std.testing.expect(!v.info.flat_blocks);
}
