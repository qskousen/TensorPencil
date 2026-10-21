//! Host halves the encoder tower walks share (`embed_*_dev.zig`, over `tower_cuda.Dev`
//! / `tower_gpu.Dev`): a ragged batch's row layout and its attention bounds, the
//! rotate-half table the device RoPE reads, and the walk over a model's host buffers
//! that eviction uses.

const std = @import("std");
const ops = @import("tp_ops");
const Weight = ops.matmul.Weight;

/// Items packed back to back: item `i` is rows `[off[i], off[i + 1])`.
pub const Ragged = struct {
    off: []usize,
    total: usize,
    longest: usize,

    pub fn init(gpa: std.mem.Allocator, lens: anytype) !Ragged {
        const off = try gpa.alloc(usize, lens.len + 1);
        off[0] = 0;
        var longest: usize = 0;
        for (lens, 0..) |l, i| {
            const n = if (@TypeOf(l) == usize) l else l.len;
            off[i + 1] = off[i] + n;
            longest = @max(longest, n);
        }
        return .{ .off = off, .total = off[lens.len], .longest = longest };
    }

    pub fn deinit(self: *Ragged, gpa: std.mem.Allocator) void {
        gpa.free(self.off);
        self.* = undefined;
    }

    pub fn items(self: Ragged) usize {
        return self.off.len - 1;
    }

    pub fn len(self: Ragged, i: usize) usize {
        return self.off[i + 1] - self.off[i];
    }

    /// Per-row `[start | end]` of the row's item, the block-diagonal attention's mask.
    /// Caller frees.
    pub fn bounds(self: Ragged, gpa: std.mem.Allocator) ![]u32 {
        const out = try gpa.alloc(u32, 2 * self.total);
        self.boundsInto(out, 0);
        return out;
    }

    /// `bounds` into `out` [2 * total]; a nonzero `radius` also clips each row to the
    /// keys within `radius` positions of it, either side.
    pub fn boundsInto(self: Ragged, out: []u32, radius: usize) void {
        std.debug.assert(out.len == 2 * self.total);
        for (0..self.items()) |i| for (self.off[i]..self.off[i + 1]) |r| {
            const lo = self.off[i];
            const hi = self.off[i + 1];
            out[r] = @intCast(if (radius == 0) lo else @max(lo, r -| radius));
            out[self.total + r] = @intCast(if (radius == 0) hi else @min(hi, r + radius + 1));
        };
    }
};

/// Key chunks per row for `attentionRows`: enough subgroups in flight to fill a GPU,
/// at least 32 keys a chunk, at most 16 chunks.
pub fn attnSplit(rows: usize, heads: usize, longest: usize) usize {
    const want = std.math.divCeil(usize, 4096, rows * heads) catch unreachable;
    return std.math.clamp(@min(want, longest / 32), 1, 16);
}

/// `[cos | sin]` rotate-half table over `rows` positions; the sine half starts at
/// `rows * head_dim / 2`. Caller frees.
pub fn ropeTable(gpa: std.mem.Allocator, rows: usize, head_dim: usize, theta: f64) ![]f32 {
    const half = head_dim / 2;
    var f = try ops.rope.rotateHalfFreqs(gpa, rows, head_dim, theta);
    defer f.deinit(gpa);
    const flat = try gpa.alloc(f32, rows * half * 2);
    @memcpy(flat[0 .. rows * half], f.cos);
    @memcpy(flat[rows * half ..], f.sin);
    return flat;
}

/// Hand `sink` every host buffer of `model` a device cache can key on: `weight(bytes)`
/// for each `Weight` and `?Weight` field, `vector(bytes)` for each f32 slice, recursing
/// into slices of structs and optional structs. Both device caches key on host
/// addresses, so a model freed without this leaves copies that the next buffer at the
/// same address is served.
pub fn forEachHostBuffer(model: anytype, sink: anytype) void {
    inline for (@typeInfo(@TypeOf(model.*)).@"struct".field_names) |name| visit(&@field(model.*, name), sink);
}

fn visit(p: anytype, sink: anytype) void {
    const F = @TypeOf(p.*);
    if (F == Weight) return sink.weight(p.bytes);
    if (F == ?Weight) return if (p.*) |w| sink.weight(w.bytes);
    if (F == []const f32 or F == []f32) return sink.vector(std.mem.sliceAsBytes(p.*));
    switch (@typeInfo(F)) {
        .pointer => |pi| if (pi.size == .slice and @typeInfo(pi.child) == .@"struct") {
            for (p.*) |*e| forEachHostBuffer(e, sink);
        },
        .optional => |o| if (@typeInfo(o.child) == .@"struct") {
            if (p.*) |*v| forEachHostBuffer(v, sink);
        },
        else => {},
    }
}
