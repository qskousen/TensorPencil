//! A Vulkan LLM's KV cache: one K and one V buffer per attention layer at the session's
//! dtype, written inside a recording batch and read by the dual flash-decoding attention.
//! A layer may be a ring (a sliding-window layer keeps only its last rows): position p
//! sits at row p % ring.

const std = @import("std");
const gpu = @import("tp_gpu").context;
const kvmod = @import("tp_core").kv_cache;

const Buf = gpu.DeviceBuffer;

pub const Kv = struct {
    k: []Buf,
    v: []Buf,
    /// Rows of each layer's ring, 0 for a linear layer.
    ring: []const usize,
    dtype: kvmod.KvDtype,
    /// Elements per row: kv heads times head dim.
    dim: usize,

    /// Device bytes for `layers` linear layers of `capacity` rows.
    pub fn windowBytes(layers: usize, capacity: usize, dim: usize, dtype: kvmod.KvDtype) usize {
        return 2 * layers * dtype.sizeBytes(capacity * dim);
    }

    /// `ring` gives each layer's ring rows (0: linear, `capacity` rows), or is empty for
    /// all linear. It is copied.
    pub fn init(gpa: std.mem.Allocator, ctx: *gpu.Context, layers: usize, capacity: usize, dim: usize, dtype: kvmod.KvDtype, ring: []const usize) !Kv {
        std.debug.assert(ring.len == 0 or ring.len == layers);
        const rings = try gpa.alloc(usize, layers);
        errdefer gpa.free(rings);
        for (rings, 0..) |*r, l| r.* = if (ring.len == 0) 0 else ring[l];
        const bufs = try gpa.alloc(Buf, 2 * layers);
        errdefer gpa.free(bufs);
        var made: usize = 0;
        errdefer for (bufs[0..made]) |*b| ctx.tensorDestroy(b);
        for (bufs, 0..) |*b, i| {
            const r = rings[i % layers];
            b.* = try ctx.tensorCreate(dtype.sizeBytes((if (r != 0) r else capacity) * dim));
            made += 1;
        }
        return .{ .k = bufs[0..layers], .v = bufs[layers..], .ring = rings, .dtype = dtype, .dim = dim };
    }

    pub fn deinit(self: *Kv, gpa: std.mem.Allocator, ctx: *gpu.Context) void {
        const all = self.k.ptr[0 .. 2 * self.k.len];
        for (all) |*b| ctx.tensorDestroy(b);
        gpa.free(all);
        gpa.free(self.ring);
        self.* = undefined;
    }

    /// Positions `[at, at + n)` of layer `l` from `k` and `v` (`[n][dim]` f32), split
    /// where a ring wraps. In-batch: `tensorCopy` would flush the recording.
    pub fn store(self: *const Kv, ctx: *gpu.Context, l: usize, k: Buf, v: Buf, at: usize, n: usize) !void {
        const r = self.ring[l];
        if (r == 0) return self.storeRows(ctx, l, k, v, at, 0, n);
        std.debug.assert(n <= r);
        const first = @min(n, r - at % r);
        try self.storeRows(ctx, l, k, v, at % r, 0, first);
        if (first < n) try self.storeRows(ctx, l, k, v, 0, first, n - first);
    }

    fn storeRows(self: *const Kv, ctx: *gpu.Context, l: usize, k: Buf, v: Buf, row: usize, src_row: usize, n: usize) !void {
        const e = n * self.dim;
        const off = row * self.dim;
        const src = src_row * self.dim;
        for ([2]Buf{ self.k[l], self.v[l] }, [2]Buf{ k, v }) |dst, s| switch (self.dtype) {
            .f16 => try ctx.opStoreKvF16(dst, off, s, src, e),
            .q8_0 => try ctx.opStoreKvQ8(dst, off, s, src, e),
            .f32 => try ctx.opElt(.copy, s, dst, null, null, .{ .u0 = @intCast(e), .u2 = @intCast(off), .u3 = @intCast(src) }, e, 1, 1),
        };
    }

    /// Layer `l`'s K and V rows for positions `[at, at + n)` as views a producer can
    /// write in place of `store`: an f32 cache whose rows do not wrap there. Null
    /// otherwise.
    pub fn rowsAt(self: *const Kv, ctx: *gpu.Context, l: usize, at: usize, n: usize) !?[2]Buf {
        if (self.dtype != .f32) return null;
        const r = self.ring[l];
        const row = if (r == 0) at else at % r;
        if (r != 0 and row + n > r) return null;
        const bytes = n * self.dim * 4;
        return .{ try ctx.view(self.k[l], row * self.dim * 4, bytes), try ctx.view(self.v[l], row * self.dim * 4, bytes) };
    }

    /// `n` queries `[n][heads][hd]` against layer `l`, query t seeing the first
    /// `kv_len0 + t` positions, or the last `window` of them, or every position below
    /// `bidir_end` (an image block). `scratch` holds `n * heads * nsplit * (hd + 2)`
    /// floats.
    pub fn attend(self: *const Kv, ctx: *gpu.Context, l: usize, q: Buf, out: Buf, scratch: Buf, kv_len0: usize, n: usize, heads: usize, kv_heads: usize, hd: usize, nsplit: usize, scale: f32, window: usize, bidir_end: usize) !void {
        try ctx.opAttnRows(q, self.k[l], self.v[l], out, scratch, kv_len0, n, heads, kv_heads, hd, nsplit, scale, fmt(self.dtype), .{ .window = window, .ring = self.ring[l], .bidir_end = bidir_end });
    }
};

fn fmt(dt: kvmod.KvDtype) gpu.KvFmt {
    return switch (dt) {
        .f32 => .f32,
        .f16 => .f16,
        .q8_0 => .q8_0,
    };
}
