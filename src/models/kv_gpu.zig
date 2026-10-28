//! A Vulkan LLM's KV cache: one K and one V buffer per attention layer at the session's
//! dtype, written inside a recording batch and read by the dual flash-decoding attention.
//! A layer may be a ring (a sliding-window layer keeps only its last rows): position p
//! sits at row p % ring. Row widths may differ by layer (gemma4's local and global
//! geometry).

const std = @import("std");
const gpu = @import("tp_gpu").context;
const kvmod = @import("tp_core").kv_cache;

const Buf = gpu.DeviceBuffer;

pub const Kv = struct {
    k: []Buf,
    v: []Buf,
    /// Rows of each layer's ring, 0 for a linear layer.
    ring: []usize,
    dtype: kvmod.KvDtype,
    /// Each layer's elements per row: kv heads times head dim.
    dims: []usize,

    /// No layers; `deinit` takes it.
    pub const empty: Kv = .{ .k = &.{}, .v = &.{}, .ring = &.{}, .dtype = .f32, .dims = &.{} };

    /// Device bytes for `layers` linear layers of `capacity` rows.
    pub fn windowBytes(layers: usize, capacity: usize, dim: usize, dtype: kvmod.KvDtype) usize {
        return 2 * layers * dtype.sizeBytes(capacity * dim);
    }

    /// `ring` gives each layer's ring rows (0: linear, `capacity` rows), or is empty for
    /// all linear. It is copied.
    pub fn init(gpa: std.mem.Allocator, ctx: *gpu.Context, layers: usize, capacity: usize, dim: usize, dtype: kvmod.KvDtype, ring: []const usize) !Kv {
        const dims = try gpa.alloc(usize, layers);
        defer gpa.free(dims);
        @memset(dims, dim);
        return initDims(gpa, ctx, capacity, dims, dtype, ring);
    }

    /// `init` with each layer's row width; `dims` and `ring` are copied.
    pub fn initDims(gpa: std.mem.Allocator, ctx: *gpu.Context, capacity: usize, dims: []const usize, dtype: kvmod.KvDtype, ring: []const usize) !Kv {
        const layers = dims.len;
        std.debug.assert(ring.len == 0 or ring.len == layers);
        const rings = try gpa.alloc(usize, layers);
        errdefer gpa.free(rings);
        for (rings, 0..) |*r, l| r.* = if (ring.len == 0) 0 else ring[l];
        const own_dims = try gpa.dupe(usize, dims);
        errdefer gpa.free(own_dims);
        const prev = ctx.llmTag(.lm_kv);
        defer ctx.setMemTag(prev);
        const bufs = try gpa.alloc(Buf, 2 * layers);
        errdefer gpa.free(bufs);
        var made: usize = 0;
        errdefer for (bufs[0..made]) |*b| ctx.tensorDestroy(b);
        for (bufs, 0..) |*b, i| {
            const l = i % layers;
            b.* = try ctx.tensorCreate(dtype.sizeBytes((if (rings[l] != 0) rings[l] else capacity) * dims[l]));
            made += 1;
        }
        return .{ .k = bufs[0..layers], .v = bufs[layers..], .ring = rings, .dtype = dtype, .dims = own_dims };
    }

    /// Every linear layer's buffers taken to `capacity` rows, the first `used` rows
    /// kept; ring layers and dropped layers are untouched. One buffer at a time, so the
    /// peak is one extra, and a failure leaves each at least its old size, so a retry
    /// resumes where it stopped. Not inside a batch.
    pub fn grow(self: *Kv, ctx: *gpu.Context, capacity: usize, used: usize) !void {
        const prev = ctx.llmTag(.lm_kv);
        defer ctx.setMemTag(prev);
        for (self.ring, 0..) |r, l| {
            if (r != 0 or self.k[l].buf == .null_handle) continue;
            const keep = self.dtype.sizeBytes(used * self.dims[l]);
            const want = self.dtype.sizeBytes(capacity * self.dims[l]);
            for ([2]*Buf{ &self.k[l], &self.v[l] }) |b| {
                if (b.size >= want) continue; // grown by an earlier, interrupted pass
                var bigger = try ctx.tensorCreate(want);
                errdefer ctx.tensorDestroy(&bigger);
                if (keep > 0) try ctx.tensorCopy(bigger, 0, b.*, 0, keep);
                ctx.tensorDestroy(b);
                b.* = bigger;
            }
        }
    }

    /// Layer `l`'s live rows (positions `[0, len)`, a ring's last `ring` of them) into
    /// `host` slot `slot`, which shares the device byte layout, then its device buffers
    /// freed: the layer has moved to the host. `host` is a `kvmod.KvCache`, linear for
    /// every layer, or a `kvmod.PerLayerKvCache` built with this cache's rings.
    pub fn dropLayer(self: *Kv, ctx: *gpu.Context, l: usize, len: usize, host: anytype, slot: usize) !void {
        for (self.hostSegments(l, len, @TypeOf(host.*))) |seg| {
            if (seg.n == 0) continue;
            const t = self.segBytes(l, seg);
            try ctx.tensorDownload(try ctx.view(self.k[l], t.off, t.bytes), host.kRowBytes(slot, t.host_row, seg.n));
            try ctx.tensorDownload(try ctx.view(self.v[l], t.off, t.bytes), host.vRowBytes(slot, t.host_row, seg.n));
        }
        self.release(ctx, l);
    }

    /// The reverse of `dropLayer`: layer `l` back on the device, a linear layer at
    /// `capacity` rows, with the live rows from `host`.
    pub fn restoreLayer(self: *Kv, ctx: *gpu.Context, l: usize, capacity: usize, len: usize, host: anytype, slot: usize) !void {
        std.debug.assert(self.k[l].buf == .null_handle);
        const prev = ctx.llmTag(.lm_kv);
        defer ctx.setMemTag(prev);
        const rows = if (self.ring[l] != 0) self.ring[l] else capacity;
        for ([2]*Buf{ &self.k[l], &self.v[l] }) |b| b.* = try ctx.tensorCreate(self.dtype.sizeBytes(rows * self.dims[l]));
        for (self.hostSegments(l, len, @TypeOf(host.*))) |seg| {
            if (seg.n == 0) continue;
            const t = self.segBytes(l, seg);
            try ctx.tensorUpload(try ctx.view(self.k[l], t.off, t.bytes), host.kRowBytes(slot, t.host_row, seg.n));
            try ctx.tensorUpload(try ctx.view(self.v[l], t.off, t.bytes), host.vRowBytes(slot, t.host_row, seg.n));
        }
    }

    /// Take ring layer `l` to `new_ring` rows, keeping positions `[len - ring, len)`:
    /// the modulus is part of every row's address, so each live row moves. Staged
    /// through the host, the one form every KV dtype shares. Not inside a batch.
    pub fn regrowRing(self: *Kv, ctx: *gpu.Context, gpa: std.mem.Allocator, l: usize, new_ring: usize, len: usize) !void {
        const old = self.ring[l];
        std.debug.assert(old != 0 and new_ring >= old);
        if (new_ring == old) return;
        const row = self.dtype.sizeBytes(self.dims[l]);
        const live = @min(len, old);
        const stage = try gpa.alloc(u8, 2 * live * row);
        defer gpa.free(stage);
        const first = len - live;
        if (self.k[l].buf != .null_handle) {
            for (kvmod.ringSegments(len, old)) |seg| {
                if (seg.n == 0) continue;
                const at = (seg.abs - first) * row;
                try ctx.tensorDownload(try ctx.view(self.k[l], seg.dev * row, seg.n * row), stage[at..][0 .. seg.n * row]);
                try ctx.tensorDownload(try ctx.view(self.v[l], seg.dev * row, seg.n * row), stage[live * row + at ..][0 .. seg.n * row]);
            }
            const prev = ctx.llmTag(.lm_kv);
            defer ctx.setMemTag(prev);
            var k = try ctx.tensorCreate(new_ring * row);
            errdefer ctx.tensorDestroy(&k);
            const v = try ctx.tensorCreate(new_ring * row);
            self.release(ctx, l);
            self.k[l] = k;
            self.v[l] = v;
            // The new window is the old one's superset: write only what was staged.
            for (kvmod.ringSegments(len, new_ring)) |seg| {
                const lo = @max(seg.abs, first);
                const n = (seg.abs + seg.n) -| lo;
                if (n == 0) continue;
                const at = (lo - first) * row;
                const dev = (seg.dev + lo - seg.abs) * row;
                try ctx.tensorUpload(try ctx.view(self.k[l], dev, n * row), stage[at..][0 .. n * row]);
                try ctx.tensorUpload(try ctx.view(self.v[l], dev, n * row), stage[live * row + at ..][0 .. n * row]);
            }
        }
        self.ring[l] = new_ring;
    }

    const SegBytes = struct { off: usize, bytes: usize, host_row: usize };

    fn segBytes(self: *const Kv, l: usize, seg: kvmod.RingSeg) SegBytes {
        return .{ .off = self.dtype.sizeBytes(seg.dev * self.dims[l]), .bytes = self.dtype.sizeBytes(seg.n * self.dims[l]), .host_row = seg.abs };
    }

    /// Device ring rows against host rows: a linear host cache holds position p at row
    /// p, a `PerLayerKvCache` with the same rings at the device row, so its copy is raw.
    fn hostSegments(self: *const Kv, l: usize, len: usize, comptime Host: type) [2]kvmod.RingSeg {
        comptime std.debug.assert(Host == kvmod.KvCache or Host == kvmod.PerLayerKvCache);
        var segs = self.segments(l, len);
        if (Host == kvmod.PerLayerKvCache) {
            for (&segs) |*seg| seg.abs = seg.dev;
        }
        return segs;
    }

    /// Free layer `l`'s device buffers; a dropped layer's are already gone.
    pub fn release(self: *Kv, ctx: *gpu.Context, l: usize) void {
        ctx.tensorDestroy(&self.k[l]);
        ctx.tensorDestroy(&self.v[l]);
    }

    /// Bytes of a ring layer's whole buffer, each of K and V.
    pub fn ringBytes(self: *const Kv, l: usize) usize {
        std.debug.assert(self.ring[l] != 0);
        return self.dtype.sizeBytes(self.ring[l] * self.dims[l]);
    }

    fn segments(self: *const Kv, l: usize, len: usize) [2]kvmod.RingSeg {
        if (self.ring[l] != 0) return kvmod.ringSegments(len, self.ring[l]);
        return .{ .{ .abs = 0, .dev = 0, .n = len }, .{ .abs = len, .dev = 0, .n = 0 } };
    }

    pub fn deinit(self: *Kv, gpa: std.mem.Allocator, ctx: *gpu.Context) void {
        const all = self.k.ptr[0 .. 2 * self.k.len];
        for (all) |*b| ctx.tensorDestroy(b);
        gpa.free(all);
        gpa.free(self.ring);
        gpa.free(self.dims);
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
        const e = n * self.dims[l];
        const off = row * self.dims[l];
        const src = src_row * self.dims[l];
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
        const bytes = n * self.dims[l] * 4;
        return .{ try ctx.view(self.k[l], row * self.dims[l] * 4, bytes), try ctx.view(self.v[l], row * self.dims[l] * 4, bytes) };
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
