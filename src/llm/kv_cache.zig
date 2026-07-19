//! Per-layer K/V cache for autoregressive decode (f32, host memory —
//! LLM_PLAN.md M2; a GPU-resident variant lands with M4).
//!
//! Layout: k/v are [n_layers][capacity][kv_dim] row-major. A forward pass
//! writes the new tokens' K/V at row `len` for every layer, then `commit`s
//! once so all layers observe the same position base.

const std = @import("std");

/// Element storage type for the KV cache K/V data. `f32` is the default and is
/// bit-exact (unchanged behavior); `f16` halves the footprint, `q8_0` (the ggml
/// 34-byte block quant: f16 scale + 32 x i8, ~1.06 B/elem) roughly quarters it.
/// Both are lossy — their output is NOT token-identical to f32.
pub const KvDtype = enum {
    f32,
    f16,
    q8_0,

    /// Storage bytes for `elems` contiguous K/V elements. This is the ONLY
    /// byte-size contract: q8_0 has no integral per-element width (34 bytes per
    /// 32-element block), so all sizing goes through here with block-aligned
    /// element counts. Every model kv_dim is a multiple of 32, so rows never
    /// split blocks.
    pub fn sizeBytes(self: KvDtype, elems: usize) usize {
        return switch (self) {
            .f32 => elems * 4,
            .f16 => elems * 2,
            .q8_0 => blk: {
                std.debug.assert(elems % q8_block_elems == 0);
                break :blk (elems / q8_block_elems) * q8_block_bytes;
            },
        };
    }

    /// Parse a CLI/config token ("f32" / "f16" / "q8_0"); null if unrecognized.
    pub fn parse(s: []const u8) ?KvDtype {
        if (std.mem.eql(u8, s, "f32")) return .f32;
        if (std.mem.eql(u8, s, "f16")) return .f16;
        if (std.mem.eql(u8, s, "q8_0")) return .q8_0;
        return null;
    }

    pub fn label(self: KvDtype) []const u8 {
        return switch (self) {
            .f32 => "f32",
            .f16 => "f16",
            .q8_0 => "q8_0",
        };
    }
};

/// ggml q8_0 block geometry: 32 elements -> f16 scale `d` + 32 x i8 = 34 bytes.
pub const q8_block_elems = 32;
pub const q8_block_bytes = 34;

/// Sizing plan for a session's KV cache: start with `initial` rows committed
/// and grow on demand up to `max`. `initial == max` is a fixed-capacity cache
/// (no growth) — required whenever the capacity is baked into device layouts
/// (speculative tree batch regions, EAGLE tap strides). `kv_dtype` selects the
/// K/V element storage type (default f32) and is threaded into every model
/// init so the device allocation and attention kernels agree on the width.
pub const Capacity = struct {
    initial: usize,
    max: usize,
    kv_dtype: KvDtype = .f32,

    pub fn fixed(n: usize) Capacity {
        return .{ .initial = n, .max = n };
    }
};

/// Default committed rows for a dynamic session (the growth floor): sessions
/// start here and grow toward --max-context as the conversation fills.
pub const initial_context = 4096;

/// Minimum growth increment (rows).
pub const grow_step = 1024;

/// Capacity to grow to when at least `min` rows are needed: geometric (1.5x,
/// at least grow_step) so repeated growth stays cheap, clamped to [min, max].
pub fn growTarget(cur: usize, min: usize, max: usize) usize {
    std.debug.assert(min <= max);
    const stepped = cur + @max(cur / 2, grow_step);
    return @min(max, @max(min, stepped));
}

/// The growth decision every stepper's `ensureCapacity` shares: from the
/// current committed capacity `cur` and the hard ceiling `max`, decide whether
/// `min_rows` needs a grow and, if so, to what capacity.
///   - `null`  → `min_rows` already fits; the caller returns without growing.
///   - `target`→ grow the cache (and RoPE tables) to this many rows.
///   - `error.ContextFull` → `min_rows` exceeds `max`; unrecoverable.
/// Single-sources the bounds-check + geometric `growTarget` that was copy-pasted
/// (and independently bug-fixed) across every CPU and CUDA stepper.
pub fn growPlan(cur: usize, max: usize, min_rows: usize) error{ContextFull}!?usize {
    if (min_rows <= cur) return null;
    if (min_rows > max) return error.ContextFull;
    return growTarget(cur, min_rows, max);
}

/// Non-f32 storage on the CPU packs the device byte layout into the f32
/// backing array (f16: two per slot; q8_0: raw 34-byte blocks) so the backing
/// array stays naturally f32-aligned and the f32 path is a zero-copy
/// sub-slice. `slotsFor(elems)` is the packed f32-slot count.
fn slotsFor(dt: KvDtype, elems: usize) usize {
    return switch (dt) {
        .f32 => elems,
        .f16 => (elems + 1) / 2, // KV rows are even, so no partial-slot aliasing
        .q8_0 => (dt.sizeBytes(elems) + 3) / 4,
    };
}

/// Pack `src` (f32) into `buf` at logical f16-element `base` (even). Each f32
/// slot holds elem 2k (low) and 2k+1 (high).
fn packF16(buf: []f32, base: usize, src: []const f32) void {
    var i: usize = 0;
    while (i + 1 < src.len) : (i += 2) {
        const lo: u16 = @bitCast(@as(f16, @floatCast(src[i])));
        const hi: u16 = @bitCast(@as(f16, @floatCast(src[i + 1])));
        buf[(base + i) / 2] = @bitCast(@as(u32, lo) | (@as(u32, hi) << 16));
    }
}

/// Expand `count` packed f16 elements at logical element `base` of `buf` into
/// `dst` (f32).
fn unpackF16(dst: []f32, buf: []const f32, base: usize, count: usize) void {
    var i: usize = 0;
    while (i + 1 < count) : (i += 2) {
        const word: u32 = @bitCast(buf[(base + i) / 2]);
        dst[i] = @floatCast(@as(f16, @bitCast(@as(u16, @truncate(word)))));
        dst[i + 1] = @floatCast(@as(f16, @bitCast(@as(u16, @truncate(word >> 16)))));
    }
}

/// Round to nearest, ties to EVEN. The q8_0 quantizer uses this instead of the
/// ggml reference's roundf (ties away from zero) so the CUDA store kernels
/// (`cvt.rni`) produce bit-identical cache bytes — a row quantized on the host
/// (offload split) matches the same row quantized on the device. Differs from
/// ggml only on exact .5 ties. Valid for |v| < 2^22 (quants are within ±127.5).
fn roundEven(v: f32) f32 {
    const magic: f32 = 8388608.0; // 2^23: adding forces the fraction out, RNE
    return if (v >= 0) (v + magic) - magic else (v - magic) + magic;
}

/// Quantize `src` (f32, block-multiple length) into ggml q8_0 blocks at logical
/// element `base` (block-aligned) of `buf`'s byte storage. Per 32-element
/// block: d = absmax/127 stored as f16, q[i] = roundEven(x[i]/d) as i8. The
/// byte layout is exactly ggml's block_q8_0 — the device caches' layout — so
/// rowBytes copies stay raw.
fn packQ80(buf: []f32, base: usize, src: []const f32) void {
    std.debug.assert(base % q8_block_elems == 0 and src.len % q8_block_elems == 0);
    const bytes = std.mem.sliceAsBytes(buf)[(base / q8_block_elems) * q8_block_bytes ..];
    var blk: usize = 0;
    while (blk * q8_block_elems < src.len) : (blk += 1) {
        const x = src[blk * q8_block_elems ..][0..q8_block_elems];
        var amax: f32 = 0;
        for (x) |e| amax = @max(amax, @abs(e));
        const d = amax / 127.0;
        const id: f32 = if (d != 0) 1.0 / d else 0.0;
        const out = bytes[blk * q8_block_bytes ..][0..q8_block_bytes];
        std.mem.writeInt(u16, out[0..2], @bitCast(@as(f16, @floatCast(d))), .little);
        for (x, out[2..]) |e, *q| q.* = @bitCast(@as(i8, @intFromFloat(roundEven(e * id))));
    }
}

/// Expand `count` q8_0 elements at logical element `base` of `buf` into `dst`
/// (f32). Same operation order as ggml's dequantize_row_q8_0 (widen d, then
/// q * d in f32), so the expansion is bit-identical to the reference.
fn unpackQ80(dst: []f32, buf: []const f32, base: usize, count: usize) void {
    std.debug.assert(base % q8_block_elems == 0 and count % q8_block_elems == 0);
    const bytes = std.mem.sliceAsBytes(buf)[(base / q8_block_elems) * q8_block_bytes ..];
    var blk: usize = 0;
    while (blk * q8_block_elems < count) : (blk += 1) {
        const in = bytes[blk * q8_block_bytes ..][0..q8_block_bytes];
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, in[0..2], .little))));
        for (dst[blk * q8_block_elems ..][0..q8_block_elems], in[2..]) |*o, q| {
            o.* = @as(f32, @floatFromInt(@as(i8, @bitCast(q)))) * d;
        }
    }
}

pub const KvCache = struct {
    /// Backing storage (f32 slots). For f16, two elements are packed per slot,
    /// so this is half the logical element count. Layer l's block starts at
    /// `l * slotsFor(capacity*kv_dim)`.
    k: []f32,
    v: []f32,
    /// f32 read scratch for the f16 path (one layer's rows expanded on view);
    /// empty for f32 (zero-copy sub-slice instead). Two, since attention reads
    /// K and V of the same layer at once.
    k_scratch: []f32 = &.{},
    v_scratch: []f32 = &.{},
    n_layers: usize,
    capacity: usize,
    kv_dim: usize,
    kv_dtype: KvDtype = .f32,
    /// Committed token count (positions 0..len are valid in every layer).
    len: usize = 0,

    pub fn init(gpa: std.mem.Allocator, n_layers: usize, capacity: usize, kv_dim: usize, kv_dtype: KvDtype) !KvCache {
        // q8_0 needs kv_dim % 64: an even block count per row keeps every row
        // (and layer) boundary 4-byte aligned in the f32-slot backing store,
        // so grow()'s slot-offset copies stay exact. All models are >= 512.
        std.debug.assert(kv_dtype != .q8_0 or kv_dim % (q8_block_elems * 2) == 0);
        const slots = slotsFor(kv_dtype, n_layers * capacity * kv_dim);
        const k = try gpa.alloc(f32, slots);
        errdefer gpa.free(k);
        const v = try gpa.alloc(f32, slots);
        errdefer gpa.free(v);
        var ks: []f32 = &.{};
        var vs: []f32 = &.{};
        if (kv_dtype != .f32) {
            ks = try gpa.alloc(f32, capacity * kv_dim);
            errdefer gpa.free(ks);
            vs = try gpa.alloc(f32, capacity * kv_dim);
        }
        return .{ .k = k, .v = v, .k_scratch = ks, .v_scratch = vs, .n_layers = n_layers, .capacity = capacity, .kv_dim = kv_dim, .kv_dtype = kv_dtype };
    }

    pub fn deinit(self: *KvCache, gpa: std.mem.Allocator) void {
        gpa.free(self.k);
        gpa.free(self.v);
        if (self.k_scratch.len != 0) gpa.free(self.k_scratch);
        if (self.v_scratch.len != 0) gpa.free(self.v_scratch);
        self.* = undefined;
    }

    /// Room left for new tokens.
    pub fn remaining(self: *const KvCache) usize {
        return self.capacity - self.len;
    }

    /// Write `n` tokens' K/V for `layer` at the uncommitted rows [len, len+n).
    pub fn write(self: *KvCache, layer: usize, k_new: []const f32, v_new: []const f32) void {
        std.debug.assert(k_new.len == v_new.len and k_new.len % self.kv_dim == 0);
        std.debug.assert(self.len * self.kv_dim + k_new.len <= self.capacity * self.kv_dim);
        const base = (layer * self.capacity + self.len) * self.kv_dim;
        switch (self.kv_dtype) {
            .f16 => {
                packF16(self.k, base, k_new);
                packF16(self.v, base, v_new);
            },
            .q8_0 => {
                packQ80(self.k, base, k_new);
                packQ80(self.v, base, v_new);
            },
            .f32 => {
                @memcpy(self.k[base..][0..k_new.len], k_new);
                @memcpy(self.v[base..][0..v_new.len], v_new);
            },
        }
    }

    /// K rows [0, len + extra) of `layer` — `extra` covers written-but-not-yet
    /// committed tokens mid-forward. For f16 this expands into `k_scratch`.
    pub fn kView(self: *KvCache, layer: usize, extra: usize) []const f32 {
        return self.view(self.k, self.k_scratch, layer, extra);
    }

    pub fn vView(self: *KvCache, layer: usize, extra: usize) []const f32 {
        return self.view(self.v, self.v_scratch, layer, extra);
    }

    /// Raw storage bytes of K rows [row, row+rows) of `layer`. The packed
    /// f16/q8_0 storage is byte-identical to the device caches' layout (a
    /// contiguous little-endian f16 array / ggml block_q8_0 array), so the
    /// CUDA steppers' migrate/promote device<->host copies stay raw (and
    /// lossless) for every dtype.
    pub fn kRowBytes(self: *KvCache, layer: usize, row: usize, rows: usize) []u8 {
        return self.rowBytes(self.k, layer, row, rows);
    }

    pub fn vRowBytes(self: *KvCache, layer: usize, row: usize, rows: usize) []u8 {
        return self.rowBytes(self.v, layer, row, rows);
    }

    fn rowBytes(self: *const KvCache, buf: []f32, layer: usize, row: usize, rows: usize) []u8 {
        std.debug.assert(layer < self.n_layers and row + rows <= self.capacity);
        const base = (layer * self.capacity + row) * self.kv_dim;
        // f16 packs 2 elements per f32 slot and q8_0 packs 32 per 34-byte
        // block; kv_dim is even (and a block multiple for q8_0) in every
        // model, so row boundaries never split a slot/block.
        std.debug.assert(self.kv_dtype != .f16 or base % 2 == 0);
        const dt = self.kv_dtype;
        return std.mem.sliceAsBytes(buf)[dt.sizeBytes(base)..][0..dt.sizeBytes(rows * self.kv_dim)];
    }

    fn view(self: *KvCache, buf: []f32, scratch: []f32, layer: usize, extra: usize) []const f32 {
        std.debug.assert(self.len + extra <= self.capacity);
        const rows_elems = (self.len + extra) * self.kv_dim;
        const layer_base = layer * self.capacity * self.kv_dim;
        switch (self.kv_dtype) {
            .f16 => unpackF16(scratch[0..rows_elems], buf, layer_base, rows_elems),
            .q8_0 => unpackQ80(scratch[0..rows_elems], buf, layer_base, rows_elems),
            .f32 => return buf[layer_base..][0..rows_elems],
        }
        return scratch[0..rows_elems];
    }

    /// Advance `len` after all layers have written the same `n` tokens.
    pub fn commit(self: *KvCache, n: usize) void {
        std.debug.assert(self.len + n <= self.capacity);
        self.len += n;
    }

    /// Roll back to `new_len` committed tokens (speculative-decode rejection):
    /// rows past `new_len` are simply overwritten by the next write.
    pub fn truncate(self: *KvCache, new_len: usize) void {
        std.debug.assert(new_len <= self.len);
        self.len = new_len;
    }

    /// Uniform cache has no sliding-window rings (all layers full context).
    /// Present so `transformer.layerForward` can query any cache generically.
    pub fn ringOf(self: *const KvCache, layer: usize) usize {
        _ = self;
        _ = layer;
        return 0;
    }

    /// Grow to `new_capacity` rows per layer. The per-layer blocks are
    /// re-strided into fresh arrays (committed rows copied, the rest left
    /// uninitialized, exactly like init). No-op when already large enough.
    pub fn grow(self: *KvCache, gpa: std.mem.Allocator, new_capacity: usize) !void {
        if (new_capacity <= self.capacity) return;
        const dt = self.kv_dtype;
        const nk = try gpa.alloc(f32, slotsFor(dt, self.n_layers * new_capacity * self.kv_dim));
        errdefer gpa.free(nk);
        const nv = try gpa.alloc(f32, slotsFor(dt, self.n_layers * new_capacity * self.kv_dim));
        errdefer gpa.free(nv);
        const used_slots = slotsFor(dt, self.len * self.kv_dim);
        for (0..self.n_layers) |l| {
            const src_base = slotsFor(dt, l * self.capacity * self.kv_dim);
            const dst_base = slotsFor(dt, l * new_capacity * self.kv_dim);
            @memcpy(nk[dst_base..][0..used_slots], self.k[src_base..][0..used_slots]);
            @memcpy(nv[dst_base..][0..used_slots], self.v[src_base..][0..used_slots]);
        }
        gpa.free(self.k);
        gpa.free(self.v);
        self.k = nk;
        self.v = nv;
        self.capacity = new_capacity;
        if (dt != .f32) {
            const ks = try gpa.alloc(f32, new_capacity * self.kv_dim);
            errdefer gpa.free(ks);
            const vs = try gpa.alloc(f32, new_capacity * self.kv_dim);
            gpa.free(self.k_scratch);
            gpa.free(self.v_scratch);
            self.k_scratch = ks;
            self.v_scratch = vs;
        }
    }
};

/// Like `KvCache` but each layer has its own `kv_dim`. Gemma 4 needs this:
/// its sliding-window layers hold 8 KV heads × 256 = 2048, while its global
/// layers hold 1 KV head × 512 = 512, so a single uniform stride would either
/// waste half the cache or misalign the attention view. Same write/commit/view
/// protocol as `KvCache`; the per-layer dims are borrowed at init (copied into
/// the caller's allocator so they outlive the source config).
pub const PerLayerKvCache = struct {
    /// Backing storage: f32 slots. For f16, two logical elements pack per slot
    /// (all layer offsets are even, so `offsets` stay LOGICAL element counts and
    /// the packed slot for logical element E is E/2 — see slotsFor/packF16).
    k: []f32,
    v: []f32,
    /// f32 read scratch (one layer's block) for the f16 path; empty for f32.
    k_scratch: []f32 = &.{},
    v_scratch: []f32 = &.{},
    n_layers: usize,
    capacity: usize,
    /// Per-layer kv_dim (owned copy).
    kv_dims: []usize,
    /// Per-layer ring size in rows (0 = full `capacity`, linear addressing).
    /// LOCAL sliding-window layers use a fixed ring so their storage doesn't
    /// grow with the conversation (TODO lever 1). Owned copy.
    rings: []usize,
    /// Logical-element base offset of each layer's block: `sum(layerRows(<l)*kv_dims)`.
    /// `offsets[n_layers]` is the total logical element count per k/v buffer.
    offsets: []usize,
    kv_dtype: KvDtype = .f32,
    len: usize = 0,

    /// Ring size of layer `l` (0 = full/linear) — read by the attention op.
    pub fn ringOf(self: *const PerLayerKvCache, l: usize) usize {
        return self.rings[l];
    }

    /// Largest single-layer block (logical elements) — the f16 read scratch size.
    fn maxBlock(capacity: usize, dims: []const usize, rings: []const usize) usize {
        var m: usize = 0;
        for (dims, 0..) |d, l| {
            const rows = if (rings.len != 0 and rings[l] != 0) rings[l] else capacity;
            m = @max(m, rows * d);
        }
        return m;
    }

    /// `rings` may be empty (no layer rings) or one entry per layer (0 = full).
    pub fn init(gpa: std.mem.Allocator, capacity: usize, kv_dims: []const usize, rings: []const usize, kv_dtype: KvDtype) !PerLayerKvCache {
        const n_layers = kv_dims.len;
        std.debug.assert(rings.len == 0 or rings.len == n_layers);
        // Like KvCache.init: q8_0 rows must hold an even block count so slot
        // arithmetic (grow) stays exact. All model dims are >= 512.
        if (kv_dtype == .q8_0) for (kv_dims) |d| std.debug.assert(d % (q8_block_elems * 2) == 0);
        const dims = try gpa.dupe(usize, kv_dims);
        errdefer gpa.free(dims);
        const rg = try gpa.alloc(usize, n_layers);
        errdefer gpa.free(rg);
        if (rings.len == 0) @memset(rg, 0) else @memcpy(rg, rings);
        const offsets = try gpa.alloc(usize, n_layers + 1);
        errdefer gpa.free(offsets);
        var acc: usize = 0;
        for (0..n_layers) |l| {
            offsets[l] = acc;
            const rows = if (rg[l] != 0) rg[l] else capacity;
            acc += rows * dims[l];
        }
        offsets[n_layers] = acc;
        const k = try gpa.alloc(f32, slotsFor(kv_dtype, acc));
        errdefer gpa.free(k);
        const v = try gpa.alloc(f32, slotsFor(kv_dtype, acc));
        errdefer gpa.free(v);
        var ks: []f32 = &.{};
        var vs: []f32 = &.{};
        if (kv_dtype != .f32) {
            const mb = maxBlock(capacity, kv_dims, rings);
            ks = try gpa.alloc(f32, mb);
            errdefer gpa.free(ks);
            vs = try gpa.alloc(f32, mb);
        }
        return .{ .k = k, .v = v, .k_scratch = ks, .v_scratch = vs, .n_layers = n_layers, .capacity = capacity, .kv_dims = dims, .rings = rg, .offsets = offsets, .kv_dtype = kv_dtype };
    }

    pub fn deinit(self: *PerLayerKvCache, gpa: std.mem.Allocator) void {
        gpa.free(self.k);
        gpa.free(self.v);
        gpa.free(self.kv_dims);
        gpa.free(self.rings);
        gpa.free(self.offsets);
        if (self.k_scratch.len != 0) gpa.free(self.k_scratch);
        if (self.v_scratch.len != 0) gpa.free(self.v_scratch);
        self.* = undefined;
    }

    pub fn remaining(self: *const PerLayerKvCache) usize {
        return self.capacity - self.len;
    }

    /// Copy/convert `src` (f32) into cache buffer `buf` at logical element
    /// `base` (even/block-aligned). f32 memcpy; f16 packs 2/slot; q8_0
    /// quantizes into 34-byte blocks.
    fn store(self: *const PerLayerKvCache, buf: []f32, base: usize, src: []const f32) void {
        switch (self.kv_dtype) {
            .f16 => packF16(buf, base, src),
            .q8_0 => packQ80(buf, base, src),
            .f32 => @memcpy(buf[base..][0..src.len], src),
        }
    }

    /// Write `n` tokens' K/V for `layer` at the uncommitted rows. Full layers
    /// append at `len`; ring layers write at `len % ring`, splitting on wrap.
    pub fn write(self: *PerLayerKvCache, layer: usize, k_new: []const f32, v_new: []const f32) void {
        const d = self.kv_dims[layer];
        std.debug.assert(k_new.len == v_new.len and (d == 0 or k_new.len % d == 0));
        const ring = self.rings[layer];
        if (ring == 0) {
            std.debug.assert(self.len * d + k_new.len <= self.capacity * d);
            const base = self.offsets[layer] + self.len * d;
            self.store(self.k, base, k_new);
            self.store(self.v, base, v_new);
            return;
        }
        const n = if (d == 0) 0 else k_new.len / d;
        std.debug.assert(n <= ring); // a single forward must fit the ring
        const start = self.len % ring;
        const first = @min(n, ring - start);
        const off = self.offsets[layer];
        self.store(self.k, off + start * d, k_new[0 .. first * d]);
        self.store(self.v, off + start * d, v_new[0 .. first * d]);
        if (first < n) {
            self.store(self.k, off, k_new[first * d ..]);
            self.store(self.v, off, v_new[first * d ..]);
        }
    }

    pub fn kView(self: *PerLayerKvCache, layer: usize, extra: usize) []const f32 {
        return self.view(self.k, self.k_scratch, layer, extra);
    }

    pub fn vView(self: *PerLayerKvCache, layer: usize, extra: usize) []const f32 {
        return self.view(self.v, self.v_scratch, layer, extra);
    }

    /// Full layers return rows [0, len+extra); ring layers return the whole
    /// ring block (the attention op indexes it by pos%ring, so it needs all
    /// rows and the caller passes `.ring = ringOf(layer)`). f16 expands into
    /// `scratch`.
    fn view(self: *PerLayerKvCache, buf: []f32, scratch: []f32, layer: usize, extra: usize) []const f32 {
        const d = self.kv_dims[layer];
        const ring = self.rings[layer];
        const count = if (ring != 0) ring * d else blk: {
            std.debug.assert(self.len + extra <= self.capacity);
            break :blk (self.len + extra) * d;
        };
        const base = self.offsets[layer];
        switch (self.kv_dtype) {
            .f16 => unpackF16(scratch[0..count], buf, base, count),
            .q8_0 => unpackQ80(scratch[0..count], buf, base, count),
            .f32 => return buf[base..][0..count],
        }
        return scratch[0..count];
    }

    /// Raw storage bytes of K rows [row, row+rows) of `layer`, addressed in the
    /// layer's own storage layout: absolute rows for full layers, ring rows
    /// (pos % ring) for ring layers. Like `KvCache.kRowBytes`, the packed f16
    /// slots are byte-identical to a contiguous little-endian f16 array — the
    /// device caches' exact layout — so a CUDA stepper's migrate/promote and
    /// ring-checkpoint copies stay raw (and lossless) for f32 and f16 alike.
    pub fn kRowBytes(self: *PerLayerKvCache, layer: usize, row: usize, rows: usize) []u8 {
        return self.rowBytes(self.k, layer, row, rows);
    }

    pub fn vRowBytes(self: *PerLayerKvCache, layer: usize, row: usize, rows: usize) []u8 {
        return self.rowBytes(self.v, layer, row, rows);
    }

    fn rowBytes(self: *const PerLayerKvCache, buf: []f32, layer: usize, row: usize, rows: usize) []u8 {
        const d = self.kv_dims[layer];
        const layer_rows = if (self.rings[layer] != 0) self.rings[layer] else self.capacity;
        std.debug.assert(layer < self.n_layers and row + rows <= layer_rows);
        const base = self.offsets[layer] + row * d;
        // f16 packs 2 elements per f32 slot and q8_0 packs 32 per 34-byte
        // block; kv_dim is even (and a block multiple for q8_0) in every
        // model, so row boundaries never split a slot/block.
        std.debug.assert(self.kv_dtype != .f16 or base % 2 == 0);
        const dt = self.kv_dtype;
        return std.mem.sliceAsBytes(buf)[dt.sizeBytes(base)..][0..dt.sizeBytes(rows * d)];
    }

    pub fn commit(self: *PerLayerKvCache, n: usize) void {
        std.debug.assert(self.len + n <= self.capacity);
        self.len += n;
    }

    pub fn truncate(self: *PerLayerKvCache, new_len: usize) void {
        std.debug.assert(new_len <= self.len);
        self.len = new_len;
    }

    /// Grow the FULL (non-ring) layers to `new_capacity`, re-striding into fresh
    /// buffers. Ring layers keep their fixed size. No-op when large enough.
    pub fn grow(self: *PerLayerKvCache, gpa: std.mem.Allocator, new_capacity: usize) !void {
        if (new_capacity <= self.capacity) return;
        const dt = self.kv_dtype;
        const new_offsets = try gpa.alloc(usize, self.n_layers + 1);
        errdefer gpa.free(new_offsets);
        var acc: usize = 0;
        for (0..self.n_layers) |l| {
            new_offsets[l] = acc;
            const rows = if (self.rings[l] != 0) self.rings[l] else new_capacity;
            acc += rows * self.kv_dims[l];
        }
        new_offsets[self.n_layers] = acc;
        const nk = try gpa.alloc(f32, slotsFor(dt, acc));
        errdefer gpa.free(nk);
        const nv = try gpa.alloc(f32, slotsFor(dt, acc));
        errdefer gpa.free(nv);
        for (0..self.n_layers) |l| {
            // Ring layers: copy the whole ring (any row may hold live data once
            // len>ring). Full layers: copy the committed rows.
            const copy_rows = if (self.rings[l] != 0) self.rings[l] else self.len;
            const used = slotsFor(dt, copy_rows * self.kv_dims[l]);
            @memcpy(nk[slotsFor(dt, new_offsets[l])..][0..used], self.k[slotsFor(dt, self.offsets[l])..][0..used]);
            @memcpy(nv[slotsFor(dt, new_offsets[l])..][0..used], self.v[slotsFor(dt, self.offsets[l])..][0..used]);
        }
        gpa.free(self.k);
        gpa.free(self.v);
        gpa.free(self.offsets);
        self.k = nk;
        self.v = nv;
        self.offsets = new_offsets;
        self.capacity = new_capacity;
        if (dt != .f32) {
            const mb = maxBlock(new_capacity, self.kv_dims, self.rings);
            const ks = try gpa.alloc(f32, mb);
            errdefer gpa.free(ks);
            const vs = try gpa.alloc(f32, mb);
            gpa.free(self.k_scratch);
            gpa.free(self.v_scratch);
            self.k_scratch = ks;
            self.v_scratch = vs;
        }
    }
};

// --- tests -----------------------------------------------------------------

test "write/commit/view round trip" {
    const gpa = std.testing.allocator;
    var cache = try KvCache.init(gpa, 2, 4, 3, .f32);
    defer cache.deinit(gpa);

    // Two tokens across both layers, then commit.
    const k0 = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const v0 = [_]f32{ -1, -2, -3, -4, -5, -6 };
    const k1 = [_]f32{ 10, 20, 30, 40, 50, 60 };
    cache.write(0, &k0, &v0);
    cache.write(1, &k1, &v0);
    try std.testing.expectEqual(@as(usize, 0), cache.len);
    try std.testing.expectEqualSlices(f32, &k0, cache.kView(0, 2));
    try std.testing.expectEqualSlices(f32, &k1, cache.kView(1, 2));
    cache.commit(2);
    try std.testing.expectEqual(@as(usize, 2), cache.len);
    try std.testing.expectEqual(@as(usize, 2), cache.remaining());

    // One more token lands at row 2 of each layer.
    const k2 = [_]f32{ 7, 8, 9 };
    cache.write(0, &k2, &k2);
    cache.commit(1);
    try std.testing.expectEqualSlices(f32, &(k0 ++ k2), cache.kView(0, 0));
    try std.testing.expectEqualSlices(f32, &(v0 ++ k2), cache.vView(0, 0));
}

test "KvCache f16 storage packs/expands (exact for f16-representable values)" {
    const gpa = std.testing.allocator;
    // dim 2, 4 rows, 2 layers, f16. Small integers are exact in f16, so the
    // round trip is bit-exact and the layout (2 elems/f32 slot) is verifiable.
    var cache = try KvCache.init(gpa, 2, 4, 2, .f16);
    defer cache.deinit(gpa);
    // Backing store is half the f32 element count (2 layers * 4 rows * 2 dim = 16
    // elems -> 8 f32 slots per buffer).
    try std.testing.expectEqual(@as(usize, 8), cache.k.len);

    const k0 = [_]f32{ 1, 2, 3, 4 }; // 2 tokens
    const v0 = [_]f32{ -1, -2, -3, -4 };
    cache.write(0, &k0, &v0);
    cache.write(1, &v0, &k0);
    cache.commit(2);
    try std.testing.expectEqualSlices(f32, &k0, cache.kView(0, 0));
    try std.testing.expectEqualSlices(f32, &v0, cache.vView(0, 0));
    try std.testing.expectEqualSlices(f32, &v0, cache.kView(1, 0));

    // grow keeps the committed f16 rows.
    try cache.grow(gpa, 8);
    try std.testing.expectEqual(@as(usize, 8), cache.capacity);
    try std.testing.expectEqualSlices(f32, &k0, cache.kView(0, 0));
    try std.testing.expectEqualSlices(f32, &k0, cache.vView(1, 0));
}

test "PerLayerKvCache f16 ring: wrap write + whole-ring view" {
    const gpa = std.testing.allocator;
    // Layer 0 full (dim 2), layer 1 ring of 3 rows (dim 2), f16.
    var cache = try PerLayerKvCache.init(gpa, 8, &.{ 2, 2 }, &.{ 0, 3 }, .f16);
    defer cache.deinit(gpa);
    // Write 5 tokens; the ring layer wraps at 3. Values exact in f16.
    for (0..5) |t| {
        const kf = [_]f32{ @floatFromInt(t * 2), @floatFromInt(t * 2 + 1) };
        cache.write(0, &kf, &kf);
        cache.write(1, &kf, &kf);
        cache.commit(1);
    }
    // Full layer: rows 0..5 present in order.
    try std.testing.expectEqualSlices(f32, &.{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }, cache.kView(0, 0));
    // Ring (3 rows, dim 2): pos%3 -> row0=pos3, row1=pos4, row2=pos2.
    const rv = cache.kView(1, 0);
    try std.testing.expectEqual(@as(usize, 6), rv.len);
    try std.testing.expectEqualSlices(f32, &.{ 6, 7 }, rv[0..2]); // pos 3
    try std.testing.expectEqualSlices(f32, &.{ 8, 9 }, rv[2..4]); // pos 4
    try std.testing.expectEqualSlices(f32, &.{ 4, 5 }, rv[4..6]); // pos 2
}

test "kRowBytes matches the device byte layout for f32 and f16" {
    const gpa = std.testing.allocator;
    // f32: raw bytes ARE the f32 rows.
    var c32 = try KvCache.init(gpa, 2, 4, 2, .f32);
    defer c32.deinit(gpa);
    const k0 = [_]f32{ 1.5, -2.25, 3, 4 }; // two tokens
    c32.write(1, &k0, &k0);
    c32.commit(2);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&k0), c32.kRowBytes(1, 0, 2));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(k0[2..4]), c32.vRowBytes(1, 1, 1));

    // f16: the packed slots must read back as a contiguous little-endian f16
    // array — the device caches' exact layout, so migrate/promote raw copies
    // are format-preserving.
    var c16 = try KvCache.init(gpa, 2, 4, 2, .f16);
    defer c16.deinit(gpa);
    c16.write(1, &k0, &k0);
    c16.commit(2);
    var expect16: [4]f16 = undefined;
    for (&expect16, k0) |*h, s| h.* = @floatCast(s);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&expect16), c16.kRowBytes(1, 0, 2));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(expect16[2..4]), c16.vRowBytes(1, 1, 1));

    // Round trip through the raw bytes (a promote after a migrate): writing
    // device-format bytes back in must reproduce the same expanded view.
    var c16b = try KvCache.init(gpa, 2, 4, 2, .f16);
    defer c16b.deinit(gpa);
    @memcpy(c16b.kRowBytes(1, 0, 2), c16.kRowBytes(1, 0, 2));
    c16b.commit(2);
    try std.testing.expectEqualSlices(f32, c16.kView(1, 0), c16b.kView(1, 0));
}

test "PerLayerKvCache row bytes match the device byte layout (full + ring, f32/f16)" {
    const gpa = std.testing.allocator;
    // Layer 0: full, dim 2. Layer 1: ring of 3 rows, dim 4 (gemma4-style mix).
    inline for ([_]KvDtype{ .f32, .f16 }) |dt| {
        var cache = try PerLayerKvCache.init(gpa, 4, &.{ 2, 4 }, &.{ 0, 3 }, dt);
        defer cache.deinit(gpa);
        const k0 = [_]f32{ 1.5, -2.25, 3, 4 }; // two tokens, dim 2
        const k1 = [_]f32{ 10, 11, 12, 13, 14, 15, 16, 17 }; // two tokens, dim 4
        cache.write(0, &k0, &k0);
        cache.write(1, &k1, &k1);
        cache.commit(2);

        // The raw rows must be the stored elements in storage order at the
        // dtype's width (the device caches' exact layout).
        if (dt == .f32) {
            try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&k0), cache.kRowBytes(0, 0, 2));
            try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(k1[4..8]), cache.vRowBytes(1, 1, 1));
        } else {
            var e0: [4]f16 = undefined;
            for (&e0, k0) |*h, s| h.* = @floatCast(s);
            try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&e0), cache.kRowBytes(0, 0, 2));
        }

        // Round trip device-format bytes into a fresh cache (a promote after a
        // migrate): views must agree, ring block included.
        var back = try PerLayerKvCache.init(gpa, 4, &.{ 2, 4 }, &.{ 0, 3 }, dt);
        defer back.deinit(gpa);
        @memcpy(back.kRowBytes(0, 0, 2), cache.kRowBytes(0, 0, 2));
        @memcpy(back.kRowBytes(1, 0, 3), cache.kRowBytes(1, 0, 3)); // whole ring
        back.commit(2);
        try std.testing.expectEqualSlices(f32, cache.kView(0, 0), back.kView(0, 0));
        try std.testing.expectEqualSlices(f32, cache.kView(1, 0)[0..8], back.kView(1, 0)[0..8]);
    }
}

test "grow preserves committed rows across the re-stride" {
    const gpa = std.testing.allocator;
    var cache = try KvCache.init(gpa, 2, 2, 3, .f32);
    defer cache.deinit(gpa);

    const k0 = [_]f32{ 1, 2, 3, 4, 5, 6 }; // two tokens
    const k1 = [_]f32{ 10, 20, 30, 40, 50, 60 };
    cache.write(0, &k0, &k1);
    cache.write(1, &k1, &k0);
    cache.commit(2);

    try cache.grow(gpa, 5);
    try std.testing.expectEqual(@as(usize, 5), cache.capacity);
    try std.testing.expectEqual(@as(usize, 3), cache.remaining());
    try std.testing.expectEqualSlices(f32, &k0, cache.kView(0, 0));
    try std.testing.expectEqualSlices(f32, &k1, cache.vView(0, 0));
    try std.testing.expectEqualSlices(f32, &k1, cache.kView(1, 0));

    // New rows land after the preserved ones.
    const k2 = [_]f32{ 7, 8, 9 };
    cache.write(0, &k2, &k2);
    cache.commit(1);
    try std.testing.expectEqualSlices(f32, &(k0 ++ k2), cache.kView(0, 0));
}

test "per-layer kv cache: variable dims, write/view/grow" {
    const gpa = std.testing.allocator;
    // Layer 0 dim 2, layer 1 dim 4 (mismatched, like gemma4 SWA vs global).
    var cache = try PerLayerKvCache.init(gpa, 3, &.{ 2, 4 }, &.{}, .f32);
    defer cache.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 0), cache.offsets[0]);
    try std.testing.expectEqual(@as(usize, 6), cache.offsets[1]); // 3*2
    try std.testing.expectEqual(@as(usize, 18), cache.offsets[2]); // 3*2 + 3*4

    const k0 = [_]f32{ 1, 2, 3, 4 }; // two tokens, dim 2
    const k1 = [_]f32{ 10, 11, 12, 13, 14, 15, 16, 17 }; // two tokens, dim 4
    cache.write(0, &k0, &k0);
    cache.write(1, &k1, &k1);
    try std.testing.expectEqualSlices(f32, &k0, cache.kView(0, 2));
    try std.testing.expectEqualSlices(f32, &k1, cache.kView(1, 2));
    cache.commit(2);
    try std.testing.expectEqual(@as(usize, 1), cache.remaining());

    try cache.grow(gpa, 8);
    try std.testing.expectEqual(@as(usize, 8), cache.capacity);
    try std.testing.expectEqualSlices(f32, &k0, cache.kView(0, 0));
    try std.testing.expectEqualSlices(f32, &k1, cache.vView(1, 0));
    // New token lands after the preserved rows in each (differently strided) layer.
    const n1 = [_]f32{ 100, 101, 102, 103 };
    cache.write(1, &n1, &n1);
    cache.commit(1);
    try std.testing.expectEqualSlices(f32, &(k1 ++ n1), cache.kView(1, 0));
}

test "per-layer ring: wrap write + whole-ring view + full-layer coexist" {
    const gpa = std.testing.allocator;
    // Layer 0: full (dim 1). Layer 1: ring of 3 rows (dim 1). Capacity 8.
    var cache = try PerLayerKvCache.init(gpa, 8, &.{ 1, 1 }, &.{ 0, 3 }, .f32);
    defer cache.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 0), cache.ringOf(0));
    try std.testing.expectEqual(@as(usize, 3), cache.ringOf(1));
    // Buffer holds layer0 (8 rows) + layer1 (3 rows) = 11 elems.
    try std.testing.expectEqual(@as(usize, 0), cache.offsets[0]);
    try std.testing.expectEqual(@as(usize, 8), cache.offsets[1]);
    try std.testing.expectEqual(@as(usize, 11), cache.offsets[2]);

    // Write 5 tokens one at a time to both layers; the ring layer wraps at 3.
    for (0..5) |t| {
        const kf = [_]f32{@floatFromInt(t)};
        cache.write(0, &kf, &kf);
        cache.write(1, &kf, &kf);
        cache.commit(1);
    }
    // Full layer: rows 0..5 = 0,1,2,3,4.
    try std.testing.expectEqualSlices(f32, &.{ 0, 1, 2, 3, 4 }, cache.kView(0, 0));
    // Ring layer (3 rows): pos%3 -> row0=pos3, row1=pos4, row2=pos2 (last 3
    // positions 2,3,4 survive, addressed by pos%3).
    const rv = cache.kView(1, 0);
    try std.testing.expectEqual(@as(usize, 3), rv.len);
    try std.testing.expectEqual(@as(f32, 3), rv[0]); // pos 3 at row 0
    try std.testing.expectEqual(@as(f32, 4), rv[1]); // pos 4 at row 1
    try std.testing.expectEqual(@as(f32, 2), rv[2]); // pos 2 at row 2

    // Grow keeps the ring intact and preserves the full layer's committed rows.
    try cache.grow(gpa, 16);
    try std.testing.expectEqual(@as(usize, 16), cache.offsets[1]); // full layer now 16 rows
    try std.testing.expectEqualSlices(f32, &.{ 0, 1, 2, 3, 4 }, cache.kView(0, 0));
    const rv2 = cache.kView(1, 0);
    try std.testing.expectEqual(@as(f32, 3), rv2[0]);
    try std.testing.expectEqual(@as(f32, 4), rv2[1]);
    try std.testing.expectEqual(@as(f32, 2), rv2[2]);
}

test "KvDtype storage size and parse" {
    try std.testing.expectEqual(@as(usize, 4), KvDtype.f32.sizeBytes(1));
    try std.testing.expectEqual(@as(usize, 2), KvDtype.f16.sizeBytes(1));
    try std.testing.expectEqual(@as(usize, 34), KvDtype.q8_0.sizeBytes(32));
    try std.testing.expectEqual(@as(usize, 34 * 32), KvDtype.q8_0.sizeBytes(1024));
    try std.testing.expectEqual(KvDtype.f32, KvDtype.parse("f32").?);
    try std.testing.expectEqual(KvDtype.f16, KvDtype.parse("f16").?);
    try std.testing.expectEqual(KvDtype.q8_0, KvDtype.parse("q8_0").?);
    try std.testing.expectEqual(@as(?KvDtype, null), KvDtype.parse("bf16"));
    // Default carrier dtype is f32 (unchanged behavior).
    try std.testing.expectEqual(KvDtype.f32, (Capacity.fixed(8)).kv_dtype);
}

test "q8_0 expand is bit-identical to the ggml reference block" {
    // Feed the golden fixture block (generated by the ggml reference
    // implementation) through unpackQ80 and expect the reference dequant bits.
    const fixtures = @import("../quants_fixtures.zig");
    var buf: [9]f32 = undefined; // 34 bytes rounded up to f32 slots
    @memcpy(std.mem.sliceAsBytes(&buf)[0..34], &fixtures.q8_0_block);
    var out: [32]f32 = undefined;
    unpackQ80(&out, &buf, 0, 32);
    for (fixtures.q8_0_expected_bits, out) |bits, got| {
        try std.testing.expectEqual(@as(f32, @bitCast(bits)), got);
    }
}

test "q8_0 quantize: scale, values, and integer exactness" {
    // A block whose absmax is 127 has d == 1.0, so integer values in
    // [-127, 127] round-trip exactly (and the stored bytes are predictable).
    var src: [64]f32 = undefined;
    for (&src, 0..) |*e, i| e.* = @floatFromInt(@as(i64, @intCast(i % 128)) - 63);
    src[0] = 127; // pin block 0's absmax
    src[32] = -127; // pin block 1's absmax
    var buf: [17 + 1]f32 = undefined; // 2 blocks = 68 bytes = 17 slots
    packQ80(&buf, 0, &src);

    const bytes = std.mem.sliceAsBytes(&buf);
    // d == 1.0 as f16 is 0x3C00.
    try std.testing.expectEqual(@as(u16, 0x3C00), std.mem.readInt(u16, bytes[0..2], .little));
    try std.testing.expectEqual(@as(u16, 0x3C00), std.mem.readInt(u16, bytes[34..36], .little));
    try std.testing.expectEqual(@as(i8, 127), @as(i8, @bitCast(bytes[2])));
    try std.testing.expectEqual(@as(i8, -127), @as(i8, @bitCast(bytes[36])));

    var out: [64]f32 = undefined;
    unpackQ80(&out, &buf, 0, 64);
    try std.testing.expectEqualSlices(f32, &src, &out);

    // All-zero block: d == 0, quants 0, expands to exact zeros.
    const zeros = [_]f32{0} ** 32;
    packQ80(&buf, 0, &zeros);
    unpackQ80(out[0..32], &buf, 0, 32);
    try std.testing.expectEqualSlices(f32, &zeros, out[0..32]);
}

test "q8_0 quantize round-trip error is within half a step" {
    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();
    var src: [128]f32 = undefined;
    for (&src) |*e| e.* = (rand.float(f32) - 0.5) * 20.0;
    var buf: [(128 / 32 * 34 + 3) / 4]f32 = undefined;
    packQ80(&buf, 0, &src);
    var out: [128]f32 = undefined;
    unpackQ80(&out, &buf, 0, 128);
    var blk: usize = 0;
    while (blk < 4) : (blk += 1) {
        var amax: f32 = 0;
        for (src[blk * 32 ..][0..32]) |e| amax = @max(amax, @abs(e));
        const step = amax / 127.0;
        for (src[blk * 32 ..][0..32], out[blk * 32 ..][0..32]) |want, got| {
            errdefer std.debug.print("blk {d}: want {d} got {d} step {d}\n", .{ blk, want, got, step });
            // Half a quantization step plus the f16 scale rounding slack
            // (dequant multiplies by the f16-rounded d: up to 127*d*2^-11 more).
            try std.testing.expect(@abs(want - got) <= step * 0.57);
        }
    }
}

test "KvCache q8_0 storage: write/view round trip and device byte layout" {
    const gpa = std.testing.allocator;
    // kv_dim 64 (the q8_0 minimum: even block count per row), 4 rows, 2 layers.
    var cache = try KvCache.init(gpa, 2, 4, 64, .q8_0);
    defer cache.deinit(gpa);
    // Backing store: 2 layers * 4 rows * 64 elems = 512 elems -> 16 blocks
    // * 34 B = 544 B -> 136 f32 slots.
    try std.testing.expectEqual(@as(usize, 136), cache.k.len);

    // Integer rows with absmax 127 in each block round-trip exactly.
    var k0: [128]f32 = undefined; // two tokens
    var v0: [128]f32 = undefined;
    for (&k0, 0..) |*e, i| e.* = @floatFromInt(@as(i64, @intCast(i % 100)) - 50);
    k0[0] = 127;
    k0[32] = 127;
    k0[64] = 127;
    k0[96] = 127;
    for (&v0, k0) |*e, s| e.* = -s;
    cache.write(0, &k0, &v0);
    cache.write(1, &v0, &k0);
    cache.commit(2);
    try std.testing.expectEqualSlices(f32, &k0, cache.kView(0, 0));
    try std.testing.expectEqualSlices(f32, &v0, cache.vView(0, 0));
    try std.testing.expectEqualSlices(f32, &v0, cache.kView(1, 0));

    // kRowBytes must be the ggml block_q8_0 device layout: 34-byte blocks,
    // f16 d then 32 quants. Row 1 of layer 0 starts at block 2.
    const row1 = cache.kRowBytes(0, 1, 1);
    try std.testing.expectEqual(@as(usize, 2 * 34), row1.len);
    try std.testing.expectEqual(@as(u16, 0x3C00), std.mem.readInt(u16, row1[0..2], .little)); // d == 1.0
    try std.testing.expectEqual(@as(i8, @intFromFloat(k0[64])), @as(i8, @bitCast(row1[2])));

    // Round trip device-format bytes into a fresh cache (promote after
    // migrate): views must agree bit-exactly.
    var back = try KvCache.init(gpa, 2, 4, 64, .q8_0);
    defer back.deinit(gpa);
    @memcpy(back.kRowBytes(0, 0, 2), cache.kRowBytes(0, 0, 2));
    back.commit(2);
    try std.testing.expectEqualSlices(f32, cache.kView(0, 0), back.kView(0, 0));

    // grow keeps the committed rows.
    try cache.grow(gpa, 8);
    try std.testing.expectEqual(@as(usize, 8), cache.capacity);
    try std.testing.expectEqualSlices(f32, &k0, cache.kView(0, 0));
    try std.testing.expectEqualSlices(f32, &k0, cache.vView(1, 0));
}

test "PerLayerKvCache q8_0: mixed dims, ring wrap, row bytes, grow" {
    const gpa = std.testing.allocator;
    // Layer 0 full (dim 64), layer 1 ring of 3 rows (dim 128) — gemma4-style.
    var cache = try PerLayerKvCache.init(gpa, 8, &.{ 64, 128 }, &.{ 0, 3 }, .q8_0);
    defer cache.deinit(gpa);

    // 5 tokens, one at a time; ring layer wraps at 3. Integer values with a
    // pinned absmax of 127 per block are exact in q8_0.
    var rows0: [5][64]f32 = undefined;
    var rows1: [5][128]f32 = undefined;
    for (0..5) |t| {
        for (&rows0[t], 0..) |*e, i| e.* = @floatFromInt(@as(i64, @intCast((t * 64 + i) % 100)) - 50);
        rows0[t][0] = 127;
        rows0[t][32] = 127;
        for (&rows1[t], 0..) |*e, i| e.* = @floatFromInt(@as(i64, @intCast((t * 128 + i) % 120)) - 60);
        for (0..4) |b| rows1[t][b * 32] = 127;
        cache.write(0, &rows0[t], &rows0[t]);
        cache.write(1, &rows1[t], &rows1[t]);
        cache.commit(1);
    }
    // Full layer: rows 0..5 in order.
    const fv = cache.kView(0, 0);
    for (0..5) |t| try std.testing.expectEqualSlices(f32, &rows0[t], fv[t * 64 ..][0..64]);
    // Ring layer (3 rows): pos%3 -> row0=pos3, row1=pos4, row2=pos2.
    const rv = cache.kView(1, 0);
    try std.testing.expectEqualSlices(f32, &rows1[3], rv[0..128]);
    try std.testing.expectEqualSlices(f32, &rows1[4], rv[128..256]);
    try std.testing.expectEqualSlices(f32, &rows1[2], rv[256..384]);

    // Raw ring-row bytes round trip into a fresh cache (gemma4 migrate/promote
    // moves whole rings).
    var back = try PerLayerKvCache.init(gpa, 8, &.{ 64, 128 }, &.{ 0, 3 }, .q8_0);
    defer back.deinit(gpa);
    @memcpy(back.kRowBytes(0, 0, 5), cache.kRowBytes(0, 0, 5));
    @memcpy(back.kRowBytes(1, 0, 3), cache.kRowBytes(1, 0, 3)); // whole ring
    back.commit(5);
    try std.testing.expectEqualSlices(f32, cache.kView(0, 0), back.kView(0, 0));
    try std.testing.expectEqualSlices(f32, cache.kView(1, 0), back.kView(1, 0));

    // Grow keeps the ring intact and the full layer's committed rows.
    try cache.grow(gpa, 16);
    try std.testing.expectEqualSlices(f32, &rows0[4], cache.kView(0, 0)[4 * 64 ..][0..64]);
    try std.testing.expectEqualSlices(f32, &rows1[3], cache.kView(1, 0)[0..128]);
}

test "roundEven ties go to even" {
    try std.testing.expectEqual(@as(f32, 2), roundEven(2.5));
    try std.testing.expectEqual(@as(f32, -2), roundEven(-2.5));
    try std.testing.expectEqual(@as(f32, 4), roundEven(3.5));
    try std.testing.expectEqual(@as(f32, 2), roundEven(1.5));
    try std.testing.expectEqual(@as(f32, 1), roundEven(0.75));
    try std.testing.expectEqual(@as(f32, 0), roundEven(0.25));
    try std.testing.expectEqual(@as(f32, -127), roundEven(-126.5001));
    try std.testing.expectEqual(@as(f32, 126), roundEven(126.4999));
}

test "growTarget is geometric and clamped" {
    try std.testing.expectEqual(@as(usize, 6144), growTarget(4096, 4097, 32768));
    try std.testing.expectEqual(@as(usize, 8192), growTarget(6000, 6001, 8192));
    try std.testing.expectEqual(@as(usize, 9000), growTarget(4096, 9000, 32768));
    try std.testing.expectEqual(@as(usize, 2048), growTarget(1024, 1025, 32768));
}

test "growPlan: fits / grows / overflows" {
    // Already fits (min_rows <= cur): no grow.
    try std.testing.expectEqual(@as(?usize, null), try growPlan(4096, 32768, 4096));
    try std.testing.expectEqual(@as(?usize, null), try growPlan(4096, 32768, 100));
    // Needs a grow: same target as growTarget.
    try std.testing.expectEqual(@as(?usize, 6144), try growPlan(4096, 32768, 4097));
    // Past the hard ceiling: unrecoverable.
    try std.testing.expectError(error.ContextFull, growPlan(4096, 8192, 8193));
}

test "truncate rolls back and rows are rewritten" {
    const gpa = std.testing.allocator;
    var cache = try KvCache.init(gpa, 1, 4, 2, .f32);
    defer cache.deinit(gpa);

    const a = [_]f32{ 1, 2, 3, 4, 5, 6 }; // three tokens
    cache.write(0, &a, &a);
    cache.commit(3);
    cache.truncate(1);
    try std.testing.expectEqual(@as(usize, 1), cache.len);
    try std.testing.expectEqual(@as(usize, 3), cache.remaining());

    // New tokens land at row 1, replacing the rolled-back ones.
    const b = [_]f32{ 9, 10 };
    cache.write(0, &b, &b);
    cache.commit(1);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 9, 10 }, cache.kView(0, 0));
}
