//! K2 Horizon CUDA inference. Fixed weights and as many routed-expert groups as
//! fit stay resident; the rest live on the host, run on the CPU for decode and
//! tiny batches, and are staged to the GPU once per prefill batch per layer.

const std = @import("std");
const k2 = @import("k2_horizon.zig");
const qwen3 = @import("qwen3.zig");
const cuda = @import("tp_gpu").cuda;
const Gguf = @import("tp_core").gguf.Gguf;
const Tokenizer = @import("tp_core").tokenizer.Tokenizer;
const chat = @import("../llm/chat.zig");
const ops = @import("tp_ops");
const kvmod = @import("tp_core").kv_cache;
const sample = @import("tp_core").sample;
const bnd = @import("tp_runtime").boundary;
const residency = @import("tp_runtime").residency;

const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;
const Growable = Backend.GrowableTensor;
const Weight = ops.matmul.Weight;

pub const CpuSplitPolicy = enum { tail, attn };

const prefill_rows = 2048;
/// Tokens per routed-expert pass. The f16 expert path dequantizes a layer's whole
/// expert group per pass, so one pass per batch; the scratch is sized for it.
const routed_chunk = prefill_rows;
/// How a batch's routed experts multiply. f16 dequantizes the group once and runs
/// one f16 GEMM per expert; mmq/gemv are the packed q6_k kernels (`TP_K2_EXPERT_GEMM`).
const ExpertGemm = enum { f16, mmq, gemv };
/// Batches up to this size run a host-resident layer's experts on the CPU; larger
/// ones upload the layer's experts to the staging buffers and run on the GPU.
const cpu_small_max = 8;
const grouped_gemv_max = 40;

const HybridProfile = struct {
    enabled: bool,
    route_ns: u64 = 0,
    transfer_ns: u64 = 0,
    values_ns: u64 = 0,
    ffn_ns: u64 = 0,

    fn tic(self: *const HybridProfile) u64 {
        return if (self.enabled) cuda.context.monoNs() else 0;
    }

    fn toc(self: *HybridProfile, start: u64, field: *u64) void {
        if (self.enabled) field.* += cuda.context.monoNs() -| start;
    }

    fn reset(self: *HybridProfile) void {
        self.route_ns = 0;
        self.transfer_ns = 0;
        self.values_ns = 0;
        self.ffn_ns = 0;
    }
};

fn offsetBuf(b: Buf, off: usize, size: usize) Buf {
    return .{ .buf = @enumFromInt(@intFromEnum(b.buf) + off), .mem = b.mem, .size = size };
}

fn nbuf(be: *Backend, weights: []const f32) !Buf {
    return .{ .buf = try be.smallBuffer(std.mem.sliceAsBytes(weights)), .mem = .null_handle, .size = 0 };
}

fn kvFmt(dt: kvmod.KvDtype) cuda.backend.KvFmt {
    return switch (dt) {
        .f32 => .f32,
        .f16 => .f16,
        .q8_0 => .q8_0,
    };
}

/// Planning cap on device-resident routed experts (`TP_K2_EXPERT_CACHE_GIB`).
fn expertCacheLimit(be: *Backend) u64 {
    const total = be.ctx.memGetInfo().total;
    const reserve: u64 = 2 << 30;
    const max_cache: u64 = 21 << 30;
    const default_limit = if (total > reserve) @min(max_cache, total - reserve) else total / 4;
    const raw = std.c.getenv("TP_K2_EXPERT_CACHE_GIB") orelse return default_limit;
    const gib = std.fmt.parseInt(u64, std.mem.span(raw), 10) catch return default_limit;
    if (gib > 64) return default_limit;
    return @min(gib << 30, default_limit);
}

fn prefillRows(max: usize) usize {
    const raw = std.c.getenv("TP_K2_PREFILL_ROWS") orelse return @min(prefill_rows, max);
    const rows = std.fmt.parseInt(usize, std.mem.span(raw), 10) catch return @min(prefill_rows, max);
    if (rows == 0 or rows > 4096) return @min(prefill_rows, max);
    return @min(rows, max);
}

fn attnSplits(seq: usize) usize {
    return if (seq == 1) 32 else if (seq < 32) 8 else 1;
}

/// Batches this long attend through the tensor-core path (f32 KV only); the
/// flash-split kernel is scalar and quadratic, and dominates long prefills.
const tc_attn_min = 64;
/// Cap on one head's f16 score plane; longer contexts take smaller query tiles.
const attn_plane_max: usize = 128 << 20;

fn groupBytes(weights: []const Weight) []const u8 {
    return weights[0].bytes.ptr[0 .. weights[0].bytes.len * weights.len];
}

/// Device copies of one host-resident layer's expert groups, refilled once per
/// prefill batch per layer, so the expert cache never sees them.
/// Packed route rows a batch of `rows` tokens can need: MMQ pads each expert to
/// 32-row tiles, the f16 GEMMs run up to 127 rows past the last expert, and the
/// per-expert fallback gathers up to `rows` rows for one expert.
fn routeSlots(c: k2.Config, rows: usize) usize {
    const max_experts = @max(c.n_experts, c.n_value_experts);
    const max_used = @max(c.n_experts_used, c.n_value_experts_used);
    return @max(rows, @min(rows, routed_chunk) * max_used + max_experts * 31 + 128);
}

/// u32 slots of the one-upload route pack: ids, scales, slot map, group table.
fn routePackSlots(c: k2.Config, rows: usize) usize {
    const max_experts = @max(c.n_experts, c.n_value_experts);
    const max_used = @max(c.n_experts_used, c.n_value_experts_used);
    const slots = routeSlots(c, rows);
    return 2 * slots + rows * max_used + slots / 8 + max_experts;
}

const Stage = struct {
    gate: Buf,
    up: Buf,
    down: Buf,
    values: ?Buf,
    layer: usize = 0,
    pending: [4]?Backend.StagedUpload = .{ null, null, null, null },
    after: ?cuda.cu.CUevent = null,

    fn bytes(experts: k2.Experts, values: ?k2.Values) u64 {
        var total: u64 = groupBytes(experts.gate).len + groupBytes(experts.up).len + groupBytes(experts.down).len;
        if (values) |v| total += groupBytes(v.weights).len;
        return total;
    }
};

const RoutePack = struct {
    counts: []usize,
    starts: []usize,
    route_ids: []u32,
    route_scales: []f32,
    expert_ids: []u32,
    /// Inverse map: the packed row of token t's k-th route, at t*used+k.
    slot_rows: []u32,
};

/// Packed routes for one batch, plus the device views the kernels read. Up to
/// this many packed rows run through the grouped q8 GEMV; more take the f16 GEMM.
const gemv_rows_max = 256;

/// Pack the routes of `seq` tokens by expert: `route_ids`/`route_scales` hold
/// (token, weight) per gathered row in expert-major, token-ascending order,
/// `expert_ids` one packed group per weight tile, and `slot_rows` the inverse
/// map the combine reads. MMQ tiles are 32 rows and pad each expert's tail with
/// zero-weight rows of token 0; GEMV groups hold up to 8 rows and f16 tiles up
/// to 128, both carrying their count.
fn packRoutes(kind: ExpertGemm, seq: usize, selected: []const usize, weights: []const f32, used: usize, n_experts: usize, out: RoutePack) GroupedRoutes {
    const mmq = kind == .mmq;
    const tile: usize = switch (kind) {
        .mmq => 32,
        .gemv => 8,
        .f16 => 128,
    };
    const counts = out.counts[0..n_experts];
    const starts = out.starts[0..n_experts];
    @memset(counts, 0);
    for (selected[0 .. seq * used]) |e| counts[e] += 1;
    var dst: usize = 0;
    var groups: usize = 0;
    for (counts, starts, 0..) |count, *start, expert| {
        start.* = dst;
        if (count == 0) continue;
        var at: usize = 0;
        while (at < count) : (at += tile) {
            const n = @min(tile, count - at);
            out.expert_ids[groups] = switch (kind) {
                .mmq => expertMmqGroup(expert, dst + at),
                .gemv => routeGroup(expert, n, dst + at),
                .f16 => cuda.kernels.hgemmGroup(expert, n, dst + at),
            };
            groups += 1;
        }
        const padded = if (mmq) (count + tile - 1) / tile * tile else count;
        @memset(out.route_ids[dst + count .. dst + padded], 0);
        @memset(out.route_scales[dst + count .. dst + padded], 0);
        dst += padded;
    }
    for (0..seq) |t| for (0..used) |slot| {
        const at = t * used + slot;
        const e = selected[at];
        out.route_ids[starts[e]] = @intCast(t);
        out.route_scales[starts[e]] = weights[at];
        out.slot_rows[at] = @intCast(starts[e]);
        starts[e] += 1;
    };
    std.debug.assert(dst >= seq * used);
    return .{ .rows = dst, .groups = groups };
}

const GroupedRoutes = struct {
    rows: usize,
    groups: usize,
    ids: Buf = .{},
    scales: Buf = .{},
    slots: Buf = .{},
    groups_d: Buf = .{},
};

fn routeGroup(expert: usize, count: usize, offset: usize) u32 {
    std.debug.assert(expert < 256 and count >= 1 and count <= 8 and offset < (1 << 20));
    return @as(u32, @intCast(expert)) |
        (@as(u32, @intCast(count)) << 8) |
        (@as(u32, @intCast(offset)) << 12);
}

fn expertMmqGroup(expert: usize, offset: usize) u32 {
    std.debug.assert(expert < 256 and offset < (1 << 20));
    return @as(u32, @intCast(expert)) | (@as(u32, @intCast(offset)) << 12);
}

test "packRoutes matches the expert-major scan" {
    const gpa = std.testing.allocator;
    const seq = 77;
    const used = 8;
    const n_experts = 100;
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    const selected = try gpa.alloc(usize, seq * used);
    defer gpa.free(selected);
    const weights = try gpa.alloc(f32, seq * used);
    defer gpa.free(weights);
    for (0..seq) |t| {
        // distinct experts per token, like a top-k
        var k: usize = 0;
        while (k < used) {
            const e = rnd.uintLessThan(usize, n_experts);
            var dup = false;
            for (selected[t * used ..][0..k]) |x| dup = dup or x == e;
            if (dup) continue;
            selected[t * used + k] = e;
            weights[t * used + k] = rnd.float(f32);
            k += 1;
        }
    }
    const cap = seq * used + n_experts * 31;
    const counts = try gpa.alloc(usize, n_experts);
    defer gpa.free(counts);
    const starts = try gpa.alloc(usize, n_experts);
    defer gpa.free(starts);
    for ([_]ExpertGemm{ .mmq, .gemv, .f16 }) |kind| {
        const mmq = kind == .mmq;
        const ids = try gpa.alloc(u32, cap);
        defer gpa.free(ids);
        const scales = try gpa.alloc(f32, cap);
        defer gpa.free(scales);
        const groups = try gpa.alloc(u32, cap / 8 + n_experts);
        defer gpa.free(groups);
        const slot_rows = try gpa.alloc(u32, seq * used);
        defer gpa.free(slot_rows);
        const got = packRoutes(kind, seq, selected, weights, used, n_experts, .{ .counts = counts, .starts = starts, .route_ids = ids, .route_scales = scales, .expert_ids = groups, .slot_rows = slot_rows });
        for (0..seq) |t| for (0..used) |k| {
            const r = slot_rows[t * used + k];
            try std.testing.expectEqual(@as(u32, @intCast(t)), ids[r]);
            try std.testing.expectEqual(weights[t * used + k], scales[r]);
        };

        // reference: scan experts in order, tokens ascending, tiling as we go
        const tile: usize = switch (kind) {
            .mmq => 32,
            .gemv => 8,
            .f16 => 128,
        };
        const ref_ids = try gpa.alloc(u32, cap);
        defer gpa.free(ref_ids);
        const ref_scales = try gpa.alloc(f32, cap);
        defer gpa.free(ref_scales);
        const ref_groups = try gpa.alloc(u32, cap / 8 + n_experts);
        defer gpa.free(ref_groups);
        var dst: usize = 0;
        var ng: usize = 0;
        for (0..n_experts) |expert| {
            var count: usize = 0;
            for (0..seq) |t| for (0..used) |slot| {
                const at = t * used + slot;
                if (selected[at] != expert) continue;
                ref_ids[dst] = @intCast(t);
                ref_scales[dst] = weights[at];
                dst += 1;
                count += 1;
                if (count == tile) {
                    ref_groups[ng] = switch (kind) {
                        .mmq => expertMmqGroup(expert, dst - tile),
                        .gemv => routeGroup(expert, tile, dst - tile),
                        .f16 => cuda.kernels.hgemmGroup(expert, tile, dst - tile),
                    };
                    ng += 1;
                    count = 0;
                }
            };
            if (count != 0) {
                const start = dst - count;
                if (mmq) while (count < tile) : (count += 1) {
                    ref_ids[dst] = 0;
                    ref_scales[dst] = 0;
                    dst += 1;
                };
                ref_groups[ng] = switch (kind) {
                    .mmq => expertMmqGroup(expert, start),
                    .gemv => routeGroup(expert, count, start),
                    .f16 => cuda.kernels.hgemmGroup(expert, count, start),
                };
                ng += 1;
            }
        }
        errdefer std.debug.print("kind={t} got rows={d} groups={d} ref rows={d} groups={d}\n", .{ kind, got.rows, got.groups, dst, ng });
        try std.testing.expectEqual(dst, got.rows);
        try std.testing.expectEqual(ng, got.groups);
        try std.testing.expectEqualSlices(u32, ref_ids[0..dst], ids[0..dst]);
        try std.testing.expectEqualSlices(f32, ref_scales[0..dst], scales[0..dst]);
        try std.testing.expectEqualSlices(u32, ref_groups[0..ng], groups[0..ng]);
    }
}

pub const CudaLM = struct {
    lm: *const k2.Model,
    be: *Backend,
    gpa: std.mem.Allocator,
    cfg: k2.Config,
    capacity: usize,
    initial_capacity: usize,
    max_capacity: usize,
    len: usize = 0,
    kv_dtype: kvmod.KvDtype,
    max_rows: usize,
    sin_off: usize,
    k_cache: []Growable,
    v_cache: []Growable,
    freqs_d: Buf,
    bufs: Bufs,
    cpu_scratch: k2.Scratch,
    route_logits: []f32,
    selected: []usize,
    selected_weights: []f32,
    route_pack: []u32,
    route_counts: []usize,
    route_starts: []usize,
    /// Per layer: its routed experts live on the host (CPU for decode and tiny
    /// batches, staged to the GPU for prefill). The migration unit of `split`.
    host_experts: []bool,
    stage: ?Stage = null,
    cpu_rows: usize,
    split: ?Split = null,
    boundary: ?bnd.Hook = null,
    io: ?std.Io = null,
    streaming_on: bool,
    hybrid_profile: HybridProfile,

    pub const prefill_batch = prefill_rows;

    pub const Split = struct {
        n_cpu: usize = 0,
        policy: CpuSplitPolicy = .tail,
        dynamic: bool = false,
        budget: u64 = 0,
        order: []usize = &.{},
        next: usize = 0,
    };

    pub fn init(gpa: std.mem.Allocator, be: *Backend, lm: *const k2.Model, cap: kvmod.Capacity) !CudaLM {
        const c = lm.cfg;
        const rows = prefillRows(cap.max);
        var self: CudaLM = undefined;
        self.lm = lm;
        self.be = be;
        self.gpa = gpa;
        self.cfg = c;
        self.capacity = cap.initial;
        self.initial_capacity = cap.initial;
        self.max_capacity = cap.max;
        self.len = 0;
        self.kv_dtype = cap.kv_dtype;
        self.max_rows = rows;
        self.sin_off = cap.max * (c.head_dim / 2);
        self.split = null;
        self.boundary = null;
        self.io = null;
        self.streaming_on = false;
        self.stage = null;
        self.cpu_rows = cpu_small_max;
        // The TC attention batches heads under this; the experts own the VRAM here.
        be.attn_scratch_budget = @min(be.attn_scratch_budget, 256 << 20);
        self.hybrid_profile = .{ .enabled = std.c.getenv("TP_K2_PROFILE_HYBRID") != null };

        var freqs = try ops.rope.rotateHalfFreqs(gpa, cap.max, c.head_dim, c.rope_theta);
        defer freqs.deinit(gpa);
        const fp = try gpa.alloc(f32, 2 * cap.max * (c.head_dim / 2));
        defer gpa.free(fp);
        @memcpy(fp[0..self.sin_off], freqs.cos);
        @memcpy(fp[self.sin_off..], freqs.sin);
        self.freqs_d = try be.tensorCreate(fp.len * 4);
        errdefer be.tensorDestroy(&self.freqs_d);
        try be.tensorUpload(self.freqs_d, std.mem.sliceAsBytes(fp));

        self.k_cache = try gpa.alloc(Growable, c.n_layers);
        errdefer gpa.free(self.k_cache);
        self.v_cache = try gpa.alloc(Growable, c.n_layers);
        errdefer gpa.free(self.v_cache);
        var nk: usize = 0;
        errdefer for (self.k_cache[0..nk]) |*buf| be.growableDestroy(buf);
        for (self.k_cache) |*buf| {
            buf.* = try be.growableCreate(cap.kv_dtype.sizeBytes(cap.initial * c.kvDim()), cap.kv_dtype.sizeBytes(cap.max * c.kvDim()));
            nk += 1;
        }
        var nv: usize = 0;
        errdefer for (self.v_cache[0..nv]) |*buf| be.growableDestroy(buf);
        for (self.v_cache) |*buf| {
            buf.* = try be.growableCreate(cap.kv_dtype.sizeBytes(cap.initial * c.kvDim()), cap.kv_dtype.sizeBytes(cap.max * c.kvDim()));
            nv += 1;
        }
        self.bufs = try Bufs.init(be, c, rows);
        errdefer self.bufs.deinit(be);
        self.cpu_scratch = try k2.Scratch.init(gpa, c, self.cpu_rows);
        errdefer self.cpu_scratch.deinit(gpa);

        const max_experts = @max(c.n_experts, c.n_value_experts);
        const max_used = @max(c.n_experts_used, c.n_value_experts_used);
        self.route_logits = try gpa.alloc(f32, rows * max_experts);
        errdefer gpa.free(self.route_logits);
        self.selected = try gpa.alloc(usize, rows * max_used);
        errdefer gpa.free(self.selected);
        self.selected_weights = try gpa.alloc(f32, rows * max_used);
        errdefer gpa.free(self.selected_weights);
        self.route_pack = try gpa.alloc(u32, routePackSlots(c, rows));
        errdefer gpa.free(self.route_pack);
        self.route_counts = try gpa.alloc(usize, max_experts);
        errdefer gpa.free(self.route_counts);
        self.route_starts = try gpa.alloc(usize, max_experts);
        errdefer gpa.free(self.route_starts);
        self.host_experts = try gpa.alloc(bool, c.n_layers);
        @memset(self.host_experts, false);
        return self;
    }

    pub fn deinit(self: *CudaLM) void {
        for (self.k_cache) |*buf| self.be.growableDestroy(buf);
        for (self.v_cache) |*buf| self.be.growableDestroy(buf);
        self.gpa.free(self.k_cache);
        self.gpa.free(self.v_cache);
        self.be.tensorDestroy(&self.freqs_d);
        self.bufs.deinit(self.be);
        self.cpu_scratch.deinit(self.gpa);
        self.gpa.free(self.route_logits);
        self.gpa.free(self.selected);
        self.gpa.free(self.selected_weights);
        self.gpa.free(self.route_pack);
        self.gpa.free(self.route_counts);
        self.gpa.free(self.route_starts);
        self.gpa.free(self.host_experts);
        if (self.split) |sp| self.gpa.free(sp.order);
        self.stageDeinit();
        self.* = undefined;
    }

    fn stageDeinit(self: *CudaLM) void {
        const st = &(self.stage orelse return);
        self.stageWait(st.layer);
        self.be.tensorDestroy(&st.gate);
        self.be.tensorDestroy(&st.up);
        self.be.tensorDestroy(&st.down);
        if (st.values) |*v| self.be.tensorDestroy(v);
        self.stage = null;
    }

    fn stagedLayer(self: *const CudaLM, l: usize) bool {
        return self.stage != null and self.host_experts[l] and self.lm.layers[l].experts != null;
    }

    fn nextStagedLayer(self: *const CudaLM, from: usize) ?usize {
        var l = from;
        while (l < self.cfg.n_layers) : (l += 1) if (self.stagedLayer(l)) return l;
        return null;
    }

    fn hostLayers(self: *const CudaLM) usize {
        var n: usize = 0;
        for (self.host_experts) |h| n += @intFromBool(h);
        return n;
    }

    /// Queue layer `l`'s expert groups into the staging buffers; the first DMA
    /// waits for `after` so the previous layer's reads of the same buffers finish.
    fn stageQueue(self: *CudaLM, l: usize, after: ?cuda.cu.CUevent) !void {
        const st = &self.stage.?;
        const layer = self.lm.layers[l];
        const e = layer.experts.?;
        st.layer = l;
        st.after = after;
        st.pending[0] = try self.be.uploadInto(st.gate, groupBytes(e.gate), after);
        st.pending[1] = try self.be.uploadInto(st.up, groupBytes(e.up), null);
        st.pending[2] = try self.be.uploadInto(st.down, groupBytes(e.down), null);
        if (layer.values) |v| st.pending[3] = try self.be.uploadInto(st.values.?, groupBytes(v.weights), null);
    }

    fn stageWait(self: *CudaLM, l: usize) void {
        const st = &self.stage.?;
        std.debug.assert(st.layer == l);
        for (&st.pending) |*p| if (p.*) |u| {
            self.be.uploadIntoWait(u);
            p.* = null;
        };
        if (st.after) |ev| {
            self.be.eventDestroy(ev);
            st.after = null;
        }
    }

    pub fn cached(self: *const CudaLM) usize {
        return self.len;
    }

    pub fn remaining(self: *const CudaLM) usize {
        return self.capacity - self.len;
    }

    pub fn capacityMax(self: *const CudaLM) usize {
        return self.max_capacity;
    }

    pub fn vocab(self: *const CudaLM) usize {
        return self.cfg.vocab;
    }

    pub fn vramUsed(self: *const CudaLM) u64 {
        return self.be.deviceUsed();
    }

    /// Grow the device KV (all layers attend on the GPU). Under a dynamic split,
    /// routed-expert layers migrate to the host first until the growth fits, and
    /// a physical OOM during the grow migrates one more and retries.
    pub fn ensureCapacity(self: *CudaLM, min_rows: usize) !void {
        if (min_rows <= self.capacity) return;
        if (min_rows > self.max_capacity) return error.ContextFull;
        const target = kvmod.growTarget(self.capacity, min_rows, self.max_capacity);
        const bytes = self.kv_dtype.sizeBytes(target * self.cfg.kvDim());
        if (self.split) |*sp| if (sp.dynamic) {
            const add = self.kv_dtype.sizeBytes((target - self.capacity) * self.cfg.kvDim());
            while (true) {
                const need = 2 * self.cfg.n_layers * add + (64 << 20);
                const free = @min(sp.budget -| self.be.deviceUsed(), self.be.headroom());
                if (free >= need) break;
                if (!(try residency.migrateNext(self))) break;
            }
        };
        grow: while (true) {
            for (self.k_cache, self.v_cache) |*kb, *vb| {
                for ([2]*Growable{ kb, vb }) |b| self.be.growableEnsure(b, bytes) catch |err| switch (err) {
                    error.DeviceOutOfMemory => {
                        if (self.split != null and try residency.migrateNext(self)) continue :grow;
                        return error.ContextFull;
                    },
                    else => return error.ContextFull,
                };
            }
            break;
        }
        self.capacity = target;
    }

    fn kvDeviceBytes(self: *const CudaLM) u64 {
        return 2 * self.cfg.n_layers * self.kv_dtype.sizeBytes(self.capacity * self.cfg.kvDim());
    }

    pub fn step(self: *CudaLM, io: std.Io, ids: []const u32, logits: []f32) !void {
        self.useIo(io);
        var off: usize = 0;
        while (off < ids.len) {
            const n = @min(self.max_rows, ids.len - off);
            try self.stepChunk(ids[off..][0..n], if (off + n == ids.len) logits else null);
            off += n;
        }
    }

    pub fn prefill(self: *CudaLM, ids: []const u32) !void {
        var off: usize = 0;
        while (off < ids.len) {
            try bnd.check(self);
            const n = @min(self.max_rows, ids.len - off);
            try self.stepChunk(ids[off..][0..n], null);
            off += n;
        }
    }

    fn stepChunk(self: *CudaLM, ids: []const u32, logits: ?[]f32) !void {
        const c = self.cfg;
        const b = &self.bufs;
        const seq = ids.len;
        const first_prefill = seq > 1 and self.len == 0;
        if (first_prefill) self.hybrid_profile.reset();
        const profile_start = self.hybrid_profile.tic();
        if (seq > 1 and self.len == 0) self.dumpPlacement("prefill-start");
        const x = try self.gpa.alloc(f32, seq * c.hidden);
        defer self.gpa.free(x);
        try qwen3.embedTokens(self.lm.embed, ids, x);
        try self.be.tensorUpload(offsetBuf(b.x, 0, x.len * 4), std.mem.sliceAsBytes(x));
        if (seq > cpu_small_max and self.stage != null)
            if (self.nextStagedLayer(0)) |l| try self.stageQueue(l, null);

        for (self.lm.layers, 0..) |layer_weights, l| {
            try self.forwardLayer(layer_weights, l, seq);
        }
        if (seq > 1 and self.len == 0) self.dumpPlacement("prefill-end");
        if (first_prefill and self.hybrid_profile.enabled) {
            const total_ns = cuda.context.monoNs() -| profile_start;
            std.debug.print("[k2 profile] total={d:.1}ms route={d:.1}ms transfer={d:.1}ms values={d:.1}ms ffn={d:.1}ms\n", .{
                @as(f64, @floatFromInt(total_ns)) / 1e6,
                @as(f64, @floatFromInt(self.hybrid_profile.route_ns)) / 1e6,
                @as(f64, @floatFromInt(self.hybrid_profile.transfer_ns)) / 1e6,
                @as(f64, @floatFromInt(self.hybrid_profile.values_ns)) / 1e6,
                @as(f64, @floatFromInt(self.hybrid_profile.ffn_ns)) / 1e6,
            });
        }
        try self.be.groupRmsNorm(
            offsetBuf(b.x, (seq - 1) * c.hidden * 4, c.hidden * 4),
            b.tmp,
            try nbuf(self.be, self.lm.final_norm),
            1,
            c.hidden,
            c.norm_groups,
            c.rms_eps,
        );
        try self.linear(b.logits, b.tmp, self.lm.head, c.vocab, c.hidden, 1);
        if (logits) |out| try self.be.tensorDownload(offsetBuf(b.logits, 0, c.vocab * 4), std.mem.sliceAsBytes(out[0..c.vocab]));
        self.len += seq;
    }

    fn dumpPlacement(self: *const CudaLM, label: []const u8) void {
        if (std.c.getenv("TP_K2_DUMP_PLACEMENT") == null) return;
        std.debug.print("[k2] {s}: host_layers={d} stage={d}MiB weights={d}MiB kv={d}MiB used={d}MiB free={d}MiB\n", .{
            label,
            self.hostLayers(),
            (if (self.stage) |st| st.gate.size + st.up.size + st.down.size + (if (st.values) |v| v.size else 0) else 0) >> 20,
            self.be.pinnedWeightBytes() >> 20,
            self.kvDeviceBytes() >> 20,
            self.be.deviceUsed() >> 20,
            self.be.ctx.memGetInfo().free >> 20,
        });
    }

    fn useIo(self: *CudaLM, io: std.Io) void {
        self.io = io;
        if (self.streaming_on) return;
        self.be.enableAsyncStreaming(io);
        self.streaming_on = true;
    }

    fn forwardLayer(self: *CudaLM, layer: k2.Layer, l: usize, seq: usize) !void {
        const c = self.cfg;
        const b = &self.bufs;
        const pos0 = self.len;
        try self.be.groupRmsNorm(b.x, b.normed, try nbuf(self.be, layer.attn_norm), seq, c.hidden, c.norm_groups, c.rms_eps);
        try self.linear(b.q, b.normed, layer.q, c.qDim(), c.hidden, seq);
        try self.linear(b.k, b.normed, layer.k, c.kvDim(), c.hidden, seq);
        const staged = seq > cpu_small_max and self.stagedLayer(l);
        if (staged) self.stageWait(l);
        if (layer.values) |values| {
            if (staged)
                try self.routedValues(values, b.normed, seq, b.v, self.stage.?.values.?)
            else if (self.host_experts[l])
                try self.cpuRoutedValues(values, seq, b.v)
            else
                try self.routedValues(values, b.normed, seq, b.v, null);
        } else try self.linear(b.v, b.normed, layer.v.?, c.kvDim(), c.hidden, seq);
        try self.be.ropeHalf(b.q, self.freqs_d, seq, c.n_heads, c.head_dim / 2, self.sin_off, pos0);
        try self.be.ropeHalf(b.k, self.freqs_d, seq, c.n_kv_heads, c.head_dim / 2, self.sin_off, pos0);
        try self.storeKv(self.k_cache[l].buf, pos0 * c.kvDim(), b.k, seq * c.kvDim());
        try self.storeKv(self.v_cache[l].buf, pos0 * c.kvDim(), b.v, seq * c.kvDim());
        try self.attention(l, seq, pos0);
        if (layer.attn_gate) |gate| {
            try self.linear(b.gate, b.normed, gate, c.qDim(), c.hidden, seq);
            try self.be.opSoftplusGate(b.attn, b.gate, seq * c.qDim());
        }
        try self.linear(b.tmp, b.attn, layer.o, c.hidden, c.qDim(), seq);
        try self.be.opAdd(b.x, b.tmp, seq * c.hidden);

        try self.be.groupRmsNorm(b.x, b.normed, try nbuf(self.be, layer.ffn_norm), seq, c.hidden, c.norm_groups, c.rms_eps);
        if (layer.experts) |experts| {
            if (staged) {
                const st = self.stage.?;
                try self.routedFfn(experts, b.normed, seq, b.tmp, .{ st.gate, st.up, st.down });
                if (self.nextStagedLayer(l + 1)) |next| try self.stageQueue(next, try self.be.recordCompute());
            } else if (self.host_experts[l])
                try self.cpuRoutedFfn(experts, seq, b.tmp)
            else
                try self.routedFfn(experts, b.normed, seq, b.tmp, null);
        } else {
            try self.linear(b.expert_a, b.normed, layer.dense_gate.?, c.intermediate, c.hidden, seq);
            try self.linear(b.expert_b, b.normed, layer.dense_up.?, c.intermediate, c.hidden, seq);
            try self.be.siluMul(b.expert_a, b.expert_b, seq * c.intermediate);
            try self.linear(b.tmp, b.expert_a, layer.dense_down.?, c.hidden, c.intermediate, seq);
        }
        try self.be.opAdd(b.x, b.tmp, seq * c.hidden);
    }

    fn attention(self: *CudaLM, l: usize, seq: usize, pos0: usize) !void {
        const c = self.cfg;
        const b = &self.bufs;
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.head_dim)));
        if (self.kv_dtype == .f32 and seq >= tc_attn_min) {
            // query tiles sized so a head's f16 score plane stays under attn_plane_max
            const kpad = std.mem.alignForward(usize, pos0 + seq, 128);
            const tile = @max(128, (attn_plane_max / (kpad * 2)) & ~@as(usize, 127));
            var t0: usize = 0;
            while (t0 < seq) {
                const n = @min(tile, seq - t0);
                try self.be.opAttnTCCausal(
                    offsetBuf(b.q, t0 * c.qDim() * 4, n * c.qDim() * 4),
                    self.k_cache[l].buf,
                    self.v_cache[l].buf,
                    offsetBuf(b.attn, t0 * c.qDim() * 4, n * c.qDim() * 4),
                    n,
                    pos0 + t0 + n,
                    c.n_heads,
                    c.n_kv_heads,
                    c.head_dim,
                    scale,
                    pos0 + t0,
                );
                t0 += n;
            }
            return;
        }
        try self.be.opAttnDecode(b.q, self.k_cache[l].buf, self.v_cache[l].buf, b.attn, b.attn_scratch, pos0 + 1, seq, c.n_heads, c.n_kv_heads, c.head_dim, attnSplits(seq), scale, 0, 0, false, kvFmt(self.kv_dtype));
    }

    fn storeKv(self: *CudaLM, dst: Buf, dst_off: usize, src: Buf, n: usize) !void {
        switch (self.kv_dtype) {
            .f32 => try self.be.tensorCopy(dst, dst_off * 4, src, 0, n * 4),
            .f16 => try self.be.opStoreKvF16(dst, dst_off, src, 0, n),
            .q8_0 => try self.be.opStoreKvQ8(dst, dst_off, src, 0, n),
        }
    }

    fn cpuRoutedValues(self: *CudaLM, values: k2.Values, seq: usize, out: Buf) !void {
        const c = self.cfg;
        std.debug.assert(seq <= self.cpu_rows);
        const io = self.io orelse return error.SplitIoUnset;
        const x = self.cpu_scratch.normed[0 .. seq * c.hidden];
        const y = self.cpu_scratch.v[0 .. seq * c.kvDim()];
        var started = self.hybrid_profile.tic();
        try self.linear(self.bufs.route_logits, self.bufs.normed, values.router, c.n_value_experts, c.hidden, seq);
        try self.selectRoutes(self.bufs.route_logits, values.bias, seq, c.n_value_experts, c.n_value_experts_used);
        self.hybrid_profile.toc(started, &self.hybrid_profile.route_ns);
        started = self.hybrid_profile.tic();
        try self.be.tensorDownload(offsetBuf(self.bufs.normed, 0, x.len * 4), std.mem.sliceAsBytes(x));
        self.hybrid_profile.toc(started, &self.hybrid_profile.transfer_ns);
        started = self.hybrid_profile.tic();
        try k2.routedValuesSelected(
            io,
            self.gpa,
            c,
            values,
            x,
            seq,
            y,
            &self.cpu_scratch,
            self.selected[0 .. seq * c.n_value_experts_used],
            self.selected_weights[0 .. seq * c.n_value_experts_used],
        );
        self.hybrid_profile.toc(started, &self.hybrid_profile.values_ns);
        started = self.hybrid_profile.tic();
        try self.be.tensorUpload(offsetBuf(out, 0, y.len * 4), std.mem.sliceAsBytes(y));
        self.hybrid_profile.toc(started, &self.hybrid_profile.transfer_ns);
    }

    fn cpuRoutedFfn(self: *CudaLM, experts: k2.Experts, seq: usize, out: Buf) !void {
        const c = self.cfg;
        std.debug.assert(seq <= self.cpu_rows);
        const io = self.io orelse return error.SplitIoUnset;
        const x = self.cpu_scratch.normed[0 .. seq * c.hidden];
        const y = self.cpu_scratch.tmp[0 .. seq * c.hidden];
        var started = self.hybrid_profile.tic();
        try self.linear(self.bufs.route_logits, self.bufs.normed, experts.router, c.n_experts, c.hidden, seq);
        try self.selectRoutes(self.bufs.route_logits, experts.bias, seq, c.n_experts, c.n_experts_used);
        self.hybrid_profile.toc(started, &self.hybrid_profile.route_ns);
        started = self.hybrid_profile.tic();
        try self.be.tensorDownload(offsetBuf(self.bufs.normed, 0, x.len * 4), std.mem.sliceAsBytes(x));
        self.hybrid_profile.toc(started, &self.hybrid_profile.transfer_ns);
        started = self.hybrid_profile.tic();
        try k2.routedFfnSparseSelected(
            io,
            self.gpa,
            c,
            experts,
            x,
            seq,
            y,
            &self.cpu_scratch,
            self.selected[0 .. seq * c.n_experts_used],
            self.selected_weights[0 .. seq * c.n_experts_used],
        );
        self.hybrid_profile.toc(started, &self.hybrid_profile.ffn_ns);
        started = self.hybrid_profile.tic();
        try self.be.tensorUpload(offsetBuf(out, 0, y.len * 4), std.mem.sliceAsBytes(y));
        self.hybrid_profile.toc(started, &self.hybrid_profile.transfer_ns);
        if (experts.shared_gate) |gate| {
            try self.linear(self.bufs.expert_a, self.bufs.normed, gate, c.shared_intermediate, c.hidden, seq);
            try self.linear(self.bufs.expert_b, self.bufs.normed, experts.shared_up.?, c.shared_intermediate, c.hidden, seq);
            try self.be.siluMul(self.bufs.expert_a, self.bufs.expert_b, seq * c.shared_intermediate);
            try self.linear(self.bufs.expert_out, self.bufs.expert_a, experts.shared_down.?, c.hidden, c.shared_intermediate, seq);
            try self.be.opAdd(out, self.bufs.expert_out, seq * c.hidden);
        }
    }

    fn selectRoutes(self: *CudaLM, logits_d: Buf, bias: []const f32, seq: usize, n_experts: usize, n_used: usize) !void {
        const logits = self.route_logits[0 .. seq * n_experts];
        try self.be.tensorDownload(offsetBuf(logits_d, 0, logits.len * 4), std.mem.sliceAsBytes(logits));
        for (0..seq) |t| {
            const row = logits[t * n_experts ..][0..n_experts];
            switch (self.cfg.gating) {
                .sigmoid => ops.act.sigmoid(row),
                .softmax => {
                    var mx = -std.math.inf(f32);
                    for (row) |v| mx = @max(mx, v);
                    var sum: f32 = 0;
                    for (row) |*v| {
                        v.* = @exp(v.* - mx);
                        sum += v.*;
                    }
                    for (row) |*v| v.* /= sum;
                },
            }
            const ids = self.selected[t * n_used ..][0..n_used];
            const weights = self.selected_weights[t * n_used ..][0..n_used];
            @memset(weights, -std.math.inf(f32));
            for (row, 0..) |prob, expert| {
                const score = prob + bias[expert];
                var at: usize = 0;
                while (at < n_used and score <= weights[at]) : (at += 1) {}
                if (at == n_used) continue;
                var j = n_used - 1;
                while (j > at) : (j -= 1) {
                    weights[j] = weights[j - 1];
                    ids[j] = ids[j - 1];
                }
                weights[at] = score;
                ids[at] = expert;
            }
            var sum: f32 = 0;
            for (ids, weights) |expert, *weight| {
                weight.* = row[expert];
                sum += weight.*;
            }
            if (self.cfg.normalize_weights) {
                for (weights) |*weight| weight.* /= @max(sum, 6.103515625e-5);
            }
            if (self.cfg.expert_scale != 1) {
                for (weights) |*weight| weight.* *= self.cfg.expert_scale;
            }
        }
    }

    /// One expert's rows for the per-expert fallback: ids then scales, one upload.
    fn gatherForExpert(self: *CudaLM, x: Buf, seq: usize, selected: []const usize, weights: []const f32, used: usize, expert: usize) !GroupedRoutes {
        const cap = seq * used;
        const ids = self.route_pack[0..cap];
        const scales = std.mem.bytesAsSlice(f32, std.mem.sliceAsBytes(self.route_pack[cap .. 2 * cap]));
        var n: usize = 0;
        for (0..seq) |t| for (0..used) |slot| {
            const at = t * used + slot;
            if (selected[at] != expert) continue;
            ids[n] = @intCast(t);
            scales[n] = weights[at];
            n += 1;
        };
        if (n == 0) return .{ .rows = 0, .groups = 0 };
        try self.be.tensorUpload(offsetBuf(self.bufs.route_pack, 0, 2 * cap * 4), std.mem.sliceAsBytes(self.route_pack[0 .. 2 * cap]));
        const routes: GroupedRoutes = .{ .rows = n, .groups = 0, .ids = offsetBuf(self.bufs.route_pack, 0, n * 4), .scales = offsetBuf(self.bufs.route_pack, cap * 4, n * 4) };
        try self.be.opGatherRows(self.bufs.expert_in, x, routes.ids, n, self.cfg.hidden);
        return routes;
    }

    /// One contiguous, uniformly typed expert group (a GGUF 3-D tensor).
    fn groupedWeights(weights: []const Weight, rows: usize, cols: usize) bool {
        if (weights.len == 0) return false;
        const first = weights[0];
        if (first.rows != rows or first.cols != cols) return false;
        for (weights, 0..) |weight, i| {
            if (weight.dtype != first.dtype or weight.rows != rows or weight.cols != cols or weight.scale != first.scale) return false;
            if (@intFromPtr(weight.bytes.ptr) != @intFromPtr(first.bytes.ptr) + i * first.bytes.len) return false;
        }
        return true;
    }

    fn expertGemm(self: *const CudaLM) ExpertGemm {
        const raw = std.c.getenv("TP_K2_EXPERT_GEMM") orelse return .f16;
        const name = std.mem.span(raw);
        if (!self.be.routedExpertGemvEnabled()) return .f16;
        if (std.mem.eql(u8, name, "mmq")) return .mmq;
        if (std.mem.eql(u8, name, "gemv")) return .gemv;
        return .f16;
    }

    /// The packed q6_k kernels want q6_k; anything else takes the f16 path. Few
    /// packed rows (decode, tiny batches) take the grouped q8 GEMV: it reads each
    /// selected expert once, where the f16 path would dequantize the whole group.
    fn expertGemmFor(self: *const CudaLM, weights: []const Weight, packed_rows: usize) ExpertGemm {
        if (weights[0].dtype != .q6_k) return .f16;
        if (packed_rows <= gemv_rows_max and self.be.routedExpertGemvEnabled()) return .gemv;
        return self.expertGemm();
    }

    /// Pack the batch's routes as [ids | scales | slot_rows | groups] in one host
    /// array, upload it once, and gather the expert input rows.
    fn prepareGroupedRoutes(self: *CudaLM, kind: ExpertGemm, x: Buf, seq: usize, selected: []const usize, weights: []const f32, used: usize, n_experts: usize) !GroupedRoutes {
        const S = seq * used;
        const R = S + (if (kind == .mmq) n_experts * 31 else 0);
        const G = R / 8 + n_experts;
        const pack = self.route_pack[0 .. 2 * R + S + G];
        const pk = packRoutes(kind, seq, selected, weights, used, n_experts, .{
            .counts = self.route_counts[0..n_experts],
            .starts = self.route_starts[0..n_experts],
            .route_ids = pack[0..R],
            .route_scales = std.mem.bytesAsSlice(f32, std.mem.sliceAsBytes(pack[R .. 2 * R])),
            .slot_rows = pack[2 * R ..][0..S],
            .expert_ids = pack[2 * R + S ..][0..G],
        });
        const n = 2 * R + S + pk.groups;
        const d = self.bufs.route_pack;
        try self.be.tensorUpload(offsetBuf(d, 0, n * 4), std.mem.sliceAsBytes(pack[0..n]));
        const routes: GroupedRoutes = .{
            .rows = pk.rows,
            .groups = pk.groups,
            .ids = offsetBuf(d, 0, pk.rows * 4),
            .scales = offsetBuf(d, R * 4, pk.rows * 4),
            .slots = offsetBuf(d, 2 * R * 4, S * 4),
            .groups_d = offsetBuf(d, (2 * R + S) * 4, pk.groups * 4),
        };
        try self.be.opGatherRows(self.bufs.expert_in, x, routes.ids, routes.rows, self.cfg.hidden);
        return routes;
    }

    /// y[routes.rows][rows] = per-expert W_e @ x over the packed rows. `dev` is a
    /// staging copy of the whole expert group; null reads the cached weights.
    fn groupedExpertLinear(self: *CudaLM, kind: ExpertGemm, y: Buf, x: Buf, weights: []const Weight, dev: ?Buf, rows: usize, cols: usize, routes: GroupedRoutes) !void {
        switch (kind) {
            .f16 => try self.expertLinearF16(y, x, weights, dev, rows, cols, routes),
            .mmq => if (dev) |w|
                try self.be.opMmqQ6ExpertsDev(y, x, w, routes.groups_d, rows, cols, routes.rows, routes.groups)
            else
                try self.be.opMmqQ6Experts(y, x, weights[0].bytes, weights.len, routes.groups_d, rows, cols, routes.rows, routes.groups),
            .gemv => {
                try self.be.opGemvQuantizeX(x, routes.rows * cols);
                if (dev) |w|
                    try self.be.opGemvQ6ExpertsDev(y, w, routes.groups_d, weights[0].scale, rows, cols, routes.rows, routes.groups)
                else
                    try self.be.opGemvQ6Experts(y, weights[0].bytes, weights.len, routes.groups_d, weights[0].scale, rows, cols, routes.rows, routes.groups);
            },
        }
    }

    /// Dequantize the whole group to f16 once, convert the packed activations
    /// once (with the 128 readable rows of slack the tiles run into), then one
    /// grouped GEMM launch over the tile table `prepareGroupedRoutes` packed.
    fn expertLinearF16(self: *CudaLM, y: Buf, x: Buf, weights: []const Weight, dev: ?Buf, rows: usize, cols: usize, routes: GroupedRoutes) !void {
        const per = rows * cols;
        const w_db = dev orelse try self.be.residentWeight(groupBytes(weights));
        const w16 = try self.be.opDequantF16(weights[0].dtype, w_db, per * weights.len);
        const a_rows = std.mem.alignForward(usize, routes.rows, 128) + 128;
        const a16 = try self.be.opActF16(x, routes.rows, a_rows, cols);
        try self.be.opGemmF16Experts(y, w16, a16, routes.groups_d, routes.groups, rows, cols);
    }

    fn isSelected(selected: []const usize, expert: usize) bool {
        for (selected) |id| if (id == expert) return true;
        return false;
    }

    fn prefetchValueExpert(self: *CudaLM, values: k2.Values, selected: []const usize, first: usize) void {
        for (first..values.weights.len) |expert| {
            if (!isSelected(selected, expert)) continue;
            self.be.prefetchWeight(values.weights[expert].bytes);
            return;
        }
    }

    fn prefetchFfnExpert(self: *CudaLM, experts: k2.Experts, selected: []const usize, first: usize) void {
        for (first..experts.gate.len) |expert| {
            if (!isSelected(selected, expert)) continue;
            self.be.prefetchWeight(experts.gate[expert].bytes);
            self.be.prefetchWeight(experts.up[expert].bytes);
            self.be.prefetchWeight(experts.down[expert].bytes);
            return;
        }
    }

    fn routedValues(self: *CudaLM, values: k2.Values, x: Buf, seq: usize, out: Buf, dev: ?Buf) !void {
        const c = self.cfg;
        try self.linear(self.bufs.route_logits, x, values.router, c.n_value_experts, c.hidden, seq);
        const selected = self.selected[0 .. seq * c.n_value_experts_used];
        const weights = self.selected_weights[0 .. seq * c.n_value_experts_used];
        try self.selectRoutes(self.bufs.route_logits, values.bias, seq, c.n_value_experts, c.n_value_experts_used);
        const used = c.n_value_experts_used;
        const kind = self.expertGemmFor(values.weights, seq * used);
        if (dev != null or (groupedWeights(values.weights, c.kvDim(), c.hidden) and (seq > 1 or kind == .gemv))) {
            var t0: usize = 0;
            while (t0 < seq) : (t0 += routed_chunk) {
                const n = @min(routed_chunk, seq - t0);
                const xs = offsetBuf(x, t0 * c.hidden * 4, n * c.hidden * 4);
                const routes = try self.prepareGroupedRoutes(kind, xs, n, selected[t0 * used ..][0 .. n * used], weights[t0 * used ..][0 .. n * used], used, c.n_value_experts);
                try self.groupedExpertLinear(kind, self.bufs.expert_a, self.bufs.expert_in, values.weights, dev, c.kvDim(), c.hidden, routes);
                try self.be.opSilu(self.bufs.expert_a, routes.rows * c.kvDim());
                try self.be.opMoeCombine(offsetBuf(out, t0 * c.kvDim() * 4, n * c.kvDim() * 4), self.bufs.expert_a, routes.slots, routes.scales, n, used, c.kvDim());
            }
            return;
        }
        try self.be.tensorZero(out, seq * c.kvDim() * 4);
        self.prefetchValueExpert(values, selected, 0);
        for (values.weights, 0..) |weight, expert| {
            const routes = try self.gatherForExpert(x, seq, selected, weights, used, expert);
            if (routes.rows == 0) continue;
            self.prefetchValueExpert(values, selected, expert + 1);
            try self.linear(self.bufs.expert_a, self.bufs.expert_in, weight, c.kvDim(), c.hidden, routes.rows);
            try self.be.opSilu(self.bufs.expert_a, routes.rows * c.kvDim());
            try self.be.opScatterAddRows(out, self.bufs.expert_a, routes.ids, routes.scales, routes.rows, c.kvDim());
        }
    }

    fn routedFfn(self: *CudaLM, experts: k2.Experts, x: Buf, seq: usize, out: Buf, dev: ?[3]Buf) !void {
        const c = self.cfg;
        try self.linear(self.bufs.route_logits, x, experts.router, c.n_experts, c.hidden, seq);
        const selected = self.selected[0 .. seq * c.n_experts_used];
        const weights = self.selected_weights[0 .. seq * c.n_experts_used];
        try self.selectRoutes(self.bufs.route_logits, experts.bias, seq, c.n_experts, c.n_experts_used);
        const used = c.n_experts_used;
        const kind = self.expertGemmFor(experts.gate, seq * used);
        if (dev != null or ((seq > 1 or kind == .gemv) and
            groupedWeights(experts.gate, c.expert_intermediate, c.hidden) and
            groupedWeights(experts.up, c.expert_intermediate, c.hidden) and
            groupedWeights(experts.down, c.hidden, c.expert_intermediate)))
        {
            var t0: usize = 0;
            while (t0 < seq) : (t0 += routed_chunk) {
                const n = @min(routed_chunk, seq - t0);
                const xs = offsetBuf(x, t0 * c.hidden * 4, n * c.hidden * 4);
                const routes = try self.prepareGroupedRoutes(kind, xs, n, selected[t0 * used ..][0 .. n * used], weights[t0 * used ..][0 .. n * used], used, c.n_experts);
                try self.groupedExpertLinear(kind, self.bufs.expert_a, self.bufs.expert_in, experts.gate, if (dev) |d| d[0] else null, c.expert_intermediate, c.hidden, routes);
                try self.groupedExpertLinear(kind, self.bufs.expert_b, self.bufs.expert_in, experts.up, if (dev) |d| d[1] else null, c.expert_intermediate, c.hidden, routes);
                try self.be.siluMul(self.bufs.expert_a, self.bufs.expert_b, routes.rows * c.expert_intermediate);
                try self.groupedExpertLinear(kind, self.bufs.expert_out, self.bufs.expert_a, experts.down, if (dev) |d| d[2] else null, c.hidden, c.expert_intermediate, routes);
                try self.be.opMoeCombine(offsetBuf(out, t0 * c.hidden * 4, n * c.hidden * 4), self.bufs.expert_out, routes.slots, routes.scales, n, used, c.hidden);
            }
        } else {
            try self.be.tensorZero(out, seq * c.hidden * 4);
            self.prefetchFfnExpert(experts, selected, 0);
            for (experts.gate, experts.up, experts.down, 0..) |gate, up, down, expert| {
                const routes = try self.gatherForExpert(x, seq, selected, weights, used, expert);
                if (routes.rows == 0) continue;
                self.prefetchFfnExpert(experts, selected, expert + 1);
                try self.linear(self.bufs.expert_a, self.bufs.expert_in, gate, c.expert_intermediate, c.hidden, routes.rows);
                try self.linear(self.bufs.expert_b, self.bufs.expert_in, up, c.expert_intermediate, c.hidden, routes.rows);
                try self.be.siluMul(self.bufs.expert_a, self.bufs.expert_b, routes.rows * c.expert_intermediate);
                try self.linear(self.bufs.expert_out, self.bufs.expert_a, down, c.hidden, c.expert_intermediate, routes.rows);
                try self.be.opScatterAddRows(out, self.bufs.expert_out, routes.ids, routes.scales, routes.rows, c.hidden);
            }
        }
        if (experts.shared_gate) |gate| {
            try self.linear(self.bufs.expert_a, x, gate, c.shared_intermediate, c.hidden, seq);
            try self.linear(self.bufs.expert_b, x, experts.shared_up.?, c.shared_intermediate, c.hidden, seq);
            try self.be.siluMul(self.bufs.expert_a, self.bufs.expert_b, seq * c.shared_intermediate);
            try self.linear(self.bufs.expert_out, self.bufs.expert_a, experts.shared_down.?, c.hidden, c.shared_intermediate, seq);
            try self.be.opAdd(out, self.bufs.expert_out, seq * c.hidden);
        }
    }

    fn linear(self: *CudaLM, y: Buf, x: Buf, weight: Weight, rows_out: usize, cols: usize, seq: usize) !void {
        if (weight.dtype.isBlockQuant()) {
            if (seq == 1) {
                try self.be.opGemvQuant(weight.dtype, y, x, weight.bytes, weight.scale, rows_out, cols);
            } else if (Backend.quantQ8NSupported(weight.dtype) and
                (seq <= grouped_gemv_max or rows_out % 128 != 0) and
                cols % 256 == 0 and rows_out % 8 == 0)
            {
                try self.be.opGemvQuantizeX(x, seq * cols);
                var off: usize = 0;
                while (off < seq) : (off += 8) {
                    const n: usize = @min(8, seq - off);
                    try self.be.opGemvQuantQ8N(
                        weight.dtype,
                        offsetBuf(y, off * rows_out * 4, n * rows_out * 4),
                        weight.bytes,
                        weight.scale,
                        rows_out,
                        cols,
                        n,
                        off,
                        seq,
                    );
                }
            } else if (weight.dtype == .q6_k and seq >= Backend.mmq_pipe_tile and
                Backend.mmqPipeSupported(weight.dtype, rows_out, cols) and
                std.c.getenv("TP_K2_MMQ6") != null)
            {
                try self.be.opMatmulQuantMmqPipe(weight.dtype, y, x, seq, weight.bytes, rows_out, cols);
            } else {
                try self.be.opMatmulQuant(weight.dtype, y, x, seq, weight.bytes, rows_out, cols);
            }
            return;
        }
        switch (weight.dtype) {
            .f32 => if (weight.scale == 1)
                self.be.opMatmulF32Lt(y, x, seq, weight.bytes, rows_out, cols, null) catch |err| switch (err) {
                    error.UnsupportedKernelArm => try self.be.opMatmul(y, 0, x, 0, seq, weight.bytes, false, rows_out, cols, weight.scale, null),
                    else => return err,
                }
            else
                try self.be.opMatmul(y, 0, x, 0, seq, weight.bytes, false, rows_out, cols, weight.scale, null),
            .bf16 => if (seq == 1)
                try self.be.opGemvBf16(y, x, weight.bytes, weight.scale, rows_out, cols)
            else
                try self.be.opMatmulBf16(y, x, seq, weight.bytes, rows_out, cols, null, false, false),
            .f16 => try self.be.opMatmulF16(y, x, seq, weight.bytes, rows_out, cols, null, false, false),
            else => return error.UnsupportedDType,
        }
    }

    pub fn stepArgmax(self: *CudaLM, io: std.Io, ids: []const u32) !u32 {
        return self.stepArgmaxPen(io, ids, &.{}, .{});
    }

    pub fn stepArgmaxPen(self: *CudaLM, io: std.Io, ids: []const u32, pen: []const sample.PenaltyEntry, params: sample.Params) !u32 {
        self.useIo(io);
        var off: usize = 0;
        while (off < ids.len) {
            const n = @min(self.max_rows, ids.len - off);
            try self.stepChunk(ids[off..][0..n], null);
            off += n;
        }
        const logits = try self.gpa.alloc(f32, self.cfg.vocab);
        defer self.gpa.free(logits);
        try self.be.tensorDownload(self.bufs.logits, std.mem.sliceAsBytes(logits));
        for (pen) |entry| logits[entry.id] = sample.penalizeLogit(logits[entry.id], entry.count, params);
        var id: usize = 0;
        for (logits[1..], 1..) |value, i| if (value > logits[id]) {
            id = i;
        };
        return @intCast(id);
    }

    pub fn maxSelect(self: *const CudaLM) usize {
        _ = self;
        return cuda.backend.topk_lanes * cuda.backend.topk_m;
    }

    pub fn stepSelect(self: *CudaLM, io: std.Io, ids: []const u32, out_id: []u32, out_logit: []f32) !usize {
        return self.stepSelectPen(io, ids, &.{}, .{}, out_id, out_logit);
    }

    pub fn stepSelectPen(self: *CudaLM, io: std.Io, ids: []const u32, pen: []const sample.PenaltyEntry, params: sample.Params, out_id: []u32, out_logit: []f32) !usize {
        self.useIo(io);
        var off: usize = 0;
        while (off < ids.len) {
            const n = @min(self.max_rows, ids.len - off);
            try self.stepChunk(ids[off..][0..n], null);
            off += n;
        }
        try self.be.opPenalize(self.bufs.logits, pen, params);
        const count = try self.be.opTopK(self.bufs.logits, self.cfg.vocab, &self.bufs.topk_v, &self.bufs.topk_i);
        try self.be.tensorDownload(self.bufs.topk_v, std.mem.sliceAsBytes(out_logit[0..count]));
        const idx = try self.gpa.alloc(f32, count);
        defer self.gpa.free(idx);
        try self.be.tensorDownload(self.bufs.topk_i, std.mem.sliceAsBytes(idx));
        for (out_id[0..count], idx) |*dst, value| dst.* = @intFromFloat(value);
        return count;
    }

    pub fn truncate(self: *CudaLM, new_len: usize) void {
        std.debug.assert(new_len <= self.len);
        self.len = new_len;
    }

    pub fn resetCache(self: *CudaLM) !void {
        self.len = 0;
    }

    pub fn checkpointBytes(self: *const CudaLM) usize {
        _ = self;
        return 0;
    }

    pub fn checkpoint(self: *CudaLM, out: []u8) !void {
        _ = self;
        std.debug.assert(out.len == 0);
    }

    pub fn restoreCheckpoint(self: *CudaLM, snap: []const u8, at: usize) !void {
        std.debug.assert(snap.len == 0);
        self.truncate(at);
    }

    pub fn reinitCache(self: *CudaLM, dtype: kvmod.KvDtype) !void {
        for (self.k_cache) |*buf| self.be.growableDestroy(buf);
        for (self.v_cache) |*buf| self.be.growableDestroy(buf);
        self.kv_dtype = dtype;
        self.len = 0;
        for (self.k_cache, self.v_cache) |*kb, *vb| {
            kb.* = try self.be.growableCreate(dtype.sizeBytes(self.capacity * self.cfg.kvDim()), dtype.sizeBytes(self.max_capacity * self.cfg.kvDim()));
            vb.* = try self.be.growableCreate(dtype.sizeBytes(self.capacity * self.cfg.kvDim()), dtype.sizeBytes(self.max_capacity * self.cfg.kvDim()));
        }
    }

    /// A new conversation: drop the context, keep the placement (the arbiter
    /// settles it again when it wants to).
    pub fn resetResidency(self: *CudaLM, budget: u64) !void {
        if (self.split) |*sp| if (budget != 0) {
            sp.budget = budget;
        };
        try self.resetCache();
    }

    /// Always arm the dynamic split (`budget == 0` = no offload): routed-expert
    /// layers migrate to the host on demand as the KV grows or the arbiter asks.
    pub fn autoOffload(self: *CudaLM, budget: u64) !bool {
        if (budget == 0) return false;
        if (self.split == null) try self.enableCpuSplit(.tail, budget, true);
        self.split.?.budget = budget;
        try self.offloadToBudget(budget);
        return true;
    }

    /// The migration order is the MoE layers front to back, continuing the prefix
    /// `warmWeights` already placed on the host.
    pub fn enableCpuSplit(self: *CudaLM, policy: CpuSplitPolicy, budget: u64, dynamic: bool) !void {
        std.debug.assert(self.split == null);
        var n: usize = 0;
        for (self.lm.layers) |layer| n += @intFromBool(layer.experts != null);
        const order = try self.gpa.alloc(usize, n);
        var k: usize = 0;
        for (self.lm.layers, 0..) |layer, l| if (layer.experts != null) {
            order[k] = l;
            k += 1;
        };
        var next: usize = 0;
        while (next < n and self.host_experts[order[next]]) next += 1;
        self.split = .{ .n_cpu = next, .policy = policy, .dynamic = dynamic, .budget = budget, .order = order, .next = next };
    }

    pub fn migrateLayer(self: *CudaLM, l: usize) !void {
        std.debug.assert(!self.host_experts[l]);
        const layer = self.lm.layers[l];
        if (layer.values) |v| self.be.evictWeightBytes(groupBytes(v.weights));
        if (layer.experts) |e| {
            self.be.evictWeightBytes(groupBytes(e.gate));
            self.be.evictWeightBytes(groupBytes(e.up));
            self.be.evictWeightBytes(groupBytes(e.down));
        }
        self.host_experts[l] = true;
        if (self.split) |*sp| sp.n_cpu += 1;
        self.ensureHostPath();
        self.dumpPlacement("migrate");
    }

    /// Uploads the layer's expert groups NOW, so `deviceUsed` tells the truth to
    /// the next promote decision: a lazy promote left two back-to-back settles
    /// each seeing the same free VRAM and promoting every layer, and the forward
    /// that followed OOM'd uploading them. A failed upload leaves the layer on the
    /// host and reports the error; the arbiter retries later.
    pub fn promoteLayer(self: *CudaLM, l: usize) !void {
        std.debug.assert(self.host_experts[l]);
        const be = self.be;
        const layer = self.lm.layers[l];
        errdefer {
            if (layer.values) |v| be.evictWeightBytes(groupBytes(v.weights));
            if (layer.experts) |e| {
                be.evictWeightBytes(groupBytes(e.gate));
                be.evictWeightBytes(groupBytes(e.up));
                be.evictWeightBytes(groupBytes(e.down));
            }
        }
        if (layer.values) |v| {
            _ = try be.residentWeight(groupBytes(v.weights));
            warmExpertWeights(be, v.weights);
        }
        if (layer.experts) |e| {
            _ = try be.residentWeight(groupBytes(e.gate));
            _ = try be.residentWeight(groupBytes(e.up));
            _ = try be.residentWeight(groupBytes(e.down));
            warmExpertWeights(be, e.gate);
            warmExpertWeights(be, e.up);
            warmExpertWeights(be, e.down);
        }
        self.host_experts[l] = false;
        if (self.split) |*sp| sp.n_cpu -= 1;
        self.dumpPlacement("promote");
    }

    pub fn offloadUntilFree(self: *CudaLM, needed: u64) !void {
        return residency.offloadUntilFree(self, needed);
    }

    pub fn offloadToBudget(self: *CudaLM, target: u64) !void {
        return residency.offloadToBudget(self, target);
    }

    pub fn promoteLayers(self: *CudaLM, budget: u64) !usize {
        return residency.promoteBack(self, budget);
    }

    /// A host layer needs either the staging buffers (GPU prefill) or a
    /// full-width CPU scratch; the first migration after warm may find neither.
    fn ensureHostPath(self: *CudaLM) void {
        if (self.stage == null and self.stageable()) self.stageAlloc() catch {
            self.stage = null;
        };
        if (self.stage == null and self.cpu_rows < self.max_rows) {
            if (k2.Scratch.init(self.gpa, self.cfg, self.max_rows)) |scratch| {
                self.cpu_scratch.deinit(self.gpa);
                self.cpu_scratch = scratch;
                self.cpu_rows = self.max_rows;
            } else |_| {}
        }
    }

    fn routedBytes(layer: k2.Layer) usize {
        var total: usize = 0;
        if (layer.values) |values| for (values.weights) |weight| {
            total += weight.bytes.len;
        };
        if (layer.experts) |experts| for (experts.gate, experts.up, experts.down) |gate, up, down| {
            total += gate.bytes.len + up.bytes.len + down.bytes.len;
        };
        return total;
    }

    fn warmExpertWeights(be: *Backend, weights: []const Weight) void {
        if (weights.len == 0) return;
        be.warmWeightGroup(weights[0].bytes, weights.len);
    }

    /// Whether every MoE layer's expert groups can run through the grouped
    /// kernels from a staging copy (one contiguous q6_k group per matrix).
    fn stageable(self: *const CudaLM) bool {
        const c = self.cfg;
        for (self.lm.layers) |layer| {
            const e = layer.experts orelse continue;
            if (!groupedWeights(e.gate, c.expert_intermediate, c.hidden) or
                !groupedWeights(e.up, c.expert_intermediate, c.hidden) or
                !groupedWeights(e.down, c.hidden, c.expert_intermediate)) return false;
            if (layer.values) |v| if (!groupedWeights(v.weights, c.kvDim(), c.hidden)) return false;
        }
        return true;
    }

    /// Grow the f16 expert scratches now so the cache plan sees them.
    fn reserveExpertScratch(self: *CudaLM) void {
        if (self.expertGemm() != .f16) return;
        const c = self.cfg;
        var elems: u64 = 0;
        for (self.lm.layers) |layer| if (layer.experts) |e| {
            elems = @max(elems, @as(u64, e.gate.len) * c.expert_intermediate * c.hidden);
            if (layer.values) |v| elems = @max(elems, @as(u64, v.weights.len) * c.kvDim() * c.hidden);
        };
        const max_used = @max(c.n_experts_used, c.n_value_experts_used);
        const a_rows = std.mem.alignForward(usize, self.max_rows * max_used, 128) + 128;
        self.be.reserveDequantScratch(elems * 2, a_rows * @max(c.hidden, c.expert_intermediate) * 2) catch {};
    }

    fn stageBytesNeeded(self: *const CudaLM) u64 {
        for (self.lm.layers) |layer| if (layer.experts) |e| return Stage.bytes(e, layer.values);
        return 0;
    }

    fn stageAlloc(self: *CudaLM) !void {
        const be = self.be;
        var gate: ?Buf = null;
        var up: ?Buf = null;
        var down: ?Buf = null;
        var values: ?Buf = null;
        errdefer {
            if (gate) |*b| be.tensorDestroy(b);
            if (up) |*b| be.tensorDestroy(b);
            if (down) |*b| be.tensorDestroy(b);
            if (values) |*b| be.tensorDestroy(b);
        }
        for (self.lm.layers) |layer| if (layer.experts) |e| {
            gate = try be.tensorCreate(groupBytes(e.gate).len);
            up = try be.tensorCreate(groupBytes(e.up).len);
            down = try be.tensorCreate(groupBytes(e.down).len);
            if (layer.values) |v| values = try be.tensorCreate(groupBytes(v.weights).len);
            self.stage = .{ .gate = gate.?, .up = up.?, .down = down.?, .values = values };
            return;
        };
    }

    fn planCpuMoeLayers(self: *CudaLM, stage_bytes: u64) usize {
        if (std.c.getenv("TP_K2_CPU_MOE_LAYERS")) |raw| {
            const count = std.fmt.parseInt(usize, std.mem.span(raw), 10) catch return 0;
            return @min(count, self.cfg.n_layers);
        }
        var keep: u64 = 0;
        for (self.lm.layers) |layer| keep += routedBytes(layer);
        const free = self.be.ctx.memGetInfo().free;
        const reserve: u64 = 1 << 30;
        const cap = expertCacheLimit(self.be);
        if (keep <= @min(cap, free -| reserve)) return 0;
        const budget = @min(cap, free -| reserve -| stage_bytes);
        var cpu_layers: usize = 0;
        for (self.lm.layers, 0..) |layer, l| {
            if (keep <= budget) break;
            const bytes = routedBytes(layer);
            if (bytes == 0) continue;
            keep -|= bytes;
            cpu_layers = l + 1;
        }
        return cpu_layers;
    }

    pub fn warmWeights(self: *CudaLM) void {
        const be = self.be;
        be.warmWeight(self.lm.embed.bytes);
        be.warmWeight(self.lm.head.bytes);
        be.warmWeight(std.mem.sliceAsBytes(self.lm.final_norm));
        for (self.lm.layers) |layer_weights| {
            be.warmWeight(std.mem.sliceAsBytes(layer_weights.attn_norm));
            be.warmWeight(layer_weights.q.bytes);
            be.warmWeight(layer_weights.k.bytes);
            if (layer_weights.v) |weight| be.warmWeight(weight.bytes);
            if (layer_weights.values) |values| be.warmWeight(values.router.bytes);
            if (layer_weights.attn_gate) |weight| be.warmWeight(weight.bytes);
            be.warmWeight(layer_weights.o.bytes);
            be.warmWeight(std.mem.sliceAsBytes(layer_weights.ffn_norm));
            if (layer_weights.experts) |experts| {
                be.warmWeight(experts.router.bytes);
                if (experts.shared_gate) |weight| be.warmWeight(weight.bytes);
                if (experts.shared_up) |weight| be.warmWeight(weight.bytes);
                if (experts.shared_down) |weight| be.warmWeight(weight.bytes);
            } else {
                be.warmWeight(layer_weights.dense_gate.?.bytes);
                be.warmWeight(layer_weights.dense_up.?.bytes);
                be.warmWeight(layer_weights.dense_down.?.bytes);
            }
        }
        self.reserveExpertScratch();
        const stage_bytes: u64 = if (self.stageable()) self.stageBytesNeeded() else 0;
        const cpu_moe_layers = self.planCpuMoeLayers(stage_bytes);
        for (self.host_experts[0..cpu_moe_layers]) |*h| h.* = true;
        if (cpu_moe_layers > 0) self.ensureHostPath();
        for (self.lm.layers, 0..) |layer, l| {
            if (self.host_experts[l]) continue;
            if (layer.values) |values| warmExpertWeights(be, values.weights);
            if (layer.experts) |experts| {
                warmExpertWeights(be, experts.gate);
                warmExpertWeights(be, experts.up);
                warmExpertWeights(be, experts.down);
            }
        }
        // Always dynamic, so KV growth migrates layers instead of failing; the
        // card is the ceiling until an arbiter hands down a budget.
        if (self.split == null) self.enableCpuSplit(.tail, be.ctx.memGetInfo().total, true) catch {};
        self.dumpPlacement("warm");
    }

    /// The migratable bytes of a layer: its routed experts. Attention, dense and
    /// shared weights always stay resident and count under `nonLayerDeviceBytes`.
    pub fn layerWeightBytes(self: *CudaLM, layer_index: usize) usize {
        return routedBytes(self.lm.layers[layer_index]);
    }

    fn fixedBytes(l: k2.Layer) usize {
        var total = l.q.bytes.len + l.k.bytes.len + l.o.bytes.len;
        if (l.v) |weight| total += weight.bytes.len;
        if (l.values) |values| total += values.router.bytes.len;
        if (l.attn_gate) |weight| total += weight.bytes.len;
        if (l.experts) |experts| {
            total += experts.router.bytes.len;
            if (experts.shared_gate) |weight| total += weight.bytes.len;
            if (experts.shared_up) |weight| total += weight.bytes.len;
            if (experts.shared_down) |weight| total += weight.bytes.len;
        } else {
            total += l.dense_gate.?.bytes.len + l.dense_up.?.bytes.len + l.dense_down.?.bytes.len;
        }
        return total;
    }

    /// Everything resident regardless of placement: fixed weights and the device KV.
    pub fn nonLayerDeviceBytes(self: *CudaLM) u64 {
        var total: u64 = self.lm.embed.bytes.len + self.lm.head.bytes.len;
        for (self.lm.layers) |l| total += fixedBytes(l);
        return total + self.kvDeviceBytes();
    }

    pub fn promoteCost(self: *CudaLM, layer_index: usize) usize {
        return self.layerWeightBytes(layer_index) + residency.promote_slack;
    }
};

const Bufs = struct {
    x: Buf,
    normed: Buf,
    q: Buf,
    k: Buf,
    v: Buf,
    attn: Buf,
    gate: Buf,
    tmp: Buf,
    expert_in: Buf,
    expert_a: Buf,
    expert_b: Buf,
    expert_out: Buf,
    route_logits: Buf,
    route_pack: Buf,
    attn_scratch: Buf,
    logits: Buf,
    argmax_v: Buf,
    argmax_i: Buf,
    argmax_out: Buf,
    topk_v: Buf,
    topk_i: Buf,

    fn init(be: *Backend, c: k2.Config, rows: usize) !Bufs {
        const routed_inner = @max(c.expert_intermediate, c.kvDim());
        const dense_inner = @max(c.intermediate, c.shared_intermediate);
        const max_out = @max(c.hidden, c.kvDim());
        const max_experts = @max(c.n_experts, c.n_value_experts);
        const route_slots = routeSlots(c, rows);
        const inner_rows = @max(rows * dense_inner, route_slots * routed_inner);
        const out_rows = @max(rows * c.hidden, route_slots * max_out);
        const attn_rows = @max(32 * 8, rows) * c.n_heads;
        var self: Bufs = undefined;
        var made: usize = 0;
        errdefer inline for (fields, 0..) |name, i| if (i < made) be.tensorDestroy(&@field(self, name));
        const sizes = [fields.len]usize{
            rows * c.hidden * 4,
            rows * c.hidden * 4,
            rows * c.qDim() * 4,
            rows * c.kvDim() * 4,
            rows * c.kvDim() * 4,
            rows * c.qDim() * 4,
            rows * c.qDim() * 4,
            rows * c.hidden * 4,
            route_slots * c.hidden * 4,
            inner_rows * 4,
            inner_rows * 4,
            out_rows * 4,
            rows * max_experts * 4,
            routePackSlots(c, rows) * 4,
            attn_rows * (c.head_dim + 4) * 4,
            c.vocab * 4,
            4096 * 4,
            4096 * 4,
            4,
            cuda.backend.topk_lanes * cuda.backend.topk_m * 4,
            cuda.backend.topk_lanes * cuda.backend.topk_m * 4,
        };
        inline for (fields, sizes) |name, size| {
            be.noteBuf(name, size);
            @field(self, name) = try be.tensorCreate(size);
            made += 1;
        }
        return self;
    }

    fn deinit(self: *Bufs, be: *Backend) void {
        inline for (fields) |name| be.tensorDestroy(&@field(self, name));
        self.* = undefined;
    }

    const fields = [_][]const u8{
        "x",            "normed",   "q",        "k",          "v",            "attn",      "gate",       "tmp",
        "expert_in",    "expert_a", "expert_b", "expert_out", "route_logits", "route_pack", "attn_scratch", "logits",
        "argmax_v",     "argmax_i", "argmax_out", "topk_v",   "topk_i",
    };
};

test "K2 Horizon CUDA produces the expected first token" {
    const path = "/home/qt/genai/lmstudio/models/K2-Horizon-MoVA-36B-A4B-Q6_K.gguf";
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var gguf = Gguf.open(gpa, io, path) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer gguf.deinit();
    var lm = try k2.Model.load(gpa, &gguf);
    defer lm.deinit();
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();
    be.pinAllWeights();
    var gpu = try CudaLM.init(gpa, be, &lm, .fixed(14));
    defer gpu.deinit();
    gpu.warmWeights();

    var tok = try Tokenizer.initFromGguf(gpa, &gguf);
    defer tok.deinit();
    const old_family = chat.family;
    const old_thinking = chat.enable_thinking;
    defer {
        chat.setFamily(old_family);
        chat.setThinking(old_thinking);
    }
    chat.applyTokenizer(&tok);
    chat.setFamily(.k2_horizon);
    chat.setThinking(false);
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    try chat.appendUser(&tok, gpa, "Hello", &ids);
    try chat.openAssistant(&tok, gpa, &ids);
    try std.testing.expectEqualSlices(u32, &.{ 0, 250018, 2672, 200, 16822, 250019, 250018, 142036, 200, 250029, 200, 250030, 200 }, ids.items);

    const id = try gpu.stepArgmax(io, ids.items);
    const decoded = try tok.decodeAlloc(gpa, &.{id});
    defer gpa.free(decoded);
    try std.testing.expectEqualStrings("Hello", decoded);

    // Migrate three more layers' experts to the host (the arbiter's path) and
    // the same prompt must give the same token through the CPU expert path.
    const before = gpu.hostLayers();
    try gpu.offloadToBudget(be.deviceUsed() - 3 * CudaLM.routedBytes(lm.layers[3]) + (1 << 20));
    errdefer std.debug.print("host layers {d} -> {d}\n", .{ before, gpu.hostLayers() });
    try std.testing.expect(gpu.hostLayers() >= before + 3);
    try gpu.resetCache();
    const id2 = try gpu.stepArgmax(io, ids.items);
    try std.testing.expectEqual(id, id2);

    // Promote them back: the groups upload at promote time, so device usage
    // grows by their bytes here, not at the next forward.
    const used_host = be.deviceUsed();
    _ = try gpu.promoteLayers(std.math.maxInt(u64));
    try std.testing.expectEqual(before, gpu.hostLayers());
    try std.testing.expect(be.deviceUsed() >= used_host + 3 * CudaLM.routedBytes(lm.layers[3]) - (64 << 20));
    try gpu.resetCache();
    try std.testing.expectEqual(id, try gpu.stepArgmax(io, ids.items));
}
