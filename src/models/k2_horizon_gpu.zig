//! K2 Horizon Vulkan inference, `k2_horizon_cuda`'s twin. The fixed weights and as many
//! routed-expert groups as fit stay resident, each group one buffer whose experts bind
//! as views (`Context.registerWeightGroup`); the rest live on the host, run on the CPU
//! for decode and small batches, and for a larger batch upload for that one layer and
//! leave again. The layer forward is `k2_horizon_dev.layer`.
//!
//! A batch's routed rows are packed by expert: one gather fills `expert_in`, each expert
//! runs its linears on its row views, and one `moe_combine` writes the result by slot, so
//! nothing accumulates across experts and the sum's order is fixed.

const std = @import("std");
const k2 = @import("k2_horizon.zig");
const k2_dev = @import("k2_horizon_dev.zig");
const qwen3 = @import("qwen3.zig");
const lin_gpu = @import("lin_gpu.zig");
const kv_gpu = @import("kv_gpu.zig");
const gpu = @import("tp_gpu").context;
const ops = @import("tp_ops");
const kvmod = @import("tp_core").kv_cache;
const sample = @import("tp_core").sample;
const init_defaults = @import("tp_core").init_defaults;
const bnd = @import("tp_runtime").boundary;
const residency = @import("tp_runtime").residency;

const Buf = gpu.DeviceBuffer;
const Weight = lin_gpu.Weight;

/// Rows per batched forward; every routed scratch holds this many tokens' routes.
const prefill_chunk = 512;
/// Batches up to this size run a host layer's experts on the CPU; a larger one takes
/// whichever of that and uploading the layer's groups `host_cost` has measured cheaper.
const cpu_small_max = 8;
/// Batches up to this size route resident blocks on the device. Each slot reads its
/// expert's weights again, which the per-expert linears do not: on a 3090 the two broke
/// even near 128 rows.
const device_route_max = 64;
/// KV chunks per head in the attention split pass: decode, then a prefill chunk.
const nsplit = 32;
const nsplit_prefill = 8;
const gemv_nchunk = 32;

fn nbuf(ctx: *gpu.Context, weights: []const f32) !Buf {
    return .{ .buf = try ctx.smallBuffer(std.mem.sliceAsBytes(weights)), .mem = .null_handle, .size = 0 };
}

/// One contiguous, uniformly typed expert group (a GGUF 3-D tensor), as one byte range.
fn groupBytes(weights: []const Weight) ?[]const u8 {
    if (weights.len == 0) return null;
    const first = weights[0];
    for (weights, 0..) |w, i| {
        if (w.dtype != first.dtype or w.bytes.len != first.bytes.len) return null;
        if (@intFromPtr(w.bytes.ptr) != @intFromPtr(first.bytes.ptr) + i * first.bytes.len) return null;
    }
    return first.bytes.ptr[0 .. first.bytes.len * weights.len];
}

/// A layer's routed weights, as the groups they upload in.
fn layerGroups(layer: k2.Layer, out: *[4][]const u8) [][]const u8 {
    var n: usize = 0;
    if (layer.values) |v| if (groupBytes(v.weights)) |g| {
        out[n] = g;
        n += 1;
    };
    if (layer.experts) |e| for ([_][]const Weight{ e.gate, e.up, e.down }) |ws| if (groupBytes(ws)) |g| {
        out[n] = g;
        n += 1;
    };
    return out[0..n];
}

fn routedBytes(layer: k2.Layer) usize {
    var total: usize = 0;
    if (layer.values) |v| for (v.weights) |w| {
        total += w.bytes.len;
    };
    if (layer.experts) |e| for (e.gate, e.up, e.down) |g, u, d| {
        total += g.bytes.len + u.bytes.len + d.bytes.len;
    };
    return total;
}

fn fixedBytes(l: k2.Layer) usize {
    var total = l.q.bytes.len + l.k.bytes.len + l.o.bytes.len;
    if (l.v) |w| total += w.bytes.len;
    if (l.values) |v| total += v.router.bytes.len;
    if (l.attn_gate) |w| total += w.bytes.len;
    if (l.experts) |e| {
        total += e.router.bytes.len;
        for ([_]?Weight{ e.shared_gate, e.shared_up, e.shared_down }) |w| if (w) |ww| {
            total += ww.bytes.len;
        };
    } else {
        total += l.dense_gate.?.bytes.len + l.dense_up.?.bytes.len + l.dense_down.?.bytes.len;
    }
    return total;
}

pub const VulkanLM = struct {
    /// The prefill batch this stepper wants the engine to hand it.
    pub const prefill_batch = prefill_chunk;
    /// `tp-llm --bench` alternates the `OpSDot` q8 decode GEMVs (toggle on: off) with
    /// the f32 ones.
    pub const bench_toggle = &gpu.Context.sdot_off;

    lm: *const k2.Model,
    be: *gpu.Context,
    gpa: std.mem.Allocator,
    cfg: k2.Config,
    lin: lin_gpu.Lin,
    lin_plan: lin_gpu.Plan,
    capacity: usize,
    initial_capacity: usize,
    max_capacity: usize,
    kv_dtype: kvmod.KvDtype = .f32,
    len: usize = 0,
    /// Polled before each prefill chunk: residency, pause and cancel (the GUI).
    boundary: ?bnd.Hook = null,
    /// The Io of the last step, which the host experts run on.
    io: ?std.Io = null,
    sin_off: u32,
    kv: kv_gpu.Kv,
    freqs_d: Buf,
    bufs: Bufs,
    cpu_scratch: k2.Scratch,
    cpu_rows: usize,
    /// Set before the first batch that has to choose (`calibrateHost`).
    host_cost: ?k2.HostCost = null,
    /// The current layer's staging of `bufs.normed` for its attention projections.
    attn_stage: ?lin_gpu.Group = null,
    route_logits: []f32,
    selected: []usize,
    selected_weights: []f32,
    pack: []u32,
    route_counts: []usize,
    route_starts: []usize,
    /// Per layer: its routed experts live on the host. The migration unit of `split`.
    host_experts: []bool,
    split: ?Split = null,

    pub const Split = struct {
        n_cpu: usize = 0,
        policy: residency.CpuSplitPolicy = .tail,
        dynamic: bool = false,
        budget: u64 = 0,
        order: []usize = &.{},
        next: usize = 0,
    };

    pub fn init(gpa: std.mem.Allocator, ctx: *gpu.Context, lm: *const k2.Model, cap: kvmod.Capacity) !VulkanLM {
        const c = lm.cfg;
        var self: VulkanLM = init_defaults.of(VulkanLM);
        self.lm = lm;
        self.be = ctx;
        self.gpa = gpa;
        self.cfg = c;
        self.capacity = @max(@min(cap.initial, cap.max), 1);
        self.initial_capacity = self.capacity;
        self.max_capacity = cap.max;
        self.kv_dtype = cap.kv_dtype;
        self.len = 0;
        self.split = null;
        self.boundary = null;
        self.io = null;
        self.sin_off = @intCast(cap.max * (c.head_dim / 2));
        self.cpu_rows = prefill_chunk;
        self.host_cost = k2.HostCost.pinned();
        self.attn_stage = null;
        ctx.weight_noise.setDepth(c.n_layers);

        const max_out = @max(@max(c.vocab, c.qDim()), @max(@max(c.intermediate, c.shared_intermediate), @max(c.hidden, @max(c.n_experts, c.n_value_experts))));
        self.lin = try lin_gpu.Lin.init(gpa, ctx, max_out, gemv_nchunk, 3);
        errdefer self.lin.deinit();
        {
            var arena = std.heap.ArenaAllocator.init(gpa);
            defer arena.deinit();
            const lins = try lm.deviceLins(arena.allocator());
            self.lin_plan = try self.lin.plan(lins, lin_gpu.Use.llm(prefill_chunk), "k2-horizon vulkan");
            var batched: std.ArrayList(Weight) = .empty;
            for (lins) |w| if (w.bytes.ptr != lm.head.bytes.ptr) try batched.append(arena.allocator(), w);
            try lin_gpu.presize(ctx, self.lin_plan, batched.items);
        }
        for (lm.layers) |layer| {
            var gs: [4][]const u8 = undefined;
            for (layerGroups(layer, &gs)) |g| try ctx.registerWeightGroup(g);
        }

        var freqs = try ops.rope.rotateHalfFreqs(gpa, cap.max, c.head_dim, c.rope_theta);
        defer freqs.deinit(gpa);
        const half = c.head_dim / 2;
        const fp = try gpa.alloc(f32, 2 * cap.max * half);
        defer gpa.free(fp);
        @memcpy(fp[0 .. cap.max * half], freqs.cos);
        @memcpy(fp[cap.max * half ..], freqs.sin);
        self.freqs_d = try ctx.tensorCreate(fp.len * 4);
        errdefer ctx.tensorDestroy(&self.freqs_d);
        try ctx.tensorUpload(self.freqs_d, std.mem.sliceAsBytes(fp));

        self.kv = try kv_gpu.Kv.init(gpa, ctx, c.n_layers, self.capacity, c.kvDim(), cap.kv_dtype, &.{});
        errdefer self.kv.deinit(gpa, ctx);
        self.bufs = try Bufs.init(ctx, c);
        errdefer self.bufs.deinit(ctx);
        self.cpu_scratch = try k2.Scratch.init(gpa, c, self.cpu_rows);
        errdefer self.cpu_scratch.deinit(gpa);

        const max_experts = @max(c.n_experts, c.n_value_experts);
        const max_used = @max(c.n_experts_used, c.n_value_experts_used);
        self.route_logits = try gpa.alloc(f32, prefill_chunk * max_experts);
        errdefer gpa.free(self.route_logits);
        self.selected = try gpa.alloc(usize, prefill_chunk * max_used);
        errdefer gpa.free(self.selected);
        self.selected_weights = try gpa.alloc(f32, prefill_chunk * max_used);
        errdefer gpa.free(self.selected_weights);
        self.pack = try gpa.alloc(u32, Bufs.packSlots(c));
        errdefer gpa.free(self.pack);
        self.route_counts = try gpa.alloc(usize, max_experts);
        errdefer gpa.free(self.route_counts);
        self.route_starts = try gpa.alloc(usize, max_experts);
        errdefer gpa.free(self.route_starts);
        self.host_experts = try gpa.alloc(bool, c.n_layers);
        @memset(self.host_experts, false);
        return self;
    }

    pub fn deinit(self: *VulkanLM) void {
        self.kv.deinit(self.gpa, self.be);
        self.be.tensorDestroy(&self.freqs_d);
        self.bufs.deinit(self.be);
        self.cpu_scratch.deinit(self.gpa);
        self.lin.deinit();
        self.gpa.free(self.route_logits);
        self.gpa.free(self.selected);
        self.gpa.free(self.selected_weights);
        self.gpa.free(self.pack);
        self.gpa.free(self.route_counts);
        self.gpa.free(self.route_starts);
        self.gpa.free(self.host_experts);
        if (self.split) |sp| self.gpa.free(sp.order);
        self.* = undefined;
    }

    pub fn cached(self: *const VulkanLM) usize {
        return self.len;
    }

    pub fn remaining(self: *const VulkanLM) usize {
        return self.capacity - self.len;
    }

    pub fn capacityMax(self: *const VulkanLM) usize {
        return self.max_capacity;
    }

    pub fn vocab(self: *const VulkanLM) usize {
        return self.cfg.vocab;
    }

    pub fn vramUsed(self: *const VulkanLM) u64 {
        return self.be.device_used;
    }

    fn kvDeviceBytes(self: *const VulkanLM) u64 {
        return kv_gpu.Kv.windowBytes(self.cfg.n_layers, self.capacity, self.cfg.kvDim(), self.kv_dtype);
    }

    /// Grow the device KV (attention always runs on the device). Under a dynamic split,
    /// routed-expert layers migrate to the host first until the growth fits, and an
    /// out-of-memory during the grow migrates one more and retries.
    pub fn ensureCapacity(self: *VulkanLM, min_rows: usize) !void {
        if (min_rows <= self.capacity) return;
        if (min_rows > self.max_capacity) return error.ContextFull;
        const to = kvmod.growTarget(self.capacity, min_rows, self.max_capacity);
        if (self.split) |*sp| if (sp.dynamic) {
            const add = self.kv_dtype.sizeBytes((to - self.capacity) * self.cfg.kvDim());
            while (true) {
                const need = 2 * self.cfg.n_layers * add + (64 << 20);
                const free = @min(sp.budget -| self.be.deviceUsed(), self.be.headroom());
                if (free >= need) break;
                if (!(try residency.migrateNext(self))) break;
            }
        };
        while (true) {
            self.kv.grow(self.be, to, self.len) catch |err| switch (err) {
                error.DeviceOutOfMemory, error.OutOfMemory => {
                    if (self.split != null and try residency.migrateNext(self)) continue;
                    return error.ContextFull;
                },
                else => return err,
            };
            break;
        }
        self.capacity = to;
    }

    pub fn step(self: *VulkanLM, io: std.Io, ids: []const u32, logits: []f32) !void {
        self.io = io;
        try self.forwardLast(ids);
        try self.be.tensorDownload(self.bufs.logits, std.mem.sliceAsBytes(logits[0..self.cfg.vocab]));
    }

    /// Forward `ids` into the cache without the head.
    pub fn prefill(self: *VulkanLM, ids: []const u32) !void {
        std.debug.assert(ids.len <= self.remaining());
        var off: usize = 0;
        while (off < ids.len) {
            try bnd.check(self);
            const n: usize = @min(prefill_chunk, ids.len - off);
            try self.forwardChunk(ids[off..][0..n], false);
            off += n;
        }
    }

    /// Every chunk of `ids`, the last position's logits left in `bufs.logits`.
    fn forwardLast(self: *VulkanLM, ids: []const u32) !void {
        std.debug.assert(ids.len >= 1 and ids.len <= self.remaining());
        var off: usize = 0;
        while (off < ids.len) {
            const n: usize = @min(prefill_chunk, ids.len - off);
            try self.forwardChunk(ids[off..][0..n], off + n == ids.len);
            off += n;
        }
    }

    /// One chunk, offloading routed-expert layers and retrying when the device runs out
    /// of memory: a failed chunk commits nothing (`len` unchanged), so moving a few
    /// layers and running it again is exact.
    fn forwardChunk(self: *VulkanLM, ids: []const u32, head: bool) !void {
        self.be.weight_noise.tick();
        while (true) {
            return self.forwardChunkOnce(ids, head) catch |err| switch (err) {
                error.DeviceOutOfMemory, error.OutOfMemory => {
                    if (self.be.batching) self.be.abortBatch();
                    try self.ensureOffloadArmed();
                    var moved: usize = 0;
                    while (moved < 4) : (moved += 1) {
                        if (!(try residency.migrateNext(self))) break;
                    }
                    if (moved == 0) return err;
                    std.log.info("[vram] k2 forward OOM at ctx {d}: moved {d} expert layer(s) to the host, retrying", .{ self.len, moved });
                    continue;
                },
                else => return err,
            };
        }
    }

    fn forwardChunkOnce(self: *VulkanLM, ids: []const u32, head: bool) !void {
        const ctx = self.be;
        const c = self.cfg;
        const b = &self.bufs;
        const seq = ids.len;
        std.debug.assert(seq > 0 and seq <= prefill_chunk and seq <= self.remaining());
        const x = try self.gpa.alloc(f32, seq * c.hidden);
        defer self.gpa.free(x);
        try qwen3.embedTokens(self.lm.embed, ids, x);
        // Before the batch opens: an upload inside it would flush the recording.
        try ctx.tensorUpload(try ctx.view(b.x, 0, seq * c.hidden * 4), std.mem.sliceAsBytes(x));
        if (seq > cpu_small_max and self.host_cost == null) try self.calibrateHost();
        try ctx.beginBatch();
        errdefer if (ctx.batching) ctx.abortBatch();
        const ctok = ops.cancel.token;
        var d: Dev = .{ .s = self, .cfg = c, .bufs = b };
        for (self.lm.layers, 0..) |layer, l| {
            if (ops.cancel.canceled(ctok)) return error.Canceled;
            ctx.weight_noise.atLayer(l);
            try k2_dev.layer(&d, layer, l, seq, self.len);
        }
        if (head) {
            const last = try ctx.view(b.x, (seq - 1) * c.hidden * 4, c.hidden * 4);
            try d.groupNorm(last, b.last, self.lm.final_norm, 1);
            ctx.weight_noise.atHead();
            try self.lin.linear(self.lin_plan, b.logits, 0, b.last, 1, self.lm.head);
        }
        try ctx.endBatch();
        self.len += seq;
    }

    /// Upload every weight now: one forward for the fixed path, then the groups of the
    /// layers that keep their experts here. A layer whose experts will not fit goes to
    /// the host first, front to back, as k2_horizon_cuda plans it.
    pub fn warmWeights(self: *VulkanLM) void {
        if (self.len != 0) return;
        const cpu_layers = self.planCpuMoeLayers();
        if (cpu_layers > 0) std.log.info("[offload] k2 experts of layers 0..{d} on the host", .{cpu_layers});
        for (self.host_experts[0..cpu_layers]) |*h| h.* = true;
        for (self.lm.layers, 0..) |layer, l| {
            if (self.host_experts[l]) continue;
            var gs: [4][]const u8 = undefined;
            for (layerGroups(layer, &gs)) |g| self.be.warmWeightGroup(g) catch |err| {
                std.log.warn("k2 vulkan warm-up: layer {d} experts: {t}", .{ l, err });
                break;
            };
        }
        const zeros: [2]u32 = @splat(0);
        self.forwardChunk(zeros[0..1], true) catch |err| std.log.warn("k2 vulkan warm-up: {t}", .{err});
        self.truncate(0);
        // Always dynamic, so KV growth migrates layers instead of failing; the card is
        // the ceiling until an arbiter hands down a budget.
        const card = if (self.be.memGetInfo()) |mi| mi.total else std.math.maxInt(u64);
        if (self.split == null) self.enableCpuSplit(.tail, card, true) catch {};
    }

    fn planCpuMoeLayers(self: *VulkanLM) usize {
        var keep: u64 = 0;
        for (self.lm.layers) |layer| keep += routedBytes(layer);
        var fixed: u64 = self.lm.head.bytes.len + self.kvDeviceBytes();
        for (self.lm.layers) |layer| fixed += fixedBytes(layer);
        // A host layer's largest group must also fit for the batch it uploads for.
        var transient: u64 = 0;
        for (self.lm.layers) |layer| transient = @max(transient, routedBytes(layer));
        const reserve: u64 = 1 << 30;
        const avail = if (self.be.memGetInfo()) |mi| (mi.free -| reserve) -| fixed else return 0;
        if (keep <= avail) return 0;
        const budget = avail -| transient;
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

    pub fn truncate(self: *VulkanLM, new_len: usize) void {
        std.debug.assert(new_len <= self.len);
        self.len = new_len;
    }

    pub fn resetCache(self: *VulkanLM) !void {
        self.len = 0;
    }

    /// The KV is append-only, so a turn boundary needs no snapshot: restoring is a
    /// truncate.
    pub fn checkpointBytes(self: *const VulkanLM) usize {
        _ = self;
        return 0;
    }

    pub fn checkpoint(self: *VulkanLM, out: []u8) !void {
        _ = self;
        std.debug.assert(out.len == 0);
    }

    pub fn restoreCheckpoint(self: *VulkanLM, snap: []const u8, at: usize) !void {
        std.debug.assert(snap.len == 0);
        self.truncate(at);
    }

    /// Re-create the cache at another element type, empty.
    pub fn reinitCache(self: *VulkanLM, dtype: kvmod.KvDtype) !void {
        var kv = try kv_gpu.Kv.init(self.gpa, self.be, self.cfg.n_layers, self.capacity, self.cfg.kvDim(), dtype, &.{});
        errdefer kv.deinit(self.gpa, self.be);
        self.kv.deinit(self.gpa, self.be);
        self.kv = kv;
        self.kv_dtype = dtype;
        self.len = 0;
    }

    // Residency: the hooks `residency.zig` drives, with k2_horizon_cuda's meaning.

    /// A new conversation: drop the context, keep the placement.
    pub fn resetResidency(self: *VulkanLM, budget: u64) !void {
        if (self.split) |*sp| if (budget != 0) {
            sp.budget = budget;
        };
        try self.resetCache();
    }

    /// Always arm the dynamic split (`budget == 0` is no offload).
    pub fn autoOffload(self: *VulkanLM, budget: u64) !bool {
        if (budget == 0) return false;
        if (self.split == null) try self.enableCpuSplit(.tail, budget, true);
        self.split.?.budget = budget;
        try self.offloadToBudget(budget);
        return true;
    }

    fn ensureOffloadArmed(self: *VulkanLM) !void {
        if (self.split != null) return;
        try self.enableCpuSplit(.tail, std.math.maxInt(u64), true);
    }

    /// The migration order is the MoE layers front to back, continuing the prefix
    /// `warmWeights` already placed on the host.
    pub fn enableCpuSplit(self: *VulkanLM, policy: residency.CpuSplitPolicy, budget: u64, dynamic: bool) !void {
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

    pub fn migrateLayer(self: *VulkanLM, l: usize) !void {
        std.debug.assert(!self.host_experts[l]);
        var gs: [4][]const u8 = undefined;
        for (layerGroups(self.lm.layers[l], &gs)) |g| self.be.evictWeightBytes(g);
        self.host_experts[l] = true;
        if (self.split) |*sp| sp.n_cpu += 1;
        std.log.debug("[offload] k2 layer {d} experts -> host at ctx {d}", .{ l, self.len });
    }

    /// Uploads the layer's groups now, so `deviceUsed` tells the next promote decision
    /// the truth. A failed upload leaves the layer on the host.
    pub fn promoteLayer(self: *VulkanLM, l: usize) !void {
        std.debug.assert(self.host_experts[l]);
        var gs: [4][]const u8 = undefined;
        const groups = layerGroups(self.lm.layers[l], &gs);
        errdefer for (groups) |g| self.be.evictWeightBytes(g);
        for (groups) |g| try self.be.warmWeightGroup(g);
        self.host_experts[l] = false;
        if (self.split) |*sp| sp.n_cpu -= 1;
        std.log.debug("[promote] k2 layer {d} experts -> device at ctx {d}", .{ l, self.len });
    }

    pub fn offloadUntilFree(self: *VulkanLM, needed: u64) !void {
        return residency.offloadUntilFree(self, needed);
    }

    pub fn offloadToBudget(self: *VulkanLM, target: u64) !void {
        return residency.offloadToBudget(self, target);
    }

    pub fn promoteLayers(self: *VulkanLM, budget: u64) !usize {
        return residency.promoteBack(self, budget);
    }

    /// The migratable bytes of a layer: its routed experts.
    pub fn layerWeightBytes(self: *VulkanLM, l: usize) usize {
        return routedBytes(self.lm.layers[l]);
    }

    /// Everything resident regardless of placement: the head, the fixed weights and the
    /// device KV. The embeddings are looked up on the host.
    pub fn nonLayerDeviceBytes(self: *VulkanLM) u64 {
        var total: u64 = self.lm.head.bytes.len;
        for (self.lm.layers) |layer| total += fixedBytes(layer);
        return total + self.kvDeviceBytes();
    }

    pub fn promoteCost(self: *VulkanLM, l: usize) usize {
        return self.layerWeightBytes(l) + residency.promote_slack;
    }

    // Sampling, on the device.

    pub fn stepArgmax(self: *VulkanLM, io: std.Io, ids: []const u32) !u32 {
        return self.stepArgmaxPen(io, ids, &.{}, .{});
    }

    pub fn stepArgmaxPen(self: *VulkanLM, io: std.Io, ids: []const u32, pen: []const sample.PenaltyEntry, sp: sample.Params) !u32 {
        self.io = io;
        try self.forwardLast(ids);
        const ctx = self.be;
        const b = &self.bufs;
        try ctx.opPenalize(b.logits, pen, sp);
        try ctx.opArgmax(b.logits, self.cfg.vocab, b.argmax_out, &b.argmax_v, &b.argmax_i);
        var id_f: [1]f32 = undefined;
        try ctx.tensorDownload(b.argmax_out, std.mem.sliceAsBytes(&id_f));
        return @intFromFloat(id_f[0]);
    }

    pub fn maxSelect(self: *const VulkanLM) usize {
        _ = self;
        return gpu.topk_lanes * gpu.topk_m;
    }

    pub fn stepSelect(self: *VulkanLM, io: std.Io, ids: []const u32, out_id: []u32, out_logit: []f32) !usize {
        return self.stepSelectPen(io, ids, &.{}, .{}, out_id, out_logit);
    }

    pub fn stepSelectPen(self: *VulkanLM, io: std.Io, ids: []const u32, pen: []const sample.PenaltyEntry, sp: sample.Params, out_id: []u32, out_logit: []f32) !usize {
        self.io = io;
        try self.forwardLast(ids);
        const ctx = self.be;
        const b = &self.bufs;
        try ctx.opPenalize(b.logits, pen, sp);
        const count = try ctx.opTopK(b.logits, self.cfg.vocab, &b.topk_v, &b.topk_i);
        std.debug.assert(count <= out_id.len and count <= out_logit.len);
        try ctx.tensorDownload(b.topk_v, std.mem.sliceAsBytes(out_logit[0..count]));
        var idx_f: [gpu.topk_lanes * gpu.topk_m]f32 = undefined;
        try ctx.tensorDownload(b.topk_i, std.mem.sliceAsBytes(idx_f[0..count]));
        for (out_id[0..count], idx_f[0..count]) |*o, f| o.* = @intFromFloat(f);
        return count;
    }

    // The routed blocks.

    fn selectRoutes(self: *VulkanLM, seq: usize, bias: []const f32, n_experts: usize, n_used: usize) !void {
        const logits = self.route_logits[0 .. seq * n_experts];
        try self.be.tensorDownload(try self.be.view(self.bufs.route_logits, 0, logits.len * 4), std.mem.sliceAsBytes(logits));
        k2.route(self.cfg, logits, bias, seq, n_experts, n_used, self.selected[0 .. seq * n_used], self.selected_weights[0 .. seq * n_used]);
    }

    /// The batch's routed rows, packed by expert (token order within one):
    /// `ids[r]` the token a row gathers, `scales[r]` its weight, `slots[t * used + s]`
    /// the row of token t's slot s. Uploaded once; `expert_in` gathered from `x`.
    const Routes = struct { rows: usize, ids: Buf, scales: Buf, slots: Buf };

    fn packRoutes(self: *VulkanLM, x: Buf, seq: usize, used: usize, n_experts: usize) !Routes {
        const ctx = self.be;
        const c = self.cfg;
        const s_n = seq * used;
        const r_pad = Bufs.padSlots(s_n);
        const sel = self.selected[0..s_n];
        const wts = self.selected_weights[0..s_n];
        const counts = self.route_counts[0..n_experts];
        const starts = self.route_starts[0..n_experts];
        @memset(counts, 0);
        for (sel) |e| counts[e] += 1;
        var acc: usize = 0;
        for (counts, starts) |n, *st| {
            st.* = acc;
            acc += n;
        }
        const ids = self.pack[0..s_n];
        const scales = std.mem.bytesAsSlice(f32, std.mem.sliceAsBytes(self.pack[r_pad..][0..s_n]));
        const slots = self.pack[2 * r_pad ..][0..s_n];
        @memset(counts, 0);
        for (0..seq) |t| for (0..used) |s| {
            const at = t * used + s;
            const e = sel[at];
            const r = starts[e] + counts[e];
            counts[e] += 1;
            ids[r] = @intCast(t);
            scales[r] = wts[at];
            slots[at] = @intCast(r);
        };
        const total = 2 * r_pad + s_n;
        try ctx.uploadRecorded(self.bufs.pack, 0, std.mem.sliceAsBytes(self.pack[0..total]));
        const routes: Routes = .{
            .rows = s_n,
            .ids = try ctx.view(self.bufs.pack, 0, s_n * 4),
            .scales = try ctx.view(self.bufs.pack, r_pad * 4, s_n * 4),
            .slots = try ctx.view(self.bufs.pack, 2 * r_pad * 4, s_n * 4),
        };
        try ctx.opElt(.gather_rows, x, self.bufs.expert_in, routes.ids, null, .{ .u0 = @intCast(s_n * c.hidden), .u1 = @intCast(c.hidden) }, s_n * c.hidden, 1, 1);
        return routes;
    }

    /// Rows `[o, o + n)` of a packed buffer `width` floats wide.
    fn rowsOf(self: *VulkanLM, b: Buf, o: usize, n: usize, width: usize) !Buf {
        return self.be.view(b, o * width * 4, n * width * 4);
    }

    /// Whether a batch of `seq` takes the device-routed block over `groups`: at most
    /// `device_route_max` rows, with each expert tensor one bindable group and nothing (weight noise, a LoRA
    /// sidecar) that the per-expert linears would add.
    fn deviceRouted(self: *const VulkanLM, seq: usize, groups: []const []const Weight) bool {
        if (device_route_off or seq > device_route_max or self.be.weight_noise.on() or self.lin_plan.lora != null) return false;
        for (groups) |ws| {
            const g = groupBytes(ws) orelse return false;
            if (g.len > lin_gpu.max_binding or !gpu.Context.dualGemv(ws[0].dtype)) return false;
        }
        return true;
    }

    /// Diagnostic: route every batch on the host, expert by expert.
    pub var device_route_off = false;

    /// The q8 activation where the per-expert linears would stage one, which keeps every
    /// routed row bit-identical to them.
    fn routedQ8(self: *const VulkanLM, dt: @import("tp_core").dtype.DType) bool {
        return self.lin.caps.dualQ8(dt);
    }

    /// Route on the device into `bufs.moe`: no download, and no route upload.
    fn routeOnDevice(self: *VulkanLM, router: Weight, bias: []const f32, seq: usize, n_experts: usize, used: usize) !void {
        const c = self.cfg;
        const b = &self.bufs;
        try self.lin.linear(self.lin_plan, b.route_logits, 0, b.normed, seq, router);
        const gating: gpu.Context.MoeGating = switch (c.gating) {
            .sigmoid => .sigmoid,
            .softmax => .softmax,
        };
        try self.be.opMoeRoute(b.route_logits, try nbuf(self.be, bias), b.moe, seq, n_experts, used, gating, c.normalize_weights, c.expert_scale, 0, Moe.of(c).cap);
    }

    /// `y[s] = W_{e_s} x[s / x_div]` for the batch's slots, into `bufs.moe` at word `off`.
    /// `x` is f32; the q8 form reads it as `stage` left it staged.
    fn routedLinear(self: *VulkanLM, ws: []const Weight, off: usize, x: Buf, slots: usize, x_div: usize) !void {
        const w = ws[0];
        try self.be.opGemvExperts(w.dtype, self.bufs.moe, off, if (self.routedQ8(w.dtype)) null else x, groupBytes(ws).?, w.scale, w.rows, w.cols, slots, x_div);
    }

    /// The q8 staging `routedLinear` over `ws` reads, if it reads one and `held` is not
    /// already it.
    fn stage(self: *VulkanLM, ws: []const Weight, x: Buf, rows: usize, held: ?lin_gpu.Group) !void {
        if (self.routedQ8(ws[0].dtype) and !self.lin.holdsQ8(held, x, rows)) _ = try self.lin.stageQ8(x, rows, ws[0].cols);
    }

    fn moeView(self: *VulkanLM, off: usize, words: usize) !Buf {
        return self.be.view(self.bufs.moe, off * 4, words * 4);
    }

    fn routedValuesDevice(self: *VulkanLM, values: k2.Values, seq: usize) !void {
        const c = self.cfg;
        const b = &self.bufs;
        const m = Moe.of(c);
        const used = c.n_value_experts_used;
        const slots = seq * used;
        const kv = c.kvDim();
        try self.routeOnDevice(values.router, values.bias, seq, c.n_value_experts, used);
        // The attention projections' staging of the same `normed`, if it is still in place.
        try self.stage(values.weights, b.normed, seq, self.attn_stage);
        try self.routedLinear(values.weights, m.a, b.normed, slots, used);
        const ya = try self.moeView(m.a, slots * kv);
        try self.be.opElt(gpu.eltAt("silu", .f32), ya, null, null, null, .{ .u0 = @intCast(slots * kv) }, slots * kv, 1, 1);
        try self.be.opElt(.moe_combine, b.v, ya, b.slots, try self.moeView(m.cap, slots), .{ .u0 = @intCast(seq * kv), .u1 = @intCast(kv), .u2 = @intCast(used) }, seq * kv, 1, 1);
    }

    fn routedFfnDevice(self: *VulkanLM, experts: k2.Experts, seq: usize) !void {
        const c = self.cfg;
        const b = &self.bufs;
        const ctx = self.be;
        const m = Moe.of(c);
        const used = c.n_experts_used;
        const slots = seq * used;
        const inter = c.expert_intermediate;
        try self.routeOnDevice(experts.router, experts.bias, seq, c.n_experts, used);
        // The shared gate and up first: their staging of `normed` is the routed pair's too.
        var held: ?lin_gpu.Group = null;
        if (experts.shared_gate) |gate| {
            const ws = [2]Weight{ gate, experts.shared_up.? };
            const g = try self.lin.prep(self.lin_plan, b.normed, seq, c.hidden, &ws);
            try self.lin.gemm(self.lin_plan, g, b.expert_a, ws[0], .{});
            try self.lin.gemm(self.lin_plan, g, b.expert_b, ws[1], .{});
            held = g;
        }
        // Gate and up share a dtype in one GGUF, and so one staging.
        std.debug.assert(experts.gate[0].dtype == experts.up[0].dtype);
        try self.stage(experts.gate, b.normed, seq, held);
        try self.routedLinear(experts.gate, m.a, b.normed, slots, used);
        try self.routedLinear(experts.up, m.b, b.normed, slots, used);
        const ya = try self.moeView(m.a, slots * inter);
        try ctx.opElt(gpu.eltAt("silu_mul", .f32), ya, try self.moeView(m.b, slots * inter), null, null, .{ .u0 = @intCast(slots * inter) }, slots * inter, 1, 1);
        // The f32 GEMV reads its rows from a binding of its own, never from the buffer it
        // writes; `expert_a` holds the shared expert's gate by now.
        const down_x = if (self.routedQ8(experts.down[0].dtype)) ya else blk: {
            const n = slots * inter;
            try ctx.opElt(.copy, ya, b.expert_in, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
            break :blk b.expert_in;
        };
        try self.stage(experts.down, down_x, slots, null);
        try self.routedLinear(experts.down, m.out, down_x, slots, 1);
        try ctx.opElt(.moe_combine, b.tmp, try self.moeView(m.out, slots * c.hidden), b.slots, try self.moeView(m.cap, slots), .{ .u0 = @intCast(seq * c.hidden), .u1 = @intCast(c.hidden), .u2 = @intCast(used) }, seq * c.hidden, 1, 1);
        if (experts.shared_gate != null) try self.sharedDown(experts, seq);
    }

    fn routedValues(self: *VulkanLM, values: k2.Values, seq: usize) !void {
        if (self.deviceRouted(seq, &.{values.weights})) return self.routedValuesDevice(values, seq);
        const c = self.cfg;
        const b = &self.bufs;
        const ctx = self.be;
        try self.lin.linear(self.lin_plan, b.route_logits, 0, b.normed, seq, values.router);
        try self.selectRoutes(seq, values.bias, c.n_value_experts, c.n_value_experts_used);
        const used = c.n_value_experts_used;
        const routes = try self.packRoutes(b.normed, seq, used, c.n_value_experts);
        const kv = c.kvDim();
        for (values.weights, 0..) |w, e| {
            const n = self.route_counts[e];
            if (n == 0) continue;
            const o = self.route_starts[e];
            const ya = try self.rowsOf(b.expert_a, o, n, kv);
            try self.lin.linear(self.lin_plan, ya, 0, try self.rowsOf(b.expert_in, o, n, c.hidden), n, w);
            try ctx.opElt(gpu.eltAt("silu", .f32), ya, null, null, null, .{ .u0 = @intCast(n * kv) }, n * kv, 1, 1);
        }
        try ctx.opElt(.moe_combine, b.v, b.expert_a, routes.slots, routes.scales, .{ .u0 = @intCast(seq * kv), .u1 = @intCast(kv), .u2 = @intCast(used) }, seq * kv, 1, 1);
    }

    fn routedFfn(self: *VulkanLM, experts: k2.Experts, seq: usize) !void {
        if (self.deviceRouted(seq, &.{ experts.gate, experts.up, experts.down })) return self.routedFfnDevice(experts, seq);
        const c = self.cfg;
        const b = &self.bufs;
        const ctx = self.be;
        try self.lin.linear(self.lin_plan, b.route_logits, 0, b.normed, seq, experts.router);
        try self.selectRoutes(seq, experts.bias, c.n_experts, c.n_experts_used);
        const used = c.n_experts_used;
        const routes = try self.packRoutes(b.normed, seq, used, c.n_experts);
        const inter = c.expert_intermediate;
        for (experts.gate, experts.up, experts.down, 0..) |g, u, dn, e| {
            const n = self.route_counts[e];
            if (n == 0) continue;
            const o = self.route_starts[e];
            const xin = try self.rowsOf(b.expert_in, o, n, c.hidden);
            const ya = try self.rowsOf(b.expert_a, o, n, inter);
            const yb = try self.rowsOf(b.expert_b, o, n, inter);
            try self.lin.linear(self.lin_plan, ya, 0, xin, n, g);
            try self.lin.linear(self.lin_plan, yb, 0, xin, n, u);
            try ctx.opElt(gpu.eltAt("silu_mul", .f32), ya, yb, null, null, .{ .u0 = @intCast(n * inter) }, n * inter, 1, 1);
            try self.lin.linear(self.lin_plan, try self.rowsOf(b.expert_out, o, n, c.hidden), 0, ya, n, dn);
        }
        try ctx.opElt(.moe_combine, b.tmp, b.expert_out, routes.slots, routes.scales, .{ .u0 = @intCast(seq * c.hidden), .u1 = @intCast(c.hidden), .u2 = @intCast(used) }, seq * c.hidden, 1, 1);
        try self.sharedExperts(experts, seq);
    }

    fn sharedExperts(self: *VulkanLM, experts: k2.Experts, seq: usize) !void {
        const gate = experts.shared_gate orelse return;
        const b = &self.bufs;
        try self.lin.linear(self.lin_plan, b.expert_a, 0, b.normed, seq, gate);
        try self.lin.linear(self.lin_plan, b.expert_b, 0, b.normed, seq, experts.shared_up.?);
        try self.sharedDown(experts, seq);
    }

    /// The shared expert past its gate and up projections (in `expert_a` and `expert_b`),
    /// added to `bufs.tmp`.
    fn sharedDown(self: *VulkanLM, experts: k2.Experts, seq: usize) !void {
        const c = self.cfg;
        const b = &self.bufs;
        const ctx = self.be;
        const n = seq * c.shared_intermediate;
        try ctx.opElt(gpu.eltAt("silu_mul", .f32), b.expert_a, b.expert_b, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
        try self.lin.linear(self.lin_plan, b.expert_out, 0, b.expert_a, seq, experts.shared_down.?);
        try ctx.opElt(gpu.eltAt("add", .f32), b.tmp, b.expert_out, null, null, .{ .u0 = @intCast(seq * c.hidden) }, seq * c.hidden, 1, 1);
    }

    /// A host layer's routes and its copy of `bufs.normed`, from one wait.
    fn routesAndNormed(self: *VulkanLM, seq: usize, bias: []const f32, n_experts: usize, n_used: usize) ![]f32 {
        const logits = self.route_logits[0 .. seq * n_experts];
        const x = self.cpu_scratch.normed[0 .. seq * self.cfg.hidden];
        try self.be.downloadMany(&.{
            .{ .db = self.bufs.route_logits, .out = std.mem.sliceAsBytes(logits) },
            .{ .db = self.bufs.normed, .out = std.mem.sliceAsBytes(x) },
        });
        k2.route(self.cfg, logits, bias, seq, n_experts, n_used, self.selected[0 .. seq * n_used], self.selected_weights[0 .. seq * n_used]);
        return x;
    }

    /// Time the first host MoE layer both ways (`k2.HostCost`): its routed blocks on the
    /// CPU, and its groups' upload. Nothing to time without a host layer.
    fn calibrateHost(self: *VulkanLM) !void {
        const l = for (self.lm.layers, 0..) |layer, l| {
            if (self.host_experts[l] and layer.experts != null) break l;
        } else return;
        const layer = self.lm.layers[l];
        const io = self.io orelse self.be.io;
        const cpu = try k2.HostCost.cpuRowNs(io, self.gpa, self.cfg, layer, &self.cpu_scratch, @min(k2.host_calibration_rows, self.cpu_rows));
        const t0 = std.Io.Clock.awake.now(self.be.io).nanoseconds;
        var gs: [4][]const u8 = undefined;
        for (layerGroups(layer, &gs)) |g| try self.be.warmWeightGroup(g);
        const up = std.Io.Clock.awake.now(self.be.io).nanoseconds - t0;
        for (layerGroups(layer, &gs)) |g| self.be.evictWeightBytes(g);
        self.host_cost = .{ .cpu_row_ns = cpu, .upload_ns = @floatFromInt(up) };
        std.log.info("[offload] k2 host layers: {d:.2} ms a row on the CPU, {d:.1} ms a layer to upload", .{ cpu / 1e6, @as(f64, @floatFromInt(up)) / 1e6 });
    }

    fn cpuRoutedValues(self: *VulkanLM, values: k2.Values, seq: usize) !void {
        const c = self.cfg;
        std.debug.assert(seq <= self.cpu_rows);
        // The context's own Io outside a step (`warmWeights`).
        const io = self.io orelse self.be.io;
        try self.lin.linear(self.lin_plan, self.bufs.route_logits, 0, self.bufs.normed, seq, values.router);
        const x = try self.routesAndNormed(seq, values.bias, c.n_value_experts, c.n_value_experts_used);
        const y = self.cpu_scratch.v[0 .. seq * c.kvDim()];
        try k2.routedValuesSelected(io, self.gpa, c, values, x, seq, y, &self.cpu_scratch, self.selected[0 .. seq * c.n_value_experts_used], self.selected_weights[0 .. seq * c.n_value_experts_used]);
        try self.be.uploadRecorded(self.bufs.v, 0, std.mem.sliceAsBytes(y));
    }

    fn cpuRoutedFfn(self: *VulkanLM, experts: k2.Experts, seq: usize) !void {
        const c = self.cfg;
        std.debug.assert(seq <= self.cpu_rows);
        // The context's own Io outside a step (`warmWeights`).
        const io = self.io orelse self.be.io;
        try self.lin.linear(self.lin_plan, self.bufs.route_logits, 0, self.bufs.normed, seq, experts.router);
        const x = try self.routesAndNormed(seq, experts.bias, c.n_experts, c.n_experts_used);
        const y = self.cpu_scratch.tmp[0 .. seq * c.hidden];
        try k2.routedFfnSparseSelected(io, self.gpa, c, experts, x, seq, y, &self.cpu_scratch, self.selected[0 .. seq * c.n_experts_used], self.selected_weights[0 .. seq * c.n_experts_used]);
        try self.be.uploadRecorded(self.bufs.tmp, 0, std.mem.sliceAsBytes(y));
        try self.sharedExperts(experts, seq);
    }

    /// `k2_horizon_dev`'s adapter. `upload` is the current layer's: a host layer whose
    /// groups come to the device for this batch, and leave after it.
    const Dev = struct {
        s: *VulkanLM,
        cfg: k2.Config,
        bufs: *Bufs,
        upload: bool = false,

        pub fn groupNorm(d: *Dev, x: Buf, out: Buf, w: []const f32, seq: usize) !void {
            const c = d.cfg;
            const ctx = d.s.be;
            try ctx.opElt(.group_rmsnorm, x, out, try nbuf(ctx, w), null, .{
                .u0 = @intCast(seq * c.norm_groups),
                .u1 = @intCast(c.hidden / c.norm_groups),
                .u2 = @intCast(c.norm_groups),
                .f0 = c.rms_eps,
            }, seq * c.norm_groups, 1, 1);
        }
        pub fn linear(d: *Dev, y: Buf, x: Buf, seq: usize, w: Weight) !void {
            try d.s.lin.linear(d.s.lin_plan, y, 0, x, seq, w);
        }
        /// One staging of `x` for every weight, held for the value experts.
        pub fn linears(d: *Dev, x: Buf, seq: usize, ys: []const Buf, ws: []const Weight) !void {
            const g = try d.s.lin.prep(d.s.lin_plan, x, seq, ws[0].cols, ws);
            for (ys, ws) |y, w| try d.s.lin.gemm(d.s.lin_plan, g, y, w, .{});
            d.s.attn_stage = g;
        }
        pub fn siluMul(d: *Dev, a: Buf, b: Buf, n: usize) !void {
            try d.s.be.opElt(gpu.eltAt("silu_mul", .f32), a, b, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
        }
        pub fn add(d: *Dev, dst: Buf, src: Buf, n: usize) !void {
            try d.s.be.opElt(gpu.eltAt("add", .f32), dst, src, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
        }
        pub fn layerBegin(d: *Dev, l: usize, seq: usize) !void {
            d.upload = seq > cpu_small_max and d.s.host_experts[l] and !d.s.host_cost.?.onCpu(seq);
        }
        pub fn values(d: *Dev, l: usize, v: k2.Values, seq: usize) !void {
            if (d.s.host_experts[l] and !d.upload)
                try d.s.cpuRoutedValues(v, seq)
            else
                try d.s.routedValues(v, seq);
        }
        pub fn rope(d: *Dev, seq: usize, pos0: usize) !void {
            const c = d.cfg;
            const b = d.bufs;
            const half = c.head_dim / 2;
            for ([2]Buf{ b.q, b.k }, [2]usize{ c.n_heads, c.n_kv_heads }) |buf, heads| {
                try d.s.be.opElt(.rope_half, buf, null, d.s.freqs_d, null, .{
                    .u0 = @intCast(seq * heads * half),
                    .u1 = @intCast(half),
                    .u2 = d.s.sin_off,
                    .u3 = @intCast(heads),
                    .u4 = @intCast(pos0),
                }, seq * heads * half, 1, 1);
            }
        }
        pub fn storeKv(d: *Dev, l: usize, seq: usize, pos0: usize) !void {
            try d.s.kv.store(d.s.be, l, d.bufs.k, d.bufs.v, pos0, seq);
        }
        pub fn attention(d: *Dev, l: usize, seq: usize, pos0: usize) !void {
            const c = d.cfg;
            const ns: usize = if (seq <= cpu_small_max) nsplit else nsplit_prefill;
            const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.head_dim)));
            try d.s.kv.attend(d.s.be, l, d.bufs.q, d.bufs.attn, d.bufs.attn_scratch, pos0 + 1, seq, c.n_heads, c.n_kv_heads, c.head_dim, ns, scale, 0, 0);
        }
        pub fn softplusGate(d: *Dev, n: usize) !void {
            try d.s.be.opElt(.softplus_gate, d.bufs.attn, d.bufs.gate, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
        }
        pub fn experts(d: *Dev, l: usize, e: k2.Experts, seq: usize) !void {
            if (d.s.host_experts[l] and !d.upload) return d.s.cpuRoutedFfn(e, seq);
            try d.s.routedFfn(e, seq);
            if (d.upload) {
                // The groups go back to the host once the batch's reads are done.
                try d.s.be.endBatch();
                var gs: [4][]const u8 = undefined;
                for (layerGroups(d.s.lm.layers[l], &gs)) |g| d.s.be.evictWeightBytes(g);
                try d.s.be.beginBatch();
            }
        }
    };
};

/// The device-routed block's buffer (`bufs.moe`), in words: each slot's expert id, its
/// weight, then the gate (or value) rows, the up rows and the down rows. The ids share the
/// buffer the GEMVs write, the one binding `opGemvExperts` has free for them.
const Moe = struct {
    cap: usize,
    a: usize,
    b: usize,
    out: usize,
    words: usize,

    const max_cap = 1024;

    fn of(c: k2.Config) Moe {
        const cap = Bufs.padSlots(device_route_max * @max(c.n_experts_used, c.n_value_experts_used));
        std.debug.assert(cap <= max_cap);
        const a = 2 * cap;
        const b = a + cap * @max(c.expert_intermediate, c.kvDim());
        const out = b + cap * c.expert_intermediate;
        return .{ .cap = cap, .a = a, .b = b, .out = out, .words = out + cap * c.hidden };
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
    last: Buf,
    route_logits: Buf,
    pack: Buf,
    expert_in: Buf,
    expert_a: Buf,
    expert_b: Buf,
    expert_out: Buf,
    attn_scratch: Buf,
    logits: Buf,
    argmax_v: Buf,
    argmax_i: Buf,
    argmax_out: Buf,
    topk_v: Buf,
    topk_i: Buf,
    moe: Buf,
    slots: Buf,

    const fields = [_][]const u8{ "x", "normed", "q", "k", "v", "attn", "gate", "tmp", "last", "route_logits", "pack", "expert_in", "expert_a", "expert_b", "expert_out", "attn_scratch", "logits", "argmax_v", "argmax_i", "argmax_out", "topk_v", "topk_i", "moe", "slots" };

    /// A packed route array's slots, padded so the next one starts bindable.
    fn padSlots(n: usize) usize {
        return std.mem.alignForward(usize, n, 64);
    }

    fn packSlots(c: k2.Config) usize {
        const s_n = prefill_chunk * @max(c.n_experts_used, c.n_value_experts_used);
        return 2 * padSlots(s_n) + s_n;
    }

    fn init(ctx: *gpu.Context, c: k2.Config) !Bufs {
        const r = prefill_chunk;
        const routed = r * @max(c.n_experts_used, c.n_value_experts_used);
        const inner = @max(r * @max(c.intermediate, c.shared_intermediate), routed * @max(c.expert_intermediate, c.kvDim()));
        const sizes = [fields.len]usize{
            r * c.hidden, // x
            r * c.hidden, // normed
            r * c.qDim(), // q
            r * c.kvDim(), // k
            r * c.kvDim(), // v
            r * c.qDim(), // attn
            r * c.qDim(), // gate
            r * c.hidden, // tmp
            c.hidden, // last
            r * @max(c.n_experts, c.n_value_experts), // route_logits
            packSlots(c), // pack
            routed * c.hidden, // expert_in
            inner, // expert_a
            inner, // expert_b
            @max(routed, r) * @max(c.hidden, c.kvDim()), // expert_out
            r * nsplit_prefill * c.n_heads * (c.head_dim + 2) + cpu_small_max * nsplit * c.n_heads * (c.head_dim + 2), // attn_scratch
            c.vocab, // logits
            4096, // argmax_v
            4096, // argmax_i
            1, // argmax_out
            gpu.topk_lanes * gpu.topk_m, // topk_v
            gpu.topk_lanes * gpu.topk_m, // topk_i
            Moe.of(c).words, // moe
            Moe.of(c).cap, // slots
        };
        var self: Bufs = undefined;
        var created: usize = 0;
        errdefer inline for (fields, 0..) |name, i| {
            if (i < created) ctx.tensorDestroy(&@field(self, name));
        };
        inline for (fields, sizes) |name, size| {
            @field(self, name) = try ctx.tensorCreate(size * 4);
            created += 1;
        }
        var ids: [Moe.max_cap]u32 = undefined;
        for (&ids, 0..) |*v, i| v.* = @intCast(i);
        try ctx.tensorUpload(self.slots, std.mem.sliceAsBytes(ids[0..sizes[sizes.len - 1]]));
        return self;
    }

    fn deinit(self: *Bufs, ctx: *gpu.Context) void {
        inline for (fields) |name| ctx.tensorDestroy(&@field(self, name));
        self.* = undefined;
    }
};

const build_options = @import("build_options");
const test_path = build_options.model_root ++ "/../llm/K2-Horizon-MoVA-36B-A4B-Q6_K.gguf";

// A decode block routed on the device runs every row the per-expert one does, bit for
// bit (fed the host's routes, the logits match to the byte); only the route weights'
// rounding may move them. On exact routes, since a q8 one turns that rounding into
// noise (5e-3).
test "vulkan k2 device-routed decode is the per-expert decode" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const f = try Fixture.open(gpa, io);
    defer f.close(gpa);
    const prompt = try testPrompt(gpa, 12);
    defer gpa.free(prompt);
    const vocab = f.lm.cfg.vocab;
    var bufs: [2][]f32 = undefined;
    for (&bufs) |*b| b.* = try gpa.alloc(f32, vocab);
    defer for (bufs) |b| gpa.free(b);
    var m = try VulkanLM.init(gpa, f.ctx, &f.lm, .{ .initial = 32, .max = 32 });
    defer m.deinit();
    m.warmWeights();
    defer VulkanLM.device_route_off = false;
    const sdot_prev = gpu.Context.sdot_off;
    defer gpu.Context.sdot_off = sdot_prev;
    gpu.Context.sdot_off = true;
    for (&bufs, [2]bool{ true, false }) |out, off| {
        VulkanLM.device_route_off = off;
        try m.resetCache();
        try m.step(io, prompt[0 .. prompt.len - 1], out);
        try m.step(io, prompt[prompt.len - 1 ..], out);
    }
    const rel = relL2(bufs[0], bufs[1]);
    errdefer std.debug.print("device-routed vs per-expert decode: rel L2 {e:.3}\n", .{rel});
    try std.testing.expect(rel < 1e-4);
    try std.testing.expectEqual(sample.argmax(bufs[0]), sample.argmax(bufs[1]));
}

// The device route must pick exactly the host's experts, in its order, a planted tie
// included; its weights only match to rounding, since the two `exp`s differ.
test "vulkan k2 routes on the device as k2_horizon.route does" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    const ctx = gpu.Context.init(gpa, io) catch return error.SkipZigTest;
    defer ctx.deinit();
    var prng = std.Random.DefaultPrng.init(9);
    const rnd = prng.random();
    const seq = 5;
    for ([_]k2.Config.Gating{ .sigmoid, .softmax }) |gating| for ([2][2]usize{ .{ 100, 8 }, .{ 64, 4 } }) |shape| {
        const n = shape[0];
        const used = shape[1];
        var cfg = std.mem.zeroes(k2.Config);
        cfg.gating = gating;
        cfg.normalize_weights = true;
        cfg.expert_scale = 2.5;
        var logits: [seq * 100]f32 = undefined;
        var bias: [100]f32 = undefined;
        for (logits[0 .. seq * n]) |*v| v.* = rnd.floatNorm(f32);
        for (bias[0..n]) |*v| v.* = (rnd.float(f32) - 0.5) * 0.2;
        logits[5] = 4;
        logits[17] = 4;
        bias[5] = 0.05;
        bias[17] = 0.05;
        var dl = try ctx.tensorCreate(seq * n * 4);
        defer ctx.tensorDestroy(&dl);
        try ctx.tensorUpload(dl, std.mem.sliceAsBytes(logits[0 .. seq * n]));
        var db = try ctx.tensorCreate(n * 4);
        defer ctx.tensorDestroy(&db);
        try ctx.tensorUpload(db, std.mem.sliceAsBytes(bias[0..n]));
        const w_off = 64;
        var out = try ctx.tensorCreate(2 * w_off * 4);
        defer ctx.tensorDestroy(&out);
        try ctx.opMoeRoute(dl, db, out, seq, n, used, if (gating == .sigmoid) .sigmoid else .softmax, true, 2.5, 0, w_off);
        var words: [2 * w_off]u32 = undefined;
        try ctx.tensorDownload(out, std.mem.sliceAsBytes(&words));
        var sel: [seq * 8]usize = undefined;
        var wts: [seq * 8]f32 = undefined;
        k2.route(cfg, logits[0 .. seq * n], bias[0..n], seq, n, used, sel[0 .. seq * used], wts[0 .. seq * used]);
        for (0..seq * used) |i| {
            const w_dev: f32 = @bitCast(words[w_off + i]);
            errdefer std.debug.print("{t} {d} experts, slot {d}: host {d} @ {d}, device {d} @ {d}\n", .{ gating, n, i, sel[i], wts[i], words[i], w_dev });
            try std.testing.expectEqual(sel[i], words[i]);
            try std.testing.expectApproxEqRel(wts[i], w_dev, 1e-5);
        }
        // The tie: the lower index first.
        try std.testing.expectEqual(@as(u32, 5), words[0]);
        try std.testing.expectEqual(@as(u32, 17), words[1]);
    };
}

/// A device and the test checkpoint, or a skip. Heap-held: the model borrows the
/// container's mapping.
const Fixture = struct {
    ctx: *gpu.Context,
    g: @import("tp_core").gguf.Gguf,
    lm: k2.Model,

    fn open(gpa: std.mem.Allocator, io: std.Io) !*Fixture {
        std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
        std.Io.Dir.cwd().access(io, test_path, .{}) catch return error.SkipZigTest;
        const f = try gpa.create(Fixture);
        errdefer gpa.destroy(f);
        f.ctx = gpu.Context.init(gpa, io) catch return error.SkipZigTest;
        errdefer f.ctx.deinit();
        f.g = try @TypeOf(f.g).open(gpa, io, test_path);
        errdefer f.g.deinit();
        f.lm = try k2.Model.load(gpa, &f.g);
        return f;
    }

    fn close(f: *Fixture, gpa: std.mem.Allocator) void {
        f.lm.deinit();
        f.g.deinit();
        f.ctx.deinit();
        gpa.destroy(f);
    }
};

fn testPrompt(gpa: std.mem.Allocator, n: usize) ![]u32 {
    const p = try gpa.alloc(u32, n);
    for (p, 0..) |*t, i| t.* = @intCast(1000 + (i * 37) % 50000);
    return p;
}

fn relL2(want: []const f32, got: []const f32) f64 {
    var num: f64 = 0;
    var den: f64 = 0;
    for (want, got) |a, b| {
        num += (@as(f64, a) - b) * (@as(f64, a) - b);
        den += @as(f64, a) * a;
    }
    return @sqrt(num / den);
}

/// The first MoE layer whose experts are on the device.
fn residentMoeLayer(m: *const VulkanLM) ?usize {
    for (m.lm.layers, 0..) |layer, l| if (layer.experts != null and !m.host_experts[l]) return l;
    return null;
}

// The device forward against the CPU one on the real model, a prefill and a decode
// step, with whatever placement the card forces: within twice the distance CUDA's
// own forward keeps from the CPU, where a CUDA device is present.
// Without the q8 GEMVs every route keeps exact activations, so the forward must match
// the CPU's to rounding. With them a 2-4 row expert batch quantizes too (CUDA dequants
// one instead), and a single row's logits then land anywhere in the format's noise,
// 0.02 to 0.27 on either backend across prompt lengths, close enough to flip a near tie.
test "vulkan k2 matches the CPU forward" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const f = try Fixture.open(gpa, io);
    defer f.close(gpa);
    const prev = ops.matmul.exact_activations;
    ops.matmul.exact_activations = true;
    defer ops.matmul.exact_activations = prev;
    const prompt = try testPrompt(gpa, 12);
    defer gpa.free(prompt);
    const vocab = f.lm.cfg.vocab;
    var bufs: [4][]f32 = undefined;
    for (&bufs) |*b| b.* = try gpa.alloc(f32, vocab);
    defer for (bufs) |b| gpa.free(b);

    {
        var cpu = try k2.CpuModel.init(gpa, &f.lm, .{ .initial = 32, .max = 32 });
        defer cpu.deinit();
        try cpu.step(io, prompt[0 .. prompt.len - 1], bufs[0]);
        try cpu.step(io, prompt[prompt.len - 1 ..], bufs[1]);
    }
    var m = try VulkanLM.init(gpa, f.ctx, &f.lm, .{ .initial = 32, .max = 32 });
    defer m.deinit();
    m.warmWeights();
    const sdot_prev = gpu.Context.sdot_off;
    defer gpu.Context.sdot_off = sdot_prev;
    for ([2]bool{ true, false }) |exact| {
        gpu.Context.sdot_off = exact;
        try m.resetCache();
        try m.step(io, prompt[0 .. prompt.len - 1], bufs[2]);
        try m.step(io, prompt[prompt.len - 1 ..], bufs[3]);
        for ([2]usize{ 0, 1 }) |i| {
            const rel = relL2(bufs[i], bufs[i + 2]);
            errdefer std.debug.print("{s}, {s}: vulkan vs cpu logits rel L2 {e:.3}\n", .{ if (exact) "exact" else "q8", if (i == 0) "prefill" else "decode", rel });
            try std.testing.expect(rel < if (exact) K2_EXACT_TOL else K2_Q8_TOL);
            if (exact) try std.testing.expectEqual(sample.argmax(bufs[i]), sample.argmax(bufs[i + 2]));
        }
    }
}
const K2_EXACT_TOL: f64 = 1e-4;
const K2_Q8_TOL: f64 = 0.4;

// One MoE layer's experts resident, then on the host: a batch past `cpu_small_max` that
// uploads the layer runs the very kernels it ran resident, so its logits must not move a
// bit; a decode step runs them on the CPU and only tracks. Promoted back, the layer is
// exactly resident again. On exact routes a batch the CPU takes matches resident too.
test "vulkan k2 moves an expert layer to the host and back" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const f = try Fixture.open(gpa, io);
    defer f.close(gpa);
    const prompt = try testPrompt(gpa, 12);
    defer gpa.free(prompt);
    const vocab = f.lm.cfg.vocab;
    var bufs: [6][]f32 = undefined;
    for (&bufs) |*b| b.* = try gpa.alloc(f32, vocab);
    defer for (bufs) |b| gpa.free(b);

    var m = try VulkanLM.init(gpa, f.ctx, &f.lm, .{ .initial = 32, .max = 32 });
    defer m.deinit();
    m.warmWeights();
    // Every host layer uploads its batches, so the one moved there runs as the rest do.
    m.host_cost = .{ .cpu_row_ns = std.math.inf(f64), .upload_ns = 0 };
    const l = residentMoeLayer(&m) orelse return error.SkipZigTest;
    try m.step(io, prompt[0 .. prompt.len - 1], bufs[0]);
    try m.step(io, prompt[prompt.len - 1 ..], bufs[1]);

    try m.migrateLayer(l);
    try m.resetCache();
    try m.step(io, prompt[0 .. prompt.len - 1], bufs[2]);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(bufs[0]), std.mem.sliceAsBytes(bufs[2]));
    try m.step(io, prompt[prompt.len - 1 ..], bufs[3]);
    const rel = relL2(bufs[1], bufs[3]);
    errdefer std.debug.print("decode with layer {d} on the host: rel L2 {e:.3}\n", .{ l, rel });
    try std.testing.expect(rel < K2_HOST_TOL);
    try std.testing.expectEqual(sample.argmax(bufs[1]), sample.argmax(bufs[3]));

    try m.promoteLayer(l);
    try m.resetCache();
    try m.step(io, prompt[0 .. prompt.len - 1], bufs[4]);
    try m.step(io, prompt[prompt.len - 1 ..], bufs[5]);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(bufs[0]), std.mem.sliceAsBytes(bufs[4]));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(bufs[1]), std.mem.sliceAsBytes(bufs[5]));

    const exact_prev = ops.matmul.exact_activations;
    defer ops.matmul.exact_activations = exact_prev;
    ops.matmul.exact_activations = true;
    const sdot_prev = gpu.Context.sdot_off;
    defer gpu.Context.sdot_off = sdot_prev;
    gpu.Context.sdot_off = true;
    try m.resetCache();
    try m.step(io, prompt[0 .. prompt.len - 1], bufs[4]);
    try m.migrateLayer(l);
    m.host_cost = .{ .cpu_row_ns = 0, .upload_ns = 1 };
    try m.resetCache();
    try m.step(io, prompt[0 .. prompt.len - 1], bufs[5]);
    const rel_cpu = relL2(bufs[4], bufs[5]);
    errdefer std.debug.print("exact batch with layer {d} on the CPU: rel L2 {e:.3}\n", .{ l, rel_cpu });
    try std.testing.expect(rel_cpu < 1e-4);
}
const K2_HOST_TOL = 5e-2;

// What the GUI does between turns: a re-step after a truncate, a reset, the device
// samplers, and a KV dtype rebuild.
test "vulkan k2 gui turn lifecycle: truncate, reset, sampling, kv rebuild" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const f = try Fixture.open(gpa, io);
    defer f.close(gpa);
    const prompt = try testPrompt(gpa, 10);
    defer gpa.free(prompt);
    const vocab = f.lm.cfg.vocab;
    var bufs: [4][]f32 = undefined;
    for (&bufs) |*b| b.* = try gpa.alloc(f32, vocab);
    defer for (bufs) |b| gpa.free(b);

    var m = try VulkanLM.init(gpa, f.ctx, &f.lm, .{ .initial = 32, .max = 32 });
    defer m.deinit();
    m.warmWeights();
    // A decode step, then the same step again after a truncate.
    try m.step(io, prompt[0 .. prompt.len - 1], bufs[3]);
    try m.step(io, prompt[prompt.len - 1 ..], bufs[0]);
    m.truncate(prompt.len - 1);
    try m.step(io, prompt[prompt.len - 1 ..], bufs[1]);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(bufs[0]), std.mem.sliceAsBytes(bufs[1]));
    try m.resetCache();
    try m.step(io, prompt, bufs[0]);
    try m.resetCache();
    const best = sample.argmax(bufs[0]);
    const dev_best = try m.stepArgmax(io, prompt);
    errdefer std.debug.print("argmax: host {d}, device {d}\n", .{ best, dev_best });
    try std.testing.expectEqual(best, dev_best);
    try m.resetCache();
    const ids = try gpa.alloc(u32, m.maxSelect());
    defer gpa.free(ids);
    const lg = try gpa.alloc(f32, m.maxSelect());
    defer gpa.free(lg);
    // The candidates come back per lane, unsorted.
    const n = try m.stepSelect(io, prompt, ids, lg);
    try std.testing.expect(n > 0);
    const top = ids[std.sort.argMax(f32, lg[0..n], {}, std.sort.asc(f32)).?];
    errdefer std.debug.print("top-k: {d} kept, best {d}\n", .{ n, top });
    try std.testing.expectEqual(best, top);
    // Exact routes for both, device and host, or the q8 GEMVs turn the f16 rounding
    // into routing flips.
    const sdot_prev = gpu.Context.sdot_off;
    defer gpu.Context.sdot_off = sdot_prev;
    gpu.Context.sdot_off = true;
    const exact_prev = ops.matmul.exact_activations;
    defer ops.matmul.exact_activations = exact_prev;
    ops.matmul.exact_activations = true;
    try m.resetCache();
    try m.step(io, prompt, bufs[0]);
    try m.reinitCache(.f16);
    try m.step(io, prompt, bufs[2]);
    const rel = relL2(bufs[0], bufs[2]);
    errdefer std.debug.print("f16 kv: rel L2 {e:.3}\n", .{rel});
    try std.testing.expect(rel < 2e-2);
}

// A cache that grows mid-conversation decodes exactly what one sized up front does. The
// plan charges the cache, so the growing one is handed the full one's host layers.
test "vulkan k2 kv growth decodes what a full-size cache does" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const f = try Fixture.open(gpa, io);
    defer f.close(gpa);
    const prompt = try testPrompt(gpa, 6);
    defer gpa.free(prompt);
    var got: [2][24]u32 = undefined;
    const host = try gpa.alloc(bool, f.lm.cfg.n_layers);
    defer gpa.free(host);
    for ([2]usize{ 64, 8 }, 0..) |initial, run| {
        var m = try VulkanLM.init(gpa, f.ctx, &f.lm, .{ .initial = initial, .max = 64 });
        defer m.deinit();
        m.warmWeights();
        if (run == 0) @memcpy(host, m.host_experts) else for (host, 0..) |want, l| {
            const have = m.host_experts[l];
            if (want and !have) try m.migrateLayer(l);
            if (have and !want) try m.promoteLayer(l);
        }
        var cur = try m.stepArgmax(io, prompt);
        for (&got[run]) |*t| {
            try m.ensureCapacity(m.cached() + 1);
            cur = try m.stepArgmax(io, &.{cur});
            t.* = cur;
        }
    }
    try std.testing.expectEqualSlices(u32, &got[0], &got[1]);
}
