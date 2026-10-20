//! qwen35 hybrid (gated DeltaNet) on Vulkan: the forward is `qwen35_dev.zig`, shared
//! with `qwen35_cuda`, over this file's `Dev`, so a prompt prefills in batched chunks
//! (GEMM projections, one multi-query attention, the chunked recurrence) and a token
//! decodes as one recorded batch. The stepper answers what `qwen35_cuda`'s does: image
//! prefill at M-RoPE grid positions, a growable KV, turn checkpoints, the CPU split and
//! weight noise.
//!
//! The recurrence and the attention are the dual kernels (`dual/attn.zig`) in decode and
//! prefill alike, so a re-prefill reproduces the incremental state exactly.

const std = @import("std");
const qwen3 = @import("qwen3.zig");
const qwen35 = @import("qwen35.zig");
const qwen35_dev = @import("qwen35_dev.zig");
const lin_gpu = @import("lin_gpu.zig");
const gpu = @import("tp_gpu").context;
const ops = @import("tp_ops");
const kvmod = @import("tp_core").kv_cache;
const kv_gpu = @import("kv_gpu.zig");
const init_defaults = @import("tp_core").init_defaults;
const sample = @import("tp_core").sample;
const bnd = @import("tp_runtime").boundary;
const residency = @import("tp_runtime").residency;
const build_options = @import("build_options");
const cuda = @import("tp_gpu").cuda;
const qwen35_cuda = @import("qwen35_cuda.zig");

const Buf = gpu.DeviceBuffer;
const Weight = ops.matmul.Weight;

/// Upload a small f32 vector (norm weight) to a pointer-cached device buffer.
fn nbuf(ctx: *gpu.Context, w: []const f32) !Buf {
    return .{ .buf = try ctx.smallBuffer(std.mem.sliceAsBytes(w)), .mem = .null_handle, .size = 0 };
}

fn zeroBuf(ctx: *gpu.Context, gpa: std.mem.Allocator, buf: Buf, bytes: usize) !void {
    const zeros = try gpa.alloc(u8, bytes);
    defer gpa.free(zeros);
    @memset(zeros, 0);
    try ctx.tensorUpload(buf, zeros);
}

/// Rows per batched forward; every activation buffer holds this many.
const prefill_chunk = 512;
/// KV chunks per head in the attention split pass, decoding and prefilling.
const nsplit = 32;
const nsplit_prefill = 8;
/// Superblock chunks per output row in the k-split GEMV.
const gemv_nchunk = 16;

/// The activation buffers, by the names `qwen35_dev` reads.
const Bufs = struct {
    x: Buf,
    normed: Buf,
    qg: Buf,
    q: Buf,
    gate: Buf,
    k: Buf,
    v: Buf,
    attn: Buf,
    attn_scratch: Buf,
    t: Buf,
    lin_qkv: Buf,
    lin_conv: Buf,
    lin_z: Buf,
    lin_o: Buf,
    /// A decode step's `[alpha | beta]`.
    ab: Buf,
    gates: Buf,
    /// A batched chunk's alpha `[n][heads]` at 0 and beta at `prefill_chunk * heads`.
    ab_n: Buf,
    mlp_gate: Buf,
    mlp_up: Buf,
    logits: Buf,
    // On-device sampling scratch: argmax per-lane max and index and the one id out;
    // top-k per-lane values and indices.
    argmax_v: Buf,
    argmax_i: Buf,
    argmax_out: Buf,
    topk_v: Buf,
    topk_i: Buf,

    fn init(ctx: *gpu.Context, cfg: qwen35.Config) !Bufs {
        const r = prefill_chunk;
        const sizes = [_]usize{
            r * cfg.hidden,
            r * cfg.hidden,
            r * cfg.qDim() * 2,
            r * cfg.qDim(),
            r * cfg.qDim(),
            r * cfg.kvDim(),
            r * cfg.kvDim(),
            r * cfg.qDim(),
            @max(cfg.n_heads * nsplit, r * cfg.n_heads * nsplit_prefill) * (cfg.head_dim + 2),
            r * cfg.hidden,
            r * cfg.convChannels(),
            r * cfg.convChannels(),
            r * cfg.linVDim(),
            r * cfg.linVDim(),
            2 * cfg.lin_v_heads,
            r * 2 * cfg.lin_v_heads,
            2 * r * cfg.lin_v_heads,
            r * cfg.intermediate,
            r * cfg.intermediate,
            cfg.vocab,
            4096,
            4096,
            1,
            gpu.topk_lanes * gpu.topk_m,
            gpu.topk_lanes * gpu.topk_m,
        };
        var s: Bufs = undefined;
        var done: usize = 0;
        errdefer inline for (@typeInfo(Bufs).@"struct".fields, 0..) |f, i| {
            if (i < done) ctx.tensorDestroy(&@field(s, f.name));
        };
        inline for (@typeInfo(Bufs).@"struct".fields, 0..) |f, i| {
            @field(s, f.name) = try ctx.tensorCreate(sizes[i] * 4);
            done = i + 1;
        }
        return s;
    }

    fn deinit(s: *Bufs, ctx: *gpu.Context) void {
        inline for (@typeInfo(Bufs).@"struct".fields) |f| ctx.tensorDestroy(&@field(s, f.name));
    }
};

pub const VulkanLM = struct {
    /// The prefill batch this stepper wants the engine to hand it.
    pub const prefill_batch = prefill_chunk;
    /// `tp-llm --bench` alternates the `OpSDot` q8 decode GEMVs (toggle on: off) with
    /// the f32 ones.
    pub const bench_toggle = &gpu.Context.sdot_off;

    be: *gpu.Context,
    lm: *const qwen35.Model,
    cfg: qwen35.Config,
    gpa: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    /// Rows the attention KV holds now; `ensureCapacity` grows it toward
    /// `max_capacity`, and `resetResidency` shrinks it back to `initial_capacity`.
    capacity: usize,
    initial_capacity: usize,
    max_capacity: usize,
    kv_dtype: kvmod.KvDtype = .f32,
    len: usize,
    /// The next token's M-RoPE position: an image moves it by its larger grid side,
    /// not by its row count.
    pos_next: usize = 0,
    bufs: Bufs,
    lin: lin_gpu.Lin,
    lin_plan: lin_gpu.Plan,
    /// One K and V per attention slot.
    kv: kv_gpu.Kv,
    /// Per-linear-layer recurrent state.
    conv_state: []Buf,
    ssm_state: []Buf,
    freqs_d: Buf,
    sin_off: usize,
    /// The step's (t, h, w) position triple, and a chunk's, one per row.
    pos3_d: Buf,
    pos3s_d: Buf,
    /// Per-linear-layer [a | dt_bias] host constants for the gates.
    a_dt: [][]f32,
    /// Polled before each prefill chunk: residency, pause and cancel (the GUI).
    boundary: ?bnd.Hook = null,
    /// The Io host layers of a split run on, set by every step.
    io: ?std.Io = null,
    /// Layers placed on the host, null when every layer is on the device.
    split: ?Split = null,

    /// The CPU split (`residency.zig` drives it), as qwen35_cuda's.
    pub const Split = struct {
        on_gpu: []bool,
        n_cpu: usize,
        policy: residency.CpuSplitPolicy,
        /// Host KV for the attention layers there (the device's byte layout) and
        /// conv/ssm for the DeltaNet ones; its length tracks `len`.
        state: qwen35.State,
        scratch: qwen35.Scratch,
        freqs: ops.rope.Freqs,
        /// The hidden while host layers run, a whole chunk of rows.
        hx: []f32,
        /// Whether the live hidden sits in `hx` rather than `bufs.x`.
        on_host: bool = false,
        /// Migrate more layers to the host as the KV grows.
        dynamic: bool = false,
        /// Migration order; `order[0..next)` are on the host.
        order: []usize = &.{},
        next: usize = 0,
        /// The device usage the dynamic split keeps under.
        budget: u64 = 0,
    };

    pub fn init(gpa: std.mem.Allocator, ctx: *gpu.Context, lm: *const qwen35.Model, cap: kvmod.Capacity) !VulkanLM {
        return initUse(gpa, ctx, lm, cap, lin_gpu.Use.llm(prefill_chunk));
    }

    /// `init` with the linear routes' policy given (see `qwen35_cuda.CudaLM.initUse`).
    pub fn initUse(gpa: std.mem.Allocator, ctx: *gpu.Context, lm: *const qwen35.Model, cap: kvmod.Capacity, use: lin_gpu.Use) !VulkanLM {
        const cfg = lm.cfg;
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();

        var self: VulkanLM = init_defaults.of(VulkanLM);
        self.be = ctx;
        self.lm = lm;
        self.cfg = cfg;
        self.gpa = gpa;
        self.capacity = @max(@min(cap.initial, cap.max), 1);
        self.initial_capacity = self.capacity;
        self.max_capacity = cap.max;
        self.kv_dtype = cap.kv_dtype;
        self.len = 0;
        self.pos_next = 0;
        self.boundary = null;
        self.io = null;
        self.split = null;
        ctx.weight_noise.setDepth(cfg.n_layers);

        var freqs = try ops.rope.rotateHalfFreqs(gpa, cap.max, cfg.rope_dim, cfg.rope_theta);
        defer freqs.deinit(gpa);
        const half = cfg.rope_dim / 2;
        const fp = try gpa.alloc(f32, 2 * cap.max * half);
        defer gpa.free(fp);
        @memcpy(fp[0 .. cap.max * half], freqs.cos);
        @memcpy(fp[cap.max * half ..], freqs.sin);
        self.sin_off = cap.max * half;
        self.freqs_d = try ctx.tensorCreate(fp.len * 4);
        try ctx.tensorUpload(self.freqs_d, std.mem.sliceAsBytes(fp));
        self.pos3_d = try ctx.tensorCreate(3 * 4);
        self.pos3s_d = try ctx.tensorCreate(prefill_chunk * 3 * 4);

        self.bufs = try Bufs.init(ctx, cfg);
        const max_out = @max(cfg.vocab, @max(cfg.convChannels(), @max(cfg.intermediate, cfg.qDim() * 2)));
        self.lin = try lin_gpu.Lin.init(gpa, ctx, max_out, gemv_nchunk, 1);
        errdefer self.lin.deinit();
        const lins = try lm.deviceLins(alloc);
        self.lin_plan = try self.lin.plan(lins, use, "qwen35 vulkan");
        // Sized for the batched routes, so not for the LM head: a prefill reads no
        // logits and a decode step runs it as a one-row GEMV, while its expansion would
        // be vocab x hidden of scratch.
        var batched: std.ArrayList(Weight) = .empty;
        for (lins) |w| if (w.bytes.ptr != lm.head.bytes.ptr) try batched.append(alloc, w);
        try lin_gpu.presize(ctx, self.lin_plan, batched.items);

        self.kv = try kv_gpu.Kv.init(gpa, ctx, cfg.nAttnLayers(), self.capacity, cfg.kvDim(), cap.kv_dtype, &.{});

        const n_lin = cfg.n_layers - cfg.nAttnLayers();
        self.conv_state = try alloc.alloc(Buf, n_lin);
        self.ssm_state = try alloc.alloc(Buf, n_lin);
        for (self.conv_state, self.ssm_state) |*cs, *ss| {
            cs.* = try ctx.tensorCreate(convBytes(cfg));
            ss.* = try ctx.tensorCreate(ssmBytes(cfg));
            try zeroBuf(ctx, gpa, cs.*, convBytes(cfg));
            try zeroBuf(ctx, gpa, ss.*, ssmBytes(cfg));
        }

        self.a_dt = try alloc.alloc([]f32, n_lin);
        var lin_idx: usize = 0;
        for (lm.layers) |*layer| switch (layer.*) {
            .linear => |*ll| {
                const buf = try alloc.alloc(f32, 2 * cfg.lin_v_heads);
                @memcpy(buf[0..cfg.lin_v_heads], ll.a);
                @memcpy(buf[cfg.lin_v_heads..], ll.dt_bias);
                self.a_dt[lin_idx] = buf;
                lin_idx += 1;
            },
            .attn => {},
        };

        self.arena = arena;
        return self;
    }

    fn convBytes(cfg: qwen35.Config) usize {
        return cfg.convChannels() * (cfg.conv_kernel - 1) * 4;
    }
    fn ssmBytes(cfg: qwen35.Config) usize {
        return cfg.lin_v_heads * cfg.lin_head_dim * cfg.lin_head_dim * 4;
    }

    pub fn deinit(self: *VulkanLM) void {
        const ctx = self.be;
        self.freeSplit();
        self.bufs.deinit(ctx);
        inline for (.{ "freqs_d", "pos3_d", "pos3s_d" }) |f| ctx.tensorDestroy(&@field(self, f));
        self.kv.deinit(self.gpa, ctx);
        for (self.conv_state) |*b| ctx.tensorDestroy(b);
        for (self.ssm_state) |*b| ctx.tensorDestroy(b);
        self.lin.deinit();
        self.arena.deinit();
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
    /// Device VRAM (bytes) this Vulkan context has allocated, the analog of the
    /// CUDA backend's `deviceUsed()`, for the end-of-response telemetry.
    pub fn vramUsed(self: *const VulkanLM) u64 {
        return self.be.device_used;
    }

    /// Grow the attention KV to hold `min_rows`, by the CUDA steppers' growth policy.
    /// A dynamic split first moves layers to the host until it can, and a real
    /// out-of-memory moves one more and retries.
    pub fn ensureCapacity(self: *VulkanLM, min_rows: usize) !void {
        if (min_rows <= self.capacity) return;
        if (min_rows > self.max_capacity) return error.ContextFull;
        const to = kvmod.growTarget(self.capacity, min_rows, self.max_capacity);
        if (self.split) |*sp| if (sp.dynamic) {
            const add = self.kv_dtype.sizeBytes((to - self.capacity) * self.cfg.kvDim());
            while (true) {
                const need = self.liveAttnSlots() * 2 * add + (32 << 20);
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
        if (self.split) |*sp| {
            sp.state.kv.grow(self.gpa, to) catch return error.ContextFull;
            sp.state.capacity = to;
            const freqs = ops.rope.rotateHalfFreqs(self.gpa, to, self.cfg.rope_dim, self.cfg.rope_theta) catch return error.ContextFull;
            sp.freqs.deinit(self.gpa);
            sp.freqs = freqs;
        }
        self.capacity = to;
    }

    /// An empty context: the recurrent state zeroed on whichever side holds it, the KV
    /// rows left to be overwritten.
    pub fn resetCache(self: *VulkanLM) !void {
        for (self.conv_state, self.ssm_state) |cs, ss| {
            try zeroBuf(self.be, self.gpa, cs, convBytes(self.cfg));
            try zeroBuf(self.be, self.gpa, ss, ssmBytes(self.cfg));
        }
        if (self.split) |*sp| {
            @memset(sp.state.conv, 0);
            @memset(sp.state.ssm, 0);
            sp.state.kv.truncate(0);
            sp.state.len = 0;
        }
        self.len = 0;
        self.pos_next = 0;
    }

    /// Re-create the KV at another element type, at the initial capacity, and empty
    /// the context: the caller re-prefills the transcript from zeroed recurrent state.
    pub fn reinitCache(self: *VulkanLM, dtype: kvmod.KvDtype) !void {
        const cfg = self.cfg;
        var kv = try kv_gpu.Kv.init(self.gpa, self.be, cfg.nAttnLayers(), self.initial_capacity, cfg.kvDim(), dtype, &.{});
        errdefer kv.deinit(self.gpa, self.be);
        if (self.split) |*sp| {
            for (sp.on_gpu, 0..) |on, l| if (!on and !cfg.isRecurrent(l)) kv.release(self.be, l / cfg.full_attn_interval);
            const state = try qwen35.State.init(self.gpa, cfg, self.initial_capacity, dtype);
            sp.state.deinit(self.gpa);
            sp.state = state;
        }
        self.kv.deinit(self.gpa, self.be);
        self.kv = kv;
        self.kv_dtype = dtype;
        self.capacity = self.initial_capacity;
        try self.resetCache();
    }

    /// A turn checkpoint's fixed size: the M-RoPE position and every DeltaNet layer's
    /// conv and ssm state. The attention KV is append-only and is truncated instead.
    pub fn checkpointBytes(self: *const VulkanLM) usize {
        const n_lin = self.cfg.n_layers - self.cfg.nAttnLayers();
        return @sizeOf(u64) + n_lin * (convBytes(self.cfg) + ssmBytes(self.cfg));
    }

    /// Snapshot the context state at the current position, each DeltaNet layer read
    /// from whichever side holds it, so a layer may move between snapshot and restore.
    pub fn checkpoint(self: *VulkanLM, out: []u8) !void {
        std.debug.assert(out.len == self.checkpointBytes());
        const pos: u64 = self.pos_next;
        @memcpy(out[0..8], std.mem.asBytes(&pos));
        try self.checkpointState(.save, out[8..]);
    }

    /// Roll back to `q` committed tokens with a snapshot taken there.
    pub fn restoreCheckpoint(self: *VulkanLM, snap: []const u8, q: usize) !void {
        std.debug.assert(snap.len == self.checkpointBytes() and q <= self.len);
        var pos: u64 = undefined;
        @memcpy(std.mem.asBytes(&pos), snap[0..8]);
        try self.checkpointState(.restore, snap[8..]);
        self.len = q;
        self.pos_next = @intCast(pos);
        if (self.split) |*sp| {
            sp.state.kv.truncate(q);
            sp.state.len = q;
        }
    }

    const CheckpointDir = enum { save, restore };

    /// The snapshot is every conv state, then every ssm state, linear-layer major.
    fn checkpointState(self: *VulkanLM, comptime dir: CheckpointDir, buf: if (dir == .save) []u8 else []const u8) !void {
        const cfg = self.cfg;
        const n_lin = cfg.n_layers - cfg.nAttnLayers();
        const cb = convBytes(cfg);
        const sb = ssmBytes(cfg);
        for (0..cfg.n_layers) |l| {
            if (!cfg.isRecurrent(l)) continue;
            const lin_idx = l - l / cfg.full_attn_interval;
            const bc = buf[lin_idx * cb ..][0..cb];
            const bs = buf[n_lin * cb + lin_idx * sb ..][0..sb];
            if (self.split == null or self.split.?.on_gpu[l]) {
                if (comptime dir == .save) {
                    try self.be.tensorDownload(self.conv_state[lin_idx], bc);
                    try self.be.tensorDownload(self.ssm_state[lin_idx], bs);
                } else {
                    try self.be.tensorUpload(self.conv_state[lin_idx], bc);
                    try self.be.tensorUpload(self.ssm_state[lin_idx], bs);
                }
            } else {
                const sp = &self.split.?;
                const hc = std.mem.sliceAsBytes(sp.state.conv[lin_idx * (cb / 4) ..][0 .. cb / 4]);
                const hs = std.mem.sliceAsBytes(sp.state.ssm[lin_idx * (sb / 4) ..][0 .. sb / 4]);
                if (comptime dir == .save) {
                    @memcpy(bc, hc);
                    @memcpy(bs, hs);
                } else {
                    @memcpy(hc, bc);
                    @memcpy(hs, bs);
                }
            }
        }
    }

    /// Forward `ids` (batched prefill for all but the last, one decode step for it) and
    /// write the last position's logits.
    pub fn step(self: *VulkanLM, io: std.Io, ids: []const u32, logits: []f32) !void {
        try self.forwardDecode(io, ids);
        try self.be.tensorDownload(self.bufs.logits, std.mem.sliceAsBytes(logits[0..self.cfg.vocab]));
    }

    /// The decode forward, leaving the last position's logits in `bufs.logits`.
    fn forwardDecode(self: *VulkanLM, io: std.Io, ids: []const u32) !void {
        self.io = io;
        std.debug.assert(ids.len >= 1 and ids.len <= self.remaining());
        if (ids.len > 1) try self.prefill(ids[0 .. ids.len - 1]);
        try self.stepOne(ids[ids.len - 1]);
    }

    /// Greedy decode without the vocab download: argmax on the device.
    pub fn stepArgmax(self: *VulkanLM, io: std.Io, ids: []const u32) !u32 {
        return self.stepArgmaxPen(io, ids, &.{}, .{});
    }

    /// `stepArgmax` with the sampling penalties scattered onto the device logits first.
    pub fn stepArgmaxPen(self: *VulkanLM, io: std.Io, ids: []const u32, pen: []const sample.PenaltyEntry, sp: sample.Params) !u32 {
        try self.forwardDecode(io, ids);
        const b = &self.bufs;
        try self.be.opPenalize(b.logits, pen, sp);
        try self.be.opArgmax(b.logits, self.cfg.vocab, b.argmax_out, &b.argmax_v, &b.argmax_i);
        var id_f: [1]f32 = undefined;
        try self.be.tensorDownload(b.argmax_out, std.mem.sliceAsBytes(&id_f));
        return @intFromFloat(id_f[0]);
    }

    /// Max candidates stepSelect can return (host buffer sizing for the engine).
    pub fn maxSelect(self: *const VulkanLM) usize {
        _ = self;
        return gpu.topk_lanes * gpu.topk_m;
    }

    /// Stochastic decode: the top-k candidates selected on the device and only those
    /// downloaded; the engine's Sampler finishes on the CPU.
    pub fn stepSelect(self: *VulkanLM, io: std.Io, ids: []const u32, out_id: []u32, out_logit: []f32) !usize {
        return self.stepSelectPen(io, ids, &.{}, .{}, out_id, out_logit);
    }

    /// `stepSelect` with the penalties applied before the top-k.
    pub fn stepSelectPen(self: *VulkanLM, io: std.Io, ids: []const u32, pen: []const sample.PenaltyEntry, sp: sample.Params, out_id: []u32, out_logit: []f32) !usize {
        try self.forwardDecode(io, ids);
        const b = &self.bufs;
        try self.be.opPenalize(b.logits, pen, sp);
        const count = try self.be.opTopK(b.logits, self.cfg.vocab, &b.topk_v, &b.topk_i);
        std.debug.assert(count <= out_id.len and count <= out_logit.len);
        try self.be.tensorDownload(b.topk_v, std.mem.sliceAsBytes(out_logit[0..count]));
        var idx_f: [gpu.topk_lanes * gpu.topk_m]f32 = undefined;
        try self.be.tensorDownload(b.topk_i, std.mem.sliceAsBytes(idx_f[0..count]));
        for (out_id[0..count], idx_f[0..count]) |*o, f| o.* = @intFromFloat(f);
        return count;
    }

    /// Prefill tokens without reading logits, in batched chunks.
    pub fn prefill(self: *VulkanLM, ids: []const u32) !void {
        const cfg = self.cfg;
        const x = try self.gpa.alloc(f32, @min(ids.len, prefill_chunk) * cfg.hidden);
        defer self.gpa.free(x);
        var pos3s: [prefill_chunk * 3]u32 = undefined;
        var off: usize = 0;
        while (off < ids.len) {
            try bnd.check(self);
            const n: usize = @min(prefill_chunk, ids.len - off);
            try qwen3.embedTokens(self.lm.embed, ids[off..][0..n], x[0 .. n * cfg.hidden]);
            for (0..n) |t| pos3s[t * 3 ..][0..3].* = @splat(@intCast(self.pos_next + t));
            try self.chunk(x[0 .. n * cfg.hidden], pos3s[0 .. n * 3], false);
            self.pos_next += n;
            off += n;
        }
    }

    /// The final-normed hidden state of every row of `ids`, prefilled at the cache's
    /// current length: what a text encoder reads of this model. `out` is
    /// `[ids.len][hidden]`.
    pub fn encode(self: *VulkanLM, ids: []const u32, out: []f32) !void {
        const hidden = self.cfg.hidden;
        std.debug.assert(out.len == ids.len * hidden);
        const x = try self.gpa.alloc(f32, @min(ids.len, prefill_chunk) * hidden);
        defer self.gpa.free(x);
        var pos3s: [prefill_chunk * 3]u32 = undefined;
        var off: usize = 0;
        while (off < ids.len) {
            const n: usize = @min(prefill_chunk, ids.len - off);
            try qwen3.embedTokens(self.lm.embed, ids[off..][0..n], x[0 .. n * hidden]);
            for (0..n) |t| pos3s[t * 3 ..][0..3].* = @splat(@intCast(self.pos_next + t));
            try self.chunk(x[0 .. n * hidden], pos3s[0 .. n * 3], true);
            self.pos_next += n;
            try self.be.downloadF32At(self.bufs.normed, 0, out[off * hidden ..][0 .. n * hidden]);
            off += n;
        }
    }

    /// Prefill one image: `embeds` are the tower's rows ([grid_w * grid_h][hidden]) at
    /// M-RoPE grid positions (t at the image start, h and w its merged row and column);
    /// the next text continues at the start plus the larger grid side.
    pub fn prefillImage(self: *VulkanLM, embeds: []const f32, grid_w: usize, grid_h: usize) !void {
        const cfg = self.cfg;
        const total = grid_w * grid_h;
        std.debug.assert(embeds.len == total * cfg.hidden and total <= self.remaining());
        const pos0: u32 = @intCast(self.pos_next);
        var pos3s: [prefill_chunk * 3]u32 = undefined;
        var off: usize = 0;
        while (off < total) {
            try bnd.check(self);
            const n: usize = @min(prefill_chunk, total - off);
            for (0..n) |t| pos3s[t * 3 ..][0..3].* = qwen35.imagePos3(pos0, grid_w, off + t);
            try self.chunk(embeds[off * cfg.hidden ..][0 .. n * cfg.hidden], pos3s[0 .. n * 3], false);
            off += n;
        }
        self.pos_next = pos0 + @max(grid_w, grid_h);
    }

    /// One batched forward over `n` embedded rows at their positions, and with `normed`
    /// the final norm of every row into `bufs.normed`.
    fn chunk(self: *VulkanLM, x: []const f32, pos3s: []const u32, normed: bool) !void {
        const n = pos3s.len / 3;
        std.debug.assert(n >= 1 and n <= prefill_chunk and n <= self.remaining());
        // Uploads before the batch opens: one inside it would flush the recording.
        try self.be.tensorUpload(try self.be.view(self.bufs.x, 0, x.len * 4), std.mem.sliceAsBytes(x));
        try self.be.tensorUpload(self.pos3s_d, std.mem.sliceAsBytes(pos3s));
        self.be.weight_noise.tick();
        try self.be.beginBatch();
        errdefer if (self.be.batching) self.be.abortBatch();
        var dev: Dev = .{ .m = self, .cfg = self.cfg, .lm = self.lm, .bufs = &self.bufs };
        try qwen35_dev.batch(&dev, n);
        if (normed) try dev.rms(self.bufs.x, self.bufs.normed, self.lm.final_norm, n, self.cfg.hidden);
        try self.be.endBatch();
        self.len += n;
    }

    fn stepOne(self: *VulkanLM, id: u32) !void {
        const cfg = self.cfg;
        const x = try self.gpa.alloc(f32, cfg.hidden);
        defer self.gpa.free(x);
        try qwen3.embedTokens(self.lm.embed, &.{id}, x);
        const p: u32 = @intCast(self.pos_next);
        try self.be.tensorUpload(try self.be.view(self.bufs.x, 0, cfg.hidden * 4), std.mem.sliceAsBytes(x));
        try self.be.tensorUpload(self.pos3_d, std.mem.sliceAsBytes(&[3]u32{ p, p, p }));
        self.be.weight_noise.tick();
        try self.be.beginBatch();
        errdefer if (self.be.batching) self.be.abortBatch();
        var dev: Dev = .{ .m = self, .cfg = cfg, .lm = self.lm, .bufs = &self.bufs };
        try qwen35_dev.decode(&dev, false, true);
        try self.be.endBatch();
        self.len += 1;
        self.pos_next += 1;
    }

    /// Upload every weight now rather than inside the first turn, by running a
    /// prefill-sized and a decode forward: those are the routes, and so the device
    /// layouts, real turns read. The context is emptied after.
    pub fn warmWeights(self: *VulkanLM) void {
        if (self.len != 0) return;
        const zeros = [_]u32{0} ** 32;
        const n = @min(zeros.len, self.capacity - 1);
        self.prefill(zeros[0..n]) catch |err| std.log.warn("qwen35 vulkan warm-up: {t}", .{err});
        self.stepOne(0) catch |err| std.log.warn("qwen35 vulkan warm-up: {t}", .{err});
        self.resetCache() catch |err| std.log.warn("qwen35 vulkan warm-up: {t}", .{err});
    }

    /// Which layer the coming launches belong to, for the weight-noise curve.
    pub fn noiseAtLayer(self: *VulkanLM, l: usize) void {
        self.be.weight_noise.atLayer(l);
    }

    // Residency: the hooks `residency.zig` drives, with qwen35_cuda's meaning.

    fn layerDeviceBytes(layer: *const qwen35.Layer) usize {
        const mlp = switch (layer.*) {
            .attn => |*a| &a.mlp,
            .linear => |*l| &l.mlp,
        };
        var n: usize = mlp.gate.bytes.len + mlp.up.bytes.len + mlp.down.bytes.len;
        switch (layer.*) {
            .attn => |*a| n += a.qg.bytes.len + a.k.bytes.len + a.v.bytes.len + a.o.bytes.len,
            .linear => |*l| n += l.qkv.bytes.len + l.z.bytes.len + l.alpha.bytes.len + l.beta.bytes.len + l.out.bytes.len,
        }
        return n;
    }

    pub fn layerWeightBytes(self: *VulkanLM, l: usize) usize {
        return layerDeviceBytes(&self.lm.layers[l]);
    }

    /// The head; the embeddings are looked up on the host.
    pub fn nonLayerDeviceBytes(self: *VulkanLM) u64 {
        return self.lm.head.bytes.len;
    }

    /// VRAM a promote of layer `l` needs: its weights, an attention layer's K/V at
    /// capacity (a DeltaNet layer's state is fixed and small), and slack.
    pub fn promoteCost(self: *VulkanLM, l: usize) usize {
        const kv_cost: usize = if (!self.cfg.isRecurrent(l)) 2 * self.kv_dtype.sizeBytes(self.capacity * self.cfg.kvDim()) else 0;
        return layerDeviceBytes(&self.lm.layers[l]) + kv_cost + residency.promote_slack;
    }

    fn liveAttnSlots(self: *const VulkanLM) usize {
        var count: usize = 0;
        for (0..self.cfg.n_layers) |l| {
            if (self.cfg.isRecurrent(l)) continue;
            if (self.split) |sp| if (!sp.on_gpu[l]) continue;
            count += 1;
        }
        return count;
    }

    fn layerWeights(layer: *const qwen35.Layer) [9][]const u8 {
        const mlp = switch (layer.*) {
            .attn => |*a| &a.mlp,
            .linear => |*l| &l.mlp,
        };
        return switch (layer.*) {
            .attn => |*a| .{ mlp.gate.bytes, mlp.up.bytes, mlp.down.bytes, a.qg.bytes, a.k.bytes, a.v.bytes, a.o.bytes, &.{}, &.{} },
            .linear => |*l| .{ mlp.gate.bytes, mlp.up.bytes, mlp.down.bytes, l.qkv.bytes, l.z.bytes, l.alpha.bytes, l.beta.bytes, l.out.bytes, &.{} },
        };
    }

    /// Move layer `l` to the host: an attention layer's live K/V, a DeltaNet layer's
    /// recurrent state, into the shadow; its device copy and weights freed.
    pub fn migrateLayer(self: *VulkanLM, l: usize) !void {
        const cfg = self.cfg;
        const sp = &self.split.?;
        if (!cfg.isRecurrent(l)) {
            const s = l / cfg.full_attn_interval;
            try self.kv.dropLayer(self.be, s, self.len, &sp.state.kv, s);
        } else {
            const lin_idx = l - l / cfg.full_attn_interval;
            const cn = convBytes(cfg) / 4;
            const sn = ssmBytes(cfg) / 4;
            try self.be.tensorDownload(self.conv_state[lin_idx], std.mem.sliceAsBytes(sp.state.conv[lin_idx * cn ..][0..cn]));
            try self.be.tensorDownload(self.ssm_state[lin_idx], std.mem.sliceAsBytes(sp.state.ssm[lin_idx * sn ..][0..sn]));
        }
        for (layerWeights(&self.lm.layers[l])) |w| if (w.len > 0) self.be.evictWeightBytes(w);
        sp.on_gpu[l] = false;
        sp.n_cpu += 1;
        std.log.debug("[offload] layer {d} ({s}) -> CPU at ctx {d} ({d}/{d} on CPU)", .{ l, if (cfg.isRecurrent(l)) "lin" else "attn", self.len, sp.n_cpu, cfg.n_layers });
    }

    /// Bring layer `l` back to the device with its state; its weights re-upload on the
    /// next forward.
    pub fn promoteLayer(self: *VulkanLM, l: usize) !void {
        const cfg = self.cfg;
        const sp = &self.split.?;
        if (!cfg.isRecurrent(l)) {
            const s = l / cfg.full_attn_interval;
            try self.kv.restoreLayer(self.be, s, self.capacity, self.len, &sp.state.kv, s);
        } else {
            const lin_idx = l - l / cfg.full_attn_interval;
            const cn = convBytes(cfg) / 4;
            const sn = ssmBytes(cfg) / 4;
            try self.be.tensorUpload(self.conv_state[lin_idx], std.mem.sliceAsBytes(sp.state.conv[lin_idx * cn ..][0..cn]));
            try self.be.tensorUpload(self.ssm_state[lin_idx], std.mem.sliceAsBytes(sp.state.ssm[lin_idx * sn ..][0..sn]));
        }
        sp.on_gpu[l] = true;
        sp.n_cpu -= 1;
        std.log.debug("[promote] layer {d} ({s}) -> GPU at ctx {d} ({d}/{d} on CPU)", .{ l, if (cfg.isRecurrent(l)) "lin" else "attn", self.len, sp.n_cpu, cfg.n_layers });
    }

    pub fn promoteLayers(self: *VulkanLM, budget: u64) !usize {
        return residency.promoteBack(self, budget);
    }

    pub fn offloadUntilFree(self: *VulkanLM, needed_free: u64) !void {
        return residency.offloadUntilFree(self, needed_free);
    }

    pub fn offloadToBudget(self: *VulkanLM, target: u64) !void {
        return residency.offloadToBudget(self, target);
    }

    /// Always arm the dynamic split (`budget == 0` is no offload).
    pub fn autoOffload(self: *VulkanLM, budget: u64) !bool {
        if (budget == 0) return false;
        try self.enableCpuSplit(.attn, budget, true);
        return true;
    }

    /// Place layers on the host until the device-resident weights fit under `budget`,
    /// as qwen35_cuda's: `.attn` sends the KV-growing attention layers first.
    pub fn enableCpuSplit(self: *VulkanLM, policy: residency.CpuSplitPolicy, budget: u64, dynamic: bool) !void {
        std.debug.assert(self.split == null);
        const cfg = self.cfg;
        const n = cfg.n_layers;
        const gpa = self.gpa;
        const per = try gpa.alloc(usize, n);
        defer gpa.free(per);
        for (self.lm.layers, 0..) |*layer, l| per[l] = layerDeviceBytes(layer);
        const kv_bytes = 2 * cfg.nAttnLayers() * self.kv_dtype.sizeBytes(self.capacity * cfg.kvDim());
        const reserve = if (dynamic) self.lm.head.bytes.len else kv_bytes + self.lm.head.bytes.len + (512 << 20);
        const order = try gpa.alloc(usize, n);
        errdefer gpa.free(order);
        switch (policy) {
            .tail => for (0..n) |i| {
                order[i] = n - 1 - i;
            },
            .attn => {
                var w: usize = 0;
                for ([2]bool{ false, true }) |recurrent| {
                    var l = n;
                    while (l > 0) {
                        l -= 1;
                        if (cfg.isRecurrent(l) == recurrent) {
                            order[w] = l;
                            w += 1;
                        }
                    }
                }
            },
        }
        const on_gpu = try gpa.alloc(bool, n);
        errdefer gpa.free(on_gpu);
        @memset(on_gpu, true);
        const n_cpu = residency.splitHostCount(self, budget, reserve, dynamic, per, order);
        if (n_cpu == 0 and !dynamic) {
            gpa.free(on_gpu);
            gpa.free(order);
            return;
        }
        var state = try qwen35.State.init(gpa, cfg, self.capacity, self.kv_dtype);
        errdefer state.deinit(gpa);
        state.kv.len = self.len;
        state.len = self.len;
        var scratch = try qwen35.Scratch.init(gpa, prefill_chunk, cfg);
        errdefer scratch.deinit(gpa);
        var freqs = try ops.rope.rotateHalfFreqs(gpa, self.capacity, cfg.rope_dim, cfg.rope_theta);
        errdefer freqs.deinit(gpa);
        const hx = try gpa.alloc(f32, prefill_chunk * cfg.hidden);
        errdefer gpa.free(hx);
        self.split = .{
            .on_gpu = on_gpu,
            .n_cpu = 0,
            .policy = policy,
            .state = state,
            .scratch = scratch,
            .freqs = freqs,
            .hx = hx,
            .dynamic = dynamic,
            .order = order,
            .next = n_cpu,
            .budget = budget,
        };
        for (order[0..n_cpu]) |l| try self.migrateLayer(l);
    }

    fn freeSplit(self: *VulkanLM) void {
        const sp = &(self.split orelse return);
        sp.state.deinit(self.gpa);
        sp.scratch.deinit(self.gpa);
        sp.freqs.deinit(self.gpa);
        self.gpa.free(sp.hx);
        self.gpa.free(sp.on_gpu);
        self.gpa.free(sp.order);
        self.split = null;
    }

    /// New chat: drop the split, shrink the KV back to its initial capacity, empty the
    /// context, and re-arm the dynamic offload.
    pub fn resetResidency(self: *VulkanLM, budget: u64) !void {
        self.freeSplit();
        const kv = try kv_gpu.Kv.init(self.gpa, self.be, self.cfg.nAttnLayers(), self.initial_capacity, self.cfg.kvDim(), self.kv_dtype, &.{});
        self.kv.deinit(self.gpa, self.be);
        self.kv = kv;
        self.capacity = self.initial_capacity;
        try self.resetCache();
        _ = try self.autoOffload(budget);
    }
};

/// `qwen35_dev`'s adapter over a `VulkanLM`.
const Dev = struct {
    m: *VulkanLM,
    cfg: qwen35.Config,
    lm: *const qwen35.Model,
    bufs: *const Bufs,

    fn ctx(d: *const Dev) *gpu.Context {
        return d.m.be;
    }

    pub fn view(d: *const Dev, b: Buf, off: usize, bytes: usize) !Buf {
        return d.ctx().view(b, off, bytes);
    }
    /// A hidden-width row takes a whole workgroup; a head's, one subgroup.
    pub fn rms(d: *const Dev, x: Buf, out: Buf, w: []const f32, rows: usize, dim: usize) !void {
        if (dim == d.cfg.hidden) return d.ctx().opAddRmsNorm(x, null, try nbuf(d.ctx(), w), out, rows, dim, d.cfg.rms_eps);
        try d.ctx().opElt(.rmsnorm, x, out, try nbuf(d.ctx(), w), null, .{ .u0 = @intCast(rows), .u1 = @intCast(dim), .f0 = d.cfg.rms_eps }, rows, 1, 1);
    }
    pub fn addRms(d: *const Dev, x: Buf, t: Buf, out: Buf, w: []const f32, rows: usize, dim: usize) !void {
        try d.ctx().opAddRmsNorm(x, t, try nbuf(d.ctx(), w), out, rows, dim, d.cfg.rms_eps);
    }
    pub fn prep(d: *const Dev, x: Buf, m: usize, cols: usize, ws: []const Weight) !lin_gpu.Group {
        return d.m.lin.prep(d.m.lin_plan, x, m, cols, ws);
    }
    pub fn gemm(d: *const Dev, g: lin_gpu.Group, y: Buf, w: Weight) !void {
        try d.m.lin.gemm(d.m.lin_plan, g, y, w, .{});
    }
    pub fn linear(d: *const Dev, y: Buf, x: Buf, m: usize, w: Weight) !void {
        try d.m.lin.linear(d.m.lin_plan, y, 0, x, m, w);
    }
    pub fn deinterleave2(d: *const Dev, src: Buf, a: Buf, b: Buf, total: usize, hd: usize) !void {
        try d.ctx().opDeinterleave2(src, a, b, total, hd);
    }
    pub fn mulSigmoid(d: *const Dev, a: Buf, g: Buf, n: usize) !void {
        try d.ctx().opElt(.sigmoid_mul, a, g, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
    }
    pub fn add(d: *const Dev, x: Buf, t: Buf, n: usize) !void {
        try d.ctx().opElt(.add, x, t, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
    }
    pub fn siluMul(d: *const Dev, a: Buf, b: Buf, n: usize) !void {
        try d.ctx().opElt(.silu_mul, a, b, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
    }
    pub fn l2normRows(d: *const Dev, x: Buf, rows: usize, dim: usize) !void {
        try d.ctx().opL2NormRows(x, rows, dim, d.cfg.rms_eps);
    }
    pub fn l2normRowsGrouped(d: *const Dev, x: Buf, rows: usize, dim: usize, group: usize, stride: usize) !void {
        try d.ctx().opL2NormRowsGrouped(x, rows, dim, group, stride, d.cfg.rms_eps);
    }
    pub fn gdnGates(d: *const Dev, ab: Buf, a_dt: []const f32, gates: Buf, heads: usize) !void {
        try d.ctx().opGdnGates(ab, try nbuf(d.ctx(), a_dt), gates, heads);
    }
    pub fn gdnConvStep(d: *const Dev, state: Buf, x: Buf, w: []const f32, out: Buf, channels: usize) !void {
        try d.ctx().opGdnConvStep(state, x, try nbuf(d.ctx(), w), out, channels, d.cfg.conv_kernel);
    }
    /// The chunk kernel at one token, so a decode step and a prefill agree exactly.
    pub fn gdnDeltaStep(d: *const Dev, state: Buf, conv: Buf, gates: Buf, o: Buf, heads: usize, dk: usize, k_heads: usize, scale: f32) !void {
        try d.ctx().opGdnDeltaChunk(state, conv, gates, o, heads, dk, k_heads, d.cfg.v_group, 1, d.cfg.convChannels(), scale);
    }
    pub fn gdnGatesBatch(d: *const Dev, alpha: Buf, beta: Buf, a_dt: []const f32, gates: Buf, heads: usize, n: usize, pitch: usize) !void {
        try d.ctx().opGdnGatesBatch(alpha, beta, try nbuf(d.ctx(), a_dt), gates, heads, n, pitch);
    }
    pub fn gdnConvBatch(d: *const Dev, state: Buf, x: Buf, w: []const f32, out: Buf, channels: usize, n: usize) !void {
        try d.ctx().opGdnConvBatch(state, x, try nbuf(d.ctx(), w), out, channels, n);
    }
    pub fn gdnDeltaChunk(d: *const Dev, state: Buf, conv: Buf, gates: Buf, o: Buf, heads: usize, dk: usize, k_heads: usize, n: usize, channels: usize, scale: f32) !void {
        try d.ctx().opGdnDeltaChunk(state, conv, gates, o, heads, dk, k_heads, d.cfg.v_group, n, channels, scale);
    }

    pub fn convState(d: *const Dev, lin_idx: usize) Buf {
        return d.m.conv_state[lin_idx];
    }
    pub fn ssmState(d: *const Dev, lin_idx: usize) Buf {
        return d.m.ssm_state[lin_idx];
    }
    pub fn aDt(d: *const Dev, lin_idx: usize) []const f32 {
        return d.m.a_dt[lin_idx];
    }
    pub fn gdnRoute(d: *const Dev, _: usize, n: usize) qwen35_dev.GdnRoute(Weight) {
        const dk = d.cfg.lin_head_dim;
        return .{ .chunk = n > 1 and (dk == 64 or dk == 128) };
    }
    pub fn abViews(d: *const Dev, _: qwen35_dev.GdnRoute(Weight), n: usize) !qwen35_dev.AbViews(Buf) {
        const heads = d.cfg.lin_v_heads;
        return .{
            .alpha = try d.view(d.bufs.ab_n, 0, n * heads * 4),
            .beta = try d.view(d.bufs.ab_n, prefill_chunk * heads * 4, n * heads * 4),
            .pitch = heads,
        };
    }

    pub fn ropeStep(d: *const Dev, qk: Buf, heads: usize) !void {
        const c = d.cfg;
        try d.ctx().opRopeImrope(qk, d.m.pos3_d, d.m.freqs_d, heads, c.rope_dim / 2, d.m.sin_off, c.rope_sections, c.head_dim);
    }
    pub fn ropeRows(d: *const Dev, qk: Buf, heads: usize, n: usize) !void {
        const c = d.cfg;
        try d.ctx().opRopeImropePos(qk, d.m.pos3s_d, d.m.freqs_d, n, heads, c.rope_dim / 2, d.m.sin_off, c.rope_sections, c.head_dim);
    }
    pub fn storeKvStep(d: *const Dev, slot: usize, comptime graph: bool) !void {
        comptime std.debug.assert(!graph);
        try d.storeKvRows(slot, 1);
    }
    pub fn storeKvRows(d: *const Dev, slot: usize, n: usize) !void {
        try d.m.kv.store(d.ctx(), slot, d.bufs.k, d.bufs.v, d.m.len, n);
    }
    pub fn attendStep(d: *const Dev, slot: usize, comptime graph: bool) !void {
        comptime std.debug.assert(!graph);
        try d.attend(slot, 1, nsplit);
    }
    pub fn attendRows(d: *const Dev, slot: usize, n: usize) !void {
        try d.attend(slot, n, nsplit_prefill);
    }
    fn attend(d: *const Dev, slot: usize, n: usize, splits: usize) !void {
        const c = d.cfg;
        const b = d.bufs;
        try d.m.kv.attend(d.ctx(), slot, b.q, b.attn, b.attn_scratch, d.m.len + 1, n, c.n_heads, c.n_kv_heads, c.head_dim, splits, 1.0 / @sqrt(@as(f32, @floatFromInt(c.head_dim))), 0, 0);
    }

    /// The hidden begins on the device; `on_host` follows it as host layers take it.
    pub fn forwardBegin(d: *const Dev, _: usize) !void {
        if (d.m.split) |*sp| sp.on_host = false;
    }
    /// Run layer `l` on the host when the split put it there; true when it did. A
    /// transfer inside the recording batch flushes it, which keeps the order.
    pub fn layerBegin(d: *const Dev, l: usize, n: usize) !bool {
        const m = d.m;
        m.noiseAtLayer(l);
        const sp = if (m.split) |*s| s else return false;
        const hidden = d.cfg.hidden;
        const x = try d.view(d.bufs.x, 0, n * hidden * 4);
        if (!sp.on_gpu[l]) {
            if (!sp.on_host) {
                try d.ctx().tensorDownload(x, std.mem.sliceAsBytes(sp.hx[0 .. n * hidden]));
                sp.on_host = true;
            }
            const host_io = m.io orelse return error.SplitIoUnset;
            try m.lm.cpuLayer(host_io, m.gpa, l, sp.hx[0 .. n * hidden], n, sp.freqs, &sp.state, &sp.scratch);
            return true;
        }
        if (sp.on_host) {
            try d.ctx().tensorUpload(x, std.mem.sliceAsBytes(sp.hx[0 .. n * hidden]));
            sp.on_host = false;
        }
        return false;
    }
    pub fn tapNormedStep(_: *const Dev, _: usize) !void {}
    pub fn tapNormedRows(_: *const Dev, _: usize, _: usize) !void {}
    pub fn tapLayer(_: *const Dev, _: usize, _: usize) !void {}
    /// The split's host state advances in step. A decode step brings the hidden back
    /// for the device head; a prefill reads no logits and the next chunk re-embeds.
    pub fn forwardEnd(d: *const Dev, n: usize, step: bool) !void {
        const sp = if (d.m.split) |*s| s else return;
        if (step and sp.on_host) {
            try d.ctx().tensorUpload(try d.view(d.bufs.x, 0, d.cfg.hidden * 4), std.mem.sliceAsBytes(sp.hx[0..d.cfg.hidden]));
            sp.on_host = false;
        }
        sp.state.kv.commit(n);
        sp.state.len += n;
    }
    pub fn headBegin(d: *const Dev) void {
        d.m.be.weight_noise.atHead();
    }
};

const test_path = build_options.model_root ++ "/../llm/Heretical-Qwen3.5-9B.Q6_K.gguf";

/// A device and the test checkpoint, or null to skip. Heap-held: the model borrows the
/// container's mapping.
const Fixture = struct {
    ctx: *gpu.Context,
    g: @import("tp_core").gguf.Gguf,
    lm: qwen35.Model,

    fn open(gpa: std.mem.Allocator, io: std.Io) !*Fixture {
        std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
        std.Io.Dir.cwd().access(io, test_path, .{}) catch return error.SkipZigTest;
        const f = try gpa.create(Fixture);
        errdefer gpa.destroy(f);
        f.ctx = gpu.Context.init(gpa, io) catch return error.SkipZigTest;
        errdefer f.ctx.deinit();
        f.g = try @TypeOf(f.g).open(gpa, io, test_path);
        errdefer f.g.deinit();
        f.lm = try qwen35.Model.load(gpa, &f.g);
        return f;
    }

    fn close(f: *Fixture, gpa: std.mem.Allocator) void {
        f.lm.deinit();
        f.g.deinit();
        f.ctx.deinit();
        gpa.destroy(f);
    }
};

fn testPrompt(comptime n: usize) [n]u32 {
    var p: [n]u32 = undefined;
    for (&p, 0..) |*t, i| t.* = @intCast(1000 + (i * 37) % 50000);
    return p;
}

/// `out.len` greedy tokens from `first`, growing the KV as the engine does.
fn greedy(m: *VulkanLM, io: std.Io, first: u32, out: []u32) !void {
    var cur = first;
    for (out) |*t| {
        try m.ensureCapacity(m.cached() + 1);
        cur = try m.stepArgmax(io, &.{cur});
        t.* = cur;
    }
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

// The GUI session's turn shape: warm-up, prefill to the boundary, a recurrent-state
// checkpoint, generate, roll back and regenerate (greedy must repeat), a reset and a
// replay, the device sampler against the downloaded logits, then a KV dtype rebuild
// that must match a stepper built at that dtype.
test "vulkan qwen35 gui turn lifecycle: checkpoint rollback, reset, sampling, kv rebuild" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const f = try Fixture.open(gpa, io);
    defer f.close(gpa);
    const prompt = testPrompt(40);
    const last = prompt[prompt.len - 1];

    var model = try VulkanLM.init(gpa, f.ctx, &f.lm, .{ .initial = 256, .max = 256 });
    defer model.deinit();
    model.warmWeights();
    try std.testing.expectEqual(@as(usize, 0), model.cached());

    try model.prefill(prompt[0 .. prompt.len - 1]);
    const q = model.cached();
    const snap = try gpa.alloc(u8, model.checkpointBytes());
    defer gpa.free(snap);
    try std.testing.expect(snap.len > 8);
    try model.checkpoint(snap);
    var seq1: [12]u32 = undefined;
    try greedy(&model, io, last, &seq1);

    try model.restoreCheckpoint(snap, q);
    try std.testing.expectEqual(q, model.cached());
    var seq2: [12]u32 = undefined;
    try greedy(&model, io, last, &seq2);
    try std.testing.expectEqualSlices(u32, &seq1, &seq2);

    try model.resetCache();
    try std.testing.expectEqual(@as(usize, 0), model.cached());
    try model.prefill(prompt[0 .. prompt.len - 1]);
    try greedy(&model, io, last, &seq2);
    try std.testing.expectEqualSlices(u32, &seq1, &seq2);

    // The device argmax and top-k against the downloaded logits of the same step.
    const logits = try gpa.alloc(f32, model.vocab());
    defer gpa.free(logits);
    try model.restoreCheckpoint(snap, q);
    try model.step(io, &.{last}, logits);
    try std.testing.expectEqual(seq1[0], @as(u32, @intCast(sample.argmax(logits))));
    try model.restoreCheckpoint(snap, q);
    var ids: [gpu.topk_lanes * gpu.topk_m]u32 = undefined;
    var vals: [gpu.topk_lanes * gpu.topk_m]f32 = undefined;
    try std.testing.expect(model.maxSelect() == ids.len);
    const count = try model.stepSelect(io, &.{last}, &ids, &vals);
    try std.testing.expect(count > 0);
    var best: usize = 0;
    for (ids[0..count], vals[0..count], 0..) |id, v, i| {
        try std.testing.expectEqual(logits[id], v);
        if (v > vals[best]) best = i;
    }
    try std.testing.expectEqual(seq1[0], ids[best]);

    // A rebuilt f16 cache against a stepper built with one.
    try model.reinitCache(.f16);
    try std.testing.expectEqual(kvmod.KvDtype.f16, model.kv_dtype);
    try std.testing.expectEqual(@as(usize, 0), model.cached());
    try model.prefill(prompt[0 .. prompt.len - 1]);
    try greedy(&model, io, last, &seq1);
    var fresh = try VulkanLM.init(gpa, f.ctx, &f.lm, .{ .initial = 256, .max = 256, .kv_dtype = .f16 });
    defer fresh.deinit();
    try fresh.prefill(prompt[0 .. prompt.len - 1]);
    try greedy(&fresh, io, last, &seq2);
    try std.testing.expectEqualSlices(u32, &seq1, &seq2);
}

// A cache that starts at 16 rows and grows through a reply decodes exactly what one
// built at full size does: growth copies rows, it changes no arithmetic.
test "vulkan qwen35 kv growth decodes what a full-size cache does" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const f = try Fixture.open(gpa, io);
    defer f.close(gpa);
    const prompt = testPrompt(24);
    var outs: [2][40]u32 = undefined;
    for (&outs, [2]usize{ 16, 256 }) |*out, initial| {
        var m = try VulkanLM.init(gpa, f.ctx, &f.lm, .{ .initial = initial, .max = 256, .kv_dtype = .f16 });
        defer m.deinit();
        try m.ensureCapacity(prompt.len);
        try m.prefill(prompt[0 .. prompt.len - 1]);
        try greedy(&m, io, prompt[prompt.len - 1], out);
        if (initial == 16) try std.testing.expect(m.cached() + m.remaining() > 32);
    }
    errdefer std.debug.print("grown {any}\nfull  {any}\n", .{ outs[0], outs[1] });
    try std.testing.expectEqualSlices(u32, &outs[1], &outs[0]);
}

// The CPU split as the GUI arms it: host layers need an Io and track the device; a
// snapshot taken all on the device restores into layers since moved to the host
// (rollback must repeat there), and after every layer comes back the continuation
// equals the never-split one exactly, since the migrate and promote copy state and
// rows byte for byte.
fn vulkanSplitBody(kv_dtype: kvmod.KvDtype) !void {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const f = try Fixture.open(gpa, io);
    defer f.close(gpa);
    const prompt = testPrompt(40);
    const last = prompt[prompt.len - 1];
    const vocab = f.lm.cfg.vocab;

    const want = try gpa.alloc(f32, vocab);
    defer gpa.free(want);
    const got = try gpa.alloc(f32, vocab);
    defer gpa.free(got);
    {
        var full = try VulkanLM.init(gpa, f.ctx, &f.lm, .{ .initial = 64, .max = 256, .kv_dtype = kv_dtype });
        defer full.deinit();
        try full.step(io, &prompt, want);
    }
    {
        var m = try VulkanLM.init(gpa, f.ctx, &f.lm, .{ .initial = 64, .max = 256, .kv_dtype = kv_dtype });
        defer m.deinit();
        // ~7 GB of layer weights: 5 GiB puts the attention layers and some more on the host.
        try m.enableCpuSplit(.attn, 5 << 30, true);
        const sp = &m.split.?;
        errdefer std.debug.print("n_cpu={d}\n", .{sp.n_cpu});
        try std.testing.expect(sp.n_cpu > 0 and sp.n_cpu < m.cfg.n_layers);
        try std.testing.expectEqual(kv_dtype, sp.state.kv.kv_dtype);
        try std.testing.expectError(error.SplitIoUnset, m.prefill(prompt[0 .. prompt.len - 1]));
        try m.resetCache();
        try std.testing.expectEqual(@as(usize, 0), sp.state.len);
        m.io = io;
        try m.step(io, &prompt, got);
        const rel = relL2(want, got);
        errdefer std.debug.print("split vs device logits rel L2 {e:.3} (kv {t})\n", .{ rel, kv_dtype });
        try std.testing.expect(rel < 5e-2);
        try std.testing.expectEqual(sample.argmax(want), sample.argmax(got));
        try std.testing.expectEqual(m.cached(), sp.state.len);
    }

    var m = try VulkanLM.init(gpa, f.ctx, &f.lm, .{ .initial = 64, .max = 256, .kv_dtype = kv_dtype });
    defer m.deinit();
    m.io = io;
    try m.prefill(prompt[0 .. prompt.len - 1]);
    const q = m.cached();
    const snap = try gpa.alloc(u8, m.checkpointBytes());
    defer gpa.free(snap);
    try m.checkpoint(snap);
    var seq1: [12]u32 = undefined;
    try greedy(&m, io, last, &seq1);

    try m.enableCpuSplit(.tail, std.math.maxInt(u64), true);
    try m.offloadToBudget(f.ctx.deviceUsed() * 4 / 5);
    try std.testing.expect(m.split.?.n_cpu > 0);
    // A restore reaches every layer where it now lives: a checkpoint taken right after
    // it reads the same bytes back, and the replies after two restores agree, logits
    // included (greedy alone can hide a stale state).
    var seq3: [12]u32 = undefined;
    var seq4: [12]u32 = undefined;
    const snap2 = try gpa.alloc(u8, snap.len);
    defer gpa.free(snap2);
    try m.restoreCheckpoint(snap, q);
    try std.testing.expectEqual(q, m.split.?.state.len);
    try m.checkpoint(snap2);
    try std.testing.expectEqualSlices(u8, snap, snap2);
    try m.step(io, &.{last}, want);
    try m.restoreCheckpoint(snap, q);
    try greedy(&m, io, last, &seq3);
    try std.testing.expectEqual(m.cached(), m.split.?.state.len);
    try m.restoreCheckpoint(snap, q);
    try m.step(io, &.{last}, got);
    try std.testing.expectEqualSlices(f32, want, got);
    try m.restoreCheckpoint(snap, q);
    try greedy(&m, io, last, &seq4);
    try std.testing.expectEqualSlices(u32, &seq3, &seq4);

    const promoted = try m.promoteLayers(std.math.maxInt(u64));
    try std.testing.expect(promoted > 0);
    try std.testing.expectEqual(@as(usize, 0), m.split.?.n_cpu);
    try m.restoreCheckpoint(snap, q);
    var seq5: [12]u32 = undefined;
    try greedy(&m, io, last, &seq5);
    errdefer std.debug.print("unsplit {any}\npromoted {any}\n", .{ seq1, seq5 });
    try std.testing.expectEqualSlices(u32, &seq1, &seq5);
}

test "vulkan qwen35 cpu split: host layers track the device, rollback repeats, promote is exact" {
    try vulkanSplitBody(.f32);
}

test "vulkan qwen35 cpu split with f16 kv" {
    try vulkanSplitBody(.f16);
}

test "vulkan qwen35 cpu split with q8_0 kv" {
    try vulkanSplitBody(.q8_0);
}

// Layers moved mid-context carry their rows and recurrent state, and moved back bring
// them back byte for byte: the logits after the promote equal an unsplit stepper's.
test "vulkan qwen35 cpu split moves live state to the host and back losslessly" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const f = try Fixture.open(gpa, io);
    defer f.close(gpa);
    const prompt = testPrompt(40);
    const k = 16;
    const n = prompt.len;
    const vocab = f.lm.cfg.vocab;
    const want = try gpa.alloc(f32, vocab);
    defer gpa.free(want);
    const got = try gpa.alloc(f32, vocab);
    defer gpa.free(got);

    for ([_]kvmod.KvDtype{ .f32, .f16, .q8_0 }) |dt| {
        {
            var ref = try VulkanLM.init(gpa, f.ctx, &f.lm, .{ .initial = 64, .max = 64, .kv_dtype = dt });
            defer ref.deinit();
            try ref.prefill(prompt[0..k]);
            try ref.prefill(prompt[k .. n - 1]);
            try ref.step(io, prompt[n - 1 ..], want);
        }
        var m = try VulkanLM.init(gpa, f.ctx, &f.lm, .{ .initial = 64, .max = 64, .kv_dtype = dt });
        defer m.deinit();
        m.io = io;
        try m.prefill(prompt[0..k]);
        try m.enableCpuSplit(.attn, 5 << 30, true);
        try std.testing.expect(m.split.?.n_cpu > 0);
        _ = try m.promoteLayers(std.math.maxInt(u64));
        try std.testing.expectEqual(@as(usize, 0), m.split.?.n_cpu);
        try m.prefill(prompt[k .. n - 1]);
        try m.step(io, prompt[n - 1 ..], got);
        errdefer std.debug.print("kv {t}: logits after the round trip differ from the unsplit run\n", .{dt});
        try std.testing.expectEqualSlices(f32, want, got);
    }
}

// The GUI's KV dtype toggle with the split armed: host layers keep no device KV, the
// host state is rebuilt at the new dtype, the placement survives, and a re-prefill runs.
test "vulkan qwen35 cpu split survives a kv dtype toggle, and a residency reset re-arms it" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const f = try Fixture.open(gpa, io);
    defer f.close(gpa);
    var model = try VulkanLM.init(gpa, f.ctx, &f.lm, .{ .initial = 64, .max = 256 });
    defer model.deinit();
    try model.enableCpuSplit(.attn, 5 << 30, true);
    try std.testing.expect(model.split.?.n_cpu > 0);
    model.io = io;
    const cfg = model.cfg;

    const prompt = testPrompt(24);
    for ([_]kvmod.KvDtype{ .f16, .q8_0, .f32 }) |dt| {
        const n_cpu_before = model.split.?.n_cpu;
        try model.reinitCache(dt);
        const sp = &model.split.?;
        try std.testing.expectEqual(dt, model.kv_dtype);
        try std.testing.expectEqual(dt, sp.state.kv.kv_dtype);
        try std.testing.expectEqual(@as(usize, 0), sp.state.len);
        try std.testing.expectEqual(n_cpu_before, sp.n_cpu);
        for (sp.on_gpu, 0..) |on, l| {
            if (cfg.isRecurrent(l)) continue;
            try std.testing.expectEqual(on, model.kv.k[l / cfg.full_attn_interval].buf != .null_handle);
        }
        try model.prefill(prompt[0 .. prompt.len - 1]);
        try std.testing.expectEqual(prompt.len - 1, sp.state.len);
        try std.testing.expect(try model.stepArgmax(io, prompt[prompt.len - 1 ..]) < cfg.vocab);
    }

    try model.ensureCapacity(200);
    try std.testing.expect(model.cached() + model.remaining() >= 200);
    try model.resetResidency(5 << 30);
    try std.testing.expectEqual(@as(usize, 0), model.cached());
    try std.testing.expectEqual(@as(usize, 64), model.remaining());
    try std.testing.expect(model.split.?.n_cpu > 0);
    try model.prefill(prompt[0 .. prompt.len - 1]);
    _ = try model.stepArgmax(io, prompt[prompt.len - 1 ..]);
}

// Weight noise through the prefill's dequant GEMM and the decode GEMV: off is exact,
// on moves both, the same draw counter reproduces both, and off again restores them.
test "vulkan qwen35 weight noise perturbs prefill and decode, reproducibly, and off is exact" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const f = try Fixture.open(gpa, io);
    defer f.close(gpa);
    var model = try VulkanLM.init(gpa, f.ctx, &f.lm, .{ .initial = 512, .max = 512 });
    defer model.deinit();
    const prompt = testPrompt(300);
    var bufs: [6][]f32 = undefined;
    for (&bufs) |*b| b.* = try gpa.alloc(f32, f.lm.cfg.vocab);
    defer for (bufs) |b| gpa.free(b);
    const off_pre, const off_dec, const on_pre, const on_dec, const re_pre, const re_dec = bufs;

    const run = struct {
        fn go(m: *VulkanLM, io_: std.Io, p: []const u32, pre: []f32, dec: []f32) !void {
            try m.resetCache();
            m.be.weight_noise.seq = 0;
            try m.step(io_, p, pre);
            try m.step(io_, &.{p[0]}, dec);
        }
    }.go;
    try run(&model, io, &prompt, off_pre, off_dec);
    const noise = &f.ctx.weight_noise;
    noise.seed = 7;
    noise.setCurve("0.05");
    try std.testing.expect(noise.on());
    try run(&model, io, &prompt, on_pre, on_dec);
    try std.testing.expect(!std.mem.eql(f32, off_pre, on_pre));
    try std.testing.expect(!std.mem.eql(f32, off_dec, on_dec));
    try run(&model, io, &prompt, re_pre, re_dec);
    try std.testing.expectEqualSlices(f32, on_pre, re_pre);
    try std.testing.expectEqualSlices(f32, on_dec, re_dec);
    noise.setCurve("");
    try run(&model, io, &prompt, re_pre, re_dec);
    try std.testing.expectEqualSlices(f32, off_pre, re_pre);
    try std.testing.expectEqualSlices(f32, off_dec, re_dec);
}

// Image rows at M-RoPE grid positions (`qwen35.imagePos3` has its own test). A 1x1 image
// sits at one position on all three axes, so it must be exactly a text token with that
// embedding; a 4x3 grid moves the next position by 4, is not the same rows at text
// positions, and tracks CUDA's prefillImage when a CUDA device is present, within the
// two backends' own disagreement, which is wider than an axis swap moves the logits.
test "vulkan qwen35 image prefill places rows on the M-RoPE grid" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const f = try Fixture.open(gpa, io);
    defer f.close(gpa);
    const cfg = f.lm.cfg;
    const prompt = testPrompt(8);
    const grid_w = 4;
    const grid_h = 3;
    const img_ids = testPrompt(grid_w * grid_h);
    const rows = try gpa.alloc(f32, img_ids.len * cfg.hidden);
    defer gpa.free(rows);
    try qwen3.embedTokens(f.lm.embed, &img_ids, rows);
    const want = try gpa.alloc(f32, cfg.vocab);
    defer gpa.free(want);
    const got = try gpa.alloc(f32, cfg.vocab);
    defer gpa.free(got);

    var m = try VulkanLM.init(gpa, f.ctx, &f.lm, .{ .initial = 64, .max = 64 });
    defer m.deinit();
    try m.prefill(prompt[0..4]);
    try m.prefill(prompt[4..5]);
    try m.step(io, prompt[5..6], want);
    try m.resetCache();
    try m.prefill(prompt[0..4]);
    const one = try gpa.alloc(f32, cfg.hidden);
    defer gpa.free(one);
    try qwen3.embedTokens(f.lm.embed, prompt[4..5], one);
    try m.prefillImage(one, 1, 1);
    try std.testing.expectEqual(@as(usize, 5), m.pos_next);
    try m.step(io, prompt[5..6], got);
    try std.testing.expectEqualSlices(f32, want, got);

    try m.resetCache();
    try m.prefill(prompt[0..4]);
    try m.prefillImage(rows, grid_w, grid_h);
    try std.testing.expectEqual(@as(usize, 4 + grid_w * grid_h), m.cached());
    try std.testing.expectEqual(@as(usize, 4 + grid_w), m.pos_next);
    try m.step(io, prompt[5..6], got);
    try m.resetCache();
    try m.prefill(prompt[0..4]);
    try m.prefill(&img_ids);
    try m.step(io, prompt[5..6], want);
    try std.testing.expect(relL2(want, got) > 1e-3);

    const be = cuda.Backend.init(gpa) catch return;
    defer be.deinit();
    var c = try qwen35_cuda.CudaLM.init(gpa, be, &f.lm, .{ .initial = 64, .max = 64 });
    defer c.deinit();
    try c.prefill(prompt[0..4]);
    try c.prefillImage(rows, grid_w, grid_h);
    try std.testing.expectEqual(@as(usize, 4 + grid_w * grid_h), c.cached());
    try std.testing.expectEqual(@as(usize, 4 + grid_w), c.pos_next);
    try c.step(io, prompt[5..6], want);
    const rel = relL2(want, got);
    errdefer std.debug.print("vulkan vs cuda image prefill rel L2 {e:.3}\n", .{rel});
    try std.testing.expect(rel < 5e-2);
}
