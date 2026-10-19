//! qwen35 hybrid (gated DeltaNet) on Vulkan: the forward is `qwen35_dev.zig`, shared
//! with `qwen35_cuda`, over this file's `Dev`, so a prompt prefills in batched chunks
//! (GEMM projections, one multi-query attention, the chunked recurrence) and a token
//! decodes as one recorded batch. Text only, KV capacity fixed at init.
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
    /// `tp-llm --bench` alternates the grouped and per-head attention kernels, which
    /// must give the same logits hash.
    pub const bench_toggle = &gpu.Context.attn_rows_grouped;

    ctx: *gpu.Context,
    lm: *const qwen35.Model,
    cfg: qwen35.Config,
    gpa: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    capacity: usize,
    len: usize,
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
        self.ctx = ctx;
        self.lm = lm;
        self.cfg = cfg;
        self.gpa = gpa;
        self.capacity = cap.max;
        self.len = 0;

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

        const n_attn = cfg.nAttnLayers();
        self.kv = try kv_gpu.Kv.init(alloc, ctx, n_attn, cap.max, cfg.kvDim(), cap.kv_dtype, &.{});

        const n_lin = cfg.n_layers - n_attn;
        const conv_bytes = cfg.convChannels() * (cfg.conv_kernel - 1) * 4;
        const ssm_bytes = cfg.lin_v_heads * cfg.lin_head_dim * cfg.lin_head_dim * 4;
        self.conv_state = try alloc.alloc(Buf, n_lin);
        self.ssm_state = try alloc.alloc(Buf, n_lin);
        for (self.conv_state, self.ssm_state) |*cs, *ss| {
            cs.* = try ctx.tensorCreate(conv_bytes);
            ss.* = try ctx.tensorCreate(ssm_bytes);
            try zeroBuf(ctx, gpa, cs.*, conv_bytes);
            try zeroBuf(ctx, gpa, ss.*, ssm_bytes);
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

    pub fn deinit(self: *VulkanLM) void {
        const ctx = self.ctx;
        self.bufs.deinit(ctx);
        inline for (.{ "freqs_d", "pos3_d", "pos3s_d" }) |f| ctx.tensorDestroy(&@field(self, f));
        self.kv.deinit(self.arena.allocator(), ctx);
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
    pub fn vocab(self: *const VulkanLM) usize {
        return self.cfg.vocab;
    }
    /// Device VRAM (bytes) this Vulkan context has allocated, the analog of the
    /// CUDA backend's `deviceUsed()`, for the end-of-response telemetry.
    pub fn vramUsed(self: *const VulkanLM) u64 {
        return self.ctx.device_used;
    }
    pub fn ensureCapacity(self: *VulkanLM, min_rows: usize) !void {
        if (min_rows > self.capacity) return error.ContextFull;
    }

    /// An empty context: the recurrent state zeroed, the KV rows left to be overwritten.
    pub fn resetCache(self: *VulkanLM) !void {
        const cfg = self.cfg;
        const conv_bytes = cfg.convChannels() * (cfg.conv_kernel - 1) * 4;
        const ssm_bytes = cfg.lin_v_heads * cfg.lin_head_dim * cfg.lin_head_dim * 4;
        for (self.conv_state, self.ssm_state) |cs, ss| {
            try zeroBuf(self.ctx, self.gpa, cs, conv_bytes);
            try zeroBuf(self.ctx, self.gpa, ss, ssm_bytes);
        }
        self.len = 0;
    }

    /// Forward `ids` (batched prefill for all but the last, one decode step for it) and
    /// write the last position's logits.
    pub fn step(self: *VulkanLM, io: std.Io, ids: []const u32, logits: []f32) !void {
        _ = io;
        std.debug.assert(ids.len >= 1 and ids.len <= self.remaining());
        if (ids.len > 1) try self.prefill(ids[0 .. ids.len - 1]);
        try self.stepOne(ids[ids.len - 1]);
        try self.ctx.tensorDownload(self.bufs.logits, std.mem.sliceAsBytes(logits));
    }

    /// Prefill tokens without reading logits, in batched chunks.
    pub fn prefill(self: *VulkanLM, ids: []const u32) !void {
        const cfg = self.cfg;
        const x = try self.gpa.alloc(f32, @min(ids.len, prefill_chunk) * cfg.hidden);
        defer self.gpa.free(x);
        var pos3s: [prefill_chunk * 3]u32 = undefined;
        var off: usize = 0;
        while (off < ids.len) {
            const n: usize = @min(prefill_chunk, ids.len - off);
            try qwen3.embedTokens(self.lm.embed, ids[off..][0..n], x[0 .. n * cfg.hidden]);
            for (0..n) |t| pos3s[t * 3 ..][0..3].* = @splat(@intCast(self.len + t));
            // Uploads before the batch opens: one inside it would flush the recording.
            try self.ctx.tensorUpload(try self.ctx.view(self.bufs.x, 0, n * cfg.hidden * 4), std.mem.sliceAsBytes(x[0 .. n * cfg.hidden]));
            try self.ctx.tensorUpload(self.pos3s_d, std.mem.sliceAsBytes(pos3s[0 .. n * 3]));
            try self.ctx.beginBatch();
            errdefer if (self.ctx.batching) self.ctx.abortBatch();
            var dev: Dev = .{ .m = self, .cfg = cfg, .lm = self.lm, .bufs = &self.bufs };
            try qwen35_dev.batch(&dev, n);
            try self.ctx.endBatch();
            self.len += n;
            off += n;
        }
    }

    /// The final-normed hidden state of every row of `ids`, prefilled at the cache's
    /// current length: what a text encoder reads of this model. `out` is
    /// `[ids.len][hidden]`.
    pub fn encode(self: *VulkanLM, ids: []const u32, out: []f32) !void {
        const cfg = self.cfg;
        const hidden = cfg.hidden;
        std.debug.assert(out.len == ids.len * hidden);
        const x = try self.gpa.alloc(f32, @min(ids.len, prefill_chunk) * hidden);
        defer self.gpa.free(x);
        var pos3s: [prefill_chunk * 3]u32 = undefined;
        var off: usize = 0;
        while (off < ids.len) {
            const n: usize = @min(prefill_chunk, ids.len - off);
            try qwen3.embedTokens(self.lm.embed, ids[off..][0..n], x[0 .. n * hidden]);
            for (0..n) |t| pos3s[t * 3 ..][0..3].* = @splat(@intCast(self.len + t));
            try self.ctx.tensorUpload(try self.ctx.view(self.bufs.x, 0, n * hidden * 4), std.mem.sliceAsBytes(x[0 .. n * hidden]));
            try self.ctx.tensorUpload(self.pos3s_d, std.mem.sliceAsBytes(pos3s[0 .. n * 3]));
            try self.ctx.beginBatch();
            errdefer if (self.ctx.batching) self.ctx.abortBatch();
            var dev: Dev = .{ .m = self, .cfg = cfg, .lm = self.lm, .bufs = &self.bufs };
            try qwen35_dev.batch(&dev, n);
            try dev.rms(self.bufs.x, self.bufs.normed, self.lm.final_norm, n, hidden);
            try self.ctx.endBatch();
            self.len += n;
            try self.ctx.downloadF32At(self.bufs.normed, 0, out[off * hidden ..][0 .. n * hidden]);
            off += n;
        }
    }

    fn stepOne(self: *VulkanLM, id: u32) !void {
        const cfg = self.cfg;
        const x = try self.gpa.alloc(f32, cfg.hidden);
        defer self.gpa.free(x);
        try qwen3.embedTokens(self.lm.embed, &.{id}, x);
        const p: u32 = @intCast(self.len);
        try self.ctx.tensorUpload(self.bufs.x, std.mem.sliceAsBytes(x));
        try self.ctx.tensorUpload(self.pos3_d, std.mem.sliceAsBytes(&[3]u32{ p, p, p }));
        try self.ctx.beginBatch();
        errdefer if (self.ctx.batching) self.ctx.abortBatch();
        var dev: Dev = .{ .m = self, .cfg = cfg, .lm = self.lm, .bufs = &self.bufs };
        try qwen35_dev.decode(&dev, false, true);
        try self.ctx.endBatch();
        self.len += 1;
    }
};

/// `qwen35_dev`'s adapter over a `VulkanLM`.
const Dev = struct {
    m: *VulkanLM,
    cfg: qwen35.Config,
    lm: *const qwen35.Model,
    bufs: *const Bufs,

    fn ctx(d: *const Dev) *gpu.Context {
        return d.m.ctx;
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

    pub fn forwardBegin(_: *const Dev, _: usize) !void {}
    pub fn layerBegin(_: *const Dev, _: usize, _: usize) !bool {
        return false;
    }
    pub fn tapNormedStep(_: *const Dev, _: usize) !void {}
    pub fn tapNormedRows(_: *const Dev, _: usize, _: usize) !void {}
    pub fn tapLayer(_: *const Dev, _: usize, _: usize) !void {}
    pub fn forwardEnd(_: *const Dev, _: usize, _: bool) !void {}
    pub fn headBegin(_: *const Dev) void {}
};
