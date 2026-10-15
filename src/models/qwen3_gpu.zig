//! GPU-resident Qwen3-VL-4B text encoder (Krea 2 conditioning).
//!
//! Mirrors `qwen3.TextEncoder.encode` with the whole 35-layer transformer on
//! the device in a single batched submission, one upload of the embedded
//! tokens in, one download of the 12-tap conditioning stack out. This
//! replaces the CPU forward's sync-per-GEMM offload (a CPU<->GPU ping-pong
//! that kept the GPU idle and made the encode latency-bound); here every op
//! is recorded into one command buffer so the GPU stays saturated.
//!
//! Parity-first: f32-accumulate GEMMs (opMatmul) and the f32 attention path
//! (attn_scores + fused-online-softmax attn_out, causal), so numerics match
//! the CPU forward up to reordering. Verified against the same ComfyUI text
//! fixture as the CPU test.

const std = @import("std");
const build_options = @import("build_options");
const qwen3 = @import("qwen3.zig");
const qwen3_enc_dev = @import("qwen3_enc_dev.zig");
const lin_gpu = @import("lin_gpu.zig");
const gpu = @import("tp_gpu").context;
const safetensors = @import("tp_core").safetensors;
const ops = @import("tp_ops");
const spec = @import("../llm/spec.zig");
const spec_limits = @import("tp_core").spec_limits;
const init_defaults = @import("tp_core").init_defaults;
const sample = @import("tp_core").sample;
const transformer = @import("transformer.zig");
const transformer_gpu = @import("transformer_gpu.zig");
const kv_gpu = @import("kv_gpu.zig");
const kvmod = @import("tp_core").kv_cache;

// Only `head_dim` is a family-wide constant here. Every other dimension is read
// off `enc.cfg` inside `encode`, because Anima's encoder is Qwen3-0.6B (28
// layers, 1024 wide, SwiGLU 3072) where krea2's and Z-Image's are 4B. As module
// constants they make a 0.6B encoder run with 2560-wide strides, which is garbage with
// no error.
const hd = qwen3.head_dim;
const half = hd / 2;
const attn_scale: f32 = 1.0 / @sqrt(@as(f32, hd));

const Buf = gpu.DeviceBuffer;

extern fn getenv(name: [*:0]const u8) ?[*:0]const u8;

/// The encoder's tap stack `[seq][enc.tapCount()][hidden]` (see `qwen3_enc_dev`).
/// Caller frees.
pub fn encode(enc: *const qwen3.TextEncoder, ctx: *gpu.Context, io: std.Io, gpa: std.mem.Allocator, ids: []const u32, act: gpu.Dt, cancel: ?*std.atomic.Value(bool)) ![]f32 {
    return encodeVision(enc, ctx, io, gpa, ids, .{}, act, cancel);
}

/// `encode` with a spliced vision payload: `qwen3_enc_dev.encode` on this backend.
pub fn encodeVision(enc: *const qwen3.TextEncoder, ctx: *gpu.Context, io: std.Io, gpa: std.mem.Allocator, ids: []const u32, vision: qwen3.TextEncoder.Vision, act: gpu.Dt, cancel: ?*std.atomic.Value(bool)) ![]f32 {
    _ = io;
    if (getenv("TP_TE_DTYPES") != null) {
        for ([_]usize{ 0, enc.layers.len - 1 }) |li| {
            const l = enc.layers[li];
            std.log.warn("layer {d}: q#{d} k#{d} v#{d} o#{d} gate#{d} up#{d} down#{d}", .{
                li,                      @intFromEnum(l.q.dtype),    @intFromEnum(l.k.dtype),  @intFromEnum(l.v.dtype),
                @intFromEnum(l.o.dtype), @intFromEnum(l.gate.dtype), @intFromEnum(l.up.dtype), @intFromEnum(l.down.dtype),
            });
        }
    }
    var d: EncDev = .{ .ctx = ctx, .gpa = gpa, .enc = enc, .act = act };
    return qwen3_enc_dev.encode(&d, enc, gpa, ids, vision, cancel);
}

/// The `qwen3_enc_dev` adapter.
pub const EncDev = struct {
    ctx: *gpu.Context,
    gpa: std.mem.Allocator,
    enc: *const qwen3.TextEncoder,
    /// The width the residual, the norms and the FFN planes are stored at; a narrowing
    /// GEMM route stages at it too (f16, else bf16).
    act: gpu.Dt,
    lin: ?lin_gpu.Lin = null,
    plan: lin_gpu.Plan = undefined,
    seq_pad: usize = 0,
    b: Bufs = undefined,
    have_bufs: bool = false,
    freqs: Buf = none,
    ds: []Buf = &.{},

    pub fn setup(d: *EncDev, seq: usize, tap_count: usize, table: []const f32, deepstack: []const []const f32) !void {
        errdefer d.teardown();
        {
            var arena = std.heap.ArenaAllocator.init(d.gpa);
            defer arena.deinit();
            const lins = try d.enc.deviceLins(arena.allocator());
            d.lin = try lin_gpu.Lin.init(d.gpa, d.ctx, lin_gpu.maxRows(lins), 0, 0);
            d.plan = lin_gpu.plan(d.ctx, lins, encoderUse(d.act == .f16), "qwen3_gpu encoder") catch return error.UnsupportedDType;
        }
        d.seq_pad = std.mem.alignForward(usize, seq, 128);
        d.b = try Bufs.init(d.ctx, d.enc.cfg, seq, d.seq_pad, tap_count, d.act);
        d.have_bufs = true;
        d.ds = try d.gpa.alloc(Buf, deepstack.len);
        @memset(d.ds, none);
        // At the residual's width, so the row-span add is one dtype.
        for (deepstack, d.ds) |feat, *b| {
            var f32b = try d.ctx.tensorCreate(feat.len * 4);
            defer if (d.act != .f32) d.ctx.tensorDestroy(&f32b);
            try d.ctx.tensorUpload(f32b, std.mem.sliceAsBytes(feat));
            if (d.act == .f32) {
                b.* = f32b;
                continue;
            }
            b.* = try d.ctx.tensorCreate(feat.len * d.act.bytes());
            b.dt = d.act;
            try d.ctx.opElt(gpu.eltCopy(.f32, d.act), f32b, b.*, null, null, .{ .u0 = @intCast(feat.len) }, feat.len, 1, 1);
        }
        d.freqs = try d.ctx.tensorCreate(table.len * 4);
        try d.ctx.tensorUpload(d.freqs, std.mem.sliceAsBytes(table));
    }

    pub fn teardown(d: *EncDev) void {
        if (d.have_bufs) d.b.deinit(d.ctx);
        d.have_bufs = false;
        for (d.ds) |*b| d.ctx.tensorDestroy(b);
        if (d.ds.len > 0) d.gpa.free(d.ds);
        d.ds = &.{};
        d.ctx.tensorDestroy(&d.freqs);
        if (d.lin) |*l| l.deinit();
        d.lin = null;
    }

    /// Into a 16-bit `b` through the f32 tap stack, which is free until the first tap.
    pub fn upload(d: *EncDev, b: Buf, data: []const f32) !void {
        if (b.dt == .f32) return d.ctx.tensorUpload(b, std.mem.sliceAsBytes(data));
        try d.ctx.tensorUpload(d.b.out, std.mem.sliceAsBytes(data));
        try d.ctx.opElt(gpu.eltCopy(.f32, b.dt), d.b.out, b, null, null, .{ .u0 = @intCast(data.len) }, data.len, 1, 1);
    }

    pub fn download(d: *EncDev, b: Buf, dst: []f32) !void {
        try d.ctx.tensorDownload(b, std.mem.sliceAsBytes(dst));
    }

    pub fn beginBatch(d: *EncDev) !void {
        try d.ctx.beginBatch();
    }

    pub fn endBatch(d: *EncDev) !void {
        try d.ctx.endBatch();
    }

    pub fn abortBatch(d: *EncDev) void {
        if (d.ctx.batching) d.ctx.abortBatch();
    }

    /// The context's own weight LRU streams what does not fit.
    pub fn prefetch(_: *EncDev, _: anytype) void {}

    pub fn rmsNorm(d: *EncDev, x: Buf, y: Buf, w: []const f32, rows: usize, dim: usize, eps: f32) !void {
        try rmsnorm(d.ctx, x, y, try nbuf(d.ctx, w), rows, dim, eps);
    }

    /// GEMM outputs are written 128-row padded.
    pub fn gemms(d: *EncDev, x: Buf, rows: usize, cols: usize, ws: []const lin_gpu.Weight, ys: []const Buf) !void {
        const l = &d.lin.?;
        const g = try l.prep(d.plan, x, rows, cols, ws);
        for (ws, ys) |w, y| try l.gemm(d.plan, g, y, w, .{ .m_pad = d.seq_pad });
    }

    pub fn rope(d: *EncDev, x: Buf, rows: usize, heads: usize) !void {
        try d.ctx.opElt(.rope_half, x, null, d.freqs, null, .{
            .u0 = @intCast(rows * heads * half),
            .u1 = half,
            .u2 = @intCast(rows * half),
            .u3 = @intCast(heads),
        }, rows * heads * half, 1, 1);
    }

    /// Gather k-major, raw scores, then the fused-softmax P@V with the causal mask.
    pub fn attention(d: *EncDev, seq: usize) !void {
        const c = d.enc.cfg;
        const b = d.b;
        try d.ctx.opElt(.gather_kmajor, b.q, null, null, b.qt, .{
            .u0 = @intCast(seq * c.n_heads * hd),
            .u1 = @intCast(c.n_heads),
            .u2 = hd,
            .u3 = @intCast(seq),
        }, seq * c.n_heads * hd, 1, 1);
        try d.ctx.opElt(.gather_kmajor, b.k, null, null, b.kt, .{
            .u0 = @intCast(seq * c.n_kv_heads * hd),
            .u1 = @intCast(c.n_kv_heads),
            .u2 = hd,
            .u3 = @intCast(seq),
        }, seq * c.n_kv_heads * hd, 1, 1);
        const dc8 = std.math.divCeil(usize, seq, 8) catch unreachable;
        try d.ctx.opElt(.attn_scores, b.qt, b.kt, null, b.s, .{
            .u0 = @intCast(seq),
            .u1 = @intCast(c.n_heads),
            .u2 = @intCast(c.n_kv_heads),
            .u3 = hd,
            .u4 = 0,
            .f0 = attn_scale,
        }, dc8, dc8, c.n_heads);
        try d.ctx.opElt(.attn_out, b.s, null, b.v, b.attn, .{
            .u0 = @intCast(seq),
            .u1 = @intCast(c.n_heads),
            .u2 = @intCast(c.n_kv_heads),
            .u3 = hd,
            .u4 = 0,
            .u5 = @intCast(seq),
            .f0 = @bitCast(@as(u32, @intCast(seq * seq))),
            .f1 = @bitCast(@as(u32, 1)),
        }, hd / 8, dc8, c.n_heads);
    }

    pub fn add(d: *EncDev, x: Buf, y: Buf, n: usize) !void {
        std.debug.assert(x.dt == y.dt);
        try d.ctx.opElt(gpu.eltAt("add", x.dt), x, y, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
    }

    /// Row offsets are whole rows of `hidden` floats, which meets any device's storage
    /// alignment for the hidden widths these encoders have.
    pub fn addRows(d: *EncDev, layer: usize, x_off: usize, f_off: usize, n: usize) !void {
        const e = d.act.bytes();
        const xv = try d.ctx.view(d.b.x, x_off * e, n * e);
        const fv = try d.ctx.view(d.ds[layer], f_off * e, n * e);
        try d.add(xv, fv, n);
    }

    pub fn siluMul(d: *EncDev, g: Buf, u: Buf, n: usize) !void {
        std.debug.assert(g.dt == u.dt);
        try d.ctx.opElt(gpu.eltAt("silu_mul", g.dt), g, u, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
    }

    pub fn copyTap(d: *EncDev, src: Buf, off: usize, n: usize) !void {
        try d.ctx.opElt(gpu.eltCopy(src.dt, .f32), src, d.b.out, null, null, .{ .u0 = @intCast(n), .u2 = @intCast(off) }, n, 1, 1);
    }
};

const none: Buf = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 };

/// What the encoder's linears run under. Its conditioning is computed once and steers
/// every sampling step, so a block quant stays on the f16 expansion (the activation
/// unquantized); `use_f16` false keeps every narrowing route off f16, which on this
/// backend means the f32-activation kernels.
fn encoderUse(use_f16: bool) lin_gpu.Use {
    return .{ .blockq = .f16, .act16 = if (use_f16) .f16 else .bf16, .int8 = lin_gpu.dual_lin.encoderInt8() };
}

/// Whether this context can run the encoder's weights at all, every linear by its own
/// storage, so a checkpoint that mixes formats by layer (a Q4_K_M GGUF is q4_k and
/// q6_k) runs here as it does on the CUDA arms.
pub fn supportsWeights(ctx: *gpu.Context, enc: *const qwen3.TextEncoder) bool {
    if (enc.layers.len == 0) return false;
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const lins = enc.deviceLins(arena.allocator()) catch return false;
    for ([_]bool{ false, true }) |half16| switch (lin_gpu.check(ctx, lins, encoderUse(half16))) {
        .ok => {},
        .refused => |r| {
            lin_gpu.dual_lin.explain(r, "qwen3_gpu encoder");
            return false;
        },
    };
    return true;
}

fn rmsnorm(ctx: *gpu.Context, in: Buf, out: Buf, weight: Buf, rows: usize, dim: usize, eps: f32) !void {
    std.debug.assert(in.dt == out.dt);
    try ctx.opElt(gpu.eltAt("rmsnorm", in.dt), in, out, weight, null, .{
        .u0 = @intCast(rows),
        .u1 = @intCast(dim),
        .f0 = eps,
    }, rows, 1, 1);
}

/// Wrap a CPU norm-weight slice as a (pointer-cached) small device buffer.
fn nbuf(ctx: *gpu.Context, weights: []const f32) !Buf {
    const buf = try ctx.smallBuffer(std.mem.sliceAsBytes(weights));
    return .{ .buf = buf, .mem = .null_handle, .size = 0 };
}

const Bufs = struct {
    x: Buf,
    normed: Buf,
    q: Buf,
    k: Buf,
    v: Buf,
    qt: Buf,
    kt: Buf,
    s: Buf,
    attn: Buf,
    gate: Buf,
    up: Buf,
    t: Buf,
    out: Buf,

    /// `act` is the encoder dial's width for the buffers `isTyped` names; the rest
    /// stay f32.
    fn init(ctx: *gpu.Context, c: qwen3.Config, seq: usize, seq_pad: usize, tap_count: usize, act: gpu.Dt) !Bufs {
        var self: Bufs = undefined;
        var created: usize = 0;
        errdefer inline for (fields, 0..) |name, i| {
            if (i < created) ctx.tensorDestroy(&@field(self, name));
        };
        // GEMM outputs (q/k/v/attn/gate/up/t) are 128-row padded for the coop
        // path (pad rows written zero); the rest are indexed by real seq.
        const sizes = [fields.len]usize{
            seq * c.hidden, // x
            seq * c.hidden, // normed
            seq_pad * c.qDim(), // q
            seq_pad * c.kvDim(), // k
            seq_pad * c.kvDim(), // v
            seq * c.qDim(), // qt (k-major)
            seq * c.kvDim(), // kt (k-major)
            c.n_heads * seq * seq, // s (all heads batched)
            seq_pad * c.qDim(), // attn
            seq_pad * c.intermediate, // gate
            seq_pad * c.intermediate, // up
            seq_pad * c.hidden, // t
            tap_count * seq * c.hidden, // out (tap-major)
        };
        inline for (fields, sizes) |name, elems| {
            const w: gpu.Dt = if (comptime isTyped(name)) act else .f32;
            @field(self, name) = try ctx.tensorCreate(elems * w.bytes());
            @field(self, name).dt = w;
            created += 1;
        }
        return self;
    }

    fn deinit(self: *Bufs, ctx: *gpu.Context) void {
        inline for (fields) |name| ctx.tensorDestroy(&@field(self, name));
        self.* = undefined;
    }

    const fields = [_][]const u8{ "x", "normed", "q", "k", "v", "qt", "kt", "s", "attn", "gate", "up", "t", "out" };

    fn isTyped(comptime name: []const u8) bool {
        inline for (.{ "x", "normed", "gate", "up", "t" }) |t| if (comptime std.mem.eql(u8, name, t)) return true;
        return false;
    }
};

/// KV-cached causal LM on the Vulkan backend (tp-llm --backend vulkan), the
/// `qwen3_cuda.CudaLM` of this backend: `transformer_gpu.decoderLayer` over this
/// stepper's ops, every linear through `lin_gpu` (so any format it takes, any head,
/// tied or not) and attention as the dual flash-decoding kernel against the cache, so a
/// chunk at any position takes the same batched forward.
///
/// A batch of at most `gemv_batch_max` rows (a speculative verify) is bit-identical
/// per row to a decode step: the linears take the grouped GEMVs and the attention the
/// decode split count.
pub const VulkanLM = struct {
    /// The prefill batch this stepper wants the engine to hand it.
    pub const prefill_batch = prefill_chunk;
    /// `tp-llm --bench` alternates the grouped and per-head attention kernels, which
    /// must give the same logits hash.
    pub const bench_toggle = &gpu.Context.attn_rows_grouped;

    lm: *const qwen3.CausalLM,
    ctx: *gpu.Context,
    gpa: std.mem.Allocator,
    cfg: qwen3.Config,
    lin: lin_gpu.Lin,
    lin_plan: lin_gpu.Plan,
    capacity: usize,
    /// Committed cache length (absolute position of the next token).
    len: usize = 0,
    sin_off: u32,
    kv: kv_gpu.Kv,
    freqs_d: Buf,
    bufs: LmBufs,
    /// `bufs.t` holds a residual not yet added to `bufs.x`: the next hidden norm adds it
    /// in the same dispatch. Anything else that reads x must `flushResidual` first.
    residual_pending: bool = false,

    /// Rows per batched forward. Each chunk expands every block-quant weight to f16
    /// once, so a bigger one amortizes that: Qwen3-4B q4_k, a 1309-token prompt, 512
    /// rows 1563 tok/s, 1024 1779, 2048 1951.
    const prefill_chunk = 2048;
    /// KV chunks per head in the attention split pass: a decode step and a verify
    /// batch, then a prefill chunk, whose query rows already fill the device.
    const nsplit = 32;
    const nsplit_prefill = 8;
    /// Largest batch that is bit-identical per row to a decode step.
    const gemv_batch_max = spec_limits.max_draft + 1;
    /// Interleaved k chunks per output column in the dense GEMVs.
    const gemv_nchunk = 32;

    /// Device bytes the KV cache reserves up front for `capacity` tokens. Vulkan has no
    /// growable buffers, so a caller sizing a weight-pin budget leaves this much free.
    pub fn kvWindowBytes(cfg: qwen3.Config, capacity: usize, dtype: kvmod.KvDtype) usize {
        return kv_gpu.Kv.windowBytes(cfg.n_layers, capacity, cfg.kvDim(), dtype);
    }

    pub fn init(gpa: std.mem.Allocator, ctx: *gpu.Context, lm: *const qwen3.CausalLM, cap: kvmod.Capacity) !VulkanLM {
        const c = lm.cfg;
        var self: VulkanLM = init_defaults.of(VulkanLM);
        self.lm = lm;
        self.ctx = ctx;
        self.gpa = gpa;
        self.cfg = c;
        self.capacity = cap.max;
        self.len = 0;
        self.sin_off = @intCast(cap.max * half);

        const max_out = @max(@max(c.vocab, c.qDim()), @max(c.intermediate, c.hidden));
        self.lin = try lin_gpu.Lin.init(gpa, ctx, max_out, gemv_nchunk, 3);
        errdefer self.lin.deinit();
        {
            var arena = std.heap.ArenaAllocator.init(gpa);
            defer arena.deinit();
            const lins = try lm.deviceLins(arena.allocator());
            self.lin_plan = try self.lin.plan(lins, lin_gpu.Use.llm(prefill_chunk), "qwen3 vulkan");
            // Not for the head: a prefill reads no logits and a step runs it as a GEMV,
            // while its expansion would be vocab x hidden of scratch.
            var batched: std.ArrayList(lin_gpu.Weight) = .empty;
            for (lins) |w| if (w.bytes.ptr != lm.head.bytes.ptr) try batched.append(arena.allocator(), w);
            try lin_gpu.presize(ctx, self.lin_plan, batched.items);
        }

        var freqs = try ops.rope.rotateHalfFreqs(gpa, cap.max, hd, c.rope_theta);
        defer freqs.deinit(gpa);
        const fp = try gpa.alloc(f32, 2 * cap.max * half);
        defer gpa.free(fp);
        @memcpy(fp[0 .. cap.max * half], freqs.cos);
        @memcpy(fp[cap.max * half ..], freqs.sin);
        self.freqs_d = try ctx.tensorCreate(fp.len * 4);
        errdefer ctx.tensorDestroy(&self.freqs_d);
        try ctx.tensorUpload(self.freqs_d, std.mem.sliceAsBytes(fp));

        self.kv = try kv_gpu.Kv.init(gpa, ctx, c.n_layers, cap.max, c.kvDim(), cap.kv_dtype, &.{});
        errdefer self.kv.deinit(gpa, ctx);
        self.bufs = try LmBufs.init(ctx, c);
        return self;
    }

    pub fn deinit(self: *VulkanLM) void {
        self.kv.deinit(self.gpa, self.ctx);
        self.ctx.tensorDestroy(&self.freqs_d);
        self.bufs.deinit(self.ctx);
        self.lin.deinit();
        self.* = undefined;
    }

    pub fn cached(self: *const VulkanLM) usize {
        return self.len;
    }

    pub fn vocab(self: *const VulkanLM) usize {
        return self.cfg.vocab;
    }

    pub fn remaining(self: *const VulkanLM) usize {
        return self.capacity - self.len;
    }

    pub fn ensureCapacity(self: *VulkanLM, min_rows: usize) !void {
        if (min_rows > self.capacity) return error.ContextFull;
    }

    /// Device VRAM (bytes) this Vulkan context has allocated, for the
    /// end-of-response telemetry.
    pub fn vramUsed(self: *const VulkanLM) u64 {
        return self.ctx.device_used;
    }

    /// Forward `ids` at positions [len, len+ids.len) in chunks of `prefill_chunk`,
    /// then write the last position's vocab logits.
    pub fn step(self: *VulkanLM, io: std.Io, ids: []const u32, logits: []f32) !void {
        try self.forwardLast(io, ids);
        try self.ctx.tensorDownload(self.bufs.logits, std.mem.sliceAsBytes(logits[0..self.cfg.vocab]));
    }

    /// `step`, with vocab logits for every new token ([ids.len, vocab] row-major): the
    /// speculative-decode verify forward, engine-capped at `gemv_batch_max` rows.
    pub fn stepAll(self: *VulkanLM, io: std.Io, ids: []const u32, logits: []f32) !void {
        _ = io;
        const ctx = self.ctx;
        const b = &self.bufs;
        const seq = ids.len;
        std.debug.assert(logits.len == seq * self.cfg.vocab);
        std.debug.assert(seq > 0 and seq <= gemv_batch_max);
        try self.layersForward(ids);
        errdefer if (ctx.batching) ctx.abortBatch();
        try self.hiddenNorm(self.lm.final_norm, seq);
        try self.lin.linear(self.lin_plan, b.logits, 0, b.normed, seq, self.lm.head);
        try ctx.endBatch();
        self.len += seq;
        try ctx.tensorDownload(b.logits, std.mem.sliceAsBytes(logits));
    }

    /// An empty context; there is no recurrent state to clear.
    pub fn resetCache(self: *VulkanLM) !void {
        self.truncate(0);
    }

    /// Roll the KV cache back to `new_len` tokens (speculative-decode rejection);
    /// device rows past it are overwritten by later steps.
    pub fn truncate(self: *VulkanLM, new_len: usize) void {
        std.debug.assert(new_len <= self.len);
        self.len = new_len;
    }

    /// Greedy decode without the vocab download: argmax on the device. Matches
    /// sample.argmax (temperature 0).
    pub fn stepArgmax(self: *VulkanLM, io: std.Io, ids: []const u32) !u32 {
        return self.stepArgmaxPen(io, ids, &.{}, .{});
    }

    /// `stepArgmax` with sampling penalties scattered onto the device logits first
    /// (opPenalize), so penalized greedy decode stays on the GPU.
    pub fn stepArgmaxPen(self: *VulkanLM, io: std.Io, ids: []const u32, pen: []const sample.PenaltyEntry, sp: sample.Params) !u32 {
        try self.forwardLast(io, ids);
        const ctx = self.ctx;
        const b = &self.bufs;
        try ctx.opPenalize(b.logits, pen, sp);
        try ctx.opArgmax(b.logits, self.cfg.vocab, b.argmax_out, &b.argmax_v, &b.argmax_i);
        var id_f: [1]f32 = undefined;
        try ctx.tensorDownload(b.argmax_out, std.mem.sliceAsBytes(&id_f));
        return @intFromFloat(id_f[0]);
    }

    /// Max candidates stepSelect can return (host buffer sizing for the engine).
    pub fn maxSelect(self: *const VulkanLM) usize {
        _ = self;
        return gpu.topk_lanes * gpu.topk_m;
    }

    /// Stochastic decode: forward `ids`, select the top-k candidates on the device and
    /// download only those (id, logit) pairs. Returns the candidate count; the engine's
    /// Sampler finishes on the CPU over this small set.
    pub fn stepSelect(self: *VulkanLM, io: std.Io, ids: []const u32, out_id: []u32, out_logit: []f32) !usize {
        return self.stepSelectPen(io, ids, &.{}, .{}, out_id, out_logit);
    }

    /// `stepSelect` with sampling penalties applied before the top-k, so the selected
    /// candidates are the true post-penalty top set.
    pub fn stepSelectPen(self: *VulkanLM, io: std.Io, ids: []const u32, pen: []const sample.PenaltyEntry, sp: sample.Params, out_id: []u32, out_logit: []f32) !usize {
        try self.forwardLast(io, ids);
        const ctx = self.ctx;
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

    /// Every chunk of `ids`, then the last position's logits left in `bufs.logits`.
    fn forwardLast(self: *VulkanLM, io: std.Io, ids: []const u32) !void {
        _ = io;
        std.debug.assert(ids.len >= 1 and ids.len <= self.remaining());
        const ctx = self.ctx;
        const b = &self.bufs;
        const hidden_ = self.cfg.hidden;
        var off: usize = 0;
        while (off < ids.len) {
            const n: usize = @min(prefill_chunk, ids.len - off);
            try self.layersForward(ids[off..][0..n]);
            errdefer if (ctx.batching) ctx.abortBatch();
            if (off + n == ids.len) {
                // Every row is normed (the residual is pending on all of them); the head
                // reads the last.
                try self.hiddenNorm(self.lm.final_norm, n);
                var last = b.normed;
                if (n > 1) {
                    try ctx.opElt(.copy, b.normed, b.t, null, null, .{
                        .u0 = @intCast(hidden_),
                        .u2 = 0,
                        .u3 = @intCast((n - 1) * hidden_),
                    }, hidden_, 1, 1);
                    last = b.t;
                }
                try self.lin.linear(self.lin_plan, b.logits, 0, last, 1, self.lm.head);
            }
            try ctx.endBatch();
            self.len += n;
            off += n;
        }
    }

    /// The layer stack over `ids` at positions [len, len+seq), inside a batch it leaves
    /// open with the final hidden states in bufs.x. Polls `ops.cancel` between layers.
    fn layersForward(self: *VulkanLM, ids: []const u32) !void {
        const ctx = self.ctx;
        const seq = ids.len;
        std.debug.assert(seq > 0 and seq <= self.remaining() and seq <= prefill_chunk);
        const pos0 = self.len;
        const hidden_ = self.cfg.hidden;

        const x = try self.gpa.alloc(f32, seq * hidden_);
        defer self.gpa.free(x);
        try qwen3.embedTokens(self.lm.embed, ids, x);
        // Before the batch opens: an upload inside it would flush the recording.
        try ctx.tensorUpload(try ctx.view(self.bufs.x, 0, seq * hidden_ * 4), std.mem.sliceAsBytes(x));

        try ctx.beginBatch();
        errdefer if (ctx.batching) ctx.abortBatch();
        self.residual_pending = false;
        const ctok = ops.cancel.token;
        for (self.lm.layers, 0..) |layer, l| {
            if (ops.cancel.canceled(ctok)) return error.Canceled;
            try transformer_gpu.decoderLayer(transformer.qwen3_spec, self, layer, l, seq, pos0);
        }
    }

    // `transformer_gpu.decoderLayer` stepper methods.

    pub fn normInput(self: *VulkanLM, layer: anytype, seq: usize) !void {
        try self.hiddenNorm(layer.input_norm, seq);
    }

    pub fn projectQKV(self: *VulkanLM, l: usize, layer: anytype, seq: usize) !void {
        const b = &self.bufs;
        const kv = try self.kvOut(l, seq);
        if (seq == 1) return self.lin.linearGroup(self.lin_plan, &.{ b.q, kv[0], kv[1] }, b.normed, &.{ layer.q, layer.k, layer.v });
        const g = try self.lin.prep(self.lin_plan, b.normed, seq, self.cfg.hidden, &.{ layer.q, layer.k, layer.v });
        try self.lin.gemm(self.lin_plan, g, b.q, layer.q, .{});
        try self.lin.gemm(self.lin_plan, g, kv[0], layer.k, .{});
        try self.lin.gemm(self.lin_plan, g, kv[1], layer.v, .{});
    }

    /// Where layer `l` projects its k and v: straight into the cache rows when it can
    /// (`Kv.rowsAt`, a batch small enough for the row-exact GEMVs), else the scratch
    /// `appendKV` stores from.
    fn kvOut(self: *VulkanLM, l: usize, seq: usize) ![2]Buf {
        const layer = self.lm.layers[l];
        if (seq <= gemv_batch_max and self.lin.writesRows(self.lin_plan, layer.k, seq) and self.lin.writesRows(self.lin_plan, layer.v, seq))
            if (try self.kv.rowsAt(self.ctx, l, self.len, seq)) |rows| return rows;
        return .{ self.bufs.k, self.bufs.v };
    }

    /// With a q/k norm, the rope too, in the same pass (`applyRope` then has nothing
    /// left): the position is the cache length, which `layersForward` passes as pos0.
    pub fn normQK(self: *VulkanLM, l: usize, layer: anytype, seq: usize) !void {
        const c = self.cfg;
        if (!c.qk_norm) return;
        const ctx = self.ctx;
        const b = &self.bufs;
        if (seq == 1) ctx.independent(2);
        try ctx.opQkNormRope(b.q, try nbuf(ctx, layer.q_norm), self.freqs_d, seq * c.n_heads, hd, c.n_heads, self.len, self.sin_off, c.rms_eps);
        try ctx.opQkNormRope((try self.kvOut(l, seq))[0], try nbuf(ctx, layer.k_norm), self.freqs_d, seq * c.n_kv_heads, hd, c.n_kv_heads, self.len, self.sin_off, c.rms_eps);
    }

    pub fn applyRope(self: *VulkanLM, l: usize, seq: usize, pos0: usize) !void {
        const ctx = self.ctx;
        const b = &self.bufs;
        const c = self.cfg;
        std.debug.assert(pos0 == self.len);
        if (c.qk_norm) return;
        if (seq == 1) ctx.independent(2);
        for ([2]Buf{ b.q, (try self.kvOut(l, seq))[0] }, [2]usize{ c.n_heads, c.n_kv_heads }) |buf, heads| {
            try ctx.opElt(.rope_half, buf, null, self.freqs_d, null, .{
                .u0 = @intCast(seq * heads * half),
                .u1 = half,
                .u2 = self.sin_off,
                .u3 = @intCast(heads),
                .u4 = @intCast(pos0),
            }, seq * heads * half, 1, 1);
        }
    }

    pub fn appendKV(self: *VulkanLM, l: usize, seq: usize, pos0: usize) !void {
        // Projected straight into the cache.
        if ((try self.kvOut(l, seq))[0].buf != self.bufs.k.buf) return;
        if (seq == 1) self.ctx.independent(2);
        try self.kv.store(self.ctx, l, self.bufs.k, self.bufs.v, pos0, seq);
    }

    pub fn attention(self: *VulkanLM, l: usize, seq: usize, pos0: usize) !void {
        const c = self.cfg;
        const b = &self.bufs;
        const ns: usize = if (seq <= gemv_batch_max) nsplit else nsplit_prefill;
        try self.kv.attend(self.ctx, l, b.q, b.attn, b.attn_scratch, pos0 + 1, seq, c.n_heads, c.n_kv_heads, hd, ns, attn_scale, 0, 0);
    }

    pub fn projectO(self: *VulkanLM, l: usize, layer: anytype, seq: usize) !void {
        _ = l;
        try self.lin.linear(self.lin_plan, self.bufs.t, 0, self.bufs.attn, seq, layer.o);
    }

    /// Deferred into the next `hiddenNorm`, which adds it in the same dispatch.
    pub fn addResidual(self: *VulkanLM, seq: usize) !void {
        _ = seq;
        std.debug.assert(!self.residual_pending);
        self.residual_pending = true;
    }

    pub fn normPreFfn(self: *VulkanLM, layer: anytype, seq: usize) !void {
        try self.hiddenNorm(layer.post_norm, seq);
    }

    pub fn projectGateUp(self: *VulkanLM, layer: anytype, seq: usize) !void {
        const b = &self.bufs;
        if (seq == 1) return self.lin.linearGroup(self.lin_plan, &.{ b.gate, b.up }, b.normed, &.{ layer.gate, layer.up });
        const g = try self.lin.prep(self.lin_plan, b.normed, seq, self.cfg.hidden, &.{ layer.gate, layer.up });
        try self.lin.gemm(self.lin_plan, g, b.gate, layer.gate, .{});
        try self.lin.gemm(self.lin_plan, g, b.up, layer.up, .{});
    }

    pub fn activate(self: *VulkanLM, comptime act: transformer.Activation, seq: usize) !void {
        const which: gpu.Elt = switch (act) {
            .silu_mul => .silu_mul,
            .gelu_tanh_mul => .gelu_mul,
        };
        const n = seq * self.cfg.intermediate;
        try self.ctx.opElt(which, self.bufs.gate, self.bufs.up, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
    }

    pub fn projectDown(self: *VulkanLM, layer: anytype, seq: usize) !void {
        try self.lin.linear(self.lin_plan, self.bufs.t, 0, self.bufs.gate, seq, layer.down);
    }

    /// `normed = rmsnorm(x) * w` over `rows` rows, adding the pending residual into x
    /// first.
    fn hiddenNorm(self: *VulkanLM, w: []const f32, rows: usize) !void {
        const b = &self.bufs;
        try self.ctx.opAddRmsNorm(b.x, if (self.residual_pending) b.t else null, try nbuf(self.ctx, w), b.normed, rows, self.cfg.hidden, self.cfg.rms_eps);
        self.residual_pending = false;
    }
};

const LmBufs = struct {
    x: Buf,
    normed: Buf,
    q: Buf,
    k: Buf,
    v: Buf,
    attn: Buf,
    gate: Buf,
    up: Buf,
    t: Buf,
    attn_scratch: Buf,
    logits: Buf,
    // GPU-argmax scratch (opArgmax): per-lane max value + index, and the 1-id out.
    argmax_v: Buf,
    argmax_i: Buf,
    argmax_out: Buf,
    // GPU top-k scratch (opTopK): per-lane top-M values + indices.
    topk_v: Buf,
    topk_i: Buf,

    /// Every activation holds a whole chunk, a multiple of the coop GEMMs' 128-row tile.
    fn init(ctx: *gpu.Context, cfg: qwen3.Config) !LmBufs {
        const r = VulkanLM.prefill_chunk;
        const sizes = [fields.len]usize{
            r * cfg.hidden, // x
            r * cfg.hidden, // normed
            r * cfg.qDim(), // q
            r * cfg.kvDim(), // k
            r * cfg.kvDim(), // v
            r * cfg.qDim(), // attn
            r * cfg.intermediate, // gate
            r * cfg.intermediate, // up
            r * cfg.hidden, // t
            @max(VulkanLM.gemv_batch_max * VulkanLM.nsplit, r * VulkanLM.nsplit_prefill) * cfg.n_heads * (hd + 2), // attn_scratch
            VulkanLM.gemv_batch_max * cfg.vocab, // logits (verify writes a row per position)
            4096, // argmax_v (>= opArgmax lane count)
            4096, // argmax_i
            1, // argmax_out
            gpu.topk_lanes * gpu.topk_m, // topk_v
            gpu.topk_lanes * gpu.topk_m, // topk_i
        };
        var self: LmBufs = undefined;
        var created: usize = 0;
        errdefer inline for (fields, 0..) |name, i| {
            if (i < created) ctx.tensorDestroy(&@field(self, name));
        };
        inline for (fields, sizes) |name, size| {
            @field(self, name) = try ctx.tensorCreate(size * 4);
            created += 1;
        }
        return self;
    }

    fn deinit(self: *LmBufs, ctx: *gpu.Context) void {
        inline for (fields) |name| ctx.tensorDestroy(&@field(self, name));
        self.* = undefined;
    }

    const fields = [_][]const u8{ "x", "normed", "q", "k", "v", "attn", "gate", "up", "t", "attn_scratch", "logits", "argmax_v", "argmax_i", "argmax_out", "topk_v", "topk_i" };
};

// Parity against the CPU encode (same weights, same fixture prompt); gated on
// the model + GPU marker. Compares the GPU-resident forward to the CPU forward
// element-wise.
test "gpu encode matches cpu encode" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const krea2_text = @import("krea2_text.zig");
    const tokenizer_mod = @import("tp_core").tokenizer;
    const te_path = build_options.model_root ++ "/text_encoders/qwen3VLInstruct4bHeretic_v10.safetensors";
    std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    std.Io.Dir.cwd().access(io, te_path, .{}) catch return error.SkipZigTest;

    var ctx = gpu.Context.init(gpa, std.testing.io) catch return error.SkipZigTest;
    defer ctx.deinit();

    var tok = try tokenizer_mod.Tokenizer.init(gpa);
    defer tok.deinit();
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    try krea2_text.buildIds(&tok, gpa, "a fluffy orange cat sitting on a windowsill", &ids);

    var st = try safetensors.SafeTensors.open(gpa, io, te_path);
    defer st.deinit();
    var enc = try qwen3.TextEncoder.load(gpa, .{ .safetensors = &st });
    defer enc.deinit();

    const want = try enc.encode(io, gpa, ids.items, null);
    defer gpa.free(want);

    // f32 (default): near-bit-parity (reduction-order noise only). f16: the residual
    // stored f16 and the coop GEMMs on f16 operands, the DiT's regime.
    inline for (.{ .{ gpu.Dt.f32, 1e-3 }, .{ gpu.Dt.f16, 1e-2 } }) |cfg| {
        const got = try encode(&enc, ctx, io, gpa, ids.items, cfg[0], null);
        defer gpa.free(got);
        try std.testing.expectEqual(want.len, got.len);
        var max_err: f32 = 0;
        var max_val: f32 = 0;
        for (want, got) |e, a| {
            max_err = if (std.math.isNan(a)) std.math.inf(f32) else @max(max_err, @abs(e - a));
            max_val = @max(max_val, @abs(e));
        }
        errdefer std.debug.print("qwen gpu parity ({t}): max_err={d:.5} max_val={d:.2}\n", .{ cfg[0], max_err, max_val });
        try std.testing.expect(max_err < cfg[1] * @max(1.0, max_val));
    }
}

// Speculative decoding on the Vulkan stepper must be byte-identical to
// vanilla greedy: the grouped 4-input GEMVs and the batched flash-decoding
// attention reproduce the decode path's summation orders bitwise. Gated on
// the model + GPU marker; kept tiny (each token is a full 36-layer forward).
test "vulkan spec decode matches vanilla greedy" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const tokenizer_mod = @import("tp_core").tokenizer;
    const chat = @import("../llm/chat.zig");
    const engine = @import("../llm/engine.zig");
    const te_path = build_options.model_root ++ "/text_encoders/qwen3VLInstruct4bHeretic_v10.safetensors";
    std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    std.Io.Dir.cwd().access(io, te_path, .{}) catch return error.SkipZigTest;

    var ctx = gpu.Context.init(gpa, std.testing.io) catch return error.SkipZigTest;
    defer ctx.deinit();

    var st = try safetensors.SafeTensors.open(gpa, io, te_path);
    defer st.deinit();
    var lm = try qwen3.CausalLM.load(gpa, .{ .safetensors = &st });
    defer lm.deinit();
    var tok = try tokenizer_mod.Tokenizer.init(gpa);
    defer tok.deinit();

    var opts: engine.Options = .{ .max_new_tokens = 3, .sampling = .{ .temperature = 0 } };

    var ids_vanilla: std.ArrayList(u32) = .empty;
    defer ids_vanilla.deinit(gpa);
    try chat.appendUser(&tok, gpa, "Count: one two three one two", &ids_vanilla);
    try chat.openAssistant(&tok, gpa, &ids_vanilla);
    var ids_spec: std.ArrayList(u32) = .empty;
    defer ids_spec.deinit(gpa);
    try ids_spec.appendSlice(gpa, ids_vanilla.items);

    {
        var model = try VulkanLM.init(gpa, ctx, &lm, try engine.capacityPlanFor(opts, ids_vanilla.items.len));
        defer model.deinit();
        _ = try engine.generate(&model, &tok, io, gpa, &ids_vanilla, opts, null);
    }
    {
        opts.spec_k = 2;
        var model = try VulkanLM.init(gpa, ctx, &lm, try engine.capacityPlanFor(opts, ids_spec.items.len));
        defer model.deinit();
        _ = try engine.generate(&model, &tok, io, gpa, &ids_spec, opts, null);
    }
    try std.testing.expectEqualSlices(u32, ids_vanilla.items, ids_spec.items);
}

// Regression for bf16 generation. A Vulkan dense matmul / decode-GEMV that handles
// only 1-byte (fp8) and 4-byte (f32) weights reads a bf16 checkpoint's 2-byte weights
// as f32 and produces garbage. bf16 is read natively (2-byte transpose
// `transpose_bf16` plus a bf16 branch in gemv_partial/gemv_partial4, weight code
// `WCode.bf16`). The pre-existing
// spec test above uses the *fp8* encoder checkpoint and only compares
// Vulkan-to-Vulkan, so it never exercised the bf16 path, hence a bf16 model AND
// a comparison against the CPU reference here. Compares the prefill's next-token
// argmax (the crisp signal: garbage diverged at token 1; bf16-GEMV-vs-CPU
// reduction-order drift can diverge multi-token decode, so we don't chase full
// token-identity on this correctness-first path).
test "vulkan bf16 prefill argmax matches cpu" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const tokenizer_mod = @import("tp_core").tokenizer;
    const chat = @import("../llm/chat.zig");
    const engine = @import("../llm/engine.zig");
    const path = build_options.model_root ++ "/text_encoders/qwen3_4b_instruct.safetensors";
    std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    std.Io.Dir.cwd().access(io, path, .{}) catch return error.SkipZigTest;

    var ctx = gpu.Context.init(gpa, std.testing.io) catch return error.SkipZigTest;
    defer ctx.deinit();

    var tok = try tokenizer_mod.Tokenizer.init(gpa);
    defer tok.deinit();
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    try chat.appendUser(&tok, gpa, "The capital of France is", &ids);
    try chat.openAssistant(&tok, gpa, &ids);

    const opts: engine.Options = .{ .max_new_tokens = 1, .sampling = .{ .temperature = 0 } };

    const argmaxOf = struct {
        fn f(logits: []const f32) usize {
            var best: usize = 0;
            for (logits, 0..) |v, i| if (v > logits[best]) {
                best = i;
            };
            return best;
        }
    }.f;

    const cpu_logits = try gpa.alloc(f32, qwen3.vocab_size);
    defer gpa.free(cpu_logits);
    const vk_logits = try gpa.alloc(f32, qwen3.vocab_size);
    defer gpa.free(vk_logits);

    // CPU reference: native bf16 forward.
    {
        var st = try safetensors.SafeTensors.open(gpa, io, path);
        defer st.deinit();
        var lm = try qwen3.CausalLM.load(gpa, .{ .safetensors = &st });
        defer lm.deinit();
        var model = try engine.CpuModel.init(gpa, &lm, try engine.capacityPlanFor(opts, ids.items.len));
        defer model.deinit();
        try model.step(io, ids.items, cpu_logits);
    }
    // Vulkan: native bf16 weights (the fix). Read as f32 before, this was garbage.
    {
        var st = try safetensors.SafeTensors.open(gpa, io, path);
        defer st.deinit();
        var lm = try qwen3.CausalLM.load(gpa, .{ .safetensors = &st });
        defer lm.deinit();
        var model = try VulkanLM.init(gpa, ctx, &lm, try engine.capacityPlanFor(opts, ids.items.len));
        defer model.deinit();
        try model.step(io, ids.items, vk_logits);
    }

    try std.testing.expectEqual(argmaxOf(cpu_logits), argmaxOf(vk_logits));
}
