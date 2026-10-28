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
const bnd = @import("tp_runtime").boundary;
const seqmod = @import("tp_runtime").seq;
const residency = @import("tp_runtime").residency;

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
                li,                     @backingInt(l.q.dtype),    @backingInt(l.k.dtype),  @backingInt(l.v.dtype),
                @backingInt(l.o.dtype), @backingInt(l.gate.dtype), @backingInt(l.up.dtype), @backingInt(l.down.dtype),
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
    /// `tp-llm --bench` alternates the `OpSDot` q8 decode GEMVs (toggle on: off) with
    /// the f32 ones.
    pub const bench_toggle = &gpu.Context.sdot_off;

    lm: *const qwen3.CausalLM,
    be: *gpu.Context,
    gpa: std.mem.Allocator,
    cfg: qwen3.Config,
    lin: lin_gpu.Lin,
    lin_plan: lin_gpu.Plan,
    /// Rows the cache holds now; `ensureCapacity` grows it toward `max_capacity`, and
    /// `resetResidency` shrinks it back to `initial_capacity`.
    capacity: usize,
    initial_capacity: usize,
    max_capacity: usize,
    kv_dtype: kvmod.KvDtype = .f32,
    /// Committed cache length (absolute position of the next token).
    len: usize = 0,
    /// Polled before each prefill chunk: residency, pause and cancel (the GUI).
    boundary: ?bnd.Hook = null,
    /// The Io of the last step, as the CUDA stepper keeps it for its host layers.
    io: ?std.Io = null,
    sin_off: u32,
    kv: kv_gpu.Kv = .empty,
    freqs_d: Buf,
    bufs: LmBufs,
    /// `bufs.t` holds a residual not yet added to `bufs.x`: the next hidden norm adds it
    /// in the same dispatch. Anything else that reads x must `flushResidual` first.
    residual_pending: bool = false,
    /// Layers placed on the host, null when every layer is on the device.
    split: ?Split = null,

    /// The CPU split (`residency.zig` drives it), as qwen3_cuda's: which layers run
    /// on the host, their K/V there, and the scheduling state.
    pub const Split = struct {
        on_gpu: []bool,
        n_cpu: usize,
        policy: residency.CpuSplitPolicy,
        /// Host K/V for the host layers, in the device's byte layout so a migrate is a
        /// copy; its length tracks `len`.
        cache: kvmod.KvCache,
        scratch: qwen3.Scratch,
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

    /// One sequence's context (see `tp_runtime.seq`). `host` is the split's
    /// `cache` and `freqs`, present exactly when a split is armed.
    pub const Seq = struct {
        capacity: usize = 0,
        max_capacity: usize = 0,
        len: usize = 0,
        kv: kv_gpu.Kv = .empty,
        host: ?Host = null,

        pub const Host = struct { cache: kvmod.KvCache, freqs: ops.rope.Freqs };
    };

    /// An empty sequence for the current layer placement.
    fn seqAlloc(self: *const VulkanLM, initial: usize, max: usize) !Seq {
        const c = self.cfg;
        var s: Seq = .{ .capacity = initial, .max_capacity = max };
        errdefer self.seqFree(&s);
        s.kv = try kv_gpu.Kv.init(self.gpa, self.be, c.n_layers, initial, c.kvDim(), self.kv_dtype, &.{});
        if (self.split) |*sp| {
            for (sp.on_gpu, 0..) |on, l| if (!on) s.kv.release(self.be, l);
            var cache = try kvmod.KvCache.init(self.gpa, c.n_layers, initial, c.kvDim(), self.kv_dtype);
            errdefer cache.deinit(self.gpa);
            s.host = .{ .cache = cache, .freqs = try ops.rope.rotateHalfFreqs(self.gpa, initial, hd, c.rope_theta) };
        }
        return s;
    }

    fn seqFree(self: *const VulkanLM, s: *Seq) void {
        s.kv.deinit(self.gpa, self.be);
        if (s.host) |*h| {
            h.cache.deinit(self.gpa);
            h.freqs.deinit(self.gpa);
        }
        s.* = .{};
    }

    /// A second, empty sequence of `rows` rows that never grows (see `qwen3_cuda`'s).
    pub fn seqCreate(self: *VulkanLM, rows: usize) !Seq {
        if (rows > self.max_capacity) return error.ContextFull;
        return self.seqAlloc(rows, rows);
    }

    /// Exchange the live sequence with `s`.
    pub fn seqSwap(self: *VulkanLM, s: *Seq) void {
        seqmod.swapShared(Seq, self, s);
        if (self.split) |*sp| {
            const h = &s.host.?;
            std.mem.swap(kvmod.KvCache, &sp.cache, &h.cache);
            std.mem.swap(ops.rope.Freqs, &sp.freqs, &h.freqs);
        } else std.debug.assert(s.host == null);
    }

    /// Free a sequence made by `seqCreate` once the device is done with it.
    pub fn seqDestroy(self: *VulkanLM, s: *Seq) void {
        self.be.waitIdle();
        self.seqFree(s);
    }

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

    /// Device bytes the KV holds at `capacity` tokens, what a caller sizing a window
    /// leaves free for the cache to grow into.
    pub fn kvWindowBytes(cfg: qwen3.Config, capacity: usize, dtype: kvmod.KvDtype) usize {
        return kv_gpu.Kv.windowBytes(cfg.n_layers, capacity, cfg.kvDim(), dtype);
    }

    pub fn init(gpa: std.mem.Allocator, ctx: *gpu.Context, lm: *const qwen3.CausalLM, cap: kvmod.Capacity) !VulkanLM {
        const c = lm.cfg;
        var self: VulkanLM = init_defaults.of(VulkanLM);
        self.lm = lm;
        self.be = ctx;
        self.gpa = gpa;
        self.cfg = c;
        self.initial_capacity = @max(@min(cap.initial, cap.max), 1);
        self.kv_dtype = cap.kv_dtype;
        self.sin_off = @intCast(cap.max * half);
        ctx.weight_noise.setDepth(c.n_layers);

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

        var first = try self.seqAlloc(self.initial_capacity, cap.max);
        self.seqSwap(&first);
        errdefer {
            var live: Seq = .{};
            self.seqSwap(&live);
            self.seqFree(&live);
        }
        self.bufs = try LmBufs.init(ctx, c);
        return self;
    }

    pub fn deinit(self: *VulkanLM) void {
        self.freeSplit();
        var live: Seq = .{};
        self.seqSwap(&live);
        self.seqFree(&live);
        self.be.tensorDestroy(&self.freqs_d);
        self.bufs.deinit(self.be);
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

    /// Grow the cache to hold `min_rows`, by the CUDA steppers' growth policy. A dynamic
    /// split first moves layers to the host until the device KV can grow, and a real
    /// out-of-memory moves one more and retries, so a full window fails only once
    /// nothing is left to move.
    pub fn ensureCapacity(self: *VulkanLM, min_rows: usize) !void {
        if (min_rows <= self.capacity) return;
        if (min_rows > self.max_capacity) return error.ContextFull;
        const to = kvmod.growTarget(self.capacity, min_rows, self.max_capacity);
        if (self.split) |*sp| if (sp.dynamic) {
            const add = self.kv_dtype.sizeBytes((to - self.capacity) * self.cfg.kvDim());
            while (true) {
                const need = self.liveSlots() * 2 * add + (32 << 20);
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
            sp.cache.grow(self.gpa, to) catch return error.ContextFull;
            const freqs = ops.rope.rotateHalfFreqs(self.gpa, to, hd, self.cfg.rope_theta) catch return error.ContextFull;
            sp.freqs.deinit(self.gpa);
            sp.freqs = freqs;
        }
        self.capacity = to;
    }

    /// Advance the committed length by `n` new rows, the host shadow with it.
    fn advance(self: *VulkanLM, n: usize) void {
        self.len += n;
        if (self.split) |*sp| sp.cache.commit(n);
    }

    pub fn capacityMax(self: *const VulkanLM) usize {
        return self.max_capacity;
    }

    /// Forward `ids` into the cache without the head: a prompt the caller samples
    /// from later.
    pub fn prefill(self: *VulkanLM, ids: []const u32) !void {
        std.debug.assert(ids.len <= self.remaining());
        var off: usize = 0;
        while (off < ids.len) {
            try bnd.check(self);
            const n: usize = @min(prefill_chunk, ids.len - off);
            try self.layersForward(ids[off..][0..n]);
            errdefer if (self.be.batching) self.be.abortBatch();
            try self.be.endBatch();
            self.advance(n);
            off += n;
        }
    }

    /// Upload every weight now rather than inside the first turn's prefill, by running
    /// a decode-sized and a prefill-sized forward at position 0: those are the routes,
    /// and so the device layouts, real turns read. Their rows are overwritten by the
    /// first prefill.
    pub fn warmWeights(self: *VulkanLM) void {
        if (self.len != 0) return;
        const n = @min(gemv_batch_max + 1, self.capacity);
        const zeros: [(gemv_batch_max + 1)]u32 = @splat(0);
        defer self.truncate(0);
        self.prefill(zeros[0..1]) catch |err| std.log.warn("qwen3 vulkan warm-up: {t}", .{err});
        self.truncate(0);
        self.forwardLast(zeros[0..n]) catch |err| std.log.warn("qwen3 vulkan warm-up: {t}", .{err});
    }

    /// Device VRAM (bytes) this Vulkan context has allocated, for the
    /// end-of-response telemetry.
    pub fn vramUsed(self: *const VulkanLM) u64 {
        return self.be.device_used;
    }

    /// Forward `ids` at positions [len, len+ids.len) in chunks of `prefill_chunk`,
    /// then write the last position's vocab logits.
    pub fn step(self: *VulkanLM, io: std.Io, ids: []const u32, logits: []f32) !void {
        self.io = io;
        try self.forwardLast(ids);
        try self.be.tensorDownload(self.bufs.logits, std.mem.sliceAsBytes(logits[0..self.cfg.vocab]));
    }

    /// `step`, with vocab logits for every new token ([ids.len, vocab] row-major): the
    /// speculative-decode verify forward, engine-capped at `gemv_batch_max` rows.
    pub fn stepAll(self: *VulkanLM, io: std.Io, ids: []const u32, logits: []f32) !void {
        _ = io;
        const ctx = self.be;
        const b = &self.bufs;
        const seq = ids.len;
        std.debug.assert(logits.len == seq * self.cfg.vocab);
        std.debug.assert(seq > 0 and seq <= gemv_batch_max);
        try self.layersForward(ids);
        errdefer if (ctx.batching) ctx.abortBatch();
        try self.hiddenNorm(self.lm.final_norm, seq);
        self.be.weight_noise.atHead();
        try self.lin.linear(self.lin_plan, b.logits, 0, b.normed, seq, self.lm.head);
        try ctx.endBatch();
        self.advance(seq);
        try ctx.tensorDownload(b.logits, std.mem.sliceAsBytes(logits));
    }

    /// An empty context; there is no recurrent state to clear.
    pub fn resetCache(self: *VulkanLM) !void {
        self.truncate(0);
    }

    /// The KV is append-only, so a turn boundary needs no snapshot: restoring it is
    /// a truncate, as for a speculative rejection.
    pub fn checkpointBytes(self: *const VulkanLM) usize {
        _ = self;
        return 0;
    }

    pub fn checkpoint(self: *VulkanLM, out: []u8) !void {
        _ = self;
        std.debug.assert(out.len == 0);
    }

    pub fn restoreCheckpoint(self: *VulkanLM, snap: []const u8, q: usize) !void {
        std.debug.assert(snap.len == 0);
        self.truncate(q);
    }

    /// Re-create the cache at another element type, empty.
    pub fn reinitCache(self: *VulkanLM, dtype: kvmod.KvDtype) !void {
        var kv = try kv_gpu.Kv.init(self.gpa, self.be, self.cfg.n_layers, self.capacity, self.cfg.kvDim(), dtype, &.{});
        errdefer kv.deinit(self.gpa, self.be);
        if (self.split) |*sp| {
            for (sp.on_gpu, 0..) |on, l| if (!on) kv.release(self.be, l);
            const cache = try kvmod.KvCache.init(self.gpa, self.cfg.n_layers, self.capacity, self.cfg.kvDim(), dtype);
            sp.cache.deinit(self.gpa);
            sp.cache = cache;
        }
        self.kv.deinit(self.gpa, self.be);
        self.kv = kv;
        self.kv_dtype = dtype;
        self.len = 0;
    }

    // Residency: the hooks `residency.zig` drives, with qwen3_cuda's meaning.

    fn layerDeviceBytes(layer: anytype) usize {
        return layer.q.bytes.len + layer.k.bytes.len + layer.v.bytes.len + layer.o.bytes.len +
            layer.gate.bytes.len + layer.up.bytes.len + layer.down.bytes.len;
    }

    pub fn layerWeightBytes(self: *VulkanLM, l: usize) usize {
        return layerDeviceBytes(&self.lm.layers[l]);
    }

    /// The head; the embeddings are looked up on the host.
    pub fn nonLayerDeviceBytes(self: *VulkanLM) u64 {
        return self.lm.head.bytes.len;
    }

    /// VRAM a promote of layer `l` needs: its weights, its K/V at capacity, and slack.
    pub fn promoteCost(self: *VulkanLM, l: usize) usize {
        const kv_at_cap = 2 * self.kv_dtype.sizeBytes(self.capacity * self.cfg.kvDim());
        return layerDeviceBytes(&self.lm.layers[l]) + kv_at_cap + residency.promote_slack;
    }

    fn liveSlots(self: *const VulkanLM) usize {
        return if (self.split) |sp| self.cfg.n_layers - sp.n_cpu else self.cfg.n_layers;
    }

    /// Move layer `l` to the host: its live K/V into the shadow, its device K/V and
    /// weights freed.
    pub fn migrateLayer(self: *VulkanLM, l: usize) !void {
        const sp = &self.split.?;
        try self.kv.dropLayer(self.be, l, self.len, &sp.cache, l);
        const layer = &self.lm.layers[l];
        for ([_][]const u8{ layer.q.bytes, layer.k.bytes, layer.v.bytes, layer.o.bytes, layer.gate.bytes, layer.up.bytes, layer.down.bytes }) |w|
            self.be.evictWeightBytes(w);
        sp.on_gpu[l] = false;
        sp.n_cpu += 1;
        std.log.debug("[offload] layer {d} -> CPU at ctx {d} ({d}/{d} on CPU)", .{ l, self.len, sp.n_cpu, self.cfg.n_layers });
    }

    /// Bring layer `l` back to the device with its K/V; its weights re-upload on the
    /// next forward.
    pub fn promoteLayer(self: *VulkanLM, l: usize) !void {
        const sp = &self.split.?;
        try self.kv.restoreLayer(self.be, l, self.capacity, self.len, &sp.cache, l);
        sp.on_gpu[l] = true;
        sp.n_cpu -= 1;
        std.log.debug("[promote] layer {d} -> GPU at ctx {d} ({d}/{d} on CPU)", .{ l, self.len, sp.n_cpu, self.cfg.n_layers });
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

    /// Always arm the dynamic split (`budget == 0` is no offload): free while the model
    /// fits, and it moves layers to the host as the KV grows.
    pub fn autoOffload(self: *VulkanLM, budget: u64) !bool {
        if (budget == 0) return false;
        try self.enableCpuSplit(.attn, budget, true);
        return true;
    }

    /// Place layers on the host until the device-resident weights fit under `budget`.
    /// `dynamic` packs the device now and migrates as the KV grows; a static split
    /// that needs no host layer leaves `split` null.
    pub fn enableCpuSplit(self: *VulkanLM, policy: residency.CpuSplitPolicy, budget: u64, dynamic: bool) !void {
        std.debug.assert(self.split == null);
        const c = self.cfg;
        const n = c.n_layers;
        const gpa = self.gpa;
        const per = try gpa.alloc(usize, n);
        defer gpa.free(per);
        for (self.lm.layers, 0..) |*layer, l| per[l] = layerDeviceBytes(layer);
        const kv_bytes = 2 * n * self.kv_dtype.sizeBytes(self.capacity * c.kvDim());
        const reserve = if (dynamic) self.lm.head.bytes.len else kv_bytes + self.lm.head.bytes.len + (512 << 20);
        // The last layer leaves first; qwen3 is all attention, so `.attn` and `.tail` agree.
        const order = try gpa.alloc(usize, n);
        errdefer gpa.free(order);
        for (0..n) |i| order[i] = n - 1 - i;
        const on_gpu = try gpa.alloc(bool, n);
        errdefer gpa.free(on_gpu);
        @memset(on_gpu, true);
        const n_cpu = residency.splitHostCount(self, budget, reserve, dynamic, per, order);
        if (n_cpu == 0 and !dynamic) {
            gpa.free(on_gpu);
            gpa.free(order);
            return;
        }
        var cache = try kvmod.KvCache.init(gpa, n, self.capacity, c.kvDim(), self.kv_dtype);
        errdefer cache.deinit(gpa);
        cache.len = self.len;
        var scratch = try qwen3.Scratch.init(gpa, prefill_chunk, c);
        errdefer scratch.deinit(gpa);
        var freqs = try ops.rope.rotateHalfFreqs(gpa, self.capacity, hd, c.rope_theta);
        errdefer freqs.deinit(gpa);
        const hx = try gpa.alloc(f32, prefill_chunk * c.hidden);
        errdefer gpa.free(hx);
        self.split = .{
            .on_gpu = on_gpu,
            .n_cpu = 0,
            .policy = policy,
            .cache = cache,
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
        sp.cache.deinit(self.gpa);
        sp.scratch.deinit(self.gpa);
        sp.freqs.deinit(self.gpa);
        self.gpa.free(sp.hx);
        self.gpa.free(sp.on_gpu);
        self.gpa.free(sp.order);
        self.split = null;
    }

    /// New chat: drop the split, shrink the KV back to its initial capacity, empty it,
    /// and re-arm the dynamic offload for the fresh context.
    pub fn resetResidency(self: *VulkanLM, budget: u64) !void {
        self.freeSplit();
        const kv = try kv_gpu.Kv.init(self.gpa, self.be, self.cfg.n_layers, self.initial_capacity, self.cfg.kvDim(), self.kv_dtype, &.{});
        self.kv.deinit(self.gpa, self.be);
        self.kv = kv;
        self.capacity = self.initial_capacity;
        try self.resetCache();
        _ = try self.autoOffload(budget);
    }

    /// Roll the KV cache back to `new_len` tokens (speculative-decode rejection);
    /// device rows past it are overwritten by later steps.
    pub fn truncate(self: *VulkanLM, new_len: usize) void {
        std.debug.assert(new_len <= self.len);
        self.len = new_len;
        if (self.split) |*sp| sp.cache.truncate(new_len);
    }

    /// Greedy decode without the vocab download: argmax on the device. Matches
    /// sample.argmax (temperature 0).
    pub fn stepArgmax(self: *VulkanLM, io: std.Io, ids: []const u32) !u32 {
        return self.stepArgmaxPen(io, ids, &.{}, .{});
    }

    /// `stepArgmax` with sampling penalties scattered onto the device logits first
    /// (opPenalize), so penalized greedy decode stays on the GPU.
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

    /// Every chunk of `ids`, then the last position's logits left in `bufs.logits`.
    fn forwardLast(self: *VulkanLM, ids: []const u32) !void {
        std.debug.assert(ids.len >= 1 and ids.len <= self.remaining());
        const ctx = self.be;
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
                self.be.weight_noise.atHead();
                try self.lin.linear(self.lin_plan, b.logits, 0, last, 1, self.lm.head);
            }
            try ctx.endBatch();
            self.advance(n);
            off += n;
        }
    }

    /// The layer stack over `ids` at positions [len, len+seq), inside a batch it leaves
    /// open with the final hidden states in bufs.x. Polls `ops.cancel` between layers.
    fn layersForward(self: *VulkanLM, ids: []const u32) !void {
        const ctx = self.be;
        const seq = ids.len;
        std.debug.assert(seq > 0 and seq <= self.remaining() and seq <= prefill_chunk);
        const pos0 = self.len;
        const hidden_ = self.cfg.hidden;

        const x = try self.gpa.alloc(f32, seq * hidden_);
        defer self.gpa.free(x);
        try qwen3.embedTokens(self.lm.embed, ids, x);
        // Before the batch opens: an upload inside it would flush the recording.
        try ctx.tensorUpload(try ctx.view(self.bufs.x, 0, seq * hidden_ * 4), std.mem.sliceAsBytes(x));

        // One perturbation per forward; nothing while sigma is 0.
        ctx.weight_noise.tick();
        try ctx.beginBatch();
        errdefer if (ctx.batching) ctx.abortBatch();
        self.residual_pending = false;
        if (self.split) |*sp| sp.on_host = false;
        const ctok = ops.cancel.token;
        const c = self.cfg;
        for (self.lm.layers, 0..) |layer, l| {
            if (ops.cancel.canceled(ctok)) return error.Canceled;
            // A host layer runs the CPU backend's own layer body; the hidden crosses
            // only where residency changes.
            if (self.split) |*sp| {
                if (!sp.on_gpu[l]) {
                    if (!sp.on_host) {
                        try self.flushResidual(seq);
                        try ctx.endBatch();
                        try ctx.tensorDownload(try ctx.view(self.bufs.x, 0, seq * hidden_ * 4), std.mem.sliceAsBytes(sp.hx[0 .. seq * hidden_]));
                        sp.on_host = true;
                    }
                    const host_io = self.io orelse return error.SplitIoUnset;
                    var sv = sp.scratch.viewSeq(seq, c);
                    const hx = sp.hx[0 .. seq * hidden_];
                    if (c.qk_norm)
                        try transformer.layerForward(transformer.qwen3_spec, .cached, host_io, self.gpa, layer, hx, seq, qwen3.dimsFor(c), sp.freqs, c.rms_eps, &sp.cache, l, pos0, false, &sv)
                    else
                        try transformer.layerForward(transformer.llama_spec, .cached, host_io, self.gpa, layer, hx, seq, qwen3.dimsFor(c), sp.freqs, c.rms_eps, &sp.cache, l, pos0, false, &sv);
                    continue;
                }
                if (sp.on_host) try self.hiddenToDevice(seq);
            }
            try transformer_gpu.decoderLayer(transformer.qwen3_spec, self, layer, l, seq, pos0);
        }
        // The head reads the device hidden, inside the batch the caller ends.
        if (self.split) |*sp| if (sp.on_host) try self.hiddenToDevice(seq);
    }

    /// The host hidden back into `bufs.x`, and a batch open again: an upload inside
    /// one would flush its recording.
    fn hiddenToDevice(self: *VulkanLM, seq: usize) !void {
        const sp = &self.split.?;
        const n = seq * self.cfg.hidden;
        try self.be.tensorUpload(try self.be.view(self.bufs.x, 0, n * 4), std.mem.sliceAsBytes(sp.hx[0..n]));
        sp.on_host = false;
        try self.be.beginBatch();
    }

    /// Add a pending residual into `bufs.x`, for a reader other than the next norm.
    fn flushResidual(self: *VulkanLM, seq: usize) !void {
        if (!self.residual_pending) return;
        const n = seq * self.cfg.hidden;
        try self.be.opElt(gpu.eltAt("add", .f32), self.bufs.x, self.bufs.t, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
        self.residual_pending = false;
    }

    // `transformer_gpu.decoderLayer` stepper methods.

    /// Which layer the coming launches belong to, for the weight-noise curve.
    pub fn noiseAtLayer(self: *VulkanLM, l: usize) void {
        self.be.weight_noise.atLayer(l);
    }

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
            if (try self.kv.rowsAt(self.be, l, self.len, seq)) |rows| return rows;
        return .{ self.bufs.k, self.bufs.v };
    }

    /// With a q/k norm, the rope too, in the same pass (`applyRope` then has nothing
    /// left): the position is the cache length, which `layersForward` passes as pos0.
    pub fn normQK(self: *VulkanLM, l: usize, layer: anytype, seq: usize) !void {
        const c = self.cfg;
        if (!c.qk_norm) return;
        const ctx = self.be;
        const b = &self.bufs;
        if (seq == 1) ctx.independent(2);
        try ctx.opQkNormRope(b.q, try nbuf(ctx, layer.q_norm), self.freqs_d, seq * c.n_heads, hd, c.n_heads, self.len, self.sin_off, c.rms_eps);
        try ctx.opQkNormRope((try self.kvOut(l, seq))[0], try nbuf(ctx, layer.k_norm), self.freqs_d, seq * c.n_kv_heads, hd, c.n_kv_heads, self.len, self.sin_off, c.rms_eps);
    }

    pub fn applyRope(self: *VulkanLM, l: usize, seq: usize, pos0: usize) !void {
        const ctx = self.be;
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
        if (seq == 1) self.be.independent(2);
        try self.kv.store(self.be, l, self.bufs.k, self.bufs.v, pos0, seq);
    }

    pub fn attention(self: *VulkanLM, l: usize, seq: usize, pos0: usize) !void {
        const c = self.cfg;
        const b = &self.bufs;
        const ns: usize = if (seq <= gemv_batch_max) nsplit else nsplit_prefill;
        try self.kv.attend(self.be, l, b.q, b.attn, b.attn_scratch, pos0 + 1, seq, c.n_heads, c.n_kv_heads, hd, ns, attn_scale, 0, 0);
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
        try self.be.opElt(which, self.bufs.gate, self.bufs.up, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
    }

    pub fn projectDown(self: *VulkanLM, layer: anytype, seq: usize) !void {
        try self.lin.linear(self.lin_plan, self.bufs.t, 0, self.bufs.gate, seq, layer.down);
    }

    /// `normed = rmsnorm(x) * w` over `rows` rows, adding the pending residual into x
    /// first.
    fn hiddenNorm(self: *VulkanLM, w: []const f32, rows: usize) !void {
        const b = &self.bufs;
        try self.be.opAddRmsNorm(b.x, if (self.residual_pending) b.t else null, try nbuf(self.be, w), b.normed, rows, self.cfg.hidden, self.cfg.rms_eps);
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
        // Per tap, so a failure says at which depth it starts.
        errdefer {
            const hidden = enc.cfg.hidden;
            const taps = enc.tapCount();
            for (0..taps) |tp| {
                var te: f32 = 0;
                var tv: f32 = 0;
                for (0..ids.items.len) |t| {
                    const off = (t * taps + tp) * hidden;
                    for (want[off..][0..hidden], got[off..][0..hidden]) |e, a| {
                        te = if (std.math.isNan(a)) std.math.inf(f32) else @max(te, @abs(e - a));
                        tv = @max(tv, @abs(e));
                    }
                }
                std.debug.print("  tap {d} (layer {d}): max_err={d:.5} max_val={d:.2}\n", .{ tp, enc.taps[tp], te, tv });
            }
        }
        errdefer std.debug.print("qwen gpu parity ({t}): max_err={d:.5} max_val={d:.2}\n", .{ cfg[0], max_err, max_val });
        try std.testing.expect(max_err < cfg[1] * @max(1.0, max_val));
    }
}

/// Two adapters, f32 and a 16-bit width, driven in lockstep through the shared forward,
/// each op's output compared as it is written: the first op that diverges is the defect.
const TraceDev = struct {
    arms: [2]*EncDev,
    ctx: *gpu.Context,
    gpa: std.mem.Allocator,
    /// Arm 0's buffers, the ones the forward names; `twin` maps each to arm 1's.
    b: Bufs = undefined,
    host: [2][]f32 = .{ &.{}, &.{} },
    scratch: Buf = none,
    steps: std.ArrayList(Step) = .empty,

    const Step = struct { what: []const u8, rel: f64, routes: [2]?lin_gpu.Route = .{ null, null } };

    fn twin(t: *const TraceDev, x: Buf) Buf {
        inline for (Bufs.fields) |name| if (@field(t.arms[0].b, name).buf == x.buf) return @field(t.arms[1].b, name);
        unreachable;
    }

    fn fetch(t: *TraceDev, x: Buf, out: []f32) !void {
        if (x.dt == .f32) return t.ctx.tensorDownload(x, std.mem.sliceAsBytes(out));
        try t.ctx.opElt(gpu.eltCopy(x.dt, .f32), x, t.scratch, null, null, .{ .u0 = @intCast(out.len) }, out.len, 1, 1);
        try t.ctx.tensorDownload(t.scratch, std.mem.sliceAsBytes(out));
    }

    fn check(t: *TraceDev, what: []const u8, x: Buf, n: usize) !void {
        try t.fetch(x, t.host[0][0..n]);
        try t.fetch(t.twin(x), t.host[1][0..n]);
        var num: f64 = 0;
        var den: f64 = 0;
        for (t.host[0][0..n], t.host[1][0..n]) |a, b| {
            num += (@as(f64, a) - b) * (@as(f64, a) - b);
            den += @as(f64, a) * a;
        }
        try t.steps.append(t.gpa, .{ .what = what, .rel = if (std.math.isFinite(num)) @sqrt(num / @max(den, 1e-30)) else std.math.inf(f64) });
    }

    pub fn setup(t: *TraceDev, seq: usize, tap_count: usize, table: []const f32, deepstack: []const []const f32) !void {
        for (t.arms) |a| try a.setup(seq, tap_count, table, deepstack);
        t.b = t.arms[0].b;
        const c = t.arms[0].enc.cfg;
        const most = @max(t.arms[0].seq_pad * @max(c.intermediate, c.qDim()), c.n_heads * seq * seq, tap_count * seq * c.hidden);
        t.scratch = try t.ctx.tensorCreate(most * 4);
        for (&t.host) |*h| h.* = try t.gpa.alloc(f32, most);
    }

    pub fn teardown(t: *TraceDev) void {
        for (t.arms) |a| a.teardown();
        t.ctx.tensorDestroy(&t.scratch);
        for (&t.host) |*h| t.gpa.free(h.*);
    }

    pub fn upload(t: *TraceDev, x: Buf, data: []const f32) !void {
        try t.arms[0].upload(x, data);
        try t.arms[1].upload(t.twin(x), data);
        try t.check("upload", x, data.len);
    }

    pub fn download(t: *TraceDev, x: Buf, dst: []f32) !void {
        try t.arms[0].download(x, dst);
    }

    // Unbatched, so every op's output can be read back as it lands.
    pub fn beginBatch(_: *TraceDev) !void {}
    pub fn endBatch(_: *TraceDev) !void {}
    pub fn abortBatch(_: *TraceDev) void {}
    pub fn prefetch(_: *TraceDev, _: anytype) void {}

    pub fn rmsNorm(t: *TraceDev, x: Buf, y: Buf, w: []const f32, rows: usize, dim: usize, eps: f32) !void {
        try t.arms[0].rmsNorm(x, y, w, rows, dim, eps);
        try t.arms[1].rmsNorm(t.twin(x), t.twin(y), w, rows, dim, eps);
        try t.check("rmsnorm", y, rows * dim);
    }

    pub fn gemms(t: *TraceDev, x: Buf, rows: usize, cols: usize, ws: []const lin_gpu.Weight, ys: []const Buf) !void {
        var ys1: [3]Buf = undefined;
        for (ys, 0..) |y, i| ys1[i] = t.twin(y);
        try t.arms[0].gemms(x, rows, cols, ws, ys);
        try t.arms[1].gemms(t.twin(x), rows, cols, ws, ys1[0..ys.len]);
        for (ws, ys) |w, y| {
            try t.check(w.tag orelse "gemm", y, rows * w.rows);
            const last = &t.steps.items[t.steps.items.len - 1];
            for (t.arms, &last.routes) |a, *r| r.* = a.lin.?.routeAt(a.plan, w, rows);
        }
    }

    pub fn rope(t: *TraceDev, x: Buf, rows: usize, heads: usize) !void {
        try t.arms[0].rope(x, rows, heads);
        try t.arms[1].rope(t.twin(x), rows, heads);
        try t.check("rope", x, rows * heads * hd);
    }

    pub fn attention(t: *TraceDev, seq: usize) !void {
        try t.arms[0].attention(seq);
        try t.arms[1].attention(seq);
        try t.check("attention", t.b.attn, seq * t.arms[0].enc.cfg.qDim());
    }

    pub fn add(t: *TraceDev, x: Buf, y: Buf, n: usize) !void {
        try t.arms[0].add(x, y, n);
        try t.arms[1].add(t.twin(x), t.twin(y), n);
        try t.check("add", x, n);
    }

    pub fn addRows(_: *TraceDev, _: usize, _: usize, _: usize, _: usize) !void {
        unreachable;
    }

    pub fn siluMul(t: *TraceDev, g: Buf, u: Buf, n: usize) !void {
        try t.arms[0].siluMul(g, u, n);
        try t.arms[1].siluMul(t.twin(g), t.twin(u), n);
        try t.check("silu_mul", g, n);
    }

    pub fn copyTap(t: *TraceDev, src: Buf, off: usize, n: usize) !void {
        try t.arms[0].copyTap(src, off, n);
        try t.arms[1].copyTap(t.twin(src), off, n);
    }
};

// Each op of two layers at f16 against the same op at f32, so a failure names the op.
test "gpu encode at f16 tracks f32 op by op" {
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
    var two = enc;
    two.layers = enc.layers[0..2];
    two.cfg.n_layers = 2;
    two.taps = &.{2};
    two.final_norm = null;

    var a32: EncDev = .{ .ctx = ctx, .gpa = gpa, .enc = &two, .act = .f32 };
    var a16: EncDev = .{ .ctx = ctx, .gpa = gpa, .enc = &two, .act = .f16 };
    var t: TraceDev = .{ .arms = .{ &a32, &a16 }, .ctx = ctx, .gpa = gpa };
    defer t.steps.deinit(gpa);
    const out = try qwen3_enc_dev.encode(&t, &two, gpa, ids.items, .{}, null);
    defer gpa.free(out);

    errdefer for (t.steps.items, 0..) |stp, i| std.debug.print("  {d:>3} {s:<34} rel L2 {e:.3}  {?t} / {?t}\n", .{ i, stp.what, stp.rel, stp.routes[0], stp.routes[1] });
    // Each row is the drift since the input, not one op's: f16 adds ~3e-3 per GEMM and
    // ends two layers near 1e-2, where a broken op lands at 0.5 on its first output.
    for (t.steps.items) |stp| try std.testing.expect(stp.rel < 5e-2);
}

// Speculative decoding on the Vulkan stepper must be byte-identical to
// vanilla greedy: the grouped 4-input GEMVs and the batched flash-decoding
// attention reproduce the decode path's summation orders bitwise. Gated on
// the model + GPU marker; kept tiny (each token is a full 36-layer forward).
// The GUI session's turn shape on the real model: warm-up, prefill to the boundary,
// a zero-byte checkpoint, generate, roll back and regenerate (greedy must repeat),
// a reset and a replay, then a KV dtype rebuild that must match a stepper built at
// that dtype.
test "vulkan gui turn lifecycle: prefill, checkpoint rollback, reset, kv rebuild" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const engine = @import("../llm/engine.zig");
    const chat = @import("../llm/chat.zig");
    const tokenizer_mod = @import("tp_core").tokenizer;
    const te_path = build_options.model_root ++ "/text_encoders/qwen3VLInstruct4bHeretic_v10.safetensors";
    std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    std.Io.Dir.cwd().access(io, te_path, .{}) catch return error.SkipZigTest;
    var ctx = gpu.Context.init(gpa, io) catch return error.SkipZigTest;
    defer ctx.deinit();

    var st = try safetensors.SafeTensors.open(gpa, io, te_path);
    defer st.deinit();
    var lm = try qwen3.CausalLM.load(gpa, .{ .safetensors = &st });
    defer lm.deinit();
    var tok = try tokenizer_mod.Tokenizer.init(gpa);
    defer tok.deinit();

    const opts: engine.Options = .{ .max_new_tokens = 4, .sampling = .{ .temperature = 0 } };
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    try chat.appendUser(&tok, gpa, "Count: one two three one two", &ids);
    try chat.openAssistant(&tok, gpa, &ids);
    const prompt_len = ids.items.len;

    var model = try VulkanLM.init(gpa, ctx, &lm, .{ .initial = 256, .max = 256 });
    defer model.deinit();
    model.warmWeights();
    try std.testing.expectEqual(@as(usize, 0), model.cached());

    try model.ensureCapacity(prompt_len);
    try model.prefill(ids.items[0 .. prompt_len - 1]);
    const q = model.cached();
    try std.testing.expectEqual(prompt_len - 1, q);
    try std.testing.expectEqual(@as(usize, 0), model.checkpointBytes());
    var empty: [0]u8 = .{};
    try model.checkpoint(&empty);
    const n1 = try engine.generate(&model, &tok, io, gpa, &ids, opts, null);
    try std.testing.expect(n1 > 0);
    const take1 = try gpa.dupe(u32, ids.items[prompt_len..]);
    defer gpa.free(take1);

    try model.restoreCheckpoint(&.{}, q);
    try std.testing.expectEqual(q, model.cached());
    ids.shrinkRetainingCapacity(prompt_len);
    _ = try engine.generate(&model, &tok, io, gpa, &ids, opts, null);
    errdefer std.debug.print("take1 {any} vs regenerated {any}\n", .{ take1, ids.items[prompt_len..] });
    try std.testing.expectEqualSlices(u32, take1, ids.items[prompt_len..]);

    try model.resetCache();
    try std.testing.expectEqual(@as(usize, 0), model.cached());
    ids.shrinkRetainingCapacity(prompt_len);
    try model.prefill(ids.items[0 .. prompt_len - 1]);
    _ = try engine.generate(&model, &tok, io, gpa, &ids, opts, null);
    errdefer std.debug.print("take1 {any} vs after reset {any}\n", .{ take1, ids.items[prompt_len..] });
    try std.testing.expectEqualSlices(u32, take1, ids.items[prompt_len..]);

    // A rebuilt f16 cache against a stepper built with one.
    try model.reinitCache(.f16);
    try std.testing.expectEqual(kvmod.KvDtype.f16, model.kv_dtype);
    ids.shrinkRetainingCapacity(prompt_len);
    try model.prefill(ids.items[0 .. prompt_len - 1]);
    _ = try engine.generate(&model, &tok, io, gpa, &ids, opts, null);
    const rebuilt = try gpa.dupe(u32, ids.items[prompt_len..]);
    defer gpa.free(rebuilt);
    var fresh = try VulkanLM.init(gpa, ctx, &lm, .{ .initial = 256, .max = 256, .kv_dtype = .f16 });
    defer fresh.deinit();
    ids.shrinkRetainingCapacity(prompt_len);
    try fresh.prefill(ids.items[0 .. prompt_len - 1]);
    _ = try engine.generate(&fresh, &tok, io, gpa, &ids, opts, null);
    errdefer std.debug.print("rebuilt f16 {any} vs fresh f16 {any}\n", .{ rebuilt, ids.items[prompt_len..] });
    try std.testing.expectEqualSlices(u32, rebuilt, ids.items[prompt_len..]);
}

// A cache that starts at 16 rows and grows twice through a reply must decode exactly
// what a cache built at full size does: growth copies rows, it changes no arithmetic.
test "vulkan kv growth decodes what a full-size cache does" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const engine = @import("../llm/engine.zig");
    const chat = @import("../llm/chat.zig");
    const tokenizer_mod = @import("tp_core").tokenizer;
    const te_path = build_options.model_root ++ "/text_encoders/qwen3VLInstruct4bHeretic_v10.safetensors";
    std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    std.Io.Dir.cwd().access(io, te_path, .{}) catch return error.SkipZigTest;
    var ctx = gpu.Context.init(gpa, io) catch return error.SkipZigTest;
    defer ctx.deinit();

    var st = try safetensors.SafeTensors.open(gpa, io, te_path);
    defer st.deinit();
    var lm = try qwen3.CausalLM.load(gpa, .{ .safetensors = &st });
    defer lm.deinit();
    var tok = try tokenizer_mod.Tokenizer.init(gpa);
    defer tok.deinit();

    const opts: engine.Options = .{ .max_new_tokens = 40, .sampling = .{ .temperature = 0 } };
    var prompt: std.ArrayList(u32) = .empty;
    defer prompt.deinit(gpa);
    try chat.appendUser(&tok, gpa, "Count from one to twenty in words.", &prompt);
    try chat.openAssistant(&tok, gpa, &prompt);

    var outs: [2][]u32 = undefined;
    for (&outs, [2]usize{ 16, 256 }) |*out, initial| {
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(gpa);
        try ids.appendSlice(gpa, prompt.items);
        var model = try VulkanLM.init(gpa, ctx, &lm, .{ .initial = initial, .max = 256, .kv_dtype = .f16 });
        defer model.deinit();
        _ = try engine.generate(&model, &tok, io, gpa, &ids, opts, null);
        if (initial == 16) try std.testing.expect(model.cached() + model.remaining() > 32);
        out.* = try gpa.dupe(u32, ids.items[prompt.items.len..]);
    }
    defer for (outs) |o| gpa.free(o);
    errdefer std.debug.print("grown {any}\nfull  {any}\n", .{ outs[0], outs[1] });
    try std.testing.expect(outs[0].len > 20);
    try std.testing.expectEqualSlices(u32, outs[1], outs[0]);
}

// The CPU split on the real model: a tight budget places layers on the host, which
// run the CPU layer body; the host KV shadow stays in step through generate and
// rollback (a rollback and regenerate must repeat), its logits track an all-device
// stepper's (a host shadow in the wrong byte layout reads garbage), and promoting
// every layer back restores full residency.
fn vulkanSplitBody(kv_dtype: kvmod.KvDtype) !void {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const engine = @import("../llm/engine.zig");
    const chat = @import("../llm/chat.zig");
    const tokenizer_mod = @import("tp_core").tokenizer;
    const te_path = build_options.model_root ++ "/text_encoders/qwen3VLInstruct4bHeretic_v10.safetensors";
    std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    std.Io.Dir.cwd().access(io, te_path, .{}) catch return error.SkipZigTest;
    var ctx = gpu.Context.init(gpa, io) catch return error.SkipZigTest;
    defer ctx.deinit();

    var st = try safetensors.SafeTensors.open(gpa, io, te_path);
    defer st.deinit();
    var lm = try qwen3.CausalLM.load(gpa, .{ .safetensors = &st });
    defer lm.deinit();
    var tok = try tokenizer_mod.Tokenizer.init(gpa);
    defer tok.deinit();

    const opts: engine.Options = .{ .max_new_tokens = 3, .sampling = .{ .temperature = 0 } };
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    try chat.appendUser(&tok, gpa, "Count: one two three one two", &ids);
    try chat.openAssistant(&tok, gpa, &ids);
    const prompt_len = ids.items.len;

    // The reference: every layer on the device.
    const vocab = lm.cfg.vocab;
    const want = try gpa.alloc(f32, vocab);
    defer gpa.free(want);
    {
        var full = try VulkanLM.init(gpa, ctx, &lm, .{ .initial = 64, .max = 256, .kv_dtype = kv_dtype });
        defer full.deinit();
        try full.step(io, ids.items, want);
    }

    var model = try VulkanLM.init(gpa, ctx, &lm, .{ .initial = 64, .max = 256, .kv_dtype = kv_dtype });
    defer model.deinit();
    // ~3.6 GB of fp8 layer weights and a ~0.8 GB head: 4 GiB puts a few tail layers
    // on the host.
    try model.enableCpuSplit(.attn, 4 << 30, true);
    const sp = &model.split.?;
    errdefer std.debug.print("n_cpu={d}\n", .{sp.n_cpu});
    try std.testing.expect(sp.n_cpu > 0 and sp.n_cpu < model.cfg.n_layers);
    try std.testing.expectEqual(kv_dtype, sp.cache.kv_dtype);

    // Host layers need an Io before the first forward, and prefill takes none.
    try std.testing.expectError(error.SplitIoUnset, model.prefill(ids.items[0 .. prompt_len - 1]));
    try std.testing.expectEqual(@as(usize, 0), model.cached());
    model.io = io;

    {
        const got = try gpa.alloc(f32, vocab);
        defer gpa.free(got);
        try model.step(io, ids.items, got);
        var num: f64 = 0;
        var den: f64 = 0;
        for (want, got) |a, b| {
            num += (@as(f64, a) - b) * (@as(f64, a) - b);
            den += @as(f64, a) * a;
        }
        const rel = @sqrt(num / den);
        errdefer std.debug.print("split vs device logits rel L2 {e:.3} (kv {t})\n", .{ rel, kv_dtype });
        try std.testing.expect(rel < 5e-2);
        try std.testing.expectEqual(sample.argmax(want), sample.argmax(got));
        model.truncate(0);
    }

    try model.ensureCapacity(prompt_len);
    try model.prefill(ids.items[0 .. prompt_len - 1]);
    const q = model.cached();
    try std.testing.expectEqual(q, sp.cache.len);
    var empty: [0]u8 = .{};
    try model.checkpoint(&empty);
    const n1 = try engine.generate(&model, &tok, io, gpa, &ids, opts, null);
    try std.testing.expect(n1 > 0);
    try std.testing.expectEqual(model.cached(), sp.cache.len);
    const take1 = try gpa.dupe(u32, ids.items[prompt_len..]);
    defer gpa.free(take1);

    try model.restoreCheckpoint(&.{}, q);
    try std.testing.expectEqual(q, sp.cache.len);
    ids.shrinkRetainingCapacity(prompt_len);
    _ = try engine.generate(&model, &tok, io, gpa, &ids, opts, null);
    errdefer std.debug.print("take1 {any} vs regenerated {any}\n", .{ take1, ids.items[prompt_len..] });
    try std.testing.expectEqualSlices(u32, take1, ids.items[prompt_len..]);

    const promoted = try model.promoteLayers(std.math.maxInt(u64));
    try std.testing.expect(promoted > 0);
    try std.testing.expectEqual(@as(usize, 0), model.split.?.n_cpu);
    try model.restoreCheckpoint(&.{}, q);
    ids.shrinkRetainingCapacity(prompt_len);
    const n3 = try engine.generate(&model, &tok, io, gpa, &ids, opts, null);
    try std.testing.expect(n3 > 0);
    try std.testing.expectEqual(model.cached(), model.split.?.cache.len);
}

test "vulkan cpu split: host layers track the device, rollback stays in sync, promote restores" {
    try vulkanSplitBody(.f32);
}

test "vulkan cpu split with f16 kv" {
    try vulkanSplitBody(.f16);
}

test "vulkan cpu split with q8_0 kv" {
    try vulkanSplitBody(.q8_0);
}

// Layers moved to the host mid-context carry their rows, and moved back bring them
// back byte for byte: after a promote the forward is all on the device again, so its
// logits must equal, exactly, a stepper that never split and was fed the same chunks.
test "vulkan cpu split moves live rows to the host and back losslessly" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const te_path = build_options.model_root ++ "/text_encoders/qwen3VLInstruct4bHeretic_v10.safetensors";
    std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    std.Io.Dir.cwd().access(io, te_path, .{}) catch return error.SkipZigTest;
    var ctx = gpu.Context.init(gpa, io) catch return error.SkipZigTest;
    defer ctx.deinit();

    var st = try safetensors.SafeTensors.open(gpa, io, te_path);
    defer st.deinit();
    var lm = try qwen3.CausalLM.load(gpa, .{ .safetensors = &st });
    defer lm.deinit();

    var prompt: [40]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(1000 + (i * 37) % 50000);
    const k = 16;
    const n = prompt.len;
    const vocab = lm.cfg.vocab;
    const want = try gpa.alloc(f32, vocab);
    defer gpa.free(want);
    const got = try gpa.alloc(f32, vocab);
    defer gpa.free(got);

    for ([_]kvmod.KvDtype{ .f32, .f16, .q8_0 }) |dt| {
        {
            var ref = try VulkanLM.init(gpa, ctx, &lm, .{ .initial = 64, .max = 64, .kv_dtype = dt });
            defer ref.deinit();
            try ref.prefill(prompt[0..k]);
            try ref.prefill(prompt[k .. n - 1]);
            try ref.step(io, prompt[n - 1 ..], want);
        }
        var m = try VulkanLM.init(gpa, ctx, &lm, .{ .initial = 64, .max = 64, .kv_dtype = dt });
        defer m.deinit();
        m.io = io;
        try m.prefill(prompt[0..k]);
        try m.enableCpuSplit(.attn, 4 << 30, true);
        try std.testing.expect(m.split.?.n_cpu > 0);

        try m.prefill(prompt[k .. n - 1]);
        try m.step(io, prompt[n - 1 ..], got);
        var num: f64 = 0;
        var den: f64 = 0;
        for (want, got) |a, b| {
            num += (@as(f64, a) - b) * (@as(f64, a) - b);
            den += @as(f64, a) * a;
        }
        const rel = @sqrt(num / den);
        errdefer std.debug.print("kv {t}: hybrid vs device rel L2 {e:.3}\n", .{ dt, rel });
        try std.testing.expect(rel < 5e-2);

        m.truncate(k);
        _ = try m.promoteLayers(std.math.maxInt(u64));
        try std.testing.expectEqual(@as(usize, 0), m.split.?.n_cpu);
        try m.prefill(prompt[k .. n - 1]);
        try m.step(io, prompt[n - 1 ..], got);
        errdefer std.debug.print("kv {t}: logits after the promote differ from the unsplit run\n", .{dt});
        try std.testing.expectEqualSlices(f32, want, got);
    }
}

// The GUI's KV dtype toggle with the split armed: host layers keep no device KV, the
// shadow is rebuilt at the new dtype, the placement survives, and a re-prefill runs.
test "vulkan cpu split survives a kv dtype toggle, and a residency reset re-arms it" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const te_path = build_options.model_root ++ "/text_encoders/qwen3VLInstruct4bHeretic_v10.safetensors";
    std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    std.Io.Dir.cwd().access(io, te_path, .{}) catch return error.SkipZigTest;
    var ctx = gpu.Context.init(gpa, io) catch return error.SkipZigTest;
    defer ctx.deinit();

    var st = try safetensors.SafeTensors.open(gpa, io, te_path);
    defer st.deinit();
    var lm = try qwen3.CausalLM.load(gpa, .{ .safetensors = &st });
    defer lm.deinit();
    var model = try VulkanLM.init(gpa, ctx, &lm, .{ .initial = 64, .max = 256 });
    defer model.deinit();
    try model.enableCpuSplit(.attn, 4 << 30, true);
    try std.testing.expect(model.split.?.n_cpu > 0);
    model.io = io;

    var prompt: [24]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(1000 + (i * 37) % 50000);
    for ([_]kvmod.KvDtype{ .f16, .q8_0, .f32 }) |dt| {
        const n_cpu_before = model.split.?.n_cpu;
        try model.reinitCache(dt);
        const sp = &model.split.?;
        try std.testing.expectEqual(dt, model.kv_dtype);
        try std.testing.expectEqual(dt, sp.cache.kv_dtype);
        try std.testing.expectEqual(@as(usize, 0), sp.cache.len);
        try std.testing.expectEqual(n_cpu_before, sp.n_cpu);
        for (sp.on_gpu, 0..) |on, l| try std.testing.expectEqual(on, model.kv.k[l].buf != .null_handle);
        try model.prefill(prompt[0 .. prompt.len - 1]);
        try std.testing.expectEqual(prompt.len - 1, sp.cache.len);
        const next = try model.stepArgmax(io, prompt[prompt.len - 1 ..]);
        try std.testing.expect(next < lm.cfg.vocab);
    }

    // A grown cache, then a reset: back to the initial capacity, empty, split re-armed.
    try model.ensureCapacity(200);
    try std.testing.expect(model.cached() + model.remaining() >= 200);
    try model.resetResidency(4 << 30);
    try std.testing.expectEqual(@as(usize, 0), model.cached());
    try std.testing.expectEqual(@as(usize, 64), model.remaining());
    try std.testing.expect(model.split.?.n_cpu > 0);
    try model.prefill(prompt[0 .. prompt.len - 1]);
    _ = try model.stepArgmax(io, prompt[prompt.len - 1 ..]);
}

// Weight noise on a block-quant model, through both routes it takes: the prefill's
// dequant GEMM and the decode GEMV. Off is the unperturbed model exactly; on moves both;
// the same seed and curve from the same draw counter reproduce both bit for bit; and
// turning it off again restores the unperturbed logits.
test "vulkan weight noise perturbs prefill and decode, reproducibly, and off is exact" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const gguf_mod = @import("tp_core").gguf;
    const path = build_options.model_root ++ "/text_encoders/Qwen3-4B-Q4_K_M.gguf";
    std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    std.Io.Dir.cwd().access(io, path, .{}) catch return error.SkipZigTest;
    var ctx = gpu.Context.init(gpa, io) catch return error.SkipZigTest;
    defer ctx.deinit();

    var g = try gguf_mod.Gguf.open(gpa, io, path);
    defer g.deinit();
    var lm = try qwen3.CausalLM.load(gpa, .{ .gguf = &g });
    defer lm.deinit();
    var model = try VulkanLM.init(gpa, ctx, &lm, .{ .initial = 512, .max = 512 });
    defer model.deinit();

    var prompt: [300]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(1000 + (i * 37) % 50000);
    const vocab = lm.cfg.vocab;
    var bufs: [6][]f32 = undefined;
    for (&bufs) |*b| b.* = try gpa.alloc(f32, vocab);
    defer for (bufs) |b| gpa.free(b);
    const off_pre, const off_dec, const on_pre, const on_dec, const re_pre, const re_dec = bufs;

    const run = struct {
        fn f(m: *VulkanLM, io_: std.Io, p: []const u32, pre: []f32, dec: []f32) !void {
            m.truncate(0);
            m.be.weight_noise.seq = 0;
            try m.step(io_, p, pre);
            try m.step(io_, &.{p[0]}, dec);
        }
    }.f;
    try run(&model, io, &prompt, off_pre, off_dec);

    const noise = &ctx.weight_noise;
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
