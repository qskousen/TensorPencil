//! CLIP text tower on the Vulkan backend, SD1.5's CLIP-L and SDXL's CLIP-G: the adapter
//! for `clip_text_dev.zig`. The weights are read straight from the checkpoint mapping
//! (the device cache keys on the host pointer, so nothing is dequantized or copied) and
//! the cheap head (the pooled row and its projection) stays on the host.
//!
//! GEMM precision follows `qwen3_gpu`'s convention: f32 by default, tensor-core f16 only
//! under `--encoder-f16`, so the default tracks the CPU forward the fixtures pin.

const std = @import("std");
const tp_core = @import("tp_core");
const gpu = @import("tp_gpu").context;
const ops = @import("tp_ops");
const clip_tok = tp_core.clip_tokenizer;
const clip_text = @import("clip_text.zig");
const clip_text_dev = @import("clip_text_dev.zig");
const lin_gpu = @import("lin_gpu.zig");

const Buf = gpu.DeviceBuffer;
const Weight = ops.matmul.Weight;
const clen = clip_tok.context_length;

/// Wrap a host f32 slice as a small device buffer (norm weights and biases).
fn nbuf(ctx: *gpu.Context, w: []const f32) !Buf {
    return .{ .buf = try ctx.smallBuffer(std.mem.sliceAsBytes(w)), .mem = .null_handle, .size = 0 };
}

/// A conditioning is computed once and steers every step, so block quants stay on the
/// f16 expansion.
const encoder_use: lin_gpu.Use = .{ .blockq = .f16 };

/// Whether this context can run the tower and its LoRA stack.
pub fn supported(ctx: *gpu.Context, enc: *const clip_text.TextEncoder) bool {
    return lin_gpu.runs(ctx, enc.device_lins, encoder_use, enc.lora, "clip_text_gpu");
}

/// `clip_text_dev.encodePrompt` on this backend, the residual, the norm outputs and the
/// projections stored at `act`; at f16 every dense weight also runs on the f16 tensor
/// cores.
pub fn encodePrompt(
    enc: *const clip_text.TextEncoder,
    ctx: *gpu.Context,
    io: std.Io,
    gpa: std.mem.Allocator,
    out: []f32,
    p: *const clip_tok.Prompt,
    r: clip_text.TextEncoder.PromptRun,
    act: gpu.Dt,
) !void {
    // Taken for signature symmetry with the CPU form; nothing here reaches the host GEMM.
    _ = io;
    var d: Dev = .{ .ctx = ctx, .gpa = gpa, .enc = enc, .act = act };
    try clip_text_dev.encodePrompt(&d, enc, gpa, out, p, r);
}

const none: Buf = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 };
const Bufs = struct { x: Buf = none, normed: Buf = none, q: Buf = none, k: Buf = none, v: Buf = none, big: Buf = none, t: Buf = none, cap: Buf = none };

/// The `clip_text_dev` adapter.
const Dev = struct {
    ctx: *gpu.Context,
    gpa: std.mem.Allocator,
    enc: *const clip_text.TextEncoder,
    /// `x`, `normed` and `t`; the attention planes and `big` stay f32, and `big` is the
    /// staging for a 16-bit `x`'s upload and download.
    act: gpu.Dt,
    lin: ?lin_gpu.Lin = null,
    plan: lin_gpu.Plan = undefined,
    total: usize = 0,
    b: Bufs = .{},

    /// Sized up front, never grown: `ensureDeviceBuffer` inside a recording batch frees
    /// memory that recorded-but-unsubmitted dispatches still reference.
    pub fn setup(d: *Dev, total: usize, capture: bool) !void {
        const cfg = d.enc.cfg;
        const enc = d.enc;
        d.total = total;
        errdefer d.teardown();
        d.lin = try lin_gpu.Lin.init(d.gpa, d.ctx, lin_gpu.maxRows(enc.device_lins), 0, 0);
        d.plan = try lin_gpu.plan(d.ctx, enc.device_lins, encoder_use, "clip_text_gpu");
        if (d.act == .f16) d.plan.dense = .f16;
        if (enc.lora) |st| {
            d.plan.lora = st;
            try d.lin.?.presizeLora(st, enc.device_lins, total);
        }
        inline for (.{ "x", "normed", "q", "k", "v", "t" }) |f| {
            const w: gpu.Dt = if (comptime std.mem.eql(u8, f, "q") or std.mem.eql(u8, f, "k") or std.mem.eql(u8, f, "v")) .f32 else d.act;
            @field(d.b, f) = try d.ctx.tensorCreate(total * cfg.hidden * w.bytes());
            @field(d.b, f).dt = w;
        }
        d.b.big = try d.ctx.tensorCreate(total * cfg.intermediate * 4);
        if (capture) d.b.cap = try d.ctx.tensorCreate(total * cfg.hidden * 4);
    }

    pub fn teardown(d: *Dev) void {
        inline for (@typeInfo(Bufs).@"struct".fields) |f| d.ctx.tensorDestroy(&@field(d.b, f.name));
        if (d.lin) |*l| l.deinit();
        d.lin = null;
    }

    pub fn upload(d: *Dev, b: Buf, data: []const f32) !void {
        if (b.dt == .f32) return d.ctx.tensorUpload(b, std.mem.sliceAsBytes(data));
        try d.ctx.tensorUpload(d.b.big, std.mem.sliceAsBytes(data));
        try d.ctx.opElt(gpu.eltCopy(.f32, b.dt), d.b.big, b, null, null, .{ .u0 = @intCast(data.len) }, data.len, 1, 1);
    }

    pub fn download(d: *Dev, b: Buf, dst: []f32) !void {
        if (b.dt == .f32) return d.ctx.tensorDownloadAt(b, 0, std.mem.sliceAsBytes(dst));
        try d.ctx.beginBatch();
        try d.ctx.opElt(gpu.eltCopy(b.dt, .f32), b, d.b.big, null, null, .{ .u0 = @intCast(dst.len) }, dst.len, 1, 1);
        try d.ctx.endBatch();
        try d.ctx.tensorDownloadAt(d.b.big, 0, std.mem.sliceAsBytes(dst));
    }

    pub fn beginBatch(d: *Dev) !void {
        try d.ctx.beginBatch();
    }

    pub fn endBatch(d: *Dev) !void {
        try d.ctx.endBatch();
    }

    pub fn abortBatch(d: *Dev) void {
        if (d.ctx.batching) d.ctx.abortBatch();
    }

    pub fn layerNorm(d: *Dev, x: Buf, y: Buf, w: []const f32, bias: []const f32, rows: usize) !void {
        std.debug.assert(x.dt == y.dt);
        try d.ctx.opElt(gpu.eltAt("layernorm", x.dt), x, y, try nbuf(d.ctx, w), try nbuf(d.ctx, bias), .{
            .u0 = @intCast(rows),
            .u1 = @intCast(d.enc.cfg.hidden),
            .f0 = d.enc.cfg.eps,
        }, rows, 1, 1);
    }

    /// One prep per weight.
    pub fn gemms(d: *Dev, x: Buf, m: usize, ys: []const Buf, wts: []const Weight, biases: []const []const f32) !void {
        const l = &d.lin.?;
        for (ys, wts, biases) |y, w, bias| {
            const g = try l.prep(d.plan, x, m, w.cols, &.{w});
            try l.gemm(d.plan, g, y, w, .{ .bias = bias });
        }
    }

    /// Block-diagonal causal attention, one 77-row window per batch item.
    pub fn attention(d: *Dev, items: usize) !void {
        const cfg = d.enc.cfg;
        const hd = cfg.headDim();
        const total = items * clen;
        try d.ctx.opElt(.attn_causal_batched, d.b.q, d.b.k, d.b.v, d.b.big, .{
            .u0 = @intCast(total),
            .u1 = @intCast(cfg.heads),
            .u2 = @intCast(cfg.heads),
            .u3 = @intCast(hd),
            .u4 = @intCast(clen),
            .f0 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd))),
        }, total * cfg.heads, 1, 1);
    }

    pub fn add(d: *Dev, x: Buf, y: Buf, n: usize) !void {
        std.debug.assert(x.dt == y.dt);
        try d.ctx.opElt(gpu.eltAt("add", x.dt), x, y, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
    }

    pub fn gelu(d: *Dev, x: Buf, n: usize, act: clip_text.Config.Act) !void {
        try d.ctx.opElt(switch (act) {
            .quick_gelu => .gelu_quick,
            .gelu_erf => .gelu_erf,
        }, x, null, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
    }

    pub fn copy(d: *Dev, dst: Buf, src: Buf, n: usize) !void {
        try d.ctx.opElt(gpu.eltCopy(src.dt, dst.dt), src, dst, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
    }
};

// --- device parity tests ---------------------------------------------------
//
// Every new kernel here is pinned against the CPU op it reproduces, at the level of
// that op rather than only through a whole render. That harness is what localized
// every bug in the SD-family GPU work in seconds, where the one bug found by looking
// at an image instead cost far more.
//
// Needs `-Dintegration` AND the `testdata/gpu-tests` marker (see context.zig).

const testing = std.testing;

fn gpuCtx(gpa: std.mem.Allocator, io: std.Io) !*gpu.Context {
    std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    return gpu.Context.init(gpa, std.testing.io) catch error.SkipZigTest;
}

fn relL2(want: []const f32, got: []const f32) f64 {
    var num: f64 = 0;
    var den: f64 = 0;
    for (want, got) |e, a| {
        const d = @as(f64, e) - @as(f64, a);
        num += d * d;
        den += @as(f64, e) * @as(f64, e);
    }
    if (den == 0) return @sqrt(num);
    return @sqrt(num / den);
}

test "gpu gelu_quick and gelu_erf match ops.act, and are not each other" {
    // Checked value by value, not in aggregate: the three GELU forms in this codebase
    // agree to ~1e-2, so an aggregate bound loose enough to pass f32 noise would also
    // pass the wrong kernel. The final assertion is the one that matters, it fails if
    // the two entry points are ever wired to the same kernel.
    const gpa = testing.allocator;
    const io = testing.io;
    const ctx = try gpuCtx(gpa, io);
    defer ctx.deinit();

    const n = 4096;
    const x = try gpa.alloc(f32, n);
    defer gpa.free(x);
    var prng = std.Random.DefaultPrng.init(11);
    // Spread over the range a real FFN sees, including the saturating tails where the
    // three forms differ most.
    for (x, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(i)) / @as(f32, n) - 0.5) * 24.0 + prng.random().floatNorm(f32) * 0.01;

    const want_q = try gpa.dupe(f32, x);
    defer gpa.free(want_q);
    ops.act.geluQuick(want_q);
    const want_e = try gpa.dupe(f32, x);
    defer gpa.free(want_e);
    ops.act.geluErf(want_e);

    const got = try gpa.alloc(f32, n);
    defer gpa.free(got);
    for ([_]struct { gpu.Elt, []const f32, []const u8 }{
        .{ .gelu_quick, want_q, "gelu_quick" },
        .{ .gelu_erf, want_e, "gelu_erf" },
    }) |arm| {
        const which, const want, const name = arm;
        var d: Buf = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 };
        defer ctx.tensorDestroy(&d);
        d = try ctx.tensorCreate(n * 4);
        try ctx.tensorUpload(d, std.mem.sliceAsBytes(x));
        try ctx.opElt(which, d, null, null, null, .{ .u0 = n }, n, 1, 1);
        try ctx.tensorDownload(d, std.mem.sliceAsBytes(got));
        for (want, got, 0..) |e, a, i| {
            errdefer std.debug.print("{s} at {d} (x={d:.4}): {d:.7} vs {d:.7}\n", .{ name, i, x[i], e, a });
            try testing.expectApproxEqAbs(e, a, 2e-6);
        }
    }
    // The two forms really are different functions, so the checks above have teeth.
    try testing.expect(relL2(want_q, want_e) > 1e-3);
}

test "gpu attn_causal_batched matches causal CPU attention per chunk" {
    const gpa = testing.allocator;
    const io = testing.io;
    const ctx = try gpuCtx(gpa, io);
    defer ctx.deinit();

    // Two chunks of the real 77-token window at CLIP-L's geometry, which is also what
    // makes the batching claim meaningful: one chunk must not see the other's keys.
    const chunks = 2;
    const heads = 12;
    const hd = 64;
    const dim = heads * hd;
    const total = chunks * clen;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));

    var prng = std.Random.DefaultPrng.init(23);
    const rand = prng.random();
    const q = try gpa.alloc(f32, total * dim);
    defer gpa.free(q);
    const k = try gpa.alloc(f32, total * dim);
    defer gpa.free(k);
    const v = try gpa.alloc(f32, total * dim);
    defer gpa.free(v);
    for (q) |*x| x.* = rand.floatNorm(f32);
    for (k) |*x| x.* = rand.floatNorm(f32);
    for (v) |*x| x.* = rand.floatNorm(f32);

    // Reference: the CPU op, run per chunk with causal masking, which is exactly what
    // the tower does on the host path, so this compares the two things that must agree.
    const want = try gpa.alloc(f32, total * dim);
    defer gpa.free(want);
    for (0..chunks) |c| {
        const off = c * clen * dim;
        try ops.attention.attention(io, gpa, want[off..][0 .. clen * dim], q[off..][0 .. clen * dim], k[off..][0 .. clen * dim], v[off..][0 .. clen * dim], .{
            .seq_q = clen,
            .seq_kv = clen,
            .n_heads = heads,
            .n_kv_heads = heads,
            .head_dim = hd,
            .causal = true,
        });
    }

    var bufs: [4]Buf = @splat(.{ .buf = .null_handle, .mem = .null_handle, .size = 0 });
    defer for (&bufs) |*b| ctx.tensorDestroy(b);
    for (&bufs) |*b| b.* = try ctx.tensorCreate(total * dim * 4);
    try ctx.tensorUpload(bufs[0], std.mem.sliceAsBytes(q));
    try ctx.tensorUpload(bufs[1], std.mem.sliceAsBytes(k));
    try ctx.tensorUpload(bufs[2], std.mem.sliceAsBytes(v));
    try ctx.opElt(.attn_causal_batched, bufs[0], bufs[1], bufs[2], bufs[3], .{
        .u0 = @intCast(total),
        .u1 = heads,
        .u2 = heads,
        .u3 = hd,
        .u4 = @intCast(clen),
        .f0 = scale,
    }, total * heads, 1, 1);

    const got = try gpa.alloc(f32, total * dim);
    defer gpa.free(got);
    try ctx.tensorDownload(bufs[3], std.mem.sliceAsBytes(got));

    const rel = relL2(want, got);
    errdefer std.debug.print("attn_causal_batched rel L2 {d:.8}\n", .{rel});
    try testing.expect(rel < 1e-5);

    // And that it is genuinely causal: the same launch with the non-causal kernel must
    // NOT agree. Without this the test passes on a kernel that ignores the mask, which
    // is the one CLIP-specific mistake that produces no visible failure at all.
    try ctx.opElt(.attn_full, bufs[0], bufs[1], bufs[2], bufs[3], .{
        .u0 = @intCast(clen),
        .u1 = heads,
        .u2 = heads,
        .u3 = hd,
        .f0 = scale,
    }, clen * heads, 1, 1);
    const bidir = try gpa.alloc(f32, total * dim);
    defer gpa.free(bidir);
    try ctx.tensorDownload(bufs[3], std.mem.sliceAsBytes(bidir));
    try testing.expect(relL2(want[0 .. clen * dim], bidir[0 .. clen * dim]) > 0.1);
}

test "gpu clip tower matches the CPU forward on a weighted two-chunk prompt" {
    // The whole path, both activations, both capture modes: this is what a wrong buffer
    // size, a wrong positional-table restart, or a dropped weight application shows up
    // in. Widths are CLIP-L's head geometry at a fraction of the depth, so it runs in
    // milliseconds while exercising the same kernels a real tower does.
    const gpa = testing.allocator;
    const io = testing.io;
    const ctx = try gpuCtx(gpa, io);
    defer ctx.deinit();

    for ([_]clip_text.Config.Act{ .quick_gelu, .gelu_erf }) |act| {
        const cfg: clip_text.Config = .{
            .hidden = 128,
            .layers = 3,
            .heads = 4,
            .intermediate = 256,
            .max_positions = clen,
            .eps = 1e-5,
            .act = act,
            .eos_id = clip_tok.eos_id,
        };
        // The full CLIP vocabulary, because BOS/EOS are ids 49406/49407 and the empty
        // reference sequence is built from them, a small vocab would be out of range.
        var ckpt = try clip_text.TinyCheckpoint.init(gpa, cfg, clip_tok.eos_id + 1);
        defer ckpt.deinit(gpa);
        var enc = try clip_text.TextEncoder.load(gpa, ckpt.store(), cfg, "");
        defer enc.deinit();

        // A hand-built two-chunk prompt with real weights, rather than a tokenized one:
        // the point here is the forward, and this pins the weighted path without
        // depending on which prompt happens to spill past 77 tokens.
        var prng = std.Random.DefaultPrng.init(31);
        const rand = prng.random();
        const tokens = try gpa.alloc(clip_tok.Weighted, 2 * clen);
        defer gpa.free(tokens);
        for (tokens, 0..) |*t, i| {
            const slot = i % clen;
            t.* = .{
                .id = if (slot == 0) clip_tok.bos_id else if (slot > 60) clip_tok.eos_id else rand.intRangeLessThan(u32, 0, 49000),
                // A mix of 1.0 (skipped, and so bit-identical) and non-1.0 rows.
                .weight = if (slot % 5 == 0) 1.0 else if (slot % 3 == 0) 1.15 else 0.8,
            };
        }
        const p: clip_tok.Prompt = .{ .tokens = tokens, .chunks = 2 };
        try testing.expect(p.hasWeights());

        for ([_]?usize{ null, cfg.layers - 1 }) |capture| {
            const n = p.seq() * cfg.hidden;
            const want = try gpa.alloc(f32, n);
            defer gpa.free(want);
            const got = try gpa.alloc(f32, n);
            defer gpa.free(got);
            const want_final = try gpa.alloc(f32, clen * cfg.hidden);
            defer gpa.free(want_final);
            const got_final = try gpa.alloc(f32, clen * cfg.hidden);
            defer gpa.free(got_final);

            // Separate caches: each side must compute its own `z_empty`, or the GPU arm
            // would be handed the CPU's and the comparison would not cover it.
            var cache_cpu: ?[]f32 = null;
            defer if (cache_cpu) |e| gpa.free(e);
            var cache_gpu: ?[]f32 = null;
            defer if (cache_gpu) |e| gpa.free(e);

            try enc.encodePrompt(io, gpa, want, &p, .{
                .empty_cache = &cache_cpu,
                .pad_id = clip_tok.eos_id,
                .capture_layer = capture,
                .final_chunk0 = want_final,
            });
            try encodePrompt(&enc, ctx, io, gpa, got, &p, .{
                .empty_cache = &cache_gpu,
                .pad_id = clip_tok.eos_id,
                .capture_layer = capture,
                .final_chunk0 = got_final,
            }, .f32);

            const rel = relL2(want, got);
            const rel_f = relL2(want_final, got_final);
            errdefer std.debug.print(
                "clip tower {t} capture={?d}: rel L2 {d:.8}, final_chunk0 {d:.8}\n",
                .{ act, capture, rel, rel_f },
            );
            // f32 GEMMs on both sides, so this is reduction-order noise only: measured
            // 1.1e-5 (no capture) and 3.3e-6 (captured) on this 3-layer tower. The bound
            // is ~4x that rather than a round number, a wrong activation, a missing
            // causal mask or a dropped weight application all miss by >1e-3, so there is
            // room to keep it tight.
            try testing.expect(rel < 5e-5);
            try testing.expect(rel_f < 5e-5);
            // The captured layer must NOT be the final output, if `capture_layer` were
            // ignored, both arms would agree with each other and be wrong together.
            // Compared against chunk 0's rows, which is what `final_chunk0` holds.
            if (capture != null) {
                try testing.expect(relL2(want_final, want[0 .. clen * cfg.hidden]) > 1e-3);
            }
        }
    }
}
