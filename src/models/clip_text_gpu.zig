//! CLIP text tower on the Vulkan backend — SD1.5's CLIP-L and SDXL's CLIP-G.
//!
//! A thin device forward over the CPU model's own weights, in the shape
//! `embed_siglip_gpu` established: every op is an existing `Context` entry point, the
//! weights are read straight from the checkpoint mapping (`opMatmul` caches a device
//! buffer keyed on the host pointer, so nothing is dequantized or copied), and the
//! cheap head — the pooled row and its projection — stays on the host.
//!
//! Three things make this not just SigLIP's tower again:
//!
//! 1. ⚠️ **The attention is CAUSAL** (`attn_causal_batched`, added for this). Every
//!    other encoder tower here reads bidirectionally. A non-causal CLIP encodes every
//!    prompt and renders every image — it is simply a different model — so there is no
//!    failure to observe if this is wrong.
//! 2. ⚠️ **The activation differs BETWEEN THE TWO TOWERS**: CLIP-L is quick-GELU
//!    (`x·σ(1.702x)`), CLIP-G is erf-GELU. `cfg.act` carries which and the kernels are
//!    separate, because the three GELU forms in this codebase agree to ~1e-2 — close
//!    enough to look right and far enough to shift style.
//! 3. **The prompt is many chunks**, so the batch axis is the chunk: a 77-row window
//!    gives 77·heads threads, which does not fill a GPU, while a long prompt has two or
//!    three windows that are independent by construction. The empty-prompt reference
//!    `z_empty` rides along as one more batch item — which is also exactly how ComfyUI
//!    computes it.
//!
//! GEMM precision follows `qwen3_gpu`'s convention: f32 by default, tensor-core f16
//! only when the caller opts in (`--encoder-f16`). The default therefore tracks the CPU
//! forward the fixtures pin, and the fast path stays a deliberate choice.
//!
//! Validated against the CPU forward by the device parity tests at the bottom
//! (`-Dintegration` plus the `testdata/gpu-tests` marker).

const std = @import("std");
const tp_core = @import("tp_core");
const gpu = @import("tp_gpu").context;
const ops = @import("tp_ops");
const safetensors = tp_core.safetensors;
const clip_tok = tp_core.clip_tokenizer;
const clip_text = @import("clip_text.zig");

const Buf = gpu.DeviceBuffer;
const Weight = ops.matmul.Weight;
const clen = clip_tok.context_length;

/// Wrap a host f32 slice as a small device buffer (norm weights and biases).
fn nbuf(ctx: *gpu.Context, w: []const f32) !Buf {
    return .{ .buf = try ctx.smallBuffer(std.mem.sliceAsBytes(w)), .mem = .null_handle, .size = 0 };
}

/// A GEMM against a checkpoint weight, routed by the dtype it was stored in — the
/// same routing `sd_unet_gpu.gemm` does, and for the same reason: a CLIP tower can
/// arrive as f32, f16, bf16, or (from ggufy) block-quantized.
///
/// Every CLIP linear carries a bias (unlike the SD UNet's attention projections), so
/// there is no zero-vector fallback here and no bias-cache aliasing hazard to guard.
fn gemm(
    ctx: *gpu.Context,
    y: Buf,
    y_off_elems: usize,
    x: Buf,
    m: usize,
    wt: Weight,
    bias: []const f32,
    use_f16: bool,
) !void {
    const rows = wt.rows;
    const cols = wt.cols;
    const coop = use_f16 and ctx.pipe_coop_f16w != .null_handle;
    switch (wt.dtype) {
        .f32 => {
            if (coop and std.mem.isAligned(@intFromPtr(wt.bytes.ptr), @alignOf(f32))) {
                const wf: []const f32 = @as([*]const f32, @ptrCast(@alignCast(wt.bytes.ptr)))[0 .. wt.bytes.len / 4];
                return ctx.opMatmulCoopF16W(y, y_off_elems, x, m, wf, rows, cols, bias);
            }
            return ctx.opMatmul(y, y_off_elems * 4, x, 0, m, wt.bytes, false, rows, cols, 1.0, bias);
        },
        // A 2-byte weight has no f32 GEMM entry point, so these take the coop path
        // regardless of `use_f16` — the storage dtype already decided the precision.
        .f16 => return ctx.opMatmulCoopF16Wh(y, y_off_elems, x, m, wt.bytes, rows, cols, bias),
        .bf16 => {
            if (ctx.pipe_coop_bf16w != .null_handle) {
                return ctx.opMatmulCoopBf16(y, y_off_elems, x, m, wt.bytes, rows, cols, bias);
            }
            return ctx.opMatmulCoopF16Wb(y, y_off_elems, x, m, wt.bytes, rows, cols, bias);
        },
        .q8_0, .q4_k, .q5_k, .q6_k, .iq4_nl => {
            if (!ctx.hasQuantPrefillGemm()) return error.UnsupportedDType;
            return ctx.opMatmulCoopQuant(wt.dtype, y, y_off_elems, x, m, wt.bytes, rows, cols, 1.0, bias, false);
        },
        else => return error.UnsupportedDType,
    }
}

/// `clip_text.TextEncoder.encodePrompt` on the device: fill `out` (`[p.seq()][hidden]`)
/// with this tower's conditioning for every chunk of `p`, weights applied.
///
/// Identical contract to the CPU form, including `r.final_chunk0` and the `r.empty_cache`
/// slot — so the caller cannot tell which ran except by speed.
pub fn encodePrompt(
    enc: *const clip_text.TextEncoder,
    ctx: *gpu.Context,
    io: std.Io,
    gpa: std.mem.Allocator,
    out: []f32,
    p: *const clip_tok.Prompt,
    r: clip_text.TextEncoder.PromptRun,
    use_f16: bool,
) !void {
    // Taken for signature symmetry with the CPU form so the two are interchangeable at
    // the call site; nothing on this path reaches for the host GEMM.
    _ = io;
    const cfg = enc.cfg;
    const h = cfg.hidden;
    const inter = cfg.intermediate;
    const heads = cfg.heads;
    const hd = cfg.headDim();
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
    std.debug.assert(out.len == p.seq() * h);

    // The empty reference rides along as one more batch item, but only when it is both
    // needed and not already cached — so a second render with the same tower pays
    // nothing, and an unweighted prompt never computes it at all.
    const need_empty = p.hasWeights() and r.mode.needsEmpty() and r.empty_cache.* == null;
    const items = p.chunks + @as(usize, if (need_empty) 1 else 0);
    const total = items * clen;

    // --- host: ids, then token + learned positional embeddings ---------------
    const ids = try gpa.alloc(u32, total);
    defer gpa.free(ids);
    for (0..p.chunks) |c| p.idsInto(ids[c * clen ..][0..clen], c);
    if (need_empty) clip_tok.emptyIds(ids[p.chunks * clen ..][0..clen], r.pad_id);

    const x_host = try gpa.alloc(f32, total * h);
    defer gpa.free(x_host);
    const row_bytes = enc.tok_embed.info.dtype.storageBytes(h);
    for (ids, 0..) |id, t| {
        if (id >= enc.vocab) return error.TokenOutOfRange;
        const dst = x_host[t * h ..][0..h];
        try safetensors.convertToF32(enc.tok_embed.info.dtype, enc.tok_embed.bytes[id * row_bytes ..][0..row_bytes], dst);
        // The positional table restarts per chunk: each window is its own sequence.
        for (dst, enc.pos_embed[(t % clen) * h ..][0..h]) |*xi, pe| xi.* += pe;
    }

    // --- device ------------------------------------------------------------
    // Sized up front and asserted, never grown: `ensureDeviceBuffer` inside a
    // recording batch discards the activation it holds and frees memory that
    // recorded-but-unsubmitted dispatches still reference (see sd_vae_gpu).
    var bufs: [7]Buf = @splat(.{ .buf = .null_handle, .mem = .null_handle, .size = 0 });
    defer for (&bufs) |*b| ctx.tensorDestroy(b);
    const sizes = [bufs.len]usize{ total * h, total * h, total * h, total * h, total * h, total * inter, total * h };
    for (&bufs, sizes) |*b, s| b.* = try ctx.tensorCreate(s * 4);
    const x_d = bufs[0];
    const normed_d = bufs[1];
    const q_d = bufs[2];
    const k_d = bufs[3];
    const v_d = bufs[4];
    const big_d = bufs[5];
    const t_d = bufs[6];
    // The captured layer has to survive to the end of the batch, so it gets its own
    // buffer rather than a mid-batch download (which would force a second submit).
    var cap_d: Buf = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 };
    defer ctx.tensorDestroy(&cap_d);
    if (r.capture_layer != null) cap_d = try ctx.tensorCreate(total * h * 4);

    try ctx.tensorUpload(x_d, std.mem.sliceAsBytes(x_host));
    try ctx.beginBatch();
    errdefer if (ctx.batching) ctx.abortBatch();

    if (r.capture_layer) |layer| if (layer == 0) {
        try ctx.opElt(.copy, x_d, cap_d, null, null, .{ .u0 = @intCast(total * h) }, total * h, 1, 1);
    };

    for (enc.layers, 0..) |*l, li| {
        // --- attention: x += o(causal_attn(ln1(x))) ---
        try ctx.opElt(.layernorm, x_d, normed_d, try nbuf(ctx, l.ln1_w), try nbuf(ctx, l.ln1_b), .{
            .u0 = @intCast(total), .u1 = @intCast(h), .f0 = cfg.eps,
        }, total, 1, 1);
        try gemm(ctx, q_d, 0, normed_d, total, l.q, l.q_b, use_f16);
        try gemm(ctx, k_d, 0, normed_d, total, l.k, l.k_b, use_f16);
        try gemm(ctx, v_d, 0, normed_d, total, l.v, l.v_b, use_f16);
        try ctx.opElt(.attn_causal_batched, q_d, k_d, v_d, big_d, .{
            .u0 = @intCast(total),
            .u1 = @intCast(heads),
            .u2 = @intCast(heads),
            .u3 = @intCast(hd),
            .u4 = @intCast(clen),
            .f0 = scale,
        }, total * heads, 1, 1);
        try gemm(ctx, t_d, 0, big_d, total, l.o, l.o_b, use_f16);
        try ctx.opElt(.add, x_d, t_d, null, null, .{ .u0 = @intCast(total * h) }, total * h, 1, 1);

        // --- mlp: x += fc2(act(fc1(ln2(x)))) ---
        try ctx.opElt(.layernorm, x_d, normed_d, try nbuf(ctx, l.ln2_w), try nbuf(ctx, l.ln2_b), .{
            .u0 = @intCast(total), .u1 = @intCast(h), .f0 = cfg.eps,
        }, total, 1, 1);
        try gemm(ctx, big_d, 0, normed_d, total, l.fc1, l.fc1_b, use_f16);
        try ctx.opElt(switch (cfg.act) {
            .quick_gelu => .gelu_quick,
            .gelu_erf => .gelu_erf,
        }, big_d, null, null, null, .{ .u0 = @intCast(total * inter) }, total * inter, 1, 1);
        try gemm(ctx, t_d, 0, big_d, total, l.fc2, l.fc2_b, use_f16);
        try ctx.opElt(.add, x_d, t_d, null, null, .{ .u0 = @intCast(total * h) }, total * h, 1, 1);

        if (r.capture_layer) |layer| if (layer == li + 1) {
            try ctx.opElt(.copy, x_d, cap_d, null, null, .{ .u0 = @intCast(total * h) }, total * h, 1, 1);
        };
    }

    // The final LayerNorm is what SD1.5 conditions on and what SDXL's pooled row is
    // read from — SDXL's *context* deliberately stops short of it.
    try ctx.opElt(.layernorm, x_d, x_d, try nbuf(ctx, enc.final_ln_w), try nbuf(ctx, enc.final_ln_b), .{
        .u0 = @intCast(total), .u1 = @intCast(h), .f0 = cfg.eps,
    }, total, 1, 1);
    try ctx.endBatch();

    // --- host: download, cache z_empty, apply the weights -------------------
    const all = try gpa.alloc(f32, total * h);
    defer gpa.free(all);
    try ctx.tensorDownload(if (r.capture_layer != null) cap_d else x_d, std.mem.sliceAsBytes(all));
    @memcpy(out, all[0 .. p.seq() * h]);

    if (r.final_chunk0) |dst| {
        if (r.capture_layer == null) {
            @memcpy(dst, all[0 .. clen * h]);
        } else {
            // Needs the post-final-LN activation, which `all` is not when a layer was
            // captured — one extra download of chunk 0's rows only.
            try ctx.tensorDownloadAt(x_d, 0, std.mem.sliceAsBytes(dst));
        }
    }

    if (!p.hasWeights()) return;
    if (need_empty) {
        const e = try gpa.alloc(f32, clen * h);
        errdefer gpa.free(e);
        @memcpy(e, all[p.chunks * clen * h ..][0 .. clen * h]);
        r.empty_cache.* = e;
    }
    // ⚠️ The load-bearing formulas are NOT reimplemented here — this calls the CPU
    // functions, so the three paths cannot disagree about either dialect's weighting.
    const empty = if (r.mode.needsEmpty()) r.empty_cache.*.? else &[_]f32{};
    for (0..p.chunks) |c| enc.applyMode(out[c * clen * h ..][0 .. clen * h], empty, p.chunk(c), r.mode);
}

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
    return gpu.Context.init(gpa) catch error.SkipZigTest;
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
    // pass the wrong kernel. The final assertion is the one that matters — it fails if
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

    // Reference: the CPU op, run per chunk with causal masking — which is exactly what
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
        .u0 = @intCast(total), .u1 = heads, .u2 = heads, .u3 = hd, .u4 = @intCast(clen), .f0 = scale,
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
        .u0 = @intCast(clen), .u1 = heads, .u2 = heads, .u3 = hd, .f0 = scale,
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
        // reference sequence is built from them — a small vocab would be out of range.
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
            }, false);

            const rel = relL2(want, got);
            const rel_f = relL2(want_final, got_final);
            errdefer std.debug.print(
                "clip tower {t} capture={?d}: rel L2 {d:.8}, final_chunk0 {d:.8}\n",
                .{ act, capture, rel, rel_f },
            );
            // f32 GEMMs on both sides, so this is reduction-order noise only: measured
            // 1.1e-5 (no capture) and 3.3e-6 (captured) on this 3-layer tower. The bound
            // is ~4x that rather than a round number — a wrong activation, a missing
            // causal mask or a dropped weight application all miss by >1e-3, so there is
            // room to keep it tight.
            try testing.expect(rel < 5e-5);
            try testing.expect(rel_f < 5e-5);
            // The captured layer must NOT be the final output — if `capture_layer` were
            // ignored, both arms would agree with each other and be wrong together.
            // Compared against chunk 0's rows, which is what `final_chunk0` holds.
            if (capture != null) {
                try testing.expect(relL2(want_final, want[0 .. clen * cfg.hidden]) > 1e-3);
            }
        }
    }
}
