//! CLIP text tower on the CUDA backend — SD1.5's CLIP-L and SDXL's CLIP-G.
//!
//! The CUDA twin of `clip_text_gpu`, in the shape `embed_siglip_cuda` established:
//! device body via `cuda.Backend` ops, cheap head (the pooled row and its projection)
//! on the host, weights borrowed from the mmap-stable CPU model and left resident
//! across calls.
//!
//! Two differences from the Vulkan arm, both because the CUDA backend already had what
//! was missing there:
//!
//! - **Attention needs no new kernel**: `Backend.attn` already takes a `causal` flag.
//!   Chunks are driven as one launch each over `dbOffset` views rather than as a
//!   block-diagonal batch, because the batched CUDA kernel is the non-causal one and a
//!   prompt has one to three chunks — not worth a fourth attention kernel.
//! - ⚠️ **Precision is NOT the caller's choice here**: `opConvF16` is the only wide GEMM
//!   entry point, so a CLIP-L-sized layer computes in f16 regardless of
//!   `--encoder-f16`. That is the same regime `sd_unet_cuda` and `embed_siglip_cuda`
//!   run in, so parity against the CPU forward is *relative* (~1e-3), not the ~1e-6 the
//!   Vulkan f32 path holds. `Session.encode`'s fallback order matters for that reason
//!   and is documented there.
//!
//! Validated by `TensorPencil sd-cuda-test`, not by a unit test: the test binary brings
//! up no CUDA context (see the SD-family section of CLAUDE.md).

const std = @import("std");
const tp_core = @import("tp_core");
const cuda = @import("tp_gpu").cuda;
const ops = @import("tp_ops");
const safetensors = tp_core.safetensors;
const clip_tok = tp_core.clip_tokenizer;
const clip_text = @import("clip_text.zig");

const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;
const Weight = ops.matmul.Weight;
const clen = clip_tok.context_length;

/// Smallest output width that goes to the tensor-core GEMM; below it the plain f32
/// kernel wins, exactly as in `sd_unet_cuda`.
const coop_min_co = 64;

/// A GEMM against a checkpoint weight, routed by its stored dtype. Every CLIP linear
/// carries a bias, so there is no zero-vector fallback and no bias-cache aliasing
/// hazard to guard against here.
fn gemm(be: *Backend, y: Buf, x: Buf, m: usize, wt: Weight, bias: []const f32) !void {
    const rows = wt.rows;
    const cols = wt.cols;
    switch (wt.dtype) {
        .f32 => {
            if (rows >= coop_min_co) return be.opConvF16(y, 0, x, m, wt.bytes, rows, cols, bias);
            return be.opMatmul(y, 0, x, 0, m, wt.bytes, false, rows, cols, 1.0, bias);
        },
        .f16 => return be.opMatmulF16(y, x, m, wt.bytes, rows, cols, bias),
        .bf16 => return be.opMatmulBf16(y, x, m, wt.bytes, rows, cols, bias),
        else => return error.UnsupportedDType,
    }
}

/// `clip_text.TextEncoder.encodePrompt` on the device. Identical contract to the CPU
/// form, including `r.final_chunk0` and the `r.empty_cache` slot.
pub fn encodePrompt(
    enc: *const clip_text.TextEncoder,
    be: *Backend,
    io: std.Io,
    gpa: std.mem.Allocator,
    out: []f32,
    p: *const clip_tok.Prompt,
    r: clip_text.TextEncoder.PromptRun,
) !void {
    // Taken for signature symmetry with the CPU form; nothing here reaches the host GEMM.
    _ = io;
    const cfg = enc.cfg;
    const h = cfg.hidden;
    const inter = cfg.intermediate;
    const heads = cfg.heads;
    const hd = cfg.headDim();
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
    std.debug.assert(out.len == p.seq() * h);

    // The empty reference rides along as one more chunk, but only when it is needed and
    // not already cached — the same policy as the Vulkan arm.
    const need_empty = p.hasWeights() and r.mode.needsEmpty() and r.empty_cache.* == null;
    const items = p.chunks + @as(usize, if (need_empty) 1 else 0);
    const total = items * clen;

    // --- host: ids, then token + learned positional embeddings ----------------
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

    // No weight scope: the f16-converted GEMM weights stay RESIDENT across calls (they
    // borrow the mmap-stable CPU model). A scope would free and re-upload the whole
    // tower every encode — a per-call cost that dwarfs the compute.
    defer {
        be.freeAttnScratch();
        be.freeConvScratch();
    }

    var bufs: [7]Buf = @splat(.{});
    defer for (&bufs) |*b| be.tensorDestroy(b);
    const sizes = [bufs.len]usize{ total * h, total * h, total * h, total * h, total * h, total * inter, total * h };
    for (&bufs, sizes) |*b, s| b.* = try be.tensorCreate(s * 4);
    const x_d = bufs[0];
    const normed_d = bufs[1];
    const q_d = bufs[2];
    const k_d = bufs[3];
    const v_d = bufs[4];
    const big_d = bufs[5];
    const t_d = bufs[6];
    // The captured layer must survive to the end of the batch, so it gets its own
    // buffer rather than a mid-batch download (which would force a second submit).
    var cap_d: Buf = .{};
    defer be.tensorDestroy(&cap_d);
    if (r.capture_layer != null) cap_d = try be.tensorCreate(total * h * 4);

    try be.tensorUpload(x_d, std.mem.sliceAsBytes(x_host));
    try be.beginBatch();
    errdefer if (be.batching()) be.abortBatch();

    if (r.capture_layer) |layer| if (layer == 0) {
        try be.opCopyOff(cap_d, 0, x_d, 0, total * h);
    };

    for (enc.layers, 0..) |*l, li| {
        // --- attention: x += o(causal_attn(ln1(x))) ---
        try be.opLayerNorm(x_d, normed_d, l.ln1_w, l.ln1_b, total, h, cfg.eps);
        try gemm(be, q_d, normed_d, total, l.q, l.q_b);
        try gemm(be, k_d, normed_d, total, l.k, l.k_b);
        try gemm(be, v_d, normed_d, total, l.v, l.v_b);
        // One causal launch per chunk: each window attends only within itself, and
        // `attn`'s batched sibling is the non-causal one.
        for (0..items) |c| {
            const off = c * clen * h * @sizeOf(f32);
            try be.attn(
                cuda.backend.dbOffset(q_d, off),
                cuda.backend.dbOffset(k_d, off),
                cuda.backend.dbOffset(v_d, off),
                cuda.backend.dbOffset(big_d, off),
                clen,
                clen,
                heads,
                heads,
                hd,
                scale,
                true,
            );
        }
        try gemm(be, t_d, big_d, total, l.o, l.o_b);
        try be.opAdd(x_d, t_d, total * h);

        // --- mlp: x += fc2(act(fc1(ln2(x)))) ---
        try be.opLayerNorm(x_d, normed_d, l.ln2_w, l.ln2_b, total, h, cfg.eps);
        try gemm(be, big_d, normed_d, total, l.fc1, l.fc1_b);
        switch (cfg.act) {
            .quick_gelu => try be.geluQuick(big_d, total * inter),
            .gelu_erf => try be.geluErf(big_d, total * inter),
        }
        try gemm(be, t_d, big_d, total, l.fc2, l.fc2_b);
        try be.opAdd(x_d, t_d, total * h);

        if (r.capture_layer) |layer| if (layer == li + 1) {
            try be.opCopyOff(cap_d, 0, x_d, 0, total * h);
        };
    }

    // The final LayerNorm is what SD1.5 conditions on and where SDXL's pooled row comes
    // from — SDXL's *context* deliberately stops short of it.
    try be.opLayerNorm(x_d, x_d, enc.final_ln_w, enc.final_ln_b, total, h, cfg.eps);
    try be.endBatch();

    // --- host: download, cache z_empty, apply the weights ---------------------
    const all = try gpa.alloc(f32, total * h);
    defer gpa.free(all);
    try be.tensorDownload(if (r.capture_layer != null) cap_d else x_d, std.mem.sliceAsBytes(all));
    @memcpy(out, all[0 .. p.seq() * h]);

    if (r.final_chunk0) |dst| {
        if (r.capture_layer == null) {
            @memcpy(dst, all[0 .. clen * h]);
        } else {
            // Needs the post-final-LN activation, which `all` is not when a layer was
            // captured — one extra download of chunk 0's rows only.
            const finals = try gpa.alloc(f32, total * h);
            defer gpa.free(finals);
            try be.tensorDownload(x_d, std.mem.sliceAsBytes(finals));
            @memcpy(dst, finals[0 .. clen * h]);
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
    // A1111's forms need no empty-prompt reference at all, so it may be unset.
    const empty = if (r.mode.needsEmpty()) r.empty_cache.*.? else &[_]f32{};
    for (0..p.chunks) |c| enc.applyMode(out[c * clen * h ..][0 .. clen * h], empty, p.chunk(c), r.mode);
}
