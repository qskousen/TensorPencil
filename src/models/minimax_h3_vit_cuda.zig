//! CUDA-backend Qwen3-VL vision tower, the device twin of `minimax_h3_vit.encode`.
//!
//! The 27-block trunk is the same tower `vit35_cuda` runs and the block loop here
//! is its shape, but the two cannot share code: `vit35` refuses DeepStack outright
//! and flattens its merger into the `Vit`, while this tower taps three block
//! outputs through mergers of their own. What they genuinely share is the rope,
//! which is why the host prep below calls `vit35.applyVisionRope`'s table layout
//! through `minimax_h3_vit`'s own helpers rather than reimplementing either.
//!
//! Host keeps only the cheap prep that `minimax_h3_vit` already exports and that
//! the CPU path uses unchanged: the position interpolation, the merge reordering
//! and the patch coordinates. Those are exactly the two conventions the module doc
//! of `minimax_h3_vit` calls out as diverging from the llama.cpp lineage, so
//! sharing them is what keeps the two paths from drifting apart.
//!
//! Two activation traps, both silent:
//!
//! 1. **The blocks use tanh-gelu and the mergers erf-gelu.** `Backend.gelu` is the
//!    tanh form and `Backend.geluErf` the other; the two agree to ~1e-2, which
//!    shifts style rather than crashing.
//! 2. **A DeepStack merger norms the POST-merge width, the main merger the
//!    PRE-merge one** (`Merger.post_merge_norm`). Same weights, same GEMMs, and
//!    normalizing at the wrong width is finite and wrong.
//!
//! The tower leaves nothing resident: its weights are cached under a scope and
//! dropped as a group, with the attention and conv scratch, because an image-sized
//! working set must not survive under the DiT that follows it.

const std = @import("std");
const h3vit = @import("minimax_h3_vit.zig");
const cuda = @import("tp_gpu").cuda;
const ops = @import("tp_ops");

const Vit = h3vit.Vit;
const Merger = h3vit.Merger;
const Config = h3vit.Config;
const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;
const Weight = ops.matmul.Weight;

/// Whether this tower's weights have a device GEMM here. Refused by dtype rather
/// than met as an `UnsupportedDType` several blocks deep.
pub fn supported(v: *const Vit) bool {
    if (!okDtype(v.patch_w)) return false;
    for (v.blocks) |*b| {
        inline for (.{ b.qkv, b.proj, b.fc1, b.fc2 }) |w| if (!okDtype(w)) return false;
    }
    if (!mergerOk(&v.merger)) return false;
    for (v.deepstack) |*m| if (!mergerOk(m)) return false;
    return true;
}

fn mergerOk(m: *const Merger) bool {
    return okDtype(m.fc1) and okDtype(m.fc2);
}

fn okDtype(w: Weight) bool {
    return switch (w.dtype) {
        .bf16, .f16, .f32 => true,
        else => false,
    };
}

fn gemm(be: *Backend, dst: Buf, src: Buf, m: usize, w: Weight, bias: []const f32) !void {
    switch (w.dtype) {
        .bf16 => try be.opMatmulBf16(dst, src, m, w.bytes, w.rows, w.cols, bias, false, false),
        .f16 => try be.opMatmulF16(dst, src, m, w.bytes, w.rows, w.cols, bias, false, false),
        .f32 => try be.opConvF16(dst, 0, src, m, w.bytes, w.rows, w.cols, bias),
        else => return error.UnsupportedDType,
    }
}

/// A sized sub-view of a device buffer (raw pointer arithmetic, CUDA only).
fn sized(b: Buf, off_bytes: usize, size: usize) Buf {
    return .{ .buf = @enumFromInt(@intFromEnum(b.buf) + off_bytes), .mem = b.mem, .size = size };
}

/// `minimax_h3_vit.encode` on the device. Same contract, same `Encoded`; the
/// numbers differ by the f16 GEMM regime, as every other device twin here does.
pub fn encode(
    v: *const Vit,
    be: *Backend,
    gpa: std.mem.Allocator,
    patches: []const f32,
    gh: usize,
    gw: usize,
) !h3vit.Encoded {
    const cfg = v.cfg;
    const np = gh * gw;
    const dim = cfg.dim;
    const hd = cfg.headDim();
    const hd_pad = std.mem.alignForward(usize, hd, 128);
    const heads = cfg.n_heads;
    const half = cfg.freqsPerAxis();
    const md = cfg.mergeDim();
    const tokens = np / (cfg.merge * cfg.merge);
    std.debug.assert(patches.len == np * cfg.patchDim());
    if (gh % cfg.merge != 0 or gw % cfg.merge != 0) return error.UnsupportedShape;

    be.weightScopeBegin();
    defer {
        be.weightScopeEnd();
        be.freeAttnScratch();
        be.freeConvScratch();
    }

    var bufs: [8]Buf = @splat(.{});
    defer for (&bufs) |*b| be.tensorDestroy(b);
    const sizes = [bufs.len]usize{
        np * dim, // x, the residual stream
        @max(np * dim, tokens * cfg.out_dim), // normed; also a merger's rows and its output
        np * 3 * dim, // fused qkv
        np * heads * hd_pad, // q, padded heads
        np * heads * hd_pad, // k
        np * heads * hd_pad, // v
        @max(np * @max(heads * hd_pad, cfg.ffn), np * cfg.patchDim()), // attn out / FFN / patch upload
        @max(np * dim, tokens * cfg.out_dim), // residual delta, and the merger output
    };
    for (&bufs, sizes) |*b, size| b.* = try be.tensorCreate(size * 4);
    const x_d = bufs[0];
    const normed_d = bufs[1];
    const qkv_d = bufs[2];
    const q_d = bufs[3];
    const k_d = bufs[4];
    const v_d = bufs[5];
    const big_d = bufs[6];
    const t_d = bufs[7];

    // Host prep, shared verbatim with the CPU path.
    const pos_row = try gpa.alloc(f32, np * dim);
    defer gpa.free(pos_row);
    h3vit.interpolatePos(pos_row, v.pos_embed, cfg.pos_grid, dim, gh, gw);
    const pos = try gpa.alloc(f32, np * dim);
    defer gpa.free(pos);
    h3vit.mergeOrder(pos, pos_row, dim, gh, gw, cfg.merge);

    const cos = try gpa.alloc(f32, cfg.pos_grid * half);
    defer gpa.free(cos);
    const sin = try gpa.alloc(f32, cfg.pos_grid * half);
    defer gpa.free(sin);
    for (0..cfg.pos_grid) |p| {
        for (0..half) |kk| {
            const exp = @as(f64, @floatFromInt(2 * kk)) / @as(f64, @floatFromInt(2 * half));
            const inv = std.math.pow(f64, cfg.rope_theta, -exp);
            const ang = @as(f64, @floatFromInt(p)) * inv;
            cos[p * half + kk] = @floatCast(@cos(ang));
            sin[p * half + kk] = @floatCast(@sin(ang));
        }
    }
    const py = try gpa.alloc(u32, np);
    defer gpa.free(py);
    const px = try gpa.alloc(u32, np);
    defer gpa.free(px);
    h3vit.patchCoords(py, px, gh, gw, cfg.merge);
    const pos2 = try gpa.alloc(u32, np * 2);
    defer gpa.free(pos2);
    for (0..np) |t| {
        pos2[t * 2] = py[t];
        pos2[t * 2 + 1] = px[t];
    }

    const sin_off = cfg.pos_grid * half;
    var freqs_d = try be.tensorCreate(2 * sin_off * 4);
    defer be.tensorDestroy(&freqs_d);
    try be.tensorUpload(sized(freqs_d, 0, sin_off * 4), std.mem.sliceAsBytes(cos));
    try be.tensorUpload(sized(freqs_d, sin_off * 4, sin_off * 4), std.mem.sliceAsBytes(sin));
    var pos2_d = try be.tensorCreate(np * 2 * 4);
    defer be.tensorDestroy(&pos2_d);
    try be.tensorUpload(pos2_d, std.mem.sliceAsBytes(pos2));

    // One output buffer per merger, so nothing waits on the device mid-batch: a
    // download inside the batch would force it once per DeepStack tap.
    var outs: [h3vit.max_deepstack + 1]Buf = @splat(.{});
    defer for (&outs) |*b| be.tensorDestroy(b);
    for (outs[0 .. cfg.n_deepstack + 1]) |*b| b.* = try be.tensorCreate(tokens * cfg.out_dim * 4);

    try be.beginBatch();
    errdefer if (be.batching()) be.abortBatch();

    // Patch embedding, then the position table already in merged order. The
    // patches arrive in merged block order too, so nothing here permutes tokens.
    try be.tensorUpload(sized(big_d, 0, np * cfg.patchDim() * 4), std.mem.sliceAsBytes(patches));
    try gemm(be, x_d, big_d, np, v.patch_w, v.patch_b);
    try be.tensorUpload(sized(t_d, 0, np * dim * 4), std.mem.sliceAsBytes(pos));
    try be.opAdd(x_d, t_d, np * dim);

    // The scale is over the REAL head width, not the padded one.
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
    for (v.blocks, 0..) |*b, bi| {
        try be.opLayerNorm(x_d, normed_d, b.norm1_w, b.norm1_b, np, dim, cfg.eps, false);
        try gemm(be, qkv_d, normed_d, np, b.qkv, b.qkv_b);
        // The projection emits [t][3][heads][hd]; attention wants three planes,
        // and the 72-wide heads pad to 128 so the tensor-core PV GEMM applies.
        try be.opHeadPad(q_d, qkv_d, np, heads, hd_pad, hd, 3 * dim, 0);
        try be.opHeadPad(k_d, qkv_d, np, heads, hd_pad, hd, 3 * dim, dim);
        try be.opHeadPad(v_d, qkv_d, np, heads, hd_pad, hd, 3 * dim, 2 * dim);
        try be.opRopeVision(q_d, pos2_d, freqs_d, np, heads, half, sin_off, hd_pad);
        try be.opRopeVision(k_d, pos2_d, freqs_d, np, heads, half, sin_off, hd_pad);
        // One image is one attention sequence, never masked within it.
        try be.opAttnTC(q_d, k_d, v_d, big_d, np, heads, heads, hd_pad, scale);
        try be.opHeadPad(normed_d, big_d, np, heads, hd, hd_pad, heads * hd_pad, 0);
        try gemm(be, t_d, normed_d, np, b.proj, b.proj_b);
        try be.opAdd(x_d, t_d, np * dim);

        try be.opLayerNorm(x_d, normed_d, b.norm2_w, b.norm2_b, np, dim, cfg.eps, false);
        try gemm(be, big_d, normed_d, np, b.fc1, b.fc1_b);
        try be.gelu(big_d, np * cfg.ffn); // tanh here; the mergers use erf
        try gemm(be, t_d, big_d, np, b.fc2, b.fc2_b);
        try be.opAdd(x_d, t_d, np * dim);

        // DeepStack reads this block's OUTPUT, not its input.
        for (cfg.deepstack_indexes[0..cfg.n_deepstack], 0..) |idx, di| {
            if (idx != bi) continue;
            try runMerger(be, &v.deepstack[di], cfg, x_d, normed_d, t_d, outs[di], np, tokens, md);
        }
    }

    try runMerger(be, &v.merger, cfg, x_d, normed_d, t_d, outs[cfg.n_deepstack], np, tokens, md);
    try be.endBatch();

    var enc: h3vit.Encoded = .{
        .merged = &.{},
        .deepstack = try gpa.alloc([]f32, cfg.n_deepstack),
        .tokens = tokens,
    };
    var n_ds: usize = 0;
    errdefer {
        for (enc.deepstack[0..n_ds]) |d| gpa.free(d);
        gpa.free(enc.deepstack);
        if (enc.merged.len > 0) gpa.free(enc.merged);
    }
    for (0..cfg.n_deepstack) |di| {
        enc.deepstack[di] = try gpa.alloc(f32, tokens * cfg.out_dim);
        n_ds += 1;
        try be.tensorDownload(sized(outs[di], 0, tokens * cfg.out_dim * 4), std.mem.sliceAsBytes(enc.deepstack[di]));
    }
    enc.merged = try gpa.alloc(f32, tokens * cfg.out_dim);
    try be.tensorDownload(sized(outs[cfg.n_deepstack], 0, tokens * cfg.out_dim * 4), std.mem.sliceAsBytes(enc.merged));
    return enc;
}

/// One merger: LayerNorm at one of two widths, the 2x2 merge, `fc2(geluErf(fc1))`.
///
/// The merge itself is free: tokens are already in block order, so four
/// consecutive rows ARE one merged row and `normed` holds either reading of it
/// (`tokens * md == np * dim`).
///
fn runMerger(
    be: *Backend,
    m: *const Merger,
    cfg: Config,
    x_d: Buf,
    normed_d: Buf,
    t_d: Buf,
    out_d: Buf,
    np: usize,
    tokens: usize,
    md: usize,
) !void {
    if (m.post_merge_norm) {
        // Merge first, then normalize the full merged width.
        try be.opLayerNorm(x_d, normed_d, m.norm_w, m.norm_b, tokens, md, cfg.eps, false);
    } else {
        // Normalize each pre-merge token, then merge (a reinterpret).
        try be.opLayerNorm(x_d, normed_d, m.norm_w, m.norm_b, np, cfg.dim, cfg.eps, false);
    }
    try gemm(be, t_d, normed_d, tokens, m.fc1, m.fc1_b);
    try be.geluErf(t_d, tokens * md);
    try gemm(be, out_d, t_d, tokens, m.fc2, m.fc2_b);
}
