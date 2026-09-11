//! GPU-resident SenseNova U1.5 on Vulkan, the device twin of `models/sensenova.zig`
//! and the sibling of `sensenova_cuda.zig`, which its stage order follows exactly so
//! the two can be diffed.
//!
//! Both passes run here, and they are different shapes rather than one shape twice:
//!
//! - the PREFIX pass runs the base MoT copy over the prompt, once per conditioning,
//!   and keeps every layer's keys and values. Its per-query key range is a buffer,
//!   not a rule: a plain prompt is `[0, i+1)` and one carrying reference pictures is
//!   block-causal, so `attnBatched` serves both and the bounds say which;
//! - the GENERATION pass runs the `_mot_gen` copy over the canvas tokens every step,
//!   unmasked, over the prefix KV concatenated with its own. That is rectangular GQA
//!   attention (`seq_q` = canvas against `seq_kv` = prefix + canvas), which is what
//!   the two-pass flash pipeline already does.
//!
//! The per-head norms and the RoPE both work on SPANS of the head rather than the
//! whole of it (64 for the sequence position, then 32 + 32 for the token's row and
//! column), which is why `group_rmsnorm` and `rope_half_span_pos` are used where the
//! other families call a plain head norm and `rope_half`.
//!
//! Two things differ from the CUDA arm, both forced by the backend rather than
//! chosen. A device buffer is an opaque HANDLE here, not a pointer, so there is no
//! `offsetBuf`: a partial read is `tensorDownloadAt` and a concatenation is the
//! `copy` kernel with source and destination offsets. And `tensorCopy` flushes an
//! open recording batch, so the per-layer concatenation goes through that kernel
//! instead, which keeps a whole layer in one submission.

const std = @import("std");
const sensenova = @import("sensenova.zig");
const lin = @import("lin.zig");
const sd_unet_gpu = @import("sd_unet_gpu.zig");
const gpu = @import("tp_gpu").context;
const ops = @import("tp_ops");

const Model = sensenova.Model;
const Config = sensenova.Config;
const Prefix = sensenova.Prefix;
const Context = gpu.Context;
const Buf = gpu.DeviceBuffer;
const Weight = ops.matmul.Weight;

/// Force the naive one-thread-per-(query, head) attention instead of the flash
/// pipeline. For A/B and for reproducing a mismatch; the device test runs both.
pub var force_naive_attn: bool = false;

/// The flash pipeline tiles queries 128 at a time and stages keys the same way.
const flash_tile = 128;

/// Cap on one im2col band, so a 3x3 over a 1024-wide canvas does not materialize a
/// [4096][9216] patch in one allocation.
const patch_band_bytes: usize = 64 << 20;

/// Widest zero bias any linear here needs (`inter`).
///
/// Passed WHOLE, never sliced: `smallBuffer` caches by host POINTER alone and sizes
/// from the first call, so once a narrow GEMM cached it a wider one would read off
/// the end. The dispatch reads only `rows` entries, so one full-width buffer serves
/// every GEMM, the rule `dit_gpu`, `zimage_gpu` and `anima_gpu` all record.
const zero_bias: [sensenova.u15_8b.inter]f32 = @splat(0);

/// Whether this context can run every linear this model has, in BOTH streams.
///
/// Both lists, not just the generation one: a checkpoint that quantized the two MoT
/// copies differently would otherwise build a session and fail inside the encode.
pub fn supported(ctx: *Context, model: *const Model) bool {
    if (model.layers.len == 0) return false;
    // This arm has no sidecar apply yet, and rendering the base GEMMs alone is a
    // different model with no error anywhere. Refusing sends the trunk to the
    // CPU, which is slow and right.
    if (model.lora != null) {
        std.log.warn("sensenova_gpu: no LoRA sidecar path on Vulkan; running the trunk on the CPU instead", .{});
        return false;
    }
    // The generation attention has no non-tensor-core form that is also fast enough
    // to render with, but it does have a correct one, so a device without the flash
    // pipeline is not refused here; `Session.tc` records what it got and `attention`
    // branches on that.
    const has_i8 = ctx.pipe_coop_i8 != .null_handle;
    const has_f16w = ctx.pipe_coop_bf16w != .null_handle or ctx.pipe_coop_f16w != .null_handle;
    const caps: lin.Caps = .{
        .f32 = true,
        .fp8 = true,
        .bf16 = has_f16w,
        .f16 = has_f16w,
        .i8 = has_i8,
        .i4 = has_i8 and ctx.hasI4Decode(),
        .w4a8 = has_i8 and ctx.hasW4A8Decode(),
        .nvfp4 = ctx.hasNvfp4Decode(),
    };
    // The convolutions are f32 weights through the f16-weight coop GEMM, and there is
    // no scalar fallback wide enough to be worth offering: without it the patch
    // embedder and fm_head have no path.
    if (!has_f16w) {
        std.log.err("sensenova_gpu: this device has no f16-weight GEMM, which the convolutions need", .{});
        return false;
    }
    for ([2][]const Weight{ model.device_lins, model.prefix_lins }) |lins| {
        if (lin.unsupported(lins, caps)) |bad| {
            std.log.err("sensenova_gpu: {s} is {t}, which this device has no GEMM for", .{ bad.tag, bad.dtype });
            return false;
        }
    }
    return true;
}

// --- shared pieces ----------------------------------------------------------

/// Wrap a host f32 slice as a (pointer-cached) small device buffer.
fn smallBuf(ctx: *Context, values: []const f32) !Buf {
    return .{ .buf = try ctx.smallBuffer(std.mem.sliceAsBytes(values)), .mem = .null_handle, .size = 0 };
}

/// A device buffer of its own for a u32 position array, owned by the caller.
///
/// NOT `smallBuffer`: that cache is keyed on the HOST POINTER, so a scratch array
/// freed here whose address is later recycled would be served this upload.
fn uploadU32(ctx: *Context, values: []const u32) !Buf {
    var buf = try ctx.tensorCreate(values.len * 4);
    errdefer ctx.tensorDestroy(&buf);
    try ctx.tensorUpload(buf, std.mem.sliceAsBytes(values));
    return buf;
}

/// A rotate-half table flattened as `cos` then `sin`, the layout every rope kernel
/// here reads with `sin_off = rows * half`.
const Table = struct {
    buf: Buf,
    half: usize,
    sin_off: usize,

    fn init(ctx: *Context, gpa: std.mem.Allocator, rows: usize, span: usize, theta: f64) !Table {
        var f = try ops.rope.rotateHalfFreqs(gpa, rows, span, theta);
        defer f.deinit(gpa);
        return upload(ctx, gpa, f, rows);
    }

    fn initInterleaved(ctx: *Context, gpa: std.mem.Allocator, rows: usize, dim: usize, theta: f64) !Table {
        const pos = try gpa.alloc(f32, rows);
        defer gpa.free(pos);
        for (pos, 0..) |*p, i| p.* = @floatFromInt(i);
        var f = try ops.rope.fluxFreqs(gpa, pos, &.{dim}, theta);
        defer f.deinit(gpa);
        return upload(ctx, gpa, f, rows);
    }

    fn upload(ctx: *Context, gpa: std.mem.Allocator, f: ops.rope.Freqs, rows: usize) !Table {
        const flat = try gpa.alloc(f32, 2 * rows * f.half);
        defer gpa.free(flat);
        @memcpy(flat[0 .. rows * f.half], f.cos[0 .. rows * f.half]);
        @memcpy(flat[rows * f.half ..], f.sin[0 .. rows * f.half]);
        var buf = try ctx.tensorCreate(flat.len * 4);
        errdefer ctx.tensorDestroy(&buf);
        try ctx.tensorUpload(buf, std.mem.sliceAsBytes(flat));
        return .{ .buf = buf, .half = f.half, .sin_off = rows * f.half };
    }

    fn deinit(self: *Table, ctx: *Context) void {
        ctx.tensorDestroy(&self.buf);
    }
};

/// Per-head RMSNorm over the head dim's two equally wide spans, in one launch.
///
/// `norm_w` is the layer's `[q_norm ++ q_norm_hw]` pair as one 128-wide vector, so
/// each half of the head is normalized against its own scale: the two spans are
/// contiguous and equally wide, which is exactly the group form.
fn groupNorm(ctx: *Context, x: Buf, w: Buf, rows: usize, dim: usize, groups: usize, eps: f32) !void {
    std.debug.assert(groups > 0 and dim % groups == 0);
    try ctx.opElt(.group_rmsnorm, x, x, w, null, .{
        .u0 = @intCast(rows * groups),
        .u1 = @intCast(dim / groups),
        .u2 = @intCast(groups),
        .f0 = eps,
    }, rows * groups, 1, 1);
}

/// The per-head norms and the three-span rope, applied to a q or k buffer in place.
fn qkPrepare(
    ctx: *Context,
    cfg: Config,
    x: Buf,
    rows: usize,
    heads: usize,
    norm_w: Buf,
    pos: [3]Buf,
    tables: [3]Table,
) !void {
    try groupNorm(ctx, x, norm_w, rows * heads, cfg.head_dim, 2, cfg.norm_eps);
    const t_span = cfg.spanT();
    const hw = cfg.spanHw();
    const offs = [3]usize{ 0, t_span, t_span + hw };
    inline for (0..3) |i| {
        const tb = tables[i];
        std.debug.assert(offs[i] + 2 * tb.half <= cfg.head_dim);
        const total = rows * heads * tb.half;
        try ctx.opElt(.rope_half_span_pos, x, pos[i], tb.buf, null, .{
            .u0 = @intCast(total),
            .u1 = @intCast(tb.half),
            .u2 = @intCast(tb.sin_off),
            .u3 = @intCast(heads),
            .u4 = @intCast(cfg.head_dim),
            .u5 = @intCast(offs[i]),
        }, total, 1, 1);
    }
}

/// A non-overlapping k x k stride-k convolution over channel-last `[h][w][ci]`:
/// gather the patch matrix, then one GEMM. `dst` is `[h/k * w/k][co]`.
fn convStride(ctx: *Context, patch: *Buf, dst: Buf, src: Buf, h: usize, w: usize, cv: sensenova.Conv) !void {
    std.debug.assert(h % cv.k == 0 and w % cv.k == 0);
    const n_out = (h / cv.k) * (w / cv.k);
    const plen = cv.k * cv.k * cv.ci;
    const total = n_out * plen;
    try ctx.ensureDeviceBuffer(patch, total * 4);
    try ctx.opElt(.im2col_stride, src, patch.*, null, null, .{
        .u0 = @intCast(total),
        .u1 = @intCast(cv.ci),
        .u2 = @intCast(cv.k),
        .u3 = @intCast(w),
    }, total, 1, 1);
    try ctx.opMatmulCoopF16W(dst, 0, patch.*, n_out, cv.w, cv.co, plen, cv.b.?);
}

/// A 3x3 pad-1 stride-1 convolution over channel-last `[h][w][ci]`, through the SD
/// family's banded im2col and the same f16-weight GEMM.
fn conv3x3(ctx: *Context, patch: *Buf, dst: *Buf, src: *const Buf, h: usize, w: usize, cv: sensenova.Conv) !void {
    std.debug.assert(cv.k == 3 and cv.stride == 1 and cv.pad == 1);
    try sd_unet_gpu.convIntoPrec(ctx, patch, dst, src, h, w, cv, .stride1, null, null, 1.0, false, false);
}

/// Concatenate within the recorded batch: `dst[dst_off..][0..n] = src[src_off..]`.
///
/// `Context.tensorCopy` would do the same bytes, but it runs on the immediate command
/// buffer and flushes any open batch, which would cut every layer into three
/// submissions.
fn copyInto(ctx: *Context, dst: Buf, dst_off: usize, src: Buf, src_off: usize, n: usize) !void {
    try ctx.opElt(.copy, src, dst, null, null, .{
        .u0 = @intCast(n),
        .u2 = @intCast(dst_off),
        .u3 = @intCast(src_off),
    }, n, 1, 1);
}

// --- linears ----------------------------------------------------------------

/// Quantize+rotate the activation once for a group of GEMMs that share it, if any
/// member needs it. Mirrors `anima_gpu.prepGroup`, and see it for why `.i4` takes the
/// int8 prep on this backend.
fn prepGroup(ctx: *Context, x: Buf, m: usize, cols: usize, group: []const Weight) !void {
    var want: lin.Prep = .none;
    for (group) |w| {
        const k = lin.prepOf(lin.kindOf(w.dtype));
        if (k == .none) continue;
        if (want != .none and want != k) {
            std.log.err("sensenova_gpu: a linear group mixes {t} and {t} activation preps; one cannot serve both", .{ want, k });
            return error.UnsupportedCheckpoint;
        }
        want = k;
    }
    switch (want) {
        .i8, .i4 => {
            // Whether the prep rotates is the checkpoint's answer (`lin.convrot`): the
            // group rotation cancels across the GEMM only when both sides apply it.
            const rot = lin.convrot(group) orelse {
                std.log.err("sensenova_gpu: a linear group mixes convrot and plain int8 weights; one activation prep cannot serve both", .{});
                return error.UnsupportedCheckpoint;
            };
            // `i4Decode` emits a weight quantized after the rotation and has no
            // unrotated build.
            if (want == .i4 and !rot) {
                std.log.err("sensenova_gpu: int4 without convrot has no unrotated decode on this backend", .{});
                return error.UnsupportedCheckpoint;
            }
            try ctx.opI8PrepR(x, m, cols, rot);
        },
        .none => {},
    }
}

/// One trunk linear. A quantized weight reads the prep state and fuses the per-row
/// rescale; a dense one goes to `gemm`.
///
/// `opI8Gemm` wants `rows % 64 == 0`; every linear the device runs here is 1024,
/// 4096 or 12288.
fn linear(ctx: *Context, y: Buf, x: Buf, m: usize, w: Weight) !void {
    switch (lin.kindOf(w.dtype)) {
        .i8 => {
            std.debug.assert(w.rows % 64 == 0);
            try ctx.opI8Gemm(y, w.bytes, w.row_scale.?, w.rows, false);
        },
        .i4 => {
            std.debug.assert(w.rows % 64 == 0);
            const wbuf = try ctx.i4Decode(w.bytes, w.rows, w.cols);
            try ctx.opI8GemmBuf(y, wbuf, &.{}, w.row_scale.?, w.rows, false);
        },
        .w4a8 => {
            std.debug.assert(w.rows % 64 == 0);
            const meta = w.w4a8.?;
            const wbuf = try ctx.w4a8Decode(w.bytes, meta.s_rel, std.mem.asBytes(meta.levels), w.rows, w.cols, meta.group_size);
            try ctx.opI8GemmBuf(y, wbuf, &.{}, w.row_scale.?, w.rows, false);
        },
        .nvfp4 => {
            std.debug.assert(w.rows % 128 == 0 and w.cols % 32 == 0);
            const meta = w.nvfp4.?;
            const zeros: []const f32 = &zero_bias;
            std.debug.assert(w.rows <= zeros.len);
            try ctx.opMatmulNvfp4(y, x, m, w.bytes, meta.scales, std.mem.asBytes(&meta.levels.bf16v), w.rows, w.cols, zeros);
        },
        else => try gemm(ctx, y, x, m, w),
    }
}

/// A dense linear. Every trunk weight is bias-free, but the f16-weight coop GEMM
/// always folds one in, so it is handed the FULL zero vector (see `zero_bias`).
fn gemm(ctx: *Context, y: Buf, x: Buf, m: usize, w: Weight) !void {
    const zeros: []const f32 = &zero_bias;
    std.debug.assert(w.rows <= zeros.len);
    switch (w.dtype) {
        .bf16 => {
            if (ctx.pipe_coop_bf16w != .null_handle) {
                try ctx.opMatmulCoopBf16(y, 0, x, m, w.bytes, w.rows, w.cols, zeros);
            } else {
                try ctx.opMatmulCoopF16Wb(y, 0, x, m, w.bytes, w.rows, w.cols, zeros);
            }
        },
        .f16 => try ctx.opMatmulCoopF16Wh(y, 0, x, m, w.bytes, w.rows, w.cols, zeros),
        .f32, .f8_e4m3 => try ctx.opMatmul(y, 0, x, 0, m, w.bytes, w.dtype == .f8_e4m3, w.rows, w.cols, w.scale, null),
        // `supported` gates this before a session is built, so reaching it is a
        // programming error rather than a bad checkpoint.
        else => return error.UnsupportedDType,
    }
}

// --- attention --------------------------------------------------------------

/// Everything the two attention forms need that is not a workspace buffer.
const AttnArgs = struct {
    /// Rows of `kcat`/`vcat` already filled with the prefix cache; 0 for the prefix
    /// pass, which attends over itself alone.
    prefix_rows: usize,
    scale: f32,
    /// Per-query key range (`u32[2*rows]`, starts then ends) when the pass is masked.
    /// Null means unmasked over the whole of `seq_kv`, which is the generation pass.
    bounds: ?Buf,
    /// True when the flash pipeline's f16 operands are wanted and sized for.
    tc: bool,
};

/// Attention from `b.q` against `b.kcat`/`b.vcat` into `b.attn`.
///
/// The two paths compute the same thing and differ in how the scores are produced.
/// The naive one is a thread per (query, head) reading the whole key range, which is
/// what the prefix pass and a mismatch hunt want; the flash one tiles it on the
/// tensor cores, which is what a render at 1024 canvas tokens against a 1289-key
/// context wants.
fn attention(ctx: *Context, cfg: Config, b: LayerBufs, rows: usize, seq_kv: usize, a: AttnArgs) !void {
    const hd = cfg.head_dim;
    const heads = cfg.n_heads;
    const n_kv = cfg.n_kv_heads;
    if (!a.tc) {
        // The bounds buffer is the mask. `Workspace`/`PrefixWs` fill an unmasked one
        // with `[0, seq_kv)` for every row, so this arm never needs a second kernel.
        return ctx.attnBatchedDispatch(rows, heads, n_kv, hd, a.scale);
    }
    const q_pad = std.mem.alignForward(usize, rows, flash_tile);
    const kv_pad = std.mem.alignForward(usize, seq_kv, flash_tile);
    // The scale rides into the f16 cast of Q rather than costing a pass of its own.
    try ctx.opElt(.f32_to_h16, b.q, null, null, b.q16, .{
        .u0 = @intCast(q_pad * heads * hd / 2),
        .u1 = @intCast(rows * heads * hd),
        .f0 = a.scale,
    }, q_pad * heads * hd / 2, 1, 1);
    try ctx.opElt(.gather_kmajor_h16, b.kcat, null, null, b.k16, .{
        .u0 = @intCast(n_kv * hd * kv_pad / 2),
        .u1 = @intCast(hd),
        .u2 = @intCast(kv_pad),
        .u3 = @intCast(seq_kv),
        .u4 = @intCast(n_kv),
    }, n_kv * hd * kv_pad / 2, 1, 1);
    try ctx.opElt(.f32_to_h16, b.vcat, null, null, b.v16, .{
        .u0 = @intCast(kv_pad * n_kv * hd / 2),
        .u1 = @intCast(seq_kv * n_kv * hd),
        .f0 = 1.0,
    }, kv_pad * n_kv * hd / 2, 1, 1);

    // The MD table is one {max, 1/sum} pair per query row per head, and it lives past
    // the attention output in the same buffer.
    const md_off = q_pad * heads * hd;
    const push: gpu.EltPush = .{
        .u0 = @intCast(heads * hd),
        .u1 = @intCast(kv_pad),
        .u2 = 0,
        .u3 = @intCast(heads / n_kv),
        .u4 = @intCast(n_kv * hd),
        .u5 = @intCast(md_off),
        .f0 = @bitCast(@as(u32, @intCast(seq_kv))),
        // The MD table is indexed by QUERY row, so its plane stride is `q_pad`, not
        // `u1`, which is the key padded length. 0 would mean "= u1" and is only right
        // where the two coincide.
        .f1 = if (q_pad == kv_pad) 0 else @bitCast(@as(u32, @intCast(q_pad))),
    };
    try ctx.opFlash(.md, b.q16, b.k16, b.v16, b.attn, push, q_pad / flash_tile, heads);
    try ctx.opFlash(.out, b.q16, b.k16, b.v16, b.attn, push, q_pad / flash_tile, heads);
}

/// The `[starts ++ ends]` table `attnBatched` reads, as the pass's mask.
///
/// `ends` is the caller's when the prompt carries reference pictures (block-causal,
/// so a picture's tokens see each other in full and the range runs past the query's
/// own position), plain causal when it does not, and the whole key range when the
/// pass is unmasked.
fn boundsTable(gpa: std.mem.Allocator, rows: usize, seq_kv: usize, kv_end: ?[]const u32) ![]u32 {
    const pair = try gpa.alloc(u32, 2 * rows);
    @memset(pair[0..rows], 0);
    if (kv_end) |ends| {
        std.debug.assert(ends.len == rows);
        @memcpy(pair[rows..], ends);
    } else if (rows == seq_kv) {
        for (pair[rows..], 0..) |*e, i| e.* = @intCast(i + 1);
    } else {
        @memset(pair[rows..], @intCast(seq_kv));
    }
    return pair;
}

// --- the prefix pass --------------------------------------------------------

/// Device buffers for one prefix encode, sized to the prompt.
const PrefixWs = struct {
    x: Buf,
    nrm: Buf,
    dlt: Buf,
    q: Buf,
    k: Buf,
    v: Buf,
    attn: Buf,
    mg: Buf,
    mu: Buf,
    bounds: Buf,

    const fields = [_][]const u8{ "x", "nrm", "dlt", "q", "k", "v", "attn", "mg", "mu", "bounds" };

    fn init(ctx: *Context, cfg: Config, seq: usize) !PrefixWs {
        // Padded to 128 rows: a quantized route launches over `align(m, 128)` and each
        // block stores a whole tile, so a buffer sized to the exact row count is
        // written off the end.
        const rows = std.mem.alignForward(usize, seq, 128);
        const sizes = [fields.len]usize{
            rows * cfg.dim * 4,
            rows * cfg.dim * 4,
            rows * cfg.dim * 4,
            rows * cfg.qDim() * 4,
            rows * cfg.kvDim() * 4,
            rows * cfg.kvDim() * 4,
            rows * cfg.qDim() * 4,
            rows * cfg.inter * 4,
            rows * cfg.inter * 4,
            2 * seq * 4,
        };
        var self: PrefixWs = undefined;
        var made: usize = 0;
        errdefer inline for (fields, 0..) |name, i| {
            if (i < made) ctx.tensorDestroy(&@field(self, name));
        };
        inline for (fields, sizes) |name, size| {
            @field(self, name) = try ctx.tensorCreate(size);
            made += 1;
        }
        return self;
    }

    fn deinit(self: *PrefixWs, ctx: *Context) void {
        inline for (fields) |name| ctx.tensorDestroy(&@field(self, name));
        self.* = undefined;
    }
};

/// Run the prompt through the base MoT copy and return the prefix KV cache, the
/// conditioning this family's `encode` produces.
pub fn prefixForward(
    model: *const Model,
    ctx: *Context,
    io: std.Io,
    gpa: std.mem.Allocator,
    layout: sensenova.PromptLayout,
    refs: []const sensenova.RefImage,
    cancel: ?*std.atomic.Value(bool),
) !Prefix {
    const cfg = model.cfg;
    const seq = layout.ids.len;
    std.debug.assert(seq > 0);

    var pre: Prefix = .{
        .kv = try gpa.alloc(f32, cfg.n_layers * 2 * seq * cfg.kvDim()),
        .seq = seq,
        .kv_dim = cfg.kvDim(),
        .n_layers = cfg.n_layers,
        .time = layout.time,
    };
    errdefer pre.deinit(gpa);

    var ws = try PrefixWs.init(ctx, cfg, seq);
    defer ws.deinit(ctx);

    // The position arrays are `usize` on the host and `u32` on the device; the tables
    // are sized to the largest position each axis actually reaches, which is 1 row for
    // the hw axes of a prompt with no pictures in it.
    const pos32 = try gpa.alloc(u32, 3 * seq);
    defer gpa.free(pos32);
    var maxpos = [3]usize{ 0, 0, 0 };
    for ([3][]const usize{ layout.pos_t, layout.pos_h, layout.pos_w }, 0..) |src, axis| {
        for (src, 0..) |v, i| {
            pos32[axis * seq + i] = @intCast(v);
            maxpos[axis] = @max(maxpos[axis], v);
        }
    }

    var tab_t = try Table.init(ctx, gpa, maxpos[0] + 1, cfg.spanT(), cfg.theta_t);
    defer tab_t.deinit(ctx);
    var tab_h = try Table.init(ctx, gpa, maxpos[1] + 1, cfg.spanHw(), cfg.theta_hw);
    defer tab_h.deinit(ctx);
    var tab_w = try Table.init(ctx, gpa, maxpos[2] + 1, cfg.spanHw(), cfg.theta_hw);
    defer tab_w.deinit(ctx);
    var pos_bufs = [3]Buf{
        try uploadU32(ctx, pos32[0..seq]),
        try uploadU32(ctx, pos32[seq..][0..seq]),
        try uploadU32(ctx, pos32[2 * seq ..][0..seq]),
    };
    defer for (&pos_bufs) |*p| ctx.tensorDestroy(p);
    const tables = [3]Table{ tab_t, tab_h, tab_w };

    {
        const pair = try boundsTable(gpa, seq, seq, layout.kv_end);
        defer gpa.free(pair);
        try ctx.tensorUpload(ws.bounds, std.mem.sliceAsBytes(pair));
    }
    {
        // The understanding tower runs on the host: it is ~30M parameters against the
        // trunk's 8B, and it runs once per conditioning rather than per step.
        const emb = try model.embedPrompt(io, gpa, layout, refs);
        defer gpa.free(emb);
        try ctx.tensorUpload(ws.x, std.mem.sliceAsBytes(emb));
    }
    // Bound BEFORE the first batch, and once: the five buffers are the same at every
    // layer, and the descriptor set is persistent, so rewriting it mid-batch would
    // touch a set already referenced by recorded commands.
    ctx.attnBatchedBind(ws.q, ws.k, ws.v, ws.attn, ws.bounds);

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));
    for (model.layers, 0..) |*layer, li| {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        try ctx.beginBatch();
        errdefer if (ctx.batching) ctx.abortBatch();
        try layerForward(ctx, cfg, &layer.base, .{
            .x = ws.x,
            .nrm = ws.nrm,
            .dlt = ws.dlt,
            .q = ws.q,
            .k = ws.k,
            .v = ws.v,
            .kcat = ws.k,
            .vcat = ws.v,
            .attn = ws.attn,
            .mg = ws.mg,
            .mu = ws.mu,
            .q16 = ws.q,
            .k16 = ws.k,
            .v16 = ws.v,
        }, seq, seq, .{
            .pos = pos_bufs,
            .tables = tables,
            .norm_q = try smallBuf(ctx, layer.base.q_norm_pair),
            .norm_k = try smallBuf(ctx, layer.base.k_norm_pair),
            .attn = .{ .prefix_rows = 0, .scale = scale, .bounds = ws.bounds, .tc = false },
        });
        try ctx.endBatch();
        // Down per layer rather than at the end: the whole cache is 172 KB per token
        // per layer, and holding 42 layers of it on the device as well as on the host
        // would cost the generation stack the room it needs.
        try ctx.tensorDownloadAt(ws.k, 0, std.mem.sliceAsBytes(pre.kv[(li * 2) * seq * cfg.kvDim() ..][0 .. seq * cfg.kvDim()]));
        try ctx.tensorDownloadAt(ws.v, 0, std.mem.sliceAsBytes(pre.kv[(li * 2 + 1) * seq * cfg.kvDim() ..][0 .. seq * cfg.kvDim()]));
    }
    return pre;
}

// --- the generation pass ----------------------------------------------------

/// Per-image cache: everything constant across sampling steps.
pub const Session = struct {
    cfg: Config,
    /// Canvas extents as the caller asked for them, and as the model pads them.
    lat_h: usize,
    lat_w: usize,
    ph: usize,
    pw: usize,
    /// Patch-embedder cell grid, then the token grid.
    gh: usize,
    gw: usize,
    th: usize,
    tw: usize,
    n_img: usize,
    prefix_seq: usize,
    /// Whether the flash pipeline is in force. Read once, here, not per call: it
    /// decides whether `Workspace` sizes the f16 attention operands at all.
    tc: bool,

    prefix_kv: Buf,
    tab_t: Table,
    tab_h: Table,
    tab_w: Table,
    tab_vx: Table,
    tab_vy: Table,
    pos_t: Buf,
    pos_h: Buf,
    pos_w: Buf,
    vis_x: Buf,
    vis_y: Buf,

    pub fn init(
        gpa: std.mem.Allocator,
        ctx: *Context,
        model: *const Model,
        lat_h: usize,
        lat_w: usize,
        pre: Prefix,
    ) !Session {
        const cfg = model.cfg;
        const ph = sensenova.paddedExtent(cfg, lat_h);
        const pw = sensenova.paddedExtent(cfg, lat_w);
        const gh = ph / cfg.patch;
        const gw = pw / cfg.patch;
        const th = ph / cfg.tokenPx();
        const tw = pw / cfg.tokenPx();

        var self: Session = .{
            .cfg = cfg,
            .lat_h = lat_h,
            .lat_w = lat_w,
            .ph = ph,
            .pw = pw,
            .gh = gh,
            .gw = gw,
            .th = th,
            .tw = tw,
            .n_img = th * tw,
            .prefix_seq = pre.seq,
            .tc = !force_naive_attn and ctx.pipe_flash_md != .null_handle and ctx.pipe_flash_out != .null_handle,
            .prefix_kv = undefined,
            .tab_t = undefined,
            .tab_h = undefined,
            .tab_w = undefined,
            .tab_vx = undefined,
            .tab_vy = undefined,
            .pos_t = undefined,
            .pos_h = undefined,
            .pos_w = undefined,
            .vis_x = undefined,
            .vis_y = undefined,
        };
        var made: usize = 0;
        errdefer {
            if (made > 0) ctx.tensorDestroy(&self.prefix_kv);
            if (made > 1) self.tab_t.deinit(ctx);
            if (made > 2) self.tab_h.deinit(ctx);
            if (made > 3) self.tab_w.deinit(ctx);
            if (made > 4) self.tab_vx.deinit(ctx);
            if (made > 5) self.tab_vy.deinit(ctx);
            const pos = [_]*Buf{ &self.pos_t, &self.pos_h, &self.pos_w, &self.vis_x, &self.vis_y };
            for (pos[0..(made -| 6)]) |b| ctx.tensorDestroy(b);
        }

        self.prefix_kv = try ctx.tensorCreate(pre.kv.len * 4);
        made += 1;
        try ctx.tensorUpload(self.prefix_kv, std.mem.sliceAsBytes(pre.kv));

        // Every image token takes the same t index, one past the whole prefix.
        self.tab_t = try Table.init(ctx, gpa, pre.time + 1, cfg.spanT(), cfg.theta_t);
        made += 1;
        self.tab_h = try Table.init(ctx, gpa, th, cfg.spanHw(), cfg.theta_hw);
        made += 1;
        self.tab_w = try Table.init(ctx, gpa, tw, cfg.spanHw(), cfg.theta_hw);
        made += 1;
        self.tab_vx = try Table.initInterleaved(ctx, gpa, gw, cfg.vis_dim / 2, cfg.theta_vis);
        made += 1;
        self.tab_vy = try Table.initInterleaved(ctx, gpa, gh, cfg.vis_dim / 2, cfg.theta_vis);
        made += 1;

        {
            const n_img = self.n_img;
            const t = try gpa.alloc(u32, n_img);
            defer gpa.free(t);
            const h = try gpa.alloc(u32, n_img);
            defer gpa.free(h);
            const w = try gpa.alloc(u32, n_img);
            defer gpa.free(w);
            for (0..n_img) |i| {
                t[i] = pre.time;
                h[i] = @intCast(i / tw);
                w[i] = @intCast(i % tw);
            }
            self.pos_t = try uploadU32(ctx, t);
            made += 1;
            self.pos_h = try uploadU32(ctx, h);
            made += 1;
            self.pos_w = try uploadU32(ctx, w);
            made += 1;
        }
        {
            const x = try gpa.alloc(u32, gh * gw);
            defer gpa.free(x);
            const y = try gpa.alloc(u32, gh * gw);
            defer gpa.free(y);
            for (0..gh * gw) |i| {
                x[i] = @intCast(i % gw);
                y[i] = @intCast(i / gw);
            }
            self.vis_x = try uploadU32(ctx, x);
            made += 1;
            self.vis_y = try uploadU32(ctx, y);
            made += 1;
        }
        return self;
    }

    pub fn deinit(self: *Session, ctx: *Context) void {
        ctx.tensorDestroy(&self.prefix_kv);
        self.tab_t.deinit(ctx);
        self.tab_h.deinit(ctx);
        self.tab_w.deinit(ctx);
        self.tab_vx.deinit(ctx);
        self.tab_vy.deinit(ctx);
        inline for (.{ "pos_t", "pos_h", "pos_w", "vis_x", "vis_y" }) |name| ctx.tensorDestroy(&@field(self, name));
        self.* = undefined;
    }
};

pub const Workspace = struct {
    patch: Buf,
    cells: Buf,
    img: Buf,
    nrm: Buf,
    dlt: Buf,
    q: Buf,
    k: Buf,
    v: Buf,
    kcat: Buf,
    vcat: Buf,
    attn: Buf,
    mg: Buf,
    mu: Buf,
    fm_a: Buf,
    fm_b: Buf,
    tvec: Buf,
    /// f16 attention operands, sized only under `Session.tc`; the naive path never
    /// reads them.
    q16: Buf,
    k16: Buf,
    v16: Buf,
    /// Per-query key range for the naive path: `[0, seq_kv)` for every canvas row.
    bounds: Buf,

    const fields = [_][]const u8{ "patch", "cells", "img", "nrm", "dlt", "q", "k", "v", "kcat", "vcat", "attn", "mg", "mu", "fm_a", "fm_b", "tvec", "q16", "k16", "v16", "bounds" };

    /// `prefix_cap` is the LONGEST prefix any branch sharing this workspace has, not
    /// the positive branch's. Under classifier-free guidance the two branches are
    /// different lengths (an edit's negative presents the pictures with no prompt, so
    /// it is shorter; a plain negative is shorter still), and sizing to one of them
    /// writes the other off the end of `kcat`.
    pub fn init(ctx: *Context, model: *const Model, sess: *const Session, prefix_cap: usize) !Workspace {
        const cfg = model.cfg;
        const rows = std.mem.alignForward(usize, sess.n_img, 128);
        const kv_seq = @max(prefix_cap, sess.prefix_seq) + sess.n_img;
        const kv_rows = std.mem.alignForward(usize, kv_seq, 128);
        const cells = sess.gh * sess.gw;
        // The fm_head's two intermediates, at their widest: after the first shuffle it
        // is `[2*th][2*tw][dim/4]`, the same element count as the token plane, and
        // after the second `[4*th][4*tw][192]`.
        const fm_max = @max(cells * cfg.vis_dim, @max(sess.n_img * cfg.dim, sess.th * 4 * sess.tw * 4 * 192));
        // The flash tiles are 128 wide and its MD table is one pair per query row per
        // head, kept past the attention output in the same buffer.
        const q_pad = std.mem.alignForward(usize, sess.n_img, flash_tile);
        const kv_pad = std.mem.alignForward(usize, kv_seq, flash_tile);
        const attn_len = @max(rows * cfg.qDim(), q_pad * cfg.qDim() + cfg.n_heads * q_pad * 2);
        const sizes = [fields.len]usize{
            // Grown on demand by the two convolution helpers.
            4,
            cells * cfg.vis_dim * 4,
            rows * cfg.dim * 4,
            rows * cfg.dim * 4,
            rows * cfg.dim * 4,
            rows * cfg.qDim() * 4,
            rows * cfg.kvDim() * 4,
            rows * cfg.kvDim() * 4,
            kv_rows * cfg.kvDim() * 4,
            kv_rows * cfg.kvDim() * 4,
            attn_len * 4,
            rows * cfg.inter * 4,
            rows * cfg.inter * 4,
            fm_max * 4,
            fm_max * 4,
            cfg.dim * 4,
            if (sess.tc) q_pad * cfg.qDim() * 2 else 4,
            if (sess.tc) kv_pad * cfg.kvDim() * 2 else 4,
            if (sess.tc) kv_pad * cfg.kvDim() * 2 else 4,
            2 * sess.n_img * 4,
        };
        var self: Workspace = undefined;
        var made: usize = 0;
        errdefer inline for (fields, 0..) |name, i| {
            if (i < made) ctx.tensorDestroy(&@field(self, name));
        };
        inline for (fields, sizes) |name, size| {
            @field(self, name) = try ctx.tensorCreate(size);
            made += 1;
        }
        return self;
    }

    pub fn deinit(self: *Workspace, ctx: *Context) void {
        inline for (fields) |name| ctx.tensorDestroy(&@field(self, name));
        self.* = undefined;
    }
};

/// One denoiser forward. `out` and `x_lat` are planar `[3][lat_h][lat_w]`; `t` is the
/// model's own timestep, `1 - sigma`.
pub fn forward(
    model: *const Model,
    ctx: *Context,
    sess: *Session,
    ws: *Workspace,
    io: std.Io,
    gpa: std.mem.Allocator,
    out: []f32,
    x_lat: []const f32,
    t: f32,
    cancel: ?*std.atomic.Value(bool),
) !void {
    const cfg = model.cfg;
    const ch = sensenova.latent_channels;
    std.debug.assert(x_lat.len == ch * sess.lat_h * sess.lat_w);
    std.debug.assert(out.len == x_lat.len);

    const xp = try sensenova.padCanvas(gpa, cfg, x_lat, ch, sess.lat_h, sess.lat_w);
    defer gpa.free(xp);

    // The patch matrix of the 16x16 embedder is built on the host: it reads the
    // canvas, which arrives on the host anyway, and it is a pure gather that would
    // otherwise cost an upload of the canvas plus a kernel for the same bytes.
    {
        const plen = cfg.patch * cfg.patch * ch;
        const patch = try gpa.alloc(f32, sess.gh * sess.gw * plen);
        defer gpa.free(patch);
        for (0..sess.gh) |cy| {
            for (0..sess.gw) |cx| {
                const dst = patch[(cy * sess.gw + cx) * plen ..][0..plen];
                for (0..cfg.patch) |dy| {
                    for (0..cfg.patch) |dx| {
                        const sy = cy * cfg.patch + dy;
                        const sx = cx * cfg.patch + dx;
                        for (0..ch) |c| {
                            dst[(dy * cfg.patch + dx) * ch + c] = xp[c * sess.ph * sess.pw + sy * sess.pw + sx];
                        }
                    }
                }
            }
        }
        try ctx.ensureDeviceBuffer(&ws.patch, patch.len * 4);
        try ctx.tensorUpload(ws.patch, std.mem.sliceAsBytes(patch));
    }
    // The scalar conditioning is two 256-wide MLPs; the host runs them in microseconds
    // and the device would need an upload either way.
    {
        const tvec = try model.timeVector(io, gpa, t, sess.ph, sess.pw);
        defer gpa.free(tvec);
        try ctx.tensorUpload(ws.tvec, std.mem.sliceAsBytes(tvec));
    }
    const seq_kv = sess.prefix_seq + sess.n_img;
    if (!sess.tc) {
        const pair = try boundsTable(gpa, sess.n_img, seq_kv, null);
        defer gpa.free(pair);
        try ctx.tensorUpload(ws.bounds, std.mem.sliceAsBytes(pair));
        ctx.attnBatchedBind(ws.q, ws.kcat, ws.vcat, ws.attn, ws.bounds);
    }

    try ctx.beginBatch();
    errdefer if (ctx.batching) ctx.abortBatch();

    // --- the generation vision tower -----------------------------------------
    const cells = sess.gh * sess.gw;
    try ctx.opMatmulCoopF16W(ws.cells, 0, ws.patch, cells, model.vision_gen.patch.w, cfg.vis_dim, cfg.patch * cfg.patch * ch, model.vision_gen.patch.b.?);
    try ctx.opElt(.gelu_erf, ws.cells, null, null, null, .{ .u0 = @intCast(cells * cfg.vis_dim) }, cells * cfg.vis_dim, 1, 1);
    const vhalf = cfg.vis_dim / 2;
    try ropeInterSpan(ctx, ws.cells, sess.vis_x, sess.tab_vx, cells, cfg.vis_dim, 0);
    try ropeInterSpan(ctx, ws.cells, sess.vis_y, sess.tab_vy, cells, cfg.vis_dim, vhalf);
    trace(ctx, "cells", 0, ws.cells, cells * cfg.vis_dim);
    try convStride(ctx, &ws.patch, ws.img, ws.cells, sess.gh, sess.gw, model.vision_gen.dense);
    trace(ctx, "vision", 0, ws.img, sess.n_img * cfg.dim);
    try ctx.opElt(.add_bias_rows, ws.img, ws.tvec, null, null, .{
        .u0 = @intCast(sess.n_img * cfg.dim),
        .u1 = @intCast(cfg.dim),
    }, sess.n_img * cfg.dim, 1, 1);
    trace(ctx, "trunk_in", 0, ws.img, sess.n_img * cfg.dim);

    // --- the trunk ------------------------------------------------------------
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));
    const pos = [3]Buf{ sess.pos_t, sess.pos_h, sess.pos_w };
    const tables = [3]Table{ sess.tab_t, sess.tab_h, sess.tab_w };
    const kv_elems = sess.prefix_seq * cfg.kvDim();
    for (model.layers, 0..) |*layer, li| {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        // The prefix half of the concatenated K/V, from the cache uploaded once per
        // conditioning. Through the copy KERNEL, so it stays in this batch.
        try copyInto(ctx, ws.kcat, 0, sess.prefix_kv, (li * 2) * kv_elems, kv_elems);
        try copyInto(ctx, ws.vcat, 0, sess.prefix_kv, (li * 2 + 1) * kv_elems, kv_elems);
        try layerForward(ctx, cfg, &layer.gen, .{
            .x = ws.img,
            .nrm = ws.nrm,
            .dlt = ws.dlt,
            .q = ws.q,
            .k = ws.k,
            .v = ws.v,
            .kcat = ws.kcat,
            .vcat = ws.vcat,
            .attn = ws.attn,
            .mg = ws.mg,
            .mu = ws.mu,
            .q16 = ws.q16,
            .k16 = ws.k16,
            .v16 = ws.v16,
        }, sess.n_img, seq_kv, .{
            .pos = pos,
            .tables = tables,
            .norm_q = try smallBuf(ctx, layer.gen.q_norm_pair),
            .norm_k = try smallBuf(ctx, layer.gen.k_norm_pair),
            .attn = .{ .prefix_rows = sess.prefix_seq, .scale = scale, .bounds = null, .tc = sess.tc },
        });
        trace(ctx, "layer", li, ws.img, sess.n_img * cfg.dim);
    }
    try ctx.opElt(.rmsnorm, ws.img, ws.img, try smallBuf(ctx, model.norm_gen), null, .{
        .u0 = @intCast(sess.n_img),
        .u1 = @intCast(cfg.dim),
        .f0 = cfg.norm_eps,
    }, sess.n_img, 1, 1);
    trace(ctx, "norm", 0, ws.img, sess.n_img * cfg.dim);

    // --- fm_head ---------------------------------------------------------------
    // The token plane IS the channel-last input the first shuffle reads.
    try pixelShuffle(ctx, ws.fm_a, ws.img, sess.th, sess.tw, cfg.fmC1(), 2);
    try conv3x3(ctx, &ws.patch, &ws.fm_b, &ws.fm_a, sess.th * 2, sess.tw * 2, model.fm1);
    try ctx.opElt(.gelu_erf, ws.fm_b, null, null, null, .{
        .u0 = @intCast(sess.th * 2 * sess.tw * 2 * cfg.fmC1()),
    }, sess.th * 2 * sess.tw * 2 * cfg.fmC1(), 1, 1);
    try pixelShuffle(ctx, ws.fm_a, ws.fm_b, sess.th * 2, sess.tw * 2, cfg.fmC2(), 2);
    try conv3x3(ctx, &ws.patch, &ws.fm_b, &ws.fm_a, sess.th * 4, sess.tw * 4, model.fm2);
    try pixelShuffle(ctx, ws.fm_a, ws.fm_b, sess.th * 4, sess.tw * 4, ch, 8);
    trace(ctx, "pred", 0, ws.fm_a, sess.ph * sess.pw * ch);
    try ctx.endBatch();

    // The x0-to-velocity conversion and the crop are three planes of arithmetic; they
    // ride back with the download rather than costing a kernel.
    const rgb = try gpa.alloc(f32, sess.ph * sess.pw * ch);
    defer gpa.free(rgb);
    try ctx.tensorDownloadAt(ws.fm_a, 0, std.mem.sliceAsBytes(rgb));
    const denom = @max(1.0 - t, cfg.velocity_eps);
    for (0..ch) |c| {
        const src = xp[c * sess.ph * sess.pw ..];
        const dst = out[c * sess.lat_h * sess.lat_w ..];
        for (0..sess.lat_h) |y| {
            for (0..sess.lat_w) |x| {
                dst[y * sess.lat_w + x] = (src[y * sess.pw + x] - rgb[(y * sess.pw + x) * ch + c]) / denom;
            }
        }
    }
}

fn ropeInterSpan(ctx: *Context, x: Buf, positions: Buf, tb: Table, rows: usize, row_dim: usize, off: usize) !void {
    std.debug.assert(off + 2 * tb.half <= row_dim);
    const total = rows * tb.half;
    try ctx.opElt(.rope_inter_span_pos, x, positions, tb.buf, null, .{
        .u0 = @intCast(total),
        .u1 = @intCast(tb.half),
        .u2 = @intCast(tb.sin_off),
        .u3 = @intCast(row_dim),
        .u4 = @intCast(off),
    }, total, 1, 1);
}

fn pixelShuffle(ctx: *Context, dst: Buf, src: Buf, h: usize, w: usize, co: usize, r: usize) !void {
    const total = h * w * co * r * r;
    try ctx.opElt(.pixel_shuffle, src, dst, null, null, .{
        .u0 = @intCast(total),
        .u1 = @intCast(co),
        .u2 = @intCast(r),
        .u3 = @intCast(w),
    }, total, 1, 1);
}

const LayerBufs = struct {
    x: Buf,
    nrm: Buf,
    dlt: Buf,
    q: Buf,
    k: Buf,
    v: Buf,
    kcat: Buf,
    vcat: Buf,
    attn: Buf,
    mg: Buf,
    mu: Buf,
    q16: Buf,
    k16: Buf,
    v16: Buf,
};

const LayerArgs = struct {
    pos: [3]Buf,
    tables: [3]Table,
    norm_q: Buf,
    norm_k: Buf,
    attn: AttnArgs,
};

fn layerForward(
    ctx: *Context,
    cfg: Config,
    s: *const sensenova.Stream,
    b: LayerBufs,
    rows: usize,
    seq_kv: usize,
    a: LayerArgs,
) !void {
    const dim = cfg.dim;
    const kv_dim = cfg.kvDim();

    try ctx.opElt(.rmsnorm, b.x, b.nrm, try smallBuf(ctx, s.in_norm), null, .{
        .u0 = @intCast(rows),
        .u1 = @intCast(dim),
        .f0 = cfg.norm_eps,
    }, rows, 1, 1);
    try prepGroup(ctx, b.nrm, rows, dim, &.{ s.q, s.k, s.v });
    try linear(ctx, b.q, b.nrm, rows, s.q);
    try linear(ctx, b.k, b.nrm, rows, s.k);
    try linear(ctx, b.v, b.nrm, rows, s.v);

    try qkPrepare(ctx, cfg, b.q, rows, cfg.n_heads, a.norm_q, a.pos, a.tables);
    try qkPrepare(ctx, cfg, b.k, rows, cfg.n_kv_heads, a.norm_k, a.pos, a.tables);
    trace(ctx, "q", 0, b.q, rows * cfg.qDim());
    trace(ctx, "k", 0, b.k, rows * kv_dim);
    trace(ctx, "v", 0, b.v, rows * kv_dim);

    // The generation stream attends over the prefix's keys and then its own; the
    // prefix pass attends over itself alone, so its `kcat` IS its `k`.
    if (a.attn.prefix_rows != 0) {
        try copyInto(ctx, b.kcat, a.attn.prefix_rows * kv_dim, b.k, 0, rows * kv_dim);
        try copyInto(ctx, b.vcat, a.attn.prefix_rows * kv_dim, b.v, 0, rows * kv_dim);
    }
    try attention(ctx, cfg, b, rows, seq_kv, a.attn);

    trace(ctx, "attn", 0, b.attn, rows * cfg.qDim());
    try prepGroup(ctx, b.attn, rows, cfg.qDim(), &.{s.o});
    try linear(ctx, b.dlt, b.attn, rows, s.o);
    try ctx.opElt(.add, b.x, b.dlt, null, null, .{ .u0 = @intCast(rows * dim) }, rows * dim, 1, 1);
    trace(ctx, "attn_res", 0, b.x, rows * dim);

    try ctx.opElt(.rmsnorm, b.x, b.nrm, try smallBuf(ctx, s.post_norm), null, .{
        .u0 = @intCast(rows),
        .u1 = @intCast(dim),
        .f0 = cfg.norm_eps,
    }, rows, 1, 1);
    try prepGroup(ctx, b.nrm, rows, dim, &.{ s.gate, s.up });
    try linear(ctx, b.mg, b.nrm, rows, s.gate);
    try linear(ctx, b.mu, b.nrm, rows, s.up);
    try ctx.opElt(.silu_mul, b.mg, b.mu, null, null, .{ .u0 = @intCast(rows * cfg.inter) }, rows * cfg.inter, 1, 1);
    try prepGroup(ctx, b.mg, rows, cfg.inter, &.{s.down});
    try linear(ctx, b.dlt, b.mg, rows, s.down);
    try ctx.opElt(.add, b.x, b.dlt, null, null, .{ .u0 = @intCast(rows * dim) }, rows * dim, 1, 1);
}

/// DIAGNOSTIC: the same stage lines `sensenova.traceActs` prints on the host
/// (`TP_SENSENOVA_TRACE`), so the two runs diff. Costs a sync and a download per call,
/// so it is env-gated and never on in a render.
var trace_on: ?bool = null;

fn trace(ctx: *Context, what: []const u8, i: usize, buf: Buf, elems: usize) void {
    if (trace_on == null) trace_on = std.c.getenv("TP_SENSENOVA_TRACE") != null;
    if (!trace_on.?) return;
    const host = ctx.gpa.alloc(f32, elems) catch return;
    defer ctx.gpa.free(host);
    const batching = ctx.batching;
    if (batching) ctx.endBatch() catch return;
    ctx.tensorDownloadAt(buf, 0, std.mem.sliceAsBytes(host)) catch return;
    var mx: f32 = 0;
    var sum: f64 = 0;
    var bad: usize = 0;
    for (host) |v| {
        if (!std.math.isFinite(v)) bad += 1 else mx = @max(mx, @abs(v));
        sum += @as(f64, v) * @as(f64, v);
    }
    std.debug.print("[sensenova-vk ] {s:<10} {d:>2}  max|x| {d:12.4}  rms {d:10.5}  n {d}  nonfinite {d}\n", .{ what, i, mx, @sqrt(sum / @as(f64, @floatFromInt(elems))), elems, bad });
    if (batching) ctx.beginBatch() catch return;
}
