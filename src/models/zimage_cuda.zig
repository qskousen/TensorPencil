//! GPU-resident Z-Image (`NextDiT`) forward on the CUDA backends (`zig-cuda`'s hand-PTX
//! and `cuda`'s vendor libraries), the CUDA twin of `zimage_gpu`.
//!
//! The host/device split is identical to the Vulkan arm's; see `zimage_gpu`'s header.
//! Two differences, both forced by the op surface:
//!
//! 1. The modulation table is the FOLDED one (`zimage.modulationTableFolded`). CUDA has
//!    no standalone `modulate`, only `rmsMod` (`out = x*inv*premul[col] + shift[col]`),
//!    so the pre-norm weight must arrive already multiplied into the scale. Vulkan takes
//!    the unfused table because it has a separate `modulate`. Both come out of one AdaLN
//!    evaluation, so they cannot drift.
//! 2. Attention is `opAttnTC`: cuDNN's fused SDPA under `cuda`, hand-PTX tensor cores
//!    under `zig-cuda`. The naive `be.attn` is one thread per (query, head) and trips the
//!    GPU watchdog at production resolutions, so it is the fallback, not the default.
//!
//! Head width is exactly 128, which both `launchHgemmB`'s P@V tiling and cuDNN handle
//! directly, so unlike the SD family there is no head padding and no `head_pad` /
//! `head_unpad` round trip. Every trunk GEMM width (3840 / 10240 / 11520) is a multiple
//! of 128 and every reduction width a multiple of 32.

const std = @import("std");
const zimage = @import("zimage.zig");
const cuda = @import("tp_gpu").cuda;
const ops = @import("tp_ops");

const DiT = zimage.DiT;
const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;
const Weight = ops.matmul.Weight;

/// Force the naive one-thread-per-(query,head) attention instead of `opAttnTC`. For
/// A/B and for reproducing a mismatch; the device test runs both.
pub var force_naive_attn: bool = false;

/// Widest zero bias any Z-Image GEMM needs. Passed WHOLE, never sliced: a bias is
/// cached by host pointer and sized from the first call's length, so a narrow layer
/// seen first would leave every wider one reading past the end. `opGemmBf16` asserts
/// `bias.len >= co` and the kernels read only `co` entries, so one full-width buffer
/// serves every GEMM.
const zero_bias: [zimage.z_image.mlp_dim]f32 = @splat(0);

/// Per-image cache: everything constant across sampling steps.
pub const Session = struct {
    cfg: zimage.Config,
    lat_h: usize,
    lat_w: usize,
    cap_padded: usize,
    n_img: usize,
    img_padded: usize,
    /// `cap_padded + img_padded`, the joint sequence the trunk runs on.
    seq: usize,

    cap_d: Buf,
    x_pad_d: Buf,
    /// Interleaved-RoPE table for the joint sequence, `cos` then `sin`.
    freqs_d: Buf,
    /// The image half's own table, a separate upload, because the `noise_refiner`
    /// runs on the image tokens alone and needs row 0 of its table to be the image's
    /// first position, not the caption's.
    img_freqs_d: Buf,

    sigmas: []f32,
    /// `[sigmas.len][modulatedBlocks * 4 * dim + dim]`, FOLDED (see the header).
    mods: []f32,
    finals: []f32,
    mod_stride: usize,

    pub fn init(
        gpa: std.mem.Allocator,
        io: std.Io,
        be: *Backend,
        model: *const DiT,
        lat_h: usize,
        lat_w: usize,
        cap: []const f32,
        cap_padded: usize,
        sigmas: []const f32,
    ) !Session {
        const cfg = model.cfg;
        std.debug.assert(cap.len == cap_padded * cfg.dim);
        const h = lat_h / cfg.patch;
        const w = lat_w / cfg.patch;
        const n_img = h * w;
        const img_padded = cfg.padded(n_img);
        const seq = cap_padded + img_padded;
        const half = cfg.head_dim / 2;

        var self: Session = .{
            .cfg = cfg,
            .lat_h = lat_h,
            .lat_w = lat_w,
            .cap_padded = cap_padded,
            .n_img = n_img,
            .img_padded = img_padded,
            .seq = seq,
            .cap_d = undefined,
            .x_pad_d = undefined,
            .freqs_d = undefined,
            .img_freqs_d = undefined,
            .sigmas = &.{},
            .mods = &.{},
            .finals = &.{},
            .mod_stride = model.modulatedBlocks() * 4 * cfg.dim + cfg.dim,
        };
        var made: usize = 0;
        errdefer {
            const bufs = [_]*Buf{ &self.cap_d, &self.x_pad_d, &self.freqs_d, &self.img_freqs_d };
            for (bufs[0..made]) |b| be.tensorDestroy(b);
            if (self.sigmas.len != 0) gpa.free(self.sigmas);
            if (self.mods.len != 0) gpa.free(self.mods);
            if (self.finals.len != 0) gpa.free(self.finals);
        }

        self.cap_d = try be.tensorCreate(cap.len * 4);
        made += 1;
        try be.tensorUpload(self.cap_d, std.mem.sliceAsBytes(cap));
        self.x_pad_d = try be.tensorCreate(cfg.dim * 4);
        made += 1;
        try be.tensorUpload(self.x_pad_d, std.mem.sliceAsBytes(model.x_pad_token));

        {
            var joint = try model.ropeFreqs(gpa, cap_padded, h, w);
            defer joint.deinit(gpa);
            const flat = try gpa.alloc(f32, 2 * seq * half);
            defer gpa.free(flat);
            @memcpy(flat[0 .. seq * half], joint.cos);
            @memcpy(flat[seq * half ..], joint.sin);
            self.freqs_d = try be.tensorCreate(flat.len * 4);
            made += 1;
            try be.tensorUpload(self.freqs_d, std.mem.sliceAsBytes(flat));

            const iflat = try gpa.alloc(f32, 2 * img_padded * half);
            defer gpa.free(iflat);
            @memcpy(iflat[0 .. img_padded * half], joint.cos[cap_padded * half ..]);
            @memcpy(iflat[img_padded * half ..], joint.sin[cap_padded * half ..]);
            self.img_freqs_d = try be.tensorCreate(iflat.len * 4);
            made += 1;
            try be.tensorUpload(self.img_freqs_d, std.mem.sliceAsBytes(iflat));
        }

        self.sigmas = try gpa.dupe(f32, sigmas);
        self.mods = try gpa.alloc(f32, sigmas.len * self.mod_stride);
        self.finals = try gpa.alloc(f32, sigmas.len * cfg.dim);
        for (sigmas, 0..) |s, i| {
            const adaln = try model.adalnInput(io, gpa, s);
            defer gpa.free(adaln);
            const tbl = try model.modulationTableFolded(io, gpa, adaln);
            defer gpa.free(tbl);
            @memcpy(self.mods[i * self.mod_stride ..][0..self.mod_stride], tbl);
            const fin = try model.finalScale(io, gpa, adaln);
            defer gpa.free(fin);
            @memcpy(self.finals[i * cfg.dim ..][0..cfg.dim], fin);
        }
        return self;
    }

    pub fn deinit(self: *Session, gpa: std.mem.Allocator, be: *Backend) void {
        be.tensorDestroy(&self.cap_d);
        be.tensorDestroy(&self.x_pad_d);
        be.tensorDestroy(&self.freqs_d);
        be.tensorDestroy(&self.img_freqs_d);
        gpa.free(self.sigmas);
        gpa.free(self.mods);
        gpa.free(self.finals);
        self.* = undefined;
    }

    fn tv(self: *const Session, sigma: f32) ?StepMod {
        for (self.sigmas, 0..) |s, i| {
            if (s == sigma) return .{
                .mods = self.mods[i * self.mod_stride ..][0..self.mod_stride],
                .final = self.finals[i * self.cfg.dim ..][0..self.cfg.dim],
            };
        }
        return null;
    }
};

const StepMod = struct { mods: []const f32, final: []const f32 };

pub const Workspace = struct {
    x_d: Buf,
    img_d: Buf,
    nrm_d: Buf,
    dlt_d: Buf,
    q_d: Buf,
    k_d: Buf,
    v_d: Buf,
    attn_d: Buf,
    mg_d: Buf,
    mu_d: Buf,
    mv_d: Buf,
    imgin_d: Buf,

    const fields = [_][]const u8{ "x_d", "img_d", "nrm_d", "dlt_d", "q_d", "k_d", "v_d", "attn_d", "mg_d", "mu_d", "mv_d", "imgin_d" };

    pub fn init(be: *Backend, model: *const DiT, lat_h: usize, lat_w: usize, cap_padded_cap: usize) !Workspace {
        const cfg = model.cfg;
        const n_img = (lat_h / cfg.patch) * (lat_w / cfg.patch);
        const img_padded = cfg.padded(n_img);
        const seq = cap_padded_cap + img_padded;
        const sizes = [fields.len]usize{
            seq * cfg.dim * 4,
            img_padded * cfg.dim * 4,
            seq * cfg.dim * 4,
            seq * cfg.dim * 4,
            seq * cfg.qDim() * 4,
            seq * cfg.kvDim() * 4,
            seq * cfg.kvDim() * 4,
            seq * cfg.qDim() * 4,
            seq * cfg.mlp_dim * 4,
            seq * cfg.mlp_dim * 4,
            (model.modulatedBlocks() * 4 * cfg.dim + cfg.dim) * 4,
            n_img * cfg.patchDim() * 4,
        };
        var self: Workspace = undefined;
        var made: usize = 0;
        errdefer inline for (fields, 0..) |name, i| {
            if (i < made) be.tensorDestroy(&@field(self, name));
        };
        inline for (fields, sizes) |name, size| {
            @field(self, name) = try be.tensorCreate(size);
            made += 1;
        }
        return self;
    }

    pub fn deinit(self: *Workspace, be: *Backend) void {
        inline for (fields) |name| be.tensorDestroy(&@field(self, name));
        self.* = undefined;
    }
};

/// Whether this backend can run Z-Image's block GEMMs. The trunk weights are dense
/// bf16 in every checkpoint seen so far; fp8 and f32 also have paths. Anything else
/// (int8/int4 convrot, ggml block quants) has no CUDA GEMM here and must stay on the
/// CPU rather than being read as the wrong dtype.
pub fn supported(model: *const DiT) bool {
    if (model.layers.len == 0) return false;
    const always = struct {
        fn f(_: @import("tp_core").dtype.DType) bool {
            return true;
        }
    }.f;
    if (zimage.unsupportedGpuLin(model, always)) |bad| {
        std.log.warn("zimage_cuda: {s} is {t}, which this backend has no GEMM for — the trunk " ++
            "runs on the CPU. Expect CPU sampling speed.", .{ bad.tag, bad.dtype });
        return false;
    }
    return true;
}

/// One denoiser forward. `out`/`x_lat` are planar `[channels][lat_h][lat_w]`.
pub fn forward(
    model: *const DiT,
    be: *Backend,
    sess: *Session,
    ws: *Workspace,
    io: std.Io,
    gpa: std.mem.Allocator,
    out: []f32,
    x_lat: []const f32,
    sigma: f32,
    cancel: ?*std.atomic.Value(bool),
) !void {
    const cfg = model.cfg;
    const dim = cfg.dim;
    const seq = sess.seq;
    const half = cfg.head_dim / 2;
    const attn_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));
    const zero_off = model.zeroShiftOffset();

    std.debug.assert(x_lat.len == cfg.channels * sess.lat_h * sess.lat_w);
    std.debug.assert(out.len == x_lat.len);

    var owned_mods: ?[]f32 = null;
    var owned_final: ?[]f32 = null;
    defer if (owned_mods) |b| gpa.free(b);
    defer if (owned_final) |b| gpa.free(b);
    const step = sess.tv(sigma) orelse blk: {
        const adaln = try model.adalnInput(io, gpa, sigma);
        defer gpa.free(adaln);
        owned_mods = try model.modulationTableFolded(io, gpa, adaln);
        owned_final = try model.finalScale(io, gpa, adaln);
        break :blk StepMod{ .mods = owned_mods.?, .final = owned_final.? };
    };

    const patches = try zimage.patchify(gpa, cfg, x_lat, sess.lat_h, sess.lat_w);
    defer gpa.free(patches);

    try be.tensorUpload(ws.mv_d, std.mem.sliceAsBytes(step.mods));
    try be.tensorUpload(ws.imgin_d, std.mem.sliceAsBytes(patches));

    try be.beginBatch();
    errdefer if (be.batching()) be.abortBatch();

    // --- the image half: x_embedder, pad, then the noise_refiner stack ---------
    // `x_embedder` is the one weight the loader materializes to f32 (it feeds the
    // fused `opMatmul`, which has no bf16 pipeline), so it takes the f32 arm.
    try be.opMatmul(ws.img_d, 0, ws.imgin_d, 0, sess.n_img, model.x_embedder.w.bytes, false, dim, cfg.patchDim(), model.x_embedder.w.scale, model.x_embedder.b);
    for (sess.n_img..sess.img_padded) |r| {
        try be.tensorCopy(ws.img_d, r * dim * 4, sess.x_pad_d, 0, dim * 4);
    }
    // Prefetch one block ahead throughout, so each block's upload overlaps the
    // previous block's compute. The two stacks are consecutive, so the last
    // `noise_refiner` iteration primes `layers[0]`.
    if (be.async_uploads and model.noise_refiner.len > 0) prefetchBlock(be, cfg, model.noise_refiner[0]);
    for (model.noise_refiner, 0..) |*blk, i| {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        if (be.async_uploads) {
            if (i + 1 < model.noise_refiner.len)
                prefetchBlock(be, cfg, model.noise_refiner[i + 1])
            else if (model.layers.len > 0)
                prefetchBlock(be, cfg, model.layers[0]);
        }
        try blockForward(be, cfg, blk, ws, ws.img_d, sess.img_padded, sess.img_freqs_d, sess.img_padded * half, i * 4 * dim, zero_off, attn_scale);
    }

    // --- the joint sequence ----------------------------------------------------
    try be.tensorCopy(ws.x_d, 0, sess.cap_d, 0, sess.cap_padded * dim * 4);
    try be.tensorCopy(ws.x_d, sess.cap_padded * dim * 4, ws.img_d, 0, sess.img_padded * dim * 4);

    for (model.layers, 0..) |*blk, i| {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        if (be.async_uploads and i + 1 < model.layers.len) prefetchBlock(be, cfg, model.layers[i + 1]);
        const base = (model.noise_refiner.len + i) * 4 * dim;
        try blockForward(be, cfg, blk, ws, ws.x_d, seq, sess.freqs_d, seq * half, base, zero_off, attn_scale);
    }
    try be.endBatch();

    // --- the final layer, on the host ------------------------------------------
    const img_rows = try gpa.alloc(f32, sess.n_img * dim);
    defer gpa.free(img_rows);
    // No `tensorDownloadAt` on this backend; take an offset view, as `dit_cuda` does.
    try be.tensorDownload(offsetBuf(ws.x_d, sess.cap_padded * dim * 4, img_rows.len * 4), std.mem.sliceAsBytes(img_rows));

    ops.norm.layerNormUnit(img_rows, img_rows, dim, cfg.final_eps);
    var row: usize = 0;
    while (row < img_rows.len) : (row += dim) {
        for (img_rows[row..][0..dim], step.final) |*v, sc| v.* = (1.0 + sc) * v.*;
    }
    const final = try gpa.alloc(f32, sess.n_img * cfg.patchDim());
    defer gpa.free(final);
    try ops.matmul.matmul(io, gpa, final, img_rows, sess.n_img, model.final_linear.w, model.final_linear.b);
    zimage.unpatchify(cfg, out, final, sess.lat_h, sess.lat_w);
}

fn blockForward(
    be: *Backend,
    cfg: zimage.Config,
    blk: anytype,
    ws: *Workspace,
    x: Buf,
    rows: usize,
    freqs: Buf,
    sin_off: usize,
    mod_base: usize,
    zero_off: usize,
    attn_scale: f32,
) !void {
    const dim = cfg.dim;
    const heads = cfg.n_heads;
    const kv_heads = cfg.n_kv_heads;
    const hd = cfg.head_dim;
    const half = hd / 2;

    // x += tanh(gate_msa) * norm2(attn(rmsMod(x, premul_attn)))
    // `premul` is `attn_norm1 * (1 + scale_msa)`, folded on the host, because
    // `rmsMod` has no place for a separate norm weight. `shift_off` points at the
    // table's trailing zero block: Z-Image's modulation has no shift.
    try be.rmsMod(x, ws.nrm_d, ws.mv_d, rows, dim, mod_base + 0 * dim, zero_off, cfg.norm_eps);

    var nvq: ops.nvfp4.Meta = undefined;
    var nvk: ops.nvfp4.Meta = undefined;
    var nvv: ops.nvfp4.Meta = undefined;
    const wq = qkvPart(blk.attn.qkv, 0, cfg.qDim(), &nvq);
    const wk = qkvPart(blk.attn.qkv, cfg.qDim(), cfg.kvDim(), &nvk);
    const wv = qkvPart(blk.attn.qkv, cfg.qDim() + cfg.kvDim(), cfg.kvDim(), &nvv);
    try gemm(be, ws.q_d, ws.nrm_d, rows, wq);
    try gemm(be, ws.k_d, ws.nrm_d, rows, wk);
    try gemm(be, ws.v_d, ws.nrm_d, rows, wv);

    // The Q/K norms take `finfo(f32).eps`, NOT the blocks' 1e-5.
    try be.qkNorm(ws.q_d, ws.q_d, try normBuf(be, blk.attn.qnorm), rows * heads, hd, cfg.qk_eps);
    try be.qkNorm(ws.k_d, ws.k_d, try normBuf(be, blk.attn.knorm), rows * kv_heads, hd, cfg.qk_eps);
    try be.rope(ws.q_d, freqs, rows, heads, half, sin_off);
    try be.rope(ws.k_d, freqs, rows, kv_heads, half, sin_off);

    if (force_naive_attn) {
        try be.attn(ws.q_d, ws.k_d, ws.v_d, ws.attn_d, rows, rows, heads, kv_heads, hd, attn_scale, false);
    } else {
        try be.opAttnTC(ws.q_d, ws.k_d, ws.v_d, ws.attn_d, rows, heads, kv_heads, hd, attn_scale);
    }
    try gemm(be, ws.dlt_d, ws.attn_d, rows, blk.attn.out);
    // The sandwich norm: a SECOND RMSNorm on the sublayer's output, inside the
    // residual. `qkNorm` is a plain weighted RMSNorm over rows of `dim`.
    try be.qkNorm(ws.dlt_d, ws.dlt_d, try normBuf(be, blk.attn_norm2), rows, dim, cfg.norm_eps);
    try be.gatedAdd(x, ws.dlt_d, ws.mv_d, rows * dim, dim, mod_base + 1 * dim);

    // x += tanh(gate_mlp) * norm2(swiglu(rmsMod(x, premul_ffn)))
    try be.rmsMod(x, ws.nrm_d, ws.mv_d, rows, dim, mod_base + 2 * dim, zero_off, cfg.norm_eps);
    try gemm(be, ws.mg_d, ws.nrm_d, rows, blk.ffn.w1);
    try gemm(be, ws.mu_d, ws.nrm_d, rows, blk.ffn.w3);
    try be.siluMul(ws.mg_d, ws.mu_d, rows * cfg.mlp_dim);
    try gemm(be, ws.dlt_d, ws.mg_d, rows, blk.ffn.w2);
    try be.qkNorm(ws.dlt_d, ws.dlt_d, try normBuf(be, blk.ffn_norm2), rows, dim, cfg.norm_eps);
    try be.gatedAdd(x, ws.dlt_d, ws.mv_d, rows * dim, dim, mod_base + 3 * dim);
}

/// Queue a block's streamable weights for async prefetch, called ONE BLOCK AHEAD so
/// the upload overlaps the previous block's compute. Keys must be the same host
/// pointers `forward` later fetches, or the prefetch is a cache miss and pure waste,
/// hence the qkv ROW VIEWS here, matching `qkvPart` exactly.
///
/// Without this the first step pays the whole ~11.6 GB upload serially: measured
/// 8.0 s for step 1 against a 2.6 s steady state at 1056x1584, which on a 9-step
/// turbo render is a fifth of the total time.
fn prefetchBlock(be: *Backend, cfg: zimage.Config, blk: anytype) void {
    const bytes = std.mem.sliceAsBytes;
    const qkv = blk.attn.qkv;
    inline for (.{ 0, 1, 2 }) |i| {
        // Prefetch only touches `.bytes`, so the sliced metadata is unused here.
        var nv: ops.nvfp4.Meta = undefined;
        const part = switch (i) {
            0 => qkvPart(qkv, 0, cfg.qDim(), &nv),
            1 => qkvPart(qkv, cfg.qDim(), cfg.kvDim(), &nv),
            else => qkvPart(qkv, cfg.qDim() + cfg.kvDim(), cfg.kvDim(), &nv),
        };
        be.prefetchWeight(part.bytes);
    }
    be.prefetchWeight(blk.attn.out.bytes);
    be.prefetchWeight(bytes(blk.attn.qnorm));
    be.prefetchWeight(bytes(blk.attn.knorm));
    be.prefetchWeight(bytes(blk.attn_norm2));
    be.prefetchWeight(bytes(blk.ffn_norm2));
    inline for (.{ blk.ffn.w1, blk.ffn.w3, blk.ffn.w2 }) |w| be.prefetchWeight(w.bytes);
}

/// A non-owning device-pointer view at a byte offset, sized to what will be read.
fn offsetBuf(b: Buf, off_bytes: usize, size: usize) Buf {
    return .{ .buf = @enumFromInt(@intFromEnum(b.buf) + off_bytes), .mem = .null_handle, .size = size };
}

/// A contiguous row range of the fused `[q_dim + 2*kv_dim, dim]` qkv weight. Zero-copy
/// `[q|k|v]` are row blocks, and the device weight cache keys on the host pointer,
/// so the three views cache separately. Part 0 shares the fused tensor's pointer, so
/// the whole tensor must never be uploaded as well.
fn qkvPart(w: Weight, row0: usize, nrows: usize, nv: *ops.nvfp4.Meta) Weight {
    const row_bytes = w.dtype.storageBytes(w.cols);
    var s = w;
    s.rows = nrows;
    s.bytes = w.bytes[row0 * row_bytes ..][0 .. nrows * row_bytes];
    // An NVFP4 weight's per-block scales have to be row-sliced too, into caller-owned
    // storage that outlives the returned `Weight`. Without it the k and v views would read
    // q's block scales, see `ops.nvfp4.Meta.rowSlice`.
    if (w.nvfp4) |m| {
        nv.* = m.rowSlice(w.cols, row0, nrows);
        s.nvfp4 = nv;
    }
    return s;
}

/// A block GEMM, dispatched by weight dtype, the same routing `dit_cuda.lin` uses.
/// Ampere+ feeds raw bf16 straight to the tensor cores; older cards take the
/// GPU-side bf16->f16 GEMM.
fn gemm(be: *Backend, y: Buf, x: Buf, m: usize, w: Weight) !void {
    const zeros: []const f32 = &zero_bias;
    std.debug.assert(w.rows <= zeros.len);
    switch (w.dtype) {
        // `null`, not `zeros`: Z-Image's block linears are all bias-free, and a
        // null bias lets the `.libs` arm write the GEMM straight into `y` instead
        // of staging through `conv_c` and re-reading the whole output to add zero.
        .bf16 => if (be.ctx.cc_major >= 8 and w.rows % 128 == 0 and w.cols % 32 == 0)
            try be.opGemmBf16(y, x, m, w.bytes, w.rows, w.cols, null)
        else
            try be.opMatmulBf16(y, x, m, w.bytes, w.rows, w.cols, zeros),
        .f8_e4m3 => try be.opMatmulFp8(y, x, m, w.bytes, w.scale, w.rows, w.cols),
        // Weight-only NVFP4: the 4-bit weight is decoded to an f16 scratch inside the GEMM
        // and the packed form stays resident. `rows % 128` / `cols % 32` come from the f16
        // GEMM it feeds; every NVFP4 layer in the shipped checkpoints satisfies both.
        .nvfp4 => {
            std.debug.assert(w.rows % 128 == 0 and w.cols % 32 == 0);
            const meta = w.nvfp4.?;
            try be.opMatmulNvfp4(y, x, m, w.bytes, meta.scales, std.mem.asBytes(&meta.levels.bf16v), w.rows, w.cols, zeros);
        },
        .f32 => try be.opMatmul(y, 0, x, 0, m, w.bytes, false, w.rows, w.cols, w.scale, null),
        // `supported` gates this before a session is built.
        else => return error.UnsupportedDType,
    }
}

/// Wrap a CPU norm-weight slice as a (pointer-cached) small device buffer.
/// `smallBuffer` returns a raw handle; the elt launchers want a `DeviceBuffer`.
fn normBuf(be: *Backend, weights: []const f32) !Buf {
    const h = try be.smallBuffer(std.mem.sliceAsBytes(weights));
    return .{ .buf = h, .size = weights.len * 4 };
}
