//! GPU-resident Z-Image (`NextDiT`) forward on Vulkan: the two `noise_refiner`
//! blocks and the whole 30-block trunk run on the device, with the small paths kept
//! on the host exactly as `dit_gpu` keeps krea2's.
//!
//! **What stays on the CPU, and why it is not laziness:**
//!
//! - The **caption half** (`cap_embedder`, the pad, and both `context_refiner`
//!   blocks). ⚠️ Those blocks are built with `modulation=False`, so the text half
//!   does not depend on the timestep at all — it is computed once per *image*, not
//!   per step, and moving it to the device would buy nothing while costing a second
//!   sequence length's worth of buffers.
//! - The **timestep MLP** and every block's AdaLN linear, precomputed for the whole
//!   schedule at `Session.init` (see `zimage.modulationTable`). A `[15360, 256]`
//!   GEMV per block is negligible next to the block, and hoisting it out of the step
//!   loop is what `dit_gpu` does with its `tvs` table for the same reason.
//! - **Patchify** and the **final layer**, both of which touch only the image rows
//!   and are a memcpy-and-a-GEMV apiece.
//!
//! **No new kernel was needed for any of this** — the whole block maps onto the
//! existing vocabulary — which is the main way this port differs from the SD family's
//! (GroupNorm, the SpatialTransformer and the LIFO skip stack were all new there).
//! Two mappings are worth naming because they are not obvious:
//!
//! - ⚠️ **Z-Image's modulation has NO SHIFT**, and the device `modulate` kernel
//!   computes `(1 + c[scale_off]) * x + c[shift_off]`. Rather than a second kernel,
//!   `zimage.modulationTable` appends one **zero block** and `shift_off` points at
//!   it. Exactly equal, and it keeps the two backends reading the same table.
//! - ⚠️ **The gates are `tanh`'d on the host**, inside that same table. A gate is one
//!   `dim`-wide vector per block per step, so this costs nothing measurable and saves
//!   a kernel.
//!
//! ⚠️ **The fused `qkv` weight is split into three zero-copy ROW VIEWS** rather than
//! de-interleaved after one wide GEMM (which is what the CPU path prefers). `[q|k|v]`
//! are contiguous row blocks of `[3*dim, dim]`, so three GEMMs write straight into
//! contiguous q/k/v buffers. The device weight cache keys on the host pointer, so the
//! three views cache as three separate device weights — which is what we want, but it
//! also means the *whole* fused tensor must never be uploaded as well (part 0 shares
//! its pointer).
//!
//! **Attention runs the tensor-core scores/PV pipeline**, with the self-contained
//! `attn_full` kept behind `force_attn_full` as the reference the device test compares
//! against — both are checked against the CPU forward independently.
//!
//! ⚠️ **The tensor-core path is a REQUIREMENT here, not an optimization.** `attn_full`
//! is one thread per (query, head) looping every key, so a 1056x1584 render (~6800
//! tokens) runs long enough to trip the GPU watchdog — `error_device_lost`, not a slow
//! render. Landing correctness-first was still right (it is how the Vulkan qwen35 port
//! went in, and it is what the fast path got validated against), but calling the fast
//! path a "follow-up" was wrong: nothing above ~768px works without it. Z-Image needs
//! no head padding for it — head_dim is exactly the 128 `coopmat.buildGemmAttnOut`
//! tiles — and every GEMM width is a multiple of 128.
//!
//! **Measured** (`zig build test -Dintegration -Dtest-filter="Z-Image gpu"`, needs
//! the `testdata/gpu-tests` marker): `attn_full` at Z-Image's own 30 x 128 head
//! geometry matches `ops.attention` at **4.9e-7** rel L2, and the whole forward on
//! real checkpoint weights matches `zimage.DiT.predict` at **2.1e-4** — the f16
//! tensor-core regime `sd_unet_cuda` already sits in, and three orders below the
//! layout mistakes these tests exist to catch.

const std = @import("std");
const zimage = @import("zimage.zig");
const gpu = @import("tp_gpu").context;
const ops = @import("tp_ops");

const DiT = zimage.DiT;
const Buf = gpu.DeviceBuffer;
const Weight = ops.matmul.Weight;

/// Two-pass softmax chunk count — 32 interleaved chunks per row so a warp covers a
/// row with coalesced reads. Same value `dit_gpu` uses.
const nchunks = 32;
/// Cap on the materialized attention-scores buffer; heads batch to fit it.
const s_bytes_cap: usize = 2 << 30;

/// Force the correctness-first `attn_full` path even where the tensor-core scores
/// pipeline exists. For A/B and for reproducing a mismatch; the device test runs both.
pub var force_attn_full: bool = false;

/// How many heads share one scores plane, given the plane size and a byte budget.
fn headsPerBatch(cfg: zimage.Config, rows_pad: usize, cap: usize, ws_s_bytes: ?usize) usize {
    const plane = rows_pad * rows_pad * 2; // f16 scores
    var hb = @max(1, @min(cfg.n_heads, cap / @max(plane, 1)));
    if (ws_s_bytes) |c| hb = @max(1, @min(hb, c / @max(plane, 1)));
    return hb;
}

fn scoresCap(budget: u64) usize {
    if (budget == 0) return s_bytes_cap;
    return @min(s_bytes_cap, @max(64 << 20, budget / 4));
}

/// Whether the tensor-core scores/PV pipeline is available and wanted.
fn useTensorCoreAttn(ctx: *gpu.Context) bool {
    return !force_attn_full and ctx.pipe_scores != .null_handle;
}

/// Per-image cache: everything constant across sampling steps.
pub const Session = struct {
    cfg: zimage.Config,
    lat_h: usize,
    lat_w: usize,
    cap_padded: usize,
    n_img: usize,
    img_padded: usize,
    /// `cap_padded + img_padded` — the joint sequence the trunk runs on.
    seq: usize,

    /// The refined caption half, `[cap_padded][dim]`, uploaded once.
    cap_d: Buf,
    /// The learned image pad token, `[dim]`, uploaded once.
    x_pad_d: Buf,
    /// Interleaved-RoPE table for the JOINT sequence: `cos` then `sin`,
    /// `seq * half` each.
    freqs_d: Buf,
    /// The image half's own table, i.e. the same positions the image tokens hold in
    /// the joint sequence. ⚠️ A separate upload rather than an offset into
    /// `freqs_d`, because `opElt` binds whole buffers: the `noise_refiner` runs on
    /// the image tokens alone and its `rope_inter` needs row 0 of its table to be the
    /// image's first position, not the caption's.
    img_freqs_d: Buf,

    /// The schedule this session precomputed for, and the per-sigma modulation
    /// tables + final-layer scales. Indexed by `tv`.
    sigmas: []f32,
    /// `[sigmas.len][modulatedBlocks * 4 * dim + dim]`, host-side; uploaded per step.
    mods: []f32,
    /// `[sigmas.len][dim]`.
    finals: []f32,
    mod_stride: usize,

    pub fn init(
        gpa: std.mem.Allocator,
        io: std.Io,
        ctx: *gpu.Context,
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
            for (bufs[0..made]) |b| ctx.tensorDestroy(b);
            if (self.sigmas.len != 0) gpa.free(self.sigmas);
            if (self.mods.len != 0) gpa.free(self.mods);
            if (self.finals.len != 0) gpa.free(self.finals);
        }

        self.cap_d = try ctx.tensorCreate(cap.len * 4);
        made += 1;
        try ctx.tensorUpload(self.cap_d, std.mem.sliceAsBytes(cap));
        self.x_pad_d = try ctx.tensorCreate(cfg.dim * 4);
        made += 1;
        try ctx.tensorUpload(self.x_pad_d, std.mem.sliceAsBytes(model.x_pad_token));

        // Both rope tables, laid out cos-then-sin the way `rope_inter` reads them
        // (`u2 = sin_off` in elements).
        {
            var joint = try model.ropeFreqs(gpa, cap_padded, h, w);
            defer joint.deinit(gpa);
            const flat = try gpa.alloc(f32, 2 * seq * half);
            defer gpa.free(flat);
            @memcpy(flat[0 .. seq * half], joint.cos);
            @memcpy(flat[seq * half ..], joint.sin);
            self.freqs_d = try ctx.tensorCreate(flat.len * 4);
            made += 1;
            try ctx.tensorUpload(self.freqs_d, std.mem.sliceAsBytes(flat));

            const iflat = try gpa.alloc(f32, 2 * img_padded * half);
            defer gpa.free(iflat);
            @memcpy(iflat[0 .. img_padded * half], joint.cos[cap_padded * half ..]);
            @memcpy(iflat[img_padded * half ..], joint.sin[cap_padded * half ..]);
            self.img_freqs_d = try ctx.tensorCreate(iflat.len * 4);
            made += 1;
            try ctx.tensorUpload(self.img_freqs_d, std.mem.sliceAsBytes(iflat));
        }

        // Precompute the whole schedule's modulation, as `dit_gpu` does with its
        // timestep vectors: it is ~2 MB per sigma and turns a per-step CPU GEMV
        // chain into a per-step upload.
        self.sigmas = try gpa.dupe(f32, sigmas);
        self.mods = try gpa.alloc(f32, sigmas.len * self.mod_stride);
        self.finals = try gpa.alloc(f32, sigmas.len * cfg.dim);
        for (sigmas, 0..) |s, i| {
            const adaln = try model.adalnInput(io, gpa, s);
            defer gpa.free(adaln);
            const tbl = try model.modulationTable(io, gpa, adaln);
            defer gpa.free(tbl);
            @memcpy(self.mods[i * self.mod_stride ..][0..self.mod_stride], tbl);
            const fin = try model.finalScale(io, gpa, adaln);
            defer gpa.free(fin);
            @memcpy(self.finals[i * cfg.dim ..][0..cfg.dim], fin);
        }
        return self;
    }

    pub fn deinit(self: *Session, gpa: std.mem.Allocator, ctx: *gpu.Context) void {
        ctx.tensorDestroy(&self.cap_d);
        ctx.tensorDestroy(&self.x_pad_d);
        ctx.tensorDestroy(&self.freqs_d);
        ctx.tensorDestroy(&self.img_freqs_d);
        gpa.free(self.sigmas);
        gpa.free(self.mods);
        gpa.free(self.finals);
        self.* = undefined;
    }

    /// The precomputed modulation + final scale for `sigma`, or null when the caller
    /// probed a sigma that is not on the schedule this session was built for (a
    /// teacher-forced measurement); `forward` then recomputes on the fly.
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

/// One step's modulation, either from the session's precomputed schedule or built
/// on the fly for an off-schedule sigma. A named type because both producers must
/// agree on it.
const StepMod = struct { mods: []const f32, final: []const f32 };

/// Per-resolution device scratch, shared by both conditioning branches.
pub const Workspace = struct {
    x_d: Buf, // [seq][dim] the joint sequence
    img_d: Buf, // [img_padded][dim] the image half during the refiner stage
    nrm_d: Buf, // [seq][dim] modulated pre-norm
    dlt_d: Buf, // [seq][dim] sublayer output
    q_d: Buf,
    k_d: Buf,
    v_d: Buf,
    attn_d: Buf,
    mg_d: Buf, // [seq][mlp_dim]
    mu_d: Buf,
    mv_d: Buf, // the whole modulation table for this step
    imgin_d: Buf, // [n_img][patchDim] raw patches
    // Tensor-core attention operands and scratch. ⚠️ Sized for the TRUNK sequence,
    // which is the longer of the two the forward runs (the `noise_refiner` sees only
    // the image half), and `hb` is recomputed per call from `s_d.size` so a shorter
    // sequence just batches more heads at once.
    qt_d: Buf, // f16 [rows_pad][q_dim], softmax scale prefolded
    kt_d: Buf, // f16 per-head k-major [kv_heads][hd][rows_pad]
    v16_d: Buf, // f16 [rows_pad][kv_dim]
    s_d: Buf, // f16 [hb][rows_pad][rows_pad]
    part_d: Buf,
    md_d: Buf,

    const fields = [_][]const u8{ "x_d", "img_d", "nrm_d", "dlt_d", "q_d", "k_d", "v_d", "attn_d", "mg_d", "mu_d", "mv_d", "imgin_d", "qt_d", "kt_d", "v16_d", "s_d", "part_d", "md_d" };

    pub fn init(ctx: *gpu.Context, model: *const DiT, lat_h: usize, lat_w: usize, cap_padded_cap: usize) !Workspace {
        const cfg = model.cfg;
        const n_img = (lat_h / cfg.patch) * (lat_w / cfg.patch);
        const img_padded = cfg.padded(n_img);
        const seq = cap_padded_cap + img_padded;
        const seq_pad = std.mem.alignForward(usize, seq, 128);
        const tc = useTensorCoreAttn(ctx);
        const hpb = if (tc) headsPerBatch(cfg, seq_pad, scoresCap(ctx.budget_override), null) else 1;
        const sizes = [fields.len]usize{
            seq * cfg.dim * 4,
            img_padded * cfg.dim * 4,
            seq * cfg.dim * 4,
            seq * cfg.dim * 4,
            // ⚠️ q/k/v are 128-row PADDED: the f16 conversion writes whole padded
            // rows, and `opAttnOut` writes `rows_pad` of them back into `attn_d`.
            seq_pad * cfg.qDim() * 4,
            seq_pad * cfg.kvDim() * 4,
            seq_pad * cfg.kvDim() * 4,
            seq_pad * cfg.qDim() * 4,
            seq * cfg.mlp_dim * 4,
            seq * cfg.mlp_dim * 4,
            (model.modulatedBlocks() * 4 * cfg.dim + cfg.dim) * 4,
            n_img * cfg.patchDim() * 4,
            if (tc) seq_pad * cfg.qDim() * 2 else 16,
            if (tc) cfg.kvDim() * seq_pad * 2 else 16,
            if (tc) seq_pad * cfg.kvDim() * 2 else 16,
            if (tc) hpb * seq_pad * seq_pad * 2 else 16,
            if (tc) hpb * seq * nchunks * 2 * 4 else 16,
            if (tc) hpb * seq_pad * 2 * 4 else 16,
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

    pub fn deinit(self: *Workspace, ctx: *gpu.Context) void {
        inline for (fields) |name| ctx.tensorDestroy(&@field(self, name));
        self.* = undefined;
    }
};

/// Whether this context can run Z-Image's block GEMMs at all. The trunk weights are
/// dense bf16 in every checkpoint seen so far, which needs one of the two f16-weight
/// tensor-core pipelines; a device without either has no path and must stay on the
/// CPU rather than silently producing something.
pub fn supported(ctx: *gpu.Context, model: *const DiT) bool {
    if (model.layers.len == 0) return false;
    // ⚠️ Every layer, and every dtype checked against what THIS device has — a bf16 weight
    // needs one of the f16-weight coop pipelines and an NVFP4 one needs the decode entry.
    const Cap = struct {
        var has_f16w: bool = false;
        var has_nvfp4: bool = false;
        fn f(dt: @import("tp_core").dtype.DType) bool {
            return switch (dt) {
                .bf16, .f16 => has_f16w,
                .nvfp4 => has_nvfp4,
                else => true,
            };
        }
    };
    Cap.has_f16w = ctx.pipe_coop_bf16w != .null_handle or ctx.pipe_coop_f16w != .null_handle;
    Cap.has_nvfp4 = ctx.hasNvfp4Decode();
    if (zimage.unsupportedGpuLin(model, Cap.f)) |bad| {
        std.log.warn("zimage_gpu: {s} is {t}, which this device has no GEMM for — the trunk " ++
            "runs on the CPU. Expect CPU sampling speed.", .{ bad.tag, bad.dtype });
        return false;
    }
    return true;
}

/// One denoiser forward. `out`/`x_lat` are planar `[channels][lat_h][lat_w]`.
pub fn forward(
    model: *const DiT,
    ctx: *gpu.Context,
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
    const sin_off: u32 = @intCast(seq * half);
    const img_sin_off: u32 = @intCast(sess.img_padded * half);
    const attn_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));
    const zero_off: u32 = @intCast(model.zeroShiftOffset());

    std.debug.assert(x_lat.len == cfg.channels * sess.lat_h * sess.lat_w);
    std.debug.assert(out.len == x_lat.len);

    // The step's modulation, from the precomputed schedule where possible.
    var owned_mods: ?[]f32 = null;
    var owned_final: ?[]f32 = null;
    defer if (owned_mods) |b| gpa.free(b);
    defer if (owned_final) |b| gpa.free(b);
    const step = sess.tv(sigma) orelse blk: {
        const adaln = try model.adalnInput(io, gpa, sigma);
        defer gpa.free(adaln);
        owned_mods = try model.modulationTable(io, gpa, adaln);
        owned_final = try model.finalScale(io, gpa, adaln);
        break :blk StepMod{ .mods = owned_mods.?, .final = owned_final.? };
    };

    const patches = try zimage.patchify(gpa, cfg, x_lat, sess.lat_h, sess.lat_w);
    defer gpa.free(patches);

    try ctx.tensorUpload(ws.mv_d, std.mem.sliceAsBytes(step.mods));
    try ctx.tensorUpload(ws.imgin_d, std.mem.sliceAsBytes(patches));

    // Pre-size the NVFP4 decode scratch to the model's widest linear BEFORE the batch
    // opens; growing it mid-forward flushes the recording batch.
    {
        const need = zimage.maxNvfp4Scratch(model, gpu.Context.nvfp4ScratchBytes);
        if (need > 0) try ctx.ensureDeviceBuffer(&ctx.nvfp4_w16, need);
    }

    try ctx.beginBatch();
    errdefer if (ctx.batching) ctx.abortBatch();

    // --- the image half: x_embedder, pad, then the noise_refiner stack ---------
    //
    // `x_embedder` is the one weight the loader materializes to f32 (it feeds the
    // fused `opMatmul`, which has no bf16 pipeline), so it takes the f32 arm.
    try ctx.opMatmul(ws.img_d, 0, ws.imgin_d, 0, sess.n_img, model.x_embedder.w.bytes, model.x_embedder.w.dtype == .f8_e4m3, dim, cfg.patchDim(), model.x_embedder.w.scale, model.x_embedder.b);
    // The learned pad token, broadcast into the tail rows.
    for (sess.n_img..sess.img_padded) |r| {
        try ctx.opElt(.copy, sess.x_pad_d, ws.img_d, null, null, .{
            .u0 = @intCast(dim),
            .u2 = @intCast(r * dim),
            .u3 = 0,
        }, dim, 1, 1);
    }
    for (model.noise_refiner, 0..) |*blk, i| {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        try blockForward(ctx, cfg, blk, ws, ws.img_d, sess.img_padded, sess.img_freqs_d, img_sin_off, @intCast(i * 4 * dim), zero_off, attn_scale);
    }

    // --- the joint sequence ----------------------------------------------------
    try ctx.opElt(.copy, sess.cap_d, ws.x_d, null, null, .{
        .u0 = @intCast(sess.cap_padded * dim),
        .u2 = 0,
        .u3 = 0,
    }, sess.cap_padded * dim, 1, 1);
    try ctx.opElt(.copy, ws.img_d, ws.x_d, null, null, .{
        .u0 = @intCast(sess.img_padded * dim),
        .u2 = @intCast(sess.cap_padded * dim),
        .u3 = 0,
    }, sess.img_padded * dim, 1, 1);

    for (model.layers, 0..) |*blk, i| {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        const base: u32 = @intCast((model.noise_refiner.len + i) * 4 * dim);
        try blockForward(ctx, cfg, blk, ws, ws.x_d, seq, sess.freqs_d, sin_off, base, zero_off, attn_scale);
    }
    try ctx.endBatch();

    // --- the final layer, on the host ------------------------------------------
    // Row-wise, so running it on the image rows alone is exactly equal to running it
    // on the whole sequence and slicing — which is what the reference does.
    const img_rows = try gpa.alloc(f32, sess.n_img * dim);
    defer gpa.free(img_rows);
    try ctx.tensorDownloadAt(ws.x_d, sess.cap_padded * dim * 4, std.mem.sliceAsBytes(img_rows));

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

/// One `JointTransformerBlock` on the device. `x` is `[rows][dim]`, modified in
/// place; `mod_base` is this block's offset into the uploaded modulation table.
fn blockForward(
    ctx: *gpu.Context,
    cfg: zimage.Config,
    blk: anytype,
    ws: *Workspace,
    x: Buf,
    rows: usize,
    freqs: Buf,
    sin_off: u32,
    mod_base: u32,
    zero_off: u32,
    attn_scale: f32,
) !void {
    const dim = cfg.dim;
    const heads = cfg.n_heads;
    const kv_heads = cfg.n_kv_heads;
    const hd = cfg.head_dim;
    const half = hd / 2;
    const total: u32 = @intCast(rows * dim);

    // --- attention: x += tanh(gate_msa) * norm2(attn(modulate(norm1(x), scale_msa)))
    try rmsNorm(ctx, x, ws.nrm_d, blk.attn_norm1, rows, dim, cfg.norm_eps);
    // ⚠️ `u3` points at the table's trailing ZERO block: Z-Image's modulation has no
    // shift, and this is how the shared `modulate` kernel expresses that.
    try ctx.opElt(.modulate, ws.nrm_d, null, ws.mv_d, null, .{
        .u0 = total,
        .u1 = @intCast(dim),
        .u2 = mod_base,
        .u3 = zero_off,
    }, rows * dim, 1, 1);

    // The fused qkv as three zero-copy row views — see the module header.
    var nvq: ops.nvfp4.Meta = undefined;
    var nvk: ops.nvfp4.Meta = undefined;
    var nvv: ops.nvfp4.Meta = undefined;
    const wq = qkvPart(blk.attn.qkv, 0, cfg.qDim(), &nvq);
    const wk = qkvPart(blk.attn.qkv, cfg.qDim(), cfg.kvDim(), &nvk);
    const wv = qkvPart(blk.attn.qkv, cfg.qDim() + cfg.kvDim(), cfg.kvDim(), &nvv);
    try gemm(ctx, ws.q_d, ws.nrm_d, rows, wq);
    try gemm(ctx, ws.k_d, ws.nrm_d, rows, wk);
    try gemm(ctx, ws.v_d, ws.nrm_d, rows, wv);

    // ⚠️ The Q/K norms take `finfo(f32).eps`, NOT the blocks' 1e-5 — see
    // `zimage.Config.qk_eps`. Same kernel, different push constant.
    ctx.independent(2);
    try rmsNorm(ctx, ws.q_d, ws.q_d, blk.attn.qnorm, rows * heads, hd, cfg.qk_eps);
    try rmsNorm(ctx, ws.k_d, ws.k_d, blk.attn.knorm, rows * kv_heads, hd, cfg.qk_eps);
    ctx.independent(2);
    try ctx.opElt(.rope_inter, ws.q_d, null, freqs, null, .{
        .u0 = @intCast(rows * heads * half),
        .u1 = @intCast(half),
        .u2 = sin_off,
        .u3 = @intCast(heads),
    }, rows * heads * half, 1, 1);
    try ctx.opElt(.rope_inter, ws.k_d, null, freqs, null, .{
        .u0 = @intCast(rows * kv_heads * half),
        .u1 = @intCast(half),
        .u2 = sin_off,
        .u3 = @intCast(kv_heads),
    }, rows * kv_heads * half, 1, 1);

    try attention(ctx, cfg, ws, rows, attn_scale);
    try gemm(ctx, ws.dlt_d, ws.attn_d, rows, blk.attn.out);
    // The sandwich norm: a SECOND RMSNorm, on the sublayer's output, inside the
    // residual. krea2 has no equivalent.
    try rmsNorm(ctx, ws.dlt_d, ws.dlt_d, blk.attn_norm2, rows, dim, cfg.norm_eps);
    // The gate was `tanh`'d on the host, inside the table.
    try ctx.opElt(.gated_add, x, ws.dlt_d, ws.mv_d, null, .{
        .u0 = total,
        .u1 = @intCast(dim),
        .u2 = mod_base + @as(u32, @intCast(dim)),
    }, rows * dim, 1, 1);

    // --- feed-forward ----------------------------------------------------------
    try rmsNorm(ctx, x, ws.nrm_d, blk.ffn_norm1, rows, dim, cfg.norm_eps);
    try ctx.opElt(.modulate, ws.nrm_d, null, ws.mv_d, null, .{
        .u0 = total,
        .u1 = @intCast(dim),
        .u2 = mod_base + 2 * @as(u32, @intCast(dim)),
        .u3 = zero_off,
    }, rows * dim, 1, 1);
    try gemm(ctx, ws.mg_d, ws.nrm_d, rows, blk.ffn.w1);
    try gemm(ctx, ws.mu_d, ws.nrm_d, rows, blk.ffn.w3);
    try ctx.opElt(.silu_mul, ws.mg_d, ws.mu_d, null, null, .{
        .u0 = @intCast(rows * cfg.mlp_dim),
    }, rows * cfg.mlp_dim, 1, 1);
    try gemm(ctx, ws.dlt_d, ws.mg_d, rows, blk.ffn.w2);
    try rmsNorm(ctx, ws.dlt_d, ws.dlt_d, blk.ffn_norm2, rows, dim, cfg.norm_eps);
    try ctx.opElt(.gated_add, x, ws.dlt_d, ws.mv_d, null, .{
        .u0 = total,
        .u1 = @intCast(dim),
        .u2 = mod_base + 3 * @as(u32, @intCast(dim)),
    }, rows * dim, 1, 1);
}

/// Attention over `rows` positions, q/k/v already normed and rope'd in `ws.q_d`/`k_d`/
/// `v_d`, result into `ws.attn_d`. Both paths are non-causal and mathematically the
/// same; they differ in how the scores are produced.
///
/// ⚠️ **The tensor-core path is not an optimization here, it is a requirement.**
/// `attn_full` is one thread per (query, head) looping every key, which at a 1056x1584
/// render (~6800 tokens) runs long enough to trip the GPU watchdog —
/// `error_device_lost`, not a slow render. It stays as the reference the device test
/// compares against, and as the fallback where `pipe_scores` is absent.
fn attention(ctx: *gpu.Context, cfg: zimage.Config, ws: *Workspace, rows: usize, scale: f32) !void {
    const heads = cfg.n_heads;
    const kv_heads = cfg.n_kv_heads;
    const hd = cfg.head_dim;
    const q_dim = cfg.qDim();
    const kv_dim = cfg.kvDim();

    if (!useTensorCoreAttn(ctx)) {
        return ctx.opElt(.attn_full, ws.q_d, ws.k_d, ws.v_d, ws.attn_d, .{
            .u0 = @intCast(rows),
            .u1 = @intCast(heads),
            .u2 = @intCast(kv_heads),
            .u3 = @intCast(hd),
            .f0 = scale,
        }, rows * heads, 1, 1);
    }

    const rows_pad = std.mem.alignForward(usize, rows, 128);
    // f16 operands: Q with the softmax scale prefolded and zero pad rows, K per-head
    // k-major with zero pad columns, V with zero pad rows — so padded keys score 0 and
    // padded-j probabilities contribute nothing to P@V.
    ctx.independent(3);
    try ctx.opElt(.f32_to_h16, ws.q_d, null, null, ws.qt_d, .{
        .u0 = @intCast(rows_pad * q_dim / 2),
        .u1 = @intCast(rows * q_dim),
        .f0 = scale,
    }, rows_pad * q_dim / 2, 1, 1);
    try ctx.opElt(.gather_kmajor_h16, ws.k_d, null, null, ws.kt_d, .{
        .u0 = @intCast(kv_dim * rows_pad / 2),
        .u1 = @intCast(hd),
        .u2 = @intCast(rows_pad),
        .u3 = @intCast(rows),
        .u4 = @intCast(kv_heads),
    }, kv_dim * rows_pad / 2, 1, 1);
    try ctx.opElt(.f32_to_h16, ws.v_d, null, null, ws.v16_d, .{
        .u0 = @intCast(rows_pad * kv_dim / 2),
        .u1 = @intCast(rows * kv_dim),
        .f0 = 1.0,
    }, rows_pad * kv_dim / 2, 1, 1);

    // ⚠️ Recomputed per call, not taken from `Workspace.init`: the two sequence
    // lengths a forward runs (image half, then the joint sequence) give different
    // plane sizes, and the buffers were sized for the longer one.
    const plane = rows_pad * rows_pad * 2;
    var hb_cap = headsPerBatch(cfg, rows_pad, scoresCap(ctx.budget_override), ws.s_d.size);
    hb_cap = @max(1, @min(hb_cap, ws.part_d.size / @max(rows * nchunks * 2 * 4, 1)));
    hb_cap = @max(1, @min(hb_cap, ws.md_d.size / @max(rows_pad * 2 * 4, 1)));
    std.debug.assert(hb_cap * plane <= ws.s_d.size);

    const s_stride: u32 = @intCast(rows_pad);
    const s_plane: u32 = @intCast(rows_pad * rows_pad);
    var h0: usize = 0;
    while (h0 < heads) : (h0 += hb_cap) {
        const hb = @min(hb_cap, heads - h0);
        try ctx.opAttnScores(ws.s_d, ws.qt_d, ws.kt_d, .{
            .u0 = @intCast(q_dim),
            .u1 = s_stride,
            .u2 = @intCast(h0),
            .u3 = @intCast(heads / kv_heads),
            .u4 = @intCast(hd * rows_pad),
            .u5 = s_plane,
        }, rows_pad / 128, rows_pad / 128, hb);
        try ctx.opElt(.softmax_partial, ws.s_d, null, null, ws.part_d, .{
            .u0 = @intCast(hb * rows * nchunks),
            .u1 = nchunks,
            .u2 = @intCast(rows),
            .u3 = s_stride,
            .u5 = s_plane,
        }, hb * rows * nchunks, 1, 1);
        try ctx.opElt(.softmax_combine, ws.part_d, null, null, ws.md_d, .{
            .u0 = @intCast(hb * rows),
            .u1 = nchunks,
            .u2 = @intCast(rows),
            .u3 = s_stride,
        }, hb * rows, 1, 1);
        try ctx.opAttnOut(ws.s_d, ws.v16_d, ws.attn_d, ws.md_d, .{
            .u0 = s_stride,
            .u1 = s_plane,
            .u2 = @intCast(h0),
            .u3 = @intCast(heads / kv_heads),
            .u4 = @intCast(kv_dim),
            .u5 = @intCast(q_dim),
            .f0 = @bitCast(@as(u32, @intCast(rows))),
            .f1 = @bitCast(s_stride), // MD rows per head plane
        }, rows_pad / 128, hb);
    }
}

/// A contiguous row range of the fused `[q_dim + 2*kv_dim, dim]` qkv weight, as a
/// `Weight` in its own right. Zero-copy: `[q|k|v]` are row blocks, so this is a
/// slice, and the device weight cache keys on the pointer.
fn qkvPart(w: Weight, row0: usize, nrows: usize, nv: *ops.nvfp4.Meta) Weight {
    const row_bytes = w.dtype.storageBytes(w.cols);
    var s = w;
    s.rows = nrows;
    s.bytes = w.bytes[row0 * row_bytes ..][0 .. nrows * row_bytes];
    // ⚠️ An NVFP4 weight's per-block scales have to be row-sliced too, into caller-owned
    // storage that outlives the returned `Weight`. Without it the k and v views would read
    // q's block scales — see `ops.nvfp4.Meta.rowSlice`.
    if (w.nvfp4) |m| {
        nv.* = m.rowSlice(w.cols, row0, nrows);
        s.nvfp4 = nv;
    }
    return s;
}

/// A block GEMM, dispatched by weight dtype. Z-Image ships dense bf16, which takes
/// one of the two f16-weight tensor-core pipelines (native bf16 where the device has
/// a bf16 coop config, else bf16→f16 at upload — both keep the conversion off the
/// host under weight streaming), exactly as `dit_gpu` routes its dense bf16 blocks.
fn gemm(ctx: *gpu.Context, y: Buf, x: Buf, m: usize, w: Weight) !void {
    // The block linears carry no bias, but the f16-weight coop GEMM always folds one
    // in, so hand it a zero vector.
    //
    // ⚠️ **The FULL slice, never `zeros[0..w.rows]`.** `smallBuffer` caches by host
    // POINTER alone, so every width would map to whichever length was uploaded first:
    // once a `dim`-wide GEMM cached a 3840-float buffer, the `mlp_dim` GEMM's
    // `bias_compact` would read 10240 floats out of it and run off the end. The
    // dispatch reads only `rows` entries, so one full-width buffer serves them all —
    // which is also exactly what `dit_gpu` does and why.
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
        // Weight-only NVFP4: decoded to an f16 scratch inside the GEMM, so the 4-bit form
        // stays resident. ⚠️ `zeros` is the FULL vector for the reason above.
        .nvfp4 => {
            std.debug.assert(w.rows % 128 == 0 and w.cols % 32 == 0);
            const meta = w.nvfp4.?;
            try ctx.opMatmulNvfp4(y, x, m, w.bytes, meta.scales, std.mem.asBytes(&meta.levels.bf16v), w.rows, w.cols, zeros);
        },
        .f32, .f8_e4m3 => try ctx.opMatmul(y, 0, x, 0, m, w.bytes, w.dtype == .f8_e4m3, w.rows, w.cols, w.scale, null),
        // `zimage.gpuLinKindSupported` gates this before a session is built, so
        // reaching here is a programming error rather than a bad checkpoint.
        else => return error.UnsupportedDType,
    }
}

/// Widest zero bias any Z-Image GEMM needs. `mlp_dim` is the largest output width.
const zero_bias: [zimage.z_image.mlp_dim]f32 = @splat(0);

/// Weighted RMSNorm over `[rows][dim]`, `x -> out` (may alias).
///
/// ⚠️ **The SUBGROUP kernel wherever it exists.** `Elt.rmsnorm` gives each THREAD a whole
/// row, so a warp's 32 loads land `dim * 4` bytes apart and each is its own sector fetch.
/// Measured at Z-Image's own geometries at 1056x1584 (`vk-norm-bench`):
///
/// | | thread/row | subgroup | |
/// |---|---|---|---|
/// | Q/K, 205440 x 128 | **37 GB/s** | **555 GB/s** | 15.2x |
/// | sandwich, 6848 x 3840 | 114 GB/s | 345 GB/s | 3.0x |
///
/// ⚠️ **Not bit-identical**: the row sum becomes a subgroup tree where it was serial. It is
/// the more accurate of the two, and `ln_mod_sg` next door already reduces that way — but it
/// does move the parity figures, so the gated forward test's bound is the thing to watch.
fn rmsNorm(ctx: *gpu.Context, x: Buf, out: Buf, weights: []const f32, rows: usize, dim: usize, eps: f32) !void {
    const w = try normBuf(ctx, weights);
    if (ctx.hasSubgroupNorm()) return ctx.opRmsNormSg(x, out, w, rows, dim, eps);
    try ctx.opElt(.rmsnorm, x, out, w, null, .{
        .u0 = @intCast(rows),
        .u1 = @intCast(dim),
        .f0 = eps,
    }, rows, 1, 1);
}

fn normBuf(ctx: *gpu.Context, weights: []const f32) !Buf {
    // smallBuffer caches by pointer; wrap the raw handle for opElt.
    const buf = try ctx.smallBuffer(std.mem.sliceAsBytes(weights));
    return .{ .buf = buf, .mem = .null_handle, .size = 0 };
}

// --- tests ------------------------------------------------------------------
//
// Two tiers. The first two need no device at all and pin the pieces a GPU port
// gets wrong — the weight split and the modulation layout, both of which are pure
// conventions that produce a plausible image when muddled. The device tests below
// them check each kernel against the CPU op it reproduces, then the whole forward
// against `zimage.DiT.predict`.

const testing = std.testing;
const test_gate = @import("../test_gate.zig");
const safetensors = @import("tp_core").safetensors;

fn relL2(want: []const f32, got: []const f32) f64 {
    var num: f64 = 0;
    var den: f64 = 0;
    for (want, got) |e, a| {
        const d = @as(f64, e) - @as(f64, a);
        num += d * d;
        den += @as(f64, e) * e;
    }
    return if (den > 0) @sqrt(num / den) else @sqrt(num);
}

test "the fused qkv splits into three row views the CPU forward agrees with" {
    // ⚠️ The GPU path does three GEMMs on row slices where the CPU path does one wide
    // GEMM and de-interleaves the result. Those must be the same linear map, and a
    // swapped or misaligned slice is not an error — q/k/v would simply be each
    // other's, which renders as structured noise. Checked against the CPU's own
    // fused GEMM rather than against a re-derivation.
    const gpa = testing.allocator;
    const io = testing.io;
    const dim = 8;
    const q_dim = 8;
    const kv_dim = 8;
    const rows = q_dim + 2 * kv_dim;
    const m = 3;

    var wbits: [rows * dim]f32 = undefined;
    for (&wbits, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 13)) * 0.25 - 1.0;
    const w = Weight.fromF32(&wbits, rows, dim);

    var x: [m * dim]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 7)) * 0.5 - 1.5;

    // One wide GEMM, as `zimage.attnForward` does.
    const fused = try gpa.alloc(f32, m * rows);
    defer gpa.free(fused);
    try ops.matmul.matmul(io, gpa, fused, &x, m, w, null);

    // Three row-view GEMMs, as `blockForward` does.
    inline for (.{ .{ 0, q_dim }, .{ q_dim, kv_dim }, .{ q_dim + kv_dim, kv_dim } }, 0..) |part, pi| {
        var nv: ops.nvfp4.Meta = undefined;
        const sub = qkvPart(w, part[0], part[1], &nv);
        try testing.expectEqual(@as(usize, part[1]), sub.rows);
        try testing.expectEqual(dim, sub.cols);
        const got = try gpa.alloc(f32, m * part[1]);
        defer gpa.free(got);
        try ops.matmul.matmul(io, gpa, got, &x, m, sub, null);
        for (0..m) |r| {
            for (0..part[1]) |c| {
                errdefer std.debug.print("part {d} row {d} col {d}\n", .{ pi, r, c });
                try testing.expectEqual(fused[r * rows + part[0] + c], got[r * part[1] + c]);
            }
        }
    }
}

test "the modulation table is laid out the way the device kernels index it" {
    // Three conventions in one buffer, none of which fail loudly:
    //  - the chunk ORDER is `scale_msa, gate_msa, scale_mlp, gate_mlp`;
    //  - the two GATE chunks are already `tanh`'d (the device has no tanh kernel);
    //  - a trailing ZERO block exists, which is what lets the shared `modulate`
    //    kernel — `(1 + c[scale_off]) * x + c[shift_off]` — express Z-Image's
    //    shift-free modulation. A missing zero block would add whatever followed.
    const gpa = testing.allocator;
    const io = testing.io;
    try test_gate.requireIntegration();
    try test_gate.requireModelFile(io, zit_ckpt);

    var ck = try safetensors.SafeTensors.open(gpa, io, zit_ckpt);
    defer ck.deinit();
    var cfg = zimage.z_image;
    cfg.n_layers = 2;
    var model = try DiT.load(gpa, .{ .safetensors = &ck }, cfg);
    defer model.deinit();

    const adaln = try model.adalnInput(io, gpa, 0.75);
    defer gpa.free(adaln);
    const tbl = try model.modulationTable(io, gpa, adaln);
    defer gpa.free(tbl);

    const n = model.modulatedBlocks();
    try testing.expectEqual(@as(usize, 4), n); // 2 noise_refiner + 2 trunk layers
    try testing.expectEqual(n * 4 * cfg.dim + cfg.dim, tbl.len);
    try testing.expectEqual(n * 4 * cfg.dim, model.zeroShiftOffset());
    for (tbl[model.zeroShiftOffset()..]) |v| try testing.expectEqual(@as(f32, 0), v);

    // Every gate is a tanh output, so strictly inside (-1, 1); the scales are not
    // bounded that way, and asserting BOTH is what distinguishes "tanh applied" from
    // "the chunks happen to be small".
    var any_scale_outside = false;
    for (0..n) |b| {
        const blk = tbl[b * 4 * cfg.dim ..][0 .. 4 * cfg.dim];
        for (blk[1 * cfg.dim ..][0..cfg.dim]) |g| try testing.expect(@abs(g) < 1.0);
        for (blk[3 * cfg.dim ..][0..cfg.dim]) |g| try testing.expect(@abs(g) < 1.0);
        for (blk[0..cfg.dim]) |sc| {
            if (@abs(sc) >= 1.0) any_scale_outside = true;
        }
    }
    try testing.expect(any_scale_outside);
}

/// A live Vulkan context, or a skip. Needs BOTH the `testdata/gpu-tests` marker and
/// `-Dintegration` (`Context.init` refuses under test without it).
fn gpuCtx(gpa: std.mem.Allocator, io: std.Io) !*gpu.Context {
    std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    return gpu.Context.init(gpa) catch error.SkipZigTest;
}

const zit_ckpt = "/home/qt/genai/comfyui/models/checkpoints/zit/unstableRevolution_V2Fp16.safetensors";

test "Z-Image gpu attention matches ops.attention at the model's head geometry" {
    // `attn_full` at 30 heads x 128 — Z-Image's exact shape, and the one place the
    // GPU path uses a different algorithm (online softmax) from the CPU's two-pass
    // form rather than the same one in a different order.
    const gpa = testing.allocator;
    const io = testing.io;
    try test_gate.requireIntegration();
    const ctx = try gpuCtx(gpa, io);
    defer ctx.deinit();

    const cfg = zimage.z_image;
    const seq = 96;
    const qd = cfg.qDim();
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));

    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    const q = try gpa.alloc(f32, seq * qd);
    defer gpa.free(q);
    const k = try gpa.alloc(f32, seq * qd);
    defer gpa.free(k);
    const v = try gpa.alloc(f32, seq * qd);
    defer gpa.free(v);
    for ([_][]f32{ q, k, v }) |buf| for (buf) |*x| {
        x.* = rnd.floatNorm(f32);
    };

    const want = try gpa.alloc(f32, seq * qd);
    defer gpa.free(want);
    try ops.attention.attention(io, gpa, want, q, k, v, .{
        .seq_q = seq,
        .seq_kv = seq,
        .n_heads = cfg.n_heads,
        .n_kv_heads = cfg.n_kv_heads,
        .head_dim = cfg.head_dim,
    });

    var q_d = try ctx.tensorCreate(q.len * 4);
    defer ctx.tensorDestroy(&q_d);
    var k_d = try ctx.tensorCreate(k.len * 4);
    defer ctx.tensorDestroy(&k_d);
    var v_d = try ctx.tensorCreate(v.len * 4);
    defer ctx.tensorDestroy(&v_d);
    var o_d = try ctx.tensorCreate(want.len * 4);
    defer ctx.tensorDestroy(&o_d);
    try ctx.tensorUpload(q_d, std.mem.sliceAsBytes(q));
    try ctx.tensorUpload(k_d, std.mem.sliceAsBytes(k));
    try ctx.tensorUpload(v_d, std.mem.sliceAsBytes(v));
    try ctx.opElt(.attn_full, q_d, k_d, v_d, o_d, .{
        .u0 = seq,
        .u1 = @intCast(cfg.n_heads),
        .u2 = @intCast(cfg.n_kv_heads),
        .u3 = @intCast(cfg.head_dim),
        .f0 = scale,
    }, seq * cfg.n_heads, 1, 1);

    const got = try gpa.alloc(f32, want.len);
    defer gpa.free(got);
    try ctx.tensorDownload(o_d, std.mem.sliceAsBytes(got));
    const rel = relL2(want, got);
    errdefer std.debug.print("attn rel L2 {e:.4}\n", .{rel});
    try testing.expect(rel < 1e-5); // measured 4.9e-7
}

test "Z-Image gpu forward matches the CPU forward on a real checkpoint" {
    // The definitive one: same weights, same latent, same conditioning, both
    // forwards. Truncated to 2 trunk layers so it loads in seconds — the loop bound
    // is not what a kernel port gets wrong, the block's shape is.
    const gpa = testing.allocator;
    const io = testing.io;
    try test_gate.requireIntegration();
    try test_gate.requireModelFile(io, zit_ckpt);
    const ctx = try gpuCtx(gpa, io);
    defer ctx.deinit();

    var ck = try safetensors.SafeTensors.open(gpa, io, zit_ckpt);
    defer ck.deinit();
    var cfg = zimage.z_image;
    cfg.n_layers = 2;
    var model = try DiT.load(gpa, .{ .safetensors = &ck }, cfg);
    defer model.deinit();
    if (!supported(ctx, &model)) return error.SkipZigTest;

    const lat = 16; // 8x8 = 64 image tokens, exactly two pad buckets
    const seq_txt = 20;
    var prng = std.Random.DefaultPrng.init(11);
    const rnd = prng.random();

    const ctxv = try gpa.alloc(f32, seq_txt * cfg.cap_dim);
    defer gpa.free(ctxv);
    for (ctxv) |*v| v.* = rnd.floatNorm(f32);
    const x_lat = try gpa.alloc(f32, cfg.channels * lat * lat);
    defer gpa.free(x_lat);
    for (x_lat) |*v| v.* = rnd.floatNorm(f32);

    const sigma: f32 = 0.75;
    const cap = try model.capTokens(io, gpa, ctxv, seq_txt);
    defer gpa.free(cap);
    const cap_padded = cfg.padded(seq_txt);

    const want = try gpa.alloc(f32, x_lat.len);
    defer gpa.free(want);
    {
        const adaln = try model.adalnInput(io, gpa, sigma);
        defer gpa.free(adaln);
        try model.predict(io, gpa, want, x_lat, lat, lat, cap, cap_padded, adaln, null);
    }

    // ⚠️ **Both attention paths, and they must agree with the CPU independently.**
    // `attn_full` is the correctness reference but trips the GPU watchdog at
    // production resolutions; the tensor-core scores/PV path is what actually runs.
    // Testing only the fast one would leave the fallback free to rot, and testing only
    // the slow one is what let the watchdog problem reach a render.
    const saved = force_attn_full;
    defer force_attn_full = saved;
    for ([_]bool{ true, false }) |full| {
        force_attn_full = full;
        if (!full and !useTensorCoreAttn(ctx)) continue; // no scores pipeline here

        var sess = try Session.init(gpa, io, ctx, &model, lat, lat, cap, cap_padded, &.{ sigma, 0 });
        defer sess.deinit(gpa, ctx);
        var ws = try Workspace.init(ctx, &model, lat, lat, cap_padded);
        defer ws.deinit(ctx);

        const got = try gpa.alloc(f32, x_lat.len);
        defer gpa.free(got);
        try forward(&model, ctx, &sess, &ws, io, gpa, got, x_lat, sigma, null);

        const rel = relL2(want, got);
        errdefer std.debug.print("zimage gpu forward ({s}) rel L2 {e:.4}\n", .{ if (full) "attn_full" else "tensor-core", rel });
        // The GEMMs run f16 tensor cores against the CPU's f32 accumulation, which is
        // the regime `sd_unet_cuda` already sits in: measured 2.15e-4 with `attn_full`
        // and 2.19e-4 with the tensor-core path — i.e. the attention choice is not
        // what the residual is made of.
        try testing.expect(rel < 1e-3);
    }
}
