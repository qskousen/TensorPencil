//! GPU-resident Anima (`MiniTrainDIT` + `LLMAdapter`) forward on the CUDA backends
//! (`zig-cuda`'s hand-PTX and `cuda`'s vendor libraries), the CUDA twin of `anima_gpu`.
//!
//! The host/device split is identical to the Vulkan arm's; see `anima_gpu`'s header.
//! Three differences, all forced by the op surface:
//!
//! 1. Self-attention is `opAttnTC`: cuDNN's fused SDPA under `cuda`, the hand-PTX
//!    tensor-core path under `zig-cuda`.
//! 2. Cross-attention is `opAttnTCRect` directly rather than through `opAttnCross`,
//!    so both arms run the same hand-PTX kernels on Anima's `seq x 512` shape.
//!    `opAttnCross` would send the `cuda` arm to cuDNN instead. It is ~8% of the
//!    step's FLOPs.
//! 3. The per-image cross-attention K/V cache is f32 here, f16 on Vulkan: the CUDA
//!    attention entry points take f32 and gather/convert internally, so there is nothing
//!    to pre-convert into. 235 MB against Vulkan's 117 MB, and the per-step
//!    re-conversion it costs is well under a millisecond.
//!
//! int8/int4 convrot runs here with the kind resolved PER BLOCK (`kindOf` per linear,
//! `prepGroup` per shared activation). A real mixed checkpoint quantizes different
//! linears in different blocks, so `dit_cuda`'s one-value-per-model `LinKind` would be
//! wrong for at least one of them. Cross-attention's k/v and the AdaLN pair may be
//! quantized too but are evaluated on the HOST, where `ops.matmul` handles convrot at
//! any shape, which is what lets the device path require `rows % 128 == 0` without
//! special-casing their 256/6144 widths.
//!
//! Head width is exactly 128, which both `launchHgemmB`'s P@V tiling and cuDNN handle
//! directly, so unlike the SD family there is no head padding. Every GEMM width (2048 /
//! 8192 / 1024) is a multiple of 128 and every reduction width a multiple of 32.

const std = @import("std");
const anima = @import("anima.zig");
const cuda = @import("tp_gpu").cuda;
const ops = @import("tp_ops");

const DiT = anima.DiT;
const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;
const Weight = ops.matmul.Weight;

/// Force the naive one-thread-per-(query, head) attention instead of the tensor-core
/// paths. For A/B and for reproducing a mismatch; `anima-cuda-test` runs both.
pub var force_naive_attn: bool = false;

/// Widest zero bias any Anima GEMM needs (the MLP's `mlp_dim`). Passed WHOLE,
/// never sliced: a bias is cached by host pointer and sized from the first call's
/// length, so a narrow layer seen first would leave every wider one reading past the
/// end. `opGemmBf16` asserts `bias.len >= co` and the kernels read only `co` entries.
const zero_bias: [anima.anima_2b.mlp_dim]f32 = @splat(0);

/// Per-image cache: everything constant across sampling steps.
pub const Session = struct {
    cfg: anima.Config,
    lat_h: usize,
    lat_w: usize,
    /// Image tokens, `(lat_h/patch) * (lat_w/patch)`.
    seq: usize,
    ctx_seq: usize,

    /// 3-axis RoPE table for the image grid: `cos` then `sin`, `seq * half` each.
    freqs_d: Buf,
    /// Cross-attention K and V for EVERY block, `[n_layers][ctx_seq][dim]` f32,
    /// projections of the adapter's output, which no step changes. See the header.
    ck_d: Buf,
    cv_d: Buf,

    /// The schedule this session precomputed for, and the FOLDED per-sigma modulation
    /// tables (see `anima.foldModulationTable`).
    sigmas: []f32,
    /// Borrowed, model-owned: the FOLDED table for every scheduled sigma, stride
    /// `mod_stride`. Per SIGMA, not per conditioning, see
    /// `anima.DiT.modulationSchedule` for why building it per session cost 0.8 s an image.
    mods: []const f32,
    mod_stride: usize,

    pub fn init(
        gpa: std.mem.Allocator,
        be: *Backend,
        model: *const DiT,
        lat_h: usize,
        lat_w: usize,
        cond: []const f32,
        ctx_seq: usize,
        sigmas: []const f32,
        /// Model-owned, shared by both conditioning branches (`modulationSchedule`).
        mods: []const f32,
    ) !Session {
        const cfg = model.cfg;
        std.debug.assert(cond.len == ctx_seq * cfg.context_dim);
        std.debug.assert(lat_h % cfg.patch == 0 and lat_w % cfg.patch == 0);
        const h = lat_h / cfg.patch;
        const w = lat_w / cfg.patch;
        const seq = h * w;
        const half = cfg.headDim() / 2;
        const d = cfg.dim;

        var self: Session = .{
            .cfg = cfg,
            .lat_h = lat_h,
            .lat_w = lat_w,
            .seq = seq,
            .ctx_seq = ctx_seq,
            .freqs_d = undefined,
            .ck_d = undefined,
            .cv_d = undefined,
            .sigmas = &.{},
            .mods = mods,
            .mod_stride = anima.modulationTableLen(cfg),
        };
        var made: usize = 0;
        errdefer {
            const bufs = [_]*Buf{ &self.freqs_d, &self.ck_d, &self.cv_d };
            for (bufs[0..made]) |b| be.tensorDestroy(b);
            if (self.sigmas.len != 0) gpa.free(self.sigmas);
        }

        {
            var f = try model.ropeFreqs(gpa, h, w);
            defer f.deinit(gpa);
            const flat = try gpa.alloc(f32, 2 * seq * half);
            defer gpa.free(flat);
            @memcpy(flat[0 .. seq * half], f.cos);
            @memcpy(flat[seq * half ..], f.sin);
            self.freqs_d = try be.tensorCreate(flat.len * 4);
            made += 1;
            try be.tensorUpload(self.freqs_d, std.mem.sliceAsBytes(flat));
        }

        // Cross-attention K/V for every block, ON THE DEVICE.
        //
        // Running these on the host costs seconds per image: 28 blocks x 2 GEMMs of
        // `[512, 2048] x [2048, 1024]` is ~120 GFLOP per conditioning, doubled under
        // CFG, measured 1.24 s at 512x768, repeated for every image in a queue. As a
        // share of a STEP that is 0.56%, which is what makes the host version look
        // cheap. A cost that is negligible per step is not therefore negligible per
        // image, and the two framings
        // needed separate measurements and only one was taken.
        //
        // The trade the old comment named is real but small: the cross `k`/`v` weights now
        // get uploaded (~235 MB) where before only the host read them. They are uploaded
        // once and cached, against 1.2 s of host GEMM per image.
        {
            // Pre-size the W4A8 decode scratch to the model's widest linear before any
            // batch opens. Growth is safe (`ensureDeviceBuffer` syncs the stream first)
            // but the first block would otherwise pay several syncs for nothing. One
            // sizing serves the cross-K/V batch below and every later forward, since it
            // is the model-wide maximum.
            const need = anima.maxW4A8Scratch(model, Backend.w4a8ScratchBytes);
            if (need > 0) try be.ensureDeviceBuffer(&be.w4a8_i8, need);

            var cond_d = try be.tensorCreate(cond.len * 4);
            defer be.tensorDestroy(&cond_d);
            try be.tensorUpload(cond_d, std.mem.sliceAsBytes(cond));
            self.ck_d = try be.tensorCreate(cfg.n_layers * ctx_seq * d * 4);
            made += 1;
            self.cv_d = try be.tensorCreate(cfg.n_layers * ctx_seq * d * 4);
            made += 1;

            try be.beginBatch();
            errdefer if (be.batching()) be.abortBatch();
            for (model.blocks, 0..) |*blk, bi| {
                const off = bi * ctx_seq * d;
                const kv_out = self.ck_d.viewF32(off);
                const vv_out = self.cv_d.viewF32(off);
                // k and v share the context activation, so one prep serves both.
                try prepGroup(be, cond_d, ctx_seq, cfg.context_dim, &.{ blk.cross_attn.k, blk.cross_attn.v });
                try lin(be, kv_out, cond_d, ctx_seq, blk.cross_attn.k);
                try lin(be, vv_out, cond_d, ctx_seq, blk.cross_attn.v);
                // K is normed, V is NOT (`v_norm = nn.Identity()`), the asymmetry
                // `DiT.projectKv` owns on the host path.
                try be.qkNorm(kv_out, kv_out, try normBuf(be, blk.cross_attn.knorm), ctx_seq * cfg.n_heads, cfg.headDim(), cfg.qk_eps);
            }
            try be.endBatch();
        }

        std.debug.assert(mods.len == sigmas.len * self.mod_stride);
        self.sigmas = try gpa.dupe(f32, sigmas);
        return self;
    }

    pub fn deinit(self: *Session, gpa: std.mem.Allocator, be: *Backend) void {
        be.tensorDestroy(&self.freqs_d);
        be.tensorDestroy(&self.ck_d);
        be.tensorDestroy(&self.cv_d);
        gpa.free(self.sigmas);
        self.* = undefined;
    }

    fn tv(self: *const Session, sigma: f32) ?[]const f32 {
        for (self.sigmas, 0..) |s, i| {
            if (s == sigma) return self.mods[i * self.mod_stride ..][0..self.mod_stride];
        }
        return null;
    }
};

/// Per-resolution device scratch, shared by both conditioning branches.
pub const Workspace = struct {
    x_d: Buf, // [seq][dim] the running activation
    nrm_d: Buf, // [seq][dim] modulated pre-norm
    dlt_d: Buf, // [seq][dim] sublayer output
    q_d: Buf,
    k_d: Buf, // self-attention only
    v_d: Buf, // self-attention only
    attn_d: Buf,
    mlp_d: Buf, // [seq][mlp_dim]
    mod_d: Buf, // this step's whole folded modulation table
    patch_d: Buf, // [seq][patchDim] raw patches

    const fields = [_][]const u8{ "x_d", "nrm_d", "dlt_d", "q_d", "k_d", "v_d", "attn_d", "mlp_d", "mod_d", "patch_d" };

    pub fn init(be: *Backend, model: *const DiT, lat_h: usize, lat_w: usize) !Workspace {
        const cfg = model.cfg;
        const seq = (lat_h / cfg.patch) * (lat_w / cfg.patch);
        const d = cfg.dim;
        // Every activation buffer is 128-ROW PADDED, because a quantized GEMM writes
        // `i8_mpad` rows, not `m`. `opI8Prep` pads the activation up to a multiple of 128
        // and `opI8Gemm` launches `grid.y = i8_mpad / 128`, so its output covers the padded
        // row count. At the validator's 192-token latent that is 256 rows into a 192-row
        // buffer, a 25% overrun, and `compute-sanitizer` named it as
        // `igemm_pipe_fused` writing 1 byte past a 1,572,864-byte allocation. The dense
        // path writes exactly `m` rows, so only the quantized path exposes it.
        const seq_pad = std.mem.alignForward(usize, seq, 128);
        const sizes = [fields.len]usize{
            seq_pad * d * 4,
            seq_pad * d * 4,
            seq_pad * d * 4,
            seq_pad * d * 4,
            seq_pad * d * 4,
            seq_pad * d * 4,
            seq_pad * d * 4,
            seq_pad * cfg.mlp_dim * 4,
            anima.modulationTableLen(cfg) * 4,
            seq * cfg.patchDim() * 4,
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

/// Whether this backend can run Anima's block GEMMs. The trunk weights are dense bf16
/// in every checkpoint seen so far; fp8 and f32 also have paths. Anything else (int8/
/// int4 convrot, ggml block quants) has no CUDA GEMM here and must stay on the CPU
/// rather than being read as the wrong dtype.
pub fn supported(model: *const DiT) bool {
    if (model.blocks.len == 0) return false;
    // EVERY block's linears, not block 0's, see `anima.unsupportedLin`. This arm HAS
    // int8/int4 convrot (`opI8Prep`/`opI8Gemm`), which the Vulkan one does not, so the
    // support set is passed rather than assumed.
    // Both CUDA arms have int8 AND int4 convrot, plus the W4A8 and NVFP4 decode kernels.
    if (anima.unsupportedLin(model, .{ .i8 = true, .i4 = true, .w4a8 = true, .nvfp4 = true }) != null) return false;
    // The W4A8 decode kernel reads four packed bytes per thread as one `u32`, so a
    // group must not straddle that word. Checked here rather than asserted in `lin`:
    // a checkpoint we cannot run belongs on the CPU, not in a panic.
    if (anima.w4a8SmallGroup(model)) |tag| {
        std.log.err("anima_cuda: {s} is W4A8 with a group_size that is not a multiple of 8; " ++
            "the CUDA decode kernel needs one (Vulkan's is general)", .{tag});
        return false;
    }
    return true;
}

const LinKind = anima.LinKind;
const kindOf = anima.linKind;

/// Quantize+rotate the activation `x` once for a group of GEMMs that share it, if any
/// member of the group needs it.
///
/// `opI8Prep` does NOT overwrite `x`, it writes int8 rows and per-row scales into
/// the backend's own `i8_x`/`i8_scale` and records `i8_cols`. So a dense GEMM in the same
/// group is free to read the f32 `x` afterwards, and the only real constraint is that the
/// prep happens before the quant GEMMs and that `cols` matches (which it does: a group
/// shares one activation, hence one reduction width).
///
/// The prep state is global and each call replaces it, so every group that contains a
/// quantized linear pays exactly one prep.
fn prepGroup(be: *Backend, x: Buf, m: usize, cols: usize, group: []const Weight) !void {
    // Grouped by the PREP a kind needs, not by the kind: int8 and W4A8 share one
    // (`anima.prepKind` says why), so a block mixing them pays a single `opI8Prep`.
    var want: anima.PrepKind = .none;
    for (group) |w| {
        const k = anima.prepKind(kindOf(w));
        if (k == .none) continue;
        // int8 and int4 in ONE group would need two preps of the same activation
        // and two live prep states, which the backend does not have. No checkpoint
        // does this; refuse loudly rather than silently use the wrong scale set.
        if (want != .none and want != k) {
            std.log.err("anima_cuda: a linear group mixes {t} and {t} activation preps; one cannot serve both", .{ want, k });
            return error.UnsupportedCheckpoint;
        }
        want = k;
    }
    switch (want) {
        .i8 => try be.opI8Prep(x, m, cols, false),
        .i4 => try be.opI4Prep(x, m, cols),
        .none => {},
    }
}

/// One block linear `y[m][w.rows] = x[m][w.cols] @ Wᵀ`. A quantized weight reads the
/// prep state (so `x` is unused for it) and fuses the per-row rescale; a dense one takes
/// the tensor-core path in `gemm`.
///
/// `opI8Gemm`/`opI4Gemm` launch `grid.x = rows / 128`, so a quantized linear needs
/// `rows % 128 == 0`. Every one the device runs here is 2048 or 8192; the odd widths
/// (the AdaLN pair's 256 and 6144, and cross-attention's k/v) are evaluated on the HOST,
/// where `ops.matmul` handles convrot regardless of shape.
fn lin(be: *Backend, y: Buf, x: Buf, m: usize, w: Weight) !void {
    switch (kindOf(w)) {
        .i8 => {
            std.debug.assert(w.rows % 128 == 0);
            try be.opI8Gemm(y, w.bytes, w.row_scale.?, w.rows, false);
        },
        .i4 => {
            std.debug.assert(w.rows % 128 == 0);
            try be.opI4Gemm(y, w.bytes, w.row_scale.?, w.rows);
        },
        // W4A8 reads the SAME prep state and runs the SAME int8 GEMM as `.i8` above; the
        // only difference is that the packed 4-bit weight is decoded into a device scratch
        // on the way in, so the 4-bit form stays resident (see `quant_weight.w4a8`).
        .w4a8 => {
            std.debug.assert(w.rows % 128 == 0);
            const meta = w.w4a8.?;
            try be.opI8GemmW4A8(y, w.bytes, meta.s_rel, std.mem.asBytes(meta.levels), w.row_scale.?, w.rows, w.cols, meta.group_size, false);
        },
        // Weight-only NVFP4: the 4-bit weight is decoded to an f16 scratch inside the
        // GEMM and the packed form stays resident. `rows % 128` / `cols % 32` come from
        // the f16 GEMM it feeds; every NVFP4 layer in the shipped checkpoints satisfies
        // both (audited across all three families).
        .nvfp4 => {
            std.debug.assert(w.rows % 128 == 0 and w.cols % 32 == 0);
            const meta = w.nvfp4.?;
            try be.opMatmulNvfp4(y, x, m, w.bytes, meta.scales, std.mem.asBytes(&meta.levels.bf16v), w.rows, w.cols, &zero_bias);
        },
        .dense => try gemm(be, y, x, m, w),
    }
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
    const d = cfg.dim;
    const seq = sess.seq;
    const hd = cfg.headDim();
    const attn_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));

    std.debug.assert(x_lat.len == cfg.channels * sess.lat_h * sess.lat_w);
    std.debug.assert(out.len == cfg.out_channels * sess.lat_h * sess.lat_w);

    var owned: ?[]f32 = null;
    defer if (owned) |b| gpa.free(b);
    const mod = sess.tv(sigma) orelse blk: {
        const tbl = try model.modulationTable(io, gpa, sigma);
        anima.foldModulationTable(cfg, tbl);
        owned = tbl;
        break :blk tbl;
    };

    const patches = try anima.patchify(gpa, cfg, x_lat, sess.lat_h, sess.lat_w);
    defer gpa.free(patches);

    try be.tensorUpload(ws.mod_d, std.mem.sliceAsBytes(mod));
    try be.tensorUpload(ws.patch_d, std.mem.sliceAsBytes(patches));

    try be.beginBatch();
    errdefer if (be.batching()) be.abortBatch();

    // `x_embedder` is one of the two weights the loader materializes to f32 (it feeds
    // the fused `opMatmul`, which has no bf16 pipeline), so it takes the f32 arm.
    try be.opMatmul(ws.x_d, 0, ws.patch_d, 0, seq, model.x_embedder.bytes, model.x_embedder.dtype == .f8_e4m3, d, cfg.patchDim(), model.x_embedder.scale, null);

    // Prefetch one block ahead throughout, so each block's upload overlaps the
    // previous block's compute. Without it the first step pays the whole weight
    // upload serially, `zimage_cuda` measured 8.0 s for step 1 against a 2.6 s
    // steady state at 1056x1584, which on a short render is a fifth of the total.
    if (be.async_uploads and model.blocks.len > 0) prefetchBlock(be, model.blocks[0]);
    for (model.blocks, 0..) |*blk, bi| {
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        if (be.async_uploads and bi + 1 < model.blocks.len) prefetchBlock(be, model.blocks[bi + 1]);
        try blockForward(be, sess, cfg, blk, ws, bi, bi * 9 * d, attn_scale);
    }
    try be.endBatch();

    // --- the final layer, on the host ------------------------------------------
    const rows = try gpa.alloc(f32, seq * d);
    defer gpa.free(rows);
    try be.tensorDownload(ws.x_d, std.mem.sliceAsBytes(rows));
    try model.finalize(io, gpa, out, rows, mod[cfg.n_layers * 9 * d ..][0 .. 2 * d], sess.lat_h, sess.lat_w);
}

fn blockForward(
    be: *Backend,
    sess: *Session,
    cfg: anima.Config,
    blk: anytype,
    ws: *Workspace,
    bi: usize,
    mod_base: usize,
    attn_scale: f32,
) !void {
    const d = cfg.dim;
    const seq = sess.seq;
    const heads = cfg.n_heads;
    const hd = cfg.headDim();
    const half = hd / 2;

    // --- self-attention --------------------------------------------------------
    // The pre-norm's scale block carries the `1 +` already (`foldModulationTable`).
    try be.lnMod(ws.x_d, ws.nrm_d, ws.mod_d, seq, d, mod_base + d, mod_base, cfg.norm_eps);
    // q/k/v share one activation, so one prep serves all three.
    try prepGroup(be, ws.nrm_d, seq, d, &.{ blk.self_attn.q, blk.self_attn.k, blk.self_attn.v });
    try lin(be, ws.q_d, ws.nrm_d, seq, blk.self_attn.q);
    try lin(be, ws.k_d, ws.nrm_d, seq, blk.self_attn.k);
    try lin(be, ws.v_d, ws.nrm_d, seq, blk.self_attn.v);

    // Anima's Q/K norms take the BLOCKS' 1e-6, not `finfo(f32).eps`, unlike
    // Z-Image, whose `RMSNorm(head_dim)` is built with no `eps` at all.
    try be.qkNorm(ws.q_d, ws.q_d, try normBuf(be, blk.self_attn.qnorm), seq * heads, hd, cfg.qk_eps);
    try be.qkNorm(ws.k_d, ws.k_d, try normBuf(be, blk.self_attn.knorm), seq * heads, hd, cfg.qk_eps);
    // SPLIT-half RoPE (pairs `(i, i + 64)`), not the interleaved `be.rope` that
    // Z-Image and krea2 use.
    try be.ropeHalf(ws.q_d, sess.freqs_d, seq, heads, half, seq * half, 0);
    try be.ropeHalf(ws.k_d, sess.freqs_d, seq, heads, half, seq * half, 0);

    if (force_naive_attn) {
        try be.attn(ws.q_d, ws.k_d, ws.v_d, ws.attn_d, seq, seq, heads, heads, hd, attn_scale, false);
    } else {
        try be.opAttnTC(ws.q_d, ws.k_d, ws.v_d, ws.attn_d, seq, heads, heads, hd, attn_scale);
    }
    try prepGroup(be, ws.attn_d, seq, d, &.{blk.self_attn.out});
    try lin(be, ws.dlt_d, ws.attn_d, seq, blk.self_attn.out);
    try be.gatedAdd(ws.x_d, ws.dlt_d, ws.mod_d, seq * d, d, mod_base + 2 * d);

    // --- cross-attention onto the adapter's output ------------------------------
    // No RoPE here at all, and K/V come from the session's per-image cache rather
    // than from two GEMMs, see the module header.
    try be.lnMod(ws.x_d, ws.nrm_d, ws.mod_d, seq, d, mod_base + 4 * d, mod_base + 3 * d, cfg.norm_eps);
    try prepGroup(be, ws.nrm_d, seq, d, &.{blk.cross_attn.q});
    try lin(be, ws.q_d, ws.nrm_d, seq, blk.cross_attn.q);
    try be.qkNorm(ws.q_d, ws.q_d, try normBuf(be, blk.cross_attn.qnorm), seq * heads, hd, cfg.qk_eps);
    {
        const off = bi * sess.ctx_seq * d;
        const ck = sess.ck_d.viewF32(off);
        const cv = sess.cv_d.viewF32(off);
        if (force_naive_attn) {
            try be.opAttnCrossNaive(ws.q_d, ck, cv, ws.attn_d, seq, sess.ctx_seq, heads, hd, attn_scale);
        } else {
            try be.opAttnTCRect(ws.q_d, ck, cv, ws.attn_d, seq, sess.ctx_seq, heads, heads, hd, attn_scale);
        }
    }
    try prepGroup(be, ws.attn_d, seq, d, &.{blk.cross_attn.out});
    try lin(be, ws.dlt_d, ws.attn_d, seq, blk.cross_attn.out);
    try be.gatedAdd(ws.x_d, ws.dlt_d, ws.mod_d, seq * d, d, mod_base + 5 * d);

    // --- MLP -------------------------------------------------------------------
    try be.lnMod(ws.x_d, ws.nrm_d, ws.mod_d, seq, d, mod_base + 7 * d, mod_base + 6 * d, cfg.norm_eps);
    try prepGroup(be, ws.nrm_d, seq, d, &.{blk.mlp1});
    try lin(be, ws.mlp_d, ws.nrm_d, seq, blk.mlp1);
    // `nn.GELU()` with the default `approximate='none'`, the erf form, not tanh.
    try be.geluErf(ws.mlp_d, seq * cfg.mlp_dim);
    // Its own prep: the reduction width here is `mlp_dim`, not `dim`, and the prep
    // state records ONE `cols`.
    try prepGroup(be, ws.mlp_d, seq, cfg.mlp_dim, &.{blk.mlp2});
    try lin(be, ws.dlt_d, ws.mlp_d, seq, blk.mlp2);
    try be.gatedAdd(ws.x_d, ws.dlt_d, ws.mod_d, seq * d, d, mod_base + 8 * d);
}

/// Queue a block's streamable weights for async prefetch, called ONE BLOCK AHEAD so
/// the upload overlaps the previous block's compute. Keys must be the same host
/// pointers `blockForward` later fetches, or the prefetch is a cache miss and pure
/// waste.
///
/// `cross_attn.k`/`.v` are deliberately absent: `Session.init` consumes them ONCE per
/// image (now on the device), and the step loop never touches them again, so prefetching
/// them here would re-upload ~235 MB per step that nothing reads.
fn prefetchBlock(be: *Backend, blk: anytype) void {
    const bytes = std.mem.sliceAsBytes;
    inline for (.{ blk.self_attn.q, blk.self_attn.k, blk.self_attn.v, blk.self_attn.out }) |w| be.prefetchWeight(w.bytes);
    be.prefetchWeight(bytes(blk.self_attn.qnorm));
    be.prefetchWeight(bytes(blk.self_attn.knorm));
    inline for (.{ blk.cross_attn.q, blk.cross_attn.out }) |w| be.prefetchWeight(w.bytes);
    be.prefetchWeight(bytes(blk.cross_attn.qnorm));
    inline for (.{ blk.mlp1, blk.mlp2 }) |w| be.prefetchWeight(w.bytes);
}

/// A block GEMM, dispatched by weight dtype, the same routing `zimage_cuda.gemm`
/// uses. Ampere+ feeds raw bf16 straight to the tensor cores; older cards take the
/// GPU-side bf16->f16 GEMM.
fn gemm(be: *Backend, y: Buf, x: Buf, m: usize, w: Weight) !void {
    const zeros: []const f32 = &zero_bias;
    std.debug.assert(w.rows <= zeros.len);
    switch (w.dtype) {
        // `null`, not `zeros`: Anima's block linears are all bias-free, and a null
        // bias lets the `.libs` arm write the GEMM straight into `y` instead of staging
        // through `conv_c` and re-reading the whole output to add zero.
        .bf16 => if (be.ctx.cc_major >= 8 and w.rows % 128 == 0 and w.cols % 32 == 0)
            try be.opGemmBf16(y, x, m, w.bytes, w.rows, w.cols, null)
        else
            try be.opMatmulBf16(y, x, m, w.bytes, w.rows, w.cols, zeros, false, false),
        .f16 => try be.opMatmulF16(y, x, m, w.bytes, w.rows, w.cols, null, false, false),
        .f8_e4m3 => try be.opMatmulFp8(y, x, m, w.bytes, w.scale, w.rows, w.cols),
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
