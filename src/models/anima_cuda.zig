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
//! Every block GEMM goes through `lin_cuda`, routed per weight, so a mixed checkpoint
//! that quantizes different linears in different blocks computes correctly. The AdaLN
//! pair is evaluated on the HOST, where `ops.matmul` handles convrot at any shape, which
//! is what lets the device path require `rows % 128 == 0` without special-casing its
//! 256/6144 widths.
//!
//! Head width is exactly 128, which both `launchHgemmB`'s P@V tiling and cuDNN handle
//! directly, so unlike the SD family there is no head padding. Every GEMM width (2048 /
//! 8192 / 1024) is a multiple of 128 and every reduction width a multiple of 32.

const std = @import("std");
const anima = @import("anima.zig");
const anima_dev = @import("anima_dev.zig");
const lin_cuda = @import("lin_cuda.zig");
const cuda = @import("tp_gpu").cuda;
const ops = @import("tp_ops");

const DiT = anima.DiT;
const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;
const Weight = ops.matmul.Weight;

/// Force the naive one-thread-per-(query, head) attention instead of the tensor-core
/// paths. For A/B and for reproducing a mismatch; `anima-cuda-test` runs both.
pub var force_naive_attn: bool = false;

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
    /// What `lin_cuda.plan` decided for this checkpoint's block linears.
    plan: lin_cuda.Plan,
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
            .plan = try lin_cuda.plan(model.device_lins, .{ .blockq = lin_cuda.blockq_gemm }, "anima cuda"),
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
            // Pre-size the decode scratches before any batch opens; one sizing serves the
            // cross-K/V batch below and every later forward.
            try lin_cuda.presize(be, self.plan, model.device_lins);
            // The patch embed runs on the fused `opMatmul`. The cross K/V below are
            // `ctx_seq` rows, the trunk's `seq`.
            try lin_cuda.attachLora(be, &self.plan, model.lora, model.device_lins, @max(seq, ctx_seq));

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
                const kv = try lin_cuda.prep(be, self.plan, cond_d, ctx_seq, cfg.context_dim, &.{ blk.cross_attn.k, blk.cross_attn.v });
                try lin_cuda.gemm(be, self.plan, kv, kv_out, blk.cross_attn.k, null);
                try lin_cuda.gemm(be, self.plan, kv, vv_out, blk.cross_attn.v, null);
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
    /// The storage of `x_d`, `nrm_d`, `dlt_d` and `mlp_d` (`anima_dev.streamDtFor`).
    act: cuda.backend.Dt,

    const fields = [_][]const u8{ "x_d", "nrm_d", "dlt_d", "q_d", "k_d", "v_d", "attn_d", "mlp_d", "mod_d", "patch_d" };

    pub fn init(be: *Backend, model: *const DiT, lat_h: usize, lat_w: usize, act: cuda.backend.Dt) !Workspace {
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
        self.act = act;
        inline for (.{ "x_d", "nrm_d", "dlt_d", "mlp_d" }) |name| @field(self, name).dt = act;
        return self;
    }

    pub fn deinit(self: *Workspace, be: *Backend) void {
        inline for (fields) |name| be.tensorDestroy(&@field(self, name));
        self.* = undefined;
    }
};

/// Whether the CUDA arms can run every block linear this model has. The refusal, if
/// any, is logged by name.
pub fn supported(model: *const DiT) bool {
    _ = lin_cuda.plan(model.device_lins, .{ .blockq = lin_cuda.blockq_gemm }, "anima cuda") catch return false;
    if (!lin_cuda.loraRunnable(model.lora, model.device_lins, "anima cuda")) return false;
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
    try anima_dev.body(Dev{ .model = model, .be = be, .sess = sess, .ws = ws }, model, sess.seq, cancel);
    try be.endBatch();
    // A 16-bit stream that overflowed can leave the output finite (`allFinite`).
    if (ws.act != .f32 and !try be.allFinite(ws.x_d, sess.seq * cfg.dim)) return error.NonFiniteActivation;

    const rows = try gpa.alloc(f32, sess.seq * cfg.dim);
    defer gpa.free(rows);
    try be.downloadF32(ws.x_d, rows);
    try model.finalize(io, gpa, out, rows, mod[cfg.n_layers * 9 * cfg.dim ..][0 .. 2 * cfg.dim], sess.lat_h, sess.lat_w);
}

/// `anima_dev`'s ops on this backend.
const Dev = struct {
    model: *const DiT,
    be: *Backend,
    sess: *Session,
    ws: *Workspace,

    fn scale(d: Dev) f32 {
        return 1.0 / @sqrt(@as(f32, @floatFromInt(d.model.cfg.headDim())));
    }

    /// `x_embedder` is one of the two weights the loader materializes to f32 (it feeds
    /// the fused `opMatmul`, which has no bf16 pipeline), so it takes the f32 arm.
    pub fn embed(d: Dev) !void {
        try lin_cuda.linear(d.be, d.sess.plan, d.ws.x_d, d.ws.patch_d, d.sess.seq, d.model.x_embedder, null);
    }

    /// Prefetch one block ahead, so each upload overlaps the previous block's compute;
    /// without it the first step pays the whole weight upload serially.
    pub fn blockBegin(d: Dev, bi: usize) !void {
        if (!d.be.async_uploads) return;
        if (bi == 0) prefetchBlock(d.be, d.model.blocks[0]);
        if (bi + 1 < d.model.blocks.len) prefetchBlock(d.be, d.model.blocks[bi + 1]);
    }

    pub fn lnMod(d: Dev, rows: usize, premul: usize, shift: usize) !void {
        const cfg = d.model.cfg;
        try d.be.lnMod(d.ws.x_d, d.ws.nrm_d, d.ws.mod_d, rows, cfg.dim, premul, shift, cfg.norm_eps);
    }

    pub fn prep(d: Dev, x: Buf, m: usize, cols: usize, group: []const Weight) !lin_cuda.Group {
        return lin_cuda.prep(d.be, d.sess.plan, x, m, cols, group);
    }

    pub fn gemms(d: Dev, g: lin_cuda.Group, ys: []const Buf, group: []const Weight) !void {
        for (ys, group) |y, w| try lin_cuda.gemm(d.be, d.sess.plan, g, y, w, null);
    }

    pub fn linear(d: Dev, y: Buf, x: Buf, m: usize, w: Weight) !void {
        const g = try lin_cuda.prep(d.be, d.sess.plan, x, m, w.cols, &.{w});
        try lin_cuda.gemm(d.be, d.sess.plan, g, y, w, null);
    }

    /// SPLIT-half RoPE (pairs `(i, i + 64)`), not the interleaved `be.rope`.
    pub fn qkNormRope(d: Dev, qnorm: []const f32, knorm: []const f32, seq: usize) !void {
        const cfg = d.model.cfg;
        const be = d.be;
        const ws = d.ws;
        const hd = cfg.headDim();
        const half = hd / 2;
        try be.qkNorm(ws.q_d, ws.q_d, try normBuf(be, qnorm), seq * cfg.n_heads, hd, cfg.qk_eps);
        try be.qkNorm(ws.k_d, ws.k_d, try normBuf(be, knorm), seq * cfg.n_heads, hd, cfg.qk_eps);
        try be.ropeHalf(ws.q_d, d.sess.freqs_d, seq, cfg.n_heads, half, seq * half, 0);
        try be.ropeHalf(ws.k_d, d.sess.freqs_d, seq, cfg.n_heads, half, seq * half, 0);
    }

    pub fn qNorm(d: Dev, qnorm: []const f32, seq: usize) !void {
        const cfg = d.model.cfg;
        try d.be.qkNorm(d.ws.q_d, d.ws.q_d, try normBuf(d.be, qnorm), seq * cfg.n_heads, cfg.headDim(), cfg.qk_eps);
    }

    pub fn selfAttn(d: Dev, seq: usize) !void {
        const cfg = d.model.cfg;
        const ws = d.ws;
        if (force_naive_attn) return d.be.attn(ws.q_d, ws.k_d, ws.v_d, ws.attn_d, seq, seq, cfg.n_heads, cfg.n_heads, cfg.headDim(), d.scale(), false);
        try d.be.opAttnTC(ws.q_d, ws.k_d, ws.v_d, ws.attn_d, seq, cfg.n_heads, cfg.n_heads, cfg.headDim(), d.scale());
    }

    pub fn crossAttn(d: Dev, bi: usize, seq: usize) !void {
        const cfg = d.model.cfg;
        const ws = d.ws;
        const off = bi * d.sess.ctx_seq * cfg.dim;
        const ck = d.sess.ck_d.viewF32(off);
        const cv = d.sess.cv_d.viewF32(off);
        if (force_naive_attn) return d.be.opAttnCrossNaive(ws.q_d, ck, cv, ws.attn_d, seq, d.sess.ctx_seq, cfg.n_heads, cfg.headDim(), d.scale());
        try d.be.opAttnTCRect(ws.q_d, ck, cv, ws.attn_d, seq, d.sess.ctx_seq, cfg.n_heads, cfg.n_heads, cfg.headDim(), d.scale());
    }

    pub fn gelu(d: Dev, n: usize) !void {
        try d.be.geluErf(d.ws.mlp_d, n);
    }

    pub fn gatedAdd(d: Dev, seq: usize, off: usize) !void {
        const dim = d.model.cfg.dim;
        try d.be.gatedAdd(d.ws.x_d, d.ws.dlt_d, d.ws.mod_d, seq * dim, dim, off);
    }
};

/// Queue a block's streamable weights for async prefetch, called ONE BLOCK AHEAD so
/// the upload overlaps the previous block's compute. Keys must be the same host
/// pointers the forward later fetches, or the prefetch is a cache miss and pure
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

/// Wrap a CPU norm-weight slice as a (pointer-cached) small device buffer.
/// `smallBuffer` returns a raw handle; the elt launchers want a `DeviceBuffer`.
fn normBuf(be: *Backend, weights: []const f32) !Buf {
    const h = try be.smallBuffer(std.mem.sliceAsBytes(weights));
    return .{ .buf = h, .size = weights.len * 4 };
}
