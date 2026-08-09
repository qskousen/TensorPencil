//! GPU-resident Krea 2 DiT forward: the whole 28-block chain runs on the
//! device (GEMMs, norms, modulation, RoPE, attention, gating) with a single
//! upload of the combined sequence going in and a single download of the
//! image tokens coming out. Small paths (timestep MLPs, txtfusion, patchify,
//! final layer) stay on the CPU where they are cheap.
//!
//! Numerics match the CPU forward up to floating-point reordering; parity is
//! tested against `DiT.forward` (gated on testdata/gpu-tests).

const std = @import("std");
const dit = @import("dit.zig");
const gpu = @import("tp_gpu").context;
const ops = @import("tp_ops");

const DiT = dit.DiT;

/// When true, each forward prints a per-category time breakdown (sync-per-op
/// submission makes host-side timing exact).
pub var profile: bool = false;

/// When true, force the full-f32 path (opMatmul GEMMs + eltwise f32
/// attention) even where the tensor-core / coop pipelines exist, the
/// hardware-fallback path, exposed for A/B (speed vs f16 rounding). Must be
/// set before Workspace.init, which sizes its buffers from the same choice.
pub var force_f32: bool = false;

const Prof = struct {
    matmul_ns: i96 = 0,
    attn_ns: i96 = 0,
    scores_ns: i96 = 0,
    smax_ns: i96 = 0,
    aout_ns: i96 = 0,
    elt_ns: i96 = 0,
    prep_ns: i96 = 0, // int8 rotate/rowscale/quantize prep (opI8Prep)
    // Packed W4A8 -> k-major int8 decode (w4a8_decode_t), its own category because it
    // is a per-GEMM cost that buys VRAM and has a bandwidth ceiling to be judged
    // against, folded into `matmul` it would just look like a slower GEMM.
    w4a8_ns: i96 = 0,
    xfer_ns: i96 = 0,
    cpu_ns: i96 = 0,
};
const F = dit.features;
const heads = dit.n_heads;
const kv_heads = dit.n_kv_heads;
const hd = dit.head_dim;
const attn_scale: f32 = 1.0 / 11.313708498984761; // 1/sqrt(128)

/// Flash attention (coopmat.buildFlashAttn) keeps the {m, 1/d} table in the
/// tail of attn_d and never materializes S, but measured SLOWER than the
/// two-pass path at DiT sizes (attn 1.49 -> 2.43 s at 1120x1680): the out
/// pass recomputes S at the global-fragment-load rate (~0.8 s), which costs
/// more than the coalesced S write + reads it eliminates (~0.5 s). It would
/// win only with both operands staged in shared, which doesn't fit the 48 KB
/// workgroup ceiling next to the S tile at this tiling. Kept behind this
/// switch (kernels are unit-tested) for smaller-seq experiments or a future
/// retiling.
const use_flash = false;
/// Full-width zero bias for the dense-bf16 block linears (they have none, but the
/// f16-weight coop GEMM always folds a bias in). One stable file-scope pointer so
/// every GEMM width shares a single cached buffer.
const zero_bias: [dit.mlp_dim]f32 = @splat(0);
/// Two-pass softmax / parallel rmsnorm chunk counts: 32 interleaved chunks
/// per row so a warp covers a row with coalesced reads.
const nchunks = 32;
const rms_ch = 32;
/// Cap on the materialized attention-scores buffer; heads batch to fit it.
const s_bytes_cap: usize = 2 << 30;

/// Per-sampling-run cache: everything constant across steps, the text
/// fusion tokens (a full CPU transformer pass), the rope table, and the
/// timestep vectors for the whole schedule, computed once and, where
/// device-resident, uploaded once.
pub const Session = struct {
    seq_txt: usize,
    lat_h: usize,
    lat_w: usize,
    /// Text tokens uploaded once; copied device-to-device into x each step.
    txt0_d: gpu.DeviceBuffer,
    txt_len: usize,
    freqs_d: gpu.DeviceBuffer,
    /// Per schedule entry: t (F) then tvec (6F).
    sigmas: []f32,
    tvs: []f32,

    pub fn init(
        gpa: std.mem.Allocator,
        io: std.Io,
        ctx: *gpu.Context,
        model: *const DiT,
        lat_h: usize,
        lat_w: usize,
        cond: []const f32,
        seq_txt: usize,
        sigmas: []const f32,
    ) !Session {
        const h = lat_h / dit.patch;
        const w = lat_w / dit.patch;
        const seq = seq_txt + h * w;
        const half = hd / 2;

        const txt_tokens = try model.textTokens(io, gpa, cond, seq_txt);
        defer gpa.free(txt_tokens);
        var txt0_d = try ctx.tensorCreate(txt_tokens.len * 4);
        errdefer ctx.tensorDestroy(&txt0_d);
        try ctx.tensorUpload(txt0_d, std.mem.sliceAsBytes(txt_tokens));

        var freqs = try DiT.ropeFreqs(gpa, seq_txt, h, w);
        defer freqs.deinit(gpa);
        std.debug.assert(freqs.half == half);
        const fp = try gpa.alloc(f32, 2 * seq * half);
        defer gpa.free(fp);
        @memcpy(fp[0 .. seq * half], freqs.cos);
        @memcpy(fp[seq * half ..], freqs.sin);
        var freqs_d = try ctx.tensorCreate(fp.len * 4);
        errdefer ctx.tensorDestroy(&freqs_d);
        try ctx.tensorUpload(freqs_d, std.mem.sliceAsBytes(fp));

        const sig = try gpa.dupe(f32, sigmas);
        errdefer gpa.free(sig);
        const tvs = try gpa.alloc(f32, sigmas.len * 7 * F);
        errdefer gpa.free(tvs);
        for (sigmas, 0..) |sg, i| {
            const tvv = try model.timestepVectors(io, gpa, sg);
            defer gpa.free(tvv.t);
            defer gpa.free(tvv.tvec);
            @memcpy(tvs[i * 7 * F ..][0..F], tvv.t);
            @memcpy(tvs[i * 7 * F + F ..][0 .. 6 * F], tvv.tvec);
        }

        return .{
            .seq_txt = seq_txt,
            .lat_h = lat_h,
            .lat_w = lat_w,
            .txt0_d = txt0_d,
            .txt_len = txt_tokens.len,
            .freqs_d = freqs_d,
            .sigmas = sig,
            .tvs = tvs,
        };
    }

    pub fn deinit(self: *Session, gpa: std.mem.Allocator, ctx: *gpu.Context) void {
        ctx.tensorDestroy(&self.txt0_d);
        ctx.tensorDestroy(&self.freqs_d);
        gpa.free(self.sigmas);
        gpa.free(self.tvs);
        self.* = undefined;
    }

    fn tv(self: *const Session, sigma: f32) ?struct { t: []const f32, tvec: []const f32 } {
        for (self.sigmas, 0..) |sg, i| {
            if (sg == sigma) {
                return .{
                    .t = self.tvs[i * 7 * F ..][0..F],
                    .tvec = self.tvs[i * 7 * F + F ..][0 .. 6 * F],
                };
            }
        }
        return null;
    }
};

/// Accumulate the time since `m` into `bucket` and restamp. Only meaningful when the
/// forward is NOT batching (`--profile on`), where each op is submit-and-wait; inside a
/// recording batch it would time recording, not execution.
fn lapNs(io_: std.Io, m: *std.Io.Timestamp, bucket: *i96) void {
    const now = std.Io.Clock.real.now(io_);
    bucket.* += now.nanoseconds - m.nanoseconds;
    m.* = now;
}

/// One int8-convrot GEMM, dispatching on how the weight is STORED: plain int8 goes
/// straight to the GEMM, a packed W4A8 weight is decoded (and k-major transposed) into
/// the context's transient scratch first. Both then run the identical int8 kernel,
/// W4A8's "A8" is precisely that the activation stays 8-bit.
///
/// Per weight rather than per model, so a checkpoint that mixes the two storage forms
/// computes correctly even where `is_i8` is decided from one block.
fn i8GemmW(io: std.Io, prof: *Prof, t_mark: *std.Io.Timestamp, ctx: *gpu.Context, y: gpu.DeviceBuffer, w: ops.matmul.Weight, c_h16: bool) !void {
    if (w.dtype == .w4a8) {
        const meta = w.w4a8.?;
        const wbuf = try ctx.w4a8Decode(
            w.bytes,
            meta.s_rel,
            std.mem.asBytes(meta.levels),
            w.rows,
            w.cols,
            meta.group_size,
        );
        lapNs(io, t_mark, &prof.w4a8_ns);
        return ctx.opI8GemmBuf(y, wbuf, &.{}, w.row_scale.?, w.rows, c_h16);
    }
    return ctx.opI8Gemm(y, w.bytes, w.row_scale.?, w.rows, c_h16);
}

/// Per-run device activation buffers, allocated once and reused every step. Allocating
/// per forward means ~20 create/destroy pairs per call, including the ~2 GiB scores
/// buffer, and each `tensorCreate` is a raw `vkAllocateMemory`.
/// Sized for a maximum sequence; one Workspace is shared by the positive and
/// negative CFG sessions (they differ only in text length). Every kernel
/// writes the regions it later reads (including zero padding), so reuse
/// across steps and sessions needs no clearing.
pub const Workspace = struct {
    seq_cap: usize,
    x_d: gpu.DeviceBuffer,
    t1_d: gpu.DeviceBuffer,
    q_d: gpu.DeviceBuffer,
    k_d: gpu.DeviceBuffer,
    v_d: gpu.DeviceBuffer,
    g_d: gpu.DeviceBuffer,
    attn_d: gpu.DeviceBuffer,
    mg_d: gpu.DeviceBuffer,
    mu_d: gpu.DeviceBuffer,
    mv_d: gpu.DeviceBuffer,
    fin_d: gpu.DeviceBuffer,
    imgin_d: gpu.DeviceBuffer,
    qt_d: gpu.DeviceBuffer,
    kt_d: gpu.DeviceBuffer,
    s_d: gpu.DeviceBuffer,
    v16_d: gpu.DeviceBuffer,
    part_d: gpu.DeviceBuffer,
    md_d: gpu.DeviceBuffer,
    rmsp_d: gpu.DeviceBuffer,
    rmsi_d: gpu.DeviceBuffer,
    h16_d: gpu.DeviceBuffer,

    pub fn init(ctx: *gpu.Context, lat_h: usize, lat_w: usize, seq_txt_cap: usize) !Workspace {
        const n_img = (lat_h / dit.patch) * (lat_w / dit.patch);
        const seq = seq_txt_cap + n_img;
        const seq_pad = std.mem.alignForward(usize, seq, 128);
        const coop = !force_f32 and ctx.pipe_coop != .null_handle;
        const tc_attn = !force_f32 and ctx.pipe_scores != .null_handle;
        const flash = use_flash and ctx.pipe_flash_md != .null_handle;
        const s_rows = if (tc_attn) seq_pad else seq;
        const s_esize: usize = if (tc_attn) 2 else 4;
        const hpb = headsPerBatch(s_rows, s_esize, scoresCap(ctx.budget_override), null);

        var ws: Workspace = undefined;
        ws.seq_cap = seq;
        var created: usize = 0;
        errdefer inline for (buf_fields, 0..) |name, i| {
            if (i < created) ctx.tensorDestroy(&@field(ws, name));
        };
        const sizes = [buf_fields.len]usize{
            seq_pad * F * 4, // x_d
            seq_pad * F * 4, // t1_d
            seq_pad * heads * hd * 4, // q_d
            seq_pad * kv_heads * hd * 4, // k_d
            seq_pad * kv_heads * hd * 4, // v_d
            seq_pad * F * 4, // g_d
            seq_pad * F * 4 + if (flash) heads * seq_pad * 2 * 4 else 0, // attn_d
            seq_pad * dit.mlp_dim * 4, // mg_d
            seq_pad * dit.mlp_dim * 4, // mu_d
            dit.n_blocks * 6 * F * 4, // mv_d
            2 * F * 4, // fin_d
            n_img * dit.channels * dit.patch * dit.patch * 4, // imgin_d
            if (tc_attn) seq_pad * heads * hd * 2 else seq * heads * hd * 4 + 32, // qt_d
            if (tc_attn) kv_heads * hd * seq_pad * 2 else seq * kv_heads * hd * 4 + 32, // kt_d
            if (flash) 16 else hpb * s_rows * s_rows * s_esize, // s_d
            if (tc_attn) seq_pad * kv_heads * hd * 2 else 16, // v16_d
            if (tc_attn and !flash) hpb * seq * nchunks * 2 * 4 else 16, // part_d
            if (tc_attn and !flash) hpb * seq_pad * 2 * 4 else 16, // md_d
            seq * rms_ch * 4, // rmsp_d
            seq * 4, // rmsi_d
            if (coop) seq_pad * dit.mlp_dim * 2 else 16, // h16_d
        };
        inline for (buf_fields, sizes) |name, size| {
            @field(ws, name) = try ctx.tensorCreate(size);
            created += 1;
        }
        return ws;
    }

    pub fn deinit(self: *Workspace, ctx: *gpu.Context) void {
        inline for (buf_fields) |name| ctx.tensorDestroy(&@field(self, name));
        self.* = undefined;
    }

    const buf_fields = [_][]const u8{
        "x_d",  "t1_d",  "q_d",    "k_d",  "v_d",     "g_d",  "attn_d",
        "mg_d", "mu_d",  "mv_d",   "fin_d", "imgin_d", "qt_d", "kt_d",
        "s_d",  "v16_d", "part_d", "md_d", "rmsp_d",  "rmsi_d", "h16_d",
    };
};

/// How many heads' scores planes fit the byte cap (and, when reusing a
/// Workspace built for a longer sequence, its actual buffer).
fn headsPerBatch(s_rows: usize, s_esize: usize, cap: usize, ws_s_bytes: ?usize) usize {
    const plane = s_rows * s_rows * s_esize;
    var hb = @max(1, @min(heads, cap / plane));
    if (ws_s_bytes) |c| hb = @max(1, @min(hb, c / plane));
    return hb;
}

/// Byte budget for the materialized attention-scores buffer. Defaults to the
/// 2 GiB cap; a `--vram-budget` shrinks it (attention batches over more head
/// groups, trading a few launches for a much smaller `s_d`, the single biggest
/// activation buffer at high resolution). Floored so at least one head fits.
fn scoresCap(budget: u64) usize {
    if (budget == 0) return s_bytes_cap;
    return @min(s_bytes_cap, @max(64 << 20, budget / 4));
}

pub fn forward(
    model: *const DiT,
    ctx: *gpu.Context,
    sess: *const Session,
    ws: *const Workspace,
    io: std.Io,
    gpa: std.mem.Allocator,
    out: []f32,
    x_lat: []const f32,
    sigma: f32,
    cancel: ?*std.atomic.Value(bool),
) !void {
    // Weight class of the DiT block linears, gated once, the same check
    // `dit_cuda.forward` makes, and for the same reason. The branches below
    // recognize int8/int4 convrot and dense bf16 and treat *everything else* as
    // raw fp8-e4m3, so a ggml block quant (a GGUF checkpoint) would be read as
    // fp8 bytes and render a blank image with no error at all. Measured: that is
    // exactly what happened once `pipeline` learned to open a GGUF, the path
    // became reachable, so it needs the refusal.
    if (!dit.gpuLinKindSupported(model.blocks[0].attn.wq.dtype)) return error.UnsupportedCheckpoint;
    // A packed W4A8 weight is `[rows][cols/2]` bytes, and the `else` arm below feeds
    // anything it does not recognize to the fp8 GEMM, the same shape as the GGUF blank
    // image this gate already exists for. `is_i8` covers it now, but keep the explicit
    // refusal so a future storage form cannot reach that arm by default.
    if (!ctx.hasW4A8Decode() and dit.anyW4A8(model)) return error.UnsupportedCheckpoint;
    if (!ctx.hasNvfp4Decode() and dit.anyNvfp4(model)) return error.UnsupportedCheckpoint;
    if (dit.anyNvfp4(model))
        try ctx.ensureDeviceBuffer(&ctx.nvfp4_w16, dit.maxNvfp4Scratch(model, gpu.Context.nvfp4ScratchBytes));
    // Pre-size the W4A8 decode scratch to the model's widest weight BEFORE the batch
    // opens. Growing it mid-forward is *safe* (Vulkan's `ensureDeviceBuffer` flushes
    // the recording batch first) but costs a submit-and-wait per growth, and the first
    // block would pay several. The SD VAE decoders allocate up front for the same
    // reason.
    if (dit.anyW4A8(model)) try ctx.ensureDeviceBuffer(&ctx.w4a8_t, dit.maxW4A8Scratch(model, gpu.Context.w4a8ScratchBytes));

    const lat_h = sess.lat_h;
    const lat_w = sess.lat_w;
    const seq_txt = sess.seq_txt;
    const h = lat_h / dit.patch;
    const w = lat_w / dit.patch;
    const n_img = h * w;
    const seq = seq_txt + n_img;
    // Cooperative-matrix GEMMs compute a tile-padded m (128-row workgroup
    // tiles); buffers that receive GEMM output are sized for seq_pad (pad
    // rows stay zero).
    const seq_pad = std.mem.alignForward(usize, seq, 128);
    // An activation capture forces the f32 GEMM path (plan item 9), for
    // coverage first and fidelity second. The cooperative-matrix fast path for fp8
    // weights converts the modulated norm to f16 and calls `opMatmulCoopH16`
    // directly, bypassing `Gemm.go`, the one place a probe can see a block linear's
    // input. Measured: a Vulkan capture recorded 39 of the model's 263 layers, every
    // one of them from another path, and the cache looked perfectly well-formed.
    // Second, what that path would hand the probe is the f16 activation, not the f32
    // quantity the CPU reference records. Capture is not the throughput case.
    const probing = ops.matmul.probe != null;
    const coop = !force_f32 and !probing and ctx.pipe_coop != .null_handle;
    var prof: Prof = .{};
    var t_mark = std.Io.Clock.real.now(io);
    const mark = lapNs;

    // CPU: timestep vectors (cached in the session for schedule sigmas) and
    // per-block modulation (tvec + block offset).
    const half = hd / 2;
    var tv_owned: bool = false;
    var tv_t: []const f32 = undefined;
    var tv_vec: []const f32 = undefined;
    if (sess.tv(sigma)) |c| {
        tv_t = c.t;
        tv_vec = c.tvec;
    } else {
        const tvv = try model.timestepVectors(io, gpa, sigma);
        tv_owned = true;
        tv_t = tvv.t;
        tv_vec = tvv.tvec;
    }
    defer if (tv_owned) {
        gpa.free(tv_t);
        gpa.free(tv_vec);
    };
    const mv = try gpa.alloc(f32, dit.n_blocks * 6 * F);
    defer gpa.free(mv);
    for (model.blocks, 0..) |*blk, b| {
        for (mv[b * 6 * F ..][0 .. 6 * F], tv_vec, blk.mod) |*m, tvv, bm| m.* = tvv + bm;
        // Slots 0/3 (pre/post modulation scale) carry the rmsnorm weight
        // prefolded: rms_apply_mod computes x*inv_rms*premul + shift.
        const base = b * 6 * F;
        for (0..F) |c| {
            mv[base + c] = (1.0 + mv[base + c]) * blk.prenorm[c];
            mv[base + 3 * F + c] = (1.0 + mv[base + 3 * F + c]) * blk.postnorm[c];
        }
    }
    // Final-layer modulation, same folded form (premul then shift).
    const fin = try gpa.alloc(f32, 2 * F);
    defer gpa.free(fin);
    for (0..F) |c| {
        fin[c] = (1.0 + tv_t[c] + model.last_mod[c]) * model.last_norm[c];
        fin[F + c] = tv_t[c] + model.last_mod[F + c];
    }

    // CPU: image patches (text tokens and rope live in the session).
    const img_in = try DiT.patchify(gpa, x_lat, lat_h, lat_w);
    defer gpa.free(img_in);

    mark(io, &t_mark, &prof.cpu_ns);

    // Device tensors come from the per-run Workspace (allocated once, reused
    // every step and by both CFG sessions).
    std.debug.assert(seq <= ws.seq_cap);
    const x_d = ws.x_d;
    const t1_d = ws.t1_d;
    const q_d = ws.q_d;
    const k_d = ws.k_d;
    const v_d = ws.v_d;
    const g_d = ws.g_d;
    const flash = use_flash and ctx.pipe_flash_md != .null_handle;
    const md_off = seq_pad * F;
    const attn_d = ws.attn_d;
    const mg_d = ws.mg_d;
    const mu_d = ws.mu_d;
    const mv_d = ws.mv_d;
    const fin_d = ws.fin_d;
    const freqs_d = sess.freqs_d;
    const imgin_d = ws.imgin_d;
    // Attention operands. With the tensor-core scores pipeline, Q converts
    // to f16 (softmax scale prefolded) and K gathers into a per-head k-major
    // f16 block with seq_pad columns (zero padded); the f32 fallback keeps
    // the old k-major gathers (+32 B pad: its edge tiles read a few elements
    // past the last position, masked at store).
    comptime std.debug.assert(hd == 128); // buildGemmScores unrolls hd/16
    const tc_attn = !force_f32 and ctx.pipe_scores != .null_handle;
    const qt_d = ws.qt_d;
    const kt_d = ws.kt_d;
    // Attention scores batched over head groups to bound VRAM (~<= 2 GiB)
    // and fit the Workspace buffers (which may be sized for a longer
    // sequence, whose per-head planes and chunk tables divide differently).
    const s_rows = if (tc_attn) seq_pad else seq;
    const s_esize: usize = if (tc_attn) 2 else 4; // coop path stores S f16
    var heads_per_batch = headsPerBatch(s_rows, s_esize, s_bytes_cap, ws.s_d.size);
    if (tc_attn and !flash) {
        heads_per_batch = @max(1, @min(heads_per_batch, ws.part_d.size / (seq * nchunks * 2 * 4)));
        heads_per_batch = @max(1, @min(heads_per_batch, ws.md_d.size / (seq_pad * 2 * 4)));
    }
    const s_d = ws.s_d;
    const v16_d = ws.v16_d;
    const part_d = ws.part_d;
    const md_d = ws.md_d;
    const rmsp_d = ws.rmsp_d;
    const rmsi_d = ws.rmsi_d;
    const h16_d = ws.h16_d;

    try ctx.tensorUpload(mv_d, std.mem.sliceAsBytes(mv));
    try ctx.tensorUpload(fin_d, std.mem.sliceAsBytes(fin));
    try ctx.tensorCopy(x_d, 0, sess.txt0_d, 0, sess.txt_len * 4);
    try ctx.tensorUpload(imgin_d, std.mem.sliceAsBytes(img_in));
    mark(io, &t_mark, &prof.xfer_ns);

    // Record the whole forward into one submission; profiling stays
    // sync-per-op so the per-category host times remain exact.
    const batched = !profile;
    if (batched) try ctx.beginBatch();
    errdefer if (ctx.batching) ctx.abortBatch();
    // first: image patches -> x rows after the text tokens.
    try probeInput(ctx, imgin_d, n_img, model.first.w);
    try ctx.opMatmul(x_d, seq_txt * F * 4, imgin_d, 0, n_img, model.first.w.bytes, model.first.w.dtype == .f8_e4m3, F, dit.channels * dit.patch * dit.patch, model.first.w.scale, model.first.b);
        mark(io, &t_mark, &prof.matmul_ns);

    const sin_off: u32 = @intCast(seq * half);
    // Zero bias for the block linears: the f16-weight coop GEMM used for dense
    // bf16 weights always folds a bias in (bias_compact), and the DiT block
    // GEMMs have none. Hand it the full-width zero bias (a stable file-scope
    // pointer): opMatmulCoopF16Wb reads only `rows` of it, so one buffer serves
    // every GEMM width, a per-width slice of a shared buffer would collide in
    // the smallBuffer cache (same pointer, first-seen length).
    const zeros: []const f32 = &zero_bias;
    const Gemm = struct {
        fn go(c: *gpu.Context, use_coop: bool, y: gpu.DeviceBuffer, x: gpu.DeviceBuffer, m: usize, m_pad: usize, w_: anytype, bias: []const f32) !void {
            // Activation capture (plan item 9): this backend owns its upload and
            // GEMM, so the `ops.matmul` probe call site never sees these. One hook
            // here covers all 224 block linears; the txtfusion tower and the
            // timestep MLPs run on the CPU path under Vulkan and are captured there.
            try probeInput(c, x, m, w_);
            // Dense bf16 weights run through the f16-weight tensor-core GEMM
            // (bf16 -> f16 at upload); its f32 output matches opMatmulCoop, so
            // the surrounding f32-operand attention/mlp path is unchanged.
            // NVFP4 must be handled BEFORE the `else` arms: `opMatmulCoop` reads a
            // 1-byte weight as raw e4m3, so a packed NVFP4 weight would decode as fp8 and
            // render a blank image with no error, the same failure this file records for
            // a GGUF block quant. Weight-only here (its activation quantization is the
            // Blackwell path), so it feeds the ordinary f16 GEMM.
            if (w_.dtype == .nvfp4) {
                const meta = w_.nvfp4.?;
                try c.opMatmulNvfp4(
                    y,
                    x,
                    m,
                    w_.bytes,
                    meta.scales,
                    std.mem.asBytes(&meta.levels.bf16v),
                    w_.rows,
                    w_.cols,
                    bias,
                );
            } else if (w_.dtype == .bf16) {
                // Native bf16 tensor cores when the device has a bf16 coop
                // config; else convert bf16->f16 on the GPU (both keep the
                // conversion off the CPU under weight streaming).
                if (c.pipe_coop_bf16w != .null_handle)
                    try c.opMatmulCoopBf16(y, 0, x, m, w_.bytes, w_.rows, w_.cols, bias)
                else
                    try c.opMatmulCoopF16Wb(y, 0, x, m, w_.bytes, w_.rows, w_.cols, bias);
            } else if (use_coop) {
                try c.opMatmulCoop(y, x, m, m_pad, w_.bytes, w_.rows, w_.cols, w_.scale);
            } else {
                try c.opMatmul(y, 0, x, 0, m, w_.bytes, true, w_.rows, w_.cols, w_.scale, null);
            }
        }
    };
    for (model.blocks, 0..) |*blk, b| {
        // Poll cancel between blocks so a stop lands mid-step rather than only
        // at step boundaries. The `errdefer` above aborts the in-flight batch.
        if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
        const mv_base: u32 = @intCast(b * 6 * F);

        // t1 = (1+pre_scale) * prenorm(x) + pre_shift, the norm weight is
        // prefolded into mv slot 0; inv-rms comes from the parallel
        // partial/combine pair (a one-thread-per-row loop over dim 6144 is
        // latency-bound).
        try ctx.opElt(.rms_partial, x_d, null, null, rmsp_d, .{
            .u0 = @intCast(seq * rms_ch),
            .u1 = F,
            .u2 = rms_ch,
        }, seq * rms_ch, 1, 1);
        try ctx.opElt(.rms_combine, rmsp_d, null, null, rmsi_d, .{
            .u0 = @intCast(seq),
            .u1 = F,
            .u2 = rms_ch,
            .f0 = 1e-5,
        }, seq, 1, 1);
        // Attention. When every consumer shares one dequant scale (the Krea
        // 2 DiT stores raw e4m3: all scales are 1), the modulated norm
        // converts straight to f16 once and feeds all four GEMMs, the f32
        // intermediate and three redundant conversions disappear.
        // int8 (convrot) weights route through the int8 tensor-core path:
        // one rotate+quantize of the modulated-norm input feeds all four
        // GEMMs (opI8Prep/opI8Gemm), producing f32 q/k/v/g, so it takes the
        // f32-output branches below (which still run tensor-core attention).
        // `.w4a8` belongs on the int8 side of every branch below, not merely "not
        // fp8": the `else` arms feed the weight to the fp8 GEMM, which would read the
        // packed nibbles as e4m3 bytes and render a blank image with no error, the same
        // failure this file already records for a GGUF block quant. Its activation prep
        // and GEMM are int8's; only the weight's storage differs (`opI8GemmW4A8`).
        const is_i8 = blk.attn.wq.dtype == .i8 or blk.attn.wq.dtype == .w4a8;
        // Dense bf16 weights take the f32-operand fallback branch (Gemm.go
        // routes them to the f16-weight coop GEMM); they must be kept out of the
        // fp8-coop shared/att16 fast paths, which assume 1-byte e4m3 weights.
        const is_bf16 = blk.attn.wq.dtype == .bf16;
        // NVFP4 must be excluded from the fp8 SHARED paths below as well, not just from
        // `Gemm.go`. Those are gated on `!is_i8 and !is_bf16`, which for a packed NVFP4
        // weight is TRUE, and they call `opMatmulCoopH16`/`opMatmulCoop`, which read a
        // 1-byte weight as raw e4m3. That is how it first behaved: the q/k/v/gate and MLP
        // GEMMs took the fp8 path, decoding nibble pairs as e4m3 bytes, and it was visible
        // only as VRAM (84 layers cached at their LOGICAL byte count, 2x the packed size,
        // 8064 MiB of an 11899 MiB total) because a 4-step 256px render still looked
        // plausible. Weight storage has to gate every arm, not the one you remembered.
        const is_nvfp4 = blk.attn.wq.dtype == .nvfp4;
        const qkv_shared = coop and !is_i8 and !is_bf16 and !is_nvfp4 and
            blk.attn.wq.scale == blk.attn.wk.scale and
            blk.attn.wq.scale == blk.attn.wv.scale and
            blk.attn.wq.scale == blk.attn.gate.scale;
        // f16 C stores (exact under f16 accumulators): q/k/g come out of
        // their GEMMs half-precision, V lands directly in the P@V layout,
        // and the qk-norm + rope + convert chain collapses to one in-place
        // pass per operand.
        const att16 = qkv_shared and tc_attn and ctx.pipe_coop_c16 != .null_handle;
        // int8 attention in f16: wq/wk/wv output f16 directly (c_h16 GEMM) so
        // the fused att16 norm/rope/gather chain applies (gate stays f32, so the
        // gate/wo/mlp path is unchanged, no f16-input prep needed).
        const i8_f16 = is_i8 and tc_attn and ctx.pipe_coop_i8_fs16 != .null_handle;
        const attn_f16 = att16 or i8_f16;
        if (qkv_shared) {
            try ctx.opElt(.rms_apply_mod_h16, x_d, h16_d, mv_d, rmsi_d, .{
                .u0 = @intCast(seq_pad * F / 2),
                .u1 = F,
                .u2 = mv_base + 0 * F,
                .u3 = mv_base + 1 * F,
                .u4 = @intCast(seq * F),
                .f0 = blk.attn.wq.scale,
            }, seq_pad * F / 2, 1, 1);
            mark(io, &t_mark, &prof.elt_ns);
            // All four GEMMs read h16_d and write disjoint outputs: no
            // barriers between them, so the small wk/wv grids (6 columns of
            // workgroups) overlap the tails of their neighbors.
            ctx.independent(4);
            inline for (.{ .{ "wq", 0 }, .{ "wk", 1 }, .{ "wv", 2 }, .{ "gate", 3 } }) |v| {
                const w_ = @field(blk.attn, v[0]);
                const dst = switch (v[1]) {
                    0 => q_d,
                    1 => k_d,
                    2 => if (att16) v16_d else v_d,
                    else => g_d,
                };
                try ctx.opMatmulCoopH16(dst, h16_d, seq_pad, w_.bytes, w_.rows, w_.cols, att16);
                mark(io, &t_mark, &prof.matmul_ns);
            }
        } else {
            try ctx.opElt(.rms_apply_mod, x_d, t1_d, mv_d, rmsi_d, .{
                .u0 = @intCast(seq * F),
                .u1 = F,
                .u2 = mv_base + 0 * F,
                .u3 = mv_base + 1 * F,
            }, seq * F, 1, 1);
            mark(io, &t_mark, &prof.elt_ns);
            if (is_i8) {
                // Prep the modulated norm once, then four int8 GEMMs share it.
                // (Overlapping them via distinct accs was measured neutral on
                // this driver, the register-tiled GEMM is the bottleneck, not
                // serialization, so keep the single-acc path to save VRAM.)
                try ctx.opI8Prep(t1_d, seq, F);
                mark(io, &t_mark, &prof.prep_ns);
                // i8_f16: q/k/v come out f16 (v lands in the P@V layout v16_d)
                // for the att16 chain; gate stays f32 (downstream unchanged).
                inline for (.{ "wq", "wk", "wv", "gate" }) |name| {
                    const w_ = @field(blk.attn, name);
                    const is_gate = comptime std.mem.eql(u8, name, "gate");
                    const dst = if (comptime std.mem.eql(u8, name, "wq")) q_d else if (comptime std.mem.eql(u8, name, "wk")) k_d else if (comptime std.mem.eql(u8, name, "wv")) (if (i8_f16) v16_d else v_d) else g_d;
                    try i8GemmW(io, &prof, &t_mark, ctx, dst, w_, i8_f16 and !is_gate);
                    mark(io, &t_mark, &prof.matmul_ns);
                }
            } else {
                try Gemm.go(ctx, coop, q_d, t1_d, seq, seq_pad, blk.attn.wq, zeros);
                mark(io, &t_mark, &prof.matmul_ns);
                try Gemm.go(ctx, coop, k_d, t1_d, seq, seq_pad, blk.attn.wk, zeros);
                mark(io, &t_mark, &prof.matmul_ns);
                try Gemm.go(ctx, coop, v_d, t1_d, seq, seq_pad, blk.attn.wv, zeros);
                mark(io, &t_mark, &prof.matmul_ns);
                try Gemm.go(ctx, coop, g_d, t1_d, seq, seq_pad, blk.attn.gate, zeros);
                mark(io, &t_mark, &prof.matmul_ns);
            }
        }
        if (attn_f16) {
            // Fused per-head norm + rope + scale, in place on the f16 GEMM
            // outputs (value-identical to the chain below, see the kernel);
            // K then gathers f16 -> f16 into the scores layout. V needs
            // nothing: its GEMM already wrote the P@V operand.
            ctx.independent(2);
            try ctx.opElt(.qknorm_rope16, q_d, try normBuf(ctx, blk.attn.qnorm), freqs_d, null, .{
                .u0 = @intCast(seq * heads),
                .u1 = half,
                .u2 = sin_off,
                .u3 = heads,
                .f0 = attn_scale,
                .f1 = 1e-5,
            }, seq * heads, 1, 1);
            mark(io, &t_mark, &prof.elt_ns);
            try ctx.opElt(.qknorm_rope16, k_d, try normBuf(ctx, blk.attn.knorm), freqs_d, null, .{
                .u0 = @intCast(seq * kv_heads),
                .u1 = half,
                .u2 = sin_off,
                .u3 = kv_heads,
                .f0 = 1.0,
                .f1 = 1e-5,
            }, seq * kv_heads, 1, 1);
            mark(io, &t_mark, &prof.elt_ns);
            try ctx.opElt(.gather_kmajor16, k_d, null, null, kt_d, .{
                .u0 = @intCast(kv_heads * hd * seq_pad / 2),
                .u1 = hd,
                .u2 = @intCast(seq_pad),
                .u3 = @intCast(seq),
                .u4 = kv_heads,
            }, kv_heads * hd * seq_pad / 2, 1, 1);
            mark(io, &t_mark, &prof.elt_ns);
        } else if (is_i8 and tc_attn) {
            // int8 f32 GEMM outputs, tensor-core attention: fuse rmsnorm+rope
            // into ONE f32 in-place pass for q and k (2 passes each -> 1), then
            // the existing converts (which zero the seq_pad tail correctly).
            // 7 f32 passes -> 5. q's attn_scale is applied at the f32_to_h16.
            ctx.independent(2);
            try ctx.opElt(.qknorm_rope_f32, q_d, try normBuf(ctx, blk.attn.qnorm), freqs_d, null, .{
                .u0 = @intCast(seq * heads),
                .u1 = half,
                .u2 = sin_off,
                .u3 = heads,
                .f0 = 1.0,
                .f1 = 1e-5,
            }, seq * heads, 1, 1);
            try ctx.opElt(.qknorm_rope_f32, k_d, try normBuf(ctx, blk.attn.knorm), freqs_d, null, .{
                .u0 = @intCast(seq * kv_heads),
                .u1 = half,
                .u2 = sin_off,
                .u3 = kv_heads,
                .f0 = 1.0,
                .f1 = 1e-5,
            }, seq * kv_heads, 1, 1);
            mark(io, &t_mark, &prof.elt_ns);
            ctx.independent(3);
            try ctx.opElt(.f32_to_h16, q_d, null, null, qt_d, .{
                .u0 = @intCast(seq_pad * heads * hd / 2),
                .u1 = @intCast(seq * heads * hd),
                .f0 = attn_scale,
            }, seq_pad * heads * hd / 2, 1, 1);
            try ctx.opElt(.gather_kmajor_h16, k_d, null, null, kt_d, .{
                .u0 = @intCast(kv_heads * hd * seq_pad / 2),
                .u1 = hd,
                .u2 = @intCast(seq_pad),
                .u3 = @intCast(seq),
                .u4 = kv_heads,
            }, kv_heads * hd * seq_pad / 2, 1, 1);
            try ctx.opElt(.f32_to_h16, v_d, null, null, v16_d, .{
                .u0 = @intCast(seq_pad * kv_heads * hd / 2),
                .u1 = @intCast(seq * kv_heads * hd),
                .f0 = 1.0,
            }, seq_pad * kv_heads * hd / 2, 1, 1);
            mark(io, &t_mark, &prof.elt_ns);
        } else {
        // The q and k chains (norm -> rope -> convert/gather) touch disjoint
        // buffers stage by stage; each stage pair/triple runs barrier-free.
        // The SUBGROUP norm, because `Elt.rmsnorm` gives each THREAD a whole 128-wide
        // head, a warp's 32 loads land 512 B apart and each is its own sector. Measured
        // at krea2's geometry at 1120x1680 (`vk-norm-bench`, 352800 x 128): 35 GB/s
        // against 558, 16x, worth ~600 ms/step across the two calls x 28 blocks.
        //
        // This `else` arm is reached by bf16 and fp8 checkpoints only, `qkv_shared`
        // requires `!is_bf16` so they fall through, while an int8 one takes the fused
        // `qknorm_rope_f32` path above and never gets here. So this is a dense-checkpoint
        // win; the int8 arm already avoided the separate norm.
        //
        // Not bit-identical: the row sum becomes a subgroup tree where it was serial (the
        // more accurate of the two). `independent(2)` still applies, one dispatch either way.
        ctx.independent(2);
        try rmsNormQk(ctx, q_d, blk.attn.qnorm, seq * heads, hd);
        mark(io, &t_mark, &prof.elt_ns);
        try rmsNormQk(ctx, k_d, blk.attn.knorm, seq * kv_heads, hd);
        mark(io, &t_mark, &prof.elt_ns);
        ctx.independent(2);
        try ctx.opElt(.rope_inter, q_d, null, freqs_d, null, .{
            .u0 = @intCast(seq * heads * half),
            .u1 = half,
            .u2 = sin_off,
            .u3 = heads,
        }, seq * heads * half, 1, 1);
        mark(io, &t_mark, &prof.elt_ns);
        try ctx.opElt(.rope_inter, k_d, null, freqs_d, null, .{
            .u0 = @intCast(seq * kv_heads * half),
            .u1 = half,
            .u2 = sin_off,
            .u3 = kv_heads,
        }, seq * kv_heads * half, 1, 1);
        mark(io, &t_mark, &prof.elt_ns);
        if (tc_attn) {
            ctx.independent(3);
            try ctx.opElt(.f32_to_h16, q_d, null, null, qt_d, .{
                .u0 = @intCast(seq_pad * heads * hd / 2),
                .u1 = @intCast(seq * heads * hd),
                .f0 = attn_scale,
            }, seq_pad * heads * hd / 2, 1, 1);
            try ctx.opElt(.gather_kmajor_h16, k_d, null, null, kt_d, .{
                .u0 = @intCast(kv_heads * hd * seq_pad / 2),
                .u1 = hd,
                .u2 = @intCast(seq_pad),
                .u3 = @intCast(seq),
                .u4 = kv_heads,
            }, kv_heads * hd * seq_pad / 2, 1, 1);
            try ctx.opElt(.f32_to_h16, v_d, null, null, v16_d, .{
                .u0 = @intCast(seq_pad * kv_heads * hd / 2),
                .u1 = @intCast(seq * kv_heads * hd),
                .f0 = 1.0,
            }, seq_pad * kv_heads * hd / 2, 1, 1);
        } else {
            ctx.independent(2);
            try ctx.opElt(.gather_kmajor, q_d, null, null, qt_d, .{
                .u0 = @intCast(seq * heads * hd),
                .u1 = heads,
                .u2 = hd,
                .u3 = @intCast(seq),
            }, seq * heads * hd, 1, 1);
            try ctx.opElt(.gather_kmajor, k_d, null, null, kt_d, .{
                .u0 = @intCast(seq * kv_heads * hd),
                .u1 = kv_heads,
                .u2 = hd,
                .u3 = @intCast(seq),
            }, seq * kv_heads * hd, 1, 1);
        }
        }
        // On the f16-C path Q was normed/roped/scaled in place: it IS the
        // scores operand.
        const q_src = if (attn_f16) q_d else qt_d;
        if (flash) {
            const push = gpu.EltPush{
                .u0 = heads * hd,
                .u1 = @intCast(seq_pad),
                .u2 = 0,
                .u3 = heads / kv_heads,
                .u4 = kv_heads * hd,
                .u5 = @intCast(md_off),
                .f0 = @bitCast(@as(u32, @intCast(seq))),
            };
            try ctx.opFlash(.md, q_src, kt_d, v16_d, attn_d, push, seq_pad / 128, heads);
            mark(io, &t_mark, &prof.scores_ns);
            mark(io, &t_mark, &prof.smax_ns);
            try ctx.opFlash(.out, q_src, kt_d, v16_d, attn_d, push, seq_pad / 128, heads);
            mark(io, &t_mark, &prof.aout_ns);
        } else {
            const s_stride: u32 = @intCast(s_rows);
            const s_plane: u32 = @intCast(s_rows * s_rows);
            var h0: usize = 0;
            while (h0 < heads) : (h0 += heads_per_batch) {
                const hb = @min(heads_per_batch, heads - h0);
                if (tc_attn) {
                    try ctx.opAttnScores(s_d, q_src, kt_d, .{
                        .u0 = heads * hd,
                        .u1 = s_stride,
                        .u2 = @intCast(h0),
                        .u3 = heads / kv_heads,
                        .u4 = @intCast(hd * seq_pad),
                        .u5 = s_plane,
                    }, seq_pad / 128, seq_pad / 128, hb);
                } else {
                    try ctx.opElt(.attn_scores, qt_d, kt_d, null, s_d, .{
                        .u0 = @intCast(seq),
                        .u1 = heads,
                        .u2 = kv_heads,
                        .u3 = hd,
                        .u4 = @intCast(h0),
                        .f0 = attn_scale,
                    }, std.math.divCeil(usize, seq, 8) catch unreachable, std.math.divCeil(usize, seq, 8) catch unreachable, hb);
                }
                mark(io, &t_mark, &prof.scores_ns);
                if (tc_attn) {
                    try ctx.opElt(.softmax_partial, s_d, null, null, part_d, .{
                        .u0 = @intCast(hb * seq * nchunks),
                        .u1 = @intCast(nchunks),
                        .u2 = @intCast(seq),
                        .u3 = s_stride,
                        .u5 = s_plane,
                    }, hb * seq * nchunks, 1, 1);
                    try ctx.opElt(.softmax_combine, part_d, null, null, md_d, .{
                        .u0 = @intCast(hb * seq),
                        .u1 = @intCast(nchunks),
                        .u2 = @intCast(seq),
                        .u3 = s_stride,
                    }, hb * seq, 1, 1);
                    mark(io, &t_mark, &prof.smax_ns);
                    try ctx.opAttnOut(s_d, v16_d, attn_d, md_d, .{
                        .u0 = s_stride,
                        .u1 = s_plane,
                        .u2 = @intCast(h0),
                        .u3 = heads / kv_heads,
                        .u4 = kv_heads * hd,
                        .u5 = heads * hd,
                        .f0 = @bitCast(@as(u32, @intCast(seq))),
                        .f1 = @bitCast(s_stride), // MD rows per head plane
                    }, seq_pad / 128, hb);
                } else {
                    try ctx.opElt(.attn_out, s_d, null, v_d, attn_d, .{
                        .u0 = @intCast(seq),
                        .u1 = heads,
                        .u2 = kv_heads,
                        .u3 = hd,
                        .u4 = @intCast(h0),
                        .u5 = s_stride,
                        .f0 = @bitCast(s_plane),
                    }, hd / 8, std.math.divCeil(usize, seq, 8) catch unreachable, hb);
                }
                mark(io, &t_mark, &prof.aout_ns);
            }
        }
        mark(io, &t_mark, &prof.attn_ns);
        if (coop and !is_i8 and !is_bf16 and !is_nvfp4) {
            if (att16) {
                try ctx.opElt(.sigmoid_mul_g16, attn_d, g_d, null, h16_d, .{
                    .u0 = @intCast(seq_pad * F / 2),
                    .u1 = @intCast(seq * F),
                    .f0 = blk.attn.wo.scale,
                }, seq_pad * F / 2, 1, 1);
            } else {
                try ctx.opElt(.sigmoid_mul_h16, attn_d, g_d, null, h16_d, .{
                    .u0 = @intCast(seq_pad * F / 2),
                    .u1 = @intCast(seq * F),
                    .f0 = blk.attn.wo.scale,
                }, seq_pad * F / 2, 1, 1);
            }
            mark(io, &t_mark, &prof.elt_ns);
            try ctx.opMatmulCoopH16(t1_d, h16_d, seq_pad, blk.attn.wo.bytes, blk.attn.wo.rows, blk.attn.wo.cols, att16);
        } else {
            try ctx.opElt(.sigmoid_mul, attn_d, g_d, null, null, .{ .u0 = @intCast(seq * F) }, seq * F, 1, 1);
            mark(io, &t_mark, &prof.elt_ns);
            if (is_i8) {
                try ctx.opI8Prep(attn_d, seq, blk.attn.wo.cols);
                mark(io, &t_mark, &prof.prep_ns);
                try i8GemmW(io, &prof, &t_mark, ctx, t1_d, blk.attn.wo, false);
            } else {
                try Gemm.go(ctx, coop, t1_d, attn_d, seq, seq_pad, blk.attn.wo, zeros);
            }
        }
        mark(io, &t_mark, &prof.matmul_ns);
        if (att16) {
            try ctx.opElt(.gated_add16, x_d, t1_d, mv_d, null, .{
                .u0 = @intCast(seq * F / 2),
                .u1 = F,
                .u2 = mv_base + 2 * F,
            }, seq * F / 2, 1, 1);
        } else {
            try ctx.opElt(.gated_add, x_d, t1_d, mv_d, null, .{
                .u0 = @intCast(seq * F),
                .u1 = F,
                .u2 = mv_base + 2 * F,
            }, seq * F, 1, 1);
        }
        mark(io, &t_mark, &prof.elt_ns);

        // MLP.
        try ctx.opElt(.rms_partial, x_d, null, null, rmsp_d, .{
            .u0 = @intCast(seq * rms_ch),
            .u1 = F,
            .u2 = rms_ch,
        }, seq * rms_ch, 1, 1);
        try ctx.opElt(.rms_combine, rmsp_d, null, null, rmsi_d, .{
            .u0 = @intCast(seq),
            .u1 = F,
            .u2 = rms_ch,
            .f0 = 1e-5,
        }, seq, 1, 1);
        const mlp_shared = coop and !is_i8 and !is_bf16 and !is_nvfp4 and blk.mlp.gate.scale == blk.mlp.up.scale;
        const mlp16 = mlp_shared and ctx.pipe_coop_c16 != .null_handle;
        if (mlp_shared) {
            try ctx.opElt(.rms_apply_mod_h16, x_d, h16_d, mv_d, rmsi_d, .{
                .u0 = @intCast(seq_pad * F / 2),
                .u1 = F,
                .u2 = mv_base + 3 * F,
                .u3 = mv_base + 4 * F,
                .u4 = @intCast(seq * F),
                .f0 = blk.mlp.gate.scale,
            }, seq_pad * F / 2, 1, 1);
            mark(io, &t_mark, &prof.elt_ns);
            ctx.independent(2);
            try ctx.opMatmulCoopH16(mg_d, h16_d, seq_pad, blk.mlp.gate.bytes, blk.mlp.gate.rows, blk.mlp.gate.cols, mlp16);
            mark(io, &t_mark, &prof.matmul_ns);
            try ctx.opMatmulCoopH16(mu_d, h16_d, seq_pad, blk.mlp.up.bytes, blk.mlp.up.rows, blk.mlp.up.cols, mlp16);
            mark(io, &t_mark, &prof.matmul_ns);
        } else {
            try ctx.opElt(.rms_apply_mod, x_d, t1_d, mv_d, rmsi_d, .{
                .u0 = @intCast(seq * F),
                .u1 = F,
                .u2 = mv_base + 3 * F,
                .u3 = mv_base + 4 * F,
            }, seq * F, 1, 1);
            mark(io, &t_mark, &prof.elt_ns);
            if (is_i8) {
                try ctx.opI8Prep(t1_d, seq, F);
                mark(io, &t_mark, &prof.prep_ns);
                try i8GemmW(io, &prof, &t_mark, ctx, mg_d, blk.mlp.gate, false);
                mark(io, &t_mark, &prof.matmul_ns);
                try i8GemmW(io, &prof, &t_mark, ctx, mu_d, blk.mlp.up, false);
                mark(io, &t_mark, &prof.matmul_ns);
            } else {
                try Gemm.go(ctx, coop, mg_d, t1_d, seq, seq_pad, blk.mlp.gate, zeros);
                mark(io, &t_mark, &prof.matmul_ns);
                try Gemm.go(ctx, coop, mu_d, t1_d, seq, seq_pad, blk.mlp.up, zeros);
                mark(io, &t_mark, &prof.matmul_ns);
            }
        }
        if (coop and !is_i8 and !is_bf16 and !is_nvfp4) {
            if (mlp16) {
                try ctx.opElt(.silu_mul16, mg_d, mu_d, null, h16_d, .{
                    .u0 = @intCast(seq_pad * dit.mlp_dim / 2),
                    .f0 = blk.mlp.down.scale,
                }, seq_pad * dit.mlp_dim / 2, 1, 1);
            } else {
                try ctx.opElt(.silu_mul_h16, mg_d, mu_d, null, h16_d, .{
                    .u0 = @intCast(seq_pad * dit.mlp_dim / 2),
                    .u1 = @intCast(seq * dit.mlp_dim),
                    .f0 = blk.mlp.down.scale,
                }, seq_pad * dit.mlp_dim / 2, 1, 1);
            }
            mark(io, &t_mark, &prof.elt_ns);
            try ctx.opMatmulCoopH16(t1_d, h16_d, seq_pad, blk.mlp.down.bytes, blk.mlp.down.rows, blk.mlp.down.cols, mlp16);
        } else {
            try ctx.opElt(.silu_mul, mg_d, mu_d, null, null, .{ .u0 = @intCast(seq * dit.mlp_dim) }, seq * dit.mlp_dim, 1, 1);
            mark(io, &t_mark, &prof.elt_ns);
            if (is_i8) {
                try ctx.opI8Prep(mg_d, seq, blk.mlp.down.cols);
                mark(io, &t_mark, &prof.prep_ns);
                try i8GemmW(io, &prof, &t_mark, ctx, t1_d, blk.mlp.down, false);
            } else {
                try Gemm.go(ctx, coop, t1_d, mg_d, seq, seq_pad, blk.mlp.down, zeros);
            }
        }
        mark(io, &t_mark, &prof.matmul_ns);
        if (mlp16) {
            try ctx.opElt(.gated_add16, x_d, t1_d, mv_d, null, .{
                .u0 = @intCast(seq * F / 2),
                .u1 = F,
                .u2 = mv_base + 5 * F,
            }, seq * F / 2, 1, 1);
        } else {
            try ctx.opElt(.gated_add, x_d, t1_d, mv_d, null, .{
                .u0 = @intCast(seq * F),
                .u1 = F,
                .u2 = mv_base + 5 * F,
            }, seq * F, 1, 1);
        }
        mark(io, &t_mark, &prof.elt_ns);
    }

    // Final layer on device: modulated rmsnorm then the 6144 -> 64 linear
    // (norm runs over all rows, the text-row waste is negligible next to
    // downloading 100 MB of hidden image rows for a CPU finalize).
    try ctx.opElt(.rms_partial, x_d, null, null, rmsp_d, .{
        .u0 = @intCast(seq * rms_ch),
        .u1 = F,
        .u2 = rms_ch,
    }, seq * rms_ch, 1, 1);
    try ctx.opElt(.rms_combine, rmsp_d, null, null, rmsi_d, .{
        .u0 = @intCast(seq),
        .u1 = F,
        .u2 = rms_ch,
        .f0 = 1e-5,
    }, seq, 1, 1);
    try ctx.opElt(.rms_apply_mod, x_d, t1_d, fin_d, rmsi_d, .{
        .u0 = @intCast(seq * F),
        .u1 = F,
        .u2 = 0,
        .u3 = F,
    }, seq * F, 1, 1);
    mark(io, &t_mark, &prof.elt_ns);
    // imgin_d's input role is long done; it is exactly n_img x 64.
    // The offset is part of the measurement: this GEMM reads the image rows only.
    try probeInputAt(ctx, t1_d, seq_txt * F * 4, n_img, model.last_linear.w);
    try ctx.opMatmul(imgin_d, 0, t1_d, seq_txt * F * 4, n_img, model.last_linear.w.bytes, model.last_linear.w.dtype == .f8_e4m3, dit.channels * dit.patch * dit.patch, F, model.last_linear.w.scale, model.last_linear.b);
    mark(io, &t_mark, &prof.matmul_ns);

    if (batched) try ctx.endBatch();

    const final_rows = try gpa.alloc(f32, n_img * dit.channels * dit.patch * dit.patch);
    defer gpa.free(final_rows);
    try ctx.tensorDownload(imgin_d, std.mem.sliceAsBytes(final_rows));
    mark(io, &t_mark, &prof.xfer_ns);
    DiT.unpatchify(out, final_rows, lat_h, lat_w);
    mark(io, &t_mark, &prof.cpu_ns);

    if (profile) {
        const ms = struct {
            fn go(ns: i96) f64 {
                return @as(f64, @floatFromInt(ns)) / 1e6;
            }
        }.go;
        std.debug.print(
            "dit gpu profile: matmul {d:.0}ms  attn {d:.0}ms (scores {d:.0} smax {d:.0} out {d:.0})  elt {d:.0}ms  prep {d:.0}ms  w4a8 {d:.0}ms  xfer {d:.0}ms  cpu {d:.0}ms\n",
            .{ ms(prof.matmul_ns), ms(prof.attn_ns + prof.scores_ns + prof.smax_ns + prof.aout_ns), ms(prof.scores_ns), ms(prof.smax_ns), ms(prof.aout_ns), ms(prof.elt_ns), ms(prof.prep_ns), ms(prof.w4a8_ns), ms(prof.xfer_ns), ms(prof.cpu_ns) },
        );
    }
}

/// Feed `ops.matmul.probe` a GEMM input that lives on the device, for an activation
/// capture running on Vulkan. See `dit_cuda.probeInput` for why this is a download
/// into the CPU accumulator rather than an on-device reduction.
///
/// int8/int4 weights are skipped. Their activation has already been rotated
/// and quantized in place, so the buffer holds a W4A4-shaped value and recording it
/// as a weight-only statistic would be a category error (ACTIVATION_AWARE hygiene
/// rule 4). `ggufy calibrate` refuses those checkpoints on a GPU backend outright.
fn probeInput(ctx: *gpu.Context, x: gpu.DeviceBuffer, m: usize, w: anytype) !void {
    return probeInputAt(ctx, x, 0, m, w);
}

fn probeInputAt(ctx: *gpu.Context, x: gpu.DeviceBuffer, off_bytes: u64, m: usize, w: anytype) !void {
    const p = ops.matmul.probe orelse return;
    if (w.dtype == .i8 or w.dtype == .i4) return;
    if (m == 0 or w.cols == 0) return;
    const host = try ctx.gpa.alloc(f32, m * w.cols);
    defer ctx.gpa.free(host);
    try ctx.tensorDownloadAt(x, off_bytes, std.mem.sliceAsBytes(host));
    p.input(p.ctx, w, host, m);
}

/// Per-head weighted RMSNorm in place over `[rows][hd]`, preferring the subgroup kernel.
/// See the call site for the measurement and for which checkpoints reach it.
fn rmsNormQk(ctx: *gpu.Context, x: gpu.DeviceBuffer, weights: []const f32, rows: usize, width: usize) !void {
    const w = try normBuf(ctx, weights);
    if (ctx.hasSubgroupNorm()) return ctx.opRmsNormSg(x, x, w, rows, width, 1e-5);
    try ctx.opElt(.rmsnorm, x, x, w, null, .{
        .u0 = @intCast(rows),
        .u1 = @intCast(width),
        .f0 = 1e-5,
    }, rows, 1, 1);
}

fn normBuf(ctx: *gpu.Context, weights: []const f32) !gpu.DeviceBuffer {
    // smallBuffer caches by pointer; wrap the raw handle for opElt.
    const buf = try ctx.smallBuffer(std.mem.sliceAsBytes(weights));
    return .{ .buf = buf, .mem = .null_handle, .size = 0 };
}

// Parity against the same ComfyUI fixture as the CPU forward test. Needs the
// model, testdata fixtures, and BOTH markers (slow + gpu).
test "gpu-resident forward matches comfyui fixture" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const safetensors = @import("tp_core").safetensors;
    std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    const dit_path = "models/diffusion_model/krea2CenterSemiraw_v10Fp8.safetensors";
    std.Io.Dir.cwd().access(io, dit_path, .{}) catch return error.SkipZigTest;
    std.Io.Dir.cwd().access(io, "testdata/dit_out.bin", .{}) catch return error.SkipZigTest;

    var ctx = gpu.Context.init(gpa) catch return error.SkipZigTest;
    defer ctx.deinit();

    const readF32 = struct {
        fn go(alloc: std.mem.Allocator, io_: std.Io, path: []const u8, n: usize) ![]f32 {
            const out = try alloc.alloc(f32, n);
            errdefer alloc.free(out);
            const file = try std.Io.Dir.cwd().openFile(io_, path, .{ .mode = .read_only });
            defer file.close(io_);
            const bytes = std.mem.sliceAsBytes(out);
            if (try file.readPositionalAll(io_, bytes, 0) != bytes.len) return error.ShortRead;
            return out;
        }
    }.go;

    const seq_txt = 14;
    const x_lat = try readF32(gpa, io, "testdata/dit_x.bin", dit.channels * 16 * 16);
    defer gpa.free(x_lat);
    const expected = try readF32(gpa, io, "testdata/dit_out.bin", dit.channels * 16 * 16);
    defer gpa.free(expected);
    const cond = try readF32(gpa, io, "testdata/text_cond.bin", seq_txt * dit.txt_layers * dit.txt_dim);
    defer gpa.free(cond);

    var st = try safetensors.SafeTensors.open(gpa, io, dit_path);
    defer st.deinit();
    var model = try DiT.load(gpa, .{ .safetensors = &st });
    defer model.deinit();

    const out = try gpa.alloc(f32, dit.channels * 16 * 16);
    defer gpa.free(out);
    var sess = try Session.init(gpa, io, ctx, &model, 16, 16, cond, seq_txt, &.{0.875});
    defer sess.deinit(gpa, ctx);
    var ws = try Workspace.init(ctx, 16, 16, seq_txt);
    defer ws.deinit(ctx);
    try forward(&model, ctx, &sess, &ws, io, gpa, out, x_lat, 0.875, null);

    var max_err: f32 = 0;
    var max_val: f32 = 0;
    for (expected, out) |e, a| {
        // NaN-robust: @max drops NaN operands, which would hide a poisoned
        // output entirely.
        max_err = if (std.math.isNan(a)) std.math.inf(f32) else @max(max_err, @abs(e - a));
        max_val = @max(max_val, @abs(e));
    }
    std.debug.print("dit gpu parity: max_err={d:.5} max_val={d:.2}\n", .{ max_err, max_val });
    // 5%: attention Q/K run in f16 on the tensor-core path (same regime as
    // ComfyUI's fp16 SDPA, so the fixture carries equivalent rounding noise
    // of its own); measured max_err is ~3.8% over 28 blocks, and same-seed
    // images against the f32-attention path are visually identical (mean
    // pixel delta ~0.9%). The scores kernel itself is verified to 5e-3
    // against an f16-rounded reference in gpu/context.zig.
    try std.testing.expect(max_err < 0.05 * @max(1.0, max_val));

    // Full-f32 path (tensor-core pipelines forced off, Workspace re-sized to
    // match): reports the f16-vs-f32 accuracy gap against the same fixture.
    {
        force_f32 = true;
        defer force_f32 = false;
        var ws32 = try Workspace.init(ctx, 16, 16, seq_txt);
        defer ws32.deinit(ctx);
        const out32 = try gpa.alloc(f32, dit.channels * 16 * 16);
        defer gpa.free(out32);
        try forward(&model, ctx, &sess, &ws32, io, gpa, out32, x_lat, 0.875, null);
        var me: f32 = 0;
        for (expected, out32) |e, a| me = if (std.math.isNan(a)) std.math.inf(f32) else @max(me, @abs(e - a));
        std.debug.print("dit gpu parity (f32): max_err={d:.5} (f16 was {d:.5})\n", .{ me, max_err });
        try std.testing.expect(me < 0.05 * @max(1.0, max_val));
    }

    // Weight streaming under memory pressure: force a budget far below the
    // model size so the LRU weight cache evicts and re-uploads every block,
    // then check the forward is BIT-IDENTICAL to the resident run (same
    // weights, same kernels, only residency changed).
    ctx.evictWeights();
    ctx.budget_override = 3 << 30;
    defer ctx.budget_override = 0;
    const out2 = try gpa.alloc(f32, dit.channels * 16 * 16);
    defer gpa.free(out2);
    try forward(&model, ctx, &sess, &ws, io, gpa, out2, x_lat, 0.875, null);
    for (out, out2) |a, b| try std.testing.expectEqual(a, b);

    // A pre-set cancel flag aborts mid-step (polled between blocks); the
    // errdefer in forward aborts the in-flight batch so the context stays
    // usable afterwards.
    var canceled = std.atomic.Value(bool).init(true);
    try std.testing.expectError(error.Canceled, forward(&model, ctx, &sess, &ws, io, gpa, out2, x_lat, 0.875, &canceled));
}

// The W4A8 decode kernel against the CPU decode it must reproduce. Needs a device
// (testdata/gpu-tests) but NO checkpoint, synthetic weights are enough, and that is
// deliberate: the kernel's correctness must be checkable without a multi-GB file on
// disk, which is exactly the situation that leaves a gated test silently self-skipping.
//
// This is also the test that would have caught the layout half. The first version of
// `w4a8_decode_t` read the row-major storage and transposed as it went (correct, and
// 1757 ms/step); the current one reads inputs that `weightBuffer` has already
// byte-transposed. Those two kernels agree on nothing about their input indexing, so a
// render is the only other thing that can tell them apart.
test "the Vulkan W4A8 decode matches ops.w4a8.decode, including the row padding" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const w4a8 = ops.w4a8;
    std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    var ctx = gpu.Context.init(gpa) catch return error.SkipZigTest;
    defer ctx.deinit();
    if (!ctx.hasW4A8Decode()) return error.SkipZigTest;

    // Two heights: one that needs no row padding and one that does, since the padded
    // slots are read by the GEMM and must decode to zero.
    const cases = [_]struct { rows: usize, cols: usize, gs: usize }{
        .{ .rows = 256, .cols = 512, .gs = 16 },
        .{ .rows = 1536, .cols = 256, .gs = 16 },
        .{ .rows = 192, .cols = 512, .gs = 32 },
        // Group sizes 8 and 4 are legal in the format and this kernel is general over
        // them, unlike the CUDA one, which needs one scale per 8 columns because it
        // decodes four packed bytes per thread. Pinning them here is what makes that a
        // documented backend difference rather than an accident.
        .{ .rows = 256, .cols = 256, .gs = 8 },
        .{ .rows = 128, .cols = 256, .gs = 4 },
    };
    // Every case's buffers stay ALIVE in one arena, and that is load-bearing rather
    // than tidy: the device weight cache keys on the HOST POINTER, so freeing a case's
    // arrays and letting the allocator hand the same address to the next case scores a
    // stale cache hit and the kernel reads the previous case's weights. (A model never
    // hits it: the weights live in the model arena for the model's lifetime.)
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const alloc = arena.allocator();
    for (cases) |c| {
        const stride = gpu.Context.w4a8ScratchBytes(c.rows, c.cols) / c.cols;
        const groups = c.cols / c.gs;

        var rnd = std.Random.DefaultPrng.init(0xA8 + c.rows);
        const r = rnd.random();
        const packed_bytes = try alloc.alloc(u8, c.rows * c.cols / 2);
        r.bytes(packed_bytes);
        const s_rel = try alloc.alloc(u8, c.rows * groups);
        // Every fp8 byte, NaN (0x7F/0xFF) included: the level table maps those to 0 on
        // both sides, so this pins that the two agree there rather than diverging on a
        // value neither reference defines.
        r.bytes(s_rel);
        const levels = try alloc.create(w4a8.Levels);
        levels.* = w4a8.Levels.init(&w4a8.fixed_lut);

        // Expected: the CPU decode, laid out k-major with zeroed row padding.
        const row_major = try alloc.alloc(i8, c.rows * c.cols);
        w4a8.decode(row_major, packed_bytes, s_rel, levels, c.rows, c.cols, c.gs);
        const want = try alloc.alloc(i8, stride * c.cols);
        @memset(want, 0);
        for (0..c.cols) |k| {
            for (0..c.rows) |row| want[k * stride + row] = row_major[row * c.cols + k];
        }

        const buf = try ctx.w4a8Decode(packed_bytes, s_rel, std.mem.asBytes(levels), c.rows, c.cols, c.gs);
        _ = buf;
        const got = try alloc.alloc(i8, stride * c.cols);
        try ctx.tensorDownload(ctx.w4a8_t, std.mem.sliceAsBytes(got));

        for (want, got, 0..) |wv, gv, i| {
            std.testing.expectEqual(wv, gv) catch |e| {
                std.debug.print("rows {d} cols {d} gs {d} stride {d}: element {d} (k {d}, row {d}) got {d}, want {d}\n", .{
                    c.rows, c.cols, c.gs, stride, i, i / stride, i % stride, gv, wv,
                });
                return e;
            };
        }
    }
}

// The Vulkan NVFP4 decode kernel against the CPU decode it must reproduce. Synthetic
// weights, so no checkpoint is needed, the durable form of the check.
//
// Covers the two things a render cannot localize and the CUDA twin does not share: the
// f16 `[k_pad][n_pad]` output layout (a different padding rule from the inputs' own
// stride) and the fact that BOTH paddings must be written, since this scratch is reused
// between weights of different shapes.
test "the Vulkan NVFP4 decode matches ops.nvfp4.decode, including both paddings" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const nvfp4 = ops.nvfp4;
    std.Io.Dir.cwd().access(io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    var ctx = gpu.Context.init(gpa) catch return error.SkipZigTest;
    defer ctx.deinit();
    if (!ctx.hasNvfp4Decode()) return error.SkipZigTest;

    const cases = [_]struct { rows: usize, cols: usize }{
        .{ .rows = 128, .cols = 256 }, // no padding on either axis
        .{ .rows = 40, .cols = 192 }, // rows pad to 128, cols to 192 (already 64-aligned)
        .{ .rows = 256, .cols = 64 }, // several row blocks, one k block
    };
    // One arena for every case: the weight cache keys on the HOST POINTER, so a freed and
    // re-allocated address would score a stale hit and decode the previous case's bytes.
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const alloc = arena.allocator();

    for (cases) |c| {
        var rnd = std.Random.DefaultPrng.init(0x4F4 + c.rows);
        const r = rnd.random();
        const packed_bytes = try alloc.alloc(u8, c.rows * c.cols / 2);
        r.bytes(packed_bytes);
        const scales = try alloc.alloc(u8, c.rows * c.cols / nvfp4.block_size);
        r.bytes(scales);
        const levels = try alloc.create(nvfp4.Levels);
        levels.* = nvfp4.Levels.init(0.0125);
        const meta: nvfp4.Meta = .{ .scales = scales, .levels = levels };

        const plain = try alloc.alloc(f32, c.rows * c.cols);
        nvfp4.decode(plain, packed_bytes, meta, c.rows, c.cols);

        const n_pad = std.mem.alignForward(usize, c.rows, 128);
        const k_pad = std.mem.alignForward(usize, c.cols, 64);
        const want = try alloc.alloc(f32, k_pad * n_pad);
        @memset(want, 0); // both paddings must come out zero
        for (0..c.cols) |k| {
            for (0..c.rows) |row| want[k * n_pad + row] = plain[row * c.cols + k];
        }

        _ = try ctx.nvfp4Decode(packed_bytes, scales, std.mem.asBytes(&levels.bf16v), c.rows, c.cols);
        const got_bits = try alloc.alloc(u16, k_pad * n_pad);
        try ctx.tensorDownload(ctx.nvfp4_w16, std.mem.sliceAsBytes(got_bits));

        for (want, got_bits, 0..) |wv, gb, i| {
            const gv: f32 = @bitCast(@as(u32, gb) << 16); // bf16 -> f32
            // A NaN block-scale byte propagates on both sides (see ops/nvfp4.zig); bf16
            // rounds the value, 8 mantissa bits = 2^-8 relative, and nothing else may
            // change it. A wrong nibble or block index moves by a whole E2M1 level.
            const both_nan = std.math.isNan(wv) and std.math.isNan(gv);
            const tol = @max(@abs(wv) * 5e-3, 1e-30);
            std.testing.expect(both_nan or @abs(gv - wv) <= tol) catch |e| {
                std.debug.print("{d}x{d} (n_pad {d}, k_pad {d}): word {d} (k {d}, row {d}) got {d}, want {d}\n", .{
                    c.rows, c.cols, n_pad, k_pad, i, i / n_pad, i % n_pad, gv, wv,
                });
                return e;
            };
        }
    }
}
