//! CUDA-backend Krea 2 DiT forward (int8 convrot checkpoint), producing the
//! same latent as the CPU/Vulkan paths so the hand-PTX backend can generate
//! like-for-like images. Follows the fallback op sequence (int8 tensor-core
//! GEMMs via `opI8Prep`/`opI8Gemm`, f32 eltwise norm/rope/gate, one naive
//! online-softmax GQA attention kernel). Small/CPU-cheap paths (text fusion,
//! timestep MLPs, patchify, unpatchify) stay on the CPU.
//!
//! Numerics match `DiT.forward` up to floating-point reordering (int8 quant +
//! ex2.approx softmax), the same regime the Vulkan int8 path runs in.
//!
//! The block activations are STORED at `Workspace.act` (`streamDtFor`), the attention
//! block at f16 under either 16-bit stream; the text rows, the patches and the rope
//! table stay f32.

const std = @import("std");
const dit = @import("dit.zig");
const dit_dev = @import("dit_dev.zig");
const lin_cuda = @import("lin_cuda.zig");
const lora_mod = @import("lora.zig");
const cuda = @import("tp_gpu").cuda;
const safetensors = @import("tp_core").safetensors;
const ops = @import("tp_ops");

const DiT = dit.DiT;
const Backend = cuda.Backend;
const DeviceBuffer = cuda.backend.DeviceBuffer;
const Dt = cuda.backend.Dt;
const tp_dtype = @import("tp_core").dtype;
const DType = tp_dtype.DType;
const Weight = ops.matmul.Weight;

const F = dit.features; // 6144
const heads = dit.n_heads; // 48
const kv_heads = dit.n_kv_heads; // 12
const hd = dit.head_dim; // 128
const half = hd / 2; // 64
const mlp_dim = dit.mlp_dim; // 16384
const n_blocks = dit.n_blocks; // 28
const patch = dit.patch; // 2
const channels = dit.channels; // 16
const txt_dim = dit.txt_dim; // 2560
const txt_heads = dit.txt_heads; // 20
const txt_layers = dit.txt_layers; // 12
const txt_mlp_dim = dit.txt_mlp_dim; // 6912
const attn_scale: f32 = 1.0 / 11.313708498984761; // 1/sqrt(128)
const eps: f32 = 1e-5;

/// MLP sequence-tile: the gate/up/down GEMMs run over chunks of this many rows
/// so the mg/mu intermediates are [tile][mlp_dim] instead of [seq][mlp_dim]
/// (512 MiB -> 128 MiB each at 1 MP). The MLP is per-row so chunks are independent.
const mlp_tile: usize = 2048;

/// Zero bias for the block linears (they have none, but opMatmulBf16 always
/// folds a bias in). A file-scope constant so every GEMM width shares one stable
/// host pointer, cachedWeight then caches it once at the widest slice.
const zero_bias: [mlp_dim]f32 = @splat(0);

/// Use the tensor-core GQA attention path (hgemm+softmax_row) instead of the
/// naive one-thread-per-(q,head) kernel. On by default, it is O(seq²) faster on
/// the tensor cores and the naive path is O(seq²) latency-bound. Toggle for A/B.
pub var use_tc_attn: bool = true;

// ==== text fusion (CUDA) =====================================================
//
// Mirrors DiT.txtFusion+textTokens on the device, ~2 TFLOP of GEMM the CPU would
// otherwise stall on. The block is the plain (no modulation, no RoPE) variant of
// the sampling block.

/// Device scratch for the text-fusion blocks, sized for the widest phase (the
/// layerwise blocks over seq_txt*12 rows). Reused by the refiner and txtmlp.
const TxtScratch = struct {
    normed: DeviceBuffer,
    q: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    g: DeviceBuffer,
    attn: DeviceBuffer,
    t1: DeviceBuffer,
    gate: DeviceBuffer, // also holds txtmlp mid (seq_txt*6144 ≤ rows_lw*6912)
    up: DeviceBuffer, // also holds txtmlp out

    fn init(be: *Backend, rows_lw: usize) !TxtScratch {
        var s: TxtScratch = undefined;
        s.normed = try be.tensorCreate(rows_lw * txt_dim * 4);
        s.q = try be.tensorCreate(rows_lw * txt_dim * 4);
        s.k = try be.tensorCreate(rows_lw * txt_dim * 4);
        s.v = try be.tensorCreate(rows_lw * txt_dim * 4);
        s.g = try be.tensorCreate(rows_lw * txt_dim * 4);
        s.attn = try be.tensorCreate(rows_lw * txt_dim * 4);
        s.t1 = try be.tensorCreate(rows_lw * txt_dim * 4);
        s.gate = try be.tensorCreate(rows_lw * txt_mlp_dim * 4);
        s.up = try be.tensorCreate(rows_lw * txt_mlp_dim * 4);
        return s;
    }
    fn deinit(s: *TxtScratch, be: *Backend) void {
        inline for (@typeInfo(TxtScratch).@"struct".fields) |f| be.tensorDestroy(&@field(s, f.name));
    }
};

/// One text-fusion linear, under `Plan.tower`: every dense weight and the integer
/// weights expanded to f16 on the f16 tensor cores, so the text-fusion precision is f16's
/// whatever the checkpoint stores, with its LoRA hits.
fn txtGemm(be: *Backend, stack: ?*const lora_mod.Stack, y: DeviceBuffer, x: DeviceBuffer, m: usize, w: Weight, bias: []const f32) !void {
    try lin_cuda.towerLinear(be, y, x, m, w, bias, stack);
}

/// One TextFusionBlock on the backend: x += attn(rmsnorm(x)); x += mlp(rmsnorm(x)).
/// `x_d` holds n_seqs sequences of seq_len rows of txt_dim (no RoPE, no mask, no
/// modulation). qkNorm with hd=txt_dim is a plain per-row RMS-norm×weight.
fn txtBlockCuda(be: *Backend, stack: ?*const lora_mod.Stack, blk: anytype, x_d: DeviceBuffer, s: *const TxtScratch, zero: []const f32, n_seqs: usize, seq_len: usize) !void {
    const rows = n_seqs * seq_len;

    // --- attention ---
    try be.qkNorm(x_d, s.normed, try normBuf(be, blk.prenorm), rows, txt_dim, eps);
    try txtGemm(be, stack, s.q, s.normed, rows, blk.attn.wq, zero);
    try txtGemm(be, stack, s.k, s.normed, rows, blk.attn.wk, zero);
    try txtGemm(be, stack, s.v, s.normed, rows, blk.attn.wv, zero);
    try txtGemm(be, stack, s.g, s.normed, rows, blk.attn.gate, zero);
    try be.qkNorm(s.q, s.q, try normBuf(be, blk.attn.qnorm), rows * txt_heads, hd, eps);
    try be.qkNorm(s.k, s.k, try normBuf(be, blk.attn.knorm), rows * txt_heads, hd, eps);
    // Each of the n_seqs sequences attends only within itself (12-long layerwise,
    // or the whole prompt for the refiner), the naive kernel handles one at a
    // time; seq_len is small so the launch count is cheap and one-time.
    for (0..n_seqs) |i| {
        const off = i * seq_len * txt_dim * 4;
        try be.attn(s.q.viewBytes(off), s.k.viewBytes(off), s.v.viewBytes(off), s.attn.viewBytes(off), seq_len, seq_len, txt_heads, txt_heads, hd, attn_scale, false);
    }
    try be.sigmoidMul(s.attn, s.g, rows * txt_dim);
    try txtGemm(be, stack, s.t1, s.attn, rows, blk.attn.wo, zero);
    try be.opAdd(x_d, s.t1, rows * txt_dim);

    // --- mlp (swiglu) ---
    try be.qkNorm(x_d, s.normed, try normBuf(be, blk.postnorm), rows, txt_dim, eps);
    try txtGemm(be, stack, s.gate, s.normed, rows, blk.mlp.gate, zero);
    try txtGemm(be, stack, s.up, s.normed, rows, blk.mlp.up, zero);
    try be.siluMul(s.gate, s.up, rows * txt_mlp_dim);
    try txtGemm(be, stack, s.t1, s.gate, rows, blk.mlp.down, zero);
    try be.opAdd(x_d, s.t1, rows * txt_dim);
}

/// CUDA port of `DiT.textTokens`: text conditioning [seq_txt*12*txt_dim] -> the
/// combined-sequence tokens [seq_txt*features] the sampler consumes. Runs the
/// whole txtfusion + txtmlp stack on the backend, so it does not stall the CPU.
/// The weights it caches are dropped before returning.
pub fn textTokensCuda(model: *const DiT, be: *Backend, io: std.Io, gpa: std.mem.Allocator, cond: []const f32) ![]f32 {
    const seq_txt = cond.len / (txt_layers * txt_dim);
    std.debug.assert(cond.len == seq_txt * txt_layers * txt_dim);
    const rows_lw = seq_txt * txt_layers;
    const stack = model.lora;
    // Scratch sized before the weight scope below, so none of it is grown inside it.
    var lins: [4 * 8 + 2]Weight = undefined;
    for (0..2) |i| {
        lins[i * 8 ..][0..8].* = dit.blockLins(&model.txt_layerwise[i]);
        lins[(2 + i) * 8 ..][0..8].* = dit.blockLins(&model.txt_refiner[i]);
    }
    lins[32] = model.txtmlp1.w;
    lins[33] = model.txtmlp3.w;
    var tower = lin_cuda.Plan.tower;
    tower.rot = lins[0].convrot != 0;
    try lin_cuda.presize(be, tower, &lins);
    if (stack) |st| try lin_cuda.presizeLora(be, st, &lins, rows_lw);

    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var x_d = try be.tensorCreate(rows_lw * txt_dim * 4);
    defer be.tensorDestroy(&x_d);
    var s = try TxtScratch.init(be, rows_lw);
    defer s.deinit(be);
    // The text weights and norm buffers are unused during sampling, so they leave the
    // cache with this scope. Not `evictWeights`: that would also drop the resident DiT
    // a persistent Session reuses for the next image.
    be.weightScopeBegin();
    defer be.weightScopeEnd();

    try be.tensorUpload(x_d, std.mem.sliceAsBytes(cond));

    // One max-width zero bias serves every no-bias linear.
    const zero = try arena.alloc(f32, txt_mlp_dim);
    @memset(zero, 0);

    // Layerwise blocks: seq_txt independent 12-long sequences (across the encoder
    // layer axis per token).
    try be.beginBatch();
    errdefer if (be.batching()) be.abortBatch();
    for (&model.txt_layerwise) |*blk| try txtBlockCuda(be, stack, blk, x_d, &s, zero, seq_txt, txt_layers);
    try be.endBatch();

    // Projector: collapse the 12-layer axis (projected[tok][d] = Σ_l pw[l]*x[tok*12+l][d]).
    // Tiny [1,12] contraction, a host round-trip is simpler than a bespoke kernel.
    {
        const work = try arena.alloc(f32, rows_lw * txt_dim);
        try be.tensorDownload(x_d, std.mem.sliceAsBytes(work));
        var pw: [txt_layers]f32 = undefined;
        try model.projectorWeights(io, arena, &pw);
        const projected = try arena.alloc(f32, seq_txt * txt_dim);
        for (0..seq_txt) |tok| {
            const dst = projected[tok * txt_dim ..][0..txt_dim];
            @memset(dst, 0);
            for (0..txt_layers) |l| {
                const src = work[(tok * txt_layers + l) * txt_dim ..][0..txt_dim];
                for (dst, src) |*d, sv| d.* += pw[l] * sv;
            }
        }
        try be.tensorUpload(x_d, std.mem.sliceAsBytes(projected));
    }

    try be.beginBatch();
    // Refiner blocks: one sequence of length seq_txt.
    for (&model.txt_refiner) |*blk| try txtBlockCuda(be, stack, blk, x_d, &s, zero, 1, seq_txt);

    // txtmlp: rmsnorm -> Linear(2560->6144) -> geluTanh -> Linear(6144->6144).
    try be.qkNorm(x_d, s.normed, try normBuf(be, model.txtmlp_norm), seq_txt, txt_dim, eps);
    try txtGemm(be, stack, s.gate, s.normed, seq_txt, model.txtmlp1.w, model.txtmlp1.b.?);
    try be.gelu(s.gate, seq_txt * F);
    try txtGemm(be, stack, s.up, s.gate, seq_txt, model.txtmlp3.w, model.txtmlp3.b.?);
    try be.endBatch();

    const out = try gpa.alloc(f32, seq_txt * F);
    errdefer gpa.free(out);
    try be.tensorDownload(s.up, std.mem.sliceAsBytes(out));
    return out;
}

/// Per-run constants: text-fusion tokens + rope table, uploaded once.
pub const Session = struct {
    seq_txt: usize,
    lat_h: usize,
    lat_w: usize,
    txt0_d: DeviceBuffer,
    txt_len: usize, // element count (seq_txt * F)
    freqs_d: DeviceBuffer,
    /// Steering directions, uploaded once. Empty when steering is off.
    dirs_d: DeviceBuffer = .{},
    /// What `lin_cuda.plan` decided for this checkpoint's block linears.
    plan: lin_cuda.Plan,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, be: *Backend, model: *const DiT, lat_h: usize, lat_w: usize, cond: []const f32, seq_txt: usize) !Session {
        const h = lat_h / patch;
        const w = lat_w / patch;
        const seq = seq_txt + h * w;
        // Refuse a checkpoint the GEMM dispatch cannot run here, by name, rather than
        // inside the first forward. krea2's widths (6144, 1536, 16384) clear every floor.
        var plan = try lin_cuda.plan(model.device_lins, .{ .blockq = lin_cuda.blockq_gemm }, "dit cuda");
        try lin_cuda.presize(be, plan, model.device_lins);
        // The patch embed and output projection run on the fused `opMatmul`.
        try lin_cuda.attachLora(be, &plan, model.lora, model.device_lins, seq);

        const txt_tokens = try textTokensCuda(model, be, io, gpa, cond);
        defer gpa.free(txt_tokens);
        var txt0_d = try be.tensorCreate(txt_tokens.len * 4);
        errdefer be.tensorDestroy(&txt0_d);
        try be.tensorUpload(txt0_d, std.mem.sliceAsBytes(txt_tokens));

        var freqs = try DiT.ropeFreqs(gpa, seq_txt, h, w);
        defer freqs.deinit(gpa);
        std.debug.assert(freqs.half == half);
        const fp = try gpa.alloc(f32, 2 * seq * half);
        defer gpa.free(fp);
        @memcpy(fp[0 .. seq * half], freqs.cos);
        @memcpy(fp[seq * half ..], freqs.sin);
        var freqs_d = try be.tensorCreate(fp.len * 4);
        errdefer be.tensorDestroy(&freqs_d);
        try be.tensorUpload(freqs_d, std.mem.sliceAsBytes(fp));

        // The model carries the steering set before a session is built, so the
        // directions upload once here rather than per forward.
        var dirs_d: DeviceBuffer = .{};
        errdefer be.tensorDestroy(&dirs_d);
        if (model.act_steer) |st| if (st.dirs.len > 0) {
            dirs_d = try be.tensorCreate(st.dirs.len * 4);
            try be.tensorUpload(dirs_d, std.mem.sliceAsBytes(st.dirs));
        };

        return .{
            .seq_txt = seq_txt,
            .lat_h = lat_h,
            .lat_w = lat_w,
            .txt0_d = txt0_d,
            .txt_len = txt_tokens.len,
            .freqs_d = freqs_d,
            .dirs_d = dirs_d,
            .plan = plan,
        };
    }

    pub fn deinit(self: *Session, be: *Backend) void {
        be.tensorDestroy(&self.txt0_d);
        be.tensorDestroy(&self.freqs_d);
        be.tensorDestroy(&self.dirs_d);
    }
};

/// Whether the plan's LoRA stack patches any of `ws`.
fn patched(plan: lin_cuda.Plan, ws: []const Weight) bool {
    const st = plan.lora orelse return false;
    for (ws) |w| if (st.forWeight(w).len > 0) return true;
    return false;
}

fn normBuf(be: *Backend, w: []const f32) !DeviceBuffer {
    return .{ .buf = try be.smallBuffer(std.mem.sliceAsBytes(w)), .mem = .null_handle, .size = w.len * 4 };
}

/// Queue all of a DiT block's streamable weights for async prefetch (called one
/// block ahead so the uploads overlap the previous block's compute). Keys must
/// match the byte slices `forward` later fetches (same host pointers -> cache hit).
fn prefetchBlock(be: *Backend, blk: anytype) void {
    const bytes = std.mem.sliceAsBytes;
    inline for (.{ blk.attn.wq, blk.attn.wk, blk.attn.wv, blk.attn.gate, blk.attn.wo }) |w| {
        be.prefetchWeight(w.bytes);
        if (w.row_scale) |rs| be.prefetchWeight(bytes(rs));
    }
    be.prefetchWeight(bytes(blk.attn.qnorm));
    be.prefetchWeight(bytes(blk.attn.knorm));
    inline for (.{ blk.mlp.gate, blk.mlp.up, blk.mlp.down }) |w| {
        be.prefetchWeight(w.bytes);
        if (w.row_scale) |rs| be.prefetchWeight(bytes(rs));
    }
}

/// The dtype the block activations are stored at when `want` is asked for. Only cuDNN's
/// attention reads 16-bit operands (the hand-PTX arm and `use_tc_attn = false` read
/// f32), and an activation capture reads what the f32 stream holds. A 16-bit stream with
/// f32 attention operands was measured on zig-cuda and does not pay: neutral on fp8 and
/// bf16, 2-4% slower on int8, whose hand-PTX GEMM stores only f32.
pub fn streamDtFor(be: *const Backend, want: Dt) Dt {
    if (want == .f32 or ops.matmul.probe != null or be.kernels != .libs or !use_tc_attn) return .f32;
    return want;
}

/// Per-run device scratch shared across sampler steps (and both CFG sessions):
/// the ~12 activation buffers a forward would otherwise `cuMemAlloc`/free every call.
/// Sized once for the largest sequence (both prompts share n_img, only seq_txt differs).
pub const Workspace = struct {
    x_d: DeviceBuffer = .{},
    imgin_d: DeviceBuffer = .{},
    mv_d: DeviceBuffer = .{},
    fin_d: DeviceBuffer = .{},
    t1_d: DeviceBuffer = .{},
    q_d: DeviceBuffer = .{},
    k_d: DeviceBuffer = .{},
    v_d: DeviceBuffer = .{},
    g_d: DeviceBuffer = .{},
    attn_d: DeviceBuffer = .{},
    mg_d: DeviceBuffer = .{},
    mu_d: DeviceBuffer = .{},
    /// The storage of `x_d`, `t1_d`, `q_d`..`g_d`, `attn_d`, `mg_d` and `mu_d`.
    act: Dt = .f32,

    pub fn init(be: *Backend, lat_h: usize, lat_w: usize, seq_txt_cap: usize, act: Dt) !Workspace {
        const n_img = (lat_h / patch) * (lat_w / patch);
        const mpad = std.mem.alignForward(usize, seq_txt_cap + n_img, 128);
        const ew = act.bytes();
        var ws: Workspace = .{ .act = act };
        errdefer ws.deinit(be);
        ws.x_d = try be.tensorCreate(mpad * F * ew);
        ws.imgin_d = try be.tensorCreate(n_img * channels * patch * patch * 4);
        ws.mv_d = try be.tensorCreate(n_blocks * 6 * F * 4);
        ws.fin_d = try be.tensorCreate(2 * F * 4);
        ws.t1_d = try be.tensorCreate(mpad * F * ew);
        ws.q_d = try be.tensorCreate(mpad * heads * hd * ew);
        ws.k_d = try be.tensorCreate(mpad * kv_heads * hd * ew);
        ws.v_d = try be.tensorCreate(mpad * kv_heads * hd * ew);
        ws.g_d = try be.tensorCreate(mpad * F * ew);
        ws.attn_d = try be.tensorCreate(mpad * heads * hd * ew);
        // mg/mu hold one MLP tile (see mlp_tile), not the full padded sequence.
        const mlp_rows = @min(mpad, std.mem.alignForward(usize, mlp_tile, 128));
        ws.mg_d = try be.tensorCreate(mlp_rows * mlp_dim * ew);
        ws.mu_d = try be.tensorCreate(mlp_rows * mlp_dim * ew);
        inline for (.{ "x_d", "t1_d", "mg_d", "mu_d" }) |f| @field(ws, f).dt = act;
        // The attention block is f16 under either 16-bit stream, which is what cuDNN is
        // handed under an f32 one. bf16 operands moved krea2's velocity 9.6e-2 against
        // f32 (fp8, 512²), f16 ones 9.5e-3.
        inline for (.{ "q_d", "k_d", "v_d", "g_d", "attn_d" }) |f| @field(ws, f).dt = if (act == .f32) .f32 else .f16;
        return ws;
    }

    pub fn deinit(self: *Workspace, be: *Backend) void {
        inline for (@typeInfo(Workspace).@"struct".fields) |f| {
            if (f.type == DeviceBuffer) be.tensorDestroy(&@field(self, f.name));
        }
    }
};

/// One DiT forward evaluation: velocity `out` from latent `x_lat` at `sigma`.
/// `ws` must be sized for a sequence ≥ seq_txt + n_img (see Workspace.init).
pub fn forward(model: *const DiT, be: *Backend, sess: *const Session, ws: *const Workspace, io: std.Io, gpa: std.mem.Allocator, out: []f32, x_lat: []const f32, sigma: f32, cancel: ?*std.atomic.Value(bool)) !void {
    const lat_h = sess.lat_h;
    const lat_w = sess.lat_w;
    const seq_txt = sess.seq_txt;
    const n_img = (lat_h / patch) * (lat_w / patch);

    const tvv = try model.timestepVectors(io, gpa, sigma);
    defer gpa.free(tvv.t);
    defer gpa.free(tvv.tvec);
    const mv = try gpa.alloc(f32, n_blocks * 6 * F);
    defer gpa.free(mv);
    const fin = try gpa.alloc(f32, 2 * F);
    defer gpa.free(fin);
    model.modulation(tvv.t, tvv.tvec, mv, fin);
    const img_in = try DiT.patchify(gpa, x_lat, lat_h, lat_w);
    defer gpa.free(img_in);

    try be.tensorUpload(ws.mv_d, std.mem.sliceAsBytes(mv));
    try be.tensorUpload(ws.fin_d, std.mem.sliceAsBytes(fin));
    if (ws.act == .f32)
        try be.tensorCopy(ws.x_d, 0, sess.txt0_d, 0, sess.txt_len * 4)
    else
        try be.opCast(sess.txt0_d, ws.x_d, sess.txt_len, 0);
    try be.tensorUpload(ws.imgin_d, std.mem.sliceAsBytes(img_in));

    try be.beginBatch();
    errdefer if (be.batching()) be.abortBatch();
    const d: Dev = .{
        .model = model,
        .be = be,
        .sess = sess,
        .ws = ws,
        .gpa = gpa,
        .bufs = .{ .x = ws.x_d, .t1 = ws.t1_d, .attn = ws.attn_d, .mv = ws.mv_d, .fin = ws.fin_d },
    };
    try dit_dev.body(d, model, seq_txt, n_img, cancel);
    try be.endBatch();
    // A 16-bit stream that overflowed can leave the output finite (`allFinite`).
    if (ws.act != .f32 and !try be.allFinite(ws.x_d, (seq_txt + n_img) * F)) return error.NonFiniteActivation;

    const final_rows = try gpa.alloc(f32, n_img * channels * patch * patch);
    defer gpa.free(final_rows);
    try be.tensorDownload(ws.imgin_d, std.mem.sliceAsBytes(final_rows));
    DiT.unpatchify(out, final_rows, lat_h, lat_w);
}

/// `dit_dev`'s ops on this backend. It never stages: its fp8 and int8 GEMMs convert or
/// quantize the f32 activation themselves.
const Dev = struct {
    model: *const DiT,
    be: *Backend,
    sess: *const Session,
    ws: *const Workspace,
    gpa: std.mem.Allocator,
    bufs: Bufs,

    pub const can_stage = false;
    const Bufs = struct { x: DeviceBuffer, t1: DeviceBuffer, attn: DeviceBuffer, mv: DeviceBuffer, fin: DeviceBuffer };

    pub fn patchEmbed(d: Dev, seq_txt: usize, n_img: usize) !void {
        try lin_cuda.linear(d.be, d.sess.plan, d.ws.x_d.viewBytes(seq_txt * F * d.ws.act.bytes()), d.ws.imgin_d, n_img, d.model.first.w, d.model.first.b);
    }

    /// Each block prefetches the NEXT block's weights so the upload overlaps this one's
    /// compute (async streaming); block 0 is fetched here too.
    pub fn blockBegin(d: Dev, b: usize) !void {
        if (!d.be.async_uploads) return;
        if (b == 0) prefetchBlock(d.be, d.model.blocks[0]);
        if (b + 1 < d.model.blocks.len) prefetchBlock(d.be, d.model.blocks[b + 1]);
    }

    pub fn attnPlan(d: Dev, _: *const dit.Block, _: usize) !dit_dev.AttnPlan(DeviceBuffer) {
        const ws = d.ws;
        return .{ .stage = null, .q = ws.q_d, .k = ws.k_d, .v = ws.v_d, .g = ws.g_d, .wo_stage = null, .wo_out = ws.t1_d };
    }

    /// Under an f32 stream, an f16 chain through the MLP where every linear is int8 on
    /// the cuBLASLt arm and none is patched: gate/up emit f16, silu_mul reads and writes
    /// it, the down prep reads it, halving the 16384-wide traffic. Per block, since a
    /// mixed checkpoint may keep some blocks dense; a sidecar accumulates in f32.
    pub fn mlpPlan(d: Dev, blk: *const dit.Block, _: usize) !dit_dev.MlpPlan(DeviceBuffer) {
        const ws = d.ws;
        const lins = [_]Weight{ blk.mlp.gate, blk.mlp.up, blk.mlp.down };
        const chain16 = d.be.kernels == .libs and lin_cuda.allI8(d.sess.plan, &lins) and !patched(d.sess.plan, &lins);
        const mdt: Dt = if (ws.act != .f32) ws.act else if (chain16) .f16 else .f32;
        return .{ .tile = mlp_tile, .stage = null, .mg = ws.mg_d.as(mdt), .mu = ws.mu_d.as(mdt), .down_stage = null, .down_out = ws.t1_d };
    }

    pub fn rmsMod(d: Dev, src: DeviceBuffer, dst: DeviceBuffer, mod: DeviceBuffer, premul: usize, shift: usize, rows: usize) !void {
        try d.be.rmsMod(src, dst, mod, rows, F, premul, shift, eps);
    }

    pub fn prep(d: Dev, x: DeviceBuffer, m: usize, cols: usize, group: []const Weight) !lin_cuda.Group {
        return lin_cuda.prep(d.be, d.sess.plan, x, m, cols, group);
    }

    pub fn gemms(d: Dev, g: lin_cuda.Group, ys: []const DeviceBuffer, group: []const Weight) !void {
        for (ys, group) |y, w| try lin_cuda.gemm(d.be, d.sess.plan, g, y, w, null);
    }

    pub fn linear(d: Dev, y: DeviceBuffer, x: DeviceBuffer, m: usize, w: Weight) !void {
        try lin_cuda.linear(d.be, d.sess.plan, y, x, m, w, null);
    }

    pub fn attention(d: Dev, blk: *const dit.Block, ap: dit_dev.AttnPlan(DeviceBuffer), seq: usize) !void {
        const be = d.be;
        const sin_off = seq * half;
        try be.qkNorm(ap.q, ap.q, try normBuf(be, blk.attn.qnorm), seq * heads, hd, eps);
        try be.qkNorm(ap.k, ap.k, try normBuf(be, blk.attn.knorm), seq * kv_heads, hd, eps);
        try be.rope(ap.q, d.sess.freqs_d, seq, heads, half, sin_off);
        try be.rope(ap.k, d.sess.freqs_d, seq, kv_heads, half, sin_off);
        if (use_tc_attn)
            try be.opAttnTC(ap.q, ap.k, ap.v, d.ws.attn_d, seq, heads, kv_heads, hd, attn_scale)
        else
            try be.attn(ap.q, ap.k, ap.v, d.ws.attn_d, seq, seq, heads, kv_heads, hd, attn_scale, false);
    }

    pub fn gateMul(d: Dev, attn: DeviceBuffer, g: DeviceBuffer, seq: usize) !void {
        try d.be.sigmoidMul(attn, g, seq * F);
    }

    pub fn gatedAdd(d: Dev, x: DeviceBuffer, t: DeviceBuffer, mod: DeviceBuffer, off: usize, rows: usize) !void {
        try d.be.gatedAdd(x, t, mod, rows * F, F, off);
    }

    pub fn rowsAt(_: Dev, x: DeviceBuffer, r0: usize, width: usize) !DeviceBuffer {
        return x.viewBytes(r0 * width * x.dt.bytes());
    }

    pub fn siluMul(d: Dev, mg: DeviceBuffer, mu: DeviceBuffer, tile: usize) !void {
        // The f32 stream's f16 chain keeps its paired kernel.
        if (mg.dt == .f16 and d.ws.act == .f32) try d.be.siluMul16(mg, mu, tile * mlp_dim) else try d.be.siluMul(mg, mu, tile * mlp_dim);
    }

    /// Reads the residual back and reuses the CPU accumulator rather than growing a
    /// kernel: one download per block is ~16 MB at 256², against the tens of seconds a
    /// CPU forward costs.
    pub fn steerCapture(d: Dev, st: dit.ActSteer, seq: usize, seq_txt: usize, b: usize) !void {
        const n = seq * F;
        const host = try d.gpa.alloc(f32, n);
        defer d.gpa.free(host);
        try d.be.downloadF32(d.ws.x_d, host);
        st.run(host, seq, seq_txt, b);
    }

    pub fn steer(d: Dev, st: dit.ActSteer, seq: usize, seq_txt: usize, b: usize, sc: f32) !void {
        if (d.sess.dirs_d.size == 0) return;
        try d.be.actSteer(d.ws.x_d, d.sess.dirs_d, seq - seq_txt, F, b * F, seq_txt, st.op == .add, st.keep_norm, sc);
    }

    /// The image rows only, an offset view past the text prefix.
    pub fn finalLinear(d: Dev, seq_txt: usize, n_img: usize) !void {
        try lin_cuda.linear(d.be, d.sess.plan, d.ws.imgin_d, d.ws.t1_d.viewBytes(seq_txt * F * d.ws.act.bytes()), n_img, d.model.last_linear.w, d.model.last_linear.b);
    }
};

// --- tests -----------------------------------------------------------------

// The CUDA W4A8 decode kernel against the CPU decode it must reproduce. Synthetic
// weights, so no checkpoint is needed, which is the point: the kernel's only earlier
// validation was that a real render came out bit-identical to the load-time
// materialization, and that check stops being reproducible the moment the checkpoint
// leaves the disk.
//
// Covers what the PTX quietly assumes and a render would not localize: the `prmt.b32`
// byte packing (its selector nibbles must stay under 8 or the instruction
// sign-replicates), the `v2.u32` store's 8-byte alignment, and the one-group-scale-per-
// thread shortcut that makes `group_size % 8 == 0` a requirement.
test "the CUDA W4A8 decode matches ops.w4a8.decode" {
    const gpa = std.testing.allocator;
    const w4a8 = ops.w4a8;
    const test_gate = @import("../test_gate.zig");
    try test_gate.requireIntegration();
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    const cases = [_]struct { rows: usize, cols: usize, gs: usize }{
        .{ .rows = 256, .cols = 512, .gs = 16 },
        .{ .rows = 128, .cols = 1024, .gs = 32 },
        .{ .rows = 64, .cols = 256, .gs = 8 }, // the smallest group the kernel allows
    };
    // Every case's buffers stay ALIVE in one arena, and that is load-bearing rather
    // than tidy: both device weight caches key on the HOST POINTER, so freeing a case's
    // arrays and letting the allocator hand the same address to the next case scores a
    // stale cache hit and the kernel reads the previous case's weights. Found the hard
    // way, this test failed on its third case with the first case's data. (A model
    // never hits it: the weights live in the model arena for the model's lifetime.)
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const alloc = arena.allocator();
    for (cases) |c| {
        var rnd = std.Random.DefaultPrng.init(0xA8 + c.cols);
        const r = rnd.random();
        const packed_bytes = try alloc.alloc(u8, c.rows * c.cols / 2);
        r.bytes(packed_bytes);
        // Every fp8 byte including NaN (0x7F/0xFF); the level table sends those to 0 on
        // both sides, so this pins that they agree rather than diverging where neither
        // reference defines a value.
        const s_rel = try alloc.alloc(u8, c.rows * c.cols / c.gs);
        r.bytes(s_rel);
        const levels = try alloc.create(w4a8.Levels);
        levels.* = w4a8.Levels.init(&w4a8.fixed_lut);

        const want = try alloc.alloc(i8, c.rows * c.cols);
        w4a8.decode(want, packed_bytes, s_rel, .{ .w4 = levels }, c.rows, c.cols, c.gs);

        const db = try be.w4a8ToI8(packed_bytes, s_rel, std.mem.asBytes(levels), c.rows, c.cols, c.gs);
        const got = try alloc.alloc(i8, c.rows * c.cols);
        try be.tensorDownload(db, std.mem.sliceAsBytes(got));

        for (want, got, 0..) |wv, gv, i| {
            std.testing.expectEqual(wv, gv) catch |e| {
                std.debug.print("rows {d} cols {d} gs {d}: element {d} (row {d}, col {d}) got {d}, want {d}\n", .{
                    c.rows, c.cols, c.gs, i, i / c.cols, i % c.cols, gv, wv,
                });
                return e;
            };
        }
    }
}

// The CUDA NVFP4 decode kernel against the CPU decode it must reproduce. Synthetic
// weights, so no checkpoint is needed.
//
// f16 output, so this is a TOLERANCE against the f32 CPU decode rather than an
// equality, but a tight one, because both sides read the same level table and f16 only
// rounds the store. The bound is set from f16's own quantum, so a nibble-order or
// block-index error (which move values by whole levels) cannot hide inside it.
test "the CUDA NVFP4 decode matches ops.nvfp4.decode" {
    const gpa = std.testing.allocator;
    const nvfp4 = ops.nvfp4;
    const test_gate = @import("../test_gate.zig");
    try test_gate.requireIntegration();
    const be = Backend.init(gpa) catch return error.SkipZigTest;
    defer be.deinit();

    const cases = [_]struct { rows: usize, cols: usize }{
        .{ .rows = 128, .cols = 256 },
        .{ .rows = 64, .cols = 1024 },
        .{ .rows = 16, .cols = 16 }, // one block per row
    };
    // One arena for every case: the weight cache keys on the HOST POINTER, so reusing an
    // address across cases would score a stale hit and decode the previous case's bytes.
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const alloc = arena.allocator();

    for (cases) |c| {
        var rnd = std.Random.DefaultPrng.init(0xF4 + c.cols);
        const r = rnd.random();
        const packed_bytes = try alloc.alloc(u8, c.rows * c.cols / 2);
        r.bytes(packed_bytes);
        // Every fp8 byte including NaN: the level table maps those identically on both
        // sides, so this pins that they agree rather than diverging where nothing defines
        // a value.
        const scales = try alloc.alloc(u8, c.rows * c.cols / nvfp4.block_size);
        r.bytes(scales);
        const levels = try alloc.create(nvfp4.Levels);
        levels.* = nvfp4.Levels.init(0.0125);
        const meta: nvfp4.Meta = .{ .scales = scales, .levels = levels };

        const want = try alloc.alloc(f32, c.rows * c.cols);
        nvfp4.decode(want, packed_bytes, meta, c.rows, c.cols);

        const db = try be.nvfp4ToBf16(packed_bytes, scales, std.mem.asBytes(&levels.bf16v), c.rows, c.cols);
        const got_bits = try alloc.alloc(u16, c.rows * c.cols);
        try be.tensorDownload(db, std.mem.sliceAsBytes(got_bits));

        for (want, got_bits, 0..) |wv, gb, i| {
            const gv: f32 = @bitCast(@as(u32, gb) << 16); // bf16 -> f32
            // A NaN block-scale byte (fp8 0x7F/0xFF) PROPAGATES here, unlike `.w4a8`
            // whose int8 target forces it to 0, and it propagates in the reference too,
            // so agreeing on NaN is the correct outcome, not a hole in the check.
            const both_nan = std.math.isNan(wv) and std.math.isNan(gv);
            // Otherwise bf16 rounds the value and nothing else may change it: 8 mantissa
            // bits is 2^-8 relative. A wrong nibble or block index moves by at least one
            // E2M1 level (>= 1/6 of the value), far outside this.
            const tol = @max(@abs(wv) * 5e-3, 1e-30);
            std.testing.expect(both_nan or @abs(gv - wv) <= tol) catch |e| {
                std.debug.print("{d}x{d}: element {d} (row {d}, col {d}) got {d}, want {d}\n", .{
                    c.rows, c.cols, i, i / c.cols, i % c.cols, gv, wv,
                });
                return e;
            };
        }
    }
}
