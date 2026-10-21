//! CUDA-backend MiniMax H3 trunk forward, the device twin of `minimax_h3.forward`.
//! The walk itself is `minimax_h3_dev.zig`; this is its adapter (`Dev`) and what the
//! backend owns.
//!
//! The split follows `dit_cuda`'s: the 50-block trunk and the output heads run on the
//! device and the small host-cheap paths stay on the CPU. Here those are the patch
//! projections, the adaLN projection, the token refiner, and the patchify/pack
//! transposes. Their combined cost at the default render is under 0.01% of a step,
//! and `adaln_proj` is `[96768, 8]`, whose 8 columns are below every tiled path's
//! floor anyway.
//!
//! Three things differ from `dit_cuda` and each is load-bearing:
//!
//! 1. **Modulation is PER SEGMENT, not per sequence.** krea2 has one modulation
//!    vector per block; H3 has one per (timestep row, modality tag), and which
//!    one a row reads depends on which segment it is in. Segments are CONTIGUOUS,
//!    so each is one `rms_mod`/`gated_add` launch over its row range with its own
//!    modulation offset. A handful of extra launches per block, no new kernel.
//! 2. **`rms_mod_par` applies no norm weight and no `1 +`**: it computes
//!    `rmsnorm(x) * premul + shift`. So the host folds `norm.weight * (1 + scale)`
//!    into `premul`. Uploading a bare `scale` there drops the norm weight AND the
//!    identity term, which is finite and wrong.
//! 3. **The fused weights are SPLIT BY ROWS on the host.** `qkv_proj` is one
//!    `[3 * inner, hidden]` tensor and `fc1` one `[2 * ffn, hidden]`, but the
//!    device wants separate q/k/v and gate/up buffers. A row range of a row-major
//!    weight is a contiguous byte range and its per-row scales slice with it, so
//!    the split is three (or two) `Weight` views over the same mapping, with no
//!    copy and no de-interleave kernel. The views are `DiT.device_lins`, built at
//!    load, and every one goes through `lin_cuda` like any family's linear.
//!
//! Numerics match `minimax_h3.forward` up to int8 quantization and the softmax
//! approximation, the same regime `dit_cuda` runs in. `minimax-h3-cuda-test`
//! checks it against the CPU forward on real weights.

const std = @import("std");
const minimax_h3 = @import("minimax_h3.zig");
const lin_cuda = @import("lin_cuda.zig");
const minimax_h3_dev = @import("minimax_h3_dev.zig");
const lin_mod = @import("lin.zig");
const lora_mod = @import("lora.zig");
const cuda = @import("tp_gpu").cuda;
const ops = @import("tp_ops");

const DiT = minimax_h3.DiT;
const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;
const Weight = ops.matmul.Weight;

/// Force the naive one-thread-per-(query, head) attention instead of the
/// tensor-core path. For A/B and for reproducing a mismatch.
pub var force_naive_attn: bool = false;

/// Run integer weights weight-only, so a device check can compare activation dtypes
/// without the W8A8 prep's re-roll noise.
pub var force_weight_only: bool = false;

const lin_use: lin_cuda.Use = .{ .blockq = .auto };

/// Whether this checkpoint's trunk, and the LoRA stack on it, can run here.
pub fn supported(dit: *const DiT) bool {
    switch (lin_cuda.check(dit.device_lins, lin_use)) {
        .ok => {},
        .refused => return false,
    }
    return lin_cuda.loraRunnable(dit.lora, dit.device_lins, "minimax_h3 cuda");
}

/// Per-render device state: everything constant across sampling steps.
pub const Session = struct {
    seq: usize,
    /// `[seq * pairs]` cos then `[seq * pairs]` sin, f32. `sin_off` is the split.
    freqs_d: Buf = .{},
    pairs: usize,

    pub fn init(be: *Backend, gpa: std.mem.Allocator, dit: *const DiT, layout: *const minimax_h3.PackedLayout) !Session {
        const cfg = dit.cfg;
        const pairs = cfg.ropePairs();
        var s: Session = .{ .seq = layout.seq_len, .pairs = pairs };
        errdefer s.deinit(be);

        // The same table the CPU path builds, uploaded once: it depends only on
        // the packed layout's position grid, which is fixed for a render.
        var freqs = try minimax_h3.ropeFreqs(gpa, layout.pos, dit.rope_inv_freq);
        defer freqs.deinit(gpa);
        const host = try gpa.alloc(f32, 2 * layout.seq_len * pairs);
        defer gpa.free(host);
        @memcpy(host[0 .. layout.seq_len * pairs], freqs.cos);
        @memcpy(host[layout.seq_len * pairs ..], freqs.sin);
        s.freqs_d = try be.tensorCreate(host.len * 4);
        try be.tensorUpload(s.freqs_d, std.mem.sliceAsBytes(host));
        return s;
    }

    pub fn sinOff(self: Session) usize {
        return self.seq * self.pairs;
    }

    pub fn deinit(self: *Session, be: *Backend) void {
        be.tensorDestroy(&self.freqs_d);
    }
};

/// Per-shape device scratch.
pub const Workspace = struct {
    x_d: Buf = .{},
    t1_d: Buf = .{},
    q_d: Buf = .{},
    k_d: Buf = .{},
    v_d: Buf = .{},
    attn_d: Buf = .{},
    /// The attention out-projection for one query band. Its own buffer, not a
    /// view into `t1_d`: `opI8Gemm` writes its row count ROUNDED UP to 128, so a
    /// band-offset view would clobber the next band's still-needed normalized
    /// rows, and the last band would run off the end.
    ao_d: Buf = .{},
    /// Q and the attention output for one band, in the attention's own bf16.
    /// Only allocated when `kv_bf16`.
    qb_d: Buf = .{},
    ob_d: Buf = .{},
    /// K and V are stored as bf16 and handed to cuDNN AS the operands, so the
    /// conversion staging that an f32 stream needs does not exist: that staging
    /// is another full copy of both, 2.62 GB at 15 s of 768x1152 on top of the
    /// 2.62 GB the f32 originals cost. bf16, not f16, because H3's V is unnormed and
    /// outgrows f16 on a real conditioning while a synthetic one stays green. Off on
    /// the hand-PTX arm, which has no bf16 attention, and under `force_naive_attn`.
    kv_bf16: bool = false,
    gate_d: Buf = .{},
    up_d: Buf = .{},
    mod_d: Buf = .{},
    /// The final layer's modulation, `[max_labels][2][hidden]`, norm folded in.
    fmod_d: Buf = .{},
    /// Per-row modulation LABEL indices (u32) for the two target segments, used
    /// only when a denoise mask relabels rows inside them. `seq` entries is an
    /// over-allocation of a few hundred KB, and sizing it from the mask would mean
    /// the workspace depended on something the layout does not carry.
    vmask_d: Buf = .{},
    amask_d: Buf = .{},
    /// What `lin_cuda.plan` decided for this checkpoint, the LoRA stack on it.
    plan: lin_cuda.Plan = undefined,

    const attn_band = minimax_h3_dev.attn_band;
    const mlp_tile = minimax_h3_dev.mlp_tile;
    const padRows = minimax_h3_dev.padRows;

    /// Sized for `Timesteps.max_labels`, NOT for the count at any one sigma.
    ///
    /// The distinct-timestep count CHANGES along the schedule: at sigma 1.0 the
    /// two streams and their condition pins collapse to 3 labels, and mid-schedule
    /// there are 4. Sizing from one sigma and uploading another's modulation
    /// overruns the buffer, which surfaces as a `CudaError` from the upload two
    /// steps into a render. The bound is small enough that reserving it is free.
    pub fn init(be: *Backend, dit: *const DiT, seq: usize, act: cuda.backend.Dt) !Workspace {
        const cfg = dit.cfg;
        const inner = cfg.n_heads * cfg.head_dim;
        const mpad = padRows(seq);
        var ws: Workspace = .{};
        errdefer ws.deinit(be);
        ws.x_d = try be.tensorCreate(mpad * cfg.hidden * act.bytes());
        ws.x_d.dt = act;
        // Band-sized, not sequence-sized: every consumer (the qkv projections, the
        // MLP, the output heads) now works a band at a time, and the widest band
        // is the attention's.
        const tband = padRows(@min(@max(attn_band, mlp_tile), seq));
        ws.t1_d = try be.tensorCreate(tband * cfg.hidden * act.bytes());
        ws.t1_d.dt = act;
        const aband = padRows(@min(attn_band, seq));
        ws.kv_bf16 = be.kernels == .libs and !force_naive_attn;
        const kvw: usize = if (ws.kv_bf16) 2 else 4;
        ws.q_d = try be.tensorCreate(aband * inner * 4);
        ws.k_d = try be.tensorCreate(mpad * inner * kvw);
        ws.v_d = try be.tensorCreate(mpad * inner * kvw);
        ws.attn_d = try be.tensorCreate(aband * inner * 4);
        ws.ao_d = try be.tensorCreate(aband * cfg.hidden * 4);
        if (ws.kv_bf16) {
            ws.qb_d = try be.tensorCreate(aband * inner * 2);
            ws.ob_d = try be.tensorCreate(aband * inner * 2);
        }
        const tile = padRows(@min(mlp_tile, seq));
        ws.gate_d = try be.tensorCreate(tile * cfg.ffn * 4);
        ws.up_d = try be.tensorCreate(tile * cfg.ffn * 4);
        // All blocks' modulation in one buffer, so the whole step uploads once.
        ws.mod_d = try be.tensorCreate(dit.blocks.len * minimax_h3.Timesteps.max_labels * 3 * 6 * cfg.hidden * 4);
        ws.fmod_d = try be.tensorCreate(minimax_h3.Timesteps.max_labels * 2 * cfg.hidden * 4);
        ws.vmask_d = try be.tensorCreate(seq * 4);
        ws.amask_d = try be.tensorCreate(seq * 4);
        ws.plan = try lin_cuda.plan(dit.device_lins, lin_use, "minimax_h3 cuda");
        if (force_weight_only) ws.plan = lin_cuda.withInt8(ws.plan, be, .weight_only);
        try lin_cuda.presize(be, ws.plan, dit.device_lins);
        try lin_cuda.presizeRows(be, ws.plan, dit.device_lins, mpad);
        try lin_cuda.attachLora(be, &ws.plan, dit.lora, dit.device_lins, mpad);
        // The attention's own device memory, reserved at the shape the forward
        // will ask for (full pack, MHA). cuDNN's plan workspace alone is 1.3 GB at
        // 15 seconds of 768x1152, and allocating it lazily inside the first
        // forward puts it AFTER the residency pass that decides how much of the
        // trunk to pin, which is how a long clip failed where a short one fit.
        if (!force_naive_attn) {
            // Reserved in the SHAPES and the DTYPE the forward will ask for. The
            // forward runs banded, so the plans are (band, seq) and (tail, seq),
            // never (seq, seq); and under bf16 storage the operands arrive in the
            // plan's own type, so there is no staging to reserve. Getting either
            // wrong reserves buffers nothing uses and leaves the real ones to be
            // allocated mid-forward, after residency has already pinned against
            // the space they need -- which is the whole reason this exists.
            const io: cuda.backend.Dt = if (ws.kv_bf16) .bf16 else .f32;
            const full = @min(attn_band, seq);
            try be.reserveAttnCudnn(full, seq, cfg.n_heads, cfg.n_heads, cfg.head_dim, io);
            const tail = seq % attn_band;
            if (tail != 0 and tail != full) try be.reserveAttnCudnn(tail, seq, cfg.n_heads, cfg.n_heads, cfg.head_dim, io);
        }
        return ws;
    }

    pub fn deinit(self: *Workspace, be: *Backend) void {
        inline for (.{ &self.x_d, &self.t1_d, &self.q_d, &self.k_d, &self.v_d, &self.attn_d, &self.ao_d, &self.qb_d, &self.ob_d, &self.gate_d, &self.up_d, &self.mod_d, &self.fmod_d, &self.vmask_d, &self.amask_d }) |b| {
            be.tensorDestroy(b);
        }
        self.* = undefined;
    }
};

/// One trunk forward on the device.
///
/// The host does everything outside the 50 blocks; `in` and the outputs are the
/// same shapes `minimax_h3.forward` takes, so the two are drop-in alternatives.
pub fn forward(
    dit: *const DiT,
    be: *Backend,
    sess: *const Session,
    ws: *Workspace,
    io: std.Io,
    gpa: std.mem.Allocator,
    layout: *const minimax_h3.PackedLayout,
    out_video: []f32,
    out_audio: []f32,
    in: minimax_h3.Inputs,
    cancel: ?*std.atomic.Value(bool),
) !void {
    const cfg = dit.cfg;
    std.debug.assert(sess.seq == layout.seq_len);
    var step = try minimax_h3.DeviceStep.init(dit, io, gpa, layout, in);
    defer step.deinit(gpa);

    try be.beginBatch();
    errdefer if (be.batching()) be.abortBatch();
    if (ws.x_d.dt == .f32) {
        try be.tensorUpload(ws.x_d, std.mem.sliceAsBytes(step.packed_h));
    } else {
        const bytes = try minimax_h3_dev.narrow(gpa, step.packed_h, ws.x_d.dt);
        defer gpa.free(bytes);
        try be.tensorUpload(ws.x_d, bytes);
    }
    try be.tensorUpload(ws.mod_d, std.mem.sliceAsBytes(step.mod));
    try be.tensorUpload(ws.fmod_d, std.mem.sliceAsBytes(step.fmod));
    // A denoise mask relabels rows inside a target segment, so those two segments
    // pick their modulation row PER ROW: the kernels take a u32 index buffer and a
    // stride, where one launch per run of equal labels would be thousands of launches
    // for a spatial mask.
    var vmask: ?Buf = null;
    var amask: ?Buf = null;
    if (step.video_idx.len > 0) {
        try be.tensorUpload(ws.vmask_d, std.mem.sliceAsBytes(step.video_idx));
        vmask = ws.vmask_d;
    }
    if (step.audio_idx.len > 0) {
        try be.tensorUpload(ws.amask_d, std.mem.sliceAsBytes(step.audio_idx));
        amask = ws.amask_d;
    }
    try minimax_h3_dev.body(Dev{ .be = be, .sess = sess, .ws = ws, .cfg = cfg, .vmask = vmask, .amask = amask }, dit, layout, &step, cancel);
    try be.endBatch();
    // A 16-bit residual that overflowed can leave the heads finite.
    if (ws.x_d.dt != .f32 and !try be.allFinite(ws.x_d, layout.seq_len * cfg.hidden)) return error.NonFiniteActivation;

    const v_rows = try gpa.alloc(f32, layout.segmentOf(.video).?.len() * cfg.videoPatchDim());
    defer gpa.free(v_rows);
    const a_rows = try gpa.alloc(f32, layout.segmentOf(.audio).?.len() * minimax_h3.audio_latent_channels);
    defer gpa.free(a_rows);
    try be.tensorDownload(ws.q_d, std.mem.sliceAsBytes(v_rows));
    try be.tensorDownload(ws.k_d, std.mem.sliceAsBytes(a_rows));
    minimax_h3.DeviceStep.finish(layout, out_video, out_audio, v_rows, a_rows);
}

/// The `minimax_h3_dev` adapter.
const Dev = struct {
    be: *Backend,
    sess: *const Session,
    ws: *Workspace,
    cfg: minimax_h3.Config,
    vmask: ?Buf,
    amask: ?Buf,

    pub fn labelIdx(d: Dev, kind: minimax_h3.Kind) ?Buf {
        return switch (kind) {
            .video => d.vmask,
            .audio => d.amask,
            else => null,
        };
    }

    pub fn rmsMod(d: Dev, src: Buf, src_row: usize, dst: Buf, dst_row: usize, table: Buf, rows: usize, premul_off: usize, shift_off: usize, norm_eps: f32, idx: ?Buf, idx_row: usize, stride: usize) !void {
        try d.be.rmsModRowsAt(src, src_row, dst, dst_row, table, rows, d.cfg.hidden, premul_off, shift_off, norm_eps, idx, idx_row, stride);
    }

    pub fn gatedAdd(d: Dev, dst: Buf, dst_row: usize, src: Buf, src_row: usize, table: Buf, total: usize, gate_off: usize, idx: ?Buf, idx_row: usize, stride: usize, scale: f32) !void {
        if (dst.dt == src.dt and scale == 1) return d.be.gatedAddRowsAt(dst, dst_row, src, src_row, table, total, d.cfg.hidden, gate_off, idx, idx_row, stride);
        try d.be.gatedAddScaledRowsAt(dst, dst_row, src, src_row, table, total, d.cfg.hidden, gate_off, idx, idx_row, stride, scale);
    }

    pub fn prep(d: Dev, x: Buf, rows: usize, cols: usize, ws: []const Weight) !lin_cuda.Group {
        return lin_cuda.prep(d.be, d.ws.plan, x, rows, cols, ws);
    }

    pub fn gemm(d: Dev, g: lin_cuda.Group, y: Buf, w: Weight) !void {
        try lin_cuda.gemm(d.be, d.ws.plan, g, y, w, null);
    }

    pub fn linear(d: Dev, y: Buf, x: Buf, rows: usize, w: Weight) !void {
        try lin_cuda.linear(d.be, d.ws.plan, y, x, rows, w, null);
    }

    /// `opMatmul` writes exactly `rows * w.rows` (no 128-row rounding), so a band lands
    /// at its own offset with nothing spilling past it.
    pub fn linearAt(d: Dev, y: Buf, y_off: usize, x: Buf, rows: usize, w: Weight, bias: []const f32) !void {
        try lin_cuda.linear(d.be, d.ws.plan, y.viewBytes(y_off * 4), x, rows, w, bias);
    }

    pub fn qkNorm(d: Dev, x: Buf, w: []const f32, rows: usize) !void {
        try d.be.qkNorm(x, x, try normBuf(d.be, w), rows * d.cfg.n_heads, d.cfg.head_dim, d.cfg.qk_norm_eps);
    }

    pub fn rope(d: Dev, x: Buf, rows: usize, pos0: usize) !void {
        try d.be.opRopeHalfPart(x, d.sess.freqs_d, rows, d.cfg.n_heads, d.sess.pairs, d.sess.sinOff(), pos0, d.cfg.head_dim);
    }

    /// With bf16 storage a band lands in the f32 band scratch (`attn_d`, idle until
    /// pass two) and `kvStore` narrows it into place; the norm and the rope run on
    /// the f32 form either way.
    pub fn kvAt(d: Dev, which: enum { k, v }, k0: usize) !Buf {
        if (d.ws.kv_bf16) return d.ws.attn_d;
        const inner = d.cfg.n_heads * d.cfg.head_dim;
        return (if (which == .k) d.ws.k_d else d.ws.v_d).viewBytes(k0 * inner * 4);
    }

    pub fn kvStore(d: Dev, which: enum { k, v }, at: Buf, k0: usize, rows: usize) !void {
        if (!d.ws.kv_bf16) return;
        const inner = d.cfg.n_heads * d.cfg.head_dim;
        try d.be.cvtF32ToBf16(at, (if (which == .k) d.ws.k_d else d.ws.v_d).viewBytes(k0 * inner * 2), rows * inner);
    }

    pub fn kvReady(_: Dev, _: usize) !void {}

    /// MHA (kv_heads == n_heads), unmasked, so a band of queries over the whole key
    /// sequence is the same arithmetic as the square launch.
    pub fn attention(d: Dev, rows: usize, seq: usize) !void {
        const cfg = d.cfg;
        const ws = d.ws;
        const inner = cfg.n_heads * cfg.head_dim;
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));
        if (force_naive_attn) {
            try d.be.attn(ws.q_d, ws.k_d, ws.v_d, ws.attn_d, rows, seq, cfg.n_heads, cfg.n_heads, cfg.head_dim, scale, false);
        } else if (ws.kv_bf16) {
            try d.be.cvtF32ToBf16(ws.q_d, ws.qb_d, rows * inner);
            // K and V are bf16 here only; the heads reuse `k_d` as f32 scratch.
            try d.be.opAttnCross(ws.qb_d.as(.bf16), ws.k_d.as(.bf16), ws.v_d.as(.bf16), ws.ob_d.as(.bf16), rows, seq, cfg.n_heads, cfg.head_dim, scale);
            try d.be.cvtBf16ToF32(ws.ob_d, ws.attn_d, rows * inner);
        } else {
            try d.be.opAttnCross(ws.q_d, ws.k_d, ws.v_d, ws.attn_d, rows, seq, cfg.n_heads, cfg.head_dim, scale);
        }
    }

    pub fn siluMul(d: Dev, g: Buf, u: Buf, n: usize) !void {
        try d.be.siluMul(g, u, n);
    }

};

fn normBuf(be: *Backend, w: []const f32) !Buf {
    return .{ .buf = try be.smallBuffer(std.mem.sliceAsBytes(w)), .mem = .null_handle, .size = w.len * 4 };
}

// --- tests -----------------------------------------------------------------

test "a fused weight's row slice is a contiguous view with its own scales" {
    // This is what splits `qkv_proj` and `fc1` without a copy: rows of a
    // row-major weight are `cols` apart with no padding, and an int8 weight's
    // per-row scales slice with them. Pairing row i's bytes with row j's scale
    // is finite and wrong, so the two must move together; and `row_off` is
    // what tells the LoRA sidecar which rows of the fused factor are this view's.
    const a = std.testing.allocator;
    const bytes: [24]u8 = @splat(0);
    const scales = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var w = Weight.init(&bytes, .i8, 6, 4);
    w.row_scale = &scales;

    const mid = try lin_mod.rowSlice(a, w, 2, 3);
    try std.testing.expectEqual(@as(usize, 3), mid.rows);
    try std.testing.expectEqual(@as(usize, 4), mid.cols);
    try std.testing.expectEqual(@as(usize, 2), mid.row_off);
    try std.testing.expectEqual(@as(usize, 12), mid.bytes.len);
    try std.testing.expectEqual(bytes[8..20].ptr, mid.bytes.ptr);
    try std.testing.expectEqualSlices(f32, &.{ 3, 4, 5 }, mid.row_scale.?);

    // The three qkv slices tile it exactly.
    var covered: usize = 0;
    for ([_]usize{ 0, 2, 4 }) |from| covered += (try lin_mod.rowSlice(a, w, from, 2)).rows;
    try std.testing.expectEqual(w.rows, covered);
}

test "modulation offsets are distinct per block, timestep and tag" {
    // The device reads one modulation vector per (block, timestep row, tag,
    // slot); two of them colliding would silently modulate a segment with
    // another's parameters.
    const cfg: minimax_h3.Config = .{
        .hidden = 8,
        .n_layers = 2,
        .refiner_layers = 1,
        .n_heads = 1,
        .head_dim = 8,
        .ffn = 8,
        .text_dim = 8,
        .time_embed_dim = 8,
        .adaln_curve_grid = 4,
        .rope_inv_freq_len = 1,
    };
    const n_labels = 3;
    var seen = std.AutoHashMap(usize, void).init(std.testing.allocator);
    defer seen.deinit();
    for (0..2) |b| {
        for (0..n_labels) |t| {
            inline for (@typeInfo(minimax_h3.Tag).@"enum".field_values) |value| {
                for (0..6) |slot| {
                    const off = minimax_h3_dev.modOff(cfg, n_labels, b, t, @fromBackingInt(value), slot);
                    try std.testing.expect(!seen.contains(off));
                    try seen.put(off, {});
                }
            }
        }
    }
    // ...and they pack the buffer exactly, with no gaps to size around.
    try std.testing.expectEqual(@as(usize, 2 * n_labels * 3 * 6), seen.count());
    try std.testing.expectEqual(@as(usize, 0), minimax_h3_dev.modOff(cfg, n_labels, 0, 0, .video, 0));
    try std.testing.expectEqual(
        @as(usize, 2 * n_labels * 3 * 6 * cfg.hidden - cfg.hidden),
        minimax_h3_dev.modOff(cfg, n_labels, 1, n_labels - 1, .audio, 5),
    );
}
