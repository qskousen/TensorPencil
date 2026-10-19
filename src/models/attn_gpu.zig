//! Full non-causal attention on Vulkan's tensor cores: per batch of heads a scores
//! GEMM into an f16 plane, a two-pass softmax over the valid keys, and the P@V GEMM.
//! `core` runs that loop over operands a caller already staged (krea2 stages q/k
//! fused with its norm and rope, the SD UNet and H3's VAE pad 64-wide heads to 128);
//! `stage` is the plain f32 staging; `run` is both, or `attn_full` (one thread per
//! query and head) where there are no tensor cores. That fallback is no fit for a
//! render: above ~768 px it runs long enough to trip the GPU watchdog,
//! `error_device_lost`.
//!
//! Queries and keys may differ in count (H3 runs a query band against every key).
//! Staged operands: Q `[q_pad][heads * hd]` f16 with the softmax scale folded in, K
//! per-head k-major `[kv_heads][hd][kv_pad]` f16, V `[kv_pad][kv_heads * hd]` f16,
//! pad rows zero, so padded keys score 0 and padded probabilities add nothing. `out`
//! must hold `q_pad` rows: the P@V store writes whole 128-row tiles. `hd` must be
//! what `coopmat.buildGemmAttnOut` tiles, 128.

const std = @import("std");
const gpu = @import("tp_gpu").context;

const Buf = gpu.DeviceBuffer;

/// Two-pass softmax chunk count, 32 interleaved chunks per row so a warp covers a
/// row with coalesced reads. H3's trunk and VAE run 8; the partials buffer is sized
/// for whichever a caller passes.
pub const nchunks = 32;
/// Default cap on the scores planes; heads batch to fit it.
pub const s_bytes_cap: usize = 2 << 30;

pub fn tensorCores(ctx: *gpu.Context) bool {
    return ctx.pipe_scores != .null_handle;
}

/// `cap` shrunk under a `--vram-budget`, floored so one plane still fits.
pub fn scoresCap(cap: usize, budget: u64) usize {
    if (budget == 0) return cap;
    return @min(cap, @max(64 << 20, budget / 4));
}

/// Heads whose scores planes fit `bytes`, at least one.
pub fn headsFitting(heads: usize, q_pad: usize, kv_pad: usize, bytes: usize) usize {
    return @max(1, @min(heads, bytes / @max(q_pad * kv_pad * 2, 1)));
}

pub const Phase = enum { scores, softmax, out };

/// A caller's per-phase timer, for a profile that splits attention.
pub const Lap = struct {
    ctx: *const anyopaque,
    f: *const fn (*const anyopaque, Phase) void,
};

/// The scores scratch, sized for the longest sequence a caller runs.
pub const Bufs = struct { s: Buf, part: Buf, md: Buf };

/// What `core` runs: `q_rows` queries against `kv_rows` keys (both unpadded).
pub const Core = struct {
    q_rows: usize,
    kv_rows: usize,
    heads: usize,
    kv_heads: usize,
    hd: usize,
    /// Most heads per batch, 0 for as many as the scratch holds.
    max_hb: usize = 0,
    nchunks: usize = nchunks,
};

pub fn core(ctx: *gpu.Context, b: Bufs, qh: Buf, kh: Buf, v16: Buf, out: Buf, c: Core, lap: ?Lap) !void {
    const q_pad = std.mem.alignForward(usize, c.q_rows, 128);
    const kv_pad = std.mem.alignForward(usize, c.kv_rows, 128);
    const q_dim = c.heads * c.hd;
    const kv_dim = c.kv_heads * c.hd;
    const plane = q_pad * kv_pad * 2;
    var hb_cap = headsFitting(c.heads, q_pad, kv_pad, b.s.size);
    hb_cap = @max(1, @min(hb_cap, b.part.size / @max(c.q_rows * c.nchunks * 2 * 4, 1)));
    hb_cap = @max(1, @min(hb_cap, b.md.size / @max(q_pad * 2 * 4, 1)));
    if (c.max_hb != 0) hb_cap = @min(hb_cap, c.max_hb);
    std.debug.assert(hb_cap * plane <= b.s.size);

    const s_plane: u32 = @intCast(q_pad * kv_pad);
    var h0: usize = 0;
    while (h0 < c.heads) : (h0 += hb_cap) {
        const hb = @min(hb_cap, c.heads - h0);
        try ctx.opAttnScores(b.s, qh, kh, .{
            .u0 = @intCast(q_dim),
            .u1 = @intCast(kv_pad),
            .u2 = @intCast(h0),
            .u3 = @intCast(c.heads / c.kv_heads),
            .u4 = @intCast(c.hd * kv_pad),
            .u5 = s_plane,
        }, kv_pad / 128, q_pad / 128, hb);
        if (lap) |l| l.f(l.ctx, .scores);
        try ctx.opElt(.softmax_partial, b.s, null, null, b.part, .{
            .u0 = @intCast(hb * c.q_rows * c.nchunks),
            .u1 = @intCast(c.nchunks),
            .u2 = @intCast(c.kv_rows),
            .u3 = @intCast(kv_pad),
            .u4 = @intCast(c.q_rows),
            .u5 = s_plane,
        }, hb * c.q_rows * c.nchunks, 1, 1);
        try ctx.opElt(.softmax_combine, b.part, null, null, b.md, .{
            .u0 = @intCast(hb * c.q_rows),
            .u1 = @intCast(c.nchunks),
            .u2 = @intCast(c.q_rows),
            .u3 = @intCast(q_pad),
        }, hb * c.q_rows, 1, 1);
        if (lap) |l| l.f(l.ctx, .softmax);
        try ctx.opAttnOut(b.s, v16, out, b.md, .{
            .u0 = @intCast(kv_pad),
            .u1 = s_plane,
            .u2 = @intCast(h0),
            .u3 = @intCast(c.heads / c.kv_heads),
            .u4 = @intCast(kv_dim),
            .u5 = @intCast(q_dim),
            .f0 = @bitCast(@as(u32, @intCast(c.kv_rows))),
            .f1 = @bitCast(@as(u32, @intCast(q_pad))),
        }, q_pad / 128, hb);
        if (lap) |l| l.f(l.ctx, .out);
    }
}

/// f32 `[rows][heads * hd]` q/k/v into `core`'s f16 operands, the softmax scale
/// folded into Q.
pub fn stage(ctx: *gpu.Context, q: Buf, k: Buf, v: Buf, qh: Buf, kh: Buf, v16: Buf, rows: usize, heads: usize, kv_heads: usize, hd: usize, scale: f32) !void {
    return stageRect(ctx, q, k, v, qh, kh, v16, rows, rows, heads, kv_heads, hd, scale);
}

/// `stage` with `q_rows` queries and `kv_rows` keys.
pub fn stageRect(ctx: *gpu.Context, q: Buf, k: Buf, v: Buf, qh: Buf, kh: Buf, v16: Buf, q_rows: usize, kv_rows: usize, heads: usize, kv_heads: usize, hd: usize, scale: f32) !void {
    const q_pad = std.mem.alignForward(usize, q_rows, 128);
    const kv_pad = std.mem.alignForward(usize, kv_rows, 128);
    const q_dim = heads * hd;
    const kv_dim = kv_heads * hd;
    ctx.independent(3);
    try ctx.opElt(.f32_to_h16, q, null, null, qh, .{
        .u0 = @intCast(q_pad * q_dim / 2),
        .u1 = @intCast(q_rows * q_dim),
        .f0 = scale,
    }, q_pad * q_dim / 2, 1, 1);
    try ctx.opElt(.gather_kmajor_h16, k, null, null, kh, .{
        .u0 = @intCast(kv_dim * kv_pad / 2),
        .u1 = @intCast(hd),
        .u2 = @intCast(kv_pad),
        .u3 = @intCast(kv_rows),
        .u4 = @intCast(kv_heads),
    }, kv_dim * kv_pad / 2, 1, 1);
    try ctx.opElt(.f32_to_h16, v, null, null, v16, .{
        .u0 = @intCast(kv_pad * kv_dim / 2),
        .u1 = @intCast(kv_rows * kv_dim),
        .f0 = 1.0,
    }, kv_pad * kv_dim / 2, 1, 1);
}

/// The f16 operands and the scores scratch for a square attention of up to `rows_cap`
/// rows, heads batched into `cap` bytes of scores.
pub const Scratch = struct {
    qh: Buf,
    kh: Buf,
    v16: Buf,
    s: Buf,
    part: Buf,
    md: Buf,

    const fields = [_][]const u8{ "qh", "kh", "v16", "s", "part", "md" };

    pub fn init(ctx: *gpu.Context, rows_cap: usize, heads: usize, kv_heads: usize, hd: usize, cap: usize) !Scratch {
        return initRect(ctx, rows_cap, rows_cap, heads, kv_heads, hd, cap);
    }

    /// `init` for up to `q_cap` queries against up to `kv_cap` keys.
    pub fn initRect(ctx: *gpu.Context, q_cap: usize, kv_cap: usize, heads: usize, kv_heads: usize, hd: usize, cap: usize) !Scratch {
        const tc = tensorCores(ctx);
        const q_pad = std.mem.alignForward(usize, q_cap, 128);
        const kv_pad = std.mem.alignForward(usize, kv_cap, 128);
        const hb = headsFitting(heads, q_pad, kv_pad, scoresCap(cap, ctx.budget_override));
        const sizes = [fields.len]usize{
            if (tc) q_pad * heads * hd * 2 else 16,
            if (tc) kv_heads * hd * kv_pad * 2 else 16,
            if (tc) kv_pad * kv_heads * hd * 2 else 16,
            if (tc) hb * q_pad * kv_pad * 2 else 16,
            if (tc) hb * q_cap * nchunks * 2 * 4 else 16,
            if (tc) hb * q_pad * 2 * 4 else 16,
        };
        var self: Scratch = undefined;
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

    pub fn deinit(self: *Scratch, ctx: *gpu.Context) void {
        inline for (fields) |name| ctx.tensorDestroy(&@field(self, name));
        self.* = undefined;
    }

    pub fn bufs(self: *const Scratch) Bufs {
        return .{ .s = self.s, .part = self.part, .md = self.md };
    }
};

pub const Shape = struct { rows: usize, heads: usize, kv_heads: usize, hd: usize, scale: f32, max_hb: usize = 0 };

/// f32 q/k/v to `out`: `stage` then `core`, or `attn_full` without tensor cores or
/// when `force_full`.
pub fn run(ctx: *gpu.Context, sc: *const Scratch, q: Buf, k: Buf, v: Buf, out: Buf, sh: Shape, force_full: bool) !void {
    if (force_full or !tensorCores(ctx)) {
        return ctx.opElt(.attn_full, q, k, v, out, .{
            .u0 = @intCast(sh.rows),
            .u1 = @intCast(sh.heads),
            .u2 = @intCast(sh.kv_heads),
            .u3 = @intCast(sh.hd),
            .f0 = sh.scale,
        }, sh.rows * sh.heads, 1, 1);
    }
    try stage(ctx, q, k, v, sc.qh, sc.kh, sc.v16, sh.rows, sh.heads, sh.kv_heads, sh.hd, sh.scale);
    try core(ctx, sc.bufs(), sc.qh, sc.kh, sc.v16, out, .{
        .q_rows = sh.rows,
        .kv_rows = sh.rows,
        .heads = sh.heads,
        .kv_heads = sh.kv_heads,
        .hd = sh.hd,
        .max_hb = sh.max_hb,
    }, null);
}

// --- tests -----------------------------------------------------------------

const testing = std.testing;
const ops = @import("tp_ops");

// Needs the `testdata/gpu-tests` marker (see context.zig).
fn gpuCtx(gpa: std.mem.Allocator) !*gpu.Context {
    std.Io.Dir.cwd().access(testing.io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    return gpu.Context.init(gpa, testing.io) catch error.SkipZigTest;
}

fn relL2(want: []const f32, got: []const f32) f64 {
    var num: f64 = 0;
    var den: f64 = 0;
    for (want, got) |e, a| {
        num += @as(f64, e - a) * (e - a);
        den += @as(f64, e) * e;
    }
    return @sqrt(num / den);
}

test "gpu rectangular attention and the offset causal kernel match the CPU op" {
    const gpa = testing.allocator;
    const ctx = try gpuCtx(gpa);
    defer ctx.deinit();
    if (!tensorCores(ctx)) return error.SkipZigTest;

    // Qwen-Image 2.1's target pass: 256 queries after a prefix, 32 heads of 128. The
    // prefix lengths: none (square), a whole tile, and one that leaves the keys unaligned.
    for ([_]usize{ 0, 128, 20 }) |pre| try rectCase(gpa, ctx, pre);
}

fn rectCase(gpa: std.mem.Allocator, ctx: *gpu.Context, pre: usize) !void {
    const q_rows = 256;
    const kv_rows = pre + q_rows;
    const heads = 32;
    const hd = 128;
    const dim = heads * hd;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));

    var prng = std.Random.DefaultPrng.init(11);
    const rand = prng.random();
    const q = try gpa.alloc(f32, q_rows * dim);
    defer gpa.free(q);
    const k = try gpa.alloc(f32, kv_rows * dim);
    defer gpa.free(k);
    const v = try gpa.alloc(f32, kv_rows * dim);
    defer gpa.free(v);
    for (q) |*x| x.* = rand.floatNorm(f32);
    for (k) |*x| x.* = rand.floatNorm(f32);
    for (v) |*x| x.* = rand.floatNorm(f32);

    const Case = struct { name: []const u8, causal: bool };
    for ([_]Case{ .{ .name = "rect", .causal = false }, .{ .name = "causal at an offset", .causal = true } }) |c| {
        // Unmasked: every query sees every key. Causal: query i sits at key `pre + i`.
        const ends = try gpa.alloc(u32, q_rows);
        defer gpa.free(ends);
        for (ends, 0..) |*e, i| e.* = if (c.causal) @intCast(pre + i + 1) else @intCast(kv_rows);
        const want = try gpa.alloc(f32, q_rows * dim);
        defer gpa.free(want);
        try ops.attention.attention(testing.io, gpa, want, q, k, v, .{
            .seq_q = q_rows,
            .seq_kv = kv_rows,
            .n_heads = heads,
            .n_kv_heads = heads,
            .head_dim = hd,
            .kv_end = ends,
        });

        var bufs: [4]Buf = @splat(.{ .buf = .null_handle, .mem = .null_handle, .size = 0 });
        defer for (&bufs) |*b| ctx.tensorDestroy(b);
        bufs[0] = try ctx.tensorCreate(std.mem.alignForward(usize, q_rows, 128) * dim * 4);
        bufs[1] = try ctx.tensorCreate(kv_rows * dim * 4);
        bufs[2] = try ctx.tensorCreate(kv_rows * dim * 4);
        bufs[3] = try ctx.tensorCreate(std.mem.alignForward(usize, q_rows, 128) * dim * 4);
        try ctx.tensorUpload(bufs[0], std.mem.sliceAsBytes(q));
        try ctx.tensorUpload(bufs[1], std.mem.sliceAsBytes(k));
        try ctx.tensorUpload(bufs[2], std.mem.sliceAsBytes(v));

        const paths = [2][]const u8{ "f32 kernel", "tensor cores" };
        for (0..if (c.causal) 1 else 2) |pi| {
            if (pi == 0) {
                try ctx.opElt(.attn_causal_batched, bufs[0], bufs[1], bufs[2], bufs[3], .{
                    .u0 = q_rows,
                    .u1 = heads,
                    .u2 = heads,
                    .u3 = hd,
                    .u4 = q_rows,
                    .u5 = @intCast(pre),
                    .u6 = if (c.causal) 0 else @intCast(kv_rows),
                    .f0 = scale,
                }, q_rows * heads, 1, 1);
            } else {
                var sc = try Scratch.initRect(ctx, q_rows, kv_rows, heads, heads, hd, s_bytes_cap);
                defer sc.deinit(ctx);
                // Recorded as one batch, the way a forward runs it.
                try ctx.beginBatch();
                try stageRect(ctx, bufs[0], bufs[1], bufs[2], sc.qh, sc.kh, sc.v16, q_rows, kv_rows, heads, heads, hd, scale);
                try core(ctx, sc.bufs(), sc.qh, sc.kh, sc.v16, bufs[3], .{ .q_rows = q_rows, .kv_rows = kv_rows, .heads = heads, .kv_heads = heads, .hd = hd }, null);
                try ctx.endBatch();
            }
            const got = try gpa.alloc(f32, q_rows * dim);
            defer gpa.free(got);
            try ctx.tensorDownload(try ctx.view(bufs[3], 0, got.len * 4), std.mem.sliceAsBytes(got));
            const rel = relL2(want, got);
            errdefer std.debug.print("prefix {d}, {s}, {s}: rel L2 {e:.3}\n", .{ pre, c.name, paths[pi], rel });
            const bound: f64 = if (pi == 0) 1e-5 else 5e-3;
            try testing.expect(rel < bound);
        }
    }
}
