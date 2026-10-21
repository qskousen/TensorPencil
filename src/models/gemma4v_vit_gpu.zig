//! The Gemma 4 `gemma4v` vision tower on Vulkan: the adapter for `gemma4v_vit_dev.zig`.
//! q, k and v stay tight `[np][heads][72]` for the 2-D rope; attention pads them to
//! 128-wide heads for the tensor-core path (`attn_gpu`), exact since the pads are zero,
//! or runs `opAttnBounded` at the true width on a device without one; linears run in each weight's storage dtype through
//! `lin_gpu.towerLinear`. Nothing stays resident after an encode (`tower_gpu.Scope`).

const std = @import("std");
const lin_gpu = @import("lin_gpu.zig");
const tower = @import("tower.zig");
const attn_gpu = @import("attn_gpu.zig");
const tower_gpu = @import("tower_gpu.zig");
const gemma4v_vit = @import("gemma4v_vit.zig");
const gemma4v_vit_dev = @import("gemma4v_vit_dev.zig");
const gpu = @import("tp_gpu").context;
const Weight = @import("tp_ops").matmul.Weight;

const Vit = gemma4v_vit.Vit;
const Buf = gpu.DeviceBuffer;

/// The head width `attn_gpu`'s tensor-core attention tiles.
const tc_hd = 128;

/// Encode interleaved RGB pixels to Gemma 4 image-token embeddings on Vulkan. Same
/// contract as `Vit.encode`.
pub fn encode(vit: *const Vit, ctx: *gpu.Context, io: std.Io, gpa: std.mem.Allocator, rgb: []const u8, width: usize, height: usize) !Vit.Encoded {
    const scope = tower_gpu.Scope.begin(ctx);
    defer scope.end(ctx, vit);
    return encodeResident(vit, ctx, io, gpa, rgb, width, height);
}

/// `encode` leaving the weights cached on the device, for a caller that encodes many
/// images (an embedding index). The cache keys on `vit`'s host pointers, so `vit`
/// must outlive it or be evicted (`tower_gpu.evictModel`) before it is freed.
pub fn encodeResident(vit: *const Vit, ctx: *gpu.Context, io: std.Io, gpa: std.mem.Allocator, rgb: []const u8, width: usize, height: usize) !Vit.Encoded {
    var d: Dev = .{ .ctx = ctx, .cfg = vit.cfg, .tl = try lin_gpu.Lin.init(gpa, ctx, lin_gpu.tower_max_out, 0, 0) };
    defer d.deinit();
    return gemma4v_vit_dev.encode(&d, vit, io, gpa, rgb, width, height);
}

const none: Buf = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 };
const Bufs = struct { x: Buf = none, normed: Buf = none, proj: Buf = none, big: Buf = none, up: Buf = none, t: Buf = none };

/// The `gemma4v_vit_dev` adapter.
const Dev = struct {
    ctx: *gpu.Context,
    cfg: gemma4v_vit.Config,
    tl: lin_gpu.Lin,
    b: Bufs = .{},
    q: Buf = none,
    k: Buf = none,
    v: Buf = none,
    ones: Buf = none,
    freqs: Buf = none,
    pos2: Buf = none,
    sin_off: usize = 0,
    /// `opAttnBounded`'s partials, then a bounds table of every row seeing all keys.
    att: Buf = none,
    att_bounds: usize = 0,
    nsplit: usize = 1,
    /// The tensor-core path: q, k, v and the output at 128-wide heads, and its scratch.
    pad: [4]Buf = .{ none, none, none, none },
    tc: ?attn_gpu.Scratch = null,

    fn deinit(d: *Dev) void {
        inline for (.{ &d.b.x, &d.b.normed, &d.b.proj, &d.b.big, &d.b.up, &d.b.t, &d.q, &d.k, &d.v, &d.ones, &d.freqs, &d.pos2, &d.att }) |b| d.ctx.tensorDestroy(b);
        for (&d.pad) |*b| d.ctx.tensorDestroy(b);
        if (d.tc) |*s| s.deinit(d.ctx);
        d.tl.deinit();
    }

    pub fn setup(d: *Dev, gpa: std.mem.Allocator, gx: usize, gy: usize) !void {
        const cfg = d.cfg;
        const ctx = d.ctx;
        const np = gx * gy;
        const hd = cfg.headDim();
        var rope = try gemma4v_vit_dev.Rope.init(gpa, cfg, gx, gy);
        defer rope.deinit(gpa);
        const ones = try gpa.alloc(f32, hd);
        defer gpa.free(ones);
        @memset(ones, 1.0);
        d.ones = try ctx.tensorCreate(hd * 4);
        try ctx.tensorUpload(d.ones, std.mem.sliceAsBytes(ones));
        d.sin_off = rope.sin_off;
        d.freqs = try ctx.tensorCreate(rope.table.len * 4);
        try ctx.tensorUpload(d.freqs, std.mem.sliceAsBytes(rope.table));
        d.pos2 = try ctx.tensorCreate(rope.pos2.len * 4);
        try ctx.tensorUpload(d.pos2, std.mem.sliceAsBytes(rope.pos2));
        const kdim = 3 * cfg.patch * cfg.patch;
        const sizes = [_]usize{ np * cfg.dim, np * cfg.dim, np * @max(cfg.dim, kdim), np * @max(cfg.ffn, cfg.dim), np * cfg.ffn, np * cfg.dim, np * cfg.dim, np * cfg.dim, np * cfg.dim };
        inline for (.{ &d.b.x, &d.b.normed, &d.b.proj, &d.b.big, &d.b.up, &d.b.t, &d.q, &d.k, &d.v }, sizes) |b, n| b.* = try ctx.tensorCreate(n * 4);

        if (attn_gpu.tensorCores(ctx) and hd <= tc_hd) {
            const q_pad = std.mem.alignForward(usize, np, 128);
            for (&d.pad, [4]usize{ np, np, np, q_pad }) |*b, n| b.* = try ctx.tensorCreate(n * cfg.n_heads * tc_hd * 4);
            d.tc = try attn_gpu.Scratch.init(ctx, np, cfg.n_heads, cfg.n_heads, tc_hd, attn_gpu.s_bytes_cap);
            return;
        }
        d.nsplit = tower.attnSplit(np, cfg.n_heads, np);
        d.att_bounds = np * cfg.n_heads * d.nsplit * (hd + 2);
        d.att = try ctx.tensorCreate((d.att_bounds + 2 * np) * 4);
        const bd = try gpa.alloc(u32, 2 * np);
        defer gpa.free(bd);
        @memset(bd[0..np], 0);
        @memset(bd[np..], @intCast(np));
        try ctx.tensorUploadAt(d.att, d.att_bounds * 4, std.mem.sliceAsBytes(bd));
    }

    pub fn beginBatch(d: *Dev) !void {
        try d.ctx.beginBatch();
    }

    pub fn endBatch(d: *Dev) !void {
        try d.ctx.endBatch();
    }

    pub fn abortBatch(d: *Dev) void {
        if (d.ctx.batching) d.ctx.abortBatch();
    }

    pub fn upload(d: *Dev, b: Buf, data: []const f32) !void {
        try d.ctx.tensorUpload(b, std.mem.sliceAsBytes(data));
    }

    pub fn download(d: *Dev, b: Buf, dst: []f32) !void {
        try d.ctx.tensorDownloadAt(b, 0, std.mem.sliceAsBytes(dst));
    }

    pub fn linear(d: *Dev, y: Buf, x: Buf, m: usize, w: Weight) !void {
        try lin_gpu.towerLinear(&d.tl, y, x, m, w, null);
    }

    pub fn rms(d: *Dev, x: Buf, y: Buf, w: []const f32, rows: usize, dim: usize) !void {
        try d.rmsBuf(x, y, try small(d.ctx, w), rows, dim);
    }

    fn rmsBuf(d: *Dev, x: Buf, y: Buf, w: Buf, rows: usize, dim: usize) !void {
        try d.ctx.opElt(.rmsnorm, x, y, w, null, .{ .u0 = @intCast(rows), .u1 = @intCast(dim), .f0 = d.cfg.eps }, rows, 1, 1);
    }

    pub fn add(d: *Dev, x: Buf, y: Buf, n: usize) !void {
        try d.ctx.opElt(.add, x, y, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
    }

    pub fn geluQuickMul(d: *Dev, g: Buf, u: Buf, n: usize) !void {
        try d.ctx.opElt(.gelu_quick_mul, g, u, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
    }

    pub fn geluMul(d: *Dev, g: Buf, u: Buf, n: usize) !void {
        try d.ctx.opElt(.gelu_mul, g, u, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
    }

    /// q and k projected, per-head normed and roped; v projected and unit-normed; then
    /// `opAttnBounded` over every key at scale 1.0, its output `[np][dim]` in `big`.
    pub fn attention(d: *Dev, blk: *const gemma4v_vit.Block, np: usize) !Buf {
        const cfg = d.cfg;
        const ctx = d.ctx;
        const heads = cfg.n_heads;
        const hd = cfg.headDim();
        const half = hd / 4;
        const pairs = np * heads * 2 * half;
        for ([2]*const Weight{ &blk.q, &blk.k }, [2][]const f32{ blk.q_norm, blk.k_norm }, [2]Buf{ d.q, d.k }) |w, nw, dst| {
            try d.linear(dst, d.b.normed, np, w.*);
            try d.rms(dst, dst, nw, np * heads, hd);
            try ctx.opElt(.rope_vision_gemma4, dst, d.pos2, d.freqs, null, .{
                .u0 = @intCast(pairs),
                .u1 = @intCast(half),
                .u2 = @intCast(d.sin_off),
                .u3 = @intCast(heads),
                .u4 = @intCast(hd),
            }, pairs, 1, 1);
        }
        try d.linear(d.v, d.b.normed, np, blk.v);
        try d.rmsBuf(d.v, d.v, d.ones, np * heads, hd);
        if (d.tc) |*sc| {
            for ([3]Buf{ d.q, d.k, d.v }, d.pad[0..3]) |src, dst| try ctx.opElt(.head_pad, src, dst, null, null, .{
                .u0 = @intCast(np * heads * tc_hd),
                .u1 = tc_hd,
                .u2 = @intCast(hd),
                .u3 = @intCast(heads * hd),
                .u4 = 0,
                .u5 = @intCast(heads),
            }, np * heads * tc_hd, 1, 1);
            try attn_gpu.run(ctx, sc, d.pad[0], d.pad[1], d.pad[2], d.pad[3], .{ .rows = np, .heads = heads, .kv_heads = heads, .hd = tc_hd, .scale = 1.0 }, false);
            // `head_unpad` takes the head count in u4, where `head_pad` takes it in u5.
            try ctx.opElt(.head_unpad, d.pad[3], d.b.big, null, null, .{
                .u0 = @intCast(np * heads * hd),
                .u1 = @intCast(hd),
                .u2 = tc_hd,
                .u4 = @intCast(heads),
            }, np * heads * hd, 1, 1);
            return d.b.big;
        }
        try ctx.opAttnBounded(d.q, d.k, d.v, d.b.big, d.att, d.att_bounds, np, heads, heads, hd, d.nsplit, 1.0);
        return d.b.big;
    }
};

fn small(ctx: *gpu.Context, w: []const f32) !Buf {
    return .{ .buf = try ctx.smallBuffer(std.mem.sliceAsBytes(w)), .mem = .null_handle, .size = 0 };
}
