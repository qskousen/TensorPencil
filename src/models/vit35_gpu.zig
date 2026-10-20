//! The Qwen3.5/3.6 vision tower on Vulkan: the adapter for `vit35_dev.zig`. q, k and v
//! are three row slices of the fused weight, so each lands in a tight `[np][heads][72]`
//! buffer; the vision rope is the shared `rope_half` over a per-token table
//! (`vit35_dev.tokenRopeTable`), and attention is the non-causal `attn_full` at the true
//! head width.
//!
//! Nothing stays resident after an encode (`tower_gpu.Scope`): no VRAM held under the
//! LLM, and no cache entry outlives the `Vit` whose pointers key it.

const std = @import("std");
const lin = @import("lin.zig");
const lin_gpu = @import("lin_gpu.zig");
const tower_gpu = @import("tower_gpu.zig");
const vit35 = @import("vit35.zig");
const vit35_dev = @import("vit35_dev.zig");
const gpu = @import("tp_gpu").context;
const Weight = @import("tp_ops").matmul.Weight;

const Vit = vit35.Vit;
const Buf = gpu.DeviceBuffer;

/// Encode interleaved RGB pixels to LLM embeddings on Vulkan. Same contract as
/// `Vit.encode`.
pub fn encode(vit: *const Vit, ctx: *gpu.Context, gpa: std.mem.Allocator, rgb: []const u8, width: usize, height: usize) !Vit.Encoded {
    const scope = tower_gpu.Scope.begin(ctx);
    var d: Dev = .{ .ctx = ctx, .cfg = vit.cfg, .tl = try lin_gpu.Lin.init(gpa, ctx, lin_gpu.tower_max_out, 0, 0) };
    defer {
        d.deinit();
        scope.end(ctx, vit);
    }
    return vit35_dev.encode(&d, vit, gpa, rgb, width, height);
}

const none: Buf = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 };
const Bufs = struct { x: Buf = none, normed: Buf = none, big: Buf = none, t: Buf = none };

/// The `vit35_dev` adapter.
const Dev = struct {
    ctx: *gpu.Context,
    cfg: vit35.Config,
    tl: lin_gpu.Lin,
    b: Bufs = .{},
    q: Buf = none,
    k: Buf = none,
    v: Buf = none,
    /// `vit35_dev.tokenRopeTable`: cos rows, then sin rows from `np * hd/2`.
    rope: Buf = none,

    fn deinit(d: *Dev) void {
        inline for (.{ &d.b.x, &d.b.normed, &d.b.big, &d.b.t, &d.q, &d.k, &d.v, &d.rope }) |b| d.ctx.tensorDestroy(b);
        d.tl.deinit();
    }

    pub fn setup(d: *Dev, gpa: std.mem.Allocator, prep: *const Vit.Prepared) !void {
        const cfg = d.cfg;
        const dim = cfg.dim;
        const np = prep.np();
        const nm = np / 4;
        const sizes = [_]usize{
            np * dim,
            np * dim,
            np * @max(dim, cfg.ffn, 3 * cfg.patch * cfg.patch),
            @max(np * dim, nm * cfg.proj_dim),
            np * dim,
            np * dim,
            np * dim,
        };
        inline for (.{ &d.b.x, &d.b.normed, &d.b.big, &d.b.t, &d.q, &d.k, &d.v }, sizes) |b, n| b.* = try d.ctx.tensorCreate(n * 4);
        const tbl = try vit35_dev.tokenRopeTable(gpa, prep, cfg.headDim() / 4);
        defer gpa.free(tbl);
        d.rope = try d.ctx.tensorCreate(tbl.len * 4);
        try d.ctx.tensorUpload(d.rope, std.mem.sliceAsBytes(tbl));
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

    pub fn linear(d: *Dev, y: Buf, x: Buf, m: usize, w: Weight, bias: []const f32) !void {
        try lin_gpu.towerLinear(&d.tl, y, x, m, w, bias);
    }

    pub fn layerNorm(d: *Dev, x: Buf, y: Buf, w: []const f32, b: []const f32, rows: usize) !void {
        try d.ctx.opElt(.layernorm, x, y, try small(d.ctx, w), try small(d.ctx, b), .{ .u0 = @intCast(rows), .u1 = @intCast(d.cfg.dim), .f0 = d.cfg.eps }, rows, 1, 1);
    }

    pub fn add(d: *Dev, x: Buf, y: Buf, n: usize) !void {
        try d.ctx.opElt(.add, x, y, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
    }

    pub fn gelu(d: *Dev, x: Buf, n: usize) !void {
        try d.ctx.opElt(.gelu, x, null, null, null, .{ .u0 = @intCast(n) }, n, 1, 1);
    }

    /// q, k and v from their row slices of the fused weight, both ropes, then
    /// `attn_full`; the output is `[np][dim]` in `big`.
    pub fn attention(d: *Dev, blk: *const vit35.Block, np: usize) !Buf {
        const cfg = d.cfg;
        const dim = cfg.dim;
        const heads = cfg.n_heads;
        const hd = cfg.headDim();
        // bf16/f16 mmproj weights carry no row sidecars, so the slices allocate nothing.
        var fba_buf: [256]u8 = undefined;
        var fba = std.heap.FixedBufferAllocator.init(&fba_buf);
        inline for (.{ d.q, d.k, d.v }, 0..) |dst, i| {
            const w = try lin.rowSlice(fba.allocator(), blk.qkv, i * dim, dim);
            try d.linear(dst, d.b.normed, np, w, blk.qkv_b[i * dim ..][0..dim]);
        }
        const n_dims = hd / 2;
        const pairs = np * heads * n_dims;
        inline for (.{ d.q, d.k }) |x| try d.ctx.opElt(.rope_half, x, null, d.rope, null, .{
            .u0 = @intCast(pairs),
            .u1 = @intCast(n_dims),
            .u2 = @intCast(np * n_dims),
            .u3 = @intCast(heads),
        }, pairs, 1, 1);
        try d.ctx.opElt(.attn_full, d.q, d.k, d.v, d.b.big, .{
            .u0 = @intCast(np),
            .u1 = @intCast(heads),
            .u2 = @intCast(heads),
            .u3 = @intCast(hd),
            .f0 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd))),
        }, np * heads, 1, 1);
        return d.b.big;
    }
};

fn small(ctx: *gpu.Context, w: []const f32) !Buf {
    return .{ .buf = try ctx.smallBuffer(std.mem.sliceAsBytes(w)), .mem = .null_handle, .size = 0 };
}
