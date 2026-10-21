//! The SigLIP2 text and visual towers on the device, written once over the tower
//! adapters (`tower_cuda.Dev`, `tower_gpu.Dev`), for one item or a batch. Every item is
//! the same length (64 text tokens, 196 patches), packed back to back; the attention is
//! block-diagonal (one item takes the plain kernel). The heads (last-token pool and
//! projection, or the MAP head) run on the host, in their single-item form for one item
//! and their batched form otherwise, as the CPU model runs them.

const std = @import("std");
const ops = @import("tp_ops");
const qwen3 = @import("qwen3.zig");
const tower = @import("tower.zig");
const siglip = @import("embed_siglip.zig");
const Weight = ops.matmul.Weight;

/// One pre-LN block's weights: fused qkv (f32, sliced by rows), the out projection and
/// the MLP, under either tower's names.
const Block = struct {
    ln1_w: []const f32,
    ln1_b: []const f32,
    qkv: []const u8,
    qkv_b: []const f32,
    out: []const u8,
    out_b: []const f32,
    ln2_w: []const f32,
    ln2_b: []const f32,
    fc1: []const u8,
    fc1_b: []const f32,
    fc2: []const u8,
    fc2_b: []const f32,
};

const Geom = struct { w: usize, inter: usize, heads: usize, hd: usize, eps: f32 };

pub fn text(d: anytype, cpu: *const siglip.TextModel, io: std.Io, gpa: std.mem.Allocator, ids_list: []const []const u32, outs: [][]f32) !void {
    const cfg = cpu.cfg;
    const n = cfg.context_length;
    const w = cfg.width;
    const items = ids_list.len;
    std.debug.assert(outs.len == items and items > 0);
    const total = items * n;

    // Each item padded to the context window, plus the position table.
    const x_host = try gpa.alloc(f32, total * w);
    defer gpa.free(x_host);
    {
        const padded = try gpa.alloc(u32, total);
        defer gpa.free(padded);
        for (ids_list, 0..) |ids, i| {
            for (padded[i * n ..][0..n], 0..) |*pd, t| pd.* = if (t < ids.len) ids[t] else 0;
        }
        try qwen3.embedTokens(cpu.token_emb, padded, x_host);
        for (0..items) |i| {
            for (x_host[i * n * w ..][0 .. n * w], cpu.pos_emb) |*xi, pe| xi.* += pe;
        }
    }

    const g: Geom = .{ .w = w, .inter = 4 * w, .heads = cfg.n_heads, .hd = cfg.head_dim, .eps = cfg.ln_eps };
    defer d.done();
    var s = try Scratch(@TypeOf(d)).init(d, gpa, items, n, g, w);
    defer s.deinit(d);
    try d.upload(s.x, x_host);
    try d.beginBatch();
    errdefer d.abortBatch();
    for (cpu.layers) |*l| try block(d, &s, g, total, .{
        .ln1_w = l.ln1_w,
        .ln1_b = l.ln1_b,
        .qkv = l.in_proj.bytes,
        .qkv_b = l.in_proj_bias,
        .out = l.out_proj.bytes,
        .out_b = l.out_proj_bias,
        .ln2_w = l.ln2_w,
        .ln2_b = l.ln2_b,
        .fc1 = l.c_fc.bytes,
        .fc1_b = l.c_fc_bias,
        .fc2 = l.c_proj.bytes,
        .fc2_b = l.c_proj_bias,
    });
    try d.layerNorm(s.x, s.x, cpu.ln_final_w, cpu.ln_final_b, total, w, cfg.ln_eps);
    try d.endBatch();
    try d.download(s.x, x_host);

    // Last-token pool, then the projection.
    const pooled = try gpa.alloc(f32, items * w);
    defer gpa.free(pooled);
    for (0..items) |i| @memcpy(pooled[i * w ..][0..w], x_host[(i * n + n - 1) * w ..][0..w]);
    const projd = try gpa.alloc(f32, items * w);
    defer gpa.free(projd);
    try ops.matmul.matmul(io, gpa, projd, pooled, items, cpu.text_proj, cpu.text_proj_bias);
    for (outs, 0..) |out, i| {
        std.debug.assert(out.len == siglip.embed_dim);
        @memcpy(out, projd[i * w ..][0..w]);
        siglip.l2normalize(out);
    }
}

pub fn visual(d: anytype, cpu: *const siglip.VisualModel, io: std.Io, gpa: std.mem.Allocator, imgs: []const []const f32, outs: [][]f32) !void {
    const cfg = cpu.cfg;
    const w = cfg.width;
    const np = cfg.nPatches();
    const pin = cfg.patchIn();
    const items = imgs.len;
    std.debug.assert(outs.len == items and items > 0);
    const total = items * np;

    const patches = try gpa.alloc(f32, total * pin);
    defer gpa.free(patches);
    for (imgs, 0..) |img, i| {
        const p = try cpu.patchify(gpa, img);
        defer gpa.free(p);
        @memcpy(patches[i * np * pin ..][0 .. np * pin], p);
    }
    const pos = try gpa.alloc(f32, total * w);
    defer gpa.free(pos);
    for (0..items) |i| @memcpy(pos[i * np * w ..][0 .. np * w], cpu.pos_emb);

    const g: Geom = .{ .w = w, .inter = cfg.mlp_dim, .heads = cfg.n_heads, .hd = cfg.head_dim, .eps = cfg.ln_eps };
    defer d.done();
    var s = try Scratch(@TypeOf(d)).init(d, gpa, items, np, g, @max(w, pin));
    defer s.deinit(d);
    try d.beginBatch();
    errdefer d.abortBatch();
    try d.upload(s.t, patches);
    try d.linear(s.x, s.t, total, Weight.init(cpu.patch_w.bytes, .f32, w, pin), cpu.patch_b);
    try d.upload(s.t, pos);
    try d.add(s.x, s.t, total * w);
    for (cpu.blocks) |*b| try block(d, &s, g, total, .{
        .ln1_w = b.norm1_w,
        .ln1_b = b.norm1_b,
        .qkv = b.qkv.bytes,
        .qkv_b = b.qkv_b,
        .out = b.attn_proj.bytes,
        .out_b = b.attn_proj_b,
        .ln2_w = b.norm2_w,
        .ln2_b = b.norm2_b,
        .fc1 = b.fc1.bytes,
        .fc1_b = b.fc1_b,
        .fc2 = b.fc2.bytes,
        .fc2_b = b.fc2_b,
    });
    try d.layerNorm(s.x, s.x, cpu.norm_w, cpu.norm_b, total, w, cfg.ln_eps);
    try d.endBatch();

    const x_host = try gpa.alloc(f32, total * w);
    defer gpa.free(x_host);
    try d.download(s.x, x_host);
    if (items == 1) try cpu.mapHead(io, gpa, x_host, outs[0]) else try cpu.mapHeadBatch(io, gpa, x_host, outs);
}

fn Scratch(comptime D: type) type {
    return struct {
        const Self = @This();
        x: D.Buffer,
        normed: D.Buffer,
        q: D.Buffer,
        k: D.Buffer,
        v: D.Buffer,
        big: D.Buffer,
        t: D.Buffer,
        bounds: ?D.Buffer = null,

        /// `t_cols` is `t`'s width: a delta, or the patch rows on their way in.
        fn init(d: D, gpa: std.mem.Allocator, items: usize, len: usize, g: Geom, t_cols: usize) !Self {
            const total = items * len;
            var s: Self = undefined;
            const fields = .{ "x", "normed", "q", "k", "v", "big", "t" };
            const cols = [_]usize{ g.w, g.w, g.w, g.w, g.w, g.inter, t_cols };
            var made: usize = 0;
            errdefer inline for (fields, 0..) |f, i| if (i < made) d.release(&@field(s, f));
            inline for (fields, cols) |f, c| {
                @field(s, f) = try d.alloc(total * c);
                made += 1;
            }
            s.bounds = null;
            if (items > 1) {
                const lens = try gpa.alloc(usize, items);
                defer gpa.free(lens);
                @memset(lens, len);
                var r = try tower.Ragged.init(gpa, lens);
                defer r.deinit(gpa);
                const bd = try r.bounds(gpa);
                defer gpa.free(bd);
                s.bounds = try d.table(u32, bd);
                d.bindAttention(s.q, s.k, s.v, s.big, s.bounds.?);
            }
            return s;
        }

        fn deinit(s: *Self, d: D) void {
            inline for (.{ "x", "normed", "q", "k", "v", "big", "t" }) |f| d.release(&@field(s, f));
            if (s.bounds) |*b| d.release(b);
        }
    };
}

fn block(d: anytype, s: anytype, g: Geom, total: usize, b: Block) !void {
    const w = g.w;
    const seg = w * w * @sizeOf(f32);
    try d.layerNorm(s.x, s.normed, b.ln1_w, b.ln1_b, total, w, g.eps);
    try d.linear(s.q, s.normed, total, Weight.init(b.qkv[0..seg], .f32, w, w), b.qkv_b[0..w]);
    try d.linear(s.k, s.normed, total, Weight.init(b.qkv[seg .. 2 * seg], .f32, w, w), b.qkv_b[w .. 2 * w]);
    try d.linear(s.v, s.normed, total, Weight.init(b.qkv[2 * seg .. 3 * seg], .f32, w, w), b.qkv_b[2 * w .. 3 * w]);
    try d.attention(s.q, s.k, s.v, s.big, total, s.bounds, 0, g.heads, g.heads, g.hd, 1.0 / @sqrt(@as(f32, @floatFromInt(g.hd))));
    try d.linear(s.t, s.big, total, Weight.init(b.out, .f32, w, w), b.out_b);
    try d.add(s.x, s.t, total * w);

    try d.layerNorm(s.x, s.normed, b.ln2_w, b.ln2_b, total, w, g.eps);
    try d.linear(s.big, s.normed, total, Weight.init(b.fc1, .f32, g.inter, w), b.fc1_b);
    try d.gelu(s.big, total * g.inter);
    try d.linear(s.t, s.big, total, Weight.init(b.fc2, .f32, w, g.inter), b.fc2_b);
    try d.add(s.x, s.t, total * w);
}
