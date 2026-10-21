//! Snowflake Arctic Embed (GTE) on the device, written once over the tower adapters
//! (`tower_cuda.Dev`, `tower_gpu.Dev`), for one item or a ragged batch: GEMMs,
//! LayerNorms and GeGLU run over every item's rows packed together, the RoPE per item,
//! and the attention block-diagonal (one item takes the plain kernel).
//!
//! GTE is post-LayerNorm: attention and MLP read the residual stream and the norm comes
//! AFTER the residual add. qkv and up_gate are packed and sliced into per-projection
//! views (f32, so row-boundary byte slices are exact). The FFN's GELU is the tanh form
//! where the CPU uses exact erf, ~1e-3 per element, inside the device parity floor. The
//! CLS pool and L2 normalize run on the host.

const std = @import("std");
const ops = @import("tp_ops");
const qwen3 = @import("qwen3.zig");
const tower = @import("tower.zig");
const es = @import("embed_snowflake.zig");
const Weight = ops.matmul.Weight;

pub fn embed(d: anytype, cpu: *const es.Model, gpa: std.mem.Allocator, ids_list: []const []const u32, outs: [][]f32) !void {
    const cfg = cpu.cfg;
    const w = cfg.hidden;
    const inter = cfg.intermediate;
    const hd = cfg.head_dim;
    const heads = cfg.n_heads;
    const seg = w * w * @sizeOf(f32);
    const useg = inter * w * @sizeOf(f32);
    std.debug.assert(outs.len == ids_list.len and ids_list.len > 0);

    var rg = try tower.Ragged.init(gpa, ids_list);
    defer rg.deinit(gpa);
    const total = rg.total;

    // Word embedding + token_type[0], then the embeddings LayerNorm, on the host.
    const x_host = try gpa.alloc(f32, total * w);
    defer gpa.free(x_host);
    for (ids_list, 0..) |ids, i| try qwen3.embedTokens(cpu.word_emb, ids, x_host[rg.off[i] * w ..][0 .. ids.len * w]);
    for (0..total) |t| {
        for (x_host[t * w ..][0..w], cpu.token_type0) |*xi, tt| xi.* += tt;
    }
    ops.norm.layerNorm(x_host, x_host, cpu.emb_ln_w, cpu.emb_ln_b, cfg.ln_eps);

    defer d.done();
    var freqs = blk: {
        const t = try tower.ropeTable(gpa, rg.longest, hd, cfg.rope_theta);
        defer gpa.free(t);
        break :blk try d.table(f32, t);
    };
    defer d.release(&freqs);
    var bufs: [8]@TypeOf(d).Buffer = undefined;
    var made: usize = 0;
    defer for (bufs[0..made]) |*b| d.release(b);
    for ([_]usize{ w, w, w, w, w, inter, inter, w }) |cols| {
        bufs[made] = try d.alloc(total * cols);
        made += 1;
    }
    const x, const q, const k, const v, const attn, const up, const gate, const t = bufs;
    var bounds: ?@TypeOf(d).Buffer = null;
    defer if (bounds) |*b| d.release(b);
    if (rg.items() > 1) {
        const bd = try rg.bounds(gpa);
        defer gpa.free(bd);
        bounds = try d.table(u32, bd);
        d.bindAttention(q, k, v, attn, bounds.?);
    }

    try d.upload(x, x_host);
    try d.beginBatch();
    errdefer d.abortBatch();
    const sin_off = rg.longest * hd / 2;
    for (cpu.layers) |*l| {
        const qkv = l.qkv.bytes;
        const ug = l.up_gate.bytes;
        try d.linear(q, x, total, Weight.init(qkv[0..seg], .f32, w, w), l.qkv_bias[0..w]);
        try d.linear(k, x, total, Weight.init(qkv[seg .. 2 * seg], .f32, w, w), l.qkv_bias[w .. 2 * w]);
        try d.linear(v, x, total, Weight.init(qkv[2 * seg .. 3 * seg], .f32, w, w), l.qkv_bias[2 * w .. 3 * w]);
        for (0..rg.items()) |i| {
            try d.rope(q, freqs, rg.off[i], rg.len(i), heads, w, hd / 2, sin_off);
            try d.rope(k, freqs, rg.off[i], rg.len(i), heads, w, hd / 2, sin_off);
        }
        try d.attention(q, k, v, attn, total, bounds, 0, heads, heads, hd, 1.0 / @sqrt(@as(f32, @floatFromInt(hd))));
        try d.linear(t, attn, total, Weight.init(l.o.bytes, .f32, w, w), l.o_bias);
        try d.add(x, t, total * w);
        try d.layerNorm(x, x, l.attn_ln_w, l.attn_ln_b, total, w, cfg.ln_eps);

        // GeGLU: up is the first split, gate the second.
        try d.linear(up, x, total, Weight.init(ug[0..useg], .f32, inter, w), null);
        try d.linear(gate, x, total, Weight.init(ug[useg .. 2 * useg], .f32, inter, w), null);
        try d.geluMul(gate, up, total * inter);
        try d.linear(t, gate, total, Weight.init(l.down.bytes, .f32, w, inter), l.down_bias);
        try d.add(x, t, total * w);
        try d.layerNorm(x, x, l.mlp_ln_w, l.mlp_ln_b, total, w, cfg.ln_eps);
    }
    try d.endBatch();

    try d.download(x, x_host);
    for (outs, 0..) |out, i| {
        std.debug.assert(out.len == es.embed_dim);
        @memcpy(out, x_host[rg.off[i] * w ..][0..w]);
        es.l2normalize(out);
    }
}
