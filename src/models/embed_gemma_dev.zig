//! EmbeddingGemma on the device, written once over the tower adapters
//! (`tower_cuda.Dev`, `tower_gpu.Dev`), for one item or a ragged batch. The head
//! (mean-pool, two Dense, L2) runs on the host via `embed_gemma.Model.head`, per item.
//!
//! A bidirectional Gemma-3 body: RMSNorms whose weights already carry the +1 (folded at
//! load), per-head QK-norm, two rope tables (global every 6th layer, local otherwise),
//! GQA, the four-norm sandwich and a tanh GeGLU (inside the device parity floor). The
//! sliding window runs as full attention, which is exact for the `seq <= window` the
//! model asserts. No GEMM here has a bias.

const std = @import("std");
const ops = @import("tp_ops");
const qwen3 = @import("qwen3.zig");
const tower = @import("tower.zig");
const eg = @import("embed_gemma.zig");
const Weight = ops.matmul.Weight;

pub fn embed(d: anytype, cpu: *const eg.Model, io: std.Io, gpa: std.mem.Allocator, ids_list: []const []const u32, outs: [][]f32) !void {
    const cfg = cpu.cfg;
    const w = cfg.hidden;
    const qd = cfg.qDim();
    const kvd = cfg.kvDim();
    const hd = cfg.head_dim;
    const heads = cfg.n_heads;
    const kvh = cfg.n_kv_heads;
    const inter = cfg.intermediate;
    const eps = cfg.rms_eps;
    std.debug.assert(outs.len == ids_list.len and ids_list.len > 0);
    for (ids_list) |ids| std.debug.assert(ids.len <= cfg.sliding_window);

    var rg = try tower.Ragged.init(gpa, ids_list);
    defer rg.deinit(gpa);
    const total = rg.total;

    // Token embedding times sqrt(hidden), on the host.
    const x_host = try gpa.alloc(f32, total * w);
    defer gpa.free(x_host);
    for (ids_list, 0..) |ids, i| try qwen3.embedTokens(cpu.embed_w, ids, x_host[rg.off[i] * w ..][0 .. ids.len * w]);
    const scale = cfg.embedScale();
    for (x_host) |*v| v.* *= scale;

    defer d.done();
    var tables: [2]@TypeOf(d).Buffer = undefined;
    var n_tables: usize = 0;
    defer for (tables[0..n_tables]) |*b| d.release(b);
    for ([2]f64{ cfg.rope_theta, cfg.rope_theta_local }) |theta| {
        const t = try tower.ropeTable(gpa, rg.longest, hd, theta);
        defer gpa.free(t);
        tables[n_tables] = try d.table(f32, t);
        n_tables += 1;
    }
    var bufs: [9]@TypeOf(d).Buffer = undefined;
    var made: usize = 0;
    defer for (bufs[0..made]) |*b| d.release(b);
    for ([_]usize{ w, w, qd, kvd, kvd, qd, inter, inter, w }) |cols| {
        bufs[made] = try d.alloc(total * cols);
        made += 1;
    }
    const x, const normed, const q, const k, const v, const attn, const gate, const up, const t = bufs;
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
    const half = hd / 2;
    const sin_off = rg.longest * half;
    for (cpu.layers, 0..) |*l, li| {
        const freqs = if (cfg.isGlobal(li)) tables[0] else tables[1];
        try d.rmsNorm(x, normed, l.input_norm, total, w, eps);
        try d.linear(q, normed, total, Weight.init(l.q.bytes, .f32, qd, w), null);
        try d.linear(k, normed, total, Weight.init(l.k.bytes, .f32, kvd, w), null);
        try d.linear(v, normed, total, Weight.init(l.v.bytes, .f32, kvd, w), null);
        try d.rmsNorm(q, q, l.q_norm, total * heads, hd, eps);
        try d.rmsNorm(k, k, l.k_norm, total * kvh, hd, eps);
        for (0..rg.items()) |i| {
            try d.rope(q, freqs, rg.off[i], rg.len(i), heads, qd, half, sin_off);
            try d.rope(k, freqs, rg.off[i], rg.len(i), kvh, kvd, half, sin_off);
        }
        try d.attention(q, k, v, attn, total, bounds, heads, kvh, hd);
        try d.linear(t, attn, total, Weight.init(l.o.bytes, .f32, w, qd), null);
        // The sandwich: the post-attention norm runs BEFORE the residual add.
        try d.rmsNorm(t, t, l.post_attn_norm, total, w, eps);
        try d.add(x, t, total * w);

        try d.rmsNorm(x, normed, l.pre_ffn_norm, total, w, eps);
        try d.linear(gate, normed, total, Weight.init(l.gate.bytes, .f32, inter, w), null);
        try d.linear(up, normed, total, Weight.init(l.up.bytes, .f32, inter, w), null);
        try d.geluMul(gate, up, total * inter);
        try d.linear(t, gate, total, Weight.init(l.down.bytes, .f32, w, inter), null);
        try d.rmsNorm(t, t, l.post_ffn_norm, total, w, eps);
        try d.add(x, t, total * w);
    }
    try d.rmsNorm(x, x, cpu.final_norm, total, w, eps);
    try d.endBatch();

    try d.download(x, x_host);
    for (outs, 0..) |out, i| {
        std.debug.assert(out.len == eg.embed_dim);
        try cpu.head(io, gpa, x_host[rg.off[i] * w ..][0 .. rg.len(i) * w], out);
    }
}
