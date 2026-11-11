//! EmbeddingGemma on the device, written once over the tower adapters
//! (`tower_cuda.Dev`, `tower_gpu.Dev`), for one item or a ragged batch. The head
//! (mean-pool, two Dense, L2) runs on the host via `embed_gemma.Model.head`, per item.
//!
//! A bidirectional Gemma-3 body: RMSNorms whose weights already carry the +1 (folded at
//! load), per-head QK-norm, two rope tables (global every 6th layer, local otherwise),
//! GQA, the four-norm sandwich and a tanh GeGLU (inside the device parity floor). No
//! GEMM here has a bias.
//!
//! Attention is `attentionRows` over a bounds table at the attention scratch's tail:
//! words `[0, 2n)` hold each row's whole item, `[2n, 4n)` the same clipped to the local
//! layers' `window_radius`.

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
    for (ids_list) |ids| if (ids.len > eg.max_seq) return error.SequenceTooLong;

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
    const Buf = @TypeOf(d).Buffer;
    var tables: [2]Buf = undefined;
    var n_tables: usize = 0;
    defer for (tables[0..n_tables]) |*b| d.release(b);
    for ([2]f64{ cfg.rope_theta, cfg.rope_theta_local }) |theta| {
        const t = try tower.ropeTable(gpa, rg.longest, hd, theta);
        defer gpa.free(t);
        tables[n_tables] = try d.table(f32, t);
        n_tables += 1;
    }
    const nsplit = tower.attnSplit(total, heads, rg.longest);
    const partials = total * heads * nsplit * (hd + 2);
    var scratch = try d.alloc(partials + 4 * total);
    defer d.release(&scratch);
    {
        const bd = try gpa.alloc(u32, 4 * total);
        defer gpa.free(bd);
        rg.boundsInto(bd[0 .. 2 * total], 0);
        rg.boundsInto(bd[2 * total ..], eg.window_radius);
        try d.uploadWordsAt(scratch, partials, bd);
    }
    var bufs: [9]Buf = undefined;
    var made: usize = 0;
    defer for (bufs[0..made]) |*b| d.release(b);
    for ([_]usize{ w, w, qd, kvd, kvd, qd, inter, inter, w }) |cols| {
        bufs[made] = try d.alloc(total * cols);
        made += 1;
    }
    const x, const normed, const q, const k, const v, const attn, const gate, const up, const t = bufs;

    try d.upload(x, x_host);
    try d.beginBatch();
    errdefer d.abortBatch();
    const half = hd / 2;
    const sin_off = rg.longest * half;
    for (cpu.layers, 0..) |*l, li| {
        const global = cfg.isGlobal(li);
        const freqs = if (global) tables[0] else tables[1];
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
        const bounds_off = partials + if (global) 0 else 2 * total;
        try d.attentionRows(q, k, v, attn, scratch, bounds_off, total, heads, kvh, hd, nsplit, 1.0 / @sqrt(@as(f32, @floatFromInt(hd))));
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

/// Device check shared by the Vulkan and CUDA tests: each long fixture case alone,
/// then all of them in one ragged batch with a short item, against the reference.
pub fn testLongAgainstReference(comptime Dev: type, handle: anytype) !void {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = try eg.loadLongReference(a, io);
    var cpu = try eg.Model.open(gpa, io, eg.test_dir);
    defer cpu.deinit();
    const d = Dev.init(&cpu);

    var out: [eg.embed_dim]f32 = undefined;
    for (cases) |c| {
        try d.embed(handle, io, gpa, c.ids, &out);
        try eg.expectLong(c, &out, min_cos);
    }

    const short = [_]u32{ 2, 23391, 1902, 1 };
    var short_cpu: [eg.embed_dim]f32 = undefined;
    try cpu.embed(io, gpa, &short, &short_cpu);
    const ids_list = try a.alloc([]const u32, cases.len + 1);
    const outs = try a.alloc([]f32, cases.len + 1);
    ids_list[0] = &short;
    for (cases, ids_list[1..]) |c, *ids| ids.* = c.ids;
    for (outs) |*o| o.* = try a.alloc(f32, eg.embed_dim);
    try d.embedBatch(handle, io, gpa, ids_list, outs);
    for (cases, outs[1..]) |c, o| try eg.expectLong(c, o, min_cos);
    const cos_short = eg.cosine(&short_cpu, outs[0]);
    errdefer std.debug.print("short item batched with long ones: device vs cpu cosine {d}\n", .{cos_short});
    try std.testing.expect(cos_short >= min_cos);

    const over = try gpa.alloc(u32, eg.max_seq + 1);
    defer gpa.free(over);
    @memset(over, 2);
    try std.testing.expectError(error.SequenceTooLong, d.embed(handle, io, gpa, over, &out));
}

const min_cos = 0.999999;
