//! EmbeddingGemma 2's text body on the device, written once over the tower adapters
//! (`tower_cuda.Dev`, `tower_gpu.Dev`), for one item or a ragged batch of text or
//! image rows. Token embedding, image framing and the head (mean pool, 512->768, L2)
//! run on the host through `embed_gemma2.Model`.
//!
//! Attention is `attentionRows` over a bounds table at the attention scratch's tail:
//! words `[0, 2n)` hold each row's whole item, `[2n, 4n)` the same clipped to the local
//! layers' radius. The
//! per-layer inputs are one 512x512 slice of the PLE projection per layer, applied to
//! the kept input embeddings. Their `1/sqrt(hidden)` scale is folded into the norm's
//! eps (eps * hidden), which is the same normalization without a scaling pass.

const std = @import("std");
const ops = @import("tp_ops");
const tower = @import("tower.zig");
const eg2 = @import("embed_gemma2.zig");
const Weight = ops.matmul.Weight;

const hidden = eg2.hidden;
const ple_dim = eg2.ple_dim;
const eps = eg2.rms_eps;

// The weightless V norm, one per head width: the device norm-weight cache keys on
// the host pointer, so the two widths must not share one.
const ones_local: [eg2.local.head_dim]f32 = @splat(1.0);
const ones_global: [eg2.global.head_dim]f32 = @splat(1.0);

/// `ids_list[i]` (framed `<bos> ... <eos>`) -> `outs[i]` [embed_dim].
pub fn embed(d: anytype, cpu: *const eg2.Model, io: std.Io, gpa: std.mem.Allocator, ids_list: []const []const u32, outs: [][]f32) !void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    try embedRows(d, cpu, io, gpa, try cpu.tokenRows(arena.allocator(), ids_list), outs);
}

/// `embed_gemma2.Model.embedRows` with the body on the device.
pub fn embedRows(d: anytype, cpu: *const eg2.Model, io: std.Io, gpa: std.mem.Allocator, rows_list: []const []const f32, outs: [][]f32) !void {
    std.debug.assert(outs.len == rows_list.len and rows_list.len > 0);
    const lens = try gpa.alloc(usize, rows_list.len);
    defer gpa.free(lens);
    for (rows_list, lens) |r, *n| n.* = r.len / hidden;
    var rg = try tower.Ragged.init(gpa, lens);
    defer rg.deinit(gpa);
    const x = try gpa.alloc(f32, rg.total * hidden);
    defer gpa.free(x);
    for (rows_list, 0..) |r, i| @memcpy(x[rg.off[i] * hidden ..][0..r.len], r);
    try encodeRows(d, cpu, gpa, x, rg);
    for (outs, 0..) |out, i| try cpu.head(io, gpa, x[rg.off[i] * hidden .. rg.off[i + 1] * hidden], out);
}

/// `images[i]` -> `outs[i]`: `vitEncode(vit, img)` (a backend's resident tower) per
/// image, then one device forward over all their rows.
pub fn embedImages(d: anytype, cpu: *const eg2.Model, io: std.Io, gpa: std.mem.Allocator, images: []const eg2.Rgb, outs: [][]f32, vit_encode: anytype) !void {
    const vit = &(cpu.vit orelse return error.NoVisionTower);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const rows = try arena.allocator().alloc([]const f32, images.len);
    for (images, rows) |img, *r| {
        var enc = try vit_encode.run(vit, io, gpa, img);
        defer enc.deinit(gpa);
        r.* = try cpu.imageRows(arena.allocator(), enc.embeds);
    }
    try embedRows(d, cpu, io, gpa, rows, outs);
}

/// `encodeRows` of `embed_gemma2.Model` on the device: `x` holds the input
/// embeddings of the items `rg` lays out and is left holding the final-normed
/// hidden states.
pub fn encodeRows(d: anytype, cpu: *const eg2.Model, gpa: std.mem.Allocator, x: []f32, rg: tower.Ragged) !void {
    const rows = rg.total;
    std.debug.assert(x.len == rows * hidden);

    defer d.done();
    const Buf = @TypeOf(d).Buffer;
    var tables: [2]Buf = undefined;
    var n_tables: usize = 0;
    defer for (tables[0..n_tables]) |*b| d.release(b);
    for ([2]eg2.Geometry{ eg2.local, eg2.global }) |g| {
        const t = try tower.ropeTable(gpa, rg.longest, g.head_dim, g.theta);
        defer gpa.free(t);
        tables[n_tables] = try d.table(f32, t);
        n_tables += 1;
    }
    const nsplit = tower.attnSplit(rows, eg2.n_heads, rg.longest);
    const partials = rows * eg2.n_heads * nsplit * (eg2.global.head_dim + 2);
    var scratch = try d.alloc(partials + 4 * rows);
    defer d.release(&scratch);
    {
        const bd = try gpa.alloc(u32, 4 * rows);
        defer gpa.free(bd);
        rg.boundsInto(bd[0 .. 2 * rows], 0);
        rg.boundsInto(bd[2 * rows ..], eg2.window);
        try d.uploadWordsAt(scratch, partials, bd);
    }

    var bufs: [10]Buf = undefined;
    var made: usize = 0;
    defer for (bufs[0..made]) |*b| d.release(b);
    for ([_]usize{ hidden, hidden, hidden, eg2.max_q_dim, eg2.max_kv_dim, eg2.max_kv_dim, eg2.max_q_dim, eg2.intermediate, eg2.intermediate, hidden }) |cols| {
        bufs[made] = try d.alloc(rows * cols);
        made += 1;
    }
    const h, const x0, const normed, const q, const k, const v, const attn, const gate, const up, const t = bufs;

    try d.upload(h, x);
    try d.upload(x0, x);
    try d.beginBatch();
    errdefer d.abortBatch();
    const proj = cpu.ple_model_proj;
    const slice_bytes = proj.dtype.storageBytes(ple_dim * hidden);
    for (cpu.layers, 0..) |*l, li| {
        const g = eg2.geometry(li);
        const hd = g.head_dim;
        const half = hd / 2;
        const freqs = if (eg2.isGlobal(li)) tables[1] else tables[0];
        const qd = g.qDim();
        const kvd = g.kvDim();

        try d.rmsNorm(h, normed, l.input_norm, rows, hidden, eps);
        try d.linear(q, normed, rows, l.q, null);
        try d.linear(k, normed, rows, l.k, null);
        try d.linear(v, normed, rows, l.v, null);
        try d.rmsNorm(q, q, l.q_norm, rows * eg2.n_heads, hd, eps);
        try d.rmsNorm(k, k, l.k_norm, rows * g.n_kv, hd, eps);
        try d.rmsNorm(v, v, if (eg2.isGlobal(li)) &ones_global else &ones_local, rows * g.n_kv, hd, eps);
        for (0..rg.items()) |i| {
            try d.rope(q, freqs, rg.off[i], rg.len(i), eg2.n_heads, qd, half, rg.longest * half);
            try d.rope(k, freqs, rg.off[i], rg.len(i), g.n_kv, kvd, half, rg.longest * half);
        }
        try d.attentionRows(q, k, v, attn, scratch, partials + if (g.window == 0) 0 else 2 * rows, rows, eg2.n_heads, g.n_kv, hd, nsplit, 1.0);
        try d.linear(t, attn, rows, l.o, null);
        try d.rmsNorm(t, t, l.post_attn_norm, rows, hidden, eps);
        try d.add(h, t, rows * hidden);

        try d.rmsNorm(h, normed, l.pre_ffn_norm, rows, hidden, eps);
        try d.linear(gate, normed, rows, l.gate, null);
        try d.linear(up, normed, rows, l.up, null);
        try d.geluMul(gate, up, rows * eg2.intermediate);
        try d.linear(t, gate, rows, l.down, null);
        try d.rmsNorm(t, t, l.post_ffn_norm, rows, hidden, eps);
        try d.add(h, t, rows * hidden);

        // Evicting `ple_model_proj` drops these slices too: both caches evict by range.
        const w_ple = Weight.init(proj.bytes[li * slice_bytes ..][0..slice_bytes], proj.dtype, ple_dim, hidden);
        try d.linear(up, x0, rows, w_ple, null);
        try d.rmsNorm(up, up, cpu.ple_proj_norm, rows, ple_dim, eps * @as(f32, hidden));
        try d.linear(gate, h, rows, l.ple_gate, null);
        try d.geluMul(gate, up, rows * ple_dim);
        try d.linear(t, gate, rows, l.ple_proj, null);
        try d.rmsNorm(t, t, l.ple_norm, rows, hidden, eps);
        try d.add(h, t, rows * hidden);
        if (l.scalar != 1.0) try d.scale(h, rows * hidden, l.scalar);
    }
    try d.rmsNorm(h, h, cpu.final_norm, rows, hidden, eps);
    try d.endBatch();
    try d.download(h, x);
}

/// Device-vs-CPU check shared by the Vulkan and CUDA tests: every fixture case one
/// at a time, then all of them as one ragged batch.
pub fn testAgainstCpu(comptime Dev: type, handle: anytype) !void {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const cases = try eg2.loadReference(arena.allocator(), io);
    var cpu = try eg2.Model.open(gpa, io, eg2.test_dir, .{});
    defer cpu.deinit();
    const d = Dev.init(&cpu);

    const ids_list = try arena.allocator().alloc([]const u32, cases.len);
    const outs = try arena.allocator().alloc([]f32, cases.len);
    var c: [eg2.embed_dim]f32 = undefined;
    for (cases, ids_list, outs) |cs, *ids, *o| {
        ids.* = cs.ids;
        o.* = try arena.allocator().alloc(f32, eg2.embed_dim);
        try cpu.embed(io, gpa, cs.ids, &c);
        var g: [eg2.embed_dim]f32 = undefined;
        try d.embed(handle, io, gpa, cs.ids, &g);
        const cos = eg2.cosine(&c, &g);
        errdefer std.debug.print("{s}: device vs cpu cosine {d}\n", .{ cs.name, cos });
        try std.testing.expect(cos >= 0.9999);
    }
    try d.embedBatch(handle, io, gpa, ids_list, outs);
    for (cases, outs) |cs, o| {
        try cpu.embed(io, gpa, cs.ids, &c);
        const cos = eg2.cosine(&c, o);
        errdefer std.debug.print("{s}: batched device vs cpu cosine {d}\n", .{ cs.name, cos });
        try std.testing.expect(cos >= 0.9999);
    }
}

/// Device-vs-CPU check of the image path, the test PNGs as one batch.
pub fn testImagesAgainstCpu(comptime Dev: type, handle: anytype) !void {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var cpu = try eg2.Model.open(gpa, io, eg2.test_dir, .{ .vision = true });
    defer cpu.deinit();
    const d = Dev.init(&cpu);
    defer d.evict(handle);

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const names = [_][]const u8{ "down", "up", "tall" };
    var images: [names.len]eg2.Rgb = undefined;
    for (names, &images) |name, *img| {
        const png = try std.Io.Dir.cwd().readFileAlloc(io, try a.print("testdata/embeddinggemma2_{s}.png", .{name}), a, .limited(8 << 20));
        const dec = try @import("tp_core").image.decodePngRgb(a, png);
        img.* = .{ .pixels = dec.pixels, .width = dec.width, .height = dec.height };
    }
    var want: [names.len][eg2.embed_dim]f32 = undefined;
    var got: [names.len][eg2.embed_dim]f32 = undefined;
    var want_s: [names.len][]f32 = undefined;
    var got_s: [names.len][]f32 = undefined;
    for (0..names.len) |i| {
        want_s[i] = &want[i];
        got_s[i] = &got[i];
    }
    try cpu.embedImageBatch(io, gpa, &images, &want_s);
    try d.embedImageBatch(handle, io, gpa, &images, &got_s);
    for (names, want, got) |name, w, g| {
        const cos = eg2.cosine(&w, &g);
        errdefer std.debug.print("{s}: device vs cpu image cosine {d}\n", .{ name, cos });
        try std.testing.expect(cos >= 0.9999);
    }
}
