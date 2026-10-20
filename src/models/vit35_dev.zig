//! The Qwen3.5/3.6 vision tower (patch embed, learned positions, pre-LN blocks with 2-D
//! vision RoPE, the 2x2 merger projector) on the device, written once for every GPU
//! backend over an adapter (`vit35_cuda.Dev`, `vit35_gpu.Dev`). The host keeps the
//! prep: resize, normalize, patch order, position interpolation, rope tables
//! (`Vit.prepare`).
//!
//! The adapter holds `b` (`x`, `normed`, `big`, `t`; sized by its `setup`, which also
//! uploads the rope tables) and supplies `linear`, `layerNorm`, `add`, `gelu`,
//! `attention` (qkv, the rope and the attention, returning the buffer holding the
//! compact `[np][dim]` output, since how a backend fits the 72-wide heads to its
//! kernel is its own) and the transfers.

const std = @import("std");
const vit35 = @import("vit35.zig");
const Weight = @import("tp_ops").matmul.Weight;

const Vit = vit35.Vit;

/// Interleaved RGB pixels to LLM embeddings; same contract as `Vit.encode`.
pub fn encode(d: anytype, vit: *const Vit, gpa: std.mem.Allocator, rgb: []const u8, width: usize, height: usize) !Vit.Encoded {
    const cfg = vit.cfg;
    const dim = cfg.dim;
    const kdim = 3 * cfg.patch * cfg.patch;

    var prep = try vit.prepare(gpa, rgb, width, height);
    defer prep.deinit(gpa);
    const np = prep.np();
    const nm = np / 4;
    const merged_dim = 4 * dim;
    try d.setup(gpa, &prep);
    const b = d.b;

    try d.beginBatch();
    errdefer d.abortBatch();
    // The positions are reordered on the host into the merged token order.
    try d.upload(b.big, prep.patches);
    try d.linear(b.x, b.big, np, Weight.fromF32(vit.patch_w, dim, kdim), vit.patch_b);
    {
        const pos_ordered = try gpa.alloc(f32, np * dim);
        defer gpa.free(pos_ordered);
        for (0..np) |t| {
            const src = prep.pos[(prep.py[t] * prep.pw + prep.px[t]) * dim ..][0..dim];
            @memcpy(pos_ordered[t * dim ..][0..dim], src);
        }
        try d.upload(b.t, pos_ordered);
        try d.add(b.x, b.t, np * dim);
    }

    for (vit.blocks) |*blk| {
        try d.layerNorm(b.x, b.normed, blk.ln1_w, blk.ln1_b, np);
        const a = try d.attention(blk, np);
        try d.linear(b.t, a, np, blk.out, blk.out_b);
        try d.add(b.x, b.t, np * dim);

        try d.layerNorm(b.x, b.normed, blk.ln2_w, blk.ln2_b, np);
        try d.linear(b.big, b.normed, np, blk.up, blk.up_b);
        try d.gelu(b.big, np * cfg.ffn);
        try d.linear(b.t, b.big, np, blk.down, blk.down_b);
        try d.add(b.x, b.t, np * dim);
    }
    try d.layerNorm(b.x, b.x, vit.post_ln_w, vit.post_ln_b, np);

    // The 2x2 merge is a reinterpret (tokens are block-ordered), then the two-layer
    // projector; nm * merged_dim == np * dim, so h0 fits `normed`.
    try d.linear(b.normed, b.x, nm, vit.mm0, vit.mm0_b);
    try d.gelu(b.normed, nm * merged_dim);
    try d.linear(b.t, b.normed, nm, vit.mm2, vit.mm2_b);
    try d.endBatch();

    const embeds = try gpa.alloc(f32, nm * cfg.proj_dim);
    errdefer gpa.free(embeds);
    try d.download(b.t, embeds);
    return .{ .embeds = embeds, .grid_w = prep.pw / 2, .grid_h = prep.ph / 2 };
}

/// The per-token rotate-half table the vision rope is, for a backend whose rope kernel
/// reads one row per position: row `t` holds `hd / 2` cos then as many sin, the first
/// `half` pairs from the token's grid row and the rest from its column, which is
/// `vit35.applyVisionRope` exactly. `[np * hd/2]` cos then `[np * hd/2]` sin.
pub fn tokenRopeTable(gpa: std.mem.Allocator, prep: *const Vit.Prepared, half: usize) ![]f32 {
    const np = prep.np();
    const n_dims = 2 * half;
    const out = try gpa.alloc(f32, 2 * np * n_dims);
    const cos = out[0 .. np * n_dims];
    const sin = out[np * n_dims ..];
    for (0..np) |t| {
        for (0..n_dims) |i| {
            const p: usize = if (i < half) prep.py[t] else prep.px[t];
            const j = if (i < half) i else i - half;
            cos[t * n_dims + i] = prep.rope_cos[p * half + j];
            sin[t * n_dims + i] = prep.rope_sin[p * half + j];
        }
    }
    return out;
}

test "the per-token rope table is applyVisionRope" {
    const gpa = std.testing.allocator;
    const half = 3;
    const hd = 4 * half;
    const heads = 2;
    const np = 5;
    const max_pos = 4;
    var cos: [max_pos * half]f32 = undefined;
    var sin: [max_pos * half]f32 = undefined;
    for (&cos, &sin, 0..) |*c, *s, i| {
        const th: f32 = @as(f32, @floatFromInt(i)) * 0.37 + 0.1;
        c.* = @cos(th);
        s.* = @sin(th);
    }
    var py = [np]u32{ 0, 0, 1, 3, 2 };
    var px = [np]u32{ 0, 1, 2, 1, 3 };
    var empty: [0]f32 = .{};
    const prep: Vit.Prepared = .{ .patches = &empty, .pos = &empty, .py = &py, .px = &px, .pw = np, .ph = 1, .rope_cos = &cos, .rope_sin = &sin };
    const tbl = try tokenRopeTable(gpa, &prep, half);
    defer gpa.free(tbl);

    var want: [np * heads * hd]f32 = undefined;
    for (&want, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.71);
    var got = want;
    vit35.applyVisionRope(&want, np, heads, hd, &py, &px, &cos, &sin, half);
    // rope_half over the table: row t, head h, pair i.
    const n_dims = hd / 2;
    for (0..np) |t| for (0..heads) |h| for (0..n_dims) |i| {
        const base = (t * heads + h) * hd;
        const c = tbl[t * n_dims + i];
        const s = tbl[np * n_dims + t * n_dims + i];
        const lo = got[base + i];
        const hi = got[base + n_dims + i];
        got[base + i] = lo * c - hi * s;
        got[base + n_dims + i] = hi * c + lo * s;
    };
    try std.testing.expectEqualSlices(f32, &want, &got);
}
