//! The Gemma 4 `gemma4v` vision tower (31B: patch embed, learned 2-D positions, 27
//! SigLIP blocks with RMS sandwich norms, per-head QK norms, a weightless V norm, 2-D
//! neox RoPE, attention scale 1.0 and a GeGLU-quick FFN) on the device, written once
//! for every GPU backend over an adapter (`gemma4v_vit_cuda.Dev`,
//! `gemma4v_vit_gpu.Dev`). The projector (3x3 pool, std affine, RMS, input projection)
//! runs on the host via `Vit.project`.
//!
//! The adapter holds `b` (`x`, `normed`, `proj`, `big`, `up`, `t`; sized by its
//! `setup`, which also uploads the rope tables and per-token positions) and supplies
//! `linear`, `rms`, `add`, `geluQuickMul`, `attention` (q/k/v with their norms and rope
//! and the attention itself, returning the buffer holding the compact `[np][dim]`
//! output) and the transfers.

const std = @import("std");
const gemma4v_vit = @import("gemma4v_vit.zig");
const Weight = @import("tp_ops").matmul.Weight;

const Vit = gemma4v_vit.Vit;

/// Interleaved RGB pixels to Gemma 4 image-token embeddings; same contract as
/// `Vit.encode`.
pub fn encode(d: anytype, vit: *const Vit, io: std.Io, gpa: std.mem.Allocator, rgb: []const u8, width: usize, height: usize) !Vit.Encoded {
    const cfg = vit.cfg;
    const dim = cfg.dim;
    const kdim = 3 * cfg.patch * cfg.patch;
    var prep = try vit.prepare(gpa, rgb, width, height);
    defer prep.deinit(gpa);
    const gx = prep.gx;
    const gy = prep.gy;
    const np = gx * gy;
    const pos_rows = try vit.posEmbedRows(gpa, gx, gy);
    defer gpa.free(pos_rows);
    try d.setup(gpa, gx, gy);
    const b = d.b;

    try d.beginBatch();
    errdefer d.abortBatch();
    try d.upload(b.proj, prep.patches);
    try d.linear(b.x, b.proj, np, Weight.fromF32(vit.patch_w, dim, kdim));
    try d.upload(b.t, pos_rows);
    try d.add(b.x, b.t, np * dim);

    for (vit.blocks) |*blk| {
        try d.rms(b.x, b.normed, blk.ln1_w, np, dim);
        const a = try d.attention(blk, np);
        try d.linear(b.t, a, np, blk.out);
        try d.rms(b.t, b.t, blk.attn_post_norm_w, np, dim);
        try d.add(b.x, b.t, np * dim);

        try d.rms(b.x, b.normed, blk.ln2_w, np, dim);
        try d.linear(b.big, b.normed, np, blk.gate);
        try d.linear(b.up, b.normed, np, blk.up);
        try d.geluQuickMul(b.big, b.up, np * cfg.ffn);
        try d.linear(b.t, b.big, np, blk.down);
        try d.rms(b.t, b.t, blk.ffn_post_norm_w, np, dim);
        try d.add(b.x, b.t, np * dim);
    }
    try d.endBatch();

    const x_host = try gpa.alloc(f32, np * dim);
    defer gpa.free(x_host);
    try d.download(b.x, x_host);
    return vit.project(io, gpa, x_host, gx, gy);
}

/// The 2-D neox rope's inputs, shared by every adapter: the frequency table over half
/// the head (`[max(gx,gy)][hd/4]` cos, then sin) and each token's (x, y).
pub const Rope = struct {
    table: []f32,
    sin_off: usize,
    pos2: []u32,

    pub fn init(gpa: std.mem.Allocator, cfg: gemma4v_vit.Config, gx: usize, gy: usize) !Rope {
        const hd = cfg.headDim();
        const half = hd / 4;
        const max_pos = @max(gx, gy);
        var freqs = try @import("tp_ops").rope.rotateHalfFreqs(gpa, max_pos, hd / 2, cfg.rope_theta);
        defer freqs.deinit(gpa);
        const sin_off = max_pos * half;
        const table = try gpa.alloc(f32, 2 * sin_off);
        errdefer gpa.free(table);
        @memcpy(table[0..sin_off], freqs.cos);
        @memcpy(table[sin_off..], freqs.sin);
        const pos2 = try gpa.alloc(u32, gx * gy * 2);
        for (0..gx * gy) |t| {
            pos2[t * 2] = @intCast(t % gx);
            pos2[t * 2 + 1] = @intCast(t / gx);
        }
        return .{ .table = table, .sin_off = sin_off, .pos2 = pos2 };
    }

    pub fn deinit(r: *Rope, gpa: std.mem.Allocator) void {
        gpa.free(r.table);
        gpa.free(r.pos2);
    }
};
