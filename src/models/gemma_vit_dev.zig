//! The Gemma 3 vision tower (SigLIP: patch embed, learned position add, pre-LN blocks)
//! on the device, written once for every GPU backend over an adapter
//! (`gemma_vit_cuda.Dev`, `gemma_vit_gpu.Dev`). The projector (4x4 avg-pool,
//! soft_emb_norm, input projection) runs on the host via `Vit.project`: 256 tokens,
//! negligible next to the blocks over 4096 patches.
//!
//! The adapter holds `b` (`x`, `normed`, `big`, `t`, each `[np]` rows wide enough for
//! its use) and supplies `linear`, `layerNorm`, `add`, `gelu`, `attention` (q/k/v and
//! the attention itself, returning the buffer holding its output, since how a backend
//! fits the 72-wide heads to its kernel is its own) and the transfers.

const std = @import("std");
const gemma_vit = @import("gemma_vit.zig");
const Weight = @import("tp_ops").matmul.Weight;

const Vit = gemma_vit.Vit;

/// Interleaved RGB pixels to Gemma image-token embeddings; same contract as `Vit.encode`.
pub fn encode(d: anytype, vit: *const Vit, io: std.Io, gpa: std.mem.Allocator, rgb: []const u8, width: usize, height: usize) !Vit.Encoded {
    const cfg = vit.cfg;
    const dim = cfg.dim;
    const side = cfg.side();
    const np = side * side;
    const kdim = 3 * cfg.patch * cfg.patch;
    const b = d.b;

    const patches = try vit.patchMatrix(gpa, rgb, width, height);
    defer gpa.free(patches);

    try d.beginBatch();
    errdefer d.abortBatch();
    // pos_embd is `[np][dim]` in the same row-major patch order, added as is.
    try d.upload(b.t, patches[0 .. np * kdim]);
    try d.linear(b.x, b.t, np, Weight.fromF32(vit.patch_w, dim, kdim), vit.patch_b);
    try d.upload(b.t, vit.pos_embd);
    try d.add(b.x, b.t, np * dim);

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
    try d.endBatch();

    const x_host = try gpa.alloc(f32, np * dim);
    defer gpa.free(x_host);
    try d.download(b.x, x_host);
    return vit.project(io, gpa, x_host);
}
