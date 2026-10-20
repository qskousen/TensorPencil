//! The Gemma 4 "unified" vision embedder (12B: no transformer blocks) on the device,
//! written once for every GPU backend over an adapter (`gemma4_vit_cuda.Dev`,
//! `gemma4_vit_gpu.Dev`): LayerNorm, patch-embed GEMM, LayerNorm, the positions,
//! LayerNorm, a weightless RMSNorm and the projection GEMM. The resize, normalize,
//! im2col and the positional-row gather stay on the host (`Vit.patchMatrix`,
//! `Vit.posEmbedRows`).
//!
//! The adapter holds `b` (`big`: the patches, then the projection out; `x`; `t`; sized
//! by its `setup`) and supplies `linear`, `layerNorm`, `rmsUnit`, `add` and the
//! transfers.

const std = @import("std");
const gemma4_vit = @import("gemma4_vit.zig");

const Vit = gemma4_vit.Vit;

/// Interleaved RGB pixels to Gemma 4 image-token embeddings; same contract as
/// `Vit.encode`.
pub fn encode(d: anytype, vit: *const Vit, gpa: std.mem.Allocator, rgb: []const u8, width: usize, height: usize) !Vit.Encoded {
    const cfg = vit.cfg;
    const dim = cfg.dim;
    const kdim = cfg.kdim();
    var pm = try vit.patchMatrix(gpa, rgb, width, height);
    defer pm.deinit(gpa);
    const np = pm.np();
    const pos = try vit.posEmbedRows(gpa, pm.n_cols, pm.n_rows);
    defer gpa.free(pos);
    try d.setup(np);
    const b = d.b;

    try d.beginBatch();
    errdefer d.abortBatch();
    try d.upload(b.big, pm.data);
    try d.layerNorm(b.big, b.big, vit.patch_norm_1_w, vit.patch_norm_1_b, np, kdim, cfg.eps_ln);
    try d.linear(b.x, b.big, np, vit.patch_w, vit.patch_b);
    try d.layerNorm(b.x, b.x, vit.patch_norm_2_w, vit.patch_norm_2_b, np, dim, cfg.eps_ln);
    try d.upload(b.t, pos);
    try d.add(b.x, b.t, np * dim);
    try d.layerNorm(b.x, b.x, vit.patch_norm_3_w, vit.patch_norm_3_b, np, dim, cfg.eps_ln);
    try d.rmsUnit(b.x, np, dim, cfg.eps_rms);
    try d.linear(b.big, b.x, np, vit.mm_proj, null);
    try d.endBatch();

    const embeds = try gpa.alloc(f32, np * cfg.proj_dim);
    errdefer gpa.free(embeds);
    try d.download(b.big, embeds);
    return .{ .embeds = embeds, .grid_w = pm.n_cols, .grid_h = pm.n_rows };
}
