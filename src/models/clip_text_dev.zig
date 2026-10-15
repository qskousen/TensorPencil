//! The CLIP text tower (SD1.5's CLIP-L, SDXL's CLIP-G) on the device, written once for
//! every GPU backend over an adapter (`clip_text_cuda.Dev`, `clip_text_gpu.Dev`).
//! Same contract as `clip_text.TextEncoder.encodePrompt`, including `r.final_chunk0` and
//! the `r.empty_cache` slot, so a caller cannot tell which ran except by speed.
//!
//! The batch axis is the chunk: one 77-row window does not fill a GPU, so every chunk
//! runs in one pass. The attention is CAUSAL within each chunk; a non-causal CLIP still
//! renders every image, so nothing would show it.
//!
//! The adapter owns its buffers (`setup` / `teardown`, then `b`: `x`, `normed`, `q`, `k`,
//! `v`, `big`, `t`, `cap`) and supplies `layerNorm`, `gemms` (linears sharing one input,
//! each with its bias), `attention` over the chunks, `add`, `gelu`, `copy` and the
//! transfers.

const std = @import("std");
const tp_core = @import("tp_core");
const clip_text = @import("clip_text.zig");

const safetensors = tp_core.safetensors;
const clip_tok = tp_core.clip_tokenizer;
const clen = clip_tok.context_length;

pub fn encodePrompt(
    d: anytype,
    enc: *const clip_text.TextEncoder,
    gpa: std.mem.Allocator,
    out: []f32,
    p: *const clip_tok.Prompt,
    r: clip_text.TextEncoder.PromptRun,
) !void {
    const h = enc.cfg.hidden;
    std.debug.assert(out.len == p.seq() * h);
    const ids = try gpa.alloc(u32, p.chunks * clen);
    defer gpa.free(ids);
    for (0..p.chunks) |c| p.idsInto(ids[c * clen ..][0..clen], c);
    try forward(d, enc, gpa, ids, r.capture_layer, out, r.final_chunk0);

    if (!p.hasWeights()) return;
    // The empty reference is its own forward, as on the CPU: riding along with the
    // prompt changes the GEMMs' row count, and with it the prompt's own encoding,
    // which made a session's first render of a weighted prompt differ from the next.
    if (r.mode.needsEmpty() and r.empty_cache.* == null) {
        var empty_ids: [clen]u32 = undefined;
        clip_tok.emptyIds(&empty_ids, r.pad_id);
        const e = try gpa.alloc(f32, clen * h);
        errdefer gpa.free(e);
        try forward(d, enc, gpa, &empty_ids, r.capture_layer, e, null);
        r.empty_cache.* = e;
    }
    // The weighting formulas are the CPU's own, so the paths cannot disagree about
    // either dialect. A1111's forms need no empty-prompt reference.
    const empty = if (r.mode.needsEmpty()) r.empty_cache.*.? else &[_]f32{};
    for (0..p.chunks) |c| enc.applyMode(out[c * clen * h ..][0 .. clen * h], empty, p.chunk(c), r.mode);
}

/// The tower over `ids`, whole 77-token windows: the captured layer (or the final one)
/// into `out`, and window 0's post-final-LN rows into `final0`.
fn forward(d: anytype, enc: *const clip_text.TextEncoder, gpa: std.mem.Allocator, ids: []const u32, capture: ?usize, out: []f32, final0: ?[]f32) !void {
    const cfg = enc.cfg;
    const h = cfg.hidden;
    const total = ids.len;
    const items = total / clen;
    std.debug.assert(out.len == total * h);

    const x_host = try gpa.alloc(f32, total * h);
    defer gpa.free(x_host);
    const row_bytes = enc.tok_embed.info.dtype.storageBytes(h);
    for (ids, 0..) |id, t| {
        if (id >= enc.vocab) return error.TokenOutOfRange;
        const dst = x_host[t * h ..][0..h];
        try safetensors.convertToF32(enc.tok_embed.info.dtype, enc.tok_embed.bytes[id * row_bytes ..][0..row_bytes], dst);
        // The positional table restarts per window: each is its own sequence.
        for (dst, enc.pos_embed[(t % clen) * h ..][0..h]) |*xi, pe| xi.* += pe;
    }

    // The captured layer gets its own buffer rather than a mid-batch download, which
    // would force a second submit.
    try d.setup(total, capture != null);
    defer d.teardown();
    const b = d.b;
    try d.upload(b.x, x_host);
    try d.beginBatch();
    errdefer d.abortBatch();

    if (capture) |layer| if (layer == 0) try d.copy(b.cap, b.x, total * h);
    for (enc.layers, 0..) |*l, li| {
        // x += o(causal_attn(ln1(x)))
        try d.layerNorm(b.x, b.normed, l.ln1_w, l.ln1_b, total);
        try d.gemms(b.normed, total, &.{ b.q, b.k, b.v }, &.{ l.q, l.k, l.v }, &.{ l.q_b, l.k_b, l.v_b });
        try d.attention(items);
        try d.gemms(b.big, total, &.{b.t}, &.{l.o}, &.{l.o_b});
        try d.add(b.x, b.t, total * h);

        // x += fc2(act(fc1(ln2(x)))); CLIP-L is quick-GELU, CLIP-G erf-GELU, and the
        // forms agree to ~1e-2, close enough to look right and far enough to shift style.
        try d.layerNorm(b.x, b.normed, l.ln2_w, l.ln2_b, total);
        try d.gemms(b.normed, total, &.{b.big}, &.{l.fc1}, &.{l.fc1_b});
        try d.gelu(b.big, total * cfg.intermediate, cfg.act);
        try d.gemms(b.big, total, &.{b.t}, &.{l.fc2}, &.{l.fc2_b});
        try d.add(b.x, b.t, total * h);

        if (capture) |layer| if (layer == li + 1) try d.copy(b.cap, b.x, total * h);
    }
    // The final LayerNorm is what SD1.5 conditions on and where SDXL's pooled row comes
    // from; SDXL's context deliberately stops short of it.
    try d.layerNorm(b.x, b.x, enc.final_ln_w, enc.final_ln_b, total);
    try d.endBatch();

    try d.download(if (capture != null) b.cap else b.x, out);
    if (final0) |dst| {
        if (capture == null) @memcpy(dst, out[0 .. clen * h]) else try d.download(b.x, dst);
    }
}
