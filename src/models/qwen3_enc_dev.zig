//! The Qwen3 text encoder (krea2's, Z-Image's, Anima's and H3's conditioning) on the
//! device, written once for every GPU backend over an adapter (`qwen3_cuda.EncDev`,
//! `qwen3_gpu.EncDev`). Returns the variant's tap stack `[seq][enc.tapCount()][hidden]`,
//! the layout the CPU `encode` returns. Caller frees.
//!
//! The tap list and the RoPE theta come off `enc`, never from constants: krea2 keeps 12
//! hidden states at theta 5e6, Z-Image one at 1e6, and both run to completion with the
//! other's values.
//!
//! A vision payload splices pre-embedded rows over the placeholder tokens (on the host,
//! where the gather already runs), replaces the 1-D rope with mrope (only when there are
//! positions, as the reference emits none without an image; both tables are the same
//! `[seq][half]` cos-then-sin shape), and adds DeepStack features at the image rows
//! after the first few layers. ⚠️ Those are LLM layer indices, not the tower's: features
//! collected from deep ViT blocks go into the FIRST decoder layers.
//!
//! The adapter owns its buffers (`setup` / `teardown`, then `b`: `x`, `normed`, `q`,
//! `k`, `v`, `attn`, `gate`, `up`, `t`, `out`) and supplies `rmsNorm`, `gemms` (linears
//! sharing one input), `rope`, causal `attention`, `add`, `addRows` (a row span of `x`
//! += a row span of a DeepStack feature), `siluMul`, `copyTap`, `prefetch` and the
//! transfers.

const std = @import("std");
const ops = @import("tp_ops");
const qwen3 = @import("qwen3.zig");

pub fn encode(d: anytype, enc: *const qwen3.TextEncoder, gpa: std.mem.Allocator, ids: []const u32, vision: qwen3.TextEncoder.Vision, cancel: ?*std.atomic.Value(bool)) ![]f32 {
    const c = enc.cfg;
    const seq = ids.len;
    std.debug.assert(seq > 0);
    const hidden = c.hidden;
    const hd = qwen3.head_dim;
    const half = hd / 2;
    const eps = c.rms_eps;
    const tap_count = enc.tapCount();

    // Through `embedTokens`, which dispatches on the embedding's own dtype: a GGUF
    // encoder's embedding may be q6_k.
    const x = try gpa.alloc(f32, seq * hidden);
    defer gpa.free(x);
    try qwen3.embedTokens(enc.embed, ids, x);
    for (vision.blocks) |blk| {
        if (blk.rows.len % hidden != 0) return error.ShapeMismatch;
        const n = blk.rows.len / hidden;
        if (blk.at + n > seq) return error.ShapeMismatch;
        @memcpy(x[blk.at * hidden ..][0..blk.rows.len], blk.rows);
    }
    const inject_rows = vision.injectRows();
    if (vision.deepstack.len > 0) {
        if (inject_rows == 0) return error.ShapeMismatch;
        for (vision.deepstack) |f| if (f.len != inject_rows * hidden) return error.ShapeMismatch;
        for (vision.inject) |sp| if (sp.start + sp.len > seq) return error.ShapeMismatch;
    }

    var freqs = if (vision.positions) |pos|
        try ops.rope.mropeInterleavedFreqs(gpa, pos, seq, hd, vision.rope_dims, c.rope_theta)
    else
        try ops.rope.rotateHalfFreqs(gpa, seq, hd, c.rope_theta);
    defer freqs.deinit(gpa);
    const table = try gpa.alloc(f32, 2 * seq * half);
    defer gpa.free(table);
    @memcpy(table[0 .. seq * half], freqs.cos);
    @memcpy(table[seq * half ..], freqs.sin);

    try d.setup(seq, tap_count, table, vision.deepstack);
    defer d.teardown();
    const b = d.b;
    try d.upload(b.x, x);
    try d.beginBatch();
    errdefer d.abortBatch();

    if (enc.layers.len > 0) d.prefetch(enc.layers[0]);
    var tap_idx: usize = 0;
    // `n_layers + 1`: a tap may sit one PAST the last layer (Anima's does), and the
    // `l >= enc.layers.len` break is what ends the loop.
    for (0..c.n_layers + 1) |l| {
        if (cancel) |cc| if (cc.load(.acquire)) return error.Canceled;
        if (l + 1 < enc.layers.len) d.prefetch(enc.layers[l + 1]);
        if (tap_idx < tap_count and enc.taps[tap_idx] == l) {
            // The hidden state entering layer l; a tap past the last layer carries the
            // final norm, which lands in `normed` first since the copy applies the
            // per-tap offset.
            var src = b.x;
            if (enc.final_norm) |w| if (l == c.n_layers) {
                try d.rmsNorm(b.x, b.normed, w, seq, hidden, eps);
                src = b.normed;
            };
            try d.copyTap(src, tap_idx * seq * hidden, seq * hidden);
            tap_idx += 1;
        }
        if (l >= enc.layers.len) break;
        const layer = enc.layers[l];

        try d.rmsNorm(b.x, b.normed, layer.input_norm, seq, hidden, eps);
        try d.gemms(b.normed, seq, hidden, &.{ layer.q, layer.k, layer.v }, &.{ b.q, b.k, b.v });
        try d.rmsNorm(b.q, b.q, layer.q_norm, seq * c.n_heads, hd, eps);
        try d.rmsNorm(b.k, b.k, layer.k_norm, seq * c.n_kv_heads, hd, eps);
        try d.rope(b.q, seq, c.n_heads);
        try d.rope(b.k, seq, c.n_kv_heads);
        try d.attention(seq);
        try d.gemms(b.attn, seq, c.qDim(), &.{layer.o}, &.{b.t});
        try d.add(b.x, b.t, seq * hidden);

        try d.rmsNorm(b.x, b.normed, layer.post_norm, seq, hidden, eps);
        try d.gemms(b.normed, seq, hidden, &.{ layer.gate, layer.up }, &.{ b.gate, b.up });
        try d.siluMul(b.gate, b.up, seq * c.intermediate);
        try d.gemms(b.gate, seq, c.intermediate, &.{layer.down}, &.{b.t});
        try d.add(b.x, b.t, seq * hidden);

        // DeepStack AFTER this layer: each injection span is a contiguous row run in
        // both the sequence and the feature, so one row-span add per span.
        if (l < vision.deepstack.len) {
            var row: usize = 0;
            for (vision.inject) |sp| {
                try d.addRows(l, sp.start * hidden, row * hidden, sp.len * hidden);
                row += sp.len;
            }
            std.debug.assert(row == inject_rows);
        }
    }
    std.debug.assert(tap_idx == tap_count);
    try d.endBatch();

    // Tap-major on the device, token-major for the caller.
    const tap_major = try gpa.alloc(f32, tap_count * seq * hidden);
    defer gpa.free(tap_major);
    try d.download(b.out, tap_major);
    const out = try gpa.alloc(f32, seq * tap_count * hidden);
    for (0..tap_count) |tp| for (0..seq) |t| {
        @memcpy(out[(t * tap_count + tp) * hidden ..][0..hidden], tap_major[(tp * seq + t) * hidden ..][0..hidden]);
    };
    return out;
}
