//! Device-memory attribution tags. Each GPU backend context can optionally tag
//! every device allocation with the pipeline component it belongs to, so the
//! GUI VRAM meter can show a MEASURED per-component breakdown (not an estimate).
//! Tagging is opt-in per context (`track_tags`); both the diffusion pipeline and
//! an LLM session turn it on, and the cost is one add per allocation.
//!
//! The counters are plain `u64`s read from the UI thread while the diffusion
//! worker updates them, a benign race (a momentarily stale number), exactly
//! like `device_used`. The per-pointer map they're derived from is only ever
//! touched on the allocating thread.
//!
//! The tags must PARTITION `device_used`: every allocation path has to attribute
//! its bytes, or the missing ones reappear as a residual under some other name
//! and the breakdown quietly stops adding up. The VMM path (`vmmCommit`) is the
//! one that is easy to forget, since it bypasses `alloc` entirely.

/// Which component an allocation belongs to. `other` is the default (anything
/// allocated outside a tagged phase, init overhead, pools); the diffusion meter
/// folds it into the "latent / working" segment.
pub const MemTag = enum(u8) {
    other = 0,
    te = 1, // text encoder (weights + encode scratch)
    dit = 2, // diffusion transformer (weights + denoise scratch)
    vae = 3, // VAE decoder (weights + decode scratch)
    latent = 4, // per-image working set (latent workspace, GPU session, live-preview decode)

    // LLM. Split finer than the diffusion tags because the thing these exist to
    // answer is "what is the non-weight, non-KV footprint made of", which was a
    // single derived residual before them.
    lm_weights = 5, // the pinned weight cache: parameters on the device
    lm_kv = 6, // K/V caches and recurrent state (the growable VMM buffers)
    lm_act = 7, // per-chunk activation buffers, sized by the prefill chunk
    lm_logits = 8, // the vocab logits buffer, sized by vocab x rows
    lm_dequant = 9, // f16 staging a quantized GEMM expands into

    pub const count = 10;

    /// Short label for logs and the debug dump.
    pub fn name(self: MemTag) []const u8 {
        return switch (self) {
            .other => "other",
            .te => "te",
            .dit => "dit",
            .vae => "vae",
            .latent => "latent",
            .lm_weights => "weights",
            .lm_kv => "kv",
            .lm_act => "act",
            .lm_logits => "logits",
            .lm_dequant => "dequant",
        };
    }
};

const std = @import("std");

test "count covers every tag" {
    // `mem_tag_used` is `[count]usize` indexed by `@intFromEnum`, so a tag added
    // without bumping `count` indexes past the end of that array on its first
    // allocation. Cheaper to catch here than as memory corruption on a GPU box.
    try std.testing.expectEqual(@typeInfo(MemTag).@"enum".fields.len, MemTag.count);
    // And the values must stay dense from 0, since that indexing assumes it.
    inline for (@typeInfo(MemTag).@"enum".fields, 0..) |f, i| {
        try std.testing.expectEqual(i, f.value);
    }
}

test "every tag has a label" {
    inline for (@typeInfo(MemTag).@"enum".fields) |f| {
        const t: MemTag = @enumFromInt(f.value);
        try std.testing.expect(t.name().len > 0);
    }
}
