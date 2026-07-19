//! Device-memory attribution tags. Each GPU backend context can optionally tag
//! every device allocation with the pipeline component it belongs to, so the
//! GUI VRAM meter can show a MEASURED per-component breakdown (not an estimate).
//! Tagging is opt-in per context (`track_tags`) — enabled only for the diffusion
//! backend, so LLM decode keeps its allocator hot path untouched.
//!
//! The counters are plain `u64`s read from the UI thread while the diffusion
//! worker updates them — a benign race (a momentarily stale number), exactly
//! like `device_used`. The per-pointer map they're derived from is only ever
//! touched on the allocating thread.

/// Which pipeline component an allocation belongs to. `other` is the default
/// (anything allocated outside a tagged phase — init overhead, pools); the
/// meter folds it into the "latent / working" segment.
pub const MemTag = enum(u8) {
    other = 0,
    te = 1, // text encoder (weights + encode scratch)
    dit = 2, // diffusion transformer (weights + denoise scratch)
    vae = 3, // VAE decoder (weights + decode scratch)
    latent = 4, // per-image working set (latent workspace, GPU session, live-preview decode)

    pub const count = 5;
};
