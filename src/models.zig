//! Model implementations (text encoder, DiT, VAE).

pub const wan_vae = @import("models/wan_vae.zig");
pub const vae_gpu = @import("models/vae_gpu.zig");
pub const vae_cuda = @import("models/vae_cuda.zig");
pub const qwen3 = @import("models/qwen3.zig");
pub const qwen3_gpu = @import("models/qwen3_gpu.zig");
pub const qwen3_cuda = @import("models/qwen3_cuda.zig");
pub const krea2_text = @import("models/krea2_text.zig");
pub const dit = @import("models/dit.zig");
pub const dit_gpu = @import("models/dit_gpu.zig");
pub const dit_cuda = @import("models/dit_cuda.zig");

test {
    _ = wan_vae;
    _ = vae_gpu;
    _ = vae_cuda;
    _ = qwen3;
    _ = qwen3_gpu;
    _ = qwen3_cuda;
    _ = krea2_text;
    _ = dit;
    _ = dit_gpu;
    _ = dit_cuda;
}
