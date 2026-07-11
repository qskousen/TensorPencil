//! Model implementations (text encoder, DiT, VAE).

pub const wan_vae = @import("models/wan_vae.zig");
pub const taehv = @import("models/taehv.zig");
pub const taehv_cuda = @import("models/taehv_cuda.zig");
pub const vae_gpu = @import("models/vae_gpu.zig");
pub const vae_cuda = @import("models/vae_cuda.zig");
pub const qwen3 = @import("models/qwen3.zig");
pub const qwen35 = @import("models/qwen35.zig");
pub const qwen35_cuda = @import("models/qwen35_cuda.zig");
pub const vit35 = @import("models/vit35.zig");
pub const vit35_cuda = @import("models/vit35_cuda.zig");
pub const qwen3_gpu = @import("models/qwen3_gpu.zig");
pub const qwen3_cuda = @import("models/qwen3_cuda.zig");
pub const eagle3 = @import("models/eagle3.zig");
pub const krea2_text = @import("models/krea2_text.zig");
pub const dit = @import("models/dit.zig");
pub const dit_gpu = @import("models/dit_gpu.zig");
pub const dit_cuda = @import("models/dit_cuda.zig");

test {
    _ = wan_vae;
    _ = vae_gpu;
    _ = vae_cuda;
    _ = qwen3;
    _ = qwen35;
    _ = qwen35_cuda;
    _ = vit35;
    _ = vit35_cuda;
    _ = qwen3_gpu;
    _ = qwen3_cuda;
    _ = eagle3;
    _ = krea2_text;
    _ = dit;
    _ = dit_gpu;
    _ = dit_cuda;
}
