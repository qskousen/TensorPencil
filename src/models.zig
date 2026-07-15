//! Model implementations (text encoder, DiT, VAE).

pub const wan_vae = @import("models/wan_vae.zig");
pub const taehv = @import("models/taehv.zig");
pub const taehv_cuda = @import("models/taehv_cuda.zig");
pub const vae_gpu = @import("models/vae_gpu.zig");
pub const vae_cuda = @import("models/vae_cuda.zig");
pub const qwen3 = @import("models/qwen3.zig");
pub const qwen35 = @import("models/qwen35.zig");
pub const gemma3 = @import("models/gemma3.zig");
pub const gemma4 = @import("models/gemma4.zig");
pub const gemma4_cuda = @import("models/gemma4_cuda.zig");
pub const gemma4_vit = @import("models/gemma4_vit.zig");
pub const gemma3_cuda = @import("models/gemma3_cuda.zig");
pub const gemma3_gpu = @import("models/gemma3_gpu.zig");
pub const gemma_vit = @import("models/gemma_vit.zig");
pub const gemma_vit_cuda = @import("models/gemma_vit_cuda.zig");
pub const gemma_vit_gpu = @import("models/gemma_vit_gpu.zig");
pub const qwen35_cuda = @import("models/qwen35_cuda.zig");
pub const vit35 = @import("models/vit35.zig");
pub const vit35_cuda = @import("models/vit35_cuda.zig");
pub const qwen3_gpu = @import("models/qwen3_gpu.zig");
pub const qwen35_gpu = @import("models/qwen35_gpu.zig");
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
    _ = gemma3;
    _ = gemma4;
    _ = gemma4_cuda;
    _ = gemma4_vit;
    _ = gemma3_cuda;
    _ = gemma3_gpu;
    _ = gemma_vit;
    _ = gemma_vit_cuda;
    _ = gemma_vit_gpu;
    _ = qwen35_cuda;
    _ = vit35;
    _ = vit35_cuda;
    _ = qwen3_gpu;
    _ = qwen35_gpu;
    _ = qwen3_cuda;
    _ = eagle3;
    _ = krea2_text;
    _ = dit;
    _ = dit_gpu;
    _ = dit_cuda;
}
