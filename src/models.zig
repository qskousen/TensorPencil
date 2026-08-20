//! Model implementations (text encoder, DiT, VAE).

pub const loader = @import("models/loader.zig");
pub const residency = @import("tp_runtime").residency;
/// The turn hook a frontend installs on a stepper (`model.boundary`), so a
/// prefill chunk loop reaches the same residency/pause/cancel intents the decode
/// loop polls between tokens.
pub const boundary = @import("tp_runtime").boundary;
pub const transformer = @import("models/transformer.zig");
pub const transformer_gpu = @import("models/transformer_gpu.zig");
pub const wan_vae = @import("models/wan_vae.zig");
pub const taehv = @import("models/taehv.zig");
pub const taehv_cuda = @import("models/taehv_cuda.zig");
pub const taehv_gpu = @import("models/taehv_gpu.zig");
pub const vae_gpu = @import("models/vae_gpu.zig");
pub const vae_cuda = @import("models/vae_cuda.zig");
pub const vae_tiled = @import("models/vae_tiled.zig");
pub const qwen3 = @import("models/qwen3.zig");
pub const qwen35 = @import("models/qwen35.zig");
pub const gemma3 = @import("models/gemma3.zig");
pub const gemma4 = @import("models/gemma4.zig");
pub const gemma4_cuda = @import("models/gemma4_cuda.zig");
pub const gemma4_vit = @import("models/gemma4_vit.zig");
pub const gemma4_vit_cuda = @import("models/gemma4_vit_cuda.zig");
pub const gemma4v_vit = @import("models/gemma4v_vit.zig");
pub const gemma4v_vit_cuda = @import("models/gemma4v_vit_cuda.zig");
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
/// SD-family (SD1.5 / SDXL) conditioner, CLIP-L and CLIP-G text towers.
pub const clip_text = @import("models/clip_text.zig");
pub const clip_text_gpu = @import("models/clip_text_gpu.zig");
pub const clip_text_cuda = @import("models/clip_text_cuda.zig");
/// SD-family denoiser, the LDM UNet.
pub const sd_unet = @import("models/sd_unet.zig");
pub const sd_unet_gpu = @import("models/sd_unet_gpu.zig");
pub const sd_unet_cuda = @import("models/sd_unet_cuda.zig");
/// SD-family VAE decoder (AutoencoderKL).
pub const sd_vae = @import("models/sd_vae.zig");
pub const sd_vae_gpu = @import("models/sd_vae_gpu.zig");
pub const sd_vae_cuda = @import("models/sd_vae_cuda.zig");
pub const lpips = @import("models/lpips.zig");
pub const embed_gemma = @import("models/embed_gemma.zig");
pub const embed_gemma_gpu = @import("models/embed_gemma_gpu.zig");
pub const embed_gemma_cuda = @import("models/embed_gemma_cuda.zig");
pub const embed_snowflake = @import("models/embed_snowflake.zig");
pub const embed_snowflake_gpu = @import("models/embed_snowflake_gpu.zig");
pub const embed_snowflake_cuda = @import("models/embed_snowflake_cuda.zig");
pub const embed_siglip = @import("models/embed_siglip.zig");
pub const embed_siglip_gpu = @import("models/embed_siglip_gpu.zig");
pub const embed_siglip_cuda = @import("models/embed_siglip_cuda.zig");
pub const quant_weight = @import("models/quant_weight.zig");
pub const dit = @import("models/dit.zig");
pub const dit_gpu = @import("models/dit_gpu.zig");
pub const dit_cuda = @import("models/dit_cuda.zig");
/// Z-Image (`NextDiT`) denoiser, the architecture "zit" checkpoints use.
pub const zimage = @import("models/zimage.zig");
pub const zimage_text = @import("models/zimage_text.zig");
pub const zimage_gpu = @import("models/zimage_gpu.zig");
pub const zimage_cuda = @import("models/zimage_cuda.zig");
/// Anima, a Cosmos-Predict2 `MiniTrainDIT` plus an `llm_adapter` that fuses a T5
/// tokenization of the prompt with a Qwen3-0.6B encode of it.
pub const anima = @import("models/anima.zig");
pub const anima_gpu = @import("models/anima_gpu.zig");
pub const anima_cuda = @import("models/anima_cuda.zig");

test {
    _ = quant_weight;
    _ = loader;
    _ = clip_text;
    _ = clip_text_gpu;
    _ = clip_text_cuda;
    _ = sd_unet;
    _ = sd_unet_gpu;
    _ = sd_unet_cuda;
    _ = sd_vae;
    _ = sd_vae_gpu;
    _ = sd_vae_cuda;
    _ = residency;
    _ = transformer;
    _ = transformer_gpu;
    _ = wan_vae;
    _ = taehv_gpu;
    _ = vae_gpu;
    _ = vae_cuda;
    _ = vae_tiled;
    _ = qwen3;
    _ = qwen35;
    _ = gemma3;
    _ = gemma4;
    _ = gemma4_cuda;
    _ = gemma4_vit;
    _ = gemma4_vit_cuda;
    _ = gemma4v_vit;
    _ = gemma4v_vit_cuda;
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
    _ = lpips;
    _ = embed_gemma;
    _ = embed_gemma_gpu;
    _ = embed_gemma_cuda;
    _ = embed_snowflake;
    _ = embed_snowflake_gpu;
    _ = embed_snowflake_cuda;
    _ = embed_siglip;
    _ = embed_siglip_gpu;
    _ = embed_siglip_cuda;
    _ = dit;
    _ = dit_gpu;
    _ = dit_cuda;
    _ = zimage;
    _ = zimage_text;
    _ = zimage_gpu;
    _ = zimage_cuda;
    _ = anima;
    _ = anima_gpu;
    _ = anima_cuda;
    // Device test relocated out of the gpu backend (it needs both tp_gpu and a
    // model CPU reference); lives here in the model tier.
    _ = @import("models/vit35_gpu_test.zig");
}
