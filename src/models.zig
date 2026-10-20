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
pub const vae_tiled = @import("models/vae_tiled.zig");
pub const qwen3 = @import("models/qwen3.zig");
pub const k2_horizon = @import("models/k2_horizon.zig");
pub const k2_horizon_cuda = @import("models/k2_horizon_cuda.zig");
pub const k2_horizon_gpu = @import("models/k2_horizon_gpu.zig");
pub const k2_horizon_dev = @import("models/k2_horizon_dev.zig");
pub const qwen35 = @import("models/qwen35.zig");
pub const gemma3 = @import("models/gemma3.zig");
pub const gemma4 = @import("models/gemma4.zig");
pub const gemma4_cuda = @import("models/gemma4_cuda.zig");
/// Gemma 4 on Vulkan.
pub const gemma4_gpu = @import("models/gemma4_gpu.zig");
pub const gemma4_vit = @import("models/gemma4_vit.zig");
pub const gemma4_vit_cuda = @import("models/gemma4_vit_cuda.zig");
pub const gemma4_vit_gpu = @import("models/gemma4_vit_gpu.zig");
pub const gemma4v_vit = @import("models/gemma4v_vit.zig");
pub const gemma4v_vit_cuda = @import("models/gemma4v_vit_cuda.zig");
pub const gemma4v_vit_gpu = @import("models/gemma4v_vit_gpu.zig");
pub const gemma3_cuda = @import("models/gemma3_cuda.zig");
pub const gemma3_gpu = @import("models/gemma3_gpu.zig");
pub const gemma_vit = @import("models/gemma_vit.zig");
pub const gemma_vit_cuda = @import("models/gemma_vit_cuda.zig");
pub const gemma_vit_gpu = @import("models/gemma_vit_gpu.zig");
pub const qwen35_cuda = @import("models/qwen35_cuda.zig");
pub const vit35 = @import("models/vit35.zig");
pub const vit35_cuda = @import("models/vit35_cuda.zig");
/// That tower on Vulkan, over the forward `vit35_dev.zig` holds for both backends.
pub const vit35_gpu = @import("models/vit35_gpu.zig");
pub const vit35_dev = @import("models/vit35_dev.zig");
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
pub const conv_gpu = @import("models/conv_gpu.zig");
pub const conv_cuda = @import("models/conv_cuda.zig");
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
/// What a block linear is (`lin`), the one linear dispatcher's routing for every
/// backend and caller (`dual_lin`), and the CUDA dispatcher every device forward on
/// either CUDA arm goes through (`lin_cuda`).
pub const lin = @import("models/lin.zig");
pub const dual_lin = @import("models/dual_lin.zig");
pub const lin_cuda = @import("models/lin_cuda.zig");
pub const lin_gpu = @import("models/lin_gpu.zig");
pub const dit = @import("models/dit.zig");
pub const dit_gpu = @import("models/dit_gpu.zig");
pub const dit_cuda = @import("models/dit_cuda.zig");
/// SenseNova U1.5, a Qwen3-shaped MoT trunk that generates in pixel space: no VAE
/// and no separate text encoder, the prompt's own KV cache is the conditioning.
pub const sensenova = @import("models/sensenova.zig");
pub const sensenova_gpu = @import("models/sensenova_gpu.zig");
pub const sensenova_cuda = @import("models/sensenova_cuda.zig");
/// Z-Image (`NextDiT`) denoiser, the architecture "zit" checkpoints use.
pub const zimage = @import("models/zimage.zig");
pub const zimage_text = @import("models/zimage_text.zig");
pub const zimage_gpu = @import("models/zimage_gpu.zig");
pub const zimage_cuda = @import("models/zimage_cuda.zig");
pub const zimage_dev = @import("models/zimage_dev.zig");
pub const sensenova_dev = @import("models/sensenova_dev.zig");
pub const minimax_h3_dev = @import("models/minimax_h3_dev.zig");
pub const anima_dev = @import("models/anima_dev.zig");
pub const mageflow_dev = @import("models/mageflow_dev.zig");
/// Anima, a Cosmos-Predict2 `MiniTrainDIT` plus an `llm_adapter` that fuses a T5
/// tokenization of the prompt with a Qwen3-0.6B encode of it.
pub const anima = @import("models/anima.zig");
pub const anima_gpu = @import("models/anima_gpu.zig");
pub const anima_cuda = @import("models/anima_cuda.zig");
/// Mage-Flow, a 12-block double-stream MMDiT at one token per latent pixel.
pub const mageflow = @import("models/mageflow.zig");
pub const mageflow_text = @import("models/mageflow_text.zig");
pub const mageflow_gpu = @import("models/mageflow_gpu.zig");
pub const mageflow_cuda = @import("models/mageflow_cuda.zig");
pub const moss_image = @import("models/moss_image.zig");
pub const moss_image_text = @import("models/moss_image_text.zig");
pub const qwen_image21 = @import("models/qwen_image21.zig");
pub const qwen_image21_text = @import("models/qwen_image21_text.zig");
pub const qwen_image21_dev = @import("models/qwen_image21_dev.zig");
pub const qwen_image21_cuda = @import("models/qwen_image21_cuda.zig");
pub const qwen_image21_gpu = @import("models/qwen_image21_gpu.zig");
pub const moss_image_dev = @import("models/moss_image_dev.zig");
pub const moss_image_cuda = @import("models/moss_image_cuda.zig");
pub const moss_image_gpu = @import("models/moss_image_gpu.zig");
pub const attn_gpu = @import("models/attn_gpu.zig");
/// Mage-Flow's VAE: a one-step diffusion codec, 128 channels at 16x, not an
/// `AutoencoderKL`.
pub const mage_vae = @import("models/mage_vae.zig");
pub const mage_vae_gpu = @import("models/mage_vae_gpu.zig");
pub const mage_vae_cuda = @import("models/mage_vae_cuda.zig");
pub const mage_vae_dev = @import("models/mage_vae_dev.zig");
/// MiniMax H3, a joint audio-video DiT: video and stereo audio denoised in one
/// packed token sequence, on two different sigma schedules.
pub const minimax_h3 = @import("models/minimax_h3.zig");
/// H3's video VAE decode side: a ViT3D over the latent grid, not a CNN.
pub const minimax_h3_vae = @import("models/minimax_h3_vae.zig");
/// H3's video VAE ENCODE side: a 3-D causal CNN, a different architecture from
/// the ViT3D decoder it shares a checkpoint with. The gate on fl2va and ref2va.
pub const minimax_h3_vae_encode = @import("models/minimax_h3_vae_encode.zig");
pub const minimax_h3_vae_encode_cuda = @import("models/minimax_h3_vae_encode_cuda.zig");
/// H3's vision tower: Qwen3-VL-32B's ViT with DeepStack. NOT `vit35.zig`, which
/// is the same tower from llama.cpp's lineage and diverges in two conventions.
pub const minimax_h3_vit = @import("models/minimax_h3_vit.zig");
/// That tower on the CUDA backend. Its own file rather than `vit35_cuda`, which
/// refuses DeepStack and flattens its merger into the `Vit`.
pub const minimax_h3_vit_cuda = @import("models/minimax_h3_vit_cuda.zig");
/// H3's prompt presentation: reference labels and spliced vision blocks, and the
/// one place the tag / injection / mrope spans are derived from.
pub const minimax_h3_present = @import("models/minimax_h3_present.zig");
/// H3's audio VAE decode side: a BigVGAN vocoder, the only 1-D convolution here.
pub const minimax_h3_audio = @import("models/minimax_h3_audio.zig");
/// H3's audio VAE encode side: the DAC down-sampler that turns a reference
/// soundtrack into `ref_audio` rows. Shares only `Conv1d` with the decode side.
pub const minimax_h3_audio_encode = @import("models/minimax_h3_audio_encode.zig");
pub const minimax_h3_audio_encode_cuda = @import("models/minimax_h3_audio_encode_cuda.zig");
pub const minimax_h3_cuda = @import("models/minimax_h3_cuda.zig");
/// H3's trunk on Vulkan, the twin of `minimax_h3_cuda`.
pub const minimax_h3_gpu = @import("models/minimax_h3_gpu.zig");
/// H3's video VAE decode on Vulkan.
pub const minimax_h3_vae_gpu = @import("models/minimax_h3_vae_gpu.zig");
/// H3's audio VAE decode on Vulkan.
pub const minimax_h3_audio_gpu = @import("models/minimax_h3_audio_gpu.zig");
pub const minimax_h3_vae_cuda = @import("models/minimax_h3_vae_cuda.zig");
pub const minimax_h3_audio_cuda = @import("models/minimax_h3_audio_cuda.zig");
/// Runtime low-rank sidecars (LoRA), applied beside a GEMM rather than merged
/// into the weight. Architecture-independent: it keys on `Weight.tag`.
pub const lora = @import("models/lora.zig");
pub const lora_cuda = @import("models/lora_cuda.zig");
pub const lora_gpu = @import("models/lora_gpu.zig");
pub const lora_sd = @import("models/lora_sd.zig");
pub const lora_adapter = @import("models/lora_adapter.zig");

test {
    _ = lora;
    _ = lora_cuda;
    _ = lora_gpu;
    _ = lora_sd;
    _ = lora_adapter;
    _ = minimax_h3_audio_cuda;
    _ = minimax_h3_audio_encode_cuda;
    _ = minimax_h3_vae_encode;
    _ = minimax_h3_vae_encode_cuda;
    _ = minimax_h3_vit;
    _ = minimax_h3_vit_cuda;
    _ = minimax_h3_present;
    _ = quant_weight;
    _ = lin;
    _ = lin_cuda;
    _ = dual_lin;
    _ = lin_gpu;
    _ = loader;
    _ = clip_text;
    _ = clip_text_gpu;
    _ = clip_text_cuda;
    _ = sd_unet;
    _ = sd_unet_gpu;
    _ = conv_gpu;
    _ = conv_cuda;
    _ = sd_unet_cuda;
    _ = sd_vae;
    _ = sd_vae_gpu;
    _ = sd_vae_cuda;
    _ = residency;
    _ = transformer;
    _ = transformer_gpu;
    _ = wan_vae;
    _ = taehv;
    _ = taehv_gpu;
    _ = vae_tiled;
    _ = qwen3;
    _ = qwen35;
    _ = gemma3;
    _ = gemma4;
    _ = gemma4_cuda;
    _ = gemma4_vit;
    _ = gemma4_vit_gpu;
    _ = gemma4_vit_cuda;
    _ = gemma4v_vit;
    _ = gemma4v_vit_gpu;
    _ = gemma4v_vit_cuda;
    _ = gemma3_cuda;
    _ = gemma3_gpu;
    _ = gemma_vit;
    _ = gemma_vit_cuda;
    _ = gemma_vit_gpu;
    _ = qwen35_cuda;
    _ = vit35;
    _ = vit35_cuda;
    _ = vit35_gpu;
    _ = vit35_dev;
    _ = qwen3_gpu;
    _ = qwen35_gpu;
    _ = gemma4_gpu;
    _ = @import("models/gemma4v_vit_dev.zig");
    _ = @import("models/gemma4_vit_dev.zig");
    _ = qwen3_cuda;
    _ = k2_horizon;
    _ = k2_horizon_cuda;
    _ = k2_horizon_gpu;
    _ = k2_horizon_dev;
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
    _ = sensenova;
    _ = sensenova_gpu;
    _ = sensenova_cuda;
    _ = zimage;
    _ = zimage_text;
    _ = zimage_gpu;
    _ = zimage_cuda;
    _ = anima;
    _ = anima_gpu;
    _ = anima_cuda;
    _ = mageflow;
    _ = mageflow_text;
    _ = mageflow_gpu;
    _ = mageflow_cuda;
    _ = moss_image;
    _ = moss_image_text;
    _ = qwen_image21;
    _ = qwen_image21_text;
    _ = qwen_image21_cuda;
    _ = qwen_image21_gpu;
    _ = attn_gpu;
    _ = moss_image_cuda;
    _ = moss_image_gpu;
    _ = mage_vae;
    _ = mage_vae_gpu;
    _ = mage_vae_cuda;
    _ = minimax_h3;
    _ = minimax_h3_vae;
    _ = minimax_h3_audio;
    _ = minimax_h3_audio_encode;
    _ = minimax_h3_cuda;
    _ = minimax_h3_gpu;
    _ = minimax_h3_vae_gpu;
    _ = minimax_h3_audio_gpu;
    _ = minimax_h3_vae_cuda;
    // Device test relocated out of the gpu backend (it needs both tp_gpu and a
    // model CPU reference); lives here in the model tier.
    _ = @import("models/vit35_gpu_test.zig");
}
