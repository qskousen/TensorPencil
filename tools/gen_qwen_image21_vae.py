#!/usr/bin/env python3
"""Pin the Qwen-Image 2.1 VAE by EXECUTING ComfyUI's own (`comfy.sd.VAE` over
`comfy/ldm/wan/vae2_2.py`: the Wan 2.2 layout at temporal kernel 1, RGBA, z 64 at 16x)
on the real checkpoint, CPU fp32.

Records:
  - the latent format (`latent_formats.QwenImage21`): mean, std, the latent->RGB
    preview factors and bias, so the Zig constants are held to ComfyUI's;
  - `z_norm`, a sampler-space latent, `z` = process_out(z_norm), and `out`, the raw
    decoder output for `z` ([4][h][w] RGBA, about [-1, 1]);
  - `image`, `VAE.decode`'s finished [h][w][4] in [0, 1], which is what a saver gets;
  - `rgb` and `enc`: a smooth RGB picture and `VAE.encode`'s latent for it, alpha
    padded at 1.0 the way the node does for a 3-channel input.

Emits src/models/assets/qwen_image21_vae.safetensors.

Usage:
    systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 \\
        /home/qt/genai/comfyui/nvenv/bin/python tools/gen_qwen_image21_vae.py
"""
import os
import sys

VAE = "/mnt/watch/models/diffusion/vae/qwen_image_2.1_vae_bf16.safetensors"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "src", "models", "assets", "qwen_image21_vae.safetensors")

sys.argv = [sys.argv[0], "--cpu", "--disable-smart-memory", "--fp32-vae"]
sys.path.insert(0, "/home/qt/genai/comfyui")
os.chdir("/home/qt/genai/comfyui")
import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()
import torch  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
import comfy.latent_formats as lf  # noqa: E402

torch.manual_seed(20261019)
sd = comfy.utils.load_torch_file(VAE)
vae = comfy.sd.VAE(sd=sd, device=torch.device("cpu"), dtype=torch.float32)
if vae.latent_channels != 64 or vae.output_channels != 4:
    raise SystemExit(f"ComfyUI read {vae.latent_channels} ch / {vae.output_channels} out, not the 2.1 VAE")
fmt = lf.QwenImage21()

out = {
    "mean": fmt.latents_mean.flatten().float().contiguous(),
    "std": fmt.latents_std.flatten().float().contiguous(),
    "rgb_factors": torch.tensor(fmt.latent_rgb_factors, dtype=torch.float32),
    "rgb_bias": torch.tensor(fmt.latent_rgb_factors_bias, dtype=torch.float32),
}

H, W = 4, 6
with torch.inference_mode():
    z_norm = torch.randn(1, 64, H, W)
    z = fmt.process_out(z_norm)
    raw = vae.first_stage_model.decode(z.float())
    image = vae.decode(z)
    if raw.shape != (1, 4, 16 * H, 16 * W):
        raise SystemExit(f"decoder gave {tuple(raw.shape)}")
    out["z_norm"] = z_norm[0].contiguous()
    out["z"] = z[0].float().contiguous()
    out["out"] = raw[0].float().contiguous()
    out["image"] = image.reshape(16 * H, 16 * W, 4).float().contiguous()

    yy, xx = torch.meshgrid(torch.linspace(-1, 1, 16 * H), torch.linspace(-1, 1, 16 * W), indexing="ij")
    rgb = torch.stack([torch.sin(3 * xx + 1), torch.cos(2 * yy - xx), xx * yy], -1) * 0.45 + 0.5
    enc = vae.encode(rgb[None])
    if enc.shape != (1, 64, H, W):
        raise SystemExit(f"encoder gave {tuple(enc.shape)}")
    out["rgb"] = rgb.contiguous()
    out["enc"] = enc[0].float().contiguous()
    print(f"raw out rms {raw.pow(2).mean().sqrt():.4f}, alpha mean {raw[0, 3].mean():.4f}, enc rms {enc.pow(2).mean().sqrt():.4f}")

comfy.utils.save_torch_file(out, OUT)
print("wrote", OUT, os.path.getsize(OUT) // 1024, "KiB")
