#!/usr/bin/env python3
"""Pin the Wan 2.2 TI2V VAE's single-frame decode (Fibo 1.5's decoder) by EXECUTING
ComfyUI's `comfy/ldm/wan/vae2_2.py` at a toy width.

What the toy keeps from the real file: 3-tap causal temporal kernels (only the last
tap sees a single frame), temporally upsampling stages whose `DupUp3D` shortcut drops
its first-chunk frames, and the patch-2 head unpatchified as `(c, r, q)`, the column
offset `r` before the row offset `q`. Seeded weights rounded to bf16 before the
reference runs, f32 on the CPU, stored as bf16 under ComfyUI's names with the latent
and the pixels.

Emits src/models/assets/wan22_ti2v_vae.safetensors, read by src/models/wan_vae.zig.

Usage (ComfyUI at or after fca60fe):
    COMFY=/path/to/comfyui /home/qt/genai/comfyui/nvenv/bin/python tools/gen_wan22_ti2v_vae.py
"""

import os
import sys

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "models", "assets", "wan22_ti2v_vae.safetensors")
COMFY = os.environ.get("COMFY", "/home/qt/genai/comfyui")
sys.argv = [sys.argv[0], "--cpu"]
sys.path.insert(0, COMFY)
os.chdir(COMFY)

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import torch  # noqa: E402
import safetensors.torch  # noqa: E402
import comfy.ldm.wan.vae2_2 as vae2_2  # noqa: E402


def main():
    torch.manual_seed(3)
    vae = vae2_2.WanVAE(dim=16, dec_dim=24, z_dim=8, dim_mult=[1, 2, 2, 2], num_res_blocks=1, temperal_downsample=[False, True, True]).float().eval()
    gen = torch.Generator().manual_seed(4)
    sd = vae.state_dict()
    for k, v in sd.items():
        if k.endswith("gamma"):
            w = 1.0 + 0.2 * torch.randn(v.shape, generator=gen)
        elif k.endswith(".bias"):
            w = 0.05 * torch.randn(v.shape, generator=gen)
        else:
            fan_in = v[0].numel() if v.ndim > 1 else 1
            w = torch.randn(v.shape, generator=gen) / fan_in ** 0.5
        sd[k] = w.to(torch.bfloat16).float()
    vae.load_state_dict(sd)
    z = torch.randn(1, 8, 3, 5, generator=gen)
    with torch.no_grad():
        img = vae.decode(z)
    print("pixels", tuple(img.shape), "rms", img.pow(2).mean().sqrt().item())
    out = {k: v.to(torch.bfloat16).contiguous() for k, v in sd.items() if k.startswith("decoder.") or k.startswith("conv2.")}
    out["test.z"] = z[0].contiguous()
    out["test.pixels"] = img[0].contiguous()
    safetensors.torch.save_file(out, OUT)
    print("wrote", OUT, os.path.getsize(OUT))


if __name__ == "__main__":
    main()
