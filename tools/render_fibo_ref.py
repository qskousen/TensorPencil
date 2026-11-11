#!/usr/bin/env python3
"""Render one Fibo 1.5 image through diffusers' own `BriaFiboPipeline` (what the
ComfyUI_Fibo-1.5 custom nodes run), for a whole-render A/B against
`TensorPencil generate` at the same prompt, seed and settings.

The initial latent is torch's CPU generator at the seed, which `--rng cpu` reproduces.
The models are offloaded to the CPU between stages so the 8B DiT fits a 24 GB card.

Usage:
    /home/qt/genai/comfyui/nvenv/bin/python tools/render_fibo_ref.py \\
        --prompt-file p.txt --seed 42 --steps 30 --cfg 5 --out scratch_out/ref.png
"""

import argparse

import torch
from diffusers import BriaFiboPipeline

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="/mnt/watch/models/diffusion/diffusers/Fibo-1.5")
ap.add_argument("--prompt-file", required=True)
ap.add_argument("--negative", default="")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--cfg", type=float, default=5.0)
ap.add_argument("--width", type=int, default=1024)
ap.add_argument("--height", type=int, default=1024)
ap.add_argument("--out", required=True)
args = ap.parse_args()

pipe = BriaFiboPipeline.from_pretrained(args.repo, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()
gen = torch.Generator("cpu").manual_seed(args.seed)
latents = torch.randn((1, 48, args.height // 16, args.width // 16), generator=gen, dtype=torch.float32)
latents = latents.permute(0, 2, 3, 1).reshape(1, -1, 48).to(torch.bfloat16)
image = pipe(prompt=open(args.prompt_file).read().strip(), negative_prompt=args.negative, height=args.height, width=args.width,
             num_inference_steps=args.steps, guidance_scale=args.cfg, latents=latents).images[0]
image.save(args.out)
print("wrote", args.out)
