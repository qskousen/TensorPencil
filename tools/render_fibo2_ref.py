#!/usr/bin/env python3
"""Render one Fibo 2.0 image through ComfyUI's own nodes, for a whole-render A/B
against `TensorPencil generate` at the same prompt, seed and settings.

`TextEncodeFibo2` (empty negative, so the null caption) and `common_ksampler` with
euler / simple, ComfyUI's template. The initial noise is torch's CPU generator at the
seed, which is what `--rng cpu` reproduces.

Usage (ComfyUI at or after fca60fe):
    COMFY=/path/to/comfyui /home/qt/genai/comfyui/nvenv/bin/python tools/render_fibo2_ref.py \\
        --prompt-file p.txt --seed 42 --steps 30 --cfg 5 --out scratch_out/ref.png
"""

import argparse
import os
import sys

COMFY = os.environ.get("COMFY", "/home/qt/genai/comfyui")
ROOT = "/mnt/watch/models/diffusion"

ap = argparse.ArgumentParser()
ap.add_argument("--dit", default=f"{ROOT}/diffusion_models/fibo/fibo2_dev_bf16.safetensors")
ap.add_argument("--te", default=f"{ROOT}/text_encoders/qwen3vl_4b_bf16.safetensors")
ap.add_argument("--vae", default=f"{ROOT}/vae/flux2-vae.safetensors")
ap.add_argument("--prompt-file", required=True)
ap.add_argument("--negative", default="")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--cfg", type=float, default=5.0)
ap.add_argument("--width", type=int, default=1024)
ap.add_argument("--height", type=int, default=1024)
ap.add_argument("--out", required=True)
args = ap.parse_args()
out_path = os.path.abspath(args.out)
prompt = open(args.prompt_file).read().strip()

sys.argv = [sys.argv[0]]
sys.path.insert(0, COMFY)
os.chdir(COMFY)

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
import nodes  # noqa: E402
from comfy_extras.nodes_fibo2 import TextEncodeFibo2  # noqa: E402
import comfy.ldm.modules.diffusionmodules.model as vae_model  # noqa: E402

# The decoder's fused-norm path needs a comfy_kitchen newer than some venvs carry; the
# unfused path is the same arithmetic.
vae_model.fused_norm_served = lambda *a, **k: False

model = comfy.sd.load_diffusion_model(args.dit)
clip = comfy.sd.load_clip(ckpt_paths=[args.te], clip_type=comfy.sd.CLIPType.FIBO2)
vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(args.vae))

res = TextEncodeFibo2.execute(clip, prompt, negative_prompt=args.negative, width=args.width, height=args.height)
positive, negative, latent = res.args
(samples,) = nodes.common_ksampler(model, args.seed, args.steps, args.cfg, "euler", "simple", positive, negative, latent)
image = vae.decode(samples["samples"])
arr = (image[0].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
Image.fromarray(arr).save(out_path)
print("wrote", out_path)
