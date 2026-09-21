#!/usr/bin/env python3
"""Render one SD1.5/SDXL image through ComfyUI, for a like-for-like comparison with
TensorPencil's own render of the same seed/prompt/steps/sampler/scheduler.

A whole-render check, which is the only thing that answers "is this sampler right on a
REAL model": the toy-denoiser fixtures in `tools/gen_sampler_fixtures.py` pin the
arithmetic against ComfyUI's own solver, but a toy denoiser is a contraction and a real
one is not, so a schedule whose last step is a huge log jump is exercised only here.

Drives ComfyUI's own `comfy.sample.sample`, the same entry point its KSampler node
uses, so the sampler, the scheduler and the noise all come from the reference.

    systemd-run --user --scope -p MemoryMax=14G -p MemorySwapMax=0 -- \\
        /home/qt/genai/comfyui/nvenv/bin/python tools/render_sd_ref.py --gpu \\
        --ckpt ~/genai/comfyui/models/checkpoints/dreamshaper_8.safetensors \\
        --prompt "a red fox in snow" --steps 8 --cfg 7.5 --seed 42 \\
        --sampler dpmpp_2m --scheduler normal --out scratch_out/ref.png

and the same render here:

    zig build run -Doptimize=ReleaseFast -- generate --dit <same> \\
        --prompt "a red fox in snow" --steps 8 --cfg 7.5 --seed 42 \\
        --width 512 --height 512 --sampler dpmpp_2m --scheduler normal \\
        --backend cuda --out scratch_out/ours.png
"""
import argparse, os, sys

COMFY = "/home/qt/genai/comfyui"
# Resolved BEFORE the chdir below, so a relative --out lands where the caller ran
# from and not inside ComfyUI's tree.
_OUT_BASE = os.getcwd()
OUR = [a for a in sys.argv[1:] if a != "--gpu"]
# ⚠️ CPU by default so the reference cannot depend on a GPU reduction order; `--gpu` is
# for when the card is free, which for an SD checkpoint is the usual case (2 GB).
_dev = ["--cpu"] if "--gpu" not in sys.argv[1:] else []
sys.argv = [sys.argv[0]] + _dev + ["--disable-smart-memory",
            "--disable-xformers", "--use-pytorch-cross-attention"]
sys.path.insert(0, COMFY)
os.chdir(COMFY)
import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()

import torch  # noqa: E402
import numpy as np  # noqa: E402
from comfy.cli_args import args as comfy_args  # noqa: E402
if not comfy_args.disable_xformers:
    raise SystemExit("ComfyUI did not take our flags; check enable_args_parsing()")

import comfy.sd  # noqa: E402
import comfy.sample  # noqa: E402
import comfy.samplers  # noqa: E402
import comfy.utils  # noqa: E402
import comfy.model_management  # noqa: E402
from PIL import Image  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--prompt", default="a red fox in snow")
ap.add_argument("--negative", default="")
ap.add_argument("--size", type=int, default=512)
ap.add_argument("--width", type=int, default=0, help="overrides --size")
ap.add_argument("--height", type=int, default=0, help="overrides --size")
ap.add_argument("--steps", type=int, default=8)
ap.add_argument("--cfg", type=float, default=7.5)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--sampler", default="euler")
ap.add_argument("--scheduler", default="normal")
ap.add_argument("--out", required=True)
a = ap.parse_args(OUR)
a.out = os.path.join(_OUT_BASE, a.out)

print(f"loading {os.path.basename(a.ckpt)}...")
model, clip, vae, _ = comfy.sd.load_checkpoint_guess_config(
    a.ckpt, output_vae=True, output_clip=True, embedding_directory=None)

cond = clip.encode_from_tokens_scheduled(clip.tokenize(a.prompt))
neg = clip.encode_from_tokens_scheduled(clip.tokenize(a.negative))

W = a.width or a.size
H = a.height or a.size
ch = model.model.latent_format.latent_channels
latent = torch.zeros([1, ch, H // 8, W // 8])
# `nodes.common_ksampler`'s noise: a CPU MT19937 generator, which is what
# `sampler.fillNoiseFrom(.torch_cpu)` reproduces.
noise = comfy.sample.prepare_noise(latent, a.seed, None)

sig = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), a.scheduler, a.steps)
print("sigmas:", [round(float(v), 6) for v in sig])

print(f"sampling {a.steps} steps of {a.sampler}/{a.scheduler} at {W}x{H}...")
samples = comfy.sample.sample(
    model, noise, a.steps, a.cfg, a.sampler, a.scheduler, cond, neg,
    latent, denoise=1.0, disable_noise=False, start_step=None,
    last_step=None, force_full_denoise=False, noise_mask=None, callback=None,
    disable_pbar=True, seed=a.seed)
print("final latent: max|x| %.3f" % float(samples.abs().max()))

img = vae.decode(samples.to(comfy.model_management.get_torch_device()))
arr = (img[0].clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
Image.fromarray(arr).save(a.out)
print(f"wrote {a.out} ({arr.shape[1]}x{arr.shape[0]})")
