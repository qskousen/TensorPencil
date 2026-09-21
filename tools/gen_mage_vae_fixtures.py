#!/usr/bin/env python3
"""Reference fixtures for Mage-VAE (`comfy/ldm/mage_flow/vae.py`).

Reference is **ComfyUI**, and the whole model fits in fp32 here (345 MB bf16),
so unlike the DiT fixture nothing is truncated: these are exact references for
the real codec at real width.

The cases are chosen for what they separate:

  - `small` (6x8 latent): one band of the per-pixel pathway, and a latent
    SMALLER than the 32x32 attention window, so it pins that the window is
    replicate-padded up rather than shrunk to fit.
  - `wide` (40x36 latent): 2x2 windows on BOTH axes, neither a whole multiple of
    32, so it pins the split itself and the padding of the last window on each
    axis. It also crosses the Zig side's NeRF band cap, pinning that banding is
    exact.
  - `enc` (80x48 image): the encoder direction, which shares only the DiCo block
    with decode and has its own affine-normed head blocks.

`cod` is stored alongside each decode output: the CoD decoder's conditioning is
where a windowing or padding mistake shows first, and comparing only the final
RGB would say "wrong" without saying where.

Run it memory-bounded:

    systemd-run --user --scope -p MemoryMax=20G -p MemorySwapMax=0 \\
        /home/qt/genai/comfyui/nvenv/bin/python tools/gen_mage_vae_fixtures.py
"""

import argparse
import hashlib
import json
import os
import sys

COMFY = "/home/qt/genai/comfyui"
DEFAULT_VAE = os.path.join(COMFY, "models/vae/mage_flow_vae_bf16.safetensors")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_OUT = os.path.join(REPO, "src", "models", "assets", "mage_vae_ref.safetensors")

SEED = 20260918
# (name, latent h, latent w)
DECODE_CASES = [("small", 6, 8), ("wide", 40, 36)]
# (name, image h, image w)
ENCODE_CASES = [("enc", 80, 48)]

OUR_ARGV = sys.argv[1:]
sys.argv = [sys.argv[0], "--cpu", "--fp32-vae", "--disable-smart-memory",
            "--disable-xformers", "--use-pytorch-cross-attention"]
sys.path.insert(0, COMFY)
os.chdir(COMFY)

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import torch  # noqa: E402
from comfy.cli_args import args as comfy_args  # noqa: E402
import comfy.model_management as _mm  # noqa: E402

if not (comfy_args.cpu and comfy_args.disable_xformers):
    raise SystemExit("ComfyUI did not take our flags — check comfy.options.enable_args_parsing()")
if _mm.xformers_enabled_vae():
    raise SystemExit("xformers is still active; it has no CPU fp32 attention kernel")

import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", default=DEFAULT_VAE)
    args = ap.parse_args(OUR_ARGV)
    if not os.path.exists(args.vae):
        raise SystemExit(f"not found: {args.vae}")

    sd = comfy.utils.load_torch_file(args.vae)
    vae = comfy.sd.VAE(sd=sd)
    model = vae.first_stage_model.to(torch.float32).eval()
    if type(model).__name__ != "MageVAE":
        raise SystemExit(f"ComfyUI picked {type(model).__name__} for this file, not MageVAE")
    if (vae.latent_channels, vae.downscale_ratio, vae.upscale_ratio) != (128, 16, 16):
        raise SystemExit(f"Mage-VAE's geometry moved: {vae.latent_channels}/{vae.downscale_ratio}/{vae.upscale_ratio}")
    # The attention really is windowed at 32; a global one would make `wide` moot.
    if model.decoder_model.y_embedder.decoder.block[1].patch_size != 32:
        raise SystemExit("the CoD decoder's attention window moved off 32")

    torch.manual_seed(SEED)
    out: dict[str, torch.Tensor] = {}
    meta = {
        "vae": os.path.basename(args.vae),
        "vae_sha256": sha256(args.vae),
        "decode_cases": json.dumps([c[0] for c in DECODE_CASES]),
        "encode_cases": json.dumps([c[0] for c in ENCODE_CASES]),
    }

    with torch.no_grad():
        for name, lh, lw in DECODE_CASES:
            z = torch.randn(1, 128, lh, lw, dtype=torch.float32)
            cond = model.decoder_model.y_embedder.decoder(z)
            rgb = model.decode(z)
            out[f"{name}.z"] = z[0].contiguous()
            out[f"{name}.cod"] = cond[0].contiguous()
            out[f"{name}.rgb"] = rgb[0].contiguous()
            meta[f"{name}.latent"] = json.dumps([lh, lw])
            print(f"{name}: latent {lh}x{lw} -> rgb {tuple(rgb.shape)}")

        for name, h, w in ENCODE_CASES:
            # The encoder takes [-1, 1], which is what `VAE.encode`'s
            # `process_input` produces from a [0, 1] image.
            img = torch.rand(1, 3, h, w, dtype=torch.float32) * 2.0 - 1.0
            lat = model.encode(img)
            out[f"{name}.img"] = img[0].contiguous()
            out[f"{name}.latent"] = lat[0].contiguous()
            meta[f"{name}.image"] = json.dumps([h, w])
            print(f"{name}: image {h}x{w} -> latent {tuple(lat.shape)}")

    os.makedirs(os.path.dirname(REF_OUT), exist_ok=True)
    comfy.utils.save_torch_file(out, REF_OUT, metadata=meta)
    print(f"wrote {REF_OUT} ({len(out)} tensors)")


if __name__ == "__main__":
    main()
