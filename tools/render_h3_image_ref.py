#!/usr/bin/env python3
"""A whole MiniMax-H3-Image render through ComfyUI, for the comparison the stage
fixtures cannot make (see tools/gen_moss_image_render.py for why one is needed).

It mirrors the release workflow (`image_minimax_h3_image_edit`): the
`MiniMaxH3ImageConditioning` node, euler, `simple`, cfg 1 against a zeroed
negative, the model's own PDD-grid sampling, and torch's CPU noise, which is what
the engine's default `--rng` reproduces.

The encoder (27 GB) and the DiT (21 GB) do not fit in this box's RAM together
under ComfyUI, so the render is TWO processes: `--phase encode` writes the
conditioning to `<out>.cond.pt` and exits, `--phase sample` reads it back and
samples and decodes. Run each under a memory cap:

    systemd-run --user --scope -p MemoryMax=40G -p MemorySwapMax=0 \\
        /home/qt/genai/comfyui/nvenv/bin/python tools/render_h3_image_ref.py \\
        --phase encode --out scratch_out/h3i_comfy
    systemd-run --user --scope -p MemoryMax=40G -p MemorySwapMax=0 \\
        /home/qt/genai/comfyui/nvenv/bin/python tools/render_h3_image_ref.py \\
        --phase sample --out scratch_out/h3i_comfy

`--no-int8-attention` swaps the file's int8 attention for ComfyUI's ordinary one,
the control row: how far ComfyUI moves from itself. `--ref-image` (repeatable)
makes it an edit; the VAE is then needed in the encode phase too.
"""

import argparse
import os
import sys

COMFY = "/home/qt/genai/comfyui"
ROOT = "/mnt/watch/models/diffusion"
DEFAULT_DIT = os.path.join(ROOT, "diffusion_models/h3-image/minimax_h3_image_ggufy_int8_convrot.safetensors")
DEFAULT_TE = os.path.join(ROOT, "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors")
DEFAULT_VAE = os.path.join(ROOT, "vae/minimax_h3_image_vae_fp16.safetensors")

# `--lowvram` lets an edit's encode (the encoder, its vision tower and the VAE)
# fit a 24 GB card.
_COMFY_FLAGS = ("--cpu", "--lowvram")
OUR_ARGV = [a for a in sys.argv[1:] if a not in _COMFY_FLAGS + ("--no-int8-attention",)]
NO_INT8_ATTENTION = "--no-int8-attention" in sys.argv[1:]
_passthrough = [a for a in sys.argv[1:] if a in _COMFY_FLAGS]
sys.argv = [sys.argv[0], "--disable-smart-memory", "--disable-xformers",
            "--use-pytorch-cross-attention"] + _passthrough
# ComfyUI wants its own directory as the cwd; paths on the command line are the caller's.
CALLER_CWD = os.getcwd()
sys.path.insert(0, COMFY)
os.chdir(COMFY)

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import comfy.sd  # noqa: E402
import comfy_kitchen  # noqa: E402

if NO_INT8_ATTENTION:
    comfy_kitchen.int8_attention_is_available = lambda device: False
import comfy.utils  # noqa: E402
import comfy.model_management  # noqa: E402
import nodes  # noqa: E402
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageConditioning  # noqa: E402


def load_image(path):
    # LoadImage's conversion: 8-bit RGB to f32 / 255, [1, H, W, 3]
    arr = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None]


def encode(args):
    clip = comfy.sd.load_clip(ckpt_paths=[args.text_encoder], embedding_directory=None,
                              clip_type=comfy.sd.CLIPType.MINIMAX)
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(args.vae)) if args.ref_image else None
    images = {"image_%d" % (i + 1): load_image(p) for i, p in enumerate(args.ref_image)}
    out = MiniMaxH3ImageConditioning.execute(clip, args.prompt, args.width, args.height, vae=vae,
                                             images=images or None)
    positive, latent = out.result if hasattr(out, "result") else out
    torch.save({"positive": positive, "latent": latent}, args.out + ".cond.pt")
    print("wrote %s.cond.pt  tokens %d  latent %s" % (args.out, positive[0][0].shape[1],
                                                       tuple(latent["samples"].shape)))


def sample(args):
    saved = torch.load(args.out + ".cond.pt", weights_only=False)
    positive, latent = saved["positive"], saved["latent"]
    (negative,) = nodes.ConditioningZeroOut().zero_out(positive)
    model = comfy.sd.load_diffusion_model(args.dit)
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(args.vae))
    import time
    t0 = time.perf_counter()
    (out,) = nodes.common_ksampler(model, args.seed, args.steps, args.cfg, "euler", "simple",
                                   positive, negative, latent, denoise=1.0)
    t1 = time.perf_counter()
    image = vae.decode(out["samples"])
    print("sampling %.1fs (%.2fs/step), decode %.1fs" % (t1 - t0, (t1 - t0) / args.steps,
                                                         time.perf_counter() - t1))
    arr = (image.reshape(-1, *image.shape[-3:])[0].detach().cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(args.out + ".png")
    comfy.utils.save_torch_file({"latent": out["samples"][0].contiguous().float().cpu()}, args.out + ".safetensors")
    print("wrote %s.png and %s.safetensors" % (args.out, args.out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("encode", "sample"), required=True)
    ap.add_argument("--dit", default=DEFAULT_DIT)
    ap.add_argument("--text-encoder", default=DEFAULT_TE)
    ap.add_argument("--vae", default=DEFAULT_VAE)
    ap.add_argument("--prompt", default="A red fox sitting in a snowy pine forest at dawn, soft golden light, photorealistic")
    ap.add_argument("--ref-image", action="append", default=[])
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="scratch_out/h3i_comfy")
    args = ap.parse_args(OUR_ARGV)
    args.out = os.path.join(CALLER_CWD, args.out)
    args.ref_image = [os.path.join(CALLER_CWD, p) for p in args.ref_image]
    encode(args) if args.phase == "encode" else sample(args)


if __name__ == "__main__":
    main()
