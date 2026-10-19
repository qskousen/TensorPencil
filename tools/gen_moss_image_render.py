#!/usr/bin/env python3
"""A whole MOSS-Image render through ComfyUI, for the comparison the per-stage
fixtures cannot make.

The three stage fixtures each pin one piece against the reference, but a correct
port of the WRONG convention passes every one of them: matching a reference
exactly still leaves the choice of reference unvalidated. Only rendering the same
prompt, seed, schedule and sampler on both sides and comparing pixels catches
that, so this drives ComfyUI's own `common_ksampler` and writes the PNG plus the
final latent.

Settings mirror the engine's for this family: euler, `simple`, shift 5.0
(`MOSSImage.sampling_settings`), and torch's CPU MT19937 for the noise, which is
what ComfyUI uses and what `--rng cpu` reproduces. ComfyUI runs this checkpoint's
attention as int8 (`comfy_kitchen_int8`, per layer in the file) and its text
encoder in W8A8, so the reference carries both.

Usage (GPU):

    systemd-run --user --scope -p MemoryMax=24G -p MemorySwapMax=0 \\
        /home/qt/genai/comfyui/nvenv/bin/python tools/gen_moss_image_render.py \\
        --out scratch_out/moss_comfy
"""

import argparse
import os
import sys

COMFY = "/home/qt/genai/comfyui"
ROOT = "/mnt/watch/models/diffusion"
DEFAULT_DIT = os.path.join(ROOT, "diffusion_models/moss/moss_image_sft-INT8_CONVROT.safetensors")
DEFAULT_TE = os.path.join(ROOT, "text_encoders/qwen3.5_9b-INT8_CONVROT.safetensors")
DEFAULT_VAE = os.path.join(ROOT, "vae/flux2-vae.safetensors")

OUR_ARGV = [a for a in sys.argv[1:] if a not in ("--fp32-unet", "--cpu", "--no-int8-attention")]
# The control row: the same render with the checkpoint's int8 attention replaced by
# ComfyUI's ordinary one, which is how far ComfyUI moves from itself.
NO_INT8_ATTENTION = "--no-int8-attention" in sys.argv[1:]
# The two flags that decide the reference's OWN precision floor. A PSNR against
# this render is uninterpretable without the bf16-vs-fp32 control, so they are
# forwarded to ComfyUI rather than fixed here.
_passthrough = [a for a in sys.argv[1:] if a in ("--fp32-unet", "--cpu")]
sys.argv = [sys.argv[0], "--disable-smart-memory", "--disable-xformers",
            "--use-pytorch-cross-attention"] + _passthrough
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dit", default=DEFAULT_DIT)
    ap.add_argument("--text-encoder", default=DEFAULT_TE)
    ap.add_argument("--vae", default=DEFAULT_VAE)
    ap.add_argument("--prompt", default="a red apple on a wooden table")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--cfg", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--rounds", type=int, default=1, help="repeat the render; the LAST is reported")
    ap.add_argument("--out", default="/tmp/moss_comfy")
    args = ap.parse_args(OUR_ARGV)

    model = comfy.sd.load_diffusion_model(args.dit)
    # The family's own shift, set explicitly so a ComfyUI default change cannot
    # silently move the schedule this comparison depends on.
    ms = model.get_model_object("model_sampling")
    ms.set_parameters(shift=args.shift, multiplier=1.0)
    clip = comfy.sd.load_clip(ckpt_paths=[args.text_encoder], embedding_directory=None,
                              clip_type=comfy.sd.CLIPType.MOSS_IMAGE)
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(args.vae))

    positive = clip.encode_from_tokens_scheduled(clip.tokenize(args.prompt))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))

    latent = {"samples": torch.zeros([1, 128, args.height // 16, args.width // 16],
                                     device=comfy.model_management.intermediate_device())}
    # Timed in two phases, because the comparison they feed has a host-side VAE
    # on one arm: a single wall clock would hide which half moved. The first
    # `rounds - 1` are warm-up, so the figure is a steady state rather than the
    # weight upload and the CUDA JIT.
    import time
    sample_s, decode_s = 0.0, 0.0
    for r in range(args.rounds):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        (out,) = nodes.common_ksampler(model, args.seed, args.steps, args.cfg, "euler", "simple",
                                       positive, negative, latent, denoise=1.0)
        samples = out["samples"]
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t1 = time.perf_counter()
        image = vae.decode(samples)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t2 = time.perf_counter()
        sample_s, decode_s = t1 - t0, t2 - t1
        print(f"  round {r + 1}: sampling {sample_s:.2f}s ({sample_s / args.steps:.3f}s/step), "
              f"vae decode {decode_s:.2f}s, total {t2 - t0:.2f}s")
    print(f"TIMING sampling {sample_s:.3f} per_step {sample_s / args.steps:.4f} decode {decode_s:.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    arr = (image[0].detach().cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(args.out + ".png")
    comfy.utils.save_torch_file({"latent": samples[0].contiguous().float().cpu()}, args.out + ".safetensors")
    print(f"wrote {args.out}.png and {args.out}.safetensors  latent {tuple(samples.shape)}")


if __name__ == "__main__":
    main()
