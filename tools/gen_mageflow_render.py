#!/usr/bin/env python3
"""A whole Mage-Flow render through ComfyUI, for the comparison the per-stage
fixtures cannot make.

The three stage fixtures each pin one piece against the reference, but a correct
port of the WRONG convention passes every one of them: matching a reference
exactly still leaves the choice of reference unvalidated. Only rendering the same
prompt, seed, schedule and sampler on both sides and comparing pixels catches
that, so this drives ComfyUI's own `common_ksampler` and writes the PNG plus the
final latent.

Settings mirror the engine's defaults for this family exactly: euler, `simple`,
shift 6.0 (`MageFlow.sampling_settings`), cfg 1.0, and torch's CPU MT19937 for
the noise, which is what ComfyUI uses and what `--rng cpu` reproduces.

Usage (GPU; the model is 8.2 GB bf16):

    systemd-run --user --scope -p MemoryMax=24G -p MemorySwapMax=0 \\
        /home/qt/genai/comfyui/nvenv/bin/python tools/gen_mageflow_render.py \\
        --out scratch_out/mageflow_comfy
"""

import argparse
import os
import sys

COMFY = "/home/qt/genai/comfyui"
DEFAULT_DIT = os.path.join(COMFY, "models/diffusion_models/mageflow/mageFlow_mageFlow4B.safetensors")
DEFAULT_TE = os.path.join(COMFY, "models/text_encoders/qwen3VLInstruct4bHeretic_v10.safetensors")
DEFAULT_VAE = os.path.join(COMFY, "models/vae/mage_flow_vae_bf16.safetensors")

OUR_ARGV = [a for a in sys.argv[1:] if a not in ("--fp32-unet", "--cpu")]
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
import comfy.utils  # noqa: E402
import comfy.model_management  # noqa: E402
import nodes  # noqa: E402
import node_helpers  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dit", default=DEFAULT_DIT)
    ap.add_argument("--text-encoder", default=DEFAULT_TE)
    ap.add_argument("--vae", default=DEFAULT_VAE)
    ap.add_argument("--prompt", default="a red apple on a wooden table")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shift", type=float, default=6.0)
    ap.add_argument("--ref-image", action="append", default=[],
                    help="Mage-Flow-Edit reference picture; repeat for more than one")
    ap.add_argument("--rounds", type=int, default=1, help="repeat the render; the LAST is reported")
    ap.add_argument("--out", default="/tmp/mageflow_comfy")
    args = ap.parse_args(OUR_ARGV)

    model = comfy.sd.load_diffusion_model(args.dit)
    # The family's own shift, set explicitly so a ComfyUI default change cannot
    # silently move the schedule this comparison depends on.
    ms = model.get_model_object("model_sampling")
    ms.set_parameters(shift=args.shift, multiplier=1.0)
    clip = comfy.sd.load_clip(ckpt_paths=[args.text_encoder], embedding_directory=None,
                              clip_type=comfy.sd.CLIPType.MAGE)
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(args.vae))

    # `TextEncodeMageFlowEdit`, inlined: the reference is resized TWICE, capped at
    # a 384 px long edge for the VL copy and taken to the RENDER's resolution for
    # the VAE copy, and the SAME references go on both conditioning branches.
    images_vl, ref_latents = [], []
    for path in args.ref_image:
        arr = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
        img = torch.from_numpy(arr)[None]
        samples = img.movedim(-1, 1)
        long_edge = max(samples.shape[3], samples.shape[2])
        if long_edge > 384:
            by = 384 / long_edge
            images_vl.append(comfy.utils.common_upscale(
                samples, max(1, round(samples.shape[3] * by)), max(1, round(samples.shape[2] * by)),
                "bicubic", "disabled").movedim(1, -1))
        else:
            images_vl.append(img)
        s = samples if (samples.shape[3] == args.width and samples.shape[2] == args.height) else \
            comfy.utils.common_upscale(samples, args.width, args.height, "bicubic", "disabled")
        ref_latents.append(vae.encode(s.movedim(1, -1)[:, :, :, :3]))

    positive = clip.encode_from_tokens_scheduled(clip.tokenize(args.prompt, images=images_vl))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(" " if images_vl else "", images=images_vl))
    if ref_latents:
        positive = node_helpers.conditioning_set_values(positive, {"reference_latents": ref_latents}, append=True)
        negative = node_helpers.conditioning_set_values(negative, {"reference_latents": ref_latents}, append=True)

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
