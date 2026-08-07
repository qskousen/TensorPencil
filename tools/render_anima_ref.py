#!/usr/bin/env python3
"""Render one Anima image through ComfyUI, for a like-for-like comparison with
TensorPencil's own render of the same seed/prompt/steps/cfg/sampler/scheduler.

Mirrors the graph in the reference PNG's embedded `prompt`, which is also ComfyUI's
own `image_anima_base_v1` template minus its turbo LoRA:

    UNETLoader -> KSampler(euler, <scheduler>, cfg)
    CLIPLoader(qwen_3_06b_base, type=stable_diffusion) -> CLIPTextEncode
    EmptyLatentImage -> VAEDecode(qwen_image_vae)

⚠️ **No `ModelSamplingAuraFlow` patch, unlike the Z-Image script.** Anima's shift is
3.0 already, from `supported_models.Anima.sampling_settings`; patching it would apply
the flux-form shift on top of the model's own discrete-flow table and change every
sigma. This is also why `--shift` is not an argument here — there is nothing to set.

⚠️ **`type=stable_diffusion` on the CLIPLoader**, which is what selects
`AnimaTEModel`. It is not a Lumina/Qwen-image type despite the encoder being a Qwen3.

⚠️ The decoder is the **Wan 2.1** VAE (`qwen_image_vae.safetensors`), byte-for-byte the
`first_stage_model.*` the checkpoint bundles — so `--vae` and the checkpoint's own copy
are the same weights, and `latent_format` is `Wan21`: `process_out` is
`z * latents_std + latents_mean` per channel, NOT Flux's `z / scale + shift`.

⚠️ **RUN IT UNDER A MEMORY BOUND.** Launching an unbounded multi-GB model load on a
loaded machine froze the desktop rather than killing the process:

    systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 -- \
        /home/qt/genai/comfyui/nvenv/bin/python tools/render_anima_ref.py --out ref.png

`--dtype` exists so the reference's OWN precision floor can be measured, by rendering
twice at different precisions and comparing those two to each other. Without that
control a dB figure against this render is uninterpretable.
"""
import argparse, os, sys

COMFY = "/home/qt/genai/comfyui"
OUR = [a for a in sys.argv[1:] if a != "--gpu"]
# ⚠️ CPU by default so the reference cannot depend on a GPU reduction order. `--gpu`
# is for when the card is free: far faster, and at bf16 the dtype is the same either
# way, so the only difference is the reduction order (and xformers, still disabled).
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
if _dev and not comfy_args.cpu:
    raise SystemExit("ComfyUI did not take --cpu; check enable_args_parsing()")

import comfy.sd  # noqa: E402
import comfy.sample  # noqa: E402
import comfy.samplers  # noqa: E402
import comfy.model_management  # noqa: E402
import latent_preview  # noqa: E402
from PIL import Image  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dit", default=f"{COMFY}/models/diffusion_models/anima/terraRising_20TerraRisingAnima.safetensors")
ap.add_argument("--te", default=f"{COMFY}/models/text_encoders/qwen_3_06b_base.safetensors")
ap.add_argument("--vae", default=f"{COMFY}/models/vae/qwen_image_vae.safetensors")
ap.add_argument("--prompt", default="1girl, black hair, (black theme)")
ap.add_argument("--negative", default="")
ap.add_argument("--negative-file", default=None)
ap.add_argument("--size", type=int, default=256)
ap.add_argument("--width", type=int, default=0, help="overrides --size")
ap.add_argument("--height", type=int, default=0, help="overrides --size")
ap.add_argument("--prompt-file", default=None, help="read the prompt from a file (long prompts)")
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--cfg", type=float, default=4.0)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--sampler", default="euler")
ap.add_argument("--scheduler", default="simple")
ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32", "fp16"])
ap.add_argument("--out", required=True)
a = ap.parse_args(OUR)

DT = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}[a.dtype]

print(f"loading denoiser ({a.dtype})...")
# No sampling patch: Anima's `sampling_settings` already carries shift 3.0 and
# multiplier 1.0. See the module docstring.
model = comfy.sd.load_diffusion_model(a.dit, model_options={"dtype": DT})

print("loading text encoder...")
clip = comfy.sd.load_clip(ckpt_paths=[a.te], embedding_directory=None,
                          clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION,
                          model_options={"dtype": DT})
prompt_text = open(a.prompt_file).read() if a.prompt_file else a.prompt
tokens = clip.tokenize(prompt_text)
cond = clip.encode_from_tokens_scheduled(tokens)
# ⚠️ A REAL negative, unlike the Z-Image script: Anima's template runs cfg 4 and the
# reference render cfg 5, so the uncond branch is evaluated and its prompt matters.
neg_text = open(a.negative_file).read() if a.negative_file else a.negative
neg = clip.encode_from_tokens_scheduled(clip.tokenize(neg_text))
del clip
comfy.model_management.cleanup_models()

W = a.width or a.size
H = a.height or a.size
# ⚠️ Through ComfyUI's own `fix_empty_latent_channels`, not hand-built: Anima's
# `latent_format` is `Wan21`, whose `latent_dimensions` is **3**, so the latent needs a
# temporal axis — `[1, 16, 1, H/8, W/8]`. `EmptyLatentImage` emits SD's 4-channel 4-D
# tensor and `nodes.common_ksampler` fixes both the channel count and the rank here.
# Skipping it fails inside `prepare_embedded_sequence` with `IndexError: tuple index
# out of range`, which says nothing about the cause.
latent = {"samples": comfy.sample.fix_empty_latent_channels(
    model, torch.zeros([1, 4, H // 8, W // 8]))}
if latent["samples"].shape[1:3] != (16, 1):
    raise SystemExit(f"unexpected latent shape {tuple(latent['samples'].shape)}")
# Exactly `nodes.common_ksampler`'s noise: `comfy.sample.prepare_noise` seeds a CPU
# MT19937 generator, which is what `sampler.fillNoiseFrom(.torch_cpu)` reproduces.
# ⚠️ Drawn on the 5-D latent, so the element ORDER is the flat `[16][h][w]` order the
# Zig side fills — the temporal axis is length 1 and contributes no stride.
noise = comfy.sample.prepare_noise(latent["samples"], a.seed, None)

print(f"sampling {a.steps} steps at {W}x{H}...")
import time as _t
_t0 = _t.perf_counter()
samples = comfy.sample.sample(
    model, noise, a.steps, a.cfg, a.sampler, a.scheduler, cond, neg,
    latent["samples"], denoise=1.0, disable_noise=False, start_step=None,
    last_step=None, force_full_denoise=False, noise_mask=None, callback=None,
    disable_pbar=True, seed=a.seed)
_dt = _t.perf_counter() - _t0
print(f"TIMING sampling {_dt:.2f}s total, {_dt / a.steps:.3f} s/step (incl. warm-up)")
sig = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), a.scheduler, a.steps)
print("sigmas:", [round(float(v), 6) for v in sig])
# ⚠️ **NO `process_out` here, and getting this wrong cost a 15 dB "disagreement" that
# was entirely this harness.** `CFGGuider.sample` already ends with
# `self.inner_model.process_latent_out(samples)` (comfy/samplers.py), which for `Wan21`
# is `z * latents_std + latents_mean`. Applying it again denormalizes twice: the
# composition survives — same subject, same pose, same colours in the same places —
# and only the tone shifts, which reads as a numerics problem in the engine rather
# than as an error in the reference. The tell is exactly the one the Z-Image
# `process_output` trap left behind: structure matching while tone does not means the
# trajectory agreed and only the output mapping did not.
#
# So `samples` here is ALREADY in the decoder's input space, which is the same space
# `pipeline.Session.decode` hands its own decoder after its own denormalization.
del model
comfy.model_management.cleanup_models()

print("decoding...")
# ⚠️ The decoder is driven DIRECTLY, on the CPU in fp32, rather than through
# `VAE.decode`: the wrapper picks its own dtype from model_management and then hands
# bf16 samples to fp32 weights ("Input type (c10::BFloat16) and bias type (float)
# should be the same"). Driving it directly also matches the arm TensorPencil is being
# compared against — its own VAE is fp32 — and this decode is small enough that CPU
# costs nothing.
#
vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(a.vae))
dec = vae.first_stage_model.float().eval().cpu()
with torch.no_grad():
    # ⚠️ The Wan VAE is a 3D (video) decoder used on ONE frame, so the latent needs a
    # temporal axis: `[B, C, T=1, H, W]`. Its `decode` also takes a `feat_cache` in
    # some ComfyUI versions; `clear_cache` first so no state leaks between calls.
    z = samples.float().cpu()
    if z.dim() == 4:
        z = z.unsqueeze(2)
    if hasattr(dec, "clear_cache"):
        dec.clear_cache()
    img = dec.decode(z)
    if img.dim() == 5:
        img = img[:, :, 0]

# ⚠️ **`(x + 1) / 2` FIRST.** The LDM decoder emits roughly [-1, 1]; it is
# `VAE.process_output` — which driving `first_stage_model.decode` directly skips —
# that maps it to [0, 1]. Clamping the raw output instead crushes every negative to
# black, which looks like a plausible image with more contrast and saturation, not
# like an error. It cost a 14 dB "disagreement" that was entirely this line.
# `image.planarF32ToRgb8` on the Zig side does the same mapping.
rgb = ((img[0].float() + 1.0) / 2.0).clamp(0, 1)
# `[3, H, W]` -> `[H, W, 3]`; the decoder is channel-first, PIL is not.
arr = (rgb.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
Image.fromarray(arr).save(a.out)
print(f"wrote {a.out}")
