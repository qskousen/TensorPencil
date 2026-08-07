#!/usr/bin/env python3
"""Render one Z-Image image through ComfyUI, for a like-for-like comparison with
TensorPencil's own render of the same seed/prompt/steps/sampler/scheduler.

Mirrors ComfyUI's own `image_z_image_turbo` template exactly:
    UNETLoader -> ModelSamplingAuraFlow(shift=3.0) -> KSampler(euler, simple, cfg 1)
    CLIPLoader(qwen_3_4b, type=lumina2) -> CLIPTextEncode
    EmptySD3LatentImage -> VAEDecode(ae.safetensors)

⚠️ CPU, and NOT fp32 by default: a fp32 copy of the 6.1B denoiser is 24.6 GB. At bf16
it is 12.3 GB, and the Qwen3-4B encoder is another 8 GB — both alive before either is
freed. The reference's own precision is then a floor under any comparison; `--dtype`
exists so that floor can itself be measured, by rendering twice at different precisions
and comparing those two against each other.

⚠️ **RUN IT UNDER A MEMORY BOUND.** Launching this unbounded on a loaded machine froze
the desktop rather than killing the process:

    systemd-run --user --scope -p MemoryMax=14G -p MemorySwapMax=0 -- \
        /home/qt/genai/comfyui/nvenv/bin/python tools/render_zimage_ref.py --out ref.png

Compare against TensorPencil's own render of the same settings:

    zig build run -Doptimize=ReleaseFast -- generate \
        --dit .../unstableRevolution_V2Fp16.safetensors --prompt "..." \
        --width 256 --height 256 --steps 8 --cfg 1 --seed 42 --backend cpu \
        --out scratch_out/ours.png
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
from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow  # noqa: E402
from PIL import Image  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dit", default=f"{COMFY}/models/checkpoints/zit/unstableRevolution_V2Fp16.safetensors")
ap.add_argument("--te", default=f"{COMFY}/models/text_encoders/qwen_3_4b.safetensors")
ap.add_argument("--vae", default=f"{COMFY}/models/vae/ae.safetensors")
ap.add_argument("--prompt", default="a photograph of an astronaut riding a horse")
ap.add_argument("--size", type=int, default=256)
ap.add_argument("--width", type=int, default=0, help="overrides --size")
ap.add_argument("--height", type=int, default=0, help="overrides --size")
ap.add_argument("--prompt-file", default=None, help="read the prompt from a file (long prompts)")
ap.add_argument("--steps", type=int, default=8)
ap.add_argument("--cfg", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--shift", type=float, default=3.0)
ap.add_argument("--sampler", default="euler")
ap.add_argument("--scheduler", default="simple")
ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32", "fp16"])
ap.add_argument("--out", required=True)
a = ap.parse_args(OUR)

DT = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}[a.dtype]

print(f"loading denoiser ({a.dtype})...")
model = comfy.sd.load_diffusion_model(a.dit, model_options={"dtype": DT})
model = ModelSamplingAuraFlow().patch_aura(model, a.shift)[0]

print("loading text encoder...")
clip = comfy.sd.load_clip(ckpt_paths=[a.te], embedding_directory=None,
                          clip_type=comfy.sd.CLIPType.LUMINA2,
                          model_options={"dtype": DT})
prompt_text = open(a.prompt_file).read() if a.prompt_file else a.prompt
tokens = clip.tokenize(prompt_text)
cond = clip.encode_from_tokens_scheduled(tokens)
# cfg 1.0 means ComfyUI never evaluates the uncond branch, but the API still wants one.
neg = clip.encode_from_tokens_scheduled(clip.tokenize(""))
del clip
comfy.model_management.cleanup_models()

W = a.width or a.size
H = a.height or a.size
latent = {"samples": torch.zeros([1, 16, H // 8, W // 8])}
# Exactly `nodes.common_ksampler`'s noise: `comfy.sample.prepare_noise` seeds a CPU
# MT19937 generator, which is what `sampler.fillNoiseFrom(.torch_cpu)` reproduces.
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
# ⚠️ No `process_out` here: `comfy.sample.sample` already applied it (`z / 0.3611 +
# 0.1159`), which is exactly what `pipeline.Session.decode` does before its decoder.
vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(a.vae))
dec = vae.first_stage_model.float().eval().cpu()
with torch.no_grad():
    img = dec.decode(samples.float().cpu())

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
