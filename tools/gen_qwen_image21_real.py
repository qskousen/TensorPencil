#!/usr/bin/env python3
"""One forward of the REAL Qwen-Image 2.1 DiT, by executing ComfyUI's own model on
the CPU: bf16 weights upcast per op to fp32, the way ComfyUI runs a bf16 file on a
device without bf16 compute.

The toy fixture (`gen_qwen_image21_forward.py`) pins the conventions at a width
where nothing is expensive; this pins what only the released shapes exercise:
128-wide heads, RoPE axes (16, 56, 56), 32 blocks of real modulation.

The conditioning is `cond.0` from `qwen_image21_te_ref.safetensors` (a real prompt
through the real encoder); the latent is seeded noise at sigma 0.25, which is exact
in bf16 both as t and as 1000 t.

Emits src/models/assets/qwen_image21_real.safetensors.

Usage:
    systemd-run --user --scope -p MemoryMax=40G -p MemorySwapMax=0 \\
        /home/qt/genai/comfyui/nvenv/bin/python tools/gen_qwen_image21_real.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIT = "/mnt/watch/models/diffusion/diffusion_models/qwen2.1/qwen_image_2.1_turbo_bf16.safetensors"
TE_REF = os.path.join(ROOT, "src", "models", "assets", "qwen_image21_te_ref.safetensors")
OUT = os.path.join(ROOT, "src", "models", "assets", "qwen_image21_real.safetensors")
H, W, SIGMA = 6, 10, 0.25

sys.argv = [sys.argv[0], "--cpu", "--disable-smart-memory", "--disable-xformers", "--use-pytorch-cross-attention"]
sys.path.insert(0, "/home/qt/genai/comfyui")
os.chdir("/home/qt/genai/comfyui")
import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()
import torch  # noqa: E402
import safetensors.torch  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.ldm.qwen_image21.model as qi  # noqa: E402
from comfy.ldm.modules.attention import attention_pytorch  # noqa: E402

qi.optimized_attention = attention_pytorch

torch.manual_seed(20261019)
patcher = comfy.sd.load_diffusion_model(DIT, model_options={"dtype": torch.bfloat16})
model = patcher.model
if model.manual_cast_dtype != torch.float32:
    raise SystemExit(f"compute dtype {model.manual_cast_dtype}, wanted fp32")
dm = model.diffusion_model
ctx = safetensors.torch.load_file(TE_REF)["cond.0"].float()

with torch.inference_mode():
    x = torch.randn(1, 64, H, W)
    ts = torch.tensor([SIGMA], dtype=torch.float32)
    y = dm(x, ts, ctx[None], transformer_options={}).float()
    if not torch.isfinite(y).all():
        raise SystemExit("non-finite output")
    print(f"out {tuple(y.shape)}, rms {y.pow(2).mean().sqrt():.4f}")

safetensors.torch.save_file({"x": x[0].contiguous(), "sigma": ts, "out": y[0].contiguous()}, OUT)
print("wrote", OUT, os.path.getsize(OUT) // 1024, "KiB")
