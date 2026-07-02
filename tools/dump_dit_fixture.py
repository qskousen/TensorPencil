"""Dump a Krea 2 DiT forward-pass parity fixture via ComfyUI.

Uses the text conditioning from dump_text_fixture.py as context, a seeded
16x16x16 latent, sigma 0.875. Weights stay fp8 (manual-cast to f32 per layer,
matching the engine's in-kernel dequant exactly).

Writes:
  testdata/dit_x.bin    f32 [16][16][16] input latent (planar)
  testdata/dit_out.bin  f32 [16][16][16] predicted velocity (planar)

Run: cd ~/genai/comfyui && venv/bin/python /dump/projects/zig/TensorPencil/tools/dump_dit_fixture.py
"""
import sys, types
import numpy as np
import torch

sys.path.insert(0, "/home/qt/genai/comfyui")
sys.argv = [sys.argv[0]]
_a = types.ModuleType("comfy_aimdo"); sys.modules["comfy_aimdo"] = _a
for s in ("host_buffer", "model_vbar", "torch", "vram_buffer"):
    m = types.ModuleType(f"comfy_aimdo.{s}"); sys.modules[f"comfy_aimdo.{s}"] = m; setattr(_a, s, m)

import comfy.cli_args
comfy.cli_args.args.cpu = True

import comfy.sd
import comfy.utils
import comfy.model_management

# Without comfy_kitchen, quant_ops.ck is unavailable; the in_training flag
# routes flux-style rope/attention through the pure-torch reference path.
comfy.model_management.in_training = True

DIT_PATH = "/dump/projects/zig/TensorPencil/models/diffusion_model/krea2CenterSemiraw_v10Fp8.safetensors"
OUT_DIR = "/dump/projects/zig/TensorPencil/testdata"

sd = comfy.utils.load_torch_file(DIT_PATH)
patcher = comfy.sd.load_diffusion_model_state_dict(sd, model_options={"dtype": torch.float8_e4m3fn})
dit = patcher.model.diffusion_model.eval()
print("model:", type(dit).__name__, "compute dtype:", patcher.model.manual_cast_dtype or patcher.model.get_dtype())

torch.manual_seed(11)
x = torch.randn(1, 16, 16, 16, dtype=torch.float32)
ctx = torch.from_numpy(np.fromfile(f"{OUT_DIR}/text_cond.bin", dtype=np.float32).reshape(1, 14, 12 * 2560))
t = torch.tensor([0.875], dtype=torch.float32)

with torch.no_grad():
    out = dit(x, timesteps=t, context=ctx)
print("out:", tuple(out.shape), "range", out.min().item(), out.max().item())

x[0].numpy().tofile(f"{OUT_DIR}/dit_x.bin")
out[0].float().numpy().tofile(f"{OUT_DIR}/dit_out.bin")
print("fixtures written")
