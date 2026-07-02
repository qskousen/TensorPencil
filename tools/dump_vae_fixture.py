"""Dump a Wan VAE decode parity fixture using ComfyUI's reference implementation.

Run from the ComfyUI directory with its venv:
  cd ~/genai/comfyui && venv/bin/python /dump/projects/zig/TensorPencil/tools/dump_vae_fixture.py
"""
import sys, types, torch
sys.path.insert(0, "/home/qt/genai/comfyui")
sys.argv = [sys.argv[0]]  # keep comfy's arg parser happy

# This ComfyUI build imports the (absent) comfy_aimdo accelerator package at
# module scope but only calls it behind aimdo_enabled=False guards — stub it.
_aimdo = types.ModuleType("comfy_aimdo")
sys.modules["comfy_aimdo"] = _aimdo
for sub in ("host_buffer", "model_vbar", "torch", "vram_buffer"):
    mod = types.ModuleType(f"comfy_aimdo.{sub}")
    sys.modules[f"comfy_aimdo.{sub}"] = mod
    setattr(_aimdo, sub, mod)

import comfy.sd
import comfy.utils

VAE_PATH = "/dump/projects/zig/TensorPencil/models/vae/krea2RealVae_v10.safetensors"
OUT_DIR = "/dump/projects/zig/TensorPencil/testdata"

sd = comfy.utils.load_torch_file(VAE_PATH)
vae = comfy.sd.VAE(sd=sd, dtype=torch.float32)
model = vae.first_stage_model.float().eval()

torch.manual_seed(7)
z = torch.randn(1, 16, 1, 8, 8, dtype=torch.float32)
with torch.no_grad():
    out = model.decode(z)  # [1, 3, 1, 64, 64], roughly [-1, 1]
print("decoded:", tuple(out.shape), "range", out.min().item(), out.max().item())

z[0, :, 0].numpy().tofile(f"{OUT_DIR}/vae_z_8x8.bin")          # [16][8][8] f32
out[0, :, 0].contiguous().numpy().tofile(f"{OUT_DIR}/vae_rgb_64.bin")  # [3][64][64] f32
print("fixtures written")
