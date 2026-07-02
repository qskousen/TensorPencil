"""Dump Krea 2 text-conditioning parity fixtures via ComfyUI (f32).

Writes:
  testdata/text_ids.bin   u32 LE token ids of the full templated prompt
  testdata/text_cond.bin  f32 [seq_stripped][12][2560] conditioning stack

Run from the ComfyUI directory with its venv:
  cd ~/genai/comfyui && venv/bin/python /dump/projects/zig/TensorPencil/tools/dump_text_fixture.py
"""
import sys, types
import numpy as np
import torch

sys.path.insert(0, "/home/qt/genai/comfyui")
sys.argv = [sys.argv[0]]
_a = types.ModuleType("comfy_aimdo"); sys.modules["comfy_aimdo"] = _a
for s in ("host_buffer", "model_vbar", "torch", "vram_buffer"):
    m = types.ModuleType(f"comfy_aimdo.{s}"); sys.modules[f"comfy_aimdo.{s}"] = m; setattr(_a, s, m)

# Imported-as-library ComfyUI never parses argv, so force CPU on the args
# object directly (the iGPU's ROCm arch is unsupported by this torch build).
import comfy.cli_args
comfy.cli_args.args.cpu = True

import comfy.sd
import comfy.utils

TE_PATH = "/dump/projects/zig/TensorPencil/models/text_encoders/qwen3VLInstruct4bHeretic_v10.safetensors"
OUT_DIR = "/dump/projects/zig/TensorPencil/testdata"
PROMPT = "a fluffy orange cat sitting on a windowsill"

# This ComfyUI build needs comfy_kitchen for its fp8 layouts; sidestep by
# dequantizing to plain f32 (weight.float() * weight_scale) before loading —
# numerically identical to what the engine does inside its GEMM.
sd = comfy.utils.load_torch_file(TE_PATH)
deq = {}
for k, v in sd.items():
    if k.endswith(".comfy_quant") or k.endswith(".weight_scale"):
        continue
    if v.dtype == torch.float8_e4m3fn:
        scale = sd.get(k + "_scale")
        w = v.float()
        if scale is not None:
            w = w * scale.float()
        deq[k] = w
    else:
        deq[k] = v.float()

import comfy.text_encoders.krea2
import comfy.sd1_clip

clip = comfy.sd.load_text_encoder_state_dicts(
    [deq],
    clip_type=comfy.sd.CLIPType.KREA2,
    model_options={"dtype": torch.float32},
)

tokens = clip.tokenize(PROMPT)
ids = np.array([t[0] for t in tokens["qwen3vl_4b"][0]], dtype=np.uint32)
ids.tofile(f"{OUT_DIR}/text_ids.bin")
print("ids:", ids.tolist())

out, _pooled, extra = clip.cond_stage_model.encode_token_weights(tokens)
print("cond:", tuple(out.shape), out.dtype, "attn mask:", "attention_mask" in extra)
# (B, seq, 12*2560) -> [seq][12][2560]
out[0].float().numpy().astype(np.float32).tofile(f"{OUT_DIR}/text_cond.bin")
print("fixtures written")
