#!/usr/bin/env python3
"""Pin Qwen-Image 2.1's DiT forward by EXECUTING ComfyUI's own model at a toy width.

`comfy/ldm/qwen_image21/model.py` with seeded f32 weights on the CPU, the weights,
inputs and outputs recorded. Conventions it pins, each a silent wrong answer:
`[text | image]` with text causal and the image attending to everything (block
causal), the text prefix modulated at t = 0 and the image at the sampled t, an
unaffine LayerNorm times (1 + scale) with tanh'd gates and no shift, RoPE on
ADJACENT pairs over three axes with text at (p, p, p) and the image centred, the
SwiGLU's fused `[gate | up]` order, the zero-centred RMSNorm on the text input, and
the final layer reading only the sampled-t row.

The `ref` case splices one reference latent into the text at a slot, which is the
edit path: its tokens sit inside the causal prefix as one block, at t = 0, and its
positions are centred on the TARGET's parity.

Sigmas are 0.25 and 0.125, exact in bf16 both as t and as 1000 t, so the reference's
rounding of the timestep to its compute dtype is a no-op here and the Zig side's
deliberate bf16 rounding agrees with it.

Two 64-wide heads with RoPE axes (8, 28, 28); the real model has 32 of 128 with
(16, 56, 56). Weights are rounded to bf16 before the reference runs and stored bf16,
written fused (`gate_up`) under `fused.` and split (`proj` = up, `gate_layer`) under
`split.`.

Emits src/models/assets/qwen_image21_forward.safetensors.

Usage:
    /home/qt/genai/comfyui/nvenv/bin/python tools/gen_qwen_image21_forward.py
"""

import os
import sys

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "models", "assets", "qwen_image21_forward.safetensors",
)

COMFY = "/home/qt/genai/comfyui"
sys.argv = [sys.argv[0], "--cpu", "--disable-smart-memory", "--disable-xformers", "--use-pytorch-cross-attention"]
sys.path.insert(0, COMFY)
os.chdir(COMFY)

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import torch  # noqa: E402
import safetensors.torch  # noqa: E402

import comfy.ops  # noqa: E402
import comfy.ldm.qwen_image21.model as qi  # noqa: E402
from comfy.ldm.modules.attention import attention_pytorch  # noqa: E402

# Bound by value at import; the CPU fp32 kernel is what a fixture wants.
qi.optimized_attention = attention_pytorch

CFG = dict(in_channels=16, out_channels=16, num_layers=2, attention_head_dim=64, num_attention_heads=2,
           context_in_dim=48, mlp_ratio=3, axes_dims_rope=(8, 28, 28))

# (name, latent h, latent w, text len, sigma, reference (h, w) or None, slot)
CASES = [
    ("even", 4, 6, 7, 0.25, None, 0),
    ("odd", 5, 3, 4, 0.125, None, 0),
    ("ref", 4, 6, 6, 0.25, (3, 5), 2),
]


def seeded_weights(model):
    sd = {}
    for name, p in model.named_parameters():
        if name.endswith("text_norm.weight"):
            t = torch.randn(p.shape) * 0.1  # zero-centred: the scale is 1 + w
        elif name.endswith("norm_q.weight") or name.endswith("norm_k.weight"):
            t = 1.0 + torch.randn(p.shape) * 0.1
        elif "modulation" in name or "norm_out.linear" in name:
            t = torch.randn(p.shape) * (0.1 / p.shape[1] ** 0.5)
        elif p.dim() == 2:
            t = torch.randn(p.shape) / (p.shape[1] ** 0.5)
        else:
            t = torch.randn(p.shape) * 0.1
        sd[name] = t.to(torch.bfloat16).float()
    return sd


def main():
    torch.manual_seed(20261019)
    model = qi.QwenImage21Transformer2DModel(**CFG, fused_mlp=True, dtype=torch.float32, device="cpu",
                                              operations=comfy.ops.disable_weight_init)
    model.eval()
    model.load_state_dict(seeded_weights(model), strict=True)
    for prm in model.parameters():
        prm.requires_grad_(False)

    out = {}
    for name, t in model.state_dict().items():
        t = t.to(torch.bfloat16)
        out["fused." + name] = t.clone()
        if name.endswith("img_mlp.gate_up.weight"):
            gate, up = t.chunk(2, dim=0)
            base = name[: -len("gate_up.weight")]
            out["split." + base + "gate_layer.weight"] = gate.clone()
            out["split." + base + "proj.weight"] = up.clone()
        else:
            out["split." + name] = t.clone()

    C = CFG["in_channels"]
    with torch.inference_mode():
        for case, h, w, n_txt, sigma, ref, slot in CASES:
            x = torch.randn(1, C, h, w)
            ctx = torch.randn(1, n_txt, CFG["context_in_dim"])
            ts = torch.tensor([sigma], dtype=torch.float32)
            kwargs = {}
            if ref is not None:
                r = torch.randn(1, C, *ref)
                kwargs = dict(ref_latents=[r], image_slots=[slot])
                out[f"{case}.ref0"] = r[0].contiguous()
                out[f"{case}.slot0"] = torch.tensor([slot], dtype=torch.int32)
            y = model(x, ts, ctx, transformer_options={}, **kwargs)
            if not torch.isfinite(y).all():
                raise SystemExit(f"{case}: non-finite output")
            out[f"{case}.x"] = x[0].contiguous()
            out[f"{case}.ctx"] = ctx[0].contiguous()
            out[f"{case}.sigma"] = ts
            out[f"{case}.out"] = y[0].contiguous()
            print(f"{case}: {h}x{w}, text {n_txt}, ref {ref}, out rms {y.pow(2).mean().sqrt():.4f}")

    safetensors.torch.save_file(out, OUT)
    print("wrote", OUT, os.path.getsize(OUT) // 1024, "KiB")


if __name__ == "__main__":
    main()
