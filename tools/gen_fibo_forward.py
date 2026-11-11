#!/usr/bin/env python3
"""Pin Fibo 1.5's DiT forward by EXECUTING diffusers' `BriaFiboTransformer2DModel` at a
toy width. ComfyUI has no model for it; the ComfyUI_Fibo-1.5 custom nodes run this one.

Seeded f32 weights, run on the CPU, with the weights, inputs and velocity recorded. Every
convention the checkpoint does not record runs as at the real width: `[text | image]`
order in the joint attention, text at rope position zero and the image on (0, h, w),
the per-block caption projection that REPLACES the second half of the text stream's
channels before every block (double and single), AdaLayerNormZero's (shift, scale, gate)
chunk order against AdaLayerNormContinuous's (scale, shift), tanh-GELU in both FFNs and
the single blocks' parallel MLP, and the timestep embedded at `sigma * 1000` with cos first.

Two heads of 32 (the real model has 24 of 128), so the per-head QK norm is not a
whole-row norm and each rope axis is real. Weights are rounded to bf16 before the
reference runs and stored as bf16 under diffusers' names.

Emits src/models/assets/fibo_forward.safetensors, read by src/models/fibo.zig.

Usage:
    /home/qt/genai/comfyui/nvenv/bin/python tools/gen_fibo_forward.py
"""

import os

import torch
import safetensors.torch
from diffusers.models.transformers.transformer_bria_fibo import BriaFiboTransformer2DModel

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "models", "assets", "fibo_forward.safetensors")

CFG = dict(patch_size=1, in_channels=16, num_layers=2, num_single_layers=2, attention_head_dim=32, num_attention_heads=2,
           joint_attention_dim=24, axes_dims_rope=[8, 12, 12], rope_theta=10000, time_theta=10000, text_encoder_dim=20)

# (name, latent h, latent w, text len, sigma)
CASES = [("a", 4, 6, 5, 0.8), ("b", 3, 5, 9, 0.3)]


def main():
    torch.manual_seed(7)
    model = BriaFiboTransformer2DModel(**CFG).float().eval()
    gen = torch.Generator().manual_seed(5)
    sd = model.state_dict()
    for k, v in sd.items():
        if v.ndim == 1 and "norm" in k:
            w = 1.0 + 0.2 * torch.randn(v.shape, generator=gen)
        elif k.endswith(".bias"):
            w = 0.05 * torch.randn(v.shape, generator=gen)
        else:
            w = torch.randn(v.shape, generator=gen) / v.shape[-1] ** 0.5
        sd[k] = w.to(torch.bfloat16).float()
    model.load_state_dict(sd)

    n_blocks = CFG["num_layers"] + CFG["num_single_layers"]
    out = {k: v.to(torch.bfloat16).contiguous() for k, v in sd.items()}
    for name, h, w, n_txt, sigma in CASES:
        g = torch.Generator().manual_seed(ord(name))
        x = torch.randn(1, CFG["in_channels"], h, w, generator=g)
        ehs = torch.randn(1, n_txt, CFG["joint_attention_dim"], generator=g)
        layers = [torch.randn(1, n_txt, CFG["text_encoder_dim"], generator=g) for _ in range(n_blocks)]
        img_ids = torch.zeros(h, w, 3)
        img_ids[..., 1] = torch.arange(h)[:, None]
        img_ids[..., 2] = torch.arange(w)[None, :]
        img_ids = img_ids.reshape(h * w, 3)
        txt_ids = torch.zeros(n_txt, 3)
        tokens = x.permute(0, 2, 3, 1).reshape(1, h * w, CFG["in_channels"])
        t = torch.tensor([sigma * 1000.0])
        with torch.no_grad():
            v = model(hidden_states=tokens, encoder_hidden_states=ehs, text_encoder_layers=layers, timestep=t,
                      img_ids=img_ids, txt_ids=txt_ids, return_dict=False)[0]
        v = v.reshape(1, h, w, CFG["in_channels"]).permute(0, 3, 1, 2)
        out[f"case.{name}.x"] = x[0].contiguous()
        out[f"case.{name}.ehs"] = ehs[0].contiguous()
        out[f"case.{name}.layers"] = torch.stack([l[0] for l in layers]).contiguous()
        out[f"case.{name}.sigma"] = torch.tensor([sigma])
        out[f"case.{name}.out"] = v[0].contiguous()
        print(name, "out rms", v.pow(2).mean().sqrt().item())
    safetensors.torch.save_file(out, OUT)
    print("wrote", OUT, os.path.getsize(OUT))


if __name__ == "__main__":
    main()
