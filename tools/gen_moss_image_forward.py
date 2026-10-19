#!/usr/bin/env python3
"""Pin MOSS-Image's DiT forward by EXECUTING ComfyUI's own model at a toy width.

`comfy/ldm/moss_image/model.py` at a small width with seeded f32 weights, run on
the CPU, with the weights, inputs and outputs recorded. Every convention the
checkpoint does not record is exercised exactly as at the real width: image
tokens BEFORE text, the 4-axis (r, f, h, w) rope applied split-half with the
image on f = text_len and its rows/columns centered on (n - 1) / 2, one shared
modulation for every block with no SiLU in front of it, the [hidden | gate] half
order of `situ`, the head modulation's [shift, scale] order, and a text refiner
that rotates text by its own index.

Two 64-wide heads (the real model has 24 of 128), so the per-head QK norm is
not a whole-row norm and each of the four rope axes is 16 wide.

Weights are rounded to bf16 BEFORE the reference runs and stored as bf16, so the
fixture is half the size and still exact. They are written twice: fused (`qkv`,
what repacked files carry) under `fused.` and split (`q`, `k`, `v`, the original
release) under `split.`, so both loader spellings answer to the same output.

Emits src/models/assets/moss_image_forward.safetensors, read by the parity test
in src/models/moss_image.zig.

Usage:
    /home/qt/genai/comfyui/nvenv/bin/python tools/gen_moss_image_forward.py
"""

import os
import sys

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "models", "assets", "moss_image_forward.safetensors",
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
import comfy.ldm.moss_image.model as moss  # noqa: E402
from comfy.ldm.modules.attention import attention_pytorch  # noqa: E402

# The model module binds its attention by value at import; rebind it to the CPU
# fp32 kernel. This fixture is about the architecture, not the attention kernel.
moss.optimized_attention = attention_pytorch

CFG = dict(dim=128, ffn_dim=160, num_heads=2, num_layers=2, text_dim=48, latent_channels=16)

# (name, latent h, latent w, text len, sigma). Odd and even sizes both, since the
# centering is (n - 1) / 2: half-integer positions on even sizes.
CASES = [
    ("even", 4, 6, 7, 0.75),
    ("odd", 5, 3, 4, 0.2),
]


def seeded_weights(model):
    # Realistically distributed, which is what gives the fixture teeth: norm
    # scales centred on one, fan-in scaled matrices so situ's tanh caps and
    # sigmoid are actually nonlinear, modulation kept small so two blocks stay
    # bounded.
    sd = {}
    for name, p in model.named_parameters():
        if name.endswith("norm1.weight") or name.endswith("norm2.weight") or name.endswith("_norm.weight") or name.endswith("norm_video.weight"):
            t = 1.0 + torch.randn(p.shape) * 0.1
        elif "modulation" in name:
            t = torch.randn(p.shape) * (0.1 if p.dim() == 1 else 0.1 / p.shape[1] ** 0.5)
        elif name.endswith(".bias"):
            t = torch.randn(p.shape) * 0.1
        elif p.dim() == 2:
            t = torch.randn(p.shape) / (p.shape[1] ** 0.5)
        else:
            t = torch.randn(p.shape) * 0.1
        sd[name] = t.to(torch.bfloat16).float()
    return sd


def main():
    torch.manual_seed(20261019)
    ops = comfy.ops.disable_weight_init
    model = moss.MOSSImage(**CFG, fused_qkv=True, dtype=torch.float32, device="cpu", operations=ops)
    model.eval()
    sd = seeded_weights(model)
    model.load_state_dict(sd, strict=True)
    for prm in model.parameters():
        prm.requires_grad_(False)

    # Scale the FFN up/gate projections so situ's caps engage (|gate| > 4 and
    # |hidden| near 25 on some lanes): a cap dropped on either side moves the output.
    for i in range(CFG["num_layers"]):
        model.layers[i].proj["ffn_up_gate"].weight.mul_(8.0)  # exact in bf16

    out = {}
    for name, t in model.state_dict().items():
        t = t.to(torch.bfloat16)
        out["fused." + name] = t.clone()
        if name.endswith("proj.qkv.weight"):
            q, k, v = t.chunk(3, dim=0)
            base = name[: -len("qkv.weight")]
            out["split." + base + "q.weight"] = q.clone()
            out["split." + base + "k.weight"] = k.clone()
            out["split." + base + "v.weight"] = v.clone()
        else:
            out["split." + name] = t.clone()

    C = CFG["latent_channels"]
    with torch.inference_mode():
        for case, h, w, n_txt, sigma in CASES:
            x = torch.randn(1, C, h, w)
            ctx = torch.randn(1, n_txt, CFG["text_dim"])
            ts = torch.tensor([sigma], dtype=torch.float32)
            y = model(x, ts, ctx, transformer_options={})
            if not torch.isfinite(y).all():
                raise SystemExit(f"{case}: non-finite output")
            out[f"{case}.x"] = x[0].contiguous()
            out[f"{case}.ctx"] = ctx[0].contiguous()
            out[f"{case}.sigma"] = ts
            out[f"{case}.out"] = y[0].contiguous()
            print(f"{case}: {h}x{w}, text {n_txt}, out rms {y.pow(2).mean().sqrt():.4f}")

    safetensors.torch.save_file(out, OUT)
    print("wrote", OUT, os.path.getsize(OUT) // 1024, "KiB")


if __name__ == "__main__":
    main()
