#!/usr/bin/env python3
"""Pin Fibo 2.0's DiT forward by EXECUTING ComfyUI's own model at a toy width.

`comfy/ldm/fibo2/model.py` at a small width with seeded f32 weights, run on the CPU,
with the weights, inputs, stage outputs and velocity recorded. Every convention the
checkpoint does not record runs exactly as at the real width: the gist count that
follows the text length (256 up to 256 tokens, then growing in steps of 64), the gist
rope planes 1..n and the image on plane max_gist + 1, the t = 0 modulation of the gist
rows against the sampled t of the image rows, the context split into one bundle for the
Perceiver and one per injection block, the tanh'd scalar cross-attention gate, and the
fused w13 with the gate half first.

Two heads of 128 (the rope axes are hard-wired to 32 + 48 + 48), so the per-head QK norm
is not a whole-row norm. Weights are rounded to bf16 BEFORE the reference runs and are
stored as bf16 under ComfyUI's fused names (`qkv`, `to_kv`, `w13`), which is what
Comfy-Org's repack ships; the Zig test derives the split and diffusers spellings from
them.

Emits src/models/assets/fibo2_forward.safetensors, read by src/models/fibo2.zig.

Usage (ComfyUI at or after fca60fe, which added Fibo 2.0):
    COMFY=/path/to/comfyui /home/qt/genai/comfyui/nvenv/bin/python tools/gen_fibo2_forward.py
"""

import os
import sys

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "models", "assets", "fibo2_forward.safetensors",
)

COMFY = os.environ.get("COMFY", "/home/qt/genai/comfyui")
sys.argv = [sys.argv[0], "--cpu", "--disable-smart-memory", "--disable-xformers", "--use-pytorch-cross-attention"]
sys.path.insert(0, COMFY)
os.chdir(COMFY)

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import torch  # noqa: E402
import safetensors.torch  # noqa: E402

import comfy.ops  # noqa: E402
import comfy.ldm.fibo2.model as fibo2  # noqa: E402
import comfy.ldm.lumina.model as lumina  # noqa: E402
from comfy.ldm.modules.attention import AttentionTensorContainer, attention_pytorch  # noqa: E402



def _cpu_attention(q, k, v, heads, mask=None, skip_reshape=False, **_kw):
    q, k, v = (t.take() if isinstance(t, AttentionTensorContainer) else t for t in (q, k, v))
    return attention_pytorch(q, k, v, heads, mask=mask, skip_reshape=skip_reshape)


# Both modules bind their attention by value at import; rebind to the CPU fp32 kernel.
fibo2.optimized_attention = _cpu_attention
lumina.optimized_attention_masked = _cpu_attention

CFG = dict(in_channels=32, dim=256, n_heads=2, n_layers=3, n_refiner_layers=1, injection_layers=(0, 2),
           cap_feat_dim=48, perceiver_layers=1, max_gist_tokens=512, ffn_dim=96, fused_ffn=True, norm_eps=1e-5)

# (name, latent h, latent w, text len, sigma). 600 tokens takes the gist count off its
# floor of 256 (to 320); 40 sits on it.
CASES = [
    ("a", 4, 6, 40, 0.7),
    ("b", 5, 3, 600, 0.25),
]


def main():
    torch.manual_seed(1234)
    model = fibo2.Fibo2(**CFG, dtype=torch.float32, device="cpu", operations=comfy.ops.disable_weight_init)
    sd = model.state_dict()
    gen = torch.Generator().manual_seed(99)
    for k, v in sd.items():
        if v.ndim == 1 and "norm" in k:
            w = 1.0 + 0.2 * torch.randn(v.shape, generator=gen)
        elif k.endswith("cross_attn_gate"):
            w = 0.8 * torch.randn(v.shape, generator=gen)
        elif k.endswith(".bias"):
            w = 0.05 * torch.randn(v.shape, generator=gen)
        elif k.endswith("latent_queries"):
            w = torch.randn(v.shape, generator=gen)
        else:
            fan_in = v.shape[-1] if v.ndim > 1 else 1
            w = torch.randn(v.shape, generator=gen) / fan_in ** 0.5
        sd[k] = w.to(torch.bfloat16).float()
    # One trunk block's q/k and one injection block's k at a mean square near the norms'
    # epsilon, so the eps (1e-5 here, not float32's as in Z-Image) changes the answer.
    for k in ("layers.1.attention.qkv.weight", "layers.0.cross_attention.to_kv.weight"):
        sd[k] = (sd[k] * 0.003).to(torch.bfloat16).float()
    # The fused FFN's hidden width is zero padded in real files; pad the last 16 rows
    # of each half here so a loader that reads the padding as data would be caught.
    pad = 16
    for k in list(sd):
        if k.endswith("w13.weight"):
            h = CFG["ffn_dim"]
            sd[k][h - pad:h] = 0
            sd[k][2 * h - pad:] = 0
        elif k.endswith("w2.weight"):
            sd[k][:, CFG["ffn_dim"] - pad:] = 0
    model.load_state_dict(sd)
    model.eval()

    acts = {}

    def hook(name):
        def f(_m, _i, o):
            acts[name] = o.detach().clone()
        return f

    model.t_embedder.register_forward_hook(hook("t_mixed"))
    model.noise_refiner[-1].register_forward_hook(hook("refiner"))
    model.perceiver.register_forward_hook(hook("gist"))
    model.layers[-1].register_forward_hook(hook("trunk"))

    out = {k: v.to(torch.bfloat16).contiguous() for k, v in sd.items()}
    out["cfg.ffn_pad"] = torch.tensor([pad], dtype=torch.int32)
    for name, h, w, n_txt, sigma in CASES:
        g = torch.Generator().manual_seed(hash(name) & 0xFFFF)
        x = torch.randn(1, CFG["in_channels"], h, w, generator=g)
        ctx = torch.randn(1, n_txt, CFG["cap_feat_dim"] * (1 + len(CFG["injection_layers"])), generator=g)
        t = torch.tensor([sigma], dtype=torch.float32)
        with torch.no_grad():
            v = model(x, t, ctx)
        out[f"case.{name}.x"] = x[0].contiguous()
        out[f"case.{name}.ctx"] = ctx[0].contiguous()
        out[f"case.{name}.sigma"] = t
        out[f"case.{name}.t_mixed"] = acts["t_mixed"].contiguous()
        out[f"case.{name}.refiner"] = acts["refiner"][0].contiguous()
        out[f"case.{name}.gist"] = acts["gist"][0].contiguous()
        out[f"case.{name}.trunk"] = acts["trunk"][0].contiguous()
        out[f"case.{name}.out"] = v[0].contiguous()
        print(name, "gist", acts["gist"].shape[1], "out rms", v.pow(2).mean().sqrt().item())
    safetensors.torch.save_file(out, OUT)
    print("wrote", OUT, os.path.getsize(OUT), "bytes")


if __name__ == "__main__":
    main()
