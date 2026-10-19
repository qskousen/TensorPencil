#!/usr/bin/env python3
"""Pin a LoRA on Qwen-Image 2.1 against ComfyUI's own LoRA path, end to end.

A toy 2.1 at the released head width (two heads of 128, so ComfyUI's RoPE axes
(16, 56, 56) apply as they do on the real model) goes through ComfyUI's detection
(`load_diffusion_model_state_dict`, which fuses the FFN) and its LoRA loader
(`load_lora_for_models`), which is where the 2.1 key map lives: a LoRA names the
FFN's two halves, `img_mlp.gate_layer` and `img_mlp.proj`, and ComfyUI slices them
out of the fused `gate_up`. The LoRA also hits `attn.to_q`, `attn.to_out.0`,
`img_mlp.out` and `proj_out`, with `lora_A`/`lora_B` and an `alpha` per target.

The keys are diffusers' bare spelling (`transformer_blocks...`), which ComfyUI maps in
full; ComfyUI drops the FFN halves under its own `diffusion_model.` spelling on a fused
checkpoint, so that one cannot be the reference here.

Records the toy weights (fused under `fused.`, split under `split.`), the LoRA under
`lora_bare.` and again under `lora_tf.` with diffusers' `transformer.` prefix, and the
outputs at strength 0 and 0.8.

Emits src/models/assets/qwen_image21_lora.safetensors.

Usage:
    /home/qt/genai/comfyui/nvenv/bin/python tools/gen_qwen_image21_lora.py
"""
import os
import sys

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "src", "models", "assets", "qwen_image21_lora.safetensors")

sys.argv = [sys.argv[0], "--cpu", "--disable-smart-memory", "--disable-xformers", "--use-pytorch-cross-attention"]
sys.path.insert(0, "/home/qt/genai/comfyui")
os.chdir("/home/qt/genai/comfyui")
import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()
import torch  # noqa: E402
import safetensors.torch  # noqa: E402
import comfy.ops  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.ldm.qwen_image21.model as qi  # noqa: E402
from comfy.ldm.modules.attention import attention_pytorch  # noqa: E402

qi.optimized_attention = attention_pytorch

CFG = dict(in_channels=16, out_channels=16, num_layers=2, attention_head_dim=128, num_attention_heads=2,
           context_in_dim=48, mlp_ratio=3)
STRENGTH = 0.8
H, W, N_TXT, SIGMA = 4, 6, 5, 0.25

torch.manual_seed(20261020)
src = qi.QwenImage21Transformer2DModel(**CFG, fused_mlp=True, dtype=torch.float32, device="cpu",
                                        operations=comfy.ops.disable_weight_init)
sd = {}
for name, p in src.named_parameters():
    if name.endswith("text_norm.weight"):
        t = torch.randn(p.shape) * 0.1
    elif name.endswith("norm_q.weight") or name.endswith("norm_k.weight"):
        t = 1.0 + torch.randn(p.shape) * 0.1
    elif "modulation" in name or "norm_out.linear" in name:
        t = torch.randn(p.shape) * (0.1 / p.shape[1] ** 0.5)
    elif p.dim() == 2:
        t = torch.randn(p.shape) / (p.shape[1] ** 0.5)
    else:
        t = torch.randn(p.shape) * 0.1
    sd[name] = t.to(torch.bfloat16).float()

# The LoRA, as a diffusers file carries it: `<name>.lora_{A,B}.weight` + `.alpha`.
d = CFG["attention_head_dim"] * CFG["num_attention_heads"]
ffn = d * CFG["mlp_ratio"]
targets = {  # name: (out, in, rank, alpha)
    "transformer_blocks.0.attn.to_q": (d, d, 4, 2.0),
    "transformer_blocks.1.attn.to_out.0": (d, d, 8, 8.0),
    "transformer_blocks.0.img_mlp.gate_layer": (ffn, d, 4, 4.0),
    "transformer_blocks.1.img_mlp.proj": (ffn, d, 6, 3.0),
    "transformer_blocks.1.img_mlp.out": (d, ffn, 4, 1.0),
    "proj_out": (CFG["out_channels"], d, 2, 2.0),
}
lora = {}
for name, (o, i, r, a) in targets.items():
    lora[f"{name}.lora_A.weight"] = (torch.randn(r, i) * 0.3).to(torch.bfloat16).float()
    lora[f"{name}.lora_B.weight"] = (torch.randn(o, r) * 0.3).to(torch.bfloat16).float()
    lora[f"{name}.alpha"] = torch.tensor(a)

patcher = comfy.sd.load_diffusion_model_state_dict({k: v.clone() for k, v in sd.items()}, model_options={"dtype": torch.float32})
if not getattr(patcher.model.diffusion_model, "fused_mlp", True):
    raise SystemExit("ComfyUI did not fuse the toy's FFN")
lpatch, _ = comfy.sd.load_lora_for_models(patcher, None, lora, STRENGTH, 0)
patched = sum(1 for k in lpatch.patches if lpatch.patches[k])
if patched != len(targets):
    raise SystemExit(f"ComfyUI patched {patched} tensors for {len(targets)} targets")

out = {}
for k, v in sd.items():
    t = v.to(torch.bfloat16)
    out["fused." + k] = t.clone()
    if k.endswith("img_mlp.gate_up.weight"):
        g, u = t.chunk(2, dim=0)
        base = k[: -len("gate_up.weight")]
        out["split." + base + "gate_layer.weight"] = g.clone()
        out["split." + base + "proj.weight"] = u.clone()
    else:
        out["split." + k] = t.clone()
for k, v in lora.items():
    t = v.to(torch.bfloat16) if v.dim() > 0 else v.float()
    out["lora_bare." + k] = t.clone()
    out["lora_tf.transformer." + k] = t.clone()

with torch.inference_mode():
    x = torch.randn(1, CFG["in_channels"], H, W)
    ctx = torch.randn(1, N_TXT, CFG["context_in_dim"])
    ts = torch.tensor([SIGMA])
    base = src
    base.load_state_dict(sd, strict=True)
    y0 = base(x, ts, ctx, transformer_options={})
    lpatch.patch_model()
    y1 = lpatch.model.diffusion_model(x, ts, ctx, transformer_options={})
    lpatch.unpatch_model()
    rel = ((y1 - y0).pow(2).mean() / y0.pow(2).mean()).sqrt()
    print(f"the LoRA moves the output by rel L2 {rel:.4f}")
    if rel < 0.05:
        raise SystemExit("the LoRA barely moves the output; the fixture would have no teeth")
out["x"] = x[0].contiguous()
out["ctx"] = ctx[0].contiguous()
out["sigma"] = ts
out["out_base"] = y0[0].contiguous()
out["out_lora"] = y1[0].contiguous()
out["strength"] = torch.tensor([STRENGTH])
safetensors.torch.save_file(out, OUT)
print("wrote", OUT, os.path.getsize(OUT) // 1024, "KiB")
