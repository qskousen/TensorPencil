#!/usr/bin/env python3
"""Write a random low-rank LoRA for a checkpoint that has no real one on disk.

For the families with no LoRA file here (Anima, Mage-Flow), so the device-vs-CPU
checks and the attach count have something to run on. It reads only the
checkpoint's safetensors HEADER, picks the linears a trained LoRA for that
family patches, and writes PEFT `lora_A` / `lora_B` pairs in bf16 under
ComfyUI's generic `diffusion_model.<name>` spelling, with `alpha == rank`.

A is N(0, 1/in) per entry, so `A x` keeps the input's scale per rank; B is
N(0, scale^2 / rank), so the delta a linear adds is about `scale` times the
input's rms. The default 0.02 moves a render visibly at strength 1 without
throwing it off the data manifold.

Usage:
    /home/qt/genai/comfyui/nvenv/bin/python tools/gen_lora_synth.py \
        --family anima <checkpoint> <out.safetensors> [--rank 32] [--scale 0.02] [--seed 7]
"""

import argparse
import json
import re
import struct

import torch
from safetensors.torch import save_file

PREFIXES = ("model.diffusion_model.",)

# What a trained LoRA for the family patches, on the checkpoint's own names.
TARGETS = {
    "anima": [
        r"blocks\.\d+\.(self_attn|cross_attn)\.(q|k|v|output)_proj",
        r"blocks\.\d+\.mlp\.layer[12]",
        r"llm_adapter\.blocks\.\d+\.(self_attn|cross_attn)\.(q|k|v|o)_proj",
    ],
    "mageflow": [
        r"transformer_blocks\.\d+\.attn\.(to_q|to_k|to_v|to_out\.0|add_q_proj|add_k_proj|add_v_proj|to_add_out)",
        r"transformer_blocks\.\d+\.(img|txt)_mlp\.net\.(0\.proj|2)",
        r"transformer_blocks\.\d+\.(img|txt)_mod\.1",
    ],
}


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=sorted(TARGETS))
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--scale", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    pats = [re.compile("^(" + p + r")\.weight$") for p in TARGETS[a.family]]
    g = torch.Generator().manual_seed(a.seed)
    out = {}
    for name, info in sorted(header(a.ckpt).items()):
        base = name
        for p in PREFIXES:
            if base.startswith(p):
                base = base[len(p):]
        m = next((p.match(base) for p in pats if p.match(base)), None)
        if m is None or len(info["shape"]) != 2:
            continue
        rows, cols = info["shape"]
        stem = "diffusion_model." + m.group(1)
        A = torch.randn(a.rank, cols, generator=g) / cols ** 0.5
        B = torch.randn(rows, a.rank, generator=g) * (a.scale / a.rank ** 0.5)
        out[stem + ".lora_A.weight"] = A.to(torch.bfloat16)
        out[stem + ".lora_B.weight"] = B.to(torch.bfloat16)
    n = len(out) // 2
    if n == 0:
        raise SystemExit("no target matched; wrong --family for this checkpoint?")
    save_file(out, a.out, metadata={"generator": "tools/gen_lora_synth.py", "family": a.family,
                                    "rank": str(a.rank), "scale": str(a.scale), "seed": str(a.seed)})
    print(f"{a.out}: {n} targets, rank {a.rank}, scale {a.scale}")


if __name__ == "__main__":
    main()
