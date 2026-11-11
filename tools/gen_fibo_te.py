#!/usr/bin/env python3
"""Reference conditioning for Fibo 1.5's text encoder, by EXECUTING diffusers'
`BriaFiboPipeline.get_prompt_embeds` on the real SmolLM3-3B at fp32 on the CPU.

What it pins: the raw prompt with nothing added (no template, no BOS; an empty prompt
is the single token 128000), every one of the 37 hidden states transformers returns
(the embeddings, each layer's output, the last one after the final norm), and the
context `[hs[-1] | hs[-2]]`. SmolLM3 is Llama-shaped with no RoPE on every fourth
layer.

Stores the token ids, the context, and the hidden states at a few layers, each as a
prefix of its columns (a wrong tap, norm or rope layer is wrong in every column).

Run it memory-bounded (3B at fp32 is 12 GB):

    systemd-run --user --scope -p MemoryMax=20G -p MemorySwapMax=0 \\
        /home/qt/genai/comfyui/nvenv/bin/python tools/gen_fibo_te.py
"""

import os

import torch
import safetensors.torch
from transformers import AutoTokenizer
from transformers.models.smollm3.modeling_smollm3 import SmolLM3ForCausalLM
from diffusers.pipelines.bria_fibo.pipeline_bria_fibo import BriaFiboPipeline

ROOT = "/mnt/watch/models/diffusion/diffusers/Fibo-1.5"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "models", "assets", "fibo_te_ref.safetensors")
COLS = 256
LAYERS = [0, 1, 3, 4, 17, 35, 36]

PROMPTS = [
    '{"short_description": "a lighthouse at dusk", "lighting": {"conditions": "golden hour"}}',
    "",
]


def main():
    tok = AutoTokenizer.from_pretrained(f"{ROOT}/tokenizer")
    te = SmolLM3ForCausalLM.from_pretrained(f"{ROOT}/text_encoder", torch_dtype=torch.float32).eval()
    pipe = BriaFiboPipeline.__new__(BriaFiboPipeline)
    pipe.tokenizer = tok
    pipe.text_encoder = te
    out = {}
    for i, p in enumerate(PROMPTS):
        with torch.no_grad():
            embeds, hidden, mask = BriaFiboPipeline.get_prompt_embeds(pipe, p, device=torch.device("cpu"), max_sequence_length=3000)
        assert len(hidden) == 37
        ids = tok(p, add_special_tokens=True).input_ids if p else [128000]
        out[f"tokens.{i}"] = torch.tensor(ids, dtype=torch.int32)
        out[f"context.{i}"] = torch.cat([embeds[0][:, :COLS], embeds[0][:, 2048:2048 + COLS]], -1).contiguous()
        for l in LAYERS:
            out[f"hidden.{i}.{l}"] = hidden[l][0][:, :COLS].contiguous()
        print(i, len(ids), "tokens")
    safetensors.torch.save_file(out, OUT, metadata={"cols": str(COLS), "layers": ",".join(map(str, LAYERS))})
    print("wrote", OUT, os.path.getsize(OUT))


if __name__ == "__main__":
    main()
