#!/usr/bin/env python3
"""Reference conditioning for Fibo 2.0's text encoder (`comfy/text_encoders/fibo2.py`):
Qwen3-VL-4B's language model, the hidden states ENTERING fifteen of its layers with no
final norm, each unit-RMS-normed at eps 1e-6 and rounded to bf16, laid out as six
bundles of three. Run by ComfyUI itself at fp32 on the CPU, through the tokenizer, so
the prompt rewriting and the template are the reference's too.

The fixture keeps every token row but only the first `COLS` columns of each of the
eighteen bundle slots: a wrong tap, a wrong slot order or a missing norm is wrong in
every column, so a prefix of each is enough. Stored as bf16, which is lossless here.

Run it memory-bounded (the 4B model at fp32 is 16 GB):

    systemd-run --user --scope -p MemoryMax=24G -p MemorySwapMax=0 \\
        env COMFY=/path/to/comfyui /home/qt/genai/comfyui/nvenv/bin/python tools/gen_fibo2_te.py
"""

import hashlib
import os
import sys

COMFY = os.environ.get("COMFY", "/home/qt/genai/comfyui")
TE = os.environ.get("TE", "/mnt/watch/models/diffusion/text_encoders/qwen3vl_4b_bf16.safetensors")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "models", "assets", "fibo2_te_ref.safetensors")
COLS = 320

sys.argv = [sys.argv[0], "--cpu", "--disable-smart-memory", "--use-pytorch-cross-attention"]
sys.path.insert(0, COMFY)
os.chdir(COMFY)

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import torch  # noqa: E402
import safetensors.torch  # noqa: E402

import comfy.sd  # noqa: E402

PROMPTS = [
    "a red fox in the snow",
    '{"short_description": "a lighthouse at dusk", "aesthetic_score": 8, "lighting": {"conditions": "golden hour"}, "objects": []}',
]


def main():
    clip = comfy.sd.load_clip(ckpt_paths=[TE], clip_type=comfy.sd.CLIPType.FIBO2, model_options={"dtype": torch.float32})
    out = {}
    for i, p in enumerate(PROMPTS):
        tokens = clip.tokenize(p)
        ids = [t[0] for t in next(iter(tokens.values()))[0]]
        cond = clip.encode_from_tokens(tokens)  # (1, L, 18 * 2560), bf16 values
        cond = cond[0].float()
        L = cond.shape[0]
        slots = cond.view(L, 18, 2560)[:, :, :COLS].contiguous()
        out[f"tokens.{i}"] = torch.tensor(ids, dtype=torch.int32)
        out[f"cond.{i}"] = slots.to(torch.bfloat16)
        print(i, L, "tokens; cond rms", cond.pow(2).mean().sqrt().item())
    with open(TE, "rb") as f:
        head = f.read(1 << 20)
    meta = {"te_head_sha256": hashlib.sha256(head).hexdigest(), "cols": str(COLS)}
    safetensors.torch.save_file(out, OUT, metadata=meta)
    print("wrote", OUT, os.path.getsize(OUT))


if __name__ == "__main__":
    main()
