#!/usr/bin/env python3
"""Reference vectors for EmbeddingGemma 300m (`google/embeddinggemma-300m`) past its
local layers' window, by running sentence-transformers itself: the bidirectional
Gemma 3 body, mean pooling, the two Dense layers and the L2 normalize. f32, eager
attention, CPU.

transformers turns the config's `sliding_window: 512` into 257 for a bidirectional
model and masks `|q - k| < 257`, so a local-layer token sees 256 positions either
side. The generator checks that it is running that convention.

Each case is `<bos>`, a prefix of one long prompt-like token stream, `<eos>`, at an
exact length (the model maximum is 2048). Per case it writes the ids, the 768-d
vector, the cosine of a bf16 run (the precision floor), and two vectors a wrong port
would produce: `vec_nowindow` (every layer global) and `vec_wide` (radius 257, one
too many). The generator refuses a corpus where either is not distinguishable.

    systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 \\
        /dump/venv-eg2/bin/python tools/gen_embeddinggemma_long.py
"""

import json
import os
import random
import sys

import torch
from sentence_transformers import SentenceTransformer

MODEL = "/dump/projects/zig/DiffKeep/Models/embeddinggemma-300m"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "testdata", "embeddinggemma_long_ref.json")

LENGTHS = {"len300": 300, "len600": 600, "len1000": 1000, "len2048": 2048}
BOS, EOS = 2, 1

WORDS = ("masterpiece best quality portrait woman man knight dragon castle forest river "
         "mountain sunset sunrise neon city street rain snow fog lantern armor sword cloak "
         "detailed intricate cinematic lighting volumetric soft shadows bokeh depth of field "
         "oil painting watercolor sketch ink anime photorealistic 35mm film grain golden hour "
         "blue red green purple silver gold eyes hair smile freckles crown flowers ruins ocean "
         "ship storm clouds stars moon galaxy robot cyberpunk steampunk baroque gothic temple").split()


def corpus():
    rng = random.Random(7)
    parts = []
    for i in range(900):
        tags = ", ".join(rng.choice(WORDS) + (f" {rng.randint(1, 99)}" if rng.random() < 0.2 else "")
                         for _ in range(rng.randint(3, 9)))
        parts.append(f"({tags}:{rng.randint(5, 15) / 10}) scene {i}")
    return ", ".join(parts)


def encode(model, ids):
    t = torch.tensor([ids])
    with torch.no_grad():
        out = model({"input_ids": t, "attention_mask": torch.ones_like(t)})
    return out["sentence_embedding"][0].float()


def cos(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


def main():
    kw = {"attn_implementation": "eager"}
    f32 = SentenceTransformer(MODEL, device="cpu", model_kwargs={"torch_dtype": torch.float32, **kw})
    bf16 = SentenceTransformer(MODEL, device="cpu", model_kwargs={"torch_dtype": torch.bfloat16, **kw})
    cfg = f32[0].auto_model.config
    if not cfg.use_bidirectional_attention or cfg.sliding_window != 257:
        sys.exit(f"unexpected window convention: bidirectional {cfg.use_bidirectional_attention}, window {cfg.sliding_window}")
    if f32.max_seq_length != 2048 or cfg.max_position_embeddings != 2048:
        sys.exit("unexpected maximum length")

    stream = f32.tokenizer(corpus(), add_special_tokens=False)["input_ids"]
    if len(stream) < 2046:
        sys.exit("corpus too short")

    out = {}
    for name, n in LENGTHS.items():
        ids = [BOS] + stream[: n - 2] + [EOS]
        v = encode(f32, ids)
        vb = encode(bf16, ids)
        out[name] = {"ids": ids, "vec": v.tolist(), "bf16_cos": cos(v, vb)}

    for key, window in (("vec_nowindow", 1 << 20), ("vec_wide", 258)):
        cfg.sliding_window = window
        for name in LENGTHS:
            v = torch.tensor(out[name]["vec"])
            w = encode(f32, out[name]["ids"])
            out[name][key] = w.tolist()
            out[name][key + "_cos"] = cos(v, w)
    cfg.sliding_window = 257

    for name, c in out.items():
        print(f"{name}: {len(c['ids'])} tokens, bf16 cos {c['bf16_cos']:.7f}, "
              f"nowindow cos {c['vec_nowindow_cos']:.7f}, wide cos {c['vec_wide_cos']:.7f}")
        if c["vec_nowindow_cos"] > 0.9995:
            sys.exit(f"{name}: the window does not change the vector; the case has no teeth")
        if 1 - c["vec_wide_cos"] < 20 * (1 - 0.9999999):
            sys.exit(f"{name}: a radius one too wide is indistinguishable")

    with open(OUT, "w") as f:
        json.dump(out, f)
    print("wrote", OUT, os.path.getsize(OUT) // 1024, "KiB")


if __name__ == "__main__":
    main()
