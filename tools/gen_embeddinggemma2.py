#!/usr/bin/env python3
"""Reference vectors for EmbeddingGemma 2's text path (`google/embeddinggemma-2`),
by running sentence-transformers itself: the role prompt, the tokenizer's
`<bos> ... <eos>` frame, the bidirectional text model, mean pooling and the
L2 normalize. f32 weights (upcast from the checkpoint's bf16), eager attention,
CPU.

Per case it writes the prompt name (`query`, `document` or null), the raw text,
the token ids (prompt included), the 768-d vector, and the cosine of a bf16 run
of the same input: that is the precision floor a port can not be judged below.

`long` is past the 1025-token reach of the local layers' bidirectional window
(|q - k| <= 512). `long.vec_nowindow` is the same input with the window removed;
the generator refuses to write a corpus where the two agree, so a port that
ignores the window fails the test.

Needs transformers new enough to know `embedding_gemma2` and
sentence-transformers >= 6.1. Run it memory-bounded:

    systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 \\
        /dump/venv-eg2/bin/python tools/gen_embeddinggemma2.py
"""

import json
import os
import sys

import torch
from sentence_transformers import SentenceTransformer

MODEL = "/dump/projects/zig/DiffKeep/Models/embeddinggemma-2"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "testdata", "embeddinggemma2_ref_vectors.json")

LONG = " ".join(
    f"Paragraph {i}: the lighthouse keeper logged wind speed, tide height and the colour of the sky at dusk."
    for i in range(60))

# name -> (prompt_name or None, text)
CASES = {
    "query_fox": ("query", "a red fox sitting on a mossy rock"),
    "doc_fox": ("document", "A red fox rests on a moss-covered boulder in the early morning light."),
    "query_code": ("query", "how do I reverse a linked list in place"),
    "doc_multilingual": ("document", "Der schnelle braune Fuchs springt über den faulen Hund. 東京は日本の首都です。"),
    "raw_short": (None, "hello"),
    "raw_symbols": (None, "  spaces,\ttabs\nand newlines -- 3.14159 ~~~ 🙂"),
    "long": ("document", LONG),
}


def encode(model, prompt_name, text):
    with torch.no_grad():
        v = model.encode([text], prompt_name=prompt_name, convert_to_tensor=True)[0]
    return v.float()


def main():
    torch.manual_seed(0)
    f32 = SentenceTransformer(MODEL, device="cpu", model_kwargs={"torch_dtype": torch.float32, "attn_implementation": "eager"})
    bf16 = SentenceTransformer(MODEL, device="cpu", model_kwargs={"torch_dtype": torch.bfloat16, "attn_implementation": "eager"})
    tok = f32.tokenizer if hasattr(f32, "tokenizer") else f32[0].tokenizer

    out = {}
    for name, (prompt_name, text) in CASES.items():
        full = (f32.prompts[prompt_name] if prompt_name else "") + text
        ids = tok(full)["input_ids"]
        if isinstance(ids[0], list):
            ids = ids[0]
        v = encode(f32, prompt_name, text)
        vb = encode(bf16, prompt_name, text)
        cos_bf16 = float(torch.dot(v, vb) / (v.norm() * vb.norm()))
        out[name] = {"prompt": prompt_name, "text": text, "ids": [int(i) for i in ids], "vec": v.tolist(), "bf16_cos": cos_bf16}
        print(f"{name}: {len(ids)} tokens, |v| {v.norm():.6f}, bf16 cos {cos_bf16:.6f}")

    if len(out["long"]["ids"]) <= 1025:
        sys.exit("long case does not exceed the window reach")
    lm = f32[0].model.language_model if hasattr(f32[0].model, "language_model") else f32[0].model.model.language_model
    lm.config.sliding_window = 1 << 20
    vn = encode(f32, "document", LONG)
    v = torch.tensor(out["long"]["vec"])
    cos_nw = float(torch.dot(v, vn) / (v.norm() * vn.norm()))
    print(f"long without window: cos {cos_nw:.6f}")
    if cos_nw > 0.9995:
        sys.exit("the window does not change the long vector; the case has no teeth")
    out["long"]["vec_nowindow"] = vn.tolist()
    out["long"]["nowindow_cos"] = cos_nw

    with open(OUT, "w") as f:
        json.dump(out, f)
    print("wrote", OUT, os.path.getsize(OUT) // 1024, "KiB")


if __name__ == "__main__":
    main()
