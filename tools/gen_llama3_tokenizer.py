#!/usr/bin/env python3
"""Extract the Llama 3 byte-level BPE (as SmolLM3, Fibo 1.5's text encoder, ships it)
into the embedded tokenizer assets, and record a token-id corpus by EXECUTING the HF
`tokenizers` library on the same file.

SmolLM3's `tokenizer.json` is Llama 3's vocabulary and merges with its own 256 added
tokens, the Llama 3 split regex (digits in runs of up to three), `ignore_merges` (a
pretoken that is itself a vocabulary entry is emitted whole), no normalizer, and a
post-processor that adds nothing: no BOS.

Writes src/core/assets/llama3_tokenizer/{vocab.json,merges.txt,added_tokens.json} and
src/core/assets/llama3_tokenizer_fixtures.json. The corpus asserts it reaches
`ignore_merges`: at least one case tokenizes differently with it off.

Usage:
    /home/qt/genai/comfyui/nvenv/bin/python tools/gen_llama3_tokenizer.py [tokenizer.json]
"""

import json
import os
import sys

from tokenizers import Tokenizer

SRC = sys.argv[1] if len(sys.argv) > 1 else "/mnt/watch/models/diffusion/diffusers/Fibo-1.5/tokenizer/tokenizer.json"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "src", "core", "assets", "llama3_tokenizer")

CORPUS = [
    "a red fox in the snow",
    "",
    "1234567 and 12 and 3.14159, year 2026",
    '{"short_description":"A lighthouse at dusk","objects":[{"description":"tower","location":"center-left"}]}',
    "Hello WORLD it's we'll they'RE I'M",
    "café 東京 🦊 naïve résumé",
    "tabs\tand\nnewlines\r\n\n  spaces   end ",
    "<|im_start|>user\nhi<|im_end|>\n<|begin_of_text|>",
    "Ελληνικά и русский текст, العربية",
    "    indented code(x) -> y; // comment",
    "sssssssssssssssssss ________________ ................",
    "émojis 👨‍👩‍👧 and ZWJ",
    # Whole-vocabulary pretokens plain BPE does not reach: only `ignore_merges` gives one id.
    "công việc ở Việt Nam, nhiều điều hợp lý",
]


def main():
    doc = json.load(open(SRC))
    model = doc["model"]
    assert model["type"] == "BPE" and model.get("ignore_merges") is True
    assert doc.get("normalizer") is None
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "vocab.json"), "w") as f:
        json.dump(model["vocab"], f, ensure_ascii=False, separators=(",", ":"))
    with open(os.path.join(OUT, "merges.txt"), "w") as f:
        for m in model["merges"]:
            f.write((m if isinstance(m, str) else " ".join(m)) + "\n")
    added = [[t["id"], t["content"]] for t in doc["added_tokens"]]
    with open(os.path.join(OUT, "added_tokens.json"), "w") as f:
        json.dump(added, f, ensure_ascii=False)

    tok = Tokenizer.from_file(SRC)
    cases = [{"text": t, "ids": tok.encode(t).ids} for t in CORPUS]
    doc_off = json.loads(json.dumps(doc))
    doc_off["model"]["ignore_merges"] = False
    tok_off = Tokenizer.from_str(json.dumps(doc_off))
    assert any(tok_off.encode(t).ids != c["ids"] for t, c in zip(CORPUS, cases)), "the corpus never reaches ignore_merges"
    with open(os.path.join(ROOT, "src", "core", "assets", "llama3_tokenizer_fixtures.json"), "w") as f:
        json.dump(cases, f, ensure_ascii=False, indent=1)
    print("vocab", len(model["vocab"]), "merges", len(model["merges"]), "added", len(added), "cases", len(cases))


if __name__ == "__main__":
    main()
