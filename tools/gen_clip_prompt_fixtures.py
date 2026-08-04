#!/usr/bin/env python3
"""Generate the CLIP prompt-parsing fixtures: attention weights + 77-token chunking.

The reference is **ComfyUI** (`comfy.sd1_clip.SDTokenizer.tokenize_with_weights`),
not A1111 and not diffusers, because ComfyUI is this engine's compatibility target
and it is what settles every convention here — none of which is derivable:

  * `BREAK` is NOT honoured (it tokenizes as the literal word `break`);
  * a bare paren multiplies the weight by 1.1 but an explicit `:w` REPLACES it;
  * a tokenized segment shorter than `max_word_length` (8) is pushed whole into the
    next chunk rather than split across the boundary;
  * `\\(` / `\\)` are literal parens hidden from the weight parser;
  * CLIP-L pads a short chunk with EOS, CLIP-G with 0.

Run with ComfyUI's own venv (NOT the ai-toolkit one):

    /home/qt/genai/comfyui/nvenv/bin/python tools/gen_clip_prompt_fixtures.py

Writes `src/core/assets/clip_tokenizer/fixtures_weighted.json`. This fixture is
model-free — it pins the tokenizer only — so the test that reads it is ungated and
lives in `tp_core`.
"""

import json
import os
import sys

COMFY = "/home/qt/genai/comfyui"
sys.path.insert(0, COMFY)
os.chdir(COMFY)

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import comfy.sd1_clip as sd1_clip  # noqa: E402
import comfy.sdxl_clip as sdxl_clip  # noqa: E402

OUT = "/dump/projects/zig/TensorPencil/src/core/assets/clip_tokenizer/fixtures_weighted.json"

# The two real prompts that exposed the bug this fixture exists for: 115 tokens over
# two chunks with three distinct weights, and a negative carrying an escaped paren.
REAL_POS = (
    "1girl, original, general, blonde hair, long hair, ponytail, light blue eyes, "
    "(shiny skin:1.1), gentle smile, bare shoulders, collarbone, "
    "[white|light blue] off-shoulder shirt, sheer sleeves, denim short shorts, "
    "holding straw hat, looking back, side angle, upper body, beach, ocean, summer, "
    "sunny day, (lens flare:0.75), breeze, BREAK, masterpiece, best quality, "
    "very aesthetic, high contrast, vibrant, highres, year 2024, newest,"
)
REAL_NEG = (
    "(worst quality, low quality:1.2), (nsfw, sexually suggestive:1.2), lowres, "
    "(monochrome:1.1), wide shot, multiple views, watermark, signature, "
    "1980s \\(style\\), 4koma, serafuku, (text:1.1),"
)

# A filler word that is exactly one token, so a boundary can be placed by counting.
FILL = " ".join(["cat"] * 70)
# `intricate` BPEs to several pieces; `(a:1.3)` is one short segment. Placed so the
# chunk boundary lands inside the parenthesized group, which is what pins the
# keep-short-segments-whole rule.
STRADDLE_SHORT = FILL + " (dog:1.3) more words here"
# A long segment across the boundary instead: >= 8 tokens, so it must be SPLIT.
STRADDLE_LONG = FILL + " (one two three four five six seven eight nine:1.3) tail"

CASES = [
    REAL_POS,
    REAL_NEG,
    "",
    "a red cat",
    "a (red:1.5) cat",
    "a (red) cat",
    "((a)) cat",  # bare nesting is cumulative: 1.21
    "((a:1.5)) cat",  # an inner absolute weight wins outright
    "(((deep))) end",  # 1.331
    "(a:0) b",  # weight 0 is the empty-prompt state, not a zero vector
    "(a:2.0) b",  # extrapolation past 1
    "(a:-1) b",  # negative weights are legal
    "1980s \\(style\\)",  # escaped parens are literal text
    "\\(a:1.5\\)",  # fully escaped: no weight at all
    "a) b (c",  # unbalanced, must be read as literal text
    "(a",
    "a)",
    "()",
    "(:1.5)",  # leading colon is text, not a separator
    "(a:)",  # unparseable weight: falls back to the bare 1.1
    "(a:not_a_number)",
    "(a:1.5:2.5)",  # rfind, so the LAST colon wins
    "BREAK",  # not a keyword here
    "one BREAK two",
    STRADDLE_SHORT,
    STRADDLE_LONG,
    FILL,  # exactly 70 content tokens: one chunk
    " ".join(["cat"] * 200),  # three chunks
    "trailing spaces   ",
    "MiXeD CaSe (EMPHASIS:1.2)",
    "emoji 🌊 and unicode café",
]


def rows(tok, text):
    chunks = tok.tokenize_with_weights(text, return_word_ids=False)
    return [
        {"ids": [int(t) for t, _ in ch], "weights": [float(w) for _, w in ch]}
        for ch in chunks
    ]


def main():
    tok_l = sd1_clip.SDTokenizer()  # pad_with_end=True  -> pad 49407
    tok_g = sdxl_clip.SDXLClipGTokenizer()  # pad_with_end=False -> pad 0
    assert tok_l.max_length == 77 and tok_l.max_word_length == 8
    assert tok_l.pad_token == 49407 and tok_g.pad_token == 0

    out = []
    for text in CASES:
        l = rows(tok_l, text)
        g = rows(tok_g, text)
        # Every chunk must be exactly max_length, or the Zig side's flat
        # [chunks][77] layout is not what the reference produced.
        for ch in l + g:
            assert len(ch["ids"]) == 77, (text, len(ch["ids"]))
        out.append({"text": text, "clip_l": l, "clip_g": g})

    with open(OUT, "w") as f:
        json.dump({"clip_prompt": out}, f, indent=1)
    n_chunks = sum(len(c["clip_l"]) for c in out)
    n_weighted = sum(
        1 for c in out for ch in c["clip_l"] if any(w != 1.0 for w in ch["weights"])
    )
    print(f"wrote {OUT}: {len(out)} prompts, {n_chunks} clip_l chunks, {n_weighted} weighted")


if __name__ == "__main__":
    main()
