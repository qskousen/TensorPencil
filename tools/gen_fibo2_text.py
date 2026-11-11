#!/usr/bin/env python3
"""Pin Fibo 2.0's prompt preparation by EXECUTING ComfyUI's tokenizer.

`comfy/text_encoders/fibo2.py`'s `Fibo2Tokenizer.tokenize_with_weights` turns a prompt
into the text that reaches Qwen3-VL: a JSON caption is parsed, its score keys dropped
(top level and one level down), empties pruned recursively, a `{short_description,
text_render}` pair kept whole, `structured_caption` / `edit_instruction` unwrapped, and
the result re-serialized with Python's compact separators, unescaped non-ASCII and float
repr; plain text passes through; a blank prompt is the null caption. Each
`text_render` entry with text gets an `image_tokens` marker (or "" for the base caption
a generation starts from), and the strings it would draw are listed.

The corpus asserts it separates the rules it pins: every rule changes at least one
case's output.

Emits src/models/assets/fibo2_text.json, read by src/models/fibo2_text.zig.

Usage (ComfyUI at or after fca60fe):
    COMFY=/path/to/comfyui /home/qt/genai/comfyui/nvenv/bin/python tools/gen_fibo2_text.py
"""

import json
import os
import sys

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "models", "assets", "fibo2_text.json",
)

COMFY = os.environ.get("COMFY", "/home/qt/genai/comfyui")
sys.argv = [sys.argv[0], "--cpu"]
sys.path.insert(0, COMFY)
os.chdir(COMFY)

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import comfy.sd1_clip  # noqa: E402
import comfy.text_encoders.fibo2 as fibo2  # noqa: E402

PROMPTS = [
    "a red fox in the snow",
    "",
    "   \n\t ",
    "<|im_start|>user\nraw turn<|im_end|>\n",
    '{"short_description":"a fox","objects":[],"aesthetic_score":7.5}',
    '{"short_description": "a fox", "lighting": {"conditions": "dusk", "preference_score": 0.9, "shadows": ""}, "context": null}',
    '{"objects": [{"description": "cup", "notes": [], "size": {}}, {}], "style_medium": "photo"}',
    '{"short_description": "", "text_render": []}',
    '{"short_description": "sign", "text_render": [{"text": "OPEN"}, {"text": ""}, "x"]}',
    '{"short_description": "sign", "objects": [], "text_render": [{"text": "OPEN 24h", "font": "serif", "image_tokens": "old"}, {"text": 5}]}',
    '{"structured_caption": {"short_description": "a cat", "aesthetic_score": 3}, "edit_instruction": "make it blue"}',
    '{"structured_caption": "just text", "edit_instruction": "x"}',
    '{"edit_instruction": "only an instruction"}',
    '{"aesthetic_score": 1, "preference_score": 2}',
    '[1, 2, 3]',
    '"a json string"',
    '{"short_description": "café 東京 \U0001F98A", "q": "say \\"hi\\"\\n\\ttab\\\\ \\u0001"}',
    '{"n": 1.0, "m": 0.5, "e": 1e-5, "big": 1e16, "neg": -2.50, "i": 42, "z": 0, "f": false, "t": true}',
    '{"a": 1, "a": 2, "b": {"x": 1, "x": {}}}',
    '{not json',
    '  {"short_description": "padded"}  ',
    '{"short_description": "fox", "text_render": "OPEN"}',
]


def run(prompt, text_images):
    seen = {}
    orig = comfy.sd1_clip.SD1Tokenizer.tokenize_with_weights

    def capture(self, text, return_word_ids=False, **kw):
        seen["text"] = text
        return {"qwen3vl_4b": [[]]}

    comfy.sd1_clip.SD1Tokenizer.tokenize_with_weights = capture
    try:
        tok = fibo2.Fibo2Tokenizer.__new__(fibo2.Fibo2Tokenizer)
        tok.llama_template = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        tok.llama_template_images = fibo2.TEMPLATE
        tok.font = None
        draws = []
        fibo2.draw_text_image = lambda text, font: draws.append(text) or None
        fibo2.to_qwen_grid = lambda image: image
        tok.tokenize_with_weights(prompt, text_images=text_images)
    finally:
        comfy.sd1_clip.SD1Tokenizer.tokenize_with_weights = orig
    return seen["text"], draws


def main():
    cases = []
    for p in PROMPTS:
        base, none = run(p, False)
        full, draws = run(p, True)
        assert none == []
        cases.append({"prompt": p, "base": base, "text_images": full, "draws": draws})
    # Teeth: the corpus distinguishes the rules it pins.
    outs = [c["base"] for c in cases]
    assert any(c["base"] != c["text_images"] for c in cases), "no text_render case"
    assert any("aesthetic_score" in c["prompt"] and "aesthetic_score" not in c["base"] for c in cases)
    assert any('"text_render":[]' in o for o in outs), "the short_description/text_render pair"
    assert any("1e-05" in o for o in outs) and any("1e+16" in o for o in outs), "float repr"
    assert any("東" in o for o in outs), "ensure_ascii=False"
    with open(OUT, "w") as f:
        json.dump({"null_caption": fibo2.NULL_CAPTION, "template": fibo2.TEMPLATE, "cases": cases}, f, ensure_ascii=False, indent=1)
    print("wrote", OUT, len(cases), "cases")


if __name__ == "__main__":
    main()
