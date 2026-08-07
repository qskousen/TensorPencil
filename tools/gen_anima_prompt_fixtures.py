#!/usr/bin/env python3
"""Reference fixtures for Anima's prompt path — BOTH of its tokenizers.

Anima conditions on two tokenizations of the same prompt string
(`comfy/text_encoders/anima.py::AnimaTokenizer`):

  * `qwen3_06b` — Qwen2 byte-level BPE, feeding the Qwen3-0.6B text encoder whose
    hidden states are the `llm_adapter`'s cross-attention *source*. ⚠️ Its weights
    are forced to 1.0 by `AnimaTokenizer.tokenize_with_weights`, so emphasis never
    reaches this branch.
  * `t5xxl` — SentencePiece **Unigram** (32100 pieces), feeding the `llm_adapter`'s
    own `Embedding(32128, 1024)` as `target_input_ids`, and carrying the per-token
    weights that multiply the adapter's output rows.

So `(a:1.5)` in an Anima prompt is a T5-side effect only. That asymmetry is not
derivable from anything; it is why this fixture exists.

⚠️ **The weighted segments are tokenized SEPARATELY**, one `self.tokenizer(word)`
call each (`sd1_clip.SDTokenizer.tokenize_with_weights`). Metaspace's
`add_prefix_space` therefore fires per segment, so `a (b:1.1) c` does not produce
the ids of `a b c` with one weight changed — it produces a different id sequence.
Reproducing the ids without reproducing the segmentation passes a naive test and
renders a different image.

⚠️ Also per segment: T5's `has_end_token` is True, so `input_ids[0:-1]` drops the
`</s>` that its `TemplateProcessing` post-processor appended — every segment. And
`has_start_token=False` for both, `pad_to_max_length=False`, `max_length=99999999`,
so there is exactly one chunk and no padding anywhere in the tokenizer. The
`llm_adapter`'s zero-pad to 512 rows happens later, in the model.

Reference is ComfyUI's own class, executed — not re-derived. Emitted as
`src/core/assets/t5_tokenizer/fixtures.json` (ungated: `tp_core` owns it, and
`tp_core` cannot import `test_gate`, so the Zig test self-skips by catching the
open error).

Usage (ComfyUI's `nvenv`, NOT the ai-toolkit venv):
    /home/qt/genai/comfyui/nvenv/bin/python tools/gen_anima_prompt_fixtures.py
"""

import argparse
import hashlib
import json
import os
import sys

COMFY = "/home/qt/genai/comfyui"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "src", "core", "assets", "t5_tokenizer", "fixtures.json")
T5_JSON = os.path.join(COMFY, "comfy/text_encoders/t5_tokenizer/tokenizer.json")

OUR_ARGV = sys.argv[1:]
sys.argv = [sys.argv[0], "--cpu"]
sys.path.insert(0, COMFY)
os.chdir(COMFY)

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

from comfy.cli_args import args as comfy_args  # noqa: E402

if not comfy_args.cpu:
    raise SystemExit("ComfyUI did not take our flags — check comfy.options.enable_args_parsing()")

import comfy.text_encoders.anima as anima_te  # noqa: E402

# The real workflow's prompts (from the reference render's `parameters` block),
# then cases chosen to stress one decision each.
CASES = [
    # --- the actual render -------------------------------------------------
    (
        "the reference render's positive prompt",
        'score_8, masterpiece, newest, absurdres, incredibly absurdres, best quality, '
        'amazing quality, very aesthetic, solo, chromatic aberration, rainbow hair, '
        'black hair, (black theme), hair bun, limited palette, tenebrism, close up, '
        'depth of field, plump lips, victorian maid dress, elegant clothes, closed mouth, '
        'swept bangs, from side, looking to the side, dark red background, large breasts, '
        'cleavage, (lipstick writing on breasts says "Lilith\'s Desire")',
    ),
    (
        "the reference render's negative prompt",
        "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia",
    ),
    # --- the empty prompt: one chunk, and for T5 an EMPTY id list ----------
    ("empty", ""),
    ("single space", " "),
    # --- segmentation: the same words, differently split -------------------
    ("no weights", "a cat sitting on a mat"),
    ("one weighted segment", "a cat (sitting:1.3) on a mat"),
    # ⚠️ A weight boundary at a SPACE is transparent — the segment's own
    # `add_prefix_space` produces the same `▁` the space would have become. Only a
    # MID-WORD split shows that segments are tokenized separately: `cat(s:1.1)` is
    # `▁cat` ++ `▁s`, never `▁cats`. This case is what gives the fixture teeth.
    ("mid-word weight split", "a cat(s:1.1) on a mat"),
    ("mid-word split, no space anywhere", "under(water:1.2)fall"),
    ("bare parens multiply", "a (cat) on a mat"),
    ("nested bare parens", "a ((cat)) on a mat"),
    # ⚠️ inner explicit weight inside an outer one — the only shape that tells
    # ComfyUI's "replace" apart from A1111's "multiply". See CLAUDE.md.
    ("explicit inside explicit", "((a:1.2):1.5) landscape"),
    ("weight at the very start", "(masterpiece:1.4) a cat"),
    ("weight at the very end", "a cat (masterpiece:1.4)"),
    ("adjacent weighted segments", "(a:1.1)(b:1.2)(c:0.9)"),
    ("escaped paren", r"a \(literal\) paren"),
    ("de-emphasis weight", "a (cat:0.5) on a mat"),
    # An empty weighted segment: `to_tokenize` drops "" before tokenizing, so
    # these must not contribute an id (nor an extra U+2581 pre-token).
    ("empty weighted segment", "a (:1.5) mat"),
    ("empty parens", "a () mat"),
    # --- Metaspace / normalizer edge cases --------------------------------
    ("leading spaces", "   leading spaces"),
    ("trailing spaces", "trailing spaces   "),
    # ⚠️ Precompiled -> Strip(right) -> Replace(/ {2,}/ -> U+2581) means an
    # interior run of spaces becomes a single U+2581 BEFORE Metaspace runs.
    ("interior double space", "two  spaces  inside"),
    ("interior triple space", "three   spaces"),
    ("tab and newline", "tab\there\nnewline"),
    ("only punctuation", "!!! ??? ... ,,,"),
    # --- the Precompiled charsmap doing real work -------------------------
    ("nfkc ligature", "the ﬁre ﬂame"),  # ﬁ ﬂ -> fi fl
    ("fullwidth latin", "Ｈｅｌｌｏ"),  # Ｈｅｌｌｏ -> Hello
    ("roman numeral", "Ⅸ and Ⅰ"),  # Ⅸ Ⅰ -> IX I
    ("nbsp", "a b"),
    ("combining accent", "café vs café"),  # NFKC composes the first
    ("superscript digit", "x² + y³"),
    ("circled digit", "①②③"),
    ("cjk", "猫が座っている"),
    ("cyrillic", "кошка на мате"),
    ("emoji", "a cat \U0001f431 on a mat"),
    ("zwj emoji sequence", "\U0001f469‍\U0001f4bb coding"),
    ("soft hyphen", "sen­tence"),
    # ⚠️ The charsmap DELETES 30 control characters (C0 minus \t\n\r\f, plus DEL and
    # a few C1). A deleted LEADING character moves the span's `offsets_original().0`
    # off 0, which is the condition Metaspace's `prepend_scheme="first"` tests — so
    # these two cases decide whether the U+2581 prefix is still added.
    ("leading deleted control char", "\x01hello"),
    ("interior deleted control char", "a\x01b"),
    ("trailing deleted control char", "hello\x01"),
    ("zero width space", "a​b"),
    # --- unigram fallback / specials --------------------------------------
    ("extra id token", "prefix <extra_id_0> suffix"),
    # ⚠️ An added token splits the string into spans that are normalized and
    # pre-tokenized independently. With spaces around it both readings of
    # Metaspace's `prepend_scheme="first"` (once per string vs once per span)
    # agree, because the span already begins with U+2581. These two do not.
    ("added token with no surrounding spaces", "prefix<extra_id_0>suffix"),
    ("added token first", "<extra_id_0>start"),
    ("eos literal", "before </s> after"),
    ("unk-ish rare glyph", "\U00010330\U00010331"),  # Gothic — outside the T5 vocab
    ("digits", "1234567890 42 3.14"),
    ("mixed case run", "CamelCase UPPER lower"),
    ("underscore words", "score_8 score_1 extra_long_tag_name"),
    ("url-ish", "https://example.com/path?q=1"),
    # --- length: past the llm_adapter's 512-row pad ------------------------
    ("long prompt over 512 t5 tokens", ", ".join(f"detailed tag number {i}" for i in range(160))),
]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(OUR_ARGV)

    tok = anima_te.AnimaTokenizer()

    # Sanity: the two sub-tokenizers must be configured the way the Zig port
    # assumes. Each of these silently changes the id stream if it moves.
    t5 = tok.t5xxl
    q3 = tok.qwen3_06b
    checks = {
        "t5.tokens_start": (t5.tokens_start, 0),
        "t5.adds_end_token": (t5.tokenizer_adds_end_token, True),
        "t5.start_token": (t5.start_token, None),
        # ⚠️ Two DIFFERENT uses of `</s>` (id 1), and only one of them survives.
        # Per segment, `input_ids[0:-1]` drops the one TemplateProcessing appended.
        # Once, at the end of the chunk, `tokenize_with_weights` appends
        # `self.end_token` — which `SDTokenizer.__init__` resolved to
        # `tokenizer('')["input_ids"][0]` == 1. So the id stream is
        # concat(segment ids without eos) ++ [1], not "no eos anywhere".
        "t5.end_token": (t5.end_token, 1),
        "t5.min_length": (t5.min_length, 1),
        "t5.pad_to_max_length": (t5.pad_to_max_length, False),
        "t5.max_word_length": (t5.max_word_length, 8),
        "qwen.tokens_start": (q3.tokens_start, 0),
        "qwen.adds_end_token": (q3.tokenizer_adds_end_token, False),
        "qwen.start_token": (q3.start_token, None),
        "qwen.end_token": (q3.end_token, None),
        "qwen.pad_to_max_length": (q3.pad_to_max_length, False),
        # `min_length=1` is the only padding either branch does: an empty prompt
        # yields one `<|endoftext|>` rather than a zero-length id list.
        "qwen.min_length": (q3.min_length, 1),
        "qwen.pad_token": (q3.pad_token, 151643),
    }
    for name, (got, want) in checks.items():
        if got != want:
            raise SystemExit(f"{name} is {got!r}, expected {want!r} — the Zig port assumes the latter")
    if t5.max_length < 1 << 20 or q3.max_length < 1 << 20:
        raise SystemExit("a sub-tokenizer gained a real max_length; chunking would now apply")

    cases = []
    for label, text in CASES:
        out = tok.tokenize_with_weights(text)
        # One chunk each, per the max_length assertion above.
        if len(out["t5xxl"]) != 1 or len(out["qwen3_06b"]) != 1:
            raise SystemExit(f"{label!r} produced multiple chunks: "
                             f"t5={len(out['t5xxl'])} qwen={len(out['qwen3_06b'])}")
        t5_pairs = out["t5xxl"][0]
        q_pairs = out["qwen3_06b"][0]
        if any(w != 1.0 for _, w in q_pairs):
            raise SystemExit(f"{label!r}: a qwen3 weight is not 1.0 — AnimaTokenizer changed")
        cases.append({
            "label": label,
            "text": text,
            "t5_ids": [int(t) for t, _ in t5_pairs],
            "t5_weights": [float(w) for _, w in t5_pairs],
            "qwen_ids": [int(t) for t, _ in q_pairs],
        })

    # Teeth check: at least one case must distinguish per-segment tokenization
    # from whole-string tokenization, and at least one must carry a non-1.0
    # weight. Without both, the fixture passes against a wrong port.
    seg = tok.t5xxl.tokenize_with_weights("a cat(s:1.1) on a mat")[0]
    whole = tok.t5xxl.tokenize_with_weights("a cats on a mat")[0]
    if [t for t, _ in seg] == [t for t, _ in whole]:
        raise SystemExit("per-segment tokenization is indistinguishable here — add a case that splits ids")
    if not any(w != 1.0 for c in cases for w in c["t5_weights"]):
        raise SystemExit("no case carries a T5 weight != 1.0")

    doc = {
        "_comment": "generated by tools/gen_anima_prompt_fixtures.py from ComfyUI's AnimaTokenizer",
        "t5_tokenizer_sha256": sha256(T5_JSON),
        "t5_unk_id": 2,
        "qwen_pad_token": 151643,
        "llm_adapter_min_rows": 512,
        "cases": cases,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")

    n_w = sum(1 for c in cases for w in c["t5_weights"] if w != 1.0)
    print(f"wrote {args.out}")
    print(f"  {len(cases)} cases, {n_w} weighted T5 tokens")
    print(f"  longest: t5={max(len(c['t5_ids']) for c in cases)} "
          f"qwen={max(len(c['qwen_ids']) for c in cases)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
