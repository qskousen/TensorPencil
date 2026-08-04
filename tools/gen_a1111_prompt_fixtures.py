#!/usr/bin/env python3
"""Reference fixtures for AUTOMATIC1111-style prompt parsing.

The reference is **A1111's own source, executed** — not a re-derivation. Its
`prompt_parser` module depends only on `re` + `lark`, so it imports and runs
standalone, and its Lark grammar is the authority on the `[a:b:when]` / `[a|b]`
scheduling that no amount of reasoning settles.

⚠️ **A1111 is AGPL-3.0, so its source is NOT vendored into this repository.** The
generator downloads it to a temp directory at run time (a maintainer-only tool, not
distributed) and records the upstream sha256 in the fixture metadata so a
regenerated fixture is traceable to an exact revision. The emitted JSON is data
derived from running it, which is what the Zig tests consume.

Three fixture groups, because three independent things have to agree:

  1. `a1111_attention` — `parse_prompt_attention`: the emphasis parser. Includes
     every upstream doctest.
  2. `a1111_schedule` — `get_learned_conditioning_prompt_schedules`: per-step prompt
     editing and alternating words. Includes every upstream doctest.
  3. `a1111_chunks` — `tokenize_line`'s chunker: BREAK boundaries and
     `comma_padding_backtrack`, down to per-token multipliers.

⚠️ **Four ways A1111 differs from ComfyUI, all load-bearing** (see
`core/prompt_a1111.zig`): `(x:w)` MULTIPLIES the enclosed range rather than replacing
its weight, so `(((house:1.3))` is 1.573; `[x]` is 1/1.1 de-emphasis where ComfyUI
reads it as literal text; `BREAK` forces a chunk boundary where ComfyUI tokenizes the
word; and a chunk boundary backtracks to the last comma. Padding, by contrast,
happens to AGREE — both pad CLIP-L with EOS and CLIP-G with 0.

Usage (ComfyUI's `nvenv`, which has both lark and transformers):
    /home/qt/genai/comfyui/nvenv/bin/python tools/gen_a1111_prompt_fixtures.py
"""

import hashlib
import importlib.util
import json
import os
import re
import sys
import tempfile
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "src", "core", "assets", "clip_tokenizer", "fixtures_a1111.json")
UPSTREAM = "https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/modules/prompt_parser.py"

# A1111 defaults (`modules/shared_options.py`): emphasis "Original", and a chunk
# boundary backtracks to a comma up to this many tokens behind.
COMMA_PADDING_BACKTRACK = 20
CHUNK_LENGTH = 75
ID_START, ID_END = 49406, 49407


def load_reference():
    """Download A1111's prompt_parser and import it. Returns (module, sha256)."""
    src = urllib.request.urlopen(UPSTREAM, timeout=60).read()
    sha = hashlib.sha256(src).hexdigest()
    d = tempfile.mkdtemp(prefix="a1111ref-")
    path = os.path.join(d, "a1111_prompt_parser.py")
    with open(path, "wb") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("a1111_prompt_parser", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, sha


# --- A1111's chunker, transcribed from modules/sd_hijack_clip.py -------------
# Kept as a faithful transcription rather than an import: `sd_hijack_clip` pulls in
# the whole webui (shared opts, devices, the model wrapper), while the loop itself is
# self-contained. The `parsed` list it consumes comes from the REAL parser above.


class Chunk:
    def __init__(self):
        self.tokens = []
        self.multipliers = []


def tokenize_line(parsed, tok, id_pad):
    """`FrozenCLIPEmbedderWithCustomWordsBase.tokenize_line`, minus embeddings."""
    comma_token = tok.get_vocab().get(",</w>")
    # Each weighted SEGMENT is tokenized on its own (content ids only) — the same
    # unit ComfyUI uses, so the two differ in weighting and chunking, not in BPE.
    tokenized = [tok(text)["input_ids"][1:-1] for text, _ in parsed]

    chunks, chunk, last_comma = [], Chunk(), -1

    def next_chunk():
        nonlocal chunk, last_comma
        to_add = CHUNK_LENGTH - len(chunk.tokens)
        if to_add > 0:
            chunk.tokens += [ID_END] * to_add
            chunk.multipliers += [1.0] * to_add
        chunk.tokens = [ID_START] + chunk.tokens + [ID_END]
        chunk.multipliers = [1.0] + chunk.multipliers + [1.0]
        last_comma = -1
        chunks.append(chunk)
        chunk = Chunk()

    for tokens, (text, weight) in zip(tokenized, parsed):
        if text == "BREAK" and weight == -1:
            next_chunk()
            continue
        position = 0
        while position < len(tokens):
            token = tokens[position]
            if token == comma_token:
                last_comma = len(chunk.tokens)
            elif (COMMA_PADDING_BACKTRACK != 0
                  and len(chunk.tokens) == CHUNK_LENGTH
                  and last_comma != -1
                  and len(chunk.tokens) - last_comma <= COMMA_PADDING_BACKTRACK):
                break_location = last_comma + 1
                reloc_tokens = chunk.tokens[break_location:]
                reloc_mults = chunk.multipliers[break_location:]
                chunk.tokens = chunk.tokens[:break_location]
                chunk.multipliers = chunk.multipliers[:break_location]
                next_chunk()
                chunk.tokens = reloc_tokens
                chunk.multipliers = reloc_mults
            if len(chunk.tokens) == CHUNK_LENGTH:
                next_chunk()
            chunk.tokens.append(token)
            chunk.multipliers.append(weight)
            position += 1

    if chunk.tokens or not chunks:
        next_chunk()

    # `process_tokens`: everything after the FIRST id_end becomes id_pad. Net effect
    # is `[BOS] content [EOS] pad…` — which is exactly ComfyUI's padding, so the two
    # engines agree here even though they arrive at it differently.
    if ID_END != id_pad:
        for c in chunks:
            i = c.tokens.index(ID_END)
            c.tokens = c.tokens[: i + 1] + [id_pad] * (len(c.tokens) - i - 1)
    return chunks


# --- cases ------------------------------------------------------------------

# Every upstream `parse_prompt_attention` doctest, plus the constructs that only
# appear in real prompts.
ATTENTION_CASES = [
    "normal text",
    "an (important) word",
    "(unbalanced",
    r"\(literal\]",
    "(unnecessary)(parens)",
    "a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).",
    "",
    "[de-emphasized]",
    "[[double de-emphasis]]",
    "(a:1.5) and [b:0.8]",          # NOTE: [b:0.8] is de-emphasis of "b:0.8"
    "(a:-1) (b:+2) (c:.5)",         # the regex accepts sign and a bare leading dot
    "(a : 1.5)",                    # whitespace around the colon
    "nested ((a:2.0) b)",           # inner absolute, then outer x1.1
    # ⚠️ These are the ONLY shapes where multiply-vs-replace diverges: an explicit
    # weight must already have another explicit weight inside its range. Without them a
    # `:w`-replaces-the-range implementation passes the whole corpus, because
    # replace-then-multiply-later coincides with multiply-then-multiply whenever the
    # inner weight is the first thing applied. Found by deliberately breaking the
    # implementation and seeing nothing fail.
    "((a:1.2):1.5)",                # multiply -> 1.8 ; replace -> 1.5
    "(x (a:1.2) y:1.5)",            # a -> 1.8, x/y -> 1.5
    "((a:1.2))",                    # 1.2 * 1.1
    "(((a:2):0.5):3)",              # 2 * 0.5 * 3 = 3.0
    "[(a:1.2):1.5]",                # parens inside brackets
    "((a:1.2) (b:0.5):2)",
    "a BREAK b",
    "a, BREAK, b",
    "BREAKfast is not a BREAK",     # \b word boundary
    r"\\ backslash",
    "(a:1.5",                       # unbalanced with a weight
    "trailing ((",
    "]stray[",
    "(a)(b)(c:2)",
    "1girl, (shiny skin:1.1), [blurry], (lens flare:0.75)",
]

# Every upstream scheduling doctest (at their documented step counts), plus the
# alternating/editing shapes a real prompt uses.
SCHEDULE_CASES = [
    ("test", 10),
    ("a [b:3]", 10),
    ("a [b: 3]", 10),
    ("a [[[b]]:2]", 10),
    ("[(a:2):3]", 10),
    ("a [b : c : 1] d", 10),
    ("a[b:[c:d:2]:1]e", 10),
    ("a [unbalanced", 10),
    ("a [b:.5] c", 10),
    ("a [{b|d{:.5] c", 10),
    ("((a][:b:c [d:3]", 10),
    ("[a|(b:1.1)]", 10),
    ("[fe|]male", 10),
    ("[fe|||]male", 10),
    # The upstream module docstring's worked example.
    ("fantasy landscape with a [mountain:lake:0.25] and [an oak:a christmas tree:0.75]"
     "[ in foreground::0.6][: in background:0.25] [shoddy:masterful:0.5]", 100),
    # Real shapes at real step counts.
    ("[white|light blue] off-shoulder shirt", 35),
    ("[white|light blue] off-shoulder shirt", 4),
    ("a [cat:dog:0.5] on a mat", 20),
    ("a [cat:dog:10] on a mat", 20),
    ("a [cat::0.3] b", 20),
    ("a [:cat:0.3] b", 20),
    ("no scheduling here", 35),
    ("[a|b|c]", 7),
    ("[a:b:0]", 10),        # when < 1 is dropped from the step set
    ("[a:b:1]", 10),
    ("[a:b:99]", 10),       # clamped to steps
    ("nested [a|[b:c:0.5]]", 8),
    ("", 10),
    # ⚠️ A bare `|` is legal ONLY directly inside a `[…]` — the grammar has no other
    # rule for it, so anywhere else the whole prompt fails to LEX and upstream falls
    # back to a single verbatim entry. That includes NovelAI-style `{a|b}`, which real
    # prompts do contain. These cases pin the rule; the upstream doctest
    # `a [{b|d{:.5] c` documents the scheduled reading but FAILS at this revision
    # (1 of its 24 doctests), so the fixture records what the code does.
    ("a|b", 10),
    ("(a|b)", 10),
    ("[(a|b)]", 10),
    ("{a|b}", 10),
    ("a [b|c] d|e", 10),
    (r"a \| b", 10),
    ("[a:[b|c]:0.5]", 10),
    ("a [b|c", 10),
    ("[[a|b]]", 4),
    # An alternation OPTION cannot contain a top-level `:` (`prompt` has no bare-colon
    # alternative), so these all fall back to verbatim — while `[a|(b:0.5)]` parses,
    # because there the `:` is inside parens.
    ("[a|b:0.5]", 4),
    ("[a|b:c]", 4),
    ("[a:1|b]", 4),
    ("[a|b:]", 4),
    ("[a|b|c:2]", 4),
    ("[a|(b:0.5)]", 4),
]

# Chunking: the interesting cases are at and around the 75-token boundary, with and
# without a nearby comma, and with BREAK.
FILL = ", ".join(["cat"] * 40)          # 40 "cat" + 39 commas = 79 content tokens
CHUNK_CASES = [
    "a red cat",
    "",
    "a BREAK b",
    "(a:1.2) BREAK (b:0.8)",
    FILL,                                # boundary lands right after a comma
    FILL + " and a very long tail of words to push well past one chunk",
    "verylongunbrokenword " * 60,        # no commas at all -> hard cut, no backtrack
    "(emphasis:1.3) " + FILL,
    # ⚠️ The backtrack fires only when the token arriving AT the 75-token boundary is not
    # itself a comma. For an alternating `word, word, …` prompt the boundary always lands
    # on the comma, so `FILL` above does NOT exercise it — verified by running the
    # reference with the setting on and off and getting identical output. These two do:
    # 35x"cat, " puts a comma at index 69, then unpunctuated words carry the boundary
    # past it (distance 6 <= 20, relocates); the second pushes the comma to index 49
    # (distance 26 > 20, so it must NOT relocate).
    ("cat, " * 35) + ("cat " * 12),
    ("cat, " * 25) + ("cat " * 30),
    "1girl, original, general, blonde hair, long hair, ponytail, light blue eyes, "
    "(shiny skin:1.1), gentle smile, bare shoulders, collarbone, off-shoulder shirt, "
    "sheer sleeves, denim short shorts, holding straw hat, looking back, side angle, "
    "upper body, beach, ocean, summer, sunny day, (lens flare:0.75), breeze, BREAK, "
    "masterpiece, best quality, very aesthetic, high contrast, vibrant, highres, "
    "year 2024, newest,",
]


def fuzz_cases(n=240, seed=20260803):
    """A seeded random corpus over the alphabet that actually stresses the grammar.

    A hand-written recursive-descent parser cannot be equivalent to Lark's Earley parser
    for arbitrary input; this is how that claim gets a number instead of a shrug. The
    seed is fixed so the committed fixture is reproducible.
    """
    import random

    rng = random.Random(seed)
    atoms = ["a", "b", "cat", "dog", " ", ", ", "x y", "1girl", "{", "}", "\\(", "\\|"]
    frags = ["(", ")", "[", "]", ":", "|", ":0.5", ":2", ":1.3", "BREAK", ".5"]
    out = []
    for _ in range(n):
        k = rng.randint(1, 9)
        parts = []
        for _ in range(k):
            parts.append(rng.choice(atoms) if rng.random() < 0.55 else rng.choice(frags))
        out.append(("".join(parts), rng.choice([4, 10, 20])))
    return out


def main():
    pp, sha = load_reference()
    from transformers import CLIPTokenizer

    tok = CLIPTokenizer(
        os.path.join(REPO, "src/core/assets/clip_tokenizer/vocab.json"),
        os.path.join(REPO, "src/core/assets/clip_tokenizer/merges.txt"),
    )

    attention = []
    for text in ATTENTION_CASES:
        parts = pp.parse_prompt_attention(text)
        attention.append({
            "text": text,
            "parts": [{"text": t, "weight": float(w)} for t, w in parts],
        })

    schedule = []
    for text, steps in SCHEDULE_CASES + fuzz_cases():
        entries = pp.get_learned_conditioning_prompt_schedules([text], steps)[0]
        schedule.append({
            "text": text,
            "steps": steps,
            "entries": [{"until": int(end), "text": t} for end, t in entries],
        })

    # Self-check: a corpus that does not distinguish `comma_padding_backtrack` on from
    # off cannot pin the feature, and a Zig implementation that simply omitted it would
    # pass. Assert the coverage rather than hope for it.
    global COMMA_PADDING_BACKTRACK
    triggering = []
    for text in CHUNK_CASES:
        parsed = pp.parse_prompt_attention(text)
        outs = {}
        for bt in (20, 0):
            COMMA_PADDING_BACKTRACK = bt
            outs[bt] = [c.tokens for c in tokenize_line(parsed, tok, ID_END)]
        COMMA_PADDING_BACKTRACK = 20
        if outs[20] != outs[0]:
            triggering.append(text[:40])
    if not triggering:
        raise SystemExit("no chunk case exercises comma_padding_backtrack — fixture would not pin it")

    chunks_out = []
    for text in CHUNK_CASES:
        parsed = pp.parse_prompt_attention(text)
        row = {"text": text, "clip_l": [], "clip_g": []}
        for key, id_pad in (("clip_l", ID_END), ("clip_g", 0)):
            for c in tokenize_line(parsed, tok, id_pad):
                assert len(c.tokens) == 77 and len(c.multipliers) == 77, text
                row[key].append({
                    "ids": [int(t) for t in c.tokens],
                    "mults": [float(m) for m in c.multipliers],
                })
        chunks_out.append(row)

    out = {
        "_reference": {
            "upstream": UPSTREAM,
            "sha256": sha,
            "comma_padding_backtrack": COMMA_PADDING_BACKTRACK,
            "emphasis": "Original",
            "note": "A1111 (AGPL-3.0) source is fetched at generation time, never vendored.",
            "backtrack_exercised_by": triggering,
        },
        "a1111_attention": attention,
        "a1111_schedule": schedule,
        "a1111_chunks": chunks_out,
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    n_sched = sum(len(c["entries"]) for c in schedule)
    n_chunks = sum(len(c["clip_l"]) for c in chunks_out)
    print(f"wrote {OUT}")
    print(f"  reference sha256 {sha[:16]}…")
    print(f"  {len(attention)} attention cases, {len(schedule)} schedule cases "
          f"({n_sched} entries), {len(chunks_out)} chunk cases ({n_chunks} clip_l chunks)")


if __name__ == "__main__":
    main()
