#!/usr/bin/env python3
"""Emit the embedded T5 SentencePiece-Unigram tokenizer assets.

Source of truth is the `tokenizer.json` that ComfyUI ships at
`comfy/text_encoders/t5_tokenizer/` — a HuggingFace *fast* tokenizer
serialization of Google's T5 SentencePiece model (Apache-2.0 upstream). Two
blobs come out of it, both consumed by `@embedFile` in
`src/core/t5_tokenizer.zig`:

  vocab.bin     32100 (piece, log-prob) records in token-id order, then the ids of
                the `added_tokens` — the pieces matched VERBATIM ahead of
                normalization, which for T5 are `<pad>`/`</s>`/`<unk>` and the 100
                `<extra_id_N>`. That list is emitted rather than derived (they are
                exactly the score-0.0 pieces today, but that coincidence is not a
                contract) and rather than hardcoded in Zig, where it could drift
                from the vocabulary it indexes.
  charsmap.bin  the `Precompiled` normalizer's `precompiled_charsmap`, i.e.
                SentencePiece's serialized darts-clone double-array trie plus
                the NUL-separated replacement blob. Base64 in the JSON, raw
                here.

⚠️ This is DATA, not code: nothing here decides tokenizer behaviour. The
behaviour is pinned separately by `tools/gen_anima_prompt_fixtures.py`, which
*executes* ComfyUI's own `AnimaTokenizer`.

Run with any Python 3 — no third-party imports, deliberately, so regenerating
the vocabulary does not need a torch environment.
"""

import argparse
import base64
import hashlib
import json
import os
import struct
import sys

DEFAULT_SRC = "/home/qt/genai/comfyui/comfy/text_encoders/t5_tokenizer/tokenizer.json"
MAGIC = b"TPT5"
VERSION = 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC, help="path to tokenizer.json")
    ap.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "src",
            "core",
            "assets",
            "t5_tokenizer",
        ),
    )
    args = ap.parse_args()

    with open(args.src, "rb") as f:
        raw = f.read()
    sha = hashlib.sha256(raw).hexdigest()
    doc = json.loads(raw)

    model = doc["model"]
    if model["type"] != "Unigram":
        raise SystemExit(f"expected a Unigram model, got {model['type']!r}")
    if model.get("byte_fallback"):
        # A byte-fallback Unigram maps unknown bytes to <0xNN> pieces instead of
        # emitting <unk>; the Zig Viterbi does not implement that branch, so
        # refuse rather than silently tokenize differently.
        raise SystemExit("byte_fallback Unigram is not supported by t5_tokenizer.zig")
    unk_id = int(model["unk_id"])
    vocab = model["vocab"]

    # The normalizer must be exactly the sequence t5_tokenizer.zig implements:
    # Precompiled(charsmap) -> Strip(right) -> Replace(/ {2,}/ -> U+2581).
    norm = doc["normalizer"]
    if norm["type"] != "Sequence":
        raise SystemExit(f"unexpected normalizer {norm['type']!r}")
    kinds = [n["type"] for n in norm["normalizers"]]
    if kinds != ["Precompiled", "Strip", "Replace"]:
        raise SystemExit(f"unexpected normalizer sequence {kinds}")
    strip = norm["normalizers"][1]
    if strip["strip_left"] or not strip["strip_right"]:
        raise SystemExit(f"unexpected Strip config {strip}")
    repl = norm["normalizers"][2]
    if repl["pattern"] != {"Regex": " {2,}"} or repl["content"] != "▁":
        raise SystemExit(f"unexpected Replace config {repl}")

    pre = doc["pre_tokenizer"]
    if pre != {
        "type": "Metaspace",
        "replacement": "▁",
        "add_prefix_space": True,
        "prepend_scheme": "first",
    }:
        raise SystemExit(f"unexpected pre_tokenizer {pre}")

    # `added_tokens` are matched verbatim ahead of normalization. For T5 every one of
    # them is also a Unigram piece at the same id, so the Zig side needs only the id
    # list plus the vocabulary's own literal — assert that alignment rather than
    # emitting a second copy of the text that could drift from it.
    added_ids = sorted(int(a["id"]) for a in doc["added_tokens"])
    for a in doc["added_tokens"]:
        if vocab[a["id"]][0] != a["content"]:
            raise SystemExit(
                f"added token {a['content']!r} (id {a['id']}) is not vocab[{a['id']}]"
            )
        # A `normalized: True` added token would be normalized before matching,
        # which the Zig side does not implement. None of T5's are.
        if a.get("normalized"):
            raise SystemExit(f"added token {a['content']!r} is `normalized`; unsupported")

    os.makedirs(args.out, exist_ok=True)

    buf = bytearray()
    buf += MAGIC
    buf += struct.pack("<IIII", VERSION, len(vocab), unk_id, len(added_ids))
    for piece, score in vocab:
        b = piece.encode("utf-8")
        if len(b) > 255:
            raise SystemExit(f"piece {piece!r} exceeds the u8 length field")
        buf += struct.pack("<fB", score, len(b))
        buf += b
    for i in added_ids:
        buf += struct.pack("<I", i)
    vocab_path = os.path.join(args.out, "vocab.bin")
    with open(vocab_path, "wb") as f:
        f.write(buf)

    charsmap = base64.b64decode(norm["normalizers"][0]["precompiled_charsmap"])
    # Format check: a u32 trie length followed by that many bytes of u32 trie
    # units, then the replacement blob. Getting this wrong is a wrong answer
    # rather than a crash, so validate the framing here where it is cheap.
    (trie_bytes,) = struct.unpack_from("<I", charsmap, 0)
    if trie_bytes % 4 or 4 + trie_bytes > len(charsmap):
        raise SystemExit(f"malformed charsmap framing: trie_bytes={trie_bytes}")
    charsmap_path = os.path.join(args.out, "charsmap.bin")
    with open(charsmap_path, "wb") as f:
        f.write(charsmap)

    with open(os.path.join(args.out, "SOURCE.txt"), "w") as f:
        f.write(
            "generated by tools/gen_t5_tokenizer_assets.py\n"
            f"source: {args.src}\n"
            f"sha256: {sha}\n"
            f"pieces: {len(vocab)}  unk_id: {unk_id}  added: {len(added_ids)}\n"
            f"charsmap: {len(charsmap)} bytes ({trie_bytes} of trie)\n"
        )

    max_piece = max(len(p.encode()) for p, _ in vocab)
    print(f"vocab.bin    {len(buf):>9} bytes  ({len(vocab)} pieces, max {max_piece} B, "
          f"{len(added_ids)} added)")
    print(f"charsmap.bin {len(charsmap):>9} bytes  ({trie_bytes} of trie)")
    print(f"source sha256 {sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
