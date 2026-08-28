#!/usr/bin/env python3
"""Pin MiniMax H3's packed-sequence layout by EXECUTING ComfyUI's own builder.

The layout is where H3 hides its silent wrong answers: the position grid is
area-normalized floats rather than indices, the video time axis is non-uniform,
audio rows are channel-major with `w` pinned to the frame grid's extremes, and
references push the target timeline out ahead of themselves. Re-deriving any of
that from the paper would pass a self-consistent test and render garbage, so
this imports comfy.ldm.minimax.model.PackedLayout and records what it produces.

Emits src/models/assets/minimax_h3_layout.json, read by the layout tests in
src/models/minimax_h3.zig.

Usage:
    /home/qt/genai/comfyui/nvenv/bin/python tools/gen_minimax_h3_fixtures.py

Needs no checkpoint: the layout is a function of shapes alone.
"""

import json
import os
import sys

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "models", "assets", "minimax_h3_layout.json",
)

COMFY = "/home/qt/genai/comfyui"
sys.path.insert(0, COMFY)
os.chdir(COMFY)

import torch  # noqa: E402

from comfy.ldm.minimax.model import PackedLayout  # noqa: E402

# Reference blocks are described to PackedLayout by shape only; the latents
# themselves never enter the layout, so these dicts carry exactly the fields it
# reads. "video_audio" and "video" are the same code path there, which is why
# the Zig side collapses them into one kind with audio_t possibly zero.
def ref_image(h, w):
    return {"kind": "image", "latent_h": h, "latent_w": w}


def ref_audio(t):
    return {"kind": "audio", "ref_audio_t": t}


def ref_video(t, h, w, audio_t=0):
    return {
        "kind": "video_audio" if audio_t else "video",
        "latent_t": t, "latent_h": h, "latent_w": w, "ref_audio_t": audio_t,
    }


def keyframe(frame_index, latent_t=0, audio_t=0):
    kf = {"resolved_frame_index": frame_index}
    if latent_t:
        # only .shape[2] is read
        kf["latent"] = torch.zeros(1, 24, latent_t, 2, 2)
    if audio_t:
        # only .shape[-1] is read
        kf["audio_latent"] = torch.zeros(1, 32, 2, audio_t)
    return kf


# Small enough to record every position, chosen to separate the things that are
# easy to get wrong: square vs non-square (the area normalization), one vs many
# video tokens (the non-uniform time axis), and each reference kind.
CASES = [
    ("t2va dev shape", dict(text_len=3, latent_t=2, latent_h=16, latent_w=16, audio_t=4)),
    ("t2va single video token", dict(text_len=1, latent_t=1, latent_h=16, latent_w=16, audio_t=1)),
    ("t2va six tokens crosses the 5-token cycle",
     dict(text_len=2, latent_t=6, latent_h=8, latent_w=8, audio_t=3)),
    ("t2va wide canvas", dict(text_len=2, latent_t=2, latent_h=16, latent_w=32, audio_t=4)),
    ("t2va tall canvas", dict(text_len=2, latent_t=2, latent_h=32, latent_w=16, audio_t=4)),
    ("fl2va first frame", dict(text_len=2, latent_t=3, latent_h=16, latent_w=16, audio_t=5,
                               keyframes=[keyframe(0, latent_t=1)])),
    ("fl2va first and last frame",
     dict(text_len=2, latent_t=3, latent_h=16, latent_w=16, audio_t=5,
          keyframes=[keyframe(0, latent_t=1), keyframe(22, latent_t=1)])),
    ("keyframe with audio", dict(text_len=2, latent_t=2, latent_h=16, latent_w=16, audio_t=6,
                                 keyframes=[keyframe(0, latent_t=1, audio_t=2)])),
    ("ref2va one image", dict(text_len=2, latent_t=2, latent_h=16, latent_w=16, audio_t=4,
                              refs=[ref_image(16, 16)])),
    ("ref2va image of a different shape",
     dict(text_len=2, latent_t=2, latent_h=16, latent_w=16, audio_t=4,
          refs=[ref_image(32, 16)])),
    ("ref2va standalone audio", dict(text_len=2, latent_t=2, latent_h=16, latent_w=16, audio_t=4,
                                     refs=[ref_audio(3)])),
    ("ref2va silent video", dict(text_len=2, latent_t=2, latent_h=16, latent_w=16, audio_t=4,
                                 refs=[ref_video(2, 16, 16)])),
    ("ref2va video with soundtrack",
     dict(text_len=2, latent_t=2, latent_h=16, latent_w=16, audio_t=4,
          refs=[ref_video(2, 16, 16, audio_t=3)])),
    # the soundtrack is longer than the video, so the block's span is the audio's
    ("ref2va soundtrack longer than its video",
     dict(text_len=2, latent_t=2, latent_h=16, latent_w=16, audio_t=8,
          refs=[ref_video(1, 16, 16, audio_t=9)])),
    ("ref2va several refs in order",
     dict(text_len=2, latent_t=2, latent_h=16, latent_w=16, audio_t=4,
          refs=[ref_image(16, 16), ref_video(2, 8, 16, audio_t=2), ref_audio(3)])),
]

# The default render, too large to record row by row: pin the arithmetic with
# the sequence length, the segment table and a checksum of the positions.
BIG_CASES = [
    ("default 1344x768 124 frames",
     dict(text_len=64, latent_t=37, latent_h=48, latent_w=84, audio_t=207)),
    ("512x512 22 frames",
     dict(text_len=32, latent_t=7, latent_h=32, latent_w=32, audio_t=37)),
]


def describe(kw):
    """The layout inputs, as the Zig side names them."""
    out = {
        "text_len": kw["text_len"],
        "latent_t": kw["latent_t"],
        "latent_h": kw["latent_h"],
        "latent_w": kw["latent_w"],
        "audio_t": kw["audio_t"],
        "keyframes": [],
        "refs": [],
    }
    for kf in kw.get("keyframes") or ():
        out["keyframes"].append({
            "frame_index": kf["resolved_frame_index"],
            "latent_t": kf["latent"].shape[2] if "latent" in kf else 0,
            "audio_t": kf["audio_latent"].shape[-1] if "audio_latent" in kf else 0,
        })
    for r in kw.get("refs") or ():
        out["refs"].append({
            # video_audio collapses into video: the reference treats them alike
            "kind": "video" if r["kind"] in ("video", "video_audio") else r["kind"],
            "latent_t": r.get("latent_t", 0),
            "latent_h": r.get("latent_h", 0),
            "latent_w": r.get("latent_w", 0),
            "audio_t": r.get("ref_audio_t", 0),
        })
    return out


def build(kw):
    return PackedLayout(
        kw["text_len"], kw["latent_t"], kw["latent_h"], kw["latent_w"], kw["audio_t"],
        keyframes=kw.get("keyframes"), refs=kw.get("refs"),
    )


def segsig(segments):
    return tuple((s["start"], s["stop"], s["kind"]) for s in segments)


def checksum(pos):
    """Order-sensitive checksum over the f64 positions.

    Deliberately not a plain sum: a sum would not notice two rows swapping,
    which is exactly the failure a permuted grid produces.
    """
    flat = pos.reshape(-1).tolist()
    acc = 0.0
    for i, v in enumerate(flat):
        acc += (i + 1) * v
    return acc


def main():
    cases = []
    for name, kw in CASES:
        layout = build(kw)
        pos = layout.position_ids
        assert pos.dtype == torch.float64, pos.dtype
        assert pos.shape == (layout.seq_len, 3), (pos.shape, layout.seq_len)
        cases.append({
            "name": name,
            "input": describe(kw),
            "seq_len": layout.seq_len,
            "segments": [{"start": a, "stop": b, "kind": k} for a, b, k in layout.segments],
            "video_cond_rows": int((~layout.img_update).sum()),
            "audio_cond_rows": int((~layout.audio_update).sum()),
            "pos": pos.reshape(-1).tolist(),
        })

    big = []
    for name, kw in BIG_CASES:
        layout = build(kw)
        big.append({
            "name": name,
            "input": describe(kw),
            "seq_len": layout.seq_len,
            "segments": [{"start": a, "stop": b, "kind": k} for a, b, k in layout.segments],
            "video_cond_rows": int((~layout.img_update).sum()),
            "audio_cond_rows": int((~layout.audio_update).sum()),
            "pos_checksum": checksum(layout.position_ids),
        })

    # A corpus with no teeth passes whatever it is compared against. Every case
    # must differ from every other in the bytes the Zig side checks, or the
    # fixture is not pinning the thing it claims to.
    sigs = {}
    for c in cases:
        sig = (c["seq_len"], segsig(c["segments"]), tuple(c["pos"]))
        assert sig not in sigs, "cases %r and %r are indistinguishable" % (sigs[sig], c["name"])
        sigs[sig] = c["name"]
    for c in big:
        sig = (c["seq_len"], segsig(c["segments"]), c["pos_checksum"])
        assert sig not in sigs, "cases %r and %r are indistinguishable" % (sigs[sig], c["name"])
        sigs[sig] = c["name"]

    # The wide and tall canvases must disagree, or the area normalization is not
    # being exercised at all.
    wide = next(c for c in cases if c["name"] == "t2va wide canvas")
    tall = next(c for c in cases if c["name"] == "t2va tall canvas")
    assert wide["pos"] != tall["pos"], "the aspect-ratio cases do not separate"

    payload = {
        "note": "generated by tools/gen_minimax_h3_fixtures.py from "
                "comfy.ldm.minimax.model.PackedLayout; do not hand-edit",
        "cases": cases,
        "big_cases": big,
    }
    with open(OUT, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
        f.write("\n")
    print("wrote %s: %d cases, %d big cases, %d bytes"
          % (OUT, len(cases), len(big), os.path.getsize(OUT)))


if __name__ == "__main__":
    sys.exit(main())
