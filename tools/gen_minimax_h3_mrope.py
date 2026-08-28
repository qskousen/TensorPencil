#!/usr/bin/env python3
"""Pin Qwen3-VL's multimodal rope: the position-id construction AND the
interleaved frequency assignment.

Both halves are intricate enough to port rather than re-derive, and both are
silent when wrong -- a rope that is subtly off produces a perfectly fluent
conditioning that describes the wrong picture.

`qwen2vl_mrope_position_ids` builds `[3][seq]` (T, H, W) ids where:
  - text runs sequentially on all three axes;
  - inside an image span, axis 0 (T) is CONSTANT, axis 1 is the merged ROW index
    (each value repeated across a row) and axis 2 the merged COLUMN index (the
    range tiled);
  - the block consumes `max(gh, gw) / 2` positions on the timeline WHATEVER its
    token count is, so text after it resumes at `start + len_max + offset` and
    `offset` accumulates `len_max - (end - start)` per image. With two or more
    references the timeline and the token index drift apart, which is why a
    two-image case is here.

`precompute_freqs_cis` then assigns the three axes to the `head_dim / 2`
frequency slots ROUND-ROBIN, not in contiguous sections:
  - slot k < 3 * rope_dims[1]  ->  axis (k % 3)
  - otherwise                  ->  axis 0
so `rope_dims = [24, 20, 20]` gives axis 0 slots 0,3,..,57 PLUS the tail 60..63.
Reading "24/20/20" as three contiguous blocks is the plausible wrong answer, and
it agrees with the right one exactly when all three axes carry the same position,
i.e. for a text-only prompt.

Emits src/models/assets/minimax_h3_mrope.safetensors.

Usage (bounded, though this one is tiny):
    systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 \
        /home/qt/genai/comfyui/nvenv/bin/python tools/gen_minimax_h3_mrope.py
"""

import json
import os
import sys

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "models", "assets", "minimax_h3_mrope.safetensors",
)

COMFY = "/home/qt/genai/comfyui"
sys.path.insert(0, COMFY)
os.chdir(COMFY)

import torch  # noqa: E402
import safetensors.torch  # noqa: E402

from comfy.text_encoders.qwen_vl import qwen2vl_mrope_position_ids  # noqa: E402
from comfy.text_encoders.llama import precompute_freqs_cis  # noqa: E402

HEAD_DIM = 128
ROPE_DIMS = [24, 20, 20]
THETA = 5000000.0

# (name, seq_len, [(index, grid_h, grid_w)]). Grids are PRE-merge, so a 6x10 grid
# is 15 merged tokens. One image, then two, because the `offset` drift only shows
# up from the second one onward.
CASES = [
    ("one_image", 40, [(6, 6, 10)]),
    ("two_images", 60, [(4, 4, 6), (30, 8, 4)]),
    ("text_only", 24, []),
]


def main():
    torch.manual_seed(20260828)
    tensors = {}
    meta_cases = {}

    for name, seq, imgs in CASES:
        embeds_info = []
        for index, gh, gw in imgs:
            size = (gh // 2) * (gw // 2)
            grid = torch.tensor([[1, gh, gw]], dtype=torch.long)
            embeds_info.append({"type": "image", "index": index, "size": size,
                                "extra": {"grid": grid}})

        pos = qwen2vl_mrope_position_ids(embeds_info, seq, torch.device("cpu"))
        if pos is None:
            # No images: the LLM falls back to a plain sequential 1-D rope, which
            # is what the text-only path we already run does.
            assert not imgs
            pos = torch.arange(0, seq, dtype=torch.float32).unsqueeze(0).repeat(3, 1)
            had_ids = 0
        else:
            had_ids = 1
        pos = pos.float()
        assert pos.shape == (3, seq), pos.shape

        cos, sin, nsin = precompute_freqs_cis(
            HEAD_DIM, pos, THETA, None, ROPE_DIMS, device=torch.device("cpu"),
            interleaved_mrope=True)
        # cos is [1, seq, head_dim]; the engine keeps a [seq][head_dim/2] table and
        # mirrors it, so store the first half only and check the mirror here.
        cos = cos.reshape(-1, HEAD_DIM)
        half = HEAD_DIM // 2
        assert torch.allclose(cos[:, :half], cos[:, half:], atol=1e-6), "cos is not mirrored"
        sin_full = torch.cat([sin.reshape(-1, half), -nsin.reshape(-1, half)], dim=-1)
        assert torch.allclose(sin_full[:, :half], sin_full[:, half:], atol=1e-6), "sin is not mirrored"

        tensors["pos.%s" % name] = pos
        tensors["cos.%s" % name] = cos[:, :half].contiguous()
        tensors["sin.%s" % name] = sin_full[:, :half].contiguous()
        meta_cases[name] = dict(seq=seq, images=[list(i) for i in imgs], had_ids=had_ids)
        print("  %-11s seq %d, %d image(s), pos axis0[0:4]=%s" %
              (name, seq, len(imgs), [int(v) for v in pos[0, :4].tolist()]))

    # --- teeth -------------------------------------------------------------
    def rel(a, b):
        return float((a - b).norm() / max(float(b.norm()), 1e-12))

    # 1. The three axes really differ inside an image span, or nothing here tests
    #    the axis assignment at all.
    p = tensors["pos.one_image"]
    idx, gh, gw = CASES[0][2][0]
    size = (gh // 2) * (gw // 2)
    span = p[:, idx:idx + size]
    assert span[0].std() == 0, "axis 0 should be constant across the block"
    assert span[1].std() > 0 and span[2].std() > 0, "axes 1 and 2 should vary"
    assert not torch.allclose(span[1], span[2]), "axes 1 and 2 are identical"

    # 2. The timeline drift: with two images the text after the second does NOT
    #    resume at its token index.
    p2 = tensors["pos.two_images"]
    last_idx, lgh, lgw = CASES[1][2][-1]
    last_size = (lgh // 2) * (lgw // 2)
    after = int(p2[0, last_idx + last_size])
    assert after != last_idx + last_size, "no timeline drift, so the offset is untested"
    print("  drift: token %d resumes at timeline position %d" % (last_idx + last_size, after))

    # 3. INTERLEAVED vs SECTIONED. The plausible wrong reading of [24,20,20] is
    #    three contiguous blocks; it must differ here and must AGREE on text-only.
    cos_i, _, _ = precompute_freqs_cis(HEAD_DIM, p, THETA, None, ROPE_DIMS,
                                       device=torch.device("cpu"), interleaved_mrope=True)
    cos_s, _, _ = precompute_freqs_cis(HEAD_DIM, p, THETA, None, ROPE_DIMS,
                                       device=torch.device("cpu"), interleaved_mrope=False)
    d_mode = rel(cos_s.reshape(-1), cos_i.reshape(-1))
    assert d_mode > 1e-3, "interleaved and sectioned mrope are indistinguishable (%.2e)" % d_mode

    p_text = tensors["pos.text_only"]
    ci, _, _ = precompute_freqs_cis(HEAD_DIM, p_text, THETA, None, ROPE_DIMS,
                                    device=torch.device("cpu"), interleaved_mrope=True)
    cs, _, _ = precompute_freqs_cis(HEAD_DIM, p_text, THETA, None, ROPE_DIMS,
                                    device=torch.device("cpu"), interleaved_mrope=False)
    d_text = rel(cs.reshape(-1), ci.reshape(-1))
    assert d_text < 1e-6, "the two modes should agree when all axes match (%.2e)" % d_text

    meta = {
        "config": json.dumps(dict(head_dim=HEAD_DIM, rope_dims=ROPE_DIMS, theta=THETA)),
        "cases": json.dumps(meta_cases),
        "note": "generated by tools/gen_minimax_h3_mrope.py from "
                "comfy.text_encoders.qwen_vl.qwen2vl_mrope_position_ids + "
                "llama.precompute_freqs_cis(interleaved_mrope=True); do not hand-edit",
    }
    tensors = {k: v.contiguous() for k, v in tensors.items()}
    safetensors.torch.save_file(tensors, OUT, metadata=meta)
    print("wrote %s: %d tensors, %.3f MB" % (OUT, len(tensors), os.path.getsize(OUT) / 1048576))
    print("  sensitivity: interleaved vs sectioned %.4f (text-only agreement %.2e)" % (d_mode, d_text))


if __name__ == "__main__":
    sys.exit(main())
