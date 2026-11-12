#!/usr/bin/env python3
"""Pin MiniMax-H3-Image's DiT forward by EXECUTING ComfyUI's own model at a tiny width.

The image model is the H3 trunk without the audio stream, packed as
[text | reference images | target image], with a PDD bank of output heads mixed
per step over the checkpoint's own sigma grid. This builds ComfyUI's
`MiniMaxH3ImageModel` at the video fixture's toy width (same seeded weight
init, tools/gen_minimax_h3_forward.py) with a 3-head bank and one reference
image, and records the output at three steps chosen to hit each kind of head
interval: [0, 1) (head 0 alone), [0, 2) (head 0 plus one offset) and [1, 3)
(head 0 plus two weighted offsets).

The three steps turn the reference's condition-row noise off
(`visual_cond_noise_aug = 1.0`) so they pin the network alone; `out.aug` repeats
step 0 with it at its default and a seed. The reference rows sit at the pinned
0.999 timestep either way.

Emits src/models/assets/minimax_h3_image_forward.safetensors, read by the image
forward test in src/models/minimax_h3.zig.

Usage:
    /home/qt/genai/comfyui/nvenv/bin/python tools/gen_minimax_h3_image_forward.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_minimax_h3_forward as base  # noqa: E402  (sets up ComfyUI's import path and attention)

import torch  # noqa: E402
import safetensors.torch  # noqa: E402

import comfy.ops  # noqa: E402
from comfy.ldm.minimax.model import MiniMaxH3ImageModel, PackedLayout  # noqa: E402

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "models", "assets", "minimax_h3_image_forward.safetensors",
)

CFG = {k: v for k, v in base.CFG.items() if k != "audio_latents_dim"}
PDD_SIGMAS = [1.0, 0.8, 0.45, 0.0]
PDD_STEPS = len(PDD_SIGMAS) - 1

TEXT_LEN = 4
LATENT_H = 4
LATENT_W = 6
REF_H = 6
REF_W = 4

AUG_SEED = 7

# (sigma, the sampler's schedule, the head interval the reference must pick)
STEPS = [
    (1.0, [1.0, 0.8, 0.45, 0.0], (0, 1)),
    (1.0, [1.0, 0.45, 0.0], (0, 2)),
    (0.8, [1.0, 0.8, 0.0], (1, 3)),
]


def main():
    torch.manual_seed(20261110)
    model = MiniMaxH3ImageModel(PDD_SIGMAS, pdd_steps=PDD_STEPS, operations=comfy.ops.disable_weight_init,
                                dtype=torch.float32, **CFG)
    model.eval()
    sd = base.init_weights(model)
    assert model.final_layer.audio_out is None and not hasattr(model, "audio_patch_proj")
    assert sd["final_layer.video_out.weight"].shape[0] == 96 * PDD_STEPS

    image = torch.randn(1, CFG["latents_dim"], 1, LATENT_H, LATENT_W, dtype=torch.float32)
    ref = torch.randn(1, CFG["latents_dim"], 1, REF_H, REF_W, dtype=torch.float32)
    text = torch.randn(1, TEXT_LEN, CFG["text_dim"], dtype=torch.float32)
    with torch.inference_mode():
        refined = model.preprocess_text_embeds(text)

    refs = [{"kind": "image", "latent_h": REF_H, "latent_w": REF_W, "latent": ref}]
    layout = PackedLayout(TEXT_LEN, 1, LATENT_H, LATENT_W, 0, refs=refs)
    assert [k for _, _, k in layout.segments] == ["text", "ref_img", "video"], layout.segments
    payload = {"layout": layout, "seed": 0, "cond_video_latents": [ref], "visual_cond_noise_aug": 1.0}

    # The head interval each step picks, by the reference's own selection code.
    picked = []
    orig_pdd = base.h3._pdd_head

    def spy(head, h, grid, start, stop):
        picked.append((start, stop))
        return orig_pdd(head, h, grid, start, stop)

    base.h3._pdd_head = spy
    outs = []
    try:
        for sigma, schedule, want in STEPS:
            opts = {"sample_sigmas": torch.tensor(schedule, dtype=torch.float32)}
            with torch.inference_mode():
                out = model._forward(image, torch.tensor([sigma * 1000.0], dtype=torch.float32), refined,
                                     transformer_options=opts, minimax_payload=payload)
            assert picked[-1] == want, (sigma, schedule, picked[-1], want)
            assert out.shape == image.shape and torch.isfinite(out).all()
            outs.append(out)
    finally:
        base.h3._pdd_head = orig_pdd

    # Teeth: steps at the same sigma with different intervals must differ, or the
    # fixture cannot tell a wrong head mix from a right one.
    rel = float((outs[0] - outs[1]).norm() / outs[0].norm())
    assert rel > 1e-2, "the [0,1) and [0,2) heads are indistinguishable (rel %.2e)" % rel
    print("  head-interval sensitivity: rel %.4f" % rel)
    # and the reference rows must matter
    with torch.inference_mode():
        alt = model._forward(image, torch.tensor([1000.0]), refined,
                             transformer_options={"sample_sigmas": torch.tensor(STEPS[0][1])},
                             minimax_payload=dict(payload, cond_video_latents=[ref * 0.5]))
    rel = float((alt - outs[0]).norm() / outs[0].norm())
    assert rel > 1e-2, "the reference rows do not reach the output (rel %.2e)" % rel
    print("  reference sensitivity: rel %.4f" % rel)

    # The reference's own condition noise (`_cond_video_rows`) at its default
    # strength, seeded: step 0's settings with the augmentation left on.
    aug_payload = {k: v for k, v in payload.items() if k != "visual_cond_noise_aug"}
    aug_payload["seed"] = AUG_SEED
    with torch.inference_mode():
        aug_out = model._forward(image, torch.tensor([1000.0]), refined,
                                 transformer_options={"sample_sigmas": torch.tensor(STEPS[0][1])},
                                 minimax_payload=aug_payload)
    rel = float((aug_out - outs[0]).norm() / outs[0].norm())
    assert rel > 1e-4, "the condition noise is invisible (rel %.2e)" % rel
    print("  condition-noise sensitivity: rel %.5f" % rel)

    tensors = dict(sd)
    tensors["out.aug"] = aug_out
    tensors["in.image"] = image
    tensors["in.ref"] = ref
    tensors["in.text"] = text
    tensors["in.refined"] = refined
    for i, o in enumerate(outs):
        tensors["out.%d" % i] = o
    meta = {
        # read by Config.detect, as the real checkpoint's is
        "config": json.dumps({"transformer": {"pdd_steps": PDD_STEPS, "pdd_sigmas": PDD_SIGMAS}}),
        "shape": json.dumps({"text_len": TEXT_LEN, "latent_h": LATENT_H, "latent_w": LATENT_W,
                             "ref_h": REF_H, "ref_w": REF_W}),
        "steps": json.dumps(STEPS),
        "note": "generated by tools/gen_minimax_h3_image_forward.py from "
                "comfy.ldm.minimax.model.MiniMaxH3ImageModel; do not hand-edit",
    }
    tensors = {k: v.contiguous() for k, v in tensors.items()}
    safetensors.torch.save_file(tensors, OUT, metadata=meta)
    print("wrote %s: %d tensors, %d bytes" % (OUT, len(tensors), os.path.getsize(OUT)))


if __name__ == "__main__":
    sys.exit(main())
