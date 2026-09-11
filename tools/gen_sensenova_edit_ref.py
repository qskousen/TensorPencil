#!/usr/bin/env python3
"""Real-checkpoint reference for SenseNova U1.5's IMAGE EDITING conditioning.

The width-reduced fixture (tools/gen_sensenova_fixtures.py) pins the whole
architecture, but it cannot carry the editing path: `_prepare_prefix` selects the
reference rows by the literal token id 151669, so a narrowed vocabulary never
matches one, and a full-vocabulary embedding table is 39 MB of asset for one
lookup. So editing is pinned here instead, on the real weights at ONE layer.

Four layers, and the first and the last are what it stores. Every convention
editing adds is upstream of the trunk -- where the `<img>` blocks are spliced into
the prompt, which token ids they are made of, the time index a whole reference
block shares, the row and column each of its tokens takes, the block-causal mask,
the ImageNet normalization the understanding tower wants and the rows its output
is written over -- so layer 0 already decides all of it. The last layer is there
because the block mask applies at EVERY depth, and a mask that is right at one
layer and drifts at the next would read as healthy in a layer-0 check.

⚠️ Bounded: a fp32 SenseNova layer is 1.5 GB and the embedding table another
2.5 GB, so run this under a memory cap like everything else here that loads torch.

Usage:
    systemd-run --user --scope -p MemoryMax=20G -p MemorySwapMax=0 \
        /home/qt/genai/comfyui/nvenv/bin/python tools/gen_sensenova_edit_ref.py
"""

import argparse
import json
import os
import sys

import numpy as np

COMFY = "/home/qt/genai/comfyui"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_OUT = os.path.join(REPO, "src", "models", "assets", "sensenova_edit_ref.safetensors")
DEFAULT_CKPT = os.path.join(
    COMFY, "models/diffusion_models/sensenova/sensenovaU158BMot_sft.safetensors")

PROMPT = "make it winter"
# 512x384 is a 16x12 token grid: 192 tokens in ONE bidirectional block.
#
# ⚠️ The size is the point, not a detail. A small reference renders correctly and a
# large one does not, so a fixture built on a 2x2 grid checks the mask's SHAPE
# while never exercising a block wide enough for anything about it to matter. It is
# also non-square, so a row/column swap in the reference's own grid shows.
REF_SIZE = (384, 512)
LAYERS = 4
SIGMA = 0.6
SEED = 20260909

OUR_ARGV = sys.argv[1:]
sys.argv = [sys.argv[0], "--cpu", "--disable-smart-memory",
            "--disable-xformers", "--use-pytorch-cross-attention"]
sys.path.insert(0, COMFY)
os.chdir(COMFY)

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import torch  # noqa: E402
import comfy.ops  # noqa: E402
import comfy.utils  # noqa: E402
from comfy.ldm.sensenova import model as sn  # noqa: E402
from comfy.ldm.sensenova import conditioning as sn_cond  # noqa: E402
import comfy.text_encoders.sensenova as sn_te  # noqa: E402
from safetensors import safe_open  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    args = ap.parse_args(OUR_ARGV)
    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"not found: {args.checkpoint}")

    sn.NUM_LAYERS = LAYERS
    src = safe_open(args.checkpoint, framework="pt")
    model = sn.SenseNovaU15(device="cpu", dtype=torch.float32, operations=comfy.ops.disable_weight_init)
    # In place, tensor by tensor: building the whole state dict first holds a
    # SECOND fp32 copy of every layer alongside the model's own, which is what
    # OOM-kills this at eight layers under a sane cap.
    with torch.no_grad():
        for name, param in model.state_dict().items():
            param.copy_(src.get_tensor(name).float())
    model.eval()

    # A SMOOTH picture, not white noise, and that is not cosmetic: a reference the
    # model can actually copy drives it very differently from one it cannot, and a
    # noise reference leaves the whole copying behaviour unexercised. Built from a
    # blurred random field so it stays deterministic and needs no asset of its own.
    #
    # u8, and divided by 255 on both sides: the fixture stores the picture the way a
    # node hands it over, and a float store would be four times the bytes for a
    # value that came from eight bits anyway.
    rng = np.random.default_rng(SEED)
    coarse = rng.random((REF_SIZE[0] // 16 + 1, REF_SIZE[1] // 16 + 1, 3), dtype=np.float32)
    smooth = np.asarray(
        torch.nn.functional.interpolate(
            torch.from_numpy(coarse).permute(2, 0, 1)[None],
            size=REF_SIZE, mode="bicubic", align_corners=False,
        )[0].permute(1, 2, 0).clamp(0, 1))
    img_u8 = (smooth * 255.0 + 0.5).astype(np.uint8)
    img = torch.from_numpy(img_u8.astype(np.float32) / 255.0)

    tok = sn_te.SenseNovaTokenizer()
    base = [int(t[0]) for t in tok.tokenize_with_weights(PROMPT)["sensenova_u15"][0]]
    refs = sn_cond.preprocess_references([img[None]])
    grids = [(max(1, -(-r.shape[-2] // 32)), max(1, -(-r.shape[-1] // 32))) for r in refs]
    ids = sn_cond.condition_input_ids(torch.tensor([base], dtype=torch.long), grids, image_only=False)
    idx = sn_cond.thw_indexes(ids, grids)
    mask = sn_cond.block_causal_mask(idx, dtype=torch.float32)

    with torch.no_grad():
        keys, values, ptime = model.preprocess_prefix(ids, refs, idx, mask)
        # And one whole denoiser forward on that prefix. 96x64 px is a 3x2 token
        # grid: not square, so a row/column swap in the canvas shows, and small
        # enough to store the velocity whole.
        gen = torch.Generator().manual_seed(SEED + 1)
        x = torch.randn(1, 3, 96, 64, generator=gen).half().float()
        # ComfyUI hands the model `1 - sigma`, not the sigma.
        t = torch.tensor([1.0 - SIGMA], dtype=torch.float32)
        v = model._forward(x, t, prefix_keys=keys, prefix_values=values, prefix_time=ptime)

    out = {
        "base_ids": torch.tensor(base, dtype=torch.int32),
        "ids": ids[0].to(torch.int32),
        "thw": idx[0].to(torch.int32),
        "time": torch.tensor([int(ptime[0])], dtype=torch.int32),
        "ref": torch.from_numpy(np.ascontiguousarray(img_u8)),  # [h][w][3] u8, the layout a node hands over
        "fwd.x": x[0],
        "fwd.t": t,
        "fwd.sigma": torch.tensor([SIGMA], dtype=torch.float32),
        "fwd.v": v[0],
    }
    # The first and the last, not all of them: the middle layers cost asset bytes
    # for a depth the two ends already bracket.
    for i in (0, LAYERS - 1):
        out[f"k.{i}"] = keys[i].squeeze(0).half().contiguous()
        out[f"v.{i}"] = values[i].squeeze(0).half().contiguous()

    meta = {
        "prompt": PROMPT,
        "layers": str(LAYERS),
        "grids": json.dumps(grids),
        "checkpoint": os.path.basename(args.checkpoint),
    }
    os.makedirs(os.path.dirname(REF_OUT), exist_ok=True)
    comfy.utils.save_torch_file(out, REF_OUT, metadata=meta)
    total = sum(v.numel() * v.element_size() for v in out.values())
    print(f"wrote {REF_OUT} ({total / 1e6:.2f} MB, {ids.shape[1]} prefix tokens, "
          f"grids {grids}, prefix_time {int(ptime[0])})")


if __name__ == "__main__":
    main()
