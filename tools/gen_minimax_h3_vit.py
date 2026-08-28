#!/usr/bin/env python3
"""Pin the Qwen3-VL vision tower H3's text encoder carries, INCLUDING DeepStack.

`models/vit35.zig` already implements a Qwen3-VL tower of the same shape (27
blocks, 1152 wide, fused qkv, 4304 FFN, 48x48 learned position embedding), but it
was ported from llama.cpp's clip.cpp and it REFUSES deepstack. Two questions have
to be answered by measurement rather than by reading:

  1. does that llama.cpp-derived forward reproduce the HF/ComfyUI one bit for bit,
     or did the two lineages diverge somewhere (position-embedding interpolation
     and the 2-D rope's pair ordering are the likely places)?
  2. what exactly does deepstack compute?

so this fixture comes from `comfy.text_encoders.qwen3vl.Qwen3VLVisionModel`, the
lineage H3's checkpoint actually is.

What it pins:

  - the tower output `merged` AND all three deepstack features, separately.
  - ⚠️ **the two mergers normalize at DIFFERENT widths.** The main merger is
    `norm(x).view(-1, merge_dim)` -- LayerNorm over the PRE-merge hidden -- and the
    deepstack merger is `norm(x.view(-1, merge_dim))`, LayerNorm over the POST-merge
    width. One set of parentheses. The real checkpoint's shapes say so out loud
    (`merger.norm` is [1152], `deepstack_merger_list.N.norm` is [4608]) and a port
    that reads them the other way is a shape error only by luck.
  - deepstack features are taken from the OUTPUT of blocks
    `deepstack_visual_indexes`, not their input, and there is one merger per index.
  - the image is normalized to [-1, 1] (mean/std 0.5), NOT with CLIP statistics.
  - a NON-SQUARE image, so an h/w swap in the patch grid, the position-embedding
    interpolation or the rope cannot pass.

Emits src/models/assets/minimax_h3_vit.safetensors.

Usage (BOUND IT -- this builds a model):
    systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 \
        /home/qt/genai/comfyui/nvenv/bin/python tools/gen_minimax_h3_vit.py
"""

import json
import os
import sys

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "models", "assets", "minimax_h3_vit.safetensors",
)

COMFY = "/home/qt/genai/comfyui"
sys.path.insert(0, COMFY)
os.chdir(COMFY)

import torch  # noqa: E402
import safetensors.torch  # noqa: E402

import comfy.ops  # noqa: E402
import comfy.text_encoders.qwen_vl  # noqa: E402
from comfy.text_encoders.qwen3vl import Qwen3VLVisionModel  # noqa: E402

# Toy widths. `patch_size`, `spatial_merge_size` and `num_position_embeddings` are
# the REAL ones: the patch grid, the 2x2 merge and the 48x48 position table are
# what the geometry depends on, and shrinking them would stop pinning it.
CONFIG = dict(
    hidden_size=64,
    intermediate_size=128,
    depth=6,
    num_heads=4,
    patch_size=16,
    temporal_patch_size=2,
    in_channels=3,
    spatial_merge_size=2,
    num_position_embeddings=2304,   # 48 x 48, the real table
    deepstack_visual_indexes=[1, 3, 5],
    out_hidden_size=128,            # the LLM's width, 5120 in the real model
)

# Non-square AND not a multiple of the 32 px merge unit, so the reference's own
# resize actually runs -- and in OPPOSITE directions per axis: 100 -> 96 (down)
# while 150 -> 160 (up), since `round(dim / 32) * 32` is nearest, not floor. An
# image already on the grid would leave the resize an identity and pin nothing.
IMG_H, IMG_W = 100, 150


def main():
    torch.manual_seed(20260827)
    torch.set_num_threads(4)

    vit = Qwen3VLVisionModel(CONFIG, device="cpu", dtype=torch.float32, ops=comfy.ops.disable_weight_init)
    vit.eval()
    n_params = sum(p.numel() for p in vit.parameters())
    print("toy ViT: %d parameters (%.2f MB f32)" % (n_params, n_params * 4 / 1048576))
    assert n_params < 5_000_000, "the 'toy' tower is not toy: %d parameters" % n_params

    sd = {}
    for name, p in vit.state_dict().items():
        if name.endswith(".bias"):
            sd[name] = torch.randn(p.shape, dtype=torch.float32) * 0.05
        elif "norm" in name and p.dim() == 1:  # LayerNorm gamma
            sd[name] = 1.0 + torch.randn(p.shape, dtype=torch.float32) * 0.1
        elif p.dim() >= 2:
            fan_in = p[0].numel()
            sd[name] = torch.randn(p.shape, dtype=torch.float32) / (fan_in ** 0.5)
        else:
            sd[name] = torch.randn(p.shape, dtype=torch.float32) * 0.05
    vit.load_state_dict(sd, strict=True)
    for prm in vit.parameters():
        prm.requires_grad_(False)

    # The exact preprocessing H3's tokenizer uses for a still reference image:
    # Qwen3-VL normalizes to [-1, 1], NOT with CLIP statistics.
    image = torch.rand(1, IMG_H, IMG_W, 3, dtype=torch.float32)
    flat, grid = comfy.text_encoders.qwen_vl.process_qwen2vl_images(
        image, patch_size=CONFIG["patch_size"], image_mean=[0.5] * 3, image_std=[0.5] * 3)
    gh, gw = int(grid[0][1]), int(grid[0][2])
    assert (gh * 16, gw * 16) != (IMG_H, IMG_W), "the resize is an identity, so it is untested"
    print("  image [%d,%d] -> resized [%d,%d] -> patches %s grid %s"
          % (IMG_H, IMG_W, gh * 16, gw * 16, list(flat.shape), grid.tolist()))

    with torch.inference_mode():
        merged, deepstack = vit(flat.to(torch.float32), grid)

    assert len(deepstack) == len(CONFIG["deepstack_visual_indexes"]), len(deepstack)
    assert merged.shape[-1] == CONFIG["out_hidden_size"], merged.shape
    for d in deepstack:
        assert d.shape == merged.shape, (d.shape, merged.shape)
        assert torch.isfinite(d).all()
    assert torch.isfinite(merged).all()
    print("  merged %s, %d deepstack features of the same shape"
          % (list(merged.shape), len(deepstack)))

    tensors = {}
    for k, v in sd.items():
        tensors["visual." + k] = v
    tensors["in.image"] = image
    tensors["in.patches"] = flat
    tensors["in.grid"] = grid.to(torch.int32)
    tensors["out.merged"] = merged
    for i, d in enumerate(deepstack):
        tensors["out.deepstack.%d" % i] = d

    # --- teeth -------------------------------------------------------------
    def rel(a, b):
        return float((a - b).norm() / b.norm())

    # 1. The deepstack features must DIFFER from each other and from `merged`.
    #    A port that ran the main merger three times would otherwise pass.
    for i, d in enumerate(deepstack):
        r = rel(d, merged)
        assert r > 0.1, "deepstack[%d] is indistinguishable from merged (%.3f)" % (i, r)
    for i in range(len(deepstack) - 1):
        r = rel(deepstack[i], deepstack[i + 1])
        assert r > 0.1, "deepstack[%d] and [%d] are the same (%.3f)" % (i, i + 1, r)

    # 2. The merger norm-width swap: applying the main merger's PRE-merge norm
    #    inside the deepstack merger (or vice versa) must move the answer. Only
    #    representable because both norms exist; this substitutes one for the other
    #    at the width they share nothing about.
    import torch.nn.functional as F
    ds0 = vit.deepstack_merger_list[0]
    with torch.inference_mode():
        x = vit.patch_embed(flat.to(torch.float32))
        x = x + vit.fast_pos_embed_interpolate(grid)
        # Re-run just far enough to reach the first deepstack index.
        rotary = vit.rot_pos_emb(grid)
        seq = x.shape[0]
        x = x.reshape(seq, -1)
        rotary = rotary.reshape(seq, -1)
        emb = torch.cat((rotary, rotary), dim=-1)
        cos = emb.cos().unsqueeze(-2)
        sin = emb.sin().unsqueeze(-2)
        sh = sin.shape[-1] // 2
        pos = (cos, sin[..., :sh], -sin[..., sh:])
        cu = torch.repeat_interleave(grid[:, 1] * grid[:, 2], grid[:, 0]).cumsum(0, dtype=torch.int32)
        cu = F.pad(cu, (1, 0), value=0)
        from comfy.ldm.modules.attention import optimized_attention_for_device
        att = optimized_attention_for_device(x.device, mask=False, small_input=True)
        for i, blk in enumerate(vit.blocks):
            x = blk(x, cu_seqlens=cu, position_embeddings=pos, optimized_attention=att)
            if i == CONFIG["deepstack_visual_indexes"][0]:
                break
        # correct: norm AFTER the merge reshape
        right = ds0.linear_fc2(F.gelu(ds0.linear_fc1(ds0.norm(x.view(-1, ds0.merge_dim)))))
        # wrong: norm BEFORE it, at the pre-merge width, the main merger's shape
        pre = F.layer_norm(x, (CONFIG["hidden_size"],),
                           ds0.norm.weight[:CONFIG["hidden_size"]],
                           ds0.norm.bias[:CONFIG["hidden_size"]], eps=1e-6)
        wrong = ds0.linear_fc2(F.gelu(ds0.linear_fc1(pre.view(-1, ds0.merge_dim))))
    d_norm = rel(wrong, right)
    assert d_norm > 1e-2, "the merger norm width is indistinguishable (%.2e)" % d_norm
    assert rel(right, deepstack[0]) < 1e-6, "the re-run did not reproduce deepstack[0]"

    meta = {
        "config": json.dumps(CONFIG),
        "image": json.dumps(dict(h=IMG_H, w=IMG_W,
                                 grid=[int(v) for v in grid[0].tolist()],
                                 patches=int(flat.shape[0]),
                                 tokens=int(merged.shape[0]))),
        "note": "generated by tools/gen_minimax_h3_vit.py from "
                "comfy.text_encoders.qwen3vl.Qwen3VLVisionModel; do not hand-edit",
    }
    tensors = {k: v.contiguous() for k, v in tensors.items()}
    safetensors.torch.save_file(tensors, OUT, metadata=meta)
    print("wrote %s: %d tensors, %.2f MB" % (OUT, len(tensors), os.path.getsize(OUT) / 1048576))
    print("  sensitivity: merger norm width %.4f" % d_norm)


if __name__ == "__main__":
    sys.exit(main())
