#!/usr/bin/env python3
"""Reference conditioning for Mage-Flow's text encoder
(`comfy/text_encoders/mage_flow.py`).

The DiT fixture feeds RANDOM context, so it says nothing about where that
context comes from. Three choices decide it, none of them recorded in any
checkpoint, and each is a silent wrong answer:

  - the chat template (identical to krea2's for t2i, a different system turn
    for edit);
  - the prefix strip, which drops everything up to and including the second
    `<|im_start|>` (+ `user` `\\n`);
  - the tap: the FINAL hidden state with `model.norm` APPLIED. ComfyUI spells
    it `layer_idx = -1` plus `layer_norm_hidden_state = True`, and krea2 reads
    the same checkpoint as a twelve-layer stack with the norm never evaluated.

⚠️ **The reference runs at the checkpoint's own dtype, not fp32.** An fp32
Qwen3-VL-4B is 16 GB and ComfyUI holds the source alive while it casts, which
does not fit here. So the stored conditioning carries a bf16-ish floor, and a
tolerance alone could not tell "right tap" from "close enough". `cond_nonorm`
is the amplification control for exactly that: the same tap with the final norm
SKIPPED. The Zig test asserts it is much further away than the floor, so the
comparison has teeth the tolerance by itself would not give it.

Run it memory-bounded:

    systemd-run --user --scope -p MemoryMax=20G -p MemorySwapMax=0 \\
        /home/qt/genai/comfyui/nvenv/bin/python tools/gen_mageflow_te.py
"""

import argparse
import hashlib
import json
import os
import sys

COMFY = "/home/qt/genai/comfyui"
DEFAULT_TE = os.path.join(COMFY, "models/text_encoders/qwen3VLInstruct4bHeretic_v10.safetensors")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_OUT = os.path.join(REPO, "src", "models", "assets", "mageflow_te_ref.safetensors")

PROMPTS = [
    "a red apple on a wooden table",
    "",  # the unconditional branch; an empty prompt is still the whole template
]

EDIT_PROMPT = "replace the background with a grassland prairie"
# 300x500 on purpose: the long edge is past the node's 384 cap so the bicubic
# resize binds, and the capped 230x384 then rounds to 224x384 rather than
# staying put, so the tower's own grid resize binds too. Non-square, because a
# square case cannot see a transposed axis.
EDIT_H, EDIT_W = 300, 500
EDIT_SEED = 7

OUR_ARGV = sys.argv[1:]
sys.argv = [sys.argv[0], "--cpu", "--disable-smart-memory",
            "--disable-xformers", "--use-pytorch-cross-attention"]
sys.path.insert(0, COMFY)
os.chdir(COMFY)

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import torch  # noqa: E402
from comfy.cli_args import args as comfy_args  # noqa: E402

if not (comfy_args.cpu and comfy_args.disable_xformers):
    raise SystemExit("ComfyUI did not take our flags — check comfy.options.enable_args_parsing()")

import comfy.sd  # noqa: E402
import comfy.text_encoders.qwen_vl  # noqa: E402
import comfy.utils  # noqa: E402


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-encoder", default=DEFAULT_TE)
    args = ap.parse_args(OUR_ARGV)
    if not os.path.exists(args.text_encoder):
        raise SystemExit(f"not found: {args.text_encoder}")

    clip = comfy.sd.load_clip(
        ckpt_paths=[args.text_encoder],
        embedding_directory=None,
        clip_type=comfy.sd.CLIPType.MAGE,
    )
    te_name = type(clip.cond_stage_model).__name__
    if "MageFlow" not in te_name:
        raise SystemExit(f"ComfyUI picked {te_name} for this file, not the Mage-Flow one")
    inner = clip.cond_stage_model.qwen3vl_4b
    if not inner.layer_norm_hidden_state:
        raise SystemExit("layer_norm_hidden_state is False; the tap this fixture pins has moved")
    if (inner.layer, inner.layer_idx) != ("hidden", -1):
        raise SystemExit(f"the tap moved to {(inner.layer, inner.layer_idx)}, expected ('hidden', -1)")

    out: dict[str, torch.Tensor] = {}
    meta = {
        "text_encoder": os.path.basename(args.text_encoder),
        "text_encoder_sha256": sha256(args.text_encoder),
        "template": clip.tokenizer.llama_template,
        "prompts": json.dumps(PROMPTS),
    }

    with torch.no_grad():
        for i, text in enumerate(PROMPTS):
            tokens = clip.tokenize(text)
            ids = [int(t) for t, _w in tokens["qwen3vl_4b"][0]]
            cond = clip.encode_from_tokens(tokens, return_pooled=False).float()
            out[f"tokens.{i}"] = torch.tensor(ids, dtype=torch.int32)
            out[f"cond.{i}"] = cond[0].contiguous()

            # The control: the same tap with the final norm skipped, which is
            # what krea2's variant of this encoder does.
            inner.layer_norm_hidden_state = False
            nonorm = clip.encode_from_tokens(tokens, return_pooled=False).float()
            inner.layer_norm_hidden_state = True
            out[f"cond_nonorm.{i}"] = nonorm[0].contiguous()

            meta[f"strip.{i}"] = str(len(ids) - cond.shape[1])
            print(f"prompt {i}: {len(ids)} tokens -> {tuple(cond.shape)} (stripped {len(ids) - cond.shape[1]})")

    # --- the edit path: one reference picture through the vision tower --------
    #
    # ⚠️ The reference is resized TWICE by ComfyUI, to different sizes, for
    # different consumers, and only the VL copy is made here: the node caps the
    # long edge at 384 (bicubic) before handing it to the tokenizer, which then
    # applies the tower's own grid resize. The DiT's copy goes to the render's
    # resolution through the VAE and is not part of this fixture.
    with torch.no_grad():
        g = torch.Generator().manual_seed(EDIT_SEED)
        # ComfyUI's IMAGE layout is [B, H, W, 3] in [0, 1].
        img = torch.rand((1, EDIT_H, EDIT_W, 3), generator=g, dtype=torch.float32)

        samples = img.movedim(-1, 1)
        long_edge = max(samples.shape[3], samples.shape[2])
        if long_edge > 384:
            scale_by = 384 / long_edge
            capped = comfy.utils.common_upscale(
                samples,
                max(1, round(samples.shape[3] * scale_by)),
                max(1, round(samples.shape[2] * scale_by)),
                "bicubic", "disabled").movedim(1, -1)
        else:
            capped = img

        tokens = clip.tokenize(EDIT_PROMPT, images=[capped])
        ids = []
        n_image_entries = 0
        for t, _w in tokens["qwen3vl_4b"][0]:
            if isinstance(t, dict):
                n_image_entries += 1
                ids.append(151655)  # the pad the descriptor replaced
            else:
                ids.append(int(t))

        # The tower alone, so a splice/mrope/deepstack mistake is separable from
        # a tower mistake -- and the two preprocessing steps before it, so a
        # resize mistake is separable from both.
        inner = clip.cond_stage_model.qwen3vl_4b
        merged, extra = inner.transformer.preprocess_embed(
            {"type": "image", "data": capped}, torch.device("cpu"))
        flat, grid = comfy.text_encoders.qwen_vl.process_qwen2vl_images(
            capped, patch_size=16, image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5])
        out["edit.capped"] = capped[0].movedim(-1, 0).contiguous()
        out["edit.patches"] = flat.float().contiguous()

        # ⚠️ **The position table is the one bf16 sum in an otherwise-f32 tower.**
        # `preprocess_embed` casts the IMAGE to f32, and `comfy.ops` then casts
        # every weight to the input's dtype, so the blocks run in f32 -- but
        # `fast_pos_embed_interpolate` looks the table up through `nn.Embedding`
        # and builds its bilinear weights at `pos_embed.weight.dtype`, which is
        # the checkpoint's bf16. Measured: that alone is 2.6e-3 on the position
        # rows and 1.4e-2 on the tower's output, which is far coarser than
        # anything else this fixture pins.
        #
        # So the reference is taken with the table UPCAST, and the bf16 figure is
        # stored beside it as the floor rather than as the target. This is the
        # one place the Zig side is deliberately more accurate than ComfyUI, and
        # it is storage precision rather than a convention.
        visual = inner.transformer.visual
        bf16_pos = visual.pos_embed.weight.data
        out["edit.merged_bf16pos"] = merged.float().contiguous()
        out["edit.pos_bf16"] = visual.fast_pos_embed_interpolate(grid).detach().float().contiguous()

        visual.pos_embed.weight.data = bf16_pos.float()
        out["edit.pos"] = visual.fast_pos_embed_interpolate(grid).detach().float().contiguous()
        merged, extra = inner.transformer.preprocess_embed(
            {"type": "image", "data": capped}, torch.device("cpu"))
        cond = clip.encode_from_tokens(tokens, return_pooled=False).float()
        visual.pos_embed.weight.data = bf16_pos

        d_pos = (out["edit.pos"] - out["edit.pos_bf16"]).abs().max().item()
        d_merged = (out["edit.merged_bf16pos"] - merged.float()).abs().max().item()
        print(f"  bf16 position table: max |delta| {d_pos:.3e} on the rows, "
              f"{d_merged:.3e} on the tower output")

        out["edit.image"] = img[0].movedim(-1, 0).contiguous()   # planar [3][h][w]
        out["edit.tokens"] = torch.tensor(ids, dtype=torch.int32)
        out["edit.cond"] = cond[0].contiguous()
        out["edit.merged"] = merged.float().contiguous()
        for i, d in enumerate(extra["deepstack"]):
            out[f"edit.deepstack.{i}"] = d.float().contiguous()
        meta["edit.prompt"] = EDIT_PROMPT
        meta["edit.grid"] = json.dumps([int(x) for x in torch.as_tensor(extra["grid"]).reshape(-1).tolist()])
        meta["edit.template"] = clip.tokenizer.llama_template_images
        meta["edit.strip"] = str(len(ids) + merged.shape[0] - n_image_entries - cond.shape[1])
        print(f"edit: {len(ids)} template tokens, grid {meta['edit.grid']}, "
              f"{merged.shape[0]} merged -> cond {tuple(cond.shape)}")

    os.makedirs(os.path.dirname(REF_OUT), exist_ok=True)
    comfy.utils.save_torch_file(out, REF_OUT, metadata=meta)
    print(f"wrote {REF_OUT} ({len(out)} tensors)")


if __name__ == "__main__":
    main()
