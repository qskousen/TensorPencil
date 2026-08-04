#!/usr/bin/env python3
"""Reference conditioning tensors for weighted, multi-chunk prompts.

`gen_clip_prompt_fixtures.py` pins the *tokenizer* half (ids, weights, chunk
boundaries) and needs no checkpoint. This pins the **encoder** half on a real one:
what ComfyUI's `CLIPTextEncode` actually hands the UNet once the chunks have been
forwarded and the attention weights applied.

That second half has one convention no amount of reasoning settles, which is the
reason this file exists:

    z[j] = (z[j] - z_empty[j]) * w + z_empty[j]

`z_empty` is the same tower run on `[BOS] [EOS] pad…` at the same capture layer, so a
weight is an interpolation away from the **empty prompt's** hidden state at that
position — not a multiply, and not A1111's mean-renormalized form either. All three
produce a plausible image; only one matches ComfyUI.

Both families are generated, because their capture layers differ (SD1.5 takes the
final output, SDXL the penultimate with the final LayerNorm skipped) and SDXL adds a
second tower, a pad-0 padding and a pooled vector that must come from **chunk 0**.

⚠️ Tied to two checkpoints by name and sha256; the Zig tests assert the hashes.

Usage (ComfyUI's `nvenv`, NOT the ai-toolkit venv):
    /home/qt/genai/comfyui/nvenv/bin/python tools/gen_clip_cond_ref.py
"""

import hashlib
import json
import os
import sys

COMFY = "/home/qt/genai/comfyui"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_OUT = os.path.join(REPO, "src", "models", "assets", "clip_cond_ref.safetensors")

CKPTS = {
    "sd15": os.path.join(COMFY, "models/checkpoints/sd1.5/perfectdeliberate_v20.safetensors"),
    # ⚠️ Despite the workflow spelling it `sd1.5/…`, perfectdeliberate_v10 is an SDXL
    # checkpoint — it carries `label_emb` and a second `conditioner.embedders` tower.
    "sdxl": os.path.join(COMFY, "models/checkpoints/sdxl/perfectdeliberate_v10.safetensors"),
}

# See gen_sdxl_fixtures.py: imported as a library, ComfyUI parses argv only after
# enable_args_parsing(), and every flag is silently a default until then.
sys.argv = [sys.argv[0], "--cpu", "--fp32-vae", "--disable-smart-memory",
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

# The two real prompts from the render that exposed the truncation bug, plus the cases
# that isolate one convention each.
REAL_POS = (
    "1girl, original, general, blonde hair, long hair, ponytail, light blue eyes, "
    "(shiny skin:1.1), gentle smile, bare shoulders, collarbone, "
    "[white|light blue] off-shoulder shirt, sheer sleeves, denim short shorts, "
    "holding straw hat, looking back, side angle, upper body, beach, ocean, summer, "
    "sunny day, (lens flare:0.75), breeze, BREAK, masterpiece, best quality, "
    "very aesthetic, high contrast, vibrant, highres, year 2024, newest,"
)
REAL_NEG = (
    "(worst quality, low quality:1.2), (nsfw, sexually suggestive:1.2), lowres, "
    "(monochrome:1.1), wide shot, multiple views, watermark, signature, "
    "1980s \\(style\\), 4koma, serafuku, (text:1.1),"
)
PROMPTS = [
    ("real_pos", REAL_POS),          # 2 chunks, weights 0.75 / 1.1
    ("real_neg", REAL_NEG),          # 1 chunk, weights 1.1 / 1.2, escaped parens
    ("plain", "a photograph of an astronaut riding a horse"),  # 1 chunk, no weights
    ("empty", ""),                   # the unconditional branch
    ("zero_w", "a (red:0) cat"),     # w=0 is exactly z_empty — the sharpest probe of it
    ("big_w", "a (red:2.0) cat"),    # and extrapolation past 1
    ("three", "masterpiece, " + "very detailed cat portrait, " * 25 + "(sunset:1.4)"),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    out = {}
    meta = {
        "reference": "ComfyUI (comfy.sd.load_checkpoint_guess_config -> CLIP.encode_from_tokens), fp32 CPU",
        "note": "cond is [chunks*77, dim]; weights applied as (z - z_empty)*w + z_empty",
    }

    for fam, ckpt in CKPTS.items():
        if not os.path.exists(ckpt):
            raise SystemExit(f"missing checkpoint: {ckpt}")
        meta[f"{fam}_checkpoint"] = os.path.basename(ckpt)
        meta[f"{fam}_sha256"] = sha256(ckpt)
        print(f"[{fam}] loading {os.path.basename(ckpt)} …", flush=True)

        # output_model=False skips the UNet entirely: this fixture is text-only, and an
        # fp32 SDXL UNet is ~10 GB of RAM for nothing.
        _model, clip, _vae, _cv = comfy.sd.load_checkpoint_guess_config(
            ckpt, output_vae=False, output_clip=True, output_model=False,
            te_model_options={"dtype": torch.float32},
        )
        assert clip is not None, f"{fam}: no CLIP in checkpoint"

        for name, text in PROMPTS:
            tokens = clip.tokenize(text)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            c = cond[0].detach().float().contiguous()
            out[f"{fam}.{name}.cond"] = c
            n_chunks = c.shape[0] // 77
            assert c.shape[0] % 77 == 0, (name, c.shape)
            if pooled is not None:
                out[f"{fam}.{name}.pooled"] = pooled[0].detach().float().contiguous()
            print(f"  {name:9s} {tuple(c.shape)}  {n_chunks} chunk(s)"
                  f"  pooled={'yes' if pooled is not None else 'no'}", flush=True)

        del clip
        import gc

        gc.collect()

    from safetensors.torch import save_file

    os.makedirs(os.path.dirname(REF_OUT), exist_ok=True)
    save_file(out, REF_OUT, metadata=meta)
    print(f"\nwrote {REF_OUT}: {len(out)} tensors, "
          f"{os.path.getsize(REF_OUT)/1e6:.1f} MB")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
