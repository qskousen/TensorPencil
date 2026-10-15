#!/usr/bin/env python3
"""Compare TensorPencil's merged SD LoRA tensors with ComfyUI's, on REAL weights.

`zig build run -- lora-merge-check <ckpt> <lora> --dump <out> <names...>` writes
the merged tensors this side produced. This script builds ComfyUI's key map over
a stand-in model whose state dict carries the checkpoint's own names
(`comfy.lora.model_lora_keys_unet` / `_clip`), loads the LoRA with
`comfy.lora.load_lora`, merges each tensor with `comfy.lora.calculate_weight` on
an fp16 temp (ComfyUI's path on a 3090), and reports how many elements differ.

The map is the reference's own, so a key this side routes to the wrong tensor,
or misses, shows up as a difference rather than agreeing by construction.

Usage:
    /home/qt/genai/comfyui/nvenv/bin/python tools/check_lora_merge_real.py <ckpt> <lora> <dump>
"""

import json
import os
import struct
import sys
import types

COMFY = "/home/qt/genai/comfyui"
sys.path.insert(0, COMFY)
os.chdir(COMFY)

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

import comfy.lora  # noqa: E402
import comfy.model_detection  # noqa: E402

UNET = "model.diffusion_model."
CLIP_L = ["conditioner.embedders.0.transformer.text_model.", "cond_stage_model.transformer.text_model."]
CLIP_G = "conditioner.embedders.1.model."


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h


def main():
    ckpt, lora_path, dump = sys.argv[1:4]
    h = header(ckpt)
    meta = {k: torch.empty(v["shape"], device="meta") for k, v in h.items() if k.startswith(UNET)}
    cfg = comfy.model_detection.detect_unet_config(meta, UNET)

    # comfy key -> (checkpoint key, row slice or None)
    where = {}
    for k in h:
        if k.startswith(UNET):
            where["diffusion_model." + k[len(UNET):]] = (k, None)
    clip_l = next((p for p in CLIP_L if any(k.startswith(p) for k in h)), None)
    if clip_l:
        for k in h:
            if k.startswith(clip_l):
                where["clip_l.transformer.text_model." + k[len(clip_l):]] = (k, None)
    has_g = any(k.startswith(CLIP_G) for k in h)
    if has_g:
        for b in range(32):
            ip = "%stransformer.resblocks.%d.attn.in_proj_weight" % (CLIP_G, b)
            if ip not in h:
                continue
            hid = h[ip]["shape"][1]
            base = "clip_g.transformer.text_model.encoder.layers.%d." % b
            for i, q in enumerate(["q_proj", "k_proj", "v_proj"]):
                where[base + "self_attn.%s.weight" % q] = (ip, (i * hid, (i + 1) * hid))
            where[base + "self_attn.out_proj.weight"] = ("%stransformer.resblocks.%d.attn.out_proj.weight" % (CLIP_G, b), None)
            where[base + "mlp.fc1.weight"] = ("%stransformer.resblocks.%d.mlp.c_fc.weight" % (CLIP_G, b), None)
            where[base + "mlp.fc2.weight"] = ("%stransformer.resblocks.%d.mlp.c_proj.weight" % (CLIP_G, b), None)

    unet_model = types.SimpleNamespace(
        state_dict=lambda: {k: None for k in where if k.startswith("diffusion_model.")},
        model_config=types.SimpleNamespace(unet_config=cfg),
    )
    clip_model = types.SimpleNamespace(state_dict=lambda: {k: None for k in where if k.startswith("clip_")})
    key_map = comfy.lora.model_lora_keys_unet(unet_model, {})
    key_map = comfy.lora.model_lora_keys_clip(clip_model, key_map)

    lora = {}
    with safe_open(lora_path, "pt") as f:
        for k in f.keys():
            lora[k] = f.get_tensor(k)
    patches = comfy.lora.load_lora(lora, key_map, log_missing=False)

    # our dump name -> comfy key
    def comfy_key(name):
        if name.startswith("clip_l:"):
            return "clip_l.transformer.text_model." + name[len("clip_l:"):]
        if name.startswith("clip_g:"):
            rest = name[len("clip_g:"):]
            return rest  # resolved below by checkpoint name
        return "diffusion_model." + name

    total = 0
    bad = 0
    with safe_open(dump, "pt") as d, safe_open(ckpt, "pt") as c:
        for name in d.keys():
            ours = d.get_tensor(name).float()
            if name.startswith("clip_g:"):
                ck_name = CLIP_G + name[len("clip_g:"):]
                keys = [(k, w[1]) for k, w in where.items() if w[0] == ck_name and k in patches]
                ref = c.get_tensor(ck_name).to(torch.float16)
                for k, rs in keys:
                    if rs is None:
                        ref = comfy.lora.calculate_weight(patches[k] if isinstance(patches[k], list) else [(1.0, patches[k], 1.0, None, None)], ref, k)
                    else:
                        part = ref[rs[0]:rs[1]].clone()
                        part = comfy.lora.calculate_weight([(1.0, patches[k], 1.0, None, None)], part, k)
                        ref[rs[0]:rs[1]] = part
            else:
                k = comfy_key(name)
                ck_name = where[k][0]
                if k not in patches:
                    print("%s: ComfyUI does not patch it" % name)
                    bad += 1
                    continue
                ref = c.get_tensor(ck_name).to(torch.float16)
                ref = comfy.lora.calculate_weight([(1.0, patches[k], 1.0, None, None)], ref, k)
            ref = ref.float().reshape(ours.shape)
            diff = (ref != ours).sum().item()
            worst = ((ref - ours).abs() / ref.abs().clamp(min=1e-3)).max().item()
            total += 1
            bad += diff > 0
            print("%-60s %8d elems, %6d differ, worst rel %.2e" % (name, ours.numel(), diff, worst))
    print("%d tensors compared, %d with any difference; ComfyUI patched %d keys" % (total, bad, len(patches)))


if __name__ == "__main__":
    main()
