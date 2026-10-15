#!/usr/bin/env python3
"""Pin the merged-LoRA path (models/lora_merge.zig) against ComfyUI's own merge.

The SD family takes a LoRA by MERGING it into the weight, the way ComfyUI does,
so every number here comes from executing the reference:

  - `comfy.model_detection.detect_unet_config` on the real SD1.5 and SDXL
    checkpoint HEADERS (meta tensors, no weight is read), then
    `comfy.utils.unet_to_diffusers` on that config. Half the SD LoRAs in the wild
    name the UNet in diffusers terms, and this map is the only bridge.
  - `comfy.lora.load_lora` + `comfy.lora.calculate_weight` per adapter kind, on a
    temp weight in fp16. That is ComfyUI's path on a GPU that prefers fp16
    (`model_management.lora_compute_dtype`), a 3090 included: the stored weight is
    cast to fp16, every patch adds its fp16-rounded delta in fp16, in file order.
    One case runs the f32 temp path instead, which is what an f32 base keeps here.

Cases cover every adapter the SD LoRAs on disk use: plain LoRA (linear, no alpha,
3x3 LoCon, Tucker `lora_mid`), LoHa (plain and CP), LoKr (full factors and
decomposed, and on a 3x3 conv), DoRA on both axes (and on a conv, and off strength
1), a `diff` (linear and conv), a two-file stack, and a negative strength. The generator asserts each case moves the weight, so no case
is vacuous. That check is why Tucker LoKr (`lokr_t2`) is absent: the reference
raises inside `torch.kron` on it, logs, and leaves the weight unpatched.

Emits src/models/assets/lora_merge.safetensors and
src/models/assets/lora_unet_diffusers.txt.

Usage:
    /home/qt/genai/comfyui/nvenv/bin/python tools/gen_lora_merge_fixtures.py
"""

import json
import os
import struct
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "src", "models", "assets")
CKPTS = {
    "sd15": "/mnt/watch/models/diffusion/checkpoints/sd1.5/perfectdeliberate_v20.safetensors",
    "sdxl": "/mnt/watch/models/diffusion/checkpoints/illustrious/illustrious_v110.safetensors",
}

COMFY = "/home/qt/genai/comfyui"
sys.path.insert(0, COMFY)
os.chdir(COMFY)

import torch  # noqa: E402
import safetensors.torch  # noqa: E402

import comfy.lora  # noqa: E402
import comfy.model_detection  # noqa: E402
import comfy.utils  # noqa: E402

DT = {"F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32, "F64": torch.float64}


def header_meta(path):
    """The checkpoint's tensors as meta tensors: shapes only, nothing read."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return {k: torch.empty(v["shape"], dtype=DT.get(v["dtype"], torch.float32), device="meta") for k, v in h.items()}


def diffusers_maps():
    lines = []
    for fam, path in CKPTS.items():
        sd = header_meta(path)
        cfg = comfy.model_detection.detect_unet_config(sd, "model.diffusion_model.")
        m = comfy.utils.unet_to_diffusers(cfg)
        n = 0
        for k in sorted(m):
            if k.endswith(".weight"):
                lines.append("%s %s %s" % (fam, k, m[k]))
                n += 1
        print("%s: %d diffusers weight names (depth %s / out %s / mid %s)" % (
            fam, n, cfg["transformer_depth"], cfg["transformer_depth_output"], cfg.get("transformer_depth_middle")))
    with open(os.path.join(ASSETS, "lora_unet_diffusers.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


def comfy_merge(base, patches, temp_dtype):
    """ComfyUI's patch_weight_to_device on one weight: temp copy, every patch in
    order, then the cast back. `patches` is [(lora_dict, stem, strength)]."""
    key = "w.weight"
    plist = []
    for lora, stem, strength in patches:
        loaded = comfy.lora.load_lora(lora, {stem: key}, log_missing=False)
        assert key in loaded, "comfy did not recognize %s" % stem
        plist.append((strength, loaded[key], 1.0, None, None))
    temp = base.to(temp_dtype, copy=True)
    out = comfy.lora.calculate_weight(plist, temp, key)
    return out.to(temp_dtype)


def main():
    diffusers_maps()

    g = torch.Generator().manual_seed(20261006)

    def rnd(*shape, s=0.3):
        return torch.randn(*shape, generator=g, dtype=torch.float32) * s

    out = {}
    cases = []

    def case(name, base, files, temp=torch.float16):
        """`files` is [(dict of suffix -> tensor, strength)]; keys land under
        `<name>.<i>.` so each file is its own namespace in one fixture."""
        patches = []
        for i, (keys, strength) in enumerate(files):
            stem = "%s.%d" % (name, i)
            lora = {}
            for sfx, t in keys.items():
                lora["%s.%s" % (stem, sfx)] = t
                out["%s.%s" % (stem, sfx)] = t.contiguous()
            patches.append((lora, stem, strength))
            out["%s.%d.strength" % (name, i)] = torch.tensor([strength], dtype=torch.float32)
        want = comfy_merge(base, patches, temp)
        moved = (want.float() - base.to(temp).float()).abs().max().item()
        assert moved > 0, "case %s does not move the weight" % name
        out["%s.base" % name] = base.contiguous()
        out["%s.want" % name] = want.contiguous()
        cases.append(name)
        print("%-14s base %s %s, %d file(s), max |change| %.3e" % (name, tuple(base.shape), base.dtype, len(files), moved))

    o, i, r = 24, 16, 4
    # Plain LoRA, alpha present, an f16 base (the common SDXL file).
    case("lora", rnd(o, i).half(), [({"lora_down.weight": rnd(r, i), "lora_up.weight": rnd(o, r), "alpha": torch.tensor(2.0)}, 1.0)])
    # No alpha: scale 1. A bf16 base, which ComfyUI still merges through fp16.
    case("noalpha", rnd(o, i).bfloat16(), [({"lora_down.weight": rnd(r, i), "lora_up.weight": rnd(o, r)}, 0.7)])
    # The f32 temp path, which an f32 base keeps here.
    case("f32", rnd(o, i), [({"lora_down.weight": rnd(r, i), "lora_up.weight": rnd(o, r), "alpha": torch.tensor(3.0)}, 1.0)], temp=torch.float32)
    # 3x3 LoCon: down [r, in, 3, 3], up [out, r, 1, 1].
    case("locon", rnd(o, i, 3, 3).half(), [({"lora_down.weight": rnd(r, i, 3, 3), "lora_up.weight": rnd(o, r, 1, 1), "alpha": torch.tensor(1.0)}, 1.0)])
    # Tucker conv: down [r, in, 1, 1], mid [r, r, 3, 3], up [out, r, 1, 1].
    case("tucker", rnd(o, i, 3, 3).half(), [({"lora_down.weight": rnd(r, i, 1, 1), "lora_mid.weight": rnd(r, r, 3, 3), "lora_up.weight": rnd(o, r, 1, 1), "alpha": torch.tensor(2.0)}, 1.0)])
    # The diffusers spelling of the same pair.
    case("peft", rnd(o, i).half(), [({"lora_A.weight": rnd(r, i), "lora_B.weight": rnd(o, r), "alpha": torch.tensor(8.0)}, 1.0)])
    # LoHa: (w1a w1b) * (w2a w2b), alpha over w1b's rows.
    case("loha", rnd(o, i).half(), [({"hada_w1_a": rnd(o, r), "hada_w1_b": rnd(r, i), "hada_w2_a": rnd(o, r), "hada_w2_b": rnd(r, i), "alpha": torch.tensor(2.0)}, 1.0)])
    # LoHa CP on a conv.
    case("loha_cp", rnd(o, i, 3, 3).half(), [({"hada_w1_a": rnd(r, o), "hada_w1_b": rnd(r, i), "hada_t1": rnd(r, r, 3, 3), "hada_w2_a": rnd(r, o), "hada_w2_b": rnd(r, i), "hada_t2": rnd(r, r, 3, 3), "alpha": torch.tensor(2.0)}, 1.0)])
    # LoKr with both factors whole: alpha is ignored (no decomposed dim).
    case("lokr_full", rnd(o, i).half(), [({"lokr_w1": rnd(4, 2, s=0.5), "lokr_w2": rnd(6, 8, s=0.5), "alpha": torch.tensor(3.0)}, 1.0)])
    # LoKr with both decomposed: alpha over w2_b's rows.
    case("lokr_dec", rnd(o, i).half(), [({"lokr_w1_a": rnd(4, 2), "lokr_w1_b": rnd(2, 2), "lokr_w2_a": rnd(6, 3), "lokr_w2_b": rnd(3, 8), "alpha": torch.tensor(1.5)}, 1.0)])
    # DoRA, magnitude per OUTPUT row ([out, 1]) and per INPUT column ([1, in]),
    # each at a strength off 1 so the blend branch runs too.
    case("dora_out", rnd(o, i).half(), [({"lora_down.weight": rnd(r, i), "lora_up.weight": rnd(o, r), "alpha": torch.tensor(4.0), "dora_scale": rnd(o, 1, s=1.0).abs() + 0.5}, 0.8)])
    case("dora_in", rnd(o, i).half(), [({"lora_down.weight": rnd(r, i), "lora_up.weight": rnd(o, r), "alpha": torch.tensor(4.0), "dora_scale": rnd(1, i, s=1.0).abs() + 0.5}, 1.0)])
    # A whole-weight diff.
    case("diff", rnd(o, i).half(), [({"diff": rnd(o, i, s=0.05)}, 0.6)])
    # A stack: two files on one weight, in order, one negative.
    case("stack", rnd(o, i).half(), [
        ({"lora_down.weight": rnd(r, i), "lora_up.weight": rnd(o, r), "alpha": torch.tensor(2.0)}, 0.8),
        ({"hada_w1_a": rnd(o, r), "hada_w1_b": rnd(r, i), "hada_w2_a": rnd(o, r), "hada_w2_b": rnd(r, i), "alpha": torch.tensor(4.0)}, -0.5),
    ])

    # The sidecar's own shapes, appended so the cases above keep their bytes. LoKr
    # on a 3x3 conv, both factors whole and both decomposed: the kron runs over
    # channel groups (`w1` mixes the groups, `w2` is the per-group conv).
    case("lokr_conv", rnd(o, i, 3, 3).half(), [({"lokr_w1": rnd(4, 2, s=0.5), "lokr_w2": rnd(6, 8, 3, 3, s=0.3), "alpha": torch.tensor(3.0)}, 0.9)])
    case("lokr_conv_dec", rnd(o, i, 3, 3).half(), [({"lokr_w1": rnd(4, 2, s=0.5), "lokr_w2_a": rnd(6, 3), "lokr_w2_b": rnd(3, 8 * 9), "alpha": torch.tensor(1.5)}, 1.0)])
    # A whole-weight diff on a conv.
    case("diff_conv", rnd(o, i, 3, 3).half(), [({"diff": rnd(o, i, 3, 3, s=0.05)}, 0.7)])
    # Input-axis DoRA off strength 1, where the blend needs a second base product,
    # and on a 3x3 conv.
    case("dora_in_s", rnd(o, i).half(), [({"lora_down.weight": rnd(r, i), "lora_up.weight": rnd(o, r), "alpha": torch.tensor(4.0), "dora_scale": rnd(1, i, s=1.0).abs() + 0.5}, 0.6)])
    case("dora_in_conv", rnd(o, i, 3, 3).half(), [({"lora_down.weight": rnd(r, i, 3, 3), "lora_up.weight": rnd(o, r, 1, 1), "alpha": torch.tensor(2.0), "dora_scale": rnd(1, i, 1, 1, s=1.0).abs() + 0.5}, 0.8)])

    out["cases"] = torch.tensor([len(cases)], dtype=torch.int32)
    meta = {"cases": ",".join(cases)}
    path = os.path.join(ASSETS, "lora_merge.safetensors")
    safetensors.torch.save_file(out, path, metadata=meta)
    print("wrote %s (%d tensors)" % (path, len(out)))


if __name__ == "__main__":
    main()
