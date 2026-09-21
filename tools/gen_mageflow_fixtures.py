#!/usr/bin/env python3
"""Reference fixtures for Mage-Flow's denoiser (`comfy/ldm/mage_flow/model.py`).

Reference is **ComfyUI**, for the reason `gen_sdxl_fixtures.py` gives: it is the
compatibility target, and everything this fixture pins is a *choice* the
checkpoint does not record — the order of the six modulation chunks, which half
of the sequence comes first inside the attention, where the RoPE origin sits,
which of `scale`/`shift` the final layer's linear emits first, and whether the
timestep table is rounded to bf16.

⚠️ **The config is DERIVED from ComfyUI, never hand-copied.**
`model_detection.detect_unet_config` reads it off the tensor shapes; this script
asserts the values it then writes into the fixture, so a ComfyUI change that
moves one of them fails here loudly instead of silently re-baselining the Zig
side against a stale constant.

## Why the reference is truncated to 2 layers

An fp32 copy of the real 4.1B denoiser is 16.5 GB, past this machine's free RAM.
Casting the reference to bf16 instead would put a ~1e-2 error floor in the
*reference*, coarser than the bugs this exists to catch: a swapped `scale`/`shift`
at the final layer and a text half concatenated after the image half both survive
1e-2 in aggregate.

All 12 blocks are identical, so 2 of them, at the real width and from the real
weights, is an exact fp32 reference for every convention above. What it
deliberately does not cover is "does the block loop run 12 times", which is a loop
bound, and which the end-to-end render comparison covers instead.

The three cases are chosen for what they separate:

  - `even`: a 4x6 latent. Centering by `n - n//2` and by `n//2` agree here, so
    this case passes under either convention and is the control.
  - `odd`: a 5x7 latent. The two conventions differ by one on both axes, so
    this is the case that pins Mage's.
  - `ref`: a 4x6 canvas plus a 3x5 reference image. Pins the edit path: the
    reference's tokens are appended, get RoPE axis 0 = 1, are centered on their
    OWN size, and are dropped from the output.

Run it memory-bounded — unbounded it can take the machine down:

    systemd-run --user --scope -p MemoryMax=20G -p MemorySwapMax=0 \\
        /home/qt/genai/comfyui/nvenv/bin/python tools/gen_mageflow_fixtures.py
"""

import argparse
import hashlib
import json
import os
import sys

COMFY = "/home/qt/genai/comfyui"
DEFAULT_CKPT = os.path.join(COMFY, "models/diffusion_models/mageflow/mageFlow_mageFlow4B.safetensors")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_OUT = os.path.join(REPO, "src", "models", "assets", "mageflow_ref.safetensors")

SEED = 20260918
# How many transformer blocks the fp32 reference keeps. See the module docstring.
REF_LAYERS = 2
SEQ_TXT = 7

# (name, canvas h, canvas w, [(ref h, ref w), ...], sigma)
CASES = [
    ("even", 4, 6, [], 0.75),
    ("odd", 5, 7, [], 0.25),
    ("ref", 4, 6, [(3, 5)], 0.5),
]

# ⚠️ ComfyUI parses argv at import time; our own flags must come off `sys.argv`
# before `import comfy.*`. `--cpu` because a reference must not depend on a GPU
# reduction order, `--disable-xformers` because ComfyUI binds its attention
# implementation at import time and prefers xformers, which has no CPU fp32 kernel.
OUR_ARGV = sys.argv[1:]
sys.argv = [sys.argv[0], "--cpu", "--disable-smart-memory",
            "--disable-xformers", "--use-pytorch-cross-attention"]
sys.path.insert(0, COMFY)
os.chdir(COMFY)

# ⚠️ Imported as a library, ComfyUI ignores argv entirely unless this is called
# first — `comfy/cli_args.py` parses `[]`. Without it every flag above silently
# defaults and the "CPU fp32" reference quietly runs on the GPU.
import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import torch  # noqa: E402
from comfy.cli_args import args as comfy_args  # noqa: E402
import comfy.model_management as _mm  # noqa: E402

if not (comfy_args.cpu and comfy_args.disable_xformers):
    raise SystemExit("ComfyUI did not take our flags — check comfy.options.enable_args_parsing()")
if _mm.xformers_enabled():
    raise SystemExit("xformers is still active; it has no CPU fp32 attention kernel")

import comfy.ops  # noqa: E402
import comfy.utils  # noqa: E402
import comfy.model_detection  # noqa: E402
from comfy.ldm.mage_flow.model import MageFlowTransformer2DModel  # noqa: E402

# What this generator (and the Zig implementation) believes Mage-Flow is.
EXPECTED = {"image_model": "mage_flow", "in_channels": 128, "num_layers": 12}
# The model's own defaults, which `detect_unet_config` does NOT report because
# they are not read off the weights. Asserted against the constructed module so a
# default change is caught rather than inherited.
EXPECTED_DIMS = {
    "patch_size": 1,
    "inner_dim": 3072,
    "num_attention_heads": 24,
    "attention_head_dim": 128,
    "joint_attention_dim": 2560,
    "axes_dims_rope": [16, 56, 56],
    "rope_theta": 10000,
}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


class LazySafetensors:
    """Read-through mapping over a safetensors file.

    ⚠️ The 8.2 GB checkpoint loaded eagerly plus an fp32 copy of the slice we want
    peaks past this machine's free RAM. `detect_unet_config` only indexes two
    tensors (it works off the key list otherwise) and a 2-layer reference needs
    about a sixth of the file, so materializing on demand keeps the peak at
    roughly what is actually used.
    """

    def __init__(self, path):
        from safetensors import safe_open

        self._f = safe_open(path, framework="pt", device="cpu")
        self._keys = list(self._f.keys())

    def keys(self):
        return self._keys

    def __contains__(self, k):
        return k in self._keys

    def __getitem__(self, k):
        return self._f.get_tensor(k)

    def get(self, k, default=None):
        return self._f.get_tensor(k) if k in self._keys else default


def detect_config(sd) -> dict:
    cfg = comfy.model_detection.detect_unet_config(sd, "", None)
    for k, want in EXPECTED.items():
        if cfg.get(k) != want:
            raise SystemExit(
                f"ComfyUI detected {k}={cfg.get(k)!r}, this generator expects {want!r}. "
                "The architecture moved — do not regenerate until the Zig side is updated.")
    return cfg


def build_ref_dit(sd, n_layers: int) -> MageFlowTransformer2DModel:
    model = MageFlowTransformer2DModel(
        num_layers=n_layers, device="cpu", dtype=torch.float32,
        operations=comfy.ops.disable_weight_init)

    got = {
        "patch_size": model.patch_size,
        "inner_dim": model.inner_dim,
        "num_attention_heads": model.transformer_blocks[0].num_attention_heads,
        "attention_head_dim": model.transformer_blocks[0].attention_head_dim,
        "joint_attention_dim": model.txt_in.in_features,
        "axes_dims_rope": list(model.pe_embedder.axes_dim),
        "rope_theta": model.pe_embedder.theta,
    }
    if got != EXPECTED_DIMS:
        raise SystemExit(f"Mage-Flow's own defaults moved: {got} != {EXPECTED_DIMS}")

    want = set(model.state_dict().keys())
    have = set(sd.keys())
    missing = want - have
    if missing:
        raise SystemExit(f"checkpoint is missing {len(missing)} tensors, e.g. {sorted(missing)[:5]}")
    model.load_state_dict({k: sd[k].float() for k in sorted(want)})
    return model.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    args = ap.parse_args(OUR_ARGV)
    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"not found: {args.checkpoint}")

    sd = LazySafetensors(args.checkpoint)
    cfg = detect_config(sd)
    model = build_ref_dit(sd, REF_LAYERS)
    del sd

    torch.manual_seed(SEED)
    out: dict[str, torch.Tensor] = {}
    meta = {
        "checkpoint": os.path.basename(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "ref_layers": str(REF_LAYERS),
        "num_layers": str(cfg["num_layers"]),
        "seq_txt": str(SEQ_TXT),
        "cases": json.dumps([c[0] for c in CASES]),
    }

    with torch.no_grad():
        for name, h, w, refs, sigma in CASES:
            x = torch.randn(1, 128, h, w, dtype=torch.float32)
            ctx = torch.randn(1, SEQ_TXT, 2560, dtype=torch.float32)
            # ComfyUI hands the model a bf16 timestep (`MageFlow.process_timestep`),
            # so the fixture's input is the rounded value, not the literal.
            t = torch.tensor([sigma], dtype=torch.bfloat16)
            ref_latents = [torch.randn(1, 128, rh, rw, dtype=torch.float32) for rh, rw in refs]
            y = model(x, t, ctx, ref_latents=ref_latents or None)

            out[f"{name}.x"] = x[0].contiguous()
            out[f"{name}.ctx"] = ctx[0].contiguous()
            out[f"{name}.out"] = y[0].contiguous()
            out[f"{name}.sigma"] = torch.tensor([float(t[0])], dtype=torch.float32)
            for i, r in enumerate(ref_latents):
                out[f"{name}.ref{i}"] = r[0].contiguous()
            meta[f"{name}.shape"] = json.dumps([h, w])
            meta[f"{name}.refs"] = json.dumps(refs)

    os.makedirs(os.path.dirname(REF_OUT), exist_ok=True)
    comfy.utils.save_torch_file(out, REF_OUT, metadata=meta)
    print(f"wrote {REF_OUT} ({len(out)} tensors)")


if __name__ == "__main__":
    main()
