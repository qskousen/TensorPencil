#!/usr/bin/env python3
"""Reference fixtures for COMPRESSED Mage-Flow ("mageflow-lowrank-modulation-v1").

The architecture is dense Mage-Flow with one substitution, so this pins only what
that substitution decides; `gen_mageflow_fixtures.py` already pins everything
else and its fixture still applies unchanged.

A compressed checkpoint replaces each block's `[6*dim][dim]` modulation linear
with a head onto a rank-r bottleneck produced by one shared `modulation_down`.
Three things about that are choices the weights do not record, and each is a
silent wrong answer:

  - **Where the SiLU goes.** The reference replaces each block's
    `Sequential(SiLU, Linear)` first element with `Identity` and applies SiLU
    ONCE, before the down-projection. Applying it per block as well still
    produces a plausible image.
  - **Which linears are compressed.** The predicate is `in == dim and
    out == 6 * dim`, so `norm_out.linear` (dim -> 2*dim) is NOT compressed and
    still reads `silu(temb)` rather than the bottleneck.
  - **That the factors stay fp32.** The trainer calibrates them in fp32 and the
    node refuses a checkpoint storing them otherwise.

The reference implementation is GPL-3.0 and is FETCHED at generation time, never
vendored, the treatment `gen_a1111_prompt_fixtures.py` gives A1111; the fixture
records the upstream sha256 so a change upstream is visible rather than silently
inherited.

Truncated to 2 layers for `gen_mageflow_fixtures.py`'s reason: an fp32 copy of
the whole trunk does not fit in this machine's RAM, every block is identical, and
the loop bound is covered by the render comparison instead.

Run it memory-bounded — unbounded it can take the machine down:

    systemd-run --user --scope -p MemoryMax=20G -p MemorySwapMax=0 \\
        /home/qt/genai/comfyui/nvenv/bin/python \\
        tools/gen_mageflow_compressed_fixtures.py
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import urllib.request

COMFY = "/home/qt/genai/comfyui"
DEFAULT_CKPT = os.path.join(
    COMFY, "models/diffusion_models/mageflow-compressed/magetrailMageflow4B_v025.safetensors")

# The node's forward, pinned to a commit so a regeneration is reproducible.
UPSTREAM = ("https://raw.githubusercontent.com/bluvoll/ComfyUI-MageFlow-Compressed/"
            "main/model.py")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_OUT = os.path.join(REPO, "src", "models", "assets", "mageflow_compressed_ref.safetensors")

SEED = 20260919
REF_LAYERS = 2
SEQ_TXT = 7

# (name, canvas h, canvas w, [(ref h, ref w), ...], sigma)
#
# Two sigmas because the whole point of the compression is the timestep pathway:
# a mistake in the bottleneck is a function of the timestep and can look fine at
# one of them.
CASES = [
    ("even", 4, 6, [], 0.75),
    ("odd", 5, 7, [], 0.25),
    ("ref", 4, 6, [(3, 5)], 0.5),
]

OUR_ARGV = sys.argv[1:]
sys.argv = [sys.argv[0], "--cpu", "--disable-smart-memory",
            "--disable-xformers", "--use-pytorch-cross-attention"]
sys.path.insert(0, COMFY)
os.chdir(COMFY)

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


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_reference():
    """Fetch the node's model.py and import it. Returns (module, sha256)."""
    src = urllib.request.urlopen(UPSTREAM, timeout=60).read()
    sha = hashlib.sha256(src).hexdigest()
    d = tempfile.mkdtemp(prefix="mfcref-")
    path = os.path.join(d, "mfc_model.py")
    with open(path, "wb") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("mfc_model", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod, sha


class LazySafetensors:
    """Read-through mapping; see `gen_mageflow_fixtures.py` for why."""

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


def checkpoint_params(path: str) -> dict:
    """The exporter's own `model_config`, which is what the node reads."""
    import struct

    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    meta = hdr.get("__metadata__", {})
    if meta.get("architecture") != "mageflow-lowrank-modulation-v1":
        raise SystemExit(f"not a compressed Mage-Flow checkpoint: {meta.get('architecture')!r}")
    return json.loads(meta["model_config"])


def build_ref_dit(mod, sd, params: dict, n_layers: int):
    dim = params["hidden_size"]
    rank = params["modulation_rank"]
    model = mod.CompressedMageFlowTransformer(
        modulation_rank=rank,
        operations=comfy.ops.disable_weight_init,
        num_layers=n_layers,
        num_attention_heads=params["num_heads"],
        attention_head_dim=dim // params["num_heads"],
        in_channels=params["in_channels"],
        out_channels=params["out_channels"],
        joint_attention_dim=params["context_in_dim"],
        axes_dims_rope=params["axes_dim"],
        device="cpu",
        dtype=torch.float32,
    )

    # The substitution actually happened: a block's modulation linear reads the
    # bottleneck and its SiLU is gone. Asserted rather than assumed, because
    # either reverting upstream would leave this fixture pinning dense behaviour
    # under a compressed name.
    head = model.transformer_blocks[0].img_mod[1]
    if head.in_features != rank or head.out_features != 6 * dim:
        raise SystemExit(f"block modulation head is {head.in_features}->{head.out_features}, "
                         f"expected {rank}->{6 * dim}")
    if not isinstance(model.transformer_blocks[0].img_mod[0], torch.nn.Identity):
        raise SystemExit("block modulation still applies its own SiLU; the reference moved")
    if model.norm_out.linear.in_features != dim:
        raise SystemExit("norm_out was compressed too; this generator assumes it is not")
    if model.patch_size != params["patch_size"]:
        raise SystemExit(f"patch_size is {model.patch_size}, checkpoint says {params['patch_size']}")

    want = set(model.state_dict().keys())
    have = set(sd.keys())
    missing = want - have
    if missing:
        raise SystemExit(f"checkpoint is missing {len(missing)} tensors, e.g. {sorted(missing)[:5]}")

    factors = {"modulation_down.weight", "modulation_down.bias"}
    for i in range(n_layers):
        for s in ("img", "txt"):
            for p in ("weight", "bias"):
                factors.add(f"transformer_blocks.{i}.{s}_mod.1.{p}")
    for k in sorted(factors):
        if sd[k].dtype != torch.float32:
            raise SystemExit(f"{k} is {sd[k].dtype}, the calibrated factors must be fp32")

    model.load_state_dict({k: sd[k].float() for k in sorted(want)})
    return model.eval(), rank


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    args = ap.parse_args(OUR_ARGV)
    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"not found: {args.checkpoint}")

    ref_mod, ref_sha = load_reference()
    params = checkpoint_params(args.checkpoint)
    sd = LazySafetensors(args.checkpoint)
    model, rank = build_ref_dit(ref_mod, sd, params, REF_LAYERS)
    del sd

    torch.manual_seed(SEED)
    out: dict[str, torch.Tensor] = {}
    meta = {
        "checkpoint": os.path.basename(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "ref_layers": str(REF_LAYERS),
        "num_layers": str(params["depth"]),
        "modulation_rank": str(rank),
        "seq_txt": str(SEQ_TXT),
        "cases": json.dumps([c[0] for c in CASES]),
        "upstream": UPSTREAM,
        "upstream_sha256": ref_sha,
        "note": "Reference (GPL-3.0) is fetched at generation time, never vendored.",
    }

    with torch.no_grad():
        for name, h, w, refs, sigma in CASES:
            x = torch.randn(1, 128, h, w, dtype=torch.float32)
            ctx = torch.randn(1, SEQ_TXT, 2560, dtype=torch.float32)
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
    print(f"wrote {REF_OUT} ({len(out)} tensors, rank {rank})")
    print(f"  reference sha256 {ref_sha[:16]}…")


if __name__ == "__main__":
    main()
