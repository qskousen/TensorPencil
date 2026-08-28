#!/usr/bin/env python3
"""Pin the LoRA sidecar against ComfyUI's own LoRA adapter.

ComfyUI MERGES a LoRA into the weight; we apply it as a runtime sidecar beside
the GEMM, because the H3 trunk is int8 and there is no lossless merge. The two
must agree exactly on the algebra, and what makes them agree is a handful of
conventions that are all silent when wrong:

  - `scale = strength * alpha / mat2.shape[0]`, i.e. divided by the rows of
    `lora_A` AS THE FILE STORES THEM. A fused qkv factor has A [3r, in] and an
    alpha already multiplied by 3, so using the per-block rank is a 3x error.
  - `delta = B @ A`, with A [rank, in] and B [out, rank]. Swapping the roles is
    representable whenever rank == in_dim, which one target here deliberately is.
  - the fused qkv's B is BLOCK DIAGONAL over 3 groups; splitting it must not
    change the answer.
  - the key spelling: `diffusion_model.<base name without .weight>.lora_A.weight`.

The reference is executed, not re-derived: `comfy.lora.load_lora` builds the
adapter and `LoRAAdapter.calculate_weight` produces the merged weight.

Emits src/models/assets/lora.safetensors.

Usage:
    /home/qt/genai/comfyui/nvenv/bin/python tools/gen_lora_fixtures.py
"""

import json
import os
import sys

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "models", "assets", "lora.safetensors",
)

COMFY = "/home/qt/genai/comfyui"
sys.path.insert(0, COMFY)
os.chdir(COMFY)

import torch  # noqa: E402
import safetensors.torch  # noqa: E402

import comfy.lora  # noqa: E402

M = 5  # activation rows

# name -> (in_dim, out_dim, rank, alpha, groups)
#
# `attn.out_proj` has rank == in_dim on purpose: that is the only shape where
# reading A as [in, rank] instead of [rank, in] still multiplies, so it is the
# only shape that can catch the transpose.
TARGETS = {
    "blocks.0.mlp.fc2":        (8, 6, 4, 2.0, 1),
    "blocks.0.attn.qkv_proj":  (8, 18, 12, 6.0, 3),
    "blocks.0.attn.out_proj":  (5, 7, 5, 1.5, 1),
}
STRENGTH = 1.0


def make_lora(g, in_dim, out_dim, rank, groups):
    """`lora_A [rank, in]` and `lora_B [out, rank]`, block diagonal when groups>1."""
    a = torch.randn(rank, in_dim, generator=g, dtype=torch.float32) * 0.4
    b = torch.zeros(out_dim, rank, dtype=torch.float32)
    if groups == 1:
        b = torch.randn(out_dim, rank, generator=g, dtype=torch.float32) * 0.4
    else:
        go, gr = out_dim // groups, rank // groups
        for i in range(groups):
            b[i * go:(i + 1) * go, i * gr:(i + 1) * gr] = (
                torch.randn(go, gr, generator=g, dtype=torch.float32) * 0.4
            )
    return a, b


def merged(lora, base, name, strength):
    """The reference's merge for one target, via comfy.lora."""
    key = name + ".weight"
    patches = comfy.lora.load_lora(lora, {"diffusion_model." + name: key}, log_missing=False)
    assert key in patches, "comfy.lora did not recognize %s" % name
    adapter = patches[key]
    return adapter.calculate_weight(base[key].clone(), key, strength, strength, None, lambda v: v)


def main():
    g = torch.Generator().manual_seed(20260827)

    lora = {}
    base = {}
    xs = {}
    for name, (in_dim, out_dim, rank, alpha, groups) in TARGETS.items():
        a, b = make_lora(g, in_dim, out_dim, rank, groups)
        lora["diffusion_model.%s.lora_A.weight" % name] = a
        lora["diffusion_model.%s.lora_B.weight" % name] = b
        lora["diffusion_model.%s.alpha" % name] = torch.tensor(alpha, dtype=torch.float32)
        base["%s.weight" % name] = torch.randn(out_dim, in_dim, generator=g, dtype=torch.float32) * 0.3
        xs[name] = torch.randn(M, in_dim, generator=g, dtype=torch.float32)

    def run(strength, alpha_scale=1.0, use_lora=None):
        src = dict(lora) if use_lora is None else use_lora
        if alpha_scale != 1.0:
            src = dict(src)
            for k in list(src):
                if k.endswith(".alpha"):
                    src[k] = src[k] * alpha_scale
        out = {}
        for name in TARGETS:
            w = merged(src, base, name, strength)
            out[name] = xs[name] @ w.t()
        return out

    ref = run(STRENGTH)
    half = run(0.5)

    tensors = {}
    tensors.update(lora)
    for k, v in base.items():
        tensors["base." + k] = v
    for name in TARGETS:
        tensors["in.%s" % name] = xs[name]
        tensors["out.%s" % name] = ref[name]
        tensors["out_half.%s" % name] = half[name]
        # The base GEMM on its own, so a Zig test can confirm the sidecar is
        # what moves the answer rather than asserting against a merged weight it
        # also computed.
        tensors["out_base.%s" % name] = xs[name] @ base["%s.weight" % name].t()

    # --- teeth -------------------------------------------------------------
    for name in TARGETS:
        d = float((ref[name] - tensors["out_base." + name]).norm() / tensors["out_base." + name].norm())
        assert d > 0.05, "%s: the LoRA barely moves the output (rel %.4f)" % (name, d)
        # strength really is linear and 0.5 really is different.
        lin = float((half[name] - 0.5 * (ref[name] + tensors["out_base." + name])).norm())
        assert lin < 1e-4, "%s: strength is not linear (%.2e)" % (name, lin)
        hd = float((half[name] - ref[name]).norm() / ref[name].norm())
        assert hd > 0.02, "%s: strength 0.5 is indistinguishable (rel %.4f)" % (name, hd)

    # Dropping the /rank is the classic scale error: it must be visible.
    for name, (_, _, rank, alpha, _) in TARGETS.items():
        no_div = run(STRENGTH, alpha_scale=float(rank))[name]
        d = float((no_div - ref[name]).norm() / ref[name].norm())
        assert d > 0.1, "%s: alpha vs alpha/rank is indistinguishable (rel %.4f)" % (name, d)

    # And for the fused target, using the per-block rank (128-style) instead of
    # the file's 3r is exactly a 3x scale error.
    fused = "blocks.0.attn.qkv_proj"
    triple = run(3.0)[fused]
    d = float((triple - ref[fused]).norm() / ref[fused].norm())
    assert d > 0.1, "the fused 3x alpha error is indistinguishable (rel %.4f)" % d

    # The transpose trap, on the one target where it is representable.
    tr = "blocks.0.attn.out_proj"
    in_dim, out_dim, rank, _, _ = TARGETS[tr]
    assert rank == in_dim, "the transpose target must have rank == in_dim to have teeth"
    swapped = dict(lora)
    swapped["diffusion_model.%s.lora_A.weight" % tr] = lora["diffusion_model.%s.lora_A.weight" % tr].t().contiguous()
    alt = run(STRENGTH, use_lora=swapped)[tr]
    d = float((alt - ref[tr]).norm() / ref[tr].norm())
    assert d > 0.05, "reading A transposed is indistinguishable (rel %.4f)" % d

    # The block-diagonal split is an optimization, so confirm the reference's B
    # really is block diagonal (if it were not, our split would drop entries).
    b = lora["diffusion_model.%s.lora_B.weight" % fused]
    _, out_dim_f, rank_f, _, groups_f = TARGETS[fused]
    go, gr = out_dim_f // groups_f, rank_f // groups_f
    for i in range(groups_f):
        for j in range(groups_f):
            blk = b[i * go:(i + 1) * go, j * gr:(j + 1) * gr]
            if i == j:
                assert blk.abs().max() > 1e-3, "diagonal block %d is empty" % i
            else:
                assert blk.abs().max() == 0.0, "off-diagonal block (%d,%d) is nonzero" % (i, j)

    meta = {
        "targets": json.dumps({k: dict(in_dim=v[0], out_dim=v[1], rank=v[2], alpha=v[3], groups=v[4])
                               for k, v in TARGETS.items()}),
        "m": str(M),
        "strength": json.dumps([STRENGTH, 0.5]),
        "note": "generated by tools/gen_lora_fixtures.py from comfy.lora.load_lora + "
                "LoRAAdapter.calculate_weight; do not hand-edit",
    }
    tensors = {k: v.contiguous() for k, v in tensors.items()}
    safetensors.torch.save_file(tensors, OUT, metadata=meta)
    print("wrote %s: %d tensors, %d bytes" % (OUT, len(tensors), os.path.getsize(OUT)))
    for name in TARGETS:
        rel = float((ref[name] - tensors["out_base." + name]).norm() / tensors["out_base." + name].norm())
        print("  %-26s sidecar moves the output by %.3f" % (name, rel))


if __name__ == "__main__":
    sys.exit(main())
