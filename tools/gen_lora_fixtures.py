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
  - the key spelling. ComfyUI accepts seven of them; `SPELLING` puts one
    target in the kohya `lora_down`/`lora_up` pair (what SenseNova's turbo LoRA
    and most civitai files ship) and the rest in PEFT's `lora_A`/`lora_B`, and
    asserts the reference merges both to the same weight. Note the reference's
    own `A_name` holds the UP matrix, so its variable names are inverted from
    PEFT's; only the shapes settle it.
  - N stacked LoRAs ADD. `out_stack` merges two independent files at different
    strengths, which is what says a stack is a sum and not a composition.

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

# Which dialect each target's factors are written in, `a` (down) first. One
# target in the kohya spelling is what gives the alias teeth.
SPELLING = {
    "blocks.0.mlp.fc2":       ("lora_A.weight", "lora_B.weight"),
    "blocks.0.attn.qkv_proj": ("lora_A.weight", "lora_B.weight"),
    "blocks.0.attn.out_proj": ("lora_down.weight", "lora_up.weight"),
}

# The second file in the stacking check, under this prefix in the same fixture
# (safetensors is one flat namespace, and `weights.Prefixed` is how a Zig test
# reads a sub-namespace as a store of its own).
L2 = "l2."
STACK = (0.75, 0.25)


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


def merged(lora, base_w, name, strength):
    """The reference's merge for one target, via comfy.lora, onto `base_w`.

    Applying two adapters in sequence is how a stack is composed: each adds its
    own delta to the weight it is handed.
    """
    key = name + ".weight"
    patches = comfy.lora.load_lora(lora, {"diffusion_model." + name: key}, log_missing=False)
    assert key in patches, "comfy.lora did not recognize %s" % name
    adapter = patches[key]
    return adapter.calculate_weight(base_w.clone(), key, strength, strength, None, lambda v: v)


def main():
    g = torch.Generator().manual_seed(20260827)

    lora = {}
    lora2 = {}
    base = {}
    xs = {}
    for name, (in_dim, out_dim, rank, alpha, groups) in TARGETS.items():
        a_sfx, b_sfx = SPELLING[name]
        a, b = make_lora(g, in_dim, out_dim, rank, groups)
        lora["diffusion_model.%s.%s" % (name, a_sfx)] = a
        lora["diffusion_model.%s.%s" % (name, b_sfx)] = b
        lora["diffusion_model.%s.alpha" % name] = torch.tensor(alpha, dtype=torch.float32)
        # The second file: independent factors, a different alpha, and always the
        # PEFT spelling, so the stack mixes dialects as well as files.
        a2, b2 = make_lora(g, in_dim, out_dim, rank, groups)
        lora2["diffusion_model.%s.lora_A.weight" % name] = a2
        lora2["diffusion_model.%s.lora_B.weight" % name] = b2
        lora2["diffusion_model.%s.alpha" % name] = torch.tensor(alpha * 0.5, dtype=torch.float32)
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
            w = merged(src, base["%s.weight" % name], name, strength)
            out[name] = xs[name] @ w.t()
        return out

    ref = run(STRENGTH)
    half = run(0.5)

    # Both files in sequence, at different strengths: the stack.
    stack = {}
    for name in TARGETS:
        w = merged(lora, base["%s.weight" % name], name, STACK[0])
        w = merged(lora2, w, name, STACK[1])
        stack[name] = xs[name] @ w.t()
    only2 = {}
    for name in TARGETS:
        w = merged(lora2, base["%s.weight" % name], name, STACK[1])
        only2[name] = xs[name] @ w.t()

    tensors = {}
    tensors.update(lora)
    for k, v in lora2.items():
        tensors[L2 + k] = v
    for k, v in base.items():
        tensors["base." + k] = v
    for name in TARGETS:
        tensors["in.%s" % name] = xs[name]
        tensors["out.%s" % name] = ref[name]
        tensors["out_half.%s" % name] = half[name]
        tensors["out_stack.%s" % name] = stack[name]
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

    # The stack really is a sum of the two deltas, not a composition, and it is
    # distinguishable from either file alone. If it were not a sum, adding the
    # sidecars beside the GEMM would not reproduce a merge.
    for name in TARGETS:
        b0 = tensors["out_base." + name]
        d1 = STACK[0] * (ref[name] - b0) / STRENGTH
        d2 = only2[name] - b0
        add = float((stack[name] - (b0 + d1 + d2)).norm() / stack[name].norm())
        assert add < 1e-5, "%s: stacking is not additive (rel %.2e)" % (name, add)
        for other, label in ((ref[name], "file 1"), (only2[name], "file 2")):
            d = float((stack[name] - other).norm() / stack[name].norm())
            assert d > 0.02, "%s: the stack is indistinguishable from %s (rel %.4f)" % (name, label, d)

    # The kohya spelling is not a second code path in the reference: the same
    # factors under either pair of names must merge to the same weight, or the
    # alias in `lora.zig` is pinned against nothing.
    spelled = [n for n in TARGETS if SPELLING[n][0] != "lora_A.weight"]
    assert spelled, "no target ships in the kohya spelling, so the alias has no teeth"
    for name in spelled:
        a_sfx, b_sfx = SPELLING[name]
        as_peft = dict(lora)
        as_peft["diffusion_model.%s.lora_A.weight" % name] = as_peft.pop("diffusion_model.%s.%s" % (name, a_sfx))
        as_peft["diffusion_model.%s.lora_B.weight" % name] = as_peft.pop("diffusion_model.%s.%s" % (name, b_sfx))
        w_a = merged(lora, base["%s.weight" % name], name, STRENGTH)
        w_b = merged(as_peft, base["%s.weight" % name], name, STRENGTH)
        assert torch.equal(w_a, w_b), "%s: %s and lora_A/lora_B disagree" % (name, a_sfx)

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
    a_sfx = SPELLING[tr][0]
    swapped = dict(lora)
    a_key = "diffusion_model.%s.%s" % (tr, a_sfx)
    swapped[a_key] = lora[a_key].t().contiguous()
    alt = run(STRENGTH, use_lora=swapped)[tr]
    d = float((alt - ref[tr]).norm() / ref[tr].norm())
    assert d > 0.05, "reading A transposed is indistinguishable (rel %.4f)" % d

    # The block-diagonal split is an optimization, so confirm the reference's B
    # really is block diagonal (if it were not, our split would drop entries).
    b = lora["diffusion_model.%s.%s" % (fused, SPELLING[fused][1])]
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
        "spelling": json.dumps(SPELLING),
        "stack": json.dumps({"prefix": L2, "strengths": list(STACK)}),
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
