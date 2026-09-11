#!/usr/bin/env python3
"""Reference fixtures for SenseNova U1.5 8B MoT, from ComfyUI's own
`comfy/ldm/sensenova/` (PR 15922).

SenseNova is not a DiT. It is one Qwen3-shaped trunk carrying TWO weight copies
per layer (Mixture of Transformers): text and reference tokens run the base copy
causally and leave a KV cache, image tokens run the `_mot_gen` copy attending
bidirectionally over themselves concatenated with that prefix KV. It generates in
PIXEL space at 32 px per token, with no VAE and no separate text encoder, and the
head predicts x0 rather than a velocity.

## Two tiers, and why the big one is a WIDTH reduction rather than a rewrite

A fp32 copy of the real model is 65 GB, so an exact reference at real width is not
possible on any machine here. What this generator does instead is run **ComfyUI's
own module objects with narrower weights**: `VisionEmbeddings.forward`,
`ConvDecoder.forward`, `Attention._project`, `DecoderLayer.forward_prefix` /
`forward_generation` and `SenseNovaU15._forward` all read their widths off the
tensors and the module globals, so patching the globals and swapping in smaller
`Conv2d`s executes the SAME code down a narrower pipe. Nothing here re-derives the
reference; the only thing a width cannot pin is a width.

Everything that is width-independent is ALSO pinned at real width, cheaply: the
three-way RoPE split (64 dims at theta 5e6, then 32 + 32 at theta 1e4), the vision
embedder's interleaved 2-D RoPE, the timestep embedding, the resolution noise
scale, the pad-to-32 rule and the sigma schedule.

## The conventions these fixtures exist to catch

Each is a silent wrong answer, not a crash:

- the two vision towers take DIFFERENT input normalization (the understanding
  tower is ImageNet mean/std over [0,1], the generation tower is raw [-1,1]) behind
  identical convolution shapes;
- the vision embedder's RoPE is INTERLEAVED pairs while the trunk's is SPLIT-HALF,
  in the same model, and the vision one puts x on the first half of the channels
  and y on the second;
- `q_norm` covers the first 64 head dims and `q_norm_hw` the last 64, both applied
  BEFORE the rope and before the hw half splits again into h and w;
- every image token shares one t index (the prefix length) and differs only in h/w;
- the generation stream attends with NO mask at all;
- padding to 32 px is replicate up to 16 and then CIRCULAR, and the output is
  cropped back;
- the head predicts x0, and `v = (x - x0) / max(1 - t, 0.02)`.

Usage (ComfyUI's `nvenv`, and bounded, as everything that loads torch here is):
    systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 \
        /home/qt/genai/comfyui/nvenv/bin/python tools/gen_sensenova_fixtures.py
"""

import argparse
import json
import math
import os
import sys

COMFY = "/home/qt/genai/comfyui"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_OUT = os.path.join(REPO, "src", "models", "assets", "sensenova_ref.safetensors")

SEED = 20260909

# ⚠️ ComfyUI parses argv at import time; our own flags must come off `sys.argv`
# before `import comfy.*`. `--cpu` because a reference must not depend on a GPU
# reduction order, `--disable-xformers` because ComfyUI binds its attention
# implementation at import time and prefers xformers, which has no CPU fp32 kernel.
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
from comfy.ldm.sensenova import model as sn  # noqa: E402
from comfy.ldm.sensenova import sampling as sn_sampling  # noqa: E402
from comfy.ldm.sensenova import conditioning as sn_cond  # noqa: E402
from comfy.ldm.modules.diffusionmodules.util import timestep_embedding  # noqa: E402
import comfy.text_encoders.sensenova as sn_te  # noqa: E402

# --- the real architecture, asserted rather than trusted ---------------------
#
# These are the values `src/models/sensenova.zig`'s `u15_8b` config carries. They
# live in ComfyUI as module constants, so a change there must fail here loudly
# instead of silently re-baselining the Zig side.
REAL = {
    "HIDDEN_SIZE": 4096,
    "INTERMEDIATE_SIZE": 12288,
    "NUM_LAYERS": 42,
    "NUM_HEADS": 32,
    "NUM_KV_HEADS": 8,
    "HEAD_DIM": 128,
    "MERGED_PATCH_SIZE": 32,
    "VOCAB_SIZE": 151936,
}

# The narrow model. `dim` must stay divisible by 16: `ConvDecoder` pixel-shuffles
# by 2, 2 and 8, so its second convolution reads dim/16 channels and must always
# write 3 * 8 * 8 = 192. The vision tower's own width is independent of it.
TINY = {
    "HIDDEN_SIZE": 128,
    "INTERMEDIATE_SIZE": 192,
    "NUM_LAYERS": 2,
    "NUM_HEADS": 4,
    "NUM_KV_HEADS": 2,
    "HEAD_DIM": 16,
    "MERGED_PATCH_SIZE": 32,
}
TINY_VIS = 32   # `patch_embedding` output width, 1024 in the real model
TINY_VOCAB = 64


def stats(t):
    f = t.detach().float().reshape(-1)
    return [float(f.mean()), float(f.norm()), float(f.abs().max())]


def check_real_config():
    for k, want in REAL.items():
        got = getattr(sn, k)
        if got != want:
            raise SystemExit(
                f"ComfyUI has sensenova.{k}={got!r}, this generator expects {want!r}. "
                "The architecture moved — do not regenerate until the Zig side is updated.")
    # The head dim splits 64 / 32 / 32; both halves must stay even for rotate-half.
    if REAL["HEAD_DIM"] % 4:
        raise SystemExit("HEAD_DIM must be divisible by 4 for the t/h/w split")


# --- tier 1: width-independent maths, at REAL width --------------------------


def op_fixtures(out, rows, meta):
    """The parts a width reduction would not exercise honestly."""
    g = torch.Generator().manual_seed(SEED)
    hd = REAL["HEAD_DIM"]

    # Three-way RoPE. Positions are deliberately unequal across the axes and
    # include 0, so a table indexed by the wrong axis cannot pass.
    seq = 5
    indexes = torch.stack((
        torch.tensor([0, 1, 2, 7, 7], dtype=torch.long),
        torch.tensor([0, 0, 0, 1, 3], dtype=torch.long),
        torch.tensor([0, 0, 0, 2, 5], dtype=torch.long),
    ))
    rope = sn._prepare_mrope(indexes, torch.device("cpu"), torch.float32)
    # [batch, heads, seq, head_dim], the layout `_apply_llm_rope` consumes.
    q = torch.randn(1, 2, seq, hd, generator=g)
    k = torch.randn(1, 1, seq, hd, generator=g)
    q_t, q_hw = q.chunk(2, dim=-1)
    k_t, k_hw = k.chunk(2, dim=-1)
    q_h, q_w = q_hw.chunk(2, dim=-1)
    k_h, k_w = k_hw.chunk(2, dim=-1)
    o_t = sn._apply_llm_rope(q_t, k_t, rope[0])
    o_h = sn._apply_llm_rope(q_h, k_h, rope[1])
    o_w = sn._apply_llm_rope(q_w, k_w, rope[2])
    out["rope.indexes"] = indexes.to(torch.int32)
    out["rope.q_in"] = q
    out["rope.k_in"] = k
    out["rope.q_out"] = torch.cat((o_t[0], o_h[0], o_w[0]), dim=-1)
    out["rope.k_out"] = torch.cat((o_t[1], o_h[1], o_w[1]), dim=-1)

    # The vision embedder's interleaved 2-D rope, on a channel count small enough
    # to store whole. x on the first half of the channels, y on the second.
    ch = 16
    tokens = 6
    patches = torch.randn(1, tokens, ch, generator=g)
    tw = 3
    idx = torch.arange(tokens)
    first = sn._apply_interleaved_rope(patches[..., : ch // 2], idx % tw, 10000.0)
    second = sn._apply_interleaved_rope(patches[..., ch // 2:], idx // tw, 10000.0)
    out["visrope.in"] = patches[0]
    out["visrope.out"] = torch.cat((first, second), dim=-1)[0]
    out["visrope.grid"] = torch.tensor([1, tokens // tw, tw], dtype=torch.int32)

    # `TimestepEmbedder`'s sinusoid: cos then sin, max_period 1e4, and the argument
    # is the raw t in [0, 1] with NO 1000x factor.
    ts = torch.tensor([0.0, 0.05, 0.25, 0.5, 0.97, 1.0], dtype=torch.float32)
    out["tsembed.t"] = ts
    out["tsembed.out"] = timestep_embedding(ts, 256, max_period=10000)

    # The resolution noise scale, and the value the noise-scale embedder is fed
    # (that same scale over 16). Sizes chosen to straddle the 16.0 cap and to
    # include one that is not a multiple of 32.
    sizes = [(64, 64), (256, 256), (1024, 1024), (1024, 1536), (2048, 2048),
             (4096, 4096), (100, 300), (8192, 8192)]
    out["noisescale.hw"] = torch.tensor(sizes, dtype=torch.int32)
    out["noisescale.scale"] = torch.tensor(
        [sn_sampling.resolution_noise_scale(h, w) for h, w in sizes], dtype=torch.float32)

    # The pad rule: replicate up to 16 on each axis, then CIRCULAR to a multiple
    # of 32. Three shapes: below 16, between, and already aligned.
    for i, (h, w) in enumerate([(5, 9), (40, 33), (32, 64)]):
        img = torch.randn(1, 3, h, w, generator=g)
        out[f"pad.{i}.in"] = img[0]
        out[f"pad.{i}.out"] = sn._pad_to_merged_patch_size(img)[0]

    # The prompt, tokenized by ComfyUI's own `SenseNovaTokenizer`. It is the whole
    # conditioning for this family, so an id off by one is a different render, and
    # the trap is not the template: the vocabulary is Qwen2.5's, whose merge table
    # differs from Qwen3's on any text containing a `#`.
    tok = sn_te.SenseNovaTokenizer()
    prompts = [
        "a cinematic red fox standing in a snowy forest at sunrise, natural colors",
        "",                       # the unconditional branch, a much shorter template
        "a #1 hashtag, C: #include <stdio.h>",   # the Qwen2.5 merge table
        "\u4e00\u53ea\u72d0\u72f8\u5728\u96ea\u5730\u91cc, a fox in the snow",  # mixed scripts
    ]
    meta["prompts"] = json.dumps(prompts)
    for i, text in enumerate(prompts):
        ids = [int(t[0]) for t in tok.tokenize_with_weights(text)["sensenova_u15"][0]]
        out[f"tok.{i}"] = torch.tensor(ids, dtype=torch.int32)

    # Image editing, pinned at the REAL vocabulary because that is what the splice
    # is written in. These three are pure functions of the ids and the reference
    # grids, and each is a silent wrong answer: the blocks go at the start of the
    # USER turn (not the end of the prompt), the labels appear only when there is
    # more than one reference, every context token of one reference shares ONE time
    # index, and the mask is that block structure OR'd with plain causality.
    #
    # The negative branch of an edit is `image_only`, which is NOT the
    # unconditional text prompt: it drops the prompt and presents the pictures.
    edit_base = [int(t[0]) for t in tok.tokenize_with_weights(
        "make the fox blue")["sensenova_u15"][0]]
    edit_cases = [
        {"grids": [(2, 3)], "image_only": False},
        {"grids": [(2, 3), (1, 2)], "image_only": False},
        {"grids": [(2, 3), (1, 2)], "image_only": True},
    ]
    meta["edit_cases"] = json.dumps([{"grids": c["grids"], "image_only": c["image_only"]} for c in edit_cases])
    base_t = torch.tensor([edit_base], dtype=torch.long)
    out["edit.base"] = torch.tensor(edit_base, dtype=torch.int32)
    for i, c in enumerate(edit_cases):
        ids = sn_cond.condition_input_ids(base_t, c["grids"], image_only=c["image_only"])
        thw = sn_cond.thw_indexes(ids, c["grids"])
        mask = sn_cond.block_causal_mask(thw, dtype=torch.float32)
        out[f"edit.{i}.ids"] = ids[0].to(torch.int32)
        out[f"edit.{i}.thw"] = thw[0].to(torch.int32)
        # The mask as an allow BITMAP: `-inf` is "blocked", 0 is "allowed", and a
        # float matrix would be three times the bytes for the same two values.
        out[f"edit.{i}.allow"] = (mask[0, 0] == 0).to(torch.uint8)

    # `preprocess_reference`: the UNDERSTANDING tower's ImageNet normalization over
    # [0, 1], which the generation tower does NOT apply to its own input.
    img01 = torch.rand(1, 5, 7, 3, generator=g)
    out["refnorm.in"] = img01[0]
    out["refnorm.out"] = sn_cond.preprocess_reference(img01)[0]

    # The sigma schedule ComfyUI's "normal" scheduler samples, at the shift the
    # workflow ships with and at the identity shift.
    for shift in (1.0, 3.0):
        key = f"sigmas.{shift:g}"
        out[key] = sn_sampling.upstream_sigmas(20, shift)
        rows[key] = stats(out[key])


# --- tier 2: the whole architecture, narrowed --------------------------------


def patch_widths():
    for k, v in TINY.items():
        setattr(sn, k, v)


def build_tiny():
    """ComfyUI's own `SenseNovaU15`, with the narrow widths in place.

    Two members are swapped after construction because their widths are written
    into `__init__` as literals rather than read from the module globals. The
    forwards are untouched and read every width off the tensors.
    """
    ops = comfy.ops.disable_weight_init
    dd = dict(device="cpu", dtype=torch.float32)
    model = sn.SenseNovaU15(operations=ops, **dd)

    dim = TINY["HIDDEN_SIZE"]
    for tower in (model.vision_model, model.fm_modules["vision_model_mot_gen"]):
        tower.embeddings.patch_embedding = ops.Conv2d(3, TINY_VIS, kernel_size=16, stride=16, **dd)
        tower.embeddings.dense_embedding = ops.Conv2d(TINY_VIS, dim, kernel_size=2, stride=2, **dd)
    head = model.fm_modules["fm_head"]
    head.conv1 = ops.Conv2d(dim // 4, dim // 4, kernel_size=3, padding=1, **dd)
    head.conv2 = ops.Conv2d(dim // 16, 192, kernel_size=3, padding=1, **dd)
    # `padding_idx` only ever affects gradients, so a small vocabulary with a
    # different one is the same function at inference.
    model.language_model.model.embed_tokens = ops.Embedding(TINY_VOCAB, dim, padding_idx=0, **dd)
    return model.eval()


def seed_weights(model):
    """⚠️ `disable_weight_init` leaves every parameter UNINITIALIZED — it allocates
    with `torch.empty` and skips reset_parameters. Filling them is not optional.

    The values are rounded to f16 before they are used, because that is the
    precision the fixture stores them at: rounding afterwards would put a
    difference in the *input* of every comparison that reads them back.
    """
    g = torch.Generator().manual_seed(SEED + 1)
    with torch.no_grad():
        for name, p in sorted(model.state_dict().items()):
            v = torch.randn(p.shape, generator=g, dtype=torch.float32)
            # Norm scales sit near 1 in a trained model; centring them there keeps
            # the narrow model's activations in a sane range over two layers.
            if name.endswith("norm.weight") or "layernorm" in name or "_norm" in name:
                v = 1.0 + 0.1 * v
            else:
                v = v / math.sqrt(p.shape[-1] if p.ndim > 1 else 1.0)
            p.copy_(v.half().float())


def capture_layers(model, cap):
    """Wrap the two `DecoderLayer` entry points. They are called as plain methods
    rather than through `__call__`, so a forward hook never fires on them."""
    cls = type(model.language_model.model.layers[0])
    orig_prefix = cls.forward_prefix
    orig_gen = cls.forward_generation
    index = {id(layer): i for i, layer in enumerate(model.language_model.model.layers)}

    def prefix(self, *a, **kw):
        r = orig_prefix(self, *a, **kw)
        i = index[id(self)]
        cap[f"prefix.{i}.h"] = r[0].detach().clone()
        cap[f"prefix.{i}.k"] = r[1].detach().clone()
        cap[f"prefix.{i}.v"] = r[2].detach().clone()
        return r

    def generation(self, *a, **kw):
        r = orig_gen(self, *a, **kw)
        cap[f"gen.{index[id(self)]}.h"] = r.detach().clone()
        return r

    cls.forward_prefix = prefix
    cls.forward_generation = generation
    return lambda: (setattr(cls, "forward_prefix", orig_prefix),
                    setattr(cls, "forward_generation", orig_gen))


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args(OUR_ARGV)

    check_real_config()

    out = {}
    rows = {}
    meta = {"real_config": json.dumps(REAL)}

    op_fixtures(out, rows, meta)

    patch_widths()
    cfg = dict(TINY)
    cfg["VIS_DIM"] = TINY_VIS
    cfg["VOCAB_SIZE"] = TINY_VOCAB
    meta["tiny_config"] = json.dumps(cfg)

    torch.manual_seed(SEED)
    model = build_tiny()
    seed_weights(model)
    for name, p in sorted(model.state_dict().items()):
        out[f"w.{name}"] = p.detach().half()

    cap = {}
    restore = capture_layers(model, cap)

    # Two cases. Case 0 is 96x64 px = 3x2 tokens, both axes already aligned, with
    # a 9-token prompt. Case 1 is 40x72 px, which pads to 64x96 and then CROPS the
    # velocity back to 40x72, on a 4-token prompt at a different sigma. A case that
    # happened to be aligned in both would let a missing pad or a missing crop pass.
    cases = [
        {"ids": [3, 17, 5, 40, 9, 61, 2, 33, 12], "h": 96, "w": 64, "sigma": 0.75},
        {"ids": [7, 22, 58, 1], "h": 40, "w": 72, "sigma": 0.25},
    ]

    with torch.no_grad():
        for ci, case in enumerate(cases):
            cap.clear()
            ids = torch.tensor([case["ids"]], dtype=torch.long)
            x = torch.randn(1, 3, case["h"], case["w"],
                            generator=torch.Generator().manual_seed(SEED + 10 + ci))
            # ComfyUI hands the model `1 - sigma` (`SenseNovaU15.process_timestep`
            # composed with the flow sampling's `timestep`), so the model's own
            # `timesteps` is not the sigma. Feeding sigma here would be a plausible
            # image at the wrong point on the trajectory.
            t = torch.tensor([1.0 - case["sigma"]], dtype=torch.float32)

            v = model._forward(x, t, text_input_ids=ids)

            out[f"fwd.{ci}.ids"] = ids[0].to(torch.int32)
            out[f"fwd.{ci}.x"] = x[0]
            out[f"fwd.{ci}.t"] = t
            out[f"fwd.{ci}.sigma"] = torch.tensor([case["sigma"]], dtype=torch.float32)
            out[f"fwd.{ci}.v"] = v[0]
            for name, t_ in cap.items():
                key = f"fwd.{ci}.{name}"
                out[key] = t_.squeeze(0)
                rows[key] = stats(t_)
            rows[f"fwd.{ci}.v"] = stats(v)

            # The prefix cache on its own, the form `Session.encode` produces. It is
            # recomputed here rather than sliced out of the run above so that the
            # standalone entry point is pinned too: `extra_conds` calls it, and the
            # in-forward path is only taken when a LoRA hook is active.
            keys, values, ptime = model.preprocess_prefix(ids)
            out[f"pre.{ci}.time"] = ptime.to(torch.int32)
            for li, (kk, vv) in enumerate(zip(keys, values)):
                out[f"pre.{ci}.k.{li}"] = kk.squeeze(0)
                out[f"pre.{ci}.v.{li}"] = vv.squeeze(0)

    restore()

    meta["stats"] = json.dumps(rows)
    os.makedirs(os.path.dirname(REF_OUT), exist_ok=True)
    # Every capture is already a clone, so this only repacks the strides of the
    # ones that came out of a transpose.
    out = {k: v.contiguous() for k, v in out.items()}
    comfy.utils.save_torch_file(out, REF_OUT, metadata=meta)
    total = sum(v.numel() * v.element_size() for v in out.values())
    print(f"wrote {REF_OUT} ({total / 1e6:.2f} MB, {len(out)} tensors, {len(rows)} stat rows)")


if __name__ == "__main__":
    main()
