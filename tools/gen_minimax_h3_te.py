#!/usr/bin/env python3
"""Pin MiniMax H3's conditioning encoder NUMERICALLY by executing ComfyUI's own
Qwen3-VL text trunk at a toy width.

The real encoder is 27 GB and cannot be a unit fixture, but nothing that makes it
what it is depends on the width: the pre-norm placement, the per-head q/k RMS norm
before rope, the 5e6 rope theta, grouped-query attention, the swiglu half order,
and -- the one this variant alone does -- consuming the UNNORMALIZED state after
the last layer, because this checkpoint has no final norm at all.

Two cases, because the second exercises three things the first cannot:

  1. **text only**, which is t2va and the path every render takes.
  2. **text with a spliced vision block**, which is fl2va/ref2va: embedding rows
     overwritten at the block's positions, the switch to multimodal rope, and
     DEEPSTACK features added at the image positions after layers 0, 1 and 2.

Conventions that are silent when wrong:

  - ⚠️ **`head_dim` is 128 and is NOT `hidden / n_heads`.** The real model is 5120
     wide over 64 heads, so q/k/v project to 8192, not 5120. The toy keeps that
     asymmetry (256 wide, 4 heads, 512 inner) because a square encoder passes an
     o_proj transpose by accident.
  - ⚠️ **No final norm.** `Qwen3VL_32BConfig` sets `final_norm=False`, so the
     forward returns the raw residual stream. Applying an RMSNorm that the
     checkpoint does not even contain is finite and wrong.
  - the rope theta is Qwen3-VL's 5e6, not plain Qwen3's 1e6, and it is not
    recoverable from the weights.
  - ⚠️ **DeepStack is added AFTER layer i**, at image positions only, for as many
    layers as there are features -- collected deep in the vision tower, injected
    shallow in the language model.
  - ⚠️ **Vision blocks switch the LLM to mrope.** With no image the reference uses a
    plain `arange`, so the text-only case pins the 1-D path and the vision case the
    3-axis one; a port that always used one of them passes exactly one of these.

⚠️ **`Llama2_.forward` mutates the `embeds` you hand it and RETURNS THAT TENSOR.**
Its residual adds are in place, so the trunk runs on the caller's buffer. Two
consequences, and the second cost a debugging pass here: every call needs its own
clone of the embeddings, and a stored output must be `.clone()`d, because
`out[0].contiguous()` on an already-contiguous view is a VIEW -- so a later
sensitivity check silently overwrote the very tensor it was being compared against
and reported a difference of exactly zero.

Emits src/models/assets/minimax_h3_te.safetensors.

Usage (BOUND IT -- this builds a model):
    systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 \
        /home/qt/genai/comfyui/nvenv/bin/python tools/gen_minimax_h3_te.py
"""

import json
import os
import sys

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "models", "assets", "minimax_h3_te.safetensors",
)

COMFY = "/home/qt/genai/comfyui"
sys.path.insert(0, COMFY)
os.chdir(COMFY)

import torch  # noqa: E402
import safetensors.torch  # noqa: E402

import comfy.ops  # noqa: E402
from comfy.text_encoders.llama import Llama2_, Qwen3VL_32BConfig  # noqa: E402
from comfy.text_encoders.qwen_vl import qwen2vl_mrope_position_ids  # noqa: E402

VOCAB = 512
HIDDEN = 128
INTER = 256
LAYERS = 3
HEADS = 2
KV_HEADS = 1
SEQ = 20
# FEWER deepstack features than layers, so the rule that injection stops after the
# last one is pinned rather than coinciding with the layer count.
DEEPSTACK = 2
# The vision block: 4 rows at index 6, from a 4x4 pre-merge grid.
VIS_AT = 6
VIS_GH, VIS_GW = 4, 4
VIS_ROWS = (VIS_GH // 2) * (VIS_GW // 2)


def main():
    torch.manual_seed(20260828)
    torch.set_num_threads(4)

    cfg = Qwen3VL_32BConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
    )
    # `head_dim`, `rope_dims`, `q_norm`/`k_norm`, `final_norm` and `lm_head` are
    # class attributes rather than dataclass fields, so they are inherited as-is.
    # That is what keeps the toy honest: head_dim stays 128 and the final norm
    # stays absent.
    assert cfg.head_dim == 128, cfg.head_dim
    assert cfg.final_norm is False and cfg.lm_head is False
    assert cfg.rope_theta == 5000000.0
    assert cfg.rope_dims == [24, 20, 20]
    assert cfg.head_dim * HEADS != HIDDEN, "the toy lost the hidden != inner asymmetry"
    assert DEEPSTACK < LAYERS, "deepstack must stop before the last layer here"

    model = Llama2_(cfg, device="cpu", dtype=torch.float32, ops=comfy.ops.disable_weight_init)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print("toy encoder: %d parameters (%.2f MB f32)" % (n_params, n_params * 4 / 1048576))
    assert n_params < 5_000_000, "the 'toy' encoder is not toy: %d parameters" % n_params
    assert model.norm is None, "this variant must have no final norm"

    sd = {}
    for name, p in model.state_dict().items():
        if name.endswith("norm.weight"):
            # RMSNorm gammas, near 1 so the norms are not degenerate.
            sd[name] = 1.0 + torch.randn(p.shape, dtype=torch.float32) * 0.1
        elif p.dim() == 2:
            sd[name] = torch.randn(p.shape, dtype=torch.float32) / (p.shape[1] ** 0.5)
        else:
            sd[name] = torch.randn(p.shape, dtype=torch.float32) * 0.05
    model.load_state_dict(sd, strict=True)
    for prm in model.parameters():
        prm.requires_grad_(False)

    ids = torch.randint(0, VOCAB, (1, SEQ), dtype=torch.long)

    # --- case 1: text only -------------------------------------------------
    with torch.inference_mode():
        text_out, _ = model(ids, dtype=torch.float32)
        text_out = text_out.clone()
    assert text_out.shape == (1, SEQ, HIDDEN), text_out.shape
    assert torch.isfinite(text_out).all()
    print("  text only : %s, std %.4f" % (list(text_out.shape), float(text_out.std())))

    # --- case 2: a spliced vision block ------------------------------------
    # The rows the tower would have produced, and one deepstack feature per layer.
    vis_rows = torch.randn(VIS_ROWS, HIDDEN, dtype=torch.float32) * 0.5
    deepstack = [torch.randn(VIS_ROWS, HIDDEN, dtype=torch.float32) * 0.3 for _ in range(DEEPSTACK)]

    with torch.inference_mode():
        embeds = model.embed_tokens(ids, out_dtype=torch.float32).clone()
        embeds[0, VIS_AT:VIS_AT + VIS_ROWS] = vis_rows

        grid = torch.tensor([[1, VIS_GH, VIS_GW]], dtype=torch.long)
        info = [{"type": "image", "index": VIS_AT, "size": VIS_ROWS, "extra": {"grid": grid}}]
        pos = qwen2vl_mrope_position_ids(info, SEQ, torch.device("cpu"))
        assert pos is not None and pos.shape == (3, SEQ), pos.shape

        mask = torch.zeros(1, SEQ, dtype=torch.bool)
        mask[0, VIS_AT:VIS_AT + VIS_ROWS] = True

        vis_out, _ = model(
            None, embeds=embeds.clone(), dtype=torch.float32,
            position_ids=pos,
            deepstack_embeds=deepstack, visual_pos_masks=mask,
        )
        vis_out = vis_out.clone()
    assert vis_out.shape == (1, SEQ, HIDDEN), vis_out.shape
    assert torch.isfinite(vis_out).all()
    print("  + vision  : %s, std %.4f" % (list(vis_out.shape), float(vis_out.std())))

    tensors = {}
    for k, v in sd.items():
        tensors["model." + k] = v
    tensors["in.ids"] = ids.to(torch.int32)
    tensors["out.text"] = text_out[0].clone()
    tensors["in.vision_rows"] = vis_rows
    tensors["in.positions"] = pos.to(torch.float32)
    for i, d in enumerate(deepstack):
        tensors["in.deepstack.%d" % i] = d
    tensors["out.vision"] = vis_out[0].clone()

    # --- teeth -------------------------------------------------------------
    def rel(a, b):
        return float((a - b).norm() / b.norm())

    # 1. The two cases must differ, or the vision payload is inert.
    d_case = rel(vis_out, text_out)
    assert d_case > 0.1, "the vision case is indistinguishable from text only (%.3f)" % d_case

    # 2. A FINAL NORM applied on the way out. This variant has none, and the
    #    checkpoint contains no `model.norm.weight` to apply, so the substitution is
    #    a unit RMSNorm -- which still has to move the answer.
    normed = text_out / torch.sqrt(text_out.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps)
    d_norm = rel(normed, text_out)
    assert d_norm > 1e-2, "a final norm is indistinguishable (%.2e)" % d_norm

    # 3. The rope theta. 1e6 (plain Qwen3) against this model's 5e6.
    cfg1e6 = Qwen3VL_32BConfig(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=INTER,
        num_hidden_layers=LAYERS, num_attention_heads=HEADS, num_key_value_heads=KV_HEADS,
        rope_theta=1000000.0,
    )
    alt = Llama2_(cfg1e6, device="cpu", dtype=torch.float32, ops=comfy.ops.disable_weight_init)
    alt.load_state_dict(sd, strict=True)
    alt.eval()
    with torch.inference_mode():
        alt_out, _ = alt(ids, dtype=torch.float32)
        alt_out = alt_out.clone()
    d_theta = rel(alt_out, text_out)
    assert d_theta > 1e-3, "the rope theta is indistinguishable (%.2e)" % d_theta

    # 4. DeepStack injected at EVERY position instead of the image ones only.
    with torch.inference_mode():
        all_mask = torch.ones(1, SEQ, dtype=torch.bool)
        wide = [d[:1].expand(SEQ, HIDDEN).contiguous() for d in deepstack]
        wrong, _ = model(
            None, embeds=embeds.clone(), dtype=torch.float32, position_ids=pos,
            deepstack_embeds=wide, visual_pos_masks=all_mask,
        )
    assert wrong.data_ptr() != vis_out.data_ptr(), "the two outputs alias; the clone did not take"
    d_mask = rel(wrong, vis_out)
    assert d_mask > 1e-2, "the deepstack position mask is indistinguishable (%.2e)" % d_mask

    # 5. mrope against the plain 1-D rope, with the same spliced embeddings. A port
    #    that never switched would differ only here.
    with torch.inference_mode():
        flat, _ = model(
            None, embeds=embeds.clone(), dtype=torch.float32,
            deepstack_embeds=deepstack, visual_pos_masks=mask,
        )
    d_rope = rel(flat, vis_out)
    assert d_rope > 1e-3, "mrope and the 1-D rope are indistinguishable (%.2e)" % d_rope

    meta = {
        "config": json.dumps(dict(vocab=VOCAB, hidden=HIDDEN, intermediate=INTER,
                                  layers=LAYERS, heads=HEADS, kv_heads=KV_HEADS,
                                  head_dim=cfg.head_dim, seq=SEQ,
                                  rope_theta=cfg.rope_theta, rope_dims=cfg.rope_dims,
                                  rms_eps=cfg.rms_norm_eps,
                                  deepstack=DEEPSTACK,
                                  vis_at=VIS_AT, vis_rows=VIS_ROWS,
                                  vis_grid=[VIS_GH, VIS_GW])),
        "note": "generated by tools/gen_minimax_h3_te.py from "
                "comfy.text_encoders.llama.Llama2_ with Qwen3VL_32BConfig; do not hand-edit",
    }
    tensors = {k: v.contiguous() for k, v in tensors.items()}
    safetensors.torch.save_file(tensors, OUT, metadata=meta)
    print("wrote %s: %d tensors, %.3f MB" % (OUT, len(tensors), os.path.getsize(OUT) / 1048576))
    print("  sensitivity: vision payload %.3f  final norm %.4f  theta %.4f  "
          "deepstack mask %.4f  mrope %.4f" % (d_case, d_norm, d_theta, d_mask, d_rope))


if __name__ == "__main__":
    sys.exit(main())
