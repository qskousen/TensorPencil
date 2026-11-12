#!/usr/bin/env python3
"""Reference fixture for ComfyUI's int8 attention (comfy-kitchen's "sage" SDPA), which
a checkpoint asks for with a `comfy_attention.config` tensor holding
`{"attention": "comfy_kitchen_int8"}`.

It runs comfy-kitchen's own CUDA kernels, the three launches the int8 path makes on
sm_80 to sm_88 with no mask: the K-anchor detector and Q/K quantizer, the V quantizer,
and the attention kernel. It records the bf16 inputs, every intermediate and the bf16
output:

    q, k, v            BF16  [heads, len, 128]
    q_int8, k_int8     I8    [heads, len, 128]      rotated, natural channel order
    q_scale            F32   [heads, ceil(lq/128)*32]
    k_scale            F32   [heads, ceil(lk/128)*4]
    v_int8             I8    [heads*128, ceil(lk/128)*128]   per 16 keys permuted
    v_scale            F32   [heads*128]
    anchor             I32   [heads]                -1 when no key is subtracted
    out                BF16  [heads, lq, 128]

The shape takes the path the H3 and MOSS denoisers do: head dim 128, more than 1024
keys (128-key tiles), a last key tile part full and a query tail inside a scale group.
Head 0's keys share an offset, so the anchor is used; head 1's are centred, so it is
not. The generator asserts both, that the one-call `int8_attention` stores the same
bytes, and that the output is farther from exact attention than bf16 rounding.

Run on an sm_80..sm_88 GPU (the 3090), in ComfyUI's environment:
    /home/qt/genai/comfyui/nvenv/bin/python -I tools/gen_sage_attn_fixture.py
"""

import importlib.metadata
import os
import sys

import torch
from safetensors.torch import save_file

import comfy_kitchen
from comfy_kitchen.backends import cuda as ck_cuda
from comfy_kitchen.backends.eager.quantization import DTYPE_TO_CODE

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "ops", "assets", "sage_attn.safetensors")

HEADS, LQ, LK, D = 2, 136, 1100, 128
CTA_K = 128


def W(t):
    return ck_cuda._wrap_for_dlpack(t)


def main():
    dev = torch.device("cuda")
    cap = torch.cuda.get_device_capability(dev)
    if not (8, 0) <= cap < (8, 9):
        sys.exit("needs an sm_80..sm_88 GPU, the path this fixture pins; got sm_%d%d" % cap)

    g = torch.Generator().manual_seed(20261111)
    q = torch.randn(HEADS, LQ, D, generator=g) * 2.0
    k = torch.randn(HEADS, LK, D, generator=g) * 2.0
    v = torch.randn(HEADS, LK, D, generator=g)
    # outlier channels, which the rotation spreads before quantizing
    q[:, :, 5] *= 6
    k[:, :, 77] *= 5
    v[:, :, 3] *= 8
    # a shared key offset in head 0 only, which the anchor removes
    k[0] += torch.randn(D, generator=g) * 4.0
    q, k, v = (t.to(torch.bfloat16).unsqueeze(0).contiguous().to(dev) for t in (q, k, v))

    stream = torch.cuda.current_stream(dev).cuda_stream
    pad_k = (LK + CTA_K - 1) // CTA_K * CTA_K
    q_int8 = torch.empty_like(q, dtype=torch.int8)
    k_int8 = torch.empty_like(k, dtype=torch.int8)
    q_scale = torch.empty(1, HEADS, (LQ + 127) // 128 * 32, dtype=torch.float32, device=dev)
    k_scale = torch.empty(1, HEADS, (LK + CTA_K - 1) // CTA_K * 4, dtype=torch.float32, device=dev)
    v_int8 = torch.empty(HEADS * D, pad_k, dtype=torch.int8, device=dev)
    v_scale = torch.empty(HEADS * D, dtype=torch.float32, device=dev)
    anchor = torch.empty(1, HEADS, dtype=torch.int32, device=dev)
    # what `prequantize_int8_attention` does, keeping the anchor indices it drops
    ck_cuda._C.sage_sdpa_quantize(W(q), W(k), W(v), W(q_int8), W(q_scale), W(k_int8), W(k_scale),
                                  W(v_int8), W(v_scale), CTA_K, DTYPE_TO_CODE[torch.bfloat16],
                                  stream, anchor.data_ptr())
    out = torch.empty(1, HEADS, LQ, D, dtype=torch.bfloat16, device=dev)
    scale = float(D ** -0.5)
    ck_cuda._C.sage_sdpa_prequantized(W(q_int8), W(k_int8), W(v_int8), W(out), W(q_scale), W(k_scale),
                                      W(v_scale), CTA_K, scale, DTYPE_TO_CODE[torch.bfloat16], stream, None)
    torch.cuda.synchronize()

    one_call = comfy_kitchen.int8_attention(q, k, v)
    assert torch.equal(one_call.view(torch.int16), out.view(torch.int16)), "one-call path stores other bytes"

    a = anchor.cpu().flatten().tolist()
    assert a[0] >= 0 and a[1] == -1, "anchor use should be [yes, no], got %s" % a
    assert not k_int8[0, 0, a[0]].any(), "the anchor key quantizes to zero"

    exact = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())
    err = (out.float() - exact).norm() / exact.norm()
    bf16_err = (exact.to(torch.bfloat16).float() - exact).norm() / exact.norm()
    print("anchor %s, rel err vs exact %.4f (bf16 rounding alone %.4f)" % (a, err, bf16_err))
    assert err > 4 * bf16_err, "int8 attention indistinguishable from exact"

    t = {
        "q": q[0], "k": k[0], "v": v[0],
        "q_int8": q_int8[0], "k_int8": k_int8[0], "q_scale": q_scale[0], "k_scale": k_scale[0],
        "v_int8": v_int8, "v_scale": v_scale, "anchor": anchor[0], "out": out[0],
    }
    meta = {"comfy_kitchen": importlib.metadata.version("comfy-kitchen"),
            "device": torch.cuda.get_device_name(dev), "scale": repr(scale)}
    save_file({n: x.contiguous().cpu() for n, x in t.items()}, OUT, metadata=meta)
    print("wrote", os.path.normpath(OUT), "comfy-kitchen", meta["comfy_kitchen"])


if __name__ == "__main__":
    main()
