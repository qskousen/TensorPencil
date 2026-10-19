#!/usr/bin/env python3
"""Reference conditioning for MOSS-Image's text encoder
(`comfy/text_encoders/moss_image.py`): stock Qwen3.5-9B, the LAST hidden state
with the final norm applied, over a chat template that ends in an OPEN think
block. ComfyUI tokenizes it with transformers' `Qwen2Tokenizer` (the Qwen2
pretokenizer regex) over Qwen3.5's own 248k vocabulary.

The reference is EXACT weight-only f32: ComfyUI runs on the CPU at fp32, and
every int8 convrot linear is dequantized to f32 just before it runs (the
kitchen's own CPU int8 GEMM quantizes the activation too, and its `dequantize`
rounds to the original bf16). Dequantizing per layer keeps the peak at one
linear instead of a 36 GB f32 model.

⚠️ **ComfyUI rounds every `(1 + w)` RMSNorm scale to bf16.** Its `RMSNorm.scale()`
adds the one in the weight's own dtype, which on this checkpoint is bf16, a 2e-3
error at every norm (measured per op: every GEMM and the gated norm agree with an
f64 recompute to 2e-7). HF and llama.cpp add it in f32. That is storage
rounding, not a convention the model was trained under. ComfyUI also holds the
checkpoint's two F32 tensors per DeltaNet layer (`A_log` and the gated norm's
weight) as bf16 parameters, another 1e-3 per layer. So `cond.*` is taken with
the scale upcast and those tensors restored from the file, and the Zig side
matches it; `cond_comfy.*` is ComfyUI as it ships, stored as the floor rather
than the target.

Also emits `ids.*`, a tokenizer-only corpus (no forward), for the cases a
pretokenizer gets wrong: digits, contractions, CJK, combining marks (decomposed
too, which only an NFC pass turns into the same ids), runs of whitespace and
newlines.

Run it memory-bounded:

    systemd-run --user --scope -p MemoryMax=18G -p MemorySwapMax=0 \\
        /home/qt/genai/comfyui/nvenv/bin/python tools/gen_moss_image_te.py
"""

import argparse
import hashlib
import json
import os
import sys

COMFY = "/home/qt/genai/comfyui"
DEFAULT_TE = "/mnt/watch/models/diffusion/text_encoders/qwen3.5_9b-INT8_CONVROT.safetensors"

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_OUT = os.path.join(REPO, "src", "models", "assets", "moss_image_te_ref.safetensors")

# The first is padded with whitespace because the tokenizer strips it.
PROMPTS = [
    "  a red fox sitting on a mossy rock, morning light\n",
    "",
]

TOKEN_CORPUS = [
    "Hello world! It's 2026, isn't it? 12345 apples.",
    "  leading and trailing spaces  ",
    "line one\nline two\n\n\tindented",
    "東京の夜景、ネオンが光る街",
    "caf\u00e9 na\u00efve r\u00e9sum\u00e9 \u00e9 \u00e4",
    "cafe\u0301 nai\u0308ve \u1100\u1161\u11a8 a\u0323\u0302",  # decomposed: transformers NFCs it first
    "emoji 🦊🌲 and symbols ©®™ <|im_end|> inside",
    "x=3.14159; y = -2e10; z=[1,2,3]",
]

OUR_ARGV = sys.argv[1:]
sys.argv = [sys.argv[0], "--cpu", "--disable-smart-memory", "--disable-xformers",
            "--use-pytorch-cross-attention", "--fp32-text-enc"]
sys.path.insert(0, COMFY)
os.chdir(COMFY)

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import torch  # noqa: E402
from comfy.cli_args import args as comfy_args  # noqa: E402

if not (comfy_args.cpu and comfy_args.fp32_text_enc):
    raise SystemExit("ComfyUI did not take our flags")

import comfy.sd  # noqa: E402
import comfy.text_encoders.llama as comfy_llama  # noqa: E402
import comfy.utils  # noqa: E402
from comfy.quant_ops import QuantizedTensor  # noqa: E402
from safetensors import safe_open  # noqa: E402


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def dequant_f32(w):
    params = w._params
    if getattr(params, "convrot", False):
        return torch.ops.comfy_kitchen.dequantize_int8_convrot_weight_dtype(w._qdata, params.scale, params.convrot_groupsize, 0)
    return torch.ops.comfy_kitchen.dequantize_int8_simple_dtype(w._qdata, params.scale, 0)


def install_dequant_hooks(model) -> int:
    n = 0
    for mod in model.modules():
        w = getattr(mod, "weight", None)
        if not isinstance(w, QuantizedTensor):
            continue

        def pre(m, _inp):
            m._q_weight = m.weight
            m.weight = torch.nn.Parameter(dequant_f32(m._q_weight), requires_grad=False)

        def post(m, _inp, _out):
            m.weight = m._q_weight
            del m._q_weight

        mod.register_forward_pre_hook(pre)
        mod.register_forward_hook(post)
        n += 1
    return n


def exact_f32_params(path, transformer):
    """Swap in the file's F32 tensors where ComfyUI holds a bf16 copy. Returns an undo."""
    saved = []
    with safe_open(path, "pt") as f:
        for name in f.keys():
            if not name.startswith("model.language_model.") or not name.endswith((".A_log", "linear_attn.norm.weight")):
                continue
            prm = transformer.get_parameter(name.replace("model.language_model.", "model."))
            saved.append((prm, prm.data))
            prm.data = f.get_tensor(name).float()

    def undo():
        for prm, data in saved:
            prm.data = data
    return len(saved), undo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-encoder", default=DEFAULT_TE)
    args = ap.parse_args(OUR_ARGV)

    clip = comfy.sd.load_clip(ckpt_paths=[args.text_encoder], embedding_directory=None,
                              clip_type=comfy.sd.CLIPType.MOSS_IMAGE)
    te_name = type(clip.cond_stage_model).__name__
    if "MOSSImage" not in te_name:
        raise SystemExit(f"ComfyUI picked {te_name}, not the MOSS-Image encoder")
    inner = clip.cond_stage_model.qwen35_9b
    if inner.layer != "last":
        raise SystemExit(f"the tap moved to {inner.layer!r}")
    hooked = install_dequant_hooks(inner)
    print(f"{te_name}: {hooked} int8 linears dequantized per call")

    out = {}
    meta = {
        "text_encoder": os.path.basename(args.text_encoder),
        "text_encoder_sha256": sha256(args.text_encoder),
        "prompts": json.dumps(PROMPTS),
        "corpus": json.dumps(TOKEN_CORPUS, ensure_ascii=False),
    }

    for i, text in enumerate(TOKEN_CORPUS):
        ids = [int(t) for t, _w in clip.tokenize(text, llama_template="{}")["qwen35_9b"][0]]
        out[f"ids.{i}"] = torch.tensor(ids, dtype=torch.int32)

    with torch.no_grad():
        for i, text in enumerate(PROMPTS):
            tokens = clip.tokenize(text)
            ids = [int(t) for t, _w in tokens["qwen35_9b"][0]]
            comfy_scale = comfy_llama.RMSNorm.scale
            comfy_llama.RMSNorm.scale = lambda self: self.weight.float() + 1.0 if self.add else self.weight
            n_f32, undo = exact_f32_params(args.text_encoder, inner.transformer)
            cond = clip.encode_from_tokens(tokens, return_pooled=False)
            undo()
            comfy_llama.RMSNorm.scale = comfy_scale
            if n_f32 != 48:
                raise SystemExit(f"restored {n_f32} F32 tensors, expected 2 per DeltaNet layer")
            as_shipped = clip.encode_from_tokens(tokens, return_pooled=False)
            out[f"cond_comfy.{i}"] = as_shipped[0].float().contiguous()
            floor = ((as_shipped - cond).norm() / cond.norm()).item()
            if cond.dtype != torch.float32:
                raise SystemExit(f"conditioning came back {cond.dtype}; the reference must be f32")
            out[f"tokens.{i}"] = torch.tensor(ids, dtype=torch.int32)
            out[f"cond.{i}"] = cond[0].float().contiguous()
            print(f"prompt {i}: {len(ids)} tokens -> {tuple(cond.shape)}, rms {cond.pow(2).mean().sqrt():.4f}, "
                  f"ComfyUI's bf16 storage moves it {floor:.3e}")

    comfy.utils.save_torch_file(out, REF_OUT, metadata=meta)
    print(f"wrote {REF_OUT} ({os.path.getsize(REF_OUT) // 1024} KiB)")


if __name__ == "__main__":
    main()
