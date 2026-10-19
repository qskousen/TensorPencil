#!/usr/bin/env python3
"""Reference conditioning for Qwen-Image 2.1's encoder
(`comfy/text_encoders/qwen_image21.py`): Qwen3-VL-8B, the last layer's output with
NO final norm, over a template whose system turn is then stripped (everything
before the second `<|im_start|>`). An empty prompt is encoded as a single space.

ComfyUI on the CPU at fp32 with the fp8-scaled checkpoint: each weight is the
file's fp8 times its scale, upcast per op, which is what the Zig side computes
too. `cond_normed` is the control: the same tap WITH the final norm, which the
Zig test asserts is far from `cond`, so the comparison has teeth.

Run it memory-bounded:
    systemd-run --user --scope -p MemoryMax=20G -p MemorySwapMax=0 \\
        /home/qt/genai/comfyui/nvenv/bin/python tools/gen_qwen_image21_te.py
"""
import os
import sys

TE = "/mnt/watch/models/diffusion/text_encoders/qwen3vl_8b_fp8_scaled.safetensors"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "src", "models", "assets", "qwen_image21_te_ref.safetensors")
PROMPTS = ["a red fox sitting on a mossy rock, morning light", ""]

sys.argv = [sys.argv[0], "--cpu", "--disable-smart-memory", "--disable-xformers",
            "--use-pytorch-cross-attention", "--fp32-text-enc"]
sys.path.insert(0, "/home/qt/genai/comfyui")
os.chdir("/home/qt/genai/comfyui")
import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()
import json  # noqa: E402
import torch  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402

clip = comfy.sd.load_clip(ckpt_paths=[TE], embedding_directory=None, clip_type=comfy.sd.CLIPType.QWEN_IMAGE)
name = type(clip.cond_stage_model).__name__
if "QwenImage21" not in name:
    raise SystemExit(f"ComfyUI picked {name}, not the Qwen-Image 2.1 encoder")
inner = clip.cond_stage_model.qwen3vl_8b
if inner.layer_norm_hidden_state:
    raise SystemExit("the tap is normed; the fixture pins the unnormed one")

out, meta = {}, {"prompts": json.dumps(PROMPTS)}
with torch.no_grad():
    for i, text in enumerate(PROMPTS):
        tokens = clip.tokenize(text, prevent_empty_text=True)
        ids = [int(t) for t, _w in tokens["qwen3vl_8b"][0]]
        cond = clip.encode_from_tokens(tokens, return_pooled=False)
        if cond.dtype != torch.float32:
            raise SystemExit(f"conditioning came back {cond.dtype}")
        inner.layer_norm_hidden_state = True
        normed = clip.encode_from_tokens(tokens, return_pooled=False)
        inner.layer_norm_hidden_state = False
        out[f"tokens.{i}"] = torch.tensor(ids, dtype=torch.int32)
        out[f"cond.{i}"] = cond[0].float().contiguous()
        out[f"cond_normed.{i}"] = normed[0].float().contiguous()
        print(f"prompt {i}: {len(ids)} tokens -> {tuple(cond.shape)}, rms {cond.pow(2).mean().sqrt():.3f}")
comfy.utils.save_torch_file(out, OUT, metadata=meta)
print("wrote", OUT, os.path.getsize(OUT) // 1024, "KiB")
