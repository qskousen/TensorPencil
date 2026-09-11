#!/usr/bin/env python3
"""Full-depth (42-layer) reference PREFIX for a SenseNova U1.5 edit conditioning.

tools/gen_sensenova_edit_ref.py pins the edit path exactly, in fp32, but only a
few layers deep, because a fp32 SenseNova layer is 1.5 GB. This one covers the
axis that cannot see: a divergence that ACCUMULATES. The block-causal mask an edit
introduces applies at every one of the 42 layers, and a mask that is right at
layer 0 and drifts after it reads as healthy in a shallow check.

Two things make a 42-layer reference fit at all:

- bf16 rather than fp32, which is why the Zig side compares at ~1e-2 rather than
  ~1e-6. That bound is the REFERENCE's precision, not ours;
- only the BASE stream is materialized. The model is built on the meta device and
  the `_mot_gen` copy -- another 16 GB, which the prefix pass never reads -- is
  left unallocated. Nothing about the forward is patched; the weights simply are
  not there for a path that does not run.

⚠️ Bounded: this peaks around 18 GB. Run it under a memory cap.

Usage:
    systemd-run --user --scope -p MemoryMax=26G -p MemorySwapMax=0 \
        /home/qt/genai/comfyui/nvenv/bin/python tools/gen_sensenova_edit_full.py
"""

import sys, os, json, numpy as np
sys.argv=[sys.argv[0],"--cpu","--disable-smart-memory","--disable-xformers","--use-pytorch-cross-attention"]
COMFY="/home/qt/genai/comfyui"; sys.path.insert(0,COMFY); os.chdir(COMFY)
import comfy.options; comfy.options.enable_args_parsing()
import torch, comfy.ops, comfy.utils
from comfy.ldm.sensenova import model as sn
from comfy.ldm.sensenova import conditioning as sn_cond
import comfy.text_encoders.sensenova as sn_te
from safetensors import safe_open

ST="/home/qt/genai/comfyui/models/diffusion_models/sensenova/sensenovaU158BMot_sft.safetensors"
OUT="/dump/projects/zig/TensorPencil/src/models/assets/sensenova_edit_full.safetensors"
PROMPT="make it winter"; REF=(384,512); SEED=20260909
DT=torch.bfloat16

sn.NUM_LAYERS=42
with torch.device("meta"):
    model = sn.SenseNovaU15(device=None, dtype=DT, operations=comfy.ops.disable_weight_init)

src=safe_open(ST, framework="pt")
def assign(name, t):
    mod=model
    parts=name.split('.')
    for p in parts[:-1]:
        mod = mod[int(p)] if p.isdigit() else getattr(mod, p)
    setattr(mod, parts[-1], torch.nn.Parameter(t, requires_grad=False))

need=[]
for name in model.state_dict().keys():
    if "_mot_gen" in name or name.startswith("fm_modules"):
        continue      # the generation copy; the prefix pass never reads it
    need.append(name)
for name in need:
    assign(name, src.get_tensor(name).to(DT))
model.eval()
print("materialized", len(need), "tensors (base stream + embedding + understanding tower)")

rng=np.random.default_rng(SEED)
coarse=rng.random((REF[0]//16+1, REF[1]//16+1, 3), dtype=np.float32)
smooth=np.asarray(torch.nn.functional.interpolate(torch.from_numpy(coarse).permute(2,0,1)[None], size=REF, mode="bicubic", align_corners=False)[0].permute(1,2,0).clamp(0,1))
img_u8=(smooth*255.0+0.5).astype(np.uint8)
img=torch.from_numpy(np.ascontiguousarray(img_u8).astype(np.float32)/255.0)

tok=sn_te.SenseNovaTokenizer()
base=[int(t[0]) for t in tok.tokenize_with_weights(PROMPT)["sensenova_u15"][0]]
refs=[r.to(DT) for r in sn_cond.preprocess_references([img[None]])]
grids=[(max(1,-(-r.shape[-2]//32)), max(1,-(-r.shape[-1]//32))) for r in refs]
ids=sn_cond.condition_input_ids(torch.tensor([base],dtype=torch.long), grids, image_only=False)
idx=sn_cond.thw_indexes(ids, grids)
mask=sn_cond.block_causal_mask(idx, dtype=DT)
with torch.no_grad():
    keys, values, ptime = model.preprocess_prefix(ids, refs, idx, mask)
print("prefix", ids.shape[1], "tokens, time", int(ptime[0]))
out={"base_ids": torch.tensor(base,dtype=torch.int32), "ids": ids[0].to(torch.int32),
     "thw": idx[0].to(torch.int32), "time": torch.tensor([int(ptime[0])],dtype=torch.int32),
     "ref": torch.from_numpy(np.ascontiguousarray(img_u8))}
for i in (0, 41):
    out[f"k.{i}"]=keys[i].squeeze(0).half().contiguous()
    out[f"v.{i}"]=values[i].squeeze(0).half().contiguous()
    print(f"layer {i}: k max {keys[i].float().abs().max():.4f} rms {keys[i].float().pow(2).mean().sqrt():.5f}")
comfy.utils.save_torch_file(out, OUT, metadata={"prompt":PROMPT,"layers":"42","grids":json.dumps(grids),
                                                "dtype":"bfloat16","checkpoint":os.path.basename(ST)})
print("wrote", OUT)
