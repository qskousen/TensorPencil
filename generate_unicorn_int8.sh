#!/usr/bin/env bash
# Regenerate the "AURORA UNICORNS" poster with the int8  convrot checkpoint
# (testdata/comfyui_int8_ref_252469767172722_1120x1680.png).
#
# Params (seed/steps/cfg/size/sampler) match the reference; override any via env:
#   SEED=123 STEPS=8 WIDTH=512 HEIGHT=512 ./generate_unicorn_int8.sh
# BACKEND defaults to zig-cuda; use cpu for the CPU path.
# Models are read from the ComfyUI models dir (NVMe); override MODELS/DIT/VAE/
# TEXT_ENCODER to relocate. MMAP=off switches to buffered reads (for ZFS).
set -euo pipefail
cd "$(dirname "$0")"

MODELS=${MODELS:-$HOME/genai/comfyui/models}
DIT=${DIT:-$MODELS/diffusion_models/krea2/krea2CenterSemiraw_v10Int8.safetensors}
VAE=${VAE:-$MODELS/vae/krea2RealVae_v10.safetensors}
TEXT_ENCODER=${TEXT_ENCODER:-$MODELS/text_encoders/qwen3VLInstruct4bHeretic_v10.safetensors}
MMAP=${MMAP:-on}
BACKEND=${BACKEND:-zig-cuda}
SEED=${SEED:-252469767172722}
STEPS=${STEPS:-20}
CFG=${CFG:-1.0}
WIDTH=${WIDTH:-1120}
HEIGHT=${HEIGHT:-1680}
OUT=${OUT:-scratch_out/int8_unicorn_ref_params.png}

# Positive prompt (verbatim from the reference image's PNG "parameters" chunk).
# Single-quoted heredoc: no escaping of the inner quotes / dashes needed.
read -r -d '' PROMPT <<'PROMPT_EOF' || true
Draw for me a comic image. It shows ladies underwear hanging on a clothes drying rack. The underwear ranges from 1700s large bloomer pants all the way to thongs. Each one is labeled with the year it was invented. As the photo goes from left to right the underwear gets smaller and smaller. The caption at the top says "Proof Global Warming is real!"
PROMPT_EOF

# Build the optimized binary if it's missing (Debug is far too slow for inference).
BIN=zig-out/bin/TensorPencil
if [[ ! -x "$BIN" ]]; then
  echo "building $BIN (ReleaseFast)..."
  zig build -Doptimize=ReleaseFast
fi

mkdir -p "$(dirname "$OUT")"
set -x
exec "$BIN" generate \
  --backend "$BACKEND" \
  --mmap "$MMAP" \
  --dit "$DIT" \
  --vae "$VAE" \
  --text-encoder "$TEXT_ENCODER" \
  --prompt "$PROMPT" \
  --negative "" \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --steps "$STEPS" \
  --cfg "$CFG" \
  --seed "$SEED" \
  --out "$OUT"
