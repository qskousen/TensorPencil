#!/usr/bin/env bash
# Regenerate the "AURORA UNICORNS" poster with the int4 (W4A4) convrot checkpoint,
# using the exact prompt + params pulled from the ComfyUI int8 reference image
# (testdata/comfyui_int8_ref_252469767172722_1120x1680.png).
#
# Params (seed/steps/cfg/size/sampler) match the reference; override any via env:
#   SEED=123 STEPS=8 WIDTH=512 HEIGHT=512 ./generate_unicorn_int4.sh
# BACKEND defaults to zig-cuda (true int4 tensor cores); use cpu for the CPU path.
set -euo pipefail
cd "$(dirname "$0")"

DIT=${DIT:-models/diffusion_model/krea2CenterSemiraw_v10Int4_CONVROT.safetensors}
BACKEND=${BACKEND:-zig-cuda}
SEED=${SEED:-252469767172722}
STEPS=${STEPS:-20}
CFG=${CFG:-1.0}
WIDTH=${WIDTH:-1120}
HEIGHT=${HEIGHT:-1680}
OUT=${OUT:-testdata/int4_unicorn_ref_params.png}

# Positive prompt (verbatim from the reference image's PNG "parameters" chunk).
# Single-quoted heredoc: no escaping of the inner quotes / dashes needed.
read -r -d '' PROMPT <<'PROMPT_EOF' || true
fantasy art, digital illustration, moody colors and lighting, dark, painterly anime style, The image presents a digitally rendered advertisement for a fictional unicorn product, bathed in soft, diffused light suggestive of a fantasy art style. The composition is vertically oriented, intended for a poster or large-scale display. The central focus is a single, realistically proportioned unicorn, its coat and horn a pearlescent white that shimmers subtly. The unicorn is positioned slightly off-center, gazing directly towards the viewer with large, expressive eyes of a deep violet hue; they convey a sense of serene intelligence. A flowing, iridescent mane and tail cascade around the creature, catching the light and creating visual complexity.
The unicorn is depicted in a clearing within a lush, overgrown forest; large, ancient trees with gnarled roots frame the composition, creating depth. The forest floor is blanketed in vibrant, bioluminescent flora—glowing moss and flowering vines—enhancing the magical atmosphere. The color palette is dominated by cool tones – blues, greens, and purples – with subtle highlights of gold and silver woven throughout.
In the lower left corner, a stylized, ornate logo is embedded. It reads, "AURORA UNICORNS - DREAM WEAVERS -" in an elegant, flowing script font rendered in metallic silver. Beneath the logo, in smaller print, is the tagline "Experience Pure Magic." Directly below the unicorn, small, sparkling particles drift upwards, suggesting enchantment and wonder. The image evokes a mood of gentle serenity, wonder, and aspirational fantasy, catering to a target audience interested in mythology, magical creatures, and aesthetically pleasing imagery.
PROMPT_EOF

# Build the optimized binary if it's missing (Debug is far too slow for inference).
BIN=zig-out/bin/TensorPencil
if [[ ! -x "$BIN" ]]; then
  echo "building $BIN (ReleaseFast)..."
  zig build -Doptimize=ReleaseFast
fi

set -x
exec "$BIN" generate \
  --backend "$BACKEND" \
  --dit "$DIT" \
  --prompt "$PROMPT" \
  --negative "" \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --steps "$STEPS" \
  --cfg "$CFG" \
  --seed "$SEED" \
  --out "$OUT"
