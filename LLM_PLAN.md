# LLM_PLAN.md — second executable for LLM inference

Plan for `tp-llm`, a second executable that runs autoregressive LLM inference
(chat / text generation), sharing the `TensorPencil` library module with the
diffusion CLI.

Usage (all sampling/backend parameters configurable; see `tp-llm` usage text):

```
zig build -Doptimize=ReleaseFast
./zig-out/bin/tp-llm --model <qwen3VL-4B.safetensors> \
    [--prompt "..."] [--backend cpu|zig-cuda|cuda|vulkan] [--system "..."] \
    [--max-tokens N] [--max-context N] [--temperature T] [--top-k K] \
    [--top-p P] [--repeat-penalty R] [--seed S] [--greedy] [--profile]
```

Without `--prompt`, tp-llm runs an interactive multi-turn chat REPL: each
stdin line is a user turn, the KV cache carries the whole conversation
(follow-up turns prefill only their new tokens; on Vulkan they run through
the flash-decoding path token-by-token, since the square attention kernel
assumes position 0), and per-turn tok/s + context usage are printed. `/exit`
or Ctrl-D quits; the session also ends when the context window fills.

## Why this is cheap: the text encoder is already an LLM

The Krea 2 text encoder (`src/models/qwen3.zig` + `qwen3_gpu.zig` +
`qwen3_cuda.zig`) is a full Qwen3-VL-4B causal decoder stack: GQA attention
(32/8 heads, head_dim 128), per-head QK-norm, rotate-half RoPE (theta 5e6),
RMSNorm, SwiGLU, causal masking — across all three backends. Reusable as-is:

- `tokenizer.zig` — Qwen BPE with both `encode` and `decodeAlloc` (vocab 151936)
- `safetensors.zig` — mmap loading, fp8-e4m3 + per-tensor-scale dequant in GEMM
- `ops/` — matmul (incl. int8/int4 convrot paths), attention, rope, norm, act
- `gpu/` — Vulkan, zig-cuda (hand PTX), cuda (cuBLASLt/cuDNN) contexts

### Checkpoint findings (verified 2026-07-08 against `qwen3VLInstruct4bHeretic_v10.safetensors`)

- No `lm_head.weight` key → Qwen3-4B-class models tie the LM head to
  `embed_tokens` (bf16, present). HF strips tied weights on save, so this is
  expected. M1 validates the tying assumption immediately — garbage output
  means it's wrong.
- Layer 35 and `model.language_model.norm.weight` **are present**; the current
  encoder deliberately skips loading them (Krea taps stop before layer 35).
  Generation needs both.
- The `model.visual.*` vision tower is in the file; the text-only path ignores
  it (no mRoPE branch for text-only inputs).

Net: the checkpoint already sitting in `~/genai/comfyui/models/text_encoders/`
works for generation with zero new downloads.

## What's missing (the encoder → generator delta)

1. **LM head** — final RMSNorm + projection to vocab logits via the tied
   embedding matrix. During decode only the last position's logits are needed:
   one `[1, hidden] × [vocab, hidden]ᵀ` GEMV per token.
2. **KV cache + decode loop** — the encoder runs one full-sequence pass.
   Generation is two phases:
   - *prefill*: existing full-seq path, additionally writing per-layer K/V
     (post-RoPE) into the cache;
   - *decode*: seq-len-1 forward attending against the cache.
   `ops/attention.zig` needs a query-len ≠ kv-len variant; everything else is
   unchanged.
3. **Token sampling** — greedy, temperature, top-k, top-p, repetition penalty.
   (`sampler.zig` is diffusion schedulers; this is separate and small.)
4. **Chat template + streaming** — Qwen3 ChatML (`<|im_start|>role\n…<|im_end|>`),
   stop tokens (`<|im_end|>`, `<|endoftext|>`), and incremental detokenization
   (byte-level BPE: buffer partial UTF-8 sequences before flushing to stdout).
5. **Config-driven dimensions** — `qwen3.zig` hardcodes 4B dims as comptime
   constants. Move hyperparams into a runtime config struct (read from tensor
   shapes and/or a `config.json`) so one implementation serves Qwen3
   0.6B–32B. Note: 8B+ models do NOT tie embeddings (separate `lm_head.weight`)
   and 0.6B/1.7B have different tap-irrelevant dims — the config must carry a
   `tied_lm_head` flag.

## Architecture

```
src/
  llm/
    config.zig     — hyperparams (layers, hidden, heads, kv_heads, ffn, vocab,
                     rope_theta, rms_eps, tied_lm_head); from shapes/config.json
    kv_cache.zig   — per-layer K/V buffers, fixed max context, position cursor
                     (CPU slices now; GPU-resident variant in M4)
    engine.zig     — prefill/decode orchestration, backend dispatch
    sample.zig     — logits → token id
    chat.zig       — ChatML templating, stop-token detection
  llm.zig          — re-exports, wired into root.zig
  llm_main.zig     — the second executable (thin CLI, like main.zig)
```

`build.zig`: second `b.addExecutable(.{ .name = "tp-llm", … })` importing the
same `TensorPencil` module, `b.installArtifact`, and a `run-llm` step. Tests
for `src/llm/*` ride the existing library test binary via `root.zig`.

### Shared transformer, two consumers (decided)

Generalize `qwen3.zig` rather than forking it. The causal transformer becomes
the shared core; the diffusion text encoder is "prefill, return tapped hidden
states" and the LLM is "prefill + lm_head + decode loop". Forking would mean
maintaining two forward passes × three backends — duplication compounds at 3×
here. Refactor risk to the working diffusion path is covered by the existing
image-generation integration tests (run `generate_unicorn_fp8.sh` class
smoke + `zig build test` after the refactor).

Concretely: `TextEncoder.load`/`encode` keep their signatures (pipeline.zig
untouched); internals move to a parameterized `CausalLM` that takes the config
struct and an optional tap list. Loading layer 35 + final norm becomes
conditional on the consumer.

## Milestones

### M1 — end-to-end correctness, deliberately dumb
CPU, greedy decoding, **no KV cache**: recompute the full sequence per token
via the existing forward pass, apply final norm + tied lm_head on the last
position, append argmax token, repeat. Quadratic and slow — but ~200 lines,
zero refactor risk, and validates tokenizer → template → forward → logits →
detokenize in one shot against the on-disk checkpoint.

*Accept:* coherent multi-sentence answer to a chat prompt; token-level match
vs HF `transformers` greedy output for a short fixed prompt (first ~20 tokens;
fp8 dequant noise may cause late divergence — eyeball threshold).

### M2 — KV cache (the real engine)
`kv_cache.zig`, prefill/decode split in `engine.zig`, incremental attention
variant in `ops/attention.zig`, config-driven dims refactor of `qwen3.zig`.
Per-token cost drops from O(n²) to O(n).

*Accept:* identical output to M1 for the same prompt/seed; unit tests for
cache append/read and qlen≠kvlen attention vs the full-seq reference.

### M3 — sampling, chat REPL, streaming (REPL landed with the post-M4 follow-up)
`sample.zig` (temperature/top-k/top-p/repetition penalty, seeded via
`torch_rng` or plain PRNG), interactive multi-turn REPL in `llm_main.zig`,
streamed stdout via `std.Io.Writer` with UTF-8-safe incremental detok,
`--prompt` one-shot mode for scripting/tests.

*Accept:* multi-turn chat holds context; stop tokens terminate cleanly;
sampling unit tests (top-k/top-p filtering on synthetic logits).

### Status (2026-07-08): M1–M4 DONE — all four backends running

Measured on the RTX 3090, fp8 checkpoint, 256-token generations (greedy
output byte-identical across all four backends; `[n tokens in Xs]` excludes
setup, includes prefill + first-use weight upload):

| backend  | tok/s | notes |
|----------|-------|-------|
| cpu      | 2.9   | host-bandwidth-bound (~4.8 GB of weights/token) |
| vulkan   | 26.5  | k-split GEMV + flash-decode eltwise kernels; dispatch-overhead-bound |
| zig-cuda | 67    | fused fp8 GEMV, warp flash-decoding, on-device bf16 LM head (hand PTX) |
| cuda     | 69    | same decode kernels; cuBLASLt prefill |

Key decode kernels added: CUDA `gemv_fp8` / `gemv_bf16` (block-per-row fused
dequant GEMV), `attn_split`/`attn_merge` (warp flash-decoding),
`qk_rmsnorm_par` (block-per-row norm — the serial one was 50% of decode);
Vulkan `gemv_partial`/`gemv_combine` (k-split GEMV, 4-col groups),
`attn_dsplit`/`attn_dmerge`, `rms_apply_w`, plus `pos0`-offset RoPE and a
seq_q≠seq_kv causal attention generalization on both backends.

Remaining perf headroom (not blocking): Vulkan is dispatch-bound (~900
dispatches/token; per-layer fusion or subgroup reductions would help);
CUDA gemv_fp8 sits at ~46% of DRAM bandwidth (vectorized 16B loads, CUDA
graphs for launch overhead). CPU decode could go f16/int8 to halve traffic.

### M4 — GPU decode + quantized weights (original plan)
Port the decode path to zig-cuda first (per existing perf history), Vulkan
after. Prefill reuses existing GEMM kernels unchanged. Decode is batch-1 GEMV,
**memory-bandwidth-bound** — a different regime from diffusion's big
compute-bound GEMMs. Fast decode wants fused dequant-and-dot GEMV kernels;
the int8/int4 convrot formats pay off double here (weight bytes ≈ tokens/sec).
KV cache becomes GPU-resident (append kernel; no per-token host round-trips).

*Ceiling math, RTX 3090 (~936 GB/s), 4B params:* bf16 ≈ 8 GB/tok → ~110 tok/s
ceiling; int4 ≈ 2 GB/tok → ~400 tok/s ceiling. Real-world 50–70% of ceiling.

*Accept:* GPU output matches CPU for greedy; tok/s reported by the CLI;
≥50 tok/s int4 on the 3090 as the initial bar.

### Later / non-goals for now
- Other architectures (Llama, Gemma) — config struct is the extension point;
  first new arch import will force weight-name mapping tables.
- GGUF import — safetensors-only keeps loading consistent; revisit if a wanted
  model only ships as GGUF.
- Batched serving, speculative decoding, paged KV — research toys after M4.
- `--backend cuda` (cuBLASLt/cuDNN) LLM path — optional; the pure backends are
  the point. Revisit only to measure a gap, as with diffusion (M10 Phase 2).

## Decisions (settled 2026-07-08)

1. Exe name: **`tp-llm`**.
2. Default max context: **4096** (KV cache preallocated; `--max-context` flag
   to override).
3. KV cache dtype: **f32**. The CPU forward pass already produces K/V as
   `[]f32` and attention consumes f32, so the cache is just retained buffers —
   no conversions. Cost at 4096 ctx: 2 × 36 layers × 4096 × 1024 × 4 B
   ≈ 1.2 GiB. f16 (halves size, doubles effective cache bandwidth) is the
   M4-era perf lever, not a starting requirement.
