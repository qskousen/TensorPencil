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

### M5 — speculative decoding (landed 2026-07-09: all four backends)

Lossless speculative decoding behind `--spec-k <n>` (0 = off): a drafter
proposes up to k tokens, one batched forward verifies them all, the KV cache
rolls back past the first rejection. Acceptance draws each drafted token with
its probability under the *fully processed* sampling distribution (penalty →
temperature → top-k → top-p, `sample.Dist`), resampling rejections from the
renormalized residual — the output distribution is exactly vanilla's, and
greedy is byte-identical (verified: verify kernels reproduce the decode
kernels' reduction order bitwise).

Pieces:
- `llm/spec.zig` — verify loop (`spec.generate`), `NgramDrafter`
  (prompt-lookup: longest 4→2-gram suffix match, most recent occurrence),
  draft/accept stats. Engine dispatches on `Options.spec_k`; backends without
  `stepAll`/`truncate` return `error.SpecUnsupported` (comptime-gated).
- Stepper interface grew `stepAll` (logits for every new row) and
  `truncate(len)` (cache rollback — a host-side counter on all backends).
- CUDA small-batch regime (`linearFp8`, seq ≤ 17): new `gemv_fp8n` /
  `gemv_bf16n` kernels — 8 weight rows per block × 4 inputs per weight read.
  Block-per-row multi-input GEMV saturates L2 re-reading the inputs (80 GB/s);
  the 8-row grouping restores 221 GB/s. `attn_split/merge` generalized to
  warp-per-(query, head, split) with per-query causal KV caps, replacing the
  naive square kernel (0.87 ms → 43 µs/layer at verify sizes). Short
  multi-turn prefills ride the same path (the old seq>1 route dequantized
  every fp8 weight to f16 scratch — ~5x weight traffic).

- Vulkan small-batch regime: `gemv_partial4`/`gemv_combine4` (4-input
  k-split GEMV, weight word read once per 4 inputs, per-column k order
  preserved → bitwise equal to the decode GEMV) and `attn_dsplit`
  generalized to thread-per-(query, head, chunk) with per-query causal KV
  caps (`u5 = seq_q`; `attn_dmerge` unchanged — it just runs with
  heads' = seq_q*heads). Follow-up (pos0 > 0) prefills now chunk through
  this path instead of going token-by-token — the old square-attention
  pos0=0 limitation is gone.

Measured (3090, greedy, 200-token runs, byte-identical output):
- zig-cuda: listing prompt 67 → 73 tok/s at k=3 (30% acceptance);
  exact-repetition ceiling 54 → 76 tok/s at k=8–16 (85% acceptance, 6–9
  tokens/verify). Novel-prose worst case ≈ break-even.
- vulkan: ≈ break-even on the listing prompt (verify ≈ decode cost because
  decode is dispatch-overhead-bound, and acceptance only buys 1.25
  tokens/verify there); the win is the multi-turn prefill fix.

The drafter, not the verify path, is now the binding constraint — a trained
draft head (EAGLE-style) or small draft model is the next lever.

**Draft-model drafting (landed 2026-07-09, same day):** `--draft-model
<path>` runs a second, smaller CausalLM as the drafter (`spec.ModelDrafter`:
greedy rollout on its own stepper + KV cache, re-synced to the accepted
prefix by truncate + prefill after rejections; drafter failure degrades to
vanilla decode). Enablers: `qwen3.Config` (runtime dims + tensor-name
prefix, auto-detected from the checkpoint — Qwen3-0.6B loads alongside the
VL 4B), cfg-aware CpuModel/CudaLM, and a bf16 linear path in CudaLM
(gemv_bf16/bf16n; the 0.6B ships bf16, no fp8 scales). Vulkan stepper is
still 4B-only (guards with error.UnsupportedModelConfig).

Measured (3090, zig-cuda, greedy, Qwen3-0.6B-*base* drafting for the 4B —
instruct scored no better): acceptance 45% (novel prose) / 79–86% (listing)
/ 94% (exact repetition); k=3 is the sweet spot. Net: listing 69 → 79 tok/s
(1.15x, beats n-gram's 73); prose 61 → 54 (0.9x); repetition 54 → 57 (worse
than free n-gram drafting there). The economics: a 0.6B draft step costs
4.9 ms (202 tok/s standalone) vs the target's 14.5 ms — both are
DISPATCH-BOUND (~700 launches/step), so the small model only gets 3x
cheaper, not the ~7x its size suggests. Break-even needs ~2.4 accepted
tokens/round at k=3. **Next lever is draft-step latency, not acceptance:**
CUDA graphs (capture the decode step, replay per token) would take the
draft to ~1.5 ms and make prose ~1.4x. Choose the drafter per workload
until then: `--spec-k` alone (n-gram, free) for repetitive/grounded text,
`--draft-model` for everything else.

### M6 — CUDA graphs for decode (landed 2026-07-09)

Single-token decode (seq == 1, zig-cuda + cuda) is captured once as a CUDA
graph and replayed per token: one 8-byte state upload + one cuGraphLaunch
instead of ~700–950 kernel launches. Full device-side indirection makes the
graph static: `g_state` (a module global shared by graph-mode kernel
variants in `elt.decode_state_ptx`) holds {token, pos0}; `embed_gather_s`
replaces the CPU embedding gather + upload, `rope_half_s` / `kv_append_s` /
`attn_split_s` read pos0 from state (per-thread math and order identical to
their param twins — logits stay bitwise equal, greedy output byte-identical).
The first decode step runs normally to warm weight residency + JIT (uploads
during capture would fail); capture failure falls back permanently.
--profile bypasses graphs (event timers cannot be captured).

Measured: 4B 69 → 70.7 tok/s (it was already GPU-memory-bound — the CPU
queues 950 async launches faster than the GPU drains them); 0.6B 202 → 219
(kernel EXECUTION latency, ~10 µs x 364 serialized kernels, is the floor,
not launch API overhead — per-layer kernel fusion is the next lever there).

### M7 — EAGLE-3 head drafter (landed 2026-07-09)

`--eagle <head.safetensors>` drafts with a trained one-layer EAGLE-3 head
(AngelSlim/Qwen3-4B_eagle3, 218M bf16) reading the TARGET's own hidden
states — the third drafter option alongside `--spec-k` (n-gram) and
`--draft-model`. Pieces:
- `CudaLM.enableTaps`: the residual stream entering 3 layers, device-
  resident for every committed position ([3][capacity][hidden], 126 MB);
  written by `copy_off` in batched forwards and by state-driven appends
  inside the decode graph.
- `models/eagle3.zig`: head loader (fc 3*hidden->hidden fusion, one
  llama-style layer over concat(normed embed, normed hidden), 32k draft
  vocab with delta `d2t`), GPU forward reusing the bf16 GEMV + batched
  flash-decode kernels, and `Eagle3Drafter` — grounds head rows from target
  features after each verify (re-syncing across rejections like
  ModelDrafter), then rolls out feeding the head its own hidden states.
- Tap convention measured empirically: OUTPUTS of layers (2, N/2, N-3) —
  i.e. taps at [3,19,34] — beat the input-side variants (51% vs 43–46%
  acceptance on the listing prompt).

Measured (3090, zig-cuda, greedy, byte-identical): listing 69 → **92 tok/s
(1.34x, k=3, 51% acceptance)** — the best speculative result in the repo;
prose 22% acceptance ≈ break-even; repetition 43% (n-gram still wins there,
free). The head was trained on vanilla Qwen3-4B-Instruct and still gets
37–57% acceptance against the Heretic-abliterated VL fp8 target.

**Matched-target control (vanilla Qwen3-4B bf16 as the target,
`qwen3_4b_instruct.safetensors`, `/no_think`):** listing acceptance 51% →
62% (54.6 → 78.5 tok/s, 1.44x), prose 22% → 24%. So the Heretic/VL/fp8
feature drift costs ~10 points of acceptance on structured text and almost
nothing on prose — the low prose ceiling is intrinsic to this head + greedy
chain drafting, not the checkpoint mismatch (consistent with AngelSlim's
reported 1.8–3.5 accept lengths). A head trained on the actual target would
recover the ~10 points; bigger prose gains need tree drafting or a
better-trained head. Note: without `/no_think` the hybrid Qwen3-4B thinks,
and thinking traces accept like prose (~29%).

Drafter menu (tok/s, zig-cuda greedy; vanilla 69/61/54):

| workload   | n-gram | 0.6B draft | EAGLE-3 |
|------------|--------|------------|---------|
| listing    | 73     | 80         | **92**  |
| prose      | 60     | 55         | 56      |
| repetition | **76** | 57         | 49      |

### M8 — tree drafting (landed 2026-07-09)

Shipped behind `--tree <nodes>` (requires `--eagle` + `--greedy`): the
drafter proposes a branching token tree (Eagle3Drafter.proposeTree:
level-synchronous beam, top-2 per node, beam 2, cumulative-logprob
ranking), one `stepAllTree` forward verifies every node (depth-based rope
via `rope_half_pos`, prefix + ancestor-chain attention via
`attn_split_tree` with decode-identical chunking, batch K/V retained at
cache rows [capacity, capacity+64)), a greedy walk keeps the deepest
matching root path, and `commitTreePath` copies the accepted rows (and
EAGLE tap rows) into the linear cache. Chain and tree coexist; dispatch is
`opts.tree_nodes > 0` + `@hasDecl` gates, v1 greedy-only as planned. The
CPU stepper implements the same interface (`ops.attentionTree`,
`CausalLM.forwardTree`) as the reference; toy acceptance-walk tests plus
gated CPU/CUDA/EAGLE real-model tests all verify byte-identical output vs
vanilla greedy (also spot-checked at 200 tokens on both targets).

**Measured (3090, zig-cuda, greedy, listing prompt, 200 tok):** trees WIN
their verify-cost class but LOSE the overall race on this rig. Matched
target (vanilla 4B bf16, /no_think): vanilla 54.7, chain k=3 **67–68.5**
(2.17 tok/verify), tree-7 44.9 (2.30 tok/verify — beats chain k=4's 2.17
at the same 2-pass verify cost), tree-15 25. VL fp8 target: chain k=3
70.2, tree-16 26, tree-31 13. The economics: grouped-GEMV verify costs one
FULL weight pass per 4 rows at ~220 GB/s effective (bf16 8 GB, fp8
3.6 GB), and beyond 17 rows the fp8 GEMM path's ~5x dequant traffic is
worse still — while this head's acceptance concentrates in its top-1
(39% chain acceptance; extra branches add only ~+0.1–0.3 tok/verify per
verify-cost doubling). Chain k=3 fills exactly one 4-row pass and stays
the global optimum.

Levers that would flip the sign (in order): (1) a faster multi-input GEMV
— 8/16 inputs per weight read instead of 4 would halve/quarter the
marginal node cost and is now the single binding constraint (gemv_fp8n/
bf16n sit at ~220 GB/s); (2) a head trained on the actual target with
calibrated top-2/3 mass (SpecForge), which is what the paper's 3–6.5x
tree numbers assume; (3) hardware where a 16–64-row verify is
compute-bound and ~free. The machinery is correct, lossless, and waiting.

Design notes below kept as-written (all implemented, one deviation:
`proposeTree` takes an explicit `max_depth` cap, and per-query kv row
indirection replaced the "extra partial + merge nsplit+1" scheme so tree
logits stay bitwise-identical to decode):

Chain drafting caps the win: a k-deep chain dies at its first wrong token
(~2.9 tokens/verify at 62% acceptance), which is why we sit at 1.3–1.4x
while the EAGLE-3 paper reports 3–6.5x — their numbers are TREE drafting
(~a few dozen draft nodes as a branching tree, one verify forward with a
tree-attention mask, keep the best accepted path; also bigger targets and
slower baselines, so don't expect the full multiplier here). Trees stay
OPTIONAL: chain (`propose`) and tree (`proposeTree`) coexist, chosen by
what the drafter implements and a CLI flag.

Design (worked out 2026-07-09, from the M5–M7 machinery):

- **Drafter interface**: optional `proposeTree(ids, tokens: []u32,
  parents: []u32) usize` — node i's token + parent index (parent < i,
  root's parent = itself/sentinel; root continues `pending`). Budget ~16–64
  nodes, depth ~6–8. spec.generate dispatches on `@hasDecl` like the
  engine's stepAll gate. Eagle3Drafter expands top-N children per node
  ranked by cumulative draft probability (needs head logits top-N, not just
  argmax — trivial, logits are already on host). ModelDrafter can emit
  trees too (top-N per step) — same interface, so the 0.6B path benefits.
- **Verify forward** (`stepAllTree`): all nodes in one batch. Three chain
  assumptions break:
  1. *Rope positions* are depth-based (prefix_len + depth(node)), not
     row-sequential — needs a rope variant taking a per-row position
     buffer (upload a small u32 array per verify).
  2. *Attention* = full committed prefix (dense, same for every node) +
     the node's ANCESTOR CHAIN inside the batch (sparse, <= depth rows).
     Reuse the flash-decode split/merge: attn_split's scratch rows are
     unnormalized partials (m, d, pad, pad, acc[hd]) at stride hd+4,
     laid out [node][head][split] — add ONE extra partial per (node, head)
     computed by a new small kernel over the node's ancestor list (batch
     K/V + ancestor index buffer), then run attn_merge with nsplit+1.
     attn_merge needs no other change.
  3. *KV cache*: in-batch nodes CANNOT append to the linear cache (sibling
     branches collide at the same position). Retain per-layer batch K/V in
     dedicated buffers ([n_layers][max_nodes][kv_dim] f32 — ~19 MB for 64
     nodes) and, after acceptance, copy the ACCEPTED PATH's rows per layer
     into the cache (copy_off per layer per accepted node; 36*depth
     dispatches, fine) instead of truncate-past-rejection.
- **Acceptance** (greedy first): walk from the root; at each node compare
  the target argmax of that node's logits row against its children's
  tokens; descend on match, else stop and emit the target token
  (correction). Emitted = accepted path + 1. Byte-identical to vanilla
  greedy by the same argument as chains. Temperature > 0 needs recursive
  multi-child rejection sampling (SpecInfer/EAGLE-style residual updates
  per rejected child) — v1 ships greedy-only trees and falls back to chain
  drafting when temperature > 0.
- **EAGLE head tree expansion**: the head's own K/V for tree nodes follows
  the same ancestor-mask problem at 1 layer — small enough to just
  re-forward each path prefix, or reuse the same extra-partial trick.
  Grounding/re-sync machinery (Eagle3Drafter.grounded) is unchanged: only
  the accepted path ever becomes grounded rows.
- **CPU backend**: ops.attention needs an ancestor-mask variant (or
  per-node scalar loops — seq is tiny); worth doing for the toy-model
  tests: a ToyModel with tree verify validates acceptance-walk + cache
  path-append against rule rollouts exactly like the chain tests in
  llm/spec.zig.

Key code entry points: `llm/spec.zig` (generate loop, drafters, toy
tests), `models/eagle3.zig` (head forward + drafter), `models/qwen3_cuda.zig`
(stepAll/layersForward/linear dispatch, taps, decode graph),
`gpu/cuda/elt.zig` (attn_split/attn_merge partial format, kv_append_s,
copy_off), `gpu/cuda/backend.zig` (opAttnDecode, graph capture). Baselines
to beat (3090, zig-cuda, greedy): chain EAGLE 92 tok/s listing / 56 prose
on the VL target (69/61 vanilla); 78.5/— on vanilla Qwen3-4B bf16 (54.6
vanilla, `/no_think`). Estimate with the AngelSlim head: ~2–2.5x
structured, ~1.5x prose; a SpecForge-trained head on the actual target
would add ~10 pts acceptance on top.

### Later / non-goals for now
- Other architectures (Llama) — config struct is the extension point;
  first new arch import will force weight-name mapping tables.
- Gemma 3 (GGUF arch "gemma3") — **cpu + zig-cuda/cuda + vulkan DONE (2026-07-14), incl. vision**:
  `models/gemma3.zig` (Config.detect + `Model` + `CpuModel` stepper). New vs
  the Qwen stack: "sandwich" norms (input / post-attention / pre-FFN /
  post-FFN RMSNorm, the two post-norms applied to the sublayer output before
  its residual add; +1 already folded into the GGUF weights), embeddings
  scaled by sqrt(hidden), GeGLU (gelu-tanh) FFN, per-head QK-norm then full
  rotate-half RoPE whose base/scale ALTERNATE by layer — every 6th layer
  (`sliding_window_pattern`) is GLOBAL (theta 1e6, linear scale 1/8, full
  causal attention), the rest LOCAL (theta 1e4, no scale, sliding-window
  causal mask window 1024). head_dim 256 with the 12B's 1/sqrt(256) attention
  scale is exactly the engine default, so ops.attention only grew a `window`
  param; ops also gained `rope.rotateHalfFreqsScaled` (freq_scale) and
  `act.geluTanhMul`. Tied LM head, no softcapping. **New tokenizer path**:
  SentencePiece (`tokenizer.ggml.model == "llama"`) —
  `Tokenizer.initSpmFromGguf` + `encodeSpm`: ▁-escape, score-ranked bigram
  merge (llama.cpp llm_tokenizer_spm), byte fallback (`<0xNN>`), and
  CONTROL/USER_DEFINED tokens matched verbatim longest-first
  (tokenizer_st_partition). Chat template family (`chat.setFamily(.gemma)`):
  `<bos><start_of_turn>{user|model}\n…<end_of_turn>\n`, no system role (its
  content prefixes the first user turn), stop on `<end_of_turn>` (106).
  Validated: SPM tokenization + full chat template token-identical to
  llama-tokenize; greedy CPU generation coherent and matching llama.cpp
  (early divergence only where GPU-f16 vs CPU-f32 flips a near-tie argmax).
  Dispatched in llm_main via `general.architecture == "gemma3"`
  (spec-decode / images / cpu-split rejected). SigLIP vision tower (mmproj)
  is the remaining follow-up.
  GPU (zig-cuda/cuda, 2026-07-14): `models/gemma3_cuda.zig` runs the stack
  device-resident — prefill in 128-row chunks (opMatmulQuant / grouped dp4a
  GEMVs), per-op decode (no graph capture: measured +0% on this
  memory-bound regime). Two RoPE tables (global theta 1e6 scale 1/8, local
  theta 1e4) live on device; each layer picks one. Two generalized backend
  primitives landed and are reused by any future arch: `be.geluMul`
  (fused GeGLU gate, `gelu_mul` PTX) and a `window` arg on `opAttnDecode`
  (sliding-window via `attn_split`/`attn_split_h256` — computes kv_start =
  max(0, kv_len - window) and partitions [kv_start, kv_len); f1 carries the
  window, 0 = full causal, so existing qwen callers are unchanged). Embed
  scale is host-side (embedding always gathers on host without a decode
  graph). Validated on the 3090: greedy byte-identical to the CPU path and
  to llama.cpp on low-entropy prompts, coherent otherwise; ~22-25 tok/s
  (12B Q4_K_M) vs ~2.3 CPU. Kernel unit test `attn decode sliding window
  matches CPU reference` exercises the kv_start>0 path (short generations
  stay under the 1024 window). Vulkan is the last backend TODO.
  VISION (2026-07-14): `models/gemma_vit.zig` — the SigLIP-So400m tower +
  Gemma projector (llama.cpp tools/mtmd clip.cpp build_gemma3 + siglip.cpp
  port, CPU f32). Pipeline: bilinear aspect-preserving resize to 896x896 +
  center pad (PAD_CEIL) + normalize ((p/255-0.5)/0.5) -> patch conv 14x14
  (im2col GEMM) -> add the learned [4096][1152] position embedding -> 27
  pre-LN SigLIP blocks (LayerNorm+bias, separate q/k/v+bias, full
  bidirectional MHA 16x72 scaled 1/sqrt(72), GELU-tanh FFN 1152->4304) ->
  post_ln -> projector: 4x4 avg-pool over the 64x64 grid -> 256 tokens,
  soft_emb_norm (RMSNorm, +1 folded into the weight), mm.input_projection
  1152->3840 (stored [in,out], transposed at load). Each image = 256 soft
  tokens, wrapped `<start_of_image>`(255999) + 256 x `<image_soft_token>`
  (262144) + `<end_of_image>`(256000), injected UNSCALED (Gemma scales only
  text embeddings by sqrt(hidden)) at sequential positions via new
  `prefill`/`prefillImage`/`forwardHidden`/`forwardRows` on the cpu + cuda
  steppers (interleaved text/image/text prefill in runGemma3, like qwen35's
  one-shot). `tp-llm --model x.gguf --mmproj mmproj.gguf --image a.png
  --prompt "..."` on cpu / zig-cuda / cuda. Validated vs llama-mtmd-cli: 274
  prompt tokens identical, caption accurate, first clause identical (then a
  near-tie divergence — f32 CPU ViT vs f16 GPU ViT + quant path).
  GPU VIT (2026-07-14): `models/gemma_vit_cuda.zig` runs the 27 SigLIP
  blocks device-side (mirrors vit35_cuda: opConvF16 patch embed, learned
  pos-embed add, head-pad 72->128 + opAttnTC full attention, LN+bias,
  gelu-tanh FFN; the cheap projector — 4x4 pool + soft_emb_norm + proj —
  stays on host via the shared `Vit.project`). 512px image encode 39s (CPU)
  -> 0.4s (GPU), identical caption; used automatically on the CUDA backends.
  DP4A DECODE (2026-07-14): gemma3_cuda.linear decode now quantizes the
  activation once (opGemvQuantizeX) and runs int8 dp4a GEMVs
  (opGemvQuantQ8 for q5_k/q6_k, opGemvQuantQ8N ng=1 for q4_k/q8_0) instead
  of the f32-dequant opGemvQuant — same regime as qwen35/llama.cpp mmvq.
  12B Q4_K_M: ~22 -> ~35 tok/s (1.6x); greedy count-prompt still
  byte-identical (the LM head keeps the f32 GEMV: vocab rows aren't %8).
  INTERACTIVE @IMAGE (2026-07-14): the tp-llm REPL @path.png mention path
  now works for gemma3 — the ViT stays loaded for the session, mentions
  encode on the GPU ViT, and inject via the shared imageTurn (ImageChat is
  now a per-arch union; the segment wrapping is family-aware).
  VULKAN (2026-07-14): `models/gemma3_gpu.zig` (VulkanLM, text-only, mirrors
  qwen35_gpu) — sandwich norms, dual global/local RoPE tables (full
  rotate-half via opRopeQwen35 half=hd/2), per-layer windowed attention,
  GeGLU, tied head, host embed×sqrt(hidden). Two generalized Vulkan
  primitives added: the `gelu_mul` eltwise kernel and a `window` arg on
  `attn_decode_q35`/opAttnDecodeQ35 (kv_start = max(0, kv_len-window); 0 =
  full causal, qwen35 caller passes 0). Validated on the 3090: greedy output
  matches the CUDA/CPU path ("...Paris", count-prompt prefix identical).
  Correctness-first — ~1.9 tok/s (12B; dispatch-bound, no batching, like the
  rest of the Vulkan LLM path); perf is the follow-up.
  VULKAN VISION (2026-07-14): `models/gemma_vit_gpu.zig` (VitGpu) runs the 27
  SigLIP blocks device-side on Vulkan; the projector stays on host (shared
  Vit.project). Vulkan has no f16 GEMM, so the block weights dequantize to
  f32 once at load and feed the f32 opMatmul (+bias). Three eltwise kernels
  added: `layernorm` (LN+bias — SigLIP; Vulkan only had RMSNorm), `gelu`
  (plain tanh-gelu), and `attn_full` (full non-causal attention, arbitrary
  head_dim — the existing `attention` kernel needs hd%32). `tp-llm --backend
  vulkan --mmproj m.gguf --image a.png` one-shot; the SigLIP encode is
  ~23.7s (per-op submit, f32 GEMMs — one-time, correctness-first) vs 0.4s
  CUDA. Interactive @image on Vulkan is not wired (one-shot only). Validated:
  Vulkan caption byte-identical to the CUDA/CPU ViT ("A red fox sits alertly
  in a snowy forest, its ..."). gemma3 now runs on ALL backends incl. vision.
- Gemma 4 (GGUF arch "gemma4") — **cpu + zig-cuda + cuda, text + vision DONE
  (2026-07-14), text token-identical vs llama.cpp; vulkan + audio follow-ups**:
  `models/gemma4.zig` (Config.detect + `Model` + `CpuModel`). Shares Gemma 3's
  sandwich norms / sqrt(hidden) embed scale / QK-norm / GeGLU / tied head /
  6-layer local:global split, but adds, per llama.cpp `src/models/gemma4.cpp`:
  (1) PER-LAYER attention geometry — LOCAL layers head_dim 256 / 8 KV heads,
  GLOBAL layers head_dim 512 / 1 KV head (MQA), so q/o/kv widths differ per
  layer and the KV cache has a per-layer stride (`kv_cache.PerLayerKvCache`);
  (2) attention score scale **1.0** (not 1/sqrt(hd) — folded into the QK norms;
  `attention.Params.scale`); (3) V is RMS-normalized per head_dim with NO
  learned weight (`norm.rmsNormUnit`), and GLOBAL layers have no v_proj so V
  reuses the RAW K projection (before k_norm/rope); (4) proportional RoPE — the
  GLOBAL layers divide the inverse frequency by `rope_freqs.weight` per dim
  (`rope.rotateHalfFreqsFactored`), replacing Gemma 3's scalar 1/8; (5) a
  per-layer scalar `out_scale` (layer_output_scale) multiplies the whole layer
  output; (6) final logits tanh-softcapped at 30 and `suppress_tokens` forced
  to -inf. The 12B QAT is **Q4_0** — added `.q4_0` to dtype/quants/gguf/matmul
  (dequant rides ggml like the other block quants). **New tokenizer path**:
  "SPM-style BPE" (`tokenizer.ggml.model == "gemma4"`) — `initGemma4FromGguf` +
  `encodeGemma4`: SPM vocab (▁-escape, no dummy prefix, `<0xNN>` byte fallback,
  verbatim CONTROL specials) but merges are RANK-ordered BPE rules over
  newline-split words (llama.cpp LLAMA_VOCAB_TYPE_BPE + pre-type "gemma4"),
  keyed `left\x00right`; a newline run that is itself a token is emitted whole.
  Chat family `chat.setFamily(.gemma4)`: `<bos><|turn>{user|model}\n…<turn|>\n`
  (turn markers 105/106, newline 107), no system role. Validated: gemma4
  tokenization token-identical to llama-tokenize; greedy CPU generation
  byte-identical to `llama-completion` ("The first 8 prime numbers are: 2, 3,
  5, 7, 11, 13, 17, and 19.\n\nThree planets are: Mars, Jupiter, and Saturn.").
  Dispatched in llm_main via `general.architecture == "gemma4"` (GPU / spec-
  decode / cpu-split rejected). **VISION on CPU** (`models/gemma4_vit.zig`): the
  mmproj `gemma4uv` "unified" embedder is NOT a SigLIP ViT (no transformer) —
  smart-resize (effective 48px patch = 16·n_merge(3), tokens ∈ [40,280], each
  dim snapped to a 48-multiple, aspect ~preserved, bilinear + PAD_CEIL onto a
  black canvas), /255 normalize (mean 0/std 1), channel-planar im2col →
  LayerNorm(patch_norm_1, over 6912, eps 1e-5) → patch-embed matmul (6912→3840)
  + bias → LayerNorm(patch_norm_2) → add learned pos (two lookup tables, x by
  column / y by row; `v.position_embd` [dim,1120,2]) → LayerNorm(patch_norm_3) →
  weightless RMSNorm (eps 1e-6) → mm.input_projection (3840→3840). `num_patches`
  tokens (NO pooling), injected UNSCALED between `<|image>` / `<image|>` markers.
  `tp-llm --backend cpu --mmproj m.gguf --image x.png --prompt "..."` one-shot.
  Validated by accurate image-specific captions (fox-in-snow; a dragon banner
  where it even OCR'd the embedded "AURORA" text; "too blurry" on a noise
  fixture) — exact token-match golden was blocked by llama-mtmd-cli crashing on
  gemma4's jinja template, but the text LLM is already token-identical so the
  embedder rides on that. Image-token blocks attend BIDIRECTIONALLY in the LLM
  (llama.cpp marks them non-causal): every image token sees the whole block,
  causal only to the prefix — prefilled in one un-chunked `bidirectional` pass
  on CPU + all GPU backends (Gemma 3 and Gemma 4). **GPU**
  (`models/gemma4_cuda.zig`, cuda + zig-cuda,
  device-resident MVP — no CPU-split/offload/streaming since the 12B fits):
  per-layer K/V Growables (kvDim(l) stride); GLOBAL-layer attention uses the
  generic naive `attn` op (arbitrary head_dim, full causal), LOCAL uses
  opAttnDecode (h256 flash-split + window); V-norm via qkNorm with a shared
  device ones buffer; out_scale via a new `opScale`/`f32_scale` kernel; the
  GLOBAL RoPE freq_factors are baked into the device table; logit softcap +
  suppress run host-side after the logits download. Two gotchas: Q4_0 had NO GPU
  support (added a `dequant_q4_0_f16` PTX kernel + a q4_0 arm in opMatmulQuant;
  q4_0 linears route through the dequant→f16 GEMM — the dp4a GEMV path has no
  q4_0 kernel), and the naive `attn` kernel's `.local accl[512]` (128 f32) OOB'd
  at head_dim 512 → bumped to accl[2048]. Text token-identical to CPU/llama.cpp
  on both backends; vision caption identical to the CPU path. **Vision on GPU**
  (`models/gemma4_vit_cuda.zig`): the shallow gemma4uv embedder runs device-side
  (opLayerNorm ×3 → patch-embed opConvF16 → +pos opAdd → weightless rms → opMatmulBf16
  projection), parity vs CPU min token cos 1.000000; wired into tp-llm and tp-gui.
  **Decode ~44
  tok/s** (token-identical, both backends; exceeds the gemma3 ~35 ref) — ~13x
  over the initial dequant-GEMM path, via two kernels: (1) `attn_split_h512`
  (flash-decode split for head_dim 512, adapted from attn_split_h256 — 16
  dims/lane; wired through opAttnDecode, replacing the naive f32 `attn` whose
  `.local accl` thrashed local memory), and (2) `gemv_q4_0_q8n` (dp4a int8-
  activation GEMV reusing the q8_0_q8n grouped machinery; unpacks the 18-byte
  q4_0 nibble block and applies the -8 offset as dot(w-8,a)=dp4a(nibble,a)-8·Σa).
  A fused f32 `gemv_q4_0` is the non-dp4a fallback; prefill uses the dequant→f16
  GEMM. REMAINING: the VULKAN backend and the AUDIO encoder (`gemma4ua`: mel
  frontend + encoder — TensorPencil has no audio path).
- ~~GGUF import~~ — DONE (2026-07): `src/gguf.zig` (container, llama.cpp→HF
  name canonicalization, config from `qwen3.*` metadata), `src/quants.zig`
  (Q8_0/Q4_K/Q5_K/Q6_K dequant, bit-exact vs ggml-quants.c golden fixtures),
  `src/weights.zig` (WeightStore over safetensors/GGUF). `tp-llm --model
  x.gguf` runs on **cpu / zig-cuda / cuda** (vulkan still gated). The
  zig-cuda path adds per-format PTX: fused GEMV (decode, `elt.gemv_q*`),
  dequant-to-f16 feeding the existing hgemm/cuBLASLt tail (prefill,
  `opMatmulQuant`), and graph-mode embed gathers in `decode_state_ptx`;
  untied `output.weight` heads supported (`CudaLM.lmHeadGemv/lmHeadAll`).
  Qwen3-4B Q4_K_M decodes at 47 tok/s greedy (word-identical to CPU), Q8_0
  at 44. Known wrinkles: no grouped-N quant GEMV yet (spec verify batches
  take the dequant-GEMM), and decode-graph capture falls back to per-op
  launches under partial-pin `--vram-budget` (mid-capture weight upload).
  Tokenizer: `Tokenizer.initFromGguf` builds the BPE from the
  `tokenizer.ggml.*` kv arrays (control tokens → verbatim specials; template
  and stop ids resolve per-vocab and reach chat.zig via
  `chat.applyTokenizer`); files without tokenizer arrays fall back to the
  embedded Qwen3 tokenizer. Unknown `tokenizer.ggml.pre` values warn and use
  the qwen2 regex — verify before trusting a new family's tokenization.
  Target model: Qwen3.6 27B Q5_K_M — SUPPORTED (2026-07): `models/qwen35.zig`
  implements the hybrid gated-DeltaNet architecture (48 linear-attention +
  16 gated-attention layers, head_dim 256, partial neox RoPE over 64 dims,
  per-channel causal conv + delta-rule recurrence ported from llama.cpp's
  qwen35.cpp / delta-net-base.cpp), with its own `State` (16-layer KV cache
  + conv/recurrent states) and CPU stepper. The qwen35 pretokenizer
  ([\p{L}\p{M}] runs) is implemented and golden-tested vs llama-tokenize.
  Validated: greedy output is token-identical to llama.cpp over 72-token
  generations. Speculative decoding rejected up front (recurrent state
  cannot roll back). Backends: cpu (~0.35 tok/s, memory-bound) and
  **zig-cuda / cuda** (`models/qwen35_cuda.zig`, ~29.7 tok/s steady decode
  on the 3090 — llama.cpp mmvq does 30.2 on the same file, parity):
  weights resident, new PTX in elt.zig — `gdn_delta_step` (block-per-v-head
  two-pass delta rule, state walks x4-unrolled for memory-level
  parallelism), `gdn_conv_step`, `gdn_gates`, `l2norm_rows`,
  `deinterleave2`, `mul_sigmoid`, `rope_half_part`, and `attn_split_h256`
  (8 dims/lane; attn_merge was already hd-generic).
  DP4A DECODE GEMV (2026-07-10, 17.4 -> 22.7 tok/s): `quantize_q8_1`
  quantizes each decode activation once to int8 (SoA: f32 d[cols/32] then
  i8 qs[cols]; opGemvQuantizeX, one launch per distinct x), and
  `gemv_q5_k_q8` / `gemv_q6_k_q8` (opGemvQuantQ8) do dp4a integer dot
  products with llama.cpp's vec_dot math (q5_K vmmq / q6_K mmvq): 16-elem
  units per lane so the inline 6-bit scale decode amortizes, sum-of-u dp4a
  for the dmin*m / -32 terms, integer sc/m muls, one v4.u32 header load.
  Kernel-level test vs a CPU emulation of the same activation quantization
  (`dp4a gemv quant kernels match CPU reference`). q8_0/q4_k keep the f32
  GEMVs (4B path unchanged, and its decode measures 72 tok/s now).
  DECODE-GRAPH CAPTURE (2026-07-10): the M6 pattern ported — after the
  first (warm) decode step the whole forward replays as one cuGraphLaunch;
  {token, len} land in g_state, the M-RoPE triple in pos3_d, KV appends
  via kv_append_s and attention length via the new `attn_split_h256_s`
  (decode_state_ptx). Gotcha: the embed table had to be gathered ONCE
  outside capture — warm steps embed on host, so the first cachedWeight
  upload (cuMemAlloc) of the 875 MB table otherwise lands inside the
  capture and CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED's it. Capture itself
  was worth ~0 (the async queue already hid launches — measured, not
  assumed) but shields the now-faster kernels from launch overhead.
  Steady decode is 33.6 ms/token: q5 GEMVs ~21 ms (~660 GB/s eff), q6
  GEMVs ~6.4 ms (563 GB/s, 2-byte-aligned 210 B blocks resist vector
  loads), gdn_delta_step ~2 ms, norms ~1.5 ms. The printed tok/s
  (~23.7) is dragged by ~2.2 s of fixed start cost — mostly the
  small-batch prefill full-dequantizing every weight (see below).
  BATCHED PREFILL
  (2026-07-10): `stepBatch` runs 128-row chunks — projections/MLP via
  opMatmulQuant (dequant-to-f16 tensor-core GEMM, output padded to 128
  rows, so activation buffers are sized pc=128), attention via
  opAttnDecode seq_q=n (`rope_imrope_pos` applies per-row M-RoPE triples),
  DeltaNet conv/delta recurrence stays sequential per token inside the
  chunk. Text prefill ~80 tok/s (409-token fox prompt ~5s vs ~35s seq),
  image setup 52s -> 6.5s. Landing it surfaced a **Zig @min type-narrowing
  footgun** (see ZIG.md "@min/@max narrow their result type"):
  `@min(prefill_chunk, ...)` yields u7, so `pos3s[0 .. n * 3]` overflowed
  and chunks with n >= 43 passed a truncated slice — stepBatch silently
  processed n' = ((n*3)&127)/3 rows. Fixed with `const n: usize =
  @min(...)`. GROUPED SMALL-BATCH PREFILL (2026-07-10 evening):
  `gemv_q5_k_q8n`/`gemv_q6_k_q8n` (opGemvQuantQ8N) stream each weight row
  once against up to 8 quantized activation rows (comptime-generated
  per-input PTX blocks; one quantize_q8_1 stages the whole chunk). Routed
  in gemm() for chunks <= 40 rows — MEASURED crossover: one grouped pass
  is ~165 us/weight vs dequant+hgemm's flat ~0.92 ms/weight/chunk, so
  grouped wins 3-6x on chat-turn-sized prefills and loses ~2.5x at 128
  rows (first attempt routed everything and made long prompts 4x slower —
  always measure the crossover). Remaining headroom: q6_k GEMV load
  alignment (repack at upload time), the grouped kernel's per-pass cost
  (165 us is ~4.7x a single-x pass for 8x the rows — the sequential
  8-input blocks expose latency).
  VISION (2026-07): `models/vit35.zig` implements the qwen3vl_merger mmproj
  (llama.cpp tools/mtmd port: smart-resize + fit/pad bilinear preprocessing,
  summed dual patch convs, antialias-interpolated 48x48 position grid,
  27 pre-LN blocks with 2-D vision RoPE {18,18} pairs, full attention,
  GELU-tanh FFN, post_ln, 2x2 merge -> mm.0/mm.2 projector; no deepstack),
  running on CPU; `tp-llm --image x.png --mmproj m.gguf` (zig-cuda/cuda)
  injects the embeddings during prefill with interleaved M-RoPE grid
  positions (`rope_imrope` kernel, sections from GGUF; text collapses to
  the 1-D path bit-identically). Validated word-identical to llama-mtmd-cli
  greedy on a 768x768 image, and semantically on non-aligned sizes. PNG
  decode (8-bit RGB/RGBA) landed in image.zig; layerNorm in ops/norm.zig.
  GPU VIT (2026-07-10): `models/vit35_cuda.zig` runs the whole tower
  device-side on the CUDA backends — 2304 patches 14.5s (threaded CPU) ->
  0.3s, 1024 patches 3.9s -> 0.2s on the 3090. Host keeps only the cheap
  prep (`Vit.prepare`, shared with the CPU path). The bf16 mmproj weights
  feed the f16 tensor-core GEMM straight from the GGUF mmap via
  `opMatmulBf16` (an `opConvF16` twin with a `bf16_to_f16_pad2d` weight
  convert; pad-handles the non-aligned ffn 4304). The 72-dim heads are
  zero-padded to 128 on device (`head_pad` restride kernel) so the DiT's
  `opAttnTC` applies unchanged (PV GEMM needs n=head_dim 128-mult; the
  pads are exact zeros end to end, scale stays 1/sqrt(72)). New eltwise
  PTX: `ln_bias_par` (two-pass LayerNorm w/ bias, block-per-row),
  `rope_vision` (pairs (p, p+36), row/col sections, per-token (py,px)),
  `head_pad`. Encode drops the backend weight cache + attention/GEMM
  scratch afterward (evictWeights/freeAttnScratch/freeConvScratch) — both
  for VRAM (the 27B needs the card) and correctness (cache keys point
  into the Vit arena the caller frees). Validated: kernel unit tests vs
  CPU refs, GPU-vs-CPU encode cos > 0.999 / rel RMSE < 0.05 on the real
  mmproj, and word-identical 27B greedy output at 512px (768px diverges
  in the tail wording only — the f16 regime under greedy amplification).
  Landing it exposed a latent decode-graph bug: `captureDecodeGraph`'s
  embed-table warm upload (~874 MB) can evict LRU weights under VRAM
  pressure, and the capture then died re-uploading them mid-capture
  (cuMemAlloc is illegal while capturing). Fixed by bailing to per-op
  decode when `evictions != 0` after the warm upload — capture now either
  succeeds or degrades cleanly. `debug_cpu_vit` in llm_main.zig A/Bs the
  CPU tower on GPU backends.
  IMAGE CHAT (2026-07-10, follow-up): interactive turns attach images as
  inline `@path.png` / `@"path with spaces.png"` mentions (multiple per
  turn, anywhere in the message — each becomes a
  <|vision_start|>[pads]<|vision_end|> block at its mention point, the
  interleaving Qwen3-VL is trained on; non-.png @tokens like emails stay
  text). Pieces: `chat.parseImageMentions` + `chat.appendUserSegments`
  (segmented user turn recording per-image pad-row offsets),
  `llm_main.imageTurn` (encode mentions on the session ViT, budget-check
  with ids rollback, then interleave prefill/prefillImage so the engine's
  cached()-based prefill takes the tail — the one-shot --image pattern per
  turn). The Vit + mmproj stay loaded for the session; VRAM safety comes
  from a new backend weight SCOPE (weightScopeBegin/End: entries cached
  inside the scope are tagged and freed as a group, evictions counter
  untouched) replacing the ViT's old evictWeights — a mid-session encode
  must not drop resident LLM weights whose device pointers the captured
  decode graph baked in. If the encode itself evicts LLM weights under
  pressure, the existing evictions guard falls decode back to per-op
  (correct, ~as fast — capture was measured +0%). Validated live: 4-turn
  session with text / single-image / two-image / recall turns (27B,
  3090), and a live-captured-graph session with a mid-session encode.
  Follow-up if wanted: graph recapture after pressure events (today one
  eviction disables capture for the session), JPEG decode, per-path
  Encoded cache.
  REPL LINE EDITOR (2026-07-10, follow-up): interactive input moved off
  the cooked-mode line reader (which fired one turn per pasted line and
  had no way to type a newline) to `llm/repl.zig` — a raw-termios,
  byte-fed editor state machine (unit-tested without a tty): Enter sends,
  Shift-Enter (kitty keyboard protocol `CSI 13;2u`, enabled via `CSI >1u`
  push around each read; harmlessly ignored elsewhere) or Alt-Enter
  (ESC CR — the fallback for terminals without the protocol; plain
  Shift-Enter is indistinguishable from Enter in legacy input) inserts a
  newline, bracketed paste (`CSI ?2004h`, 200~/201~) keeps pasted
  newlines literal so a multi-line paste is ONE message, backspace edits
  within the line, Ctrl-C cancels the message (ISIG is off only while
  reading; generation runs cooked so Ctrl-C still kills it), Ctrl-D ends
  the session. Piped stdin keeps one-message-per-line (scripted sessions
  unchanged); @image mention parsing treats '\n' as a path delimiter.
  Validated through a `script` pty: Shift-Enter + 3-line paste = two
  turns, correct replies, mode sequences balanced around each read.
  IMAGE FORMATS VIA LIBVIPS (2026-07-10, follow-up): --image and
  @mentions decode through system libvips (jpeg incl. progressive, png,
  webp, gif, tiff; EXIF autorotation; alpha flattened over white) — the
  C shim `lib/vips/vips_helper.c` + `src/vips.zig` wrapper are ported
  from DiffKeep (its varargs C API can't be @cImport'ed directly).
  DELIBERATELY linked into the tp-llm EXECUTABLE only: the TensorPencil
  library module stays pure Zig (image.zig's PNG encode/decode
  untouched); building tp-llm now needs libvips-dev + pkg-config — see
  CLAUDE.md Dependencies. Validated e2e: baseline jpg (fur colors right
  => channel order right), progressive jpg + webp judged identical to
  the source png, EXIF-rotated jpg seen upright, png-via-vips unchanged.
  CHASING THE VIT PARITY TEST'S FLAKINESS FOUND A LATENT BACKEND BUG: CUDA
  `ensureDeviceBuffer` freed the old buffer without syncing — queued
  kernels could still read it (cuMemFree doesn't wait), an intermittent
  use-after-free whenever grow-on-demand scratch resized mid-batch (the
  ViT's per-GEMM conv-scratch growth hit it ~weekly-lottery style;
  Vulkan's ensureDeviceBuffer has had the flush invariant all along —
  see gpu lab notes). Now syncs before reallocating; vit parity is
  bit-stable across runs (min token cos 0.999996, rel 0.0039).
- Batched serving, paged KV — research toys.
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
