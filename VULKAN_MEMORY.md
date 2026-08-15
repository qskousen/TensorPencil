# Vulkan: undo the "no shared memory" contortions (subgroup-op rework)

> **2026-07-23 LANDMARK BASELINE — read first.** llama.cpp's own Vulkan backend
> on the SAME model+GPU (Impish 12B IQ4_NL, RTX 3090) = **82.77 tok/s** tg128
> (`llama-bench`), *higher* than its CUDA (79.75). Our best Vulkan decode is
> **~24 tok/s** (repack-dp4a); u16 sg-dp4a ~11.7. We are **~3.5× slower than
> llama.cpp on Vulkan** — and llama.cpp reports `int dot: 0` on this 3090 (FLOAT
> DMMV path, NO dp4a) yet still hits 83. So the whole dp4a-GEMV effort below
> (10→24 tok/s) was polishing a path fundamentally ~3.5× behind the reference.
> **The real lever is overall decode efficiency** (kernel quality + per-op
> dispatch/sync overhead: 24 tok/s = 42 ms/tok vs 83 = 12 ms/tok), NOT dp4a.
> Next: profile where our 42 ms/token goes vs llama.cpp. Repro: build llama.cpp
> with `-DGGML_VULKAN=ON -DGGML_CUDA=OFF` (needs `glslc`+`spirv-headers`),
> `llama-bench -dev Vulkan0 -p 0 -n 128 -ngl 99` (build at ~/genai/llama.cpp/build-vk).

Working list for reworking Vulkan compute kernels that were built around the
assumption that shared/workgroup memory is unusable. That assumption is a
**Zig-0.16-SPIR-V-codegen** limitation, not a hardware/driver one — see below —
and there's a **verified escape hatch: subgroup ops** (no workgroup storage
class needed).

## Background (why these exist)

- Zig's SPIR-V backend emits workgroup-memory modules that **hang NVIDIA 580**
  (`DEVICE_LOST` at dispatch, even 1×1); RADV/llvmpipe run them fine. It is
  codegen-specific: hand-authored SPIR-V using workgroup memory
  (`coopmat.zig buildGemmShared`) runs on the same driver. See `ZIG.md`
  ("SPIR-V kernels" section) and PLAN.md M9.
- So every kernel in `src/gpu/kernels/eltwise.zig` avoids shared memory. Cross-
  thread **reductions** were therefore either (a) split into a `*_partial` →
  `*_combine` pair that round-trips partials through **global** memory, or
  (b) done one-thread-per-row (serial when rows≈1, i.e. LLM decode).
- **Escape hatch (VERIFIED on 580.173.02):** subgroup-scope ops
  (`OpGroupNonUniformFAdd` reduce, `OpGroupNonUniformShuffleXor` butterfly)
  need no workgroup storage. Emit from Zig via inline asm (same trick as
  `OpSDot`); inject `GroupNonUniform` (61) + `GroupNonUniformArithmetic` (63)
  / `GroupNonUniformShuffle` (65) capabilities via `spv.withCapabilities`.
  Proven by the `"subgroup reduce runs on device"` test (context.zig) — a
  32-lane subgroup summed 32 ones → 32.0, no hang.
- Gotcha: the Zig backend **segfaults compiling a single-kernel module** that
  uses a subgroup/dp4a inline-asm op; put such kernels in a multi-kernel module.
- The CUDA backend (`src/gpu/cuda/elt.zig`) is the reference: every op below has
  a CUDA twin using a **shared-mem tree reduction** (block-per-row) or a
  **`shfl.bfly`** warp reduction — that's the shape we're recovering.

## Prerequisite: a shared subgroup-reduce primitive

Build once, reuse everywhere below:
- [x] `subgroupReduceAdd` inline-asm helper (f32) landed in a dedicated
      multi-kernel module `src/gpu/kernels/subgroup.zig` (hosts `subgroup_sum`
      probe + `rmsnorm_sg`; kept ≥2 kernels to dodge the single-kernel-module
      segfault). `subgroupReduceMax` / butterfly-shuffle variant: add when the
      GEMV / softmax work needs them.
- [x] Caps injected via `spv.withCapabilities` (`cap_group_nonuniform` +
      `cap_group_nonuniform_arithmetic`, no extension — subgroup arithmetic is
      core Vulkan 1.1). Module builds independent of `has_int_dot`. `subgroup_sum`
      moved OUT of the dp4a module (dp4a keeps only its 3 dp4a kernels + the
      DotProduct caps); `Context.subgroupProbe` now reads `pipe_sg[0]`.
- [ ] Pin subgroup size 32 where needed. NOT yet done — `rmsnorm_sg` dispatches
      LocalSize x=32 and relies on native subgroup==32 (NVIDIA wave32). Correct
      on the 580; revisit for AMD wave64 (a 32-wide wg is a partial subgroup
      there — non-uniform reduce over active lanes *should* still be correct, but
      unverified). The coop path's `VK_EXT_subgroup_size_control` pin is the model
      to copy if a device needs it.

## Tier 1 — per-layer, per-token; LLM-decode primary

- [~] **Cooperative GEMV** (block-quant). DONE for the 5 block-quant dtypes
      (q8_0/q4_k/q5_k/q6_k/iq4_nl) as a SCALAR (f32 dequant-multiply) cooperative
      kernel: `gemv_q*_sg` in subgroup.zig + `Context.opGemvQuantSg`, one subgroup
      per row over the RAW row-major weight (weightBufferRaw), lane l owns element
      l of each block, one `subgroupReduceAdd`. LocalSize 256 (8 subgroups/wg) for
      occupancy. Wired into `VulkanLM.gemvW` behind `TP_VK_SG_GEMV`. **Correctness
      VERIFIED** (device test `"gpu block-quant gemv matches cpu reference"` now
      covers the subgroup path for all 5 dtypes incl. iq4_nl; greedy 12B output
      byte-identical to baseline). **DROPS the `_t` transpose, the dp4a repack,
      AND the partials/combine pass.**
      MEASURED (12B IQ4_NL+Q8_0, RTX 3090, clean GPU, 16-vs-128 decode subtraction):
        - scalar `_t` default (opGemvQuantT): ~9.8 tok/s, ~7.2 GB VRAM
        - cooperative (opGemvQuantSg):        ~9.6 tok/s, ~6.9 GB VRAM  ← NEUTRAL speed, lower VRAM
        - dp4a opt-in (opGemvDp4a, repack):   ~22  tok/s, ~13  GB VRAM  ← 2.2×, but ~2× VRAM
      So the scalar cooperative matches the scalar default (both bandwidth/compute
      parity) at lower VRAM + fewer dispatches — a modest structural win, but it
      does NOT capture the headline 2.2×. That requires the DP4A part below.
      (First A/B was contaminated by a concurrent DiffKeep GPU load — 22s vs 16s
      "regression" vanished on a quiet GPU. Lesson: check `nvidia-smi
      --query-compute-apps` before trusting a tok/s delta.)
- [x] **Cooperative GEMV — DP4A — MEASURED DEAD END for speed.** Built
      `gemv_q8_0_sg_dp4a` + `gemv_iq4_nl_sg_dp4a` (dp4a module, GroupNonUniform
      caps added) + `Context.opGemvQuantSgDp4a`, wired behind `TP_VK_SG_DP4A`.
      Reuses `quant_act_i8`; reads RAW quads via a 1–2-load shift-combine `wquad`
      (the +2 misalignment). **Correctness VERIFIED** (device test extended, q8_0
      + iq4_nl match CPU ref). But MEASURED (same 12B, quiet GPU, isolated):
        - cooperative-dp4a (raw):  ~10 tok/s, ~6.9 GB
        - repack-dp4a (opGemvDp4a): ~24 tok/s, ~13 GB
      So it does NOT capture the dp4a win — plateaus at scalar-cooperative speed.
      ROOT CAUSE (isolated, not the byte-assembly — cutting wquad 4→2 loads moved
      nothing): **parallelism**. repack-dp4a k-SPLITS each row across nchunk=32
      chunks → rows×32 threads, short per-thread loops, + perfectly-aligned
      coalesced repack. The cooperative kernel is ONE subgroup per row → only
      `rows` subgroups, each lane serial over ALL blocks → 32× less parallelism
      for large rows. The subgroup reduce can't span multiple subgroups, so
      matching the k-split would need subgroups-handle-chunks + a combine pass
      (≈ the existing _t+combine shape, but with a subgroup reduce) — a different,
      more complex kernel. The doc's "contiguous quads → dp4a, no repack" premise
      is TRUE for correctness but the one-subgroup-per-row shape gives up the
      k-split parallelism that makes repack-dp4a fast.
- [x] **dp4a over the `_t` layout + k-split — ALSO a dead end; yields the
      FUNDAMENTAL result.** Built `gemv_q8_0_t_dp4a` + `gemv_iq4_nl_t_dp4a`
      (dp4a.zig) + `Context.opGemvQuantTDp4a` (`TP_VK_T_DP4A`): repack-dp4a's fast
      k-split shape but reading the existing `_t` buffer (weightBufferRawT) — no
      repack, no extra VRAM (7213 MiB), shares the prefill buffer (no collision,
      GEMM prefill stays on). Correct (device test extended). But MEASURED ~8
      tok/s — SLOWER than scalar (~9.8) and nowhere near repack (~24). With
      k-split now present the bottleneck is the **4-load-per-quad assembly**
      (a row's `_t` bytes are 32 B apart → 4 `tbyte` loads + shift-assembly per
      quad vs the repack's 1 aligned load); the shift-assembly ALU even exceeds
      scalar's mul savings.
      **CORRECTION (2026-07-23, after reading llama.cpp's Vulkan backend): my
      "fundamental dead end" claim below was WRONG.** llama.cpp's `mul_mat_vecq
      .comp` does fast dp4a decode GEMV from the RAW layout, no repack — it reads
      quant blocks through 16/32-bit-ALIGNED struct-aliased buffer views
      (`block_q8_0_packed16`, `block_q4_K_packed32`), unpacks to int8 IN REGISTERS
      (incl. iq4_nl LUT), and reduces with `subgroupAdd` over a k-split where the
      workgroup == 1 subgroup and the subgroup size is PINNED via
      VK_EXT_subgroup_size_control. So low-VRAM raw dp4a IS achievable (their fast
      path is exactly the no-shmem subgroup variant we're allowed to use). My
      sg-dp4a/t-dp4a were slow from IMPLEMENTATION gaps, not a fundamental limit:
      (1) byte-assembled quad loads (wbyte/tbyte) instead of aligned packed
      16/32-bit loads; (2) no pinned subgroup size on the GEMV pipelines; (3) a
      lane→memory mapping (phase-strided blocks 34 B apart) that doesn't coalesce
      cleanly. REWRITE DONE + MEASURED: matched mul_mat_vecq — contiguous K-chunk
      mapping, LocalSize 32 (1 subgroup/wg, many small wgs), AND aligned u16 loads
      (WU16 aliased view + StorageBuffer16BitAccess; Zig SPIR-V accepts u16
      storage, runs on 580, correct 13/13). Result ~11.5 tok/s (up from ~10.8
      byte-assembly) vs repack ~24 — still ~2× off. ISOLATED remaining cause:
      repack reads a PRE-PACKED u32 (1 load→dot4); the raw kernel does a u16-load
      + shift-or PACK per quad ≈ 2× the non-dp4a instructions/dp4a, and decode is
      instruction-bound (repack has the SAME weight bandwidth yet runs 2×). The
      pre-packed u32 IS the repack layout (2× VRAM for iq4_nl). Untried levers:
      NUM_ROWS>1 (amortize activation/overhead across rows, llama.cpp rm_stdq≈2 —
      maybe +20-40%, won't fix per-quad pack); u8vec4 direct load via 8-bit
      storage (already enabled — 1 load, no pack; helps q8_0, not iq4_nl nibbles).
      NUM_ROWS=4 ALSO TRIED (each subgroup computes 4 rows, reusing activation
      loads): ~11.7 tok/s — no meaningful change, confirming the bottleneck is the
      per-quad weight unpack (not activation/overhead). FINAL: every llama.cpp
      technique applied (contiguous mapping, aligned u16 loads, subgroup reduce,
      1 subgroup/wg, NUM_ROWS=4) → raw-layout dp4a caps ~11.7 vs repack ~24. The
      per-quad u16-load+shift-pack-to-u32 is the irreducible ~2× cost of NOT
      pre-packing; only the repack's pre-packed u32 avoids it (2× VRAM for iq4_nl).
      NET: u16 sg-dp4a (TP_VK_SG_DP4A, ~11.7) is the best low-VRAM dp4a option
      (vs scalar ~9.8); repack stays the speed king (opt-in, 2× VRAM);
      q8_0-repack-default shipped (the practical win). Zig SPIR-V u16-storage now
      proven working (new capability). _t_dp4a (~8, TP_VK_T_DP4A) is strictly
      dominated — removal candidate.
      (Superseded "fundamental" reasoning kept below for the record.)
      ~~FUNDAMENTAL CONCLUSION (measured + reasoned across sg-dp4a, t-dp4a):
      repack-dp4a's ~24 tok/s = k-split AND 1-load *aligned* quads; BOTH required.
      Aligned quads only exist in the row-interleaved repack layout; in-register
      int8 assembly (from raw or `_t`) always erases the dp4a win.~~
        - q8_0: the repack is only ~6% larger than raw (1152 vs 1088 B / 32-row
          group) — dp4a speed is ~free. **LANDED: `VulkanLM.dp4aRepack(dt)` makes
          q8_0 use repack-dp4a by DEFAULT (decode + prefill, layout-consistent);
          iq4_nl stays opt-in (TP_VK_DP4A, ~2× VRAM).** MEASURED 12B (196 iq4_nl +
          80 q8_0): ~11.1 tok/s vs ~9.8 scalar (+13% — only the 80 q8_0 layers
          sped up here; a q8_0-heavy model gets the full ~2.4×), VRAM +~0.3%.
        - iq4_nl: dp4a NEEDS int8 = inherently 2× the 4-bit form (stored=repack/
          fast, or assembled-per-read=slow). NO low-VRAM dp4a for iq4_nl exists —
          fundamental: scalar-speed @4-bit OR dp4a-speed @2×.
      So `TP_VK_SG_DP4A` / `TP_VK_T_DP4A` are measured-dead-end experimental
      kernels (behind flags for the record; removal candidates). Shipping wins
      from the whole GEMV effort: scalar cooperative (lower VRAM, neutral speed)
      + the q8_0-repack-default insight.
- [ ] **Cooperative GEMV — dense** (bf16/fp8/f32). Left on the existing k-split
      `gemv_partial`/`combine` for now: dense weights are stored in the shared
      GEMM k-major layout (NOT contorted like the block-quant `_t`/repack), so a
      cooperative version would need a separate raw row-major buffer (extra VRAM)
      or a non-coalesced k-major read. Lower priority than the DP4A item.
      *LLM only* (diffusion linears are m=seq → coop GEMM, not GEMV).
- [~] **One-pass RMSNorm.** DONE for the LLM (`rmsnorm_sg` in subgroup.zig +
      `Context.opRmsNormSg`): one subgroup per row, lanes stride the row summing
      squares → single subgroup reduce → each lane writes normed*w. Replaces the
      3-pass `rms_partial`/`rms_combine`/`rms_apply_w` (`VulkanLM.normWide`, fires
      ~2×/layer/token). **Correctness: VERIFIED** — device test `"gpu subgroup
      rmsnorm matches cpu reference"` (rows∈{1,4,3}, dim∈{6144,6157}) passes on
      580 (no DEVICE_LOST — the escape hatch works for a real reduction kernel).
      **tok/s: MEASURED NEUTRAL** on a 12B (llama-arch) Vulkan decode: baseline
      ~10.7 vs subgroup ~10.6 tok/s (2-run 16-vs-128 subtraction, within
      run-to-run noise) — as predicted, a 12B's per-token cost is GEMV-dominated,
      not norm-dominated. Currently **opt-in via `TP_VK_SG_RMS`** (not yet
      default): neutral+correct on the one model testable here, but the regime
      where the dispatch-count drop would show (small model / high tok/s) has no
      Vulkan-runnable checkpoint on this box (all local qwen3 GGUFs have
      block-quant embed tables Vulkan rejects; only F16-embed GGUF is this 12B).
      Flip to default once the GEMV work lands a fast small-model path to measure.
      STILL TODO: diffusion side (`rms_apply_mod` variant; dit_gpu.zig:443/809/905)
      — needs a modulation-aware `rmsnorm_mod_sg`. CUDA twin: `rms_mod`/`qk_rmsnorm`.

## Tier 2 — per-token / per-block; medium

- [~] **Decode attention** `attn_dsplit_gemma` → `attn_dmerge`: DONE for the
      gemma/qwen35 shape as `attn_decode_sg` (subgroup.zig) + `Context.
      opAttnDecodeSg`: one subgroup per head, lanes = 32 KV chunks, the online-
      softmax merge folded via `subgroupReduceMax` (running maxes) + `subgroup
      ReduceAdd` (reweighted dsum, then each acc element) — NO global scratch, NO
      attn_dmerge dispatch. **Correct** (device test extended vs the CPU ref,
      window+ring+GQA+bidirectional). Wired into `gemma3_gpu.attention` behind
      `TP_VK_SG_ATTN` (f32 KV only). tok/s NOT measured — the only local
      Vulkan-runnable model (the 12B) is qwen3/llama-arch, which uses the
      *batched* `attn_dsplit` path (per-query causal, seq_q>1), NOT the gemma
      single-query path; benchmarking would need a gemma model or a second folded
      `attn_dsplit` variant. Expected NEUTRAL per this session's consistent
      pattern (every subgroup rework = correct + VRAM/dispatch win, tok/s-neutral,
      because the heavy QK/PV compute is untouched). f16/q8 KV variants + the
      batched `attn_dsplit` fold: not built.
- [ ] **Attention softmax** `softmax_partial` → `softmax_combine`: max+sum
      reduction over keys, split via global partials. Subgroup-reduce in one
      pass. **Diffusion only** — DiT attention (dit_gpu.zig:726/733) and VAE
      mid-block attention (vae_gpu.zig:336/343). (No LLM stepper uses it: LLM
      decode is `attn_dsplit`/`dmerge`, prefill is `attn_scores`/`attn_out` with
      online softmax in registers — leave those.)

## Tier 3 — once/token or one-shot; minor

- [ ] **argmax / top-k** (`argmax_reduce`+`argmax_final`, `topk_reduce`): lane-
      partial reductions over the 131k vocab. Subgroup simplifies, but runs once
      per token (sampling). *LLM only.*
- [ ] **`vae_norm`**: serial per-row L2/group-norm reduction (eltwise.zig ~137).
      Subgroup-reduce. *Diffusion VAE decode* (one-shot at the end). Low.
- [ ] **`l2norm_rows`** (qwen35 DeltaNet), **`layernorm`** (SigLIP ViT): one-
      thread-per-row reductions; niche/one-shot. Low.

## Not candidates (checked)

- Per-head QK-norm (`rmsnorm`, one-thread-per-row): `seq*n_heads` rows × 128
  elems — parallel across rows, no split. Fine as-is.
- Elementwise (`add`, `silu_mul`, `gelu_mul`, `rope_half`, `copy`, `modulate`,
  …): no reduction.
- Coop GEMM / coop attention (`buildGemm*`, `opFlash`, `opAttnScores`): already
  hand-authored SPIR-V with correct subgroup/workgroup use. Diffusion's dominant
  cost lives here — **not** contorted; its ceiling is MMA-issue density
  (PLAN.md), a separate problem.

## Does this cover the diffusion side?

**Partially, and secondarily.** Diffusion has its own contorted reductions —
RMSNorm (shared with LLM), plus `softmax_partial`/`combine` and `vae_norm`
(diffusion-only) — so the Tier 1 RMSNorm and Tier 2 softmax items help it. But
diffusion time is dominated by the **coop GEMMs + coop attention**,
which are already on the hand-authored tensor-core path and are *not* affected
by the no-shared-memory limitation. So:
- **LLM decode**: these reworks are the primary perf lever (decode is all
  GEMV + norms + split attention; no coop GEMM to hide behind).
- **Diffusion**: expect modest gains on norms/softmax only; the matmul ceiling
  is unchanged. Don't expect a step change here from this work.

## Verification pattern

Each rework: keep the old kernel, add the subgroup variant behind a flag,
compare (a) numerics vs the old path (gated test, like `"subgroup reduce runs on
device"`) and (b) tok/s via 2-run subtraction (`--max-tokens 16` vs `128`,
same prompt; the 3090 idles clocks so single short runs are noisy). Land as
default only after both check out.
