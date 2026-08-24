- begin filling in holes in the capabilities grid (BACKEND.md)
- add more sampling methods
- gui: studio (image_view) still uses its own form layout; bring the parameter form onto the shared chip/section primitives
- the CPU steppers' `prefill()` is still one un-chunked forward over the whole
  tail, so it has no boundary to stop or re-plan at. Inert today (the GUI is
  CUDA-only and the CLI arms no cancel), but any CPU-backed frontend needs it
- even a tiny bit of offloading of gemma4 31b is extremely slow
- diffusion model weights are "bouncing" during steps, vram-wise
- there's no good visiblity of "how much of the model is in vram" for either side
- llm weight noise (`--weight-noise`, BACKEND.md 6) covers gemma4's q4_k/q5_k/q6_k
  GEMVs and the q4_k-family MMQ pipe. Unwired: every other arch's stepper (only
  `gemma4_cuda` declares `noiseAtLayer`), q4_0 (so the Gemma 4 12B QAT files get
  nothing), the q6_k/q8_0/q1_0/q2_0 MMQ pipes, the dequant-to-f16 fallback, and the
  fp8/bf16/int8-convrot GEMMs. No longer SILENT — the GUI hides the controls and
  the CLI warns, both off `llm.session.weightNoiseSupported` — but the coverage gap
  is the real fix. q4_0 is the cheapest next one: `gemv_q4_0_q8n` / `gemv_q4_0`
  take the same `%rd10`/`%rd1` snippet as the k-quants
- llm weight noise: the interesting use is not diversity but UNCERTAINTY. Sample
  one token k times under independent perturbations: a lead that survives is
  redundantly encoded (the model knows it), one that flips is riding a few fragile
  weights (a guess). Logit entropy cannot tell confidently-right from
  confidently-wrong; this can, in principle. Needs a k-sample harness and a
  calibration set to know whether it actually correlates with hallucination
- llm weight noise: sweep the curve against factual recall vs grammar on a real
  benchmark to turn it into a quant error-budget oracle (how much scale precision
  an arch tolerates, PER DEPTH, measurable before writing a new format's kernel).
  The placement result (BACKEND.md 6) says the useful output is a sigma-vs-depth
  tolerance profile, not one number, and it suggests mixed-precision quantization
  should follow that profile rather than ggml's fixed early-layer bump. Everything
  measured so far is single prompts eyeballed, not a curve
- llm prefill: port `buildMmqPipeQ1_0`'s cp.async double-buffered pipeline to the
  UNPACKED variants of `buildMmqPipeQ4K` (q5_k, iq4_xs), leaving q4_k/q6_k on the
  single-buffered path. `TP_MMQ_NOSTAGE` measures the ceiling: exposed A-staging is
  worth 0% on q4_k (raw copy), 12% on q5_k, 30% on iq4_xs, and q1_0's own header
  records 65.0 -> 86.9 TOPS for the same rewrite. ~10% end-to-end on a q5_k model
- llm prefill: q6_k's MMQ pipe is correct but measures ~1.0x against dequant+f16, so
  `mmqPipeFaster` leaves it off. It is the only k-quant still on the f16 route. Worth
  retrying with PLAIN s8 staging, the lever that took q5_k/iq4_xs from ~32 to ~68 TOPS
- llm: reasoning markers come from `chat.reasoningFor(family)`, a static guess per
  family. A fine-tune that emits different markers than its base is mis-split
  LIVE, not just on reload. The real source is the model's own chat template;
  the open marker could be observed from the rendered generation prompt (where
  `recordThoughtPrimed` already looks), the close marker cannot, so this likely
  needs an explicit per-model override to be correct
- gui: a transcript does not record the system prompt it was generated under, so
  reloading replays an old conversation under current settings (e.g. with or
  without the image-tool description). Same class as the markers were
- gui: replaying a conversation into a DIFFERENT model feeds the old model's
  reasoning markup in as literal content; real chat templates vary in whether
  they keep prior reasoning at all (Qwen's drops it)
