- gpu: `kernels/dual.zig` holds every elementwise kernel, every dequantizer and the row
  reductions for both GPU arms. Still per backend: the block-quant GEMVs (CUDA warp-per-row
  dp4a/f16 kernels, Vulkan `_t`/`_sg` kernels), attention, GEMMs. A shared subgroup-per-row
  GEMV over the RAW ggml layout would replace both arms' scalar fallbacks; the dp4a/MMQ fast
  paths stay hand-tuned. The Vulkan half of that is done as five hand kernels in
  `kernels/subgroup.zig` (q4_0/iq4_xs/q1_0/q2_0 now decode there instead of dequantizing the
  whole weight per token); putting them in `dual/` instead would cost the CUDA arm PTX-JIT
  time for kernels it never calls, which is why they are not there. Shared memory stays out until
  Zig-emitted workgroup memory stops hanging NVIDIA on Vulkan; that is also what keeps
  `qk_rmsnorm_par` alive for the one-row decode norm
- gpu: the DiT's norms are one fused row kernel each (`rms_mod` / `rms_mod_h16`), but the
  LLM's `normWide` stays on the 3-pass chain, MEASURED: at the decode's one row a
  subgroup-per-row kernel is 32 lanes on one multiprocessor and costs 6% of decode
  (30.9 -> 29.0 tok/s, 8B q8_0). A row kernel that splits a wide row across subgroups
  needs a cross-subgroup reduce, which is the workgroup memory Vulkan cannot have here
- gpu: `dual_ptx` builds for one CUDA generation at a time; `-Dcuda-sm` / `-Dptx-isa`
  now set it for every PTX module at once. Untested past sm_86: only the ISA header and
  the lowering flags are parameterized, and a kernel using an instruction the older
  target lacks would still be written by hand
- begin filling in holes in the capabilities grid (BACKEND.md)
- add more sampling methods
- gui: studio (image_view) still uses its own form layout; bring the parameter form onto the shared chip/section primitives
- preview: the TAESD ladder only has the Wan decoder (`taehv`), so the settings
  "Preview decoder" row appears for Krea2 and Anima alone. Opening it to SD1.5 / SDXL
  (`taesd_decoder.pth`, `taesdxl_decoder.pth`) and Z-Image (`taef1_decoder.pth`) needs
  (a) a reader for PyTorch's zip+pickle `.pth` container, or a `tools/convert_taesd.py`
  that rewrites them as safetensors, and (b) a `taesd.zig` decoder (TAESD is TAEHV
  without the temporal memory blocks; `wan_vae.loadConv` and the taehv CPU/CUDA/Vulkan
  kernels are the parts to reuse) plus a `previewFits` arm per latent format in
  `gui/model_spec.zig`. The catalog and the settings row need nothing else: a file
  that answers `previewFits` for a family is offered for it
- the CPU steppers' `prefill()` is still one un-chunked forward over the whole
  tail, so it has no boundary to stop or re-plan at. Inert today (the GUI is
  CUDA-only and the CLI arms no cancel), but any CPU-backed frontend needs it
- even a tiny bit of offloading of gemma4 31b is extremely slow
- diffusion model weights are "bouncing" during steps, vram-wise
- there's no good visiblity of "how much of the model is in vram" for either side
- llm weight noise (`--weight-noise`, BACKEND.md 6) now reaches every block-quant GEMV and
  every MMQ pipe, on the gemma3, gemma4, qwen3 (and llama) and qwen35 steppers. Still
  unwired: k2-horizon's stepper, the dequant-to-f16 fallback (the batched route for q4_0
  and iq4_nl), `opGemvQuantQ8Batch`, and the fp8/bf16/int8-convrot GEMMs
- llm weight noise: ⚠️ injecting the jitter into a hand-PTX kernel by CONCATENATION lands
  it on whatever line the string ended on, and a Zig multiline literal's last line carries
  no newline — so a kernel whose last line ended in a `//` comment silently swallowed the
  `ld.param` of sigma, leaving it uninitialized (read as 0 = noise off) with no error
  anywhere. `every injected weight-noise statement starts its own line` now fails on it.
  That test also found that `cuda/elt.zig`'s tests had NEVER run: nothing in the module
  tree referenced the file, so its two PTX guard tests were dead
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
- llm: reasoning markers are now read from the model's own chat template
  (`chat.observeReasoning`, both halves of a pair or nothing) and `tp-llm
  --reasoning-markers` overrides them. Two gaps left: a fine-tune whose markers are
  in NEITHER `known_reasoning` nor the flag still falls back to the family guess,
  and tp-gui has no override field, so a GUI user cannot answer for such a model
- gui: a transcript does not record the system prompt it was generated under, so
  reloading replays an old conversation under current settings (e.g. with or
  without the image-tool description). Same class as the markers were
- gui: replaying a conversation into a DIFFERENT model feeds the old model's
  reasoning markup in as literal content; real chat templates vary in whether
  they keep prior reasoning at all (Qwen's drops it)
- h3 text encoder: the 50-layer encode streams all 23 GB every prompt (2.4 s) because the
  weight cache is LRU and a sequential walk larger than the cache evicts each layer just
  before its next use. Keeping the first ~25 layers resident across prompts (a pin
  scoped to the encoder that the DiT's `evictUnpinned` still drops, or MRU-aware
  eviction for scans) would halve it. Only matters if the DiT leaves that VRAM free
