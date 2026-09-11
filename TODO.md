- gpu: still per backend, and candidates for `kernels/dual.zig`: the block-quant GEMVs
  (CUDA warp-per-row dp4a/f16, Vulkan `_t`/`_sg`), attention, GEMMs. A shared
  subgroup-per-row GEMV over the RAW ggml layout would replace both arms' scalar
  fallbacks, with the dp4a/MMQ fast paths staying hand-tuned. Vulkan's half of it is
  already five hand kernels in `kernels/subgroup.zig`, deliberately NOT in `dual/`:
  moving them there costs the CUDA arm PTX-JIT time for kernels it never calls
- gpu: shared memory stays out of the Zig-emitted kernels until Vulkan stops hanging
  NVIDIA on workgroup memory. That blocker is what keeps `qk_rmsnorm_par` alive for
  the one-row decode norm, and what parks the LLM's `normWide` on the 3-pass chain:
  splitting a wide row across subgroups needs a cross-subgroup reduce. MEASURED, so
  don't retry it as-is at one row: a subgroup-per-row `normWide` is 32 lanes on one
  multiprocessor and costs 6% of decode (30.9 -> 29.0 tok/s, 8B q8_0)
- cuda: the PTX targets ONE compute capability (`-Dcuda-sm` / `-Dptx-isa`, default 86,
  applied to every PTX module at once). Forward JIT onto sm_89/90/120 should work and is
  UNVERIFIED; anything below sm_80 cannot run the hand kernels at all (cp.async,
  mma.m16n8k32), and a kernel needing an instruction an older target lacks would still
  have to be written by hand. A startup check against `cc_major` that says so, rather
  than a JIT log, is the missing piece
- lora: the Vulkan arm has no sidecar apply, so `sensenova_gpu.supported` refuses a
  model with one attached and the trunk falls back to the CPU. `sensenova_gpu.linear`
  is the single funnel to hang it off; the kernel is two `opMatmulCoopBf16` calls plus
  a scaled add, and `sensenova-vk-test --lora` already takes the axis
- lora: krea2, Z-Image and Anima route their device GEMMs through `lin_cuda` too, so
  each is a `lora` field on the DiT plus a `plan.lora` assignment plus a host funnel;
  they currently REFUSE a `--lora` rather than ignoring it. The SD UNets need conv
  factors (kohya's `[r, in, kh, kw]` plus `lora_mid`) and a text-encoder LoRA
  (`lora_te_` / `lora_te1_` / `lora_te2_`), neither of which `lora.zig` has
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
- llm weight noise (`--weight-noise`, BACKEND.md 6) is unwired on k2-horizon's stepper,
  the dequant-to-f16 fallback (the batched route for q4_0 and iq4_nl),
  `opGemvQuantQ8Batch`, and the fp8/bf16/int8-convrot GEMMs
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
- llm: two gaps left in reasoning-marker detection (`chat.observeReasoning` plus
  `tp-llm --reasoning-markers`): a fine-tune whose markers are in NEITHER
  `known_reasoning` nor the flag still falls back to the family guess, and tp-gui has
  no override field, so a GUI user cannot answer for such a model
- gui: a transcript does not record the system prompt it was generated under, so
  reloading replays an old conversation under current settings (e.g. with or
  without the image-tool description). Same class as the markers were
- gui: replaying a conversation into a DIFFERENT model feeds the old model's
  reasoning markup in as literal content; real chat templates vary in whether
  they keep prior reasoning at all (Qwen's drops it)
- sensenova: the 64x64 canvas in the device tests carries ~25x the 96x40 case's
  device-vs-CPU error at the same depth, on BOTH arms and with the same reference
  magnitude, so it is neither a backend nor a small denominator (the printed control row
  rules that out). It does not move the full-render figure, but nothing explains it
- sensenova: image EDITING renders the reference's structure exactly and its TONE
  wrong: at a 1024^2 canvas with a photographic reference of 16 tokens or more the
  output posterizes (46-54% of pixels clipped, against 0% for the same prompt with no
  reference). It is not the conditioning and not a device arm. Measured: the edit
  prefix matches ComfyUI to 2.1e-4 at fp32 over 4 layers and to the bf16 floor
  (2.6e-2 k, 5.3e-2 v) over all 42; a whole denoiser forward on that prefix matches
  `_forward` to 9.5e-7; ids, thw indexes and the block-causal mask match cell by
  cell; CPU and CUDA agree. It depends on the reference's CONTENT (a smooth
  photographic reference posterizes, white noise at the same shape does not) and on
  the canvas (256^2 and 512^2 canvases are clean, 1024^2 is not). Unmeasured, and the
  reason this is still open: whether ComfyUI's own 42-layer render of the same
  checkpoint differs, because a 42-layer fp32 reference does not fit this box's RAM.
  Next probe is `img_cfg_scale` -- SenseNova's own `modeling_neo_chat.py` guides
  editing with TWO scales and ComfyUI's port carries one, though at cfg 1 the two
  agree, which is where the defect already shows
- sensenova: q4_k is measured as unusable for this architecture and the receipt is in
  BACKEND.md 2G. What is NOT measured is where the ceiling actually is: q8_0, q6_k and
  the int8-convrot conversions (`jtreminio/SenseNova-U1.5-8B-MoT-int8_convrot`) are all
  wired through `lin_cuda` already and none has been rendered on either GPU arm. That
  also leaves the 4096 / 12288 `i8_prep_cols` entries the Vulkan port added untested,
  so a quantized checkpoint there may silently take the 3-pass prep fallback
- h3 text encoder: the 50-layer encode streams all 23 GB every prompt (2.4 s) because the
  weight cache is LRU and a sequential walk larger than the cache evicts each layer just
  before its next use. Keeping the first ~25 layers resident across prompts (a pin
  scoped to the encoder that the DiT's `evictUnpinned` still drops, or MRU-aware
  eviction for scans) would halve it. Only matters if the DiT leaves that VRAM free
- windows: what is left is the system C libraries, since every Zig source compiles and
  the SDL3/dvui link succeeds. libvips and the libav set have to come from somewhere with
  `.pc` files, which is what the release workflow's MSYS2 step is for. Nothing on Windows
  has been executed (`zig build test -Dtarget=x86_64-windows` cannot RUN the binaries),
  so `filemap`'s section+view mapping and `dynlib`'s LoadLibraryW arm are both UNRUN
- macos: no Metal backend, so a Mac is CPU-only, which the README's own numbers put at
  ~290 s/step. The Vulkan arm would reach it through MoltenVK (`openVulkanLib` already
  names the dylibs) but coopmat is unlikely to be there, so the non-coop GEMM path is
  what would have to carry it
