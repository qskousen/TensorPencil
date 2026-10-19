- one weight path for every model: every weight GEMM and conv goes through one
  dispatcher per backend, so a format, a kernel or a LoRA route added once reaches
  every family. Done: every linear goes through `lin_cuda` / `lin_gpu` and every
  convolution through `conv_cuda` / `conv_gpu`. In order, each
  with an interleaved A/B and, where the kernels do not change, a byte-identical render:
  1. DONE: SD UNet linears on CUDA through `lin_cuda` (`run`, `Plan.dense`,
     `takesF16`); byte-identical renders on f16 SDXL, f32 SD1.5 and with a LoRA, and
     int8 / q4_k SDXL now render. Measured (`diffusion-profile`, quiet GPU, 1024²,
     one forward): f16 137-146 ms, int8 convrot 122 ms, q4_k 196-199 ms. The bias
     now rides every route's epilogue but fp8, the f16 block-quant decode and MMQ
     (`lin_cuda.gemmBias`); what is left of int8's GEMM win goes to its activation
     prep (8 ms, once per input since q/k/v share one). q4_k pays its per-GEMM
     decode and twice the elementwise launches
  2. DONE: CLIP on CUDA through `lin_cuda`, loader on `quant_weight.load`; a bundled
     int8 SDXL loads and renders on its own towers. Dense CLIP renders pixel-identical.
     Integer towers run weight-only (`lin_cuda.encoder_int8`): conditioning against
     the dense source 0.0084-0.0089 rel L2, W8A8 0.012-0.013 (`cond-diff`).
  3. DONE for the SD UNet: convs stay in the checkpoint's order and dtype, every CUDA
     conv reads one f16 device weight (`Backend.weightF16`), block quants decode per
     call. q4_k SDXL DiT VRAM 3306 -> 2141 MB; the SD/Flux VAE (`sd_vae`) is on it too,
     f16 SDXL peak RSS 11.08 -> 8.09 GB; `wan_vae` (Anima 6.74 -> 6.37 GB) and `mage_vae`
     (Mage-Flow -0.7 GB) too, and the conv LoRA's A (f16, file order); Vulkan keeps no
     f32 copy of any of them where it has the coop GEMM.
  - These real-model test files are gone from this machine and their tests skip: the
    `krea2CenterSemiraw_v10{Fp8,Int8,Int8-ASYM_W4A8_INT8,Int8-INT4_CONVROT_SR}` DiTs,
    `anim-q4k-calib.gguf`, `lpips_alex`, the two Anima int8/W4A8 conversions,
    `models/assets/minimax_h3_vae.safetensors`.
  4. The SD sidecar's cost: now +12-16% per step on SDXL (was +55%); a LoRA keeps
     the f16 chain and the int8 A path is wired. A shared A GEMM is measured out
     (≤1.8 ms/step ceiling), and so is running the sidecar first (B's read of the
     output plane is 1.2 ms/step, 1.7%, `lora-sidecar-bench`'s `b_overwrite`). What is
     left is ~9.5 ms/step of B side on SD1.5 768² beyond that read, not yet attributed
     between its GEMM, the `lo` bf16 pass and launch count. And LoKr, DoRA and whole diffs, which it refuses today, each
     have an exact runtime form (LoKr's kron is two reshaped GEMMs, DoRA a per-row
     or per-column scale computed once at attach)
  5. `dual_lin`: one linear dispatcher, `dual.zig`'s idea applied to the GEMMs. In
     place: `models/dual_lin.zig` (Route, Policy, `routeOf`, `Plan`, `check`, all over a
     comptime `Caps` adapter) and one executor per backend, `models/lin_cuda.zig` and
     `models/lin_gpu.zig`, which every diffusion trunk, text encoder and LLM stepper on
     both backends calls. Device checks: `lin-cuda-test`,
     `lin-vk-test`. Left:
     - SenseNova's generation check cannot tell rounding-flip drift from a defect past
       a few layers on the 4-token canvas (BACKEND.md 2G). Mage-Flow's split (weight-only
       walk, noise magnitude, per-route emulation; BACKEND.md 2H) holds at any depth and
       is the shape to move SenseNova's onto.
     - Vulkan's fp8 coop GEMM is built with f16 accumulators unconditionally
       (`coopmat.coop_acc_h16`), while its f16-store sibling is gated on the device
       offering them (`coop_acc_f16`): Intel Alchemist advertises f32/s32 accumulate only,
       so the fp8 pipe asks the Arc for a configuration it never offered. Untested there.
     - `--vram-budget` on CUDA refuses a 72 MB allocation mid-forward on a 1024² krea2 int8
       render at 6, 10 and 16 GiB, intermittently at 16 (two runs in three), with every
       unevictable entry pinned (~190) or an unconsumed prefetch (13): the pin claims leave
       the activations no room under a budget.
     - Vulkan's W8A16 (an explicit `--dit-act f16` / `--te-act f16` on int8) runs ~1.5x
       a W8A8 krea2 step: the f16 coop GEMM is 44-52 TF/s against int8's 77-90
       (`vk-wo-bench`), and the pack is no longer the cost. The f16/bf16 coop GEMM's own
       rate (~50 of the card's ~71 at f32 accumulate) is the lever. It is also 61% of a
       1024² Wan decode on Vulkan now that the conv gathers into its operand.
     - Vulkan's fused int8 prep at 25600 (four chunks) rotates twice at 99 GB/s; it runs
       only under an explicit W8A8 on a 32B encoder.
     - LLM parity, warm (`tp-llm --bench`, 1309-token prompt, greedy ids identical):
       Qwen3-4B q4_k: Vulkan pp 1951 / tg 63 (`qwen3_gpu`, one batched forward at any
       position, 2048-row chunks); Qwen3.5-9B q6_k: Vulkan pp 1074 / tg 37.5, CUDA 1550 /
       84 (`qwen35_dev.zig` on both). Left:
       - Vulkan decode is now GEMV-bound: Qwen3-4B q4_k at short context (`TP_VK_PROFILE`)
         spends 74% of GPU time in the dual GEMVs. On the f32 route decode takes, the Vulkan
         build now matches or beats zig-cuda's build of the same source (`dual-gemv-bench`,
         q4_k 480-500 GB/s, q6_k 460-530, after the k-quants' 16-element units and two rows
         a subgroup, which took Qwen3-4B decode 91 -> 112 tok/s); what CUDA
         has beyond that is its dp4a q8 route, which Vulkan cannot match without `OpSDot`.
         Done on the dispatch side: residual add + hidden norm in one workgroup kernel
         (`coopmat.buildAddRms`), q/k norm + rope in one (`qk_norm_rope`), K/V projected
         straight into an f32 cache, the merge's loads batched, a one-subgroup argmax;
         67.5 -> 79.9 tok/s under `TP_VK_PROFILE` (89 without it). Left: silu_mul into the down GEMV (0.13 ms a token), the
         attention merge (7 us a call).
       - Prefill attention: the old square tiled f32 kernel was 8.5% faster than the
         grouped split one on a fresh 1309-row chunk (measured), and both are scalar; a
         coopmat flash kernel is the lever. A block-quant prefill also re-expands every
         weight to f16 per chunk (why the chunk is 2048: 512 rows was 1563 tok/s).
       - Every dense weight is cached twice on Vulkan (GEMV and coop layouts): a bf16 4B
         holds 16 GB. One layout both read would halve it.
       - Vulkan decode against CUDA: ~8.9 ms a token vs 7.6 (Qwen3-4B q4_k, 112 vs 131 tok/s).
         The GEMV kernels now match CUDA's in a back-to-back bench; what is left is each
         GEMV's ramp and drain between barriers and the small ones (k is 1.5 MB), so the
         lever is the step's structure, not a kernel: fewer barrier-separated levels
         (silu_mul folded into the down GEMV, more `independent` groups), or a persistent
         decode kernel. Design it before building it.
       - The q8 decode GEMV is wired on Vulkan (`opGemvQuantizeX`, `gemv_dual_q8`) but off,
         and an `OpSDot` module would not turn it on: MEASURED with `dp4a` emitting real
         `OpSDot` (capabilities injected, `dual-gemv-bench`), q8 trails the f32 route on
         the k-quants (q4_k 371 vs 394 GB/s, q5_k 447 vs 472, q6_k equal) and gains ~5% on
         q8_0 only, before its quantize pass, while the f32 route already runs q8_0 at
         800+ GB/s. A q8 kernel that decodes in 16-element units is what could change that.
       - The one-thread-per-query `Backend.attn` kernel in its remaining callers (the DiT
         text fusion, anima, the embedders, the H3 VAE and audio encoder, CLIP), most of
         them non-causal, which `opAttnDecode` has no mode for.
       - CUDA's f16 KV path changes the greedy tokens on Qwen3.5-9B where f32, q8_0 and
         Vulkan f16 do not (predates the shared forward).
  6. The f16 weight-only integer routes (`*_wo16`, krea2's text fusion) have no
     checkpoint here that reaches them; only `lin-cuda-test` and `lin-vk-test` run them.
- activation precision as a per-component dial (text encoder, denoiser, VAE):
  f32 / f16 / bf16, default `auto`. Its prerequisites are done: one device forward per
  family and VAE (`models/*_dev.zig`), typed device buffers, dtype-generic `dual.zig`
  kernels and a dtype-aware `lin_cuda` / `lin_gpu`, so the dial is written once per
  family. The machinery is in (`src/precision.zig`, `Session.actFor`/`widen`, the
  per-step finite check, the VAE ladder, CLI and GUI); what a component honours is
  the SD UNet (cuda and Vulkan), krea2's DiT (cuda and Vulkan; zig-cuda stays f32) and
  the SD/Flux VAEs on every GPU arm, Z-Image, Anima and Mage-Flow on every arm (default
  f32: an explicit f16 overflows at step 1 on every checkpoint), and SenseNova's canvas
  on every arm (default f32; f16 runs through a scaled residual). SenseNova's overflow was the residual (~7e4 at the last
  layer of the last step) and the down projection's output (~5e4): the stream is stored
  divided by 16 with the RMSNorm eps scaled to match and the delta kept f32
  (`sensenova_dev.streamDiv`), which made f16 viable. Z-Image and Anima are the same
  shape of problem; trace WHERE they overflow (their residual, a delta or the MLP) and
  try the same split before settling for f32. H3's trunk takes the same split (residual
  /256, default f32). The Wan, H3 video and Mage VAEs and every text encoder (CLIP, Qwen3, SenseNova's
  prompt pass) follow the dial on every arm; every device check takes `--act`.
  DiT defaults are f16 only where raw f16's range
  fits with no workaround, else f32; never ComfyUI's bf16 (on krea2 bf16 costs ~7x f16's
  velocity error at the same speed). Vulkan: krea2's
  fp8 forward's larger perturbation floor is the fp8 GEMM's f16 accumulators and costs the
  forward nothing against the CPU (BACKEND.md 2A); its int8 prep reads only f32, so a 16-bit stream pays a
  widening copy per group. The PNG field is our own (`Activation dtype`), since
  neither A1111 nor Civitai has one. Vulkan's 3x3 convs now gather straight into
  the coop GEMM's 16-bit operand; its LINEARS still re-cast and re-pad an operand that
  is already 16-bit (`coopGemmPad`), and reading an aligned one directly is open. fp8 is
  not a chain dtype; it is a later GEMM-input route like W8A8, on cards with fp8
  tensor cores. The rules:
  - `auto` = a per-family static rule (each component's default and allowed
    dtypes; ComfyUI's choices: f16 SD UNet, bf16 DiTs, f32 where range forces)
    plus a finite check on every step's output. No calibration forward, nothing
    written next to a checkpoint.
  - A known overflow with an exact fix (a power-of-two prescale, an f32 scores
    plane) takes the fix, keeping the dtype.
  - On a non-finite step: fail with an error naming the component, dtype and step;
    the CLI logs it, the GUI toasts it (over the wire from a remote host), and the
    render reruns at the fallback, f16 then bf16 then f32, skipping what the card
    cannot run fast.
  - The fallback is REMEMBERED per component for the session, so later images do
    not retry; it clears on unload or a model switch, survives LoRA swaps, and a
    change of that component's dial clears it. Reported once; each PNG records
    the dtype it actually used.
  - An explicit dtype choice gets the check and no fallback.
  - Receipt for whether bf16 is an acceptable SD fallback: SDXL at f32 / f16 /
    bf16, one seed; if bf16 is clearly worse than f16, SD falls back to f32
- mageflow: the Vulkan bf16 GEMM runs at 38 TFLOP/s against cuBLASLt's 52, and it
  is at its OWN ceiling, not being called badly (`anima-vk-bench` reports 36.8-40.2
  at every shape). Kernel work: tiling, double-buffering, swizzle. The same bench
  shows int8 at 72-75 TFLOP/s on the same card, so a `mage_flow_t2i_int8`
  checkpoint is the bigger practical win and needs no new code, just the file
- mageflow: Vulkan attention materializes an 856 MB scores plane at 1024x1024 and
  reads it twice (softmax table, then PV), ~31 GB a forward, which is why it is 25%
  of the step against the CUDA arm's 15% on cuDNN's fused kernel. Query banding
  would keep a tile in L2; the blocker is that the coopmat scores shader comes out
  of a SPIR-V builder with no query offset, unlike the f32 `attn_scores` elt kernel
  which already bands
- mageflow: the 64-row TEXT stream is 8% of the Vulkan GEMM time for 1.5% of the
  FLOPs, because m=64 leaves 24 workgroups on 82 SMs. Fusing q/k/v into one GEMM
  per stream triples the occupancy and reads the activation once instead of three
  times; the cost is concatenated weights, ~1.4 GB more VRAM, and dropping the
  zero-copy mapping
- mageflow: the DiT step is 1.22x slower than ComfyUI's (0.755 vs 0.621 s/step at
  1024x1024 cfg 5, interleaved on the same 3090). WHERE is not measured. Ruled out:
  the GEMM arm, since `--backend cuda` and `zig-cuda` time identically because
  `lin_cuda`'s `.bf16` route takes hand-PTX `opGemmBf16` on both. Two candidates for
  an isolation: the separate `opAddBiasRows` pass after every block GEMM, which
  ComfyUI folds into cuBLASLt's epilogue (an `opGemmBf16` that takes a real bias
  would remove ~10 GB/step of traffic, ESTIMATED not measured), and the per-block
  text q/k/v copies. Our VAE is 1.69x FASTER than ComfyUI's on CUDA and another
  3.7x faster again on Vulkan (0.3 s against 1.1 s at 1024x1024, unexplained and
  worth a look from the other direction: what the CUDA VAE is doing that the
  Vulkan one is not), so the codec is not where to look
- mageflow: the VAE ENCODE side is still host-only (`mage_vae.encode`), which costs
  a fixed second or so per reference image on the edit path rather than per step.
  It shares the DiCo block and the 2 head blocks with decode, so the device arm is
  mostly assembled already; what it needs new is the `[3][h][w]` patch embed at
  stride 16 and the affine `LayerNorm2d` the head blocks use. Host on Vulkan too,
  for the same reason
- mageflow: the edit path is wired and pinned, but Mage-Flow-EDIT is a SEPARATE
  checkpoint (`mage_flow_edit_bf16` / `mage_flow_edit_int8_convrot`) that is not
  on this box. The T2I checkpoint runs the edit path correctly and ignores the
  instruction, which is what ComfyUI does with the same file. Nothing to write;
  the file is the gap
- gpu: still per backend, and candidates for `kernels/dual.zig`: CUDA's block-quant decode
  GEMVs (warp-per-row dp4a/f16; Vulkan's are the dual ones), its attention and the GEMMs.
  CUDA keeps its own attention: the dual pair cost it 17% of qwen35 prefill, and 7% once
  the dual attention grouped its query heads (`attn_rows_g*`, CUDA's `attn_split_g` idea;
  Qwen3.5-9B q6_k pp 1578 vs 1694, interleaved, `qwen35_cuda.dual_kernels`). What is left
  of the 7% is unattributed between the attention and the GDN chunk.
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
- lora: only SenseNova and MiniMax H3 have a sidecar path at all, and tp-gui now drives
  LoRAs PER IMAGE (a list change swaps the stack on the live session instead of
  reloading), so the missing family funnels below are what gates the feature for
  everyone else. `Family.supportsLora` is the list the UI asks first; it and
  `Session.attachLoras` must stay in step
- lora: krea2, Z-Image and Anima route their device GEMMs through `lin_cuda` too, so
  each is a `lora` field on the DiT plus a `plan.lora` assignment plus a host funnel;
  they currently REFUSE a `--lora` rather than ignoring it. The SD UNets need conv
  factors (kohya's `[r, in, kh, kw]` plus `lora_mid`) and a text-encoder LoRA
  (`lora_te_` / `lora_te1_` / `lora_te2_`), neither of which `lora.zig` has
- begin filling in holes in the capabilities grid (BACKEND.md)
- add more sampling methods
- preview: the TAESD ladder only has the Wan decoder (`taehv`), so the settings
  "Preview decoder" row appears for Krea2 and Anima alone. Opening it to SD1.5 / SDXL
  (`taesd_decoder.pth`, `taesdxl_decoder.pth`) and Z-Image (`taef1_decoder.pth`) needs
  (a) a reader for PyTorch's zip+pickle `.pth` container, or a `tools/convert_taesd.py`
  that rewrites them as safetensors, and (b) a `taesd.zig` decoder (TAESD is TAEHV
  without the temporal memory blocks; `wan_vae.loadConv` and the taehv CPU/CUDA/Vulkan
  kernels are the parts to reuse) plus a `previewFits` arm per latent format in
  `shared/model_spec.zig`. The catalog and the settings row need nothing else: a file
  that answers `previewFits` for a family is offered for it
- the CPU steppers' `prefill()` is still one un-chunked forward over the whole
  tail, so it has no boundary to stop or re-plan at. Inert today (the GUI is
  CUDA-only and the CLI arms no cancel), but any CPU-backed frontend needs it
- even a tiny bit of offloading of gemma4 31b is extremely slow
- diffusion model weights are "bouncing" during steps, vram-wise
- sdxl `perfectdeliberate_v10` renders SOLID WHITE at 1024^2 on the `cuda` backend and
  correctly on `zig-cuda`; `blackMAGICXL_v145` is fine on both. One arm AND one file, so
  an interaction, and white with no error is the f16 overflow signature. Next probe is
  `sd_vae.Config.act_f16` off for this file, plus a non-finite check on the residual
- cond noise: a trajectory axis (noise varying over the denoising steps; high sigma sets
  composition, low sigma texture). Needs a per-step re-upload on each device session
  (5 families x 2 backends), or `Cond.Schedule` with K buckets at K sessions' worth of
  fused text
- cond noise: multiplicative mode (`x *= 1 + sigma*u`). Stays nearer the manifold but
  gives up the ray property
- cond noise/steer are studio-only (per image), so a render the CHAT model asks for never
  carries them: the knobs are not in `config.host_fields`. Putting them there needs
  `Diffuser.opts` to own the curve and term text, since settings buffers are replaced
  under it. Also unwired: a saved curve library and a shape sparkline
- cond steer: terms whose directions overlap partly cancel where their signs oppose. The
  engine logs the overlap; orthogonalizing terms against each other would fix the
  magnitudes at the cost of order-dependence
- act-derive builds a full `Session`, so it loads the VAE and never decodes: the capture
  stops at the denoiser. A lazy decoder in `Session.init`, or a capture path that skips
  it, saves that load
- act steer moves global attributes and cannot place an object: the capture averages the
  residual over image tokens and the apply adds one vector to every token, so neither end
  carries position. A `moon` direction shifts time of day instead of drawing a moon.
  Per-token placement is the fix and it is a much larger change. The health check is the
  cosine between directions for UNRELATED concepts against a random-direction control:
  near the 1/sqrt(features) floor means the derive is separating concepts, high means it
  has collapsed onto one axis again
- gui: `ui-probe --click=X,Y` injects a press/release and prints what took keyboard
  focus, but nothing runs it as a check. An `--expect-focus` arm plus a build step would
  make it a regression guard; the probe already falls back to SDL's dummy driver
- cond steer: a direction is re-derived on every encode, so a batch of N images pays N
  text encodes per term. A cache on the Session keyed by (text, model) is the fix; the SD
  arm's `empty_ref` is the precedent
- cond noise: the krea2 tap sweep says the DiT leans on the DEEP encoder taps (2.6x the
  effect of the shallow ones at a mirrored budget, BACKEND.md 2K). Turn that into a TE
  quantization budget: quantize taps 0-3 harder than 8-11 and check the render against
  the dense encoder on an INTERMEDIATE (the conditioning), not on an image
- cond noise: text-encoder WEIGHT noise is the other half and a different feature -- it
  would ride the existing block-quant kernels rather than a host buffer, and it answers
  the encoder's quantization budget directly
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
- gemma3 vision: `lmstudio-community/gemma-3-12b-it-GGUF/mmproj-model-f16.gguf` fails to load
  with `ShapeMismatch` on every backend, where the Starshine file's `mmproj-F16.gguf` works.
- llm: two gaps left in reasoning-marker detection (`chat.observeReasoning` plus
  `tp-llm --reasoning-markers`): a fine-tune whose markers are in NEITHER
  `known_reasoning` nor the flag still falls back to the family guess, and tp-gui has
  no override field, so a GUI user cannot answer for such a model
- serve: the privacy hardening the canary suite does NOT cover, in the plan but unbuilt:
  a `Content` type whose formatters print a length (so logging a prompt is a compile
  error rather than a review question), a `-Dprivacy-audit` build refusing `{any}` under
  `serve/`, and a runtime taint check on emitted log lines. What exists instead: the
  sinks are gone from the engine, the two content dumps are compiled out under
  `-Dprivacy`, `serve/privacy_test.zig` walks for a canary with teeth, and
  `serve_main.zig`'s module doc lists every write path, diffed against a real strace run.
  A lost host is respawned once; a second loss is final until restart, by design. The scheduler places at enqueue and never moves a job afterwards, so a host
  that dies with pending work fails it rather than re-placing it; `chat_contention` and
  the cold-start step cost are assumed figures with no measurement behind them. Two
  local hosts share one card with two arbiters that only see each other as "others
  hold": a 31B chat model beside a diffusion pipeline prefilled at 1 tok/s, so the
  second local host is for a second card or a small model until the scheduler weighs
  VRAM. A non-NVIDIA host now reports its card (Vulkan heap budget for VRAM, DRM fdinfo
  for busy time), with two caveats worth knowing: the busy figure is THIS PROCESS's, not
  the whole card's, so another program's work on a shared GPU is invisible; and i915's
  counter overshoots (107% measured), so it is clamped. Remote host gaps: the token sits
  in the settings file in plaintext (a `token_file` per host, or a keychain shim, would move it); and on
  Windows std's TLS client falls back to the OS certificate store when the pinned
  bundle lacks the presented issuer, so there a peer with any OS-trusted certificate
  gets past the pin and is sent the token (tls.zig's client, which has no such
  fallback, is the strict alternative; Linux and macOS pin strictly today). A remote
  host has no `--idle-exit`, and one client at a time is still the rule. Model sync is
  one direction only (client to host) and offers just the image checkpoint, not its
  side files or LoRAs; a host never asks for a file, and nothing removes one
- gui: a transcript does not record the system prompt it was generated under, so
  reloading replays an old conversation under current settings (e.g. with or
  without the image-tool description). Same class as the markers were
- gui: replaying a conversation into a DIFFERENT model feeds the old model's
  reasoning markup in as literal content; real chat templates vary in whether
  they keep prior reasoning at all (Qwen's drops it)
- lora: DoRA over LoKr refuses (no file seen); in a stack, a DoRA's norms read the
  base weight where ComfyUI's merge reads it with the earlier files merged in; a
  denoiser adapter on a convolution refuses (the SD path takes them)
- lora: `textxl` (SDXL, old diffusers `processor.to_k_lora` spelling) loads its UNet
  half; whether its `text_encoder.*.lora_linear_layer` keys reach CLIP is unchecked
  (its conditioning moved by nothing measurable)
- zig-cuda: the hand-PTX int8 GEMM (`opI8GemmDevWs`, `fusedFn`) stores only f32, and
  `lin_cuda.typedOn` widens a 16-bit activation before its prep even though `opI8PrepR`
  reads either width. So a 16-bit stream costs an int8 linear a widen and a cast: krea2
  int8 at f16 storage measured 2-4% slower than f32 on zig-cuda, which is why its stream
  stays f32 there. A 16-bit epilogue store plus a typed prep read would make it pay, on
  every int8 family (the SD UNet's int8 checkpoints too)
- sensenova: the 64x64 canvas in the device tests carries ~25x the 96x40 case's
  device-vs-CPU error at the same depth, on BOTH arms and with the same reference
  magnitude, so it is neither a backend nor a small denominator (the printed control row
  rules that out). It does not move the full-render figure, but nothing explains it. At
  all 42 layers it fails the check's bound (8.2e-2 against 4.4e-2 on cuda, f32), which was
  fitted to 32 layers
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
- cuda: a context that faulted in tp-gui with both models resident (image generated,
  then a chat turn) is now reported once, by `Context.check`, with the kernels the ring
  still holds. The FAULT ITSELF is unattributed: the only capture is aftermath, every
  call in the context already failing with the same sticky code. The next occurrence
  should name a kernel; `CUDA_LAUNCH_BLOCKING=1` pins it to one launch. Suspect the
  paths the handover exercises, LLM promote-at-boundary and the growable KV, not the
  copy that reports it
- hosts: a failed render is now reported and moved to a host that has not had it
  (`Hosts.takeFailure` / `replay`). The move itself is UNCOVERED by a unit test: the
  fake `Remote` these tests build is `undefined` apart from `failed`, so posting to one
  faults, and a slot has to be up to be chosen. The decision is tested directly
  (`placeExcluding` with a skip list, the `Asked` bookkeeping); what is not is that the
  replayed enqueue reaches the wire. A `Remote` that can be stood up with a link that
  answers nothing would close it, and `hosts-probe` is where it would run
- sampler: eleven of ComfyUI's are in (`core/sampler.zig`), over four steppers plus a
  mid-step model callback, each pinned against ComfyUI's own body on both families.
  What is left, roughly by how often anyone picks it: `uni_pc` / `uni_pc_bh2` (its own
  multistep predictor-corrector, the biggest single addition left), `ddim` (needs its
  own eta and an alpha-bar walk rather than a sigma one), `lms` (Adams-Bashforth, whose
  coefficients the reference gets from `scipy.integrate.quad` over a polynomial, so the
  port integrates it in closed form), `res_multistep`, `ipndm` / `ipndm_v`, `deis`,
  `heunpp2`, and the `_cfg_pp` family (which needs the UNCONDITIONAL prediction as well
  as the guided one, i.e. a second output from the denoiser rather than a second call).
  `dpm_fast` / `dpm_adaptive` do not fit at all: they pick their own step count.
  A new one is a `Kind`, an arm on whichever stepper machine it belongs to, and a
  fixture in `tools/gen_sampler_fixtures.py`; the trap is that ComfyUI ships a SEPARATE
  `_RF` body for several of them and dispatches on `CONST`, which a port that reads
  only the eps form gets plausibly wrong on krea2 and Z-Image
- tp-gui fonts: bold CJK runs render regular since the Bold face went (10.5 MB).
  FreeType's synthetic bold (`FT_GlyphSlot_Embolden`, em/24) is a fair stand-in at
  UI sizes, measured in-app, but it is a dvui change: `Font.Cache.getOrCreate`
  knows when bold was asked and only a regular source matched, and the two
  `FT_Load_Char` sites embolden the slot. ~20 lines, prototyped; needs a dvui fork
  or an upstream PR, then fonts.zig routes tier-2 bold as `.bold` again
- tp-gui fonts: the faces embed uncompressed, 21.5 MB. Measured on the 10 MB CJK
  face in ReleaseFast: std zstd inflates at 22 MB/s (0.77 ratio, ~1 s for the set,
  8 s in Debug), std xz at 27 MB/s (0.70), std flate at 190 MB/s but only 0.86, so
  ~3 MB for 110 ms. libzstd does the same frame in 10 ms. `tools/gen_fonts.py
  --pack zstd` emits the frames; wiring them in is worth it only with a libzstd
  dependency, which is a 4.5 MB saving to weigh against a new C library
- MOSS-Image, what support left open:
  1. int8 attention. The checkpoint asks for `comfy_kitchen_int8` per layer, and that
     choice alone moves ComfyUI's render 16 dB (BACKEND.md 2L), so matching ComfyUI as
     shipped needs a quantized QK^T route; it is also most of ComfyUI's speed lead
     (attention is a third of our 1024² forward, `moss-image-cuda-test --prof`).
  2. Speed: 1.17 s/step against ComfyUI's 0.694 at 1024². Measured per forward: GEMMs
     217 ms, attention 165 ms, elementwise 89 ms (464 launches: the `v_div` pair, rmsMod,
     situ, gatedAdd, both qk norms and ropes per block) on the 3090. `situ` could fold
     into the down projection's activation quant as ComfyUI's does.
  3. `er_sde`, the sampler ComfyUI's MOSS templates use. Not in `core/sampler.zig`.
  4. `MOSSImageLayoutPrompt`, ComfyUI's bounding-box layout prompt (a JSON caption with
     per-box captions and colours), for tp-gui.
