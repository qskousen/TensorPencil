- Zig 0.17 port, what is left:
  - Three SPIR-V backend bugs to report upstream (Codeberg), each with a minimal repro: an
    access chain into an array at a comptime-known index is typed as a pointer to the
    array (`spv.fixAccessChainTypes` repairs it; drop the pass once fixed, the
    well-typed-modules test in `gpu/context.zig` says when); a signed-to-unsigned
    `@intCast` emits no conversion, so a `/` or `%` on it is invalid (`convt1dCa` uses
    `@bitCast`); inline-asm `OpDecorate` on a variable is dropped without an error.
  - A native Windows build (redpc has only Zig 0.16): build.zig compiles the SPIR-V
    kernels as Compile steps again, which 0.16 could not do on a Windows host, and a
    Windows test run finds the pinned DLLs through a PATH entry that is relative to the
    build root.
  - `@SpirvType(.runtime_array)` can declare a buffer without the fixed `[1 << 28]f32`
    bound the kernels carry (see `spv.largestArrayBytes`).
  - Vulkan decode on gemma-4 12B q4_0 is within ~1.5% of the 0.16 build (it was 3.5%
    before `dual.zig`'s entry kept its body out of line); per-kernel GPU time is within
    1-2%, at the noise of `TP_VK_PROFILE`. Not chased further.
  - `NoContraction` on the f32 quant GEMVs (needed so a routed row equals its one-row
    GEMV bit for bit under 0.17) costs ~2.3% on that path (`vk-gemv-bench` f32 column).
  - `forEntry` cuts are built from a whole-module copy per kernel at load (5.9 MB
    mmaps, extra kernel time at setup); cutting from one shared parse would save it.
- weight dispatch (`dual_lin`, `lin_cuda` / `lin_gpu`, `conv_cuda` / `conv_gpu`), what is left:
  - the SD LoRA sidecar costs +12-16% per step on SDXL; ~9.5 ms/step of B side on SD1.5
    768² is not yet attributed between its GEMM, the `lo` bf16 pass and launch count.
    LoKr, DoRA and whole diffs, which it refuses, each have an exact runtime form (LoKr's
    kron is two reshaped GEMMs, DoRA a per-row or per-column scale computed at attach)
  - SenseNova's generation check cannot tell rounding-flip drift from a defect past a few
    layers on the 4-token canvas (BACKEND.md 2G). Mage-Flow's split (weight-only walk,
    noise magnitude, per-route emulation; BACKEND.md 2H) is the shape to move it onto.
  - Vulkan's fp8 coop GEMM is built with f16 accumulators unconditionally
    (`coopmat.coop_acc_h16`), while its f16-store sibling is gated on the device offering
    them (`coop_acc_f16`): Intel Alchemist offers f32/s32 only. Untested there.
  - `--vram-budget` on CUDA refuses a 72 MB allocation mid-forward on a 1024² krea2 int8
    render at 6, 10 and 16 GiB, intermittently at 16, with every unevictable entry pinned
    (~190) or an unconsumed prefetch (13): the pin claims leave the activations no room.
  - Vulkan's W8A16 (an explicit `--dit-act f16` / `--te-act f16` on int8) runs ~1.5x a
    W8A8 krea2 step: the f16 coop GEMM is 44-52 TF/s against int8's 77-90 (`vk-wo-bench`).
    The f16/bf16 coop GEMM's own rate (~50 of the card's ~71) is the lever; it is also
    61% of a 1024² Wan decode on Vulkan.
  - Vulkan's fused int8 prep at 25600 (four padded chunks) rotates twice, 242 GB/s on the
    3090 and 160 on the RX 6800 XT; it runs only under an explicit W8A8 on a 32B encoder.
  - `opMatmulIntWoF32` (W8A32) rotates the whole WEIGHT per GEMM through `rotate_fwht`,
    one thread per group, which spills its 256 floats to scratch.
  - Vulkan linears still re-cast and re-pad an operand that is already 16-bit
    (`coopGemmPad`); reading an aligned one directly is open.
  - The f16 weight-only integer routes (`*_wo16`, krea2's text fusion) have no checkpoint
    here that reaches them; only `lin-cuda-test` and `lin-vk-test` run them.
  - These real-model test files are gone from this machine and their tests skip: the
    `krea2CenterSemiraw_v10{Fp8,Int8,Int8-ASYM_W4A8_INT8,Int8-INT4_CONVROT_SR}` DiTs,
    `anim-q4k-calib.gguf`, `lpips_alex`, the two Anima int8/W4A8 conversions,
    `models/assets/minimax_h3_vae.safetensors`.
- LLM on Vulkan, against CUDA (Qwen3-4B q4_k 112 vs 131 tok/s decode):
  - decode is GEMV-bound (74% of GPU time) and the GEMVs match CUDA's back to back; what
    is left is each GEMV's ramp and drain between barriers, so the lever is the step's
    structure: silu_mul folded into the down GEMV (0.13 ms a token), more `independent`
    groups, or a persistent decode kernel. Design it before building it. The attention
    merge is 7 us a call.
  - prefill attention is scalar; a coopmat flash kernel is the lever. A block-quant
    prefill re-expands every weight to f16 per chunk (why the chunk is 2048).
  - every dense weight is cached twice (GEMV and coop layouts): a bf16 4B holds 16 GB.
  - the q8 decode GEMV is wired but off, and `OpSDot` alone would not turn it on
    (MEASURED: q8 trails the f32 route on the k-quants). A q8 kernel decoding in
    16-element units is what could change that.
- the one-thread-per-query `Backend.attn` kernel in its remaining callers (the DiT text
  fusion, anima, the embedders, the H3 VAE and audio encoder, CLIP), most of them
  non-causal, which `opAttnDecode` has no mode for
- CUDA's f16 KV path changes the greedy tokens on Qwen3.5-9B where f32, q8_0 and Vulkan
  f16 do not
- activation precision: Z-Image and Anima default to f32 because an explicit f16
  overflows at step 1. Trace WHERE (residual, a delta or the MLP) and try SenseNova's
  split (`sensenova_dev.streamDiv`) before settling for f32. fp8 as a GEMM-input route
  (like W8A8) on cards with fp8 tensor cores is unbuilt. Receipt still owed for whether
  bf16 is an acceptable SD fallback: SDXL at f32 / f16 / bf16, one seed; if bf16 is
  clearly worse than f16, SD falls back to f32
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
- mageflow: the edit path has never run against Mage-Flow-EDIT (`mage_flow_edit_bf16` /
  `mage_flow_edit_int8_convrot`), which is not on this box
- mageflow: the VAE ENCODE side is still host-only (`mage_vae.encode`), which costs
  a fixed second or so per reference image on the edit path rather than per step.
  It shares the DiCo block and the 2 head blocks with decode, so the device arm is
  mostly assembled already; what it needs new is the `[3][h][w]` patch embed at
  stride 16 and the affine `LayerNorm2d` the head blocks use. Host on Vulkan too,
  for the same reason
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
- begin filling in holes in the capabilities grid (BACKEND.md)
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
- serve: the privacy hardening the canary suite does NOT cover: a `Content` type whose
  formatters print a length (so logging a prompt is a compile error rather than a review
  question), a `-Dprivacy-audit` build refusing `{any}` under `serve/`, and a runtime
  taint check on emitted log lines
- hosts: the scheduler places at enqueue and never moves a job afterwards, so a host that
  dies with pending work fails it rather than re-placing it. `chat_contention` and the
  cold-start step cost are assumed figures with no measurement behind them
- hosts: two local hosts share one card with two arbiters that only see each other as
  "others hold" (a 31B chat model beside a diffusion pipeline prefilled at 1 tok/s), so a
  second local host is for a second card or a small model until the scheduler weighs VRAM
- hosts: a non-NVIDIA host's busy figure is THIS PROCESS's, not the whole card's, so another
  program's work on a shared GPU is invisible; i915's counter overshoots (107%) and is clamped
- remote hosts: the token sits in the settings file in plaintext (a `token_file` per host,
  or a keychain shim, would move it); no `--idle-exit`; one client at a time; nothing
  removes a model file from a host
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
- tp-gui chat runs every arch on Vulkan. There is no CPU arm: `chat.zig` refuses `.cpu`. A
  new stepper joins `Arch` only once `llm/stepper.zig`'s ledger line for it is empty.
- k2-horizon on Vulkan decodes at 85% of CUDA (BACKEND.md 9). What is left is GPU kernel
  time, the q6_k `_sdot` GEMVs below half the card's bandwidth at its shapes; `gemvx`
  reading each weight row once for several of an expert's rows would let device routing
  take batches past 64.
- `k2.HostCost` compares a 128-row CPU pass against one unoverlapped stage; CUDA's staging
  overlaps compute at large batches, so between ~100 and ~200 rows it may pick the CPU
  where staging was cheaper. Unmeasured.
- Vulkan fp8 on a device that flushes f16 subnormals (`fp8_wide`) expands every weight to
  f16 per GEMM and runs without krea2's staged paths: 3.51 s a forward on the RX 6800 XT,
  0.97 on the 3090 forced, against 0.61 for the folded GEMM there. The fp8 coop kernel with
  a device-screened warp tile (its tile is fixed at 64x64, and with f32 accumulators it was
  4x slower than the f32 GEMM on that card) would read 1-byte weights and skip the expansion.
- sensenova: int4 convrot is the FORMAT, not the file or the reader (ggufy's bytes equal
  comfy_kitchen's; our GEMM equals its decode to 2e-6). Two separate costs: the WEIGHTS of
  the gen copy's `mlp_mot_gen.gate_proj` / `up_proj` (ggufy rule file keeping them and
  `mlp.down_proj` at int8: `sensenova_u15_int4_convrot_mlp8`, 13.0 GB, velocity 0.043 vs
  0.22, clean weight-only and on Vulkan), and CUDA's 4-bit ACTIVATIONS, which still grain
  that file under `auto`. CUDA has no int4-weight int8-activation route; Vulkan already
  runs one (decode to int8) and renders it clean. Conditioning stays 0.55 from the base
  copy's int4 roles.
- sensenova: `sensenova_u15_asym_w4a8_int8` renders washed-out WHITE on CUDA and Vulkan
  (512², 28 steps, cfg 4, seed 42) with no non-finite warning, though `quant-attrib`
  (8 steps, cfg 1) puts its velocity error at 0.053 and its conditioning at 0.88.
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
- vulkan decode: Qwen3.5-9B Q6_K gains 2% from the `OpSDot` GEMVs on the RX 6800 XT where
  its q6_k kernel measures 1.44x; profile what else a token spends there (its GDN layers,
  attention). The q4_0 `_sdot` GEMV still trails CUDA's dp4a one by ~25% at the large shapes.
- vulkan barriers: synchronization validation is clean on AMD (BACKEND.md 5); the NVIDIA-only
  paths are unvalidated, since the layer is not installed on the 3090 box. The levers it
  measured: write-after-write on reused scratch is 21% of a decode's barriers
  (alternate the buffers in the model), and binding extents would clear the 10% that fall
  on another offset, once every GEMM's padded write is known to stay inside its view.
- windows: libvips and the libav set have to come from somewhere with `.pc` files (the
  release workflow's MSYS2 step). The gated suite on the Windows box intermittently
  ends "test runner failed to respond for 1m" after every test passed, only seen on runs
  that recompiled; cause unconfirmed
- macos: no Metal backend, so a Mac is CPU-only, which the README's own numbers put at
  ~290 s/step. The Vulkan arm would reach it through MoltenVK (`openVulkanLib` already
  names the dylibs) but coopmat is unlikely to be there, so the non-coop GEMM path is
  what would have to carry it
- cuda: a context faulted in tp-gui with both models resident (image generated, then a
  chat turn). `Context.check` reports it with the kernels the ring holds, but the FAULT
  ITSELF is unattributed: the only capture is aftermath, every
  call in the context already failing with the same sticky code. The next occurrence
  should name a kernel; `CUDA_LAUNCH_BLOCKING=1` pins it to one launch. Suspect the
  paths the handover exercises, LLM promote-at-boundary and the growable KV, not the
  copy that reports it
- hosts: moving a failed render to a host that has not had it (`Hosts.takeFailure` /
  `replay`) is UNCOVERED by a unit test: the
  fake `Remote` these tests build is `undefined` apart from `failed`, so posting to one
  faults, and a slot has to be up to be chosen. The decision is tested directly
  (`placeExcluding` with a skip list, the `Asked` bookkeeping); what is not is that the
  replayed enqueue reaches the wire. A `Remote` that can be stood up with a link that
  answers nothing would close it, and `hosts-probe` is where it would run
- sampler: ComfyUI's samplers not yet in `core/sampler.zig`, roughly by how often anyone
  picks one: `uni_pc` / `uni_pc_bh2` (its own
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
- tp-gui fonts: bold CJK runs render regular (no Bold face embedded).
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
  5. `effectiveAct(.encoder)` still answers f32 for MOSS: the Qwen3.5 encode runs on the
     device (`runQwen35`) at whatever `CudaLM`/`VulkanLM` store, which was never read off.
     The PNG's activation field says f32 regardless.
- Qwen-Image 2.1, what support left open:
  1. Reference editing. The CPU DiT takes `Ref`s spliced at a slot (the toy fixture pins
     it); missing are the 2.2-layout VAE ENCODER (alpha padded at 1.0 as ComfyUI does),
     the 8B encoder's vision tower presenting the reference to the text, the slot syntax
     in the prompt, and the device arms' image prefix parts beyond the rect call.
  2. The f16 DiT stream measured slower than f32 in one run (2.15 against 1.38 s/step,
     int8, CUDA, shared card). An interleaved same-binary A/B decides the default; a
     profile first, since "flat or slower f16" has been dispatcher staging before.
  3. The VAE at f32 only: f16 overflows it and bf16 is 2-3e-2. Find which tensor
     outgrows f16 (an exact power-of-two prescale as elsewhere) before calling it.
  4. The prefix's text runs on Vulkan go through the per-thread `attn_causal_batched`.
     Fine at a prompt's length and run once per conditioning; long prompts or image
     prefixes want the tensor-core path with a causal row bound.
