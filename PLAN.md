# TensorPencil — Krea 2 Inference Plan

## Status (updated as milestones land)

- [x] M1 foundations: dtype (fp8-e4m3 LUT verified vs torch), tensor, mmap safetensors loader
- [x] M2 ops: fp8-dequant GEMM (~270 GFLOP/s), streaming GQA attention, RoPE (flux + rotate-half), RMSNorm, activations — all torch-fixture tested
- [x] M3 tokenizer: Qwen2 byte-level BPE, exact parity with transformers
- [x] M4 VAE decoder + PNG: parity 1e-6 vs ComfyUI (T=1 collapses causal 3D convs to 2D)
- [x] M5 Qwen3-VL text encoder: 12-layer tap conditioning, parity ~1e-6 relative
- [x] M6 DiT: full SingleStreamDiT forward, parity max_err 4e-4 on ±4.5 outputs (slow test gated on `testdata/slow-tests`)
- [x] M7 sampler + pipeline + `generate` CLI — first end-to-end image 2026-07-01 (256px/4 steps ≈ 85 s on CPU)
- [x] M8 CPU performance pass: packed outer-product GEMM (270 GFLOP/s -> 1.2-1.7 TFLOP/s),
      SIMD exp for softmax/activations. 256px: 20s -> 7.3s/step; 512px 8-step image in ~4.5 min.
- [x] M9 Vulkan backend (infrastructure): pure-Zig loader bindings (`src/gpu/vk.zig`,
      machine-verified against vulkan_core.h), compute context, Zig-authored SPIR-V kernels
      (fp8/f32 GEMM, naive + shared-memory tiled), and a load-time SPIR-V sanitizer
      (`src/gpu/spv.zig`) that makes the 0.16 backend's output spirv-val-clean
      (LocalSize splice, logical addressing, workgroup ArrayStride strip, decoration dedupe).
      Numerics exact vs CPU. `gpu-test` / `bench-matmul` CLI diagnostics; GPU unit test gated
      on `testdata/gpu-tests`.
- [x] M9 GPU speedup on NVIDIA via shared-memory-free kernels: weights transposed to k-major
      at upload (coalesced without workgroup storage), 8x8 register tile per thread, k
      unrolled x4, branchless fp8 decode, device-local activations with explicit staging
      copies. 1.74 TFLOP/s on the 3090 (vs 1.2-1.4 CPU), bit-exact, correct on
      NVIDIA/RADV/llvmpipe. `generate --gpu on` routes large f8/f32 GEMMs
      (`ops.matmul.gpu_min_flops` threshold); DiT weight transpose+upload amortizes into the
      first step (~20 s for the 12.4 GiB model). 256px: 7.3 -> 5.1 s/step; 512px:
      32.3 -> ~28 s/step (attention stays on CPU and grows quadratically — see follow-ups).
- [x] M9 GPU-resident DiT forward: transpose moved onto the GPU (model load ~20 s -> ~1.7 s,
      straight DMA + device transpose kernel), and the whole 28-block chain now runs on the
      device (eltwise/norm/RoPE/attention kernels in src/gpu/kernels/eltwise.zig; orchestration
      in src/models/dit_gpu.zig; one upload in, one download out per step). Parity vs ComfyUI
      fixture: max_err 3e-4. Step times with --gpu on: 256px 1.7 s (CPU 7.3), 512px 6.1 s
      (CPU 32.3) — 4-5x. Sync-per-op submissions still cost ~100 ms/step; batching into one
      command buffer per block/step is the next easy win.
- [x] M9 attention GEMM-ification: profile-driven (`generate --profile on`; sync-per-op makes
      host timing exact). At 1024px attention was 26 s of a 42 s step; replaced the streaming
      kernel with attn_scores/softmax_rows/attn_out (8x8 register tiles, scores buffer batched
      over head groups to cap VRAM at ~2 GiB). 1024px: 41.8 -> 30.9 s/step (matmul 14.3 s
      @ ~7 TFLOP/s, attn 15.7 s, rest <1 s); 512px: 6.1 -> 5.0 s/step. Parity 2.9e-4.
- [x] Attention overhaul round 2 (profile-driven): k-major per-head gathers made the
      scores/out GEMM kernels coalesced (~1 s each at 1024px), and the sub-profile exposed
      the real culprit — the row-softmax kernel (12.7 s, one latency-bound thread per row).
      Deleted it entirely by fusing ONLINE softmax into attn_out (scores stay raw; running
      max/denominator in registers). Attention 15 -> 2.1 s at 1024px. Step times:
      1024px 30.9 -> 16.6 s, 512px 5.0 -> 4.2 s. Parity 3e-4 throughout.
- [x] Tensor cores via VK_KHR_cooperative_matrix: hand-assembled SPIR-V GEMM
      (src/gpu/coopmat.zig — a word-level Asm emitter; spirv-val-clean). f16 A/B with f32
      accumulate; weights decode fp8->f16 on device per call (scale folded), activations
      f32->f16 with zero row padding to multiples of 64. Two variants: buildGemm (naive,
      one 16x16 tile/workgroup — bandwidth-bound, no faster than f32) and buildGemmShared
      (4 subgroups, 64x64 tile, A/B slabs staged through WORKGROUP MEMORY — which works,
      proving the NVIDIA fault is specific to Zig-backend-emitted modules, not workgroup
      storage itself). GEMMs 14.4 -> 6.3 s at 1024px (~16 TFLOP/s effective incl. decode).
      Parity 1.5e-3 (f16 rounding; same regime ComfyUI runs). Step times: 1024px 9.2 s,
      512px 1.8 s, full 1024 image ~75 s end to end.
- [x] Coop GEMM round 2 (profile + bench-driven; `bench-matmul` now times the coop path
      at DiT shapes): fp8->f16 decode fused into the staging loop (B is read as raw
      e4m3 u32 words and SWAR-decoded — no separate decode pass, no w_h16 scratch, the
      dequant scale rides on the f32->f16 activation conversion), 128x128 workgroup
      tile over a 64-deep k slab (2 barriers per 64 k), A tiles cooperative-loaded
      straight from global, phi-carried SSA accumulators. Each change measured neutral
      -to-small: ALL variants floor at ~23-25 TF/s on 4224x6144x6144 — not bandwidth
      (wider tiles: no change), not decode ALU (SWAR: no change), not acc spills (phi:
      no change); the ceiling is MMA issue density / sync structure. 1024px matmul
      4.7 -> 4.4 s/step.
- [x] Batched submission infrastructure: Context.beginBatch/endBatch record every op
      into one command buffer (per-op descriptor-set ring of 1024, global compute
      barrier between dispatches, flush cap 512); immediate work (weight upload/
      transpose, transfers) runs on a second command buffer mid-recording; uploads/
      downloads and device-buffer growth flush the pending batch first (growing x_h16
      mid-batch destroyed a buffer still referenced by recorded dispatches -> NVRM
      Xid 109 DEVICE_LOST; fixed). Net perf ~neutral at current sizes — measured
      per-op fence overhead was small, contrary to the earlier estimate — but it is
      the foundation for async pipelining and keeps the queue fed.
- [x] Attention scores on tensor cores (coopmat.buildGemmScores): straight-line coop
      kernel, no workgroup memory, gid.z = head, GQA via a group push param; Q
      converts to f16 with the softmax scale prefolded (f32_to_h16), K gathers to a
      per-head k-major f16 block padded to seq_pad columns (gather_kmajor_h16);
      attn_out gained stride push params so the f32 path stays as the non-coop
      fallback. Scores 0.95 -> 0.38 s at 1024px; kernel verified to 5e-3 against an
      f16-rounded reference. Q/K in f16 costs DiT-fixture parity 1.5e-3 -> 3.8%
      (28-block accumulation; same regime as ComfyUI's fp16 SDPA — same-seed images
      vs the f32 path are visually identical, mean pixel delta 0.9%), test threshold
      raised to 5% with rationale in the test.
      Step times: 1024px 7.6 -> 6.7 s (matmul 4.4, attn 1.5, elt 0.44, xfer 0.18,
      cpu 0.18); 512px ~1.8 s (matmul-bound, unchanged).
- [x] attn_out on tensor cores: two-pass softmax (softmax_partial/softmax_combine
      eltwise kernels -> per-row {max, 1/denom}) + coopmat.buildGemmAttnOut, which
      computes P = exp(S-m)/d from the raw f32 scores during A staging (GLSL.std.450
      Exp) and runs P@V on cooperative matrices against a zero-padded f16 V. Two
      hard-won lessons baked into the kernels: (1) padded-j columns hold S = 0, and
      with a negative row max exp(0-m)/d overflows f16 -> Inf * V(0) = NaN poisoning
      whole rows — P is clamped to 0 for j >= seq (and the parity test is now
      NaN-robust: @max silently drops NaN); (2) per-thread reduction chunks must be
      INTERLEAVED (j = chunk, chunk+32, ...) — contiguous chunks put warp lanes 2 KB
      apart and cost ~30x bandwidth (2.4 s -> 0.15 s for the softmax pass).
      Attention at 1024px: 1.46 -> 1.0 s (scores 0.38, softmax 0.15, P@V 0.49,
      was 1.05 for the old fused-online-softmax kernel). Isolated P@V unit test
      vs f16-rounded reference; DiT parity unchanged (0.173, within the 5% gate).
- [x] Coop GEMM double buffering (two 32-deep k sub-slabs, next-slab loads issued
      before the MMA section): measured NEUTRAL — min 13.6 ms vs 12.7-14.1 across
      all prior variants on 4224x6144x6144. Kept (same perf, and the staging
      structure is what buildGemmAttnOut reuses). Conclusion stands: the ~23-25
      TF/s ceiling is MMA issue density on this driver, not latency/bandwidth.
- [x] Eltwise pass: dim-6144 rmsnorm was one-thread-per-row (latency-bound; a few
      thousand serial threads) -> rms_partial (32 interleaved chunks/row) +
      rms_combine + rms_apply_mod, with the norm weight prefolded into modulation
      slots 0/3 on the CPU (mv now stores (1+scale)*w). silu/sigmoid gates fused
      into the f16 conversions feeding mlp.down / wo (silu_mul_h16,
      sigmoid_mul_h16 + opMatmulCoopH16 — skips a full f32 round trip). elt 0.40
      -> 0.26 s at 1024px.
      Step times after all three: 1024px 6.7 -> 5.9 s (matmul 4.2, attn 1.0, elt
      0.26, xfer 0.18, cpu 0.18); 512px 1.8 -> 1.6 s.
- [x] 1:1 ComfyUI comparison (metadata pulled from a real ComfyUI output PNG:
      1122x1683 -> 1120x1680, 20 steps, euler/simple, cfg 1.0, seed 80085, LLM-
      enhanced 311-token prompt; the prompt text lives in the PNG's iTXt
      `parameters` chunk). Speed: 14.2-14.5 s/step steady state, ~330 s wall
      including model loads / ~315 s excluding, vs ComfyUI's ~85 s outside model
      loading => ~3.7x at this size (seq ~7661 is even more GEMM-dominated than
      1 MP). The comparison pair in testdata/ was later re-rendered from a 
      different reference image (see working notes for the current files); numbers
      in this entry are from the original seed-80085 run.
- [x] Torch-bit-exact noise (src/torch_rng.zig): with the RNG divergence
      identified as the entire compositional difference (everything else was
      measured at <=1e-3), fillNoise now reproduces torch.randn's CPU path
      bit-for-bit — MT19937, 24-bit uniforms, and the AVX2 Cephes Box-Muller
      from avx_mathfun.h including the shipping wheel's exact fma-contraction
      choices (recovered empirically against fixtures: the build fuses each
      pending multiply into its consuming add — the cos polynomial's final
      mul becomes an fmsub, and log's y*z fuses into the e*q1 add). Verified
      bit-for-bit on the full 470400-element comparison latent (fixtures via
      tools/dump_randn.py; Intel hosts take an MKL path torch-side that would
      NOT match). Result: a ComfyUI seed now renders ComfyUI's image — same
      composition, ~0.6-2% mean pixel delta depending on the image
      (texture-level drift from 20 steps of fp16 kernel-order differences;
      bit-identical pixels would require matching cuBLAS/FlashAttention
      reduction orders and is not a meaningful target). Current testdata pair:
      seed 252469767172722, 0.61% mean delta — also confirms the 64-bit
      seeding path, the pinned fixtures all used small seeds.
- [x] Warp retiling — the 23-25 TF/s plateau explained and broken. The ceiling
      was the MMA-per-fragment-load ratio: 16-row warp tiles issue ~1 fragment
      load (ldmatrix / global coop load) per MMA, and the load pipe saturates
      long before the tensor pipe. Re-tiling each warp to a 2x2 grid of 64x64
      tiles (4 A x 4 B fragments -> 16 MMAs, ratio 2.0) took the fp8 GEMM from
      13.3 ms to 8.2 ms on 4224x6144x6144 (38.8 TF/s); the same change to the
      scores and P@V kernels (where fragment loads are GLOBAL, so it matters
      double) gave scores 0.38 -> 0.25 s and P@V 0.49 -> 0.40 s at 1024px.
      Also tried and REJECTED with measurements: f16 accumulators (zero gain —
      the driver appears to lower f16-acc coopmat to the same HMMA path;
      reverted to f32 accs, which are better numerics anyway). Parity
      bit-identical throughout (accumulation order per output unchanged).
      Step times: 1024px 5.9 -> 4.2 s (matmul 2.96, attn 0.80, elt 0.34);
      512px 1.6 -> 1.2 s; 1120x1680 14.3 -> 10.9 s (ComfyUI gap 3.7x -> 2.9x).
- [x] Session cache + device-resident final layer: dit_gpu.Session computes
      once per run what the forward used to redo every step — the text-fusion
      tokens (a full CPU transformer pass, expensive at long prompts), the
      rope table (uploaded once, reused), and the timestep vectors for every
      schedule sigma. The final layer (modulated rmsnorm + 6144->64 linear)
      now runs on the GPU (reusing the rms_* kernels with the norm weight
      folded into the final modulation, and opMatmul with an x-offset), so
      the per-step download shrinks from the full hidden image rows (~100 MB
      at 1 MP) to the 64-channel patch tokens (~1 MB). Per-step xfer 175 ->
      38 ms, cpu 180 -> 1 ms. Step times: 1024px 4.2 -> 4.0 s, 512px 1.2 ->
      0.9 s, 1120x1680 10.9 -> 8.8 s (long prompts benefited most).
      Also measured and rejected this round: f16 coopmat accumulators AGAIN
      at ratio 2.0 (still neutral — the driver lowers f16-acc to the same
      HMMA rate; they only save registers), and 64x128 warp tiles (32 f32
      master accumulators = 256 regs/thread, guaranteed spill; f16 masters
      would blow the parity budget).
- [x] Scores in f16: the scores kernel converts its accumulators and stores S
      half-precision; softmax_partial reads interleaved f16-pair words and the
      P@V staging decodes pairs (per-row m/1/d loads amortized per word). This
      halves every byte S moves (one write + two reads per step). Attention
      0.80 -> 0.59 s at 1024px (P@V 0.40 -> 0.21, softmax 0.16 -> 0.10);
      parity 0.1734 -> 0.1752 (noise-level). uvec4 128-bit staging loads for
      the fp8 GEMM measured NEUTRAL (kept — fewer instructions, same speed);
      a true single-pass softmax remains impossible without flash-style
      rescaling of opaque coop accumulators, so the two-pass + f16-S shape is
      the settled design. Step times: 1024px ~4.0 s, 512px ~0.9-1.0 s,
      1120x1680 8.8 -> 8.2 s.
- [x] A staged through shared (the untried structural variant — and the win):
      buildGemmShared now stages BOTH operands through one workgroup f16
      array (B decoded into [0,8192), A uvec4-copied into [8192,16384); 32 KB
      per wg, still 3 wgs/SM), double-buffered with the same 2-barriers-per-
      64k structure; all fragment loads are now ldmatrix-from-shared. Binding
      0 was repurposed as a uvec4 view of the f16 activations (the old u32 B
      view was dead). fp8 GEMM 8.2 -> 6.2 ms on 4224x6144x6144 (38.8 -> 51.7
      TF/s, ~73% of the 3090's f32-acc tensor peak) — so the old "MMA issue
      density" ceiling was really the direct-global A fragment loads. Also
      tried and REJECTED: A row stride 34 to spread shared banks (neutral vs
      32 — the driver's shared coop loads evidently swizzle already; the
      stride stays parameterized in buildGemmShared as documentation).
      Parity: gpu matmul unit test + DiT fixture unchanged (0.17520).
      Step times: 1024px 4.0 -> 2.8 s (matmul 1.98, attn 0.52, elt 0.27),
      512px ~0.95 -> 0.7 s, 1120x1680 8.2 -> ~5.85 s (ComfyUI gap ~1.55x).
- [x] VAE decode on GPU (src/models/vae_gpu.zig): 30.6 -> 2.5 s at
      1120x1680. Went simpler than the planned coop-GEMM route — the
      existing f32 register-tile GEMM (opMatmul: any shape, bias built in,
      weights auto-transposed/cached; the packed [co][9ci] conv layout is
      already its expected [rows][cols]) plus two new eltwise kernels:
      `im2col` (banded f32 patch builder with an upsample flag that fuses
      the nearest-exact 2x resample into the gather — the 1.4 GB upsampled
      intermediate never exists) and `vae_norm` (per-position channel L2
      norm with optional fused silu). eltwise FBuf bound raised to 1<<28
      (5.4 MP activations are ~180M f32). Mid-block attention profiled at
      17.9 s of the first cut's 20.4 s -> moved to tensor cores:
      buildGemmScores parameterized by head_dim (DiT 128 / VAE 384
      pipelines), and P@V runs the 384-wide single head as three fake
      128-column heads sharing one scores/MD plane (buildGemmAttnOut's MD
      plane stride became the f1 push word; S plane stride 0). Attention
      0.09 s on GPU. Parity: vae_gpu decode vs ComfyUI fixture max_err
      2.1e-4 (f16 attention rounding; CPU path is 4e-6), gpu-gated conv
      unit test (plain + fused-2x) vs CPU reference. CPU attention core
      kept as fallback when coop pipelines are absent.
      Full 1120x1680 20-step image: ~127 s wall including all model loads
      (~121 s excluding; ComfyUI ~85 s excl. loads => overall gap ~1.4x).
- [x] Attention round 3 (profile at 1120x1680: matmul 3.64, attn 1.77
      (scores 0.86 / smax 0.27 / out 0.63), elt 0.54):
      * Scores store coalescing: the scores kernel's direct cooperative
        stores scattered 32 B tile rows across the s_stride and measured
        ~184 GB/s; the 128x128 f16 output tile now bounces through a 32 KB
        workgroup slab and copies out as coalesced u32 words (S rebound at
        binding 3 as a u32 view). scores 0.86 -> ~0.62 s; parity
        bit-identical. Applies to both head_dim variants.
      * Flash attention BUILT AND REJECTED at DiT sizes: two-pass recompute
        (coopmat.buildFlashMd/buildFlashOut, hd 128 — pass 1 computes
        {m, 1/d} with S only ever in a col-major shared tile; pass 2
        recomputes S and does P@V; MD rides in attn_d's tail since K/Q/OUT/V
        exhaust the bindings). Correct (unit test vs CPU reference in
        context.zig) but attn measured 1.49 -> 2.43 s: the out pass
        recomputes S at the global-fragment-load rate (~0.8 s), more than
        the S write+reads it saves (~0.5 s); Q-resident-in-shared made it
        worse still (48 KB -> 2 wgs/SM). Flash only wins here if the
        S-compute gets staged operands, which don't fit 48 KB next to the
        S tile at this tiling (a 64-q retiling with staged K+Q at 40 KB is
        the open idea). Kept behind `use_flash = false` in dit_gpu.zig.
      * Conversion dedupe: all Krea 2 DiT fp8 weights are scale-free, so
        the modulated norm now converts once to f16 (new fused
        rms_apply_mod_h16 kernel — no f32 intermediate round trip) feeding
        all four attention GEMMs, and once for mlp gate+up: 6 conversions
        per block -> 2 (~30 GB/step less traffic). Measured neutral within
        the box's thermal noise band (falls back per-block if scales ever
        differ). Parity 0.17520 unchanged.
      Net: category-level scores win (~0.2 s/step) is real but end-of-day
      wall clocks are thermally confounded (same 2m07s for the 20-step
      1120x1680 run, steps 5.9-6.1 s hot vs 5.85 cold). Re-baseline cold
      before the next perf round.
- [ ] NEXT — gap assessment vs ComfyUI (as of 2026-07-02): ~121 s vs ~85 s
      excl. loads at 1120x1680/20 steps = 1.42x. Per-step decomposition of
      the gap: GEMM +0.3-0.7 s (51.7 TF/s vs cuBLAS f16 ~55-65 effective),
      attention +0.5-0.7 s (two-pass-over-S vs FlashAttention-2), eltwise
      +0.2-0.3 s (less fusion). Realistic target with the levers below is
      ~1.2-1.3x; FULL parity is likely blocked structurally: Vulkan on
      NVIDIA caps workgroup shared memory at 48 KB (verified: vulkaninfo
      reports maxComputeSharedMemorySize = 49152 on this driver, no opt-in
      exists) where CUDA allows 99 KB/block via dynamic opt-in on the same
      GA102 silicon (cc 8.6, NVIDIA Ampere tuning guide) — exactly what
      killed the flash tiling — and the driver's opaque coopmat lowering
      leaves no cp.async/
      SASS-level control for the last ~15% of GEMM efficiency (f16
      accumulators would buy occupancy but cost parity). First action next
      perf round: re-baseline on a COLD box — this session's wall clocks
      drifted ~5% warm (see working notes).
      Remaining levers, in order:
      1. GEMM residual (~27% off tensor peak): occupancy — 4-warp wgs at
         ~165 regs run ~12 warps/SM; try 8-warp wgs = 2x4 warp grid of
         64x64 over a 128x256 wg tile (rows%256 holds for all coop-path
         weights: 6144/16384/1536). B slab doubles to 32 KB so the total
         with the A slab is 48 KB = the device max; 2 wgs/SM needs regs
         <= 128/thread — likely requires f16 accumulators (parity risk).
      2. Attention residual (~1.5 s at 1120x1680): the S compute runs at
         ~1 TF/s because Q/K cooperative fragment loads are global with
         12-15 KB row strides. Ideas: stage the K block (16 KB) in shared
         inside the scores kernel next to the 32 KB S bounce (exactly 48
         KB), or revisit flash with a 64-q workgroup (Q 16 KB + K 16 KB +
         S 8 KB = 40 KB, all fragment loads ldmatrix).
      3. VAE decode residual (~2.5 s): the f32 GEMM runs the ~11 TFLOP of
         convs at ~7 TF/s; an f16-B variant of the coop kernel (B staging
         becomes a plain f16 copy like the A slab) would cut decode to
         ~1 s.
      4. Cleanup candidates (no perf impact): coopmat.buildGemm (naive
         single-tile reference) and the eltwise kernels `modulate`,
         `attention`, `softmax_rows` are dead code now; delete or keep as
         documented references. Also: `zig build test` prints a spurious
         "failed command ... --listen=-" stderr line while the Build
         Summary reports all steps succeeded — pre-existing test-runner
         quirk (binary passes when run directly), not a test failure.
- [ ] M9 follow-ups:
      * Blocked upstream: NVIDIA 580 faults (DEVICE_LOST) on any Zig-emitted kernel using
        workgroup memory — even spirv-val-clean at workgroup size 1x1; RADV/llvmpipe run the
        same modules fine. Hand-ASSEMBLED workgroup kernels (coopmat.zig) work fine, so this
        blocks only Zig-emitted ones (eltwise.zig stays workgroup-free). Retest on newer
        Zig / file upstream with the minimal repro.
      * Perf-lab notes (this box): the GPU clock governor idles at ~500 MHz and only
        boosts under sustained queue pressure, so single-dispatch timings swing +-40%
        — compare variants by the min over >=8 reps (`bench-matmul`), or profile whole
        steps. A desktop session shares the 3090; long single submissions trip the
        preemption watchdog (NVRM Xid 109), hence the 512-dispatch batch flush cap.

Working notes for the perf effort (how to measure):
- `zig build -Doptimize=ReleaseFast`, then `zig-out/bin/TensorPencil bench-matmul`
  for GEMM variants (coop section, DiT shapes, take the min) and
  `generate ... --gpu on --profile on` for per-category step breakdowns
  (profile forces sync-per-op so host timing is exact; batched step times
  come from a normal `--profile off` run's step lines).
- GPU-gated tests: `touch testdata/gpu-tests && zig build test`, remove the
  marker after. DiT parity gate is 5% (currently 0.175) — rationale in the
  test; the parity check is NaN-robust on purpose.
- The ComfyUI 1:1 comparison: prompt text in the reference PNG's iTXt
  `parameters` chunk (see testdata/comfyui_ref_252469767172722_1120x1680.png),
  settings 1120x1680 (ComfyUI rounds the requested 1122x1683), 20 steps,
  euler/simple, cfg 1.0, seed 252469767172722; our seed reproduces ComfyUI's
  exact noise (src/torch_rng.zig), current pair 0.61% mean pixel delta.
  ComfyUI reference: ~85 s end-to-end excl. model load
  (~3.75 s/step-equivalent, measured on the original seed-80085 run).

Reference-fixture dumps live in `tools/dump_*.py` (run with ComfyUI's venv; note:
`~/genai/comfyui/nvenv` is the runtime venv with comfy_kitchen/comfy_aimdo, `venv` needs stubs).
Krea 2 turbo defaults: 8 steps, cfg 1.0, euler, "simple" scheduler, shift 1.15.

## Context

TensorPencil is a pure-Zig (0.16.0, no C deps) text-to-image inference engine. This plan is laser-focused on one goal: **run Krea 2 end-to-end on the CPU**, producing a correct image from a text prompt, then move to a **Vulkan backend** for usable speed. The repo is currently the stock `zig init` scaffold; everything below is greenfield.

The three checkpoints are already in `models/`:

| File | What it is | Size | dtypes |
|---|---|---|---|
| `diffusion_model/krea2CenterSemiraw_v10Fp8.safetensors` | Krea 2 DiT (~12.3B params) | 11.9 GiB | F8_E4M3 (raw, no scales) |
| `text_encoders/qwen3VLInstruct4bHeretic_v10.safetensors` | Qwen3-VL-4B Instruct | 4.9 GiB | F8_E4M3 + F32 `weight_scale` per tensor + BF16 norms/embeddings |
| `vae/krea2RealVae_v10.safetensors` | Wan 2.1-style 3D causal VAE | 484 MiB | F32 |

Reference implementations: ComfyUI at `~/genai/comfyui` (has a native Krea 2 implementation — see file map below) and `~/genai/ggufy` (Zig 0.16 safetensors parsing patterns we can adapt).

Hardware: Ryzen 9800X3D (8 cores, AVX-512), 64 GB RAM, RTX 3090 (Vulkan 1.4) for phase 2.

---

## Architecture spec (extracted from ComfyUI source + checkpoint headers)

ComfyUI reference files (read these when implementing each piece):
- DiT: `comfy/ldm/krea2/model.py` (`SingleStreamDiT`)
- Text encoder: `comfy/text_encoders/krea2.py`, `comfy/text_encoders/qwen3vl.py`
- RoPE/timestep helpers: `comfy/ldm/flux/layers.py`, `comfy/ldm/flux/math.py`
- Sampling: `comfy/model_sampling.py` (`ModelSamplingFlux`), `comfy/supported_models.py:1823` (shift=1.15)
- VAE: `comfy/ldm/wan/vae.py`, `comfy/latent_formats.py:493` (`Wan21`)

### Pipeline

```
prompt ──Qwen tokenizer──▶ Qwen3-VL-4B (text-only) ──12-layer tap──▶ (seq, 12, 2560)
                                                                          │
noise (16ch latent) ──▶ Euler flow-matching loop over SingleStreamDiT ◀───┘  (CFG: 2 passes/step)
                                    │
                        denoised latent (16, H/8, W/8)
                                    │
                        per-channel de-norm ──▶ Wan 2.1 VAE decode ──▶ RGB ──▶ PNG
```

### DiT — `SingleStreamDiT` (checkpoint prefix: bare / `blocks.N.*`)

- hidden 6144, **28 identical single-stream blocks** (no FLUX double/single split), patch 2, 16 latent channels → input proj `first: Linear(64→6144, bias)`; output `last.linear: Linear(6144→64, bias)`.
- Attention: separate `wq/wk/wv/wo` + **sigmoid gate** (`wo(attn * sigmoid(gate(x)))`); **GQA 48 Q / 12 KV heads, head_dim 128** (KV repeated 4×); per-head **RMS QK-norm before RoPE**; no biases.
- MLP: SwiGLU `down(silu(gate(x)) * up(x))`, inner dim 16384.
- Norms: RMSNorm, `(1+scale)` convention (scale stored zero-centered), computed in f32.
- Modulation (AdaLN-single): `t = tmlp(sinusoidal(t·1000, dim 256))`; `tvec = tproj(t)` (6·6144). Each block adds its learned `mod.lin` (6·6144) to `tvec`, chunks into `prescale/preshift/pregate/postscale/postshift/postgate`:
  `x += pregate·attn((1+prescale)·prenorm(x)+preshift); x += postgate·mlp((1+postscale)·postnorm(x)+postshift)`.
- Final layer modulated by **`t` (pre-tproj)**, 2 chunks: `linear((1+scale)·norm(x)+shift)`, then unpatchify, take image-token slice only.
- **RoPE: theta=1000** (not 10000!), 3 axes, dims `[32, 48, 48]` over head_dim 128. Text tokens at position (0,0,0); image tokens (0, row, col).
- Sequence = concat(text_tokens, image_tokens); full attention, mask only if the text attention mask isn't all-ones.
- Text fusion (`txtfusion.*`): input (B, seq, 12, 2560) → 2 `layerwise_blocks` (each of the 12 layers independently, dim 2560, 20 heads, no GQA) → `projector: Linear(12→1)` collapses layer axis → 2 `refiner_blocks` (with text mask) → `txtmlp`: RMSNorm(2560) → Linear(2560→6144) → GELU(tanh) → Linear(6144→6144).
- No guidance embedding — **standard CFG** with negative prompt.
- `model_sampling.sigmas [10000]` in the checkpoint is just ComfyUI's serialized schedule buffer; ignore it and compute the schedule ourselves.

### Text encoder — Qwen3-VL-4B (prefix `model.language_model.*`)

- 36 decoder layers, hidden 2560, GQA 32 Q / 8 KV heads, head_dim 128, QK-norm, SwiGLU inner 9728, vocab 151936, RMSNorm. Causal attention, standard Qwen3 RoPE. **Skip the vision tower (`model.visual.*`) entirely** — text-only conditioning.
- No `lm_head` needed — we only want hidden states.
- Conditioning = hidden states tapped at layers `[2,5,8,11,14,17,20,23,26,29,32,35]` (see `krea2.py:17`) → (seq, 12, 2560).
- Tokenizer: GPT-2-style byte-level BPE, files at `comfy/text_encoders/qwen25_tokenizer/{vocab.json,merges.txt}` — **vendor these into `models/tokenizer/`**.
- Prompt template (`krea2.py:20`): system message "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:" in `<|im_start|>…<|im_end|>` chat format; after encoding, strip the prefix up through the second `<|im_start|>`(151644)+`user`(872)+`\n`(198). Negative prompt = same template around the (possibly empty) negative text.
- fp8 weights carry per-tensor F32 `weight_scale` (ComfyUI quant format): dequant = `f32(fp8_value) * weight_scale`. `comfy_quant` U8 tensors are metadata markers — parse and ignore.

### Sampler — rectified flow / flow matching

- `sigma(t) = exp(mu)/(exp(mu) + (1/t - 1))` with **shift mu = ln(1.15)** applied via `flux_time_shift(1.15, 1.0, t)`; sigmas over `t = linspace(1, 0, steps+1)` (ComfyUI convention), timestep fed to model = sigma.
- Euler step on `x = sigma·noise + (1−sigma)·x0`; model predicts velocity; `denoised = x − v·sigma`; `x_next = x + (sigma_next − sigma)·v`.
- CFG: `v = v_neg + cfg·(v_pos − v_neg)`, two model evaluations per step.

### VAE — Wan 2.1 (16 latent channels, F32)

- 3D causal convs (kernels `[kT,kH,kW]`), 8× spatial down/up, base width 96→384, mid-block with single-head self-attention, scale-only norm (`gamma [C,1,1,1]`). For still images run with T=1 (causal temporal padding).
- Latent normalization is **per-channel mean/std** (16 values each, `latent_formats.py:519`), `scale_factor = 1.0`: `decode input = latent · std + mean` (i.e. `process_out`). Decoder path: `post_quant_conv (conv2, 16→16, 1×1×1)` → decoder. Encoder not needed for t2i (only `conv2` + `decoder.*` tensors).

---

## Design decisions

- **Weight storage**: keep DiT weights **raw fp8-e4m3 in memory** (12.9 GB) and dequantize inside the matmul's panel-packing step via a 256-entry comptime f32 LUT (× `weight_scale` where present). Avoids a 26 GB bf16 blow-up; dequant cost amortizes since GEMM packs panels anyway. Norm scales / biases / small tensors dequant to f32 at load. VAE loads as f32 directly. Activations f32 throughout; f32 accumulation everywhere.
- **Staged execution to bound RAM**: load Qwen (5 GB) → encode prompts → free; load DiT (13 GB) → sample; free; load VAE (0.5 GB) → decode. Peak ≈ 14 GB + activations. (Note: ~55 GB of the 64 GB is currently in use on this box — ComfyUI should be stopped for big runs either way.)
- **File loading**: mmap the safetensors files read-only (`std.posix.mmap` — ZIG.md documents the 0.16 packed-struct flags). Header parsing follows ggufy's approach (`~/genai/ggufy/src/Safetensor.zig:107` — u64 LE length + `std.json.parseFromSlice(std.json.Value, …)`), but on the mmapped slice instead of reader calls.
- **Threading**: persistent `std.Thread` pool (8 workers), parallelize GEMM over row blocks and attention over heads. SIMD via `@Vector` (comptime lane width; AVX-512 on this box).
- **Image output**: PNG writer using **stored (uncompressed) deflate blocks** — a valid PNG needs only chunk framing + CRC32 + adler32, no compressor; ~4 MB for 1024². Real deflate (or `std.compress.flate` if 0.16 has a compressor) can come later.
- **RNG**: `src/torch_rng.zig` reproduces torch.randn's CPU path bit-for-bit
  (MT19937 + 24-bit uniforms + the AVX2 Cephes Box-Muller from avx_mathfun.h,
  including the shipping wheel's exact fma-contraction choices — pinned against
  fixtures from tools/dump_randn.py, one of them the full 470400-element
  comparison latent). A ComfyUI seed therefore reproduces the identical initial
  noise and the same composition; remaining pixel differences are per-step
  kernel-numerics only. Caveat: torch on Intel hosts takes an MKL VSL path with
  different output — we match the generic/AMD path.
- **CPU expectations**: at 12.3B params, a 1024×1024 CFG step is ~100 GFLOP·token-count territory — minutes per step. CPU milestone targets **512×512, few steps** for correctness; Vulkan is the usability phase.

## Module layout (all under `src/`, re-exported from `root.zig`)

```
tensor.zig        Tensor (dtype, shape, strides, data slice), views, f32 helpers
dtype.zig         fp8-e4m3 LUT, bf16<->f32, dtype sizes
safetensors.zig   mmap + header parse -> name->TensorInfo map; typed accessors
ops/matmul.zig    packed-panel GEMM, fp8/bf16 dequant in pack, thread pool
ops/attention.zig softmax attention, GQA, masks
ops/rope.zig      3-axis rope (theta param), qwen rope
ops/norm.zig      rmsnorm (1+scale), qk-norm
ops/act.zig       silu, gelu-tanh, sigmoid, swiglu
tokenizer.zig     byte-level BPE (vocab.json + merges.txt), chat template
models/qwen3.zig  text encoder (36 layers, 12-layer tap)
models/dit.zig    SingleStreamDiT (blocks, txtfusion, modulation, final layer)
models/vae.zig    Wan 2.1 decoder (causal 3D convs, T=1)
sampler.zig       flux_time_shift schedule, Euler flow-matching loop, CFG
image.zig         PNG writer (stored deflate), PPM for debugging
pipeline.zig      staged orchestration: encode -> sample -> decode
main.zig          CLI: generate --prompt --negative --out --steps --cfg --width --height --seed
```

## Milestones

Each milestone lands with unit tests; parity milestones compare against reference tensors dumped from ComfyUI (small Python dump scripts in `tools/`, run with ComfyUI's venv — dev-time only, not a runtime dependency). Comparison tolerance: max-abs-err thresholds appropriate to fp8 weights (~1e-2 relative for full layers, tighter for pure ops).

1. **Foundations**: `dtype.zig` (e4m3 LUT — verify all 256 values against a Python reference table checked into the test), `tensor.zig`, `safetensors.zig` (mmap, header parse; test: open all three real files, assert known tensor names/shapes/dtypes; plus synthetic-file unit tests).
2. **Ops**: GEMM (+ fp8 pack-dequant), rmsnorm, softmax attention w/ GQA, rope (both variants), activations. Tests: small-matrix golden values + property tests (`std.testing.fuzz` where cheap).
3. **Tokenizer**: byte-level BPE + Krea 2 chat template + prefix-strip logic. Test: token ids match ComfyUI's for a set of prompts (dumped fixture).
4. **VAE decode + PNG** (smallest model, f32 — first visible output): decode a ComfyUI-dumped latent, compare pixels; write PNG.
5. **Qwen3-VL text encoder**: full 36-layer forward, 12-layer tap. Parity: hidden-state stack vs ComfyUI dump for fixed prompts.
6. **DiT forward**: single forward pass parity vs ComfyUI dump (fixed latent, sigma, text embedding). This is the big one — build it block-type by block-type with per-submodule parity dumps (txtfusion out, block 0 out, final out).
7. **Sampler + pipeline + CLI**: schedule values parity test (pure math); then end-to-end `zig build run -- generate --prompt "..."` at 512×512 with ComfyUI-dumped noise → image should visually match ComfyUI's output for the same inputs.
8. **CPU performance pass**: threading coverage, AVX-512 kernel tuning, flash-style attention tiling, benchmarks (`-Doptimize=ReleaseFast`). Target: usable 512×512 iteration time.
9. **Phase 2 — Vulkan**: load `libvulkan` at runtime via `std.DynLib` + hand-declared extern fns (no C compilation — stays pure Zig); compute kernels written in Zig compiled to SPIR-V via Zig's SPIR-V backend. Start by offloading GEMM + attention for the DiT loop on the RTX 3090, keep CPU fallback. Detailed design deferred until CPU path is correct.

## Verification (end-to-end)

- `zig build test` green at every milestone (per CLAUDE.md: fix any breakage).
- Parity fixtures: `tools/dump_reference.py` (uses ComfyUI's code paths directly) writes `.bin` tensors into `testdata/`; Zig tests load and compare. Keep fixtures small (short prompts, 256×256 latents) so tests stay fast; large e2e checks live behind a CLI flag rather than `zig build test`.
- Final acceptance: same prompt/seed-noise/steps/cfg through ComfyUI and TensorPencil produce visually equivalent images at 512×512 and 1024×1024 (fp8 nondeterminism means near-identical, not bit-identical).

## Risks / open items

- **Wan VAE causal-conv details** (temporal padding, norm exactness) are the most fiddly part of parity; mitigated by doing VAE first with direct tensor dumps.
- **fp8 e4m3 semantics**: e4m3fn (no inf, 448 max) vs e4m3 — must match torch's `float8_e4m3fn`; the LUT test against a Python-generated table settles it.
- **`std.compress.flate` compressor availability in 0.16** — if absent, stored-block PNG is the fallback (already the plan).
- **Attention memory** at 1024×1024 (4096+ tokens, 48 heads): needs tiled/streaming softmax by milestone 8; naive full-matrix is fine at 512×512.
