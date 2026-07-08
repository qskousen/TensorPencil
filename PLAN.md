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
         LANDED (2026-07-02): buildGemmShared parameterized
         (warps8/acc_h16 toggles at the top of coopmat.zig; fp8 pipe only,
         the VAE f16w pipe stays 4-warp since its n_pads aren't %256;
         dispatch divides rows by coopmat.coop_wgn). Parameterization is
         regression-clean (parity 0.17520 exact at old-default toggles).
         Same-session A/B on bench-matmul 4224x6144x6144:
           4-warp f32-acc baseline:  6.4 ms (49.7 TF/s that session)
           8-warp f32-acc:           7.5 ms — register-gated as predicted
             (~165 regs x 256 threads > 1 wg/SM); toggle kept for retest.
           8-warp + f16 accs:        5.5 ms (57.8 TF/s, 81% of tensor
             peak) — the occupancy theory pays once regs fit 2 wgs/SM.
         Step-level (1024px, same-session A/B): matmul 2.24 -> 1.88 s.
         Numerics: DiT parity 0.17520 -> 0.16875 (marginally BETTER —
         different rounding, same regime); same-seed image vs f32 accs is
         compositionally identical at 0.53% mean pixel delta, the same
         class as our drift vs ComfyUI. The coop matmul unit test gate is
         now regime-aware (4 f16 ulps of the row's absolute-product sum —
         rationale in the test). warps8 + acc_h16 are the DEFAULTS now.
         Residual caveat: f16 accumulators cap at 65504; intermediate
         sums are bounded by the row's absolute-product sum, fine on real
         weights (fixture + full images), pathological inputs could
         saturate — watch for it if future models misbehave.
         RE-BASELINED (same day, free GPU, warm-ish box): 1120x1680
         20-step steady state 5.5 s/step (second run mean 5.59), ~112 s
         excl model loads (110 sampling + 1.8 decode), 2m00s wall incl
         loads => ComfyUI gap 112/85 = 1.32x (was 1.42x). Comparison
         repro regenerated with the f16-acc numerics: 0.64% mean pixel
         delta vs the ComfyUI ref (was 0.61% — same class).
         Streaming curve (--vram-budget flag, same session, 3-step runs):
         16 GiB ~6.8-7.1 s/step, 12 GiB ~7.2-8.0, 8 GiB ~7.4-10.3,
         4 GiB ~6.8 — i.e. +25-90%, noisy and NON-monotonic (the sync
         eviction design: flush + DeviceWaitIdle per eviction dominates
         over pure PCIe bandwidth; mid budgets thrash hit/miss the most).
         An async v2 would flatten this; good enough for degraded mode.
      2. Attention residual (~1.5 s at 1120x1680): the S compute runs at
         ~1 TF/s because Q/K cooperative fragment loads are global with
         12-15 KB row strides. K-STAGING BUILT AND MEASURED NEUTRAL
         (2026-07-02): buildGemmScores can stage each 64-deep K slab into
         the first 16 KB of the existing 32 KB S-bounce slab (dead until
         after the MMAs) — B fragments become ldmatrix-from-shared, zero
         extra workgroup memory, bit-identical output. Back-to-back A/B at
         1120x1680: scores ~700 ms/step BOTH ways. So global K fragment
         loads are NOT the scores bottleneck (same lesson as A-stride-34:
         the driver's coop global loads are better than theory says).
         Kept behind coopmat.scores_stage_k = false.
         FLASH K-STAGING ALSO BUILT AND MEASURED (same day, closing the
         "staged operands would fix flash" theory): buildFlashAttn gained
         stage_k (K tile 16 KB next to the 16 KB S tile = 32 KB, 3 wgs/SM
         vs 6; one shared copy replaces the 4x-redundant per-warp global K
         fragments). Three-way A/B at 1120x1680, attn per step:
           two-pass incumbent          1.75 s (scores .70 smax .30 out .73)
           flash, global K             2.67 s (md .86, out 1.81)
           flash, staged K             2.64 s (md 1.13, out 1.50)
         Mechanism now understood: staged K sped the recompute-heavy out
         pass 17% (loads DO matter there) but slowed the light md pass by
         the same amount (occupancy 6->3 wgs/SM + copy + barrier), and
         even a perfect out pass leaves flash ~0.6 s behind — the
         materialized-f16-S two-pass is simply cheaper than recomputing S
         twice at these shapes. 64-q/32-q retilings would stage Q too
         (already known L1-served) at even lower occupancy — dead end,
         not built. Flash stays behind use_flash=false / flash_stage_k=
         false; attention micro-optimization is CLOSED on this backend
         unless the driver's coopmat lowering changes.
      3. Cleanup candidates (no perf impact): coopmat.buildGemm (naive
         single-tile reference) and the eltwise kernels `modulate`,
         `attention`, `softmax_rows` are dead code now; delete or keep as
         documented references. Also: `zig build test` prints a spurious
         "failed command ... --listen=-" stderr line while the Build
         Summary reports all steps succeeded — pre-existing test-runner
         quirk (binary passes when run directly), not a test failure.
- [x] VAE convs on tensor cores (lever 3, 2026-07-02): buildGemmShared
      gained a `b_f16` variant — B staging is a plain uvec4 f16 copy (two
      quads per 16-element chunk) instead of the SWAR e4m3 decode;
      everything else (tiling, double buffer, MMA, C store) identical.
      Weights convert once on the CPU to zero-padded k-major f16
      (weightBufferF16From32: co pads to the 128 n-tile, 9*ci pads to the
      64 k-slab; a few MB per conv, cached like the fp8 uploads) and the
      activation conversion pads the k tail (f32_to_h16_pad — im2col
      output stays tight). C lands in a column-padded scratch and a new
      bias_compact eltwise kernel strips the pad + adds the conv bias in
      one pass into the tight destination. Convs with co >= 96 route to
      tensor cores (384/192/96 stages); post_quant (16) and the 3-channel
      head stay on the f32 GEMM. Decode at 1120x1680: 2.5 -> 1.7-1.8 s
      (the residual includes the one-time CPU weight convert + upload,
      im2col, norms, and the small non-coop convs). VAE fixture parity
      2.1e-4 -> 8.6e-4 max_err (f16 weight rounding, same regime as the
      f16 attention; comparison image delta vs ComfyUI unchanged at
      0.61%). DiT parity untouched (0.17520). Full 1120x1680 20-step
      image now ~119-120 s excl. loads (gap ~1.4x, GEMM-bound).
- [x] VRAM offloading v1 (2026-07-02): exhaustion degrades to weight
      streaming instead of a crash. (Note: the originally-sketched "free
      DiT weights before VAE" quick win turned out to ALREADY exist —
      pipeline.zig's evictWeights() call predates this; the measured
      17.5 GiB peak is during SAMPLING, not decode.) What landed, all in
      the context layer (dit_gpu untouched — the lazy weightBuffer cache
      makes streaming fall out of the cache layer):
      * VK_EXT_memory_budget (vk.zig bindings machine-checked against
        vulkan_core.h; enabled when present, physical-device-level query
        via core-1.1 vkGetPhysicalDeviceMemoryProperties2) gives a live
        budget that sees OTHER processes' VRAM; fallback is 90% of the
        device-local heap. Context.budget_override is the test hook.
      * device_used accounting on every createBuffer/free (all teardown
        routed through freeDeviceBuffer).
      * weights map is LRU-stamped; reserveForWeights evicts LRU entries
        (flushing the pending batch first — recorded dispatches may
        reference them; the Xid 109 lesson) until the new upload fits the
        budget headroom. Since the DiT walks blocks in a fixed order, LRU
        eviction IS sequential block streaming; evicted weights re-upload
        on next use through the existing cached-upload path.
      * Reactive backstop: createBuffer distinguishes
        error_out_of_device_memory and retries after evicting, so
        activation/scratch allocations survive pressure too.
      Verified: gpu-gated test forces budget_override = 3 GiB (model is
      12.4 GiB) — the streamed forward is BIT-IDENTICAL to the resident
      one; and the real-world repro (1120x1680 with a game holding 2.2 GB
      — previously died with error_out_of_device_memory) now completes at
      5.5 s/step (light eviction, barely above the unpressured estimate).
      Follow-ups for a v2 if streaming cost ever matters: async overlap
      via a dedicated transfer queue + double-buffered weight slots (the
      batched-submission work is the foundation; sync re-uploads flush
      the batch per eviction today), and composing with int8/CONVROT
      (halves streaming bandwidth).
- [x] Orchestration round (2026-07-03, same-session A/B at 1120x1680,
      4-step runs, steps 2-4): baseline 5.5-5.8 s/step -> 4.9-5.0 s/step
      (~11%); 20-step run 106.7 s sampling + 1.9 decode ~= 108.6 s excl
      loads (steady state crept 5.0 -> 5.3 warm) => gap ~1.28x. Everything
      below is BIT-IDENTICAL end to end (4-step PNGs match the pre-change
      baseline byte for byte, and the 20-step render matches the checked-in
      comfyui_repro PNG); DiT parity 0.16875 unchanged. Three items:
      * Workspace (dit_gpu.Workspace): the ~20 per-step tensorCreate/
        Destroy pairs (incl. the ~2 GiB scores buffer; each a raw
        vkAllocateMemory) moved to a per-run struct shared by both CFG
        sessions (sized for the longer text; heads_per_batch re-derives
        from the actual seq and clamps to the workspace buffers). Profile
        xfer 61 -> 3 ms/step; also decouples step buffers from the VRAM
        budget/eviction churn. Step 1 (weight upload) 7.3 -> 6.7 s.
      * Barrier elision (Context.independent(n)): opEnd skips the global
        compute barrier inside declared-independent groups (QKV/gate GEMM
        quad, gate/up pair, q/k norm and rope pairs, conversion triple).
        Measured NEUTRAL at 1120x1680 batched step times — the driver
        evidently already absorbs the small-grid tails (same lesson as
        K-staging / A-stride-34: theory says win, this driver says no).
        Kept: zero cost, correct (mid-group flush is a stronger sync), and
        smaller sizes with sub-wave grids may yet benefit. opMatmulCoop
        asserts it is never grouped (it records two dependent ops).
      * f16 C-store + fused f16 eltwise chain (the elt win: 573 -> 284
        ms/step): buildGemmShared gained c_h16 (binding 2 becomes an f16
        array, the accumulator stores directly — with acc_h16 the f32
        store was just a widening of f16 values, so f16 C is EXACT, which
        is why everything stays bit-identical). pipe_coop_c16 rides the
        fp8 pipe's toggles (requires coop_acc_h16). On this path q/k/g
        come out of their GEMMs f16, V's GEMM writes v16_d in the P@V
        layout directly (its f32 buffer and conversion die), qt_d dies (Q
        is normed/roped/scaled IN PLACE by the fused qknorm_rope16 kernel
        — one pass instead of rmsnorm + rope_inter + f32_to_h16, same
        operation order so f32 math is value-identical), K uses fused
        norm/rope + a raw-u16 gather (gather_kmajor16), the MLP gate/up
        GEMMs store f16 read by silu_mul16, and wo/down store f16 t1
        consumed by gated_add16. sigmoid_mul_g16 reads the f16 gate next
        to the f32 attention out. All old kernels/paths remain as
        fallbacks (per-block scale check can still route to them; the f32
        buffers stay allocated for that, ~140 MB slack).
      Follow-up candidates if another round is wanted: attn_out f16 C
      store (P@V accs are f32, so it would round — needs a parity look),
      Workspace sizes could shrink q/k/g/t1/mg/mu to f16 when the c16
      pipe exists, s_d ping-pong to overlap head-group scores with P@V.
- [x] INT8 ConvRot support (2026-07-03) — the `krea2CenterSemiraw_v10Int8`
      checkpoint (ComfyUI `int8_tensorwise` + `convrot`). Format: weights stored
      as `W_rot = W @ H^T` quantized int8 per output-row (F32 `weight_scale
      [rows,1]`), where H is a size-256 *regular* Hadamard (kron of the 4x4
      block) applied in groups along the input dim. H is symmetric+orthonormal
      (H=H^T, H@H=I). Only the 8 per-block DiT linears are I8; everything else
      (first/last/tmlp/tproj/txtfusion/txtmlp/norms) is F32/BF16. Keys nested
      under `model.diffusion_model.` (fp8 file uses bare `blocks.N`).
      * CPU (done + validated): `src/ops/convrot.zig` (comptime H256 for tests +
        `rotate` via a radix-4 fast Walsh-Hadamard — 4 passes stride 1/4/16/64,
        then /16). `Weight.row_scale`+`Weight.convrot` in matmul.zig dequant+
        rotate in both the packed and small-m paths. DiT `Loader` struct auto-
        detects the prefix and wires per-row scale + rotation onto I8 tensors.
        `--dit <path>` CLI flag selects fp8/int8 (auto-detected). Validated:
        int8 GEMM ≈ fp8 GEMM at 3.6% rel RMSE on real weights; coherent 256px
        CPU image. The naive matvec rotation was 222 s/step; the fast transform
        cut it to ~11 s/step (fp8 CPU is ~5 s/step).
      * GPU probe: `gpu-test` now dumps every coop config
        (Context.dump_coop_configs). 3090 exposes s8*s8->s32 at 16x16x32 and
        16x8x32 (int8 tensor cores; K=32 = 2x the f16 K). So the int8 IMMA
        speedup is reachable, not blocked at the config level.
      * GPU int8 GEMM (done + validated, `gpu-i8-test`): `coopmat.buildGemmI8`
        hand-assembled s8*s8->s32 kernel, MulAdd signed operands mask 0xF, Int8+
        StorageBuffer8BitAccess caps. Parameterized to an MTxNT (2x4) register
        tile — each subgroup computes mt*nt 16x16 tiles reusing loaded A/B
        fragments. NO workgroup memory (dodges the Zig/NVIDIA device-lost
        hazard). 0 mismatches vs CPU. Tile sweep @ 4224x6144x6144: 1x1 ~12,
        2x2 ~23, 4x2 ~29, 2x4 ~36, **4x4 ~46 TFLOP/s** (chosen; fp8 shared-mem
        kernel is ~57). Beyond 4x4 hits register pressure; i8_mt=i8_nt=4 set.
      * GPU full int8 linear (done + validated): `Context.opMatmulI8` chains
        eltwise `rotate` (x@H via an uploaded 256x256 H buffer), `rowmax_i8`
        (per-row absmax->scale), `quantize_i8` (pack 4 int8/word), the coop
        GEMM, and `scale_i32` (s32 * act_scale * weight_scale -> f32). Wiring
        exact vs a CPU replica; int8 accuracy 0.7% rel vs f32 (the rotation
        tames activation quantization well).
      * Prep optimization (all validated, `gpu-i8-test` batched full linear @
        4224x6144x6144): started 54 ms. (1) `rotate` coalesced by indexing H
        symmetrically as [l][local] (consecutive threads -> consecutive H
        addresses): 54 -> 19 ms. (2) fast Walsh-Hadamard `rotate_fwht` (one
        thread per 256-group, radix-4, 256-f32 private array): 19 -> 15 ms
        (private-array spill is now the floor — a shared-mem/subgroup-shuffle
        rotate would go faster but hits the Zig workgroup hazard). (3) fused
        per-group abs-max into rotate_fwht + cheap `rowscale_i8` reduce
        (replaces the latency-bound one-thread-per-row `rowmax_i8`): 15 -> 14 ms.
        Breakdown now ~7 ms GEMM (4x4 tile) + ~7 ms prep. fp8 GEMM same shape is
        ~5.5 ms, so a naive per-GEMM int8 linear is ~2.5x SLOWER than fp8.
      * KEY: the standalone bench runs full prep PER GEMM, but in the DiT one
        rotated+quantized activation feeds 4 GEMMs (wq/wk/wv/gate) or 2 (mlp
        gate/up) — prep amortizes ~4x/2x (same as fp8's shared f32->f16). So the
        real per-GEMM overhead is far lower than the proxy; dit_gpu integration
        is needed to know if int8 actually wins.
- [x] INT8 ConvRot dit_gpu integration + end-to-end result (2026-07-03).
      `opMatmulI8` split into `opI8Prep` (rotate+quantize once) + `opI8Gemm`
      (per-weight GEMM+rescale) so one prepped activation feeds wq/wk/wv/gate
      and mlp gate/up. dit_gpu.forward gains an `is_i8` route (gated on
      `blk.attn.wq.dtype == .i8`; fp8 path byte-for-byte untouched): int8 blocks
      take the f32-output branches (still tensor-core attention via the
      f32->f16 gather/convert), swapping the 8 block GEMMs to int8. Produces
      correct, coherent images (512px fox verified; testdata/int8_gpu_smoke.png).
      RESULT @ 1120x1680, warm: **int8 8.28 s/step vs fp8 4.80 s/step — int8 is
      1.7x SLOWER.** Profile (sync-per-op, steady state): matmul int8 5401 vs
      fp8 2858 ms (1.9x — the register-tiled int8 GEMM is slower than fp8's
      shared-mem kernel, and the 4 qkv int8 GEMMs SERIALIZE because they share
      i8_acc, vs fp8's overlapped independent(4)); attn ~equal (both TC); elt
      int8 1184 vs fp8 261 ms (+920 ms = the rotate/quantize/scale prep fp8
      doesn't pay). Conclusion: matches the memory-note thesis — beating fp8
      needs the GEMM pushed past fp8 (shared-mem #2, int8 has ~2x peak) AND the
      prep cut hard; the full ~2x ComfyUI speedup likely needs a single fused
      rotate+quantize+IMMA+scale kernel (large). int8's win is accuracy/VRAM,
      not yet speed, on this Vulkan path.
- [ ] INT8 ConvRot follow-ups / forks (pursue to find the best options):
      * [tried, NEUTRAL] Overlapping the qkv/mlp int8 GEMMs via distinct accs
        (opI8GemmRaw/opI8Scale/i8AccPool exist for it) gave 8.33 vs 8.28 s/step
        — no change. So the 1.9x matmul gap is the register-tiled GEMM being
        slower in-DiT, NOT serialization (the driver already overlaps tails, same
        lesson as the fp8 barrier-elision experiment). Reverted to single-acc to
        save VRAM. => the shared-mem GEMM (#2) is the ONLY lever for the matmul.
      * TOP FORK for speed (#2): shared-memory staged int8 kernel (adapt the fp8
        `buildGemmShared`, or write a simpler single-buffered s8 one — stage s8
        A/B through workgroup memory, no decode, s32 accum). int8 has ~2x
        tensor-core peak, so this could take the GEMM from ~46 past fp8's ~57
        TFLOP/s toward ~90-110. Rough math: if matmul drops 5401->~1900 ms, int8
        total ~= 1900 + 1369(attn) + 1184(prep) ~= 4.5 s/step ~= fp8's 4.8 — i.e.
        roughly PARITY, maybe a slight win. The full ComfyUI ~1.9x needs a single
        fused rotate+quantize+IMMA+scale kernel (eliminates the ~920ms prep and
        the s32->f32 round trip) — a large project.
      * FORK (#2, deferred from the register-tile decision): a full
        shared-memory staged int8 kernel like the fp8 `buildGemmShared`
        (stage s8 operands through workgroup memory — half the bytes of the
        f16 staging, no decode; double-buffer k-slabs). Should beat the 36
        TFLOP/s register-tiled kernel and possibly the 57 TFLOP/s fp8 path
        (int8 has ~2x tensor-core peak). ~800 lines of SPIR-V; higher risk.
      * [resolved] MTxNT tile tuning — swept {1,2,4}^2: 4x4 best at ~46 TFLOP/s
        (i8_mt/i8_nt=4 in coopmat.zig). 6x6/8x8 untested (register pressure +
        the tuning harness's small check cases don't divide the bigger tiles).
      * NEXT: wire the int8 path into dit_gpu.forward (it currently hardcodes
        the fp8 coop path). Needs the int8 weights routed through opMatmulI8
        (fold the modulation into the pre-rotation input, reuse the fused
        norm/gate kernels where possible), then a real batched DiT s/it to
        compare against fp8's ~5 s/step. This is the true speedup test.
      * NEXT: DP4A alternate GEMM (VK_KHR_shader_integer_dot_product / OpSDot)
        — the second path the comparison wants. NOTE: coopmat and DP4A both do
        EXACT int8*int8->i32, so their ACCURACY is identical; the fork between
        them is purely SPEED (tensor cores vs integer-dot ALUs). The 0.7%-vs-f32
        number above is the int8-quantization accuracy, common to both.
      * [partly resolved] GPU rotate: coalesced matvec + fast Walsh-Hadamard
        done (54->14 ms full linear). Remaining fork: the FWHT's 256-f32 private
        array spills; a shared-mem or subgroup-shuffle rotate (hand-assembled to
        dodge the Zig workgroup hazard) could cut the rotate further. Also could
        fuse quantize into the rotate once the per-row scale is known (needs the
        two-pass structure) and skip the f32 xr round-trip.
      * FORK: activation quantization uses round-half-away (@round) vs torch's
        round-half-even; negligible at int8 noise but worth a parity check if
        matching ComfyUI pixels exactly matters.

### Fused int8 ConvRot kernel — detailed design (the path to a real speedup)

GOAL: get int8 convrot BELOW fp8's ~4.8 s/step @1120x1680 (currently 8.28).
The two costs to kill: (1) the register-tiled GEMM (~46 vs fp8's ~57 TFLOP/s),
(2) the ~920 ms/step of separate prep passes with full-tensor round-trips.

Current per-linear dataflow (5 GPU passes, each a full DRAM round-trip):
  x(f32) --rotate_fwht--> xr(f32) + partials
        --rowscale_i8--> act_scale
  xr,act_scale --quantize_i8--> x_i8(int8)          [xr write+read round-trip]
  x_i8 --opMatmulCoopI8--> acc(s32)                 [register-tiled, no reuse]
  acc,act_scale,weight_scale --scale_i32--> y(f32)  [s32 write+read round-trip]
Shared across GEMMs: prep (rotate/scale/quantize) is done ONCE per shared-input
group (qkv/gate, mlp gate/up); wo and mlp.down prep individually.

HARD CONSTRAINT that shapes the design: dynamic per-row (per-token) quant needs
the FULL row's abs-max of x_rot before any element can be quantized (rotation is
orthonormal so it preserves L2 but NOT L-inf — no shortcut). And a full row is
6144 or 16384 f32 (24 or 64 KB) vs the 48 KB workgroup-shared cap. So a single
"do everything for a tile in one kernel" is only possible for small K; the
general case stays two kernels (prep, then GEMM). Also: ANY kernel using
workgroup(shared) memory must be HAND-ASSEMBLED SPIR-V (coopmat.zig style) —
Zig-emitted workgroup kernels DEVICE_LOST on this NVIDIA driver (see ZIG.md).

Recommended staged plan (do in order; each is independently shippable + testable):

  STAGE A — fold the rescale into the GEMM C-store (needs #2 first, or bolt
  onto the current register-tiled kernel as a quick check). The int8 coop GEMM,
  at its OpCooperativeMatrixStoreKHR, instead of storing s32 to a buffer, reads
  act_scale[row] (broadcast down the 16 rows of the C tile) and weight_scale[col]
  (down the 16 cols), computes f32(acc)*act_scale*weight_scale, and stores f32 y
  directly. Kills the s32 acc buffer + the whole scale_i32 pass (one of the ~4
  prep passes/block, plus a 104 MB s32 write + read). Implementation: the coop C
  fragment is 16x16 s32; extract-to-f32 then multiply needs per-element access to
  the fragment — either OpCooperativeMatrixStoreKHR to shared then a threaded
  scale+store, or use the coopmat element-access. Simplest first cut: store s32
  to shared (small, one tile), then 32 threads scale+write the 16x16 tile to y
  with act_scale/weight_scale from small buffers. Bindings gain act_scale (per
  row-tile) + weight_scale (per col-tile). Push gains their base offsets.
  Expected: removes ~1 pass + the s32 round-trip; modest on its own, but it is
  the prerequisite for the GEMM to emit f32 directly.

  STAGE B — fused prep kernel (rotate + rowmax + quantize in ONE launch, no xr
  round-trip). One workgroup per row (or small row-tile). Hand-assembled SPIR-V
  (uses shared memory). Per row: (i) the 32 threads cooperatively FWHT-rotate the
  row's 256-groups, writing rotated f32 into shared [cols] (24 KB for cols=6144,
  fits; cols=16384 = 64 KB does NOT fit -> that linear (mlp.down input) either
  stays on the 3-pass path or tiles the row across 2 shared slabs with a
  two-level max), (ii) reduce abs-max over shared -> act_scale, (iii) quantize
  shared f32 -> packed int8 output + write act_scale. Fuses 3 passes -> 1 and
  removes the xr(f32) 104 MB write+read. Keep the existing eltwise rotate_fwht/
  rowscale_i8/quantize_i8 as the fallback for the 16384 case and for validation.

  STAGE C (stretch, small-K only) — merge prep into the GEMM prologue: a
  per-output-tile kernel that stages its M-row activation tile, rotates+maxes+
  quantizes it in shared, then runs the IMMA k-loop from that shared int8 tile +
  streamed weights, rescaling at store. Only viable when M*K int8 fits shared
  (small linears); general K makes the shared activation tile too big, so this is
  a targeted optimization, not the general path. Probably NOT worth it given A+B.

  PREREQUISITE / parallel track — #2 shared-memory int8 GEMM (see the TOP FORK
  bullet above). Stage A's fused rescale rides on whichever GEMM kernel is used;
  pairing it with the shared-mem GEMM is what pushes the matmul past fp8. Adapt
  coopmat.buildGemmShared (fp8, double-buffered uvec4 staging) to s8: stage s8
  A/B through workgroup memory (half the bytes of the f16 staging, NO decode
  step — simpler than fp8), s32 accumulators, K=32 slabs, MulAdd operands 0xF.

VALIDATION: extend gpu-i8-test — it already checks wiring-exact vs a CPU replica
and 0.7% accuracy vs f32 for the full linear; add cases that exercise the fused
kernels and keep the rel-vs-cpu-sim < 1e-3 gate. Then end-to-end s/it vs fp8 at
1120x1680 (both bit-comparable images already validated). Profiler:
`generate --gpu on --profile on` prints matmul/attn/elt category ms.

REALISTIC CEILING: A+B remove ~2-3 round-trips and ~1-2 passes/linear; with the
shared-mem GEMM the matmul target is ~1.9 s (from 5.4) and prep drops toward
fp8's ~260 ms. Optimistic int8 ~= 1.9 + 1.4(attn) + ~0.4(prep) ~= 3.7 s/step vs
fp8 4.8 => ~1.3x, approaching ComfyUI's 1.9x (they also fuse; the residual gap is
the extra rotate work int8 fundamentally does). Attention is already TC-equal;
one more lever if needed: give int8 the att16 fully-fused f16 attention path
(currently int8 takes the f32-input TC path, slightly less fused).

### Fused int8 ConvRot kernel — EXECUTION LOG (2026-07-03, this session)

Goal: get int8 below fp8's 4.8 s/step @1120x1680 (was 8.28). Same-session
baseline (bench, min-of-8, 4224x6144x6144): fp8 shared GEMM 5.3 ms (60 TF/s),
int8 register GEMM 6.9 ms (46 TF/s), full int8 linear 14.2 ms (~7 GEMM + ~7 prep).

- [x] **Fork #2 — shared-memory staged int8 GEMM (`coopmat.buildGemmSharedI8`).**
      The big lever, and it OVER-delivered. 128x128 workgroup tile, 2x2 warp grid
      of 64x64 (mt=nt=4, 16 MMAs/32k, 2 ks/step), K_STEP=64, SINGLE-buffered
      (barrier/stage/barrier/MMA). Design decision that made it clean: the shared
      arrays are **u32-typed** (staging is a plain uvec-free u32 copy, NO decode —
      half the bytes of the fp8 f16 staging) and the s8 cooperative fragment loads
      read that u32 workgroup memory with **u32-unit strides** (the standard
      "int8 matrix from a uint buffer" GLSL pattern). This sidesteps any 8-bit
      workgroup-write capability question entirely and needs only Int8 +
      CooperativeMatrixKHR + VulkanMemoryModel caps (no StorageBuffer8BitAccess —
      global A/B are read as u32 during staging, never as s8).
      FORK RESOLVED (u32-shared vs s8-shared): tried u32-shared FIRST (fewest caps,
      cleanest stores); it validated 0-mismatch on the first build, so the s8-shared
      byte-store fallback was never needed.
      RESULT (same-session bench): **3.72 ms best (85.7 TF/s)** — 1.85x the register
      int8 kernel (6.9 ms) and **1.4x faster than the fp8 shared GEMM** (5.3 ms).
      Blew past the PLAN's "≈parity with fp8" ceiling (int8's ~2x tensor peak is
      real once staging is shared). 0 mismatches on 128x128x64 and 256x384x320
      correctness cases (gpu-i8-test). Routing: `opMatmulCoopI8` picks the shared
      pipe when m_pad%128==0 && rows%128==0 && cols%64==0 (all DiT-block shapes;
      `opI8Prep` now pads m to 128), else the register kernel (small shapes / bench
      check cases). Toggle `coopmat.i8_shared`.
      => GEMM is no longer the int8 bottleneck. Full linear 14.2 -> 10.9 ms; the
      remaining ~7 ms is PREP (rotate/rowscale/quantize + the scale_i32 pass).
      Next: Stage A (fuse rescale into C-store) + Stage B (fused prep) attack prep.
- [x] **End-to-end DiT with shared GEMM (same-session, 1120x1680, 5-step batched):
      int8 6.2 s/step vs fp8 4.6 s/step** — int8 1.35x SLOWER (was 1.7x @8.28 pre-GEMM).
      Profiled category breakdown (sync-per-op), int8 vs fp8:
        matmul  3585 vs 2899  (+686)
        attn    1431 vs 1406  (~0, both tensor-core)
        elt     1187 vs  274  (+913 = int8 rotate/rowscale/quantize/scale prep +
                               the un-fused f32 norm/rope/gather path vs fp8's f16 fused chain)
      Two gaps: matmul +686, elt/prep +913. The int8 shared GEMM benches at
      87-95 TF/s at real DiT shapes (7680x6144x6144 6.63ms, 7680x16384x6144 17.3ms,
      7680x6144x16384 16.2ms), implying ~2070ms of DiT matmul if sustained — but
      in-DiT it's 3585. The 1.73x in-DiT efficiency loss (fp8 doesn't suffer it) is
      NOT occupancy:
- [x, FORK REJECTED] **8-warp (2x4) int8 GEMM** for occupancy (buildGemmSharedI8
      `warps8`: 8 accs/warp = 64 s32/thread -> ~2 wgs/SM vs 4-warp's 128 regs ->
      ~1 wg/SM). Correct (0-mismatch) but SLOWER both isolated (7.65 vs 6.63 ms
      @7680x6144x6144 — lower MMA/load ratio) AND in-DiT (matmul 3821 vs 3585,
      same-session). Same lesson as every prior occupancy experiment on this
      driver (fp8 8-warp, barrier-elision, A-stride-34): the driver doesn't reward
      raising occupancy at DiT sizes. Reverted to 4-warp (`coop_i8_warps8 = false`);
      the toggle stays for other drivers. => the +686 matmul gap is most likely
      clock/scheduling (the mixed in-DiT workload doesn't sustain the boost the
      tight bench loop does) or single-buffered load-latency stalls; double-buffer
      / larger-K_STEP is the remaining unexplored matmul fork (deferred — smaller
      and less certain than the +913 elt/prep gap).
- [x] **Stage A — fuse the act*weight rescale into the shared GEMM C-store**
      (`buildGemmSharedI8(warps8, fuse_scale)`, `Context.opMatmulCoopI8Fused`).
      Kills the s32 accumulator buffer (mlp.up's was ~500 MB!) + the whole
      scale_i32 pass (an m*rows s32 write + read per GEMM). At the merge, each
      16x16 s32 C fragment coop-stores into a per-subgroup s32 workgroup scratch,
      then the 32 lanes read it back and write y = f32(acc)*act_scale[row]*
      weight_scale[col] (f32, element-wise). Binding budget solved WITHOUT a 5th
      binding: binding 3 = ONE scale buffer [act(m_pad) | weight(rows)] f32,
      assembled per GEMM by a new `scale_concat` eltwise kernel (single compute
      dispatch, so the batch's compute barriers serialize it hazard-free vs the
      GEMM — no transfer/compute race, no Xid 109). Push u0 (=m_pad) is the split.
      Correctness: fused kernel 0.0 rel err vs CPU (s32->f32 + f32 scale is exact
      for real magnitudes); the DiT int8 linear route (opI8Gemm -> fused, opMatmulI8
      for wo/down) validates wiring rel 0.0; 512px fox image coherent + identical
      regime. RESULT (same-session batched, 1120x1680): **int8 6.2 -> 6.0 s/step**
      (~0.2 s; profiled matmul 3585->~3320 as the scale folds into the store,
      elt loses the 8 scale_i32 passes but gains the tiny scale_concat) + the
      ~500 MB s32 buffers are gone (VRAM). Kept ON by default (accuracy-neutral).
      VRAM peak at 1120x1680 = 21.4/24.5 GiB (NO streaming; GPU 100% util — the
      gaps are genuinely compute/clock-bound, not bandwidth/eviction).

- [x] **Stage B — fused prep kernel (`coopmat.buildFusedPrepI8(cols)`): rotate
      FWHT + per-row abs-max + dynamic quantize in ONE hand-assembled
      f16-shared-memory kernel.** THE prep killer. Isolating prep with a profiler
      sub-timer showed it was **2160 ms/step** — the single biggest int8 overhead
      (as large as matmul; fp8 pays zero), NOT the +913 "elt" I'd mis-attributed.
      Design: one workgroup (32 or 64 threads) per row; (0) coalesced f32->f16
      load into a padded shared buffer [ng groups][257] (PAD 257 = bank-conflict-
      free: lane t -> bank (t*257)%32 = t), (1) threads 0..ng-1 each FWHT +
      normalize their group IN SHARED (kills the private-array spill that was the
      prep floor) + emit the group abs-max, (2) thread 0 reduces -> per-row scale,
      (3) all threads coalesced-quantize shared -> packed int8. Hand-assembled
      (Zig-emitted workgroup kernels DEVICE_LOST on this driver) with a
      counted-loop helper + explicit current-block tracking for the phis; 4 FWHT
      passes are bf-loops with mask/shift index math. **f16 shared** was the key
      second decision: it makes cols=16384 (mlp.down) fit (32.9 KB) AND gives
      cols=6144 3-4 workgroups/SM (12.3 KB) vs 1 for f32 — so ALL 4 preps/block
      run the fused kernel (two builds, cols 6144 + 16384; picked in opI8Prep).
      Validated first correct build (no DEVICE_LOST): gpu-i8-test 128x128x6144 and
      x16384 both rel-vs-f32 ~0.009 (int8 accuracy UNCHANGED — the f16 rotation
      error stays inside int8 quant noise; the GPU-f16-vs-CPU-f32-replica rel is
      ~0.4%, gated at 1e-2 for cols>=6144 with rationale). 1120x1680 fox image
      clean. RESULT: **prep 2160 -> 1010 (f32 Stage B, 3 preps) -> 187 ms**
      (f16 Stage B, all 4 preps). Step (batched, same-session): 6.0 -> 5.0 -> 4.2.

- [x] **More levers (2026-07-03 cont.): int8 4.2 -> 4.0 s/step.**
      * Double-buffered int8 shared GEMM (`coop_i8_double_buf`; issue next k-slab's
        global loads into registers before the current MMA). Correct (0-mismatch).
        Measured ~NEUTRAL isolated (3.63 vs 3.72 ms) AND in-DiT (matmul 2060 vs
        2100) — same as the fp8 double-buffering result: this driver already hides
        the load latency, so the ~85 TF/s (30% of int8 tensor peak) is the driver's
        coopmat-lowering ceiling, not a single-buffer stall. Kept (correct, marginal).
      * int8 attention-setup fusion: new `qknorm_rope_f32` eltwise kernel folds
        rmsnorm+rope into ONE f32 in-place pass for q and k (the int8 path took the
        un-fused 7-pass f32 route; att16's qknorm_rope16 is value-identical, so the
        f32 adaptation is safe — kept the existing f32_to_h16/gather converts which
        zero the seq_pad tail correctly). Attention setup 7 -> 5 passes; **elt 500
        -> 370 ms**; step 4.2 -> 4.0 (batched, same-session). Coherent fox, fp8
        parity 0.16875 unchanged. Remaining elt vs fp8 (274) needs the c_h16 int8
        GEMM + full f16 chain (risky; diminishing) — matmul/attn are at ceilings.

- [x] **f16 attention chain for int8 (2026-07-03 cont.): int8 4.0 -> 3.9 s/step,
      elt 500 -> 300 ms (near fp8's 274).** Gave int8 attention the fp8 att16
      treatment: wq/wk/wv int8 GEMMs now store f16 directly (`buildGemmSharedI8`
      c_h16 param -> `pipe_coop_i8_fs16`; the fused-rescale threaded store
      FConverts to f16), so the norm/rope chain collapses to att16's 3 fused f16
      passes (qknorm_rope16 in-place on q/k, gather_kmajor16, v lands in v16_d from
      the GEMM) instead of int8's 5-pass f32 route. GATE stays f32 (c_h16=false),
      so the gate/wo/mlp path is untouched — no f16-input prep needed (contained).
      dit_gpu gates the att16 branch on `attn_f16 = att16 or i8_f16`.
      BUG FOUND + FIXED: the C-array ArrayStride was hardcoded 4 (s32/f32); f16 C
      needs stride 2 or the f16 stores scatter (first image was noise). One-line
      fix. Then coherent 512px + 1120x1680 fox; fp8 parity 0.16875 unchanged;
      int8 accuracy rel-vs-f32 0.0089/0.0094 unchanged. Remaining elt vs fp8
      (~26 ms) is the still-f32 gate/mlp/modulation ops (sigmoid_mul, silu_mul,
      rms_apply_mod, gated_add) — fusing them to f16 needs the mlp/gate GEMMs f16
      + f16-input Stage B prep for wo/down; ~26 ms for a big integration, NOT
      worth it (elt is already at fp8 parity).

### Fused int8 ConvRot — SESSION OUTCOME (2026-07-03): int8 BEATS fp8

**int8 8.28 -> 3.9 s/step steady state @1120x1680; fp8 is 4.6 -> int8 is ~15-18%
FASTER.** (Was 1.7x SLOWER at session start.) Three kernels did it, all validated
end-to-end (clean 1120x1680 fox; fp8 path bit-identical at 0.16875 parity;
gpu-i8-test green; all `zig build test` green):
  1. **Shared-memory int8 GEMM** (register-tiled -> u32-staged-shared, 46 -> 85+
     TF/s, beats fp8's ~57). The unlock.
  2. **Stage A** — fused act*weight rescale in the GEMM C-store (kills the s32 acc
     buffer + scale_i32 pass). matmul now 2050 vs fp8 2900 — int8 matmul is FASTER.
  3. **Stage B** — fused f16-shared prep (rotate+rowmax+quantize one kernel):
     prep 2160 -> 187 ms.
Final same-session profiled steady state: int8 matmul 2100 / attn 1390 / elt 500 /
prep 187 = ~4.2 s; fp8 matmul 2900 / attn 1400 / elt 274 = ~4.6 s.

This OVERTURNS the earlier standing assessment that fp8 was unbeatable on this
Vulkan/NVIDIA path — int8's ~2x tensor peak, once the shared-mem GEMM + fused
prep remove the overheads, wins outright while ALSO giving better accuracy
(0.9% vs f32) and, with Stage A, less VRAM. See [[comfyui-gap-assessment]]
(now updated), [[int8-convrot-notes]], [[gpu-perf-lab-notes]].

FORKS this session: shared GEMM (DONE, unlock) · Stage A scale fusion (DONE) ·
Stage B f16 fused prep (DONE, the prep killer) · 8-warp occupancy (REJECTED,
worse) · VRAM streaming (ruled out).
REMAINING (optional, int8 already wins): int8 elt is still un-fused f32 (500 vs
fp8's fused-f16 274) — routing int8 through the f16 eltwise chain would widen the
lead ~200 ms but needs a c_h16 GEMM output + dit_gpu integration; matmul residual
is clock-bound (occupancy rejected). [CORRECTION 2026-07-03: the "~PARITY with
ComfyUI" framing that used to live here was WRONG — the ~85 s / ~4.25 s-equiv
number is ComfyUI's FP8 run, and Ampere has NO fp8 tensor cores (it upcasts to
f16 → a soft target, the only thing our int8 edges). ComfyUI's INT8 path
(`torch._int_mm` → cuBLASLt IMMA + flash-attn) samples at **~2 s/it vs our
~4 s/step — ~2x faster.** The gap is kernel efficiency under Vulkan's structural
ceilings, not the silicon. See the M10 CUDA backend section below.]
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

## M10 — CUDA backend experiment (int8 convrot): close the ~2x ComfyUI gap

### Why (corrected baseline)

Our int8 Vulkan path is **~4 s/step @1120x1680**; ComfyUI int8 is **~2 s/it — ~2x
faster** (user-verified). This is NOT a silicon gap (same 3090, same int8 IMMA
peak) and NOT an API gap (the CUDA Driver API and Vulkan host API are equivalent
C-ABIs we call the same way). It is a **kernel-efficiency gap**: ComfyUI leans on
`cuBLASLt` (int8 GEMM near hardware peak) + flash-attn, while our Vulkan coopmat
GEMM tops out at ~85 TOPS (~30% of peak) held back by three Vulkan-only ceilings —
**48 KB shared-memory cap, no `cp.async`, and opaque coopmat lowering** (no
SASS/scheduling control). See [[comfyui-gap-assessment]] / [[int8-convrot-notes]].

Two experiments, in order, to decompose that 2x and decide if it's worth closing:

- **Phase 1 — hand-rolled Zig + PTX CUDA backend.** Preserves "pure Zig, no C
  deps" *by the project's own standard*: `libcuda` is a system driver we `dlopen`
  exactly like `libvulkan`, and PTX is device IR we hand-emit exactly like SPIR-V
  (`coopmat.zig`). Tests how much of the 2x is recoverable purely by breaking the
  Vulkan ceilings (>48 KB dynamic shared + `cp.async` + hand-written IMMA tiling).
  Honest ceiling: PTX exposes `mma`/`cp.async`, but `ptxas` (closed) owns
  register allocation + scheduling, so this likely lands **between** Vulkan (85
  TOPS) and cuBLASLt — it may close *part* of the gap, not all.
- **Phase 2 — link cuBLASLt (and optionally flash-attn/cuDNN).** This is the thing
  that actually reaches ComfyUI speed, because it *is* the kernels ComfyUI uses.
  It **breaks pure-Zig** (a closed-source C/CUDA math-library dependency, NVIDIA-
  only) — so it lives behind a build flag, Vulkan stays the default portable path,
  and the tradeoff is documented, not silently taken.

The comparison across {Vulkan, hand-PTX CUDA, cuBLASLt CUDA} vs ComfyUI ~2 s/it
answers the real question: how much is "our kernels" (Phase 1 recovers) vs "their
libraries" (only Phase 2 recovers).

### Architecture: a backend seam (enabling refactor)

Today the GPU path is Vulkan-specific: `src/gpu/context.zig` (`Context` — buffers,
`op*` methods, `beginBatch`/`endBatch`/`independent`), `coopmat.zig` (SPIR-V),
`eltwise.zig`; `dit_gpu.zig` / `vae_gpu.zig` orchestrate against `Context`
directly. Adding CUDA alongside needs a **backend interface** covering exactly the
surface those orchestrators use:
- buffers: create / free / upload / download / grow, device-used accounting;
- GEMM variants: `opMatmul`, `opMatmulCoop*`, int8 `opI8Prep` / `opI8Gemm` /
  `opMatmulCoopI8Fused`;
- eltwise kernels (norm/rope/gather/quantize/scale/gate families);
- submission: batch record + flush, `independent(n)` barrier grouping, sync.

Plan: extract a `GpuBackend` vtable (or comptime-dispatched tagged union — pick
whichever keeps `dit_gpu` readable; a vtable is simplest for two backends), move
the current `Context` behind `VulkanBackend`, and add `CudaBackend`. `dit_gpu`
becomes backend-agnostic. **Do the seam only once Phase 1's standalone GEMM proves
CUDA is worth it** — don't refactor on spec.

### Phase 1 — hand-rolled Zig + PTX (stay pure-Zig)

Each step is independently shippable + testable. The correctness oracle is the
existing CPU replica in `gpu-i8-test` (backend-agnostic — same numeric gates:
wiring rel 0.0, accuracy ~0.9% vs f32). Measurement mirrors `bench-matmul`
(min-of-≥8; the clock governor caveat in [[gpu-perf-lab-notes]] applies identically
to CUDA — the 3090 idles ~500 MHz, boosts under queue pressure).

#### EXECUTION LOG (2026-07-03, hand-PTX CUDA backend — unattended build)

Toolchain on this box (verified): driver 580.173.02 / CUDA 13.0, `libcuda.so.1`
present, `libnvidia-ptxjitcompiler.so.1` (built-in JIT), `ptxas`/`nvdisasm`/
`cuobjdump` at `/usr/local/cuda-13.0/bin` (dev-time PTX validation only; runtime
JITs via the driver). Zig 0.16.0. Device: RTX 3090 sm_86, 82 SMs, **opt-in shared
99 KB/block, 100 KB/SM** (vs Vulkan's hard 48 KB — the unlock), 1800 MHz boost.

Offline PTX de-risk (all assemble under `ptxas -arch=sm_86`, SASS confirmed):
`mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` → `IMMA.16832.S8.S8`;
`cp.async.cg.shared.global` → `LDGSTS.E.BYPASS.128` + `LDGDEPBAR` (the Vulkan-
denied async copy); `.extern .shared` dynamic shared; `ldmatrix.x4.b16`.
m16n8k32 s8 fragment layout (verified from generated PTX): A=4×b32, B=2×b32,
C/D=4×b32(s32); `groupID=lane>>2, tid=lane&3`; A[a0..3]=(row groupID/+8, k
tid*4{+16}); B[b0,b1]=(k tid*4{+16}, col groupID); C[c0..3]=(row groupID/+8,
col tid*2{+0,+1}). First-bringup load = plain `ld.shared.b32` per the table
(ldmatrix is a later throughput opt).

- [x] **1.0 bindings** (`src/gpu/cuda/cu.zig`) — dlopen libcuda.so.1, all driver
      entry points as `callconv(.c)` externs; ABI hazards handled (CUdeviceptr=u64;
      `_v2` suffixes on cuMemAlloc/Free/Memcpy*/Memset*/CtxDestroy/StreamDestroy/
      EventDestroy/DeviceTotalMem/MemGetInfo/ModuleGetGlobal; cuCtxCreate_v2 simple
      form). Test loads driver + queries device (green with -lc).
- [x] **1.2 PTX emitter** (`src/gpu/cuda/ptx.zig`) — text builder: per-class
      virtual registers (%r/%rd/%f/%rs/%p), labels, formatted lines, `.reg`-count
      finalize, module preamble. Unit test green.
- [x] **1.1 smoke test** (`cuda-test`) — `Context` (device/ctx/stream, PTX JIT via
      cuModuleLoadDataEx with error-log capture + sm_86 target, buffers, launch,
      cuEvent timing) in `src/gpu/cuda/context.zig`; vadd PTX JIT+launch+readback
      verified. `cuda-test` prints caps + "OK". Pure-Zig line preserved.
- [x] **1.3 int8 IMMA GEMM — DECISION GATE PASSED (1.5-1.6x over Vulkan).**
      Two kernels, both bit-exact vs the CPU integer oracle (0 mismatches incl.
      128x256x6144 real k-depth):
      * `igemm_v0` (hand-PTX string) — correctness reference: 1 warp / 16x8 tile,
        fragments straight from global, no reuse. 9.5 TOP/s @4224x6144x6144.
        Confirmed the m16n8k32 s8 fragment layout is exactly right. KEY finding:
        for `mma.row.col`, B col-major K×N == the natural row-major weight W[n][k]
        — so NO k-major transpose (the Vulkan coopmat path needed one).
      * `igemm_smem` (EMITTER-generated, `buildIgemmSmem`) — 128x128 block, 4 warps
        (2x2 of 64x64), 128 s32 accumulators/thread, K_STEP=64, 16 KB STATIC
        shared, synchronous staging, 32 MMAs/k-step. 168 regs/thread.
        **@4224x6144x6144: 2.31 ms = 138 TOP/s (Vulkan ~85 => 1.63x).**
        DiT shapes: 7680x6144x6144 4.41 ms (Vulkan 6.63 => 1.50x), 131 TOP/s;
        7680x16384x6144 (mlp gate/up) 13.2 ms, 117 TOP/s; 7680x6144x16384
        (mlp.down) 15.0 ms, 103 TOP/s. ~49% of the 3090's ~284 TOPS int8 peak
        (Vulkan sat at ~30%). And this is BEFORE cp.async / >48 KB dynamic shared
        — pure tiling + static shared already clears the gate. `cuda-i8-test` CLI.
- [x] **1.3b GEMM perf round — cp.async NEUTRAL (compute-bound), GEMM locked.**
      `igemm_pipe` (`buildIgemmPipe`, EMITTER): cp.async.cg double-buffered
      (LDGSTS), K_STEP param. Bit-exact (0 mismatches). @4224x6144x6144 2.32 ms
      (137 TOP/s) vs smem 2.36 — ~1-2%, i.e. NEUTRAL, same lesson as the Vulkan
      double-buffer post-mortem ("this driver already hides load latency"). So
      the GEMM is MMA-issue/compute-bound, not load-bound => deeper K_STEP via
      >48 KB shared won't move it either (skipped for the GEMM; that lever is
      saved for attention/flash where the 48 KB cap was the actual Vulkan
      blocker). GEMM LOCKED at ~135 TOP/s (1.5-1.6x Vulkan, ~49% of the 3090's
      ~284 TOPS int8 peak; cuBLASLt ~60-70% would need ldmatrix + SASS-level
      scheduling we don't have — diminishing, per the plan's time-box).
      DiT shapes (pipe): qkv 7680x6144x6144 4.34 ms (Vulkan 6.63 => 1.53x),
      mlp gate/up 12.9 ms (120 TOP/s), mlp.down 14.2 ms (109 TOP/s).
      Both kernels stay (smem = no-opt-in fallback; pipe = default).
- [x] **1.5 hand-PTX fused prep + full int8 linear — VALIDATED bit-comparably.**
      * `buildPrep(cols)` (EMITTER): one 256-thread block/row, loads x[row] into
        DYNAMIC shared f32 (>48 KB — the lever), parallel radix-4 FWHT per
        256-group (bit-identical to convrot.rotate: each butterfly output is a
        fixed 4-input sum, so parallel order == serial order), *0.0625 normalize
        + per-row abs-max (order-independent max), scale=max(absmax/127,1e-12),
        round-half-away (copysign) quantize -> packed int8 + act_scale.
        KEY: cols=16384 rotates in FULL f32 (66 KB dynamic shared) — the Vulkan
        path was forced to f16 there by the 48 KB cap (its wiring rel ~0.4%);
        CUDA's wiring rel is 3.7e-5 (essentially exact).
      * `irescale` (hand-PTX): y = f32(acc)*act_scale[row]*weight_scale[col].
      * Full linear (prep->igemm_pipe->rescale) vs the `gpu-i8-test` CPU oracle:
        wiring rel 0.000000-0.000052 (bit-exact), int8 accuracy 0.78-0.94% vs f32
        (matches Vulkan's ~0.9%). `cuda-i8-test`.
      * DiT-shape full-linear timing (prep+gemm+rescale, min-of-10):
        qkv 7680x6144x6144   5.17 ms (prep 0.38 / gemm 4.34 / rescale 0.45)
        mlp gate/up 7680x16384x6144  14.5 ms (0.35 / 12.9 / 1.19)
        mlp.down 7680x6144x16384     16.6 ms (1.78 / 14.4 / 0.45)
        vs Vulkan's int8 shared GEMM alone at qkv shape = 6.63 ms => the full
        CUDA linear (5.17) already beats Vulkan's GEMM-only. prep is cheap
        (f32 rotation in >48 KB shared is efficient); rescale not yet fused into
        the GEMM C-store (Stage-A fusion is a known follow-up, ~0.45-1.2 ms).
- [x] **1.4 DECISION GATE: PASS.** Hand-PTX recovers a real, validated fraction of
      the 2x ComfyUI gap: int8 GEMM 1.5-1.6x over Vulkan, full linear correct +
      faster, exercising all three Vulkan-denied levers (>48 KB shared used for
      exact f32 prep; cp.async built [neutral for the compute-bound GEMM];
      explicit mma.sync IMMA tiling = the GEMM win). Continue Phase 1.
- [x] **Stage-A fused rescale (`buildIgemmPipe(..,fuse=true)` -> `igemm_pipe_fused`).**
      y = f32(acc)*act_scale[row]*weight_scale[col] folded into the C-store: no s32
      acc buffer (mlp.up's was ~500 MB), no separate rescale pass. Bit-exact vs the
      oracle (fused rel == unfused). Full linear: qkv 5.15->4.78 ms, mlp gate/up
      14.48->13.31 ms (the 500 MB acc round-trip is gone), mlp.down ~neutral. VRAM
      + bandwidth win regardless of the timing delta.

### M10 Phase 1 — SESSION OUTCOME (2026-07-03): hand-PTX CUDA backend, decision gate PASSED

A complete pure-Zig CUDA Driver-API + hand-emitted-PTX backend was built and
validated in one unattended session. It breaks the Vulkan structural ceilings
and beats the Vulkan int8 path on the DiT GEMM/linear:

| stage | kernel | shape | result |
|-------|--------|-------|--------|
| GEMM | `igemm_pipe` (mma.m16n8k32.s8, cp.async, 128x128 tile) | 4224^3 | **138 TOP/s, 1.5-1.6x Vulkan (~85)**, ~49% of int8 peak |
| GEMM | " | qkv 7680x6144x6144 | 4.34 ms vs Vulkan 6.63 (1.53x) |
| linear | prep + gemm + rescale, vs CPU oracle | validate | wiring rel ~0 (bit-exact), int8 acc 0.9% |
| linear (fused) | prep + `igemm_pipe_fused` | qkv | 4.78 ms full |

What made it work vs Vulkan's ~30%-of-peak coopmat: (1) explicit `mma.sync` IMMA
tiling (the win); (2) for `mma.row.col`, B col-major K×N == the natural row-major
weight, so no k-major transpose; (3) >48 KB dynamic shared lets the prep rotate
cols=16384 in exact f32 (Vulkan was forced to f16). cp.async was NEUTRAL (the
GEMM is MMA-issue-bound, same lesson as the Vulkan double-buffer post-mortem).

Files: `src/gpu/cuda/{cu.zig, ptx.zig, context.zig, kernels.zig}`, `src/gpu/cuda.zig`;
CLI `cuda-test` / `cuda-i8-test`. `zig build test` green; Vulkan path byte-untouched.
Memory: [[cuda-ptx-backend]].

- [x] **1.6 attention on CUDA — DONE (validated end-to-end).** Three kernels:
      * `buildHgemm` (EMITTER) — f16 tensor-core GEMM C(f32)=A(f16)@B(f16)^T,
        `mma.m16n8k16.f32.f16.f16.f32` -> `HMMA.16816.F32`, 128x128 tile / 2x2
        warps / 128 f32 accs / K_STEP=32, adapted from the int8 tiling. rel vs
        f16-CPU = 0.00000 (bit-exact) at 128^3, 128x256x256, 256x128x512.
      * `softmax_row` (hand-PTX) — row softmax with prefolded scale, f32 in ->
        f16 out, ex2.approx (log2e-folded), 2-pass max/sum tree reduction over
        256 threads, pads j>=seq to 0.
      * Full single-head attention (scores=Q@K^T via hgemm -> softmax_row ->
        P@V via hgemm with V pre-transposed to [hd][seq]): **rel vs f32-CPU
        0.00027** (well inside the f16 regime). `cuda-attn-test`.
      Remaining glue for DiT integration: GQA head-batching (gid.z=head, kv=
      head/group), a GPU V-transpose/gather (host-transposed in the test), and
      seq_pad handling — mechanical, not new tensor-core work.

- [x] **1.7 CUDA DiT forward — END-TO-END LIKE-FOR-LIKE IMAGES.** Rather than
      genericize dit_gpu (a large refactor with `.null_handle`/struct-field
      couplings — see the recipe), wrote a dedicated `src/models/dit_cuda.zig`
      driving a `cuda.Backend` (`src/gpu/cuda/backend.zig`) — the GpuBackend
      surface on top of the driver Context + PTX kernels:
      * buffers (tensorCreate/Destroy/Upload/Download/Copy/ensureDeviceBuffer/
        smallBuffer), batch==stream, weight cache by host ptr.
      * `opI8Prep`/`opI8Gemm` (validated int8 fused-rescale GEMM), `opMatmul`
        (naive f32, for the non-int8 first/last), and correctness-first f32
        eltwise kernels in `cuda/elt.zig` (rms_mod fused rmsnorm+AdaLN, qk_rmsnorm,
        rope, one naive online-softmax GQA `attn`, sigmoid_mul, silu_mul,
        gated_add) — all assemble under ptxas sm_86.
      * dit_cuda.forward follows the recipe's fallback op sequence (28 blocks:
        prenorm-mod → qkv/gate int8 GEMMs → qk-norm → rope → attn → sigmoid-gate
        → wo → residual → postnorm-mod → mlp int8 GEMMs → silu → down → residual;
        first/last via opMatmul). Text fusion / timestep / patchify stay on CPU.
      * `cuda-dit-test`: **CUDA DiT vs CPU int8 forward rel RMSE 0.01367** (1.4%,
        int8 + ex2.approx regime); **CUDA 1.58 s vs CPU 9.8 s** (6.2x) at 256px.
      * `pipeline.zig --dit-cuda on`: full CPU-encode → CUDA-DiT → CPU-VAE →
        PNG. 256px/6-step fox: CUDA **1.0 s/step** vs CPU **9.9 s/step** (~10x).
        **Image vs CPU int8: 0.15% mean pixel delta** (max 16/255, 41% pixels
        bit-identical) — like-for-like, indistinguishable. The hand-PTX PTX
        backend now generates real images.

REMAINING (perf / breadth, next session):
- The DiT eltwise/attention kernels are correctness-first (one-thread, naive
  attention) — not yet tuned; the int8 GEMM is the fast part. Full-speed s/step
  needs the tensor-core attention path (opAttnScores/opAttnOut, hgemm+softmax
  primitives already built) + fewer per-step buffer allocs (a Workspace) + the
  f16 int8-GEMM output (c_h16 / pipe_coop_i8_fs16). Larger images (1MP) need the
  head-batched attention (s-buffer cap) — the naive attn is fine at 256-512px.
- Projected tuned end-to-end (GEMM-dominated): Vulkan int8 ~3.9 s/step -> CUDA
  ~3.0-3.3 (~1.2x); ComfyUI ~2 s/step stays ahead (cuBLASLt + flash-attn). The
  hand-PTX ceiling is ptxas-owned scheduling, as predicted.

### M10 Phase 1 — perf pass (2026-07-04): tensor-core attention + parallel elt

Profile-driven (`cuda-dit-test <path> <lat>` prints a per-category + attention
sub-category cuEvent breakdown; sync-per-op). All changes validated: `cuda-dit-test`
rel RMSE cuda-vs-cpu 0.0136 -> 0.0481 (the f16-attention regime, same as Vulkan/
ComfyUI fp16 SDPA; well under the 0.08 gate); `zig build test` green; real 512px
and 1120x1680 fox images coherent.

Batched steady-state s/step (RTX 3090, int8 checkpoint):
| size          | before (naive attn) | after      | note |
|---------------|---------------------|------------|------|
| 256px         | 0.97 s              | **0.12 s** | 8x   |
| 512px         | 6.27 s              | **0.39 s** | 16x  |
| 1024px        | >400 s (timed out)  | **2.12 s** | usable |
| 1408px (~1MP) | ~40 s+              | **5.10 s** | —    |

- [x] **CUDA per-category profiler** (`Backend.profile`/`prof`, cuEvent sync-per-op):
      matmul / prep / attn(gather+scatter) / elt, plus attention sub-timers
      scores / softmax / pv. This drove the whole pass.
- [x] **Tensor-core GQA attention** — the huge win. Replaced the naive
      one-thread-per-(query,head) O(seq²)-serial kernel with hgemm(scores) ->
      softmax_row -> hgemm(P@V). Three new gather/scatter kernels (elt.zig):
      `gather_head` (Q/K -> contiguous [mpad][hd] f16, zero-pad rows>=seq),
      `gather_vt` (V -> transposed [hd][mpad] f16), `scatter_head` (O -> the
      interleaved [seq][heads][hd] out). GQA via KV head = h/group; seq padded
      to mpad (128-multiple). At 512px attn 5595 -> 180 ms (31x).
- [x] **Head-batched attention** (`opAttnTCBatched`, grid.z=G) — the per-head PV
      GEMM has n=hd=128 (grid.x=1, under-occupied); batching G heads per launch
      fills the SMs and collapses ~7·(n_heads/G) launches. G derived so the
      scores+probs scratch fits `attn_scratch_budget` (2 GiB). Batched
      `gather_head_b`/`gather_vt_b`/`scatter_head_b` (one launch/head-group,
      GQA unified via a group_div param); `buildHgemm(batched)` adds a gid.z
      per-head-stride offset; softmax runs flattened over gs·mpad rows (S is
      contiguous [gs][mpad][mpad]). Validated batched-vs-loop-vs-CPU
      (`cuda-attn-cmp`). attn 75 -> 9.7 ms at 256px.
- [x] **Parallel `rms_mod`** (elt) — the naive fused-rmsnorm+AdaLN was
      one-thread-per-row with a serial dim-6144 reduction (only 264 threads =
      2 blocks on 82 SMs at 256px). Rewrote as one block (256 threads) per row
      with a shared tree reduction (`rms_mod_par`, bit-identical math). elt
      311 -> 11 ms at 256px (27x).
- [x] **f16 scores** (`buildHgemm(batched, c_f16)` -> `hgemm_batched_c16`,
      `softmax_row_f16`) — S stored/read f16 (halves the S write + softmax reads,
      the memory-bound cost at large seq). KEY BUG + FIX: real DiT scores exceed
      f16's 65504 max (qk-norm weights) -> Inf -> exp(Inf-Inf)=NaN. FIX (matches
      Vulkan's "Q prescaled by the softmax scale"): the softmax scale is prefolded
      into the scores GEMM C-store (`mul.f32 acc,scale` before the f16 convert),
      so f16 S holds scale·(Q·K) (in range) and the f32 accumulator holds the true
      value; softmax then uses scale=1. (This bug hid behind small random test
      values — `cuda-attn-cmp` now tests std up to 3.0.) attn at 1408px 2910 ->
      2367 ms; step 5.73 -> 5.10 s.
- [x] **Workspace** (`dit_cuda.Workspace`) — the ~12 per-forward device buffers
      (raw cuMemAlloc/free each) moved to a per-run struct shared by both CFG
      sessions (sized for the longer prompt). Small perf delta at these sizes
      (host overhead is mostly the CPU modulation-vector loop + launch gaps) but
      avoids VRAM churn/fragmentation and is required for clean large-image runs.
- [x] **End-to-end**: `generate --dit-cuda on` renders coherent 512px (0.53 s/step,
      5.3 s total incl. CPU VAE) and 1120x1680 (4.98 s/step) images.

Attention sub-profile @1408px (seq~7752): scores 862 / softmax 867 / pv 451 ms.
Scores is shallow-k (k=hd=128) at ~24 TFLOP/s; softmax is bandwidth-bound on 3
S-reads (f16). Both scale ~O(seq²).

RESULT vs baselines @1120x1680: **CUDA ~4.8 s/step** — beats Vulkan (~3.9) and
ComfyUI (~2) at ≤512px, but is ~1.2x BEHIND Vulkan at 1MP (attention: CUDA's
two-pass/3-read-softmax vs Vulkan's fused-online-softmax ~1.4 s). Also: the CUDA
backend keeps all int8 weights resident (no VRAM streaming) — 1120x1680 with the
Vulkan VAE + other-process VRAM OOMs; run VAE on CPU (`--gpu off`) or free the GPU.

REMAINING attention levers (in order of confidence):
- [DONE 2026-07-04 — see "attention fusion" below] Softmax + fused attn-out.
  Landed BOTH the 2-pass-softmax idea AND the fused attn-out in one shot: the
  softmax pass became a single-pass FLASH reduction emitting only per-row
  {max, 1/sum} (softmax_md_f16, one S read, no P write), and the P@V GEMM
  (hgemm_attnout) recomputes P = exp(S-max)/sum from S+MD during its A-staging —
  no P materialization at all. softmax @1MP 928→212 ms; step 5.42→4.68 s.
- scores GEMM: shallow k=128 caps it at ~24 TFLOP/s; now the single biggest
  attention cost (915 ms @1MP). A purpose-built scores kernel (ldmatrix instead
  of the 4×ld.shared.b32 fragment loads, K-staging) or larger tiles could help,
  and ldmatrix would speed the P@V GEMM too. Broad/risky rewrite of the validated
  MMA fragment loads; rated diminishing. Untried.
- CUDA VRAM streaming (LRU weight eviction like the Vulkan path) for 1MP with a
  busy GPU. [DONE 2026-07-04 — see below.]

### M10 Phase 1 — CUDA VRAM streaming (2026-07-04)

Ported the Vulkan weight-streaming design (previously CUDA kept all 14 GiB int8
weights resident → 1120x1680 OOM'd once the Vulkan VAE + other-process VRAM ate
the card). backend.zig: LRU-stamped weight cache, `budgetHeadroom` (live
`cuMemGetInfo` free — sees other processes, the CUDA analog of VK_EXT_memory_budget
— min'd with the `--vram-budget` ceiling), `reserveForWeights` (evict LRU until the
upload fits), `evictOneWeight` (cuStreamSynchronize before cuMemFree), and a
`tensorCreate` OOM-retry backstop. On the fixed block-order walk, LRU eviction =
sequential weight streaming; evicted weights re-upload from the mmap'd checkpoint.
context.zig: alloc distinguishes CUDA_ERROR_OUT_OF_MEMORY; added memGetInfo.
- **KEY FIX** (`CUDA_ERROR_ILLEGAL_ADDRESS` at tight budgets): evictOneWeight must
  never evict the MOST-recently-used weight. opI8Gemm fetches its weight, then its
  weight_scale; when the non-weight footprint (workspace + attn scratch) alone
  exceeds the budget, headroom pins to 0, so the weight_scale fetch's reserve was
  freeing the just-fetched weight → the GEMM read freed memory. (Vulkan dodges this
  because eviction flushes recorded ops already bound to the weight; CUDA launches
  after both fetches, so the MRU is protected in evictOneWeight instead.)
- Validated BIT-IDENTICAL (`cuda-stream-test`): streamed forward == resident, 0/N
  elems differ, at budgets 1 & 2 GiB (256/1024px).
- **RESULT @1120x1680, --vram-budget 2 (apples-to-apples):**
  | backend | resident | streamed | loss |
  |---------|----------|----------|------|
  | CUDA    | 4.98 s   | 5.62 s   | ~13% |
  | Vulkan  | 4.86 s   | 5.86 s   | ~20% |
  CUDA streaming MATCHES/BEATS Vulkan. 1120x1680 with the Vulkan VAE now completes
  (was OOM). The ~10% target is optimistic for the SYNC design — Vulkan itself is
  ~20% at this budget (both upload synchronously; the transfer time serializes with
  compute). Loss shrinks with compute size (256px 650% → 1024px 28-37% → 1MP ~13%).
- ASYNC streaming — INVESTIGATED (2026-07-04), infra built, NOT shipped (needs a
  prefetch thread). The idea: overlap weight re-upload with compute. Findings:
  * The two serializers are the sync UPLOAD (cuMemcpyHtoD) and the per-eviction
    cuStreamSynchronize. Built a transfer stream + per-weight upload events +
    deferred-free eviction (record a compute-stream event at eviction, reclaim the
    buffer lazily via cuEventQuery — no full sync).
  * BLOCKER 1: the checkpoint mmap can't be page-locked (cuMemHostRegister ->
    CUDA_ERROR_NOT_SUPPORTED for the READ_ONLY flag / INVALID_VALUE plain, on a
    read-only file-backed mapping), so cuMemcpyHtoDAsync from it degrades to sync.
  * BLOCKER 2: a driver-pinned staging ring (cuMemAllocHost + memcpy mmap->pinned
    -> async DMA, no thread) measured SLOWER than sync: the driver's cuMemcpyHtoD
    already pipelines its internal staging, while the explicit ring puts a
    single-threaded ~1.4 s/step mmap->pinned memcpy on the critical path.
    Measured: 1024px budget2 50.9% (vs sync 37%); 1408px budget3 20.3% (vs sync
    ~15%). Bit-identical throughout.
  * Vulkan streaming is ALSO synchronous (grep: no transfer queue / prefetch), and
    measured ~20% at budget2 — WORSE than CUDA sync's ~13%. So the "~10% like
    vulkan" target was optimistic; v1 sync already matches/beats Vulkan.
  * BLOCKER 3 (the prefetch thread — BUILT + measured, still worse): implemented
    the full design — a lock-free SPSC prefetch thread (Zig 0.16 removed
    std.Thread.Mutex/Condition, so std.atomic.Value + Thread.yield), a block-ahead
    driver in dit_cuda.forward (prefetchBlock N+1 during block N), deferred-free
    eviction, and in-flight-protect in the evictor. Bit-identical, no hang/race.
    But 1024px budget2: 54.6% (vs sync 37%, no-thread ring 50.9%) — the thread's
    single-threaded mmap→pinned memcpy + cross-thread CUDA driver-lock contention
    + the per-weight cuMemAlloc/Free/event churn (~280 alloc+free/step, present in
    BOTH sync and async but amplified by the thread) never let the DMA overlap win.
  * FINAL CONCLUSION: explicit async weight streaming does NOT beat the driver's
    synchronous cuMemcpyHtoD (internally pipelined) on this platform — verified 3
    ways (direct-pin blocked, no-thread ring worse, thread worse). Production ships
    the SYNC v1 (beats Vulkan). The async infra (cu.zig cuMemAllocHost/
    cuMemcpyHtoDAsync/etc; context.zig staging ring + events; backend prefetch
    thread + deferred-free) stays dormant + documented, OFF in production, exercised
    only by `cuda-stream-test`.
  * CHURN REDUCTION (size-bucketed weight-buffer pool — BUILT + measured, then
    REVERTED): recycle evicted weight buffers by exact size instead of cuMemFree,
    reuse on re-upload instead of cuMemAlloc (pooled bytes counted as reclaimable
    headroom, trimmed to the budget under pressure). Measured WITHIN NOISE: 1024px
    budget2 37%→33.6%, but budget1 28%→32.6% (the trim overhead hurts pure-stream
    budgets where the pool can't reuse). CONFIRMS the streaming loss is
    PCIe-TRANSFER-bound, not churn-bound: the pool recycles buffers but the weight
    DATA is still re-uploaded every step, so the transfer (the real cost) is
    unchanged. Reverted for cleanliness (marginal + mixed). No software technique
    (churn pool, async overlap) meaningfully reduces streaming loss on this
    platform — it is the raw PCIe cost of re-uploading the streamed weights, and
    the only real lever is holding more resident (bigger budget). SYNC v1 ships.

### M10 Phase 1 — attention fusion (2026-07-04): flash softmax + fused attn-out

The ranked attention levers #1 (cheaper softmax) and #2 (eliminate P
materialization) landed together — the second subsumes the first. Same-session
A/B on `cuda-dit-test`, `attn_fused` toggle in backend.zig (default on; the
materialized `softmax_row_f16 → P → hgemm_batched` path stays as the A/B/fallback
reference behind `attn_fused=false`).

- [x] **`softmax_md_f16` (kernels.zig): single-pass FLASH reduction.** Replaces the
      3-read `softmax_row_f16` (max pass + sum pass + write-P pass) with ONE pass
      over S that maintains a per-thread running (m, d) and block-reduces the
      FlashAttention way — M=max(mᵢ), D=Σ dᵢ·exp2((mᵢ-M)·log2e) — emitting only the
      per-row MD table {max, 1/sum} (f32 pair). No P write. Robustness: the running
      max inits to **-FLT_MAX (not -inf)** so combining two empty lanes gives
      m-M = 0 (finite) instead of -inf-(-inf) = NaN; every real score > -FLT_MAX so
      the max is unchanged, and empty lanes (d=0) contribute 0. (Attention seq ≥ 256
      here so every lane has a valid column, but the sentinel keeps it general.)
- [x] **`hgemm_attnout` (buildHgemm `attnout` flag): fused P@V.** The P@V GEMM's A
      operand is the raw scores S (f16); during shared-staging each element is
      turned into a softmax probability P[q][j] = exp2((S-max[q])·log2e)·inv[q] read
      from MD (pad keys j≥seq → 0 via selp), then P@Vt runs on tensor cores exactly
      as before. Zero P materialization — no P write (softmax) and no P read (PV,
      it reads S instead, same bytes). The A-row q = row0+rowq+i·8 is **clamped to
      mpad-1** for the MD lookup so the redundant/pad staging rows (buildHgemm's
      staging over-reads by design) can't fault MD; the transformed garbage they
      write lands in shared slots the MMA ignores (idempotent, same as the plain
      copy). C-store, B(Vt)-staging, tiling all identical to hgemm_batched.
- [x] **Wiring**: opAttnTCBatched branches on `attn_fused`; drops the `attn_p`
      buffer for a tiny `attn_md` ({max,1/sum}), which also lets the head-batch `g`
      ~double (per-head scratch mpad²·2 vs ·4) → **half the launches** (peak VRAM
      unchanged: S grows to fill the 2 GiB budget where S+P used to share it).
- **Validation**: `cuda-attn-cmp` fused-vs-CPU rel bit-for-bit with the materialized
      path to 5 dp (3e-4 std .3 → 1.8e-3 std 3.0 — the f16-S/P rounding dominates
      the flash-vs-linear sum reorder); DiT rel-vs-CPU 0.04744 unchanged; `zig build
      test` green; `cuda-stream-test` bit-identical (0/N); coherent 512px fox
      (scratch_out/fused_fox_512.png).
- **RESULT (same-session A/B, batched steady-state s/step + sync-per-op profile):**
  | size  | before | after | softmax | pv | sync total |
  |-------|--------|-------|---------|-----|-----------|
  | 1024px| 2.246 s| **2.064 s** | 222→54 | 141→161 | 1972→1834 |
  | 1408px| 5.418 s| **4.676 s** | 928→212 | 470→512 | 4999→4346 |
  Softmax −77% (4.4× at 1MP); pv +9% (the exp-in-staging cost, well under the
  softmax saving); scores unchanged. Net −13.7% s/step at 1MP. The +42 ms pv is
  the exp recompute during A-staging (some rows staged/exp'd twice by buildHgemm's
  redundant staging — accepted, dwarfed by the softmax win).
- NEXT attention lever: scores GEMM (now 915 ms @1MP, the biggest attention cost)
  is shallow-k=128 at ~24 TFLOP/s — needs ldmatrix / K-staging / bigger tiles (also
  speeds pv). Broad rewrite of the validated MMA fragment loads; deferred.

### M10 Phase 1 — full CUDA pipeline: encoder + VAE (2026-07-04)

`--backend zig-cuda` now runs the WHOLE pipeline on the hand-PTX CUDA backend —
text encoder, DiT, and VAE — with no Vulkan context at all. (This also replaced
the old three-flag CLI: `--gpu`/`--dit-cuda` collapsed into
`--backend cpu|vulkan|zig-cuda`.) All weights stream through the CUDA weight
cache, so `--vram-budget` degrades to weight streaming across all three stages
and coexists with other GPU workloads via the live `cuMemGetInfo` budget
(verified: budgeted 512px runs stream and complete).

- [x] **fp8-e4m3 GEMM (`opMatmulFp8`).** The encoder's GEMM weights are fp8-e4m3
      (+f32 per-tensor scale); the CUDA backend had no fp8 path. Kept the weights
      fp8 in the cache (4 GiB, streaming-friendly) and decode per GEMM into an f16
      scratch (`dequant_fp8_f16`: the checkpoint's 256-entry e4m3→f32 LUT × scale
      → f16), convert activations f32→f16 (`f32_to_f16`, m→128 pad zeroed), then
      the validated f16 `buildHgemm`. Validated vs a CPU fp8 reference: rel RMSE
      **0.00029** across encoder shapes.
- [x] **CUDA text encoder (`qwen3_cuda.zig`).** The Vulkan `qwen3_gpu` mirror on
      CUDA: 35-layer Qwen3 in one batched submission. fp8 GEMMs (opMatmulFp8),
      RMSNorm/QK-norm reuse `qkNorm`, new `rope_half` (rotate-half RoPE) + `add`
      (residual) kernels, causal flag added to the naive GQA attention (parity-
      first — the prompt-length seq makes the O(seq²) kernel a ~0.2 s one-time
      cost), tap snapshots via `tensorCopy`, embed gather + rope table CPU-side.
      Validated (`cuda-encode-test`) vs CPU encode: **rel RMSE 0.00010**.
- [x] **CUDA VAE decode (`vae_cuda.zig`).** The Vulkan `vae_gpu` mirror on CUDA:
      3x3 convs as banded `im2col` (new kernel, +fused nearest-exact 2x upsample)
      + GEMM; `vae_norm` (new, per-position channel L2 norm + fused silu); `add`
      residuals; the mid-block single head (dim 384) REUSES the DiT tensor-core
      attention (`opAttnTC` with n_heads=1, hd=384 — no VAE-specific attention
      kernel; the Vulkan "3 fake heads" split was a coopmat limitation the CUDA
      MMA doesn't have). `freeAttnScratch` drops the ~seq² scores plane after the
      mid block. Validated (`cuda-vae-test`) vs CPU decode.
      * First cut used the f32 register GEMM for convs — CORRECT but slow (27 s
        @1MP). Added `opConvF16`: f32→f16 zero-padded weight+activation
        (`f32_to_f16_pad2d`, co→128 / k→32 / m→128) → buildHgemm → `bias_compact`
        (strip col pad + add bias); co≥96 convs route through it. **VAE decode
        @1MP 27.3 → 1.23 s (22×)**; @512px 5.66 → 0.25 s. Parity rel RMSE
        1e-5 (f32) → 2.5e-4 (f16 conv rounding, same regime as Vulkan's f16-w
        VAE, well under the 5e-3 gate).
- **End-to-end (all CUDA)**: coherent 512px (4.8 s total: enc 0.5 + DiT 0.53×6 +
      VAE 0.3) and 1024px (19.3 s: 2.09 s/step + VAE 1.1) fox images; `zig build
      test` green.
- Follow-up: the encoder attention is naive-causal-f32 (fine at prompt lengths);
      a tensor-core causal path (mask j>q in softmax_md + hgemm_attnout) would
      only matter for very long prompts.

### M10 Phase 1 — audit-driven perf follow-ups (2026-07-04)

An 8-subsystem audit (each kernel deep-read, every proposed optimization
adversarially verified against the dead-end log above) found the 2x ComfyUI gap
is a kernel-efficiency gap concentrated in the int8 GEMM (~49% of peak) and the
shallow-k attention scores GEMM — both capped by ptxas/driver opacity. cuBLASLt
(the only path that closes it) is blocked by the pure-Zig invariant. What remains
inside the pure-Zig constraint: `ldmatrix` on the GEMM/attention fragment loads
(the real, neutral-risk swing) plus a set of small, bit-identical fusions and one
streaming-policy bug-fix. All the "bigger" ideas (cuBLASLt, true single-pass flash
attention [already shipped via `attn_fused`], cp.async on scores [same-op Vulkan
K-staging was neutral], CUDA Graphs [GPU already 100% util], multi-stream overlap
[one GEMM saturates 82 SMs]) were verified dead or blocked.

Validation per step: `cuda-i8-test` (GEMM/prep vs CPU oracle, bit-exact),
`cuda-dit-test <int8-ckpt> <lat>` (full-forward rel-RMSE + per-category profile),
`cuda-stream-test` (streamed == resident, bit-identical), `bench-matmul` (GEMM
min-of-≥8), `generate --backend zig-cuda` (coherent image). GPU-gated unit tests
need `touch testdata/gpu-tests` (remove after). Measure with the ±40% clock-noise
caveat — min-of-≥8 or whole-step profiles.

Batch A — cheap, safe, bit-identical CUDA wins + the streaming bug-fix:
- [ ] **A1. `opI8Prep`: memset only pad rows `[m..mpad)`, skip when `m==mpad`.**
      The prep kernel fully overwrites rows `0..m-1`; zeroing them in `backend.zig`
      before launch is pure waste. ~17 ms/step @1MP, zero correctness risk.
- [ ] **A2. Cache `timestepVectors` per-schedule-sigma in `dit_cuda.Session`**
      (parity with `dit_gpu`'s `tvs`). `forward` recomputes `timestepVectors(sigma)`
      (reads ~900 MB tproj1) every step on the CPU critical path; precompute for the
      fixed schedule sigmas at `Session.init`, look up in `forward`. ~10-40 ms/step.
- [ ] **A3. Fuse qk-RMSNorm + RoPE into one in-place CUDA pass (per q, per k).**
      Currently 4 separate passes (qkNorm q/k, rope q/k) round-trip q(190 MB)+k
      through VRAM. Fuse to 2 passes. ~18 ms/step, bit-identical (f32 round-trip).
- [ ] **A4. Fold the gated residual add into the int8 GEMM fused C-store**
      (`wo` / `mlp.down`). `gated_add` is a separate naive-f32 kernel; fold
      `fma(mod, y, x)` into `igemm_pipe_fused`'s epilogue. ~29 ms/step, accuracy-neutral.
- [ ] **A5. Fold `rms_mod` + silu/sigmoid gates into `buildPrep`'s shared load.**
      The remaining un-fused-f32 CUDA elt ops; fold into the prep kernel's pre-FWHT
      shared store. ~80-90 ms/step (closes the CUDA elt gap; Vulkan did the f16-chain
      equivalent). Largest of Batch A.
- [ ] **A6. MRU-ward residency instead of LRU for cyclic DiT weight streaming.**
      The block loop is a cyclic reference string; LRU is provably pessimal on a
      loop larger than the cache (~0 hits, streams everything). MRU-ward eviction
      keeps a stable resident subset (~C-1 hits/step). Fixes the non-monotonic
      budget curve (8 GiB slower than 4 GiB). Situational (partial-pressure only),
      but the correct policy. Must protect the per-op MRU fetch window (2 fetches:
      weight then weight_scale). Validate bit-identical via `cuda-stream-test`.

Batch B — the real swing at the gap (higher risk; a neutral result is itself the
answer — that the ~49% floor is the driver's coopmat lowering, not our tiling):
- [ ] **B1. `ldmatrix.x4` s8 fragment loads for the int8 GEMM (`igemm_pipe`).**
      Kills the verified 4-way shared-bank conflict + cuts ~60% of `ld.shared`
      instructions in the MMA inner loop. Target 49% → ~57-62% of int8 peak,
      ~0.3-0.5 s/step @1MP. Gate on `bench-matmul` min-of-≥8; bit-exact vs oracle.
- [ ] **B2. `ldmatrix` fragment loads in `buildHgemm` (attention scores + pv).**
      Same technique on the shallow-k=128 attention GEMMs (scores is the biggest
      attention cost at ~915 ms @1MP). Reuses B1's rewrite. ~0.1-0.26 s @1MP.

- [ ] **1.0 CUDA Driver API bindings (`src/gpu/cuda/cu.zig`).** `std.DynLib` load of
      `libcuda.so.1` + hand-declared externs (mirror `vk.zig`; machine-check
      signatures against the CUDA driver headers, no linking, no `nvcc`): `cuInit`,
      `cuDeviceGet`, `cuCtxCreate/Destroy`, `cuMemAlloc/Free`, `cuMemcpyHtoD/DtoH`,
      `cuModuleLoadDataEx`, `cuModuleGetFunction`, `cuLaunchKernel`,
      `cuFuncSetAttribute`, `cuStreamCreate/Synchronize`, `cuEventCreate/Record/
      Elapsed` (device-side timing, avoids host-clock noise). Pure-Zig preserved.
- [ ] **1.1 Toolchain smoke test.** Hand-write a trivial PTX vector-add string
      (`.version 7.x`, `.target sm_86`, `.address_size 64`), `cuModuleLoadDataEx`
      (driver JITs via built-in ptxas — capture the JIT error log via the
      `CU_JIT_ERROR_LOG_BUFFER` option), launch, verify DtoH. This validates
      bindings→module→launch→readback end-to-end (the analog of M9's first SPIR-V
      kernel). New CLI: `cuda-test`, gated on `testdata/gpu-tests` like `gpu-test`.
- [ ] **1.2 PTX emitter scaffold (`src/gpu/cuda/ptx.zig`).** A small text-assembly
      helper (register decls, labels, loops) — the PTX analog of `coopmat.zig`'s
      word-level SPIR-V `Asm`. Keep kernels as authored PTX strings; the emitter
      just reduces boilerplate. (Alternative considered: Zig's own SPIR-V-style
      backend does NOT target PTX/NVPTX cleanly in 0.16 — hand PTX is the path.)
- [ ] **1.3 Hand-PTX int8 IMMA GEMM — the core Phase-1 experiment.** Ampere int8
      tensor core: `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`. The two
      Vulkan-denied levers to exploit:
      * **>48 KB dynamic shared**: `cuFuncSetAttribute(f,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, N)` — up to ~101 KB on
        sm_86. Lets the A/B s8 tiles + deeper K-staging live in shared (our Vulkan
        int8 GEMM is capped at the 16 KB u32 layout by the 48 KB wall).
      * **`cp.async.cg.shared.global`** (16 B) + `cp.async.commit_group` /
        `wait_group` — real async global→shared overlap, the thing Vulkan
        double-buffering could only fake (it measured neutral there).
      Benchmark on `bench-matmul`'s DiT shapes (4224x6144x6144 and the
      7680x{6144,16384} cases) vs the Vulkan 85-TOPS baseline. **This single number
      is the Phase-1 verdict.** Expectation: > 85, likely < cuBLASLt (ptxas owns
      scheduling; matching cuBLAS needs SASS control we don't have).
      Fork if it stalls: try `wgmma`-style larger tiles, more `cp.async` pipeline
      stages, and swizzled shared layouts — but cap the effort (this is
      re-implementing a slice of CUTLASS by hand; diminishing returns are expected).
- [ ] **1.4 Decision gate.** If 1.3 doesn't beat Vulkan by a margin that would
      matter end-to-end (say ≥1.3x on GEMM), STOP Phase 1 here and jump to Phase 2
      — the pure-Zig ceiling is then demonstrably not enough, which is itself a
      publishable result for [[comfyui-gap-assessment]]. If it does: continue.
- [ ] **1.5 Hand-PTX fused prep** (rotate FWHT + per-row abs-max + quantize),
      porting `buildFusedPrepI8`'s design — now with >48 KB shared, the cols=16384
      (mlp.down) row fits without the f16-shared compromise, and f32 rotation is
      back on the table (accuracy headroom). Validate vs CPU replica.
- [ ] **1.6 Hand-PTX attention.** Port the two-pass scores/softmax/PV, OR attempt
      real flash-attention now that >48 KB shared makes the K+Q+S tiling that was
      rejected under Vulkan's 48 KB cap actually fit (that was the explicit
      "open idea" in the Vulkan attention post-mortem — CUDA removes the blocker).
- [ ] **1.7 Backend seam + `dit_gpu` port** (the refactor above), then the real
      measurement: full int8 s/step on CUDA vs Vulkan vs ComfyUI ~2 s/it, at
      1120x1680, via `generate --gpu cuda --profile on`. Reuse the bit-comparable
      image validation + fp8/int8 parity gates.

### Phase 2 — library-backed CUDA backend (`--backend cuda`): cuBLASLt + cuDNN

The 4th backend. Phase 1 (hand-PTX) recovered ~1.5-1.6x on the GEMM but plateaus
at ~49% of int8 peak — the residual to ComfyUI (~2 s/it) is the closed libraries
ComfyUI itself runs (cuBLASLt IMMA near peak + cuDNN/flash-attn). Phase 2 links
those. Decisions (2026-07-08):

- **dlopen, not a build flag.** `libcublasLt.so.13` and `libcudnn.so.9` load at
      runtime via `std.DynLib` + hand-declared externs — the *identical* mechanism
      as `libcuda`/`libvulkan` (no `nvcc`, no headers, no link step). Missing `.so`
      degrades gracefully like the other GPU backends. Keeps the project's own
      "pure-Zig = runtime-loaded, hand-declared" definition; the only new thing is a
      closed-source *math library* dependency (vs just the driver), documented in
      README/CLAUDE.md. Verified present: cuBLASLt 13.2 (`/usr/local/cuda`), cuDNN
      9.23.2 (full graph stack incl. the fused-SDPA engines).
- **New `--backend cuda`** beside `--backend zig-cuda` (hand-PTX). Both drive the
      SAME `dit_cuda`/`qwen3_cuda`/`vae_cuda` orchestrators through the existing
      `cuda.Backend` seam; `cuda` sets `kernels = .libs` and the heavy op methods
      branch to the library path. zig-cuda stays intact for the comparison table.
- **Full scope**: cuBLASLt GEMM (int8 + fp8/f16) + cuDNN fused SDPA attention +
      cuDNN conv for the VAE.

**The seam is already built.** `cuda.Backend` (backend.zig) is the GpuBackend
abstraction the orchestrators use; only these methods swap their compute (same
signatures, so `dit_cuda`/etc. are untouched):

| op method | `.hand_ptx` (today) | `.libs` (new) |
|-----------|---------------------|---------------|
| `opI8Gemm` | `igemm_pipe_fused` (~49% peak) | cuBLASLt IMMA `CUDA_R_8I`→`R_32I` |
| `opMatmulFp8` / f16 GEMM | `buildHgemm` | cuBLASLt `R_16F`→`R_32F` |
| `opAttnTCBatched` | hgemm+`softmax_md`+fused PV | cuDNN fused SDPA |
| `opConvF16` (VAE) | im2col+hgemm | `cudnnConvolutionForward` |

Everything else — weight cache + LRU/streaming, the int8 **prep** (rotate FWHT +
per-row quantize) and **rescale**, all `elt.zig`, profiler — stays byte-identical.
cuBLASLt does ONLY the GEMM; our prep/rescale wrap it. Because s8·s8→s32 is exact
integer math, the int8 GEMM result is **bit-identical to the CPU integer oracle**
(`cuda-i8-test` wiring rel ~0); f16/fp8 GEMM and attention land in the f16 regime.

- [x] **2.0 dlopen bindings + backend/CLI wiring.** `src/gpu/cuda/cublaslt.zig`
      (DynLib `libcublasLt.so.13`, create/destroy/version + the matmul-descriptor
      API, status→string, dtype/compute/order constants) and
      `src/gpu/cuda/cudnn.zig` (DynLib `libcudnn.so.9`, create/destroy/version/
      setStream/errorString). Added `Backend.KernelMode`/`Libs` + `initLibs` (loads
      libs, creates handles bound to our stream, allocs a 32 MB cuBLASLt
      workspace); `deinit` frees them. `--backend cuda` in `pipeline.Backend` +
      `main.zig` help; `cuda-libs-test` smoke CLI. Ops still hand-PTX until 2.1 —
      `--backend cuda` == `zig-cuda` behaviorally for now, but with the libs loaded
      and reporting versions.
- [x] **2.1 cuBLASLt int8 IMMA GEMM (the headline number) — VERDICT: DECISIVE.**
      `opI8Gemm` routes to `cublasLtMatmul` in `.libs` mode; prep + `irescale`
      (per-row act-scale × per-col weight-scale, into a reintroduced s32 acc
      scratch) stay ours. KEY simplification vs the plan's expectation: **no COL32
      transform needed** — the GEMM is cuBLASLt's ordinary-COL TN case
      (`opA(W)=T, opB(A)=N`, `COMPUTE_32I`, s32 D). Our natural layouts map
      directly: weight W[n][k] row-major == col-major [k][n] (the A operand,
      transposed); prepped activation A[m][k] == col-major [k][m] (B, non-T); the
      s32 D lands col-major [n][m] == row-major [m][n], exactly what `irescale`
      consumes. Heuristic-selected algo + descriptors are cached per (n,m,k)
      (`LtPlan`), so the timed matmul is a pure enqueue and the in-DiT path pays
      the heuristic once. `cuda-libs-i8-test` (new CLI): **0 mismatches vs the CPU
      integer oracle** (bit-exact, as predicted for exact s8·s8→s32), then min-of-N
      TOP/s at the DiT shapes (RTX 3090):
      | shape (m×n×k)        | cuBLASLt   | hand-PTX igemm_pipe | Vulkan |
      |----------------------|------------|---------------------|--------|
      | 4224×6144×6144       | 259.5 TOP/s (1.23 ms) | 138 (2.31 ms) | ~85 |
      | 7680×6144×6144 (qkv) | 262.4 TOP/s (2.21 ms) | 131 (4.34 ms) | — |
      | 7680×16384×6144 (up) | 264.4 TOP/s (5.85 ms) | 117 (12.9 ms) | — |
      | 7680×6144×16384 (dn) | 302.3 TOP/s (5.11 ms) | 103 (14.2 ms) | — |
      **~1.9–2.9× over hand-PTX, ~3× over Vulkan, ~91–106% of the ~284-TOPS int8
      peak** — i.e. the GEMM gap to ComfyUI is essentially CLOSED (cuBLASLt IS
      ComfyUI's kernel). The DiT is now attention/prep/elt-bound, not GEMM-bound;
      confirm at the s/step level in 2.3.
- [x] **2.2 cuBLASLt f16/fp8 GEMM — DONE (unit-validated).** `ltPlan`
      generalized over an `LtKind {i8, f16}` (shared TN layout mapping; f16 uses
      R_16F A/B, R_32F D, `COMPUTE_32F` = HMMA with f32 accumulate) + `ltRun`;
      `ltMatmulF16` is the drop-in for the hand-PTX `buildHgemm`. Routed
      `opMatmulFp8` (encoder fp8: decode→f16 stays ours, GEMM→cuBLASLt) and
      `opConvF16` (VAE convs) through it in `.libs` mode; the hand-PTX path stays
      as the `.hand_ptx` else-branch. DiT first/last stay on the naive f32 kernel
      (`opMatmul`) — tiny, low value. `cuda-libs-f16-test` (new CLI): **rel vs a
      CPU f32-accumulate reference 1e-6** (f16 inputs widen exactly; only
      reduction order differs), and min-of-N: **65.8 TFLOP/s @4224³** (vs Vulkan
      coop 51.7, ~93% of the ~71 TF/s f32-acc f16 peak), 57.3 @ enc-mlp, 62.1 @
      vae-conv shapes. NOTE: end-to-end validation (encoder/VAE via
      `cuda-encode-test`/`cuda-vae-test`/`generate`) needs the checkpoints, which
      are ABSENT in the current checkout — deferred to 2.3, to run where the
      14 GB int8 DiT + encoder + VAE are present.
- [x] **2.3 measure GEMM-only — DONE, attention-bound confirmed.** Ran on the
      real int8 checkpoint (`~/genai/comfyui/models/...krea2CenterSemiraw_v10Int8`,
      see [[model-checkpoint-paths]]; NOT in the repo). `cuda-dit-test <ckpt> <lat>
      [libs]` gained a `libs` switch to profile the cuBLASLt backend head-to-head
      vs hand-PTX in one session.
      * **Correctness**: full libs DiT forward vs CPU int8 ref rel RMSE **0.04519**
        (f16 regime, == hand-PTX's 0.048); full `generate --backend cuda` 512px
        renders a coherent AURORA-UNICORNS image (encoder fp8→cuBLASLt-f16, DiT
        int8→cuBLASLt, VAE conv→cuBLASLt-f16 all exercised): 0.59 s/step, 5.9 s
        total incl. loads. scratch_out/cuda_2p3_512.png.
      * **1024px (lat=128, seq~4104) same-session profile:**
        | category | hand-PTX | cuBLASLt libs |
        |----------|----------|---------------|
        | matmul   | 943.9 ms | **496.6 ms (1.90×)** |
        | attn (scores 263 + softmax 55 + pv 163) | 465 ms | 481 ms (unchanged) |
        | elt / prep / gather | ~330 ms | ~332 ms (unchanged) |
        | **s/step (batched)** | **2.011 s** | **1.548 s (1.30×)** |
      The 2.1 GEMM win (~1.9×) lands intact in-DiT; the ~23% s/step drop is exactly
      that. **Attention (scores/pv/softmax, hand-PTX, O(seq²)) is now the largest
      category (~480 ms) — it overtakes matmul.** So 2.4 (cuDNN fused SDPA) is the
      real next lever, and matters more at 1120x1680 (seq~7661) where attention
      dominates hardest. Plan sequencing (GEMM first, then attention) confirmed.
- [x] **2.4 cuDNN fused SDPA attention — DONE, ~80× on the attention GEMMs.**
      Used the dedicated `CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR` (op 41) —
      a single fused flash-attention node, NOT a hand-built softmax subgraph
      (huge de-risk). Headers: the `nvidia-cudnn-cu12` wheel headers in
      `~/genai/comfyui/nvenv` (9.13; backend enums stable across 9.x) — no apt
      needed. `cudnn.zig` gained the backend-graph bindings (Create/Set/Get/
      Finalize/Execute) + a `SdpaPlan` (tensors Q/K/V/O + by-value f32 scale →
      SDPA op → operation graph → HEUR_MODE_A → engine-config → execution plan,
      workspace-sized; cached per shape). `opAttnCudnn` converts the DiT's f32
      [seq][heads][hd] q/k/v to f16 (new `f16_to_f32` elt for O back), runs the
      op, converts O back — no per-head gather/scatter, no S materialization, no
      seq padding, **native GQA**.
      * **Isolation test (`cuda-libs-attn-test`)**: MHA + GQA (8/2) vs CPU rel
        **3e-4**; timing at DiT shapes — **1024px (s=4104, 48/12) 6.07 ms** vs
        hand-PTX ~481 ms (**~80×**); 1408px (s=7752) 22.8 ms vs ~1600 ms (~70×);
        ~68 TFLOP/s ≈ f16 tensor peak.
      * **In-DiT (`cuda-dit-test <ckpt> 128 libs`)**: rel vs CPU **0.04525**
        (unchanged from hand-PTX 0.04519); coherent 512px `generate --backend
        cuda` image (scratch_out/cuda_2p4_512.png). 1024px s/step **1.548 →
        1.257 (1.23×; 1.60× total vs hand-PTX 2.011)**. Attention category
        480 → 233 ms — the residual is the f32↔f16 conversions around the op
        (raw SDPA ~6 ms/block); fusing f16 into rope/qknorm (Vulkan-c16-style)
        is the follow-up, and cuDNN's edge grows at higher res (seq²).
- [x] **2.5 cuDNN conv for VAE — DONE (as predicted, marginal).** `cudnn.zig`
      gained the legacy conv-API bindings + a `ConvPlan` (NHWC f16 X/W → f16 Y,
      3×3 pad-1 stride-1 cross-correlation, `IMPLICIT_PRECOMP_GEMM`, tensor-op
      math — no im2col materialization). KEY: the checkpoint weight is stored
      `[co][kh][kw][ci]` (wan_vae.zig) and activations are channel-last
      `[h·w][ci]` — both EXACTLY cuDNN's NHWC layout, so zero layout
      reconciliation. `opConvCudnn` (convert src+weight f32→f16, conv, new
      `bias_add_f16` elt folds the per-channel bias into the f32 dst); wired into
      `vae_cuda.conv` for the big (co≥96) NON-upsample 3×3 convs in `.libs` mode.
      Upsample convs keep the fused im2col-2×-resample path (cuDNN can't fuse the
      resample); they still use cuBLASLt for the GEMM. Validated end-to-end:
      coherent 512px `generate --backend cuda` image (scratch_out/cuda_2p5_512.png;
      VAE conv errors would be glaring — none), decode 0.6 → 0.4 s @512px.
      Marginal (VAE is a one-time cost), exactly as scoped.
- [x] **2.6 deliverable — DONE. The gap is closed from ~2.4× to 1.42×.**
      Same-session `generate --backend {cuda,zig-cuda}` at 1120×1680, int8 convrot,
      8 steps, steady-state (steps 2–8 flat); coherent production-quality poster
      (scratch_out/cuda_2p6_1120x1680.png — crisp legible "AURORA UNICORNS" text).

      | backend                 | int8 GEMM (DiT shape) | full s/step | vs ComfyUI |
      |-------------------------|-----------------------|-------------|-----------|
      | Vulkan                  | ~85 TOPS              | ~4.86 s     | ~2.4×     |
      | CUDA hand-PTX (zig-cuda)| ~135 TOP/s (~49%)     | 4.76 s      | ~2.4×     |
      | **CUDA + cuBLASLt/cuDNN** | **259–302 TOP/s (~peak)** | **2.83 s** | **1.42×** |
      | ComfyUI int8            | cuBLASLt (near peak)  | ~2.0 s      | 1.0×      |

      **Verdict on the original M10 question**: the closed libraries recover most
      of the gap — **1.68× over our own hand-PTX in the same session**, GEMM at
      hardware peak and attention at the flash kernel. The residual ~1.42× to
      ComfyUI is NOT the heavy kernels (those now match); it's **orchestration /
      fusion** — the un-fused f32 eltwise chain (~267 ms/step at 1024px) and the
      f32↔f16 conversions around the cuDNN attention (ComfyUI fuses these into its
      kernels). So the lean-runtime thesis lands qualified: our thinner dispatch
      does NOT yet edge ComfyUI, because ComfyUI's advantage was never dispatch
      overhead — it's kernel fusion we haven't matched. Closing the last 1.4×
      means the eltwise-fusion follow-ups below, not more library calls.
      Follow-ups (bit-identical, deferred): (1) fuse f16 conversion into
      rope/qknorm so cuDNN attention reads f16 directly (Vulkan-c16-style) —
      removes the ~2 ms/block converts; (2) fuse the int8 rms_mod/silu/sigmoid
      f32 eltwise chain (the biggest remaining category); (3) cache the cuDNN conv
      ConvPlan per shape.
- [~] **2.7 f16-activation-chain redesign — Stage A (MLP) landed, MEASURED NEUTRAL.**
      Built the f16 MLP sub-chain: `irescale_h16` (int8 GEMM → f16), `silu_mul_h16`,
      `buildPrep(in_f16)` (prep reads f16), wired in `dit_cuda` behind
      `mlp_f16 = kernels==.libs and !is_i4`. Validated: rel-vs-CPU 0.04668 (in
      regime), coherent 1120×1680 poster. **Clean same-session A/B @1120×1680:
      f16 MLP 2.77 vs f32 MLP 2.79 s/step — ~0.7%, NEUTRAL.** Kept (correct,
      halves MLP activation DRAM for memory-pressured regimes), but the hypothesis
      is FALSIFIED: the "elt = 267 ms" in the sync-per-op profile is an artifact
      (a device sync after each of ~280 elt ops/step); the real BATCHED eltwise
      cost is small. So follow-ups (1)/(2) above won't close the gap. CORRECTED
      residual: the 1.42× is matmul (cuBLASLt ~peak) + SDPA (cuDNN ~peak) — both
      == ComfyUI's kernels, immovable — plus the int8 prep + s32→f32 `irescale`
      round-trip (cuBLASLt can't fuse per-row×per-col dequant into its IMMA
      epilogue) plus per-step host/launch overhead (batched wall ≫ device
      sync-sum). Real levers (larger/uncertain): (a) fuse dequant into the GEMM
      epilogue — NOT via CUTLASS (header-only C++, needs nvcc build-dep, no
      dlopen'able .so; cuBLASLt already IS NVIDIA's CUTLASS-based GEMM) but via a
      cuDNN backend-graph int8-MATMUL→POINTWISE(row)→POINTWISE(col) fused op,
      staying pure-dlopen; (b) CUDA Graphs for the host overhead.
- [x] **2.7b cuDNN fused int8 GEMM+dequant — BUILT + VALIDATED, MEASURED NEUTRAL
      (the pure-dlopen ceiling).** `cudnn.MatmulDequantPlan`: one backend-graph op
      (int8 MATMUL -> s32 virtual -> POINTWISE mul act_scale[row] -> POINTWISE mul
      weight_scale[col] -> f32/f16 D), the dlopen alternative to a CUTLASS epilogue
      (weight [n][k] viewed as [k][n] via strides). Cached per (m,n,k,d_f16);
      `use_fused_i8` toggle in `opI8GemmLibs`. `cuda-libs-i8fused-test`: **rel vs
      CPU 0.000000 (bit-exact)**; in-DiT rel 0.04547; coherent image. The irescale
      diagnostic (skip the pass -> valid timing) confirmed it IS a real batched
      cost: **1120x1680 2.55 s/step with irescale skipped vs 2.77 with (~0.22 s,
      ~8%)** — unlike the eltwise. BUT the clean same-session A/B: **fused 2.72 vs
      cuBLASLt+irescale 2.71 s/step — NEUTRAL.** The fusion removes the ~0.22 s
      round-trip, but **cuDNN's int8 matmul is ~0.22 s/step slower than cuBLASLt's
      IMMA**, canceling exactly. CONCLUSION: within pure-dlopen you get peak GEMM
      (cuBLASLt, no fusion) XOR fused dequant (cuDNN, slower GEMM), never both —
      they net to a wash. Closing the last 1.42x needs a CUTLASS-class fused-
      epilogue GEMM (nvcc build-dep, out of model) or CUDA Graphs for host overhead
      (untried). **1.42x is the effective floor for the pure-dlopen library
      backend.** Ships cuBLASLt+irescale (proven, equal speed); the fused graph
      stays dormant + documented (`use_fused_i8=false`).

### Risks / notes (Phase 2)

- **IMMA layout (COL32)** is the one real int8 wrinkle — mitigated by
  transform-once-at-upload, which fits the existing weight-cache/streaming pattern.
- **cuDNN backend graph API** is verbose and version-sensitive — mitigated by the
  cuBLASLt-batched-GEMM attention fallback (2.4).
- **VRAM accounting**: cuBLASLt/cuDNN allocate workspace *outside* our
  `device_used` accountant → the streaming budget under-counts. Register their
  workspaces or reserve headroom for them.
- **Version skew**: dlopen the explicit `/usr/local/cuda/lib64/libcublasLt.so.13`
  ahead of the system `.so.12`.
- **Pure-Zig line**: crossed deliberately for `--backend cuda` (closed math libs,
  dlopen'd); CPU/Vulkan/zig-cuda stay pure and default. Same dlopen mechanism as
  the driver, so no build-time dependency is added.
- Clock-governor + Xid 109 watchdog caveats from [[gpu-perf-lab-notes]] apply to
  CUDA identically; use `cuEvent` device timing and keep submissions bounded.

## Context

TensorPencil is a pure-Zig (0.16.0, no C deps) text-to-image inference engine. This plan is laser-focused on one goal: **run Krea 2 end-to-end on the CPU**, producing a correct image from a text prompt, then move to a **Vulkan backend** for usable speed. The repo is currently the stock `zig init` scaffold; everything below is greenfield.

The three checkpoints are already in `models/`:

| File | What it is | Size | dtypes |
|---|---|---|---|
| `diffusion_model/krea2CenterSemiraw_v10Fp8.safetensors` | Krea 2 DiT (~12.3B params) | 11.9 GiB | F8_E4M3 (raw, no scales) |
| `diffusion_model/krea2CenterSemiraw_v10Int8.safetensors` | Krea 2 DiT (int8 ConvRot) | 14 GiB | I8 `W_rot` + F32 `weight_scale [rows,1]` per-row + `comfy_quant` U8 marker; norms/small linears BF16/F32; keys under `model.diffusion_model.` |
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
