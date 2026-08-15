- begin filling in holes in the capabilities grid (BACKEND.md)
- add more sampling methods
- make integration tests shorter?
- gui: studio (image_view) still uses its own form layout; bring the parameter form onto the shared chip/section primitives
- prefill (at least on cpu) isn't cancellable between tokens
- prefill doesn't try to resize when more ram is available
- even a tiny bit of offloading of gemma4 31b is extremely slow
- let's evict the whole diffusion model if needed to fit the llm on the gpu when there's no contention
- diffusion model weights are "bouncing" during steps, vram-wise
- there's no good visiblity of "how much of the model is in vram" for either side
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
