//! Shared ggml (llama.cpp tensor library) C bindings, available to the whole
//! codebase as `@import("ggml")`. Built from the vendor/ggml submodule with
//! AVX2/FMA/F16C/BMI2 + CPU-repack flags (see build_ggml.zig) and linked into
//! every executable/test. The project deliberately depends on ggml for its
//! fast CPU quant kernels — pure-Zig fallbacks remain where useful.

pub const c = @cImport({
    @cDefine("NDEBUG", "1");
    @cInclude("ggml.h");
    @cInclude("ggml-cpu.h");
});
