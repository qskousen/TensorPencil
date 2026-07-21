//! Shared ggml (llama.cpp tensor library) C bindings, available as
//! `@import("ggml")` when built with ggml enabled (`-Dggml`, default on). Built
//! from the fetched `ggml` package dependency (see build.zig.zon / build_ggml.zig)
//! with AVX2/FMA/F16C/BMI2 + CPU-repack flags and linked into every artifact
//! that uses the TensorPencil module. The project depends on ggml for its fast
//! CPU quant kernels; with `-Dggml=false` this module is not imported and the
//! GGUF block-quant paths return error.QuantBackendUnavailable (see quants.zig).

pub const c = @cImport({
    @cDefine("NDEBUG", "1");
    @cInclude("ggml.h");
    @cInclude("ggml-cpu.h");
});
