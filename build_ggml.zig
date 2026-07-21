const std = @import("std");

// Source paths are relative to the ggml package root. The package is supplied
// as a fetched dependency (see build.zig.zon `.ggml`, formerly a git submodule
// under vendor/ggml), so all files are resolved via `ggml_dep.path(...)` — this
// keeps the build working for external consumers who obtain us through
// `zig fetch` (git submodules are not fetched by the Zig package manager).

// ── Compiler flag arrays ─────────────────────────────────────────────────────

const base_flags = [_][]const u8{
    "-O3",
    "-fno-sanitize=undefined",
    "-ffast-math",
    "-fno-finite-math-only", // keeps inf/nan behaviour sane despite ffast-math
    "-Wno-macro-redefined",  // _GNU_SOURCE is set both by Zig and by GGML headers
    // Suppress any AVX512 paths Zig's native CPU detection might activate.
    // These are unset globally because ggml.c and other base files have
    // AVX512 code paths that require AVX512-specific headers we're not linking.
    "-U__AVX512F__",
    "-U__AVX512BF16__",
    "-U__AVX512VNNI__",
    "-U__AVX512VBMI__",
    "-U__AVX512CD__",
    "-U__AVX512BW__",
    "-U__AVX512DQ__",
    "-U__AVX512VL__",
};

const base_c_flags   = base_flags ++ [_][]const u8{ "-std=c11" };
const base_cpp_flags = base_flags ++ [_][]const u8{ "-std=c++17" };

// Compiler ISA flags — tell the compiler it may EMIT these instructions
const x86_isa_flags = [_][]const u8{
    "-msse4.2",
    "-mavx",
    "-mavx2",
    "-mfma",
    "-mf16c",
    "-mbmi2",
};

// GGML's own feature defines — tell GGML's #ifdefs which paths to compile
// Both sets are required: ISA flags alone are not enough.
const x86_ggml_defines = [_][]const u8{
    "-DGGML_SSE42",
    "-DGGML_AVX",
    "-DGGML_AVX2",
    "-DGGML_FMA",
    "-DGGML_F16C",
    "-DGGML_BMI2",
};

const x86_c_flags   = base_c_flags   ++ x86_isa_flags ++ x86_ggml_defines;
const x86_cpp_flags = base_cpp_flags ++ x86_isa_flags ++ x86_ggml_defines;

// ── Source file lists (relative to the ggml package root) ────────────────────

// Group 1 — ggml-base: pure logic, no SIMD
const base_c_sources = [_][]const u8{
    "src/ggml.c",
    "src/ggml-alloc.c",
    "src/ggml-quants.c",
};
const base_cpp_sources = [_][]const u8{
    "src/ggml.cpp",
    "src/ggml-backend.cpp",
    // Added in recent ggml: the "meta" backend buffer (deferred/no-alloc tensor
    // contexts) referenced by ggml-backend.cpp / ggml-alloc.c.
    "src/ggml-backend-meta.cpp",
    "src/ggml-opt.cpp",
    "src/ggml-threading.cpp",
    "src/gguf.cpp",
};

// Group 2 — ggml-backend-reg: needs C++ exceptions, no SIMD
// Intentionally separate so we never pass -fno-exceptions to it.
const reg_cpp_sources = [_][]const u8{
    "src/ggml-backend-reg.cpp",
};

// Group 3 — ggml-cpu: shared CPU kernel files (get SIMD flags on x86)
const cpu_c_sources = [_][]const u8{
    "src/ggml-cpu/ggml-cpu.c",
    "src/ggml-cpu/quants.c",
};
const cpu_cpp_sources = [_][]const u8{
    "src/ggml-cpu/ggml-cpu.cpp",
    "src/ggml-cpu/repack.cpp",
    "src/ggml-cpu/hbm.cpp",
    "src/ggml-cpu/traits.cpp",
    "src/ggml-cpu/binary-ops.cpp",
    "src/ggml-cpu/unary-ops.cpp",
    "src/ggml-cpu/vec.cpp",
    "src/ggml-cpu/ops.cpp",
    // AMX files compile fine without AMX flags — they just won't activate.
    // Include them so the symbol table is complete on x86 builds.
    "src/ggml-cpu/amx/amx.cpp",
    "src/ggml-cpu/amx/mmq.cpp",
};

// Group 3 — arch-specific CPU sources
const cpu_x86_c_sources = [_][]const u8{
    "src/ggml-cpu/arch/x86/quants.c",
};
const cpu_x86_cpp_sources = [_][]const u8{
    "src/ggml-cpu/arch/x86/repack.cpp",
};

const cpu_arm_c_sources = [_][]const u8{
    "src/ggml-cpu/arch/arm/quants.c",
};
const cpu_arm_cpp_sources = [_][]const u8{
    "src/ggml-cpu/arch/arm/repack.cpp",
};

// ── Public API ───────────────────────────────────────────────────────────────

/// Build a static GGML library and link it into `exe`.
pub fn link(
    b: *std.Build,
    exe: *std.Build.Step.Compile,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    ggml_dep: *std.Build.Dependency,
) void {
    const lib = buildLib(b, target, optimize, ggml_dep);
    exe.root_module.linkLibrary(lib);
}

/// Build and return the static GGML library from the fetched `ggml_dep`.
/// Use this if you need to inspect or further configure the library step.
pub fn buildLib(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    ggml_dep: *std.Build.Dependency,
) *std.Build.Step.Compile {
    const lib = b.addLibrary(.{
        .name = "ggml",
        .linkage = .static,
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libcpp = true,
        }),
    });

    addSources(lib, target, ggml_dep);
    return lib;
}

// ── Internal implementation ──────────────────────────────────────────────────

fn addSources(
    lib: *std.Build.Step.Compile,
    target: std.Build.ResolvedTarget,
    ggml_dep: *std.Build.Dependency,
) void {
    const arch = target.result.cpu.arch;
    const os   = target.result.os.tag;
    const is_x86 = arch == .x86_64 or arch == .x86;
    const is_arm = arch == .aarch64;

    // Include paths (into the fetched ggml package).
    lib.root_module.addIncludePath(ggml_dep.path("include"));
    lib.root_module.addIncludePath(ggml_dep.path("src"));
    lib.root_module.addIncludePath(ggml_dep.path("src/ggml-cpu"));

    // Platform macros — use addCMacro rather than -D flags to avoid
    // the "macro redefined" warning from Zig's built-in definitions.
    switch (os) {
        .linux   => lib.root_module.addCMacro("_GNU_SOURCE",            ""),
        .windows => lib.root_module.addCMacro("_CRT_SECURE_NO_WARNINGS",""),
        .macos   => lib.root_module.addCMacro("_DARWIN_C_SOURCE",       ""),
        else     => {},
    }
    // Required by GGML to emit version/commit strings
    lib.root_module.addCMacro("GGML_VERSION", "\"0.0.0\"");
    lib.root_module.addCMacro("GGML_COMMIT",  "\"unknown\"");
    // Activates weight repacking optimisation in the CPU backend
    lib.root_module.addCMacro("GGML_USE_CPU_REPACK", "");

    // Windows-specific: GGML uses __attribute__((format(...))) which MSVC
    // doesn't understand; the macro definition below disables it.
    if (os == .windows) {
        lib.root_module.addCMacro("GGML_ATTRIBUTE_FORMAT(...)", "");
    }

    // All C++ files need libc++ (satisfies exception/RTTI runtime symbols,
    // including the SEH glue on Windows that was causing link errors).

    const addC = struct {
        fn f(l: *std.Build.Step.Compile, dep: *std.Build.Dependency, files: []const []const u8, flags: []const []const u8) void {
            for (files) |sub| l.root_module.addCSourceFile(.{ .file = dep.path(sub), .flags = flags });
        }
    }.f;

    // ── Group 1: ggml-base (no SIMD) ────────────────────────────────────────
    addC(lib, ggml_dep, &base_c_sources,   &base_c_flags);
    addC(lib, ggml_dep, &base_cpp_sources, &base_cpp_flags);

    // ── Group 2: backend registry (C++ exceptions required, no SIMD) ────────
    addC(lib, ggml_dep, &reg_cpp_sources, &base_cpp_flags);

    // ── Group 3: CPU backend (SIMD flags on x86) ────────────────────────────
    const c_flags_cpu   = if (is_x86) &x86_c_flags   else &base_c_flags;
    const cpp_flags_cpu = if (is_x86) &x86_cpp_flags  else &base_cpp_flags;

    addC(lib, ggml_dep, &cpu_c_sources,   c_flags_cpu);
    addC(lib, ggml_dep, &cpu_cpp_sources, cpp_flags_cpu);

    // Arch-specific sources
    if (is_x86) {
        addC(lib, ggml_dep, &cpu_x86_c_sources,   &x86_c_flags);
        addC(lib, ggml_dep, &cpu_x86_cpp_sources, &x86_cpp_flags);
    } else if (is_arm) {
        addC(lib, ggml_dep, &cpu_arm_c_sources,   &base_c_flags);
        addC(lib, ggml_dep, &cpu_arm_cpp_sources, &base_cpp_flags);
    }
}
