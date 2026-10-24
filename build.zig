const std = @import("std");
const Translator = @import("translate_c").Translator;
const ggml_build = @import("build_ggml.zig");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const is_windows = target.result.os.tag == .windows;

    // The gated suite is real models and real kernels, so Debug makes it ~6x
    // slower for nothing: `-Dintegration` therefore defaults to ReleaseSafe,
    // which keeps every assert and safety check the tests rely on. The fast unit
    // suite stays Debug (its cost is compile time, not run time). An explicit
    // `-Doptimize=` still wins over both.
    const integration = b.option(bool, "integration", "Also run the slow GPU + real-model integration tests") orelse false;
    // Declared directly rather than via `standardOptimizeOption`, which turns
    // `-Doptimize=<mode>` into a `-Drelease` bool as soon as a preferred mode is
    // set. Same option name and meaning as the standard one; only the DEFAULT is
    // ours to choose.
    const optimize = b.option(
        std.lang.Optimize,
        "optimize",
        "Prioritize performance, safety, or binary size",
    ) orelse if (integration) std.lang.Optimize.safe else std.lang.Optimize.debug;

    // Declare every option before the first `dependencyLazy`: on a checkout missing a
    // lazy package, `build` returns at that call, and the runner then rejects any -D
    // flag declared after it as an invalid option instead of fetching and rerunning.

    // `-Dself-hosted` builds the executables with Zig's self-hosted x86_64
    // codegen backend instead of LLVM (backend A/B benchmarking). `null` = Zig's
    // default (LLVM in Release). SPIR-V kernels always stay self-hosted.
    const self_hosted = b.option(bool, "self-hosted", "Build executables with the self-hosted backend (no LLVM)") orelse false;
    const bundle_libs = is_windows or (b.option(bool, "bundle-libs", "Linux: link the pinned libvips and FFmpeg packages and install them beside the executables, instead of the system's (a release build)") orelse false);
    // Privacy: the code paths that print the user's prompt or reply are
    // COMPILED OUT of the engine layer, which since the socket split is linked
    // only by tp-serve and the engine probes (tp-gui has no engine). A flag is
    // a promise; a missing code path is a fact. `-Dprivacy=false` puts the
    // dumps back for a debugging build of chat-probe.
    const privacy = b.option(bool, "privacy", "Compile the prompt/reply debug dumps out of the engine layer (default true)") orelse true;
    // Where the real-model tests and device commands look for checkpoints.
    const model_root = b.option([]const u8, "model-root", "Root of the diffusion model folders (checkpoints/, vae/, text_encoders/, ...)") orelse "/mnt/watch/models/diffusion";
    const test_filter = b.option([]const u8, "test-filter", "Only run tests whose name matches this substring");
    const test_filters: []const []const u8 = if (test_filter) |f| &.{f} else &.{};
    // `-Dtest-timing`: swap Zig's runner for one that reports per-test wall time
    // (tools/test_timing_runner.zig), for finding what makes the gated suite slow.
    const test_timing = b.option(bool, "test-timing", "Report per-test wall time instead of the normal runner") orelse false;

    const translate_c = b.dependency("translate_c", .{});

    // GPU compute kernels: Zig source compiled to SPIR-V by the self-hosted
    // backend (the LLVM backend does not support spirv_kernel) and embedded
    // into the library as raw module bytes. One object per kernel: the SPIR-V
    // backend only supports workgroup storage in a single entry point per module.
    // SPIR-V 1.4 for workgroup-storage variables in entry-point interfaces.
    const spirv_target = b.resolveTargetQuery(.{
        .cpu_arch = .spirv64,
        .os_tag = .vulkan,
        .cpu_model = .baseline,
        .cpu_features_add = std.Target.spirv.featureSet(&.{ .v1_4, .float16, .int8, .int16 }),
    });
    const kernel_names = [_][]const u8{ "matmul_f8", "matmul_f32", "transpose", "eltwise", "attn_batched", "dual" };
    var kernel_spv: [kernel_names.len]std.Build.LazyPath = undefined;
    for (kernel_names, &kernel_spv) |kname, *spv| {
        const obj = b.addObject(.{
            .name = kname,
            .root_module = b.createModule(.{
                .root_source_file = b.path(b.fmt("src/gpu/kernels/{s}.zig", .{kname})),
                .target = spirv_target,
                .optimize = .fast,
            }),
            .use_llvm = false,
        });
        spv.* = obj.getEmittedBin();
    }

    // The CUDA generation every PTX module is built for: the hand kernels' headers
    // (`cuda/ptx.zig` reads both from build_options), `dual`'s PTX below, and
    // the JIT all follow these two, so a second generation is one flag.
    const cuda_sm = b.option(u32, "cuda-sm", "CUDA compute capability the PTX targets, e.g. 86 or 89 (default 86)") orelse 86;
    const ptx_isa = b.option([]const u8, "ptx-isa", "PTX ISA version in every kernel header (default 8.0)") orelse "8.0";
    const cuda_march = b.fmt("sm_{d}", .{cuda_sm});
    const cuda_model = nvptxModel(cuda_march) orelse std.debug.panic("-Dcuda-sm={d}: Zig has no nvptx cpu model {s}", .{ cuda_sm, cuda_march });

    // `dual` also compiles to PTX for the CUDA backend: the object's assembly, at the
    // ISA version the hand-written kernels use.
    const dual_ptx_obj = b.addObject(.{
        .name = "dual",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/gpu/kernels/dual.zig"),
            .target = b.resolveTargetQuery(.{
                .cpu_arch = .nvptx64,
                .os_tag = .cuda,
                .cpu_model = .{ .explicit = cuda_model },
                .cpu_features_add = std.Target.nvptx.featureSet(&.{ptxFeature(ptx_isa)}),
            }),
            .optimize = .fast,
            .strip = true,
        }),
        .use_llvm = true,
    });
    const dual_ptx = dual_ptx_obj.getEmittedAsm();

    // The engine itself is pure Zig; libc is linked solely so std.DynLib can
    // dlopen the system Vulkan loader (Zig's own ELF loader cannot initialize
    // glibc-linked driver stacks). No C code is compiled or bound.
    const mod = b.addModule("TensorPencil", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    // tp_core: the foundational primitive layer (src/core/), its own module so
    // consumers can depend on it in isolation. The umbrella `mod` and every
    // higher layer import it by name as "tp_core". ggml + build_options get
    // wired into it below (quants needs both).
    const core_mod = b.addModule("tp_core", .{
        .root_source_file = b.path("src/core/core.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    mod.addImport("tp_core", core_mod);

    // tp_ops: CPU numeric kernels (src/ops/), depends on tp_core. Its own module
    // so consumers can pull the ops layer without the GPU/model layers. ggml is
    // wired below (matmul's block-quant GEMV).
    const ops_mod = b.addModule("tp_ops", .{
        .root_source_file = b.path("src/ops.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    ops_mod.addImport("tp_core", core_mod);
    mod.addImport("tp_ops", ops_mod);

    // tp_gpu: GPU backends (Vulkan loader + Zig SPIR-V kernels, CUDA driver-API
    // PTX; src/gpu/). Depends on tp_core + tp_ops. The SPIR-V kernel blobs are
    // embedded here (reachable as @embedFile("matmul_f8_spv") from gpu/context).
    // build_options is wired below.
    const gpu_mod = b.addModule("tp_gpu", .{
        .root_source_file = b.path("src/gpu.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    gpu_mod.addImport("tp_core", core_mod);
    gpu_mod.addImport("tp_ops", ops_mod);
    for (kernel_names, kernel_spv) |kname, spv| {
        gpu_mod.addAnonymousImport(b.fmt("{s}_spv", .{kname}), .{ .root_source_file = spv });
    }
    gpu_mod.addAnonymousImport("dual_ptx", .{ .root_source_file = dual_ptx });
    mod.addImport("tp_gpu", gpu_mod);

    // tp_runtime: the offload/scheduling tier (VRAM arbiter + residency planner;
    // src/runtime/). Pure std, no dependency on the other layers.
    const runtime_mod = b.addModule("tp_runtime", .{
        .root_source_file = b.path("src/runtime/runtime.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    mod.addImport("tp_runtime", runtime_mod);

    // tp_models: the model tier, NN architectures (src/models/) plus the LLM
    // generation machinery (src/llm/) that drives them, in one module because
    // they are mutually recursive. Depends on all lower layers. build_options is
    // wired below (its test_gate).
    const models_mod = b.addModule("tp_models", .{
        .root_source_file = b.path("src/tp_models.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    models_mod.addImport("tp_core", core_mod);
    models_mod.addImport("tp_ops", ops_mod);
    models_mod.addImport("tp_gpu", gpu_mod);
    models_mod.addImport("tp_runtime", runtime_mod);
    mod.addImport("tp_models", models_mod);

    // ggml (llama.cpp tensor lib) is an OPTIONAL dependency (default on, gated
    // by `-Dggml`): its AVX2 CPU quant kernels are ~30x faster than our Zig ones
    // and back the GGUF block-quant (q4_k/q5_k/q6_k/q8_0) dequant + GEMV paths
    // (@import("ggml")). Fetched rather than vendored so external consumers get
    // it via `zig fetch`, and lazy so `-Dggml=false` skips the fetch/compile
    // entirely; the library's block-quant dtypes then return
    // error.QuantBackendUnavailable (see src/quants.zig, ops/matmul.zig).
    // The static lib + libc++ are linked into `mod`, so every artifact that
    // transitively uses the TensorPencil module picks them up.
    const have_ggml = b.option(bool, "ggml", "Link ggml for GGUF block-quant support (default true)") orelse true;
    var ggml_mod: ?*std.Build.Module = null;
    if (have_ggml) {
        const ggml_dep = try b.dependencyLazy("ggml", .{});
        const ggml_lib = ggml_build.buildLib(b, target, .fast, ggml_dep);
        const ggml_c: Translator = .init(translate_c, .{
            .name = "ggml_c",
            .c_source_file = b.addWriteFiles().add("ggml_c.h",
                \\#define NDEBUG 1
                \\#include <ggml.h>
                \\#include <ggml-cpu.h>
                \\
            ),
            .target = target,
            .optimize = optimize,
        });
        ggml_c.addIncludePath(ggml_dep.path("include"));
        const gm = b.createModule(.{
            .root_source_file = b.path("src/ggml.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{.{ .name = "ggml_c", .module = ggml_c.mod }},
        });
        mod.addImport("ggml", gm);
        mod.linkLibrary(ggml_lib);
        mod.link_libcpp = true;
        // quants lives in tp_core, so ggml is wired there too (matmul in the
        // umbrella still needs its own `ggml` import above).
        core_mod.addImport("ggml", gm);
        core_mod.linkLibrary(ggml_lib);
        core_mod.link_libcpp = true;
        // matmul (tp_ops) also calls ggml directly; the static lib + libc++
        // reach the final link via tp_core, but the module import is needed here.
        ops_mod.addImport("ggml", gm);
        ggml_mod = gm;
    }

    // `zig build test` runs only the fast CPU unit suite (~15 s). The slow
    // integration tests (GPU device tests and the real-model LLM / parity tests
    // that load multi-GB checkpoints) are gated behind `-Dintegration`. When it's
    // off, GPU `init` fails in test builds (so GPU tests self-skip via their
    // `catch SkipZigTest`) and the heavy model tests skip via
    // `test_gate.requireIntegration`.
    const use_llvm: ?bool = if (self_hosted) false else null;

    const build_opts = b.addOptions();
    build_opts.addOption(bool, "integration", integration);
    build_opts.addOption([]const u8, "model_root", model_root);
    build_opts.addOption(bool, "privacy", privacy);
    // Tie the compile-time flag to whether ggml was actually wired, so a build
    // without it compiles the block-quant paths to a clean runtime error instead
    // of a missing-module compile error.
    build_opts.addOption(bool, "have_ggml", ggml_mod != null);
    build_opts.addOption(u32, "cuda_sm", cuda_sm);
    build_opts.addOption([]const u8, "ptx_isa", ptx_isa);
    const opts_mod = build_opts.createModule();
    mod.addImport("build_options", opts_mod);
    core_mod.addImport("build_options", opts_mod);
    gpu_mod.addImport("build_options", opts_mod);
    models_mod.addImport("build_options", opts_mod);

    const exe = b.addExecutable(.{
        .name = "TensorPencil",
        .use_llvm = use_llvm,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .link_libc = true,
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "TensorPencil", .module = mod },
            },
        }),
    });

    // Clip MUXING (H.264 + AAC) behind a tiny C shim over libav*, the sibling of
    // the libvips decode shim. Linked into the diffusion EXECUTABLE only: the
    // TensorPencil library module stays pure Zig, so the engine hands back a
    // `Clip` of plain data and this writes the container. Building needs
    // libavformat/libavcodec/libavutil/libswscale/libswresample + pkg-config,
    // or the pinned packages on Windows and under -Dbundle-libs (see `Pinned`).
    const av_libs = [_][]const u8{ "avformat", "avcodec", "avutil", "swscale", "swresample" };
    const av_module = shimModule(b, translate_c, target, optimize, "av", "lib/video", "av_helper");
    const av_dlls = if (is_windows) try Pinned.link(b, av_module, &.{.{
        .dep = "ffmpeg_win",
        .includes = &.{"include"},
        .libs = &.{ "lib/avformat.lib", "lib/avcodec.lib", "lib/avutil.lib", "lib/swscale.lib", "lib/swresample.lib" },
        .dll_dir = "bin",
        .licenses = &.{"LICENSE.txt"},
    }}) else if (bundle_libs) try Pinned.link(b, av_module, &.{.{
        .dep = "ffmpeg_linux",
        .includes = &.{"include"},
        .shared = &.{ "lib/libavformat.so.62", "lib/libavcodec.so.62", "lib/libavutil.so.60", "lib/libswscale.so.9", "lib/libswresample.so.6" },
        .licenses = &.{"LICENSE.txt"},
    }}) else blk: {
        for (av_libs) |lib| av_module.linkSystemLibrary(lib, .{});
        break :blk null;
    };
    if (av_dlls) |l| l.installWith(b.getInstallStep());
    exe.root_module.addImport("av", av_module);

    b.installArtifact(exe);

    const run_step = b.step("run", "Run the app");
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);
    // Run from the installation directory rather than the cache directory.
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.addPassthruArgs();

    // tp-llm: the LLM inference CLI, a second thin driver over the same
    // TensorPencil library module.
    const llm_exe = b.addExecutable(.{
        .name = "tp-llm",
        .use_llvm = use_llvm,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/llm_main.zig"),
            .link_libc = true,
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "TensorPencil", .module = mod },
            },
        }),
    });
    // Image DECODE for tp-llm (--image / @mentions): libvips behind a tiny C
    // shim (its varargs C API cannot be called from Zig), ported from DiffKeep.
    // Linked into the executables only; the TensorPencil library module stays
    // pure Zig. Building needs libvips-dev + pkg-config, or the pinned packages
    // on Windows and under -Dbundle-libs (see `Pinned`).
    const vips_module = shimModule(b, translate_c, target, optimize, "vips", "lib/vips", "vips_helper");
    const vips_dlls = if (is_windows) try Pinned.link(b, vips_module, &.{.{
        .dep = "vips_win",
        .includes = &.{ "include", "include/glib-2.0", "lib/glib-2.0/include" },
        .libs = &.{ "lib/libvips.lib", "lib/libgio-2.0.lib", "lib/libgobject-2.0.lib", "lib/libglib-2.0.lib" },
        .dll_dir = "bin",
        .licenses = &.{ "LICENSE", "versions.json" },
    }}) else if (bundle_libs) try Pinned.link(b, vips_module, &.{
        .{
            .dep = "vips_linux_dev",
            .includes = &.{ "include", "include/glib-2.0" },
        },
        .{
            .dep = "vips_linux",
            .includes = &.{"lib/glib-2.0/include"},
            // One library: sharp's build folds libvips' dependencies and glib into it.
            .shared = &.{"lib/libvips-cpp.so.8.18.7"},
            .licenses = &.{ "README.md", "versions.json" },
        },
    }) else blk: {
        vips_module.linkSystemLibrary("vips", .{});
        break :blk null;
    };
    if (vips_dlls) |l| l.installWith(b.getInstallStep());
    llm_exe.root_module.addImport("vips", vips_module);

    b.installArtifact(llm_exe);

    const run_llm_step = b.step("run-llm", "Run tp-llm");
    const run_llm_cmd = b.addRunArtifact(llm_exe);
    run_llm_cmd.step.dependOn(b.getInstallStep());
    run_llm_cmd.addPassthruArgs();
    run_llm_step.dependOn(&run_llm_cmd.step);

    // The tp-serve protocol layer: wire types, certificates, framing. Below every
    // layer (they all carry its types) and free of known-folders, so its tests
    // ride in the fast `test` step.
    const serve_mod = b.createModule(.{
        .root_source_file = b.path("src/serve/serve.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "tp_core", .module = core_mod },
            .{ .name = "TensorPencil", .module = mod },
        },
    });

    // NOTE on every `dependencyLazy` below: the build script cannot see which step
    // was requested, so a fresh checkout fetches all of them on any `zig build`,
    // `test` included. `.lazy` buys the manifest, not the step: what it saves is a
    // checkout that never builds these at all.

    // The tp-gui / tp-serve layers: `shared` is what both hold, `engine` is
    // what owns a GPU, `client` is tp-gui's own state. Each is a module rather
    // than a directory because a test or probe rooted in one directory cannot
    // `@import` a file above it.
    const known_folders = (try b.dependencyLazy("known_folders", .{})).module("known-folders");
    const shared_mod = b.createModule(.{
        .root_source_file = b.path("src/shared/shared.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "TensorPencil", .module = mod },
            .{ .name = "serve", .module = serve_mod },
            .{ .name = "known-folders", .module = known_folders },
        },
    });
    const engine_mod = b.createModule(.{
        .root_source_file = b.path("src/engine/engine.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "TensorPencil", .module = mod },
            .{ .name = "serve", .module = serve_mod },
            .{ .name = "shared", .module = shared_mod },
            .{ .name = "build_options", .module = opts_mod },
        },
    });
    const client_mod = b.createModule(.{
        .root_source_file = b.path("src/client/client.zig"),
        .link_libc = true, // localtime_r, for history's local-day grouping
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "TensorPencil", .module = mod },
            .{ .name = "serve", .module = serve_mod },
            .{ .name = "shared", .module = shared_mod },
        },
    });

    // tp-serve: the engine host behind a socket. `zig build serve` installs
    // it; the gui steps install it too, since tp-gui spawns it as the sibling
    // binary of its own installed path. tls.zig is both ends of a remote link
    // (`serve_tls`): the daemon's server, and the client each binary that dials
    // a remote host installs. Only those binaries and tls-spike link it, so no
    // test binary carries it.
    const tls_dep = try b.dependencyLazy("tls", .{ .target = target, .optimize = optimize });
    const serve_tls_mod = b.createModule(.{
        .root_source_file = b.path("src/serve_tls.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "serve", .module = serve_mod },
            .{ .name = "tls", .module = tls_dep.module("tls") },
        },
    });
    const serve_step = b.step("serve", "Build tp-serve (the engine host daemon)");
    const serve_exe = b.addExecutable(.{
        .name = "tp-serve",
        .use_llvm = use_llvm,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/serve_main.zig"),
            .link_libc = true,
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "TensorPencil", .module = mod },
                .{ .name = "serve", .module = serve_mod },
                .{ .name = "shared", .module = shared_mod },
                .{ .name = "engine", .module = engine_mod },
                .{ .name = "serve_tls", .module = serve_tls_mod },
            },
        }),
    });
    const serve_install = b.addInstallArtifact(serve_exe, .{});
    serve_step.dependOn(&serve_install.step);

    // tp-gui: desktop GUI (dvui + SDL3), a conversational image studio over
    // tp-serve hosts. The GUI deps (dvui/SDL3) are BUILT only for the gui steps.
    const gui_step = b.step("gui", "Build tp-gui (desktop GUI)");
    const run_gui_step = b.step("run-gui", "Run tp-gui");
    const gui_test_step = b.step("gui-test", "Run tp-gui config unit tests (not part of `test`)");
    const dvui_dep = try b.dependencyLazy("dvui", .{
        .target = target,
        .optimize = optimize,
        .backend = .sdl3,
    });
    const dvui_mod = dvui_dep.module("dvui_sdl3");
    const gui_exe = b.addExecutable(.{
        .name = "tp-gui",
        .use_llvm = use_llvm,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/gui_main.zig"),
            .link_libc = true,
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "TensorPencil", .module = mod },
                .{ .name = "dvui", .module = dvui_mod },
                .{ .name = "backend", .module = dvui_dep.module("sdl3") },
                // Reuse the same libvips decode shim as tp-llm (jpeg/webp/
                // etc. for dropped / @mentioned images).
                .{ .name = "vips", .module = vips_module },
                // Platform config-dir resolution for the settings file.
                .{ .name = "known-folders", .module = known_folders },
                .{ .name = "serve", .module = serve_mod },
                .{ .name = "shared", .module = shared_mod },
                // No `engine`: tp-gui runs none, and the build says so.
                .{ .name = "client", .module = client_mod },
                .{ .name = "serve_tls", .module = serve_tls_mod },
            },
        }),
    });
    // No windowing library is linked: SDL3 loads X11 and Wayland itself at run
    // time, so a release does not require any one of them on the user's system.
    // Install only via the `gui` / `run-gui` steps, never the default
    // step, so `zig build` / `zig build test` stay free of the GUI deps.
    const install_gui = b.addInstallArtifact(gui_exe, .{});
    install_gui.step.dependOn(&serve_install.step);
    if (vips_dlls) |l| l.installWith(&install_gui.step);
    gui_step.dependOn(&install_gui.step);

    // The installed binary, not the cached artifact: tp-gui finds
    // tp-serve beside its own executable.
    const run_gui_cmd = b.addRunFile(b.graph.path(.install_bin, b.fmt("tp-gui{s}", .{target.result.exeFileExt()})));
    run_gui_cmd.step.dependOn(&install_gui.step);
    run_gui_cmd.addPassthruArgs();
    run_gui_step.dependOn(&run_gui_cmd.step);

    // md-probe: renders a markdown torture document to a PNG through
    // the real fonts + renderer (`zig build md-probe -- out.png`).
    // The renderer's failure modes are visual, so this is the cheap
    // way to eyeball changes without driving a chat session.
    const md_probe_step = b.step("md-probe", "Render the markdown probe document to a PNG");
    const md_probe_exe = b.addExecutable(.{
        .name = "md-probe",
        .use_llvm = use_llvm,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/gui/md_probe.zig"),
            .link_libc = true,
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "dvui", .module = dvui_mod },
                .{ .name = "backend", .module = dvui_dep.module("sdl3") },
            },
        }),
    });
    linkWindowing(md_probe_exe, target);
    const run_md_probe = b.addRunArtifact(md_probe_exe);
    run_md_probe.addPassthruArgs();
    md_probe_step.dependOn(&run_md_probe.step);

    // ui-probe: renders the whole chat workspace (or, with --settings, the
    // settings form) from canned data to a PNG
    // (`zig build ui-probe -- out.png [w h] [--states|--settings]`). No
    // model, no GPU, no engine: the failure modes of this screen are all
    // visual, and driving the real app to see them needs a loaded
    // checkpoint. It LINKS TensorPencil (the settings form inspects
    // checkpoints and evaluates noise curves) but starts nothing.
    const ui_probe_step = b.step("ui-probe", "Render the chat workspace to a PNG from canned data");
    const ui_probe_exe = b.addExecutable(.{
        .name = "ui-probe",
        .use_llvm = use_llvm,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/gui/ui_probe.zig"),
            .link_libc = true,
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "dvui", .module = dvui_mod },
                .{ .name = "backend", .module = dvui_dep.module("sdl3") },
                .{ .name = "TensorPencil", .module = mod },
                .{ .name = "serve", .module = serve_mod },
                .{ .name = "shared", .module = shared_mod },
                .{ .name = "client", .module = client_mod },
            },
        }),
    });
    linkWindowing(ui_probe_exe, target);
    const run_ui_probe = b.addRunArtifact(ui_probe_exe);
    run_ui_probe.addPassthruArgs();
    ui_probe_step.dependOn(&run_ui_probe.step);

    // The three layers, one test binary each: settings, catalog, model
    // spec, tool-call parsing, turn stats and meter math (shared); the
    // chat and diffusion engines' pure helpers, the scanner and the
    // sampler ABI (engine); selection, history and the prompt library
    // (client). All CPU-only. Kept off the default `test` step, which
    // stays free of known-folders.
    for ([_]*std.Build.Module{ shared_mod, engine_mod, client_mod }) |m| {
        gui_test_step.dependOn(&testRun(b, b.addTest(.{ .root_module = m, .filters = test_filters })).step);
    }

    // Pure-std gui files: the markdown parser and the viewer zoom/pan math. The
    // dvui-facing views that use them stay out of the test build.
    for ([_][]const u8{ "src/gui/markdown.zig", "src/gui/viewmath.zig" }) |path| {
        const t = b.addTest(.{
            .root_module = b.createModule(.{
                .root_source_file = b.path(path),
                .target = target,
                .optimize = optimize,
            }),
            .filters = test_filters,
        });
        gui_test_step.dependOn(&testRun(b, t).step);
    }

    // Font tier routing (fonts.zig: the cmap coverage parse plus which face each
    // codepoint lands on, whose failure mode is a tofu box otherwise only visible
    // in a screenshot) and the design system (style.zig: the nominal-px ->
    // cap-height conversion, and that every role names a family we register,
    // since dvui silently substitutes its built-in face otherwise). Both need the
    // dvui module but no window.
    for ([_][]const u8{ "src/gui/fonts.zig", "src/gui/style.zig" }) |path| {
        const t = b.addTest(.{
            .root_module = b.createModule(.{
                .root_source_file = b.path(path),
                .link_libc = true,
                .target = target,
                .optimize = optimize,
                .imports = &.{
                    .{ .name = "dvui", .module = dvui_mod },
                },
            }),
            .filters = test_filters,
        });
        linkWindowing(t, target);
        gui_test_step.dependOn(&testRun(b, t).step);
    }

    // The views with arithmetic in them: the meter's segment maths, the
    // status bar's sampling, and the studio form's defaults.
    for ([_][]const u8{
        "src/gui/meter.zig",
        "src/gui/status_bar.zig",
        "src/gui/image_view.zig",
        "src/gui/model_lib.zig",
    }) |path| {
        const t = b.addTest(.{
            .root_module = b.createModule(.{
                .root_source_file = b.path(path),
                .link_libc = true,
                .target = target,
                .optimize = optimize,
                .imports = &.{
                    .{ .name = "TensorPencil", .module = mod },
                    .{ .name = "dvui", .module = dvui_mod },
                    .{ .name = "serve", .module = serve_mod },
                    .{ .name = "shared", .module = shared_mod },
                    .{ .name = "client", .module = client_mod },
                },
            }),
            .filters = test_filters,
        });
        linkWindowing(t, target);
        gui_test_step.dependOn(&testRun(b, t).step);
    }

    // The engine and client probes. Each is a CLI that prints its own usage; the
    // ones that spawn tp-serve install it first, since they find it as a sibling.
    // catalog-probe and chat-probe link no dvui, which keeps the engine and
    // shared layers free of it.
    const Probe = struct {
        name: []const u8,
        desc: []const u8,
        root: []const u8,
        imports: []const std.Build.Module.Import,
        /// Installed as well as run, so a tool that must wrap the binary
        /// (compute-sanitizer, gdb) has a stable path built by THIS build.
        install: bool = true,
        spawns_serve: bool = false,
        tls: bool = false,
    };
    const probes = [_]Probe{
        .{
            .name = "catalog-probe",
            .desc = "Scan model folders and print what tp-gui takes each file for",
            .root = "src/engine/catalog_probe.zig",
            .imports = &.{ .{ .name = "shared", .module = shared_mod }, .{ .name = "engine", .module = engine_mod } },
            .install = false,
        },
        .{
            .name = "chat-probe",
            .desc = "Run a tp-gui chat session headlessly and dump the raw reply",
            .root = "src/engine/chat_probe.zig",
            .imports = &.{ .{ .name = "TensorPencil", .module = mod }, .{ .name = "shared", .module = shared_mod }, .{ .name = "engine", .module = engine_mod } },
        },
        .{
            .name = "driver-probe",
            .desc = "Drive the engine host with wire requests and print its events",
            .root = "src/engine/driver_probe.zig",
            .imports = &.{ .{ .name = "TensorPencil", .module = mod }, .{ .name = "serve", .module = serve_mod }, .{ .name = "shared", .module = shared_mod }, .{ .name = "engine", .module = engine_mod } },
        },
        .{
            .name = "serve-probe",
            .desc = "Drive a spawned tp-serve over its socket and print its events",
            .root = "src/client/serve_probe.zig",
            .imports = &.{ .{ .name = "TensorPencil", .module = mod }, .{ .name = "serve", .module = serve_mod }, .{ .name = "shared", .module = shared_mod }, .{ .name = "client", .module = client_mod } },
            .spawns_serve = true,
        },
        .{
            .name = "hosts-probe",
            .desc = "Drive two spawned tp-serve hosts as tp-gui does and check the split",
            .root = "src/client/hosts_probe.zig",
            .imports = &.{ .{ .name = "TensorPencil", .module = mod }, .{ .name = "serve", .module = serve_mod }, .{ .name = "shared", .module = shared_mod }, .{ .name = "client", .module = client_mod } },
            .spawns_serve = true,
            .tls = true,
        },
    };
    for (probes) |p| {
        const probe_exe = b.addExecutable(.{
            .name = p.name,
            .use_llvm = use_llvm,
            .root_module = b.createModule(.{
                .root_source_file = b.path(p.root),
                .link_libc = true,
                .target = target,
                .optimize = optimize,
                .imports = p.imports,
            }),
        });
        if (p.tls) probe_exe.root_module.addImport("serve_tls", serve_tls_mod);
        const probe_step = b.step(p.name, p.desc);
        const probe_run = b.addRunArtifact(probe_exe);
        probe_run.addPassthruArgs();
        if (p.install) {
            const probe_install = b.addInstallArtifact(probe_exe, .{});
            if (p.spawns_serve) probe_install.step.dependOn(&serve_install.step);
            probe_step.dependOn(&probe_install.step);
            probe_run.step.dependOn(&probe_install.step);
        }
        probe_step.dependOn(&probe_run.step);
    }

    // tls-spike: the remote link end to end on a loopback socket with no
    // engine: the real server routing behind tls.zig, the real client link
    // (tls.zig's client pinned to the minted certificate), the right token,
    // a wrong token, a wrong pin, a pushed multi-MiB frame and a multi-MiB
    // upload. Exits non-zero on any failure.
    {
        const spike_exe = b.addExecutable(.{
            .name = "tls-spike",
            .use_llvm = use_llvm,
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/tls_spike.zig"),
                .target = target,
                .optimize = optimize,
                .imports = &.{
                    .{ .name = "serve", .module = serve_mod },
                    .{ .name = "serve_tls", .module = serve_tls_mod },
                },
            }),
        });
        b.step("tls-spike", "Run the remote-link gate on a loopback socket").dependOn(&b.addRunArtifact(spike_exe).step);
    }

    // Benchmarks, always ReleaseFast. ggml-bench (TensorPencil's CPU quant
    // kernels against ggml's) and qgemv-bench (grouped-N dp4a quant GEMV against
    // dequant->f16 GEMM on device) need direct `ggml` C access, so they exist
    // only when ggml is linked.
    const Bench = struct { name: []const u8, desc: []const u8, root: []const u8, ggml: bool = false };
    const benches = [_]Bench{
        .{ .name = "embed-bench", .desc = "Build+run the tp.embed encoder throughput profiler", .root = "src/embed_bench.zig" },
        .{ .name = "ggml-bench", .desc = "Build+run the ggml vs TensorPencil CPU-kernel benchmark", .root = "src/ggml_bench.zig", .ggml = true },
        .{ .name = "qgemv-bench", .desc = "Build+run the grouped-N GEMV vs dequant-GEMM device benchmark", .root = "src/qgemv_bench.zig", .ggml = true },
    };
    for (benches) |bench| {
        if (bench.ggml and ggml_mod == null) continue;
        const bench_exe = b.addExecutable(.{
            .name = bench.name,
            .root_module = b.createModule(.{
                .root_source_file = b.path(bench.root),
                .link_libc = true,
                .target = target,
                .optimize = .fast,
                .imports = &.{
                    .{ .name = "TensorPencil", .module = mod },
                },
            }),
        });
        if (bench.ggml) bench_exe.root_module.addImport("ggml", ggml_mod.?);
        const bench_run = b.addRunArtifact(bench_exe);
        bench_run.addPassthruArgs();
        b.step(bench.name, bench.desc).dependOn(&bench_run.step);
    }

    const test_runner: ?std.Build.Step.Compile.TestRunner = if (test_timing)
        .{ .path = b.path("tools/test_timing_runner.zig"), .mode = .simple }
    else
        null;

    // Each layer module is its own compilation, so its `test` blocks run from
    // their own test binary.
    const test_step = b.step("test", "Run tests");
    const test_roots = [_]*std.Build.Module{ serve_mod, mod, core_mod, ops_mod, gpu_mod, runtime_mod, models_mod, exe.root_module };
    // The device tests share one GPU: side by side under Windows one binary stopped
    // answering the build runner for over a minute, and in series they cost no more time.
    // A claim of all the memory runs them one at a time without ordering them, so a
    // failure skips no other, and no peak can exceed it (a binary mapping a checkpoint
    // bigger than half the RAM printed a red "exceeding" line on a green run).
    const one_at_a_time: u64 = if (integration) std.process.totalSystemMemory() catch 0 else 0;
    for (test_roots) |root| {
        const run = testRun(b, b.addTest(.{
            .root_module = root,
            .filters = test_filters,
            .test_runner = test_runner,
        }));
        if (root == exe.root_module) if (av_dlls) |l| l.findFrom(b, run, is_windows);
        if (one_at_a_time > 1) run.step.max_rss = one_at_a_time;
        test_step.dependOn(&run.step);
    }

    // `check`: compile every artifact any other step builds, running none of them
    // (the probes spawn daemons and read configs).
    const check = b.step("check", "Compile every artifact without running anything");
    var seen: std.AutoHashMapUnmanaged(*std.Build.Step, void) = .empty;
    var stack: std.ArrayList(*std.Build.Step) = .empty;
    for (b.top_level_steps.values()) |tl| try stack.append(b.allocator, &tl.step);
    while (stack.pop()) |s| {
        if ((try seen.getOrPut(b.allocator, s)).found_existing) continue;
        if (s.cast(std.Build.Step.Compile) != null) check.dependOn(s);
        try stack.appendSlice(b.allocator, s.dependencies.items);
    }
}

/// A Zig module over one of the C shims in lib/: `<dir>/<name>.c` compiled in,
/// `<dir>/<name>.h` translated and imported as "c". The caller links the library
/// the shim wraps.
fn shimModule(
    b: *std.Build,
    translate_c: *std.Build.Dependency,
    target: std.Build.ResolvedTarget,
    optimize: std.lang.Optimize,
    zig_name: []const u8,
    dir: []const u8,
    name: []const u8,
) *std.Build.Module {
    const header: Translator = .init(translate_c, .{
        .name = b.fmt("{s}_c", .{zig_name}),
        .c_source_file = b.path(b.fmt("{s}/{s}.h", .{ dir, name })),
        .target = target,
        .optimize = optimize,
    });
    const module = b.createModule(.{
        .root_source_file = b.path(b.fmt("src/{s}.zig", .{zig_name})),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "c", .module = header.mod }},
    });
    module.addCSourceFile(.{ .file = b.path(b.fmt("{s}/{s}.c", .{ dir, name })), .flags = &.{} });
    module.addIncludePath(b.path(dir));
    return module;
}

/// A run of test binary `t`. A cross build compiles it and skips the run.
fn testRun(b: *std.Build, t: *std.Build.Step.Compile) *std.Build.Step.Run {
    const run = b.addRunArtifact(t);
    run.skip_foreign_checks = true;
    return run;
}

/// The libraries SDL3 needs at link time when a dvui binary is linked on Linux.
fn linkWindowing(c: *std.Build.Step.Compile, target: std.Build.ResolvedTarget) void {
    if (target.result.os.tag != .linux) return;
    for ([_][]const u8{ "X11", "Xcursor", "Xi", "wayland-client" }) |lib| c.root_module.linkSystemLibrary(lib, .{});
}

/// The `std.Target.nvptx.cpu` model named `name` (`sm_86`), if Zig has one.
fn nvptxModel(name: []const u8) ?*const std.Target.Cpu.Model {
    inline for (@typeInfo(std.Target.nvptx.cpu).@"struct".decl_names) |d| {
        if (std.mem.eql(u8, d, name)) return &@field(std.Target.nvptx.cpu, d);
    }
    return null;
}

/// The nvptx cpu feature for a PTX ISA version: "8.0" -> `.ptx80`.
fn ptxFeature(isa: []const u8) std.Target.nvptx.Feature {
    var name: [16]u8 = undefined;
    var w = std.Io.Writer.fixed(&name);
    w.writeAll("ptx") catch unreachable;
    for (isa) |ch| if (ch != '.') w.writeByte(ch) catch std.debug.panic("-Dptx-isa={s}: expected major.minor", .{isa});
    return std.meta.stringToEnum(std.Target.nvptx.Feature, w.buffered()) orelse
        std.debug.panic("-Dptx-isa={s}: Zig has no nvptx feature {s}", .{ isa, w.buffered() });
}

/// A prebuilt library package (`build.zig.zon`) standing in for pkg-config: what a
/// Windows build always links, and a Linux one under -Dbundle-libs, so a release
/// carries its own libvips and FFmpeg rather than a distribution's.
const Pinned = struct {
    dep: []const u8,
    includes: []const []const u8 = &.{},
    /// Import libs, linked by exact path: Zig's `-l` search does not try
    /// every naming scheme these packages use (`libvips.lib`).
    libs: []const []const u8 = &.{},
    /// A directory whose DLLs install beside the executables (Windows).
    dll_dir: ?[]const u8 = null,
    /// Shared libraries, each linked by path and installed beside the executables
    /// under its own name (its soname), which they find through `$ORIGIN`.
    shared: []const []const u8 = &.{},
    /// Installed to `share/licenses/<dep>/`: the libraries ship with their terms.
    licenses: []const []const u8 = &.{},

    const Linked = struct {
        /// Install the libraries beside the executables, and their licenses.
        installs: []const *std.Build.Step,
        /// Where the libraries are, for a run of a binary that is not installed (a test).
        dir: []const u8,

        fn installWith(self: Linked, step: *std.Build.Step) void {
            for (self.installs) |s| step.dependOn(s);
        }

        fn findFrom(self: Linked, b: *std.Build, run: *std.Build.Step.Run, windows: bool) void {
            if (!windows) return run.setEnvironmentVariable("LD_LIBRARY_PATH", self.dir);
            const path = b.graph.environ_map.get("PATH") orelse "";
            run.setEnvironmentVariable("PATH", b.fmt("{s};{s}", .{ self.dir, path }));
        }
    };

    /// Points `module` at every package.
    fn link(b: *std.Build, module: *std.Build.Module, pkgs: []const Pinned) !Linked {
        var installs: std.ArrayList(*std.Build.Step) = .empty;
        var dir: ?[]const u8 = null;
        var shared = false;
        for (pkgs) |pkg| {
            const d = try b.dependencyLazy(pkg.dep, .{});
            for (pkg.includes) |p| module.addIncludePath(d.path(p));
            for (pkg.libs) |p| module.addObjectFile(d.path(p));
            if (pkg.dll_dir) |sub| {
                // Recursive, so vips' loader modules keep the subdirectory it looks in.
                const dlls = b.addInstallDirectory(.{
                    .source_dir = d.path(sub),
                    .install_dir = .bin,
                    .install_subdir = "",
                    .include_extensions = &.{".dll"},
                });
                try installs.append(b.allocator, &dlls.step);
                dir = try d.builder.root.joinString(b.allocator, sub);
            }
            for (pkg.shared) |p| {
                shared = true;
                module.addObjectFile(d.path(p));
                try installs.append(b.allocator, &b.addInstallBinFile(d.path(p), std.Io.Dir.path.basename(p)).step);
                dir = try d.builder.root.joinString(b.allocator, std.Io.Dir.path.dirname(p).?);
            }
            for (pkg.licenses) |p| {
                try installs.append(b.allocator, &b.addInstallFile(d.path(p), b.fmt("share/licenses/{s}/{s}", .{ pkg.dep, std.Io.Dir.path.basename(p) })).step);
            }
        }
        if (shared) module.addRPathSpecial("$ORIGIN");
        return .{ .installs = installs.items, .dir = dir.? };
    }
};
