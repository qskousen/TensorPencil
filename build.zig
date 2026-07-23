const std = @import("std");
const ggml_build = @import("build_ggml.zig");

// Although this function looks imperative, it does not perform the build
// directly and instead it mutates the build graph (`b`) that will be then
// executed by an external runner. The functions in `std.Build` implement a DSL
// for defining build steps and express dependencies between them, allowing the
// build runner to parallelize the build automatically (and the cache system to
// know when a step doesn't need to be re-run).
pub fn build(b: *std.Build) void {
    // Standard target options allow the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});

    // Host toolchain workaround: recent glibc/binutils (e.g. Arch/CachyOS,
    // gcc 16+) ship crt startup objects whose .sframe section uses
    // R_X86_64_PC64 relocations the Zig 0.16 self-hosted ELF linker cannot
    // process, breaking every libc-linked build. `tools/patch-crt.sh`
    // generates a sframe-stripped crt dir + .zig-crt/libc.txt; if present,
    // route all native compiles through it (via the global libc-file fallback
    // in std.Build.Step.Compile). Absent on hosts with an older toolchain, so
    // this is a no-op there. An explicit `--libc` on the CLI takes precedence.
    if (b.libc_file == null) {
        if (b.build_root.handle.access(b.graph.io, ".zig-crt/libc.txt", .{})) |_| {
            b.libc_file = b.pathFromRoot(".zig-crt/libc.txt");
        } else |_| {}
    }
    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = b.standardOptimizeOption(.{});
    // It's also possible to define more custom flags to toggle optional features
    // of this build script using `b.option()`. All defined flags (including
    // target and optimize options) will be listed when running `zig build --help`
    // in this directory.

    // This creates a module, which represents a collection of source files alongside
    // some compilation options, such as optimization mode and linked system libraries.
    // Zig modules are the preferred way of making Zig code available to consumers.
    // addModule defines a module that we intend to make available for importing
    // to our consumers. We must give it a name because a Zig package can expose
    // multiple modules and consumers will need to be able to specify which
    // module they want to access.
    // GPU compute kernels: Zig source compiled to SPIR-V by the self-hosted
    // backend (the LLVM backend does not support spirv_kernel) and embedded
    // into the library as raw module bytes.
    // One object per kernel: the 0.16 SPIR-V backend only supports workgroup
    // storage in a single entry point per module.
    const kernel_names = [_][]const u8{ "matmul_f8", "matmul_f32", "transpose", "eltwise", "attn_batched", "dp4a", "subgroup" };
    var kernel_objs: [kernel_names.len]*std.Build.Step.Compile = undefined;
    for (kernel_names, 0..) |kname, i| {
        kernel_objs[i] = b.addObject(.{
            .name = kname,
            .root_module = b.createModule(.{
                .root_source_file = b.path(b.fmt("src/gpu/kernels/{s}.zig", .{kname})),
                .target = b.resolveTargetQuery(.{
                    .cpu_arch = .spirv64,
                    .os_tag = .vulkan,
                    // SPIR-V 1.5 (Vulkan 1.2): required for workgroup-storage
                    // variables in entry-point interfaces (shared-memory tiles).
                    .cpu_features_add = std.Target.spirv.featureSet(&.{ .v1_4, .float16 }),
                }),
                .optimize = .ReleaseFast,
            }),
            .use_llvm = false,
        });
    }

    // The engine itself is pure Zig; libc is linked solely so std.DynLib can
    // dlopen the system Vulkan loader (Zig's own ELF loader cannot initialize
    // glibc-linked driver stacks). No C code is compiled or bound.
    const mod = b.addModule("TensorPencil", .{
        // The root source file is the "entry point" of this module. Users of
        // this module will only be able to access public declarations contained
        // in this file, which means that if you have declarations that you
        // intend to expose to consumers that were defined in other files part
        // of this module, you will have to make sure to re-export them from
        // the root file.
        .root_source_file = b.path("src/root.zig"),
        // Later on we'll use this module as the root module of a test executable
        // which requires us to specify a target.
        .target = target,
        .link_libc = true,
    });

    // tp_core: the foundational primitive layer (src/core/), its own module so
    // consumers can depend on it in isolation. The umbrella `mod` and every
    // higher layer import it by name as "tp_core". ggml + build_options get
    // wired into it below (quants needs both).
    const core_mod = b.addModule("tp_core", .{
        .root_source_file = b.path("src/core/core.zig"),
        .target = target,
        .link_libc = true,
    });
    mod.addImport("tp_core", core_mod);

    // tp_ops: CPU numeric kernels (src/ops/), depends on tp_core. Its own module
    // so consumers can pull the ops layer without the GPU/model layers. ggml is
    // wired below (matmul's block-quant GEMV).
    const ops_mod = b.addModule("tp_ops", .{
        .root_source_file = b.path("src/ops.zig"),
        .target = target,
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
        .link_libc = true,
    });
    gpu_mod.addImport("tp_core", core_mod);
    gpu_mod.addImport("tp_ops", ops_mod);
    for (kernel_names, kernel_objs) |kname, obj| {
        gpu_mod.addAnonymousImport(b.fmt("{s}_spv", .{kname}), .{ .root_source_file = obj.getEmittedBin() });
    }
    mod.addImport("tp_gpu", gpu_mod);

    // tp_runtime: the offload/scheduling tier (VRAM arbiter + residency planner;
    // src/runtime/). Pure std — no dependency on the other layers.
    const runtime_mod = b.addModule("tp_runtime", .{
        .root_source_file = b.path("src/runtime/runtime.zig"),
        .target = target,
        .link_libc = true,
    });
    mod.addImport("tp_runtime", runtime_mod);

    // tp_models: the model tier — NN architectures (src/models/) plus the LLM
    // generation machinery (src/llm/) that drives them, in one module because
    // they are mutually recursive. Depends on all lower layers. build_options is
    // wired below (its test_gate).
    const models_mod = b.addModule("tp_models", .{
        .root_source_file = b.path("src/tp_models.zig"),
        .target = target,
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
    // (@import("ggml")). Now fetched (was a git submodule under vendor/ggml) so
    // external consumers get it via `zig fetch`, and lazy so `-Dggml=false`
    // skips the fetch/compile entirely — the library's block-quant dtypes then
    // return error.QuantBackendUnavailable (see src/quants.zig, ops/matmul.zig).
    // The static lib + libc++ are linked into `mod`, so every artifact that
    // transitively uses the TensorPencil module picks them up.
    const have_ggml = b.option(bool, "ggml", "Link ggml for GGUF block-quant support (default true)") orelse true;
    var ggml_mod: ?*std.Build.Module = null;
    if (have_ggml) {
        if (b.lazyDependency("ggml", .{})) |ggml_dep| {
            const ggml_lib = ggml_build.buildLib(b, target, .ReleaseFast, ggml_dep);
            const gm = b.createModule(.{
                .root_source_file = b.path("src/ggml.zig"),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            });
            gm.addIncludePath(ggml_dep.path("include"));
            mod.addImport("ggml", gm);
            mod.linkLibrary(ggml_lib);
            mod.link_libcpp = true;
            // quants lives in tp_core now, so wire ggml there too (matmul in the
            // umbrella still needs its own `ggml` import above).
            core_mod.addImport("ggml", gm);
            core_mod.linkLibrary(ggml_lib);
            core_mod.link_libcpp = true;
            // matmul (tp_ops) also calls ggml directly; the static lib + libc++
            // reach the final link via tp_core, but the module import is needed here.
            ops_mod.addImport("ggml", gm);
            ggml_mod = gm;
        }
    }

    // `zig build test` runs only the fast CPU unit suite (~15 s). The slow
    // integration tests — GPU (CUDA/Vulkan) device tests and the real-model LLM
    // / parity tests that load multi-GB checkpoints and run inference in Debug —
    // are gated behind `-Dintegration`. When it's off, GPU `init` fails in test
    // builds (so GPU tests self-skip via their `catch SkipZigTest`) and the heavy
    // model tests skip via `test_gate.requireIntegration`. Full suite:
    // `zig build test -Dintegration` (needs a device + the models/ checkpoints).
    // `-Dself-hosted` builds the executables with Zig's self-hosted x86_64
    // codegen backend instead of LLVM (backend A/B benchmarking). `null` = Zig's
    // default (LLVM in Release). SPIR-V kernels always stay self-hosted.
    const self_hosted = b.option(bool, "self-hosted", "Build executables with the self-hosted backend (no LLVM)") orelse false;
    const use_llvm: ?bool = if (self_hosted) false else null;

    const integration = b.option(bool, "integration", "Also run the slow GPU + real-model integration tests") orelse false;
    const build_opts = b.addOptions();
    build_opts.addOption(bool, "integration", integration);
    // Tie the compile-time flag to whether ggml was actually wired: if the
    // dependency is disabled (or unavailable), the library compiles its
    // block-quant paths to a clean runtime error instead of a missing-module
    // compile error.
    build_opts.addOption(bool, "have_ggml", ggml_mod != null);
    const opts_mod = build_opts.createModule();
    mod.addImport("build_options", opts_mod);
    core_mod.addImport("build_options", opts_mod);
    gpu_mod.addImport("build_options", opts_mod);
    models_mod.addImport("build_options", opts_mod);

    // Here we define an executable. An executable needs to have a root module
    // which needs to expose a `main` function. While we could add a main function
    // to the module defined above, it's sometimes preferable to split business
    // logic and the CLI into two separate modules.
    //
    // If your goal is to create a Zig library for others to use, consider if
    // it might benefit from also exposing a CLI tool. A parser library for a
    // data serialization format could also bundle a CLI syntax checker, for example.
    //
    // If instead your goal is to create an executable, consider if users might
    // be interested in also being able to embed the core functionality of your
    // program in their own executable in order to avoid the overhead involved in
    // subprocessing your CLI tool.
    //
    // If neither case applies to you, feel free to delete the declaration you
    // don't need and to put everything under a single module.
    const exe = b.addExecutable(.{
        .name = "TensorPencil",
        .use_llvm = use_llvm,
        .root_module = b.createModule(.{
            // b.createModule defines a new module just like b.addModule but,
            // unlike b.addModule, it does not expose the module to consumers of
            // this package, which is why in this case we don't have to give it a name.
            .root_source_file = b.path("src/main.zig"),
            .link_libc = true,
            // Target and optimization levels must be explicitly wired in when
            // defining an executable or library (in the root module), and you
            // can also hardcode a specific target for an executable or library
            // definition if desireable (e.g. firmware for embedded devices).
            .target = target,
            .optimize = optimize,
            // List of modules available for import in source files part of the
            // root module.
            .imports = &.{
                // Here "TensorPencil" is the name you will use in your source code to
                // import this module (e.g. `@import("TensorPencil")`). The name is
                // repeated because you are allowed to rename your imports, which
                // can be extremely useful in case of collisions (which can happen
                // importing modules from different packages).
                .{ .name = "TensorPencil", .module = mod },
            },
        }),
    });

    // This declares intent for the executable to be installed into the
    // install prefix when running `zig build` (i.e. when executing the default
    // step). By default the install prefix is `zig-out/` but can be overridden
    // by passing `--prefix` or `-p`.
    b.installArtifact(exe);

    // This creates a top level step. Top level steps have a name and can be
    // invoked by name when running `zig build` (e.g. `zig build run`).
    // This will evaluate the `run` step rather than the default step.
    // For a top level step to actually do something, it must depend on other
    // steps (e.g. a Run step, as we will see in a moment).
    const run_step = b.step("run", "Run the app");

    // This creates a RunArtifact step in the build graph. A RunArtifact step
    // invokes an executable compiled by Zig. Steps will only be executed by the
    // runner if invoked directly by the user (in the case of top level steps)
    // or if another step depends on it, so it's up to you to define when and
    // how this Run step will be executed. In our case we want to run it when
    // the user runs `zig build run`, so we create a dependency link.
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);

    // By making the run step depend on the default step, it will be run from the
    // installation directory rather than directly from within the cache directory.
    run_cmd.step.dependOn(b.getInstallStep());

    // This allows the user to pass arguments to the application in the build
    // command itself, like this: `zig build run -- arg1 arg2 etc`
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    // tp-llm: LLM inference CLI (see LLM_PLAN.md), a second thin driver over
    // the same TensorPencil library module.
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
    // Image DECODE for tp-llm (--image / @mentions): system libvips behind
    // a tiny C shim (its varargs C API is un-importable directly), ported
    // from DiffKeep. Linked into the tp-llm EXECUTABLE only — the
    // TensorPencil library module stays pure Zig. Building needs
    // libvips-dev + pkg-config.
    const vips_module = b.createModule(.{
        .root_source_file = b.path("src/vips.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    vips_module.addCSourceFile(.{
        .file = b.path("lib/vips/vips_helper.c"),
        .flags = &.{},
    });
    vips_module.addIncludePath(b.path("lib/vips"));
    vips_module.linkSystemLibrary("vips", .{});
    llm_exe.root_module.addImport("vips", vips_module);

    b.installArtifact(llm_exe);

    const run_llm_step = b.step("run-llm", "Run tp-llm");
    const run_llm_cmd = b.addRunArtifact(llm_exe);
    run_llm_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_llm_cmd.addArgs(args);
    }
    run_llm_step.dependOn(&run_llm_cmd.step);

    // tp-gui: desktop GUI (dvui + SDL3) — a conversational image studio that
    // drives the same TensorPencil library (LLM chat + diffusion). GUI deps
    // (dvui/SDL3, X11/Wayland) are lazy: only fetched/built when the gui or
    // run-gui step is actually requested, so `zig build test` stays lean and
    // dependency-free. Mirrors DiffKeep's proven dvui-0.5.0-dev wiring.
    const gui_step = b.step("gui", "Build tp-gui (desktop GUI)");
    const run_gui_step = b.step("run-gui", "Run tp-gui");
    const gui_test_step = b.step("gui-test", "Run tp-gui config unit tests (not part of `test`)");
    if (b.lazyDependency("dvui", .{
        .target = target,
        .optimize = optimize,
        .backend = .sdl3,
    })) |dvui_dep| {
        if (b.lazyDependency("known_folders", .{})) |kf| {
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
                        .{ .name = "dvui", .module = dvui_dep.module("dvui_sdl3") },
                        .{ .name = "backend", .module = dvui_dep.module("sdl3") },
                        // Reuse the same libvips decode shim as tp-llm (jpeg/webp/
                        // etc. for dropped / @mentioned images).
                        .{ .name = "vips", .module = vips_module },
                        // Platform config-dir resolution for the settings file.
                        .{ .name = "known-folders", .module = kf.module("known-folders") },
                    },
                }),
            });
            // SDL3 backend windowing system libraries (per DiffKeep's build.zig).
            if (target.result.os.tag == .linux) {
                gui_exe.root_module.linkSystemLibrary("X11", .{});
                gui_exe.root_module.linkSystemLibrary("Xcursor", .{});
                gui_exe.root_module.linkSystemLibrary("Xi", .{});
                gui_exe.root_module.linkSystemLibrary("wayland-client", .{});
            }
            // Install only via the `gui` / `run-gui` steps, never the default
            // step, so `zig build` / `zig build test` stay free of the GUI deps.
            const install_gui = b.addInstallArtifact(gui_exe, .{});
            gui_step.dependOn(&install_gui.step);

            const run_gui_cmd = b.addRunArtifact(gui_exe);
            run_gui_cmd.step.dependOn(&install_gui.step);
            if (b.args) |args| run_gui_cmd.addArgs(args);
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
                        .{ .name = "dvui", .module = dvui_dep.module("dvui_sdl3") },
                        .{ .name = "backend", .module = dvui_dep.module("sdl3") },
                    },
                }),
            });
            if (target.result.os.tag == .linux) {
                md_probe_exe.root_module.linkSystemLibrary("X11", .{});
                md_probe_exe.root_module.linkSystemLibrary("Xcursor", .{});
                md_probe_exe.root_module.linkSystemLibrary("Xi", .{});
                md_probe_exe.root_module.linkSystemLibrary("wayland-client", .{});
            }
            const run_md_probe = b.addRunArtifact(md_probe_exe);
            if (b.args) |args| run_md_probe.addArgs(args);
            md_probe_step.dependOn(&run_md_probe.step);

            // GUI config unit tests. Kept off the default `test` step (which
            // stays free of GUI deps); `config.zig` only pulls std + known-folders.
            const gui_config_tests = b.addTest(.{
                .root_module = b.createModule(.{
                    .root_source_file = b.path("src/gui/config.zig"),
                    .target = target,
                    .optimize = optimize,
                    .imports = &.{
                        .{ .name = "known-folders", .module = kf.module("known-folders") },
                    },
                }),
            });
            gui_test_step.dependOn(&b.addRunArtifact(gui_config_tests).step);

            // Tool-call (`<image>…</image>`) parser unit tests. Pure std, no
            // GUI/engine deps — kept off the default `test` step for the same
            // reason as the config tests.
            const gui_toolcall_tests = b.addTest(.{
                .root_module = b.createModule(.{
                    .root_source_file = b.path("src/gui/toolcall.zig"),
                    .target = target,
                    .optimize = optimize,
                }),
            });
            gui_test_step.dependOn(&b.addRunArtifact(gui_toolcall_tests).step);

            // Markdown parser unit tests. Pure std — the dvui-facing
            // markdown_view.zig stays out of the test build.
            const gui_markdown_tests = b.addTest(.{
                .root_module = b.createModule(.{
                    .root_source_file = b.path("src/gui/markdown.zig"),
                    .target = target,
                    .optimize = optimize,
                }),
            });
            gui_test_step.dependOn(&b.addRunArtifact(gui_markdown_tests).step);

            // Viewer zoom/pan math unit tests. Pure std — the dvui-facing
            // viewer.zig stays out of the test build.
            const gui_viewmath_tests = b.addTest(.{
                .root_module = b.createModule(.{
                    .root_source_file = b.path("src/gui/viewmath.zig"),
                    .target = target,
                    .optimize = optimize,
                }),
            });
            gui_test_step.dependOn(&b.addRunArtifact(gui_viewmath_tests).step);

            // Diffusion-engine pure-helper tests (clampDim / parseGenAttrs /
            // seed advance). Pulls in the TensorPencil module (for pipeline
            // types) + known-folders (via config.zig), but stays CPU-only.
            const gui_diffuser_tests = b.addTest(.{
                .root_module = b.createModule(.{
                    .root_source_file = b.path("src/gui/diffuser.zig"),
                    .target = target,
                    .optimize = optimize,
                    .imports = &.{
                        .{ .name = "TensorPencil", .module = mod },
                        .{ .name = "known-folders", .module = kf.module("known-folders") },
                    },
                }),
            });
            gui_test_step.dependOn(&b.addRunArtifact(gui_diffuser_tests).step);

            // Chat-session pure-helper tests (Message variants + the ‹/›
            // regenerate-navigation semantics). Same deps as the diffuser
            // tests (chat.zig imports it); CPU-only.
            const gui_chat_tests = b.addTest(.{
                .root_module = b.createModule(.{
                    .root_source_file = b.path("src/gui/chat.zig"),
                    .target = target,
                    .optimize = optimize,
                    .imports = &.{
                        .{ .name = "TensorPencil", .module = mod },
                        .{ .name = "known-folders", .module = kf.module("known-folders") },
                    },
                }),
            });
            gui_test_step.dependOn(&b.addRunArtifact(gui_chat_tests).step);
        }
    }

    // embed-bench: throughput profiler for the tp.embed encoders (per-item vs
    // batched forwards, DIFFKEEP.md M8). Only needs the TensorPencil module +
    // a device for the GPU backends. `zig build embed-bench -- [opts]`.
    {
        const eb_step = b.step("embed-bench", "Build+run the tp.embed encoder throughput profiler");
        const eb_exe = b.addExecutable(.{
            .name = "embed-bench",
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/embed_bench.zig"),
                .link_libc = true,
                .target = target,
                .optimize = .ReleaseFast,
                .imports = &.{
                    .{ .name = "TensorPencil", .module = mod },
                },
            }),
        });
        const run_eb = b.addRunArtifact(eb_exe);
        if (b.args) |args| run_eb.addArgs(args);
        eb_step.dependOn(&run_eb.step);
    }

    // ggml-bench / qgemv-bench: CPU-kernel + device GEMV benchmarks that need
    // direct `ggml` C access, so they only exist when ggml is linked
    // (`-Dggml`, default on). ggml links transitively via the TensorPencil
    // module; the shared `ggml` import gives direct C access.
    if (ggml_mod) |ggml_mod_v| {
        // ggml-bench: benchmark comparing TensorPencil's CPU quant kernels
        // against ggml's. `zig build ggml-bench`.
        const bench_step = b.step("ggml-bench", "Build+run the ggml vs TensorPencil CPU-kernel benchmark");
        const bench_exe = b.addExecutable(.{
            .name = "ggml-bench",
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/ggml_bench.zig"),
                .link_libc = true,
                .target = target,
                .optimize = .ReleaseFast,
                .imports = &.{
                    .{ .name = "TensorPencil", .module = mod },
                    .{ .name = "ggml", .module = ggml_mod_v },
                },
            }),
        });
        const run_bench = b.addRunArtifact(bench_exe);
        if (b.args) |args| run_bench.addArgs(args);
        bench_step.dependOn(&run_bench.step);

        // qgemv-bench: grouped-N dp4a quant GEMV vs dequant->f16 GEMM, on
        // device. `zig build qgemv-bench`. Measures whether grouped-N is a real
        // gain over the m>1 dequant-GEMM fallback for small (speculative-verify)
        // batches.
        const qb_step = b.step("qgemv-bench", "Build+run the grouped-N GEMV vs dequant-GEMM device benchmark");
        const qb_exe = b.addExecutable(.{
            .name = "qgemv-bench",
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/qgemv_bench.zig"),
                .link_libc = true,
                .target = target,
                .optimize = .ReleaseFast,
                .imports = &.{
                    .{ .name = "TensorPencil", .module = mod },
                    .{ .name = "ggml", .module = ggml_mod_v },
                },
            }),
        });
        const run_qb = b.addRunArtifact(qb_exe);
        if (b.args) |args| run_qb.addArgs(args);
        qb_step.dependOn(&run_qb.step);
    }

    // Creates an executable that will run `test` blocks from the provided module.
    // Here `mod` needs to define a target, which is why earlier we made sure to
    // set the releative field.
    const test_filter = b.option([]const u8, "test-filter", "Only run tests whose name matches this substring");
    const test_filters: []const []const u8 = if (test_filter) |f| &.{f} else &.{};
    const mod_tests = b.addTest(.{
        .root_module = mod,
        .filters = test_filters,
    });

    // A run step that will run the test executable.
    const run_mod_tests = b.addRunArtifact(mod_tests);

    // Each carved layer module is its own compilation, so its `test` blocks run
    // from their own test binary. tp_core first.
    const core_tests = b.addTest(.{
        .root_module = core_mod,
        .filters = test_filters,
    });
    const run_core_tests = b.addRunArtifact(core_tests);

    const ops_tests = b.addTest(.{
        .root_module = ops_mod,
        .filters = test_filters,
    });
    const run_ops_tests = b.addRunArtifact(ops_tests);

    const gpu_tests = b.addTest(.{
        .root_module = gpu_mod,
        .filters = test_filters,
    });
    const run_gpu_tests = b.addRunArtifact(gpu_tests);

    const runtime_tests = b.addTest(.{
        .root_module = runtime_mod,
        .filters = test_filters,
    });
    const run_runtime_tests = b.addRunArtifact(runtime_tests);

    const models_tests = b.addTest(.{
        .root_module = models_mod,
        .filters = test_filters,
    });
    const run_models_tests = b.addRunArtifact(models_tests);

    // Creates an executable that will run `test` blocks from the executable's
    // root module. Note that test executables only test one module at a time,
    // hence why we have to create two separate ones.
    const exe_tests = b.addTest(.{
        .root_module = exe.root_module,
        .filters = test_filters,
    });

    // A run step that will run the second test executable.
    const run_exe_tests = b.addRunArtifact(exe_tests);

    // A top level step for running all tests. dependOn can be called multiple
    // times and since the two run steps do not depend on one another, this will
    // make the two of them run in parallel.
    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_core_tests.step);
    test_step.dependOn(&run_ops_tests.step);
    test_step.dependOn(&run_gpu_tests.step);
    test_step.dependOn(&run_runtime_tests.step);
    test_step.dependOn(&run_models_tests.step);
    test_step.dependOn(&run_exe_tests.step);

    // Just like flags, top level steps are also listed in the `--help` menu.
    //
    // The Zig build system is entirely implemented in userland, which means
    // that it cannot hook into private compiler APIs. All compilation work
    // orchestrated by the build system will result in other Zig compiler
    // subcommands being invoked with the right flags defined. You can observe
    // these invocations when one fails (or you pass a flag to increase
    // verbosity) to validate assumptions and diagnose problems.
    //
    // Lastly, the Zig build system is relatively simple and self-contained,
    // and reading its source code will allow you to master it.
}
