# ZIG.md — Zig 0.16.0 Notes

Breaking changes and gotchas encountered in this codebase when working with Zig 0.16.0.
Update this file whenever a new 0.16.0 change is found or an existing workaround is resolved.

---

## Namespaces move between Zig versions — search before assuming "removed"

Zig is pre-1.0; the std namespace layout churns constantly. A decl that isn't
where an older version (or your memory) put it has usually **moved**, not been
deleted. Before concluding something is gone or writing a raw-syscall
workaround, grep the std lib for it:

```sh
grep -rn "pub const Mutex\|pub fn futexWait" "$(zig env | grep std_dir ...)"  # or ~/.zvm/<ver>/lib/std
```

Recent moves seen here: the blocking primitives (`Mutex`, `Condition`,
`Semaphore`) and the futex/sleep helpers all migrated from `std.Thread`/
`std.time` onto **`std.Io`** (they now take an `io: std.Io` argument). See the
individual entries below.

## `std.Io` futex (cross-platform, replaces `std.Thread.Futex`)

`std.Thread.Futex` is gone; use `std.Io`'s futex, which is cross-platform
(Linux futex / Windows `WaitOnAddress` / darwin ulock / WASM). The
`Uncancelable` variant is safe to call from a raw `std.Thread` (it doesn't touch
the Io runtime's task/cancellation state), so you can park/wake a plain worker
thread without a raw `linux.futex` syscall:

```zig
// consumer parks until the u32 changes from `expect`:
io.futexWaitUncancelable(u32, &wake.raw, expect);   // wake: std.atomic.Value(u32)
// producer bumps the value (release) THEN wakes:
_ = wake.fetchAdd(1, .release);
io.futexWake(u32, &wake.raw, 1);
```

---

## Self-hosted linker chokes on glibc CRT `.sframe` (recent toolchains)

On hosts with a recent glibc/binutils (e.g. Arch/CachyOS, gcc 16+), the C
runtime startup objects `Scrt1.o`/`crt1.o` carry an SFrame unwind section
(`.sframe` + `.rela.sframe`) that uses `R_X86_64_PC64` relocations. Zig 0.16's
self-hosted ELF linker (the default for native Debug builds) cannot process
that relocation type, so **every libc-linked build fails**:

```
fatal linker error: unhandled relocation type R_X86_64_PC64 at offset 0x1c
    note: in .../crt1.o:.sframe
```

The exit-1 build even writes a corrupt binary ("exec format error"). Forcing
`-flld` is not a fallback here — the bundled LLD segfaults on this system.

`.sframe` is only stack-unwind metadata for the tiny `_start` stub, so stripping
it is harmless. `tools/patch-crt.sh` builds `.zig-crt/` — a mirror of the system
crt dir with those objects sframe-stripped (via `objcopy --remove-section`) — plus
`.zig-crt/libc.txt`. `build.zig` auto-detects that file and sets `b.libc_file`
(the global libc-file fallback for all compiles), so a plain `zig build` links
cleanly. The dir is host-specific, `.gitignore`d, and a no-op on hosts with an
older toolchain (e.g. the CUDA desktop). Re-run the script after a glibc upgrade.

---

## ArrayList is now always unmanaged

`std.ArrayList` no longer stores an allocator internally. The allocator is passed at every call site.

```zig
// Old
var list = std.ArrayList(T).init(allocator);
list.append(item);
list.deinit();

// New (0.16.0)
var list: std.ArrayList(T) = .empty;
list.append(allocator, item);
list.deinit(allocator);
```

Affected methods: `.deinit(alloc)`, `.append(alloc, item)`, `.appendSlice(alloc, items)`,
`.resize(alloc, n)`, `.ensureUnusedCapacity(alloc, n)`, `.clearAndFree(alloc)`, `.orderedRemove(idx)` (no alloc needed).

## `callconv(.C)` → `callconv(.c)`

Calling convention tags are now lowercase.

```zig
fn myFn() callconv(.c) void { ... }  // was .C
```

## `std.mem.trimRight` → `std.mem.trimEnd`

```zig
std.mem.trimEnd(u8, str, "\n")  // was trimRight
```

## `std.Thread.Semaphore` removed

Use `std.atomic.Value(bool)` for a simple spinlock, or `std.Io.Semaphore` (requires an `std.Io` argument).

## `std.Thread.Mutex` removed

Use `std.Io.Mutex` instead. Lock and unlock require an `std.Io` argument:

```zig
mutex: std.Io.Mutex = std.Io.Mutex.init,

mutex.lockUncancelable(io);
defer mutex.unlock(io);
```

## `std.time.nanoTimestamp()` removed

Use the `std.Io` clock API instead:

```zig
const now_ns = std.Io.Clock.real.now(io).nanoseconds;
```

`std.time.Timer`/`std.time.Instant` and `std.posix.clock_gettime` are gone too.
Without an `io` in scope (e.g. a `test` block), reach for the raw syscall:

```zig
var ts: std.os.linux.timespec = undefined;
_ = std.os.linux.clock_gettime(std.os.linux.CLOCK.MONOTONIC, &ts);
const ns = @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
```

## File / directory I/O uses `std.Io`

Most I/O operations now require passing an `std.Io` value through the call stack.
`std.Io.Dir.cwd()` replaces `std.fs.cwd()`, and file operations take `io` as the first argument.

```zig
var dir = try std.Io.Dir.openDirAbsolute(io, path, .{});
defer dir.close(io);
var file = try dir.openFile(io, "foo.txt", .{});
defer file.close(io);
```

## `std.ArrayList` / `std.ArrayListUnmanaged` default init is `.empty`

`.{}` no longer works as a default initializer for these types; use `.empty`:

```zig
var list: std.ArrayList(T) = .empty;
var list: std.ArrayListUnmanaged(T) = .empty;
```

This also applies to struct field defaults:
```zig
items: std.ArrayListUnmanaged(T) = .empty,
```

## `std.BoundedArray` removed

Use a plain fixed-size array with a count instead:

```zig
// Was: var tasks = std.BoundedArray(T, N){};  tasks.append(v) catch {};  tasks.slice()
var tasks: [N]T = undefined;
var task_count: usize = 0;
tasks[task_count] = v; task_count += 1;  // guard task_count < N yourself
tasks[0..task_count]
```

## `std.process.run` signature changed

```zig
// 0.16.0 — takes allocator and io
const result = try std.process.run(allocator, io, .{ .argv = &.{ "git", "describe" } });
```

## `std.posix.PROT`, `std.posix.MAP` are packed structs, not flag namespaces

In 0.16.0, `PROT` and `MAP` are `packed struct(u32)` with boolean fields.
Use struct literal syntax instead of bitwise OR:

```zig
// Old (does not compile)
std.posix.PROT.READ | std.posix.PROT.WRITE

// New (0.16.0)
std.posix.PROT{ .READ = true, .WRITE = true }
// or inline as the argument:
.{ .READ = true, .WRITE = true }
```

Same applies to `MAP`:
```zig
// Old
std.posix.MAP.SHARED

// New
std.posix.MAP{ .TYPE = .SHARED }
// or inline:
.{ .TYPE = .SHARED }
```

## `std.posix.mprotect` removed

Like the other thin wrappers below, use the raw Linux syscall (takes the
packed `PROT` struct; address must be page-aligned, length is rounded up
by the kernel):

```zig
const rc = std.os.linux.mprotect(bytes.ptr, bytes.len, .{ .READ = true, .WRITE = true });
if (std.posix.errno(rc) != .SUCCESS) return false;
```

## `std.posix.write`, `std.posix.close`, `std.posix.ftruncate` removed

These thin POSIX wrappers were removed. For contexts without `std.Io`, use raw
Linux syscalls instead:

```zig
_ = std.os.linux.write(fd, buf.ptr, buf.len);
_ = std.os.linux.close(fd);
const rc = std.os.linux.ftruncate(fd, length);
if (std.posix.errno(rc) != .SUCCESS) return error.FtruncateFailed;
```

## `std.time.sleep` removed

Use `std.Io.sleep(io, duration, clock)` instead. `duration` is `std.Io.Duration{ .nanoseconds = N }`,
`clock` is `.real` (the only member of `std.Io.Clock`):

```zig
std.Io.sleep(io, .{ .nanoseconds = 1_000_000 }, .real) catch {};  // 1ms
```

## SPIR-V kernels (0.16 self-hosted backend)

- Compile with `-target spirv64-vulkan -fno-llvm`. Release modes without
  `-fno-llvm` silently pick the LLVM backend, which does not support
  `callconv(.spirv_kernel)` (and can segfault or emit empty output).
- `std.gpu.executionMode(...)` does not work ("cannot set execution mode in
  assembly"); the module comes out without `OpExecutionMode LocalSize`.
  `src/gpu/spv.zig` patches it into the binary at load time instead.
- Storage buffers: declare `extern var buf: T addrspace(.storage_buffer)` where
  `T` is an `extern struct` wrapping the array, then add the Vulkan-required
  decorations via inline asm inside the kernel. Types are passed to asm with
  the `"t"` constraint, variables with `""`:

```zig
asm volatile (
    \\OpDecorate %t Block
    \\OpMemberDecorate %t 0 Offset 0
    \\OpDecorate %v DescriptorSet 0
    \\OpDecorate %v Binding 0
    : : [t] "t" (Buf), [v] "" (&buf));
```

- Kernel entry points must not take parameters (that emits a Linkage module,
  not executable by Vulkan); use decorated globals + push constants.
- Embedding the .spv: `b.addObject` with the spirv target + `.use_llvm = false`,
  then `mod.addAnonymousImport("name", .{ .root_source_file = obj.getEmittedBin() })`
  and `@embedFile("name")`.
- Workgroup (shared) memory: needs SPIR-V >= 1.4 in the target
  (`.cpu_features_add = std.Target.spirv.featureSet(&.{.v1_4})`; plain
  `-mcpu baseline` fails with "storage class must be ... but is workgroup").
  Shared arrays must be 1-D (multi-dim arrays are rejected) and may be
  referenced from only ONE entry point per module — split kernels into one
  module each. The backend also emits invalid duplicate layout decorations
  and ArrayStride on workgroup arrays; `src/gpu/spv.zig` scrubs the module
  into spirv-val-clean shape at load time.
- Known open issue: even validator-clean workgroup-memory kernels hang the
  NVIDIA 580 driver (VK_ERROR_DEVICE_LOST at dispatch, even at workgroup
  size 1x1), while RADV and llvmpipe execute them correctly. Non-workgroup
  kernels run fine on NVIDIA. Tracked in PLAN.md M9. **This is specific to
  Zig-BACKEND-EMITTED workgroup memory** — hand-authored SPIR-V using workgroup
  memory (`coopmat.zig buildGemmShared`) runs fine on the same driver, so the
  fault is in the Zig codegen's workgroup-memory output, not the driver or
  workgroup storage itself.
- Escape hatch — **subgroup ops instead of workgroup memory** (VERIFIED on
  580.173.02): subgroup-scope ops (`OpGroupNonUniformFAdd`, `…Shuffle*`, coop
  matrix) use no workgroup storage class, so they sidestep the hang. Emit them
  from Zig via inline asm exactly like `OpSDot` — e.g.
  `%r = OpGroupNonUniformFAdd %f32t %scope Reduce %val` with `[scope] "" (@as(u32,3))`
  (Subgroup scope materializes as a constant id); inject the `GroupNonUniform`
  (61) + `GroupNonUniformArithmetic` (63) capabilities via `spv.withCapabilities`.
  This is the path to a CUDA-style warp-cooperative GEMV (row-major weights,
  coalesced, no `_t` transpose) without shared memory. Confirmed by the
  `"subgroup reduce runs on device"` test (one 32-lane subgroup sums 32 ones →
  32.0; a hang would surface as DEVICE_LOST, not a wrong value).
- Note: the Zig SPIR-V backend **segfaults compiling a single-kernel module**
  that uses a subgroup/dp4a inline-asm op; the same kernel compiles fine in a
  multi-kernel module. Park such probes alongside other kernels (e.g. `dp4a.zig`).

## `@min`/`@max` with a comptime bound narrows the result type

`@min(512, x)` where `x: usize` yields a `u10` (smallest type holding 0..512), so
subsequent arithmetic happens in that narrow type and can overflow where the
"same" expression in `usize` would be fine:

```zig
const kl = @min(KC, cols - kc0);   // u10 when KC = 512!
_ = kl * NR;                       // panics: integer overflow in u10

const kl: usize = @min(KC, cols - kc0);  // fix: annotate the result type
```

The panic's source location points at the arithmetic, and the printed values
look impossible — check the *type* of the operands (disassembly shows a
narrow `mulw`/mask if it happened).

## Tests get an `Io` via `std.testing.io`

Tests that need file I/O use the testing-provided instance (only valid inside tests):

```zig
const io = std.testing.io;
var tmp = std.testing.tmpDir(.{});
defer tmp.cleanup(); // no io argument
try tmp.dir.writeFile(io, .{ .sub_path = "f", .data = bytes });
```

## `std.Io.Dir` has no `realpathAlloc`

`fs.Dir.realpathAlloc` did not survive the move to `std.Io.Dir`. Instead of resolving
an absolute path, pass the directory itself and open relative to it
(e.g. `SafeTensors.openIn(gpa, io, dir, sub_path)`).

## `std.process.Child.Term` tags are lowercase in 0.16.0

```zig
// Old
if (result.term != .Exited or result.term.Exited != 0) { ... }

// New (0.16.0)
if (result.term != .exited or result.term.exited != 0) { ... }
```

## `Io.Reader.takeDelimiterExclusive` does not consume the delimiter

`takeDelimiterExclusive('\n')` returns the line and tosses only `line.len`
bytes — the `\n` stays buffered, so the next call returns an empty line
forever (infinite loop hazard in read-line REPLs). For line reading use
`takeDelimiter`, which consumes the delimiter and returns `null` at EOF:

```zig
var stdin_reader: Io.File.Reader = .initStreaming(.stdin(), io, &buf);
const stdin = &stdin_reader.interface;
const line = (try stdin.takeDelimiter('\n')) orelse break; // null = EOF
```

## `zig test src/root.zig` needs `-lc` for dlopen-based (GPU) tests

The CUDA/Vulkan backends `dlopen` their drivers. Without libc, `std.DynLib`
falls back to Zig's own ELF loader, which cannot load glibc-heavy libraries
like `libcuda.so.1` — `Api.load()` fails and every GPU-gated test silently
SKIPs. `zig build test` links libc already; for direct single-file runs use:

```sh
zig test -lc src/root.zig --test-filter "<name>"
```

## `zig build test` prints "failed command:" for long-running tests (cosmetic)

A test binary containing a test that runs for a long stretch without output
(e.g. the ~90 s Debug krea2 parity test) makes `zig build test` print
`failed command: ...zig-cache/o/<hash>/test ... --listen=-` — with no test
name, no error, and **exit code 0**. The test actually ran to completion
(strace shows the binary exiting 0 after finishing), all tests pass when
the binary is run directly, and an immediate rerun reports the step as
cached success. Reproduced at a clean checkout (de2a5a0), independent of
the `testdata/gpu-tests` marker; `--test-timeout 10m` does not change it —
it appears to be the 0.16 build-runner's listen-protocol response-timeout
handling misreporting a silent-but-alive test runner. Judge test runs by
the exit code (or run the binary directly with
`zig test -lc src/root.zig --test-filter "<name>"`), not by the presence
of this line.

## `@min`/`@max` narrow their result type — `@min(65, x) * 3` overflows u7

`@TypeOf(@min(65, some_usize))` is **`u7`**, not usize: when one operand is
an untyped `comptime_int`, `@min`/`@max` narrow the result type to fit the
comptime bound (accepted proposal ziglang/zig#14039, since 0.11). So

```zig
const n = @min(prefill_chunk, ids.len - off); // n: u7 when prefill_chunk = 65!
try stepBatch(x, pos3s[0 .. n * 3]);          // n*3 is u7 math: 195 overflows
```

is an arithmetic overflow by the language rules. In ReleaseSafe it panics
"integer overflow"; in ReleaseFast it silently truncates — slice lengths
arrived as `(n*3) & 0x7f` (n=56 → 40, not 168), which cost a full day of
"memory corruption" chasing. Two traps stacked:

1. The overflow is invisible at the use site — nothing looks narrower than
   usize, and the panic only fires when the runtime value is big enough
   (n ≥ 43 here), so small tests pass.
2. In optimized builds all overflow checks in a function share one panic
   block, so the ReleaseSafe panic reports an **arbitrary unrelated source
   line** (we saw `Allocator.zig rawFree`, unrelated `@intCast` lines).
   Don't trust panic line numbers through shared panic pads; find the real
   site by breakpointing the `jo`/`jb` jumps into the pad (objdump for the
   pad address), or by checking `@TypeOf` on suspect expressions.

Fix: annotate the result type at the `@min` — `const n: usize = @min(...)`.
The result is then usize and all downstream arithmetic is full-width.
Audit hint: any `@min(comptime_const, runtime)` (or `@max`) whose result
feeds arithmetic or slice bounds needs the annotation.

## Passing tests must be silent on stderr — else `zig build test` prints a spurious red "failed command:" line

Any test that writes to stderr **while passing** (e.g. a `std.debug.print`
diagnostic) makes the build runner emit output that looks exactly like a
failure — a dim step subtree, the captured stderr, and a red
`failed command: .../test --listen=-` line — even though the step
succeeded and the summary says `N passed; 0 failed`. Don't burn time
debugging it: check `zig build test --summary all`; if it says
`test success`, nothing failed.

Mechanism (Zig 0.16 cosmetic bug): `Step/Run.zig` sets
`step.result_failed_command` before spawning every test binary and never
clears it on the `zig_test` success path; `build_runner.zig`'s `makeStep`
routes ANY step with non-empty `result_stderr` through
`printErrorMessages` ("no matter the result"), which unconditionally
appends the `failed command:` line when `result_failed_command` is set
(default `verbose` error style).

Rule: tests print diagnostics **only on failure**, via `errdefer`:

```zig
errdefer std.debug.print("parity: max_err={d:.6}\n", .{max_err}); // silent when green
try std.testing.expect(max_err < 5e-3);
```

The print then lands inside the real failure report where it's useful.
(`ZIG_BUILD_ERROR_STYLE=minimal` only hides the `failed command:` line,
not the stderr dump — fix the test, not the env.)

## `var x: T = undefined` bypasses EVERY field default

A struct's `field: T = default` only runs for struct-literal initialization. A
value declared `undefined` and then filled field-by-field keeps garbage in every
field the code forgets to assign.

```zig
const LM = struct {
    lm: *const Model,
    bidir_prefill: bool = false, // NOT applied below
};

var self: LM = undefined;
self.lm = lm;                    // bidir_prefill is now whatever was on the stack
```

The garbage is a plausible value, not a crash: an unset `bidir_prefill` ran TEXT
prefill under the image block's bidirectional attention mask in about half of
all processes, and the model stopped reasoning. It is invisible to buffer
zeroing and to compute-sanitizer (the field is a host struct member), and stable
under `setarch -R`, which is the signature of an uninitialised SCALAR.

Use `core/init_defaults.zig` (`tp_core.init_defaults`): `var self: T = init_defaults.of(T)` applies every
declared default and leaves the rest undefined, so an init keeps its
field-by-field shape without the class of bug.

## A slice of an inline array in a by-value return dangles at the `return`

`WeightStore.get` hands back a `TensorView` **by value**, and `view.info.shape`
holds its dims in an inline `[8]usize`. So `view.info.shape.slice()` points into
that temporary. Using it while the view is still in scope is fine, which is what
every model loader does:

```zig
const view = store.get(name) orelse return error.MissingTensor;
const shape = view.info.shape.slice();   // fine: `view` outlives `shape`
```

Wrapping the same two lines in a helper that returns the slice does not:

```zig
fn shapeOf(store: WeightStore, name: []const u8) ![]const usize {
    const view = store.get(name) orelse return error.MissingTensor;
    return view.info.shape.slice();      // `view` dies HERE
}
```

The read-back is `0x5555555555555555` (6148914691236517205), not a segfault, so
it surfaces as a nonsense dimension somewhere downstream rather than at the bad
access. It is also position-dependent: the first few callers happened to read
intact stack, and only a value captured early and used late came back garbage.

Copy the dims out instead of borrowing them. Applies to any accessor returning a
slice of a field of a by-value struct, not just `Shape`.

## `Dir.makePath` → `Dir.createDirPath`

`std.Io.Dir` spells the recursive-mkdir as `createDirPath(dir, io, sub_path)`.
The single-level form is `createDir(dir, io, sub_path, permissions)`. There is no
`makePath` / `makeDir` any more, so the old names fail to compile rather than
silently doing nothing.

## `{e}` is the scientific-notation specifier, not `{d:.3e}`

`std.fmt` rejects `{d:.3e}` with "extraneous trailing character 'e'". Use `{e}`
for scientific notation (precision is not settable the same way) or `{d}` /
`{d:.3}` for decimal.

## A truncating integer divide in emitted kernel code is silent, not fatal

`buildPrep` sized its FWHT work as `ngroups * 64 / 256` butterflies per thread.
Where that divides exactly the kernel is correct; where it does not, the emitted
loop simply runs one iteration short and the tail of every row keeps its
unrotated basis. No error, no assert, no NaN — just a GEMM in the wrong basis,
22% off, on one model and not the others.

The file already guarded the `== 0` case ("would silently skip the rotation
entirely") and reasoned about exactly this hazard, but only at the boundary. A
truncated non-zero count is the same bug with a smaller blast radius, which is
what made it survive.

When emitting code, any `a / b` that decides HOW MUCH WORK a thread does needs
`(a + b - 1) / b` plus a tail guard, or an explicit refusal. Prefer exposing the
count as a plain function and testing the covering property without a device —
`prepButterflyIters` and its test are the pattern.

## A reference's "tiling" flag can be semantic, not an optimization

MiniMax H3's video VAE defaults to `tiling = True`, and its decode entry point
therefore ALWAYS goes through the tiled path: 256 px tiles, >=64 px overlap,
blended. That is not a memory bound to be skipped on a big GPU — the transformer
inside never sees more than `tile / patch` tokens per axis, and running it on a
whole frame produces per-patch incoherence (a visible grid wherever the image has
detail).

It hid because a frame at or below the tile size is exactly ONE tile, so the
small shapes every fixture used were identical either way, and the fixture
generator had set `tiling = False` to "simplify".

Before treating a reference's tiling/chunking as an optimization, check what its
own default entry point calls, and make at least one fixture LARGER than the tile.

## MP4 drops unknown metadata keys, silently

`av_dict_set(&fc->metadata, "parameters", ...)` then `avformat_write_header` writes
NOTHING for an MP4: isom keeps only the handful of keys it defines, and an
unrecognized one is discarded with no error and no warning. Pass the muxer option
`movflags = use_metadata_tags` to `avformat_write_header` and unknown keys go into
a udta atom instead.

Check a metadata write actually landed (`ffprobe -show_entries format_tags`)
rather than trusting the return code — it is 0 either way.

## cuBLASLt's `beta` is a free accumulate, and it matters more than the GEMM

`cublasLtMatmul` takes C and D separately, but our `ltRun` already hands it the
same buffer and layout for both, so `beta = 1` turns any GEMM into
`D += alpha * (A @ B)` at no cost beyond reading D. The plan cache is keyed on
shape, not on beta, so an existing cached plan serves both.

The reason to reach for it: a GEMM followed by a separate scaled-add kernel is
three extra passes over the f32 output plane (write the scratch, read it,
read-modify-write the destination) on top of the GEMM's own write. For the LoRA
sidecar at MiniMax H3's widths that was ~70 GB of DRAM traffic per step, measured
at roughly TWICE the two GEMMs it served. It also means a scratch plane that does
not need to exist (205 MB at H3's native canvas).

Two traps around measuring this:

- A FLOP count cannot see it. The estimate that motivated the sidecar said 3-4%
  of a step; the GEMMs alone were +7% and the unfused accumulate another +19%.
- The hand-PTX `hgemm` writes its C tiles unconditionally, so the `zig-cuda` arm
  has no equivalent. `opGemmBf16Acc` returns `error.UnsupportedKernelArm` there
  rather than overwriting, because silently overwriting would drop the base GEMM's
  output — a plausible wrong render.

## A whole-render wall clock measures the page cache, not the change

Timing a multi-GB-model render end to end on a box whose free RAM is smaller than
the checkpoint gives a number dominated by first-touch reads. Observed: the SAME
1-step configuration took 42 s and 131 s in two runs an hour apart, and a 24-step
render came out FASTER than an 8-step one.

Difference step counts, or better, instrument the loop and print the per-step time
(`generateClip` now does). Even then, interleave the A and B configurations within
a round and repeat: this 3090's steady-state step time drifts ±30% between rounds
while a no-LoRA baseline held 0.78-0.81 s, so only within-round pairs compare.

## cuBLASLt has no f32 arm until you add one, and TF32 is not it

`ltPlan` in `gpu/cuda/backend.zig` had `LtKind = { i8, f16, bf16 }`, so the only
f32-precision GEMM available was a one-thread-per-output kernel. Adding an `.f32`
arm is four lines (`R_32F` for A/B and D, `COMPUTE_32F`, `R_32F` scale) and gives a
properly tiled f32 GEMM.

Use `CUBLAS_COMPUTE_32F`, **not** `CUBLAS_COMPUTE_32F_FAST_TF32`. TF32 keeps 10
mantissa bits, which is the same precision as the f16 path an f32 arm exists to
avoid; the point is the other 13 bits, not the tensor cores.

Measured on MiniMax H3's BigVGAN vocoder (`minimax-h3-audio-cuda-test`), device
against the f32 CPU reference:

| conv GEMM | rel L2 | worst sample | time at 37 latent frames |
|---|---|---|---|
| f16 tensor cores | 2.4e-3 | 8.3e-3 | 82 ms |
| f32 cuBLASLt | 9.5e-6 | 2.8e-5 | **75 ms** |
| f32 naive kernel | 9.5e-6 | 2.8e-5 | 518 ms |

Two things worth carrying: for a VOCODER, -42 dB with 74% of samples past 1e-4 is
an audible noise floor, not "close enough"; and the tensor-core path was **not
faster** at these widths, because the per-conv weight-pad and activation-convert
passes cost more than the MMA wins. Reaching for f16 because it is the fast option
was wrong on both axes.

## A 1-D conv's replicate padding and slicing vanish if you write it as a gather

BigVGAN's anti-aliased activation is seven ops in the reference: pad, stride-2
transposed conv, slice, scale by the ratio, snake, pad, stride-2 conv. Written as a
gather over the OUTPUT index it is two kernels with one intermediate: replicate
padding becomes `max(0, min(len-1, i))` on the source index, the slice becomes a
constant added to it, and a transposed conv's contributing taps at output `t` are
exactly `j == (t + pad) mod stride`, i.e. `k / stride` of them.

The same trick removes the atomics a scattered transposed conv would need. And
device signals want to be CHANNEL-LAST for it: a warp then covers consecutive
channels at one time step, which is consecutive memory, and it removes the
transpose an im2col GEMM otherwise pays on both sides. The conv weight has to be
permuted to match the patch matrix's column order (`(tap, in_ch)`), which is a
load-time pass, not a kernel.

## A conditioned output's level is not a bug until the conditioning is swept

MiniMax H3's audio came out at -52 dBFS and I spent a round concluding the sampler's
audio carry was wrong: swept the flow shifts, instrumented the latent handed to the
vocoder, checked the carry against the reference line by line. All of it held up,
which should have been the signal.

The level is conditioned by the PROMPT. The same seed and shifts with an explicit
`overall_soundscape:` section render at -17.7 dBFS instead of -52.4 — **35 dB from
the text alone**. The prompt I had been using for every render in the session
described a fox in snow and no sound at all.

Two transferable mistakes:

- **I swept an axis the quantity does not ride** (the shifts) while holding fixed the
  axis it does (the prompt), and read the resulting flatness as evidence of a defect.
  A flat sweep is evidence the axis is wrong, not that the code is.
- **I measured a proxy and treated it as the quantity.** The audio latent's rms moves
  0.327 -> 0.395 across that 35 dB output swing, so it could never have answered the
  question I was asking it.

Before calling a generative model's output level, colour, or sharpness a defect,
vary the conditioning first, and measure the output rather than an intermediate.
