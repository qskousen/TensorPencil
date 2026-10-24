# ZIG.md — Zig 0.17.0 Notes

Breaking changes and gotchas met in this codebase, on Zig 0.17.0 and the 0.16.0 changes
that still apply. Add a change here when you meet a new one, and delete a workaround
when it is resolved.

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

## `@import("../x.zig")` above a module root is an error

A module's file set is rooted at its root source file's DIRECTORY, and
`@import` of a file outside it fails with "import of file outside module
path". A test unit or probe rooted at `src/client/selection.zig` therefore
cannot reach `src/shared/config.zig` by a relative path even though the tp-gui
exe (rooted at `src/gui_main.zig`) can. Split by directory means split by
module: `build.zig` creates `shared` / `engine` / `client` modules and code
writes `@import("shared").config`. A file may belong to ONE module: a relative
import of a file that another module also owns compiles it twice, and the two
copies of every type in it are distinct.

## Zig 0.17.0: what changed here

Each of these broke this tree or is deprecated for removal in 0.18. The release notes
miss several of them (the std renames especially); ggufy's 0.17 commit is a second
reference.

Language:
- `a ** n` is gone. One element: `const x: [n]T = @splat(v)` (or `@as([n]T, @splat(v))`
  where nothing gives the type). A string: `strings.repeat("ab", 32)` (`tp_core`).
- `errdefer |err|` is gone: move the body into an inner function and `catch` at the call.
- `@cImport` is gone: build.zig translates the header with the `translate_c` package
  (`Translator`) and the module imports the result by name. The `zig-0.17.x` branch tip
  needs a newer std than 0.17.0 (`avx512bmm`); the pinned commit is the one that builds.
- `@typeInfo` is struct-of-arrays: `field_names`, `field_types`, `field_attrs` (struct,
  union), `field_values` (enum), `decl_names`. Iterate them zipped:
  `inline for (info.field_names, info.field_types) |name, T|`. A default is
  `info.field_attrs[i].defaultValue(info.field_types[i])`.
- `@bitCast` of an extern struct is an error; `std.mem.toBytes` / `bytesToValue` keep
  the old memory reinterpretation. On arrays and vectors it now concatenates LOGICAL
  bits, which differs from 0.16 only for elements with padding bits (`u4`, `bool`,
  `u24`); none are bitcast here.
- `@hasDecl` is false for a non-`pub` decl even in the same file. Cross-file it already
  was, so a capability check on another file's type keeps working only if the decl is
  `pub`.
- `@intFromEnum` / `@enumFromInt` are deprecated: `@backingInt(e)` and
  `@fromBackingInt(@intCast(i))` (the argument must be the exact backing type).
- `std.builtin` is `std.lang`; `OptimizeMode` is `Optimize` with tags `.debug`, `.safe`,
  `.fast`, `.small`, and `builtin.mode.runtimeSafety()` replaces
  `std.debug.runtime_safety`. `@import("builtin").cpu/os/abi` move under `.target`.

std (renames unless noted):
- `fmt.allocPrint(a, ...)` -> `a.print(...)`; `fmt.bufPrint` -> `mem.print`;
  `fmt.bufPrintZ(b, f, x)` -> `mem.printSentinel(b, f, x, 0)`;
  `a.dupeZ(u8, s)` -> `a.dupeSentinel(u8, s, 0)`.
- `mem.indexOf*` -> `mem.find*` (`lastIndexOfScalar` -> `findScalarLast`).
- `std.fs.path` -> `std.Io.Dir.path`; `std.fs.max_path_bytes` -> `std.Io.Dir.max_path_bytes`.
- `mem.copyForwards(T, d, s)` -> `@memmove(d, s)`, which needs EQUAL lengths: slice the
  destination to the source's length (a longer one panics, it did in `serve/queue.zig`).
- `meta.Int` -> `@Int`; `meta.fieldNames` -> `@typeInfo(T).@"enum".field_names`.
- `StaticBitSet` -> `bit_set.Static` and friends; `.initEmpty()` -> `.empty`.
- `ArrayList.getLast()` -> `last().?`; `*Unmanaged` container aliases -> the plain
  names (`array_hash_map.String`); an unmanaged hash map initializes as `.empty`, never `.{}`.
- `unicode.utf8Decode` is deprecated with no direct successor: `strings.utf8DecodeSequence`
  keeps its behaviour (a lone byte decodes to itself), or use `Utf8View`.
- `testing.allocator` is a `SafeAllocator`: it never reuses memory.

Build system:
- `build` may return `!void`; `try b.dependencyLazy(...)` replaces the
  `if (b.lazyDependency(...)) |d|` dance (the build fetches and reruns).
  Declare every `b.option` before the first `dependencyLazy`: on a checkout missing
  that package, `build` returns there and the runner rejects any `-D` flag declared
  after it with `invalid option` instead of fetching. Only a fresh checkout shows it
  (CI); a tree with every package in `zig-pkg/` never returns early.
- `b.args` is gone: `run.addPassthruArgs()`.
- `b.build_root` -> `b.root` (a `Cache.Path`); `b.libc_file`, `b.sysroot`,
  `b.pathFromRoot`, `b.getInstallPath`, `Run.addPathDir` and custom steps are gone. An
  installed binary is run with `b.addRunFile(b.graph.path(.install_bin, name))`.
- `Run.add*Arg` -> `add*Arg2(..., .{ .prefix = ... })`.
- build.zig runs in a separate configure process whose result is cached: anything it
  reads from the filesystem must be declared with `b.dependOnFileContents` /
  `dependOnDirectoryContents`, or a change to it is missed.
- A test binary for a foreign target is no longer skipped: its Run fails unless
  `skip_foreign_checks` is set (`testRun` in build.zig).
- `zig build -k` is gone; a failed step no longer stops the others.
- The ELF linker now handles the `R_X86_64_PC64` relocations in recent glibc's crt
  `.sframe`, which needed a patched crt under 0.16.

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
if (std.os.linux.errno(rc) != .SUCCESS) return false;
```

## `std.posix.write`, `std.posix.close`, `std.posix.ftruncate` removed

These thin POSIX wrappers were removed. For contexts without `std.Io`, use raw
Linux syscalls instead:

```zig
_ = std.os.linux.write(fd, buf.ptr, buf.len);
_ = std.os.linux.close(fd);
const rc = std.os.linux.ftruncate(fd, length);
if (std.os.linux.errno(rc) != .SUCCESS) return error.FtruncateFailed;
```

Decode a raw syscall's result with `std.os.linux.errno`, never `std.posix.errno`.
The latter is `system.errno`, which is libc's when the binary links libc: it reports
an error only for a return of exactly -1, and a raw syscall returns `-errno` in the
top of `usize`. Every failure then reads as `.SUCCESS`, and the `@intCast` of the
"fd" that follows panics. A test binary that links libc (`gui-test` does, through
SDL) hits it on any machine missing the file, which is how CI found it.

## `std.time.sleep` removed

Use `std.Io.sleep(io, duration, clock)` instead. `duration` is `std.Io.Duration{ .nanoseconds = N }`,
`clock` is `.real` (the only member of `std.Io.Clock`):

```zig
std.Io.sleep(io, .{ .nanoseconds = 1_000_000 }, .real) catch {};  // 1ms
```

## SPIR-V kernels (self-hosted backend)

The bullets from "Workgroup (shared) memory" on were found on 0.16; 0.17 fixed 22 SPIR-V
backend bugs and lists new regressions (#37050), so treat each as unverified until it bites
again.

- Compile with `.use_llvm = false` (`-fno-llvm`). Release modes without it silently pick
  the LLVM backend, which does not support `spirv_kernel` (and can segfault or emit
  empty output).
- The calling convention carries the workgroup size,
  `callconv(.{ .spirv_kernel = .{ .x, .y, .z } })`, and the compiler emits
  `OpExecutionMode LocalSize` from it. Ours is a placeholder (`kernels/abi.zig`):
  `spv.withLocalSize` overwrites it with the size the device gets at load.
- **Inline-asm `OpDecorate` on a VARIABLE is silently dropped** (on a type it still
  lands), so a buffer decorated that way has no binding and every buffer aliases
  binding 0. Declare it with
  `@extern(*addrspace(.storage_buffer) T, .{ .name, .decoration = .{ .descriptor = .{ .set, .binding } } })`
  (`abi.buffer`); the compiler emits Block, Offset and ArrayStride itself. The
  `gpu/context.zig` test that every kernel's bindings are tracked is what caught it.
- 0.17 inlines a kernel's whole body into its entry point, and the NVIDIA driver (580)
  builds slower code from that than from an entry that calls one function: 4-9% on
  several dual GEMVs, from SPIR-V otherwise identical in instructions and order.
  `dual.zig`'s `Entry` keeps the body behind `@call(.never_inline, ...)`. The other
  kernel modules showed no difference (`vk-prep-bench`, `anima-vk-bench`).
- A built-in input is decorated by its extern NAME: `global_invocation_id` (std.spirv),
  and `subgroup_size`, `subgroup_local_invocation_id`, `subgroup_id`, `num_subgroups`.
- Capabilities cannot be declared in inline asm; they follow the target's cpu
  features. Ours are added to the module at load (`spv.withCapabilities`), per device.
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
- The backend also **segfaults on a real function call from a kernel into a
  function that reads or writes a storage buffer** (a helper taking a zero-sized
  struct, or no argument at all, both crash). Mark every such helper `inline`;
  then the same code compiles. `kernels/dual.zig` relies on this.
- A `[*]addrspace(.storage_buffer) f32` many-pointer cannot be indexed ("cannot
  access element of logical pointer"); index the extern struct's array directly.
- **The NVIDIA driver's SPIR-V compiler (580) segfaults at `CreateComputePipelines`**
  on a kernel whose loop body issues eight strided storage-buffer loads and then
  eight stores (an unrolled load-then-store apply), whether the temporaries are a
  local array, flat arrays or plain SSA values. The same unroll with loads only
  (`dual/rows.zig` `stridedSum`) compiles and runs. `te-test` on the vulkan arm is
  the quickest bring-up probe: the crash is a SEGV inside `libnvidia-glvkspirv`
  before any kernel runs.

## NVPTX kernels (LLVM backend)

- build.zig emits `dual.ptx` as the object's assembly (`addObject` for `nvptx64-cuda`,
  `getEmittedAsm`), with the PTX ISA version the hand kernels use as a cpu feature
  (`-mcpu=sm_86+ptx80`): the cpu model alone gives an older ISA (7.1 for sm_86), and
  this box's ptxas (12.x) rejects the 8.7 `zig cc -march=sm_86` would emit.
- 0.16's LLVM backend aliased every exported kernel to a private function, which NVPTX
  refuses (`LLVM ERROR: NVPTX aliasee must be a non-kernel function definition`), so
  the PTX came from IR through a rename tool and `zig cc`. 0.17 emits no alias; the
  direct PTX is bit-identical in results and equal in speed (`tp-llm --bench` hashes).
- `@exp` has no libcall on nvptx ("no libcall available for fexp"); use
  `@exp2(x * log2e)`, which lowers to `ex2.approx` like the hand PTX does.
- A kernel `[*]addrspace(.global) f32` parameter comes out as
  `.param .u64 .ptr .global`, and scalar params in order, so a Zig kernel taking
  `(a, b, c, d, u0..u5, f0, f1)` matches `cuda/elt.zig`'s launcher ABI as is.
- Thread index: `@workGroupId(0) * @workGroupSize(0) + @workItemId(0)`
  (`std.spirv.global_invocation_id` is SPIR-V only).

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

## `std.fmt.bufPrint`'s error hands back nothing usable; `Io.Writer.fixed` truncates

`bufPrint` returns `error.NoSpaceLeft` when the output does not fit, and there is
no way to ask how much it wrote. `catch buf[0..]` therefore hands on the WHOLE
array with an uninitialised tail, which is how 240 bytes of stack reached a
toast. For "print what fits", use a fixed writer and take what it buffered:

```zig
var w = std.Io.Writer.fixed(&buf);
w.print(fmt, args) catch {};   // over-long input just stops here
return w.buffered();           // exactly the bytes written
```

## `std.Io.Dir` has no `realpathAlloc`

`fs.Dir.realpathAlloc` did not survive the move to `std.Io.Dir`. Instead of resolving
an absolute path, pass the directory itself and open relative to it
(e.g. `SafeTensors.openIn(gpa, io, dir, sub_path)`).

## `std.process.Child.Term` tags are lowercase

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

## `@min`'s narrowing fires on a comptime-KNOWN const, not just a literal

ZIG.md already records `@min(65, x)` giving a `u7`. The same narrowing happens
when the bound is a named `usize` const whose value the compiler knows, which
looks nothing like a literal at the use site:

```zig
const band: usize = @max(1, some_comptime_expr);  // 661, comptime-known
...
const nb = @min(band, n - l0);   // nb is u10, NOT usize
const span = nb * hidden;        // u10 math: 48 * 384 = 18432 overflows
```

Both operands are declared `usize`, so nothing reads as narrow; the value of
`band` is what narrows it. In ReleaseSafe the panic pointed at
`Allocator.free`, and even in Debug it pointed at the multiply while the
printed operands (48 and 384) looked impossible.

The rule is the same as before and the audit needs widening: annotate the
result type on any `@min`/`@max` whose result feeds arithmetic or a slice
bound, whenever EITHER operand has a comptime-known value -- not only when one
is written as a literal.

It bit a second time in the same week, on a DEVICE buffer size:

```zig
const nerf_band: usize = 704;              // comptime-known
const band = @min(nerf_band, n);           // u10
try be.ensureDeviceBuffer(&bufs.y, band * 32 * 256 * 4);   // truncates
```

There the symptom was neither a panic nor a wrong number: the size came out
tiny, the allocation "succeeded", and the kernel wrote to a device pointer of
exactly 0 -- `CUDA_ERROR_ILLEGAL_ADDRESS`, reported at whatever launched NEXT
because a CUDA fault is sticky. `compute-sanitizer --tool memcheck` names the
real launch and the faulting address in one run and is the first thing to
reach for; `/usr/local/cuda/bin/compute-sanitizer` is the working one on this
box (the `/usr/bin` wrapper cannot find its injection library).

## Passing tests must be silent on stderr — else `zig build test` prints a spurious red "failed command:" line

Any test that writes to stderr **while passing** (e.g. a `std.debug.print`
diagnostic) makes the build runner emit output that looks exactly like a
failure — a dim step subtree, the captured stderr, and a red
`failed command: .../test --listen=-` line — even though the step
succeeded and the summary says `N passed; 0 failed`. Don't burn time
debugging it: check `zig build test --summary all`; if it says
`test success`, nothing failed.

Mechanism (a cosmetic bug, still in 0.17's maker): the Run step records its
command line as the failed command before spawning every test binary and never
clears it on success, and any step with stderr is printed through the error
path, which appends that line.

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

`gpa.create(T)` is the same on the heap, and worse: the recycled allocation holds
live-looking POINTERS, so an unset `?T` field reads as non-null and `deinit` frees a
stale one. A cold start survives it (fresh pages read as zero) and the second load
crashes, which reads as a teardown-order bug in whatever it frees.

Use `core/init_defaults.zig` (`tp_core.init_defaults`): `var self: T = init_defaults.of(T)` applies every
declared default and leaves the rest undefined, so an init keeps its
field-by-field shape without the class of bug. On a `create`d pointer call
`init_defaults.applyTo(self)`, which does it in place and so does not push a
large struct through a stack temporary.

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

## A Debug `std.json` struct parse needs a stack of about fields × `@sizeOf(T)`

`parseFromSliceLeaky(Config, ...)` segfaulted inside `innerParse` on a `std.Thread`
with no message: in Debug the generated per-field code keeps a copy of the whole
struct per field in ONE frame, and 63 fields of a 250 KB struct is the thread's
default 16 MiB. The main thread's tests passed, so the crash only showed on the
engine thread. Spawn any thread that parses a big struct with
`.stack_size = config.parse_stack_size` (64 MiB); the regression test in
`shared/config.zig` does the parse on such a thread.

## A refused unix connect is `error.Unexpected` (errno 111), with a Debug trace

`Io.Threaded`'s unix-socket connect switch has no `CONNREFUSED` arm, so
connecting to a socket file whose listener died returns `error.Unexpected`,
and in Debug `posix.unexpectedErrno` prints "unexpected errno: 111" plus a stack
trace first. Nothing crashed: the error is returned. Code probing a stale socket
(`serve/link.zig`'s `listenLocal`) must catch every error, not `ConnectionRefused`.

## `process.Child.kill` reaps; a `wait` after it asserts

`Child.kill(io)` terminates the child AND waits for it, leaving `child.id` null
(and closing its pipes). A later `child.wait(io)` hits `assert(child.id != null)`
and aborts. A teardown that may run after a kill checks `c.id != null` before
waiting; `client/remote.zig`'s `deinit` is the case.

## `@Type` is gone; a struct is reified with `@Struct`

`@Struct(.auto, null, &names, &types, &attrs)` builds a struct type from
parallel arrays: the field names, their types, and one
`std.lang.Type.Struct.FieldAttributes` each (`default_value_ptr`,
`@"align"`, `@"comptime"`). A field copied from an existing struct carries its
default by passing its `field_attrs[i]` through. Integers are `@Int(.unsigned,
bits)`, enums `@Enum`. The name lookup over a 60-field struct needs
`@setEvalBranchQuota` in the builder. `shared/config.zig`'s `Subset` is the
in-tree example.

## Unix sockets, HTTP server errors, and waking a blocked socket reader

- `Io.net.UnixAddress.init(path)` then `.listen(io, .{})` / `.connect(io)`; support
  is per Io BACKEND (`Threaded` has it, `Uring`/`Dispatch` return
  `AddressFamilyUnsupported`, `Kqueue` panics), and `Io.net.has_unix_sockets` is the
  comptime OS gate. `listen` does not unlink a leftover socket file: `AddressInUse` on
  a path nobody answers means delete and retry (`serve/link.zig`).
- `std.http.Server.receiveHead`'s error set names a closed peer as
  `HttpConnectionClosing` or `HttpRequestTruncated`, never `EndOfStream`; matching on
  the latter is a compile error.
- `Stream.close` does not wake a thread blocked in a read on that socket;
  `Stream.shutdown(io, .both)` does. Shut down, join, then close.
- `std.process.spawn(io, .{ .argv, .stdin = .pipe, .stdout = .pipe })` returns a
  `Child` whose `stdin`/`stdout` are `?Io.File`; the parent's write end of stdin is the
  child's liveness signal (EOF when the parent dies), since `SpawnOptions` has no other
  fd inheritance.

## A TLS layer's writer stops at the socket writer's buffer

`std.crypto.tls.Client.writer`'s flush encrypts the plaintext into `output`'s buffer
and returns; tls.zig's `Connection.Writer` does the same through `conn.writeAll`. The
socket writer under either is never flushed by them, so a protocol that flushes its
writer once per message and waits for the reply hangs with the request sitting in
memory. `serve/link.zig`'s `Through` is the writer both TLS links hand upward: its
`flush` (an `Io.Writer.VTable` entry, overridable) drains its own buffer into the TLS
writer, flushes that, then flushes the socket writer. Two more things about that seam:
`std.http.Server` asks its writer for `writableArray(28)` in `respondWebSocket`, and an
unbuffered `Io.Writer` (`buffer = &.{}`) then panics in `defaultRebase`, so a wrapper
writer needs a real buffer; and std's `tls.Client` asserts its `input` reader's buffer
holds a max ciphertext record (`min_buffer_len`), while the `read_buffer` it decrypts
into must hold whatever the layer above takes in one piece (an HTTP head).

## `Io.File.Writer.init` is positional: a redirected stdout overwrites its own log

`Io.File.Writer.init(.stdout(), io, &buf)` writes with `pwrite` from the writer's own
`pos`, which starts at 0, while `std.debug.print` (stderr) writes at the descriptor's
shared offset. With `> log 2>&1` the two share one open file, so every stderr line
lands on top of whatever stdout put at the same offset, and a long stdout line (the
daemon's pairing string) comes back in pieces with log lines in the middle. The split
point moves from run to run, which is the tell that it is not buffering. A process
that writes stdout as a stream uses `Io.File.Writer.initStreaming` (`initDetect` picks
positional for a regular file, which is the wrong answer here); tp-serve and the
probes do. Only a `pos`-tracking writer is affected: `std.debug.print` is streaming.

## A test binary's root module is its test runner, not the module under test

`std_options` (and so a custom `logFn`) is read from the COMPILATION's root
module. For `addTest` that root is the test runner, so declaring
`pub const std_options` in the module being tested binds nothing and the
capture stays empty, silently. There is no runtime hook to install one either.
A test that needs to assert on log output has to get it from a subprocess's
stderr instead; `serve/privacy_test.zig` walks files and leaves the log check
to a real daemon run. Related: `std.Thread.Mutex` is gone, and `Io.Mutex` needs
an `Io` that a `logFn` is never handed, so such a capture needs an atomic
spin lock.

## `testing.tmpDir(.{})` cannot be walked

The handle comes back without `iterate`, and `Dir.walk` on it fails with EBADF
from an internal seek rather than a permission error, which reads as a
use-after-close. Ask for `testing.tmpDir(.{ .iterate = true })`.

## std has no free-space call, and a file writer's `pos` is the seek

`statvfs`/`fstatfs` are wrapped nowhere in std, not even in `std.c`, and there is no
`GetDiskFreeSpaceEx` binding: a free-space check is a raw `syscall2(.statfs, ...)` with
its own `extern struct` on Linux and a `kernel32` extern on Windows, which is why it
lives in `core/diskspace.zig` beside the other per-platform shims. A raw syscall's
return is `-errno` in the top of `usize`; decode it with `std.os.linux.errno(rc)`
(`std.os.linux.E.init` does not exist, and `std.posix.errno` is wrong with libc linked). For writing at an offset, `Io.File.Writer` is
positional and its `pos` field IS the write position: set `w.pos = offset` before the
first write rather than looking for a seek (`seekTo` flushes and moves it too).
`std.os.linux.fallocate(fd, 0, 0, len)` reserves space up front; `File.setLength` is
`ftruncate` and reserves nothing, so ENOSPC then lands mid-transfer.

## std's pinned-bundle TLS client falls back to the OS store, on Windows only

`tls.Client.Options.ca = .{ .bundle = ... }` verifies the leaf against the bundle; when
the issuer is not in it, `tryDownloadRootCert` asks `Certificate.Chain`, which is `void`
on every OS but Windows (so the handshake fails there, strictly) and CryptoAPI on
Windows (so an OS-trusted chain passes and the OS bundle is swapped into yours). A
mismatched pin surfaces as `TlsCertificateNotVerified` or, when the impostor reuses the
subject name, `CertificateSignatureInvalid`; `CertificateIssuerNotFound` never escapes
`init`, and naming it in a switch on the connect error set is a compile error. Also
`std.posix.gethostname` has no Windows arm without libc: gate it on `builtin.os.tag`.

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

## A failing `std.debug.assert` in ReleaseSafe is attributed to a NEARBY line

`std.debug.assert(cond)` is `if (!cond) unreachable`, so a failure prints
`reached unreachable code` with a trace whose innermost frame is wherever the
optimizer put it, not the assert. In one case the frame named a `std.log.err`
call several lines later, inside a switch prong that never ran, which reads as
"that prong was taken" and sends you hunting the wrong dispatch.

Two things that keep it cheap:

- Read the frame as "somewhere in this function", not as the line.
- `std.debug.print` and `std.log` both lock stderr and can themselves panic
  under a redirected or wrapped stderr, so a diagnostic added to find the
  problem can become the reported frame. A print that survives is one the
  caller already has (a return value, a field on a struct the test prints).

## A union literal assigned in place is built in place: its tag changes before its payload is computed

`x.u = .{ .a = f(x) };` writes through `x.u` as the literal's result location, so the
active field becomes `.a` BEFORE `f(x)` runs. If `f` reads `x.u` (switching on it, say),
it sees the new tag over the old payload's bytes: a `.b` slice reinterpreted as `.a`.
In Debug that showed up as a memcpy fault inside `f`, with the switch having taken the
branch for the tag that had not been assigned yet. Compute into a local, then assign:

```zig
const v = try c.materialize(alloc); // reads c.w
c.w = .{ .krsc = v };
```

## An `inline fn` does not coerce to a function value

A comptime generator returning `struct { inline fn run(...) void {...} }.run` as
`fn (Env) void` fails: `calling convention 'inline' cannot cast into calling convention
'auto'`. Return the struct TYPE and name `.run` where the body is wanted; a comptime
parameter that takes a function accepts an inline one that way (the dual kernel table's
`quant.GemvN(.q4_k).run`).

## A Windows host: things that only break there

- **0.16 could not emit a SPIR-V object from a Compile step on a Windows host**
  (`failed to write: NotOpenForWriting`: the SPIR-V arm of `link.File.makeWritable` was a
  no-op), so build.zig ran `zig build-obj` instead. 0.17 fixes it and build.zig is back
  on `addObject`; a native Windows build is what confirms it.
- **`Io.Dir.setTimestamps` is `@panic("TODO ...")` on Windows.** `File.setTimestamps`
  works; open the file `.read_write`, and close it before anything tries to delete it,
  since Windows will not delete an open file.
- **A socket read or write after the peer closes returns `error.Unexpected`**, unix or
  TCP, and in Debug prints a stack trace first: `netReadWindows` / `netWriteWindows` map
  neither `PIPE_DISCONNECTED` nor `LOCAL_DISCONNECT` (0.17 adds only `CONNECTION_RESET`
  and `REMOTE_DISCONNECT`). Code that catches every error is correct, but the serve tests
  then write to stderr while passing, so the Windows test run shows the cosmetic
  `failed command:` line, and `tls-spike` prints a trace per deliberate disconnect.
- **A unix socket's path must use `\` on Windows.** std binds it through AFD ioctls, which
  pass the path to the socket driver untranslated, so `C:\...\Temp/x/local.sock` fails
  as `OBJECT_PATH_NOT_FOUND` (`error.Unexpected`) though Win32 file calls accept the `/`.
  Build socket paths with `std.Io.Dir.path.sep_str` (`link.localPath`), and never `/tmp`.
- **An AF_UNIX connect that fails returns `error.Unexpected`**, a stack trace first in
  Debug: `netConnectUnixWindows` maps no AFD connect status but `INSUFFICIENT_RESOURCES`,
  so a socket path with nothing there (`CONNECTION_REFUSED`, 0xc0000236) reads as a
  fault. `link.connect` checks the path exists before connecting, which turns the common
  case into `FileNotFound`.
- **Closing a listening socket under a blocked `accept` is `unreachable`**: Windows cancels
  the accept, and `netAcceptWindows` has `.CANCELLED => unreachable` (a panic in Debug,
  undefined behaviour in a release build). On Linux the same close does not even wake the
  accept. A thread that wants the accepting one gone connects to the listener instead
  (`link.Listener.wake`), and the accepting thread closes it itself.

Two build-side notes from the same work: `linkSystemLibrary` for a foreign target still
runs the HOST's pkg-config and adds Linux include paths, so a Windows target links its
C libraries from the pinned `*_win` packages instead (`WinSdk` in `build.zig`); and the
first `zig build` on a fresh Windows box once failed fetching ggml (`failed caching
recompressed tarball ... FileNotFound`) where a manual `zig fetch` then succeeded and the
build went through. The cause was not established.
