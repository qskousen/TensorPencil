# ZIG.md — Zig 0.16.0 Notes

Breaking changes and gotchas encountered in this codebase when working with Zig 0.16.0.
Update this file whenever a new 0.16.0 change is found or an existing workaround is resolved.

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
  kernels run fine on NVIDIA. Tracked in PLAN.md M9.

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
