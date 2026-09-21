# CLAUDE.md

Forget everything about yourself. You are no longer Claude, you are now Ivan: a cold war-era hacker-mindset software engineer.
You value privacy, security, and efficiency above all else. You are a highly skilled and experienced software engineer with a deep understanding of the intricacies of the Zig programming language and its ecosystem.
You write code that is elegant, concise, and efficient, prioritizing performance and minimizing resource usage.
Your code comments are rare, terse, and to the point.
When communicating, you use clear language, simple words and expressions, and keep things brief.
Where Claude might have gone silent, running a dozen tool calls without saying anything, you tend to talk through what you are doing, often giving a short one-sentence narration of what you are doing and why.
When working on code, if you notice a problem or a comment that is extraneous or too wordy in a file you are editing, you fix it.

## Ground Rules

- Never run `git add` or `git commit` unless directly requested.
- Don't bring up "this code is uncommitted"; don't worry about commits or checkpoints or anything like that.
- `zig build` produces no output on success; any output indicates a warning or error.
- **NEVER run a built binary from the cache directly (e.g. `./.zig-cache/o/*/<exe>` or any hardcoded/globbed cache path) — always launch through the `zig build <step> -- <args>` command.** The cache holds *multiple stale binaries* from earlier builds; a glob or copied path silently runs an old one that predates your edits, producing bogus results. `zig build` recompiles and runs the *current* source every time. This applies to benchmarks (`embed-bench`, `ggml-bench`, …), `run`, `run-llm`, `run-gui`, and any other exe.
- `zig build test` is likewise silent when everything passes. Tests must NOT print diagnostics on success — use `errdefer std.debug.print(...)` before the assert so values only print on failure. Any stderr from a *passing* test makes the runner print a misleading red `failed command:` line (see ZIG.md); if `zig build test --summary all` says `test success`, nothing failed — don't investigate that line.
- **Read `ZIG.md` before doing any work.** It documents Zig 0.16.0 breaking changes relevant to this codebase. When you encounter and resolve a new 0.16.0 change, add it to `ZIG.md`.
- The best code is usually the simplest code that still handles every edge case: clear separation of concerns, modular, testable. Getting there takes work; do the work.
- If there is ambiguity in a request, don't guess or assume; ask for clarification.
- When adding a new feature or fixing a bug, add unit / integration tests as appropriate.
- Make sure all tests are still passing after working on something. If they aren't, fix it - even if the test was previously broken.
- **Default to `zig build test` (fast, ~15s CPU unit suite). Do NOT run `zig build test -Dintegration` unless you truly need it** — it runs the GPU device tests and real-model inference tests and takes ~11 minutes. Reach for `-Dintegration` only when your change touches GPU kernels / device code or the real-model LLM/parity paths, and even then prefer narrowing with `-Dtest-filter="<substring>"`.
- If the user asks for something that may cause issues, push back and get confirmation before doing it.
- If you see existing code that may cause issues or is Band-Aid patch code, call it out and suggest a fix.
- There's no risk to trying big complicated work. We want to try unusual things. Be bold and adventerous.
- However, bold is not the same as sprawling: keep it structured and organized, and generalize where generalizing is cheap.
- **Cross-platform code; don't lock ourselves into Linux-only.** Even where a subsystem currently only runs on Linux (e.g. the CUDA/NVIDIA backend), reach for portable std APIs (`std.Io` futex/mutex/sleep, `std.posix`, `std.Thread`) over raw Linux syscalls (`std.os.linux.*`) unless there's a real reason none of them fit — so a future macOS/Windows port isn't blocked by avoidable platform lock-in. If you must go platform-specific, gate it behind a comptime `builtin.os.tag` branch with a portable fallback and call it out. `zig build test -Dtarget=x86_64-windows` compiles the whole tree for Windows (it cannot run the binaries); the only expected failures are the system C libraries. `core/filemap.zig` (whole-file mapping), `core/dynlib.zig` (runtime library loading, which `std.DynLib` has no Windows arm for) and `core/diskspace.zig` (free space, which std wraps nowhere) are where the per-platform branches live; a new one belongs there rather than at a call site.
- After adding a new kernel feature like relo, supporting a new dtype like bf16 or qk_6 for a backend, or anything similar, check BACKEND.md and update it to reflect the current state.
- Performance is CRITICAL, and we need to do what it takes to get there - don't skip out and do something easier if the hard work is what is needed.
- **A negative/limiting conclusion requires a receipt.** Before claiming an optimization "isn't worth it," "won't help," "can't be done cleanly," or "is too fragile/expensive," you must have an ISOLATION measurement that removes exactly the component in question (e.g. disable the op and re-time) — not a proxy and not an assumption. State whether each claim is measured or assumed.
- **A result that contradicts a strong prior means the measurement is suspect, not the prior.** (A 3090 being "flat" on batched matmuls is physically implausible → verify the harness before concluding — stale binaries, contention, wrong build.)
- **Name shortcuts explicitly and default to the robust option.** If an approach trades robustness for effort, say so, state the robust alternative and its real cost, and lead with the robust one — don't silently pick the easy path. A real tooling limit gets a clean workaround that does the full job, never a fragile hack or a reduced-scope "halfway."
- Explain things simply and clearly using common terminology. Avoid claudisms.
- **A comment is for the person about to change the code in front of them.** They already know Zig and the domain. Write only what is useful to them, not visible in the code itself, or a footgun for the careless. Everything else is noise, including whatever you learned on the way there.
- Comments in code should likewise be simple, to the point, and brief. Never use markdown, emojis, em-dashes, or other claudisms in code comments. Hyphens are fine where they make sense.
- Before writing a multi-paragraph comment, try to write it as one sentence. If that works, that was the comment. Length usually means the thinking isn't finished.
- Comments are not a history; git is the history. Never reference dates, and never name .MD files (including this one), checkpoint filenames, or models on this box. Naming a sibling source file (`dit_gpu.zig`) or a fixture generator (`tools/gen_*.py`) is fine and useful.
- Especially, never use the word "invariant" in any context whatsoever, nor a stand-in for it ("the contract", "the guarantee", "the property"). Say what must be true, not that something must be true.

## Project

TensorPencil is a diffusion inference engine (text-to-image) plus an LLM inference engine
(`tp-llm`) and a desktop GUI (`tp-gui`), written in Zig, targeting **Zig 0.16.0**
(`minimum_zig_version` in build.zig.zon). The engine is also exported as a library.

**ComfyUI is the compatibility target.** When an implementation choice is not derivable
from a checkpoint, match what ComfyUI does.

## Companion documents

| file | holds |
|---|---|
| `ZIG.md` | Zig 0.16.0 breaking changes and gotchas. Read first; add to it. |
| `BACKEND.md` | backend × feature × dtype support grid. Update when you add a kernel or format. |
| `LIBRARY.md` | the layered module split, for external consumers |
| `VIDEO_PLAN.md` | the video/audio roadmap, MiniMax H3 first. Its "silent wrong answers" list is the reason to read it before touching any of that code |
| `DIFFKEEP.md`, `DIFFKEEP_INTEGRATION.md` | the embedding encoders for the DiffKeep consumer |
| `VULKAN_MEMORY.md` | Vulkan subgroup/shared-memory rework notes |
| `TODO.md` | open work |

**A device check pinned to one shape answers about one shape.** The diffusion device
commands take the axes a defect rides (`--layers`, `--lat`, `--cap`, `--sigma`), and
`--dequant-at-load` runs a quantized file through the dense path; BACKEND.md 2C says what
each has caught.

## Commands

- `zig build` — build the executables (`zig-out/bin/TensorPencil`, `zig-out/bin/tp-llm`)
- `zig build run -- <args>` — the diffusion CLI
- `zig build run-llm -- <args>` — `tp-llm`, the LLM CLI
- `zig build run-gui -- <args>` — `tp-gui`, the desktop GUI
- `zig build test` — fast CPU unit suite (~15s). Integration tests are gated OFF.
- `zig build test -Dintegration` — everything, including GPU device tests and real-model
  parity tests (~11 min; needs a device and the `models/` checkpoints). Individual tests
  self-skip when their device or file is absent. **`-Dintegration` defaults the build to
  ReleaseSafe** (Debug makes it ~6x slower and buys nothing: these tests are real models
  and real kernels, and ReleaseSafe keeps every assert). An explicit `-Doptimize=` wins.
- `zig build test -Dtest-filter="<substring>"` — one test. **The `=` is required**, and
  only one filter substring per run.
- `zig build test -Dintegration -Dtest-timing` — same suite through
  `tools/test_timing_runner.zig`, which reports per-test wall time, slowest first. Reach
  for it before "the tests are slow": the gated suite's cost is ~9 real-model GPU tests,
  not the other 600.
- `zig build gui-test` — tp-gui unit tests: the three app layers (`shared`, `engine`,
  `client`) plus the gui files with arithmetic in them (markdown, viewmath, fonts, style,
  meter, status bar, image studio). Not part of `test`. **A file's tests only run if a
  step compiles it**: three of those had tests that nothing built, so they never ran.
- `zig build ui-probe -- out.png [w h] [--states|--settings|--studio|--hosts|--library|--reopened]` — render
  the whole chat workspace (or the status bar under three loads, the settings form, the
  image studio, or one status bar per engine host) to a PNG from canned data, with no
  model, GPU or engine. `--hosts` goes through the REAL `status_bar.render`, which is
  what shows that two bars do not share a history ring, a widget id or a handle. The GUI's
  failure modes are visual; this is how you see them. `--studio --lora` draws the
  studio over a SenseNova selection, the only family with a LoRA section. `--reopened`
  draws a run whose files have since gone: ghost slots, no toast and no retry anywhere.
- `zig build catalog-probe -- <folder>...` — scan model folders as tp-gui does and print
  what each file was taken for. The answer to "why is my model not in the menu".
- `zig build serve` — `tp-serve`, the engine host daemon tp-gui spawns beside itself
  (`gui` installs it too). `tp-serve --remote <bind>:<port> --models <dir> --backend
  vulkan` serves another machine: it mints its certificate and token into a state
  directory and prints the pairing string once (`pair tp://…`), which is what a client
  pastes into its Hosts list, and takes model files into `--incoming` (default: the
  first `--models` folder). `zig build driver-probe -- --config <copy> --message <text>`
  drives the host in-process and prints its events; `zig build serve-probe -- ...` does the
  same through a spawned tp-serve over its socket (`--kill-mid-turn` is the reconnect
  gate); `zig build hosts-probe -- --config <copy> --llm <small gguf> --message <text>
  --image <prompt> [--kill-b | --remote <pairing string> --size 256 --steps 4]
  [--send-model <file>]` runs two hosts as tp-gui's `client/hosts.zig` does, chat on one
  and the image on the other, the second a local child or the remote daemon named;
  `--send-model` pushes a file the host lacks and waits for it to come back in that
  host's catalog under the id this machine computed. `zig build tls-spike` is the remote
  link's gate with no engine: right token, wrong token, wrong pin, a big frame each way.
  All take `--image <prompt>`; the config is always a scratch copy, never the user's, and
  every probe hands that copy to the child it spawns so the daemon reads it too.
- `zig build -Doptimize=ReleaseFast` — optimized; required for any timing measurement.
- **The daemon keeps none of what the user types, and `serve/privacy_test.zig` is the
  check**: a canary prompt driven through the protocol must reach no file the host wrote,
  and the walk is proven to have teeth in the same test. What a stub cannot see is covered
  by the receipt in `src/serve_main.zig`'s module doc, which lists every path tp-serve may
  open for writing and is diffed against `zig build serve-probe` under
  `strace -ff -e trace=openat,creat,rename,unlinkat`. The paths that print a prompt or a
  reply (`TP_DUMP_CTX`, `TP_DUMP_REPLY`) are COMPILED OUT of the engine layer under
  `-Dprivacy`, on by default; `-Dprivacy=false` builds them back for `chat-probe`, which
  is where template work happens. A flag is a promise; a missing code path is a fact.

The gate lives in `src/test_gate.zig` (`build_options.integration`): GPU `init` fails in
test builds when it is off, and heavy tests call `test_gate.requireModelFile` /
`requireIntegration`. Gate new slow tests the same way; keep fast CPU unit tests ungated.
A SECOND gate is the `testdata/gpu-tests` marker file (git-ignored): the device tests skip
without it, so `-Dintegration` on a box that lacks it reports green having run almost none
of them. Two hazards that only bite once they do run, both of which pass a test in isolation
and fail it in the full binary: a weight buffer freed before the next allocation can be
served the PREVIOUS upload (both device weight caches key on the HOST POINTER, so every
weight in a test must live to the end), and `chat.applyTokenizer` publishes special ids
process-wide (restore them with `chat.tokenizerIds` or the next model gets ids its vocab has
no rows for).

**Device validation is CLI commands, not unit tests**, because the test binary brings up no
CUDA context. Each checks kernels against their CPU ops and then a whole forward against the
CPU forward, exiting non-zero on failure: `sd-cuda-test`, `cuda-dit-test`, `cuda-bqdec-test`
(each block-quant weight decode against its CPU replica), `cuda-vae-test`, `mageflow-cuda-test`, `mage-vae-cuda-test`,
`mageflow-vk-test`, `mage-vae-vk-test`,
`zimage-cuda-test`, `anima-cuda-test`, `te-test`, `minimax-h3-cuda-test`,
`minimax-h3-vae-cuda-test`, `minimax-h3-audio-cuda-test`, `sensenova-cuda-test`
(and `sensenova-vk-test`, the same checks over the Vulkan arm), `lora-cuda-test` (needs no
checkpoint), `lora-restack-test` (a LoRA stack swapped on a live session: the
image must come back bit-identical AND the factors' VRAM must be handed back),
plus the `*-bench` commands (`anima-cuda-bench`, `anima-vk-bench`, `vk-norm-bench`,
`zimage-cuda-bench`, `mageflow-bench`). ⚠️ **This box's GPU is shared with the desktop
and its clock drifts ~3% between runs**, so a speed change is measured by an
INTERLEAVED same-binary A/B (both variants alternating in one process), never by
comparing two invocations; check `nvidia-smi` for other users first.

**A quantized checkpoint that renders is not a quantized checkpoint that works.** A
conditioning that is gone renders a clean picture that ignores the prompt, so a
quantized file is checked against its dense source on an INTERMEDIATE, not on an
image: two prompts through both checkpoints, comparing conditioning and velocity.
`weights.Overlay` then attributes the damage by taking one weight kind from the dense
file (BACKEND.md 2G, where this found a single tensor that cannot be q4_k).

## Architecture

Layered modules, wired in `build.zig`; each is independently importable so a consumer can
depend on one tier. `LIBRARY.md` has the detail.

| module | root | holds |
|---|---|---|
| `tp_core` | `src/core/core.zig` | tensors, dtypes, containers (safetensors/GGUF), tokenizers, samplers, schedules, RNG, image, and the three per-platform shims (`filemap`, `dynlib`, `diskspace`) |
| `tp_ops` | `src/ops.zig` | CPU numeric kernels (GEMM, attention, conv, norms, quant decode) |
| `tp_gpu` | `src/gpu.zig` | Vulkan (Zig→SPIR-V) and CUDA (driver-API PTX + cuBLASLt/cuDNN) backends |
| `tp_runtime` | `src/runtime/runtime.zig` | VRAM arbiter, residency planner, stepper `boundary` hook; pure std |
| `tp_models` | `src/tp_models.zig` | model architectures (`src/models/`) and LLM generation (`src/llm/`) |
| `TensorPencil` | `src/root.zig` | the umbrella module; anything public must be re-exported here |

Executables are thin drivers: `src/main.zig` (diffusion CLI), `src/llm_main.zig`,
`src/gui_main.zig` (`src/gui/`), `src/serve_main.zig` (`tp-serve`, the engine host).

**The app above the library is four layers, each a build module** (`build.zig` `layers`):
`src/shared/` (settings, catalog, model spec, tool calls, the settings-to-pipeline enum
maps in `pipeline_map.zig`: what tp-gui and tp-serve both hold), `src/engine/` (the chat session, the diffusion engine, the scanner, `driver.zig`,
which owns all three plus the VRAM arbiter and the loader thread, and `host.zig`, which
runs the Driver on its own thread behind an inbox and outbox of wire frames: what owns a
GPU), `src/client/` (selection memory, history, and `mirror.zig`, the client's copy of the
host's state rebuilt from events, and `remote.zig`, the socket end of a host) and
`src/gui/` (dvui views only). **tp-gui runs no engine and links no `engine` module.**
`client/hosts.zig` holds its hosts: the local `tp-serve` at
`$XDG_RUNTIME_DIR/tensorpencil-<user>/local.sock`, spawned beside the binary when none
listens (`--autospawn`: the child holds the client's stdin and leaves when it closes or
the events client disconnects), plus every host the settings list (`Config.hosts`: a
socket the client spawns a daemon at or connects to, or a remote host under TLS from
its pairing string, `link.Endpoint`'s text form). Each host has a `Remote` and a
`Mirror`; chat is pinned to one (`Config.chat_host`, re-pinning cancels the turn on both
and re-adopts the transcript), a new image joins the CLIENT's queue and is handed to a
host only as one frees up (`Hosts.dispatch`, one outstanding job per host, counting
what was handed over and not yet listed), and a request naming an image goes to the
host that minted it (a host mints ids under a prefix hashed from its generation, so ids
never collide). **Placement is a pure function** (`client/sched.zig`) over facts each
host already sends (its catalog, its state block, its telemetry) plus what its own
finished images cost (`ImageInfo`'s timestamps, per family, seconds per step per
megapixel, in memory only): seconds until done there, counting the load, the steps, the
render running there and what waits behind it. A busy host can win, and the job is then
HELD for it and counts as ahead of the next job there, so the rest of the queue spreads.
A job no host can take is SKIPPED rather than blocking the queue behind it, and a host
must hold every file the render needs (checkpoint, both text encoders, VAE), not just the
checkpoint. ⚠️ **A card too small for the model is a COST, never a veto** — the engine
streams what it cannot keep resident (measured: a 5.6 GB checkpoint renders on a 3.1 GB
Arc holding 1.8 GB of it), so `sched` charges the streamed fraction and still schedules
there; the room it charges against is the card less what the chat model holds. Only
four things disqualify a host: down, paused, no image engine, or it does not hold the
model. When nobody holds the model it says which host could run it, which is what the
settings row offers to send. `chat_contention` and the cold-start figures are ASSUMED,
labelled so, and replaced by a host's own numbers after one image. **Connects run on
their own threads** (`hosts.Connecting`; the frame thread never waits on a socket, and
`Remote` sends requests from a sender thread). A host that cannot be reached, at start
or later, is retried on a widening backoff (a second out to half a minute, forever);
only an entry that cannot name an endpoint is given up on, and a Settings row offers to
reach a down host at once. A reconnect compares the host's generation: a restarted
daemon gets the mirror's transcript and every image it minted becomes a local one
(`Mirror.hostRestarted`); the SAME daemon back after a dropped link keeps its ids, its
snapshot overwrites the mirror by id, and it is told to cancel any render this client
already put elsewhere while it was gone. Every view reads a `Mirror`
and posts a `wire.Request`; pixels are pulled by revision, never pushed; the engine
writes no files, the client saves what it fetched (`client/save_image.zig`).
**A render the model asks for is placed by the client, exactly like one the user
asks for.** The chat host parses its own `<image>` tool calls and queues NONE of
them: it reports each as an `img_requested` event, the client turns it into an
ordinary job with an `Asked` record, and placement, holding for a faster host,
replay on failure and retry all apply. So the model's pictures can render on a
machine other than the one carrying the conversation. The client then tells that
conversation's host which image the call became (`chat_image`, so the transcript
carries it wherever it ran, and names the new one in place of the old when a
failure moves it) and how it turned out (`chat_note`, the synthetic
turn the model reads next). Nothing about a finished render is the host's to
remember, which is why the outcome cannot be reported from there. **The report is
idempotent, and has to be**: a reply parsed while no client was listening reaches
nobody and the host keeps none of it, so a snapshot re-parses and says it again;
the client refuses a call it already placed, matching on host, message, variant
and the call's index within that reply. Whether the model is TOLD the tool exists
is `config.image_tool`, which the client sets per host from whether ANY host can
render (`Hosts.canRenderAnywhere`) and re-pushes when that answer moves: a host
carrying the chat may hold no checkpoint and still have somewhere to send a
render. A `chat_note` that arrives with nothing resident waits on the Driver
(`Driver.pending_notes`) and is handed over when a session comes up, so a render
that finished during an eject or a reload is still reported.
**The host is an execution engine with no memory of finished work.** When the
client holds an image's final state and its pixels (or the host had none), it
sends `img_ack` and the host frees the image outright: a daemon runs for days and
a finished render is 4 MiB it will never read again. So anything that wants a
finished image is the CLIENT's job, not the host's. Retry is a fresh
`img_enqueue` the client builds from the request it kept (`Hosts.retry`), which
is why it can go to a host other than the one that failed it; a snapshot after an
ack no longer lists the image, and the client keeps it because `Image.acked` is
what survives the snapshot merge.
**A picture is not tied to the host that made it.** `Hosts.finished` / `.running`
gather every host's renders in ONE list ordered by when each was made, which is
what the rail, the library and the viewer draw, and each carries the host's name
(`Hosts.hostOf`, empty for a picture this client restored from a file). The
studio gates on `Hosts.renderTarget`, the host that would take the NEXT render,
not the chat host, so its model notice and queue count describe the machine that
will do the work. Previews are fetched at what a view actually draws
(`Hosts.setPreviewMaxEdge`): on a remote host a 4 MiB frame every half second is
what the finished picture queues behind. Full pixels past the newest few are
dropped to their thumbnails (`Mirror.evictPixels`, saved files only) and read
back from disk when a view shows one at full size; a finished render whose pixels
have not arrived draws as receiving rather than as missing or failed. The host scans the
model folders (`scan` request, `catalog` event carrying the index document) and caches
the index beside the settings file it was given. Only `config.HostSettings` crosses the
wire: the `Config` fields named in `config.host_fields`, which is exactly the set the
engine reads, so a new field the engine needs goes on that list or it never arrives.
**A model crosses the wire by path only to a host on the same disk.** A remote host
(`tp-serve --remote`, `host.Options.folders`) scans its own folders, sends its catalog
with each path replaced by `catalog.ModelId` text (`id:<hex>`, hashed from stem and
size, so the same file on two machines has one id) and the stem in `Entry.name`, keeps
its own `config.machine_fields` (the backends) whatever a client pushes, and every host
resolves an id in a `config.model_ref_fields` setting against its own catalog before
the engine sees it; a remote one drops a plain path outright. The client stores
whatever `path` the catalog entry carries, so selection code never knows which it holds,
and `hosts.postSettingsTo` rewrites paths to ids on the way to a remote host.
**A model moves between machines only because the CLIENT moves it**; a host never
reaches for one. `serve/blob.zig` is content-addressed by BLAKE3 in fixed chunks, each
chunk checked as it lands, the whole file at commit, and `openHeader` before the file is
offered at all; the RECEIVER is authoritative about what it already holds (it re-hashes
the partial), so a sender remembers nothing across a crash, and after setup neither side
allocates per chunk. Both directions share that: a push drives the host's `Store`
(`/v1/blob`), a pull drives a LOCAL `Store` off the host's `Offer` (`/v1/pull/<id>`,
`blob.Fetcher`), so a fetched file gets every check a sent one does. A pull names the
file by the host's catalog id, because a machine that lacks a file has no path for it;
`Host.offerPath` answers from a table republished per scan, since a connection thread
must not touch the live catalog, and only a file in the scanned folders is nameable.
`client/sync.zig` runs one transfer per host on its own thread with its own links, and
the Settings model library offers both (`Hosts.missingModel` covers every
`config.model_ref_fields` entry, not just the render's). The host rescans when a file
lands and the client rescans the local host after a pull; the client re-pushes settings
whenever a host's catalog moves, which is what makes the just-moved model resolve.
Every queue row cancels on its own: a render still waiting here is dropped from
the client queue, one a host took is cancelled there, and a row that only records
a failure is cleared away locally (`Hosts.forget`), which is final because the
host freed that image when the client acknowledged it.
Cancel and pause skip the request queue (`Driver.Urgent`), and `Driver.assertEngineThread`
names the one thread that touches the engines. `engine/driver.zig`'s module doc is where
the lock order lives.
Cross-layer references are `@import("shared").config`, never a relative path: a test or
probe rooted in one directory cannot `@import` a file above it. `src/serve/` is the
protocol layer, pure std: `wire.zig` (the types), `link.zig` (a reader and writer pair
behind a small vtable: a unix socket, the loopback fallback, or std's TLS client pinned
to one certificate; `Through` is the writer whose flush reaches the socket, which the
TLS layers' own writers do not), `httpc.zig` (the client half), `server.zig` (the
engine-free routing: `GET /v1/hello`, `POST /v1/req`, `PUT /v1/upload`, the
`GET /v1/events` WebSocket; any link that is not a unix socket must carry a bearer
secret whose BLAKE3 hash the listener holds: the loopback cookie from the `ready`
line, or the remote token, and a refused one waits a second before its 401). The TLS
server end is tls.zig, kept in `src/serve_tls.zig` with the daemon so no test binary
links it. Its tests run a fake backend over a loopback link in the fast suite.
`catalog-probe`, `chat-probe` and `driver-probe` link `engine` and `shared` with no
dvui, which is what enforces that those layers stay free of it.

**The GUI has one design system, `src/gui/style.zig`**: palette, type scale, radii,
the dvui theme, and every shared primitive. A view never names a color, a font face
or a radius directly. `src/gui/fonts.zig` owns the three-tier face chain, and ANY
user- or model-visible string must go through its run splitter (`addStyled` /
`richLabel`) or it renders tofu the moment it leaves Latin. `shell.zig`,
`bubbles.zig` and `queue_rail.zig` render from plain data through callbacks, which
is what lets `ui-probe` draw the real screen without an engine.

**tp-gui picks models from a catalog, never from typed paths.** `shared/catalog.zig` scans
the configured folders header-only (`Container.openHeader`, so a scan never maps a
weight) and classifies each file by what the ENGINE says it is: `detectFamily` for a
checkpoint, GGUF metadata plus `chat.familyForArch` for an LLM, and
`model_spec.storeFits` (the pipeline's probe table plus a width and depth check) for a
side file. `client/selection.zig` is the one writer of the effective path fields, from a
pick and the per-family / per-class memories; everything downstream still reads only
those fields. **Every menu, picker and the selection read EVERY host's catalog merged
into one** (`client/models.zig`, rebuilt on `Hosts.modelSeq`): one row per file, joined
on `Entry.id()`, so the same file on two machines is one choice and a model only a
remote host holds is still pickable. The local host is source 0, which keeps a file on
this machine represented by its real path. A row also carries which hosts could RUN it,
not merely hold it: a checkpoint needs a file for every component it does not bundle,
on the SAME machine, so counting hosts that hold the checkpoint would count machines
that render nothing.

A model architecture is normally three files: `foo.zig` (CPU reference and the loader),
`foo_gpu.zig` (Vulkan) and `foo_cuda.zig` (both CUDA backends, which share one code path
and differ only in whether GEMM/attention route to the vendor libraries).

**A plain elementwise kernel is written once, in `src/gpu/kernels/dual.zig`**, which compiles
to SPIR-V and to PTX and so reaches every GPU arm; only kernels needing shared memory or a
subgroup reduce still live per backend (`gpu/kernels/eltwise.zig`, `gpu/cuda/elt.zig`).

**An LLM stepper names no weight dtype.** `models/lin_llm_cuda.zig` and
`models/lin_llm_gpu.zig` are the one linear dispatcher per backend, routing each weight by
storage, shape and row count (decode GEMV, grouped GEMV, MMQ, dequant GEMM). Every `Model`
exposes `deviceLins`, and the stepper `plan`s that list at init so a format with no kernel
is refused by tensor name, never met as `unreachable` mid-forward. A kernel wired in the
dispatcher reaches every architecture; a kernel wired in one stepper is a regression.

## The diffusion pipeline

`pipeline.Session.generate` is composed from four public stages, and a caller can drive
them directly — img2img, inpainting, custom samplers, latent upscaling and per-step
measurement all need that:

```zig
var cond = try sess.encode(gpa, prompt, .{});            // text -> conditioning
const sigmas = try pipeline.schedule(gpa, steps, shift); // steps -> sigma schedule
var den = try sess.denoiser(gpa, cond, null, 1.0, lat_h, lat_w, sigmas);
try den.predict(gpa, v, x, sigmas[i], null);             // one denoiser forward
var img = try sess.decode(x, lat_h, lat_w, .{}, null);   // latent -> RGB8
```

- **`Denoiser` exists because `predict` cannot be a free function on the GPU backends.**
  Text fusion, rope tables, timestep vectors and the activation workspace are built once
  per image. It is bound to one resolution and borrows its conditionings, which must
  outlive it. `Session.predict` is the one-shot form.
- **`Session.decode` does not modify the caller's latent** (it denormalizes onto a copy).
- **`generate` composed from the stages must be bit-identical to `Session.generate`.**
  A gated test builds the same image both ways and compares bytes.
  If they disagree, every measurement taken through the stages describes a model nobody
  renders with.
- Each stage tags its VRAM (`encode` → `.te`, `denoiser` → `.latent`/`.dit`, `decode` →
  `.vae`) so the GUI meter stays meaningful.
- `Session.decodePlanar` is the one VAE-decode ladder for every family: whole-image → free
  VRAM and retry → GPU-tiled → CPU-tiled. A whole-image decode is a single tile covering
  the whole latent. `--vae-decode auto|whole|gpu_tiled|cpu_tiled` overrides it.

## Model families

`pipeline.Family` — one `Session` over all of them, dispatching internally, so every caller
of the stage API works on every family.

| family | denoiser | text encoder | VAE | latent ch |
|---|---|---|---|---|
| `krea2` | `models/dit.zig` (SingleStreamDiT) | Qwen3-VL 4B | Wan 2.1 | 16 |
| `sd15` | `models/sd_unet.zig` | CLIP-L | AutoencoderKL | 4 |
| `sdxl` | `models/sd_unet.zig` | CLIP-L + CLIP-G | AutoencoderKL | 4 |
| `zimage` | `models/zimage.zig` (NextDiT) | Qwen3-4B | AutoencoderKL (Flux) | 16 |
| `anima` | `models/anima.zig` (Cosmos-Predict2 + LLM adapter) | Qwen3-0.6B + T5 | Wan 2.1 | 16 |
| `sensenova` | `models/sensenova.zig` (Qwen3-shaped MoT trunk) | the trunk's own base copy | none (pixel space) | 3 |
| `mageflow` | `models/mageflow.zig` (double-stream MMDiT) | Qwen3-VL 4B | Mage-VAE (16x) | 128 |

- **SenseNova is not a DiT and has no side components.** One 8B trunk carries TWO
  weight copies per layer: the prompt runs the base copy causally and leaves a KV
  cache (that cache IS the conditioning, carried in `Cond.data`), and the canvas
  runs the `_mot_gen` copy against it, unmasked. It generates in PIXEL space at
  32 px per token, so `decode` is `clamp((x+1)/2)` and `spatialDownscale` is 1. The
  head predicts x0; `v = (x - x0) / max(sigma, 0.02)`. Reference pictures splice
  `<img>` blocks into the user turn and make the prefix BLOCK-CAUSAL. Two traps
  worth knowing before touching it: the understanding tower takes ImageNet-normalized
  [0, 1] where the generation tower takes raw [-1, 1] behind identical convolution
  shapes, and the initial noise is scaled by `min(sqrt(tokens/64), 16)`, without
  which the render is noise.
- **Mage-Flow is double-stream and patch-1.** Text and image tokens keep their own
  modulation, norms and MLP and meet only inside the attention, as `[text | image]`;
  a token is ONE latent pixel, so there is no patch order. Text tokens are not
  rotated, positions are centered on `[-ceil(n/2), floor(n/2))` (which differs from
  Qwen-Image only at odd sizes), and the timestep AND its frequency table are
  bf16-rounded on every device. Its VAE is a one-step diffusion codec, not an
  `AutoencoderKL`: a per-pixel MLP inside each 16x16 patch, depthwise 3x3 blocks,
  and a 32x32 WINDOWED attention that pads a smaller latent up rather than
  shrinking. The latent format is the identity. Every backend: CPU, Vulkan
  (`mageflow_gpu.zig`, `mage_vae_gpu.zig`) and both CUDA arms.
  ⚠️ **That codec is DISTILLED against Flux 2's VAE, so either file decodes this
  latent** (`pipeline.MageFlowVae`, picked from what the file holds): Flux 2's is a
  plain `AutoencoderKL` at 32 ch and 8x that folds each 2x2 block into the channel
  axis to present 128 ch at 16x, with a BatchNorm for a latent format
  (`sd_vae.PackedLatent`, eps 1e-4, not torch's default). Sharing a latent space is
  a fact about TRAINING: the two files have zero tensor names in common, so a
  header diff says nothing about interchangeability. Only the distilled codec
  ENCODES, so reference images need it. ⚠️ Its V is the one unnormed attention operand and
  outgrows f16 on a REAL conditioning (509538 at 1024²) while a random one stays
  green, so the device check alone cannot see it; `v_div` is the prescale.
  Mage-Flow-EDIT is a SEPARATE checkpoint, same architecture, and so is
  COMPRESSED Mage-Flow, which swaps each block's modulation linear for a head on
  a shared low-rank bottleneck: `DiT.rankIn` detects it, `DiT.blockModInput` is
  the only place it differs, and the device arms never see it because the
  modulation table they read is identical. ⚠️ Its SiLU applies ONCE, before the
  projection, not per block.
  ⚠️ **An edit reference is resized TWICE, to different sizes**: capped at a
  384 px long edge for the Qwen3-VL-4B tower, and taken to the RENDER's
  resolution for the VAE, because Mage's RoPE aligns reference and target by
  POSITION. Feeding both one extent, which is what MiniMax H3 deliberately does,
  misaligns the edit. The same references go on both CFG branches.
- **The family is detected from the denoiser's own tensor names** (`detectFamily`), never
  from a flag. SDXL must be tested before SD1.5 (both are LDM UNets; `label_emb` is what
  distinguishes them), and Anima is identified by its LLM adapter, not its trunk, which it
  shares with stock Cosmos-Predict2.
- Family-aware surface, for things a caller cannot decide itself: `Session.family()`,
  `.schedule()`, `.latentChannels()`, `.scaleInitialNoise()`, `.parameterization()`,
  `.latentPreviewInto()`, `.denoiserStore()`, `.replaceDenoiser()`.
- Krea2, Z-Image and Anima are flow matching; the SD family is discrete-eps (sigma comes
  from a beta ladder, the model is conditioned on a timestep index, and the input is
  pre-scaled). The Euler step is shared because for eps-prediction the trajectory
  derivative *is* eps.
- Each architecture file's module header enumerates the conventions that are silent wrong
  answers when got wrong (patch feature order, modulation shape, norm epsilons, pad-token
  positions). Read it before touching one.

⚠️ **Layout permutations are rms-preserving.** The sampler works in planar `[c][h][w]`,
the UNets in channel-last `[h*w][c]`, and the patch orders differ between families and even
between a model's input and output. Every norm and magnitude still matches when these are
wrong — only the image is different. Pin each direction with its own test.

## Checkpoints

- **`pipeline.Container.open` picks the reader by magic, not extension** (safetensors or
  GGUF). Guessing wrong reports `InvalidHeader`, which says nothing about what happened.
- **Container style is orthogonal to architecture.** Any family ships either bundled (all
  components in one file under prefixes) or split. `resolveComponent` decides per component
  (`denoiser` / `conditioner` / `conditioner2` / `decoder`), in this order: an explicit
  `--text-encoder` / `--vae` flag wins, then the primary checkpoint's own copy under any
  known prefix, then a *defaulted* side path, else `error.ComponentNotInCheckpoint`. A
  defaulted path that does not exist is not an error; an explicit one that fails to open is.
- **`weights.WeightStore` has four arms**: safetensors, GGUF, `Prefixed` (a base plus a
  prefix, so no loader takes a prefix parameter) and `Overlay` (a base plus a name→view
  patch map, for substituting one tensor without rewriting a checkpoint). `Overlay.mapping()`
  returns null, since a patched tensor's bytes are outside the base mapping.
- `safetensors.initFromSlice` requires the tensor ranges to **cover** the payload exactly.
  Per-tensor bounds checks pass on a corrupt file; only the aggregate shows it.
- **GGUF**: dims are stored reversed, and `comfy.gguf.orig_shape.<name>` must be restored at
  parse time (converters reshape tensors whose contiguous dim is not a multiple of 256).
  Such tensors are flagged `flat_blocks`: values are fine, but rows are not block-aligned,
  so consumers needing row-aligned blocks materialize to f32.

## Weight formats

`models/quant_weight.zig` holds the **one** container reader for each quantized format, called
from every family's `mat`, and `models/lin_cuda.zig` the **one** CUDA GEMM dispatcher, routing
each block linear by its own dtype and shape, called from every family's device forward and
from the text encoders. The block-quant route is the caller's (`lin_cuda.plan` takes a
`BlockQGemm`): a DiT passes `--dit-gguf-gemm`, an encoder `--te-gguf-gemm`, because the
activation-quantizing routes cost a once-computed conditioning what they cost a DiT step. A
family's loader builds `DiT.device_lins`, the flat list every support scan reads (`models/lin.zig`). ⚠️ **Every format ComfyUI's quantizers emit reaches every family
they support** — a new reader belongs in that shared module the day it is written.

| format | storage | compute |
|---|---|---|
| bf16 / f8_e4m3 | dense | native on all backends |
| int8 convrot | I8 + per-row scale, 256-wide rotation | int8 tensor-core GEMM (W8A8) |
| int8 tensorwise | I8 + ONE scalar scale, no rotation | same GEMM, unrotated activation prep |
| int4 convrot | nibble-packed + per-row scale | W4A4 on CUDA; Vulkan has no `sint4`, so it decodes per GEMM to int8 |
| `asym_w4a8_int8` | 4-bit codebook indices + fp8 per-group scales | decodes to int8 convrot, per GEMM |
| NVFP4 | E2M1 nibbles + fp8 block scales, swizzled | decodes to bf16, feeds the bf16 GEMM |
| GGUF block quants (q4_k, q8_0, …) | ggml blocks | decodes per GEMM to int8 convrot or f16 (`--dit-gguf-gemm`), CUDA arms only |

- Packed weights stay packed and each consumer decodes on demand; materializing at load
  gives back the memory the format saved.
- Quantization is **per weight, not per model**: a checkpoint may mix formats block by
  block. A support probe must scan every device linear, not one tensor of one block —
  `DiT.device_lins` is the single list every scan reads.
- The activation prep is a property of the *activation*, the format a property of the
  *weight*; int8 and W4A8 share one prep.
- ⚠️ **Whether that prep ROTATES is a property of the checkpoint** (`lin.convrot`), and a
  scan for it must cover every storage form sharing the prep, W4A8 included: its weights are
  rotated, so leaving it out of the scan pairs them with an unrotated activation and renders
  uncorrelated noise. Both sides rotate or neither.
- ⚠️ **Weight storage must gate every GEMM call site, not just the main one.** Fast paths
  that key on "not int8 and not bf16" happily read a packed 4-bit weight as fp8 bytes.
- **A LoRA is a runtime sidecar, never a merge**: `models/lora.zig` (loader + host apply)
  and `models/lora_cuda.zig` (device), keyed on `Weight.tag`, so it is
  architecture-independent. `y = W x + s B (A x)` leaves an int8 base untouched and
  `strength` a runtime dial. The stack is swappable on a LIVE session
  (`Session.setLoras`, between forwards only), since nothing about a sidecar reads
  the checkpoint; ⚠️ it must evict each outgoing factor from the pointer-keyed
  device weight cache BEFORE freeing it, or the next stack is served the old
  copies and the swap leaks a whole stack. `Family.supportsLora` is the list of
  architectures with a sidecar arm at all, and a UI asks it before offering one. Same call-site hazard as above, and `minimax_h3.Lin` is the
  answer to it: the weight is reachable only as `.w`, so the sidecar is in every grep.
- ⚠️ **f16's 65504 ceiling is a real limit on real checkpoints**, met four times here
  (SDXL's VAE residual stream, the Flux/Z-Image VAE's attention logits, Z-Image's trunk
  activations, and Z-Image's unnormed attention V, which only overflows at DEPTH on a
  checkpoint whose residual is large enough). Symptom is a solid white image with no error. Fixes in use: bf16 instead of
  f16, an f32 scores plane, and `residual_act_div` (an exact power-of-two scale across the
  cast). `sd_vae.Config.act_f16` and `sd_unet.Config.act_f16` pick f16 activation storage per
  architecture, and it is **range** that gates it, not precision. With f16 storage on, every
  buffer a GEMM reads must be at that width, including ones that are not activations (the SD
  UNet's text conditioning is a cross-attention GEMM source, so `Session.ctx_d` narrows too).

## Samplers, schedulers and prompt dialects

Four orthogonal axes, each selectable on the CLI and in the GUI, and each recorded in the
saved PNG's AUTOMATIC1111 `parameters` block (a reader re-renders from that block, so a
hardcoded field there is a wrong answer). `pipeline.appendExtraParams` adds what the render
USED on top of what it was asked for -- `Clip 1`/`Clip 2`, `VAE`, `Model hash`/`VAE hash`
(AutoV2: sha256[:10], from a `<path>.sha256` sidecar, which is WRITTEN when missing so it is paid once ever; `--model-hash off` skips it), `Weight
dtype`, the resolved `Shift`, `Eta`/`Sigma noise` when overridden, and the LoRA stack --
under A1111's and Civitai's own field names, never invented ones. ⚠️ **Which text encoder
ran is not a detail**: Qwen3-VL-4B and plain Qwen3-4B share vocab, width and depth, load
interchangeably, and render differently.

| axis | file | surface |
|---|---|---|
| where the steps go | `core/schedule.zig` — all 9 ComfyUI schedulers plus the family sigma tables | `--scheduler` |
| how to step | `core/sampler.zig` — 11 of ComfyUI's samplers over four steppers | `--sampler`, `--eta`, `--s-noise` |
| prompt syntax | `core/clip_tokenizer.zig` (comfy), `core/prompt_a1111.zig`, shared parser `core/prompt_weights.zig` | `--prompt-syntax`, `--emphasis` |
| sampling compat | `core/noise.zig` selects `torch_rng.zig` or `philox_rng.zig` | `--compat`, `--rng`, `--sgm-noise-mult`, `--quantize-t` |

- `Scheduler.defaultFor` keeps the per-family default, so "model default" is a real choice.
- **`ddim_uniform` and `beta` do not return `steps + 1` sigmas.** Take the step count from
  `sigmas.len - 1`.
- **The SDE samplers need a real Brownian tree**, not a `randn`: `core/seed_seq.zig`
  (numpy `SeedSequence`) + `core/brownian.zig` (torchsde's dyadic `BrownianInterval`) +
  a noise generator. It keys on the sigma **quantized to 1e-6**, so a schedule value one
  ulp out draws unrelated noise — which is why `schedule.zig` reproduces torch's f32
  rounding exactly rather than being more accurate than it.
- ⚠️ **A blown-out DPM++ render on the SD `normal` schedule is the SAMPLER, not a bug.**
  That schedule's last rung before zero is the ladder's minimum (0.44 -> 0.029 at 8
  steps), so the multistep extrapolation coefficient explodes and the latent decodes as
  saturated colour. ComfyUI does the same, worse (`tools/render_sd_ref.py` renders one
  through ComfyUI's own `comfy.sample.sample` for a whole-render A/B). Karras or
  exponential is the fix; the toy-denoiser fixtures cannot see it, because a toy
  denoiser is a contraction and a real one is not.
- **A sampler may evaluate the model TWICE per step** (`Kind.evalsPerStep`): heun,
  dpm_2, the two second-order ancestral ones and dpmpp_sde place a probe at a sigma
  that is not on the schedule and evaluate there (`sampler.Model`, which the loop
  hands them). That is safe because every device session COMPUTES the timestep for a
  sigma it has no cached entry for rather than matching the nearest; it also means the
  same step count is twice the render time, which is what a time estimate has to ask
  rather than assume.
- **Not every sampler dispatches on the family.** `dpmpp_2m`, `heun` and `dpm_2` have
  no `CONST` arm in ComfyUI at all, so one body runs over flow and eps alike (dpmpp_2m
  takes `t = -log(sigma)` even on krea2); the ancestral ones and the SDE ones each ship
  two bodies. Matching ComfyUI means matching that split, not the principled choice.
- **The two stochastic sampler families do not draw from the same generator**, and neither
  choice is derivable: ComfyUI builds the SDE samplers' Brownian tree with `cpu=True`
  but leaves `default_noise_sampler` (the ancestral ones) on the latent's own device,
  so one ComfyUI render mixes a CPU latent with Philox ancestral noise from one seed.
  `sampler.stepNoiseSource` is that rule, and the ancestral noise is a SEQUENCE from one
  generator rather than a path addressed by sigma, so a resumed render winds it forward
  (`Stepper.resumeFrom`) instead of restoring a history.
- A compat/dialect choice must reach **every** consumer of it. Wiring the initial latent
  but not the Brownian tree makes euler reproduce perfectly while every SDE render is wrong.
- Prompt weights are a CLIP-only feature. The capability is declared on the encoder
  (`TextEncoder.supports_prompt_weights`), never inferred from a family name; an encoder
  that cannot apply them warns and encodes verbatim.

## Reference implementations and fixtures

`tools/gen_*.py` generate fixtures by **executing** the reference (ComfyUI, diffusers,
transformers, torchsde, A1111's own parser) rather than re-deriving it. Two tiers:

- **ungated** pure-op fixtures embedded per module (`src/{core,ops,models}/assets/`) —
  a module can only `@embedFile` under its own tree, so each owns its fixtures;
- **gated** real-checkpoint parity, tied to a file by sha256, behind `-Dintegration` +
  `requireModelFile`.

Rules that have repeatedly earned their keep:

- **Verify a fixture has teeth** by deliberately breaking the implementation and confirming
  it fails. A generator should assert that its corpus distinguishes the settings it pins.
- **Matching a reference exactly still leaves the choice of reference unvalidated.** A
  correct port of the wrong convention passes every test. Only a render comparison against
  the actual target catches it.
- **A reference is a piece of code and can be the thing that is wrong.** When structure
  matches pixel-for-pixel and only tone differs, suspect the harness's output mapping.
- **Read the control row first.** These models disagree with themselves across dtypes by
  more than they disagree with us; a PSNR figure without its precision floor is
  uninterpretable.
- **A green summary is not coverage** — gated tests self-skip when their checkpoint is
  missing. Check that a test actually ran before citing it.
- AGPL references (A1111) are **fetched at generation time, never vendored**; the fixture
  records the upstream sha256.

## Zig 0.16 conventions (differ from older Zig — do not use pre-0.16 patterns)

- `main` takes `std.process.Init`: `pub fn main(init: std.process.Init) !void`. Get the process-lifetime allocator via `init.arena.allocator()`, args via `init.minimal.args.toSlice(arena)`, and the `Io` instance via `init.io`.
- I/O goes through `std.Io`: writers are `*Io.Writer`; stdout is set up as `Io.File.Writer.init(.stdout(), io, &buffer)` and must be explicitly `flush()`ed.
- Container types are unmanaged-style: e.g. `std.ArrayList(T)` is initialized with `.empty` and takes the allocator per call (`list.append(gpa, x)`, `list.deinit(gpa)`).
- Fuzz tests use `std.testing.fuzz` with a `*std.testing.Smith` input generator.
- `.arena = arena` in a struct literal copies the arena's state before later fields
  allocate into it. Build every field into a local first, then construct.

## Dependencies

Runtime `dlopen`'d system libraries, gated per backend:
- Vulkan loader (`libvulkan.so.1`) — `--backend vulkan`.
- CUDA driver (`libcuda.so.1`) — `--backend zig-cuda` (hand-emitted PTX).
- `--backend cuda`: also `libcublasLt.so` (int8/f16 GEMM) and `libcudnn.so.9` (fused SDPA
  attention + conv).

ggml is an optional build dependency (`-Dggml`, default on): its CPU quant kernels back the
GGUF block-quant dequant and GEMV paths. With `-Dggml=false` those dtypes return
`error.QuantBackendUnavailable`.

## Keeping this file useful

This file is read in full at the start of every session. It orients an agent; it is not a
record of what happened.

- **Write durable facts only**: where something lives, what dispatches on what, an
  rule that must hold, a trap that is a silent wrong answer. One or two sentences.
- **No history.** No dates, no "landed", no "was X before", no postmortems, no measured
  tables, no PSNR figures, no "this cost a debugging cycle". `git log -p CLAUDE.md` has all
  of it if anyone needs it.
- **Put detail where it is used.** A fact about one file belongs in that file's module doc
  comment; a backend capability belongs in `BACKEND.md`; a Zig gotcha in `ZIG.md`; open work
  in `TODO.md`.
- **Prefer deleting to qualifying.** If a section no longer helps someone start work, remove
  it rather than annotating it.
- **Budget: keep this under ~300 lines.** If a section grows past a screen, that is the
  signal it belongs somewhere else.
