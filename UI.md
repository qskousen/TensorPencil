# TensorPencil — screen `2a` (Chat workspace)
## Implementation spec for Zig + dvui

Target window: **1200 × 705** logical px at scale 1.0 (44 title + 660 body + ~66 status bar).
All values below are logical px; multiply by `dvui.windowNaturalScale()` only if you hardcode
anything outside dvui's own `Rect`/`Options` machinery.

---

## 1. Palette

```zig
const C = struct {
    // surfaces
    pub const canvas       = dvui.Color.fromHex("#0b0d0f"); // chat pane bg / card bg
    pub const chrome       = dvui.Color.fromHex("#101317"); // title bar, status bar
    pub const rail         = dvui.Color.fromHex("#0e1114"); // left + right side rails
    pub const sunken       = dvui.Color.fromHex("#12161a"); // composer, vram track bg
    pub const spark_bg     = dvui.Color.fromHex("#0f1215"); // sparkline well
    pub const card         = dvui.Color.fromHex("#0f1316"); // tool-call card
    pub const raised       = dvui.Color.fromHex("#1c2126"); // buttons, "New chat"
    pub const raised_dim   = dvui.Color.fromHex("#151a1e"); // secondary buttons
    pub const chip         = dvui.Color.fromHex("#141a1e"); // quick-setting chips
    pub const sel_row      = dvui.Color.fromHex("#1a1f24"); // selected conversation
    pub const bubble_user  = dvui.Color.fromHex("#1e2429"); // user message bubble
    pub const seg_track    = dvui.Color.fromHex("#181c20"); // Chat/Studio toggle well
    pub const queue_active = dvui.Color.fromHex("#141a1f");
    pub const queue_idle   = dvui.Color.fromHex("#111417");
    pub const meter_track  = dvui.Color.fromHex("#22282d");

    // text
    pub const text_hi      = dvui.Color.fromHex("#e6e8ea"); // primary
    pub const text         = dvui.Color.fromHex("#c9cdd1"); // agent prose
    pub const text_dim     = dvui.Color.fromHex("#8b9196"); // secondary / labels
    pub const text_faint   = dvui.Color.fromHex("#6d7378"); // monospace captions
    pub const text_ghost   = dvui.Color.fromHex("#5c6368"); // placeholders, section heads
    pub const text_ink     = dvui.Color.fromHex("#0b0d0f"); // on light fills

    // accents — RESERVED MEANINGS, do not reuse
    pub const amber        = dvui.Color.fromHex("#f0a03c"); // "machine is working" only
    pub const blue         = dvui.Color.fromHex("#6ea8f0"); // user-owned controls/selection
    pub const danger       = dvui.Color.fromHex("#d9584f");

    // VRAM segment colors — LLM side is cool, diffusion side is warm
    pub const vram_sys     = dvui.Color.fromHex("#3f464c"); // hatch fg, other programs
    pub const vram_sys_alt = dvui.Color.fromHex("#343b41"); // hatch bg
    pub const vram_llm     = dvui.Color.fromHex("#3f9c8f"); // weights
    pub const vram_ctx     = dvui.Color.fromHex("#4f7fb0"); // kv cache
    pub const vram_ovh     = dvui.Color.fromHex("#6d7378"); // runtime overhead
    pub const vram_free    = dvui.Color.fromHex("#1b2024"); // rail gap
    pub const vram_lat     = dvui.Color.fromHex("#d9584f"); // latents
    pub const vram_vae     = dvui.Color.fromHex("#9a6fd8");
    pub const vram_te      = dvui.Color.fromHex("#e8c33f"); // text encoder
    pub const vram_dit     = dvui.Color.fromHex("#e08a2e"); // unet / dit
};

// hairlines — always alpha over the surface, never a solid grey
const hairline      = dvui.Color{ .r=255,.g=255,.b=255,.a=18 }; // ~.07 — chrome edges
const hairline_soft = dvui.Color{ .r=255,.g=255,.b=255,.a=15 }; // ~.06 — inside rails
const hairline_hi   = dvui.Color{ .r=255,.g=255,.b=255,.a=28 }; // ~.11 — composer border
```

**Rule:** amber means *the machine is doing something* (progress, running job, VRAM
pressure). Blue means *the user owns this* (selection, focus, a value they set). Never
use amber for a button and never use blue for progress.

---

## 2. Type

Two families, both variable-free weights:

| role | family | size | weight | line |
|---|---|---|---|---|
| app name | IBM Plex Sans | 12 | 600 | 1.0, +0.03em tracking |
| UI label / button | IBM Plex Sans | 10.5–11.5 | 500 | 1.0 |
| conversation row | IBM Plex Sans | 11.5 | 400 | 1.35 |
| chat prose + bubbles | IBM Plex Sans | 12.5 | 400 | 1.55–1.65 |
| composer placeholder | IBM Plex Sans | 12.5 | 400 | 1.0 |
| section head (TODAY) | IBM Plex Mono | 9.5 | 500 | 1.0, +0.1em, uppercase |
| telemetry value | IBM Plex Mono | 10 | 500 | 1.0 |
| telemetry label | IBM Plex Mono | 8.5 | 500 | 1.0, +0.07em |
| legend chip | IBM Plex Mono | 8.5 | 400 | 1.0 |
| metadata / params | IBM Plex Mono | 9.5–11 | 400 | 1.0–1.5 |

```zig
const F = struct {
    pub const app     = dvui.Font{ .name = "sans_semibold", .size = 12 };
    pub const ui      = dvui.Font{ .name = "sans_medium",   .size = 11 };
    pub const ui_sm   = dvui.Font{ .name = "sans_medium",   .size = 10.5 };
    pub const row     = dvui.Font{ .name = "sans",          .size = 11.5 };
    pub const prose   = dvui.Font{ .name = "sans",          .size = 12.5 };
    pub const mono    = dvui.Font{ .name = "mono",          .size = 10.5 };
    pub const mono_sm = dvui.Font{ .name = "mono",          .size = 8.5 };
    pub const mono_hd = dvui.Font{ .name = "mono_medium",   .size = 9.5 };
    pub const mono_val= dvui.Font{ .name = "mono_medium",   .size = 10 };
};
```

The names are **roles, not files** — see §12, which is where the real work is. Never
reference a concrete face (`PlexSans-Regular`) from a call site; a role resolves to a
fallback chain at draw time.

dvui has no letter-spacing. For the uppercase mono labels (`TODAY`, `EARLIER`, `RECENT`,
`GPU/CPU/VRAM`) either accept default tracking or draw per-glyph with `dvui.renderText` at
+0.8px advance. Accepting default is fine; it costs a little density.

---

## 3. Radii, borders, spacing

```zig
const R = struct {
    pub const chip   = dvui.Rect.all(5);  // legend swatch 1, pills 5–6
    pub const button = dvui.Rect.all(6);
    pub const panel  = dvui.Rect.all(7);  // queue rows, New chat
    pub const card   = dvui.Rect.all(9);  // tool-call card
    pub const input  = dvui.Rect.all(10); // composer
    pub const track  = dvui.Rect.all(2);  // vram bar, sparkline well
};
```

Spacing scale actually used: **2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 24, 30**.
Gaps dominate; there are almost no margins. Every sibling group is a box with a `gap`,
so in dvui use `.box(.horizontal/.vertical, .{})` plus explicit `.margin` on children,
or a small helper:

```zig
fn gapBox(src: std.builtin.SourceLocation, dir: dvui.enums.Direction, gap: f32, opts: dvui.Options) *dvui.BoxWidget {
    // set .margin on each child to gap/2, or emit spacers between children
}
```

---

## 4. Window layout

```
┌─ title bar ─────────────────────────────────────────── h 44, bg chrome ┐
│ traffic lights · TensorPencil · [Chat|Studio] ······ model chips       │
├─ body ────────────────────────────────────────────── h 660 (flex) ─────┤
│ sidebar 206 │ chat column (flex, min 0)        │ queue rail 272        │
├─ status bar ────────────────────────── h ~66, bg chrome, full width ───┤
│ GPU 86 │ CPU 86 │ VRAM 86 ║ two-sided vram track + legend (flex)       │
└────────────────────────────────────────────────────────────────────────┘
```

Three vertical bands, fixed-fixed-flex-fixed. The status bar spans the **whole window**,
under all three columns — that is deliberate: telemetry is app-global, not a property of
the queue.

```zig
pub fn frame(state: *AppState) !void {
    var win = dvui.box(@src(), .vertical, .{ .expand = .both, .background = true,
                                             .color_fill = .{ .color = C.canvas } });
    defer win.deinit();

    try titleBar(state);                       // 44
    {
        var body = dvui.box(@src(), .horizontal, .{ .expand = .both });
        defer body.deinit();
        try sidebar(state);                    // 206 fixed
        try chatColumn(state);                 // expand .both, min_size_content.w = 0
        try queueRail(state);                  // 272 fixed
    }
    try statusBar(&state.telemetry);           // ~66
}
```

---

## 5. Title bar (h 44)

- padding `0 14`, gap 16, bottom border `hairline`.
- Traffic lights: three 10px circles, `#2c3237`, gap 6. Decorative on Win/Linux — on macOS
  let the OS draw them and just reserve 62px.
- `TensorPencil` in `F.app`, `C.text_hi`.
- **Segmented toggle**: well `seg_track`, radius 8, 1px `hairline`, padding 2, gap 2.
  Active segment: fill `text_hi`, label `text_ink`, radius 6, padding `6 12`.
  Inactive: no fill, label `text_dim`. Bind to ⌘1 / ⌘2.
- Right group (`margin-left:auto`): two chips — `sdxl-turbo · fp16 ▾` (a dropdown) and
  `qwen-7b + sdxl-turbo` (static, shows both residents). Chip = fill `#171b1f`,
  1px `hairline`, radius 6, padding `6 9`, `F.mono` in `text_dim`.

---

## 6. Sidebar (w 206)

bg `rail`, right border `hairline_soft`, padding `14 12`, vertical gap 16.

1. **New chat** — full width, fill `raised`, 1px `#ffffff17`, radius 7, padding 9,
   centered `+ New chat` in Plex Sans 11.5/500.
2. **Conversation list** — gap 3. Section heads (`TODAY`, `EARLIER`) in `F.mono_hd`,
   `text_ghost`, padding `0 8 7`. Rows: padding 8, radius 6.
    - selected: fill `sel_row`, title `text_hi`, plus a 9.5px mono sub-line in `C.blue`
      (`2 studio edits`) — blue because it records the *user's* manual edits.
    - unselected: no fill, title `text_dim`, no sub-line.
    - hover: fill `#ffffff0a`.
3. **Footer** pinned bottom (`margin-top:auto`), top border `hairline_soft`, padding-top 10.
   `Models` row carries an amber `73%` right-aligned while a download runs; `Settings` below.

---

## 7. Chat column (flex)

Two children: scrollable transcript (flex, `min_size 0`, clipped) and a fixed composer block.

### Transcript — padding `24 30`, vertical gap 18, `overflow: hidden` → `dvui.scrollArea`

- **User bubble** — right-aligned, max width 400, fill `bubble_user`, 1px `hairline`,
  radius `10 10 3 10` (the 3 is the tail corner, bottom-right), padding `11 14`, prose 12.5/1.55.
- **Agent prose** — no bubble, no avatar. Max width 560, `C.text`, 12.5/1.65. This asymmetry
  is load-bearing: the agent is the page, the user is a card on it.
- **Tool-call card** — max width 600, fill `card`, 1px `#ffffff17`, radius 9, clipped.
    - header: padding `10 12`, bottom border `hairline_soft`, gap 9 —
      an amber tag `image.generate` (fill `amber @ 12% alpha`, radius 4, padding `4 6`,
      mono 10/500), then `×4 · 1024² · seed 8812` in `text_dim`, then a right-aligned
      disclosure `▸ prompt & reasoning`.
    - body: 4-column grid of 1:1 thumbs, gap 8, padding 12, radius 5. Selected thumb gets a
      1.5px `C.blue` border (no glow, no scale).
    - actions: padding `0 12 12`, gap 8 — primary `Open in Studio ⌘2` (fill `text_hi`,
      label `text_ink`), two secondaries (fill `raised_dim`, 1px `hairline`), then a
      right-aligned `render 03 selected` caption in mono 10 `text_ghost`.
- **Returned-from-Studio card** — max width 600, 2px left border in `C.blue`,
  fill `blue @ 5% alpha`, radius `0 8 8 0`, padding `11 13`, gap 12: a 44px thumb,
  a title (`Returned from Studio`, Sans 11.5/500) with a mono diff line
  (`cfg 5.0 → 4.2 · steps 30 → 34 · +coastal-mist 0.30`), and a right-aligned
  `Continue here` outline button. This is the whole Chat↔Studio handoff made visible —
  do not drop it.

Placeholder images in this mockup are 45° stripes (`#1c2126` / `#161a1e`, 6px period);
in the real app they are decoded textures. Keep the same two greys for the *loading*
placeholder so nothing pops on swap.

### Composer block — top border `hairline_soft`, padding `12 30 18`, vertical gap 10

- **Quick settings row**, gap 8: three chips (`3:2 ▾`, `×4 ▾`, `quality: balanced ▾`) —
  fill `chip`, 1px `#ffffff14`, radius 6, padding `6 9`, mono 10.5 `text_dim`.
  Right-aligned `All settings in Studio ↗` in Sans 10.5 `text_ghost`.
  These three are the only knobs a beginner ever needs; everything else lives in Studio.
- **Input** — 1px `hairline_hi`, radius 10, fill `sunken`, padding `12 14`, gap 12:
  a flexible text entry (placeholder `Describe an image, or ask for changes…` 12.5
  `text_ghost`), a `+ reference` affordance in mono 10.5, and a `Send` button
  (fill `text_hi`, `text_ink`, radius 6, padding `7 14`).

---

## 8. Queue rail (w 272)

bg `rail`, left border `hairline_soft`, column layout. **No telemetry here** — it all moved
to the status bar.

1. **Tab strip** — padding `10 12`, bottom border `hairline_soft`, gap 2.
   `Queue · 2` active (fill `text_hi`, `text_ink`, radius 5, padding 7, flex 1),
   `Library` inactive (`text_dim`).
2. **Body** — padding 12, gap 9, clipped:
    - *running row*: fill `queue_active`, 1px `amber @ 22% alpha`, radius 7, padding 9, gap 10.
      52px thumb; title Sans 11 `text_hi`; `18 / 34 · 4.2 it/s` mono 9.5 `text_dim`;
      3px progress track `meter_track` with an amber fill at 53%.
    - *queued row*: fill `queue_idle`, 1px `hairline_soft`, 52px **dashed** placeholder
      (1px `#ffffff1f`), sub-line `queued · ~11 s · from Studio` in `text_ghost`.
      Provenance matters — say when a job came from Studio.
    - *hint row*: `drag to reorder` left, `Pause all` right, mono 10 `text_ghost`.
    - `RECENT` head, then a 3-col grid of 1:1 thumbs, gap 7, radius 6.

Drag-to-reorder in dvui: track `dvui.dragging()` on the row, reorder your slice on drop,
and give rows stable ids via `.id_extra = job.id`.

---

## 9. Status bar (h ~66) — the two-sided VRAM meter

Full window width, bg `chrome`, top border `hairline`, padding `9 14 10`, horizontal gap 14.

### 9.1 Left: three history meters (86px each, gap 10)

Per meter, vertical gap 4:
- header row: label mono 8.5/500 `text_faint`, value right-aligned mono 10/500 in the
  meter's own color.
- sparkline well: h 22, fill `spark_bg`, radius 2, padding 2, **26 bars**, gap 1,
  bottom-aligned, `flex:1` each, height = value%.

Colors: GPU and CPU always `C.blue`. VRAM is `C.blue` under 80%, `C.amber` at 80–95%,
`C.danger` above 95% — that threshold is the only place the bar shouts.

Nothing else: no clock speeds, no `x / y GB`, no second progress bar. The number and the
history are the whole meter.

```zig
const Meter = struct {
    label: []const u8,
    history: [26]f32,   // ring buffer, oldest at index 0, sampled ~1 Hz
    value: f32,         // 0..1
    color: dvui.Color,
};

fn meter(src: std.builtin.SourceLocation, m: Meter) !void {
    var b = dvui.box(src, .vertical, .{ .min_size_content = .{ .w = 86 } });
    defer b.deinit();

    var hdr = dvui.box(@src(), .horizontal, .{ .expand = .horizontal });
    try dvui.labelNoFmt(@src(), m.label, .{ .font = F.mono_sm, .color_text = .{ .color = C.text_faint } });
    try dvui.label(@src(), "{d:.0}%", .{ m.value * 100 },
        .{ .gravity_x = 1.0, .font = F.mono_val, .color_text = .{ .color = m.color } });
    hdr.deinit();

    // sparkline: one manual pass is cheaper than 26 widgets
    var well = dvui.box(@src(), .horizontal, .{
        .min_size_content = .{ .h = 22 }, .expand = .horizontal,
        .background = true, .color_fill = .{ .color = C.spark_bg },
        .corner_radius = R.track, .padding = dvui.Rect.all(2),
    });
    const r = well.data().contentRect();
    const bw = (r.w - 25) / 26;           // 26 bars, 1px gaps
    for (m.history, 0..) |v, i| {
        const h = @max(1.0, v * r.h);
        try dvui.pathFillConvex(  // or dvui.Rect{...} via renderRect
            dvui.Rect{ .x = r.x + @as(f32, @floatFromInt(i)) * (bw + 1),
                       .y = r.y + r.h - h, .w = bw, .h = h },
            m.color);
    }
    well.deinit();
}
```

### 9.2 Divider

1px, `hairline`, stretched to the bar's content height, 2px vertical margin.

### 9.3 Right: the two-sided track (flex, min 0)

Three stacked rows, gap 4:

**(a) Ownership rail — h 3.** Segments, left to right:
`sys 10%` (`vram_sys`) · `1px` gap (`canvas`) · `LLM total 21%` (`vram_llm`) ·
`flex free` (`vram_free`) · `DIFFUSION total 51%` (`vram_dit`).
The rail's only job is to say *which side owns what*, so its boundary is the fastest read
of who is winning. Derive both spans by summing the track's segments — never hardcode
them separately or they drift.

**(b) Track — h 20**, fill `sunken`, radius 2, clipped. One continuous 24 GB scale read
from both ends:

| order | segment | color | note |
|---|---|---|---|
| 1 | `sys` | 45° hatch, `vram_sys` on `vram_sys_alt`, 4px period | other programs; pinned far left, outside the LLM bracket |
| — | 1px divider `canvas` | | hard edge: not ours to reclaim |
| 2 | `weights` | `vram_llm` | LLM grows **rightward** |
| 3 | `ctx` | `vram_ctx` | kv cache, the segment that grows per turn |
| 4 | `ovh` | `vram_ovh` | |
| 5 | **free** | none (shows `sunken`) | flex — the gap IS free VRAM |
| 6 | `lat` | `vram_lat` | diffusion grows **leftward**, so its order mirrors |
| 7 | `vae` | `vram_vae` | |
| 8 | `te` | `vram_te` | |
| 9 | `dit` | `vram_dit` | |

**No text inside the bars. Ever.** No segment labels, no amounts, no free readout. The bar
is pure color; every number lives in the legend row beneath it.

**(c) Legend row** — mono 8.5, gap 10 within a side, sides pushed apart by a flexible
middle. Handedness matches the bar exactly:

- left group: `LLM 4.8G →` (500 weight, `vram_llm`), then swatch+label pairs
  `llm 3.6G`, `ctx 0.8G · 6k`, `ovh 0.4G`, `sys 2.4G`.
- middle (flex, centered): `4.7G free` — `text_ghost`, turning `amber` under ~2G.
- right group: `TE 3.3G`, `diffusion 7.9G`, `VAE 0.8G`, `lat 0.1G`, then
  `← DIFFUSION 12.1G` (500 weight, `vram_dit`).

Swatch = 6×6, radius 1, gap 5 to its label.

**Unloaded engine:** its segments are simply absent and the gap widens — no ghost slot, no
dashed placeholder, no eviction marks or handover animation. The legend labels go to
`text_ghost` with an em-dash (`llm —`), and the group total reads `LLM 0.0G`. The bar
states only what is resident right now.

```zig
const VramState = struct {
    total_bytes: u64,
    sys: u64,                          // other processes
    llm: struct { weights: u64, ctx: u64, ovh: u64, ctx_tokens: u32 },
    dif: struct { te: u64, dit: u64, vae: u64, lat: u64 },

    fn llmTotal(v: @This()) u64 { return v.llm.weights + v.llm.ctx + v.llm.ovh; }
    fn difTotal(v: @This()) u64 { return v.dif.te + v.dif.dit + v.dif.vae + v.dif.lat; }
    fn free(v: @This()) u64 { return v.total_bytes - v.sys - v.llmTotal() - v.difTotal(); }
};
```

Lay the track out by computing each segment's width as
`@round(fraction * track_width)` in one pass, giving the **free** segment the remainder,
so rounding error never shows up as a seam. The rail is a second pass over the same
fractions.

---

## 10. Behaviours worth preserving

1. **Telemetry is global.** It sits under all three columns and never migrates into a panel.
2. **Two accents, two meanings.** Amber = machine working. Blue = user's own choice.
   The whole screen only ever uses these two against neutrals.
3. **Every generation is a tool call in the transcript**, with prompt and reasoning behind
   one disclosure — visible depth, collapsed by default.
4. **Studio edits come back as messages.** The conversation is the record of what happened,
   including things the user did by hand.
5. **Queue rows carry provenance** (`from Studio`) and the running row is the only amber
   thing in the rail.
6. **Sparklines are ~1 Hz, 26 samples (~26 s window).** Sample on a timer, not per frame,
   or the graph turns into noise at 60 fps. `dvui.timerSet(1_000_000)` and push one sample.

---

## 11. dvui gotchas for this layout

- **`min_size_content.w = 0` on the chat column**, otherwise its long prose lines force the
  window wider than 1200 and the side rails get squeezed.
- **Clipping**: dvui boxes don't clip by default. The transcript and the queue body both
  rely on it — use `dvui.scrollArea` (which clips) or set `.clip = true` (`clipSet`) around
  the manual draws.
- **Right alignment** (`margin-left:auto` in the mockup) is `.gravity_x = 1.0` on the child
  inside an expanded horizontal box.
- **Hairlines**: use `.border` with an alpha color, not a filled 1px box, so the edge sits
  correctly over both `chrome` and `rail`.
- **Hatch fill** for `sys` has no dvui primitive: draw it yourself with a clip rect and a
  loop of 4px-wide rotated quads via `dvui.pathStroke`, or bake an 8×8 tiling texture once
  at startup and `dvui.renderImage` it — the texture route is cheaper and crisper.
- **Rounded per-corner radii** (the `10 10 3 10` chat bubble) map to
  `dvui.Rect{ .x = tl, .y = tr, .w = br, .h = bl }` in `.corner_radius`.
- **Fonts**: register the role chains at startup via `dvui.addFont` — see §12.

---

## 12. Unicode, scripts, and font strategy

This is the part of the spec most likely to cost you weeks, so it gets its own section.
There are **three separate problems** and they are usually conflated:

1. **Coverage** — does a glyph exist for this codepoint?
2. **Shaping** — do the glyphs combine correctly? (Arabic joining, Devanagari conjuncts and
   reordering, Thai mark stacking, Latin ligatures, Korean jamo composition)
3. **Bidi** — does mixed RTL/LTR text lay out in the right visual order?

dvui's text stack (stb_truetype + a per-font glyph cache) solves **none** of these for you.
It renders a codepoint run left-to-right through one font. So decide up front how far you
intend to go, because the answers are very different in cost.

### 12.1 No single font file can do it

A TrueType font is capped at 65,535 glyphs; Unicode has ~150,000 assigned codepoints. Even
the "everything" fonts don't: Noto Sans is a *family* of ~200 files, and the CJK fonts alone
are 15–20 MB each. Anyone who tells you to "just ship one Unicode font" is quietly dropping
CJK, or Arabic, or both.

So: a **fallback chain per role**, resolved per codepoint.

### 12.2 Recommended chain — keep IBM Plex, extend it

Good news for the design: IBM Plex is itself a large multi-script superfamily under the OFL,
so you can keep the intended texture across most of the world's readers and only leave Plex
for the long tail.

| tier | sans role | mono role | covers |
|---|---|---|---|
| 1 | IBM Plex Sans | IBM Plex Mono | Latin, Greek, Cyrillic, Vietnamese |
| 2 | IBM Plex Sans Arabic | *(sans)* | Arabic, Persian, Urdu |
| 2 | IBM Plex Sans Hebrew | *(sans)* | Hebrew |
| 2 | IBM Plex Sans Devanagari / Thai | *(sans)* | Hindi, Marathi, Thai |
| 2 | IBM Plex Sans JP / KR / SC / TC | *(sans)* | Japanese, Korean, Chinese |
| 3 | Noto Sans \<script\> | *(sans)* | everything Plex lacks — Bengali, Tamil, Khmer, Ethiopic, Armenian, Georgian… |
| 4 | Noto Sans Symbols 2 / Math | *(sans)* | arrows, box drawing, `▾ · × ² ⌘ ↗ ⏸ ⊘` |
| 5 | last-resort `.notdef` box | | never render nothing |

Tier 4 matters more than it looks: the UI copy in this design already uses `·`, `×`, `²`,
`→`, `↗`, `⌘`, `▾`, `⏸`, `⊘`. Plex covers most, but verify each one at build time rather
than discovering a tofu box in a screenshot.

**Mono deliberately does not fall back to mono.** The mono role only ever renders digits,
Latin identifiers, model names, and units (`4.2 it/s`, `19.3 / 24 G`). If a user's *content*
(a prompt, a chat title) needs a script Plex Mono lacks, fall back to the **sans** chain for
that run — a proportional Devanagari run beats a tofu row. Reserve true mono for numerics,
where you also want tabular figures.

### 12.3 Loading: bundle a core, lazy-load the rest

Bundling every tier is ~60 MB. Instead:

- **Embed** tier 1 + tier 4 (Latin/Greek/Cyrillic + symbols), ~1.5 MB — the app always works.
- **Ship the rest as files next to the binary** and `mmap`/read them on first need. dvui's
  `addFont(name, bytes, ttc_index)` takes a byte slice, so a lazily-read file is fine as long
  as you keep the slice alive for the program's lifetime (dvui does **not** copy it).
- Register lazily but **resolve deterministically**: build the coverage index at startup from
  each face's `cmap` (parse it once, cache a bitset per face to disk), not by trial-rendering.

```zig
const FontRole = enum { sans, sans_medium, sans_semibold, mono, mono_medium };

const Face = struct {
    dvui_name: []const u8,
    coverage: *const CodepointSet, // built from cmap, cached
    loaded: bool,
    path: []const u8,
};

const chains = std.EnumArray(FontRole, []const *Face){ /* tiers, in order */ };

/// Split a UTF-8 string into runs that share one face.
fn runsFor(role: FontRole, text: []const u8, out: *std.ArrayList(Run)) !void {
    var it = std.unicode.Utf8Iterator{ .bytes = text, .i = 0 };
    var cur: ?*Face = null;
    var start: usize = 0;
    while (it.nextCodepointSlice()) |s| {
        const cp = try std.unicode.utf8Decode(s);
        const face = pick(role, cp) orelse lastResort;
        if (cur != null and face != cur.?) {
            try out.append(.{ .face = cur.?, .bytes = text[start .. it.i - s.len] });
            start = it.i - s.len;
        }
        cur = face;
    }
    if (cur) |f| try out.append(.{ .face = f, .bytes = text[start..] });
}
```

Then every label in this spec goes through one helper that draws runs back to back instead of
calling `dvui.labelNoFmt` directly. Do this **on day one** — retrofitting it across 60 call
sites is the expensive version.

**Cache the runs.** Static UI strings (`Open in Studio ⌘2`) resolve once; only user content
(chat text, model names, filenames) needs re-resolution, and that changes rarely. Key the
cache on `(role, string ptr/hash)`.

### 12.4 Metrics and vertical rhythm

Mixing faces breaks the line rhythm this design depends on:

- **Normalize by cap height, not em.** Plex Sans JP at the same nominal size looks larger;
  scale each fallback face by a per-face factor so cap heights match tier 1. Store the factor
  next to the coverage set.
- **Freeze line height per role**, don't derive it from the tallest face in the run —
  otherwise a single CJK character in a chat title changes that row's height and the whole
  sidebar reflows. The chat prose row is 12.5 × 1.65 ≈ 21px; keep it 21px regardless.
- **CJK is not monospace-compatible.** Ideographs are full-width; in the 272px queue rail a
  Japanese title fits ~13 characters. Budget by *width*, not character count, and ellipsize
  by measuring.
- **Tabular figures** for all telemetry: the sparkline values, `18 / 34 · 4.2 it/s`, and the
  legend amounts jitter horizontally if digits are proportional. Plex Mono is tabular by
  default; if a fallback isn't, right-align the number in a fixed-width box (the meters
  already reserve 86px, so this is nearly free).

### 12.5 Shaping and bidi — decide the tier of support

Be explicit with yourself about which of these you're shipping:

- **Tier A — Latin/Greek/Cyrillic only.** No extra deps. Codepoint→glyph is correct.
- **Tier B — + CJK.** Still no shaping needed (ideographs don't combine), but you need line
  breaking *between* characters rather than at spaces (UAX #14). ~200 lines of table-driven
  code, or the `zg` Zig Unicode library.
- **Tier C — + Arabic/Hebrew/Indic/Thai.** You need real shaping and bidi. There is no
  credible pure-Zig implementation yet: link **HarfBuzz** (C, builds fine from `build.zig`)
  and **SheenBidi** (small, C, permissive) or FriBidi. Budget a week to wire
  shape→glyph-indices→dvui's atlas, because you will be feeding dvui *glyph ids*, not
  codepoints, which means bypassing `dvui.renderText` for a custom run renderer.
- **Tier D — emoji.** Color emoji are CBDT/sbix/COLRv1 bitmap or layered-vector tables that
  stb_truetype won't touch. If you need them, render via a separate path (decode the bitmap
  strike yourself and `renderImage`). Given this design uses no emoji, I'd ship monochrome
  Noto Emoji at most and move on.

My recommendation for a v1: **Tier B**, with the run-splitting architecture from 12.3 in
place so Tier C is a renderer swap and not a rewrite. Put a known-limitation note in the
release rather than shipping mis-shaped Arabic, which reads as broken rather than absent.

### 12.6 Input, not just output

Coverage is half the problem — users have to *type* those scripts:

- **IME**: CJK/Indic input needs preedit (composition) support. SDL gives you
  `SDL_TEXTEDITING`; dvui's `TextEntryWidget` handles committed text but you'll want to draw
  the underlined preedit run yourself, and call `SDL_SetTextInputRect` so the OS candidate
  window lands under the caret — in this layout, under the composer at the window foot.
- **Grapheme clusters**: backspace, arrow keys, and selection must move by cluster, not
  codepoint, or users delete half a Devanagari syllable or an emoji ZWJ sequence.
- **Normalization**: normalize to NFC before hashing or comparing prompt text, or
  identical-looking prompts produce different cache keys.

### 12.7 Testing

Keep a fixture file of one string per script and render it into every text slot in the app
(chat bubble, conversation row, queue title, composer, tooltip) in CI, then diff screenshots.
The failure modes are visual — tofu, clipping, baseline drift, reflowed rows — and none of
them raise an error at runtime.

---

## 13. What shipped, and where it diverges

This section is the record of the built screen, written after the fact. Where it
contradicts the sections above, this section is right.

### Modules

| file | holds |
|---|---|
| `src/gui/style.zig` | the palette, type scale, radii, hairlines, the dvui `Theme`, and the shared primitives (chip, chip dropdown, segmented toggle, section head, sparkline, history meter, hatch, marks) |
| `src/gui/fonts.zig` | the three-tier face chain and per-codepoint run splitting |
| `src/gui/shell.zig` | title bar, sidebar, band arithmetic |
| `src/gui/bubbles.zig` | transcript leaves (user bubble, agent prose, tool card) and the composer frame |
| `src/gui/queue_rail.zig` | the right rail |
| `src/gui/meter.zig` | the two-sided VRAM track, its handles and legend |
| `src/gui/history.zig` | the conversation store (format, index, day grouping) |
| `src/gui/framing.zig` | ratio + megapixels → model-legal dimensions |
| `src/gui/ui_probe.zig` | `zig build ui-probe` — renders the whole screen from canned data to a PNG |

`shell`, `bubbles` and `queue_rail` render from plain data and report through
callbacks, which is what lets `ui-probe` draw the real screen with no model, no
GPU and no engine. Add `--states` to render the status bar under three loads.

### Title bar (§5)

No traffic lights and no `TensorPencil` wordmark: the mockup was imitating a
macOS window, and the real window has a title bar of its own (the SDL window
title is `TensorPencil`). The bar starts with the Chat/Studio toggle, which is
the first thing on it that is actually a control.

### Type (§2)

dvui's `Font.size` is the height of a capital M, and `line_height_factor`
multiplies the face's hhea span, NOT the cap height. So the spec's nominal px
and line multiples are converted in `style.role`: `size = nominal × 0.698`,
`factor = line ÷ 1.30`. Both constants are read back out of the font file by a
test, because getting either wrong is invisible until you look at the screen.

A useful accident: two faces at the same dvui size have the same cap height, so
the cross-face normalization §12.4 asks for is free.

### Marks (§12.2 tier 4)

`▾ ▸ ✕` are missing from **every** face we bundle, and Plex has none of
`▾ ⌘ ⏸ ⊘ ⏏ ✕ ▸ ⚙`. UI affordance marks are therefore vector icons
(`style.Mark`), not glyphs. Text decorations that ARE glyphs (`◦ ▪ ▎ ─`) must go
through `fonts.addStyled`; emitting one with a bare `addText` puts tofu on the
screen, which is how the markdown probe caught it.

### Tool-call card (§7)

Trimmed to what the engine actually does. No `Vary`, no `Upscale ×2`, no
keyboard hint on `Open in Studio` (which loads that render's exact parameters
into the Studio form).

The card keeps its own `prompt` disclosure. Separately, a collapsible
**`Tool call`** section sits where the model actually emitted the call, in the
same sunken-well-with-a-blue-edge treatment as a reasoning block, because it is
the same kind of thing: the machine's working, available but not in the way.

That means the reply renders **in document order** — prose, call section, card,
prose — rather than stripping every call out and hanging all the images off the
end, which read out of order whenever the model talked between generations.
Consecutive calls group into one card, so a four-image request is still one 2x2
grid. The walk is a pure function (`toolcall.segments`) with tests, because its
failure mode is an off-by-one that shows the WRONG PICTURE under a card rather
than an error.

A layout consequence worth knowing: the chat column's width is now COMPUTED as
`band − sidebar − rail` rather than left to the box layout. `min_size_content.w
= 0` is not enough — a child's min size still propagates up, so one long
unbroken line (a raw tool call, a URL) grew the column and squeezed both rails
to slivers. `Layout.sidebar_w` / `rail_w` are therefore OUTER widths, chrome
included, and each rail subtracts its own padding and border.

**The card and the rail hand off rather than duplicate.** While a call is
rendering, its tiles are dashed empty slots and the caption reads
`rendering n of m · in queue`; the live pixels are in the rail. As each image
lands it fills its slot and leaves the queue. An image is in exactly one place
at any instant.

There is no "Returned from Studio" card — that handoff does not exist.

### Queue rail (§8)

`RECENT` is gone from the Queue tab, for the same no-duplication reason: a
finished image is already in the transcript that asked for it. The Queue tab
shows in-flight work only and drains to empty; **Library** holds every finished
image, chat and studio.

Rows carry provenance (`from Studio`) and drag to reorder via dvui's
`ReorderWidget`, with a dim handle — the mockup shows none, but an invisible
drag target is a feature nobody finds. Reordering calls `Diffuser.movePending`,
which is UI-thread-only and refuses a generating image.

### Composer (§7)

Quick settings are **aspect ratio** (a menu) and a **megapixel budget** (typed,
so "1.8" is as easy as a preset), plus the reasoning toggle where the model
supports it. `framing.zig` maps the pair to model-legal dimensions, rounding to
a multiple of 64 and deriving height from the snapped width so the aspect does
not drift.

**These chips are the only control for image size.** Settings has no
width/height rows: two controls writing one value means whichever was touched
last silently wins, and that is exactly what happened — Settings and Studio each
seed their form buffers ONCE, so a chip change left both holding the old numbers
and Settings' Apply wrote them straight back over it. `config.framing_ratio` and
`config.framing_mp` are now the persisted source; `width`/`height` are derived
by `Config.applyFraming` and both dependent views are told to re-seed.

The chip menu is hand-rolled rather than `dvui.dropdown`, which hardcodes
`.avoid = .none` and so always opens downward — off the bottom of the window,
since the composer sits at the window foot.

### VRAM meter (§9)

The bar keeps its two draggable handles and its eject/pause buttons, which the
mockup has no equivalent for. They sit with their own side's legend group, so
the bar's handedness carries through to the controls that act on it.

The bar is **74px, not 66**, and its type is two steps above the spec: the
legend is 11.5 and the percentages 12.5, because these are numbers a user
compares against each other and 8.5 is a size you can see but not read. The
legend measures itself and drops entries by importance (`ovh`, `lat`, `VAE`
first) until the row fits, so a narrow window loses detail rather than silently
clipping the diffusion total off the end.

The three sparklines get **a hue each** — green GPU, blue CPU, violet VRAM —
rather than the spec's uniform blue: three identical graphs side by side are
three things you have to read the label of. VRAM keeps its threshold
escalation to amber at 80% and danger above 95%, since it is the only one of
the three that actually fails.

**The limit handle is on the LEFT.** It marks the RESERVE the user asks to keep
free, which is what `vram.resolve` means by it: the effective reserve is
`max(requested, sys)`, so our block starts at whichever is further right. When
other programs hold more than the request, the handle sits left of the block's
real edge — the meter showing why the handle has stopped binding. The split
handle sits in the free gap, where the two engines actually meet.

### A trap: text layouts paint their own background

`TextLayoutWidget.defaults` sets `.background = true` with `.style = .content`,
so a prose layout left alone paints a CANVAS-COLOURED block, 6px padding and
all, over whatever it sits in. Inside a chat bubble that covers the bubble's
fill completely, which is why the message colours looked wrong long after the
palette was right. `markdown_view.Opts.prose` now defaults to
`.background = false`; a caller that wants one (the thought block) says so.

### Sidebar (§6)

Deleting is a ✕ revealed on hover, ARMED by the first click and confirmed by the
second (the same two-step the VRAM meter's unload button uses), plus right-click
for a pointer-free route. Right-click alone was the original design and it is
undiscoverable; an unguarded ✕ is one stray click away from losing a
conversation, so it gets both.

Two event-ordering traps live in that one row. It is a plain box rather than a
`ButtonWidget`, because a button processes its events BEFORE its children exist
and would swallow the click meant for the ✕ inside it. And its title is drawn by
`fonts.richLine`, which builds its `TextLayoutWidget` by hand purely to SKIP
`processEvents`: a text layout claims the mouse for text selection, so the title
ate every click that landed on the words and you had to aim at the gap between
the text and the row's edge. The row's own click is therefore tested after the children have had
their turn, and hover comes from the previous frame (the background is painted
at init, before hover can be known) — invisible at frame rate.

The list is ordered by `updated_ms`, newest first, and the conversation you are
reading is moved to the front on open — IN MEMORY. Persisting that would be a
write on open, which is exactly what broke the ordering once already: `force` on
the save path was bypassing the change check as well as the busy check, so
switching conversations rewrote the OUTGOING one with a fresh timestamp and
browsing reshuffled the whole list. Reading must not write.

A real conversation list, backed by `history.zig`: length-prefixed transcripts
in `<config dir>/conversations`, grouped TODAY / YESTERDAY / EARLIER by local
calendar day, autosaved after each completed turn, right-click to delete.
Attachments and generated images are NOT stored — reopening a chat gives the
words back, not the pictures.

Each assistant turn also stores an **image record per saved render** (path,
size, steps, seed). Reopening a conversation rebuilds those as finished
`GenImage`s decoded from disk and hands them to the engine, so they appear in
the card, in Library and in the viewer like anything else — and it sets
`Variant.images_scanned`, which is what stops the last assistant turn's tool
calls being re-dispatched and the images being GENERATED AGAIN on every reload.

An image record stores only where the file is, not how it was made. **The PNG is
the record**: it already carries its own AUTOMATIC1111 `parameters` block, so
"Open in Studio" on a reopened conversation reads the file
(`image.pngText` + `diffuser.parseA1111Params` → `image_view.loadFromFile`)
rather than the transcript keeping a second copy that could disagree with the
image it describes. It also means images saved before any of this restore
correctly, and images made by ComfyUI or A1111 would too.

On load a render is matched to one already in the engine's list by its file path
(`Diffuser.findSaved`) and reused; building a second one appended a duplicate to
Library on every open.

Only images that were actually written are recorded; saving is optional and this
respects that setting rather than writing behind the user's back. A run whose
files are gone (or were never saved) still shows its `Tool call` section, with a
line saying which of the two it is. Nothing is ever silently re-rendered.

The transcript keeps the raw `<image>` markup either way, so `replayTranscript`
rebuilds the model's context byte-identically and continuing an old conversation
is unaffected.

Each assistant turn also stores **the reasoning markers it was generated with**
and the model that produced it. Markers are a property of the model that wrote
the text, not of whatever is loaded now, so splitting a stored reply with the
live model's markers is wrong across a model swap and impossible with nothing
loaded — which is how a reopened conversation showed a bare `</think>` in its
prose. The strings are stored rather than the family they came from: a family is
a key into a table we still edit, and a fine-tune can emit markers its base
family does not.

Turns with no recorded markers (anything saved before this) fall back to the live
model's, and with none loaded are NOT split at all. Guessing would silently eat
or invent part of a reply, which is worse than showing the markers.

**Image outcomes reach the model** (`image_tool_result`, on by default). When a
render finishes, fails or is canceled, a short note is queued and flushed into
the transcript at the next turn boundary, as its own message rather than glued
onto the user's — putting words in the user's mouth in their own transcript is
not a thing to do for the sake of a status line. It is shown as a centred quiet
line, not a bubble, and stored with a `note` role token.

Flushed at a turn boundary rather than the instant the image lands, because an
image can finish mid-generation and splicing a message in then would race the
worker and invalidate the KV it is decoding against. The consequence is that a
note reaches the model when the user next speaks; making it reach an idle
conversation would mean the assistant replying unprompted, which is a different
feature. Off restores fire-and-forget exactly.

The `<image>` tool needs no equivalent: it is a HARNESS convention (prose in the
system prompt plus a fixed parser), not the model's own tool format, so it cannot
drift. Native tool calling is the opposite — template-rendered and probe-measured
in `chat_template.zig` — but tool turns are not stored in a transcript, so the
question does not arise there yet.

Each assistant turn also stores whether its prompt had the reasoning block
already OPEN (`assistant_primed` in place of `assistant`). Without it a reloaded
primed reply parses as having no thought at all and the entire reasoning block
spills into the answer as prose. Old files, which have no such token, read as
not primed.
