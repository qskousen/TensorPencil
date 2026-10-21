//! Runtime low-rank sidecars (LoRA), applied beside a GEMM instead of merged
//! into the weight.
//!
//! `y = W x + s B (A x)`, with `s = strength * alpha / rank`. The base weight is
//! never touched, which is the whole point here: the shipping MiniMax H3 trunk is
//! int8 convrot, so merging would mean dequantizing 20 GB, adding the delta, and
//! requantizing, and the result would only be reproducible if the exact original
//! quantizer bake were. A sidecar has no round trip, and `strength` stays a
//! runtime dial.
//!
//! Three things here are silent wrong answers:
//!
//! 1. **`s = alpha / A.rows`, not `alpha / training_rank`.** ComfyUI's
//!    `LoRAAdapter.calculate_weight` divides by `mat2.shape[0]`, the rows of the
//!    `lora_A` tensor as it sits in the file. A fused factor whose A is three
//!    rank-128 blocks concatenated has 384 rows and an alpha that was multiplied
//!    by 3 to match, so both readings give the same number only if you use the
//!    file's own shape. Deriving it before any splitting is what makes that
//!    impossible to get wrong, and it is why the scale sits on the `Target` and
//!    not on each `Factor`.
//! 2. **`B` may be block diagonal.** A LoRA over a fused `qkv_proj` is three
//!    independent factors stacked: `A [3r, in]` concatenated and `B [3 out, 3r]`
//!    block diagonal. Treating it as one dense rank-`3r` factor is numerically
//!    identical and costs 3x the bytes and 3x the FLOPs *of the B GEMM* (the A
//!    GEMM is the same work either way, so H3's fused qkv comes out ~2.1x
//!    cheaper overall). `Target.load` tests for the structure on the actual data
//!    (exact zero off the diagonal, early-out on the first nonzero) rather than
//!    trusting a name, so a dense LoRA pays one comparison and a block-diagonal
//!    one pays a read it was going to do anyway.
//! 3. **A sidecar must gate every GEMM call site.** A fast path that skips it
//!    produces a finite, plausible, wrong image with no error anywhere. `Lin` in
//!    `minimax_h3.zig` is how that is made structural: the weight is reachable
//!    only as `.w`, so the sidecar is beside it in every grep.
//!
//! A base tensor `blocks.3.attn.qkv_proj.weight` is patched by
//! `diffusion_model.blocks.3.attn.qkv_proj.<factors>`, and `Sidecar.forWeight`
//! does that name transform, so a model loader passes the `Weight.tag` it
//! already has. A bundled checkpoint's tag carries `model.diffusion_model.`,
//! which `baseName` strips: ComfyUI holds that same tensor as
//! `diffusion_model.X`, which is what the file's keys are written against.
//!
//! Some families' LoRAs name a tensor the checkpoint does not have: Z-Image's
//! are written against diffusers' split `to_q/to_k/to_v`, which ComfyUI's
//! `z_image_to_diffusers` maps onto row blocks of the fused `attention.qkv`.
//! `Rename` is that map, passed per family to `Stack.add`. A slotted rename
//! merges its keys into one target with a factor per slot, each at its own
//! `out_off` and with its own `gain` (the slot's alpha / rank), so nothing is
//! copied and a device arm that runs q, k and v as three row views
//! (`Weight.row_off`) applies each slot's factor to its own view.
//!
//! `dialects` is the spellings those factors come in, from
//! `comfy/weight_adapter/lora.py`. Only the suffixes differ; the algebra does
//! not. ⚠️ **The reference's own `A_name` holds `lora_B`**, inverted from PEFT's
//! naming, so reading comfy's variable names rather than its shapes swaps the
//! two. What is fixed is the shape: `a` is `[rank, in_dim]` (`lora_A`,
//! `lora_down`) and `b` is `[out_dim, rank]` (`lora_B`, `lora_up`), and comfy
//! divides alpha by `mat2.shape[0]`, which is `a`'s rows.
//!
//! Three sibling keys mean a file this loader must REFUSE rather than read as a
//! plain LoRA: `lora_mid` (the conv CP form, three matrices), `dora_scale` (DoRA
//! rescales the base weight, so the low-rank part alone is the wrong magnitude)
//! and `reshape_weight`. Each is a finite, plausible, wrong render otherwise.
//!
//! A bf16 factor whose rank is not a multiple of `rank_align` is zero-padded at
//! load (zero rows of A, zero columns of B), which adds exactly nothing and is
//! what lets a device GEMM tile a rank-8 file. `Target.scale` is read from the
//! file's rank before that, so the padding cannot move it; everything after
//! load, host apply included, sees only the padded shape.
//!
//! `strength` is NOT folded into a factor. `Target.scale` holds the file's own
//! `alpha / rank` and the dial is multiplied at apply time, so a GUI slider does
//! not mean re-reading a gigabyte. `Stack` is N files over one model, each with
//! its own live dial; their deltas add, so order does not matter and stacking is
//! a merged name index.

const std = @import("std");
const ops = @import("tp_ops");
const weights_mod = @import("tp_core").weights;

/// This file's source hashed, for a cache of what a LoRA file was taken for: the
/// catalog mixes it into its rules fingerprint, so an edit here re-scans.
pub const source_fingerprint: u64 = blk: {
    @setEvalBranchQuota(1 << 24);
    break :blk std.hash.Wyhash.hash(0, @embedFile("lora.zig"));
};

const Weight = ops.matmul.Weight;
const WeightStore = weights_mod.WeightStore;
const adapter = @import("lora_adapter.zig");
const DType = @import("tp_core").dtype.DType;

/// DIAGNOSTIC: round the host apply's activation to bf16 before each GEMM, the
/// way the device apply's `opGemmBf16` does.
///
/// This is the control row for a whole-render device-vs-host comparison. The
/// device sidecar sits at bf16's own 1.66e-3 per GEMM (`lora-cuda-test` prints
/// it beside the residual), and without this there is no way to tell that floor
/// amplified by a trajectory from a defect in the apply.
pub var host_bf16_act: bool = false;

/// Rows the host apply processes at a time. Bounds the `[rows][out]`
/// intermediate: at the default H3 render an untiled one would be 38k x 21504 x
/// 4 B = 3.3 GB for a delta. Rows are independent, so this is free.
const host_band: usize = 512;

/// How one file spells a factor pair, `a` first. Tried in this order, which is
/// the reference's, so a file carrying two spellings resolves the way ComfyUI
/// resolves it.
pub const Dialect = struct {
    /// `[rank][in_dim]`, comfy's `mat2`.
    a: []const u8,
    /// `[out_dim][rank]`, comfy's `mat1`.
    b: []const u8,
};

pub const dialects = [_]Dialect{
    .{ .a = ".lora_down.weight", .b = ".lora_up.weight" }, // kohya, and civitai at large
    .{ .a = "_lora.down.weight", .b = "_lora.up.weight" }, // diffusers
    .{ .a = ".lora_A.weight", .b = ".lora_B.weight" }, // PEFT
    .{ .a = ".lora.down.weight", .b = ".lora.up.weight" },
    .{ .a = ".lora_A", .b = ".lora_B" }, // mochi
    .{ .a = ".lora_linear_layer.down.weight", .b = ".lora_linear_layer.up.weight" },
    .{ .a = ".lora_A.default.weight", .b = ".lora_B.default.weight" },
};

/// Suffixes that make a file something other than a plain LoRA. Reading one as a
/// plain LoRA renders finitely and wrongly, so `load` refuses instead.
const refuse = [_][]const u8{".reshape_weight"};

/// One rank-`rank` factor covering output rows `[out_off, out_off + out_rows)`
/// of a logical linear.
///
/// A plain LoRA is one factor at offset 0 spanning the whole output. A fused one
/// is several, which is what turns a block-diagonal `B` into independent work.
pub const Factor = struct {
    /// `[rank][in_dim]`.
    a: Weight,
    /// `[out_rows][rank]`.
    b: Weight,
    out_off: usize,
    /// Multiplied onto `Target.scale`. 1 except in a slotted merge, where each
    /// slot came from its own key with its own alpha.
    gain: f32 = 1,

    /// The padded rank for a factor `Sidecar.load` built.
    pub fn rank(f: Factor) usize {
        return f.a.rows;
    }
};

/// A LoKr term: `delta = kron(w1, w2)` over output rows `[out_off, out_off + ol * ok)`
/// and the weight's input (a conv's `[in][kh][kw]`), so output row `i * ok + k` and
/// input channel `j * in_n + n` take `w1[i][j] * w2[k][n]`. It runs as `w2` over each
/// of the `im` input channel groups and then `w1` across the groups, so nothing
/// weight-sized is ever built.
pub const Kron = struct {
    /// `[ol][im]`, on the host: it is small.
    w1: []const f32,
    ol: usize,
    im: usize,
    /// `[ok][in_n]`, or a conv's `[ok][in_n][kh][kw]` in torch order. bf16 for a
    /// linear, f16 for a conv, as a LoCon's `A` is.
    w2: Weight,
    out_off: usize = 0,
    /// Multiplied onto `Target.scale`.
    gain: f32 = 1,

    pub fn outRows(k: Kron) usize {
        return k.ol * k.w2.rows;
    }
};

/// A whole-weight difference over output rows `[out_off, out_off + w.rows)` in the
/// base's layout (a conv's `[out][in][kh][kw]`): `weight += strength * diff`, with no
/// `Target.scale`.
pub const Diff = struct {
    w: Weight,
    out_off: usize = 0,
};

/// DoRA (`weight_decompose`) over the target's factors, which carry its ratio already
/// (folded into `B`'s rows on the output axis, into `A`'s columns on the input one).
/// What is left at apply is the base's own share of the rescale: per output row
/// `y += s * (c - 1) * (y - bias)`, per input channel `y += s * W ((d - 1) x)`, a
/// second base product.
pub const Dora = struct {
    on_out: bool,
    /// `c` per output row or `d` per input channel: magnitude over (norm + eps).
    ratio: []const f32,
};

/// A DoRA stem waiting for its base weight (`Target.pending`).
pub const Pending = struct { patch: adapter.Patch, shape: [2]usize };

/// The sidecar for one linear.
pub const Target = struct {
    factors: []const Factor,
    kron: []const Kron = &.{},
    diffs: []const Diff = &.{},
    /// Whole targets only (`Stack.resolve` refuses a row view of one).
    dora: ?Dora = null,
    /// A DoRA stem whose ratio needs the base weight, which a loader with no model
    /// first sees at attach: `Stack.resolve` builds its factor and `dora` there.
    pending: ?Pending = null,
    in_dim: usize,
    out_dim: usize,
    /// `alpha / a.rows`, from the file's own shapes. Derived once before any
    /// block-diagonal split, which is what makes a fused alpha read right, so it
    /// belongs to the target and not to each factor. The runtime dial is
    /// multiplied on top at apply time.
    scale: f32,
    /// The base tensor's name, for diagnostics.
    tag: []const u8,
    /// File keys this target was built from: 1, or the slots a rename merged.
    keys: u8 = 1,

    /// The factor whose output rows contain `[row0, row0 + n)`, or null when the
    /// range spans two.
    pub fn factorAt(t: *const Target, row0: usize, n: usize) ?Factor {
        for (t.factors) |f| {
            if (row0 >= f.out_off and row0 + n <= f.out_off + f.b.rows) return f;
        }
        return null;
    }

    /// FLOPs one apply costs over `m` rows, for reporting the overhead honestly.
    pub fn flops(t: *const Target, m: usize) usize {
        var n: usize = 0;
        for (t.factors) |f| n += 2 * m * f.rank() * (t.in_dim + f.b.rows);
        return n;
    }

    /// `applyHost` over output rows `[row0, row0 + n)` only, into a `y` whose
    /// column 0 is row `row0`. A factor must lie wholly inside or outside the
    /// window, which `Stack.resolve` checked at attach.
    pub fn applyHostRows(t: *const Target, io: std.Io, gpa: std.mem.Allocator, y: []f32, y_stride: usize, x: []const f32, m: usize, strength: f32, row0: usize, n: usize) !void {
        var fb: [max_slots]Factor = undefined;
        var kb: [max_slots]Kron = undefined;
        var db: [max_slots]Diff = undefined;
        var view = t.*;
        view.factors = rowsOf(Factor, &fb, t.factors, row0, n);
        view.kron = rowsOf(Kron, &kb, t.kron, row0, n);
        view.diffs = rowsOf(Diff, &db, t.diffs, row0, n);
        view.out_dim = n;
        return view.applyHost(io, gpa, y, y_stride, x, m, strength);
    }

    /// The terms of `all` inside rows `[row0, row0 + n)`, moved to that origin. A term
    /// must lie wholly inside or outside, which `Stack.resolve` checked at attach.
    fn rowsOf(comptime T: type, buf: *[max_slots]T, all: []const T, row0: usize, n: usize) []const T {
        var k: usize = 0;
        for (all) |e| {
            const rows = termRows(e);
            if (e.out_off >= row0 + n or e.out_off + rows <= row0) continue;
            std.debug.assert(e.out_off >= row0 and e.out_off + rows <= row0 + n);
            buf[k] = e;
            buf[k].out_off -= row0;
            k += 1;
        }
        return buf[0..k];
    }

    /// Whether some term covers exactly rows inside `[row0, row0 + n)`, and none
    /// straddles them: a row view applies only what lies wholly in its rows.
    pub fn rowsFit(t: *const Target, row0: usize, n: usize) bool {
        inline for (.{ t.factors, t.kron, t.diffs }) |all| for (all) |e| {
            const r = termRows(e);
            const inside = e.out_off >= row0 and e.out_off + r <= row0 + n;
            const outside = e.out_off >= row0 + n or e.out_off + r <= row0;
            if (!inside and !outside) return false;
        };
        return true;
    }

    /// The bytes every term holds, for residency and the eviction walk.
    pub fn termBytes(t: *const Target) usize {
        var n: usize = 0;
        for (t.factors) |f| n += f.a.bytes.len + f.b.bytes.len;
        for (t.kron) |k| n += k.w2.bytes.len + k.w1.len * 4;
        for (t.diffs) |d| n += d.w.bytes.len;
        if (t.dora) |d| n += d.ratio.len * 4;
        return n;
    }

    /// `y[m][y_stride] += strength * scale * B (A x)`, factor `f` writing columns
    /// `[f.out_off, f.out_off + f.b.rows)`.
    ///
    /// `y_stride` is the destination row stride, which is NOT `out_dim` when the
    /// caller holds a fused output: the CPU forward's qkv buffer is
    /// `[seq][3 * inner]` and each factor lands in its own third of every row.
    pub fn applyHost(
        t: *const Target,
        io: std.Io,
        gpa: std.mem.Allocator,
        y: []f32,
        y_stride: usize,
        x: []const f32,
        m: usize,
        strength: f32,
    ) !void {
        std.debug.assert(x.len >= m * t.in_dim);
        std.debug.assert(y.len >= m * y_stride);

        var max_rank: usize = 0;
        var max_out: usize = 0;
        for (t.factors) |f| {
            max_rank = @max(max_rank, f.rank());
            max_out = @max(max_out, f.b.rows);
        }
        const band = @min(host_band, m);
        const lo = try gpa.alloc(f32, band * max_rank);
        defer gpa.free(lo);
        const hi = try gpa.alloc(f32, band * max_out);
        defer gpa.free(hi);

        // The bf16 control row, when asked for: one scratch band of the
        // activation, rounded, standing in for `x`.
        const xr: ?[]f32 = if (host_bf16_act) try gpa.alloc(f32, band * t.in_dim) else null;
        defer if (xr) |b| gpa.free(b);
        const lor: ?[]f32 = if (host_bf16_act) try gpa.alloc(f32, band * max_rank) else null;
        defer if (lor) |b| gpa.free(b);

        const s = t.scale * strength;
        var r0: usize = 0;
        while (r0 < m) : (r0 += band) {
            const n = @min(band, m - r0);
            var xin = x[r0 * t.in_dim ..][0 .. n * t.in_dim];
            if (xr) |b| {
                roundBf16(b[0 .. n * t.in_dim], xin);
                xin = b[0 .. n * t.in_dim];
            }
            for (t.factors) |f| {
                const r = f.rank();
                const o = f.b.rows;
                try ops.matmul.matmul(io, gpa, lo[0 .. n * r], xin, n, f.a, null);
                if (lor) |b| {
                    roundBf16(b[0 .. n * r], lo[0 .. n * r]);
                    @memcpy(lo[0 .. n * r], b[0 .. n * r]);
                }
                try ops.matmul.matmul(io, gpa, hi[0 .. n * o], lo[0 .. n * r], n, f.b, null);
                const sf = s * f.gain;
                for (0..n) |i| {
                    const dst = y[(r0 + i) * y_stride + f.out_off ..][0..o];
                    const src = hi[i * o ..][0..o];
                    for (dst, src) |*d, v| d.* += sf * v;
                }
            }
            for (t.kron) |k| try kronHost(io, gpa, y[r0 * y_stride ..], y_stride, xin, n, k, s * k.gain);
            for (t.diffs) |d| {
                const o = d.w.rows;
                const dh = try gpa.alloc(f32, n * o);
                defer gpa.free(dh);
                try ops.matmul.matmul(io, gpa, dh, xin, n, d.w, null);
                for (0..n) |i| {
                    const dst = y[(r0 + i) * y_stride + d.out_off ..][0..o];
                    for (dst, dh[i * o ..][0..o]) |*e, v| e.* += strength * v;
                }
            }
        }
    }
};

fn termRows(e: anytype) usize {
    return switch (@TypeOf(e)) {
        Factor => e.b.rows,
        Kron => e.outRows(),
        Diff => e.w.rows,
        else => @compileError("not a sidecar term"),
    };
}

/// `y[m][stride] += g * kron(w1, w2) x` for a linear, at the term's rows. `x` is
/// `[m][im * in_n]`, which read as `[m * im][in_n]` is the w2 GEMM's own input.
fn kronHost(io: std.Io, gpa: std.mem.Allocator, y: []f32, y_stride: usize, x: []const f32, m: usize, k: Kron, g: f32) !void {
    const ok = k.w2.rows;
    const h = try gpa.alloc(f32, m * k.im * ok);
    defer gpa.free(h);
    try ops.matmul.matmul(io, gpa, h, x, m * k.im, k.w2, null);
    kronMix(y, y_stride, h, m, k, g);
}

/// `y[p][out_off + i * ok + q] += g * sum_j w1[i][j] * h[p * im + j][q]`: the cross-group
/// half of a LoKr term, `h` being `w2`'s output per group.
fn kronMix(y: []f32, y_stride: usize, h: []const f32, rows_n: usize, k: Kron, g: f32) void {
    const ok = k.w2.rows;
    for (0..rows_n) |p| for (0..k.ol) |i| {
        const dst = y[p * y_stride + k.out_off + i * ok ..][0..ok];
        for (0..k.im) |j| {
            const a = g * k.w1[i * k.im + j];
            for (dst, h[(p * k.im + j) * ok ..][0..ok]) |*d, v| d.* += a * v;
        }
    };
}

/// `y[p][o] += s * (c[o] - 1) * (y[p][o] - bias[o])`: an output-axis DoRA's rescale of
/// the base product, which must run before the target's own delta lands in `y`.
pub fn doraRows(y: []f32, y_stride: usize, m: usize, c: []const f32, bias: ?[]const f32, s: f32) void {
    for (0..m) |p| for (y[p * y_stride ..][0..c.len], c, 0..) |*e, r, o| {
        const b = if (bias) |bb| bb[o] else 0;
        e.* += s * (r - 1) * (e.* - b);
    };
}

/// `xd[p][j] = (d[j] - 1) * x[p][j]`: an input-axis DoRA's share of the base product.
pub fn doraCols(xd: []f32, x: []const f32, m: usize, d: []const f32) void {
    for (0..m) |p| for (xd[p * d.len ..][0..d.len], x[p * d.len ..][0..d.len], d) |*o, v, r| {
        o.* = (r - 1) * v;
    };
}

/// `dst[p][off + o] += g * src[p][o]` over `n` pixels of `rows` channels.
fn addAt(dst: []f32, stride: usize, src: []const f32, n: usize, rows_n: usize, off: usize, g: f32) void {
    for (0..n) |p| for (dst[p * stride + off ..][0..rows_n], src[p * rows_n ..][0..rows_n]) |*e, v| {
        e.* += g * v;
    };
}

/// One loaded LoRA file: the name index plus the arena holding the repacked
/// block-diagonal factors. The mapping it was read from must outlive it, since
/// the dense factors are views into it.
pub const Sidecar = struct {
    arena: std.heap.ArenaAllocator,
    /// Base tensor name (without the `diffusion_model.` prefix) -> its target.
    index: std.StringHashMapUnmanaged(Target),
    /// Denoiser-prefixed keys whose SPELLING this loader does not know. A file
    /// made entirely of those loads as an empty sidecar, which would otherwise
    /// render as if no LoRA had been asked for. A LoRA for a different
    /// architecture is a separate case, caught by nothing attaching.
    unclaimed: usize,
    /// Bytes the rank padding added on top of the file's own factors.
    pad_bytes: usize = 0,

    pub fn deinit(self: *Sidecar) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// The prefix ComfyUI's generic format puts on a denoiser tensor name.
    pub const prefix = "diffusion_model.";

    /// Load every factor pair in `store`, in any dialect, keyed by the base
    /// tensor name it patches. `keys` is the family's key map, empty for the
    /// families whose LoRAs use the checkpoint's own names under `diffusion_model.`.
    ///
    /// The runtime dial is not taken here: see `Stack.File.strength`.
    pub fn load(gpa: std.mem.Allocator, store: WeightStore, keys: KeyMap) !Sidecar {
        const renames = keys.renames;
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();

        var index: std.StringHashMapUnmanaged(Target) = .empty;
        // Renamed keys wait here so a collision with a native spelling of the
        // same tensor is caught whatever order the file lists them in.
        var renamed: std.array_hash_map.String([max_slots]?Target) = .empty;
        var slots_of: std.StringHashMapUnmanaged(u8) = .empty;

        var unclaimed: usize = 0;
        var pad_bytes: usize = 0;
        for (store.names()) |full| {
            // Only denoiser keys are this loader's business; anything else in
            // the file belongs to another component (a text-encoder LoRA) and is
            // not evidence of a problem.
            const pfx = keys.prefixOf(full) orelse continue;

            if (refusedSuffix(full)) |sfx| {
                std.log.err("lora: {s} is a {s} adapter, which is a different calculation and not a plain LoRA", .{ full, sfx[1..] });
                return error.UnsupportedLora;
            }

            // A stem is reached through one key alone: a plain pair's `a`, or the key
            // that names a LoKr, LoHa or diff; every sibling is claimed without being
            // iterated to. The dialect a pair loads under is the FIRST that matches.
            var stem: []const u8 = undefined;
            var t: Target = undefined;
            if (for (adapter.stem_suffixes) |sfx| {
                if (std.mem.endsWith(u8, full, sfx)) break sfx;
            } else null) |sfx| {
                stem = full[pfx.len .. full.len - sfx.len];
                t = try adapterTarget(gpa, alloc, store, full[0 .. full.len - sfx.len], stem);
            } else {
                const d: Dialect = for (dialects) |cand| {
                    if (std.mem.endsWith(u8, full, cand.a)) break cand;
                } else {
                    if (!claimedSuffix(full)) unclaimed += 1;
                    continue;
                };
                const full_stem = full[0 .. full.len - d.a.len];
                stem = full[pfx.len .. full.len - d.a.len];
                // A pair with a Tucker core or a DoRA magnitude is not a plain pair.
                var kb: [512]u8 = undefined;
                const odd = for ([_][]const u8{ ".lora_mid.weight", ".dora_scale" }) |sib| {
                    if (store.get(std.mem.print(&kb, "{s}{s}", .{ full_stem, sib }) catch break false) != null) break true;
                } else false;
                t = if (odd) try adapterTarget(gpa, alloc, store, full_stem, stem) else try loadTarget(alloc, store, full_stem, stem, d, &pad_bytes);
            }
            if (renameFor(renames, stem)) |r| {
                const base = try alloc.print("{s}{s}.weight", .{ stem[0 .. stem.len - r.from.len], r.to });
                const gop = try renamed.getOrPut(alloc, base);
                if (!gop.found_existing) {
                    gop.value_ptr.* = @splat(null);
                    try slots_of.put(alloc, base, r.slots);
                }
                if (gop.value_ptr[r.slot] != null) return dupTarget(base);
                gop.value_ptr[r.slot] = t;
            } else {
                const base = try alloc.print("{s}.weight", .{stem});
                try index.put(alloc, base, t);
            }
        }

        var it = renamed.iterator();
        while (it.next()) |e| {
            const base = e.key_ptr.*;
            const gop = try index.getOrPut(alloc, base);
            if (gop.found_existing) return dupTarget(base);
            gop.value_ptr.* = try mergeSlots(alloc, base, e.value_ptr.*[0..slots_of.get(base).?]);
        }

        return .{
            .arena = arena,
            .index = index,
            .unclaimed = unclaimed,
            .pad_bytes = pad_bytes,
        };
    }

    fn dupTarget(base: []const u8) error{DuplicateTarget} {
        std.log.err("lora: two keys in this file patch {s}", .{base});
        return error.DuplicateTarget;
    }

    /// File keys across every target, i.e. what the file itself names.
    pub fn keyCount(self: *const Sidecar) usize {
        var n: usize = 0;
        var it = self.index.valueIterator();
        while (it.next()) |t| n += t.keys;
        return n;
    }

    /// The base tensor stem `name` is the `a` factor of, or null when it is not
    /// one (a `b` key, an alpha, or another component's tensor).
    ///
    /// Keyed on the `a` side alone, which is what makes one pass over the names
    /// visit each target exactly once.
    pub fn factorStem(name: []const u8) ?[]const u8 {
        if (!std.mem.startsWith(u8, name, prefix)) return null;
        for (dialects) |d| {
            if (std.mem.endsWith(u8, name, d.a)) return name[prefix.len .. name.len - d.a.len];
        }
        return null;
    }

    /// The `refuse` suffix `name` carries, or null. Split from the reporting in
    /// `load` so a test can assert the decision without making every passing run
    /// print: `std.testing.log_level` has no level below `err`.
    pub fn refusedSuffix(name: []const u8) ?[]const u8 {
        for (refuse) |sfx| if (std.mem.endsWith(u8, name, sfx)) return sfx;
        return null;
    }

    /// Whether `name` is a key some dialect or adapter accounts for, so it is not
    /// evidence of a spelling this loader missed. `a` keys are handled by the caller.
    fn claimedSuffix(name: []const u8) bool {
        for (adapter.adapter_suffixes) |sfx| if (std.mem.endsWith(u8, name, sfx)) return true;
        for (dialects) |d| if (std.mem.endsWith(u8, name, d.b)) return true;
        return false;
    }

    /// What this LoRA has for one base weight.
    ///
    /// `mismatch` is separate from `none` on purpose. A LoRA trained against a
    /// different variant of the same architecture has all the right NAMES and
    /// the wrong widths; its factors would still multiply, into the wrong
    /// columns. Treating that as "no entry" leaves a trunk running with some of
    /// its sidecars applied, which is a plausible wrong render. The caller
    /// decides how loud to be, which is also what keeps this pure enough to
    /// assert on in a test.
    pub const Lookup = union(enum) {
        none,
        ok: *const Target,
        /// The named entry, whose shape disagrees with the base weight.
        mismatch: *const Target,
    };

    /// The target patching `w`, by its `tag`.
    ///
    /// Returns a pointer into `index`, so it stays valid as long as nothing is
    /// added, which nothing is after `load`.
    pub fn forWeight(self: *const Sidecar, w: Weight) Lookup {
        const tag = w.tag orelse return .none;
        const t = self.index.getPtr(baseName(tag)) orelse return .none;
        if (t.in_dim != w.cols or t.out_dim != w.rows) return .{ .mismatch = t };
        return .{ .ok = t };
    }

    /// How many targets this LoRA holds, for the load message.
    pub fn count(self: *const Sidecar) usize {
        return self.index.count();
    }

    /// Total factor bytes, i.e. what this costs resident.
    pub fn bytes(self: *const Sidecar) usize {
        var n: usize = 0;
        var it = self.index.valueIterator();
        while (it.next()) |t| n += t.termBytes();
        return n;
    }
};

/// N LoRA files over one model, each with its own live strength.
///
/// The deltas ADD, so order does not change the result; the hit list is kept in
/// the order the files were given anyway, so a float sum is the same every run.
pub const Stack = struct {
    files: std.ArrayList(File) = .empty,
    /// Base tensor name -> every file's target for it. Keys are borrowed from
    /// the sidecars' own indexes, so every file outlives this map.
    index: std.StringHashMapUnmanaged(std.ArrayList(Hit)) = .empty,
    /// Per-file hit counts for the attach walk in progress. Only `beginAttach`
    /// / `resolve` / `finishAttach` touch these.
    hits: [max_files]usize = @splat(0),
    /// Storage behind every `Hit.a_i8`.
    quant: std.ArrayList(QuantA) = .empty,

    pub const File = struct {
        side: Sidecar,
        /// The dial, read at every apply. Nothing folds it into a factor, so
        /// moving it costs nothing and no second copy can disagree with it.
        strength: f32,
        /// The file it came from, for diagnostics and the GUI row. Owned.
        path: []const u8,
    };

    pub const Hit = struct {
        target: *const Target,
        /// Index into `files`, so an apply reads the live strength.
        file: u32,
        /// The target's one factor's A as int8 (`quantizeA`), for a base whose int8
        /// activation prep the device sidecar reads instead of the f32 activation.
        /// Null when nothing asked for it. Its bytes live in `quant`.
        a_i8: ?Weight = null,
        /// Whether the attach walk in progress met a linear for this target.
        seen: bool = false,
    };

    pub fn deinit(self: *Stack, gpa: std.mem.Allocator) void {
        for (self.quant.items) |q| q.free(gpa);
        self.quant.deinit(gpa);
        var it = self.index.valueIterator();
        while (it.next()) |list| list.deinit(gpa);
        self.index.deinit(gpa);
        for (self.files.items) |*f| {
            f.side.deinit();
            gpa.free(f.path);
        }
        self.files.deinit(gpa);
        self.* = undefined;
    }

    /// Load one file and merge it in. `store` must outlive the stack: the dense
    /// factors are views into its mapping.
    ///
    /// Returns how many targets the file holds, which is not yet how many will
    /// apply; `check` answers that against a model.
    pub fn add(
        self: *Stack,
        gpa: std.mem.Allocator,
        store: WeightStore,
        strength: f32,
        path: []const u8,
        keys: KeyMap,
    ) !usize {
        if (self.files.items.len >= max_files) return error.TooManyLoras;
        var side = try Sidecar.load(gpa, store, keys);
        errdefer side.deinit();
        return self.addSidecar(gpa, side, strength, path);
    }

    /// `add` for a sidecar built elsewhere (the SD family's, from its own key
    /// map). Takes `side` over on success.
    pub fn addSidecar(self: *Stack, gpa: std.mem.Allocator, side: Sidecar, strength: f32, path: []const u8) !usize {
        if (self.files.items.len >= max_files) return error.TooManyLoras;
        const owned_path = try gpa.dupe(u8, path);
        errdefer gpa.free(owned_path);

        const idx: u32 = @intCast(self.files.items.len);
        try self.files.append(gpa, .{ .side = side, .strength = strength, .path = owned_path });
        // Past this point the file is owned by `files` and freed by `deinit`.
        errdefer _ = self.files.pop();

        const own = &self.files.items[idx].side;
        var it = own.index.iterator();
        while (it.next()) |e| {
            const gop = try self.index.getOrPut(gpa, e.key_ptr.*);
            if (!gop.found_existing) gop.value_ptr.* = .empty;
            try gop.value_ptr.append(gpa, .{ .target = e.value_ptr, .file = idx });
        }
        return own.count();
    }

    /// Every hit against `w`, empty when nothing patches it.
    ///
    /// Shape disagreements are not reported here. `check` has refused them
    /// already, so the forward path carries no branch for a case that cannot
    /// reach it.
    pub fn forWeight(self: *const Stack, w: Weight) []const Hit {
        const tag = w.tag orelse return &.{};
        const list = self.index.getPtr(baseName(tag)) orelse return &.{};
        return list.items;
    }

    /// Zero the per-file counters before an attach walk.
    ///
    /// The walk is the model's own, not a shared linear list, because only the
    /// model knows which linears it actually runs. Each family calls
    /// `beginAttach`, then `resolve` for every linear, then `finishAttach`.
    pub fn beginAttach(self: *Stack) void {
        self.hits = @splat(0);
        var it = self.index.valueIterator();
        while (it.next()) |list| for (list.items) |*h| {
            h.seen = false;
        };
    }

    /// The validating lookup: `forWeight` plus the width check, and it counts the
    /// hit against its file.
    ///
    /// A LoRA trained against another variant of the same architecture has all
    /// the right NAMES and the wrong widths; its factors would still multiply,
    /// into the wrong columns. Refusing beats skipping, since a trunk with only
    /// some of its sidecars applied renders plausibly and wrongly.
    ///
    /// A row view (`Weight.row_off`, or fewer rows than the target) takes the
    /// factor covering its rows, so it must sit inside one: a device arm asks the
    /// sidecar for its own row range and nothing else. The walk must also resolve
    /// the whole tensor, which is where the full shape is checked.
    pub fn resolve(self: *Stack, w: Weight) ![]const Hit {
        const tag = w.tag orelse return &.{};
        const list = self.index.getPtr(baseName(tag)) orelse return &.{};
        for (list.items) |*h| {
            const t = h.target;
            const whole = w.row_off == 0 and w.rows == t.out_dim;
            if (t.in_dim != w.cols or w.row_off + w.rows > t.out_dim) {
                std.log.err("lora: {s}: {s} is {d}x{d} but the checkpoint's is {d}x{d}", .{
                    self.files.items[h.file].path, t.tag, t.out_dim, t.in_dim, w.rows, w.cols,
                });
                return error.ShapeMismatch;
            }
            if (!whole and !t.rowsFit(w.row_off, w.rows)) {
                std.log.err("lora: {s}: {s} rows [{d},{d}) straddle two of its terms", .{
                    self.files.items[h.file].path, t.tag, w.row_off, w.row_off + w.rows,
                });
                return error.ShapeMismatch;
            }
            if (!whole and (t.dora != null or t.pending != null)) {
                std.log.err("lora: {s}: {s} is DoRA on a row view, whose norms span the whole tensor", .{ self.files.items[h.file].path, t.tag });
                return error.UnsupportedLora;
            }
            if (t.pending != null) try self.finishDora(h.file, tag, w);
            self.hits[h.file] += 1;
            h.seen = true;
        }
        return list.items;
    }

    /// Build a pending DoRA target's factor and ratio from its base weight `w`, once.
    fn finishDora(self: *Stack, file: usize, tag: []const u8, w: Weight) !void {
        const side = &self.files.items[file].side;
        const t = side.index.getPtr(baseName(tag)).?;
        const pd = t.pending orelse return;
        const gpa = std.heap.smp_allocator;
        const wf = ops.matmul.materializeF32(gpa, w) catch |err| {
            std.log.err("lora: {s} is DoRA over a {t} weight, whose norms this cannot read", .{ t.tag, w.dtype });
            return err;
        };
        defer if (wf.bytes.ptr != w.bytes.ptr) gpa.free(wf.bytes);
        const mode: adapter.Mode = if (w.dtype == .f32) .f32 else .f16;
        const vals = try gpa.dupe(f32, @as([*]const f32, @ptrCast(@alignCast(wf.bytes.ptr)))[0 .. w.rows * w.cols]);
        defer gpa.free(vals);
        if (mode == .f16) for (vals) |*x| {
            x.* = adapter.r16(x.*);
        };
        const term = try adapter.termFor(gpa, side.arena.allocator(), pd.patch, .{ .w = vals, .mode = mode }, &pd.shape, pd.shape[0], 0);
        t.factors = try side.arena.allocator().dupe(Factor, &.{term.factor.f});
        t.dora = term.factor.dora;
        t.pending = null;
    }

    /// `resolve` over a list, for a walk that does not keep the hits.
    pub fn resolveAll(self: *Stack, ws: []const Weight) !void {
        for (ws) |w| _ = try self.resolve(w);
    }

    /// Close an attach walk, refusing a file that patched nothing and warning,
    /// by name, about every target that matched no linear.
    ///
    /// Returns how many of the files' keys will apply. A file with no hits is a
    /// LoRA for another architecture and is refused. A file that hits SOME
    /// linears is applied, as ComfyUI does ("lora key not loaded"), but the
    /// rest would otherwise render as a plausible wrong image, so each miss and
    /// a count go to the log.
    pub fn finishAttach(self: *const Stack, gpa: std.mem.Allocator) !usize {
        if (self.fileWithNoHits()) |i| {
            const f = self.files.items[i];
            std.log.err("lora: {s} patches none of this checkpoint's linears ({d} targets in the file)", .{ f.path, f.side.count() });
            return error.NoMatchingTargets;
        }
        var applied: usize = 0;
        for (self.files.items, 0..) |f, i| {
            if (f.side.unclaimed > 0) {
                std.log.warn("lora: {d} keys in {s} use a spelling this loader does not know", .{ f.side.unclaimed, f.path });
            }
            const c = self.coverage(i);
            applied += c.applied;
            if (c.applied == c.total) continue;
            const names = try self.missed(gpa, i);
            defer gpa.free(names);
            for (names) |tag| std.log.warn("lora: {s}: {s} matches no linear this model runs; not applied", .{ f.path, tag });
            std.log.warn("lora: {s}: ONLY {d} of {d} targets applied, {d} ignored", .{ f.path, c.applied, c.total, c.total - c.applied });
        }
        return applied;
    }

    pub const Coverage = struct { applied: usize, total: usize };

    /// File keys of `file` the last walk met a linear for, against all of them.
    pub fn coverage(self: *const Stack, file: usize) Coverage {
        var c: Coverage = .{ .applied = 0, .total = 0 };
        var it = self.index.valueIterator();
        while (it.next()) |list| for (list.items) |h| {
            if (h.file != file) continue;
            c.total += h.target.keys;
            if (h.seen) c.applied += h.target.keys;
        };
        return c;
    }

    /// Tags of `file`'s targets the last walk met no linear for, sorted. Caller
    /// frees the slice; the names belong to the stack.
    pub fn missed(self: *const Stack, gpa: std.mem.Allocator, file: usize) ![][]const u8 {
        var out: std.ArrayList([]const u8) = .empty;
        errdefer out.deinit(gpa);
        var it = self.index.valueIterator();
        while (it.next()) |list| for (list.items) |h| {
            if (h.file == file and !h.seen) try out.append(gpa, h.target.tag);
        };
        std.mem.sort([]const u8, out.items, {}, lessStr);
        return out.toOwnedSlice(gpa);
    }

    fn lessStr(_: void, a: []const u8, b: []const u8) bool {
        return std.mem.lessThan(u8, a, b);
    }

    /// The first file the walk found no linear for, or null. Split from the
    /// reporting for the same reason as `Sidecar.refusedSuffix`.
    pub fn fileWithNoHits(self: *const Stack) ?usize {
        for (self.files.items, 0..) |_, i| if (self.hits[i] == 0) return i;
        return null;
    }

    /// The host apply for every hit against `w`, in file order. `bias` is what the
    /// base GEMM added into `y`, which an output-axis DoRA must not rescale.
    pub fn applyHost(
        self: *const Stack,
        io: std.Io,
        gpa: std.mem.Allocator,
        y: []f32,
        y_stride: usize,
        x: []const f32,
        m: usize,
        w: Weight,
        bias: ?[]const f32,
    ) !void {
        for (self.forWeight(w)) |h| {
            const s = self.files.items[h.file].strength;
            // Exact: adding `0 * delta` changes nothing, so a dial at zero is
            // free rather than two GEMMs whose result is discarded.
            if (s == 0) continue;
            // A row view of a fused tensor (CLIP-G's q, k, v thirds) writes a
            // buffer of its own rows, so only its factors apply, at its origin.
            if (w.row_off != 0 or w.rows != h.target.out_dim) {
                try h.target.applyHostRows(io, gpa, y, y_stride, x, m, s, w.row_off, w.rows);
                continue;
            }
            if (h.target.pending != null) return error.UnsupportedLora;
            if (h.target.dora) |d| {
                if (d.on_out) {
                    doraRows(y, y_stride, m, d.ratio, bias, s);
                } else {
                    const xd = try gpa.alloc(f32, m * w.cols);
                    defer gpa.free(xd);
                    doraCols(xd, x, m, d.ratio);
                    const t = try gpa.alloc(f32, m * w.rows);
                    defer gpa.free(t);
                    try ops.matmul.matmul(io, gpa, t, xd, m, w, null);
                    for (0..m) |i| for (y[i * y_stride ..][0..w.rows], t[i * w.rows ..][0..w.rows]) |*e, v| {
                        e.* += s * v;
                    };
                }
            }
            try h.target.applyHost(io, gpa, y, y_stride, x, m, s);
        }
    }

    /// Every hit against a convolution, onto `dst` (channel-last `[oh*ow][co]`, `c`'s
    /// bias in it) from `src` (`[h*w][ci]`), the CPU twin of the device conv sidecars.
    /// A 1x1 is the linear apply over pixels. A 3x3 factor's `A` is itself a
    /// convolution to `rank` channels; a LoKr term's `w2` is a convolution over each
    /// input channel group (the grouped convolution its kron is); a diff is a whole
    /// extra convolution.
    pub fn applyHostConv(self: *const Stack, io: std.Io, gpa: std.mem.Allocator, dst: []f32, src: []const f32, h: usize, w: usize, c: ops.conv.Conv2d) !void {
        const cw = c.gemmWeight();
        const hits = self.forWeight(cw);
        if (hits.len == 0) return;
        if (c.k == 1 and c.stride == 1) return self.applyHost(io, gpa, dst, c.co, src, h * w, cw, c.b);
        const n = c.outH(h) * c.outW(w);
        const tmp = try gpa.alloc(f32, n * c.co);
        defer gpa.free(tmp);
        for (hits) |hit| {
            const s = self.files.items[hit.file].strength;
            if (s == 0) continue;
            const t = hit.target;
            if (t.dora) |d| {
                if (d.on_out) {
                    doraRows(dst, c.co, n, d.ratio, c.b, s);
                } else {
                    // W ((d - 1) x): the base convolution, bias-free, on a rescaled input.
                    const xd = try gpa.alloc(f32, h * w * c.ci);
                    defer gpa.free(xd);
                    doraCols(xd, src, h * w, d.ratio);
                    var base = c;
                    base.b = null;
                    try ops.conv.conv2d(io, gpa, tmp, xd, h, w, base);
                    for (dst[0 .. n * c.co], tmp) |*e, v| e.* += s * v;
                }
            }
            for (t.factors) |f| {
                const r = f.rank();
                const lo = try gpa.alloc(f32, n * r);
                defer gpa.free(lo);
                const a_conv: ops.conv.Conv2d = .{ .w = .{ .torch = .{ .w = f.a } }, .co = r, .ci = c.ci, .k = c.k, .stride = c.stride, .pad = c.pad };
                try ops.conv.conv2d(io, gpa, lo, src, h, w, a_conv);
                const o = f.b.rows;
                try ops.matmul.matmul(io, gpa, tmp[0 .. n * o], lo, n, f.b, null);
                addAt(dst, c.co, tmp, n, o, f.out_off, s * t.scale * f.gain);
            }
            for (t.kron) |k| {
                const in_n = c.ci / k.im;
                const ok = k.w2.rows;
                const xg = try gpa.alloc(f32, h * w * in_n);
                defer gpa.free(xg);
                const hg = try gpa.alloc(f32, n * ok);
                defer gpa.free(hg);
                // `[n][im][ok]`, the per-group outputs side by side, which is the
                // layout `kronMix` reads.
                const hk = try gpa.alloc(f32, n * k.im * ok);
                defer gpa.free(hk);
                const g_conv: ops.conv.Conv2d = .{ .w = .{ .torch = .{ .w = k.w2 } }, .co = ok, .ci = in_n, .k = c.k, .stride = c.stride, .pad = c.pad };
                for (0..k.im) |j| {
                    for (0..h * w) |px| @memcpy(xg[px * in_n ..][0..in_n], src[px * c.ci + j * in_n ..][0..in_n]);
                    try ops.conv.conv2d(io, gpa, hg, xg, h, w, g_conv);
                    for (0..n) |px| @memcpy(hk[(px * k.im + j) * ok ..][0..ok], hg[px * ok ..][0..ok]);
                }
                kronMix(dst, c.co, hk, n, k, s * t.scale * k.gain);
            }
            for (t.diffs) |d| {
                const d_conv: ops.conv.Conv2d = .{ .w = .{ .torch = .{ .w = d.w } }, .co = d.w.rows, .ci = c.ci, .k = c.k, .stride = c.stride, .pad = c.pad };
                try ops.conv.conv2d(io, gpa, tmp[0 .. n * d.w.rows], src, h, w, d_conv);
                addAt(dst, c.co, tmp, n, d.w.rows, d.out_off, s);
            }
        }
    }

    /// Add every hit's delta against `w` into `dst`, `w` itself as `[rows][cols]`
    /// f32, for a small weight a family applies by other means than a GEMM. Runs
    /// the host apply on an identity, so it is exactly what `applyHost` adds.
    pub fn addDelta(self: *const Stack, io: std.Io, gpa: std.mem.Allocator, w: Weight, dst: []f32) !void {
        if (self.forWeight(w).len == 0) return;
        // DoRA rescales the base product, which an identity apply never computes.
        for (self.forWeight(w)) |h| if (h.target.dora != null) {
            std.log.err("lora: {s} is DoRA, which this weight's apply cannot rescale", .{h.target.tag});
            return error.UnsupportedLora;
        };
        std.debug.assert(dst.len == w.rows * w.cols);
        const eye = try gpa.alloc(f32, w.cols * w.cols);
        defer gpa.free(eye);
        @memset(eye, 0);
        for (0..w.cols) |i| eye[i * w.cols + i] = 1;
        const yt = try gpa.alloc(f32, w.cols * w.rows);
        defer gpa.free(yt);
        @memset(yt, 0);
        try self.applyHost(io, gpa, yt, w.rows, eye, w.cols, w, null);
        for (0..w.rows) |o| for (0..w.cols) |i| {
            dst[o * w.cols + i] += yt[i * w.rows + o];
        };
    }

    /// Widest `[m][rank]` and `[m][out]` any single apply needs, for sizing one
    /// shared device scratch across the whole stack.
    pub fn maxFactor(self: *const Stack) struct { rank: usize, out: usize } {
        var rank: usize = 0;
        var out: usize = 0;
        for (self.files.items) |f| {
            var it = f.side.index.valueIterator();
            while (it.next()) |t| for (t.factors) |fa| {
                rank = @max(rank, fa.rank());
                out = @max(out, fa.b.rows);
            };
        }
        return .{ .rank = rank, .out = out };
    }

    /// Total factor bytes across every file, i.e. what the stack costs resident.
    pub fn bytes(self: *const Stack) usize {
        var n: usize = 0;
        for (self.files.items) |f| n += f.side.bytes();
        return n;
    }

    /// The part of `bytes` that is rank padding.
    pub fn padBytes(self: *const Stack) usize {
        var n: usize = 0;
        for (self.files.items) |f| n += f.side.pad_bytes;
        return n;
    }

    /// Visit every factor's raw bytes, the rank-padded copies included.
    ///
    /// ⚠️ The device backends key their weight cache on the HOST POINTER they
    /// were handed, and for a fused linear that is a SUB-SLICE of the B factor
    /// (`lora_cuda.applyRange` takes `f.b.bytes[b_off..]`), not the address
    /// yielded here. A stack about to be freed must have the whole RANGE of each
    /// of these evicted from that cache FIRST (`evictWeightRange`): the
    /// allocator hands the same addresses to the next stack, and the cache would
    /// then serve the OLD factors under the new file's name -- a finite,
    /// plausible, wrong image with no error anywhere.
    pub fn forEachFactorBytes(self: *const Stack, ctx: anytype, comptime visit: fn (@TypeOf(ctx), []const u8) void) void {
        for (self.files.items) |f| {
            var it = f.side.index.valueIterator();
            while (it.next()) |t| {
                for (t.factors) |fa| {
                    visit(ctx, fa.a.bytes);
                    visit(ctx, fa.b.bytes);
                }
                for (t.kron) |k| {
                    visit(ctx, k.w2.bytes);
                    visit(ctx, std.mem.sliceAsBytes(k.w1));
                }
                for (t.diffs) |d| visit(ctx, d.w.bytes);
                if (t.dora) |d| visit(ctx, std.mem.sliceAsBytes(d.ratio));
            }
        }
        // The row scale is uploaded by its own host pointer too.
        for (self.quant.items) |q| {
            visit(ctx, std.mem.sliceAsBytes(q.q));
            visit(ctx, std.mem.sliceAsBytes(q.scale));
        }
    }

    /// Give every single-factor hit against `w` an int8 A rotated iff `rotate`, so a
    /// device sidecar can run its A GEMM on the base's staged int8 activation.
    /// Idempotent; a hit already quantized the other way is requantized.
    pub fn quantizeHits(self: *Stack, gpa: std.mem.Allocator, w: Weight, rotate: bool) !void {
        const tag = w.tag orelse return;
        const list = self.index.getPtr(baseName(tag)) orelse return;
        const want: u32 = if (rotate) @intCast(ops.convrot.group_size) else 0;
        for (list.items) |*h| {
            if (h.target.factors.len != 1) continue;
            if (h.a_i8) |a| if (a.convrot == want) continue;
            try self.quant.ensureUnusedCapacity(gpa, 1);
            const q = try quantizeA(gpa, h.target.factors[0].a, rotate);
            self.quant.appendAssumeCapacity(q);
            h.a_i8 = q.weight(h.target.factors[0].a.cols, rotate);
        }
    }
};

/// `A [rank][in]` as int8 per row: rotated by the convrot group Hadamard iff the
/// activation prep it pairs with rotates, then `q = round(a / s)` with
/// `s = absmax / 127`, rows padded to 128 with zero rows of scale 0 (the int8 GEMM
/// launches `rows / 128` tiles). The file's `alpha / rank` is NOT folded in; it stays
/// on `Target.scale`.
pub const QuantA = struct {
    q: []i8,
    scale: []f32,

    pub fn free(self: QuantA, gpa: std.mem.Allocator) void {
        gpa.free(self.q);
        gpa.free(self.scale);
    }

    /// As a `Weight` the CPU matmul reads back to `A` and the device int8 GEMM takes.
    pub fn weight(self: QuantA, cols: usize, rotated: bool) Weight {
        var w = Weight.init(std.mem.sliceAsBytes(self.q), .i8, self.scale.len, cols);
        w.row_scale = self.scale;
        w.convrot = if (rotated) @intCast(ops.convrot.group_size) else 0;
        return w;
    }
};

pub fn quantizeA(gpa: std.mem.Allocator, a: Weight, rotate: bool) !QuantA {
    if (rotate and a.cols % ops.convrot.group_size != 0) return error.ShapeMismatch;
    const rows = std.mem.alignForward(usize, a.rows, 128);
    const q = try gpa.alloc(i8, rows * a.cols);
    errdefer gpa.free(q);
    const scale = try gpa.alloc(f32, rows);
    errdefer gpa.free(scale);
    @memset(q, 0);
    @memset(scale, 0);
    const row = try gpa.alloc(f32, a.cols);
    defer gpa.free(row);
    const es = a.dtype.storageBytes(a.cols);
    for (0..a.rows) |r| {
        try tp_core.safetensors.convertToF32(a.dtype, a.bytes[r * es ..][0..es], row);
        if (rotate) ops.convrot.rotate(row);
        var amax: f32 = 0;
        for (row) |v| amax = @max(amax, @abs(v));
        if (amax == 0) continue;
        const s = amax / 127.0;
        scale[r] = s;
        for (q[r * a.cols ..][0..a.cols], row) |*d, v| d.* = @intFromFloat(@round(v / s));
    }
    return .{ .q = q, .scale = scale };
}

/// Files one stack can hold. A dial per file is a GUI row, so the cap is what
/// fits on screen rather than anything the algebra needs.
pub const max_files: usize = 16;

/// One key spelling a family's LoRAs use for a tensor its checkpoint names
/// differently, matched on the end of the file's stem.
/// How a family's LoRA files name its tensors: ComfyUI's `diffusion_model.` prefix,
/// plus any the family's trainers write (`prefixes`, tried in order after it; "" is
/// diffusers' bare spelling), and the key renames onto fused tensors.
pub const KeyMap = struct {
    renames: []const Rename = &.{},
    prefixes: []const []const u8 = &.{},

    /// The prefix `full` is written under, or null when it is none of this map's.
    fn prefixOf(self: KeyMap, full: []const u8) ?[]const u8 {
        if (std.mem.startsWith(u8, full, Sidecar.prefix)) return Sidecar.prefix;
        for (self.prefixes) |p| if (std.mem.startsWith(u8, full, p)) return p;
        return null;
    }
};

pub const Rename = struct {
    /// As the LoRA spells it, e.g. `.attention.to_q`.
    from: []const u8,
    /// As the checkpoint does, e.g. `.attention.qkv`.
    to: []const u8,
    /// Which of `slots` equal row blocks of `to` this key patches. One slot is a
    /// plain rename.
    slot: u8 = 0,
    slots: u8 = 1,
};

const max_slots = 4;

/// The checkpoint prefixes a loader's tag may carry that a LoRA's names never do.
const checkpoint_prefixes = [_][]const u8{"model.diffusion_model."};

/// `tag` as a LoRA names it.
pub fn baseName(tag: []const u8) []const u8 {
    for (checkpoint_prefixes) |p| if (std.mem.startsWith(u8, tag, p)) return tag[p.len..];
    return tag;
}

fn renameFor(renames: []const Rename, stem: []const u8) ?Rename {
    for (renames) |r| if (std.mem.endsWith(u8, stem, r.from)) return r;
    return null;
}

/// One target over `slots` equal row blocks, each slot's factors moved to its
/// block and carrying that slot's scale as a gain. A missing slot is left out,
/// which is a zero delta on those rows.
fn mergeSlots(alloc: std.mem.Allocator, base: []const u8, slots: []const ?Target) !Target {
    var in_dim: usize = 0;
    var sub_out: usize = 0;
    var n: [3]usize = .{ 0, 0, 0 };
    var keys: u8 = 0;
    for (slots) |s| if (s) |t| {
        if ((in_dim != 0 and t.in_dim != in_dim) or (sub_out != 0 and t.out_dim != sub_out)) {
            std.log.err("lora: the keys merged into {s} disagree on shape", .{base});
            return error.ShapeMismatch;
        }
        if (t.pending != null) {
            std.log.err("lora: {s} is DoRA on one slot of a fused tensor, whose norms span the whole tensor", .{base});
            return error.UnsupportedLora;
        }
        in_dim = t.in_dim;
        sub_out = t.out_dim;
        n[0] += t.factors.len;
        n[1] += t.kron.len;
        n[2] += t.diffs.len;
        keys += 1;
    };
    const factors = try alloc.alloc(Factor, n[0]);
    const kron = try alloc.alloc(Kron, n[1]);
    const diffs = try alloc.alloc(Diff, n[2]);
    var i: [3]usize = .{ 0, 0, 0 };
    for (slots, 0..) |s, si| if (s) |t| {
        for (t.factors) |f| {
            factors[i[0]] = f;
            factors[i[0]].out_off = si * sub_out + f.out_off;
            factors[i[0]].gain = f.gain * t.scale;
            i[0] += 1;
        }
        for (t.kron) |k| {
            kron[i[1]] = k;
            kron[i[1]].out_off = si * sub_out + k.out_off;
            kron[i[1]].gain = k.gain * t.scale;
            i[1] += 1;
        }
        for (t.diffs) |d| {
            diffs[i[2]] = d;
            diffs[i[2]].out_off = si * sub_out + d.out_off;
            i[2] += 1;
        }
    };
    return .{
        .factors = factors,
        .kron = kron,
        .diffs = diffs,
        .in_dim = in_dim,
        .out_dim = slots.len * sub_out,
        .scale = 1,
        .tag = base[0 .. base.len - ".weight".len],
        .keys = keys,
    };
}

/// What a loaded bf16 factor's rank is padded to: the k slab of the device B
/// GEMM, which contracts over the rank.
pub const rank_align: usize = 32;

/// `f` with its rank zero-padded to `rank_align`, copied into `alloc`. A factor
/// already aligned, or not bf16, comes back as is.
pub fn padRank(alloc: std.mem.Allocator, f: Factor) !Factor {
    const r = f.rank();
    if (f.a.dtype != .bf16 or f.b.dtype != .bf16 or r % rank_align == 0) return f;
    const rp = std.mem.alignForward(usize, r, rank_align);
    const in_dim = f.a.cols;
    const out = f.b.rows;
    std.debug.assert(f.b.cols == r and f.a.bytes.len == r * in_dim * 2 and f.b.bytes.len == out * r * 2);

    // Word-aligned: the device kernels read these as 16- and 32-bit words.
    const a = try alloc.alignedAlloc(u8, .of(u32), rp * in_dim * 2);
    @memcpy(a[0 .. r * in_dim * 2], f.a.bytes);
    @memset(a[r * in_dim * 2 ..], 0);
    const b = try alloc.alignedAlloc(u8, .of(u32), out * rp * 2);
    @memset(b, 0);
    for (0..out) |o| @memcpy(b[o * rp * 2 ..][0 .. r * 2], f.b.bytes[o * r * 2 ..][0 .. r * 2]);
    return .{ .a = Weight.init(a, .bf16, rp, in_dim), .b = Weight.init(b, .bf16, out, rp), .out_off = f.out_off, .gain = f.gain };
}

/// A stem that is not a plain pair, read as ComfyUI reads it (`lora_adapter.parse`) and
/// made into a target with one term. A DoRA stem waits for its base weight at attach
/// (`Target.pending`). Linears only: a denoiser's adapters patch none of its convs.
fn adapterTarget(gpa: std.mem.Allocator, alloc: std.mem.Allocator, store: WeightStore, full_stem: []const u8, stem: []const u8) !Target {
    var consumed: adapter.Consumed = .empty;
    defer consumed.deinit(gpa);
    const p = (adapter.parse(gpa, store, full_stem, &consumed) catch |err| switch (err) {
        error.UnsupportedAdapter => {
            std.log.err("lora: {s} is an adapter form the sidecar does not apply (set_weight, or Tucker LoKr)", .{stem});
            return error.UnsupportedLora;
        },
        else => return err,
    }) orelse return error.MissingTensor;
    const sh = try adapter.patchShape(p);
    if (sh.len != 2) {
        std.log.err("lora: {s} patches a convolution, which a denoiser sidecar does not apply", .{stem});
        return error.UnsupportedLora;
    }
    const shape = [2]usize{ sh.dims[0], sh.dims[1] };
    var t: Target = .{ .factors = &.{}, .in_dim = shape[1], .out_dim = shape[0], .scale = 1, .tag = try alloc.dupe(u8, stem) };
    if (p.dora != null) {
        if (p.kind == .lokr) {
            std.log.err("lora: {s} is DoRA over LoKr, which the sidecar does not apply", .{stem});
            return error.UnsupportedLora;
        }
        t.pending = .{ .patch = p, .shape = shape };
        return t;
    }
    switch (try adapter.termFor(gpa, alloc, p, null, &shape, shape[0], 0)) {
        .factor => |f| t.factors = try alloc.dupe(Factor, &.{f.f}),
        .kron => |k| t.kron = try alloc.dupe(Kron, &.{k}),
        .diff => |d| t.diffs = try alloc.dupe(Diff, &.{d}),
    }
    return t;
}

fn loadTarget(
    alloc: std.mem.Allocator,
    store: WeightStore,
    full_stem: []const u8,
    stem: []const u8,
    d: Dialect,
    pad_bytes: *usize,
) !Target {
    var buf: [512]u8 = undefined;

    const a_view = store.get(try std.mem.print(&buf, "{s}{s}", .{ full_stem, d.a })) orelse return error.MissingTensor;
    const b_view = store.get(try std.mem.print(&buf, "{s}{s}", .{ full_stem, d.b })) orelse {
        std.log.err("lora: {s} has a {s} but no {s}", .{ stem, d.a[1..], d.b[1..] });
        return error.MissingTensor;
    };

    const a_shape = a_view.info.shape.slice();
    const b_shape = b_view.info.shape.slice();
    if (a_shape.len != 2 or b_shape.len != 2) {
        std.log.err("lora: {s} factors are {d}-D and {d}-D, expected 2-D", .{ stem, a_shape.len, b_shape.len });
        return error.ShapeMismatch;
    }
    const rank = a_shape[0];
    const in_dim = a_shape[1];
    const out_dim = b_shape[0];
    if (b_shape[1] != rank) {
        std.log.err("lora: {s} has A [{d},{d}] and B [{d},{d}]; B's second dim must be A's first", .{ stem, rank, in_dim, out_dim, b_shape[1] });
        return error.ShapeMismatch;
    }
    if (rank == 0 or in_dim == 0 or out_dim == 0) return error.ShapeMismatch;

    // alpha / A.rows, from the file's OWN shapes and before any split. A fused
    // factor's alpha is pre-multiplied to match its concatenated rank, so this
    // is the one reading that is right for both.
    var alpha: f32 = @floatFromInt(rank);
    if (store.get(try std.mem.print(&buf, "{s}.alpha", .{full_stem}))) |av| {
        // Rank 0 (`shape: []`) is how the turbo LoRAs ship it, and `Shape.count`
        // reads that as one element.
        alpha = av.asScalarF32() catch {
            std.log.err("lora: {s}.alpha has {d} entries, expected 1", .{ stem, av.info.elemCount() });
            return error.ShapeMismatch;
        };
    }
    const scale = alpha / @as(f32, @floatFromInt(rank));

    const a_dt = a_view.info.dtype;
    const b_dt = b_view.info.dtype;
    if (!ops.matmul.supportsDType(a_dt) or !ops.matmul.supportsDType(b_dt)) {
        std.log.err("lora: {s} factors are {t}/{t}, which have no GEMM path", .{ stem, a_dt, b_dt });
        return error.UnsupportedDType;
    }

    const a_full = Weight.init(a_view.bytes, a_dt, rank, in_dim);
    const b_full = Weight.init(b_view.bytes, b_dt, out_dim, rank);

    const groups = blockGroups(b_full);
    const factors = try alloc.alloc(Factor, groups);
    if (groups == 1) {
        factors[0] = .{ .a = a_full, .b = b_full, .out_off = 0 };
    } else {
        const gr = rank / groups;
        const go = out_dim / groups;
        for (factors, 0..) |*f, g| {
            f.* = .{
                // A's groups are contiguous ROWS, so they are views, no copy.
                .a = rowSlice(a_full, g * gr, gr),
                // B's are a sub-block of a wider tensor and must be repacked.
                .b = try subBlock(alloc, b_full, g * go, go, g * gr, gr),
                .out_off = g * go,
            };
        }
    }
    // After the split, so each block pads on its own; `scale` above is already
    // the file's.
    for (factors) |*f| {
        const was = f.a.bytes.len + f.b.bytes.len;
        f.* = try padRank(alloc, f.*);
        pad_bytes.* += f.a.bytes.len + f.b.bytes.len - was;
    }

    return .{
        .factors = factors,
        .in_dim = in_dim,
        .out_dim = out_dim,
        .scale = scale,
        .tag = try alloc.dupe(u8, stem),
    };
}

/// How many block-diagonal groups `b` splits into, 1 when it is dense.
///
/// Tested on the data, not on a name: a fused `qkv_proj` LoRA has three
/// rank-`r` blocks down the diagonal of a `[3 out, 3 r]` tensor and exact zeros
/// elsewhere. Checking the first off-diagonal element first means a dense B
/// costs one load.
fn blockGroups(b: Weight) usize {
    if (elemBytes(b.dtype) == 0) return 1;
    // Only the fusions that occur: 3 for qkv, 2 for a gate/value pair. A larger
    // group count would be a different architecture's fusion and is not worth
    // guessing at.
    for ([_]usize{ 3, 2 }) |g| {
        if (b.rows % g != 0 or b.cols % g != 0) continue;
        if (b.cols / g == 0) continue;
        if (isBlockDiagonal(b, g)) return g;
    }
    return 1;
}

fn isBlockDiagonal(b: Weight, groups: usize) bool {
    const es = elemBytes(b.dtype);
    const go = b.rows / groups;
    const gr = b.cols / groups;
    for (0..b.rows) |r| {
        const diag = r / go;
        const row = b.bytes[r * b.cols * es ..][0 .. b.cols * es];
        for (0..groups) |g| {
            if (g == diag) continue;
            if (!allZero(row[g * gr * es ..][0 .. gr * es])) return false;
        }
    }
    return true;
}

/// Zero bytes are zero in bf16, f16 and f32 alike (and -0.0 would read as
/// nonzero here, which is the safe direction: it declines the split).
fn allZero(bytes: []const u8) bool {
    for (bytes) |c| if (c != 0) return false;
    return true;
}

/// `dst = bf16(src)`, back in f32, round-to-nearest-even. The same conversion
/// `lora-cuda-test` builds its floor row from, which is what makes the two
/// numbers comparable.
fn roundBf16(dst: []f32, src: []const f32) void {
    const f32ToBf16 = @import("tp_core").dtype.f32ToBf16;
    for (dst, src) |*d, v| d.* = @bitCast(@as(u32, f32ToBf16(v)) << 16);
}

fn elemBytes(dt: DType) usize {
    return switch (dt) {
        .f32 => 4,
        .bf16, .f16 => 2,
        else => 0,
    };
}

/// Rows `[from, from + n)` of a row-major weight, as a view. Contiguous, since
/// rows are `cols` elements apart with no padding.
fn rowSlice(w: Weight, from: usize, n: usize) Weight {
    const stride = w.dtype.storageBytes(w.cols);
    var out = w;
    out.bytes = w.bytes[from * stride ..][0 .. n * stride];
    out.rows = n;
    return out;
}

/// A `[nr][nc]` sub-block of a row-major weight, copied into `alloc` so it is
/// contiguous. Only the block-diagonal split needs this, and only for B.
fn subBlock(alloc: std.mem.Allocator, w: Weight, r0: usize, nr: usize, c0: usize, nc: usize) !Weight {
    const es = elemBytes(w.dtype);
    std.debug.assert(es != 0);
    const dst = try alloc.alignedAlloc(u8, .of(u32), nr * nc * es);
    for (0..nr) |r| {
        const src = w.bytes[((r0 + r) * w.cols + c0) * es ..][0 .. nc * es];
        @memcpy(dst[r * nc * es ..][0 .. nc * es], src);
    }
    return Weight.init(dst, w.dtype, nr, nc);
}

// --- tests -----------------------------------------------------------------

const testing = std.testing;
const tp_core = @import("tp_core");

const reference_fixture = @embedFile("assets/lora.safetensors");

test "the sidecar reproduces ComfyUI's merged LoRA" {
    // The reference MERGES the delta into the weight; we add it beside the GEMM.
    // Everything that makes those two the same calculation is pinned here at
    // once: `alpha / A.rows`, `delta = B @ A`, the key spelling, the strength
    // dial, and the block-diagonal split of a fused qkv. See
    // tools/gen_lora_fixtures.py, which executes `comfy.lora` and asserts each
    // of those distinguishes the corpus.
    const gpa = testing.allocator;
    const io = testing.io;

    var st = try tp_core.safetensors.SafeTensors.initFromSlice(gpa, reference_fixture);
    defer st.deinit();
    const store: WeightStore = .{ .safetensors = &st };

    const targets = [_]struct { name: []const u8, groups: usize }{
        .{ .name = "blocks.0.mlp.fc2", .groups = 1 },
        .{ .name = "blocks.0.attn.qkv_proj", .groups = 3 },
        // Written in the kohya spelling by the generator, so this target is what
        // makes the pass below cover the alias as well.
        .{ .name = "blocks.0.attn.out_proj", .groups = 1 },
    };
    // ...and that stays true only while the fixture still ships one. A
    // regenerated corpus that dropped the spelling would otherwise reduce this
    // test's coverage in silence.
    try testing.expect(store.get("diffusion_model.blocks.0.attn.out_proj.lora_down.weight") != null);

    for ([_]f32{ 1.0, 0.5 }) |strength| {
        var side = try Sidecar.load(gpa, store, .{});
        defer side.deinit();
        try testing.expectEqual(targets.len, side.count());

        for (targets) |spec| {
            var buf: [128]u8 = undefined;
            const base_name = try std.mem.print(&buf, "base.{s}.weight", .{spec.name});
            const bv = store.get(base_name) orelse return error.MissingTensor;
            const bs = bv.info.shape.slice();
            const out_dim = bs[0];
            const in_dim = bs[1];

            var w = Weight.init(bv.bytes, bv.info.dtype, out_dim, in_dim);
            // `forWeight` keys on the checkpoint name, so the fixture's base
            // tensors are looked up under the name a loader would have set.
            const tag = try gpa.print("{s}.weight", .{spec.name});
            defer gpa.free(tag);
            w.tag = tag;

            const t = switch (side.forWeight(w)) {
                .ok => |p| p,
                else => {
                    std.debug.print("no sidecar found for {s}\n", .{spec.name});
                    return error.MissingTensor;
                },
            };
            // The block-diagonal split is a property of the data, so the fixture
            // is what proves the detector fires on a real fused factor.
            try testing.expectEqual(spec.groups, t.factors.len);

            const in_name = try std.mem.print(&buf, "in.{s}", .{spec.name});
            const x = try (store.get(in_name) orelse return error.MissingTensor).toF32Alloc(gpa);
            defer gpa.free(x);
            const m = x.len / in_dim;

            const want_prefix: []const u8 = if (strength == 1.0) "out" else "out_half";
            const want_name = try std.mem.print(&buf, "{s}.{s}", .{ want_prefix, spec.name });
            const want = try (store.get(want_name) orelse return error.MissingTensor).toF32Alloc(gpa);
            defer gpa.free(want);

            // Base GEMM then sidecar, exactly as a forward does it.
            const got = try gpa.alloc(f32, m * out_dim);
            defer gpa.free(got);
            try ops.matmul.matmul(io, gpa, got, x, m, w, null);
            try t.applyHost(io, gpa, got, out_dim, x, m, strength);

            const rel = relL2(want, got);
            errdefer std.debug.print("{s} @ strength {d}: rel {e}\nwant {any}\ngot  {any}\n", .{ spec.name, strength, rel, want, got });
            try testing.expect(rel < 1e-6);

            // ...and the sidecar is what moved it. Without this the test passes
            // on a base weight that already happened to match.
            const bare_name = try std.mem.print(&buf, "out_base.{s}", .{spec.name});
            const bare = try (store.get(bare_name) orelse return error.MissingTensor).toF32Alloc(gpa);
            defer gpa.free(bare);
            try testing.expect(relL2(want, bare) > 0.05);
        }
    }
}

test "a shape disagreement is distinguished from no entry at all" {
    // Both mean "do not apply this factor", and only one of them means the user
    // handed over the wrong file. Collapsing them leaves a trunk running with
    // some of its sidecars applied, which renders plausibly and wrongly.
    const gpa = testing.allocator;

    var st = try tp_core.safetensors.SafeTensors.initFromSlice(gpa, reference_fixture);
    defer st.deinit();
    const store: WeightStore = .{ .safetensors = &st };
    var side = try Sidecar.load(gpa, store, .{});
    defer side.deinit();

    const bytes: [(16 * 9 * 4)]u8 = @splat(0);
    var w = Weight.init(&bytes, .f32, 16, 9); // fc2 is 6x8 in the fixture
    w.tag = "blocks.0.mlp.fc2.weight";
    try testing.expect(side.forWeight(w) == .mismatch);

    // A name the LoRA does not cover is normal: a LoRA covers a subset.
    var other = Weight.init(&bytes, .f32, 16, 9);
    other.tag = "blocks.7.mlp.fc1.weight";
    try testing.expect(side.forWeight(other) == .none);

    // An untagged weight cannot be matched at all, and must not match the first
    // entry by accident.
    const untagged = Weight.init(&bytes, .f32, 16, 9);
    try testing.expect(side.forWeight(untagged) == .none);

    // Nothing in this file went unrecognized, so a real mismatch (a LoRA for
    // another architecture) is visible as a nonzero count.
    try testing.expectEqual(@as(usize, 0), side.unclaimed);
    try testing.expect(side.bytes() > 0);
}

test "N stacked LoRAs add, and each file keeps its own live dial" {
    // Two independent files at different strengths, against the reference
    // merging both in sequence. This is what says a stack is a SUM: if it were
    // a composition, adding sidecars beside one GEMM could not reproduce it.
    const gpa = testing.allocator;
    const io = testing.io;

    var st = try tp_core.safetensors.SafeTensors.initFromSlice(gpa, reference_fixture);
    defer st.deinit();
    const store: WeightStore = .{ .safetensors = &st };
    // The second file lives under `l2.` in the same fixture; a prefixed view is
    // a store of its own, which is exactly what a second container would be.
    var pfx = try weights_mod.Prefixed.init(gpa, store, "l2.");
    defer pfx.deinit(gpa);

    const s1: f32 = 0.75;
    const s2: f32 = 0.25;
    var stack: Stack = .{};
    defer stack.deinit(gpa);
    try testing.expectEqual(@as(usize, 3), try stack.add(gpa, store, s1, "file1", .{}));
    try testing.expectEqual(@as(usize, 3), try stack.add(gpa, pfx.store(), s2, "file2", .{}));

    for ([_][]const u8{ "blocks.0.mlp.fc2", "blocks.0.attn.qkv_proj", "blocks.0.attn.out_proj" }) |name| {
        var buf: [128]u8 = undefined;
        const bv = store.get(try std.mem.print(&buf, "base.{s}.weight", .{name})) orelse return error.MissingTensor;
        const out_dim = bv.info.shape.slice()[0];
        const in_dim = bv.info.shape.slice()[1];

        var w = Weight.init(bv.bytes, bv.info.dtype, out_dim, in_dim);
        const tag = try gpa.print("{s}.weight", .{name});
        defer gpa.free(tag);
        w.tag = tag;

        // Both files patch this weight, and the walk must say so.
        stack.beginAttach();
        try testing.expectEqual(@as(usize, 2), (try stack.resolve(w)).len);

        const x = try (store.get(try std.mem.print(&buf, "in.{s}", .{name})) orelse return error.MissingTensor).toF32Alloc(gpa);
        defer gpa.free(x);
        const m = x.len / in_dim;
        const want = try (store.get(try std.mem.print(&buf, "out_stack.{s}", .{name})) orelse return error.MissingTensor).toF32Alloc(gpa);
        defer gpa.free(want);
        const bare = try (store.get(try std.mem.print(&buf, "out_base.{s}", .{name})) orelse return error.MissingTensor).toF32Alloc(gpa);
        defer gpa.free(bare);

        const got = try gpa.alloc(f32, m * out_dim);
        defer gpa.free(got);
        try ops.matmul.matmul(io, gpa, got, x, m, w, null);
        try stack.applyHost(io, gpa, got, out_dim, x, m, w, null);

        const rel = relL2(want, got);
        errdefer std.debug.print("{s}: stack rel {e}\n", .{ name, rel });
        try testing.expect(rel < 1e-6);
        // Both files moved it, so neither dial is being dropped.
        try testing.expect(relL2(want, bare) > 0.05);

        // The dial is live: zeroing file 2 leaves file 1's delta alone, with no
        // reload anywhere. `out` is file 1 at strength 1, so its delta scales.
        const only1 = try (store.get(try std.mem.print(&buf, "out.{s}", .{name})) orelse return error.MissingTensor).toF32Alloc(gpa);
        defer gpa.free(only1);
        const want1 = try gpa.alloc(f32, m * out_dim);
        defer gpa.free(want1);
        for (want1, bare, only1) |*d, b, o| d.* = b + s1 * (o - b);

        stack.files.items[1].strength = 0;
        @memset(got, 0);
        try ops.matmul.matmul(io, gpa, got, x, m, w, null);
        try stack.applyHost(io, gpa, got, out_dim, x, m, w, null);
        const rel1 = relL2(want1, got);
        errdefer std.debug.print("{s}: file1-only rel {e}\n", .{ name, rel1 });
        try testing.expect(rel1 < 1e-6);
        // ...and that really is a different picture from the stack.
        try testing.expect(relL2(want, want1) > 0.02);
        stack.files.items[1].strength = s2;
    }
}

test "a file that patches nothing is refused rather than ignored" {
    // A LoRA for another architecture has none of the right names. Loading it
    // and rendering anyway is a flag that did nothing, silently.
    const gpa = testing.allocator;

    var st = try tp_core.safetensors.SafeTensors.initFromSlice(gpa, reference_fixture);
    defer st.deinit();
    const store: WeightStore = .{ .safetensors = &st };

    var stack: Stack = .{};
    defer stack.deinit(gpa);
    _ = try stack.add(gpa, store, 1.0, "file1", .{});

    const bytes: [16]u8 = @splat(0);
    var other = Weight.init(&bytes, .f32, 2, 2);
    other.tag = "some.other.arch.weight";

    stack.beginAttach();
    _ = try stack.resolve(other);
    // The predicate, not `finishAttach`: the latter reports through `std.log.err`,
    // and there is no test log level below it, so asserting the error itself
    // would make every passing run print.
    try testing.expectEqual(@as(?usize, 0), stack.fileWithNoHits());
}

test "only a reshaped pad is refused by name; every other adapter is read" {
    // The predicate rather than `load`, which reports through `std.log.err`.
    try testing.expectEqualStrings(".reshape_weight", Sidecar.refusedSuffix("diffusion_model.x.reshape_weight").?);
    for ([_][]const u8{ ".dora_scale", ".lora_mid.weight", ".lokr_w1", ".hada_w1_a", ".diff", ".alpha" }) |sfx| {
        try testing.expect(Sidecar.refusedSuffix(sfx) == null);
    }
    for (dialects) |d| {
        try testing.expect(Sidecar.refusedSuffix(d.a) == null);
        try testing.expect(Sidecar.refusedSuffix(d.b) == null);
    }
}

test "the generic loader reproduces ComfyUI's merge for every adapter on a linear" {
    const gpa = testing.allocator;
    const io = testing.io;
    var fx = try tp_core.safetensors.SafeTensors.initFromSlice(gpa, @embedFile("assets/lora_merge.safetensors"));
    defer fx.deinit();
    const fstore: WeightStore = .{ .safetensors = &fx };
    var prng = std.Random.DefaultPrng.init(23);
    const rnd = prng.random();
    for ([_][]const u8{ "lora", "peft", "loha", "lokr_full", "lokr_dec", "diff", "dora_out", "dora_in", "dora_in_s" }) |name| {
        var buf: [64]u8 = undefined;
        const stem = try std.mem.print(&buf, "{s}.0", .{name});
        // The case's own keys, renamed under the denoiser prefix onto `l.0.weight`.
        var entries: std.ArrayList(struct { []const u8, []const u8, []const usize, []const u8 }) = .empty;
        defer {
            for (entries.items) |e| {
                gpa.free(e[0]);
                gpa.free(e[2]);
            }
            entries.deinit(gpa);
        }
        for (fstore.names()) |n| {
            if (!std.mem.startsWith(u8, n, stem) or n.len == stem.len or n[stem.len] != '.') continue;
            if (std.mem.endsWith(u8, n, ".strength")) continue;
            const v = fstore.get(n).?;
            const dt = switch (v.info.dtype) {
                .f32 => "F32",
                .f16 => "F16",
                .bf16 => "BF16",
                else => return error.TestUnexpectedResult,
            };
            // The shape is copied out: `v` is this iteration's own copy of the view.
            try entries.append(gpa, .{ try gpa.print("diffusion_model.l.0{s}", .{n[stem.len..]}), dt, try gpa.dupe(usize, v.info.shape.slice()), v.bytes });
        }
        const blob = try safetensorsBlob(gpa, entries.items);
        defer gpa.free(blob);
        var st = try tp_core.safetensors.SafeTensors.initFromSlice(gpa, blob);
        defer st.deinit();
        const strength = try (fstore.get(try std.mem.print(&buf, "{s}.0.strength", .{name})).?).asScalarF32();

        var stack: Stack = .{};
        defer stack.deinit(gpa);
        _ = try stack.add(gpa, .{ .safetensors = &st }, strength, name, .{});

        const base_v = fstore.get(try std.mem.print(&buf, "{s}.base", .{name})).?;
        const want_v = fstore.get(try std.mem.print(&buf, "{s}.want", .{name})).?;
        const out = base_v.info.shape.slice()[0];
        const cols = base_v.info.elemCount() / out;
        var base_w = Weight.init(base_v.bytes, base_v.info.dtype, out, cols);
        base_w.tag = "model.diffusion_model.l.0.weight";
        stack.beginAttach();
        _ = try stack.resolve(base_w);
        _ = try stack.finishAttach(gpa);

        const m: usize = 6;
        const x = try gpa.alloc(f32, m * cols);
        defer gpa.free(x);
        for (x) |*v| v.* = rnd.floatNorm(f32);
        const y = try gpa.alloc(f32, 3 * m * out);
        defer gpa.free(y);
        const y_base = y[0 .. m * out];
        const y_want = y[m * out ..][0 .. m * out];
        const y_got = y[2 * m * out ..];
        try ops.matmul.matmul(io, gpa, y_base, x, m, base_w, null);
        try ops.matmul.matmul(io, gpa, y_want, x, m, Weight.init(want_v.bytes, want_v.info.dtype, out, cols), null);
        @memcpy(y_got, y_base);
        try stack.applyHost(io, gpa, y_got, out, x, m, base_w, null);
        for (y_want, y_got, y_base) |*a_, *g, b0| {
            a_.* -= b0;
            g.* -= b0;
        }
        const rel = relL2(y_want, y_got);
        errdefer std.debug.print("{s}: rel {e}\n", .{ name, rel });
        try testing.expect(rel < 1e-2);
    }
}

/// Relative L2 of `got` against `want`.
fn relL2(want: []const f32, got: []const f32) f64 {
    std.debug.assert(want.len == got.len);
    var l2_ref: f64 = 0;
    var l2_err: f64 = 0;
    for (want, got) |e, a| {
        l2_ref += @as(f64, e) * e;
        l2_err += @as(f64, e - a) * (e - a);
    }
    return if (l2_ref > 0) @sqrt(l2_err / l2_ref) else @sqrt(l2_err);
}

test "the scale comes from the file's own A rows, so a fused alpha reads right" {
    // The one number that is silent when wrong. A fused qkv factor ships A with
    // 3r rows and alpha multiplied by 3; a per-block reading (alpha/r) and a
    // whole-tensor reading (3 alpha / 3 r) must agree, and they only do if the
    // divisor is the file's A row count.
    const r: usize = 4;
    const per_block = 8.0 / @as(f32, @floatFromInt(r));
    const fused = (3.0 * 8.0) / @as(f32, @floatFromInt(3 * r));
    try testing.expectEqual(per_block, fused);
    // ...and dividing the fused alpha by the per-block rank is the 3x error.
    try testing.expectApproxEqAbs(@as(f32, 3.0), (3.0 * 8.0 / @as(f32, @floatFromInt(r))) / per_block, 1e-6);
}

test "a block-diagonal B is detected and a dense one is not" {
    // 6x6 f32: three 2x2 blocks down the diagonal.
    var diag: [36]f32 = @splat(0);
    for (0..3) |g| {
        for (0..2) |i| for (0..2) |j| {
            diag[(g * 2 + i) * 6 + g * 2 + j] = @floatFromInt(g + 1);
        };
    }
    const wd = Weight.fromF32(&diag, 6, 6);
    try testing.expectEqual(@as(usize, 3), blockGroups(wd));

    // One nonzero off the diagonal and it must decline, since splitting would
    // drop that entry entirely.
    var broken = diag;
    broken[0 * 6 + 5] = 1e-30;
    try testing.expectEqual(@as(usize, 1), blockGroups(Weight.fromF32(&broken, 6, 6)));

    // A dense B: no group count fits.
    var dense: [36]f32 = undefined;
    for (&dense, 0..) |*v, i| v.* = @floatFromInt(i + 1);
    try testing.expectEqual(@as(usize, 1), blockGroups(Weight.fromF32(&dense, 6, 6)));

    // 2-way fusion is found too, and 3 is preferred when both would fit (an
    // all-zero B, where the choice does not matter).
    var two: [16]f32 = @splat(0);
    for (0..2) |g| {
        for (0..2) |i| for (0..2) |j| {
            two[(g * 2 + i) * 4 + g * 2 + j] = 1;
        };
    }
    try testing.expectEqual(@as(usize, 2), blockGroups(Weight.fromF32(&two, 4, 4)));
}

test "splitting a block-diagonal factor reproduces the dense product" {
    // The whole justification for the split: same numbers, a third of the work.
    // If these disagree the render is finite and wrong.
    const r: usize = 2;
    const in_dim: usize = 3;
    const out_g: usize = 2;
    const g: usize = 3;

    var prng = std.Random.DefaultPrng.init(0x10ea9);
    const rnd = prng.random();

    var a: [g * r * in_dim]f32 = undefined;
    for (&a) |*v| v.* = rnd.floatNorm(f32);
    var b: [g * out_g * g * r]f32 = @splat(0);
    for (0..g) |gi| {
        for (0..out_g) |i| for (0..r) |j| {
            b[(gi * out_g + i) * (g * r) + gi * r + j] = rnd.floatNorm(f32);
        };
    }
    var x: [4 * in_dim]f32 = undefined;
    for (&x) |*v| v.* = rnd.floatNorm(f32);

    const a_w = Weight.fromF32(&a, g * r, in_dim);
    const b_w = Weight.fromF32(&b, g * out_g, g * r);
    const scale: f32 = 0.0625;

    const dense: Target = .{
        .factors = &.{.{ .a = a_w, .b = b_w, .out_off = 0 }},
        .in_dim = in_dim,
        .out_dim = g * out_g,
        .scale = scale,
        .tag = "dense",
    };

    // The same tensor, split the way `loadTarget` would.
    var split_factors: [g]Factor = undefined;
    var blocks: [g][out_g * r]f32 = undefined;
    for (0..g) |gi| {
        for (0..out_g) |i| for (0..r) |j| {
            blocks[gi][i * r + j] = b[(gi * out_g + i) * (g * r) + gi * r + j];
        };
        split_factors[gi] = .{
            .a = rowSlice(a_w, gi * r, r),
            .b = Weight.fromF32(&blocks[gi], out_g, r),
            .out_off = gi * out_g,
        };
    }
    const split: Target = .{
        .factors = &split_factors,
        .in_dim = in_dim,
        .out_dim = g * out_g,
        .scale = scale,
        .tag = "split",
    };

    try testing.expectEqual(@as(usize, 3), blockGroups(b_w));

    var y_dense: [4 * g * out_g]f32 = @splat(0);
    var y_split: [4 * g * out_g]f32 = @splat(0);
    try dense.applyHost(testing.io, testing.allocator, &y_dense, g * out_g, &x, 4, 1.0);
    try split.applyHost(testing.io, testing.allocator, &y_split, g * out_g, &x, 4, 1.0);
    errdefer std.debug.print("dense {any}\nsplit {any}\n", .{ y_dense, y_split });
    for (y_dense, y_split) |d, s| try testing.expectApproxEqAbs(d, s, 1e-5);
    // And the delta is not trivially zero, or the comparison proves nothing.
    var mag: f32 = 0;
    for (y_dense) |v| mag = @max(mag, @abs(v));
    try testing.expect(mag > 1e-3);
    // The split really does less work. Only the B GEMM shrinks (by `g`); the A
    // GEMM is the same either way, so the total ratio depends on the shape and
    // is well under `g`. At H3's real fused qkv it is ~2.1x.
    try testing.expect(split.flops(4) < dense.flops(4));
}

fn randomBf16(gpa: std.mem.Allocator, rnd: std.Random, n: usize, sd: f32) ![]u8 {
    const f32ToBf16 = tp_core.dtype.f32ToBf16;
    const b = try gpa.alloc(u8, n * 2);
    for (0..n) |i| std.mem.writeInt(u16, b[i * 2 ..][0..2], f32ToBf16(rnd.floatNorm(f32) * sd), .little);
    return b;
}

test "the int8 A is rotated exactly when asked, padded to the tile, and reads back to A" {
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x1a8);
    const rnd = prng.random();
    const rank: usize = 3; // pads to 128
    const in_dim: usize = 512;
    const a_bytes = try randomBf16(gpa, rnd, rank * in_dim, 0.02);
    defer gpa.free(a_bytes);
    const a = Weight.init(a_bytes, .bf16, rank, in_dim);
    const a_f = try ops.matmul.materializeF32(gpa, a);
    defer gpa.free(@as([]const f32, @ptrCast(@alignCast(a_f.bytes))));
    const a32: []const f32 = @ptrCast(@alignCast(a_f.bytes));

    for ([_]bool{ false, true }) |rot| {
        const q = try quantizeA(gpa, a, rot);
        defer q.free(gpa);
        try testing.expectEqual(@as(usize, 128), q.scale.len);
        try testing.expectEqual(@as(usize, 128 * in_dim), q.q.len);
        for (q.scale[rank..]) |s| try testing.expectEqual(@as(f32, 0), s);
        for (q.q[rank * in_dim ..]) |v| try testing.expectEqual(@as(i8, 0), v);

        // Raw dequant against A itself and against A rotated: it matches exactly one.
        const deq = try gpa.alloc(f32, rank * in_dim);
        defer gpa.free(deq);
        for (0..rank * in_dim) |i| deq[i] = @as(f32, @floatFromInt(q.q[i])) * q.scale[i / in_dim];
        const a_rot = try gpa.dupe(f32, a32);
        defer gpa.free(a_rot);
        ops.convrot.rotate(a_rot);
        const vs_plain = relL2(a32, deq);
        const vs_rot = relL2(a_rot, deq);
        errdefer std.debug.print("rot {} vs plain {e} vs rotated {e}\n", .{ rot, vs_plain, vs_rot });
        try testing.expect((if (rot) vs_rot else vs_plain) < 1e-2);
        try testing.expect((if (rot) vs_plain else vs_rot) > 0.5);

        // Each row's largest magnitude lands on 127 exactly, so the scale is the
        // row's own absmax and nothing else multiplied into it.
        for (0..rank) |r| {
            var m: i8 = 0;
            for (q.q[r * in_dim ..][0..in_dim]) |v| m = @max(m, if (v < 0) -v else v);
            try testing.expectEqual(@as(i8, 127), m);
        }

        // As a Weight, the convrot flag is what undoes the rotation on the way back.
        const w = q.weight(in_dim, rot);
        const back = try ops.matmul.materializeF32(gpa, w);
        const back32: []const f32 = @ptrCast(@alignCast(back.bytes));
        defer gpa.free(back32);
        const rel = relL2(a32, back32[0 .. rank * in_dim]);
        errdefer std.debug.print("rot {} read-back rel {e}\n", .{ rot, rel });
        try testing.expect(rel < 1e-2);
    }
    // A rotation needs whole groups.
    const narrow = Weight.init(a_bytes[0 .. rank * 200 * 2], .bf16, rank, 200);
    try testing.expectError(error.ShapeMismatch, quantizeA(gpa, narrow, true));
}

test "quantizeHits leaves the file's scale out, follows the rotation, and skips fused targets" {
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x7a6);
    const rnd = prng.random();
    const in_dim: usize = 256;
    const rank: usize = 128;
    const a_bytes = try randomBf16(gpa, rnd, rank * in_dim, 0.02);
    defer gpa.free(a_bytes);
    const b_bytes = try randomBf16(gpa, rnd, 128 * rank, 0.02);
    defer gpa.free(b_bytes);
    const f: Factor = .{ .a = Weight.init(a_bytes, .bf16, rank, in_dim), .b = Weight.init(b_bytes, .bf16, 128, rank), .out_off = 0 };
    const t: Target = .{ .factors = &.{f}, .in_dim = in_dim, .out_dim = 128, .scale = 0.0625, .tag = "one" };
    const fused: Target = .{ .factors = &.{ f, f }, .in_dim = in_dim, .out_dim = 256, .scale = 1, .tag = "two" };

    var stack: Stack = .{};
    defer stack.deinit(gpa);
    for ([_]struct { []const u8, *const Target }{ .{ "one.weight", &t }, .{ "two.weight", &fused } }) |e| {
        var list: std.ArrayList(Stack.Hit) = .empty;
        try list.append(gpa, .{ .target = e[1], .file = 0 });
        try stack.index.put(gpa, e[0], list);
    }
    var w = Weight.init(a_bytes[0 .. 128 * in_dim], .i8, 128, in_dim);
    w.tag = "one.weight";
    var wf = Weight.init(a_bytes[0 .. 256 * in_dim / 2], .i8, 128, in_dim);
    wf.tag = "two.weight";

    try stack.quantizeHits(gpa, w, true);
    try stack.quantizeHits(gpa, wf, true);
    try testing.expect(stack.forWeight(wf)[0].a_i8 == null);
    const q1 = stack.forWeight(w)[0].a_i8.?;
    try testing.expectEqual(@as(u32, 256), q1.convrot);
    // Idempotent: the same answer is not built twice.
    try stack.quantizeHits(gpa, w, true);
    try testing.expectEqual(@as(usize, 1), stack.quant.items.len);
    try testing.expectEqual(q1.bytes.ptr, stack.forWeight(w)[0].a_i8.?.bytes.ptr);
    // The other basis is a new copy.
    try stack.quantizeHits(gpa, w, false);
    try testing.expectEqual(@as(u32, 0), stack.forWeight(w)[0].a_i8.?.convrot);
    try testing.expectEqual(@as(usize, 2), stack.quant.items.len);

    // A read back is A, not 0.0625 A.
    const back = try ops.matmul.materializeF32(gpa, stack.forWeight(w)[0].a_i8.?);
    const back32: []const f32 = @ptrCast(@alignCast(back.bytes));
    defer gpa.free(back32);
    const a_f = try ops.matmul.materializeF32(gpa, f.a);
    const a32: []const f32 = @ptrCast(@alignCast(a_f.bytes));
    defer gpa.free(a32);
    try testing.expect(relL2(a32, back32) < 1e-2);

    // Eviction sees both copies' bytes and scales.
    const Count = struct {
        n: usize = 0,
        fn visit(c: *@This(), _: []const u8) void {
            c.n += 1;
        }
    };
    var c: Count = .{};
    stack.forEachFactorBytes(&c, Count.visit);
    try testing.expectEqual(@as(usize, 2 * 2), c.n);
}

/// A safetensors file of `tensors` (name, dtype string, shape, bytes), in order.
pub fn safetensorsBlob(gpa: std.mem.Allocator, tensors: []const struct { []const u8, []const u8, []const usize, []const u8 }) ![]u8 {
    var header: std.ArrayList(u8) = .empty;
    defer header.deinit(gpa);
    var off: usize = 0;
    try header.append(gpa, '{');
    for (tensors, 0..) |t, i| {
        if (i > 0) try header.append(gpa, ',');
        try header.print(gpa, "\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ t[0], t[1] });
        for (t[2], 0..) |d, j| try header.print(gpa, "{s}{d}", .{ if (j > 0) "," else "", d });
        try header.print(gpa, "],\"data_offsets\":[{d},{d}]}}", .{ off, off + t[3].len });
        off += t[3].len;
    }
    try header.append(gpa, '}');
    const out = try gpa.alloc(u8, 8 + header.items.len + off);
    std.mem.writeInt(u64, out[0..8], header.items.len, .little);
    @memcpy(out[8..][0..header.items.len], header.items);
    var p = 8 + header.items.len;
    for (tensors) |t| {
        @memcpy(out[p..][0..t[3].len], t[3]);
        p += t[3].len;
    }
    return out;
}

test "a rank-8 factor pads to the slab, keeps the file's scale, and applies unchanged" {
    const gpa = testing.allocator;
    const io = testing.io;
    var prng = std.Random.DefaultPrng.init(0x9ad8);
    const rnd = prng.random();
    const in_dim: usize = 64;
    const out_dim: usize = 128;
    const r8: usize = 8;
    const r32: usize = 32;
    const a8 = try randomBf16(gpa, rnd, r8 * in_dim, 0.1);
    defer gpa.free(a8);
    const b8 = try randomBf16(gpa, rnd, out_dim * r8, 0.1);
    defer gpa.free(b8);
    const a32 = try randomBf16(gpa, rnd, r32 * in_dim, 0.1);
    defer gpa.free(a32);
    const b32 = try randomBf16(gpa, rnd, out_dim * r32, 0.1);
    defer gpa.free(b32);
    const alpha: f32 = 4;

    const blob = try safetensorsBlob(gpa, &.{
        .{ "diffusion_model.blk.fc.alpha", "F32", &.{}, std.mem.asBytes(&alpha) },
        .{ "diffusion_model.blk.fc.lora_down.weight", "BF16", &.{ r8, in_dim }, a8 },
        .{ "diffusion_model.blk.fc.lora_up.weight", "BF16", &.{ out_dim, r8 }, b8 },
        .{ "diffusion_model.blk.wide.lora_down.weight", "BF16", &.{ r32, in_dim }, a32 },
        .{ "diffusion_model.blk.wide.lora_up.weight", "BF16", &.{ out_dim, r32 }, b32 },
    });
    defer gpa.free(blob);
    var st = try tp_core.safetensors.SafeTensors.initFromSlice(gpa, blob);
    defer st.deinit();
    const store: WeightStore = .{ .safetensors = &st };

    var stack: Stack = .{};
    defer stack.deinit(gpa);
    try testing.expectEqual(@as(usize, 2), try stack.add(gpa, store, 1.0, "rank8", .{}));

    const t = stack.index.get("blk.fc.weight").?.items[0].target;
    // alpha / the FILE's rows: 4 / 8, not 4 / 32.
    try testing.expectEqual(@as(f32, 0.5), t.scale);
    const f = t.factors[0];
    try testing.expectEqual(r32, f.rank());
    try testing.expectEqual(r32, f.b.cols);
    try testing.expectEqualSlices(u8, a8, f.a.bytes[0 .. r8 * in_dim * 2]);
    try testing.expect(allZero(f.a.bytes[r8 * in_dim * 2 ..]));
    for (0..out_dim) |o| {
        const row = f.b.bytes[o * r32 * 2 ..][0 .. r32 * 2];
        try testing.expectEqualSlices(u8, b8[o * r8 * 2 ..][0 .. r8 * 2], row[0 .. r8 * 2]);
        try testing.expect(allZero(row[r8 * 2 ..]));
    }
    try testing.expectEqual((r32 - r8) * (in_dim + out_dim) * 2, stack.padBytes());
    try testing.expectEqual(a8.len + b8.len + a32.len + b32.len + stack.padBytes(), stack.bytes());

    // An aligned rank stays a view into the file, and no alpha means alpha = rank.
    const tw = stack.index.get("blk.wide.weight").?.items[0].target;
    try testing.expectEqual(@as(f32, 1), tw.scale);
    try testing.expectEqual(store.get("diffusion_model.blk.wide.lora_down.weight").?.bytes.ptr, tw.factors[0].a.bytes.ptr);
    try testing.expectEqual(store.get("diffusion_model.blk.wide.lora_up.weight").?.bytes.ptr, tw.factors[0].b.bytes.ptr);

    // Against the file's own rank-8 factor at alpha / 8, computed directly.
    const m: usize = 5;
    const x = try gpa.alloc(f32, m * in_dim);
    defer gpa.free(x);
    for (x) |*v| v.* = rnd.floatNorm(f32);
    const y = try gpa.alloc(f32, m * out_dim);
    defer gpa.free(y);
    @memset(y, 0);
    const base: [(out_dim * in_dim * 2)]u8 = @splat(0);
    var w = Weight.init(&base, .bf16, out_dim, in_dim);
    w.tag = "blk.fc.weight";
    try stack.applyHost(io, gpa, y, out_dim, x, m, w, null);
    const bf = struct {
        fn at(b: []const u8, i: usize) f64 {
            return @as(f32, @bitCast(@as(u32, std.mem.readInt(u16, b[i * 2 ..][0..2], .little)) << 16));
        }
    }.at;
    var worst: f64 = 0;
    for (0..m) |i| for (0..out_dim) |o| {
        var acc: f64 = 0;
        for (0..r8) |r| {
            var lo: f64 = 0;
            for (0..in_dim) |c| lo += bf(a8, r * in_dim + c) * x[i * in_dim + c];
            acc += bf(b8, o * r8 + r) * lo;
        }
        worst = @max(worst, @abs(alpha / 8.0 * acc - y[i * out_dim + o]));
    };
    errdefer std.debug.print("padded apply worst abs err {e}\n", .{worst});
    try testing.expect(worst < 1e-5);

    // Eviction walks the padded copies, not the file bytes they replaced.
    const Seen = struct {
        a: [*]const u8,
        b: [*]const u8,
        hits: usize = 0,
        fn visit(s: *@This(), bytes: []const u8) void {
            if (bytes.ptr == s.a or bytes.ptr == s.b) s.hits += 1;
        }
    };
    var seen: Seen = .{ .a = f.a.bytes.ptr, .b = f.b.bytes.ptr };
    stack.forEachFactorBytes(&seen, Seen.visit);
    try testing.expectEqual(@as(usize, 2), seen.hits);
    try testing.expect(f.a.bytes.ptr != store.get("diffusion_model.blk.fc.lora_down.weight").?.bytes.ptr);
}

test "applyHost writes into a fused destination's own column range" {
    // The CPU forward's qkv buffer is [seq][3 * inner]; a factor covering the
    // middle third must leave the other two untouched. Writing at the wrong
    // stride corrupts a neighbour instead of erroring.
    const in_dim: usize = 2;
    const a = [_]f32{ 1, 0 };
    const b = [_]f32{ 2, 3 };
    const t: Target = .{
        .factors = &.{.{
            .a = Weight.fromF32(&a, 1, in_dim),
            .b = Weight.fromF32(&b, 2, 1),
            .out_off = 2,
        }},
        .in_dim = in_dim,
        .out_dim = 6,
        .scale = 1.0,
        .tag = "mid",
    };
    const x = [_]f32{ 1, 9, 2, 9 };
    var y: [2 * 6]f32 = @splat(0);
    try t.applyHost(testing.io, testing.allocator, &y, 6, &x, 2, 1.0);
    // x row 0 is [1, 9] -> lo = 1 -> hi = [2, 3] at columns 2 and 3.
    try testing.expectEqualSlices(f32, &.{ 0, 0, 2, 3, 0, 0, 0, 0, 4, 6, 0, 0 }, &y);
}

test "applyHost accumulates rather than overwriting" {
    // It is a sidecar: the base GEMM's output is already in `y`. Overwriting is
    // the LoRA rendering alone, which looks like a broken model rather than a
    // broken sidecar.
    const a = [_]f32{1};
    const b = [_]f32{1};
    const t: Target = .{
        .factors = &.{.{ .a = Weight.fromF32(&a, 1, 1), .b = Weight.fromF32(&b, 1, 1), .out_off = 0 }},
        .in_dim = 1,
        .out_dim = 1,
        .scale = 2.0,
        .tag = "acc",
    };
    var y = [_]f32{100};
    const x = [_]f32{3};
    try t.applyHost(testing.io, testing.allocator, &y, 1, &x, 1, 1.0);
    try testing.expectEqual(@as(f32, 106), y[0]);
}

test "the host apply bands rows without changing the answer" {
    // `host_band` bounds the intermediate; a band boundary that dropped or
    // double-counted rows would show only at a specific sequence length.
    const in_dim: usize = 3;
    const out_dim: usize = 2;
    const rank: usize = 2;
    const m: usize = host_band + 7;

    var prng = std.Random.DefaultPrng.init(0xba7d);
    const rnd = prng.random();
    var a: [rank * in_dim]f32 = undefined;
    for (&a) |*v| v.* = rnd.floatNorm(f32);
    var b: [out_dim * rank]f32 = undefined;
    for (&b) |*v| v.* = rnd.floatNorm(f32);

    const t: Target = .{
        .factors = &.{.{
            .a = Weight.fromF32(&a, rank, in_dim),
            .b = Weight.fromF32(&b, out_dim, rank),
            .out_off = 0,
        }},
        .in_dim = in_dim,
        .out_dim = out_dim,
        .scale = 0.5,
        .tag = "band",
    };

    const x = try testing.allocator.alloc(f32, m * in_dim);
    defer testing.allocator.free(x);
    for (x) |*v| v.* = rnd.floatNorm(f32);
    const y = try testing.allocator.alloc(f32, m * out_dim);
    defer testing.allocator.free(y);
    @memset(y, 0);
    try t.applyHost(testing.io, testing.allocator, y, out_dim, x, m, 1.0);

    // Every row, computed directly.
    for (0..m) |i| {
        for (0..out_dim) |o| {
            var acc: f32 = 0;
            for (0..rank) |r| {
                var lo: f32 = 0;
                for (0..in_dim) |c| lo += a[r * in_dim + c] * x[i * in_dim + c];
                acc += b[o * rank + r] * lo;
            }
            errdefer std.debug.print("row {d} col {d}: got {e} want {e}\n", .{ i, o, y[i * out_dim + o], 0.5 * acc });
            try testing.expectApproxEqAbs(0.5 * acc, y[i * out_dim + o], 1e-4);
        }
    }
}

test "a target no linear matches is counted and named, and a bundled tag still matches" {
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x3155);
    const rnd = prng.random();
    const in_dim: usize = 32;
    const out_dim: usize = 128;
    const r: usize = 32;
    var bufs: [6][]u8 = undefined;
    for (&bufs, 0..) |*b, i| b.* = try randomBf16(gpa, rnd, if (i % 2 == 0) r * in_dim else out_dim * r, 0.1);
    defer for (bufs) |b| gpa.free(b);
    const blob = try safetensorsBlob(gpa, &.{
        .{ "diffusion_model.blk.a.lora_A.weight", "BF16", &.{ r, in_dim }, bufs[0] },
        .{ "diffusion_model.blk.a.lora_B.weight", "BF16", &.{ out_dim, r }, bufs[1] },
        .{ "diffusion_model.blk.b.lora_A.weight", "BF16", &.{ r, in_dim }, bufs[2] },
        .{ "diffusion_model.blk.b.lora_B.weight", "BF16", &.{ out_dim, r }, bufs[3] },
        .{ "diffusion_model.blk.bogus.lora_A.weight", "BF16", &.{ r, in_dim }, bufs[4] },
        .{ "diffusion_model.blk.bogus.lora_B.weight", "BF16", &.{ out_dim, r }, bufs[5] },
    });
    defer gpa.free(blob);
    var st = try tp_core.safetensors.SafeTensors.initFromSlice(gpa, blob);
    defer st.deinit();
    var stack: Stack = .{};
    defer stack.deinit(gpa);
    try testing.expectEqual(@as(usize, 3), try stack.add(gpa, .{ .safetensors = &st }, 1.0, "f", .{}));

    const base: [(out_dim * in_dim * 2)]u8 = @splat(0);
    var wa = Weight.init(&base, .bf16, out_dim, in_dim);
    // The bundled checkpoint prefix is not part of the name a LoRA patches.
    wa.tag = "model.diffusion_model.blk.a.weight";
    var wb = wa;
    wb.tag = "blk.b.weight";
    stack.beginAttach();
    try testing.expectEqual(@as(usize, 1), (try stack.resolve(wa)).len);
    try stack.resolveAll(&.{ wb, wb });
    // A weight resolved twice is still one target.
    try testing.expectEqual(Stack.Coverage{ .applied = 2, .total = 3 }, stack.coverage(0));
    const names = try stack.missed(gpa, 0);
    defer gpa.free(names);
    try testing.expectEqual(@as(usize, 1), names.len);
    try testing.expectEqualStrings("blk.bogus", names[0]);
    // A fresh walk forgets the last one.
    stack.beginAttach();
    try testing.expectEqual(Stack.Coverage{ .applied = 0, .total = 3 }, stack.coverage(0));
}

test "split q/k/v keys merge onto row blocks of the fused tensor, each with its own alpha" {
    // Z-Image's LoRAs are written against diffusers' `to_q/to_k/to_v`; ComfyUI
    // patches rows [0,h), [h,2h), [2h,3h) of `qkv` with them. A missing slot is
    // no delta, and each slot keeps its own alpha / rank.
    const gpa = testing.allocator;
    const io = testing.io;
    var prng = std.Random.DefaultPrng.init(0x9e7a);
    const rnd = prng.random();
    const in_dim: usize = 64;
    const h: usize = 128;
    const r: usize = 8;
    const aq = try randomBf16(gpa, rnd, r * in_dim, 0.1);
    defer gpa.free(aq);
    const bq = try randomBf16(gpa, rnd, h * r, 0.1);
    defer gpa.free(bq);
    const ak = try randomBf16(gpa, rnd, r * in_dim, 0.1);
    defer gpa.free(ak);
    const bk = try randomBf16(gpa, rnd, h * r, 0.1);
    defer gpa.free(bk);
    const alpha_k: f32 = 2;
    const blob = try safetensorsBlob(gpa, &.{
        .{ "diffusion_model.l.0.attention.to_q.lora_A.weight", "BF16", &.{ r, in_dim }, aq },
        .{ "diffusion_model.l.0.attention.to_q.lora_B.weight", "BF16", &.{ h, r }, bq },
        .{ "diffusion_model.l.0.attention.to_k.alpha", "F32", &.{}, std.mem.asBytes(&alpha_k) },
        .{ "diffusion_model.l.0.attention.to_k.lora_A.weight", "BF16", &.{ r, in_dim }, ak },
        .{ "diffusion_model.l.0.attention.to_k.lora_B.weight", "BF16", &.{ h, r }, bk },
    });
    defer gpa.free(blob);
    var st = try tp_core.safetensors.SafeTensors.initFromSlice(gpa, blob);
    defer st.deinit();
    const keys = [_]Rename{
        .{ .from = ".attention.to_q", .to = ".attention.qkv", .slot = 0, .slots = 3 },
        .{ .from = ".attention.to_k", .to = ".attention.qkv", .slot = 1, .slots = 3 },
        .{ .from = ".attention.to_v", .to = ".attention.qkv", .slot = 2, .slots = 3 },
    };
    var stack: Stack = .{};
    defer stack.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), try stack.add(gpa, .{ .safetensors = &st }, 1.0, "z", .{ .renames = &keys }));
    try testing.expectEqual(@as(usize, 2), stack.files.items[0].side.keyCount());

    const t = stack.index.get("l.0.attention.qkv.weight").?.items[0].target;
    try testing.expectEqual(3 * h, t.out_dim);
    try testing.expectEqual(@as(usize, 2), t.factors.len);
    // Each slot's factor covers its own rows and nothing else, so a device row
    // view finds it and a view across two slots finds none.
    try testing.expectEqual(h, t.factorAt(h, h).?.out_off);
    try testing.expect(t.factorAt(2 * h, h) == null);
    try testing.expect(t.factorAt(h / 2, h) == null);

    const m: usize = 3;
    const x = try gpa.alloc(f32, m * in_dim);
    defer gpa.free(x);
    for (x) |*v| v.* = rnd.floatNorm(f32);
    const y = try gpa.alloc(f32, m * 3 * h);
    defer gpa.free(y);
    @memset(y, 0);
    const base: [(3 * h * in_dim * 2)]u8 = @splat(0);
    var w = Weight.init(&base, .bf16, 3 * h, in_dim);
    w.tag = "l.0.attention.qkv.weight";
    try stack.applyHost(io, gpa, y, 3 * h, x, m, w, null);

    const bf = struct {
        fn at(b: []const u8, i: usize) f64 {
            return @as(f32, @bitCast(@as(u32, std.mem.readInt(u16, b[i * 2 ..][0..2], .little)) << 16));
        }
    }.at;
    var worst: f64 = 0;
    for (0..m) |i| for (0..3 * h) |o| {
        const slot = o / h;
        var want: f64 = 0;
        if (slot < 2) {
            const a = if (slot == 0) aq else ak;
            const b = if (slot == 0) bq else bk;
            const s: f64 = if (slot == 0) 1 else alpha_k / @as(f64, @floatFromInt(r));
            for (0..r) |k| {
                var lo: f64 = 0;
                for (0..in_dim) |c| lo += bf(a, k * in_dim + c) * x[i * in_dim + c];
                want += s * bf(b, (o - slot * h) * r + k) * lo;
            }
        }
        worst = @max(worst, @abs(want - y[i * 3 * h + o]));
    };
    errdefer std.debug.print("merged apply worst abs err {e}\n", .{worst});
    try testing.expect(worst < 1e-5);
}
