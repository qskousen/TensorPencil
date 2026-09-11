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
//! already has.
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
//! `strength` is NOT folded into a factor. `Target.scale` holds the file's own
//! `alpha / rank` and the dial is multiplied at apply time, so a GUI slider does
//! not mean re-reading a gigabyte. `Stack` is N files over one model, each with
//! its own live dial; their deltas add, so order does not matter and stacking is
//! a merged name index.

const std = @import("std");
const ops = @import("tp_ops");
const weights_mod = @import("tp_core").weights;

const Weight = ops.matmul.Weight;
const WeightStore = weights_mod.WeightStore;
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
const refuse = [_][]const u8{ ".lora_mid.weight", ".dora_scale", ".reshape_weight" };

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

    pub fn rank(f: Factor) usize {
        return f.a.rows;
    }
};

/// The sidecar for one linear.
pub const Target = struct {
    factors: []const Factor,
    in_dim: usize,
    out_dim: usize,
    /// `alpha / a.rows`, from the file's own shapes. Derived once before any
    /// block-diagonal split, which is what makes a fused alpha read right, so it
    /// belongs to the target and not to each factor. The runtime dial is
    /// multiplied on top at apply time.
    scale: f32,
    /// The base tensor's name, for diagnostics.
    tag: []const u8,

    /// FLOPs one apply costs over `m` rows, for reporting the overhead honestly.
    pub fn flops(t: *const Target, m: usize) usize {
        var n: usize = 0;
        for (t.factors) |f| n += 2 * m * f.rank() * (t.in_dim + f.b.rows);
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
                for (0..n) |i| {
                    const dst = y[(r0 + i) * y_stride + f.out_off ..][0..o];
                    const src = hi[i * o ..][0..o];
                    for (dst, src) |*d, v| d.* += s * v;
                }
            }
        }
    }
};

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

    pub fn deinit(self: *Sidecar) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// The prefix ComfyUI's generic format puts on a denoiser tensor name.
    pub const prefix = "diffusion_model.";

    /// Load every factor pair in `store`, in any dialect, keyed by the base
    /// tensor name it patches.
    ///
    /// The runtime dial is not taken here: see `Stack.File.strength`.
    pub fn load(gpa: std.mem.Allocator, store: WeightStore) !Sidecar {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();

        var index: std.StringHashMapUnmanaged(Target) = .empty;

        var unclaimed: usize = 0;
        for (store.names()) |full| {
            // Only denoiser keys are this loader's business; anything else in
            // the file belongs to another component (a text-encoder LoRA) and is
            // not evidence of a problem.
            if (!std.mem.startsWith(u8, full, prefix)) continue;

            if (refusedSuffix(full)) |sfx| {
                std.log.err("lora: {s} is a {s} adapter, which is a different calculation and not a plain LoRA", .{ full, sfx[1..] });
                return error.UnsupportedLora;
            }

            // The dialect a stem loads under is the FIRST that matches, and a
            // stem is reached through its `a` key alone, so `b` and `alpha` are
            // claimed here without being iterated to.
            const d: Dialect = for (dialects) |cand| {
                if (std.mem.endsWith(u8, full, cand.a)) break cand;
            } else {
                if (!claimedSuffix(full)) unclaimed += 1;
                continue;
            };

            const stem = full[prefix.len .. full.len - d.a.len];
            const t = try loadTarget(alloc, store, full[0 .. full.len - d.a.len], stem, d);
            const base = try std.fmt.allocPrint(alloc, "{s}.weight", .{stem});
            try index.put(alloc, base, t);
        }

        return .{
            .arena = arena,
            .index = index,
            .unclaimed = unclaimed,
        };
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

    /// Whether `name` is a key some dialect accounts for, so it is not evidence
    /// of a spelling this loader missed. `a` keys are handled by the caller.
    fn claimedSuffix(name: []const u8) bool {
        if (std.mem.endsWith(u8, name, ".alpha")) return true;
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
        const t = self.index.getPtr(tag) orelse return .none;
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
        while (it.next()) |t| {
            for (t.factors) |f| n += f.a.bytes.len + f.b.bytes.len;
        }
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
    index: std.StringHashMapUnmanaged(std.ArrayListUnmanaged(Hit)) = .empty,
    /// Per-file hit counts for the attach walk in progress. Only `beginAttach`
    /// / `resolve` / `finishAttach` touch these.
    hits: [max_files]usize = @splat(0),

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
    };

    pub fn deinit(self: *Stack, gpa: std.mem.Allocator) void {
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
    ) !usize {
        if (self.files.items.len >= max_files) return error.TooManyLoras;
        const owned_path = try gpa.dupe(u8, path);
        errdefer gpa.free(owned_path);
        var side = try Sidecar.load(gpa, store);
        errdefer side.deinit();

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
        const list = self.index.getPtr(tag) orelse return &.{};
        return list.items;
    }

    /// Zero the per-file counters before an attach walk.
    ///
    /// The walk is the model's own, not a shared linear list, because only the
    /// model knows which linears it actually runs. Each family calls
    /// `beginAttach`, then `resolve` for every linear, then `finishAttach`.
    pub fn beginAttach(self: *Stack) void {
        self.hits = @splat(0);
    }

    /// The validating lookup: `forWeight` plus the width check, and it counts the
    /// hit against its file.
    ///
    /// A LoRA trained against another variant of the same architecture has all
    /// the right NAMES and the wrong widths; its factors would still multiply,
    /// into the wrong columns. Refusing beats skipping, since a trunk with only
    /// some of its sidecars applied renders plausibly and wrongly.
    pub fn resolve(self: *Stack, w: Weight) ![]const Hit {
        const hits = self.forWeight(w);
        for (hits) |h| {
            const t = h.target;
            if (t.in_dim != w.cols or t.out_dim != w.rows) {
                std.log.err("lora: {s}: {s} is {d}x{d} but the checkpoint's is {d}x{d}", .{
                    self.files.items[h.file].path, t.tag, t.out_dim, t.in_dim, w.rows, w.cols,
                });
                return error.ShapeMismatch;
            }
            self.hits[h.file] += 1;
        }
        return hits;
    }

    /// Close an attach walk, refusing a file that patched nothing.
    ///
    /// Returns how many (linear, file) pairs will apply. A file with no hits is a
    /// LoRA for another architecture, which is a mistake worth reporting rather
    /// than a render that quietly ignores the flag.
    pub fn finishAttach(self: *const Stack) !usize {
        if (self.fileWithNoHits()) |i| {
            const f = self.files.items[i];
            std.log.err("lora: {s} patches none of this checkpoint's linears ({d} targets in the file)", .{ f.path, f.side.count() });
            return error.NoMatchingTargets;
        }
        var total: usize = 0;
        for (self.files.items, 0..) |f, i| {
            if (f.side.unclaimed > 0) {
                std.log.warn("lora: {d} keys in {s} use a spelling this loader does not know", .{ f.side.unclaimed, f.path });
            }
            total += self.hits[i];
        }
        return total;
    }

    /// The first file the walk found no linear for, or null. Split from the
    /// reporting for the same reason as `Sidecar.refusedSuffix`.
    pub fn fileWithNoHits(self: *const Stack) ?usize {
        for (self.files.items, 0..) |_, i| if (self.hits[i] == 0) return i;
        return null;
    }

    /// The host apply for every hit against `w`, in file order.
    pub fn applyHost(
        self: *const Stack,
        io: std.Io,
        gpa: std.mem.Allocator,
        y: []f32,
        y_stride: usize,
        x: []const f32,
        m: usize,
        w: Weight,
    ) !void {
        for (self.forWeight(w)) |h| {
            const s = self.files.items[h.file].strength;
            // Exact: adding `0 * delta` changes nothing, so a dial at zero is
            // free rather than two GEMMs whose result is discarded.
            if (s == 0) continue;
            try h.target.applyHost(io, gpa, y, y_stride, x, m, s);
        }
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
};

/// Files one stack can hold. A dial per file is a GUI row, so the cap is what
/// fits on screen rather than anything the algebra needs.
pub const max_files: usize = 16;

fn loadTarget(
    alloc: std.mem.Allocator,
    store: WeightStore,
    full_stem: []const u8,
    stem: []const u8,
    d: Dialect,
) !Target {
    var buf: [512]u8 = undefined;

    const a_view = store.get(try std.fmt.bufPrint(&buf, "{s}{s}", .{ full_stem, d.a })) orelse return error.MissingTensor;
    const b_view = store.get(try std.fmt.bufPrint(&buf, "{s}{s}", .{ full_stem, d.b })) orelse {
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
    if (store.get(try std.fmt.bufPrint(&buf, "{s}.alpha", .{full_stem}))) |av| {
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
    const dst = try alloc.alloc(u8, nr * nc * es);
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
        var side = try Sidecar.load(gpa, store);
        defer side.deinit();
        try testing.expectEqual(targets.len, side.count());

        for (targets) |spec| {
            var buf: [128]u8 = undefined;
            const base_name = try std.fmt.bufPrint(&buf, "base.{s}.weight", .{spec.name});
            const bv = store.get(base_name) orelse return error.MissingTensor;
            const bs = bv.info.shape.slice();
            const out_dim = bs[0];
            const in_dim = bs[1];

            var w = Weight.init(bv.bytes, bv.info.dtype, out_dim, in_dim);
            // `forWeight` keys on the checkpoint name, so the fixture's base
            // tensors are looked up under the name a loader would have set.
            const tag = try std.fmt.allocPrint(gpa, "{s}.weight", .{spec.name});
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

            const in_name = try std.fmt.bufPrint(&buf, "in.{s}", .{spec.name});
            const x = try (store.get(in_name) orelse return error.MissingTensor).toF32Alloc(gpa);
            defer gpa.free(x);
            const m = x.len / in_dim;

            const want_prefix: []const u8 = if (strength == 1.0) "out" else "out_half";
            const want_name = try std.fmt.bufPrint(&buf, "{s}.{s}", .{ want_prefix, spec.name });
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
            const bare_name = try std.fmt.bufPrint(&buf, "out_base.{s}", .{spec.name});
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
    var side = try Sidecar.load(gpa, store);
    defer side.deinit();

    const bytes = [_]u8{0} ** (16 * 9 * 4);
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
    try testing.expectEqual(@as(usize, 3), try stack.add(gpa, store, s1, "file1"));
    try testing.expectEqual(@as(usize, 3), try stack.add(gpa, pfx.store(), s2, "file2"));

    for ([_][]const u8{ "blocks.0.mlp.fc2", "blocks.0.attn.qkv_proj", "blocks.0.attn.out_proj" }) |name| {
        var buf: [128]u8 = undefined;
        const bv = store.get(try std.fmt.bufPrint(&buf, "base.{s}.weight", .{name})) orelse return error.MissingTensor;
        const out_dim = bv.info.shape.slice()[0];
        const in_dim = bv.info.shape.slice()[1];

        var w = Weight.init(bv.bytes, bv.info.dtype, out_dim, in_dim);
        const tag = try std.fmt.allocPrint(gpa, "{s}.weight", .{name});
        defer gpa.free(tag);
        w.tag = tag;

        // Both files patch this weight, and the walk must say so.
        stack.beginAttach();
        try testing.expectEqual(@as(usize, 2), (try stack.resolve(w)).len);

        const x = try (store.get(try std.fmt.bufPrint(&buf, "in.{s}", .{name})) orelse return error.MissingTensor).toF32Alloc(gpa);
        defer gpa.free(x);
        const m = x.len / in_dim;
        const want = try (store.get(try std.fmt.bufPrint(&buf, "out_stack.{s}", .{name})) orelse return error.MissingTensor).toF32Alloc(gpa);
        defer gpa.free(want);
        const bare = try (store.get(try std.fmt.bufPrint(&buf, "out_base.{s}", .{name})) orelse return error.MissingTensor).toF32Alloc(gpa);
        defer gpa.free(bare);

        const got = try gpa.alloc(f32, m * out_dim);
        defer gpa.free(got);
        try ops.matmul.matmul(io, gpa, got, x, m, w, null);
        try stack.applyHost(io, gpa, got, out_dim, x, m, w);

        const rel = relL2(want, got);
        errdefer std.debug.print("{s}: stack rel {e}\n", .{ name, rel });
        try testing.expect(rel < 1e-6);
        // Both files moved it, so neither dial is being dropped.
        try testing.expect(relL2(want, bare) > 0.05);

        // The dial is live: zeroing file 2 leaves file 1's delta alone, with no
        // reload anywhere. `out` is file 1 at strength 1, so its delta scales.
        const only1 = try (store.get(try std.fmt.bufPrint(&buf, "out.{s}", .{name})) orelse return error.MissingTensor).toF32Alloc(gpa);
        defer gpa.free(only1);
        const want1 = try gpa.alloc(f32, m * out_dim);
        defer gpa.free(want1);
        for (want1, bare, only1) |*d, b, o| d.* = b + s1 * (o - b);

        stack.files.items[1].strength = 0;
        @memset(got, 0);
        try ops.matmul.matmul(io, gpa, got, x, m, w, null);
        try stack.applyHost(io, gpa, got, out_dim, x, m, w);
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
    _ = try stack.add(gpa, store, 1.0, "file1");

    const bytes = [_]u8{0} ** 16;
    var other = Weight.init(&bytes, .f32, 2, 2);
    other.tag = "some.other.arch.weight";

    stack.beginAttach();
    _ = try stack.resolve(other);
    // The predicate, not `finishAttach`: the latter reports through `std.log.err`,
    // and there is no test log level below it, so asserting the error itself
    // would make every passing run print.
    try testing.expectEqual(@as(?usize, 0), stack.fileWithNoHits());
}

test "a DoRA or conv-CP adapter is refused by name, not read as a plain LoRA" {
    // Both are a different calculation: DoRA rescales the base weight and
    // `lora_mid` is a three-matrix decomposition. Reading either as a plain
    // LoRA renders finitely, plausibly and wrongly.
    // The predicate rather than `load`, which reports through `std.log.err`.
    try testing.expectEqualStrings(".dora_scale", Sidecar.refusedSuffix("diffusion_model.blocks.0.mlp.fc2.dora_scale").?);
    try testing.expectEqualStrings(".lora_mid.weight", Sidecar.refusedSuffix("diffusion_model.blocks.0.attn.qkv.lora_mid.weight").?);
    try testing.expectEqualStrings(".reshape_weight", Sidecar.refusedSuffix("diffusion_model.x.reshape_weight").?);
    // A plain LoRA's own keys are not refused, or nothing would load at all.
    for (dialects) |d| {
        try testing.expect(Sidecar.refusedSuffix(d.a) == null);
        try testing.expect(Sidecar.refusedSuffix(d.b) == null);
    }
    try testing.expect(Sidecar.refusedSuffix("diffusion_model.blocks.0.mlp.fc2.alpha") == null);
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
