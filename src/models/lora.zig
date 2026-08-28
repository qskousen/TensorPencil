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
//!    file's own shape. Deriving `s` before any splitting is what makes that
//!    impossible to get wrong, and it is why `split` takes the scale rather than
//!    computing one.
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
//! Key convention is ComfyUI's generic LoRA format (`comfy/lora.py`): a base
//! tensor `blocks.3.attn.qkv_proj.weight` is patched by
//! `diffusion_model.blocks.3.attn.qkv_proj.{lora_A.weight, lora_B.weight, alpha}`.
//! `Sidecar.forWeight` does that name transform, so a model loader passes the
//! `Weight.tag` it already has.

const std = @import("std");
const ops = @import("tp_ops");
const weights_mod = @import("tp_core").weights;

const Weight = ops.matmul.Weight;
const WeightStore = weights_mod.WeightStore;
const DType = @import("tp_core").dtype.DType;

/// Rows the host apply processes at a time. Bounds the `[rows][out]`
/// intermediate: at the default H3 render an untiled one would be 38k x 21504 x
/// 4 B = 3.3 GB for a delta. Rows are independent, so this is free.
const host_band: usize = 512;

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
    /// `strength * alpha / rank`, folded once at attach time.
    scale: f32,

    pub fn rank(f: Factor) usize {
        return f.a.rows;
    }
};

/// The sidecar for one linear.
pub const Target = struct {
    factors: []const Factor,
    in_dim: usize,
    out_dim: usize,
    /// The base tensor's name, for diagnostics.
    tag: []const u8,

    /// FLOPs one apply costs over `m` rows, for reporting the overhead honestly.
    pub fn flops(t: *const Target, m: usize) usize {
        var n: usize = 0;
        for (t.factors) |f| n += 2 * m * f.rank() * (t.in_dim + f.b.rows);
        return n;
    }

    /// `y[m][y_stride] += s * B (A x)`, factor `f` writing columns
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

        var r0: usize = 0;
        while (r0 < m) : (r0 += band) {
            const n = @min(band, m - r0);
            for (t.factors) |f| {
                const r = f.rank();
                const o = f.b.rows;
                try ops.matmul.matmul(io, gpa, lo[0 .. n * r], x[r0 * t.in_dim ..][0 .. n * t.in_dim], n, f.a, null);
                try ops.matmul.matmul(io, gpa, hi[0 .. n * o], lo[0 .. n * r], n, f.b, null);
                for (0..n) |i| {
                    const dst = y[(r0 + i) * y_stride + f.out_off ..][0..o];
                    const src = hi[i * o ..][0..o];
                    for (dst, src) |*d, s| d.* += f.scale * s;
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
    /// Denoiser-prefixed keys whose SPELLING this loader does not know (the
    /// kohya `lora_up`/`lora_down` dialect, say). A file made entirely of those
    /// loads as an empty sidecar, which would otherwise render as if no LoRA had
    /// been asked for. A LoRA for a different architecture is a separate case,
    /// caught by nothing attaching.
    unclaimed: usize,

    pub fn deinit(self: *Sidecar) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// The prefix ComfyUI's generic format puts on a denoiser tensor name.
    pub const prefix = "diffusion_model.";

    /// Load every `lora_A`/`lora_B` pair in `store`, keyed by the base tensor
    /// name it patches.
    ///
    /// `strength` is folded into each factor's scale here, so changing it means
    /// reloading. That is deliberate: nothing downstream then holds a second
    /// copy of the dial that could disagree with this one.
    pub fn load(gpa: std.mem.Allocator, store: WeightStore, strength: f32) !Sidecar {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();

        var index: std.StringHashMapUnmanaged(Target) = .empty;
        const names = store.names();

        const a_suffix = ".lora_A.weight";
        var unclaimed: usize = 0;
        for (names) |full| {
            // Only denoiser keys are this loader's business; anything else in
            // the file belongs to another component (a text-encoder LoRA) and is
            // not evidence of a problem.
            if (!std.mem.startsWith(u8, full, prefix)) continue;
            var known = false;
            inline for (.{ a_suffix, ".lora_B.weight", ".alpha" }) |sfx| {
                if (std.mem.endsWith(u8, full, sfx)) known = true;
            }
            if (!known) unclaimed += 1;
            if (!std.mem.endsWith(u8, full, a_suffix)) continue;
            const stem = full[prefix.len .. full.len - a_suffix.len];

            const t = try loadTarget(alloc, store, full[0 .. full.len - a_suffix.len], stem, strength);
            const base = try std.fmt.allocPrint(alloc, "{s}.weight", .{stem});
            try index.put(alloc, base, t);
        }

        return .{
            .arena = arena,
            .index = index,
            .unclaimed = unclaimed,
        };
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

fn loadTarget(
    alloc: std.mem.Allocator,
    store: WeightStore,
    full_stem: []const u8,
    stem: []const u8,
    strength: f32,
) !Target {
    var buf: [256]u8 = undefined;

    const a_view = store.get(try std.fmt.bufPrint(&buf, "{s}.lora_A.weight", .{full_stem})) orelse return error.MissingTensor;
    const b_view = store.get(try std.fmt.bufPrint(&buf, "{s}.lora_B.weight", .{full_stem})) orelse {
        std.log.err("lora: {s} has a lora_A but no lora_B", .{stem});
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
        const vals = try av.toF32Alloc(alloc);
        defer alloc.free(vals);
        if (vals.len != 1) {
            std.log.err("lora: {s}.alpha has {d} entries, expected 1", .{ stem, vals.len });
            return error.ShapeMismatch;
        }
        alpha = vals[0];
    }
    const scale = strength * alpha / @as(f32, @floatFromInt(rank));

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
        factors[0] = .{ .a = a_full, .b = b_full, .out_off = 0, .scale = scale };
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
                .scale = scale,
            };
        }
    }

    return .{
        .factors = factors,
        .in_dim = in_dim,
        .out_dim = out_dim,
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
        .{ .name = "blocks.0.attn.out_proj", .groups = 1 },
    };

    for ([_]f32{ 1.0, 0.5 }) |strength| {
        var side = try Sidecar.load(gpa, store, strength);
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
            try t.applyHost(io, gpa, got, out_dim, x, m);

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
    var side = try Sidecar.load(gpa, store, 1.0);
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
        .factors = &.{.{ .a = a_w, .b = b_w, .out_off = 0, .scale = scale }},
        .in_dim = in_dim,
        .out_dim = g * out_g,
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
            .scale = scale,
        };
    }
    const split: Target = .{
        .factors = &split_factors,
        .in_dim = in_dim,
        .out_dim = g * out_g,
        .tag = "split",
    };

    try testing.expectEqual(@as(usize, 3), blockGroups(b_w));

    var y_dense: [4 * g * out_g]f32 = @splat(0);
    var y_split: [4 * g * out_g]f32 = @splat(0);
    try dense.applyHost(testing.io, testing.allocator, &y_dense, g * out_g, &x, 4);
    try split.applyHost(testing.io, testing.allocator, &y_split, g * out_g, &x, 4);
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
            .scale = 1.0,
        }},
        .in_dim = in_dim,
        .out_dim = 6,
        .tag = "mid",
    };
    const x = [_]f32{ 1, 9, 2, 9 };
    var y: [2 * 6]f32 = @splat(0);
    try t.applyHost(testing.io, testing.allocator, &y, 6, &x, 2);
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
        .factors = &.{.{ .a = Weight.fromF32(&a, 1, 1), .b = Weight.fromF32(&b, 1, 1), .out_off = 0, .scale = 2.0 }},
        .in_dim = 1,
        .out_dim = 1,
        .tag = "acc",
    };
    var y = [_]f32{100};
    const x = [_]f32{3};
    try t.applyHost(testing.io, testing.allocator, &y, 1, &x, 1);
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
            .scale = 0.5,
        }},
        .in_dim = in_dim,
        .out_dim = out_dim,
        .tag = "band",
    };

    const x = try testing.allocator.alloc(f32, m * in_dim);
    defer testing.allocator.free(x);
    for (x) |*v| v.* = rnd.floatNorm(f32);
    const y = try testing.allocator.alloc(f32, m * out_dim);
    defer testing.allocator.free(y);
    @memset(y, 0);
    try t.applyHost(testing.io, testing.allocator, y, out_dim, x, m);

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
