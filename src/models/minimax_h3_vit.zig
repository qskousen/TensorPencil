//! MiniMax H3's vision tower: Qwen3-VL-32B's ViT with DeepStack.
//!
//! A separate file from `vit35.zig`, which implements the same tower on paper (27
//! blocks, 1152 wide, fused qkv, 4304 FFN, 48x48 learned position table, and even
//! the same merger norm under llama.cpp's `v.post_ln` name). That one was ported
//! from llama.cpp's clip.cpp, and TWO conventions diverge from the HF lineage this
//! checkpoint is:
//!
//! 1. **The position-embedding interpolation.** llama.cpp resamples the 48x48
//!    table with `BILINEAR | ANTIALIAS`, `align_corners = false`. The reference
//!    takes `linspace(0, 47, n)` with plain floor/ceil bilinear weights, i.e.
//!    align_corners TRUE and no antialias. Different numbers at every grid that is
//!    not natively 48x48, which is every real image.
//! 2. **The patch embedding's temporal pair.** `temporal_patch_size` is 2, and
//!    `vit35` SUMS the two 16x16 convolutions -- right only when both temporal
//!    slots hold the same still. H3's reference VIDEOS feed two DISTINCT frames per
//!    block, so the sum is wrong there. Here the flattened patch already interleaves
//!    both frames in `(channel, temporal, ky, kx)` order and the embedding is one
//!    GEMM against the Conv3d weight, which handles either case.
//!
//! The rope agrees between the two lineages and `vit35.applyVisionRope` is reused:
//! split-half over pairs `(d, d + head_dim/2)`, the first quarter of the head keyed
//! by the patch ROW and the second by the patch COLUMN.
//!
//! Conventions that are silent wrong answers, pinned by
//! `tools/gen_minimax_h3_vit.py`:
//!
//! - ⚠️ **The two mergers normalize at DIFFERENT widths, and it is one set of
//!   parentheses.** The main merger is `norm(x).view(-1, merge_dim)`, LayerNorm over
//!   the PRE-merge hidden; a deepstack merger is `norm(x.view(-1, merge_dim))`,
//!   LayerNorm over the POST-merge width. The checkpoint says so out loud
//!   (`merger.norm` is `[1152]`, `deepstack_merger_list.N.norm` is `[4608]`).
//! - ⚠️ **The block MLP is gelu-TANH; both mergers are erf gelu.** Same activation
//!   name, two different functions, and the difference is small enough to look like
//!   noise.
//! - DeepStack features come from the OUTPUT of blocks `deepstack_indexes`
//!   (`[8, 16, 24]` here) and are injected into LLM decoder layers `0, 1, 2` at
//!   image-token positions only. Collected deep, injected shallow.
//! - Token order is the 2x2-MERGED block order throughout
//!   `(block_row, block_col, intra_row, intra_col)`; the patch flattening, the
//!   position interpolation and the rope all agree on it, and they have to.

const std = @import("std");
const tp_core = @import("tp_core");
const weights_mod = tp_core.weights;
const ops = @import("tp_ops");
const vit35 = @import("vit35.zig");

const WeightStore = weights_mod.WeightStore;
const Weight = ops.matmul.Weight;

/// DeepStack merger count. Fixed at three across every Qwen3-VL size.
pub const max_deepstack = 4;

pub const Config = struct {
    dim: usize,
    n_heads: usize,
    ffn: usize,
    n_blocks: usize,
    patch: usize,
    temporal_patch: usize,
    merge: usize,
    in_channels: usize,
    /// Side of the learned position grid; `num_position_embeddings` is its square.
    pos_grid: usize,
    /// The LLM's hidden width, i.e. what both merger kinds project to.
    out_dim: usize,
    n_deepstack: usize,
    deepstack_indexes: [max_deepstack]usize,
    eps: f32 = 1e-6,
    rope_theta: f32 = 10000.0,

    pub fn headDim(self: Config) usize {
        return self.dim / self.n_heads;
    }

    /// Width one merged token carries before the merger's projections.
    pub fn mergeDim(self: Config) usize {
        return self.dim * self.merge * self.merge;
    }

    /// Values one patch contributes to the embedding GEMM: the flattened Conv3d
    /// kernel, in `(channel, temporal, ky, kx)` order.
    pub fn patchDim(self: Config) usize {
        return self.in_channels * self.temporal_patch * self.patch * self.patch;
    }

    /// Rope frequencies per axis. The rotary width is `head_dim / 2` and the two
    /// axes split it, so each gets `head_dim / 4`.
    pub fn freqsPerAxis(self: Config) usize {
        return self.headDim() / 4;
    }

    /// The 32B (and 8B) tower. 4B differs in width, depth and indexes.
    pub const qwen3vl_32b: Config = .{
        .dim = 1152,
        .n_heads = 16,
        .ffn = 4304,
        .n_blocks = 27,
        .patch = 16,
        .temporal_patch = 2,
        .merge = 2,
        .in_channels = 3,
        .pos_grid = 48,
        .out_dim = 5120,
        .n_deepstack = 3,
        .deepstack_indexes = .{ 8, 16, 24, 0 },
    };
};

pub const Block = struct {
    norm1_w: []f32,
    norm1_b: []f32,
    norm2_w: []f32,
    norm2_b: []f32,
    qkv: Weight,
    qkv_b: []f32,
    proj: Weight,
    proj_b: []f32,
    fc1: Weight,
    fc1_b: []f32,
    fc2: Weight,
    fc2_b: []f32,
};

/// One merger. `post_merge_norm` is what distinguishes the two kinds: the main
/// merger norms the PRE-merge hidden and a deepstack merger the POST-merge width.
pub const Merger = struct {
    norm_w: []f32,
    norm_b: []f32,
    fc1: Weight,
    fc1_b: []f32,
    fc2: Weight,
    fc2_b: []f32,
    post_merge_norm: bool,
};

pub const Vit = struct {
    arena: std.heap.ArenaAllocator,
    cfg: Config,
    /// `[dim][patchDim]`, the Conv3d kernel flattened.
    patch_w: Weight,
    patch_b: []f32,
    /// `[pos_grid * pos_grid][dim]`.
    pos_embed: []f32,
    blocks: []Block,
    merger: Merger,
    deepstack: []Merger,

    pub fn deinit(self: *Vit) void {
        self.arena.deinit();
        self.* = undefined;
    }

    pub fn load(gpa: std.mem.Allocator, store: WeightStore, cfg: Config) !Vit {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();
        const l: L = .{ .alloc = alloc, .store = store };

        const blocks = try alloc.alloc(Block, cfg.n_blocks);
        for (blocks, 0..) |*b, i| {
            b.* = .{
                .norm1_w = try l.vec("blocks.{d}.norm1.weight", .{i}, cfg.dim),
                .norm1_b = try l.vec("blocks.{d}.norm1.bias", .{i}, cfg.dim),
                .norm2_w = try l.vec("blocks.{d}.norm2.weight", .{i}, cfg.dim),
                .norm2_b = try l.vec("blocks.{d}.norm2.bias", .{i}, cfg.dim),
                .qkv = try l.mat("blocks.{d}.attn.qkv.weight", .{i}, 3 * cfg.dim, cfg.dim),
                .qkv_b = try l.vec("blocks.{d}.attn.qkv.bias", .{i}, 3 * cfg.dim),
                .proj = try l.mat("blocks.{d}.attn.proj.weight", .{i}, cfg.dim, cfg.dim),
                .proj_b = try l.vec("blocks.{d}.attn.proj.bias", .{i}, cfg.dim),
                .fc1 = try l.mat("blocks.{d}.mlp.linear_fc1.weight", .{i}, cfg.ffn, cfg.dim),
                .fc1_b = try l.vec("blocks.{d}.mlp.linear_fc1.bias", .{i}, cfg.ffn),
                .fc2 = try l.mat("blocks.{d}.mlp.linear_fc2.weight", .{i}, cfg.dim, cfg.ffn),
                .fc2_b = try l.vec("blocks.{d}.mlp.linear_fc2.bias", .{i}, cfg.dim),
            };
        }

        const deepstack = try alloc.alloc(Merger, cfg.n_deepstack);
        for (deepstack, 0..) |*m, i| {
            m.* = try l.merger("deepstack_merger_list.{d}", .{i}, cfg, true);
        }

        // Every arena allocation before `.arena = arena` snapshots it.
        const patch_w = try l.mat("patch_embed.proj.weight", .{}, cfg.dim, cfg.patchDim());
        const patch_b = try l.vec("patch_embed.proj.bias", .{}, cfg.dim);
        const pos_embed = try l.vec("pos_embed.weight", .{}, cfg.pos_grid * cfg.pos_grid * cfg.dim);
        const merger = try l.merger("merger", .{}, cfg, false);

        return .{
            .arena = arena,
            .cfg = cfg,
            .patch_w = patch_w,
            .patch_b = patch_b,
            .pos_embed = pos_embed,
            .blocks = blocks,
            .merger = merger,
            .deepstack = deepstack,
        };
    }
};

const L = struct {
    alloc: std.mem.Allocator,
    store: WeightStore,

    fn view(s: L, comptime fmt: []const u8, args: anytype) !weights_mod.TensorView {
        var buf: [160]u8 = undefined;
        const nm = try std.fmt.bufPrint(&buf, fmt, args);
        return s.store.get(nm) orelse {
            std.log.err("minimax_h3_vit: missing tensor {s}", .{nm});
            return error.MissingTensor;
        };
    }

    fn vec(s: L, comptime fmt: []const u8, args: anytype, len: usize) ![]f32 {
        const v = try s.view(fmt, args);
        if (v.info.elemCount() != len) {
            std.log.err("minimax_h3_vit: a tensor has {d} elements, expected {d}", .{ v.info.elemCount(), len });
            return error.ShapeMismatch;
        }
        return v.toF32Alloc(s.alloc);
    }

    /// A 2-D weight, materialized to f32. The tower is bf16 in the checkpoint and
    /// small enough (a few hundred MB) that keeping it packed buys nothing here.
    fn mat(s: L, comptime fmt: []const u8, args: anytype, rows: usize, cols: usize) !Weight {
        const v = try s.view(fmt, args);
        if (v.info.elemCount() != rows * cols) {
            std.log.err("minimax_h3_vit: a weight has {d} elements, expected {d}x{d}", .{ v.info.elemCount(), rows, cols });
            return error.ShapeMismatch;
        }
        return Weight.fromF32(try v.toF32Alloc(s.alloc), rows, cols);
    }

    fn merger(s: L, comptime fmt: []const u8, args: anytype, cfg: Config, post_merge: bool) !Merger {
        const md = cfg.mergeDim();
        // The one asymmetry: a deepstack merger's LayerNorm is the POST-merge
        // width, the main merger's the PRE-merge one. Reading the width from the
        // checkpoint rather than assuming makes a swapped pair a load error.
        const norm_dim = if (post_merge) md else cfg.dim;
        return .{
            .norm_w = try s.vec(fmt ++ ".norm.weight", args, norm_dim),
            .norm_b = try s.vec(fmt ++ ".norm.bias", args, norm_dim),
            .fc1 = try s.mat(fmt ++ ".linear_fc1.weight", args, md, md),
            .fc1_b = try s.vec(fmt ++ ".linear_fc1.bias", args, md),
            .fc2 = try s.mat(fmt ++ ".linear_fc2.weight", args, cfg.out_dim, md),
            .fc2_b = try s.vec(fmt ++ ".linear_fc2.bias", args, cfg.out_dim),
            .post_merge_norm = post_merge,
        };
    }
};

// --- position embedding ---------------------------------------------------

/// The reference's `fast_pos_embed_interpolate`: bilinear over the learned
/// `pos_grid x pos_grid` table at `linspace(0, pos_grid - 1, n)` sample points.
///
/// ⚠️ This is align_corners=TRUE and there is NO antialiasing, which is what
/// separates it from `vit35`'s ggml resampler. The endpoints land exactly on the
/// table's first and last row, and a target grid coarser than the table simply
/// point-samples with fractional weights rather than averaging a footprint.
///
/// `out` is `[gh * gw][dim]` in ROW-MAJOR patch order; the merge permutation
/// happens afterwards.
pub fn interpolatePos(out: []f32, pos: []const f32, pos_grid: usize, dim: usize, gh: usize, gw: usize) void {
    std.debug.assert(out.len == gh * gw * dim);
    const last: f64 = @floatFromInt(pos_grid - 1);

    for (0..gh) |i| {
        // linspace(0, pos_grid - 1, gh): a single sample sits at 0, not at the
        // centre, which is the align_corners convention.
        const fy: f64 = if (gh == 1) 0 else last * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(gh - 1));
        const y0: usize = @intFromFloat(fy);
        const y1 = @min(y0 + 1, pos_grid - 1);
        const dy: f32 = @floatCast(fy - @as(f64, @floatFromInt(y0)));
        for (0..gw) |j| {
            const fx: f64 = if (gw == 1) 0 else last * @as(f64, @floatFromInt(j)) / @as(f64, @floatFromInt(gw - 1));
            const x0: usize = @intFromFloat(fx);
            const x1 = @min(x0 + 1, pos_grid - 1);
            const dx: f32 = @floatCast(fx - @as(f64, @floatFromInt(x0)));

            const w00 = (1 - dy) * (1 - dx);
            const w01 = (1 - dy) * dx;
            const w10 = dy * (1 - dx);
            const w11 = dy * dx;
            const r00 = pos[(y0 * pos_grid + x0) * dim ..][0..dim];
            const r01 = pos[(y0 * pos_grid + x1) * dim ..][0..dim];
            const r10 = pos[(y1 * pos_grid + x0) * dim ..][0..dim];
            const r11 = pos[(y1 * pos_grid + x1) * dim ..][0..dim];
            const dst = out[(i * gw + j) * dim ..][0..dim];
            for (dst, r00, r01, r10, r11) |*d, a, b, c, e| {
                d.* = a * w00 + b * w01 + c * w10 + e * w11;
            }
        }
    }
}

/// Row-major patch order -> the 2x2-merged block order every other stage speaks:
/// `(block_row, block_col, intra_row, intra_col)`.
pub fn mergeOrder(out: []f32, x: []const f32, dim: usize, gh: usize, gw: usize, merge: usize) void {
    std.debug.assert(gh % merge == 0 and gw % merge == 0);
    const bh = gh / merge;
    const bw = gw / merge;
    var o: usize = 0;
    for (0..bh) |br| {
        for (0..bw) |bc| {
            for (0..merge) |ir| {
                for (0..merge) |ic| {
                    const src = ((br * merge + ir) * gw + bc * merge + ic) * dim;
                    @memcpy(out[o * dim ..][0..dim], x[src..][0..dim]);
                    o += 1;
                }
            }
        }
    }
}

/// Per-token patch row and column, in merged block order. These are the rope's
/// two axes.
pub fn patchCoords(py: []u32, px: []u32, gh: usize, gw: usize, merge: usize) void {
    const bh = gh / merge;
    const bw = gw / merge;
    std.debug.assert(py.len == gh * gw and px.len == gh * gw);
    var o: usize = 0;
    for (0..bh) |br| {
        for (0..bw) |bc| {
            for (0..merge) |ir| {
                for (0..merge) |ic| {
                    py[o] = @intCast(br * merge + ir);
                    px[o] = @intCast(bc * merge + ic);
                    o += 1;
                }
            }
        }
    }
}

// --- image preprocessing --------------------------------------------------

/// The patch grid an image of `(h, w)` resizes to, and the resized extent.
///
/// The reference STRETCHES to `round(dim / factor) * factor` per axis, with
/// `factor = patch * merge`. It does not preserve the aspect ratio and it does not
/// pad -- `vit35`'s llama.cpp-derived policy does both, which is a third
/// divergence between the two lineages on top of the position interpolation and
/// the temporal patch.
///
/// `round` is nearest, so an axis can move in EITHER direction: 100 -> 96 down
/// while 150 -> 160 up. A floor would quietly always shrink.
pub fn resizeTarget(cfg: Config, h: usize, w: usize, min_pixels: usize, max_pixels: usize) struct { h: usize, w: usize } {
    const factor = cfg.patch * cfg.merge;
    const ff: f64 = @floatFromInt(factor);
    var hb = @max(factor, roundToFactor(@floatFromInt(h), ff));
    var wb = @max(factor, roundToFactor(@floatFromInt(w), ff));
    const area: f64 = @floatFromInt(h * w);
    if (hb * wb > max_pixels) {
        const beta = @sqrt(area / @as(f64, @floatFromInt(max_pixels)));
        hb = @max(factor, floorToFactor(@as(f64, @floatFromInt(h)) / beta, ff));
        wb = @max(factor, floorToFactor(@as(f64, @floatFromInt(w)) / beta, ff));
    } else if (hb * wb < min_pixels) {
        const beta = @sqrt(@as(f64, @floatFromInt(min_pixels)) / area);
        hb = @max(factor, ceilToFactor(@as(f64, @floatFromInt(h)) * beta, ff));
        wb = @max(factor, ceilToFactor(@as(f64, @floatFromInt(w)) * beta, ff));
    }
    return .{ .h = hb, .w = wb };
}

/// Python's `round`, which is banker's rounding (half to EVEN), not half-up.
/// `round(2.5)` is 2 there and 3 in most other languages, and an axis exactly
/// half a factor over the grid is the case that differs.
fn pyRound(v: f64) f64 {
    const r = @round(v);
    if (@abs(v - @trunc(v)) == 0.5) {
        // `@round` goes away from zero on a tie; Python goes to even.
        const t = @trunc(v);
        return if (@mod(t, 2.0) == 0.0) t else t + std.math.sign(v);
    }
    return r;
}

fn roundToFactor(v: f64, factor: f64) usize {
    return @intFromFloat(pyRound(v / factor) * factor);
}
fn floorToFactor(v: f64, factor: f64) usize {
    return @intFromFloat(@floor(v / factor) * factor);
}
fn ceilToFactor(v: f64, factor: f64) usize {
    return @intFromFloat(@ceil(v / factor) * factor);
}

/// Bilinear resample with `align_corners = false` and NO antialiasing, which is
/// what `F.interpolate(mode="bilinear", align_corners=False)` does.
///
/// `src` is planar `[3][sh][sw]`, `dst` planar `[3][dh][dw]`. align_corners=false
/// puts the sample at `(i + 0.5) * scale - 0.5`, i.e. pixel CENTRES; the
/// align_corners=true form `i * (n-1)/(m-1)` is `vit35`'s and shifts everything by
/// half a pixel at every scale but 1.
pub fn resizeBilinear(dst: []f32, src: []const f32, sh: usize, sw: usize, dh: usize, dw: usize) void {
    std.debug.assert(src.len == 3 * sh * sw and dst.len == 3 * dh * dw);
    const ys: f64 = @as(f64, @floatFromInt(sh)) / @as(f64, @floatFromInt(dh));
    const xs: f64 = @as(f64, @floatFromInt(sw)) / @as(f64, @floatFromInt(dw));
    for (0..dh) |i| {
        const fy = @max(0.0, (@as(f64, @floatFromInt(i)) + 0.5) * ys - 0.5);
        const y0: usize = @min(sh - 1, @as(usize, @intFromFloat(fy)));
        const y1 = @min(sh - 1, y0 + 1);
        const dy: f32 = @floatCast(fy - @as(f64, @floatFromInt(y0)));
        for (0..dw) |j| {
            const fx = @max(0.0, (@as(f64, @floatFromInt(j)) + 0.5) * xs - 0.5);
            const x0: usize = @min(sw - 1, @as(usize, @intFromFloat(fx)));
            const x1 = @min(sw - 1, x0 + 1);
            const dx: f32 = @floatCast(fx - @as(f64, @floatFromInt(x0)));
            for (0..3) |c| {
                const p = src[c * sh * sw ..];
                const v00 = p[y0 * sw + x0];
                const v01 = p[y0 * sw + x1];
                const v10 = p[y1 * sw + x0];
                const v11 = p[y1 * sw + x1];
                const top = v00 + (v01 - v00) * dx;
                const bot = v10 + (v11 - v10) * dx;
                dst[c * dh * dw + i * dw + j] = top + (bot - top) * dy;
            }
        }
    }
}

/// A preprocessed reference image: the flattened patch matrix the tower takes,
/// plus its patch grid.
pub const Prepared = struct {
    /// `[gh * gw][patchDim]`, in MERGED block order.
    patches: []f32,
    grid_h: usize,
    grid_w: usize,

    pub fn deinit(self: *Prepared, gpa: std.mem.Allocator) void {
        gpa.free(self.patches);
        self.* = undefined;
    }

    pub fn tokens(self: Prepared) usize {
        return (self.grid_h / 2) * (self.grid_w / 2);
    }
};

/// `process_qwen2vl_images` for one still: resize, normalize to [-1, 1], and
/// flatten to the patch matrix.
///
/// `rgb` is planar `[3][h][w]` in [0, 1]. Two things are load-bearing:
///
/// - the normalization is mean/std 0.5, i.e. `[0,1] -> [-1,1]`, NOT the CLIP
///   statistics Qwen2.5-VL uses;
/// - a STILL is repeated across the temporal patch, so both temporal slots hold
///   the same frame. That is the case a summed patch embedding would also get
///   right; a reference VIDEO's two distinct frames is the case it would not.
pub fn preprocessStill(
    gpa: std.mem.Allocator,
    cfg: Config,
    rgb: []const f32,
    h: usize,
    w: usize,
    min_pixels: usize,
    max_pixels: usize,
) !Prepared {
    const t = resizeTarget(cfg, h, w, min_pixels, max_pixels);
    const resized = try gpa.alloc(f32, 3 * t.h * t.w);
    defer gpa.free(resized);
    resizeBilinear(resized, rgb, h, w, t.h, t.w);
    for (resized) |*v| v.* = (v.* - 0.5) / 0.5;

    const gh = t.h / cfg.patch;
    const gw = t.w / cfg.patch;
    const out = try gpa.alloc(f32, gh * gw * cfg.patchDim());
    errdefer gpa.free(out);
    // Two temporal slots holding the same frame.
    flattenPatches(out, resized, resized, cfg, t.h, t.w);
    return .{ .patches = out, .grid_h = gh, .grid_w = gw };
}

/// `process_video_block` for one temporal PAIR: two DISTINCT frames filling the
/// temporal patch, rather than a still repeated.
///
/// This is the case `vit35` cannot express. It sums the two 16x16 convolutions of
/// the patch embedding, which is exactly right when both temporal slots hold the
/// same frame and wrong here, where they hold consecutive frames of a reference
/// video. Everything else matches `preprocessStill`.
///
/// `f0` and `f1` are planar `[3][h][w]` in [0, 1] at the SAME size, already at the
/// reference video's canvas.
pub fn preprocessPair(
    gpa: std.mem.Allocator,
    cfg: Config,
    f0: []const f32,
    f1: []const f32,
    h: usize,
    w: usize,
    min_pixels: usize,
    max_pixels: usize,
) !Prepared {
    const t = resizeTarget(cfg, h, w, min_pixels, max_pixels);
    const r0 = try gpa.alloc(f32, 3 * t.h * t.w);
    defer gpa.free(r0);
    const r1 = try gpa.alloc(f32, 3 * t.h * t.w);
    defer gpa.free(r1);
    resizeBilinear(r0, f0, h, w, t.h, t.w);
    resizeBilinear(r1, f1, h, w, t.h, t.w);
    for (r0) |*v| v.* = (v.* - 0.5) / 0.5;
    for (r1) |*v| v.* = (v.* - 0.5) / 0.5;

    const gh = t.h / cfg.patch;
    const gw = t.w / cfg.patch;
    const out = try gpa.alloc(f32, gh * gw * cfg.patchDim());
    errdefer gpa.free(out);
    flattenPatches(out, r0, r1, cfg, t.h, t.w);
    return .{ .patches = out, .grid_h = gh, .grid_w = gw };
}

/// Flatten two temporal frames into `[gh * gw][3 * 2 * patch * patch]`.
///
/// Row order is the MERGED block order `(block_row, block_col, intra_row,
/// intra_col)`; within a row the order is `(channel, temporal, ky, kx)`, which is
/// exactly the Conv3d kernel's own flattening, so the patch embedding is one GEMM
/// against it. Both orders come from the reference's single permute
/// `(0, 3, 6, 4, 7, 2, 1, 5, 8)` and neither is guessable.
pub fn flattenPatches(out: []f32, f0: []const f32, f1: []const f32, cfg: Config, h: usize, w: usize) void {
    const p = cfg.patch;
    const m = cfg.merge;
    const gh = h / p;
    const gw = w / p;
    const bh = gh / m;
    const bw = gw / m;
    const pd = cfg.patchDim();
    std.debug.assert(out.len == gh * gw * pd);

    var row: usize = 0;
    for (0..bh) |br| {
        for (0..bw) |bc| {
            for (0..m) |ir| {
                for (0..m) |ic| {
                    const py = (br * m + ir) * p;
                    const px = (bc * m + ic) * p;
                    const dst = out[row * pd ..][0..pd];
                    var k: usize = 0;
                    for (0..cfg.in_channels) |c| {
                        for (0..cfg.temporal_patch) |tt| {
                            const src = if (tt == 0) f0 else f1;
                            const plane = src[c * h * w ..];
                            for (0..p) |ky| {
                                for (0..p) |kx| {
                                    dst[k] = plane[(py + ky) * w + px + kx];
                                    k += 1;
                                }
                            }
                        }
                    }
                    row += 1;
                }
            }
        }
    }
}

// --- multimodal rope positions --------------------------------------------

/// One spliced image, as the position builder sees it: where its embedding rows
/// start, how many there are, and its PRE-merge patch grid.
pub const ImageSpan = struct {
    index: usize,
    size: usize,
    grid_h: usize,
    grid_w: usize,

    /// Positions the block consumes on the shared timeline, which is NOT its
    /// token count: `max(gh, gw) / 2`. The difference is what makes the timeline
    /// and the token index drift apart across several references.
    pub fn timelineSpan(self: ImageSpan) usize {
        return @max(self.grid_h, self.grid_w) / 2;
    }

    pub fn mergedH(self: ImageSpan) usize {
        return self.grid_h / 2;
    }
    pub fn mergedW(self: ImageSpan) usize {
        return self.grid_w / 2;
    }
};

/// The reference's `qwen2vl_mrope_position_ids`: `out` is `[3][seq]` (T, H, W).
///
/// Text runs sequentially on all three axes. Inside an image span axis 0 is
/// CONSTANT, axis 1 is the merged ROW index and axis 2 the merged COLUMN index,
/// both in row-major order over the merged tokens.
///
/// ⚠️ **The timeline and the token index drift apart.** A block occupies `size`
/// tokens but consumes `max(gh, gw) / 2` timeline positions, so text after it
/// resumes at `start + timelineSpan + offset` and `offset` accumulates
/// `timelineSpan - size` per image. With one reference the drift is invisible
/// because nothing follows it that a second block would shift; with two it moves
/// every position after the first.
///
/// `images` must be ascending by `index` and non-overlapping.
pub fn mropePositions(out: []f32, seq: usize, images: []const ImageSpan) !void {
    std.debug.assert(out.len == 3 * seq);
    if (images.len == 0) {
        // No images means no multimodal rope at all in the reference (it returns
        // no ids and the LLM falls back to a plain 1-D sequence). Filling all
        // three axes identically is that same thing, and it keeps a caller from
        // needing two paths.
        for (0..seq) |t| {
            for (0..3) |ax| out[ax * seq + t] = @floatFromInt(t);
        }
        return;
    }

    @memset(out, 0);
    var offset: isize = 0;
    var prev_end: usize = 0;
    for (images, 0..) |img, i| {
        const start = img.index;
        const end = start + img.size;
        if (start < prev_end or end > seq or img.size == 0) return error.BadImageSpan;
        if (img.grid_h % 2 != 0 or img.grid_w % 2 != 0) return error.BadImageSpan;
        if (img.mergedH() * img.mergedW() != img.size) return error.BadImageSpan;

        // Text before the FIRST image only: later images leave the run before
        // them alone, because the previous iteration's tail already wrote it.
        if (i == 0) {
            for (0..start) |t| {
                for (0..3) |ax| out[ax * seq + t] = @floatFromInt(t);
            }
        }

        const span = img.timelineSpan();
        const base: isize = @as(isize, @intCast(start)) + offset;

        // The tail, rewritten by every image; the last one wins.
        const start_next: isize = @as(isize, @intCast(span + start)) + offset;
        for (end..seq) |t| {
            const v: isize = start_next + @as(isize, @intCast(t - end));
            for (0..3) |ax| out[ax * seq + t] = @floatFromInt(v);
        }

        // Axis 0 is constant over the block; axes 1 and 2 are the merged row and
        // column, row-major.
        const mh = img.mergedH();
        const mw = img.mergedW();
        for (0..img.size) |k| {
            out[0 * seq + start + k] = @floatFromInt(base);
            out[1 * seq + start + k] = @floatFromInt(base + @as(isize, @intCast(k / mw)));
            out[2 * seq + start + k] = @floatFromInt(base + @as(isize, @intCast(k % mw)));
        }
        _ = mh;

        offset += @as(isize, @intCast(span)) - @as(isize, @intCast(img.size));
        prev_end = end;
    }
}

// --- forward --------------------------------------------------------------

/// The tower's output: the merged tokens the LLM splices in, plus one deepstack
/// feature per configured index, each the same shape.
pub const Encoded = struct {
    /// `[tokens][out_dim]`.
    merged: []f32,
    /// `n_deepstack` slices, each `[tokens][out_dim]`.
    deepstack: [][]f32,
    tokens: usize,

    pub fn deinit(self: *Encoded, gpa: std.mem.Allocator) void {
        gpa.free(self.merged);
        for (self.deepstack) |d| gpa.free(d);
        gpa.free(self.deepstack);
        self.* = undefined;
    }
};

/// `patches` is `[gh * gw][patchDim]`, already normalized and flattened in
/// `(channel, temporal, ky, kx)` order, in ROW-MAJOR patch order as
/// `process_qwen2vl_images` emits it.
pub fn encode(
    v: *const Vit,
    io: std.Io,
    gpa: std.mem.Allocator,
    patches: []const f32,
    gh: usize,
    gw: usize,
) !Encoded {
    const cfg = v.cfg;
    const np = gh * gw;
    const dim = cfg.dim;
    const hd = cfg.headDim();
    std.debug.assert(patches.len == np * cfg.patchDim());
    if (gh % cfg.merge != 0 or gw % cfg.merge != 0) return error.UnsupportedShape;

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    // Patch embedding: one GEMM against the flattened Conv3d kernel. The patch
    // layout already interleaves both temporal frames, so nothing here has to
    // know whether they are two distinct frames or one still twice.
    //
    // ⚠️ `patches` ARRIVES in merged block order, because the reference's
    // `process_qwen2vl_images` permutes on the way out. So nothing here permutes
    // the tokens; what gets permuted is the POSITION table, which is interpolated
    // row-major and then reordered to match. Permuting the tokens as well applies
    // the merge twice and pairs every token with another patch's position, which
    // is a plausible-looking 0.75 relative error rather than a crash.
    const x = try a.alloc(f32, np * dim);
    try ops.matmul.matmul(io, gpa, x, patches, np, v.patch_w, v.patch_b);

    const pos_row = try a.alloc(f32, np * dim);
    interpolatePos(pos_row, v.pos_embed, cfg.pos_grid, dim, gh, gw);
    const pos = try a.alloc(f32, np * dim);
    mergeOrder(pos, pos_row, dim, gh, gw, cfg.merge);
    for (x, pos) |*d, p| d.* += p;

    // Rope tables over the two axes, `freqsPerAxis` frequencies each.
    const half = cfg.freqsPerAxis();
    const cos = try a.alloc(f32, cfg.pos_grid * half);
    const sin = try a.alloc(f32, cfg.pos_grid * half);
    {
        const max_hw = @max(gh, gw);
        std.debug.assert(max_hw <= cfg.pos_grid);
        for (0..cfg.pos_grid) |p| {
            for (0..half) |k| {
                const exp = @as(f64, @floatFromInt(2 * k)) / @as(f64, @floatFromInt(2 * half));
                const inv = std.math.pow(f64, cfg.rope_theta, -exp);
                const ang = @as(f64, @floatFromInt(p)) * inv;
                cos[p * half + k] = @floatCast(@cos(ang));
                sin[p * half + k] = @floatCast(@sin(ang));
            }
        }
    }
    const py = try a.alloc(u32, np);
    const px = try a.alloc(u32, np);
    patchCoords(py, px, gh, gw, cfg.merge);

    const norm = try a.alloc(f32, np * dim);
    const qkv = try a.alloc(f32, np * 3 * dim);
    const q = try a.alloc(f32, np * dim);
    const k = try a.alloc(f32, np * dim);
    const vv = try a.alloc(f32, np * dim);
    const att = try a.alloc(f32, np * dim);
    const ffn = try a.alloc(f32, np * cfg.ffn);
    const proj = try a.alloc(f32, np * dim);

    var ds_out = try gpa.alloc([]f32, cfg.n_deepstack);
    var n_ds: usize = 0;
    errdefer {
        for (ds_out[0..n_ds]) |d| gpa.free(d);
        gpa.free(ds_out);
    }

    for (v.blocks, 0..) |*b, bi| {
        // Attention half.
        for (0..np) |t| {
            ops.norm.layerNorm(norm[t * dim ..][0..dim], x[t * dim ..][0..dim], b.norm1_w, b.norm1_b, cfg.eps);
        }
        try ops.matmul.matmul(io, gpa, qkv, norm, np, b.qkv, b.qkv_b);
        // The projection emits `[t][3][heads][hd]`; attention wants three planes.
        for (0..np) |t| {
            const src = qkv[t * 3 * dim ..];
            @memcpy(q[t * dim ..][0..dim], src[0..dim]);
            @memcpy(k[t * dim ..][0..dim], src[dim..][0..dim]);
            @memcpy(vv[t * dim ..][0..dim], src[2 * dim ..][0..dim]);
        }
        vit35.applyVisionRope(q, np, cfg.n_heads, hd, py, px, cos, sin, half);
        vit35.applyVisionRope(k, np, cfg.n_heads, hd, py, px, cos, sin, half);
        try ops.attention.attention(io, gpa, att, q, k, vv, .{
            .seq_q = np,
            .seq_kv = np,
            .n_heads = cfg.n_heads,
            .n_kv_heads = cfg.n_heads,
            .head_dim = hd,
            // One image is one attention sequence; the reference's `cu_seqlens`
            // batches several images and never masks within one.
            .causal = false,
        });
        try ops.matmul.matmul(io, gpa, proj, att, np, b.proj, b.proj_b);
        for (x, proj) |*d, p| d.* += p;

        // MLP half. gelu-TANH here; the mergers use erf gelu.
        for (0..np) |t| {
            ops.norm.layerNorm(norm[t * dim ..][0..dim], x[t * dim ..][0..dim], b.norm2_w, b.norm2_b, cfg.eps);
        }
        try ops.matmul.matmul(io, gpa, ffn, norm, np, b.fc1, b.fc1_b);
        ops.act.geluTanh(ffn);
        try ops.matmul.matmul(io, gpa, proj, ffn, np, b.fc2, b.fc2_b);
        for (x, proj) |*d, p| d.* += p;

        // DeepStack: from this block's OUTPUT, not its input.
        for (v.cfg.deepstack_indexes[0..cfg.n_deepstack], 0..) |idx, di| {
            if (idx != bi) continue;
            ds_out[di] = try runMerger(io, gpa, a, &v.deepstack[di], cfg, x, np);
            n_ds += 1;
        }
    }

    const merged = try runMerger(io, gpa, a, &v.merger, cfg, x, np);
    return .{ .merged = merged, .deepstack = ds_out, .tokens = np / (cfg.merge * cfg.merge) };
}

/// A merger: LayerNorm (at one of two widths), 2x2 merge, `fc2(gelu(fc1(.)))`.
///
/// `post_merge_norm` is the whole difference between the two kinds, and it is
/// `norm(x.view(...))` against `norm(x).view(...)` in the reference. Both use ERF
/// gelu, unlike the blocks' tanh.
fn runMerger(
    io: std.Io,
    gpa: std.mem.Allocator,
    a: std.mem.Allocator,
    m: *const Merger,
    cfg: Config,
    x: []const f32,
    np: usize,
) ![]f32 {
    const dim = cfg.dim;
    const md = cfg.mergeDim();
    const tokens = np / (cfg.merge * cfg.merge);

    // The merge itself is free: tokens are already in block order, so four
    // consecutive rows ARE one merged row.
    const rows = try a.alloc(f32, tokens * md);
    if (m.post_merge_norm) {
        // Merge, then normalize the full merged width.
        @memcpy(rows, x[0 .. tokens * md]);
        for (0..tokens) |t| {
            ops.norm.layerNorm(rows[t * md ..][0..md], rows[t * md ..][0..md], m.norm_w, m.norm_b, cfg.eps);
        }
    } else {
        // Normalize each pre-merge token, then merge.
        for (0..np) |t| {
            ops.norm.layerNorm(rows[t * dim ..][0..dim], x[t * dim ..][0..dim], m.norm_w, m.norm_b, cfg.eps);
        }
    }

    const hidden = try a.alloc(f32, tokens * md);
    try ops.matmul.matmul(io, gpa, hidden, rows, tokens, m.fc1, m.fc1_b);
    ops.act.geluErf(hidden);
    const out = try gpa.alloc(f32, tokens * cfg.out_dim);
    errdefer gpa.free(out);
    try ops.matmul.matmul(io, gpa, out, hidden, tokens, m.fc2, m.fc2_b);
    return out;
}

// --- tests ----------------------------------------------------------------

const testing = std.testing;
const vit_fixture = @embedFile("assets/minimax_h3_vit.safetensors");

fn relL2(want: []const f32, got: []const f32) f64 {
    std.debug.assert(want.len == got.len);
    var l2_ref: f64 = 0;
    var l2_err: f64 = 0;
    for (want, got) |e, g| {
        l2_ref += @as(f64, e) * e;
        l2_err += @as(f64, e - g) * (e - g);
    }
    return if (l2_ref > 0) @sqrt(l2_err / l2_ref) else @sqrt(l2_err);
}

test "position interpolation is align_corners, not a resampling" {
    // The divergence from `vit35`: the reference samples the table at
    // `linspace(0, grid - 1, n)`, so the endpoints land EXACTLY on the first and
    // last row whatever `n` is. A ggml-style resampler with align_corners=false
    // offsets by half a cell and antialiases, which is a different table at every
    // grid but the native one.
    const grid = 4;
    const dim = 1;
    var pos: [grid * grid]f32 = undefined;
    for (&pos, 0..) |*p, i| p.* = @floatFromInt(i);

    // At the native grid it is the identity.
    var same: [grid * grid]f32 = undefined;
    interpolatePos(&same, &pos, grid, dim, grid, grid);
    for (same, pos) |g, w| try testing.expectApproxEqAbs(w, g, 1e-6);

    // Coarser: 2 samples of a 4-wide axis land on 0 and 3, NOT on 0.5 and 2.5.
    var two: [2 * 2]f32 = undefined;
    interpolatePos(&two, &pos, grid, dim, 2, 2);
    try testing.expectApproxEqAbs(@as(f32, 0), two[0], 1e-6); // (0,0)
    try testing.expectApproxEqAbs(@as(f32, 3), two[1], 1e-6); // (0,3)
    try testing.expectApproxEqAbs(@as(f32, 12), two[2], 1e-6); // (3,0)
    try testing.expectApproxEqAbs(@as(f32, 15), two[3], 1e-6); // (3,3)

    // A single sample sits at the ORIGIN, not the centre.
    var one: [1]f32 = undefined;
    interpolatePos(&one, &pos, grid, dim, 1, 1);
    try testing.expectApproxEqAbs(@as(f32, 0), one[0], 1e-6);

    // Finer than the table: 7 samples of a 4-wide axis step by 0.5, so the
    // interior ones are exact half-way blends.
    var fine: [7 * 1]f32 = undefined;
    interpolatePos(&fine, &pos, grid, dim, 1, 7);
    try testing.expectApproxEqAbs(@as(f32, 0.5), fine[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.0), fine[2], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 3.0), fine[6], 1e-6);
}

test "merged block order pairs each token with its own patch coordinates" {
    // Three stages have to agree on the token order -- the patch flattening, the
    // position interpolation and the rope -- and the order is
    // (block_row, block_col, intra_row, intra_col). Any disagreement pairs a token
    // with another patch's position, which is plausible and wrong.
    const gh = 4;
    const gw = 6;
    const merge = 2;
    var py: [gh * gw]u32 = undefined;
    var px: [gh * gw]u32 = undefined;
    patchCoords(&py, &px, gh, gw, merge);

    // First block covers rows 0-1, cols 0-1, in that nesting.
    try testing.expectEqualSlices(u32, &.{ 0, 0, 1, 1 }, py[0..4]);
    try testing.expectEqualSlices(u32, &.{ 0, 1, 0, 1 }, px[0..4]);
    // Second block steps the COLUMN, not the row.
    try testing.expectEqualSlices(u32, &.{ 0, 0, 1, 1 }, py[4..8]);
    try testing.expectEqualSlices(u32, &.{ 2, 3, 2, 3 }, px[4..8]);
    // Row-major would have given (0,0),(0,1),(0,2),(0,3) here instead.
    try testing.expect(px[2] != 2);

    // `mergeOrder` moves the same permutation, so a row-major table indexed by
    // (py, px) reproduces it.
    var src: [gh * gw]f32 = undefined;
    for (&src, 0..) |*s, i| s.* = @floatFromInt(i);
    var dst: [gh * gw]f32 = undefined;
    mergeOrder(&dst, &src, 1, gh, gw, merge);
    for (0..gh * gw) |t| {
        try testing.expectEqual(@as(f32, @floatFromInt(py[t] * gw + px[t])), dst[t]);
    }
}

const mrope_fixture = @embedFile("assets/minimax_h3_mrope.safetensors");

test "multimodal rope positions and frequencies match the reference" {
    // Both halves of mrope, from tools/gen_minimax_h3_mrope.py: the position-id
    // construction (`qwen2vl_mrope_position_ids`) and the INTERLEAVED frequency
    // assignment (`precompute_freqs_cis(interleaved_mrope=True)`).
    //
    // The two-image case is the one that matters: a block consumes
    // `max(gh, gw) / 2` timeline positions but occupies `size` tokens, so the
    // timeline drifts behind the token index and every position after the first
    // block moves. One image alone cannot see that.
    const gpa = testing.allocator;

    var st = try tp_core.safetensors.SafeTensors.initFromSlice(gpa, mrope_fixture);
    defer st.deinit();
    const store: WeightStore = .{ .safetensors = &st };

    const head_dim: usize = 128;
    const dims: [3]usize = .{ 24, 20, 20 };
    const theta: f64 = 5000000.0;

    const cases = [_]struct {
        name: []const u8,
        seq: usize,
        images: []const ImageSpan,
    }{
        .{ .name = "one_image", .seq = 40, .images = &.{
            .{ .index = 6, .size = 15, .grid_h = 6, .grid_w = 10 },
        } },
        .{ .name = "two_images", .seq = 60, .images = &.{
            .{ .index = 4, .size = 6, .grid_h = 4, .grid_w = 6 },
            .{ .index = 30, .size = 8, .grid_h = 8, .grid_w = 4 },
        } },
        .{ .name = "text_only", .seq = 24, .images = &.{} },
    };

    for (cases) |c| {
        var buf: [48]u8 = undefined;
        const want_pos = try (store.get(try std.fmt.bufPrint(&buf, "pos.{s}", .{c.name})) orelse
            return error.MissingTensor).toF32Alloc(gpa);
        defer gpa.free(want_pos);

        const pos = try gpa.alloc(f32, 3 * c.seq);
        defer gpa.free(pos);
        try mropePositions(pos, c.seq, c.images);

        for (0..3) |ax| {
            for (0..c.seq) |t| {
                errdefer std.debug.print("{s} axis {d} token {d}: got {d} want {d}\n", .{
                    c.name, ax, t, pos[ax * c.seq + t], want_pos[ax * c.seq + t],
                });
                try testing.expectEqual(want_pos[ax * c.seq + t], pos[ax * c.seq + t]);
            }
        }

        // ...and the frequency table those positions produce.
        var freqs = try ops.rope.mropeInterleavedFreqs(gpa, pos, c.seq, head_dim, dims, theta);
        defer freqs.deinit(gpa);
        const want_cos = try (store.get(try std.fmt.bufPrint(&buf, "cos.{s}", .{c.name})) orelse
            return error.MissingTensor).toF32Alloc(gpa);
        defer gpa.free(want_cos);
        const want_sin = try (store.get(try std.fmt.bufPrint(&buf, "sin.{s}", .{c.name})) orelse
            return error.MissingTensor).toF32Alloc(gpa);
        defer gpa.free(want_sin);
        try testing.expectEqual(want_cos.len, freqs.cos.len);
        const rel_c = relL2(want_cos, freqs.cos);
        const rel_s = relL2(want_sin, freqs.sin);
        errdefer std.debug.print("{s}: cos rel {e} sin rel {e}\n", .{ c.name, rel_c, rel_s });
        try testing.expect(rel_c < 1e-6 and rel_s < 1e-6);
    }
}

test "the timeline drifts behind the token index across several images" {
    // The `offset` accumulation, stated directly. A block occupies `size` tokens
    // and consumes `max(gh, gw) / 2` timeline positions; when those differ, every
    // position after it shifts. Dropping the offset leaves a plausible-looking
    // rope that mis-places the second reference and everything after it.
    const gpa = testing.allocator;
    const seq: usize = 60;
    const images = [_]ImageSpan{
        .{ .index = 4, .size = 6, .grid_h = 4, .grid_w = 6 },
        .{ .index = 30, .size = 8, .grid_h = 8, .grid_w = 4 },
    };
    // First block: 6 tokens, timeline span max(4,6)/2 = 3, so it gives back 3.
    try testing.expectEqual(@as(usize, 3), images[0].timelineSpan());
    try testing.expectEqual(@as(usize, 4), images[1].timelineSpan());

    const pos = try gpa.alloc(f32, 3 * seq);
    defer gpa.free(pos);
    try mropePositions(pos, seq, &images);

    // Text before the first block is the identity.
    for (0..4) |t| try testing.expectEqual(@as(f32, @floatFromInt(t)), pos[t]);
    // After the LAST block the timeline is behind the token index by the total
    // drift: (6 - 3) + (8 - 4) = 7.
    const after = images[1].index + images[1].size;
    try testing.expectEqual(@as(f32, @floatFromInt(after - 7)), pos[after]);
    // ...and it keeps stepping by one from there.
    try testing.expectEqual(pos[after] + 1, pos[after + 1]);

    // Inside a block: axis 0 constant, axis 1 the merged row, axis 2 the column.
    const b = images[1];
    const base = pos[b.index];
    for (0..b.size) |k| {
        try testing.expectEqual(base, pos[b.index + k]);
        try testing.expectEqual(base + @as(f32, @floatFromInt(k / b.mergedW())), pos[seq + b.index + k]);
        try testing.expectEqual(base + @as(f32, @floatFromInt(k % b.mergedW())), pos[2 * seq + b.index + k]);
    }

    // Malformed spans are refused rather than writing a plausible wrong table.
    for ([_][]const ImageSpan{
        &.{.{ .index = 4, .size = 5, .grid_h = 4, .grid_w = 6 }}, // size != mh*mw
        &.{.{ .index = 4, .size = 6, .grid_h = 5, .grid_w = 6 }}, // odd grid
        &.{ .{ .index = 10, .size = 6, .grid_h = 4, .grid_w = 6 }, .{ .index = 12, .size = 6, .grid_h = 4, .grid_w = 6 } }, // overlap
        &.{.{ .index = 58, .size = 6, .grid_h = 4, .grid_w = 6 }}, // past the end
    }) |bad| {
        try testing.expectError(error.BadImageSpan, mropePositions(pos, seq, bad));
    }
}

test "image preprocessing reproduces the reference's patch matrix" {
    // The fixture stores the RAW image beside the patches the reference derived
    // from it, so this closes the loop from pixels to tower input. The image is
    // 100x150, which resizes to 96x160 -- DOWN in one axis and UP in the other,
    // because `round(dim / 32) * 32` is nearest rather than floor. An image
    // already on the 32 px grid would leave the resize an identity and pin
    // nothing, which is what the first version of this fixture did.
    const gpa = testing.allocator;

    var st = try tp_core.safetensors.SafeTensors.initFromSlice(gpa, vit_fixture);
    defer st.deinit();
    const store: WeightStore = .{ .safetensors = &st };

    const cfg: Config = .{
        .dim = 64,
        .n_heads = 4,
        .ffn = 128,
        .n_blocks = 6,
        .patch = 16,
        .temporal_patch = 2,
        .merge = 2,
        .in_channels = 3,
        .pos_grid = 48,
        .out_dim = 128,
        .n_deepstack = 3,
        .deepstack_indexes = .{ 1, 3, 5, 0 },
    };
    const img_h: usize = 100;
    const img_w: usize = 150;

    // The resize really moves both axes, in opposite directions.
    const t = resizeTarget(cfg, img_h, img_w, 3136, 12845056);
    try testing.expectEqual(@as(usize, 96), t.h);
    try testing.expectEqual(@as(usize, 160), t.w);
    try testing.expect(t.h < img_h and t.w > img_w);

    // The fixture's image is [1][h][w][3] interleaved; the engine works planar.
    const hwc = try (store.get("in.image") orelse return error.MissingTensor).toF32Alloc(gpa);
    defer gpa.free(hwc);
    try testing.expectEqual(img_h * img_w * 3, hwc.len);
    const planar = try gpa.alloc(f32, 3 * img_h * img_w);
    defer gpa.free(planar);
    for (0..img_h) |y| {
        for (0..img_w) |x| {
            for (0..3) |c| planar[c * img_h * img_w + y * img_w + x] = hwc[(y * img_w + x) * 3 + c];
        }
    }

    var prep = try preprocessStill(gpa, cfg, planar, img_h, img_w, 3136, 12845056);
    defer prep.deinit(gpa);
    try testing.expectEqual(@as(usize, 6), prep.grid_h);
    try testing.expectEqual(@as(usize, 10), prep.grid_w);

    const want = try (store.get("in.patches") orelse return error.MissingTensor).toF32Alloc(gpa);
    defer gpa.free(want);
    try testing.expectEqual(want.len, prep.patches.len);
    const rel = relL2(want, prep.patches);
    errdefer std.debug.print("patches rel L2 {e}\n", .{rel});
    // The bilinear resample is the only lossy step and it is f32 both sides.
    try testing.expect(rel < 1e-5);

    // ...and running the tower on OUR patches reproduces the tower output the
    // fixture stored, so the whole pixels-to-features path is closed.
    var pfx = try weights_mod.Prefixed.init(gpa, store, "visual.");
    defer pfx.deinit(gpa);
    var v = try Vit.load(gpa, pfx.store(), cfg);
    defer v.deinit();
    var got = try encode(&v, testing.io, gpa, prep.patches, prep.grid_h, prep.grid_w);
    defer got.deinit(gpa);
    const merged = try (store.get("out.merged") orelse return error.MissingTensor).toF32Alloc(gpa);
    defer gpa.free(merged);
    const rel_m = relL2(merged, got.merged);
    errdefer std.debug.print("merged-from-our-pixels rel L2 {e}\n", .{rel_m});
    try testing.expect(rel_m < 1e-4);
}

test "the resize is nearest-to-grid with banker's rounding, not floor" {
    // `round(dim / 32) * 32` in Python is banker's rounding, so an axis exactly
    // half a factor over the grid goes to EVEN rather than up. 48 is 1.5 factors
    // and rounds to 2 (64, even); 112 is 3.5 and rounds to 4 (128, even) -- but
    // 80 is 2.5 and rounds DOWN to 2 (64), which half-up would send to 96.
    const cfg: Config = .{
        .dim = 64, .n_heads = 4, .ffn = 128, .n_blocks = 1, .patch = 16,
        .temporal_patch = 2, .merge = 2, .in_channels = 3, .pos_grid = 48,
        .out_dim = 128, .n_deepstack = 0, .deepstack_indexes = .{ 0, 0, 0, 0 },
    };
    const big: usize = 1 << 28; // no clamping
    try testing.expectEqual(@as(usize, 96), resizeTarget(cfg, 100, 100, 0, big).h);
    try testing.expectEqual(@as(usize, 160), resizeTarget(cfg, 150, 150, 0, big).w);
    // The ties: 80 / 32 == 2.5 -> 2 (even), 112 / 32 == 3.5 -> 4 (even).
    try testing.expectEqual(@as(usize, 64), resizeTarget(cfg, 80, 80, 0, big).h);
    try testing.expectEqual(@as(usize, 128), resizeTarget(cfg, 112, 112, 0, big).h);
    // Never below one factor, however small the input.
    try testing.expectEqual(@as(usize, 32), resizeTarget(cfg, 3, 3, 0, big).h);
    // A tiny image is scaled UP to clear min_pixels rather than left as one patch.
    const up = resizeTarget(cfg, 8, 8, 3136, big);
    try testing.expect(up.h * up.w >= 3136);
}

test "a video block's two temporal slots hold DISTINCT frames" {
    // The case `vit35` cannot express: it SUMS the patch embedding's two temporal
    // convolutions, which is right only when both slots hold the same frame. Here
    // they hold consecutive frames of a reference video, and the flattening has to
    // keep them apart -- interleaved as (channel, temporal, ky, kx) so the Conv3d
    // kernel's own flattening lines up.
    const gpa = testing.allocator;
    const cfg: Config = .{
        .dim = 64, .n_heads = 4, .ffn = 128, .n_blocks = 1, .patch = 16,
        .temporal_patch = 2, .merge = 2, .in_channels = 3, .pos_grid = 48,
        .out_dim = 128, .n_deepstack = 0, .deepstack_indexes = .{ 0, 0, 0, 0 },
    };
    const h: usize = 32;
    const w: usize = 32;
    // Two constant frames at different levels, so a swap or a sum is visible.
    const f0 = try gpa.alloc(f32, 3 * h * w);
    defer gpa.free(f0);
    const f1 = try gpa.alloc(f32, 3 * h * w);
    defer gpa.free(f1);
    @memset(f0, 0.25);
    @memset(f1, 0.75);

    var pair = try preprocessPair(gpa, cfg, f0, f1, h, w, 0, 1 << 28);
    defer pair.deinit(gpa);

    // Normalization maps 0.25 -> -0.5 and 0.75 -> +0.5.
    const p = cfg.patch;
    const row = pair.patches[0..cfg.patchDim()];
    // Within a row the layout is (channel, temporal, ky, kx), so the first
    // `p * p` values are channel 0 / temporal 0 and the next `p * p` are
    // channel 0 / temporal 1.
    for (row[0 .. p * p]) |v| try testing.expectApproxEqAbs(@as(f32, -0.5), v, 1e-5);
    for (row[p * p .. 2 * p * p]) |v| try testing.expectApproxEqAbs(@as(f32, 0.5), v, 1e-5);

    // A STILL puts the same value in both slots, which is what makes the summed
    // embedding correct there and only there.
    var still = try preprocessStill(gpa, cfg, f0, h, w, 0, 1 << 28);
    defer still.deinit(gpa);
    const srow = still.patches[0..cfg.patchDim()];
    for (srow[0 .. 2 * p * p]) |v| try testing.expectApproxEqAbs(@as(f32, -0.5), v, 1e-5);

    // Swapping the pair really changes the patches, so the ordering is pinned.
    var swapped = try preprocessPair(gpa, cfg, f1, f0, h, w, 0, 1 << 28);
    defer swapped.deinit(gpa);
    try testing.expect(relL2(pair.patches, swapped.patches) > 0.5);
}

test "the resize samples pixel centres, not corners" {
    // align_corners=FALSE puts the sample at `(i + 0.5) * scale - 0.5`. The
    // align_corners=TRUE form `i * (n-1)/(m-1)` is `vit35`'s and differs by half a
    // pixel at every scale but 1, which is a whole-image shift rather than noise.
    //
    // A 2 -> 4 upscale of [0, 1] makes the difference visible: centres give
    // 0, 0.25, 0.75, 1 (the ends CLAMP), corners give 0, 1/3, 2/3, 1.
    var src = [_]f32{ 0, 1 } ** 3; // three channels of [0, 1]
    var dst: [3 * 1 * 4]f32 = undefined;
    resizeBilinear(&dst, &src, 1, 2, 1, 4);
    try testing.expectApproxEqAbs(@as(f32, 0.0), dst[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.25), dst[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.75), dst[2], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.0), dst[3], 1e-6);
    // Not the align_corners=true answer.
    try testing.expect(@abs(dst[1] - 1.0 / 3.0) > 0.05);
    // Identity at the same size.
    var same: [3 * 1 * 2]f32 = undefined;
    resizeBilinear(&same, &src, 1, 2, 1, 2);
    try testing.expectApproxEqAbs(@as(f32, 0.0), same[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.0), same[1], 1e-6);
}

test "the vision tower matches the reference, deepstack included" {
    // ComfyUI's own `Qwen3VLVisionModel` at a toy width, from
    // tools/gen_minimax_h3_vit.py. The image is non-square (96x160 -> a 6x10 patch
    // grid) so an h/w swap in the patch grid, the position interpolation or the
    // rope cannot pass.
    const gpa = testing.allocator;
    const io = testing.io;

    var st = try tp_core.safetensors.SafeTensors.initFromSlice(gpa, vit_fixture);
    defer st.deinit();
    const store: WeightStore = .{ .safetensors = &st };
    var pfx = try weights_mod.Prefixed.init(gpa, store, "visual.");
    defer pfx.deinit(gpa);

    const cfg: Config = .{
        .dim = 64,
        .n_heads = 4,
        .ffn = 128,
        .n_blocks = 6,
        .patch = 16,
        .temporal_patch = 2,
        .merge = 2,
        .in_channels = 3,
        .pos_grid = 48,
        .out_dim = 128,
        .n_deepstack = 3,
        .deepstack_indexes = .{ 1, 3, 5, 0 },
    };
    var v = try Vit.load(gpa, pfx.store(), cfg);
    defer v.deinit();
    // The two merger kinds normalize at different widths, and the loader reads
    // each from the checkpoint. A swap would have failed above.
    try testing.expectEqual(cfg.dim, v.merger.norm_w.len);
    try testing.expectEqual(cfg.mergeDim(), v.deepstack[0].norm_w.len);
    try testing.expect(!v.merger.post_merge_norm and v.deepstack[0].post_merge_norm);

    const patches = try (store.get("in.patches") orelse return error.MissingTensor).toF32Alloc(gpa);
    defer gpa.free(patches);
    const gh: usize = 6;
    const gw: usize = 10;
    try testing.expectEqual(gh * gw * cfg.patchDim(), patches.len);

    var got = try encode(&v, io, gpa, patches, gh, gw);
    defer got.deinit(gpa);
    try testing.expectEqual(@as(usize, gh * gw / 4), got.tokens);

    const want = try (store.get("out.merged") orelse return error.MissingTensor).toF32Alloc(gpa);
    defer gpa.free(want);
    try testing.expectEqual(want.len, got.merged.len);
    {
        const rel = relL2(want, got.merged);
        errdefer std.debug.print("merged rel L2 {e}\n", .{rel});
        try testing.expect(rel < 1e-5);
    }

    for (0..cfg.n_deepstack) |i| {
        var buf: [32]u8 = undefined;
        const nm = try std.fmt.bufPrint(&buf, "out.deepstack.{d}", .{i});
        const dw = try (store.get(nm) orelse return error.MissingTensor).toF32Alloc(gpa);
        defer gpa.free(dw);
        const rel = relL2(dw, got.deepstack[i]);
        errdefer std.debug.print("deepstack[{d}] rel L2 {e}\n", .{ i, rel });
        try testing.expect(rel < 1e-5);
        // ...and it is genuinely a different feature from the tower output, so a
        // port that ran the main merger three times would fail here.
        try testing.expect(relL2(want, got.deepstack[i]) > 0.1);
    }
}
