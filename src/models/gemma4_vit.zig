//! Gemma 4 "unified" vision embedder (mmproj GGUF, arch "clip", projector
//! "gemma4uv"). Unlike Gemma 3's SigLIP tower there is NO transformer — the
//! "token merging" is baked into a large conv patch, so this is a shallow
//! embedder. CPU forward, f32 compute. Ported from llama.cpp
//! tools/mtmd/models/gemma4uv.cpp + clip.cpp/mtmd-image.cpp preprocessing.
//!
//! Pipeline (per llama.cpp):
//!   - Smart-resize the image so its area lands in [40, 280] patch tokens,
//!     each dim snapped to a 48-px multiple (aspect ~preserved), bilinear +
//!     PAD_CEIL onto a black canvas; normalize to [0,1] (mean 0, std 1).
//!   - im2col 48x48 stride-48 patches (channel-planar) -> [n_patches, 6912].
//!   - LayerNorm (patch_norm_1, over 6912, eps 1e-5).
//!   - patch-embed matmul (6912 -> 3840) + bias.
//!   - LayerNorm (patch_norm_2, over 3840).
//!   - add learned positional embeddings: two lookup tables (x by column, y by
//!     row) indexed by the patch's grid position.
//!   - LayerNorm (patch_norm_3 / pos_norm).
//!   - weightless RMSNorm (eps 1e-6).
//!   - mm.input_projection (3840 -> 3840) -> [n_patches, 3840] LLM embeddings.
//!
//! One image is `n_patches = (W/48)*(H/48)` tokens (variable, NO pooling); the
//! LLM injects them UNSCALED between the `<|image>` / `<image|>` markers.
//!
//! Simplification vs llama.cpp: image tokens use the LLM's causal attention
//! here (llama.cpp marks them non-causal / bidirectional). Same approximation
//! Gemma 3's CPU path makes; revisit if caption quality needs it.

const std = @import("std");
const gguf_mod = @import("../gguf.zig");
const weights_mod = @import("../weights.zig");
const ops = @import("../ops.zig");
const loader = @import("loader.zig");

const Gguf = gguf_mod.Gguf;
const WeightStore = weights_mod.WeightStore;
const Weight = ops.matmul.Weight;

/// Token-merge factor folded into the conv patch (llama.cpp gemma4uv sets
/// patch_size *= n_merge, then n_merge = 1).
const n_merge = 3;
/// Per-image token bounds (llama.cpp set_limit_image_tokens(40, 280)).
const min_tokens = 40;
const max_tokens = 280;

pub const Config = struct {
    dim: usize,
    proj_dim: usize,
    /// Effective conv patch (raw patch_size * n_merge).
    patch: usize,
    /// Positional-table length per axis (v.position_embd ne1).
    pos_size: usize,
    /// LayerNorm eps (llama.cpp hardcodes 1e-5 for the 3 vision LayerNorms).
    eps_ln: f32,
    /// Pre-projection RMSNorm eps (hparams.eps, default 1e-6).
    eps_rms: f32,

    pub fn kdim(self: Config) usize {
        return 3 * self.patch * self.patch;
    }
    pub fn minPixels(self: Config) usize {
        return min_tokens * self.patch * self.patch;
    }
    pub fn maxPixels(self: Config) usize {
        return max_tokens * self.patch * self.patch;
    }

    pub fn detect(g: *const Gguf) !Config {
        const arch = g.getStr("general.architecture") orelse return error.UnknownModelConfig;
        if (!std.mem.eql(u8, arch, "clip")) return error.UnknownModelConfig;
        const proj = g.getStr("clip.vision.projector_type") orelse return error.UnknownModelConfig;
        if (!std.mem.eql(u8, proj, "gemma4uv")) return error.UnknownModelConfig;

        const raw_patch: usize = @intCast(g.getUint("clip.vision.patch_size") orelse return error.UnknownModelConfig);
        const dim: usize = @intCast(g.getUint("clip.vision.embedding_length") orelse return error.UnknownModelConfig);
        const proj_dim: usize = @intCast(g.getUint("clip.vision.projection_dim") orelse dim);

        // Positional table: v.position_embd.weight is [dim, pos_size, 2].
        const pos = g.get("v.position_embd.weight") orelse return error.UnknownModelConfig;
        if (pos.info.elemCount() % (dim * 2) != 0) return error.UnknownModelConfig;
        const pos_size = pos.info.elemCount() / (dim * 2);

        return .{
            .dim = dim,
            .proj_dim = proj_dim,
            .patch = raw_patch * n_merge,
            .pos_size = pos_size,
            .eps_ln = 1e-5,
            .eps_rms = @floatCast(g.getFloat("clip.vision.attention.layer_norm_epsilon") orelse 1e-6),
        };
    }
};

pub const Vit = struct {
    arena: std.heap.ArenaAllocator,
    cfg: Config,
    /// Patch-embed conv as [dim][6912] im2col GEMM weight, + bias.
    patch_w: Weight,
    patch_b: []const f32,
    patch_norm_1_w: []const f32, // over kdim (6912)
    patch_norm_1_b: []const f32,
    patch_norm_2_w: []const f32, // over dim (3840)
    patch_norm_2_b: []const f32,
    patch_norm_3_w: []const f32,
    patch_norm_3_b: []const f32,
    /// Positional tables, flat [2][pos_size][dim] (x = table 0, y = table 1).
    pos_embd: []const f32,
    /// mm.input_projection [proj_dim][dim] (ggml_mul_mat orientation: no transpose).
    mm_proj: Weight,

    pub fn load(gpa: std.mem.Allocator, g: *const Gguf) !Vit {
        const cfg = try Config.detect(g);
        const store: WeightStore = .{ .gguf = g };
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();

        // Compute all loads into locals BEFORE building the struct: the arena
        // must be captured into the returned Vit only after every allocation,
        // or `.arena` snapshots a stale buffer list and the later allocs leak.
        const patch_w = try loader.matrix(store, "v.patch_embd.weight", cfg.dim, cfg.kdim());
        const patch_b = try loader.vector(alloc, store, "v.patch_embd.bias", cfg.dim);
        const pn1_w = try loader.vector(alloc, store, "v.patch_norm.1.weight", cfg.kdim());
        const pn1_b = try loader.vector(alloc, store, "v.patch_norm.1.bias", cfg.kdim());
        const pn2_w = try loader.vector(alloc, store, "v.patch_norm.2.weight", cfg.dim);
        const pn2_b = try loader.vector(alloc, store, "v.patch_norm.2.bias", cfg.dim);
        const pn3_w = try loader.vector(alloc, store, "v.patch_norm.3.weight", cfg.dim);
        const pn3_b = try loader.vector(alloc, store, "v.patch_norm.3.bias", cfg.dim);
        const pos_embd = try loader.vector(alloc, store, "v.position_embd.weight", 2 * cfg.pos_size * cfg.dim);
        const mm_proj = try loader.matrix(store, "mm.input_projection.weight", cfg.proj_dim, cfg.dim);
        return .{
            .arena = arena,
            .cfg = cfg,
            .patch_w = patch_w,
            .patch_b = patch_b,
            .patch_norm_1_w = pn1_w,
            .patch_norm_1_b = pn1_b,
            .patch_norm_2_w = pn2_w,
            .patch_norm_2_b = pn2_b,
            .patch_norm_3_w = pn3_w,
            .patch_norm_3_b = pn3_b,
            .pos_embd = pos_embd,
            .mm_proj = mm_proj,
        };
    }

    pub fn deinit(self: *Vit) void {
        self.arena.deinit();
        self.* = undefined;
    }

    pub const Encoded = struct {
        /// [n_patches][proj_dim] projected image-token embeddings.
        embeds: []f32,
        grid_w: usize, // patch columns (W/patch)
        grid_h: usize, // patch rows (H/patch)

        pub fn deinit(self: *Encoded, gpa: std.mem.Allocator) void {
            gpa.free(self.embeds);
            self.* = undefined;
        }
    };

    /// Preprocessed patches: the im2col matrix [np][kdim] (row-major patch
    /// order) plus the patch-grid dims. Shared by the CPU and CUDA encoders.
    pub const Patches = struct {
        data: []f32, // gpa-owned
        n_cols: usize,
        n_rows: usize,
        pub fn np(self: Patches) usize {
            return self.n_cols * self.n_rows;
        }
        pub fn deinit(self: *Patches, gpa: std.mem.Allocator) void {
            gpa.free(self.data);
            self.* = undefined;
        }
    };

    /// Smart-resize + normalize + im2col: RGB pixels -> [np][kdim] patch matrix
    /// (channel-planar 48x48 patches, row-major patch order).
    pub fn patchMatrix(self: *const Vit, gpa: std.mem.Allocator, rgb: []const u8, width: usize, height: usize) !Patches {
        const cfg = self.cfg;
        const patch = cfg.patch;
        const kdim = cfg.kdim();
        const tgt = smartResize(width, height, patch, cfg.minPixels(), cfg.maxPixels());
        const n_cols = tgt.w / patch;
        const n_rows = tgt.h / patch;
        const np = n_cols * n_rows;

        const chw = try preprocess(gpa, rgb, width, height, tgt.w, tgt.h);
        defer gpa.free(chw);
        const patches = try gpa.alloc(f32, np * kdim);
        errdefer gpa.free(patches);
        for (0..n_rows) |gy| {
            for (0..n_cols) |gx| {
                const row = patches[(gy * n_cols + gx) * kdim ..][0..kdim];
                for (0..3) |c| {
                    for (0..patch) |ky| {
                        const src = chw[c * tgt.h * tgt.w + (gy * patch + ky) * tgt.w + gx * patch ..][0..patch];
                        @memcpy(row[(c * patch + ky) * patch ..][0..patch], src);
                    }
                }
            }
        }
        return .{ .data = patches, .n_cols = n_cols, .n_rows = n_rows };
    }

    /// The per-patch learned positional embedding as a dense [np][dim] buffer
    /// (row p = table_x[col] + table_y[row]), for a single opAdd on device or
    /// host. gpa-owned.
    pub fn posEmbedRows(self: *const Vit, gpa: std.mem.Allocator, n_cols: usize, n_rows: usize) ![]f32 {
        const cfg = self.cfg;
        const dim = cfg.dim;
        const np = n_cols * n_rows;
        const tbl_x = self.pos_embd[0 .. cfg.pos_size * dim];
        const tbl_y = self.pos_embd[cfg.pos_size * dim .. 2 * cfg.pos_size * dim];
        const out = try gpa.alloc(f32, np * dim);
        errdefer gpa.free(out);
        for (0..np) |p| {
            const ex = tbl_x[(p % n_cols) * dim ..][0..dim];
            const ey = tbl_y[(p / n_cols) * dim ..][0..dim];
            const dst = out[p * dim ..][0..dim];
            for (dst, ex, ey) |*d, vx, vy| d.* = vx + vy;
        }
        return out;
    }

    /// Encode interleaved RGB pixels to LLM image-token embeddings (CPU).
    pub fn encode(self: *const Vit, io: std.Io, gpa: std.mem.Allocator, rgb: []const u8, width: usize, height: usize) !Encoded {
        const cfg = self.cfg;
        const dim = cfg.dim;

        var pm = try self.patchMatrix(gpa, rgb, width, height);
        defer pm.deinit(gpa);
        const np = pm.np();
        const patches = pm.data;

        // patch_norm_1 (over kdim) -> patch-embed matmul -> patch_norm_2.
        ops.norm.layerNorm(patches, patches, self.patch_norm_1_w, self.patch_norm_1_b, cfg.eps_ln);
        const x = try gpa.alloc(f32, np * dim);
        defer gpa.free(x);
        try ops.matmul.matmul(io, gpa, x, patches, np, self.patch_w, self.patch_b);
        ops.norm.layerNorm(x, x, self.patch_norm_2_w, self.patch_norm_2_b, cfg.eps_ln);

        // Add learned positional embeddings, then patch_norm_3.
        const pos = try self.posEmbedRows(gpa, pm.n_cols, pm.n_rows);
        defer gpa.free(pos);
        for (x, pos) |*d, p| d.* += p;
        ops.norm.layerNorm(x, x, self.patch_norm_3_w, self.patch_norm_3_b, cfg.eps_ln);

        // weightless RMSNorm -> projection.
        ops.norm.rmsNormUnit(x, x, dim, cfg.eps_rms);
        const embeds = try gpa.alloc(f32, np * cfg.proj_dim);
        errdefer gpa.free(embeds);
        try ops.matmul.matmul(io, gpa, embeds, x, np, self.mm_proj, null);
        return .{ .embeds = embeds, .grid_w = pm.n_cols, .grid_h = pm.n_rows };
    }
};

const Size = struct { w: usize, h: usize };

/// llama.cpp img_tool::calc_size_preserved_ratio "smart resize": snap each dim
/// to a `patch` multiple, then clamp total area into [min_px, max_px] while
/// keeping ~aspect ratio (mtmd-image.cpp smart_resize).
fn smartResize(sw: usize, sh: usize, patch: usize, min_px: usize, max_px: usize) Size {
    const m: f64 = @floatFromInt(patch);
    const W: f64 = @floatFromInt(sw);
    const H: f64 = @floatFromInt(sh);
    const roundBy = struct {
        fn f(x: f64, mm: f64) f64 {
            return @round(x / mm) * mm;
        }
    }.f;
    const floorBy = struct {
        fn f(x: f64, mm: f64) f64 {
            return @floor(x / mm) * mm;
        }
    }.f;
    const ceilBy = struct {
        fn f(x: f64, mm: f64) f64 {
            return @ceil(x / mm) * mm;
        }
    }.f;

    var h_bar = @max(m, roundBy(H, m));
    var w_bar = @max(m, roundBy(W, m));
    const max_f: f64 = @floatFromInt(max_px);
    const min_f: f64 = @floatFromInt(min_px);
    if (h_bar * w_bar > max_f) {
        const beta = @sqrt(H * W / max_f);
        h_bar = @max(m, floorBy(H / beta, m));
        w_bar = @max(m, floorBy(W / beta, m));
    } else if (h_bar * w_bar < min_f) {
        const beta = @sqrt(min_f / (H * W));
        h_bar = ceilBy(H * beta, m);
        w_bar = ceilBy(W * beta, m);
    }
    return .{ .w = @intFromFloat(w_bar), .h = @intFromFloat(h_bar) };
}

/// Resize `rgb` (interleaved, sw x sh) to `tw x th` and normalize to planar
/// CHW f32 in [0,1] (mean 0 / std 1). Aspect-preserving align-corners bilinear
/// (truncating u8, matching llama.cpp), PAD_CEIL onto a black canvas.
fn preprocess(gpa: std.mem.Allocator, rgb: []const u8, sw: usize, sh: usize, tw: usize, th: usize) ![]f32 {
    const scale = @min(
        @as(f64, @floatFromInt(tw)) / @as(f64, @floatFromInt(sw)),
        @as(f64, @floatFromInt(th)) / @as(f64, @floatFromInt(sh)),
    );
    const nw = @min(@as(usize, @intFromFloat(@ceil(@as(f64, @floatFromInt(sw)) * scale))), tw);
    const nh = @min(@as(usize, @intFromFloat(@ceil(@as(f64, @floatFromInt(sh)) * scale))), th);

    const resized = try gpa.alloc(u8, nw * nh * 3);
    defer gpa.free(resized);
    const x_ratio: f64 = if (nw > 1) @as(f64, @floatFromInt(sw - 1)) / @as(f64, @floatFromInt(nw - 1)) else 0;
    const y_ratio: f64 = if (nh > 1) @as(f64, @floatFromInt(sh - 1)) / @as(f64, @floatFromInt(nh - 1)) else 0;
    for (0..nh) |y| {
        const pyf = @as(f64, @floatFromInt(y)) * y_ratio;
        const y0 = @min(@as(usize, @intFromFloat(pyf)), sh - 1);
        const y1 = @min(y0 + 1, sh - 1);
        const yf = pyf - @as(f64, @floatFromInt(y0));
        for (0..nw) |xx| {
            const pxf = @as(f64, @floatFromInt(xx)) * x_ratio;
            const x0 = @min(@as(usize, @intFromFloat(pxf)), sw - 1);
            const x1 = @min(x0 + 1, sw - 1);
            const xf = pxf - @as(f64, @floatFromInt(x0));
            for (0..3) |c| {
                const p00: f64 = @floatFromInt(rgb[(y0 * sw + x0) * 3 + c]);
                const p01: f64 = @floatFromInt(rgb[(y0 * sw + x1) * 3 + c]);
                const p10: f64 = @floatFromInt(rgb[(y1 * sw + x0) * 3 + c]);
                const p11: f64 = @floatFromInt(rgb[(y1 * sw + x1) * 3 + c]);
                const top = p00 + (p01 - p00) * xf;
                const bottom = p10 + (p11 - p10) * xf;
                resized[(y * nw + xx) * 3 + c] = @intFromFloat(top + (bottom - top) * yf);
            }
        }
    }

    const off_x = (tw - nw) / 2;
    const off_y = (th - nh) / 2;
    const out = try gpa.alloc(f32, 3 * tw * th);
    errdefer gpa.free(out);
    for (0..3) |c| {
        const plane = out[c * tw * th ..][0 .. tw * th];
        @memset(plane, 0.0); // black pad -> 0/255 = 0
        for (0..nh) |y| {
            for (0..nw) |xx| {
                const p: f32 = @floatFromInt(resized[(y * nw + xx) * 3 + c]);
                plane[(y + off_y) * tw + (xx + off_x)] = p / 255.0;
            }
        }
    }
    return out;
}

// --- tests -----------------------------------------------------------------

test "gemma4 vit loads from real mmproj" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const path = "/home/qt/genai/lmstudio/models/mmproj-gemma-4-12b-it-qat-q4_0.gguf";
    std.Io.Dir.cwd().access(io, path, .{}) catch return error.SkipZigTest;

    var g = try Gguf.open(gpa, io, path);
    defer g.deinit();
    var vit = try Vit.load(gpa, &g);
    defer vit.deinit();

    const cfg = vit.cfg;
    try std.testing.expectEqual(@as(usize, 3840), cfg.dim);
    try std.testing.expectEqual(@as(usize, 3840), cfg.proj_dim);
    try std.testing.expectEqual(@as(usize, 48), cfg.patch); // 16 * 3
    try std.testing.expectEqual(@as(usize, 6912), cfg.kdim());
    try std.testing.expectEqual(@as(usize, 1120), cfg.pos_size);
    try std.testing.expectEqual(@as(usize, 92160), cfg.minPixels());
    try std.testing.expectEqual(@as(usize, 645120), cfg.maxPixels());

    // Smart-resize snaps to 48-multiples within the token budget.
    const s = smartResize(1000, 600, cfg.patch, cfg.minPixels(), cfg.maxPixels());
    try std.testing.expectEqual(@as(usize, 0), s.w % 48);
    try std.testing.expectEqual(@as(usize, 0), s.h % 48);
    const toks = (s.w / 48) * (s.h / 48);
    try std.testing.expect(toks >= 40 and toks <= 280);
}
