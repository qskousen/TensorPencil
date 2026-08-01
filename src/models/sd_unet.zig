//! The Stable Diffusion UNet (LDM `openaimodel.UNetModel`) — SD1.5 today, SDXL by
//! configuration, mirroring `diffusers.UNet2DConditionModel` numerically and the LDM
//! single-file checkpoint's names structurally.
//!
//! This is the first *convolutional*, multi-resolution architecture in this engine;
//! everything else here is a stack of identical transformer blocks. Activations are
//! channel-last `[h*w][c]` f32, the same convention `wan_vae` and `ops.conv` use, so
//! a "row" is one pixel's channels and a GEMM over `h*w` rows is a 1x1 convolution.
//!
//! ## The graph, since the checkpoint's names do not show it
//!
//! ```
//! t --> sinusoidal(320) --> time_embed.0 --> silu --> time_embed.2 --> emb[1280]
//!
//! x[4] --> input_blocks.0.0 (conv 3x3)                          --> skip
//!          input_blocks.1,2   = [ResBlock, SpatialTransformer]   --> skip each
//!          input_blocks.3.0   = Downsample (conv 3x3 stride 2)   --> skip
//!          input_blocks.4,5   = [ResBlock(320->640), SpatialTransformer]
//!          input_blocks.6.0   = Downsample
//!          input_blocks.7,8   = [ResBlock(640->1280), SpatialTransformer]
//!          input_blocks.9.0   = Downsample
//!          input_blocks.10,11 = [ResBlock]            (no attention at 1/8 scale)
//!          middle_block       = [ResBlock, SpatialTransformer, ResBlock]
//!          output_blocks.0..11: each concatenates the matching skip, then
//!                               [ResBlock, (SpatialTransformer)?, (Upsample)?]
//! out.0 (GroupNorm) --> silu --> out.2 (conv 3x3) --> eps[4]
//! ```
//!
//! Four things are easy to get wrong and produce a model that still renders:
//!
//! 1. **The skip stack is LIFO and includes the stem.** `input_blocks.0`'s output is
//!    pushed too, so there are 12 skips for 12 output blocks; popping them in the
//!    wrong order gives a plausible-but-wrong image.
//! 2. **`ff.net.0.proj` is a GEGLU whose halves are (value, gate) in that order**,
//!    and the gate uses **erf** GELU (`F.gelu`'s default), not the tanh
//!    approximation. Swapping the halves is a silent quality loss.
//! 3. **`attn1`/`attn2` q/k/v have no bias**; only `to_out.0` does. A loader that
//!    expects biases everywhere fails loudly, but one that *invents* zeros for
//!    `to_out` would not.
//! 4. **The timestep embedding is diffusers' convention** — `flip_sin_to_cos` with
//!    `downscale_freq_shift = 0` — i.e. cos half first. See `ops.act`/the fixture:
//!    the wrong order is a smooth reparameterization the model half-absorbs.
//!
//! Loads from any `WeightStore` (safetensors or GGUF) and keeps large weights in
//! their checkpoint dtype so a quantized UNet runs and `ops.matmul.probe` can
//! attribute every GEMM — which is the entire point of adding this architecture:
//! ggufy's quantization programme has measured exactly one model so far.

const std = @import("std");
const tp_core = @import("tp_core");
const safetensors = tp_core.safetensors;
const weights_mod = tp_core.weights;
const ops = @import("tp_ops");

const DType = tp_core.dtype.DType;
const WeightStore = weights_mod.WeightStore;
const Weight = ops.matmul.Weight;
const Conv2d = ops.conv.Conv2d;

/// Everything that differs between SD1.5 and SDXL except the weights themselves.
pub const Config = struct {
    /// Latent channels in and out (4 for the SD-family AutoencoderKL).
    channels: usize,
    /// Base width; each level multiplies it.
    model_channels: usize,
    /// Per-level channel multipliers, outermost first.
    channel_mult: []const usize,
    /// ResBlocks per level (2 for both SD1.5 and SDXL).
    layers_per_block: usize,
    /// Which levels carry a SpatialTransformer, outermost first. SD1.5 attends at
    /// the first three of four levels.
    attn_levels: []const bool,
    /// Transformer blocks inside each SpatialTransformer (1 for SD1.5; SDXL varies
    /// per level, which is why this will become a slice when SDXL lands).
    transformer_depth: usize,
    /// Attention heads. LDM's `num_heads`: head_dim is the level's channels / this.
    heads: usize,
    /// Cross-attention context width (CLIP-L's 768 for SD1.5).
    context_dim: usize,
    /// Time-embedding width; LDM uses `4 * model_channels`.
    time_embed_dim: usize,
    /// GroupNorm groups (32 everywhere in the SD family) and its epsilon.
    norm_groups: usize,
    norm_eps: f32,

    pub fn levels(self: Config) usize {
        return self.channel_mult.len;
    }
};

pub const sd15: Config = .{
    .channels = 4,
    .model_channels = 320,
    .channel_mult = &.{ 1, 2, 4, 4 },
    .layers_per_block = 2,
    .attn_levels = &.{ true, true, true, false },
    .transformer_depth = 1,
    .heads = 8,
    .context_dim = 768,
    .time_embed_dim = 1280,
    .norm_groups = 32,
    .norm_eps = 1e-5,
};

// --- weight groups ----------------------------------------------------------

const Linear = struct {
    w: Weight,
    b: ?[]const f32,
};

const GroupNormW = struct {
    w: []const f32,
    b: []const f32,
};

const ResBlock = struct {
    in_norm: GroupNormW,
    in_conv: Conv2d,
    emb: Linear,
    out_norm: GroupNormW,
    out_conv: Conv2d,
    /// 1x1 convolution, present only where the channel count changes.
    skip: ?Conv2d,
    in_ch: usize,
    out_ch: usize,
};

const CrossAttn = struct {
    q: Weight,
    k: Weight,
    v: Weight,
    out: Linear,
    /// Width of the keys/values: the block's own channels for self-attention,
    /// `context_dim` for cross-attention.
    kv_dim: usize,
};

const TransformerBlock = struct {
    norm1: LayerNormW,
    attn1: CrossAttn,
    norm2: LayerNormW,
    attn2: CrossAttn,
    norm3: LayerNormW,
    /// GEGLU: `[2 * inner, ch]`, halves are (value, gate).
    ff_proj: Linear,
    ff_out: Linear,
    inner: usize,
};

const LayerNormW = struct {
    w: []const f32,
    b: []const f32,
};

const SpatialTransformer = struct {
    norm: GroupNormW,
    proj_in: Conv2d,
    blocks: []TransformerBlock,
    proj_out: Conv2d,
    channels: usize,
};

/// One entry of `input_blocks` / `output_blocks`: LDM stores a `TimestepEmbedSequential`
/// whose members are positional, so this mirrors "whatever was at index 0, 1, 2".
const Stage = struct {
    res: ?ResBlock = null,
    attn: ?SpatialTransformer = null,
    /// Downsample (input side) or Upsample (output side); both are one conv.
    sample: ?Conv2d = null,
    sample_kind: enum { none, down, up } = .none,
};

pub const UNet = struct {
    arena: std.heap.ArenaAllocator,
    cfg: Config,
    time_1: Linear,
    time_2: Linear,
    stem: Conv2d,
    input_stages: []Stage,
    mid_res1: ResBlock,
    mid_attn: SpatialTransformer,
    mid_res2: ResBlock,
    output_stages: []Stage,
    out_norm: GroupNormW,
    out_conv: Conv2d,

    pub fn load(gpa: std.mem.Allocator, store: WeightStore, cfg: Config, prefix: []const u8) !UNet {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();
        const l: Loader = .{ .store = store, .alloc = alloc, .pfx = prefix, .cfg = cfg };

        const time_1 = try l.linear("time_embed.0", .{}, cfg.time_embed_dim, cfg.model_channels, true);
        const time_2 = try l.linear("time_embed.2", .{}, cfg.time_embed_dim, cfg.time_embed_dim, true);
        const stem = try l.conv("input_blocks.0.0", .{}, cfg.model_channels, cfg.channels, 3, 1, 1);

        // --- input side ---
        var in_stages: std.ArrayList(Stage) = .empty;
        var ch = cfg.model_channels;
        var idx: usize = 1; // input_blocks.0 is the stem
        for (cfg.channel_mult, 0..) |mult, level| {
            const out_ch = cfg.model_channels * mult;
            for (0..cfg.layers_per_block) |_| {
                var st: Stage = .{ .res = try l.res("input_blocks.{d}.0", .{idx}, ch, out_ch) };
                if (cfg.attn_levels[level]) {
                    st.attn = try l.spatial("input_blocks.{d}.1", .{idx}, out_ch);
                }
                try in_stages.append(alloc, st);
                ch = out_ch;
                idx += 1;
            }
            // Every level but the last ends with a stride-2 convolution.
            if (level + 1 < cfg.levels()) {
                try in_stages.append(alloc, .{
                    .sample = try l.conv("input_blocks.{d}.0.op", .{idx}, ch, ch, 3, 2, 1),
                    .sample_kind = .down,
                });
                idx += 1;
            }
        }

        const mid_res1 = try l.res("middle_block.0", .{}, ch, ch);
        const mid_attn = try l.spatial("middle_block.1", .{}, ch);
        const mid_res2 = try l.res("middle_block.2", .{}, ch, ch);

        // --- output side ---
        // Channel bookkeeping mirrors LDM's own loop: it tracks the stack of skip
        // widths pushed on the way down, and each output ResBlock consumes
        // `ch + skip_ch` inputs.
        var skip_widths: std.ArrayList(usize) = .empty;
        defer skip_widths.deinit(alloc);
        try skip_widths.append(alloc, cfg.model_channels); // the stem's output
        {
            var c = cfg.model_channels;
            for (cfg.channel_mult, 0..) |mult, level| {
                const out_ch = cfg.model_channels * mult;
                for (0..cfg.layers_per_block) |_| {
                    try skip_widths.append(alloc, out_ch);
                    c = out_ch;
                }
                if (level + 1 < cfg.levels()) try skip_widths.append(alloc, c);
            }
        }

        var out_stages: std.ArrayList(Stage) = .empty;
        idx = 0;
        var level: usize = cfg.levels();
        while (level > 0) {
            level -= 1;
            const out_ch = cfg.model_channels * cfg.channel_mult[level];
            // One extra ResBlock per level on the way up (LDM: layers_per_block + 1).
            for (0..cfg.layers_per_block + 1) |j| {
                const skip_ch = skip_widths.pop().?;
                var st: Stage = .{ .res = try l.res("output_blocks.{d}.0", .{idx}, ch + skip_ch, out_ch) };
                ch = out_ch;
                var member: usize = 1;
                if (cfg.attn_levels[level]) {
                    st.attn = try l.spatial("output_blocks.{d}.1", .{idx}, out_ch);
                    member = 2;
                }
                // The upsample rides on the LAST block of each level except the
                // innermost-first one, and its member index depends on whether an
                // attention block preceded it — which is why this is positional.
                if (level > 0 and j == cfg.layers_per_block) {
                    st.sample = try l.convMember("output_blocks.{d}.{d}.conv", .{ idx, member }, ch, ch, 3, 1, 1);
                    st.sample_kind = .up;
                }
                try out_stages.append(alloc, st);
                idx += 1;
            }
        }

        const out_norm = try l.groupNorm("out.0", .{}, cfg.model_channels);
        const out_conv = try l.conv("out.2", .{}, cfg.channels, cfg.model_channels, 3, 1, 1);

        return .{
            .arena = arena,
            .cfg = cfg,
            .time_1 = time_1,
            .time_2 = time_2,
            .stem = stem,
            .input_stages = in_stages.items,
            .mid_res1 = mid_res1,
            .mid_attn = mid_attn,
            .mid_res2 = mid_res2,
            .output_stages = out_stages.items,
            .out_norm = out_norm,
            .out_conv = out_conv,
        };
    }

    pub fn deinit(self: *UNet) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

// --- loading ----------------------------------------------------------------

const Loader = struct {
    store: WeightStore,
    alloc: std.mem.Allocator,
    pfx: []const u8,
    cfg: Config,

    fn name(l: Loader, buf: []u8, comptime fmt: []const u8, args: anytype, suffix: []const u8) ![]u8 {
        var fbs = std.Io.Writer.fixed(buf);
        try fbs.writeAll(l.pfx);
        try fbs.print(fmt, args);
        try fbs.writeAll(suffix);
        return fbs.buffered();
    }

    fn view(l: Loader, nm: []const u8) !safetensors.TensorView {
        return l.store.get(nm) orelse {
            std.log.err("sd_unet: missing {s}", .{nm});
            return error.MissingTensor;
        };
    }

    /// A 2-D weight for a GEMM. Materializes any dtype the GEMM cannot read (an
    /// SD1.5 merge in the wild stores f64), and tags it with its checkpoint name so
    /// `ops.matmul.probe` can attribute the GEMM.
    fn mat(l: Loader, comptime fmt: []const u8, args: anytype, suffix: []const u8, rows: usize, cols: usize) !Weight {
        var buf: [200]u8 = undefined;
        const nm = try l.name(&buf, fmt, args, suffix);
        const v = try l.view(nm);
        const shape = v.info.shape.slice();
        // A 1x1 convolution is stored [out, in, 1, 1] and *is* a GEMM over pixels.
        const flat_cols = blk: {
            var c: usize = 1;
            for (shape[1..]) |d| c *= d;
            break :blk c;
        };
        if (shape.len < 2 or shape[0] != rows or flat_cols != cols) {
            std.log.err("sd_unet: {s} has shape {any} ({t}), expected [{d}, {d}]", .{ nm, shape, v.info.dtype, rows, cols });
            return error.ShapeMismatch;
        }
        var w = if (ops.matmul.supportsDType(v.info.dtype))
            Weight.init(v.bytes, v.info.dtype, rows, cols)
        else
            Weight.fromF32(try v.toF32Alloc(l.alloc), rows, cols);
        w.tag = try l.alloc.dupe(u8, nm);
        return w;
    }

    fn vec(l: Loader, comptime fmt: []const u8, args: anytype, suffix: []const u8, len: usize) ![]f32 {
        var buf: [200]u8 = undefined;
        const nm = try l.name(&buf, fmt, args, suffix);
        const v = try l.view(nm);
        if (v.info.elemCount() != len) {
            std.log.err("sd_unet: {s} has {d} elements, expected {d}", .{ nm, v.info.elemCount(), len });
            return error.ShapeMismatch;
        }
        return v.toF32Alloc(l.alloc);
    }

    fn linear(l: Loader, comptime fmt: []const u8, args: anytype, rows: usize, cols: usize, bias: bool) !Linear {
        return .{
            .w = try l.mat(fmt, args, ".weight", rows, cols),
            .b = if (bias) try l.vec(fmt, args, ".bias", rows) else null,
        };
    }

    fn groupNorm(l: Loader, comptime fmt: []const u8, args: anytype, ch: usize) !GroupNormW {
        return .{
            .w = try l.vec(fmt, args, ".weight", ch),
            .b = try l.vec(fmt, args, ".bias", ch),
        };
    }

    fn layerNorm(l: Loader, comptime fmt: []const u8, args: anytype, ch: usize) !LayerNormW {
        return .{
            .w = try l.vec(fmt, args, ".weight", ch),
            .b = try l.vec(fmt, args, ".bias", ch),
        };
    }

    /// A k×k convolution. `ops.conv` wants weights packed `[k][k][ci][co]`, so this
    /// repacks at load — once per weight rather than once per forward.
    fn conv(l: Loader, comptime fmt: []const u8, args: anytype, co: usize, ci: usize, k: usize, stride: usize, pad: usize) !Conv2d {
        return l.convMember(fmt, args, co, ci, k, stride, pad);
    }

    fn convMember(l: Loader, comptime fmt: []const u8, args: anytype, co: usize, ci: usize, k: usize, stride: usize, pad: usize) !Conv2d {
        var buf: [200]u8 = undefined;
        const nm = try l.name(&buf, fmt, args, ".weight");
        const v = try l.view(nm);
        const shape = v.info.shape.slice();
        if (shape.len != 4 or shape[0] != co or shape[1] != ci or shape[2] != k or shape[3] != k) {
            std.log.err("sd_unet: {s} has shape {any}, expected [{d}, {d}, {d}, {d}]", .{ nm, shape, co, ci, k, k });
            return error.ShapeMismatch;
        }
        const torch_w = try v.toF32Alloc(l.alloc);
        defer l.alloc.free(torch_w);
        // The GEMM this convolution becomes carries the checkpoint name, so an
        // activation capture can attribute it (see `ops.conv.Conv2d.tag`).
        const tag = try l.alloc.dupe(u8, nm);
        const packed_w = try ops.conv.packWeight(l.alloc, torch_w, co, ci, k);
        return .{
            .w = packed_w,
            .tag = tag,
            .b = try l.vec(fmt, args, ".bias", co),
            .co = co,
            .ci = ci,
            .k = k,
            .stride = stride,
            .pad = pad,
        };
    }

    fn res(l: Loader, comptime fmt: []const u8, args: anytype, in_ch: usize, out_ch: usize) !ResBlock {
        // The RELATIVE base (no store prefix): the inner helpers add the prefix
        // themselves, and passing an already-prefixed base through them doubled it —
        // `model.diffusion_model.model.diffusion_model.input_blocks…`, which surfaces
        // as a missing tensor rather than as the naming bug it is.
        var buf: [200]u8 = undefined;
        const base = try std.fmt.bufPrint(&buf, fmt, args);
        // The names are positional inside LDM's `nn.Sequential`s: in_layers is
        // [GroupNorm, SiLU, Conv], out_layers is [GroupNorm, SiLU, Dropout, Conv].
        const in_norm = try l.groupNorm("{s}.in_layers.0", .{base}, in_ch);
        const in_conv = try l.conv("{s}.in_layers.2", .{base}, out_ch, in_ch, 3, 1, 1);
        const emb = try l.linear("{s}.emb_layers.1", .{base}, out_ch, l.cfg.time_embed_dim, true);
        const out_norm = try l.groupNorm("{s}.out_layers.0", .{base}, out_ch);
        const out_conv = try l.conv("{s}.out_layers.3", .{base}, out_ch, out_ch, 3, 1, 1);
        const skip: ?Conv2d = if (in_ch != out_ch)
            try l.conv("{s}.skip_connection", .{base}, out_ch, in_ch, 1, 1, 0)
        else
            null;
        return .{
            .in_norm = in_norm,
            .in_conv = in_conv,
            .emb = emb,
            .out_norm = out_norm,
            .out_conv = out_conv,
            .skip = skip,
            .in_ch = in_ch,
            .out_ch = out_ch,
        };
    }

    fn spatial(l: Loader, comptime fmt: []const u8, args: anytype, ch: usize) !SpatialTransformer {
        var buf: [200]u8 = undefined;
        const base = try std.fmt.bufPrint(&buf, fmt, args); // relative; see `res`
        const norm = try l.groupNorm("{s}.norm", .{base}, ch);
        const proj_in = try l.conv("{s}.proj_in", .{base}, ch, ch, 1, 1, 0);
        const blocks = try l.alloc.alloc(TransformerBlock, l.cfg.transformer_depth);
        for (blocks, 0..) |*b, i| {
            const inner = ch * 4;
            b.* = .{
                .norm1 = try l.layerNorm("{s}.transformer_blocks.{d}.norm1", .{ base, i }, ch),
                // q/k/v carry NO bias in LDM's CrossAttention; only to_out.0 does.
                .attn1 = .{
                    .q = try l.mat("{s}.transformer_blocks.{d}.attn1.to_q", .{ base, i }, ".weight", ch, ch),
                    .k = try l.mat("{s}.transformer_blocks.{d}.attn1.to_k", .{ base, i }, ".weight", ch, ch),
                    .v = try l.mat("{s}.transformer_blocks.{d}.attn1.to_v", .{ base, i }, ".weight", ch, ch),
                    .out = try l.linear("{s}.transformer_blocks.{d}.attn1.to_out.0", .{ base, i }, ch, ch, true),
                    .kv_dim = ch,
                },
                .norm2 = try l.layerNorm("{s}.transformer_blocks.{d}.norm2", .{ base, i }, ch),
                .attn2 = .{
                    .q = try l.mat("{s}.transformer_blocks.{d}.attn2.to_q", .{ base, i }, ".weight", ch, ch),
                    .k = try l.mat("{s}.transformer_blocks.{d}.attn2.to_k", .{ base, i }, ".weight", ch, l.cfg.context_dim),
                    .v = try l.mat("{s}.transformer_blocks.{d}.attn2.to_v", .{ base, i }, ".weight", ch, l.cfg.context_dim),
                    .out = try l.linear("{s}.transformer_blocks.{d}.attn2.to_out.0", .{ base, i }, ch, ch, true),
                    .kv_dim = l.cfg.context_dim,
                },
                .norm3 = try l.layerNorm("{s}.transformer_blocks.{d}.norm3", .{ base, i }, ch),
                .ff_proj = try l.linear("{s}.transformer_blocks.{d}.ff.net.0.proj", .{ base, i }, inner * 2, ch, true),
                .ff_out = try l.linear("{s}.transformer_blocks.{d}.ff.net.2", .{ base, i }, ch, inner, true),
                .inner = inner,
            };
        }
        const proj_out = try l.conv("{s}.proj_out", .{base}, ch, ch, 1, 1, 0);
        return .{ .norm = norm, .proj_in = proj_in, .blocks = blocks, .proj_out = proj_out, .channels = ch };
    }
};

// --- forward ----------------------------------------------------------------

/// Scratch buffers for one resolution, allocated once per denoiser rather than per
/// step. A UNet forward touches activations from `[h*w][320]` down to
/// `[h/8·w/8][1280]`, so the workspace is sized by the largest of them.
pub const Workspace = struct {
    gpa: std.mem.Allocator,
    /// Two ping-pong activation buffers, each big enough for any stage.
    a: []f32,
    b: []f32,
    /// Skip-connection stack: `input_stages.len + 1` activations, contiguous.
    skips: [][]f32,
    /// Attention scratch (q/k/v/out at the widest stage) and the FF buffer.
    q: []f32,
    k: []f32,
    v: []f32,
    attn: []f32,
    ff: []f32,
    emb: []f32,

    pub fn init(gpa: std.mem.Allocator, u: *const UNet, lat_h: usize, lat_w: usize) !Workspace {
        const cfg = u.cfg;
        // The widest activation is the stem's, the deepest is the innermost level's;
        // one buffer that fits every (positions × channels) product covers both.
        var max_elems: usize = 0;
        var h = lat_h;
        var w = lat_w;
        for (cfg.channel_mult, 0..) |mult, level| {
            const ch = cfg.model_channels * mult;
            // An output-side ResBlock sees `ch + skip_ch` channels, at most 2× the
            // level's own width plus one level below it.
            max_elems = @max(max_elems, h * w * (ch * 3));
            if (level + 1 < cfg.levels()) {
                h = (h + 1) / 2;
                w = (w + 1) / 2;
            }
        }
        const widest_ch = cfg.model_channels * cfg.channel_mult[cfg.channel_mult.len - 1];
        const max_positions = lat_h * lat_w;

        const self: Workspace = .{
            .gpa = gpa,
            .a = try gpa.alloc(f32, max_elems),
            .b = try gpa.alloc(f32, max_elems),
            .skips = try gpa.alloc([]f32, u.input_stages.len + 1),
            .q = try gpa.alloc(f32, max_positions * widest_ch),
            .k = try gpa.alloc(f32, max_positions * widest_ch),
            .v = try gpa.alloc(f32, max_positions * widest_ch),
            .attn = try gpa.alloc(f32, max_positions * widest_ch),
            .ff = try gpa.alloc(f32, max_positions * widest_ch * 8),
            .emb = try gpa.alloc(f32, cfg.time_embed_dim),
        };
        for (self.skips) |*s| s.* = &.{};
        return self;
    }

    pub fn deinit(self: *Workspace) void {
        for (self.skips) |s| if (s.len > 0) self.gpa.free(s);
        self.gpa.free(self.skips);
        self.gpa.free(self.a);
        self.gpa.free(self.b);
        self.gpa.free(self.q);
        self.gpa.free(self.k);
        self.gpa.free(self.v);
        self.gpa.free(self.attn);
        self.gpa.free(self.ff);
        self.gpa.free(self.emb);
        self.* = undefined;
    }

    fn setSkip(self: *Workspace, i: usize, data: []const f32) !void {
        if (self.skips[i].len != data.len) {
            if (self.skips[i].len > 0) self.gpa.free(self.skips[i]);
            self.skips[i] = try self.gpa.alloc(f32, data.len);
        }
        @memcpy(self.skips[i], data);
    }
};

/// The sinusoidal timestep embedding, in diffusers' convention: **cos half first**
/// (`flip_sin_to_cos`) and an exponent denominator of `half` (`downscale_freq_shift
/// = 0`). Both are choices, and both are silently absorbable if wrong, which is why
/// the fixture pins them.
pub fn timestepEmbedding(out: []f32, timestep: f32) void {
    const dim = out.len;
    const half = dim / 2;
    // f64 internals, deliberately: diffusers computes this in f32, and at i = 0 the
    // argument is the timestep itself (~1000), so a 1e-7 relative slip in `freq`
    // becomes ~1e-4 in `cos(freq)`. Computing more accurately than the reference
    // bounds the disagreement by the *reference's* own rounding instead of stacking
    // two errors — which is why the fixture tolerance is 2e-4 and not 1e-6.
    const log_max: f64 = @log(10000.0);
    for (0..half) |i| {
        const exponent = -log_max * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(half));
        const freq = @exp(exponent) * @as(f64, timestep);
        out[i] = @floatCast(@cos(freq));
        out[half + i] = @floatCast(@sin(freq));
    }
    if (dim % 2 != 0) out[dim - 1] = 0;
}

/// eps = UNet(x, t, context). `x` and `out` are channel-last `[lat_h*lat_w][channels]`;
/// `context` is `[ctx_seq][context_dim]`.
pub fn forward(
    u: *const UNet,
    io: std.Io,
    gpa: std.mem.Allocator,
    ws: *Workspace,
    out: []f32,
    x: []const f32,
    lat_h: usize,
    lat_w: usize,
    timestep: f32,
    context: []const f32,
    ctx_seq: usize,
) !void {
    const cfg = u.cfg;
    std.debug.assert(x.len == lat_h * lat_w * cfg.channels);
    std.debug.assert(out.len == x.len);
    std.debug.assert(context.len == ctx_seq * cfg.context_dim);

    // --- time embedding ---
    {
        const sin = try gpa.alloc(f32, cfg.model_channels);
        defer gpa.free(sin);
        timestepEmbedding(sin, timestep);
        const hidden = try gpa.alloc(f32, cfg.time_embed_dim);
        defer gpa.free(hidden);
        try ops.matmul.matmul(io, gpa, hidden, sin, 1, u.time_1.w, u.time_1.b);
        ops.act.silu(hidden);
        try ops.matmul.matmul(io, gpa, ws.emb, hidden, 1, u.time_2.w, u.time_2.b);
    }

    // --- stem ---
    var h = lat_h;
    var w = lat_w;
    var cur = ws.a;
    var alt = ws.b;
    var ch = cfg.model_channels;
    try ops.conv.conv2d(io, gpa, cur[0 .. h * w * ch], x, h, w, u.stem);
    try ws.setSkip(0, cur[0 .. h * w * ch]);

    // --- input side ---
    for (u.input_stages, 0..) |stage, si| {
        if (stage.res) |rb| {
            try applyRes(io, gpa, ws, rb, cur, alt, h, w, cfg);
            std.mem.swap([]f32, &cur, &alt);
            ch = rb.out_ch;
        }
        if (stage.attn) |st| {
            try applySpatial(io, gpa, ws, st, cur[0 .. h * w * ch], h, w, context, ctx_seq, cfg);
        }
        if (stage.sample_kind == .down) {
            const oh = (h + 1) / 2;
            const ow = (w + 1) / 2;
            try ops.conv.conv2d(io, gpa, alt[0 .. oh * ow * ch], cur[0 .. h * w * ch], h, w, stage.sample.?);
            std.mem.swap([]f32, &cur, &alt);
            h = oh;
            w = ow;
        }
        try ws.setSkip(si + 1, cur[0 .. h * w * ch]);
    }

    // --- middle ---
    try applyRes(io, gpa, ws, u.mid_res1, cur, alt, h, w, cfg);
    std.mem.swap([]f32, &cur, &alt);
    ch = u.mid_res1.out_ch;
    try applySpatial(io, gpa, ws, u.mid_attn, cur[0 .. h * w * ch], h, w, context, ctx_seq, cfg);
    try applyRes(io, gpa, ws, u.mid_res2, cur, alt, h, w, cfg);
    std.mem.swap([]f32, &cur, &alt);
    ch = u.mid_res2.out_ch;

    // --- output side ---
    var skip_top = u.input_stages.len + 1;
    for (u.output_stages) |stage| {
        skip_top -= 1;
        const skip = ws.skips[skip_top];
        const skip_ch = skip.len / (h * w);
        // Concatenate along channels: [h*w][ch + skip_ch], current first.
        {
            const total = ch + skip_ch;
            var p: usize = h * w;
            // Walk backwards so an in-place widening never overwrites unread input.
            while (p > 0) {
                p -= 1;
                const dst = alt[p * total ..][0..total];
                @memcpy(dst[0..ch], cur[p * ch ..][0..ch]);
                @memcpy(dst[ch..], skip[p * skip_ch ..][0..skip_ch]);
            }
            std.mem.swap([]f32, &cur, &alt);
            ch = total;
        }
        const rb = stage.res.?;
        try applyRes(io, gpa, ws, rb, cur, alt, h, w, cfg);
        std.mem.swap([]f32, &cur, &alt);
        ch = rb.out_ch;
        if (stage.attn) |st| {
            try applySpatial(io, gpa, ws, st, cur[0 .. h * w * ch], h, w, context, ctx_seq, cfg);
        }
        if (stage.sample_kind == .up) {
            // Nearest-neighbour 2x, then a 3x3 convolution — LDM's `Upsample`.
            const oh = h * 2;
            const ow = w * 2;
            const up = try gpa.alloc(f32, oh * ow * ch);
            defer gpa.free(up);
            for (0..oh) |y| {
                for (0..ow) |xx| {
                    const src = cur[((y / 2) * w + (xx / 2)) * ch ..][0..ch];
                    @memcpy(up[(y * ow + xx) * ch ..][0..ch], src);
                }
            }
            try ops.conv.conv2d(io, gpa, alt[0 .. oh * ow * ch], up, oh, ow, stage.sample.?);
            std.mem.swap([]f32, &cur, &alt);
            h = oh;
            w = ow;
        }
    }

    // --- head ---
    ops.norm.groupNorm(cur[0 .. h * w * ch], cur[0 .. h * w * ch], ch, cfg.norm_groups, u.out_norm.w, u.out_norm.b, cfg.norm_eps);
    ops.act.silu(cur[0 .. h * w * ch]);
    try ops.conv.conv2d(io, gpa, out, cur[0 .. h * w * ch], h, w, u.out_conv);
}

fn applyRes(
    io: std.Io,
    gpa: std.mem.Allocator,
    ws: *Workspace,
    rb: ResBlock,
    src: []f32,
    dst: []f32,
    h: usize,
    w: usize,
    cfg: Config,
) !void {
    const n = h * w;
    const in_n = n * rb.in_ch;
    const out_n = n * rb.out_ch;

    // in_layers: GroupNorm -> SiLU -> conv3x3. The norm is out-of-place into scratch
    // because `src` is still needed for the residual.
    const tmp = try gpa.alloc(f32, in_n);
    defer gpa.free(tmp);
    ops.norm.groupNorm(tmp, src[0..in_n], rb.in_ch, cfg.norm_groups, rb.in_norm.w, rb.in_norm.b, cfg.norm_eps);
    ops.act.silu(tmp);
    try ops.conv.conv2d(io, gpa, dst[0..out_n], tmp, h, w, rb.in_conv);

    // emb_layers: SiLU on the shared time embedding, then a per-channel bias.
    {
        const e = try gpa.alloc(f32, cfg.time_embed_dim);
        defer gpa.free(e);
        @memcpy(e, ws.emb);
        ops.act.silu(e);
        const proj = try gpa.alloc(f32, rb.out_ch);
        defer gpa.free(proj);
        try ops.matmul.matmul(io, gpa, proj, e, 1, rb.emb.w, rb.emb.b);
        for (0..n) |p| {
            const row = dst[p * rb.out_ch ..][0..rb.out_ch];
            for (row, proj) |*o, pv| o.* += pv;
        }
    }

    // out_layers: GroupNorm -> SiLU -> conv3x3.
    const tmp2 = try gpa.alloc(f32, out_n);
    defer gpa.free(tmp2);
    ops.norm.groupNorm(tmp2, dst[0..out_n], rb.out_ch, cfg.norm_groups, rb.out_norm.w, rb.out_norm.b, cfg.norm_eps);
    ops.act.silu(tmp2);
    const conved = try gpa.alloc(f32, out_n);
    defer gpa.free(conved);
    try ops.conv.conv2d(io, gpa, conved, tmp2, h, w, rb.out_conv);

    // residual: the skip projection when the width changes, else the input itself.
    if (rb.skip) |sk| {
        try ops.conv.conv2d(io, gpa, dst[0..out_n], src[0..in_n], h, w, sk);
        for (dst[0..out_n], conved) |*o, c| o.* += c;
    } else {
        @memcpy(dst[0..out_n], src[0..out_n]);
        for (dst[0..out_n], conved) |*o, c| o.* += c;
    }
}

fn applySpatial(
    io: std.Io,
    gpa: std.mem.Allocator,
    ws: *Workspace,
    st: SpatialTransformer,
    x: []f32,
    h: usize,
    w: usize,
    context: []const f32,
    ctx_seq: usize,
    cfg: Config,
) !void {
    const n = h * w;
    const ch = st.channels;
    std.debug.assert(x.len == n * ch);

    // GroupNorm -> proj_in (1x1) into the transformer's residual stream.
    const stream = try gpa.alloc(f32, n * ch);
    defer gpa.free(stream);
    ops.norm.groupNorm(stream, x, ch, cfg.norm_groups, st.norm.w, st.norm.b, cfg.norm_eps);
    const projected = try gpa.alloc(f32, n * ch);
    defer gpa.free(projected);
    try ops.conv.conv2d(io, gpa, projected, stream, h, w, st.proj_in);
    @memcpy(stream, projected);

    const head_dim = ch / cfg.heads;
    const norm_buf = try gpa.alloc(f32, n * ch);
    defer gpa.free(norm_buf);

    for (st.blocks) |b| {
        // attn1: self-attention over pixels, no mask.
        ops.norm.layerNorm(norm_buf, stream, b.norm1.w, b.norm1.b, cfg.norm_eps);
        try ops.matmul.matmul(io, gpa, ws.q[0 .. n * ch], norm_buf, n, b.attn1.q, null);
        try ops.matmul.matmul(io, gpa, ws.k[0 .. n * ch], norm_buf, n, b.attn1.k, null);
        try ops.matmul.matmul(io, gpa, ws.v[0 .. n * ch], norm_buf, n, b.attn1.v, null);
        try ops.attention.attention(io, gpa, ws.attn[0 .. n * ch], ws.q[0 .. n * ch], ws.k[0 .. n * ch], ws.v[0 .. n * ch], .{
            .seq_q = n,
            .seq_kv = n,
            .n_heads = cfg.heads,
            .n_kv_heads = cfg.heads,
            .head_dim = head_dim,
        });
        try ops.matmul.matmul(io, gpa, norm_buf, ws.attn[0 .. n * ch], n, b.attn1.out.w, b.attn1.out.b);
        for (stream, norm_buf) |*s, r| s.* += r;

        // attn2: cross-attention onto the text conditioning.
        ops.norm.layerNorm(norm_buf, stream, b.norm2.w, b.norm2.b, cfg.norm_eps);
        try ops.matmul.matmul(io, gpa, ws.q[0 .. n * ch], norm_buf, n, b.attn2.q, null);
        try ops.matmul.matmul(io, gpa, ws.k[0 .. ctx_seq * ch], context, ctx_seq, b.attn2.k, null);
        try ops.matmul.matmul(io, gpa, ws.v[0 .. ctx_seq * ch], context, ctx_seq, b.attn2.v, null);
        try ops.attention.attention(io, gpa, ws.attn[0 .. n * ch], ws.q[0 .. n * ch], ws.k[0 .. ctx_seq * ch], ws.v[0 .. ctx_seq * ch], .{
            .seq_q = n,
            .seq_kv = ctx_seq,
            .n_heads = cfg.heads,
            .n_kv_heads = cfg.heads,
            .head_dim = head_dim,
        });
        try ops.matmul.matmul(io, gpa, norm_buf, ws.attn[0 .. n * ch], n, b.attn2.out.w, b.attn2.out.b);
        for (stream, norm_buf) |*s, r| s.* += r;

        // ff: GEGLU. `proj` emits [value | gate] per row, in that order, and the gate
        // takes the ERF gelu — diffusers' `GEGLU` uses `F.gelu`'s default.
        ops.norm.layerNorm(norm_buf, stream, b.norm3.w, b.norm3.b, cfg.norm_eps);
        const two = 2 * b.inner;
        try ops.matmul.matmul(io, gpa, ws.ff[0 .. n * two], norm_buf, n, b.ff_proj.w, b.ff_proj.b);
        const gated = try gpa.alloc(f32, n * b.inner);
        defer gpa.free(gated);
        for (0..n) |p| {
            const row = ws.ff[p * two ..][0..two];
            const value = row[0..b.inner];
            const gate = row[b.inner..];
            const dst = gated[p * b.inner ..][0..b.inner];
            for (dst, value, gate) |*d, val, g| d.* = val * ops.act.geluErfScalar(g);
        }
        try ops.matmul.matmul(io, gpa, norm_buf, gated, n, b.ff_out.w, b.ff_out.b);
        for (stream, norm_buf) |*s, r| s.* += r;
    }

    // proj_out (1x1) and the outer residual.
    try ops.conv.conv2d(io, gpa, projected, stream, h, w, st.proj_out);
    for (x, projected) |*o, p| o.* += p;
}

// --- tests ------------------------------------------------------------------

const testing = std.testing;
const test_gate = @import("../test_gate.zig");

const ref_path = "src/models/assets/sd15_ref.safetensors";
const sd15_ckpt = "/home/qt/genai/comfyui/models/checkpoints/sd1.5/perfectdeliberate_v20.safetensors";

/// torch planar `[c][h][w]` -> our channel-last `[h*w][c]`.
fn plannarToChannelLast(dst: []f32, src: []const f32, c: usize, h: usize, w: usize) void {
    std.debug.assert(dst.len == c * h * w and src.len == dst.len);
    for (0..h * w) |p| {
        for (0..c) |ci| dst[p * c + ci] = src[ci * h * w + p];
    }
}

test "the timestep embedding matches diffusers' get_timestep_embedding" {
    const gpa = testing.allocator;
    const Case = struct { timestep: f32, dim: usize, expected: []const f32 };
    const Fixtures = struct { timestep_embedding: []const Case };
    var parsed = try std.json.parseFromSlice(
        Fixtures,
        gpa,
        @embedFile("../ops/assets/sd15_op_fixtures.json"),
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();

    for (parsed.value.timestep_embedding) |c| {
        const out = try gpa.alloc(f32, c.dim);
        defer gpa.free(out);
        timestepEmbedding(out, c.timestep);
        for (c.expected, out, 0..) |e, a, i| {
            errdefer std.debug.print("t={d} idx {d}: expected {d:.6} got {d:.6}\n", .{ c.timestep, i, e, a });
            // 2e-4: see the f64 note in `timestepEmbedding`. The residual is the
            // reference's own f32 rounding amplified by cos of a large argument, so a
            // tighter bound would be pinning torch's arithmetic order, not our math.
            try testing.expectApproxEqAbs(e, a, 2e-4);
        }
    }
}

test "the SD1.5 UNet matches diffusers.UNet2DConditionModel on a real checkpoint" {
    const gpa = testing.allocator;
    const io = testing.io;
    try test_gate.requireIntegration();
    try test_gate.requireModelFile(io, sd15_ckpt);
    try test_gate.requireModelFile(io, ref_path);

    var ref = try safetensors.SafeTensors.open(gpa, io, ref_path);
    defer ref.deinit();
    var ck = try safetensors.SafeTensors.open(gpa, io, sd15_ckpt);
    defer ck.deinit();

    var u = try UNet.load(gpa, .{ .safetensors = &ck }, sd15, "model.diffusion_model.");
    defer u.deinit();

    const latent = try ref.get("unet.latent").?.toF32Alloc(gpa); // [2][4][16][16]
    defer gpa.free(latent);
    const context = try ref.get("clip.hidden").?.toF32Alloc(gpa); // [2][77][768]
    defer gpa.free(context);
    const eps_ref = try ref.get("unet.eps").?.toF32Alloc(gpa);
    defer gpa.free(eps_ref);
    const temb_ref = try ref.get("unet.temb").?.toF32Alloc(gpa); // [2][1280]
    defer gpa.free(temb_ref);
    const t_view = ref.get("unet.timestep").?;
    const timesteps = std.mem.bytesAsSlice(i32, t_view.bytes);

    const lat = 16;
    const n = lat * lat;
    const cfg = sd15;
    const per_item = n * cfg.channels;
    const ctx_seq = 77;

    // The time path on its own first: it feeds every ResBlock, so a wrong embedding
    // convention corrupts the whole forward uniformly and looks like nothing specific.
    {
        var ws = try Workspace.init(gpa, &u, lat, lat);
        defer ws.deinit();
        const x = try gpa.alloc(f32, per_item);
        defer gpa.free(x);
        @memset(x, 0);
        const out = try gpa.alloc(f32, per_item);
        defer gpa.free(out);
        try forward(&u, io, gpa, &ws, out, x, lat, lat, @floatFromInt(timesteps[0]), context[0 .. ctx_seq * cfg.context_dim], ctx_seq);
        for (temb_ref[0..cfg.time_embed_dim], ws.emb, 0..) |e, a, i| {
            errdefer std.debug.print("temb idx {d}: expected {d:.6} got {d:.6}\n", .{ i, e, a });
            try testing.expectApproxEqAbs(e, a, 2e-4);
        }
    }

    for (0..2) |b| {
        var ws = try Workspace.init(gpa, &u, lat, lat);
        defer ws.deinit();

        const x = try gpa.alloc(f32, per_item);
        defer gpa.free(x);
        plannarToChannelLast(x, latent[b * per_item ..][0..per_item], cfg.channels, lat, lat);

        const out = try gpa.alloc(f32, per_item);
        defer gpa.free(out);
        try forward(
            &u,
            io,
            gpa,
            &ws,
            out,
            x,
            lat,
            lat,
            @floatFromInt(timesteps[b]),
            context[b * ctx_seq * cfg.context_dim ..][0 .. ctx_seq * cfg.context_dim],
            ctx_seq,
        );

        const want = try gpa.alloc(f32, per_item);
        defer gpa.free(want);
        plannarToChannelLast(want, eps_ref[b * per_item ..][0..per_item], cfg.channels, lat, lat);

        var max_abs: f32 = 0;
        var l2_ref: f64 = 0;
        var l2_err: f64 = 0;
        for (want, out) |e, a| {
            max_abs = @max(max_abs, @abs(e - a));
            l2_ref += @as(f64, e) * e;
            l2_err += @as(f64, e - a) * (e - a);
        }
        const rel = @sqrt(l2_err / l2_ref);
        errdefer std.debug.print("unet eps batch {d}: rel L2 {d:.6} max_abs {d:.6}\n", .{ b, rel, max_abs });
        // f32 accumulation order differs from torch's throughout a 25-block network,
        // so this is a numeric-agreement bound, not bit-identity. 1e-3 relative over
        // 1024 outputs is far tighter than any structural error could survive.
        try testing.expect(rel < 1e-3);
    }
}

test "the UNet still matches diffusers at the size images are actually rendered at" {
    // The 16x16 case above cannot catch a scale-dependent bug: there `ops.conv`'s
    // im2col fits one band and self-attention sees 256 positions, where a 512x512
    // render bands 3+ ways and attends over 4096. A grid artifact with a 4-latent-pixel
    // period reached a rendered image before this test existed.
    const gpa = testing.allocator;
    const io = testing.io;
    try test_gate.requireIntegration();
    try test_gate.requireModelFile(io, sd15_ckpt);
    try test_gate.requireModelFile(io, ref_path);

    var ref = try safetensors.SafeTensors.open(gpa, io, ref_path);
    defer ref.deinit();
    var ck = try safetensors.SafeTensors.open(gpa, io, sd15_ckpt);
    defer ck.deinit();

    var u = try UNet.load(gpa, .{ .safetensors = &ck }, sd15, "model.diffusion_model.");
    defer u.deinit();

    const lat = 64;
    const cfg = sd15;
    const n = lat * lat * cfg.channels;
    const ctx_seq = 77;

    const latent = try ref.get("unet_big.latent").?.toF32Alloc(gpa);
    defer gpa.free(latent);
    const eps_ref = try ref.get("unet_big.eps").?.toF32Alloc(gpa);
    defer gpa.free(eps_ref);
    const context = try ref.get("clip.hidden").?.toF32Alloc(gpa);
    defer gpa.free(context);
    const t_view = ref.get("unet.timestep").?;
    const timesteps = std.mem.bytesAsSlice(i32, t_view.bytes);

    var ws = try Workspace.init(gpa, &u, lat, lat);
    defer ws.deinit();

    const x = try gpa.alloc(f32, n);
    defer gpa.free(x);
    plannarToChannelLast(x, latent, cfg.channels, lat, lat);

    const out = try gpa.alloc(f32, n);
    defer gpa.free(out);
    try forward(&u, io, gpa, &ws, out, x, lat, lat, @floatFromInt(timesteps[0]), context[0 .. ctx_seq * cfg.context_dim], ctx_seq);

    const want = try gpa.alloc(f32, n);
    defer gpa.free(want);
    plannarToChannelLast(want, eps_ref, cfg.channels, lat, lat);

    var l2_ref: f64 = 0;
    var l2_err: f64 = 0;
    for (want, out) |e, a| {
        l2_ref += @as(f64, e) * e;
        l2_err += @as(f64, e - a) * (e - a);
    }
    const rel = @sqrt(l2_err / l2_ref);

    // ⚠️ A global rel-L2 bound hides a *structured* error, and structure is what
    // matters: a periodic error compounds coherently over a sampling loop where random
    // error cancels. So this also checks that the error is not concentrated on a
    // spatial grid — the artifact that reached a rendered image was a 4-latent-pixel
    // checkerboard sitting under a rel L2 of well under 1e-3.
    // Period 4, not 2: the artifact that reached a rendered image modulated the latent
    // with a 4-pixel period (one pixel at the UNet's 16x16 level, upsampled twice), and
    // a 2x2 phase test is blind to it.
    const period = 4;
    var err_by_phase: [period * period]f64 = @splat(0);
    var cnt_by_phase: [period * period]f64 = @splat(0);
    for (0..lat) |y| {
        for (0..lat) |xx| {
            const phase = ((y % period) * period) + (xx % period);
            for (0..cfg.channels) |c| {
                const idx = (y * lat + xx) * cfg.channels + c;
                const d = @as(f64, want[idx]) - out[idx];
                err_by_phase[phase] += d * d;
                cnt_by_phase[phase] += 1;
            }
        }
    }
    var lo: f64 = std.math.inf(f64);
    var hi: f64 = 0;
    for (err_by_phase, cnt_by_phase) |e, cnt| {
        const rms = @sqrt(e / cnt);
        lo = @min(lo, rms);
        hi = @max(hi, rms);
    }
    errdefer std.debug.print(
        "unet eps at latent {d}: rel L2 {e:.4}; per-2x2-phase error rms {e:.4}..{e:.4} (ratio {d:.2})\n",
        .{ lat, rel, lo, hi, if (lo > 0) hi / lo else 0 },
    );
    try testing.expect(rel < 1e-3);
    // A clean implementation has no phase preference; a checkerboard has a large one.
    try testing.expect(hi / lo < 2.0);

    // ⚠️ **A second forward through the SAME workspace must give the same answer.** The
    // parity tests above each build a fresh `Workspace` and call `forward` once; a real
    // render calls it 40 times (20 steps x CFG) through one workspace, so any state
    // that leaks between calls is invisible to a single-shot test and compounds
    // coherently over a sampling loop.
    const again = try gpa.alloc(f32, n);
    defer gpa.free(again);
    try forward(&u, io, gpa, &ws, again, x, lat, lat, @floatFromInt(timesteps[0]), context[0 .. ctx_seq * cfg.context_dim], ctx_seq);
    var max_drift: f32 = 0;
    for (out, again) |a, b| max_drift = @max(max_drift, @abs(a - b));
    errdefer std.debug.print("second forward through the same workspace drifted by {e:.4}\n", .{max_drift});
    try testing.expectEqual(@as(f32, 0), max_drift);
}
