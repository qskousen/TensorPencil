//! Wan 2.1 VAE decoder (the Krea 2 VAE), specialized for still images.
//!
//! With a single frame (T=1) and no feature cache, every causal 3D conv in
//! ComfyUI's reference (`comfy/ldm/wan/vae.py`) sees only zero padding in
//! front, so just the *last* temporal slice of each kernel touches real data
//! and the whole decoder collapses to 2D convs; the `time_conv`s are skipped
//! entirely. Weights are repacked accordingly at load time.
//!
//! This checkpoint: base dim 96, z_dim 16, dim_mult [1,2,4,4], 2 res blocks
//! per stage, attention only in the middle block. Norms are per-position
//! channel L2 norms (F.normalize * sqrt(C) * gamma, eps 1e-12).
//!
//! Activations are channel-last [h*w, c] f32; the public API takes/returns
//! torch-style planar [c][h][w].

const std = @import("std");
const safetensors = @import("../safetensors.zig");
const ops = @import("../ops.zig");

const SafeTensors = safetensors.SafeTensors;
const Weight = ops.matmul.Weight;

pub const latent_channels = 16;
pub const spatial_scale = 8;

/// Per-channel latent normalization from comfy/latent_formats.py (Wan21).
/// The DiT samples in normalized space; decode input = z * std + mean.
pub const latents_mean = [latent_channels]f32{ -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921 };
pub const latents_std = [latent_channels]f32{ 2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160 };

/// Linear latent→RGB approximation (WAN 2.1 factors, from ComfyUI's
/// latent_formats.Wan21). Maps the 16-channel sampler latent to a rough RGB
/// preview with a per-pixel 16×3 matmul — no VAE needed, cheap enough to run
/// every sampling step for a live preview.
pub const latent_rgb_factors = [latent_channels][3]f32{
    .{ -0.1299, -0.1692, 0.2932 },
    .{ 0.0671, 0.0406, 0.0442 },
    .{ 0.3568, 0.2548, 0.1747 },
    .{ 0.0372, 0.2344, 0.1420 },
    .{ 0.0313, 0.0189, -0.0328 },
    .{ 0.0296, -0.0956, -0.0665 },
    .{ -0.3477, -0.4059, -0.2925 },
    .{ 0.0166, 0.1902, 0.1975 },
    .{ -0.0412, 0.0267, -0.1364 },
    .{ -0.1293, 0.0740, 0.1636 },
    .{ 0.0680, 0.3019, 0.1128 },
    .{ 0.0032, 0.0581, 0.0639 },
    .{ -0.1251, 0.0927, 0.1699 },
    .{ 0.0060, -0.0633, 0.0005 },
    .{ 0.3477, 0.2275, 0.2950 },
    .{ 0.1984, 0.0913, 0.1861 },
};
pub const latent_rgb_bias = [3]f32{ -0.1835, -0.0868, -0.3360 };

/// Fill `rgb_out` ([zh*zw][3] RGB8) with the latent2rgb preview of the planar
/// [16][zh*zw] normalized sampler latent `z`.
pub fn latentPreviewInto(rgb_out: []u8, z: []const f32, zh: usize, zw: usize) void {
    const plane = zh * zw;
    std.debug.assert(rgb_out.len >= plane * 3 and z.len >= latent_channels * plane);
    for (0..plane) |p| {
        var acc = latent_rgb_bias;
        inline for (0..latent_channels) |c| {
            const v = z[c * plane + p];
            acc[0] += v * latent_rgb_factors[c][0];
            acc[1] += v * latent_rgb_factors[c][1];
            acc[2] += v * latent_rgb_factors[c][2];
        }
        inline for (0..3) |ch| {
            const u = std.math.clamp((acc[ch] + 1.0) * 0.5, 0.0, 1.0) * 255.0;
            rgb_out[p * 3 + ch] = @intFromFloat(u);
        }
    }
}

pub const Conv2d = struct {
    /// Repacked [co][kh][kw][ci] to match im2col patch layout.
    w: []const f32,
    b: []const f32,
    co: usize,
    ci: usize,
    k: usize, // 1 or 3, padding (k-1)/2
};

pub const ResBlock = struct {
    norm1: []const f32, // gamma [ci]
    conv1: Conv2d,
    norm2: []const f32,
    conv2: Conv2d,
    shortcut: ?Conv2d, // 1x1 when ci != co
};

pub const AttnBlock = struct {
    norm: []const f32,
    qkv: Conv2d, // 1x1, co = 3*ci
    proj: Conv2d, // 1x1
};

pub const Layer = union(enum) {
    res: ResBlock,
    /// nearest-exact 2x upsample followed by a 3x3 conv halving channels.
    up: Conv2d,
};

pub const Decoder = struct {
    arena: std.heap.ArenaAllocator,
    post_quant: Conv2d, // "conv2", 1x1 16->16
    conv_in: Conv2d, // 16 -> 384
    mid_res1: ResBlock,
    mid_attn: AttnBlock,
    mid_res2: ResBlock,
    ups: []Layer,
    head_norm: []const f32,
    head_conv: Conv2d, // 96 -> 3

    pub fn load(gpa: std.mem.Allocator, st: *const SafeTensors) !Decoder {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();

        const ups = try alloc.alloc(Layer, 15);
        // Stage layout for dim 96, mult [1,2,4,4], 2 res blocks (see module doc).
        // upsamples.{0,1,2}: res 384; .3: up 384->192; .4: res 192->384 (shortcut);
        // .{5,6}: res 384; .7: up 384->192; .{8,9,10}: res 192; .11: up 192->96;
        // .{12,13,14}: res 96.
        var idx: usize = 0;
        const stages = [4]struct { in: usize, out: usize, up: bool }{
            .{ .in = 384, .out = 384, .up = true },
            .{ .in = 192, .out = 384, .up = true },
            .{ .in = 192, .out = 192, .up = true },
            .{ .in = 96, .out = 96, .up = false },
        };
        for (stages) |stage| {
            var ci = stage.in;
            for (0..3) |_| {
                ups[idx] = .{ .res = try loadRes(alloc, st, "decoder.upsamples", idx, ci, stage.out) };
                ci = stage.out;
                idx += 1;
            }
            if (stage.up) {
                const name = try std.fmt.allocPrint(alloc, "decoder.upsamples.{d}.resample.1", .{idx});
                ups[idx] = .{ .up = try loadConv(alloc, st, name, stage.out, stage.out / 2, 3) };
                idx += 1;
            }
        }
        std.debug.assert(idx == 15);

        // All arena allocations must happen before `arena` is copied into the
        // result — chunks allocated afterwards would be missed by deinit.
        const post_quant = try loadConv(alloc, st, "conv2", 16, 16, 1);
        const conv_in = try loadConv(alloc, st, "decoder.conv1", 16, 384, 3);
        const mid_res1 = try loadRes(alloc, st, "decoder.middle", 0, 384, 384);
        const mid_attn: AttnBlock = .{
            .norm = try loadGamma(alloc, st, "decoder.middle.1.norm.gamma", 384),
            .qkv = try loadConv(alloc, st, "decoder.middle.1.to_qkv", 384, 1152, 1),
            .proj = try loadConv(alloc, st, "decoder.middle.1.proj", 384, 384, 1),
        };
        const mid_res2 = try loadRes(alloc, st, "decoder.middle", 2, 384, 384);
        const head_norm = try loadGamma(alloc, st, "decoder.head.0.gamma", 96);
        const head_conv = try loadConv(alloc, st, "decoder.head.2", 96, 3, 3);

        return .{
            .arena = arena,
            .post_quant = post_quant,
            .conv_in = conv_in,
            .mid_res1 = mid_res1,
            .mid_attn = mid_attn,
            .mid_res2 = mid_res2,
            .ups = ups,
            .head_norm = head_norm,
            .head_conv = head_conv,
        };
    }

    pub fn deinit(self: *Decoder) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// Rough upper bound on the PEAK device VRAM (bytes) a whole-image GPU decode
    /// of a [16][zh][zw] latent needs for its ACTIVATIONS: the three grow-and-keep
    /// ping-pong buffers (x/t/u in vae_cuda / vae_gpu, each sized to the largest
    /// activation `max(h·w·ch)`) plus the capped im2col patch band. Mirrors
    /// decode()'s resolution/channel walk (res keeps resolution; up doubles h,w).
    /// Deliberately approximate — it can't see the opaque cuBLASLt/cuDNN conv
    /// workspace — so callers pre-free with a margin and keep a reactive fallback.
    /// (For a tiled decode, call with the tile side to bound a single tile.)
    pub fn estimatePeakBytes(self: *const Decoder, zh: usize, zw: usize) u64 {
        var h = zh;
        var w = zw;
        var max_act: u64 = @as(u64, zh) * zw * self.conv_in.co; // latent res, 384ch
        for (self.ups) |layer| switch (layer) {
            .res => |rb| {
                const e = @as(u64, h) * w * @max(rb.conv1.ci, rb.conv2.co);
                if (e > max_act) max_act = e;
            },
            .up => |cv| {
                h *= 2; // fused nearest-2x upsample: output lands at the doubled res
                w *= 2;
                const e = @as(u64, h) * w * cv.co;
                if (e > max_act) max_act = e;
            },
        };
        const head_in = @as(u64, h) * w * self.head_conv.ci; // full res, 96ch
        if (head_in > max_act) max_act = head_in;
        const patch_band: u64 = 256 << 20; // vae_cuda patch_band_bytes cap
        return 3 * max_act * 4 + patch_band;
    }

    /// Decode a VAE-space latent (planar [16][zh][zw], already denormalized)
    /// to planar [3][8*zh][8*zw] pixels in [-1, 1]. Caller frees the result.
    pub fn decode(self: *const Decoder, io: std.Io, gpa: std.mem.Allocator, z: []const f32, zh: usize, zw: usize) ![]f32 {
        std.debug.assert(z.len == latent_channels * zh * zw);

        // planar -> channel-last
        var x = try planarToRows(gpa, z, latent_channels, zh * zw);
        var h = zh;
        var w = zw;

        x = try self.applyConv(io, gpa, x, h, w, self.post_quant);
        x = try self.applyConv(io, gpa, x, h, w, self.conv_in);
        x = try self.applyRes(io, gpa, x, h, w, self.mid_res1);
        x = try self.applyAttn(io, gpa, x, h, w, self.mid_attn);
        x = try self.applyRes(io, gpa, x, h, w, self.mid_res2);

        for (self.ups) |layer| switch (layer) {
            .res => |rb| x = try self.applyRes(io, gpa, x, h, w, rb),
            .up => |conv| {
                const up = try nearest2x(gpa, x, h, w, conv.ci);
                gpa.free(x);
                x = up;
                h *= 2;
                w *= 2;
                x = try self.applyConv(io, gpa, x, h, w, conv);
            },
        };

        // head: norm + silu + conv
        channelRmsNorm(x, self.head_norm);
        ops.act.silu(x);
        x = try self.applyConv(io, gpa, x, h, w, self.head_conv);

        defer gpa.free(x);
        return rowsToPlanar(gpa, x, 3, h * w);
    }

    /// conv2d that frees the input and returns a fresh output buffer.
    fn applyConv(self: *const Decoder, io: std.Io, gpa: std.mem.Allocator, x: []f32, h: usize, w: usize, conv: Conv2d) ![]f32 {
        _ = self;
        const out = try gpa.alloc(f32, h * w * conv.co);
        errdefer gpa.free(out);
        try conv2d(io, gpa, out, x, h, w, conv);
        gpa.free(x);
        return out;
    }

    fn applyRes(self: *const Decoder, io: std.Io, gpa: std.mem.Allocator, x: []f32, h: usize, w: usize, rb: ResBlock) ![]f32 {
        _ = self;
        const n = h * w;
        // t = silu(norm1(x))
        const t = try gpa.alloc(f32, n * rb.conv1.ci);
        defer gpa.free(t);
        @memcpy(t, x);
        channelRmsNorm(t, rb.norm1);
        ops.act.silu(t);
        // u = conv1(t)
        const u = try gpa.alloc(f32, n * rb.conv1.co);
        defer gpa.free(u);
        try conv2d(io, gpa, u, t, h, w, rb.conv1);
        // u = silu(norm2(u)); out = conv2(u)
        channelRmsNorm(u, rb.norm2);
        ops.act.silu(u);
        const out = try gpa.alloc(f32, n * rb.conv2.co);
        errdefer gpa.free(out);
        try conv2d(io, gpa, out, u, h, w, rb.conv2);
        // residual: out += shortcut(x)
        if (rb.shortcut) |sc| {
            const shortcut = try gpa.alloc(f32, n * sc.co);
            defer gpa.free(shortcut);
            try conv2d(io, gpa, shortcut, x, h, w, sc);
            for (out, shortcut) |*o, s| o.* += s;
        } else {
            for (out, x) |*o, s| o.* += s;
        }
        gpa.free(x);
        return out;
    }

    fn applyAttn(self: *const Decoder, io: std.Io, gpa: std.mem.Allocator, x: []f32, h: usize, w: usize, ab: AttnBlock) ![]f32 {
        _ = self;
        const n = h * w;
        const c = ab.qkv.ci;
        const t = try gpa.alloc(f32, n * c);
        defer gpa.free(t);
        @memcpy(t, x);
        channelRmsNorm(t, ab.norm);

        // q/k/v = 1x1 conv slices of the packed qkv weight ([3c][ci] row-major).
        const q = try gpa.alloc(f32, n * c);
        defer gpa.free(q);
        const k_ = try gpa.alloc(f32, n * c);
        defer gpa.free(k_);
        const v = try gpa.alloc(f32, n * c);
        defer gpa.free(v);
        inline for (.{ q, k_, v }, 0..) |dst, part| {
            const wslice = ab.qkv.w[part * c * c ..][0 .. c * c];
            const bslice = ab.qkv.b[part * c ..][0..c];
            try ops.matmul.matmul(io, gpa, dst, t, n, Weight.fromF32(wslice, c, c), bslice);
        }

        // Single-head attention over all spatial positions.
        const attn_out = try gpa.alloc(f32, n * c);
        defer gpa.free(attn_out);
        try ops.attention.attention(io, gpa, attn_out, q, k_, v, .{
            .seq_q = n,
            .seq_kv = n,
            .n_heads = 1,
            .n_kv_heads = 1,
            .head_dim = c,
        });

        const out = try gpa.alloc(f32, n * c);
        errdefer gpa.free(out);
        try ops.matmul.matmul(io, gpa, out, attn_out, n, Weight.fromF32(ab.proj.w, c, c), ab.proj.b);
        for (out, x) |*o, s| o.* += s;
        gpa.free(x);
        return out;
    }
};

// --- building blocks -------------------------------------------------------

/// Per-position channel norm: x[i] = x[i] / max(||x||_2, 1e-12) * sqrt(c) * gamma.
fn channelRmsNorm(x: []f32, gamma: []const f32) void {
    const c = gamma.len;
    const scale = @sqrt(@as(f32, @floatFromInt(c)));
    var row: usize = 0;
    while (row < x.len) : (row += c) {
        const xr = x[row..][0..c];
        var sum: f32 = 0;
        for (xr) |v| sum += v * v;
        const inv = scale / @max(@sqrt(sum), 1e-12);
        for (xr, gamma) |*v, g| v.* *= inv * g;
    }
}

/// Zero-padded conv over channel-last activations via row-blocked im2col +
/// the shared GEMM. `out` is [h*w, conv.co]; may not alias `in`.
pub fn conv2d(io: std.Io, gpa: std.mem.Allocator, out: []f32, in: []const f32, h: usize, w: usize, conv: Conv2d) !void {
    const ci = conv.ci;
    std.debug.assert(in.len == h * w * ci);
    std.debug.assert(out.len == h * w * conv.co);

    if (conv.k == 1) {
        return ops.matmul.matmul(io, gpa, out, in, h * w, Weight.fromF32(conv.w, conv.co, ci), conv.b);
    }
    std.debug.assert(conv.k == 3);

    const patch_len = 9 * ci;
    const weight = Weight.fromF32(conv.w, conv.co, patch_len);
    const max_rows = @max(1, (16 << 20) / (w * patch_len * 4));
    const buf = try gpa.alloc(f32, @min(h, max_rows) * w * patch_len);
    defer gpa.free(buf);

    var y0: usize = 0;
    while (y0 < h) : (y0 += max_rows) {
        const yn = @min(max_rows, h - y0);
        for (0..yn) |dy| {
            const y = y0 + dy;
            for (0..w) |x| {
                const patch = buf[(dy * w + x) * patch_len ..][0..patch_len];
                for (0..3) |ky| {
                    for (0..3) |kx| {
                        const dst = patch[(ky * 3 + kx) * ci ..][0..ci];
                        const sy = @as(isize, @intCast(y + ky)) - 1;
                        const sx = @as(isize, @intCast(x + kx)) - 1;
                        if (sy < 0 or sy >= h or sx < 0 or sx >= w) {
                            @memset(dst, 0);
                        } else {
                            const src_at = (@as(usize, @intCast(sy)) * w + @as(usize, @intCast(sx))) * ci;
                            @memcpy(dst, in[src_at..][0..ci]);
                        }
                    }
                }
            }
        }
        try ops.matmul.matmul(io, gpa, out[y0 * w * conv.co ..][0 .. yn * w * conv.co], buf[0 .. yn * w * patch_len], yn * w, weight, conv.b);
    }
}

/// Nearest-exact 2x spatial upsample of channel-last activations.
pub fn nearest2x(gpa: std.mem.Allocator, x: []const f32, h: usize, w: usize, c: usize) ![]f32 {
    const out = try gpa.alloc(f32, 4 * h * w * c);
    for (0..2 * h) |y| {
        for (0..2 * w) |xx| {
            const src = x[(y / 2 * w + xx / 2) * c ..][0..c];
            @memcpy(out[(y * 2 * w + xx) * c ..][0..c], src);
        }
    }
    return out;
}

pub fn planarToRows(gpa: std.mem.Allocator, planar: []const f32, c: usize, n: usize) ![]f32 {
    const out = try gpa.alloc(f32, planar.len);
    for (0..n) |i| {
        for (0..c) |ch| out[i * c + ch] = planar[ch * n + i];
    }
    return out;
}

pub fn rowsToPlanar(gpa: std.mem.Allocator, rows: []const f32, c: usize, n: usize) ![]f32 {
    const out = try gpa.alloc(f32, rows.len);
    for (0..n) |i| {
        for (0..c) |ch| out[ch * n + i] = rows[i * c + ch];
    }
    return out;
}

// --- weight loading --------------------------------------------------------

fn loadGamma(alloc: std.mem.Allocator, st: *const SafeTensors, name: []const u8, c: usize) ![]f32 {
    const view = st.get(name) orelse return error.MissingTensor;
    if (view.info.elemCount() != c) return error.ShapeMismatch;
    return view.toF32Alloc(alloc);
}

/// Load a conv weight (+bias), collapsing a causal temporal axis if present
/// (last kt slice) and repacking to [co][kh][kw][ci].
pub fn loadConv(alloc: std.mem.Allocator, st: *const SafeTensors, prefix: []const u8, ci: usize, co: usize, k: usize) !Conv2d {
    const wname = try std.fmt.allocPrint(alloc, "{s}.weight", .{prefix});
    const bname = try std.fmt.allocPrint(alloc, "{s}.bias", .{prefix});
    const wview = st.get(wname) orelse return error.MissingTensor;

    const shape = wview.info.shape.slice();
    const kt: usize = switch (shape.len) {
        5 => shape[2],
        4 => 1,
        else => return error.ShapeMismatch,
    };
    if (shape[0] != co or shape[1] != ci or shape[shape.len - 2] != k or shape[shape.len - 1] != k)
        return error.ShapeMismatch;

    const raw = try wview.toF32Alloc(alloc);
    defer alloc.free(raw);
    const packed_w = try alloc.alloc(f32, co * k * k * ci);
    for (0..co) |o| {
        for (0..ci) |i| {
            for (0..k) |ky| {
                for (0..k) |kx| {
                    // raw [co][ci][kt][k][k], temporal slice kt-1
                    const src = (((o * ci + i) * kt + (kt - 1)) * k + ky) * k + kx;
                    packed_w[((o * k + ky) * k + kx) * ci + i] = raw[src];
                }
            }
        }
    }

    // Bias may be absent (bias=False convs, e.g. TAEHV's TGrow/stage convs) —
    // synthesize zeros then.
    const bias = if (st.get(bname)) |bv| blk: {
        if (bv.info.elemCount() != co) return error.ShapeMismatch;
        break :blk try bv.toF32Alloc(alloc);
    } else blk: {
        const z = try alloc.alloc(f32, co);
        @memset(z, 0);
        break :blk z;
    };
    return .{ .w = packed_w, .b = bias, .co = co, .ci = ci, .k = k };
}

fn loadRes(alloc: std.mem.Allocator, st: *const SafeTensors, base: []const u8, idx: usize, ci: usize, co: usize) !ResBlock {
    const p = try std.fmt.allocPrint(alloc, "{s}.{d}", .{ base, idx });
    const n1 = try std.fmt.allocPrint(alloc, "{s}.residual.0.gamma", .{p});
    const c1 = try std.fmt.allocPrint(alloc, "{s}.residual.2", .{p});
    const n2 = try std.fmt.allocPrint(alloc, "{s}.residual.3.gamma", .{p});
    const c2 = try std.fmt.allocPrint(alloc, "{s}.residual.6", .{p});
    const sc = try std.fmt.allocPrint(alloc, "{s}.shortcut", .{p});
    return .{
        .norm1 = try loadGamma(alloc, st, n1, ci),
        .conv1 = try loadConv(alloc, st, c1, ci, co, 3),
        .norm2 = try loadGamma(alloc, st, n2, co),
        .conv2 = try loadConv(alloc, st, c2, co, co, 3),
        .shortcut = if (ci != co) try loadConv(alloc, st, sc, ci, co, 1) else null,
    };
}

// --- tests -----------------------------------------------------------------

fn readF32File(gpa: std.mem.Allocator, io: std.Io, path: []const u8, n: usize) ![]f32 {
    const out = try gpa.alloc(f32, n);
    errdefer gpa.free(out);
    const bytes = std.mem.sliceAsBytes(out);
    const file = try std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_only });
    defer file.close(io);
    const got = try file.readPositionalAll(io, bytes, 0);
    if (got != bytes.len) return error.ShortRead;
    return out;
}

// Parity against ComfyUI's WanVAE on the real checkpoint. Fixtures come from
// tools/dump_vae_fixture.py; skipped when the model or fixtures are absent.
test "decode matches comfyui reference" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const vae_path = "models/vae/krea2RealVae_v10.safetensors";
    std.Io.Dir.cwd().access(io, vae_path, .{}) catch return error.SkipZigTest;
    std.Io.Dir.cwd().access(io, "testdata/vae_z_8x8.bin", .{}) catch return error.SkipZigTest;

    const z = try readF32File(gpa, io, "testdata/vae_z_8x8.bin", 16 * 8 * 8);
    defer gpa.free(z);
    const expected = try readF32File(gpa, io, "testdata/vae_rgb_64.bin", 3 * 64 * 64);
    defer gpa.free(expected);

    var st = try SafeTensors.open(gpa, io, vae_path);
    defer st.deinit();
    var dec = try Decoder.load(gpa, &st);
    defer dec.deinit();

    const out = try dec.decode(io, gpa, z, 8, 8);
    defer gpa.free(out);

    // Keep the raw output around for offline analysis of parity failures.
    std.Io.Dir.cwd().writeFile(io, .{ .sub_path = "testdata/vae_out_zig.bin", .data = std.mem.sliceAsBytes(out) }) catch {};

    var max_err: f32 = 0;
    var sum_err: f64 = 0;
    for (expected, out) |e, a| {
        const err = @abs(e - a);
        max_err = @max(max_err, err);
        sum_err += err;
    }
    const mean_err = sum_err / @as(f64, @floatFromInt(out.len));
    std.debug.print("vae parity: max_err={d:.6} mean_err={d:.6}\n", .{ max_err, mean_err });
    try std.testing.expect(max_err < 5e-3);
    try std.testing.expect(mean_err < 5e-4);
}

fn stageErr(gpa: std.mem.Allocator, io: std.Io, name: []const u8, rows: []const f32, c: usize, n: usize) !f32 {
    var buf: [64]u8 = undefined;
    const path = try std.fmt.bufPrint(&buf, "testdata/vae_stage_{s}.bin", .{name});
    const ref = try readF32File(gpa, io, path, c * n);
    defer gpa.free(ref);
    var max_err: f32 = 0;
    for (0..n) |i| {
        for (0..c) |ch| max_err = @max(max_err, @abs(rows[i * c + ch] - ref[ch * n + i]));
    }
    return max_err;
}

// Stage-by-stage parity walk; only runs when the debug stage dumps from
// tools/dump_vae_stages.py are present.
test "decode stage parity" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    std.Io.Dir.cwd().access(io, "models/vae/krea2RealVae_v10.safetensors", .{}) catch return error.SkipZigTest;
    std.Io.Dir.cwd().access(io, "testdata/vae_stage_postquant.bin", .{}) catch return error.SkipZigTest;

    const z = try readF32File(gpa, io, "testdata/vae_z_8x8.bin", 16 * 8 * 8);
    defer gpa.free(z);
    var st = try SafeTensors.open(gpa, io, "models/vae/krea2RealVae_v10.safetensors");
    defer st.deinit();
    var dec = try Decoder.load(gpa, &st);
    defer dec.deinit();

    var x = try planarToRows(gpa, z, latent_channels, 64);
    var h: usize = 8;
    var w: usize = 8;

    x = try dec.applyConv(io, gpa, x, h, w, dec.post_quant);
    std.debug.print("postquant err {d:.7}\n", .{try stageErr(gpa, io, "postquant", x, 16, h * w)});
    x = try dec.applyConv(io, gpa, x, h, w, dec.conv_in);
    std.debug.print("convin err {d:.7}\n", .{try stageErr(gpa, io, "convin", x, 384, h * w)});
    x = try dec.applyRes(io, gpa, x, h, w, dec.mid_res1);
    std.debug.print("mid0 err {d:.7}\n", .{try stageErr(gpa, io, "mid0", x, 384, h * w)});
    x = try dec.applyAttn(io, gpa, x, h, w, dec.mid_attn);
    std.debug.print("mid1 err {d:.7}\n", .{try stageErr(gpa, io, "mid1", x, 384, h * w)});
    x = try dec.applyRes(io, gpa, x, h, w, dec.mid_res2);
    std.debug.print("mid2 err {d:.7}\n", .{try stageErr(gpa, io, "mid2", x, 384, h * w)});

    for (dec.ups, 0..) |layer, i| {
        var c: usize = undefined;
        switch (layer) {
            .res => |rb| {
                x = try dec.applyRes(io, gpa, x, h, w, rb);
                c = rb.conv2.co;
            },
            .up => |conv| {
                const up = try nearest2x(gpa, x, h, w, conv.ci);
                gpa.free(x);
                x = up;
                h *= 2;
                w *= 2;
                x = try dec.applyConv(io, gpa, x, h, w, conv);
                c = conv.co;
            },
        }
        var namebuf: [16]u8 = undefined;
        const name = try std.fmt.bufPrint(&namebuf, "up{d}", .{i});
        std.debug.print("{s} err {d:.7}\n", .{ name, try stageErr(gpa, io, name, x, c, h * w) });
    }
    gpa.free(x);
}

test "channel rms norm" {
    // One position, c=4: ||x|| = sqrt(30), scale sqrt(4) = 2.
    var x = [_]f32{ 1, 2, 3, 4 };
    const gamma = [_]f32{ 1, 1, 2, 0.5 };
    channelRmsNorm(&x, &gamma);
    const inv = 2.0 / @sqrt(@as(f32, 30.0));
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 * inv), x[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0 * inv * 2.0), x[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0 * inv * 0.5), x[3], 1e-6);
}

test "conv2d identity kernel" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    // 3x3 conv, 1 channel, kernel = delta at center: output == input.
    var w: [9]f32 = @splat(0);
    w[4] = 1; // [ky=1][kx=1][ci=0]
    const b = [_]f32{0};
    const conv: Conv2d = .{ .w = &w, .b = &b, .co = 1, .ci = 1, .k = 3 };
    const in = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    var out: [12]f32 = undefined;
    try conv2d(io, gpa, &out, &in, 3, 4, conv);
    try std.testing.expectEqualSlices(f32, &in, &out);
}

test "conv2d zero padding at borders" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    // Kernel = delta at top-left tap: out[y][x] = in[y-1][x-1] (0 outside).
    var w: [9]f32 = @splat(0);
    w[0] = 1;
    const b = [_]f32{0};
    const conv: Conv2d = .{ .w = &w, .b = &b, .co = 1, .ci = 1, .k = 3 };
    const in = [_]f32{ 1, 2, 3, 4 };
    var out: [4]f32 = undefined;
    try conv2d(io, gpa, &out, &in, 2, 2, conv);
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 0, 1 }, &out);
}

test "nearest2x" {
    const gpa = std.testing.allocator;
    const in = [_]f32{ 1, 2, 3, 4 }; // 2x2, c=1
    const out = try nearest2x(gpa, &in, 2, 2, 1);
    defer gpa.free(out);
    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4 }, out);
}
