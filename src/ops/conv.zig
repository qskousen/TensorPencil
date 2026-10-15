//! General 2D convolution and max-pooling over channel-last activations,
//! expressed as banded `im2col` + the shared GEMM.
//!
//! `wan_vae.conv2d` predates this and covers only the VAE's case (k ∈ {1,3},
//! stride 1, same-padding); its bytes are pinned by the VAE parity fixtures, so
//! it stays as it is. This is the general form, arbitrary kernel, stride and
//! zero-padding, PyTorch's floor semantics for the output size, needed by
//! feature towers like LPIPS-AlexNet (k11 s4 p2, k5 s1 p2, k3 s1 p1, maxpool
//! k3 s2). A test pins it bit-for-bit against `wan_vae.conv2d` on the shape
//! they share, so the two cannot drift.
//!
//! Layout, everywhere here: activations are `[h][w][c]` (channel-last, the
//! layout that makes an im2col patch a contiguous GEMM row), and conv weights
//! are `[co][kh][kw][ci]` to match: pre-packed (`packWeight` converts from
//! PyTorch's `[co][ci][kh][kw]`), or kept in the checkpoint's own order and dtype
//! and converted by whoever reads them (`Weights.torch`).

const std = @import("std");
const matmul = @import("matmul.zig");

const Weight = matmul.Weight;

/// Default cap on the im2col patch band, in bytes. Bands iterate over output
/// rows; `conv2dBanded` takes it explicitly so a test can force a split.
pub const default_band_bytes = 16 << 20;

/// A convolution's weights.
pub const Weights = union(enum) {
    /// f32 `[co][kh][kw][ci]`, the im2col patch order. See `packWeight`.
    krsc: []const f32,
    /// The checkpoint's own tensor in its stored dtype. A block quant here may tile the
    /// FLAT sequence (`flat_blocks`), so its rows are not block-aligned and only a
    /// whole-tensor decode reads it.
    torch: Torch,

    /// `[co][ci][kt][kh][kw]`, `co` rows of `ci*kt*kh*kw`, of which only temporal slice
    /// `kt - 1` is read: a causal video kernel run on one frame. `kt = 1` is a plain
    /// `[co][ci][kh][kw]` conv.
    pub const Torch = struct { w: Weight, kt: usize = 1 };
};

/// How a device conv's gather samples its source; one enum for every backend, since
/// it is the im2col kernels' `f0` encoding.
pub const SampleMode = enum(u2) { stride1 = 0, upsample2x = 1, stride2 = 2 };

/// A 1-D convolution as the GEMM its device gather feeds: `w` is `[out_ch][k * in_ch]`,
/// taps outer, the column order `im2col1d` writes.
pub const Conv1dGemm = struct {
    w: Weight,
    bias: ?[]const f32,
    in_ch: usize,
    k: usize,
    dilation: usize = 1,
    padding: usize = 0,
    stride: usize = 1,

    pub fn outCh(c: Conv1dGemm) usize {
        return c.w.rows;
    }

    pub fn outLen(c: Conv1dGemm, in_len: usize) usize {
        const eff = c.dilation * (c.k - 1) + 1;
        const padded = in_len + 2 * c.padding;
        if (padded < eff) return 0;
        return (padded - eff) / c.stride + 1;
    }
};

pub const Conv2d = struct {
    w: Weights,
    /// `[co]`, or null for a bias-free conv.
    b: ?[]const f32 = null,
    co: usize,
    ci: usize,
    k: usize,
    stride: usize = 1,
    pad: usize = 0,
    /// Checkpoint tensor name, when the loader knows it, carried onto the `Weight`
    /// this convolution's im2col GEMM is issued with.
    ///
    /// Without it a convolutional model is invisible to `ops.matmul.probe`. A
    /// conv here *is* a GEMM, so the probe fires either way, but an untagged call
    /// cannot be attributed to a layer, so an activation capture of a UNet would
    /// record only its attention and feed-forward linears and silently omit the
    /// convolutions holding most of its parameters. That is the same class of partial
    /// coverage as the Vulkan capture that recorded 39 of 263 layers and looked
    /// well-formed.
    tag: ?[]const u8 = null,

    pub fn patchLen(self: Conv2d) usize {
        return self.k * self.k * self.ci;
    }

    /// The repacked weights, for a reader that takes only those. A checkpoint-order
    /// conv reaching one is a missing repack, and must not be a silent misread in
    /// ReleaseFast, where an inactive union field is not checked.
    pub fn krsc(self: Conv2d) []const f32 {
        return switch (self.w) {
            .krsc => |s| s,
            .torch => @panic("ops.conv: a checkpoint-order conv reached a reader of repacked f32; repack it first"),
        };
    }

    /// The `Weight` this convolution's GEMM is: `co` rows over the im2col patch, tagged.
    /// A LoRA target is keyed and width-checked against it, and a probe files under it.
    /// For a `torch` conv it carries no bytes, so no GEMM can be handed it by mistake.
    pub fn gemmWeight(self: Conv2d) Weight {
        var w = switch (self.w) {
            .krsc => |s| Weight.fromF32(s, self.co, self.patchLen()),
            .torch => |t| Weight{ .bytes = &.{}, .dtype = t.w.dtype, .rows = self.co, .cols = self.patchLen() },
        };
        w.tag = self.tag;
        return w;
    }

    /// f32 `[co][kh][kw][ci]` in `gpa`, decoded and repacked from a `torch` conv or
    /// copied from a `krsc` one. Caller owns it.
    pub fn materialize(self: Conv2d, gpa: std.mem.Allocator) ![]f32 {
        switch (self.w) {
            .krsc => |s| return gpa.dupe(f32, s),
            .torch => |t| {
                const kk = self.k * self.k;
                const flat = try gpa.alloc(f32, self.co * self.ci * t.kt * kk);
                defer gpa.free(flat);
                try @import("tp_core").safetensors.convertToF32(t.w.dtype, t.w.bytes, flat);
                const out = try gpa.alloc(f32, self.co * self.patchLen());
                for (0..self.co) |o| for (0..self.ci) |i| for (0..kk) |tap| {
                    out[(o * kk + tap) * self.ci + i] = flat[((o * self.ci + i) * t.kt + (t.kt - 1)) * kk + tap];
                };
                return out;
            },
        }
    }

    pub fn outH(self: Conv2d, h: usize) usize {
        return outDim(h, self.k, self.stride, self.pad);
    }

    pub fn outW(self: Conv2d, w: usize) usize {
        return outDim(w, self.k, self.stride, self.pad);
    }
};

/// PyTorch's output extent: `floor((n + 2*pad − k) / stride) + 1`. Zero when the
/// padded input is smaller than the kernel.
pub fn outDim(n: usize, k: usize, stride: usize, pad: usize) usize {
    const padded = n + 2 * pad;
    if (padded < k) return 0;
    return (padded - k) / stride + 1;
}

/// Repack a PyTorch conv weight `[co][ci][kh][kw]` into the `[co][kh][kw][ci]`
/// im2col patch order `Conv2d` expects. Caller owns the result.
pub fn packWeight(gpa: std.mem.Allocator, torch_w: []const f32, co: usize, ci: usize, k: usize) ![]f32 {
    std.debug.assert(torch_w.len == co * ci * k * k);
    const out = try gpa.alloc(f32, torch_w.len);
    for (0..co) |o| {
        for (0..ci) |i| {
            for (0..k) |ky| {
                for (0..k) |kx| {
                    out[((o * k + ky) * k + kx) * ci + i] = torch_w[((o * ci + i) * k + ky) * k + kx];
                }
            }
        }
    }
    return out;
}

/// `out[oy][ox][co] = Σ in[oy*s−p+ky][ox*s−p+kx][ci] * w[co][ky][kx][ci] + b[co]`,
/// zero outside the input. `out` must be `outH(h)*outW(w)*co` long and may not
/// alias `in`.
pub fn conv2d(
    io: std.Io,
    gpa: std.mem.Allocator,
    out: []f32,
    in: []const f32,
    h: usize,
    w: usize,
    conv: Conv2d,
) !void {
    return conv2dBanded(io, gpa, out, in, h, w, conv, default_band_bytes);
}

/// `conv2d` with an explicit patch-band cap. The band size must not change the
/// result, only the peak memory, which is what the banding test asserts.
pub fn conv2dBanded(
    io: std.Io,
    gpa: std.mem.Allocator,
    out: []f32,
    in: []const f32,
    h: usize,
    w: usize,
    conv: Conv2d,
    band_bytes: usize,
) !void {
    const oh = conv.outH(h);
    const ow = conv.outW(w);
    std.debug.assert(in.len == h * w * conv.ci);
    std.debug.assert(out.len == oh * ow * conv.co);
    if (oh == 0 or ow == 0) return;

    // A checkpoint-order weight is decoded and repacked per call: this is the
    // reference path, and keeping it packed is what keeps a quantized file small.
    var owned: ?[]f32 = null;
    defer if (owned) |o| gpa.free(o);
    const wk: []const f32 = switch (conv.w) {
        .krsc => |s| s,
        .torch => blk: {
            owned = try conv.materialize(gpa);
            break :blk owned.?;
        },
    };

    // 1x1 stride-1 unpadded is a plain GEMM over positions, no patch buffer.
    if (conv.k == 1 and conv.stride == 1 and conv.pad == 0) {
        var w1: Weight = Weight.fromF32(wk, conv.co, conv.ci);
        w1.tag = conv.tag;
        return matmul.matmul(io, gpa, out, in, h * w, w1, conv.b);
    }

    const patch_len = conv.patchLen();
    var weight: Weight = Weight.fromF32(wk, conv.co, patch_len);
    weight.tag = conv.tag;
    const max_rows = @max(1, band_bytes / (ow * patch_len * 4));
    const buf = try gpa.alloc(f32, @min(oh, max_rows) * ow * patch_len);
    defer gpa.free(buf);

    var oy0: usize = 0;
    while (oy0 < oh) : (oy0 += max_rows) {
        const rows = @min(max_rows, oh - oy0);
        for (0..rows) |dy| {
            const oy = oy0 + dy;
            const sy0 = @as(isize, @intCast(oy * conv.stride)) - @as(isize, @intCast(conv.pad));
            for (0..ow) |ox| {
                const sx0 = @as(isize, @intCast(ox * conv.stride)) - @as(isize, @intCast(conv.pad));
                const patch = buf[(dy * ow + ox) * patch_len ..][0..patch_len];
                for (0..conv.k) |ky| {
                    const sy = sy0 + @as(isize, @intCast(ky));
                    const row_oob = sy < 0 or sy >= h;
                    for (0..conv.k) |kx| {
                        const dst = patch[(ky * conv.k + kx) * conv.ci ..][0..conv.ci];
                        const sx = sx0 + @as(isize, @intCast(kx));
                        if (row_oob or sx < 0 or sx >= w) {
                            @memset(dst, 0);
                        } else {
                            const at = (@as(usize, @intCast(sy)) * w + @as(usize, @intCast(sx))) * conv.ci;
                            @memcpy(dst, in[at..][0..conv.ci]);
                        }
                    }
                }
            }
        }
        try matmul.matmul(
            io,
            gpa,
            out[oy0 * ow * conv.co ..][0 .. rows * ow * conv.co],
            buf[0 .. rows * ow * patch_len],
            rows * ow,
            weight,
            conv.b,
        );
    }
}

/// PyTorch `max_pool2d(kernel_size=k, stride=stride)` over channel-last
/// activations: no padding, no dilation, `ceil_mode=false`. `out` must be
/// `outDim(h...)*outDim(w...)*c` long and may not alias `in`.
pub fn maxPool2d(out: []f32, in: []const f32, h: usize, w: usize, c: usize, k: usize, stride: usize) void {
    const oh = outDim(h, k, stride, 0);
    const ow = outDim(w, k, stride, 0);
    std.debug.assert(in.len == h * w * c);
    std.debug.assert(out.len == oh * ow * c);

    for (0..oh) |oy| {
        for (0..ow) |ox| {
            const dst = out[(oy * ow + ox) * c ..][0..c];
            const y0 = oy * stride;
            const x0 = ox * stride;
            @memcpy(dst, in[((y0 * w) + x0) * c ..][0..c]);
            for (0..k) |ky| {
                for (0..k) |kx| {
                    if (ky == 0 and kx == 0) continue;
                    const src = in[(((y0 + ky) * w) + x0 + kx) * c ..][0..c];
                    for (dst, src) |*d, s| d.* = @max(d.*, s);
                }
            }
        }
    }
}

/// Depthwise convolution (`groups == channels`), stride 1, zero-padded:
/// `out[y][x][c] = Σ in[y+ky-pad][x+kx-pad][c] * w[c][ky][kx] + b[c]`.
///
/// Not an im2col GEMM: each output channel reads only its own input channel, so
/// the patch matrix a GEMM needs would be block-diagonal and almost all zero.
/// Written as a direct gather instead, which keeps the innermost loop over
/// CHANNELS and therefore contiguous in this file's channel-last layout.
///
/// The weight stays in PyTorch's `[c][1][kh][kw]` order (that is already
/// channel-outermost), so there is no `packWeight` step for it.
pub fn depthwiseConv2d(
    out: []f32,
    in: []const f32,
    h: usize,
    w: usize,
    c: usize,
    k: usize,
    pad: usize,
    weight: []const f32,
    bias: ?[]const f32,
) void {
    std.debug.assert(in.len == h * w * c);
    std.debug.assert(weight.len == c * k * k);
    const oh = outDim(h, k, 1, pad);
    const ow = outDim(w, k, 1, pad);
    std.debug.assert(out.len == oh * ow * c);

    for (0..oh) |oy| {
        for (0..ow) |ox| {
            const dst = out[(oy * ow + ox) * c ..][0..c];
            if (bias) |b| @memcpy(dst, b) else @memset(dst, 0);
            for (0..k) |ky| {
                const iy = oy + ky;
                if (iy < pad or iy - pad >= h) continue;
                for (0..k) |kx| {
                    const ix = ox + kx;
                    if (ix < pad or ix - pad >= w) continue;
                    const src = in[(((iy - pad) * w) + ix - pad) * c ..][0..c];
                    for (dst, src, 0..) |*d, s, ci| d.* += s * weight[(ci * k + ky) * k + kx];
                }
            }
        }
    }
}

/// In-place ReLU.
pub fn relu(x: []f32) void {
    for (x) |*v| v.* = @max(v.*, 0);
}


// --- tests -----------------------------------------------------------------

const fixtures_json = @embedFile("assets/conv_fixtures.json");

const ConvCase = struct {
    name: []const u8,
    h: usize,
    w: usize,
    ci: usize,
    co: usize,
    k: usize,
    stride: usize,
    pad: usize,
    out_h: usize,
    out_w: usize,
    /// `[h][w][ci]`, channel-last.
    x: []const f32,
    /// `[co][ci][k][k]`, PyTorch order, the test packs it, so `packWeight` is
    /// under test too.
    weight: []const f32,
    bias: []const f32,
    /// `[out_h][out_w][co]`, channel-last.
    out: []const f32,
};

const PoolCase = struct {
    name: []const u8,
    h: usize,
    w: usize,
    c: usize,
    k: usize,
    stride: usize,
    out_h: usize,
    out_w: usize,
    x: []const f32,
    out: []const f32,
};

const Fixtures = struct { conv_cases: []const ConvCase, pool_cases: []const PoolCase };

fn loadFixtures(gpa: std.mem.Allocator) !std.json.Parsed(Fixtures) {
    return std.json.parseFromSlice(Fixtures, gpa, fixtures_json, .{ .ignore_unknown_fields = true });
}

test "conv2d matches torch conv2d on the LPIPS-AlexNet shapes" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var parsed = try loadFixtures(gpa);
    defer parsed.deinit();

    for (parsed.value.conv_cases) |c| {
        const packed_w = try packWeight(gpa, c.weight, c.co, c.ci, c.k);
        defer gpa.free(packed_w);
        const conv: Conv2d = .{
            .w = .{ .krsc = packed_w },
            .b = c.bias,
            .co = c.co,
            .ci = c.ci,
            .k = c.k,
            .stride = c.stride,
            .pad = c.pad,
        };
        try std.testing.expectEqual(c.out_h, conv.outH(c.h));
        try std.testing.expectEqual(c.out_w, conv.outW(c.w));

        const out = try gpa.alloc(f32, c.out_h * c.out_w * c.co);
        defer gpa.free(out);
        try conv2d(io, gpa, out, c.x, c.h, c.w, conv);

        for (out, c.out, 0..) |got, want, i| {
            std.testing.expectApproxEqAbs(want, got, 2e-5) catch |e| {
                std.debug.print("conv case {s}: element {d}: got {d} want {d}\n", .{ c.name, i, got, want });
                return e;
            };
        }
    }
}

test "maxPool2d matches torch max_pool2d" {
    const gpa = std.testing.allocator;
    var parsed = try loadFixtures(gpa);
    defer parsed.deinit();

    for (parsed.value.pool_cases) |c| {
        try std.testing.expectEqual(c.out_h, outDim(c.h, c.k, c.stride, 0));
        try std.testing.expectEqual(c.out_w, outDim(c.w, c.k, c.stride, 0));
        const out = try gpa.alloc(f32, c.out_h * c.out_w * c.c);
        defer gpa.free(out);
        maxPool2d(out, c.x, c.h, c.w, c.c, c.k, c.stride);
        for (out, c.out, 0..) |got, want, i| {
            std.testing.expectApproxEqAbs(want, got, 1e-6) catch |e| {
                std.debug.print("pool case {s}: element {d}: got {d} want {d}\n", .{ c.name, i, got, want });
                return e;
            };
        }
    }
}

test "the patch band cap changes the result only in the last bits" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const h = 33;
    const w = 21;
    const ci = 3;
    const co = 4;
    const k = 11;

    const in = try gpa.alloc(f32, h * w * ci);
    defer gpa.free(in);
    const wt = try gpa.alloc(f32, co * k * k * ci);
    defer gpa.free(wt);
    var rng = std.Random.DefaultPrng.init(7);
    const r = rng.random();
    for (in) |*v| v.* = r.float(f32) * 2 - 1;
    for (wt) |*v| v.* = r.float(f32) * 2 - 1;

    const conv: Conv2d = .{ .w = .{ .krsc = wt }, .co = co, .ci = ci, .k = k, .stride = 4, .pad = 2 };
    const oh = conv.outH(h);
    const ow = conv.outW(w);
    try std.testing.expect(oh > 4); // enough output rows for the cap to split

    const whole = try gpa.alloc(f32, oh * ow * co);
    defer gpa.free(whole);
    try conv2dBanded(io, gpa, whole, in, h, w, conv, default_band_bytes);

    // One output row per band, and then a cap below one row (clamped to one).
    // Neither is *bit*-identical to the unbanded call: a band is one GEMM, and
    // the CPU GEMM's reduction order depends on its row count `m` (see
    // matmul.small_m_max), so the last bits move. Measured here: max abs 1.1e-5
    // on values of magnitude ~7, i.e. ~1.6e-6 relative. That is why the band cap
    // is a fixed constant rather than a tuning knob - a caller comparing two
    // images must convolve them at the same cap, which `conv2d` guarantees.
    const one_row = ow * conv.patchLen() * 4;
    const banded = try gpa.alloc(f32, oh * ow * co);
    defer gpa.free(banded);
    for ([_]usize{ one_row, 1 }) |cap| {
        try conv2dBanded(io, gpa, banded, in, h, w, conv, cap);
        var maxd: f64 = 0;
        for (whole, banded) |x, y| maxd = @max(maxd, @abs(@as(f64, x) - @as(f64, y)));
        std.testing.expect(maxd < 5e-5) catch |e| {
            std.debug.print("band cap {d}: max abs deviation {d}\n", .{ cap, maxd });
            return e;
        };
    }
}

test "depthwise conv equals a dense conv with a block-diagonal weight" {
    // Cross-check rather than a fixture: a depthwise conv IS the dense conv whose
    // off-diagonal input channels are zero, so `conv2d` (already pinned against
    // torch) is the reference. Anything the gather gets wrong -- a transposed
    // weight, a padding sign, the wrong channel stride -- differs from it.
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const h = 5;
    const w = 7;
    const c = 6;
    const k = 3;
    const pad = 1;

    var prng = std.Random.DefaultPrng.init(0xd0f9);
    const rand = prng.random();
    const in = try gpa.alloc(f32, h * w * c);
    defer gpa.free(in);
    for (in) |*v| v.* = rand.float(f32) * 2 - 1;
    const dw = try gpa.alloc(f32, c * k * k);
    defer gpa.free(dw);
    for (dw) |*v| v.* = rand.float(f32) * 2 - 1;
    const bias = try gpa.alloc(f32, c);
    defer gpa.free(bias);
    for (bias) |*v| v.* = rand.float(f32);

    // [co][ci][kh][kw] with ci != co zeroed, then packed like any other conv.
    const dense = try gpa.alloc(f32, c * c * k * k);
    defer gpa.free(dense);
    @memset(dense, 0);
    for (0..c) |ci| {
        for (0..k * k) |kk| dense[(ci * c + ci) * k * k + kk] = dw[ci * k * k + kk];
    }
    const packed_w = try packWeight(gpa, dense, c, c, k);
    defer gpa.free(packed_w);

    const want = try gpa.alloc(f32, h * w * c);
    defer gpa.free(want);
    try conv2d(io, gpa, want, in, h, w, .{ .w = .{ .krsc = packed_w }, .b = bias, .co = c, .ci = c, .k = k, .pad = pad });

    const got = try gpa.alloc(f32, h * w * c);
    defer gpa.free(got);
    depthwiseConv2d(got, in, h, w, c, k, pad, dw, bias);

    var maxd: f64 = 0;
    for (want, got) |a, b| maxd = @max(maxd, @abs(@as(f64, a) - b));
    errdefer std.debug.print("depthwise vs dense: max abs {d}\n", .{maxd});
    try std.testing.expect(maxd < 1e-6);
}

test "a checkpoint-order conv computes exactly what its repacked twin does" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const h = 9;
    const w = 7;
    const ci = 5;
    const co = 6;
    const k = 3;
    var rng = std.Random.DefaultPrng.init(11);
    const r = rng.random();
    const in = try gpa.alloc(f32, h * w * ci);
    defer gpa.free(in);
    for (in) |*v| v.* = r.float(f32) * 2 - 1;
    // f16-representable, so the f16 copy below holds the same values.
    const torch_w = try gpa.alloc(f32, co * ci * k * k);
    defer gpa.free(torch_w);
    for (torch_w) |*v| v.* = @as(f16, @floatCast(r.float(f32) * 2 - 1));
    const half = try gpa.alloc(f16, torch_w.len);
    defer gpa.free(half);
    for (half, torch_w) |*d, v| d.* = @floatCast(v);
    const packed_w = try packWeight(gpa, torch_w, co, ci, k);
    defer gpa.free(packed_w);

    const base: Conv2d = .{ .w = .{ .krsc = packed_w }, .co = co, .ci = ci, .k = k, .stride = 2, .pad = 1 };
    const n = base.outH(h) * base.outW(w) * co;
    const want = try gpa.alloc(f32, n);
    defer gpa.free(want);
    try conv2d(io, gpa, want, in, h, w, base);

    const got = try gpa.alloc(f32, n);
    defer gpa.free(got);
    for ([_]Weight{
        Weight.init(std.mem.sliceAsBytes(torch_w), .f32, co, ci * k * k),
        Weight.init(std.mem.sliceAsBytes(half), .f16, co, ci * k * k),
    }) |tw| {
        var c = base;
        c.w = .{ .torch = .{ .w = tw } };
        try conv2d(io, gpa, got, in, h, w, c);
        errdefer std.debug.print("{t} torch-order conv differs from the repacked one\n", .{tw.dtype});
        try std.testing.expectEqualSlices(f32, want, got);
    }
}
