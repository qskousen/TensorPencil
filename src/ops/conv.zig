//! General 2D convolution and max-pooling over **channel-last** activations,
//! expressed as banded `im2col` + the shared GEMM.
//!
//! `wan_vae.conv2d` predates this and covers only the VAE's case (k ∈ {1,3},
//! stride 1, same-padding); its bytes are pinned by the VAE parity fixtures, so
//! it stays as it is. This is the general form — arbitrary kernel, stride and
//! zero-padding, PyTorch's floor semantics for the output size — needed by
//! feature towers like LPIPS-AlexNet (k11 s4 p2, k5 s1 p2, k3 s1 p1, maxpool
//! k3 s2). A test pins it bit-for-bit against `wan_vae.conv2d` on the shape
//! they share, so the two cannot drift.
//!
//! Layout, everywhere here: activations are `[h][w][c]` (channel-last, the
//! layout that makes an im2col patch a contiguous GEMM row), and conv weights
//! are pre-packed `[co][kh][kw][ci]` to match. `packWeight` converts from
//! PyTorch's `[co][ci][kh][kw]`.

const std = @import("std");
const matmul = @import("matmul.zig");

const Weight = matmul.Weight;

/// Default cap on the im2col patch band, in bytes. Bands iterate over output
/// rows; `conv2dBanded` takes it explicitly so a test can force a split.
pub const default_band_bytes = 16 << 20;

pub const Conv2d = struct {
    /// `[co][kh][kw][ci]`, i.e. im2col patch order. See `packWeight`.
    w: []const f32,
    /// `[co]`, or null for a bias-free conv.
    b: ?[]const f32 = null,
    co: usize,
    ci: usize,
    k: usize,
    stride: usize = 1,
    pad: usize = 0,
    /// Checkpoint tensor name, when the loader knows it — carried onto the `Weight`
    /// this convolution's im2col GEMM is issued with.
    ///
    /// ⚠️ **Without it a convolutional model is invisible to `ops.matmul.probe`.** A
    /// conv here *is* a GEMM, so the probe fires either way, but an untagged call
    /// cannot be attributed to a layer — so an activation capture of a UNet would
    /// record only its attention and feed-forward linears and silently omit the
    /// convolutions holding most of its parameters. That is the same class of partial
    /// coverage as the Vulkan capture that recorded 39 of 263 layers and looked
    /// well-formed.
    tag: ?[]const u8 = null,

    pub fn patchLen(self: Conv2d) usize {
        return self.k * self.k * self.ci;
    }

    pub fn outH(self: Conv2d, h: usize) usize {
        return outDim(h, self.k, self.stride, self.pad);
    }

    pub fn outW(self: Conv2d, w: usize) usize {
        return outDim(w, self.k, self.stride, self.pad);
    }
};

/// PyTorch's output extent: `floor((n + 2·pad − k) / stride) + 1`. Zero when the
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

/// `out[oy][ox][co] = Σ in[oy·s−p+ky][ox·s−p+kx][ci] · w[co][ky][kx][ci] + b[co]`,
/// zero outside the input. `out` must be `outH(h)·outW(w)·co` long and may not
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
/// result — only the peak memory — which is what the banding test asserts.
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

    // 1x1 stride-1 unpadded is a plain GEMM over positions — no patch buffer.
    if (conv.k == 1 and conv.stride == 1 and conv.pad == 0) {
        var w1: Weight = Weight.fromF32(conv.w, conv.co, conv.ci);
        w1.tag = conv.tag;
        return matmul.matmul(io, gpa, out, in, h * w, w1, conv.b);
    }

    const patch_len = conv.patchLen();
    var weight: Weight = Weight.fromF32(conv.w, conv.co, patch_len);
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
/// `outDim(h,…)·outDim(w,…)·c` long and may not alias `in`.
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
    /// `[co][ci][k][k]`, PyTorch order — the test packs it, so `packWeight` is
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
            .w = packed_w,
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

    const conv: Conv2d = .{ .w = wt, .co = co, .ci = ci, .k = k, .stride = 4, .pad = 2 };
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
