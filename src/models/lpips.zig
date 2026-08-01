//! LPIPS — Learned Perceptual Image Patch Similarity (Zhang et al. 2018), the
//! AlexNet variant, v0.1.
//!
//! LPIPS is not "a perceptual distance"; it is one specific computation, and
//! anything else cannot be compared against a published LPIPS number:
//!
//!   1. input in [-1, 1], then the ImageNet-ish affine `(x − shift) / scale`
//!   2. torchvision AlexNet `features`, tapped after each of the 5 ReLUs
//!   3. per-position unit-normalization *across channels* of each tap
//!   4. squared difference of the two normalized taps
//!   5. the learned per-channel weights `lin` (a 1x1 conv, no bias)
//!   6. spatial mean per layer, summed over the 5 layers
//!
//! Validated against the `lpips` pip package to 1e-5 on the fixtures in
//! `assets/lpips_fixtures.json`, with per-slice feature checksums so a mismatch
//! localizes to a conv instead of being "the number is wrong".
//!
//! Weights come from `tools/gen_lpips_fixtures.py` (torchvision AlexNet features
//! + the LPIPS `lin` weights, ~10 MB f32). They are a user-supplied checkpoint
//! like every other model here, not committed to the repo.
//!
//! AlexNet rather than VGG16 because the LPIPS authors recommend it as the
//! forward metric and it is ~25x cheaper; the interface has room for VGG16 if
//! comparability with a VGG-reporting paper is ever needed.

const std = @import("std");
const tp_core = @import("tp_core");
const ops = @import("tp_ops");

const safetensors = tp_core.safetensors;
const SafeTensors = safetensors.SafeTensors;
const conv = ops.conv;

/// Channel counts of the 5 taps.
pub const chns = [5]usize{ 64, 192, 384, 256, 256 };

/// torchvision `alexnet.features` indices of the 5 convs, which is how the
/// weight tensors are named in the checkpoint.
const feature_idx = [5]usize{ 0, 3, 6, 8, 10 };

/// Per-tap: the conv's geometry, and whether a 3x2 maxpool precedes it.
const layer_spec = [5]struct { k: usize, stride: usize, pad: usize, pool: bool }{
    .{ .k = 11, .stride = 4, .pad = 2, .pool = false },
    .{ .k = 5, .stride = 1, .pad = 2, .pool = true },
    .{ .k = 3, .stride = 1, .pad = 1, .pool = true },
    .{ .k = 3, .stride = 1, .pad = 1, .pool = false },
    .{ .k = 3, .stride = 1, .pad = 1, .pool = false },
};

const pool_k = 3;
const pool_stride = 2;

/// The eps inside `normalize_tensor`, which is added to the norm rather than to
/// its square — worth stating because the two differ at the 1e-5 the fixtures
/// check.
const norm_eps = 1e-10;

/// One tapped activation: `[h][w][c]`, channel-last.
pub const Tap = struct {
    data: []f32,
    h: usize,
    w: usize,
    c: usize,

    pub fn positions(self: Tap) usize {
        return self.h * self.w;
    }
};

/// Errors this module raises itself, on top of whatever the safetensors reader,
/// the allocator and the GEMM raise. `ImageTooSmall` means the tower would
/// produce a zero-extent tap — below ~32x32 for AlexNet.
pub const Error = error{ MissingTensor, ShapeMismatch, SizeMismatch, ImageTooSmall };

pub const Lpips = struct {
    arena: std.heap.ArenaAllocator,
    convs: [5]conv.Conv2d,
    /// Learned per-channel weights, `[chns[i]]`.
    lin: [5][]const f32,
    shift: [3]f32,
    scale: [3]f32,

    /// Load from an open safetensors container (see the module doc for the
    /// tensor names). Weights are copied into the returned arena, so `st` may be
    /// closed immediately after.
    pub fn load(gpa: std.mem.Allocator, st: *const SafeTensors) !Lpips {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();

        var convs: [5]conv.Conv2d = undefined;
        var lin: [5][]const f32 = undefined;
        for (0..5) |i| {
            const spec = layer_spec[i];
            const ci = if (i == 0) 3 else chns[i - 1];
            const co = chns[i];

            const wname = try std.fmt.allocPrint(a, "features.{d}.weight", .{feature_idx[i]});
            const bname = try std.fmt.allocPrint(a, "features.{d}.bias", .{feature_idx[i]});
            const wv = st.get(wname) orelse return error.MissingTensor;
            const shape = wv.info.shape.slice();
            if (shape.len != 4 or shape[0] != co or shape[1] != ci or shape[2] != spec.k or shape[3] != spec.k)
                return error.ShapeMismatch;
            const raw = try wv.toF32Alloc(a);
            const packed_w = try conv.packWeight(a, raw, co, ci, spec.k);
            a.free(raw);

            const bv = st.get(bname) orelse return error.MissingTensor;
            const bias = try bv.toF32Alloc(a);
            if (bias.len != co) return error.ShapeMismatch;

            convs[i] = .{
                .w = packed_w,
                .b = bias,
                .co = co,
                .ci = ci,
                .k = spec.k,
                .stride = spec.stride,
                .pad = spec.pad,
            };

            const lname = try std.fmt.allocPrint(a, "lin.{d}.weight", .{i});
            const lv = st.get(lname) orelse return error.MissingTensor;
            const lw = try lv.toF32Alloc(a);
            if (lw.len != co) return error.ShapeMismatch;
            lin[i] = lw;
        }

        var self: Lpips = .{
            .arena = arena,
            .convs = convs,
            .lin = lin,
            .shift = undefined,
            .scale = undefined,
        };
        for ([_][]const u8{ "scaling.shift", "scaling.scale" }, 0..) |name, which| {
            const v = st.get(name) orelse return error.MissingTensor;
            const f = try v.toF32Alloc(a);
            if (f.len != 3) return error.ShapeMismatch;
            const dst = if (which == 0) &self.shift else &self.scale;
            dst[0] = f[0];
            dst[1] = f[1];
            dst[2] = f[2];
        }
        return self;
    }

    /// Open `path` and load. Convenience for callers that have no other use for
    /// the container.
    pub fn loadPath(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !Lpips {
        var st = try SafeTensors.open(gpa, io, path);
        defer st.deinit();
        return load(gpa, &st);
    }

    pub fn deinit(self: *Lpips) void {
        self.arena.deinit();
    }

    /// The 5 tapped feature maps for one image, **before** normalization — i.e.
    /// exactly what conv+ReLU produced, which is what the per-layer fixture
    /// checksums pin. `pixels` is `[h][w][3]` u8 RGB. Caller frees each
    /// `Tap.data` with `gpa`.
    pub fn forward(
        self: *const Lpips,
        io: std.Io,
        gpa: std.mem.Allocator,
        pixels: []const u8,
        width: usize,
        height: usize,
    ) ![5]Tap {
        if (pixels.len != width * height * 3) return error.SizeMismatch;

        var taps: [5]Tap = undefined;
        var filled: usize = 0;
        errdefer for (taps[0..filled]) |t| gpa.free(t.data);

        // `scratch` is the one intermediate buffer this function owns: the
        // preprocessed input, or a pooled map. Once a tap is produced it owns
        // its buffer and also serves as the next layer's input, so scratch goes
        // back to null and nothing is copied.
        var scratch: ?[]f32 = null;
        errdefer if (scratch) |s| gpa.free(s);

        // [-1, 1] then the affine, channel-last so a conv patch is contiguous.
        const pre = try gpa.alloc(f32, pixels.len);
        scratch = pre;
        var i: usize = 0;
        while (i < pixels.len) : (i += 3) {
            inline for (0..3) |c| {
                const v = @as(f32, @floatFromInt(pixels[i + c])) / 255.0 * 2.0 - 1.0;
                pre[i + c] = (v - self.shift[c]) / self.scale[c];
            }
        }

        var input: []const f32 = pre;
        var ih = height;
        var iw = width;

        for (0..5) |layer| {
            const c = self.convs[layer];
            if (layer_spec[layer].pool) {
                const ph = conv.outDim(ih, pool_k, pool_stride, 0);
                const pw = conv.outDim(iw, pool_k, pool_stride, 0);
                if (ph == 0 or pw == 0) return error.ImageTooSmall;
                const pooled = try gpa.alloc(f32, ph * pw * c.ci);
                conv.maxPool2d(pooled, input, ih, iw, c.ci, pool_k, pool_stride);
                if (scratch) |s| gpa.free(s); // no-op when the input was a tap
                scratch = pooled;
                input = pooled;
                ih = ph;
                iw = pw;
            }

            const oh = c.outH(ih);
            const ow = c.outW(iw);
            if (oh == 0 or ow == 0) return error.ImageTooSmall;
            const out = try gpa.alloc(f32, oh * ow * c.co);
            {
                errdefer gpa.free(out);
                try conv.conv2d(io, gpa, out, input, ih, iw, c);
            }
            conv.relu(out);

            if (scratch) |s| {
                gpa.free(s);
                scratch = null;
            }
            taps[layer] = .{ .data = out, .h = oh, .w = ow, .c = c.co };
            filled = layer + 1;
            input = out;
            ih = oh;
            iw = ow;
        }
        return taps;
    }

    /// LPIPS distance between two equally sized `[h][w][3]` u8 RGB images, and
    /// the 5 per-layer contributions it is the sum of.
    ///
    /// Lower is more similar; 0 for identical input. Typical range 0–1, though
    /// nothing bounds it above.
    pub fn distancePerLayer(
        self: *const Lpips,
        io: std.Io,
        gpa: std.mem.Allocator,
        a: []const u8,
        b: []const u8,
        width: usize,
        height: usize,
        per_layer: *[5]f64,
    ) !f64 {
        if (a.len != b.len) return error.SizeMismatch;

        var ta = try self.forward(io, gpa, a, width, height);
        defer for (ta) |t| gpa.free(t.data);
        var tb = try self.forward(io, gpa, b, width, height);
        defer for (tb) |t| gpa.free(t.data);

        var total: f64 = 0;
        for (0..5) |i| {
            normalizeChannels(ta[i]);
            normalizeChannels(tb[i]);
            const n = ta[i].positions();
            const c = ta[i].c;
            const lin = self.lin[i];
            var acc: f64 = 0;
            for (0..n) |p| {
                const ra = ta[i].data[p * c ..][0..c];
                const rb = tb[i].data[p * c ..][0..c];
                var s: f64 = 0;
                for (ra, rb, lin) |x, y, wgt| {
                    const d = x - y;
                    s += @as(f64, wgt) * @as(f64, d) * @as(f64, d);
                }
                acc += s;
            }
            per_layer[i] = acc / @as(f64, @floatFromInt(n));
            total += per_layer[i];
        }
        return total;
    }

    /// `distancePerLayer` without the breakdown.
    pub fn distance(
        self: *const Lpips,
        io: std.Io,
        gpa: std.mem.Allocator,
        a: []const u8,
        b: []const u8,
        width: usize,
        height: usize,
    ) !f64 {
        var per_layer: [5]f64 = undefined;
        return self.distancePerLayer(io, gpa, a, b, width, height, &per_layer);
    }
};

/// In-place `x /= ‖x‖₂ + eps` per spatial position, across channels — the
/// reference's `normalize_tensor`.
fn normalizeChannels(t: Tap) void {
    var p: usize = 0;
    while (p < t.positions()) : (p += 1) {
        const row = t.data[p * t.c ..][0..t.c];
        var sq: f32 = 0;
        for (row) |v| sq += v * v;
        const inv = 1.0 / (@sqrt(sq) + norm_eps);
        for (row) |*v| v.* *= inv;
    }
}

// --- tests -----------------------------------------------------------------

const test_gate = @import("../test_gate.zig");
const image = tp_core.image;

const weights_path = "models/lpips/lpips_alex.safetensors";
const fixtures_json = @embedFile("assets/lpips_fixtures.json");

const FeatSum = struct { c: usize, h: usize, w: usize, sum: f64, sumsq: f64, max: f64 };
const Pair = struct {
    name: []const u8,
    width: usize,
    height: usize,
    png_a: []const u8,
    png_b: []const u8,
    lpips: f64,
    per_layer: [5]f64,
    feat_a: [5]FeatSum,
    feat_b: [5]FeatSum,
};
const Fixtures = struct { chns: [5]usize, pairs: []const Pair };

fn decodeB64Png(gpa: std.mem.Allocator, b64: []const u8) !image.DecodedPng {
    const dec = std.base64.standard.Decoder;
    const n = try dec.calcSizeForSlice(b64);
    const bytes = try gpa.alloc(u8, n);
    defer gpa.free(bytes);
    try dec.decode(bytes, b64);
    return image.decodePngRgb(gpa, bytes);
}

test "lpips matches the reference implementation on the fixture pairs" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    try test_gate.requireModelFile(io, weights_path);

    var parsed = try std.json.parseFromSlice(Fixtures, gpa, fixtures_json, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    try std.testing.expectEqualSlices(usize, &chns, &parsed.value.chns);

    var net = try Lpips.loadPath(gpa, io, weights_path);
    defer net.deinit();

    for (parsed.value.pairs) |p| {
        const img_a = try decodeB64Png(gpa, p.png_a);
        defer gpa.free(img_a.pixels);
        const img_b = try decodeB64Png(gpa, p.png_b);
        defer gpa.free(img_b.pixels);
        try std.testing.expectEqual(p.width, img_a.width);
        try std.testing.expectEqual(p.height, img_a.height);

        // Per-slice feature checksums first: they localize a failure to a conv,
        // which the distance alone cannot.
        for ([_]struct { px: []const u8, want: [5]FeatSum }{
            .{ .px = img_a.pixels, .want = p.feat_a },
            .{ .px = img_b.pixels, .want = p.feat_b },
        }) |side| {
            const taps = try net.forward(io, gpa, side.px, p.width, p.height);
            defer for (taps) |t| gpa.free(t.data);
            for (taps, side.want, 0..) |t, want, i| {
                try std.testing.expectEqual(want.c, t.c);
                try std.testing.expectEqual(want.h, t.h);
                try std.testing.expectEqual(want.w, t.w);
                var sum: f64 = 0;
                var sumsq: f64 = 0;
                var mx: f64 = 0;
                for (t.data) |v| {
                    sum += v;
                    sumsq += @as(f64, v) * @as(f64, v);
                    mx = @max(mx, v);
                }
                // Relative: these are sums over up to 10^5 f32 activations, so a
                // GEMM's reduction order alone moves the last few digits.
                inline for (.{ .{ want.sum, sum, "sum" }, .{ want.sumsq, sumsq, "sumsq" }, .{ want.max, mx, "max" } }) |cmp| {
                    const rel = @abs(cmp[0] - cmp[1]) / @max(1e-6, @abs(cmp[0]));
                    std.testing.expect(rel < 1e-4) catch |e| {
                        std.debug.print("{s} tap {d} {s}: torch {d} got {d} (rel {d})\n", .{ p.name, i, cmp[2], cmp[0], cmp[1], rel });
                        return e;
                    };
                }
            }
        }

        var per_layer: [5]f64 = undefined;
        const got = try net.distancePerLayer(io, gpa, img_a.pixels, img_b.pixels, p.width, p.height, &per_layer);
        for (per_layer, p.per_layer, 0..) |g, want, i| {
            std.testing.expectApproxEqAbs(want, g, 1e-5) catch |e| {
                std.debug.print("{s} layer {d}: torch {d} got {d}\n", .{ p.name, i, want, g });
                return e;
            };
        }
        std.testing.expectApproxEqAbs(p.lpips, got, 1e-5) catch |e| {
            std.debug.print("{s}: torch {d} got {d}\n", .{ p.name, p.lpips, got });
            return e;
        };
    }
}

test "lpips of an image against itself is exactly zero" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    try test_gate.requireModelFile(io, weights_path);

    var net = try Lpips.loadPath(gpa, io, weights_path);
    defer net.deinit();

    const w = 64;
    const h = 64;
    const px = try gpa.alloc(u8, w * h * 3);
    defer gpa.free(px);
    var prng = std.Random.DefaultPrng.init(19);
    prng.random().bytes(px);

    try std.testing.expectEqual(@as(f64, 0), try net.distance(io, gpa, px, px, w, h));
    // 16x16 is below the tower's floor rather than silently producing a number.
    try std.testing.expectError(error.ImageTooSmall, net.distance(io, gpa, px[0 .. 16 * 16 * 3], px[0 .. 16 * 16 * 3], 16, 16));
}
