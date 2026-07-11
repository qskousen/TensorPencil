//! TAEHV (Tiny AutoEncoder for Hunyuan/WanVideo) decoder — the "approx VAE"
//! ComfyUI uses for fast previews. This is the WAN 2.1 variant (taew2_1),
//! wired here as a CPU prototype to validate fidelity before a GPU port.
//!
//! TAEHV is a *temporal* video AE. For a single still image (T=1) we take the
//! first-frame path: MemBlock "past" is zeros, and each temporal-upscale TGrow
//! (a 1x1 conv that would split channels into 2 frames) keeps the first frame
//! (its first half of channels). All convs are 2D. Input is the raw sampler
//! latent (no scaling for taew2_1); output is [0,1] → RGB8. Architecture read
//! from ComfyUI comfy/taesd/taehv.py + the taew2_1 checkpoint keys.
const std = @import("std");
const safetensors = @import("../safetensors.zig");
const wan_vae = @import("wan_vae.zig");

const SafeTensors = safetensors.SafeTensors;
const Conv2d = wan_vae.Conv2d;

pub const latent_channels = 16;
pub const spatial_scale = 8;

pub const MemBlock = struct {
    // conv0 is (2n -> n) in the model, taking cat[x, past]. Since past is
    // always zero for a still image, the past-half weights vanish, so we keep
    // only the first-half input channels → a plain (n -> n) conv.
    conv0: Conv2d, // (n -> n) 3x3
    conv2: Conv2d, // (n -> n) 3x3
    conv4: Conv2d, // (n -> n) 3x3
    n: usize,
};

pub const Stage = struct {
    mb: [3]MemBlock,
    tgrow: Conv2d, // 1x1, co = n (first frame kept; second half of a stride-2 tgrow dropped)
    sc: Conv2d, // stage conv, 3x3, n -> n_next
    n: usize,
};

pub const Decoder = struct {
    arena: std.heap.ArenaAllocator,
    conv_in: Conv2d, // decoder.1: 16 -> 256, 3x3
    stages: [3]Stage,
    head_conv: Conv2d, // decoder.22: 64 -> 3, 3x3

    pub fn load(gpa: std.mem.Allocator, st: *const SafeTensors) !Decoder {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();

        const conv_in = try wan_vae.loadConv(a, st, "decoder.1", 16, 256, 3);
        // (n, first mem-block index, tgrow index, tgrow output ch, stage-conv
        // index, stage-conv out).
        const specs = [3]struct { n: usize, mb0: usize, tg: usize, tg_co: usize, sc: usize, sc_co: usize }{
            .{ .n = 256, .mb0 = 3, .tg = 7, .tg_co = 256, .sc = 8, .sc_co = 128 },
            .{ .n = 128, .mb0 = 9, .tg = 13, .tg_co = 256, .sc = 14, .sc_co = 64 },
            .{ .n = 64, .mb0 = 15, .tg = 19, .tg_co = 128, .sc = 20, .sc_co = 64 },
        };
        var stages: [3]Stage = undefined;
        for (&stages, specs) |*stage, s| {
            var mb: [3]MemBlock = undefined;
            for (&mb, 0..) |*b, j| {
                const base = s.mb0 + j;
                // conv0: load full (ci=2n) then keep the first-half input
                // channels (the x side of cat[x, past=0]).
                var c0 = try loadSub(a, st, base, "conv.0", s.n * 2, s.n, 3);
                c0 = try firstHalfInput(a, c0);
                b.* = .{
                    .conv0 = c0,
                    .conv2 = try loadSub(a, st, base, "conv.2", s.n, s.n, 3),
                    .conv4 = try loadSub(a, st, base, "conv.4", s.n, s.n, 3),
                    .n = s.n,
                };
            }
            // TGrow 1x1 conv (decoder.<tg>.conv). Load full output channels,
            // then keep only the first `n` (the first-frame slice).
            const tg_name = try std.fmt.allocPrint(a, "decoder.{d}.conv", .{s.tg});
            var tg = try wan_vae.loadConv(a, st, tg_name, s.n, s.tg_co, 1);
            tg.co = s.n; // first-frame: first n of the (n or 2n) output channels
            tg.b = tg.b[0..s.n]; // keep bias aligned with the sliced output
            const sc_name = try std.fmt.allocPrint(a, "decoder.{d}", .{s.sc});
            stage.* = .{
                .mb = mb,
                .tgrow = tg,
                .sc = try wan_vae.loadConv(a, st, sc_name, s.n, s.sc_co, 3),
                .n = s.n,
            };
        }

        const head_conv = try wan_vae.loadConv(a, st, "decoder.22", 64, 3, 3);
        return .{ .arena = arena, .conv_in = conv_in, .stages = stages, .head_conv = head_conv };
    }

    pub fn deinit(self: *Decoder) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// Decode planar [16][zh][zw] sampler latent to RGB8 [8*zh][8*zw][3].
    pub fn decode(self: *const Decoder, io: std.Io, gpa: std.mem.Allocator, z: []const f32, zh: usize, zw: usize) ![]u8 {
        std.debug.assert(z.len == latent_channels * zh * zw);
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const a = arena.allocator();

        // planar -> channel-last rows, with the input Clamp tanh(x/3)*3.
        var x = try wan_vae.planarToRows(a, z, latent_channels, zh * zw);
        for (x) |*v| v.* = std.math.tanh(v.* / 3.0) * 3.0;

        var h = zh;
        var w = zw;
        x = try conv(io, a, x, h, w, self.conv_in);
        for (x) |*v| v.* = @max(0.0, v.*); // ReLU after conv_in

        for (self.stages) |stage| {
            for (stage.mb) |b| x = try memBlock(io, a, x, h, w, b);
            x = try wan_vae.nearest2x(a, x, h, w, stage.n);
            h *= 2;
            w *= 2;
            x = try conv(io, a, x, h, w, stage.tgrow); // 1x1 (first-frame slice)
            x = try conv(io, a, x, h, w, stage.sc); // 3x3, no activation
        }

        for (x) |*v| v.* = @max(0.0, v.*); // ReLU before head
        x = try conv(io, a, x, h, w, self.head_conv); // -> [h*w][3]

        const rgb = try gpa.alloc(u8, h * w * 3);
        for (rgb, 0..) |*o, i| {
            o.* = @intFromFloat(std.math.clamp(x[i], 0.0, 1.0) * 255.0);
        }
        return rgb;
    }
};

/// MemBlock with zero "past": out = ReLU(conv4(ReLU(conv2(ReLU(conv0(x))))) + x).
/// conv0 already dropped its past-half input channels at load.
fn memBlock(io: std.Io, a: std.mem.Allocator, x: []const f32, h: usize, w: usize, b: MemBlock) ![]f32 {
    var t = try conv(io, a, x, h, w, b.conv0);
    for (t) |*v| v.* = @max(0.0, v.*);
    t = try conv(io, a, t, h, w, b.conv2);
    for (t) |*v| v.* = @max(0.0, v.*);
    t = try conv(io, a, t, h, w, b.conv4);
    for (t, 0..) |*v, i| v.* = @max(0.0, v.* + x[i]); // + skip(identity), fuse ReLU
    return t;
}

fn conv(io: std.Io, a: std.mem.Allocator, in: []const f32, h: usize, w: usize, cv: Conv2d) ![]f32 {
    const out = try a.alloc(f32, h * w * cv.co);
    try wan_vae.conv2d(io, a, out, in, h, w, cv);
    return out;
}

/// Keep only the first ci/2 input channels of a conv (w is [co][k][k][ci],
/// channel-innermost). Used for MemBlock conv0 whose past-half input is zero.
fn firstHalfInput(a: std.mem.Allocator, cv: Conv2d) !Conv2d {
    const ci2 = cv.ci;
    const ci = ci2 / 2;
    const w = try a.alloc(f32, cv.co * cv.k * cv.k * ci);
    for (0..cv.co * cv.k * cv.k) |blk| {
        @memcpy(w[blk * ci ..][0..ci], cv.w[blk * ci2 ..][0..ci]);
    }
    return .{ .w = w, .b = cv.b, .co = cv.co, .ci = ci, .k = cv.k };
}

fn loadSub(a: std.mem.Allocator, st: *const SafeTensors, base: usize, sub: []const u8, ci: usize, co: usize, k: usize) !Conv2d {
    const name = try std.fmt.allocPrint(a, "decoder.{d}.{s}", .{ base, sub });
    return wan_vae.loadConv(a, st, name, ci, co, k);
}
