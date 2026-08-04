//! Which generator standard-normal noise comes from — the one thing ComfyUI and
//! AUTOMATIC1111 disagree about that makes the same seed a *different picture* rather
//! than a slightly different one.
//!
//! Two engines, one selector: `torch_rng.zig` (torch's CPU MT19937 + AVX2 `normal_fill`,
//! what ComfyUI draws with) and `philox_rng.zig` (NVIDIA's Philox4x32-10, what A1111
//! draws with by default, since its `randn_source` defaults to `"GPU"`).
//!
//! ⚠️ **This has to apply to BOTH noise consumers, and missing the second one would be
//! invisible on the default sampler.** The initial latent is the obvious one. But the SDE
//! samplers' Brownian tree also draws a `torch.randn` per node, and the k-diffusion
//! commit A1111 pins (`ab527a9a`) builds `torchsde.BrownianTree` on `zeros_like(x)` with
//! **no `cpu` kwarg** — i.e. on the CUDA tensor's own device — where ComfyUI's fork forces
//! that tree to the CPU. So under A1111 every tree node is a Philox draw too. Euler would
//! have reproduced perfectly while every SDE render stayed wrong, which is the failure
//! shape this codebase keeps recording: fix one of two paths and the tests still pass.

const std = @import("std");
const torch_rng = @import("torch_rng.zig");
const philox_rng = @import("philox_rng.zig");

/// Named after the two ecosystems' settings rather than the algorithms, since that is how
/// a user encounters them: A1111 calls this option `randn_source` (`CPU` vs `GPU`/`NV`).
pub const Source = enum {
    /// torch's CPU generator — ComfyUI's `prepare_noise`, and A1111's `randn_source="CPU"`.
    torch_cpu,
    /// NVIDIA's Philox — A1111's default (`randn_source="GPU"`, and its `"NV"` CPU
    /// imitation of the same thing).
    nv_philox,
};

/// Fill `out` with seeded standard normals from `src`.
pub fn randn(out: []f32, seed: u64, src: Source) void {
    switch (src) {
        // ⚠️ Requires `out.len >= 16` — torch's `normal_fill` path redraws an
        // overlapping final block below that. Every latent and every Brownian node is
        // far larger; the philox arm has no such floor.
        .torch_cpu => torch_rng.randn(out, seed),
        .nv_philox => philox_rng.randn(out, seed),
    }
}

test "the two sources are unrelated, not perturbed" {
    // The point of the option: this is not a precision difference. If a future refactor
    // made one silently fall through to the other, correlation is what would catch it.
    var a: [4096]f32 = undefined;
    var b: [4096]f32 = undefined;
    randn(&a, 1234, .torch_cpu);
    randn(&b, 1234, .nv_philox);

    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (a, b) |x, y| {
        dot += @as(f64, x) * @as(f64, y);
        na += @as(f64, x) * @as(f64, x);
        nb += @as(f64, y) * @as(f64, y);
    }
    const cos = @abs(dot) / @sqrt(na * nb);
    errdefer std.debug.print("cosine {d}\n", .{cos});
    try std.testing.expect(cos < 0.05);

    // Both are still standard normal, which is the other half of "unrelated": a bug that
    // produced garbage would also decorrelate.
    for ([_]f64{ na, nb }) |ss| {
        const rms = @sqrt(ss / @as(f64, @floatFromInt(a.len)));
        errdefer std.debug.print("rms {d}\n", .{rms});
        try std.testing.expectApproxEqAbs(@as(f64, 1.0), rms, 0.05);
    }
}
