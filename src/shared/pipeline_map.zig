//! The settings enums against the pipeline's. They are separate enums because
//! the config's carry UI labels and a stable serialized spelling, and each map
//! is written out rather than derived from tag names so a member with no twin
//! is a compile error here, not a silently wrong row.
const std = @import("std");
const tp = @import("TensorPencil");
const pipeline = tp.pipeline;
const config = @import("config.zig");

/// `auto` resolves to this machine's devices, so call it where the engine runs.
pub fn toPipelineBackend(b: config.Backend) pipeline.Backend {
    return switch (b) {
        .cpu => .cpu,
        .vulkan => .vulkan,
        .zig_cuda => .zig_cuda,
        .cuda => .cuda,
        .auto => pipeline.detectBackend(),
    };
}

/// The backend `b` names, or null for `auto`, which a client cannot resolve for a
/// host it does not run on.
pub fn explicitBackend(b: config.Backend) ?pipeline.Backend {
    return if (b == .auto) null else toPipelineBackend(b);
}

pub fn fromPipelineBackend(b: pipeline.Backend) config.Backend {
    return switch (b) {
        .cpu => .cpu,
        .vulkan => .vulkan,
        .zig_cuda => .zig_cuda,
        .cuda => .cuda,
    };
}

pub fn toPipelineVae(v: config.VaeDecode) pipeline.VaeDecode {
    return switch (v) {
        .auto => .auto,
        .whole => .whole,
        .gpu_tiled => .gpu_tiled,
        .cpu_tiled => .cpu_tiled,
    };
}

pub fn toPipelineAct(p: config.Precision) ?pipeline.Act {
    return switch (p) {
        .auto => null,
        .f32 => .f32,
        .f16 => .f16,
        .bf16 => .bf16,
    };
}

pub fn toPipelineSyntax(s: config.PromptSyntax) pipeline.PromptSyntax {
    return switch (s) {
        .comfy => .comfy,
        .a1111 => .a1111,
    };
}

pub fn toPipelineCondShape(s: config.CondShape) pipeline.CondNoise.Shape {
    return switch (s) {
        .shared => .shared,
        .independent => .independent,
    };
}

pub fn toPipelineSteerMode(c: config.SteerMode) pipeline.CondNoise.Mode {
    return switch (c) {
        .prompt => .prompt,
        .empty => .empty,
        .orthogonal => .orthogonal,
        .concat => .concat,
    };
}

pub fn toPipelineActOp(o: config.ActOp) pipeline.dit_act.Op {
    return switch (o) {
        .add => .add,
        .gain => .gain,
    };
}

pub fn toPipelineCompat(c: config.Compat) pipeline.Compat {
    return switch (c) {
        .comfy => .comfy,
        .a1111 => .a1111,
    };
}

pub fn toPipelineEmphasis(e: config.Emphasis) pipeline.Emphasis {
    return switch (e) {
        .original => .original,
        .no_norm => .no_norm,
        .ignore => .ignore,
    };
}

pub fn toPipelineSampler(s: config.Sampler) tp.sampler.Kind {
    return switch (s) {
        .euler => .euler,
        .euler_ancestral => .euler_ancestral,
        .heun => .heun,
        .dpm_2 => .dpm_2,
        .dpm_2_ancestral => .dpm_2_ancestral,
        .dpmpp_2s_ancestral => .dpmpp_2s_ancestral,
        .dpmpp_sde => .dpmpp_sde,
        .dpmpp_2m => .dpmpp_2m,
        .dpmpp_2m_sde => .dpmpp_2m_sde,
        .dpmpp_2m_sde_heun => .dpmpp_2m_sde_heun,
        .dpmpp_3m_sde => .dpmpp_3m_sde,
    };
}

/// `.default` -> null, meaning "the architecture's own", which
/// `Session.scheduleWith` resolves per family.
pub fn toPipelineScheduler(s: config.Scheduler) ?tp.sampler.Scheduler {
    return switch (s) {
        .default => null,
        .normal => .normal,
        .karras => .karras,
        .exponential => .exponential,
        .sgm_uniform => .sgm_uniform,
        .simple => .simple,
        .ddim_uniform => .ddim_uniform,
        .beta => .beta,
        .linear_quadratic => .linear_quadratic,
        .kl_optimal => .kl_optimal,
        .reference => .reference,
        .turbo => .turbo,
    };
}

/// The inverses, for a form seeded from a render that already happened.
pub fn fromPipelineSampler(k: tp.sampler.Kind) config.Sampler {
    return switch (k) {
        .euler => .euler,
        .euler_ancestral => .euler_ancestral,
        .heun => .heun,
        .dpm_2 => .dpm_2,
        .dpm_2_ancestral => .dpm_2_ancestral,
        .dpmpp_2s_ancestral => .dpmpp_2s_ancestral,
        .dpmpp_sde => .dpmpp_sde,
        .dpmpp_2m => .dpmpp_2m,
        .dpmpp_2m_sde => .dpmpp_2m_sde,
        .dpmpp_2m_sde_heun => .dpmpp_2m_sde_heun,
        .dpmpp_3m_sde => .dpmpp_3m_sde,
    };
}

pub fn fromPipelineScheduler(s: ?tp.sampler.Scheduler) config.Scheduler {
    const v = s orelse return .default;
    return switch (v) {
        .normal => .normal,
        .karras => .karras,
        .exponential => .exponential,
        .sgm_uniform => .sgm_uniform,
        .simple => .simple,
        .ddim_uniform => .ddim_uniform,
        .beta => .beta,
        .linear_quadratic => .linear_quadratic,
        .kl_optimal => .kl_optimal,
        .reference => .reference,
        .turbo => .turbo,
    };
}

/// Round a requested edge to a multiple of 16 (the pipeline's requirement)
/// within sane bounds.
pub fn clampDim(n: usize) usize {
    const c = std.math.clamp(n, 256, 4096);
    return c / 16 * 16;
}

test "clampDim rounds to multiple of 16 within bounds" {
    try std.testing.expectEqual(@as(usize, 1024), clampDim(1024));
    try std.testing.expectEqual(@as(usize, 1024), clampDim(1030));
    try std.testing.expectEqual(@as(usize, 256), clampDim(10));
    try std.testing.expectEqual(@as(usize, 4096), clampDim(99999));
    try std.testing.expectEqual(@as(usize, 512), clampDim(519));
}

test "auto names no backend a client could check, and resolves to one where it runs" {
    try std.testing.expectEqual(@as(?pipeline.Backend, null), explicitBackend(.auto));
    try std.testing.expectEqual(@as(?pipeline.Backend, .vulkan), explicitBackend(.vulkan));
    try std.testing.expectEqual(pipeline.detectBackend(), toPipelineBackend(.auto));
}

test "the sampler and scheduler maps invert each other" {
    inline for (@typeInfo(config.Sampler).@"enum".field_values) |value| {
        const s: config.Sampler = @fromBackingInt(value);
        try std.testing.expectEqual(s, fromPipelineSampler(toPipelineSampler(s)));
    }
    inline for (@typeInfo(config.Scheduler).@"enum".field_values) |value| {
        const s: config.Scheduler = @fromBackingInt(value);
        try std.testing.expectEqual(s, fromPipelineScheduler(toPipelineScheduler(s)));
    }
}
