//! The settings enums against the pipeline's. They are separate enums because
//! the config's carry UI labels and a stable serialized spelling, and each map
//! is written out rather than derived from tag names so a member with no twin
//! is a compile error here, not a silently wrong row.
const std = @import("std");
const tp = @import("TensorPencil");
const pipeline = tp.pipeline;
const config = @import("config.zig");

pub fn toPipelineBackend(b: config.Backend) pipeline.Backend {
    return switch (b) {
        .cpu => .cpu,
        .vulkan => .vulkan,
        .zig_cuda => .zig_cuda,
        .cuda => .cuda,
    };
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

pub fn toPipelineSyntax(s: config.PromptSyntax) pipeline.PromptSyntax {
    return switch (s) {
        .comfy => .comfy,
        .a1111 => .a1111,
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

test "the sampler and scheduler maps invert each other" {
    inline for (std.meta.fields(config.Sampler)) |f| {
        const s: config.Sampler = @enumFromInt(f.value);
        try std.testing.expectEqual(s, fromPipelineSampler(toPipelineSampler(s)));
    }
    inline for (std.meta.fields(config.Scheduler)) |f| {
        const s: config.Scheduler = @enumFromInt(f.value);
        try std.testing.expectEqual(s, fromPipelineScheduler(toPipelineScheduler(s)));
    }
}
