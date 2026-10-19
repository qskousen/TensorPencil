//! The entry table of dual.zig, in a file the host can import: the Vulkan Context
//! reads workgroup sizes from it and the CUDA Backend launches with them, so a
//! kernel's shape is written once. dual.zig checks that every body here has a row
//! and every row a body.

const std = @import("std");

pub const Kind = enum {
    /// One thread per element (or f16 pair): the host passes the thread count.
    elems,
    /// One subgroup per row, striding: the host passes the row count and sizes the
    /// launch with `rowGroups`.
    rows,
};

/// An activation buffer's element type.
pub const Dt = enum {
    f32,
    f16,
    bf16,

    pub fn bytes(d: Dt) usize {
        return if (d == .f32) 4 else 2;
    }
};

/// The entry of a body that takes an element type, at `dt`: `base` itself at f32.
pub fn typedName(comptime base: []const u8, comptime dt: Dt) [:0]const u8 {
    return switch (dt) {
        .f32 => base ++ "",
        .f16 => base ++ "_f16",
        .bf16 => base ++ "_bf16",
    };
}

/// The `copy` instance from `in` to `out`. Only one side may be 16-bit when they differ.
pub fn copyName(comptime in: Dt, comptime out: Dt) [:0]const u8 {
    if (in == out) return typedName("copy", in);
    if (in != .f32 and out != .f32) @compileError("copy: no 16-bit to 16-bit cast");
    return "copy_" ++ @tagName(in) ++ "_" ++ @tagName(out);
}

/// Gather every 3x3 conv's patch through an f32 buffer and a convert pass rather than
/// straight into the GEMM operand, on both backends, for an A/B
/// (`diffusion-profile --ab convpatch`, `wan-vae-test`).
pub var force_conv_patch: bool = false;

/// Rows per chunk of `col_mean`'s first pass; the second sums `ceil(rows / this)` partials.
pub const col_mean_chunk: usize = 64;

/// Where a band of a 3x3 convolution reads its source, as `im2col_h16` / `im2col_bf16`
/// take it.
pub const ConvTaps = struct {
    ci: usize,
    src_w: usize,
    /// The grid the taps are addressed in: the OUTPUT height when upsampling.
    extent_h: usize,
    out_w: usize,
    /// The band's first output position.
    first: usize,
    /// 0 stride 1, 1 fused nearest 2x upsample, 2 stride 2 (the kernel's f0).
    mode: u2,

    /// Whether the kernel can carry this shape into a `k_pad`-wide operand: two
    /// dimensions share each of two 32-bit push words, and a thread writes eight columns.
    pub fn fits(t: ConvTaps, k_pad: usize) bool {
        return k_pad % 8 == 0 and k_pad <= 0xffff and t.ci <= 0xffff and t.src_w <= 0xffff and t.extent_h <= 0xffff;
    }

    /// The u0..u6 push words for an `[m_pad][k_pad]` operand whose first `m` rows are
    /// real; u0 is also the thread count.
    pub fn push(t: ConvTaps, m_pad: usize, k_pad: usize, m: usize) [7]u32 {
        std.debug.assert(t.fits(k_pad) and m <= m_pad);
        return .{
            @intCast(m_pad * k_pad / 8),
            @intCast(9 * t.ci),
            @intCast(t.ci | k_pad << 16),
            @intCast(t.src_w | t.extent_h << 16),
            @intCast(t.out_w),
            @intCast(t.first),
            @intCast(m),
        };
    }
};

pub const Entry = struct {
    name: [:0]const u8,
    /// Threads per workgroup.
    wg: u32,
    kind: Kind = .elems,
};

pub const elem_wg: u32 = 256;
pub const row_wg: u32 = 256;

/// Subgroup width the host assumes when sizing a row launch. A wider subgroup only
/// leaves subgroups idle: the row kernels stride by the real count.
pub const host_sg: u32 = 32;

pub const entries = [_]Entry{
    .{ .name = "add", .wg = elem_wg },
    .{ .name = "add_relu", .wg = elem_wg },
    .{ .name = "add_scaled", .wg = elem_wg },
    .{ .name = "compact_acc", .wg = elem_wg },
    .{ .name = "relu", .wg = elem_wg },
    .{ .name = "silu", .wg = elem_wg },
    .{ .name = "silu_mul", .wg = elem_wg },
    .{ .name = "sigmoid_mul", .wg = elem_wg },
    .{ .name = "gelu", .wg = elem_wg },
    .{ .name = "gelu_mul", .wg = elem_wg },
    .{ .name = "gelu_quick", .wg = elem_wg },
    .{ .name = "gelu_quick_mul", .wg = elem_wg },
    .{ .name = "gelu_erf", .wg = elem_wg },
    .{ .name = "geglu", .wg = elem_wg },
    .{ .name = "softplus_gate", .wg = elem_wg },
    .{ .name = "copy", .wg = elem_wg },
    .{ .name = "scale_f32", .wg = elem_wg },
    .{ .name = "concat_ch", .wg = elem_wg },
    .{ .name = "scale_concat", .wg = elem_wg },
    .{ .name = "scale_i32", .wg = elem_wg },
    .{ .name = "quantize_i8", .wg = elem_wg },
    .{ .name = "rms_apply_w", .wg = elem_wg },
    .{ .name = "rms_partial", .wg = elem_wg },
    .{ .name = "rms_combine", .wg = elem_wg },
    .{ .name = "modulate", .wg = elem_wg },
    .{ .name = "gated_add", .wg = elem_wg },
    .{ .name = "gated_add16", .wg = elem_wg },
    .{ .name = "rope_inter", .wg = elem_wg },
    .{ .name = "rope_half", .wg = elem_wg },
    .{ .name = "rope_half_pos", .wg = elem_wg },
    .{ .name = "rope_half_part", .wg = elem_wg },
    .{ .name = "deinterleave2", .wg = elem_wg },
    .{ .name = "deinterleave3", .wg = elem_wg },
    .{ .name = "gdn_gates", .wg = elem_wg },
    .{ .name = "gdn_gates_batch", .wg = elem_wg },
    .{ .name = "gdn_conv_step", .wg = elem_wg },
    .{ .name = "gdn_conv_batch", .wg = elem_wg },
    .{ .name = "gdn_conv_state", .wg = elem_wg },
    .{ .name = "penalize", .wg = elem_wg },
    .{ .name = "argmax_reduce", .wg = elem_wg },
    .{ .name = "argmax_final", .wg = row_wg, .kind = .rows },
    .{ .name = "topk_reduce", .wg = elem_wg },
    .{ .name = "head_pad_h16", .wg = elem_wg },
    .{ .name = "head_unpad", .wg = elem_wg },
    .{ .name = "head_pad", .wg = elem_wg },
    .{ .name = "gather_head", .wg = elem_wg },
    .{ .name = "gather_vt", .wg = elem_wg },
    .{ .name = "scatter_head", .wg = elem_wg },
    .{ .name = "gather_head_b", .wg = elem_wg },
    .{ .name = "gather_vt_b", .wg = elem_wg },
    .{ .name = "scatter_head_b", .wg = elem_wg },
    .{ .name = "gather_kmajor", .wg = elem_wg },
    .{ .name = "gather_kmajor_h16", .wg = elem_wg },
    .{ .name = "gather_kmajor16", .wg = elem_wg },
    .{ .name = "f32_to_h16", .wg = elem_wg },
    .{ .name = "pad_h16", .wg = elem_wg },
    .{ .name = "pad_bf16", .wg = elem_wg },
    .{ .name = "conv_w_h16", .wg = elem_wg },
    .{ .name = "bf16_to_f32", .wg = elem_wg },
    .{ .name = "f16_to_f32", .wg = elem_wg },
    .{ .name = "silu_mul_h16", .wg = elem_wg },
    .{ .name = "sigmoid_mul_h16", .wg = elem_wg },
    .{ .name = "silu_mul16", .wg = elem_wg },
    .{ .name = "sigmoid_mul_g16", .wg = elem_wg },
    .{ .name = "kv_store_f16", .wg = elem_wg },
    .{ .name = "transpose_kmajor16", .wg = elem_wg },
    .{ .name = "bias_compact", .wg = elem_wg },
    .{ .name = "bias_add", .wg = elem_wg },
    .{ .name = "add_bias_rows", .wg = elem_wg },
    .{ .name = "gn_apply", .wg = elem_wg },
    .{ .name = "im2col", .wg = elem_wg },
    .{ .name = "im2col_sd", .wg = elem_wg },
    .{ .name = "gather_rows", .wg = elem_wg },
    .{ .name = "scatter_add_rows", .wg = elem_wg },
    .{ .name = "moe_combine", .wg = elem_wg },
    .{ .name = "gemv_combine", .wg = elem_wg },
    .{ .name = "gemv_combine4", .wg = elem_wg },
    .{ .name = "modulate_pr", .wg = elem_wg },
    .{ .name = "gated_add_pr", .wg = elem_wg },
    .{ .name = "dw_conv3", .wg = elem_wg },
    .{ .name = "col_mean", .wg = elem_wg },
    .{ .name = "mul_cols_sigmoid", .wg = elem_wg },
    .{ .name = "nerf_feat", .wg = elem_wg },
    .{ .name = "patch_scatter", .wg = elem_wg },
    .{ .name = "win_gather", .wg = elem_wg },
    .{ .name = "win_scatter", .wg = elem_wg },
    .{ .name = "rope_imrope", .wg = elem_wg },
    .{ .name = "rope_imrope_pos", .wg = elem_wg },
    .{ .name = "rope_vision", .wg = elem_wg },
    .{ .name = "rope_vision_gemma4", .wg = elem_wg },
    .{ .name = "rope_half_span_pos", .wg = elem_wg },
    .{ .name = "rope_inter_span_pos", .wg = elem_wg },
    .{ .name = "pixel_shuffle", .wg = elem_wg },
    .{ .name = "im2col_stride", .wg = elem_wg },
    .{ .name = "im2col1d", .wg = elem_wg },
    .{ .name = "im2col3d", .wg = elem_wg },
    .{ .name = "aa_up_snake", .wg = elem_wg },
    .{ .name = "aa_down", .wg = elem_wg },
    .{ .name = "convt1d_ca", .wg = elem_wg },
    .{ .name = "snake1d_ca", .wg = elem_wg },
    .{ .name = "mean_heads_pool", .wg = elem_wg },
    .{ .name = "dequant_fp8_f16", .wg = elem_wg },
    .{ .name = "dequant_fp8_bf16", .wg = elem_wg },
    .{ .name = "dequant_q8_0_f16", .wg = elem_wg },
    .{ .name = "dequant_q8_0_bf16", .wg = elem_wg },
    .{ .name = "dequant_q4_0_f16", .wg = elem_wg },
    .{ .name = "dequant_q4_0_bf16", .wg = elem_wg },
    .{ .name = "dequant_q1_0_f16", .wg = elem_wg },
    .{ .name = "dequant_q1_0_bf16", .wg = elem_wg },
    .{ .name = "dequant_q2_0_g64_f16", .wg = elem_wg },
    .{ .name = "dequant_q2_0_g64_bf16", .wg = elem_wg },
    .{ .name = "dequant_q2_0_g128_f16", .wg = elem_wg },
    .{ .name = "dequant_q2_0_g128_bf16", .wg = elem_wg },
    .{ .name = "dequant_iq4_nl_f16", .wg = elem_wg },
    .{ .name = "dequant_iq4_nl_bf16", .wg = elem_wg },
    .{ .name = "dequant_iq4_xs_f16", .wg = elem_wg },
    .{ .name = "dequant_iq4_xs_bf16", .wg = elem_wg },
    .{ .name = "dequant_q4_k_f16", .wg = elem_wg },
    .{ .name = "dequant_q4_k_bf16", .wg = elem_wg },
    .{ .name = "dequant_q5_k_f16", .wg = elem_wg },
    .{ .name = "dequant_q5_k_bf16", .wg = elem_wg },
    .{ .name = "dequant_q6_k_f16", .wg = elem_wg },
    .{ .name = "dequant_q6_k_bf16", .wg = elem_wg },
    .{ .name = "dequant_q2_k_f16", .wg = elem_wg },
    .{ .name = "dequant_q2_k_bf16", .wg = elem_wg },
    .{ .name = "dequant_q3_k_f16", .wg = elem_wg },
    .{ .name = "dequant_q3_k_bf16", .wg = elem_wg },
    .{ .name = "dequant_iq2_xxs_f16", .wg = elem_wg },
    .{ .name = "dequant_iq2_xxs_bf16", .wg = elem_wg },
    .{ .name = "dequant_iq2_xs_f16", .wg = elem_wg },
    .{ .name = "dequant_iq2_xs_bf16", .wg = elem_wg },
    .{ .name = "dequant_iq3_xxs_f16", .wg = elem_wg },
    .{ .name = "dequant_iq3_xxs_bf16", .wg = elem_wg },
    .{ .name = "dequant_iq3_s_f16", .wg = elem_wg },
    .{ .name = "dequant_iq3_s_bf16", .wg = elem_wg },
    .{ .name = "gn_combine", .wg = elem_wg },
    .{ .name = "softmax_partial", .wg = elem_wg },
    .{ .name = "softmax_combine", .wg = elem_wg },
    .{ .name = "rowscale_i8", .wg = elem_wg },
    .{ .name = "i8row_rot_bf16", .wg = elem_wg },
    .{ .name = "i4row_rot_bf16", .wg = elem_wg },
    .{ .name = "dequant_i8row_bf16", .wg = elem_wg },
    .{ .name = "i8row_rot_f16", .wg = elem_wg },
    .{ .name = "i4row_rot_f16", .wg = elem_wg },
    .{ .name = "dequant_i8row_f16", .wg = elem_wg },
    .{ .name = "i8row_rot_f32", .wg = elem_wg },
    .{ .name = "i4row_rot_f32", .wg = elem_wg },
    .{ .name = "dequant_i8row_f32", .wg = elem_wg },
    .{ .name = "wo_unpack_f32", .wg = elem_wg },
    .{ .name = "transpose_pad_f32", .wg = elem_wg },
    .{ .name = "im2col_h16", .wg = elem_wg },
    .{ .name = "im2col_h16_f16", .wg = elem_wg },
    .{ .name = "im2col_h16_bf16", .wg = elem_wg },
    .{ .name = "im2col_bf16", .wg = elem_wg },
    .{ .name = "im2col_bf16_f16", .wg = elem_wg },
    .{ .name = "im2col_bf16_bf16", .wg = elem_wg },
    .{ .name = "qknorm_rope", .wg = elem_wg },
    .{ .name = "l2norm_rows", .wg = row_wg, .kind = .rows },
    .{ .name = "l2norm_rows_g", .wg = row_wg, .kind = .rows },
    .{ .name = "act_steer", .wg = row_wg, .kind = .rows },
    .{ .name = "attn_rows_f32", .wg = row_wg, .kind = .rows },
    .{ .name = "attn_rows_f16", .wg = row_wg, .kind = .rows },
    .{ .name = "attn_rows_q8", .wg = row_wg, .kind = .rows },
    .{ .name = "attn_merge", .wg = elem_wg },
    .{ .name = "quantize_q8_1", .wg = elem_wg },
    .{ .name = "gdn_delta_chunk_d64", .wg = 64 },
    .{ .name = "gdn_delta_chunk_d128", .wg = 128 },
    .{ .name = "rmsnorm", .wg = row_wg, .kind = .rows },
    .{ .name = "qk_norm_rope", .wg = row_wg, .kind = .rows },
    .{ .name = "group_rmsnorm", .wg = row_wg, .kind = .rows },
    .{ .name = "rms_mod", .wg = row_wg, .kind = .rows },
    .{ .name = "rms_mod_h16", .wg = row_wg, .kind = .rows },
    .{ .name = "layernorm", .wg = row_wg, .kind = .rows },
    .{ .name = "ln_mod", .wg = row_wg, .kind = .rows },
    .{ .name = "gn_stats", .wg = row_wg, .kind = .rows },
    .{ .name = "rowmax_i8", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv_dual_q2_k", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv_dual_q2_k_q8", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv_dual_q3_k", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv_dual_q3_k_q8", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv_dual_q4_k", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv_dual_q4_k_q8", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv_dual_q8_0", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv_dual_q8_0_q8", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv_dual_q4_0", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv_dual_q4_0_q8", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv_dual_iq4_nl", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv_dual_iq4_nl_q8", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv_dual_iq4_xs", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv_dual_iq4_xs_q8", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv_dual_q5_k", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv_dual_q5_k_q8", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv_dual_q6_k", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv_dual_q6_k_q8", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv_dual_q1_0", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv_dual_q1_0_q8", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv_dual_q2_0_g64", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv_dual_q2_0_g64_q8", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv_dual_q2_0_g128", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv_dual_q2_0_g128_q8", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv_dual_iq2_xxs", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv_dual_iq2_xxs_q8", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv_dual_iq2_xs", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv_dual_iq2_xs_q8", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv_dual_iq3_xxs", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv_dual_iq3_xxs_q8", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv_dual_iq3_s", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv_dual_iq3_s_q8", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv4_dual_q2_k", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv4_dual_q3_k", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv4_dual_q4_k", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv4_dual_q8_0", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv4_dual_q4_0", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv4_dual_iq4_nl", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv4_dual_iq4_xs", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv4_dual_q5_k", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv4_dual_q6_k", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv4_dual_q1_0", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv4_dual_q2_0_g64", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv4_dual_q2_0_g128", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv4_dual_iq2_xxs", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv4_dual_iq2_xs", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv4_dual_iq3_xxs", .wg = row_wg, .kind = .rows },
    .{ .name = "gemv4_dual_iq3_s", .wg = row_wg, .kind = .rows },
    .{ .name = "attn_rows_g2_f32", .wg = row_wg, .kind = .rows },
    .{ .name = "attn_rows_g2_f16", .wg = row_wg, .kind = .rows },
    .{ .name = "attn_rows_g2_q8", .wg = row_wg, .kind = .rows },
    .{ .name = "attn_rows_g4_f32", .wg = row_wg, .kind = .rows },
    .{ .name = "attn_rows_g4_f16", .wg = row_wg, .kind = .rows },
    .{ .name = "attn_rows_g4_q8", .wg = row_wg, .kind = .rows },
    .{ .name = "attn_rows_g8_f32", .wg = row_wg, .kind = .rows },
    .{ .name = "attn_rows_g8_f16", .wg = row_wg, .kind = .rows },
    .{ .name = "attn_rows_g8_q8", .wg = row_wg, .kind = .rows },
    // The f16 and bf16 instances of the bodies that take an element type.
    .{ .name = "add_f16", .wg = elem_wg },
    .{ .name = "add_bf16", .wg = elem_wg },
    .{ .name = "geglu_f16", .wg = elem_wg },
    .{ .name = "geglu_bf16", .wg = elem_wg },
    .{ .name = "bias_compact_f16", .wg = elem_wg },
    .{ .name = "bias_compact_bf16", .wg = elem_wg },
    .{ .name = "add_bias_rows_f16", .wg = elem_wg },
    .{ .name = "add_bias_rows_bf16", .wg = elem_wg },
    .{ .name = "gn_apply_f16", .wg = elem_wg },
    .{ .name = "gn_apply_bf16", .wg = elem_wg },
    .{ .name = "im2col_sd_f16", .wg = elem_wg },
    .{ .name = "im2col_sd_bf16", .wg = elem_wg },
    .{ .name = "qknorm_rope_f16", .wg = elem_wg },
    .{ .name = "qknorm_rope_bf16", .wg = elem_wg },
    .{ .name = "layernorm_f16", .wg = row_wg, .kind = .rows },
    .{ .name = "layernorm_bf16", .wg = row_wg, .kind = .rows },
    .{ .name = "layernorm_f32_f16", .wg = row_wg, .kind = .rows },
    .{ .name = "layernorm_f32_bf16", .wg = row_wg, .kind = .rows },
    .{ .name = "gn_stats_f16", .wg = row_wg, .kind = .rows },
    .{ .name = "gn_stats_bf16", .wg = row_wg, .kind = .rows },
    .{ .name = "pad_h16_f16", .wg = elem_wg },
    .{ .name = "pad_h16_bf16", .wg = elem_wg },
    .{ .name = "pad_bf16_f16", .wg = elem_wg },
    .{ .name = "pad_bf16_bf16", .wg = elem_wg },
    .{ .name = "bias_add_f16", .wg = elem_wg },
    .{ .name = "bias_add_bf16", .wg = elem_wg },
    .{ .name = "conv_w_bf16", .wg = elem_wg },
    .{ .name = "bias_add_bf16_f32", .wg = elem_wg },
    .{ .name = "scale_i32_f16", .wg = elem_wg },
    .{ .name = "scale_i32_bf16", .wg = elem_wg },
    .{ .name = "copy_f16", .wg = elem_wg },
    .{ .name = "copy_bf16", .wg = elem_wg },
    .{ .name = "copy_f32_f16", .wg = elem_wg },
    .{ .name = "copy_f32_bf16", .wg = elem_wg },
    .{ .name = "copy_f16_f32", .wg = elem_wg },
    .{ .name = "copy_bf16_f32", .wg = elem_wg },
    .{ .name = "concat_ch_f16", .wg = elem_wg },
    .{ .name = "concat_ch_bf16", .wg = elem_wg },
    .{ .name = "compact_acc_f16", .wg = elem_wg },
    .{ .name = "compact_acc_bf16", .wg = elem_wg },
    .{ .name = "silu_mul_f16", .wg = elem_wg },
    .{ .name = "silu_mul_bf16", .wg = elem_wg },
    .{ .name = "sigmoid_mul_f16", .wg = elem_wg },
    .{ .name = "sigmoid_mul_bf16", .wg = elem_wg },
    .{ .name = "gated_add_f16", .wg = elem_wg },
    .{ .name = "gated_add_bf16", .wg = elem_wg },
    .{ .name = "rope_inter_f16", .wg = elem_wg },
    .{ .name = "rope_inter_bf16", .wg = elem_wg },
    .{ .name = "act_steer_f16", .wg = row_wg, .kind = .rows },
    .{ .name = "act_steer_bf16", .wg = row_wg, .kind = .rows },
    .{ .name = "vae_norm", .wg = row_wg, .kind = .rows },
    .{ .name = "vae_norm_f16", .wg = row_wg, .kind = .rows },
    .{ .name = "vae_norm_bf16", .wg = row_wg, .kind = .rows },
    .{ .name = "rmsnorm_f16", .wg = row_wg, .kind = .rows },
    .{ .name = "rmsnorm_bf16", .wg = row_wg, .kind = .rows },
    .{ .name = "rms_mod_f16", .wg = row_wg, .kind = .rows },
    .{ .name = "rms_mod_bf16", .wg = row_wg, .kind = .rows },
    .{ .name = "rms_mod_h16_f16", .wg = row_wg, .kind = .rows },
    .{ .name = "rms_mod_h16_bf16", .wg = row_wg, .kind = .rows },
    .{ .name = "modulate_f16", .wg = elem_wg },
    .{ .name = "modulate_bf16", .wg = elem_wg },
    .{ .name = "kron_mix", .wg = elem_wg },
    .{ .name = "kron_mix_f16", .wg = elem_wg },
    .{ .name = "kron_mix_bf16", .wg = elem_wg },
    .{ .name = "kron_stack", .wg = elem_wg },
    .{ .name = "kron_stack_f16", .wg = elem_wg },
    .{ .name = "kron_stack_bf16", .wg = elem_wg },
    .{ .name = "dora_rows", .wg = elem_wg },
    .{ .name = "dora_rows_f16", .wg = elem_wg },
    .{ .name = "dora_rows_bf16", .wg = elem_wg },
    .{ .name = "dora_cols", .wg = elem_wg },
    .{ .name = "dora_cols_f16", .wg = elem_wg },
    .{ .name = "dora_cols_bf16", .wg = elem_wg },
    .{ .name = "gelu_erf_f16", .wg = elem_wg },
    .{ .name = "gelu_erf_bf16", .wg = elem_wg },
    .{ .name = "silu_f16", .wg = elem_wg },
    .{ .name = "silu_bf16", .wg = elem_wg },
    .{ .name = "modulate_pr_f16", .wg = elem_wg },
    .{ .name = "modulate_pr_bf16", .wg = elem_wg },
    .{ .name = "gated_add_pr_f16", .wg = elem_wg },
    .{ .name = "gated_add_pr_bf16", .wg = elem_wg },
    .{ .name = "dw_conv3_f16", .wg = elem_wg },
    .{ .name = "dw_conv3_bf16", .wg = elem_wg },
    .{ .name = "col_mean_f16", .wg = elem_wg },
    .{ .name = "col_mean_bf16", .wg = elem_wg },
    .{ .name = "mul_cols_sigmoid_f16", .wg = elem_wg },
    .{ .name = "mul_cols_sigmoid_bf16", .wg = elem_wg },
    .{ .name = "nerf_feat_f16", .wg = elem_wg },
    .{ .name = "nerf_feat_bf16", .wg = elem_wg },
    .{ .name = "ln_mod_f16", .wg = row_wg, .kind = .rows },
    .{ .name = "ln_mod_bf16", .wg = row_wg, .kind = .rows },
    .{ .name = "gelu_f16", .wg = elem_wg },
    .{ .name = "gelu_bf16", .wg = elem_wg },
    .{ .name = "flag_nonfinite", .wg = elem_wg },
    .{ .name = "flag_nonfinite_f16", .wg = elem_wg },
    .{ .name = "flag_nonfinite_bf16", .wg = elem_wg },
    .{ .name = "add_scaled_f16", .wg = elem_wg },
    .{ .name = "add_scaled_bf16", .wg = elem_wg },
    .{ .name = "gated_add_scaled", .wg = elem_wg },
    .{ .name = "gated_add_scaled_f16", .wg = elem_wg },
    .{ .name = "gated_add_scaled_bf16", .wg = elem_wg },
    .{ .name = "situ", .wg = elem_wg },
    .{ .name = "situ_f16", .wg = elem_wg },
    .{ .name = "situ_bf16", .wg = elem_wg },
    .{ .name = "dup_up_add", .wg = elem_wg },
    .{ .name = "dup_up_add_f16", .wg = elem_wg },
    .{ .name = "dup_up_add_bf16", .wg = elem_wg },
    .{ .name = "w6a8_decode", .wg = elem_wg },
    .{ .name = "w6a8_decode_t", .wg = elem_wg },
};

pub fn get(comptime name: []const u8) Entry {
    @setEvalBranchQuota(400_000);
    inline for (entries) |e| if (comptime std.mem.eql(u8, e.name, name)) return e;
    @compileError("dual_table: no entry named " ++ name);
}

pub fn wgOf(comptime name: []const u8) u32 {
    return get(name).wg;
}

/// Workgroups for a row kernel over `rows` rows.
pub fn rowGroups(rows: usize) usize {
    return (rows * host_sg + row_wg - 1) / row_wg;
}
