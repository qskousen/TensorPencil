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
    .{ .name = "geglu_h16", .wg = elem_wg },
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
    .{ .name = "argmax_final", .wg = elem_wg },
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
    .{ .name = "f32_to_h16_pad", .wg = elem_wg },
    .{ .name = "f32_to_bf16_pad", .wg = elem_wg },
    .{ .name = "h16_to_h16_pad", .wg = elem_wg },
    .{ .name = "bf16_to_h16_pad", .wg = elem_wg },
    .{ .name = "f16_to_f32", .wg = elem_wg },
    .{ .name = "add_h16", .wg = elem_wg },
    .{ .name = "silu_mul_h16", .wg = elem_wg },
    .{ .name = "sigmoid_mul_h16", .wg = elem_wg },
    .{ .name = "silu_mul16", .wg = elem_wg },
    .{ .name = "sigmoid_mul_g16", .wg = elem_wg },
    .{ .name = "kv_store_f16", .wg = elem_wg },
    .{ .name = "pack_h16_kmajor", .wg = elem_wg },
    .{ .name = "bias_compact", .wg = elem_wg },
    .{ .name = "bias_compact_h16", .wg = elem_wg },
    .{ .name = "bias_add_f16", .wg = elem_wg },
    .{ .name = "bias_add_h16", .wg = elem_wg },
    .{ .name = "add_bias_rows", .wg = elem_wg },
    .{ .name = "add_bias_rows_h16", .wg = elem_wg },
    .{ .name = "gn_apply", .wg = elem_wg },
    .{ .name = "gn_apply_h16", .wg = elem_wg },
    .{ .name = "vae_norm", .wg = elem_wg },
    .{ .name = "im2col", .wg = elem_wg },
    .{ .name = "im2col_sd", .wg = elem_wg },
    .{ .name = "im2col_sd_h16", .wg = elem_wg },
    .{ .name = "gather_rows", .wg = elem_wg },
    .{ .name = "scatter_add_rows", .wg = elem_wg },
    .{ .name = "moe_combine", .wg = elem_wg },
    .{ .name = "gemv_combine", .wg = elem_wg },
    .{ .name = "gemv_combine4", .wg = elem_wg },
    .{ .name = "rope_imrope", .wg = elem_wg },
    .{ .name = "rope_imrope_pos", .wg = elem_wg },
    .{ .name = "rope_vision", .wg = elem_wg },
    .{ .name = "rope_vision_gemma4", .wg = elem_wg },
    .{ .name = "rope_half_span_pos", .wg = elem_wg },
    .{ .name = "rope_inter_span_pos", .wg = elem_wg },
    .{ .name = "pixel_shuffle", .wg = elem_wg },
    .{ .name = "im2col_stride", .wg = elem_wg },
    .{ .name = "im2col1d", .wg = elem_wg },
    .{ .name = "aa_up_snake", .wg = elem_wg },
    .{ .name = "aa_down", .wg = elem_wg },
    .{ .name = "convt1d_ca", .wg = elem_wg },
    .{ .name = "snake1d_ca", .wg = elem_wg },
    .{ .name = "mean_heads_pool", .wg = elem_wg },
    .{ .name = "dequant_fp8_f16", .wg = elem_wg },
    .{ .name = "dequant_fp8_bf16", .wg = elem_wg },
    .{ .name = "dequant_fp8_f32", .wg = elem_wg },
    .{ .name = "dequant_q8_0_f16", .wg = elem_wg },
    .{ .name = "dequant_q8_0_bf16", .wg = elem_wg },
    .{ .name = "dequant_q8_0_f32", .wg = elem_wg },
    .{ .name = "dequant_q4_0_f16", .wg = elem_wg },
    .{ .name = "dequant_q4_0_bf16", .wg = elem_wg },
    .{ .name = "dequant_q4_0_f32", .wg = elem_wg },
    .{ .name = "dequant_q1_0_f16", .wg = elem_wg },
    .{ .name = "dequant_q1_0_bf16", .wg = elem_wg },
    .{ .name = "dequant_q1_0_f32", .wg = elem_wg },
    .{ .name = "dequant_q2_0_g64_f16", .wg = elem_wg },
    .{ .name = "dequant_q2_0_g64_bf16", .wg = elem_wg },
    .{ .name = "dequant_q2_0_g64_f32", .wg = elem_wg },
    .{ .name = "dequant_q2_0_g128_f16", .wg = elem_wg },
    .{ .name = "dequant_q2_0_g128_bf16", .wg = elem_wg },
    .{ .name = "dequant_q2_0_g128_f32", .wg = elem_wg },
    .{ .name = "dequant_iq4_nl_f16", .wg = elem_wg },
    .{ .name = "dequant_iq4_nl_bf16", .wg = elem_wg },
    .{ .name = "dequant_iq4_nl_f32", .wg = elem_wg },
    .{ .name = "dequant_iq4_xs_f16", .wg = elem_wg },
    .{ .name = "dequant_iq4_xs_bf16", .wg = elem_wg },
    .{ .name = "dequant_iq4_xs_f32", .wg = elem_wg },
    .{ .name = "dequant_q4_k_f16", .wg = elem_wg },
    .{ .name = "dequant_q4_k_bf16", .wg = elem_wg },
    .{ .name = "dequant_q4_k_f32", .wg = elem_wg },
    .{ .name = "dequant_q5_k_f16", .wg = elem_wg },
    .{ .name = "dequant_q5_k_bf16", .wg = elem_wg },
    .{ .name = "dequant_q5_k_f32", .wg = elem_wg },
    .{ .name = "dequant_q6_k_f16", .wg = elem_wg },
    .{ .name = "dequant_q6_k_bf16", .wg = elem_wg },
    .{ .name = "dequant_q6_k_f32", .wg = elem_wg },
    .{ .name = "gn_combine", .wg = elem_wg },
    .{ .name = "softmax_partial", .wg = elem_wg },
    .{ .name = "softmax_combine", .wg = elem_wg },
    .{ .name = "rowscale_i8", .wg = elem_wg },
    .{ .name = "qknorm_rope16", .wg = elem_wg },
    .{ .name = "qknorm_rope_f32", .wg = elem_wg },
    .{ .name = "l2norm_rows", .wg = row_wg, .kind = .rows },
    .{ .name = "l2norm_rows_g", .wg = row_wg, .kind = .rows },
    .{ .name = "rmsnorm", .wg = row_wg, .kind = .rows },
    .{ .name = "group_rmsnorm", .wg = row_wg, .kind = .rows },
    .{ .name = "rms_mod", .wg = row_wg, .kind = .rows },
    .{ .name = "rms_mod_h16", .wg = row_wg, .kind = .rows },
    .{ .name = "layernorm", .wg = row_wg, .kind = .rows },
    .{ .name = "layernorm_h16", .wg = row_wg, .kind = .rows },
    .{ .name = "ln_mod", .wg = row_wg, .kind = .rows },
    .{ .name = "gn_stats", .wg = row_wg, .kind = .rows },
    .{ .name = "gn_stats_h16", .wg = row_wg, .kind = .rows },
    .{ .name = "rowmax_i8", .wg = row_wg, .kind = .rows },
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
