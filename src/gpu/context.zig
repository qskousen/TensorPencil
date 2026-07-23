//! Vulkan compute context: loads libvulkan at runtime (std.DynLib — no C
//! linkage), owns one compute queue on the best available GPU, and executes
//! the embedded SPIR-V matmul kernels.
//!
//! v1 execution model is synchronous: activations travel over mapped
//! host-visible buffers each call, weights are uploaded once into
//! device-local memory and cached by their host pointer. Good enough to
//! offload DiT-sized GEMMs; keeping activations resident on the GPU across a
//! whole forward pass is the planned next step.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
pub const vk = @import("vk.zig");
const spv = @import("spv.zig");
const coopmat = @import("coopmat.zig");
const convrot = @import("tp_ops").convrot;
const mem_tag = @import("mem_tag.zig");
const sample = @import("tp_core").sample;
pub const MemTag = mem_tag.MemTag;

const matmul_f8_spv = @embedFile("matmul_f8_spv");
const matmul_f32_spv = @embedFile("matmul_f32_spv");
const transpose_spv = @embedFile("transpose_spv");
const eltwise_spv = @embedFile("eltwise_spv");
const attn_batched_spv = @embedFile("attn_batched_spv");

/// Push constants for the standalone `attn_batched` kernel (matches the Push
/// struct in kernels/attn_batched.zig: u0=total, u1=n_heads, u2=n_kv, u3=hd,
/// f0=scale). Distinct layout from EltPush (f0 at offset 16, not 24).
const AttnBatchedPush = extern struct {
    u0: u32 = 0,
    u1: u32 = 0,
    u2: u32 = 0,
    u3: u32 = 0,
    f0: f32 = 0,
};

/// Probe hook: when true, `Context.init` logs every cooperative-matrix config
/// the device advertises (component types + M/N/K + scope). Used by
/// `TensorPencil gpu-test` to survey int8 tensor-core availability.
pub var dump_coop_configs: bool = false;

/// Short name for a coopmat component type; panic-safe for values outside the
/// known set (drivers may advertise types this build doesn't enumerate).
fn componentName(t: vk.ComponentTypeKHR) []const u8 {
    return switch (t) {
        .float16 => "f16",
        .float32 => "f32",
        .float64 => "f64",
        .sint8 => "s8",
        .sint16 => "s16",
        .sint32 => "s32",
        .sint64 => "s64",
        .uint8 => "u8",
        .uint16 => "u16",
        .uint32 => "u32",
        .uint64 => "u64",
        .bfloat16 => "bf16",
        else => "?",
    };
}

const wg_x = 16; // threads per workgroup, column direction
const wg_y = 16; // threads per workgroup, token direction
// Transpose kernel workgroup shape (dispatch math in weightBuffer).
const tr_wg_x = 64;
const tr_wg_y = 4;
// Per-thread register tile; must match src/gpu/kernels/common.zig (tm/tn).
const tile_m = 8;
const tile_n = 8;

/// Dense-GEMV weight storage dtype. The integer values are the `u4` dtype flag
/// the gemv_partial / gemv_partial4 kernels branch on; `esize()` is the
/// per-element byte count weightBuffer transposes with (and picks the matching
/// transpose pipeline: 1→f8, 2→bf16, 4→f32).
pub const WCode = enum(u32) {
    f32 = 0,
    f8 = 1,
    bf16 = 2,

    pub fn esize(self: WCode) u64 {
        return switch (self) {
            .f8 => 1,
            .bf16 => 2,
            .f32 => 4,
        };
    }
};

const Push = extern struct {
    m: u32,
    rows: u32,
    cols: u32,
    w_stride: u32,
    has_bias: u32,
    scale: f32,
};

/// GPU top-k selection (opTopK): `topk_lanes` lanes each keep their `topk_m`
/// highest candidates, so the host does exact top-k over topk_lanes*topk_m
/// downloaded pairs. `topk_m` MUST match kernels/eltwise.zig. Exact in practice
/// for the sampler's k (<= 512): far more slots than any lane can plausibly own.
pub const topk_m = 8;
pub const topk_lanes = 1024;

/// Eltwise/attention kernels and their workgroup shapes (kernels/eltwise.zig).
/// KV storage format of the LLM decode-attention / KV store ops (mirrors
/// llm.kv_cache.KvDtype). q8_0 is the ggml 34-byte block (f16 scale + 32 i8).
pub const KvFmt = enum { f32, f16, q8_0 };

pub const Elt = enum(usize) { rmsnorm, rms_partial, rms_combine, rms_apply_mod, rms_apply_mod_h16, modulate, gated_add, add, silu_mul, sigmoid_mul, silu_mul_h16, sigmoid_mul_h16, rope_inter, attention, gather_kmajor, gather_kmajor_h16, attn_scores, softmax_partial, softmax_combine, softmax_rows, attn_out, f32_to_h16, f32_to_h16_pad, vae_norm, im2col, bias_compact, qknorm_rope16, gather_kmajor16, silu_mul16, sigmoid_mul_g16, gated_add16, rope_half, copy, rotate, rotate_fwht, rowmax_i8, rowscale_i8, quantize_i8, scale_i32, scale_concat, qknorm_rope_f32, rms_apply_w, attn_dsplit, attn_dmerge, gemv_partial, gemv_combine, gemv_partial4, gemv_combine4, gemv_q8_0, gemv_q4_k, gemv_q5_k, gemv_q6_k, l2norm_rows, deinterleave2, gdn_gates, gdn_conv_step, gdn_delta_step, rope_qwen35, attn_decode_q35, attn_dsplit_gemma, gemv_q6_k_t, gemv_q8_0_t, gemv_q4_k_t, gemv_q5_k_t, gelu_mul, gelu, layernorm, attn_full, f32_to_bf16_pad, relu, add_relu, argmax_reduce, argmax_final, topk_reduce, attn_dsplit_gemma_f16, kv_store_f16, penalize, attn_dsplit_gemma_q8, kv_store_q8_0 };
const elt_entry_sizes = [_]EntrySize{
    .{ .name = "rmsnorm", .x = 64, .y = 1 },
    .{ .name = "rms_partial", .x = 256, .y = 1 },
    .{ .name = "rms_combine", .x = 256, .y = 1 },
    .{ .name = "rms_apply_mod", .x = 256, .y = 1 },
    .{ .name = "rms_apply_mod_h16", .x = 256, .y = 1 },
    .{ .name = "modulate", .x = 256, .y = 1 },
    .{ .name = "gated_add", .x = 256, .y = 1 },
    .{ .name = "add", .x = 256, .y = 1 },
    .{ .name = "silu_mul", .x = 256, .y = 1 },
    .{ .name = "sigmoid_mul", .x = 256, .y = 1 },
    .{ .name = "silu_mul_h16", .x = 256, .y = 1 },
    .{ .name = "sigmoid_mul_h16", .x = 256, .y = 1 },
    .{ .name = "rope_inter", .x = 256, .y = 1 },
    .{ .name = "attention", .x = 4, .y = 64 },
    .{ .name = "gather_kmajor", .x = 256, .y = 1 },
    .{ .name = "gather_kmajor_h16", .x = 256, .y = 1 },
    .{ .name = "attn_scores", .x = 16, .y = 16 },
    .{ .name = "softmax_partial", .x = 256, .y = 1 },
    .{ .name = "softmax_combine", .x = 256, .y = 1 },
    .{ .name = "softmax_rows", .x = 64, .y = 1 },
    .{ .name = "attn_out", .x = 16, .y = 16 },
    .{ .name = "f32_to_h16", .x = 256, .y = 1 },
    .{ .name = "f32_to_h16_pad", .x = 256, .y = 1 },
    .{ .name = "vae_norm", .x = 64, .y = 1 },
    .{ .name = "im2col", .x = 256, .y = 1 },
    .{ .name = "bias_compact", .x = 256, .y = 1 },
    .{ .name = "qknorm_rope16", .x = 64, .y = 1 },
    .{ .name = "gather_kmajor16", .x = 256, .y = 1 },
    .{ .name = "silu_mul16", .x = 256, .y = 1 },
    .{ .name = "sigmoid_mul_g16", .x = 256, .y = 1 },
    .{ .name = "gated_add16", .x = 256, .y = 1 },
    .{ .name = "rope_half", .x = 256, .y = 1 },
    .{ .name = "copy", .x = 256, .y = 1 },
    .{ .name = "rotate", .x = 256, .y = 1 },
    .{ .name = "rotate_fwht", .x = 64, .y = 1 },
    .{ .name = "rowmax_i8", .x = 64, .y = 1 },
    .{ .name = "rowscale_i8", .x = 64, .y = 1 },
    .{ .name = "quantize_i8", .x = 256, .y = 1 },
    .{ .name = "scale_i32", .x = 256, .y = 1 },
    .{ .name = "scale_concat", .x = 256, .y = 1 },
    .{ .name = "qknorm_rope_f32", .x = 64, .y = 1 },
    .{ .name = "rms_apply_w", .x = 256, .y = 1 },
    .{ .name = "attn_dsplit", .x = 256, .y = 1 },
    .{ .name = "attn_dmerge", .x = 256, .y = 1 },
    .{ .name = "gemv_partial", .x = 256, .y = 1 },
    .{ .name = "gemv_combine", .x = 256, .y = 1 },
    .{ .name = "gemv_partial4", .x = 256, .y = 1 },
    .{ .name = "gemv_combine4", .x = 256, .y = 1 },
    .{ .name = "gemv_q8_0", .x = 256, .y = 1 },
    .{ .name = "gemv_q4_k", .x = 256, .y = 1 },
    .{ .name = "gemv_q5_k", .x = 256, .y = 1 },
    .{ .name = "gemv_q6_k", .x = 256, .y = 1 },
    .{ .name = "l2norm_rows", .x = 64, .y = 1 },
    .{ .name = "deinterleave2", .x = 256, .y = 1 },
    .{ .name = "gdn_gates", .x = 32, .y = 1 },
    .{ .name = "gdn_conv_step", .x = 256, .y = 1 },
    .{ .name = "gdn_delta_step", .x = 16, .y = 1 },
    .{ .name = "rope_qwen35", .x = 256, .y = 1 },
    .{ .name = "attn_decode_q35", .x = 16, .y = 1 },
    .{ .name = "attn_dsplit_gemma", .x = 256, .y = 1 },
    .{ .name = "gemv_q6_k_t", .x = 256, .y = 1 },
    .{ .name = "gemv_q8_0_t", .x = 256, .y = 1 },
    .{ .name = "gemv_q4_k_t", .x = 256, .y = 1 },
    .{ .name = "gemv_q5_k_t", .x = 256, .y = 1 },
    .{ .name = "gelu_mul", .x = 256, .y = 1 },
    .{ .name = "gelu", .x = 256, .y = 1 },
    .{ .name = "layernorm", .x = 64, .y = 1 },
    .{ .name = "attn_full", .x = 64, .y = 1 },
    .{ .name = "f32_to_bf16_pad", .x = 256, .y = 1 },
    .{ .name = "relu", .x = 256, .y = 1 },
    .{ .name = "add_relu", .x = 256, .y = 1 },
    .{ .name = "argmax_reduce", .x = 256, .y = 1 },
    .{ .name = "argmax_final", .x = 64, .y = 1 },
    .{ .name = "topk_reduce", .x = 256, .y = 1 },
    .{ .name = "attn_dsplit_gemma_f16", .x = 256, .y = 1 },
    .{ .name = "kv_store_f16", .x = 256, .y = 1 },
    .{ .name = "penalize", .x = 256, .y = 1 },
    .{ .name = "attn_dsplit_gemma_q8", .x = 256, .y = 1 },
    .{ .name = "kv_store_q8_0", .x = 256, .y = 1 },
};

/// Push block shared by all eltwise entries; meaning per entry (see kernels).
pub const EltPush = extern struct {
    u0: u32 = 0,
    u1: u32 = 0,
    u2: u32 = 0,
    u3: u32 = 0,
    u4: u32 = 0,
    u5: u32 = 0,
    f0: f32 = 0,
    f1: f32 = 0,
    // Appended after f0/f1 so existing kernels' field offsets are unchanged.
    u6: u32 = 0,
};

const TransposePush = extern struct {
    rows: u32,
    cols: u32,
    stride: u32,
    k_pad: u32 = 0, // bf16_to_f16_coopw only
};

pub const Error = error{
    VulkanUnavailable,
    NoSuitableDevice,
    VulkanFailed,
    /// Device allocation failed even after evicting every cached weight
    /// buffer — the working set genuinely doesn't fit.
    DeviceOutOfMemory,
    OutOfMemory,
    /// A GGUF block-quant dtype without a Vulkan GEMV kernel yet.
    UnsupportedDType,
} || spv.Error;

/// Patch a Zig-emitted kernel into strict-Vulkan shape (LocalSize per entry,
/// logical addressing, no workgroup ArrayStrides — see spv.zig) and create
/// the module. All listed entry points get the same workgroup size.
const EntrySize = struct { name: []const u8, x: u32, y: u32 };

fn createKernelModule(gpa: std.mem.Allocator, d: *const Dispatch, device: vk.Device, code: []const u8, entries: []const EntrySize, out: *vk.ShaderModule) Error!void {
    var current = try gpa.alignedAlloc(u8, .of(u32), code.len);
    @memcpy(current, code);
    for (entries) |entry| {
        const sized = try spv.withLocalSize(gpa, current, entry.name, entry.x, entry.y, 1);
        gpa.free(current);
        current = sized;
    }
    defer gpa.free(current);
    const logical = try spv.withLogicalAddressing(gpa, current);
    defer gpa.free(logical);
    const stripped = try spv.stripWorkgroupStrides(gpa, logical);
    defer gpa.free(stripped);
    const patched = try spv.dedupeDecorations(gpa, stripped);
    defer gpa.free(patched);
    try check(d.CreateShaderModule(device, &.{
        .code_size = patched.len,
        .p_code = @ptrCast(@alignCast(patched.ptr)),
    }, null, out));
}

fn openVulkanLib() ?std.DynLib {
    const candidates = [_][]const u8{
        "libvulkan.so.1",
        "/usr/lib/x86_64-linux-gnu/libvulkan.so.1",
        "/usr/lib64/libvulkan.so.1",
        "/usr/lib/libvulkan.so.1",
    };
    for (candidates) |path| {
        if (std.DynLib.open(path)) |lib| return lib else |_| {}
    }
    return null;
}

fn check(r: vk.Result) Error!void {
    if (r != .success) {
        std.log.err("vulkan call failed: {t}", .{r});
        return error.VulkanFailed;
    }
}

/// Create a compute pipeline for a cooperative-matrix kernel, pinning the
/// subgroup size when the device needs it. The coopmat SPIR-V assumes
/// warp == subgroup == 32 lanes (every coop kernel declares `LocalSize 32,
/// NWARPS, 1`). On a device whose native subgroup size isn't 32 — AMD RADV
/// runs wave64 — the cooperative-matrix operands span the wrong lanes and the
/// GEMM silently produces all zeros. `coop_sg` is 32 there (and the pipeline
/// gets a RequiredSubgroupSize create-info via VK_EXT_subgroup_size_control,
/// core 1.3); it is null on wave32 devices (e.g. NVIDIA), leaving the created
/// pipeline byte-identical to the un-pinned path.
fn createCoopPipe(
    d: *const Dispatch,
    device: vk.Device,
    module: vk.ShaderModule,
    layout: vk.PipelineLayout,
    coop_sg: ?u32,
    out: *vk.Pipeline,
) Error!void {
    var sgs: vk.PipelineShaderStageRequiredSubgroupSizeCreateInfo =
        .{ .required_subgroup_size = coop_sg orelse 0 };
    const info: vk.ComputePipelineCreateInfo = .{
        .stage = .{
            .module = module,
            .p_name = "main",
            .p_next = if (coop_sg != null) &sgs else null,
        },
        .layout = layout,
    };
    try check(d.CreateComputePipelines(device, .null_handle, 1, @ptrCast(&info), null, @ptrCast(out)));
}

/// Function table. Field names match the command names minus the `vk` prefix;
/// loading resolves "vk" ++ field name through vkGetInstanceProcAddr.
const Dispatch = struct {
    DestroyInstance: vk.PfnDestroyInstance,
    EnumeratePhysicalDevices: vk.PfnEnumeratePhysicalDevices,
    GetPhysicalDeviceProperties: vk.PfnGetPhysicalDeviceProperties,
    GetPhysicalDeviceQueueFamilyProperties: vk.PfnGetPhysicalDeviceQueueFamilyProperties,
    GetPhysicalDeviceMemoryProperties: vk.PfnGetPhysicalDeviceMemoryProperties,
    GetPhysicalDeviceMemoryProperties2: vk.PfnGetPhysicalDeviceMemoryProperties2,
    EnumerateDeviceExtensionProperties: vk.PfnEnumerateDeviceExtensionProperties,
    CreateDevice: vk.PfnCreateDevice,
    DestroyDevice: vk.PfnDestroyDevice,
    GetDeviceQueue: vk.PfnGetDeviceQueue,
    CreateBuffer: vk.PfnCreateBuffer,
    DestroyBuffer: vk.PfnDestroyBuffer,
    GetBufferMemoryRequirements: vk.PfnGetBufferMemoryRequirements,
    AllocateMemory: vk.PfnAllocateMemory,
    FreeMemory: vk.PfnFreeMemory,
    BindBufferMemory: vk.PfnBindBufferMemory,
    MapMemory: vk.PfnMapMemory,
    UnmapMemory: vk.PfnUnmapMemory,
    CreateShaderModule: vk.PfnCreateShaderModule,
    DestroyShaderModule: vk.PfnDestroyShaderModule,
    CreateDescriptorSetLayout: vk.PfnCreateDescriptorSetLayout,
    DestroyDescriptorSetLayout: vk.PfnDestroyDescriptorSetLayout,
    CreatePipelineLayout: vk.PfnCreatePipelineLayout,
    DestroyPipelineLayout: vk.PfnDestroyPipelineLayout,
    CreateComputePipelines: vk.PfnCreateComputePipelines,
    DestroyPipeline: vk.PfnDestroyPipeline,
    CreateDescriptorPool: vk.PfnCreateDescriptorPool,
    DestroyDescriptorPool: vk.PfnDestroyDescriptorPool,
    AllocateDescriptorSets: vk.PfnAllocateDescriptorSets,
    UpdateDescriptorSets: vk.PfnUpdateDescriptorSets,
    CreateCommandPool: vk.PfnCreateCommandPool,
    DestroyCommandPool: vk.PfnDestroyCommandPool,
    AllocateCommandBuffers: vk.PfnAllocateCommandBuffers,
    BeginCommandBuffer: vk.PfnBeginCommandBuffer,
    EndCommandBuffer: vk.PfnEndCommandBuffer,
    ResetCommandBuffer: vk.PfnResetCommandBuffer,
    CmdBindPipeline: vk.PfnCmdBindPipeline,
    CmdBindDescriptorSets: vk.PfnCmdBindDescriptorSets,
    CmdDispatch: vk.PfnCmdDispatch,
    CmdPipelineBarrier: vk.PfnCmdPipelineBarrier,
    CmdCopyBuffer: vk.PfnCmdCopyBuffer,
    CmdPushConstants: vk.PfnCmdPushConstants,
    QueueSubmit: vk.PfnQueueSubmit,
    QueueWaitIdle: vk.PfnQueueWaitIdle,
    DeviceWaitIdle: vk.PfnDeviceWaitIdle,
    CreateFence: vk.PfnCreateFence,
    DestroyFence: vk.PfnDestroyFence,
    WaitForFences: vk.PfnWaitForFences,
    ResetFences: vk.PfnResetFences,
};

const HostBuffer = struct {
    buf: vk.Buffer = .null_handle,
    mem: vk.DeviceMemory = .null_handle,
    size: u64 = 0,
    mapped: ?[*]u8 = null,
};

pub const DeviceBuffer = struct {
    buf: vk.Buffer,
    mem: vk.DeviceMemory,
    size: u64,
    /// Component attribution for the VRAM meter (set from the context's current
    /// tag at createBuffer; read at freeDeviceBuffer). `.other` unless tagging is
    /// active (diffusion backend only).
    tag: MemTag = .other,
};

/// Cached device weight buffer + LRU stamp. Since the DiT walks its blocks
/// in the same order every step, LRU eviction under memory pressure turns
/// the cache into sequential block streaming (evicted weights re-upload on
/// next use).
const WeightEntry = struct {
    db: DeviceBuffer,
    last_use: u64,
    /// Pinned entries (first-touch, up to pin_budget) are immune to eviction.
    pinned: bool = false,
};

pub const Context = struct {
    gpa: std.mem.Allocator,
    lib: std.DynLib,
    d: Dispatch,
    instance: vk.Instance,
    device: vk.Device,
    queue: vk.Queue,
    queue_family: u32,
    mem_props: vk.PhysicalDeviceMemoryProperties,
    /// Kept for live memory-budget queries (VK_EXT_memory_budget).
    phys: vk.PhysicalDevice,
    has_memory_budget: bool = false,
    /// Device-local heap index backing our allocations.
    device_heap: u32 = 0,
    /// Bytes we hold in buffers created through createBuffer.
    device_used: u64 = 0,
    /// Per-component device-byte attribution (MEASURED, for the GUI VRAM meter);
    /// only maintained when `track_tags` is set (diffusion backend). `mem_tag` is
    /// applied to new device buffers; the pipeline sets it around each phase.
    track_tags: bool = false,
    mem_tag: MemTag = .other,
    mem_tag_used: [MemTag.count]u64 = @splat(0),
    /// Monotonic stamp for the weight cache's LRU order.
    use_counter: u64 = 0,
    /// Test hook: when nonzero, caps the budget headroom calculation so
    /// weight streaming can be forced without exhausting real VRAM.
    budget_override: u64 = 0,
    /// First-touch weight pinning: newly cached weights are pinned (immune to
    /// eviction) until their total reaches this cap; later weights stream.
    /// For a fixed repeating walk (LLM decode) this turns the LRU cliff —
    /// where any cap below full residency re-uploads EVERYTHING — into cost
    /// proportional to the streamed fraction. 0 = off. Must stay off for the
    /// diffusion pipeline: first-touch would pin the single-use text encoder
    /// and stream the whole DiT.
    pin_budget: u64 = 0,
    /// Bytes currently claimed against pin_budget (device sizes, padded).
    pinned_bytes: u64 = 0,
    device_name: [64]u8,
    device_name_len: usize,
    /// Native subgroup (wave) size the device reported (0 = not queried).
    subgroup_size: u32 = 0,
    /// Subgroup size pinned on the coop pipelines (0 = none needed/applied).
    coop_pinned_subgroup: u32 = 0,
    /// Coop was advertised but disabled because the kernels' subgroup width
    /// couldn't be pinned on this device (fell back to register/f32).
    coop_disabled_by_subgroup: bool = false,
    /// f16*f16->f32 subgroup cooperative-matrix shape (0 = unsupported).
    coop_m: u32 = 0,
    coop_n: u32 = 0,
    coop_k: u32 = 0,
    /// sint8*sint8->sint32 subgroup cooperative-matrix shape (0 = unsupported).
    /// Probed for the int8 (convrot) tensor-core GEMM path.
    coop_i8_m: u32 = 0,
    coop_i8_n: u32 = 0,
    coop_i8_k: u32 = 0,

    cmd_pool: vk.CommandPool,
    cmd: vk.CommandBuffer,
    /// Second command buffer for immediate work (weight upload/transpose,
    /// transfers) issued while a batch is being recorded on `cmd`.
    cmd_now: vk.CommandBuffer,
    fence: vk.Fence,

    /// Batch recording: while active, op* dispatches are recorded into `cmd`
    /// — one descriptor set per op from the preallocated ring, a global
    /// compute->compute barrier between dispatches — and submitted once at
    /// endBatch, so the queue stays fed and per-op fence waits disappear.
    batch_pool: vk.DescriptorPool,
    batch_sets: []vk.DescriptorSet,
    batching: bool = false,
    batch_n: usize = 0,
    /// Countdown for `independent`: while > 1, opEnd skips the inter-dispatch
    /// barrier so the group's dispatches may overlap on the device.
    indep_remaining: usize = 0,

    dsl: vk.DescriptorSetLayout,
    pipeline_layout: vk.PipelineLayout,
    shader_f8: vk.ShaderModule,
    shader_f32: vk.ShaderModule,
    pipe_f8: vk.Pipeline,
    pipe_f32: vk.Pipeline,
    dsl_tr: vk.DescriptorSetLayout,
    pipeline_layout_tr: vk.PipelineLayout,
    shader_tr: vk.ShaderModule,
    pipe_tr_f8: vk.Pipeline,
    pipe_tr_f32: vk.Pipeline,
    pipe_tr_bf16: vk.Pipeline, // bf16 weight -> bf16 k-major dense-GEMV layout (2 cols/u32)
    pipe_tr_bf16w: vk.Pipeline, // bf16 weight -> f16 k-major coop layout (GPU convert, fallback)
    pipe_tr_bf16raw: vk.Pipeline, // bf16 weight -> bf16 k-major coop layout (native, no convert)
    pipeline_layout_e: vk.PipelineLayout,
    shader_e: vk.ShaderModule,
    pipes_e: [elt_entry_sizes.len]vk.Pipeline,
    // Standalone block-diagonal batched-attention kernel (its own module +
    // 5-buffer set: a=q,b=k,c=v,d=out,e=bounds). Separate from the eltwise
    // module, which is at the SPIR-V backend's per-module entry-point limit.
    shader_ab: vk.ShaderModule,
    dsl_ab: vk.DescriptorSetLayout,
    pipeline_layout_ab: vk.PipelineLayout,
    pipe_ab: vk.Pipeline,
    pool_ab: vk.DescriptorPool,
    desc_set_ab: vk.DescriptorSet,
    shader_coop: vk.ShaderModule = .null_handle,
    pipe_coop: vk.Pipeline = .null_handle,
    /// fp8 coop GEMM with an f16 C store (exact under f16 accumulators;
    /// present iff pipe_coop is and coop_acc_h16 is on).
    shader_coop_c16: vk.ShaderModule = .null_handle,
    pipe_coop_c16: vk.Pipeline = .null_handle,
    /// f16-weight coop GEMM (VAE convs; present iff pipe_coop is).
    shader_coop_f16w: vk.ShaderModule = .null_handle,
    pipe_coop_f16w: vk.Pipeline = .null_handle,
    shader_coop_bf16w: vk.ShaderModule = .null_handle,
    pipe_coop_bf16w: vk.Pipeline = .null_handle, // native bf16 A/B coop GEMM (null = use f16 fallback)
    /// int8 tensor-core GEMM s8*s8->s32 (present iff coop_i8_m != 0).
    shader_coop_i8: vk.ShaderModule = .null_handle,
    pipe_coop_i8: vk.Pipeline = .null_handle,
    /// Shared-memory-staged int8 GEMM (coopmat.buildGemmSharedI8): faster for
    /// large 128-multiple shapes; the register-tiled pipe stays the fallback.
    shader_coop_i8_sh: vk.ShaderModule = .null_handle,
    pipe_coop_i8_sh: vk.Pipeline = .null_handle,
    /// Stage A: shared int8 GEMM with the act*weight rescale fused into the
    /// C-store (outputs f32 directly, kills the s32 acc buffer + scale_i32 pass).
    /// Binding 3 = scale buffer [act(m_pad) | weight(rows)] f32.
    shader_coop_i8_fs: vk.ShaderModule = .null_handle,
    pipe_coop_i8_fs: vk.Pipeline = .null_handle,
    /// f16-C variant of the fused GEMM (stores f16 directly) so int8 attention
    /// GEMM outputs feed the fused f16 eltwise chain (att16/mlp16), no convert.
    shader_coop_i8_fs16: vk.ShaderModule = .null_handle,
    pipe_coop_i8_fs16: vk.Pipeline = .null_handle,
    /// Stage B: fused int8 prep (rotate FWHT + rowmax + quantize) in one
    /// f16-shared-memory kernel. Two builds: cols=6144 (qkv/wo/mlp-gu) and
    /// cols=16384 (mlp.down). Replaces the 3-pass chain + its xr round-trip.
    shader_i8_prep6144: vk.ShaderModule = .null_handle,
    pipe_i8_prep6144: vk.Pipeline = .null_handle,
    shader_i8_prep16384: vk.ShaderModule = .null_handle,
    pipe_i8_prep16384: vk.Pipeline = .null_handle,
    /// Tensor-core attention GEMMs (present iff pipe_coop is).
    shader_scores: vk.ShaderModule = .null_handle,
    pipe_scores: vk.Pipeline = .null_handle,
    shader_scores_vae: vk.ShaderModule = .null_handle,
    pipe_scores_vae: vk.Pipeline = .null_handle,
    shader_attn_out: vk.ShaderModule = .null_handle,
    pipe_attn_out: vk.Pipeline = .null_handle,
    shader_flash_md: vk.ShaderModule = .null_handle,
    pipe_flash_md: vk.Pipeline = .null_handle,
    shader_flash_out: vk.ShaderModule = .null_handle,
    pipe_flash_out: vk.Pipeline = .null_handle,
    desc_pool: vk.DescriptorPool,
    desc_set: vk.DescriptorSet,
    desc_set_tr: vk.DescriptorSet,

    /// Host-visible staging: activations in/out, bias, weight upload.
    x_buf: HostBuffer = .{},
    y_buf: HostBuffer = .{},
    bias_buf: HostBuffer = .{},
    staging: HostBuffer = .{},
    /// Device-local activation buffers (the shader never reads across PCIe).
    x_dev: DeviceBuffer = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 },
    y_dev: DeviceBuffer = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 },
    /// f16 activation scratch for the cooperative-matrix path.
    x_h16: DeviceBuffer = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 },
    /// Column-padded f32 output scratch for the f16-weight coop path.
    y_pad: DeviceBuffer = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 },
    /// Device-local scratch for raw (untransposed) weight uploads.
    raw_dev: DeviceBuffer = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 },
    /// int8-path scratch: rotated f32 activations, per-row act scale, packed
    /// int8 activations, s32 GEMM accumulator, and the resident Hadamard.
    i8_xr: DeviceBuffer = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 },
    i8_scale: DeviceBuffer = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 },
    i8_x: DeviceBuffer = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 },
    i8_acc: DeviceBuffer = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 },
    // Extra s32 accumulators so a group of int8 GEMMs sharing one prepped
    // activation (qkv/gate, mlp gate/up) can run overlapped instead of
    // serializing on a single acc buffer.
    i8_acc1: DeviceBuffer = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 },
    i8_acc2: DeviceBuffer = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 },
    i8_acc3: DeviceBuffer = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 },
    i8_hadamard: DeviceBuffer = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 },
    i8_partials: DeviceBuffer = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 },
    /// Stage A: assembled [act(m_pad) | weight(rows)] scale buffer for the
    /// fused-rescale GEMM (rebuilt per GEMM by scale_concat; batch-safe).
    i8_scalecat: DeviceBuffer = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 },
    /// Shape of the activation last prepped by opI8Prep (consumed by opI8Gemm).
    i8_m: usize = 0,
    i8_cols: usize = 0,
    i8_mpad: usize = 0,
    /// Tiny valid buffer bound to unused descriptor slots.
    dummy: DeviceBuffer = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 },
    /// Per-step penalty-entry wire for opPenalize (id + subtract f32 pairs).
    pen_wire: DeviceBuffer = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 },
    /// Small cached device uploads (biases, norm weights), keyed by host ptr.
    small_bufs: std.AutoHashMapUnmanaged(usize, DeviceBuffer) = .empty,
    /// Device-local transposed weight buffers, keyed by host weight pointer,
    /// LRU-evicted when the device memory budget runs out.
    weights: std.AutoHashMapUnmanaged(usize, WeightEntry) = .empty,

    pub fn init(gpa: std.mem.Allocator) Error!*Context {
        // Integration tests are gated behind `-Dintegration`: fail init in test
        // builds when it's off so the Vulkan tests self-skip (they `catch SkipZigTest`).
        if (builtin.is_test and !build_options.integration) return error.VulkanUnavailable;
        var lib = openVulkanLib() orelse return error.VulkanUnavailable;
        errdefer lib.close();
        const gipa = lib.lookup(vk.PfnGetInstanceProcAddr, "vkGetInstanceProcAddr") orelse
            return error.VulkanUnavailable;

        // Instance.
        var instance: vk.Instance = .null_handle;
        {
            const create: vk.PfnCreateInstance = @ptrCast(gipa(.null_handle, "vkCreateInstance") orelse
                return error.VulkanUnavailable);
            const app_info: vk.ApplicationInfo = .{
                .p_application_name = "TensorPencil",
                .application_version = 0,
                .p_engine_name = "TensorPencil",
                .engine_version = 0,
                .api_version = vk.API_VERSION_1_2,
            };
            try check(create(&.{ .p_application_info = &app_info }, null, &instance));
        }

        var d: Dispatch = undefined;
        inline for (@typeInfo(Dispatch).@"struct".fields) |field| {
            const pfn = gipa(instance, "vk" ++ field.name) orelse return error.VulkanUnavailable;
            @field(d, field.name) = @ptrCast(pfn);
        }
        errdefer d.DestroyInstance(instance, null);

        // Pick a physical device: prefer discrete, require a compute queue.
        var phys: vk.PhysicalDevice = .null_handle;
        var queue_family: u32 = 0;
        var props: vk.PhysicalDeviceProperties = undefined;
        {
            var count: u32 = 0;
            try check(d.EnumeratePhysicalDevices(instance, &count, null));
            if (count == 0) return error.NoSuitableDevice;
            const devices = try gpa.alloc(vk.PhysicalDevice, count);
            defer gpa.free(devices);
            try check(d.EnumeratePhysicalDevices(instance, &count, devices.ptr));

            var best_score: i32 = -1;
            for (devices[0..count]) |dev| {
                var p: vk.PhysicalDeviceProperties = undefined;
                d.GetPhysicalDeviceProperties(dev, &p);
                var qcount: u32 = 0;
                d.GetPhysicalDeviceQueueFamilyProperties(dev, &qcount, null);
                var qprops: [16]vk.QueueFamilyProperties = undefined;
                qcount = @min(qcount, 16);
                d.GetPhysicalDeviceQueueFamilyProperties(dev, &qcount, &qprops);
                const family: ?u32 = for (qprops[0..qcount], 0..) |qp, i| {
                    if (qp.queue_flags & vk.QueueFlagBits.compute != 0) break @intCast(i);
                } else null;
                if (family == null) continue;
                const score: i32 = if (p.device_type == .discrete_gpu) 2 else 1;
                if (score > best_score) {
                    best_score = score;
                    phys = dev;
                    queue_family = family.?;
                    props = p;
                }
            }
            if (phys == .null_handle) return error.NoSuitableDevice;
        }

        // Cooperative matrix support: pick an f16xf16->f32 subgroup config for
        // the current fp8/f16 GEMM path, and separately record an
        // sint8xsint8->sint32 subgroup config for the int8 (convrot) path.
        var coop_m: u32 = 0;
        var coop_n: u32 = 0;
        var coop_k: u32 = 0;
        var coop_i8_m: u32 = 0;
        var coop_i8_n: u32 = 0;
        var coop_i8_k: u32 = 0;
        // Native bf16 A/B -> f32 config (SPV_KHR_bfloat16), same 16x16x16 tile as
        // f16: lets the dense bf16 DiT compute in bf16 with no f16 round trip.
        var coop_bf16 = false;
        if (gipa(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR")) |pfn| {
            const get_props: vk.PfnGetPhysicalDeviceCooperativeMatrixPropertiesKHR = @ptrCast(pfn);
            var count: u32 = 0;
            if (get_props(phys, &count, null) == .success and count > 0) {
                const props_buf = gpa.alloc(vk.CooperativeMatrixPropertiesKHR, count) catch null;
                if (props_buf) |pb| {
                    defer gpa.free(pb);
                    for (pb) |*cp| cp.* = .{};
                    if (get_props(phys, &count, pb.ptr) == .success) {
                        for (pb[0..count]) |cp| {
                            if (dump_coop_configs) {
                                std.log.info("coopmat cfg: {d}x{d}x{d} A={s} B={s} C={s} R={s} scope={d} sat={d}", .{
                                    cp.m_size,                     cp.n_size,                cp.k_size,
                                    componentName(cp.a_type),      componentName(cp.b_type), componentName(cp.c_type),
                                    componentName(cp.result_type), @intFromEnum(cp.scope),   @intFromBool(cp.saturating_accumulation == vk.TRUE),
                                });
                            }
                            if (coop_m == 0 and cp.scope == .subgroup and cp.a_type == .float16 and cp.b_type == .float16 and
                                cp.c_type == .float32 and cp.result_type == .float32 and cp.m_size == 16)
                            {
                                coop_m = cp.m_size;
                                coop_n = cp.n_size;
                                coop_k = cp.k_size;
                            }
                            if (coop_i8_m == 0 and cp.scope == .subgroup and cp.a_type == .sint8 and cp.b_type == .sint8 and
                                cp.c_type == .sint32 and cp.result_type == .sint32)
                            {
                                coop_i8_m = cp.m_size;
                                coop_i8_n = cp.n_size;
                                coop_i8_k = cp.k_size;
                            }
                            if (cp.scope == .subgroup and cp.a_type == .bfloat16 and cp.b_type == .bfloat16 and
                                cp.c_type == .float32 and cp.result_type == .float32 and cp.m_size == 16 and cp.n_size == 16 and cp.k_size == 16)
                            {
                                coop_bf16 = true;
                            }
                        }
                    }
                }
            }
        }
        if (dump_coop_configs) {
            std.log.info("selected f16 coop: {d}x{d}x{d}; int8 coop: {d}x{d}x{d}", .{
                coop_m, coop_n, coop_k, coop_i8_m, coop_i8_n, coop_i8_k,
            });
        }

        // The coop kernels are authored for a fixed subgroup width
        // (coopmat.subgroup_lanes = 32; see there). We query the device's
        // *native* subgroup size at runtime and adapt — nothing is hardcoded
        // per GPU; the only constant is the kernels' own width, and every
        // decision is driven by the queried native size and the device's
        // advertised [min,max] pinnable range:
        //   - native size already matches  -> coop runs as-is (NVIDIA wave32);
        //   - native differs but the width is pinnable -> pin it via
        //     VK_EXT_subgroup_size_control (AMD RADV wave64 -> pin 32);
        //   - native differs and the width is NOT pinnable -> disable coop
        //     (coop_m/coop_i8_m = 0) so the model falls back to the correct
        //     register-tiled / f32 path instead of feeding coopmat the wrong
        //     lanes (which returns zeros).
        const want_lanes = coopmat.subgroup_lanes;
        var coop_sg: ?u32 = null;
        var native_subgroup: u32 = 0;
        var coop_disabled_by_subgroup = false;
        if (coop_m != 0 or coop_i8_m != 0) {
            const getp2: ?vk.PfnGetPhysicalDeviceProperties2 =
                if (gipa(instance, "vkGetPhysicalDeviceProperties2")) |p|
                    @ptrCast(p)
                else if (gipa(instance, "vkGetPhysicalDeviceProperties2KHR")) |p|
                    @ptrCast(p)
                else
                    null;
            if (getp2) |getp| {
                var sg: vk.PhysicalDeviceSubgroupProperties = .{};
                var sgc: vk.PhysicalDeviceSubgroupSizeControlProperties = .{ .p_next = &sg };
                var props2: vk.PhysicalDeviceProperties2 = .{ .p_next = &sgc };
                getp(phys, &props2);
                native_subgroup = sg.subgroup_size;
                if (sg.subgroup_size != want_lanes) {
                    const pinnable = sgc.min_subgroup_size <= want_lanes and
                        sgc.max_subgroup_size >= want_lanes and
                        (sgc.required_subgroup_size_stages & vk.ShaderStage.compute) != 0;
                    if (pinnable) {
                        coop_sg = want_lanes;
                    } else {
                        // Can't make coopmat correct on this device; fall back.
                        coop_m = 0;
                        coop_n = 0;
                        coop_k = 0;
                        coop_i8_m = 0;
                        coop_i8_n = 0;
                        coop_i8_k = 0;
                        coop_disabled_by_subgroup = true;
                    }
                }
            }
            if (dump_coop_configs) {
                std.log.info("coopmat: native subgroup {d}, kernels want {d}; {s}", .{
                    native_subgroup,
                    want_lanes,
                    if (coop_disabled_by_subgroup)
                        "not pinnable -> coop disabled, register/f32 fallback"
                    else if (coop_sg != null)
                        "pinned to kernel width"
                    else
                        "native matches, no pin needed",
                });
            }
        }

        // VK_EXT_memory_budget: proactive VRAM budgeting for weight
        // streaming (accounts for other processes; heap-size fallback
        // otherwise).
        var has_memory_budget = false;
        {
            var count: u32 = 0;
            if (d.EnumerateDeviceExtensionProperties(phys, null, &count, null) == .success and count > 0) {
                if (gpa.alloc(vk.ExtensionProperties, count)) |exts| {
                    defer gpa.free(exts);
                    if (d.EnumerateDeviceExtensionProperties(phys, null, &count, exts.ptr) == .success) {
                        for (exts[0..count]) |e| {
                            if (std.mem.eql(u8, std.mem.sliceTo(&e.extension_name, 0), "VK_EXT_memory_budget")) {
                                has_memory_budget = true;
                            }
                        }
                    }
                } else |_| {}
            }
        }

        // Logical device with the features the Zig SPIR-V backend requires.
        var device: vk.Device = .null_handle;
        {
            var coop_features: vk.PhysicalDeviceCooperativeMatrixFeaturesKHR = .{
                .cooperative_matrix = vk.TRUE,
            };
            // Enable subgroup-size control so the coop pipelines can pin size 32
            // (chained after coop_features, which is always present when
            // coop_sg is set). Left disabled on wave32 devices.
            var sgc_features: vk.PhysicalDeviceSubgroupSizeControlFeatures =
                .{ .subgroup_size_control = vk.TRUE };
            if (coop_sg != null) coop_features.p_next = &sgc_features;
            // Native bf16 cooperative matrix: chain the bf16 feature after the
            // subgroup-size one (or coop_features if no sg control) when the
            // device advertised a bf16 coop config.
            var bf16_features: vk.PhysicalDeviceShaderBfloat16FeaturesKHR = .{
                .shader_bfloat16_type = vk.TRUE,
                .shader_bfloat16_cooperative_matrix = vk.TRUE,
            };
            if (coop_bf16) {
                if (coop_sg != null) sgc_features.p_next = &bf16_features else coop_features.p_next = &bf16_features;
            }
            var features12: vk.PhysicalDeviceVulkan12Features = .{
                .shader_int8 = vk.TRUE,
                .storage_buffer_8bit_access = vk.TRUE,
                .buffer_device_address = vk.TRUE,
                .shader_float16 = vk.TRUE,
                .vulkan_memory_model = vk.TRUE,
                .p_next = if (coop_m != 0) &coop_features else null,
            };
            var features: vk.PhysicalDeviceFeatures = .{};
            features.shader_int64 = vk.TRUE;
            features.shader_int16 = vk.TRUE;
            const priority: f32 = 1.0;
            const queue_info: vk.DeviceQueueCreateInfo = .{
                .queue_family_index = queue_family,
                .queue_count = 1,
                .p_queue_priorities = @ptrCast(&priority),
            };
            var exts: [3][*:0]const u8 = undefined;
            var ext_n: u32 = 0;
            if (coop_m != 0) {
                exts[ext_n] = "VK_KHR_cooperative_matrix";
                ext_n += 1;
            }
            if (has_memory_budget) {
                exts[ext_n] = "VK_EXT_memory_budget";
                ext_n += 1;
            }
            if (coop_bf16) {
                exts[ext_n] = "VK_KHR_shader_bfloat16";
                ext_n += 1;
            }
            try check(d.CreateDevice(phys, &.{
                .p_next = &features12,
                .queue_create_info_count = 1,
                .p_queue_create_infos = @ptrCast(&queue_info),
                .enabled_extension_count = ext_n,
                .pp_enabled_extension_names = if (ext_n != 0) &exts else null,
                .p_enabled_features = &features,
            }, null, &device));
        }
        errdefer d.DestroyDevice(device, null);

        var queue: vk.Queue = .null_handle;
        d.GetDeviceQueue(device, queue_family, 0, &queue);

        var mem_props: vk.PhysicalDeviceMemoryProperties = undefined;
        d.GetPhysicalDeviceMemoryProperties(phys, &mem_props);
        var device_heap: u32 = 0;
        for (0..mem_props.memory_heap_count) |i| {
            if (mem_props.memory_heaps[i].flags & vk.MemoryHeapFlagBits.device_local != 0) {
                device_heap = @intCast(i);
                break;
            }
        }

        // Command pool + primary buffer + fence.
        var cmd_pool: vk.CommandPool = .null_handle;
        try check(d.CreateCommandPool(device, &.{
            .flags = vk.CommandPoolCreate.reset_command_buffer,
            .queue_family_index = queue_family,
        }, null, &cmd_pool));
        errdefer d.DestroyCommandPool(device, cmd_pool, null);
        var cmds: [2]vk.CommandBuffer = .{ .null_handle, .null_handle };
        try check(d.AllocateCommandBuffers(device, &.{
            .command_pool = cmd_pool,
            .level = .primary,
            .command_buffer_count = 2,
        }, &cmds));
        const cmd = cmds[0];
        const cmd_now = cmds[1];
        var fence: vk.Fence = .null_handle;
        try check(d.CreateFence(device, &.{}, null, &fence));
        errdefer d.DestroyFence(device, fence, null);

        // Shader modules (with LocalSize patched in).
        var shader_f8: vk.ShaderModule = .null_handle;
        try createKernelModule(gpa, &d, device, matmul_f8_spv, &.{.{ .name = "matmul_f8", .x = wg_x, .y = wg_y }}, &shader_f8);
        errdefer d.DestroyShaderModule(device, shader_f8, null);
        var shader_f32: vk.ShaderModule = .null_handle;
        try createKernelModule(gpa, &d, device, matmul_f32_spv, &.{.{ .name = "matmul_f32", .x = wg_x, .y = wg_y }}, &shader_f32);
        errdefer d.DestroyShaderModule(device, shader_f32, null);
        var shader_tr: vk.ShaderModule = .null_handle;
        try createKernelModule(gpa, &d, device, transpose_spv, &.{
            .{ .name = "transpose_f8", .x = tr_wg_x, .y = tr_wg_y },
            .{ .name = "transpose_f32", .x = tr_wg_x, .y = tr_wg_y },
            .{ .name = "transpose_bf16", .x = tr_wg_x, .y = tr_wg_y },
            .{ .name = "bf16_to_f16_coopw", .x = tr_wg_x, .y = tr_wg_y },
            .{ .name = "bf16_coopw", .x = tr_wg_x, .y = tr_wg_y },
        }, &shader_tr);
        var shader_e: vk.ShaderModule = .null_handle;
        try createKernelModule(gpa, &d, device, eltwise_spv, &elt_entry_sizes, &shader_e);
        errdefer d.DestroyShaderModule(device, shader_e, null);
        errdefer d.DestroyShaderModule(device, shader_tr, null);

        var dsl: vk.DescriptorSetLayout = .null_handle;
        {
            var bindings: [4]vk.DescriptorSetLayoutBinding = undefined;
            for (&bindings, 0..) |*b, i| {
                b.* = .{
                    .binding = @intCast(i),
                    .descriptor_type = .storage_buffer,
                    .descriptor_count = 1,
                    .stage_flags = vk.ShaderStage.compute,
                    .p_immutable_samplers = null,
                };
            }
            try check(d.CreateDescriptorSetLayout(device, &.{
                .binding_count = bindings.len,
                .p_bindings = &bindings,
            }, null, &dsl));
        }
        errdefer d.DestroyDescriptorSetLayout(device, dsl, null);

        var pipeline_layout: vk.PipelineLayout = .null_handle;
        {
            const push_range: vk.PushConstantRange = .{
                .stage_flags = vk.ShaderStage.compute,
                .offset = 0,
                .size = @sizeOf(Push),
            };
            try check(d.CreatePipelineLayout(device, &.{
                .set_layout_count = 1,
                .p_set_layouts = @ptrCast(&dsl),
                .push_constant_range_count = 1,
                .p_push_constant_ranges = @ptrCast(&push_range),
            }, null, &pipeline_layout));
        }
        errdefer d.DestroyPipelineLayout(device, pipeline_layout, null);

        var pipes: [2]vk.Pipeline = .{ .null_handle, .null_handle };
        {
            const infos = [2]vk.ComputePipelineCreateInfo{
                .{ .stage = .{ .module = shader_f8, .p_name = "matmul_f8" }, .layout = pipeline_layout },
                .{ .stage = .{ .module = shader_f32, .p_name = "matmul_f32" }, .layout = pipeline_layout },
            };
            try check(d.CreateComputePipelines(device, .null_handle, infos.len, &infos, null, &pipes));
        }
        errdefer for (pipes) |p| d.DestroyPipeline(device, p, null);

        // Transpose pipeline: 2 buffers + small push range.
        var dsl_tr: vk.DescriptorSetLayout = .null_handle;
        {
            var bindings: [2]vk.DescriptorSetLayoutBinding = undefined;
            for (&bindings, 0..) |*bd, i| {
                bd.* = .{
                    .binding = @intCast(i),
                    .descriptor_type = .storage_buffer,
                    .descriptor_count = 1,
                    .stage_flags = vk.ShaderStage.compute,
                    .p_immutable_samplers = null,
                };
            }
            try check(d.CreateDescriptorSetLayout(device, &.{
                .binding_count = bindings.len,
                .p_bindings = &bindings,
            }, null, &dsl_tr));
        }
        errdefer d.DestroyDescriptorSetLayout(device, dsl_tr, null);
        var pipeline_layout_tr: vk.PipelineLayout = .null_handle;
        {
            const push_range: vk.PushConstantRange = .{
                .stage_flags = vk.ShaderStage.compute,
                .offset = 0,
                .size = @sizeOf(TransposePush),
            };
            try check(d.CreatePipelineLayout(device, &.{
                .set_layout_count = 1,
                .p_set_layouts = @ptrCast(&dsl_tr),
                .push_constant_range_count = 1,
                .p_push_constant_ranges = @ptrCast(&push_range),
            }, null, &pipeline_layout_tr));
        }
        errdefer d.DestroyPipelineLayout(device, pipeline_layout_tr, null);
        var pipes_tr: [5]vk.Pipeline = @splat(.null_handle);
        {
            const infos = [5]vk.ComputePipelineCreateInfo{
                .{ .stage = .{ .module = shader_tr, .p_name = "transpose_f8" }, .layout = pipeline_layout_tr },
                .{ .stage = .{ .module = shader_tr, .p_name = "transpose_f32" }, .layout = pipeline_layout_tr },
                .{ .stage = .{ .module = shader_tr, .p_name = "transpose_bf16" }, .layout = pipeline_layout_tr },
                .{ .stage = .{ .module = shader_tr, .p_name = "bf16_to_f16_coopw" }, .layout = pipeline_layout_tr },
                .{ .stage = .{ .module = shader_tr, .p_name = "bf16_coopw" }, .layout = pipeline_layout_tr },
            };
            try check(d.CreateComputePipelines(device, .null_handle, infos.len, &infos, null, &pipes_tr));
        }
        errdefer for (pipes_tr) |pp| d.DestroyPipeline(device, pp, null);

        // Eltwise pipelines share the 4-buffer matmul set layout with a
        // wider push range.
        var pipeline_layout_e: vk.PipelineLayout = .null_handle;
        {
            const push_range: vk.PushConstantRange = .{
                .stage_flags = vk.ShaderStage.compute,
                .offset = 0,
                .size = @sizeOf(EltPush),
            };
            try check(d.CreatePipelineLayout(device, &.{
                .set_layout_count = 1,
                .p_set_layouts = @ptrCast(&dsl),
                .push_constant_range_count = 1,
                .p_push_constant_ranges = @ptrCast(&push_range),
            }, null, &pipeline_layout_e));
        }
        errdefer d.DestroyPipelineLayout(device, pipeline_layout_e, null);
        var pipes_e: [elt_entry_sizes.len]vk.Pipeline = @splat(vk.Pipeline.null_handle);
        {
            var infos: [elt_entry_sizes.len]vk.ComputePipelineCreateInfo = undefined;
            var names: [elt_entry_sizes.len][64]u8 = undefined;
            for (&infos, elt_entry_sizes, 0..) |*info, es, i| {
                @memset(&names[i], 0);
                @memcpy(names[i][0..es.name.len], es.name);
                info.* = .{ .stage = .{ .module = shader_e, .p_name = @ptrCast(&names[i]) }, .layout = pipeline_layout_e };
            }
            try check(d.CreateComputePipelines(device, .null_handle, infos.len, &infos, null, &pipes_e));
        }
        errdefer for (pipes_e) |pp| d.DestroyPipeline(device, pp, null);

        // Standalone batched-attention kernel: own shader module, 5-buffer set
        // (a=q,b=k,c=v,d=out,e=bounds), and one persistent descriptor set (the
        // 5 buffers are stable within a forward, so no ring is needed).
        var shader_ab: vk.ShaderModule = .null_handle;
        try createKernelModule(gpa, &d, device, attn_batched_spv, &.{.{ .name = "attn_batched", .x = 64, .y = 1 }}, &shader_ab);
        errdefer d.DestroyShaderModule(device, shader_ab, null);
        var dsl_ab: vk.DescriptorSetLayout = .null_handle;
        {
            var bindings: [5]vk.DescriptorSetLayoutBinding = undefined;
            for (&bindings, 0..) |*bd, i| bd.* = .{
                .binding = @intCast(i),
                .descriptor_type = .storage_buffer,
                .descriptor_count = 1,
                .stage_flags = vk.ShaderStage.compute,
                .p_immutable_samplers = null,
            };
            try check(d.CreateDescriptorSetLayout(device, &.{ .binding_count = bindings.len, .p_bindings = &bindings }, null, &dsl_ab));
        }
        errdefer d.DestroyDescriptorSetLayout(device, dsl_ab, null);
        var pipeline_layout_ab: vk.PipelineLayout = .null_handle;
        {
            const push_range: vk.PushConstantRange = .{ .stage_flags = vk.ShaderStage.compute, .offset = 0, .size = @sizeOf(AttnBatchedPush) };
            try check(d.CreatePipelineLayout(device, &.{
                .set_layout_count = 1,
                .p_set_layouts = @ptrCast(&dsl_ab),
                .push_constant_range_count = 1,
                .p_push_constant_ranges = @ptrCast(&push_range),
            }, null, &pipeline_layout_ab));
        }
        errdefer d.DestroyPipelineLayout(device, pipeline_layout_ab, null);
        var pipe_ab: vk.Pipeline = .null_handle;
        {
            const info = [1]vk.ComputePipelineCreateInfo{.{ .stage = .{ .module = shader_ab, .p_name = "attn_batched" }, .layout = pipeline_layout_ab }};
            try check(d.CreateComputePipelines(device, .null_handle, 1, &info, null, @ptrCast(&pipe_ab)));
        }
        errdefer d.DestroyPipeline(device, pipe_ab, null);
        var pool_ab: vk.DescriptorPool = .null_handle;
        {
            const pool_size: vk.DescriptorPoolSize = .{ .type = .storage_buffer, .descriptor_count = 5 };
            try check(d.CreateDescriptorPool(device, &.{ .max_sets = 1, .pool_size_count = 1, .p_pool_sizes = @ptrCast(&pool_size) }, null, &pool_ab));
        }
        errdefer d.DestroyDescriptorPool(device, pool_ab, null);
        var desc_set_ab: vk.DescriptorSet = .null_handle;
        try check(d.AllocateDescriptorSets(device, &.{ .descriptor_pool = pool_ab, .descriptor_set_count = 1, .p_set_layouts = @ptrCast(&dsl_ab) }, @ptrCast(&desc_set_ab)));

        // Cooperative-matrix GEMM pipeline (hand-assembled SPIR-V), when the
        // device supports 16x16x16 f16->f32 subgroup matrices.
        var shader_coop: vk.ShaderModule = .null_handle;
        var pipe_coop: vk.Pipeline = .null_handle;
        var shader_coop_c16: vk.ShaderModule = .null_handle;
        var pipe_coop_c16: vk.Pipeline = .null_handle;
        var shader_coop_f16w: vk.ShaderModule = .null_handle;
        var pipe_coop_f16w: vk.Pipeline = .null_handle;
        var shader_coop_bf16w: vk.ShaderModule = .null_handle;
        var pipe_coop_bf16w: vk.Pipeline = .null_handle;
        if (coop_m == 16 and coop_n == 16 and coop_k == 16) {
            inline for (.{
                .{ false, coopmat.coop_warps8, coopmat.coop_acc_h16, false, &shader_coop },
                .{ true, false, false, false, &shader_coop_f16w },
            }) |v| {
                const code = try coopmat.buildGemmShared(gpa, v[0], v[1], v[2], v[3], false);
                defer gpa.free(code);
                try check(d.CreateShaderModule(device, &.{
                    .code_size = code.len,
                    .p_code = @ptrCast(@alignCast(code.ptr)),
                }, null, v[4]));
            }
            // f16 C store rides the fp8 pipe's toggles; it only exists (and
            // is only exact) with f16 accumulators.
            if (coopmat.coop_acc_h16) {
                const code = try coopmat.buildGemmShared(gpa, false, coopmat.coop_warps8, true, true, false);
                defer gpa.free(code);
                try check(d.CreateShaderModule(device, &.{
                    .code_size = code.len,
                    .p_code = @ptrCast(@alignCast(code.ptr)),
                }, null, &shader_coop_c16));
            }
            // Native bf16 f16-weight-style GEMM (f16-weight path with bf16 A/B).
            if (coop_bf16) {
                const code = try coopmat.buildGemmShared(gpa, true, false, false, false, true);
                defer gpa.free(code);
                try check(d.CreateShaderModule(device, &.{
                    .code_size = code.len,
                    .p_code = @ptrCast(@alignCast(code.ptr)),
                }, null, &shader_coop_bf16w));
            }
        }
        errdefer if (shader_coop != .null_handle) d.DestroyShaderModule(device, shader_coop, null);
        errdefer if (shader_coop_c16 != .null_handle) d.DestroyShaderModule(device, shader_coop_c16, null);
        errdefer if (shader_coop_f16w != .null_handle) d.DestroyShaderModule(device, shader_coop_f16w, null);
        errdefer if (shader_coop_bf16w != .null_handle) d.DestroyShaderModule(device, shader_coop_bf16w, null);
        if (shader_coop != .null_handle) {
            inline for (.{ .{ shader_coop, &pipe_coop }, .{ shader_coop_f16w, &pipe_coop_f16w } }) |v| {
                try createCoopPipe(&d, device, v[0], pipeline_layout, coop_sg, v[1]);
            }
            if (shader_coop_c16 != .null_handle) {
                try createCoopPipe(&d, device, shader_coop_c16, pipeline_layout, coop_sg, &pipe_coop_c16);
            }
            if (shader_coop_bf16w != .null_handle) {
                try createCoopPipe(&d, device, shader_coop_bf16w, pipeline_layout, coop_sg, &pipe_coop_bf16w);
            }
        }
        errdefer if (pipe_coop != .null_handle) d.DestroyPipeline(device, pipe_coop, null);
        errdefer if (pipe_coop_c16 != .null_handle) d.DestroyPipeline(device, pipe_coop_c16, null);
        errdefer if (pipe_coop_f16w != .null_handle) d.DestroyPipeline(device, pipe_coop_f16w, null);

        // int8 tensor-core GEMM (s8*s8->s32), independent of the f16 coop
        // support: gated on the probed sint8 cooperative-matrix config.
        var shader_coop_i8: vk.ShaderModule = .null_handle;
        var pipe_coop_i8: vk.Pipeline = .null_handle;
        if (coop_i8_m == 16 and coop_i8_n == 16 and coop_i8_k == 32) {
            const code = try coopmat.buildGemmI8(gpa, coopmat.i8_mt, coopmat.i8_nt);
            defer gpa.free(code);
            try check(d.CreateShaderModule(device, &.{
                .code_size = code.len,
                .p_code = @ptrCast(@alignCast(code.ptr)),
            }, null, &shader_coop_i8));
            try createCoopPipe(&d, device, shader_coop_i8, pipeline_layout, coop_sg, &pipe_coop_i8);
        }
        errdefer if (shader_coop_i8 != .null_handle) d.DestroyShaderModule(device, shader_coop_i8, null);
        errdefer if (pipe_coop_i8 != .null_handle) d.DestroyPipeline(device, pipe_coop_i8, null);

        // Shared-memory-staged int8 GEMM (Fork #2): the fast path for large
        // 128-multiple shapes.
        var shader_coop_i8_sh: vk.ShaderModule = .null_handle;
        var pipe_coop_i8_sh: vk.Pipeline = .null_handle;
        var shader_coop_i8_fs: vk.ShaderModule = .null_handle;
        var pipe_coop_i8_fs: vk.Pipeline = .null_handle;
        var shader_coop_i8_fs16: vk.ShaderModule = .null_handle;
        var pipe_coop_i8_fs16: vk.Pipeline = .null_handle;
        if (coop_i8_m == 16 and coop_i8_n == 16 and coop_i8_k == 32 and coopmat.i8_shared) {
            // (fuse_scale, c_h16, shader*, pipe*): raw-s32, fused-f32, fused-f16.
            inline for (.{
                .{ false, false, &shader_coop_i8_sh, &pipe_coop_i8_sh },
                .{ true, false, &shader_coop_i8_fs, &pipe_coop_i8_fs },
                .{ true, true, &shader_coop_i8_fs16, &pipe_coop_i8_fs16 },
            }) |cfg| {
                const code = try coopmat.buildGemmSharedI8(gpa, coopmat.coop_i8_warps8, cfg[0], coopmat.coop_i8_double_buf, cfg[1]);
                defer gpa.free(code);
                try check(d.CreateShaderModule(device, &.{
                    .code_size = code.len,
                    .p_code = @ptrCast(@alignCast(code.ptr)),
                }, null, cfg[2]));
                try createCoopPipe(&d, device, cfg[2].*, pipeline_layout, coop_sg, cfg[3]);
            }
        }
        errdefer if (shader_coop_i8_fs16 != .null_handle) d.DestroyShaderModule(device, shader_coop_i8_fs16, null);
        errdefer if (pipe_coop_i8_fs16 != .null_handle) d.DestroyPipeline(device, pipe_coop_i8_fs16, null);
        errdefer if (shader_coop_i8_sh != .null_handle) d.DestroyShaderModule(device, shader_coop_i8_sh, null);
        errdefer if (pipe_coop_i8_sh != .null_handle) d.DestroyPipeline(device, pipe_coop_i8_sh, null);
        errdefer if (shader_coop_i8_fs != .null_handle) d.DestroyShaderModule(device, shader_coop_i8_fs, null);
        errdefer if (pipe_coop_i8_fs != .null_handle) d.DestroyPipeline(device, pipe_coop_i8_fs, null);

        // Stage B: fused int8 prep kernels (f16-shared FWHT; hand-assembled),
        // one per convrot cols (6144, 16384).
        var shader_i8_prep6144: vk.ShaderModule = .null_handle;
        var pipe_i8_prep6144: vk.Pipeline = .null_handle;
        var shader_i8_prep16384: vk.ShaderModule = .null_handle;
        var pipe_i8_prep16384: vk.Pipeline = .null_handle;
        if (coop_i8_m == 16 and coop_i8_n == 16 and coop_i8_k == 32 and coopmat.i8_shared) {
            inline for (.{ .{ 6144, &shader_i8_prep6144, &pipe_i8_prep6144 }, .{ 16384, &shader_i8_prep16384, &pipe_i8_prep16384 } }) |cfg| {
                const code = try coopmat.buildFusedPrepI8(gpa, cfg[0]);
                defer gpa.free(code);
                try check(d.CreateShaderModule(device, &.{
                    .code_size = code.len,
                    .p_code = @ptrCast(@alignCast(code.ptr)),
                }, null, cfg[1]));
                const info: vk.ComputePipelineCreateInfo = .{
                    .stage = .{ .module = cfg[1].*, .p_name = "main" },
                    .layout = pipeline_layout,
                };
                try check(d.CreateComputePipelines(device, .null_handle, 1, @ptrCast(&info), null, @ptrCast(cfg[2])));
            }
        }
        errdefer if (shader_i8_prep6144 != .null_handle) d.DestroyShaderModule(device, shader_i8_prep6144, null);
        errdefer if (pipe_i8_prep6144 != .null_handle) d.DestroyPipeline(device, pipe_i8_prep6144, null);
        errdefer if (shader_i8_prep16384 != .null_handle) d.DestroyShaderModule(device, shader_i8_prep16384, null);
        errdefer if (pipe_i8_prep16384 != .null_handle) d.DestroyPipeline(device, pipe_i8_prep16384, null);

        // Attention-scores GEMM on the same cooperative-matrix support; uses
        // the eltwise pipeline layout (same 4-buffer set, EltPush-sized push).
        // Two head_dim variants: 128 (DiT) and 384 (VAE mid-block).
        var shader_scores: vk.ShaderModule = .null_handle;
        var pipe_scores: vk.Pipeline = .null_handle;
        var shader_scores_vae: vk.ShaderModule = .null_handle;
        var pipe_scores_vae: vk.Pipeline = .null_handle;
        if (shader_coop != .null_handle) {
            inline for (.{ .{ 128, &shader_scores }, .{ 384, &shader_scores_vae } }) |v| {
                const code = try coopmat.buildGemmScores(gpa, v[0], coopmat.scores_stage_k);
                defer gpa.free(code);
                try check(d.CreateShaderModule(device, &.{
                    .code_size = code.len,
                    .p_code = @ptrCast(@alignCast(code.ptr)),
                }, null, v[1]));
            }
        }
        errdefer if (shader_scores != .null_handle) d.DestroyShaderModule(device, shader_scores, null);
        errdefer if (shader_scores_vae != .null_handle) d.DestroyShaderModule(device, shader_scores_vae, null);
        if (shader_scores != .null_handle) {
            inline for (.{ .{ shader_scores, &pipe_scores }, .{ shader_scores_vae, &pipe_scores_vae } }) |v| {
                try createCoopPipe(&d, device, v[0], pipeline_layout_e, coop_sg, v[1]);
            }
        }
        errdefer if (pipe_scores != .null_handle) d.DestroyPipeline(device, pipe_scores, null);
        errdefer if (pipe_scores_vae != .null_handle) d.DestroyPipeline(device, pipe_scores_vae, null);

        var shader_attn_out: vk.ShaderModule = .null_handle;
        var pipe_attn_out: vk.Pipeline = .null_handle;
        if (shader_coop != .null_handle) {
            const code = try coopmat.buildGemmAttnOut(gpa);
            defer gpa.free(code);
            try check(d.CreateShaderModule(device, &.{
                .code_size = code.len,
                .p_code = @ptrCast(@alignCast(code.ptr)),
            }, null, &shader_attn_out));
        }
        errdefer if (shader_attn_out != .null_handle) d.DestroyShaderModule(device, shader_attn_out, null);
        if (shader_attn_out != .null_handle) {
            try createCoopPipe(&d, device, shader_attn_out, pipeline_layout_e, coop_sg, &pipe_attn_out);
        }
        errdefer if (pipe_attn_out != .null_handle) d.DestroyPipeline(device, pipe_attn_out, null);

        // Flash attention (head_dim 128): the scores matrix never touches
        // global memory (see coopmat.buildFlashAttn).
        var shader_flash_md: vk.ShaderModule = .null_handle;
        var pipe_flash_md: vk.Pipeline = .null_handle;
        var shader_flash_out: vk.ShaderModule = .null_handle;
        var pipe_flash_out: vk.Pipeline = .null_handle;
        if (shader_coop != .null_handle) {
            inline for (.{
                .{ coopmat.buildFlashMd, &shader_flash_md, &pipe_flash_md },
                .{ coopmat.buildFlashOut, &shader_flash_out, &pipe_flash_out },
            }) |v| {
                const code = try v[0](gpa);
                defer gpa.free(code);
                try check(d.CreateShaderModule(device, &.{
                    .code_size = code.len,
                    .p_code = @ptrCast(@alignCast(code.ptr)),
                }, null, v[1]));
                try createCoopPipe(&d, device, v[1].*, pipeline_layout_e, coop_sg, v[2]);
            }
        }
        errdefer if (shader_flash_md != .null_handle) d.DestroyShaderModule(device, shader_flash_md, null);
        errdefer if (pipe_flash_md != .null_handle) d.DestroyPipeline(device, pipe_flash_md, null);
        errdefer if (shader_flash_out != .null_handle) d.DestroyShaderModule(device, shader_flash_out, null);
        errdefer if (pipe_flash_out != .null_handle) d.DestroyPipeline(device, pipe_flash_out, null);

        var desc_pool: vk.DescriptorPool = .null_handle;
        {
            const pool_size: vk.DescriptorPoolSize = .{ .type = .storage_buffer, .descriptor_count = 6 };
            try check(d.CreateDescriptorPool(device, &.{
                .max_sets = 2,
                .pool_size_count = 1,
                .p_pool_sizes = @ptrCast(&pool_size),
            }, null, &desc_pool));
        }
        errdefer d.DestroyDescriptorPool(device, desc_pool, null);
        var desc_set: vk.DescriptorSet = .null_handle;
        try check(d.AllocateDescriptorSets(device, &.{
            .descriptor_pool = desc_pool,
            .descriptor_set_count = 1,
            .p_set_layouts = @ptrCast(&dsl),
        }, @ptrCast(&desc_set)));
        var desc_set_tr: vk.DescriptorSet = .null_handle;
        try check(d.AllocateDescriptorSets(device, &.{
            .descriptor_pool = desc_pool,
            .descriptor_set_count = 1,
            .p_set_layouts = @ptrCast(&dsl_tr),
        }, @ptrCast(&desc_set_tr)));

        // Descriptor-set ring for batched recording (one set per recorded
        // op; a DiT step at 1024px is ~800 dispatches).
        const batch_ring = 1024;
        var batch_pool: vk.DescriptorPool = .null_handle;
        {
            const pool_size: vk.DescriptorPoolSize = .{ .type = .storage_buffer, .descriptor_count = batch_ring * 4 };
            try check(d.CreateDescriptorPool(device, &.{
                .max_sets = batch_ring,
                .pool_size_count = 1,
                .p_pool_sizes = @ptrCast(&pool_size),
            }, null, &batch_pool));
        }
        errdefer d.DestroyDescriptorPool(device, batch_pool, null);
        const batch_sets = try gpa.alloc(vk.DescriptorSet, batch_ring);
        errdefer gpa.free(batch_sets);
        {
            const layouts = try gpa.alloc(vk.DescriptorSetLayout, batch_ring);
            defer gpa.free(layouts);
            @memset(layouts, dsl);
            try check(d.AllocateDescriptorSets(device, &.{
                .descriptor_pool = batch_pool,
                .descriptor_set_count = batch_ring,
                .p_set_layouts = layouts.ptr,
            }, batch_sets.ptr));
        }

        const self = try gpa.create(Context);
        self.* = .{
            .gpa = gpa,
            .lib = lib,
            .d = d,
            .instance = instance,
            .device = device,
            .queue = queue,
            .queue_family = queue_family,
            .mem_props = mem_props,
            .phys = phys,
            .has_memory_budget = has_memory_budget,
            .device_heap = device_heap,
            .device_name = undefined,
            .device_name_len = 0,
            .coop_m = coop_m,
            .coop_n = coop_n,
            .coop_k = coop_k,
            .coop_i8_m = coop_i8_m,
            .coop_i8_n = coop_i8_n,
            .coop_i8_k = coop_i8_k,
            .cmd_pool = cmd_pool,
            .cmd = cmd,
            .cmd_now = cmd_now,
            .fence = fence,
            .batch_pool = batch_pool,
            .batch_sets = batch_sets,
            .dsl = dsl,
            .pipeline_layout = pipeline_layout,
            .shader_f8 = shader_f8,
            .shader_f32 = shader_f32,
            .pipe_f8 = pipes[0],
            .pipe_f32 = pipes[1],
            .dsl_tr = dsl_tr,
            .pipeline_layout_tr = pipeline_layout_tr,
            .shader_tr = shader_tr,
            .pipe_tr_f8 = pipes_tr[0],
            .pipe_tr_f32 = pipes_tr[1],
            .pipe_tr_bf16 = pipes_tr[2],
            .pipe_tr_bf16w = pipes_tr[3],
            .pipe_tr_bf16raw = pipes_tr[4],
            .pipeline_layout_e = pipeline_layout_e,
            .shader_e = shader_e,
            .pipes_e = pipes_e,
            .shader_ab = shader_ab,
            .dsl_ab = dsl_ab,
            .pipeline_layout_ab = pipeline_layout_ab,
            .pipe_ab = pipe_ab,
            .pool_ab = pool_ab,
            .desc_set_ab = desc_set_ab,
            .shader_coop = shader_coop,
            .pipe_coop = pipe_coop,
            .shader_coop_i8 = shader_coop_i8,
            .pipe_coop_i8 = pipe_coop_i8,
            .shader_coop_i8_sh = shader_coop_i8_sh,
            .pipe_coop_i8_sh = pipe_coop_i8_sh,
            .shader_coop_i8_fs = shader_coop_i8_fs,
            .pipe_coop_i8_fs = pipe_coop_i8_fs,
            .shader_coop_i8_fs16 = shader_coop_i8_fs16,
            .pipe_coop_i8_fs16 = pipe_coop_i8_fs16,
            .shader_i8_prep6144 = shader_i8_prep6144,
            .pipe_i8_prep6144 = pipe_i8_prep6144,
            .shader_i8_prep16384 = shader_i8_prep16384,
            .pipe_i8_prep16384 = pipe_i8_prep16384,
            .shader_coop_c16 = shader_coop_c16,
            .pipe_coop_c16 = pipe_coop_c16,
            .shader_coop_f16w = shader_coop_f16w,
            .pipe_coop_f16w = pipe_coop_f16w,
            .shader_coop_bf16w = shader_coop_bf16w,
            .pipe_coop_bf16w = pipe_coop_bf16w,
            .shader_scores = shader_scores,
            .pipe_scores = pipe_scores,
            .shader_scores_vae = shader_scores_vae,
            .pipe_scores_vae = pipe_scores_vae,
            .shader_attn_out = shader_attn_out,
            .pipe_attn_out = pipe_attn_out,
            .shader_flash_md = shader_flash_md,
            .pipe_flash_md = pipe_flash_md,
            .shader_flash_out = shader_flash_out,
            .pipe_flash_out = pipe_flash_out,
            .desc_pool = desc_pool,
            .desc_set = desc_set,
            .desc_set_tr = desc_set_tr,
        };
        const name_z = std.mem.sliceTo(&props.device_name, 0);
        self.device_name_len = @min(name_z.len, self.device_name.len);
        @memcpy(self.device_name[0..self.device_name_len], name_z[0..self.device_name_len]);
        self.subgroup_size = native_subgroup;
        self.coop_pinned_subgroup = coop_sg orelse 0;
        self.coop_disabled_by_subgroup = coop_disabled_by_subgroup;
        return self;
    }

    pub fn deinit(self: *Context) void {
        _ = self.d.DeviceWaitIdle(self.device);
        var it = self.weights.valueIterator();
        while (it.next()) |wb| {
            self.freeDeviceBuffer(wb.db);
        }
        self.weights.deinit(self.gpa);
        {
            var sit = self.small_bufs.valueIterator();
            while (sit.next()) |sb| {
                self.freeDeviceBuffer(sb.*);
            }
            self.small_bufs.deinit(self.gpa);
        }
        for ([_]*DeviceBuffer{ &self.x_dev, &self.y_dev, &self.raw_dev, &self.dummy, &self.pen_wire, &self.x_h16, &self.y_pad, &self.i8_xr, &self.i8_scale, &self.i8_x, &self.i8_acc, &self.i8_acc1, &self.i8_acc2, &self.i8_acc3, &self.i8_hadamard, &self.i8_partials, &self.i8_scalecat }) |db| {
            if (db.buf != .null_handle) {
                self.freeDeviceBuffer(db.*);
            }
        }
        for ([_]*HostBuffer{ &self.x_buf, &self.y_buf, &self.bias_buf, &self.staging }) |hb| {
            self.destroyHostBuffer(hb);
        }
        self.d.DestroyDescriptorPool(self.device, self.batch_pool, null);
        self.gpa.free(self.batch_sets);
        self.d.DestroyDescriptorPool(self.device, self.desc_pool, null);
        self.d.DestroyPipeline(self.device, self.pipe_f8, null);
        self.d.DestroyPipeline(self.device, self.pipe_f32, null);
        self.d.DestroyPipeline(self.device, self.pipe_tr_bf16, null);
        self.d.DestroyPipeline(self.device, self.pipe_tr_bf16w, null);
        self.d.DestroyPipeline(self.device, self.pipe_tr_bf16raw, null);
        self.d.DestroyPipeline(self.device, self.pipe_tr_f8, null);
        self.d.DestroyPipeline(self.device, self.pipe_tr_f32, null);
        for (self.pipes_e) |pp| self.d.DestroyPipeline(self.device, pp, null);
        if (self.pipe_coop != .null_handle) self.d.DestroyPipeline(self.device, self.pipe_coop, null);
        if (self.shader_coop != .null_handle) self.d.DestroyShaderModule(self.device, self.shader_coop, null);
        if (self.pipe_coop_i8 != .null_handle) self.d.DestroyPipeline(self.device, self.pipe_coop_i8, null);
        if (self.shader_coop_i8 != .null_handle) self.d.DestroyShaderModule(self.device, self.shader_coop_i8, null);
        if (self.pipe_coop_i8_sh != .null_handle) self.d.DestroyPipeline(self.device, self.pipe_coop_i8_sh, null);
        if (self.shader_coop_i8_sh != .null_handle) self.d.DestroyShaderModule(self.device, self.shader_coop_i8_sh, null);
        if (self.pipe_coop_i8_fs != .null_handle) self.d.DestroyPipeline(self.device, self.pipe_coop_i8_fs, null);
        if (self.shader_coop_i8_fs != .null_handle) self.d.DestroyShaderModule(self.device, self.shader_coop_i8_fs, null);
        if (self.pipe_coop_i8_fs16 != .null_handle) self.d.DestroyPipeline(self.device, self.pipe_coop_i8_fs16, null);
        if (self.shader_coop_i8_fs16 != .null_handle) self.d.DestroyShaderModule(self.device, self.shader_coop_i8_fs16, null);
        if (self.pipe_i8_prep6144 != .null_handle) self.d.DestroyPipeline(self.device, self.pipe_i8_prep6144, null);
        if (self.shader_i8_prep6144 != .null_handle) self.d.DestroyShaderModule(self.device, self.shader_i8_prep6144, null);
        if (self.pipe_i8_prep16384 != .null_handle) self.d.DestroyPipeline(self.device, self.pipe_i8_prep16384, null);
        if (self.shader_i8_prep16384 != .null_handle) self.d.DestroyShaderModule(self.device, self.shader_i8_prep16384, null);
        if (self.pipe_coop_c16 != .null_handle) self.d.DestroyPipeline(self.device, self.pipe_coop_c16, null);
        if (self.shader_coop_c16 != .null_handle) self.d.DestroyShaderModule(self.device, self.shader_coop_c16, null);
        if (self.pipe_coop_f16w != .null_handle) self.d.DestroyPipeline(self.device, self.pipe_coop_f16w, null);
        if (self.shader_coop_f16w != .null_handle) self.d.DestroyShaderModule(self.device, self.shader_coop_f16w, null);
        if (self.pipe_coop_bf16w != .null_handle) self.d.DestroyPipeline(self.device, self.pipe_coop_bf16w, null);
        if (self.shader_coop_bf16w != .null_handle) self.d.DestroyShaderModule(self.device, self.shader_coop_bf16w, null);
        if (self.pipe_scores != .null_handle) self.d.DestroyPipeline(self.device, self.pipe_scores, null);
        if (self.shader_scores != .null_handle) self.d.DestroyShaderModule(self.device, self.shader_scores, null);
        if (self.pipe_scores_vae != .null_handle) self.d.DestroyPipeline(self.device, self.pipe_scores_vae, null);
        if (self.shader_scores_vae != .null_handle) self.d.DestroyShaderModule(self.device, self.shader_scores_vae, null);
        if (self.pipe_attn_out != .null_handle) self.d.DestroyPipeline(self.device, self.pipe_attn_out, null);
        if (self.shader_attn_out != .null_handle) self.d.DestroyShaderModule(self.device, self.shader_attn_out, null);
        if (self.pipe_flash_md != .null_handle) self.d.DestroyPipeline(self.device, self.pipe_flash_md, null);
        if (self.shader_flash_md != .null_handle) self.d.DestroyShaderModule(self.device, self.shader_flash_md, null);
        if (self.pipe_flash_out != .null_handle) self.d.DestroyPipeline(self.device, self.pipe_flash_out, null);
        if (self.shader_flash_out != .null_handle) self.d.DestroyShaderModule(self.device, self.shader_flash_out, null);
        self.d.DestroyPipelineLayout(self.device, self.pipeline_layout, null);
        self.d.DestroyPipelineLayout(self.device, self.pipeline_layout_tr, null);
        self.d.DestroyPipelineLayout(self.device, self.pipeline_layout_e, null);
        self.d.DestroyShaderModule(self.device, self.shader_f8, null);
        self.d.DestroyShaderModule(self.device, self.shader_f32, null);
        self.d.DestroyShaderModule(self.device, self.shader_tr, null);
        self.d.DestroyShaderModule(self.device, self.shader_e, null);
        self.d.DestroyPipeline(self.device, self.pipe_ab, null);
        self.d.DestroyPipelineLayout(self.device, self.pipeline_layout_ab, null);
        self.d.DestroyShaderModule(self.device, self.shader_ab, null);
        self.d.DestroyDescriptorPool(self.device, self.pool_ab, null);
        self.d.DestroyDescriptorSetLayout(self.device, self.dsl_ab, null);
        self.d.DestroyDescriptorSetLayout(self.device, self.dsl, null);
        self.d.DestroyDescriptorSetLayout(self.device, self.dsl_tr, null);
        self.d.DestroyFence(self.device, self.fence, null);
        self.d.DestroyCommandPool(self.device, self.cmd_pool, null);
        self.d.DestroyDevice(self.device, null);
        self.d.DestroyInstance(self.instance, null);
        self.lib.close();
        const gpa = self.gpa;
        gpa.destroy(self);
    }

    pub fn deviceName(self: *const Context) []const u8 {
        return self.device_name[0..self.device_name_len];
    }

    /// Write a one-line summary of the cooperative-matrix (tensor-core) setup:
    /// whether coop GEMMs are active, and how the device's subgroup width was
    /// reconciled with the kernels'. Purely informational.
    pub fn writeCoopStatus(self: *const Context, w: *std.Io.Writer) !void {
        const coop_on = self.pipe_coop != .null_handle;
        if (coop_on and self.coop_pinned_subgroup != 0) {
            try w.print("coopmat: on (subgroup pinned to {d}, device native {d})\n", .{
                self.coop_pinned_subgroup, self.subgroup_size,
            });
        } else if (coop_on) {
            try w.print("coopmat: on (native subgroup {d})\n", .{self.subgroup_size});
        } else if (self.coop_disabled_by_subgroup) {
            try w.print("coopmat: off (device subgroup {d} not pinnable to {d}); register/f32 fallback\n", .{
                self.subgroup_size, coopmat.subgroup_lanes,
            });
        } else {
            try w.print("coopmat: off (not supported by device); register/f32 path\n", .{});
        }
    }

    fn findMemoryType(self: *const Context, type_bits: u32, flags: u32) Error!u32 {
        for (0..self.mem_props.memory_type_count) |i| {
            const bit = @as(u32, 1) << @intCast(i);
            if (type_bits & bit != 0 and self.mem_props.memory_types[i].property_flags & flags == flags) {
                return @intCast(i);
            }
        }
        return error.NoSuitableDevice;
    }

    fn createBuffer(self: *Context, size: u64, usage: u32, mem_flags: u32) Error!DeviceBuffer {
        while (true) {
            if (self.createBufferRaw(size, usage, mem_flags)) |db_raw| {
                var db = db_raw;
                db.tag = self.mem_tag;
                self.device_used += size;
                if (self.track_tags) self.mem_tag_used[@intFromEnum(self.mem_tag)] += size;
                return db;
            } else |err| {
                if (err != error.DeviceOutOfMemory) return err;
                // Degrade to weight re-uploads instead of failing: drop the
                // least-recently-used cached weight buffer and retry.
                if (!self.evictOneWeight()) return err;
            }
        }
    }

    fn createBufferRaw(self: *Context, size: u64, usage: u32, mem_flags: u32) Error!DeviceBuffer {
        var buf: vk.Buffer = .null_handle;
        try check(self.d.CreateBuffer(self.device, &.{ .size = size, .usage = usage }, null, &buf));
        errdefer self.d.DestroyBuffer(self.device, buf, null);
        var reqs: vk.MemoryRequirements = undefined;
        self.d.GetBufferMemoryRequirements(self.device, buf, &reqs);
        var mem: vk.DeviceMemory = .null_handle;
        const r = self.d.AllocateMemory(self.device, &.{
            .allocation_size = reqs.size,
            .memory_type_index = try self.findMemoryType(reqs.memory_type_bits, mem_flags),
        }, null, &mem);
        if (r == .error_out_of_device_memory) return error.DeviceOutOfMemory;
        try check(r);
        errdefer self.d.FreeMemory(self.device, mem, null);
        try check(self.d.BindBufferMemory(self.device, buf, mem, 0));
        return .{ .buf = buf, .mem = mem, .size = size };
    }

    /// Central device-buffer free — keeps `device_used` honest. All
    /// DeviceBuffer teardown must route through here.
    fn freeDeviceBuffer(self: *Context, db: DeviceBuffer) void {
        self.d.DestroyBuffer(self.device, db.buf, null);
        self.d.FreeMemory(self.device, db.mem, null);
        self.device_used -|= db.size;
        if (self.track_tags) self.mem_tag_used[@intFromEnum(db.tag)] -|= db.size;
    }

    /// Enable MEASURED per-component device-memory attribution (VRAM meter).
    pub fn enableMemTags(self: *Context) void {
        self.track_tags = true;
    }

    /// Free resident weights so the footprint fits `budget` (GUI VRAM limit
    /// lowered while idle). Coarse: if over budget, drop the whole weight cache;
    /// the next generation re-uploads what fits. Caller ensures no work is in
    /// flight (DeviceWaitIdle inside evictWeights).
    pub fn trimToBudget(self: *Context, budget: u64) void {
        if (self.device_used <= budget) return;
        self.evictWeights();
    }
    /// Set the tag applied to subsequent device allocations (pipeline brackets
    /// each phase). Only meaningful when `track_tags` is on.
    pub fn setMemTag(self: *Context, tag: MemTag) void {
        self.mem_tag = tag;
    }
    /// Live device bytes attributed to `tag` (0 unless tagging is active).
    pub fn memTagUsed(self: *const Context, tag: MemTag) u64 {
        return self.mem_tag_used[@intFromEnum(tag)];
    }

    fn destroyHostBuffer(self: *Context, hb: *HostBuffer) void {
        if (hb.buf == .null_handle) return;
        if (hb.mapped != null) self.d.UnmapMemory(self.device, hb.mem);
        self.d.DestroyBuffer(self.device, hb.buf, null);
        self.d.FreeMemory(self.device, hb.mem, null);
        self.device_used -|= hb.size;
        hb.* = .{};
    }

    /// Grow-on-demand mapped host-visible storage buffer.
    fn ensureHostBuffer(self: *Context, hb: *HostBuffer, size: u64) Error![*]u8 {
        if (hb.size >= size) return hb.mapped.?;
        self.destroyHostBuffer(hb);
        const alloc_size = std.math.ceilPowerOfTwo(u64, @max(size, 1 << 16)) catch return error.OutOfMemory;
        const db = try self.createBuffer(
            alloc_size,
            vk.BufferUsage.storage_buffer | vk.BufferUsage.transfer_src | vk.BufferUsage.transfer_dst,
            vk.MemoryProperty.host_visible | vk.MemoryProperty.host_coherent,
        );
        var mapped: ?*anyopaque = null;
        try check(self.d.MapMemory(self.device, db.mem, 0, vk.WHOLE_SIZE, 0, &mapped));
        hb.* = .{ .buf = db.buf, .mem = db.mem, .size = alloc_size, .mapped = @ptrCast(mapped.?) };
        return hb.mapped.?;
    }

    /// Device-local weight buffer in the transposed k-major layout
    /// (element (k, col) at k * w_stride + col, w_stride = align(rows, tile_n)),
    /// uploaded once and cached by host pointer. The raw bytes are DMA'd as-is
    /// and transposed on the GPU (kernels/transpose.zig) — no CPU-side work
    /// beyond the staging memcpy.
    fn weightBuffer(self: *Context, bytes: []const u8, esize: u64, rows: usize, cols: usize) Error!vk.Buffer {
        const key = @intFromPtr(bytes.ptr);
        self.use_counter += 1;
        if (self.weights.getPtr(key)) |e| {
            e.last_use = self.use_counter;
            return e.db.buf;
        }

        const stride = std.mem.alignForward(u64, rows, tile_n);
        const total = stride * cols * esize;
        self.reserveForWeights(total);
        const db = try self.createBuffer(
            total,
            vk.BufferUsage.storage_buffer | vk.BufferUsage.transfer_dst,
            vk.MemoryProperty.device_local,
        );
        errdefer {
            self.d.DestroyBuffer(self.device, db.buf, null);
            self.d.FreeMemory(self.device, db.mem, null);
        }

        // Raw upload (chunked through staging), then transpose device-side.
        // Runs on the immediate command buffer: a batch recording on `cmd`
        // (which submits later) never touches this brand-new buffer.
        const cb = self.nowCmd();
        const raw_size = std.mem.alignForward(u64, bytes.len, 4);
        try self.ensureDeviceBuffer(&self.raw_dev, raw_size);
        const chunk: u64 = 256 << 20;
        const mapped = try self.ensureHostBuffer(&self.staging, @min(raw_size, chunk));
        var off: u64 = 0;
        while (off < bytes.len) {
            const n: u64 = @min(chunk, bytes.len - off);
            @memcpy(mapped[0..@intCast(n)], bytes[@intCast(off)..][0..@intCast(n)]);
            try self.beginCmdBuf(cb);
            const region: vk.BufferCopy = .{ .src_offset = 0, .dst_offset = off, .size = n };
            self.d.CmdCopyBuffer(cb, self.staging.buf, self.raw_dev.buf, 1, @ptrCast(&region));
            try self.submitAndWaitBuf(cb);
            off += n;
        }

        // Transpose dispatch: raw_dev -> db.
        {
            const buf_infos = [2]vk.DescriptorBufferInfo{
                .{ .buffer = self.raw_dev.buf },
                .{ .buffer = db.buf },
            };
            var writes: [2]vk.WriteDescriptorSet = undefined;
            for (&writes, 0..) |*wr, i| {
                wr.* = .{
                    .dst_set = self.desc_set_tr,
                    .dst_binding = @intCast(i),
                    .dst_array_element = 0,
                    .descriptor_count = 1,
                    .descriptor_type = .storage_buffer,
                    .p_image_info = null,
                    .p_buffer_info = @ptrCast(&buf_infos[i]),
                    .p_texel_buffer_view = null,
                };
            }
            self.d.UpdateDescriptorSets(self.device, writes.len, &writes, 0, null);

            const push: TransposePush = .{
                .rows = @intCast(rows),
                .cols = @intCast(cols),
                .stride = @intCast(stride),
            };
            // f8: one u32 (4 cols) per invocation in x; bf16: 2 cols; f32: 1.
            const x_items: u64 = switch (esize) {
                1 => stride / 4,
                2 => stride / 2,
                else => stride,
            };
            const tr_pipe = switch (esize) {
                1 => self.pipe_tr_f8,
                2 => self.pipe_tr_bf16,
                else => self.pipe_tr_f32,
            };
            try self.beginCmdBuf(cb);
            self.d.CmdBindPipeline(cb, .compute, tr_pipe);
            self.d.CmdBindDescriptorSets(cb, .compute, self.pipeline_layout_tr, 0, 1, @ptrCast(&self.desc_set_tr), 0, null);
            self.d.CmdPushConstants(cb, self.pipeline_layout_tr, vk.ShaderStage.compute, 0, @sizeOf(TransposePush), &push);
            self.d.CmdDispatch(
                cb,
                @intCast(std.math.divCeil(u64, x_items, tr_wg_x) catch unreachable),
                @intCast(std.math.divCeil(usize, cols, tr_wg_y) catch unreachable),
                1,
            );
            const to_shader: vk.BufferMemoryBarrier = .{
                .src_access_mask = vk.Access.shader_write,
                .dst_access_mask = vk.Access.shader_read,
                .buffer = db.buf,
                .offset = 0,
                .size = vk.WHOLE_SIZE,
            };
            self.d.CmdPipelineBarrier(cb, vk.PipelineStage.compute_shader, vk.PipelineStage.compute_shader, 0, 0, null, 1, @ptrCast(&to_shader), 0, null);
            try self.submitAndWaitBuf(cb);
        }

        try self.weights.put(self.gpa, key, .{ .db = db, .last_use = self.use_counter, .pinned = self.pinNew(total) });
        return db.buf;
    }

    /// Like weightBuffer, but uploads the bytes VERBATIM (no k-major transpose)
    /// — for GGUF block-quant weights whose row-major block layout the quant
    /// GEMV kernels read directly. Cached by host pointer, same residency.
    fn weightBufferRaw(self: *Context, bytes: []const u8) Error!vk.Buffer {
        const key = @intFromPtr(bytes.ptr);
        self.use_counter += 1;
        if (self.weights.getPtr(key)) |e| {
            e.last_use = self.use_counter;
            return e.db.buf;
        }
        const total = std.mem.alignForward(u64, bytes.len, 4);
        self.reserveForWeights(total);
        const db = try self.createBuffer(
            total,
            vk.BufferUsage.storage_buffer | vk.BufferUsage.transfer_dst,
            vk.MemoryProperty.device_local,
        );
        errdefer {
            self.d.DestroyBuffer(self.device, db.buf, null);
            self.d.FreeMemory(self.device, db.mem, null);
        }
        const cb = self.nowCmd();
        const chunk: u64 = 256 << 20;
        const mapped = try self.ensureHostBuffer(&self.staging, @min(total, chunk));
        var off: u64 = 0;
        while (off < bytes.len) {
            const n: u64 = @min(chunk, bytes.len - off);
            @memcpy(mapped[0..@intCast(n)], bytes[@intCast(off)..][0..@intCast(n)]);
            try self.beginCmdBuf(cb);
            const region: vk.BufferCopy = .{ .src_offset = 0, .dst_offset = off, .size = n };
            self.d.CmdCopyBuffer(cb, self.staging.buf, db.buf, 1, @ptrCast(&region));
            try self.submitAndWaitBuf(cb);
            off += n;
        }
        try self.weights.put(self.gpa, key, .{ .db = db, .last_use = self.use_counter, .pinned = self.pinNew(total) });
        return db.buf;
    }

    /// Like weightBufferRaw, but byte-transposes the row-major block-quant
    /// weight into 32-row groups so the `*_t` GEMV kernels read coalesced:
    /// logical byte `j` of row `row` moves to grp_base + j*32 (grp_base =
    /// (row/32)*row_bytes*32 + row%32). The bytes are unchanged (bit-identical
    /// dequant); only their order changes. Transpose is host-side and one-time
    /// (cached by host pointer). Rows are padded up to a multiple of 32; the
    /// pad slots are never read (out-of-range threads exit).
    fn weightBufferRawT(self: *Context, bytes: []const u8, row_bytes: usize) Error!vk.Buffer {
        const key = @intFromPtr(bytes.ptr);
        self.use_counter += 1;
        if (self.weights.getPtr(key)) |e| {
            e.last_use = self.use_counter;
            return e.db.buf;
        }
        const G: usize = 32;
        std.debug.assert(bytes.len % row_bytes == 0);
        const rows = bytes.len / row_bytes;
        const ngrp = (rows + G - 1) / G;
        const packed_size = ngrp * G * row_bytes;
        const total = std.mem.alignForward(u64, packed_size, 4);
        self.reserveForWeights(total);
        const db = try self.createBuffer(
            total,
            vk.BufferUsage.storage_buffer | vk.BufferUsage.transfer_dst,
            vk.MemoryProperty.device_local,
        );
        errdefer {
            self.d.DestroyBuffer(self.device, db.buf, null);
            self.d.FreeMemory(self.device, db.mem, null);
        }

        // Host byte-transpose into a scratch, then upload verbatim.
        const scratch = try self.gpa.alloc(u8, @intCast(total));
        defer self.gpa.free(scratch);
        if (packed_size != bytes.len) @memset(scratch, 0); // zero pad slots
        var row: usize = 0;
        while (row < rows) : (row += 1) {
            const base = (row / G) * row_bytes * G + (row % G);
            const src = bytes[row * row_bytes ..][0..row_bytes];
            var j: usize = 0;
            while (j < row_bytes) : (j += 1) scratch[base + j * G] = src[j];
        }

        const cb = self.nowCmd();
        const chunk: u64 = 256 << 20;
        const mapped = try self.ensureHostBuffer(&self.staging, @min(total, chunk));
        var off: u64 = 0;
        while (off < scratch.len) {
            const n: u64 = @min(chunk, scratch.len - off);
            @memcpy(mapped[0..@intCast(n)], scratch[@intCast(off)..][0..@intCast(n)]);
            try self.beginCmdBuf(cb);
            const region: vk.BufferCopy = .{ .src_offset = 0, .dst_offset = off, .size = n };
            self.d.CmdCopyBuffer(cb, self.staging.buf, db.buf, 1, @ptrCast(&region));
            try self.submitAndWaitBuf(cb);
            off += n;
        }
        try self.weights.put(self.gpa, key, .{ .db = db, .last_use = self.use_counter, .pinned = self.pinNew(total) });
        return db.buf;
    }

    /// Grow-on-demand device-local storage buffer.
    pub fn ensureDeviceBuffer(self: *Context, db: *DeviceBuffer, size: u64) Error!void {
        if (db.size >= size) return;
        if (db.buf != .null_handle) {
            // Recorded-but-unsubmitted dispatches may still reference the
            // old buffer; submit them before destroying it.
            if (self.batching) try self.flushBatch();
            self.freeDeviceBuffer(db.*);
            db.* = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 };
        }
        // Grow with slack to avoid realloc churn, but don't power-of-two
        // round large buffers: a 722 MB activation plane would jump to
        // 1 GiB, wasting ~40% of VRAM per buffer — invisible on an empty
        // card, fatal when another process holds most of it. Pow2 stays for
        // small buffers (cheap, avoids churn); large ones round to 8 MiB.
        const min_sz = @max(size, 1 << 16);
        const alloc_size = if (min_sz <= (8 << 20))
            std.math.ceilPowerOfTwo(u64, min_sz) catch return error.OutOfMemory
        else
            std.mem.alignForward(u64, min_sz, 8 << 20);
        db.* = try self.createBuffer(
            alloc_size,
            vk.BufferUsage.storage_buffer | vk.BufferUsage.transfer_src | vk.BufferUsage.transfer_dst,
            vk.MemoryProperty.device_local,
        );
    }

    // --- device-resident API (GPU-resident DiT forward) ---------------------

    pub fn tensorCreate(self: *Context, size: u64) Error!DeviceBuffer {
        return self.createBuffer(
            size,
            vk.BufferUsage.storage_buffer | vk.BufferUsage.transfer_src | vk.BufferUsage.transfer_dst,
            vk.MemoryProperty.device_local,
        );
    }

    pub fn tensorDestroy(self: *Context, db: *DeviceBuffer) void {
        if (db.buf == .null_handle) return;
        self.freeDeviceBuffer(db.*);
        db.* = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 };
    }

    pub fn tensorUpload(self: *Context, db: DeviceBuffer, bytes: []const u8) Error!void {
        return self.tensorUploadAt(db, 0, bytes);
    }

    pub fn tensorUploadAt(self: *Context, db: DeviceBuffer, dst_off: u64, bytes: []const u8) Error!void {
        // The target may be written by already-recorded ops; order them first.
        if (self.batching) try self.flushBatch();
        const cb = self.nowCmd();
        const chunk: u64 = 256 << 20;
        const mapped = try self.ensureHostBuffer(&self.staging, @min(@as(u64, bytes.len), chunk));
        var off: u64 = 0;
        while (off < bytes.len) {
            const n: u64 = @min(chunk, bytes.len - off);
            @memcpy(mapped[0..@intCast(n)], bytes[@intCast(off)..][0..@intCast(n)]);
            try self.beginCmdBuf(cb);
            const region: vk.BufferCopy = .{ .src_offset = 0, .dst_offset = dst_off + off, .size = n };
            self.d.CmdCopyBuffer(cb, self.staging.buf, db.buf, 1, @ptrCast(&region));
            try self.submitAndWaitBuf(cb);
            off += n;
        }
    }

    pub fn tensorDownload(self: *Context, db: DeviceBuffer, out: []u8) Error!void {
        return self.tensorDownloadAt(db, 0, out);
    }

    /// Synchronous device-to-device copy (used to reseed per-step buffers
    /// from cached uploads without a host round trip). Runs on the immediate
    /// command buffer; call before beginBatch or after a flush.
    pub fn tensorCopy(self: *Context, dst: DeviceBuffer, dst_off: u64, src: DeviceBuffer, src_off: u64, size: u64) Error!void {
        if (self.batching) try self.flushBatch();
        const cb = self.nowCmd();
        try self.beginCmdBuf(cb);
        const region: vk.BufferCopy = .{ .src_offset = src_off, .dst_offset = dst_off, .size = size };
        self.d.CmdCopyBuffer(cb, src.buf, dst.buf, 1, @ptrCast(&region));
        try self.submitAndWaitBuf(cb);
    }

    pub fn tensorDownloadAt(self: *Context, db: DeviceBuffer, src_off: u64, out: []u8) Error!void {
        // Recorded-but-unsubmitted ops may produce the data being read.
        if (self.batching) try self.flushBatch();
        const cb = self.nowCmd();
        const chunk: u64 = 256 << 20;
        const mapped = try self.ensureHostBuffer(&self.staging, @min(@as(u64, out.len), chunk));
        var off: u64 = 0;
        while (off < out.len) {
            const n: u64 = @min(chunk, out.len - off);
            try self.beginCmdBuf(cb);
            const region: vk.BufferCopy = .{ .src_offset = src_off + off, .dst_offset = 0, .size = n };
            self.d.CmdCopyBuffer(cb, db.buf, self.staging.buf, 1, @ptrCast(&region));
            const to_host: vk.BufferMemoryBarrier = .{
                .src_access_mask = vk.Access.transfer_write,
                .dst_access_mask = vk.Access.host_read,
                .buffer = self.staging.buf,
                .offset = 0,
                .size = vk.WHOLE_SIZE,
            };
            self.d.CmdPipelineBarrier(cb, vk.PipelineStage.transfer, vk.PipelineStage.host, 0, 0, null, 1, @ptrCast(&to_host), 0, null);
            try self.submitAndWaitBuf(cb);
            @memcpy(out[@intCast(off)..][0..@intCast(n)], mapped[0..@intCast(n)]);
            off += n;
        }
    }

    /// Small cached device upload (biases, norm weights, modulation vectors
    /// that are stable per model), keyed by host pointer.
    pub fn smallBuffer(self: *Context, bytes: []const u8) Error!vk.Buffer {
        const key = @intFromPtr(bytes.ptr);
        if (self.small_bufs.get(key)) |db| return db.buf;
        const db = try self.tensorCreate(std.mem.alignForward(u64, bytes.len, 4));
        errdefer {
            var mut = db;
            self.tensorDestroy(&mut);
        }
        try self.tensorUpload(db, bytes);
        try self.small_bufs.put(self.gpa, key, db);
        return db.buf;
    }

    fn dummyBuf(self: *Context) Error!vk.Buffer {
        if (self.dummy.buf == .null_handle) {
            self.dummy = try self.tensorCreate(16);
        }
        return self.dummy.buf;
    }

    /// Write the op's four buffers into a descriptor set and return it: the
    /// shared set outside a batch, the next ring set inside one (callers
    /// must opBegin() first, which guarantees ring capacity).
    fn bind4(self: *Context, bufs: [4]vk.Buffer) vk.DescriptorSet {
        return self.bind4Off(bufs, .{ 0, 0, 0, 0 });
    }

    fn bind4Off(self: *Context, bufs: [4]vk.Buffer, offs: [4]u64) vk.DescriptorSet {
        const set = if (self.batching) blk: {
            const s = self.batch_sets[self.batch_n];
            self.batch_n += 1;
            break :blk s;
        } else self.desc_set;
        var buf_infos: [4]vk.DescriptorBufferInfo = undefined;
        var writes: [4]vk.WriteDescriptorSet = undefined;
        for (&buf_infos, &writes, bufs, 0..) |*bi, *wr, buf, i| {
            bi.* = .{ .buffer = buf, .offset = offs[i] };
            wr.* = .{
                .dst_set = set,
                .dst_binding = @intCast(i),
                .dst_array_element = 0,
                .descriptor_count = 1,
                .descriptor_type = .storage_buffer,
                .p_image_info = null,
                .p_buffer_info = @ptrCast(bi),
                .p_texel_buffer_view = null,
            };
        }
        self.d.UpdateDescriptorSets(self.device, writes.len, &writes, 0, null);
        return set;
    }

    /// GEMM between device-resident tensors (weights cached/transposed as in
    /// `matmul`); x_off/y_off are byte offsets of the first row.
    pub fn opMatmul(
        self: *Context,
        y: DeviceBuffer,
        y_off: u64,
        x: DeviceBuffer,
        x_off: u64,
        m: usize,
        w_bytes: []const u8,
        dtype_f8: bool,
        rows: usize,
        cols: usize,
        scale: f32,
        bias: ?[]const f32,
    ) Error!void {
        const w_buf = try self.weightBuffer(w_bytes, if (dtype_f8) 1 else 4, rows, cols);
        const bias_buf = if (bias) |bv| try self.smallBuffer(std.mem.sliceAsBytes(bv)) else try self.dummyBuf();

        const push: Push = .{
            .m = @intCast(m),
            .rows = @intCast(rows),
            .cols = @intCast(cols),
            .w_stride = @intCast(std.mem.alignForward(usize, rows, tile_n)),
            .has_bias = @intFromBool(bias != null),
            .scale = scale,
        };
        try self.opBegin();
        var set = self.bind4Off(.{ w_buf, x.buf, y.buf, bias_buf }, .{ 0, x_off, y_off, 0 });
        self.d.CmdBindPipeline(self.cmd, .compute, if (dtype_f8) self.pipe_f8 else self.pipe_f32);
        self.d.CmdBindDescriptorSets(self.cmd, .compute, self.pipeline_layout, 0, 1, @ptrCast(&set), 0, null);
        self.d.CmdPushConstants(self.cmd, self.pipeline_layout, vk.ShaderStage.compute, 0, @sizeOf(Push), &push);
        self.d.CmdDispatch(
            self.cmd,
            @intCast(std.math.divCeil(usize, rows, wg_x * tile_n) catch unreachable),
            @intCast(std.math.divCeil(usize, m, wg_y * tile_m) catch unreachable),
            1,
        );
        try self.opEnd();
    }

    /// m=1 GEMV over the same cached transposed weights as opMatmul, split
    /// over `nchunk` interleaved k chunks (gemv_partial + gemv_combine) so
    /// the GPU sees rows*nchunk threads instead of the tiled GEMM's rows/8.
    /// `partials` must hold rows*nchunk f32; `y_off_elems` offsets the
    /// destination (the chunked LM head).
    pub fn opGemv(
        self: *Context,
        y: DeviceBuffer,
        y_off_elems: usize,
        x: DeviceBuffer,
        partials: DeviceBuffer,
        w_bytes: []const u8,
        wcode: WCode,
        rows: usize,
        cols: usize,
        scale: f32,
        nchunk: usize,
    ) Error!void {
        try self.opGemvPartial(x, partials, w_bytes, wcode, rows, cols, nchunk);
        try self.opGemvCombine(y, y_off_elems, partials, rows, scale, nchunk);
    }

    /// The two opGemv halves, exposed separately so callers can batch several
    /// GEMVs' partial passes (distinct partials buffers) into one
    /// `independent` group, then their combines into another.
    pub fn opGemvPartial(
        self: *Context,
        x: DeviceBuffer,
        partials: DeviceBuffer,
        w_bytes: []const u8,
        wcode: WCode,
        rows: usize,
        cols: usize,
        nchunk: usize,
    ) Error!void {
        std.debug.assert(rows % 4 == 0);
        const w_buf = try self.weightBuffer(w_bytes, wcode.esize(), rows, cols);
        const w_db: DeviceBuffer = .{ .buf = w_buf, .mem = .null_handle, .size = 0 };
        try self.opElt(.gemv_partial, w_db, x, null, partials, .{
            .u0 = @intCast((rows / 4) * nchunk),
            .u1 = @intCast(cols),
            .u2 = @intCast(nchunk),
            .u3 = @intCast(std.mem.alignForward(usize, rows, tile_n)),
            .u4 = @intFromEnum(wcode),
            .u5 = @intCast(rows),
        }, (rows / 4) * nchunk, 1, 1);
    }

    /// Fused dequant GEMV for GGUF block-quant weights (decode, m=1): one
    /// thread per output row dequants that row's blocks on the fly and dots
    /// them with x. y[rows] = scale * (dequant(W) @ x). cols % 32 == 0. The
    /// weight is uploaded raw (row-major blocks) via weightBufferRaw.
    pub fn opGemvQuant(
        self: *Context,
        dt: @import("tp_core").dtype.DType,
        y: DeviceBuffer,
        y_off: usize,
        x: DeviceBuffer,
        w_bytes: []const u8,
        scale: f32,
        rows: usize,
        cols: usize,
    ) Error!void {
        std.debug.assert(cols % 32 == 0);
        const entry: Elt = switch (dt) {
            .q8_0 => .gemv_q8_0,
            .q4_k => .gemv_q4_k,
            .q5_k => .gemv_q5_k,
            .q6_k => .gemv_q6_k,
            else => return error.UnsupportedDType,
        };
        const w_buf = try self.weightBufferRaw(w_bytes);
        const w_db: DeviceBuffer = .{ .buf = w_buf, .mem = .null_handle, .size = 0 };
        try self.opElt(entry, w_db, x, null, y, .{
            .u0 = @intCast(rows),
            .u1 = @intCast(cols),
            .u2 = @intCast(y_off),
            .f0 = scale,
        }, rows, 1, 1);
    }

    /// Coalesced + k-split block-quant GEMV: same math as opGemvQuant but
    /// reads the 32-row-group transposed weight (weightBufferRawT + a `*_t`
    /// kernel) so a warp touches consecutive bytes, and splits each row across
    /// `nchunk` superblock chunks so the GPU has enough warps to hide memory
    /// latency. `partials` holds rows*nchunk f32; gemv_combine reduces+scales.
    /// Output is bit-close to opGemvQuant (only the reduction is chunked).
    pub fn opGemvQuantT(
        self: *Context,
        dt: @import("tp_core").dtype.DType,
        y: DeviceBuffer,
        y_off: usize,
        x: DeviceBuffer,
        w_bytes: []const u8,
        scale: f32,
        rows: usize,
        cols: usize,
        nchunk: usize,
        partials: DeviceBuffer,
    ) Error!void {
        std.debug.assert(cols % 32 == 0);
        const row_bytes: usize = switch (dt) {
            .q8_0 => (cols / 32) * 34,
            .q4_k => (cols / 256) * 144,
            .q5_k => (cols / 256) * 176,
            .q6_k => (cols / 256) * 210,
            else => return error.UnsupportedDType,
        };
        const entry: Elt = switch (dt) {
            .q8_0 => .gemv_q8_0_t,
            .q4_k => .gemv_q4_k_t,
            .q5_k => .gemv_q5_k_t,
            .q6_k => .gemv_q6_k_t,
            else => return error.UnsupportedDType,
        };
        if (dt != .q8_0) std.debug.assert(cols % 256 == 0);
        const w_buf = try self.weightBufferRawT(w_bytes, row_bytes);
        const w_db: DeviceBuffer = .{ .buf = w_buf, .mem = .null_handle, .size = 0 };
        try self.opElt(entry, w_db, x, null, partials, .{
            .u0 = @intCast(rows * nchunk),
            .u1 = @intCast(cols),
            .u2 = @intCast(nchunk),
            .u3 = @intCast(rows),
        }, rows * nchunk, 1, 1);
        try self.opGemvCombine(y, y_off, partials, rows, scale, nchunk);
    }

    // --- qwen35 hybrid (gated DeltaNet) ops ---
    pub fn opL2NormRows(self: *Context, x: DeviceBuffer, rows: usize, dim: usize, eps: f32) Error!void {
        try self.opElt(.l2norm_rows, x, null, null, null, .{ .u0 = @intCast(rows), .u1 = @intCast(dim), .f0 = eps }, rows, 1, 1);
    }
    pub fn opDeinterleave2(self: *Context, qg: DeviceBuffer, q: DeviceBuffer, gate: DeviceBuffer, total: usize, hd: usize) Error!void {
        try self.opElt(.deinterleave2, qg, null, q, gate, .{ .u0 = @intCast(total), .u1 = @intCast(hd) }, total, 1, 1);
    }
    pub fn opGdnGates(self: *Context, alpha_beta: DeviceBuffer, a_dt: DeviceBuffer, out: DeviceBuffer, heads: usize) Error!void {
        try self.opElt(.gdn_gates, alpha_beta, a_dt, null, out, .{ .u0 = @intCast(heads) }, heads, 1, 1);
    }
    pub fn opGdnConvStep(self: *Context, conv_state: DeviceBuffer, x: DeviceBuffer, conv_w: DeviceBuffer, out: DeviceBuffer, channels: usize, taps: usize) Error!void {
        try self.opElt(.gdn_conv_step, conv_state, x, conv_w, out, .{ .u0 = @intCast(channels), .u1 = @intCast(taps) }, channels, 1, 1);
    }
    pub fn opGdnDeltaStep(self: *Context, state: DeviceBuffer, conv_out: DeviceBuffer, gates: DeviceBuffer, o: DeviceBuffer, heads: usize, d: usize, k_heads: usize, scale: f32) Error!void {
        try self.opElt(.gdn_delta_step, state, conv_out, gates, o, .{ .u0 = @intCast(heads), .u1 = @intCast(d), .u2 = @intCast(k_heads), .f0 = scale }, heads, 1, 1);
    }
    /// Partial rotate-half RoPE over the first rope_dim head dims (text-only
    /// decode). qk in place; freqs = cos then sin at sin_off. half = rope_dim/2.
    pub fn opRopeQwen35(self: *Context, qk: DeviceBuffer, freqs: DeviceBuffer, n_heads: usize, half: usize, sin_off: usize, head_dim: usize, pos: usize) Error!void {
        try self.opElt(.rope_qwen35, qk, null, freqs, null, .{ .u0 = @intCast(n_heads * half), .u1 = @intCast(half), .u2 = @intCast(sin_off), .u3 = @intCast(head_dim), .u4 = @intCast(pos) }, n_heads * half, 1, 1);
    }
    /// Causal GQA attention for one decode query (online softmax, one thread
    /// per query head). k/v caches: position j at j*(n_kv*hd) + kvh*hd.
    /// KV chunks per head in the flash-decoding split (attn_dsplit_gemma):
    /// turns the old one-thread-per-head attention into heads*nsplit threads.
    pub const attn_decode_nsplit = 64;

    /// Causal GQA decode attention (one query), flash-split across nsplit kv
    /// chunks then merged — parallelizes the naive one-thread-per-head kernel.
    /// `scratch` holds n_heads*attn_decode_nsplit*(head_dim+2) f32 partials.
    /// `window` (> 0) restricts to the last `window` keys; `ring` (> 0) enables
    /// sliding-window ring addressing (KV storage row = pos%ring, gemma3 LOCAL
    /// layers). 0 = full causal / linear.
    pub fn opAttnDecodeQ35(self: *Context, q: DeviceBuffer, k_cache: DeviceBuffer, v_cache: DeviceBuffer, out: DeviceBuffer, scratch: DeviceBuffer, n_heads: usize, n_kv: usize, head_dim: usize, kv_len: usize, scale: f32, window: usize, ring: usize, kv_end: usize, kv_fmt: KvFmt) Error!void {
        const nsplit = attn_decode_nsplit;
        const entry: Elt = switch (kv_fmt) {
            .f32 => .attn_dsplit_gemma,
            .f16 => .attn_dsplit_gemma_f16,
            .q8_0 => .attn_dsplit_gemma_q8,
        };
        // kv_end > kv_len marks a bidirectional image-block query (attends the
        // whole block forward); 0 (or == kv_len) is the plain causal decode bound.
        try self.opElt(entry, q, k_cache, v_cache, scratch, .{
            .u0 = @intCast(kv_len),
            .u1 = @intCast(n_heads),
            .u2 = @intCast(n_kv),
            .u3 = @intCast(head_dim),
            .u4 = nsplit,
            .u5 = @intCast(window),
            .f0 = scale,
            .f1 = @bitCast(@as(u32, @intCast(ring))),
            .u6 = @intCast(kv_end),
        }, n_heads * nsplit, 1, 1);
        try self.opElt(.attn_dmerge, scratch, null, null, out, .{
            .u0 = @intCast(n_heads),
            .u1 = @intCast(head_dim),
            .u2 = nsplit,
        }, n_heads * head_dim, 1, 1);
    }

    /// Convert-store `n` f32 K/V elements from `src` (+`src_off` elems) into the
    /// f16-packed cache `dst` at element offset `dst_off` (the f16-cache analogue
    /// of `opElt(.copy, ...)`). `n`, `dst_off`, and `src_off` are element counts;
    /// `n` and `dst_off` must be even (whole KV rows are) so each packed slot is
    /// written by exactly one thread. See kv_store_f16 in eltwise.zig.
    pub fn opStoreKvF16(self: *Context, dst: DeviceBuffer, dst_off: usize, src: DeviceBuffer, src_off: usize, n: usize) Error!void {
        try self.opElt(.kv_store_f16, src, dst, null, null, .{
            .u0 = @intCast(n / 2),
            .u2 = @intCast(dst_off / 2),
            .u3 = @intCast(src_off),
        }, n / 2, 1, 1);
    }

    /// Quantize-store `n` f32 K/V elements from `src` (+`src_off` elems) into
    /// the q8_0 block cache `dst` at element offset `dst_off`: one thread per
    /// PAIR of 32-element blocks (a lone 34-byte block ends mid-word — see
    /// kv_store_q8_0 in eltwise.zig). `n` and `dst_off` must be multiples of
    /// 64 (kv_dim always is).
    pub fn opStoreKvQ8(self: *Context, dst: DeviceBuffer, dst_off: usize, src: DeviceBuffer, src_off: usize, n: usize) Error!void {
        std.debug.assert(n % 64 == 0 and dst_off % 64 == 0);
        try self.opElt(.kv_store_q8_0, src, dst, null, null, .{
            .u0 = @intCast(n / 64),
            .u2 = @intCast(dst_off),
            .u3 = @intCast(src_off),
        }, n / 64, 1, 1);
    }

    pub fn opGemvCombine(
        self: *Context,
        y: DeviceBuffer,
        y_off_elems: usize,
        partials: DeviceBuffer,
        rows: usize,
        scale: f32,
        nchunk: usize,
    ) Error!void {
        try self.opElt(.gemv_combine, partials, null, null, y, .{
            .u0 = @intCast(rows),
            .u1 = @intCast(nchunk),
            .u2 = @intCast(y_off_elems),
            .f0 = scale,
        }, rows, 1, 1);
    }

    /// opGemvPartial for four input vectors at once (speculative-decode
    /// verify): each weight word is read once for all four inputs; results
    /// are bitwise equal to four single-input GEMVs. `x` must have 4 rows of
    /// backing store past `x_off_elems`; `partials` must hold 4*rows*nchunk
    /// f32. rows % 8 == 0.
    pub fn opGemvPartial4(
        self: *Context,
        x: DeviceBuffer,
        x_off_elems: usize,
        partials: DeviceBuffer,
        w_bytes: []const u8,
        wcode: WCode,
        rows: usize,
        cols: usize,
        nchunk: usize,
    ) Error!void {
        std.debug.assert(rows % 8 == 0);
        const w_buf = try self.weightBuffer(w_bytes, wcode.esize(), rows, cols);
        const w_db: DeviceBuffer = .{ .buf = w_buf, .mem = .null_handle, .size = 0 };
        try self.opElt(.gemv_partial4, w_db, x, null, partials, .{
            .u0 = @intCast((rows / 8) * nchunk),
            .u1 = @intCast(cols),
            .u2 = @intCast(nchunk),
            .u3 = @intCast(std.mem.alignForward(usize, rows, tile_n)),
            .u4 = @intFromEnum(wcode),
            .u5 = @intCast(rows),
            .f1 = @bitCast(@as(u32, @intCast(x_off_elems))),
        }, (rows / 8) * nchunk, 1, 1);
    }

    /// Reduce gemv_partial4 partials into `n` (1..4) live outputs, each at
    /// y_off + i*y_stride elements.
    pub fn opGemvCombine4(
        self: *Context,
        y: DeviceBuffer,
        y_off_elems: usize,
        y_stride_elems: usize,
        partials: DeviceBuffer,
        rows: usize,
        scale: f32,
        nchunk: usize,
        n: usize,
    ) Error!void {
        std.debug.assert(n >= 1 and n <= 4);
        try self.opElt(.gemv_combine4, partials, null, null, y, .{
            .u0 = @intCast(rows),
            .u1 = @intCast(nchunk),
            .u2 = @intCast(y_off_elems),
            .u3 = @intCast(y_stride_elems),
            .u4 = @intCast(n),
            .f0 = scale,
        }, n * rows, 1, 1);
    }

    /// Tensor-core GEMM: the kernel reads the cached raw fp8 k-major weights
    /// directly and decodes e4m3 -> f16 in its staging loop; the dequant
    /// scale is folded into the f32 -> f16 activation conversion. `m_pad`
    /// (multiple of 128) rows are computed; `y` must hold m_pad rows. rows
    /// must be a multiple of 128 (n tile), cols of 64 (k slab), stride == rows.
    pub fn opMatmulCoop(
        self: *Context,
        y: DeviceBuffer,
        x: DeviceBuffer,
        m: usize,
        m_pad: usize,
        w_bytes: []const u8,
        rows: usize,
        cols: usize,
        scale: f32,
    ) Error!void {
        std.debug.assert(m_pad % 128 == 0);
        // Two dependent recorded ops — must not sit inside an `independent`
        // group, which would elide the barrier between them.
        std.debug.assert(self.indep_remaining == 0);
        try self.ensureDeviceBuffer(&self.x_h16, m_pad * cols * 2);

        // x f32 -> f16 with the weight scale folded in (zero pad rows).
        try self.opElt(.f32_to_h16, x, null, null, self.x_h16, .{
            .u0 = @intCast(m_pad * cols / 2),
            .u1 = @intCast(m * cols),
            .f0 = scale,
        }, m_pad * cols / 2, 1, 1);
        try self.opMatmulCoopH16(y, self.x_h16, m_pad, w_bytes, rows, cols, false);
    }

    /// Tensor-core GEMM whose f16 activations were already produced (the
    /// fused gate kernels fold the weight scale in themselves): just the
    /// coop dispatch on the cached raw fp8 weights. With `c16` the C tile
    /// stores half-precision (y is an f16 buffer) — exact under f16
    /// accumulators; callers must check pipe_coop_c16 exists.
    pub fn opMatmulCoopH16(
        self: *Context,
        y: DeviceBuffer,
        x16: DeviceBuffer,
        m_pad: usize,
        w_bytes: []const u8,
        rows: usize,
        cols: usize,
        c16: bool,
    ) Error!void {
        std.debug.assert(if (c16) self.pipe_coop_c16 != .null_handle else self.pipe_coop != .null_handle);
        std.debug.assert(rows % coopmat.coop_wgn == 0 and cols % 64 == 0 and m_pad % 128 == 0);
        const w_buf = try self.weightBuffer(w_bytes, 1, rows, cols);
        const push: [4]u32 = .{ @intCast(m_pad), @intCast(rows), @intCast(cols), @intCast(rows) };
        try self.opBegin();
        // The kernel stages both operands through workgroup memory via
        // uvec4 views: binding 0 = activations, binding 3 = raw fp8
        // weights (binding 1 is unused; the layout still needs a buffer).
        var set = self.bind4(.{ x16.buf, x16.buf, y.buf, w_buf });
        self.d.CmdBindPipeline(self.cmd, .compute, if (c16) self.pipe_coop_c16 else self.pipe_coop);
        self.d.CmdBindDescriptorSets(self.cmd, .compute, self.pipeline_layout, 0, 1, @ptrCast(&set), 0, null);
        self.d.CmdPushConstants(self.cmd, self.pipeline_layout, vk.ShaderStage.compute, 0, 16, &push);
        self.d.CmdDispatch(self.cmd, @intCast(rows / coopmat.coop_wgn), @intCast(m_pad / 128), 1);
        try self.opEnd();
    }

    /// Raw int8 tensor-core GEMM: `y(s32)[m_pad][rows] = x(s8)[m_pad][cols] @ W^T`,
    /// where `w_bytes` is the raw int8 weight [rows][cols] (transposed to k-major
    /// on upload, cached by pointer — same 1-byte path as fp8). No dequant scale
    /// here; the caller scales the s32 result by act/weight per-row scales.
    /// `x_i8` is row-major s8 [m_pad][cols]; `y` holds m_pad*rows s32.
    /// m_pad/rows multiples of 16, cols a multiple of 32.
    pub fn opMatmulCoopI8(
        self: *Context,
        y: DeviceBuffer,
        x_i8: DeviceBuffer,
        m_pad: usize,
        w_bytes: []const u8,
        rows: usize,
        cols: usize,
    ) Error!void {
        const mtile = 16 * coopmat.i8_mt;
        const ntile = 16 * coopmat.i8_nt;
        std.debug.assert(self.pipe_coop_i8 != .null_handle);
        std.debug.assert(m_pad % mtile == 0 and rows % ntile == 0 and cols % 32 == 0);
        const w_buf = try self.weightBuffer(w_bytes, 1, rows, cols);
        const w_stride: u32 = @intCast(std.mem.alignForward(usize, rows, tile_n));
        const push: [4]u32 = .{ @intCast(m_pad), @intCast(rows), @intCast(cols), w_stride };
        // Prefer the shared-memory kernel when the shape fits its 128x128 wg
        // tile / 64-deep k step (all DiT-block shapes qualify); else the
        // register-tiled kernel (small shapes, the standalone check cases).
        const use_sh = self.pipe_coop_i8_sh != .null_handle and
            m_pad % 128 == 0 and rows % 128 == 0 and cols % 64 == 0 and w_stride % 4 == 0;
        try self.opBegin();
        var set = self.bind4(.{ w_buf, x_i8.buf, y.buf, try self.dummyBuf() });
        if (use_sh) {
            self.d.CmdBindPipeline(self.cmd, .compute, self.pipe_coop_i8_sh);
            self.d.CmdBindDescriptorSets(self.cmd, .compute, self.pipeline_layout, 0, 1, @ptrCast(&set), 0, null);
            self.d.CmdPushConstants(self.cmd, self.pipeline_layout, vk.ShaderStage.compute, 0, 16, &push);
            self.d.CmdDispatch(self.cmd, @intCast(rows / 128), @intCast(m_pad / 128), 1);
        } else {
            self.d.CmdBindPipeline(self.cmd, .compute, self.pipe_coop_i8);
            self.d.CmdBindDescriptorSets(self.cmd, .compute, self.pipeline_layout, 0, 1, @ptrCast(&set), 0, null);
            self.d.CmdPushConstants(self.cmd, self.pipeline_layout, vk.ShaderStage.compute, 0, 16, &push);
            self.d.CmdDispatch(self.cmd, @intCast(rows / ntile), @intCast(m_pad / mtile), 1);
        }
        try self.opEnd();
    }

    /// Stage A: shared int8 GEMM with fused rescale — `y(f32)[m_pad][rows] =
    /// (x @ W^T) * act_scale[row] * weight_scale[col]` in one kernel (no s32 acc
    /// buffer, no scale_i32 pass). `scale_buf` holds [act(m_pad) | weight(rows)]
    /// f32 (act at index 0, weight at index m_pad). Shape must fit the shared
    /// kernel (m_pad/rows %128, cols %64). Returns false if unavailable/unfit so
    /// the caller can fall back to the s32 GEMM + scale_i32 path.
    pub fn opMatmulCoopI8Fused(
        self: *Context,
        y: DeviceBuffer,
        x_i8: DeviceBuffer,
        m_pad: usize,
        w_bytes: []const u8,
        rows: usize,
        cols: usize,
        scale_buf: vk.Buffer,
        c_h16: bool,
    ) Error!bool {
        const pipe = if (c_h16) self.pipe_coop_i8_fs16 else self.pipe_coop_i8_fs;
        const w_stride: u32 = @intCast(std.mem.alignForward(usize, rows, tile_n));
        if (pipe == .null_handle or
            m_pad % 128 != 0 or rows % 128 != 0 or cols % 64 != 0 or w_stride % 4 != 0)
            return false;
        const w_buf = try self.weightBuffer(w_bytes, 1, rows, cols);
        const push: [4]u32 = .{ @intCast(m_pad), @intCast(rows), @intCast(cols), w_stride };
        try self.opBegin();
        var set = self.bind4(.{ w_buf, x_i8.buf, y.buf, scale_buf });
        self.d.CmdBindPipeline(self.cmd, .compute, pipe);
        self.d.CmdBindDescriptorSets(self.cmd, .compute, self.pipeline_layout, 0, 1, @ptrCast(&set), 0, null);
        self.d.CmdPushConstants(self.cmd, self.pipeline_layout, vk.ShaderStage.compute, 0, 16, &push);
        self.d.CmdDispatch(self.cmd, @intCast(rows / 128), @intCast(m_pad / 128), 1);
        try self.opEnd();
        return true;
    }

    /// Upload the resident group-Hadamard (256x256 f32) once for the int8 path.
    fn ensureHadamard(self: *Context) Error!void {
        if (self.i8_hadamard.buf != .null_handle) return;
        const bytes = std.mem.asBytes(&convrot.H);
        try self.ensureDeviceBuffer(&self.i8_hadamard, bytes.len);
        try self.tensorUpload(self.i8_hadamard, bytes);
    }

    /// Full int8 (convrot) linear: `y[m][rows] = x[m][cols] @ W^T`, where
    /// `w_bytes` is the raw int8 weight [rows][cols] (pre-rotated `W_rot`) and
    /// `weight_scale` its per-output-row scale [rows]. Rotates x by the group
    /// Hadamard, dynamically quantizes it per row to int8, runs the tensor-core
    /// s8*s8->s32 GEMM, then rescales by act_scale*weight_scale. `cols` a
    /// multiple of 256 (rotation group), `rows` a multiple of 16.
    /// int8 activation prep: rotate x by the group Hadamard, dynamically
    /// quantize per row to int8, leaving the packed int8 activation (i8_x) and
    /// per-row scale (i8_scale) on the device for one or more opI8Gemm calls
    /// (the DiT shares one prepped activation across wq/wk/wv/gate and mlp
    /// gate/up). `cols` a multiple of 256.
    pub fn opI8Prep(self: *Context, x: DeviceBuffer, m: usize, cols: usize) Error!void {
        std.debug.assert(cols % convrot.group_size == 0);
        const ng = cols / convrot.group_size;
        // Pad to 128 rows when the shared-mem GEMM is enabled (its wg tile is
        // 128x128); still a multiple of the register kernel's 64-row tile.
        const m_align: usize = if (coopmat.i8_shared) 128 else 16 * coopmat.i8_mt;
        const m_pad = std.mem.alignForward(usize, m, m_align);
        self.i8_m = m;
        self.i8_cols = cols;
        self.i8_mpad = m_pad;
        try self.ensureDeviceBuffer(&self.i8_scale, m * 4);
        try self.ensureDeviceBuffer(&self.i8_x, m_pad * cols); // 1 byte/elem

        // Stage B: one fused kernel (rotate FWHT + rowmax + quantize in f16
        // shared) replaces the 3-pass chain + its xr round-trip. One build per
        // convrot cols (6144 for qkv/wo/mlp-gu, 16384 for mlp.down).
        const prep_pipe: vk.Pipeline = switch (cols) {
            6144 => self.pipe_i8_prep6144,
            16384 => self.pipe_i8_prep16384,
            else => .null_handle,
        };
        if (prep_pipe != .null_handle) {
            try self.opBegin();
            var set = self.bind4(.{ x.buf, self.i8_x.buf, self.i8_scale.buf, try self.dummyBuf() });
            self.d.CmdBindPipeline(self.cmd, .compute, prep_pipe);
            self.d.CmdBindDescriptorSets(self.cmd, .compute, self.pipeline_layout, 0, 1, @ptrCast(&set), 0, null);
            self.d.CmdDispatch(self.cmd, @intCast(m), 1, 1);
            try self.opEnd();
            return;
        }

        try self.ensureDeviceBuffer(&self.i8_xr, m * cols * 4);
        try self.ensureDeviceBuffer(&self.i8_partials, m * ng * 4);
        try self.opElt(.rotate_fwht, x, self.i8_xr, null, self.i8_partials, .{
            .u0 = @intCast(m * ng),
        }, m * ng, 1, 1);
        try self.opElt(.rowscale_i8, self.i8_partials, self.i8_scale, null, null, .{
            .u0 = @intCast(m),
            .u1 = @intCast(ng),
        }, m, 1, 1);
        try self.opElt(.quantize_i8, self.i8_xr, self.i8_x, null, self.i8_scale, .{
            .u0 = @intCast(m_pad * cols / 4),
            .u1 = @intCast(cols),
            .u2 = @intCast(m * cols),
        }, m_pad * cols / 4, 1, 1);
    }

    /// int8 GEMM + rescale against the last opI8Prep activation: `y[m][rows] =
    /// x_prepped @ W^T`, `w_bytes` the raw int8 [rows][cols], `weight_scale`
    /// its per-row scale. `rows` a multiple of 16*i8_nt.
    /// `c_h16` stores y half-precision (feeds the fused f16 attention chain);
    /// only honored on the fused shared path (falls back to f32 s32+scale_i32).
    pub fn opI8Gemm(self: *Context, y: DeviceBuffer, w_bytes: []const u8, weight_scale: []const f32, rows: usize, c_h16: bool) Error!void {
        std.debug.assert(self.pipe_coop_i8 != .null_handle);
        std.debug.assert(rows % (16 * coopmat.i8_nt) == 0);
        const m = self.i8_m;
        const ws_buf: DeviceBuffer = .{
            .buf = try self.smallBuffer(std.mem.sliceAsBytes(weight_scale)),
            .mem = .null_handle,
            .size = 0,
        };
        // Stage A: fused rescale — assemble [act(m_pad) | weight(rows)] into one
        // scale buffer (batch-barrier-safe scale_concat dispatch) and run the
        // GEMM that stores y directly (f16 if c_h16), no s32 acc / scale_i32 pass.
        if (self.pipe_coop_i8_fs != .null_handle and
            self.i8_mpad % 128 == 0 and rows % 128 == 0 and self.i8_cols % 64 == 0)
        {
            const total = self.i8_mpad + rows;
            try self.ensureDeviceBuffer(&self.i8_scalecat, total * 4);
            try self.opElt(.scale_concat, self.i8_scale, self.i8_scalecat, ws_buf, null, .{
                .u0 = @intCast(total),
                .u1 = @intCast(m),
                .u2 = @intCast(self.i8_mpad),
            }, total, 1, 1);
            if (try self.opMatmulCoopI8Fused(y, self.i8_x, self.i8_mpad, w_bytes, rows, self.i8_cols, self.i8_scalecat.buf, c_h16))
                return;
        }
        // Fallback: s32 GEMM + separate scale_i32 pass.
        try self.ensureDeviceBuffer(&self.i8_acc, self.i8_mpad * rows * 4);
        try self.opMatmulCoopI8(self.i8_acc, self.i8_x, self.i8_mpad, w_bytes, rows, self.i8_cols);
        try self.opElt(.scale_i32, self.i8_acc, y, ws_buf, self.i8_scale, .{
            .u0 = @intCast(m * rows),
            .u1 = @intCast(rows),
        }, m * rows, 1, 1);
    }

    /// One of the 4 int8 accumulator buffers (round-robin), so a group of
    /// GEMMs sharing the prepped activation can use distinct accs and overlap.
    pub fn i8AccPool(self: *Context, i: usize) *DeviceBuffer {
        return switch (i & 3) {
            0 => &self.i8_acc,
            1 => &self.i8_acc1,
            2 => &self.i8_acc2,
            else => &self.i8_acc3,
        };
    }

    /// int8 GEMM into a caller-chosen accumulator (records a single dispatch,
    /// so a set of these can run inside one `independent` group). Pairs with
    /// opI8Scale, which reads the same acc after the group's barrier.
    pub fn opI8GemmRaw(self: *Context, acc: *DeviceBuffer, w_bytes: []const u8, rows: usize) Error!void {
        std.debug.assert(rows % (16 * coopmat.i8_nt) == 0);
        try self.ensureDeviceBuffer(acc, self.i8_mpad * rows * 4);
        try self.opMatmulCoopI8(acc.*, self.i8_x, self.i8_mpad, w_bytes, rows, self.i8_cols);
    }

    /// Rescale an int8 GEMM accumulator: y = acc * act_scale * weight_scale.
    pub fn opI8Scale(self: *Context, y: DeviceBuffer, acc: DeviceBuffer, weight_scale: []const f32, rows: usize) Error!void {
        const ws_buf: DeviceBuffer = .{
            .buf = try self.smallBuffer(std.mem.sliceAsBytes(weight_scale)),
            .mem = .null_handle,
            .size = 0,
        };
        try self.opElt(.scale_i32, acc, y, ws_buf, self.i8_scale, .{
            .u0 = @intCast(self.i8_m * rows),
            .u1 = @intCast(rows),
        }, self.i8_m * rows, 1, 1);
    }

    /// Full int8 (convrot) linear (prep + one GEMM); the standalone/bench path.
    pub fn opMatmulI8(
        self: *Context,
        y: DeviceBuffer,
        x: DeviceBuffer,
        m: usize,
        w_bytes: []const u8,
        weight_scale: []const f32,
        rows: usize,
        cols: usize,
    ) Error!void {
        std.debug.assert(self.indep_remaining == 0);
        try self.opI8Prep(x, m, cols);
        try self.opI8Gemm(y, w_bytes, weight_scale, rows, false);
    }

    /// Device f16 k-major weight buffer converted from tight f32
    /// [rows][cols] on the CPU (element (k, col) at k * n_pad + col; zeros
    /// in both pads so the GEMM's padded tiles contribute nothing).
    /// Uploaded once and cached by host pointer; VAE conv weights are a few
    /// MB each so the CPU pass is trivial next to the decode.
    fn weightBufferF16From32(self: *Context, w: []const f32, rows: usize, cols: usize) Error!vk.Buffer {
        const key = @intFromPtr(w.ptr);
        self.use_counter += 1;
        if (self.weights.getPtr(key)) |e| {
            e.last_use = self.use_counter;
            return e.db.buf;
        }

        const n_pad = std.mem.alignForward(usize, rows, 128);
        const k_pad = std.mem.alignForward(usize, cols, 64);
        const half = self.gpa.alloc(u16, k_pad * n_pad) catch return error.OutOfMemory;
        defer self.gpa.free(half);
        @memset(half, 0);
        for (0..rows) |r| {
            for (0..cols) |k| {
                half[k * n_pad + r] = @bitCast(@as(f16, @floatCast(w[r * cols + k])));
            }
        }
        const bytes = std.mem.sliceAsBytes(half);

        self.reserveForWeights(bytes.len);
        const db = try self.createBuffer(
            bytes.len,
            vk.BufferUsage.storage_buffer | vk.BufferUsage.transfer_dst,
            vk.MemoryProperty.device_local,
        );
        errdefer {
            self.d.DestroyBuffer(self.device, db.buf, null);
            self.d.FreeMemory(self.device, db.mem, null);
        }
        // Chunked staging upload + visibility barrier (same recipe as
        // weightBuffer, minus the transpose — the layout is built on the CPU).
        const cb = self.nowCmd();
        const chunk: u64 = 256 << 20;
        const mapped = try self.ensureHostBuffer(&self.staging, @min(bytes.len, chunk));
        var off: u64 = 0;
        while (off < bytes.len) {
            const n: u64 = @min(chunk, bytes.len - off);
            @memcpy(mapped[0..@intCast(n)], bytes[@intCast(off)..][0..@intCast(n)]);
            try self.beginCmdBuf(cb);
            const region: vk.BufferCopy = .{ .src_offset = 0, .dst_offset = off, .size = n };
            self.d.CmdCopyBuffer(cb, self.staging.buf, db.buf, 1, @ptrCast(&region));
            const to_shader: vk.BufferMemoryBarrier = .{
                .src_access_mask = vk.Access.transfer_write,
                .dst_access_mask = vk.Access.shader_read,
                .buffer = db.buf,
                .offset = 0,
                .size = vk.WHOLE_SIZE,
            };
            self.d.CmdPipelineBarrier(cb, vk.PipelineStage.transfer, vk.PipelineStage.compute_shader, 0, 0, null, 1, @ptrCast(&to_shader), 0, null);
            try self.submitAndWaitBuf(cb);
            off += n;
        }

        try self.weights.put(self.gpa, key, .{ .db = db, .last_use = self.use_counter, .pinned = self.pinNew(bytes.len) });
        return db.buf;
    }

    /// Like weightBufferF16From32, but the source is raw bf16 bytes (a dense
    /// bf16 DiT checkpoint) and the bf16 -> f16 conversion + k-major transpose
    /// runs ON THE GPU (bf16_to_f16_coopw): raw bf16 streams to the device once,
    /// then one compute dispatch produces the [k_pad][n_pad] f16 layout the
    /// f16-weight coop GEMM reads — no single-threaded CPU conversion loop (the
    /// bottleneck when a >VRAM model re-streams weights every step). f16 keeps
    /// more mantissa bits than bf16, so the round trip is near-lossless for the
    /// < 1 weight range. Cached by host pointer, same residency accounting.
    fn weightBufferF16FromBf16(self: *Context, w_bytes: []const u8, rows: usize, cols: usize, native: bool) Error!vk.Buffer {
        const key = @intFromPtr(w_bytes.ptr);
        self.use_counter += 1;
        if (self.weights.getPtr(key)) |e| {
            e.last_use = self.use_counter;
            return e.db.buf;
        }
        std.debug.assert(w_bytes.len == rows * cols * 2);
        // native: keep the weight bf16 (pipe_tr_bf16raw) for the bf16 coop GEMM;
        // else convert to f16 (pipe_tr_bf16w) for the f16 fallback. Both emit a
        // 16-bit k-major [k_pad][n_pad] buffer, so only the pipeline differs.
        const tr_pipe = if (native) self.pipe_tr_bf16raw else self.pipe_tr_bf16w;

        const n_pad = std.mem.alignForward(usize, rows, 128);
        const k_pad = std.mem.alignForward(usize, cols, 64);
        const total: u64 = @as(u64, k_pad) * n_pad * 2; // f16 out
        self.reserveForWeights(total);
        const db = try self.createBuffer(
            total,
            vk.BufferUsage.storage_buffer | vk.BufferUsage.transfer_dst,
            vk.MemoryProperty.device_local,
        );
        errdefer {
            self.d.DestroyBuffer(self.device, db.buf, null);
            self.d.FreeMemory(self.device, db.mem, null);
        }

        // Raw bf16 -> device scratch (chunked through staging), then the
        // convert+transpose dispatch builds the coop f16 layout device-side.
        const cb = self.nowCmd();
        const raw_size = std.mem.alignForward(u64, w_bytes.len, 4);
        try self.ensureDeviceBuffer(&self.raw_dev, raw_size);
        const chunk: u64 = 256 << 20;
        const mapped = try self.ensureHostBuffer(&self.staging, @min(raw_size, chunk));
        var off: u64 = 0;
        while (off < w_bytes.len) {
            const n: u64 = @min(chunk, w_bytes.len - off);
            @memcpy(mapped[0..@intCast(n)], w_bytes[@intCast(off)..][0..@intCast(n)]);
            try self.beginCmdBuf(cb);
            const region: vk.BufferCopy = .{ .src_offset = 0, .dst_offset = off, .size = n };
            self.d.CmdCopyBuffer(cb, self.staging.buf, self.raw_dev.buf, 1, @ptrCast(&region));
            try self.submitAndWaitBuf(cb);
            off += n;
        }

        {
            const buf_infos = [2]vk.DescriptorBufferInfo{
                .{ .buffer = self.raw_dev.buf },
                .{ .buffer = db.buf },
            };
            var writes: [2]vk.WriteDescriptorSet = undefined;
            for (&writes, 0..) |*wr, i| {
                wr.* = .{
                    .dst_set = self.desc_set_tr,
                    .dst_binding = @intCast(i),
                    .dst_array_element = 0,
                    .descriptor_count = 1,
                    .descriptor_type = .storage_buffer,
                    .p_image_info = null,
                    .p_buffer_info = @ptrCast(&buf_infos[i]),
                    .p_texel_buffer_view = null,
                };
            }
            self.d.UpdateDescriptorSets(self.device, writes.len, &writes, 0, null);

            const push: TransposePush = .{
                .rows = @intCast(rows),
                .cols = @intCast(cols),
                .stride = @intCast(n_pad),
                .k_pad = @intCast(k_pad),
            };
            try self.beginCmdBuf(cb);
            self.d.CmdBindPipeline(cb, .compute, tr_pipe);
            self.d.CmdBindDescriptorSets(cb, .compute, self.pipeline_layout_tr, 0, 1, @ptrCast(&self.desc_set_tr), 0, null);
            self.d.CmdPushConstants(cb, self.pipeline_layout_tr, vk.ShaderStage.compute, 0, @sizeOf(TransposePush), &push);
            self.d.CmdDispatch(
                cb,
                @intCast(std.math.divCeil(usize, n_pad / 2, tr_wg_x) catch unreachable),
                @intCast(std.math.divCeil(usize, k_pad, tr_wg_y) catch unreachable),
                1,
            );
            const to_shader: vk.BufferMemoryBarrier = .{
                .src_access_mask = vk.Access.shader_write,
                .dst_access_mask = vk.Access.shader_read,
                .buffer = db.buf,
                .offset = 0,
                .size = vk.WHOLE_SIZE,
            };
            self.d.CmdPipelineBarrier(cb, vk.PipelineStage.compute_shader, vk.PipelineStage.compute_shader, 0, 0, null, 1, @ptrCast(&to_shader), 0, null);
            try self.submitAndWaitBuf(cb);
        }

        try self.weights.put(self.gpa, key, .{ .db = db, .last_use = self.use_counter, .pinned = self.pinNew(total) });
        return db.buf;
    }

    /// Tensor-core GEMM for f32 weights (the VAE convs): B converts once to
    /// zero-padded k-major f16 (cached), the f32 activations convert per
    /// call with the k tail padded (f32_to_h16_pad, since 9*ci is rarely a
    /// multiple of 64), C lands in the column-padded y_pad scratch, and
    /// bias_compact strips the pad and adds the conv bias into `y` at
    /// element offset `y_off`.
    pub fn opMatmulCoopF16W(
        self: *Context,
        y: DeviceBuffer,
        y_off: usize,
        x: DeviceBuffer,
        m: usize,
        w: []const f32,
        rows: usize,
        cols: usize,
        bias: []const f32,
    ) Error!void {
        std.debug.assert(self.pipe_coop_f16w != .null_handle);
        const w_buf = try self.weightBufferF16From32(w, rows, cols);
        return self.coopF16WDispatch(y, y_off, x, m, w_buf, rows, cols, bias, false);
    }

    /// Same f16-weight tensor-core GEMM as opMatmulCoopF16W, but the weight is a
    /// raw bf16 tensor (a dense bf16 DiT block linear). bf16 -> f16 conversion
    /// happens once at upload (weightBufferF16FromBf16, cached); everything else
    /// — f16 activations, coop dispatch, bias-compact — is identical, so the
    /// f32 output matches opMatmulCoop's contract and the bf16 DiT can reuse the
    /// backend's f32-operand attention/mlp path unchanged.
    pub fn opMatmulCoopF16Wb(
        self: *Context,
        y: DeviceBuffer,
        y_off: usize,
        x: DeviceBuffer,
        m: usize,
        w_bytes: []const u8,
        rows: usize,
        cols: usize,
        bias: []const f32,
    ) Error!void {
        std.debug.assert(self.pipe_coop_f16w != .null_handle);
        const w_buf = try self.weightBufferF16FromBf16(w_bytes, rows, cols, false);
        return self.coopF16WDispatch(y, y_off, x, m, w_buf, rows, cols, bias, false);
    }

    /// Native bf16 twin of opMatmulCoopF16Wb: the raw bf16 weight is kept bf16
    /// (k-major transpose only) and the activation converts f32 -> bf16, so the
    /// bf16 tensor cores consume both operands directly — no f16 round trip.
    /// Requires a bf16 cooperative-matrix config (pipe_coop_bf16w).
    pub fn opMatmulCoopBf16(
        self: *Context,
        y: DeviceBuffer,
        y_off: usize,
        x: DeviceBuffer,
        m: usize,
        w_bytes: []const u8,
        rows: usize,
        cols: usize,
        bias: []const f32,
    ) Error!void {
        std.debug.assert(self.pipe_coop_bf16w != .null_handle);
        const w_buf = try self.weightBufferF16FromBf16(w_bytes, rows, cols, true);
        return self.coopF16WDispatch(y, y_off, x, m, w_buf, rows, cols, bias, true);
    }

    /// Shared body of the f16/bf16-weight coop GEMM: the weight is already
    /// uploaded as k-major padded 16-bit (`w_buf`); convert x to f16 (or bf16
    /// when `bf16_ab`), run the coop dispatch into the padded scratch, then strip
    /// the pad and add `bias` into `y`.
    fn coopF16WDispatch(
        self: *Context,
        y: DeviceBuffer,
        y_off: usize,
        x: DeviceBuffer,
        m: usize,
        w_buf: vk.Buffer,
        rows: usize,
        cols: usize,
        bias: []const f32,
        bf16_ab: bool,
    ) Error!void {
        const n_pad = std.mem.alignForward(usize, rows, 128);
        const k_pad = std.mem.alignForward(usize, cols, 64);
        const m_pad = std.mem.alignForward(usize, m, 128);
        try self.ensureDeviceBuffer(&self.x_h16, m_pad * k_pad * 2);
        try self.ensureDeviceBuffer(&self.y_pad, m_pad * n_pad * 4);

        // x f32 [m][cols] -> f16/bf16 [m_pad][k_pad], zeros in both pads.
        try self.opElt(if (bf16_ab) .f32_to_bf16_pad else .f32_to_h16_pad, x, null, null, self.x_h16, .{
            .u0 = @intCast(m_pad * k_pad / 2),
            .u1 = @intCast(cols),
            .u2 = @intCast(k_pad),
            .u3 = @intCast(m),
            .f0 = 1.0,
        }, m_pad * k_pad / 2, 1, 1);

        const push: [4]u32 = .{ @intCast(m_pad), @intCast(n_pad), @intCast(k_pad), @intCast(n_pad) };
        try self.opBegin();
        var set = self.bind4(.{ self.x_h16.buf, self.x_h16.buf, self.y_pad.buf, w_buf });
        self.d.CmdBindPipeline(self.cmd, .compute, if (bf16_ab) self.pipe_coop_bf16w else self.pipe_coop_f16w);
        self.d.CmdBindDescriptorSets(self.cmd, .compute, self.pipeline_layout, 0, 1, @ptrCast(&set), 0, null);
        self.d.CmdPushConstants(self.cmd, self.pipeline_layout, vk.ShaderStage.compute, 0, 16, &push);
        self.d.CmdDispatch(self.cmd, @intCast(n_pad / 128), @intCast(m_pad / 128), 1);
        try self.opEnd();

        const bias_buf: DeviceBuffer = .{
            .buf = try self.smallBuffer(std.mem.sliceAsBytes(bias)),
            .mem = .null_handle,
            .size = 0,
        };
        try self.opElt(.bias_compact, self.y_pad, bias_buf, null, y, .{
            .u0 = @intCast(m * rows),
            .u1 = @intCast(rows),
            .u2 = @intCast(n_pad),
            .u3 = @intCast(y_off),
        }, m * rows, 1, 1);
    }

    /// Attention-scores tensor-core GEMM (coopmat.buildGemmScores): grid is
    /// (j_tiles, q_tiles, heads_in_batch); push carries the strides/head
    /// offsets (see the kernel doc). Requires head_dim == 128.
    pub fn opAttnScores(
        self: *Context,
        s: DeviceBuffer,
        q16: DeviceBuffer,
        k16: DeviceBuffer,
        push: EltPush,
        gx: usize,
        gy: usize,
        gz: usize,
    ) Error!void {
        return self.opAttnScoresPipe(self.pipe_scores, s, q16, k16, push, gx, gy, gz);
    }

    /// head_dim-384 variant (VAE mid-block attention).
    pub fn opAttnScoresVae(
        self: *Context,
        s: DeviceBuffer,
        q16: DeviceBuffer,
        k16: DeviceBuffer,
        push: EltPush,
        gx: usize,
        gy: usize,
        gz: usize,
    ) Error!void {
        return self.opAttnScoresPipe(self.pipe_scores_vae, s, q16, k16, push, gx, gy, gz);
    }

    fn opAttnScoresPipe(
        self: *Context,
        pipe: vk.Pipeline,
        s: DeviceBuffer,
        q16: DeviceBuffer,
        k16: DeviceBuffer,
        push: EltPush,
        gx: usize,
        gy: usize,
        gz: usize,
    ) Error!void {
        std.debug.assert(pipe != .null_handle);
        try self.opBegin();
        // Binding 3 is the same S buffer viewed as u32 (coalesced copy-out).
        var set = self.bind4(.{ k16.buf, q16.buf, s.buf, s.buf });
        self.d.CmdBindPipeline(self.cmd, .compute, pipe);
        self.d.CmdBindDescriptorSets(self.cmd, .compute, self.pipeline_layout_e, 0, 1, @ptrCast(&set), 0, null);
        self.d.CmdPushConstants(self.cmd, self.pipeline_layout_e, vk.ShaderStage.compute, 0, @sizeOf(EltPush), &push);
        self.d.CmdDispatch(self.cmd, @intCast(gx), @intCast(gy), @intCast(gz));
        try self.opEnd();
    }

    /// Flash-attention pass (coopmat.buildFlashAttn): grid is
    /// (1, q_tiles, heads). `out` carries both the f32 attention output and
    /// the MD table at the push u5 offset; `v16` is unused by the md pass
    /// (any valid buffer).
    pub fn opFlash(
        self: *Context,
        which: enum { md, out },
        q16: DeviceBuffer,
        k16: DeviceBuffer,
        v16: DeviceBuffer,
        out: DeviceBuffer,
        push: EltPush,
        gy: usize,
        gz: usize,
    ) Error!void {
        const pipe = switch (which) {
            .md => self.pipe_flash_md,
            .out => self.pipe_flash_out,
        };
        std.debug.assert(pipe != .null_handle);
        try self.opBegin();
        var set = self.bind4(.{ k16.buf, q16.buf, out.buf, v16.buf });
        self.d.CmdBindPipeline(self.cmd, .compute, pipe);
        self.d.CmdBindDescriptorSets(self.cmd, .compute, self.pipeline_layout_e, 0, 1, @ptrCast(&set), 0, null);
        self.d.CmdPushConstants(self.cmd, self.pipeline_layout_e, vk.ShaderStage.compute, 0, @sizeOf(EltPush), &push);
        self.d.CmdDispatch(self.cmd, 1, @intCast(gy), @intCast(gz));
        try self.opEnd();
    }

    /// Attention P@V tensor-core GEMM (coopmat.buildGemmAttnOut): grid is
    /// (1, q_tiles, heads_in_batch); P is computed from raw scores + the
    /// two-pass softmax {m, 1/d} buffer during staging.
    pub fn opAttnOut(
        self: *Context,
        s: DeviceBuffer,
        v16: DeviceBuffer,
        out: DeviceBuffer,
        md: DeviceBuffer,
        push: EltPush,
        gy: usize,
        gz: usize,
    ) Error!void {
        std.debug.assert(self.pipe_attn_out != .null_handle);
        try self.opBegin();
        var set = self.bind4(.{ s.buf, v16.buf, out.buf, md.buf });
        self.d.CmdBindPipeline(self.cmd, .compute, self.pipe_attn_out);
        self.d.CmdBindDescriptorSets(self.cmd, .compute, self.pipeline_layout_e, 0, 1, @ptrCast(&set), 0, null);
        self.d.CmdPushConstants(self.cmd, self.pipeline_layout_e, vk.ShaderStage.compute, 0, @sizeOf(EltPush), &push);
        self.d.CmdDispatch(self.cmd, 1, @intCast(gy), @intCast(gz));
        try self.opEnd();
    }

    /// Dispatch one eltwise/attention kernel over device tensors. Buffers
    /// map to bindings a..d; pass null for unused slots.
    pub fn opElt(
        self: *Context,
        which: Elt,
        a: ?DeviceBuffer,
        b: ?DeviceBuffer,
        c: ?DeviceBuffer,
        dd: ?DeviceBuffer,
        push: EltPush,
        total_x: usize,
        total_y: usize,
        total_z: usize,
    ) Error!void {
        const dummy = try self.dummyBuf();
        const es = elt_entry_sizes[@intFromEnum(which)];
        try self.opBegin();
        var set = self.bind4(.{
            if (a) |t| t.buf else dummy,
            if (b) |t| t.buf else dummy,
            if (c) |t| t.buf else dummy,
            if (dd) |t| t.buf else dummy,
        });
        self.d.CmdBindPipeline(self.cmd, .compute, self.pipes_e[@intFromEnum(which)]);
        self.d.CmdBindDescriptorSets(self.cmd, .compute, self.pipeline_layout_e, 0, 1, @ptrCast(&set), 0, null);
        self.d.CmdPushConstants(self.cmd, self.pipeline_layout_e, vk.ShaderStage.compute, 0, @sizeOf(EltPush), &push);
        self.d.CmdDispatch(
            self.cmd,
            @intCast(std.math.divCeil(usize, total_x, es.x) catch unreachable),
            @intCast(std.math.divCeil(usize, @max(total_y, 1), es.y) catch unreachable),
            @intCast(@max(total_z, 1)),
        );
        try self.opEnd();
    }

    /// Bind the 5 buffers (q,k,v,out,bounds) for `attnBatchedDispatch`. Call
    /// ONCE before `beginBatch` — the buffers are stable across a forward's
    /// layers, so one persistent descriptor set (no ring) suffices, and updating
    /// it before the batch avoids touching a set referenced by recorded commands.
    /// `bounds` is a device u32[2*total]: bounds[q]=item-start, bounds[total+q]=
    /// item-end (exclusive) for query row q.
    pub fn attnBatchedBind(self: *Context, q: DeviceBuffer, k: DeviceBuffer, v: DeviceBuffer, out: DeviceBuffer, bounds: DeviceBuffer) void {
        const bufs = [5]vk.Buffer{ q.buf, k.buf, v.buf, out.buf, bounds.buf };
        var bi: [5]vk.DescriptorBufferInfo = undefined;
        var wr: [5]vk.WriteDescriptorSet = undefined;
        for (&bi, &wr, bufs, 0..) |*b_, *w_, buf, i| {
            b_.* = .{ .buffer = buf, .offset = 0 };
            w_.* = .{
                .dst_set = self.desc_set_ab,
                .dst_binding = @intCast(i),
                .dst_array_element = 0,
                .descriptor_count = 1,
                .descriptor_type = .storage_buffer,
                .p_image_info = null,
                .p_buffer_info = @ptrCast(b_),
                .p_texel_buffer_view = null,
            };
        }
        self.d.UpdateDescriptorSets(self.device, wr.len, &wr, 0, null);
    }

    /// Block-diagonal batched non-causal attention over the buffers last bound by
    /// `attnBatchedBind`: one launch over all total*n_heads (query,head) threads,
    /// each query attending only its own item's key range (from the bounds
    /// buffer). B× the parallelism of a per-item attention loop — the lever that
    /// fills the GPU for short-sequence encoders.
    pub fn attnBatchedDispatch(self: *Context, total: usize, n_heads: usize, kv_heads: usize, hd: usize, scale: f32) Error!void {
        try self.opBegin();
        self.d.CmdBindPipeline(self.cmd, .compute, self.pipe_ab);
        self.d.CmdBindDescriptorSets(self.cmd, .compute, self.pipeline_layout_ab, 0, 1, @ptrCast(&self.desc_set_ab), 0, null);
        const push: AttnBatchedPush = .{ .u0 = @intCast(total), .u1 = @intCast(n_heads), .u2 = @intCast(kv_heads), .u3 = @intCast(hd), .f0 = scale };
        self.d.CmdPushConstants(self.cmd, self.pipeline_layout_ab, vk.ShaderStage.compute, 0, @sizeOf(AttnBatchedPush), &push);
        self.d.CmdDispatch(self.cmd, @intCast((total * n_heads + 63) / 64), 1, 1);
        try self.opEnd();
    }

    /// Argmax over `logits` (device [vocab] f32) → token id written as an exact
    /// f32 into out_id[0] (download 4 bytes, not the ~608 KB vocab). Two passes:
    /// `lanes` stride-scanners each find a local max, then one thread reduces
    /// them, tie-breaking to the lowest index (matches sample.argmax). Caller
    /// owns the two working buffers (>= lanes f32 each). No workgroup memory.
    pub fn opArgmax(
        self: *Context,
        logits: DeviceBuffer,
        vocab: usize,
        out_id: DeviceBuffer,
        scratch_v: *DeviceBuffer,
        scratch_i: *DeviceBuffer,
    ) Error!void {
        // Explicit usize: `@min(4096, vocab)` would otherwise narrow to a type
        // bounded by 4096 (~u13), so `lanes * 4` overflows (#14039).
        const lanes: usize = @min(@as(usize, 4096), vocab);
        try self.ensureDeviceBuffer(scratch_v, lanes * 4);
        try self.ensureDeviceBuffer(scratch_i, lanes * 4);
        try self.opElt(.argmax_reduce, logits, null, scratch_v.*, scratch_i.*, .{
            .u0 = @intCast(lanes),
            .u1 = @intCast(vocab),
        }, lanes, 1, 1);
        try self.opElt(.argmax_final, scratch_v.*, scratch_i.*, null, out_id, .{
            .u0 = @intCast(lanes),
        }, 1, 1, 1);
    }

    /// Top-k candidate selection over `logits` (device [vocab] f32). `topk_lanes`
    /// lanes each keep their `topk_m` highest (value,index); the caller downloads
    /// the `topk_lanes * topk_m` candidates and does exact top-k on the CPU (a
    /// few KB vs the ~608 KB vocab). Caller owns out_val/out_idx (>= that many
    /// f32 each). Returns the candidate count written. No download here.
    /// Apply the sampling penalties (sample.zig penalizeLogit) to the
    /// resident logits before an on-device argmax/top-k: uploads the tiny
    /// (id, subtract) f32 wire and scatters one thread per unique recent
    /// token. Same formula as the CPU applyPenalties (division is subject to
    /// Vulkan's FP32 ULP tolerance). No-op on an empty entry list.
    pub fn opPenalize(self: *Context, logits: DeviceBuffer, entries: []const sample.PenaltyEntry, sp: sample.Params) Error!void {
        if (entries.len == 0) return;
        var wire: [2 * sample.max_penalty_window]f32 = undefined;
        const w = sample.packPenaltyWireF32(entries, sp, &wire);
        try self.ensureDeviceBuffer(&self.pen_wire, w.len * 4);
        try self.tensorUpload(self.pen_wire, std.mem.sliceAsBytes(w));
        try self.opElt(.penalize, logits, self.pen_wire, null, null, .{
            .u0 = @intCast(entries.len),
            .f0 = sp.repeat_penalty,
        }, entries.len, 1, 1);
    }

    pub fn opTopK(self: *Context, logits: DeviceBuffer, vocab: usize, out_val: *DeviceBuffer, out_idx: *DeviceBuffer) Error!usize {
        const lanes: usize = @min(@as(usize, topk_lanes), vocab);
        const count = lanes * topk_m;
        try self.ensureDeviceBuffer(out_val, count * 4);
        try self.ensureDeviceBuffer(out_idx, count * 4);
        try self.opElt(.topk_reduce, logits, null, out_val.*, out_idx.*, .{
            .u0 = @intCast(lanes),
            .u1 = @intCast(vocab),
        }, lanes, 1, 1);
        return count;
    }

    /// Drop a cached weight buffer (e.g. when its model is unloaded).
    pub fn evictWeights(self: *Context) void {
        _ = self.d.DeviceWaitIdle(self.device);
        var it = self.weights.valueIterator();
        while (it.next()) |wb| {
            self.freeDeviceBuffer(wb.db);
        }
        self.weights.clearRetainingCapacity();
        self.pinned_bytes = 0;
        var sit = self.small_bufs.valueIterator();
        while (sit.next()) |sb| {
            self.freeDeviceBuffer(sb.*);
        }
        self.small_bufs.clearRetainingCapacity();
    }

    /// Evict cached weights LRU-first — INCLUDING pinned ones (under VAE-decode
    /// memory pressure the resident DiT pin must yield) — until at least `want`
    /// bytes are freed or nothing more is evictable; return the bytes freed.
    /// The "make just enough room, keep the rest resident" counterpart to the
    /// full `evictWeights` nuke: used on a VAE-decode OOM so the untouched
    /// weights stay cached instead of being dropped and re-uploaded. Dropped
    /// weights re-upload from their mmap on next use.
    pub fn evictToFree(self: *Context, want: u64) u64 {
        if (want == 0) return 0;
        if (self.batching) self.flushBatch() catch {};
        _ = self.d.DeviceWaitIdle(self.device);
        var freed: u64 = 0;
        while (freed < want) {
            var lru_key: usize = undefined;
            var lru_use: u64 = std.math.maxInt(u64);
            var it = self.weights.iterator();
            while (it.next()) |e| {
                if (e.value_ptr.last_use < lru_use) {
                    lru_use = e.value_ptr.last_use;
                    lru_key = e.key_ptr.*;
                }
            }
            if (lru_use == std.math.maxInt(u64)) break; // cache empty
            const e = self.weights.fetchRemove(lru_key).?;
            if (e.value.pinned) self.pinned_bytes -|= e.value.db.size;
            freed += e.value.db.size;
            self.freeDeviceBuffer(e.value.db);
        }
        return freed;
    }

    /// Free device memory available to us right now (live driver view;
    /// VK_EXT_memory_budget sees other processes). Used at setup to default
    /// the weight-pin budget so a model that fits stays fully resident.
    pub fn liveVram(self: *Context) u64 {
        return self.liveHeadroom();
    }

    /// How many more bytes we may allocate before hitting the device
    /// memory budget. VK_EXT_memory_budget sees other processes' usage
    /// live; the fallback assumes 90% of the device-local heap is ours.
    fn budgetHeadroom(self: *Context) u64 {
        // An explicit --vram-budget is a CEILING on our own footprint, but it
        // must still yield to what's physically free: another process holding
        // most of the card can leave less than the budget available. Take the
        // min of the two so the budget never over-promises under external
        // pressure (previously it ignored the live query entirely, which is
        // why the same budget behaved fine on an empty card and OOM'd when
        // ~17 GB was held elsewhere).
        const live = self.liveHeadroom();
        if (self.budget_override != 0) {
            return @min(self.budget_override -| self.device_used, live);
        }
        return live;
    }

    /// Headroom from the driver's live view (VK_EXT_memory_budget sees other
    /// processes; heap-size * 0.9 minus our own usage is the fallback).
    fn liveHeadroom(self: *Context) u64 {
        if (self.has_memory_budget) {
            var budget: vk.PhysicalDeviceMemoryBudgetPropertiesEXT = .{
                .heap_budget = @splat(0),
                .heap_usage = @splat(0),
            };
            var props2: vk.PhysicalDeviceMemoryProperties2 = .{
                .p_next = &budget,
                .memory_properties = undefined,
            };
            self.d.GetPhysicalDeviceMemoryProperties2(self.phys, &props2);
            const cap = budget.heap_budget[self.device_heap];
            if (cap > 0) return (cap * 95 / 100) -| budget.heap_usage[self.device_heap];
        }
        return (self.mem_props.memory_heaps[self.device_heap].size * 9 / 10) -| self.device_used;
    }

    /// Evict the least-recently-used cached weight buffer (false when the
    /// cache is empty). Any recorded-but-unsubmitted dispatches may still
    /// reference it, so the pending batch is flushed first.
    fn evictOneWeight(self: *Context) bool {
        var lru_key: usize = undefined;
        var lru_use: u64 = std.math.maxInt(u64);
        var it = self.weights.iterator();
        while (it.next()) |e| {
            if (e.value_ptr.pinned) continue; // pinned prefix never streams
            if (e.value_ptr.last_use < lru_use) {
                lru_use = e.value_ptr.last_use;
                lru_key = e.key_ptr.*;
            }
        }
        if (lru_use == std.math.maxInt(u64)) return false;
        if (self.batching) self.flushBatch() catch return false;
        _ = self.d.DeviceWaitIdle(self.device);
        const e = self.weights.fetchRemove(lru_key).?;
        self.freeDeviceBuffer(e.value.db);
        return true;
    }

    /// Make room for a `need`-byte weight upload: evict LRU weights while
    /// the budget lacks headroom. Exhaustion thereby degrades to per-step
    /// weight re-uploads (streaming) instead of an allocation failure; if
    /// this still can't free enough, createBuffer's OOM retry is the
    /// reactive backstop.
    fn reserveForWeights(self: *Context, need: u64) void {
        while (self.budgetHeadroom() < need) {
            if (!self.evictOneWeight()) return;
        }
    }

    /// Claim pin residency for a newly cached `size`-byte weight (first-touch
    /// order): true while the claims fit under pin_budget.
    fn pinNew(self: *Context, size: u64) bool {
        if (self.pinned_bytes + size > self.pin_budget) return false;
        self.pinned_bytes += size;
        return true;
    }

    fn beginCmdBuf(self: *Context, cb: vk.CommandBuffer) Error!void {
        try check(self.d.ResetCommandBuffer(cb, 0));
        try check(self.d.BeginCommandBuffer(cb, &.{
            .flags = vk.CommandBufferUsage.one_time_submit,
        }));
    }

    fn submitAndWaitBuf(self: *Context, cb: vk.CommandBuffer) Error!void {
        try check(self.d.EndCommandBuffer(cb));
        var cb_mut = cb;
        const submit: vk.SubmitInfo = .{
            .command_buffer_count = 1,
            .p_command_buffers = @ptrCast(&cb_mut),
        };
        try check(self.d.QueueSubmit(self.queue, 1, @ptrCast(&submit), self.fence));
        try check(self.d.WaitForFences(self.device, 1, @ptrCast(&self.fence), vk.TRUE, std.math.maxInt(u64)));
        try check(self.d.ResetFences(self.device, 1, @ptrCast(&self.fence)));
    }

    fn beginCmd(self: *Context) Error!void {
        return self.beginCmdBuf(self.cmd);
    }

    fn submitAndWait(self: *Context) Error!void {
        return self.submitAndWaitBuf(self.cmd);
    }

    /// Command buffer for immediate (non-batched) work: transfers and weight
    /// preparation use `cmd_now` while a batch is being recorded on `cmd`.
    fn nowCmd(self: *Context) vk.CommandBuffer {
        return if (self.batching) self.cmd_now else self.cmd;
    }

    // --- batched submission -------------------------------------------------

    /// Start recording op* dispatches into one command buffer. Weight/bias
    /// cache misses keep working (they run on `cmd_now` and complete before
    /// the batch is submitted); uploads/downloads of batch-visible tensors
    /// flush the pending recording first.
    pub fn beginBatch(self: *Context) Error!void {
        std.debug.assert(!self.batching);
        try self.beginCmd();
        self.batching = true;
        self.batch_n = 0;
    }

    /// Submit everything recorded since beginBatch and wait for completion.
    pub fn endBatch(self: *Context) Error!void {
        std.debug.assert(self.batching);
        std.debug.assert(self.indep_remaining == 0); // group left open
        self.batching = false;
        try self.submitAndWait();
    }

    /// Abandon an in-progress batch (error paths): the recorded commands are
    /// dropped; the command buffer is reset on next use.
    pub fn abortBatch(self: *Context) void {
        self.batching = false;
        self.indep_remaining = 0;
    }

    /// Submit the recording so far, wait, and resume with a fresh command
    /// buffer and descriptor ring.
    fn flushBatch(self: *Context) Error!void {
        std.debug.assert(self.batching);
        try self.submitAndWait();
        try self.beginCmd();
        self.batch_n = 0;
    }

    /// Per-op prologue: outside a batch, start the one-shot command buffer;
    /// inside one, make room in the descriptor ring.
    fn opBegin(self: *Context) Error!void {
        if (self.batching) {
            if (self.batch_n == self.batch_sets.len) try self.flushBatch();
        } else {
            try self.beginCmd();
        }
    }

    /// Longest recording per submission. Uncapped batches trip the display
    /// GPU's preemption watchdog (NVRM Xid 109 CTX SWITCH TIMEOUT) once a
    /// single submission runs for seconds; ~64 dispatches keeps submissions
    /// around 100 ms while still eliding ~98% of the fence waits.
    const batch_flush_limit = 512;

    /// Declare that the next `n` recorded ops are pairwise independent (no
    /// op reads or writes a buffer another op in the group writes): no
    /// barrier is recorded between them, so their dispatches may overlap on
    /// the device — the tail waves of one fill with the next. One barrier is
    /// still recorded after the group's last op. A mid-group flush (ring or
    /// dispatch cap) is harmless: submit-and-wait is a stronger sync than
    /// the elided barrier. Sync (non-batched) mode ignores grouping.
    pub fn independent(self: *Context, n: usize) void {
        std.debug.assert(self.indep_remaining == 0);
        self.indep_remaining = n;
    }

    /// Per-op epilogue: outside a batch, submit-and-wait; inside one, order
    /// this dispatch's writes before every later dispatch's reads/writes
    /// (unless the op is inside an `independent` group).
    fn opEnd(self: *Context) Error!void {
        const in_group = self.indep_remaining > 1;
        if (self.indep_remaining > 0) self.indep_remaining -= 1;
        if (self.batching) {
            if (!in_group) {
                const bar: vk.MemoryBarrier = .{
                    .src_access_mask = vk.Access.shader_write,
                    .dst_access_mask = vk.Access.shader_read | vk.Access.shader_write,
                };
                self.d.CmdPipelineBarrier(self.cmd, vk.PipelineStage.compute_shader, vk.PipelineStage.compute_shader, 0, 1, @ptrCast(&bar), 0, null, 0, null);
            }
            if (self.batch_n >= batch_flush_limit) try self.flushBatch();
        } else {
            try self.submitAndWait();
        }
    }

    /// y[m, rows] = x[m, cols] @ W^T (+ bias). `dtype_f8` selects the fp8
    /// kernel; otherwise W bytes are f32 words. One synchronous submission:
    /// stage x -> copy to device -> dispatch -> copy y back -> read.
    pub fn matmul(
        self: *Context,
        y: []f32,
        x: []const f32,
        m: usize,
        w_bytes: []const u8,
        dtype_f8: bool,
        rows: usize,
        cols: usize,
        scale: f32,
        bias: ?[]const f32,
    ) Error!void {
        std.debug.assert(!self.batching); // host-visible path is sync-only
        const w_buf = try self.weightBuffer(w_bytes, if (dtype_f8) 1 else 4, rows, cols);
        const x_size = x.len * 4;
        const y_size = y.len * 4;
        const x_mapped = try self.ensureHostBuffer(&self.x_buf, x_size);
        _ = try self.ensureHostBuffer(&self.y_buf, y_size);
        const bias_mapped = try self.ensureHostBuffer(&self.bias_buf, if (bias) |b| b.len * 4 else 16);
        try self.ensureDeviceBuffer(&self.x_dev, x_size);
        try self.ensureDeviceBuffer(&self.y_dev, y_size);
        @memcpy(x_mapped[0..x_size], std.mem.sliceAsBytes(x));
        if (bias) |b| @memcpy(bias_mapped[0 .. b.len * 4], std.mem.sliceAsBytes(b));

        // Bind current buffers.
        const buf_infos = [4]vk.DescriptorBufferInfo{
            .{ .buffer = w_buf },
            .{ .buffer = self.x_dev.buf },
            .{ .buffer = self.y_dev.buf },
            .{ .buffer = self.bias_buf.buf },
        };
        var writes: [4]vk.WriteDescriptorSet = undefined;
        for (&writes, 0..) |*wr, i| {
            wr.* = .{
                .dst_set = self.desc_set,
                .dst_binding = @intCast(i),
                .dst_array_element = 0,
                .descriptor_count = 1,
                .descriptor_type = .storage_buffer,
                .p_image_info = null,
                .p_buffer_info = @ptrCast(&buf_infos[i]),
                .p_texel_buffer_view = null,
            };
        }
        self.d.UpdateDescriptorSets(self.device, writes.len, &writes, 0, null);

        const push: Push = .{
            .m = @intCast(m),
            .rows = @intCast(rows),
            .cols = @intCast(cols),
            .w_stride = @intCast(std.mem.alignForward(usize, rows, tile_n)),
            .has_bias = @intFromBool(bias != null),
            .scale = scale,
        };

        try self.beginCmd();
        {
            const region: vk.BufferCopy = .{ .src_offset = 0, .dst_offset = 0, .size = x_size };
            self.d.CmdCopyBuffer(self.cmd, self.x_buf.buf, self.x_dev.buf, 1, @ptrCast(&region));
            const to_shader: vk.BufferMemoryBarrier = .{
                .src_access_mask = vk.Access.transfer_write,
                .dst_access_mask = vk.Access.shader_read,
                .buffer = self.x_dev.buf,
                .offset = 0,
                .size = vk.WHOLE_SIZE,
            };
            self.d.CmdPipelineBarrier(self.cmd, vk.PipelineStage.transfer, vk.PipelineStage.compute_shader, 0, 0, null, 1, @ptrCast(&to_shader), 0, null);
        }
        self.d.CmdBindPipeline(self.cmd, .compute, if (dtype_f8) self.pipe_f8 else self.pipe_f32);
        self.d.CmdBindDescriptorSets(self.cmd, .compute, self.pipeline_layout, 0, 1, @ptrCast(&self.desc_set), 0, null);
        self.d.CmdPushConstants(self.cmd, self.pipeline_layout, vk.ShaderStage.compute, 0, @sizeOf(Push), &push);
        self.d.CmdDispatch(
            self.cmd,
            @intCast(std.math.divCeil(usize, rows, wg_x * tile_n) catch unreachable),
            @intCast(std.math.divCeil(usize, m, wg_y * tile_m) catch unreachable),
            1,
        );
        {
            const to_copy: vk.BufferMemoryBarrier = .{
                .src_access_mask = vk.Access.shader_write,
                .dst_access_mask = vk.Access.transfer_read,
                .buffer = self.y_dev.buf,
                .offset = 0,
                .size = vk.WHOLE_SIZE,
            };
            self.d.CmdPipelineBarrier(self.cmd, vk.PipelineStage.compute_shader, vk.PipelineStage.transfer, 0, 0, null, 1, @ptrCast(&to_copy), 0, null);
            const region: vk.BufferCopy = .{ .src_offset = 0, .dst_offset = 0, .size = y_size };
            self.d.CmdCopyBuffer(self.cmd, self.y_dev.buf, self.y_buf.buf, 1, @ptrCast(&region));
            const to_host: vk.BufferMemoryBarrier = .{
                .src_access_mask = vk.Access.transfer_write,
                .dst_access_mask = vk.Access.host_read,
                .buffer = self.y_buf.buf,
                .offset = 0,
                .size = vk.WHOLE_SIZE,
            };
            self.d.CmdPipelineBarrier(self.cmd, vk.PipelineStage.transfer, vk.PipelineStage.host, 0, 0, null, 1, @ptrCast(&to_host), 0, null);
        }
        try self.submitAndWait();

        @memcpy(std.mem.sliceAsBytes(y), self.y_buf.mapped.?[0..y_size]);
    }
};

// GPU tests run only when a Vulkan device is actually present, and only
// when opted in via the `testdata/gpu-tests` marker: the NVIDIA 580 driver
// faults on Zig-emitted workgroup-storage kernels (validator-clean; fine on
// RADV/llvmpipe), which would kill the test process. See PLAN.md (M9).
test "gpu gdn kernels match cpu reference" {
    const gpa = std.testing.allocator;
    std.Io.Dir.cwd().access(std.testing.io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    var ctx = Context.init(gpa) catch return error.SkipZigTest;
    defer ctx.deinit();
    var prng = std.Random.DefaultPrng.init(555);
    const rand = prng.random();
    const expect = std.testing.expectApproxEqAbs;

    // --- gdn_gates: decay = exp(a*softplus(alpha+dt)), beta = sigmoid(braw) ---
    {
        const heads = 16;
        const ab = try gpa.alloc(f32, 2 * heads); // [alpha | beta_raw]
        defer gpa.free(ab);
        const adt = try gpa.alloc(f32, 2 * heads); // [a | dt_bias]
        defer gpa.free(adt);
        for (ab) |*v| v.* = rand.floatNorm(f32);
        for (adt) |*v| v.* = rand.floatNorm(f32) * 0.5;
        var ab_d = try ctx.tensorCreate(2 * heads * 4);
        var adt_d = try ctx.tensorCreate(2 * heads * 4);
        var g_d = try ctx.tensorCreate(2 * heads * 4);
        defer {
            ctx.tensorDestroy(&ab_d);
            ctx.tensorDestroy(&adt_d);
            ctx.tensorDestroy(&g_d);
        }
        try ctx.tensorUpload(ab_d, std.mem.sliceAsBytes(ab));
        try ctx.tensorUpload(adt_d, std.mem.sliceAsBytes(adt));
        try ctx.opGdnGates(ab_d, adt_d, g_d, heads);
        const g = try gpa.alloc(f32, 2 * heads);
        defer gpa.free(g);
        try ctx.tensorDownload(g_d, std.mem.sliceAsBytes(g));
        for (0..heads) |h| {
            const s = ab[h] + adt[heads + h];
            const sp = if (s > 20.0) s else @log(1.0 + @exp(s));
            try expect(@exp(adt[h] * sp), g[h], 1e-4);
            try expect(1.0 / (1.0 + @exp(-ab[heads + h])), g[heads + h], 1e-4);
        }
    }

    // --- gdn_conv_step: depthwise causal conv (taps 4) + SiLU, rolling state ---
    {
        const ch = 96;
        const taps = 4;
        const st = try gpa.alloc(f32, ch * (taps - 1));
        defer gpa.free(st);
        const x = try gpa.alloc(f32, ch);
        defer gpa.free(x);
        const w = try gpa.alloc(f32, ch * taps);
        defer gpa.free(w);
        for (st) |*v| v.* = rand.floatNorm(f32);
        for (x) |*v| v.* = rand.floatNorm(f32);
        for (w) |*v| v.* = rand.floatNorm(f32);
        // CPU reference (mutates copies).
        const st_ref = try gpa.dupe(f32, st);
        defer gpa.free(st_ref);
        const out_ref = try gpa.alloc(f32, ch);
        defer gpa.free(out_ref);
        for (0..ch) |c| {
            const s = st_ref[c * (taps - 1) ..][0 .. taps - 1];
            const ww = w[c * taps ..][0..taps];
            var acc: f32 = ww[taps - 1] * x[c];
            for (0..taps - 1) |k| acc += ww[k] * s[k];
            for (0..taps - 2) |k| s[k] = s[k + 1];
            s[taps - 2] = x[c];
            out_ref[c] = acc / (1.0 + @exp(-acc));
        }
        var st_d = try ctx.tensorCreate(ch * (taps - 1) * 4);
        var x_d = try ctx.tensorCreate(ch * 4);
        var w_d = try ctx.tensorCreate(ch * taps * 4);
        var o_d = try ctx.tensorCreate(ch * 4);
        defer {
            ctx.tensorDestroy(&st_d);
            ctx.tensorDestroy(&x_d);
            ctx.tensorDestroy(&w_d);
            ctx.tensorDestroy(&o_d);
        }
        try ctx.tensorUpload(st_d, std.mem.sliceAsBytes(st));
        try ctx.tensorUpload(x_d, std.mem.sliceAsBytes(x));
        try ctx.tensorUpload(w_d, std.mem.sliceAsBytes(w));
        try ctx.opGdnConvStep(st_d, x_d, w_d, o_d, ch, taps);
        const o = try gpa.alloc(f32, ch);
        defer gpa.free(o);
        const st_out = try gpa.alloc(f32, ch * (taps - 1));
        defer gpa.free(st_out);
        try ctx.tensorDownload(o_d, std.mem.sliceAsBytes(o));
        try ctx.tensorDownload(st_d, std.mem.sliceAsBytes(st_out));
        for (o, out_ref) |gv, rv| try expect(rv, gv, 1e-4);
        for (st_out, st_ref) |gv, rv| try expect(rv, gv, 1e-4);
    }

    // --- gdn_delta_step: per-head delta rule over dd x dd state ---
    {
        const heads = 4;
        const dd = 16;
        const kheads = 2;
        const scale: f32 = 1.0 / @sqrt(@as(f32, dd));
        const qkdim = kheads * dd;
        const vdim = heads * dd;
        const co = try gpa.alloc(f32, 2 * qkdim + vdim); // q | k | v
        defer gpa.free(co);
        const gates = try gpa.alloc(f32, 2 * heads); // decay | beta
        defer gpa.free(gates);
        const state = try gpa.alloc(f32, heads * dd * dd);
        defer gpa.free(state);
        for (co) |*v| v.* = rand.floatNorm(f32) * 0.3;
        for (state) |*v| v.* = rand.floatNorm(f32) * 0.1;
        for (0..heads) |h| {
            gates[h] = 0.9 + rand.float(f32) * 0.09; // decay in (0.9,0.99)
            gates[heads + h] = rand.float(f32); // beta in (0,1)
        }
        // CPU reference.
        const st_ref = try gpa.dupe(f32, state);
        defer gpa.free(st_ref);
        const o_ref = try gpa.alloc(f32, vdim);
        defer gpa.free(o_ref);
        var m: [128]f32 = undefined;
        var dl: [128]f32 = undefined;
        for (0..heads) |h| {
            const decay = gates[h];
            const beta = gates[heads + h];
            const kh = co[qkdim + (h % kheads) * dd ..][0..dd];
            const qh = co[(h % kheads) * dd ..][0..dd];
            const vh = co[2 * qkdim + h * dd ..][0..dd];
            const S = st_ref[h * dd * dd ..][0 .. dd * dd];
            for (0..dd) |j| m[j] = 0;
            for (0..dd) |i| {
                for (0..dd) |j| {
                    S[i * dd + j] *= decay;
                    m[j] += S[i * dd + j] * kh[i];
                }
            }
            for (0..dd) |j| dl[j] = (vh[j] - m[j]) * beta;
            const oh = o_ref[h * dd ..][0..dd];
            for (0..dd) |j| oh[j] = 0;
            for (0..dd) |i| {
                const qi = qh[i] * scale;
                for (0..dd) |j| {
                    S[i * dd + j] += kh[i] * dl[j];
                    oh[j] += S[i * dd + j] * qi;
                }
            }
        }
        var co_d = try ctx.tensorCreate(co.len * 4);
        var g_d = try ctx.tensorCreate(gates.len * 4);
        var s_d = try ctx.tensorCreate(state.len * 4);
        var o_d = try ctx.tensorCreate(vdim * 4);
        defer {
            ctx.tensorDestroy(&co_d);
            ctx.tensorDestroy(&g_d);
            ctx.tensorDestroy(&s_d);
            ctx.tensorDestroy(&o_d);
        }
        try ctx.tensorUpload(co_d, std.mem.sliceAsBytes(co));
        try ctx.tensorUpload(g_d, std.mem.sliceAsBytes(gates));
        try ctx.tensorUpload(s_d, std.mem.sliceAsBytes(state));
        try ctx.opGdnDeltaStep(s_d, co_d, g_d, o_d, heads, dd, kheads, scale);
        const o = try gpa.alloc(f32, vdim);
        defer gpa.free(o);
        const s_out = try gpa.alloc(f32, state.len);
        defer gpa.free(s_out);
        try ctx.tensorDownload(o_d, std.mem.sliceAsBytes(o));
        try ctx.tensorDownload(s_d, std.mem.sliceAsBytes(s_out));
        for (o, o_ref) |gv, rv| try expect(rv, gv, 1e-3);
        for (s_out, st_ref) |gv, rv| try expect(rv, gv, 1e-3);
    }
}

test "gpu block-quant gemv matches cpu reference" {
    const gpa = std.testing.allocator;
    std.Io.Dir.cwd().access(std.testing.io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    var ctx = Context.init(gpa) catch return error.SkipZigTest;
    defer ctx.deinit();
    const dtypes = @import("tp_core").dtype;
    const quants = @import("tp_core").quants;

    const rows = 64;
    const cols = 512; // two 256-elem super-blocks: exercises the shared scale table
    var prng = std.Random.DefaultPrng.init(77);
    const rand = prng.random();

    const x = try gpa.alloc(f32, cols);
    defer gpa.free(x);
    for (x) |*v| v.* = rand.floatNorm(f32) * 2.0 - 1.0;
    var x_d = try ctx.tensorCreate(cols * 4);
    var y_d = try ctx.tensorCreate(rows * 4);
    defer {
        ctx.tensorDestroy(&x_d);
        ctx.tensorDestroy(&y_d);
    }
    try ctx.tensorUpload(x_d, std.mem.sliceAsBytes(x));

    const y = try gpa.alloc(f32, rows);
    defer gpa.free(y);
    const row_f32 = try gpa.alloc(f32, cols);
    defer gpa.free(row_f32);

    const dts = [_]dtypes.DType{ .q8_0, .q4_k, .q5_k, .q6_k };
    const d16: u16 = 0x2A66; // ~0.05
    const min16: u16 = 0x251F; // ~0.02

    // All weight buffers stay alive for the whole test: the device weight
    // cache is keyed by host pointer, so free-then-realloc at the same address
    // would alias a stale upload.
    var ws: [dts.len][]u8 = undefined;
    inline for (dts, 0..) |dt, i| {
        ws[i] = try gpa.alloc(u8, dt.storageBytes(rows * cols));
        rand.bytes(ws[i]);
        const bb = dt.blockBytes();
        var off: usize = 0;
        while (off < ws[i].len) : (off += bb) {
            switch (dt) {
                .q8_0 => std.mem.writeInt(u16, ws[i][off..][0..2], d16, .little),
                .q4_k, .q5_k => {
                    std.mem.writeInt(u16, ws[i][off..][0..2], d16, .little);
                    std.mem.writeInt(u16, ws[i][off + 2 ..][0..2], min16, .little);
                },
                .q6_k => std.mem.writeInt(u16, ws[i][off + 208 ..][0..2], d16, .little),
                else => unreachable,
            }
        }
    }
    defer for (ws) |w| gpa.free(w);

    inline for (dts, 0..) |dt, i| {
        try ctx.opGemvQuant(dt, y_d, 0, x_d, ws[i], 1.0, rows, cols);
        try ctx.tensorDownload(y_d, std.mem.sliceAsBytes(y));

        const row_bytes = dt.storageBytes(cols);
        for (0..rows) |r| {
            quants.dequantSlice(dt, ws[i][r * row_bytes ..][0..row_bytes], 0, cols, row_f32);
            var acc: f64 = 0;
            for (row_f32, x) |wv, xv| acc += @as(f64, wv) * xv;
            std.testing.expectApproxEqAbs(@as(f32, @floatCast(acc)), y[r], 2e-2) catch |e| {
                std.debug.print("dtype {s} row {d}: gpu {d} cpu {d}\n", .{ @tagName(dt), r, y[r], acc });
                return e;
            };
        }
    }

    // Transposed + k-split path (opGemvQuantT): same CPU reference. Uses
    // SEPARATE weight buffers (the device cache is keyed by host pointer, and
    // the raw vs transposed uploads must not alias).
    const nchunk = 16;
    var part_d = try ctx.tensorCreate(rows * nchunk * 4);
    defer ctx.tensorDestroy(&part_d);
    var wsT: [dts.len][]u8 = undefined;
    inline for (dts, 0..) |_, i| wsT[i] = try gpa.dupe(u8, ws[i]);
    defer for (wsT) |w| gpa.free(w);
    inline for (dts, 0..) |dt, i| {
        try ctx.opGemvQuantT(dt, y_d, 0, x_d, wsT[i], 1.0, rows, cols, nchunk, part_d);
        try ctx.tensorDownload(y_d, std.mem.sliceAsBytes(y));

        const row_bytes = dt.storageBytes(cols);
        for (0..rows) |r| {
            quants.dequantSlice(dt, wsT[i][r * row_bytes ..][0..row_bytes], 0, cols, row_f32);
            var acc: f64 = 0;
            for (row_f32, x) |wv, xv| acc += @as(f64, wv) * xv;
            std.testing.expectApproxEqAbs(@as(f32, @floatCast(acc)), y[r], 2e-2) catch |e| {
                std.debug.print("dtype {s} (transposed) row {d}: gpu {d} cpu {d}\n", .{ @tagName(dt), r, y[r], acc });
                return e;
            };
        }
    }
}

test "gpu matmul matches cpu reference" {
    const gpa = std.testing.allocator;
    std.Io.Dir.cwd().access(std.testing.io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    var ctx = Context.init(gpa) catch return error.SkipZigTest;
    defer ctx.deinit();
    std.debug.print("gpu device: {s}\n", .{ctx.deviceName()});

    const m = 33;
    const rows = 70;
    const cols = 129;
    var prng = std.Random.DefaultPrng.init(9);
    const rand = prng.random();

    // f32 weights.
    {
        const wdata = try gpa.alloc(f32, rows * cols);
        defer gpa.free(wdata);
        for (wdata) |*v| v.* = rand.floatNorm(f32);
        const x = try gpa.alloc(f32, m * cols);
        defer gpa.free(x);
        for (x) |*v| v.* = rand.floatNorm(f32);
        const bias = try gpa.alloc(f32, rows);
        defer gpa.free(bias);
        for (bias) |*v| v.* = rand.floatNorm(f32);

        const y = try gpa.alloc(f32, m * rows);
        defer gpa.free(y);
        try ctx.matmul(y, x, m, std.mem.sliceAsBytes(wdata), false, rows, cols, 1.0, bias);

        for (0..m) |t| {
            for (0..rows) |r| {
                var want: f64 = bias[r];
                for (0..cols) |k| want += @as(f64, x[t * cols + k]) * wdata[r * cols + k];
                try std.testing.expectApproxEqAbs(@as(f32, @floatCast(want)), y[t * rows + r], 2e-3);
            }
        }
    }

    // fp8 weights with scale, cooperative-matrix path (fused e4m3 decode).
    if (ctx.pipe_coop != .null_handle) {
        const dtypes = @import("tp_core").dtype;
        const cm = 100;
        const m_pad = 128;
        const crows = 256;
        const ccols = 192;
        const scale: f32 = 0.5;
        const wbytes = try gpa.alloc(u8, crows * ccols);
        defer gpa.free(wbytes);
        for (wbytes) |*bb| {
            bb.* = rand.int(u8);
            if (bb.* & 0x7F == 0x7F) bb.* &= 0xF0; // avoid e4m3fn NaN patterns
        }
        const x = try gpa.alloc(f32, cm * ccols);
        defer gpa.free(x);
        for (x) |*v| v.* = rand.floatNorm(f32);

        var x_d = try ctx.tensorCreate(m_pad * ccols * 4);
        defer ctx.tensorDestroy(&x_d);
        var y_d = try ctx.tensorCreate(m_pad * crows * 4);
        defer ctx.tensorDestroy(&y_d);
        try ctx.tensorUpload(x_d, std.mem.sliceAsBytes(x));
        try ctx.opMatmulCoop(y_d, x_d, cm, m_pad, wbytes, crows, ccols, scale);
        const y = try gpa.alloc(f32, cm * crows);
        defer gpa.free(y);
        try ctx.tensorDownload(y_d, std.mem.sliceAsBytes(y));

        // Reference with f16-rounded operands (the kernel's exact regime),
        // f64 accumulation. With f16 ACCUMULATORS (coop_acc_h16) each MMA
        // step rounds the running sum at ~2^-11 relative of its magnitude,
        // which is bounded by the row's absolute-product sum — this test's
        // random e4m3 bytes drive that into the thousands (measured
        // max_abs ~4 on sums bounded ~6700, i.e. a few f16 ulps), so the
        // gate scales with that bound. The exact rounding depends on the
        // hardware's MMA accumulation order: NVIDIA lands within ~4 ulps
        // (2^-9), AMD RADV (wave32-pinned coopmat) a touch wider (~want_abs/
        // 483), so the gate is want_abs/256 (~8 ulps) to cover both. This is
        // still tight enough to catch the wrong-subgroup-size all-zero failure
        // (100% off) and any gross regression. The model-level gate stays the
        // DiT parity fixture (0.169 with f16 accs).
        for (0..cm) |t| {
            for (0..crows) |r| {
                var want: f64 = 0;
                var want_abs: f64 = 0;
                for (0..ccols) |k| {
                    const xa: f16 = @floatCast(x[t * ccols + k] * scale);
                    const wa: f16 = @floatCast(dtypes.f8e4m3ToF32(wbytes[r * ccols + k]));
                    const prod = @as(f64, @floatCast(xa)) * @as(f64, @floatCast(wa));
                    want += prod;
                    want_abs += @abs(prod);
                }
                const wantf: f32 = @floatCast(want);
                const tol: f32 = if (coopmat.coop_acc_h16)
                    @max(5e-3, @as(f32, @floatCast(want_abs)) / 256.0)
                else
                    5e-3;
                try std.testing.expectApproxEqAbs(wantf, y[t * crows + r], tol);
            }
        }
    }

    // Tensor-core attention scores vs an f16-rounded CPU reference.
    if (ctx.pipe_scores != .null_handle) {
        const seq = 200;
        const seq_pad = 256;
        const n_heads = 4;
        const n_kv = 2;
        const hd = 128;
        const scale: f32 = 0.25;
        const q = try gpa.alloc(f32, seq * n_heads * hd);
        defer gpa.free(q);
        for (q) |*v| v.* = rand.floatNorm(f32);
        const k = try gpa.alloc(f32, seq * n_kv * hd);
        defer gpa.free(k);
        for (k) |*v| v.* = rand.floatNorm(f32);

        var q_d = try ctx.tensorCreate(seq_pad * n_heads * hd * 4);
        defer ctx.tensorDestroy(&q_d);
        var q16_d = try ctx.tensorCreate(seq_pad * n_heads * hd * 2);
        defer ctx.tensorDestroy(&q16_d);
        var k_d = try ctx.tensorCreate(seq * n_kv * hd * 4);
        defer ctx.tensorDestroy(&k_d);
        var k16_d = try ctx.tensorCreate(n_kv * hd * seq_pad * 2);
        defer ctx.tensorDestroy(&k16_d);
        var s_d = try ctx.tensorCreate(n_heads * seq_pad * seq_pad * 2);
        defer ctx.tensorDestroy(&s_d);
        try ctx.tensorUpload(q_d, std.mem.sliceAsBytes(q));
        try ctx.tensorUpload(k_d, std.mem.sliceAsBytes(k));

        try ctx.opElt(.f32_to_h16, q_d, null, null, q16_d, .{
            .u0 = seq_pad * n_heads * hd / 2,
            .u1 = seq * n_heads * hd,
            .f0 = scale,
        }, seq_pad * n_heads * hd / 2, 1, 1);
        try ctx.opElt(.gather_kmajor_h16, k_d, null, null, k16_d, .{
            .u0 = n_kv * hd * seq_pad / 2,
            .u1 = hd,
            .u2 = seq_pad,
            .u3 = seq,
            .u4 = n_kv,
        }, n_kv * hd * seq_pad / 2, 1, 1);
        try ctx.opAttnScores(s_d, q16_d, k16_d, .{
            .u0 = n_heads * hd,
            .u1 = seq_pad,
            .u2 = 0,
            .u3 = n_heads / n_kv,
            .u4 = hd * seq_pad,
            .u5 = seq_pad * seq_pad,
        }, seq_pad / 128, seq_pad / 128, n_heads);

        const s16 = try gpa.alloc(f16, n_heads * seq_pad * seq_pad);
        defer gpa.free(s16);
        try ctx.tensorDownload(s_d, std.mem.sliceAsBytes(s16));
        const s = try gpa.alloc(f32, n_heads * seq_pad * seq_pad);
        defer gpa.free(s);
        for (s, s16) |*dst, src| dst.* = @floatCast(src);
        for (0..n_heads) |h| {
            const kvh = h / (n_heads / n_kv);
            for (0..seq) |qi| {
                for (0..seq) |ji| {
                    var want: f64 = 0;
                    for (0..hd) |kk| {
                        const qa: f16 = @floatCast(q[(qi * n_heads + h) * hd + kk] * scale);
                        const ka: f16 = @floatCast(k[(ji * n_kv + kvh) * hd + kk]);
                        want += @as(f64, @floatCast(qa)) * @as(f64, @floatCast(ka));
                    }
                    // S is stored f16 (|S| up to ~10 here -> ~5e-3 ulp).
                    const got = s[(h * seq_pad + qi) * seq_pad + ji];
                    try std.testing.expectApproxEqAbs(@as(f32, @floatCast(want)), got, 3e-2);
                }
            }
        }

        // Tensor-core P@V with the two-pass softmax, on the same scores.
        if (ctx.pipe_attn_out != .null_handle) {
            const vv = try gpa.alloc(f32, seq * n_kv * hd);
            defer gpa.free(vv);
            for (vv) |*x| x.* = rand.floatNorm(f32);
            var v_d = try ctx.tensorCreate(seq * n_kv * hd * 4);
            defer ctx.tensorDestroy(&v_d);
            var v16_d = try ctx.tensorCreate(seq_pad * n_kv * hd * 2);
            defer ctx.tensorDestroy(&v16_d);
            var part_d = try ctx.tensorCreate(n_heads * seq * 32 * 2 * 4);
            defer ctx.tensorDestroy(&part_d);
            var md_d = try ctx.tensorCreate(n_heads * seq_pad * 2 * 4);
            defer ctx.tensorDestroy(&md_d);
            var o_d = try ctx.tensorCreate(seq_pad * n_heads * hd * 4);
            defer ctx.tensorDestroy(&o_d);
            try ctx.tensorUpload(v_d, std.mem.sliceAsBytes(vv));
            try ctx.opElt(.f32_to_h16, v_d, null, null, v16_d, .{
                .u0 = seq_pad * n_kv * hd / 2,
                .u1 = seq * n_kv * hd,
                .f0 = 1.0,
            }, seq_pad * n_kv * hd / 2, 1, 1);
            try ctx.opElt(.softmax_partial, s_d, null, null, part_d, .{
                .u0 = n_heads * seq * 32,
                .u1 = 32,
                .u2 = seq,
                .u3 = seq_pad,
                .u5 = seq_pad * seq_pad,
            }, n_heads * seq * 32, 1, 1);
            try ctx.opElt(.softmax_combine, part_d, null, null, md_d, .{
                .u0 = n_heads * seq,
                .u1 = 32,
                .u2 = seq,
                .u3 = seq_pad,
            }, n_heads * seq, 1, 1);
            try ctx.opAttnOut(s_d, v16_d, o_d, md_d, .{
                .u0 = seq_pad,
                .u1 = seq_pad * seq_pad,
                .u2 = 0,
                .u3 = n_heads / n_kv,
                .u4 = n_kv * hd,
                .u5 = n_heads * hd,
                .f0 = @bitCast(@as(u32, seq)),
                .f1 = @bitCast(@as(u32, seq_pad)), // MD rows per head plane
            }, seq_pad / 128, n_heads);
            const o = try gpa.alloc(f32, seq * n_heads * hd);
            defer gpa.free(o);
            try ctx.tensorDownload(o_d, std.mem.sliceAsBytes(o));

            // Reference from the downloaded scores: exact softmax, f16 P/V.
            const p_row = try gpa.alloc(f64, seq);
            defer gpa.free(p_row);
            for (0..n_heads) |h| {
                const kvh = h / (n_heads / n_kv);
                for (0..seq) |qi| {
                    var rmax: f64 = -std.math.inf(f64);
                    for (0..seq) |ji| rmax = @max(rmax, s[(h * seq_pad + qi) * seq_pad + ji]);
                    var dsum: f64 = 0;
                    for (0..seq) |ji| {
                        p_row[ji] = @exp(s[(h * seq_pad + qi) * seq_pad + ji] - rmax);
                        dsum += p_row[ji];
                    }
                    for (0..hd) |cc| {
                        var want: f64 = 0;
                        for (0..seq) |ji| {
                            const pa: f16 = @floatCast(p_row[ji] / dsum);
                            const va: f16 = @floatCast(vv[(ji * n_kv + kvh) * hd + cc]);
                            want += @as(f64, @floatCast(pa)) * @as(f64, @floatCast(va));
                        }
                        const got = o[(qi * n_heads + h) * hd + cc];
                        try std.testing.expectApproxEqAbs(@as(f32, @floatCast(want)), got, 1e-2);
                    }
                }
            }
        }
    }

    // fp8 weights with scale.
    {
        const dtypes = @import("tp_core").dtype;
        const wbytes = try gpa.alloc(u8, rows * cols);
        defer gpa.free(wbytes);
        for (wbytes) |*b| b.* = rand.int(u8) & 0x7e;
        const x = try gpa.alloc(f32, m * cols);
        defer gpa.free(x);
        for (x) |*v| v.* = rand.floatNorm(f32);

        const y = try gpa.alloc(f32, m * rows);
        defer gpa.free(y);
        try ctx.matmul(y, x, m, wbytes, true, rows, cols, 0.125, null);

        for (0..m) |t| {
            for (0..rows) |r| {
                var want: f64 = 0;
                for (0..cols) |k| want += @as(f64, x[t * cols + k]) * dtypes.f8e4m3ToF32(wbytes[r * cols + k]) * 0.125;
                try std.testing.expectApproxEqAbs(@as(f32, @floatCast(want)), y[t * rows + r], 2e-3);
            }
        }
    }
}

// Flash attention (md + out passes) against a CPU reference computed in the
// kernels' exact regime: f16 Q/K/V, f16-rounded scores, f32 softmax, f16 P.
test "flash attention matches reference" {
    const gpa = std.testing.allocator;
    std.Io.Dir.cwd().access(std.testing.io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    var ctx = Context.init(gpa) catch return error.SkipZigTest;
    defer ctx.deinit();
    if (ctx.pipe_flash_md == .null_handle) return error.SkipZigTest;

    const seq = 200;
    const seq_pad = 256;
    const n_heads = 4;
    const n_kv = 2;
    const hd = 128;
    const scale: f32 = 0.25;
    var prng = std.Random.DefaultPrng.init(77);
    const rand = prng.random();

    const q = try gpa.alloc(f32, seq * n_heads * hd);
    defer gpa.free(q);
    for (q) |*v| v.* = rand.floatNorm(f32);
    const k = try gpa.alloc(f32, seq * n_kv * hd);
    defer gpa.free(k);
    for (k) |*v| v.* = rand.floatNorm(f32);
    const vv = try gpa.alloc(f32, seq * n_kv * hd);
    defer gpa.free(vv);
    for (vv) |*v| v.* = rand.floatNorm(f32);

    var q_d = try ctx.tensorCreate(seq * n_heads * hd * 4);
    defer ctx.tensorDestroy(&q_d);
    var q16_d = try ctx.tensorCreate(seq_pad * n_heads * hd * 2);
    defer ctx.tensorDestroy(&q16_d);
    var k_d = try ctx.tensorCreate(seq * n_kv * hd * 4);
    defer ctx.tensorDestroy(&k_d);
    var k16_d = try ctx.tensorCreate(n_kv * hd * seq_pad * 2);
    defer ctx.tensorDestroy(&k16_d);
    var v_d = try ctx.tensorCreate(seq * n_kv * hd * 4);
    defer ctx.tensorDestroy(&v_d);
    var v16_d = try ctx.tensorCreate(seq_pad * n_kv * hd * 2);
    defer ctx.tensorDestroy(&v16_d);
    const mdoff = seq_pad * n_heads * hd;
    var o_d = try ctx.tensorCreate((mdoff + n_heads * seq_pad * 2) * 4);
    defer ctx.tensorDestroy(&o_d);
    try ctx.tensorUpload(q_d, std.mem.sliceAsBytes(q));
    try ctx.tensorUpload(k_d, std.mem.sliceAsBytes(k));
    try ctx.tensorUpload(v_d, std.mem.sliceAsBytes(vv));

    try ctx.opElt(.f32_to_h16, q_d, null, null, q16_d, .{
        .u0 = seq_pad * n_heads * hd / 2,
        .u1 = seq * n_heads * hd,
        .f0 = scale,
    }, seq_pad * n_heads * hd / 2, 1, 1);
    try ctx.opElt(.gather_kmajor_h16, k_d, null, null, k16_d, .{
        .u0 = n_kv * hd * seq_pad / 2,
        .u1 = hd,
        .u2 = seq_pad,
        .u3 = seq,
        .u4 = n_kv,
    }, n_kv * hd * seq_pad / 2, 1, 1);
    try ctx.opElt(.f32_to_h16, v_d, null, null, v16_d, .{
        .u0 = seq_pad * n_kv * hd / 2,
        .u1 = seq * n_kv * hd,
        .f0 = 1.0,
    }, seq_pad * n_kv * hd / 2, 1, 1);

    const push: EltPush = .{
        .u0 = n_heads * hd,
        .u1 = seq_pad,
        .u2 = 0,
        .u3 = n_heads / n_kv,
        .u4 = n_kv * hd,
        .u5 = mdoff,
        .f0 = @bitCast(@as(u32, seq)),
    };
    try ctx.opFlash(.md, q16_d, k16_d, v16_d, o_d, push, seq_pad / 128, n_heads);
    try ctx.opFlash(.out, q16_d, k16_d, v16_d, o_d, push, seq_pad / 128, n_heads);

    const o = try gpa.alloc(f32, mdoff + n_heads * seq_pad * 2);
    defer gpa.free(o);
    try ctx.tensorDownload(o_d, std.mem.sliceAsBytes(o));

    // Reference: f16 scores from f16 operands, exact softmax, f16 P/V.
    const s_row = try gpa.alloc(f64, seq);
    defer gpa.free(s_row);
    for (0..n_heads) |h| {
        const kvh = h / (n_heads / n_kv);
        for (0..seq) |qi| {
            var rmax: f64 = -std.math.inf(f64);
            for (0..seq) |ji| {
                var dot: f64 = 0;
                for (0..hd) |kk| {
                    const qa: f16 = @floatCast(q[(qi * n_heads + h) * hd + kk] * scale);
                    const ka: f16 = @floatCast(k[(ji * n_kv + kvh) * hd + kk]);
                    dot += @as(f64, @floatCast(qa)) * @as(f64, @floatCast(ka));
                }
                const s16: f16 = @floatCast(dot);
                s_row[ji] = @floatCast(s16);
                rmax = @max(rmax, s_row[ji]);
            }
            var dsum: f64 = 0;
            for (0..seq) |ji| dsum += @exp(s_row[ji] - rmax);
            for (0..hd) |cc| {
                var want: f64 = 0;
                for (0..seq) |ji| {
                    const pa: f16 = @floatCast(@exp(s_row[ji] - rmax) / dsum);
                    const va: f16 = @floatCast(vv[(ji * n_kv + kvh) * hd + cc]);
                    want += @as(f64, @floatCast(pa)) * @as(f64, @floatCast(va));
                }
                const got = o[(qi * n_heads + h) * hd + cc];
                try std.testing.expectApproxEqAbs(@as(f32, @floatCast(want)), got, 1e-2);
            }
        }
    }
}

// Bidirectional image-block attention on Vulkan: opAttnDecodeQ35 is single
// query, so an image block is prefilled token-by-token but each query gets
// kv_end = block end (bidirectional forward) while the sliding window still
// tracks its own position. This validates the u6=kv_end path of
// attn_dsplit_gemma against a CPU reference for one such query.
test "opAttnDecodeQ35 bidirectional kv_end matches cpu reference" {
    const gpa = std.testing.allocator;
    std.Io.Dir.cwd().access(std.testing.io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    var ctx = Context.init(gpa) catch return error.SkipZigTest;
    defer ctx.deinit();

    const n_heads = 4;
    const n_kv = 2;
    const hd = 256; // gemma3 head_dim
    const window = 24;
    const kv_len = 61; // this query's causal position + 1 (query at abs pos 60)
    const kv_end = 100; // block end: query attends forward to here (bidirectional)
    const scale: f32 = 1.0 / @sqrt(@as(f32, hd));
    var prng = std.Random.DefaultPrng.init(202);
    const rand = prng.random();

    const q = try gpa.alloc(f32, n_heads * hd);
    defer gpa.free(q);
    const k = try gpa.alloc(f32, kv_end * n_kv * hd);
    defer gpa.free(k);
    const v = try gpa.alloc(f32, kv_end * n_kv * hd);
    defer gpa.free(v);
    for (q) |*x| x.* = rand.floatNorm(f32) * 0.3;
    for (k) |*x| x.* = rand.floatNorm(f32) * 0.3;
    for (v) |*x| x.* = rand.floatNorm(f32);

    // Reference: attend [kv_start, kv_end), kv_start windowed off kv_len.
    const kv_start: usize = if (kv_len > window) kv_len - window else 0;
    const ref = try gpa.alloc(f32, n_heads * hd);
    defer gpa.free(ref);
    const scores = try gpa.alloc(f32, kv_end);
    defer gpa.free(scores);
    for (0..n_heads) |h| {
        const kvh = h / (n_heads / n_kv);
        var mx: f32 = -std.math.inf(f32);
        for (kv_start..kv_end) |j| {
            var s: f32 = 0;
            for (0..hd) |c| s += q[h * hd + c] * k[(j * n_kv + kvh) * hd + c];
            scores[j] = s * scale;
            mx = @max(mx, scores[j]);
        }
        var den: f32 = 0;
        for (kv_start..kv_end) |j| {
            scores[j] = @exp(scores[j] - mx);
            den += scores[j];
        }
        for (0..hd) |c| {
            var acc: f32 = 0;
            for (kv_start..kv_end) |j| acc += scores[j] * v[(j * n_kv + kvh) * hd + c];
            ref[h * hd + c] = acc / den;
        }
    }

    var q_d = try ctx.tensorCreate(q.len * 4);
    defer ctx.tensorDestroy(&q_d);
    var k_d = try ctx.tensorCreate(k.len * 4);
    defer ctx.tensorDestroy(&k_d);
    var v_d = try ctx.tensorCreate(v.len * 4);
    defer ctx.tensorDestroy(&v_d);
    var o_d = try ctx.tensorCreate(q.len * 4);
    defer ctx.tensorDestroy(&o_d);
    var s_d = try ctx.tensorCreate(n_heads * Context.attn_decode_nsplit * (hd + 2) * 4);
    defer ctx.tensorDestroy(&s_d);
    try ctx.tensorUpload(q_d, std.mem.sliceAsBytes(q));
    try ctx.tensorUpload(k_d, std.mem.sliceAsBytes(k));
    try ctx.tensorUpload(v_d, std.mem.sliceAsBytes(v));

    try ctx.beginBatch();
    errdefer if (ctx.batching) ctx.abortBatch();
    // ring=0 (linear KV); kv_end > kv_len marks the bidirectional block.
    try ctx.opAttnDecodeQ35(q_d, k_d, v_d, o_d, s_d, n_heads, n_kv, hd, kv_len, scale, window, 0, kv_end, .f32);
    try ctx.endBatch();

    const got = try gpa.alloc(f32, q.len);
    defer gpa.free(got);
    try ctx.tensorDownload(o_d, std.mem.sliceAsBytes(got));
    var worst: f32 = 0;
    for (ref, got) |e, a| worst = @max(worst, @abs(e - a));
    try std.testing.expect(worst <= 2e-3);
}

test "vulkan q8_0 KV: kv_store_q8_0 matches host packing, attention matches reference" {
    const gpa = std.testing.allocator;
    std.Io.Dir.cwd().access(std.testing.io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    var ctx = Context.init(gpa) catch return error.SkipZigTest;
    defer ctx.deinit();
    const kvmod = @import("tp_core").kv_cache;

    const n_heads = 4;
    const n_kv = 2;
    const hd = 256; // gemma3/qwen35 head_dim
    const kv_dim = n_kv * hd;
    const kv_len = 61;
    const scale: f32 = 1.0 / @sqrt(@as(f32, hd));
    var prng = std.Random.DefaultPrng.init(203);
    const rand = prng.random();

    const q = try gpa.alloc(f32, n_heads * hd);
    defer gpa.free(q);
    const k_raw = try gpa.alloc(f32, kv_len * kv_dim);
    defer gpa.free(k_raw);
    const v_raw = try gpa.alloc(f32, kv_len * kv_dim);
    defer gpa.free(v_raw);
    for (q) |*x| x.* = rand.floatNorm(f32) * 0.3;
    for (k_raw) |*x| x.* = rand.floatNorm(f32) * 0.3;
    for (v_raw) |*x| x.* = rand.floatNorm(f32);
    @memset(k_raw[32..64], 0); // exercise the d == 0 block

    // Host truth: quantized bytes (device layout) + dequantized views.
    var cache = try kvmod.KvCache.init(gpa, 1, kv_len, kv_dim, .q8_0);
    defer cache.deinit(gpa);
    cache.write(0, k_raw, v_raw);
    cache.commit(kv_len);
    const expect_k = cache.kRowBytes(0, 0, kv_len);

    var kf_d = try ctx.tensorCreate(k_raw.len * 4);
    defer ctx.tensorDestroy(&kf_d);
    var vf_d = try ctx.tensorCreate(v_raw.len * 4);
    defer ctx.tensorDestroy(&vf_d);
    var k_d = try ctx.tensorCreate(expect_k.len);
    defer ctx.tensorDestroy(&k_d);
    var v_d = try ctx.tensorCreate(expect_k.len);
    defer ctx.tensorDestroy(&v_d);
    try ctx.tensorUpload(kf_d, std.mem.sliceAsBytes(k_raw));
    try ctx.tensorUpload(vf_d, std.mem.sliceAsBytes(v_raw));

    // Device-side quantize-store (row at a time, like appendKV).
    try ctx.beginBatch();
    errdefer if (ctx.batching) ctx.abortBatch();
    for (0..kv_len) |row| {
        try ctx.opStoreKvQ8(k_d, row * kv_dim, kf_d, row * kv_dim, kv_dim);
        try ctx.opStoreKvQ8(v_d, row * kv_dim, vf_d, row * kv_dim, kv_dim);
    }
    try ctx.endBatch();

    // Device quantization must match the host packQ80 byte for byte (RNE on
    // both sides; f32 add/mul are correctly rounded in Vulkan, and div-by-127
    // rounds correctly on every desktop driver we target).
    const got_bytes = try gpa.alloc(u8, expect_k.len);
    defer gpa.free(got_bytes);
    try ctx.tensorDownload(k_d, got_bytes);
    {
        var nbad: usize = 0;
        for (expect_k, got_bytes, 0..) |e, g, i| {
            if (e != g and nbad < 16) {
                std.debug.print("byte {d} (block {d} off {d}): want {x:0>2} got {x:0>2}\n", .{ i, i / 34, i % 34, e, g });
                nbad += 1;
            }
        }
    }
    try std.testing.expectEqualSlices(u8, expect_k, got_bytes);

    // Attention over the q8_0 cache vs the host-dequantized reference.
    const k = cache.kView(0, 0);
    const v = cache.vView(0, 0);
    const ref = try gpa.alloc(f32, n_heads * hd);
    defer gpa.free(ref);
    const scores = try gpa.alloc(f32, kv_len);
    defer gpa.free(scores);
    for (0..n_heads) |h| {
        const kvh = h / (n_heads / n_kv);
        var mx: f32 = -std.math.inf(f32);
        for (0..kv_len) |j| {
            var s: f32 = 0;
            for (0..hd) |c| s += q[h * hd + c] * k[j * kv_dim + kvh * hd + c];
            scores[j] = s * scale;
            mx = @max(mx, scores[j]);
        }
        var den: f32 = 0;
        for (0..kv_len) |j| {
            scores[j] = @exp(scores[j] - mx);
            den += scores[j];
        }
        for (0..hd) |c| {
            var acc: f32 = 0;
            for (0..kv_len) |j| acc += scores[j] * v[j * kv_dim + kvh * hd + c];
            ref[h * hd + c] = acc / den;
        }
    }

    var q_d = try ctx.tensorCreate(q.len * 4);
    defer ctx.tensorDestroy(&q_d);
    var o_d = try ctx.tensorCreate(q.len * 4);
    defer ctx.tensorDestroy(&o_d);
    var s_d = try ctx.tensorCreate(n_heads * Context.attn_decode_nsplit * (hd + 2) * 4);
    defer ctx.tensorDestroy(&s_d);
    try ctx.tensorUpload(q_d, std.mem.sliceAsBytes(q));
    try ctx.beginBatch();
    try ctx.opAttnDecodeQ35(q_d, k_d, v_d, o_d, s_d, n_heads, n_kv, hd, kv_len, scale, 0, 0, 0, .q8_0);
    try ctx.endBatch();

    const got = try gpa.alloc(f32, q.len);
    defer gpa.free(got);
    try ctx.tensorDownload(o_d, std.mem.sliceAsBytes(got));
    var worst: f32 = 0;
    for (ref, got) |e, a| worst = @max(worst, @abs(e - a));
    try std.testing.expect(worst <= 5e-3);
}

test "gpu argmax matches cpu argmax (incl. tie -> lowest index)" {
    const gpa = std.testing.allocator;
    std.Io.Dir.cwd().access(std.testing.io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    var ctx = Context.init(gpa) catch return error.SkipZigTest;
    defer ctx.deinit();

    var prng = std.Random.DefaultPrng.init(0xA11CE);
    const rand = prng.random();
    // A spread of vocab sizes incl. non-multiples of the lane count and a tiny
    // one; plus a forced exact tie to check the lowest-index tie-break.
    const vocabs = [_]usize{ 1, 7, 4096, 151936, 99991 };
    for (vocabs) |vocab| {
        const logits = try gpa.alloc(f32, vocab);
        defer gpa.free(logits);
        for (logits) |*v| v.* = rand.floatNorm(f32) * 5.0;
        if (vocab > 10) {
            // Two exact-equal maxima: CPU + GPU must both pick the lower index.
            logits[vocab / 3] = 1000.0;
            logits[vocab / 3 + 5] = 1000.0;
        }

        var lg = try ctx.tensorCreate(vocab * 4);
        var out = try ctx.tensorCreate(4);
        var sv: DeviceBuffer = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 };
        var si: DeviceBuffer = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 };
        defer {
            ctx.tensorDestroy(&lg);
            ctx.tensorDestroy(&out);
            ctx.tensorDestroy(&sv);
            ctx.tensorDestroy(&si);
        }
        try ctx.tensorUpload(lg, std.mem.sliceAsBytes(logits));
        try ctx.opArgmax(lg, vocab, out, &sv, &si);

        var got_f: [1]f32 = undefined;
        try ctx.tensorDownload(out, std.mem.sliceAsBytes(&got_f));
        const got: u32 = @intFromFloat(got_f[0]);
        try std.testing.expectEqual(sample.argmax(logits), got);
    }
}

test "gpu topk selects the true top-k set" {
    const gpa = std.testing.allocator;
    std.Io.Dir.cwd().access(std.testing.io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    var ctx = Context.init(gpa) catch return error.SkipZigTest;
    defer ctx.deinit();

    var prng = std.Random.DefaultPrng.init(0x70FF);
    const rand = prng.random();
    const vocabs = [_]usize{ 5, 4096, 151936 };
    for (vocabs) |vocab| {
        const logits = try gpa.alloc(f32, vocab);
        defer gpa.free(logits);
        for (logits) |*v| v.* = rand.floatNorm(f32) * 6.0;

        var lg = try ctx.tensorCreate(vocab * 4);
        var ov: DeviceBuffer = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 };
        var oi: DeviceBuffer = .{ .buf = .null_handle, .mem = .null_handle, .size = 0 };
        defer {
            ctx.tensorDestroy(&lg);
            ctx.tensorDestroy(&ov);
            ctx.tensorDestroy(&oi);
        }
        try ctx.tensorUpload(lg, std.mem.sliceAsBytes(logits));
        const count = try ctx.opTopK(lg, vocab, &ov, &oi);

        const vals = try gpa.alloc(f32, count);
        defer gpa.free(vals);
        const idxs = try gpa.alloc(f32, count);
        defer gpa.free(idxs);
        try ctx.tensorDownload(ov, std.mem.sliceAsBytes(vals));
        try ctx.tensorDownload(oi, std.mem.sliceAsBytes(idxs));

        // Build candidates and take exact top-k on the host, compare to the
        // reference top-k over the full logits. k spans small and large.
        for ([_]usize{ 1, 20, 512 }) |k_want| {
            const k = @min(k_want, vocab);
            const cands = try gpa.alloc(sample.Candidate, count);
            defer gpa.free(cands);
            for (cands, vals, idxs) |*c, v, id| c.* = .{ .id = @intFromFloat(id), .logit = v };
            std.mem.sort(sample.Candidate, cands, {}, sample.candDesc);
            // reference: sort a full candidate list
            const refc = try gpa.alloc(sample.Candidate, vocab);
            defer gpa.free(refc);
            for (refc, logits, 0..) |*c, v, i| c.* = .{ .id = @intCast(i), .logit = v };
            std.mem.sort(sample.Candidate, refc, {}, sample.candDesc);
            for (0..k) |r| try std.testing.expectEqual(refc[r].id, cands[r].id);
        }
    }
}

// The device penalize kernel mirrors the CPU applyPenalties formula. Vulkan
// guarantees correctly-rounded f32 mul/sub but allows ~2.5 ULP on division
// (the positive-logit repeat branch), so penalized elements compare with a
// tight tolerance while untouched elements must be bit-identical.
test "gpu penalize matches cpu applyPenalties" {
    const gpa = std.testing.allocator;
    std.Io.Dir.cwd().access(std.testing.io, "testdata/gpu-tests", .{}) catch return error.SkipZigTest;
    var ctx = Context.init(gpa) catch return error.SkipZigTest;
    defer ctx.deinit();

    var prng = std.Random.DefaultPrng.init(0x9E4A17);
    const rand = prng.random();
    const p: sample.Params = .{ .repeat_penalty = 1.3, .presence_penalty = 0.4, .frequency_penalty = 0.17, .repeat_last_n = 64 };
    const vocabs = [_]usize{ 128, 151936 };
    for (vocabs) |vocab| {
        const logits = try gpa.alloc(f32, vocab);
        defer gpa.free(logits);
        for (logits) |*v| v.* = rand.floatNorm(f32) * 5.0; // mixed signs: both penalty branches

        // A recent window with repeats (counts > 1) and out-of-order ids.
        var recent: [40]u32 = undefined;
        for (&recent) |*t| t.* = rand.uintLessThan(u32, @intCast(vocab));
        recent[7] = recent[3]; // guaranteed duplicates
        recent[21] = recent[3];

        const want = try gpa.dupe(f32, logits);
        defer gpa.free(want);
        sample.applyPenalties(want, &recent, p);

        var lg = try ctx.tensorCreate(vocab * 4);
        defer ctx.tensorDestroy(&lg);
        try ctx.tensorUpload(lg, std.mem.sliceAsBytes(logits));
        var scratch: [sample.max_penalty_window]sample.PenaltyEntry = undefined;
        const entries = sample.collectPenalties(&recent, p, &scratch);
        try ctx.opPenalize(lg, entries, p);

        const penalized = try gpa.alloc(bool, vocab);
        defer gpa.free(penalized);
        @memset(penalized, false);
        for (entries) |e| penalized[e.id] = true;

        const got = try gpa.alloc(f32, vocab);
        defer gpa.free(got);
        try ctx.tensorDownload(lg, std.mem.sliceAsBytes(got));
        for (want, got, penalized) |w, g, was_hit| {
            if (was_hit) {
                try std.testing.expect(@abs(g - w) <= @max(1e-5, 1e-6 * @abs(w)));
            } else {
                try std.testing.expectEqual(w, g); // untouched: bit-identical
            }
        }
    }
}
