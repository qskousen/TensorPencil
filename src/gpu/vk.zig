//! Hand-written Zig bindings for the Vulkan compute API.
//!
//! Every declaration in this file was hand-verified against
//! `/usr/include/vulkan/vulkan_core.h` (Vulkan 1.3.275, `VK_HEADER_VERSION 275`).
//! Only the compute subset TensorPencil needs is covered: instance/device setup,
//! buffers and device memory, shader modules, descriptor sets, compute pipelines,
//! command buffers, barriers, and fences. No windowing, images, or graphics.
//!
//! Pure Zig: no `@cImport`, no C compilation. All types are plain `extern struct`
//! / `enum` definitions with C ABI layout, and all commands are function-pointer
//! types (`Pfn*`) meant to be resolved at runtime via `vkGetInstanceProcAddr` /
//! `vkGetDeviceProcAddr` from a dynamically loaded `libvulkan.so.1`.

// ---------------------------------------------------------------------------
// Basic scalar types (vulkan_core.h lines ~94-97)
// ---------------------------------------------------------------------------

/// `typedef uint32_t VkBool32;`
pub const Bool32 = u32;
/// `typedef uint64_t VkDeviceSize;`
pub const DeviceSize = u64;
/// `typedef uint32_t VkFlags;`
pub const Flags = u32;

pub const TRUE: Bool32 = 1;
pub const FALSE: Bool32 = 0;

// ---------------------------------------------------------------------------
// API constants (vulkan_core.h lines ~124-138)
// ---------------------------------------------------------------------------

/// `#define VK_WHOLE_SIZE (~0ULL)`
pub const WHOLE_SIZE: u64 = ~@as(u64, 0);
/// `#define VK_QUEUE_FAMILY_IGNORED (~0U)`
pub const QUEUE_FAMILY_IGNORED: u32 = ~@as(u32, 0);
/// `#define VK_MAX_MEMORY_TYPES 32U`
pub const MAX_MEMORY_TYPES = 32;
/// `#define VK_MAX_MEMORY_HEAPS 16U`
pub const MAX_MEMORY_HEAPS = 16;
/// `#define VK_MAX_PHYSICAL_DEVICE_NAME_SIZE 256U`
pub const MAX_PHYSICAL_DEVICE_NAME_SIZE = 256;
/// `#define VK_MAX_EXTENSION_NAME_SIZE 256U`
pub const MAX_EXTENSION_NAME_SIZE = 256;
/// `#define VK_UUID_SIZE 16U`
pub const UUID_SIZE = 16;

/// `VK_MAKE_API_VERSION(variant, major, minor, patch)`
pub fn makeApiVersion(variant: u32, major: u32, minor: u32, patch: u32) u32 {
    return (variant << 29) | (major << 22) | (minor << 12) | patch;
}

pub const API_VERSION_1_0: u32 = makeApiVersion(0, 1, 0, 0);
pub const API_VERSION_1_1: u32 = makeApiVersion(0, 1, 1, 0);
pub const API_VERSION_1_2: u32 = makeApiVersion(0, 1, 2, 0);
pub const API_VERSION_1_3: u32 = makeApiVersion(0, 1, 3, 0);

// ---------------------------------------------------------------------------
// Handles
//
// Dispatchable handles are `typedef struct Vk*_T* Vk*;` in C — pointer-sized.
// Non-dispatchable handles are 64-bit on all targets
// (`VK_DEFINE_NON_DISPATCHABLE_HANDLE` → uint64_t on non-64-bit platforms).
// Both are represented as non-exhaustive enums so `.null_handle` replaces
// VK_NULL_HANDLE and accidental integer mixing is a compile error.
// ---------------------------------------------------------------------------

// Dispatchable (pointer-sized).
pub const Instance = enum(usize) { null_handle = 0, _ };
pub const PhysicalDevice = enum(usize) { null_handle = 0, _ };
pub const Device = enum(usize) { null_handle = 0, _ };
pub const Queue = enum(usize) { null_handle = 0, _ };
pub const CommandBuffer = enum(usize) { null_handle = 0, _ };

// Non-dispatchable (always 64-bit).
pub const Buffer = enum(u64) { null_handle = 0, _ };
pub const DeviceMemory = enum(u64) { null_handle = 0, _ };
pub const Fence = enum(u64) { null_handle = 0, _ };
pub const Semaphore = enum(u64) { null_handle = 0, _ };
pub const ShaderModule = enum(u64) { null_handle = 0, _ };
pub const Pipeline = enum(u64) { null_handle = 0, _ };
pub const PipelineLayout = enum(u64) { null_handle = 0, _ };
pub const PipelineCache = enum(u64) { null_handle = 0, _ };
pub const DescriptorSetLayout = enum(u64) { null_handle = 0, _ };
pub const DescriptorPool = enum(u64) { null_handle = 0, _ };
pub const DescriptorSet = enum(u64) { null_handle = 0, _ };
pub const CommandPool = enum(u64) { null_handle = 0, _ };
pub const Sampler = enum(u64) { null_handle = 0, _ };
pub const BufferView = enum(u64) { null_handle = 0, _ };
pub const ImageView = enum(u64) { null_handle = 0, _ };
pub const RenderPass = enum(u64) { null_handle = 0, _ };
pub const Framebuffer = enum(u64) { null_handle = 0, _ };

/// `VkAllocationCallbacks` — we never provide host allocation callbacks;
/// only `?*const AllocationCallbacks == null` is ever passed.
pub const AllocationCallbacks = opaque {};

/// `VkImageMemoryBarrier` — TensorPencil uses no images; only a null pointer
/// with `imageMemoryBarrierCount == 0` is ever passed to vkCmdPipelineBarrier.
pub const ImageMemoryBarrier = opaque {};

// ---------------------------------------------------------------------------
// VkResult (exact values from the header)
// ---------------------------------------------------------------------------

pub const Result = enum(i32) {
    success = 0,
    not_ready = 1,
    timeout = 2,
    event_set = 3,
    event_reset = 4,
    incomplete = 5,
    error_out_of_host_memory = -1,
    error_out_of_device_memory = -2,
    error_initialization_failed = -3,
    error_device_lost = -4,
    error_memory_map_failed = -5,
    error_layer_not_present = -6,
    error_extension_not_present = -7,
    error_feature_not_present = -8,
    error_incompatible_driver = -9,
    error_too_many_objects = -10,
    error_format_not_supported = -11,
    error_fragmented_pool = -12,
    error_unknown = -13,
    error_out_of_pool_memory = -1000069000,
    error_invalid_external_handle = -1000072003,
    error_fragmentation = -1000161000,
    error_invalid_opaque_capture_address = -1000257000,
    pipeline_compile_required = 1000297000,
    _,
};

// ---------------------------------------------------------------------------
// VkStructureType (only the values used by structs below; exact from header)
// ---------------------------------------------------------------------------

pub const StructureType = enum(i32) {
    application_info = 0,
    instance_create_info = 1,
    device_queue_create_info = 2,
    device_create_info = 3,
    submit_info = 4,
    memory_allocate_info = 5,
    mapped_memory_range = 6,
    fence_create_info = 8,
    buffer_create_info = 12,
    shader_module_create_info = 16,
    pipeline_shader_stage_create_info = 18,
    compute_pipeline_create_info = 29,
    pipeline_layout_create_info = 30,
    descriptor_set_layout_create_info = 32,
    descriptor_pool_create_info = 33,
    descriptor_set_allocate_info = 34,
    write_descriptor_set = 35,
    copy_descriptor_set = 36,
    command_pool_create_info = 39,
    command_buffer_allocate_info = 40,
    command_buffer_inheritance_info = 41,
    command_buffer_begin_info = 42,
    buffer_memory_barrier = 44,
    memory_barrier = 46,
    physical_device_vulkan_1_2_features = 51,
    physical_device_features_2 = 1000059000,
    physical_device_memory_properties_2 = 1000059006,
    memory_allocate_flags_info = 1000060000,
    physical_device_memory_budget_properties_ext = 1000237000,
    _,
};

// ---------------------------------------------------------------------------
// Plain enums (exact values from header)
// ---------------------------------------------------------------------------

pub const PhysicalDeviceType = enum(i32) {
    other = 0,
    integrated_gpu = 1,
    discrete_gpu = 2,
    virtual_gpu = 3,
    cpu = 4,
    _,
};

pub const SharingMode = enum(i32) {
    exclusive = 0,
    concurrent = 1,
    _,
};

pub const DescriptorType = enum(i32) {
    uniform_buffer = 6,
    storage_buffer = 7,
    _,
};

pub const PipelineBindPoint = enum(i32) {
    graphics = 0,
    compute = 1,
    _,
};

pub const CommandBufferLevel = enum(i32) {
    primary = 0,
    secondary = 1,
    _,
};

/// Only referenced by `DescriptorImageInfo` (unused by compute-buffer code).
pub const ImageLayout = enum(i32) {
    @"undefined" = 0,
    general = 1,
    _,
};

/// `VkShaderStageFlagBits` — used as a typed (single-bit) field in
/// `PipelineShaderStageCreateInfo`; a C enum, 4 bytes.
pub const ShaderStageFlagBits = enum(u32) {
    vertex = 0x00000001,
    fragment = 0x00000010,
    compute = 0x00000020,
    _,
};

// ---------------------------------------------------------------------------
// Flags (all `typedef VkFlags ...` = u32). Bit constants are grouped in
// namespaces so call sites read `vk.BufferUsage.storage_buffer | ...`.
// ---------------------------------------------------------------------------

pub const InstanceCreateFlags = Flags;
pub const DeviceCreateFlags = Flags;
pub const DeviceQueueCreateFlags = Flags;
pub const BufferCreateFlags = Flags;
pub const ShaderModuleCreateFlags = Flags;
pub const DescriptorSetLayoutCreateFlags = Flags;
pub const PipelineLayoutCreateFlags = Flags;
pub const PipelineShaderStageCreateFlags = Flags;
pub const PipelineCreateFlags = Flags;
pub const DescriptorPoolCreateFlags = Flags;
pub const MemoryMapFlags = Flags;
pub const MemoryHeapFlags = Flags;
pub const MemoryPropertyFlags = Flags;
pub const MemoryAllocateFlags = Flags;
pub const QueueFlags = Flags;
pub const BufferUsageFlags = Flags;
pub const ShaderStageFlags = Flags;
pub const PipelineStageFlags = Flags;
pub const AccessFlags = Flags;
pub const DependencyFlags = Flags;
pub const CommandPoolCreateFlags = Flags;
pub const CommandBufferUsageFlags = Flags;
pub const CommandBufferResetFlags = Flags;
pub const FenceCreateFlags = Flags;
pub const QueryControlFlags = Flags;
pub const QueryPipelineStatisticFlags = Flags;
pub const SampleCountFlags = Flags;

/// `VkQueueFlagBits`
pub const QueueFlagBits = struct {
    pub const graphics: QueueFlags = 0x00000001;
    pub const compute: QueueFlags = 0x00000002;
    pub const transfer: QueueFlags = 0x00000004;
};

/// `VkMemoryPropertyFlagBits`
pub const MemoryProperty = struct {
    pub const device_local: MemoryPropertyFlags = 0x00000001;
    pub const host_visible: MemoryPropertyFlags = 0x00000002;
    pub const host_coherent: MemoryPropertyFlags = 0x00000004;
    pub const host_cached: MemoryPropertyFlags = 0x00000008;
};

/// `VkMemoryHeapFlagBits`
pub const MemoryHeapFlagBits = struct {
    pub const device_local: MemoryHeapFlags = 0x00000001;
};

/// `VkBufferUsageFlagBits`
pub const BufferUsage = struct {
    pub const transfer_src: BufferUsageFlags = 0x00000001;
    pub const transfer_dst: BufferUsageFlags = 0x00000002;
    pub const uniform_buffer: BufferUsageFlags = 0x00000010;
    pub const storage_buffer: BufferUsageFlags = 0x00000020;
    pub const shader_device_address: BufferUsageFlags = 0x00020000;
};

/// `VkShaderStageFlagBits` as mask bits (for `stageFlags` fields).
pub const ShaderStage = struct {
    pub const compute: ShaderStageFlags = 0x00000020;
};

/// `VkPipelineStageFlagBits`
pub const PipelineStage = struct {
    pub const top_of_pipe: PipelineStageFlags = 0x00000001;
    pub const compute_shader: PipelineStageFlags = 0x00000800;
    pub const transfer: PipelineStageFlags = 0x00001000;
    pub const bottom_of_pipe: PipelineStageFlags = 0x00002000;
    pub const host: PipelineStageFlags = 0x00004000;
};

/// `VkAccessFlagBits`
pub const Access = struct {
    pub const shader_read: AccessFlags = 0x00000020;
    pub const shader_write: AccessFlags = 0x00000040;
    pub const transfer_read: AccessFlags = 0x00000800;
    pub const transfer_write: AccessFlags = 0x00001000;
    pub const host_read: AccessFlags = 0x00002000;
    pub const host_write: AccessFlags = 0x00004000;
};

/// `VkCommandPoolCreateFlagBits`
pub const CommandPoolCreate = struct {
    pub const transient: CommandPoolCreateFlags = 0x00000001;
    pub const reset_command_buffer: CommandPoolCreateFlags = 0x00000002;
};

/// `VkCommandBufferUsageFlagBits`
pub const CommandBufferUsage = struct {
    pub const one_time_submit: CommandBufferUsageFlags = 0x00000001;
};

/// `VkFenceCreateFlagBits`
pub const FenceCreate = struct {
    pub const signaled: FenceCreateFlags = 0x00000001;
};

/// `VkMemoryAllocateFlagBits`
pub const MemoryAllocate = struct {
    pub const device_address: MemoryAllocateFlags = 0x00000002;
};

// ---------------------------------------------------------------------------
// Structs — field order and types exactly as in vulkan_core.h.
// sType-carrying structs default their s_type/p_next (and zero their flags)
// so call sites stay terse.
// ---------------------------------------------------------------------------

pub const ApplicationInfo = extern struct {
    s_type: StructureType = .application_info,
    p_next: ?*const anyopaque = null,
    p_application_name: ?[*:0]const u8 = null,
    application_version: u32 = 0,
    p_engine_name: ?[*:0]const u8 = null,
    engine_version: u32 = 0,
    api_version: u32,
};

pub const InstanceCreateInfo = extern struct {
    s_type: StructureType = .instance_create_info,
    p_next: ?*const anyopaque = null,
    flags: InstanceCreateFlags = 0,
    p_application_info: ?*const ApplicationInfo = null,
    enabled_layer_count: u32 = 0,
    pp_enabled_layer_names: ?[*]const [*:0]const u8 = null,
    enabled_extension_count: u32 = 0,
    pp_enabled_extension_names: ?[*]const [*:0]const u8 = null,
};

pub const PhysicalDeviceLimits = extern struct {
    max_image_dimension_1d: u32,
    max_image_dimension_2d: u32,
    max_image_dimension_3d: u32,
    max_image_dimension_cube: u32,
    max_image_array_layers: u32,
    max_texel_buffer_elements: u32,
    max_uniform_buffer_range: u32,
    max_storage_buffer_range: u32,
    max_push_constants_size: u32,
    max_memory_allocation_count: u32,
    max_sampler_allocation_count: u32,
    buffer_image_granularity: DeviceSize,
    sparse_address_space_size: DeviceSize,
    max_bound_descriptor_sets: u32,
    max_per_stage_descriptor_samplers: u32,
    max_per_stage_descriptor_uniform_buffers: u32,
    max_per_stage_descriptor_storage_buffers: u32,
    max_per_stage_descriptor_sampled_images: u32,
    max_per_stage_descriptor_storage_images: u32,
    max_per_stage_descriptor_input_attachments: u32,
    max_per_stage_resources: u32,
    max_descriptor_set_samplers: u32,
    max_descriptor_set_uniform_buffers: u32,
    max_descriptor_set_uniform_buffers_dynamic: u32,
    max_descriptor_set_storage_buffers: u32,
    max_descriptor_set_storage_buffers_dynamic: u32,
    max_descriptor_set_sampled_images: u32,
    max_descriptor_set_storage_images: u32,
    max_descriptor_set_input_attachments: u32,
    max_vertex_input_attributes: u32,
    max_vertex_input_bindings: u32,
    max_vertex_input_attribute_offset: u32,
    max_vertex_input_binding_stride: u32,
    max_vertex_output_components: u32,
    max_tessellation_generation_level: u32,
    max_tessellation_patch_size: u32,
    max_tessellation_control_per_vertex_input_components: u32,
    max_tessellation_control_per_vertex_output_components: u32,
    max_tessellation_control_per_patch_output_components: u32,
    max_tessellation_control_total_output_components: u32,
    max_tessellation_evaluation_input_components: u32,
    max_tessellation_evaluation_output_components: u32,
    max_geometry_shader_invocations: u32,
    max_geometry_input_components: u32,
    max_geometry_output_components: u32,
    max_geometry_output_vertices: u32,
    max_geometry_total_output_components: u32,
    max_fragment_input_components: u32,
    max_fragment_output_attachments: u32,
    max_fragment_dual_src_attachments: u32,
    max_fragment_combined_output_resources: u32,
    max_compute_shared_memory_size: u32,
    max_compute_work_group_count: [3]u32,
    max_compute_work_group_invocations: u32,
    max_compute_work_group_size: [3]u32,
    sub_pixel_precision_bits: u32,
    sub_texel_precision_bits: u32,
    mipmap_precision_bits: u32,
    max_draw_indexed_index_value: u32,
    max_draw_indirect_count: u32,
    max_sampler_lod_bias: f32,
    max_sampler_anisotropy: f32,
    max_viewports: u32,
    max_viewport_dimensions: [2]u32,
    viewport_bounds_range: [2]f32,
    viewport_sub_pixel_bits: u32,
    min_memory_map_alignment: usize,
    min_texel_buffer_offset_alignment: DeviceSize,
    min_uniform_buffer_offset_alignment: DeviceSize,
    min_storage_buffer_offset_alignment: DeviceSize,
    min_texel_offset: i32,
    max_texel_offset: u32,
    min_texel_gather_offset: i32,
    max_texel_gather_offset: u32,
    min_interpolation_offset: f32,
    max_interpolation_offset: f32,
    sub_pixel_interpolation_offset_bits: u32,
    max_framebuffer_width: u32,
    max_framebuffer_height: u32,
    max_framebuffer_layers: u32,
    framebuffer_color_sample_counts: SampleCountFlags,
    framebuffer_depth_sample_counts: SampleCountFlags,
    framebuffer_stencil_sample_counts: SampleCountFlags,
    framebuffer_no_attachments_sample_counts: SampleCountFlags,
    max_color_attachments: u32,
    sampled_image_color_sample_counts: SampleCountFlags,
    sampled_image_integer_sample_counts: SampleCountFlags,
    sampled_image_depth_sample_counts: SampleCountFlags,
    sampled_image_stencil_sample_counts: SampleCountFlags,
    storage_image_sample_counts: SampleCountFlags,
    max_sample_mask_words: u32,
    timestamp_compute_and_graphics: Bool32,
    timestamp_period: f32,
    max_clip_distances: u32,
    max_cull_distances: u32,
    max_combined_clip_and_cull_distances: u32,
    discrete_queue_priorities: u32,
    point_size_range: [2]f32,
    line_width_range: [2]f32,
    point_size_granularity: f32,
    line_width_granularity: f32,
    strict_lines: Bool32,
    standard_sample_locations: Bool32,
    optimal_buffer_copy_offset_alignment: DeviceSize,
    optimal_buffer_copy_row_pitch_alignment: DeviceSize,
    non_coherent_atom_size: DeviceSize,
};

pub const PhysicalDeviceSparseProperties = extern struct {
    residency_standard_2d_block_shape: Bool32,
    residency_standard_2d_multisample_block_shape: Bool32,
    residency_standard_3d_block_shape: Bool32,
    residency_aligned_mip_size: Bool32,
    residency_non_resident_strict: Bool32,
};

pub const PhysicalDeviceProperties = extern struct {
    api_version: u32,
    driver_version: u32,
    vendor_id: u32,
    device_id: u32,
    device_type: PhysicalDeviceType,
    device_name: [MAX_PHYSICAL_DEVICE_NAME_SIZE]u8,
    pipeline_cache_uuid: [UUID_SIZE]u8,
    limits: PhysicalDeviceLimits,
    sparse_properties: PhysicalDeviceSparseProperties,
};

pub const Extent3D = extern struct {
    width: u32,
    height: u32,
    depth: u32,
};

pub const QueueFamilyProperties = extern struct {
    queue_flags: QueueFlags,
    queue_count: u32,
    timestamp_valid_bits: u32,
    min_image_transfer_granularity: Extent3D,
};

pub const MemoryType = extern struct {
    property_flags: MemoryPropertyFlags,
    heap_index: u32,
};

pub const MemoryHeap = extern struct {
    size: DeviceSize,
    flags: MemoryHeapFlags,
};

pub const PhysicalDeviceMemoryProperties = extern struct {
    memory_type_count: u32,
    memory_types: [MAX_MEMORY_TYPES]MemoryType,
    memory_heap_count: u32,
    memory_heaps: [MAX_MEMORY_HEAPS]MemoryHeap,
};

pub const PhysicalDeviceMemoryProperties2 = extern struct {
    s_type: StructureType = .physical_device_memory_properties_2,
    p_next: ?*anyopaque = null,
    memory_properties: PhysicalDeviceMemoryProperties,
};

/// `VkPhysicalDeviceMemoryBudgetPropertiesEXT` (VK_EXT_memory_budget)
pub const PhysicalDeviceMemoryBudgetPropertiesEXT = extern struct {
    s_type: StructureType = .physical_device_memory_budget_properties_ext,
    p_next: ?*anyopaque = null,
    heap_budget: [MAX_MEMORY_HEAPS]DeviceSize,
    heap_usage: [MAX_MEMORY_HEAPS]DeviceSize,
};

pub const ExtensionProperties = extern struct {
    extension_name: [MAX_EXTENSION_NAME_SIZE]u8,
    spec_version: u32,
};

pub const DeviceQueueCreateInfo = extern struct {
    s_type: StructureType = .device_queue_create_info,
    p_next: ?*const anyopaque = null,
    flags: DeviceQueueCreateFlags = 0,
    queue_family_index: u32,
    queue_count: u32,
    p_queue_priorities: [*]const f32,
};

pub const DeviceCreateInfo = extern struct {
    s_type: StructureType = .device_create_info,
    p_next: ?*const anyopaque = null,
    flags: DeviceCreateFlags = 0,
    queue_create_info_count: u32,
    p_queue_create_infos: [*]const DeviceQueueCreateInfo,
    enabled_layer_count: u32 = 0,
    pp_enabled_layer_names: ?[*]const [*:0]const u8 = null,
    enabled_extension_count: u32 = 0,
    pp_enabled_extension_names: ?[*]const [*:0]const u8 = null,
    p_enabled_features: ?*const PhysicalDeviceFeatures = null,
};

pub const PhysicalDeviceFeatures = extern struct {
    robust_buffer_access: Bool32 = FALSE,
    full_draw_index_uint32: Bool32 = FALSE,
    image_cube_array: Bool32 = FALSE,
    independent_blend: Bool32 = FALSE,
    geometry_shader: Bool32 = FALSE,
    tessellation_shader: Bool32 = FALSE,
    sample_rate_shading: Bool32 = FALSE,
    dual_src_blend: Bool32 = FALSE,
    logic_op: Bool32 = FALSE,
    multi_draw_indirect: Bool32 = FALSE,
    draw_indirect_first_instance: Bool32 = FALSE,
    depth_clamp: Bool32 = FALSE,
    depth_bias_clamp: Bool32 = FALSE,
    fill_mode_non_solid: Bool32 = FALSE,
    depth_bounds: Bool32 = FALSE,
    wide_lines: Bool32 = FALSE,
    large_points: Bool32 = FALSE,
    alpha_to_one: Bool32 = FALSE,
    multi_viewport: Bool32 = FALSE,
    sampler_anisotropy: Bool32 = FALSE,
    texture_compression_etc2: Bool32 = FALSE,
    texture_compression_astc_ldr: Bool32 = FALSE,
    texture_compression_bc: Bool32 = FALSE,
    occlusion_query_precise: Bool32 = FALSE,
    pipeline_statistics_query: Bool32 = FALSE,
    vertex_pipeline_stores_and_atomics: Bool32 = FALSE,
    fragment_stores_and_atomics: Bool32 = FALSE,
    shader_tessellation_and_geometry_point_size: Bool32 = FALSE,
    shader_image_gather_extended: Bool32 = FALSE,
    shader_storage_image_extended_formats: Bool32 = FALSE,
    shader_storage_image_multisample: Bool32 = FALSE,
    shader_storage_image_read_without_format: Bool32 = FALSE,
    shader_storage_image_write_without_format: Bool32 = FALSE,
    shader_uniform_buffer_array_dynamic_indexing: Bool32 = FALSE,
    shader_sampled_image_array_dynamic_indexing: Bool32 = FALSE,
    shader_storage_buffer_array_dynamic_indexing: Bool32 = FALSE,
    shader_storage_image_array_dynamic_indexing: Bool32 = FALSE,
    shader_clip_distance: Bool32 = FALSE,
    shader_cull_distance: Bool32 = FALSE,
    shader_float64: Bool32 = FALSE,
    shader_int64: Bool32 = FALSE,
    shader_int16: Bool32 = FALSE,
    shader_resource_residency: Bool32 = FALSE,
    shader_resource_min_lod: Bool32 = FALSE,
    sparse_binding: Bool32 = FALSE,
    sparse_residency_buffer: Bool32 = FALSE,
    sparse_residency_image_2d: Bool32 = FALSE,
    sparse_residency_image_3d: Bool32 = FALSE,
    sparse_residency_2_samples: Bool32 = FALSE,
    sparse_residency_4_samples: Bool32 = FALSE,
    sparse_residency_8_samples: Bool32 = FALSE,
    sparse_residency_16_samples: Bool32 = FALSE,
    sparse_residency_aliased: Bool32 = FALSE,
    variable_multisample_rate: Bool32 = FALSE,
    inherited_queries: Bool32 = FALSE,
};

pub const PhysicalDeviceFeatures2 = extern struct {
    s_type: StructureType = .physical_device_features_2,
    p_next: ?*anyopaque = null, // `void* pNext` (mutable) in the header
    features: PhysicalDeviceFeatures = .{},
};

pub const PhysicalDeviceVulkan12Features = extern struct {
    s_type: StructureType = .physical_device_vulkan_1_2_features,
    p_next: ?*anyopaque = null, // `void* pNext` (mutable) in the header
    sampler_mirror_clamp_to_edge: Bool32 = FALSE,
    draw_indirect_count: Bool32 = FALSE,
    storage_buffer_8bit_access: Bool32 = FALSE,
    uniform_and_storage_buffer_8bit_access: Bool32 = FALSE,
    storage_push_constant8: Bool32 = FALSE,
    shader_buffer_int64_atomics: Bool32 = FALSE,
    shader_shared_int64_atomics: Bool32 = FALSE,
    shader_float16: Bool32 = FALSE,
    shader_int8: Bool32 = FALSE,
    descriptor_indexing: Bool32 = FALSE,
    shader_input_attachment_array_dynamic_indexing: Bool32 = FALSE,
    shader_uniform_texel_buffer_array_dynamic_indexing: Bool32 = FALSE,
    shader_storage_texel_buffer_array_dynamic_indexing: Bool32 = FALSE,
    shader_uniform_buffer_array_non_uniform_indexing: Bool32 = FALSE,
    shader_sampled_image_array_non_uniform_indexing: Bool32 = FALSE,
    shader_storage_buffer_array_non_uniform_indexing: Bool32 = FALSE,
    shader_storage_image_array_non_uniform_indexing: Bool32 = FALSE,
    shader_input_attachment_array_non_uniform_indexing: Bool32 = FALSE,
    shader_uniform_texel_buffer_array_non_uniform_indexing: Bool32 = FALSE,
    shader_storage_texel_buffer_array_non_uniform_indexing: Bool32 = FALSE,
    descriptor_binding_uniform_buffer_update_after_bind: Bool32 = FALSE,
    descriptor_binding_sampled_image_update_after_bind: Bool32 = FALSE,
    descriptor_binding_storage_image_update_after_bind: Bool32 = FALSE,
    descriptor_binding_storage_buffer_update_after_bind: Bool32 = FALSE,
    descriptor_binding_uniform_texel_buffer_update_after_bind: Bool32 = FALSE,
    descriptor_binding_storage_texel_buffer_update_after_bind: Bool32 = FALSE,
    descriptor_binding_update_unused_while_pending: Bool32 = FALSE,
    descriptor_binding_partially_bound: Bool32 = FALSE,
    descriptor_binding_variable_descriptor_count: Bool32 = FALSE,
    runtime_descriptor_array: Bool32 = FALSE,
    sampler_filter_minmax: Bool32 = FALSE,
    scalar_block_layout: Bool32 = FALSE,
    imageless_framebuffer: Bool32 = FALSE,
    uniform_buffer_standard_layout: Bool32 = FALSE,
    shader_subgroup_extended_types: Bool32 = FALSE,
    separate_depth_stencil_layouts: Bool32 = FALSE,
    host_query_reset: Bool32 = FALSE,
    timeline_semaphore: Bool32 = FALSE,
    buffer_device_address: Bool32 = FALSE,
    buffer_device_address_capture_replay: Bool32 = FALSE,
    buffer_device_address_multi_device: Bool32 = FALSE,
    vulkan_memory_model: Bool32 = FALSE,
    vulkan_memory_model_device_scope: Bool32 = FALSE,
    vulkan_memory_model_availability_visibility_chains: Bool32 = FALSE,
    shader_output_viewport_index: Bool32 = FALSE,
    shader_output_layer: Bool32 = FALSE,
    subgroup_broadcast_dynamic_id: Bool32 = FALSE,
};

pub const BufferCreateInfo = extern struct {
    s_type: StructureType = .buffer_create_info,
    p_next: ?*const anyopaque = null,
    flags: BufferCreateFlags = 0,
    size: DeviceSize,
    usage: BufferUsageFlags,
    sharing_mode: SharingMode = .exclusive,
    queue_family_index_count: u32 = 0,
    p_queue_family_indices: ?[*]const u32 = null,
};

pub const MemoryRequirements = extern struct {
    size: DeviceSize,
    alignment: DeviceSize,
    memory_type_bits: u32,
};

pub const MemoryAllocateInfo = extern struct {
    s_type: StructureType = .memory_allocate_info,
    p_next: ?*const anyopaque = null,
    allocation_size: DeviceSize,
    memory_type_index: u32,
};

pub const MemoryAllocateFlagsInfo = extern struct {
    s_type: StructureType = .memory_allocate_flags_info,
    p_next: ?*const anyopaque = null,
    flags: MemoryAllocateFlags = 0,
    device_mask: u32 = 0,
};

pub const MappedMemoryRange = extern struct {
    s_type: StructureType = .mapped_memory_range,
    p_next: ?*const anyopaque = null,
    memory: DeviceMemory,
    offset: DeviceSize,
    size: DeviceSize,
};

pub const ShaderModuleCreateInfo = extern struct {
    s_type: StructureType = .shader_module_create_info,
    p_next: ?*const anyopaque = null,
    flags: ShaderModuleCreateFlags = 0,
    code_size: usize, // in bytes, must be a multiple of 4
    p_code: [*]const u32,
};

pub const DescriptorSetLayoutBinding = extern struct {
    binding: u32,
    descriptor_type: DescriptorType,
    descriptor_count: u32,
    stage_flags: ShaderStageFlags,
    p_immutable_samplers: ?[*]const Sampler = null,
};

pub const DescriptorSetLayoutCreateInfo = extern struct {
    s_type: StructureType = .descriptor_set_layout_create_info,
    p_next: ?*const anyopaque = null,
    flags: DescriptorSetLayoutCreateFlags = 0,
    binding_count: u32,
    p_bindings: ?[*]const DescriptorSetLayoutBinding,
};

pub const PushConstantRange = extern struct {
    stage_flags: ShaderStageFlags,
    offset: u32,
    size: u32,
};

pub const PipelineLayoutCreateInfo = extern struct {
    s_type: StructureType = .pipeline_layout_create_info,
    p_next: ?*const anyopaque = null,
    flags: PipelineLayoutCreateFlags = 0,
    set_layout_count: u32 = 0,
    p_set_layouts: ?[*]const DescriptorSetLayout = null,
    push_constant_range_count: u32 = 0,
    p_push_constant_ranges: ?[*]const PushConstantRange = null,
};

pub const SpecializationMapEntry = extern struct {
    constant_id: u32,
    offset: u32,
    size: usize,
};

pub const SpecializationInfo = extern struct {
    map_entry_count: u32 = 0,
    p_map_entries: ?[*]const SpecializationMapEntry = null,
    data_size: usize = 0,
    p_data: ?*const anyopaque = null,
};

pub const PipelineShaderStageCreateInfo = extern struct {
    s_type: StructureType = .pipeline_shader_stage_create_info,
    p_next: ?*const anyopaque = null,
    flags: PipelineShaderStageCreateFlags = 0,
    stage: ShaderStageFlagBits = .compute,
    module: ShaderModule,
    p_name: [*:0]const u8 = "main",
    p_specialization_info: ?*const SpecializationInfo = null,
};

pub const ComputePipelineCreateInfo = extern struct {
    s_type: StructureType = .compute_pipeline_create_info,
    p_next: ?*const anyopaque = null,
    flags: PipelineCreateFlags = 0,
    stage: PipelineShaderStageCreateInfo,
    layout: PipelineLayout,
    base_pipeline_handle: Pipeline = .null_handle,
    base_pipeline_index: i32 = -1,
};

pub const DescriptorPoolSize = extern struct {
    type: DescriptorType,
    descriptor_count: u32,
};

pub const DescriptorPoolCreateInfo = extern struct {
    s_type: StructureType = .descriptor_pool_create_info,
    p_next: ?*const anyopaque = null,
    flags: DescriptorPoolCreateFlags = 0,
    max_sets: u32,
    pool_size_count: u32,
    p_pool_sizes: [*]const DescriptorPoolSize,
};

pub const DescriptorSetAllocateInfo = extern struct {
    s_type: StructureType = .descriptor_set_allocate_info,
    p_next: ?*const anyopaque = null,
    descriptor_pool: DescriptorPool,
    descriptor_set_count: u32,
    p_set_layouts: [*]const DescriptorSetLayout,
};

pub const DescriptorBufferInfo = extern struct {
    buffer: Buffer,
    offset: DeviceSize = 0,
    range: DeviceSize = WHOLE_SIZE,
};

/// Only referenced by `WriteDescriptorSet.p_image_info` (unused for buffers).
pub const DescriptorImageInfo = extern struct {
    sampler: Sampler,
    image_view: ImageView,
    image_layout: ImageLayout,
};

pub const WriteDescriptorSet = extern struct {
    s_type: StructureType = .write_descriptor_set,
    p_next: ?*const anyopaque = null,
    dst_set: DescriptorSet,
    dst_binding: u32,
    dst_array_element: u32 = 0,
    descriptor_count: u32,
    descriptor_type: DescriptorType,
    p_image_info: ?[*]const DescriptorImageInfo = null,
    p_buffer_info: ?[*]const DescriptorBufferInfo = null,
    p_texel_buffer_view: ?[*]const BufferView = null,
};

pub const CopyDescriptorSet = extern struct {
    s_type: StructureType = .copy_descriptor_set,
    p_next: ?*const anyopaque = null,
    src_set: DescriptorSet,
    src_binding: u32,
    src_array_element: u32,
    dst_set: DescriptorSet,
    dst_binding: u32,
    dst_array_element: u32,
    descriptor_count: u32,
};

pub const CommandPoolCreateInfo = extern struct {
    s_type: StructureType = .command_pool_create_info,
    p_next: ?*const anyopaque = null,
    flags: CommandPoolCreateFlags = 0,
    queue_family_index: u32,
};

pub const CommandBufferAllocateInfo = extern struct {
    s_type: StructureType = .command_buffer_allocate_info,
    p_next: ?*const anyopaque = null,
    command_pool: CommandPool,
    level: CommandBufferLevel = .primary,
    command_buffer_count: u32,
};

pub const CommandBufferInheritanceInfo = extern struct {
    s_type: StructureType = .command_buffer_inheritance_info,
    p_next: ?*const anyopaque = null,
    render_pass: RenderPass = .null_handle,
    subpass: u32 = 0,
    framebuffer: Framebuffer = .null_handle,
    occlusion_query_enable: Bool32 = FALSE,
    query_flags: QueryControlFlags = 0,
    pipeline_statistics: QueryPipelineStatisticFlags = 0,
};

pub const CommandBufferBeginInfo = extern struct {
    s_type: StructureType = .command_buffer_begin_info,
    p_next: ?*const anyopaque = null,
    flags: CommandBufferUsageFlags = 0,
    p_inheritance_info: ?*const CommandBufferInheritanceInfo = null,
};

pub const SubmitInfo = extern struct {
    s_type: StructureType = .submit_info,
    p_next: ?*const anyopaque = null,
    wait_semaphore_count: u32 = 0,
    p_wait_semaphores: ?[*]const Semaphore = null,
    p_wait_dst_stage_mask: ?[*]const PipelineStageFlags = null,
    command_buffer_count: u32,
    p_command_buffers: [*]const CommandBuffer,
    signal_semaphore_count: u32 = 0,
    p_signal_semaphores: ?[*]const Semaphore = null,
};

pub const FenceCreateInfo = extern struct {
    s_type: StructureType = .fence_create_info,
    p_next: ?*const anyopaque = null,
    flags: FenceCreateFlags = 0,
};

pub const MemoryBarrier = extern struct {
    s_type: StructureType = .memory_barrier,
    p_next: ?*const anyopaque = null,
    src_access_mask: AccessFlags,
    dst_access_mask: AccessFlags,
};

pub const BufferMemoryBarrier = extern struct {
    s_type: StructureType = .buffer_memory_barrier,
    p_next: ?*const anyopaque = null,
    src_access_mask: AccessFlags,
    dst_access_mask: AccessFlags,
    src_queue_family_index: u32 = QUEUE_FAMILY_IGNORED,
    dst_queue_family_index: u32 = QUEUE_FAMILY_IGNORED,
    buffer: Buffer,
    offset: DeviceSize = 0,
    size: DeviceSize = WHOLE_SIZE,
};

pub const BufferCopy = extern struct {
    src_offset: DeviceSize,
    dst_offset: DeviceSize,
    size: DeviceSize,
};

// ---------------------------------------------------------------------------
// Command function-pointer types (`PFN_vk*`), resolved at runtime.
// Signatures translated 1:1 from vulkan_core.h; VKAPI_PTR is the default
// C calling convention on Linux x86_64.
// ---------------------------------------------------------------------------

/// `PFN_vkVoidFunction` — what the proc-addr loaders return; cast to a
/// concrete `Pfn*` type with `@ptrCast`.
pub const PfnVoidFunction = ?*const fn () callconv(.c) void;

/// Resolved from `libvulkan.so.1` via dlopen/dlsym (or the OS equivalent).
pub const PfnGetInstanceProcAddr = *const fn (Instance, [*:0]const u8) callconv(.c) ?*const fn () callconv(.c) void;
pub const PfnGetDeviceProcAddr = *const fn (Device, [*:0]const u8) callconv(.c) PfnVoidFunction;

// Instance / physical-device level.
pub const PfnCreateInstance = *const fn (*const InstanceCreateInfo, ?*const AllocationCallbacks, *Instance) callconv(.c) Result;
pub const PfnDestroyInstance = *const fn (Instance, ?*const AllocationCallbacks) callconv(.c) void;
pub const PfnEnumeratePhysicalDevices = *const fn (Instance, *u32, ?[*]PhysicalDevice) callconv(.c) Result;
pub const PfnGetPhysicalDeviceProperties = *const fn (PhysicalDevice, *PhysicalDeviceProperties) callconv(.c) void;
pub const PfnGetPhysicalDeviceQueueFamilyProperties = *const fn (PhysicalDevice, *u32, ?[*]QueueFamilyProperties) callconv(.c) void;
pub const PfnGetPhysicalDeviceMemoryProperties = *const fn (PhysicalDevice, *PhysicalDeviceMemoryProperties) callconv(.c) void;
pub const PfnGetPhysicalDeviceMemoryProperties2 = *const fn (PhysicalDevice, *PhysicalDeviceMemoryProperties2) callconv(.c) void;
pub const PfnEnumerateDeviceExtensionProperties = *const fn (PhysicalDevice, ?[*:0]const u8, *u32, ?[*]ExtensionProperties) callconv(.c) Result;
pub const PfnCreateDevice = *const fn (PhysicalDevice, *const DeviceCreateInfo, ?*const AllocationCallbacks, *Device) callconv(.c) Result;
pub const PfnDestroyDevice = *const fn (Device, ?*const AllocationCallbacks) callconv(.c) void;
pub const PfnGetDeviceQueue = *const fn (Device, u32, u32, *Queue) callconv(.c) void;

// Buffers and memory.
pub const PfnCreateBuffer = *const fn (Device, *const BufferCreateInfo, ?*const AllocationCallbacks, *Buffer) callconv(.c) Result;
pub const PfnDestroyBuffer = *const fn (Device, Buffer, ?*const AllocationCallbacks) callconv(.c) void;
pub const PfnGetBufferMemoryRequirements = *const fn (Device, Buffer, *MemoryRequirements) callconv(.c) void;
pub const PfnAllocateMemory = *const fn (Device, *const MemoryAllocateInfo, ?*const AllocationCallbacks, *DeviceMemory) callconv(.c) Result;
pub const PfnFreeMemory = *const fn (Device, DeviceMemory, ?*const AllocationCallbacks) callconv(.c) void;
pub const PfnBindBufferMemory = *const fn (Device, Buffer, DeviceMemory, DeviceSize) callconv(.c) Result;
pub const PfnMapMemory = *const fn (Device, DeviceMemory, DeviceSize, DeviceSize, MemoryMapFlags, *?*anyopaque) callconv(.c) Result;
pub const PfnUnmapMemory = *const fn (Device, DeviceMemory) callconv(.c) void;
pub const PfnFlushMappedMemoryRanges = *const fn (Device, u32, [*]const MappedMemoryRange) callconv(.c) Result;

// Shaders, layouts, pipelines.
pub const PfnCreateShaderModule = *const fn (Device, *const ShaderModuleCreateInfo, ?*const AllocationCallbacks, *ShaderModule) callconv(.c) Result;
pub const PfnDestroyShaderModule = *const fn (Device, ShaderModule, ?*const AllocationCallbacks) callconv(.c) void;
pub const PfnCreateDescriptorSetLayout = *const fn (Device, *const DescriptorSetLayoutCreateInfo, ?*const AllocationCallbacks, *DescriptorSetLayout) callconv(.c) Result;
pub const PfnDestroyDescriptorSetLayout = *const fn (Device, DescriptorSetLayout, ?*const AllocationCallbacks) callconv(.c) void;
pub const PfnCreatePipelineLayout = *const fn (Device, *const PipelineLayoutCreateInfo, ?*const AllocationCallbacks, *PipelineLayout) callconv(.c) Result;
pub const PfnDestroyPipelineLayout = *const fn (Device, PipelineLayout, ?*const AllocationCallbacks) callconv(.c) void;
pub const PfnCreateComputePipelines = *const fn (Device, PipelineCache, u32, [*]const ComputePipelineCreateInfo, ?*const AllocationCallbacks, [*]Pipeline) callconv(.c) Result;
pub const PfnDestroyPipeline = *const fn (Device, Pipeline, ?*const AllocationCallbacks) callconv(.c) void;

// Descriptors.
pub const PfnCreateDescriptorPool = *const fn (Device, *const DescriptorPoolCreateInfo, ?*const AllocationCallbacks, *DescriptorPool) callconv(.c) Result;
pub const PfnDestroyDescriptorPool = *const fn (Device, DescriptorPool, ?*const AllocationCallbacks) callconv(.c) void;
pub const PfnAllocateDescriptorSets = *const fn (Device, *const DescriptorSetAllocateInfo, [*]DescriptorSet) callconv(.c) Result;
pub const PfnUpdateDescriptorSets = *const fn (Device, u32, ?[*]const WriteDescriptorSet, u32, ?[*]const CopyDescriptorSet) callconv(.c) void;

// Command pools and buffers.
pub const PfnCreateCommandPool = *const fn (Device, *const CommandPoolCreateInfo, ?*const AllocationCallbacks, *CommandPool) callconv(.c) Result;
pub const PfnDestroyCommandPool = *const fn (Device, CommandPool, ?*const AllocationCallbacks) callconv(.c) void;
pub const PfnAllocateCommandBuffers = *const fn (Device, *const CommandBufferAllocateInfo, [*]CommandBuffer) callconv(.c) Result;
pub const PfnBeginCommandBuffer = *const fn (CommandBuffer, *const CommandBufferBeginInfo) callconv(.c) Result;
pub const PfnEndCommandBuffer = *const fn (CommandBuffer) callconv(.c) Result;
pub const PfnResetCommandBuffer = *const fn (CommandBuffer, CommandBufferResetFlags) callconv(.c) Result;

// Command recording.
pub const PfnCmdBindPipeline = *const fn (CommandBuffer, PipelineBindPoint, Pipeline) callconv(.c) void;
pub const PfnCmdBindDescriptorSets = *const fn (CommandBuffer, PipelineBindPoint, PipelineLayout, u32, u32, [*]const DescriptorSet, u32, ?[*]const u32) callconv(.c) void;
pub const PfnCmdDispatch = *const fn (CommandBuffer, u32, u32, u32) callconv(.c) void;
pub const PfnCmdPipelineBarrier = *const fn (CommandBuffer, PipelineStageFlags, PipelineStageFlags, DependencyFlags, u32, ?[*]const MemoryBarrier, u32, ?[*]const BufferMemoryBarrier, u32, ?*const ImageMemoryBarrier) callconv(.c) void;
pub const PfnCmdCopyBuffer = *const fn (CommandBuffer, Buffer, Buffer, u32, [*]const BufferCopy) callconv(.c) void;
pub const PfnCmdPushConstants = *const fn (CommandBuffer, PipelineLayout, ShaderStageFlags, u32, u32, *const anyopaque) callconv(.c) void;

// Submission and synchronization.
pub const PfnQueueSubmit = *const fn (Queue, u32, ?[*]const SubmitInfo, Fence) callconv(.c) Result;
pub const PfnQueueWaitIdle = *const fn (Queue) callconv(.c) Result;
pub const PfnDeviceWaitIdle = *const fn (Device) callconv(.c) Result;
pub const PfnCreateFence = *const fn (Device, *const FenceCreateInfo, ?*const AllocationCallbacks, *Fence) callconv(.c) Result;
pub const PfnDestroyFence = *const fn (Device, Fence, ?*const AllocationCallbacks) callconv(.c) void;
pub const PfnWaitForFences = *const fn (Device, u32, [*]const Fence, Bool32, u64) callconv(.c) Result;
pub const PfnResetFences = *const fn (Device, u32, [*]const Fence) callconv(.c) Result;

// ---------------------------------------------------------------------------
// Compile-time layout checks against sizes/offsets mandated by the C ABI.
// ---------------------------------------------------------------------------

comptime {
    const std = @import("std");
    // Handle sizes.
    std.debug.assert(@sizeOf(Instance) == @sizeOf(*anyopaque));
    std.debug.assert(@sizeOf(Buffer) == 8);
    // Struct sizes on x86_64 (verified against C sizeof).
    std.debug.assert(@sizeOf(PhysicalDeviceFeatures) == 55 * 4);
    std.debug.assert(@sizeOf(Extent3D) == 12);
    std.debug.assert(@sizeOf(QueueFamilyProperties) == 24);
    std.debug.assert(@sizeOf(MemoryRequirements) == 24);
    std.debug.assert(@sizeOf(BufferCopy) == 24);
    std.debug.assert(@sizeOf(PushConstantRange) == 12);
    std.debug.assert(@sizeOf(DescriptorPoolSize) == 8);
    std.debug.assert(@sizeOf(MemoryType) == 8);
    std.debug.assert(@sizeOf(MemoryHeap) == 16);
    std.debug.assert(@sizeOf(PhysicalDeviceMemoryProperties) == 4 + 32 * 8 + 4 + 16 * 16);
    std.debug.assert(@offsetOf(PhysicalDeviceProperties, "limits") == 4 * 4 + 4 + 256 + 16 + 4); // +4 padding for u64 alignment
    std.debug.assert(@sizeOf(DescriptorBufferInfo) == 24);
    std.debug.assert(@sizeOf(BufferMemoryBarrier) == 56);
    std.debug.assert(@sizeOf(SubmitInfo) == 72);
    std.debug.assert(@sizeOf(WriteDescriptorSet) == 64);
    std.debug.assert(@sizeOf(ComputePipelineCreateInfo) == 96);
    std.debug.assert(@sizeOf(PipelineShaderStageCreateInfo) == 48);
}

// --- VK_KHR_cooperative_matrix (verified against vulkan_core.h 1.3.275) ----

pub const ComponentTypeKHR = enum(i32) {
    float16 = 0,
    float32 = 1,
    float64 = 2,
    sint8 = 3,
    sint16 = 4,
    sint32 = 5,
    sint64 = 6,
    uint8 = 7,
    uint16 = 8,
    uint32 = 9,
    uint64 = 10,
    _,
};

pub const ScopeKHR = enum(i32) {
    device = 1,
    workgroup = 2,
    subgroup = 3,
    queue_family = 5,
    _,
};

pub const CooperativeMatrixPropertiesKHR = extern struct {
    s_type: StructureType = @enumFromInt(1000506001),
    p_next: ?*anyopaque = null,
    m_size: u32 = 0,
    n_size: u32 = 0,
    k_size: u32 = 0,
    a_type: ComponentTypeKHR = .float16,
    b_type: ComponentTypeKHR = .float16,
    c_type: ComponentTypeKHR = .float16,
    result_type: ComponentTypeKHR = .float16,
    saturating_accumulation: Bool32 = 0,
    scope: ScopeKHR = .subgroup,
};

pub const PhysicalDeviceCooperativeMatrixFeaturesKHR = extern struct {
    s_type: StructureType = @enumFromInt(1000506000),
    p_next: ?*anyopaque = null,
    cooperative_matrix: Bool32 = 0,
    cooperative_matrix_robust_buffer_access: Bool32 = 0,
};

pub const PfnGetPhysicalDeviceCooperativeMatrixPropertiesKHR = *const fn (PhysicalDevice, *u32, ?[*]CooperativeMatrixPropertiesKHR) callconv(.c) Result;
