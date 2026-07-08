//! cuDNN bindings — pure Zig, runtime-loaded via std.DynLib.
//!
//! Loaded like the driver (`cu.zig`) and cuBLASLt (`cublaslt.zig`): `dlopen` the
//! dispatch shim `libcudnn.so.9` (it pulls the graph/engine sub-libraries in
//! itself), hand-declare the C-ABI entry points, no linking / headers / nvcc.
//!
//! Phase-2 milestone 2.0 binds only the handle lifecycle + version/error/stream
//! surface (enough for the `cuda-libs-test` smoke). The fused-SDPA (attention,
//! 2.4) and convolution (VAE, 2.5) descriptor APIs are added in their own
//! milestones — the graph API is verbose, so it lands next to the code that
//! drives it.

const std = @import("std");
const cu = @import("cu.zig");

pub const Handle = ?*anyopaque;
pub const BackendDescriptor = ?*anyopaque; // cudnnBackendDescriptor_t = void*
pub const Status = c_int;
pub const SUCCESS: Status = 0;

/// cuDNN backend-graph enum values, machine-extracted from cudnn_graph.h (9.13;
/// stable across 9.x). Grouped so call sites read `cudnn.b.DESC_TENSOR` etc.
pub const b = struct {
    // cudnnBackendDescriptorType_t
    pub const DESC_POINTWISE: c_int = 0;
    pub const DESC_ENGINE: c_int = 2;
    pub const DESC_ENGINECFG: c_int = 3;
    pub const DESC_ENGINEHEUR: c_int = 4;
    pub const DESC_EXECUTION_PLAN: c_int = 5;
    pub const DESC_OPERATIONGRAPH: c_int = 15;
    pub const DESC_VARIANT_PACK: c_int = 16;
    pub const DESC_TENSOR: c_int = 17;
    pub const DESC_MATMUL: c_int = 18;
    pub const DESC_OPERATION_MATMUL: c_int = 19;
    pub const DESC_OPERATION_POINTWISE: c_int = 13;
    pub const DESC_OPERATION_SDPA_FWD: c_int = 41;

    // cudnnPointwiseMode_t
    pub const POINTWISE_MUL: c_int = 1;
    // extra cudnnDataType_t
    pub const DATA_INT8: c_int = 3;

    // matmul + pointwise op attributes (fused int8 GEMM + dequant graph)
    pub const ATTR_MATMUL_COMP_TYPE: c_int = 1500;
    pub const ATTR_OPERATION_MATMUL_ADESC: c_int = 1520;
    pub const ATTR_OPERATION_MATMUL_BDESC: c_int = 1521;
    pub const ATTR_OPERATION_MATMUL_CDESC: c_int = 1522;
    pub const ATTR_OPERATION_MATMUL_DESC: c_int = 1523;
    pub const ATTR_POINTWISE_MODE: c_int = 0;
    pub const ATTR_POINTWISE_MATH_PREC: c_int = 1;
    pub const ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR: c_int = 750;
    pub const ATTR_OPERATION_POINTWISE_XDESC: c_int = 751;
    pub const ATTR_OPERATION_POINTWISE_BDESC: c_int = 752;
    pub const ATTR_OPERATION_POINTWISE_YDESC: c_int = 753;

    // cudnnBackendAttributeType_t
    pub const TYPE_HANDLE: c_int = 0;
    pub const TYPE_DATA_TYPE: c_int = 1;
    pub const TYPE_BOOLEAN: c_int = 2;
    pub const TYPE_INT64: c_int = 3;
    pub const TYPE_FLOAT: c_int = 4;
    pub const TYPE_VOID_PTR: c_int = 6;
    pub const TYPE_HEUR_MODE: c_int = 8;
    pub const TYPE_POINTWISE_MODE: c_int = 14;
    pub const TYPE_BACKEND_DESCRIPTOR: c_int = 15;

    // cudnnDataType_t
    pub const DATA_FLOAT: c_int = 0;
    pub const DATA_HALF: c_int = 2;
    pub const DATA_INT32: c_int = 4;

    // cudnnTensorFormat_t / cudnnConvolutionMode_t / cudnnMathType_t /
    // cudnnConvolutionFwdAlgo_t (legacy conv API, for the VAE).
    pub const TENSOR_NCHW: c_int = 0;
    pub const TENSOR_NHWC: c_int = 1;
    pub const CONV_CROSS_CORRELATION: c_int = 1;
    pub const MATH_TENSOR_OP: c_int = 1;
    pub const CONV_FWD_ALGO_IMPLICIT_PRECOMP_GEMM: c_int = 1;

    // cudnnBackendHeurMode_t
    pub const HEUR_MODE_A: c_int = 3;
    pub const HEUR_MODE_INSTANT: c_int = 0;

    // cudnnBackendAttributeName_t (subset we set)
    pub const ATTR_ENGINEHEUR_MODE: c_int = 200;
    pub const ATTR_ENGINEHEUR_OPERATION_GRAPH: c_int = 201;
    pub const ATTR_ENGINEHEUR_RESULTS: c_int = 202;
    pub const ATTR_ENGINECFG_ENGINE: c_int = 300;
    pub const ATTR_EXECUTION_PLAN_HANDLE: c_int = 400; // deprecated but still accepted
    pub const ATTR_EXECUTION_PLAN_ENGINE_CONFIG: c_int = 401;
    pub const ATTR_EXECUTION_PLAN_WORKSPACE_SIZE: c_int = 402;
    pub const ATTR_TENSOR_BYTE_ALIGNMENT: c_int = 900;
    pub const ATTR_TENSOR_DATA_TYPE: c_int = 901;
    pub const ATTR_TENSOR_DIMENSIONS: c_int = 902;
    pub const ATTR_TENSOR_STRIDES: c_int = 903;
    pub const ATTR_TENSOR_UNIQUE_ID: c_int = 906;
    pub const ATTR_TENSOR_IS_VIRTUAL: c_int = 907;
    pub const ATTR_TENSOR_IS_BY_VALUE: c_int = 908;
    pub const ATTR_ENGINE_OPERATION_GRAPH: c_int = 1300;
    pub const ATTR_ENGINE_GLOBAL_INDEX: c_int = 1301;
    pub const ATTR_OPERATIONGRAPH_OPS: c_int = 801;
    pub const ATTR_OPERATIONGRAPH_HANDLE: c_int = 800;
    pub const ATTR_VARIANT_PACK_UNIQUE_IDS: c_int = 1000;
    pub const ATTR_VARIANT_PACK_DATA_POINTERS: c_int = 1001;
    pub const ATTR_VARIANT_PACK_WORKSPACE: c_int = 1003;
    pub const ATTR_OPERATION_SDPA_FWD_QDESC: c_int = 2800;
    pub const ATTR_OPERATION_SDPA_FWD_KDESC: c_int = 2801;
    pub const ATTR_OPERATION_SDPA_FWD_VDESC: c_int = 2802;
    pub const ATTR_OPERATION_SDPA_FWD_ODESC: c_int = 2803;
    pub const ATTR_OPERATION_SDPA_FWD_STATSDESC: c_int = 2804;
    pub const ATTR_OPERATION_SDPA_FWD_SCALEDESC: c_int = 2805;
};

const PFN_Create = *const fn (*Handle) callconv(.c) Status;
const PFN_Destroy = *const fn (Handle) callconv(.c) Status;
const PFN_GetVersion = *const fn () callconv(.c) usize;
const PFN_SetStream = *const fn (Handle, cu.CUstream) callconv(.c) Status;
const PFN_GetErrorString = *const fn (Status) callconv(.c) [*:0]const u8;
// Legacy conv-API opaque descriptor handles (all void* typedefs).
pub const TensorDescriptor = ?*anyopaque;
pub const FilterDescriptor = ?*anyopaque;
pub const ConvDescriptor = ?*anyopaque;

const PFN_CreateTensorDescriptor = *const fn (*TensorDescriptor) callconv(.c) Status;
const PFN_SetTensor4dDescriptor = *const fn (TensorDescriptor, c_int, c_int, c_int, c_int, c_int, c_int) callconv(.c) Status;
const PFN_DestroyTensorDescriptor = *const fn (TensorDescriptor) callconv(.c) Status;
const PFN_CreateFilterDescriptor = *const fn (*FilterDescriptor) callconv(.c) Status;
const PFN_SetFilter4dDescriptor = *const fn (FilterDescriptor, c_int, c_int, c_int, c_int, c_int, c_int) callconv(.c) Status;
const PFN_DestroyFilterDescriptor = *const fn (FilterDescriptor) callconv(.c) Status;
const PFN_CreateConvolutionDescriptor = *const fn (*ConvDescriptor) callconv(.c) Status;
const PFN_SetConvolution2dDescriptor = *const fn (ConvDescriptor, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int) callconv(.c) Status;
const PFN_SetConvolutionMathType = *const fn (ConvDescriptor, c_int) callconv(.c) Status;
const PFN_DestroyConvolutionDescriptor = *const fn (ConvDescriptor) callconv(.c) Status;
const PFN_GetConvolutionForwardWorkspaceSize = *const fn (Handle, TensorDescriptor, FilterDescriptor, ConvDescriptor, TensorDescriptor, c_int, *usize) callconv(.c) Status;
const PFN_ConvolutionForward = *const fn (Handle, *const anyopaque, TensorDescriptor, ?*const anyopaque, FilterDescriptor, ?*const anyopaque, ConvDescriptor, c_int, ?*anyopaque, usize, *const anyopaque, TensorDescriptor, ?*anyopaque) callconv(.c) Status;

const PFN_BackendCreateDescriptor = *const fn (c_int, *BackendDescriptor) callconv(.c) Status;
const PFN_BackendDestroyDescriptor = *const fn (BackendDescriptor) callconv(.c) Status;
const PFN_BackendSetAttribute = *const fn (BackendDescriptor, c_int, c_int, i64, ?*const anyopaque) callconv(.c) Status;
const PFN_BackendGetAttribute = *const fn (BackendDescriptor, c_int, c_int, i64, *i64, ?*anyopaque) callconv(.c) Status;
const PFN_BackendFinalize = *const fn (BackendDescriptor) callconv(.c) Status;
const PFN_BackendExecute = *const fn (Handle, BackendDescriptor, BackendDescriptor) callconv(.c) Status;

pub const Error = error{CudnnError};

/// Resolved cuDNN entry points (one per process, owned by the caller).
pub const Api = struct {
    lib: std.DynLib,

    cudnnCreate: PFN_Create,
    cudnnDestroy: PFN_Destroy,
    cudnnGetVersion: PFN_GetVersion,
    cudnnSetStream: PFN_SetStream,
    cudnnGetErrorString: PFN_GetErrorString,
    cudnnBackendCreateDescriptor: PFN_BackendCreateDescriptor,
    cudnnBackendDestroyDescriptor: PFN_BackendDestroyDescriptor,
    cudnnBackendSetAttribute: PFN_BackendSetAttribute,
    cudnnBackendGetAttribute: PFN_BackendGetAttribute,
    cudnnBackendFinalize: PFN_BackendFinalize,
    cudnnBackendExecute: PFN_BackendExecute,
    cudnnCreateTensorDescriptor: PFN_CreateTensorDescriptor,
    cudnnSetTensor4dDescriptor: PFN_SetTensor4dDescriptor,
    cudnnDestroyTensorDescriptor: PFN_DestroyTensorDescriptor,
    cudnnCreateFilterDescriptor: PFN_CreateFilterDescriptor,
    cudnnSetFilter4dDescriptor: PFN_SetFilter4dDescriptor,
    cudnnDestroyFilterDescriptor: PFN_DestroyFilterDescriptor,
    cudnnCreateConvolutionDescriptor: PFN_CreateConvolutionDescriptor,
    cudnnSetConvolution2dDescriptor: PFN_SetConvolution2dDescriptor,
    cudnnSetConvolutionMathType: PFN_SetConvolutionMathType,
    cudnnDestroyConvolutionDescriptor: PFN_DestroyConvolutionDescriptor,
    cudnnGetConvolutionForwardWorkspaceSize: PFN_GetConvolutionForwardWorkspaceSize,
    cudnnConvolutionForward: PFN_ConvolutionForward,

    /// dlopen cuDNN's dispatch shim and resolve the symbols (handle lifecycle +
    /// the backend-graph API used for fused SDPA attention).
    pub fn load() Error!Api {
        var lib = std.DynLib.open("libcudnn.so.9") catch
            std.DynLib.open("/usr/lib/x86_64-linux-gnu/libcudnn.so.9") catch
            std.DynLib.open("libcudnn.so") catch return error.CudnnError;
        errdefer lib.close();

        var api: Api = undefined;
        api.lib = lib;
        inline for (.{
            "cudnnCreate",
            "cudnnDestroy",
            "cudnnGetVersion",
            "cudnnSetStream",
            "cudnnGetErrorString",
            "cudnnBackendCreateDescriptor",
            "cudnnBackendDestroyDescriptor",
            "cudnnBackendSetAttribute",
            "cudnnBackendGetAttribute",
            "cudnnBackendFinalize",
            "cudnnBackendExecute",
            "cudnnCreateTensorDescriptor",
            "cudnnSetTensor4dDescriptor",
            "cudnnDestroyTensorDescriptor",
            "cudnnCreateFilterDescriptor",
            "cudnnSetFilter4dDescriptor",
            "cudnnDestroyFilterDescriptor",
            "cudnnCreateConvolutionDescriptor",
            "cudnnSetConvolution2dDescriptor",
            "cudnnSetConvolutionMathType",
            "cudnnDestroyConvolutionDescriptor",
            "cudnnGetConvolutionForwardWorkspaceSize",
            "cudnnConvolutionForward",
        }) |name| {
            const T = @TypeOf(@field(api, name));
            const sym: [:0]const u8 = name;
            @field(api, name) = api.lib.lookup(T, sym) orelse return error.CudnnError;
        }
        return api;
    }

    pub fn deinit(self: *Api) void {
        self.lib.close();
    }

    pub fn errString(self: *const Api, s: Status) []const u8 {
        return std.mem.span(self.cudnnGetErrorString(s));
    }

    pub fn check(self: *const Api, s: Status, comptime what: []const u8) Error!void {
        if (s == SUCCESS) return;
        std.debug.print("cuDNN {s} failed: {s} ({d})\n", .{ what, self.errString(s), s });
        return error.CudnnError;
    }
};

/// A finalized cuDNN fused-SDPA-forward execution plan for one (b,hq,hkv,s,d)
/// shape: O[b,hq,s,d] = softmax(scale · Q·Kᵀ) · V, non-causal, f16 in/out with
/// f32 softmax. Built once per shape (the heuristic + plan finalize are the
/// expensive host steps); `execute` just binds pointers into a variant pack.
///
/// Tensor UIDs are fixed: Q=1 K=2 V=3 O=4 SCALE=5. Q/K/V/O are device f16 in
/// [b,h,s,d] dim order with caller-supplied strides (our DiT tensors are stored
/// [s,h,d], so the s/h strides are swapped vs a contiguous [b,h,s,d]); the
/// attention scale is a by-value host f32.
pub const SdpaPlan = struct {
    plan: BackendDescriptor = null,
    // descriptors kept alive for the plan's lifetime (freed in deinit).
    keep: [16]BackendDescriptor = @splat(null),
    n_keep: usize = 0,
    workspace_bytes: usize = 0,

    const uid_q: i64 = 1;
    const uid_k: i64 = 2;
    const uid_v: i64 = 3;
    const uid_o: i64 = 4;
    const uid_scale: i64 = 5;

    fn track(self: *SdpaPlan, d: BackendDescriptor) void {
        if (self.n_keep < self.keep.len) {
            self.keep[self.n_keep] = d;
            self.n_keep += 1;
        }
    }

    /// Build a finalized f16 tensor descriptor [b,h,s,d] with the given strides
    /// and unique id. `by_value` scalars live on the host; `virtual` intermediates
    /// have no data pointer.
    fn tensor(self: *SdpaPlan, api: *const Api, uid: i64, dtype: c_int, dims: *const [4]i64, strides: *const [4]i64, by_value: bool) Error!BackendDescriptor {
        var t: BackendDescriptor = null;
        try api.check(api.cudnnBackendCreateDescriptor(b.DESC_TENSOR, &t), "CreateDescriptor(tensor)");
        self.track(t);
        var dt = dtype;
        var id = uid;
        var al: i64 = 16;
        var bv: u8 = if (by_value) 1 else 0;
        var virt: u8 = 0;
        try api.check(api.cudnnBackendSetAttribute(t, b.ATTR_TENSOR_DATA_TYPE, b.TYPE_DATA_TYPE, 1, &dt), "tensor dtype");
        try api.check(api.cudnnBackendSetAttribute(t, b.ATTR_TENSOR_DIMENSIONS, b.TYPE_INT64, 4, dims), "tensor dims");
        try api.check(api.cudnnBackendSetAttribute(t, b.ATTR_TENSOR_STRIDES, b.TYPE_INT64, 4, strides), "tensor strides");
        try api.check(api.cudnnBackendSetAttribute(t, b.ATTR_TENSOR_UNIQUE_ID, b.TYPE_INT64, 1, &id), "tensor uid");
        try api.check(api.cudnnBackendSetAttribute(t, b.ATTR_TENSOR_BYTE_ALIGNMENT, b.TYPE_INT64, 1, &al), "tensor align");
        try api.check(api.cudnnBackendSetAttribute(t, b.ATTR_TENSOR_IS_VIRTUAL, b.TYPE_BOOLEAN, 1, &virt), "tensor virtual");
        try api.check(api.cudnnBackendSetAttribute(t, b.ATTR_TENSOR_IS_BY_VALUE, b.TYPE_BOOLEAN, 1, &bv), "tensor byvalue");
        try api.check(api.cudnnBackendFinalize(t), "Finalize(tensor)");
        return t;
    }

    pub fn build(api: *const Api, handle: Handle, bsz: usize, hq: usize, hkv: usize, s: usize, d: usize) Error!SdpaPlan {
        var self: SdpaPlan = .{};
        errdefer self.deinit(api);

        const S: i64 = @intCast(s);
        const D: i64 = @intCast(d);
        const HQ: i64 = @intCast(hq);
        const HKV: i64 = @intCast(hkv);
        const B: i64 = @intCast(bsz);
        // Our tensors are stored [b][s][h][d] (h and d contiguous per token), so
        // in [b,h,s,d] dim order the strides are: b=h*s*d, h=d, s=h*d, d=1.
        const q_dims = [4]i64{ B, HQ, S, D };
        const q_str = [4]i64{ HQ * S * D, D, HQ * D, 1 };
        const kv_dims = [4]i64{ B, HKV, S, D };
        const kv_str = [4]i64{ HKV * S * D, D, HKV * D, 1 };
        const o_dims = [4]i64{ B, HQ, S, D };
        const o_str = [4]i64{ HQ * S * D, D, HQ * D, 1 };
        const sc_dims = [4]i64{ 1, 1, 1, 1 };
        const sc_str = [4]i64{ 1, 1, 1, 1 };

        const qt = try self.tensor(api, uid_q, b.DATA_HALF, &q_dims, &q_str, false);
        const kt = try self.tensor(api, uid_k, b.DATA_HALF, &kv_dims, &kv_str, false);
        const vt = try self.tensor(api, uid_v, b.DATA_HALF, &kv_dims, &kv_str, false);
        const ot = try self.tensor(api, uid_o, b.DATA_HALF, &o_dims, &o_str, false);
        const sct = try self.tensor(api, uid_scale, b.DATA_FLOAT, &sc_dims, &sc_str, true);

        var op: BackendDescriptor = null;
        try api.check(api.cudnnBackendCreateDescriptor(b.DESC_OPERATION_SDPA_FWD, &op), "CreateDescriptor(sdpa)");
        self.track(op);
        var q_ = qt;
        var k_ = kt;
        var v_ = vt;
        var o_ = ot;
        var sc_ = sct;
        try api.check(api.cudnnBackendSetAttribute(op, b.ATTR_OPERATION_SDPA_FWD_QDESC, b.TYPE_BACKEND_DESCRIPTOR, 1, @ptrCast(&q_)), "sdpa q");
        try api.check(api.cudnnBackendSetAttribute(op, b.ATTR_OPERATION_SDPA_FWD_KDESC, b.TYPE_BACKEND_DESCRIPTOR, 1, @ptrCast(&k_)), "sdpa k");
        try api.check(api.cudnnBackendSetAttribute(op, b.ATTR_OPERATION_SDPA_FWD_VDESC, b.TYPE_BACKEND_DESCRIPTOR, 1, @ptrCast(&v_)), "sdpa v");
        try api.check(api.cudnnBackendSetAttribute(op, b.ATTR_OPERATION_SDPA_FWD_ODESC, b.TYPE_BACKEND_DESCRIPTOR, 1, @ptrCast(&o_)), "sdpa o");
        try api.check(api.cudnnBackendSetAttribute(op, b.ATTR_OPERATION_SDPA_FWD_SCALEDESC, b.TYPE_BACKEND_DESCRIPTOR, 1, @ptrCast(&sc_)), "sdpa scale");
        try api.check(api.cudnnBackendFinalize(op), "Finalize(sdpa)");

        var graph: BackendDescriptor = null;
        try api.check(api.cudnnBackendCreateDescriptor(b.DESC_OPERATIONGRAPH, &graph), "CreateDescriptor(graph)");
        self.track(graph);
        var h = handle;
        var op_ = op;
        try api.check(api.cudnnBackendSetAttribute(graph, b.ATTR_OPERATIONGRAPH_HANDLE, b.TYPE_HANDLE, 1, @ptrCast(&h)), "graph handle");
        try api.check(api.cudnnBackendSetAttribute(graph, b.ATTR_OPERATIONGRAPH_OPS, b.TYPE_BACKEND_DESCRIPTOR, 1, @ptrCast(&op_)), "graph ops");
        try api.check(api.cudnnBackendFinalize(graph), "Finalize(graph)");

        var heur: BackendDescriptor = null;
        try api.check(api.cudnnBackendCreateDescriptor(b.DESC_ENGINEHEUR, &heur), "CreateDescriptor(heur)");
        self.track(heur);
        var g_ = graph;
        var mode = b.HEUR_MODE_A;
        try api.check(api.cudnnBackendSetAttribute(heur, b.ATTR_ENGINEHEUR_OPERATION_GRAPH, b.TYPE_BACKEND_DESCRIPTOR, 1, @ptrCast(&g_)), "heur graph");
        try api.check(api.cudnnBackendSetAttribute(heur, b.ATTR_ENGINEHEUR_MODE, b.TYPE_HEUR_MODE, 1, @ptrCast(&mode)), "heur mode");
        try api.check(api.cudnnBackendFinalize(heur), "Finalize(heur)");

        // Pull the ranked engine configs (pre-created, heuristic fills them).
        var cfgs: [16]BackendDescriptor = @splat(null);
        for (&cfgs) |*c| try api.check(api.cudnnBackendCreateDescriptor(b.DESC_ENGINECFG, c), "CreateDescriptor(enginecfg)");
        var returned: i64 = 0;
        try api.check(api.cudnnBackendGetAttribute(heur, b.ATTR_ENGINEHEUR_RESULTS, b.TYPE_BACKEND_DESCRIPTOR, cfgs.len, &returned, @ptrCast(&cfgs)), "heur results");
        if (returned < 1) {
            std.debug.print("cuDNN SDPA: heuristic returned no engine configs\n", .{});
            for (cfgs) |c| _ = api.cudnnBackendDestroyDescriptor(c);
            return error.CudnnError;
        }

        // Try configs in rank order until one finalizes into an execution plan.
        var plan: BackendDescriptor = null;
        var chosen: usize = 0;
        var i: usize = 0;
        while (i < @as(usize, @intCast(returned))) : (i += 1) {
            var p: BackendDescriptor = null;
            if (api.cudnnBackendCreateDescriptor(b.DESC_EXECUTION_PLAN, &p) != SUCCESS) continue;
            var hh = handle;
            var cfg = cfgs[i];
            _ = api.cudnnBackendSetAttribute(p, b.ATTR_EXECUTION_PLAN_HANDLE, b.TYPE_HANDLE, 1, @ptrCast(&hh));
            _ = api.cudnnBackendSetAttribute(p, b.ATTR_EXECUTION_PLAN_ENGINE_CONFIG, b.TYPE_BACKEND_DESCRIPTOR, 1, @ptrCast(&cfg));
            if (api.cudnnBackendFinalize(p) == SUCCESS) {
                plan = p;
                chosen = i;
                break;
            }
            _ = api.cudnnBackendDestroyDescriptor(p);
        }
        for (cfgs, 0..) |c, ci| {
            if (ci != chosen or plan == null) _ = api.cudnnBackendDestroyDescriptor(c);
        }
        if (plan == null) {
            std.debug.print("cuDNN SDPA: no engine config finalized into a plan ({d} tried)\n", .{returned});
            return error.CudnnError;
        }
        self.track(cfgs[chosen]); // engine config backs the plan; keep it alive
        self.track(plan);
        self.plan = plan;

        var ws: i64 = 0;
        var n: i64 = 0;
        try api.check(api.cudnnBackendGetAttribute(plan, b.ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, b.TYPE_INT64, 1, &n, &ws), "plan workspace");
        self.workspace_bytes = @intCast(ws);
        return self;
    }

    /// Bind pointers + run. q/k/v/o are device f16; `scale` is a host f32 (by-value);
    /// `workspace` is a device pointer with >= workspace_bytes (may be null if 0).
    pub fn execute(self: *const SdpaPlan, api: *const Api, handle: Handle, q: u64, k: u64, v: u64, o: u64, scale: *const f32, workspace: u64) Error!void {
        var vpack: BackendDescriptor = null;
        try api.check(api.cudnnBackendCreateDescriptor(b.DESC_VARIANT_PACK, &vpack), "CreateDescriptor(vpack)");
        defer _ = api.cudnnBackendDestroyDescriptor(vpack);
        var uids = [_]i64{ uid_q, uid_k, uid_v, uid_o, uid_scale };
        var ptrs = [_]?*anyopaque{
            @ptrFromInt(q), @ptrFromInt(k), @ptrFromInt(v), @ptrFromInt(o), @constCast(@ptrCast(scale)),
        };
        var ws_ptr: ?*anyopaque = if (workspace != 0) @ptrFromInt(workspace) else null;
        try api.check(api.cudnnBackendSetAttribute(vpack, b.ATTR_VARIANT_PACK_UNIQUE_IDS, b.TYPE_INT64, uids.len, @ptrCast(&uids)), "vpack uids");
        try api.check(api.cudnnBackendSetAttribute(vpack, b.ATTR_VARIANT_PACK_DATA_POINTERS, b.TYPE_VOID_PTR, ptrs.len, @ptrCast(&ptrs)), "vpack ptrs");
        try api.check(api.cudnnBackendSetAttribute(vpack, b.ATTR_VARIANT_PACK_WORKSPACE, b.TYPE_VOID_PTR, 1, @ptrCast(&ws_ptr)), "vpack workspace");
        try api.check(api.cudnnBackendFinalize(vpack), "Finalize(vpack)");
        try api.check(api.cudnnBackendExecute(handle, self.plan, vpack), "Execute");
    }

    pub fn deinit(self: *SdpaPlan, api: *const Api) void {
        var i: usize = self.n_keep;
        while (i > 0) {
            i -= 1;
            _ = api.cudnnBackendDestroyDescriptor(self.keep[i]);
        }
        self.* = .{};
    }
};

/// A legacy-API 3×3 (pad 1, stride 1, cross-correlation) NHWC convolution for the
/// VAE: f16 X/W tensor-core conv (f32 accumulate) → f16 Y, algo
/// IMPLICIT_PRECOMP_GEMM (no im2col materialization). Descriptors are cheap to
/// build (no heuristic), so this is created/destroyed per conv call.
pub const ConvPlan = struct {
    x: TensorDescriptor = null,
    y: TensorDescriptor = null,
    w: FilterDescriptor = null,
    conv: ConvDescriptor = null,
    workspace_bytes: usize = 0,

    pub fn build(api: *const Api, handle: Handle, h: usize, w: usize, ci: usize, co: usize) Error!ConvPlan {
        var self: ConvPlan = .{};
        errdefer self.deinit(api);
        try api.check(api.cudnnCreateTensorDescriptor(&self.x), "CreateTensor(x)");
        try api.check(api.cudnnCreateTensorDescriptor(&self.y), "CreateTensor(y)");
        try api.check(api.cudnnCreateFilterDescriptor(&self.w), "CreateFilter");
        try api.check(api.cudnnCreateConvolutionDescriptor(&self.conv), "CreateConv");
        const hi: c_int = @intCast(h);
        const wi: c_int = @intCast(w);
        const cii: c_int = @intCast(ci);
        const coi: c_int = @intCast(co);
        // X/Y are NHWC f16 (N=1); W is [co][3][3][ci] NHWC f16.
        try api.check(api.cudnnSetTensor4dDescriptor(self.x, b.TENSOR_NHWC, b.DATA_HALF, 1, cii, hi, wi), "SetTensor(x)");
        try api.check(api.cudnnSetTensor4dDescriptor(self.y, b.TENSOR_NHWC, b.DATA_HALF, 1, coi, hi, wi), "SetTensor(y)");
        try api.check(api.cudnnSetFilter4dDescriptor(self.w, b.DATA_HALF, b.TENSOR_NHWC, coi, cii, 3, 3), "SetFilter");
        try api.check(api.cudnnSetConvolution2dDescriptor(self.conv, 1, 1, 1, 1, 1, 1, b.CONV_CROSS_CORRELATION, b.DATA_FLOAT), "SetConv2d");
        try api.check(api.cudnnSetConvolutionMathType(self.conv, b.MATH_TENSOR_OP), "SetMathType");
        var ws: usize = 0;
        try api.check(api.cudnnGetConvolutionForwardWorkspaceSize(handle, self.x, self.w, self.conv, self.y, b.CONV_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, &ws), "ConvWorkspace");
        self.workspace_bytes = ws;
        return self;
    }

    pub fn execute(self: *const ConvPlan, api: *const Api, handle: Handle, x: u64, w: u64, y: u64, workspace: u64) Error!void {
        var alpha: f32 = 1;
        var beta: f32 = 0;
        try api.check(api.cudnnConvolutionForward(
            handle,
            @ptrCast(&alpha),
            self.x,
            @ptrFromInt(x),
            self.w,
            @ptrFromInt(w),
            self.conv,
            b.CONV_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
            if (workspace != 0) @ptrFromInt(workspace) else null,
            self.workspace_bytes,
            @ptrCast(&beta),
            self.y,
            @ptrFromInt(y),
        ), "ConvolutionForward");
    }

    pub fn deinit(self: *ConvPlan, api: *const Api) void {
        if (self.conv != null) _ = api.cudnnDestroyConvolutionDescriptor(self.conv);
        if (self.w != null) _ = api.cudnnDestroyFilterDescriptor(self.w);
        if (self.y != null) _ = api.cudnnDestroyTensorDescriptor(self.y);
        if (self.x != null) _ = api.cudnnDestroyTensorDescriptor(self.x);
        self.* = .{};
    }
};

/// Turn a finalized operation graph into an execution plan via HEUR_MODE_A,
/// trying ranked engine configs until one finalizes. Returns the plan, the
/// engine config backing it (caller must keep both alive), and the workspace
/// size. Shared by the graph-based plans.
fn planFromGraph(api: *const Api, handle: Handle, graph: BackendDescriptor) Error!struct { plan: BackendDescriptor, cfg: BackendDescriptor, ws: usize } {
    var heur: BackendDescriptor = null;
    try api.check(api.cudnnBackendCreateDescriptor(b.DESC_ENGINEHEUR, &heur), "CreateDescriptor(heur)");
    defer _ = api.cudnnBackendDestroyDescriptor(heur);
    var g_ = graph;
    var mode = b.HEUR_MODE_A;
    try api.check(api.cudnnBackendSetAttribute(heur, b.ATTR_ENGINEHEUR_OPERATION_GRAPH, b.TYPE_BACKEND_DESCRIPTOR, 1, @ptrCast(&g_)), "heur graph");
    try api.check(api.cudnnBackendSetAttribute(heur, b.ATTR_ENGINEHEUR_MODE, b.TYPE_HEUR_MODE, 1, @ptrCast(&mode)), "heur mode");
    try api.check(api.cudnnBackendFinalize(heur), "Finalize(heur)");

    var cfgs: [16]BackendDescriptor = @splat(null);
    for (&cfgs) |*c| try api.check(api.cudnnBackendCreateDescriptor(b.DESC_ENGINECFG, c), "CreateDescriptor(enginecfg)");
    var returned: i64 = 0;
    try api.check(api.cudnnBackendGetAttribute(heur, b.ATTR_ENGINEHEUR_RESULTS, b.TYPE_BACKEND_DESCRIPTOR, cfgs.len, &returned, @ptrCast(&cfgs)), "heur results");

    var plan: BackendDescriptor = null;
    var chosen: usize = 0;
    var i: usize = 0;
    while (i < @as(usize, @intCast(returned))) : (i += 1) {
        var p: BackendDescriptor = null;
        if (api.cudnnBackendCreateDescriptor(b.DESC_EXECUTION_PLAN, &p) != SUCCESS) continue;
        var hh = handle;
        var cfg = cfgs[i];
        _ = api.cudnnBackendSetAttribute(p, b.ATTR_EXECUTION_PLAN_HANDLE, b.TYPE_HANDLE, 1, @ptrCast(&hh));
        _ = api.cudnnBackendSetAttribute(p, b.ATTR_EXECUTION_PLAN_ENGINE_CONFIG, b.TYPE_BACKEND_DESCRIPTOR, 1, @ptrCast(&cfg));
        if (api.cudnnBackendFinalize(p) == SUCCESS) {
            plan = p;
            chosen = i;
            break;
        }
        _ = api.cudnnBackendDestroyDescriptor(p);
    }
    for (cfgs, 0..) |c, ci| {
        if (ci != chosen or plan == null) _ = api.cudnnBackendDestroyDescriptor(c);
    }
    if (plan == null) return error.CudnnError;
    var ws: i64 = 0;
    var n: i64 = 0;
    try api.check(api.cudnnBackendGetAttribute(plan, b.ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, b.TYPE_INT64, 1, &n, &ws), "plan workspace");
    return .{ .plan = plan, .cfg = cfgs[chosen], .ws = @intCast(ws) };
}

/// Fused int8 GEMM + per-row×per-col dequant, as one cuDNN op graph — the
/// dlopen-compatible alternative to a CUTLASS epilogue. Computes
/// D[m][n] (f32) = (A[m][k]·B[k][n] in s32) · act_scale[m] · weight_scale[n],
/// so the s32 accumulator never round-trips to DRAM and there is no separate
/// `irescale` pass. B is the weight stored [n][k] row-major, viewed as [k][n]
/// via strides (the transpose). Built once per (m,n,k). UIDs: A=1 B=2 C=3(virt)
/// actScale=4 S1=5(virt) wScale=6 D=7.
pub const MatmulDequantPlan = struct {
    plan: BackendDescriptor = null,
    keep: [24]BackendDescriptor = @splat(null),
    n_keep: usize = 0,
    workspace_bytes: usize = 0,

    const uid_a: i64 = 1;
    const uid_b: i64 = 2;
    const uid_c: i64 = 3;
    const uid_as: i64 = 4;
    const uid_s1: i64 = 5;
    const uid_ws: i64 = 6;
    const uid_d: i64 = 7;

    fn track(self: *MatmulDequantPlan, d: BackendDescriptor) void {
        if (self.n_keep < self.keep.len) {
            self.keep[self.n_keep] = d;
            self.n_keep += 1;
        }
    }

    fn tensor3(self: *MatmulDequantPlan, api: *const Api, uid: i64, dtype: c_int, dims: *const [3]i64, strides: *const [3]i64, is_virtual: bool) Error!BackendDescriptor {
        var t: BackendDescriptor = null;
        try api.check(api.cudnnBackendCreateDescriptor(b.DESC_TENSOR, &t), "CreateDescriptor(tensor)");
        self.track(t);
        var dt = dtype;
        var id = uid;
        var al: i64 = 16;
        var virt: u8 = if (is_virtual) 1 else 0;
        try api.check(api.cudnnBackendSetAttribute(t, b.ATTR_TENSOR_DATA_TYPE, b.TYPE_DATA_TYPE, 1, &dt), "tensor dtype");
        try api.check(api.cudnnBackendSetAttribute(t, b.ATTR_TENSOR_DIMENSIONS, b.TYPE_INT64, 3, dims), "tensor dims");
        try api.check(api.cudnnBackendSetAttribute(t, b.ATTR_TENSOR_STRIDES, b.TYPE_INT64, 3, strides), "tensor strides");
        try api.check(api.cudnnBackendSetAttribute(t, b.ATTR_TENSOR_UNIQUE_ID, b.TYPE_INT64, 1, &id), "tensor uid");
        try api.check(api.cudnnBackendSetAttribute(t, b.ATTR_TENSOR_BYTE_ALIGNMENT, b.TYPE_INT64, 1, &al), "tensor align");
        try api.check(api.cudnnBackendSetAttribute(t, b.ATTR_TENSOR_IS_VIRTUAL, b.TYPE_BOOLEAN, 1, &virt), "tensor virtual");
        try api.check(api.cudnnBackendFinalize(t), "Finalize(tensor)");
        return t;
    }

    fn pointwiseOp(self: *MatmulDequantPlan, api: *const Api, x: BackendDescriptor, bt: BackendDescriptor, y: BackendDescriptor) Error!BackendDescriptor {
        var pw: BackendDescriptor = null;
        try api.check(api.cudnnBackendCreateDescriptor(b.DESC_POINTWISE, &pw), "CreateDescriptor(pw)");
        self.track(pw);
        var mode = b.POINTWISE_MUL;
        var prec = b.DATA_FLOAT;
        try api.check(api.cudnnBackendSetAttribute(pw, b.ATTR_POINTWISE_MODE, b.TYPE_POINTWISE_MODE, 1, &mode), "pw mode");
        try api.check(api.cudnnBackendSetAttribute(pw, b.ATTR_POINTWISE_MATH_PREC, b.TYPE_DATA_TYPE, 1, &prec), "pw prec");
        try api.check(api.cudnnBackendFinalize(pw), "Finalize(pw)");
        var op: BackendDescriptor = null;
        try api.check(api.cudnnBackendCreateDescriptor(b.DESC_OPERATION_POINTWISE, &op), "CreateDescriptor(op pw)");
        self.track(op);
        var pw_ = pw;
        var x_ = x;
        var b_ = bt;
        var y_ = y;
        try api.check(api.cudnnBackendSetAttribute(op, b.ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, b.TYPE_BACKEND_DESCRIPTOR, 1, @ptrCast(&pw_)), "op pw desc");
        try api.check(api.cudnnBackendSetAttribute(op, b.ATTR_OPERATION_POINTWISE_XDESC, b.TYPE_BACKEND_DESCRIPTOR, 1, @ptrCast(&x_)), "op pw x");
        try api.check(api.cudnnBackendSetAttribute(op, b.ATTR_OPERATION_POINTWISE_BDESC, b.TYPE_BACKEND_DESCRIPTOR, 1, @ptrCast(&b_)), "op pw b");
        try api.check(api.cudnnBackendSetAttribute(op, b.ATTR_OPERATION_POINTWISE_YDESC, b.TYPE_BACKEND_DESCRIPTOR, 1, @ptrCast(&y_)), "op pw y");
        try api.check(api.cudnnBackendFinalize(op), "Finalize(op pw)");
        return op;
    }

    pub fn build(api: *const Api, handle: Handle, m: usize, n: usize, k: usize, d_f16: bool) Error!MatmulDequantPlan {
        var self: MatmulDequantPlan = .{};
        errdefer self.deinit(api);
        const M: i64 = @intCast(m);
        const N: i64 = @intCast(n);
        const K: i64 = @intCast(k);
        const d_type: c_int = if (d_f16) b.DATA_HALF else b.DATA_FLOAT;
        // A [1,m,k] row-major; B [1,k,n] = weight[n][k] viewed transposed
        // (k-stride 1, n-stride k); C/S1/D [1,m,n] row-major; scales broadcast.
        const a = try self.tensor3(api, uid_a, b.DATA_INT8, &.{ 1, M, K }, &.{ M * K, K, 1 }, false);
        const bt = try self.tensor3(api, uid_b, b.DATA_INT8, &.{ 1, K, N }, &.{ K * N, 1, K }, false);
        const c = try self.tensor3(api, uid_c, b.DATA_INT32, &.{ 1, M, N }, &.{ M * N, N, 1 }, true);
        const as_ = try self.tensor3(api, uid_as, b.DATA_FLOAT, &.{ 1, M, 1 }, &.{ M, 1, 1 }, false);
        const s1 = try self.tensor3(api, uid_s1, b.DATA_FLOAT, &.{ 1, M, N }, &.{ M * N, N, 1 }, true);
        const ws = try self.tensor3(api, uid_ws, b.DATA_FLOAT, &.{ 1, 1, N }, &.{ N, N, 1 }, false);
        const d = try self.tensor3(api, uid_d, d_type, &.{ 1, M, N }, &.{ M * N, N, 1 }, false);

        var mm_desc: BackendDescriptor = null;
        try api.check(api.cudnnBackendCreateDescriptor(b.DESC_MATMUL, &mm_desc), "CreateDescriptor(matmul)");
        self.track(mm_desc);
        var comp = b.DATA_INT32;
        try api.check(api.cudnnBackendSetAttribute(mm_desc, b.ATTR_MATMUL_COMP_TYPE, b.TYPE_DATA_TYPE, 1, &comp), "matmul comptype");
        try api.check(api.cudnnBackendFinalize(mm_desc), "Finalize(matmul desc)");

        var op_mm: BackendDescriptor = null;
        try api.check(api.cudnnBackendCreateDescriptor(b.DESC_OPERATION_MATMUL, &op_mm), "CreateDescriptor(op matmul)");
        self.track(op_mm);
        var a_ = a;
        var b_ = bt;
        var c_ = c;
        var md_ = mm_desc;
        try api.check(api.cudnnBackendSetAttribute(op_mm, b.ATTR_OPERATION_MATMUL_ADESC, b.TYPE_BACKEND_DESCRIPTOR, 1, @ptrCast(&a_)), "mm a");
        try api.check(api.cudnnBackendSetAttribute(op_mm, b.ATTR_OPERATION_MATMUL_BDESC, b.TYPE_BACKEND_DESCRIPTOR, 1, @ptrCast(&b_)), "mm b");
        try api.check(api.cudnnBackendSetAttribute(op_mm, b.ATTR_OPERATION_MATMUL_CDESC, b.TYPE_BACKEND_DESCRIPTOR, 1, @ptrCast(&c_)), "mm c");
        try api.check(api.cudnnBackendSetAttribute(op_mm, b.ATTR_OPERATION_MATMUL_DESC, b.TYPE_BACKEND_DESCRIPTOR, 1, @ptrCast(&md_)), "mm desc");
        try api.check(api.cudnnBackendFinalize(op_mm), "Finalize(op matmul)");

        const op_pw1 = try self.pointwiseOp(api, c, as_, s1); // C · act_scale → S1
        const op_pw2 = try self.pointwiseOp(api, s1, ws, d); // S1 · weight_scale → D

        var graph: BackendDescriptor = null;
        try api.check(api.cudnnBackendCreateDescriptor(b.DESC_OPERATIONGRAPH, &graph), "CreateDescriptor(graph)");
        self.track(graph);
        var h = handle;
        var ops = [_]BackendDescriptor{ op_mm, op_pw1, op_pw2 };
        try api.check(api.cudnnBackendSetAttribute(graph, b.ATTR_OPERATIONGRAPH_HANDLE, b.TYPE_HANDLE, 1, @ptrCast(&h)), "graph handle");
        try api.check(api.cudnnBackendSetAttribute(graph, b.ATTR_OPERATIONGRAPH_OPS, b.TYPE_BACKEND_DESCRIPTOR, 3, @ptrCast(&ops)), "graph ops");
        try api.check(api.cudnnBackendFinalize(graph), "Finalize(graph)");

        const r = planFromGraph(api, handle, graph) catch {
            std.debug.print("cuDNN fused int8 matmul+dequant: no engine config for m={d} n={d} k={d}\n", .{ m, n, k });
            return error.CudnnError;
        };
        self.track(r.cfg);
        self.track(r.plan);
        self.plan = r.plan;
        self.workspace_bytes = r.ws;
        return self;
    }

    pub fn execute(self: *const MatmulDequantPlan, api: *const Api, handle: Handle, a: u64, wt: u64, act_scale: u64, weight_scale: u64, d: u64, workspace: u64) Error!void {
        var vpack: BackendDescriptor = null;
        try api.check(api.cudnnBackendCreateDescriptor(b.DESC_VARIANT_PACK, &vpack), "CreateDescriptor(vpack)");
        defer _ = api.cudnnBackendDestroyDescriptor(vpack);
        var uids = [_]i64{ uid_a, uid_b, uid_as, uid_ws, uid_d };
        var ptrs = [_]?*anyopaque{ @ptrFromInt(a), @ptrFromInt(wt), @ptrFromInt(act_scale), @ptrFromInt(weight_scale), @ptrFromInt(d) };
        var ws_ptr: ?*anyopaque = if (workspace != 0) @ptrFromInt(workspace) else null;
        try api.check(api.cudnnBackendSetAttribute(vpack, b.ATTR_VARIANT_PACK_UNIQUE_IDS, b.TYPE_INT64, uids.len, @ptrCast(&uids)), "vpack uids");
        try api.check(api.cudnnBackendSetAttribute(vpack, b.ATTR_VARIANT_PACK_DATA_POINTERS, b.TYPE_VOID_PTR, ptrs.len, @ptrCast(&ptrs)), "vpack ptrs");
        try api.check(api.cudnnBackendSetAttribute(vpack, b.ATTR_VARIANT_PACK_WORKSPACE, b.TYPE_VOID_PTR, 1, @ptrCast(&ws_ptr)), "vpack workspace");
        try api.check(api.cudnnBackendFinalize(vpack), "Finalize(vpack)");
        try api.check(api.cudnnBackendExecute(handle, self.plan, vpack), "Execute");
    }

    pub fn deinit(self: *MatmulDequantPlan, api: *const Api) void {
        var i: usize = self.n_keep;
        while (i > 0) {
            i -= 1;
            _ = api.cudnnBackendDestroyDescriptor(self.keep[i]);
        }
        self.* = .{};
    }
};

test "cudnn loads (skips without the library)" {
    var api = Api.load() catch return error.SkipZigTest;
    defer api.deinit();
    try std.testing.expect(api.cudnnGetVersion() > 0);
}
