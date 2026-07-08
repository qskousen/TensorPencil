//! CUDA compute backend (pure Zig: runtime-loaded driver + hand-emitted PTX).
//!
//! The Vulkan path (`gpu/context.zig`) hits three NVIDIA-only structural
//! ceilings — a 48 KB shared-memory cap, no `cp.async`, and opaque coopmat
//! lowering. This backend goes through the CUDA Driver API and JITs
//! hand-written PTX, unlocking >48 KB dynamic shared (`cuFuncSetAttribute`),
//! `cp.async` global->shared overlap, and explicit `mma.sync` IMMA tiling.
//!
//! Phase 1 goal (see PLAN.md M10): a hand-PTX int8 GEMM that beats the Vulkan
//! ~85 TOPS baseline, validated bit-for-bit against the same CPU oracle as
//! `gpu-i8-test`. Kept behind its own CLI (`cuda-test` / `cuda-i8-test`); the
//! Vulkan path stays the portable default.

const std = @import("std");

pub const cu = @import("cuda/cu.zig");
pub const ptx = @import("cuda/ptx.zig");
pub const kernels = @import("cuda/kernels.zig");
pub const elt = @import("cuda/elt.zig");
pub const context = @import("cuda/context.zig");
pub const Context = context.Context;
pub const Buffer = context.Buffer;
pub const backend = @import("cuda/backend.zig");
pub const Backend = backend.Backend;

/// cuBLASLt / cuDNN bindings for the library-backed `--backend cuda` (Phase 2):
/// dlopen'd closed math libraries, same loading mechanism as the driver.
pub const cublaslt = @import("cuda/cublaslt.zig");
pub const cudnn = @import("cuda/cudnn.zig");

test {
    _ = cu;
    _ = ptx;
    _ = kernels;
    _ = context;
    _ = backend;
    _ = cublaslt;
    _ = cudnn;
}
