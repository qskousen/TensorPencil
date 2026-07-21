//! Cooperative cancellation for long CPU kernels.
//!
//! A threadlocal token, set by CPU entry points that own a cancel flag (DiT
//! forward, VAE decode, text encode) for the duration of their compute, and
//! polled inside the threaded matmul/attention kernels between row panels and
//! k-blocks — so a cancel lands in milliseconds even when a single GEMM takes
//! seconds. Threadlocal (not a plain global) so a diffusion worker's cancel
//! can't leak into an LLM forward running concurrently on another thread: the
//! kernels capture the token ONCE on the calling thread and pass it into
//! their `std.Io.Group` tasks as a plain value.
//!
//! Worker tasks return `void`, so on cancel they just bail early (leaving the
//! output partial); the coordinating kernel re-checks the token afterwards
//! and returns `error.Canceled` (already in the kernels' error sets via
//! `std.Io.Cancelable`).

const std = @import("std");

pub const Token = ?*const std.atomic.Value(bool);

/// Set around CPU compute by entry points that own a cancel flag; read by
/// `matmul()`/`attention()` at entry on the calling thread.
pub threadlocal var token: Token = null;

pub inline fn canceled(tok: Token) bool {
    return if (tok) |t| t.load(.acquire) else false;
}
