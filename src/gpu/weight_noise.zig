//! Live weight-noise state both GPU backends' block-quant kernels read per launch.

const std = @import("std");
const noise_curve = @import("tp_core").noise_curve;

/// Weight noise: the two values every block-quant GEMM/GEMV kernel reads out of
/// the parameter slots the ABI already carried unused. See `cuda/wnoise.zig` for
/// what the kernels do with them; `dtypeHonorsWeightNoise` is which ones do.
///
/// Sigma is a FUNCTION OF DEPTH, not a constant: `core/noise_curve.zig` evaluates
/// the user's expression once per layer into `table`, and a launch reads the slot
/// for the layer it belongs to. That is what lets a curve perturb the early stack
/// hard while leaving the last layers and the LM head almost untouched, which is
/// the difference between a differently-reasoned answer and gibberish.
///
/// `table` is written whole by whoever owns the curve (a UI thread) and read one
/// slot at a time by the worker at every launch, hence atomic per slot. A publish
/// that lands mid-forward shows a mix of two curves for one token and is coherent
/// again on the next, which for a noise knob is not worth a lock. `depth_i` is
/// worker-only: the model sets it as it walks the stack.
pub const WeightNoise = struct {
    /// Longest stack this can describe. Slot `l` is decoder layer `l`; the slot at
    /// the live `depth` is the LM head.
    pub const max_depth = 256;
    /// Longest curve expression kept. Generous for `0.08*(1-t)^2`.
    pub const max_expr = 192;

    table: [max_depth]std.atomic.Value(u32) = @splat(.init(0)),
    /// Any slot nonzero, so the per-launch key lookup can be skipped outright.
    any: std.atomic.Value(bool) = .init(false),
    /// Decoder layer count, from the model at init. 0 until then.
    depth: u32 = 0,
    /// Which slot the launches now belong to.
    depth_i: u32 = 0,
    /// Advanced once per forward so each one draws a fresh perturbation.
    seq: u32 = 0,
    /// Base of the whole run's noise stream. Same seed + same curve + same prompt
    /// reproduces a generation exactly, which is what makes a sweep readable:
    /// without it every setting is one sample of an unrepeatable draw.
    seed: u32 = 0,

    /// The curve itself. Unlocked, because nothing shares it: `setDepth` is called
    /// exactly once from the model's init, and `setCurve` only from whichever
    /// thread owns the curve (the CLI at bring-up, a GUI's UI thread once loading
    /// has finished). What DOES cross threads is `table`, and that is atomic per
    /// slot. Calling `setDepth` from a worker while a UI edits the curve would
    /// break that reasoning, so don't: it is an init-time call.
    expr_buf: [max_expr]u8 = @splat(0),
    expr_len: usize = 0,
    /// The curve's `a`: how MUCH, kept apart from the curve's WHERE so the amount
    /// can be a knob nobody has to edit an expression to turn. 1 leaves a curve
    /// written with literal amplitudes meaning exactly what it says.
    amount: f32 = 1,

    /// Set the curve (any `noise_curve` expression; a bare number is a flat one).
    /// An empty or malformed curve is off. Re-evaluates the table if the layer
    /// count is already known.
    pub fn setCurve(self: *WeightNoise, expr: []const u8) void {
        self.expr_len = @min(expr.len, max_expr);
        @memcpy(self.expr_buf[0..self.expr_len], expr[0..self.expr_len]);
        self.refill();
    }

    /// Set the amount the curve's `a` reads. Non-finite and negative clamp to 0
    /// (off): this is fed straight from a text box.
    pub fn setAmount(self: *WeightNoise, v: f32) void {
        self.amount = if (std.math.isFinite(v) and v > 0) v else 0;
        self.refill();
    }

    /// Tell it how deep the stack is. Init-time only; see `expr_buf`.
    pub fn setDepth(self: *WeightNoise, n_layers: usize) void {
        self.depth = @intCast(@min(n_layers, max_depth - 1));
        self.refill();
    }

    fn refill(self: *WeightNoise) void {
        const expr = self.expr_buf[0..self.expr_len];
        var any = false;
        for (&self.table, 0..) |*slot, i| {
            var v: f32 = 0;
            if (self.depth != 0 and i <= self.depth and expr.len != 0) {
                // t spans 0 at the first layer to exactly 1 at the head slot, so a
                // curve decaying in t protects the head without being asked to.
                const l: f32 = @floatFromInt(i);
                const n: f32 = @floatFromInt(self.depth);
                v = noise_curve.sanitize(noise_curve.eval(expr, .{ .t = l / n, .l = l, .n = n, .a = self.amount }) catch 0);
            }
            slot.store(@bitCast(v), .monotonic);
            if (v != 0) any = true;
        }
        self.any.store(any, .release);
    }

    /// Sigma for the layer the launches now belong to.
    pub fn sigma(self: *const WeightNoise) f32 {
        return @bitCast(self.table[@min(self.depth_i, max_depth - 1)].load(.monotonic));
    }

    /// Whether the curve is nonzero ANYWHERE. A per-launch gate would have to be
    /// per-layer, but this only guards the key lookup, which is free to happen on
    /// a layer whose own sigma is 0.
    pub fn on(self: *const WeightNoise) bool {
        return self.any.load(.acquire);
    }

    /// Point the following launches at a decoder layer, or at the LM head.
    pub fn atLayer(self: *WeightNoise, l: usize) void {
        self.depth_i = @intCast(@min(l, max_depth - 1));
    }

    pub fn atHead(self: *WeightNoise) void {
        self.depth_i = self.depth;
    }

    /// Draw the next perturbation. One call per forward.
    pub fn tick(self: *WeightNoise) void {
        self.seq +%= 1;
    }

    /// The per-launch key: everything that identifies "this weight, this forward"
    /// folded host-side, so the kernel spends one parameter slot instead of three
    /// and only has to mix in the block's own offset.
    pub fn key(self: *const WeightNoise, weight_id: u32) u32 {
        var h = self.seed ^ (self.seq *% 0x9E3779B1) ^ (weight_id *% 0x85EBCA6B);
        h ^= h >> 16;
        h *%= 0x7FEB352D;
        h ^= h >> 15;
        return h;
    }
};
