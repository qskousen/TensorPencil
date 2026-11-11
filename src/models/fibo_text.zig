//! Fibo 1.5 prompt conditioning (diffusers' `BriaFiboPipeline.get_prompt_embeds`): the
//! raw prompt through SmolLM3's tokenizer with nothing added, an empty one being the
//! single token 128000, then every hidden state SmolLM3 returns.
//!
//! The DiT reads two things off those states (`Context`): the context
//! `[hs[-1] | hs[-2]]`, and one state per block, the last `blocks` of them, or all of
//! them with the last repeated when there are fewer states than blocks (37 states, 46
//! blocks: blocks 37..45 all read the final, normed one).

const std = @import("std");
const tp_core = @import("tp_core");
const qwen3 = @import("qwen3.zig");

const Tokenizer = tp_core.tokenizer.Tokenizer;

/// What the pipeline substitutes for an empty prompt.
pub const empty_id: u32 = 128000;

pub fn buildIds(tok: *const Tokenizer, gpa: std.mem.Allocator, text: []const u8, out: *std.ArrayList(u32)) !void {
    if (text.len == 0) return out.append(gpa, empty_id);
    try tok.encode(gpa, text, out);
}

/// Which hidden state block `b` of `blocks` reads, of `n_states`.
pub fn stateFor(b: usize, blocks: usize, n_states: usize) usize {
    if (n_states >= blocks) return n_states - blocks + b;
    return @min(b, n_states - 1);
}

/// The DiT's inputs from the encoder's states, `taps` token-major
/// `[text_len][n_states][hidden]`: `ehs` `[text_len][2 * hidden]` and `layers`
/// `[blocks][text_len][hidden]`. Caller frees both.
pub const Context = struct {
    ehs: []f32,
    layers: []f32,

    pub fn init(gpa: std.mem.Allocator, taps: []const f32, text_len: usize, n_states: usize, hidden: usize, blocks: usize) !Context {
        std.debug.assert(taps.len == text_len * n_states * hidden and n_states >= 2);
        const ehs = try gpa.alloc(f32, text_len * 2 * hidden);
        errdefer gpa.free(ehs);
        const layers = try gpa.alloc(f32, blocks * text_len * hidden);
        for (0..text_len) |t| {
            const row = taps[t * n_states * hidden ..][0 .. n_states * hidden];
            @memcpy(ehs[t * 2 * hidden ..][0..hidden], row[(n_states - 1) * hidden ..][0..hidden]);
            @memcpy(ehs[t * 2 * hidden + hidden ..][0..hidden], row[(n_states - 2) * hidden ..][0..hidden]);
            for (0..blocks) |b| {
                const s = stateFor(b, blocks, n_states);
                @memcpy(layers[(b * text_len + t) * hidden ..][0..hidden], row[s * hidden ..][0..hidden]);
            }
        }
        return .{ .ehs = ehs, .layers = layers };
    }

    pub fn deinit(self: *Context, gpa: std.mem.Allocator) void {
        gpa.free(self.ehs);
        gpa.free(self.layers);
        self.* = undefined;
    }
};

// --- tests ------------------------------------------------------------------

const testing = std.testing;

test "Fibo 1.5 blocks past the last state read the last" {
    try testing.expectEqual(@as(usize, 0), stateFor(0, 46, 37));
    try testing.expectEqual(@as(usize, 36), stateFor(36, 46, 37));
    try testing.expectEqual(@as(usize, 36), stateFor(45, 46, 37));
    try testing.expectEqual(@as(usize, 13), stateFor(3, 30, 40));
}

const test_gate = @import("../test_gate.zig");
const build_options = @import("build_options");
const te_ref_path = "src/models/assets/fibo_te_ref.safetensors";
const te_ckpt = build_options.model_root ++ "/text_encoders/smollm3_3b_fibo15_bf16.safetensors";

test "the Fibo 1.5 conditioning matches diffusers' on the real SmolLM3" {
    const gpa = testing.allocator;
    const io = testing.io;
    const safetensors = tp_core.safetensors;
    try test_gate.requireIntegration();
    try test_gate.requireModelFile(io, te_ckpt);
    var ref = try safetensors.SafeTensors.open(gpa, io, te_ref_path);
    defer ref.deinit();
    var ck = try safetensors.SafeTensors.open(gpa, io, te_ckpt);
    defer ck.deinit();
    var enc = try qwen3.TextEncoder.loadVariant(gpa, .{ .safetensors = &ck }, .fibo);
    defer enc.deinit();
    var tok = try Tokenizer.initLlama3(gpa);
    defer tok.deinit();
    const ops = @import("tp_ops");
    const prev = ops.matmul.exact_activations;
    ops.matmul.exact_activations = true;
    defer ops.matmul.exact_activations = prev;

    const prompts = [_][]const u8{ "{\"short_description\": \"a lighthouse at dusk\", \"lighting\": {\"conditions\": \"golden hour\"}}", "" };
    const hidden = enc.cfg.hidden;
    const n_states = enc.tapCount();
    for (prompts, 0..) |p, i| {
        var kb: [32]u8 = undefined;
        const tv = try ref.require(try std.mem.print(&kb, "tokens.{d}", .{i}));
        const want_ids = try gpa.alloc(u32, tv.bytes.len / 4);
        defer gpa.free(want_ids);
        for (want_ids, 0..) |*id, j| id.* = @intCast(std.mem.readInt(i32, tv.bytes[j * 4 ..][0..4], .little));
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(gpa);
        try buildIds(&tok, gpa, p, &ids);
        try testing.expectEqualSlices(u32, want_ids, ids.items);

        const taps = try enc.encode(io, gpa, ids.items, null);
        defer gpa.free(taps);
        const n = ids.items.len;
        const check = struct {
            fn f(want: []const f32, cols: usize, got_rows: []const f32, stride: usize, off: usize, rows: usize) f64 {
                var num: f64 = 0;
                var den: f64 = 0;
                for (0..rows) |r| for (0..cols) |c| {
                    const w = want[r * cols + c];
                    const g = got_rows[r * stride + off + c];
                    num += @as(f64, w - g) * (w - g);
                    den += @as(f64, w) * w;
                };
                return @sqrt(num / @max(den, 1e-30));
            }
        }.f;
        for ([_]usize{ 0, 1, 3, 4, 17, 35, 36 }) |l| {
            const want = try (try ref.require(try std.mem.print(&kb, "hidden.{d}.{d}", .{ i, l }))).toF32Alloc(gpa);
            defer gpa.free(want);
            const cols = want.len / n;
            const rel = check(want, cols, taps, n_states * hidden, l * hidden, n);
            errdefer std.debug.print("prompt {d} state {d}: rel L2 {e:.3}\n", .{ i, l, rel });
            try testing.expect(rel < 1e-4);
        }
    }
}
