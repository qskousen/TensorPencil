//! CLIP text tower on the CUDA backend, SD1.5's CLIP-L and SDXL's CLIP-G: the adapter
//! for `clip_text_dev.zig`. The cheap head (the pooled row and its projection) stays on
//! the host, and the weights are borrowed from the mmap-stable CPU model and left
//! resident across calls.
//!
//! Every linear goes through `lin_cuda` with `Plan.dense = .f16`, so a dense layer
//! computes in f16 regardless of `--encoder-f16`, the regime `sd_unet_cuda` runs in.
//! Parity against the CPU forward is therefore relative (~1e-3), not the ~1e-6 the
//! Vulkan f32 path holds; `Session.encode`'s fallback order matters for that reason.
//!
//! Validated by `TensorPencil sd-cuda-test`, not by a unit test, because the test binary
//! brings up no CUDA context.

const std = @import("std");
const tp_core = @import("tp_core");
const cuda = @import("tp_gpu").cuda;
const ops = @import("tp_ops");
const clip_tok = tp_core.clip_tokenizer;
const clip_text = @import("clip_text.zig");
const clip_text_dev = @import("clip_text_dev.zig");

const Backend = cuda.Backend;
const Buf = cuda.backend.DeviceBuffer;
const Dt = cuda.backend.Dt;
const Weight = ops.matmul.Weight;
const lin_cuda = @import("lin_cuda.zig");
const clen = clip_tok.context_length;

/// The tower's plan: the encoders' block-quant and integer routes, dense weights on
/// the f16 tensor cores, and its LoRA stack, applied inside `lin_cuda.run`.
pub fn planFor(be: *const Backend, enc: *const clip_text.TextEncoder) !lin_cuda.Plan {
    var p = lin_cuda.withInt8(try lin_cuda.plan(enc.device_lins, .{ .blockq = lin_cuda.encoder_gemm }, "clip_text cuda"), be, lin_cuda.dual_lin.encoderInt8());
    p.dense = .f16;
    p.lora = enc.lora;
    return p;
}

/// `clip_text_dev.encodePrompt` on this backend.
pub fn encodePrompt(
    enc: *const clip_text.TextEncoder,
    be: *Backend,
    io: std.Io,
    gpa: std.mem.Allocator,
    out: []f32,
    p: *const clip_tok.Prompt,
    r: clip_text.TextEncoder.PromptRun,
    /// The encoder dial's width for the residual, the norm outputs and the projections.
    act: Dt,
) !void {
    // Taken for signature symmetry with the CPU form; nothing here reaches the host GEMM.
    _ = io;
    var d: Dev = .{ .be = be, .enc = enc, .act = act };
    try clip_text_dev.encodePrompt(&d, enc, gpa, out, p, r);
}

const Bufs = struct { x: Buf = .{}, normed: Buf = .{}, q: Buf = .{}, k: Buf = .{}, v: Buf = .{}, big: Buf = .{}, t: Buf = .{}, cap: Buf = .{} };

/// The `clip_text_dev` adapter.
const Dev = struct {
    be: *Backend,
    enc: *const clip_text.TextEncoder,
    /// `x`, `normed` and `t`; the attention planes and `big` stay f32, and `big` is the
    /// staging for a 16-bit `x`'s upload and download.
    act: Dt = .f32,
    plan: lin_cuda.Plan = undefined,
    total: usize = 0,
    b: Bufs = .{},

    pub fn setup(d: *Dev, total: usize, capture: bool) !void {
        const cfg = d.enc.cfg;
        d.total = total;
        d.plan = planFor(d.be, d.enc) catch return error.UnsupportedDType;
        try lin_cuda.presize(d.be, d.plan, d.enc.device_lins);
        try lin_cuda.presizeRows(d.be, d.plan, d.enc.device_lins, total);
        try lin_cuda.attachLora(d.be, &d.plan, d.enc.lora, d.enc.device_lins, total);
        // The int8 routes store whole 128-row tiles.
        const rows_pad = std.mem.alignForward(usize, total, 128);
        errdefer d.teardown();
        d.b.x = try d.be.tensorCreate(total * cfg.hidden * d.act.bytes());
        d.b.x.dt = d.act;
        inline for (.{ "normed", "q", "k", "v", "t" }) |f| {
            const w: Dt = if (comptime std.mem.eql(u8, f, "normed") or std.mem.eql(u8, f, "t")) d.act else .f32;
            @field(d.b, f) = try d.be.tensorCreate(rows_pad * cfg.hidden * w.bytes());
            @field(d.b, f).dt = w;
        }
        d.b.big = try d.be.tensorCreate(rows_pad * cfg.intermediate * 4);
        if (capture) d.b.cap = try d.be.tensorCreate(total * cfg.hidden * 4);
    }

    /// No weight scope: the f16-converted GEMM weights stay RESIDENT across calls (they
    /// borrow the mmap-stable CPU model), where a scope would re-upload the whole tower
    /// every encode.
    pub fn teardown(d: *Dev) void {
        d.be.freeAttnScratch();
        d.be.freeConvScratch();
        inline for (@typeInfo(Bufs).@"struct".field_names) |name| d.be.tensorDestroy(&@field(d.b, name));
    }

    pub fn upload(d: *Dev, b: Buf, data: []const f32) !void {
        if (b.dt == .f32) return d.be.tensorUpload(b, std.mem.sliceAsBytes(data));
        try d.be.tensorUpload(d.b.big.slice(0, data.len * 4), std.mem.sliceAsBytes(data));
        try d.be.opCast(d.b.big, b, data.len, 0);
    }

    pub fn download(d: *Dev, b: Buf, dst: []f32) !void {
        const src = if (b.dt == .f32) b else blk: {
            try d.be.opCast(b, d.b.big, dst.len, 0);
            break :blk d.b.big;
        };
        try d.be.tensorDownload(src.slice(0, dst.len * 4), std.mem.sliceAsBytes(dst));
    }

    pub fn beginBatch(d: *Dev) !void {
        try d.be.beginBatch();
    }

    pub fn endBatch(d: *Dev) !void {
        try d.be.endBatch();
    }

    pub fn abortBatch(d: *Dev) void {
        if (d.be.batching()) d.be.abortBatch();
    }

    pub fn layerNorm(d: *Dev, x: Buf, y: Buf, w: []const f32, bias: []const f32, rows: usize) !void {
        try d.be.opLayerNorm(x, y, w, bias, rows, d.enc.cfg.hidden, d.enc.cfg.eps);
    }

    /// Under one activation prep. `ys` must hold `align(m, 128)` rows.
    pub fn gemms(d: *Dev, x: Buf, m: usize, ys: []const Buf, wts: []const Weight, biases: []const []const f32) !void {
        const g = try lin_cuda.prep(d.be, d.plan, x, m, wts[0].cols, wts);
        for (ys, wts, biases) |y, wt, bias| try lin_cuda.gemm(d.be, d.plan, g, y, wt, bias);
    }

    /// One causal launch per chunk over `dbOffset` views: the batched CUDA attention is
    /// the non-causal one, and a prompt has one to three chunks.
    pub fn attention(d: *Dev, items: usize) !void {
        const cfg = d.enc.cfg;
        const h = cfg.hidden;
        const hd = cfg.headDim();
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        for (0..items) |c| {
            const off = c * clen * h * @sizeOf(f32);
            try d.be.attn(
                cuda.backend.dbOffset(d.b.q, off),
                cuda.backend.dbOffset(d.b.k, off),
                cuda.backend.dbOffset(d.b.v, off),
                cuda.backend.dbOffset(d.b.big, off),
                clen,
                clen,
                cfg.heads,
                cfg.heads,
                hd,
                scale,
                true,
            );
        }
    }

    pub fn add(d: *Dev, x: Buf, y: Buf, n: usize) !void {
        try d.be.opAdd(x, y, n);
    }

    pub fn gelu(d: *Dev, x: Buf, n: usize, act: clip_text.Config.Act) !void {
        switch (act) {
            .quick_gelu => try d.be.geluQuick(x, n),
            .gelu_erf => try d.be.geluErf(x, n),
        }
    }

    pub fn copy(d: *Dev, dst: Buf, src: Buf, n: usize) !void {
        if (src.dt == dst.dt) return d.be.opCopyOff(dst, 0, src, 0, n);
        try d.be.opCast(src, dst, n, 0);
    }
};
