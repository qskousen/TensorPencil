//! The adapter forms a LoRA file carries and their exact arithmetic, for every family:
//! `parse` reads one stem as ComfyUI's `load_lora` does, `apply` is ComfyUI's merge of it
//! into one tensor (the oracle a sidecar is checked against), and `termFor` turns it
//! into a `lora.Target` term.
//!
//! Three things here are silent wrong answers, each a finite plausible render:
//!
//! 1. **Each adapter's scale.** alpha over the DOWN factor's rows (LoRA), over
//!    `hada_w1_b`'s rows (LoHa), over the decomposed factor's rows and 1.0 when
//!    LoKr has none.
//! 2. **A 3x3 conv's column order.** A LoCon `A` stays in torch's `[r][ci][ky][kx]`
//!    and runs as a checkpoint-order conv (`ops.conv.Weights.torch`), which every
//!    backend reorders to its patch order itself; reading it as patch order multiplies
//!    the wrong taps.
//! 3. **The merge's rounding**, for the oracle: on a GPU that prefers fp16 (a 3090)
//!    ComfyUI casts the weight to fp16, adds each file's delta rounded to fp16, in
//!    file order, and keeps fp16. DoRA's output-axis norm reads the BASE weight, its
//!    input-axis norm the patched one.
//!
//! `tools/gen_lora_merge_fixtures.py` pins the merge by executing ComfyUI, and the
//! sidecar is checked against the merge.

const std = @import("std");
const tp_core = @import("tp_core");
const ops = @import("tp_ops");
const lora_mod = @import("lora.zig");

const weights_mod = tp_core.weights;
const WeightStore = weights_mod.WeightStore;
const TensorView = weights_mod.TensorView;
const Weight = ops.matmul.Weight;

// --- parsing one stem --------------------------------------------------------

/// One stem's patch, as the reference's `load_lora` reads it.
pub const Patch = struct {
    kind: Kind,
    /// The file's `alpha`, null when absent.
    alpha: ?f32,
    /// DoRA magnitude, which turns the add into `weight_decompose`.
    dora: ?TensorView,

    pub const Kind = union(enum) {
        lora: struct { up: TensorView, down: TensorView, mid: ?TensorView },
        loha: struct { w1a: TensorView, w1b: TensorView, w2a: TensorView, w2b: TensorView, t1: ?TensorView, t2: ?TensorView },
        lokr: struct { w1: ?TensorView, w2: ?TensorView, w1a: ?TensorView, w1b: ?TensorView, w2a: ?TensorView, w2b: ?TensorView },
        /// A whole-weight difference, `weight += strength * diff`.
        diff: TensorView,
    };
};

/// Key names a file's walk consumed, for the "not loaded" report.
pub const Consumed = std.StringHashMapUnmanaged(void);

/// Every key suffix `parse` reads besides the plain dialects', so a loader can tell
/// an adapter's keys from a spelling it does not know.
pub const adapter_suffixes = [_][]const u8{
    ".alpha",     ".dora_scale", ".lora_mid.weight", ".hada_w1_a", ".hada_w1_b", ".hada_w2_a", ".hada_w2_b", ".hada_t1", ".hada_t2",
    ".lokr_w1",   ".lokr_w2",    ".lokr_w1_a",       ".lokr_w1_b", ".lokr_w2_a", ".lokr_w2_b", ".lokr_t2",  ".diff",    ".w_norm",
    ".set_weight",
};

/// The suffixes that alone name a stem `parse` must read, i.e. what is not a plain pair.
pub const stem_suffixes = [_][]const u8{ ".lokr_w1", ".lokr_w1_a", ".hada_w1_a", ".diff" };

fn probe(store: WeightStore, buf: []u8, stem: []const u8, suffix: []const u8, consumed: *Consumed, gpa: std.mem.Allocator) !?TensorView {
    const key = std.fmt.bufPrint(buf, "{s}{s}", .{ stem, suffix }) catch return null;
    const v = store.get(key) orelse return null;
    // The store's own name string, which outlives this walk.
    try consumed.put(gpa, v.info.name, {});
    return v;
}

/// A one-element tensor as f32, integer dtypes included: some files store alpha
/// as an int64.
fn scalar(v: TensorView) !f32 {
    if (v.info.elemCount() != 1) return error.LengthMismatch;
    return switch (v.info.dtype) {
        .i64 => @floatFromInt(std.mem.readInt(i64, v.bytes[0..8], .little)),
        .i32 => @floatFromInt(std.mem.readInt(i32, v.bytes[0..4], .little)),
        else => v.asScalarF32(),
    };
}

/// The patch the reference reads at `stem`, or null when the file has none.
/// Adapter kinds are tried in its order and a later match replaces an earlier one;
/// a `.diff` replaces any of them. `set_weight` and Tucker LoKr come back as
/// `error.UnsupportedAdapter` (the reference raises on the latter itself and
/// leaves the weight unpatched).
pub fn parse(gpa: std.mem.Allocator, store: WeightStore, stem: []const u8, consumed: *Consumed) !?Patch {
    var buf: [512]u8 = undefined;
    const alpha: ?f32 = if (try probe(store, &buf, stem, ".alpha", consumed, gpa)) |v| try scalar(v) else null;
    const dora = try probe(store, &buf, stem, ".dora_scale", consumed, gpa);
    var kind: ?Patch.Kind = null;

    for (lora_mod.dialects, 0..) |d, i| {
        const up = try probe(store, &buf, stem, d.b, consumed, gpa) orelse continue;
        const down = try probe(store, &buf, stem, d.a, consumed, gpa) orelse return error.ShapeMismatch;
        const mid = if (i == 0) try probe(store, &buf, stem, ".lora_mid.weight", consumed, gpa) else null;
        kind = .{ .lora = .{ .up = up, .down = down, .mid = mid } };
        break;
    }
    if (try probe(store, &buf, stem, ".hada_w1_a", consumed, gpa)) |w1a| {
        const t1 = try probe(store, &buf, stem, ".hada_t1", consumed, gpa);
        kind = .{ .loha = .{
            .w1a = w1a,
            .w1b = try probe(store, &buf, stem, ".hada_w1_b", consumed, gpa) orelse return error.ShapeMismatch,
            .w2a = try probe(store, &buf, stem, ".hada_w2_a", consumed, gpa) orelse return error.ShapeMismatch,
            .w2b = try probe(store, &buf, stem, ".hada_w2_b", consumed, gpa) orelse return error.ShapeMismatch,
            .t1 = t1,
            .t2 = if (t1 != null) try probe(store, &buf, stem, ".hada_t2", consumed, gpa) orelse return error.ShapeMismatch else null,
        } };
    }
    {
        const w1 = try probe(store, &buf, stem, ".lokr_w1", consumed, gpa);
        const w2 = try probe(store, &buf, stem, ".lokr_w2", consumed, gpa);
        const w1a = try probe(store, &buf, stem, ".lokr_w1_a", consumed, gpa);
        const w1b = try probe(store, &buf, stem, ".lokr_w1_b", consumed, gpa);
        const w2a = try probe(store, &buf, stem, ".lokr_w2_a", consumed, gpa);
        const w2b = try probe(store, &buf, stem, ".lokr_w2_b", consumed, gpa);
        if (w1 != null or w2 != null or w1a != null or w2a != null) {
            if (try probe(store, &buf, stem, ".lokr_t2", consumed, gpa) != null) return error.UnsupportedAdapter;
            if ((w1 == null and (w1a == null or w1b == null)) or (w2 == null and (w2a == null or w2b == null))) return error.ShapeMismatch;
            kind = .{ .lokr = .{ .w1 = w1, .w2 = w2, .w1a = w1a, .w1b = w1b, .w2a = w2a, .w2b = w2b } };
        }
    }
    if (try probe(store, &buf, stem, ".w_norm", consumed, gpa)) |w| kind = .{ .diff = w };
    if (try probe(store, &buf, stem, ".diff", consumed, gpa)) |w| kind = .{ .diff = w };
    if (try probe(store, &buf, stem, ".set_weight", consumed, gpa) != null) return error.UnsupportedAdapter;
    const k = kind orelse return null;
    return .{ .kind = k, .alpha = alpha, .dora = dora };
}

// --- the math ----------------------------------------------------------------

pub inline fn r16(x: f32) f32 {
    return @floatCast(@as(f16, @floatCast(x)));
}

/// How a weight is merged: the reference's fp16 temp, or f32 for an f32 base.
pub const Mode = enum { f16, f32 };

inline fn rnd(mode: Mode, x: f32) f32 {
    return if (mode == .f16) r16(x) else x;
}

fn rows(v: TensorView) usize {
    return v.info.shape.slice()[0];
}

fn inner(v: TensorView) usize {
    return v.info.elemCount() / rows(v);
}

/// `a[m][k] @ b[k][n]`, summed over `k` in order. Small: only factors go
/// through here, never a whole weight.
fn mm(gpa: std.mem.Allocator, a: []const f32, b: []const f32, m: usize, k: usize, n: usize) ![]f32 {
    const y = try gpa.alloc(f32, m * n);
    for (0..m) |i| {
        const row = y[i * n ..][0..n];
        @memset(row, 0);
        for (0..k) |j| axpy(row, a[i * k + j], b[j * n ..][0..n]);
    }
    return y;
}

/// `y += a * x`, fused, as the reference's GEMM accumulates: summing the rank in
/// order with one rounding per step is what makes an f32 merge bit-exact.
inline fn axpy(y: []f32, a: f32, x: []const f32) void {
    const V = 16;
    const av: @Vector(V, f32) = @splat(a);
    var i: usize = 0;
    while (i + V <= y.len) : (i += V) y[i..][0..V].* = @mulAdd(@Vector(V, f32), av, x[i..][0..V].*, y[i..][0..V].*);
    while (i < y.len) : (i += 1) y[i] = @mulAdd(f32, a, x[i], y[i]);
}

/// `einsum("i j k l, j r, i p -> p r k l", t, wb, wa)`: the CP core LoHa uses.
fn cp(gpa: std.mem.Allocator, t: []const f32, ts: []const usize, wb: []const f32, wb_cols: usize, wa: []const f32, wa_cols: usize) ![]f32 {
    const ri = ts[0];
    const rj = ts[1];
    const kk = ts[2] * ts[3];
    const out = try gpa.alloc(f32, wa_cols * wb_cols * kk);
    @memset(out, 0);
    for (0..ri) |i| for (0..rj) |j| for (0..kk) |s| {
        const tv = t[(i * rj + j) * kk + s];
        for (0..wa_cols) |p| {
            const a = tv * wa[i * wa_cols + p];
            for (0..wb_cols) |r| out[(p * wb_cols + r) * kk + s] += a * wb[j * wb_cols + r];
        }
    };
    return out;
}

fn f32Of(gpa: std.mem.Allocator, v: TensorView) ![]f32 {
    return v.toF32Alloc(gpa);
}

/// A patch with its factors in f32, ready to produce the difference one output
/// row at a time. Every element of a row is its own dot product, so the row
/// order changes no value.
pub const Prepared = struct {
    /// `(strength * alpha) * diff`, the product of two Python floats.
    s: f32,
    /// The alpha ratio alone, which DoRA scales by instead.
    ratio: f32,
    /// Elements per output row.
    cols: usize,
    kind: union(enum) {
        /// `diff[o] = sum_r up[o][r] * down[r]`.
        lora: struct { up: []f32, down: []f32, r: usize },
        /// `m1 * m2`, each a factor product (or a whole CP product).
        loha: struct { a1: []f32, b1: []f32, a2: []f32, b2: []f32, r1: usize, r2: usize, full1: ?[]f32 = null, full2: ?[]f32 = null },
        /// `kron(w1, w2)`: row `i*c + k` is `w1[i][j] * w2[k][l][s]`.
        lokr: struct { w1: []f32, w2: []f32, a: usize, b: usize, c: usize, d: usize, s: usize },
        diff: []f32,
    },

    pub fn deinit(self: *Prepared, gpa: std.mem.Allocator) void {
        switch (self.kind) {
            .lora => |l| {
                gpa.free(l.up);
                gpa.free(l.down);
            },
            .loha => |h| {
                for ([_][]f32{ h.a1, h.b1, h.a2, h.b2 }) |x| gpa.free(x);
                if (h.full1) |x| gpa.free(x);
                if (h.full2) |x| gpa.free(x);
            },
            .lokr => |kr| {
                gpa.free(kr.w1);
                gpa.free(kr.w2);
            },
            .diff => |d| gpa.free(d),
        }
    }

    /// Output row `o`'s difference, unscaled, into `dst` (`cols` long).
    pub fn row(self: *const Prepared, o: usize, dst: []f32, tmp: []f32) void {
        switch (self.kind) {
            .lora => |l| {
                @memset(dst, 0);
                for (0..l.r) |j| axpy(dst, l.up[o * l.r + j], l.down[j * self.cols ..][0..self.cols]);
            },
            .loha => |h| {
                if (h.full1) |f1| {
                    for (dst, f1[o * self.cols ..][0..self.cols], h.full2.?[o * self.cols ..][0..self.cols]) |*d, x, y| d.* = y * x;
                    return;
                }
                @memset(dst, 0);
                @memset(tmp, 0);
                for (0..h.r1) |j| axpy(dst, h.a1[o * h.r1 + j], h.b1[j * self.cols ..][0..self.cols]);
                for (0..h.r2) |j| axpy(tmp, h.a2[o * h.r2 + j], h.b2[j * self.cols ..][0..self.cols]);
                for (dst, tmp) |*d, y| d.* = y * d.*;
            },
            .lokr => |kr| {
                const i = o / kr.c;
                const k = o % kr.c;
                for (0..kr.b) |j| {
                    const w = kr.w1[i * kr.b + j];
                    for (0..kr.d) |l| for (0..kr.s) |e| {
                        dst[(j * kr.d + l) * kr.s + e] = w * kr.w2[(k * kr.d + l) * kr.s + e];
                    };
                }
            },
            .diff => |d| @memcpy(dst, d[o * self.cols ..][0..self.cols]),
        }
    }
};

pub fn prepare(gpa: std.mem.Allocator, p: Patch, out_rows: usize, numel: usize, strength: f32) !Prepared {
    const cols = numel / out_rows;
    var ratio: f32 = 1.0;
    const kind: @FieldType(Prepared, "kind") = switch (p.kind) {
        .diff => |dv| blk: {
            if (dv.info.elemCount() != numel) return error.ShapeMismatch;
            break :blk .{ .diff = try f32Of(gpa, dv) };
        },
        .lora => |l| blk: {
            const r = rows(l.down);
            var down = try f32Of(gpa, l.down);
            errdefer gpa.free(down);
            var k = inner(l.down);
            if (l.mid) |mid_v| {
                // down2[i, c, s] = sum_j down[j, c] * mid[i, j, s]
                const mid = try f32Of(gpa, mid_v);
                defer gpa.free(mid);
                const ms = mid_v.info.shape.slice();
                if (ms.len != 4 or ms[0] != r or ms[1] != r) return error.ShapeMismatch;
                const kk = ms[2] * ms[3];
                const cin = k;
                const d2 = try gpa.alloc(f32, r * cin * kk);
                @memset(d2, 0);
                for (0..r) |i| for (0..r) |j| for (0..cin) |c| {
                    const dv = down[j * cin + c];
                    for (0..kk) |s| d2[(i * cin + c) * kk + s] += dv * mid[(i * r + j) * kk + s];
                };
                gpa.free(down);
                down = d2;
                k = cin * kk;
            }
            if (inner(l.up) != r or rows(l.up) != out_rows or k != cols) return error.ShapeMismatch;
            if (p.alpha) |a| ratio = @floatCast(@as(f64, a) / @as(f64, @floatFromInt(r)));
            break :blk .{ .lora = .{ .up = try f32Of(gpa, l.up), .down = down, .r = r } };
        },
        .loha => |h| blk: {
            if (p.alpha) |a| ratio = @floatCast(@as(f64, a) / @as(f64, @floatFromInt(rows(h.w1b))));
            if (h.t1) |t1| {
                const f1 = try lohaCp(gpa, h.w1a, h.w1b, t1);
                errdefer gpa.free(f1);
                const f2 = try lohaCp(gpa, h.w2a, h.w2b, h.t2.?);
                errdefer gpa.free(f2);
                if (f1.len != numel or f2.len != numel) return error.ShapeMismatch;
                const none = try gpa.alloc(f32, 0);
                break :blk .{ .loha = .{ .a1 = none, .b1 = try gpa.alloc(f32, 0), .a2 = try gpa.alloc(f32, 0), .b2 = try gpa.alloc(f32, 0), .r1 = 0, .r2 = 0, .full1 = f1, .full2 = f2 } };
            }
            if (rows(h.w1a) != out_rows or rows(h.w2a) != out_rows or inner(h.w1b) != cols or inner(h.w2b) != cols or
                inner(h.w1a) != rows(h.w1b) or inner(h.w2a) != rows(h.w2b)) return error.ShapeMismatch;
            break :blk .{ .loha = .{
                .a1 = try f32Of(gpa, h.w1a),
                .b1 = try f32Of(gpa, h.w1b),
                .a2 = try f32Of(gpa, h.w2a),
                .b2 = try f32Of(gpa, h.w2b),
                .r1 = rows(h.w1b),
                .r2 = rows(h.w2b),
            } };
        },
        .lokr => |kr| blk: {
            var dim: ?usize = null;
            const w1 = if (kr.w1) |w| try f32Of(gpa, w) else w: {
                dim = rows(kr.w1b.?);
                break :w try factor(gpa, kr.w1a.?, kr.w1b.?);
            };
            errdefer gpa.free(w1);
            const ab: [2]usize = if (kr.w1) |w| .{ rows(w), inner(w) } else .{ rows(kr.w1a.?), inner(kr.w1b.?) };
            const w2 = if (kr.w2) |w| try f32Of(gpa, w) else w: {
                dim = rows(kr.w2b.?);
                break :w try factor(gpa, kr.w2a.?, kr.w2b.?);
            };
            errdefer gpa.free(w2);
            // w2 as [c][d][s]: a 4-D whole factor carries its kernel.
            const cds: [3]usize = if (kr.w2) |w| c: {
                const sh = w.info.shape.slice();
                break :c .{ sh[0], sh[1], if (sh.len == 4) sh[2] * sh[3] else 1 };
            } else .{ rows(kr.w2a.?), inner(kr.w2b.?), 1 };
            if (ab[0] * cds[0] != out_rows or ab[1] * cds[1] * cds[2] != cols) return error.ShapeMismatch;
            if (p.alpha != null and dim != null) ratio = @floatCast(@as(f64, p.alpha.?) / @as(f64, @floatFromInt(dim.?)));
            break :blk .{ .lokr = .{ .w1 = w1, .w2 = w2, .a = ab[0], .b = ab[1], .c = cds[0], .d = cds[1], .s = cds[2] } };
        },
    };
    return .{ .s = @floatCast(@as(f64, strength) * @as(f64, ratio)), .ratio = ratio, .cols = cols, .kind = kind };
}

fn factor(gpa: std.mem.Allocator, a_v: TensorView, b_v: TensorView) ![]f32 {
    const a = try f32Of(gpa, a_v);
    defer gpa.free(a);
    const b = try f32Of(gpa, b_v);
    defer gpa.free(b);
    if (inner(a_v) != rows(b_v)) return error.ShapeMismatch;
    return mm(gpa, a, b, rows(a_v), rows(b_v), inner(b_v));
}

fn lohaCp(gpa: std.mem.Allocator, wa_v: TensorView, wb_v: TensorView, t_v: TensorView) ![]f32 {
    const t = try f32Of(gpa, t_v);
    defer gpa.free(t);
    const wa = try f32Of(gpa, wa_v);
    defer gpa.free(wa);
    const wb = try f32Of(gpa, wb_v);
    defer gpa.free(wb);
    const ts = t_v.info.shape.slice();
    if (ts.len != 4 or rows(wa_v) != ts[0] or rows(wb_v) != ts[1]) return error.ShapeMismatch;
    return cp(gpa, t, ts, wb, inner(wb_v), wa, inner(wa_v));
}

/// Add one prepared patch to row `w` of the temp weight, the reference's way.
pub inline fn addRow(w: []f32, d: []const f32, p: *const Prepared, strength: f32, mode: Mode) void {
    if (p.kind == .diff) {
        // The reference casts a diff to the weight's dtype BEFORE scaling it.
        for (w, d) |*x, y| x.* = rnd(mode, x.* + rnd(mode, strength * rnd(mode, y)));
    } else {
        for (w, d) |*x, y| x.* = rnd(mode, x.* + rnd(mode, p.s * y));
    }
}

/// Apply one patch to `w`, the target region of the temp weight, in place.
/// `shape` is the region's shape, outermost first.
pub fn apply(gpa: std.mem.Allocator, w: []f32, shape: []const usize, p: Patch, strength: f32, mode: Mode) !void {
    var prep = try prepare(gpa, p, shape[0], w.len, strength);
    defer prep.deinit(gpa);
    if (p.dora) |ds| return decompose(gpa, w, shape, &prep, ds, strength, mode);
    const d = try gpa.alloc(f32, 2 * prep.cols);
    defer gpa.free(d);
    for (0..shape[0]) |o| {
        prep.row(o, d[0..prep.cols], d[prep.cols..]);
        addRow(w[o * prep.cols ..][0..prep.cols], d[0..prep.cols], &prep, strength, mode);
    }
}

/// DoRA's patched weight `w + ratio * diff` (`wc`) and its ratio vector
/// `magnitude / (norm + eps)`, per output row or per input channel, as the reference
/// computes both. Caller frees both.
const DoraParts = struct { wc: []f32, ratio: []f32, on_out: bool };

fn doraParts(gpa: std.mem.Allocator, w: []const f32, shape: []const usize, prep: *const Prepared, ds_v: TensorView, mode: Mode) !DoraParts {
    const ds = try f32Of(gpa, ds_v);
    defer gpa.free(ds);
    const out = shape[0];
    const cols = if (shape.len > 1) shape[1] else 1;
    const kk = w.len / (out * cols);
    const output_axis = ds_v.info.shape.slice()[0] == out;
    if (ds.len != (if (output_axis) out else cols)) return error.ShapeMismatch;

    const wc = try gpa.alloc(f32, w.len);
    errdefer gpa.free(wc);
    const tmp = try gpa.alloc(f32, prep.cols);
    defer gpa.free(tmp);
    for (0..out) |o| {
        const row = wc[o * prep.cols ..][0..prep.cols];
        prep.row(o, row, tmp);
        for (row, w[o * prep.cols ..][0..prep.cols]) |*c, x| c.* = rnd(mode, x + rnd(mode, c.* * prep.ratio));
    }

    const eps: f32 = if (mode == .f16) std.math.floatEps(f16) else std.math.floatEps(f32);
    const acc = try gpa.alloc(f64, ds.len);
    defer gpa.free(acc);
    @memset(acc, 0);
    // Output axis: the BASE weight's row norms. Input axis: the PATCHED weight's
    // per-input-channel norms. Both as the reference computes them.
    for (0..out) |o| for (0..cols) |c| for (0..kk) |s| {
        const e = (o * cols + c) * kk + s;
        if (output_axis) acc[o] += @as(f64, w[e]) * w[e] else acc[c] += @as(f64, wc[e]) * wc[e];
    };
    const ratio = try gpa.alloc(f32, ds.len);
    for (ratio, acc, ds) |*r, a, m| r.* = rnd(mode, m / rnd(mode, rnd(mode, @floatCast(@sqrt(a))) + eps));
    return .{ .wc = wc, .ratio = ratio, .on_out = output_axis };
}

/// `weight_decompose`, DoRA, over the whole region: its norms span it.
fn decompose(gpa: std.mem.Allocator, w: []f32, shape: []const usize, prep: *const Prepared, ds_v: TensorView, strength: f32, mode: Mode) !void {
    const parts = try doraParts(gpa, w, shape, prep, ds_v, mode);
    defer gpa.free(parts.ratio);
    const wc = parts.wc;
    defer gpa.free(wc);
    const out = shape[0];
    const cols = if (shape.len > 1) shape[1] else 1;
    const kk = w.len / (out * cols);
    for (0..out) |o| for (0..cols) |c| for (0..kk) |s| {
        const e = (o * cols + c) * kk + s;
        const i = if (parts.on_out) o else c;
        wc[e] = rnd(mode, wc[e] * parts.ratio[i]);
    };
    if (strength != 1.0) {
        for (w, wc) |*x, c| x.* = rnd(mode, x.* + rnd(mode, strength * rnd(mode, c - x.*)));
    } else @memcpy(w, wc);
}

// --- as a sidecar ------------------------------------------------------------

/// A patch as an exact low-rank pair in f32: `diff = b @ a`, `b` `[out][r]` and
/// `a` `[r][cols]` with `cols` in torch's `[in][kh][kw]` order, and the alpha
/// ratio the reference scales it by.
const Pieces = struct {
    b: []f32,
    a: []f32,
    r: usize,
    ratio: f32,

    fn deinit(self: Pieces, gpa: std.mem.Allocator) void {
        gpa.free(self.b);
        gpa.free(self.a);
    }
};

/// One LoHa half as a rank-`r` pair: `w_a @ w_b`, or for the CP form
/// `m[p, (c, s)] = sum_i wa[i][p] * (sum_j t[i][j][s] * wb[j][c])`.
fn lohaHalf(gpa: std.mem.Allocator, wa_v: TensorView, wb_v: TensorView, t_v: ?TensorView, out_rows: usize, cols: usize) !struct { b: []f32, a: []f32, r: usize } {
    if (t_v) |tv| {
        const ts = tv.info.shape.slice();
        if (ts.len != 4) return error.ShapeMismatch;
        const ri = ts[0];
        const rj = ts[1];
        const kk = ts[2] * ts[3];
        const ci = inner(wb_v);
        if (rows(wa_v) != ri or rows(wb_v) != rj or inner(wa_v) != out_rows or ci * kk != cols) return error.ShapeMismatch;
        const t = try f32Of(gpa, tv);
        defer gpa.free(t);
        const wa = try f32Of(gpa, wa_v);
        defer gpa.free(wa);
        const wb = try f32Of(gpa, wb_v);
        defer gpa.free(wb);
        const b = try gpa.alloc(f32, out_rows * ri);
        for (0..out_rows) |o| for (0..ri) |i| {
            b[o * ri + i] = wa[i * out_rows + o];
        };
        const a = try gpa.alloc(f32, ri * cols);
        @memset(a, 0);
        for (0..ri) |i| for (0..rj) |j| for (0..ci) |c| {
            const w = wb[j * ci + c];
            for (0..kk) |e| a[i * cols + c * kk + e] += t[(i * rj + j) * kk + e] * w;
        };
        return .{ .b = b, .a = a, .r = ri };
    }
    if (rows(wa_v) != out_rows or inner(wb_v) != cols or inner(wa_v) != rows(wb_v)) return error.ShapeMismatch;
    return .{ .b = try f32Of(gpa, wa_v), .a = try f32Of(gpa, wb_v), .r = rows(wb_v) };
}

fn pieces(gpa: std.mem.Allocator, p: Patch, out_rows: usize, cols: usize) !Pieces {
    switch (p.kind) {
        .lora => |l| {
            // `prepare` already folds a Tucker core into `down`; reuse it.
            var prep = try prepare(gpa, .{ .kind = p.kind, .alpha = p.alpha, .dora = null }, out_rows, out_rows * cols, 1);
            const lr = prep.kind.lora;
            _ = l;
            prep.kind = .{ .diff = &.{} };
            return .{ .b = lr.up, .a = lr.down, .r = lr.r, .ratio = prep.ratio };
        },
        .loha => |h| {
            const h1 = try lohaHalf(gpa, h.w1a, h.w1b, h.t1, out_rows, cols);
            defer {
                gpa.free(h1.b);
                gpa.free(h1.a);
            }
            const h2 = try lohaHalf(gpa, h.w2a, h.w2b, h.t2, out_rows, cols);
            defer {
                gpa.free(h2.b);
                gpa.free(h2.a);
            }
            // (B1 A1) * (B2 A2) elementwise is B A at rank r1*r2, with
            // B[o][(i, j)] = B1[o][i] B2[o][j] and A[(i, j)][c] = A1[i][c] A2[j][c].
            const r = h1.r * h2.r;
            const b = try gpa.alloc(f32, out_rows * r);
            errdefer gpa.free(b);
            for (0..out_rows) |o| for (0..h1.r) |i| for (0..h2.r) |j| {
                b[o * r + i * h2.r + j] = h1.b[o * h1.r + i] * h2.b[o * h2.r + j];
            };
            const a = try gpa.alloc(f32, r * cols);
            for (0..h1.r) |i| for (0..h2.r) |j| for (0..cols) |c| {
                a[(i * h2.r + j) * cols + c] = h1.a[i * cols + c] * h2.a[j * cols + c];
            };
            const ratio: f32 = if (p.alpha) |al| @floatCast(@as(f64, al) / @as(f64, @floatFromInt(rows(h.w1b)))) else 1.0;
            return .{ .b = b, .a = a, .r = r, .ratio = ratio };
        },
        .lokr, .diff => return error.UnsupportedLora,
    }
}

/// Bytes of an f32 slice as bf16, into `alloc`.
fn toF16(alloc: std.mem.Allocator, x: []const f32) ![]u8 {
    const out = try alloc.alloc(u8, x.len * 2);
    tp_core.dtype.f32ToF16Row(x, out);
    return out;
}

fn toBf16(alloc: std.mem.Allocator, x: []const f32) ![]u8 {
    const out = try alloc.alloc(u8, x.len * 2);
    tp_core.dtype.f32ToBf16Row(x, out);
    return out;
}

/// What one patch becomes in a sidecar target.
pub const Term = union(enum) {
    factor: struct { f: lora_mod.Factor, dora: ?lora_mod.Dora },
    kron: lora_mod.Kron,
    diff: lora_mod.Diff,
};

/// The sidecar term for one patch over a base tensor of `shape` (`base`, for DoRA's
/// norms). A low-rank factor has its rank padded to `lora_mod.rank_align` with zeros
/// (exact). A 3x3 conv's `a`, a conv LoKr's `w2` and a conv diff are f16 in the file's
/// own `[r][ci][ky][kx]` order, run as checkpoint-order convs (their values stay small,
/// and most files ship them f16); every other weight is bf16, which the device GEMMs
/// read.
/// A base weight as DoRA's norms read it: f32 values, rounded to f16 first when the
/// reference's merge would hold it in fp16 (`mode`).
pub const Base = struct { w: []const f32, mode: Mode };

/// `v` as a `Base`. Caller frees `.w`.
pub fn baseOf(gpa: std.mem.Allocator, v: TensorView) !Base {
    const mode: Mode = if (v.info.dtype == .f32) .f32 else .f16;
    const w = try v.toF32Alloc(gpa);
    if (mode == .f16) for (w) |*x| {
        x.* = r16(x.*);
    };
    return .{ .w = w, .mode = mode };
}

/// The shape of the weight `p` patches, from its own tensors, for a loader that has no
/// model to read it from: `[out][in]`, or `[out][in][kh][kw]` where the patch says so.
/// A decomposed LoKr `w2` cannot say, and is read as a linear's.
pub fn patchShape(p: Patch) !struct { dims: [4]usize, len: usize } {
    var d: [4]usize = .{ 0, 0, 1, 1 };
    switch (p.kind) {
        .diff => |v| {
            const sh = v.info.shape.slice();
            if (sh.len != 2 and sh.len != 4) return error.ShapeMismatch;
            @memcpy(d[0..sh.len], sh);
            return .{ .dims = d, .len = sh.len };
        },
        .lora => |l| {
            const dn = l.down.info.shape.slice();
            d[0] = rows(l.up);
            d[1] = if (dn.len >= 2) dn[1] else return error.ShapeMismatch;
            const k = if (l.mid) |m| m.info.shape.slice() else dn;
            if (k.len == 4) {
                d[2] = k[2];
                d[3] = k[3];
            }
        },
        .loha => |h| {
            if (h.t1) |t| {
                d[0] = inner(h.w1a);
                const ts = t.info.shape.slice();
                d[2] = ts[2];
                d[3] = ts[3];
            } else d[0] = rows(h.w1a);
            d[1] = inner(h.w1b) / (d[2] * d[3]);
        },
        .lokr => |k| {
            const ab: [2]usize = if (k.w1) |w| .{ rows(w), inner(w) } else .{ rows(k.w1a.?), inner(k.w1b.?) };
            if (k.w2) |w| {
                const sh = w.info.shape.slice();
                d[0] = ab[0] * sh[0];
                d[1] = ab[1] * sh[1];
                if (sh.len == 4) {
                    d[2] = sh[2];
                    d[3] = sh[3];
                }
            } else {
                d[0] = ab[0] * rows(k.w2a.?);
                d[1] = ab[1] * inner(k.w2b.?);
            }
        },
    }
    return .{ .dims = d, .len = if (d[2] * d[3] == 1) 2 else 4 };
}

pub fn termFor(gpa: std.mem.Allocator, alloc: std.mem.Allocator, p: Patch, base: ?Base, shape: []const usize, out_rows: usize, out_off: usize) !Term {
    const cols = blk: {
        var n: usize = 1;
        for (shape[1..]) |d| n *= d;
        break :blk n;
    };
    const conv3 = shape.len == 4 and shape[2] > 1;
    switch (p.kind) {
        .lokr => {
            if (p.dora != null) return error.UnsupportedLora;
            var prep = try prepare(gpa, p, out_rows, out_rows * cols, 1);
            defer prep.deinit(gpa);
            const kr = prep.kind.lokr;
            const w2_cols = kr.d * kr.s;
            const w2 = if (conv3) try toF16(alloc, kr.w2) else try toBf16(alloc, kr.w2);
            return .{ .kron = .{
                .w1 = try alloc.dupe(f32, kr.w1),
                .ol = kr.a,
                .im = kr.b,
                .w2 = Weight.init(w2, if (conv3) .f16 else .bf16, kr.c, w2_cols),
                .out_off = out_off,
                .gain = prep.ratio,
            } };
        },
        .diff => {
            var prep = try prepare(gpa, p, out_rows, out_rows * cols, 1);
            defer prep.deinit(gpa);
            const d = prep.kind.diff;
            const bytes = if (conv3) try toF16(alloc, d) else try toBf16(alloc, d);
            return .{ .diff = .{ .w = Weight.init(bytes, if (conv3) .f16 else .bf16, out_rows, cols), .out_off = out_off } };
        },
        .lora, .loha => {},
    }
    const pc = try pieces(gpa, p, out_rows, cols);
    defer pc.deinit(gpa);

    // DoRA's ratio, folded into the factor: B's rows on the output axis, A's input
    // channels on the input axis. What the base needs is left to the apply.
    var dora: ?lora_mod.Dora = null;
    if (p.dora) |ds| {
        const b = base orelse return error.UnsupportedLora;
        const w = b.w;
        var prep = try prepare(gpa, .{ .kind = p.kind, .alpha = p.alpha, .dora = null }, out_rows, w.len, 1);
        defer prep.deinit(gpa);
        const parts = try doraParts(gpa, w, shape, &prep, ds, b.mode);
        defer gpa.free(parts.wc);
        defer gpa.free(parts.ratio);
        const kk = cols / (if (shape.len > 1) shape[1] else 1);
        if (parts.on_out) {
            for (0..out_rows) |o| for (pc.b[o * pc.r ..][0..pc.r]) |*x| {
                x.* *= parts.ratio[o];
            };
        } else {
            for (0..pc.r) |j| for (pc.a[j * cols ..][0..cols], 0..) |*x, c| {
                x.* *= parts.ratio[c / kk];
            };
        }
        dora = .{ .on_out = parts.on_out, .ratio = try alloc.dupe(f32, parts.ratio) };
    }

    const rp = std.mem.alignForward(usize, pc.r, lora_mod.rank_align);
    const b = try gpa.alloc(f32, out_rows * rp);
    defer gpa.free(b);
    @memset(b, 0);
    for (0..out_rows) |o| @memcpy(b[o * rp ..][0..pc.r], pc.b[o * pc.r ..][0..pc.r]);

    const a = try gpa.alloc(f32, rp * cols);
    defer gpa.free(a);
    @memset(a, 0);
    @memcpy(a[0 .. pc.r * cols], pc.a);

    return .{ .factor = .{ .f = .{
        .a = if (conv3) Weight.init(try toF16(alloc, a), .f16, rp, cols) else Weight.init(try toBf16(alloc, a), .bf16, rp, cols),
        .b = Weight.init(try toBf16(alloc, b), .bf16, out_rows, rp),
        .out_off = out_off,
        .gain = pc.ratio,
    }, .dora = dora } };
}

// --- tests -------------------------------------------------------------------

const testing = std.testing;
const fixture = @embedFile("assets/lora_merge.safetensors");

fn fixtureStore(st: *const tp_core.safetensors.SafeTensors) WeightStore {
    return .{ .safetensors = st };
}

test "every adapter merges as ComfyUI merges it" {
    const gpa = testing.allocator;
    var st = try tp_core.safetensors.SafeTensors.initFromSlice(gpa, fixture);
    defer st.deinit();
    const store = fixtureStore(&st);
    const cases = [_][]const u8{ "lora", "noalpha", "f32", "locon", "tucker", "peft", "loha", "loha_cp", "lokr_full", "lokr_dec", "dora_out", "dora_in", "diff", "stack", "lokr_conv", "lokr_conv_dec", "diff_conv", "dora_in_s", "dora_in_conv" };
    for (cases) |name| {
        var buf: [64]u8 = undefined;
        const base_v = try store.require(try std.fmt.bufPrint(&buf, "{s}.base", .{name}));
        const want_v = try store.require(try std.fmt.bufPrint(&buf, "{s}.want", .{name}));
        const mode: Mode = if (base_v.info.dtype == .f32) .f32 else .f16;
        const w = try base_v.toF32Alloc(gpa);
        defer gpa.free(w);
        if (mode == .f16) for (w) |*x| {
            x.* = r16(x.*);
        };
        var fi: usize = 0;
        while (true) : (fi += 1) {
            var sb: [64]u8 = undefined;
            const stem = try std.fmt.bufPrint(&sb, "{s}.{d}", .{ name, fi });
            const sv = store.get(try std.fmt.bufPrint(&buf, "{s}.strength", .{stem})) orelse break;
            var consumed: Consumed = .empty;
            defer consumed.deinit(gpa);
            const p = (try parse(gpa, store, stem, &consumed)).?;
            try apply(gpa, w, base_v.info.shape.slice(), p, try sv.asScalarF32(), mode);
        }
        const want = try want_v.toF32Alloc(gpa);
        defer gpa.free(want);
        // Bit-exact: every rounding is the reference's, at the same point.
        var off: usize = 0;
        for (w, want) |g, x| off += @intFromBool(g != x);
        errdefer std.debug.print("{s}: {d} of {d} differ\n", .{ name, off, w.len });
        try testing.expectEqual(@as(usize, 0), off);
    }
}

test "a sidecar reproduces the merge's product for every adapter, linear and conv" {
    const gpa = testing.allocator;
    const io = testing.io;
    var st = try tp_core.safetensors.SafeTensors.initFromSlice(gpa, fixture);
    defer st.deinit();
    const store = fixtureStore(&st);
    const cases = [_][]const u8{ "lora", "noalpha", "f32", "locon", "tucker", "peft", "loha", "loha_cp", "lokr_full", "lokr_dec", "dora_out", "dora_in", "diff", "lokr_conv", "lokr_conv_dec", "diff_conv", "dora_in_s", "dora_in_conv" };
    var prng = std.Random.DefaultPrng.init(17);
    const rnd_g = prng.random();
    for (cases) |name| {
        var buf: [64]u8 = undefined;
        const base_v = try store.require(try std.fmt.bufPrint(&buf, "{s}.base", .{name}));
        const want_v = try store.require(try std.fmt.bufPrint(&buf, "{s}.want", .{name}));
        const strength = try (try store.require(try std.fmt.bufPrint(&buf, "{s}.0.strength", .{name}))).asScalarF32();
        const shape = base_v.info.shape.slice();
        const out = shape[0];
        const cols = base_v.info.elemCount() / out;
        var sb: [64]u8 = undefined;
        var consumed: Consumed = .empty;
        defer consumed.deinit(gpa);
        const p = (try parse(gpa, store, try std.fmt.bufPrint(&sb, "{s}.0", .{name}), &consumed)).?;

        var arena = std.heap.ArenaAllocator.init(gpa);
        const alloc = arena.allocator();
        const base = try baseOf(gpa, base_v);
        defer gpa.free(base.w);
        const term = try termFor(gpa, alloc, p, base, shape, out, 0);
        var t: lora_mod.Target = .{ .factors = &.{}, .in_dim = cols, .out_dim = out, .scale = 1, .tag = "w.weight" };
        switch (term) {
            .factor => |f| {
                t.factors = try alloc.dupe(lora_mod.Factor, &.{f.f});
                t.dora = f.dora;
            },
            .kron => |k| t.kron = try alloc.dupe(lora_mod.Kron, &.{k}),
            .diff => |d| t.diffs = try alloc.dupe(lora_mod.Diff, &.{d}),
        }
        var index: std.StringHashMapUnmanaged(lora_mod.Target) = .empty;
        try index.put(alloc, "w.weight", t);
        var stack: lora_mod.Stack = .{};
        defer stack.deinit(gpa);
        _ = try stack.addSidecar(gpa, .{ .arena = arena, .index = index, .unclaimed = 0 }, strength, name);

        // Each weight as the CPU forward reads it: f32, torch order.
        const wb = try base_v.toF32Alloc(gpa);
        defer gpa.free(wb);
        const ww = try want_v.toF32Alloc(gpa);
        defer gpa.free(ww);
        var base_w = Weight.init(std.mem.sliceAsBytes(wb), .f32, out, cols);
        base_w.tag = "w.weight";
        const want_w = Weight.init(std.mem.sliceAsBytes(ww), .f32, out, cols);

        const conv = shape.len == 4 and shape[2] > 1;
        const hw: usize = 5;
        const m: usize = if (conv) hw * hw else 7;
        const in_ch: usize = if (conv) shape[1] else cols;
        const x = try gpa.alloc(f32, m * in_ch);
        defer gpa.free(x);
        for (x) |*v| v.* = rnd_g.floatNorm(f32);
        const y_base = try gpa.alloc(f32, m * out);
        defer gpa.free(y_base);
        const y_want = try gpa.alloc(f32, m * out);
        defer gpa.free(y_want);
        const y_got = try gpa.alloc(f32, m * out);
        defer gpa.free(y_got);
        if (conv) {
            const mk = struct {
                fn c(w: Weight, co: usize, ci: usize) ops.conv.Conv2d {
                    return .{ .w = .{ .torch = .{ .w = w } }, .co = co, .ci = ci, .k = 3, .pad = 1, .tag = "w.weight" };
                }
            }.c;
            try ops.conv.conv2d(io, gpa, y_base, x, hw, hw, mk(base_w, out, in_ch));
            try ops.conv.conv2d(io, gpa, y_want, x, hw, hw, mk(want_w, out, in_ch));
            @memcpy(y_got, y_base);
            try stack.applyHostConv(io, gpa, y_got, x, hw, hw, mk(base_w, out, in_ch));
        } else {
            try ops.matmul.matmul(io, gpa, y_base, x, m, base_w, null);
            try ops.matmul.matmul(io, gpa, y_want, x, m, want_w, null);
            @memcpy(y_got, y_base);
            try stack.applyHost(io, gpa, y_got, out, x, m, base_w, null);
        }
        for (y_want, y_got, y_base) |*a, *g, b0| {
            a.* -= b0;
            g.* -= b0;
        }
        const rel = relErr(y_want, y_got);
        errdefer std.debug.print("{s}: rel {e}\n", .{ name, rel });
        // 16-bit factors against an fp16 merge, so a rounding of each, not exactness.
        // LoHa's rank r1*r2 factor is a product of two bf16-rounded halves (1.1e-2 on
        // the CP form); every other kind sits at 0.04-0.8%.
        try testing.expect(rel < @as(f64, if (std.mem.startsWith(u8, name, "loha")) 2e-2 else 1e-2));
    }
}

fn relErr(want: []const f32, got: []const f32) f64 {
    var num: f64 = 0;
    var den: f64 = 0;
    for (want, got) |w, g| {
        num += (@as(f64, w) - g) * (@as(f64, w) - g);
        den += @as(f64, w) * w;
    }
    return @sqrt(num / @max(den, 1e-30));
}
