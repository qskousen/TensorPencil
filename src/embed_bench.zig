//! embed-bench: throughput profiler for the `tp.embed` encoders. `zig build
//! embed-bench -- [opts]`. Times the DiffKeep encoders (Snowflake,
//! EmbeddingGemma 1 and 2, SigLIP2 text, SigLIP2 visual) on a chosen backend, at a
//! chosen batch size, so we can measure the win from true batched forwards
//! against the per-item baseline (batch=1).
//!
//! The harness drives the public façade `embedTextBatch`/`embedImageBatch` in
//! chunks of `--batch`, so batch=1 reproduces the per-item loop and larger
//! batches exercise whatever batched path the façade routes to.
//!
//!   --backend cpu|vulkan|cuda|cuda-libs   compute backend (default cpu)
//!   --model all|snowflake|gemma|gemma2|siglip-text|siglip-visual|gemma2-image  (default all)
//!   --budget N                  gemma2-image soft tokens per image: 70|140|280|560|1120 (default 280)
//!   --stages                    gemma2-image: time preprocessing, tower and text body apart
//!   --img WxH                   gemma2-image source size, RGB8 resized by the encoder (default 1024x768)
//!   --count N                   total items to encode (default 64)
//!   --batch B                   items per batched forward (default 1)
//!   --seq S                     approx target token count for text (default ~16)
//!   --reps R                    timed repetitions, best-of reported (default 3)
//!   --hash                      also print an FNV hash of every item's embedding, so
//!                               two builds can be compared byte for byte

const std = @import("std");
const tp = @import("TensorPencil");

const embed = tp.embed;

fn nowNs(io: std.Io) u64 {
    return @intCast(std.Io.Clock.real.now(io).nanoseconds);
}

const Which = enum { snowflake, gemma, gemma2, siglip_text, siglip_visual };

const model_dir = struct {
    const snowflake = "../DiffKeep/Models/snowflake-arctic-embed-m-v2.0";
    const gemma = "../DiffKeep/Models/embeddinggemma-300m";
    const gemma2 = "../DiffKeep/Models/embeddinggemma-2";
    const siglip = "../DiffKeep/Models/ViT-B-16-SigLIP2-timm";
};

const Args = struct {
    backend_name: []const u8 = "cpu",
    model: []const u8 = "all",
    count: usize = 64,
    batch: usize = 1,
    seq: usize = 16,
    reps: usize = 3,
    prof: bool = false,
    hash: bool = false,
    img_w: usize = 1024,
    img_h: usize = 768,
    stages: bool = false,
    budget: usize = 280,
};

fn parseArgs(argv: []const []const u8) !Args {
    var a: Args = .{};
    var i: usize = 1;
    while (i < argv.len) : (i += 1) {
        const arg = argv[i];
        const next = if (i + 1 < argv.len) argv[i + 1] else "";
        if (std.mem.eql(u8, arg, "--backend")) {
            a.backend_name = next;
            i += 1;
        } else if (std.mem.eql(u8, arg, "--model")) {
            a.model = next;
            i += 1;
        } else if (std.mem.eql(u8, arg, "--count")) {
            a.count = try std.fmt.parseInt(usize, next, 10);
            i += 1;
        } else if (std.mem.eql(u8, arg, "--batch")) {
            a.batch = try std.fmt.parseInt(usize, next, 10);
            i += 1;
        } else if (std.mem.eql(u8, arg, "--seq")) {
            a.seq = try std.fmt.parseInt(usize, next, 10);
            i += 1;
        } else if (std.mem.eql(u8, arg, "--reps")) {
            a.reps = try std.fmt.parseInt(usize, next, 10);
            i += 1;
        } else if (std.mem.eql(u8, arg, "--img")) {
            const x = std.mem.findScalar(u8, next, 'x') orelse return error.BadImageSize;
            a.img_w = try std.fmt.parseInt(usize, next[0..x], 10);
            a.img_h = try std.fmt.parseInt(usize, next[x + 1 ..], 10);
            i += 1;
        } else if (std.mem.eql(u8, arg, "--budget")) {
            a.budget = try std.fmt.parseInt(usize, next, 10);
            i += 1;
        } else if (std.mem.eql(u8, arg, "--stages")) {
            a.stages = true;
        } else if (std.mem.eql(u8, arg, "--hash")) {
            a.hash = true;
        } else if (std.mem.eql(u8, arg, "--prof")) {
            a.prof = true;
        }
    }
    return a;
}

// Dump the CUDA backend's per-category device profiler (sync-per-op timing:
// each op fences the stream, so `ms` is real device time and `ops` is the op
// count, the count reveals how the per-item attention/RoPE loop inflates the
// attn/elt categories with batch size).
fn dumpCudaProf(be: *tp.gpu.cuda.Backend) void {
    var tot: f64 = 0;
    for (be.prof.ms) |ms| tot += ms;
    p("  [cuda per-op profile, sync-per-op] total {d:.2} ms over one batch\n", .{tot});
    for (@typeInfo(tp.gpu.cuda.Backend.ProfCat).@"enum".field_names, be.prof.ms, be.prof.n) |name, ms, n| {
        if (n == 0) continue;
        const per = ms / @as(f64, @floatFromInt(n));
        const pct = if (tot > 0) 100.0 * ms / tot else 0;
        p("    {s:<12} {d:>8.2} ms {d:>5.0}%  {d:>6} ops {d:>8.4} ms/op\n", .{ name, ms, pct, n, per });
    }
}

// Build `count` distinct-ish text inputs of roughly `seq` tokens each. The
// base phrase is ~14 BPE tokens; we repeat it to approach the target and
// suffix the index so tokenization differs per item.
fn buildTexts(gpa: std.mem.Allocator, count: usize, seq: usize) ![][]const u8 {
    const base = "a photograph of a red bicycle leaning against a brick wall ";
    const reps = @max(1, seq / 12);
    const texts = try gpa.alloc([]const u8, count);
    for (texts, 0..) |*t, i| {
        var buf: std.ArrayList(u8) = .empty;
        for (0..reps) |_| try buf.appendSlice(gpa, base);
        const suffix = try gpa.print("#{d}", .{i});
        try buf.appendSlice(gpa, suffix);
        t.* = try buf.toOwnedSlice(gpa);
    }
    return texts;
}

fn buildImages(gpa: std.mem.Allocator, count: usize) ![][]const f32 {
    const n = 3 * 224 * 224;
    const imgs = try gpa.alloc([]const f32, count);
    for (imgs, 0..) |*im, i| {
        const buf = try gpa.alloc(f32, n);
        // deterministic pattern in the preprocessed [-1,1] range
        for (buf, 0..) |*v, j| v.* = @sin(@as(f32, @floatFromInt((j + i * 7) % 257)) * 0.0245);
        im.* = buf;
    }
    return imgs;
}

const p = std.debug.print;

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    const io = init.io;
    const argv = try init.minimal.args.toSlice(gpa);
    const a = try parseArgs(argv);

    const backend: embed.Backend = blk: {
        if (std.mem.eql(u8, a.backend_name, "cpu")) break :blk .cpu;
        if (std.mem.eql(u8, a.backend_name, "vulkan")) {
            const ctx = try tp.gpu.context.Context.init(gpa, io);
            break :blk .{ .vulkan = ctx };
        }
        if (std.mem.eql(u8, a.backend_name, "cuda")) {
            const be = try tp.gpu.cuda.Backend.init(gpa);
            break :blk .{ .cuda = be };
        }
        if (std.mem.eql(u8, a.backend_name, "cuda-libs")) {
            const be = try tp.gpu.cuda.Backend.initLibs(gpa);
            break :blk .{ .cuda = be };
        }
        return error.UnknownBackend;
    };

    p("embed-bench: backend={s} count={d} batch={d} seq~{d} reps={d}\n", .{ a.backend_name, a.count, a.batch, a.seq, a.reps });
    p("{s:<16} {s:>10} {s:>12} {s:>12}\n", .{ "model", "ms/item", "items/s", "ms/batch" });

    const want_all = std.mem.eql(u8, a.model, "all");
    const sel = struct {
        fn on(name: []const u8, m: []const u8, all: bool) bool {
            return all or std.mem.eql(u8, m, name);
        }
    }.on;

    if (sel("snowflake", a.model, want_all)) try benchText(gpa, io, .snowflake, backend, a);
    if (sel("gemma", a.model, want_all)) try benchText(gpa, io, .gemma, backend, a);
    if (sel("gemma2", a.model, want_all)) try benchText(gpa, io, .gemma2, backend, a);
    if (sel("siglip-text", a.model, want_all)) try benchText(gpa, io, .siglip_text, backend, a);
    if (sel("siglip-visual", a.model, want_all)) try benchImage(gpa, io, backend, a);
    if (sel("gemma2-image", a.model, want_all)) try benchGemma2Image(gpa, io, backend, a);
    // Tear down the handle: `TP_VK_PROFILE` reports at `Context.deinit`.
    switch (backend) {
        .cpu => {},
        .vulkan => |ctx| ctx.deinit(),
        .cuda => |be| be.deinit(),
    }
}

fn benchText(gpa: std.mem.Allocator, io: std.Io, which: Which, backend: embed.Backend, a: Args) !void {
    const kind: embed.TextModelKind = switch (which) {
        .snowflake => .arctic_embed_m_v2,
        .gemma => .embeddinggemma,
        .gemma2 => .embeddinggemma2,
        .siglip_text => .siglip2_text,
        .siglip_visual => unreachable,
    };
    const dir = switch (which) {
        .snowflake => model_dir.snowflake,
        .gemma => model_dir.gemma,
        .gemma2 => model_dir.gemma2,
        .siglip_text => model_dir.siglip,
        .siglip_visual => unreachable,
    };
    std.Io.Dir.cwd().access(io, dir, .{}) catch {
        p("{s:<16} (skipped: {s} missing)\n", .{ @tagName(which), dir });
        return;
    };
    var enc = try embed.TextEncoder.open(gpa, io, kind, dir, backend);
    defer enc.deinit();

    const texts = try buildTexts(gpa, a.count, a.seq);
    const opts: embed.Options = .{ .role = .document };

    // Warmup (also primes weights / device buffers).
    {
        const w = try enc.embedTextBatch(gpa, texts[0..@min(a.batch, a.count)], opts);
        for (w) |v| gpa.free(v);
        gpa.free(w);
    }

    var best: u64 = std.math.maxInt(u64);
    for (0..a.reps) |_| {
        const t0 = nowNs(io);
        var off: usize = 0;
        while (off < a.count) {
            const end = @min(off + a.batch, a.count);
            const vecs = try enc.embedTextBatch(gpa, texts[off..end], opts);
            for (vecs) |v| gpa.free(v);
            gpa.free(vecs);
            off = end;
        }
        best = @min(best, nowNs(io) - t0);
    }
    report(@tagName(which), best, a);
    if (a.hash) {
        var h: std.hash.Fnv1a_64 = .init();
        var off: usize = 0;
        while (off < a.count) {
            const end = @min(off + a.batch, a.count);
            const vecs = try enc.embedTextBatch(gpa, texts[off..end], opts);
            for (vecs) |v| {
                h.update(std.mem.sliceAsBytes(v));
                gpa.free(v);
            }
            gpa.free(vecs);
            off = end;
        }
        p("{s:<16} hash {x:0>16}\n", .{ @tagName(which), h.final() });
    }

    // Optional: one profiled batch on CUDA (per-category device timing).
    if (a.prof) switch (backend) {
        .cuda => |be| {
            be.profile = true;
            be.prof.reset();
            const vecs = try enc.embedTextBatch(gpa, texts[0..@min(a.batch, a.count)], opts);
            for (vecs) |v| gpa.free(v);
            gpa.free(vecs);
            be.profile = false;
            dumpCudaProf(be);
        },
        else => {},
    };
}

fn benchImage(gpa: std.mem.Allocator, io: std.Io, backend: embed.Backend, a: Args) !void {
    const dir = model_dir.siglip;
    std.Io.Dir.cwd().access(io, dir, .{}) catch {
        p("{s:<16} (skipped: {s} missing)\n", .{ "siglip-visual", dir });
        return;
    };
    var enc = try embed.ImageEncoder.open(gpa, io, dir, backend);
    defer enc.deinit();

    const imgs = try buildImages(gpa, a.count);
    {
        const w = try enc.embedImageBatch(gpa, imgs[0..@min(a.batch, a.count)]);
        for (w) |v| gpa.free(v);
        gpa.free(w);
    }

    var best: u64 = std.math.maxInt(u64);
    for (0..a.reps) |_| {
        const t0 = nowNs(io);
        var off: usize = 0;
        while (off < a.count) {
            const end = @min(off + a.batch, a.count);
            const vecs = try enc.embedImageBatch(gpa, imgs[off..end]);
            for (vecs) |v| gpa.free(v);
            gpa.free(vecs);
            off = end;
        }
        best = @min(best, nowNs(io) - t0);
    }
    report("siglip-visual", best, a);
    if (a.hash) {
        var h: std.hash.Fnv1a_64 = .init();
        var off: usize = 0;
        while (off < a.count) {
            const end = @min(off + a.batch, a.count);
            const vecs = try enc.embedImageBatch(gpa, imgs[off..end]);
            for (vecs) |v| {
                h.update(std.mem.sliceAsBytes(v));
                gpa.free(v);
            }
            gpa.free(vecs);
            off = end;
        }
        p("{s:<16} hash {x:0>16}\n", .{ "siglip-visual", h.final() });
    }

    if (a.prof) switch (backend) {
        .cuda => |be| {
            be.profile = true;
            be.prof.reset();
            const vecs = try enc.embedImageBatch(gpa, imgs[0..@min(a.batch, a.count)]);
            for (vecs) |v| gpa.free(v);
            gpa.free(vecs);
            be.profile = false;
            dumpCudaProf(be);
        },
        else => {},
    };
}

fn report(name: []const u8, best_ns: u64, a: Args) void {
    const total_ms = @as(f64, @floatFromInt(best_ns)) / 1e6;
    const per_item = total_ms / @as(f64, @floatFromInt(a.count));
    const items_s = 1000.0 / per_item;
    const n_batches = std.math.divCeil(usize, a.count, a.batch) catch 1;
    const per_batch = total_ms / @as(f64, @floatFromInt(n_batches));
    p("{s:<16} {d:>10.3} {d:>12.1} {d:>12.3}\n", .{ name, per_item, items_s, per_batch });
}

// Smooth fields plus fine stripes, distinct per image, so the resize does real work.
fn buildRgb(gpa: std.mem.Allocator, count: usize, w: usize, h: usize) ![]embed.Rgb {
    const imgs = try gpa.alloc(embed.Rgb, count);
    for (imgs, 0..) |*im, i| {
        const px = try gpa.alloc(u8, 3 * w * h);
        for (0..h) |y| for (0..w) |x| {
            const o = (y * w + x) * 3;
            px[o] = @intCast((x * 255 / w + i * 13) % 256);
            px[o + 1] = @intCast((y * 255 / h + i * 29) % 256);
            px[o + 2] = if ((x / 3 + y / 5 + i) % 2 == 0) 230 else 20;
        };
        im.* = .{ .pixels = px, .width = w, .height = h };
    }
    return imgs;
}

fn benchGemma2Image(gpa: std.mem.Allocator, io: std.Io, backend: embed.Backend, a: Args) !void {
    const dir = model_dir.gemma2;
    std.Io.Dir.cwd().access(io, dir, .{}) catch {
        p("{s:<16} (skipped: {s} missing)\n", .{ "gemma2-image", dir });
        return;
    };
    var enc = try embed.MultimodalEncoder.open(gpa, io, dir, backend);
    defer enc.deinit();
    try enc.setImageTokens(a.budget);
    const imgs = try buildRgb(gpa, a.count, a.img_w, a.img_h);
    p("{s:<16} source {d}x{d}, {d} tokens\n", .{ "gemma2-image", a.img_w, a.img_h, a.budget });
    {
        const w = try enc.embedImageBatch(gpa, imgs[0..@min(a.batch, a.count)]);
        for (w) |v| gpa.free(v);
        gpa.free(w);
    }
    var best: u64 = std.math.maxInt(u64);
    for (0..a.reps) |_| {
        const t0 = nowNs(io);
        var off: usize = 0;
        while (off < a.count) {
            const end = @min(off + a.batch, a.count);
            const vecs = try enc.embedImageBatch(gpa, imgs[off..end]);
            for (vecs) |v| gpa.free(v);
            gpa.free(vecs);
            off = end;
        }
        best = @min(best, nowNs(io) - t0);
    }
    report("gemma2-image", best, a);
    if (a.stages) try gemma2Stages(gpa, io, &enc, imgs, backend, a.prof);
}

// One image at a time, each stage timed alone (best of 3 per stage): the CPU
// preprocessing, the vision tower (whose own preprocessing is subtracted), and the
// text body over the image's rows.
fn gemma2Stages(gpa: std.mem.Allocator, io: std.Io, enc: *embed.MultimodalEncoder, imgs: []const embed.Rgb, backend: embed.Backend, prof_tower: bool) !void {
    const models = tp.models;
    const m = &enc.text.impl.embeddinggemma2;
    const vit = &m.vit.?;
    const img = imgs[0];
    var t_prep: u64 = std.math.maxInt(u64);
    var t_vit: u64 = std.math.maxInt(u64);
    var t_body: u64 = std.math.maxInt(u64);
    var out: [embed.dim]f32 = undefined;
    var outs = [_][]f32{&out};
    for (0..3) |_| {
        var t0 = nowNs(io);
        var prep = try vit.prepare(gpa, img.pixels, img.width, img.height);
        t_prep = @min(t_prep, nowNs(io) - t0);
        prep.deinit(gpa);

        if (prof_tower) switch (backend) {
            .cuda => |be| {
                be.profile = true;
                be.prof.reset();
            },
            else => {},
        };
        t0 = nowNs(io);
        var e = switch (backend) {
            .cpu => try vit.encode(io, gpa, img.pixels, img.width, img.height),
            .cuda => |be| try models.gemma4v_vit_cuda.encodeResident(vit, be, io, gpa, img.pixels, img.width, img.height),
            .vulkan => |ctx| try models.gemma4v_vit_gpu.encodeResident(vit, ctx, io, gpa, img.pixels, img.width, img.height),
        };
        t_vit = @min(t_vit, nowNs(io) - t0);
        if (prof_tower) switch (backend) {
            .cuda => |be| {
                be.profile = false;
                dumpCudaProf(be);
            },
            else => {},
        };
        defer e.deinit(gpa);
        const rows = try m.imageRows(gpa, e.embeds);
        defer gpa.free(rows);

        t0 = nowNs(io);
        const rl = [_][]const f32{rows};
        switch (backend) {
            .cpu => try m.embedRows(io, gpa, &rl, &outs),
            .cuda => |be| try models.embed_gemma2_dev.embedRows(models.tower_cuda.Dev{ .be = be }, m, io, gpa, &rl, &outs),
            .vulkan => |ctx| {
                var tl = try models.lin_gpu.Lin.init(gpa, ctx, models.lin_gpu.tower_max_out, 0, 0);
                defer tl.deinit();
                try models.embed_gemma2_dev.embedRows(models.tower_gpu.Dev{ .ctx = ctx, .tl = &tl }, m, io, gpa, &rl, &outs);
            },
        }
        t_body = @min(t_body, nowNs(io) - t0);
    }
    const ms = struct {
        fn f(ns: u64) f64 {
            return @as(f64, @floatFromInt(ns)) / 1e6;
        }
    }.f;
    p("  stages: preprocess {d:.1} ms, tower {d:.1} ms, text body {d:.1} ms\n", .{ ms(t_prep), ms(t_vit -| t_prep), ms(t_body) });
}
