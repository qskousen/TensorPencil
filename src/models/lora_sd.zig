//! The SD family's LoRAs: the key map, the adapter reader, and `sidecars`, which
//! turns a file into `lora.zig` sidecar targets for the UNet and both CLIP towers.
//! `merge` is ComfyUI's merge reproduced exactly; nothing renders through it, it
//! is the oracle a sidecar is checked against and what `lora-merge-check` runs.
//!
//! The adapter forms and their arithmetic are `lora_adapter.zig`'s.
//!
//! **The key map** is a silent wrong answer, a finite plausible render. A LoRA names a
//! UNet weight in LDM terms (`lora_unet_input_blocks_4_1_...`), in diffusers terms
//! (`lora_unet_down_blocks_1_attentions_0_...`) or as PEFT keys, and names CLIP's by
//! `lora_te1`/`lora_te2`. Half the SD files in the wild use each UNet spelling.
//! `keyMap` builds the reference's map the reference's way: it walks the model's own
//! tensor names and probes the file, and `unetToDiffusers` ports
//! `comfy.utils.unet_to_diffusers`. CLIP-G in an LDM checkpoint stores q, k and v as
//! row blocks of one `in_proj_weight`, so a target may be a row range.

const std = @import("std");
const tp_core = @import("tp_core");
const ops = @import("tp_ops");
const lora_mod = @import("lora.zig");
const adapter = @import("lora_adapter.zig");
const Patch = adapter.Patch;
const Consumed = adapter.Consumed;
const Mode = adapter.Mode;
const Prepared = adapter.Prepared;
const parse = adapter.parse;
const prepare = adapter.prepare;
const apply = adapter.apply;
const addRow = adapter.addRow;
const r16 = adapter.r16;
const termFor = adapter.termFor;
const sd_unet = @import("sd_unet.zig");

/// This file's source hashed, for a cache of what a LoRA file was taken for: the
/// catalog mixes it into its rules fingerprint, so an edit here re-scans.
pub const source_fingerprint: u64 = blk: {
    @setEvalBranchQuota(1 << 24);
    break :blk std.hash.Wyhash.hash(0, @embedFile("lora_sd.zig"));
};

const weights_mod = tp_core.weights;
const WeightStore = weights_mod.WeightStore;
const TensorView = weights_mod.TensorView;
const Overlay = weights_mod.Overlay;
const convertToF32 = tp_core.safetensors.convertToF32;
const Weight = ops.matmul.Weight;

pub const Component = enum(u2) { unet, clip_l, clip_g };

/// What one LoRA key stem patches.
pub const Target = struct {
    comp: Component,
    /// The tensor's name in its component's store.
    name: []const u8,
    /// A row block of a fused tensor; `rows == 0` is the whole tensor.
    row_off: usize = 0,
    rows: usize = 0,
};

/// LoRA stem -> target, in insertion order so a merge walks it reproducibly.
pub const KeyMap = std.array_hash_map.String(Target);

/// The component stores a merge reads, and the UNet layout the diffusers names
/// derive from.
pub const Model = struct {
    unet: WeightStore,
    unet_cfg: sd_unet.Config,
    clip_l: ?WeightStore = null,
    clip_l_layers: usize = 0,
    clip_g: ?WeightStore = null,
    clip_g_layers: usize = 0,
    clip_g_hidden: usize = 0,

    fn store(self: Model, c: Component) ?WeightStore {
        return switch (c) {
            .unet => self.unet,
            .clip_l => self.clip_l,
            .clip_g => self.clip_g,
        };
    }
};

// --- the key map -------------------------------------------------------------

/// One `unet_to_diffusers` weight entry: the diffusers name and the LDM name.
pub const Pair = struct { diffusers: []const u8, ldm: []const u8 };

const resnet_map = [_][2][]const u8{
    .{ "in_layers.2.weight", "conv1.weight" },
    .{ "emb_layers.1.weight", "time_emb_proj.weight" },
    .{ "out_layers.3.weight", "conv2.weight" },
    .{ "skip_connection.weight", "conv_shortcut.weight" },
    .{ "in_layers.0.weight", "norm1.weight" },
    .{ "out_layers.0.weight", "norm2.weight" },
};
const attn_weights = [_][]const u8{ "proj_in.weight", "proj_out.weight", "norm.weight" };
const block_weights = [_][]const u8{
    "norm1.weight",          "norm2.weight",          "norm3.weight",
    "attn1.to_q.weight",     "attn1.to_k.weight",     "attn1.to_v.weight",
    "attn1.to_out.0.weight", "attn2.to_q.weight",     "attn2.to_k.weight",
    "attn2.to_v.weight",     "attn2.to_out.0.weight", "ff.net.0.proj.weight",
    "ff.net.2.weight",
};
const basic_map = [_][2][]const u8{
    .{ "class_embedding.linear_1.weight", "label_emb.0.0.weight" },
    .{ "class_embedding.linear_2.weight", "label_emb.0.2.weight" },
    .{ "add_embedding.linear_1.weight", "label_emb.0.0.weight" },
    .{ "add_embedding.linear_2.weight", "label_emb.0.2.weight" },
    .{ "conv_in.weight", "input_blocks.0.0.weight" },
    .{ "conv_norm_out.weight", "out.0.weight" },
    .{ "conv_out.weight", "out.2.weight" },
    .{ "time_embedding.linear_1.weight", "time_embed.0.weight" },
    .{ "time_embedding.linear_2.weight", "time_embed.2.weight" },
};

/// The `.weight` entries of `comfy.utils.unet_to_diffusers` for `cfg`. Its
/// per-level transformer depths come from `attn_levels`/`transformer_depth`, and
/// the middle block attends at the innermost level's depth.
pub fn unetToDiffusers(arena: std.mem.Allocator, cfg: sd_unet.Config) ![]Pair {
    var out: std.ArrayList(Pair) = .empty;
    const levels = cfg.levels();
    const nrb = cfg.layers_per_block;
    const P = struct {
        fn add(a: std.mem.Allocator, list: *std.ArrayList(Pair), comptime df: []const u8, dargs: anytype, comptime lf: []const u8, largs: anytype) !void {
            try list.append(a, .{ .diffusers = try a.print(df, dargs), .ldm = try a.print(lf, largs) });
        }
    };
    for (0..levels) |x| {
        const depth = if (cfg.attn_levels[x]) cfg.transformer_depth[x] else 0;
        var n: usize = 1 + (nrb + 1) * x;
        for (0..nrb) |i| {
            for (resnet_map) |rm| try P.add(arena, &out, "down_blocks.{d}.resnets.{d}.{s}", .{ x, i, rm[1] }, "input_blocks.{d}.0.{s}", .{ n, rm[0] });
            if (depth > 0) {
                for (attn_weights) |b| try P.add(arena, &out, "down_blocks.{d}.attentions.{d}.{s}", .{ x, i, b }, "input_blocks.{d}.1.{s}", .{ n, b });
                for (0..depth) |t| for (block_weights) |b| {
                    try P.add(arena, &out, "down_blocks.{d}.attentions.{d}.transformer_blocks.{d}.{s}", .{ x, i, t, b }, "input_blocks.{d}.1.transformer_blocks.{d}.{s}", .{ n, t, b });
                };
            }
            n += 1;
        }
        try P.add(arena, &out, "down_blocks.{d}.downsamplers.0.conv.weight", .{x}, "input_blocks.{d}.0.op.weight", .{n});
    }
    const mid_depth = cfg.transformer_depth[levels - 1];
    for (attn_weights) |b| try P.add(arena, &out, "mid_block.attentions.0.{s}", .{b}, "middle_block.1.{s}", .{b});
    for (0..mid_depth) |t| for (block_weights) |b| {
        try P.add(arena, &out, "mid_block.attentions.0.transformer_blocks.{d}.{s}", .{ t, b }, "middle_block.1.transformer_blocks.{d}.{s}", .{ t, b });
    };
    for ([_][2]usize{ .{ 0, 0 }, .{ 1, 2 } }) |in| for (resnet_map) |rm| {
        try P.add(arena, &out, "mid_block.resnets.{d}.{s}", .{ in[0], rm[1] }, "middle_block.{d}.{s}", .{ in[1], rm[0] });
    };
    for (0..levels) |x| {
        // Up block `x` is level `levels - 1 - x`: the reference pops its output
        // depths from the END of an outermost-first list.
        const lv = levels - 1 - x;
        const depth = if (cfg.attn_levels[lv]) cfg.transformer_depth[lv] else 0;
        var n: usize = (nrb + 1) * x;
        const l = nrb + 1;
        for (0..l) |i| {
            var c: usize = 1;
            for (resnet_map) |rm| try P.add(arena, &out, "up_blocks.{d}.resnets.{d}.{s}", .{ x, i, rm[1] }, "output_blocks.{d}.0.{s}", .{ n, rm[0] });
            if (depth > 0) {
                c += 1;
                for (attn_weights) |b| try P.add(arena, &out, "up_blocks.{d}.attentions.{d}.{s}", .{ x, i, b }, "output_blocks.{d}.1.{s}", .{ n, b });
                for (0..depth) |t| for (block_weights) |b| {
                    try P.add(arena, &out, "up_blocks.{d}.attentions.{d}.transformer_blocks.{d}.{s}", .{ x, i, t, b }, "output_blocks.{d}.1.transformer_blocks.{d}.{s}", .{ n, t, b });
                };
            }
            if (i == l - 1) try P.add(arena, &out, "up_blocks.{d}.upsamplers.0.conv.weight", .{x}, "output_blocks.{d}.{d}.conv.weight", .{ n, c });
            n += 1;
        }
    }
    for (basic_map) |bm| try out.append(arena, .{ .diffusers = bm[0], .ldm = bm[1] });
    return out.toOwnedSlice(arena);
}

/// `LORA_CLIP_MAP`: the HF suffix and the underscore spelling kohya keys use.
const clip_map = [_][2][]const u8{
    .{ "mlp.fc1", "mlp_fc1" },
    .{ "mlp.fc2", "mlp_fc2" },
    .{ "self_attn.k_proj", "self_attn_k_proj" },
    .{ "self_attn.q_proj", "self_attn_q_proj" },
    .{ "self_attn.v_proj", "self_attn_v_proj" },
    .{ "self_attn.out_proj", "self_attn_out_proj" },
};

fn underscored(arena: std.mem.Allocator, prefix: []const u8, name: []const u8) ![]u8 {
    const out = try arena.print("{s}{s}", .{ prefix, name });
    std.mem.replaceScalar(u8, out[prefix.len..], '.', '_');
    return out;
}

fn stripWeight(name: []const u8) ?[]const u8 {
    return if (std.mem.endsWith(u8, name, ".weight")) name[0 .. name.len - ".weight".len] else null;
}

/// CLIP-G's tensor for an HF suffix in the open_clip spelling an LDM SDXL
/// checkpoint stores it under: q/k/v are row blocks of the fused `in_proj_weight`,
/// in that order (how the reference splits it at load).
fn openClipTarget(arena: std.mem.Allocator, b: usize, hf: []const u8, hidden: usize) !Target {
    const qkv = [_][]const u8{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj" };
    for (qkv, 0..) |q, i| if (std.mem.eql(u8, hf, q)) return .{
        .comp = .clip_g,
        .name = try arena.print("transformer.resblocks.{d}.attn.in_proj_weight", .{b}),
        .row_off = i * hidden,
        .rows = hidden,
    };
    const rest = [_][2][]const u8{
        .{ "self_attn.out_proj", "attn.out_proj" },
        .{ "mlp.fc1", "mlp.c_fc" },
        .{ "mlp.fc2", "mlp.c_proj" },
    };
    for (rest) |r| if (std.mem.eql(u8, hf, r[0])) return .{
        .comp = .clip_g,
        .name = try arena.print("transformer.resblocks.{d}.{s}.weight", .{ b, r[1] }),
    };
    unreachable;
}

/// The reference's key map over this model: every spelling a LoRA may use for a
/// weight that exists here. A later spelling of the same stem replaces an earlier
/// one, as the reference's dict does.
pub fn keyMap(arena: std.mem.Allocator, m: Model) !KeyMap {
    var map: KeyMap = .empty;
    // UNet, LDM names: `lora_unet_<name with _>` and the generic `diffusion_model.<name>`.
    for (m.unet.names()) |name| {
        const base = stripWeight(name) orelse continue;
        const t: Target = .{ .comp = .unet, .name = name };
        try map.put(arena, try underscored(arena, "lora_unet_", base), t);
        try map.put(arena, try arena.print("diffusion_model.{s}", .{base}), t);
    }
    // UNet, diffusers names, only where the LDM tensor exists.
    for (try unetToDiffusers(arena, m.unet_cfg)) |p| {
        if (m.unet.get(p.ldm) == null) continue;
        const t: Target = .{ .comp = .unet, .name = p.ldm };
        const base = stripWeight(p.diffusers).?;
        try map.put(arena, try underscored(arena, "lora_unet_", base), t);
        try map.put(arena, try underscored(arena, "lycoris_", base), t);
        for ([_][]const u8{ "", "unet." }) |pfx| {
            const proc = try std.mem.replaceOwned(u8, arena, base, ".to_", ".processor.to_");
            const key = if (std.mem.endsWith(u8, proc, ".to_out.0")) proc[0 .. proc.len - 2] else proc;
            try map.put(arena, try arena.print("{s}{s}", .{ pfx, key }), t);
        }
    }
    // CLIP-L, always HF naming relative to `text_model.`. `lora_te_` reaches it
    // on SDXL too, as in the reference.
    if (m.clip_l) |st| for (0..m.clip_l_layers) |b| for (clip_map) |c| {
        const name = try arena.print("encoder.layers.{d}.{s}.weight", .{ b, c[0] });
        if (st.get(name) == null) continue;
        const t: Target = .{ .comp = .clip_l, .name = name };
        try map.put(arena, try arena.print("lora_te_text_model_encoder_layers_{d}_{s}", .{ b, c[1] }), t);
        try map.put(arena, try arena.print("lora_te1_text_model_encoder_layers_{d}_{s}", .{ b, c[1] }), t);
        try map.put(arena, try arena.print("text_encoder.text_model.encoder.layers.{d}.{s}", .{ b, c[0] }), t);
    };
    if (m.clip_g) |st| {
        const open_clip = st.get("transformer.resblocks.0.attn.in_proj_weight") != null;
        for (0..m.clip_g_layers) |b| for (clip_map) |c| {
            const t = if (open_clip) try openClipTarget(arena, b, c[0], m.clip_g_hidden) else Target{
                .comp = .clip_g,
                .name = try arena.print("encoder.layers.{d}.{s}.weight", .{ b, c[0] }),
            };
            if (st.get(t.name) == null) continue;
            try map.put(arena, try arena.print("lora_te2_text_model_encoder_layers_{d}_{s}", .{ b, c[1] }), t);
            try map.put(arena, try arena.print("text_encoder_2.text_model.encoder.layers.{d}.{s}", .{ b, c[0] }), t);
        };
    }
    return map;
}

/// The factor whose second dimension is a linear's input width, per spelling.
const input_factors = [_][]const u8{
    ".lora_down.weight", "_lora.down.weight",              ".lora_A.weight",         ".lora.down.weight",
    ".lora_A",           ".lora_linear_layer.down.weight", ".lora_A.default.weight", ".hada_w1_b",
    ".diff",
};

/// Whether a LoRA file is for this SD UNet, from its header alone. It must name
/// one of this layout's UNet weights in a spelling `keyMap` reads, every kohya
/// UNet stem it carries must exist in this layout, it carries no CLIP-G key
/// unless this is SDXL, and any cross-attention `to_k` it patches must take this
/// UNet's context width. SD1.5 and SDXL share most LDM names, so the names a file
/// carries that the other layout LACKS, and that width, are what place it; a LoKr
/// file hides the width, so for it only the names can.
pub fn sdFits(gpa: std.mem.Allocator, store: WeightStore, cfg: sd_unet.Config) bool {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    return sdFitsIn(arena.allocator(), store, cfg) catch false;
}

fn sdFitsIn(a: std.mem.Allocator, store: WeightStore, cfg: sd_unet.Config) !bool {
    const pairs = try unetToDiffusers(a, cfg);
    var kohya: std.StringHashMapUnmanaged(void) = .empty;
    for (pairs) |p| {
        try kohya.put(a, try underscored(a, "lora_unet_", stripWeight(p.ldm).?), {});
        try kohya.put(a, try underscored(a, "lora_unet_", stripWeight(p.diffusers).?), {});
    }
    const sdxl = cfg.context_dim == sd_unet.sdxl.context_dim;
    for (store.names()) |name| {
        // A kohya stem carries no dots, so the key up to its first is the stem.
        const stem = name[0 .. std.mem.findScalar(u8, name, '.') orelse name.len];
        if (std.mem.startsWith(u8, stem, "lora_unet_") and !kohya.contains(stem)) return false;
        if (!sdxl and std.mem.startsWith(u8, stem, "lora_te2_")) return false;
    }
    var named = false;
    var buf: [512]u8 = undefined;
    for (pairs) |p| {
        const to_k = std.mem.endsWith(u8, p.ldm, "attn2.to_k.weight");
        if (named and !to_k) continue;
        const ldm = stripWeight(p.ldm).?;
        const dif = stripWeight(p.diffusers).?;
        const proc = try std.mem.replaceOwned(u8, a, dif, ".to_", ".processor.to_");
        const stems = [_][]const u8{
            try underscored(a, "lora_unet_", ldm),
            try a.print("diffusion_model.{s}", .{ldm}),
            try underscored(a, "lora_unet_", dif),
            try underscored(a, "lycoris_", dif),
            proc,
            try a.print("unet.{s}", .{proc}),
        };
        for (stems) |stem| {
            const in_dim = stemInput(store, &buf, stem) orelse continue;
            named = true;
            if (to_k and in_dim != 0 and in_dim != cfg.context_dim) return false;
        }
    }
    return named;
}

/// Keys of an adapter form `sidecars` refuses, so the menu need not offer it.
const no_sidecar = [_][]const u8{ ".set_weight", ".lokr_t2" };

/// Whether `sidecars` can take every adapter in the file, from its key names: not
/// `no_sidecar`, and not DoRA over LoKr, which a file carrying both is taken to be.
pub fn sidecarTakes(store: WeightStore) bool {
    var lokr = false;
    var dora = false;
    for (store.names()) |n| {
        for (no_sidecar) |sfx| if (std.mem.endsWith(u8, n, sfx)) return false;
        if (std.mem.find(u8, n, ".lokr_") != null) lokr = true;
        if (std.mem.endsWith(u8, n, ".dora_scale")) dora = true;
    }
    return !(lokr and dora);
}

/// A stem's input width if the file patches it, 0 when its form hides the width
/// (LoKr), or null when the file does not patch it.
fn stemInput(store: WeightStore, buf: []u8, stem: []const u8) ?usize {
    for (input_factors) |sfx| {
        const v = store.get(std.mem.print(buf, "{s}{s}", .{ stem, sfx }) catch return null) orelse continue;
        const sh = v.info.shape.slice();
        return if (sh.len >= 2) sh[1] else 0;
    }
    for ([_][]const u8{ ".lokr_w1", ".lokr_w1_a", ".lokr_w2", ".lokr_w2_a" }) |sfx| {
        if (store.get(std.mem.print(buf, "{s}{s}", .{ stem, sfx }) catch return null) != null) return 0;
    }
    return null;
}

// --- the whole merge ---------------------------------------------------------

/// One LoRA file in a stack.
pub const File = struct {
    store: WeightStore,
    strength: f32,
    path: []const u8,
};

const Hit = struct { file: u32, target: Target, patch: Patch };

/// A merged stack: the substituted tensors and the overlays that serve them.
/// Every pointer in here must outlive the models loaded through `store`.
pub const Merged = struct {
    arena: std.heap.ArenaAllocator,
    overlays: [3]?*Overlay = .{ null, null, null },
    /// Stems applied per file, in file order.
    applied: []usize = &.{},
    tensors: usize = 0,
    bytes: usize = 0,

    pub fn deinit(self: *Merged, gpa: std.mem.Allocator) void {
        for (self.overlays) |o| if (o) |ov| {
            ov.deinit(gpa);
            gpa.destroy(ov);
        };
        self.arena.deinit();
        self.* = undefined;
    }

    /// The store a component loads from: its overlay, or `base` when nothing
    /// patched it.
    pub fn store(self: *const Merged, c: Component, base: WeightStore) WeightStore {
        return if (self.overlays[@backingInt(c)]) |ov| ov.store() else base;
    }
};

/// One merged tensor, written by its own task.
const Job = struct {
    base: TensorView,
    hits: []const Hit,
    mode: Mode,
    out: []u8,
    err: ?anyerror = null,
};

/// Merge `files`, in order, into the tensors they patch, one task per tensor. A
/// file that patches nothing is refused, since it is a LoRA for another
/// architecture; keys a file carries that nothing here reads are reported and
/// skipped, as the reference does.
///
/// The tasks' scratch comes from `std.heap.smp_allocator`, never `gpa`, which a
/// caller may hand over as an arena: an arena is not thread-safe.
pub fn merge(gpa: std.mem.Allocator, io: std.Io, m: Model, files: []const File) !Merged {
    var result: Merged = .{ .arena = .init(gpa) };
    errdefer result.deinit(gpa);
    const arena = result.arena.allocator();
    result.applied = try arena.alloc(usize, files.len);
    @memset(result.applied, 0);

    var scratch = std.heap.ArenaAllocator.init(gpa);
    defer scratch.deinit();
    const sa = scratch.allocator();
    const map = try keyMap(sa, m);

    // Per component, tensor name -> its hits in file order.
    var per: [3]std.array_hash_map.String(std.ArrayList(Hit)) = .{ .empty, .empty, .empty };
    for (files, 0..) |f, fi| {
        var consumed: Consumed = .empty;
        defer consumed.deinit(gpa);
        var it = map.iterator();
        while (it.next()) |e| {
            const p = parse(gpa, f.store, e.key_ptr.*, &consumed) catch |err| switch (err) {
                error.UnsupportedAdapter => {
                    std.log.warn("lora: {s}: {s} is an adapter form this merge does not apply; skipped", .{ f.path, e.key_ptr.* });
                    continue;
                },
                else => {
                    std.log.err("lora: {s}: {s} is malformed ({t})", .{ f.path, e.key_ptr.*, err });
                    return err;
                },
            } orelse continue;
            const gop = try per[@backingInt(e.value_ptr.comp)].getOrPut(sa, e.value_ptr.name);
            if (!gop.found_existing) gop.value_ptr.* = .empty;
            try gop.value_ptr.append(sa, .{ .file = @intCast(fi), .target = e.value_ptr.*, .patch = p });
            result.applied[fi] += 1;
        }
        var missed: usize = 0;
        for (f.store.names()) |n| {
            if (consumed.contains(n)) continue;
            if (missed < 8) std.log.warn("lora: {s}: key not loaded: {s}", .{ f.path, n });
            missed += 1;
        }
        if (result.applied[fi] == 0) {
            std.log.err("lora: {s} patches none of this checkpoint's weights", .{f.path});
            return error.NoMatchingTargets;
        }
        if (missed > 0) std.log.warn("lora: {s}: ONLY {d} of {d} keys loaded, {d} ignored", .{
            f.path, f.store.names().len - missed, f.store.names().len, missed,
        });
    }

    // The output buffers and the overlays are made here, single-threaded; the
    // tasks only fill bytes they were handed.
    var jobs: std.ArrayList(Job) = .empty;
    for (&per, 0..) |*comp_jobs, ci| {
        if (comp_jobs.count() == 0) continue;
        const base = m.store(@fromBackingInt(@intCast(ci))).?;
        const ov = try gpa.create(Overlay);
        ov.* = .{ .base = base };
        result.overlays[ci] = ov;
        var it = comp_jobs.iterator();
        while (it.next()) |e| {
            const v = base.get(e.key_ptr.*).?;
            const mode: Mode = switch (v.info.dtype) {
                .f32, .f64 => .f32,
                else => .f16,
            };
            const n = v.info.elemCount();
            const out = try arena.alloc(u8, if (mode == .f16) n * 2 else n * 4);
            try ov.put(gpa, try arena.dupe(u8, e.key_ptr.*), if (mode == .f16) .f16 else .f32, out);
            try jobs.append(sa, .{ .base = v, .hits = e.value_ptr.items, .mode = mode, .out = out });
            result.tensors += 1;
            result.bytes += out.len;
        }
    }

    var group: std.Io.Group = .init;
    defer group.cancel(io);
    for (jobs.items) |*j| group.async(io, runJob, .{ std.heap.smp_allocator, j, files });
    try group.await(io);
    for (jobs.items) |j| if (j.err) |err| {
        std.log.err("lora: cannot merge into {s} {any} ({t})", .{ j.base.info.name, j.base.info.shape.slice(), err });
        return err;
    };
    return result;
}

fn runJob(gpa: std.mem.Allocator, j: *Job, files: []const File) void {
    mergeTensor(gpa, j, files) catch |err| {
        j.err = err;
    };
}

fn mergeTensor(gpa: std.mem.Allocator, j: *Job, files: []const File) !void {
    const v = j.base;
    const full = v.info.shape.slice();
    const n_rows = full[0];
    const cols = v.info.elemCount() / n_rows;
    const es = v.info.dtype.storageBytes(cols);

    // DoRA's norms span the whole region, so a tensor carrying one merges whole.
    const whole = for (j.hits) |h| {
        if (h.patch.dora != null) break true;
    } else false;
    if (whole) return mergeWhole(gpa, j, files);

    var preps: [lora_mod.max_files * 3]Prepared = undefined;
    var n_prep: usize = 0;
    defer for (preps[0..n_prep]) |*p| p.deinit(gpa);
    for (j.hits) |h| {
        const region_rows = if (h.target.rows == 0) n_rows else h.target.rows;
        preps[n_prep] = try prepare(gpa, h.patch, region_rows, region_rows * cols, files[h.file].strength);
        n_prep += 1;
    }
    const buf = try gpa.alloc(f32, 3 * cols);
    defer gpa.free(buf);
    const w = buf[0..cols];
    const d = buf[cols .. 2 * cols];
    const tmp = buf[2 * cols ..];
    for (0..n_rows) |o| {
        try convertToF32(v.info.dtype, v.bytes[o * es ..][0..es], w);
        if (j.mode == .f16) for (w) |*x| {
            x.* = r16(x.*);
        };
        for (j.hits, preps[0..n_prep]) |h, *p| {
            const t = h.target;
            if (t.rows != 0 and (o < t.row_off or o >= t.row_off + t.rows)) continue;
            p.row(if (t.rows == 0) o else o - t.row_off, d, tmp);
            addRow(w, d, p, files[h.file].strength, j.mode);
        }
        writeOut(j, o * cols, w);
    }
}

fn mergeWhole(gpa: std.mem.Allocator, j: *Job, files: []const File) !void {
    const v = j.base;
    const full = v.info.shape.slice();
    const w = try v.toF32Alloc(gpa);
    defer gpa.free(w);
    if (j.mode == .f16) for (w) |*x| {
        x.* = r16(x.*);
    };
    const row_elems = v.info.elemCount() / full[0];
    for (j.hits) |h| {
        const t = h.target;
        var shape_buf: [4]usize = undefined;
        const region = if (t.rows == 0) w else w[t.row_off * row_elems ..][0 .. t.rows * row_elems];
        const shape = if (t.rows == 0) full else blk: {
            shape_buf[0] = t.rows;
            @memcpy(shape_buf[1..full.len], full[1..]);
            break :blk shape_buf[0..full.len];
        };
        try apply(gpa, region, shape, h.patch, files[h.file].strength, j.mode);
    }
    writeOut(j, 0, w);
}

/// Write merged values starting at element `at` into the job's output.
fn writeOut(j: *Job, at: usize, w: []const f32) void {
    switch (j.mode) {
        .f16 => tp_core.dtype.f32ToF16Row(w, j.out[at * 2 ..][0 .. w.len * 2]),
        .f32 => @memcpy(j.out[at * 4 ..][0 .. w.len * 4], std.mem.sliceAsBytes(w)),
    }
}

/// One file's sidecars for an SD model, one per component, keyed by the tensor
/// names each component's loader tags its weights with.
pub const SdSidecars = struct {
    sides: [3]?lora_mod.Sidecar = .{ null, null, null },
    /// Stems the file patches here.
    applied: usize = 0,

    pub fn deinit(self: *SdSidecars) void {
        for (&self.sides) |*s| if (s.*) |*x| x.deinit();
        self.* = undefined;
    }
};

/// Build one file's sidecars over `m`: the same key map and adapter reading as
/// `merge`, so a file reaches exactly what it would merge into. A file that
/// patches nothing is refused; keys nothing here reads are reported and skipped,
/// as the reference does. `set_weight`, Tucker LoKr and DoRA over LoKr refuse the
/// file by name.
pub fn sidecars(gpa: std.mem.Allocator, m: Model, store: WeightStore, path: []const u8) !SdSidecars {
    var out: SdSidecars = .{};
    errdefer out.deinit();
    var scratch = std.heap.ArenaAllocator.init(gpa);
    defer scratch.deinit();
    const sa = scratch.allocator();
    const map = try keyMap(sa, m);

    var arenas: [3]?std.heap.ArenaAllocator = .{ null, null, null };
    errdefer for (&arenas) |*ar| if (ar.*) |*x| x.deinit();
    const Terms = struct {
        factors: std.ArrayList(lora_mod.Factor) = .empty,
        kron: std.ArrayList(lora_mod.Kron) = .empty,
        diffs: std.ArrayList(lora_mod.Diff) = .empty,
        dora: ?lora_mod.Dora = null,
    };
    var targets: [3]std.array_hash_map.String(Terms) = .{ .empty, .empty, .empty };
    var keys: [3]std.StringHashMapUnmanaged(u8) = .{ .empty, .empty, .empty };

    var consumed: Consumed = .empty;
    defer consumed.deinit(gpa);
    var it = map.iterator();
    while (it.next()) |e| {
        const p = parse(gpa, store, e.key_ptr.*, &consumed) catch |err| switch (err) {
            error.UnsupportedAdapter => {
                std.log.err("lora: {s}: {s} is an adapter form the sidecar does not apply", .{ path, e.key_ptr.* });
                return error.UnsupportedLora;
            },
            else => return err,
        } orelse continue;
        const t = e.value_ptr.*;
        if (p.dora != null and t.rows != 0) {
            std.log.err("lora: {s}: {s} is DoRA on a row block of a fused tensor, whose norms span the whole tensor", .{ path, e.key_ptr.* });
            return error.UnsupportedLora;
        }
        const ci = @backingInt(t.comp);
        if (arenas[ci] == null) arenas[ci] = .init(gpa);
        const alloc = arenas[ci].?.allocator();
        const base = m.store(t.comp).?.get(t.name).?;
        const shape = base.info.shape.slice();
        const out_rows = if (t.rows == 0) shape[0] else t.rows;
        const db: ?adapter.Base = if (p.dora != null) try adapter.baseOf(gpa, base) else null;
        defer if (db) |b| gpa.free(b.w);
        const term = termFor(gpa, alloc, p, db, shape, out_rows, t.row_off) catch |err| {
            const what = if (err == error.UnsupportedLora) "DoRA over LoKr, which the sidecar does not apply" else "a shape this weight cannot take";
            std.log.err("lora: {s}: {s} is {s} ({t})", .{ path, e.key_ptr.*, what, err });
            return if (err == error.UnsupportedLora) error.UnsupportedLora else err;
        };
        const gop = try targets[ci].getOrPut(sa, t.name);
        if (!gop.found_existing) gop.value_ptr.* = .{};
        switch (term) {
            .factor => |f| {
                try gop.value_ptr.factors.append(sa, f.f);
                if (f.dora) |d| gop.value_ptr.dora = d;
            },
            .kron => |k| try gop.value_ptr.kron.append(sa, k),
            .diff => |d| try gop.value_ptr.diffs.append(sa, d),
        }
        const kg = try keys[ci].getOrPut(sa, t.name);
        kg.value_ptr.* = if (kg.found_existing) kg.value_ptr.* + 1 else 1;
        out.applied += 1;
    }

    var missed: usize = 0;
    for (store.names()) |n| {
        if (consumed.contains(n)) continue;
        if (missed < 8) std.log.warn("lora: {s}: key not loaded: {s}", .{ path, n });
        missed += 1;
    }
    if (out.applied == 0) {
        std.log.err("lora: {s} patches none of this checkpoint's weights", .{path});
        return error.NoMatchingTargets;
    }
    if (missed > 0) std.log.warn("lora: {s}: ONLY {d} of {d} keys loaded, {d} ignored", .{
        path, store.names().len - missed, store.names().len, missed,
    });

    for (&targets, 0..) |*tm, ci| {
        if (tm.count() == 0) continue;
        const alloc = arenas[ci].?.allocator();
        var index: std.StringHashMapUnmanaged(lora_mod.Target) = .empty;
        var tit = tm.iterator();
        while (tit.next()) |e| {
            const v = m.store(@fromBackingInt(@intCast(ci))).?.get(e.key_ptr.*).?;
            const shape = v.info.shape.slice();
            const name = try alloc.dupe(u8, e.key_ptr.*);
            try index.put(alloc, name, .{
                .factors = try alloc.dupe(lora_mod.Factor, e.value_ptr.factors.items),
                .kron = try alloc.dupe(lora_mod.Kron, e.value_ptr.kron.items),
                .diffs = try alloc.dupe(lora_mod.Diff, e.value_ptr.diffs.items),
                .dora = e.value_ptr.dora,
                .in_dim = v.info.elemCount() / shape[0],
                .out_dim = shape[0],
                .scale = 1,
                .tag = name,
                .keys = keys[ci].get(e.key_ptr.*).?,
            });
        }
        out.sides[ci] = .{ .arena = arenas[ci].?, .index = index, .unclaimed = 0 };
        arenas[ci] = null;
    }
    return out;
}

// --- tests -------------------------------------------------------------------

const testing = std.testing;
const fixture = @embedFile("assets/lora_merge.safetensors");
const diffusers_fixture = @embedFile("assets/lora_unet_diffusers.txt");

test "the diffusers map is the reference's, for both SD UNets" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]struct { []const u8, sd_unet.Config }{ .{ "sd15", sd_unet.sd15 }, .{ "sdxl", sd_unet.sdxl } }) |fam| {
        var want: std.StringHashMapUnmanaged([]const u8) = .empty;
        var lines = std.mem.tokenizeScalar(u8, diffusers_fixture, '\n');
        while (lines.next()) |line| {
            var f = std.mem.tokenizeScalar(u8, line, ' ');
            if (!std.mem.eql(u8, f.next().?, fam[0])) continue;
            const d = f.next().?;
            try want.put(a, d, f.next().?);
        }
        const got = try unetToDiffusers(a, fam[1]);
        // The reference's dict collapses the two `label_emb` spellings into four
        // distinct diffusers keys, as here; nothing else repeats.
        var seen: std.StringHashMapUnmanaged(void) = .empty;
        for (got) |p| {
            errdefer std.debug.print("{s}: {s} -> {s}\n", .{ fam[0], p.diffusers, p.ldm });
            try testing.expect(!(try seen.getOrPut(a, p.diffusers)).found_existing);
            try testing.expectEqualStrings(want.get(p.diffusers) orelse return error.TestUnexpectedResult, p.ldm);
        }
        errdefer std.debug.print("{s}: {d} names here, {d} in the reference\n", .{ fam[0], got.len, want.count() });
        try testing.expectEqual(want.count(), got.len);
    }
}

fn fixtureStore(st: *const tp_core.safetensors.SafeTensors) WeightStore {
    return .{ .safetensors = st };
}

test "a CLIP-G row block and the LDM and diffusers spellings all reach their tensors" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // A UNet store with the names the map needs, shapes irrelevant.
    var hdr: std.ArrayList(u8) = .empty;
    const names = [_][]const u8{
        "input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight",
        "output_blocks.2.2.conv.weight",
        "middle_block.1.transformer_blocks.0.ff.net.2.weight",
    };
    try hdr.append(a, '{');
    for (names, 0..) |n, i| {
        if (i > 0) try hdr.append(a, ',');
        try hdr.print(a, "\"{s}\":{{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[{d},{d}]}}", .{ n, i * 4, i * 4 + 4 });
    }
    try hdr.append(a, '}');
    const file = try a.alloc(u8, 8 + hdr.items.len + names.len * 4);
    std.mem.writeInt(u64, file[0..8], hdr.items.len, .little);
    @memcpy(file[8..][0..hdr.items.len], hdr.items);
    @memset(file[8 + hdr.items.len ..], 0);
    var st = try tp_core.safetensors.SafeTensors.initFromSlice(a, file);
    defer st.deinit();

    const g_hdr = "{\"transformer.resblocks.3.attn.in_proj_weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},\"transformer.resblocks.0.attn.in_proj_weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[4,8]}}";
    const gfile = try a.alloc(u8, 8 + g_hdr.len + 8);
    std.mem.writeInt(u64, gfile[0..8], g_hdr.len, .little);
    @memcpy(gfile[8..][0..g_hdr.len], g_hdr);
    @memset(gfile[8 + g_hdr.len ..], 0);
    var gst = try tp_core.safetensors.SafeTensors.initFromSlice(a, gfile);
    defer gst.deinit();

    const map = try keyMap(a, .{ .unet = .{ .safetensors = &st }, .unet_cfg = sd_unet.sdxl, .clip_g = .{ .safetensors = &gst }, .clip_g_layers = 32, .clip_g_hidden = 1280 });
    const want = [_]struct { []const u8, []const u8, usize }{
        .{ "lora_unet_input_blocks_4_1_transformer_blocks_0_attn2_to_k", names[0], 0 },
        .{ "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_k", names[0], 0 },
        .{ "unet.down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor.to_k", names[0], 0 },
        .{ "lora_unet_up_blocks_0_upsamplers_0_conv", names[1], 0 },
        .{ "lora_unet_mid_block_attentions_0_transformer_blocks_0_ff_net_2", names[2], 0 },
        .{ "lora_te2_text_model_encoder_layers_3_self_attn_v_proj", "transformer.resblocks.3.attn.in_proj_weight", 2 * 1280 },
    };
    for (want) |w| {
        errdefer std.debug.print("{s}\n", .{w[0]});
        const t = map.get(w[0]) orelse return error.TestUnexpectedResult;
        try testing.expectEqualStrings(w[1], t.name);
        try testing.expectEqual(w[2], t.row_off);
    }
    // No q/k/v of a layer the store lacks.
    try testing.expect(map.get("lora_te2_text_model_encoder_layers_5_self_attn_q_proj") == null);
}

