//! MiniMax H3's prompt PRESENTATION: reference labels and spliced vision blocks.
//!
//! H3's conditioning is not chat-templated. It is raw prompt text with per-
//! reference labels and vision blocks spliced in, and the label ORDER and
//! ORDINALS are part of what the model was trained on:
//!
//!     t2va   <prompt>
//!     fl2va  "<Picture 1>: " <block> ["<Picture 2>: " <block>] <prompt>
//!     ref2va per reference in request order, 1-based ordinals PER TYPE:
//!              image -> "<Picture i>: " <block>
//!              audio -> "<Audio j>: "                    (audio never enters the ViT)
//!              video -> "<Video k>: " then per 2-frame temporal block
//!                       "<T.T seconds>" <block>
//!            then <prompt>
//!
//! ⚠️ **The ordinals count per TYPE, not globally**, so an image followed by an
//! audio followed by an image is `<Picture 1>`, `<Audio 1>`, `<Picture 2>`. A
//! global counter gives `<Picture 1>`, `<Audio 2>`, `<Picture 3>`, which is
//! fluent, plausible and refers to things the prompt never mentions.
//!
//! ⚠️ **An audio reference gets a LABEL and no block.** It never reaches the
//! vision tower; its rows enter the DiT through the packed layout instead. Giving
//! it a block would put audio-shaped nothing into the LLM.
//!
//! ⚠️ **A reference video's soundtrack label is emitted BEFORE its video label**
//! (`<Audio j>` then `<Video k>`), because the DiT packs the soundtrack's rows
//! immediately before the video's. The two orders have to agree.
//!
//! What this module produces is one source for the four things that must line up,
//! since deriving them separately is how they drift:
//!
//!   - the token ids;
//!   - the modality TAG spans (widened by one on each side, so the flanking
//!     markers are tagged with the image);
//!   - the DEEPSTACK injection spans (NOT widened);
//!   - the mrope `ImageSpan`s (index, size and the pre-merge grid).

const std = @import("std");
const tp_core = @import("tp_core");
const tokenizer_mod = tp_core.tokenizer;
const h3 = @import("minimax_h3.zig");
const vit = @import("minimax_h3_vit.zig");

const Tokenizer = tokenizer_mod.Tokenizer;

/// `<|vision_start|>` / `<|vision_end|>`. The flanking markers a vision block
/// sits between, and the reason the tag span is two rows wider than the block.
pub const vision_start: u32 = 151652;
pub const vision_end: u32 = 151653;

/// One reference, in request order. Grids are PRE-merge patch grids, i.e. what
/// the ViT's resize produced; a block occupies `(gh/2) * (gw/2)` rows.
pub const Item = union(enum) {
    image: Grid,
    /// A standalone audio reference, or a video's soundtrack. Label only.
    audio,
    /// A reference video: one vision block per temporal pair, each labelled with
    /// its midpoint timestamp. `soundtrack` emits an `<Audio j>` label first.
    video: struct {
        grid: Grid,
        /// Temporal blocks, i.e. `ceil(sampled_frames / 2)`.
        blocks: usize,
        /// Midpoint seconds per block, `blocks` long.
        timestamps: []const f32,
        soundtrack: bool = false,
    },

    pub const Grid = struct {
        h: usize,
        w: usize,

        /// Rows one block of this grid occupies.
        pub fn tokens(g: Grid) usize {
            return (g.h / 2) * (g.w / 2);
        }
    };
};

/// The assembled prompt. `ids` is what the encoder embeds; everything else
/// describes where the vision blocks landed.
pub const Presentation = struct {
    arena: std.heap.ArenaAllocator,
    ids: []u32,
    /// One per spliced vision block, in emission order. Both span kinds and the
    /// mrope spans derive from these.
    blocks: []h3.VisionBlock,
    /// Pre-merge grid per block, parallel to `blocks`.
    grids: []Item.Grid,

    pub fn deinit(self: *Presentation) void {
        self.arena.deinit();
        self.* = undefined;
    }

    pub fn seq(self: *const Presentation) usize {
        return self.ids.len;
    }

    /// Modality tag spans for `PackedLayout.build`, widened by one on each side.
    pub fn tagSpans(self: *const Presentation, gpa: std.mem.Allocator) ![]h3.VisionSpan {
        const out = try gpa.alloc(h3.VisionSpan, self.blocks.len);
        for (out, self.blocks) |*o, b| o.* = b.tagSpan();
        return out;
    }

    /// DeepStack injection spans, NOT widened.
    pub fn injectSpans(self: *const Presentation, gpa: std.mem.Allocator, comptime Span: type) ![]Span {
        const out = try gpa.alloc(Span, self.blocks.len);
        for (out, self.blocks) |*o, b| {
            const s = b.injectSpan();
            o.* = .{ .start = s.start, .len = s.len };
        }
        return out;
    }

    /// mrope spans: the block plus the grid its axes come from.
    pub fn imageSpans(self: *const Presentation, gpa: std.mem.Allocator) ![]vit.ImageSpan {
        const out = try gpa.alloc(vit.ImageSpan, self.blocks.len);
        for (out, self.blocks, self.grids) |*o, b, g| o.* = .{
            .index = b.index,
            .size = b.size,
            .grid_h = g.h,
            .grid_w = g.w,
        };
        return out;
    }
};

/// Build the presentation: labels, vision blocks, then the prompt.
///
/// `items` is in REQUEST order and the ordinals are counted per type as it walks.
pub fn build(
    gpa: std.mem.Allocator,
    tok: *const Tokenizer,
    text: []const u8,
    items: []const Item,
) !Presentation {
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const alloc = arena.allocator();

    var ids: std.ArrayList(u32) = .empty;
    var blocks: std.ArrayList(h3.VisionBlock) = .empty;
    var grids: std.ArrayList(Item.Grid) = .empty;

    var n_image: usize = 0;
    var n_audio: usize = 0;
    var n_video: usize = 0;
    var buf: [64]u8 = undefined;

    const addText = struct {
        fn go(a: std.mem.Allocator, t: *const Tokenizer, list: *std.ArrayList(u32), s: []const u8) !void {
            if (s.len == 0) return;
            try t.encode(a, s, list);
        }
    }.go;

    for (items) |item| {
        switch (item) {
            .image => |g| {
                n_image += 1;
                try addText(alloc, tok, &ids, try std.fmt.bufPrint(&buf, "<Picture {d}>: ", .{n_image}));
                try addBlock(alloc, &ids, &blocks, &grids, g);
            },
            .audio => {
                // Label only: audio never enters the vision tower.
                n_audio += 1;
                try addText(alloc, tok, &ids, try std.fmt.bufPrint(&buf, "<Audio {d}>: ", .{n_audio}));
            },
            .video => |v| {
                if (v.timestamps.len != v.blocks) return error.BadPresentation;
                // The soundtrack's label comes FIRST, matching the order the DiT
                // packs its rows in.
                if (v.soundtrack) {
                    n_audio += 1;
                    try addText(alloc, tok, &ids, try std.fmt.bufPrint(&buf, "<Audio {d}>: ", .{n_audio}));
                }
                n_video += 1;
                try addText(alloc, tok, &ids, try std.fmt.bufPrint(&buf, "<Video {d}>: ", .{n_video}));
                for (v.timestamps) |ts| {
                    // One decimal, matching the reference's "%.1f seconds".
                    try addText(alloc, tok, &ids, try std.fmt.bufPrint(&buf, "<{d:.1} seconds>", .{ts}));
                    try addBlock(alloc, &ids, &blocks, &grids, v.grid);
                }
            },
        }
    }

    try addText(alloc, tok, &ids, text);
    // An empty presentation still needs a row: the reference emits one pad token
    // rather than an empty sequence.
    if (ids.items.len == 0) try ids.append(alloc, tokenizer_mod.pad_token);

    return .{
        .arena = arena,
        .ids = try ids.toOwnedSlice(alloc),
        .blocks = try blocks.toOwnedSlice(alloc),
        .grids = try grids.toOwnedSlice(alloc),
    };
}

/// `<|vision_start|>` + `n` placeholder rows + `<|vision_end|>`.
///
/// The placeholders are real pad ids rather than a sentinel, because the encoder
/// embeds every id BEFORE pasting the vision rows over them: an out-of-vocabulary
/// marker would fail the embedding lookup instead of being replaced.
fn addBlock(
    alloc: std.mem.Allocator,
    ids: *std.ArrayList(u32),
    blocks: *std.ArrayList(h3.VisionBlock),
    grids: *std.ArrayList(Item.Grid),
    g: Item.Grid,
) !void {
    if (g.h % 2 != 0 or g.w % 2 != 0 or g.h == 0 or g.w == 0) return error.BadPresentation;
    const n = g.tokens();
    try ids.append(alloc, vision_start);
    const index = ids.items.len;
    try ids.appendNTimes(alloc, tokenizer_mod.pad_token, n);
    try ids.append(alloc, vision_end);
    try blocks.append(alloc, .{ .index = index, .size = n });
    try grids.append(alloc, g);
}

// --- tests ----------------------------------------------------------------

const testing = std.testing;

test "reference ordinals count per type, not globally" {
    // `<Picture 1>`, `<Audio 1>`, `<Picture 2>` -- not 1, 2, 3. A global counter
    // produces a fluent prompt referring to things it never labelled.
    const gpa = testing.allocator;
    var tok = try Tokenizer.init(gpa);
    defer tok.deinit();

    var p = try build(gpa, &tok, "a prompt", &.{
        .{ .image = .{ .h = 4, .w = 6 } },
        .audio,
        .{ .image = .{ .h = 2, .w = 2 } },
    });
    defer p.deinit();

    // Round-trip the ids back to text to read the labels the model will see.
    const text = try tok.decodeAlloc(gpa, p.ids);
    defer gpa.free(text);
    errdefer std.debug.print("presented: {s}\n", .{text});
    try testing.expect(std.mem.indexOf(u8, text, "<Picture 1>: ") != null);
    try testing.expect(std.mem.indexOf(u8, text, "<Audio 1>: ") != null);
    try testing.expect(std.mem.indexOf(u8, text, "<Picture 2>: ") != null);
    try testing.expect(std.mem.indexOf(u8, text, "<Audio 2>") == null);
    // ...and the ordering is request order, so Audio 1 sits between the pictures.
    const pic1 = std.mem.indexOf(u8, text, "<Picture 1>").?;
    const aud1 = std.mem.indexOf(u8, text, "<Audio 1>").?;
    const pic2 = std.mem.indexOf(u8, text, "<Picture 2>").?;
    try testing.expect(pic1 < aud1 and aud1 < pic2);
    // The prompt comes last.
    try testing.expect(std.mem.indexOf(u8, text, "a prompt").? > pic2);

    // Two blocks, one per IMAGE: the audio reference contributes a label only.
    try testing.expectEqual(@as(usize, 2), p.blocks.len);
    try testing.expectEqual(@as(usize, 6), p.blocks[0].size); // 4/2 * 6/2
    try testing.expectEqual(@as(usize, 1), p.blocks[1].size); // 2/2 * 2/2
}

test "a reference video's soundtrack is labelled before the video itself" {
    // The `<Audio j>` label comes FIRST, matching the order the DiT packs the
    // block's two row streams in -- and the ordinals still count per type, so a
    // standalone soundtrack after the video is `<Audio 2>`.
    const gpa = testing.allocator;
    var tok = try Tokenizer.init(gpa);
    defer tok.deinit();

    const ts = [_]f32{ 0.25, 0.75 };
    var p = try build(gpa, &tok, "prompt", &.{
        .{ .video = .{ .grid = .{ .h = 4, .w = 4 }, .blocks = 2, .timestamps = &ts, .soundtrack = true } },
        .audio,
    });
    defer p.deinit();

    const text = try tok.decodeAlloc(gpa, p.ids);
    defer gpa.free(text);
    errdefer std.debug.print("presented: {s}\n", .{text});
    const aud1 = std.mem.indexOf(u8, text, "<Audio 1>: ").?;
    const vid1 = std.mem.indexOf(u8, text, "<Video 1>: ").?;
    const aud2 = std.mem.indexOf(u8, text, "<Audio 2>: ").?;
    try testing.expect(aud1 < vid1);
    try testing.expect(vid1 < aud2);
    try testing.expect(std.mem.indexOf(u8, text, "<Video 2>") == null);
    // Two blocks, one per temporal pair; neither audio label contributes one.
    try testing.expectEqual(@as(usize, 2), p.blocks.len);
}

test "a vision block sits between its markers, and the spans reflect that" {
    // The block's rows are flanked by `<|vision_start|>` / `<|vision_end|>`, the
    // tag span covers both markers and the injection span covers neither. This is
    // the one place the two are derived, so it is the one place to check them
    // against the actual id stream.
    const gpa = testing.allocator;
    var tok = try Tokenizer.init(gpa);
    defer tok.deinit();

    var p = try build(gpa, &tok, "x", &.{.{ .image = .{ .h = 4, .w = 4 } }});
    defer p.deinit();
    try testing.expectEqual(@as(usize, 1), p.blocks.len);
    const b = p.blocks[0];
    try testing.expectEqual(@as(usize, 4), b.size);

    // The id just before the block is the start marker, just after is the end.
    try testing.expectEqual(vision_start, p.ids[b.index - 1]);
    try testing.expectEqual(vision_end, p.ids[b.index + b.size]);
    // The rows under the block are embeddable placeholders, not a sentinel.
    for (0..b.size) |i| try testing.expectEqual(tokenizer_mod.pad_token, p.ids[b.index + i]);

    // The tag span covers both markers; the injection span covers only the rows.
    const tags = try p.tagSpans(gpa);
    defer gpa.free(tags);
    try testing.expectEqual(b.index - 1, tags[0].start);
    try testing.expectEqual(b.index + b.size + 1, tags[0].stop());
    try testing.expectEqual(vision_start, p.ids[tags[0].start]);
    try testing.expectEqual(vision_end, p.ids[tags[0].stop() - 1]);

    const spans = try p.imageSpans(gpa);
    defer gpa.free(spans);
    try testing.expectEqual(b.index, spans[0].index);
    try testing.expectEqual(@as(usize, 4), spans[0].grid_h);
    try testing.expectEqual(@as(usize, 4), spans[0].grid_w);
    try testing.expectEqual(b.size, spans[0].size);
}

test "a reference video labels its soundtrack before itself, one block per pair" {
    // The DiT packs a reference video's soundtrack rows immediately BEFORE its
    // video rows, so the labels have to come out in that order too. And each
    // temporal pair gets its own timestamped block, which is what makes a video
    // several vision blocks rather than one.
    const gpa = testing.allocator;
    var tok = try Tokenizer.init(gpa);
    defer tok.deinit();

    const ts = [_]f32{ 0.0, 1.0, 2.0 };
    var p = try build(gpa, &tok, "p", &.{.{ .video = .{
        .grid = .{ .h = 4, .w = 4 },
        .blocks = 3,
        .timestamps = &ts,
        .soundtrack = true,
    } }});
    defer p.deinit();

    const text = try tok.decodeAlloc(gpa, p.ids);
    defer gpa.free(text);
    errdefer std.debug.print("presented: {s}\n", .{text});
    const aud1 = std.mem.indexOf(u8, text, "<Audio 1>").?;
    const vid1 = std.mem.indexOf(u8, text, "<Video 1>").?;
    try testing.expect(aud1 < vid1);
    try testing.expect(std.mem.indexOf(u8, text, "<0.0 seconds>") != null);
    try testing.expect(std.mem.indexOf(u8, text, "<2.0 seconds>") != null);

    // Three blocks, all sharing the video's grid, ascending and non-overlapping.
    try testing.expectEqual(@as(usize, 3), p.blocks.len);
    for (p.blocks, p.grids) |b, g| {
        try testing.expectEqual(@as(usize, 4), b.size);
        try testing.expectEqual(@as(usize, 4), g.h);
    }
    for (0..p.blocks.len - 1) |i| {
        try testing.expect(p.blocks[i].index + p.blocks[i].size < p.blocks[i + 1].index);
    }

    // A timestamp count that disagrees with the block count is refused rather
    // than emitting fewer blocks than the payload will carry.
    try testing.expectError(error.BadPresentation, build(gpa, &tok, "p", &.{.{ .video = .{
        .grid = .{ .h = 4, .w = 4 },
        .blocks = 3,
        .timestamps = ts[0..2],
    } }}));
}

test "a text-only presentation is exactly the prompt" {
    // t2va must be unchanged by all of this: no labels, no blocks, and the same
    // ids the plain tokenizer gives.
    const gpa = testing.allocator;
    var tok = try Tokenizer.init(gpa);
    defer tok.deinit();

    var p = try build(gpa, &tok, "a red fox", &.{});
    defer p.deinit();
    try testing.expectEqual(@as(usize, 0), p.blocks.len);

    var plain: std.ArrayList(u32) = .empty;
    defer plain.deinit(gpa);
    try tok.encode(gpa, "a red fox", &plain);
    try testing.expectEqualSlices(u32, plain.items, p.ids);

    // ...and an empty prompt is one pad token, not an empty sequence.
    var empty = try build(gpa, &tok, "", &.{});
    defer empty.deinit();
    try testing.expectEqualSlices(u32, &.{tokenizer_mod.pad_token}, empty.ids);
}
