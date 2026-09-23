#!/usr/bin/env python3
"""Trim the upstream Noto files down to what dvui can draw.

dvui renders one codepoint at a time through FT_Load_Char and reads no GSUB or
GPOS, so a glyph that only a shaper could select (vertical alternates, locale
forms, ligatures, emoji sequences) is dead weight, as are the layout and
vertical-metrics tables and a variable font's deltas. The output keeps hints,
every cmap-reachable glyph and every codepoint, so it renders identically
through dvui's FreeType. A FreeType built with HarfBuzz is different: its
autohinter reads GSUB to group glyphs, so a few emoji hint differently there.

Only the Regular CJK weight ships: a bold face is another 10 MB for a weight
change, so bold CJK runs render regular (fonts.zig routes them that way).

Inputs (drop them in --src):
  NotoSansCJK-Regular.ttc
      https://github.com/notofonts/noto-cjk/releases  (Sans, "TTC" language-specific OTC)
  NotoEmoji.ttf   (variable)
      https://github.com/googlefonts/noto-emoji/tree/main/fonts  NotoEmoji[wght].ttf

--pack zstd writes each face as a zstd frame instead (0.77 of raw for these
hinted CFF faces, 0.64 for the emoji TrueType). Off by default: std's decoder
inflates at ~22 MB/s, a second of startup in ReleaseFast and eight in Debug
for 4.5 MB, and fonts.zig embeds the raw files. Only a libzstd link makes it
free.

  pip install fonttools zstandard
  tools/gen_fonts.py --src <dir with the three files> --out src/gui/fonts
"""
import argparse, hashlib, os, sys
from fontTools.ttLib import TTCollection, TTFont
from fontTools import subset
from fontTools.varLib.instancer import instantiateVariableFont, OverlapMode

# The files the shipped fonts were cut from. A different hash is a warning, not
# an error: Noto updates are fine, this is so a diff in the output has a cause.
KNOWN = {
    'NotoSansCJK-Regular.ttc': 'b76b0433203017ca80401b2ee0dd69350349871c4b19d504c34dbdd80541690a',
    'NotoEmoji.ttf': 'de6c18832938afc99caf132b39d6a30a19bac7f2e812e28db2535b4608d27551',
}
PACK = False
DROP = ['GSUB', 'GPOS', 'BASE', 'vmtx', 'vhea', 'VORG', 'MATH']
# Codepoints fonts.zig routes to each face; the output must still have them.
MUST_CJK = [0x65E5, 0xD55C, 0x2318, 0x2500, 0x25AA, 0x20BB7]  # 日 한 ⌘ ─ ▪ 𠮷
MUST_EMOJI = [0x1F98A, 0x23F8, 0xFE0F, 0x200D]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def check_input(path):
    name = os.path.basename(path)
    got = sha256(path)
    if KNOWN.get(name) != got:
        print(f'warning: {name} is not the file the shipped fonts came from ({got[:12]})', file=sys.stderr)


def trim(font, must):
    """Keep only cmap-reachable glyphs and drop the tables nothing reads."""
    cmap = font.getBestCmap()
    missing = [hex(c) for c in must if c not in cmap]
    assert not missing, f'input lacks {missing}'
    opts = subset.Options()
    opts.drop_tables += DROP
    opts.notdef_outline = True  # a miss must stay a visible tofu box
    opts.hinting = True
    opts.name_IDs = ['*']
    opts.layout_features = []
    ss = subset.Subsetter(opts)
    ss.populate(unicodes=list(cmap))
    ss.subset(font)
    return set(cmap)


def reachable(f):
    """Glyphs FT_Load_Char can reach: the cmap targets plus, in a TrueType
    font, the components those composites are built from."""
    seen = {'.notdef'}
    todo = list(set(f.getBestCmap().values()))
    glyf = f['glyf'] if 'glyf' in f else None
    while todo:
        g = todo.pop()
        if g in seen:
            continue
        seen.add(g)
        if glyf is not None and glyf[g].isComposite():
            todo += [c.glyphName for c in glyf[g].components]
    return seen


def pack(path):
    """Replace `path` with `path`.zst and return the compressed path."""
    import zstandard
    raw = open(path, 'rb').read()
    z = zstandard.ZstdCompressor(level=19, write_content_size=True, write_checksum=False).compress(raw)
    assert zstandard.ZstdDecompressor().decompress(z) == raw
    assert zstandard.get_frame_parameters(z).content_size == len(raw)
    dst = path + '.zst'
    with open(dst, 'wb') as f:
        f.write(z)
    os.remove(path)
    print(f'  {os.path.basename(dst):28s} {len(z) // 1024:6d} KB  ({len(z) / len(raw):.2f} of raw)')
    return dst


def verify(path, before, must):
    f = TTFont(path)
    cmap = f.getBestCmap()
    assert set(cmap) == before, 'subset lost codepoints'
    assert set(f.getGlyphOrder()) == reachable(f), 'unreachable glyphs survived'
    for t in DROP:
        assert t not in f, f'{t} survived'
    assert 'fvar' not in f and 'gvar' not in f, 'still variable'
    assert not f.reader.file.name.endswith('.ttc')
    for c in must:
        assert c in cmap, hex(c)
    kinds = {(t.platformID, t.platEncID, t.format) for t in f['cmap'].tables}
    assert (3, 1, 4) in kinds and (3, 10, 12) in kinds, kinds  # what fonts.zig parses, plus astral
    print(f'  {os.path.basename(path):28s} {os.path.getsize(path) // 1024:6d} KB  {f["maxp"].numGlyphs} glyphs  {len(cmap)} codepoints')


def cjk(src, out):
    inp = os.path.join(src, 'NotoSansCJK-Regular.ttc')
    check_input(inp)
    font = TTCollection(inp).fonts[0]  # face 0 is the JP Sans face, the one dvui opens
    assert font['name'].getDebugName(4) == 'Noto Sans CJK JP'
    before = trim(font, MUST_CJK)
    dst = os.path.join(out, 'NotoSansCJKjp-Regular.otf')
    font.save(dst)
    verify(dst, before, MUST_CJK)
    if PACK:
        pack(dst)


def emoji(src, out):
    inp = os.path.join(src, 'NotoEmoji.ttf')
    check_input(inp)
    font = TTFont(inp)
    axes = {a.axisTag: a.defaultValue for a in font['fvar'].axes}
    # The default instance is what FreeType drew anyway. The instancer's default
    # sets OVERLAP_SIMPLE on every glyph, which sends FreeType's rasterizer down
    # its oversampling path and changes every bitmap; leave the flags as found.
    font = instantiateVariableFont(font, axes, overlap=OverlapMode.KEEP_AND_DONT_SET_FLAGS)
    before = trim(font, MUST_EMOJI)
    dst = os.path.join(out, 'NotoEmoji-Regular.ttf')
    font.save(dst)
    verify(dst, before, MUST_EMOJI)
    if PACK:
        pack(dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--out', default='src/gui/fonts')
    ap.add_argument('--pack', choices=['none', 'zstd'], default='none')
    a = ap.parse_args()
    global PACK
    PACK = a.pack == 'zstd'
    cjk(a.src, a.out)
    emoji(a.src, a.out)


if __name__ == '__main__':
    main()
