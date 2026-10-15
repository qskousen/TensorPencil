"""Truncate a kohya-spelled LoRA to its top-R singular directions, and write the
same factors zero-padded to a wider rank as a second file.

    lora_rank_truncate.py <src.safetensors> <out_rank_r> <out_padded> --rank 32 --pad-to 128

Per target, the applied delta is `alpha / r0 * up @ down`. Its SVD comes from two
QRs and a small SVD, so no `[out, in]` product is ever formed. The rank-R file keeps
the source's alpha, so the loader's `alpha / A.rows` divides by R and the factors
absorb `R / r0`. The padded file carries the SAME bf16 factor bytes with zero rows
of `down` and zero columns of `up` and alpha scaled by `pad_to / R`, so both files
apply the same delta and a renderer must produce the same image from either.

Output is streamed one tensor at a time; memory stays at one target's factors.
"""

import argparse
import json
import struct

import torch
from safetensors import safe_open

DOWN = ".lora_down.weight"
UP = ".lora_up.weight"
ALPHA = ".alpha"


def truncate(down, up, alpha, rank):
    r0 = down.shape[0]
    a = down.double()
    b = up.double()
    qb, rb = torch.linalg.qr(b)          # b = qb rb
    qa, ra = torch.linalg.qr(a.T)        # a = ra^T qa^T
    u, s, vh = torch.linalg.svd(rb @ ra.T)
    # up' @ down' must equal (R / r0) * top_R(up @ down), split evenly.
    s = s[:rank] * (rank / r0)
    root = s.sqrt()
    up_r = (qb @ u[:, :rank]) * root
    down_r = root[:, None] * (vh[:rank] @ qa.T)
    return down_r.to(down.dtype).contiguous(), up_r.to(up.dtype).contiguous()


def header_and_order(src, names, rank, pad_to):
    """Tensor specs for both outputs, in one fixed order."""
    specs = {"r": [], "p": []}
    for stem in names:
        for kind in (DOWN, UP, ALPHA):
            key = stem + kind
            sl = src.get_slice(key)
            shape = list(sl.get_shape())
            dt = sl.get_dtype()
            if kind == DOWN:
                sr, sp = [rank, shape[1]], [pad_to, shape[1]]
            elif kind == UP:
                sr, sp = [shape[0], rank], [shape[0], pad_to]
            else:
                sr = sp = shape
            specs["r"].append((key, dt, sr))
            specs["p"].append((key, dt, sp))
    return specs


def elem_bytes(dt):
    return {"BF16": 2, "F16": 2, "F32": 4}[dt]


def open_out(path, specs):
    hdr = {}
    off = 0
    for key, dt, shape in specs:
        n = elem_bytes(dt)
        for d in shape:
            n *= d
        hdr[key] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + n]}
        off += n
    blob = json.dumps(hdr, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    f = open(path, "wb")
    f.write(struct.pack("<Q", len(blob)))
    f.write(blob)
    return f


def raw(t):
    return t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out_rank")
    ap.add_argument("out_padded")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--pad-to", type=int, default=128)
    args = ap.parse_args()
    assert args.pad_to > args.rank

    with safe_open(args.src, "pt") as src:
        keys = list(src.keys())
        names = sorted(k[: -len(DOWN)] for k in keys if k.endswith(DOWN))
        assert len(keys) == 3 * len(names), "only down/up/alpha triples are handled"
        specs = header_and_order(src, names, args.rank, args.pad_to)
        fr = open_out(args.out_rank, specs["r"])
        fp = open_out(args.out_padded, specs["p"])
        kept = []
        for stem in names:
            down = src.get_tensor(stem + DOWN)
            up = src.get_tensor(stem + UP)
            alpha = src.get_tensor(stem + ALPHA)
            assert down.shape[0] > args.rank
            d_r, u_r = truncate(down, up, float(alpha), args.rank)
            d_p = torch.zeros((args.pad_to, d_r.shape[1]), dtype=d_r.dtype)
            d_p[: args.rank] = d_r
            u_p = torch.zeros((u_r.shape[0], args.pad_to), dtype=u_r.dtype)
            u_p[:, : args.rank] = u_r
            # A power of two, so exact in bf16.
            a_p = (alpha.double() * (args.pad_to / args.rank)).to(alpha.dtype)
            assert float(a_p) / args.pad_to == float(alpha) / args.rank
            for f, d, u, a in ((fr, d_r, u_r, alpha), (fp, d_p, u_p, a_p)):
                f.write(raw(d))
                f.write(raw(u))
                f.write(raw(a))

            # How much of the delta the top R directions keep, for the record.
            s = torch.linalg.svdvals(torch.linalg.qr(up.double())[1] @ torch.linalg.qr(down.double().T)[1].T)
            kept.append(float((s[: args.rank] ** 2).sum() / (s**2).sum()))
        fr.close()
        fp.close()
    print(f"{len(names)} targets, rank {args.rank} (padded copy at {args.pad_to}); energy kept min {min(kept):.3f} mean {sum(kept) / len(kept):.3f}")


if __name__ == "__main__":
    main()
