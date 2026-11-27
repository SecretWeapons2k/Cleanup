"""Side-by-side check of FRAME_DONOR against BOX_CELL: original | BOX_CELL | FRAME_DONOR | difference (x5).

    python compare_frame_donor.py --old ~/cleanup_compare/STAGE2_MALL_CELL --new ~/cleanup_compare/STAGE2_FRAME_TEST \
        --out ~/cleanup_compare/frame_compare

--old: a Stage-2 run that has the BOX_CELL tiles, --new: the run with the FRAME_DONOR tiles. Only regions that really
got a FRAME_DONOR tile are shown (a refused region has none, see sibling_misses.csv). Reads nothing but images."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _load(path: Path):
    im = np.asarray(Image.open(path).convert("RGBA"))
    return im[..., :3].astype(np.float32), im[..., 3].astype(np.float32)


def _over_grey(rgb, a):
    al = (a / 255.0)[..., None]
    return (rgb * al + 128.0 * (1 - al)).clip(0, 255).astype(np.uint8)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--old", type=Path, required=True, help="run with the BOX_CELL tiles")
    p.add_argument("--new", type=Path, required=True, help="run with the FRAME_DONOR tiles")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max", type=int, default=60, help="at most this many sheets (worst difference first)")
    args = p.parse_args(argv)
    old_dir = args.old.expanduser() / "debug_candidates" / "regions"
    new_dir = args.new.expanduser() / "debug_candidates" / "regions"
    args.out.expanduser().mkdir(parents=True, exist_ok=True)
    items, selected, with_donor, no_cell = [], 0, 0, 0
    for js in sorted(new_dir.glob("*.json")):
        data = json.loads(js.read_text(encoding="utf-8"))
        for e in data.get("regions", []):
            selected += 1
            cands = e.get("candidates", {})
            if "FRAME_DONOR" not in cands:
                continue
            with_donor += 1
            stem = js.stem
            fd = new_dir / cands["FRAME_DONOR"]
            orig = new_dir / e["original"]
            bc = old_dir / f"{stem}__r{e['region']}__BOX_CELL.png"
            if not (fd.is_file() and orig.is_file() and bc.is_file()):
                no_cell += 1
                continue
            o, oa = _load(orig)
            b, ba = _load(bc)
            f, fa = _load(fd)
            if not (o.shape == b.shape == f.shape):
                no_cell += 1
                continue
            d = np.abs(b - f).max(axis=-1)
            items.append((float(np.percentile(d, 99)), stem, e["region"], o, oa, b, ba, f, fa, d))
    items.sort(key=lambda x: -x[0])                     # the biggest differences first: where to look
    for score, stem, reg, o, oa, b, ba, f, fa, d in items[: args.max]:
        diff = np.clip(d * 5, 0, 255).astype(np.uint8)
        sheet = np.hstack([_over_grey(o, oa), _over_grey(b, ba), _over_grey(f, fa), np.dstack([diff] * 3)])
        Image.fromarray(sheet).save(args.out.expanduser() / f"{score:05.1f}_{stem}_r{reg}.png")
    print(f"{selected} Regionen im neuen Lauf, {with_donor} mit FRAME_DONOR-Kachel, {no_cell} nicht vergleichbar (BOX_CELL fehlt)")
    print(f"{min(len(items), args.max)} Vergleichsbilder (links: Original | BOX_CELL | FRAME_DONOR | Differenz x5) in {args.out}")
    if items:
        v = np.array([i[0] for i in items])
        print(f"Unterschied BOX_CELL/FRAME_DONOR (p99 je Region): median {np.median(v):.1f}, p90 {np.percentile(v, 90):.1f}, max {v.max():.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
