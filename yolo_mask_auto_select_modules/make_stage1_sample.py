"""Small stage-1 run for a stage-2 test: only the given images, everything else linked (originals stay untouched).

    python make_stage1_sample.py --stage1-run ~/cleanup_compare/STAGE1_M_ALL --out ~/cleanup_compare/STAGE1_SAMPLE \
        --images m003_001,m003_002,m004_005,m000_005

--stage1-run may be a folder ABOVE the run: the first folder below it that has stage1_features.csv with the images is used."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _images_in(csv_path: Path) -> set[str]:
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        return {r["image"] for r in csv.DictReader(f)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--images", required=True, help="comma separated, with or without .png")
    args = p.parse_args(argv)
    want = {(x.strip() if x.strip().endswith(".png") else x.strip() + ".png") for x in args.images.split(",") if x.strip()}
    root = args.stage1_run.expanduser()
    cands = sorted(root.rglob("stage1_features.csv"), key=lambda q: len(q.parts))
    cands = [c for c in cands if len(c.relative_to(root).parts) <= 4]
    print("Gefundene stage1_features.csv unter", root)
    best = None
    for c in cands:
        have = want & _images_in(c)
        print(f"  {c}  ({len(have)} von {len(want)} Bildern)")
        if best is None and have:
            best = (c, have)
    if best is None:
        print("Keines der Bilder gefunden. Gib mit --stage1-run den Ordner an, der zum Stage-2-Lauf gehoert.")
        return 1
    csv_path, have = best
    run = csv_path.parent
    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r["image"] in want]
        names = reader.fieldnames
    with (out / "stage1_features.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=names)
        w.writeheader()
        w.writerows(rows)
    for item in run.iterdir():
        if item.name != "stage1_features.csv" and not (out / item.name).exists():
            (out / item.name).symlink_to(item)
    print(f"\nVerwendet: {run}\n{len(rows)} Regionen aus {len(have)} Bildern -> {out}")
    missing = want - have
    if missing:
        print("Nicht in diesem Lauf:", ", ".join(sorted(missing)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
