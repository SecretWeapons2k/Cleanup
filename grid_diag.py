"""Diagnose: was sagt der Grid-Detektor zu jedem Bild, und welches Raster wird daraus?

Aufruf aus dem Projektordner (laeuft in Sekunden, veraendert nichts):
    uv run python grid_diag.py --features ~/cleanup_compare/STAGE1_CELLS3/stage1_features.csv \
        --out ~/cleanup_compare/STAGE1_CELLS3/grid_diag

Schreibt grid_diag.csv (eine Zeile je Bild), eine Zusammenfassung auf der Konsole und
montage_*.png (Bilder mit Raster in Magenta eingezeichnet bzw. ohne Raster, mit Namen).
Die Bilder stammen aus config.DEFAULT_IMAGES_DIR (oder --images-dir).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from yolo_mask_auto_select_modules import cells
from yolo_mask_auto_select_modules.config import DEFAULT_IMAGES_DIR


def montage(items: list[tuple[str, Image.Image]], path: Path, per: int = 12) -> None:
    cw, ch, lab, cols_n = 340, 190, 18, 4
    for part, start in enumerate(range(0, len(items), per), start=1):
        chunk = items[start:start + per]
        rows_n = (len(chunk) + cols_n - 1) // cols_n
        sheet = Image.new("RGB", (cols_n * (cw + 8), rows_n * (ch + lab + 8)), (40, 40, 40))
        d = ImageDraw.Draw(sheet)
        for i, (label, im) in enumerate(chunk):
            t = im.copy()
            t.thumbnail((cw, ch))
            x, y = (i % cols_n) * (cw + 8), (i // cols_n) * (ch + lab + 8)
            sheet.paste(t, (x, y))
            d.text((x + 2, y + ch + 2), label, fill=(255, 255, 255))
        sheet.save(path.with_name(f"{path.stem}_{part}.png"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--montage-count", type=int, default=36)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    feats = pd.read_csv(args.features)
    regions = feats.groupby("image").size()
    rows, with_grid, without = [], [], []
    for name in sorted(feats["image"].unique()):
        path = args.images_dir / name
        if not path.is_file():
            rows.append({"image": name, "gate": "Datei nicht gefunden"})
            continue
        im = Image.open(path).convert("RGBA")
        a = np.asarray(im)
        rgb, alpha = a[..., :3].copy(), a[..., 3].copy()
        gate = cells.gate_reason(rgb, alpha)
        grid = cells.detect_cell_grid(rgb, alpha)
        rows.append({
            "image": name, "width": im.width, "height": im.height, "regions": int(regions[name]),
            "transparent_frac": round(float((alpha < cells.ALPHA_GAP_LEVEL).mean()), 3),
            "gate": gate, "grid": grid.summary() if grid else "", "cells": grid.n_cells if grid else 0,
        })
        if grid:
            with_grid.append((f"{name} {grid.summary()} ({gate})", Image.fromarray(cells.draw_grid(rgb, grid))))
        else:
            bg = Image.new("RGBA", im.size, (200, 200, 200, 255))
            bg.alpha_composite(im)
            without.append((f"{name} {gate}", bg.convert("RGB")))

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "grid_diag.csv", index=False)
    df["gate_kurz"] = df["gate"].astype(str).str.replace(r"[:(].*$", "", regex=True)
    print("Urteil des Detektors (gate):")
    print(df["gate_kurz"].value_counts().to_string())
    print(f"\nBilder mit Raster: {int((df['grid'].fillna('') != '').sum())} | ohne: {int((df['grid'].fillna('') == '').sum())}")
    print("\nRaster-Typen:")
    print(df[df["grid"].fillna("") != ""]["grid"].value_counts().head(15).to_string())
    big = df[df["cells"].fillna(0) >= 20].sort_values("cells", ascending=False)
    if len(big):
        print("\nBilder mit >= 20 Zellen (bitte stichprobenhaft pruefen):")
        print(big[["image", "grid", "cells", "regions", "width", "height"]].head(12).to_string(index=False))
    montage(with_grid[:args.montage_count], args.out / "montage_mit_raster")
    montage(without[:args.montage_count], args.out / "montage_ohne_raster")
    print(f"\nGeschrieben: {args.out}/grid_diag.csv, montage_mit_raster_*.png, montage_ohne_raster_*.png")


if __name__ == "__main__":
    main()
