"""Stage 1 of the sieve: BOX_ONLY only.

Every image is filled with the Box-only mask (all anchors minus protected zones) - one LaMa
run per image, no YOLO. For every text region the result is measured (gate_features) and
cropped (original / result / mask overlay) so that it can be labelled "usable / unusable"
quickly (build_gate_labeler.py) and later be judged automatically by the gate.

Output (in --output):
    stage1_features.csv      one row per region: bbox, integrity flags, all measured features
    labeling/<stem>.json     index of the crops of one image
    labeling/<stem>__r<k>__{orig,result,overlay}.png
    stage1_run_stats.json
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .candidate_masks import enforce_protection
from .config import (
    DEFAULT_ANCHORS_CSV, DEFAULT_IMAGES_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_PROTECTED_ZONES_CSV,
    FILL_REPAIR_ENABLED, LAMA_BATCH_SIZE, LAMA_PARALLEL_PROCESSES, POSTPROCESS_WORKERS,
    PROTECTED_ZONE_MARGIN_PX, REGION_MERGE_PX, REGION_SCORE_MARGIN_PX,
)
from .fill_repair import repair_flat_fill_artifacts
from .gate_features import context_type, image_features, lama_integrity, region_features
from .geometry import boxes_union_mask, protected_mask, read_protected_zones, resolve_protected_zones_csv
from .io_utils import (
    choose_output_dir, choose_random_sample, discover_sources, find_iopaint,
    read_anchor_csv, remove_small_alpha_islands, rgb_array,
)
from .cells import (CELLS_VERSION, SPLIT_REGIONS_BY_CELLS, USE_CELL_WINDOWS, detect_cell_grid, region_windows,
                    split_plan_by_cells)
from .regions import plan_regions, region_window
from .reports import write_csv_with_union_fields


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1 of the sieve: fill every region with Box-only and measure the result.")
    p.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    p.add_argument("--anchors", type=Path, default=DEFAULT_ANCHORS_CSV)
    p.add_argument("--protected-zones", type=Path, default=DEFAULT_PROTECTED_ZONES_CSV)
    p.add_argument("--no-protected-zones", action="store_true")
    p.add_argument("--image", help="Process only this image, e.g. m004_015.png")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--random-sample", type=int, default=0)
    p.add_argument("--min-m-images", type=int, default=3)
    p.add_argument("--seed", type=int, default=20261018)
    p.add_argument("--model", default="lama")
    p.add_argument("--lama-device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--lama-processes", type=int, default=LAMA_PARALLEL_PROCESSES)
    p.add_argument("--batch-size", type=int, default=LAMA_BATCH_SIZE)
    p.add_argument("--workers", type=int, default=POSTPROCESS_WORKERS, help="Images measured concurrently")
    p.add_argument("--region-merge-px", type=int, default=REGION_MERGE_PX)
    p.add_argument("--no-fill-repair", action="store_true")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR.parent / "stage1_runs")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def _run_iopaint(iopaint: str, model: str, device: str, images: Path, masks: Path, out: Path,
                 workers: int, tmp: Path, cwd: Path) -> int:
    """Run IOPaint on one staged batch, optionally as several processes (same scheme as the main pipeline)."""
    files = sorted(p.name for p in images.iterdir())
    workers = max(1, min(workers, len(files)))
    base = [iopaint, "run", f"--model={model}", f"--device={device}"]
    out.mkdir(parents=True, exist_ok=True)
    if workers <= 1:
        return subprocess.run(base + [f"--image={images}", f"--mask={masks}", f"--output={out}"], cwd=cwd, check=False).returncode
    root = tmp / "lama_shards"
    cmds = []
    for k in range(workers):
        (root / f"s{k}" / "images").mkdir(parents=True, exist_ok=True)
        (root / f"s{k}" / "masks").mkdir(parents=True, exist_ok=True)
        cmds.append(base + [f"--image={root / f's{k}' / 'images'}", f"--mask={root / f's{k}' / 'masks'}", f"--output={out}"])
    for i, name in enumerate(files):
        k = i % workers
        shutil.move(str(images / name), str(root / f"s{k}" / "images" / name))
        shutil.move(str(masks / name), str(root / f"s{k}" / "masks" / name))
    procs = [subprocess.Popen(c, cwd=cwd) for c in cmds]
    try:
        codes = [pr.wait() for pr in procs]
    finally:
        for pr in procs:
            if pr.poll() is None:
                pr.terminate()
    shutil.rmtree(root, ignore_errors=True)
    return next((c for c in codes if c != 0), 0)


def _rgba_crop(rgb: np.ndarray, alpha: np.ndarray, win: tuple[int, int, int, int]) -> Image.Image:
    x1, y1, x2, y2 = win
    arr = np.dstack((rgb[y1:y2 + 1, x1:x2 + 1], alpha[y1:y2 + 1, x1:x2 + 1])).astype(np.uint8)
    return Image.fromarray(np.ascontiguousarray(arr), mode="RGBA")


def _overlay_crop(rgb: np.ndarray, mask: np.ndarray, win: tuple[int, int, int, int]) -> Image.Image:
    x1, y1, x2, y2 = win
    base = rgb[y1:y2 + 1, x1:x2 + 1].astype(np.float32)
    m = mask[y1:y2 + 1, x1:x2 + 1]
    tint = np.array([255.0, 35.0, 185.0], dtype=np.float32)
    base[m] = base[m] * 0.48 + tint * 0.52
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")


def process_image(
    *, name: str, stem: str, source_path: Path, mask: np.ndarray, protected: np.ndarray, pred_path: Path,
    pred_alpha_path: Path, out_dir: Path, merge_px: int, use_fill_repair: bool,
) -> list[dict[str, Any]]:
    original_rgb, original_alpha = rgb_array(source_path)
    pred_rgb = pred_alpha = None
    if pred_path.is_file():
        with Image.open(pred_path) as im:
            pred_rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    if pred_alpha_path.is_file():
        with Image.open(pred_alpha_path) as im:
            pred_alpha = np.asarray(im.convert("RGB").convert("L"), dtype=np.uint8)

    integrity = lama_integrity(original_rgb, pred_rgb, mask)
    aligned = pred_rgb is not None and pred_alpha is not None and pred_rgb.shape == original_rgb.shape \
        and pred_alpha.shape == original_alpha.shape
    final_rgb, final_alpha = original_rgb.copy(), original_alpha.copy()
    meta: dict[str, float] = {}
    if aligned:
        final_rgb[mask] = pred_rgb[mask]
        if use_fill_repair:
            final_rgb, fr = repair_flat_fill_artifacts(original_rgb, final_rgb, mask, protected)
            meta.update(fr)
        final_alpha[mask] = pred_alpha[mask]
        final_rgb, final_alpha, a_meta = remove_small_alpha_islands(final_rgb, final_alpha, original_alpha, protected=protected)
        meta.update(a_meta)
        meta.update(enforce_protection(final_rgb, final_alpha, original_rgb, original_alpha, protected))

    height, width = original_rgb.shape[:2]
    plan = plan_regions(mask, merge_px)
    # Windows follow the sprite cells (union of the cells a region touches); without a detectable
    # grid they fall back to bbox +- REGION_SCORE_MARGIN_PX exactly as before.
    grid = detect_cell_grid(original_rgb, original_alpha) if USE_CELL_WINDOWS else None
    split_origin: dict[int, int] = {}
    if SPLIT_REGIONS_BY_CELLS:     # one region per sprite cell (a whole strip is no useful unit)
        plan, split_origin = split_plan_by_cells(plan, grid)
    windows = region_windows(plan.core_labels, plan.bboxes, grid, REGION_SCORE_MARGIN_PX)
    grid_label = grid.summary() if grid else ""
    rows: list[dict[str, Any]] = []
    index: dict[str, Any] = {"image": name, "imageSize": [width, height], "full": f"{stem}__full.png", "regions": [],
                             "cellGrid": grid_label}
    thumb = Image.fromarray(np.ascontiguousarray(np.dstack((original_rgb, original_alpha)).astype(np.uint8)), mode="RGBA")
    thumb.thumbnail((480, 480), Image.Resampling.LANCZOS)
    thumb.save(out_dir / "labeling" / f"{stem}__full.png")
    img_feats = image_features(original_rgb, original_alpha)
    for k in range(1, plan.count + 1):
        bbox = plan.bboxes[k - 1]
        rw = windows[k - 1]
        win = rw.rect
        x1, y1, x2, y2 = win
        sl = (slice(y1, y2 + 1), slice(x1, x2 + 1))
        terr = plan.territory[sl] == k
        region_mask = mask[sl] & terr
        other = mask[sl] & ~terr
        # Result inside this territory only; everything else stays the untouched original.
        res_rgb = np.where(terr[..., None], final_rgb[sl], original_rgb[sl])
        res_alpha = np.where(terr, final_alpha[sl], original_alpha[sl])
        feats = region_features(
            src_rgb=original_rgb[sl], src_alpha=original_alpha[sl],
            res_rgb=res_rgb, res_alpha=res_alpha, mask=region_mask, other_mask=other,
        )
        base = f"{stem}__r{k}"
        _rgba_crop(original_rgb, original_alpha, win).save(out_dir / "labeling" / f"{base}__orig.png")
        Image.fromarray(np.ascontiguousarray(np.dstack((res_rgb, res_alpha)).astype(np.uint8)), mode="RGBA").save(
            out_dir / "labeling" / f"{base}__result.png")
        _overlay_crop(original_rgb, mask & (plan.territory == k), win).save(out_dir / "labeling" / f"{base}__overlay.png")
        index["regions"].append({
            "region": k, "bbox": list(bbox), "window": list(win), "windowSource": rw.source, "splitFrom": split_origin.get(k, ""),
            "original": f"{base}__orig.png", "result": f"{base}__result.png", "overlay": f"{base}__overlay.png",
        })
        rows.append({
            "image": name, "region": k, "region_count": plan.count,
            "window_source": rw.source, "cell_grid": grid_label, "split_from": split_origin.get(k, ""),
            "bbox_x1": bbox[0], "bbox_y1": bbox[1], "bbox_x2": bbox[2], "bbox_y2": bbox[3],
            "asset_type": img_feats["asset_type"],
            "context_type": context_type(float(feats.get("ring_flat_frac", 1.0))),
            "lama_aligned": 1.0 if aligned else 0.0,
            **integrity,
            "fill_repair_pixels_replaced": meta.get("fill_repair_pixels_replaced", 0.0),
            "protected_pixels_restored": meta.get("protected_pixels_restored", 0.0),
            **{key: f"{value:.6f}" if isinstance(value, float) else value for key, value in feats.items()},
            **{key: f"{value:.6f}" for key, value in img_feats.items() if key != "asset_type"},
        })
    (out_dir / "labeling" / f"{stem}.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    return rows


def main(argv: list[str] | None = None) -> int:
    project = Path(__file__).resolve().parent.parent
    args = parse_args(argv)
    anchors_path = args.anchors.expanduser().resolve()
    images_dir = args.images_dir.expanduser().resolve()
    try:
        _fields, _rows, anchors_by_image = read_anchor_csv(anchors_path)
        names = set(anchors_by_image)
        if args.random_sample:
            if args.image or args.limit:
                raise ValueError("Use --random-sample without --image/--limit")
            ordered = choose_random_sample(names, args.random_sample, args.min_m_images, args.seed)
        else:
            sel = {Path(args.image).name} if args.image else set(names)
            if args.image and Path(args.image).name not in names:
                raise ValueError(f"No valid anchor rows for image {args.image}")
            ordered = sorted(sel, key=str.casefold)
            if args.limit:
                ordered = ordered[:args.limit]
        if not ordered:
            raise ValueError("No images selected.")
        sources = discover_sources(images_dir, set(ordered), project)
        zones_by_image: dict[str, list[dict[str, Any]]] = {}
        zones_path = None if args.no_protected_zones else resolve_protected_zones_csv(anchors_path, args.protected_zones)
        if zones_path is not None:
            zones_by_image = read_protected_zones(zones_path)
        iopaint = find_iopaint()
        output_dir = choose_output_dir(args.output, project)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    use_fill_repair = FILL_REPAIR_ENABLED and not args.no_fill_repair
    print(f"Stage 1 (BOX_ONLY): {len(ordered)} images -> {output_dir}")
    if args.dry_run:
        print("DRY RUN: paths checked; nothing written.")
        return 0
    (output_dir / "labeling").mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    seconds: dict[str, float] = defaultdict(float)
    skipped_no_mask: list[str] = []
    t_run = time.perf_counter()
    batches = math.ceil(len(ordered) / max(1, args.batch_size))
    with tempfile.TemporaryDirectory(prefix="stage1_") as tmp_s:
        tmp = Path(tmp_s)
        for b in range(batches):
            names_b = ordered[b * args.batch_size:(b + 1) * args.batch_size]
            print(f"\n--- Batch {b + 1}/{batches} ({len(names_b)} Bilder) ---", flush=True)
            images_stage, masks_stage, raw = tmp / "in_images", tmp / "in_masks", tmp / "out"
            for d in (images_stage, masks_stage, raw):
                shutil.rmtree(d, ignore_errors=True)
                d.mkdir(parents=True)
            staged: dict[str, tuple[str, str, np.ndarray, np.ndarray, str]] = {}
            t0 = time.perf_counter()
            for name in names_b:
                rgb, alpha = rgb_array(sources[name])
                h, w = rgb.shape[:2]
                prot = protected_mask(zones_by_image.get(name, []), w, h, PROTECTED_ZONE_MARGIN_PX)
                mask = boxes_union_mask(anchors_by_image[name], w, h) & ~prot
                if not mask.any():
                    skipped_no_mask.append(name)
                    continue
                stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in Path(name).stem)
                f_rgb, f_alpha = f"{stem}__box_only_rgb.png", f"{stem}__box_only_alpha.png"
                Image.fromarray(rgb, mode="RGB").save(images_stage / f_rgb)
                Image.fromarray(alpha, mode="L").convert("RGB").save(images_stage / f_alpha)
                m_img = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
                m_img.save(masks_stage / f_rgb)
                m_img.save(masks_stage / f_alpha)
                staged[name] = (f_rgb, f_alpha, mask, prot, stem)
            seconds["staging"] += time.perf_counter() - t0
            if not staged:
                continue
            t0 = time.perf_counter()
            code = _run_iopaint(iopaint, args.model, args.lama_device, images_stage, masks_stage, raw,
                                args.lama_processes, tmp, project)
            seconds["lama"] += time.perf_counter() - t0
            if code != 0:
                print(f"ERROR: IOPaint failed in batch {b + 1} with code {code}", file=sys.stderr)
                return code or 1

            t0 = time.perf_counter()

            def _one(name: str) -> list[dict[str, Any]]:
                f_rgb, f_alpha, mask, prot, stem = staged[name]
                return process_image(
                    name=name, stem=stem, source_path=sources[name], mask=mask, protected=prot,
                    pred_path=raw / f_rgb, pred_alpha_path=raw / f_alpha, out_dir=output_dir,
                    merge_px=max(0, args.region_merge_px), use_fill_repair=use_fill_repair,
                )

            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
                for rows in pool.map(_one, list(staged)):
                    all_rows.extend(rows)
            seconds["measure_and_crops"] += time.perf_counter() - t0
            write_csv_with_union_fields(output_dir / "stage1_features.csv",
                                        sorted(all_rows, key=lambda r: (str(r["image"]).casefold(), int(r["region"]))))

    stats = {
        "images": len(ordered), "regions": len(all_rows), "images_without_mask": skipped_no_mask,
        "seconds": {k: round(v, 2) for k, v in seconds.items()}, "total_seconds": round(time.perf_counter() - t_run, 2),
        "region_merge_px": args.region_merge_px, "fill_repair": use_fill_repair,
        "split_regions_by_cells": bool(USE_CELL_WINDOWS and SPLIT_REGIONS_BY_CELLS), "cells_version": CELLS_VERSION,
    }
    (output_dir / "stage1_run_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    misaligned = sum(1 for r in all_rows if float(r["lama_aligned"]) == 0.0 or float(r["lama_outside_frac_gt8"]) > 0.02)
    print(f"\n=== STAGE 1 COMPLETE ===\nRegionen: {len(all_rows)} aus {len(ordered)} Bildern")
    if misaligned:
        print(f"WARNUNG: {misaligned} Regionen mit auffälliger LaMa-Ausgabe (Verschiebung/Größe) - siehe stage1_features.csv (lama_*)")
    print(f"Merkmale: {output_dir / 'stage1_features.csv'}\nLabel-Ausschnitte: {output_dir / 'labeling'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
