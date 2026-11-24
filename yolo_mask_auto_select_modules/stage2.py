"""Stage 2 of the sieve: alternative methods for regions where Box-only is risky or failed.

Input is a stage-1 run (stage1_features.csv). Regions are selected by their surrounding
(``context_type``: structured / mixed, computed from ``ring_flat_frac`` if the CSV has no such
column) and/or by human labels (``--labels``: every region labelled unusable). For each selected
region every method below produces a result; each is measured (gate_features) and cropped in the
format of build_ranker.py, together with the stage-1 Box-only result as the reference tile.
Mark the unusable tiles in the ranker (Shift+click) to learn which method works for which
surrounding and in which order the sieve should try them.

Methods need no YOLO: smaller masks than Box-only (text strokes only) and/or a fill that cannot
invent structure (Telea). Output (in --output):
    stage2_features.csv                         one row per (region, method)
    debug_candidates/regions/<stem>.json        ranker index (candidates incl. BOX_ONLY)
    debug_candidates/regions/<stem>__r<k>__<METHOD>.png
    stage2_run_stats.json
"""
from __future__ import annotations

import argparse
import csv
import threading
import json
import math
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image

from .candidate_masks import enforce_protection, group_identical_masks
from .config import (
    DEFAULT_ANCHORS_CSV, DEFAULT_IMAGES_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_PROTECTED_ZONES_CSV,
    FILL_REPAIR_ENABLED, LAMA_BATCH_SIZE, LAMA_PARALLEL_PROCESSES, POSTPROCESS_WORKERS,
    PROTECTED_ZONE_MARGIN_PX, REGION_SCORE_MARGIN_PX,
)
from .fill_repair import repair_flat_fill_artifacts
from .gate_features import context_type, lama_integrity, region_features
from .geometry import boxes_union_mask, protected_mask, read_protected_zones, resolve_protected_zones_csv
from .io_utils import (
    choose_output_dir, discover_sources, find_iopaint, make_box_only_mask, read_anchor_csv, remove_small_alpha_islands,
    rgb_array,
)
from .cells import USE_CELL_WINDOWS, detect_cell_grid, region_windows, split_plan_for_image
from .cells import CELLS_VERSION, SPLIT_REGIONS_BY_CELLS
from .donor import MISS_FIELDS, DonorBank, explain_no_donor, find_donors, stamp
from .siblings import SiblingBank, SiblingResult, find_tiles, sibling_fill
from . import siblings_v1
from .regions import plan_regions, region_window
from .reports import write_csv_with_union_fields
from .scoring import make_contrast_mask, make_smart_ink_mask
from .stage1 import _rgba_crop, _run_iopaint

TELEA_RADIUS = 3
# A method whose mask covers (almost) the whole region box is Box-only in disguise (smart-ink falls back
# to the full box when it finds no / too much "ink"). Such a tile carries no new information: skip it.
BOX_LIKE_COVERAGE = 0.85

MaskFn = Callable[[np.ndarray, np.ndarray, list[dict[str, Any]]], np.ndarray]

# name -> (filler, mask builder). Order = default order in which the sieve would try them.
METHODS: dict[str, tuple[str, MaskFn]] = {
    "SMART_INK": ("lama", lambda rgb, a, b: make_smart_ink_mask(rgb, b)),
    "SMART_INK_R4": ("lama", lambda rgb, a, b: make_smart_ink_mask(rgb, b, dilation_radius=4)),
    "SMART_INK_T10": ("lama", lambda rgb, a, b: make_smart_ink_mask(rgb, b, contrast_threshold=0.10, dilation_radius=3)),
    "CONTRAST_T38": ("lama", lambda rgb, a, b: make_contrast_mask(rgb, a, b, 0.38)),
    # Stamp from an identical, text-free copy elsewhere in the image (see donor.py). The mask is the
    # whole box; regions without a verified clean donor get no tile.
    "DONOR": ("donor", lambda rgb, a, b: boxes_union_mask(b, rgb.shape[1], rgb.shape[0])),
    # Consensus of sibling cells (same frame, different text): see siblings.py. Whole box as mask; regions
    # without enough clean-voting siblings get no tile (reason in sibling_misses.csv).
    "SIBLING": ("sibling", lambda rgb, a, b: boxes_union_mask(b, rgb.shape[1], rgb.shape[0])),
    # Frozen earlier version of SIBLING (before the glow rework, no bank) to compare both in the ranker.
    "SIBLING_V1": ("sibling_v1", lambda rgb, a, b: boxes_union_mask(b, rgb.shape[1], rgb.shape[0])),
}
# SMART_TELEA / SMART_R4_TELEA were removed after the stage-2 ranking: they never rescued a region.


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2 of the sieve: alternative methods for risky / failed Box-only regions.")
    p.add_argument("--stage1-run", type=Path, required=True, help="Output folder of yolo_stage1.py (contains stage1_features.csv)")
    p.add_argument("--labels", type=Path, default=None, help="gate_labels CSV; regions labelled 'unusable' are always included")
    p.add_argument("--select", choices=("nonflat", "all", "unusable"), default="nonflat",
                   help="nonflat: structured+mixed surroundings (+ labelled unusable); all: every region; unusable: labelled only")
    p.add_argument("--methods", default=",".join(METHODS), help="Comma-separated subset of: " + ", ".join(METHODS))
    p.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    p.add_argument("--anchors", type=Path, default=DEFAULT_ANCHORS_CSV)
    p.add_argument("--protected-zones", type=Path, default=DEFAULT_PROTECTED_ZONES_CSV)
    p.add_argument("--no-protected-zones", action="store_true")
    p.add_argument("--model", default="lama")
    p.add_argument("--lama-device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--lama-processes", type=int, default=LAMA_PARALLEL_PROCESSES)
    p.add_argument("--batch-size", type=int, default=LAMA_BATCH_SIZE)
    p.add_argument("--workers", type=int, default=POSTPROCESS_WORKERS)
    p.add_argument("--region-merge-px", type=int, default=None, help="Default: value of the stage-1 run (must match it)")
    p.add_argument("--no-fill-repair", action="store_true")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR.parent / "stage2_runs")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--donor-bank", type=Path, default=None,
                   help="Folder of the donor bank: clean donors remembered across images (see donor.py)")
    p.add_argument("--donor-near", action="store_true",
                   help="Also allow donors whose ring differs slightly (marked donor_tier=near); seam must still match")
    p.add_argument("--donor-bank-learn", action="store_true",
                   help="Also save newly found clean donors into the bank (used from the NEXT run on)")
    p.add_argument("--sibling-bank", type=Path, default=None,
                   help="Folder of the sibling bank: frames whose background the sibling vote verified, remembered across images (see siblings.py)")
    p.add_argument("--sibling-bank-learn", action="store_true",
                   help="Also save newly verified frame pixels into the sibling bank (used from the NEXT run on)")
    return p.parse_args(argv)


def read_selection(stage1_run: Path, labels_path: Path | None, mode: str) -> tuple[dict[str, dict[int, dict[str, str]]], int | None]:
    """Return ({image: {region: info}}, merge_px of the stage-1 run)."""
    feats_path = stage1_run / "stage1_features.csv"
    if not feats_path.is_file():
        raise FileNotFoundError(f"stage1_features.csv not found in {stage1_run}")
    labels: dict[tuple[str, int], str] = {}
    if labels_path is not None:
        with labels_path.expanduser().open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                labels[(row["image"], int(row["region"]))] = row["label"]
    selected: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    with feats_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            region = int(row["region"])
            ctx = row.get("context_type") or context_type(float(row["ring_flat_frac"]))
            lab = labels.get((row["image"], region), "")
            take = (
                mode == "all"
                or (mode == "unusable" and lab == "unusable")
                or (mode == "nonflat" and (ctx != "flat" or lab == "unusable"))
            )
            if take:
                selected[row["image"]][region] = {"context_type": ctx, "stage1_label": lab}
    merge_px = None
    stats = stage1_run / "stage1_run_stats.json"
    if stats.is_file():
        merge_px = json.loads(stats.read_text(encoding="utf-8")).get("region_merge_px")
    return dict(selected), merge_px


def stage1_cells_version(stage1_run: Path) -> int:
    """Version of the cell detection stage 1 was run with (older runs did not record it: 0)."""
    stats = stage1_run / "stage1_run_stats.json"
    if not stats.is_file():
        return 0
    return int(json.loads(stats.read_text(encoding="utf-8")).get("cells_version", 0))


def stage1_split_flag(stage1_run: Path) -> bool:
    """Was stage 1 run with regions split by cells? (older runs did not record it: False)"""
    stats = stage1_run / "stage1_run_stats.json"
    if not stats.is_file():
        return False
    return bool(json.loads(stats.read_text(encoding="utf-8")).get("split_regions_by_cells", False))


def _telea(rgb: np.ndarray, alpha: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = mask.astype(np.uint8) * 255
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    out = cv2.cvtColor(cv2.inpaint(bgr, m, TELEA_RADIUS, cv2.INPAINT_TELEA), cv2.COLOR_BGR2RGB)
    out_alpha = cv2.inpaint(np.ascontiguousarray(alpha), m, TELEA_RADIUS, cv2.INPAINT_TELEA)
    return out, out_alpha


_MISS_LOCK = threading.Lock()


def _log_donor_miss(out_dir: Path, image: str, region: int, info: dict[str, Any]) -> None:
    """One row per region that got no stamp, with the reason (donor_misses.csv in the run folder)."""
    path = out_dir / "donor_misses.csv"
    with _MISS_LOCK:
        new = not path.exists()
        with path.open("a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["image", "region", *MISS_FIELDS])
            if new:
                w.writeheader()
            w.writerow({"image": image, "region": region, **info})


def _log_sibling_miss(out_dir: Path, image: str, region: int, reason: str, meta: dict[str, Any], method: str = "SIBLING") -> None:
    """One row per region the sibling method refused (sibling_misses.csv in the run folder)."""
    path = out_dir / "sibling_misses.csv"
    fields = ["image", "region", "method", "reason", "sibling_count", "sibling_same_size_tiles", "sibling_alpha_rejected",
              "sibling_colour_rejected", "sibling_gate_pixels_min", "sibling_voted_frac", "sibling_hole_frac",
              "sibling_changed_frac", "sibling_entry_id", "sibling_bank_frac", "sibling_parts", "sibling_off_tile_frac",
              "sibling_tiles_in_image", "sibling_region_visible_frac", "sibling_biggest_tile_px", "sibling_region_px",
              "sibling_ink_px", "sibling_shared_ink_frac", "sibling_skipped_px"]
    with _MISS_LOCK:
        new = not path.exists()
        with path.open("a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow({"image": image, "region": region, "method": method, "reason": reason, **meta})


def process_image(
    *, name: str, stem: str, source_path: Path, protected: np.ndarray, plan, selected: dict[int, dict[str, str]],
    masks: dict[str, np.ndarray], groups: dict[str, str], staged: dict[str, tuple[str, str]], raw: Path,
    out_dir: Path, stage1_run: Path, use_fill_repair: bool, bank: DonorBank | None = None,
    donor_near: bool = False, sibling_bank: SiblingBank | None = None, body_mask: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], int]:
    original_rgb, original_alpha = rgb_array(source_path)
    height, width = original_rgb.shape[:2]
    # Same cell windows as stage 1 (both detect the grid from the same pixels, so they agree). Needed
    # before the fill loop: the donor stamp cuts its ring at the cell.
    grid = detect_cell_grid(original_rgb, original_alpha) if USE_CELL_WINDOWS else None
    windows = region_windows(plan.core_labels, plan.bboxes, grid, REGION_SCORE_MARGIN_PX)
    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    integrity: dict[str, dict[str, float]] = {}
    repair_meta: dict[str, float] = {}
    donor_meta: dict[int, dict[str, Any]] = {}
    sibling_meta: dict[tuple[str, int], dict[str, Any]] = {}

    for method, mask in masks.items():
        filler = METHODS[method][0]
        rep = groups.get(method, method)
        if rep != method:                       # identical mask to an earlier LaMa method
            results[method] = results[rep]
            integrity[method] = integrity[rep]
            continue
        if filler == "lama":
            s_name, a_name = staged[method]
            pred_rgb = pred_alpha = None
            if (raw / s_name).is_file():
                with Image.open(raw / s_name) as im:
                    pred_rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
            if (raw / a_name).is_file():
                with Image.open(raw / a_name) as im:
                    pred_alpha = np.asarray(im.convert("RGB").convert("L"), dtype=np.uint8)
            integrity[method] = lama_integrity(original_rgb, pred_rgb, mask)
            if pred_rgb is None or pred_alpha is None or pred_rgb.shape != original_rgb.shape or pred_alpha.shape != original_alpha.shape:
                results[method] = (original_rgb.copy(), original_alpha.copy())
                continue
            f_rgb = original_rgb.copy()
            f_rgb[mask] = pred_rgb[mask]
            if use_fill_repair:
                f_rgb, _ = repair_flat_fill_artifacts(original_rgb, f_rgb, mask, protected)
            f_alpha = original_alpha.copy()
            f_alpha[mask] = pred_alpha[mask]
        elif filler == "donor":
            f_rgb, f_alpha = original_rgb.copy(), original_alpha.copy()
            stamped = np.zeros(mask.shape, dtype=bool)
            for k in selected:
                region_m = mask & (plan.territory == k)
                if not region_m.any():
                    continue
                rw_k = windows[k - 1]
                bounds = rw_k.rect if rw_k.source == "cell" else None      # ring ends at the sprite cell
                donors = find_donors(original_rgb, original_alpha, region_m, allow_near=donor_near, bounds=bounds)
                got = stamp(original_rgb, original_alpha, region_m, donors)
                meta: dict[str, Any] = {}
                if got is not None:                  # clean copy in the same image
                    s_rgb, s_alpha, d = got
                    meta = {"donor_source": "image", "donor_dx": d.dx, "donor_dy": d.dy,
                            "donor_ring_msd": f"{d.ring_msd:.4f}", "donor_bg_de_p95": f"{d.bg_de_p95:.2f}",
                            "donor_bg_pixels": d.bg_pixels, "donor_color_checked": int(d.color_checked),
                            "donor_tier": d.tier, "donor_ring_bad_frac": f"{d.ring_bad_frac:.4f}"}
                    if bank is not None and d.tier == "exact":              # only exact donors are remembered
                        meta["donor_entry_id"] = bank.add(original_rgb, original_alpha, region_m, d,
                                                          source_image=name, region=k) or ""
                elif bank is not None:               # clean copy remembered from another image
                    usable = [x for x in bank.match(original_rgb, original_alpha, region_m,
                                                    allow_near=donor_near, bounds=bounds) if x.usable]
                    if usable:
                        b = usable[0]
                        s_rgb, s_alpha = bank.stamp(original_rgb, original_alpha, region_m, b)
                        meta = {"donor_source": "bank", "donor_entry_id": b.entry_id,
                                "donor_ring_msd": f"{b.ring_msd:.4f}", "donor_bg_de_p95": f"{b.bg_de_p95:.2f}",
                                "donor_bg_pixels": b.bg_pixels, "donor_color_checked": int(b.color_checked),
                                "donor_tier": b.tier, "donor_ring_bad_frac": f"{b.ring_bad_frac:.4f}"}
                if not meta:
                    _log_donor_miss(out_dir, name, k, explain_no_donor(original_rgb, original_alpha, region_m, bank, bounds=bounds))
                    continue                         # no verified clean donor for this region
                f_rgb[region_m] = s_rgb[region_m]
                f_alpha[region_m] = s_alpha[region_m]
                stamped |= region_m
                donor_meta[k] = meta
            masks[method] = stamped              # only regions that really got a stamp keep a tile
            integrity[method] = {"lama_size_mismatch": 0.0, "lama_outside_mean": 0.0, "lama_outside_frac_gt8": 0.0}
        elif filler in ("sibling", "sibling_v1"):
            f_rgb, f_alpha = original_rgb.copy(), original_alpha.copy()
            stamped = np.zeros(mask.shape, dtype=bool)
            zone = plan.core_labels > 0                       # every text box of the image: the colour gate looks outside them
            tiles = find_tiles(original_alpha)
            if len(tiles) < 3 and grid is not None and grid.n_cells >= 3:
                tiles = [grid.rect_of(i) for i in range(grid.n_cells)]      # opaque sheet: the grid cells are the tiles
            for k in selected:
                region_m = mask & (plan.territory == k)
                if not region_m.any():
                    continue
                if filler == "sibling_v1":
                    res = siblings_v1.sibling_fill(original_rgb, original_alpha, region_m, zone, tiles,
                                                   source_image=name, region=k, body_mask=body_mask)
                    ok_type = siblings_v1.SiblingResult
                else:
                    res = sibling_fill(original_rgb, original_alpha, region_m, zone, tiles,
                                       bank=sibling_bank, source_image=name, region=k, body_mask=body_mask)
                    ok_type = SiblingResult
                if not isinstance(res, ok_type):
                    _log_sibling_miss(out_dir, name, k, res.reason, res.meta, method)
                    continue
                f_rgb[region_m] = res.rgb[region_m]
                stamped |= region_m
                sibling_meta[(method, k)] = res.meta
            masks[method] = stamped                          # only regions that really got a fill keep a tile
            integrity[method] = {"lama_size_mismatch": 0.0, "lama_outside_mean": 0.0, "lama_outside_frac_gt8": 0.0}
        else:
            t_rgb, t_alpha = _telea(original_rgb, original_alpha, mask)
            f_rgb, f_alpha = original_rgb.copy(), original_alpha.copy()
            f_rgb[mask] = t_rgb[mask]
            f_alpha[mask] = t_alpha[mask]
            integrity[method] = {"lama_size_mismatch": 0.0, "lama_outside_mean": 0.0, "lama_outside_frac_gt8": 0.0}
        f_rgb, f_alpha, _ = remove_small_alpha_islands(f_rgb, f_alpha, original_alpha, protected=protected)
        enforce_protection(f_rgb, f_alpha, original_rgb, original_alpha, protected)
        results[method] = (f_rgb, f_alpha)

    reg_dir = out_dir / "debug_candidates" / "regions"
    index: dict[str, Any] = {"image": name, "regions": []}
    rows: list[dict[str, Any]] = []
    for k, info in sorted(selected.items()):
        bbox = plan.bboxes[k - 1]
        rw = windows[k - 1]
        win = rw.rect
        x1, y1, x2, y2 = win
        sl = (slice(y1, y2 + 1), slice(x1, x2 + 1))
        terr = plan.territory[sl] == k
        entry: dict[str, Any] = {"region": k, "bbox": list(bbox), "window": list(win), "window_source": rw.source, "auto_winner": "BOX_ONLY",
                                 "candidates": {}, "stage1_label": info["stage1_label"], "context_type": info["context_type"]}
        entry["skipped_box_like"] = []
        box_px = int((plan.core_labels[sl] == k).sum())
        stage1_crop = stage1_run / "labeling" / f"{stem}__r{k}__result.png"
        if stage1_crop.is_file():
            with Image.open(stage1_crop) as _im1:
                same_window = _im1.size == (x2 - x1 + 1, y2 - y1 + 1)
            if same_window:
                shutil.copyfile(stage1_crop, reg_dir / f"{stem}__r{k}__BOX_ONLY.png")
                entry["candidates"]["BOX_ONLY"] = f"{stem}__r{k}__BOX_ONLY.png"
            else:
                # Stage 1 was run with another window (older code): its tile does not line up with
                # the new tiles. Re-run stage 1 with the current code to get the reference tile.
                entry["box_only_window_mismatch"] = True
        for method, (res_rgb_full, res_alpha_full) in results.items():
            region_mask = masks[method][sl] & terr
            if not region_mask.any():
                continue                         # this method changes nothing here: no tile
            if box_px and METHODS[method][0] not in ("donor", "sibling", "sibling_v1") and int(region_mask.sum()) >= BOX_LIKE_COVERAGE * box_px:
                entry["skipped_box_like"].append(method)
                continue                         # Box-only in disguise: no new information
            other = masks[method][sl] & ~terr
            res_rgb = np.where(terr[..., None], res_rgb_full[sl], original_rgb[sl])
            res_alpha = np.where(terr, res_alpha_full[sl], original_alpha[sl])
            feats = region_features(
                src_rgb=original_rgb[sl], src_alpha=original_alpha[sl], res_rgb=res_rgb, res_alpha=res_alpha,
                mask=region_mask, other_mask=other,
            )
            fname = f"{stem}__r{k}__{method}.png"
            Image.fromarray(np.ascontiguousarray(np.dstack((res_rgb, res_alpha)).astype(np.uint8)), mode="RGBA").save(reg_dir / fname)
            entry["candidates"][method] = fname
            rows.append({
                "image": name, "region": k, "method": method, "filler": METHODS[method][0],
                "context_type": info["context_type"], "stage1_label": info["stage1_label"],
                **integrity[method],
                **(donor_meta.get(k, {}) if method == "DONOR" else {}),
                **(sibling_meta.get((method, k), {}) if method in ("SIBLING", "SIBLING_V1") else {}),
                **{key: f"{value:.6f}" if isinstance(value, float) else value for key, value in feats.items()},
            })
        orig_name = f"{stem}__r{k}__original.png"
        _rgba_crop(original_rgb, original_alpha, win).save(reg_dir / orig_name)
        entry["original"] = orig_name
        index["regions"].append(entry)
    (reg_dir / f"{stem}.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    return rows, sum(len(e["skipped_box_like"]) for e in index["regions"])


def main(argv: list[str] | None = None) -> int:
    project = Path(__file__).resolve().parent.parent
    args = parse_args(argv)
    anchors_path = args.anchors.expanduser().resolve()
    images_dir = args.images_dir.expanduser().resolve()
    stage1_run = args.stage1_run.expanduser().resolve()
    try:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
        unknown = [m for m in methods if m not in METHODS]
        if unknown or not methods:
            raise ValueError(f"Unknown/empty --methods: {unknown or args.methods!r}; available: {', '.join(METHODS)}")
        selection, stage1_merge = read_selection(stage1_run, args.labels, args.select)
        if not selection:
            raise ValueError("No regions selected (check --select / --labels).")
        merge_px = args.region_merge_px if args.region_merge_px is not None else (stage1_merge if stage1_merge is not None else 8)
        if stage1_merge is not None and merge_px != stage1_merge:
            raise ValueError(f"--region-merge-px {merge_px} differs from the stage-1 run ({stage1_merge}); region numbers would not match")
        now_split = bool(USE_CELL_WINDOWS and SPLIT_REGIONS_BY_CELLS)
        if stage1_split_flag(stage1_run) != now_split:
            raise ValueError(
                f"The stage-1 run was made {'with' if stage1_split_flag(stage1_run) else 'without'} splitting regions "
                f"by cells, this code does {'' if now_split else 'not '}split: region numbers would not match. "
                "Re-run stage 1 with the current code (or set SPLIT_REGIONS_BY_CELLS in cells.py accordingly).")
        if now_split and stage1_cells_version(stage1_run) != CELLS_VERSION:
            raise ValueError(
                f"The stage-1 run used cell detection version {stage1_cells_version(stage1_run)}, this code has "
                f"version {CELLS_VERSION}: regions would be split differently and the region numbers would not match. "
                "Re-run stage 1 with the current code.")
        _f, _r, anchors_by_image = read_anchor_csv(anchors_path)
        names = sorted(selection, key=str.casefold)
        missing = [n for n in names if n not in anchors_by_image]
        if missing:
            raise ValueError(f"Images of the stage-1 run missing in anchors CSV: {missing[:5]}")
        sources = discover_sources(images_dir, set(names), project)
        zones_by_image: dict[str, list[dict[str, Any]]] = {}
        zones_path = None if args.no_protected_zones else resolve_protected_zones_csv(anchors_path, args.protected_zones)
        if zones_path is not None:
            zones_by_image = read_protected_zones(zones_path)
        iopaint = find_iopaint()
        output_dir = choose_output_dir(args.output, project)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    n_regions = sum(len(v) for v in selection.values())
    use_fill_repair = FILL_REPAIR_ENABLED and not args.no_fill_repair
    print(f"Stage 2: {n_regions} Regionen in {len(names)} Bildern, Methoden: {', '.join(methods)} -> {output_dir}")
    if args.dry_run:
        print("DRY RUN: paths checked; nothing written.")
        return 0
    (output_dir / "debug_candidates" / "regions").mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    box_like_skipped = 0
    bank = DonorBank(args.donor_bank, learn=args.donor_bank_learn) if args.donor_bank else None
    sibling_bank = SiblingBank(args.sibling_bank, learn=args.sibling_bank_learn) if args.sibling_bank else None
    if sibling_bank is not None:
        print(f"Sibling bank: {len(sibling_bank)} entries from {sibling_bank.path}" + (" (learning)" if sibling_bank.learn else ""))
    if bank is not None:
        print(f"Donor bank: {len(bank)} entries from {bank.path}" + (" (learning)" if bank.learn else ""))
    seconds: dict[str, float] = defaultdict(float)
    t_run = time.perf_counter()
    batches = math.ceil(len(names) / max(1, args.batch_size))
    with tempfile.TemporaryDirectory(prefix="stage2_") as tmp_s:
        tmp = Path(tmp_s)
        for b in range(batches):
            names_b = names[b * args.batch_size:(b + 1) * args.batch_size]
            print(f"\n--- Batch {b + 1}/{batches} ({len(names_b)} Bilder) ---", flush=True)
            images_stage, masks_stage, raw = tmp / "in_images", tmp / "in_masks", tmp / "out"
            for d in (images_stage, masks_stage, raw):
                shutil.rmtree(d, ignore_errors=True)
                d.mkdir(parents=True)
            prepared: dict[str, dict[str, Any]] = {}
            n_lama_jobs = 0
            t0 = time.perf_counter()
            for name in names_b:
                rgb, alpha = rgb_array(sources[name])
                h, w = rgb.shape[:2]
                boxes = anchors_by_image[name]
                prot = protected_mask(zones_by_image.get(name, []), w, h, PROTECTED_ZONE_MARGIN_PX)
                box_mask = boxes_union_mask(boxes, w, h) & ~prot
                plan = plan_regions(box_mask, max(0, merge_px))
                plan, _split_origin = split_plan_for_image(plan, rgb, alpha)    # same split as stage 1
                sel = {k: v for k, v in selection[name].items() if 1 <= k <= plan.count}
                if len(sel) != len(selection[name]):
                    print(f"WARN: {name}: Regionsnummern passen nicht zum Stage-1-Lauf (merge_px?)", file=sys.stderr)
                if not sel:
                    continue
                territory_sel = np.isin(plan.territory, list(sel))
                masks: dict[str, np.ndarray] = {}
                for m in methods:
                    mask = np.asarray(METHODS[m][1](rgb, alpha, boxes), dtype=bool) & box_mask & territory_sel
                    if mask.any():
                        masks[m] = mask
                stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in Path(name).stem)
                lama_masks = {m: mk for m, mk in masks.items() if METHODS[m][0] == "lama"}
                groups = group_identical_masks(lama_masks) if lama_masks else {}
                staged: dict[str, tuple[str, str]] = {}
                for m, mk in lama_masks.items():
                    if groups[m] != m:
                        continue
                    f_rgb, f_alpha = f"{stem}__{m.lower()}_rgb.png", f"{stem}__{m.lower()}_alpha.png"
                    Image.fromarray(rgb, mode="RGB").save(images_stage / f_rgb)
                    Image.fromarray(alpha, mode="L").convert("RGB").save(images_stage / f_alpha)
                    mi = Image.fromarray(mk.astype(np.uint8) * 255, mode="L")
                    mi.save(masks_stage / f_rgb)
                    mi.save(masks_stage / f_alpha)
                    staged[m] = (f_rgb, f_alpha)
                    n_lama_jobs += 1
                body_boxes = [bx for bx in boxes if str(bx.get("type", "")).casefold() == "body"]
                body_mask = (np.asarray(make_box_only_mask(w, h, body_boxes), dtype=bool) & box_mask) if body_boxes else None
                prepared[name] = {"stem": stem, "prot": prot, "plan": plan, "sel": sel, "masks": masks,
                                  "groups": groups, "staged": staged, "body": body_mask}
            seconds["staging"] += time.perf_counter() - t0
            if n_lama_jobs:
                t0 = time.perf_counter()
                code = _run_iopaint(iopaint, args.model, args.lama_device, images_stage, masks_stage, raw,
                                    args.lama_processes, tmp, project)
                seconds["lama"] += time.perf_counter() - t0
                if code != 0:
                    print(f"ERROR: IOPaint failed in batch {b + 1} with code {code}", file=sys.stderr)
                    return code or 1
            t0 = time.perf_counter()

            def _one(name: str) -> tuple[list[dict[str, Any]], int]:
                pr = prepared[name]
                return process_image(
                    name=name, stem=pr["stem"], source_path=sources[name], protected=pr["prot"], plan=pr["plan"],
                    selected=pr["sel"], masks=pr["masks"], groups=pr["groups"], staged=pr["staged"], raw=raw,
                    out_dir=output_dir, stage1_run=stage1_run, use_fill_repair=use_fill_repair, bank=bank,
                    donor_near=args.donor_near, sibling_bank=sibling_bank, body_mask=pr.get("body"),
                )

            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
                for rows, skipped in pool.map(_one, list(prepared)):
                    all_rows.extend(rows)
                    box_like_skipped += skipped
            seconds["measure_and_crops"] += time.perf_counter() - t0
            write_csv_with_union_fields(output_dir / "stage2_features.csv",
                                        sorted(all_rows, key=lambda r: (str(r["image"]).casefold(), int(r["region"]), str(r["method"]))))

    have = {(r["image"], int(r["region"])) for r in all_rows}
    no_alternative = sorted((img, k) for img, regs in selection.items() for k in regs if (img, k) not in have)
    (output_dir / "regions_without_alternative.csv").write_text(
        "image,region\n" + "".join(f"{img},{k}\n" for img, k in no_alternative), encoding="utf-8")
    bank_new = bank.save() if bank is not None else 0
    sibling_new = sibling_bank.save() if sibling_bank is not None else 0
    stats = {"donor_bank": str(bank.path) if bank is not None else None,
             "donor_bank_entries_at_start": len(bank) if bank is not None else 0, "donor_bank_entries_learned": bank_new,
             "sibling_bank": str(sibling_bank.path) if sibling_bank is not None else None,
             "sibling_bank_entries_at_start": len(sibling_bank) if sibling_bank is not None else 0,
             "sibling_bank_entries_learned": sibling_new,
             "images": len(names), "regions": n_regions, "methods": methods, "rows": len(all_rows),
             "regions_without_alternative": len(no_alternative),
             "select": args.select, "region_merge_px": merge_px, "method_tiles_skipped_box_like": box_like_skipped,
             "seconds": {k: round(v, 2) for k, v in seconds.items()}, "total_seconds": round(time.perf_counter() - t_run, 2)}
    (output_dir / "stage2_run_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\n=== STAGE 2 COMPLETE ===\nRegionen: {n_regions} | Ergebniszeilen: {len(all_rows)} | "
          f"als Box-only-Ersatz verworfen (Maske >= {int(BOX_LIKE_COVERAGE * 100)} % der Box): {box_like_skipped}")
    if no_alternative:
        print(f"{len(no_alternative)} Regionen ohne Alternative (alle Methoden Box-ähnlich): regions_without_alternative.csv "
              "- brauchen andere Methoden (z. B. YOLO-Masken / neue Füllung)")
    print(f"Ranker bauen:  uv run python build_ranker.py {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
