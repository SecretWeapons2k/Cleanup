#!/usr/bin/env python3
"""Audit when YOLO masks are suitable versus box-only masks, without training.

The script reruns the existing YOLO segmentation model (inference only) so it can
inspect individual instance masks, not just the already-unioned mask images.
For each image it:
  * checks whether every red/body anchor box intersects at least one YOLO mask;
  * estimates the distance of each YOLO instance mask to the nearest OCR anchor;
  * trims each YOLO instance mask pixel-by-pixel to a distance band around OCR boxes;
  * saves a box-only baseline and visual overlays for side-by-side inspection.

It never modifies source images, anchors.csv, raw/, cleaned/, or existing tests.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from yolo_mask_auto_select_modules.config import (
    ALPHA_VISIBILITY_THRESHOLD,
    BODY_BOX_MIN_DETAIL_ENERGY_COVERAGE,
    BODY_BOX_MIN_HIGH_CONTRAST_COVERAGE,
    BODY_BOX_MIN_HIGH_CONTRAST_PIXELS,
    BODY_BOX_MIN_OVERLAP_FRACTION,
    BODY_BOX_MIN_OVERLAP_PIXELS,
    OUTLINE_LEAK_DETAIL_THRESHOLD,
)
from yolo_mask_auto_select_modules.scoring import ink_strength_map

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULT_IMAGES_DIR = Path.home() / "Downloads/yabai proj/yabaiproject - Copy/mit_kanji"
DEFAULT_WEIGHTS = Path.home() / ".cache/cleanup-models/manga-inpainter-best.pt"
DISTANCE_SWEEP_DEFAULT = "0,4,8,12,16"
CONFIDENCE_THRESHOLDS = (0.05, 0.10)


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    p.add_argument("--anchors", type=Path, default=project / "anchors.csv")
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--output", type=Path, default=project / "yolo_mask_routing_audit")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--min-conf", type=float, default=0.025,
                   help="Inference floor; keep <= 0.05 for the 0.05 test")
    p.add_argument("--dilation", type=int, default=2,
                   help="Dilate each instance mask before matching/filtering")
    p.add_argument("--distances", default=DISTANCE_SWEEP_DEFAULT,
                   help="Comma-separated max distance in pixels from any anchor box")
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--limit", type=int, default=0, help="Optional image limit for a pilot run")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def norm_name(value: str) -> str:
    return " ".join(value.casefold().replace("_", " ").replace("-", " ").split())


def requested_class(name: str) -> bool:
    normalized = norm_name(name)
    return normalized in {
        "text", "text region", "dialogue text", "printed text", "caption",
        "sfx", "sound effect", "sound effects", "onomatopoeia",
    }


def image_name_from(row: dict[str, str]) -> str:
    return Path((row.get("image") or "").strip()).name


def read_anchors(path: Path) -> tuple[dict[str, list[dict[str, Any]]], set[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"anchors CSV not found: {path}")
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"image", "type", "x1", "y1", "x2", "y2"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError("CSV missing columns: " + ", ".join(sorted(missing)))
        for row_number, row in enumerate(reader, start=2):
            name = image_name_from(row)
            if not name:
                continue
            try:
                x1, y1, x2, y2 = [float(row[k]) for k in ("x1", "y1", "x2", "y2")]
                if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                print(f"WARN: invalid box ignored at CSV line {row_number}: {name}", file=sys.stderr)
                continue
            label = (row.get("type") or "").strip().casefold()
            by_image[name].append({
                "type": label, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "anchor_id": (row.get("anchor_id") or "").strip(),
                "row": (row.get("row") or "").strip(),
                "col": (row.get("col") or "").strip(),
                "csv_line": row_number,
            })
    if not by_image:
        raise ValueError(f"No valid anchor boxes found in {path}")
    return by_image, set(by_image)


def discover_images(images_dir: Path, known_names: set[str]) -> list[Path]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"source images directory not found: {images_dir}")
    found: dict[str, list[Path]] = defaultdict(list)
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.name in known_names:
            found[p.name].append(p)
    duplicates = {name: vals for name, vals in found.items() if len(vals) > 1}
    if duplicates:
        examples = "; ".join(f"{name}: {len(vals)} files" for name, vals in list(duplicates.items())[:8])
        raise ValueError("Duplicate source-image basenames; refusing to guess: " + examples)
    paths = [vals[0] for _, vals in found.items()]
    return sorted(paths, key=lambda p: (p.name.casefold(), str(p)))


def rect_bounds(box: dict[str, Any], w: int, h: int) -> tuple[int, int, int, int] | None:
    x1 = max(0, min(w - 1, math.floor(box["x1"])))
    y1 = max(0, min(h - 1, math.floor(box["y1"])))
    x2 = max(0, min(w - 1, math.ceil(box["x2"])))
    y2 = max(0, min(h - 1, math.ceil(box["y2"])))
    if x2 < x1 or y2 < y1:
        return None
    return x1, y1, x2, y2


def draw_box_mask(size: tuple[int, int], boxes: list[dict[str, Any]]) -> np.ndarray:
    w, h = size
    mask = np.zeros((h, w), np.uint8)
    for box in boxes:
        bounds = rect_bounds(box, w, h)
        if bounds:
            cv2.rectangle(mask, (bounds[0], bounds[1]), (bounds[2], bounds[3]), 255, -1)
    return mask


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)


def polygon_mask(poly: Any, w: int, h: int, dilation: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.asarray(poly, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
        return mask
    pts = np.rint(pts).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    cv2.fillPoly(mask, [pts], 255)
    return dilate(mask, dilation)


def min_distance_to_boxes(instance_mask: np.ndarray, box_distance: np.ndarray) -> float:
    ys, xs = np.nonzero(instance_mask)
    if len(xs) == 0:
        return float("inf")
    return float(np.min(box_distance[ys, xs]))


def overlap_with_box(mask: np.ndarray, box: dict[str, Any]) -> int:
    h, w = mask.shape
    bounds = rect_bounds(box, w, h)
    if not bounds:
        return 0
    x1, y1, x2, y2 = bounds
    return int(np.count_nonzero(mask[y1:y2 + 1, x1:x2 + 1]))


def body_box_coverage_metrics(
    candidate_mask: np.ndarray,
    source_detail: np.ndarray,
    source_alpha: np.ndarray,
    box: dict[str, Any],
) -> dict[str, Any]:
    """Measure meaningful YOLO coverage of a body OCR box, not just any overlap.

    Ratios are calculated over visible source pixels in the box. Detail-energy
    coverage measures how much original local contrast is actually covered;
    high-contrast coverage adds a check for missed outlines/strokes. Boxes with
    no measurable contrast cannot fail the contrast-ratio test, but still need a
    non-trivial absolute mask overlap.
    """
    h, w = candidate_mask.shape
    bounds = rect_bounds(box, w, h)
    if bounds is None:
        return {
            "raw_overlap_nonzero": False, "overlap_pixels": 0,
            "visible_box_pixels": 0, "minimum_required_overlap_pixels": BODY_BOX_MIN_OVERLAP_PIXELS,
            "box_overlap_fraction": 0.0, "detail_energy_coverage_ratio": 0.0,
            "high_contrast_coverage_ratio": float("nan"), "high_contrast_pixels": 0,
            "covered_high_contrast_pixels": 0, "combined_detail_coverage_score": 0.0,
            "meaningfully_covered": False,
        }
    x1, y1, x2, y2 = bounds
    region_alpha = source_alpha[y1:y2 + 1, x1:x2 + 1]
    visible = region_alpha > ALPHA_VISIBILITY_THRESHOLD
    mask_crop = candidate_mask[y1:y2 + 1, x1:x2 + 1] > 0
    covered = visible & mask_crop
    detail_crop = source_detail[y1:y2 + 1, x1:x2 + 1]
    visible_pixels = int(np.count_nonzero(visible))
    overlap_pixels = int(np.count_nonzero(covered))
    minimum_overlap = max(
        int(BODY_BOX_MIN_OVERLAP_PIXELS),
        int(math.ceil(visible_pixels * BODY_BOX_MIN_OVERLAP_FRACTION)),
    )
    box_detail_energy = float(np.sum(detail_crop[visible]))
    covered_detail_energy = float(np.sum(detail_crop[covered]))
    if box_detail_energy > 1e-6:
        detail_coverage = float(np.clip(covered_detail_energy / box_detail_energy, 0.0, 1.0))
    else:
        # There is no contrast energy to lose, so do not invent a detail failure.
        detail_coverage = 1.0

    high_contrast = (detail_crop >= OUTLINE_LEAK_DETAIL_THRESHOLD) & visible
    high_count = int(np.count_nonzero(high_contrast))
    covered_high = int(np.count_nonzero(high_contrast & covered))
    high_coverage = (float(covered_high / high_count) if high_count else float("nan"))

    enough_absolute_overlap = overlap_pixels >= minimum_overlap
    if high_count >= BODY_BOX_MIN_HIGH_CONTRAST_PIXELS:
        # Both ratios must pass: good average energy coverage must not hide a
        # largely omitted high-contrast outline, and vice versa.
        meaningful = (
            enough_absolute_overlap
            and detail_coverage >= BODY_BOX_MIN_DETAIL_ENERGY_COVERAGE
            and high_coverage >= BODY_BOX_MIN_HIGH_CONTRAST_COVERAGE
        )
        combined = 0.65 * detail_coverage + 0.35 * high_coverage
    else:
        meaningful = enough_absolute_overlap and detail_coverage >= BODY_BOX_MIN_DETAIL_ENERGY_COVERAGE
        combined = detail_coverage

    return {
        "raw_overlap_nonzero": overlap_pixels > 0,
        "overlap_pixels": overlap_pixels,
        "visible_box_pixels": visible_pixels,
        "minimum_required_overlap_pixels": minimum_overlap,
        "box_overlap_fraction": float(overlap_pixels / max(visible_pixels, 1)),
        "detail_energy_coverage_ratio": detail_coverage,
        "high_contrast_coverage_ratio": high_coverage,
        "high_contrast_pixels": high_count,
        "covered_high_contrast_pixels": covered_high,
        "combined_detail_coverage_score": float(combined),
        "meaningfully_covered": bool(meaningful),
    }


def build_overlay(rgb: np.ndarray, candidate: np.ndarray, boxes: list[dict[str, Any]]) -> Image.Image:
    base = rgb.astype(np.float32).copy()
    selected = candidate > 0
    tint = np.array([255.0, 35.0, 185.0], dtype=np.float32)
    base[selected] = base[selected] * 0.48 + tint * 0.52
    im = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(im)
    w, h = im.size
    for b in boxes:
        bounds = rect_bounds(b, w, h)
        if not bounds:
            continue
        color = (225, 45, 45) if b["type"] == "body" else ((170, 90, 215) if b["type"] == "header" else (255, 170, 0))
        draw.rectangle(bounds, outline=color, width=2)
    return im


def choose_output_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    stamp = time.strftime("run_%Y%m%d_%H%M%S")
    if path.exists() and any(path.iterdir()):
        candidate = path / stamp
        suffix = 1
        while candidate.exists():
            candidate = path / f"{stamp}_{suffix}"
            suffix += 1
        return candidate
    return path


def save_contact_sheets(images: list[tuple[str, Path]], output: Path, label: str, per_sheet: int = 16) -> None:
    if not images:
        return
    output.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    for start in range(0, len(images), per_sheet):
        chunk = images[start:start + per_sheet]
        cards: list[Image.Image] = []
        for name, path in chunk:
            with Image.open(path) as source:
                im = source.convert("RGB")
            scale = min(1.0, 320 / max(im.size))
            if scale < 1:
                im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.Resampling.LANCZOS)
            card = Image.new("RGB", (im.width, im.height + 25), "white")
            card.paste(im, (0, 25))
            ImageDraw.Draw(card).text((5, 5), name[:42], font=font, fill=(10, 10, 10))
            cards.append(card)
        cw = max(c.width for c in cards)
        ch = max(c.height for c in cards)
        cols = 4
        rows = (len(cards) + cols - 1) // cols
        gap = margin = 8
        sheet = Image.new("RGB", (margin * 2 + cols * cw + (cols - 1) * gap,
                                   margin * 2 + 18 + rows * ch + (rows - 1) * gap), (225, 225, 225))
        d = ImageDraw.Draw(sheet)
        d.text((margin, 2), label, fill=(0, 0, 0), font=font)
        for i, card in enumerate(cards):
            x = margin + (i % cols) * (cw + gap)
            y = margin + 18 + (i // cols) * (ch + gap)
            sheet.paste(card, (x, y))
        sheet.save(output / f"sheet_{start // per_sheet + 1:03d}.jpg", quality=88)


def main() -> int:
    args = parse_args()
    anchors_path = args.anchors.expanduser().resolve()
    images_dir = args.images_dir.expanduser().resolve()
    weights_path = args.weights.expanduser().resolve()
    if args.imgsz < 320 or args.min_conf < 0 or args.min_conf > 0.05:
        print("ERROR: --imgsz must be >=320 and --min-conf must be in [0, 0.05]", file=sys.stderr)
        return 2
    if args.dilation < 0 or args.limit < 0:
        print("ERROR: --dilation and --limit must be >=0", file=sys.stderr)
        return 2
    try:
        distances = sorted({int(x.strip()) for x in args.distances.split(",") if x.strip() != ""})
        if not distances or distances[0] < 0:
            raise ValueError
    except ValueError:
        print("ERROR: --distances must be comma-separated non-negative integers, e.g. 0,4,8,12,16", file=sys.stderr)
        return 2

    try:
        anchors_by_image, known_names = read_anchors(anchors_path)
        images = discover_images(images_dir, known_names)
        if args.limit:
            images = images[:args.limit]
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if not images:
        print("ERROR: no source images matched filenames in anchors.csv", file=sys.stderr)
        return 2

    print(f"Anchors CSV: {anchors_path}")
    print(f"Source images: {images_dir}")
    print(f"Matched images: {len(images)} / {len(known_names)} anchor image names")
    print(f"Checkpoint: {weights_path}")
    print(f"Confidence tests: {CONFIDENCE_THRESHOLDS}; mask proximity thresholds: {distances}px; dilation: {args.dilation}px")
    print("This is inference-only; no training and no LaMa run.")
    if args.dry_run:
        if weights_path.is_file():
            print(f"Checkpoint exists ({weights_path.stat().st_size / 1024 / 1024:.1f} MiB).")
        else:
            print("WARNING: checkpoint not found.")
        print("DRY RUN: no inference or output files created.")
        return 0
    if not weights_path.is_file():
        print("ERROR: YOLO checkpoint not found. Pass --weights explicitly.", file=sys.stderr)
        return 3

    try:
        from ultralytics import YOLO
        import torch
        device = "cuda:0" if args.device == "cuda" else ("cpu" if args.device == "cpu" else ("cuda:0" if torch.cuda.is_available() else "cpu"))
        model = YOLO(str(weights_path))
        names_raw = getattr(model, "names", {})
        if isinstance(names_raw, dict):
            class_names = {int(k): str(v) for k, v in names_raw.items()}
        else:
            class_names = {i: str(v) for i, v in enumerate(names_raw)}
        print(f"Loaded model; classes: {class_names}; device: {device}")
    except Exception as exc:
        print(f"ERROR loading YOLO: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 4

    output = choose_output_dir(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report_rows: list[dict[str, Any]] = []
    box_rows: list[dict[str, Any]] = []
    instance_rows: list[dict[str, Any]] = []
    overlays_by_distance: dict[int, list[tuple[str, Path]]] = {d: [] for d in distances}
    box_only_overlays: list[tuple[str, Path]] = []
    error_count = 0

    for image_idx, image_path in enumerate(images, start=1):
        name = image_path.name
        print(f"[{image_idx}/{len(images)}] {name}", flush=True)
        try:
            with Image.open(image_path) as source:
                rgba = source.convert("RGBA")
                rgb = np.asarray(rgba.convert("RGB"), dtype=np.uint8)
                source_alpha = np.asarray(rgba.getchannel("A"), dtype=np.uint8)
            h, w = rgb.shape[:2]
            # A hue-independent detail map lets routing check whether YOLO
            # covers substantial text-like contrast inside each Body OCR box.
            source_detail = ink_strength_map(rgb)
            boxes = anchors_by_image.get(name, [])
            body_boxes = [b for b in boxes if b["type"] == "body"]
            header_boxes = [b for b in boxes if b["type"] == "header"]
            all_box_mask = draw_box_mask((w, h), boxes)
            header_mask = draw_box_mask((w, h), header_boxes)
            box_mask_dir = output / "box_only" / "masks"
            box_overlay_dir = output / "box_only" / "overlays"
            box_mask_dir.mkdir(parents=True, exist_ok=True)
            box_overlay_dir.mkdir(parents=True, exist_ok=True)
            box_mask_path = box_mask_dir / f"{Path(name).stem}.png"
            box_overlay_path = box_overlay_dir / f"{Path(name).stem}.jpg"
            Image.fromarray(all_box_mask, mode="L").save(box_mask_path)
            build_overlay(rgb, all_box_mask, boxes).save(box_overlay_path, quality=90)
            box_only_overlays.append((name, box_overlay_path))
            # Exact Euclidean distance (in pixels) to the nearest anchor-box pixel.
            # Pixels inside any OCR box have distance 0.
            distance_to_anchor = cv2.distanceTransform(
                (all_box_mask == 0).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
            )

            prediction = model.predict(
                source=cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), imgsz=args.imgsz,
                conf=args.min_conf, iou=0.70, device=device, retina_masks=True, verbose=False,
            )
            if not prediction:
                raise RuntimeError("model returned no prediction result")
            result = prediction[0]
            result_boxes = getattr(result, "boxes", None)
            result_masks = getattr(result, "masks", None)
            confs = result_boxes.conf.detach().cpu().tolist() if result_boxes is not None and len(result_boxes) else []
            cls_ids = result_boxes.cls.detach().cpu().tolist() if result_boxes is not None and len(result_boxes) else []
            polygons = list(result_masks.xy) if result_masks is not None else []
            if len(polygons) != len(confs):
                if len(confs) or len(polygons):
                    raise RuntimeError(f"detection/mask count mismatch: {len(confs)} detections, {len(polygons)} masks")
            instances: list[dict[str, Any]] = []
            for ix, (conf, cls_id, poly) in enumerate(zip(confs, cls_ids, polygons)):
                class_name = class_names.get(int(cls_id), f"class_{int(cls_id)}")
                if not requested_class(class_name):
                    continue
                imask = polygon_mask(poly, w, h, args.dilation)
                pixels = int(np.count_nonzero(imask))
                if pixels == 0:
                    continue
                dist = min_distance_to_boxes(imask, distance_to_anchor)
                hits = []
                for bidx, box in enumerate(body_boxes):
                    overlap_px = overlap_with_box(imask, box)
                    if overlap_px > 0:
                        hits.append((bidx, overlap_px))
                instances.append({
                    "index": ix, "class_name": class_name, "confidence": float(conf),
                    "mask": imask, "pixels": pixels, "distance": dist, "body_hits": hits,
                    "poly": poly,
                })

            # Evaluate both confidence choices against the same inference.
            # Distance filtering is PIXEL-WISE: within each instance, retain only
            # pixels whose distance to the nearest OCR box is <= d. This trims
            # distant parts of a mask rather than accepting/rejecting the whole instance.
            distance_allowed = {d: (distance_to_anchor <= (d + 1e-6)) for d in distances}
            for threshold in CONFIDENCE_THRESHOLDS:
                eligible = [inst for inst in instances if inst["confidence"] + 1e-9 >= threshold]
                # Build the eligible union once. A single pixel overlap no longer
                # counts as a successful body-box detection: assess original detail
                # coverage and high-contrast stroke coverage within each box.
                untrimmed_union = np.zeros((h, w), dtype=np.uint8)
                for inst in eligible:
                    untrimmed_union = cv2.bitwise_or(untrimmed_union, inst["mask"])
                pixels_before_clip = int(np.count_nonzero(untrimmed_union))
                body_coverage = [
                    body_box_coverage_metrics(untrimmed_union, source_detail, source_alpha, body)
                    for body in body_boxes
                ]
                missing = [i for i, coverage in enumerate(body_coverage)
                           if not coverage["meaningfully_covered"]]
                raw_hit_count = sum(int(coverage["raw_overlap_nonzero"]) for coverage in body_coverage)
                if not body_boxes:
                    route = "MANUAL_REVIEW_NO_BODY_BOXES"
                elif missing:
                    route = "BOX_MASK_FALLBACK_WEAK_COVERAGE"
                else:
                    route = "YOLO_CANDIDATE"

                for d in distances:
                    allowed_pixels = distance_allowed[d]
                    yolo_union = np.zeros((h, w), dtype=np.uint8)
                    partially_clipped_instances = 0
                    clipped_away_instances = 0
                    for inst in eligible:
                        original_mask = inst["mask"]
                        clipped_mask = cv2.bitwise_and(original_mask, allowed_pixels.astype(np.uint8) * 255)
                        original_pixels = int(np.count_nonzero(original_mask))
                        clipped_pixels = int(np.count_nonzero(clipped_mask))
                        if clipped_pixels == 0:
                            clipped_away_instances += 1
                        elif clipped_pixels < original_pixels:
                            partially_clipped_instances += 1
                        yolo_union = cv2.bitwise_or(yolo_union, clipped_mask)
                    candidate = cv2.bitwise_or(yolo_union, header_mask)
                    total_yolo_pixels = int(np.count_nonzero(yolo_union))
                    pixels_removed_by_clip = max(0, pixels_before_clip - total_yolo_pixels)
                    yolo_inside_boxes = int(np.count_nonzero((yolo_union > 0) & (all_box_mask > 0)))
                    yolo_outside_boxes = total_yolo_pixels - yolo_inside_boxes
                    mask_path = ""
                    overlay_path = ""
                    # Only persist the main 0.05 candidates; 0.10 is a metrics-only control.
                    if threshold == 0.05:
                        output_mask_dir = output / "conf_005" / f"distance_{d:02d}" / "masks"
                        output_overlay_dir = output / "conf_005" / f"distance_{d:02d}" / "overlays"
                        output_mask_dir.mkdir(parents=True, exist_ok=True)
                        output_overlay_dir.mkdir(parents=True, exist_ok=True)
                        mask_file = output_mask_dir / f"{Path(name).stem}.png"
                        overlay_file = output_overlay_dir / f"{Path(name).stem}.jpg"
                        Image.fromarray(candidate, mode="L").save(mask_file)
                        build_overlay(rgb, candidate, boxes).save(overlay_file, quality=90)
                        mask_path = str(mask_file)
                        overlay_path = str(overlay_file)
                        overlays_by_distance[d].append((name, overlay_file))

                    report_rows.append({
                        "image": name, "confidence_threshold": f"{threshold:.2f}", "max_distance_px": d,
                        "route_suggestion": route,
                        "body_boxes": len(body_boxes),
                        "body_boxes_hit_by_any_eligible_yolo_mask": raw_hit_count,
                        "body_boxes_meaningfully_covered_by_yolo_mask": len(body_boxes) - len(missing),
                        "body_boxes_missing_or_weakly_covered": len(missing),
                        "body_boxes_missing_yolo_mask": len(missing),
                        "missing_body_box_indices_1based": ";".join(str(i + 1) for i in missing),
                        "weak_coverage_body_box_indices_1based": ";".join(str(i + 1) for i in missing),
                        "minimum_required_body_box_detail_coverage": BODY_BOX_MIN_DETAIL_ENERGY_COVERAGE,
                        "minimum_required_body_box_high_contrast_coverage": BODY_BOX_MIN_HIGH_CONTRAST_COVERAGE,
                        "eligible_instances_before_clip": len(eligible),
                        "instances_with_pixels_after_clip": len(eligible) - clipped_away_instances,
                        "instances_partially_clipped": partially_clipped_instances,
                        "instances_clipped_away_completely": clipped_away_instances,
                        "yolo_mask_pixels_before_distance_clip_union": pixels_before_clip,
                        "yolo_pixels_removed_by_distance_clip_union": pixels_removed_by_clip,
                        "yolo_mask_pixels_after_distance_clip_union": total_yolo_pixels,
                        "yolo_pixels_inside_any_anchor_box": yolo_inside_boxes,
                        "yolo_pixels_outside_anchor_boxes": yolo_outside_boxes,
                        "yolo_pixels_outside_box_fraction": f"{yolo_outside_boxes / max(1, total_yolo_pixels):.6f}",
                        "candidate_mask_pixels_including_header_boxes": int(np.count_nonzero(candidate)),
                        "candidate_mask_path": str(mask_path), "overlay_path": str(overlay_path),
                    })

                # Per-box detail at each confidence threshold. `covered_by_yolo`
                # now means meaningful coverage, while raw overlap is preserved
                # separately to expose the former one-pixel false positive.
                for bidx, body in enumerate(body_boxes):
                    coverage = body_coverage[bidx]
                    total_overlap = sum(overlap_with_box(inst["mask"], body) for inst in eligible)
                    matching_confidences = [inst["confidence"] for inst in eligible if overlap_with_box(inst["mask"], body) > 0]
                    box_rows.append({
                        "image": name, "confidence_threshold": f"{threshold:.2f}",
                        "body_box_index_1based": bidx + 1,
                        "anchor_id": body["anchor_id"], "row": body["row"], "col": body["col"],
                        "box_x1": body["x1"], "box_y1": body["y1"], "box_x2": body["x2"], "box_y2": body["y2"],
                        "covered_by_yolo": int(coverage["meaningfully_covered"]),
                        "raw_overlap_nonzero": int(coverage["raw_overlap_nonzero"]),
                        "overlap_pixel_count_union": coverage["overlap_pixels"],
                        "overlap_pixel_count_sum": total_overlap,
                        "visible_box_pixels": coverage["visible_box_pixels"],
                        "minimum_required_overlap_pixels": coverage["minimum_required_overlap_pixels"],
                        "box_overlap_fraction": f"{coverage['box_overlap_fraction']:.6f}",
                        "detail_energy_coverage_ratio": f"{coverage['detail_energy_coverage_ratio']:.6f}",
                        "high_contrast_coverage_ratio": (f"{coverage['high_contrast_coverage_ratio']:.6f}" if math.isfinite(coverage["high_contrast_coverage_ratio"]) else ""),
                        "high_contrast_pixels": coverage["high_contrast_pixels"],
                        "covered_high_contrast_pixels": coverage["covered_high_contrast_pixels"],
                        "combined_detail_coverage_score": f"{coverage['combined_detail_coverage_score']:.6f}",
                        "coverage_status": "meaningful" if coverage["meaningfully_covered"] else "weak_or_missing",
                        "matching_instance_confidences": ";".join(f"{v:.4f}" for v in matching_confidences),
                    })

            for inst in instances:
                for threshold in CONFIDENCE_THRESHOLDS:
                    if inst["confidence"] + 1e-9 < threshold:
                        continue
                    instance_rows.append({
                        "image": name, "confidence_threshold": f"{threshold:.2f}", "instance_index": inst["index"],
                        "class_name": inst["class_name"], "confidence": f"{inst['confidence']:.6f}",
                        "mask_pixels_after_dilation": inst["pixels"],
                        "minimum_distance_to_any_anchor_px": f"{inst['distance']:.3f}",
                        "intersected_body_box_indices_1based": ";".join(str(i + 1) for i, _ in inst["body_hits"]),
                        "intersected_body_box_count": len(inst["body_hits"]),
                    })
        except Exception as exc:
            error_count += 1
            print(f"  ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            report_rows.append({"image": name, "status": f"error: {type(exc).__name__}: {exc}"})

    def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        keys = list(dict.fromkeys(key for row in rows for key in row.keys()))
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    write_csv(output / "routing_report.csv", report_rows)
    write_csv(output / "body_box_coverage.csv", box_rows)
    write_csv(output / "instances.csv", instance_rows)
    save_contact_sheets(box_only_overlays, output / "box_only" / "contact_sheets", "Box-only masks (all body + header rectangles)")
    for d, entries in overlays_by_distance.items():
        save_contact_sheets(entries, output / "conf_005" / f"distance_{d:02d}" / "contact_sheets", f"YOLO conf 0.05 pixelwise trim: retain mask pixels <= {d}px from nearest anchor box + headers")

    # Aggregate a quick summary for stdout/README.
    main_rows = [r for r in report_rows if r.get("confidence_threshold") == "0.05"]
    unique_images = len({r["image"] for r in main_rows})
    by_d: list[str] = []
    for d in distances:
        subset = [r for r in main_rows if int(r["max_distance_px"]) == d]
        fallbacks = len({r["image"] for r in subset if r.get("route_suggestion") == "BOX_MASK_FALLBACK"})
        avg_after = float(np.mean([int(r["instances_with_pixels_after_clip"]) for r in subset])) if subset else 0.0
        avg_removed = float(np.mean([int(r["yolo_pixels_removed_by_distance_clip_union"]) for r in subset])) if subset else 0.0
        by_d.append(f"{d}px: box fallback {fallbacks}/{len(subset)}, mean instances with pixels after clip {avg_after:.2f}, mean union pixels clipped {avg_removed:.0f}")

    readme = [
        "YOLO mask routing audit (inference only; no training and no LaMa).",
        f"Images source: {images_dir}", f"Anchors CSV: {anchors_path}", f"Weights: {weights_path}",
        f"Images processed: {unique_images}", f"Distance sweep: {distances}; dilation: {args.dilation}px",
        f"Body-box meaningful coverage thresholds: overlap >= {BODY_BOX_MIN_OVERLAP_PIXELS}px / {BODY_BOX_MIN_OVERLAP_FRACTION:.2%} of visible box; detail energy >= {BODY_BOX_MIN_DETAIL_ENERGY_COVERAGE:.0%}; high-contrast pixels >= {BODY_BOX_MIN_HIGH_CONTRAST_COVERAGE:.0%} when present.",
        "Route rule: every type=body box must have a non-trivial YOLO overlap and sufficient coverage of source local-contrast energy and high-contrast pixels; one-pixel hits do not count. Weak coverage routes to BOX_MASK_FALLBACK.",
        "Proximity rule: pixel-wise trim each eligible YOLO mask. Keep only mask pixels whose distance to the nearest anchor-box pixel is <= the configured limit. Always union header rectangles in full.",
        "This clips distant parts of a mask, even when another part of the same instance is close to a box; instances are not accepted/rejected as indivisible objects.",
        "Use routing_report.csv for per-image summary, body_box_coverage.csv for raw overlap and detail-coverage ratios per body box, instances.csv for per-instance confidence/minimum distance, and contact sheets to visually tune clipping.",
        "This is a heuristic report; the best distance must be chosen from the contact sheets and LaMa results, not inferred from overlap statistics alone.",
        "Never modified source images, anchors.csv, raw/, cleaned/, or existing audit results.",
        "",
        *by_d,
    ]
    (output / "README.txt").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print("\n=== ROUTING AUDIT COMPLETE ===")
    print(f"Processed images with report rows: {unique_images}; errors: {error_count}")
    print(f"Output: {output}")
    print(f"Per-image report: {output / 'routing_report.csv'}")
    print(f"Per-body-box details: {output / 'body_box_coverage.csv'}")
    print(f"Per-instance details: {output / 'instances.csv'}")
    print("Per-pixel distance-clip summary (conf=0.05):")
    for line in by_d:
        print("  " + line)
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
