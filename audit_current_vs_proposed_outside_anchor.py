#!/usr/bin/env python3
"""Read-only direct comparison of current vs proposed candidate outside OCR anchors.

Uses only saved original images, candidate images, anchor CSV, and existing audit CSVs.
No inference; does not modify scoring.py, winner_report.csv, or candidate images.
"""
from __future__ import annotations

import csv
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

try:
    import cv2
    import numpy as np
    import audit_anchor_color_rule_holdout as audit
except Exception as exc:
    print(f"FEHLER beim Import: {type(exc).__name__}: {exc}")
    print("Bitte im Cleanup-Projekt mit uv run python ausführen.")
    raise

AUDIT_DIR = audit.DEBUG_DIR / "counterfactual_color_rule_full_population_20261024_214800"
SWEEP_DIR = AUDIT_DIR / "sensitivity_sweep_20261024_220012"
OUTSIDE_DIR = SWEEP_DIR / "outside_anchor_preservation_20261024_220824"
CHANGED_CSV = AUDIT_DIR / "full_population_changed_review.csv"
STABILITY_CSV = SWEEP_DIR / "sensitivity_image_stability.csv"
PADS = (0, 2, 4, 8)
THRESHOLDS = (8.0, 12.0, 20.0)
MIN_REDUCTION_PCT = 90.0
STABLE_MIN_SCENARIOS = 18


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Benötigte Datei fehlt: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row.keys()))
    if not fields:
        fields = ["image"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def as_int(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, "")))
    except (TypeError, ValueError):
        return default


def norm_stem(name: str) -> str:
    return Path(name.strip()).stem.lower()


def load_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Bild nicht lesbar: {path}")
    return img


def anchor_mask(anchors: list[dict[str, str]], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    # Candidate images are compared in the original image's pixel coordinate system.
    for row in anchors:
        box = audit.clipped_box(row, width, height)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        mask[y1:y2, x1:x2] = 255
    return mask


def lab_units(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    # Approximate CIE L*a*b* units from OpenCV's 8-bit encoding.
    lab[:, :, 0] *= 100.0 / 255.0
    lab[:, :, 1:] -= 128.0
    return lab


def direct_diff_metrics(current_bgr: np.ndarray, proposed_bgr: np.ndarray,
                        mask: np.ndarray) -> dict[str, object]:
    delta = np.linalg.norm(lab_units(current_bgr) - lab_units(proposed_bgr), axis=2)
    result: dict[str, object] = {}
    for pad in PADS:
        region = mask
        if pad:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
            region = cv2.dilate(mask, kernel, iterations=1)
        inside = region > 0
        outside = ~inside
        result[f"outside_pixels_pad{pad}"] = int(np.count_nonzero(outside))
        result[f"anchor_buffer_area_pct_pad{pad}"] = round(100.0 * float(np.mean(inside)), 5)
        for label, selected in (("outside", outside), ("inside", inside)):
            vals = delta[selected]
            if vals.size:
                for threshold in THRESHOLDS:
                    result[f"{label}_gt{int(threshold)}_pct_pad{pad}"] = round(
                        100.0 * float(np.count_nonzero(vals > threshold)) / vals.size, 5
                    )
                result[f"{label}_delta_mean_pad{pad}"] = round(float(np.mean(vals)), 5)
                result[f"{label}_delta_p95_pad{pad}"] = round(float(np.percentile(vals, 95)), 5)
            else:
                for threshold in THRESHOLDS:
                    result[f"{label}_gt{int(threshold)}_pct_pad{pad}"] = 0.0
                result[f"{label}_delta_mean_pad{pad}"] = 0.0
                result[f"{label}_delta_p95_pad{pad}"] = 0.0
    result["whole_image_gt12_pct"] = round(100.0 * float(np.count_nonzero(delta > 12.0)) / delta.size, 5)
    result["whole_image_delta_mean"] = round(float(np.mean(delta)), 5)
    return result


def make_panel(img: np.ndarray, boxes: list[tuple[int, int, int, int]], title: str,
               tile_w: int = 450, tile_h: int = 210) -> np.ndarray:
    marked = img.copy()
    h, w = marked.shape[:2]
    for x1, y1, x2, y2 in boxes:
        xx1, yy1 = max(0, x1), max(0, y1)
        xx2, yy2 = min(w - 1, x2 - 1), min(h - 1, y2 - 1)
        if xx2 >= xx1 and yy2 >= yy1:
            cv2.rectangle(marked, (xx1, yy1), (xx2, yy2), (0, 0, 255), 1, cv2.LINE_AA)
    scale = min((tile_w - 12) / max(1, w), (tile_h - 34) / max(1, h))
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    thumb = cv2.resize(marked, (nw, nh), interpolation=interp)
    panel = np.full((tile_h, tile_w, 3), 248, dtype=np.uint8)
    cv2.putText(panel, title[:60], (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (25, 25, 25), 1, cv2.LINE_AA)
    x0, y0 = (tile_w - nw) // 2, 26 + (tile_h - 26 - nh) // 2
    panel[y0:y0 + nh, x0:x0 + nw] = thumb
    return panel


def main() -> None:
    for path in (AUDIT_DIR, SWEEP_DIR, OUTSIDE_DIR, audit.ORIGINAL_DIR, audit.CANDIDATE_DIR):
        if not path.exists():
            raise FileNotFoundError(f"Benötigter vorhandener Pfad fehlt: {path}")

    changed = read_csv(CHANGED_CSV)
    stability = {norm_stem(r.get("image", "")): r for r in read_csv(STABILITY_CSV)}
    anchors_by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(audit.find_anchor_csv()):
        if row.get("image", "").strip():
            anchors_by_image[norm_stem(row["image"])].append(row)

    original_map: dict[str, list[Path]] = defaultdict(list)
    for p in audit.ORIGINAL_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in audit.IMAGE_EXTS:
            original_map[p.stem.lower()].append(p)

    targets = []
    for row in changed:
        st = stability.get(norm_stem(row.get("image", "")), {})
        if as_int(st, "stable_alternative_18_of_24") < 1:
            continue
        if as_float(row, "source_color_gt16_reduction_pct") < MIN_REDUCTION_PCT:
            continue
        targets.append(row)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTSIDE_DIR / f"direct_current_proposed_delta_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)

    metrics: list[dict[str, object]] = []
    review_panels: list[tuple[dict, np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int, int, int]]]] = []
    skipped: Counter[str] = Counter()
    print(f"Robuste Vorschläge für direkten Vergleich: {len(targets)}")
    print("CPU/Lesen; keine GPU-Inferenz; keine Produktionsdateien oder Kandidatenbilder werden verändert.")

    for idx, row in enumerate(targets, 1):
        image_name = row["image"]
        anchors = anchors_by_image.get(norm_stem(image_name), [])
        if not anchors:
            skipped["no_anchors"] += 1
            continue
        try:
            original_path = audit.find_original(image_name, original_map)
            original = load_bgr(original_path)
            current_path = audit.candidate_path(image_name, row["current_candidate"])
            proposed_path = audit.candidate_path(image_name, row["recommended_candidate"])
            if not current_path.is_file() or not proposed_path.is_file():
                skipped["candidate_missing"] += 1
                continue
            current_raw = load_bgr(current_path)
            proposed_raw = load_bgr(proposed_path)
        except Exception as exc:
            skipped[f"load_error:{type(exc).__name__}"] += 1
            print(f"WARNUNG {image_name}: {type(exc).__name__}: {exc}")
            continue

        original_h, original_w = original.shape[:2]
        current_shape_mismatch = current_raw.shape[:2] != (original_h, original_w)
        proposed_shape_mismatch = proposed_raw.shape[:2] != (original_h, original_w)
        # Canonical comparison frame is the original's size; any resizing is reported.
        current = current_raw if not current_shape_mismatch else cv2.resize(
            current_raw, (original_w, original_h), interpolation=cv2.INTER_AREA
        )
        proposed = proposed_raw if not proposed_shape_mismatch else cv2.resize(
            proposed_raw, (original_w, original_h), interpolation=cv2.INTER_AREA
        )
        boxes = [b for a in anchors if (b := audit.clipped_box(a, original_w, original_h)) is not None]
        mask = anchor_mask(anchors, original_w, original_h)
        rec: dict[str, object] = {
            "image": image_name,
            "current_candidate": row["current_candidate"],
            "recommended_candidate": row["recommended_candidate"],
            "sensitivity_stable_18_of_24": 1,
            "source_color_gt16_reduction_pct": as_float(row, "source_color_gt16_reduction_pct"),
            "current_source_color_pixels_gt16": as_int(row, "current_source_color_pixels_gt16"),
            "recommended_source_color_pixels_gt16": as_int(row, "recommended_source_color_pixels_gt16"),
            "objective_improvement": as_float(row, "objective_improvement"),
            "anchor_count": len(anchors),
            "original_width": original_w,
            "original_height": original_h,
            "current_original_size_mismatch": int(current_shape_mismatch),
            "proposed_original_size_mismatch": int(proposed_shape_mismatch),
        }
        rec.update(direct_diff_metrics(current, proposed, mask))
        outside_pct = as_float({"v": str(rec.get("outside_gt12_pct_pad8", 0))}, "v")
        if current_shape_mismatch or proposed_shape_mismatch:
            status = "SIZE_MISMATCH_REVIEW"
        elif outside_pct <= 0.25:
            status = "LOW_OUTSIDE_DIFFERENCE"
        elif outside_pct <= 1.0:
            status = "REVIEW_OUTSIDE_DIFFERENCE"
        else:
            status = "HIGH_OUTSIDE_DIFFERENCE"
        rec["direct_outside_difference_status"] = status
        metrics.append(rec)
        # Create visual sheets only for meaningful direct changes outside the 8px buffer.
        if status != "LOW_OUTSIDE_DIFFERENCE":
            review_panels.append((rec, original, current, proposed, boxes))
        if idx % 25 == 0 or idx == len(targets):
            print(f"Vorschläge verglichen: {idx}/{len(targets)} | Datensätze: {len(metrics)}")

    metrics.sort(key=lambda r: (
        -as_float({"v": str(r.get("outside_gt12_pct_pad8", 0))}, "v"),
        str(r.get("image", "")),
    ))
    write_csv(out_dir / "direct_current_proposed_outside_metrics.csv", metrics)

    review_dir = out_dir / "direct_difference_context_sheets"
    review_dir.mkdir()
    # If many are flagged, show the 32 largest differences; CSV retains every row.
    review_panels.sort(key=lambda item: -as_float({"v": str(item[0].get("outside_gt12_pct_pad8", 0))}, "v"))
    review_panels = review_panels[:32]
    for start in range(0, len(review_panels), 4):
        batch = review_panels[start:start + 4]
        tile_w, tile_h = 450, 210
        canvas = np.full((len(batch) * (tile_h + 28), tile_w * 3, 3), 248, dtype=np.uint8)
        for j, (rec, original, current, proposed, boxes) in enumerate(batch):
            y0 = j * (tile_h + 28)
            label = (f'{rec["image"]}: outside ΔE>12={rec["outside_gt12_pct_pad8"]}% '
                     f'| {rec["current_candidate"]} -> {rec["recommended_candidate"]}')
            cv2.putText(canvas, label[:140], (6, y0 + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                        (25, 25, 25), 1, cv2.LINE_AA)
            panels = [
                make_panel(original, boxes, "ORIGINAL (red = OCR anchors)", tile_w, tile_h),
                make_panel(current, boxes, f'CURRENT: {rec["current_candidate"]}', tile_w, tile_h),
                make_panel(proposed, boxes, f'PROPOSED: {rec["recommended_candidate"]}', tile_w, tile_h),
            ]
            for col, panel in enumerate(panels):
                canvas[y0 + 28:y0 + 28 + tile_h, col * tile_w:(col + 1) * tile_w] = panel
        cv2.imwrite(str(review_dir / f"direct_difference_{start // 4 + 1:03d}.jpg"), canvas,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    counts = Counter(str(r["direct_outside_difference_status"]) for r in metrics)
    high = sum(1 for r in metrics if r["direct_outside_difference_status"] == "HIGH_OUTSIDE_DIFFERENCE")
    review = sum(1 for r in metrics if r["direct_outside_difference_status"] == "REVIEW_OUTSIDE_DIFFERENCE")
    lines = [
        "DIRECT CURRENT-VS-PROPOSED OUTSIDE-ANCHOR AUDIT (READ-ONLY)",
        f"Timestamp: {stamp}",
        f"Targets evaluated: {len(metrics)} (selected from stable proposals with >= {MIN_REDUCTION_PCT:.0f}% source-color residual reduction)",
        "Metric: direct CIE Lab Delta E 76 between CURRENT and PROPOSED candidate images, outside OCR anchors and 0/2/4/8 px buffers.",
        "This tests whether selecting the proposed candidate directly changes pixels outside anchors; it is not a complete visual-quality score.",
        f"Status counts (pad=8, Delta E>12): {dict(counts)}",
        f"Rows with medium/high outside difference: {review + high}",
        f"High outside difference (>1% of outside pixels, Delta E>12): {high}",
        f"Medium outside difference (>0.25% to 1%): {review}",
        f"Rows written: {out_dir / 'direct_current_proposed_outside_metrics.csv'}",
        f"Comparison sheets for medium/high differences (up to 32 largest): {review_dir}",
        f"Skipped: {dict(skipped)}",
        "No GPU inference; no candidate images, winner_report.csv, or scoring.py modified.",
    ]
    (out_dir / "direct_current_proposed_outside_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nFEHLER: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
