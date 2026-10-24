#!/usr/bin/env python3
"""Read-only background-preservation check for stable color-rule proposals.

Uses existing original images, saved candidate images, OCR anchors, and the
already-computed sensitivity/full-population CSVs. No inference and no writes
to production files or candidate images.
"""
from __future__ import annotations

import csv
import traceback
from collections import defaultdict, Counter
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

RUN_DIR = audit.RUN_DIR
DEBUG_DIR = audit.DEBUG_DIR
AUDIT_DIR = DEBUG_DIR / "counterfactual_color_rule_full_population_20261024_214800"
SWEEP_DIR = AUDIT_DIR / "sensitivity_sweep_20261024_220012"
CHANGED_CSV = AUDIT_DIR / "full_population_changed_review.csv"
STABILITY_CSV = SWEEP_DIR / "sensitivity_image_stability.csv"
ANCHOR_PADS = (0, 2, 4, 8)
PIXEL_THRESHOLDS = (8.0, 12.0, 20.0)
MIN_REDUCTION_PCT = 90.0


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Benötigte Datei fehlt: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
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


def anchor_mask_for_image(
    anchors: list[dict[str, str]], original_shape: tuple[int, int], candidate_shape: tuple[int, int]
) -> np.ndarray:
    oh, ow = original_shape
    ch, cw = candidate_shape
    sx, sy = cw / max(1, ow), ch / max(1, oh)
    mask = np.zeros((ch, cw), dtype=np.uint8)
    for row in anchors:
        box = audit.clipped_box(row, ow, oh)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        cx1 = max(0, min(cw - 1, int(round(x1 * sx))))
        cy1 = max(0, min(ch - 1, int(round(y1 * sy))))
        cx2 = max(0, min(cw, int(round(x2 * sx))))
        cy2 = max(0, min(ch, int(round(y2 * sy))))
        if cx2 > cx1 and cy2 > cy1:
            mask[cy1:cy2, cx1:cx2] = 255
    return mask


def diff_metrics(original_bgr: np.ndarray, candidate_bgr: np.ndarray,
                 anchor_mask: np.ndarray, prefix: str, size_mismatch: bool) -> dict[str, object]:
    ch, cw = candidate_bgr.shape[:2]
    if original_bgr.shape[:2] != (ch, cw):
        interp = cv2.INTER_AREA if original_bgr.shape[0] > ch or original_bgr.shape[1] > cw else cv2.INTER_CUBIC
        comparison = cv2.resize(original_bgr, (cw, ch), interpolation=interp)
    else:
        comparison = original_bgr
    # Convert OpenCV's 8-bit Lab encoding to approximate CIE L*a*b* units
    # before computing Delta E 76, so thresholds 8/12/20 have a clearer scale.
    orig_lab = cv2.cvtColor(comparison, cv2.COLOR_BGR2LAB).astype(np.float32)
    cand_lab = cv2.cvtColor(candidate_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    orig_lab[:, :, 0] *= (100.0 / 255.0)
    cand_lab[:, :, 0] *= (100.0 / 255.0)
    orig_lab[:, :, 1:] -= 128.0
    cand_lab[:, :, 1:] -= 128.0
    delta = np.linalg.norm(cand_lab - orig_lab, axis=2)
    result: dict[str, object] = {f"{prefix}_size_mismatch": int(size_mismatch)}
    for pad in ANCHOR_PADS:
        if pad:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
            region = cv2.dilate(anchor_mask, k, iterations=1)
        else:
            region = anchor_mask
        outside = region == 0
        n = int(np.count_nonzero(outside))
        result[f"{prefix}_outside_pixels_pad{pad}"] = n
        result[f"{prefix}_anchor_buffer_area_pct_pad{pad}"] = round(100 * (1 - n / max(1, outside.size)), 4)
        if not n:
            for threshold in PIXEL_THRESHOLDS:
                result[f"{prefix}_outside_gt{int(threshold)}_pct_pad{pad}"] = 0.0
            result[f"{prefix}_outside_delta_mean_pad{pad}"] = 0.0
            result[f"{prefix}_outside_delta_p95_pad{pad}"] = 0.0
            continue
        vals = delta[outside]
        for threshold in PIXEL_THRESHOLDS:
            result[f"{prefix}_outside_gt{int(threshold)}_pct_pad{pad}"] = round(
                100 * float(np.count_nonzero(vals > threshold)) / n, 5
            )
        result[f"{prefix}_outside_delta_mean_pad{pad}"] = round(float(np.mean(vals)), 5)
        result[f"{prefix}_outside_delta_p95_pad{pad}"] = round(float(np.percentile(vals, 95)), 5)
    return result


def marked_fit_panel(img: np.ndarray, boxes: list[tuple[int, int, int, int]], title: str,
                     tile_w: int = 430, tile_h: int = 250) -> np.ndarray:
    marked = img.copy()
    h, w = marked.shape[:2]
    for x1, y1, x2, y2 in boxes:
        xx1, yy1 = max(0, x1), max(0, y1)
        xx2, yy2 = min(w - 1, x2 - 1), min(h - 1, y2 - 1)
        if xx2 >= xx1 and yy2 >= yy1:
            cv2.rectangle(marked, (xx1, yy1), (xx2, yy2), (0, 0, 255), 1, cv2.LINE_AA)
    scale = min((tile_w - 12) / max(1, w), (tile_h - 36) / max(1, h))
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    thumb = cv2.resize(marked, (nw, nh), interpolation=interp)
    panel = np.full((tile_h, tile_w, 3), 248, dtype=np.uint8)
    cv2.putText(panel, title[:62], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (25, 25, 25), 1, cv2.LINE_AA)
    x0, y0 = (tile_w - nw) // 2, 28 + (tile_h - 28 - nh) // 2
    panel[y0:y0 + nh, x0:x0 + nw] = thumb
    return panel


def main() -> None:
    for path in (AUDIT_DIR, SWEEP_DIR, audit.ORIGINAL_DIR, audit.CANDIDATE_DIR):
        if not path.exists():
            raise FileNotFoundError(f"Benötigter vorhandener Pfad fehlt: {path}")
    changed = read_csv(CHANGED_CSV)
    stability = {norm_stem(r.get("image", "")): r for r in read_csv(STABILITY_CSV)}
    anchors_by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(audit.find_anchor_csv()):
        if row.get("image", "").strip():
            anchors_by_image[norm_stem(row["image"])].append(row)
    original_map: dict[str, list[Path]] = defaultdict(list)
    for image_path in audit.ORIGINAL_DIR.rglob("*"):
        if image_path.is_file() and image_path.suffix.lower() in audit.IMAGE_EXTS:
            original_map[image_path.stem.lower()].append(image_path)

    # Evaluate only the robust, strong-reduction subset: stable in >=18/24
    # combinations and at least 90% fewer source-color residual pixels.
    targets = []
    for row in changed:
        st = stability.get(norm_stem(row.get("image", "")), {})
        if as_int(st, "stable_alternative_18_of_24") != 1:
            continue
        if as_float(row, "source_color_gt16_reduction_pct") < MIN_REDUCTION_PCT:
            continue
        targets.append(row)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = SWEEP_DIR / f"outside_anchor_preservation_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    metrics: list[dict[str, object]] = []
    panels_to_render: list[tuple[dict, np.ndarray, np.ndarray, np.ndarray, list[tuple[int,int,int,int]]]] = []
    skipped: Counter[str] = Counter()

    print(f"Robuste Vorschläge für Prüfung: {len(targets)}")
    print("CPU/Lesen; kein Modelllauf; Produktionsbericht und Kandidatenbilder bleiben unverändert.")

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
            current = load_bgr(current_path)
            proposed = load_bgr(proposed_path)
        except Exception as exc:
            skipped[f"load_error:{type(exc).__name__}"] += 1
            print(f"WARNUNG {image_name}: {type(exc).__name__}: {exc}")
            continue

        h, w = original.shape[:2]
        boxes = []
        for ar in anchors:
            box = audit.clipped_box(ar, w, h)
            if box:
                boxes.append(box)
        current_mask = anchor_mask_for_image(anchors, (h, w), current.shape[:2])
        proposed_mask = anchor_mask_for_image(anchors, (h, w), proposed.shape[:2])
        size_mismatch = current.shape[:2] != original.shape[:2] or proposed.shape[:2] != original.shape[:2]
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
            "original_width": w,
            "original_height": h,
            "current_width": current.shape[1],
            "current_height": current.shape[0],
            "proposed_width": proposed.shape[1],
            "proposed_height": proposed.shape[0],
            "size_mismatch": int(size_mismatch),
        }
        rec.update(diff_metrics(original, current, current_mask, "current", current.shape[:2] != (h, w)))
        rec.update(diff_metrics(original, proposed, proposed_mask, "proposed", proposed.shape[:2] != (h, w)))
        cur_pad8 = float(rec["current_outside_gt12_pct_pad8"])
        prop_pad8 = float(rec["proposed_outside_gt12_pct_pad8"])
        increase_pp = prop_pad8 - cur_pad8
        rec["proposed_minus_current_outside_gt12_pct_pad8_pp"] = round(increase_pp, 5)
        if size_mismatch:
            status = "SIZE_MISMATCH_REVIEW"
        elif prop_pad8 > 0.5 or increase_pp > 0.25:
            status = "CHECK_OUTSIDE_ANCHORS"
        elif prop_pad8 <= 0.25 and increase_pp <= 0.10:
            status = "LIKELY_SAFE_OUTSIDE_ANCHORS"
        else:
            status = "INTERMEDIATE_REVIEW"
        rec["outside_guard_status"] = status
        metrics.append(rec)
        if status in {"CHECK_OUTSIDE_ANCHORS", "SIZE_MISMATCH_REVIEW", "INTERMEDIATE_REVIEW"}:
            panels_to_render.append((rec, original, current, proposed, boxes))
        if idx % 25 == 0 or idx == len(targets):
            print(f"Vorschläge geprüft: {idx}/{len(targets)} | Datensätze: {len(metrics)}")

    metrics.sort(key=lambda r: (r["outside_guard_status"], -as_float({"x": str(r.get("proposed_minus_current_outside_gt12_pct_pad8_pp", 0))}, "x")))
    write_csv(out_dir / "outside_anchor_preservation_metrics.csv", metrics)

    flagged_dir = out_dir / "flagged_context_sheets"
    flagged_dir.mkdir()
    for start in range(0, len(panels_to_render), 4):
        batch = panels_to_render[start:start + 4]
        tile_w, tile_h = 430, 250
        canvas = np.full((len(batch) * (tile_h + 30), tile_w * 3, 3), 248, dtype=np.uint8)
        for j, (rec, original, current, proposed, boxes) in enumerate(batch):
            y0 = j * (tile_h + 30)
            label = (f'{rec["image"]}: {rec["outside_guard_status"]} | '
                     f'outside Δpp={rec["proposed_minus_current_outside_gt12_pct_pad8_pp"]}')
            cv2.putText(canvas, label[:150], (6, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (25,25,25), 1, cv2.LINE_AA)
            current_boxes = []
            for ar in anchors:
                b = audit.clipped_box(ar, h, w)
                if b:
                    x1, y1, x2, y2 = b
                    current_boxes.append((round(x1 * current.shape[1] / w), round(y1 * current.shape[0] / h),
                                          round(x2 * current.shape[1] / w), round(y2 * current.shape[0] / h)))
            proposed_boxes = []
            for ar in anchors:
                b = audit.clipped_box(ar, h, w)
                if b:
                    x1, y1, x2, y2 = b
                    proposed_boxes.append((round(x1 * proposed.shape[1] / w), round(y1 * proposed.shape[0] / h),
                                           round(x2 * proposed.shape[1] / w), round(y2 * proposed.shape[0] / h)))
            panels = [
                marked_fit_panel(original, boxes, "ORIGINAL (red = OCR anchors)", tile_w, tile_h),
                marked_fit_panel(current, current_boxes, f'CURRENT: {rec["current_candidate"]}', tile_w, tile_h),
                marked_fit_panel(proposed, proposed_boxes, f'PROPOSED: {rec["recommended_candidate"]}', tile_w, tile_h),
            ]
            for col, panel in enumerate(panels):
                canvas[y0 + 30:y0 + 30 + tile_h, col*tile_w:(col+1)*tile_w] = panel
        cv2.imwrite(str(flagged_dir / f"outside_anchor_flagged_{start//4+1:03d}.jpg"), canvas,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    counts = Counter(str(r["outside_guard_status"]) for r in metrics)
    lines = [
        "OUTSIDE-ANCHOR PRESERVATION AUDIT (READ-ONLY)",
        f"Timestamp: {stamp}",
        f"Stable proposals evaluated: {len(metrics)} (target set: {len(targets)})",
        f"Stability: alternative selected in >=18/24 parameter combinations; source-color residual reduction >= {MIN_REDUCTION_PCT:.0f}%.",
        "Metric: original-vs-candidate CIE Lab Delta E 76 outside OCR anchors, with anchor buffers of 0/2/4/8 px.",
        "This is a collateral-change guard, not a full visual-quality score.",
        "Thresholds for triage use pad=8 and Lab delta > 12: CHECK if proposed outside change > 0.5% or rises > 0.25 percentage points versus current; LIKELY_SAFE if <=0.25% and rise <=0.10 pp; intermediate values are reviewed.",
        f"Status counts: {dict(counts)}",
        f"Rows written: {out_dir / 'outside_anchor_preservation_metrics.csv'}",
        f"Flagged/intermediate comparison sheets: {flagged_dir}",
        f"Skipped: {dict(skipped)}",
        "No GPU inference; no candidate images, winner_report.csv, or scoring.py modified.",
    ]
    (out_dir / "outside_anchor_preservation_summary.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nFEHLER: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
