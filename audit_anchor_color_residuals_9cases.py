#!/usr/bin/env python3
"""CPU-only color-residual audit for the nine manually reviewed Cleanup cases.

Reads the existing local-anchor quality audit and candidate images, then writes
an anchor-level CSV and an image/candidate summary CSV. It does not modify
production scripts, candidate images, or source images.
"""
from __future__ import annotations

import csv
import sys
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

try:
    import cv2
    import numpy as np
except Exception as exc:  # report a useful error instead of a short traceback only
    print(f"FEHLER beim Importieren von OpenCV/numpy: {type(exc).__name__}: {exc}")
    print("Benutze den bestehenden Cleanup-.venv-Python, in dem OpenCV/numpy installiert sind.")
    input("Enter zum Schließen ...")
    raise

PROJECT = Path.home() / "Downloads" / "Cleanup"
RUN_DIR = PROJECT / "yolo_mask_winners" / "full_recheck_after_detection_20261024_030641"
DEBUG_DIR = RUN_DIR / "debug_candidates"
CANDIDATE_DIR = DEBUG_DIR / "cleaned_images"
QUALITY_AUDIT = DEBUG_DIR / "local_anchor_quality_audit_9cases_20261024_052534.csv"
ORIGINAL_DIR = Path.home() / "Downloads" / "yabai proj" / "yabaiproject - Copy" / "mit_kanji"

CASES = {
    "h003_000", "h008_001", "m002_000", "m003_015", "m003_016",
    "m003_017", "m004_002", "m004_005", "m004_006",
}
IMAGE_EXTS = {".png", ".bmp", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
PAD = 6

DETAIL_EXTRA_FIELDS = [
    "context_valid", "source_color_direction_valid", "source_color_strength",
    "visible_target_pixels", "source_color_residual_mean", "source_color_residual_p90",
    "source_color_pixels_gt8", "source_color_pixels_gt16", "chroma_residual_p90",
    "chroma_pixels_gt12", "chroma_pixels_gt20",
]
SUMMARY_FIELDS = [
    "image", "current_winner", "candidate", "anchor_count", "context_valid_anchors",
    "source_color_valid_anchors", "visible_target_pixels", "source_color_residual_mean",
    "source_color_residual_p90", "source_color_pixels_gt8", "source_color_pixels_gt16",
    "chroma_residual_p90", "chroma_pixels_gt12", "chroma_pixels_gt20",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Eingabe-CSV fehlt: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def find_original(image_name: str, original_map: dict[str, list[Path]]) -> Path:
    stem = Path(image_name).stem.lower()
    choices = original_map.get(stem, [])
    if not choices:
        raise FileNotFoundError(
            f"Originalbild für {image_name} nicht gefunden unter {ORIGINAL_DIR}."
        )
    if len(choices) > 1:
        exact = [p for p in choices if p.name.lower() == image_name.lower()]
        if len(exact) == 1:
            return exact[0]
        choices = sorted(choices)
        raise RuntimeError(f"Mehrere Originale für {image_name}: " + ", ".join(map(str, choices[:6])))
    return choices[0]


def candidate_path(image_name: str, candidate: str) -> Path:
    stem = Path(image_name).stem
    suffix = candidate.strip().lower()
    path = CANDIDATE_DIR / f"{stem}__{suffix}.png"
    if path.is_file():
        return path
    # Fallback to case-insensitive filename matching, without assuming extension.
    target = f"{stem}__{suffix}".lower()
    matches = [p for p in CANDIDATE_DIR.glob(f"{stem}__*") if p.stem.lower() == target and p.suffix.lower() in IMAGE_EXTS]
    if len(matches) == 1:
        return matches[0]
    available = sorted(p.name for p in CANDIDATE_DIR.glob(f"{stem}__*"))
    raise FileNotFoundError(
        f"Kandidatbild fehlt: {path}. Vorhandene Varianten für {stem}: {available[:20]}"
    )


def load_lab(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Bild kann nicht gelesen werden: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)


def clipped_box(row: dict[str, str], w: int, h: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(w, int(round(float(row["x1"])))))
    y1 = max(0, min(h, int(round(float(row["y1"])))))
    x2 = max(0, min(w, int(round(float(row["x2"])))))
    y2 = max(0, min(h, int(round(float(row["y2"])))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Ungültige Anchor-Box für {row.get('anchor_id')}: {(x1, y1, x2, y2)}")
    return x1, y1, x2, y2


def roi_and_ring(lab: np.ndarray, box: tuple[int, int, int, int], pad: int = PAD):
    h, w = lab.shape[:2]
    x1, y1, x2, y2 = box
    roi_ab = lab[y1:y2, x1:x2, 1:3].reshape(-1, 2)
    ox1, oy1 = max(0, x1 - pad), max(0, y1 - pad)
    ox2, oy2 = min(w, x2 + pad), min(h, y2 + pad)
    outer = lab[oy1:oy2, ox1:ox2, 1:3]
    keep = np.ones(outer.shape[:2], dtype=bool)
    keep[y1 - oy1:y2 - oy1, x1 - ox1:x2 - ox1] = False
    ring = outer[keep]
    if ring.size == 0 or roi_ab.size == 0:
        return roi_ab, None
    return roi_ab, np.median(ring, axis=0)


def source_direction(original_lab: np.ndarray, box: tuple[int, int, int, int]):
    roi, bg = roi_and_ring(original_lab, box)
    if bg is None or len(roi) < 6:
        return None, 0.0, 0
    delta = roi - bg
    mag = np.linalg.norm(delta, axis=1)
    # Robustly estimate a color direction from the strongest chromatic pixels
    # instead of the near-zero mean across the whole OCR rectangle.
    cutoff = max(5.0, float(np.percentile(mag, 85)))
    strong = delta[mag >= cutoff]
    strong_mag = mag[mag >= cutoff]
    if len(strong) < max(6, int(np.ceil(len(roi) * 0.002))):
        return None, float(np.median(strong_mag)) if len(strong_mag) else 0.0, 0
    unit = strong / np.maximum(strong_mag[:, None], 1e-6)
    mean_unit = np.mean(unit, axis=0)
    coherence = float(np.linalg.norm(mean_unit))
    direction_norm = float(np.linalg.norm(mean_unit))
    strength = float(np.median(strong_mag)) * coherence
    if coherence < 0.22 or strength < 3.5 or direction_norm < 1e-6:
        return None, strength, 0
    return mean_unit / direction_norm, strength, 1


def quantile_or_zero(values: list[np.ndarray], q: float) -> float:
    if not values:
        return 0.0
    merged = np.concatenate([v.reshape(-1) for v in values if v.size]) if any(v.size for v in values) else np.array([], dtype=np.float32)
    return float(np.percentile(merged, q)) if merged.size else 0.0


def main() -> None:
    if not DEBUG_DIR.is_dir():
        raise FileNotFoundError(f"Cleanup-Ausgabeordner fehlt: {DEBUG_DIR}")
    if not ORIGINAL_DIR.is_dir():
        raise FileNotFoundError(f"Originalbildordner fehlt: {ORIGINAL_DIR}")
    if not CANDIDATE_DIR.is_dir():
        raise FileNotFoundError(f"Kandidatenordner fehlt: {CANDIDATE_DIR}")

    quality_rows = read_csv(QUALITY_AUDIT)
    required = {"image", "current_winner", "candidate", "anchor_id", "type", "x1", "y1", "x2", "y2"}
    if not quality_rows or not required.issubset(quality_rows[0]):
        raise ValueError(f"Die Qualität-Audit-CSV hat nicht die erwarteten Spalten: {sorted(required)}")
    selected = [r for r in quality_rows if Path(r["image"]).stem.lower() in CASES]
    if not selected:
        raise ValueError(f"Keine der neun gesuchten Bilddateien in {QUALITY_AUDIT} gefunden.")

    # Build an index of original files once, supporting PNG/BMP/JPEG/WebP/TIFF.
    original_map: dict[str, list[Path]] = defaultdict(list)
    for p in ORIGINAL_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            original_map[p.stem.lower()].append(p)

    # Reprocess one image/candidate at a time; summary stores raw pixel distributions
    # so the aggregate p90 is computed across pixels, not by averaging anchor p90s.
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        grouped[(row["image"], row["candidate"])].append(row)

    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    case_image_cache: dict[str, np.ndarray] = {}
    case_original_path: dict[str, Path] = {}
    total_groups = len(grouped)
    valid_dirs = 0

    for idx, ((image_name, candidate), anchors) in enumerate(sorted(grouped.items()), start=1):
        stem = Path(image_name).stem.lower()
        if stem not in case_image_cache:
            original_path = find_original(image_name, original_map)
            case_original_path[stem] = original_path
            case_image_cache[stem] = load_lab(original_path)
        original_lab = case_image_cache[stem]
        cand_path = candidate_path(image_name, candidate)
        cand_lab = load_lab(cand_path)
        orig_h, orig_w = original_lab.shape[:2]
        cand_h, cand_w = cand_lab.shape[:2]
        sx, sy = cand_w / orig_w, cand_h / orig_h

        all_source_pos: list[np.ndarray] = []
        all_chroma: list[np.ndarray] = []
        sum_visible = sum_gt8 = sum_gt16 = sum_chroma12 = sum_chroma20 = 0
        context_count = source_valid_count = 0
        current_winner = anchors[0].get("current_winner", "")

        for row in anchors:
            orig_box = clipped_box(row, orig_w, orig_h)
            direction, strength, direction_valid = source_direction(original_lab, orig_box)
            x1, y1, x2, y2 = orig_box
            cand_box = (int(round(x1 * sx)), int(round(y1 * sy)), int(round(x2 * sx)), int(round(y2 * sy)))
            cand_box = clipped_box({"x1": cand_box[0], "y1": cand_box[1], "x2": cand_box[2], "y2": cand_box[3], "anchor_id": row.get("anchor_id", "")}, cand_w, cand_h)
            cand_roi, cand_bg = roi_and_ring(cand_lab, cand_box)
            context_valid = int(cand_bg is not None)
            if context_valid:
                context_count += 1
                delta = cand_roi - cand_bg
                chroma = np.linalg.norm(delta, axis=1)
                all_chroma.append(chroma.astype(np.float32))
                sum_chroma12 += int(np.count_nonzero(chroma > 12))
                sum_chroma20 += int(np.count_nonzero(chroma > 20))
            else:
                delta = np.zeros((0, 2), dtype=np.float32)
                chroma = np.zeros((0,), dtype=np.float32)

            if direction_valid and context_valid:
                source_valid_count += 1
                proj = delta @ direction
                positive = np.clip(proj, 0, None).astype(np.float32)
                all_source_pos.append(positive)
                visible = int(np.count_nonzero(proj > 4))
                gt8 = int(np.count_nonzero(proj > 8))
                gt16 = int(np.count_nonzero(proj > 16))
            else:
                positive = np.zeros((0,), dtype=np.float32)
                visible = gt8 = gt16 = 0
            sum_visible += visible
            sum_gt8 += gt8
            sum_gt16 += gt16

            detail = {k: row.get(k, "") for k in [
                "image", "current_winner", "candidate", "is_current_winner", "anchor_id",
                "type", "x1", "y1", "x2", "y2",
            ] if k in row}
            detail.update({
                "context_valid": context_valid,
                "source_color_direction_valid": direction_valid,
                "source_color_strength": round(float(strength), 6),
                "visible_target_pixels": visible,
                "source_color_residual_mean": round(float(np.mean(positive)), 6) if positive.size else 0.0,
                "source_color_residual_p90": round(float(np.percentile(positive, 90)), 6) if positive.size else 0.0,
                "source_color_pixels_gt8": gt8,
                "source_color_pixels_gt16": gt16,
                "chroma_residual_p90": round(float(np.percentile(chroma, 90)), 6) if chroma.size else 0.0,
                "chroma_pixels_gt12": int(np.count_nonzero(chroma > 12)),
                "chroma_pixels_gt20": int(np.count_nonzero(chroma > 20)),
            })
            detail_rows.append(detail)

        if source_valid_count:
            valid_dirs += 1
        source_merged = np.concatenate(all_source_pos) if all_source_pos else np.array([], dtype=np.float32)
        chroma_merged = np.concatenate(all_chroma) if all_chroma else np.array([], dtype=np.float32)
        summary_rows.append({
            "image": image_name,
            "current_winner": current_winner,
            "candidate": candidate,
            "anchor_count": len(anchors),
            "context_valid_anchors": context_count,
            "source_color_valid_anchors": source_valid_count,
            "visible_target_pixels": sum_visible,
            "source_color_residual_mean": round(float(np.mean(source_merged)), 6) if source_merged.size else 0.0,
            "source_color_residual_p90": round(float(np.percentile(source_merged, 90)), 6) if source_merged.size else 0.0,
            "source_color_pixels_gt8": sum_gt8,
            "source_color_pixels_gt16": sum_gt16,
            "chroma_residual_p90": round(float(np.percentile(chroma_merged, 90)), 6) if chroma_merged.size else 0.0,
            "chroma_pixels_gt12": sum_chroma12,
            "chroma_pixels_gt20": sum_chroma20,
        })
        print(f"[{idx:>3}/{total_groups}] OK: {image_name} | {candidate}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_path = DEBUG_DIR / f"local_anchor_color_residual_audit_9cases_{stamp}.csv"
    summary_path = DEBUG_DIR / f"local_anchor_color_residual_summary_9cases_{stamp}.csv"

    # DictWriter uses explicit complete field lists; missing fields are never silently ignored.
    detail_fields = [
        "image", "current_winner", "candidate", "is_current_winner", "anchor_id", "type",
        "x1", "y1", "x2", "y2", *DETAIL_EXTRA_FIELDS,
    ]
    with detail_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(detail_rows)
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n========== FARBREST-DIAGNOSE ABGESCHLOSSEN ==========")
    print(f"Eingabezeilen: {len(selected)}")
    print(f"Bild/Kandidaten-Gruppen: {len(summary_rows)}")
    print(f"Detailzeilen geschrieben: {len(detail_rows)}")
    print(f"Gruppen mit mindestens einem gültigen Quellfarbvektor: {valid_dirs}")
    print(f"Detail:  {detail_path}")
    print(f"Summary: {summary_path}")
    print("Nur CPU/Lesen; keine GPU-Inferenz und keine Produktionsdateien geändert.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nFEHLER: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        print("Das Terminal bleibt offen.")
        try:
            input("Enter zum Schließen ...")
        except EOFError:
            pass
        sys.exit(1)
