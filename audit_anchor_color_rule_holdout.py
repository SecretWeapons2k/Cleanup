#!/usr/bin/env python3
"""Independent, CPU-only holdout audit for the color-residual candidate rule.

This script samples 24 cases from the existing full Cleanup run, excluding the
nine cases used to tune the rule. It computes color residuals for the available
candidate images, applies the rule as a counterfactual, and writes review CSVs
and comparison sheets. It does not run model inference or edit production files.
"""
from __future__ import annotations

import csv
import random
import sys
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

try:
    import cv2
    import numpy as np
except Exception as exc:
    print(f"FEHLER beim Importieren von OpenCV/numpy: {type(exc).__name__}: {exc}")
    print("Verwende den vorhandenen Cleanup-.venv-Python.")
    input("Enter zum Schließen ...")
    raise

PROJECT = Path.home() / "Downloads" / "Cleanup"
RUN_DIR = PROJECT / "yolo_mask_winners" / "full_recheck_after_detection_20261024_030641"
DEBUG_DIR = RUN_DIR / "debug_candidates"
CANDIDATE_DIR = DEBUG_DIR / "cleaned_images"
WINNER_REPORT = RUN_DIR / "winner_report.csv"
CANDIDATE_SCORES = DEBUG_DIR / "candidate_scores.csv"
ORIGINAL_DIR = Path.home() / "Downloads" / "yabai proj" / "yabaiproject - Copy" / "mit_kanji"
ANCHOR_CSV_OPTIONS = [
    PROJECT / "anchors.csv",
    Path.home() / "Downloads" / "yabai proj" / "processed_output" / "anchors.csv",
    DEBUG_DIR / "anchors.csv",
]

CALIBRATION_CASES = {
    "h003_000", "h008_001", "m002_000", "m003_015", "m003_016",
    "m003_017", "m004_002", "m004_005", "m004_006",
}
SAMPLE_PER_GROUP = 8
RANDOM_SEED = 20261024
REDUCTION_RATIO_MAX = 0.75
CHROMA_P90_RATIO_MAX = 1.05
COLOR_WEIGHT = 0.24
PAD = 6
IMAGE_EXTS = {".png", ".bmp", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV-Datei fehlt: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def find_anchor_csv() -> Path:
    for path in ANCHOR_CSV_OPTIONS:
        if path.is_file():
            return path
    checked = "\n".join(f"  - {p}" for p in ANCHOR_CSV_OPTIONS)
    raise FileNotFoundError("anchors.csv wurde nicht gefunden. Geprüfte Pfade:\n" + checked)


def as_float(row: dict[str, str], key: str, default: float | None = None) -> float:
    value = row.get(key, "")
    if value is None or str(value).strip() == "":
        if default is None:
            raise ValueError(f"Fehlender numerischer Wert in Spalte {key!r}: {row}")
        return default
    return float(value)


def find_original(image_name: str, original_map: dict[str, list[Path]]) -> Path:
    stem = Path(image_name).stem.lower()
    choices = original_map.get(stem, [])
    if len(choices) == 1:
        return choices[0]
    exact = [p for p in choices if p.name.lower() == image_name.lower()]
    if len(exact) == 1:
        return exact[0]
    if not choices:
        raise FileNotFoundError(f"Originalbild für {image_name} nicht gefunden in {ORIGINAL_DIR}")
    raise RuntimeError(f"Mehrere Originalbilder für {image_name}: " + ", ".join(map(str, choices[:8])))


def candidate_path(image_name: str, candidate: str) -> Path:
    stem = Path(image_name).stem
    path = CANDIDATE_DIR / f"{stem}__{candidate.lower()}.png"
    if path.is_file():
        return path
    key = f"{stem}__{candidate}".lower()
    matches = [p for p in CANDIDATE_DIR.glob(f"{stem}__*")
               if p.suffix.lower() in IMAGE_EXTS and p.stem.lower() == key]
    if len(matches) == 1:
        return matches[0]
    return Path()


def load_lab(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Bild kann nicht gelesen werden: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)


def load_color(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Bild kann nicht gelesen werden: {path}")
    return image


def clipped_box(row: dict[str, str], w: int, h: int) -> tuple[int, int, int, int] | None:
    try:
        x1 = max(0, min(w, int(round(as_float(row, "x1")))))
        y1 = max(0, min(h, int(round(as_float(row, "y1")))))
        x2 = max(0, min(w, int(round(as_float(row, "x2")))))
        y2 = max(0, min(h, int(round(as_float(row, "y2")))))
    except (ValueError, TypeError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
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
    cutoff = max(5.0, float(np.percentile(mag, 85)))
    mask = mag >= cutoff
    strong = delta[mask]
    strong_mag = mag[mask]
    if len(strong) < max(6, int(np.ceil(len(roi) * 0.002))):
        return None, float(np.median(strong_mag)) if len(strong_mag) else 0.0, 0
    unit = strong / np.maximum(strong_mag[:, None], 1e-6)
    mean_unit = np.mean(unit, axis=0)
    coherence = float(np.linalg.norm(mean_unit))
    strength = float(np.median(strong_mag)) * coherence
    if coherence < 0.22 or strength < 3.5 or coherence < 1e-6:
        return None, strength, 0
    return mean_unit / coherence, strength, 1


def group_for_candidate(candidate: str) -> str:
    c = candidate.upper()
    if c.startswith("CONTRAST"):
        return "contrast"
    if c.startswith("HYBRID"):
        return "hybrid"
    return "other"


def select_holdout(winners: list[dict[str, str]], anchor_map: dict[str, list[dict[str, str]]],
                   scores_by_image: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    rng = random.Random(RANDOM_SEED)
    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in winners:
        image = row.get("image", "").strip()
        stem = Path(image).stem.lower()
        current = row.get("selected_distance_px", "").strip()
        if not image or not current or stem in CALIBRATION_CASES:
            continue
        if not anchor_map.get(stem) or not scores_by_image.get(stem):
            continue
        variants = {r.get("candidate", "") for r in scores_by_image[stem]}
        if current not in variants:
            continue
        current_path = candidate_path(image, current)
        if not current_path.is_file():
            continue
        buckets[group_for_candidate(current)].append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected: list[dict[str, str]] = []
    for group in ("contrast", "hybrid", "other"):
        selected.extend(buckets.get(group, [])[:SAMPLE_PER_GROUP])
    if len(selected) < SAMPLE_PER_GROUP * 3:
        chosen_names = {r.get("image") for r in selected}
        rest = [r for name, bucket in buckets.items() for r in bucket if r.get("image") not in chosen_names]
        rng.shuffle(rest)
        selected.extend(rest[:SAMPLE_PER_GROUP * 3 - len(selected)])
    # Stable order for output once sample selection is complete.
    return sorted(selected, key=lambda r: r.get("image", ""))


def metric_for_candidate(image_name: str, candidate: str,
                         anchors: list[dict[str, str]], original_lab: np.ndarray,
                         candidate_lab: np.ndarray) -> dict[str, float | int]:
    oh, ow = original_lab.shape[:2]
    ch, cw = candidate_lab.shape[:2]
    sx, sy = cw / ow, ch / oh
    all_source: list[np.ndarray] = []
    all_chroma: list[np.ndarray] = []
    sum_gt8 = sum_gt16 = sum_chroma12 = sum_chroma20 = 0
    context_count = direction_count = 0
    for row in anchors:
        obox = clipped_box(row, ow, oh)
        if obox is None:
            continue
        direction, _, direction_valid = source_direction(original_lab, obox)
        x1, y1, x2, y2 = obox
        cbox = (int(round(x1 * sx)), int(round(y1 * sy)),
                int(round(x2 * sx)), int(round(y2 * sy)))
        cbox = clipped_box({"x1": str(cbox[0]), "y1": str(cbox[1]),
                            "x2": str(cbox[2]), "y2": str(cbox[3])}, cw, ch)
        if cbox is None:
            continue
        cand_roi, cand_bg = roi_and_ring(candidate_lab, cbox)
        if cand_bg is None:
            continue
        context_count += 1
        delta = cand_roi - cand_bg
        chroma = np.linalg.norm(delta, axis=1)
        all_chroma.append(chroma.astype(np.float32))
        sum_chroma12 += int(np.count_nonzero(chroma > 12))
        sum_chroma20 += int(np.count_nonzero(chroma > 20))
        if direction_valid and direction is not None:
            direction_count += 1
            proj = delta @ direction
            positive = np.clip(proj, 0, None).astype(np.float32)
            all_source.append(positive)
            sum_gt8 += int(np.count_nonzero(proj > 8))
            sum_gt16 += int(np.count_nonzero(proj > 16))
    src = np.concatenate(all_source) if all_source else np.array([], dtype=np.float32)
    chr_ = np.concatenate(all_chroma) if all_chroma else np.array([], dtype=np.float32)
    return {
        "anchor_count": len(anchors), "context_valid_anchors": context_count,
        "source_color_valid_anchors": direction_count,
        "source_color_residual_mean": float(np.mean(src)) if src.size else 0.0,
        "source_color_residual_p90": float(np.percentile(src, 90)) if src.size else 0.0,
        "source_color_pixels_gt8": sum_gt8, "source_color_pixels_gt16": sum_gt16,
        "chroma_residual_p90": float(np.percentile(chr_, 90)) if chr_.size else 0.0,
        "chroma_pixels_gt12": sum_chroma12, "chroma_pixels_gt20": sum_chroma20,
    }


def make_contact_sheets(review_rows: list[dict[str, str]], output_dir: Path,
                        original_map: dict[str, list[Path]], anchor_map: dict[str, list[dict[str, str]]]) -> int:
    if not review_rows:
        return 0
    tile_w, tile_h, label_h = 900, 300, 54
    per_sheet = 6
    sheets = 0
    for start in range(0, len(review_rows), per_sheet):
        batch = review_rows[start:start + per_sheet]
        row_h = tile_h + label_h
        canvas = np.full((len(batch) * row_h, tile_w * 2, 3), 248, dtype=np.uint8)
        for i, row in enumerate(batch):
            image_name = row["image"]
            y0 = i * row_h
            label = f'{image_name} | proposed: {row["recommended_candidate"]}'
            cv2.putText(canvas, label[:120], (12, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (20, 20, 20), 1, cv2.LINE_AA)
            current = row["current_candidate"]
            proposed = row["recommended_candidate"]
            for col, candidate, tag in ((0, current, "CURRENT"), (1, proposed, "PROPOSED")):
                path = candidate_path(image_name, candidate)
                if not path.is_file():
                    continue
                image = load_color(path)
                h, w = image.shape[:2]
                original_path = find_original(image_name, original_map)
                original = load_color(original_path)
                oh, ow = original.shape[:2]
                sx, sy = w / max(1, ow), h / max(1, oh)
                # Scale original OCR anchors to this candidate's dimensions, then
                # crop to their shared envelope with a small context margin.
                boxes = []
                for anchor in anchor_map.get(image_name, []):
                    obox = clipped_box(anchor, ow, oh)
                    if obox is None:
                        continue
                    ax1, ay1, ax2, ay2 = obox
                    boxes.append((int(round(ax1 * sx)), int(round(ay1 * sy)),
                                  int(round(ax2 * sx)), int(round(ay2 * sy))))
                if boxes:
                    x1 = max(0, min(b[0] for b in boxes) - 12)
                    y1 = max(0, min(b[1] for b in boxes) - 8)
                    x2 = min(w, max(b[2] for b in boxes) + 12)
                    y2 = min(h, max(b[3] for b in boxes) + 8)
                    image = image[y1:y2, x1:x2]
                ih, iw = image.shape[:2]
                scale = min((tile_w - 12) / max(1, iw), (tile_h - 8) / max(1, ih))
                nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
                interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
                thumb = cv2.resize(image, (nw, nh), interpolation=interp)
                xx = col * tile_w + (tile_w - nw) // 2
                yy = y0 + label_h + (tile_h - nh) // 2
                canvas[yy:yy + nh, xx:xx + nw] = thumb
                cv2.putText(canvas, tag, (col * tile_w + 12, y0 + 43), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (40, 40, 40), 1, cv2.LINE_AA)
        target = output_dir / f"holdout_comparison_{sheets + 1:02d}.jpg"
        cv2.imwrite(str(target), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        sheets += 1
    return sheets


def main() -> None:
    for p in (DEBUG_DIR, CANDIDATE_DIR, ORIGINAL_DIR, WINNER_REPORT, CANDIDATE_SCORES):
        if not p.exists():
            raise FileNotFoundError(f"Benötigter vorhandener Projektpfad fehlt: {p}")
    anchors_path = find_anchor_csv()
    print(f"Ankerdatei: {anchors_path}")
    winner_rows = read_csv(WINNER_REPORT)
    score_rows = read_csv(CANDIDATE_SCORES)
    # In the actual full-run candidate_scores.csv, the candidate identifier is
    # stored in "distance_px"; some diagnostic exports call it "candidate".
    # Normalize both schemas without changing the source CSV.
    for score_row in score_rows:
        candidate_value = (score_row.get("candidate") or "").strip()
        if not candidate_value:
            candidate_value = (score_row.get("distance_px") or "").strip()
        score_row["candidate"] = candidate_value
    anchor_rows = read_csv(anchors_path)
    if not winner_rows or not {"image", "selected_distance_px"}.issubset(winner_rows[0]):
        raise ValueError(f"winner_report.csv hat nicht die erwarteten Spalten: {list(winner_rows[0]) if winner_rows else []}")
    if not score_rows or not {"image", "candidate"}.issubset(score_rows[0]):
        raise ValueError(f"candidate_scores.csv hat nicht die erwarteten Spalten: {list(score_rows[0]) if score_rows else []}")
    score_headers = set(score_rows[0])
    structure_key = "structure_score" if "structure_score" in score_headers else None
    if not structure_key:
        raise ValueError("candidate_scores.csv enthält keine Spalte structure_score; der Regel-Gegenversuch kann dann nicht korrekt reproduziert werden. Vorhandene Spalten: " + ", ".join(score_rows[0].keys()))
    required_anchor = {"image", "x1", "y1", "x2", "y2"}
    if not anchor_rows or not required_anchor.issubset(anchor_rows[0]):
        raise ValueError(f"anchors.csv hat nicht die erwarteten Spalten: {list(anchor_rows[0]) if anchor_rows else []}")

    # Key image-indexed tables by lowercase stem so "scene.png" and "scene"
    # in anchors/diagnostic CSVs still refer to the same image.
    anchor_map: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in anchor_rows:
        image_value = row.get("image", "").strip()
        if image_value:
            anchor_map[Path(image_value).stem.lower()].append(row)
    scores_by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in score_rows:
        image_value = row.get("image", "").strip()
        if image_value and row.get("candidate", "").strip():
            scores_by_image[Path(image_value).stem.lower()].append(row)

    original_map: dict[str, list[Path]] = defaultdict(list)
    for p in ORIGINAL_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            original_map[p.stem.lower()].append(p)

    selected = select_holdout(winner_rows, anchor_map, scores_by_image)
    if not selected:
        raise RuntimeError("Keine unabhängigen Holdout-Fälle gefunden. Prüfe Anker- und Kandidatenpfade.")
    bucket_counts = defaultdict(int)
    for row in selected:
        bucket_counts[group_for_candidate(row["selected_distance_px"])] += 1
    print(f"Holdout-Auswahl (außerhalb der 9 Kalibrierungsfälle): {len(selected)} Bilder")
    print("Strata: " + ", ".join(f"{k}={v}" for k, v in sorted(bucket_counts.items())))
    print("Keine GPU-Inferenz. Fortschritt wird pro Kandidatenbild ausgegeben.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = DEBUG_DIR / f"independent_color_rule_holdout_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    all_summary: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    changed_reviews: list[dict[str, str]] = []
    processed = 0
    total_candidates = sum(len(scores_by_image[Path(r["image"]).stem.lower()]) for r in selected)

    for image_i, winner in enumerate(selected, start=1):
        image_name = winner["image"].strip()
        current = winner["selected_distance_px"].strip()
        image_key = Path(image_name).stem.lower()
        anchors = anchor_map[image_key]
        original_path = find_original(image_name, original_map)
        original_lab = load_lab(original_path)
        candidates = scores_by_image[image_key]
        candidate_scores_map = {r["candidate"]: r for r in candidates}
        summaries_for_image: list[dict[str, object]] = []
        for candidate_row in candidates:
            candidate = candidate_row["candidate"]
            img_path = candidate_path(image_name, candidate)
            if not img_path.is_file():
                continue
            candidate_lab = load_lab(img_path)
            metrics = metric_for_candidate(image_name, candidate, anchors, original_lab, candidate_lab)
            score = as_float(candidate_row, structure_key)
            record: dict[str, object] = {
                "image": image_name,
                "sample_group": group_for_candidate(current),
                "current_candidate": current,
                "candidate": candidate,
                "is_current": int(candidate == current),
                "base_score_field": structure_key,
                "base_score": round(score, 8),
                **metrics,
            }
            all_summary.append(record)
            summaries_for_image.append(record)
            processed += 1
            print(f"[{processed:>3}/{total_candidates}] {image_name} | {candidate}")

        cur = next((r for r in summaries_for_image if r["candidate"] == current), None)
        if cur is None:
            decisions.append({"image": image_name, "current_candidate": current, "recommended_candidate": current,
                              "rule_triggered": 0, "decision_reason": "current_candidate_not_audited"})
            continue
        current_count = int(cur["source_color_pixels_gt16"])
        current_valid = int(cur["source_color_valid_anchors"])
        current_p90 = float(cur["chroma_residual_p90"])
        # Score the current winner using the same objective as alternatives:
        # its residual ratio is current_count/current_count == 1 when count > 0.
        current_objective = float(cur["base_score"]) + (COLOR_WEIGHT if current_count > 0 else 0.0)
        eligible_alternatives = []
        if current_valid > 0 and current_count > 0:
            for alt in summaries_for_image:
                if alt["candidate"] == current or int(alt["source_color_valid_anchors"]) <= 0:
                    continue
                if int(alt["source_color_pixels_gt16"]) > REDUCTION_RATIO_MAX * current_count:
                    continue
                if float(alt["chroma_residual_p90"]) > CHROMA_P90_RATIO_MAX * current_p90:
                    continue
                ratio = int(alt["source_color_pixels_gt16"]) / max(1, current_count)
                alt["rule_objective"] = float(alt["base_score"]) + COLOR_WEIGHT * ratio
                eligible_alternatives.append(alt)
        if eligible_alternatives:
            best_alternative = min(
                eligible_alternatives,
                key=lambda r: (float(r["rule_objective"]), float(r["base_score"]), str(r["candidate"])),
            )
            # Do not switch merely because an alternative passes the color guards:
            # it must also improve the combined objective over the current winner.
            if float(best_alternative["rule_objective"]) + 1e-9 < current_objective:
                best = best_alternative
                recommended = str(best["candidate"])
                triggered = recommended != current
                reason = "alternative_improved_current_objective"
            else:
                best, recommended, triggered = None, current, False
                reason = "best_alternative_not_better_than_current_objective"
        else:
            best, recommended, triggered = None, current, False
            reason = "no_alternative_passed_color_guards"
        decision = {
            "image": image_name,
            "sample_group": group_for_candidate(current),
            "current_candidate": current,
            "recommended_candidate": recommended,
            "rule_triggered": int(triggered),
            "decision_reason": reason,
            "valid_source_color_anchors_current": current_valid,
            "current_source_color_pixels_gt16": current_count,
            "recommended_source_color_pixels_gt16": int(best["source_color_pixels_gt16"]) if best else current_count,
            "source_color_gt16_reduction_pct": round(100 * (1 - int(best["source_color_pixels_gt16"]) / max(1, current_count)), 2) if best and current_count else 0.0,
            "current_chroma_residual_p90": round(current_p90, 6),
            "recommended_chroma_residual_p90": round(float(best["chroma_residual_p90"]), 6) if best else round(current_p90, 6),
            "current_base_score": round(float(cur["base_score"]), 8),
            "current_rule_objective": round(current_objective, 8),
            "recommended_base_score": round(float(best["base_score"]), 8) if best else round(float(cur["base_score"]), 8),
            "recommended_rule_objective": round(float(best["rule_objective"]), 8) if best else round(current_objective, 8),
            "candidate_variant_count_audited": len(summaries_for_image),
            "anchor_count": len(anchors),
        }
        decisions.append(decision)
        if triggered:
            changed_reviews.append({
                "image": image_name,
                "sample_group": group_for_candidate(current),
                "current_candidate": current,
                "recommended_candidate": recommended,
                "rule_triggered": "1",
                "source_color_gt16_reduction_pct": str(decision["source_color_gt16_reduction_pct"]),
                "current_source_color_pixels_gt16": str(current_count),
                "recommended_source_color_pixels_gt16": str(best["source_color_pixels_gt16"]),
                "current_chroma_residual_p90": str(decision["current_chroma_residual_p90"]),
                "recommended_chroma_residual_p90": str(decision["recommended_chroma_residual_p90"]),
                "visual_choice": "",  # fill only after inspecting contact sheets
                "review_notes": "",
            })

    fields = list(dict.fromkeys(k for row in all_summary for k in row.keys())) if all_summary else ["image", "candidate"]
    with (output_dir / "holdout_candidate_metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(all_summary)
    decision_fields = list(decisions[0].keys()) if decisions else ["image", "current_candidate", "recommended_candidate"]
    with (output_dir / "holdout_rule_decisions.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=decision_fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(decisions)
    review_fields = ["image", "sample_group", "current_candidate", "recommended_candidate", "rule_triggered",
                     "source_color_gt16_reduction_pct", "current_source_color_pixels_gt16",
                     "recommended_source_color_pixels_gt16", "current_chroma_residual_p90",
                     "recommended_chroma_residual_p90", "visual_choice", "review_notes"]
    with (output_dir / "holdout_manual_review.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=review_fields, extrasaction="raise")
        writer.writeheader(); writer.writerows(changed_reviews)

    sheet_count = make_contact_sheets(changed_reviews, output_dir, original_map, anchor_map)
    print("\n========== INDEPENDENTER HOLDOUT-TEST ABGESCHLOSSEN ==========")
    print(f"Unabhängige Bilder ausgewählt: {len(selected)}")
    print(f"Kandidatenvarianten geprüft: {len(all_summary)}")
    print(f"Regel würde Auswahl ändern: {len(changed_reviews)}")
    print(f"Vergleichsbögen geschrieben: {sheet_count}")
    print(f"Ausgabeordner: {output_dir}")
    print("Die Spalte visual_choice bleibt absichtlich leer; die unabhängige Sichtprüfung steht noch aus.")
    print("Nur CPU/Lesen. Keine GPU-Inferenz, keine Produktionsdateien und keine Kandidatenbilder verändert.")


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
