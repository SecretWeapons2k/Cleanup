#!/usr/bin/env python3
"""Read-only CPU counterfactual audit of the color-residual rule over the saved full run.

This script reuses the existing holdout audit's metrics and thresholds. It does NOT
run inference, edit scoring.py, change the saved winner report, or overwrite candidate
images. It writes a timestamped report folder beneath the existing run's debug_candidates.
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
    print("Bitte im Cleanup-Projekt mit `uv run python ...` ausführen.")
    raise

PROJECT = audit.PROJECT
RUN_DIR = audit.RUN_DIR
DEBUG_DIR = audit.DEBUG_DIR
CANDIDATE_DIR = audit.CANDIDATE_DIR
WINNER_REPORT = audit.WINNER_REPORT
CANDIDATE_SCORES = audit.CANDIDATE_SCORES
ORIGINAL_DIR = audit.ORIGINAL_DIR
CALIBRATION_CASES = set(audit.CALIBRATION_CASES)


def norm_stem(value: str) -> str:
    return Path(value.strip()).stem.lower()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    if not fieldnames:
        fieldnames = ["image"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_color(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Bild kann nicht gelesen werden: {path}")
    return image


def add_anchor_rectangles(image: np.ndarray, boxes: list[tuple[int, int, int, int]], sx: float, sy: float) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    for x1, y1, x2, y2 in boxes:
        xx1 = max(0, min(w - 1, int(round(x1 * sx))))
        yy1 = max(0, min(h - 1, int(round(y1 * sy))))
        xx2 = max(0, min(w - 1, int(round(x2 * sx)) - 1))
        yy2 = max(0, min(h - 1, int(round(y2 * sy)) - 1))
        if xx2 >= xx1 and yy2 >= yy1:
            cv2.rectangle(out, (xx1, yy1), (xx2, yy2), (0, 0, 255), 1, cv2.LINE_AA)
    return out


def crop_anchor_envelope(image: np.ndarray, boxes: list[tuple[int, int, int, int]], sx: float, sy: float, pad: int = 12) -> np.ndarray:
    h, w = image.shape[:2]
    if not boxes:
        return image
    scaled = [(int(round(x1*sx)), int(round(y1*sy)), int(round(x2*sx)), int(round(y2*sy))) for x1,y1,x2,y2 in boxes]
    x1 = max(0, min(b[0] for b in scaled) - pad)
    y1 = max(0, min(b[1] for b in scaled) - pad)
    x2 = min(w, max(b[2] for b in scaled) + pad)
    y2 = min(h, max(b[3] for b in scaled) + pad)
    if x2 <= x1 or y2 <= y1:
        return image
    return image[y1:y2, x1:x2]


def render_panel(image: np.ndarray, boxes: list[tuple[int, int, int, int]], sx: float, sy: float,
                 tile_w: int = 480, tile_h: int = 230) -> np.ndarray:
    marked = add_anchor_rectangles(image, boxes, sx, sy)
    crop = crop_anchor_envelope(marked, boxes, sx, sy)
    h, w = crop.shape[:2]
    scale = min((tile_w - 12) / max(1, w), (tile_h - 12) / max(1, h))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    thumb = cv2.resize(crop, (nw, nh), interpolation=interp)
    panel = np.full((tile_h, tile_w, 3), 248, dtype=np.uint8)
    x0, y0 = (tile_w - nw) // 2, (tile_h - nh) // 2
    panel[y0:y0+nh, x0:x0+nw] = thumb
    return panel


def make_context_sheets(changed: list[dict], original_map: dict[str, list[Path]], anchor_map: dict[str, list[dict[str, str]]], output_dir: Path) -> int:
    if not changed:
        return 0
    tile_w, tile_h, label_h = 480, 230, 48
    columns = 3
    per_sheet = 5
    sheets = 0
    for start in range(0, len(changed), per_sheet):
        batch = changed[start:start+per_sheet]
        row_h = label_h + tile_h
        canvas = np.full((len(batch)*row_h, columns*tile_w, 3), 248, dtype=np.uint8)
        for i, record in enumerate(batch):
            image_name = record["image"]
            stem = norm_stem(image_name)
            originals = original_map.get(stem, [])
            if len(originals) != 1:
                # Mirror the existing audit's ambiguity check; do not guess an original.
                try:
                    original_path = audit.find_original(image_name, original_map)
                except Exception:
                    continue
            else:
                original_path = originals[0]
            original = load_color(original_path)
            oh, ow = original.shape[:2]
            anchors = anchor_map.get(stem, [])
            boxes = []
            for anchor in anchors:
                box = audit.clipped_box(anchor, ow, oh)
                if box:
                    boxes.append(box)
            panels = []
            panels.append(render_panel(original, boxes, 1.0, 1.0, tile_w, tile_h))
            for candidate in (record["current_candidate"], record["recommended_candidate"]):
                path = audit.candidate_path(image_name, candidate)
                if not path.is_file():
                    panels.append(np.full((tile_h, tile_w, 3), 248, dtype=np.uint8))
                    continue
                img = load_color(path)
                ch, cw = img.shape[:2]
                panels.append(render_panel(img, boxes, cw/max(1,ow), ch/max(1,oh), tile_w, tile_h))
            y0 = i * row_h
            header = (f'{image_name}: {record["current_candidate"]} -> {record["recommended_candidate"]} '
                      f'| Δobjective={record.get("objective_improvement", "")}')
            cv2.putText(canvas, header[:140], (8, y0+18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20,20,20), 1, cv2.LINE_AA)
            for col, (title, panel) in enumerate(zip(("ORIGINAL (red = OCR anchor)", "CURRENT", "PROPOSED"), panels)):
                x0 = col*tile_w
                cv2.putText(canvas, title, (x0+8, y0+42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (35,35,35), 1, cv2.LINE_AA)
                canvas[y0+label_h:y0+label_h+tile_h, x0:x0+tile_w] = panel
        target = output_dir / f"full_population_context_{sheets+1:03d}.jpg"
        cv2.imwrite(str(target), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        sheets += 1
    return sheets


def main() -> None:
    for path in (DEBUG_DIR, CANDIDATE_DIR, ORIGINAL_DIR, WINNER_REPORT, CANDIDATE_SCORES):
        if not path.exists():
            raise FileNotFoundError(f"Benötigter vorhandener Projektpfad fehlt: {path}")

    anchors_path = audit.find_anchor_csv()
    winner_rows = audit.read_csv(WINNER_REPORT)
    score_rows = audit.read_csv(CANDIDATE_SCORES)
    anchor_rows = audit.read_csv(anchors_path)
    if not winner_rows or not {"image", "selected_distance_px"}.issubset(winner_rows[0]):
        raise ValueError("winner_report.csv hat nicht die erwarteten Spalten.")
    if not score_rows:
        raise ValueError("candidate_scores.csv ist leer oder nicht lesbar.")
    if not anchor_rows or not {"image", "x1", "y1", "x2", "y2"}.issubset(anchor_rows[0]):
        raise ValueError("anchors.csv hat nicht die erwarteten Spalten.")

    for row in score_rows:
        if not (row.get("candidate") or "").strip():
            row["candidate"] = (row.get("distance_px") or "").strip()

    score_headers = set(score_rows[0])
    structure_key = "structure_score" if "structure_score" in score_headers else None
    if not structure_key:
        raise ValueError("candidate_scores.csv enthält keine Spalte structure_score.")

    anchor_map: dict[str, list[dict[str, str]]] = defaultdict(list)
    scores_by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in anchor_rows:
        if row.get("image", "").strip():
            anchor_map[norm_stem(row["image"])].append(row)
    for row in score_rows:
        if row.get("image", "").strip() and row.get("candidate", "").strip():
            scores_by_image[norm_stem(row["image"])].append(row)

    original_map: dict[str, list[Path]] = defaultdict(list)
    for p in ORIGINAL_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in audit.IMAGE_EXTS:
            original_map[p.stem.lower()].append(p)

    # Reproduce the audit's deterministic 24-case sample for labeling only.
    sampled_holdout = audit.select_holdout(winner_rows, anchor_map, scores_by_image)
    holdout_stems = {norm_stem(row["image"]) for row in sampled_holdout}

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = DEBUG_DIR / f"counterfactual_color_rule_full_population_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    all_metrics: list[dict] = []
    decisions: list[dict] = []
    changed: list[dict] = []
    total_candidate_count = sum(len(scores_by_image.get(norm_stem(row.get("image", "")), [])) for row in winner_rows)
    processed_candidates = 0
    skipped = Counter()

    print(f"Vollpopulation: {len(winner_rows)} Gewinnerzeilen; Kandidatenvarianten geplant: {total_candidate_count}")
    print("CPU/Lesen; keine GPU-Inferenz; keine Produktionsdateien oder Kandidatenbilder werden verändert.")

    for image_i, winner in enumerate(winner_rows, start=1):
        image_name = winner.get("image", "").strip()
        current = winner.get("selected_distance_px", "").strip()
        stem = norm_stem(image_name)
        anchors = anchor_map.get(stem, [])
        candidates = scores_by_image.get(stem, [])
        if not image_name or not current:
            skipped["missing_winner_identity"] += 1
            continue
        if not anchors:
            skipped["no_anchors"] += 1
            decisions.append({"image":image_name,"current_candidate":current,"recommended_candidate":current,"rule_triggered":0,"decision_reason":"no_anchors","is_calibration_case":int(stem in CALIBRATION_CASES),"is_holdout_sample":int(stem in holdout_stems)})
            continue
        if not candidates:
            skipped["no_candidate_rows"] += 1
            decisions.append({"image":image_name,"current_candidate":current,"recommended_candidate":current,"rule_triggered":0,"decision_reason":"no_candidate_rows","is_calibration_case":int(stem in CALIBRATION_CASES),"is_holdout_sample":int(stem in holdout_stems)})
            continue
        try:
            original_path = audit.find_original(image_name, original_map)
            original_lab = audit.load_lab(original_path)
        except Exception as exc:
            skipped["original_load_error"] += 1
            decisions.append({"image":image_name,"current_candidate":current,"recommended_candidate":current,"rule_triggered":0,"decision_reason":f"original_error:{type(exc).__name__}","is_calibration_case":int(stem in CALIBRATION_CASES),"is_holdout_sample":int(stem in holdout_stems)})
            continue

        image_metrics: list[dict] = []
        for candidate_row in candidates:
            candidate = candidate_row.get("candidate", "").strip()
            if not candidate:
                continue
            img_path = audit.candidate_path(image_name, candidate)
            if not img_path.is_file():
                skipped["candidate_file_missing"] += 1
                continue
            try:
                candidate_lab = audit.load_lab(img_path)
                metrics = audit.metric_for_candidate(image_name, candidate, anchors, original_lab, candidate_lab)
                base_score = audit.as_float(candidate_row, structure_key)
            except Exception as exc:
                skipped["candidate_metric_error"] += 1
                print(f"WARNUNG {image_name} | {candidate}: {type(exc).__name__}: {exc}")
                continue
            record = {
                "image":image_name,
                "sample_group":audit.group_for_candidate(current),
                "current_candidate":current,
                "candidate":candidate,
                "is_current":int(candidate == current),
                "base_score_field":structure_key,
                "base_score":round(base_score,8),
                "is_calibration_case":int(stem in CALIBRATION_CASES),
                "is_holdout_sample":int(stem in holdout_stems),
                **metrics,
            }
            all_metrics.append(record)
            image_metrics.append(record)
            processed_candidates += 1
        cur = next((rec for rec in image_metrics if rec["candidate"] == current), None)
        if cur is None:
            skipped["current_candidate_missing_or_unscored"] += 1
            decisions.append({"image":image_name,"current_candidate":current,"recommended_candidate":current,"rule_triggered":0,"decision_reason":"current_candidate_not_audited","candidate_variant_count_audited":len(image_metrics),"anchor_count":len(anchors),"is_calibration_case":int(stem in CALIBRATION_CASES),"is_holdout_sample":int(stem in holdout_stems)})
            continue

        current_count = int(cur["source_color_pixels_gt16"])
        current_valid = int(cur["source_color_valid_anchors"])
        current_p90 = float(cur["chroma_residual_p90"])
        current_objective = float(cur["base_score"]) + (audit.COLOR_WEIGHT if current_count > 0 else 0.0)
        eligible = []
        if current_valid > 0 and current_count > 0:
            for alt in image_metrics:
                if alt["candidate"] == current or int(alt["source_color_valid_anchors"]) <= 0:
                    continue
                if int(alt["source_color_pixels_gt16"]) > audit.REDUCTION_RATIO_MAX * current_count:
                    continue
                if float(alt["chroma_residual_p90"]) > audit.CHROMA_P90_RATIO_MAX * current_p90:
                    continue
                ratio = int(alt["source_color_pixels_gt16"]) / max(1,current_count)
                alt["rule_objective"] = float(alt["base_score"]) + audit.COLOR_WEIGHT * ratio
                eligible.append(alt)
        if eligible:
            best_alt = min(eligible,key=lambda rec:(float(rec["rule_objective"]),float(rec["base_score"]),str(rec["candidate"])))
            if float(best_alt["rule_objective"]) + 1e-9 < current_objective:
                best = best_alt
                recommended = str(best["candidate"])
                triggered = recommended != current
                reason = "alternative_improved_current_objective"
            else:
                best = None
                recommended = current
                triggered = False
                reason = "best_alternative_not_better_than_current_objective"
        else:
            best = None
            recommended = current
            triggered = False
            reason = "no_alternative_passed_color_guards"

        decision = {
            "image":image_name,
            "sample_group":audit.group_for_candidate(current),
            "current_candidate":current,
            "recommended_candidate":recommended,
            "rule_triggered":int(triggered),
            "decision_reason":reason,
            "valid_source_color_anchors_current":current_valid,
            "current_source_color_pixels_gt16":current_count,
            "recommended_source_color_pixels_gt16":int(best["source_color_pixels_gt16"]) if best else current_count,
            "source_color_gt16_reduction_pct":round(100*(1-int(best["source_color_pixels_gt16"])/max(1,current_count)),2) if best and current_count else 0.0,
            "current_chroma_residual_p90":round(current_p90,6),
            "recommended_chroma_residual_p90":round(float(best["chroma_residual_p90"]),6) if best else round(current_p90,6),
            "current_base_score":round(float(cur["base_score"]),8),
            "current_rule_objective":round(current_objective,8),
            "recommended_base_score":round(float(best["base_score"]),8) if best else round(float(cur["base_score"]),8),
            "recommended_rule_objective":round(float(best["rule_objective"]),8) if best else round(current_objective,8),
            "objective_improvement":round(current_objective-float(best["rule_objective"]),8) if best else 0.0,
            "candidate_variant_count_audited":len(image_metrics),
            "anchor_count":len(anchors),
            "is_calibration_case":int(stem in CALIBRATION_CASES),
            "is_holdout_sample":int(stem in holdout_stems),
        }
        decisions.append(decision)
        if triggered:
            changed.append(decision.copy())
        if image_i % 25 == 0 or image_i == len(winner_rows):
            print(f"Gewinner geprüft: {image_i}/{len(winner_rows)} | Kandidatenbilder geprüft: {processed_candidates} | Wechsel bisher: {len(changed)}")

    write_csv(output_dir/"full_population_candidate_metrics.csv",all_metrics)
    write_csv(output_dir/"full_population_decisions.csv",decisions)
    write_csv(output_dir/"full_population_changed_review.csv",changed)

    # Compact overview: separate calibration set, predetermined holdout sample, and remainder.
    strata = {"calibration_9":lambda d: bool(d.get("is_calibration_case",0)),
              "holdout_24":lambda d: bool(d.get("is_holdout_sample",0)) and not bool(d.get("is_calibration_case",0)),
              "remaining_population":lambda d: not bool(d.get("is_calibration_case",0)) and not bool(d.get("is_holdout_sample",0))}
    summary_lines = [
        "FULL-POPULATION COUNTERFACTUAL COLOR-RULE AUDIT",
        f"Timestamp: {stamp}",
        f"Run: {RUN_DIR}",
        f"Anchors: {audit.find_anchor_csv()}",
        "Mode: CPU/read-only; no GPU inference; production scoring.py untouched; no candidate files modified.",
        f"Winner rows: {len(winner_rows)}",
        f"Candidates scored by this audit: {processed_candidates}",
        f"Decision rows written: {len(decisions)}",
        f"Proposed winner changes: {len(changed)}",
        "",
        "DECISION STRATA (calibration and holdout counts are labels, not separate reruns):",
    ]
    for label, predicate in strata.items():
        group=[d for d in decisions if predicate(d)]
        summary_lines.append(f"{label}: images={len(group)}, proposed_changes={sum(int(d.get('rule_triggered',0)) for d in group)}")
    summary_lines += ["", "CHANGED WINNERS:"]
    if not changed:
        summary_lines.append("(none)")
    else:
        for d in changed:
            summary_lines.append(
                f"{d['image']}: {d['current_candidate']} -> {d['recommended_candidate']} | "
                f"objective {d.get('current_rule_objective')} -> {d.get('recommended_rule_objective')} | "
                f"residual pixels {d.get('current_source_color_pixels_gt16')} -> {d.get('recommended_source_color_pixels_gt16')} | "
                f"calibration={d.get('is_calibration_case')} holdout_sample={d.get('is_holdout_sample')}"
            )
    summary_lines += ["", "SKIPPED/WARN COUNTS:"]
    for key, value in sorted(skipped.items()):
        summary_lines.append(f"{key}: {value}")
    (output_dir/"full_population_summary.txt").write_text("\n".join(summary_lines)+"\n",encoding="utf-8")

    sheet_count = make_context_sheets(changed,original_map,anchor_map,output_dir)
    print("\n========== FULL-POPULATION AUDIT FERTIG ==========")
    print(f"Bilder in winner_report.csv: {len(winner_rows)}")
    print(f"Kandidatenvarianten ausgewertet: {processed_candidates}")
    print(f"Vorgeschlagene Auswahlwechsel: {len(changed)}")
    print(f"Vergleichsbögen ORIGINAL/CURRENT/PROPOSED: {sheet_count}")
    print(f"Ausgabeordner: {output_dir}")
    print("Nur CPU/Lesen. Keine GPU-Inferenz; scoring.py, winner_report.csv und Kandidatenbilder unverändert.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nFEHLER: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        print("Das Terminal bleibt offen.")
        input("Enter zum Schließen ...")
        raise
