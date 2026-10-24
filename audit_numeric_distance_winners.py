#!/usr/bin/env python3
"""Read-only color-rule audit for the 145 numeric DISTANCE_* current winners.

The earlier full-population audit evaluated the five named mask candidates but
could not evaluate the current winner where winner_report.csv stores a numeric
selected_distance_px (0, 2, 4, 6, 8, 10). This script maps those values to the
saved DISTANCE_XXPX candidate image, measures its existing color residuals, and
compares it against the same five named candidate variants using the existing
rule. It does not run inference or modify production reports/candidate images.
"""
from __future__ import annotations

import csv
import re
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import audit_anchor_color_rule_holdout as audit
import audit_anchor_color_rule_full_population as full_audit

AUDIT_DIR = audit.DEBUG_DIR / "counterfactual_color_rule_full_population_20261024_214800"
METRICS_CSV = AUDIT_DIR / "full_population_candidate_metrics.csv"
WINNER_REPORT = audit.WINNER_REPORT
CANDIDATE_NAMES = ("BOX_ONLY", "SMART_INK", "CONTRAST_T38", "HYBRID_D2", "HYBRID_D6")
EPSILON = 1e-9


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Benötigte Datei fehlt: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(dict.fromkeys(k for row in rows for k in row))
    if not fields:
        fields = ["image"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def norm_stem(name: str) -> str:
    return Path(name.strip()).stem.lower()


def as_int(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, "")))
    except (TypeError, ValueError):
        return default


def num(row: dict[str, str], *keys: str) -> float:
    for key in keys:
        value = row.get(key, "")
        if value is not None and str(value).strip():
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    raise ValueError(f"Keine gültige Zahl in Spalten {keys}: {row}")


def numeric_distance(value: str) -> int | None:
    raw = (value or "").strip()
    if not re.fullmatch(r"\d+(?:\.0+)?", raw):
        return None
    distance = int(float(raw))
    if distance not in (0, 2, 4, 6, 8, 10):
        return None
    return distance


def main() -> None:
    for path in (AUDIT_DIR, audit.CANDIDATE_DIR, audit.ORIGINAL_DIR, WINNER_REPORT, METRICS_CSV):
        if not path.exists():
            raise FileNotFoundError(f"Benötigter vorhandener Pfad fehlt: {path}")

    winners = read_csv(WINNER_REPORT)
    all_candidate_metrics = read_csv(METRICS_CSV)
    if not winners or not {"image", "selected_distance_px"}.issubset(winners[0]):
        raise ValueError("winner_report.csv hat nicht die erwarteten Spalten.")
    required = {"image", "candidate", "base_score", "source_color_valid_anchors",
                "source_color_pixels_gt16", "chroma_residual_p90"}
    if not all_candidate_metrics or required - set(all_candidate_metrics[0]):
        missing = sorted(required - set(all_candidate_metrics[0])) if all_candidate_metrics else sorted(required)
        raise ValueError("Candidate metrics CSV fehlen Spalten: " + ", ".join(missing))

    anchors_by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(audit.find_anchor_csv()):
        if row.get("image", "").strip():
            anchors_by_image[norm_stem(row["image"])].append(row)

    original_map: dict[str, list[Path]] = defaultdict(list)
    for p in audit.ORIGINAL_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in audit.IMAGE_EXTS:
            original_map[p.stem.lower()].append(p)

    named_metrics_by_stem: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in all_candidate_metrics:
        if row.get("image", "").strip() and row.get("candidate", "").strip():
            named_metrics_by_stem[norm_stem(row["image"])].append(row)

    targets = []
    for winner in winners:
        distance = numeric_distance(winner.get("selected_distance_px", ""))
        if distance is not None:
            targets.append((winner, distance))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = AUDIT_DIR / f"numeric_distance_winner_audit_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)

    decisions: list[dict[str, object]] = []
    current_metric_rows: list[dict[str, object]] = []
    changed: list[dict[str, object]] = []
    skipped: Counter[str] = Counter()

    print(f"Numerische Distanz-Gewinner zur Prüfung: {len(targets)}")
    print("CPU/Lesen; keine GPU-Inferenz; keine Kandidatenbilder oder Produktionsdateien werden verändert.")

    for idx, (winner, distance) in enumerate(targets, 1):
        image_name = winner["image"].strip()
        stem = norm_stem(image_name)
        current = f"DISTANCE_{distance:02d}PX"
        anchors = anchors_by_image.get(stem, [])
        if not anchors:
            skipped["no_anchors"] += 1
            decisions.append({"image": image_name, "raw_selected_distance_px": distance,
                              "current_candidate": current, "recommended_candidate": current,
                              "rule_triggered": 0, "decision_reason": "no_anchors"})
            continue

        try:
            original_path = audit.find_original(image_name, original_map)
            original_lab = audit.load_lab(original_path)
            current_path = audit.candidate_path(image_name, current)
            if not current_path.is_file():
                skipped["current_distance_candidate_missing"] += 1
                decisions.append({"image": image_name, "raw_selected_distance_px": distance,
                                  "current_candidate": current, "recommended_candidate": current,
                                  "rule_triggered": 0, "decision_reason": "current_distance_candidate_missing",
                                  "expected_current_path": str(audit.CANDIDATE_DIR / f"{Path(image_name).stem}__distance_{distance:02d}px.png")})
                continue
            current_lab = audit.load_lab(current_path)
            current_color_metrics = audit.metric_for_candidate(
                image_name, current, anchors, original_lab, current_lab
            )
            current_base = num(winner, "selected_structure_score", "selected_score")
        except Exception as exc:
            skipped[f"current_metric_error:{type(exc).__name__}"] += 1
            decisions.append({"image": image_name, "raw_selected_distance_px": distance,
                              "current_candidate": current, "recommended_candidate": current,
                              "rule_triggered": 0, "decision_reason": f"current_metric_error:{type(exc).__name__}",
                              "error": str(exc)[:300]})
            continue

        current_record: dict[str, object] = {
            "image": image_name, "sample_group": "distance", "current_candidate": current,
            "candidate": current, "is_current": 1, "base_score_field": "selected_structure_score",
            "base_score": round(current_base, 8), **current_color_metrics,
        }
        current_metric_rows.append(current_record)
        current_count = int(current_color_metrics["source_color_pixels_gt16"])
        current_valid = int(current_color_metrics["source_color_valid_anchors"])
        current_chroma_p90 = float(current_color_metrics["chroma_residual_p90"])
        current_objective = current_base + (audit.COLOR_WEIGHT if current_count > 0 else 0.0)

        alternatives = [r for r in named_metrics_by_stem.get(stem, [])
                       if r.get("candidate", "").strip().upper() in CANDIDATE_NAMES]
        alt_by_name = {r["candidate"].strip().upper(): r for r in alternatives}
        missing_alternatives = sorted(set(CANDIDATE_NAMES) - set(alt_by_name))
        if missing_alternatives:
            skipped["named_candidate_metrics_missing"] += 1

        eligible = []
        if current_valid > 0 and current_count > 0:
            for candidate_name in CANDIDATE_NAMES:
                alt = alt_by_name.get(candidate_name)
                if alt is None or as_int(alt, "source_color_valid_anchors") <= 0:
                    continue
                alt_count = as_int(alt, "source_color_pixels_gt16")
                alt_p90 = num(alt, "chroma_residual_p90")
                if alt_count > audit.REDUCTION_RATIO_MAX * current_count:
                    continue
                if alt_p90 > audit.CHROMA_P90_RATIO_MAX * current_chroma_p90:
                    continue
                alt_base = num(alt, "base_score")
                ratio = alt_count / max(1, current_count)
                alt_objective = alt_base + audit.COLOR_WEIGHT * ratio
                eligible.append({"candidate": candidate_name, "base_score": alt_base,
                                 "source_color_pixels_gt16": alt_count,
                                 "source_color_valid_anchors": as_int(alt, "source_color_valid_anchors"),
                                 "chroma_residual_p90": alt_p90, "ratio": ratio,
                                 "rule_objective": alt_objective})

        best = min(eligible, key=lambda r: (r["rule_objective"], r["base_score"], r["candidate"])) if eligible else None
        if best and float(best["rule_objective"]) + EPSILON < current_objective:
            recommended = str(best["candidate"])
            triggered = recommended != current
            reason = "alternative_improved_numeric_current_objective"
        else:
            best = None
            recommended = current
            triggered = False
            reason = "no_eligible_alternative_better_than_numeric_current"

        chosen_count = int(best["source_color_pixels_gt16"]) if best else current_count
        rec: dict[str, object] = {
            "image": image_name,
            "raw_selected_distance_px": distance,
            "sample_group": "distance",
            "current_candidate": current,
            "recommended_candidate": recommended,
            "rule_triggered": int(triggered),
            "decision_reason": reason,
            "current_base_score": round(current_base, 8),
            "current_source_color_valid_anchors": current_valid,
            "current_source_color_pixels_gt16": current_count,
            "recommended_source_color_pixels_gt16": chosen_count,
            "source_color_gt16_reduction_pct": round(100 * (1 - chosen_count / max(1, current_count)), 2) if best and current_count else 0.0,
            "current_chroma_residual_p90": round(current_chroma_p90, 6),
            "current_rule_objective": round(current_objective, 8),
            "recommended_base_score": round(float(best["base_score"]), 8) if best else round(current_base, 8),
            "recommended_chroma_residual_p90": round(float(best["chroma_residual_p90"]), 6) if best else round(current_chroma_p90, 6),
            "recommended_rule_objective": round(float(best["rule_objective"]), 8) if best else round(current_objective, 8),
            "objective_improvement": round(current_objective - float(best["rule_objective"]), 8) if best else 0.0,
            "eligible_alternative_count": len(eligible),
            "named_candidate_metrics_count": len(alt_by_name),
            "missing_alternative_metrics": ";".join(missing_alternatives),
            "anchor_count": len(anchors),
            "current_candidate_path": str(current_path),
        }
        decisions.append(rec)
        if triggered:
            changed.append(rec.copy())
        if idx % 25 == 0 or idx == len(targets):
            print(f"Geprüft: {idx}/{len(targets)} | Distanzkandidaten vermessen: {len(current_metric_rows)} | Vorschläge: {len(changed)}")

    write_csv(out_dir / "numeric_distance_current_metrics.csv", current_metric_rows)
    write_csv(out_dir / "numeric_distance_decisions.csv", decisions)
    write_csv(out_dir / "numeric_distance_changed_review.csv", changed)

    # Reuse the already-reviewed original/current/proposed comparison renderer.
    sheet_count = full_audit.make_context_sheets(changed, original_map, anchors_by_image, out_dir)
    transitions = Counter(f"{r['current_candidate']} -> {r['recommended_candidate']}" for r in changed)
    reason_counts = Counter(str(r.get("decision_reason", "")) for r in decisions)
    lines = [
        "NUMERIC DISTANCE WINNER COLOR-RULE AUDIT (READ-ONLY)",
        f"Timestamp: {stamp}",
        f"Numeric distance winner rows found: {len(targets)}",
        f"Current distance candidate metrics computed: {len(current_metric_rows)}",
        f"Decision rows: {len(decisions)}",
        f"Proposed changes from numeric current winners: {len(changed)}",
        f"Candidate files / metric skips: {dict(skipped)}",
        f"Rule constants: COLOR_WEIGHT={audit.COLOR_WEIGHT}, REDUCTION_RATIO_MAX={audit.REDUCTION_RATIO_MAX}, CHROMA_P90_RATIO_MAX={audit.CHROMA_P90_RATIO_MAX}",
        "Named alternatives are taken from the previously computed full-population candidate metrics; the actual selected DISTANCE_XXPX winner is newly measured from its saved candidate image.",
        "No GPU inference; no scoring.py, winner_report.csv, or candidate images modified.",
        "",
        "DECISION REASONS:",
        *[f"{k}: {v}" for k, v in sorted(reason_counts.items())],
        "",
        "PROPOSED TRANSITIONS:",
        *([f"{k}: {v}" for k, v in sorted(transitions.items(), key=lambda kv: (-kv[1], kv[0]))] or ["(none)"]),
        "",
        f"Output directory: {out_dir}",
        f"Comparison sheets written: {sheet_count}",
        "This remains a counterfactual audit; it does not change the current selection.",
    ]
    (out_dir / "numeric_distance_audit_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nFEHLER: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
