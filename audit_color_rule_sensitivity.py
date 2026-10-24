#!/usr/bin/env python3
"""Read-only sensitivity sweep for the anchor-color candidate selection rule.

Reads the already-computed full_population_candidate_metrics.csv and evaluates
parameter combinations. It does not read candidate images, run GPU inference,
or modify production files/candidate images.
"""
from __future__ import annotations

import csv
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

WEIGHTS = (0.12, 0.18, 0.24, 0.30)
REDUCTION_LIMITS = (0.50, 0.65, 0.75)
CHROMA_P90_LIMITS = (1.00, 1.05)
EPSILON = 1e-9
EXPECTED_BASELINE_CHANGES = 166
BASELINE_WEIGHT = 0.24
BASELINE_REDUCTION_LIMIT = 0.75
BASELINE_CHROMA_P90_LIMIT = 1.05
CANDIDATE_ORDER = ("BOX_ONLY", "SMART_INK", "CONTRAST_T38", "HYBRID_D2", "HYBRID_D6")


def find_project_root() -> Path:
    candidates = [Path.cwd(), Path(__file__).resolve().parent]
    for start in candidates:
        for folder in (start, *start.parents):
            if (folder / "yolo_mask_winners" / "full_recheck_after_detection_20261024_030641").is_dir():
                return folder
    raise SystemExit(
        "FEHLER: Projekt-Root nicht gefunden. Bitte im Ordner ~/Downloads/Cleanup starten."
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def evaluate_image(
    rows: list[dict[str, str]], weight: float, reduction_limit: float, chroma_limit: float
) -> dict[str, object]:
    current_name = rows[0].get("current_candidate", "")
    cur = next((row for row in rows if row.get("candidate") == current_name), None)
    if cur is None:
        return {
            "current_candidate": current_name,
            "recommended_candidate": current_name,
            "rule_triggered": 0,
            "reason": "current_candidate_row_missing",
            "current_count": 0,
            "recommended_count": 0,
            "current_objective": "",
            "recommended_objective": "",
            "objective_improvement": 0.0,
        }

    current_count = as_int(cur, "source_color_pixels_gt16")
    current_valid = as_int(cur, "source_color_valid_anchors")
    current_p90 = as_float(cur, "chroma_residual_p90")
    current_base = as_float(cur, "base_score")
    current_objective = current_base + (weight if current_count > 0 else 0.0)

    eligible: list[tuple[float, float, str, dict[str, str], float]] = []
    if current_valid > 0 and current_count > 0:
        for alt in rows:
            alt_name = alt.get("candidate", "")
            if alt_name == current_name or as_int(alt, "source_color_valid_anchors") <= 0:
                continue
            alt_count = as_int(alt, "source_color_pixels_gt16")
            alt_p90 = as_float(alt, "chroma_residual_p90")
            if alt_count > reduction_limit * current_count:
                continue
            if alt_p90 > chroma_limit * current_p90:
                continue
            ratio = alt_count / max(1, current_count)
            alt_base = as_float(alt, "base_score")
            alt_objective = alt_base + weight * ratio
            eligible.append((alt_objective, alt_base, alt_name, alt, ratio))

    if not eligible:
        return {
            "current_candidate": current_name,
            "recommended_candidate": current_name,
            "rule_triggered": 0,
            "reason": "no_alternative_passed_color_guards",
            "current_count": current_count,
            "recommended_count": current_count,
            "current_objective": current_objective,
            "recommended_objective": current_objective,
            "objective_improvement": 0.0,
        }

    alt_objective, _, alt_name, alt, _ = min(eligible, key=lambda item: (item[0], item[1], item[2]))
    if alt_objective + EPSILON < current_objective:
        recommended = alt_name
        recommended_count = as_int(alt, "source_color_pixels_gt16")
        return {
            "current_candidate": current_name,
            "recommended_candidate": recommended,
            "rule_triggered": int(recommended != current_name),
            "reason": "alternative_improved_current_objective",
            "current_count": current_count,
            "recommended_count": recommended_count,
            "current_objective": current_objective,
            "recommended_objective": alt_objective,
            "objective_improvement": current_objective - alt_objective,
        }

    return {
        "current_candidate": current_name,
        "recommended_candidate": current_name,
        "rule_triggered": 0,
        "reason": "best_alternative_not_better_than_current_objective",
        "current_count": current_count,
        "recommended_count": current_count,
        "current_objective": current_objective,
        "recommended_objective": current_objective,
        "objective_improvement": 0.0,
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    project = find_project_root()
    audit_dir = (
        project / "yolo_mask_winners" / "full_recheck_after_detection_20261024_030641"
        / "debug_candidates" / "counterfactual_color_rule_full_population_20261024_214800"
    )
    metrics_path = audit_dir / "full_population_candidate_metrics.csv"
    decisions_path = audit_dir / "full_population_decisions.csv"
    if not metrics_path.is_file():
        raise SystemExit(f"FEHLER: Metrics-CSV nicht gefunden: {metrics_path}")

    metric_rows = read_csv(metrics_path)
    if not metric_rows:
        raise SystemExit("FEHLER: Metrics-CSV ist leer.")
    required = {
        "image", "current_candidate", "candidate", "source_color_valid_anchors",
        "source_color_pixels_gt16", "chroma_residual_p90", "base_score"
    }
    missing = required - set(metric_rows[0])
    if missing:
        raise SystemExit("FEHLER: In Metrics-CSV fehlen Spalten: " + ", ".join(sorted(missing)))

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in metric_rows:
        grouped[row["image"]].append(row)
    image_names = sorted(grouped)

    baseline_map: dict[str, tuple[str, int]] = {}
    if decisions_path.is_file():
        for row in read_csv(decisions_path):
            baseline_map[row.get("image", "")] = (
                row.get("recommended_candidate", ""), as_int(row, "rule_triggered")
            )

    output_dir = audit_dir / ("sensitivity_sweep_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_dir.mkdir(parents=True, exist_ok=False)

    long_rows: list[dict[str, object]] = []
    grid_rows: list[dict[str, object]] = []
    baseline_recommendations: dict[str, str] = {}
    baseline_change_flags: dict[str, int] = {}
    image_results_by_scenario: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)

    scenarios = [
        (weight, reduction_limit, chroma_limit)
        for weight in WEIGHTS
        for reduction_limit in REDUCTION_LIMITS
        for chroma_limit in CHROMA_P90_LIMITS
    ]

    for weight, reduction_limit, chroma_limit in scenarios:
        scenario_id = f"w{weight:.2f}_r{reduction_limit:.2f}_p{chroma_limit:.2f}"
        scenario_results: list[dict[str, object]] = []
        for image_name in image_names:
            result = evaluate_image(grouped[image_name], weight, reduction_limit, chroma_limit)
            current = str(result["current_candidate"])
            recommended = str(result["recommended_candidate"])
            expected = baseline_map.get(image_name)
            baseline_mismatch: object = ""
            if (weight, reduction_limit, chroma_limit) == (
                BASELINE_WEIGHT, BASELINE_REDUCTION_LIMIT, BASELINE_CHROMA_P90_LIMIT
            ):
                baseline_recommendations[image_name] = recommended
                baseline_change_flags[image_name] = int(result["rule_triggered"])
                if expected is not None:
                    baseline_mismatch = int(recommended != expected[0] or int(result["rule_triggered"]) != expected[1])
            row = {
                "scenario_id": scenario_id,
                "color_weight": weight,
                "reduction_ratio_max": reduction_limit,
                "chroma_p90_ratio_max": chroma_limit,
                "image": image_name,
                "current_candidate": current,
                "recommended_candidate": recommended,
                "rule_triggered": result["rule_triggered"],
                "decision_reason": result["reason"],
                "current_source_color_pixels_gt16": result["current_count"],
                "recommended_source_color_pixels_gt16": result["recommended_count"],
                "current_objective": result["current_objective"],
                "recommended_objective": result["recommended_objective"],
                "objective_improvement": result["objective_improvement"],
                "baseline_decision_mismatch": baseline_mismatch,
            }
            long_rows.append(row)
            scenario_results.append(row)
            image_results_by_scenario[image_name][scenario_id] = row

        changed = [row for row in scenario_results if as_int(row, "rule_triggered") == 1]
        dest_counts = Counter(str(row["recommended_candidate"]) for row in changed)
        gains = [as_float(row, "objective_improvement") for row in changed]
        residuals = [as_int(row, "recommended_source_color_pixels_gt16") for row in changed]
        mismatch_count = sum(as_int(row, "baseline_decision_mismatch") for row in scenario_results if row.get("baseline_decision_mismatch", "") != "")
        grid_row: dict[str, object] = {
            "scenario_id": scenario_id,
            "color_weight": weight,
            "reduction_ratio_max": reduction_limit,
            "chroma_p90_ratio_max": chroma_limit,
            "images_evaluated": len(image_names),
            "proposed_changes": len(changed),
            "proposed_change_pct": round(100 * len(changed) / max(1, len(image_names)), 2),
            "changes_from_CONTRAST_T38": sum(str(r["current_candidate"]) == "CONTRAST_T38" for r in changed),
            "changes_to_BOX_ONLY": dest_counts.get("BOX_ONLY", 0),
            "changes_to_SMART_INK": dest_counts.get("SMART_INK", 0),
            "changes_to_CONTRAST_T38": dest_counts.get("CONTRAST_T38", 0),
            "changes_to_HYBRID_D2": dest_counts.get("HYBRID_D2", 0),
            "changes_to_HYBRID_D6": dest_counts.get("HYBRID_D6", 0),
            "mean_objective_improvement": round(statistics.mean(gains), 8) if gains else 0.0,
            "median_objective_improvement": round(statistics.median(gains), 8) if gains else 0.0,
            "recommended_residual_zero": sum(x == 0 for x in residuals),
            "recommended_residual_at_most_10": sum(x <= 10 for x in residuals),
            "baseline_decision_mismatches": mismatch_count,
        }
        grid_rows.append(grid_row)

    stability_rows: list[dict[str, object]] = []
    for image_name in image_names:
        rows = list(image_results_by_scenario[image_name].values())
        current = str(rows[0]["current_candidate"])
        recommendations = Counter(str(row["recommended_candidate"]) for row in rows)
        alternatives = Counter(str(row["recommended_candidate"]) for row in rows if str(row["recommended_candidate"]) != current)
        changed_scenarios = sum(as_int(row, "rule_triggered") for row in rows)
        if alternatives:
            dominant_alt, dominant_alt_count = sorted(alternatives.items(), key=lambda item: (-item[1], item[0]))[0]
        else:
            dominant_alt, dominant_alt_count = "", 0
        baseline_expected = baseline_map.get(image_name, ("", 0))[0]
        baseline_recommended = baseline_recommendations.get(image_name, current)
        baseline_mismatch = int(baseline_expected != baseline_recommended) if baseline_expected else ""
        stability_rows.append({
            "image": image_name,
            "current_candidate": current,
            "baseline_recommended_candidate": baseline_recommended,
            "baseline_rule_triggered": baseline_change_flags.get(image_name, 0),
            "baseline_decision_mismatch": baseline_mismatch,
            "scenarios_tested": len(rows),
            "scenarios_selecting_any_alternative": changed_scenarios,
            "change_rate_pct": round(100 * changed_scenarios / max(1, len(rows)), 2),
            "dominant_alternative": dominant_alt,
            "dominant_alternative_scenarios": dominant_alt_count,
            "dominant_alternative_rate_pct_of_all_scenarios": round(100 * dominant_alt_count / max(1, len(rows)), 2),
            "distinct_alternative_candidates": len(alternatives),
            "dominant_alternative_is_baseline_pick": int(bool(dominant_alt) and dominant_alt == baseline_recommended),
            "stable_alternative_18_of_24": int(bool(dominant_alt) and dominant_alt_count >= 18),
            "all_scenarios_choose_current": int(not alternatives),
        })

    long_fields = [
        "scenario_id", "color_weight", "reduction_ratio_max", "chroma_p90_ratio_max", "image",
        "current_candidate", "recommended_candidate", "rule_triggered", "decision_reason",
        "current_source_color_pixels_gt16", "recommended_source_color_pixels_gt16",
        "current_objective", "recommended_objective", "objective_improvement", "baseline_decision_mismatch"
    ]
    grid_fields = list(grid_rows[0]) if grid_rows else []
    stability_fields = list(stability_rows[0]) if stability_rows else []
    write_csv(output_dir / "sensitivity_recommendations_long.csv", long_rows, long_fields)
    write_csv(output_dir / "sensitivity_grid_summary.csv", grid_rows, grid_fields)
    write_csv(output_dir / "sensitivity_image_stability.csv", stability_rows, stability_fields)

    baseline_scenario = next((r for r in grid_rows if
        float(r["color_weight"]) == BASELINE_WEIGHT and
        float(r["reduction_ratio_max"]) == BASELINE_REDUCTION_LIMIT and
        float(r["chroma_p90_ratio_max"]) == BASELINE_CHROMA_P90_LIMIT), None)
    baseline_mismatches = baseline_scenario["baseline_decision_mismatches"] if baseline_scenario else "unavailable"
    stable_count = sum(as_int(row, "stable_alternative_18_of_24") for row in stability_rows)
    stable_baseline_count = sum(
        as_int(row, "stable_alternative_18_of_24") and as_int(row, "dominant_alternative_is_baseline_pick")
        for row in stability_rows
    )

    summary_lines = [
        "ANCHOR COLOR RULE — READ-ONLY SENSITIVITY SWEEP",
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Metrics input: {metrics_path}",
        f"Decision reference: {decisions_path if decisions_path.is_file() else 'not found'}",
        "Mode: CSV-only; no image loading; no GPU; no production files or candidate images changed.",
        f"Images: {len(image_names)}",
        f"Candidate metrics rows: {len(metric_rows)}",
        f"Parameter combinations: {len(scenarios)}",
        f"Weight values: {', '.join(str(x) for x in WEIGHTS)}",
        f"Maximum residual ratios: {', '.join(str(x) for x in REDUCTION_LIMITS)}",
        f"Maximum chroma-p90 ratios: {', '.join(str(x) for x in CHROMA_P90_LIMITS)}",
        "",
        "BASELINE CHECK (weight=0.24, reduction<=0.75, chroma-p90<=1.05):",
        f"Computed proposed changes: {baseline_scenario['proposed_changes'] if baseline_scenario else 'unavailable'}",
        f"Expected based on original audit: {EXPECTED_BASELINE_CHANGES}",
        f"Differences versus full_population_decisions.csv: {baseline_mismatches}",
        "",
        "HOW TO READ THE OUTPUTS:",
        "sensitivity_grid_summary.csv: change counts for each parameter combination.",
        "sensitivity_image_stability.csv: per-image stability across the parameter sweep.",
        "sensitivity_recommendations_long.csv: full per-image result for every combination.",
        f"Images with the same alternative selected in at least 18/{len(scenarios)} combinations: {stable_count}",
        f"Of those, alternatives matching the baseline recommendation: {stable_baseline_count}",
        "",
        "This is a sensitivity analysis, not a production change. Do not apply all switches automatically based only on this report.",
    ]
    (output_dir / "sensitivity_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("========== SENSITIVITY SWEEP FERTIG ==========")
    print(f"Bilder: {len(image_names)} | Kandidatenmetriken: {len(metric_rows)}")
    print(f"Parameterkombinationen: {len(scenarios)}")
    if baseline_scenario:
        print(f"Baseline-Wechsel (0.24 / 0.75 / 1.05): {baseline_scenario['proposed_changes']} (erwartet {EXPECTED_BASELINE_CHANGES})")
        print(f"Abweichungen zur Original-Entscheidung: {baseline_mismatches}")
    print(f"Stabiler Alternativkandidat in mindestens 18/{len(scenarios)} Kombinationen: {stable_count}")
    print(f"Ausgabeordner: {output_dir}")
    print("Nur vorhandene CSV-Metriken gelesen. Keine Bilder/GPU-Inferenz; scoring.py, winner_report.csv und Kandidaten unverändert.")


if __name__ == "__main__":
    main()
