#!/usr/bin/env python3
"""Build a read-only shadow selection manifest for vetted color-rule proposals.

Joins the full-population proposals, sensitivity stability, and direct current-vs-
proposed outside-anchor comparison. It does not copy or modify candidate images,
scoring.py, winner_report.csv, or any production output. The resulting manifest is
for review only and must not be fed into production automatically.
"""
from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

import audit_anchor_color_rule_holdout as audit

PROJECT = Path.home() / "Downloads" / "Cleanup"
RUN_DIR = PROJECT / "yolo_mask_winners" / "full_recheck_after_detection_20261024_030641"
DEBUG_DIR = RUN_DIR / "debug_candidates"
AUDIT_DIR = DEBUG_DIR / "counterfactual_color_rule_full_population_20261024_214800"
SWEEP_DIR = AUDIT_DIR / "sensitivity_sweep_20261024_220012"
DIRECT_DIR = (
    SWEEP_DIR / "outside_anchor_preservation_20261024_220824"
    / "direct_current_proposed_delta_20261024_221316"
)

WINNER_REPORT = RUN_DIR / "winner_report.csv"
PROPOSALS = AUDIT_DIR / "full_population_changed_review.csv"
AUDIT_DECISIONS = AUDIT_DIR / "full_population_decisions.csv"
STABILITY = SWEEP_DIR / "sensitivity_image_stability.csv"
DIRECT_METRICS = DIRECT_DIR / "direct_current_proposed_outside_metrics.csv"
CANDIDATE_DIR = DEBUG_DIR / "cleaned_images"
MIN_REDUCTION_PCT = 90.0
MAX_OUTSIDE_GT12_PCT_PAD8 = 0.25
MIN_STABLE_SCENARIOS = 18
TOTAL_SCENARIOS = 24


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Benötigte Datei fehlt: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    else:
        fields = ["image"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def num(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def image_key(value: str) -> str:
    return Path(value.strip()).stem.lower()


def canonical_candidate(value: str) -> str:
    """Convert numeric winner-report distances (e.g. 4) to DISTANCE_04PX."""
    value = value.strip()
    if value.isdigit():
        return f"DISTANCE_{int(value):02d}PX"
    return value


def main() -> None:
    winner_rows = read_csv(WINNER_REPORT)
    proposal_rows = read_csv(PROPOSALS)
    audit_decision_rows = read_csv(AUDIT_DECISIONS)
    stability_rows = read_csv(STABILITY)
    direct_rows = read_csv(DIRECT_METRICS)

    stability_by_image = {image_key(row.get("image", "")): row for row in stability_rows}
    direct_by_image = {image_key(row.get("image", "")): row for row in direct_rows}
    proposal_by_image = {image_key(row.get("image", "")): row for row in proposal_rows}
    decision_by_image = {image_key(row.get("image", "")): row for row in audit_decision_rows}
    unassessed_distance_rows = sum(1 for row in winner_rows if row.get("selected_distance_px", "").strip().isdigit())
    current_not_audited_rows = sum(1 for row in audit_decision_rows if row.get("decision_reason", "") == "current_candidate_not_audited")

    accepted: dict[str, dict[str, str]] = {}
    exclusion_counts: Counter[str] = Counter()
    for key, row in proposal_by_image.items():
        stable = stability_by_image.get(key, {})
        direct = direct_by_image.get(key, {})
        if int(num(stable, "stable_alternative_18_of_24")) != 1:
            exclusion_counts["not_stable_in_18_of_24"] += 1
            continue
        if num(row, "source_color_gt16_reduction_pct") < MIN_REDUCTION_PCT:
            exclusion_counts["residual_reduction_below_90_percent"] += 1
            continue
        if not direct:
            exclusion_counts["missing_direct_outside_audit"] += 1
            continue
        if int(num(direct, "current_original_size_mismatch")) or int(num(direct, "proposed_original_size_mismatch")):
            exclusion_counts["image_size_mismatch"] += 1
            continue
        if num(direct, "outside_gt12_pct_pad8", 999.0) > MAX_OUTSIDE_GT12_PCT_PAD8:
            exclusion_counts["outside_anchor_difference_over_threshold"] += 1
            continue
        current = row.get("current_candidate", "").strip()
        proposed = row.get("recommended_candidate", "").strip()
        if not current or not proposed:
            exclusion_counts["missing_candidate_name"] += 1
            continue
        accepted[key] = row

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = SWEEP_DIR / f"shadow_selection_manifest_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)

    manifest: list[dict[str, object]] = []
    overrides: list[dict[str, object]] = []
    missing_candidate_files: list[dict[str, object]] = []
    for winner in winner_rows:
        image = winner.get("image", "").strip()
        key = image_key(image)
        raw_current = winner.get("selected_distance_px", "").strip()
        current = canonical_candidate(raw_current)
        proposal = accepted.get(key)
        proposed_raw = proposal.get("recommended_candidate", current).strip() if proposal else current
        proposed = canonical_candidate(proposed_raw)
        changed = bool(proposal and proposed and proposed != current)
        direct = direct_by_image.get(key, {})
        stability = stability_by_image.get(key, {})
        decision = decision_by_image.get(key, {})
        expected_candidates = {current, proposed}
        for candidate in expected_candidates:
            if candidate:
                canonical = canonical_candidate(candidate)
                expected_path = CANDIDATE_DIR / f"{Path(image).stem}__{canonical.lower()}.png"
                resolved_path = audit.candidate_path(image, canonical)
                if not resolved_path.is_file():
                    missing_candidate_files.append({
                        "image": image,
                        "candidate": candidate,
                        "expected_path": str(expected_path),
                        "resolved_path": str(resolved_path),
                    })

        enriched: dict[str, object] = dict(winner)
        enriched.update({
            "shadow_current_candidate": current,
            "shadow_candidate": proposed,
            "shadow_override": int(changed),
            "shadow_eligibility_reason": (
                "stable_18_of_24+residual_reduction_ge_90pct+outside_deltaE12_pad8_le_0.25pct"
                if proposal else (
                    "retain_current_distance_candidate_not_evaluated_by_color_rule"
                    if raw_current.isdigit() or decision.get("decision_reason", "") == "current_candidate_not_audited"
                    else "retain_current_candidate"
                )
            ),
            "source_color_gt16_before": proposal.get("current_source_color_pixels_gt16", "") if proposal else "",
            "source_color_gt16_after": proposal.get("recommended_source_color_pixels_gt16", "") if proposal else "",
            "source_color_reduction_pct": proposal.get("source_color_gt16_reduction_pct", "") if proposal else "",
            "objective_improvement_counterfactual": proposal.get("objective_improvement", "") if proposal else "",
            "stability_scenarios_selecting_alternative": stability.get("dominant_alternative_scenarios", "") if proposal else "",
            "direct_outside_gt12_pct_pad8": direct.get("outside_gt12_pct_pad8", "") if proposal else "",
        })
        manifest.append(enriched)
        if changed:
            overrides.append({
                "image": image,
                "current_candidate": current,
                "shadow_candidate": proposed,
                "source_color_gt16_before": proposal.get("current_source_color_pixels_gt16", ""),
                "source_color_gt16_after": proposal.get("recommended_source_color_pixels_gt16", ""),
                "source_color_reduction_pct": proposal.get("source_color_gt16_reduction_pct", ""),
                "objective_improvement_counterfactual": proposal.get("objective_improvement", ""),
                "stable_alternative_scenarios": stability.get("dominant_alternative_scenarios", ""),
                "outside_gt12_pct_pad8": direct.get("outside_gt12_pct_pad8", ""),
            })

    write_csv(out_dir / "shadow_candidate_manifest.csv", manifest)
    write_csv(out_dir / "shadow_overrides.csv", overrides)
    write_csv(out_dir / "missing_candidate_files.csv", missing_candidate_files)

    transitions = Counter(f"{row['current_candidate']} -> {row['shadow_candidate']}" for row in overrides)
    lines = [
        "COLOR-RULE SHADOW SELECTION MANIFEST (READ-ONLY)",
        f"Timestamp: {stamp}",
        f"Winner rows: {len(winner_rows)}",
        f"Accepted shadow overrides: {len(overrides)}",
        f"Retained current candidates: {len(manifest) - len(overrides)}",
        f"Eligible proposals after guards: {len(accepted)}",
        f"Candidate files missing: {len(missing_candidate_files)}",
        f"Numeric selected_distance_px winner rows: {unassessed_distance_rows}",
        f"Rows marked current_candidate_not_audited by source full-population audit: {current_not_audited_rows}",
        "Coverage caveat: numeric distance winners are path-normalized to DISTANCE_XXPX and retained unchanged; the source color-rule audit did not score these current distance candidates, so this manifest makes no color-rule override recommendation for them.",
        f"Stability gate: alternative selected in at least {MIN_STABLE_SCENARIOS}/{TOTAL_SCENARIOS} sensitivity combinations",
        f"Residual reduction gate: >= {MIN_REDUCTION_PCT:.0f}%",
        f"Outside-anchor gate: direct current-vs-proposed Delta E > 12 on <= {MAX_OUTSIDE_GT12_PCT_PAD8:.2f}% of outside pixels after 8px anchor buffer",
        "No candidate image copied or altered; no production scoring/report changed.",
        "This is a sidecar/shadow manifest only, not a production winner report.",
        "",
        "OVERRIDE TRANSITIONS:",
    ]
    lines.extend(f"{transition}: {count}" for transition, count in transitions.most_common())
    lines.extend(["", "EXCLUSIONS / WARNINGS:"])
    lines.extend(f"{reason}: {count}" for reason, count in exclusion_counts.most_common())
    if not exclusion_counts:
        lines.append("none")
    lines.extend([
        "",
        f"Manifest: {out_dir / 'shadow_candidate_manifest.csv'}",
        f"Overrides: {out_dir / 'shadow_overrides.csv'}",
        f"Missing candidate-file report: {out_dir / 'missing_candidate_files.csv'}",
    ])
    (out_dir / "shadow_manifest_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if missing_candidate_files:
        raise SystemExit("FEHLER: Mindestens eine referenzierte Kandidatendatei fehlt; Details in missing_candidate_files.csv")


if __name__ == "__main__":
    main()
