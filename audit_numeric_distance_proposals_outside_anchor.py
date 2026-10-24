#!/usr/bin/env python3
"""Read-only follow-up audit for numeric DISTANCE winner proposals.

1. Recomputes the same 24 sensitivity combinations used by
   audit_color_rule_sensitivity.py, but for the 42 proposals originating from
   the numeric DISTANCE audit.
2. Sends only proposals passing BOTH existing gates (same alternative in >=18/24
   scenarios and >=90% source-color residual reduction) through the existing
   direct CURRENT-vs-PROPOSED outside-anchor Delta-E audit.

CPU/read-only. No GPU inference. Does not alter scoring.py, winner_report.csv,
source images, or candidate images. All new files go into a uniquely timestamped
folder under the numeric distance audit directory.
"""
from __future__ import annotations

import csv
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import audit_anchor_color_rule_holdout as audit
import audit_color_rule_sensitivity as sensitivity
import audit_current_vs_proposed_outside_anchor as outside_audit

AUDIT_DIR = audit.DEBUG_DIR / "counterfactual_color_rule_full_population_20261024_214800"
SENSITIVITY_DIR = AUDIT_DIR / "sensitivity_sweep_20261024_220012"
MIN_REDUCTION_PCT = 90.0
MIN_STABLE_SCENARIOS = 18
WEIGHTS = (0.12, 0.18, 0.24, 0.30)
REDUCTION_LIMITS = (0.50, 0.65, 0.75)
CHROMA_P90_LIMITS = (1.00, 1.05)
CANDIDATE_ORDER = ("BOX_ONLY", "SMART_INK", "CONTRAST_T38", "HYBRID_D2", "HYBRID_D6")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Benötigte Datei fehlt: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    if not fields:
        fields = ["image"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def norm_stem(name: str) -> str:
    return Path((name or "").strip()).stem.lower()


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def build_candidate_rows(
    image: str,
    current_candidate: str,
    current_row: dict[str, str],
    named_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Combine the newly measured DISTANCE row with existing named-candidate metrics."""
    current = dict(current_row)
    current["image"] = image
    current["candidate"] = current_candidate
    current["current_candidate"] = current_candidate
    result = [current]
    seen = {current_candidate}
    for row in named_rows:
        candidate = row.get("candidate", "").strip().upper()
        if candidate not in CANDIDATE_ORDER or candidate in seen:
            continue
        item = dict(row)
        item["current_candidate"] = current_candidate
        item["candidate"] = candidate
        result.append(item)
        seen.add(candidate)
    return result


def main() -> None:
    if not AUDIT_DIR.is_dir() or not SENSITIVITY_DIR.is_dir():
        raise FileNotFoundError(
            "Audit-Verzeichnis nicht gefunden. Bitte im Projektordner ~/Downloads/Cleanup starten."
        )

    # The three uploaded runs had identical proposal CSVs; choose the newest folder.
    distance_dirs = sorted(
        p for p in AUDIT_DIR.glob("numeric_distance_winner_audit_*")
        if (p / "numeric_distance_changed_review.csv").is_file()
        and (p / "numeric_distance_current_metrics.csv").is_file()
        and (p / "numeric_distance_decisions.csv").is_file()
    )
    if not distance_dirs:
        raise FileNotFoundError("Kein vollständiger numeric_distance_winner_audit_* Ordner gefunden.")
    distance_dir = distance_dirs[-1]

    changed_path = distance_dir / "numeric_distance_changed_review.csv"
    current_path = distance_dir / "numeric_distance_current_metrics.csv"
    decisions_path = distance_dir / "numeric_distance_decisions.csv"
    named_metrics_path = AUDIT_DIR / "full_population_candidate_metrics.csv"
    for path in (changed_path, current_path, decisions_path, named_metrics_path,
                 audit.ORIGINAL_DIR, audit.CANDIDATE_DIR):
        if not path.exists():
            raise FileNotFoundError(f"Benötigter vorhandener Pfad fehlt: {path}")

    changed = read_csv(changed_path)
    current_rows = read_csv(current_path)
    decisions = read_csv(decisions_path)
    named_metrics = read_csv(named_metrics_path)

    current_by_stem = {norm_stem(r.get("image", "")): r for r in current_rows}
    decision_by_stem = {norm_stem(r.get("image", "")): r for r in decisions}
    named_by_stem: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in named_metrics:
        if row.get("image", "").strip():
            named_by_stem[norm_stem(row["image"])].append(row)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = distance_dir / f"numeric_distance_outside_anchor_followup_{stamp}"
    run_root.mkdir(parents=True, exist_ok=False)

    stability_rows: list[dict[str, object]] = []
    qualifying: list[dict[str, object]] = []
    all_decisions: list[dict[str, object]] = []
    skipped: Counter[str] = Counter()

    print(f"Numeric-Audit-Ordner: {distance_dir}")
    print(f"Vorschläge für die Sensitivitätsprüfung: {len(changed)}")
    print("Sensitivität: 24 Kombinationen; CSV-only; keine GPU-Inferenz.")

    for proposal in changed:
        image = proposal.get("image", "").strip()
        stem = norm_stem(image)
        current_candidate = proposal.get("current_candidate", "").strip().upper()
        current_row = current_by_stem.get(stem)
        baseline = decision_by_stem.get(stem, {}).get("recommended_candidate", current_candidate).strip().upper()
        if current_row is None:
            skipped["current_metric_row_missing"] += 1
            continue

        candidates = build_candidate_rows(
            image, current_candidate, current_row, named_by_stem.get(stem, [])
        )
        missing = sorted(set(CANDIDATE_ORDER) - {
            r.get("candidate", "").strip().upper() for r in candidates
        })
        if missing:
            skipped["named_candidate_metrics_missing"] += 1
            continue

        selected_by_scenario: Counter[str] = Counter()
        scenario_count = 0
        for weight in WEIGHTS:
            for reduction_limit in REDUCTION_LIMITS:
                for chroma_limit in CHROMA_P90_LIMITS:
                    result = sensitivity.evaluate_image(
                        candidates, weight, reduction_limit, chroma_limit
                    )
                    selected_by_scenario[str(result.get("recommended_candidate", current_candidate))] += 1
                    scenario_count += 1

        alternative_counts = {
            name: count for name, count in selected_by_scenario.items()
            if name != current_candidate
        }
        if alternative_counts:
            dominant, dominant_count = sorted(
                alternative_counts.items(), key=lambda item: (-item[1], item[0])
            )[0]
        else:
            dominant, dominant_count = "", 0

        same_alternative_stable = bool(
            dominant and dominant == baseline and dominant_count >= MIN_STABLE_SCENARIOS
        )
        reduction_pct = as_float(proposal, "source_color_gt16_reduction_pct")
        passes_reduction = reduction_pct >= MIN_REDUCTION_PCT
        passes_both = same_alternative_stable and passes_reduction
        row: dict[str, object] = {
            **proposal,
            "scenarios_tested": scenario_count,
            "baseline_recommended_candidate": baseline,
            "dominant_alternative": dominant,
            "dominant_alternative_scenarios": dominant_count,
            "stable_alternative_18_of_24": int(same_alternative_stable),
            "passes_reduction_ge_90pct": int(passes_reduction),
            "passes_both_prechecks": int(passes_both),
            "scenario_selection_counts": ";".join(
                f"{name}={count}" for name, count in sorted(
                    selected_by_scenario.items(), key=lambda item: (-item[1], item[0])
                )
            ),
        }
        stability_rows.append({
            "image": image,
            "current_candidate": current_candidate,
            "baseline_recommended_candidate": baseline,
            "dominant_alternative": dominant,
            "dominant_alternative_scenarios": dominant_count,
            "scenarios_tested": scenario_count,
            "stable_alternative_18_of_24": int(same_alternative_stable),
        })
        all_decisions.append(row)
        if passes_both:
            qualifying.append(proposal)

    write_csv(run_root / "numeric_distance_sensitivity_stability.csv", stability_rows)
    write_csv(run_root / "numeric_distance_sensitivity_all_proposals.csv", all_decisions)
    write_csv(run_root / "numeric_distance_qualifying_changes_for_outside_anchor.csv", qualifying)

    stable_count = sum(int(r["stable_alternative_18_of_24"]) for r in stability_rows)
    reduction_count = sum(int(r["passes_reduction_ge_90pct"]) for r in all_decisions)
    print(f"Sensitivitätsdatensätze: {len(stability_rows)}")
    print(f"Baseline-Alternative stabil in >=18/24 Szenarien: {stable_count}")
    print(f"Residualreduktion >=90%: {reduction_count}")
    print(f"Beide Vorprüfungen bestanden: {len(qualifying)}")
    print("Ausgewählte Bilder:")
    for row in qualifying:
        print(f"  {row['image']}: {row['current_candidate']} -> {row['recommended_candidate']} "
              f"| Reduktion {row['source_color_gt16_reduction_pct']}%")
    print(f"Neue Follow-up-Dateien: {run_root}")

    # Reuse the existing direct Delta-E audit logic, redirecting every input/output
    # path to this isolated follow-up folder. Existing 121-proposal audit files stay intact.
    outside_audit.AUDIT_DIR = AUDIT_DIR
    outside_audit.SWEEP_DIR = SENSITIVITY_DIR
    outside_audit.OUTSIDE_DIR = run_root
    outside_audit.CHANGED_CSV = run_root / "numeric_distance_qualifying_changes_for_outside_anchor.csv"
    outside_audit.STABILITY_CSV = run_root / "numeric_distance_sensitivity_stability.csv"
    outside_audit.MIN_REDUCTION_PCT = MIN_REDUCTION_PCT
    outside_audit.STABLE_MIN_SCENARIOS = MIN_STABLE_SCENARIOS
    print("\n========== DIREKTER HINTERGRUNDSCHUTZ-TEST ==========")
    outside_audit.main()

    # Create a NEW combined shadow manifest. Only additions passing the exact
    # outside-anchor gate are promoted into this sidecar; the previous manifest
    # remains unchanged and no production report is touched.
    direct_dirs = sorted(run_root.glob("direct_current_proposed_delta_*"))
    if not direct_dirs:
        raise FileNotFoundError("Der direkte Delta-E-Audit hat keinen Ergebnisordner erzeugt.")
    direct_dir = direct_dirs[-1]
    direct_metrics_path = direct_dir / "direct_current_proposed_outside_metrics.csv"
    if not direct_metrics_path.is_file():
        raise FileNotFoundError(f"Direkte Outside-Metriken fehlen: {direct_metrics_path}")
    direct_metrics = read_csv(direct_metrics_path)

    previous_manifest_dirs = sorted(
        p for p in SENSITIVITY_DIR.glob("shadow_selection_manifest_*")
        if (p / "shadow_candidate_manifest.csv").is_file()
        and (p / "shadow_overrides.csv").is_file()
    )
    if not previous_manifest_dirs:
        raise FileNotFoundError("Der bisherige Shadow-Manifest-Stand wurde nicht gefunden.")
    previous_manifest_dir = previous_manifest_dirs[-1]
    manifest_rows = read_csv(previous_manifest_dir / "shadow_candidate_manifest.csv")
    override_rows = read_csv(previous_manifest_dir / "shadow_overrides.csv")
    manifest_by_stem = {norm_stem(r.get("image", "")): r for r in manifest_rows}
    proposal_by_stem = {norm_stem(r.get("image", "")): r for r in changed}
    stability_by_stem = {norm_stem(str(r.get("image", ""))): r for r in stability_rows}

    approved: dict[str, dict[str, str]] = {}
    outside_rejections: Counter[str] = Counter()
    for row in direct_metrics:
        stem = norm_stem(row.get("image", ""))
        status = row.get("direct_outside_difference_status", "")
        try:
            outside_pct = float(row.get("outside_gt12_pct_pad8", ""))
        except (TypeError, ValueError):
            outside_pct = float("inf")
        size_mismatch = (
            row.get("current_original_size_mismatch", "1") != "0"
            or row.get("proposed_original_size_mismatch", "1") != "0"
        )
        if status != "LOW_OUTSIDE_DIFFERENCE" or outside_pct > 0.25 or size_mismatch:
            outside_rejections[status or "missing_status"] += 1
            continue
        proposal = proposal_by_stem.get(stem)
        current_shadow = manifest_by_stem.get(stem)
        if proposal is None or current_shadow is None:
            outside_rejections["not_in_existing_manifest_or_proposal"] += 1
            continue
        if current_shadow.get("shadow_current_candidate", "").upper() != proposal.get("current_candidate", "").upper():
            outside_rejections["current_candidate_mismatch"] += 1
            continue
        required_candidates = (
            proposal.get("current_candidate", ""),
            proposal.get("recommended_candidate", ""),
        )
        missing = False
        for candidate in required_candidates:
            candidate_path = audit.candidate_path(proposal.get("image", ""), candidate)
            if not candidate_path.is_file():
                outside_rejections["candidate_file_missing"] += 1
                missing = True
        if missing:
            continue
        approved[stem] = row

    old_override_count = sum(1 for r in manifest_rows if str(r.get("shadow_override", "0")) == "1")
    new_override_rows: list[dict[str, object]] = []
    for stem, direct in approved.items():
        manifest_row = manifest_by_stem[stem]
        proposal = proposal_by_stem[stem]
        stability = stability_by_stem.get(stem, {})
        shadow_candidate = proposal.get("recommended_candidate", "")
        manifest_row.update({
            "shadow_candidate": shadow_candidate,
            "shadow_override": 1,
            "shadow_eligibility_reason": (
                "numeric_distance_audit+stable_18_of_24+residual_reduction_ge_90pct+"
                "outside_deltaE12_pad8_le_0.25pct"
            ),
            "source_color_gt16_before": proposal.get("current_source_color_pixels_gt16", ""),
            "source_color_gt16_after": proposal.get("recommended_source_color_pixels_gt16", ""),
            "source_color_reduction_pct": proposal.get("source_color_gt16_reduction_pct", ""),
            "objective_improvement_counterfactual": proposal.get("objective_improvement", ""),
            "stability_scenarios_selecting_alternative": stability.get("dominant_alternative_scenarios", ""),
            "direct_outside_gt12_pct_pad8": direct.get("outside_gt12_pct_pad8", ""),
        })
        new_override_rows.append({
            "image": proposal.get("image", ""),
            "current_candidate": proposal.get("current_candidate", ""),
            "shadow_candidate": shadow_candidate,
            "source_color_gt16_before": proposal.get("current_source_color_pixels_gt16", ""),
            "source_color_gt16_after": proposal.get("recommended_source_color_pixels_gt16", ""),
            "source_color_reduction_pct": proposal.get("source_color_gt16_reduction_pct", ""),
            "objective_improvement_counterfactual": proposal.get("objective_improvement", ""),
            "stable_alternative_scenarios": stability.get("dominant_alternative_scenarios", ""),
            "outside_gt12_pct_pad8": direct.get("outside_gt12_pct_pad8", ""),
        })

    all_overrides = [dict(r) for r in override_rows]
    all_overrides.extend(dict(r) for r in new_override_rows)

    missing_candidate_files: list[dict[str, object]] = []
    for row in manifest_rows:
        image_name = row.get("image", "")
        for candidate_field in ("shadow_current_candidate", "shadow_candidate"):
            candidate = row.get(candidate_field, "").strip()
            if not candidate:
                continue
            candidate_path = audit.candidate_path(image_name, candidate)
            if not candidate_path.is_file():
                missing_candidate_files.append({
                    "image": image_name,
                    "candidate": candidate,
                    "resolved_path": str(candidate_path),
                })

    combined_dir = run_root / "combined_shadow_manifest_after_numeric_audit"
    combined_dir.mkdir(parents=True, exist_ok=False)
    write_csv(combined_dir / "shadow_candidate_manifest.csv", manifest_rows)
    write_csv(combined_dir / "shadow_overrides.csv", all_overrides)
    write_csv(combined_dir / "missing_candidate_files.csv", missing_candidate_files)

    total_overrides = sum(1 for r in manifest_rows if str(r.get("shadow_override", "0")) == "1")
    approved_names = [proposal_by_stem[stem].get("image", stem) for stem in approved]
    rejected_names = [
        r.get("image", "") for r in qualifying
        if norm_stem(r.get("image", "")) not in approved
    ]
    summary_lines = [
        "COMBINED SHADOW MANIFEST AFTER NUMERIC DISTANCE AUDIT (READ-ONLY)",
        f"Timestamp: {stamp}",
        f"Base manifest: {previous_manifest_dir}",
        f"Audit folder: {distance_dir}",
        f"Numeric audit proposals: {len(changed)}",
        f"Passed sensitivity >= {MIN_STABLE_SCENARIOS}/24 and residual reduction >= {MIN_REDUCTION_PCT:.0f}%: {len(qualifying)}",
        f"Passed direct outside-anchor guard: {len(approved)}",
        f"Existing accepted shadow overrides preserved: {old_override_count}",
        f"New numeric overrides added: {len(approved)}",
        f"Total shadow overrides: {total_overrides}",
        f"Retained current selections: {len(manifest_rows) - total_overrides}",
        f"Missing candidate files: {len(missing_candidate_files)}",
        f"Outside-guard exclusions/warnings: {dict(outside_rejections)}",
        "No production scoring.py, winner_report.csv, or candidate images modified.",
        "This is a new sidecar shadow manifest; the previous shadow manifest is unchanged.",
        "",
        "NEW NUMERIC OVERRIDES:",
        *(approved_names or ["(none)" ]),
        "",
        "QUALIFYING BUT NOT ADDED:",
        *(rejected_names or ["(none)" ]),
        "",
        f"Manifest: {combined_dir / 'shadow_candidate_manifest.csv'}",
        f"Overrides: {combined_dir / 'shadow_overrides.csv'}",
        f"Missing-candidate report: {combined_dir / 'missing_candidate_files.csv'}",
    ]
    (combined_dir / "shadow_manifest_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n========== KOMBINIERTER SHADOW-STAND ==========")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nFEHLER: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        print("Es wurden nur neue Follow-up-Ausgaben geschrieben; vorhandene Audit-/Produktionsdateien bleiben unangetastet.")
        raise
