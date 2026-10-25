#!/usr/bin/env python3
"""Read-only shadow audit der Kandidatenauswahl für den vorhandenen Produktionslauf."""

import csv
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT = Path.home() / "Downloads/Cleanup"
RUN = (
    PROJECT
    / "yolo_mask_winners"
    / "full_recheck_after_alpha_island_025_20261025_022053"
)

CANDIDATES_CSV = RUN / "debug_candidates" / "candidate_scores.csv"
WINNERS_CSV = RUN / "winner_report.csv"

ABSOLUTE_ANCHOR_LIMIT = 0.50
TEXT_SCORE_BOUNDARY_SLACK = 0.0001
TOLERANCE_FLOOR = 0.02
TOLERANCE_CEILING = 0.035
TOLERANCE_RANGE_FRACTION = 0.05


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f"Datei fehlt: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def number(row, key, default=float("nan")):
    try:
        value = float(row.get(key, ""))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError, OverflowError):
        return default


def candidate_id(row):
    return str(row.get("distance_px", "")).strip()


def label(value):
    value = str(value or "").strip()
    try:
        n = float(value)
        if math.isfinite(n) and n.is_integer():
            return f"DISTANCE_{int(n):02d}PX"
    except (TypeError, ValueError, OverflowError):
        pass
    return value or "UNKNOWN"


def tie_key(row):
    cid = candidate_id(row)
    try:
        n = float(cid)
        if math.isfinite(n) and n.is_integer():
            return (0, int(n))
    except (TypeError, ValueError, OverflowError):
        pass

    if cid == "SMART_INK":
        return (1, 0)

    return (2, 9999)


def anchor_ratio(row):
    value = number(row, "max_text_anchor_retained_detail_ratio")
    if math.isfinite(value):
        return value

    value = number(row, "max_anchor_retained_detail_ratio", 0.0)
    return value if math.isfinite(value) else 0.0


def choose(rows, eligible):
    options = [row for row, ok in zip(rows, eligible) if ok]
    if not options:
        return "NO_ACCEPTABLE_CANDIDATE"

    winner = min(
        options,
        key=lambda row: (
            number(row, "structure_score", 1e9),
            tie_key(row),
        ),
    )
    return label(candidate_id(winner))


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = list(dict.fromkeys(
        key for row in rows for key in row.keys()
    ))

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    candidate_rows = read_csv(CANDIDATES_CSV)
    winner_rows = read_csv(WINNERS_CSV)

    required = {
        "image",
        "distance_px",
        "text_quality_score",
        "structure_score",
    }
    if not candidate_rows or not required.issubset(candidate_rows[0]):
        raise ValueError(
            "Unerwartetes CSV-Schema. Vorhandene Spalten: "
            + ", ".join(candidate_rows[0] if candidate_rows else [])
        )

    groups = defaultdict(list)
    for row in candidate_rows:
        groups[Path(row["image"]).name].append(row)

    winner_report = {
        Path(row["image"]).name: row
        for row in winner_rows
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RUN / "debug_candidates" / f"selection_rule_shadow_{stamp}"
    out.mkdir(parents=True, exist_ok=False)

    summaries = []
    details = []

    for image, rows in sorted(groups.items()):
        rows = list(rows)

        text_scores = [
            number(row, "text_quality_score", 1.0)
            for row in rows
        ]
        best_text = min(text_scores)
        spread = max(text_scores) - best_text

        computed_tolerance = min(
            TOLERANCE_CEILING,
            max(TOLERANCE_FLOOR, TOLERANCE_RANGE_FRACTION * spread),
        )
        stored_tolerance = number(
            rows[0], "text_quality_tolerance", computed_tolerance
        )
        tolerance = (
            stored_tolerance
            if math.isfinite(stored_tolerance)
            else computed_tolerance
        )

        text_ok = [
            score <= best_text + tolerance
            + TEXT_SCORE_BOUNDARY_SLACK + 1e-9
            for score in text_scores
        ]
        ratios = [anchor_ratio(row) for row in rows]

        has_clean_candidate = any(
            ok and ratio <= ABSOLUTE_ANCHOR_LIMIT + 1e-9
            for ok, ratio in zip(text_ok, ratios)
        )

        if has_clean_candidate:
            anchor_limit = ABSOLUTE_ANCHOR_LIMIT
        else:
            reference = [
                ratio for ratio, ok in zip(ratios, text_ok) if ok
            ]
            if not reference:
                reference = ratios
            anchor_limit = min(1.0, min(reference) + 0.02)

        joint_exists = any(
            ok and ratio <= anchor_limit + 1e-9
            for ok, ratio in zip(text_ok, ratios)
        )
        fallback_mode = not joint_exists

        # Bisherige Auswahlregeln nachbilden, jedoch ohne Box-Only-Rescue.
        base_eligible = [
            ratio <= anchor_limit + 1e-9
            and (ok or fallback_mode)
            for ratio, ok in zip(ratios, text_ok)
        ]
        no_rescue_winner = choose(rows, base_eligible)

        # Strenger Schatten-Gate: kein automatischer Fallback über 0.50.
        strict_eligible = [
            ok and ratio <= ABSOLUTE_ANCHOR_LIMIT + 1e-9
            for ok, ratio in zip(text_ok, ratios)
        ]
        strict_winner = choose(rows, strict_eligible)

        report = winner_report.get(image, {})
        reported_winner = label(report.get("selected_distance_px", ""))

        box = next(
            (row for row in rows if candidate_id(row) == "BOX_ONLY"),
            None,
        )
        nonbox = [
            row for row in rows if candidate_id(row) != "BOX_ONLY"
        ]
        numeric = []
        for row in nonbox:
            try:
                float(candidate_id(row))
                numeric.append(row)
            except (TypeError, ValueError, OverflowError):
                pass

        best_nonbox_text = (
            min(nonbox, key=lambda r: number(r, "text_quality_score", 1.0))
            if nonbox else None
        )
        best_numeric_text = (
            min(numeric, key=lambda r: number(r, "text_quality_score", 1.0))
            if numeric else None
        )

        box_rescue = any(
            number(row, "box_only_rescue_applied", 0.0) >= 0.5
            for row in rows
        )

        box_text = number(box, "text_quality_score") if box else float("nan")
        box_structure = number(box, "structure_score") if box else float("nan")

        # Signed difference: positive means Box-Only has the better text score.
        all_nonbox_minus_box = (
            number(best_nonbox_text, "text_quality_score") - box_text
            if best_nonbox_text and box else float("nan")
        )
        numeric_advantage = (
            max(
                0.0,
                number(best_numeric_text, "text_quality_score") - box_text,
            )
            if best_numeric_text and box else float("nan")
        )

        better_eligible_nonbox = [
            row for row, ok in zip(rows, base_eligible)
            if ok
            and candidate_id(row) != "BOX_ONLY"
            and box is not None
            and number(row, "structure_score", 1e9) < box_structure - 1e-9
        ]

        rescue_overrode_better = (
            box_rescue
            and reported_winner == "BOX_ONLY"
            and bool(better_eligible_nonbox)
        )

        summaries.append({
            "image": image,
            "reported_winner": reported_winner,
            "candidate_count": len(rows),
            "best_text_quality_score": f"{best_text:.6f}",
            "text_quality_tolerance": f"{tolerance:.6f}",
            "recomputed_anchor_limit": f"{anchor_limit:.6f}",
            "absolute_clean_candidate_exists": int(has_clean_candidate),
            "base_winner_without_box_rescue": no_rescue_winner,
            "no_rescue_differs_from_report": int(
                no_rescue_winner != reported_winner
            ),
            "strict_shadow_winner": strict_winner,
            "strict_candidate_count": sum(strict_eligible),
            "strict_gate_no_acceptable_candidate": int(
                not any(strict_eligible)
            ),
            "box_only_rescue_applied": int(box_rescue),
            "box_text_advantage_vs_numeric_distances": (
                f"{numeric_advantage:.6f}"
                if math.isfinite(numeric_advantage) else ""
            ),
            "best_nonbox_candidate_by_text": (
                label(candidate_id(best_nonbox_text))
                if best_nonbox_text else ""
            ),
            "best_nonbox_minus_box_text_score": (
                f"{all_nonbox_minus_box:.6f}"
                if math.isfinite(all_nonbox_minus_box) else ""
            ),
            "rescue_overrode_better_eligible_nonbox": int(
                rescue_overrode_better
            ),
        })

        for row, ok, ratio, base_ok, strict_ok in zip(
            rows, text_ok, ratios, base_eligible, strict_eligible
        ):
            details.append({
                "image": image,
                "candidate": label(candidate_id(row)),
                "text_quality_score": row.get("text_quality_score", ""),
                "structure_score": row.get("structure_score", ""),
                "current_score_after_selection": row.get("score", ""),
                "max_text_anchor_retained_detail_ratio": f"{ratio:.6f}",
                "raw_text_score_passes_tolerance": int(ok),
                "recomputed_base_eligible_without_rescue": int(base_ok),
                "strict_absolute_gate_eligible": int(strict_ok),
                "stored_anchor_eligibility_limit": row.get(
                    "anchor_detail_eligibility_limit", ""
                ),
                "stored_text_quality_eligible_after_rescue": row.get(
                    "text_quality_eligible", ""
                ),
                "box_only_rescue_applied": row.get(
                    "box_only_rescue_applied", ""
                ),
                "is_reported_winner": int(
                    label(candidate_id(row)) == reported_winner
                ),
            })

    write_csv(out / "image_shadow_summary.csv", summaries)
    write_csv(out / "candidate_shadow_details.csv", details)

    rescue_count = sum(
        int(row["box_only_rescue_applied"]) for row in summaries
    )
    override_count = sum(
        int(row["rescue_overrode_better_eligible_nonbox"])
        for row in summaries
    )
    strict_none_count = sum(
        int(row["strict_gate_no_acceptable_candidate"])
        for row in summaries
    )
    changed_no_rescue = sum(
        int(row["no_rescue_differs_from_report"])
        for row in summaries
    )

    summary_lines = [
        "SELECTION-RULE SHADOW AUDIT (READ-ONLY)",
        f"Candidate-CSV: {CANDIDATES_CSV}",
        f"Winner-Report: {WINNERS_CSV}",
        f"Bilder: {len(summaries)}",
        f"Kandidatenzeilen: {len(details)}",
        f"Bilder mit Box-Only-Rescue: {rescue_count}",
        f"Rescue-Fälle mit besserem eligible Nicht-Box-Kandidaten: {override_count}",
        f"Bilder ohne Kandidat am strengen absoluten Gate: {strict_none_count}",
        f"Winner-Wechsel ohne Box-Only-Rescue: {changed_no_rescue}",
        "Keine Produktionsdateien wurden verändert.",
    ]
    (out / "summary.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )

    print("\n".join(summary_lines))
    print("\n===== ZIELFÄLLE =====")
    for target in ("h008_001.png", "m001_001.png"):
        summary = next(
            (row for row in summaries if row["image"] == target), None
        )
        if summary is None:
            print(f"{target}: nicht im Kandidaten-CSV gefunden")
            continue

        print(
            f"\n{target}: gemeldet={summary['reported_winner']}; "
            f"ohne Rescue={summary['base_winner_without_box_rescue']}; "
            f"strenges Gate={summary['strict_shadow_winner']}; "
            f"Ankerlimit={summary['recomputed_anchor_limit']}"
        )

        for row in details:
            if row["image"] == target:
                print(
                    f"  {row['candidate']:16s} "
                    f"text={float(row['text_quality_score']):.6f} "
                    f"struct={float(row['structure_score']):.6f} "
                    f"anchor={row['max_text_anchor_retained_detail_ratio']} "
                    f"text_ok={row['raw_text_score_passes_tolerance']} "
                    f"base_ok={row['recomputed_base_eligible_without_rescue']} "
                    f"strict_ok={row['strict_absolute_gate_eligible']}"
                )

    print(f"\nAusgabe: {out}")
    print("FERTIG – Produktionsdateien unverändert.")


if __name__ == "__main__":
    main()
