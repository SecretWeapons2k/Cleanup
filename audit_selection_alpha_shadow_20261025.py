#!/usr/bin/env python3
"""Read-only Auswahl- und Alpha-Schattenaudit für den bestehenden Cleanup-Lauf."""

import csv
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT = Path.home() / "Downloads/Cleanup"
RUN = (
    PROJECT / "yolo_mask_winners"
    / "full_recheck_after_alpha_island_025_20261025_022053"
)

CANDIDATE_CSV = RUN / "debug_candidates" / "candidate_scores.csv"
WINNER_CSV = RUN / "winner_report.csv"
SOURCE_IMAGE = (
    Path.home() / "Downloads/yabai proj"
    / "yabaiproject - Copy" / "mit_kanji" / "m001_001.png"
)
CLEANED_DIR = RUN / "debug_candidates" / "cleaned_images"
WINNER_MASK = RUN / "masks" / "m001_001.png"

ABSOLUTE_ANCHOR_LIMIT = 0.50
BOUNDARY_SLACK = 0.0001
TOLERANCE_FLOOR = 0.02
TOLERANCE_CEILING = 0.035
TOLERANCE_FRACTION = 0.05

ALPHA_THRESHOLDS = (1, 8, 16, 32, 64, 128)

ALPHA_BANDS = (
    ("alpha_0", 0, 0),
    ("alpha_1_8", 1, 8),
    ("alpha_9_16", 9, 16),
    ("alpha_17_32", 17, 32),
    ("alpha_33_64", 33, 64),
    ("alpha_65_128", 65, 128),
    ("alpha_129_255", 129, 255),
)


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f"Datei nicht gefunden: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def number(row, key, default=float("nan")):
    try:
        value = float(row.get(key, ""))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError, OverflowError):
        return default


def candidate_label(value):
    value = str(value or "").strip()

    try:
        numeric = float(value)
        if math.isfinite(numeric) and numeric.is_integer():
            return f"DISTANCE_{int(numeric):02d}PX"
    except (TypeError, ValueError, OverflowError):
        pass

    return value.upper() or "UNKNOWN"


def tie_key(row):
    value = str(row.get("distance_px", "")).strip()

    try:
        numeric = float(value)
        if math.isfinite(numeric) and numeric.is_integer():
            return (0, int(numeric))
    except (TypeError, ValueError, OverflowError):
        pass

    if value.upper() == "SMART_INK":
        return (1, 0)

    return (2, 9999)


def anchor_ratio(row):
    value = number(row, "max_text_anchor_retained_detail_ratio")
    if math.isfinite(value):
        return value

    value = number(row, "max_anchor_retained_detail_ratio", 0.0)
    return value if math.isfinite(value) else 0.0


def select_best(rows, eligible):
    options = [
        row for row, is_eligible in zip(rows, eligible)
        if is_eligible
    ]

    if not options:
        return "NO_ACCEPTABLE_CANDIDATE"

    winner = min(
        options,
        key=lambda row: (
            number(row, "structure_score", 1e9),
            tie_key(row),
        ),
    )
    return candidate_label(winner.get("distance_px"))


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


def load_rgba(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGBA"), dtype=np.uint8).copy()


def run_selection_shadow(out):
    rows = read_csv(CANDIDATE_CSV)
    reports = read_csv(WINNER_CSV)

    required = {
        "image", "distance_px", "text_quality_score", "structure_score",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(
            "Unerwartetes Schema in candidate_scores.csv. "
            f"Vorhandene Spalten: {list(rows[0]) if rows else []}"
        )

    grouped = defaultdict(list)
    for row in rows:
        grouped[Path(row["image"]).name].append(row)

    report_by_image = {
        Path(row["image"]).name: row for row in reports
    }

    image_results = []
    candidate_results = []

    for image_name, candidates in sorted(grouped.items()):
        report = report_by_image.get(image_name, {})

        scores = [
            number(row, "text_quality_score", 1.0)
            for row in candidates
        ]
        best_text = min(scores)
        spread = max(scores) - best_text

        calculated_tolerance = min(
            TOLERANCE_CEILING,
            max(TOLERANCE_FLOOR, TOLERANCE_FRACTION * spread),
        )
        stored_tolerance = number(
            candidates[0], "text_quality_tolerance",
            calculated_tolerance,
        )
        tolerance = (
            stored_tolerance
            if math.isfinite(stored_tolerance)
            else calculated_tolerance
        )

        text_ok = [
            score <= best_text + tolerance + BOUNDARY_SLACK + 1e-9
            for score in scores
        ]
        ratios = [anchor_ratio(row) for row in candidates]

        absolute_clean_exists = any(
            ok and ratio <= ABSOLUTE_ANCHOR_LIMIT + 1e-9
            for ok, ratio in zip(text_ok, ratios)
        )

        if absolute_clean_exists:
            adaptive_limit = ABSOLUTE_ANCHOR_LIMIT
        else:
            eligible_ratios = [
                ratio for ratio, ok in zip(ratios, text_ok) if ok
            ]
            reference = eligible_ratios if eligible_ratios else ratios
            adaptive_limit = min(1.0, min(reference) + 0.02)

        adaptive_eligible = [
            ok and ratio <= adaptive_limit + 1e-9
            for ok, ratio in zip(text_ok, ratios)
        ]
        fallback_mode = not any(adaptive_eligible)

        base_eligible = [
            ratio <= adaptive_limit + 1e-9
            and (ok or fallback_mode)
            for ok, ratio in zip(ratios, text_ok)
        ]

        # Diese Grenze ist eine strenge Diagnosevariante, keine Produktionsregel.
        strict_eligible = [
            ok and ratio <= ABSOLUTE_ANCHOR_LIMIT + 1e-9
            for ok, ratio in zip(text_ok, ratios)
        ]

        reported = candidate_label(report.get("selected_distance_px"))
        without_rescue = select_best(candidates, base_eligible)
        strict_winner = select_best(candidates, strict_eligible)

        box = next(
            (row for row in candidates
             if str(row.get("distance_px", "")).strip().upper() == "BOX_ONLY"),
            None,
        )

        box_structure = (
            number(box, "structure_score", float("inf"))
            if box else float("inf")
        )

        better_nonbox = [
            row for row, ok in zip(candidates, base_eligible)
            if ok
            and str(row.get("distance_px", "")).strip().upper() != "BOX_ONLY"
            and number(row, "structure_score", 1e9) < box_structure - 1e-9
        ]

        rescue_applied = (
            number(report, "box_only_rescue_applied", 0.0) >= 0.5
        )
        rescue_overrode_better = (
            rescue_applied
            and reported == "BOX_ONLY"
            and bool(better_nonbox)
        )

        image_results.append({
            "image": image_name,
            "reported_winner": reported,
            "candidate_count": len(candidates),
            "best_text_quality_score": f"{best_text:.6f}",
            "text_quality_tolerance": f"{tolerance:.6f}",
            "adaptive_anchor_limit": f"{adaptive_limit:.6f}",
            "absolute_clean_candidate_exists": int(absolute_clean_exists),
            "candidate_count_passing_strict_gate": sum(strict_eligible),
            "base_winner_without_box_rescue": without_rescue,
            "no_rescue_differs_from_report": int(without_rescue != reported),
            "strict_shadow_winner": strict_winner,
            "strict_gate_has_no_acceptable_candidate": int(
                not any(strict_eligible)
            ),
            "box_only_rescue_applied": int(rescue_applied),
            "rescue_overrode_better_eligible_nonbox": int(
                rescue_overrode_better
            ),
        })

        for row, ok, ratio, base_ok, strict_ok in zip(
            candidates, text_ok, ratios, base_eligible, strict_eligible
        ):
            candidate_results.append({
                "image": image_name,
                "candidate": candidate_label(row.get("distance_px")),
                "text_quality_score": row.get("text_quality_score", ""),
                "structure_score": row.get("structure_score", ""),
                "stored_selection_score": row.get("score", ""),
                "max_text_anchor_retained_detail_ratio": f"{ratio:.6f}",
                "text_score_passes_tolerance": int(ok),
                "adaptive_anchor_limit": f"{adaptive_limit:.6f}",
                "base_eligible_without_rescue": int(base_ok),
                "strict_absolute_gate_eligible": int(strict_ok),
                "is_reported_winner": int(
                    candidate_label(row.get("distance_px")) == reported
                ),
            })

    write_csv(out / "image_shadow_summary.csv", image_results)
    write_csv(out / "candidate_shadow_details.csv", candidate_results)

    return image_results, candidate_results


def run_alpha_sweep(out):
    # Bestehende Produktionsfunktion nur lesend für einen einheitlichen
    # Detailindikator verwenden; Scorewerte werden dabei nicht verändert.
    from yolo_mask_auto_select_modules.scoring import ink_strength_map

    source = load_rgba(SOURCE_IMAGE)
    mask = np.asarray(
        Image.open(WINNER_MASK).convert("L"), dtype=np.uint8
    ) > 0

    if source.shape[:2] != mask.shape:
        raise ValueError(
            f"Original-/Maskengröße unterscheiden sich: "
            f"{source.shape[:2]} vs. {mask.shape}"
        )

    files = sorted(CLEANED_DIR.glob("m001_001__*.png"))
    if not files:
        raise FileNotFoundError(
            f"Keine Kandidatenbilder gefunden: {CLEANED_DIR}/m001_001__*.png"
        )

    outputs = [("SOURCE", source)]
    for path in files:
        outputs.append((path.stem.replace("m001_001__", "").upper(),
                        load_rgba(path)))

    source_alpha = source[:, :, 3]
    zones = {
        "whole_image": np.ones(mask.shape, dtype=bool),
        "winner_mask": mask,
        "outside_winner_mask": ~mask,
        "source_visible_inside_mask": mask & (source_alpha > 16),
        "source_alpha_le16_inside_mask": mask & (source_alpha <= 16),
    }

    threshold_rows = []
    band_rows = []
    overview_rows = []

    for name, rgba in outputs:
        if rgba.shape != source.shape:
            raise ValueError(
                f"Bildgröße stimmt nicht bei {name}: "
                f"{rgba.shape} vs {source.shape}"
            )

        rgb = rgba[:, :, :3]
        alpha = rgba[:, :, 3]
        detail = ink_strength_map(rgb)
        high_detail_020 = detail >= 0.20
        high_detail_028 = detail >= 0.28

        alpha_changed = alpha != source_alpha
        rgb_changed = np.any(rgb != source[:, :, :3], axis=2)

        overview_rows.append({
            "candidate": name,
            "alpha_changed_pixels_inside_mask": int(
                np.count_nonzero(alpha_changed & mask)
            ),
            "alpha_changed_pixels_outside_mask": int(
                np.count_nonzero(alpha_changed & ~mask)
            ),
            "rgb_changed_pixels_inside_mask": int(
                np.count_nonzero(rgb_changed & mask)
            ),
            "rgb_changed_pixels_outside_mask": int(
                np.count_nonzero(rgb_changed & ~mask)
            ),
            "alpha_1_16_detail_ge020_inside_mask": int(
                np.count_nonzero(
                    mask & (alpha >= 1) & (alpha <= 16) & high_detail_020
                )
            ),
            "alpha_1_16_detail_ge028_inside_mask": int(
                np.count_nonzero(
                    mask & (alpha >= 1) & (alpha <= 16) & high_detail_028
                )
            ),
            "alpha_1_16_pixels_inside_mask": int(
                np.count_nonzero(mask & (alpha >= 1) & (alpha <= 16))
            ),
        })

        for threshold in ALPHA_THRESHOLDS:
            visible = alpha > threshold

            for zone_name, zone in zones.items():
                selected = zone & visible
                threshold_rows.append({
                    "candidate": name,
                    "alpha_threshold": threshold,
                    "region": zone_name,
                    "region_pixels": int(np.count_nonzero(zone)),
                    "pixels_alpha_gt_threshold": int(
                        np.count_nonzero(selected)
                    ),
                    "detail_ge020_and_alpha_gt_threshold": int(
                        np.count_nonzero(selected & high_detail_020)
                    ),
                    "detail_ge028_and_alpha_gt_threshold": int(
                        np.count_nonzero(selected & high_detail_028)
                    ),
                })

        for band_name, low, high in ALPHA_BANDS:
            in_band = (alpha >= low) & (alpha <= high)

            for zone_name, zone in zones.items():
                selected = zone & in_band
                values = detail[selected]

                band_rows.append({
                    "candidate": name,
                    "alpha_band": band_name,
                    "alpha_min": low,
                    "alpha_max": high,
                    "region": zone_name,
                    "pixels_in_band": int(np.count_nonzero(selected)),
                    "detail_ge020_pixels": int(
                        np.count_nonzero(selected & high_detail_020)
                    ),
                    "detail_ge028_pixels": int(
                        np.count_nonzero(selected & high_detail_028)
                    ),
                    "mean_ink_strength": (
                        f"{float(np.mean(values)):.6f}"
                        if values.size else ""
                    ),
                })

    write_csv(out / "m001_001_alpha_threshold_sweep.csv", threshold_rows)
    write_csv(out / "m001_001_alpha_band_detail.csv", band_rows)
    write_csv(out / "m001_001_alpha_overview.csv", overview_rows)

    return overview_rows


def main():
    for path in (CANDIDATE_CSV, WINNER_CSV, SOURCE_IMAGE, WINNER_MASK):
        if not path.is_file():
            raise FileNotFoundError(f"Benötigte Datei fehlt: {path}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = RUN / "debug_candidates" / f"selection_alpha_shadow_{stamp}"
    out.mkdir(parents=True, exist_ok=False)

    image_results, candidate_results = run_selection_shadow(out)
    alpha_results = run_alpha_sweep(out)

    rescue_count = sum(
        row["box_only_rescue_applied"] for row in image_results
    )
    override_count = sum(
        row["rescue_overrode_better_eligible_nonbox"]
        for row in image_results
    )
    no_rescue_switches = sum(
        row["no_rescue_differs_from_report"] for row in image_results
    )
    strict_none = sum(
        row["strict_gate_has_no_acceptable_candidate"]
        for row in image_results
    )
    adaptive_one = sum(
        float(row["adaptive_anchor_limit"]) >= 0.999999
        for row in image_results
    )

    summary = [
        "SELECTION + ALPHA SHADOW AUDIT (READ-ONLY)",
        f"Kandidaten-CSV: {CANDIDATE_CSV}",
        f"Gewinnerbericht: {WINNER_CSV}",
        f"Bilder im Auswahlaudit: {len(image_results)}",
        f"Kandidatenzeilen: {len(candidate_results)}",
        f"Bilder mit Box-Only-Rescue: {rescue_count}",
        f"Rescue-Fälle mit besserem eligible Nicht-Box-Kandidaten: {override_count}",
        f"Bilder, deren Gewinner ohne Rescue abweicht: {no_rescue_switches}",
        f"Bilder ohne Kandidat am strengen absoluten Gate: {strict_none}",
        f"Bilder mit adaptiver Ankergrenze >= 1.0: {adaptive_one}",
        f"Alpha-Kandidaten einschließlich Original: {len(alpha_results)}",
        "",
        "Alpha-Dateien:",
        "  m001_001_alpha_threshold_sweep.csv",
        "  m001_001_alpha_band_detail.csv",
        "  m001_001_alpha_overview.csv",
        "",
        "Auswahldateien:",
        "  image_shadow_summary.csv",
        "  candidate_shadow_details.csv",
        "",
        "Keine Produktionsdateien wurden verändert.",
    ]
    (out / "summary.txt").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )

    print("\n".join(summary))

    print("\n===== AUSWAHL: h008_001 und m001_001 =====")
    for target in ("h008_001.png", "m001_001.png"):
        row = next(
            (item for item in image_results if item["image"] == target),
            None,
        )
        if row:
            print(
                f"{target}: report={row['reported_winner']}; "
                f"ohne Rescue={row['base_winner_without_box_rescue']}; "
                f"striktes Gate={row['strict_shadow_winner']}; "
                f"Ankergrenze={row['adaptive_anchor_limit']}; "
                f"strikt zulässig={row['candidate_count_passing_strict_gate']}"
            )

    print("\n===== ALPHA: m001_001 =====")
    for row in alpha_results:
        print(
            f"{row['candidate']:20s} "
            f"alpha-geändert innen={row['alpha_changed_pixels_inside_mask']:6d} "
            f"außen={row['alpha_changed_pixels_outside_mask']:4d} "
            f"Detail alpha 1-16 (>=0.20)="
            f"{row['alpha_1_16_detail_ge020_inside_mask']:5d}"
        )

    print(f"\nAusgabeordner:\n{out}")
    print("FERTIG – Produktionsdateien unverändert.")


if __name__ == "__main__":
    main()
