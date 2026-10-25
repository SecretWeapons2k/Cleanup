#!/usr/bin/env python3
"""Methodenunabhaengiger RGBA-Schattenvergleich fuer vier Hardcases."""

import csv
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from yolo_mask_auto_select_modules.scoring import ink_strength_map

PROJECT = Path.home() / "Downloads" / "Cleanup"
RUN = PROJECT / "yolo_mask_winners" / (
    "full_recheck_after_alpha_island_025_20261025_022053"
)
DEBUG = RUN / "debug_candidates"
CANDIDATES = DEBUG / "cleaned_images"
SOURCE = (
    Path.home() / "Downloads" / "yabai proj"
    / "yabaiproject - Copy" / "mit_kanji"
)
ANCHORS = (
    Path.home() / "Downloads" / "yabai proj"
    / "processed_output" / "anchors.csv"
)
SCORES = DEBUG / "candidate_scores.csv"
WINNERS = RUN / "winner_report.csv"

TARGETS = ("h003_000", "h008_001", "m001_001", "m002_000")

BACKGROUNDS = {
    "dark": (24, 24, 24),
    "neutral": (128, 128, 128),
    "light": (240, 240, 240),
}

SUFFIXES = {
    "HYBRID_D2": "hybrid_d2",
    "HYBRID_D6": "hybrid_d6",
    "CONTRAST_T38": "contrast_t38",
    "SMART_INK": "smart_ink",
    "BOX_ONLY": "box_only",
}

ALPHA_THRESHOLDS = (16, 64, 128, 240)


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    fields = list(dict.fromkeys(
        key for row in rows for key in row
    )) or ["image"]

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fields, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def load_rgba(path):
    if not path.is_file():
        raise FileNotFoundError(f"Datei fehlt: {path}")

    with Image.open(path) as im:
        return np.asarray(im.convert("RGBA"), dtype=np.uint8).copy()


def composite(rgba, background):
    rgb = rgba[:, :, :3].astype(np.float32)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    bg = np.asarray(background, dtype=np.float32).reshape(1, 1, 3)

    result = rgb * alpha + bg * (1.0 - alpha)
    return np.clip(result, 0, 255).astype(np.uint8)


def candidate_suffix(candidate_id):
    value = str(candidate_id).strip()

    if value.isdigit():
        return f"distance_{int(value):02d}px"

    key = value.upper()
    if key not in SUFFIXES:
        raise ValueError(f"Unbekannter Kandidat: {candidate_id}")

    return SUFFIXES[key]


def parse_boxes(rows, width, height):
    boxes = []

    for row in rows:
        try:
            x1, y1, x2, y2 = (
                int(round(float(row[key])))
                for key in ("x1", "y1", "x2", "y2")
            )
        except (ValueError, TypeError, KeyError):
            continue

        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width - 1, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height - 1, y2))

        if x2 >= x1 and y2 >= y1:
            boxes.append({
                "box": (x1, y1, x2, y2),
                "anchor_id": row.get("anchor_id", ""),
                "row": row.get("row", ""),
                "col": row.get("col", ""),
                "type": row.get("type", ""),
                "text": row.get("text", ""),
            })

    return boxes


def make_box_mask(boxes, shape):
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)

    for item in boxes:
        x1, y1, x2, y2 = item["box"]
        mask[y1:y2 + 1, x1:x2 + 1] = True

    return mask


def detail_ratio(source_detail, candidate_detail, region):
    """Gleiche Positionen vergleichen; niedriger bedeutet weniger Restdetail."""
    src = source_detail[region].astype(np.float64)
    fin = candidate_detail[region].astype(np.float64)

    source_energy = float(src.sum())
    if source_energy <= 1e-9:
        return 0.0, 0.0, 0.0

    overlap = float(np.minimum(src, fin).sum() / source_energy)
    density = float(fin.sum() / source_energy)

    # Gleiche Grundform wie die bestehende Detailbewertung,
    # aber auf dem alpha-korrekt zusammengesetzten Bild.
    retained = 0.90 * np.clip(overlap, 0.0, 1.0)
    retained += 0.10 * np.clip(density, 0.0, 1.0)

    return float(retained), overlap, density


def alpha_metrics(source_rgba, candidate_rgba, box_mask):
    src_a = source_rgba[:, :, 3]
    fin_a = candidate_rgba[:, :, 3]

    result = {
        "source_alpha_mass_inside_boxes": int(src_a[box_mask].sum()),
        "candidate_alpha_mass_inside_boxes": int(fin_a[box_mask].sum()),
        "alpha_changed_pixels_inside_boxes": int(
            np.count_nonzero((src_a != fin_a) & box_mask)
        ),
    }

    denominator = max(int(src_a[box_mask].sum()), 1)
    result["alpha_mass_ratio_inside_boxes"] = round(
        int(fin_a[box_mask].sum()) / denominator, 6
    )

    for threshold in ALPHA_THRESHOLDS:
        result[f"source_alpha_gt_{threshold}_inside_boxes"] = int(
            np.count_nonzero((src_a > threshold) & box_mask)
        )
        result[f"candidate_alpha_gt_{threshold}_inside_boxes"] = int(
            np.count_nonzero((fin_a > threshold) & box_mask)
        )

    return result


def pareto_frontier(rows):
    """Markiert Pareto-Dominanz, ohne Kandidaten automatisch auszuwaehlen."""
    objectives = (
        "rgba_residual_global_worst_background",
        "rgba_residual_worst_anchor_worst_background",
        "structure_score",
        "sprite_texture_distortion_ratio",
    )

    for row in rows:
        a = []
        for field in objectives:
            try:
                a.append(float(row[field]))
            except (ValueError, TypeError, KeyError):
                a.append(float("inf"))
        row["_objectives"] = a

    for row in rows:
        current = row["_objectives"]
        dominators = []

        for other in rows:
            if other is row:
                continue

            candidate = other["_objectives"]
            if all(np.isfinite(v) for v in candidate + current):
                no_worse = all(a <= b + 1e-12
                               for a, b in zip(candidate, current))
                strictly_better = any(a < b - 1e-12
                                      for a, b in zip(candidate, current))
                if no_worse and strictly_better:
                    dominators.append(other["candidate"])

        row["pareto_dominated_by_count"] = len(dominators)
        row["pareto_dominated_by"] = ";".join(dominators)
        row["pareto_frontier"] = int(not dominators)
        del row["_objectives"]


def main():
    for path in (RUN, DEBUG, CANDIDATES, SOURCE, ANCHORS, SCORES, WINNERS):
        if not path.exists():
            raise FileNotFoundError(f"Eingabe fehlt: {path}")

    anchor_rows = read_csv(ANCHORS)
    score_rows = read_csv(SCORES)
    winner_rows = read_csv(WINNERS)

    anchors_by_image = {}
    for name in TARGETS:
        anchors_by_image[name] = [
            row for row in anchor_rows
            if Path(row.get("image", "")).stem.casefold() == name.casefold()
        ]

    current_winner = {
        Path(row["image"]).stem: row.get("selected_distance_px", "")
        for row in winner_rows
        if Path(row.get("image", "")).stem in TARGETS
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = DEBUG / f"rgba_visual_shadow_{stamp}"
    out.mkdir(parents=True, exist_ok=False)

    candidate_output = []
    anchor_output = []
    summary = [
        "RGBA VISUAL SHADOW AUDIT (READ-ONLY)",
        f"Ausgabe: {out}",
        "Keine GPU-Inferenz.",
        "Keine Produktionsdatei wird veraendert.",
        "Alle Methoden erhalten dieselben Messkriterien.",
        "Keine automatische Gewinneraenderung.",
        "",
    ]

    scores_by_image = {}
    for row in score_rows:
        image_name = Path(row.get("image", "")).stem
        if image_name in TARGETS:
            scores_by_image.setdefault(image_name, []).append(row)

    for name in TARGETS:
        source_path = SOURCE / f"{name}.png"
        source = load_rgba(source_path)
        h, w = source.shape[:2]

        boxes = parse_boxes(anchors_by_image[name], w, h)
        if not boxes:
            raise RuntimeError(f"Keine OCR-Boxen fuer {name}")

        box_union = make_box_mask(boxes, (h, w))
        candidate_rows_for_image = []

        # Original-Detail auf denselben festen Hintergruenden berechnen.
        source_details = {}
        for bg_name, bg_color in BACKGROUNDS.items():
            src_comp = composite(source, bg_color)
            source_details[bg_name] = ink_strength_map(src_comp)

        for score_row in scores_by_image.get(name, []):
            cid = score_row.get("distance_px", "")
            suffix = candidate_suffix(cid)
            path = CANDIDATES / f"{name}__{suffix}.png"
            candidate = load_rgba(path)

            if candidate.shape[:2] != source.shape[:2]:
                raise RuntimeError(f"Groesse stimmt nicht: {path}")

            global_by_bg = {}
            anchor_by_bg = {}
            per_anchor_records = []

            for bg_name, bg_color in BACKGROUNDS.items():
                final_comp = composite(candidate, bg_color)
                final_detail = ink_strength_map(final_comp)

                global_value, global_overlap, global_density = detail_ratio(
                    source_details[bg_name], final_detail, box_union
                )
                global_by_bg[bg_name] = global_value

                anchor_values = []

                for index, item in enumerate(boxes, start=1):
                    x1, y1, x2, y2 = item["box"]
                    region = np.zeros((h, w), dtype=bool)
                    region[y1:y2 + 1, x1:x2 + 1] = True

                    residual, overlap, density = detail_ratio(
                        source_details[bg_name], final_detail, region
                    )
                    anchor_values.append(residual)

                    per_anchor_records.append({
                        "image": name,
                        "candidate": cid,
                        "background": bg_name,
                        "anchor_index": index,
                        "anchor_id": item["anchor_id"],
                        "row": item["row"],
                        "col": item["col"],
                        "type": item["type"],
                        "text": item["text"],
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "rgba_retained_detail_ratio": round(residual, 6),
                        "same_location_overlap_ratio": round(overlap, 6),
                        "detail_density_ratio": round(density, 6),
                    })

                anchor_by_bg[bg_name] = (
                    max(anchor_values) if anchor_values else 0.0
                )
                anchor_output.extend(per_anchor_records[-len(boxes):])

            summary_row = {
                "image": name,
                "candidate": cid,
                "candidate_file": str(path),
                "current_winner": int(
                    str(current_winner.get(name, "")).upper()
                    == str(cid).upper()
                    or (
                        str(current_winner.get(name, "")).isdigit()
                        and str(cid).isdigit()
                        and int(current_winner[name]) == int(cid)
                    )
                ),
                "rgba_residual_global_mean_background": round(
                    float(np.mean(list(global_by_bg.values()))), 6
                ),
                "rgba_residual_global_worst_background": round(
                    float(max(global_by_bg.values())), 6
                ),
                "rgba_residual_worst_anchor_mean_background": round(
                    float(np.mean(list(anchor_by_bg.values()))), 6
                ),
                "rgba_residual_worst_anchor_worst_background": round(
                    float(max(anchor_by_bg.values())), 6
                ),
                **alpha_metrics(source, candidate, box_union),
            }

            for bg_name, value in global_by_bg.items():
                summary_row[f"rgba_residual_global_{bg_name}"] = round(
                    float(value), 6
                )

            for bg_name, value in anchor_by_bg.items():
                summary_row[f"rgba_residual_worst_anchor_{bg_name}"] = round(
                    float(value), 6
                )

            for key in (
                "score", "structure_score", "text_quality_score",
                "best_text_quality_score", "residual_inside_box_ratio",
                "residual_micro_component_area",
                "residual_text_large_component_area",
                "max_text_anchor_retained_detail_ratio",
                "sprite_contour_displacement_ratio",
                "sprite_texture_distortion_ratio",
                "boundary_color_artifact_ratio",
                "text_quality_eligible", "anchor_detail_eligible",
            ):
                summary_row[f"existing_{key}"] = score_row.get(key, "")

            candidate_rows_for_image.append(summary_row)

        pareto_frontier(candidate_rows_for_image)
        candidate_output.extend(candidate_rows_for_image)

        current = current_winner.get(name, "UNKNOWN")
        winner_metric = next(
            (
                r for r in candidate_rows_for_image
                if str(r["candidate"]).upper() == str(current).upper()
                or (
                    str(r["candidate"]).isdigit()
                    and str(current).isdigit()
                    and int(r["candidate"]) == int(current)
                )
            ),
            None,
        )

        frontier_names = [
            str(row["candidate"])
            for row in candidate_rows_for_image
            if row["pareto_frontier"]
        ]

        summary += [
            f"===== {name} =====",
            f"Bisheriger Gewinner: {current}",
            f"Kandidaten ausgewertet: {len(candidate_rows_for_image)}",
            "Pareto-Front (keine einzelne Methode wird bevorzugt): "
            + (", ".join(frontier_names) if frontier_names else "keine"),
        ]

        if winner_metric:
            summary += [
                "Bisheriger Gewinner - alpha-korrigierter globaler "
                "Restdetailwert (Mittel der drei Hintergruende): "
                f"{winner_metric['rgba_residual_global_mean_background']}",
                "Bisheriger Gewinner - schlimmster OCR-Anker "
                "(schlechtester Hintergrund): "
                f"{winner_metric['rgba_residual_worst_anchor_worst_background']}",
                "Bisheriger Gewinner - Alpha-Massenverhaeltnis in OCR-Boxen: "
                f"{winner_metric['alpha_mass_ratio_inside_boxes']}",
            ]

        summary.append("")

    write_csv(out / "rgba_candidate_shadow_metrics.csv", candidate_output)
    write_csv(out / "rgba_per_anchor_shadow_metrics.csv", anchor_output)
    (out / "summary.txt").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )

    print("\n".join(summary))
    print("\n========== DATEIEN ==========")
    print("Ordner:", out)
    print("Kandidatenmetriken:", out / "rgba_candidate_shadow_metrics.csv")
    print("Ankermetriken:", out / "rgba_per_anchor_shadow_metrics.csv")
    print("Zusammenfassung:", out / "summary.txt")
    print("\nNur Diagnoseausgaben; keine Produktionsdatei veraendert.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nFEHLER: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        print("Keine Produktionsdatei wurde veraendert.")
