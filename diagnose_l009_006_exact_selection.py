#!/usr/bin/env python3
"""Exakter Auswahl-Replay für l009_006; verändert keine Produktionsdateien."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from yolo_mask_auto_select_modules.config import (
    CONTRAST_MASK_THRESHOLDS,
    DEFAULT_ANCHORS_CSV,
    DEFAULT_IMAGES_DIR,
    DEFAULT_WEIGHTS,
)
from yolo_mask_auto_select_modules.io_utils import (
    discover_sources,
    make_box_only_mask,
    read_anchor_csv,
    read_box_mask,
    rgb_array,
    write_filtered_anchors,
)
from yolo_mask_auto_select_modules.scoring import (
    ink_strength_map,
    make_contrast_mask,
    make_smart_ink_mask,
    outline_leak_metrics,
    score_candidate,
    select_candidate,
)

PROJECT = Path(__file__).resolve().parent
IMAGE_STEM = "l009_006"

RUN = (
    PROJECT
    / "yolo_mask_winners"
    / "full_recheck_after_alpha_island_025_20261025_022053"
)
DEBUG = RUN / "debug_candidates"
CLEANED = DEBUG / "cleaned_images"
SCORES_CSV = DEBUG / "candidate_scores.csv"
WINNER_CSV = RUN / "winner_report.csv"
WINNER_MASK = RUN / "masks" / f"{IMAGE_STEM}.png"

DISTANCES = [0, 2, 4, 6, 8, 10]
CANDIDATE_IDS = [
    *DISTANCES,
    "HYBRID_D2",
    "HYBRID_D6",
    *[
        f"CONTRAST_T{int(round(t * 100)):02d}"
        for t in CONTRAST_MASK_THRESHOLDS
    ],
    "SMART_INK",
    "BOX_ONLY",
]

IGNORE_METRICS = {
    # Diese drei Felder stammen aus dem damaligen Alpha-Postprocessing,
    # nicht aus score_candidate() oder der Auswahlfunktion.
    "alpha_small_object_removal_applied",
    "alpha_small_components_removed",
    "alpha_small_pixels_removed",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Datei fehlt: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("Keine Datensaetze.\n", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def full_precision(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return ""
    return format(number, ".17g")


def rounded_like_production(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.6f}" if math.isfinite(number) else ""


def load_mask(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Maskenkandidat fehlt: {path}")
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8) > 0


def load_rgba(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Gespeichertes Kandidatenbild fehlt: {path}")
    with Image.open(path) as im:
        rgba = np.asarray(im.convert("RGBA"), dtype=np.uint8)
    return rgba[..., :3].copy(), rgba[..., 3].copy()


def candidate_token(cid: int | str) -> str:
    if cid == "SMART_INK":
        return "smart_ink"
    if cid == "BOX_ONLY":
        return "box_only"
    if isinstance(cid, str):
        return cid.lower()
    return f"distance_{cid:02d}px"


def get_image_name(anchors_by_image: dict[str, list[dict]]) -> str:
    hits = [
        name for name in anchors_by_image
        if Path(name).stem.casefold() == IMAGE_STEM.casefold()
    ]
    if len(hits) != 1:
        raise RuntimeError(
            f"Erwartet genau einen Anchor-Dateinamen für {IMAGE_STEM}; "
            f"gefunden: {hits}"
        )
    return hits[0]


def main() -> int:
    print("=" * 72)
    print("EXAKTER AUSWAHL-REPLAY: l009_006")
    print("Nur Diagnose; keine Produktionsdateien werden geändert.")
    print("=" * 72)

    for path in (PROJECT / "yolo_mask_routing_audit.py",
                 SCORES_CSV, WINNER_CSV, CLEANED, DEFAULT_WEIGHTS):
        if not path.exists():
            raise FileNotFoundError(f"Benötigte Datei/Pfad fehlt: {path}")

    anchors_path = DEFAULT_ANCHORS_CSV.expanduser().resolve()
    images_dir = DEFAULT_IMAGES_DIR.expanduser().resolve()
    weights_path = DEFAULT_WEIGHTS.expanduser().resolve()

    fieldnames, anchor_rows, anchors_by_image = read_anchor_csv(anchors_path)
    image_name = get_image_name(anchors_by_image)
    source_paths = discover_sources(images_dir, {image_name}, PROJECT)
    source_path = source_paths[image_name]

    print(f"Bild:       {image_name}")
    print(f"Original:   {source_path}")
    print(f"Anchors:    {anchors_path}")
    print(f"YOLO:       {weights_path}")
    print("Gespeicherte LaMa-Kandidaten werden wiederverwendet.")

    # Vorhandene Originaldaten und gespeicherte Kandidaten einlesen.
    original_rgb, original_alpha = rgb_array(source_path)
    image_h, image_w = original_alpha.shape
    anchors = anchors_by_image[image_name]
    body_boxes = [
        b for b in anchors
        if str(b.get("type", "")).casefold() == "body"
    ]
    body_box_mask = (
        make_box_only_mask(image_w, image_h, body_boxes) > 0
    )
    source_detail = ink_strength_map(original_rgb)

    archived_score_rows = [
        row for row in read_csv(SCORES_CSV)
        if Path(row.get("image", "")).stem.casefold()
        == IMAGE_STEM.casefold()
    ]
    archived_by_candidate = {
        row.get("distance_px", "").strip(): row
        for row in archived_score_rows
    }
    if len(archived_by_candidate) != len(CANDIDATE_IDS):
        raise RuntimeError(
            f"Erwartet {len(CANDIDATE_IDS)} archivierte Kandidatenscores, "
            f"gefunden: {len(archived_by_candidate)}. "
            "Prüfe candidate_scores.csv."
        )

    winner_report_rows = [
        row for row in read_csv(WINNER_CSV)
        if Path(row.get("image", "")).stem.casefold()
        == IMAGE_STEM.casefold()
    ]
    if len(winner_report_rows) != 1:
        raise RuntimeError(
            f"Erwartet genau eine Winner-Report-Zeile, gefunden: "
            f"{len(winner_report_rows)}"
        )
    archived_winner_report = winner_report_rows[0]
    archived_winner = archived_winner_report.get(
        "selected_distance_px", ""
    ).strip()

    # Nur ein Bild an den vorhandenen Routing-Audit übergeben.
    # Damit werden die sechs YOLO-Distanzmasken neu erzeugt, nicht LaMa.
    with tempfile.TemporaryDirectory(
        prefix="l009_006_exact_selection_"
    ) as temp_string:
        temp = Path(temp_string)
        temp_anchors = temp / "anchors_l009_006.csv"
        audit_parent = temp / "routing_audit"

        write_filtered_anchors(
            temp_anchors,
            fieldnames,
            anchor_rows,
            {image_name},
        )

        command = [
            sys.executable,
            str(PROJECT / "yolo_mask_routing_audit.py"),
            "--images-dir", str(images_dir),
            "--anchors", str(temp_anchors),
            "--weights", str(weights_path),
            "--output", str(audit_parent),
            "--imgsz", "1280",
            "--min-conf", "0.025",
            "--dilation", "2",
            "--distances", "0,2,4,6,8,10",
            "--device", "auto",
        ]

        print("\n[1/4] YOLO-Masken für ein Bild rekonstruieren ...")
        print("Befehl:", " ".join(command))
        proc = subprocess.run(command, cwd=PROJECT, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Routing-Audit fehlgeschlagen, Exit-Code {proc.returncode}"
            )

        reports = list(audit_parent.rglob("routing_report.csv"))
        if not reports:
            raise FileNotFoundError(
                f"routing_report.csv fehlt unter {audit_parent}"
            )
        audit_out = max(reports, key=lambda p: p.stat().st_mtime).parent

        print("\n[2/4] Kandidatenmasken exakt aus den Maskenquellen aufbauen ...")

        box_mask = load_mask(
            audit_out / "box_only" / "masks" / f"{IMAGE_STEM}.png"
        )
        masks: dict[int, np.ndarray] = {}
        for distance in DISTANCES:
            masks[distance] = load_mask(
                audit_out
                / "conf_005"
                / f"distance_{distance:02d}"
                / "masks"
                / f"{IMAGE_STEM}.png"
            )

        smart_mask = make_smart_ink_mask(original_rgb, anchors)
        contrast_masks = {
            f"CONTRAST_T{int(round(t * 100)):02d}": make_contrast_mask(
                original_rgb, original_alpha, anchors, float(t)
            )
            for t in CONTRAST_MASK_THRESHOLDS
        }

        hybrid_masks = {
            "HYBRID_D2": box_mask | masks[2],
            "HYBRID_D6": box_mask | masks[6],
        }

        candidate_masks: dict[int | str, np.ndarray] = {
            **masks,
            **hybrid_masks,
            **contrast_masks,
            "SMART_INK": smart_mask,
            "BOX_ONLY": box_mask,
        }

        yolo_reference_mask = masks[max(DISTANCES)]
        scoring_envelope = yolo_reference_mask | box_mask | smart_mask

        scored = []
        precision_rows = []
        comparison_rows = []
        metric_match_count = 0
        metric_compare_count = 0

        print("\n[3/4] Gespeicherte Kandidaten erneut bewerten ...")

        for cid in CANDIDATE_IDS:
            mask = candidate_masks[cid]
            if mask.shape != original_alpha.shape:
                raise RuntimeError(
                    f"Maskenform stimmt nicht für {cid}: {mask.shape}; "
                    f"Original: {original_alpha.shape}"
                )

            token = candidate_token(cid)
            cleaned_path = CLEANED / f"{IMAGE_STEM}__{token}.png"
            final_rgb, final_alpha = load_rgba(cleaned_path)

            if final_rgb.shape[:2] != original_alpha.shape:
                raise RuntimeError(
                    f"Bildform stimmt nicht für {cid}: {final_rgb.shape[:2]}"
                )

            leak = outline_leak_metrics(
                source_detail,
                original_alpha,
                body_box_mask,
                mask,
            )
            metrics = score_candidate(
                original_rgb,
                original_alpha,
                final_rgb,
                final_alpha,
                mask,
                scoring_envelope,
                box_mask,
                body_box_mask,
                yolo_reference_mask,
                outline_leak_precomputed=leak,
            )

            # Wie in der Pipeline: erster Tuple-Wert ist der Score vor
            # select_candidate(); die Auswahl verändert danach metrics.
            scored.append((metrics["score"], cid, mask, metrics))

            precision_row = {
                "candidate": str(cid),
                "stored_winner": str(cid) == archived_winner,
            }
            for key, value in metrics.items():
                precision_row[key] = full_precision(value)
            precision_rows.append(precision_row)

        # Wichtig: echte Produktionsauswahlfunktion und Reihenfolge verwenden.
        winner, best_text_score, tolerance = select_candidate(scored)
        _initial_score, replay_winner, replay_mask, replay_metrics = winner

        # Nach select_candidate() vergleichen: die Funktion schreibt unter
        # anderem eligibility und finalen Auswahl-Score in die Metriken.
        for _tuple_score, cid, _mask, metrics in scored:
            cid_key = str(cid)
            archived = archived_by_candidate.get(cid_key)
            if archived is None:
                raise RuntimeError(
                    f"Archivierter Kandidatenscore fehlt: {cid_key}"
                )

            for key, value in metrics.items():
                if key not in archived or key in IGNORE_METRICS:
                    continue

                actual = rounded_like_production(value)
                expected = archived.get(key, "").strip()
                matches = actual == expected

                metric_compare_count += 1
                metric_match_count += int(matches)
                comparison_rows.append({
                    "candidate": cid_key,
                    "metric": key,
                    "archived_6dp": expected,
                    "replayed_6dp": actual,
                    "match": matches,
                    "replayed_full_precision": full_precision(value),
                })

        # Gewinner-Maske mit der tatsächlich archivierten Maske vergleichen.
        stored_mask = load_mask(WINNER_MASK)
        winner_mask_exact = (
            stored_mask.shape == replay_mask.shape
            and np.array_equal(stored_mask, replay_mask)
        )
        changed_pixels = (
            int(np.count_nonzero(stored_mask != replay_mask))
            if stored_mask.shape == replay_mask.shape
            else -1
        )

        # Exakten Rang aus den bereits von select_candidate() veränderten
        # Metriken ausgeben; Sortierschlüssel wie in scoring.py.
        def tie_break(candidate_id):
            if isinstance(candidate_id, int):
                return (0, candidate_id)
            if candidate_id == "SMART_INK":
                return (1, 0)
            return (2, 9999)

        ranked = sorted(
            scored,
            key=lambda item: (
                item[3]["score"],
                item[3]["structure_score"],
                item[3]["text_quality_score"],
                tie_break(item[1]),
            ),
        )

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT / "diagnostics" / (
            f"l009_006_exact_selection_replay_{stamp}"
        )
        output_dir.mkdir(parents=True, exist_ok=False)

        # Volle Präzision der nachgerechneten Metriken speichern.
        write_csv(output_dir / "scores_full_precision.csv", precision_rows)
        write_csv(
            output_dir / "archived_vs_replayed_metrics.csv",
            comparison_rows,
        )

        ranking_rows = []
        for rank, item in enumerate(ranked, start=1):
            _score, cid, mask, metrics = item
            ranking_rows.append({
                "rank": rank,
                "candidate": str(cid),
                "is_replay_winner": cid == replay_winner,
                "is_archived_winner": str(cid) == archived_winner,
                "selector_score_full_precision": full_precision(
                    metrics["score"]
                ),
                "structure_score_full_precision": full_precision(
                    metrics["structure_score"]
                ),
                "text_quality_score_full_precision": full_precision(
                    metrics["text_quality_score"]
                ),
                "anchor_detail_eligible": metrics.get(
                    "anchor_detail_eligible", ""
                ),
                "text_quality_eligible": metrics.get(
                    "text_quality_eligible", ""
                ),
                "residual_micro_component_area": metrics.get(
                    "residual_micro_component_area", ""
                ),
                "mask_pixels": int(np.count_nonzero(mask)),
            })
        write_csv(output_dir / "replay_ranking.csv", ranking_rows)

        mismatch_rows = [
            row for row in comparison_rows if row["match"] != "True"
        ]
        # CSV DictWriter kann boolesche Werte als "True"/"False" schreiben;
        # dieser Vergleich deckt beide Schreibweisen ab.
        mismatch_rows = [
            row for row in comparison_rows
            if str(row["match"]).casefold() != "true"
        ]

        summary = {
            "image": image_name,
            "archived_winner": archived_winner,
            "replayed_winner": str(replay_winner),
            "winner_changed": archived_winner != str(replay_winner),
            "stored_winner_mask_pixel_exact": winner_mask_exact,
            "stored_winner_mask_different_pixels": changed_pixels,
            "best_text_score_full_precision": full_precision(best_text_score),
            "text_tolerance_full_precision": full_precision(tolerance),
            "metric_fields_compared": metric_compare_count,
            "metric_fields_matching_at_6dp": metric_match_count,
            "metric_fields_different_at_6dp": len(mismatch_rows),
            "score_comparison_passed": (
                metric_compare_count > 0 and not mismatch_rows
            ),
            "production_files_modified": False,
            "lama_rerun": False,
            "yolo_rerun_for_this_image": True,
            "output_directory": str(output_dir),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        with (output_dir / "summary.txt").open(
            "w", encoding="utf-8"
        ) as f:
            f.write("EXAKTER AUSWAHL-REPLAY: l009_006\n")
            f.write("=" * 55 + "\n")
            f.write(f"Archivierter Gewinner: {archived_winner}\n")
            f.write(f"Replay-Gewinner:       {replay_winner}\n")
            f.write(f"Gewinner geändert:     {summary['winner_changed']}\n")
            f.write(
                "Gewinner-Maske pixelgenau identisch: "
                f"{winner_mask_exact}\n"
            )
            f.write(f"Unterschiedliche Maskenpixel: {changed_pixels}\n")
            f.write(
                f"Metrikfelder geprüft: {metric_compare_count}\n"
            )
            f.write(
                f"Übereinstimmung bei 6 Dezimalstellen: "
                f"{metric_match_count}\n"
            )
            f.write(
                f"Abweichungen: {len(mismatch_rows)}\n"
            )
            f.write(
                f"Scorevergleich bestanden: "
                f"{summary['score_comparison_passed']}\n"
            )
            f.write("LaMa erneut ausgeführt: Nein\n")
            f.write("Produktionsdateien geändert: Nein\n")

            if mismatch_rows:
                f.write("\nErste Abweichungen:\n")
                for row in mismatch_rows[:30]:
                    f.write(
                        f"  {row['candidate']} / {row['metric']}: "
                        f"Archiv={row['archived_6dp']!r}, "
                        f"Replay={row['replayed_6dp']!r}\n"
                    )

        print("\n" + "=" * 72)
        print("ERGEBNIS")
        print("=" * 72)
        print("Archivierter Gewinner:", archived_winner)
        print("Replay-Gewinner:      ", replay_winner)
        print("Gewinner geändert:    ", summary["winner_changed"])
        print("Gewinner-Maske pixelgenau identisch:", winner_mask_exact)
        print("Unterschiedliche Maskenpixel:", changed_pixels)
        print(
            f"Metrikfelder passend bei 6 Dezimalstellen: "
            f"{metric_match_count}/{metric_compare_count}"
        )
        print("Scorevergleich bestanden:", summary["score_comparison_passed"])
        print("\nRangfolge nach dem Replay:")
        for row in ranking_rows:
            print(
                f"  {row['rank']:2}. {row['candidate']:<14} "
                f"score={row['selector_score_full_precision']} "
                f"text={row['text_quality_score_full_precision']} "
                f"structure={row['structure_score_full_precision']}"
            )
        print("\nDiagnoseordner:", output_dir)
        print("Darin: summary.txt, summary.json, replay_ranking.csv,")
        print("       scores_full_precision.csv, archived_vs_replayed_metrics.csv")

        if not winner_mask_exact:
            print(
                "\nHINWEIS: Die rekonstruierte Gewinner-Maske stimmt nicht "
                "pixelgenau mit dem Archiv überein. Der Replay ist dann "
                "noch kein exakter Nachweis für das damalige Ergebnis."
            )
        if mismatch_rows:
            print(
                "\nHINWEIS: Scores weichen vom Archiv ab. Details stehen in "
                "archived_vs_replayed_metrics.csv; Ursache nicht ungeprüft "
                "der Rundung zuschreiben."
            )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(f"\nFEHLER: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
