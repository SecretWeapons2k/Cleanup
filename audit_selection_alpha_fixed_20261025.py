#!/usr/bin/env python3
"""Read-only: echte Auswahlfunktion ohne Rescue plus Alpha-Komposit-Diagnose."""

import csv
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import yolo_mask_auto_select_modules.scoring as scoring

PROJECT = Path.home() / "Downloads/Cleanup"
RUN = (
    PROJECT / "yolo_mask_winners"
    / "full_recheck_after_alpha_island_025_20261025_022053"
)
CANDIDATE_CSV = RUN / "debug_candidates" / "candidate_scores.csv"
WINNER_CSV = RUN / "winner_report.csv"
CLEANED = RUN / "debug_candidates" / "cleaned_images"
SOURCE = (
    Path.home() / "Downloads/yabai proj"
    / "yabaiproject - Copy" / "mit_kanji" / "m001_001.png"
)
MASK = RUN / "masks" / "m001_001.png"

ALPHA_THRESHOLDS = (16, 64, 128)
DETAIL_THRESHOLDS = (0.10, 0.20, 0.28)
BACKGROUNDS = {
    "WHITE": (255, 255, 255),
    "MID_GRAY": (128, 128, 128),
    "BLACK": (0, 0, 0),
}


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


def candidate_id(raw):
    raw = str(raw or "").strip()
    try:
        x = float(raw)
        if math.isfinite(x) and x.is_integer():
            return int(x)
    except (TypeError, ValueError, OverflowError):
        pass
    return raw.upper()


def label(raw):
    value = candidate_id(raw)
    if isinstance(value, int):
        return f"DISTANCE_{value:02d}PX"
    return value or "UNKNOWN"


def metric_dict(row):
    result = {}
    for key, value in row.items():
        if key in {"image", "distance_px"}:
            result[key] = value
            continue
        if value is None or not str(value).strip():
            result[key] = 0.0
            continue
        try:
            x = float(value)
            result[key] = x if math.isfinite(x) else 0.0
        except (ValueError, TypeError, OverflowError):
            result[key] = value
    return result


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_selection_audit(out):
    raw_rows = read_csv(CANDIDATE_CSV)
    report_rows = read_csv(WINNER_CSV)

    required = {"image", "distance_px", "text_quality_score", "structure_score"}
    if not raw_rows or not required.issubset(raw_rows[0]):
        raise ValueError("Unerwartetes CSV-Schema in candidate_scores.csv")

    grouped = defaultdict(list)
    for row in raw_rows:
        grouped[Path(row["image"]).name].append(row)

    reports = {Path(r["image"]).name: r for r in report_rows}
    image_results = []
    candidate_results = []

    old_rescue = scoring.BOX_ONLY_RESCUE_ENABLED
    scoring.BOX_ONLY_RESCUE_ENABLED = False

    try:
        for image_name, rows in sorted(grouped.items()):
            scored = []
            metrics_by_id = {}

            for row in rows:
                cid = candidate_id(row.get("distance_px"))
                metrics = metric_dict(row)
                metrics_by_id[cid] = metrics
                scored.append((
                    number(metrics, "score", 0.0),
                    cid,
                    np.zeros((1, 1), dtype=np.uint8),
                    metrics,
                ))

            # Verwendet die echte Auswahlfunktion. Nur Rescue ist
            # temporär deaktiviert; Produktionsdateien bleiben unverändert.
            chosen, best_text, tolerance = scoring.select_candidate(scored)
            no_rescue_winner = label(chosen[1])

            report = reports.get(image_name, {})
            reported_winner = label(report.get("selected_distance_px"))
            rescue_applied = number(report, "box_only_rescue_applied", 0.0) >= 0.5

            ratios = [
                number(m, "text_quality_score", 1.0)
                for m in metrics_by_id.values()
            ]
            best_score = min(ratios) if ratios else float("nan")
            spread = max(ratios) - best_score if ratios else 0.0
            strict_tolerance = min(
                scoring.TEXT_SCORE_TOLERANCE_CEILING,
                max(
                    scoring.TEXT_SCORE_TOLERANCE_FLOOR,
                    scoring.TEXT_SCORE_TOLERANCE_RANGE_FRACTION * spread,
                ),
            )

            strict_candidates = []
            for cid, metrics in metrics_by_id.items():
                text_ok = (
                    number(metrics, "text_quality_score", 1.0)
                    <= best_score + strict_tolerance
                    + scoring.TEXT_SCORE_BOUNDARY_SLACK + 1e-9
                )
                anchor = number(
                    metrics,
                    "max_text_anchor_retained_detail_ratio",
                    number(metrics, "max_anchor_retained_detail_ratio", 0.0),
                )
                if (
                    text_ok
                    and anchor <= scoring.MAX_ANCHOR_RETAINED_DETAIL_FOR_ELIGIBILITY + 1e-9
                ):
                    strict_candidates.append(cid)

            current_without_rescue_disagrees = (
                not rescue_applied and no_rescue_winner != reported_winner
            )
            rescue_changed_winner = (
                rescue_applied and no_rescue_winner != reported_winner
            )

            image_results.append({
                "image": image_name,
                "reported_winner": reported_winner,
                "winner_with_real_selector_rescue_disabled": no_rescue_winner,
                "reported_rescue_applied": int(rescue_applied),
                "rescue_would_change_winner": int(rescue_changed_winner),
                "unexpected_mismatch_when_rescue_was_not_applied": int(
                    current_without_rescue_disagrees
                ),
                "candidate_count": len(rows),
                "strict_absolute_gate_candidate_count": len(strict_candidates),
                "strict_gate_no_acceptable_candidate": int(not strict_candidates),
                "best_text_quality_score": f"{best_score:.6f}",
                "strict_text_tolerance": f"{strict_tolerance:.6f}",
            })

            for _score, cid, _mask, metrics in scored:
                candidate_results.append({
                    "image": image_name,
                    "candidate": label(cid),
                    "reported_winner": reported_winner,
                    "real_selector_without_rescue_winner": no_rescue_winner,
                    "text_quality_score": metrics.get("text_quality_score", ""),
                    "structure_score": metrics.get("structure_score", ""),
                    "selector_score_without_rescue": metrics.get("score", ""),
                    "anchor_ratio": metrics.get(
                        "max_text_anchor_retained_detail_ratio", ""
                    ),
                    "anchor_limit_recomputed_by_selector": metrics.get(
                        "anchor_detail_eligibility_limit", ""
                    ),
                    "eligible_without_rescue_in_real_selector": metrics.get(
                        "text_quality_eligible", ""
                    ),
                    "reported_winner_this_candidate": int(
                        label(cid) == reported_winner
                    ),
                    "no_rescue_winner_this_candidate": int(
                        label(cid) == no_rescue_winner
                    ),
                })
    finally:
        scoring.BOX_ONLY_RESCUE_ENABLED = old_rescue

    write_csv(out / "image_shadow_summary.csv", image_results)
    write_csv(out / "candidate_shadow_details.csv", candidate_results)

    rescue_count = sum(r["reported_rescue_applied"] for r in image_results)
    switched = sum(r["rescue_would_change_winner"] for r in image_results)
    unexpected = sum(
        r["unexpected_mismatch_when_rescue_was_not_applied"]
        for r in image_results
    )
    strict_none = sum(r["strict_gate_no_acceptable_candidate"] for r in image_results)

    # Eine Korrektheitsprüfung: Wenn ausschließlich die Rescue geändert
    # wurde, können außerhalb ihrer aktivierten Fälle keine Gewinner wechseln.
    invariant_ok = switched <= rescue_count and unexpected == 0

    summary = [
        "FIXED SELECTOR + ALPHA COMPOSITE AUDIT (READ-ONLY)",
        f"Bilder: {len(image_results)}",
        f"Kandidatenzeilen: {len(candidate_results)}",
        f"Bilder mit aktiver Rescue im ursprünglichen Bericht: {rescue_count}",
        f"Gewinner, die sich bei deaktivierter Rescue ändern: {switched}",
        f"Ungeklärte Unterschiede bei ursprünglich inaktiver Rescue: {unexpected}",
        f"Bilder ohne Kandidat am strengen absoluten Gate: {strict_none}",
        f"Auswahlvergleich bestanden: {invariant_ok}",
        "Produktion unverändert.",
    ]
    (out / "selection_summary.txt").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )
    print("\n".join(summary))

    print("\n===== ZIELFÄLLE =====")
    for target in ("h008_001.png", "m001_001.png"):
        row = next((r for r in image_results if r["image"] == target), None)
        if row:
            print(
                f"{target}: aktuell={row['reported_winner']}; "
                f"ohne Rescue={row['winner_with_real_selector_rescue_disabled']}; "
                f"strikt zulässige Kandidaten="
                f"{row['strict_absolute_gate_candidate_count']}; "
                f"Rescue aktiv={row['reported_rescue_applied']}"
            )

    return invariant_ok


def load_rgba(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGBA"), dtype=np.uint8).copy()


def run_alpha_composite_audit(out):
    from yolo_mask_auto_select_modules.scoring import ink_strength_map

    source = load_rgba(SOURCE)
    mask = np.asarray(Image.open(MASK).convert("L"), dtype=np.uint8) > 0

    if source.shape[:2] != mask.shape:
        raise ValueError("Originalbild und Gewinner-Maske haben verschiedene Größen.")

    files = sorted(CLEANED.glob("m001_001__*.png"))
    if not files:
        raise FileNotFoundError(f"Keine Kandidaten in {CLEANED}")

    images = [("SOURCE", source)]
    images.extend(
        (p.stem.replace("m001_001__", "").upper(), load_rgba(p))
        for p in files
    )

    regions = {
        "winner_mask": mask,
        "source_visible_inside_mask": mask & (source[:, :, 3] > 16),
        "outside_winner_mask": ~mask,
    }

    composite_rows = []
    boundary_rows = []

    for name, rgba in images:
        if rgba.shape != source.shape:
            raise ValueError(f"Falsche Größe bei Kandidat {name}: {rgba.shape}")

        alpha = rgba[:, :, 3]
        rgb = rgba[:, :, :3].astype(np.float32)
        a = alpha[:, :, None].astype(np.float32) / 255.0

        # Lokalen RGB-Kontrast nach tatsächlicher Alpha-Komposition messen.
        for bg_name, bg_rgb in BACKGROUNDS.items():
            bg = np.asarray(bg_rgb, dtype=np.float32)[None, None, :]
            composite = np.clip(
                np.rint(rgb * a + bg * (1.0 - a)), 0, 255
            ).astype(np.uint8)
            detail = ink_strength_map(composite)

            for region_name, region in regions.items():
                values = detail[region]
                for threshold in DETAIL_THRESHOLDS:
                    composite_rows.append({
                        "candidate": name,
                        "background": bg_name,
                        "region": region_name,
                        "detail_threshold": threshold,
                        "pixels_above_detail_threshold": int(
                            np.count_nonzero(values >= threshold)
                        ),
                        "region_pixels": int(values.size),
                        "mean_detail_strength": (
                            f"{float(values.mean()):.6f}" if values.size else ""
                        ),
                    })

        # Alpha-Konturen getrennt erfassen; das ist keine OCR-Erkennung.
        for threshold in ALPHA_THRESHOLDS:
            visible = (alpha > threshold).astype(np.uint8)
            eroded = cv2.erode(
                visible, np.ones((3, 3), dtype=np.uint8), iterations=1
            )
            boundary = (visible > 0) & (eroded == 0)

            boundary_rows.append({
                "candidate": name,
                "alpha_threshold": threshold,
                "alpha_edge_pixels_inside_winner_mask": int(
                    np.count_nonzero(boundary & mask)
                ),
                "alpha_edge_pixels_outside_winner_mask": int(
                    np.count_nonzero(boundary & ~mask)
                ),
                "visible_pixels_inside_winner_mask": int(
                    np.count_nonzero((alpha > threshold) & mask)
                ),
                "visible_pixels_outside_winner_mask": int(
                    np.count_nonzero((alpha > threshold) & ~mask)
                ),
            })

    write_csv(out / "m001_001_composited_detail.csv", composite_rows)
    write_csv(out / "m001_001_alpha_edge_counts.csv", boundary_rows)

    print("\nAlpha-Komposit-Auswertung erstellt:")
    print(" ", out / "m001_001_composited_detail.csv")
    print(" ", out / "m001_001_alpha_edge_counts.csv")


def main():
    for path in (CANDIDATE_CSV, WINNER_CSV, SOURCE, MASK):
        if not path.is_file():
            raise FileNotFoundError(f"Datei fehlt: {path}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = RUN / "debug_candidates" / f"selection_alpha_fixed_{stamp}"
    out.mkdir(parents=True, exist_ok=False)

    invariant_ok = run_selection_audit(out)
    run_alpha_composite_audit(out)

    print("\nAusgabeordner:", out)
    if not invariant_ok:
        print(
            "WARNUNG: Der Selektorvergleich hat noch ungeklärte Unterschiede. "
            "Gewinnerwechsel nicht interpretieren."
        )
    else:
        print("Auswahl-Invarianten bestanden.")
    print("FERTIG – Produktionsdateien unverändert.")


if __name__ == "__main__":
    main()
