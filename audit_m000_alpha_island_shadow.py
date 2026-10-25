#!/usr/bin/env python3
"""Read-only Schattenaudit der Alpha-Komponentenregel für m000_006 und s001_006."""

import csv
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path.home() / "Downloads" / "Cleanup"
RUN = PROJECT / "yolo_mask_winners" / "full_recheck_after_detection_20261024_030641"
DEBUG = RUN / "debug_candidates"
CANDIDATES = DEBUG / "cleaned_images"
SOURCE = (
    Path.home() / "Downloads" / "yabai proj"
    / "yabaiproject - Copy" / "mit_kanji"
)

IMAGES = ("m000_006", "s001_006")
VARIANTS = [
    ("distance_00px", "DISTANCE_00PX"),
    ("distance_02px", "DISTANCE_02PX"),
    ("distance_04px", "DISTANCE_04PX"),
    ("distance_06px", "DISTANCE_06PX"),
    ("distance_08px", "DISTANCE_08PX"),
    ("distance_10px", "DISTANCE_10PX"),
    ("hybrid_d2", "HYBRID_D2"),
    ("hybrid_d6", "HYBRID_D6"),
    ("contrast_t38", "CONTRAST_T38"),
    ("smart_ink", "SMART_INK"),
    ("box_only", "BOX_ONLY"),
]

ALPHA_THRESHOLD = 16
MAX_COMPONENT_AREA = 64
CHECKER_SIZE = 8


def read_rgba(path: Path) -> np.ndarray:
    """Liest ein Bild als RGBA; das Original wird nicht verändert."""
    if not path.is_file():
        raise FileNotFoundError(f"Datei fehlt: {path}")
    with Image.open(path) as im:
        return np.asarray(im.convert("RGBA"), dtype=np.uint8).copy()


def source_transparent_ratio(alpha: np.ndarray) -> float:
    """Anteil der Originalpixel mit Alpha <= 16."""
    return float(np.mean(alpha <= ALPHA_THRESHOLD))


def component_stats(alpha: np.ndarray) -> dict:
    """Zählt sichtbare Pixel und zusammenhängende Komponenten."""
    visible = (alpha > ALPHA_THRESHOLD).astype(np.uint8)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        visible, connectivity=8
    )

    areas = (
        stats[1:, cv2.CC_STAT_AREA]
        if count > 1 else np.empty(0, dtype=np.int32)
    )
    small = areas[areas <= MAX_COMPONENT_AREA]

    return {
        "alpha_gt_0": int(np.count_nonzero(alpha > 0)),
        "alpha_gt_16": int(np.count_nonzero(alpha > 16)),
        "alpha_gt_64": int(np.count_nonzero(alpha > 64)),
        "alpha_gt_128": int(np.count_nonzero(alpha > 128)),
        "alpha_gt_240": int(np.count_nonzero(alpha > 240)),
        "visible_component_count": int(areas.size),
        "small_component_count_le_64": int(small.size),
        "small_component_pixels_le_64": int(small.sum()),
        "largest_visible_component": int(areas.max()) if areas.size else 0,
    }


def apply_shadow_rule(rgba: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Wendet die vorhandene 64-Pixel-Komponentenregel auf eine Kopie an.
    Anders als im normalen Produktionsablauf wird hier die globale
    Transparenz-Schwelle bewusst NICHT als Aktivierungsbedingung benutzt.
    """
    out = rgba.copy()
    alpha = out[:, :, 3]
    visible = (alpha > ALPHA_THRESHOLD).astype(np.uint8)

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        visible, connectivity=8
    )

    remove = np.zeros(visible.shape, dtype=bool)
    removed_components = 0

    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= MAX_COMPONENT_AREA:
            remove |= labels == label
            removed_components += 1

    removed_pixels = int(np.count_nonzero(remove))
    out[remove, :3] = 0
    out[remove, 3] = 0

    return out, {
        "shadow_removed_components": removed_components,
        "shadow_removed_pixels": removed_pixels,
    }


def composite_checkerboard(rgba: np.ndarray) -> Image.Image:
    """Stellt Alpha korrekt auf einem Schachbrett dar."""
    h, w = rgba.shape[:2]
    yy, xx = np.indices((h, w))
    tiles = ((xx // CHECKER_SIZE + yy // CHECKER_SIZE) % 2).astype(bool)

    bg = np.empty((h, w, 3), dtype=np.float32)
    bg[tiles] = (226, 226, 226)
    bg[~tiles] = (184, 184, 184)

    rgb = rgba[:, :, :3].astype(np.float32)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    comp = np.clip(rgb * alpha + bg * (1.0 - alpha), 0, 255).astype(np.uint8)
    return Image.fromarray(comp, mode="RGB")


def fit_image(im: Image.Image, max_w: int, max_h: int) -> Image.Image:
    """Skaliert ohne Verzerrung; beim Vergrößern bleiben Pixel gut erkennbar."""
    scale = min(max_w / im.width, max_h / im.height)
    size = (
        max(1, round(im.width * scale)),
        max(1, round(im.height * scale)),
    )
    return im.resize(size, Image.Resampling.NEAREST)


def save_m000_grid(path: Path, items: list) -> None:
    """Vorher/Nachher-Kontaktbogen für die sehr breite Textzeile."""
    width = 1400
    row_h = 100
    canvas = Image.new("RGB", (width, row_h * len(items) * 2), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    y = 0

    for item in items:
        name = item["candidate"]
        for state, rgba, visible_count, removed in (
            ("BEFORE", item["before"], item["before_stats"]["alpha_gt_16"], None),
            ("SHADOW", item["after"], item["after_stats"]["alpha_gt_16"],
             item["shadow_removed_pixels"]),
        ):
            draw.text(
                (10, y + 5),
                f"{name} | {state} | alpha>16={visible_count}"
                + (f" | entfernt={removed}" if removed is not None else ""),
                fill=(20, 20, 20),
                font=font,
            )
            comp = composite_checkerboard(rgba)
            thumb = fit_image(comp, width - 24, 62)
            canvas.paste(thumb, (12, y + 30))
            y += row_h

    canvas.save(path)


def save_s001_grid(path: Path, items: list) -> None:
    """Vergleicht alle Kandidaten in Vorher-/Nachher-Paaren."""
    cols = 3
    tile_w = 500
    tile_h = 340
    rows = (len(items) + cols - 1) // cols

    canvas = Image.new(
        "RGB", (cols * tile_w, rows * tile_h), (248, 248, 248)
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for i, item in enumerate(items):
        x = (i % cols) * tile_w
        y = (i // cols) * tile_h

        draw.text(
            (x + 8, y + 5),
            f"{item['candidate']} | entfernt={item['shadow_removed_pixels']} px",
            fill=(20, 20, 20),
            font=font,
        )
        draw.text((x + 10, y + 26), "BEFORE", fill=(30, 30, 30), font=font)
        draw.text((x + 255, y + 26), "SHADOW", fill=(30, 30, 30), font=font)

        before = fit_image(composite_checkerboard(item["before"]), 225, 270)
        after = fit_image(composite_checkerboard(item["after"]), 225, 270)

        canvas.paste(before, (x + 8, y + 48))
        canvas.paste(after, (x + 255, y + 48))

    canvas.save(path)


def main() -> None:
    for path in (PROJECT, RUN, CANDIDATES, SOURCE):
        if not path.is_dir():
            raise FileNotFoundError(f"Verzeichnis fehlt: {path}")

    # Erst alle Eingaben prüfen. Bis dahin wird nichts geschrieben.
    sources = {}
    candidate_paths = {}

    for name in IMAGES:
        sources[name] = SOURCE / f"{name}.png"
        if not sources[name].is_file():
            raise FileNotFoundError(f"Original fehlt: {sources[name]}")

        for suffix, label in VARIANTS:
            p = CANDIDATES / f"{name}__{suffix}.png"
            if not p.is_file():
                raise FileNotFoundError(f"Kandidat fehlt: {p}")
            candidate_paths[(name, label)] = p

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = DEBUG / f"alpha_island_shadow_{stamp}"
    out.mkdir(parents=True, exist_ok=False)

    rows = []
    prepared = {name: [] for name in IMAGES}
    summary = [
        "ALPHA-ISLAND SHADOW AUDIT (READ-ONLY)",
        f"Ausgabe: {out}",
        f"Alpha-Sichtbarkeitsschwelle: > {ALPHA_THRESHOLD}",
        f"Maximal entfernte Komponentenfläche: <= {MAX_COMPONENT_AREA} Pixel",
        "Die 40%-Aktivierungsschwelle wird nur für diesen Schattenvergleich umgangen.",
        "Jeder Kandidat wird im Speicher kopiert; Produktionsdateien werden nicht verändert.",
        "",
    ]

    for name in IMAGES:
        source = read_rgba(sources[name])
        h, w = source.shape[:2]
        source_ratio = source_transparent_ratio(source[:, :, 3])

        summary += [
            name,
            f"  Bildgröße: {w}x{h}",
            f"  Transparenz im Original (Alpha <= {ALPHA_THRESHOLD}): "
            f"{source_ratio * 100:.2f}%",
            f"  Normale 40%-Aktivierung würde laufen: "
            f"{'JA' if source_ratio >= 0.40 else 'NEIN'}",
        ]

        for suffix, label in VARIANTS:
            path = candidate_paths[(name, label)]
            before = read_rgba(path)

            if before.shape[:2] != (h, w):
                raise RuntimeError(
                    f"Bildgröße passt nicht: {path}; "
                    f"erwartet {(h, w)}, erhalten {before.shape[:2]}"
                )

            before_stats = component_stats(before[:, :, 3])
            after, shadow_metrics = apply_shadow_rule(before)
            after_stats = component_stats(after[:, :, 3])

            item = {
                "image": name,
                "candidate": label,
                "before": before,
                "after": after,
                "before_stats": before_stats,
                "after_stats": after_stats,
                **shadow_metrics,
            }
            prepared[name].append(item)

            row = {
                "image": name,
                "candidate": label,
                "source_transparent_ratio_pct": round(source_ratio * 100, 4),
                "normal_40pct_guard_would_run": int(source_ratio >= 0.40),
                "alpha_threshold": ALPHA_THRESHOLD,
                "max_component_area": MAX_COMPONENT_AREA,
                **{f"before_{k}": v for k, v in before_stats.items()},
                **shadow_metrics,
                **{f"after_{k}": v for k, v in after_stats.items()},
            }
            rows.append(row)

            summary.append(
                f"  {label:14s} | sichtbar vorher={before_stats['alpha_gt_16']:5d}"
                f" | nachher={after_stats['alpha_gt_16']:5d}"
                f" | entfernte Pixel={shadow_metrics['shadow_removed_pixels']:5d}"
                f" | entfernte Komponenten={shadow_metrics['shadow_removed_components']:4d}"
            )

        summary.append("")

    metrics_path = out / "candidate_shadow_metrics.csv"
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with metrics_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    save_m000_grid(out / "m000_006_shadow_comparison.png", prepared["m000_006"])
    save_s001_grid(out / "s001_006_shadow_comparison.png", prepared["s001_006"])

    summary += [
        "WICHTIGER VERGLEICH",
    ]

    for name, candidate in (
        ("m000_006", "DISTANCE_02PX"),
        ("m000_006", "CONTRAST_T38"),
        ("s001_006", "DISTANCE_00PX"),
        ("s001_006", "SMART_INK"),
    ):
        item = next(
            x for x in prepared[name] if x["candidate"] == candidate
        )
        summary.append(
            f"  {name}/{candidate}: "
            f"{item['before_stats']['alpha_gt_16']} -> "
            f"{item['after_stats']['alpha_gt_16']} sichtbare Pixel; "
            f"{item['shadow_removed_pixels']} Pixel durch Schattenregel entfernt."
        )

    summary += [
        "",
        "Ausgaben:",
        f"  {metrics_path}",
        f"  {out / 'm000_006_shadow_comparison.png'}",
        f"  {out / 's001_006_shadow_comparison.png'}",
        f"  {out / 'summary.txt'}",
        "",
        "Nur neue Diagnoseausgaben. Keine Produktionsdatei, Maske oder",
        "vorhandener Kandidat wurde überschrieben.",
    ]

    summary_path = out / "summary.txt"
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")

    print("\n".join(summary))
    print("\n========== FERTIG ==========")
    print("Diagnoseordner:", out)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nFEHLER: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        print("Es wurden keine Produktionsdateien verändert.")
