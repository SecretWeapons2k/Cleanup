#!/usr/bin/env python3
"""Alpha-aware Diagnose für m000_006 und s001_006.
Nur lesend: keine Produktionsdateien oder Kandidaten werden verändert.
"""

import csv
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path.home() / "Downloads" / "Cleanup"
RUN = PROJECT / "yolo_mask_winners" / "full_recheck_after_detection_20261024_030641"
DEBUG = RUN / "debug_candidates"
CANDIDATES = DEBUG / "cleaned_images"
MASKS = RUN / "masks"
SOURCE = (
    Path.home() / "Downloads" / "yabai proj"
    / "yabaiproject - Copy" / "mit_kanji"
)

ANCHOR_PATHS = [
    PROJECT / "anchors.csv",
    Path.home() / "Downloads" / "yabai proj" / "processed_output" / "anchors.csv",
    DEBUG / "anchors.csv",
]

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

IMAGE_EXTS = {".png", ".bmp", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ["image"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stem(name):
    return Path(name).stem.casefold()


def read_image(path):
    """Liefert BGR und Alpha separat; ignoriert Alpha nicht."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if raw is None:
        raise RuntimeError(f"Bild nicht lesbar: {path}")

    if raw.ndim == 2:
        bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        alpha = np.full(raw.shape, 255, np.uint8)
        has_alpha = False
    elif raw.shape[2] == 4:
        bgr = raw[:, :, :3]
        alpha = raw[:, :, 3]
        has_alpha = True
    else:
        bgr = raw[:, :, :3]
        alpha = np.full(raw.shape[:2], 255, np.uint8)
        has_alpha = False

    return bgr, alpha, has_alpha


def checkerboard(h, w, cell=8):
    yy, xx = np.indices((h, w))
    tiles = ((xx // cell + yy // cell) % 2).astype(bool)

    bg = np.empty((h, w, 3), np.uint8)
    bg[tiles] = (226, 226, 226)
    bg[~tiles] = (188, 188, 188)
    return bg


def composite(bgr, alpha):
    """Transparente Pixel werden über einem Schachbrett dargestellt."""
    bg = checkerboard(*alpha.shape)
    a = alpha.astype(np.float32)[:, :, None] / 255.0

    return np.clip(
        bgr.astype(np.float32) * a
        + bg.astype(np.float32) * (1.0 - a),
        0, 255
    ).astype(np.uint8)


def find_source(name):
    matches = [
        p for p in SOURCE.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
        and p.stem.casefold() == name.casefold()
    ]

    exact = [p for p in matches if p.name.casefold() == f"{name}.png"]

    if len(exact) == 1:
        return exact[0]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Original nicht gefunden: {name}")

    raise RuntimeError(f"Mehrere mögliche Originale für {name}: {matches}")


def load_anchors(path):
    grouped = defaultdict(list)

    for row in read_csv(path):
        if row.get("image", "").strip():
            grouped[stem(row["image"])].append(row)

    return grouped


def get_boxes(rows, w, h):
    boxes = []

    for row in rows:
        try:
            x1, y1, x2, y2 = (
                int(round(float(row[k])))
                for k in ("x1", "y1", "x2", "y2")
            )
        except (ValueError, TypeError, KeyError):
            continue

        x1, x2 = max(0, min(w - 1, x1)), max(0, min(w - 1, x2))
        y1, y2 = max(0, min(h - 1, y1)), max(0, min(h - 1, y2))

        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))

    return boxes


def box_mask(boxes, shape):
    h, w = shape
    mask = np.zeros((h, w), np.uint8)

    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    return mask > 0


def draw_boxes(img, boxes):
    out = img.copy()

    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), (20, 20, 245), 2)

    return out


def fit_tile(img, width, height, label):
    canvas = np.full((height, width, 3), 248, np.uint8)
    cv2.putText(
        canvas, label[:130], (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (25, 25, 25), 1, cv2.LINE_AA
    )

    h, w = img.shape[:2]
    scale = min((width - 16) / max(1, w), (height - 38) / max(1, h))

    nw = max(1, round(w * scale))
    nh = max(1, round(h * scale))

    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST
    thumb = cv2.resize(img, (nw, nh), interpolation=interpolation)

    x0 = (width - nw) // 2
    y0 = 30 + (height - 30 - nh) // 2

    canvas[y0:y0 + nh, x0:x0 + nw] = thumb
    return canvas


def save_vertical_sheet(path, rows, width=1600, height=190):
    canvas = np.full((len(rows) * height, width, 3), 248, np.uint8)

    for i, (label, img) in enumerate(rows):
        canvas[i * height:(i + 1) * height] = fit_tile(
            img, width, height, label
        )

    cv2.imwrite(str(path), canvas)


def alpha_stats(bgr, alpha, has_alpha):
    visible = alpha > 16

    return {
        "has_alpha_channel": int(has_alpha),
        "width": alpha.shape[1],
        "height": alpha.shape[0],
        "visible_pixels_alpha_gt16": int(visible.sum()),
        "transparent_pixels_alpha_le16": int((~visible).sum()),
        "transparent_ratio_pct": round(100 * float((~visible).mean()), 4),
    }


def main():
    for path in (PROJECT, RUN, CANDIDATES, MASKS, SOURCE):
        if not path.exists():
            raise FileNotFoundError(f"Pfad fehlt: {path}")

    anchors_path = next(
        (p for p in ANCHOR_PATHS if p.is_file()), None
    )

    if anchors_path is None:
        raise FileNotFoundError("Keine passende anchors.csv gefunden.")

    anchors = load_anchors(anchors_path)

    out = DEBUG / (
        "diagnostic_alpha_m000_s001_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out.mkdir(parents=True, exist_ok=False)

    stats = []
    summary = [
        "Alpha-aware Diagnose für m000_006 und s001_006",
        f"Originalbilder: {SOURCE}",
        f"Anchor-CSV: {anchors_path}",
        f"Ausgabeordner: {out}",
        "Transparente Pixel werden für Vorschauen über Schachbrett dargestellt.",
        "Rot = OCR-Boxen; Cyan = gespeicherte Auswahlmaske.",
        "Keine GPU-Inferenz; keine Produktionsdateien verändert.",
        "",
    ]

    # ---------------------------------------------------------
    # m000_006: Box- und Maskenabdeckung untersuchen
    # ---------------------------------------------------------
    name = "m000_006"
    source_path = find_source(name)
    src, src_alpha, src_has_alpha = read_image(source_path)
    h, w = src_alpha.shape

    boxes = get_boxes(anchors.get(name, []), w, h)
    boxes_union = box_mask(boxes, (h, w))

    mask_path = MASKS / f"{name}.png"
    selected_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    mask_dims_match = selected_mask is not None and selected_mask.shape == (h, w)

    if selected_mask is not None and not mask_dims_match:
        selected_mask = cv2.resize(
            selected_mask, (w, h), interpolation=cv2.INTER_NEAREST
        )

    selected = (
        selected_mask > 0 if selected_mask is not None
        else np.zeros((h, w), bool)
    )

    original_preview = composite(src, src_alpha)
    box_preview = draw_boxes(original_preview, boxes)

    mask_preview = draw_boxes(original_preview, boxes)

    if selected_mask is not None:
        contours, _ = cv2.findContours(
            (selected_mask > 0).astype(np.uint8),
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(mask_preview, contours, -1, (255, 220, 0), 1)

    rows = [
        ("ORIGINAL + OCR-BOXEN (rot)", box_preview),
        ("ORIGINAL + BOXEN (rot) + AUSWAHLMASKE (cyan)", mask_preview),
    ]

    for suffix, label in VARIANTS:
        p = CANDIDATES / f"{name}__{suffix}.png"

        if not p.is_file():
            rows.append((f"{label} — DATEI FEHLT", np.full_like(src, 248)))
            continue

        bgr, alpha, has_alpha = read_image(p)
        st = alpha_stats(bgr, alpha, has_alpha)
        stats.append({"image": name, "candidate": label, **st})

        rows.append((
            f"{label} | sichtbare Pixel alpha>16: {st['visible_pixels_alpha_gt16']}",
            composite(bgr, alpha)
        ))

    save_vertical_sheet(
        out / "m000_006_alpha_aware_candidates.png", rows
    )

    visible_source = src_alpha > 16
    visible_count = int(visible_source.sum())
    outside_box = int((visible_source & ~boxes_union).sum())
    outside_mask = int((visible_source & ~selected).sum())

    summary += [
        "m000_006 — BOX-/MASKEN-DIAGNOSE",
        f"Original: {source_path}",
        f"Bildgröße: {w}x{h}",
        f"Original besitzt Alpha-Kanal: {src_has_alpha}",
        f"Anzahl OCR-Boxen: {len(boxes)}",
        f"Boxen bedecken {100 * float(boxes_union.mean()):.2f}% des Bildes",
        f"Pixel mit Original-Alpha > 16: {visible_count}",
        f"Solche Pixel außerhalb der OCR-Boxen: {outside_box}",
        f"Solche Pixel außerhalb der Auswahlmaske: {outside_mask}",
        f"Maske entspricht Originalabmessungen: {mask_dims_match}",
        f"Auswahlbericht: {RUN / 'winner_report.csv'}",
        "",
        "HINWEIS: Bei einem opaken Original enthalten diese Pixelzahlen auch",
        "Hintergrund. Dann die Overlay-Grafik visuell beurteilen und die",
        "Zahlen nicht als direkte Schriftmessung interpretieren.",
        "",
    ]

    # ---------------------------------------------------------
    # s001_006: Alle gespeicherten Kandidaten alpha-korrekt zeigen
    # ---------------------------------------------------------
    name = "s001_006"
    source_path = find_source(name)
    src, src_alpha, src_has_alpha = read_image(source_path)
    h, w = src_alpha.shape
    boxes = get_boxes(anchors.get(name, []), w, h)

    tiles = [(
        "ORIGINAL + OCR-BOXEN (rot)",
        draw_boxes(composite(src, src_alpha), boxes)
    )]

    for suffix, label in VARIANTS:
        p = CANDIDATES / f"{name}__{suffix}.png"

        if not p.is_file():
            continue

        bgr, alpha, has_alpha = read_image(p)
        st = alpha_stats(bgr, alpha, has_alpha)
        stats.append({"image": name, "candidate": label, **st})

        tiles.append((
            f"{label} | sichtbare Pixel alpha>16: {st['visible_pixels_alpha_gt16']}",
            composite(bgr, alpha)
        ))

    cols, tile_w, tile_h = 3, 480, 380
    nrows = (len(tiles) + cols - 1) // cols
    sheet = np.full((nrows * tile_h, cols * tile_w, 3), 248, np.uint8)

    for i, (label, img) in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * tile_h:(r + 1) * tile_h,
              c * tile_w:(c + 1) * tile_w] = fit_tile(
                  img, tile_w, tile_h, label
              )

    cv2.imwrite(str(out / "s001_006_alpha_aware_candidates.png"), sheet)

    write_csv(out / "candidate_alpha_visibility.csv", stats)

    summary += [
        "s001_006 — ALPHA-KORREKTE KANDIDATENANSICHT",
        f"Original: {source_path}",
        f"Bildgröße: {w}x{h}",
        f"Original besitzt Alpha-Kanal: {src_has_alpha}",
        f"Anzahl OCR-Boxen: {len(boxes)}",
        "",
        "Ausgabe: s001_006_alpha_aware_candidates.png",
        "Alpha-Messwerte: candidate_alpha_visibility.csv",
        "Die Vorschauen berücksichtigen den Alphakanal.",
        "RGB-Farbwerte unter transparenten Pixeln werden nicht als sichtbar",
        "dargestellt. Dadurch ist dieser Vergleich besser mit den tatsächlichen",
        "Run-Kontaktbögen vergleichbar.",
        "",
        "Alle Ergebnisse sind neu angelegt. Produktionsdateien, vorhandene",
        "Kandidatenbilder, Masken und winner_report.csv bleiben unverändert.",
    ]

    summary_path = out / "diagnostic_summary.txt"
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")

    print("\n".join(summary))
    print("\nNEUE DATEIEN:")
    print(out / "m000_006_alpha_aware_candidates.png")
    print(out / "s001_006_alpha_aware_candidates.png")
    print(out / "candidate_alpha_visibility.csv")
    print(summary_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nFEHLER: {type(exc).__name__}: {exc}")
        traceback.print_exc()