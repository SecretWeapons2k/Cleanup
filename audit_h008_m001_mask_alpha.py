#!/usr/bin/env python3
"""Read-only Masken-/Alpha-Audit fuer h008_001 und m001_001."""

import csv
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT = Path.home() / "Downloads" / "Cleanup"

RUN = (
    PROJECT / "yolo_mask_winners"
    / "full_recheck_after_alpha_island_025_20261025_022053"
)

SOURCE = (
    Path.home() / "Downloads" / "yabai proj"
    / "yabaiproject - Copy" / "mit_kanji"
)

ANCHOR_PATHS = [
    PROJECT / "anchors.csv",
    Path.home() / "Downloads" / "yabai proj"
    / "processed_output" / "anchors.csv",
    RUN / "debug_candidates" / "anchors.csv",
]

IMAGES = ("h008_001", "m001_001")

VARIANTS = {
    "h008_001": [
        ("distance_10px", "DISTANCE_10PX"),
        ("hybrid_d2", "HYBRID_D2"),
        ("hybrid_d6", "HYBRID_D6"),
        ("contrast_t38", "CONTRAST_T38"),
        ("box_only", "BOX_ONLY"),
    ],
    "m001_001": [
        ("distance_00px", "DISTANCE_00PX"),
        ("hybrid_d2", "HYBRID_D2"),
        ("hybrid_d6", "HYBRID_D6"),
        ("contrast_t38", "CONTRAST_T38"),
        ("box_only", "BOX_ONLY"),
    ],
}

ALPHA_THRESHOLD = 16
DETAIL_THRESHOLDS = (8, 16, 24)


def find_anchor_file():
    for path in ANCHOR_PATHS:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "anchors.csv wurde an keinem der bekannten Pfade gefunden:\n"
        + "\n".join(str(p) for p in ANCHOR_PATHS)
    )


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


def stem(name):
    return Path(name).stem.casefold()


def read_rgba(path):
    if not path.is_file():
        raise FileNotFoundError(f"Datei fehlt: {path}")

    with Image.open(path) as im:
        original_mode = im.mode
        rgba = np.asarray(im.convert("RGBA"), dtype=np.uint8).copy()

    return rgba, ("A" in original_mode)


def read_mask(path, shape):
    if not path.is_file():
        raise FileNotFoundError(f"Maske fehlt: {path}")

    with Image.open(path) as im:
        mask = np.asarray(im.convert("L"), dtype=np.uint8) > 0

    if mask.shape != shape:
        raise RuntimeError(
            f"Maskenabmessungen stimmen nicht: {path}; "
            f"erwartet {shape}, erhalten {mask.shape}"
        )

    return mask


def parse_anchors(path):
    grouped = defaultdict(list)

    for row in read_csv(path):
        if row.get("image", "").strip():
            grouped[stem(row["image"])].append(row)

    return grouped


def get_box(row, width, height):
    try:
        x1, y1, x2, y2 = (
            int(round(float(row[k])))
            for k in ("x1", "y1", "x2", "y2")
        )
    except (ValueError, TypeError, KeyError):
        return None

    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))

    if x2 < x1 or y2 < y1:
        return None

    return x1, y1, x2, y2


def box_slice(box):
    x1, y1, x2, y2 = box
    return np.s_[y1:y2 + 1, x1:x2 + 1]


def checkerboard(h, w, cell=8):
    yy, xx = np.indices((h, w))
    tiles = ((xx // cell + yy // cell) % 2).astype(bool)

    bg = np.empty((h, w, 3), np.uint8)
    bg[tiles] = (226, 226, 226)
    bg[~tiles] = (188, 188, 188)

    return bg


def composite_rgba(rgba):
    rgb = rgba[:, :, :3].astype(np.float32)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    bg = checkerboard(*rgba.shape[:2]).astype(np.float32)

    out = rgb * alpha + bg * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def overlay_mask_and_boxes(rgba, mask, boxes):
    base = composite_rgba(rgba).astype(np.float32)

    # Cyan zeigt die tatsaechlich maskierte Flaeche.
    cyan = np.array([0, 210, 240], dtype=np.float32)
    base[mask] = 0.60 * base[mask] + 0.40 * cyan

    out = Image.fromarray(
        np.clip(base, 0, 255).astype(np.uint8), mode="RGB"
    )
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()

    for i, item in enumerate(boxes, start=1):
        x1, y1, x2, y2 = item["box"]
        draw.rectangle(
            (x1, y1, x2, y2), outline=(255, 30, 30), width=2
        )
        label = f"{i}: {item['row'].get('type', '?')}"
        draw.text((x1 + 2, max(0, y1 - 12)), label,
                  fill=(255, 30, 30), font=font)

    return out


def fit_tile(im, max_w=900, max_h=350):
    scale = min(max_w / im.width, max_h / im.height)
    size = (
        max(1, round(im.width * scale)),
        max(1, round(im.height * scale)),
    )
    return im.resize(size, Image.Resampling.NEAREST)


def save_contact_sheet(path, items):
    tile_w = 930
    tile_h = 390
    canvas = Image.new(
        "RGB", (tile_w, tile_h * len(items)), (248, 248, 248)
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for i, (label, im) in enumerate(items):
        y = i * tile_h
        draw.text((8, y + 5), label, fill=(20, 20, 20), font=font)
        thumb = fit_tile(im, tile_w - 16, tile_h - 35)
        canvas.paste(thumb, (8, y + 27))

    canvas.save(path)


def make_detail_map(rgba):
    rgb = rgba[:, :, :3]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    blur = cv2.GaussianBlur(
        gray, (0, 0), sigmaX=3.0, sigmaY=3.0
    )

    return np.abs(
        gray.astype(np.int16) - blur.astype(np.int16)
    )


def alpha_metrics(rgba):
    alpha = rgba[:, :, 3]
    visible = alpha > ALPHA_THRESHOLD

    result = {
        "height": alpha.shape[0],
        "width": alpha.shape[1],
        "alpha_gt_0": int(np.count_nonzero(alpha > 0)),
        "alpha_gt_16": int(np.count_nonzero(alpha > 16)),
        "alpha_gt_64": int(np.count_nonzero(alpha > 64)),
        "alpha_gt_128": int(np.count_nonzero(alpha > 128)),
        "alpha_gt_240": int(np.count_nonzero(alpha > 240)),
        "alpha_le_16": int(np.count_nonzero(alpha <= 16)),
    }

    result["transparent_ratio_alpha_le_16"] = round(
        result["alpha_le_16"] / alpha.size, 6
    )

    result["visible_ratio_alpha_gt_16"] = round(
        np.count_nonzero(visible) / alpha.size, 6
    )

    return result


def main():
    anchors_path = find_anchor_file()

    candidate_dir = RUN / "debug_candidates" / "cleaned_images"
    mask_dir = RUN / "masks"

    for path in (PROJECT, RUN, SOURCE, candidate_dir, mask_dir):
        if not path.is_dir():
            raise FileNotFoundError(f"Verzeichnis fehlt: {path}")

    anchors = parse_anchors(anchors_path)

    # Eingaben pruefen, bevor der Diagnoseordner angelegt wird.
    prepared = {}

    for name in IMAGES:
        source_path = SOURCE / f"{name}.png"
        source_rgba, has_alpha = read_rgba(source_path)

        h, w = source_rgba.shape[:2]
        selected_mask = read_mask(
            mask_dir / f"{name}.png", (h, w)
        )

        all_anchors = anchors.get(name.casefold(), [])
        box_items = []

        for row in all_anchors:
            box = get_box(row, w, h)
            if box is not None:
                box_items.append({"row": row, "box": box})

        if not box_items:
            raise RuntimeError(
                f"Keine gueltigen OCR-Boxen fuer {name} in {anchors_path}"
            )

        candidates = []

        for suffix, label in VARIANTS[name]:
            path = candidate_dir / f"{name}__{suffix}.png"
            rgba, _ = read_rgba(path)

            if rgba.shape[:2] != (h, w):
                raise RuntimeError(
                    f"Abmessungen stimmen nicht: {path}; "
                    f"Original {(h, w)}, Kandidat {rgba.shape[:2]}"
                )

            candidates.append({
                "label": label,
                "path": path,
                "rgba": rgba,
            })

        prepared[name] = {
            "source_path": source_path,
            "source": source_rgba,
            "source_has_alpha": has_alpha,
            "mask": selected_mask,
            "boxes": box_items,
            "candidates": candidates,
        }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = RUN / "debug_candidates" / f"mask_alpha_h008_m001_{stamp}"
    out.mkdir(parents=True, exist_ok=False)

    anchor_metrics = []
    image_metrics = []
    candidate_metrics = []
    summary = [
        "MASK- UND ALPHA-AUDIT (READ-ONLY)",
        f"Ausgabe: {out}",
        f"OCR-Datei: {anchors_path}",
        f"Aktuelle Ausgabe: {RUN}",
        f"Sichtbarkeitsschwelle: Alpha > {ALPHA_THRESHOLD}",
        "Lokaler Kontrast ist ein Diagnoseproxy, keine Texterkennung.",
        "Keine Produktionsdatei wird veraendert.",
        "",
    ]

    for name, data in prepared.items():
        src = data["source"]
        alpha = src[:, :, 3]
        mask = data["mask"]
        h, w = alpha.shape

        visible = alpha > ALPHA_THRESHOLD
        detail = make_detail_map(src)

        union_boxes = np.zeros((h, w), dtype=bool)

        for item in data["boxes"]:
            union_boxes[box_slice(item["box"])] = True

        src_stats = alpha_metrics(src)

        global_metrics = {
            "image": name,
            "source_has_alpha_channel": int(data["source_has_alpha"]),
            **{f"source_{k}": v for k, v in src_stats.items()},
            "winner_mask_pixels": int(np.count_nonzero(mask)),
            "visible_source_pixels_inside_winner_mask": int(
                np.count_nonzero(visible & mask)
            ),
            "visible_source_pixels_outside_winner_mask": int(
                np.count_nonzero(visible & ~mask)
            ),
            "visible_source_pixels_outside_all_ocr_boxes": int(
                np.count_nonzero(visible & ~union_boxes)
            ),
            "winner_mask_pixels_outside_all_ocr_boxes": int(
                np.count_nonzero(mask & ~union_boxes)
            ),
            "ocr_box_count": len(data["boxes"]),
        }

        image_metrics.append(global_metrics)

        summary += [
            f"===== {name} =====",
            f"Original: {data['source_path']}",
            f"Originalgroesse: {w}x{h}",
            f"Original besitzt Alpha-Kanal: {data['source_has_alpha']}",
            f"OCR-Boxen: {len(data['boxes'])}",
            f"Originalpixel Alpha > 16: {src_stats['alpha_gt_16']}",
            f"Sichtbare Originalpixel innerhalb Gewinner-Maske: "
            f"{global_metrics['visible_source_pixels_inside_winner_mask']}",
            f"Sichtbare Originalpixel ausserhalb Gewinner-Maske: "
            f"{global_metrics['visible_source_pixels_outside_winner_mask']}",
            f"Sichtbare Originalpixel ausserhalb aller OCR-Boxen: "
            f"{global_metrics['visible_source_pixels_outside_all_ocr_boxes']}",
            "",
        ]

        # Pro OCR-Box Abdeckung und sichtbare Restpixel messen.
        for i, item in enumerate(data["boxes"], start=1):
            row = item["row"]
            box = item["box"]
            sl = box_slice(box)

            box_mask = mask[sl]
            box_visible = visible[sl]
            box_detail = detail[sl]

            area = int(box_mask.size)
            visible_count = int(np.count_nonzero(box_visible))
            unmasked_visible = int(
                np.count_nonzero(box_visible & ~box_mask)
            )
            masked_visible = int(
                np.count_nonzero(box_visible & box_mask)
            )

            rec = {
                "image": name,
                "anchor_index": i,
                "anchor_id": row.get("anchor_id", ""),
                "row": row.get("row", ""),
                "col": row.get("col", ""),
                "type": row.get("type", ""),
                "text": row.get("text", ""),
                "text_lines": row.get("text_lines", ""),
                "x1": box[0], "y1": box[1],
                "x2": box[2], "y2": box[3],
                "box_area_px": area,
                "mask_pixels_in_box": int(np.count_nonzero(box_mask)),
                "mask_coverage_pct": round(
                    100.0 * np.count_nonzero(box_mask) / max(area, 1), 3
                ),
                "source_visible_pixels_alpha_gt16": visible_count,
                "source_visible_pixels_masked": masked_visible,
                "source_visible_pixels_unmasked": unmasked_visible,
            }

            for threshold in DETAIL_THRESHOLDS:
                strong = box_detail >= threshold
                rec[f"detail_pixels_ge_{threshold}"] = int(
                    np.count_nonzero(strong)
                )
                rec[f"detail_pixels_ge_{threshold}_unmasked"] = int(
                    np.count_nonzero(strong & ~box_mask)
                )

            anchor_metrics.append(rec)

        # Alpha-Bilder aller Kandidaten gegen Original vergleichen.
        for cand in data["candidates"]:
            rgba = cand["rgba"]
            cand_alpha = rgba[:, :, 3]
            alpha_changes = cand_alpha != alpha
            changed_inside = int(np.count_nonzero(alpha_changes & mask))
            changed_outside = int(np.count_nonzero(alpha_changes & ~mask))

            row = {
                "image": name,
                "candidate": cand["label"],
                "candidate_path": str(cand["path"]),
                **alpha_metrics(rgba),
                "alpha_changed_pixels_inside_winner_mask": changed_inside,
                "alpha_changed_pixels_outside_winner_mask": changed_outside,
                "source_visible_pixels_inside_mask": int(
                    np.count_nonzero(visible & mask)
                ),
                "source_visible_pixels_that_became_alpha_le16_inside_mask": int(
                    np.count_nonzero(visible & mask & (cand_alpha <= 16))
                ),
                "source_visible_pixels_still_visible_inside_mask": int(
                    np.count_nonzero(visible & mask & (cand_alpha > 16))
                ),
                "visible_candidate_pixels_outside_winner_mask": int(
                    np.count_nonzero((cand_alpha > 16) & ~mask)
                ),
            }

            candidate_metrics.append(row)

            # Ausgabe fuer m001: pruefen, ob Alpha ausserhalb der Maske
            # unveraendert geblieben ist.
            summary.append(
                f"{name}/{cand['label']}: "
                f"alpha>16 vorher={src_stats['alpha_gt_16']}, "
                f"nachher={row['alpha_gt_16']}; "
                f"Alpha-Aenderungen innerhalb Maske={changed_inside}, "
                f"ausserhalb Maske={changed_outside}"
            )

        # Overlay: Quelle + ausgewaehlte Maske + OCR-Boxen.
        overlay = overlay_mask_and_boxes(
            src, mask, data["boxes"]
        )
        overlay.save(out / f"{name}_source_mask_boxes.png")

        # Kandidatenvergleich auf alpha-korrektem Schachbrett.
        sheet_items = [
            ("SOURCE | original alpha-aware", Image.fromarray(composite_rgba(src)))
        ]

        for cand in data["candidates"]:
            sheet_items.append((
                f"{cand['label']} | alpha-aware",
                Image.fromarray(composite_rgba(cand["rgba"]))
            ))

        save_contact_sheet(
            out / f"{name}_alpha_candidate_comparison.png",
            sheet_items
        )

        # Sichtbarkeit, OCR-Box und Maske getrennt speichern.
        Image.fromarray(composite_rgba(src)).save(
            out / f"{name}_source_alpha_checkerboard.png"
        )

        summary.append("")

    write_csv(out / "anchor_mask_coverage.csv", anchor_metrics)
    write_csv(out / "image_alpha_summary.csv", image_metrics)
    write_csv(out / "candidate_alpha_metrics.csv", candidate_metrics)

    (out / "summary.txt").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )

    print("\n".join(summary))
    print("\n========== DIAGNOSEDATEIEN ==========")
    print("Ordner:", out)
    print("Zusammenfassung:", out / "summary.txt")
    print("OCR-/Maskenabdeckung:", out / "anchor_mask_coverage.csv")
    print("Alpha-Metriken:", out / "candidate_alpha_metrics.csv")
    print("Bildmetriken:", out / "image_alpha_summary.csv")

    for name in IMAGES:
        print("Overlay:", out / f"{name}_source_mask_boxes.png")
        print("Vergleich:", out / f"{name}_alpha_candidate_comparison.png")

    print("\nKeine Produktionsdatei wurde veraendert.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nFEHLER: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        print("Es wurden keine Produktionsdateien veraendert.")
