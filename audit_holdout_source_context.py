#!/usr/bin/env python3
"""Read-only, CPU-only visual comparison: source vs current vs proposed cleanup candidates."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

HOME = Path.home()
PROJECT = HOME / "Downloads" / "Cleanup"
RUN_DIR = PROJECT / "yolo_mask_winners" / "full_recheck_after_detection_20261024_030641"
DEBUG_DIR = RUN_DIR / "debug_candidates"
CANDIDATE_DIR = DEBUG_DIR / "cleaned_images"
ORIGINAL_DIR = HOME / "Downloads" / "yabai proj" / "yabaiproject - Copy" / "mit_kanji"
HOLDOUT_DIR = DEBUG_DIR / "independent_color_rule_holdout_20261024_061544"
REVIEW_CSV = HOLDOUT_DIR / "holdout_manual_review.csv"
OUT_FILE = HOLDOUT_DIR / "holdout_source_context_comparison_01.jpg"
ANCHOR_CSV_OPTIONS = [
    PROJECT / "anchors.csv",
    HOME / "Downloads" / "yabai proj" / "processed_output" / "anchors.csv",
    DEBUG_DIR / "anchors.csv",
]
IMAGE_EXTS = {".png", ".bmp", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}

TILE_W, TILE_H, LABEL_H = 640, 260, 58
PAD_X, PAD_Y = 12, 8


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Bild kann nicht gelesen werden: {path}")
    return image


def find_unique_by_stem(stem: str, index: dict[str, list[Path]], description: str) -> Path:
    choices = index.get(stem.lower(), [])
    exact = [p for p in choices if p.stem.lower() == stem.lower()]
    if len(exact) == 1:
        return exact[0]
    if len(choices) == 1:
        return choices[0]
    if not choices:
        raise FileNotFoundError(f"{description} nicht gefunden für {stem}")
    raise RuntimeError(f"Mehrere {description} für {stem}: " + ", ".join(map(str, choices[:8])))


def candidate_path(image_name: str, candidate: str) -> Path:
    stem = Path(image_name).stem
    expected = CANDIDATE_DIR / f"{stem}__{candidate.lower()}.png"
    if expected.is_file():
        return expected
    matches = [p for p in CANDIDATE_DIR.glob(f"{stem}__*")
               if p.suffix.lower() in IMAGE_EXTS and p.stem.lower() == f"{stem}__{candidate}".lower()]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Kandidatenbild fehlt/uneindeutig: {image_name} | {candidate}")


def get_box(row: dict[str, str], w: int, h: int) -> tuple[int, int, int, int] | None:
    try:
        x1 = max(0, min(w, int(round(float(row["x1"])))) )
        y1 = max(0, min(h, int(round(float(row["y1"])))) )
        x2 = max(0, min(w, int(round(float(row["x2"])))) )
        y2 = max(0, min(h, int(round(float(row["y2"])))) )
    except (KeyError, ValueError, TypeError):
        return None
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def annotate_and_crop(image: np.ndarray, original_size: tuple[int, int], anchors: list[dict[str, str]]) -> np.ndarray:
    h, w = image.shape[:2]
    ow, oh = original_size
    sx, sy = w / max(1, ow), h / max(1, oh)
    boxes: list[tuple[int, int, int, int]] = []
    for anchor in anchors:
        obox = get_box(anchor, ow, oh)
        if obox is None:
            continue
        x1, y1, x2, y2 = obox
        box = (
            max(0, min(w, int(round(x1 * sx)))),
            max(0, min(h, int(round(y1 * sy)))),
            max(0, min(w, int(round(x2 * sx)))),
            max(0, min(h, int(round(y2 * sy)))),
        )
        if box[2] > box[0] and box[3] > box[1]:
            boxes.append(box)
            cv2.rectangle(image, (box[0], box[1]), (box[2] - 1, box[3] - 1), (0, 0, 255), 1)
    if not boxes:
        return image
    x1 = max(0, min(b[0] for b in boxes) - PAD_X)
    y1 = max(0, min(b[1] for b in boxes) - PAD_Y)
    x2 = min(w, max(b[2] for b in boxes) + PAD_X)
    y2 = min(h, max(b[3] for b in boxes) + PAD_Y)
    return image[y1:y2, x1:x2]


def paste_tile(canvas: np.ndarray, image: np.ndarray, col: int, y0: int, label: str) -> None:
    x0 = col * TILE_W
    cv2.putText(canvas, label[:100], (x0 + 10, y0 + 43), cv2.FONT_HERSHEY_SIMPLEX,
                0.53, (20, 20, 20), 1, cv2.LINE_AA)
    h, w = image.shape[:2]
    scale = min((TILE_W - 12) / max(1, w), (TILE_H - 8) / max(1, h))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    thumb = cv2.resize(image, (nw, nh), interpolation=interp)
    xx = x0 + (TILE_W - nw) // 2
    yy = y0 + LABEL_H + (TILE_H - nh) // 2
    canvas[yy:yy + nh, xx:xx + nw] = thumb


def main() -> None:
    for path in (PROJECT, RUN_DIR, CANDIDATE_DIR, ORIGINAL_DIR, REVIEW_CSV):
        if not path.exists():
            raise FileNotFoundError(f"Benötigter vorhandener Pfad fehlt: {path}")
    anchor_path = next((p for p in ANCHOR_CSV_OPTIONS if p.is_file()), None)
    if anchor_path is None:
        raise FileNotFoundError("anchors.csv nicht gefunden. Geprüfte Pfade:\n" + "\n".join(map(str, ANCHOR_CSV_OPTIONS)))

    reviews = read_csv(REVIEW_CSV)
    if not reviews:
        raise RuntimeError(f"Keine Review-Zeilen in {REVIEW_CSV}")
    anchors_by_stem: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(anchor_path):
        if row.get("image", "").strip():
            anchors_by_stem[Path(row["image"].strip()).stem.lower()].append(row)

    originals_by_stem: dict[str, list[Path]] = defaultdict(list)
    for path in ORIGINAL_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            originals_by_stem[path.stem.lower()].append(path)

    row_h = TILE_H + LABEL_H
    canvas = np.full((len(reviews) * row_h, TILE_W * 3, 3), 248, dtype=np.uint8)
    print(f"Review-Zeilen: {len(reviews)}")
    print(f"Ankerdatei: {anchor_path}")
    print("Es werden nur vorhandene Bilder gelesen; keine GPU-Inferenz und keine Produktionsdateien werden verändert.")

    for i, row in enumerate(reviews):
        image_name = row["image"].strip()
        stem = Path(image_name).stem.lower()
        current_name = row["current_candidate"].strip()
        proposed_name = row["recommended_candidate"].strip()
        anchors = anchors_by_stem.get(stem, [])
        if not anchors:
            raise RuntimeError(f"Keine OCR-Anker für {image_name}")
        original_path = find_unique_by_stem(Path(image_name).stem, originals_by_stem, "Originalbild")
        current_path = candidate_path(image_name, current_name)
        proposed_path = candidate_path(image_name, proposed_name)

        original = load_image(original_path)
        oh, ow = original.shape[:2]
        original_crop = annotate_and_crop(original.copy(), (ow, oh), anchors)
        current = load_image(current_path)
        current_crop = annotate_and_crop(current.copy(), (ow, oh), anchors)
        proposed = load_image(proposed_path)
        proposed_crop = annotate_and_crop(proposed.copy(), (ow, oh), anchors)

        y0 = i * row_h
        heading = f"{image_name} | {current_name} -> {proposed_name}"
        cv2.putText(canvas, heading[:150], (10, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (20, 20, 20), 1, cv2.LINE_AA)
        paste_tile(canvas, original_crop, 0, y0, "SOURCE (red = OCR anchor)")
        paste_tile(canvas, current_crop, 1, y0, f"CURRENT: {current_name} (red = anchor)")
        paste_tile(canvas, proposed_crop, 2, y0, f"PROPOSED: {proposed_name} (red = anchor)")

    cv2.imwrite(str(OUT_FILE), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    print(f"Vergleich geschrieben: {OUT_FILE}")
    print("Bitte den Vergleichsbogen visuell prüfen: Ist im PROPOSED-Bild nur alter Text entfernt, oder verschwinden auch Motiv-/Hintergrunddetails innerhalb der roten OCR-Anker?")


if __name__ == "__main__":
    main()
