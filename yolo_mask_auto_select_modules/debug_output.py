"""Debug image generation: cleaned candidates, mask overlays and paired previews."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

def _pillow_safe_label(label: str) -> str:
    """Keep debug labels printable with Pillow's built-in bitmap font (Latin-1)."""
    replacements = {
        "—": "-", "–": "-", "−": "-", "→": "->",
        "≤": "<=", "≥": ">=", "×": "x", "…": "...",
    }
    for source, replacement in replacements.items():
        label = label.replace(source, replacement)
    try:
        label.encode("latin-1")
    except UnicodeEncodeError:
        label = label.encode("latin-1", errors="replace").decode("latin-1")
    return label

def save_debug_contact_sheet(
    original_rgba: np.ndarray,
    items: list[tuple[str, np.ndarray]],
    output_path: Path,
    *,
    tile_width: int = 420,
    tile_height: int = 340,
) -> None:
    """Write a contact sheet containing only finished images, never masks.

    RGBA images are composited over a subtle checkerboard for previewing alpha.
    The individual PNGs preserve source alpha outside the mask and use LaMa's
    alpha reconstruction inside the mask.
    """
    if not items:
        return

    def preview_rgb(array: np.ndarray) -> Image.Image:
        arr = np.asarray(array, dtype=np.uint8)
        if arr.ndim == 2:
            return Image.fromarray(arr, mode="L").convert("RGB")
        if arr.shape[2] == 3:
            return Image.fromarray(arr, mode="RGB")
        if arr.shape[2] != 4:
            raise ValueError(f"Unsupported debug image shape: {arr.shape}")
        foreground = Image.fromarray(arr, mode="RGBA")
        alpha = foreground.getchannel("A")
        w, h = foreground.size
        tile = 16
        checker = Image.new("RGB", (w, h), (238, 238, 238))
        draw = ImageDraw.Draw(checker)
        for y in range(0, h, tile):
            for x in range(0, w, tile):
                if ((x // tile) + (y // tile)) % 2:
                    draw.rectangle((x, y, min(x + tile - 1, w - 1), min(y + tile - 1, h - 1)), fill=(205, 205, 205))
        checker.paste(foreground.convert("RGB"), mask=alpha)
        return checker

    all_items = [("Original", original_rgba), *items]
    columns = 3
    rows = math.ceil(len(all_items) / columns)
    label_height = 30
    canvas = Image.new("RGB", (columns * tile_width, rows * (tile_height + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for index, (label, array) in enumerate(all_items):
        col = index % columns
        row = index // columns
        x0 = col * tile_width
        y0 = row * (tile_height + label_height)
        is_winner = "[WINNER]" in label.upper()

        # Make the selected candidate visible at thumbnail size, not just in text.
        cell_box = (x0 + 2, y0 + 2, x0 + tile_width - 3, y0 + tile_height + label_height - 3)
        if is_winner:
            draw.rectangle((x0, y0, x0 + tile_width - 1, y0 + label_height - 1), fill=(255, 225, 110))
            draw.rectangle(cell_box, outline=(235, 125, 20), width=5)
            label = "WINNER | " + label.replace("[WINNER]", "").strip(" |-")
        draw.text((x0 + 8, y0 + 7), _pillow_safe_label(label), fill="black", font=font)
        tile = preview_rgb(array)
        tile.thumbnail((tile_width - 16, tile_height - 12), Image.Resampling.LANCZOS)
        px = x0 + (tile_width - tile.width) // 2
        py = y0 + label_height + (tile_height - tile.height) // 2
        canvas.paste(tile, (px, py))
        if is_winner:
            draw.rectangle(cell_box, outline=(235, 125, 20), width=5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)

def build_debug_mask_overlay(
    original_rgb: np.ndarray,
    candidate_mask: np.ndarray,
    boxes: list[dict[str, Any]],
) -> np.ndarray:
    """Render a magenta candidate-mask tint over raw RGB and outline OCR boxes.

    Body boxes are red; header boxes are violet. The image is RGB (not RGBA) so
    the mask tint remains visible even where the source alpha is transparent.
    """
    base = np.asarray(original_rgb, dtype=np.float32).copy()
    selected = np.asarray(candidate_mask).astype(bool)
    tint = np.array([255.0, 35.0, 185.0], dtype=np.float32)
    base[selected] = base[selected] * 0.48 + tint * 0.52
    overlay = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(overlay)
    width, height = overlay.size
    for box in boxes:
        try:
            x1 = max(0, min(width - 1, int(round(float(box["x1"])))))
            y1 = max(0, min(height - 1, int(round(float(box["y1"])))))
            x2 = max(0, min(width - 1, int(round(float(box["x2"])))))
            y2 = max(0, min(height - 1, int(round(float(box["y2"])))))
        except (KeyError, TypeError, ValueError):
            continue
        if x2 < x1 or y2 < y1:
            continue
        box_type = str(box.get("type", "")).casefold()
        color = (225, 45, 45) if box_type == "body" else ((170, 90, 215) if box_type == "header" else (255, 170, 0))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
    return np.asarray(overlay, dtype=np.uint8)

def save_debug_side_by_side_sheet(
    pairs: list[tuple[str, np.ndarray, str, np.ndarray]],
    output_path: Path,
    *,
    tile_width: int = 600,
    tile_height: int = 255,
) -> None:
    """Save one row per distance: raw+mask overlay on the left, cleaned result on right."""
    if not pairs:
        return

    def preview_rgb(array: np.ndarray) -> Image.Image:
        arr = np.asarray(array, dtype=np.uint8)
        if arr.ndim == 2:
            return Image.fromarray(arr, mode="L").convert("RGB")
        if arr.shape[2] == 3:
            return Image.fromarray(arr, mode="RGB")
        if arr.shape[2] != 4:
            raise ValueError(f"Unsupported debug image shape: {arr.shape}")
        foreground = Image.fromarray(arr, mode="RGBA")
        alpha = foreground.getchannel("A")
        w, h = foreground.size
        tile = 16
        checker = Image.new("RGB", (w, h), (238, 238, 238))
        draw = ImageDraw.Draw(checker)
        for y in range(0, h, tile):
            for x in range(0, w, tile):
                if ((x // tile) + (y // tile)) % 2:
                    draw.rectangle((x, y, min(x + tile - 1, w - 1), min(y + tile - 1, h - 1)), fill=(205, 205, 205))
        checker.paste(foreground.convert("RGB"), mask=alpha)
        return checker

    label_height = 28
    row_height = tile_height + label_height
    canvas = Image.new("RGB", (tile_width * 2, row_height * len(pairs)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for row, (left_label, left_image, right_label, right_image) in enumerate(pairs):
        y0 = row * row_height
        is_winner = "[WINNER]" in (left_label + " " + right_label).upper()
        row_box = (2, y0 + 2, tile_width * 2 - 3, y0 + row_height - 3)
        if is_winner:
            draw.rectangle((0, y0, tile_width * 2 - 1, y0 + label_height - 1), fill=(255, 225, 110))
            left_label = "WINNER | " + left_label.replace("[WINNER]", "").strip(" |-")
            right_label = "WINNER | " + right_label.replace("[WINNER]", "").strip(" |-")
        draw.text((8, y0 + 7), _pillow_safe_label(left_label), fill="black", font=font)
        draw.text((tile_width + 8, y0 + 7), _pillow_safe_label(right_label), fill="black", font=font)
        for col, array in enumerate((left_image, right_image)):
            tile = preview_rgb(array)
            tile.thumbnail((tile_width - 16, tile_height - 8), Image.Resampling.LANCZOS)
            x0 = col * tile_width
            x = x0 + (tile_width - tile.width) // 2
            y = y0 + label_height + (tile_height - tile.height) // 2
            canvas.paste(tile, (x, y))
        if is_winner:
            # Outline the entire paired raw-overlay/result row so it remains obvious
            # when several candidates are viewed together at reduced scale.
            draw.rectangle(row_box, outline=(235, 125, 20), width=5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)
