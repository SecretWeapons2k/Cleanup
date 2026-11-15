"""Shared geometry: polygon anchors (instead of plain boxes) and protected zones.

Used by the pipeline, the scoring module and the routing audit so that all of them
rasterise anchors and protected zones in exactly the same way.

Box dictionaries keep their rectangle (``x1..y2``). If a usable polygon exists, it is
stored under ``"poly"`` (float array, shape (N, 2)); axis-aligned rectangles are
stored as plain boxes with ``poly=None`` so the historic rectangle code path and its
pixel semantics (inclusive floor/ceil rectangle) stay untouched.
"""
from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import (
    PROTECTED_ZONES_FILENAME,
    USE_BODY_MASK_POLYGONS,
    USE_HEADER_POLYGONS,
)


def resolve_protected_zones_csv(
    anchors_path: Path,
    explicit_path: Path | None = None,
) -> Path | None:
    """Resolve an explicit protected-zones CSV or discover it beside anchors.csv.

    An explicit path is returned even when it does not exist so the subsequent
    CSV reader can raise a clear FileNotFoundError. Without an explicit path,
    protected_zones.csv is optional and is only returned when present next to
    the selected anchors CSV.
    """
    if explicit_path is not None:
        return explicit_path.expanduser().resolve()
    candidate = anchors_path.expanduser().resolve().with_name(PROTECTED_ZONES_FILENAME)
    return candidate if candidate.is_file() else None


def parse_polygon(text: str | None) -> np.ndarray | None:
    """Parse ``"x,y;x,y;..."``. Returns None for empty or invalid input."""
    if not text or not str(text).strip():
        return None
    try:
        pts = [tuple(float(v) for v in part.split(",")) for part in str(text).split(";") if part.strip()]
        arr = np.asarray(pts, dtype=np.float64)
    except ValueError:
        return None
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] != 2 or not np.isfinite(arr).all():
        return None
    return arr


def _is_axis_aligned_rect(poly: np.ndarray) -> bool:
    return (
        poly.shape[0] == 4
        and len(set(poly[:, 0].tolist())) == 2
        and len(set(poly[:, 1].tolist())) == 2
    )


def parse_polygons(text: str | None) -> list[np.ndarray] | None:
    """Parse Yabai mask_polygon; disconnected components are separated by '|'."""
    if not text or not str(text).strip():
        return None
    pieces = [piece.strip() for piece in str(text).split('|')]
    if not pieces or any(not piece for piece in pieces):
        return None
    polygons: list[np.ndarray] = []
    for piece in pieces:
        polygon = parse_polygon(piece)
        if polygon is None:
            return None
        polygons.append(polygon)
    return polygons


def anchor_polygon(row: dict[str, Any]) -> np.ndarray | list[np.ndarray] | None:
    """Return white cleaning geometry; None means use the red rectangle fallback.

    Body anchors use ONLY mask_polygon. The separate polygon field is the
    original OCR contour and must not replace the red box if the white mask is absent.
    Header anchors continue to use their rotated OCR polygon.
    """
    kind = str(row.get("type") or "").strip().casefold()
    if kind == "body":
        if not USE_BODY_MASK_POLYGONS:
            return None
        polygons = parse_polygons(row.get("mask_polygon"))
        if not polygons:
            return None
        if len(polygons) == 1 and _is_axis_aligned_rect(polygons[0]):
            return None
        return polygons[0] if len(polygons) == 1 else polygons

    if not USE_HEADER_POLYGONS:
        return None
    poly = parse_polygon(row.get("polygon"))
    if poly is None or _is_axis_aligned_rect(poly):
        return None
    return poly



def box_region(box: dict[str, Any], width: int, height: int) -> tuple[int, int, int, int, np.ndarray] | None:
    """Return ``(x1, y1, x2, y2, crop_mask)`` with inclusive pixel bounds.

    ``crop_mask`` is a bool array of shape ``(y2-y1+1, x2-x1+1)``. Rectangles use the
    historic floor/ceil clamp; polygons use rounded vertices clamped to the image.
    """
    poly_value = box.get("poly")
    if poly_value is None:
        try:
            x1 = max(0, min(width - 1, math.floor(float(box["x1"]))))
            y1 = max(0, min(height - 1, math.floor(float(box["y1"]))))
            x2 = max(0, min(width - 1, math.ceil(float(box["x2"]))))
            y2 = max(0, min(height - 1, math.ceil(float(box["y2"]))))
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        if x2 < x1 or y2 < y1:
            return None
        return x1, y1, x2, y2, np.ones((y2 - y1 + 1, x2 - x1 + 1), dtype=bool)

    raw_polygons = [poly_value] if isinstance(poly_value, np.ndarray) else list(poly_value)
    if not raw_polygons:
        return None
    points: list[np.ndarray] = []
    for raw_poly in raw_polygons:
        pts = np.rint(np.asarray(raw_poly, dtype=np.float64)).astype(np.int32)
        if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
            return None
        pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
        points.append(pts)
    all_points = np.concatenate(points, axis=0)
    x1, y1 = int(all_points[:, 0].min()), int(all_points[:, 1].min())
    x2, y2 = int(all_points[:, 0].max()), int(all_points[:, 1].max())
    crop = np.zeros((y2 - y1 + 1, x2 - x1 + 1), dtype=np.uint8)
    offset = np.array([x1, y1], dtype=np.int32)
    cv2.fillPoly(crop, [pts - offset for pts in points], 255)
    return x1, y1, x2, y2, crop > 0


def box_mask(box: dict[str, Any], width: int, height: int) -> np.ndarray:
    """Full-image bool mask of one anchor (polygon if present, else rectangle)."""
    out = np.zeros((height, width), dtype=bool)
    region = box_region(box, width, height)
    if region is not None:
        x1, y1, x2, y2, crop = region
        out[y1:y2 + 1, x1:x2 + 1] |= crop
    return out


def boxes_union_mask(boxes: list[dict[str, Any]], width: int, height: int) -> np.ndarray:
    out = np.zeros((height, width), dtype=bool)
    for box in boxes:
        region = box_region(box, width, height)
        if region is not None:
            x1, y1, x2, y2, crop = region
            out[y1:y2 + 1, x1:x2 + 1] |= crop
    return out


def inset_region(crop: np.ndarray, inset_x: int, inset_y: int) -> np.ndarray:
    """Erode a region mask by (inset_x, inset_y) pixels; fall back to the full region if it vanishes."""
    h, w = crop.shape
    padded = np.pad(crop.astype(np.uint8), ((inset_y, inset_y), (inset_x, inset_x)))
    kernel = np.ones((2 * inset_y + 1, 2 * inset_x + 1), dtype=np.uint8)
    eroded = cv2.erode(padded, kernel)[inset_y:inset_y + h, inset_x:inset_x + w] > 0
    return eroded if eroded.any() else crop


# --------------------------------------------------------------------------- protected zones

def read_protected_zones(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Read ``protected_zones.csv`` -> ``{image_name: [zone_box, ...]}``.

    Required column: ``image``. Geometry from ``polygon`` or, if missing/invalid, from
    ``x1,y1,x2,y2``. Invalid rows are skipped with a warning.
    """
    if not path.is_file():
        raise FileNotFoundError(f"protected zones CSV not found: {path}")
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if "image" not in (reader.fieldnames or []):
            raise ValueError("protected zones CSV missing column: image")
        for n, row in enumerate(reader, start=2):
            name = Path((row.get("image") or "").strip()).name
            if not name:
                continue
            poly = parse_polygon(row.get("polygon"))
            try:
                if poly is not None:
                    x1, y1 = poly.min(axis=0)
                    x2, y2 = poly.max(axis=0)
                else:
                    x1, y1, x2, y2 = (float(row[k]) for k in ("x1", "y1", "x2", "y2"))
                if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 < x1 or y2 < y1:
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                print(f"WARN: invalid protected zone at CSV row {n}; ignored", file=sys.stderr)
                continue
            if poly is not None and _is_axis_aligned_rect(poly):
                poly = None
            by_image[name].append({
                "zone_id": (row.get("zone_id") or "").strip(),
                "text": (row.get("text") or "").strip(),
                "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                "poly": poly,
            })
    return dict(by_image)


def protected_mask(zones: list[dict[str, Any]], width: int, height: int, margin_px: int = 0) -> np.ndarray:
    """Full-image bool mask of all protected zones, optionally grown by ``margin_px``."""
    mask = boxes_union_mask(zones, width, height)
    if margin_px > 0 and mask.any():
        k = margin_px * 2 + 1
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((k, k), dtype=np.uint8)) > 0
    return mask
