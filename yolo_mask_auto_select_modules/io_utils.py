"""Source discovery, CSV/box utilities and alpha post-processing."""
from __future__ import annotations

import csv
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .config import *

def choose_random_sample(
    available_names: set[str], sample_count: int, min_m_count: int, seed: int
) -> list[str]:
    """Choose a reproducible sample with at least min_m_count M-series images.

    When enough non-M images are available, the remaining slots are drawn from
    other series so the test is not dominated by M-series pages.
    """
    names = sorted(available_names, key=str.casefold)
    if sample_count <= 0:
        raise ValueError("--random-sample must be greater than zero")
    if sample_count > len(names):
        raise ValueError(f"Requested {sample_count} random images, but only {len(names)} anchor images are available")
    if min_m_count < 0 or min_m_count > sample_count:
        raise ValueError("--min-m-images must be between 0 and --random-sample")

    m_names = [name for name in names if Path(name).stem.casefold().startswith("m")]
    non_m_names = [name for name in names if name not in set(m_names)]
    if len(m_names) < min_m_count:
        raise ValueError(
            f"Need at least {min_m_count} M-series images, but anchors.csv contains only {len(m_names)}"
        )

    rng = random.Random(seed)
    chosen_m = rng.sample(m_names, min_m_count)
    slots_left = sample_count - len(chosen_m)

    if len(non_m_names) >= slots_left:
        chosen_other = rng.sample(non_m_names, slots_left)
    else:
        chosen_other = list(non_m_names)
        extra_slots = slots_left - len(chosen_other)
        remaining_m = [name for name in m_names if name not in set(chosen_m)]
        if extra_slots > len(remaining_m):
            raise ValueError("Not enough distinct images to complete the requested sample")
        chosen_other.extend(rng.sample(remaining_m, extra_slots))

    chosen = chosen_m + chosen_other
    rng.shuffle(chosen)
    return chosen

def image_name_from(row: dict[str, str]) -> str:
    return Path((row.get("image") or "").strip()).name

def read_anchor_csv(path: Path) -> tuple[list[str], list[dict[str, str]], dict[str, list[dict[str, Any]]]]:
    if not path.is_file():
        raise FileNotFoundError(f"anchors CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        required = {"image", "type", "x1", "y1", "x2", "y2"}
        missing = required - set(fieldnames)
        if missing:
            raise ValueError("anchors CSV missing columns: " + ", ".join(sorted(missing)))
        rows = list(reader)
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for n, row in enumerate(rows, start=2):
        name = image_name_from(row)
        if not name:
            continue
        try:
            coords = [float(row[k]) for k in ("x1", "y1", "x2", "y2")]
            x1, y1, x2, y2 = coords
            if not all(math.isfinite(v) for v in coords) or x2 <= x1 or y2 <= y1:
                continue
        except (KeyError, TypeError, ValueError):
            print(f"WARN: invalid anchor coordinates at CSV row {n}; ignored", file=sys.stderr)
            continue
        by_image[name].append({
            "type": (row.get("type") or "").strip().casefold(),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })
    if not by_image:
        raise ValueError(f"No valid anchor boxes found in {path}")
    return fieldnames, rows, dict(by_image)

def discover_sources(images_dir: Path, names: set[str], project: Path) -> dict[str, Path]:
    # Match the routing-audit script: the configured image collection is authoritative.
    # Do not also scan Cleanup/raw, because it may intentionally contain duplicate
    # copies of images selected from the larger source collection.
    found: dict[str, list[Path]] = defaultdict(list)
    root = images_dir.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Source image directory not found: {root}")
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS and path.name in names:
            found[path.name].append(path)
    missing = sorted(names - set(found))
    duplicates = {name: paths for name, paths in found.items() if len(paths) > 1}
    if missing:
        raise FileNotFoundError(
            f"Could not locate {len(missing)} source image(s): {', '.join(missing[:12])}. "
            f"Check --images-dir ({images_dir})."
        )
    if duplicates:
        detail = "; ".join(f"{name}: {len(paths)} files" for name, paths in list(duplicates.items())[:8])
        raise ValueError("Ambiguous source image basenames; refusing to guess: " + detail)
    return {name: paths[0] for name, paths in found.items()}

def find_iopaint() -> str:
    candidate = Path(sys.executable).absolute().with_name("iopaint")
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    found = shutil.which("iopaint")
    if found:
        return found
    raise FileNotFoundError(
        f"IOPaint was not found beside Python ({candidate}) or on PATH. "
        "Activate Cleanup/.venv before running this script."
    )

def choose_output_dir(path: Path, project: Path) -> Path:
    resolved = path.expanduser().resolve()
    for protected in (project / "raw", project / "cleaned", project / ".venv"):
        protected = protected.resolve()
        if resolved == protected or protected in resolved.parents:
            raise ValueError(f"Output may not be inside protected folder {protected}")
    if resolved.exists() and any(resolved.iterdir()):
        stamp = time.strftime("run_%Y%m%d_%H%M%S")
        candidate = resolved / stamp
        n = 1
        while candidate.exists():
            candidate = resolved / f"{stamp}_{n}"
            n += 1
        return candidate
    return resolved

def write_filtered_anchors(path: Path, fieldnames: list[str], rows: list[dict[str, str]], names: set[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row for row in rows if image_name_from(row) in names)

def read_routing_rows(path: Path) -> dict[str, dict[str, str]]:
    by_image: dict[str, dict[str, str]] = {}
    if not path.is_file():
        raise FileNotFoundError(f"Routing report missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("confidence_threshold") == "0.05" and row.get("image"):
                by_image.setdefault(row["image"], row)
    return by_image

def read_box_mask(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Expected box-only mask does not exist: {path}")
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8) > 0

def rgb_array(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(path) as im:
        rgba = im.convert("RGBA")
        return np.asarray(rgba.convert("RGB"), dtype=np.uint8), np.asarray(rgba.getchannel("A"), dtype=np.uint8)

def make_box_only_mask(width: int, height: int, boxes: list[dict[str, Any]]) -> np.ndarray:
    """Build the same inclusive rectangle union as the routing-audit script."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for box in boxes:
        try:
            x1 = max(0, min(width - 1, math.floor(float(box["x1"]))))
            y1 = max(0, min(height - 1, math.floor(float(box["y1"]))))
            x2 = max(0, min(width - 1, math.ceil(float(box["x2"]))))
            y2 = max(0, min(height - 1, math.ceil(float(box["y2"]))))
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if x2 >= x1 and y2 >= y1:
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return mask

def remove_small_alpha_islands(
    rgb: np.ndarray, alpha: np.ndarray, source_alpha: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Remove tiny disconnected visible-alpha components for transparent-canvas sprites.

    Activation is based on the SOURCE image, not the candidate output, so every
    distance candidate for the same image follows the same rule. The transparency
    ratio counts source pixels whose alpha is <= ALPHA_VISIBILITY_THRESHOLD.
    Components are 8-connected; any isolated visible component up to the configured
    size is made fully transparent. RGB is set to zero at removed pixels to avoid
    leaving hidden colour data behind.
    """
    transparent_ratio = float(np.mean(source_alpha <= ALPHA_VISIBILITY_THRESHOLD))
    metadata = {
        "source_transparent_ratio": transparent_ratio,
        "alpha_small_object_removal_applied": 0.0,
        "alpha_small_components_removed": 0.0,
        "alpha_small_pixels_removed": 0.0,
    }
    if (not POSTPROCESS_REMOVE_SMALL_ALPHA_ISLANDS
            or transparent_ratio < ALPHA_SMALL_OBJECT_MIN_TRANSPARENT_RATIO):
        return rgb, alpha, metadata

    visible = (alpha > ALPHA_VISIBILITY_THRESHOLD).astype(np.uint8)
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(visible, connectivity=8)
    remove = np.zeros(visible.shape, dtype=bool)
    removed_components = 0
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= ALPHA_SMALL_OBJECT_MAX_PIXELS:
            remove |= labels == label
            removed_components += 1

    if removed_components:
        rgb_out = rgb.copy()
        alpha_out = alpha.copy()
        rgb_out[remove] = 0
        alpha_out[remove] = 0
    else:
        rgb_out, alpha_out = rgb, alpha

    metadata.update({
        "alpha_small_object_removal_applied": 1.0,
        "alpha_small_components_removed": float(removed_components),
        "alpha_small_pixels_removed": float(np.count_nonzero(remove)),
    })
    return rgb_out, alpha_out, metadata


