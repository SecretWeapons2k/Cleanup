"""Single source of truth for all candidate masks of one image.

Every candidate mask (YOLO distances, hybrids, contrast, smart-ink, box-only) and every
reference mask used for scoring is built here, with the protected zones already
subtracted. LaMa staging, compositing and scoring all use this one result, so the mask
LaMa inpaints is always the mask that is composited and scored.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .geometry import protected_mask
from .io_utils import make_box_only_mask
from .scoring import make_contrast_mask, make_smart_ink_mask

HYBRID_DISTANCES = (2, 6)


@dataclass
class ImageMaskSet:
    candidates: dict[int | str, np.ndarray]
    box_mask: np.ndarray
    smart_mask: np.ndarray
    body_box_mask: np.ndarray
    yolo_reference_mask: np.ndarray
    scoring_envelope: np.ndarray
    protected: np.ndarray
    zones: list[dict[str, Any]] = field(default_factory=list)


def stage_token(candidate_id: int | str) -> str:
    """File-name token used when staging a candidate for LaMa (unchanged naming)."""
    if isinstance(candidate_id, int):
        return f"d{candidate_id:02d}"
    if candidate_id.startswith("HYBRID_D"):
        return f"hybrid_d{int(candidate_id[len('HYBRID_D'):]):02d}"
    return candidate_id.lower()


def build_image_mask_set(
    *,
    rgb: np.ndarray,
    alpha: np.ndarray,
    boxes: list[dict[str, Any]],
    zones: list[dict[str, Any]],
    yolo_masks: dict[int, np.ndarray],
    box_only_mask: np.ndarray,
    contrast_thresholds: tuple[float, ...],
    contrast_candidate_ids: list[str],
    protected_margin_px: int = 0,
) -> ImageMaskSet:
    height, width = rgb.shape[:2]
    protected = protected_mask(zones, width, height, protected_margin_px)
    has_protection = bool(protected.any())

    def prot(mask: np.ndarray) -> np.ndarray:
        mask = np.asarray(mask) > 0
        return mask & ~protected if has_protection else mask

    masks = {d: prot(m) for d, m in yolo_masks.items()}
    box_mask = prot(box_only_mask)
    smart_mask = prot(make_smart_ink_mask(rgb, boxes))
    body_boxes = [b for b in boxes if str(b.get("type", "")).casefold() == "body"]
    body_box_mask = prot(make_box_only_mask(width, height, body_boxes))

    candidates: dict[int | str, np.ndarray] = dict(sorted(masks.items()))
    for d in HYBRID_DISTANCES:
        candidates[f"HYBRID_D{d}"] = box_mask | masks[d]
    for threshold, candidate_id in zip(contrast_thresholds, contrast_candidate_ids):
        candidates[candidate_id] = prot(make_contrast_mask(rgb, alpha, boxes, threshold))
    candidates["SMART_INK"] = smart_mask
    candidates["BOX_ONLY"] = box_mask

    yolo_reference = masks[max(masks)]
    return ImageMaskSet(
        candidates=candidates,
        box_mask=box_mask,
        smart_mask=smart_mask,
        body_box_mask=body_box_mask,
        yolo_reference_mask=yolo_reference,
        scoring_envelope=yolo_reference | box_mask | smart_mask,
        protected=protected,
        zones=zones,
    )


def enforce_protection(
    final_rgb: np.ndarray, final_alpha: np.ndarray,
    original_rgb: np.ndarray, original_alpha: np.ndarray,
    protected: np.ndarray,
) -> dict[str, float]:
    """Hard guarantee: protected pixels are restored to the original (in place).

    Returns metrics; ``protected_pixels_restored`` > 0 means some earlier step touched a
    protected pixel (a bug worth investigating) - the pixels are restored regardless.
    """
    count = int(np.count_nonzero(protected))
    if not count:
        return {"protected_pixels": 0.0, "protected_pixels_restored": 0.0}
    differs = (final_rgb != original_rgb).any(axis=-1) | (final_alpha != original_alpha)
    restored = int(np.count_nonzero(differs & protected))
    final_rgb[protected] = original_rgb[protected]
    final_alpha[protected] = original_alpha[protected]
    return {"protected_pixels": float(count), "protected_pixels_restored": float(restored)}
