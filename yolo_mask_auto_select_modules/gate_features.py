"""Absolute quality features for ONE filled region (basis of the usability gate).

The old scorer compares candidates against each other (best text score + tolerance). A sieve
needs the opposite: a verdict about a single result that does not depend on any competitor.
This module only MEASURES; it makes no decision. Thresholds are calibrated later against
human labels (see calibrate_gate.py), so every value here is a plain, interpretable number.

All inputs are crops of the same window (uint8 RGB / alpha, bool masks):
    src_*        original image
    res_*        result after filling (and repair / protection)
    mask         pixels of THIS region that were filled
    other_mask   filled pixels of OTHER regions inside the window (excluded from the context ring)
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

from .config import INK_STRENGTH_THRESHOLD
from .scoring import ink_strength_map

RING_INNER_PX = 2          # guard band next to the mask that is not used as "context"
RING_OUTER_PX = 10         # context ring reaches this far from the mask
DEV_LEVELS = 11            # same deviation level as fill_repair uses for a visible blob
DARK_LEVEL = 20            # luma difference to the context that counts as a clear dark/bright blob
RING_FLAT_LEVELS = 15
# Context type of a region, from how flat its surrounding ring is (data: flat 1.6 % unusable,
# structured 37 % unusable for Box-only on 192 labelled regions).
CONTEXT_FLAT_MIN = 0.90
CONTEXT_MIXED_MIN = 0.50
# Provisional split of whole assets into flat UI sprites vs detailed images (check the distribution!).
ASSET_FLAT_FRAC_MIN = 0.75
IMAGE_FLAT_RANGE = 12


def _luma(rgb: np.ndarray) -> np.ndarray:
    f = rgb.astype(np.float32)
    return f[..., 0] * 0.299 + f[..., 1] * 0.587 + f[..., 2] * 0.114


def _largest_component(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    labels, count = ndi.label(mask, structure=np.ones((3, 3), dtype=bool))
    return int(np.bincount(labels.ravel())[1:].max()) if count else 0


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    k = 2 * radius + 1
    return cv2.dilate(mask.astype(np.uint8), np.ones((k, k), np.uint8)) > 0


def _gradient_magnitude(luma: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def _edges(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)
    return cv2.Canny(gray, 50, 150) > 0


def context_type(ring_flat_frac: float) -> str:
    if ring_flat_frac >= CONTEXT_FLAT_MIN:
        return "flat"
    if ring_flat_frac >= CONTEXT_MIXED_MIN:
        return "mixed"
    return "structured"


def image_features(rgb: np.ndarray, alpha: np.ndarray) -> dict[str, float]:
    """Asset-level descriptors: how flat / detailed is the whole sprite (opaque pixels only)."""
    opaque = alpha > 16
    n = int(opaque.sum())
    feats: dict[str, float] = {"image_opaque_frac": n / float(opaque.size)}
    if n == 0:
        feats.update({"image_flat_frac": 1.0, "image_edge_density": 0.0, "image_unique_colors": 0.0})
        feats["asset_type"] = "flat_ui"
        return feats
    k = np.ones((5, 5), np.uint8)
    rng = (cv2.dilate(rgb, k).astype(np.int16) - cv2.erode(rgb, k).astype(np.int16)).max(axis=2)
    feats["image_flat_frac"] = float((rng[opaque] <= IMAGE_FLAT_RANGE).mean())
    feats["image_edge_density"] = float(_edges(rgb)[opaque].mean())
    packed = (rgb[opaque][:, 0].astype(np.uint32) << 16) | (rgb[opaque][:, 1].astype(np.uint32) << 8) | rgb[opaque][:, 2]
    feats["image_unique_colors"] = float(np.unique(packed).size)
    feats["asset_type"] = "flat_ui" if feats["image_flat_frac"] >= ASSET_FLAT_FRAC_MIN else "detailed"
    return feats


def region_features(
    *,
    src_rgb: np.ndarray, src_alpha: np.ndarray,
    res_rgb: np.ndarray, res_alpha: np.ndarray,
    mask: np.ndarray, other_mask: np.ndarray | None = None,
) -> dict[str, float]:
    mask = np.asarray(mask, dtype=bool)
    other = np.zeros_like(mask) if other_mask is None else np.asarray(other_mask, dtype=bool)
    mask_px = int(mask.sum())
    feats: dict[str, float] = {"mask_px": float(mask_px), "mask_frac_window": mask_px / float(mask.size)}
    if mask_px == 0:
        return feats

    ring = _dilate(mask, RING_OUTER_PX) & ~_dilate(mask, RING_INNER_PX) & ~other & ~mask
    ring_px = int(ring.sum())
    feats["ring_px"] = float(ring_px)

    res_luma = _luma(res_rgb)
    src_luma = _luma(src_rgb)
    have_ring = ring_px >= 20

    # ---- context (what does the surrounding look like?) ------------------------------------
    if have_ring:
        ring_rgb = res_rgb[ring].astype(np.float32)
        ring_med = np.median(ring_rgb, axis=0)
        feats["ring_luma_std"] = float(res_luma[ring].std())
        feats["ring_flat_frac"] = float((np.abs(ring_rgb - ring_med).max(axis=1) <= RING_FLAT_LEVELS).mean())
        ring_luma_med = float(np.median(res_luma[ring]))
    else:
        ring_med = np.median(res_rgb[mask].astype(np.float32), axis=0)
        feats["ring_luma_std"] = 0.0
        feats["ring_flat_frac"] = 1.0
        ring_luma_med = float(np.median(res_luma[mask]))

    # ---- text residue ------------------------------------------------------------------------
    src_ink = ink_strength_map(np.ascontiguousarray(src_rgb)) > INK_STRENGTH_THRESHOLD
    res_ink = ink_strength_map(np.ascontiguousarray(res_rgb)) > INK_STRENGTH_THRESHOLD
    src_ink_frac = float(src_ink[mask].mean())
    res_ink_frac = float(res_ink[mask].mean())
    ring_ink_frac = float(res_ink[ring].mean()) if have_ring else 0.0
    feats["src_ink_frac"] = src_ink_frac
    feats["res_ink_frac"] = res_ink_frac
    feats["ring_ink_frac"] = ring_ink_frac
    feats["res_ink_excess"] = max(0.0, res_ink_frac - ring_ink_frac)
    feats["ink_removed_ratio"] = 1.0 - (res_ink_frac / src_ink_frac) if src_ink_frac > 0 else 1.0
    feats["res_ink_max_cc_px"] = float(_largest_component(res_ink & mask))
    unchanged_ink = src_ink & mask & (np.abs(res_rgb.astype(np.int16) - src_rgb.astype(np.int16)).max(axis=2) <= 6)
    feats["unchanged_ink_px"] = float(unchanged_ink.sum())
    feats["unchanged_ink_max_cc_px"] = float(_largest_component(unchanged_ink))

    # ---- fill artefacts (dark / bright blobs, rings, smudges) --------------------------------
    f = np.ascontiguousarray(res_rgb)
    ref = cv2.medianBlur(f, 25)
    dev = np.abs(f.astype(np.int16) - ref.astype(np.int16)).max(axis=2)
    dev_mask = (dev > DEV_LEVELS) & mask
    feats["dev_frac"] = float(dev_mask.sum()) / mask_px
    feats["dev_max_cc_px"] = float(_largest_component(dev_mask))
    diff_to_ring = ring_luma_med - res_luma
    dark = (diff_to_ring > DARK_LEVEL) & mask
    bright = (-diff_to_ring > DARK_LEVEL) & mask
    feats["dark_p99"] = float(np.percentile(diff_to_ring[mask], 99))
    feats["bright_p99"] = float(np.percentile(-diff_to_ring[mask], 99))
    feats["dark_cc_px"] = float(_largest_component(dark))
    feats["bright_cc_px"] = float(_largest_component(bright))
    feats["mean_shift"] = float(np.abs(res_rgb[mask].astype(np.float32).mean(axis=0) - ring_med).max())
    std_in = float(res_luma[mask].std())
    feats["std_in_luma"] = std_in
    feats["std_ratio"] = std_in / (feats["ring_luma_std"] + 1.0)

    # ---- seam at the mask border -------------------------------------------------------------
    inner = mask & ~(cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    outer = _dilate(mask, 1) & ~mask & ~other
    if inner.any() and outer.any():
        feats["seam_color_diff"] = float(np.abs(
            res_rgb[inner].astype(np.float32).mean(axis=0) - res_rgb[outer].astype(np.float32).mean(axis=0)
        ).max())
        grad = _gradient_magnitude(res_luma)
        band = inner | outer
        feats["seam_grad_ratio"] = float(grad[band].mean() / (grad[ring].mean() + 1.0)) if have_ring else 0.0
    else:
        feats["seam_color_diff"] = 0.0
        feats["seam_grad_ratio"] = 0.0

    # ---- structure that continues across the mask border (outlines, frames, patterns) -------
    e_src = _edges(src_rgb)
    e_res = _dilate(_edges(res_rgb), 2)
    labels, count = ndi.label(e_src, structure=np.ones((3, 3), dtype=bool))
    crossing_inside = np.zeros_like(mask)
    if count:
        in_counts = np.bincount(labels[mask], minlength=count + 1)
        out_counts = np.bincount(labels[ring], minlength=count + 1) if have_ring else np.zeros(count + 1, int)
        crossing = (in_counts > 0) & (out_counts >= 6)
        crossing[0] = False
        if crossing.any():
            crossing_inside = np.isin(labels, np.nonzero(crossing)[0]) & mask
    crossing_px = int(crossing_inside.sum())
    feats["crossing_edge_px"] = float(crossing_px)
    feats["crossing_kept_ratio"] = float((crossing_inside & e_res).sum()) / crossing_px if crossing_px else 1.0
    feats["res_edge_density_in"] = float(_edges(res_rgb)[mask].mean())
    feats["ring_edge_density"] = float(_edges(res_rgb)[ring].mean()) if have_ring else 0.0

    # ---- alpha ---------------------------------------------------------------------------------
    feats["alpha_changed_frac"] = float((res_alpha[mask] != src_alpha[mask]).mean())
    feats["src_transparent_frac"] = float((src_alpha <= 16).mean())
    return feats


def lama_integrity(
    src_rgb: np.ndarray, pred_rgb: np.ndarray | None, mask: np.ndarray,
) -> dict[str, float]:
    """Whole-image sanity check of the raw inpainting output.

    Outside the mask the output must be (nearly) the original. A shifted, rolled, resized or
    otherwise misaligned output shows up here as a large difference far from the mask.
    """
    if pred_rgb is None or pred_rgb.shape != src_rgb.shape:
        return {"lama_size_mismatch": 1.0, "lama_outside_mean": 255.0, "lama_outside_frac_gt8": 1.0}
    far = ~_dilate(mask, 6)
    if not far.any():
        return {"lama_size_mismatch": 0.0, "lama_outside_mean": 0.0, "lama_outside_frac_gt8": 0.0}
    diff = np.abs(pred_rgb.astype(np.int16) - src_rgb.astype(np.int16)).max(axis=2)[far]
    return {
        "lama_size_mismatch": 0.0,
        "lama_outside_mean": float(diff.mean()),
        "lama_outside_frac_gt8": float((diff > 8).mean()),
    }
