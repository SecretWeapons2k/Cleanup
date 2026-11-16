"""Post-fill sanity check for inpainting results (LaMa).

LaMa sometimes invents small dark dots/rings/smudges inside a mask on flat
backgrounds (e.g. button sprites). This module finds such blobs *inside the
mask* by comparing the fill with a robust local background (large median
filter) and, only if the surroundings of a blob are flat, replaces the blob
with that smoothed background.

Why not compare against cv2 Navier-Stokes (previous version): NS bleeds dark
glyph halos from just outside the mask into the mask, so it often agrees with
the artefact (ring/smudge on f002_004) and the blob is never detected.

Blobs next to frames/outlines/textures are left alone: there the ring around
the blob is not flat or contains too many pixels far from its median, or the
blob continues across the mask border into the unmasked image.
Only RGB inside the mask is changed; protected zones are never touched.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi


def repair_flat_fill_artifacts(
    original_rgb: np.ndarray,          # kept for API compatibility (not needed any more)
    filled_rgb: np.ndarray,
    mask: np.ndarray,
    protected: np.ndarray | None = None,
    *,
    median_ksize: int = 25,            # odd; blob must cover < 50 % of the window
    dev_levels: int = 11,              # min. max-channel deviation from local background
    min_cluster_px: int = 8,
    ring_inner: int = 3,
    ring_outer: int = 8,
    ring_keep_levels: int = 15,        # ring pixels within this distance of ring median are "background"
    ring_min_keep_frac: float = 0.6,   # else: frame/outline/texture nearby -> skip
    flat_ring_std: float = 12.0,
    grow_px: int = 5,
    soften_sigma: float = 2.0,
    max_contact_aspect: float = 2.5,   # lines crossing the mask border are skipped
) -> tuple[np.ndarray, dict[str, float]]:
    """Return (repaired_rgb, meta). Inputs: uint8 HxWx3 and a bool HxW mask."""
    mask = np.asarray(mask).astype(bool)
    meta = {"fill_repair_clusters_found": 0.0, "fill_repair_clusters_repaired": 0.0,
            "fill_repair_pixels_replaced": 0.0}
    if not mask.any():
        return filled_rgb, meta

    f = np.ascontiguousarray(filled_rgb)
    ref = cv2.medianBlur(f, int(median_ksize) | 1)
    dev = np.abs(f.astype(np.int16) - ref.astype(np.int16)).max(axis=2)
    seeds = ndi.binary_opening((dev > dev_levels) & mask, iterations=1)
    labels, count = ndi.label(seeds)
    if count == 0:
        return filled_rgb, meta

    prot = None if protected is None else np.asarray(protected).astype(bool)
    soft_ref = cv2.GaussianBlur(ref.astype(np.float32), (0, 0), soften_sigma)
    out = filled_rgb.astype(np.float32).copy()
    changed = np.zeros(mask.shape, dtype=bool)

    for idx in range(1, count + 1):
        cluster = labels == idx
        if int(cluster.sum()) < min_cluster_px:
            continue
        meta["fill_repair_clusters_found"] += 1
        # Structure that continues outside the mask (frame/outline crossing the mask
        # border) is real image content, not a fill artefact: leave it alone.
        # Compact blobs (dots/rings) that merely touch the border are still repaired
        # (inside the mask only); elongated ones (lines) are skipped.
        outside_contact = ndi.binary_dilation(cluster, iterations=2) & ~mask
        if int((dev[outside_contact] > dev_levels).sum()) >= 3:
            ys, xs = np.nonzero(cluster)
            h, w = int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1)
            if max(h, w) / max(1, min(h, w)) >= max_contact_aspect:
                continue
        ring = (ndi.binary_dilation(cluster, iterations=ring_outer)
                & ~ndi.binary_dilation(cluster, iterations=ring_inner))
        ring_px = f[ring].astype(np.float32)
        if ring_px.shape[0] < 20:
            continue
        med = np.median(ring_px, axis=0)
        keep = np.abs(ring_px - med).max(axis=1) <= ring_keep_levels
        if keep.mean() < ring_min_keep_frac or keep.sum() < 10:
            continue  # frame/outline/texture around: leave LaMa result alone
        if float(ring_px[keep].std(axis=0).max()) > flat_ring_std:
            continue
        region = ndi.binary_dilation(cluster, iterations=grow_px) & mask
        if prot is not None:
            region &= ~prot
        if not region.any():
            continue
        w = cv2.GaussianBlur(region.astype(np.float32), (0, 0), 1.0)
        w = np.where(region, np.maximum(w, 0.0), w) * mask  # never leave the mask
        if prot is not None:
            w = w * (~prot)
        out = out * (1.0 - w[..., None]) + soft_ref * w[..., None]
        changed |= w > 0
        meta["fill_repair_clusters_repaired"] += 1

    repaired = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    repaired[~changed] = filled_rgb[~changed]
    meta["fill_repair_pixels_replaced"] = float((repaired != filled_rgb).any(axis=2).sum())
    return repaired, meta
