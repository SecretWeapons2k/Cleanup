"""Post-fill sanity check for inpainting results (LaMa).

Finds clusters inside the candidate mask where the LaMa fill disagrees strongly
with a deterministic fill (cv2 Navier-Stokes) AND where the local surrounding is
flat. Only such clusters are replaced by the deterministic fill. Clusters next to
frames/outlines/textures are left untouched (LaMa is better there).
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi


def repair_flat_fill_artifacts(
    original_rgb: np.ndarray,
    filled_rgb: np.ndarray,
    mask: np.ndarray,
    protected: np.ndarray | None = None,
    *,
    min_cluster_px: int = 25,
    disagree_levels: int = 45,
    flat_ring_std: float = 20.0,
    ring_inner: int = 2,
    ring_outer: int = 6,
    grow_px: int = 2,
    inpaint_radius: int = 4,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return (repaired_rgb, meta). Inputs are uint8 HxWx3 and a bool HxW mask."""
    mask = mask.astype(bool)
    meta = {"fill_repair_clusters_found": 0.0, "fill_repair_clusters_repaired": 0.0,
            "fill_repair_pixels_replaced": 0.0}
    if not mask.any():
        return filled_rgb, meta

    ns = cv2.inpaint(np.ascontiguousarray(original_rgb), mask.astype(np.uint8) * 255,
                     inpaint_radius, cv2.INPAINT_NS)
    diff = np.abs(filled_rgb.astype(np.int16) - ns.astype(np.int16)).max(axis=2)
    seeds = ndi.binary_opening((diff > disagree_levels) & mask, iterations=1)
    labels, count = ndi.label(seeds)

    out = filled_rgb.copy()
    for idx in range(1, count + 1):
        cluster = labels == idx
        if int(cluster.sum()) < min_cluster_px:
            continue
        meta["fill_repair_clusters_found"] += 1
        ring = (ndi.binary_dilation(cluster, iterations=ring_outer)
                & ~ndi.binary_dilation(cluster, iterations=ring_inner))
        ring_px = filled_rgb[ring].astype(np.float32)
        if ring_px.shape[0] < 20 or float(ring_px.std(axis=0).max()) > flat_ring_std:
            continue  # not flat around the cluster: leave LaMa result alone
        med = np.median(ring_px, axis=0)
        lama_dev = np.abs(filled_rgb[cluster].astype(np.float32) - med).max(axis=1).mean()
        ns_dev = np.abs(ns[cluster].astype(np.float32) - med).max(axis=1).mean()
        if ns_dev * 2.0 > lama_dev:
            continue  # deterministic fill is not clearly closer to the surroundings
        region = ndi.binary_dilation(cluster, iterations=grow_px) & mask
        if protected is not None:
            region &= ~protected.astype(bool)
        out[region] = ns[region]
        meta["fill_repair_clusters_repaired"] += 1
        meta["fill_repair_pixels_replaced"] += float(region.sum())
    return out, meta
