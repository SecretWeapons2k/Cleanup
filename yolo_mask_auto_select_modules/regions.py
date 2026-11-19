"""Per-region (per-textbox) candidate selection.

Until now one candidate (one mask + its LaMa fill) won for the WHOLE image, so an image
with two text boxes that need different masks ended with a compromise. Here the anchors
are grouped into spatial regions (clusters of nearby boxes); every region gets its own
scoring and its own winner, and the final result is assembled from the per-region winners.

Design rules
------------
* ``score_candidate`` / ``select_candidate`` are reused UNCHANGED. A region is scored by
  handing them a *restricted* view of the image: everything outside the region's territory
  is the untouched original, all masks are cut to the territory, and the arrays are cropped
  to the region plus a safety margin (all metrics are local, so the crop is exact).
* A *territory* is the part of the image that is closer to this region's boxes than to any
  other region's boxes. Territories partition the image, so every pixel of the final result
  comes from exactly one region winner.
* With a single region the legacy whole-image path is used (see pipeline); this module is
  only entered for images with two or more regions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .scoring import (
    ink_strength_map, outline_leak_metrics, score_candidate, select_candidate,
)

CandId = int | str
Composite = tuple[np.ndarray, np.ndarray]  # (rgb uint8 HxWx3, alpha uint8 HxW)


@dataclass
class RegionPlan:
    count: int
    core_labels: np.ndarray          # int32 HxW, 0 = not a box pixel, k = box pixel of region k
    territory: np.ndarray            # int32 HxW, 1..count (every pixel belongs to one region)
    bboxes: list[tuple[int, int, int, int]] = field(default_factory=list)  # x1,y1,x2,y2 inclusive


@dataclass
class RegionResult:
    region: int
    bbox: tuple[int, int, int, int]            # core box bbox (inclusive)
    window: tuple[int, int, int, int]          # crop used for scoring/previews (inclusive x1,y1,x2,y2)
    winner_id: CandId
    winner_metrics: dict[str, float]
    best_text_score: float
    text_tolerance: float
    scored: list[tuple[float, CandId, dict[str, float]]]   # (score, candidate id, metrics) for all candidates


def plan_regions(box_mask: np.ndarray, merge_px: int) -> RegionPlan:
    """Group box pixels into regions: boxes closer than ``2*merge_px`` px share a region."""
    box_mask = np.asarray(box_mask, dtype=bool)
    h, w = box_mask.shape
    if not box_mask.any():
        return RegionPlan(0, np.zeros((h, w), np.int32), np.ones((h, w), np.int32), [])
    src = box_mask.astype(np.uint8)
    if merge_px > 0:
        k = 2 * int(merge_px) + 1
        src = cv2.dilate(src, np.ones((k, k), np.uint8))
    count_plus_one, labels = cv2.connectedComponents(src, connectivity=8)
    count = count_plus_one - 1
    core = np.where(box_mask, labels, 0).astype(np.int32)

    if count == 1:
        territory = np.ones((h, w), np.int32)
    else:
        best = np.full((h, w), np.inf, dtype=np.float32)
        territory = np.zeros((h, w), np.int32)
        for k in range(1, count + 1):
            dist = cv2.distanceTransform((core != k).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_5)
            closer = dist < best
            best[closer] = dist[closer]
            territory[closer] = k

    bboxes: list[tuple[int, int, int, int]] = []
    for k in range(1, count + 1):
        ys, xs = np.nonzero(core == k)
        bboxes.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))
    return RegionPlan(count, core, territory, bboxes)


def region_window(bbox: tuple[int, int, int, int], margin: int, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (max(0, x1 - margin), max(0, y1 - margin),
            min(width - 1, x2 + margin), min(height - 1, y2 + margin))


def select_per_region(
    *,
    plan: RegionPlan,
    original_rgb: np.ndarray,
    original_alpha: np.ndarray,
    composites: dict[CandId, Composite],
    candidate_masks: dict[CandId, np.ndarray],
    groups: dict[CandId, CandId],
    extra_metrics: dict[CandId, dict[str, float]],
    box_mask: np.ndarray,
    body_box_mask: np.ndarray,
    yolo_reference_mask: np.ndarray,
    scoring_envelope: np.ndarray,
    margin_px: int,
) -> list[RegionResult]:
    """Score every candidate inside every region and select one winner per region.

    ``extra_metrics[cid]`` (alpha post-processing / fill-repair counters) is merged into
    the metrics of each region exactly as the whole-image path does.
    """
    height, width = original_rgb.shape[:2]
    candidate_ids = list(candidate_masks)
    results: list[RegionResult] = []

    for k in range(1, plan.count + 1):
        bbox = plan.bboxes[k - 1]
        x1, y1, x2, y2 = region_window(bbox, margin_px, width, height)
        sl = (slice(y1, y2 + 1), slice(x1, x2 + 1))
        terr = plan.territory[sl] == k
        rgb0 = original_rgb[sl]
        alpha0 = original_alpha[sl]
        box_c = np.asarray(box_mask)[sl] & terr
        body_c = np.asarray(body_box_mask)[sl] & terr
        ref_c = np.asarray(yolo_reference_mask)[sl] & terr
        env_c = np.asarray(scoring_envelope)[sl] & terr
        source_detail = ink_strength_map(rgb0)

        scored: list[tuple[float, CandId, np.ndarray, dict[str, float]]] = []
        metrics_by_rep: dict[CandId, dict[str, float]] = {}
        for cid in candidate_ids:
            rep = groups.get(cid, cid)
            mask_c = np.asarray(candidate_masks[cid])[sl] & terr
            if rep != cid:
                metrics = dict(metrics_by_rep[rep])
                metrics["mask_is_duplicate"] = 1.0
            else:
                f_rgb_full, f_alpha_full = composites[cid]
                # Region view: candidate result inside the territory, untouched original elsewhere.
                c_rgb = np.where(terr[..., None], f_rgb_full[sl], rgb0)
                c_alpha = np.where(terr, f_alpha_full[sl], alpha0)
                leak = outline_leak_metrics(source_detail, alpha0, body_c, mask_c)
                metrics = score_candidate(
                    rgb0, alpha0, c_rgb, c_alpha, mask_c, env_c, box_c, body_c, ref_c,
                    outline_leak_precomputed=leak,
                )
                metrics.update(extra_metrics.get(cid, {}))
                metrics["mask_is_duplicate"] = 0.0
                metrics_by_rep[cid] = dict(metrics)
            scored.append((metrics["score"], cid, mask_c, metrics))

        winner, best_text, tolerance = select_candidate(scored)
        results.append(RegionResult(
            region=k, bbox=bbox, window=(x1, y1, x2, y2),
            winner_id=winner[1], winner_metrics=winner[3],
            best_text_score=best_text, text_tolerance=tolerance,
            scored=[(s, cid, m) for s, cid, _mask, m in scored],
        ))
    return results


def merge_region_winners(
    plan: RegionPlan,
    original_rgb: np.ndarray,
    original_alpha: np.ndarray,
    composites: dict[CandId, Composite],
    candidate_masks: dict[CandId, np.ndarray],
    winners: dict[int, CandId],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble the final RGB/alpha/mask: each territory is taken from its region's winner."""
    rgb = original_rgb.copy()
    alpha = original_alpha.copy()
    mask = np.zeros(original_alpha.shape, dtype=bool)
    for k, cid in winners.items():
        sel = plan.territory == k
        rgb[sel] = composites[cid][0][sel]
        alpha[sel] = composites[cid][1][sel]
        mask |= np.asarray(candidate_masks[cid], dtype=bool) & sel
    return rgb, alpha, mask


def winners_label(results: list[RegionResult]) -> str:
    """Compact text such as ``r1=D10;r2=SMART_INK`` for reports."""
    return ";".join(f"r{r.region}={r.winner_id}" for r in results)


def region_rows(name: str, results: list[RegionResult]) -> list[dict[str, Any]]:
    """One report row per region (winner only) for region_report.csv."""
    rows: list[dict[str, Any]] = []
    for r in results:
        m = r.winner_metrics
        rows.append({
            "image": name, "region": r.region,
            "bbox_x1": r.bbox[0], "bbox_y1": r.bbox[1], "bbox_x2": r.bbox[2], "bbox_y2": r.bbox[3],
            "selected_candidate": str(r.winner_id),
            "selected_score": f"{m['score']:.6f}",
            "selected_text_quality_score": f"{m['text_quality_score']:.6f}",
            "selected_structure_score": f"{m['structure_score']:.6f}",
            "best_text_quality_score": f"{r.best_text_score:.6f}",
            "text_quality_tolerance": f"{r.text_tolerance:.6f}",
            "outline_leak_score": f"{m['outline_leak_score']:.6f}",
            "residual_inside_box_ratio": f"{m['residual_inside_box_ratio']:.6f}",
            "box_only_rescue_applied": int(m.get("box_only_rescue_applied", 0.0)),
            "residual_spot_tiebreak_applied": int(m.get("residual_spot_tiebreak_applied", 0.0)),
        })
    return rows
