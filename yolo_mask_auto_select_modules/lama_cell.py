"""LaMa on the sprite cell instead of the whole image (methods ``BOX_CELL``, ``SMART_INK_CELL``).

In a sprite sheet the whole image shows LaMa a dozen near-identical cells as "context"; the cell of one
text box is the context that matters. Here every region gets its own job: the crop of its window (the same
window stage 1 uses: the sprite cell if a grid is detected, otherwise bbox +- margin), the mask inside that
window and the alpha as a second image (the pipeline inpaints alpha through the same mask). The result is
pasted back into the region's territory only, so every pixel still comes from exactly one region.

The functions are pure numpy (no IOPaint) so that they can be tested without LaMa.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Rect = tuple[int, int, int, int]            # x1, y1, x2, y2 inclusive


@dataclass
class CellJob:
    region: int
    win: Rect
    rgb: np.ndarray            # h, w, 3 uint8: crop of the image
    alpha_rgb: np.ndarray      # h, w, 3 uint8: crop of the alpha, as a grey RGB image
    mask: np.ndarray           # h, w uint8 (255 = inpaint): ALL masks inside the window
    own: np.ndarray            # h, w bool: pixels of this region's mask inside its territory (what gets pasted)


def _sl(win: Rect) -> tuple[slice, slice]:
    x1, y1, x2, y2 = win
    return slice(y1, y2 + 1), slice(x1, x2 + 1)


def make_jobs(rgb: np.ndarray, alpha: np.ndarray, mask: np.ndarray, territory: np.ndarray,
              regions: list[int], windows: list[Rect]) -> list[CellJob]:
    """One job per region that has mask pixels in its territory.

    ``windows[k - 1]`` is the window of region k. The crop mask contains the masks of the NEIGHBOUR regions
    inside the window as well: their text must not act as context (LaMa could copy it into this region);
    their fill is simply not pasted."""
    mask = np.asarray(mask, dtype=bool)
    jobs: list[CellJob] = []
    for k in regions:
        win = tuple(int(v) for v in windows[k - 1])
        sl = _sl(win)
        own = mask[sl] & (territory[sl] == k)
        if not own.any():
            continue
        crop_mask = mask[sl]
        jobs.append(CellJob(
            region=k, win=win,                                   # type: ignore[arg-type]
            rgb=np.ascontiguousarray(rgb[sl]),
            alpha_rgb=np.ascontiguousarray(np.repeat(alpha[sl][..., None], 3, axis=-1)),
            mask=(crop_mask.astype(np.uint8) * 255), own=own,
        ))
    return jobs


def paste_region(win: Rect, own: np.ndarray, pred_rgb: np.ndarray | None, pred_alpha: np.ndarray | None,
                 out_rgb: np.ndarray, out_alpha: np.ndarray, stamped: np.ndarray) -> bool:
    """Paste the prediction for one window into the full-size result (in place), only where ``own`` is set.
    False if LaMa gave nothing usable (missing file or a crop of the wrong size)."""
    h, w = own.shape
    if pred_rgb is None or pred_alpha is None or pred_rgb.shape[:2] != (h, w) or pred_alpha.shape[:2] != (h, w):
        return False
    sl = _sl(win)
    view_rgb, view_alpha, view_st = out_rgb[sl], out_alpha[sl], stamped[sl]
    view_rgb[own] = pred_rgb[own]
    view_alpha[own] = pred_alpha[own]
    view_st[own] = True
    return True


def paste(job: CellJob, pred_rgb: np.ndarray | None, pred_alpha: np.ndarray | None,
          out_rgb: np.ndarray, out_alpha: np.ndarray, stamped: np.ndarray) -> bool:
    return paste_region(job.win, job.own, pred_rgb, pred_alpha, out_rgb, out_alpha, stamped)


def merge_integrity(parts: list[dict[str, float]]) -> dict[str, float]:
    """Worst value of every integrity metric over all jobs of an image (all of them are 'smaller is better')."""
    out: dict[str, float] = {}
    for p in parts:
        for key, value in p.items():
            out[key] = max(out.get(key, float("-inf")), float(value))
    return out
