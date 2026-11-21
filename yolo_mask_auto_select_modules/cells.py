"""Cell grid detection for sprite sheets and per-region windows that follow the cells.

Why: ``regions.region_window`` crops "box bbox + 48 px". The anchor boxes are text boxes and
smaller than the sprite cell, so the window cuts through buttons and neighbouring cells.
Here the image is split into equal cells and a region's window is the union of the cells it
touches. Regions themselves (``plan_regions``) are NOT changed, so region numbers and all
existing labels stay valid.

Detection (pixels only, no anchors needed)
------------------------------------------
For each axis and each cell count n (largest first) the n-1 interior boundaries of an equal
split must each be supported by evidence within +-2 px:
  * gap : the line is (almost) fully transparent  (sprites separated by transparent gaps), or
  * edge: a strong colour/alpha jump across the line (opaque tiles side by side).
The largest n whose weakest boundary is still supported wins. Larger n than the truth fails
because the extra boundaries run through cell interiors; smaller n is a subset of true
boundaries and is only used when nothing larger passes.

Not handled: unequal cell sizes, sheets with an outer margin, rotated sheets. In those cases
the detector returns ``None`` and the caller keeps the old margin window.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

USE_CELL_WINDOWS = True    # False = always the old bbox +- margin window
EDGE_RATIO_MIN = 3.0       # boundary edge energy / median edge energy needed for "edge" evidence
GAP_FRACTION_MIN = 0.90    # fraction of transparent pixels on the line needed for "gap" evidence
MIN_CELL_PX = 14           # smallest allowed cell side
MAX_CELLS_PER_AXIS = 16
SNAP_PX = 2                # boundary may move this far from the equal split to hit its evidence
MIN_CELL_OPAQUE_FRAC = 0.30  # every cell must contain content (guards mostly transparent images)
ALPHA_GAP_LEVEL = 15
# Weakest boundary must reach this support (1.0 = evidence threshold). With a single boundary
# (n=2) one stray edge in a picture is enough to fool the detector, so it must be clearly strong.
SUPPORT_MIN = 1.5
# A single row (n x 1) is split wrongly far too often: gaps between glyphs look like cell
# boundaries (20 of 140 single buttons cut out of the test sheets). Sheets with button states
# have several rows, so a grid needs at least this many rows; single rows keep the margin window.
MIN_ROWS = 2
# Single-row strips (e.g. 4 text bars next to each other) are allowed again, but only with very
# clear boundaries: wrong splits of single buttons reached support 4.05, the 5 test strips 7.5-9.7.
SINGLE_ROW_SUPPORT_MIN = 5.0
SINGLE_ROW_MIN_COLS = 2
# Among the cell counts that pass, a finer grid than the truth only passes by catching random
# edges inside the cells, so its boundaries are weaker on average. Only counts whose mean support
# (per boundary capped at SUPPORT_CAP) reaches this share of the best mean stay in the race.
MEAN_SUPPORT_KEEP = 0.8
SUPPORT_CAP = 10.0
SUPPORT_MIN_TWO_CELLS = 3.0
MIN_REGION_CELL_PX = 8     # a cell counts for a region only with this many box pixels ...
MIN_REGION_CELL_FRAC = 0.02  # ... and this fraction of the region's box pixels

Rect = tuple[int, int, int, int]  # x1, y1, x2, y2 inclusive


@dataclass
class CellGrid:
    col_bounds: list[int]            # first pixel of each column + width as last element
    row_bounds: list[int]            # first pixel of each row + height as last element
    col_support: float               # weakest boundary support (accepted from 1.5; 3.0 for two cells)
    row_support: float
    width: int
    height: int
    col_index: np.ndarray = field(repr=False, default=None)  # per x: column number
    row_index: np.ndarray = field(repr=False, default=None)  # per y: row number

    @property
    def n_cols(self) -> int:
        return len(self.col_bounds) - 1

    @property
    def n_rows(self) -> int:
        return len(self.row_bounds) - 1

    def cell_rect(self, row: int, col: int) -> Rect:
        return (self.col_bounds[col], self.row_bounds[row],
                self.col_bounds[col + 1] - 1, self.row_bounds[row + 1] - 1)

    def cell_at(self, x: int, y: int) -> tuple[int, int]:
        return int(self.row_index[y]), int(self.col_index[x])

    def summary(self) -> str:
        return f"{self.n_cols}x{self.n_rows}"


@dataclass
class RegionWindow:
    rect: Rect
    source: str                      # "cell" or "margin"
    cells: list[tuple[int, int]] = field(default_factory=list)  # (row, col) touched


def _bounds_from_n(length: int, n: int) -> list[int]:
    return [int(round(i * length / n)) for i in range(n + 1)]


def _axis_signals(arr: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """Edge energy between neighbouring lines (len L-1) and transparent fraction per line (len L)."""
    diff = np.abs(np.diff(arr, axis=axis)).max(axis=2)
    other = 1 - axis
    edge = diff.sum(axis=other)
    gap = (arr[..., 3] < ALPHA_GAP_LEVEL).mean(axis=other)
    return edge, gap


def _boundary_score(edge: np.ndarray, gap: np.ndarray, med: float, b: int) -> float:
    """Evidence for a cell boundary at b (first pixel of the next cell)."""
    length = len(gap)
    g = gap[max(0, b - 1):min(length, b + 1)].max() / GAP_FRACTION_MIN
    e = edge[max(0, b - 3):min(len(edge), b + 1)].max() / (med * EDGE_RATIO_MIN) if len(edge) else 0.0
    return float(max(g, e))


def _edge_baseline(edge: np.ndarray) -> float:
    """Typical edge energy of a line; floored so piecewise-constant images cannot give ratios of 1e9."""
    return max(float(np.median(edge)), 0.1 * float(np.mean(edge)), 1.0)


def _fit_axis(edge: np.ndarray, gap: np.ndarray, length: int, n: int) -> tuple[list[int], float]:
    med = _edge_baseline(edge)
    bounds = [0]
    worst = np.inf
    for i in range(1, n):
        expect = int(round(i * length / n))
        cands = range(max(1, expect - SNAP_PX), min(length - 1, expect + SNAP_PX) + 1)
        best_b, best_s = expect, -1.0
        for b in cands:
            s = _boundary_score(edge, gap, med, b)
            # higher evidence wins; at equal evidence the position nearest to the equal split
            if s > best_s + 1e-9 or (abs(s - best_s) <= 1e-9 and abs(b - expect) < abs(best_b - expect)):
                best_b, best_s = b, s
        bounds.append(best_b)
        worst = min(worst, best_s)
    bounds.append(length)
    return bounds, float(worst if n > 1 else np.inf)


def _cells_have_content(alpha: np.ndarray, col_b: list[int], row_b: list[int]) -> bool:
    opaque = alpha >= ALPHA_GAP_LEVEL
    for r in range(len(row_b) - 1):
        for c in range(len(col_b) - 1):
            cell = opaque[row_b[r]:row_b[r + 1], col_b[c]:col_b[c + 1]]
            if cell.size == 0 or cell.mean() < MIN_CELL_OPAQUE_FRAC:
                return False
    return True


def detect_cell_grid(rgb: np.ndarray, alpha: np.ndarray) -> CellGrid | None:
    """Return the cell grid of a sprite sheet, or None when no regular grid (>1 cell) is found."""
    h, w = alpha.shape
    arr = np.concatenate([rgb.astype(np.float32), alpha[..., None].astype(np.float32)], axis=2)
    ex, gx = _axis_signals(arr, axis=1)   # across columns
    ey, gy = _axis_signals(arr, axis=0)   # across rows

    def best_axis(edge, gap, length):
        found = []
        med = _edge_baseline(edge)
        for n in range(2, MAX_CELLS_PER_AXIS + 1):
            if length / n < MIN_CELL_PX:
                break
            bounds, worst = _fit_axis(edge, gap, length, n)
            if worst >= (SUPPORT_MIN_TWO_CELLS if n == 2 else SUPPORT_MIN):
                scores = [min(_boundary_score(edge, gap, med, b), SUPPORT_CAP) for b in bounds[1:-1]]
                found.append((n, bounds, worst, float(np.mean(scores))))
        if not found:
            return []
        best_mean = max(f[3] for f in found)
        return [(n, b, wst) for n, b, wst, m in found if m >= MEAN_SUPPORT_KEEP * best_mean]

    cols = best_axis(ex, gx, w)
    rows = best_axis(ey, gy, h)
    col_opts = [(1, [0, w], np.inf)] + cols
    row_opts = [(1, [0, h], np.inf)] + rows
    # Largest combination (>= MIN_ROWS rows) whose cells all contain content (guards mostly
    # transparent images).
    for n_c, col_b, col_s in reversed(col_opts):
        for n_r, row_b, row_s in reversed(row_opts):
            if n_r < MIN_ROWS:
                if n_r != 1 or n_c < SINGLE_ROW_MIN_COLS or col_s < SINGLE_ROW_SUPPORT_MIN:
                    continue
            if _cells_have_content(alpha, col_b, row_b):
                col_index = np.zeros(w, np.int32)
                row_index = np.zeros(h, np.int32)
                for i in range(len(col_b) - 1):
                    col_index[col_b[i]:col_b[i + 1]] = i
                for i in range(len(row_b) - 1):
                    row_index[row_b[i]:row_b[i + 1]] = i
                return CellGrid(col_b, row_b, float(col_s), float(row_s), w, h, col_index, row_index)
    return None


def region_windows(
    core_labels: np.ndarray,
    bboxes: list[Rect],
    grid: CellGrid | None,
    margin: int,
) -> list[RegionWindow]:
    """Window per region: union of the cells its box pixels touch (plus the box bbox itself).

    Without a grid, or for a region whose pixels hit no cell, the old window is used:
    bbox +- margin, clipped to the image.
    """
    h, w = core_labels.shape
    out: list[RegionWindow] = []
    for k, (bx1, by1, bx2, by2) in enumerate(bboxes, start=1):
        old = (max(0, bx1 - margin), max(0, by1 - margin), min(w - 1, bx2 + margin), min(h - 1, by2 + margin))
        if grid is None:
            out.append(RegionWindow(old, "margin"))
            continue
        ys, xs = np.nonzero(core_labels[by1:by2 + 1, bx1:bx2 + 1] == k)
        ys, xs = ys + by1, xs + bx1
        cell_ids = grid.row_index[ys].astype(np.int64) * grid.n_cols + grid.col_index[xs]
        ids, counts = np.unique(cell_ids, return_counts=True)
        keep = [(int(i // grid.n_cols), int(i % grid.n_cols)) for i, c in zip(ids, counts)
                if c >= MIN_REGION_CELL_PX and c >= MIN_REGION_CELL_FRAC * len(ys)]
        if not keep:
            out.append(RegionWindow(old, "margin"))
            continue
        rects = [grid.cell_rect(r, c) for r, c in keep]
        x1 = min(min(r[0] for r in rects), bx1)
        y1 = min(min(r[1] for r in rects), by1)
        x2 = max(max(r[2] for r in rects), bx2)
        y2 = max(max(r[3] for r in rects), by2)
        out.append(RegionWindow((x1, y1, x2, y2), "cell", sorted(keep)))
    return out


def draw_grid(rgb: np.ndarray, grid: CellGrid | None) -> np.ndarray:
    """Debug picture: the image with cell boundaries drawn in magenta."""
    img = rgb.copy()
    if grid is None:
        return img
    for x in grid.col_bounds[1:-1]:
        img[:, max(0, x - 1):x + 1] = (255, 0, 255)
    for y in grid.row_bounds[1:-1]:
        img[max(0, y - 1):y + 1, :] = (255, 0, 255)
    return img
