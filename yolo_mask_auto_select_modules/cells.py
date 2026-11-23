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

CELLS_VERSION = 3          # raise whenever detection changes: stage 2 refuses a stage-1 run made with another version
USE_CELL_WINDOWS = True    # False = always the old bbox +- margin window
# Which detection decides: "grid_detector" = the project's detector (grid_gate.py, incl. its exclusions
# X_ALPHA / X_NOBOX / X_SINGLE / X_CV_*); "legacy" = the two own detectors below (equal tiles, sprites).
CELL_DETECTOR = "grid_detector"
SPLIT_REGIONS_BY_CELLS = True  # False = keep regions that span several cells (old numbering for those)
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

# Sprites as connected alpha areas (sheets whose buttons have ragged outlines and no common gap)
SPRITE_MIN_PX = 400           # a connected area must have this many opaque pixels to count as a sprite
SPRITE_MIN_COUNT = 4
SPRITE_MAX_COUNT = 64
SPRITE_SIZE_TOLERANCE = 0.30  # width and height within +-30 % of the median
SPRITE_SIZE_SHARE = 0.80      # share of the sprites that must be of similar size
SPRITE_MARGIN_PX = 2
SPRITE_CENTRE_TOLERANCE = 0.04  # tie rule: a sprite centre may be this far (share of the tile width) off the tile centre

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
    source: str = "tiles"                  # "grid" / "lalpha" / "leiste" (grid_detector) or "tiles" (legacy)

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

    # interface shared with SpriteCells
    def cell_ids(self) -> np.ndarray:
        """int32 HxW: id of the cell every pixel belongs to (row * n_cols + col)."""
        return (self.row_index[:, None].astype(np.int32) * self.n_cols + self.col_index[None, :]).astype(np.int32)

    def rect_of(self, cell_id: int) -> Rect:
        return self.cell_rect(cell_id // self.n_cols, cell_id % self.n_cols)

    @property
    def n_cells(self) -> int:
        return self.n_cols * self.n_rows


@dataclass
class SpriteCells:
    """Cells = connected opaque areas (sprites). Same interface as CellGrid for windows and splitting."""
    rects: list[Rect]
    ids: np.ndarray = field(repr=False)     # int32 HxW: nearest sprite for every pixel
    width: int = 0
    height: int = 0
    n_cols: int = 0                          # only for the summary (rows/columns of the sprite centres)
    n_rows: int = 0

    def summary(self) -> str:
        n = len(self.rects)
        return f"{self.n_cols}x{self.n_rows}s" if self.n_cols * self.n_rows == n else f"{n}s"

    def cell_ids(self) -> np.ndarray:
        return self.ids

    def rect_of(self, cell_id: int) -> Rect:
        return self.rects[cell_id]

    @property
    def n_cells(self) -> int:
        return len(self.rects)


@dataclass
class RegionWindow:
    rect: Rect
    source: str                      # "cell" or "margin"
    cells: list[int] = field(default_factory=list)  # ids of the cells touched (CellGrid: row * n_cols + col)


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


def _detect_tile_grid(rgb: np.ndarray, alpha: np.ndarray) -> CellGrid | None:
    """Equal tiles with transparent gaps or clear edges between them (see module docstring)."""
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


def _cluster_count(values: list[float], gap: float) -> int:
    """Number of groups when sorted values are split wherever neighbours are further apart than ``gap``."""
    v = sorted(values)
    return 1 + sum(1 for a, b in zip(v, v[1:]) if b - a > gap)


def _detect_sprites(alpha: np.ndarray) -> SpriteCells | None:
    """Sheet of equally sized sprites separated by transparency, found as connected opaque areas."""
    import cv2

    h, w = alpha.shape
    opaque = (alpha >= ALPHA_GAP_LEVEL).astype(np.uint8)
    if not 0.05 <= 1.0 - float(opaque.mean()) <= 0.95:
        return None
    n, labels, stats, _ = cv2.connectedComponentsWithStats(opaque, connectivity=8)
    big = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= SPRITE_MIN_PX]
    if not SPRITE_MIN_COUNT <= len(big) <= SPRITE_MAX_COUNT:
        return None
    ws = np.array([stats[i, cv2.CC_STAT_WIDTH] for i in big], float)
    hs = np.array([stats[i, cv2.CC_STAT_HEIGHT] for i in big], float)
    similar = (np.abs(ws / np.median(ws) - 1) <= SPRITE_SIZE_TOLERANCE) & (np.abs(hs / np.median(hs) - 1) <= SPRITE_SIZE_TOLERANCE)
    if similar.mean() < SPRITE_SIZE_SHARE:
        return None
    keep = [i for i, ok in zip(big, similar) if ok]
    rects: list[Rect] = []
    for i in keep:
        x, y = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        bw, bh = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        rects.append((max(0, x - SPRITE_MARGIN_PX), max(0, y - SPRITE_MARGIN_PX),
                      min(w - 1, x + bw - 1 + SPRITE_MARGIN_PX), min(h - 1, y + bh - 1 + SPRITE_MARGIN_PX)))
    order = sorted(range(len(keep)), key=lambda j: (rects[j][1] // max(1, int(np.median(hs)) // 2), rects[j][0]))
    keep = [keep[j] for j in order]
    rects = [rects[j] for j in order]
    # Every pixel gets the nearest kept sprite (gaps, fragments, glow belong to the closest one).
    sites = np.ones((h, w), np.uint8)
    for i in keep:
        sites[labels == i] = 0
    _dist, dt_labels = cv2.distanceTransformWithLabels(sites, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_CCOMP)
    to_id = {}
    for idx, i in enumerate(keep):
        yy, xx = np.nonzero(labels == i)
        to_id[int(dt_labels[yy[0], xx[0]])] = idx
    ids = np.zeros((h, w), np.int32)
    for dt_label, idx in to_id.items():
        ids[dt_labels == dt_label] = idx
    cx = [(r[0] + r[2]) / 2 for r in rects]
    cy = [(r[1] + r[3]) / 2 for r in rects]
    n_cols = _cluster_count(cx, float(np.median(ws)) * 0.6)
    n_rows = _cluster_count(cy, float(np.median(hs)) * 0.6)
    return SpriteCells(rects, ids, w, h, n_cols, n_rows)


def _detect_legacy(rgb: np.ndarray, alpha: np.ndarray) -> "CellGrid | SpriteCells | None":
    """Own detection (no exclusion rules): equal tiles, else sprites as connected opaque areas.

    When both exist the one with more cells wins; on a tie the grid (exact tiles).
    """
    grid = _detect_tile_grid(rgb, alpha)
    sprites = _detect_sprites(alpha)
    if sprites is None:
        return grid
    if grid is None or sprites.n_cells > grid.n_cells:
        return sprites
    # Same number of cells: the exact tiles win, unless the sprites do not sit centred in their tiles
    # (rows shifted sideways against each other: the sprites are right, the tile lines cut through them).
    ids = grid.cell_ids()
    for x1, y1, x2, y2 in sprites.rects:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        tx1, ty1, tx2, ty2 = grid.rect_of(int(ids[cy, cx]))
        off_x = abs((x1 + x2) / 2 - (tx1 + tx2) / 2) / max(1, tx2 - tx1 + 1)
        if off_x > SPRITE_CENTRE_TOLERANCE:        # only sideways: rows are rarely of exactly equal height
            return sprites
    return grid


def _grid_from_bounds(col_b: list[int], row_b: list[int], w: int, h: int, source: str) -> CellGrid | None:
    """CellGrid from column/row boundaries; the outer cells reach to the image edge."""
    def clean(bounds: list[int], length: int) -> list[int]:
        b = sorted({int(round(x)) for x in bounds if 0 < int(round(x)) < length})
        return [0] + b + [length]
    cols, rows = clean(col_b[1:-1], w), clean(row_b[1:-1], h)
    if len(cols) < 2 or len(rows) < 2 or (len(cols) - 1) * (len(rows) - 1) < 2:
        return None
    col_index = np.zeros(w, np.int32)
    row_index = np.zeros(h, np.int32)
    for i in range(len(cols) - 1):
        col_index[cols[i]:cols[i + 1]] = i
    for i in range(len(rows) - 1):
        row_index[rows[i]:rows[i + 1]] = i
    return CellGrid(cols, rows, 0.0, 0.0, w, h, col_index, row_index, source)


def gate_reason(rgb: np.ndarray, alpha: np.ndarray) -> str:
    """Verdict of the project's grid detector for this image: its category or the exclusion (X_...)."""
    from .grid_gate import analyze_grid_from_contours

    bgra = np.dstack((rgb[..., ::-1], alpha)).astype(np.uint8)
    data, reason = analyze_grid_from_contours(alpha, bgra)
    return data["category"] if data else str(reason)


def detect_cell_grid(rgb: np.ndarray, alpha: np.ndarray, detector: str | None = None) -> "CellGrid | SpriteCells | None":
    """Cells of a sprite sheet, or None. ``detector`` defaults to CELL_DETECTOR.

    "grid_detector": the project's detector decides, including its exclusions; an image it rejects
    (X_ALPHA, X_NOBOX, X_SINGLE, X_CV_W, X_CV_A) gets no grid. "legacy": the own detectors.
    """
    mode = detector or CELL_DETECTOR
    if mode == "legacy":
        return _detect_legacy(rgb, alpha)
    from .grid_gate import analyze_grid_from_contours

    h, w = alpha.shape
    bgra = np.dstack((rgb[..., ::-1], alpha)).astype(np.uint8)
    data, _reason = analyze_grid_from_contours(alpha, bgra)
    if data is None:
        return None
    return _grid_from_bounds(data["col_bounds"], data["row_bounds"], w, h, str(data["category"]))


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
    cell_map = grid.cell_ids() if grid is not None else None
    for k, (bx1, by1, bx2, by2) in enumerate(bboxes, start=1):
        old = (max(0, bx1 - margin), max(0, by1 - margin), min(w - 1, bx2 + margin), min(h - 1, by2 + margin))
        if grid is None:
            out.append(RegionWindow(old, "margin"))
            continue
        ys, xs = np.nonzero(core_labels[by1:by2 + 1, bx1:bx2 + 1] == k)
        ys, xs = ys + by1, xs + bx1
        ids, counts = np.unique(cell_map[ys, xs], return_counts=True)
        keep = [int(i) for i, c in zip(ids, counts) if c >= MIN_REGION_CELL_PX and c >= MIN_REGION_CELL_FRAC * len(ys)]
        if not keep:
            out.append(RegionWindow(old, "margin"))
            continue
        rects = [grid.rect_of(i) for i in keep]
        x1 = min(min(r[0] for r in rects), bx1)
        y1 = min(min(r[1] for r in rects), by1)
        x2 = max(max(r[2] for r in rects), bx2)
        y2 = max(max(r[3] for r in rects), by2)
        out.append(RegionWindow((x1, y1, x2, y2), "cell", sorted(keep)))
    return out


def split_plan_by_cells(plan, grid: CellGrid | None):
    """Split every region that spans several sprite cells into one region per cell.

    Why: neighbouring text boxes of a strip lie closer than the merge distance, so a whole strip
    becomes ONE region and the ranker shows the whole strip. A cell is the natural unit.

    Numbering is stable: a region inside one cell keeps its number; of a split region the piece in
    the first cell (row-major) keeps the number, the other pieces get new numbers count+1, count+2, ...
    (so labels of unsplit regions stay valid). Territories are re-divided among the pieces.
    Returns (new plan, {new_region_number: old_region_number} for the pieces that were split off,
    plus the kept piece itself); without a grid the plan is returned unchanged.
    """
    import cv2

    if grid is None or plan.count == 0:
        return plan, {}
    h, w = plan.core_labels.shape
    cell_id = grid.cell_ids().astype(np.int64)
    core = plan.core_labels.copy()
    territory = plan.territory.copy()
    bboxes = list(plan.bboxes)
    next_id = plan.count + 1
    origin: dict[int, int] = {}
    for k in range(1, plan.count + 1):
        reg = plan.core_labels == k
        ys, xs = np.nonzero(reg)
        ids, counts = np.unique(cell_id[ys, xs], return_counts=True)
        keep = sorted(int(i) for i, c in zip(ids, counts) if c >= MIN_REGION_CELL_PX and c >= MIN_REGION_CELL_FRAC * len(ys))
        if len(keep) <= 1:
            continue
        # Pieces; slivers in other cells join the piece they are closest to.
        pieces = {cid: reg & (cell_id == cid) for cid in keep}
        loose = reg & ~np.isin(cell_id, keep)
        numbers = {cid: (k if i == 0 else next_id + i - 1) for i, cid in enumerate(keep)}
        next_id += len(keep) - 1

        def nearest_piece(sel: np.ndarray) -> np.ndarray:
            best = np.full((h, w), np.inf, np.float32)
            owner = np.zeros((h, w), np.int32)
            for cid in keep:
                d = cv2.distanceTransform((~pieces[cid]).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_5)
                closer = (d < best) & sel
                best[closer] = d[closer]
                owner[closer] = numbers[cid]
            return owner
        if loose.any():
            owner = nearest_piece(loose)
            for cid in keep:
                pieces[cid] = pieces[cid] | (loose & (owner == numbers[cid]))
        old_territory = plan.territory == k
        owner = nearest_piece(old_territory)
        for cid in keep:
            n = numbers[cid]
            core[pieces[cid]] = n
            territory[old_territory & (owner == n)] = n
            py, px = np.nonzero(pieces[cid])
            box = (int(px.min()), int(py.min()), int(px.max()), int(py.max()))
            if n == k:
                bboxes[k - 1] = box
            else:
                bboxes.append(box)
            origin[n] = k
    new_plan = plan.__class__(count=next_id - 1, core_labels=core, territory=territory, bboxes=bboxes)
    return new_plan, origin


def split_plan_for_image(plan, rgb: np.ndarray, alpha: np.ndarray):
    """Detect the grid of this image and split the plan (stage 1 and stage 2 must both use this)."""
    if not (USE_CELL_WINDOWS and SPLIT_REGIONS_BY_CELLS):
        return plan, {}
    return split_plan_by_cells(plan, detect_cell_grid(rgb, alpha))


def draw_grid(rgb: np.ndarray, grid: "CellGrid | SpriteCells | None") -> np.ndarray:
    """Debug picture: the image with the cells drawn in magenta (boundaries or sprite rectangles)."""
    img = rgb.copy()
    if grid is None:
        return img
    if isinstance(grid, SpriteCells):
        for x1, y1, x2, y2 in grid.rects:
            img[y1:y2 + 1, [x1, x2]] = (255, 0, 255)
            img[[y1, y2], x1:x2 + 1] = (255, 0, 255)
        return img
    for x in grid.col_bounds[1:-1]:
        img[:, max(0, x - 1):x + 1] = (255, 0, 255)
    for y in grid.row_bounds[1:-1]:
        img[max(0, y - 1):y + 1, :] = (255, 0, 255)
    return img
