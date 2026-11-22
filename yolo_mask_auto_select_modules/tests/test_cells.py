"""Tests for cells.py (synthetic sheets only, no real images needed).

Run from the repo root, with cells.py placed in yolo_mask_auto_select_modules/:
    uv run python -m pytest yolo_mask_auto_select_modules/tests/test_cells.py
"""
from __future__ import annotations

import numpy as np

from yolo_mask_auto_select_modules import cells


def _sheet(cols: int, rows: int, cw: int, ch: int, *, gap: int = 0, seed: int = 0, blobs: int = 3):
    """Opaque tiles with a dark 1px frame, or tiles separated by transparent gaps (gap > 0)."""
    rng = np.random.default_rng(seed)
    w = cols * cw
    h = rows * ch
    rgb = np.zeros((h, w, 3), np.uint8)
    alpha = np.full((h, w), 255, np.uint8)
    for r in range(rows):
        for c in range(cols):
            x0, y0 = c * cw, r * ch
            base = rng.integers(40, 200, 3)
            rgb[y0:y0 + ch, x0:x0 + cw] = base
            rgb[y0 + 5:y0 + ch - 5, x0 + 5:x0 + cw - 5] = base // 3        # darker body
            for _ in range(blobs):                                           # "text" blobs, random places
                tx = x0 + int(rng.integers(8, cw - 30))
                ty = y0 + int(rng.integers(8, ch - 14))
                rgb[ty:ty + int(rng.integers(3, 7)), tx:tx + int(rng.integers(8, 22))] = 230
            noise = rng.integers(-6, 7, (ch, cw, 1))                     # film grain
            rgb[y0:y0 + ch, x0:x0 + cw] = np.clip(rgb[y0:y0 + ch, x0:x0 + cw].astype(int) + noise, 0, 255)
            if gap:
                alpha[y0:y0 + ch, x0:x0 + gap // 2 + 1] = 0
                alpha[y0:y0 + ch, x0 + cw - (gap - gap // 2):x0 + cw] = 0
                alpha[y0:y0 + gap // 2 + 1, x0:x0 + cw] = 0
                alpha[y0 + ch - (gap - gap // 2):y0 + ch, x0:x0 + cw] = 0
    return rgb, alpha


def test_detects_opaque_grid():
    rgb, alpha = _sheet(4, 5, 90, 40)
    grid = cells.detect_cell_grid(rgb, alpha)
    assert grid is not None and (grid.n_cols, grid.n_rows) == (4, 5)
    assert grid.col_bounds == [0, 90, 180, 270, 360]
    assert grid.row_bounds == [0, 40, 80, 120, 160, 200]


def test_detects_transparent_gap_grid():
    rgb, alpha = _sheet(3, 5, 116, 57, gap=4)
    grid = cells.detect_cell_grid(rgb, alpha)
    assert grid is not None and (grid.n_cols, grid.n_rows) == (3, 5)


def test_single_cell_and_noise_give_none():
    rgb, alpha = _sheet(1, 1, 141, 57)
    assert cells.detect_cell_grid(rgb, alpha) is None
    rng = np.random.default_rng(1)
    noise = rng.integers(0, 255, (240, 360, 3), dtype=np.uint8)
    assert cells.detect_cell_grid(noise, np.full((240, 360), 255, np.uint8)) is None


def test_single_button_with_one_text_line_gets_no_grid():
    # Known limit: a single cell with several strong inner edges (2-3 bright blobs) is split
    # wrongly in 12-32 % of synthetic cases; with one text blob (a normal button) never.
    for seed in range(40):
        rgb, alpha = _sheet(1, 1, 141, 57, seed=seed, blobs=1)
        assert cells.detect_cell_grid(rgb, alpha) is None


def test_strip_with_very_clear_boundaries_is_detected():
    rgb, alpha = _sheet(4, 1, 264, 72)
    rgb[:, 264::264, :] = 255                       # bright 1px separators = very clear boundaries
    grid = cells.detect_cell_grid(rgb, alpha)
    assert grid is not None and (grid.n_cols, grid.n_rows) == (4, 1)


def test_single_column_of_states_is_detected():
    rgb, alpha = _sheet(1, 5, 120, 45)
    grid = cells.detect_cell_grid(rgb, alpha)
    assert grid is not None and (grid.n_cols, grid.n_rows) == (1, 5)


def test_mostly_transparent_image_gives_none():
    rgb = np.zeros((200, 300, 3), np.uint8)
    alpha = np.zeros((200, 300), np.uint8)
    alpha[90:110, 140:160] = 255
    assert cells.detect_cell_grid(rgb, alpha) is None


def test_region_window_is_union_of_touched_cells():
    rgb, alpha = _sheet(3, 2, 100, 50)
    grid = cells.detect_cell_grid(rgb, alpha)
    core = np.zeros(alpha.shape, np.int32)
    core[10:20, 120:160] = 1                       # region 1: inside cell (row 0, col 1)
    core[10:20, 80:130] = 2                        # region 2: spans columns 0 and 1
    core[60:70, 205:215] = 3                       # region 3: cell (row 1, col 2)
    bboxes = [(120, 10, 159, 19), (80, 10, 129, 19), (205, 60, 214, 69)]
    wins = cells.region_windows(core, bboxes, grid, margin=48)
    assert wins[0].rect == (100, 0, 199, 49) and wins[0].source == "cell"
    assert wins[1].rect == (0, 0, 199, 49) and wins[1].cells == [0, 1]
    assert wins[2].rect == (200, 50, 299, 99)


def test_sliver_in_neighbour_cell_is_ignored():
    rgb, alpha = _sheet(2, 2, 100, 50)
    grid = cells.detect_cell_grid(rgb, alpha)
    assert grid is not None and (grid.n_cols, grid.n_rows) == (2, 2)
    core = np.zeros(alpha.shape, np.int32)
    core[10:30, 40:101] = 1                        # 1 px column leaks into cell 1
    wins = cells.region_windows(core, [(40, 10, 100, 29)], grid, margin=48)
    assert wins[0].cells == [0]
    assert wins[0].rect[2] >= 100                  # but the bbox itself always stays inside


def test_without_grid_falls_back_to_margin_window():
    core = np.zeros((100, 200), np.int32)
    core[40:50, 90:110] = 1
    wins = cells.region_windows(core, [(90, 40, 109, 49)], None, margin=48)
    assert wins[0].rect == (42, 0, 157, 97) and wins[0].source == "margin"


# ---- region split by cells -------------------------------------------------------------------

class _Plan:
    """Minimal stand-in with the fields of regions.RegionPlan."""
    def __init__(self, count, core_labels, territory, bboxes):
        self.count, self.core_labels, self.territory, self.bboxes = count, core_labels, territory, bboxes


def _plan_one_region_over(shape, boxes):
    core = np.zeros(shape, np.int32)
    for x1, y1, x2, y2 in boxes:
        core[y1:y2 + 1, x1:x2 + 1] = 1
    ys, xs = np.nonzero(core)
    return _Plan(1, core, np.ones(shape, np.int32), [(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))])


def test_region_over_several_cells_is_split_into_one_region_per_cell():
    rgb, alpha = _sheet(3, 2, 100, 50)
    grid = cells.detect_cell_grid(rgb, alpha)
    plan = _plan_one_region_over(alpha.shape, [(10, 10, 89, 39), (110, 10, 189, 39), (210, 10, 289, 39)])
    new, origin = cells.split_plan_by_cells(plan, grid)
    assert new.count == 3 and origin == {1: 1, 2: 1, 3: 1}
    assert [int((new.core_labels == k).sum()) for k in (1, 2, 3)] == [80 * 30] * 3
    assert np.array_equal(new.core_labels > 0, plan.core_labels > 0)        # box pixels unchanged
    assert set(np.unique(new.territory)) == {1, 2, 3}                       # every pixel has an owner
    wins = cells.region_windows(new.core_labels, new.bboxes, grid, 48)
    assert [w.rect for w in wins] == [(0, 0, 99, 49), (100, 0, 199, 49), (200, 0, 299, 49)]


def test_unsplit_regions_keep_their_number_and_split_pieces_get_new_ones():
    rgb, alpha = _sheet(3, 2, 100, 50)
    grid = cells.detect_cell_grid(rgb, alpha)
    core = np.zeros(alpha.shape, np.int32)
    core[10:40, 10:190] = 1                       # region 1 spans cells 0 and 1
    core[60:90, 210:290] = 2                      # region 2: one cell (row 1, col 2)
    terr = np.where(core > 0, core, 0)
    terr[terr == 0] = 1
    boxes = [(10, 10, 189, 39), (210, 60, 289, 89)]
    new, origin = cells.split_plan_by_cells(_Plan(2, core, terr, boxes), grid)
    assert new.count == 3
    assert int((new.core_labels == 2).sum()) == 30 * 80                      # region 2 untouched, same number
    assert origin == {1: 1, 3: 1} and 2 not in origin                         # piece in cell 1 became region 3


def test_split_without_grid_changes_nothing():
    plan = _plan_one_region_over((50, 100), [(10, 10, 80, 40)])
    new, origin = cells.split_plan_by_cells(plan, None)
    assert new is plan and origin == {}


# ---- sprites as connected opaque areas ---------------------------------------------------------

def _ragged_sheet(cols, rows, sw=120, sh=40, gap=10, shift=(0, 7, 14), used_cols=None, seed=0):
    """Transparent sheet with ragged blob sprites; each row may be shifted sideways (no common gap column)."""
    import cv2
    rng = np.random.default_rng(seed)
    w, h = cols * (sw + gap), rows * (sh + gap)
    rgb = np.zeros((h, w, 3), np.uint8)
    alpha = np.zeros((h, w), np.uint8)
    for r in range(rows):
        for c in range(used_cols or cols):
            x0 = c * (sw + gap) + shift[r % len(shift)]
            y0 = r * (sh + gap)
            pts = np.array([[x0 + rng.integers(0, 6), y0 + rng.integers(0, 6)], [x0 + sw - rng.integers(0, 6), y0 + rng.integers(0, 6)],
                            [x0 + sw - rng.integers(0, 6), y0 + sh - rng.integers(0, 6)], [x0 + rng.integers(0, 6), y0 + sh - rng.integers(0, 6)]])
            cv2.fillPoly(alpha, [pts.astype(np.int32)], 255)
            cv2.fillPoly(rgb, [pts.astype(np.int32)], (60 + 20 * r, 40, 90))
    return rgb, alpha


def test_ragged_sprites_with_shifted_rows_are_found():
    rgb, alpha = _ragged_sheet(4, 3)
    grid = cells.detect_cell_grid(rgb, alpha)
    assert isinstance(grid, cells.SpriteCells) and grid.n_cells == 12 and grid.summary() == "4x3s"
    assert len(np.unique(grid.cell_ids())) == 12 and grid.cell_ids().min() >= 0


def test_sprites_with_an_empty_column_are_found():
    rgb, alpha = _ragged_sheet(4, 3, used_cols=3)                         # like m004_004: 4th column empty
    grid = cells.detect_cell_grid(rgb, alpha)
    assert isinstance(grid, cells.SpriteCells) and grid.n_cells == 9 and grid.summary() == "3x3s"


def test_few_or_unequal_islands_are_no_sprite_sheet():
    import cv2
    rgb = np.zeros((200, 400, 3), np.uint8)
    alpha = np.zeros((200, 400), np.uint8)
    cv2.rectangle(alpha, (10, 10), (80, 40), 255, -1)                      # one sprite only
    assert cells._detect_sprites(alpha) is None
    cv2.rectangle(alpha, (150, 20), (390, 190), 255, -1)                   # big island of another size
    cv2.rectangle(alpha, (10, 100), (40, 120), 255, -1)
    cv2.rectangle(alpha, (60, 130), (130, 190), 255, -1)
    assert cells._detect_sprites(alpha) is None                            # four islands, sizes differ


def test_region_window_and_split_follow_sprites():
    rgb, alpha = _ragged_sheet(4, 2, shift=(0, 7))
    grid = cells.detect_cell_grid(rgb, alpha)
    sw = 120 + 10
    core = np.zeros(alpha.shape, np.int32)
    core[15:30, 20:230] = 1                         # one region over sprites 0 and 1 of row 0
    ys, xs = np.nonzero(core)
    plan = _Plan(1, core, np.ones(alpha.shape, np.int32), [(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))])
    wins = cells.region_windows(plan.core_labels, plan.bboxes, grid, 48)
    assert wins[0].source == "cell" and len(wins[0].cells) == 2
    new, origin = cells.split_plan_by_cells(plan, grid)
    assert new.count == 2 and origin == {1: 1, 2: 1}
    wins2 = cells.region_windows(new.core_labels, new.bboxes, grid, 48)
    assert all(len(w.cells) == 1 for w in wins2)
    assert wins2[0].rect[2] < sw * 2 and wins2[1].rect[0] >= sw - 8          # windows follow the single sprites


def test_tiles_win_a_tie_only_when_no_sprite_crosses_a_tile_boundary():
    rgb, alpha = _sheet(3, 5, 116, 57, gap=4)                              # separated tiles, same cell count both ways
    assert isinstance(cells.detect_cell_grid(rgb, alpha), cells.CellGrid)
    rgb, alpha = _ragged_sheet(4, 3, shift=(0, 7, 14))                     # rows shifted: tile lines cut sprites
    assert isinstance(cells.detect_cell_grid(rgb, alpha), cells.SpriteCells)
