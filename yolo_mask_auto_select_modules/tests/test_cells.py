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
    assert wins[1].rect == (0, 0, 199, 49) and wins[1].cells == [(0, 0), (0, 1)]
    assert wins[2].rect == (200, 50, 299, 99)


def test_sliver_in_neighbour_cell_is_ignored():
    rgb, alpha = _sheet(2, 2, 100, 50)
    grid = cells.detect_cell_grid(rgb, alpha)
    assert grid is not None and (grid.n_cols, grid.n_rows) == (2, 2)
    core = np.zeros(alpha.shape, np.int32)
    core[10:30, 40:101] = 1                        # 1 px column leaks into cell 1
    wins = cells.region_windows(core, [(40, 10, 100, 29)], grid, margin=48)
    assert wins[0].cells == [(0, 0)]
    assert wins[0].rect[2] >= 100                  # but the bbox itself always stays inside


def test_without_grid_falls_back_to_margin_window():
    core = np.zeros((100, 200), np.int32)
    core[40:50, 90:110] = 1
    wins = cells.region_windows(core, [(90, 40, 109, 49)], None, margin=48)
    assert wins[0].rect == (42, 0, 157, 97) and wins[0].source == "margin"
