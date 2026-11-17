"""The cropped fill_repair must be bit-identical to the full-image computation."""
from __future__ import annotations

import unittest

import numpy as np

from yolo_mask_auto_select_modules.fill_repair import _repair_core, repair_flat_fill_artifacts


class FillRepairCropEquivalenceTests(unittest.TestCase):
    def test_random_scenes_match_full_image_result_exactly(self):
        rng = np.random.default_rng(12345)
        repaired_cases = 0
        for case in range(60):
            h, w = int(rng.integers(120, 260)), int(rng.integers(120, 260))
            base = np.full((h, w, 3), rng.integers(60, 230, 3), np.uint8)
            if case % 3 == 0:                                   # textured / frame content
                base[:, ::17] = rng.integers(0, 255, 3)
            mh, mw = int(rng.integers(15, 60)), int(rng.integers(15, 70))
            # include masks touching every image border
            y = int(rng.choice([0, h - mh, rng.integers(0, h - mh)]))
            x = int(rng.choice([0, w - mw, rng.integers(0, w - mw)]))
            mask = np.zeros((h, w), bool); mask[y:y + mh, x:x + mw] = True
            filled = base.copy()
            for _ in range(int(rng.integers(0, 4))):            # dark dots inside the mask
                by, bx = int(rng.integers(y, y + mh - 6)), int(rng.integers(x, x + mw - 6))
                filled[by:by + 5, bx:bx + 5] = rng.integers(0, 40)
            if case % 4 == 0:                                   # line crossing the mask border
                filled[y + mh // 2, max(0, x - 10):x + mw // 2] = 10
            prot = None
            if case % 5 == 0:
                prot = np.zeros((h, w), bool); prot[y:y + 8, x:x + 8] = True
            full, meta_full = _repair_core(base, filled, mask, prot)
            crop, meta_crop = repair_flat_fill_artifacts(base, filled, mask, prot)
            self.assertTrue(np.array_equal(full, crop), f"case {case}")
            self.assertEqual(meta_full, meta_crop, f"case {case}")
            repaired_cases += int(meta_full["fill_repair_clusters_repaired"] > 0)
        self.assertGreater(repaired_cases, 10)                  # the test really exercises repairs

    def test_empty_mask_returns_input_unchanged(self):
        f = np.full((50, 50, 3), 100, np.uint8)
        out, meta = repair_flat_fill_artifacts(f, f, np.zeros((50, 50), bool))
        self.assertTrue(np.array_equal(out, f))
        self.assertEqual(meta["fill_repair_pixels_replaced"], 0.0)


if __name__ == "__main__":
    unittest.main()
