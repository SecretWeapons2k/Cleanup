"""Tests for the absolute gate features (stage 1 of the sieve)."""
import unittest

import numpy as np

from yolo_mask_auto_select_modules.gate_features import lama_integrity, region_features

H, W = 140, 220
BOX = (60, 50, 160, 90)


def _scene():
    src = np.full((H, W, 3), 150, np.uint8)
    for x in range(BOX[0] + 6, BOX[2] - 6, 9):
        src[BOX[1] + 8:BOX[3] - 8, x:x + 3] = 15
    alpha = np.full((H, W), 255, np.uint8)
    mask = np.zeros((H, W), bool)
    mask[BOX[1]:BOX[3] + 1, BOX[0]:BOX[2] + 1] = True
    return src, alpha, mask


def _feats(res, src, alpha, mask):
    return region_features(src_rgb=src, src_alpha=alpha, res_rgb=res, res_alpha=alpha, mask=mask)


class FeatureTests(unittest.TestCase):
    def test_clean_fill_has_no_residue_or_blob(self):
        src, alpha, mask = _scene()
        res = src.copy(); res[mask] = 150
        f = _feats(res, src, alpha, mask)
        self.assertEqual(f["res_ink_frac"], 0.0)
        self.assertEqual(f["dark_cc_px"], 0.0)
        self.assertGreater(f["src_ink_frac"], 0.1)
        self.assertEqual(f["ink_removed_ratio"], 1.0)

    def test_untouched_text_is_detected_as_residue(self):
        src, alpha, mask = _scene()
        f = _feats(src.copy(), src, alpha, mask)
        self.assertGreater(f["res_ink_frac"], 0.1)
        self.assertLess(f["ink_removed_ratio"], 0.1)
        self.assertGreater(f["res_ink_max_cc_px"], 50)

    def test_dark_blob_is_detected(self):
        src, alpha, mask = _scene()
        res = src.copy(); res[mask] = 150
        yy, xx = np.mgrid[:H, :W]
        res[((yy - 70) ** 2 + (xx - 110) ** 2) < 81] = 25
        f = _feats(res, src, alpha, mask)
        self.assertGreater(f["dark_p99"], 60)
        self.assertGreater(f["dark_cc_px"], 100)
        self.assertGreater(f["dev_frac"], 0.02)

    def test_crossing_line_destroyed_is_detected(self):
        src, alpha, mask = _scene()
        src[:, 100:103] = 30                      # frame line crossing the whole mask
        res = src.copy(); res[mask] = 150         # fill wipes the line inside the mask
        f = _feats(res, src, alpha, mask)
        self.assertGreater(f["crossing_edge_px"], 50)
        self.assertLess(f["crossing_kept_ratio"], 0.2)

    def test_crossing_line_preserved_is_kept(self):
        src, alpha, mask = _scene()
        src[:, 100:103] = 30
        res = src.copy(); res[mask] = 150; res[:, 100:103] = 30
        f = _feats(res, src, alpha, mask)
        self.assertGreater(f["crossing_kept_ratio"], 0.8)


class IntegrityTests(unittest.TestCase):
    def test_identical_outside_mask_is_ok_and_rolled_output_is_flagged(self):
        src, _a, mask = _scene()
        src[:, :, 0] = (np.arange(W)[None, :] * 3 % 256).astype(np.uint8)   # some horizontal detail
        ok = lama_integrity(src, src.copy(), mask)
        self.assertEqual(ok["lama_outside_frac_gt8"], 0.0)
        rolled = lama_integrity(src, np.roll(src, 40, axis=1), mask)
        self.assertGreater(rolled["lama_outside_frac_gt8"], 0.5)
        self.assertEqual(lama_integrity(src, src[:, :-5], mask)["lama_size_mismatch"], 1.0)


if __name__ == "__main__":
    unittest.main()
