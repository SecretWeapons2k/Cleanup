"""Regression tests for detail scoring, Box-only rescue, and meaningful YOLO coverage."""
from __future__ import annotations

import unittest
import numpy as np

from yolo_mask_auto_select_modules.scoring import _retained_detail_ratio, select_candidate
from yolo_mask_routing_audit import body_box_coverage_metrics


class RetainedDetailRatioTests(unittest.TestCase):
    def test_new_generated_texture_has_only_ten_percent_density_weight(self):
        source = np.zeros((3, 3), dtype=np.float32)
        source[1, 1] = 1.0
        source[1, 2] = 1.0
        final = np.zeros((3, 3), dtype=np.float32)
        final[1, 1] = 1.0   # one source detail remains at the same position
        final[0, 0] = 1.0   # one new LaMa-generated detail elsewhere
        region = np.ones((3, 3), dtype=bool)
        visible = np.ones((3, 3), dtype=bool)
        value = _retained_detail_ratio(source, final, region, visible, visible)
        # same_location = 0.5, detail_density = 1.0 => 0.9*0.5 + 0.1*1.0
        self.assertAlmostEqual(value, 0.55, places=6)


class BoxOnlyRescueTests(unittest.TestCase):
    def test_box_only_wins_when_inside_box_residual_is_materially_lower(self):
        scored = []
        for distance, inside, text, structure in [
            (0, .16, .30, .03), (2, .16, .29, .04), (4, .16, .28, .05),
            (6, .16, .27, .04), (8, .16, .26, .03), (10, .16, .25, .02),
        ]:
            metrics = {
                "text_quality_score": text,
                "text_quality_base_score": text,
                "structure_score": structure,
                "residual_inside_box_ratio": inside,
                "outline_leak_score": .13,
            }
            scored.append((structure, distance, np.zeros((2, 2), dtype=bool), metrics))
        box_metrics = {
            "text_quality_score": .30,
            "text_quality_base_score": .30,
            "structure_score": .04,
            "residual_inside_box_ratio": .12,
            "outline_leak_score": 0.0,
        }
        scored.append((.40, "BOX_ONLY", np.zeros((2, 2), dtype=bool), box_metrics))
        winner, _best, _tolerance = select_candidate(scored)
        self.assertEqual(winner[1], "BOX_ONLY")
        self.assertEqual(box_metrics["box_only_rescue_applied"], 1.0)


class MeaningfulBodyCoverageTests(unittest.TestCase):
    def setUp(self):
        self.box = {"x1": 0, "y1": 0, "x2": 39, "y2": 19}
        self.alpha = np.full((20, 40), 255, dtype=np.uint8)
        self.detail = np.zeros((20, 40), dtype=np.float32)
        self.detail[5:15, 8:30] = 0.8

    def test_one_pixel_hit_is_not_meaningful_coverage(self):
        mask = np.zeros((20, 40), dtype=np.uint8)
        mask[10, 12] = 255
        result = body_box_coverage_metrics(mask, self.detail, self.alpha, self.box)
        self.assertTrue(result["raw_overlap_nonzero"])
        self.assertFalse(result["meaningfully_covered"])

    def test_well_covered_high_contrast_region_passes(self):
        mask = np.zeros((20, 40), dtype=np.uint8)
        mask[5:15, 8:30] = 255
        result = body_box_coverage_metrics(mask, self.detail, self.alpha, self.box)
        self.assertTrue(result["meaningfully_covered"])
        self.assertGreaterEqual(result["detail_energy_coverage_ratio"], 0.99)


if __name__ == "__main__":
    unittest.main()
