"""Regression tests for per-region (per-textbox) selection."""
import unittest

import numpy as np

from yolo_mask_auto_select_modules.regions import (
    merge_region_winners, plan_regions, select_per_region, winners_label,
)

H, W = 120, 260
A = (20, 40, 100, 80)
B = (160, 40, 240, 80)


def _boxmask(*boxes):
    m = np.zeros((H, W), bool)
    for x1, y1, x2, y2 in boxes:
        m[y1:y2 + 1, x1:x2 + 1] = True
    return m


def _scene():
    orig = np.full((H, W, 3), 140, np.uint8)
    for x1, y1, x2, y2 in (A, B):
        for x in range(x1 + 4, x2 - 4, 8):
            orig[y1 + 6:y2 - 6, x:x + 3] = 20
    return orig, np.full((H, W), 255, np.uint8)


def _cleaned(orig, *boxes):
    img = orig.copy()
    for x1, y1, x2, y2 in boxes:
        img[y1:y2 + 1, x1:x2 + 1] = 140
    return img


class PlanTests(unittest.TestCase):
    def test_far_boxes_are_separate_regions_and_territories_partition_image(self):
        plan = plan_regions(_boxmask(A, B), merge_px=8)
        self.assertEqual(plan.count, 2)
        self.assertTrue(((plan.territory >= 1) & (plan.territory <= 2)).all())
        self.assertTrue((plan.territory[_boxmask(A)] == 1).all())
        self.assertTrue((plan.territory[_boxmask(B)] == 2).all())

    def test_large_merge_distance_joins_regions(self):
        self.assertEqual(plan_regions(_boxmask(A, B), merge_px=40).count, 1)

    def test_empty_box_mask_has_no_regions(self):
        self.assertEqual(plan_regions(np.zeros((H, W), bool), 8).count, 0)


class SelectionTests(unittest.TestCase):
    def test_each_region_picks_its_own_best_candidate(self):
        orig, alpha = _scene()
        bm = _boxmask(A, B)
        cands = {
            "ONLY_A": (_cleaned(orig, A), _boxmask(A)),
            "ONLY_B": (_cleaned(orig, B), _boxmask(B)),
            "NOTHING": (orig.copy(), np.zeros((H, W), bool)),
        }
        comps = {c: (v[0], alpha) for c, v in cands.items()}
        masks = {c: v[1] for c, v in cands.items()}
        plan = plan_regions(bm, 8)
        res = select_per_region(
            plan=plan, original_rgb=orig, original_alpha=alpha, composites=comps,
            candidate_masks=masks, groups={c: c for c in cands}, extra_metrics={},
            box_mask=bm, body_box_mask=bm, yolo_reference_mask=bm, scoring_envelope=bm, margin_px=48,
        )
        self.assertEqual(winners_label(res), "r1=ONLY_A;r2=ONLY_B")
        rgb, _a, mask = merge_region_winners(plan, orig, alpha, comps, masks, {r.region: r.winner_id for r in res})
        self.assertEqual(int((rgb[A[1]:A[3], A[0]:A[2]] < 60).sum()), 0)
        self.assertEqual(int((rgb[B[1]:B[3], B[0]:B[2]] < 60).sum()), 0)
        self.assertTrue(mask[_boxmask(A)].all() and mask[_boxmask(B)].all())


if __name__ == "__main__":
    unittest.main()
