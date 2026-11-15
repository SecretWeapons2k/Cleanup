"""Tests for polygon anchors and protected zones."""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from yolo_mask_auto_select_modules import geometry
from yolo_mask_auto_select_modules.candidate_masks import (
    build_image_mask_set, enforce_protection, stage_token,
)
from yolo_mask_auto_select_modules.io_utils import (
    make_box_only_mask, read_anchor_csv, remove_small_alpha_islands,
)
from yolo_mask_auto_select_modules.scoring import make_contrast_mask, make_smart_ink_mask
from yolo_mask_routing_audit import body_box_coverage_metrics, overlap_with_box

# L-shaped body polygon inside the rectangle (10,5)-(49,24): upper-right notch missing.
L_POLY = "10,5;49,5;49,14;30,14;30,24;10,24"
L_ROW = {"type": "body", "mask_polygon": L_POLY, "polygon": "10,5;49,5;49,24;10,24"}


def l_box() -> dict:
    return {"type": "body", "x1": 10, "y1": 5, "x2": 49, "y2": 24, "poly": geometry.anchor_polygon(L_ROW)}


class ParsingTests(unittest.TestCase):
    def test_parse_polygon(self):
        self.assertEqual(geometry.parse_polygon("1,2;3,4;5,6").shape, (3, 2))
        for bad in ("", None, "1,2;3,4", "a,b;c,d;e,f", "1,2,3;4,5,6;7,8,9"):
            self.assertIsNone(geometry.parse_polygon(bad))

    def test_body_uses_mask_polygon_and_rectangles_stay_rectangles(self):
        self.assertIsNotNone(geometry.anchor_polygon(L_ROW))
        rect = {"type": "body", "mask_polygon": "1,1;9,1;9,5;1,5", "polygon": "1,1;9,1;9,5;1,5"}
        self.assertIsNone(geometry.anchor_polygon(rect))

    def test_body_uses_red_rectangle_only_when_white_mask_is_missing(self):
        row = {
            "type": "body",
            "mask_polygon": "",
            "polygon": L_POLY,  # This is the OCR contour, not the white cleaning mask.
        }
        self.assertIsNone(geometry.anchor_polygon(row))
        box = {"x1": 10, "y1": 5, "x2": 49, "y2": 24, "poly": geometry.anchor_polygon(row)}
        mask = make_box_only_mask(60, 30, [box]) > 0
        self.assertTrue(mask[20, 40])  # the red rectangle fallback covers the notch

    def test_header_uses_rotated_polygon(self):
        header = {"type": "header", "mask_polygon": "", "polygon": "569,27;628,31;627,55;568,51"}
        self.assertIsNotNone(geometry.anchor_polygon(header))

    def test_flags_restore_rectangle_behaviour(self):
        with mock.patch.object(geometry, "USE_BODY_MASK_POLYGONS", False):
            self.assertIsNone(geometry.anchor_polygon(L_ROW))
        header = {"type": "header", "polygon": "569,27;628,31;627,55;568,51"}
        with mock.patch.object(geometry, "USE_HEADER_POLYGONS", False):
            self.assertIsNone(geometry.anchor_polygon(header))


class PolygonMaskTests(unittest.TestCase):
    def test_polygon_excludes_notch_rectangle_does_not(self):
        poly_mask = make_box_only_mask(60, 30, [l_box()]) > 0
        rect_mask = make_box_only_mask(60, 30, [{"x1": 10, "y1": 5, "x2": 49, "y2": 24}]) > 0
        self.assertTrue(poly_mask[20, 15])      # lower-left arm
        self.assertFalse(poly_mask[20, 40])     # notch
        self.assertTrue(rect_mask[20, 40])
        self.assertTrue(np.all(rect_mask >= poly_mask))

    def test_read_anchor_csv_keeps_polygon(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "anchors.csv"
            with path.open("w", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["image", "type", "x1", "y1", "x2", "y2", "polygon", "mask_polygon"])
                w.writeheader()
                w.writerow({"image": "a.png", "type": "body", "x1": 10, "y1": 5, "x2": 49, "y2": 24,
                            "polygon": L_ROW["polygon"], "mask_polygon": L_POLY})
                w.writerow({"image": "a.png", "type": "header", "x1": 0, "y1": 0, "x2": 5, "y2": 5,
                            "polygon": "0,0;5,0;5,5;0,5", "mask_polygon": ""})
            _fields, _rows, by_image = read_anchor_csv(path)
        body, header = by_image["a.png"]
        self.assertIsNotNone(body["poly"])
        self.assertIsNone(header["poly"])

    def test_contrast_and_smart_ink_stay_inside_polygon(self):
        rng = np.random.default_rng(3)
        rgb = rng.integers(0, 255, (30, 60, 3), dtype=np.uint8)
        alpha = np.full((30, 60), 255, dtype=np.uint8)
        poly_region = make_box_only_mask(60, 30, [l_box()]) > 0
        self.assertFalse((make_contrast_mask(rgb, alpha, [l_box()]) & ~poly_region).any())
        self.assertFalse((make_smart_ink_mask(rgb, [l_box()]) & ~poly_region).any())

    def test_coverage_uses_polygon_and_exclude_mask(self):
        h, w = 30, 60
        alpha = np.full((h, w), 255, dtype=np.uint8)
        detail = np.zeros((h, w), dtype=np.float32)
        detail[5:25, 10:50] = 0.8
        full = np.full((h, w), 255, dtype=np.uint8)
        self.assertEqual(overlap_with_box(full, l_box()), int((make_box_only_mask(w, h, [l_box()]) > 0).sum()))
        # A mask covering only the notch must not count as covering the polygon.
        notch = np.zeros((h, w), dtype=np.uint8)
        notch[16:25, 32:50] = 255
        self.assertEqual(body_box_coverage_metrics(notch, detail, alpha, l_box())["overlap_pixels"], 0)
        # Excluded pixels outside the polygon change nothing ...
        left_only = np.zeros((h, w), dtype=np.uint8)
        left_only[5:25, 10:30] = 255
        exclude = np.zeros((h, w), dtype=bool)
        exclude[16:25, 32:50] = True  # lies in the notch (outside the polygon): must change nothing
        base = body_box_coverage_metrics(left_only, detail, alpha, l_box())
        self.assertEqual(body_box_coverage_metrics(left_only, detail, alpha, l_box(), exclude_mask=exclude), base)
        # ... while excluded pixels inside the polygon are no longer required coverage.
        inside = np.zeros((h, w), dtype=bool)
        inside[5:15, 31:50] = True
        with_exclude = body_box_coverage_metrics(left_only, detail, alpha, l_box(), exclude_mask=inside)
        self.assertLess(with_exclude["visible_box_pixels"], base["visible_box_pixels"])
        self.assertGreater(with_exclude["detail_energy_coverage_ratio"], base["detail_energy_coverage_ratio"])


class ProtectedZoneTests(unittest.TestCase):
    def _csv(self, tmp: str) -> Path:
        path = Path(tmp) / "protected_zones.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["image", "zone_id", "x1", "y1", "x2", "y2", "polygon", "text"])
            w.writeheader()
            w.writerow({"image": "a.png", "zone_id": "a#z1", "x1": 2, "y1": 3, "x2": 6, "y2": 5,
                        "polygon": "2,3;6,3;6,5;2,5", "text": "OK"})
            w.writerow({"image": "a.png", "zone_id": "bad", "x1": "x", "y1": 0, "x2": 1, "y2": 1, "polygon": "", "text": ""})
        return path

    def test_default_protected_csv_is_discovered_next_to_anchors(self):
        with tempfile.TemporaryDirectory() as tmp:
            anchors = Path(tmp) / "anchors.csv"
            anchors.touch()
            expected = Path(tmp) / "protected_zones.csv"
            self.assertIsNone(geometry.resolve_protected_zones_csv(anchors))
            expected.touch()
            self.assertEqual(geometry.resolve_protected_zones_csv(anchors), expected)

    def test_explicit_protected_csv_overrides_sibling_discovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            anchors = Path(tmp) / "anchors.csv"
            explicit = Path(tmp) / "elsewhere" / "zones.csv"
            resolved = geometry.resolve_protected_zones_csv(anchors, explicit)
            self.assertEqual(resolved, explicit.resolve())

    def test_read_and_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            zones = geometry.read_protected_zones(self._csv(tmp))
        self.assertEqual(len(zones["a.png"]), 1)          # invalid row skipped
        mask = geometry.protected_mask(zones["a.png"], 20, 10)
        self.assertTrue(mask[3, 2] and mask[5, 6])         # inclusive corners
        self.assertEqual(int(mask.sum()), 5 * 3)
        self.assertGreater(int(geometry.protected_mask(zones["a.png"], 20, 10, margin_px=1).sum()), 15)

    def test_no_candidate_touches_protected_pixels(self):
        rng = np.random.default_rng(7)
        h, w = 30, 60
        rgb = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        alpha = np.full((h, w), 255, dtype=np.uint8)
        zone = {"x1": 20.0, "y1": 8.0, "x2": 35.0, "y2": 20.0, "poly": None}
        yolo = {d: np.ones((h, w), dtype=bool) for d in (0, 2, 4, 6, 8, 10)}
        ms = build_image_mask_set(
            rgb=rgb, alpha=alpha, boxes=[l_box()], zones=[zone], yolo_masks=yolo,
            box_only_mask=make_box_only_mask(w, h, [l_box()]) > 0,
            contrast_thresholds=(0.38,), contrast_candidate_ids=["CONTRAST_T38"],
        )
        self.assertEqual(list(ms.candidates)[:6], [0, 2, 4, 6, 8, 10])
        self.assertEqual(list(ms.candidates)[6:], ["HYBRID_D2", "HYBRID_D6", "CONTRAST_T38", "SMART_INK", "BOX_ONLY"])
        self.assertTrue(ms.protected.any())
        for cid, mask in ms.candidates.items():
            self.assertFalse((mask & ms.protected).any(), cid)
        for ref in (ms.box_mask, ms.body_box_mask, ms.yolo_reference_mask, ms.scoring_envelope):
            self.assertFalse((ref & ms.protected).any())

    def test_without_zones_masks_equal_unprotected_masks(self):
        h, w = 30, 60
        rgb = np.random.default_rng(1).integers(0, 255, (h, w, 3), dtype=np.uint8)
        alpha = np.full((h, w), 255, dtype=np.uint8)
        yolo = {d: np.ones((h, w), dtype=bool) for d in (0, 2, 4, 6, 8, 10)}
        box_only = make_box_only_mask(w, h, [l_box()]) > 0
        ms = build_image_mask_set(rgb=rgb, alpha=alpha, boxes=[l_box()], zones=[], yolo_masks=yolo,
                                  box_only_mask=box_only, contrast_thresholds=(0.38,),
                                  contrast_candidate_ids=["CONTRAST_T38"])
        self.assertFalse(ms.protected.any())
        self.assertTrue(np.array_equal(ms.candidates["BOX_ONLY"], box_only))

    def test_alpha_island_touching_protected_is_kept(self):
        h, w = 20, 40
        rgb = np.full((h, w, 3), 90, dtype=np.uint8)
        alpha = np.zeros((h, w), dtype=np.uint8)
        alpha[2:5, 2:5] = 255          # island A (protected)
        alpha[10:13, 20:23] = 255      # island B (free)
        protected = np.zeros((h, w), dtype=bool)
        protected[3, 3] = True
        _rgb, out_alpha, meta = remove_small_alpha_islands(rgb, alpha, alpha, protected=protected)
        self.assertTrue((out_alpha[2:5, 2:5] == 255).all())
        self.assertTrue((out_alpha[10:13, 20:23] == 0).all())
        self.assertEqual(meta["alpha_small_components_removed"], 1.0)

    def test_enforce_protection_restores_and_reports(self):
        h, w = 10, 10
        orig_rgb = np.full((h, w, 3), 50, dtype=np.uint8)
        orig_alpha = np.full((h, w), 255, dtype=np.uint8)
        f_rgb, f_alpha = orig_rgb.copy(), orig_alpha.copy()
        f_rgb[2, 2] = 0
        f_alpha[3, 3] = 0
        protected = np.zeros((h, w), dtype=bool)
        protected[2:4, 2:4] = True
        meta = enforce_protection(f_rgb, f_alpha, orig_rgb, orig_alpha, protected)
        self.assertEqual(meta, {"protected_pixels": 4.0, "protected_pixels_restored": 2.0})
        self.assertTrue(np.array_equal(f_rgb, orig_rgb) and np.array_equal(f_alpha, orig_alpha))

    def test_stage_tokens_keep_historic_file_names(self):
        self.assertEqual([stage_token(c) for c in (0, 10, "HYBRID_D2", "HYBRID_D6", "CONTRAST_T38", "SMART_INK", "BOX_ONLY")],
                         ["d00", "d10", "hybrid_d02", "hybrid_d06", "contrast_t38", "smart_ink", "box_only"])


if __name__ == "__main__":
    unittest.main()
