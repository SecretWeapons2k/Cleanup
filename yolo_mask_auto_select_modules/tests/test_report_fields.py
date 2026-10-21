"""Regression test for dynamically added candidate score fields."""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from yolo_mask_auto_select_modules.reports import write_csv_with_union_fields


class CandidateScoreCsvRegressionTests(unittest.TestCase):
    def test_new_box_only_rescue_metrics_are_written_without_error(self) -> None:
        rows = [{
            "image": "q000_000.png",
            "distance_px": "BOX_ONLY",
            "box_only_rescue_box_inside_residual": "0.123000",
            "box_only_rescue_best_yolo_inside_residual": "0.456000",
            "box_only_rescue_inside_detail_advantage": "0.333000",
        }]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "candidate_scores.csv"
            fields = write_csv_with_union_fields(
                output, rows, preferred_fields=["image", "distance_px"]
            )
            self.assertTrue(output.is_file())
            self.assertTrue(set(rows[0]).issubset(fields))
            with output.open("r", encoding="utf-8-sig", newline="") as stream:
                saved = list(csv.DictReader(stream))
            self.assertEqual(saved, rows)

    def test_future_unknown_score_keys_are_added_automatically(self) -> None:
        rows = [{"image": "example.png", "future_metric": "0.75"}]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "scores.csv"
            fields = write_csv_with_union_fields(output, rows, preferred_fields=["image"])
            self.assertEqual(fields, ["image", "future_metric"])


if __name__ == "__main__":
    unittest.main()
