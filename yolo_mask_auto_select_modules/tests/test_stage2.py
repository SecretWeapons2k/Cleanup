"""Tests for stage-2 selection logic and the Telea filler."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from yolo_mask_auto_select_modules.stage2 import METHODS, _telea, read_selection


def _write_stage1(tmp: Path, rows: list[dict], with_context: bool) -> None:
    cols = ["image", "region", "ring_flat_frac"] + (["context_type"] if with_context else [])
    with (tmp / "stage1_features.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows({k: r[k] for k in cols} for r in rows)
    (tmp / "stage1_run_stats.json").write_text(json.dumps({"region_merge_px": 8}))


ROWS = [
    {"image": "a.png", "region": "1", "ring_flat_frac": "1.0", "context_type": "flat"},
    {"image": "a.png", "region": "2", "ring_flat_frac": "0.3", "context_type": "structured"},
    {"image": "b.png", "region": "1", "ring_flat_frac": "0.7", "context_type": "mixed"},
    {"image": "c.png", "region": "1", "ring_flat_frac": "0.95", "context_type": "flat"},
]


class SelectionTests(unittest.TestCase):
    def _run(self, mode, labels=None, with_context=True):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            _write_stage1(tmp, ROWS, with_context)
            lp = None
            if labels:
                lp = tmp / "labels.csv"
                with lp.open("w", newline="", encoding="utf-8-sig") as f:
                    w = csv.writer(f)
                    w.writerow(["image", "region", "label"])
                    w.writerows(labels)
            sel, merge = read_selection(tmp, lp, mode)
            return {(i, k) for i, regs in sel.items() for k in regs}, merge

    def test_nonflat_takes_structured_and_mixed(self):
        got, merge = self._run("nonflat")
        self.assertEqual(got, {("a.png", 2), ("b.png", 1)})
        self.assertEqual(merge, 8)

    def test_context_is_derived_when_column_missing(self):
        got, _ = self._run("nonflat", with_context=False)
        self.assertEqual(got, {("a.png", 2), ("b.png", 1)})

    def test_labelled_unusable_flat_region_is_included(self):
        got, _ = self._run("nonflat", labels=[("c.png", 1, "unusable"), ("a.png", 1, "usable")])
        self.assertEqual(got, {("a.png", 2), ("b.png", 1), ("c.png", 1)})

    def test_unusable_mode_only_labelled(self):
        got, _ = self._run("unusable", labels=[("c.png", 1, "unusable")])
        self.assertEqual(got, {("c.png", 1)})

    def test_all_mode(self):
        got, _ = self._run("all")
        self.assertEqual(len(got), 4)


class MethodTests(unittest.TestCase):
    def test_telea_only_changes_masked_pixels_and_removes_stripe(self):
        rgb = np.full((60, 80, 3), 150, np.uint8)
        rgb[:, 40:43] = 10
        alpha = np.full((60, 80), 255, np.uint8)
        mask = np.zeros((60, 80), bool)
        mask[:, 38:46] = True
        out, out_alpha = _telea(rgb, alpha, mask)
        self.assertTrue((out[~mask] == rgb[~mask]).all())
        self.assertGreater(int(out[:, 40:43].min()), 100)
        self.assertTrue((out_alpha == 255).all())

    def test_method_registry_is_consistent(self):
        for name, (filler, fn) in METHODS.items():
            self.assertIn(filler, ("lama", "telea"), name)
            self.assertTrue(callable(fn), name)


if __name__ == "__main__":
    unittest.main()
