"""Tests: identical-mask grouping, dedup/no-dedup equivalence of the full pipeline, fill_repair."""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
from PIL import Image

from yolo_mask_auto_select_modules import pipeline
from yolo_mask_auto_select_modules.candidate_masks import group_identical_masks
from yolo_mask_auto_select_modules.fill_repair import repair_flat_fill_artifacts


class GroupIdenticalMasksTests(unittest.TestCase):
    def test_identical_masks_map_to_first_candidate(self):
        a = np.zeros((6, 6), bool); a[1:3, 1:3] = True
        b = a.copy()
        c = a.copy(); c[5, 5] = True
        groups = group_identical_masks({0: a, 2: b, 4: c, "BOX_ONLY": a.copy()})
        self.assertEqual(groups, {0: 0, 2: 0, 4: 4, "BOX_ONLY": 0})

    def test_one_pixel_difference_is_never_merged(self):
        a = np.zeros((40, 40), bool); a[10:20, 10:20] = True
        b = a.copy(); b[0, 0] = True
        self.assertNotEqual(group_identical_masks({1: a, 2: b})[2], 1)


class FillRepairTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((80, 80), bool); self.mask[10:70, 10:70] = True
        self.flat = np.full((80, 80, 3), 200, np.uint8)

    def test_dark_blob_on_flat_background_is_repaired_inside_mask_only(self):
        f = self.flat.copy(); f[38:44, 38:44] = 20
        out, meta = repair_flat_fill_artifacts(self.flat, f, self.mask)
        self.assertGreaterEqual(meta["fill_repair_clusters_repaired"], 1.0)
        self.assertGreater(int(out[40, 40].min()), 150)
        self.assertTrue(np.array_equal(out[~self.mask], f[~self.mask]))

    def test_clean_fill_is_untouched(self):
        out, meta = repair_flat_fill_artifacts(self.flat, self.flat.copy(), self.mask)
        self.assertTrue(np.array_equal(out, self.flat))
        self.assertEqual(meta["fill_repair_pixels_replaced"], 0.0)

    def test_protected_pixels_are_never_changed(self):
        f = self.flat.copy(); f[38:44, 38:44] = 20
        prot = np.zeros_like(self.mask); prot[36:46, 36:46] = True
        out, _ = repair_flat_fill_artifacts(self.flat, f, self.mask, prot)
        self.assertTrue(np.array_equal(out[prot], f[prot]))

    def test_frame_line_across_flat_area_is_kept(self):
        f = self.flat.copy(); f[40:42, 10:70] = 20   # long line, not a blob
        out, _ = repair_flat_fill_artifacts(self.flat, f, self.mask)
        self.assertTrue(np.array_equal(out[40:42, 20:60], f[40:42, 20:60]))


_INVOCATIONS: list[int] = []   # one entry per fake IOPaint process start


def _fake_environment(tmp: Path, n_images: int = 1):
    h, w = 90, 140
    yy, xx = np.mgrid[0:h, 0:w]
    images = tmp / "images"; images.mkdir()
    names, lines = [], ["image,type,x1,y1,x2,y2"]
    for i in range(n_images):
        rgb = np.stack([(xx * (1.5 + 0.3 * i) + 30 + 17 * i) % 256, (yy * 2 + 60 + 11 * i) % 256,
                        np.full((h, w), 120 + 9 * i)], -1).astype(np.uint8)
        rgb[30:50, 40:100] = 15 + 5 * i                 # "text" block
        rgb[38:42, 45:95 - 4 * i] = 240
        name = f"t{i + 1:03d}_000.png"
        Image.fromarray(rgb).save(images / name)
        names.append(name)
        lines.append(f"{name},body,40,30,100,50")
    anchors = tmp / "anchors.csv"
    anchors.write_text("\n".join(lines) + "\n", encoding="utf-8")
    weights = tmp / "w.pt"; weights.write_bytes(b"x")
    return images, anchors, weights, names


def _run_main(tmp: Path, extra: list[str], lama_calls: list[int], n_images: int = 1):
    images, anchors, weights, names = _fake_environment(tmp, n_images)
    h, w = 90, 140
    out = tmp / "out"

    def fake_run(cmd, cwd=None, check=False):
        cmd = [str(c) for c in cmd]
        if cmd[0] == sys.executable:               # routing audit
            audit_out = Path(cmd[cmd.index("--output") + 1]) / "run"
            box = np.zeros((h, w), np.uint8); box[30:51, 40:101] = 255
            rows = ["confidence_threshold,image,route_suggestion"]
            for nm in names:
                for d in (0, 2, 4, 6, 8, 10):          # D6/D8/D10 saturate -> identical masks
                    dd = min(d, 6)
                    m = cv2.dilate(box, np.ones((2 * dd + 1, 2 * dd + 1), np.uint8)) if dd else box
                    path = audit_out / "conf_005" / f"distance_{d:02d}" / "masks" / nm
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(m).save(path)
                bpath = audit_out / "box_only" / "masks" / nm
                bpath.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(box).save(bpath)
                rows.append(f"0.05,{nm},TEST")
            (audit_out / "routing_report.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        else:                                       # fake IOPaint: deterministic fill
            _INVOCATIONS.append(1)
            img_dir = Path([c for c in cmd if c.startswith("--image=")][0].split("=", 1)[1])
            msk_dir = Path([c for c in cmd if c.startswith("--mask=")][0].split("=", 1)[1])
            out_dir = Path([c for c in cmd if c.startswith("--output=")][0].split("=", 1)[1])
            for f in img_dir.glob("*.png"):
                lama_calls.append(1)
                arr = np.asarray(Image.open(f).convert("RGB")).copy()
                m = np.asarray(Image.open(msk_dir / f.name).convert("L")) > 0
                arr[m] = np.rint(cv2.blur(arr, (15, 15))[m] * 0.97).astype(np.uint8)
                Image.fromarray(arr).save(out_dir / f.name)
        return types.SimpleNamespace(returncode=0)

    class FakePopen:
        def __init__(self, cmd, cwd=None):
            self.returncode = fake_run(cmd, cwd=cwd).returncode
        def wait(self): return self.returncode
        def poll(self): return self.returncode
        def terminate(self): pass

    argv = ["x", "--images-dir", str(images), "--anchors", str(anchors), "--weights", str(weights),
            "--no-protected-zones", "--output", str(out), *extra]
    with mock.patch.object(sys, "argv", argv), \
         mock.patch.object(pipeline, "find_iopaint", return_value="iopaint"), \
         mock.patch.object(pipeline.subprocess, "run", side_effect=fake_run), \
         mock.patch.object(pipeline.subprocess, "Popen", FakePopen):
        rc = pipeline.main()
    return rc, out


def _read(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


class PipelineDedupEquivalenceTests(unittest.TestCase):
    def test_dedup_gives_identical_scores_winner_and_mask_with_fewer_lama_jobs(self):
        results = {}
        for label, extra in (("dedup", []), ("plain", ["--no-dedup"])):
            with tempfile.TemporaryDirectory() as tmp:
                calls: list[int] = []
                rc, out = _run_main(Path(tmp), extra, calls)
                self.assertEqual(rc, 0)
                scores = _read(out / "debug_candidates" / "candidate_scores.csv")
                winner = _read(out / "winner_report.csv")
                mask = np.asarray(Image.open(out / "masks" / "t001_000.png"))
                stats = json.loads((out / "run_stats.json").read_text())
                results[label] = (scores, winner, mask, len(calls), stats)
        s_d, w_d, m_d, n_d, st_d = results["dedup"]
        s_p, w_p, m_p, n_p, _ = results["plain"]
        self.assertEqual(n_p, 22)                      # 11 candidates x (rgb + alpha)
        self.assertLess(n_d, n_p)                      # D8/D10 (and more) reused
        self.assertEqual(n_d, st_d["lama_runs_rgb_plus_alpha"])
        for row in s_d + s_p:
            row.pop("mask_is_duplicate")
        self.assertEqual(s_d, s_p)                     # every metric of every candidate identical
        self.assertEqual(w_d, w_p)
        self.assertTrue(np.array_equal(m_d, m_p))
        self.assertEqual([r["distance_px"] for r in s_d], [r["distance_px"] for r in s_p])
        self.assertIn("fill_repair_clusters_found", w_d[0])

    def test_parallel_iopaint_workers_give_identical_results(self):
        results = {}
        for n in (1, 3):
            with tempfile.TemporaryDirectory() as tmp:
                _INVOCATIONS.clear()
                calls: list[int] = []
                rc, out = _run_main(Path(tmp), ["--lama-processes", str(n)], calls)
                self.assertEqual(rc, 0)
                results[n] = (_read(out / "debug_candidates" / "candidate_scores.csv"), _read(out / "winner_report.csv"),
                              np.asarray(Image.open(out / "masks" / "t001_000.png")), len(calls), len(_INVOCATIONS))
        self.assertEqual(results[1][4], 1)             # sequential: one IOPaint process
        self.assertEqual(results[3][4], 3)             # three concurrent workers
        self.assertEqual(results[1][3], results[3][3]) # every input processed exactly once
        self.assertEqual(results[1][0], results[3][0])
        self.assertEqual(results[1][1], results[3][1])
        self.assertTrue(np.array_equal(results[1][2], results[3][2]))

    def test_threaded_postprocessing_matches_sequential_for_many_images(self):
        results = {}
        for n in (1, 4):
            with tempfile.TemporaryDirectory() as tmp:
                rc, out = _run_main(Path(tmp), ["--postprocess-workers", str(n)], [], n_images=6)
                self.assertEqual(rc, 0)
                masks = {f.name: np.asarray(Image.open(f)) for f in sorted((out / "masks").glob("*.png"))}
                results[n] = (_read(out / "debug_candidates" / "candidate_scores.csv"),
                              _read(out / "winner_report.csv"), masks,
                              json.loads((out / "run_stats.json").read_text())["postprocess_workers"])
        self.assertEqual(len(results[1][1]), 6)
        self.assertEqual(results[1][0], results[4][0])          # every metric of every candidate
        self.assertEqual(results[1][1], results[4][1])
        self.assertEqual(results[1][2].keys(), results[4][2].keys())
        for k in results[1][2]:
            self.assertTrue(np.array_equal(results[1][2][k], results[4][2][k]), k)
        self.assertEqual((results[1][3], results[4][3]), (1, 4))
        self.assertEqual([r["image"] for r in results[4][1]], sorted(r["image"] for r in results[4][1]))

    def test_failing_worker_aborts_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(pipeline.shutil, "move", side_effect=OSError("disk full")):
                rc, out = _run_main(Path(tmp), ["--lama-processes", "2"], [])
            self.assertNotEqual(rc, 0)
            self.assertFalse((out / "masks" / "t001_000.png").exists())

    def test_no_debug_images_flag_skips_pngs_but_keeps_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, out = _run_main(Path(tmp), ["--no-debug-images"], [])
            self.assertEqual(rc, 0)
            self.assertTrue((out / "debug_candidates" / "candidate_scores.csv").is_file())
            self.assertFalse((out / "debug_candidates" / "cleaned_images").exists())
            self.assertFalse((out / "debug_candidates" / "contact_sheets").exists())
            self.assertTrue((out / "masks" / "t001_000.png").is_file())


if __name__ == "__main__":
    unittest.main()
