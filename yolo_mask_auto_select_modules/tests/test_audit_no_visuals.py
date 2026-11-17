"""--no-visuals must change nothing except the skipped overlay/contact-sheet files."""
from __future__ import annotations

import csv
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

import yolo_mask_routing_audit as audit


class _T:
    def __init__(self, v): self.v = list(v)
    def detach(self): return self
    def cpu(self): return self
    def tolist(self): return list(self.v)
    def __len__(self): return len(self.v)


class _Boxes:
    conf = _T([0.9])
    cls = _T([3])
    def __len__(self): return 1


class _FakeYOLO:
    names = {3: "text"}
    def __init__(self, *_a, **_k): pass
    def predict(self, **_k):
        poly = np.array([[40, 30], [100, 30], [100, 52], [40, 52]], dtype=float)
        return [types.SimpleNamespace(boxes=_Boxes(),
                                      masks=types.SimpleNamespace(xy=[poly]))]


def _run(tmp: Path, extra: list[str]) -> Path:
    images = tmp / "images"; images.mkdir(exist_ok=True)
    rgb = np.full((90, 140, 3), 150, np.uint8); rgb[30:50, 40:100] = 20
    Image.fromarray(rgb).save(images / "t001_000.png")
    anchors = tmp / "anchors.csv"
    anchors.write_text("image,type,x1,y1,x2,y2\nt001_000.png,body,40,30,100,50\n", encoding="utf-8")
    weights = tmp / "w.pt"; weights.write_bytes(b"x")
    out = tmp / ("out_" + ("nv" if extra else "v"))
    argv = ["audit", "--images-dir", str(images), "--anchors", str(anchors), "--weights", str(weights),
            "--output", str(out), "--device", "cpu", "--distances", "0,2,4,6,8,10", *extra]
    with mock.patch.object(sys, "argv", argv), \
         mock.patch.dict(sys.modules, {"ultralytics": types.SimpleNamespace(YOLO=_FakeYOLO),
                                  "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))}):
        rc = audit.main()
    assert rc == 0, rc
    return next(out.parent.glob(out.name + "*")) if not (out / "routing_report.csv").exists() else out


def _report(d: Path):
    with (d / "routing_report.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r.pop("overlay_path", None)
        r["candidate_mask_path"] = Path(r["candidate_mask_path"]).relative_to(d).as_posix() if r.get("candidate_mask_path") else ""
    return rows


class AuditNoVisualsTests(unittest.TestCase):
    def test_masks_and_report_identical_visuals_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            v = _run(Path(tmp), [])
            nv = _run(Path(tmp), ["--no-visuals"])
            self.assertEqual(_report(v), _report(nv))
            masks_v = sorted((v / "conf_005").glob("distance_*/masks/*.png"))
            self.assertEqual(len(masks_v), 6)
            for m in masks_v:
                other = nv / m.relative_to(v)
                self.assertTrue(np.array_equal(np.asarray(Image.open(m)), np.asarray(Image.open(other))))
            self.assertTrue(np.array_equal(np.asarray(Image.open(v / "box_only/masks/t001_000.png")),
                                           np.asarray(Image.open(nv / "box_only/masks/t001_000.png"))))
            self.assertTrue(any(v.rglob("*.jpg")))
            self.assertFalse(any(nv.rglob("*.jpg")))


if __name__ == "__main__":
    unittest.main()
