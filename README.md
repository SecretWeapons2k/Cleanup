# Cleanup — YOLO/LaMa image cleanup

The repository contains the production pipeline for removing selected text
regions from images while preserving nearby sprite details, outlines, alpha,
and protected regions.

## Production files

- `yolo_mask_auto_select.py` — command-line entry point.
- `yolo_mask_routing_audit.py` — required child process. The main pipeline runs
  it to generate YOLO distance masks and the routing report; do not remove it.
- `yolo_mask_auto_select_modules/` — pipeline, mask generation, geometry,
  scoring, fill repair, image I/O, debug output and reports.
- `yolo_mask_auto_select_modules/tests/` — regression tests.
- `config.py` — central paths, candidate settings and performance parameters.

## Default input paths

- Anchors: `~/Downloads/yabai proj/output/csv/anchors.csv`
- Source images: `~/Downloads/yabai proj/mit_kanji/`
- Optional protected zones: `protected_zones.csv` beside the selected anchors
  CSV (or pass `--protected-zones`).
- YOLO weights: `~/.cache/cleanup-models/manga-inpainter-best.pt`
- Default output: `yolo_mask_winners/` in the repository. Use `--output` to
  write runs elsewhere.

All input paths can be overridden through the CLI. Current settings belong in
`yolo_mask_auto_select_modules/config.py`; old run reports are not configuration.

## Run

From the repository root:

```bash
uv run python yolo_mask_auto_select.py --help
uv run python yolo_mask_auto_select.py --random-sample 10 --min-m-images 3 --seed 20261018 --dry-run
```

For a single-image check:

```bash
uv run python yolo_mask_auto_select.py --image f002_004.png --output ~/cleanup_compare/one_image
```

Remove `--dry-run` to process images. The pipeline requires the local IOPaint
installation and the external YOLO checkpoint; neither model weights nor run
outputs belong in this source repository.

## Tests

```bash
uv run python -m compileall -q yolo_mask_auto_select_modules yolo_mask_auto_select.py yolo_mask_routing_audit.py
uv run python -m unittest discover -s yolo_mask_auto_select_modules/tests -t . -v
```

The unit tests exercise the selector, mask geometry/protection, reporting,
fill repair and pipeline orchestration with test doubles. They do not replace a
real GPU/IOPaint run or visual review of the resulting images.

## Outputs and version control

Run outputs, masks, candidate images, contact sheets and reports are user data.
Keep them outside the source tree (or under the ignored `yolo_mask_winners/`
folder). Do not commit generated test runs or diagnostic archives. Compare
`candidate_scores.csv`, `winner_report.csv` and image files only when they come
from the same run directory.
