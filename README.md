# Cleanup — YOLO/LaMa image cleanup

This repository contains a YOLO/LaMa pipeline for removing unwanted text or
other selected elements from images while trying to preserve sprite details,
colors, outlines, and surrounding image structure.

The main workflow generates mask candidates, evaluates them, selects a winner,
and stores masks, cleaned candidates, and reports under `yolo_mask_winners/`.

## Main pipeline

`yolo_mask_auto_select.py` is the command-line entry point. Keep it beside
`yolo_mask_auto_select_modules/` and `yolo_mask_routing_audit.py`.

The module package separates the implementation:

- `config.py`: shared paths, model settings, thresholds, and scoring configuration.
  The default `anchors.csv` is read from
  `~/Downloads/yabai proj/output/csv/anchors.csv`; optional
  `protected_zones.csv` is discovered beside the selected anchors file.
- `cli.py`: command-line arguments.
- `io_utils.py`: image/CSV discovery, anchor handling, masks, and I/O helpers.
- `scoring.py`: candidate metrics and candidate selection.
- `pipeline.py`: YOLO/LaMa orchestration, evaluation, selection, and output.
- `debug_output.py`: overlays, contact sheets, and comparison images.
- `reports.py`: CSV output with dynamically collected metric columns.

Candidate variants include YOLO distance-clipped masks and a Box-only fallback.
The currently active variant set and thresholds are defined in
`yolo_mask_auto_select_modules/config.py`; do not infer current values from
old run logs or archived configuration backups.

## Supporting tools

These are separate utilities, not duplicate pipeline implementations:

- `yolo_mask_routing_audit.py`: reruns YOLO inference to inspect instance
  masks, distance clipping, and body-box coverage. It writes routing reports,
  per-box metrics, instance details, and visual contact sheets. It does not
  run LaMa or modify source images.
- `label_tool.py`: local browser-based UI for manually rating generated
  candidates. It writes `labels.csv` in the selected run directory.
- `train_ranker.py`: trains and evaluates a candidate ranker from the
  candidate score CSV and manual labels.
- `build_final_picks.py`: combines ranker predictions with manual labels and
  can copy the final accepted candidate images into a separate folder.
- `scripts/cleanup_repo.py`: reports repository cleanup candidates by
  default; with `--apply`, removes Python caches and archives old backups,
  the known accidental patch dump, and root-level log files outside the repo.

## CSV and source paths

The pipeline, standalone routing audit, and labeling tool share the defaults
from `yolo_mask_auto_select_modules/config.py`:

- CSV directory: `~/Downloads/yabai proj/output/csv/`
- Anchors: `anchors.csv` in that directory.
- Optional protected zones: `protected_zones.csv` in that directory.
- Source images: `~/Downloads/yabai proj/yabaiproject - Copy/mit_kanji/`

All three tools support their existing command-line path overrides. If the
anchors CSV is moved, pass its new path with `--anchors`; protected zones
will then be discovered beside that file unless an explicit protected CSV
path is supplied. Run-specific CSVs such as `labels.csv` and
`candidate_scores.csv` remain inside their respective run directories.

## Run the pipeline

From the repository root:

```bash
source .venv/bin/activate
python -m py_compile yolo_mask_auto_select.py yolo_mask_routing_audit.py yolo_mask_auto_select_modules/*.py
python yolo_mask_auto_select.py --random-sample 10 --min-m-images 3 --seed 20261018 --dry-run
```

Remove `--dry-run` only when you intend to start processing. Check the
arguments with `python yolo_mask_auto_select.py --help` before a larger run.

## Tests

Run the repository's unit tests with:

```bash
python -m unittest discover -s yolo_mask_auto_select_modules/tests -v
```

The tests cover dynamic candidate-score CSV columns, Box-only rescue behavior,
residual micro-components, and meaningful YOLO body-box coverage. A test result
must come from an actual test run; the existence of test files is not evidence
that the current checkout passes.

## Result and diagnostic data

- `yolo_mask_winners/` contains generated winner masks, candidate images,
  score CSVs, reports, and other run outputs. It is intentionally excluded
  from Git because it is large and is treated as user data.
- `diagnostics/` contains selected diagnostic evidence and replay reports.
  These files are intentionally kept.
- The older `requirements-before-iopaint.txt` is a historical environment
  snapshot, not a complete current dependency lockfile.

Never delete result images or diagnostic reports solely because they are
generated. Remove them only after confirming that they are no longer needed.

## Current repository notes — 15 November 2026

- The latest repository cleanup removed tracked Python bytecode/cache files,
  old source backups, and one accidental diff dump. Their recoverable copies
  are kept in the local archive created next to the checkout.
- The two root-level run logs are being moved into that same kind of external
  archive by the updated cleanup utility; logs nested inside `diagnostics/`
  remain untouched.
- The diagnostic replay for `l009_006` recorded the same winner
  (`HYBRID_D2`) as the archived result, an identical winner mask, and matching
  values for all 1,111 compared metric fields. This was a one-image replay,
  not a new full-dataset model run.
- No claim about the current full test suite or a new full-dataset run is made
  by this documentation update. Verify either with the commands above.
