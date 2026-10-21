# Modular Cleanup YOLO/LaMa selector

## Files to keep together

Copy the complete bundle into `~/Downloads/Cleanup/` so the layout is:

```text
Cleanup/
  yolo_mask_auto_select.py             # executable entry point
  yolo_mask_routing_audit.py           # KEEP the existing companion script; required by the pipeline
  yolo_mask_auto_select_modules/
    __init__.py
    config.py                          # settings and thresholds
    cli.py                             # command-line arguments
    io_utils.py                        # CSV, image discovery, alpha cleanup, optional OCR
    scoring.py                         # text leak, text residual, contour and structure metrics
    debug_output.py                    # overlays and side-by-side previews
    reports.py                         # robust CSV writer (union of actual metric fields)
    pipeline.py                        # workflow orchestration
```

Do not move only `yolo_mask_auto_select.py`: it imports the package folder. Keep `yolo_mask_routing_audit.py` next to the launcher.

## Install/replace

Download the updated ZIP, then extract its contents into `~/Downloads/Cleanup/`, allowing replacement of both `yolo_mask_auto_select.py` and `yolo_mask_routing_audit.py`. The existing `anchors.csv`, `.venv`, `raw/`, `cleaned/`, and output directories are not included or modified by this bundle.

## Run

```bash
cd ~/Downloads/Cleanup
source .venv/bin/activate
python -m py_compile yolo_mask_auto_select.py yolo_mask_routing_audit.py yolo_mask_auto_select_modules/*.py
python yolo_mask_auto_select.py --random-sample 10 --min-m-images 3 --seed 20261018 --dry-run
python yolo_mask_auto_select.py --random-sample 10 --min-m-images 3 --seed 20261018
```

## Fixed report failure

The candidate CSV writer now merges the preferred columns with every key actually present in candidate score dictionaries before writing. This includes the three newer Box-only rescue fields that triggered `ValueError: dict contains fields not in fieldnames`. New metric keys are also automatically included rather than aborting at the very end of a long LaMa run.

## Regression test for the previous CSV crash

```bash
cd ~/Downloads/Cleanup
source .venv/bin/activate
python -m unittest discover -s yolo_mask_auto_select_modules/tests -v
```

This checks that the three `box_only_rescue_*` metrics and future newly introduced score keys are added to the CSV header instead of aborting `DictWriter`.


## Latest scoring and routing safeguards

- The near-best text tolerance is tightened (`0.003` floor, `0.008` ceiling, `0.05` of score spread).
- Original detail retained at the same location has 90% of `_retained_detail_ratio`; new LaMa texture density contributes 10%.
- Box-only rescue thresholds are lower so a meaningful text-cleanup advantage or a significant inside-box improvement can override structure-only selection.
- Routing no longer accepts a single overlapping pixel as a sufficient body-box hit. It checks absolute overlap, source-detail energy coverage, and coverage of high-contrast pixels, all configurable in `config.py`. Per-box ratios appear in `body_box_coverage.csv`.
- A box with insufficient meaningful coverage triggers the existing Box-only fallback. Boxes with meaningful but still imperfect coverage continue to the seven-candidate comparison (six YOLO distances plus Box-only).
- These thresholds are cautious initial values, not universal truths. Review the new per-box coverage report and debug previews on the random sample before scaling to the full collection.

Regression tests:

```bash
python -m unittest discover -s yolo_mask_auto_select_modules/tests -v
```
