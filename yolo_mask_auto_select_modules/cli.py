"""Command-line argument parsing."""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import (
    DEFAULT_ANCHORS_CSV,
    DEFAULT_DISTANCES,
    DEFAULT_IMAGES_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WEIGHTS,
)

def parse_args(project: Path) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Automatically compare YOLO mask distances against Box-only using LaMa.")
    p.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR,
                   help="Original source image folder")
    p.add_argument("--anchors", type=Path, default=DEFAULT_ANCHORS_CSV,
                   help="Path to anchors CSV file")
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                   help="Path to YOLO segmentation weights")
    p.add_argument("--image", help="Process only this image, e.g. m004_015.png; default: all anchor images")
    p.add_argument("--limit", type=int, default=0, help="Optional cap on selected images (0 = all)")
    p.add_argument("--random-sample", type=int, default=0,
                   help="Select this many random images; requires no --image or --limit")
    p.add_argument("--min-m-images", type=int, default=3,
                   help="Minimum number of sampled images whose filename stem starts with M/m (default: 3)")
    p.add_argument("--seed", type=int, default=20261018,
                   help="Seed for repeatable random sampling; dry-run and real run select the same images")
    p.add_argument("--distances", default=DEFAULT_DISTANCES,
                   help="Comma-separated pixel distances; default: 0,2,4,6,8,10")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--dilation", type=int, default=2)
    p.add_argument("--confidence", type=float, default=0.05,
                   help="Routing threshold. This script currently supports 0.05 only.")
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto",
                   help="YOLO inference device")
    p.add_argument("--lama-device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--model", default="lama", help="IOPaint model name")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help="Directory for winning masks, reports and debug files")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()