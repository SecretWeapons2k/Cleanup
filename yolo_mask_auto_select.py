#!/usr/bin/env python3
"""Automatically choose between six clipped YOLO masks and a Box-only mask.

Implementation modules live in yolo_mask_auto_select_modules/. Keep that folder
beside this script and keep yolo_mask_routing_audit.py in this project directory.
"""
from yolo_mask_auto_select_modules.pipeline import main

if __name__ == "__main__":
    raise SystemExit(main())
