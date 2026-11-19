"""End-to-end YOLO + LaMa candidate generation, evaluation and winner persistence with batching."""
from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import types
from concurrent.futures import ThreadPoolExecutor
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .candidate_masks import (
    ImageMaskSet, build_image_mask_set, enforce_protection, group_identical_masks, stage_token,
)
from .cli import parse_args
from .config import *
from .debug_output import build_debug_mask_overlay, save_debug_contact_sheet, save_debug_side_by_side_sheet
from .geometry import read_protected_zones, resolve_protected_zones_csv
from .io_utils import (
    choose_output_dir,
    choose_random_sample,
    discover_sources,
    find_iopaint,
    image_name_from,
    make_box_only_mask,
    read_anchor_csv,
    read_box_mask,
    read_routing_rows,
    remove_small_alpha_islands,
    rgb_array,
    write_filtered_anchors,
)
from .reports import write_csv_with_union_fields
from .regions import (
    merge_region_winners, plan_regions, region_rows, region_window, select_per_region, winners_label,
)
from .fill_repair import repair_flat_fill_artifacts
from .scoring import (
    ink_strength_map, make_contrast_mask, make_smart_ink_mask, outline_leak_metrics,
    score_candidate, select_candidate,
)

class RunStats:
    """Wall-clock per pipeline phase plus job counters (written to run_stats.json)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.seconds: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)
        self.batch_log: list[dict[str, Any]] = []

    def add(self, phase: str, seconds: float) -> None:
        with self._lock:
            self.seconds[phase] += seconds

    def timer(self, phase: str):
        stats = self

        class _Timer:
            def __enter__(self_inner):
                self_inner.t0 = time.perf_counter()

            def __exit__(self_inner, *exc):
                stats.add(phase, time.perf_counter() - self_inner.t0)

        return _Timer()


def sort_dist_key(val: Any) -> int:
    s = str(val)
    if s.isdigit():
        return int(s)
    if s == "SMART_INK":
        return 9998
    return 9999

def main() -> int:
    project = Path(__file__).resolve().parent.parent
    args = parse_args(project)
    anchors_path = args.anchors.expanduser().resolve()
    images_dir = args.images_dir.expanduser().resolve()
    weights_path = args.weights.expanduser().resolve()
    audit_script = project / "yolo_mask_routing_audit.py"

    if args.confidence != 0.05:
        print("ERROR: this version expects --confidence 0.05, matching the successful test.", file=sys.stderr)
        return 2
    if args.imgsz < 320 or args.dilation < 0 or args.limit < 0 or args.random_sample < 0:
        print("ERROR: --imgsz must be >=320; --dilation, --limit, and --random-sample must be >=0.", file=sys.stderr)
        return 2
    try:
        distances = sorted({int(x.strip()) for x in args.distances.split(",") if x.strip()})
        if distances != [0, 2, 4, 6, 8, 10]:
            raise ValueError
    except ValueError:
        print("ERROR: --distances must be exactly 0,2,4,6,8,10 for this version.", file=sys.stderr)
        return 2

    contrast_thresholds = tuple(float(value) for value in CONTRAST_MASK_THRESHOLDS)
    contrast_candidate_ids = [
        f"CONTRAST_T{int(round(value * 100)):02d}" for value in contrast_thresholds
    ]
    if (
        any(not 0.0 <= value <= 1.0 for value in contrast_thresholds)
        or len(set(contrast_candidate_ids)) != len(contrast_candidate_ids)
    ):
        print(
            "ERROR: CONTRAST_MASK_THRESHOLDS must be unique after percentage rounding "
            "and each value must be between 0 and 1.",
            file=sys.stderr,
        )
        return 2

    try:
        fieldnames, anchor_rows, anchors_by_image = read_anchor_csv(anchors_path)
        available_names = set(anchors_by_image)
        if args.random_sample:
            if args.image:
                raise ValueError("Use either --image or --random-sample, not both")
            if args.limit:
                raise ValueError("Use either --limit or --random-sample, not both")
            ordered_names = choose_random_sample(
                available_names, args.random_sample, args.min_m_images, args.seed
            )
        else:
            selected = set(available_names)
            if args.image:
                requested = Path(args.image).name
                if requested not in selected:
                    raise ValueError(f"No valid anchor rows for image {requested} in {anchors_path}")
                selected = {requested}
            ordered_names = sorted(selected, key=str.casefold)
            if args.limit:
                ordered_names = ordered_names[:args.limit]
        selected = set(ordered_names)
        if not ordered_names:
            raise ValueError("No images selected.")
        stems = [Path(name).stem for name in ordered_names]
        if len(stems) != len(set(stems)):
            raise ValueError("Selected anchor images contain duplicate stems with different extensions; audit masks would collide.")
        source_paths = discover_sources(images_dir, selected, project)

        yolo_precheck_names = selected
        if yolo_precheck_names:
            if not audit_script.is_file():
                raise FileNotFoundError(f"Required routing script missing: {audit_script}")
            if not weights_path.is_file():
                raise FileNotFoundError(f"YOLO checkpoint not found: {weights_path}")

        zones_path = None
        zones_by_image: dict[str, list[dict[str, Any]]] = {}
        if not args.no_protected_zones:
            zones_path = resolve_protected_zones_csv(
                anchors_path, args.protected_zones
            )
            if zones_path is not None:
                zones_by_image = read_protected_zones(zones_path)
                unknown = sorted(set(zones_by_image) - available_names)
                if unknown:
                    print(f"WARN: protected zones for {len(unknown)} image(s) not in anchors CSV: "
                          f"{', '.join(unknown[:8])}", file=sys.stderr)

        iopaint = find_iopaint()
        output_dir = choose_output_dir(args.output, project)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    batch_size = getattr(sys.modules[__name__], "LAMA_BATCH_SIZE", 25)
    save_cleaned = getattr(sys.modules[__name__], "DEBUG_SAVE_CLEANED_IMAGES", True) and DEBUG_SAVE_CANDIDATES
    save_comparison = getattr(sys.modules[__name__], "DEBUG_SAVE_COMPARISON_SHEET", True) and DEBUG_SAVE_CANDIDATES
    save_overlays = getattr(sys.modules[__name__], "DEBUG_SAVE_RAW_MASK_OVERLAYS", False) and DEBUG_SAVE_CANDIDATES
    save_side_by_side = getattr(sys.modules[__name__], "DEBUG_SAVE_SIDE_BY_SIDE_SHEET", False) and DEBUG_SAVE_CANDIDATES
    if args.no_debug_images:
        save_cleaned = save_comparison = save_overlays = save_side_by_side = False
    use_fill_repair = FILL_REPAIR_ENABLED and not args.no_fill_repair
    use_dedup = DEDUPLICATE_IDENTICAL_MASKS and not args.no_dedup
    use_regions = PER_REGION_SELECTION_ENABLED and not args.no_per_region
    region_merge_px = max(0, int(args.region_merge_px))
    stats = RunStats()
    run_t0 = time.perf_counter()
    dedup_rows: list[dict[str, Any]] = []

    print(f"Selected images: {len(ordered_names)} (Batch-Größe: {batch_size} Bilder pro Durchlauf)")
    print(f"Persistente Ausgabe: {output_dir}")
    if zones_path is not None:
        selected_zone_count = sum(len(zones_by_image.get(n, [])) for n in ordered_names)
        print(f"Schutzzonen: {zones_path} ({selected_zone_count} Zonen in der Auswahl)")
    else:
        print("Schutzzonen: keine")
    print(f"Fill-Repair={use_fill_repair}, Maskendedup={use_dedup}, Auswahl pro Region={use_regions} (merge {region_merge_px}px)")
    print(f"Debug-Optionen: Cleaned={save_cleaned}, Comparison-Sheet={save_comparison}, Overlays={save_overlays}, Side-By-Side={save_side_by_side}")

    if args.dry_run:
        print("DRY RUN: dependencies and source paths checked; no inference or output files created.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    output_masks = output_dir / "masks"
    output_masks.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "winner_report.csv"
    debug_report_path = output_dir / "debug_candidates" / "candidate_scores.csv"

    try:
        with tempfile.TemporaryDirectory(prefix="yolo_mask_auto_select_") as tmp_string:
            tmp = Path(tmp_string)
            yolo_selected_names = selected
            temp_anchors = tmp / "anchors_for_yolo.csv"
            audit_parent = tmp / "routing_audit"
            audit_out = audit_parent

            if yolo_selected_names:
                write_filtered_anchors(temp_anchors, fieldnames, anchor_rows, yolo_selected_names)
                audit_cmd = [
                    sys.executable, str(audit_script),
                    "--images-dir", str(images_dir),
                    "--anchors", str(temp_anchors),
                    "--weights", str(weights_path),
                    "--output", str(audit_parent),
                    "--imgsz", str(args.imgsz),
                    "--min-conf", "0.025",
                    "--dilation", str(args.dilation),
                    "--distances", ",".join(map(str, distances)),
                    "--device", args.device,
                    "--no-visuals",
                ]
                if zones_path is not None:
                    audit_cmd += ["--protected", str(zones_path),
                                  "--protected-margin", str(PROTECTED_ZONE_MARGIN_PX)]
                print(f"\n[1/3] Building YOLO candidate masks for {len(yolo_selected_names)} images...", flush=True)
                with stats.timer("yolo_routing_audit"):
                    proc = subprocess.run(audit_cmd, cwd=project, check=False)
                if proc.returncode != 0:
                    print(f"ERROR: routing audit failed with code {proc.returncode}", file=sys.stderr)
                    return proc.returncode or 1

                report_candidates = list(audit_parent.rglob("routing_report.csv"))
                if not report_candidates:
                    raise FileNotFoundError(f"Routing report missing under audit output: {audit_parent}")
                report_path_audit = max(report_candidates, key=lambda item: item.stat().st_mtime)
                audit_out = report_path_audit.parent
                route_by_image = read_routing_rows(report_path_audit)
            else:
                route_by_image = {}

            box_masks_dir = audit_out / "box_only" / "masks"
            box_masks_dir.mkdir(parents=True, exist_ok=True)

            mask_paths: dict[tuple[str, int], Path] = {}
            candidate_images: list[str] = []

            for name in ordered_names:
                route = str(
                    route_by_image.get(name, {}).get("route_suggestion", "") or ""
                )

                for d in distances:
                    path = (
                        audit_out
                        / "conf_005"
                        / f"distance_{d:02d}"
                        / "masks"
                        / f"{Path(name).stem}.png"
                    )

                    if not path.is_file():
                        raise FileNotFoundError(
                            f"YOLO candidate mask missing: image={name!r}, "
                            f"distance={d}px, path={path}, "
                            f"audit_route={route or '<missing>'}"
                        )

                    mask_paths[(name, d)] = path

                candidate_images.append(name)

            def yolo_masks_for(name: str) -> dict[int, np.ndarray]:
                result: dict[int, np.ndarray] = {}
                for d in distances:
                    with Image.open(mask_paths[(name, d)]) as mask_im:
                        result[d] = np.asarray(mask_im.convert("L"), dtype=np.uint8) > 0
                return result

            def mask_set_for(name: str, rgb: np.ndarray, alpha: np.ndarray) -> ImageMaskSet:
                return build_image_mask_set(
                    rgb=rgb, alpha=alpha,
                    boxes=anchors_by_image.get(name, []),
                    zones=zones_by_image.get(name, []),
                    yolo_masks=yolo_masks_for(name),
                    box_only_mask=read_box_mask(box_masks_dir / f"{Path(name).stem}.png"),
                    contrast_thresholds=contrast_thresholds,
                    contrast_candidate_ids=contrast_candidate_ids,
                    protected_margin_px=PROTECTED_ZONE_MARGIN_PX,
                )

            # ==================================================================
            # [2/3] & [3/3] BATCH-VERARBEITUNG IN HÄPPCHEN (SCHONT /tmp UND RAM)
            # ==================================================================
            total_images = len(ordered_names)
            num_batches = math.ceil(total_images / batch_size)
            all_report_rows: list[dict[str, Any]] = []
            all_debug_score_rows: list[dict[str, Any]] = []
            all_region_rows: list[dict[str, Any]] = []

            print(f"\n[2/3] & [3/3] Processing LaMa and Selection in {num_batches} Batches:", flush=True)

            for b_idx in range(num_batches):
                start_i = b_idx * batch_size
                end_i = min(start_i + batch_size, total_images)
                batch_names = ordered_names[start_i:end_i]
                print(f"\n--- Batch {b_idx + 1}/{num_batches} (Bilder {start_i + 1}-{end_i} von {total_images}) ---", flush=True)

                images_stage = tmp / "lama_inputs" / "images"
                masks_stage = tmp / "lama_inputs" / "masks"
                raw_stage = tmp / "lama_outputs"
                images_stage.mkdir(parents=True, exist_ok=True)
                masks_stage.mkdir(parents=True, exist_ok=True)
                raw_stage.mkdir(parents=True, exist_ok=True)

                stage_names: dict[tuple[str, int | str], tuple[str, str]] = {}
                batch_sets: dict[str, ImageMaskSet] = {}
                batch_groups: dict[str, dict[int | str, int | str]] = {}

                with stats.timer("mask_generation_and_staging"):
                    for name in batch_names:
                        rgb, alpha = rgb_array(source_paths[name])
                        stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in Path(name).stem)
                        mask_set = mask_set_for(name, rgb, alpha)
                        batch_sets[name] = mask_set
                        groups = (group_identical_masks(mask_set.candidates) if use_dedup
                                  else {cid: cid for cid in mask_set.candidates})
                        batch_groups[name] = groups

                        # Every UNIQUE candidate mask (protected zones already subtracted) goes to LaMa
                        # exactly as it is later composited and scored; duplicates reuse the result.
                        for cid, cmask in mask_set.candidates.items():
                            stats.counts["candidates_total"] += 1
                            if groups[cid] != cid:
                                stats.counts["candidates_deduplicated"] += 1
                                dedup_rows.append({
                                    "image": name, "candidate": str(cid), "identical_to": str(groups[cid]),
                                    "mask_pixels": int(np.count_nonzero(cmask)),
                                })
                                continue
                            stats.counts["lama_mask_jobs"] += 1
                            token = stage_token(cid)
                            staged_rgb = f"{stem}__{token}_rgb.png"
                            staged_alpha = f"{stem}__{token}_alpha.png"
                            Image.fromarray(rgb, mode="RGB").save(images_stage / staged_rgb)
                            Image.fromarray(alpha, mode="L").convert("RGB").save(images_stage / staged_alpha)
                            mask_image = Image.fromarray(cmask.astype(np.uint8) * 255, mode="L")
                            mask_image.save(masks_stage / staged_rgb)
                            mask_image.save(masks_stage / staged_alpha)
                            stage_names[(name, cid)] = (staged_rgb, staged_alpha)

                lama_cmd = [iopaint, "run", f"--model={args.model}", f"--device={args.lama_device}",
                            f"--image={images_stage}", f"--mask={masks_stage}", f"--output={raw_stage}"]
                t_lama = time.perf_counter()
                n_workers = max(1, int(args.lama_processes))
                with stats.timer("lama_iopaint"):
                    staged_files = sorted(p_.name for p_ in images_stage.iterdir())
                    n_workers = min(n_workers, len(staged_files))
                    if n_workers <= 1:
                        proc = subprocess.run(lama_cmd, cwd=project, check=False)
                        worker_codes = [proc.returncode]
                    else:
                        # Round-robin shards: every worker gets its own image/mask folder and the
                        # same output folder (file names are unique), so results are unchanged.
                        shard_root = tmp / "lama_shards"
                        raw_stage.mkdir(parents=True, exist_ok=True)
                        shard_cmds = []
                        for k in range(n_workers):
                            (shard_root / f"s{k}" / "images").mkdir(parents=True, exist_ok=True)
                            (shard_root / f"s{k}" / "masks").mkdir(parents=True, exist_ok=True)
                            shard_cmds.append([iopaint, "run", f"--model={args.model}", f"--device={args.lama_device}",
                                               f"--image={shard_root / f's{k}' / 'images'}",
                                               f"--mask={shard_root / f's{k}' / 'masks'}", f"--output={raw_stage}"])
                        for i, fname in enumerate(staged_files):
                            k = i % n_workers
                            shutil.move(str(images_stage / fname), str(shard_root / f"s{k}" / "images" / fname))
                            shutil.move(str(masks_stage / fname), str(shard_root / f"s{k}" / "masks" / fname))
                        workers = [subprocess.Popen(cmd, cwd=project) for cmd in shard_cmds]
                        try:
                            worker_codes = [w.wait() for w in workers]
                        finally:
                            for w in workers:
                                if w.poll() is None:
                                    w.terminate()
                        shutil.rmtree(shard_root, ignore_errors=True)
                    proc = types.SimpleNamespace(returncode=next((c_ for c_ in worker_codes if c_ != 0), 0))
                n_inputs = len(stage_names) * 2
                stats.batch_log.append({"batch": b_idx + 1, "lama_inputs": n_inputs, "workers": n_workers,
                                        "seconds": round(time.perf_counter() - t_lama, 3)})
                print(f"  IOPaint-Batch {b_idx + 1}: {n_inputs} Eingaben, {n_workers} Prozess(e), in {time.perf_counter() - t_lama:.1f}s", flush=True)
                if proc.returncode != 0:
                    print(f"ERROR: IOPaint failed in Batch {b_idx + 1} with code {proc.returncode}", file=sys.stderr)
                    return proc.returncode or 1

                def _process_image(name):
                    """Composite, score and select one image; returns (score_rows, report_rows, log_line).

                    Side effects are limited to this image's own output files, so images can run concurrently.
                    """
                    all_debug_score_rows: list[dict[str, Any]] = []
                    all_report_rows: list[dict[str, Any]] = []
                    original_rgb, original_alpha = rgb_array(source_paths[name])
                    mask_set = batch_sets.pop(name)
                    groups = batch_groups.pop(name)
                    source_detail = ink_strength_map(original_rgb)
                    box_mask = mask_set.box_mask
                    candidate_masks = mask_set.candidates
                    candidate_ids: list[int | str] = list(candidate_masks)
                    body_box_mask = mask_set.body_box_mask
                    yolo_reference_mask = mask_set.yolo_reference_mask
                    scoring_envelope = mask_set.scoring_envelope

                    scored: list[tuple[float, int | str, np.ndarray, dict[str, float]]] = []
                    debug_results: dict[int | str, np.ndarray] = {}
                    final_composites: dict[int | str, tuple[np.ndarray, np.ndarray]] = {}
                    alpha_postprocess_by_candidate: dict[int | str, dict[str, float]] = {}

                    t_comp = time.perf_counter()
                    for cid in candidate_ids:
                        if groups[cid] != cid:
                            final_composites[cid] = final_composites[groups[cid]]
                            alpha_postprocess_by_candidate[cid] = alpha_postprocess_by_candidate[groups[cid]]
                            continue
                        s_name, a_name = stage_names[(name, cid)]

                        with Image.open(raw_stage / s_name) as p_im:
                            pred_rgb = np.asarray(p_im.convert("RGB"), dtype=np.uint8)
                        with Image.open(raw_stage / a_name) as p_im:
                            pred_alpha = np.asarray(p_im.convert("RGB").convert("L"), dtype=np.uint8)

                        c_mask = candidate_masks[cid]
                        f_rgb = original_rgb.copy()
                        f_rgb[c_mask] = pred_rgb[c_mask]
                        if use_fill_repair:
                            with stats.timer("fill_repair"):
                                f_rgb, fr_meta = repair_flat_fill_artifacts(original_rgb, f_rgb, c_mask, mask_set.protected)
                        else:
                            fr_meta = {"fill_repair_clusters_found": 0.0, "fill_repair_clusters_repaired": 0.0,
                                       "fill_repair_pixels_replaced": 0.0}
                        f_alpha = original_alpha.copy()
                        f_alpha[c_mask] = pred_alpha[c_mask]
                        f_rgb, f_alpha, a_meta = remove_small_alpha_islands(
                            f_rgb, f_alpha, original_alpha, protected=mask_set.protected
                        )
                        a_meta.update(fr_meta)
                        a_meta.update(enforce_protection(
                            f_rgb, f_alpha, original_rgb, original_alpha, mask_set.protected
                        ))
                        final_composites[cid] = (f_rgb, f_alpha)
                        alpha_postprocess_by_candidate[cid] = a_meta
                    stats.add("composite_and_postprocess", time.perf_counter() - t_comp)

                    def _cand_token(cid) -> str:
                        return (
                            "smart_ink" if cid == "SMART_INK"
                            else "box_only" if cid == "BOX_ONLY"
                            else str(cid).lower() if isinstance(cid, str)
                            else f"distance_{cid:02d}px"
                        )

                    def _save_candidate_debug(cid, final_rgba, c_mask) -> None:
                        stem = Path(name).stem
                        if save_cleaned:
                            cl_path = output_dir / "debug_candidates" / "cleaned_images" / f"{stem}__{_cand_token(cid)}.png"
                            cl_path.parent.mkdir(parents=True, exist_ok=True)
                            Image.fromarray(final_rgba, mode="RGBA").save(cl_path)
                        # Overlays nur speichern, falls explizit gewünscht
                        if save_overlays:
                            ov = build_debug_mask_overlay(original_rgb, c_mask, anchors_by_image.get(name, []), zones=mask_set.zones)
                            ov_path = output_dir / "debug_candidates" / "raw_mask_overlays" / f"{stem}__{_cand_token(cid)}_overlay.png"
                            ov_path.parent.mkdir(parents=True, exist_ok=True)
                            Image.fromarray(ov, mode="RGB").save(ov_path)

                    def _save_region_crops(specs) -> None:
                        """Ausschnitte je Region und Kandidat + JSON-Index: Grundlage für das Ranking pro Textbox."""
                        stem = Path(name).stem
                        reg_dir = output_dir / "debug_candidates" / "regions"
                        reg_dir.mkdir(parents=True, exist_ok=True)
                        index = {"image": name, "regions": []}
                        original_rgba_full = np.dstack((original_rgb, original_alpha)).astype(np.uint8)
                        for region_no, bbox, window, auto_winner in specs:
                            wx1, wy1, wx2, wy2 = window
                            entry = {"region": region_no, "bbox": list(bbox), "window": list(window),
                                     "auto_winner": auto_winner, "candidates": {}}
                            for cid in candidate_ids:
                                crop_name = f"{stem}__r{region_no}__{_cand_token(cid)}.png"
                                Image.fromarray(debug_results[cid][wy1:wy2 + 1, wx1:wx2 + 1], mode="RGBA").save(reg_dir / crop_name)
                                entry["candidates"][str(cid)] = crop_name
                            orig_name = f"{stem}__r{region_no}__original.png"
                            Image.fromarray(original_rgba_full[wy1:wy2 + 1, wx1:wx2 + 1], mode="RGBA").save(reg_dir / orig_name)
                            entry["original"] = orig_name
                            index["regions"].append(entry)
                        (reg_dir / f"{stem}.json").write_text(json.dumps(index, indent=1), encoding="utf-8")

                    plan = plan_regions(box_mask, region_merge_px) if use_regions else None
                    region_mode = plan is not None and plan.count >= 2
                    region_results: list = []
                    extra_contact_items: list[tuple[str, np.ndarray]] = []

                    t_score = time.perf_counter()
                    if region_mode:
                        # ---- Auswahl pro Region: jede Textbox-Region bekommt ihren eigenen Gewinner ----
                        with stats.timer("region_scoring"):
                            region_results = select_per_region(
                                plan=plan, original_rgb=original_rgb, original_alpha=original_alpha,
                                composites=final_composites, candidate_masks=candidate_masks,
                                groups=groups, extra_metrics=alpha_postprocess_by_candidate,
                                box_mask=box_mask, body_box_mask=body_box_mask,
                                yolo_reference_mask=yolo_reference_mask, scoring_envelope=scoring_envelope,
                                margin_px=REGION_SCORE_MARGIN_PX,
                            )
                        winners_by_region = {r.region: r.winner_id for r in region_results}
                        m_rgb, m_alpha, best_mask = merge_region_winners(
                            plan, original_rgb, original_alpha, final_composites, candidate_masks, winners_by_region
                        )
                        protection_meta = enforce_protection(m_rgb, m_alpha, original_rgb, original_alpha, mask_set.protected)
                        # Ehrliche Kontrolle: das zusammengesetzte Ergebnis einmal als Ganzes bewerten.
                        merged_leak = outline_leak_metrics(source_detail, original_alpha, body_box_mask, best_mask)
                        best_metrics = score_candidate(
                            original_rgb, original_alpha, m_rgb, m_alpha, best_mask,
                            scoring_envelope, box_mask, body_box_mask, yolo_reference_mask,
                            outline_leak_precomputed=merged_leak,
                        )
                        # Rescue-/Tie-Break-Felder und Füllreparatur kommen aus den Region-Gewinnern;
                        # die Region mit dem schlechtesten Textscore ist die Referenz für den Bericht.
                        primary = max(region_results, key=lambda r: r.winner_metrics["text_quality_score"])
                        for key in (
                            "best_text_quality_score", "text_quality_tolerance", "box_only_rescue_applied",
                            "box_only_rescue_reason_code", "box_only_rescue_final_text_advantage",
                            "box_only_rescue_inside_detail_advantage", "residual_spot_tiebreak_applied",
                            "residual_spot_tiebreak_structure_margin", "residual_spot_tiebreak_area_reduction",
                            "source_transparent_ratio",
                        ):
                            if key in primary.winner_metrics:
                                best_metrics[key] = primary.winner_metrics[key]
                        # Füllreparatur-Zähler sind je Kandidat ganzbildbezogen: je verwendetem Gewinner einmal zählen.
                        for key in ("fill_repair_clusters_found", "fill_repair_clusters_repaired", "fill_repair_pixels_replaced"):
                            best_metrics[key] = float(sum(
                                alpha_postprocess_by_candidate[cid].get(key, 0.0) for cid in set(winners_by_region.values())
                            ))
                        best_metrics.update(protection_meta)
                        best_metrics["mask_is_duplicate"] = 0.0
                        best_candidate_id = winners_label(region_results)
                        best_score = float(best_metrics["score"])
                        best_text_score = float(primary.best_text_score)
                        text_tolerance = float(primary.text_tolerance)
                        scored = []
                        merged_rgba = np.dstack((m_rgb, m_alpha)).astype(np.uint8)
                        if save_cleaned or save_comparison or save_overlays:
                            for cid in candidate_ids:
                                debug_results[cid] = np.dstack(final_composites[cid]).astype(np.uint8)
                                _save_candidate_debug(cid, debug_results[cid], candidate_masks[cid])
                            _save_candidate_debug("PER_REGION", merged_rgba, best_mask)
                            extra_contact_items.append(("PER-REGION cleaned [WINNER]", merged_rgba))
                        if save_cleaned:
                            _save_region_crops([(r.region, r.bbox, r.window, str(r.winner_id)) for r in region_results])
                    else:
                        # ---- Alter Ganzbild-Pfad (unverändert): ein Gewinner für das ganze Bild ----
                        metrics_by_rep: dict[int | str, dict[str, float]] = {}
                        for cid in candidate_ids:
                            c_rgb, c_alpha = final_composites[cid]
                            c_mask = candidate_masks[cid]
                            if groups[cid] != cid:
                                # Same mask => same composite => same metrics. Copy the dict because
                                # select_candidate() writes rescue/tie-break fields into it.
                                metrics = dict(metrics_by_rep[groups[cid]])
                                metrics["mask_is_duplicate"] = 1.0
                                final_rgba = debug_results[groups[cid]]
                            else:
                                leak = outline_leak_metrics(source_detail, original_alpha, body_box_mask, c_mask)
                                metrics = score_candidate(
                                    original_rgb, original_alpha, c_rgb, c_alpha,
                                    c_mask, scoring_envelope, box_mask, body_box_mask,
                                    yolo_reference_mask,
                                    outline_leak_precomputed=leak,
                                )
                                metrics.update(alpha_postprocess_by_candidate[cid])
                                metrics["mask_is_duplicate"] = 0.0
                                metrics_by_rep[cid] = dict(metrics)
                                final_rgba = np.dstack((c_rgb, c_alpha)).astype(np.uint8)
                            scored.append((metrics["score"], cid, c_mask, metrics))
                            debug_results[cid] = final_rgba
                            _save_candidate_debug(cid, final_rgba, c_mask)

                        stats.add("scoring_and_debug_images", time.perf_counter() - t_score)
                        with stats.timer("winner_selection"):
                            winner, best_text_score, text_tolerance = select_candidate(scored)
                        best_score, best_candidate_id, best_mask, best_metrics = winner
                        if use_regions and save_cleaned and plan is not None and plan.count == 1:
                            _save_region_crops([(
                                1, plan.bboxes[0],
                                region_window(plan.bboxes[0], REGION_SCORE_MARGIN_PX, original_rgb.shape[1], original_rgb.shape[0]),
                                str(best_candidate_id),
                            )])
                    if region_mode:
                        stats.add("scoring_and_debug_images", time.perf_counter() - t_score)

                    # Gewinner-Maske sofort final speichern
                    out_mask = output_masks / f"{Path(name).stem}.png"
                    Image.fromarray((best_mask.astype(np.uint8) * 255), mode="L").save(out_mask)

                    # Comparison Contact-Sheet speichern
                    if save_comparison:
                        stem = Path(name).stem
                        original_rgba = np.dstack((original_rgb, original_alpha)).astype(np.uint8)
                        result_items = []
                        for cid in candidate_ids:
                            label = (
                                "SMART-INK" if cid == "SMART_INK"
                                else "BOX-ONLY" if cid == "BOX_ONLY"
                                else cid.replace("_", "-") if isinstance(cid, str)
                                else f"{cid}px"
                            )
                            winner_suffix = " [WINNER]" if (not region_mode and cid == best_candidate_id) else ""
                            result_items.append((f"{label} cleaned{winner_suffix}", debug_results[cid]))
                        result_items.extend(extra_contact_items)
                        save_debug_contact_sheet(
                            original_rgba, result_items,
                            output_dir / "debug_candidates" / "contact_sheets" / f"{stem}_cleaned_comparison.jpg"
                        )

                    # Score Zeilen erfassen
                    def _fmt_metrics(m: dict[str, float]) -> dict[str, str]:
                        return {key: (f"{value:.6f}" if math.isfinite(value) else "") for key, value in m.items()}

                    if region_mode:
                        for r in region_results:
                            for _s, cid, m in r.scored:
                                all_debug_score_rows.append({
                                    "image": name, "distance_px": str(cid), "region": str(r.region), **_fmt_metrics(m),
                                })
                        all_debug_score_rows.append({
                            "image": name, "distance_px": "PER_REGION", "region": "0", **_fmt_metrics(best_metrics),
                        })
                    else:
                        by_candidate = {cid: m for _s, cid, _m, m in scored}
                        for cid in candidate_ids:
                            all_debug_score_rows.append({
                                "image": name, "distance_px": str(cid), "region": "1", **_fmt_metrics(by_candidate[cid]),
                            })

                    winner_is_smart = best_candidate_id == "SMART_INK"
                    winner_is_box = best_candidate_id == "BOX_ONLY"
                    winner_is_hybrid = best_candidate_id in {"HYBRID_D2", "HYBRID_D6"}
                    winner_is_contrast = (
                        isinstance(best_candidate_id, str)
                        and best_candidate_id in contrast_candidate_ids
                    )
                    route_lbl = (
                        "PER_REGION_SELECTED" if region_mode
                        else "SMART_INK_COMPETITION_SELECTED" if winner_is_smart
                        else "BOX_ONLY_COMPETITION_SELECTED" if winner_is_box
                        else "HYBRID_COMPETITION_SELECTED" if winner_is_hybrid
                        else "CONTRAST_MASK_COMPETITION_SELECTED" if winner_is_contrast
                        else "YOLO_CONF005_AUTO_SELECTED"
                    )
                    audit_suggestion = str(
                        route_by_image.get(name, {}).get("route_suggestion", "") or ""
                    )
                    rescue_applied = bool(
                        best_metrics.get("box_only_rescue_applied", 0.0)
                    )
                    rescue_reason_code = int(
                        round(float(
                            best_metrics.get("box_only_rescue_reason_code", 0.0)
                        ))
                    )

                    rescue_reason_labels = {
                        1: "BOX_ONLY rescue: text advantage + YOLO outline leak",
                        2: "BOX_ONLY rescue: text advantage >= 0.020",
                        3: "BOX_ONLY rescue: inside residual advantage",
                        4: "BOX_ONLY rescue: combined text and inside advantage",
                    }

                    if rescue_applied:
                        rsn = rescue_reason_labels.get(
                            rescue_reason_code,
                            f"BOX_ONLY rescue: unknown reason code {rescue_reason_code}",
                        )
                    elif bool(best_metrics.get("residual_spot_tiebreak_applied", 0.0)):
                        area_reduction = float(
                            best_metrics.get("residual_spot_tiebreak_area_reduction", 0.0)
                        )
                        structure_margin = float(
                            best_metrics.get("residual_spot_tiebreak_structure_margin", 0.0)
                        )
                        rsn = (
                            "residual micro-component tie-break: "
                            f"area reduced by {area_reduction:.0f}px; "
                            f"structure cost +{structure_margin:.3f}"
                        )
                    else:
                        rsn = "best two-stage score"

                    all_report_rows.append({
                        "image": name,
                        "route": route_lbl,
                        "routing_audit_suggestion": audit_suggestion,
                        "selected_distance_px": str(best_candidate_id),
                        "selected_score": f"{best_score:.6f}", "selected_text_quality_score": f"{best_metrics['text_quality_score']:.6f}",
                        "selected_structure_score": f"{best_metrics['structure_score']:.6f}", "best_text_quality_score": f"{best_text_score:.6f}",
                        "text_quality_tolerance": f"{text_tolerance:.6f}", "outline_leak_score": f"{best_metrics['outline_leak_score']:.6f}",
                        "residual_inside_box_ratio": f"{best_metrics['residual_inside_box_ratio']:.6f}",
                        "source_transparent_ratio": f"{best_metrics['source_transparent_ratio']:.6f}",
                        "box_only_rescue_applied": int(
                            best_metrics.get("box_only_rescue_applied", 0.0)
                        ),
                        "box_only_rescue_reason_code": rescue_reason_code,
                        "box_only_rescue_final_text_advantage": (
                            f"{float(best_metrics.get('box_only_rescue_final_text_advantage', 0.0)):.6f}"
                        ),
                        "box_only_rescue_inside_detail_advantage": (
                            f"{float(best_metrics.get('box_only_rescue_inside_detail_advantage', 0.0)):.6f}"
                        ),
                        "fill_repair_clusters_found": int(best_metrics.get("fill_repair_clusters_found", 0.0)),
                        "fill_repair_clusters_repaired": int(best_metrics.get("fill_repair_clusters_repaired", 0.0)),
                        "fill_repair_pixels_replaced": int(best_metrics.get("fill_repair_pixels_replaced", 0.0)),
                        "protected_pixels": int(best_metrics.get("protected_pixels", 0.0)),
                        "protected_pixels_restored": int(best_metrics.get("protected_pixels_restored", 0.0)),
                        "region_count": plan.count if region_mode else 1,
                        "region_winners": str(best_candidate_id),
                        "reason": rsn,
                    })

                    if region_mode:
                        score_text = " | ".join(
                            f"r{r.region}:best={r.winner_id},text={r.winner_metrics['text_quality_score']:.3f},"
                            f"shape={r.winner_metrics['structure_score']:.3f}" for r in region_results
                        )
                        region_report_rows = region_rows(name, region_results)
                    else:
                        score_text = " | ".join(f"{cid}:text={m['text_quality_score']:.3f},inside={m['residual_inside_box_ratio']:.3f},shape={m['structure_score']:.3f}" for _score, cid, _mask, m in scored)
                        region_report_rows = []
                    return all_debug_score_rows, all_report_rows, f"  {name}: {score_text} -> WINNER {best_candidate_id}", region_report_rows

                t_pp = time.perf_counter()
                pp_workers = max(1, min(int(args.postprocess_workers), len(batch_names)))
                if pp_workers == 1:
                    image_results = [_process_image(n_) for n_ in batch_names]
                else:
                    cv2_threads = cv2.getNumThreads()
                    cv2.setNumThreads(1)   # parallelism comes from the images; avoids oversubscription
                    try:
                        with ThreadPoolExecutor(max_workers=pp_workers) as pool:
                            image_results = list(pool.map(_process_image, batch_names))  # keeps input order
                    finally:
                        cv2.setNumThreads(cv2_threads)
                stats.add("postprocess_wall_clock", time.perf_counter() - t_pp)
                for rows_debug, rows_report, log_line, rows_region in image_results:
                    all_debug_score_rows.extend(rows_debug)
                    all_report_rows.extend(rows_report)
                    all_region_rows.extend(rows_region)
                    print(log_line)

                # CSVs nach jedem Batch aktualisieren (Crash-Schutz)
                cols = list(dict.fromkeys(key for row in all_report_rows for key in row.keys()))
                with report_path.open("w", encoding="utf-8-sig", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=cols)
                    writer.writeheader()
                    writer.writerows(sorted(all_report_rows, key=lambda r: str(r.get("image", "")).casefold()))

                if all_region_rows:
                    write_csv_with_union_fields(
                        output_dir / "region_report.csv",
                        sorted(all_region_rows, key=lambda r: (str(r["image"]).casefold(), int(r["region"]))),
                    )

                if DEBUG_SAVE_CANDIDATES and all_debug_score_rows:
                    debug_report_path.parent.mkdir(parents=True, exist_ok=True)
                    write_csv_with_union_fields(
                        debug_report_path,
                        sorted(all_debug_score_rows, key=lambda r: (str(r["image"]).casefold(), sort_dist_key(r["distance_px"]))),
                    )

                # ==============================================================
                # TEMPORÄREN SPEICHER DIESES BATCHES SOFORT LEEREN!
                # ==============================================================
                shutil.rmtree(images_stage, ignore_errors=True)
                shutil.rmtree(masks_stage, ignore_errors=True)
                shutil.rmtree(raw_stage, ignore_errors=True)

            stats.seconds["total_wall_clock"] = time.perf_counter() - run_t0
            if dedup_rows:
                write_csv_with_union_fields(output_dir / "mask_dedup_report.csv", dedup_rows)
            stats_doc = {
                "images": total_images,
                "batch_size": batch_size,
                "lama_runs_rgb_plus_alpha": 2 * stats.counts["lama_mask_jobs"],
                "lama_runs_without_dedup": 2 * stats.counts["candidates_total"],
                "counts": dict(stats.counts),
                "lama_batches": stats.batch_log,
                "postprocess_workers": max(1, int(args.postprocess_workers)),
                "seconds": {k: round(v, 3) for k, v in stats.seconds.items()},
                "fill_repair_enabled": use_fill_repair,
                "dedup_enabled": use_dedup,
                "per_region_enabled": use_regions,
                "region_merge_px": region_merge_px,
                "images_with_multiple_regions": len({r["image"] for r in all_region_rows}),
                "regions_total": len(all_region_rows),
                "debug_images_enabled": bool(save_cleaned or save_comparison or save_overlays),
            }
            (output_dir / "run_stats.json").write_text(json.dumps(stats_doc, indent=2), encoding="utf-8")
            print("Run stats: " + json.dumps(stats_doc["seconds"]) + f" | LaMa jobs {stats_doc['lama_runs_rgb_plus_alpha']}"
                  f" (ohne Dedup {stats_doc['lama_runs_without_dedup']})")
            print("\n=== AUTO-SELECTION COMPLETE ===")
            print(f"Winner masks: {output_masks}")
            print(f"Decision report: {report_path}")
            if DEBUG_SAVE_CANDIDATES:
                print(f"Debug Candidates Folder: {output_dir / 'debug_candidates'}")
            print(f"Winner masks saved: {len(all_report_rows)} / {total_images} images")
            return 0
    except KeyboardInterrupt:
        print("\nInterrupted; cache removed.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2