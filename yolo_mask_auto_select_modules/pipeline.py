"""End-to-end YOLO + LaMa candidate generation, evaluation and winner persistence with batching."""
from __future__ import annotations

import csv
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .cli import parse_args
from .config import *
from .debug_output import build_debug_mask_overlay, save_debug_contact_sheet, save_debug_side_by_side_sheet
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
from .scoring import (
    ink_strength_map, make_contrast_mask, make_smart_ink_mask, outline_leak_metrics,
    score_candidate, select_candidate,
)

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

    print(f"Selected images: {len(ordered_names)} (Batch-Größe: {batch_size} Bilder pro Durchlauf)")
    print(f"Persistente Ausgabe: {output_dir}")
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
                ]
                print(f"\n[1/3] Building YOLO candidate masks for {len(yolo_selected_names)} images...", flush=True)
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

            outline_leak_cache: dict[tuple[str, int | str], dict[str, float]] = {}
            for name in ordered_names:
                source_rgb, source_alpha = rgb_array(source_paths[name])
                h_img, w_img = source_alpha.shape
                body_boxes = [b for b in anchors_by_image.get(name, []) if str(b.get("type", "")).casefold() == "body"]
                body_box_mask = make_box_only_mask(w_img, h_img, body_boxes) > 0
                source_detail = ink_strength_map(source_rgb)
                box_only_mask = read_box_mask(box_masks_dir / f"{Path(name).stem}.png")
                smart_mask = make_smart_ink_mask(source_rgb, anchors_by_image.get(name, []))

                if name in candidate_images:
                    cached_yolo_masks: dict[int, np.ndarray] = {}
                    for d in distances:
                        with Image.open(mask_paths[(name, d)]) as mask_im:
                            c_mask = np.asarray(mask_im.convert("L"), dtype=np.uint8) > 0
                        cached_yolo_masks[d] = c_mask
                        outline_leak_cache[(name, d)] = outline_leak_metrics(source_detail, source_alpha, body_box_mask, c_mask)

                    # Hybride Kandidaten: vollständige Box-Maske OR YOLO-Maske.
                    for d in (2, 6):
                        hybrid_mask = (np.asarray(box_only_mask) > 0) | cached_yolo_masks[d]
                        outline_leak_cache[(name, f"HYBRID_D{d}")] = outline_leak_metrics(
                            source_detail, source_alpha, body_box_mask, hybrid_mask
                        )

                outline_leak_cache[(name, "BOX_ONLY")] = outline_leak_metrics(source_detail, source_alpha, body_box_mask, box_only_mask)
                outline_leak_cache[(name, "SMART_INK")] = outline_leak_metrics(source_detail, source_alpha, body_box_mask, smart_mask)
                for threshold, candidate_id in zip(contrast_thresholds, contrast_candidate_ids):
                    contrast_mask = make_contrast_mask(
                        source_rgb, source_alpha, anchors_by_image.get(name, []), threshold
                    )
                    outline_leak_cache[(name, candidate_id)] = outline_leak_metrics(
                        source_detail, source_alpha, body_box_mask, contrast_mask
                    )

            # ==================================================================
            # [2/3] & [3/3] BATCH-VERARBEITUNG IN HÄPPCHEN (SCHONT /tmp UND RAM)
            # ==================================================================
            total_images = len(ordered_names)
            num_batches = math.ceil(total_images / batch_size)
            all_report_rows: list[dict[str, Any]] = []
            all_debug_score_rows: list[dict[str, Any]] = []

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

                stage_meta: dict[tuple[str, int], tuple[str, str]] = {}
                hybrid_stage_meta: dict[tuple[str, int], tuple[str, str]] = {}
                contrast_stage_meta: dict[tuple[str, str], tuple[str, str]] = {}
                contrast_masks_by_batch: dict[tuple[str, str], np.ndarray] = {}
                box_stage_meta: dict[str, tuple[str, str]] = {}
                smart_stage_meta: dict[str, tuple[str, str]] = {}

                for name in batch_names:
                    rgb, alpha = rgb_array(source_paths[name])
                    stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in Path(name).stem)

                    if name in candidate_images:
                        for d in distances:
                            staged_name = f"{stem}__d{d:02d}_rgb.png"
                            alpha_name = f"{stem}__d{d:02d}_alpha.png"
                            Image.fromarray(rgb, mode="RGB").save(images_stage / staged_name)
                            Image.fromarray(alpha, mode="L").convert("RGB").save(images_stage / alpha_name)
                            c_mask_path = mask_paths[(name, d)]
                            shutil.copy2(c_mask_path, masks_stage / staged_name)
                            shutil.copy2(c_mask_path, masks_stage / alpha_name)
                            stage_meta[(name, d)] = (staged_name, alpha_name)

                    # Box-Only stagen
                    staged_box = f"{stem}__box_only_rgb.png"
                    alpha_box = f"{stem}__box_only_alpha.png"
                    Image.fromarray(rgb, mode="RGB").save(images_stage / staged_box)
                    Image.fromarray(alpha, mode="L").convert("RGB").save(images_stage / alpha_box)
                    b_path = box_masks_dir / f"{Path(name).stem}.png"
                    shutil.copy2(b_path, masks_stage / staged_box)
                    shutil.copy2(b_path, masks_stage / alpha_box)
                    box_stage_meta[name] = (staged_box, alpha_box)

                    # HYBRID_D2 / HYBRID_D6: full box mask OR distance-clipped YOLO mask.
                    box_mask_for_hybrid = np.asarray(read_box_mask(b_path)) > 0
                    for d in (2, 6):
                        with Image.open(mask_paths[(name, d)]) as mask_im:
                            yolo_mask_for_hybrid = np.asarray(
                                mask_im.convert("L"), dtype=np.uint8
                            ) > 0
                        hybrid_mask = box_mask_for_hybrid | yolo_mask_for_hybrid

                        staged_hybrid = f"{stem}__hybrid_d{d:02d}_rgb.png"
                        alpha_hybrid = f"{stem}__hybrid_d{d:02d}_alpha.png"
                        Image.fromarray(rgb, mode="RGB").save(images_stage / staged_hybrid)
                        Image.fromarray(alpha, mode="L").convert("RGB").save(images_stage / alpha_hybrid)
                        hybrid_mask_image = Image.fromarray(
                            hybrid_mask.astype(np.uint8) * 255, mode="L"
                        )
                        hybrid_mask_image.save(masks_stage / staged_hybrid)
                        hybrid_mask_image.save(masks_stage / alpha_hybrid)
                        hybrid_stage_meta[(name, d)] = (staged_hybrid, alpha_hybrid)

                    # Kontrastmasken stagen: jede Schwelle wird wie die
                    # bestehenden Kandidaten durch LaMa geschickt und danach bewertet.
                    for threshold, candidate_id in zip(contrast_thresholds, contrast_candidate_ids):
                        contrast_mask = make_contrast_mask(
                            rgb, alpha, anchors_by_image.get(name, []), threshold
                        )
                        contrast_masks_by_batch[(name, candidate_id)] = contrast_mask
                        token = candidate_id.lower()
                        staged_contrast = f"{stem}__{token}_rgb.png"
                        alpha_contrast = f"{stem}__{token}_alpha.png"
                        Image.fromarray(rgb, mode="RGB").save(images_stage / staged_contrast)
                        Image.fromarray(alpha, mode="L").convert("RGB").save(images_stage / alpha_contrast)
                        contrast_mask_image = Image.fromarray(
                            contrast_mask.astype(np.uint8) * 255, mode="L"
                        )
                        contrast_mask_image.save(masks_stage / staged_contrast)
                        contrast_mask_image.save(masks_stage / alpha_contrast)
                        contrast_stage_meta[(name, candidate_id)] = (
                            staged_contrast, alpha_contrast
                        )

                    # Smart-Ink stagen
                    smart_mask = make_smart_ink_mask(rgb, anchors_by_image.get(name, []))
                    staged_smart = f"{stem}__smart_ink_rgb.png"
                    alpha_smart = f"{stem}__smart_ink_alpha.png"
                    Image.fromarray(rgb, mode="RGB").save(images_stage / staged_smart)
                    Image.fromarray(alpha, mode="L").convert("RGB").save(images_stage / alpha_smart)
                    Image.fromarray((smart_mask.astype(np.uint8) * 255), mode="L").save(masks_stage / staged_smart)
                    Image.fromarray((smart_mask.astype(np.uint8) * 255), mode="L").save(masks_stage / alpha_smart)
                    smart_stage_meta[name] = (staged_smart, alpha_smart)

                lama_cmd = [iopaint, "run", f"--model={args.model}", f"--device={args.lama_device}",
                            f"--image={images_stage}", f"--mask={masks_stage}", f"--output={raw_stage}"]
                proc = subprocess.run(lama_cmd, cwd=project, check=False)
                if proc.returncode != 0:
                    print(f"ERROR: IOPaint failed in Batch {b_idx + 1} with code {proc.returncode}", file=sys.stderr)
                    return proc.returncode or 1

                for name in batch_names:
                    original_rgb, original_alpha = rgb_array(source_paths[name])
                    box_mask = read_box_mask(box_masks_dir / f"{Path(name).stem}.png")
                    smart_mask = make_smart_ink_mask(original_rgb, anchors_by_image.get(name, []))

                    masks: dict[int, np.ndarray] = {}
                    for d in distances:
                        with Image.open(mask_paths[(name, d)]) as mask_im:
                            masks[d] = np.asarray(mask_im.convert("L"), dtype=np.uint8) > 0

                    # Eigene, echte Hybridkandidaten; nicht nur ein Scoring-Envelope.
                    hybrid_masks: dict[str, np.ndarray] = {
                        f"HYBRID_D{d}": (box_mask | masks[d]) for d in (2, 6)
                    }
                    contrast_masks: dict[str, np.ndarray] = {
                        candidate_id: contrast_masks_by_batch[(name, candidate_id)]
                        for candidate_id in contrast_candidate_ids
                    }
                    yolo_reference_mask = masks[max(distances)]
                    scoring_envelope = yolo_reference_mask | box_mask | smart_mask
                    candidate_ids: list[int | str] = [
                        *distances, "HYBRID_D2", "HYBRID_D6",
                        *contrast_candidate_ids, "SMART_INK", "BOX_ONLY"
                    ]
                    candidate_masks: dict[int | str, np.ndarray] = {
                        **masks,
                        **hybrid_masks,
                        **contrast_masks,
                        "SMART_INK": smart_mask,
                        "BOX_ONLY": box_mask,
                    }

                    body_boxes = [b for b in anchors_by_image.get(name, []) if str(b.get("type", "")).casefold() == "body"]
                    body_box_mask = make_box_only_mask(original_rgb.shape[1], original_rgb.shape[0], body_boxes) > 0

                    scored: list[tuple[float, int | str, np.ndarray, dict[str, float]]] = []
                    debug_results: dict[int | str, np.ndarray] = {}
                    final_composites: dict[int | str, tuple[np.ndarray, np.ndarray]] = {}
                    alpha_postprocess_by_candidate: dict[int | str, dict[str, float]] = {}

                    for cid in candidate_ids:
                        if isinstance(cid, int):
                            s_name, a_name = stage_meta[(name, cid)]
                        elif cid == "HYBRID_D2":
                            s_name, a_name = hybrid_stage_meta[(name, 2)]
                        elif cid == "HYBRID_D6":
                            s_name, a_name = hybrid_stage_meta[(name, 6)]
                        elif cid in contrast_candidate_ids:
                            s_name, a_name = contrast_stage_meta[(name, cid)]
                        elif cid == "BOX_ONLY":
                            s_name, a_name = box_stage_meta[name]
                        else:
                            s_name, a_name = smart_stage_meta[name]

                        with Image.open(raw_stage / s_name) as p_im:
                            pred_rgb = np.asarray(p_im.convert("RGB"), dtype=np.uint8)
                        with Image.open(raw_stage / a_name) as p_im:
                            pred_alpha = np.asarray(p_im.convert("RGB").convert("L"), dtype=np.uint8)

                        c_mask = candidate_masks[cid]
                        f_rgb = original_rgb.copy()
                        f_rgb[c_mask] = pred_rgb[c_mask]
                        f_alpha = original_alpha.copy()
                        f_alpha[c_mask] = pred_alpha[c_mask]
                        f_rgb, f_alpha, a_meta = remove_small_alpha_islands(f_rgb, f_alpha, original_alpha)
                        final_composites[cid] = (f_rgb, f_alpha)
                        alpha_postprocess_by_candidate[cid] = a_meta

                    for cid in candidate_ids:
                        c_rgb, c_alpha = final_composites[cid]
                        c_mask = candidate_masks[cid]
                        metrics = score_candidate(
                            original_rgb, original_alpha, c_rgb, c_alpha,
                            c_mask, scoring_envelope, box_mask, body_box_mask,
                            yolo_reference_mask,
                            outline_leak_precomputed=outline_leak_cache[(name, cid)],
                        )
                        metrics.update(alpha_postprocess_by_candidate[cid])
                        scored.append((metrics["score"], cid, c_mask, metrics))

                        final_rgba = np.dstack((c_rgb, c_alpha)).astype(np.uint8)
                        debug_results[cid] = final_rgba

                        # Cleaned-PNG speichern
                        if save_cleaned:
                            stem = Path(name).stem
                            l_tok = (
                                "smart_ink" if cid == "SMART_INK"
                                else "box_only" if cid == "BOX_ONLY"
                                else str(cid).lower() if isinstance(cid, str)
                                else f"distance_{cid:02d}px"
                            )
                            cl_path = output_dir / "debug_candidates" / "cleaned_images" / f"{stem}__{l_tok}.png"
                            cl_path.parent.mkdir(parents=True, exist_ok=True)
                            Image.fromarray(final_rgba, mode="RGBA").save(cl_path)

                        # Overlays nur speichern, falls explizit gewünscht
                        if save_overlays:
                            stem = Path(name).stem
                            ov = build_debug_mask_overlay(original_rgb, c_mask, anchors_by_image.get(name, []))
                            o_tok = (
                                "smart_ink" if cid == "SMART_INK"
                                else "box_only" if cid == "BOX_ONLY"
                                else str(cid).lower() if isinstance(cid, str)
                                else f"distance_{cid:02d}px"
                            )
                            ov_path = output_dir / "debug_candidates" / "raw_mask_overlays" / f"{stem}__{o_tok}_overlay.png"
                            ov_path.parent.mkdir(parents=True, exist_ok=True)
                            Image.fromarray(ov, mode="RGB").save(ov_path)

                    winner, best_text_score, text_tolerance = select_candidate(scored)
                    best_score, best_candidate_id, best_mask, best_metrics = winner

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
                            winner_suffix = " [WINNER]" if cid == best_candidate_id else ""
                            result_items.append((f"{label} cleaned{winner_suffix}", debug_results[cid]))
                        save_debug_contact_sheet(
                            original_rgba, result_items,
                            output_dir / "debug_candidates" / "contact_sheets" / f"{stem}_cleaned_comparison.jpg"
                        )

                    # Score Zeilen erfassen
                    by_candidate = {cid: m for _s, cid, _m, m in scored}
                    for cid in candidate_ids:
                        m = by_candidate[cid]
                        all_debug_score_rows.append({
                            "image": name, "distance_px": str(cid),
                            **{key: (f"{value:.6f}" if math.isfinite(value) else "") for key, value in m.items()},
                        })

                    winner_is_smart = best_candidate_id == "SMART_INK"
                    winner_is_box = best_candidate_id == "BOX_ONLY"
                    winner_is_hybrid = best_candidate_id in {"HYBRID_D2", "HYBRID_D6"}
                    winner_is_contrast = (
                        isinstance(best_candidate_id, str)
                        and best_candidate_id in contrast_candidate_ids
                    )
                    route_lbl = (
                        "SMART_INK_COMPETITION_SELECTED" if winner_is_smart
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
                        "reason": rsn,
                    })

                    score_text = " | ".join(f"{cid}:text={m['text_quality_score']:.3f},inside={m['residual_inside_box_ratio']:.3f},shape={m['structure_score']:.3f}" for _score, cid, _mask, m in scored)
                    print(f"  {name}: {score_text} -> WINNER {best_candidate_id}")

                # CSVs nach jedem Batch aktualisieren (Crash-Schutz)
                cols = list(dict.fromkeys(key for row in all_report_rows for key in row.keys()))
                with report_path.open("w", encoding="utf-8-sig", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=cols)
                    writer.writeheader()
                    writer.writerows(sorted(all_report_rows, key=lambda r: str(r.get("image", "")).casefold()))

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