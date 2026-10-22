"""Image-based quality metrics and two-stage candidate selection."""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from .config import *

def edge_map_lab(rgb: np.ndarray) -> np.ndarray:
    """Build a hue-independent contour map from Lab luminance and chroma."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    return (
        (cv2.Canny(lab[:, :, 0], 40, 120) > 0)
        | (cv2.Canny(lab[:, :, 1], 18, 55) > 0)
        | (cv2.Canny(lab[:, :, 2], 18, 55) > 0)
    )

def _safe_distance_map(edge_map: np.ndarray) -> np.ndarray:
    return cv2.distanceTransform((~edge_map).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

def sprite_contour_metrics(
    original_rgb: np.ndarray,
    final_rgb: np.ndarray,
    original_alpha: np.ndarray,
    final_alpha: np.ndarray,
    candidate_nontext_region: np.ndarray,
    available_region: np.ndarray,
    diagnostics: dict[str, float] | None = None,
) -> tuple[float, float]:
    """Measure contour displacement and optionally expose diagnostics."""

    def record(**values: float) -> None:
        if diagnostics is not None:
            diagnostics.update({
                key: float(value)
                for key, value in values.items()
            })

    diagnostic_keys = {
        "diagnostic_contour_source_point_count": 0.0,
        "diagnostic_contour_final_point_count": 0.0,
        "diagnostic_contour_available_source_edge_count": 0.0,
        "diagnostic_contour_available_final_edge_count": 0.0,
        "diagnostic_contour_source_to_final_ratio": 0.0,
        "diagnostic_contour_final_to_source_ratio": 0.0,
        "diagnostic_contour_source_to_final_capped_px": 0.0,
        "diagnostic_contour_final_to_source_capped_px": 0.0,
        "diagnostic_contour_orientation_matched_count": 0.0,
        "diagnostic_contour_match_radius_px": float(CONTOUR_MATCH_RADIUS_PX),
    }

    if not np.any(candidate_nontext_region):
        record(**diagnostic_keys)
        return 0.0, 0.0

    source_edges = (
        edge_map_lab(original_rgb)
        & (original_alpha > 16)
        & available_region
    )
    final_edges = (
        edge_map_lab(final_rgb)
        & (final_alpha > 16)
        & available_region
    )

    source_points = source_edges & candidate_nontext_region
    final_points = final_edges & candidate_nontext_region

    diagnostic_keys.update({
        "diagnostic_contour_source_point_count":
            float(np.count_nonzero(source_points)),
        "diagnostic_contour_final_point_count":
            float(np.count_nonzero(final_points)),
        "diagnostic_contour_available_source_edge_count":
            float(np.count_nonzero(source_edges)),
        "diagnostic_contour_available_final_edge_count":
            float(np.count_nonzero(final_edges)),
    })

    if not np.any(source_points) and not np.any(final_points):
        record(**diagnostic_keys)
        return 0.0, 0.0

    final_distance = _safe_distance_map(final_edges)
    source_distance = _safe_distance_map(source_edges)
    radius = float(CONTOUR_MATCH_RADIUS_PX)

    source_to_final_px = (
        float(np.mean(np.minimum(final_distance[source_points], radius)))
        if np.any(source_points) else 0.0
    )
    final_to_source_px = (
        float(np.mean(np.minimum(source_distance[final_points], radius)))
        if np.any(final_points) else 0.0
    )

    source_to_final = source_to_final_px / radius
    final_to_source = final_to_source_px / radius

    displacement = float(np.clip(
        0.60 * source_to_final + 0.40 * final_to_source,
        0.0, 1.0,
    ))

    src_gx, src_gy, src_mag = gradient_features(original_rgb)
    fin_gx, fin_gy, fin_mag = gradient_features(final_rgb)

    matched = (
        source_points
        & final_points
        & (src_mag > 50.0)
        & (fin_mag > 50.0)
    )

    if np.any(matched):
        denominator = src_mag[matched] * fin_mag[matched]
        dot = (
            src_gx[matched] * fin_gx[matched]
            + src_gy[matched] * fin_gy[matched]
        )
        orientation_loss = float(np.mean(
            1.0 - np.clip(
                np.abs(dot) / np.maximum(denominator, 1e-6),
                0.0, 1.0,
            )
        ))
    else:
        orientation_loss = 0.0

    diagnostic_keys.update({
        "diagnostic_contour_source_to_final_ratio": source_to_final,
        "diagnostic_contour_final_to_source_ratio": final_to_source,
        "diagnostic_contour_source_to_final_capped_px": source_to_final_px,
        "diagnostic_contour_final_to_source_capped_px": final_to_source_px,
        "diagnostic_contour_orientation_matched_count":
            float(np.count_nonzero(matched)),
    })
    record(**diagnostic_keys)

    return displacement, float(np.clip(orientation_loss, 0.0, 1.0))


def ink_strength_map(rgb: np.ndarray) -> np.ndarray:
    """Return hue-independent local contrast strength in [0, 1]."""
    lab_u8 = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab = lab_u8.astype(np.float32)
    min_dim = min(lab.shape[:2])
    k = min(9, min_dim if min_dim % 2 == 1 else min_dim - 1)
    if k >= 3:
        local = cv2.medianBlur(lab_u8, k).astype(np.float32)
        luminance_delta = np.abs(lab[:, :, 0] - local[:, :, 0])
        da = lab[:, :, 1] - local[:, :, 1]
        db = lab[:, :, 2] - local[:, :, 2]
        chroma_delta = np.sqrt(da * da + db * db)
    else:
        luminance_delta = np.zeros(lab.shape[:2], dtype=np.float32)
        chroma_delta = np.zeros(lab.shape[:2], dtype=np.float32)

    luminance_strength = luminance_delta / 30.0
    chroma_strength = chroma_delta / 24.0
    return np.clip(np.maximum(luminance_strength, chroma_strength), 0.0, 1.0)

def make_smart_ink_mask(
    rgb: np.ndarray,
    boxes: list[dict[str, Any]],
    inset_px: int = SMART_INK_INSET_PX,
    contrast_threshold: float = SMART_INK_CONTRAST_THRESHOLD,
    dilation_radius: int = SMART_INK_DILATION_RADIUS,
) -> np.ndarray:
    """Mask high-contrast text strokes inside OCR boxes while preserving borders and background."""
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    contrast = ink_strength_map(rgb)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilation_radius * 2 + 1, dilation_radius * 2 + 1)
    )

    for b in boxes:
        try:
            x1 = max(0, min(w - 1, int(math.floor(float(b["x1"])))))
            y1 = max(0, min(h - 1, int(math.floor(float(b["y1"])))))
            x2 = max(0, min(w - 1, int(math.ceil(float(b["x2"])))))
            y2 = max(0, min(h - 1, int(math.ceil(float(b["y2"])))))
        except (KeyError, TypeError, ValueError):
            continue
        if x2 <= x1 or y2 <= y1:
            continue

        ix1 = min(x1 + inset_px, x2)
        iy1 = min(y1 + inset_px, y2)
        ix2 = max(x2 - inset_px, x1)
        iy2 = max(y2 - inset_px, y1)

        if ix2 <= ix1 or iy2 <= iy1:
            ix1, iy1, ix2, iy2 = x1, y1, x2, y2

        box_contrast = contrast[iy1:iy2, ix1:ix2]
        text_pixels = (box_contrast >= contrast_threshold).astype(np.uint8) * 255
        if np.any(text_pixels):
            text_closed = cv2.morphologyEx(text_pixels, cv2.MORPH_CLOSE, close_kernel)
            text_dilated = cv2.dilate(text_closed, dilate_kernel, iterations=1)
            mask[iy1:iy2, ix1:ix2] = text_dilated
        else:
            mask[iy1:iy2, ix1:ix2] = 255

    return mask > 0

def gradient_features(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return gx, gy, mag

def boundary_artifact_metrics(
    original_rgb: np.ndarray,
    final_rgb: np.ndarray,
    mask: np.ndarray,
    scoring_domain: np.ndarray,
    ignore_region: np.ndarray,
) -> tuple[float, float]:
    source_lab = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    final_lab = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    src_gx, src_gy, src_mag = gradient_features(original_rgb)
    fin_gx, fin_gy, fin_mag = gradient_features(final_rgb)

    color_penalties: list[np.ndarray] = []
    geometry_penalties: list[np.ndarray] = []
    pairs = [
        ((slice(None), slice(0, -1)), (slice(None), slice(1, None))),
        ((slice(0, -1), slice(None)), (slice(1, None), slice(None))),
    ]

    def orientation_gap(gx_a: np.ndarray, gy_a: np.ndarray,
                        gx_b: np.ndarray, gy_b: np.ndarray) -> np.ndarray:
        ma = np.sqrt(gx_a * gx_a + gy_a * gy_a)
        mb = np.sqrt(gx_b * gx_b + gy_b * gy_b)
        denom = ma * mb
        dot = gx_a * gx_b + gy_a * gy_b
        out = np.zeros_like(denom, dtype=np.float32)
        valid = denom > 25.0
        out[valid] = 1.0 - np.clip(np.abs(dot[valid]) / denom[valid], 0.0, 1.0)
        return out

    for sl_a, sl_b in pairs:
        ma, mb = mask[sl_a], mask[sl_b]
        choose_a = ma & ~mb & scoring_domain[sl_a] & ~ignore_region[sl_a]
        choose_b = mb & ~ma & scoring_domain[sl_b] & ~ignore_region[sl_b]
        if not np.any(choose_a) and not np.any(choose_b):
            continue

        src_la, src_lb = source_lab[sl_a], source_lab[sl_b]
        fin_la, fin_lb = final_lab[sl_a], final_lab[sl_b]
        src_colour_jump = np.linalg.norm(src_la - src_lb, axis=2) / 100.0
        fin_colour_jump = np.linalg.norm(fin_la - fin_lb, axis=2) / 100.0
        colour_increase = np.clip(fin_colour_jump - src_colour_jump, 0.0, 1.0)

        src_mag_a, src_mag_b = src_mag[sl_a], src_mag[sl_b]
        fin_mag_a, fin_mag_b = fin_mag[sl_a], fin_mag[sl_b]
        src_mag_step = np.abs(src_mag_a - src_mag_b) / 1020.0
        fin_mag_step = np.abs(fin_mag_a - fin_mag_b) / 1020.0
        step_increase = np.clip(fin_mag_step - src_mag_step, 0.0, 1.0)

        src_gap = orientation_gap(src_gx[sl_a], src_gy[sl_a], src_gx[sl_b], src_gy[sl_b])
        fin_gap = orientation_gap(fin_gx[sl_a], fin_gy[sl_a], fin_gx[sl_b], fin_gy[sl_b])
        orientation_increase = np.clip(fin_gap - src_gap, 0.0, 1.0)
        geometry_increase = 0.65 * step_increase + 0.35 * orientation_increase

        for choose in (choose_a, choose_b):
            if np.any(choose):
                color_penalties.append(colour_increase[choose])
                geometry_penalties.append(geometry_increase[choose])

    colour_score = float(np.mean(np.concatenate(color_penalties))) if color_penalties else 0.0
    geometry_score = float(np.mean(np.concatenate(geometry_penalties))) if geometry_penalties else 0.0
    return colour_score, geometry_score

def _retained_detail_ratio(
    source_detail: np.ndarray,
    final_detail: np.ndarray,
    region: np.ndarray,
    source_visible: np.ndarray,
    final_visible: np.ndarray,
) -> float:
    src = source_detail * region * source_visible
    fin = final_detail * region * final_visible
    source_energy = float(np.sum(src))
    if source_energy <= 1e-6:
        return 0.0
    same_location = float(np.sum(np.minimum(src, fin))) / source_energy
    detail_density = float(np.sum(fin)) / source_energy
    return float(0.90 * np.clip(same_location, 0.0, 1.0)
                 + 0.10 * np.clip(detail_density, 0.0, 1.0))

def residual_blob_contrast_ratio(
    final_rgb: np.ndarray,
    final_alpha: np.ndarray,
    text_target_region: np.ndarray,
    local_context_region: np.ndarray,
) -> float:
    """Detect low-/mid-frequency residue AND solid dark/bright discolored blocks."""
    visible = np.asarray(final_alpha) > 16
    target = np.asarray(text_target_region, dtype=bool) & visible
    context = np.asarray(local_context_region, dtype=bool) & visible
    if int(np.count_nonzero(target)) < 5:
        return 0.0

    lab = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lum = lab[:, :, 0]
    ca = lab[:, :, 1]
    cb = lab[:, :, 2]

    lum_sigma3 = cv2.GaussianBlur(lum, (0, 0), sigmaX=3.0, sigmaY=3.0)
    lum_sigma12 = cv2.GaussianBlur(lum, (0, 0), sigmaX=12.0, sigmaY=12.0)
    local_luma = np.abs(lum - lum_sigma12) / max(RESIDUAL_BLOB_LUMA_SCALE, 1e-6)
    medium_luma = np.abs(lum_sigma3 - lum_sigma12) / max(RESIDUAL_BLOB_LUMA_SCALE, 1e-6)

    a_sigma3 = cv2.GaussianBlur(ca, (0, 0), sigmaX=3.0, sigmaY=3.0)
    b_sigma3 = cv2.GaussianBlur(cb, (0, 0), sigmaX=3.0, sigmaY=3.0)
    a_sigma12 = cv2.GaussianBlur(ca, (0, 0), sigmaX=12.0, sigmaY=12.0)
    b_sigma12 = cv2.GaussianBlur(cb, (0, 0), sigmaX=12.0, sigmaY=12.0)
    chroma = np.sqrt((a_sigma3 - a_sigma12) ** 2 + (b_sigma3 - b_sigma12) ** 2)
    chroma = chroma / max(RESIDUAL_BLOB_CHROMA_SCALE, 1e-6)

    response = (
        0.40 * np.clip(local_luma, 0.0, 1.0)
        + 0.35 * np.clip(medium_luma, 0.0, 1.0)
        + 0.25 * np.clip(chroma, 0.0, 1.0)
    )
    target_values = response[target]
    target_signal = 0.65 * float(np.mean(target_values)) + 0.35 * float(np.percentile(target_values, 75))

    radius = max(1, int(RESIDUAL_BLOB_RING_RADIUS_PX))
    kernel_size = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated_target = cv2.dilate(target.astype(np.uint8), kernel) > 0
    ring = dilated_target & ~target & context
    if np.any(ring):
        ring_values = response[ring]
        context_signal = 0.65 * float(np.mean(ring_values)) + 0.35 * float(np.percentile(ring_values, 75))

        # DC-Unterschied fängt flächige Verfärbungen / pechschwarze Balken ab:
        target_lum_mean = float(np.mean(lum[target]))
        ring_lum_mean = float(np.mean(lum[ring]))
        lum_dc_shift = abs(target_lum_mean - ring_lum_mean) / max(RESIDUAL_BLOB_LUMA_SCALE, 1e-6)

        target_ca_mean = float(np.mean(ca[target]))
        target_cb_mean = float(np.mean(cb[target]))
        ring_ca_mean = float(np.mean(ca[ring]))
        ring_cb_mean = float(np.mean(cb[ring]))
        chroma_dc_shift = math.sqrt((target_ca_mean - ring_ca_mean) ** 2 + (target_cb_mean - ring_cb_mean) ** 2) / max(RESIDUAL_BLOB_CHROMA_SCALE, 1e-6)

        dc_smudge_penalty = 0.70 * np.clip(lum_dc_shift, 0.0, 1.0) + 0.30 * np.clip(chroma_dc_shift, 0.0, 1.0)
    else:
        context_signal = 0.0
        dc_smudge_penalty = 0.0

    ac_excess = max(0.0, target_signal - context_signal)
    ac_blob = float(np.clip(ac_excess / max(RESIDUAL_BLOB_FULL_SCALE, 1e-6), 0.0, 1.0))
    return max(ac_blob, dc_smudge_penalty)

def residual_micro_component_metrics(
    source_detail: np.ndarray,
    final_detail: np.ndarray,
    text_region: np.ndarray,
    source_visible: np.ndarray,
    final_visible: np.ndarray,
    threshold: float = OUTLINE_LEAK_DETAIL_THRESHOLD,
    max_component_area: int = RESIDUAL_MICRO_COMPONENT_MAX_AREA,
) -> dict[str, float]:
    """Measure tiny high-contrast details retained at source-text locations.

    Aggregate text-detail metrics can dilute isolated small remnants. This
    metric counts connected components that are high-detail in both source
    and candidate, inside the OCR-linked text corridor. Components larger
    than max_component_area remain covered by the existing aggregate metrics.
    """
    target = (
        np.asarray(text_region, dtype=bool)
        & np.asarray(source_visible, dtype=bool)
        & np.asarray(final_visible, dtype=bool)
    )
    retained_components = (
        target
        & (np.asarray(source_detail) >= float(threshold))
        & (np.asarray(final_detail) >= float(threshold))
    )

    if not np.any(retained_components):
        return {
            "residual_micro_component_area": 0.0,
            "residual_micro_component_count": 0.0,
            "residual_micro_component_peak_area": 0.0,
        }

    component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        retained_components.astype(np.uint8),
        connectivity=8,
    )
    if component_count <= 1:
        small_areas = np.empty((0,), dtype=np.int32)
    else:
        areas = stats[1:, cv2.CC_STAT_AREA]
        small_areas = areas[
            (areas > 0) & (areas <= max(1, int(max_component_area)))
        ]

    return {
        "residual_micro_component_area": float(np.sum(small_areas)),
        "residual_micro_component_count": float(small_areas.size),
        "residual_micro_component_peak_area": (
            float(np.max(small_areas)) if small_areas.size else 0.0
        ),
    }


def estimate_ocr_linked_text_corridor(
    source_detail: np.ndarray,
    box_mask: np.ndarray,
    near_ring: np.ndarray,
    yolo_reference_mask: np.ndarray,
    threshold: float = INK_STRENGTH_THRESHOLD,
    extra_padding: int = 2,
    diagnostics: dict[str, float] | None = None,
) -> np.ndarray:
    """Find text-linked detail, with a narrow fallback for OCR boxes lacking linked ink.

    Normal path: use high-contrast connected components linked to the YOLO mask.
    Fallback path: if an individual OCR box has no source-detail energy in that
    corridor, include only thresholded detail components inside that box plus a
    one-pixel edge margin. Do not fall back to the entire OCR box: its other
    pixels may belong to sprite structure that must remain protected.
    """
    detail_binary = (source_detail >= threshold).astype(np.uint8)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    detail_binary = cv2.morphologyEx(detail_binary, cv2.MORPH_CLOSE, close_kernel)
    component_count, labels = cv2.connectedComponents(detail_binary, connectivity=8)

    allowed_region = np.asarray(box_mask, dtype=bool) | np.asarray(near_ring, dtype=bool)
    linked = np.zeros(box_mask.shape, dtype=bool)

    if component_count > 1:
        link_radius = max(1, min(extra_padding, 3))
        link_size = 2 * link_radius + 1
        link_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (link_size, link_size)
        )
        linked_yolo_core = cv2.dilate(
            yolo_reference_mask.astype(np.uint8), link_kernel
        ) > 0
        seed = (detail_binary > 0) & box_mask & linked_yolo_core
        component_labels = np.unique(labels[seed])
        component_labels = component_labels[component_labels != 0]

        if component_labels.size:
            linked = np.isin(labels, component_labels)
            linked &= allowed_region
            if extra_padding > 0:
                size = 2 * extra_padding + 1
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (size, size)
                )
                linked = cv2.dilate(linked.astype(np.uint8), kernel) > 0
                linked &= allowed_region

    # Recover only high-detail pixels inside an OCR box when the YOLO-linked
    # corridor contributes no measurable source-detail energy to that box.
    # The recovery is restricted to the actual detail components and a 1px
    # margin around them, never to the whole rectangular OCR anchor.
    anchor_count, anchor_labels = cv2.connectedComponents(
        np.asarray(box_mask, dtype=np.uint8), connectivity=8
    )
    fallback_mask = np.zeros(box_mask.shape, dtype=bool)
    fallback_anchor_count = 0
    fallback_core_pixels = 0
    padding_px = min(max(int(extra_padding), 0), 1)
    if padding_px:
        pad_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * padding_px + 1, 2 * padding_px + 1)
        )

    for anchor_id in range(1, anchor_count):
        anchor_region = anchor_labels == anchor_id
        current_energy = float(np.sum(source_detail[linked & anchor_region]))
        if current_energy > 1e-6:
            continue

        fallback_core = (detail_binary > 0) & anchor_region
        core_count = int(np.count_nonzero(fallback_core))
        if core_count == 0:
            continue

        fallback_anchor_count += 1
        fallback_core_pixels += core_count

        if padding_px:
            local_allowed = cv2.dilate(
                anchor_region.astype(np.uint8), pad_kernel
            ) > 0
            recovered = cv2.dilate(
                fallback_core.astype(np.uint8), pad_kernel
            ) > 0
            recovered &= local_allowed & allowed_region
        else:
            recovered = fallback_core & allowed_region

        fallback_mask |= recovered

    added_pixels = fallback_mask & ~linked
    linked |= fallback_mask

    if diagnostics is not None:
        diagnostics.update({
            "text_corridor_fallback_anchor_count": float(fallback_anchor_count),
            "text_corridor_fallback_core_pixels": float(fallback_core_pixels),
            "text_corridor_fallback_added_pixels": float(
                np.count_nonzero(added_pixels)
            ),
            "text_corridor_fallback_added_detail_pixels": float(
                np.count_nonzero(added_pixels & (source_detail >= threshold))
            ),
        })

    return linked

def outline_leak_metrics(
    source_detail: np.ndarray,
    source_alpha: np.ndarray,
    body_box_mask: np.ndarray,
    candidate_mask: np.ndarray,
) -> dict[str, float]:
    visible = source_alpha > 16
    body_region = body_box_mask & visible
    leak_region = body_region & ~candidate_mask
    total_energy = float(np.sum(source_detail[body_region]))
    leak_energy = float(np.sum(source_detail[leak_region]))
    leak_ratio = float(np.clip(leak_energy / max(total_energy, 1e-6), 0.0, 1.0)) if total_energy > 1e-6 else 0.0

    high_contrast = source_detail >= OUTLINE_LEAK_DETAIL_THRESHOLD
    leak_pixels = int(np.count_nonzero(leak_region & high_contrast))
    visible_pixels = int(np.count_nonzero(body_region))
    leak_density = float(leak_pixels / max(visible_pixels, 1))
    absolute_component = float(np.clip(
        leak_density / max(OUTLINE_LEAK_ABSOLUTE_DENSITY_FULL_SCALE, 1e-6), 0.0, 1.0
    ))
    score = float(np.clip(0.70 * leak_ratio + 0.30 * absolute_component, 0.0, 1.0))
    return {
        "outline_leak_ratio": leak_ratio,
        "outline_leak_absolute_density": leak_density,
        "outline_leak_score": score,
        "outline_leak_high_contrast_pixels": float(leak_pixels),
    }

def _peak_residual_ratio(
    source_detail: np.ndarray,
    final_detail: np.ndarray,
    region: np.ndarray,
    source_visible: np.ndarray,
    final_visible: np.ndarray,
    threshold: float = 0.20,
    min_pixels: int = 8,
) -> float:
    """Erkennt konzentrierte weiße Schmierer und Schrift-Antialiasing-Reste.
    
    Selbst winzige Cluster aus 10-20 hellen Pixeln auf dunklem Grund ruinieren
    das Bild für das menschliche Auge. Diese Funktion isoliert die intensivsten Reste.
    """
    valid = region & source_visible & final_visible
    if not np.any(valid):
        return 0.0
    src = source_detail[valid]
    fin = final_detail[valid]
    surviving = np.minimum(src, fin)
    
    # Nur Pixel betrachten, an denen im Original tatsächlich Schriftkontrast vorlag
    high_contrast_surviving = surviving[src >= threshold]
    if high_contrast_surviving.size < min_pixels:
        return 0.0
        
    top_count = max(min_pixels, min(32, int(high_contrast_surviving.size * 0.15)))
    top_values = np.partition(high_contrast_surviving, -top_count)[-top_count:]
    peak_mean = float(np.mean(top_values))
    
    if peak_mean <= 0.10:
        return 0.0
    return float(np.clip((peak_mean - 0.10) / 0.30, 0.0, 1.0))


# Lokale Schutzschranke gegen Kandidaten, die einen ganzen
# getrennten Ankerbereich praktisch unverändert stehen lassen.
# Wird nur zur Zulässigkeitsprüfung verwendet, nicht als Strukturkosten.
MAX_ANCHOR_RETAINED_DETAIL_FOR_ELIGIBILITY = 0.50


def _peak_residual_diagnostics(
    source_detail: np.ndarray,
    final_detail: np.ndarray,
    region: np.ndarray,
    source_visible: np.ndarray,
    final_visible: np.ndarray,
    prefix: str,
    threshold: float = 0.20,
    min_pixels: int = 8,
) -> dict[str, float]:
    """Expose peak statistics without changing the production score."""
    valid = region & source_visible & final_visible

    if not np.any(valid):
        return {
            f"{prefix}_raw_mean": 0.0,
            f"{prefix}_score": 0.0,
            f"{prefix}_candidate_pixel_count": 0.0,
            f"{prefix}_top_count": 0.0,
        }

    src = source_detail[valid]
    fin = final_detail[valid]
    surviving = np.minimum(src, fin)

    high_contrast_surviving = surviving[src >= threshold]
    candidate_count = int(high_contrast_surviving.size)

    if candidate_count == 0:
        raw_mean = 0.0
        top_count = 0
        peak_score = 0.0
    elif candidate_count < min_pixels:
        # Zu wenige Pixel für den produktiven Peakscore.
        # Den Mittelwert dennoch diagnostisch ausgeben.
        raw_mean = float(np.mean(high_contrast_surviving))
        top_count = candidate_count
        peak_score = 0.0
    else:
        top_count = max(
            min_pixels,
            min(32, int(candidate_count * 0.15)),
        )
        top_values = np.partition(
            high_contrast_surviving,
            -top_count,
        )[-top_count:]
        raw_mean = float(np.mean(top_values))

        if raw_mean <= 0.10:
            peak_score = 0.0
        else:
            peak_score = float(np.clip(
                (raw_mean - 0.10) / 0.30,
                0.0,
                1.0,
            ))

    return {
        f"{prefix}_raw_mean": raw_mean,
        f"{prefix}_score": peak_score,
        f"{prefix}_candidate_pixel_count": float(candidate_count),
        f"{prefix}_top_count": float(top_count),
    }


def score_candidate(
    original_rgb: np.ndarray,
    original_alpha: np.ndarray,
    final_rgb: np.ndarray,
    final_alpha: np.ndarray,
    mask: np.ndarray,
    scoring_envelope: np.ndarray,
    box_mask: np.ndarray,
    body_box_mask: np.ndarray,
    yolo_reference_mask: np.ndarray,
    outline_leak_precomputed: dict[str, float] | None = None,
) -> dict[str, float]:
    distance_to_box = cv2.distanceTransform((~box_mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    box_expanded = distance_to_box <= 10.000001
    source_visible = original_alpha > 16
    final_visible = final_alpha > 16
    near_ring = box_expanded & ~box_mask

    source_detail = ink_strength_map(original_rgb)
    final_detail = ink_strength_map(final_rgb)

    # Lokale Restdetails getrennt untersuchen.
    # Die einzelnen Rechtecke in box_mask sind räumlich getrennt.
    # Der bisherige globale Mittelwert kann einen komplett
    # unveränderten Bereich durch 19 andere Bereiche verdünnen.
    anchor_count, anchor_labels = cv2.connectedComponents(
        np.asarray(box_mask, dtype=np.uint8),
        connectivity=8,
    )

    anchor_residuals: list[tuple[int, float]] = []

    for anchor_id in range(1, anchor_count):
        anchor_region = anchor_labels == anchor_id
        source_anchor_visible = anchor_region & source_visible

        if not np.any(source_anchor_visible):
            continue

        region_source_energy = float(
            np.sum(source_detail[source_anchor_visible])
        )

        # Bereiche ohne relevante ursprüngliche Detailenergie
        # sollen keine künstlichen Extremwerte erzeugen.
        if region_source_energy <= 1e-6:
            continue

        anchor_residual = _retained_detail_ratio(
            source_detail,
            final_detail,
            anchor_region,
            source_visible,
            final_visible,
        )

        anchor_residuals.append(
            (
                anchor_id,
                float(np.clip(anchor_residual, 0.0, 1.0)),
            )
        )

    if anchor_residuals:
        max_anchor_id, max_anchor_retention = max(
            anchor_residuals,
            key=lambda entry: entry[1],
        )
    else:
        max_anchor_id = 0
        max_anchor_retention = 0.0

    corridor_fallback_diagnostics: dict[str, float] = {}
    text_corridor = estimate_ocr_linked_text_corridor(
        source_detail, box_mask, near_ring, yolo_reference_mask,
        extra_padding=TEXT_CORRIDOR_DILATION_PX,
        diagnostics=corridor_fallback_diagnostics,
    )
    if not np.any(text_corridor):
        fallback_target = box_mask & (source_detail >= OUTLINE_LEAK_DETAIL_THRESHOLD) & source_visible
        if np.any(fallback_target):
            fallback_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            text_corridor = cv2.dilate(fallback_target.astype(np.uint8), fallback_kernel) > 0
            text_corridor &= box_expanded

    # Textbezogene lokale Restprüfung:
    # Anders als max_anchor_retained_detail_ratio berücksichtigt
    # dieser Wert ausschließlich den ermittelten Textkorridor.
    # Dadurch sollen erhaltene Haare und Sprite-Kanten nicht
    # automatisch als verbliebene Schrift gelten.
    # Bereits vorhandene Detailwerte pro Anker als Rückfalloption.
    # Sie werden nur verwendet, wenn im jeweiligen Anker kein
    # ausreichend großer Textkorridor ausgewertet werden kann.
    anchor_residual_lookup = {
        int(anchor_id): float(retention)
        for anchor_id, retention in anchor_residuals
    }

    text_anchor_residuals: list[tuple[int, float]] = []
    text_anchor_fallback_region_count = 0

    for anchor_id in range(1, anchor_count):
        anchor_region = anchor_labels == anchor_id
        text_region = anchor_region & text_corridor
        visible_text_region = text_region & source_visible

        visible_text_count = int(
            np.count_nonzero(visible_text_region)
        )
        source_text_energy = float(
            np.sum(source_detail[visible_text_region])
        )

        if (
            visible_text_count >= 5
            and source_text_energy > 1e-6
        ):
            # Normalfall: ausschließlich den ermittelten Textkorridor
            # zur Beurteilung der verbliebenen Details verwenden.
            text_anchor_residual = _retained_detail_ratio(
                source_detail,
                final_detail,
                text_region,
                source_visible,
                final_visible,
            )
        else:
            # Kein auswertbarer Textkorridor in diesem Anker:
            # auf den bereits berechneten allgemeinen Detailwert
            # genau dieses Ankers zurückgreifen.
            fallback_ratio = anchor_residual_lookup.get(anchor_id)

            if fallback_ratio is None:
                continue

            text_anchor_residual = fallback_ratio
            text_anchor_fallback_region_count += 1

        text_anchor_residuals.append(
            (
                anchor_id,
                float(np.clip(text_anchor_residual, 0.0, 1.0)),
            )
        )

    if text_anchor_residuals:
        max_text_anchor_id, max_text_anchor_retention = max(
            text_anchor_residuals,
            key=lambda entry: entry[1],
        )
    else:
        max_text_anchor_id = 0
        max_text_anchor_retention = 0.0

    # --- REGEL 1: PEAK-SCHMIERER ERKENNUNG IM KASTEN ---
    inside_energy = _retained_detail_ratio(source_detail, final_detail, box_mask, source_visible, final_visible)
    inside_peak = _peak_residual_ratio(source_detail, final_detail, box_mask, source_visible, final_visible)
    # Kombiniere Gesamtenergie mit dem Spitzenwert (Schmierer schlagen voll durch!)
    inside_residual = float(np.clip(0.55 * inside_energy + 0.45 * inside_peak, 0.0, 1.0))

    # DIAGNOSTIK: dieselben Kennzahlen zusätzlich nur im Textkorridor
    # innerhalb der Box berechnen. Diese Werte verändern die Auswahl
    # noch nicht; sie dienen dem Vergleich mit der bisherigen Boxwertung.
    corridor_inside_region = box_mask & text_corridor
    corridor_source_pixels = corridor_inside_region & source_visible
    corridor_inside_pixel_count = int(
        np.count_nonzero(corridor_source_pixels)
    )
    corridor_source_energy = float(
        np.sum(source_detail[corridor_source_pixels])
    )


    # DIAGNOSTIK: Stärke und Anzahl der Peak-Pixel getrennt messen.
    # Die produktiven inside_peak- und Textscore-Berechnungen bleiben
    # unverändert.
    peak_diagnostics: dict[str, float] = {}

    peak_diagnostics.update(_peak_residual_diagnostics(
        source_detail,
        final_detail,
        box_mask,
        source_visible,
        final_visible,
        prefix="diagnostic_box_peak",
    ))

    peak_diagnostics.update(_peak_residual_diagnostics(
        source_detail,
        final_detail,
        corridor_inside_region,
        source_visible,
        final_visible,
        prefix="diagnostic_text_corridor_peak",
    ))

    if (
        corridor_inside_pixel_count >= 5
        and corridor_source_energy > 1e-6
    ):
        corridor_inside_energy = _retained_detail_ratio(
            source_detail,
            final_detail,
            corridor_inside_region,
            source_visible,
            final_visible,
        )
        corridor_inside_peak = _peak_residual_ratio(
            source_detail,
            final_detail,
            corridor_inside_region,
            source_visible,
            final_visible,
        )
        corridor_inside_residual = float(np.clip(
            0.55 * corridor_inside_energy
            + 0.45 * corridor_inside_peak,
            0.0,
            1.0,
        ))
        corridor_inside_fallback_used = 0.0
    else:
        # Bei zu wenig auswertbarem Textkorridor auf die bisherige
        # Boxwertung zurückfallen, aber diesen Fall explizit markieren.
        corridor_inside_energy = inside_energy
        corridor_inside_peak = inside_peak
        corridor_inside_residual = inside_residual
        corridor_inside_fallback_used = 1.0

    micro_component_metrics = residual_micro_component_metrics(
        source_detail,
        final_detail,
        box_expanded & text_corridor,
        source_visible,
        final_visible,
    )

    halo_text_region = near_ring & text_corridor & ~box_mask
    if np.any(halo_text_region):
        halo_energy = _retained_detail_ratio(source_detail, final_detail, halo_text_region, source_visible, final_visible)
        halo_peak = _peak_residual_ratio(source_detail, final_detail, halo_text_region, source_visible, final_visible)
        halo_residual = float(np.clip(0.55 * halo_energy + 0.45 * halo_peak, 0.0, 1.0))
        residual_text_detail = float(np.clip(0.85 * inside_residual + 0.15 * halo_residual, 0.0, 1.0))
    else:
        halo_residual = 0.0
        residual_text_detail = float(np.clip(inside_residual, 0.0, 1.0))

    residual_blob = residual_blob_contrast_ratio(
        final_rgb, final_alpha, text_corridor, box_expanded & source_visible
    )
    visual_text_residual = float(np.clip(
        (1.0 - RESIDUAL_BLOB_VISUAL_WEIGHT) * residual_text_detail
        + RESIDUAL_BLOB_VISUAL_WEIGHT * residual_blob,
        0.0, 1.0,
    ))

    leak = (
        outline_leak_precomputed
        if outline_leak_precomputed is not None
        else outline_leak_metrics(source_detail, original_alpha, body_box_mask, mask)
    )
    base_text_quality = visual_text_residual
    text_quality_score = float(np.clip(
        (1.0 - OUTLINE_LEAK_TEXT_WEIGHT) * base_text_quality
        + OUTLINE_LEAK_TEXT_WEIGHT * leak["outline_leak_score"], 0.0, 1.0
    ))

    # Alternative Textbewertung für den Vergleich:
    # Nur der Inside-Anteil wird auf den Textkorridor eingeschränkt.
    # Halo, Blob-Metrik und Leak-Gewichtung bleiben gleich.
    if np.any(halo_text_region):
        corridor_residual_text_detail = float(np.clip(
            0.85 * corridor_inside_residual
            + 0.15 * halo_residual,
            0.0,
            1.0,
        ))
    else:
        corridor_residual_text_detail = corridor_inside_residual

    corridor_visual_text_residual = float(np.clip(
        (1.0 - RESIDUAL_BLOB_VISUAL_WEIGHT)
        * corridor_residual_text_detail
        + RESIDUAL_BLOB_VISUAL_WEIGHT * residual_blob,
        0.0,
        1.0,
    ))

    corridor_base_text_quality = corridor_visual_text_residual

    corridor_text_quality_score = float(np.clip(
        (1.0 - OUTLINE_LEAK_TEXT_WEIGHT) * corridor_base_text_quality
        + OUTLINE_LEAK_TEXT_WEIGHT * leak["outline_leak_score"],
        0.0,
        1.0,
    ))

    available_nontext_region = box_expanded & source_visible & ~text_corridor
    candidate_nontext_region = mask & source_visible & ~text_corridor

    contour_debug: dict[str, float] = {}
    candidate_contour_displacement, contour_orientation_loss = sprite_contour_metrics(
        original_rgb, final_rgb, original_alpha, final_alpha,
        candidate_nontext_region, available_nontext_region,
        diagnostics=contour_debug,
    )

    # Gemeinsame Konturwertung: Alle Kandidaten im selben räumlichen Bereich.
    # Die kandidatenabhängige Konturwertung bleibt als Diagnosewert erhalten.
    common_contour_region = (
        np.asarray(scoring_envelope, dtype=bool)
        & available_nontext_region
    )

    common_contour_debug: dict[str, float] = {}
    common_contour_displacement, common_contour_orientation_loss = (
        sprite_contour_metrics(
            original_rgb,
            final_rgb,
            original_alpha,
            final_alpha,
            common_contour_region,
            common_contour_region,
            diagnostics=common_contour_debug,
        )
    )

    common_contour_components: dict[str, float] = {
        "diagnostic_common_contour_displacement_ratio":
            common_contour_displacement,
        "diagnostic_common_contour_orientation_loss_ratio":
            common_contour_orientation_loss,
        **{
            key.replace(
                "diagnostic_contour_",
                "diagnostic_common_contour_",
                1,
            ): value
            for key, value in common_contour_debug.items()
        },
    }


    # Produktive Konturverschiebung auf gemeinsamer Vergleichsregion.
    # Die Orientierungskomponente bleibt unverändert.
    contour_displacement = common_contour_displacement

    _src_gx, _src_gy, src_mag = gradient_features(original_rgb)
    _fin_gx, _fin_gy, fin_mag = gradient_features(final_rgb)
    source_edge = np.clip(src_mag / 1020.0, 0.0, 1.0)
    final_edge = np.clip(fin_mag / 1020.0, 0.0, 1.0)
    source_texture = cv2.GaussianBlur(source_edge, (0, 0), sigmaX=2.0, sigmaY=2.0)
    final_texture = cv2.GaussianBlur(final_edge, (0, 0), sigmaX=2.0, sigmaY=2.0)
    texture_delta = np.abs(source_texture - final_texture)
    fixed_region = scoring_envelope & available_nontext_region
    texture_denominator = max(
        float(np.sum(source_texture[fixed_region])),
        0.01 * int(np.count_nonzero(fixed_region)), 1e-6
    )
    texture_distortion = float(np.clip(
        np.sum(texture_delta[candidate_nontext_region]) / texture_denominator, 0.0, 1.0
    ))

    # --- REGEL 3A: PEAK COLOR-SHIFT (FARBFLECKEN-ERKENNUNG z.B. h008_001) ---
    source_lab = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    final_lab = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    delta_e = np.linalg.norm(source_lab - final_lab, axis=2)
    colour_change = np.clip(delta_e / 100.0, 0.0, 1.0)
    fixed_pixel_denominator = max(int(np.count_nonzero(fixed_region)), 1)
    avg_colour_shift = float(np.clip(
        np.sum(colour_change[candidate_nontext_region]) / fixed_pixel_denominator, 0.0, 1.0
    ))
    
    if np.any(candidate_nontext_region):
        shifts = colour_change[candidate_nontext_region]
        bad_shifts = shifts[shifts >= 0.20]  # Lokale Farbsprünge ab Delta-E 20
        if bad_shifts.size >= 8:
            k = min(24, bad_shifts.size)
            peak_color_shift = float(np.mean(np.partition(bad_shifts, -k)[-k:]))
        else:
            peak_color_shift = 0.0
    else:
        peak_color_shift = 0.0
    sprite_colour_shift = float(np.clip(0.60 * avg_colour_shift + 0.40 * peak_color_shift, 0.0, 1.0))

    boundary_colour, boundary_geometry = boundary_artifact_metrics(
        original_rgb, final_rgb, mask, box_expanded & source_visible, text_corridor
    )

    # --- REGEL 3B: PFEIL- & UI-SCHUTZ AUSSERHALB DER BOX (z.B. d002_000) ---
    # Prüft, ob außerhalb der Box liegende Nicht-Text-Kanten von der Maske rasiert wurden
    external_masked = mask & ~box_mask & ~text_corridor & source_visible
    source_external_edges = edge_map_lab(original_rgb) & external_masked
    final_external_edges = edge_map_lab(final_rgb) & external_masked
    erased_external_edges = int(np.count_nonzero(source_external_edges & ~final_external_edges))
    external_edge_destruction_score = float(np.clip(erased_external_edges / 12.0, 0.0, 1.0))

    components: dict[str, float] = {
        "residual_text_detail_ratio": residual_text_detail,
        "residual_inside_peak": float(inside_peak),
        "residual_inside_energy": float(inside_energy),
        **peak_diagnostics,
        "residual_blob_contrast_ratio": residual_blob,
        "visual_text_residual_score": visual_text_residual,
        "text_quality_base_score": float(np.clip(base_text_quality, 0.0, 1.0)),
        "text_quality_score": text_quality_score,
        **leak,
        **micro_component_metrics,
        **corridor_fallback_diagnostics,
        **contour_debug,
        **common_contour_components,
        "diagnostic_candidate_contour_displacement_ratio":
            candidate_contour_displacement,
        "sprite_contour_displacement_ratio": contour_displacement,
        "sprite_contour_orientation_loss_ratio": contour_orientation_loss,
        "sprite_texture_distortion_ratio": texture_distortion,
        "sprite_color_shift_ratio": sprite_colour_shift,
        "peak_color_shift_ratio": peak_color_shift,
        "boundary_color_artifact_ratio": float(np.clip(boundary_colour, 0.0, 1.0)),
        "boundary_geometry_artifact_ratio": float(np.clip(boundary_geometry, 0.0, 1.0)),
        "external_edge_destruction_score": external_edge_destruction_score,
        "residual_inside_box_ratio": float(np.clip(inside_residual, 0.0, 1.0)),
        "residual_inside_text_corridor_energy": float(corridor_inside_energy),
        "residual_inside_text_corridor_peak": float(corridor_inside_peak),
        "residual_inside_text_corridor_ratio": float(corridor_inside_residual),
        "text_corridor_inside_pixel_count": float(corridor_inside_pixel_count),
        "text_corridor_inside_fallback_used": float(corridor_inside_fallback_used),
        "text_quality_score_corridor_test": float(corridor_text_quality_score),
        "residual_near_box_ratio": float(np.clip(halo_residual, 0.0, 1.0)),
        "max_anchor_retained_detail_ratio": float(max_anchor_retention),
        "max_anchor_retained_detail_region": float(max_anchor_id),
        "anchor_region_count": float(len(anchor_residuals)),
        "max_text_anchor_retained_detail_ratio": float(max_text_anchor_retention),
        "max_text_anchor_retained_detail_region": float(max_text_anchor_id),
        "text_anchor_region_count": float(len(text_anchor_residuals)),
        "text_anchor_fallback_region_count": float(
            text_anchor_fallback_region_count
        ),
    }
    structure_score = sum(STRUCTURE_WEIGHTS[key] * components[key] for key in STRUCTURE_WEIGHTS)
    components["structure_score"] = float(structure_score)
    components["score"] = float(structure_score)
    for key, weight in STRUCTURE_WEIGHTS.items():
        components[f"weighted_{key}"] = weight * components[key]
    components["weighted_outline_leak_score"] = OUTLINE_LEAK_TEXT_WEIGHT * components["outline_leak_score"]
    return components

def select_candidate(
    scored: list[tuple[float, int | str, np.ndarray, dict[str, float]]],
) -> tuple[tuple[float, int | str, np.ndarray, dict[str, float]], float, float]:
    """Wählt den besten Kandidaten mit fehlerresistenter Box-Only-Rescue."""
    if not scored:
        raise ValueError("No scored mask candidates")

    text_scores = [item[3]["text_quality_score"] for item in scored]
    best_text = min(text_scores)
    spread = max(text_scores) - best_text
    tolerance = float(np.clip(
        TEXT_SCORE_TOLERANCE_RANGE_FRACTION * spread,
        TEXT_SCORE_TOLERANCE_FLOOR,
        TEXT_SCORE_TOLERANCE_CEILING,
    ))
    ceiling = best_text + tolerance

    # Zusätzliche lokale Schranke:
    # Ein Kandidat mit einem stark verbliebenen Einzelbereich
    # darf nicht allein wegen eines guten Gesamtdurchschnitts gewinnen.
    anchor_ratios = [
        float(item[3].get(
            "max_text_anchor_retained_detail_ratio",
            item[3].get("max_anchor_retained_detail_ratio", 0.0),
        ))
        for item in scored
    ]

    text_quality_flags = [
        float(item[3]["text_quality_score"]) <= ceiling + 1e-9
        for item in scored
    ]

    # Ein "sauberer" Kandidat zählt für den absoluten Grenzwert nur,
    # wenn er auch die globale Textqualitätsgrenze erfüllt.
    has_anchor_clean_candidate = any(
        text_ok
        and ratio <= MAX_ANCHOR_RETAINED_DETAIL_FOR_ELIGIBILITY + 1e-9
        for ratio, text_ok in zip(anchor_ratios, text_quality_flags)
    )

    if has_anchor_clean_candidate:
        anchor_eligibility_limit = (
            MAX_ANCHOR_RETAINED_DETAIL_FOR_ELIGIBILITY
        )
    else:
        # Wenn kein global textqualitativ zulässiger Kandidat den
        # absoluten Grenzwert erreicht, dient der beste lokale Wert
        # innerhalb der textqualitativ zulässigen Kandidaten als
        # Referenz. Dadurch wird ein Kandidat wie BOX_ONLY nicht
        # allein wegen eines geringfügig höheren Detailrests gesperrt.
        text_eligible_anchor_ratios = [
            ratio
            for ratio, text_ok in zip(anchor_ratios, text_quality_flags)
            if text_ok
        ]

        reference_ratios = (
            text_eligible_anchor_ratios
            if text_eligible_anchor_ratios
            else anchor_ratios
        )

        anchor_eligibility_limit = min(
            1.0,
            min(reference_ratios) + 0.02,
        )

    # Die gemeinsame Zulässigkeitsprüfung muss dieselbe lokale Metrik
    # benutzen wie die anschließende Prüfung jedes Kandidaten.
    joint_eligible_exists = any(
        text_ok
        and ratio <= anchor_eligibility_limit + 1e-9
        for ratio, text_ok in zip(anchor_ratios, text_quality_flags)
    )

    # Die Textqualitätsgrenze wird nur dann übergangen, wenn tatsächlich
    # kein Kandidat beide Bedingungen erfüllt.
    anchor_fallback_mode = not joint_eligible_exists

    box_item = next((item for item in scored if item[1] == "BOX_ONLY"), None)
    yolo_items = [item for item in scored if isinstance(item[1], int)]

    box_rescue = False
    rescue_advantage = 0.0
    rescue_yolo_leak = 0.0
    rescue_reason = 0.0
    rescue_inside_advantage = 0.0
    rescue_box_inside_residual = float("nan")
    rescue_best_yolo_inside_residual = float("nan")

    if BOX_ONLY_RESCUE_ENABLED and box_item is not None and yolo_items:
        best_yolo_by_text = min(yolo_items, key=lambda item: item[3]["text_quality_score"])
        # Für die Artefaktreferenz alle Alternativen berücksichtigen,
        # einschließlich HYBRID_D2, HYBRID_D6 und SMART_INK.
        # Die separate YOLO-Text-/Inside-Referenz bleibt unverändert.
        best_non_box_by_structure = min(
            (item for item in scored if item[1] != "BOX_ONLY"),
            key=lambda item: item[3]["structure_score"],
        )
        best_yolo_text_score = float(best_yolo_by_text[3]["text_quality_score"])
        box_text_score = float(box_item[3]["text_quality_score"])
        
        # 1. SCHRANKE: Wenn YOLO bereits absolut sauber ist (< 0.03), NIEMALS retten!
        yolo_already_clean = best_yolo_text_score <= 0.030

        # 2. SCHRANKE: Wenn Box-Only selbst Halluzinationen/Schmierer erzeugt, NIEMALS retten!
        # Strukturkosten entscheiden weiterhin über die Rangfolge,
        # sind aber keine eigenständige Artefaktprüfung.
        # Der Farbwechsel wird relativ zum besten YOLO-Textkandidaten
        # verglichen, damit ein allen Kandidaten gemeinsamer hoher Wert
        # BOX_ONLY nicht pauschal blockiert.
        box_peak_shift = float(
            box_item[3].get("peak_color_shift_ratio", 0.0)
        )
        # Peak-Farbverschiebung mit dem strukturell besten YOLO-
        # Kandidaten vergleichen. Der Text-Benchmark bleibt getrennt.
        comparator_peak_shift = float(
            best_non_box_by_structure[3].get(
                "peak_color_shift_ratio", box_peak_shift
            )
        )

        box_has_artifacts = (
            box_peak_shift > comparator_peak_shift + 0.15
            or box_item[3].get("residual_blob_contrast_ratio", 0.0) > 0.10
            or float(
                box_item[3].get(
                    "max_text_anchor_retained_detail_ratio",
                    box_item[3].get("max_anchor_retained_detail_ratio", 0.0),
                )
            ) > anchor_eligibility_limit + 1e-9
        )

        if not yolo_already_clean and not box_has_artifacts:
            rescue_advantage = max(0.0, best_yolo_text_score - box_text_score)
            
            best_yolo_by_inside = min(yolo_items, key=lambda item: item[3].get("residual_inside_box_ratio", 1.0))
            box_inside = float(box_item[3].get("residual_inside_box_ratio", 1.0))
            yolo_inside = float(best_yolo_by_inside[3].get("residual_inside_box_ratio", 1.0))
            rescue_inside_advantage = max(0.0, yolo_inside - box_inside)
            rescue_yolo_leak = float(best_yolo_by_text[3].get("outline_leak_score", 0.0))

            # Reale Schwellenwerte statt Rausch-Empfindlichkeit (0.02 statt 0.006)
            if rescue_advantage >= 0.020:
                box_rescue = True
                rescue_reason = 2.0
            elif rescue_advantage >= 0.010 and rescue_yolo_leak >= BOX_ONLY_RESCUE_MIN_YOLO_LEAK_SCORE:
                box_rescue = True
                rescue_reason = 1.0
            elif yolo_inside >= 0.12 and rescue_inside_advantage >= 0.04:
                box_rescue = True
                rescue_reason = 3.0
            elif (
                rescue_advantage >= 0.015
                and rescue_inside_advantage >= 0.025
            ):
                # Zusätzliche relative Regel:
                # BOX_ONLY muss gleichzeitig beim Textscore und beim
                # Inside-Residual merklich besser sein.
                box_rescue = True
                rescue_reason = 4.0

    for _score, candidate_id, _mask, metrics in scored:
        text_eligible = (
            metrics["text_quality_score"] <= ceiling + 1e-9
        )

        anchor_eligible = (
            float(metrics.get(
                "max_text_anchor_retained_detail_ratio",
                metrics.get("max_anchor_retained_detail_ratio", 0.0),
            )) <= anchor_eligibility_limit + 1e-9
        )

        default_eligible = anchor_eligible and (
            text_eligible or anchor_fallback_mode
        )

        # Eine evidenzbasierte BOX_ONLY-Rescue darf die globale Textscore-Grenze
        # übergehen. Die lokale Anker-/Detail-Schranke bleibt weiterhin aktiv.
        if box_rescue:
            eligible = candidate_id == "BOX_ONLY" and anchor_eligible
        else:
            eligible = default_eligible

        metrics["anchor_detail_eligible"] = (
            1.0 if anchor_eligible else 0.0
        )
        metrics["anchor_detail_eligibility_limit"] = float(
            anchor_eligibility_limit
        )
        metrics["anchor_detail_fallback_mode"] = (
            1.0 if anchor_fallback_mode else 0.0
        )

        metrics["best_text_quality_score"] = float(best_text)
        metrics["text_quality_tolerance"] = tolerance
        metrics["text_quality_eligible"] = 1.0 if eligible else 0.0
        metrics["box_only_rescue_applied"] = 1.0 if box_rescue else 0.0
        metrics["box_only_rescue_final_text_advantage"] = rescue_advantage
        metrics["box_only_rescue_best_yolo_leak_score"] = rescue_yolo_leak
        metrics["box_only_rescue_inside_detail_advantage"] = rescue_inside_advantage
        metrics["box_only_rescue_box_inside_residual"] = rescue_box_inside_residual
        metrics["box_only_rescue_best_yolo_inside_residual"] = rescue_best_yolo_inside_residual
        metrics["box_only_rescue_reason_code"] = rescue_reason
        metrics["residual_spot_tiebreak_applied"] = 0.0
        metrics["residual_spot_tiebreak_structure_margin"] = 0.0
        metrics["residual_spot_tiebreak_area_reduction"] = 0.0
        metrics["score"] = metrics["structure_score"] if eligible else 2.0 + metrics["structure_score"]

    def tie_break(candidate_id: int | str) -> tuple[int, int]:
        if isinstance(candidate_id, int):
            return (0, candidate_id)
        if candidate_id == "SMART_INK":
            return (1, 0)
        return (2, 9999)

    winner = min(scored, key=lambda item: (
        item[3]["score"], item[3]["structure_score"], item[3]["text_quality_score"], tie_break(item[1])
    ))
    # Konservative Zweitentscheidung: Bei nahezu gleichem Struktur-Score
    # darf ein Kandidat mit deutlich weniger kleinen Textresten gewinnen.
    baseline_winner = winner
    baseline_structure = float(baseline_winner[3]["structure_score"])
    baseline_micro_area = float(
        baseline_winner[3].get("residual_micro_component_area", 0.0)
    )
    near_optimal = [
        item
        for item in scored
        if float(item[3].get("score", 2.0)) < 2.0
        and float(item[3]["structure_score"])
        <= baseline_structure + RESIDUAL_SPOT_STRUCTURE_TOLERANCE + 1e-9
    ]
    if near_optimal:
        cleanest_area = min(
            float(item[3].get("residual_micro_component_area", 0.0))
            for item in near_optimal
        )
        area_reduction = baseline_micro_area - cleanest_area
        if area_reduction >= RESIDUAL_SPOT_MIN_AREA_REDUCTION_PX:
            cleaner_candidates = [
                item
                for item in near_optimal
                if float(item[3].get("residual_micro_component_area", 0.0))
                == cleanest_area
            ]
            cleaner_winner = min(cleaner_candidates, key=lambda item: (
                item[3]["structure_score"],
                item[3]["text_quality_score"],
                tie_break(item[1]),
            ))
            if cleaner_winner[1] != baseline_winner[1]:
                cleaner_winner[3]["residual_spot_tiebreak_applied"] = 1.0
                cleaner_winner[3]["residual_spot_tiebreak_structure_margin"] = max(
                    0.0,
                    float(cleaner_winner[3]["structure_score"]) - baseline_structure,
                )
                cleaner_winner[3]["residual_spot_tiebreak_area_reduction"] = float(
                    area_reduction
                )
                winner = cleaner_winner


    return winner, float(best_text), tolerance