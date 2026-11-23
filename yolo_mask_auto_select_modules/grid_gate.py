"""Grid gate: the grid detector of the project (grid_detector.py), vendored unchanged.

Its thresholds and rejection rules were tuned on the real image set and decide which images get a
cell grid at all (X_ALPHA, X_NOBOX, X_SINGLE, X_CV_W, X_CV_A are exclusions). cells.py uses it as the
authority. Only the debug drawing (process_image / main) was left out; the constants, compute_cv,
estimate_button_count_grid and analyze_grid_from_contours are copied verbatim, so results equal
the stand-alone script. Change thresholds in BOTH places or only here and treat this one as the
source of truth for the pipeline.

    data, reason = analyze_grid_from_contours(alpha, bgra)   # bgra: HxWx4 uint8 in BGR(A) order
    data is None  -> image excluded, ``reason`` says why (e.g. "X_ALPHA:0.83")
    data["category"] in ("grid", "lalpha", "leiste"), data["col_bounds"], data["row_bounds"]
"""
import cv2
import numpy as np

SHOW_DEBUG_DOTS = True  # kept so that analyze_grid_from_contours stays identical to grid_detector.py

# ------------------------------------------------------------------------------
# VARIABLENSTACK / SCHWELLENWERTE
# ------------------------------------------------------------------------------

# Max. Transparenz-Anteil des Bildes (Alpha < 15), bevor das Bild verworfen wird.
# Was es tut: Filtert weitgehend leere oder extrem transparente Bilder raus.
# Bereich: 0.10 bis 0.90 | Standard: 0.60
MAX_TRANSPARENCY_RATIO = 0.60

# Mindestgröße (Breite & Höhe in Pixeln) für eine Bounding-Box, um als gültiges Element zu gelten.
# Was es tut: Filtert kleines Pixelrauschen und einzelne isolierte Artefakte aus.
# Bereich: 4 bis 50 Pixel | Standard: 12
MIN_BOX_DIMENSION = 12

# Mindest-Solidity (Verhältnis von echter Konturfläche zu Bounding-Box-Fläche).
# Was es tut: Filtert hohle Formen, U-Formen und ausgefranste Pinselstrich-Reste (Solidity zu gering).
# Bereich: 0.10 bis 0.70 | Standard: 0.80
MIN_BOX_SOLIDITY = 0.71

# Max. Variationskoeffizient (CV_w = std/mean) der BLAUEN GITTERZELLEN-BREITEN.
# Was es tut: Misst ausschließlich die Gleichmäßigkeit der Spaltenbreiten zwischen den blauen Linien.
# Bereich: 0.05 bis 0.50 | Standard: 0.30
MAX_CV_WIDTH = 0.15

# Max. Variationskoeffizient (CV_a = std/mean) der BLAUEN GITTERZELLEN-FLÄCHEN.
# Was es tut: Misst ausschließlich die Gleichmäßigkeit der durch die blauen Linien aufgespannten Kachelflächen.
# Bereich: 0.10 bis 0.60 | Standard: 0.35
MAX_CV_AREA = 0.35

# Mindest-Füllrate einer Einzelbox (relativer Anteil von Alpha-Pixeln im Rechteck).
# Was es tut: Prüft, ob eine zusammenhängende Box flächig/massiv ist (potenzielle Button-Leiste).
# Bereich: 0.50 bis 0.95 | Standard: 0.80
LEISTE_MIN_FILL_RATIO = 0.80

# Mindest-Seitenverhältnis (Breite / Höhe) für die Einstufung als zusammenhängende Leiste.
# Was es tut: Verhindert, dass quadratische Einzelobjekte fälschlicherweise als Leiste geprüft werden.
# Bereich: 1.5 bis 5.0 | Standard: 2.5
LEISTE_MIN_ASPECT_RATIO = 2.5


def compute_cv(values):
    """Berechnet den Variationskoeffizienten (CV = std / mean).
    Gibt 0.0 zurück, wenn die Liste weniger als 2 Werte enthält oder mean == 0 ist.
    """
    if not values or len(values) < 2:
        return 0.0
    mean_val = float(np.mean(values))
    if mean_val == 0:
        return 0.0
    return float(np.std(values)) / mean_val


def estimate_button_count_grid(roi_alpha, roi_gray=None):
    """Bestimmt N und die genauen Trennstellen durch synchrone Peak-Erkennung 
    (Top * Bottom) unter Einhaltung eines Mindestabstands (min_dist).
    """
    bh, bw = roi_alpha.shape
    target_img = roi_gray if roi_gray is not None else roi_alpha

    margin_h = max(3, int(bh * 0.08))
    top_margin = target_img[:margin_h, :]
    bot_margin = target_img[-margin_h:, :]

    grad_top = np.abs(cv2.Sobel(top_margin, cv2.CV_64F, 1, 0, ksize=3))
    grad_bot = np.abs(cv2.Sobel(bot_margin, cv2.CV_64F, 1, 0, ksize=3))

    raw_top = np.sum(grad_top, axis=0)
    raw_bot = np.sum(grad_bot, axis=0)

    notch_signal = raw_top * raw_bot

    kernel_size = max(3, int(round(bw / 150.0)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    smoothed_signal = cv2.GaussianBlur(notch_signal.reshape(1, -1), (kernel_size, 1), 0).flatten()

    crop_start = int(bw * 0.04)
    crop_end = int(bw * 0.96)
    inner_signal = smoothed_signal[crop_start:crop_end]

    if len(inner_signal) == 0 or np.max(inner_signal) < 1.0:
        return 1, []

    max_val = np.max(inner_signal)
    min_val = np.min(inner_signal)

    threshold = min_val + (max_val - min_val) * 0.25
    min_dist = max(12, int(bh * 0.60))

    peak_x_coords = []
    for x in range(crop_start, crop_end):
        val = smoothed_signal[x]
        if val > threshold:
            w_start = max(0, x - min_dist // 2)
            w_end = min(bw, x + min_dist // 2 + 1)
            
            if val == np.max(smoothed_signal[w_start:w_end]):
                if not peak_x_coords or (x - peak_x_coords[-1]) >= min_dist:
                    peak_x_coords.append(x)

    estimated_segments = len(peak_x_coords) + 1
    return estimated_segments, peak_x_coords


def analyze_grid_from_contours(alpha_channel, cv_img=None):
    h, w = alpha_channel.shape

    # =========================================================================
    # ALPHA-FILTER (Transparenz-Prüfung)
    # =========================================================================
    transparent_pixels = np.count_nonzero(alpha_channel < 15)
    transparent_ratio = transparent_pixels / (w * h)

    if transparent_ratio > MAX_TRANSPARENCY_RATIO:
        return None, f"X_ALPHA:{transparent_ratio:.2f}"

    # Binarisierung & Konturen finden
    _, thresh = cv2.threshold(alpha_channel, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw_contour_count = len(contours)

    valid_boxes = []
    rejected_solidities = []
    for cnt in contours:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bw > MIN_BOX_DIMENSION and bh > MIN_BOX_DIMENSION:
            box_area = bw * bh
            if box_area > 0:
                contour_area = cv2.contourArea(cnt)
                solidity = contour_area / float(box_area)
                if solidity >= MIN_BOX_SOLIDITY:
                    valid_boxes.append((bx, by, bw, bh))
                else:
                    rejected_solidities.append(solidity)

    if len(valid_boxes) == 0:
        if rejected_solidities:
            return None, f"X_NOBOX(Solidity:{max(rejected_solidities):.2f})"
        else:
            return None, f"X_NOBOX(Raw:{raw_contour_count})"

    min_x = min(b[0] for b in valid_boxes)
    min_y = min(b[1] for b in valid_boxes)
    max_x = max(b[0] + b[2] for b in valid_boxes)
    max_y = max(b[1] + b[3] for b in valid_boxes)

    category = None
    col_bounds = []
    row_bounds = []
    strip_segments = 1
    cut_dots = []

    # =========================================================================
    # BLAUE GITTERLINIEN ERZEUGEN
    # =========================================================================
    if len(valid_boxes) == 1:
        bx, by, bw, bh = valid_boxes[0]
        roi_alpha = alpha_channel[by : by + bh, bx : bx + bw]
        fill_ratio = np.count_nonzero(roi_alpha) / (bw * bh)
        is_area_uniform = fill_ratio > LEISTE_MIN_FILL_RATIO
        aspect_ratio = bw / float(bh)

        if is_area_uniform and bw >= LEISTE_MIN_ASPECT_RATIO * bh:
            y_m1 = int(bh * 0.30)
            y_m2 = int(bh * 0.70)
            mid_roi_alpha = roi_alpha[y_m1:y_m2, :]
            col_max_alpha = np.max(mid_roi_alpha, axis=0)

            # Einzelne Box mit Alpha-Lücken (lalpha)
            if np.any(col_max_alpha < 100):
                gap_mask = (col_max_alpha >= 100).astype(np.uint8) * 255
                sub_cnts, _ = cv2.findContours(gap_mask.reshape(1, -1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                sub_boxes = []
                for c in sub_cnts:
                    sx, _, sbw, _ = cv2.boundingRect(c)
                    if sbw > MIN_BOX_DIMENSION:
                        sub_boxes.append((bx + sx, bx + sx + sbw))
                sub_boxes.sort(key=lambda b: b[0])

                if sub_boxes:
                    col_bounds = [sub_boxes[0][0]]
                    for i in range(len(sub_boxes) - 1):
                        gap_mid = (sub_boxes[i][1] + sub_boxes[i+1][0]) // 2
                        col_bounds.append(gap_mid)
                    col_bounds.append(sub_boxes[-1][1])
                    row_bounds = [by, by + bh]
                    category = "lalpha"

            # Topf "leiste" (wenn kein lalpha)
            if category is None:
                roi_bgr = cv_img[by : by + bh, bx : bx + bw, :3] if cv_img is not None else None
                roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY) if roi_bgr is not None else None

                estimated_segments, peak_x_coords = estimate_button_count_grid(roi_alpha, roi_gray)
                strip_segments = estimated_segments

                if estimated_segments == 1:
                    col_bounds = [bx, bx + bw]
                else:
                    col_bounds = [bx] + [bx + px for px in peak_x_coords] + [bx + bw]

                row_bounds = [by, by + bh]
                category = "leiste"

                if SHOW_DEBUG_DOTS and estimated_segments > 1:
                    dot_y = int(by + bh / 2)
                    for cx in col_bounds[1:-1]:
                        cut_dots.append((int(cx), dot_y, True))
        else:
            return None, f"X_SINGLE(F:{fill_ratio:.2f},AR:{aspect_ratio:.1f})"

    else:
        # MEHRERE BOXEN
        cell_w = int(np.median([b[2] for b in valid_boxes]))
        cell_h = int(np.median([b[3] for b in valid_boxes]))

        y_coords = [b[1] for b in valid_boxes]
        is_single_row = (max(y_coords) - min(y_coords)) < cell_h * 0.5

        if is_single_row:
            sorted_boxes = sorted(valid_boxes, key=lambda b: b[0])
            col_bounds = [sorted_boxes[0][0]]
            for i in range(len(sorted_boxes) - 1):
                box_i_end = sorted_boxes[i][0] + sorted_boxes[i][2]
                box_next_start = sorted_boxes[i+1][0]
                gap_mid = (box_i_end + box_next_start) // 2
                col_bounds.append(gap_mid)
            col_bounds.append(sorted_boxes[-1][0] + sorted_boxes[-1][2])
            row_bounds = [min_y, max_y]
            category = "lalpha"

        else:
            # 2D-GRID (Exakte gleichmäßige Aufteilung ohne Rand-Artefakte)
            gap_x_list, gap_y_list = [], []
            for i in range(len(valid_boxes)):
                for j in range(i + 1, len(valid_boxes)):
                    b1, b2 = valid_boxes[i], valid_boxes[j]
                    if abs(b1[1] - b2[1]) < cell_h * 0.4:
                        gx = abs(b2[0] - (b1[0] + b1[2]))
                        if 0 <= gx < cell_w: gap_x_list.append(gx)
                    if abs(b1[0] - b2[0]) < cell_w * 0.4:
                        gy = abs(b2[1] - (b1[1] + b1[3]))
                        if 0 <= gy < cell_h: gap_y_list.append(gy)

            gap_x = int(np.median(gap_x_list)) if gap_x_list else 0
            gap_y = int(np.median(gap_y_list)) if gap_y_list else 0

            span_x = max_x - min_x
            span_y = max_y - min_y

            num_cols = max(1, int(round((span_x + gap_x) / float(cell_w + gap_x))))
            num_rows = max(1, int(round((span_y + gap_y) / float(cell_h + gap_y))))

            step_x = span_x / float(num_cols)
            step_y = span_y / float(num_rows)

            col_bounds = [int(round(min_x + i * step_x)) for i in range(num_cols + 1)]
            row_bounds = [int(round(min_y + j * step_y)) for j in range(num_rows + 1)]

            category = "grid"

    # =========================================================================
    # EXKLUSIVE PRÜFUNG DER BLAUEN GITTERZELLEN (CV_W & CV_A)
    # =========================================================================
    cell_widths = [col_bounds[i+1] - col_bounds[i] for i in range(len(col_bounds) - 1)]
    cell_heights = [row_bounds[j+1] - row_bounds[j] for j in range(len(row_bounds) - 1)]

    # Flächen aller erzeugten blauen Gitterzellen berechnen
    cell_areas = [w_i * h_j for w_i in cell_widths for h_j in cell_heights]

    cv_w = compute_cv(cell_widths)
    cv_a = compute_cv(cell_areas)

    if cv_w > MAX_CV_WIDTH:
        return None, f"X_CV_W:{cv_w:.2f}"
    if cv_a > MAX_CV_AREA:
        return None, f"X_CV_A:{cv_a:.2f}"

    cell_w = int(np.median(cell_widths)) if cell_widths else 0
    cell_h = int(np.median(cell_heights)) if cell_heights else 0

    return {
        "start_x": col_bounds[0], "start_y": row_bounds[0],
        "max_x": col_bounds[-1], "max_y": row_bounds[-1],
        "cell_w": cell_w, "cell_h": cell_h,
        "gap_x": 0, "gap_y": 0,
        "is_strip": category in ["leiste", "lalpha"],
        "strip_segments": strip_segments,
        "category": category,
        "col_bounds": col_bounds,
        "row_bounds": row_bounds,
        "cut_dots": cut_dots,
    }, None
