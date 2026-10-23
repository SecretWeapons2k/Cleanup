Offline calibration of colour residuals from existing cleaned candidate PNGs.
Run directory: /home/secret/Downloads/Cleanup/yolo_mask_winners/contrast_t38_padding4_hard10_20261023_201544
Images requested: h007_000.png, h009_000.png, l002_002.png, m003_009.png, m003_014.png, m004_003.png, m004_004.png, m004_015.png, m005_015.png, m008_012.png
Source detail threshold: 0.200; output detail threshold: 0.120; Lab tolerance: 18.00
Per-anchor color matching uses the same V3 relaxed bright-fill prototypes and largest-cluster alternate.
For each anchor, ratios are matched output pixels divided by number of source detail pixels assigned to the prototype. Values may exceed 1 and are not probabilities.
Experimental dual residual index = max(body bright-fill maximum ratio, header bright-fill maximum ratio, body largest-cluster p90 ratio).
The index is a residual-only diagnostic ranking, NOT a production winner; it intentionally ignores structure quality and has no veto threshold applied.
Inspect per_anchor_dual_color_residuals.csv and candidate_dual_color_summary.csv before considering production changes.
CPU-only. Does not run YOLO/LaMa. Existing candidate images, winner reports, anchors, and scoring code are not modified.
