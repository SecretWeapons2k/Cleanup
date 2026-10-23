V2 offline calibration using local and repeated-text consensus colour prototypes.
Run directory: /home/secret/Downloads/Cleanup/yolo_mask_winners/contrast_t38_padding4_hard10_20261023_201544
Images requested: h007_000.png, h009_000.png, l002_002.png, m003_009.png, m003_014.png, m004_003.png, m004_004.png, m004_015.png, m005_015.png, m008_012.png
Source detail threshold=0.200; candidate detail threshold=0.120; Lab tolerance=18.0.
Local source prototype minimum=25 pixels; per-anchor source support minimum=10 pixels.
Repeated-text group tolerance=22.0 Lab; minimum inlier anchors=3.
An image receives a residual-only ranking only if combined local/group prototype coverage >= 0.65 and each present anchor type has at least 0.50 coverage.
No prototype or insufficient support is represented as missing/blank, never a zero residual.
Ratios are capped at 1 for rankings; raw matched pixel counts and bounded zone density are retained separately.
The combined index is diagnostic-only and ignores structure. Do not integrate it into production scoring without visual validation.
Overlays: yellow = local per-anchor fill match; magenta = repeated-text consensus match; white = overlap; header boxes cyan, body boxes yellow.
CPU-only. No YOLO/LaMa inference. No production scoring or existing candidate files changed.
