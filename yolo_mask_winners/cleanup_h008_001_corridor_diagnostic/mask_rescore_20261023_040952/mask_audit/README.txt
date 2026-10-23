YOLO mask routing audit (inference only; no training and no LaMa).
Images source: /home/secret/Downloads/yabai proj/yabaiproject - Copy/mit_kanji
Anchors CSV: /home/secret/Downloads/Cleanup/yolo_mask_winners/cleanup_h008_001_corridor_diagnostic/mask_rescore_20261023_040952/anchors_h008_001.csv
Weights: /home/secret/.cache/cleanup-models/manga-inpainter-best.pt
Images processed: 1
Distance sweep: [0, 2, 4, 6, 8, 10]; dilation: 2px
Body-box meaningful coverage thresholds: overlap >= 12px / 0.20% of visible box; detail energy >= 35%; high-contrast pixels >= 30% when present.
Route rule: every type=body box must have a non-trivial YOLO overlap and sufficient coverage of source local-contrast energy and high-contrast pixels; one-pixel hits do not count. Weak coverage routes to BOX_MASK_FALLBACK.
Proximity rule: pixel-wise trim each eligible YOLO mask. Keep only mask pixels whose distance to the nearest anchor-box pixel is <= the configured limit. Always union header rectangles in full.
This clips distant parts of a mask, even when another part of the same instance is close to a box; instances are not accepted/rejected as indivisible objects.
Use routing_report.csv for per-image summary, body_box_coverage.csv for raw overlap and detail-coverage ratios per body box, instances.csv for per-instance confidence/minimum distance, and contact sheets to visually tune clipping.
This is a heuristic report; the best distance must be chosen from the contact sheets and LaMa results, not inferred from overlap statistics alone.
Never modified source images, anchors.csv, raw/, cleaned/, or existing audit results.

0px: box fallback 0/1, mean instances with pixels after clip 75.00, mean union pixels clipped 9805
2px: box fallback 0/1, mean instances with pixels after clip 75.00, mean union pixels clipped 7062
4px: box fallback 0/1, mean instances with pixels after clip 75.00, mean union pixels clipped 5520
6px: box fallback 0/1, mean instances with pixels after clip 75.00, mean union pixels clipped 4109
8px: box fallback 0/1, mean instances with pixels after clip 75.00, mean union pixels clipped 2844
10px: box fallback 0/1, mean instances with pixels after clip 75.00, mean union pixels clipped 1766
