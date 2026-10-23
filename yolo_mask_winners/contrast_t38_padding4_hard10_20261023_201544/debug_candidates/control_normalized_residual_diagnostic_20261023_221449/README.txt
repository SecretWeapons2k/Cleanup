Control-normalized source-supported glyph residual diagnostic (offline only).
Run directory: /home/secret/Downloads/Cleanup/yolo_mask_winners/contrast_t38_padding4_hard10_20261023_201544
Calibration directory: /home/secret/Downloads/Cleanup/yolo_mask_winners/contrast_t38_padding4_hard10_20261023_201544/debug_candidates/color_guard_calibration_v2_20261023_215610
Source anchors: /home/secret/Downloads/yabai proj/processed_output/anchors.csv
Source images: /home/secret/Downloads/yabai proj/yabaiproject - Copy/mit_kanji
Source detail threshold: 0.120; primary final detail threshold: 0.080.
Control is the same OCR interior excluding source glyph support dilated by 5px.
Per-anchor normalized excess = max(0, final target detail density - final control detail density) / (source target detail density - source control detail density), capped at 1; source density gap must be >= 0.04.
Candidate rankings require sufficient color-prototype and control-zone coverage; detail-only fallback is not counted as reliable evidence.
Overlay: magenta = final local contrast inside the source-supported glyph zone; green = final local contrast in the nearby control zone; cyan = source-supported glyph contour; yellow = OCR box.
These metrics are diagnostic only and ignore structure quality. They are not added to production scoring and cannot select production winners.
CPU-only; no YOLO/LaMa inference; no existing candidates, score files, masks, or production code modified.
