Repeated-text geometric glyph-template residual diagnostic v2 (offline only).
Run directory: /home/secret/Downloads/Cleanup/yolo_mask_winners/contrast_t38_padding4_hard10_20261023_201544
Calibration directory: /home/secret/Downloads/Cleanup/yolo_mask_winners/contrast_t38_padding4_hard10_20261023_201544/debug_candidates/color_guard_calibration_v2_20261023_215610
Source anchors: /home/secret/Downloads/yabai proj/processed_output/anchors.csv
Source images: /home/secret/Downloads/yabai proj/yabaiproject - Copy/mit_kanji
Normalized template grid: 256x64; consensus threshold: 0.350.
For each anchor, the source glyph template is built leave-one-out from other anchors with identical normalized OCR text and type.
Template donors use local Lab-colour prototype plus local-detail support; an unscorable target can borrow geometry from valid donors.
Candidate detail density uses the FULL template zone as denominator; transparent output pixels count as removed, not as missing samples.
Radius 0 is strict template support, radius 1 adds a 1px band, radius 2 adds a 2px band.
Overlay colors: magenta = details inside exact template; orange = the 1px band; green = the 2px-only outer band; cyan = template contour; yellow = OCR box.
Ranking is diagnostic only, based on worst template residual at each radius and sufficient repeated-template coverage. It ignores sprite-structure preservation.
Unique-text anchors without repeated donors remain unscorable rather than falling back silently.
CPU-only; no YOLO/LaMa inference; no masks, candidate images, production scores, or winner reports modified.
