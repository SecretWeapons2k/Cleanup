Offline chroma-penalty blend simulation
Run directory: /home/secret/Downloads/Cleanup/yolo_mask_winners/contrast_t38_padding4_hard10_20261023_201544
Color diagnostic: /home/secret/Downloads/Cleanup/yolo_mask_winners/contrast_t38_padding4_hard10_20261023_201544/debug_candidates/repeated_glyph_color_residual_diagnostic_20261023_224202
Grid: radius=(0, 1, 2); chroma tolerance ab=(12, 20, 28); weights=(0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
Minimum template coverage: 0.80
Candidate formula: structure_score + weight * normalized_color_residual_index
Eligible rows only: text_quality_eligible and anchor_detail_eligible must be true.
No inference, no production code or saved images/scores changed.

IMPORTANT LIMITATIONS
- Diagnostic residual ignores sprite-structure quality and is used only as a simulated additive penalty.
- The simulation does NOT replay the production residual-micro-component tie-break or BOX_ONLY rescue. The actual winner and the structure-only minimum are included so disagreements are visible.
- Images with insufficient template coverage are not blended; their actual production winner is carried through unchanged.
- A zero residual is not automatically proof of a visually perfect cleanup. Inspect the overlays and candidate images before integrating anything.

Main files:
- blend_winner_summary.csv: one row per image/parameter setting/weight
- candidate_blend_scores.csv: ranked candidate details where coverage is adequate
- parameter_summary.csv: compact parameter grid summary
