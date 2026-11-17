# Optimierungsbericht 20261117_132006

Backup: `/home/secret/cleanup_backups/Cleanup_backup_20261117_132006`

## Geaenderte Dateien
- `yolo_mask_auto_select_modules/pipeline.py`
- `yolo_mask_auto_select_modules/config.py`
- `yolo_mask_routing_audit.py`
- `yolo_mask_auto_select_modules/tests/test_audit_no_visuals.py`

## Validierung
- syntax: OK
- unit_tests: OK

```
test_new_generated_texture_has_only_ten_percent_density_weight (yolo_mask_auto_select_modules.tests.test_scoring_and_routing.RetainedDetailRatioTests.test_new_generated_texture_has_only_ten_percent_density_weight) ... ok

----------------------------------------------------------------------
Ran 39 tests in 0.827s

OK
```

## Nicht Teil dieser Stufe
- Umstrukturierung nach src/cleanup/, pyproject.toml/uv.lock (erst nach Baseline-Messung)
- Laufzeit-/VRAM-Vergleich vorher/nachher: erfordert echten Lauf mit GPU (`run_stats.json` im Ausgabeordner liefert die Phasenzeiten)
- Bewertung der Cleaning-Methoden/Scoring-Metriken: erfordert die vollstaendigen candidate_scores.csv

Wiederherstellung: `python optimize_cleanup.py --restore /home/secret/cleanup_backups/Cleanup_backup_20261117_132006`