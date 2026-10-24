# Unabhängige visuelle Holdout-Prüfung

**Datum:** 2026-10-24  
**Projekt:** `Cleanup`  
**Holdout-Lauf:** `independent_color_rule_holdout_20261024_064402`  
**Status:** Diagnostische Prüfung abgeschlossen; keine Produktionsänderung freigegeben.

## Umfang

Der CPU-only-Holdout untersuchte 24 unabhängige Bilder und 120 bereits vorhandene Kandidatenvarianten. Die aktualisierte Entscheidungslogik würde fünf Gewinner wechseln. Die Vergleichsbögen wurden visuell geprüft. Es wurden keine GPU-Inferenz ausgeführt und keine Kandidatenbilder verändert.

## Visuelle Entscheidungen

| Bild | Bisheriger Gewinner | Vorschlag | Entscheidung | Beobachtung |
|---|---|---|---|---|
| `c000_002.png` | `BOX_ONLY` | `HYBRID_D2` | Vorschlag beibehalten | Der Vorschlag reduziert die letzten sichtbaren Farbreste weiter. Es bleiben allenfalls sehr schwache Spuren. |
| `l008_025.png` | `BOX_ONLY` | `HYBRID_D2` | Vorschlag beibehalten | Der Vorschlag wirkt sauberer; die sichtbaren Restspuren werden deutlich reduziert. |
| `o016_002.png` | `CONTRAST_T38` | `BOX_ONLY` | Vorschlag beibehalten | Die orangefarbene Schrift ist im Vorschlag weitgehend entfernt. |
| `o016_012.png` | `CONTRAST_T38` | `SMART_INK` | Vorschlag beibehalten | Fast die gesamte orangefarbene Schrift ist entfernt; einzelne schwache Reste bleiben sichtbar. |
| `o018_039.png` | `BOX_ONLY` | `HYBRID_D2` | Vorschlag beibehalten | Der kleine blaue Rückstand verschwindet im Vorschlag. |
| `n008_005.png` | `CONTRAST_T38` | `HYBRID_D2` (früherer Vorschlag) | Bisherigen Gewinner beibehalten | Nach korrigierter Zielfunktionsprüfung ist der Vorschlag schlechter: aktueller kombinierter Wert `0.665383`, HYBRID_D2 `0.676899` (niedriger ist besser). Der Hintergrund ist zudem stark strukturiert und cyanblau; die Farbmetrik allein reicht für einen Wechsel nicht aus. |

## Technische Feststellungen

- Die frühere Holdout-Logik verglich zulässige Alternativen untereinander, ohne sicherzustellen, dass die beste Alternative auch besser als der aktuelle Gewinner war.
- Die korrigierte Audit-Logik verlangt, dass die Alternative den kombinierten Wert des aktuellen Gewinners tatsächlich verbessert. Dadurch bleibt `n008_005` bei `CONTRAST_T38`.
- `source_color_pixels_gt16` zählt eine gerichtete Farbabweichung in den OCR-Ankerbereichen. Es ist ein Proxy für verbleibende ursprüngliche Schriftfarbe, keine vollständige Qualitätsmetrik und kein Nachweis, dass Hintergrunddetails unverändert geblieben sind.

## Ergebnis und Grenzen

Für diesen unabhängigen Holdout sind die fünf verbliebenen Wechsel visuell plausibel. Das ist eine visuelle Einschätzung der vorhandenen Vergleichsbilder, keine pixelgenaue Validierung aller Bilder und keine Freigabe für einen automatischen Produktionswechsel.

**Produktionsstatus:** `scoring.py` unverändert. Keine Produktionsauswahl verändert. Die Datei `holdout_manual_review.csv` bleibt unverändert; diese Dokumentation hält die visuelle Einschätzung separat fest.

## Referenzdateien

Ausgabeordner:
`yolo_mask_winners/full_recheck_after_detection_20261024_030641/debug_candidates/independent_color_rule_holdout_20261024_064402/`

Wichtige Dateien:
- `holdout_rule_decisions.csv`
- `holdout_candidate_metrics.csv`
- `holdout_comparison_01.jpg`
- `holdout_source_context_comparison_01.jpg` aus dem vorherigen Kontextvergleich-Lauf `independent_color_rule_holdout_20261024_061544`
