cd ~/Downloads/Cleanup || exit 1

cat > PROJECT_HANDOFF.md <<'EOF'
# PROJECT HANDOFF — Cleanup YOLO/LaMa

Stand: 19. Oktober 2026
Projektpfad: ~/Downloads/Cleanup
Sprache der Zusammenarbeit: Deutsch

## 1. Projektziel

Automatische Entfernung unerwünschter Schrift und anderer Elemente aus
Bildern. Sprite-Details, Farben, Konturen und natürliche Bildstrukturen
sollen möglichst erhalten bleiben.

Die Pipeline vergleicht mehrere Maskierungsvarianten, führt LaMa-Inpainting
aus, bewertet die Kandidaten und speichert das ausgewählte Ergebnis.

## 2. Aktuelle Projektstruktur

yolo_mask_auto_select.py
  Schlanker Programmeinstieg für die modulare YOLO/LaMa-Pipeline.

yolo_mask_routing_audit.py
  Eigenständiges Diagnosewerkzeug für Maskenabdeckung und Routing.
  Enthält unter anderem body_box_coverage_metrics() und
  erzeugt Diagnoseberichte und Vergleichsbilder.

yolo_mask_auto_select_modules/config.py
  Zentrale Pfade, Modell- und Laufparameter, Schwellenwerte,
  Bewertungsgewichte und Einstellungen für Masken-/Alpha-Nachbearbeitung.

yolo_mask_auto_select_modules/cli.py
  Verarbeitet Kommandozeilenargumente.

yolo_mask_auto_select_modules/io_utils.py
  CSV- und Bildverarbeitung, Quellenfindung, Box-Masken,
  Alpha-Nachbearbeitung und optionale OCR-Hilfsfunktionen.

yolo_mask_auto_select_modules/scoring.py
  Bildmetriken und Kandidatenbewertung.
  Wichtige Funktionen:
  - sprite_contour_metrics()
  - ink_strength_map()
  - make_smart_ink_mask()
  - boundary_artifact_metrics()
  - _retained_detail_ratio()
  - residual_blob_contrast_ratio()
  - estimate_ocr_linked_text_corridor()
  - outline_leak_metrics()
  - _peak_residual_ratio()
  - score_candidate()
  - select_candidate()

yolo_mask_auto_select_modules/pipeline.py
  Orchestriert die Verarbeitung: Kandidaten, YOLO/LaMa,
  Bewertung, Auswahl und Speicherung der Gewinnergebnisse.

yolo_mask_auto_select_modules/debug_output.py
  Debug-Kontaktbögen, Masken-Overlays und Vorher-/Nachher-Vergleiche.

yolo_mask_auto_select_modules/reports.py
  CSV-Ausgabe mit dynamischer Vereinigung der tatsächlich vorhandenen
  Bewertungsfelder. Verhindert Fehler durch zusätzliche Score-Spalten.

yolo_mask_auto_select_modules/tests/test_report_fields.py
  Regressionstest für dynamische CSV-Felder.

yolo_mask_auto_select_modules/tests/test_scoring_and_routing.py
  Regressionstests für Detailbewertung, Box-only Rescue und
  aussagekräftige YOLO-Abdeckung.

## 3. Maskenrouting und Kandidatenauswahl

Konfigurierte YOLO-Distanzen:
0, 2, 4, 6, 8, 10 Pixel.

Zusätzlich existiert eine Box-only-Maskenvariante.
Bei ausreichender inhaltlicher Maskenabdeckung werden die verfügbaren
Kandidaten verglichen. Bei unzureichender Abdeckung greift der
vorgesehene Box-only-Fallback.

Die Entscheidung darf nicht nur auf einem beliebigen einzelnen
überlappenden Pixel basieren. Relevant sind unter anderem absolute
Überlappung, Detailenergie und hochkontrastige Bildbereiche.

Die Bewertung berücksichtigt unter anderem Schriftreste, Detailerhaltung,
Konturen, Strukturveränderungen, Farbverschiebungen und Artefakte.

Box-only Rescue kann die normale Kandidatenauswahl beeinflussen,
wenn die Box-only-Variante hinsichtlich Schriftentfernung oder
Verbesserung innerhalb der Box ausreichend vorteilhaft ist.

## 4. Aktueller technischer Arbeitsschwerpunkt

Box-only-Rescue-Bewertung und die zugehörigen Schwellenwerte untersuchen.

Im bisherigen Gespräch wurden unter anderem folgende Metriken betrachtet:

- residual_text_detail_ratio
- residual_inside_peak
- residual_inside_energy
- residual_blob_contrast_ratio
- visual_text_residual_score
- ocr_text_residual_ratio

Der genaue aktuelle Entscheidungszweig und die aktuellen Funktionen
müssen vor Änderungen anhand des lokalen Quellcodes überprüft werden.
Die Werte aus älteren Paketversionen sind nicht automatisch aktuell.

### Wichtiger Dokumentationsabgleich

Die README nennt für die Near-best-Texttoleranz:
- Floor: 0.003
- Ceiling: 0.008
- Range fraction: 0.05

Die bei der Bestandsaufnahme ausgegebene config.py enthielt dagegen:
- TEXT_SCORE_TOLERANCE_FLOOR = 0.02
- TEXT_SCORE_TOLERANCE_CEILING = 0.035
- TEXT_SCORE_TOLERANCE_RANGE_FRACTION = 0.05

Vor weiteren Änderungen müssen wir untersuchen, ob README oder
aktive Konfiguration veraltet ist oder die Werte unterschiedlich
verwendet werden. Keine Werte allein aufgrund der Dokumentation ändern.

## 5. Berichte und Testläufe

Ergebnisverzeichnis:
yolo_mask_winners/

Bei der letzten Bestandsaufnahme war das nach Verzeichnisnamen
neueste Laufverzeichnis:
yolo_mask_winners/run_20261019_163711/

Darin:
- winner_report.csv
- debug_candidates/
- masks/

Der Bericht war ungefähr 136 KB groß. Frühere gelistete Läufe hatten
deutlich kleinere winner_report.csv-Dateien. Daraus allein lässt sich
noch keine Aussage über die Bildqualität oder den Erfolg ableiten.

Vorhandene Tests:
- tests/test_report_fields.py
- tests/test_scoring_and_routing.py

Testergebnisse müssen nach einem tatsächlichen Testlauf dokumentiert
werden. Der bloße Dateibestand bedeutet nicht, dass die Tests bestanden sind.

## 6. Wichtige Befehle

Projekt öffnen:
cd ~/Downloads/Cleanup

Virtuelle Umgebung aktivieren:
source .venv/bin/activate

Syntaxprüfung:
python -m py_compile yolo_mask_auto_select.py \
  yolo_mask_routing_audit.py \
  yolo_mask_auto_select_modules/*.py

Unit-Tests:
python -m unittest discover \
  -s yolo_mask_auto_select_modules/tests -v

Bisher dokumentierter Dry-Run:
python yolo_mask_auto_select.py \
  --random-sample 10 --min-m-images 3 \
  --seed 20261018 --dry-run

## 7. Regeln für die Zusammenarbeit

- Vor Änderungen den aktuellen lokalen Code berücksichtigen.
- Ergebnisse bereits durchgeführter Tests nicht ohne Grund wiederholen.
- Zwischen bestätigten Ursachen, ausgeschlossenen Ursachen und offenen
  Fragen unterscheiden.
- Schwellenwerte nicht ohne Bezug zu den konkreten Metriken ändern.
- Bei größeren Python-Dateien bevorzugt gezielte Patch-Skripte verwenden.
- Änderungen an config.py, scoring.py und pipeline.py gemeinsam betrachten,
  wenn ihre Logik voneinander abhängt.
- Vor einem längeren Lauf möglichst Syntaxprüfung, passende Tests und
  einen kleinen Diagnose-/Dry-Run durchführen.
- Bildqualität anhand der tatsächlichen Reports und Vergleichsbilder prüfen.
- Keine Aussage über bestandene Tests machen, solange kein Ergebnis vorliegt.

## 8. Grenzen dieser Übergabe

Diese Datei beschreibt die bekannte Architektur und den dokumentierten
Arbeitsstand. Sie enthält nicht den gesamten Quellcode und aktualisiert
sich nicht automatisch.

Bei einem neuen Chat diese Datei zusammen mit dem aktuellen konkreten
Arbeitsproblem und den neuesten relevanten Testergebnissen übergeben.
Vor Codeänderungen gegebenenfalls aktuelle Funktionsausschnitte aus dem
lokalen Projekt nachreichen.
EOF

echo
echo "Projektübergabe erstellt:"
ls -lh PROJECT_HANDOFF.md
echo
echo "Datei: $(pwd)/PROJECT_HANDOFF.md"