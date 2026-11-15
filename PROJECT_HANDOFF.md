# PROJECT HANDOFF — Cleanup YOLO/LaMa

Stand: 15. November 2026
Projektpfad: ~/Downloads/Cleanup
Sprache der Zusammenarbeit: Deutsch

## 1. Projektziel

Das Projekt entfernt unerwünschte Schrift und andere ausgewählte Elemente aus
Bildern. Sprite-Details, Farben, Konturen und natürliche Bildstrukturen sollen
möglichst erhalten bleiben.

Die Hauptpipeline erzeugt mehrere Maskenkandidaten, verarbeitet diese mit
YOLO/LaMa, bewertet die Ergebnisse und speichert pro Bild eine ausgewählte
Variante samt Masken, Kandidatenbildern und Berichten.

## 2. Hauptpipeline und Module

`yolo_mask_auto_select.py`
  Schlanker Programmeinstieg. Importiert `main()` aus dem Pipeline-Modul.

`yolo_mask_auto_select_modules/config.py`
  Zentrale Pfade, Modell- und Laufparameter, aktive Kandidatenvarianten,
  Schwellenwerte und Bewertungsgewichte. Die gemeinsamen Standardpfade sind:
  - Anchors: `~/Downloads/yabai proj/processed_output/anchors.csv`
  - Quellbilder: `~/Downloads/yabai proj/yabaiproject - Copy/mit_kanji/`
  - optionale Schutzzonen: `protected_zones.csv` neben der aktiven Anchors-CSV.
  Pipeline, Routing-Audit und Label-Tool verwenden dieselben Defaults.
  Abweichende Speicherorte können per CLI angegeben werden. Für aktuelle
  Einstellungen ist ausschließlich diese Datei maßgeblich; alte Logs und
  Backups können abweichende historische Werte enthalten.

`yolo_mask_auto_select_modules/cli.py`
  Kommandozeilenargumente für Bilder, Anchors, YOLO-Gewichte, Stichprobe,
  Distanzen, Geräte, Ausgabe und Dry-Run.

`yolo_mask_auto_select_modules/io_utils.py`
  Bild- und CSV-Verarbeitung, Quellenfindung, Anchor-/Box-Masken,
  Alpha-Nachbearbeitung und zugehörige I/O-Helfer.

`yolo_mask_auto_select_modules/scoring.py`
  Metriken zur Kandidatenbewertung und Auswahl. Dazu gehören unter anderem
  Textreste, Detailerhaltung, Konturen, Strukturveränderungen, Leaks,
  Box-only-Rescue und Residual-Micro-Components.

`yolo_mask_auto_select_modules/pipeline.py`
  Orchestriert YOLO-Kandidaten, Maskenvarianten, LaMa-Verarbeitung,
  Bewertung, Auswahl und Speicherung.

`yolo_mask_auto_select_modules/debug_output.py`
  Debug-Overlays, Kontaktbögen und Vorher-/Nachher-Vergleiche.

`yolo_mask_auto_select_modules/reports.py`
  CSV-Ausgabe mit der Vereinigung der bevorzugten und tatsächlich vorhandenen
  Metrikfelder. Verhindert Abbrüche, wenn neue Score-Felder hinzukommen.

## 3. Eigenständige Hilfsprogramme

Diese Programme haben getrennte Aufgaben und sind keine Duplikate der
Hauptpipeline:

`yolo_mask_routing_audit.py`
  Eigenständiges YOLO-Inferenz-/Diagnosewerkzeug. Verwendet dieselben
  Standardpfade für `anchors.csv`, Quellbilder und optionale Schutzzonen
  wie die Hauptpipeline. Prüft Maskenabdeckung,
  Abstand zu OCR-Ankern und Body-Box-Coverage. Erzeugt unter anderem
  `routing_report.csv`, `body_box_coverage.csv`, `instances.csv` und
  visuelle Kontaktbögen. Kein LaMa-Lauf und keine Änderung der Quelldateien.

`label_tool.py`
  Lokale Browser-Oberfläche zur manuellen Bewertung der berechneten Kandidaten.
  Speichert die Labels im jeweiligen Laufverzeichnis.

`train_ranker.py`
  Trainiert und bewertet einen Kandidaten-Ranker anhand von
  `candidate_scores.csv` und manuellen Labels. Schreibt unter anderem
  `ranker_picks.csv`, `cv_report.txt` und das trainierte Modell.

`build_final_picks.py`
  Kombiniert Ranker-Auswahl und manuelle Labels. Kann akzeptierte
  Kandidatenbilder in einen separaten Ordner kopieren; Quelldateien werden
  dabei nicht verschoben.

`scripts/cleanup_repo.py`
  Repository-Wartung. Standardmäßig nur Dry-Run-Bericht. Mit `--apply`
  werden Python-Caches entfernt und alte Backups, der bekannte versehentliche
  Diff-Dump sowie Root-Logs außerhalb des Repositories archiviert.
  Anwendungslogik und Ergebnisdaten werden nicht automatisch gelöscht.

## 4. Maskenvarianten und Auswahl

Die konfigurierten YOLO-Distanzen und zusätzlichen Kandidaten werden in
`config.py` festgelegt. Historische Läufe verwendeten unter anderem
Distanzen 0, 2, 4, 6, 8 und 10 Pixel sowie Box-only-, Smart-Ink- und
Hybrid-Varianten. Vor einem neuen Lauf die aktuell aktiven Werte im Code
prüfen, statt historische Logs als Konfiguration zu behandeln.

Die Auswahl berücksichtigt nicht nur die rohe Maskenüberlappung. Für das
Routing sind unter anderem absolute Überlappung, Detailenergie und
hochkontrastige Pixel relevant. Bei unzureichender inhaltlicher Abdeckung
kann der Box-only-Fallback greifen.

Die Scoring-Logik berücksichtigt unter anderem Schriftreste, Detailerhaltung,
Konturen, Strukturveränderungen, Farbverschiebungen und Artefakte. Die genauen
Metriken und Entscheidungszweige müssen vor einer Änderung im aktuellen
`scoring.py` und `pipeline.py` überprüft werden.

## 5. Ergebnis- und Diagnoseverzeichnisse

`yolo_mask_winners/`
  Enthält Laufverzeichnisse mit Gewinner-Masken, Kandidatenbildern,
  `candidate_scores.csv`, `winner_report.csv` und Debug-Ausgaben.
  Dieses Verzeichnis ist groß und wird in Git ignoriert. Es ist Nutzdaten-
  und Ergebnisbestand, kein temporärer Cache.

`diagnostics/`
  Enthält gezielte Diagnose- und Replay-Ergebnisse. Diese Daten bleiben
  erhalten, solange sie als Evidenz für Auswahl- oder Qualitätsvergleiche
  dienen.

Der gespeicherte Replay-Bericht für `l009_006` vom 25. Oktober 2026 meldet:
- archivierter und erneut bestimmter Gewinner: `HYBRID_D2`
- Gewinner-Maske pixelgenau identisch
- 1.111 verglichene Metrikfelder, alle auf sechs Dezimalstellen identisch
- keine Änderung an Produktionsdateien
- kein erneuter LaMa-Lauf

Das ist ein gezielter Einzelbild-Replay und kein neuer vollständiger
716-Bilder-Lauf.

## 6. Letzte bekannte Laufdaten

Der archivierte `full_run.log` dokumentierte einen Lauf mit 716 ausgewählten
Bildern und 716 gespeicherten Gewinner-Masken am 19. Oktober 2026.
Das Log wird aus dem Repository heraus in ein lokales Archiv verschoben.
Es ist ein historischer Lauf und belegt nicht den Zustand eines späteren
vollständigen Reprocessings.

Vorhandene Ergebnisdaten vom Lauf
`full_recheck_after_alpha_island_025_20261025_022053` umfassen unter anderem
`winner_report.csv`, Kandidaten-Scores, Masken und Debug-Ausgaben.

## 7. Tests und wichtige Befehle

Projektverzeichnis:

```bash
cd ~/Downloads/Cleanup
source .venv/bin/activate
```

Syntaxprüfung:

```bash
python -m py_compile yolo_mask_auto_select.py yolo_mask_routing_audit.py yolo_mask_auto_select_modules/*.py
```

Unit-Tests:

```bash
python -m unittest discover -s yolo_mask_auto_select_modules/tests -v
```

Kleiner Pipeline-Dry-Run:

```bash
python yolo_mask_auto_select.py --random-sample 10 --min-m-images 3 --seed 20261018 --dry-run
```

Routing-Audit-Hilfe:

```bash
python yolo_mask_routing_audit.py --help
```

Manuelle Kandidatenbewertung:

```bash
python label_tool.py --run-dir <LAUFVERZEICHNIS>
```

Ranker-Training benötigt Score- und Label-CSV aus demselben Bewertungsworkflow:

```bash
python train_ranker.py --scores <candidate_scores.csv> --labels <labels.csv> --out <ranker_out>
```

Finale Auswahl:

```bash
python build_final_picks.py --run-dir <LAUFVERZEICHNIS>
```

Repository-Cleanup:

```bash
python scripts/cleanup_repo.py
python scripts/cleanup_repo.py --apply
```

## 8. Zusammenarbeit und Sicherheitsregeln

- Vor Codeänderungen den aktuellen lokalen Code und die tatsächlich aktive
  Konfiguration prüfen.
- Zwischen bestätigten Ursachen, ausgeschlossenen Ursachen und offenen
  Fragen unterscheiden.
- Schwellenwerte nicht ohne Bezug zu konkreten Metriken ändern.
- Änderungen an `config.py`, `scoring.py` und `pipeline.py` gemeinsam
  betrachten, wenn ihre Logik voneinander abhängt.
- Vor längeren Läufen Syntaxprüfung, passende Tests und einen kleinen
  Diagnose-/Dry-Run durchführen.
- Bildqualität anhand tatsächlicher Reports und Vergleichsbilder prüfen.
- Keine Aussage über bestandene Tests machen, solange kein Testlauf vorliegt.
- Generierte Ergebnisbilder und Diagnoseberichte nicht pauschal löschen.
- Backups vor einer Entfernung außerhalb des Repositorys archivieren.

## 9. Bekannte Dokumentationsgrenzen

Diese Übergabe beschreibt die am 15. November 2026 geprüfte Repository-
Struktur und die oben genannten gespeicherten Diagnoseergebnisse. Sie ersetzt
keine Prüfung der aktuellen lokalen Laufdaten und aktualisiert sich nicht
automatisch.

Bei einem neuen Arbeitsschritt immer den aktuellen Quellcode, die letzten
relevanten Testergebnisse und die konkreten zu untersuchenden Bilder oder
Reports heranziehen.
