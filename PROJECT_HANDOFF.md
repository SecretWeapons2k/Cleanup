# Projektübergabe — Cleanup

Stand: 18. November 2026  
Projektpfad: `~/Downloads/Cleanup`  
Zweck: Japanische Schrift/ausgewählte Textbereiche aus Bildern entfernen und
Sprite-Details, Konturen, Farben und Alpha möglichst erhalten.

## 1. Produktionsstruktur

- `yolo_mask_auto_select.py`: CLI-Einstieg; ruft `pipeline.main()` auf.
- `yolo_mask_routing_audit.py`: **zwingender Laufzeitbestandteil**. Die
  Hauptpipeline startet dieses Skript als Unterprozess, um YOLO-Masken,
  Distanzvarianten und den Routing-Report aufzubauen.
- `yolo_mask_auto_select_modules/config.py`: Pfade, aktive Kandidaten,
  Schwellenwerte und Laufzeitparameter.
- `cli.py`: Argumente und Laufzeitoptionen.
- `candidate_masks.py`: Maskenkandidaten, Schutzlogik und Masken-Deduplizierung.
- `geometry.py`: Anchor-Polygone, Box-Geometrie und geschützte Zonen.
- `io_utils.py`: CSV-/Bild-I/O, Eingabefindung und Alpha-Nachbearbeitung.
- `scoring.py`: Textreste, Detailerhalt, Konturen, Leaks und Kandidatenauswahl.
- `fill_repair.py`: optionale Nachprüfung/Reparatur kleiner LaMa-Füllartefakte.
- `pipeline.py`: YOLO-/LaMa-Orchestrierung, Compositing, Bewertung und Ausgabe.
- `debug_output.py` und `reports.py`: Debug-Bilder und CSV-Berichte.
- `tests/`: Regressionstests für die Kernlogik.

Das Routing-Audit darf bei einer Bereinigung nicht entfernt werden: `pipeline.py`
prüft seine Existenz und startet es mit `--no-visuals` als Teil jedes regulären
Laufs.

## 2. Eingaben und Standardpfade

- Anchors: `~/Downloads/yabai proj/output/csv/anchors.csv`
- Quellbilder: `~/Downloads/yabai proj/mit_kanji/`
- Optionale Schutzzonen: `protected_zones.csv` neben der gewählten Anchors-CSV
- YOLO-Gewichte: `~/.cache/cleanup-models/manga-inpainter-best.pt`
- Standardausgabe: `yolo_mask_winners/` im Projektordner

Die Pfade sind in `config.py` zentral definiert und können über CLI-Argumente
ausgetauscht werden. Modellgewichte und Bilddaten liegen außerhalb des
Quellcode-Repositories.

## 3. Aktuelle Pipeline

Der reguläre Ablauf:

1. Anchors und Quellbilder laden und Auswahl bestimmen.
2. Routing-Audit ausführen und YOLO-Distanzmasken erzeugen.
3. Kandidatenmasken einschließlich Box-/Smart-Ink-/Contrast-/Hybridvarianten
   aufbauen und geschützte Zonen ausschließen.
4. Pixelgleiche Masken deduplizieren, damit identische Kandidaten nur einmal
   durch LaMa und Scoring laufen.
5. RGB und Alpha mit IOPaint/LaMa verarbeiten, wieder zusammensetzen und
   geschützte Pixel auf die Originalwerte zurücksetzen.
6. Optional `fill_repair` auf den gefüllten RGB-Bereich anwenden.
7. Kandidaten bewerten, Gewinner wählen und Reports/Masken/Debug-Ausgaben
   schreiben.

Die aktive Kandidatenkonfiguration und Schwellenwerte stehen ausschließlich
in `config.py` und den Produktionsmodulen. Historische Ausgaben sind keine
maßgebliche Konfiguration.

## 4. Wichtige aktuelle Einstellungen

- YOLO-Distanzen: `0,2,4,6,8,10` (die CLI validiert genau diese Menge).
- Contrast-Kandidat: `CONTRAST_T38` (`CONTRAST_MASK_THRESHOLDS = (0.38,)`).
- Weitere Kandidaten: `SMART_INK`, `BOX_ONLY` und konfigurierte Hybridvarianten.
- `DEDUPLICATE_IDENTICAL_MASKS = True`.
- `FILL_REPAIR_ENABLED = True`.
- `LAMA_BATCH_SIZE = 100`.
- `LAMA_PARALLEL_PROCESSES = 2`.
- `POSTPROCESS_WORKERS = 6`.
- Debug-Bilder werden standardmäßig gemäß den `DEBUG_SAVE_*`-Schaltern
  gespeichert; `--no-debug-images` deaktiviert sie für Produktionsläufe.

Für A/B-Tests können `--no-dedup` und `--no-fill-repair` verwendet werden.

## 5. Lauf- und Testbefehle

Syntaxprüfung:

```bash
uv run python -m compileall -q yolo_mask_auto_select_modules yolo_mask_auto_select.py yolo_mask_routing_audit.py
```

Unit-Tests:

```bash
uv run python -m unittest discover -s yolo_mask_auto_select_modules/tests -t . -v
```

Kleiner Dry-Run:

```bash
uv run python yolo_mask_auto_select.py --random-sample 10 --min-m-images 3 --seed 20261018 --dry-run
```

Einzelbild:

```bash
uv run python yolo_mask_auto_select.py --image f002_004.png --output ~/cleanup_compare/one_image
```

Vollständiger Lauf:

```bash
uv run python yolo_mask_auto_select.py --output ~/cleanup_compare/<NAME>
```

`--dry-run` entfernt starten, wenn wirklich Bilder verarbeitet werden sollen.

## 6. Bestätigter Performance-Stand und offene Auswertung

Die fünf Performance-Stufen sind bereits in den Produktionscode übernommen:

1. Identische Kandidatenmasken nur einmal verarbeiten; Reports behalten alle
   Kandidaten-IDs. Maskensatz pro Bild nur einmal aufbauen.
2. `fill_repair` auf den relevanten Bildausschnitt beschränken; bitgleiches
   Verhalten wurde in Tests geprüft.
3. Routing-Audit ohne temporäre Visualisierungen ausführen und Batchgröße auf
   100 erhöhen.
4. Zwei IOPaint-Prozesse parallel betreiben.
5. Compositing/Nachbearbeitung/Scoring über sechs Worker parallelisieren.

Laut dokumentiertem Vergleich waren bei der 40-Bilder-Stichprobe die
Kandidaten-Scores (ohne neue Dedup-Spalte), Gewinner und `winner_report.csv`
zwischen den Stufenläufen identisch. Das ist kein Beleg für identisches
GPU-Verhalten auf jedem möglichen Datensatz.

**Vollständiger Lauf nach Stufe 5:** 716 Bilder, 8 Batches, keine gemeldeten
Fehler, Laufzeit `52m20,6s`. Ausgabeordner: `~/cleanup_compare/FULL_stage5/`.
Die Dateien `winner_report.csv`, `candidate_scores.csv`,
`mask_dedup_report.csv` und `run_stats.json` dieses Laufs müssen noch gemeinsam
ausgewertet werden. Ein Vergleich der Gewinner mit dem vorherigen
Vollständiglaufs steht ebenfalls noch aus.

## 7. Arbeitsregeln

- `config.py`, `scoring.py`, `candidate_masks.py`, `geometry.py` und
  `pipeline.py` bei Änderungen zusammen betrachten.
- Vor Änderungen Tests und einen kleinen Dry-Run ausführen.
- Visuelle Qualität anhand von Original, Maske und Gewinnerbild aus demselben
  Laufverzeichnis beurteilen.
- Keine Score-Gewichte ändern, um einen Maskierungs- oder Inpaintingfehler zu
  kaschieren.
- Nutzerergebnisse und vollständige Laufverzeichnisse außerhalb des
  Quellcode-Repositories aufbewahren.
- Nach Codeänderungen Syntaxprüfung und die vollständige Unit-Test-Suite
  ausführen; Tests mit Attrappen belegen nicht das Verhalten eines echten
  GPU-/IOPaint-Laufs.

## 8. Grenzen des aktuellen Wissens

Die Bildqualität des vollständigen 716-Bilder-Laufs nach Stufe 5 ist noch nicht
visuell abgenommen. Auch die Auswertung seiner CSV-Reports steht aus. Aus den
Unit-Tests oder der Laufzeit allein darf keine Aussage über die visuelle
Qualität der Gewinner abgeleitet werden.
