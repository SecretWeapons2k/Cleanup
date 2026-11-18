# PROJECT HANDOFF — Cleanup YOLO/LaMa

Stand: 18. November 2026 (nach Performance-Optimierung Stufe 1–5, siehe Abschnitt 11)
Projektpfad: ~/Downloads/Cleanup
Sprache der Zusammenarbeit: Deutsch

Kennzeichnung in diesem Dokument:
- **Bestätigt** = anhand konkreter Dateien/Zahlen geprüft (Quelle genannt).
- **Hypothese** = plausibel, aber nicht geprüft.
- **Offen** = noch nicht getestet oder nicht entschieden.

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
  - CSV-Verzeichnis: `~/Downloads/yabai proj/output/csv/`
  - Anchors: `anchors.csv` in diesem Verzeichnis
  - optionale Schutzzonen: `protected_zones.csv` im selben Verzeichnis
  - Quellbilder: `~/Downloads/yabai proj/mit_kanji/`
  Pipeline, Routing-Audit und Label-Tool verwenden dieselben Defaults.
  Abweichende Speicherorte können per CLI angegeben werden. Für aktuelle
  Einstellungen ist ausschließlich diese Datei maßgeblich; alte Logs und
  Backups können abweichende historische Werte enthalten.

`yolo_mask_auto_select_modules/cli.py`
  Kommandozeilenargumente für Bilder, Anchors, YOLO-Gewichte, Stichprobe,
  Distanzen, Geräte, Ausgabe und Dry-Run. Relevant für Einzeltests:
  `--image <Dateiname>` (nur dieses Bild) und `--output <Basisordner>`
  (ist der Ordner leer, wird direkt dort geschrieben; sonst entsteht darin
  `run_JJJJMMTT_HHMMSS`).
  Seit 17.11.2026 zusätzlich: `--no-dedup`, `--no-fill-repair`,
  `--no-debug-images`, `--lama-processes N`, `--postprocess-workers N`
  (Details in Abschnitt 11).

`yolo_mask_auto_select_modules/candidate_masks.py`
  Baut die Maskenkandidaten (Distanz-, Box-only-, Smart-Ink-, Contrast-,
  Hybrid-Varianten) und `enforce_protection`. Neu: `group_identical_masks()`
  erkennt pixelgleiche Kandidatenmasken (Hash als Vorfilter, Gleichheit wird
  immer Pixel für Pixel geprüft).

`yolo_mask_auto_select_modules/io_utils.py`
  Bild- und CSV-Verarbeitung, Quellenfindung, Anchor-/Box-Masken,
  Alpha-Nachbearbeitung und zugehörige I/O-Helfer.

`yolo_mask_auto_select_modules/scoring.py`
  Metriken zur Kandidatenbewertung und Auswahl. Dazu gehören unter anderem
  Textreste, Detailerhaltung, Konturen, Strukturveränderungen, Leaks,
  Box-only-Rescue und Residual-Micro-Components.

`yolo_mask_auto_select_modules/pipeline.py`
  Orchestriert YOLO-Kandidaten, Maskenvarianten, LaMa-Verarbeitung,
  Bewertung, Auswahl und Speicherung. RGB und Alpha werden getrennt als
  Stage-Bilder durch LaMa geschickt und im Composite-Block zusammengesetzt
  (`f_rgb[c_mask] = pred_rgb[c_mask]`, `f_alpha[c_mask] = pred_alpha[c_mask]`,
  danach `remove_small_alpha_islands` und `enforce_protection`).
  Seit 17.11.2026: pixelgleiche Masken gehen nur einmal durch LaMa/Scoring,
  IOPaint läuft pro Batch in N gleichzeitigen Prozessen, und Compositing,
  `fill_repair`, Scoring und Auswahl der Bilder eines Batches laufen in einem
  Thread-Pool (Abschnitt 11). Der Maskensatz wird pro Bild nur noch einmal
  gebaut (vorher zweimal). Pro Lauf entstehen `run_stats.json` und
  `mask_dedup_report.csv`.

`yolo_mask_auto_select_modules/debug_output.py`
  Debug-Overlays, Kontaktbögen und Vorher-/Nachher-Vergleiche.

`yolo_mask_auto_select_modules/reports.py`
  CSV-Ausgabe mit der Vereinigung der bevorzugten und tatsächlich vorhandenen
  Metrikfelder. Verhindert Abbrüche, wenn neue Score-Felder hinzukommen.

`yolo_mask_auto_select_modules/fill_repair.py`  **(16.11.2026; technisch geprüft am 17.11.2026)**
  Post-Fill-Sanity-Check für LaMa-Ergebnisse, siehe Abschnitt 9. Technisch
  geprüft durch Unit-Tests und Läufe; die visuelle Wirkung auf die
  Problembilder ist **noch nicht abgenommen**.

## 3. Eigenständige Hilfsprogramme

Diese Programme haben getrennte Aufgaben und sind keine Duplikate der
Hauptpipeline:

`yolo_mask_routing_audit.py`
  Eigenständiges YOLO-Inferenz-/Diagnosewerkzeug. Verwendet dieselben
  Standardpfade für `anchors.csv`, Quellbilder und optionale Schutzzonen
  wie die Hauptpipeline. Prüft Maskenabdeckung, Abstand zu OCR-Ankern und
  Body-Box-Coverage. Erzeugt unter anderem `routing_report.csv`,
  `body_box_coverage.csv`, `instances.csv` und visuelle Kontaktbögen.
  Kein LaMa-Lauf und keine Änderung der Quelldateien. Neu: `--no-visuals`
  lässt Overlays und Kontaktbögen weg (Masken und Report unverändert); die
  Hauptpipeline setzt den Schalter immer, weil ihr Audit-Ordner temporär ist.

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

`optimize_cleanup/optimize_cleanup*.py`  (NEU, 17.11.2026)
  Einmalige Migrationsskripte Stufe 1–5 (Abschnitt 11.6). Eigenständige
  Einzeldateien; nach erfolgreicher Anwendung nur noch als Dokumentation/
  Rollback-Werkzeug nötig.

`scripts/cleanup_repo.py`
  Repository-Wartung. Standardmäßig nur Dry-Run-Bericht. Mit `--apply`
  werden Python-Caches entfernt und alte Backups, der bekannte versehentliche
  Diff-Dump sowie Root-Logs außerhalb des Repositories archiviert.
  Anwendungslogik und Ergebnisdaten werden nicht automatisch gelöscht.

## 4. Maskenvarianten und Auswahl

Die konfigurierten YOLO-Distanzen und zusätzlichen Kandidaten werden in
`config.py` festgelegt. Historische Läufe verwendeten unter anderem
Distanzen 0, 2, 4, 6, 8 und 10 Pixel sowie Box-only-, Smart-Ink-, Contrast-
(z. B. `CONTRAST_T38`) und Hybrid-Varianten (`HYBRID_D2`, `HYBRID_D6`).
Vor einem neuen Lauf die aktuell aktiven Werte im Code prüfen, statt
historische Logs als Konfiguration zu behandeln.

Die Auswahl berücksichtigt nicht nur die rohe Maskenüberlappung. Für das
Routing sind unter anderem absolute Überlappung, Detailenergie und
hochkontrastige Pixel relevant. Bei unzureichender inhaltlicher Abdeckung
kann der Box-only-Fallback greifen (im Report z. B. als
`BOX_MASK_FALLBACK_WEAK_COVERAGE`).

Beobachtetes Auswahlverhalten (aus `candidate_scores.csv` / `winner_report.csv`
des Laufs `run_20261116_113900`; Code in `scoring.py`/`pipeline.py` bei einer
Änderung immer neu prüfen):
- Der finale Score entspricht dem `structure_score`; nicht eligible
  Kandidaten (`text_quality_eligible = 0`) erhalten +2,0.
- Das Eligibility-Gate hat eine Toleranz von 0,02 auf den `text_quality_score`
  relativ zum besten Kandidaten. Innerhalb dieser Toleranz entscheidet allein
  der `structure_score`.
- Ein Micro-Component-Tie-Break kann den strukturell besten Kandidaten durch
  einen mit weniger Restfläche ersetzen (Beispiel `d002_001`: SMART_INK
  → CONTRAST_T38, Restfläche −230 px, Struktur +0,016).
- D4, D6, D8 und D10 liefern bei einigen Bildern identische Masken/Ergebnisse
  (z. B. `d002_001`: pixelgenau identisch im Vormittagslauf).

## 5. Ergebnis- und Diagnoseverzeichnisse

`yolo_mask_winners/`
  Enthält Laufverzeichnisse mit Gewinner-Masken (`masks/`), Kandidatenbildern
  (`debug_candidates/cleaned_images/`), Kontaktbögen
  (`debug_candidates/contact_sheets/`), `debug_candidates/candidate_scores.csv`,
  `winner_report.csv` und Debug-Ausgaben. Dieses Verzeichnis ist groß und wird
  in Git ignoriert. Es ist Nutzdaten- und Ergebnisbestand, kein temporärer Cache.

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

**Vollständiger Lauf mit dem Stand nach Stufe 5 (18.11.2026, Bestätigt aus der
Terminalausgabe des Nutzers):** 716 Bilder, 8 Batches, ohne Fehler,
`real 52m20,6s`. Ausgabeordner: `~/cleanup_compare/FULL_stage5/` (außerhalb des
Repositorys). **Die Ergebnisdateien dieses Laufs (`winner_report.csv`,
`candidate_scores.csv`, `mask_dedup_report.csv`, `run_stats.json`) sind noch
nicht ausgewertet.** Ein Vergleich der Gewinner mit einem früheren
vollständigen Lauf steht aus. Einzelheiten und Zeiten: Abschnitt 11.3.

Der archivierte `full_run.log` dokumentierte einen Lauf mit 716 ausgewählten
Bildern und 716 gespeicherten Gewinner-Masken am 19. Oktober 2026.
Das Log wird aus dem Repository heraus in ein lokales Archiv verschoben.
Es ist ein historischer Lauf und belegt nicht den Zustand eines späteren
vollständigen Reprocessings.

Vorhandene Ergebnisdaten vom Lauf
`full_recheck_after_alpha_island_025_20261025_022053` umfassen unter anderem
`winner_report.csv`, Kandidaten-Scores, Masken und Debug-Ausgaben.

Lauf `run_20261116_113900` (16.11.2026) enthält u. a. `d002_001` (und vermutlich
`f002_004`; Gesamtumfang nicht geprüft). Zwischen diesem Lauf und dem
vorherigen Vormittagslauf haben sich Scores geändert (z. B. `CONTRAST_T38` in
`d002_001`: Micro-Komponenten 18 → 0, Eligibility nein → ja, D0 jetzt eligible).
Laut Nutzer wurde nur geändert, dass weiße Boxen statt der roten verwendet
werden, falls vorhanden (laut Nutzer war das schon vorher so). Die Ursache der
Score-Änderung ist **offen**.

**Wichtig für jede Analyse:** `candidate_scores.csv`, `winner_report.csv`,
Masken und Kandidatenbilder müssen aus **demselben** Laufverzeichnis stammen.
Am 16.11.2026 passten ein älteres `candidate_scores.csv` und ein neueres
`winner_report.csv` nicht zusammen (z. B. `text_quality_score` von
`CONTRAST_T38`: 0,345 vs. 0,261), was zu falschen Schlüssen führte.

## 7. Tests und wichtige Befehle

Projektverzeichnis (der Nutzer startet mit `uv run`; `source .venv/bin/activate`
ist gleichwertig):

```bash
cd ~/Downloads/Cleanup
```

Syntaxprüfung:

```bash
uv run python -m compileall -q yolo_mask_auto_select_modules yolo_mask_auto_select.py yolo_mask_routing_audit.py
```

Unit-Tests:

```bash
uv run python -m unittest discover -s yolo_mask_auto_select_modules/tests -t . -v
```

Teststand (Bestätigt): Die Stufenskripte aus Abschnitt 11.6 führen nach jeder
Stufe Syntaxprüfung und die komplette Suite auf dem System des Nutzers aus;
alle Stufen meldeten `syntax: OK` und `unit_tests: OK`. Die Suite umfasst im
Sandbox-Stand 42 Tests (28 bestehende, 14 neue). Neue Testdateien:
`test_dedup_and_fill_repair.py` (Maskengruppierung, `fill_repair`,
Pipeline-Gleichheit mit/ohne Dedup, mit 1/3 IOPaint-Prozessen, mit 1/4
Nachbearbeitungs-Threads, Fehlerpfad), `test_fill_repair_crop.py` (60
Zufallsszenen: zugeschnittenes `fill_repair` bitgleich zur Vollbild-Variante)
und `test_audit_no_visuals.py` (Audit mit/ohne `--no-visuals`, mit
Dummy-YOLO). Die Pipeline-Tests ersetzen YOLO-Audit und IOPaint durch
Attrappen und belegen die **Logik**, nicht das GPU-Verhalten.

Kleiner Pipeline-Dry-Run:

```bash
uv run python yolo_mask_auto_select.py --random-sample 10 --min-m-images 3 --seed 20261018 --dry-run
```

Einzelbild-Testlauf (eigener Ausgabeordner, `yolo_mask_winners` bleibt sauber):

```bash
uv run python yolo_mask_auto_select.py --image f002_004.png --output ~/Downloads/Cleanup/test_runs
```

Vollständiger Lauf im Hintergrund (GPU währenddessen **frei lassen**, siehe 11.5):

```bash
nohup bash -c 'time uv run python yolo_mask_auto_select.py --output ~/cleanup_compare/<NAME>' > /tmp/full.log 2>&1 &
tail -f /tmp/full.log | grep -E "Batch|real|ERROR|COMPLETE"
```

`fill_repair` ein-/ausschalten für Vergleiche: `--no-fill-repair` bzw.
`FILL_REPAIR_ENABLED` in `config.py`. (Die frühere Umgebungsvariable
`FILL_REPAIR=0/1` existiert nicht.)

Routing-Audit-Hilfe:

```bash
uv run python yolo_mask_routing_audit.py --help
```

Manuelle Kandidatenbewertung (braucht die Debug-Bilder, also **nicht** mit
`--no-debug-images` erzeugte Läufe):

```bash
uv run python label_tool.py --run-dir <LAUFVERZEICHNIS>
```

Ranker-Training benötigt Score- und Label-CSV aus demselben Bewertungsworkflow:

```bash
uv run python train_ranker.py --scores <candidate_scores.csv> --labels <labels.csv> --out <ranker_out>
```

Finale Auswahl:

```bash
uv run python build_final_picks.py --run-dir <LAUFVERZEICHNIS>
```

Repository-Cleanup:

```bash
uv run python scripts/cleanup_repo.py
uv run python scripts/cleanup_repo.py --apply
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
- **Grundsatz (16.11.2026):** Fehler nach Möglichkeit im Cleaning (Maske,
  Inpainting, Alpha-Behandlung) beheben und nicht durch Anpassen von
  Score-Zahlen. Score-Änderungen verschieben Fehler leicht auf andere Bilder.
- Vor jeder Analyse prüfen, **welchem Kandidaten** eine Masken- oder Bilddatei
  gehört (am 16.11. war `d002_001.png` die YOLO-Maske D4–D10, nicht die
  Gewinnermaske) und ob alle Dateien aus demselben Lauf stammen.
- Kontaktbogen-JPGs sind komprimiert und bei breiten Bildern verkleinert;
  für Pixel-Analysen die PNGs aus `cleaned_images/` verwenden.

## 9. Arbeitsstand 16. November 2026: Cleaning-Artefakte

### 9.1 Problem

Nach dem Cleaning bleiben bei einzelnen Bildern Artefakte zurück:
dunkle Punkte/Ringe (`d002_001`), ein schwarzer Punkt in einem Button
(`f002_004`) und dunkle, halbtransparente Flecken an Glyph-Positionen bei
Sprites mit dunklem Glow (`m001_001`, `n004_003`, `n005_000`).

### 9.2 Befunde

**`d002_001` (Panels mit Ringen/Punkten)**
- Bestätigt: Die Maske des Gewinners (`CONTRAST_T38`, Maskendatei
  `d002_001_b3a150.png` aus dem Lauf) deckt Text und Halo vollständig ab
  (0 Alpha-Pixel ≥ 215 und 0 RGB-Pixel > 60 außerhalb der Maske).
  Die Ringe waren daher von LaMa erzeugt, kein Abdeckungsproblem.
- Widerlegt: Die erste Hypothese „Maske zu knapp“ gilt nicht für diesen
  Gewinner.
- Nutzermeldung: `d002_001` ist inzwischen „gefixt“, vermutlich durch bessere
  Boxen. Von mir nicht verifiziert.

**YOLO-Masken (D0–D10) und „。“**
- Bestätigt (`d002_001`): Die YOLO-Maske lässt zwei „。“ am Ende von Zeile 2 in
  Panel 2 und 3 teilweise außerhalb (99 bzw. 70 Alpha-Pixel, Abstand 1 px
  zur Maske). Box-only-, Contrast- und Hybrid-Masken decken sie ab.
- Prototyp (nicht eingebaut): hochkontrastige Komponenten, die ≤ 2 px an der
  Maske liegen und teilweise außerhalb liegen, vollständig in die Maske
  aufnehmen. Ergebnis: Maske 30.241 → 30.685 px, danach kein Alpha-Rest außerhalb.
  Zurückgestellt, weil der Gewinner das Problem nicht hatte.

**Alpha-Kanal wird von LaMa „erfunden“**
- Bestätigt (`d002_001`): Höchster Alpha im veränderten Bereich 253 bis 254
  (CONTRAST_T38, D4, BOX_ONLY) bei Original-Maximum 245; HYBRID_D6 bleibt bei
  206. Ein deterministischer Fill (cv2 TELEA) liefert 203 bis 209.
- Ursache im Code: Alpha wird als Graubild separat durch LaMa geschickt
  (`pipeline.py`, Composite-Block).

**`f002_004` (Sprite-Sheet mit Buttons)**
- Bestätigt: Gewinner `SMART_INK` (Route `SMART_INK_COMPETITION_SELECTED`,
  Grund `BOX_MASK_FALLBACK_WEAK_COVERAGE`) hat einen schwarzen Punkt
  (ca. 12×11 px, RGB (6, 14, 22) auf Umfeld (86, 3, 31)) innerhalb der
  Gewinnermaske (`f002_004.png` im Lauf).
- Bestätigt: `CONTRAST_T38` hat keinen Punkt. SMART_INK gewinnt über den
  Struktur-Score (0,479 vs. 0,532); beide sind eligible
  (`text_quality_score` 0,573 vs. 0,560, Toleranz 0,02).
- Bestätigt: Die Metriken erkennen den Punkt nicht (Micro-Komponenten SMART_INK
  6/64 px vs. CONTRAST_T38 13/104 px).
- Beobachtet: D0–D10 lassen Textgeister stehen; Hybrid und Box-only erzeugen
  waagerechte Schlieren.
- Spaltenzuordnung der CSV-Zeilen erfolgte über den Header aus
  `candidate_scores_1.csv` (109 Spalten; Annahme gleiches Schema).

**`m001_001` (schwarzer Glow, Box-only gewinnt)**
- Bestätigt: Die Box-only-Maske deckt 29.892 von 35.685 px ab (84 %).
- Bestätigt: Im Original haben 27.407 px in der Maske Alpha > 10 (Mittel 161);
  der „schwarze Glow“ ist ein halbtransparenter dunkler Hof hinter den
  Glyphen. Außerhalb der Maske: Alpha Mittel 6,9, Maximum 66.
- Bestätigt: LaMa-Ergebnis in der Maske hat Alpha Mittel 76,5, Maximum 189
  (16.685 px > 60); ein deterministischer Fill (cv2 NS) hat Mittel 17,5,
  Maximum 63 (302 px > 60), mit schwachen senkrechten Schlieren links.
- Hypothese: Ein deterministischer Alpha-Fill löst diese Bildklasse. Für
  `n004_003` und `n005_000` liegen nur verkleinerte Kontaktbögen vor.

### 9.3 Aktuelle Implementierung von `fill_repair.py` (Stand 17.11.2026)

Die alte Beschreibung als Navier-Stokes-Vergleich ist überholt. Der aktuelle
Code in `yolo_mask_auto_select_modules/fill_repair.py` verwendet einen
Medianfilter als lokale Hintergrundreferenz:

1. Medianfilter über das gefüllte RGB-Bild (`median_ksize=25`).
2. Abweichende Pixel innerhalb der aktuellen Kandidatenmaske werden über
   maximale Kanalabweichung erkannt (`dev_levels=11`).
3. Zusammenhängende Cluster werden nach Mindestgröße gefiltert
   (`min_cluster_px=8`).
4. Cluster, deren Struktur über den Maskenrand hinaus weiterläuft und
   länglich ist, werden übersprungen.
5. Der Umgebungsring wird auf Homogenität geprüft
   (`ring_inner=3`, `ring_outer=8`, `ring_keep_levels=15`,
   `ring_min_keep_frac=0.6`, `flat_ring_std=12.0`).
6. Geeignete Cluster werden innerhalb der Maske und außerhalb geschützter
   Zonen mit einer weichgezeichneten Medianreferenz ersetzt
   (`grow_px=5`, `soften_sigma=2.0`).

Die Funktion verändert ausschließlich RGB-Pixel innerhalb der Maske.
Geschützte Pixel bleiben unangetastet; Alpha wird von dieser Funktion nicht
bearbeitet. Die Parameter sind direkt in `fill_repair.py` definiert und
nicht in `config.py` zentralisiert.

**Integration in der Pipeline (Stand 17.11.2026, bestätigt im Quellcode):**
- `pipeline.py` importiert `repair_flat_fill_artifacts`.
- Der Aufruf erfolgt unmittelbar nach dem Einsetzen des LaMa-RGB-Ergebnisses
  in die Kandidatenmaske.
- **Neu:** Der Aufruf ist über `FILL_REPAIR_ENABLED` (`config.py`) bzw.
  `--no-fill-repair` schaltbar (Standard: an, also unverändertes Verhalten).
- **Neu:** Die `fr_meta`-Zähler (`fill_repair_clusters_found`,
  `fill_repair_clusters_repaired`, `fill_repair_pixels_replaced`) stehen pro
  Kandidat in `candidate_scores.csv` und für den Gewinner in
  `winner_report.csv`.
- **Neu:** `repair_flat_fill_artifacts` ist ein Wrapper, der die bisherige
  Funktion (jetzt `_repair_core`) auf einem Bildausschnitt um die Maske
  (Rand: `median_ksize//2 + ring_outer + grow_px + 24` px) ausführt. Das
  Ergebnis ist bitgleich zur Vollbild-Berechnung (Bestätigt: 60
  Zufallsszenen im Test, außerdem identische Scores im 40-Bilder-Lauf). Ist
  der Ausschnitt ≥ 80 % des Bildes, wird wie früher das ganze Bild berechnet.
  Gemessene Wirkung: gering (Abschnitt 11.4), weil die Maskenbereiche der
  Gewinner meist fast das ganze Bild abdecken (Bestätigt für die
  Gewinnermasken der 40-Bilder-Stichprobe: 78 % im Rückfall).

### 9.4 Aktueller gespeicherter Testlauf: `run_20261116_225034`

Im Repository liegt ein Ergebnislauf vom 16.11.2026 mit genau einem
ausgewerteten Bild: `f002_004.png`.

**Aus dem Laufbericht bestätigt:**
- Gewinner: `SMART_INK`.
- Route: `SMART_INK_COMPETITION_SELECTED`.
- Routing-Audit-Vorschlag: `BOX_MASK_FALLBACK_WEAK_COVERAGE`.
- Ausgewiesener Score: 0,479091.
- Textqualität: 0,572759.
- Struktur-Score: 0,479091.
- Beste Textqualität eines Kandidaten: 0,559216.

Die zugehörige `candidate_scores.csv` enthält sechs Distanzkandidaten
(0, 2, 4, 6, 8, 10) sowie `SMART_INK`, `HYBRID_D2`, `HYBRID_D6`,
`CONTRAST_T38` und `BOX_ONLY`. Der Gewinnerbericht weist SMART_INK aus.
Dieser gespeicherte Lauf ist **kein vollständiger Reprocess** und belegt
nicht, dass der verbliebene schwarze Punkt visuell vollständig entfernt ist.
Die Ergebnisdaten enthalten in diesem Laufbericht keine belastbare
Vorher/Nachher-Auswertung mit deaktivierter und aktivierter Reparatur.

Wichtig: Der Laufordnername und die CSV-Dateien belegen einen Ein-Bild-Lauf.
Ob die darin gespeicherten PNGs den aktuellen Code exakt widerspiegeln,
muss bei einer erneuten visuellen Prüfung anhand der Laufdateien und des
zugehörigen Commits abgeglichen werden.

### 9.5 Was gegenüber früheren Übergaben korrigiert wurde

- Der alte Navier-Stokes-Ablauf samt den Parametern
  `disagree_levels=45`, `flat_ring_std=20`, `grow_px=2` und
  `inpaint_radius=4` ist nicht mehr die aktuelle Implementierung.
- Die alte Aussage, der Import fehle und der Pipeline-Aufruf breche mit
  `NameError` ab, ist überholt.
- Die früheren Offline-Ergebnisse (z. B. 126 ersetzte Pixel bei
  `f002_004`) gehören zur alten Methode.
- **Seit 17.11.2026 erledigt:** `FILL_REPAIR`-Schalter (als Config-Wert und
  CLI-Flag, nicht als Umgebungsvariable), `fr_meta` im Report,
  Unit-Tests für `fill_repair` (`test_dedup_and_fill_repair.py`,
  `test_fill_repair_crop.py`).

### 9.6 Nächste sinnvolle Schritte

1. **Gezielter visueller Test:** die fünf Problembilder (`d002_001`,
   `f002_004`, `m001_001`, `n004_003`, `n005_000`) mit dem aktuellen Stand
   ausführen und Original, Gewinner-Maske und Gewinnerbild aus **demselben**
   Lauf prüfen. Bisher gibt es dafür nach den Änderungen keine Sichtprüfung.
2. **Wirkung von `fill_repair` messen:** Vergleichslauf mit und ohne
   `--no-fill-repair` über dieselben Bilder; die neuen Zähler in
   `candidate_scores.csv` zeigen, bei wie vielen Kandidaten die Reparatur
   überhaupt greift. Erst danach entscheiden, ob Parameter geändert werden.
3. Keine Score-Gewichte oder Auswahlregeln verändern, um einen
   Cleaning-Fehler zu kaschieren (Grundsatz aus Abschnitt 8).

## 10. Bekannte Dokumentationsgrenzen

Diese Übergabe beschreibt den am 18. November 2026 bekannten Stand anhand der
oben genannten Dateien, Messungen und Analysen. Sie ersetzt keine Prüfung der aktuellen
lokalen Laufdaten und aktualisiert sich nicht automatisch.

Nicht verifiziert (Stand 18.11.2026): die visuelle Qualität der Ergebnisse
nach den Optimierungen (Scores und Gewinner sind in den Vergleichsläufen
identisch, Bilder wurden nicht angesehen), der Zustand von `d002_001`
(„gefixt“ laut Nutzer), die Auswertung der Ergebnisdateien des
716-Bilder-Laufs `FULL_stage5` und ein Vergleich dieses Laufs mit einem
früheren vollständigen Lauf. Die Sandbox-Tests (Dummy-YOLO/-IOPaint) belegen
keine GPU-Eigenschaften.

Bei einem neuen Arbeitsschritt immer den aktuellen Quellcode, die letzten
relevanten Testergebnisse und die konkreten zu untersuchenden Bilder oder
Reports heranziehen.

## 11. Arbeitsstand 17./18. November 2026: Performance-Optimierung

Ziel war ein deutlich schnellerer vollständiger Lauf **ohne** Änderung der
Reinigungsergebnisse. Alle Stufen wurden an derselben 40-Bilder-Stichprobe
(`--random-sample 40`, Standard-Seed) gegen den Lauf `B_dedup` verglichen:
`candidate_scores.csv` (ohne die neue Spalte `mask_is_duplicate`),
Gewinner (`selected_distance_px`) und `winner_report.csv` waren in **jedem**
Vergleichslauf identisch (Bestätigt, Nutzerausgaben). Das belegt
Reproduzierbarkeit von LaMa/GPU zwischen Läufen mit gleichen Eingaben, aber
nur für diese 40 Bilder.

Einschränkung der Baseline: Referenzlauf A (`--no-dedup`, Batch 25) lief mit
dem **gepatchten** Code im alten Verhalten, nicht mit dem unveränderten
Original. Die Gleichwertigkeit zum Original (Maskensatz nur einmal bauen,
Outline-Leak im Batch statt vorab) ist durch Code-Lesen begründet, nicht durch
einen Lauf des unveränderten Codes belegt.

### 11.1 Befunde der Analyse (Bestätigt)

- Pro Bild: 11 Kandidaten × 2 (RGB + Alpha) = 22 LaMa-Eingaben.
- Der Maskensatz (YOLO-PNGs laden, Contrast-/Smart-Ink-Masken) wurde pro Bild
  zweimal gebaut (Outline-Leak-Vorberechnung + Batch).
- `fill_repair` lief immer; seine Metriken wurden verworfen.
- Das YOLO-Audit schrieb pro Bild Overlays für alle Distanzen, Box-only und
  Kontaktbögen in einen temporären Ordner, der nach dem Lauf gelöscht wird.
  Das Audit läuft mit Konfidenz 0,05 und 0,10; die Pipeline liest nur `0.05`
  (`0.10` ist laut Audit-Code nur eine Metrik-Kontrolle, bleibt unverändert).
- IOPaint arbeitet mit `HDStrategy.CROP` (Trigger 800 px, Rand 128 px)
  und verarbeitet Dateien einzeln nacheinander.
- Alpha-LaMa für deckende Bilder auszulassen ist **verworfen**: Nur 1 von 40
  Bildern hat durchgehend Alpha 255, und bei diesem liefert LaMa in allen 11
  Kandidaten nicht exakt 255 zurück (Pixelwerte würden sich ändern).

### 11.2 Umgesetzte Änderungen

| Stufe | Änderung | Ergebnis-Auswirkung |
|---|---|---|
| 1 | Identische Masken nur einmal durch LaMa/Compositing/Scoring (`DEDUPLICATE_IDENTICAL_MASKS`, `--no-dedup`); Reports behalten alle Kandidaten-IDs, Metriken werden kopiert; `mask_dedup_report.csv`. Maskensatz nur einmal pro Bild. `FILL_REPAIR_ENABLED`/`--no-fill-repair`; `fr_meta` in Reports. `run_stats.json` (Phasenzeiten, LaMa-Jobs). `--no-debug-images`. | keine (bestätigt) |
| 2 | `fill_repair` auf Ausschnitt um die Maske (bitgleich); Zeit je IOPaint-Batch in `run_stats.json`. | keine (bestätigt) |
| 3 | Audit mit `--no-visuals` (Pipeline setzt es immer); `LAMA_BATCH_SIZE` 25 → 100. | keine (bestätigt) |
| 4 | `LAMA_PARALLEL_PROCESSES = 2` (`--lama-processes`): Eingaben reihum auf N Shard-Ordner, N IOPaint-Prozesse gleichzeitig, gleicher Ausgabeordner. | keine (bestätigt für 2) |
| 5 | `POSTPROCESS_WORKERS = 6` (`--postprocess-workers`): Thread-Pool über die Bilder eines Batches (Reihenfolge bleibt); OpenCV-Threads während des Pools auf 1. | keine (bestätigt) |

`run_stats.json` zählt `composite_and_postprocess`, `fill_repair` und
`scoring_and_debug_images` als **Summe der Thread-Sekunden**; die echte Dauer
des Blocks steht in `postprocess_wall_clock`. `fill_repair` ist in
`composite_and_postprocess` enthalten (nicht addieren).

### 11.3 Messwerte (40-Bilder-Stichprobe, Bestätigt aus Nutzerausgaben)

| Lauf | Stand | Gesamt (`total_wall_clock`) | LaMa |
|---|---|---|---|
| A | Original-Verhalten (`--no-dedup`, Batch 25) | 456 s | 316,5 s |
| B | + Dedup (LaMa-Läufe 880 → 610, −30,7 %) | 401 s | 279,6 s |
| C/D | + `fill_repair`-Ausschnitt | 386 / 390 s | 271 / 277 s |
| E | ein Batch (40) | 347 s | 231,1 s |
| F | + Audit ohne Visuals, Batch 100 | 351 s | 237,6 s |
| H | wie F, `--no-debug-images` | real 5m29 | – |
| I_p2 | + 2 IOPaint-Prozesse | 243 s | 145,7 s |
| K_pp1 | wie I_p2, 1 Nachbearb.-Thread | 243 s | 145,6 s |
| K_pp6 | + 6 Nachbearb.-Threads | **183 s** | 150,8 s |

Rauschen zwischen identischen Läufen: etwa 2–3 % (C vs. D). Der erste Lauf A
war vermutlich durch kalte Dateicaches leicht erhöht (auch das unveränderte
YOLO-Audit war dort ~10 % langsamer als in B). Der Gewinn durch Dedup allein
ist deshalb nicht sauber von diesem Effekt zu trennen.

Phasenverteilung Lauf A (Anteile): LaMa 69 %, `fill_repair` 16 %, Scoring +
Debug-Bilder 6,5 %, YOLO-Audit 5,5 %, Maskenbau 1,6 %.
Debug-Bilder speichern kostet ~7,5 s je 40 Bilder (24,1 → 16,6 s).
`yolo_routing_audit` fiel in den späteren Läufen von ~22 s auf ~6 s;
**Ursache offen** (Hypothese: warmer MIOpen-Kernelcache).

GPU-Auslastung im Lauf F (sysfs, 1-s-Takt, Bestätigt): Gesamtlauf im Mittel
25 %; im LaMa-Fenster (198 s) im Mittel 42 % und 42 % der Zeit unter 10 %;
87 s nach dem LaMa-Fenster Leerlauf (CPU-Nachbearbeitung); VRAM-Spitze
1,56 GB bei ~0,6 GB Grundlast.

**Vollständiger Lauf `FULL_stage5` (716 Bilder, Stufe 5, 2 IOPaint-Prozesse,
Batch 100):** `real 52m20,6s`. IOPaint je Batch:

| Batch | Bilder | Eingaben | Zeit | s/Eingabe |
|---|---|---|---|---|
| 1 | 1–100 | 1658 | 517,9 s | 0,31 |
| 2 | 101–200 | 1630 | 481,0 s | 0,30 |
| 3 | 201–300 | 1672 | 237,3 s | 0,14 |
| 4 | 301–400 | 2112 | 805,6 s | 0,38 |
| 5 | 401–500 | 1504 | 206,0 s | 0,14 |
| 6 | 501–600 | 1366 | 151,6 s | 0,11 |
| 7 | 601–700 | 1326 | 126,4 s | 0,10 |
| 8 | 701–716 | 230 | 97,4 s | 0,42 (Startzeit dominiert) |

Summe IOPaint ≈ 2622 s (≈ 43,7 min, ~83 % des Laufs). Die Zeit je Eingabe
schwankt zwischen Batches um mehr als Faktor 3 (Batches 1, 2, 4 deutlich
langsamer). **Ursache offen** (Hypothesen: größere Bilder in diesen Batches
wegen der Crop-Strategie; Fremdlast auf der GPU; nicht geprüft).
Ursprüngliche Hochrechnung des Ausgangsstands: ≈ 136 min (aus der
40-Bilder-Messung A extrapoliert, **kein** gemessener Wert); Hochrechnung der
Stufe 5: 50–60 min, gemessen 52,3 min.

### 11.4 Was wenig brachte / Einordnung

- `fill_repair`-Ausschnitt: nur ~11 % weniger `fill_repair`-Zeit, weil die
  Maskenbereiche meist fast das ganze Bild abdecken (siehe 9.3). Profil an
  einem Bild: ~96 ms je Aufruf; Medianfilter ≈ die Hälfte, Dilatationen/
  Erosionen (~220 Aufrufe) und Gaußfilter der Rest.
- Audit ohne Visuals: ~6 % der Audit-Zeit.
- Ein IOPaint-Start kostet geschätzt ~44 s Fixzeit (Dreipunktschätzung aus
  Läufen mit 1 bzw. 2 Batches, grob).

### 11.5 Betriebshinweise (wichtig)

- **Die GPU muss während des Laufs frei sein.** Zwei Testläufe mit
  `--lama-processes 2` und `3` brachen ab, während der Nutzer nebenher ein
  anderes GPU-Programm nutzte (VRAM-Spitze 16,2 GB, kein
  `candidate_scores.csv`). Die Fehlermeldung wurde nicht gesehen (die Ausgabe
  war gefiltert); die Wiederholung mit freier GPU lief fehlerfrei und lieferte
  identische Ergebnisse. Ursache daher **sehr wahrscheinlich
  Speicherknappheit durch die Fremdlast** (Hypothese, nicht durch die
  Fehlermeldung belegt).
- Der Test mit `--lama-processes 3` steht aus (laut Nutzer reichen zwei
  vorerst).
- Bei einem Fehler in einem IOPaint-Prozess bricht der Lauf ab (kein
  stilles Weiterarbeiten).
- Größere Batches bedeuten größeren Verlust bei Abbruch (CSV-Reports werden
  nach jedem Batch geschrieben, ein Batch hat jetzt bis zu 100 Bilder).
- `/tmp` ist ein tmpfs (31 GB, im RAM); Staging-Daten eines Batches liegen
  deutlich darunter, sind aber nicht kostenlos.
- Für Läufe, die mit `label_tool.py` bewertet werden sollen, **nicht**
  `--no-debug-images` verwenden.

### 11.6 Migrationsskripte `optimize_cleanup/`

`optimize_cleanup.py` (Stufe 1) sowie `optimize_cleanup_stage2.py` bis
`optimize_cleanup_stage5.py`. Jede Datei ist eigenständig (Patch eingebettet),
benötigt nur die Standardbibliothek und läuft von jedem Verzeichnis:

```bash
uv run python optimize_cleanup/optimize_cleanup_stageN.py            # Dry-Run, ändert nichts
uv run python optimize_cleanup/optimize_cleanup_stageN.py --apply    # Backup, Patch, Validierung
uv run python optimize_cleanup/optimize_cleanup_stageN.py --restore <BACKUP_DIR>
```

Eigenschaften (Bestätigt im Test gegen eine Kopie des Ausgangsstands; alle
fünf Stufen wurden vom Nutzer erfolgreich angewendet):
- Stufen müssen **in Reihenfolge** angewendet werden; jede prüft per SHA-256,
  dass die Zieldateien exakt dem erwarteten Stand entsprechen, und bricht
  sonst ab (keine heuristischen Textersetzungen). **Folge:** Wer eine dieser
  Dateien (`pipeline.py`, `config.py`, `cli.py`, `candidate_masks.py`,
  `fill_repair.py`, `yolo_mask_routing_audit.py`, die Testdatei
  `test_dedup_and_fill_repair.py`) von Hand ändert, blockiert spätere
  Stufen; nicht ausgeführte Stufen dann neu abgleichen.
- Backup vor dem Patch nach `~/cleanup_backups/<Projekt>_backup_<Zeitstempel>`
  (außerhalb des Repositorys, nie überschrieben); ausgenommen sind `.venv`,
  `.git`, `test_runs/`, `diagnostics/`, `yolo_mask_winners/`.
- Bericht je Ausführung in `optimization_reports/<Zeitstempel>/`
  (`inventory.json`: AST-Bestandsaufnahme, `report.md`).
- Die Skripte ändern weder Originalbilder noch Ergebnisordner noch `.venv`.

### 11.7 Neue Ausgabedateien und Spalten

- `<Ausgabe>/run_stats.json`: Phasenzeiten, Zähler (`candidates_total`,
  `candidates_deduplicated`, `lama_mask_jobs`), `lama_runs_rgb_plus_alpha`,
  `lama_runs_without_dedup`, `lama_batches` (Eingaben, Prozesse, Sekunden),
  `postprocess_workers`, Schalterstände.
- `<Ausgabe>/mask_dedup_report.csv`: je zusammengelegtem Kandidaten
  `image`, `candidate`, `identical_to`, `mask_pixels`.
- `candidate_scores.csv`: neue Spalten `mask_is_duplicate` (0/1) und
  `fill_repair_*`; `winner_report.csv`: `fill_repair_*`.
  Spaltenvergleiche mit älteren Läufen müssen diese Spalten ausnehmen.

### 11.8 Offene Punkte (priorisiert)

1. **Ergebnisse von `FULL_stage5` auswerten** (Gewinnerhäufigkeit je Kandidat,
   Dedup-Quote über alle 716 Bilder, Bildtypen, Zeit je Phase) und mit einem
   früheren vollständigen Lauf vergleichen, z. B. dem Lauf
   `full_recheck_after_alpha_island_025_20261025_022053` (nur vergleichen,
   wenn Dateien aus demselben Lauf stammen; Codestand und Konfiguration
   können sich unterscheiden).
2. **Sichtprüfung der fünf Problembilder** (Abschnitt 9.6).
3. **Kandidaten- und Scoring-Analyse** (Auftrag Abschnitt 3/4): welche
   Methoden gewinnen wie oft, welche Masken sind identisch bzw. fast
   identisch, welche Metriken ändern Gewinner, Kosten je Metrik. Grundlage
   sollten die Daten aus Punkt 1 sein; `scoring.py` (1525 Zeilen) wurde bisher
   nur im Überblick gelesen, nicht analysiert. Keine Methode entfernen oder
   Gewichte ändern ohne diese Daten.
4. **Weitere Laufzeit:** IOPaint macht ~83 % der Zeit. Optionen: Ursache der
   stark schwankenden Zeit je Eingabe klären (Batches 1, 2, 4); Nachbearbeitung
   eines Batches mit dem LaMa-Lauf des nächsten überlappen (die GPU wartet
   sonst während der Nachbearbeitung); `--lama-processes 3` testen; weniger
   Kandidaten (nur nach Punkt 3).
5. **Umstrukturierung** nach dem Zielbild des Projektauftrags
   (`src/cleanup/`, Pakete `pipeline/masks/cleaning/scoring/io/debug`,
   `tests/`, `scripts/`, `tools/`): **noch nicht begonnen.** Erst nach
   Punkt 1–3, da Dateiverschiebungen die hashgeschützte Stufenkette brechen
   und die Analyse auf dem aktuellen Code aufbaut.
6. **Dependency-Verwaltung** (`pyproject.toml`/`uv.lock`, ROCm-Besonderheiten,
   IOPaint): **noch nicht begonnen.** `requirements-before-iopaint.txt` ist
   weiterhin nur ein historischer Snapshot. Die funktionierende
   ROCm-Umgebung nicht anfassen.
7. **Entscheidung über Debug-Standardwerte:** `DEBUG_SAVE_*` stehen weiter auf
   `True` (Standardverhalten unverändert), weil `label_tool.py` und
   `build_final_picks.py` die Kandidatenbilder lesen; `--no-debug-images`
   existiert als Produktivschalter.
8. Die Aufräumarbeiten aus dem Auftrag (Dateiumbenennungen, zentrale Pfade,
   Handoff nach der Umstrukturierung erneut anpassen) sind offen.
