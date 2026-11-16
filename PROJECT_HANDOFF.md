# PROJECT HANDOFF — Cleanup YOLO/LaMa

Stand: 16. November 2026
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
  (darin entsteht `run_JJJJMMTT_HHMMSS`).

`yolo_mask_auto_select_modules/candidate_masks.py`
  Baut die Maskenkandidaten (Distanz-, Box-only-, Smart-Ink-, Contrast-,
  Hybrid-Varianten) und `enforce_protection`.

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

`yolo_mask_auto_select_modules/debug_output.py`
  Debug-Overlays, Kontaktbögen und Vorher-/Nachher-Vergleiche.

`yolo_mask_auto_select_modules/reports.py`
  CSV-Ausgabe mit der Vereinigung der bevorzugten und tatsächlich vorhandenen
  Metrikfelder. Verhindert Abbrüche, wenn neue Score-Felder hinzukommen.

`yolo_mask_auto_select_modules/fill_repair.py`  **(NEU, 16.11.2026, Integration noch nicht abgenommen)**
  Post-Fill-Sanity-Check für LaMa-Ergebnisse, siehe Abschnitt 9.

## 3. Eigenständige Hilfsprogramme

Diese Programme haben getrennte Aufgaben und sind keine Duplikate der
Hauptpipeline:

`yolo_mask_routing_audit.py`
  Eigenständiges YOLO-Inferenz-/Diagnosewerkzeug. Verwendet dieselben
  Standardpfade für `anchors.csv`, Quellbilder und optionale Schutzzonen
  wie die Hauptpipeline. Prüft Maskenabdeckung, Abstand zu OCR-Ankern und
  Body-Box-Coverage. Erzeugt unter anderem `routing_report.csv`,
  `body_box_coverage.csv`, `instances.csv` und visuelle Kontaktbögen.
  Kein LaMa-Lauf und keine Änderung der Quelldateien.

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

Teststand: Laut Nutzer lief die Suite am 16.11.2026 **vor** dem Einbau von
`fill_repair` ohne Probleme durch. Nach dem Einbau wurde sie nicht erneut
ausgeführt; für `fill_repair.py` existieren noch keine Unit-Tests.

Kleiner Pipeline-Dry-Run:

```bash
python yolo_mask_auto_select.py --random-sample 10 --min-m-images 3 --seed 20261018 --dry-run
```

Einzelbild-Testlauf (eigener Ausgabeordner, `yolo_mask_winners` bleibt sauber):

```bash
python yolo_mask_auto_select.py --image f002_004.png --output ~/Downloads/Cleanup/test_runs
```

Vorher/Nachher-Vergleich für `fill_repair` (Schalter per Umgebungsvariable,
sofern so in `pipeline.py` eingebaut, siehe Abschnitt 9):

```bash
FILL_REPAIR=0 python yolo_mask_auto_select.py --image f002_004.png --output ~/Downloads/Cleanup/test_runs
FILL_REPAIR=1 python yolo_mask_auto_select.py --image f002_004.png --output ~/Downloads/Cleanup/test_runs
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

### 9.3 `fill_repair.py` (RGB-Post-Fill-Check)

Funktion `repair_flat_fill_artifacts(original_rgb, filled_rgb, mask, protected)`
→ `(repaired_rgb, meta)`.

Ablauf:
1. Deterministischer Vergleichsfill der Maske mit cv2 Navier-Stokes.
2. Cluster in der Maske, in denen LaMa und NS um mehr als 45 Stufen
   (max. Kanaldifferenz) abweichen, mindestens 25 px groß.
3. Nur wenn das Umfeld des Clusters flach ist (Ring-Std ≤ 20) und NS klar näher
   am Umfeld liegt als LaMa (Faktor 2), wird der Cluster (+2 px) durch NS ersetzt.
4. Geschützte Zonen werden nicht verändert. Nur RGB, Alpha bleibt unberührt.

Standardparameter: `min_cluster_px=25`, `disagree_levels=45`,
`flat_ring_std=20.0`, `ring_inner=2`, `ring_outer=6`, `grow_px=2`,
`inpaint_radius=4`. Meta-Felder: `fill_repair_clusters_found`,
`fill_repair_clusters_repaired`, `fill_repair_pixels_replaced`.

Offline-Testergebnisse (gespeicherte PNGs, ohne Pipeline-Lauf):
- `f002_004` SMART_INK: 2 Cluster gefunden, 1 repariert (126 px); Punkt von
  (6, 14, 22) auf (63, 9, 32). Ein schwach dunklerer Kreisrand bleibt.
- `f002_004` CONTRAST_T38: 14 Cluster gefunden, 0 repariert (alle an Rahmen-
  linien, Ring-Std ≥ 40,9; der Punkt hatte 12,8). Der Grenzwert 20 beruht auf
  diesem Abstand und ist **an nur zwei Bildern abgeleitet**.
- `d002_001` (Bild aus dem Vormittagslauf, `CONTRAST_T38`): Ring in Panel 4
  großteils entfernt, zwei Blobs in Panel 5 bleiben; höchster RGB-Wert in der
  Maske 77 → 52. Die Reparatur ist also unvollständig.

Einbau in `pipeline.py` (Composite-Block, direkt nach
`f_rgb[c_mask] = pred_rgb[c_mask]`):

```python
from .fill_repair import repair_flat_fill_artifacts   # bei den relativen Imports

fr_meta = {}
if os.environ.get("FILL_REPAIR", "1") == "1":          # Schalter (vorgeschlagen)
    f_rgb, fr_meta = repair_flat_fill_artifacts(original_rgb, f_rgb, c_mask, mask_set.protected)
...
a_meta.update(fr_meta)    # nach remove_small_alpha_islands(...)
```

Status: Der Nutzer hat die Aufrufzeile eingefügt; der erste Testlauf
(`--image f002_004.png`) brach mit `NameError: name 'repair_flat_fill_artifacts'
is not defined` ab, weil der Import in `pipeline.py` fehlte (ggf. auch
`import os`). Ob der Lauf nach dem Nachtragen des Imports durchläuft und was
der Schalter im Detail tut, ist **offen** (aktuellen Stand von `pipeline.py`
prüfen, nicht diesem Dokument vertrauen).

### 9.4 Entscheidungen und Rückmeldungen des Nutzers

- Fehler im Cleaning beheben, nicht durch Anpassen von Score-Zahlen.
- Alpha-LaMa-Lauf darf entfallen, wenn das die Ergebnisse nicht stark
  beeinflusst (halbiert die LaMa-Läufe pro Kandidat). Umsetzung als
  schaltbare Option, vorab Vergleich mit/ohne.
- Alpha-Fill: Nutzer hat keine Präferenz zwischen „für alle Bilder“ und
  „nur bei transparentem Umfeld“; vorgeschlagen ist die vorsichtige Variante.
- Es gibt nur wenige Beispiele für den Fehlertyp; außer dem schwarzen Glow
  keine weiteren ähnlichen Bilder.

### 9.5 Offene Punkte / nächste Schritte

1. Import in `pipeline.py` nachtragen, Testlauf `f002_004` mit
   `FILL_REPAIR=0` und `=1` vergleichen (Punkt sollte nur mit 1 verschwinden,
   falls SMART_INK wieder gewinnt; Gewinner kann abweichen).
2. Unit-Tests für `fill_repair.py` schreiben; Gesamt-Suite nach dem Einbau
   erneut ausführen.
3. Alpha-Fill (deterministisch, z. B. cv2 NS/TELEA auf dem Alpha-Kanal) mit
   Gate „Umfeld überwiegend transparent“ (Rand-Alpha bei `m001_001`: Mittel 6,9)
   prototypisch bauen und an `m001_001`, `n004_003`, `n005_000` prüfen.
   Grenzwert für das Gate ist unbekannt; Sprites mit echter Alpha-Struktur
   dürfen nicht schlechter werden. Dafür werden Originale (RGBA), Gewinner-
   masken und PNG-Kandidatenbilder dieser Bilder benötigt.
4. Optional: Komponenten-Vervollständigung für YOLO-Masken („。“, siehe 9.2).
5. Optional: LaMa pro Maskenregion mit Kontextrand statt auf dem ganzen Streifen
   testen (Hypothese: Ringe entstehen durch zu großen/schmalen Bildstreifen mit
   mehreren Panels; **nicht geprüft**, da LaMa in der Analyseumgebung nicht
   verfügbar war).
6. Vor einer Festlegung einen Regressionslauf auf einer Stichprobe
   (`--random-sample`) vorher/nachher durchführen und Reports/Bilder
   vergleichen.
7. Scoring-Beobachtungen (Abschnitt 4: Gate-Toleranz 0,02, Struktur-Score
   dominiert, Micro-Metrik erkennt dunkle Punkte nicht) sind dokumentiert, aber
   bewusst nicht geändert.

## 10. Bekannte Dokumentationsgrenzen

Diese Übergabe beschreibt den am 16. November 2026 bekannten Stand anhand der
oben genannten Dateien und Analysen. Sie ersetzt keine Prüfung der aktuellen
lokalen Laufdaten und aktualisiert sich nicht automatisch.

Nicht von mir ausgeführt oder verifiziert: LaMa-Läufe (in der Analyse-
umgebung nicht verfügbar), der aktuelle Inhalt von `pipeline.py` nach den
Nutzeränderungen, der Pipeline-Lauf mit `fill_repair`, die Unit-Tests nach dem
Einbau und die Behauptung, `d002_001` sei inzwischen gefixt.

Bei einem neuen Arbeitsschritt immer den aktuellen Quellcode, die letzten
relevanten Testergebnisse und die konkreten zu untersuchenden Bilder oder
Reports heranziehen.
