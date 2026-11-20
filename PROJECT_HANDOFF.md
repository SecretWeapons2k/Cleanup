# Projektübergabe — Cleanup

Stand: 20. November 2026  
Projektpfad: `~/Downloads/Cleanup`  
Zweck: Japanische Schrift/ausgewählte Textbereiche aus Bildern entfernen und
Sprite-Details, Konturen, Farben und Alpha möglichst erhalten.

Diese Übergabe ersetzt die vom 18. November. Sie beschreibt den Wechsel von
„ein Gewinner pro Bild per Scorer“ zu einem **Sieb mit Brauchbarkeits-Prüfung**.

## 1. Zielbild: Sieb statt Ranking

Hauptziel ist ein Cleaner, der **nur brauchbare Ergebnisse** liefert (angestrebt
99–100 %). Es wird nicht mehr der „beste“ Kandidat gesucht. Jede Textbox-Region
läuft durch Stufen, die erste brauchbare gewinnt:

1. **Stufe 1:** `BOX_ONLY` (alle Anchors minus Schutzzonen) + LaMa. Prüfung:
   brauchbar? Dann fertig.
2. **Stufe 2:** alternative Methoden (kleinere Masken, anderes Füllen).
3. **Stufe 3 (nicht gebaut):** YOLO-Masken (D0–D10) und neue Füllmethoden für
   Regionen, in denen Stufe 2 keine Alternative liefert.
4. **Nichts bestanden:** Region wird zur manuellen Prüfung markiert, nicht
   stillschweigend ausgeliefert.

Gründe: spart Rechenzeit (ein LaMa-Lauf statt ~11 Kandidaten), und die Frage
„brauchbar ja/nein“ ist besser messbar als „welcher ist der beste“. Bewusst in
Kauf genommen: ein besseres Ergebnis, das nie berechnet wird.

**Wichtig:** Die Prüfung ist noch **nicht gebaut/kalibriert**. Stufe 1 und 2
liefern derzeit Messwerte und Ausschnitte zum Labeln; ein Orchestrator, der die
Stufen verkettet und entscheidet, existiert noch nicht.

Fehlerarten sind ungleich teuer: ein falsches „brauchbar“ liefert ein schlechtes
Bild aus (vermeiden), ein falsches „unbrauchbar“ kostet nur Rechenzeit. Die
Prüfung ist deshalb bewusst streng einzustellen.

## 2. Befunde aus den Läufen (echte Daten, Stand 20.11.)

### 2.1 Ranking des alten Pools (20 Bilder, 51 Regionen)
- Nur 1 von 51 Regionen ohne brauchbaren Kandidaten: der Pool war fast immer gut.
- Der alte Scorer traf die Wahl nur in 19 von 50 Fällen (38 %). Er bevorzugte zu
  kleine Masken (D0: 23× gewählt, 17× widersprochen) und `CONTRAST_T38` (10×,
  8× widersprochen). Du wähltest in 40 von 50 Regionen `BOX_ONLY` (inkl. Gruppen
  pixelgleicher Kandidaten).
- Der Score ist relativ (bester Kandidat + Toleranz) und taugt nicht als
  absolute Prüfung.

### 2.2 Brauchbarkeit von `BOX_ONLY` (Stufe 1, 80 Bilder, 192 Regionen)
Gelabelt: 170 brauchbar, 22 unbrauchbar (88,5 % brauchbar). Gründe der 22:
Fleck/Artefakt 12, Textrest 6, Naht 1, Struktur 1, ohne Grund 2.

Der Umfeld-Typ der Maske (`ring_flat_frac`, Ring 2–10 px um die Maske) erklärt
den Ausfall am besten:

| Umfeld | Regionen | unbrauchbar |
|---|---|---|
| flach (≥ 0,90) | 124 | 2 (1,6 %) |
| gemischt (0,5–0,9) | 17 | 1 (5,9 %) |
| strukturiert (< 0,5) | 51 | 19 (37 %) |

- Strukturiertes Umfeld: die aktuellen Merkmale trennen brauchbar/unbrauchbar
  schlecht (AUC meist < 0,8). Eine Prüfung über alle Regionen zusammen, die alle
  Fehler fängt, würde ~81 % der guten Regionen verwerfen (5-fach nach Bildern
  validiert). Dort sollen andere Methoden zuerst probiert werden.
- Flaches Umfeld: die 2 Fehler sind winzige Textreste (7–8 px). Einfache
  Merkmale fangen sie nicht (brauchbare Regionen haben teils größere
  „Rest“-Werte). Ein neues Merkmal `unchanged_ink_px` (unveränderte Tinten-Pixel)
  soll sie fangen; mit 2 Beispielen ist keine Schwelle setzbar.
- Unbrauchbare Fälle haben meist große Masken in strukturiertem Umfeld; LaMa
  erfindet dort Inhalt. Das spricht für kleinere Masken in Stufe 2.
- `fill_repair` ersetzt auf flachem Grund dunkle Strukturen im Maskenbereich,
  auch stehengebliebene Streifen. Das kann echte Probleme verdecken. Alle Labels
  beziehen sich auf das Ergebnis **nach** der Reparatur.
- Die LaMa-Integritätsprüfung (Ausgabe außerhalb der Maske = Original) schlug
  bei keiner der 192 Regionen an.

### 2.3 Das „Abschneiden“ in den Ausschnitten
Beobachtet wurden harte Kanten und Fragmente am Rand der Original-Kachel.
Wahrscheinlichste Erklärung (nicht bestätigt): der Ausschnitt ist Region plus
48 px Rand (`REGION_SCORE_MARGIN_PX`) und das Fenster schneidet durch Nachbar-
Inhalt; die Original-Kachel ist ein unveränderter Ausschnitt des Quellbildes.
Bestätigung: Quellbild einmal komplett öffnen. Der Labeler zeigt dazu „Lage im
Bild“ (nur bei Läufen mit der neuen Stufe 1).

### 2.4 Grenzen der Daten
- 22 Negativbeispiele in 192 Regionen; viele Regionen stammen aus wenigen Bildern
  mit je 5 Boxen (ähnlich, nicht unabhängig). Immer nach Bildern gruppiert
  validieren.
- Für „Fehlerrate < 1 %“ (95 %) braucht es bei 0 Fehlern rund 300 geprüfte
  Regionen; die Obergrenze bei 192 Regionen und 0 Fehlern wäre ~2 %.
- Die Labels sind eine einzelne Beurteilung (deine). `o017_102` r1: Rest „passend“;
  `m003_019` r2: gleichfarbige Nachbarzellen bekommen den Rand hin (Notizen).

## 3. Dateien

Im Repo-Root: `yolo_mask_auto_select.py` (alte Pipeline), `yolo_mask_routing_audit.py`
(von der alten Pipeline als Unterprozess genutzt, nicht entfernen), neu siehe unten.

### Neu
| Datei | Zweck |
|---|---|
| `yolo_stage1.py` + `modules/stage1.py` | Stufe 1: nur `BOX_ONLY`, ohne YOLO/Audit; Integritätsprüfung, Merkmale, Ausschnitte, Übersichtsbild |
| `yolo_stage2.py` + `modules/stage2.py` | Stufe 2: Methoden-Varianten für riskante/fehlgeschlagene Regionen |
| `modules/gate_features.py` | absolute Prüfmerkmale (~35 Werte), Kontext-Typ, Asset-Merkmale |
| `modules/regions.py` | Regionen (Textbox-Cluster), Territorien, Auswahl/Zusammensetzen pro Region |
| `build_gate_labeler.py` | Schnell-Oberfläche: Box-only brauchbar? (J/N, Gründe, Rahmen zeichnen) |
| `build_ranker.py` | Oberfläche pro Region: Kacheln, beste wählen, Shift = unbrauchbar, Enter = fertig |
| `ranker_server.py` | lokaler Server: speichert CSVs nach `ranker_out/` (Autosave + Export) |
| `eval_ranking.py` | Auswertung der Ranking-CSV (Brauchbarkeit, Kaskade, Verwechslungen) |
| `modules/tests/test_regions.py`, `test_gate_features.py`, `test_stage2.py` | 17 neue Tests (in meiner Umgebung grün) |

### Geändert
- `modules/pipeline.py`: Auswahl pro Region (Modus nur bei ≥ 2 Regionen;
  Einzelregion-Bilder laufen den alten Pfad, mit `--no-per-region` ist das
  Ergebnis identisch zum Original), Region-Ausschnitte für den Ranker,
  `region_report.csv`.
- `modules/config.py`: `PER_REGION_SELECTION_ENABLED`, `REGION_MERGE_PX = 8`,
  `REGION_SCORE_MARGIN_PX = 48`.
- `modules/cli.py`: `--no-per-region`, `--region-merge-px`.

Unverändert: `scoring.py`, `candidate_masks.py`, `geometry.py`, `io_utils.py`,
`fill_repair.py`, `debug_output.py`, `reports.py`.

`modules` steht für `yolo_mask_auto_select_modules/`.

## 4. Ablauf und Befehle

```bash
# Stufe 1 (nur BOX_ONLY, schnell)
uv run python yolo_stage1.py --random-sample 80 --min-m-images 10 --seed 20261120 \
  --output ~/cleanup_compare/STAGE1_X

# Labeln: Server starten (öffnet http://127.0.0.1:8765/gate_labeler.html)
uv run python build_gate_labeler.py ~/cleanup_compare/STAGE1_X   # ggf. run_...-Unterordner

# Stufe 2 für riskante Regionen (nonflat + als unbrauchbar gelabelte)
uv run python yolo_stage2.py --stage1-run ~/cleanup_compare/STAGE1_X \
  --labels ranker_out/gate_labels_<id>.csv --output ~/cleanup_compare/STAGE2_X

# Stufe-2-Ergebnisse bewerten (Shift = unbrauchbar, Enter = fertig)
uv run python build_ranker.py ~/cleanup_compare/STAGE2_X

# Auswertung
uv run python eval_ranking.py ranker_out/ranking_<id>.csv ~/cleanup_compare/STAGE2_X
```

- Existiert der Ausgabeordner und ist nicht leer, legt die Pipeline einen
  Unterordner `run_JJJJMMTT_HHMMSS` an; die Builder dann mit diesem aufrufen.
- Die CSVs landen in `ranker_out/` (Autosave nach jeder Änderung, Export mit
  Zeitstempel). `--export-dir`, `--port`, `--no-serve`, `--no-browser` verfügbar.
- Der Browser-Speicher gilt pro Adresse. Über `127.0.0.1:8765` geöffnet sieht die
  Seite Labels von `file://` nicht; „CSV importieren“ nutzen. Ist Port 8765
  belegt, wird der nächste genommen (neue Adresse, eigener Speicher).
- Alte Stufe-1-Läufe (vor 20.11.) haben kein Übersichtsbild und keine
  `context_type`-Spalte; Stufe 2 leitet den Typ aus `ring_flat_frac` ab. Für
  „Lage im Bild“ Stufe 1 neu ausführen.
- Stufe 2 muss mit demselben `--region-merge-px` laufen wie Stufe 1 (wird aus
  `stage1_run_stats.json` übernommen und geprüft), sonst passen Regionsnummern
  nicht.
- Alte Pipeline: `uv run python yolo_mask_auto_select.py ...` (siehe README).

## 5. Stufe 2 im Detail

Auswahl (`--select nonflat`, Standard): Regionen mit Umfeld strukturiert/gemischt
plus alle gelabelten unbrauchbaren. Bei den echten Labels sind das 70 Regionen in
37 Bildern (`unusable`: 22 in 16, `all`: 192 in 80).

Methoden (alle ohne YOLO; Reihenfolge = vorgesehene Probierreihenfolge):
`SMART_INK`, `SMART_INK_R4` (Aufweitung 4), `SMART_INK_T10` (Schwelle 0,10,
Aufweitung 3), `CONTRAST_T38` (jeweils LaMa), `SMART_TELEA`, `SMART_R4_TELEA`
(OpenCV-Telea, deterministisch, erfindet keine Flecken).

Wichtig: `make_smart_ink_mask` fällt auf die **ganze Box** zurück, wenn es keine
Tinte findet, und findet auf rauschigem Hintergrund Tinte fast überall. Methoden,
deren Maske ≥ 85 % der Box deckt, sind `BOX_ONLY` in Verkleidung und werden
verworfen (`BOX_LIKE_COVERAGE`). Regionen ohne verbleibende Alternative stehen in
`regions_without_alternative.csv`: sie brauchen Stufe 3.

Ausgabe im Ranker-Format; `BOX_ONLY` aus Stufe 1 liegt als Referenz-Kachel dabei.

## 6. Prüfmerkmale (`gate_features.py`)

Nur Messwerte, keine Entscheidung. Gruppen: Kontext (`ring_flat_frac`,
`ring_luma_std`), Textrest (`res_ink_frac`, `ink_removed_ratio`,
`res_ink_max_cc_px`, `unchanged_ink_px`), Artefakte (`dev_frac`, `dark_cc_px`,
`bright_cc_px`, `dark_p99`, `mean_shift`, `std_ratio`), Naht (`seam_color_diff`,
`seam_grad_ratio`), Struktur über den Rand (`crossing_edge_px`,
`crossing_kept_ratio`), Alpha, Integrität (`lama_*`). Asset-Ebene:
`asset_type` (vorläufige Schwelle 0,75 auf `image_flat_frac`, Verteilung noch
ansehen), `image_edge_density`, `image_unique_colors`.
Kontext-Typ: flat ≥ 0,90, mixed ≥ 0,50, sonst structured.

## 7. Nicht verifiziert (bitte nicht als belegt behandeln)

- Stufe 2 ist nur mit Attrappen (OpenCV statt IOPaint) und synthetischen Bildern
  getestet, **noch nicht mit echtem LaMa/echten Bildern gelaufen**.
- Stufe 1 und die Oberflächen liefen bei dir echt; die Prüfung („Gate“) ist nicht
  kalibriert, es gibt keine automatische Entscheidung.
- Pipeline-Modus „pro Region“: Das gespeicherte Ergebnis ist eine Mischmaske aus
  Region-Gewinnern; wird sie später neu durch LaMa geschickt, kann die Füllung vom
  bewerteten Bild leicht abweichen (v. a. bei dicht liegenden Regionen).
  `selected_distance_px` hat bei Mehrregionen-Bildern das Format `r1=10;r2=SMART_INK`.
- Die Erklärung für das „Abschneiden“ (Fensterrand) ist plausibel, nicht bestätigt.
- Die alte Test-Suite (`modules/tests/` vor meinen Ergänzungen) lag mir nicht vor
  und wurde nicht ausgeführt.
- Vollständiger 716-Bilder-Lauf nach Stufe 5 (alte Pipeline): visuelle
  Abnahme und Report-Auswertung stehen weiterhin aus.

## 8. Offene Aufgaben (Reihenfolge)

1. Stufe 2 mit echten Daten laufen lassen, Ergebnisse im Ranker bewerten
   (Shift = unbrauchbar), `eval_ranking.py` auswerten: Brauchbar-Quote je Methode
   und Umfeld, greedy-Kaskade, Regionen ohne Alternative.
2. Stufe 3 entwerfen: YOLO-Masken (D0–D10, `SMART_INK` war bei einigen Bildern
   sehr gut) und neue Füllmethoden für strukturierte Flächen (analytische
   Fläche/Verlauf mit Rauschen, LaMa mit Kontext-Ausschnitt). `fill_repair`
   (25-px-Median glättet, überspringt Rahmen/Texturen) durch rauschangepasste
   Ersetzung ablösen.
3. Prüfung kalibrieren: nach Umfeld getrennt, bewusst streng, Kreuzvalidierung
   nach Bildern; Hauptzahl = Rate „fälschlich brauchbar“. Ziel ≥ 300 gelabelte
   Regionen aus Bildern, die nicht im Probelauf waren.
4. Orchestrator `yolo_sieve.py`: Stufen verketten, pro Region die erste bestandene
   Methode übernehmen, Rest markieren; entscheiden, ob ein finaler LaMa-Lauf auf
   der Mischmaske nötig ist.
5. Rahmen-Labels (22 vorhanden) nutzen, um zu prüfen, ob Merkmale an der
   markierten Stelle anschlagen.
6. README und Tests der alten Pipeline an den neuen Stand anpassen;
   `ranker_out/` in `.gitignore` aufnehmen.

## 9. Arbeitsregeln

- Nutzerergebnisse, Labels (`ranker_out/`) und Läufe außerhalb des Repositories
  oder in ignorierten Ordnern halten.
- Keine Schwellen oder Gewichte „nach Gefühl“ ändern: nur mit Labels und nach
  Bildern gruppierter Validierung. Kennzahl, die zählt: fälschlich als brauchbar
  eingestufte Ergebnisse.
- Visuelle Qualität nur anhand von Original, Maske und Ergebnis aus demselben
  Laufverzeichnis beurteilen; Labels beziehen sich auf das Ergebnis nach
  `fill_repair`.
- Nach Codeänderungen Syntaxprüfung und alle Tests ausführen; Tests mit
  Attrappen belegen nicht das Verhalten eines echten GPU-/IOPaint-Laufs.
- Eine fertige Prüfung gilt erst, wenn ihre Fehlerrate an ungesehenen Bildern
  gemessen wurde.
