#!/usr/bin/env python3
"""Create a single browser gallery for the 26 approved numeric-distance review cases.
Read-only with respect to production files and candidate images.
"""
from __future__ import annotations

import csv
import html
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import audit_anchor_color_rule_holdout as audit
import audit_anchor_color_rule_full_population as full_audit

AUDIT_ROOT = audit.DEBUG_DIR / "counterfactual_color_rule_full_population_20261024_214800"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def norm_stem(name: str) -> str:
    return Path(name.strip()).stem.lower()


def main() -> None:
    if not AUDIT_ROOT.is_dir():
        raise SystemExit(f"Audit-Ordner fehlt: {AUDIT_ROOT}")

    csv_matches = sorted(AUDIT_ROOT.glob(
        "numeric_distance_winner_audit_*/numeric_distance_outside_anchor_followup_*/"
        "numeric_distance_qualifying_changes_for_outside_anchor.csv"
    ))
    if not csv_matches:
        raise SystemExit("FEHLER: numeric_distance_qualifying_changes_for_outside_anchor.csv nicht gefunden.")
    changes_csv = csv_matches[-1]
    changes = sorted(read_csv(changes_csv), key=lambda row: row.get("image", "").lower())
    if not changes:
        raise SystemExit(f"FEHLER: Keine Vorschläge in {changes_csv}")

    required = {"image", "current_candidate", "recommended_candidate"}
    if required - set(changes[0]):
        raise SystemExit("FEHLER: Die Vorschlags-CSV enthält nicht die erwarteten Spalten.")

    anchors_path = audit.find_anchor_csv()
    anchor_map: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(anchors_path):
        if row.get("image", "").strip():
            anchor_map[norm_stem(row["image"])].append(row)

    original_map: dict[str, list[Path]] = defaultdict(list)
    for path in audit.ORIGINAL_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in audit.IMAGE_EXTS:
            original_map[path.stem.lower()].append(path)

    # Preflight: do not create partial sheets if any required source is missing.
    missing: list[str] = []
    for row in changes:
        image_name = row["image"]
        stem = norm_stem(image_name)
        if not anchor_map.get(stem):
            missing.append(f"Anker fehlen: {image_name}")
        try:
            audit.find_original(image_name, original_map)
        except Exception as exc:
            missing.append(f"Original fehlt/mehrdeutig: {image_name} ({exc})")
        for col in ("current_candidate", "recommended_candidate"):
            candidate = row[col]
            if not audit.candidate_path(image_name, candidate).is_file():
                missing.append(f"Kandidat fehlt: {image_name} / {candidate}")
    if missing:
        raise SystemExit("FEHLER beim Vorabtest; keine Kontaktbögen erstellt:\n- " + "\n- ".join(missing))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = changes_csv.parent / f"visual_review_contact_sheets_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)

    count = full_audit.make_context_sheets(changes, original_map, anchor_map, out_dir)
    sheets = sorted(out_dir.glob("full_population_context_*.jpg"))
    if count == 0 or not sheets:
        raise SystemExit("FEHLER: Das Rendern der Kontaktbögen hat keine Bilder erzeugt.")

    cards = "\n".join(
        f'<section><h2>Kontaktbogen {i:02d}</h2>'
        f'<a href="{html.escape(path.name)}"><img loading="lazy" src="{html.escape(path.name)}" '
        f'alt="Kontaktbogen {i:02d}"></a></section>'
        for i, path in enumerate(sheets, start=1)
    )
    image_names = " · ".join(html.escape(row["image"]) for row in changes)
    index = out_dir / "index.html"
    index.write_text(
        "<!doctype html><html lang='de'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Cleanup – visuelle Kandidatenprüfung</title><style>"
        "body{font:16px/1.45 system-ui,sans-serif;max-width:1500px;margin:24px auto;padding:0 16px;color:#222}"
        "h1{font-size:1.5rem}section{margin:28px 0;padding-top:10px;border-top:1px solid #bbb}"
        "img{display:block;width:100%;height:auto;border:1px solid #ccc}"
        "p{overflow-wrap:anywhere}</style></head><body>"
        "<h1>Visuelle Prüfung der vorgeschlagenen Kandidatenwechsel</h1>"
        f"<p><b>Fälle:</b> {len(changes)} · <b>Kontaktbögen:</b> {len(sheets)}</p>"
        "<p>Je Fall: ORIGINAL (rote Rechtecke = OCR-Anker), CURRENT (bisherige Auswahl), "
        "PROPOSED (vorgeschlagene Auswahl). Die Kontaktbögen sind nur für Review; "
        "Produktionsdateien und Kandidatenbilder werden nicht verändert.</p>"
        f"<p><b>Quell-CSV:</b> {html.escape(str(changes_csv))}</p>"
        f"<p><b>Enthaltene Bilder:</b> {image_names}</p>{cards}</body></html>\n",
        encoding="utf-8",
    )

    print("========== KONTAKTBOGEN-GALERIE FERTIG ==========")
    print(f"Geprüfte Vorschläge: {len(changes)}")
    print(f"Kontaktbögen erstellt: {len(sheets)}")
    print(f"Galerie: {index}")
    print("Nur neue Review-Dateien geschrieben; Produktionsdateien und Kandidaten unverändert.")
    try:
        subprocess.Popen(["xdg-open", str(index)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("Galerie wird im Standardbrowser geöffnet.")
    except OSError as exc:
        print(f"Galerie nicht automatisch geöffnet: {exc}")
        print(f"Manuell öffnen: {index}")


if __name__ == "__main__":
    main()
