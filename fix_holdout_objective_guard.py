#!/usr/bin/env python3
"""Patch only the diagnostic holdout audit so alternatives must beat current objective.

Creates a timestamped backup. Does not edit scoring.py or run inference.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

project = Path.home() / "Downloads" / "Cleanup"
target = project / "audit_anchor_color_rule_holdout.py"
if not target.is_file():
    print(f"FEHLER: Audit-Script nicht gefunden: {target}")
    sys.exit(2)

source = target.read_text(encoding="utf-8-sig")
old_anchor = '''        current_p90 = float(cur["chroma_residual_p90"])
        eligible_alternatives = []
'''
new_anchor = '''        current_p90 = float(cur["chroma_residual_p90"])
        # Score the current winner using the same objective as alternatives:
        # its residual ratio is current_count/current_count == 1 when count > 0.
        current_objective = float(cur["base_score"]) + (COLOR_WEIGHT if current_count > 0 else 0.0)
        eligible_alternatives = []
'''
old_choice = '''        if eligible_alternatives:
            best = min(eligible_alternatives, key=lambda r: (float(r["rule_objective"]), float(r["base_score"]), str(r["candidate"])))
            recommended = str(best["candidate"])
            triggered = recommended != current
            reason = "alternative_passed_color_guards"
        else:
            best, recommended, triggered = None, current, False
            reason = "no_alternative_passed_color_guards"
'''
new_choice = '''        if eligible_alternatives:
            best_alternative = min(
                eligible_alternatives,
                key=lambda r: (float(r["rule_objective"]), float(r["base_score"]), str(r["candidate"])),
            )
            # Do not switch merely because an alternative passes the color guards:
            # it must also improve the combined objective over the current winner.
            if float(best_alternative["rule_objective"]) + 1e-9 < current_objective:
                best = best_alternative
                recommended = str(best["candidate"])
                triggered = recommended != current
                reason = "alternative_improved_current_objective"
            else:
                best, recommended, triggered = None, current, False
                reason = "best_alternative_not_better_than_current_objective"
        else:
            best, recommended, triggered = None, current, False
            reason = "no_alternative_passed_color_guards"
'''

checks = [
    (old_anchor, new_anchor, "current objective insertion"),
    (old_choice, new_choice, "candidate selection comparison"),
    ('            "current_base_score": round(float(cur["base_score"]), 8),\n',
     '            "current_base_score": round(float(cur["base_score"]), 8),\n            "current_rule_objective": round(current_objective, 8),\n',
     "objective output column"),
    ('            "recommended_rule_objective": round(float(best["rule_objective"]), 8) if best else "",\n',
     '            "recommended_rule_objective": round(float(best["rule_objective"]), 8) if best else round(current_objective, 8),\n',
     "recommended objective output"),
]
for old, new, label in checks:
    count = source.count(old)
    if count != 1:
        print(f"FEHLER: Erwartete genau eine Stelle für {label}, gefunden: {count}.")
        print("Es wurde nichts geändert. Bitte Audit-Script-Version prüfen.")
        sys.exit(3)

patched = source
for old, new, _ in checks:
    patched = patched.replace(old, new, 1)

backup = target.with_name(target.name + ".bak_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
backup.write_text(source, encoding="utf-8")
target.write_text(patched, encoding="utf-8")

print("========== PATCH ERFOLGREICH ==========")
print(f"Geändert: {target}")
print(f"Backup:   {backup}")
print("Geändert wurde nur die Entscheidungslogik des Holdout-Audits.")
print("scoring.py, Produktionsauswahl und Kandidatenbilder wurden nicht verändert.")
print("Nächster Schritt: das Audit mit dem vorhandenen Cleanup-.venv erneut ausführen.")
