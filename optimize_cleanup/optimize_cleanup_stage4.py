#!/usr/bin/env python3
"""optimize_cleanup_stage4.py - staged, abortable optimisation of the Cleanup project.

STAGE 4 (run AFTER stages 1-3): LaMa runs in N concurrent IOPaint processes per batch
(LAMA_PARALLEL_PROCESSES = 2, CLI --lama-processes; 1 = old sequential behaviour).

Stage 1 applied only changes that are verified to leave every score,
winner and mask unchanged:
  * pixel-identical candidate masks go through LaMa / compositing / scoring once
  * the mask set is built once per image (was twice)
  * FILL_REPAIR switch + fill-repair metrics in the reports
  * per-phase timers and LaMa job counters (run_stats.json)
  * --no-debug-images production switch

The restructuring into src/cleanup/ is deliberately NOT part of this stage: it should
follow after the baseline measurements, see the final report.

Usage (works from any directory, stdlib only):
    python optimize_cleanup.py                 # inventory + dry run, changes nothing
    python optimize_cleanup.py --apply         # backup, patch, validate
    python optimize_cleanup.py --restore BACKUP_DIR

Never touched: originals, test_runs/, diagnostics/, .venv, yolo_mask_winners/.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

EMBEDDED_PAYLOAD = {
    'yolo_mask_auto_select_modules/pipeline.py':
        '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',
    'yolo_mask_auto_select_modules/config.py':
        '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',
    'yolo_mask_auto_select_modules/cli.py':
        'eNqdV9tu4kgQfecrSn4CCTyQuShKxEpMwmYikYAC2dXui9XYhd07dre3u52E+fqpbtuQi+3sLA8JdndVnbqdKjzPu5BZxkQ0SrlA'
        'YCouMhQGcqY0F7HveV5vp2QGQbArTKEwCIBnuVQGmBDSMMOl0L1e/U7FVhJLmZyZJOXbWmBFj73yxA+l2PG4Pun3gD6X899n94tN'
        'MLu9+La8WwcX6z+GLw4ur9cbOpyvX76+vpldzdd0ele+X8xuZsFqdjdbLOaLYHW3JIn1a6Hl/WZ1vzkK1e/p+mZ+sZlfBn8vb+cN'
        'IP6cX19925C2Qa8X4c5FCgNyXPdzJf/B0Jw5Twcw+u0QD/+WZahzFuKZU5bD9Hg2q4K+sk+qH6EOFc9tYKferDAyoyCHLE33EMqM'
        'RBD+Wi6WkDH9HSKuDRMhamAx40Ib+CqfRlLQ5cImEBbshvneoLTqsygK6hz3vdGIZyxGPYq48oZg9jlOLfQhkGOsSM20LcSvPgmm'
        '+dRbKh5zwVLQslAhglMOO5lGqFoREPhEKt1t/k1FNNq3smAkVCqBbsOOp9hqm/JlKF8YjX5Ige9gaC2MRizW9iMnPAcb4GyASZgB'
        'ISGknuMRM0h5pMQmTMR4Dl6TRvfxKjxnR42B0+iH+gEEPhnruknwuftD4DvgBvCJykS3xkHIhlCwsKxAbSR1vVEFeh3+XseCrr3x'
        'Fh9QWAzMJYNrqoZCRK1AHpHHiXknEYcGfLcQXJtojK16R1RQG+jsBzJf6VGSOkuDayeTEHx3PgT0Yx+y8fhTMJ589nMRn8MhPdSo'
        'VQrK2+3GUp5xU/vKhTm6Oq4RLB0NUEuFLCcY5ExahrdUDf2xJZI0HbQaUVRnMhtpluUpthhrj+Ta2St9pymxh1JdZf4cFP5bcEVA'
        'qKKr6AE5XvvWBirjYpRV5NOI6WMHphsueFZkIIpsiwrkDkrnDkF5TKRG1/mCWBe0wYz+MGV02ZI3HzLoH/L1sT12GjFqhHcyPvky'
        'GU9OOyNHgHYUC4U5MsO2KdbBc3C5Kxq1H6lCUL1EdI/SbB90HXO0N/G9KjpMAO9tq7wamM1MZec/uUqDhdnKyvkTpse58qyyx8OT'
        '4afhl+HpcDLu6KBY/2gM2uTkdNzuROoatDnabVJuiYiQYNZyu1Sy56Xtjz93uH4nC2NnpEmohBOaUz5sbKWX8xfCQimyRL2vi9zu'
        'KRqsQkcH7SM1wgfuABEF0Bc97XuMhji98MIiYu5/XniDI8rqvB2n4zEudqisr1BZaKUVRvlsQNFm3b1v7VQZYfqssjyr/UCQ18sV'
        'LR0G3C2w3daNKi8JtaXpO/e2ltoVVZKghnIwATmRw5aZMIH+hFhSE1XRRc7SIVCqYYsJe+C0qbT3P5VHXpjuafR6j2zEeUkcGdIc'
        '3TtGeORC2LKzC5weWoJwtWVJIMJtETvq6pzW7tqRP39tVNNUiwon4fJ2Bvo7zyFMkQlq/+Nesrq9InTUY4b0g04QK5DyAVXK9rpr'
        'W+kf9AQ6JFTlmmKlrfeUGec82GVWG04j81FxY1AMutymuKQjChdz6+qveX3JteNgqyModfh5mQ833Szz9mcfvlY7NteyG0uEUZH/'
        'n9i7jcKyllvqX+yBNiI1ERPr9P9jmVZTpBlMKaSQfrsJkn32a2XwEziDWEI=',
    'yolo_mask_auto_select_modules/tests/test_dedup_and_fill_repair.py':
        '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'
        'ZwrR/wP9yZDy',
}
MANIFEST_DATA = {
  "expected_original_sha256": {
    "yolo_mask_auto_select_modules/pipeline.py": "51737f043bdbc6c5518284dee6308f68ab79f9c1b78ce066e64af7c5b986463c",
    "yolo_mask_auto_select_modules/config.py": "4888a8025cd8ccb98bd010b52c791c99a9636352c79698aaf244904677bd42d4",
    "yolo_mask_auto_select_modules/cli.py": "da4627b2b75c0f7554030c17c630327bd694a3636df2c9812771be52cd259508",
    "yolo_mask_auto_select_modules/tests/test_dedup_and_fill_repair.py": "cee28cc14b3bcef5ee24234c40b396df7d66cd0c0bb024bfeba109b111fc3af4"
  },
  "new_files": []
}


def payload_bytes(rel: str) -> bytes:
    import base64, zlib
    return zlib.decompress(base64.b64decode(EMBEDDED_PAYLOAD[rel]))

EXCLUDE_DIRS = {".venv", ".git", "__pycache__", "test_runs", "diagnostics", "yolo_mask_winners",
                "node_modules", ".pytest_cache"}


def fail(msg: str) -> None:
    print(f"\nABBRUCH: {msg}", file=sys.stderr)
    raise SystemExit(2)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def py_files(root: Path):
    for p in sorted(root.rglob("*.py")):
        if not (set(p.relative_to(root).parts) & EXCLUDE_DIRS):
            yield p


# ------------------------------------------------------------------ phase 1
def inventory(project: Path) -> dict:
    defs: dict[str, list[str]] = {}
    refs: set[str] = set()
    files = {}
    for p in py_files(project):
        rel = str(p.relative_to(project))
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            fail(f"Syntaxfehler in {rel}: {exc}")
        imports = sorted({
            (n.module or "") if isinstance(n, ast.ImportFrom) else a.name
            for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
            for a in (n.names if isinstance(n, ast.Import) else [None])
        })
        top = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        for n in top:
            defs.setdefault(n, []).append(rel)
        for n in ast.walk(tree):
            if isinstance(n, ast.Name):
                refs.add(n.id)
            elif isinstance(n, ast.Attribute):
                refs.add(n.attr)
            elif isinstance(n, ast.ImportFrom):
                refs.update(a.name for a in n.names)
        abs_paths = sorted({
            n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and (n.value.startswith(("/", "~/")) and len(n.value) > 3)
        })
        files[rel] = {"lines": len(p.read_text(encoding="utf-8").splitlines()),
                      "imports": imports, "top_level_defs": top, "absolute_path_literals": abs_paths}
    unreferenced = {n: where for n, where in defs.items()
                    if n not in refs and not n.startswith(("test_", "Test", "_test", "main"))}
    return {"files": files, "possibly_unreferenced_defs (heuristic, NOT a removal proof)": unreferenced,
            "test_files": [f for f in files if "/test" in f or f.startswith("test")],
            "config_files": [str(p.relative_to(project)) for p in project.glob("*.toml")]
                            + [str(p.relative_to(project)) for p in project.glob("requirements*.txt")]}


# ------------------------------------------------------------------ phase 2
def make_backup(project: Path, root: Path) -> Path:
    dest = root / f"{project.name}_backup_{time.strftime('%Y%m%d_%H%M%S')}"
    if dest.exists():
        fail(f"Backup existiert bereits, wird nie ueberschrieben: {dest}")
    try:
        dest.resolve().relative_to(project.resolve())
        fail("Backup-Verzeichnis darf nicht innerhalb des Projekts liegen.")
    except ValueError:
        pass
    root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(project, dest, ignore=shutil.ignore_patterns(*EXCLUDE_DIRS, "*.pyc"))
    return dest


# ------------------------------------------------------------------ phase 3
def check_patch_preconditions(project: Path) -> tuple[dict, list[str]]:
    m = MANIFEST_DATA
    problems = []
    for rel, expected in m["expected_original_sha256"].items():
        target = project / rel
        if not target.is_file():
            problems.append(f"fehlt: {rel}")
        elif sha256(target) != expected:
            if sha256(target) == hashlib.sha256(payload_bytes(rel)).hexdigest():
                continue  # already patched
            problems.append(f"Datei weicht vom analysierten Stand ab (nicht automatisch patchbar): {rel}")
    for rel in m["new_files"]:
        t = project / rel
        if t.exists() and sha256(t) != hashlib.sha256(payload_bytes(rel)).hexdigest():
            problems.append(f"Neue Datei wuerde bestehende, andere Datei ueberschreiben: {rel}")
    return m, problems


def apply_patch(project: Path, manifest: dict) -> list[str]:
    changed = []
    for rel in [*manifest["expected_original_sha256"], *manifest["new_files"]]:
        data, dst = payload_bytes(rel), project / rel
        if dst.is_file() and dst.read_bytes() == data:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        changed.append(rel)
    return changed


# ------------------------------------------------------------------ phase 5
def validate(project: Path, python: str) -> dict:
    res = {}
    comp = subprocess.run([python, "-m", "compileall", "-q", "yolo_mask_auto_select_modules",
                           "yolo_mask_auto_select.py", "yolo_mask_routing_audit.py"],
                          cwd=project, capture_output=True, text=True)
    res["syntax"] = {"ok": comp.returncode == 0, "output": (comp.stdout + comp.stderr)[-800:]}
    tests = subprocess.run([python, "-m", "unittest", "discover", "-s",
                            "yolo_mask_auto_select_modules/tests", "-t", ".", "-v"],
                           cwd=project, capture_output=True, text=True)
    tail = (tests.stdout + tests.stderr).strip().splitlines()[-6:]
    res["unit_tests"] = {"ok": tests.returncode == 0, "tail": tail}
    return res


def restore(backup: Path, project: Path) -> None:
    m = MANIFEST_DATA
    for rel in m["expected_original_sha256"]:
        shutil.copy2(backup / rel, project / rel)
        print(f"wiederhergestellt: {rel}")
    for rel in m["new_files"]:
        t = project / rel
        if t.exists() and not (backup / rel).exists():
            t.rename(t.with_suffix(t.suffix + f".removed_{time.strftime('%Y%m%d_%H%M%S')}"))
            print(f"zur Seite gelegt: {rel}")


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", type=Path, default=next((c for c in (HERE, HERE.parent) if (c / "yolo_mask_auto_select_modules").is_dir()),
                                 Path.home() / "Downloads" / "Cleanup"))
    ap.add_argument("--backup-root", type=Path, default=Path.home() / "cleanup_backups")
    ap.add_argument("--apply", action="store_true", help="Backup, patch and validate (default: dry run)")
    ap.add_argument("--restore", type=Path, metavar="BACKUP_DIR")
    ap.add_argument("--python", default=sys.executable, help="Interpreter of the project .venv for the tests")
    args = ap.parse_args()
    project = args.project.expanduser().resolve()
    if not (project / "yolo_mask_auto_select_modules" / "pipeline.py").is_file():
        fail(f"Kein Cleanup-Projekt unter {project} (--project angeben).")

    if args.restore:
        restore(args.restore.expanduser().resolve(), project)
        return 0

    stamp = time.strftime("%Y%m%d_%H%M%S")
    report_dir = project / "optimization_reports" / stamp
    report_dir.mkdir(parents=True, exist_ok=False)

    print("[Phase 1] Bestandsaufnahme ...")
    inv = inventory(project)
    (report_dir / "inventory.json").write_text(json.dumps(inv, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  {len(inv['files'])} Python-Dateien, {len(inv['test_files'])} Testdateien")

    print("[Phase 3] Voraussetzungen des Patches pruefen ...")
    manifest, problems = check_patch_preconditions(project)
    if problems:
        fail("Patch wird nicht angewendet:\n  - " + "\n  - ".join(problems))
    print("  alle Zieldateien entsprechen dem analysierten Stand.")

    if not args.apply:
        print(f"\nDRY RUN beendet. Bericht: {report_dir}\nMit --apply ausfuehren, um Backup + Patch durchzufuehren.")
        return 0

    print("[Phase 2] Backup ...")
    backup = make_backup(project, args.backup_root.expanduser().resolve())
    print(f"  {backup}")

    print("[Phase 3] Patch anwenden ...")
    changed = apply_patch(project, manifest)
    for rel in changed:
        print(f"  geaendert: {rel}")

    print("[Phase 5] Validierung ...")
    val = validate(project, args.python)
    for k, v in val.items():
        print(f"  {k}: {'OK' if v['ok'] else 'FEHLER'}")
    ok = all(v["ok"] for v in val.values())

    lines = [f"# Optimierungsbericht {stamp}", "", f"Backup: `{backup}`", "",
             "## Geaenderte Dateien", *[f"- `{c}`" for c in changed], "",
             "## Validierung", *[f"- {k}: {'OK' if v['ok'] else 'FEHLER'}" for k, v in val.items()], "",
             "```", *val["unit_tests"]["tail"], "```", "",
             "## Nicht Teil dieser Stufe",
             "- Umstrukturierung nach src/cleanup/, pyproject.toml/uv.lock (erst nach Baseline-Messung)",
             "- Laufzeit-/VRAM-Vergleich vorher/nachher: erfordert echten Lauf mit GPU "
             "(`run_stats.json` im Ausgabeordner liefert die Phasenzeiten)",
             "- Bewertung der Cleaning-Methoden/Scoring-Metriken: erfordert die vollstaendigen candidate_scores.csv",
             "",
             "Wiederherstellung: `python optimize_cleanup.py --restore " + str(backup) + "`"]
    (report_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nBericht: {report_dir / 'report.md'}")
    if not ok:
        print(f"FEHLER bei der Validierung. Wiederherstellen mit:\n  python {Path(__file__).name} --restore {backup}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
