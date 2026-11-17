#!/usr/bin/env python3
"""optimize_cleanup_stage3.py - staged, abortable optimisation of the Cleanup project.

STAGE 3 (run AFTER stages 1 and 2): routing audit skips overlays/contact sheets when called by the
pipeline (--no-visuals; masks and metrics unchanged) and LAMA_BATCH_SIZE 25 -> 100.

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
    'yolo_mask_routing_audit.py':
        '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'
        'uAyfL0jfbjG38WIT44vDsDqvf/C/pQB3mw==',
    'yolo_mask_auto_select_modules/tests/test_audit_no_visuals.py':
        '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',
}
MANIFEST_DATA = {
  "expected_original_sha256": {
    "yolo_mask_auto_select_modules/pipeline.py": "39a398712ab0a85c5607547ccfdf922b1026de8885663620c3ea8e9365860cba",
    "yolo_mask_auto_select_modules/config.py": "ee8ee075f77590d5a024b8a7c75db0deb415b3fb51251865f79e8ea8fac99ead",
    "yolo_mask_routing_audit.py": "a32ec36749d208fea9fc256d6ddb74ee3254d9ff46ce877f4ad378da4bb2edba"
  },
  "new_files": [
    "yolo_mask_auto_select_modules/tests/test_audit_no_visuals.py"
  ]
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
