#!/usr/bin/env python3
"""optimize_cleanup_stage2.py - staged, abortable optimisation of the Cleanup project.

STAGE 2 (run AFTER stage 1): bit-identical cropped fill_repair (median filter only over the
mask area) and per-batch IOPaint timing in run_stats.json.

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
    'yolo_mask_auto_select_modules/fill_repair.py':
        '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',
    'yolo_mask_auto_select_modules/tests/test_fill_repair_crop.py':
        'eNqVVk1v4zYQvftXEFoUkLGKYidxNtAivSy2vfVQ5GYEBC2NYmIpUktSG6lF/3tnRNmRbDnt6pBYw5k3bz44oyiKnvbAcmvqGgpW'
        'SqW4hVpIy6rGebYDtpP+ShagvcyFYt4wjwZlo9SVrMQL2pqqbrzw0ug0iqJFaU3FOC8b31jgnMmqNtYzobUJWm6xGGSNlt6D80eB'
        'bqq6Y8IxXS8CUGeU4ZVw37hovOEOFOSeV6ZoFLh0zHeAGF55biwkbHgplfC8VxbWy1LkHkksciWcY7+h+M9e7Qtm4ev3Rv4QCnQO'
        'T8jMxQeOKb1+EQ6W2YLhU0DJSM6t0IWpuMtBg0OqPt9zSg/v04N0XKM8hxadqi7GAMoBgR6rX9gjRpsGlBRRBamjPF7f3N5tlm+q'
        'PUcoeI4kHFqtjkelsYykTGqGQC8Q369GTujZJ+wVbaT2MWKn+B9ewDp0skrYDaovk/cOJ1g7ctWzpjjjmLATdosIE+t7Mr5dhRNU'
        'bvDkYYoky8D7F3bLHjGijP338wHT3lJvFeyalVZU1ILoU/vFqSoR3WYJy7L1p2dkPKFH7DYbYjexqzCaaj5VqHwpUXj06TRNH1Av'
        'V00BjPrX4dVp8r3EgsMPsB0Lt2dnbAF2YteNnOd7I3OIt0h2z656cqdBBPny+cR7O4/yStqv5yhBfoZCzEOp/wJrXKg1ZmBnjFp+'
        '7o+3Xdaxjz21Nmvp1yvl+sk2MIGi64c1e+yrkuam7uKpL+pi/tbCZ0lGlnfL5TKb5rgQ9hsrjHdo6XBO9eOJeJ13Q4e827nK4kEI'
        'AZNwP1vgFoPrQwsaZ9ghuO2uy3aEtCFP2Y5MNnONd7e6dA/ufuIeKKn70e0cddUh7rmWGlEcAr2+ZjfY6KIlOi1GtcaJMYRIZ8R6'
        'vZqA1NZ4lP5hNFwgvwnkz1wPlpe6iI6HLno4NNHDfA/hvEHW4EU/YlFlPO5j6qxkCDTpk5H04NNk07IbQOgnlefilvifkDTVU1wm'
        'YD1xjjFSYa3oOHxvhIoDbXKGQZdRn62/6e8/0UWcr73lMdQR4fchTtbEx9DuR5xtNNqZHMeT89iT/GAVPbNf2ag5R4R+tyBQN546'
        'SKhxZoc0tiPtRyQklOoYtGBzSZQCAC7gyRqFqvZd2PQWcLxrx6XGDwve6HxPA6E43Z3leAdtsI03w65Zr1ZzC8c0PmTx/YqXmN9k'
        '1KsB+dCts6mZq3nvrZzXf6vttBy1bEH1xVAip2IkbJViNfD7qMRPKo27Dj+o8I5FHDMlNedRSMfxI4WkOFb/BUOI4OY=',
}
MANIFEST_DATA = {
  "expected_original_sha256": {
    "yolo_mask_auto_select_modules/pipeline.py": "7ca133a6f93480dc9e1311feeb30d6ab7b1573837837dda534f1606335f45e6f",
    "yolo_mask_auto_select_modules/fill_repair.py": "116de52c793c5da90be46a4b4d7f347b84662b924be8df8a4fba7b703ba14e34"
  },
  "new_files": [
    "yolo_mask_auto_select_modules/tests/test_fill_repair_crop.py"
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
