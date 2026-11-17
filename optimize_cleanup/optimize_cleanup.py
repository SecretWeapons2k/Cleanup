#!/usr/bin/env python3
"""optimize_cleanup.py - staged, abortable optimisation of the Cleanup project.

Stage 1 (this script) applies only changes that are verified to leave every score,
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
    'yolo_mask_auto_select_modules/candidate_masks.py':
        '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',
    'yolo_mask_auto_select_modules/config.py':
        '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',
    'yolo_mask_auto_select_modules/cli.py':
        'eNqdV9tO40gQfc9XlPyUSI4nMBchUFbKQJZBGkhEwq52X6zGLtu9Y3d7u9tA5uu3um0nXGyjWT8AdndVnTp1xfO8c1kUTMTTnAsE'
        'ptKqQGGgZEpzkQae540SJQsIw6QylcIwBF6UUhlgQkjDDJdCj0btN5VaSaxlSmaynN+3Amt6HdUnQSRFwtP2ZDwCei6Wvy/uvm/D'
        'xc35t9XtJjzf/OG/OLi42mzpcLl5+fnqenG53NDp7cvvq7vt+m779vv6drVdnm+XF+Hfq5tlh50/l1eX37ZkZTIaxZg4MjAk3/S4'
        'VPIfjMypc2YC09/2Lgc3rEBdsghPnbIS5oezRcPr2r6pcYw6Ury03M29RWVkQTxGLM93EMmCRBD+Wn1fQcH0D4i5NkxEqIGljAtt'
        '4Kt8mkpBlysbI/jOrlngTWqrAYvjsA3j2JtOecFS1NOYK88HsytxbqH7QI6xKjfzPhZfPRnm5dxbKZ5ywXLQslIRglMOicxjVL0I'
        'CHwmlR42/ybonfatLBgJjUqg25DwHHttU7wMxQvj6U8p8B0MvYnRicXafuSEZ28DnA0wGTMgJERUVjxmBimOFNiMiRTPwOvS6B6v'
        'wXN60Bg6jUGkH0Dgk7Gumwyfu+8DT4AbwCdKE93Lg5AdVLCozkBtJBW2URV6A/5epYKuvfEWH1BYDMwFg2vKhkrEvUAekaeZeScQ'
        '+wJ8NxFcmWhMrXrXi6A1MFgPZL7RoyRVlgZXTiYj+O7cBwzSAIrZ7FM4O/oclCI9g314qFCbENS3+43lvOCm9ZULc3B11iJYuTZA'
        'JRWxkmCQM3lNb60axjPbSPJ80mtEUZ7JYqpZUebYY6yfyY2zV/tOg2AHtbrG/Bko/LfiioBQRjfsATne+tYHquBiWjTNpxPTxwFM'
        '11zwoipAVMU9KpAJ1M7tSXnMpEZX+YK6LmiDBf1gyui6JK8/FDDex+tjP3caMe6Edzw7/nI0OzoZZI4AJcSFwhKZYfc5tuQ5uNwl'
        'jdpNVSUoX2K6R2G2L7rlHO1NfC+L9hPAe1sqr2Zid6eyI55cpcHCbGaV/Anzw1x5ltkz/9j/5H/xT/yj2UAFpfpnJ2lHxyezfidy'
        'V6DdbPdJuT0hRoLZyiW5ZM9TO5h9HnD9VlbGzkiTUQpnNKcC2NpMr+cvRJVSZIlqX1elXUU0WIWuHfSP1BgfuANELYD+0POxx2iI'
        '0wcvqmLmfpeVNzmgbM77cbo+xkWCyvoKjYXetsIonh0o+qy7772VKmPMn2WWZ7XvG+TVak1LhwF3C2y19SoiqsvKDHf21ztZJxkX'
        '1G8imkk7V12PXAgbQrsMad8Wm4uTLagY76vUtYHByeeuHXrRr409mhBx5SQcB6egf/ASohyZoFI6zPj1zSWho3w1pB90htiAlA+o'
        'crbTQ5N/vNcT6ohQ1SPfSlvvqQU658EuhtpwGj+PihuDYjLkNvGST4ku5la/X/P6gmvXz6yOsNYRlHU83KSwXWy8+PC12Ve5lsNY'
        'Yoyr8v9w76az7QBuQX6xU1lG2qZGFTym6oZ7zNgDp+W0H03TkbvB1EIK6V8dQbLPNv/Jf9gUGEw=',
    'yolo_mask_auto_select_modules/pipeline.py':
        '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'
        'GmyvPMWx1vC62wwVg0Q3bU0ujy6P8KqtyTEdMiLZqoz3oNcQ0YBtgw8GvgebHKktcAkB6EFXvid4C6xqGHrw4rlTQHfyw97eV+7B'
        'h8vTAt8OQyavFMlPIneday6G+LOdbe6QJUrEu6DowoBuELuO47CdLt3JFnTkh/TKmICcPWfP2p2ShHtgl1zaZTZmtWuhK0wC5Sxu'
        '5O7Urv1Ua612Ud6xFteiUcv4yfVAt7TdorwaFRNNQo8ohmKef++Jq61fiEs8ZH6algk6tlHub4Hxe8oS7+91oPmyCQ1oR7xCDh4D'
        'fwWxU46xA9GfmxCBS3Id4B12grDIV6j7MG5aD6WXBbLjyEsaiftCawNFstLnY1bEp287a63n6kZPgUNu16zj7doo2yyRPAQm0HSu'
        'y78aMDddxhCdB5xW4SqTtyPa96A6UOevZlKbfdQsbKOihLx3PnYEpwmgwA9qiUO2n5nMKvqPJ1RIYDdZORVsEcySeEhLtRHe3JUv'
        'oo76ry6Aaow4CzPt+omty6weOXtzCOi+68WTdx3EFTyoF+sqL0bOl/POaxB2XxAVTDJweFj2EA/FRjCxIjQYtLn02rHdfm++7Mm3'
        'VyPba4wKR6ynbn73kW6vRzWnoANLfiNTPbio+ZjWNngE30YYGxYpNhF02Fo1St79A6QuyJ5qMGpZ7lm91rOpxLUNKbnstDZQqZJj'
        'F+LQTWWRi+wlqYQnT1kKXOU0Zt5+aY24lf56rS0YlKsdYQlptZtOfja10Fp5m5EQWwnAG45Skd9ub6et7hbNtHmKo24v77O5naR4'
        'dqBP9wlHO1n6GiqxIEciVPgzI9Z3581amKta759tg/+ZMuI0Lp+FI1K2xJ6TP6YnLU18hs507vRECawLWQeIpGfX9rRtozs0mUPt'
        '1mp9f2l/m1WY/haLq0xKpzXrzdQq7o4bpSSLu/jXDtHfmpjbONfTFlvcljcTKUMKXTCH/U8mz77LHPkI34zXizZLCb6SLQSfKnXq'
        'cqyb3npLUbXBMKLjJfOrYtlM5ljFVpkizWp1njJjjG7Twovsi85t8NYikZ9n65rfUfochLYKdfFjCg7V7Jb9yn2+kMfUvLsMuZAG'
        'uCOLmW8SL70diluh64TN4jC/ARG3JdG62R1/Sk34h/oK/h19dKDs8ansKN1rEj84BGJZLfdWanf6ieMGxoOBH4wFXQ8O3NhYZfPh'
        'N8M0uIHSiD+gQR8b4p7fecvVVrjZkT4/kN47h0D4T1RgzvV7bcfYO6sD3qE/t9zzWze86hXpfkL5cYgKP0rXKCbCfUzEvBGeL+WQ'
        'rPxzOcSrplOLjRfWU9TeeOdN52GpgumfttDe8elVs3djWE5Eu3rQWFrtYJWvJjH2SjJ1qjPUrnwAE2rpLv3UatNQjXPrM16oJBBe'
        'Tt6fnZ7//q/zyQm7OJscvXkHU/3waHIBw0u3QMHfi9MfTs8v2fFkArX+9MdTJb5h7SSLLOHlA982C24iHASONwimLcJRhtfOCn8M'
        'eH4EqRl40LFfQ1we9IACRB/Jpb2Zzds0xHcOqrdqax83GOwi/fWb2l2Jqrgx1tawW/VeuD6dea6HzurqolHpFqe6CBvFhU9QV7v9'
        'qV6TTlIDB1JcHXXxs8FiFwDA7bPnPZcwdOJD1uCuCepqI7r6lR4NCEVlQ5wrMHUEDXPXkOMP1dd3I5mzvrfZl3Ltzmb3ZJN1WelY'
        'wSs5azzCS8vRGaqc62qAE8NbhqCyxrrFbiINhHKJpQ+OQAeqXxxRRWoTVoUl5VW5imiWPwJtWMKgkethYonjrxbL1MwlEg/t4mUj'
        '432rZqgrtzPIbyycrySrR+CmvWBNSK/yEZtaUGWO7hzdtIYCRpeDyXrP2gT12bQxRAFMZnwbcXZIn0ZpxlQSUUBkNffj7xFoSobL'
        'UUOxHHV0esJw0et4cjlBBdwINjd+ouUvcVVX8fkRety0gBzyWUBf5RXaYoTJ6dxcNnyDrNE3GLXezUbf6GBvipuuf4jxFrbSt1FA'
        'Np5V9zo822zRRRJFX37SpeIMWRtAW77RTV6BVkGcfzeDcpTi41B/5U/XsZf4R6iyk9Uyq33N4+9R/g5vLpqBA8zloTl/m2/Rvvpy'
        '7//ia1T/C70fcGU=',
    'yolo_mask_auto_select_modules/tests/test_dedup_and_fill_repair.py':
        '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',
}
MANIFEST_DATA = {
  "expected_original_sha256": {
    "yolo_mask_auto_select_modules/candidate_masks.py": "2cf0c96a06d79978f2bf49986bdb089caa1dfb0966a318ad7d4cf73cfdcfe059",
    "yolo_mask_auto_select_modules/config.py": "93cd3b51ff714a892ffdcfe91d12890a12193592088c49fc5dfeac333b4656f3",
    "yolo_mask_auto_select_modules/cli.py": "6309f1c234c76c25c17f3a135642b631832259ea140c8bd2ad00816aba50e3f0",
    "yolo_mask_auto_select_modules/pipeline.py": "c1db15d5ff7abfc0d496d955fa6af19a35ea3b2bf48d2b7c3b2e2e1d03e74808"
  },
  "new_files": [
    "yolo_mask_auto_select_modules/tests/test_dedup_and_fill_repair.py"
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
