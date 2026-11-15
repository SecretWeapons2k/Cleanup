#!/usr/bin/env python3
"""Conservative repository cleanup for the Cleanup YOLO/LaMa project.

Default: report only, make no changes.
Apply:   python scripts/cleanup_repo.py --apply

The script removes regenerable Python bytecode/cache files. Old source
backups, the known accidental patch dump, and root-level historical logs are
moved to a timestamped archive next to the repository before disappearing
from the working tree.

Diagnostics, tests, source code, requirements snapshots, and all
yolo_mask_winners data are deliberately preserved. A static import audit
may report possible orphan modules, but it never deletes code.
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import os
import re
import shutil
from pathlib import Path


BACKUP_NAME_RE = re.compile(
    r"(?:\.bak(?:[._-].*)?|\.backup(?:[._-].*)?|\.old|\.orig)$",
    re.IGNORECASE,
)
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "node_modules",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
}
PROTECTED_ROOT_DIRS = {
    "yolo_mask_winners",  # selected masks, candidate images and reports
}


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def walk_repository(root: Path):
    """Yield paths without following symlinks or entering protected trees."""
    for current, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        kept_dirs: list[str] = []
        for dirname in sorted(dirnames):
            candidate = current_path / dirname
            if dirname in SKIP_DIRS or candidate.is_symlink():
                continue
            if current_path == root and dirname in PROTECTED_ROOT_DIRS:
                continue
            kept_dirs.append(dirname)
        dirnames[:] = kept_dirs
        yield current_path, list(sorted(filenames)), list(sorted(dirnames))


def is_known_patch_dump(path: Path) -> bool:
    """Recognise the one accidental Git diff file in the inspected repo."""
    expected_name = "tore -- " + chr(92)
    if path.name != expected_name or not path.is_file() or path.is_symlink():
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="replace")[:4096]
    except OSError:
        return False
    # The committed file contains ANSI color escapes around a unified Git diff.
    return "diff --git a/" in text and "+++ b/" in text


def collect_cleanup_targets(root: Path) -> dict[str, list[Path]]:
    targets: dict[str, list[Path]] = {
        "cache_dirs": [],
        "bytecode_files": [],
        "backup_files": [],
        "patch_dumps": [],
        "log_files": [],
    }

    for current, filenames, dirnames in walk_repository(root):
        for dirname in dirnames:
            if dirname == "__pycache__":
                cache_dir = current / dirname
                if cache_dir.is_dir() and not cache_dir.is_symlink():
                    targets["cache_dirs"].append(cache_dir)

        for filename in filenames:
            path = current / filename
            if path.is_symlink() or not path.is_file():
                continue
            lower_name = filename.lower()

            if path.suffix.lower() in {".pyc", ".pyo"}:
                # Files inside a cache directory are removed with that directory.
                if "__pycache__" not in path.parts:
                    targets["bytecode_files"].append(path)
                continue

            if BACKUP_NAME_RE.search(lower_name):
                targets["backup_files"].append(path)
                continue

            # Only root-level logs are archived. Logs nested inside diagnostics
            # remain with their reports and evidence.
            if path.suffix.lower() == ".log" and path.parent == root:
                targets["log_files"].append(path)

            if is_known_patch_dump(path):
                targets["patch_dumps"].append(path)

    # The specific accidental patch dump has no normal extension and may have
    # already been matched by a generic category only if its name changes.
    patch_path = root / ("tore -- " + chr(92))
    if is_known_patch_dump(patch_path) and patch_path not in targets["patch_dumps"]:
        targets["patch_dumps"].append(patch_path)

    for key in targets:
        targets[key] = sorted(set(targets[key]))
    return targets


def python_module_name(path: Path, root: Path) -> str | None:
    relative = path.relative_to(root)
    if relative.suffix != ".py":
        return None
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    if not parts:
        return None
    return ".".join(parts)


def has_main_guard(tree: ast.AST) -> bool:
    for node in getattr(tree, "body", []):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
            and any(
                isinstance(op, ast.Eq)
                and isinstance(comp, ast.Constant)
                and comp.value == "__main__"
                for op, comp in zip(test.ops, test.comparators)
            )
        ):
            return True
    return False


def resolve_from_import(current_module: str, path: Path, node: ast.ImportFrom) -> set[str]:
    """Resolve a Python ImportFrom node to possible in-repo module names."""
    if node.level == 0:
        base = node.module or ""
    else:
        parts = current_module.split(".")
        # An __init__.py represents its package; regular modules import relative
        # to their containing package.
        is_init = path.name == "__init__.py"
        package_parts = parts if is_init else parts[:-1]
        keep = max(0, len(package_parts) - node.level + 1)
        base_parts = package_parts[:keep]
        if node.module:
            base_parts.extend(node.module.split("."))
        base = ".".join(base_parts)

    resolved: set[str] = set()
    if base:
        resolved.add(base)
    for alias in node.names:
        if alias.name == "*":
            continue
        candidate = f"{base}.{alias.name}" if base else alias.name
        resolved.add(candidate)
    return resolved


def audit_unreferenced_python(root: Path) -> tuple[list[str], list[str]]:
    """Return (unreachable modules, parse errors); never changes or deletes code."""
    source_files: list[Path] = []
    for current, filenames, _ in walk_repository(root):
        for filename in filenames:
            if filename.endswith(".py"):
                source_files.append(current / filename)

    module_to_path: dict[str, Path] = {}
    for path in source_files:
        module = python_module_name(path, root)
        if module:
            module_to_path[module] = path

    graph: dict[str, set[str]] = {name: set() for name in module_to_path}
    entrypoints: set[str] = set()
    parse_errors: list[str] = []

    for module, path in module_to_path.items():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeError, SyntaxError) as exc:
            parse_errors.append(f"{path.relative_to(root)}: {exc}")
            continue

        if has_main_guard(tree) or "tests" in path.parts or path.name.startswith("test_"):
            entrypoints.add(module)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported = alias.name
                    if imported in module_to_path:
                        graph[module].add(imported)
                    else:
                        # Also account for imports that name a package prefix.
                        for candidate in module_to_path:
                            if candidate.startswith(imported + "."):
                                graph[module].add(candidate)
            elif isinstance(node, ast.ImportFrom):
                for imported in resolve_from_import(module, path, node):
                    if imported in module_to_path:
                        graph[module].add(imported)

    reachable: set[str] = set(entrypoints)
    pending = list(entrypoints)
    while pending:
        current = pending.pop()
        for child in graph.get(current, set()):
            if child not in reachable:
                reachable.add(child)
                pending.append(child)

    # Package __init__ files are implicit imports, not standalone dead modules.
    unreachable = [
        str(module_to_path[name].relative_to(root))
        for name in sorted(module_to_path)
        if name not in reachable and module_to_path[name].name != "__init__.py"
    ]
    return unreachable, parse_errors


def format_report(root: Path, targets: dict[str, list[Path]], orphans: list[str], parse_errors: list[str]) -> str:
    def rel(items: list[Path]) -> list[str]:
        return [str(p.relative_to(root)) for p in items]

    lines = [
        f"Repository: {root}",
        "Mode: DRY RUN (no files changed)",
        "",
        "SAFE CLEANUP CANDIDATES (only applied with --apply)",
        f"  __pycache__ directories: {len(targets['cache_dirs'])}",
        f"  standalone .pyc/.pyo files: {len(targets['bytecode_files'])}",
        f"  source backup files: {len(targets['backup_files'])}",
        f"  confirmed accidental diff dumps: {len(targets['patch_dumps'])}",
        "",
        "PRESERVED BY DESIGN",
        f"  root-level .log files to archive: {len(targets['log_files'])}",
        "  yolo_mask_winners/: all masks, images, CSV reports and run data",
        "  diagnostics/, tests/, current Python source and requirements snapshots",
        "",
        "POTENTIALLY UNREFERENCED PYTHON MODULES (review only; never auto-deleted)",
    ]
    lines.extend(f"  - {item}" for item in orphans) if orphans else lines.append("  none detected")

    if parse_errors:
        lines += ["", "FILES NOT INCLUDED IN IMPORT AUDIT (parse/read errors)"]
        lines.extend(f"  - {item}" for item in parse_errors)

    for title, key in (
        ("CACHE DIRECTORIES", "cache_dirs"),
        ("STANDALONE BYTECODE", "bytecode_files"),
        ("BACKUP FILES TO ARCHIVE", "backup_files"),
        ("ACCIDENTAL DIFF DUMP TO ARCHIVE", "patch_dumps"),
    ):
        items = rel(targets[key])
        if items:
            lines += ["", title]
            lines.extend(f"  - {item}" for item in items)

    if targets["log_files"]:
        lines += ["", "ROOT-LEVEL LOGS TO ARCHIVE"]
        lines.extend(f"  - {item}" for item in rel(targets["log_files"]))

    lines += [
        "",
        "No application logic is deleted. Unreferenced-module findings are hints only:",
        "scripts may be launched manually or from outside the repository.",
    ]
    return "\n".join(lines)


def apply_cleanup(root: Path, targets: dict[str, list[Path]]) -> Path | None:
    archived_files = (
        targets["backup_files"] + targets["patch_dumps"] + targets["log_files"]
    )
    archive_dir: Path | None = None

    if archived_files:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = root.parent / f".{root.name}_cleanup_archive_{stamp}"
        # Never overwrite an earlier archive.
        archive_dir.mkdir(parents=False, exist_ok=False)
        for source in archived_files:
            relative = source.relative_to(root)
            destination = archive_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))

    for cache_dir in sorted(targets["cache_dirs"], key=lambda p: len(p.parts), reverse=True):
        if cache_dir.exists() and cache_dir.is_dir() and not cache_dir.is_symlink():
            shutil.rmtree(cache_dir)

    for bytecode in targets["bytecode_files"]:
        if bytecode.is_file() and not bytecode.is_symlink():
            bytecode.unlink()

    return archive_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Conservative cleanup of generated Python files and obsolete code backups."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="apply safe cleanup; backup files are archived outside the repository first",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=default_repo_root(),
        help="repository root (defaults to the parent of this script's directory)",
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"repository root does not exist or is not a directory: {root}")

    targets = collect_cleanup_targets(root)
    orphans, parse_errors = audit_unreferenced_python(root)
    report = format_report(root, targets, orphans, parse_errors)

    if not args.apply:
        print(report)
        print("\nNothing changed. Run again with --apply to perform the safe cleanup.")
        return 0

    print(report.replace("Mode: DRY RUN (no files changed)", "Mode: APPLY"))
    archive_dir = apply_cleanup(root, targets)
    print("\nCLEANUP COMPLETE")
    print(f"  Removed cache directories: {len(targets['cache_dirs'])}")
    print(f"  Removed standalone bytecode files: {len(targets['bytecode_files'])}")
    print(f"  Archived backup/diff/log files: {len(targets['backup_files']) + len(targets['patch_dumps']) + len(targets['log_files'])}")
    if archive_dir:
        print(f"  Recovery archive: {archive_dir}")
    else:
        print("  Recovery archive: not needed")
    print("  Nested diagnostics logs, diagnostics and result images were not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
