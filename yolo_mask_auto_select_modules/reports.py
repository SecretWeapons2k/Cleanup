"""CSV report helpers with forward-compatible columns."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def write_csv_with_union_fields(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    preferred_fields: Sequence[str] | None = None,
) -> list[str]:
    """Write rows using preferred column order plus every observed field.

    This deliberately avoids DictWriter's `fields not in fieldnames` failure when
    newly added metrics appear in candidate dictionaries. No fields are discarded.
    """
    columns: list[str] = []
    seen: set[str] = set()
    for key in [*(preferred_fields or ()), *(key for row in rows for key in row.keys())]:
        key = str(key)
        if key not in seen:
            seen.add(key)
            columns.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    return columns
