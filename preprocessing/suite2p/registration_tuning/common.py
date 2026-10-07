"""Shared helpers for the registration-tuning workflow."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], *, overwrite: bool = False) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {path}. Pass --overwrite only when intentional."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def slug(value: str) -> str:
    result = "".join(character if character.isalnum() or character in "-_" else "-" for character in value)
    return result.strip("-") or "unnamed"


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def paths_overlap(first: Path, second: Path) -> bool:
    first = first.resolve()
    second = second.resolve()
    return is_relative_to(first, second) or is_relative_to(second, first)


def require_columns(rows: Iterable[dict[str, str]], required: set[str], label: str) -> list[dict[str, str]]:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"{label} is empty")
    missing = required - set(materialized[0])
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")
    return materialized

