"""Load reproducible registration grids and named presets."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Registration configuration must be a JSON object")
    version = payload.get("suite2p_version")
    fixed = payload.get("fixed_registration")
    grid = payload.get("parameter_grid")
    presets = payload.get("presets")
    if not isinstance(version, str) or not version:
        raise ValueError("suite2p_version must be a non-empty string")
    if not isinstance(fixed, dict):
        raise TypeError("fixed_registration must be a JSON object")
    if not isinstance(grid, dict):
        raise TypeError("parameter_grid must be a JSON object")
    if not isinstance(presets, dict):
        raise TypeError("presets must be a JSON object")
    for name, values in grid.items():
        if not isinstance(name, str) or not name:
            raise ValueError("parameter_grid keys must be non-empty strings")
        if not isinstance(values, list) or not values:
            raise ValueError(f"parameter_grid[{name!r}] must be a non-empty list")
        canonical_values = [
            json.dumps(value, sort_keys=True, separators=(",", ":"))
            for value in values
        ]
        if len(canonical_values) != len(set(canonical_values)):
            raise ValueError(f"parameter_grid[{name!r}] contains duplicate values")
    for name, overrides in presets.items():
        if not isinstance(name, str) or not name:
            raise ValueError("preset names must be non-empty strings")
        if not isinstance(overrides, dict):
            raise TypeError(f"preset {name!r} must be a JSON object")
    return payload


def build_sweep_candidates(
    config: dict[str, Any], selected_parameters: list[str]
) -> list[dict[str, object]]:
    if not selected_parameters:
        raise ValueError("A sweep requires at least one selected parameter")
    if len(selected_parameters) != len(set(selected_parameters)):
        raise ValueError("Sweep parameter names must not be repeated")

    grid: dict[str, list[Any]] = config["parameter_grid"]
    unknown = [name for name in selected_parameters if name not in grid]
    if unknown:
        raise ValueError(
            f"Selected parameters are absent from parameter_grid: {unknown}; "
            f"available: {list(grid)}"
        )

    # The JSON grid order, rather than CLI order, defines stable task ordering.
    ordered_names = [name for name in grid if name in selected_parameters]
    fixed = dict(config["fixed_registration"])
    candidates: list[dict[str, object]] = []
    for values in itertools.product(*(grid[name] for name in ordered_names)):
        overrides = dict(zip(ordered_names, values))
        registration = {**fixed, **overrides}
        name = "__".join(
            f"{parameter}={json.dumps(value, separators=(',', ':'))}"
            for parameter, value in overrides.items()
        )
        candidates.append({"name": name, "registration": registration})
    return candidates


def build_preset_candidate(
    config: dict[str, Any], preset: str
) -> dict[str, object]:
    presets: dict[str, dict[str, Any]] = config["presets"]
    if preset not in presets:
        raise ValueError(
            f"Unknown preset {preset!r}; available presets: {list(presets)}"
        )
    registration = {
        **dict(config["fixed_registration"]),
        **dict(presets[preset]),
    }
    return {"name": preset, "registration": registration}
