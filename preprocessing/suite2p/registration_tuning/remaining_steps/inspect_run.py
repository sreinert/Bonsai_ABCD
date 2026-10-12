#!/usr/bin/env python3
"""Display registration failure details and the exact resolved Suite2p settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_run(run_dir: Path) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    run_dir = run_dir.resolve()
    status = read_json(run_dir / "status.json")
    configuration = read_json(run_dir / "resolved_suite2p_configuration.json")
    if configuration is None:
        provenance = read_json(run_dir / "provenance.json")
        if provenance and "settings" in provenance and "db" in provenance:
            configuration = {
                "suite2p_version": provenance.get("suite2p_version"),
                "db": provenance["db"],
                "settings": provenance["settings"],
            }
    if configuration is None:
        raise FileNotFoundError(
            "No resolved Suite2p configuration was written. Registration likely "
            "failed before Suite2p settings were constructed. Inspect status.json "
            f"and the Slurm stderr log for {run_dir}."
        )
    return status, configuration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--all-settings",
        action="store_true",
        help="print the complete Suite2p settings instead of registration only",
    )
    args = parser.parse_args()

    status, configuration = load_run(args.run_dir)
    if status:
        print("Run state:", status.get("state", "unknown"))
        if status.get("error"):
            print("Error:", status["error"])
    else:
        print("Run state: status.json not found")

    settings = configuration["settings"]
    print("Suite2p version:", configuration.get("suite2p_version", "unknown"))
    print("fs:", settings.get("fs"))
    print("tau:", settings.get("tau"))
    print("torch_device:", settings.get("torch_device"))
    print("\nDatabase passed to Suite2p:")
    print(json.dumps(configuration["db"], indent=2, sort_keys=True))
    label = "All resolved Suite2p settings" if args.all_settings else "Resolved registration settings"
    payload = settings if args.all_settings else settings["registration"]
    print(f"\n{label}:")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
