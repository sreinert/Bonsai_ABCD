#!/usr/bin/env python3
"""Move registration metadata and previews into the unified workflow layout."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def move_if_present(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    os.replace(source, destination)


def organize(run_dir: Path, workflow_dir: Path, qc_dir: Path) -> None:
    plane_dir = run_dir / "suite2p" / "plane0"
    workflow_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    for source_name, destination_name in (
        ("status.json", "registration_status.json"),
        ("provenance.json", "registration_provenance.json"),
        ("resolved_suite2p_configuration.json", "resolved_suite2p_configuration.json"),
    ):
        move_if_present(run_dir / source_name, workflow_dir / destination_name)
    for path in sorted(run_dir.glob("registered_frames*.png")):
        move_if_present(path, qc_dir / path.name)
    if plane_dir.is_dir():
        for pattern in ("meanImg*.png", "meanImg*.tiff"):
            for path in sorted(plane_dir.glob(pattern)):
                move_if_present(path, qc_dir / path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workflow-dir", type=Path, required=True)
    parser.add_argument("--qc-dir", type=Path, required=True)
    args = parser.parse_args()
    organize(args.run_dir.resolve(), args.workflow_dir.resolve(), args.qc_dir.resolve())


if __name__ == "__main__":
    main()
