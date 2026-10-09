#!/usr/bin/env python3
"""Create one isolated full-session task for the default Cellpose pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import read_csv, require_columns, write_csv
from make_full_session_tasks import REGISTRATION
from run_candidate import TASK_COLUMNS


def build_pilot_task(source_manifest: Path, task_id: int) -> dict[str, object]:
    rows = require_columns(read_csv(source_manifest), TASK_COLUMNS, "source manifest")
    if task_id < 0 or task_id >= len(rows):
        raise IndexError(f"Task {task_id} is outside 0..{len(rows) - 1}")
    source = rows[task_id]
    source_run_dir = Path(source["run_dir"])
    run_dir = source_run_dir / "suite2p_pilots" / "default-cpsam"
    return {
        **source,
        "task_id": 0,
        "candidate_name": "default-cpsam-pilot",
        "candidate_id": "default-cpsam-pilot",
        "registration_json": json.dumps(REGISTRATION, sort_keys=True),
        "run_dir": str(run_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-task-id", type=int, default=0)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    row = build_pilot_task(args.source_manifest, args.source_task_id)
    run_dir = Path(str(row["run_dir"]))
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Pilot output already exists and will not be overwritten: {run_dir}"
        )
    write_csv(args.manifest, [row], overwrite=args.overwrite)
    print(
        f"Pilot session: {row['mouse_id']} / {row['session_id']}\n"
        f"Pilot output:  {row['run_dir']}\n"
        f"Manifest:      {args.manifest}"
    )


if __name__ == "__main__":
    main()
