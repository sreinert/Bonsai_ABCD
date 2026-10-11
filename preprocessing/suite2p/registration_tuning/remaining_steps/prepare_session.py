#!/usr/bin/env python3
"""List tested sessions or create one task for the selected session."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


WORKFLOW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_DIR))

from common import paths_overlap, read_csv, require_columns, slug, write_csv  # noqa: E402
from run_candidate import TASK_COLUMNS  # noqa: E402

try:  # Support both direct script execution and package imports in tests.
    from .config import FS, NCHANNELS, NPLANES, REGISTRATION, SUITE2P_VERSION, TAU
except ImportError:
    from config import FS, NCHANNELS, NPLANES, REGISTRATION, SUITE2P_VERSION, TAU


def unique_sessions(source_manifest: Path) -> list[dict[str, Any]]:
    """Return one representative row for each raw session in manifest order."""
    rows = require_columns(read_csv(source_manifest), TASK_COLUMNS, "source manifest")
    sessions: list[dict[str, Any]] = []
    positions: dict[tuple[str, str, str, str], int] = {}
    for source_row, row in enumerate(rows):
        key = (
            row["mouse_id"],
            row["session_id"],
            row["input_path"],
            row["tiff_files_json"],
        )
        if key not in positions:
            positions[key] = len(sessions)
            sessions.append(
                {
                    "source_task_id": source_row,
                    "row": row,
                    "candidates": [row["candidate_name"]],
                }
            )
        else:
            candidate = row["candidate_name"]
            candidates = sessions[positions[key]]["candidates"]
            if candidate not in candidates:
                candidates.append(candidate)
    return sessions


def print_sessions(source_manifest: Path) -> None:
    sessions = unique_sessions(source_manifest)
    print("index\tmouse\tsession\trole\tsource task\tregistration candidates")
    for index, item in enumerate(sessions):
        row = item["row"]
        print(
            f"{index}\t{row['mouse_id']}\t{row['session_id']}\t"
            f"{row.get('selection_role', '')}\t{item['source_task_id']}\t"
            f"{', '.join(item['candidates'])}"
        )


def build_task(
    source_manifest: Path, session_index: int, output_root: Path
) -> dict[str, object]:
    sessions = unique_sessions(source_manifest)
    if session_index < 0 or session_index >= len(sessions):
        raise IndexError(
            f"Session index {session_index} is outside 0..{len(sessions) - 1}"
        )
    selected = sessions[session_index]
    source = selected["row"]
    input_path = Path(source["input_path"]).resolve()
    run_dir = (
        output_root.resolve()
        / slug(source["mouse_id"])
        / slug(source["session_id"])
        / "chosen-registration-cpsam"
    )
    if paths_overlap(run_dir, input_path):
        raise ValueError(f"Output {run_dir} overlaps raw input {input_path}")
    return {
        **source,
        "task_id": 0,
        "candidate_name": "chosen-registration-cpsam",
        "candidate_id": "chosen-registration-cpsam",
        "suite2p_version": SUITE2P_VERSION,
        "fs": FS,
        "tau": TAU,
        "nplanes": NPLANES,
        "nchannels": NCHANNELS,
        "registration_json": json.dumps(REGISTRATION, sort_keys=True),
        "run_dir": str(run_dir),
        "source_manifest": str(source_manifest.resolve()),
        "source_task_id": selected["source_task_id"],
        "source_candidates": json.dumps(selected["candidates"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true", help="list unique sessions")
    action.add_argument("--session-index", type=int, help="select an index from --list")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    if args.list:
        print_sessions(args.source_manifest)
        return
    if args.output_root is None or args.manifest is None:
        parser.error("--session-index requires --output-root and --manifest")

    task = build_task(args.source_manifest, args.session_index, args.output_root)
    run_dir = Path(str(task["run_dir"]))
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Output already exists and will not be overwritten: {run_dir}"
        )
    write_csv(args.manifest, [task])
    print(f"Selected: {task['mouse_id']} / {task['session_id']}")
    print(f"Input:    {task['input_path']}")
    print(f"Output:   {task['run_dir']}")
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
