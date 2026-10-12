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

from common import (  # noqa: E402
    paths_overlap,
    read_csv,
    remap_mounted_path,
    require_columns,
    slug,
    write_csv,
)
from discover_sessions import validate_tiff_header  # noqa: E402
from run_candidate import TASK_COLUMNS  # noqa: E402

try:  # Support both direct script execution and package imports in tests.
    from .config import FS, NCHANNELS, NPLANES, REGISTRATION, SUITE2P_VERSION, TAU
except ImportError:
    from config import FS, NCHANNELS, NPLANES, REGISTRATION, SUITE2P_VERSION, TAU


def unique_sessions(source_manifest: Path) -> list[dict[str, Any]]:
    """Return one representative row per mouse/session in manifest order."""
    rows = require_columns(read_csv(source_manifest), TASK_COLUMNS, "source manifest")
    sessions: list[dict[str, Any]] = []
    positions: dict[tuple[str, str], int] = {}
    for source_row, row in enumerate(rows):
        key = (row["mouse_id"], row["session_id"])
        if key not in positions:
            positions[key] = len(sessions)
            sessions.append(
                {
                    "source_task_id": source_row,
                    "row": row,
                }
            )
    return sessions


def print_sessions(source_manifest: Path) -> None:
    sessions = unique_sessions(source_manifest)
    print("index\tmouse\tsession\trole\tsource manifest row")
    for index, item in enumerate(sessions):
        row = item["row"]
        print(
            f"{index}\t{row['mouse_id']}\t{row['session_id']}\t"
            f"{row.get('selection_role', '')}\t{item['source_task_id']}"
        )


def discover_full_session(source_input: Path) -> tuple[Path, list[Path]]:
    """Resolve the session's funcimg directory and include every direct TIFF."""
    source_input = remap_mounted_path(source_input, must_exist=True)
    funcimg = next(
        (path for path in (source_input, *source_input.parents) if path.name == "funcimg"),
        None,
    )
    if funcimg is None or not funcimg.is_dir():
        raise NotADirectoryError(
            f"Could not locate the raw funcimg directory above {source_input}"
        )
    tiff_files = sorted(
        path.resolve()
        for path in funcimg.iterdir()
        if path.is_file()
        and not path.name.startswith("._")
        and path.suffix.lower() in {".tif", ".tiff"}
    )
    if not tiff_files:
        raise FileNotFoundError(
            f"No direct TIFF files found for the full session in {funcimg}"
        )
    for path in tiff_files:
        validate_tiff_header(path)
    return funcimg.resolve(), tiff_files


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
    input_path, tiff_files = discover_full_session(Path(source["input_path"]))
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
        "input_path": str(input_path),
        "tiff_files_json": json.dumps([str(path) for path in tiff_files]),
        "frame_count": "",
        "registration_json": json.dumps(REGISTRATION, sort_keys=True),
        "run_dir": str(run_dir),
        "source_manifest": str(source_manifest.resolve()),
        "source_task_id": selected["source_task_id"],
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
    print(f"Input:    {task['input_path']} (all direct TIFF files)")
    print(f"TIFFs:    {len(json.loads(str(task['tiff_files_json'])))}")
    print(f"Output:   {task['run_dir']}")
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
