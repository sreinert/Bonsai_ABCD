#!/usr/bin/env python3
"""Create Suite2p tasks for the first and last full session per cohort-2 mouse."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from common import write_csv
from discover_sessions import parse_date, resolve_data_root, validate_tiff_header


TIFF_SUFFIXES = {".tif", ".tiff"}
SESSION_PATTERN = re.compile(r"^ses-(?P<processed>.+?)_date-\d{8}T\d{6}(?:_|$)")
REGISTRATION = {
    "nonrigid": True,
    "block_size": [128, 128],
    "two_step_registration": True,
    "nimg_init": 1000,
    "maxregshift": 0.2,
    "maxregshiftNR": 10,
    "align_by_chan2": True,
    "smooth_sigma": 3.0,
    "smooth_sigma_time": 0,
    "snr_thresh": 1.2,
    "norm_frames": True,
    "do_bidiphase": False,
    "bidiphase": 0.0,
    "batch_size": 100,
    "reg_tif": False,
    "reg_tif_chan2": False,
}


@dataclass(frozen=True)
class FullSession:
    mouse_id: str
    raw_session_id: str
    processed_session_id: str
    input_path: Path
    tiff_files: tuple[Path, ...]


def processed_session_id(raw_session_id: str) -> str:
    match = SESSION_PATTERN.match(raw_session_id)
    if match is None:
        raise ValueError(
            f"Session {raw_session_id!r} does not match "
            "ses-<processed-id>_date-YYYYMMDDTHHMMSS"
        )
    return match.group("processed")


def discover_mouse(mouse_path: Path) -> list[FullSession]:
    sessions: list[FullSession] = []
    for session_path in sorted(path for path in mouse_path.glob("ses-*") if path.is_dir()):
        if parse_date(session_path.name) is None:
            continue
        input_path = session_path / "funcimg"
        if not input_path.is_dir():
            continue
        tiff_files = tuple(
            sorted(
                path.resolve()
                for path in input_path.iterdir()
                if path.is_file() and path.suffix.lower() in TIFF_SUFFIXES
            )
        )
        if not tiff_files:
            continue
        for path in tiff_files:
            validate_tiff_header(path)
        sessions.append(
            FullSession(
                mouse_id=mouse_path.name,
                raw_session_id=session_path.name,
                processed_session_id=processed_session_id(session_path.name),
                input_path=input_path.resolve(),
                tiff_files=tiff_files,
            )
        )
    return sorted(
        sessions,
        key=lambda session: (parse_date(session.raw_session_id), session.raw_session_id),
    )


def first_last(sessions: list[FullSession]) -> list[tuple[str, FullSession]]:
    if not sessions:
        return []
    if len(sessions) == 1:
        return [("only", sessions[0])]
    return [("first", sessions[0]), ("last", sessions[-1])]


def build_tasks(data_root: Path, processed_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for mouse_path in sorted(path for path in data_root.glob("sub-*") if path.is_dir()):
        for role, session in first_last(discover_mouse(mouse_path)):
            run_dir = processed_root / session.mouse_id / session.processed_session_id
            complete = run_dir / "status.json"
            ops_path = run_dir / "suite2p" / "plane0" / "ops.npy"
            if complete.is_file() and ops_path.is_file():
                continue
            suite2p_dir = run_dir / "suite2p"
            if suite2p_dir.exists() and any(suite2p_dir.iterdir()):
                raise FileExistsError(
                    f"Incomplete existing Suite2p output requires inspection: {suite2p_dir}"
                )
            rows.append(
                {
                    "task_id": len(rows),
                    "mouse_id": session.mouse_id,
                    "session_id": session.raw_session_id,
                    "selection_role": role,
                    "candidate_name": "full-session-align-chan2-sigma-3",
                    "candidate_id": "full-session-align-chan2-sigma-3",
                    "suite2p_version": "1.1.0",
                    "input_path": str(session.input_path),
                    "tiff_files_json": json.dumps([str(path) for path in session.tiff_files]),
                    "frame_count": "",
                    "fs": 45.0,
                    "tau": 0.4,
                    "nplanes": 1,
                    "nchannels": 2,
                    "registration_json": json.dumps(REGISTRATION, sort_keys=True),
                    "run_dir": str(run_dir.resolve()),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("AtAp_20260119_SequenceCompression"),
        help="Absolute path or path relative to the MRSIC projects mount",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = resolve_data_root(args.project_root)
    data_root = project_root / "rawdata" / "cohort2"
    processed_root = project_root / "processed" / "cohort2"
    rows = build_tasks(data_root, processed_root)
    if not rows:
        print("No incomplete first/last full sessions found.")
        return
    write_csv(args.manifest, rows, overwrite=args.overwrite)
    print(f"Wrote {len(rows)} full-session tasks to {args.manifest}")
    for row in rows:
        print(
            f"{row['mouse_id']}\t{row['selection_role']}\t{row['session_id']}"
            f"\t{row['run_dir']}"
        )


if __name__ == "__main__":
    main()
