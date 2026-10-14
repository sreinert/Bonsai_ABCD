#!/usr/bin/env python3
"""Prepare one exact raw session for the complete Suite2p workflow."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


WORKFLOW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_DIR))

from common import is_relative_to, paths_overlap, write_csv  # noqa: E402
from discover_sessions import resolve_data_root, validate_tiff_header  # noqa: E402
from make_full_session_tasks import (  # noqa: E402
    REGISTRATION,
    TIFF_SUFFIXES,
    processed_session_id,
)


REQUIRED_FINAL_OUTPUTS = (
    "ops.npy",
    "stat.npy",
    "iscell.npy",
    "spks.npy",
    "F.npy",
    "Fneu.npy",
    "F_chan2.npy",
    "Fneu_chan2.npy",
)


def build_pipeline(
    project_root: Path,
    mouse_id: str,
    session_id: str,
    *,
    overwrite: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    project = resolve_data_root(project_root)
    for label, value in (("mouse ID", mouse_id), ("session ID", session_id)):
        if value in {"", ".", ".."} or Path(value).name != value:
            raise ValueError(f"{label} must be one directory name, received {value!r}")
    raw_root = (project / "rawdata" / "cohort2").resolve()
    raw_session = (raw_root / mouse_id / session_id).resolve()
    if not is_relative_to(raw_session, raw_root):
        raise ValueError(f"Raw session escapes the cohort data root: {raw_session}")
    funcimg = raw_session / "funcimg"
    if not funcimg.is_dir():
        raise NotADirectoryError(f"Raw funcimg directory not found: {funcimg}")
    tiffs = sorted(
        path.resolve()
        for path in funcimg.iterdir()
        if path.is_file()
        and not path.name.startswith("._")
        and path.suffix.lower() in TIFF_SUFFIXES
    )
    if not tiffs:
        raise FileNotFoundError(f"No direct TIFF files found in {funcimg}")
    for path in tiffs:
        validate_tiff_header(path)

    processed_id = processed_session_id(session_id)
    processed_root = (project / "processed" / "cohort2").resolve()
    run_dir = (processed_root / mouse_id / processed_id).resolve()
    suite2p_dir = run_dir / "suite2p"
    plane_dir = suite2p_dir / "plane0"
    qc_dir = suite2p_dir / "qc"
    workflow_dir = suite2p_dir / ".workflow"
    if paths_overlap(run_dir, funcimg):
        raise ValueError(f"Output {run_dir} overlaps raw input {funcimg}")
    if not is_relative_to(suite2p_dir, processed_root) or suite2p_dir.name != "suite2p":
        raise ValueError(f"Unsafe Suite2p output directory: {suite2p_dir}")

    complete = (qc_dir / "index.html").is_file() and all(
        (plane_dir / name).is_file() for name in REQUIRED_FINAL_OUTPUTS
    )
    existing_output = suite2p_dir.exists() and any(suite2p_dir.iterdir())
    if existing_output and overwrite:
        if suite2p_dir.is_symlink():
            raise ValueError(f"Refusing to overwrite symlinked output: {suite2p_dir}")
        print(
            f"--overwrite: removing existing Suite2p output only: {suite2p_dir}",
            file=sys.stderr,
        )
        shutil.rmtree(suite2p_dir)
        complete = False
    elif existing_output and not complete:
        raise FileExistsError(
            "Partial or inconsistent Suite2p output already exists; inspect it "
            f"before rerunning or pass --overwrite: {suite2p_dir}"
        )

    row = {
        "task_id": 0,
        "mouse_id": mouse_id,
        "session_id": session_id,
        "selection_role": "requested",
        "candidate_name": "full-session-align-chan2-sigma-3",
        "candidate_id": "full-session-align-chan2-sigma-3",
        "suite2p_version": "1.1.0",
        "input_path": str(funcimg.resolve()),
        "tiff_files_json": json.dumps([str(path) for path in tiffs]),
        "frame_count": "",
        "fs": 45.0,
        "tau": 0.4,
        "nplanes": 1,
        "nchannels": 2,
        "registration_json": json.dumps(REGISTRATION, sort_keys=True),
        "run_dir": str(run_dir),
    }
    paths = {
        "already_complete": complete,
        "project_root": str(project),
        "raw_session": str(raw_session.resolve()),
        "input_path": str(funcimg.resolve()),
        "run_dir": str(run_dir),
        "suite2p_dir": str(suite2p_dir),
        "plane_dir": str(plane_dir),
        "qc_dir": str(qc_dir),
        "workflow_dir": str(workflow_dir),
        "report": str(qc_dir / "index.html"),
        "tiff_count": len(tiffs),
        "overwrote_existing": bool(existing_output and overwrite),
    }
    return row, paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mouse_id")
    parser.add_argument("session_id")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("AtAp_20260119_SequenceCompression"),
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--paths-json", type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove only the selected processed session's existing suite2p directory.",
    )
    args = parser.parse_args()

    row, paths = build_pipeline(
        args.project_root,
        args.mouse_id,
        args.session_id,
        overwrite=args.overwrite,
    )
    if paths["already_complete"]:
        if args.paths_json:
            args.paths_json.write_text(json.dumps(paths, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(paths, indent=2))
        return
    manifest = args.manifest or Path(str(paths["workflow_dir"])) / "manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    write_csv(manifest, [row])
    paths["manifest"] = str(manifest.resolve())
    if args.paths_json:
        args.paths_json.write_text(json.dumps(paths, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
