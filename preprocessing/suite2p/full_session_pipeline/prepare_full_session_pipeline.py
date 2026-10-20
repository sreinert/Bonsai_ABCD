#!/usr/bin/env python3
"""Prepare one exact raw session for the complete Suite2p workflow."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np


PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_DIR))

from full_session_common import (  # noqa: E402
    is_relative_to,
    paths_overlap,
    resolve_data_root,
    validate_tiff_header,
    write_csv,
)
from full_session_config import (  # noqa: E402
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

REGISTRATION_OUTPUTS = (
    "data.bin",
    "data_chan2.bin",
    "db.npy",
    "ops.npy",
    "reg_outputs.npy",
)

DOWNSTREAM_PLANE_OUTPUTS = (
    "stat.npy",
    "iscell.npy",
    "spks.npy",
    "F.npy",
    "Fneu.npy",
    "F_chan2.npy",
    "Fneu_chan2.npy",
)


def file_snapshot(paths: list[Path]) -> dict[str, dict[str, int]]:
    return {
        str(path): {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for path in paths
    }


def validate_registration_for_resume(suite2p_dir: Path, tiffs: list[Path]) -> dict[str, Any]:
    """Require a complete, internally consistent registration before resuming."""
    if suite2p_dir.is_symlink():
        raise ValueError(f"Refusing to resume through symlinked output: {suite2p_dir}")
    plane_dir = suite2p_dir / "plane0"
    workflow_dir = suite2p_dir / ".workflow"
    for name in REGISTRATION_OUTPUTS:
        path = plane_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Cannot resume: registration output is missing: {path}")
    status_path = workflow_dir / "registration_status.json"
    if not status_path.is_file():
        raise FileNotFoundError(f"Cannot resume without registration status: {status_path}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("state") != "complete":
        raise RuntimeError(
            f"Cannot resume: registration state is {status.get('state')!r} in {status_path}"
        )
    recorded_snapshot = status.get("source_snapshot_after")
    if recorded_snapshot and recorded_snapshot != file_snapshot(tiffs):
        raise RuntimeError("Cannot resume because raw TIFF metadata changed after registration")

    db = np.load(plane_dir / "db.npy", allow_pickle=True).item()
    reg_outputs = np.load(plane_dir / "reg_outputs.npy", allow_pickle=True).item()
    ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
    ly, lx, nframes = (int(db[key]) for key in ("Ly", "Lx", "nframes"))
    if min(ly, lx, nframes) < 1:
        raise RuntimeError(
            f"Cannot resume with invalid registration dimensions: {(ly, lx, nframes)}"
        )
    expected_bytes = ly * lx * nframes * np.dtype(np.int16).itemsize
    for name in ("data.bin", "data_chan2.bin"):
        path = plane_dir / name
        if path.stat().st_size != expected_bytes:
            raise RuntimeError(
                f"Cannot resume: {path} has {path.stat().st_size} bytes; "
                f"expected {expected_bytes}"
            )
    mean_image = np.asarray(reg_outputs.get("meanImg"))
    badframes = np.asarray(reg_outputs.get("badframes"), dtype=bool).ravel()
    if mean_image.shape != (ly, lx):
        raise RuntimeError(
            f"Cannot resume: registered mean image has shape {mean_image.shape}; "
            f"expected {(ly, lx)}"
        )
    if badframes.size != nframes:
        raise RuntimeError(
            f"Cannot resume: bad-frame mask has length {badframes.size}; expected {nframes}"
        )
    if not isinstance(ops, dict):
        raise TypeError(f"Cannot resume: registration ops is not a dictionary: {plane_dir / 'ops.npy'}")
    return {
        "registration_status": str(status_path),
        "nframes": nframes,
        "Ly": ly,
        "Lx": lx,
        "expected_binary_bytes": expected_bytes,
    }


def clean_downstream_outputs(suite2p_dir: Path) -> list[str]:
    """Remove only products at or after detection, preserving registration and logs."""
    if suite2p_dir.is_symlink() or suite2p_dir.name != "suite2p":
        raise ValueError(f"Unsafe Suite2p resume directory: {suite2p_dir}")
    removed: list[str] = []
    plane_dir = suite2p_dir / "plane0"
    for name in DOWNSTREAM_PLANE_OUTPUTS:
        path = plane_dir / name
        if path.exists():
            path.unlink()
            removed.append(str(path))
    for path in (
        suite2p_dir / "qc" / "detection",
        suite2p_dir / "qc" / "traces",
        suite2p_dir / ".workflow" / "trace_extraction",
    ):
        if path.exists():
            shutil.rmtree(path)
            removed.append(str(path))
    for path in (
        suite2p_dir / "qc" / "index.html",
        suite2p_dir / "qc" / "summary.png",
        suite2p_dir / "qc" / "session_metrics.csv",
        suite2p_dir / "qc" / "calculation.json",
        suite2p_dir / ".workflow" / "provenance.json",
    ):
        if path.exists():
            path.unlink()
            removed.append(str(path))
    for path in suite2p_dir.glob(".trace-output-*"):
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
            removed.append(str(path))
    return removed


def build_pipeline(
    project_root: Path,
    mouse_id: str,
    session_id: str,
    *,
    overwrite: bool = False,
    resume: bool = False,
    rerun_detection: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if sum((overwrite, resume, rerun_detection)) > 1:
        raise ValueError(
            "--overwrite, --resume, and --rerun-detection are mutually exclusive"
        )
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
    resume_validation: dict[str, Any] | None = None
    if existing_output and overwrite:
        if suite2p_dir.is_symlink():
            raise ValueError(f"Refusing to overwrite symlinked output: {suite2p_dir}")
        print(
            f"--overwrite: removing existing Suite2p output only: {suite2p_dir}",
            file=sys.stderr,
        )
        shutil.rmtree(suite2p_dir)
        complete = False
    elif existing_output and rerun_detection:
        resume_validation = validate_registration_for_resume(suite2p_dir, tiffs)
        complete = False
    elif rerun_detection and not existing_output:
        raise FileNotFoundError(
            "Cannot rerun detection because no existing Suite2p registration "
            f"was found: {suite2p_dir}"
        )
    elif existing_output and resume and not complete:
        resume_validation = validate_registration_for_resume(suite2p_dir, tiffs)
    elif resume and not existing_output:
        raise FileNotFoundError(
            f"Cannot resume because no existing Suite2p registration was found: {suite2p_dir}"
        )
    elif existing_output and not complete:
        raise FileExistsError(
            "Partial or inconsistent Suite2p output already exists; inspect it "
            f"before rerunning, pass --resume, or pass --overwrite: {suite2p_dir}"
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
        "resume": bool(resume_validation),
        "rerun_detection": rerun_detection,
        "start_stage": "detection" if resume_validation else "registration",
        "resume_validation": resume_validation,
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse validated completed registration and restart at detection.",
    )
    parser.add_argument(
        "--rerun-detection",
        action="store_true",
        help=(
            "For a completed pipeline, preserve validated registration and replace "
            "detection, trace-extraction, and QC outputs."
        ),
    )
    args = parser.parse_args()

    row, paths = build_pipeline(
        args.project_root,
        args.mouse_id,
        args.session_id,
        overwrite=args.overwrite,
        resume=args.resume,
        rerun_detection=args.rerun_detection,
    )
    if paths["already_complete"]:
        if args.paths_json:
            args.paths_json.write_text(json.dumps(paths, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(paths, indent=2))
        return
    manifest = args.manifest or Path(str(paths["workflow_dir"])) / "manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    write_csv(
        manifest,
        [row],
        overwrite=args.resume or args.rerun_detection,
    )
    paths["manifest"] = str(manifest.resolve())
    if args.paths_json:
        args.paths_json.write_text(json.dumps(paths, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
