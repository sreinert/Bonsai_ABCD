#!/usr/bin/env python3
"""Run registration, detection, extraction, and QC sequentially in one job."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_DIR))

from full_session_common import read_csv, write_json_atomic  # noqa: E402
from prepare_full_session_pipeline import (  # noqa: E402
    clean_downstream_outputs,
    validate_registration_for_resume,
)


TRACE_BUNDLE_NAMES = (
    "F.npy",
    "Fneu.npy",
    "F_chan2.npy",
    "Fneu_chan2.npy",
    "spks.npy",
    "iscell.npy",
    "stat.npy",
    "ops.npy",
)


def backup_trace_bundle(plane_dir: Path, backup_root: Path) -> Path:
    """Keep a recoverable snapshot of the current trace/ROI bundle."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = backup_root / timestamp
    suffix = 1
    while backup_dir.exists():
        backup_dir = backup_root / f"{timestamp}-{suffix}"
        suffix += 1
    backup_dir.mkdir(parents=True)
    for name in TRACE_BUNDLE_NAMES:
        source = plane_dir / name
        if source.is_file():
            destination = backup_dir / name
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
    return backup_dir


def restore_trace_bundle(plane_dir: Path, backup_dir: Path) -> None:
    """Restore a saved bundle atomically per file after a failed rerun."""
    for name in TRACE_BUNDLE_NAMES:
        backup = backup_dir / name
        target = plane_dir / name
        if backup.is_file():
            temporary = target.with_name(f".{target.name}.restore-{os.getpid()}")
            shutil.copy2(backup, temporary)
            os.replace(temporary, target)
        else:
            target.unlink(missing_ok=True)


def prepare_extraction_rerun(suite2p_dir: Path) -> tuple[Path, Path, Path]:
    """Back up traces and clear only outputs that extraction/QC will replace."""
    plane_dir = suite2p_dir / "plane0"
    workflow_dir = suite2p_dir / ".workflow"
    qc_dir = suite2p_dir / "qc"
    backup_dir = backup_trace_bundle(
        plane_dir, workflow_dir / "extraction_rerun_backups"
    )
    stat_backup = backup_dir / "stat.npy"
    iscell_backup = backup_dir / "iscell.npy"
    if not stat_backup.is_file() or not iscell_backup.is_file():
        raise FileNotFoundError(
            "Extraction rerun requires current stat.npy and iscell.npy"
        )
    for name in TRACE_BUNDLE_NAMES:
        if name not in {"stat.npy", "ops.npy"}:
            (plane_dir / name).unlink(missing_ok=True)
    for path in (
        qc_dir / "traces",
        workflow_dir / "trace_extraction",
    ):
        if path.exists():
            shutil.rmtree(path)
    for path in (
        qc_dir / "index.html",
        qc_dir / "summary.png",
        qc_dir / "session_metrics.csv",
        qc_dir / "calculation.json",
        workflow_dir / "provenance.json",
        workflow_dir / "qc_provenance.json",
    ):
        path.unlink(missing_ok=True)
    return backup_dir, stat_backup, iscell_backup


def run_stage(name: str, command: list[str], state: dict[str, Any], status_path: Path) -> None:
    started = time.time()
    state["current_stage"] = name
    state["stages"][name] = {"state": "running", "started_unix": started}
    write_json_atomic(status_path, state)
    try:
        subprocess.run(command, check=True)
    except BaseException:
        state["state"] = "failed"
        state["stages"][name] = {
            "state": "failed",
            "started_unix": started,
            "finished_unix": time.time(),
        }
        write_json_atomic(status_path, state)
        raise
    state["stages"][name] = {
        "state": "complete",
        "started_unix": started,
        "finished_unix": time.time(),
        "elapsed_seconds": time.time() - started,
    }
    write_json_atomic(status_path, state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--script-dir", type=Path, required=True)
    parser.add_argument("--python-executable", type=Path, required=True)
    parser.add_argument(
        "--start-stage",
        choices=("registration", "detection", "trace_extraction"),
        default="registration",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    script_dir = args.script_dir.resolve()
    python_executable = str(args.python_executable.resolve())
    suite2p_dir = run_dir / "suite2p"
    plane_dir = suite2p_dir / "plane0"
    qc_dir = suite2p_dir / "qc"
    workflow_dir = suite2p_dir / ".workflow"
    status_path = workflow_dir / "status.json"
    started = time.time()
    state: dict[str, Any] = {
        "state": "running",
        "started_unix": started,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "execution_mode": (
            "slurm" if os.environ.get("SLURM_JOB_ID") else "current_node"
        ),
        "run_dir": str(run_dir),
        "stages": {},
        "start_stage": args.start_stage,
    }
    write_json_atomic(status_path, state)

    registration_command = [
        python_executable,
        "-u",
        str(PIPELINE_DIR / "run_registration.py"),
        "--manifest",
        str(args.manifest.resolve()),
        "--task-id",
        "0",
        "--device",
        "cuda",
    ]
    try:
        if args.start_stage == "registration":
            try:
                run_stage("registration", registration_command, state, status_path)
            finally:
                subprocess.run(
                    [
                        python_executable,
                        str(script_dir / "organize_registration_outputs.py"),
                        "--run-dir",
                        str(run_dir),
                        "--workflow-dir",
                        str(workflow_dir),
                        "--qc-dir",
                        str(qc_dir / "registration"),
                    ],
                    check=True,
                )
        elif args.start_stage == "detection":
            rows = read_csv(args.manifest.resolve())
            if len(rows) != 1:
                raise ValueError(
                    f"Resume manifest must contain exactly one row: {args.manifest}"
                )
            tiffs = [
                Path(value).resolve()
                for value in json.loads(rows[0]["tiff_files_json"])
            ]
            validation = validate_registration_for_resume(suite2p_dir, tiffs)
            removed = clean_downstream_outputs(suite2p_dir)
            state["stages"]["registration"] = {
                "state": "reused",
                "registration_status": validation["registration_status"],
            }
            state["resume_cleanup"] = removed
            write_json_atomic(status_path, state)

        else:
            rows = read_csv(args.manifest.resolve())
            if len(rows) != 1:
                raise ValueError(
                    f"Rerun manifest must contain exactly one row: {args.manifest}"
                )
            tiffs = [
                Path(value).resolve()
                for value in json.loads(rows[0]["tiff_files_json"])
            ]
            validation = validate_registration_for_resume(suite2p_dir, tiffs)
            backup_dir, stat_backup, iscell_backup = prepare_extraction_rerun(
                suite2p_dir
            )
            state["stages"]["registration"] = {
                "state": "reused",
                "registration_status": validation["registration_status"],
            }
            state["stages"]["detection"] = {
                "state": "preserved_gui_rois",
                "stat_source": str(stat_backup),
                "iscell_source": str(iscell_backup),
            }
            state["trace_bundle_backup"] = str(backup_dir)
            write_json_atomic(status_path, state)

        if args.start_stage != "trace_extraction":
            run_stage(
                "detection",
                [
                    python_executable,
                    "-u",
                    str(script_dir / "run_roi_detection.py"),
                    "--run-dir",
                    str(run_dir),
                    "--output-dir",
                    str(qc_dir / "detection"),
                    "--stat-output",
                    str(plane_dir / "stat.npy"),
                    "--device",
                    "cuda",
                ],
                state,
                status_path,
            )
        extraction_command = [
            python_executable,
            "-u",
            str(script_dir / "run_trace_extraction.py"),
            "--run-dir",
            str(run_dir),
            "--detection-dir",
            str(qc_dir / "detection"),
            "--stat-path",
            str(
                stat_backup
                if args.start_stage == "trace_extraction"
                else plane_dir / "stat.npy"
            ),
            "--output-dir",
            str(plane_dir),
            "--metadata-dir",
            str(workflow_dir / "trace_extraction"),
            "--allow-output-dir-nonempty",
            "--device",
            "cuda",
        ]
        if args.start_stage == "trace_extraction":
            extraction_command.extend(
                ["--use-current-stat", "--iscell-path", str(iscell_backup)]
            )
        run_stage(
            "trace_extraction",
            extraction_command,
            state,
            status_path,
        )
        extraction_completed = True
        run_stage(
            "qc",
            [
                python_executable,
                "-u",
                str(script_dir / "generate_full_session_qc.py"),
                "--run-dir",
                str(run_dir),
                "--script-dir",
                str(script_dir),
            ],
            state,
            status_path,
        )
        finished = time.time()
        state.update(
            {
                "state": "complete",
                "current_stage": None,
                "finished_unix": finished,
                "elapsed_seconds": finished - started,
                "report": str(qc_dir / "index.html"),
            }
        )
        write_json_atomic(status_path, state)
        write_json_atomic(workflow_dir / "provenance.json", state)
        print(f"Full Suite2p pipeline complete: {plane_dir}")
        print(f"QC report: {qc_dir / 'index.html'}")
    except Exception as exc:
        if (
            args.start_stage == "trace_extraction"
            and "backup_dir" in locals()
            and not locals().get("extraction_completed", False)
        ):
            restore_trace_bundle(plane_dir, backup_dir)
            state["trace_bundle_restored_after_failure"] = True
        state.update(
            {
                "state": "failed",
                "finished_unix": time.time(),
                "elapsed_seconds": time.time() - started,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
        write_json_atomic(status_path, state)
        raise


if __name__ == "__main__":
    main()
