#!/usr/bin/env python3
"""Run registration, detection, extraction, and QC sequentially in one job."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


WORKFLOW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_DIR))

from common import read_csv, write_json_atomic  # noqa: E402
from remaining_steps.prepare_full_session_pipeline import (  # noqa: E402
    clean_downstream_outputs,
    validate_registration_for_resume,
)


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
        "--start-stage", choices=("registration", "detection"), default="registration"
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
        str(WORKFLOW_DIR / "run_candidate.py"),
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
        else:
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

        run_stage(
            "detection",
            [
                python_executable,
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
        run_stage(
            "trace_extraction",
            [
                python_executable,
                str(script_dir / "run_trace_extraction.py"),
                "--run-dir",
                str(run_dir),
                "--detection-dir",
                str(qc_dir / "detection"),
                "--stat-path",
                str(plane_dir / "stat.npy"),
                "--output-dir",
                str(plane_dir),
                "--metadata-dir",
                str(workflow_dir / "trace_extraction"),
                "--allow-output-dir-nonempty",
                "--device",
                "cuda",
            ],
            state,
            status_path,
        )
        run_stage(
            "qc",
            [
                python_executable,
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
