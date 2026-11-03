"""Persist and execute immutable registration task manifests."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from common import write_json_atomic


def _string_rows(rows: list[dict[str, object]]) -> list[dict[str, str]]:
    return [{key: str(value) for key, value in row.items()} for row in rows]


def write_or_validate_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("Task manifest cannot be empty")
    expected = _string_rows(rows)
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
        if existing != expected:
            raise RuntimeError(
                f"Existing manifest does not match this experiment: {path}. "
                "Choose a new output root for changed sessions or parameters."
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_or_validate_experiment(path: Path, payload: dict[str, Any]) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(
                f"Existing experiment specification does not match: {path}. "
                "Choose a new output root for changed sessions or parameters."
            )
        return
    write_json_atomic(path, payload)


def run_direct(
    manifest: Path,
    task_count: int,
    output_root: Path,
    runner: Path,
    *,
    device: str,
) -> None:
    if task_count < 1:
        raise ValueError("Direct execution requires at least one task")
    logs = output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    python_executable = os.environ.get("PYTHON_EXECUTABLE", sys.executable)
    if not Path(python_executable).is_file():
        raise FileNotFoundError(f"Python executable not found: {python_executable}")
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    for task_id in range(task_count):
        stdout_path = logs / f"direct_task_{task_id}.out"
        stderr_path = logs / f"direct_task_{task_id}.err"
        print(f"Running task {task_id}/{task_count - 1}; logs: {stdout_path}")
        with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open(
            "a", encoding="utf-8"
        ) as stderr:
            subprocess.run(
                [
                    python_executable,
                    "-u",
                    str(runner),
                    "--manifest",
                    str(manifest),
                    "--task-id",
                    str(task_id),
                    "--device",
                    device,
                ],
                check=True,
                stdout=stdout,
                stderr=stderr,
                env=environment,
            )


def submit_slurm(
    manifest: Path,
    task_count: int,
    output_root: Path,
    array_runner: Path,
    repo_root: Path,
    *,
    input_kind: str,
) -> None:
    if task_count < 1:
        raise ValueError("Slurm submission requires at least one task")
    defaults = {
        "small": {"concurrency": 6, "time": "04:00:00"},
        "full": {"concurrency": 4, "time": "12:00:00"},
    }[input_kind]
    concurrency = int(os.environ.get("MAX_CONCURRENT", defaults["concurrency"]))
    if concurrency < 1:
        raise ValueError("MAX_CONCURRENT must be positive")
    array_spec = os.environ.get(
        "TASK_RANGE", f"0-{task_count - 1}%{concurrency}"
    )
    time_limit = os.environ.get("SLURM_TIME", str(defaults["time"]))
    exclude_nodes = os.environ.get("EXCLUDE_NODES", "gpu-350-02,gpu-350-04")
    python_executable = os.environ.get("PYTHON_EXECUTABLE", sys.executable)
    if not Path(python_executable).is_file():
        raise FileNotFoundError(f"Python executable not found: {python_executable}")
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"

    subprocess.run(
        [
            python_executable,
            "-c",
            (
                "import importlib.metadata,sys; "
                "v=importlib.metadata.version('suite2p'); "
                "assert v=='1.1.0', f'expected suite2p==1.1.0, found {v}'; "
                "print('Validated', sys.executable, 'suite2p', v)"
            ),
        ],
        check=True,
        env=environment,
    )

    logs = output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    command = [
        "sbatch",
        f"--array={array_spec}",
        f"--time={time_limit}",
        f"--chdir={output_root}",
        f"--output={logs}/s2p_reg_%A_%a.out",
        f"--error={logs}/s2p_reg_%A_%a.err",
    ]
    if exclude_nodes:
        command.append(f"--exclude={exclude_nodes}")
    command.extend(
        [str(array_runner), str(manifest), str(repo_root), python_executable]
    )
    subprocess.run(command, check=True, env=environment)
