#!/usr/bin/env python3
"""Run one registration-only Suite2p task from a generated manifest."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from common import (
    is_relative_to,
    paths_overlap,
    read_csv,
    remap_mounted_path,
    require_columns,
    write_json_atomic,
)
from export_mean_images import export_mean_images


TASK_COLUMNS = {
    "task_id",
    "mouse_id",
    "session_id",
    "candidate_name",
    "suite2p_version",
    "input_path",
    "tiff_files_json",
    "fs",
    "tau",
    "nplanes",
    "nchannels",
    "registration_json",
    "run_dir",
}


def read_task(manifest: Path, task_id: int) -> dict[str, str]:
    rows = require_columns(read_csv(manifest), TASK_COLUMNS, "task manifest")
    if task_id < 0 or task_id >= len(rows):
        raise IndexError(f"Task {task_id} is outside 0..{len(rows) - 1}")
    row = rows[task_id]
    if int(row["task_id"]) != task_id:
        raise ValueError("task_id values must match zero-based CSV row order")
    return row


def file_snapshot(paths: list[Path]) -> dict[str, dict[str, int]]:
    return {
        str(path): {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for path in paths
    }


def validate_paths(row: dict[str, str]) -> tuple[Path, Path, list[Path]]:
    input_path = remap_mounted_path(Path(row["input_path"]), must_exist=True)
    run_dir = remap_mounted_path(Path(row["run_dir"]), must_exist=False)
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_path}")
    if paths_overlap(run_dir, input_path):
        raise ValueError(
            f"Refusing unsafe task: output {run_dir} overlaps input {input_path}"
        )
    files = [
        remap_mounted_path(Path(value), must_exist=True)
        for value in json.loads(row["tiff_files_json"])
    ]
    if not files:
        raise ValueError("Task has no input TIFF files")
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"Input TIFF not found: {path}")
        if not is_relative_to(path, input_path):
            raise ValueError(f"Input TIFF is outside declared input directory: {path}")
    return input_path, run_dir, files


def build_suite2p_configuration(
    row: dict[str, str], input_path: Path, run_dir: Path, files: list[Path], device: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    import suite2p

    registration = json.loads(row["registration_json"])
    settings = suite2p.default_settings()
    unknown_registration_keys = set(registration) - set(settings["registration"])
    if unknown_registration_keys:
        raise ValueError(
            "Candidate contains registration settings unknown to this Suite2p "
            f"version: {sorted(unknown_registration_keys)}"
        )
    if isinstance(registration.get("block_size"), list):
        registration["block_size"] = tuple(registration["block_size"])
    if float(registration.get("smooth_sigma_time", 0)) != 0:
        raise ValueError("This project requires smooth_sigma_time=0")
    if float(registration.get("bidiphase", 0)) != 0:
        raise ValueError("This project requires bidiphase=0")
    if bool(registration.get("do_bidiphase", False)):
        raise ValueError("This project requires do_bidiphase=False")
    settings["torch_device"] = device
    settings["fs"] = float(row["fs"])
    settings["tau"] = float(row["tau"])
    settings["run"]["do_registration"] = 2
    frame_count_text = row.get("frame_count", "").strip()
    if frame_count_text:
        timepoints_per_channel = int(frame_count_text) // int(row["nchannels"])
        settings["run"]["do_regmetrics"] = timepoints_per_channel >= 1500
    else:
        # Discovery intentionally avoids walking every TIFF page over network
        # storage. Suite2p will determine the movie length while reading it.
        settings["run"]["do_regmetrics"] = True
    settings["run"]["do_detection"] = False
    settings["run"]["do_deconvolution"] = False
    settings["io"]["delete_bin"] = False
    settings["io"]["move_bin"] = False
    settings["io"]["save_ops_orig"] = True
    settings["registration"].update(registration)

    db: dict[str, Any] = {
        "data_path": [str(input_path)],
        "file_list": [str(path) for path in files],
        "save_path0": str(run_dir),
        "input_format": "tif",
        "nplanes": int(row["nplanes"]),
        "nchannels": int(row["nchannels"]),
        "keep_movie_raw": bool(registration.get("two_step_registration", False)),
    }
    return db, settings


def find_ops_files(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("suite2p/plane*/ops.npy"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    row = read_task(args.manifest, args.task_id)
    input_path, run_dir, tiff_files = validate_paths(row)
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    if status_path.exists():
        previous = json.loads(status_path.read_text(encoding="utf-8"))
        if previous.get("state") == "complete" and find_ops_files(run_dir):
            recorded_snapshot = previous.get("source_snapshot_after")
            if recorded_snapshot and file_snapshot(tiff_files) != recorded_snapshot:
                raise RuntimeError(
                    "Input TIFF metadata changed after this task completed; create a "
                    "new manifest/output root rather than reusing the old result"
                )
            mean_image_exports = export_mean_images(run_dir, require_chan2=True)
            previous["mean_image_exports"] = [
                str(path) for path in mean_image_exports
            ]
            write_json_atomic(status_path, previous)
            print(f"Task {args.task_id} is already complete: {run_dir}")
            print(
                "Mean images: "
                + ", ".join(str(path) for path in mean_image_exports)
            )
            return

    started = time.time()
    source_before = file_snapshot(tiff_files)
    provenance = {
        "task": row,
        "manifest": str(args.manifest.resolve()),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "started_unix": started,
        "raw_input_policy": "read-only; output path validated as non-overlapping",
        "source_snapshot_before": source_before,
    }
    write_json_atomic(status_path, {"state": "running", **provenance})

    try:
        required_version = row["suite2p_version"]
        installed_version = importlib.metadata.version("suite2p")
        if installed_version != required_version:
            raise RuntimeError(
                f"Task requires suite2p=={required_version}, found {installed_version}"
            )

        import suite2p
        import torch

        torch_version = importlib.metadata.version("torch")
        torch_module = getattr(torch, "__file__", None)
        if not hasattr(torch, "cuda"):
            raise RuntimeError(
                "The imported torch module does not expose torch.cuda. "
                f"Imported from {torch_module!r}; torch distribution is "
                f"{torch_version}."
            )
        cuda_available = torch.cuda.is_available()
        if args.device == "cuda" and not cuda_available:
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

        db, settings = build_suite2p_configuration(
            row, input_path, run_dir, tiff_files, args.device
        )
        write_json_atomic(
            run_dir / "provenance.json",
            {
                **provenance,
                "suite2p_version": installed_version,
                "torch_version": torch_version,
                "torch_module": str(torch_module),
                "cuda_available": cuda_available,
                "torch_cuda_version": getattr(
                    getattr(torch, "version", None), "cuda", None
                ),
                "cuda_device": (
                    torch.cuda.get_device_name(0) if cuda_available else None
                ),
                "db": db,
                "settings": settings,
            },
        )

        result = suite2p.run_s2p(settings=settings, db=db)
        ops_files = find_ops_files(run_dir)
        if not ops_files:
            raise RuntimeError("Suite2p returned without creating suite2p/plane*/ops.npy")
        mean_image_exports = export_mean_images(run_dir, require_chan2=True)
        source_after = file_snapshot(tiff_files)
        if source_after != source_before:
            raise RuntimeError("Raw TIFF metadata changed while Suite2p was running")
        finished = time.time()
        write_json_atomic(
            status_path,
            {
                "state": "complete",
                **provenance,
                "suite2p_version": installed_version,
                "finished_unix": finished,
                "elapsed_seconds": finished - started,
                "ops_files": [str(path) for path in ops_files],
                "mean_image_exports": [str(path) for path in mean_image_exports],
                "suite2p_return": repr(result),
                "source_snapshot_after": source_after,
            },
        )
        print(f"Completed task {args.task_id}: {run_dir}")
        print("Mean images: " + ", ".join(str(path) for path in mean_image_exports))
    except Exception as exc:
        finished = time.time()
        write_json_atomic(
            status_path,
            {
                "state": "failed",
                **provenance,
                "finished_unix": finished,
                "elapsed_seconds": finished - started,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()
