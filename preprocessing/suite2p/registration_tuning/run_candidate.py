#!/usr/bin/env python3
"""Run one registration-only Suite2p task from a generated manifest."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import resource
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

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

# These are Suite2p-generated pre-registration movies. Restrict cleanup to
# this explicit allowlist inside the processed run directory. Registered
# data.bin and data_chan2.bin are permanent downstream inputs.
RAW_BINARY_NAMES = {
    "data_raw.bin",
    "data_raw_chan2.bin",
    "data_chan2_raw.bin",
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
    if not bool(registration.get("nonrigid", False)):
        raise ValueError("This project requires nonrigid=True")
    if not bool(registration.get("two_step_registration", False)):
        raise ValueError("This project requires two_step_registration=True")
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
        "functional_chan": 1,
        "keep_movie_raw": bool(registration.get("two_step_registration", False)),
    }
    return db, settings


def find_ops_files(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("suite2p/plane*/ops.npy"))


def export_registered_montages(ops_files: list[Path]) -> list[Path]:
    # Import lazily so the runner does not initialise matplotlib before Suite2p.
    from evaluate import load_ops, make_registered_montage

    exports: list[Path] = []
    for ops_path in ops_files:
        ops, outputs = load_ops(ops_path)
        for channel in (1, 2):
            exported = make_registered_montage(
                ops_path, ops, outputs, channel=channel
            )
            if exported is None:
                raise RuntimeError(
                    f"Could not export the channel-{channel} registered-frame "
                    f"montage from {ops_path}; derived binaries were retained"
                )
            exports.append(exported)
    return exports


def validate_registered_binaries(
    run_dir: Path, *, require_chan2: bool
) -> list[dict[str, Any]]:
    retained: list[dict[str, Any]] = []
    plane_dirs = sorted(run_dir.glob("suite2p/plane*"))
    if not plane_dirs:
        raise FileNotFoundError(f"No Suite2p plane directories found under {run_dir}")
    for plane_dir in plane_dirs:
        db_path = plane_dir / "db.npy"
        if not db_path.is_file():
            raise FileNotFoundError(f"Missing Suite2p database: {db_path}")
        db = np.load(db_path, allow_pickle=True).item()
        expected_bytes = int(db["nframes"]) * int(db["Ly"]) * int(db["Lx"]) * 2
        names = ["data.bin", *(["data_chan2.bin"] if require_chan2 else [])]
        for name in names:
            path = plane_dir / name
            if not path.is_file():
                raise FileNotFoundError(f"Required registered binary is missing: {path}")
            size = path.stat().st_size
            if size != expected_bytes:
                raise RuntimeError(
                    f"Registered binary has {size} bytes, expected {expected_bytes}: {path}"
                )
            retained.append({"path": str(path), "bytes": size})
    return retained


def remove_generated_raw_binaries(run_dir: Path) -> list[dict[str, Any]]:
    removed: list[dict[str, Any]] = []
    resolved_run_dir = run_dir.resolve()
    for path in sorted(run_dir.glob("suite2p/plane*/*.bin")):
        if path.name not in RAW_BINARY_NAMES:
            continue
        resolved = path.resolve()
        if not is_relative_to(resolved, resolved_run_dir):
            raise ValueError(f"Refusing to remove binary outside processed run: {resolved}")
        size = path.stat().st_size
        path.unlink()
        removed.append({"path": str(path), "bytes": size})
    return removed


def collect_resource_usage(torch_module: Any | None = None) -> dict[str, Any]:
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes. Production jobs run on Linux.
    peak_rss_bytes = int(max_rss if sys.platform == "darwin" else max_rss * 1024)
    usage: dict[str, Any] = {
        "peak_python_rss_bytes": peak_rss_bytes,
        "slurm_mem_per_node": os.environ.get("SLURM_MEM_PER_NODE"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
    }
    try:
        if torch_module is not None and torch_module.cuda.is_available():
            usage["peak_cuda_allocated_bytes"] = int(
                torch_module.cuda.max_memory_allocated()
            )
            usage["peak_cuda_reserved_bytes"] = int(
                torch_module.cuda.max_memory_reserved()
            )
    except Exception as exc:
        usage["cuda_measurement_error"] = repr(exc)
    return usage


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
            montage_exports = export_registered_montages(find_ops_files(run_dir))
            retained_binaries = validate_registered_binaries(
                run_dir, require_chan2=int(row["nchannels"]) == 2
            )
            removed_raw_binaries = remove_generated_raw_binaries(run_dir)
            previous["mean_image_exports"] = [
                str(path) for path in mean_image_exports
            ]
            previous["registered_montage_exports"] = [
                str(path) for path in montage_exports
            ]
            previous["retained_registered_binaries"] = retained_binaries
            previous["removed_generated_raw_binaries"] = [
                *previous.get("removed_generated_raw_binaries", []),
                *removed_raw_binaries,
            ]
            write_json_atomic(status_path, previous)
            print(f"Task {args.task_id} is already complete: {run_dir}")
            print(
                "Mean images: "
                + ", ".join(str(path) for path in mean_image_exports)
            )
            print("Registered-frame montages: " + ", ".join(map(str, montage_exports)))
            if removed_raw_binaries:
                freed = sum(item["bytes"] for item in removed_raw_binaries)
                print(
                    f"Removed {len(removed_raw_binaries)} generated raw binary movie(s); "
                    f"freed {freed / 1024**3:.2f} GiB"
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

    torch_module: Any | None = None
    try:
        import suite2p

        required_version = row["suite2p_version"]
        installed_version = importlib.metadata.version("suite2p")
        if installed_version != required_version:
            raise RuntimeError(
                f"Task requires suite2p=={required_version}, found "
                f"{installed_version} from {suite2p.__file__} using {sys.executable}"
            )

        import torch

        torch_module = torch

        torch_version = importlib.metadata.version("torch")
        torch_module_path = getattr(torch, "__file__", None)
        if not hasattr(torch, "cuda"):
            raise RuntimeError(
                "The imported torch module does not expose torch.cuda. "
                f"Imported from {torch_module_path!r}; torch distribution is "
                f"{torch_version}."
            )
        cuda_available = torch.cuda.is_available()
        if args.device == "cuda" and not cuda_available:
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

        db, settings = build_suite2p_configuration(
            row, input_path, run_dir, tiff_files, args.device
        )
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        write_json_atomic(
            run_dir / "provenance.json",
            {
                **provenance,
                "suite2p_version": installed_version,
                "torch_version": torch_version,
                "torch_module": str(torch_module_path),
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
        montage_exports = export_registered_montages(ops_files)
        retained_binaries = validate_registered_binaries(
            run_dir, require_chan2=int(row["nchannels"]) == 2
        )
        source_after = file_snapshot(tiff_files)
        if source_after != source_before:
            raise RuntimeError("Raw TIFF metadata changed while Suite2p was running")
        removed_raw_binaries = remove_generated_raw_binaries(run_dir)
        finished = time.time()
        resource_usage = collect_resource_usage(torch_module)
        write_json_atomic(
            status_path,
            {
                "state": "complete",
                **provenance,
                "suite2p_version": installed_version,
                "finished_unix": finished,
                "elapsed_seconds": finished - started,
                "resource_usage": resource_usage,
                "ops_files": [str(path) for path in ops_files],
                "mean_image_exports": [str(path) for path in mean_image_exports],
                "registered_montage_exports": [
                    str(path) for path in montage_exports
                ],
                "retained_registered_binaries": retained_binaries,
                "removed_generated_raw_binaries": removed_raw_binaries,
                "suite2p_return": repr(result),
                "source_snapshot_after": source_after,
            },
        )
        print(f"Completed task {args.task_id}: {run_dir}")
        print("Mean images: " + ", ".join(str(path) for path in mean_image_exports))
        print("Registered-frame montages: " + ", ".join(map(str, montage_exports)))
        print(
            "Peak Python RSS: "
            f"{resource_usage['peak_python_rss_bytes'] / 1024**3:.2f} GiB"
        )
        if "peak_cuda_reserved_bytes" in resource_usage:
            print(
                "Peak CUDA memory reserved: "
                f"{resource_usage['peak_cuda_reserved_bytes'] / 1024**3:.2f} GiB"
            )
        if removed_raw_binaries:
            freed = sum(item["bytes"] for item in removed_raw_binaries)
            print(
                f"Removed {len(removed_raw_binaries)} generated raw binary movie(s); "
                f"freed {freed / 1024**3:.2f} GiB"
            )
    except Exception as exc:
        finished = time.time()
        write_json_atomic(
            status_path,
            {
                "state": "failed",
                **provenance,
                "finished_unix": finished,
                "elapsed_seconds": finished - started,
                "resource_usage": collect_resource_usage(torch_module),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()
