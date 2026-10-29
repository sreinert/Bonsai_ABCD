#!/usr/bin/env python3
"""Extract and deconvolve traces from the reviewed CPSAM detection output."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import os
import platform
import socket
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import skew


PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_DIR))

from full_session_common import write_json_atomic

try:  # Support both direct script execution and package imports in tests.
    from .config import CLASSIFICATION, FS, SUITE2P_VERSION, TAU
except ImportError:
    from full_session_config import CLASSIFICATION, FS, SUITE2P_VERSION, TAU


PLANE_OUTPUTS = (
    "F.npy",
    "Fneu.npy",
    "F_chan2.npy",
    "Fneu_chan2.npy",
    "iscell.npy",
    "spks.npy",
)
REQUIRED_ROI_KEYS = {"ypix", "xpix", "lam", "overlap"}


def atomic_save(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
    os.replace(temporary, path)


def one_plane(run_dir: Path) -> Path:
    planes = sorted(run_dir.glob("suite2p/plane*"))
    if len(planes) != 1:
        raise ValueError(
            f"Expected exactly one Suite2p plane in {run_dir}, found {len(planes)}"
        )
    return planes[0]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binary_snapshot(paths: list[Path]) -> dict[str, dict[str, int]]:
    return {
        str(path): {
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in paths
    }


def configure_settings(settings: dict[str, Any], device: str) -> dict[str, Any]:
    """Lock recording metadata, device, deconvolution, and classifier choice."""
    settings["fs"] = FS
    settings["tau"] = TAU
    settings["torch_device"] = device
    settings["run"]["do_registration"] = 0
    settings["run"]["do_detection"] = False
    settings["run"]["do_deconvolution"] = True
    # A fixed built-in classifier is reproducible across cluster user accounts.
    settings["classification"]["classifier_path"] = None
    settings["classification"]["use_builtin_classifier"] = True
    settings["classification"]["preclassify"] = 0.0
    return settings


def apply_cell_probability_threshold(
    iscell: np.ndarray, threshold: float
) -> np.ndarray:
    """Apply the chosen classifier cutoff while preserving cell probabilities."""
    if iscell.ndim != 2 or iscell.shape[1] != 2:
        raise ValueError(f"iscell has shape {iscell.shape}; expected (n_rois, 2)")
    thresholded = iscell.copy()
    thresholded[:, 0] = thresholded[:, 1] > threshold
    return thresholded


def load_reviewed_detection(
    detection_dir: Path, stat_path: Path | None = None
) -> tuple[np.ndarray, dict[str, Any]]:
    status_path = detection_dir / "status.json"
    stat_path = stat_path or detection_dir / "stat.npy"
    for required in (status_path, stat_path):
        if not required.is_file():
            raise FileNotFoundError(f"Required detection output is missing: {required}")

    import json

    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("state") != "complete":
        raise RuntimeError(
            f"Detection is not complete according to {status_path}: "
            f"state={status.get('state')!r}"
        )
    stat = np.load(stat_path, allow_pickle=True)
    validate_stat(stat)
    recorded_count = status.get("n_rois")
    if recorded_count is not None and int(recorded_count) != len(stat):
        raise RuntimeError(
            f"Detection status records {recorded_count} ROIs but stat.npy has {len(stat)}"
        )
    return stat, status


def promote_output_bundle(output_dir: Path, arrays: dict[str, Any]) -> None:
    """Stage, validate, and promote one complete Suite2p output bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".trace-output-", dir=output_dir.parent)
    )
    staged = temporary / "staged"
    backups = temporary / "backups"
    staged.mkdir()
    backups.mkdir()
    promoted: list[Path] = []
    replaced: list[tuple[Path, Path]] = []
    try:
        for name, value in arrays.items():
            atomic_save(staged / name, value)
        for name in arrays:
            target = output_dir / name
            if target.exists():
                if name not in {"stat.npy", "ops.npy"}:
                    raise FileExistsError(f"Refusing to overwrite trace output: {target}")
                backup = backups / name
                os.replace(target, backup)
                replaced.append((target, backup))
            os.replace(staged / name, target)
            promoted.append(target)
    except BaseException:
        for target in reversed(promoted):
            if target.exists():
                target.unlink()
        for target, backup in reversed(replaced):
            if backup.exists():
                os.replace(backup, target)
        raise
    finally:
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)


def validate_stat(stat: np.ndarray) -> None:
    if stat.ndim != 1 or len(stat) == 0:
        raise ValueError(f"Expected a non-empty 1-D stat array, got shape {stat.shape}")
    for index, roi in enumerate(stat):
        if not isinstance(roi, dict):
            raise TypeError(f"ROI {index} is not a dictionary")
        missing = REQUIRED_ROI_KEYS - set(roi)
        if missing:
            raise ValueError(f"ROI {index} is missing Suite2p fields: {sorted(missing)}")
        lengths = {len(np.asarray(roi[key])) for key in REQUIRED_ROI_KEYS}
        if len(lengths) != 1 or next(iter(lengths)) == 0:
            raise ValueError(f"ROI {index} has empty or inconsistent pixel arrays")


def validate_outputs(
    stat: np.ndarray,
    F: np.ndarray,
    Fneu: np.ndarray,
    F_chan2: np.ndarray,
    Fneu_chan2: np.ndarray,
    spks: np.ndarray,
    iscell: np.ndarray,
    expected_frames: int,
) -> None:
    expected_trace_shape = (len(stat), expected_frames)
    arrays = {
        "F": F,
        "Fneu": Fneu,
        "F_chan2": F_chan2,
        "Fneu_chan2": Fneu_chan2,
        "spks": spks,
    }
    for name, array in arrays.items():
        if array.shape != expected_trace_shape:
            raise RuntimeError(
                f"{name} has shape {array.shape}; expected {expected_trace_shape}"
            )
        if not np.all(np.isfinite(array)):
            raise RuntimeError(f"{name} contains non-finite values")
    if iscell.shape != (len(stat), 2):
        raise RuntimeError(
            f"iscell has shape {iscell.shape}; expected {(len(stat), 2)}"
        )
    if not np.all(np.isfinite(iscell)):
        raise RuntimeError("iscell contains non-finite values")


def validate_registered_binary_sizes(db: dict[str, Any], paths: list[Path]) -> int:
    expected_bytes = int(db["nframes"]) * int(db["Ly"]) * int(db["Lx"]) * 2
    for path in paths:
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            raise RuntimeError(
                f"Registered binary has {actual_bytes} bytes; expected "
                f"{expected_bytes}: {path}"
            )
    return expected_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--detection-dir",
        type=Path,
        help="Reviewed detection directory (default: RUN_DIR/roi_detection/cpsam).",
    )
    parser.add_argument(
        "--stat-path",
        type=Path,
        help="Detected stat.npy when it is stored outside DETECTION_DIR.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Isolated output directory (default: RUN_DIR/trace_extraction/cpsam).",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        help="Status/provenance directory (default: OUTPUT_DIR).",
    )
    parser.add_argument(
        "--allow-output-dir-nonempty",
        action="store_true",
        help="Allow writing validated outputs into an existing Suite2p plane directory.",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    import suite2p
    import torch

    suite2p_version = importlib.metadata.version("suite2p")
    if suite2p_version != SUITE2P_VERSION:
        raise RuntimeError(
            f"Trace extraction requires suite2p=={SUITE2P_VERSION}, "
            f"found {suite2p_version}"
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    run_dir = args.run_dir.resolve()
    plane_dir = one_plane(run_dir)
    detection_dir = (
        args.detection_dir.resolve()
        if args.detection_dir
        else run_dir / "roi_detection" / "cpsam"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else run_dir / "trace_extraction" / "cpsam"
    )
    metadata_dir = (
        args.metadata_dir.resolve() if args.metadata_dir else output_dir
    )
    stat_path = args.stat_path.resolve() if args.stat_path else None
    if (
        not args.allow_output_dir_nonempty
        and output_dir.exists()
        and any(output_dir.iterdir())
    ):
        raise FileExistsError(f"Refusing to overwrite trace output: {output_dir}")
    for name in PLANE_OUTPUTS:
        if (plane_dir / name).exists():
            raise FileExistsError(
                f"Registered plane must remain detection/extraction-free; found "
                f"{plane_dir / name}"
            )

    db_path = plane_dir / "db.npy"
    reg_outputs_path = plane_dir / "reg_outputs.npy"
    data_path = plane_dir / "data.bin"
    data_chan2_path = plane_dir / "data_chan2.bin"
    for required in (db_path, reg_outputs_path, data_path, data_chan2_path):
        if not required.is_file():
            raise FileNotFoundError(f"Required registered input is missing: {required}")

    stat, detection_status = load_reviewed_detection(detection_dir, stat_path)
    resolved_stat_path = stat_path or detection_dir / "stat.npy"
    db = np.load(db_path, allow_pickle=True).item()
    reg_outputs = np.load(reg_outputs_path, allow_pickle=True).item()
    detect_outputs_path = detection_dir / "detect_outputs.npy"
    detect_outputs = (
        np.load(detect_outputs_path, allow_pickle=True).item()
        if detect_outputs_path.is_file()
        else {}
    )
    settings = configure_settings(suite2p.default_settings(), args.device)
    device = torch.device(args.device)
    binaries = [data_path, data_chan2_path]
    expected_binary_bytes = validate_registered_binary_sizes(db, binaries)
    binaries_before = binary_snapshot(binaries)

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    status_path = metadata_dir / "status.json"
    started = time.time()
    write_json_atomic(status_path, {"state": "running", "started_unix": started})

    try:
        extraction_started = time.time()
        with (
            suite2p.io.BinaryFile(
                Ly=int(db["Ly"]), Lx=int(db["Lx"]), filename=data_path,
                n_frames=int(db["nframes"]), write=False,
            ) as f_reg,
            suite2p.io.BinaryFile(
                Ly=int(db["Ly"]), Lx=int(db["Lx"]), filename=data_chan2_path,
                n_frames=int(db["nframes"]), write=False,
            ) as f_reg_chan2,
        ):
            F, Fneu, F_chan2, Fneu_chan2 = suite2p.extraction_wrapper(
                stat,
                f_reg,
                f_reg_chan2=f_reg_chan2,
                settings=settings["extraction"],
                device=device,
            )
        extraction_seconds = time.time() - extraction_started

        corrected = F - settings["extraction"]["neuropil_coefficient"] * Fneu
        variance = corrected.var(axis=1)
        snr = np.divide(
            0.5 * np.diff(corrected, axis=1).var(axis=1),
            variance,
            out=np.full(len(stat), np.inf),
            where=variance > 0,
        )
        snr = 1.0 - snr
        sk = skew(corrected, axis=1)
        standard_deviation = np.std(corrected, axis=1)
        for index, roi in enumerate(stat):
            roi["snr"] = float(snr[index])
            roi["skew"] = float(sk[index])
            roi["std"] = float(standard_deviation[index])

        classification_started = time.time()
        iscell = suite2p.classify(
            stat=stat,
            classfile=suite2p.classification.builtin_classfile,
        )
        iscell = apply_cell_probability_threshold(
            iscell, CLASSIFICATION["cell_probability_threshold"]
        )
        classification_seconds = time.time() - classification_started

        deconvolution_started = time.time()
        baseline_corrected = suite2p.extraction.preprocess(
            F=corrected.copy(),
            fs=settings["fs"],
            batch_size=settings["extraction"]["batch_size"],
            device=device,
            **settings["dcnv_preprocess"],
        )
        spks = suite2p.extraction.oasis(
            F=baseline_corrected,
            batch_size=settings["extraction"]["batch_size"],
            tau=settings["tau"],
            fs=settings["fs"],
        )
        deconvolution_seconds = time.time() - deconvolution_started

        validate_outputs(
            stat, F, Fneu, F_chan2, Fneu_chan2, spks, iscell,
            expected_frames=int(db["nframes"]),
        )
        binaries_after = binary_snapshot(binaries)
        if binaries_after != binaries_before:
            raise RuntimeError("Registered binary metadata changed during read-only extraction")

        plane_times = {
            "extraction": extraction_seconds,
            "classification": classification_seconds,
            "deconvolution": deconvolution_seconds,
            "total_trace_job": time.time() - started,
        }
        ops = {**db, **settings, **reg_outputs, **detect_outputs}
        ops["db"] = db
        ops["settings"] = settings
        ops["reg_outputs"] = reg_outputs
        ops["detect_outputs"] = detect_outputs
        ops["plane_times"] = plane_times

        promote_output_bundle(
            output_dir,
            {
                "F.npy": F,
                "Fneu.npy": Fneu,
                "F_chan2.npy": F_chan2,
                "Fneu_chan2.npy": Fneu_chan2,
                "spks.npy": spks,
                "iscell.npy": iscell,
                "stat.npy": stat,
                "ops.npy": ops,
            },
        )

        finished = time.time()
        provenance = {
            "state": "complete",
            "started_unix": started,
            "finished_unix": finished,
            "elapsed_seconds": finished - started,
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "suite2p_version": suite2p_version,
            "torch_version": importlib.metadata.version("torch"),
            "torch_cuda_version": getattr(torch.version, "cuda", None),
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "run_dir": str(run_dir),
            "plane_dir": str(plane_dir),
            "detection_dir": str(detection_dir),
            "detection_stat_sha256": file_sha256(resolved_stat_path),
            "detection_state": detection_status.get("state"),
            "output_dir": str(output_dir),
            "n_rois": int(len(stat)),
            "n_cells": int(np.count_nonzero(iscell[:, 0])),
            "iscell_probability_threshold": CLASSIFICATION[
                "cell_probability_threshold"
            ],
            "n_frames": int(F.shape[1]),
            "functional_channel": int(db.get("functional_chan", 1)),
            "anatomical_channel_extracted": True,
            "registered_binaries_before": binaries_before,
            "registered_binaries_after": binaries_after,
            "expected_bytes_per_registered_binary": expected_binary_bytes,
            "settings": {
                "fs": settings["fs"],
                "tau": settings["tau"],
                "classification": settings["classification"],
                "extraction": settings["extraction"],
                "dcnv_preprocess": settings["dcnv_preprocess"],
            },
            "timings_seconds": plane_times,
            "outputs": {
                name: str(output_dir / name)
                for name in (
                    "F.npy", "Fneu.npy", "F_chan2.npy", "Fneu_chan2.npy",
                    "spks.npy", "iscell.npy", "stat.npy", "ops.npy",
                )
            },
        }
        write_json_atomic(metadata_dir / "provenance.json", provenance)
        write_json_atomic(status_path, provenance)
        print(
            f"Trace extraction completed: {len(stat)} ROIs, "
            f"{provenance['n_cells']} classified cells, {F.shape[1]} frames"
        )
        print(f"Outputs: {output_dir}")
    except Exception as exc:
        write_json_atomic(
            status_path,
            {
                "state": "failed",
                "started_unix": started,
                "finished_unix": time.time(),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()
