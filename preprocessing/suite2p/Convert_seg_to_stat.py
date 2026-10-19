#!/usr/bin/env python3
"""Replace Suite2p ROIs and traces with a reviewed Cellpose segmentation."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "full_session_pipeline"))

from full_session_common import is_relative_to, resolve_data_root  # noqa: E402
from full_session_config import SUITE2P_VERSION  # noqa: E402


DEFAULT_PROJECT_ROOT = Path("AtAp_20260119_SequenceCompression")
SESSION_PATTERN = re.compile(
    r"^ses-(?P<processed>.+?)_date-\d{8}(?:T\d{6})?(?:_|$)",
    re.IGNORECASE,
)
SEGMENTATION_NAMES = ("meanImg_chan1_seg.npy", "meanImg_seg.npy")


def processed_session_id(session: str) -> str:
    """Accept a full raw-session name or the short processed-session ID."""
    if not session or session in {".", ".."} or Path(session).name != session:
        raise ValueError(f"Session must be one directory name, received {session!r}")
    match = SESSION_PATTERN.match(session)
    if match is not None:
        return match.group("processed")
    return session.removeprefix("ses-")


def resolve_paths(
    subject: str,
    session: str,
    *,
    project_root: Path,
    processed_root: Path | None = None,
    meanimg_folder: Path | None = None,
    segmentation: Path | None = None,
) -> tuple[Path, Path]:
    """Return the reviewed segmentation and Suite2p plane directory."""
    if not subject or subject in {".", ".."} or Path(subject).name != subject:
        raise ValueError(f"Subject must be one directory name, received {subject!r}")
    if processed_root is None:
        processed_root = resolve_data_root(project_root) / "processed" / "cohort2"
    else:
        processed_root = processed_root.expanduser().resolve()

    run_dir = (processed_root / subject / processed_session_id(session)).resolve()
    if not is_relative_to(run_dir, processed_root):
        raise ValueError(f"Resolved session escapes the processed root: {run_dir}")
    suite2p_dir = run_dir / "suite2p"
    plane_dir = suite2p_dir / "plane0"
    if not plane_dir.is_dir():
        raise NotADirectoryError(f"Suite2p plane directory not found: {plane_dir}")

    if segmentation is not None:
        segmentation_path = segmentation.expanduser().resolve()
    else:
        if meanimg_folder is None:
            segmentation_dir = suite2p_dir / "qc" / "registration"
        else:
            expanded = meanimg_folder.expanduser()
            segmentation_dir = (
                expanded if expanded.is_absolute() else suite2p_dir / expanded
            ).resolve()
        if not segmentation_dir.is_dir():
            raise NotADirectoryError(
                f"Mean-image folder not found: {segmentation_dir}"
            )
        candidates = [segmentation_dir / name for name in SEGMENTATION_NAMES]
        existing = [path for path in candidates if path.is_file()]
        if not existing:
            raise FileNotFoundError(
                "Manual Cellpose segmentation not found; checked: "
                + ", ".join(str(path) for path in candidates)
            )
        if len(existing) > 1:
            raise RuntimeError(
                "More than one supported segmentation file exists; pass "
                f"--segmentation explicitly: {existing}"
            )
        segmentation_path = existing[0]
    if not segmentation_path.is_file():
        raise FileNotFoundError(f"Segmentation file not found: {segmentation_path}")
    return segmentation_path, plane_dir


def load_masks(path: Path) -> np.ndarray:
    payload = np.load(path, allow_pickle=True)
    if payload.ndim == 0:
        payload = payload.item()
    if isinstance(payload, dict):
        if "masks" not in payload:
            raise KeyError(f"Cellpose segmentation has no 'masks' entry: {path}")
        masks = np.asarray(payload["masks"])
    else:
        masks = np.asarray(payload)
    if masks.ndim != 2:
        raise ValueError(f"Expected a 2-D Cellpose mask, got shape {masks.shape}: {path}")
    if not np.all(np.isfinite(masks)):
        raise ValueError(f"Cellpose mask contains non-finite values: {path}")
    if not np.all(masks == np.floor(masks)) or np.any(masks < 0):
        raise ValueError(f"Cellpose mask labels must be non-negative integers: {path}")
    masks = masks.astype(np.int32, copy=False)
    if not np.any(masks > 0):
        raise ValueError(f"Cellpose mask contains no labelled ROIs: {path}")
    return masks


def raw_stat_from_masks(masks: np.ndarray) -> np.ndarray:
    """Construct the minimal ROI dictionaries consumed by Suite2p roi_stats."""
    stat: list[dict[str, Any]] = []
    for label in np.unique(masks):
        if label == 0:
            continue
        ypix, xpix = np.nonzero(masks == label)
        npix = len(ypix)
        stat.append(
            {
                "ypix": ypix,
                "xpix": xpix,
                "npix": npix,
                "lam": np.ones(npix, dtype=np.float32),
                "med": [float(np.mean(ypix)), float(np.mean(xpix))],
            }
        )
    return np.asarray(stat, dtype=object)


def validate_trace_outputs(
    stat: np.ndarray, arrays: dict[str, np.ndarray], expected_frames: int
) -> None:
    expected_trace_shape = (len(stat), expected_frames)
    for name in ("F.npy", "Fneu.npy", "spks.npy"):
        if arrays[name].shape != expected_trace_shape:
            raise RuntimeError(
                f"{name} has shape {arrays[name].shape}; expected {expected_trace_shape}"
            )
        if not np.all(np.isfinite(arrays[name])):
            raise RuntimeError(f"{name} contains non-finite values")
    for name in ("F_chan2.npy", "Fneu_chan2.npy"):
        if name in arrays:
            if arrays[name].shape != expected_trace_shape:
                raise RuntimeError(
                    f"{name} has shape {arrays[name].shape}; expected {expected_trace_shape}"
                )
            if not np.all(np.isfinite(arrays[name])):
                raise RuntimeError(f"{name} contains non-finite values")
    if arrays["iscell.npy"].shape != (len(stat), 2):
        raise RuntimeError(
            f"iscell.npy has shape {arrays['iscell.npy'].shape}; "
            f"expected {(len(stat), 2)}"
        )
    if not np.all(np.isfinite(arrays["iscell.npy"])):
        raise RuntimeError("iscell.npy contains non-finite values")


def backup_and_promote(
    plane_dir: Path, arrays: dict[str, np.ndarray], backup_root: Path
) -> Path:
    """Stage a complete output bundle, retain old files, and then promote it."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = backup_root / timestamp
    suffix = 1
    while backup_dir.exists():
        backup_dir = backup_root / f"{timestamp}-{suffix}"
        suffix += 1

    temporary = Path(tempfile.mkdtemp(prefix=".manual-seg-", dir=plane_dir))
    staged = temporary / "staged"
    staged.mkdir()
    try:
        for name, value in arrays.items():
            with (staged / name).open("wb") as handle:
                np.save(handle, value)

        backup_dir.mkdir(parents=True)
        for name in arrays:
            source = plane_dir / name
            if not source.exists():
                continue
            destination = backup_dir / name
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)

        replaced: list[str] = []
        try:
            for name in arrays:
                os.replace(staged / name, plane_dir / name)
                replaced.append(name)
        except BaseException:
            for name in replaced:
                backup = backup_dir / name
                if backup.exists():
                    os.replace(backup, plane_dir / name)
                else:
                    (plane_dir / name).unlink(missing_ok=True)
            raise
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return backup_dir


def convert(
    segmentation_path: Path,
    plane_dir: Path,
    *,
    device_name: str,
    dry_run: bool,
) -> tuple[int, int | None, Path | None]:
    masks = load_masks(segmentation_path)
    ops_path = plane_dir / "ops.npy"
    data_path = plane_dir / "data.bin"
    for required in (ops_path, data_path):
        if not required.is_file():
            raise FileNotFoundError(f"Required Suite2p file is missing: {required}")
    ops = np.load(ops_path, allow_pickle=True).item()
    Ly, Lx = int(ops["Ly"]), int(ops["Lx"])
    if masks.shape != (Ly, Lx):
        raise ValueError(
            f"Segmentation shape {masks.shape} does not match Suite2p {(Ly, Lx)}"
        )
    raw_stat = raw_stat_from_masks(masks)
    if dry_run:
        return len(raw_stat), None, None

    import suite2p
    import torch
    from scipy.stats import skew
    from suite2p.detection import roi_stats

    suite2p_version = importlib.metadata.version("suite2p")
    if suite2p_version != SUITE2P_VERSION:
        raise RuntimeError(
            f"Manual segmentation import requires suite2p=={SUITE2P_VERSION}, "
            f"found {suite2p_version}"
        )
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    device = torch.device(device_name)

    settings = ops.get("settings")
    if not isinstance(settings, dict):
        settings = suite2p.default_settings()
    extraction_settings = ops.get("extraction", settings["extraction"])
    dcnv_settings = ops.get("dcnv_preprocess", settings["dcnv_preprocess"])
    fs = float(ops.get("fs", settings["fs"]))
    tau = float(ops.get("tau", settings["tau"]))
    nframes = int(ops.get("nframes", data_path.stat().st_size // (Ly * Lx * 2)))
    expected_binary_bytes = nframes * Ly * Lx * 2
    data_chan2_path = plane_dir / "data_chan2.bin"
    binaries = [data_path]
    if data_chan2_path.is_file():
        binaries.append(data_chan2_path)
    for binary in binaries:
        if binary.stat().st_size != expected_binary_bytes:
            raise RuntimeError(
                f"Registered binary has {binary.stat().st_size} bytes; expected "
                f"{expected_binary_bytes}: {binary}"
            )

    stat = roi_stats(raw_stat, Ly, Lx)
    binary_kwargs = {"Ly": Ly, "Lx": Lx, "n_frames": nframes, "write": False}
    with suite2p.io.BinaryFile(filename=data_path, **binary_kwargs) as f_reg:
        if data_chan2_path.is_file():
            with suite2p.io.BinaryFile(
                filename=data_chan2_path, **binary_kwargs
            ) as f_reg_chan2:
                F, Fneu, F_chan2, Fneu_chan2 = suite2p.extraction_wrapper(
                    stat,
                    f_reg,
                    f_reg_chan2=f_reg_chan2,
                    settings=extraction_settings,
                    device=device,
                )
        else:
            F, Fneu, F_chan2, Fneu_chan2 = suite2p.extraction_wrapper(
                stat,
                f_reg,
                f_reg_chan2=None,
                settings=extraction_settings,
                device=device,
            )

    corrected = F - extraction_settings["neuropil_coefficient"] * Fneu
    variance = corrected.var(axis=1)
    snr = 1.0 - np.divide(
        0.5 * np.diff(corrected, axis=1).var(axis=1),
        variance,
        out=np.full(len(stat), np.inf),
        where=variance > 0,
    )
    for index, roi in enumerate(stat):
        roi["snr"] = float(snr[index])
        roi["skew"] = float(skew(corrected[index]))
        roi["std"] = float(np.std(corrected[index]))

    iscell = suite2p.classify(
        stat=stat, classfile=suite2p.classification.builtin_classfile
    )
    baseline_corrected = suite2p.extraction.preprocess(
        F=corrected.copy(),
        fs=fs,
        batch_size=extraction_settings["batch_size"],
        device=device,
        **dcnv_settings,
    )
    spks = suite2p.extraction.oasis(
        F=baseline_corrected,
        batch_size=extraction_settings["batch_size"],
        tau=tau,
        fs=fs,
    )

    arrays = {
        "F.npy": F,
        "Fneu.npy": Fneu,
        "iscell.npy": iscell,
        "spks.npy": spks,
        "stat.npy": stat,
    }
    channel_2 = (F_chan2, Fneu_chan2)
    if any(value is not None and np.asarray(value).size for value in channel_2):
        if any(value is None or not np.asarray(value).size for value in channel_2):
            raise RuntimeError("Suite2p returned an incomplete channel-2 trace pair")
        arrays["F_chan2.npy"] = np.asarray(F_chan2)
        arrays["Fneu_chan2.npy"] = np.asarray(Fneu_chan2)
    validate_trace_outputs(stat, arrays, nframes)
    backup_dir = backup_and_promote(
        plane_dir,
        arrays,
        plane_dir.parent / ".workflow" / "manual_segmentation_backups",
    )
    return len(stat), int(np.count_nonzero(iscell[:, 0])), backup_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject", help="Subject directory, for example sub-02")
    parser.add_argument(
        "session",
        help=(
            "Full raw session name (ses-full002_date-...) or short processed "
            "session ID (full002)"
        ),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Project path or name under the standard /ceph or /Volumes mount",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        help="Override PROJECT_ROOT/processed/cohort2",
    )
    parser.add_argument(
        "--meanimg-folder",
        type=Path,
        help=(
            "Folder containing meanImg_chan1_seg.npy or meanImg_seg.npy. "
            "Relative paths are resolved below the session's suite2p directory "
            "(default: qc/registration; use plane0 for the legacy layout)"
        ),
    )
    parser.add_argument(
        "--segmentation",
        type=Path,
        help="Exact segmentation file; overrides --meanimg-folder",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and validate inputs without importing Suite2p or changing files",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    segmentation_path, plane_dir = resolve_paths(
        args.subject,
        args.session,
        project_root=args.project_root,
        processed_root=args.processed_root,
        meanimg_folder=args.meanimg_folder,
        segmentation=args.segmentation,
    )
    print(f"Segmentation: {segmentation_path}")
    print(f"Suite2p plane: {plane_dir}")
    n_rois, n_cells, backup_dir = convert(
        segmentation_path,
        plane_dir,
        device_name=args.device,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(f"Dry run succeeded: {n_rois} manual ROIs; no files changed.")
        return
    print(f"Updated Suite2p outputs: {n_rois} ROIs, {n_cells} classified cells.")
    print(f"Previous outputs backed up to: {backup_dir}")


if __name__ == "__main__":
    main()
