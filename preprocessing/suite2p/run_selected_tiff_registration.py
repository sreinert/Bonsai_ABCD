#!/usr/bin/env python3
"""Register one selected TIFF without running detection or trace extraction.

Edit REGISTRATION_PARAMETERS below to try different registration settings, then
run this script with the TIFF path and a new output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import tifffile


# ---------------------------------------------------------------------------
# EDIT REGISTRATION PARAMETERS HERE
#
# These are the settings currently used by the full-session pipeline. Suite2p
# defaults are used for settings that are not listed. To test another value,
# edit it here and use a different output directory for the new run.
# ---------------------------------------------------------------------------
REGISTRATION_PARAMETERS = {
    "align_by_chan2": True,
    "smooth_sigma": 3.0,
    "smooth_sigma_time": 0,
    "norm_frames": True,
    "nimg_init": 1000,
    "do_bidiphase": False,
    "bidiphase": 0.0,
    "nonrigid": True,
    "block_size": [128, 128],
    "two_step_registration": True,

    # Uncomment/edit any settings you want to test:
    # "maxregshift": 0.2,
    # "maxregshiftNR": 10,
    # "snr_thresh": 1.2,
}


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR / "full_session_pipeline"
RUNNER = PIPELINE_DIR / "run_registration.py"
SUITE2P_VERSION = "1.1.0"
FS = 45.0
TAU = 0.4
NPLANES = 1
NCHANNELS = 2


def tiff_frame_count(path: Path) -> int:
    """Count TIFF pages/timepoints without loading the image data."""
    with tifffile.TiffFile(path) as tif:
        if len(tif.pages) > 1:
            return len(tif.pages)
        if not tif.series:
            return len(tif.pages)
        shape = tuple(int(value) for value in tif.series[0].shape)
        axes = tif.series[0].axes
        if "T" in axes:
            return shape[axes.index("T")]
        if len(shape) > 2:
            count = 1
            for dimension in shape[:-2]:
                count *= dimension
            return count
        return len(tif.pages)


def make_manifest_row(tiff: Path, run_dir: Path, frame_count: int) -> dict[str, object]:
    return {
        "task_id": 0,
        "mouse_id": "manual",
        "session_id": tiff.stem,
        "selection_role": "manually-selected-tiff",
        "candidate_name": "manual-registration",
        "candidate_id": "manual-registration",
        "suite2p_version": SUITE2P_VERSION,
        "input_path": str(tiff.parent),
        "tiff_files_json": json.dumps([str(tiff)]),
        "frame_count": frame_count,
        "fs": FS,
        "tau": TAU,
        "nplanes": NPLANES,
        "nchannels": NCHANNELS,
        "registration_json": json.dumps(REGISTRATION_PARAMETERS, sort_keys=True),
        "run_dir": str(run_dir),
    }


def write_or_validate_manifest(path: Path, row: dict[str, object]) -> None:
    """Create the internal manifest, or ensure an existing run matches it."""
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            existing_rows = list(csv.DictReader(handle))
        expected = {key: str(value) for key, value in row.items()}
        if existing_rows != [expected]:
            raise RuntimeError(
                "This output directory was created with a different TIFF or different "
                "registration parameters. Use a new output directory for this test: "
                f"{path.parent.parent}"
            )
        return

    run_dir = path.parent.parent
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            "Output directory is not empty and was not created by this script; "
            f"choose a new output directory: {run_dir}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tiff", type=Path, help="one TIFF stack to register")
    parser.add_argument("output_dir", type=Path, help="new directory for this run")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    tiff = args.tiff.expanduser().resolve()
    run_dir = args.output_dir.expanduser().resolve()
    if not tiff.is_file():
        raise FileNotFoundError(f"Selected TIFF does not exist: {tiff}")
    if tiff.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError(f"Selected input is not a TIFF: {tiff}")
    if (
        run_dir == tiff.parent
        or tiff.parent in run_dir.parents
        or run_dir in tiff.parent.parents
    ):
        raise ValueError("Output directory must be outside the raw TIFF directory")

    frame_count = tiff_frame_count(tiff)
    manifest = run_dir / ".workflow" / "registration_manifest.csv"
    row = make_manifest_row(tiff, run_dir, frame_count)
    write_or_validate_manifest(manifest, row)

    print(f"Selected TIFF: {tiff}")
    print(f"TIFF frames:   {frame_count}")
    print(f"Output:        {run_dir}")
    print("Registration parameters:")
    print(json.dumps(REGISTRATION_PARAMETERS, indent=2, sort_keys=True))

    subprocess.run(
        [
            sys.executable,
            "-u",
            str(RUNNER),
            "--manifest",
            str(manifest),
            "--task-id",
            "0",
            "--device",
            args.device,
        ],
        check=True,
    )

    plane_dir = run_dir / "suite2p" / "plane0"
    print(f"Registration complete: {plane_dir}")
    print(f"Mean-image TIFF:       {plane_dir / 'meanImg_chan1.tiff'}")
    print(f"Mean-image PNG:        {plane_dir / 'meanImg_chan1.png'}")


if __name__ == "__main__":
    main()
