#!/usr/bin/env python3
"""Regenerate percentile-contrast QC for a completed detection experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:  # Support direct script execution and package imports in tests.
    from .run_detection_experiment import save_qc
except ImportError:
    from run_detection_experiment import save_qc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args()

    experiment_dir = args.experiment_dir.resolve()
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
    if experiment_dir.parent.name != "experiments":
        raise ValueError(
            "Expected <run-dir>/roi_detection/experiments/<experiment-name>, got "
            f"{experiment_dir}"
        )
    run_dir = experiment_dir.parents[2]
    plane_dirs = sorted(run_dir.glob("suite2p/plane*"))
    if len(plane_dirs) != 1:
        raise ValueError(f"Expected exactly one Suite2p plane under {run_dir}")

    required = {
        "image": experiment_dir / "cellpose_input.npy",
        "crop_masks": experiment_dir / "cellpose_masks_crop.npy",
        "full_masks": experiment_dir / "roi_masks.npy",
        "stat": experiment_dir / "stat.npy",
        "reg_outputs": plane_dirs[0] / "reg_outputs.npy",
    }
    for path in required.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required QC input is missing: {path}")

    provenance_path = experiment_dir / "provenance.json"
    provenance = (
        json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance_path.is_file()
        else {}
    )
    settings = provenance.get("settings", {})
    title = (
        f"{experiment_dir.name}: {settings.get('model', 'Cellpose')} on "
        f"{settings.get('input_image', 'saved input')}"
    )
    reg_outputs = np.load(required["reg_outputs"], allow_pickle=True).item()
    outputs = save_qc(
        experiment_dir,
        np.load(required["image"]),
        np.asarray(reg_outputs["meanImg"]),
        np.load(required["crop_masks"]),
        np.load(required["full_masks"]),
        np.load(required["stat"], allow_pickle=True),
        title,
    )
    print("Refreshed QC with 1st–99th percentile display contrast")
    print(f"QC image: {outputs['qc']}")
    print(f"HTML report: {outputs['html']}")


if __name__ == "__main__":
    main()
