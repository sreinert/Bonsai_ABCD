from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


WORKFLOW_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(WORKFLOW_DIR))

import make_default_cellpose_pilot_task as pilot_task  # noqa: E402
import run_default_cellpose_detection as detection  # noqa: E402


def test_pilot_task_is_isolated_and_uses_fixed_registration(tmp_path: Path) -> None:
    source = tmp_path / "full_sessions.csv"
    row = {
        "task_id": 0,
        "mouse_id": "sub-02",
        "session_id": "ses-example_date-20260824T120630",
        "selection_role": "first",
        "candidate_name": "old",
        "candidate_id": "old",
        "suite2p_version": "1.1.0",
        "input_path": str(tmp_path / "raw" / "funcimg"),
        "tiff_files_json": json.dumps([str(tmp_path / "raw" / "funcimg" / "a.tif")]),
        "frame_count": "",
        "fs": 45.0,
        "tau": 0.4,
        "nplanes": 1,
        "nchannels": 2,
        "registration_json": "{}",
        "run_dir": str(tmp_path / "processed" / "sub-02" / "example"),
    }
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    result = pilot_task.build_pilot_task(source, 0)
    registration = json.loads(str(result["registration_json"]))
    assert Path(str(result["run_dir"])).name == "default-cpsam"
    assert Path(str(result["run_dir"])).parent.name == "suite2p_pilots"
    assert registration == {
        "align_by_chan2": True,
        "bidiphase": 0.0,
        "block_size": [128, 128],
        "do_bidiphase": False,
        "nimg_init": 1000,
        "nonrigid": True,
        "norm_frames": True,
        "smooth_sigma": 3.0,
        "smooth_sigma_time": 0,
        "two_step_registration": True,
    }


def test_masks_from_stat_and_boundaries() -> None:
    stat = np.asarray(
        [
            {"ypix": np.array([1, 1, 2]), "xpix": np.array([1, 2, 1])},
            {"ypix": np.array([3]), "xpix": np.array([3])},
        ],
        dtype=object,
    )
    masks = detection.masks_from_stat(stat, 5, 5)
    assert set(np.unique(masks)) == {0, 1, 2}
    assert detection.boundaries(masks).sum() == 4


def test_default_cellpose_configuration_changes_only_metadata_device_and_algorithm() -> None:
    settings = {
        "fs": 10.0,
        "tau": 1.0,
        "torch_device": "cpu",
        "diameter": [12.0, 12.0],
        "detection": {
            "algorithm": "sparsery",
            "cellpose_settings": {
                "cellpose_model": "cpsam",
                "img": "max_proj / meanImg",
                "flow_threshold": 0.4,
                "cellprob_threshold": 0.0,
            },
        },
    }
    result = detection.configure_default_cellpose(settings, "cuda")
    assert result["fs"] == 45.0
    assert result["tau"] == 0.4
    assert result["torch_device"] == "cuda"
    assert result["diameter"] == [12.0, 12.0]
    assert result["detection"]["algorithm"] == "cellpose"
    assert result["detection"]["cellpose_settings"] == {
        "cellpose_model": "cpsam",
        "img": "max_proj / meanImg",
        "flow_threshold": 0.4,
        "cellprob_threshold": 0.0,
    }
