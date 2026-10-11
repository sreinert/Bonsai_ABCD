from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT_DIR = Path(__file__).parents[1]
WORKFLOW_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(WORKFLOW_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

import config  # noqa: E402
import prepare_session  # noqa: E402
import run_roi_detection  # noqa: E402


def source_row(tmp_path: Path, *, task_id: int, session: str, candidate: str) -> dict[str, object]:
    input_path = tmp_path / "raw" / "sub-02" / session / "funcimg"
    return {
        "task_id": task_id,
        "mouse_id": "sub-02",
        "session_id": session,
        "selection_role": "first",
        "candidate_name": candidate,
        "candidate_id": candidate,
        "suite2p_version": "1.1.0",
        "input_path": str(input_path),
        "tiff_files_json": json.dumps([str(input_path / "movie.tif")]),
        "frame_count": "",
        "fs": 45.0,
        "tau": 0.4,
        "nplanes": 1,
        "nchannels": 2,
        "registration_json": "{}",
        "run_dir": str(tmp_path / "old" / session / candidate),
    }


def write_source(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_unique_sessions_collapses_grid_candidates(tmp_path: Path) -> None:
    source = tmp_path / "tasks.csv"
    rows = [
        source_row(tmp_path, task_id=0, session="ses-a", candidate="candidate-a"),
        source_row(tmp_path, task_id=1, session="ses-a", candidate="candidate-b"),
        source_row(tmp_path, task_id=2, session="ses-b", candidate="candidate-a"),
    ]
    write_source(source, rows)

    sessions = prepare_session.unique_sessions(source)

    assert len(sessions) == 2
    assert sessions[0]["source_task_id"] == 0
    assert sessions[0]["candidates"] == ["candidate-a", "candidate-b"]
    assert sessions[1]["row"]["session_id"] == "ses-b"


def test_builds_exactly_one_isolated_task_with_locked_parameters(tmp_path: Path) -> None:
    source = tmp_path / "tasks.csv"
    write_source(
        source,
        [source_row(tmp_path, task_id=0, session="ses-a", candidate="old")],
    )

    task = prepare_session.build_task(source, 0, tmp_path / "processed")

    assert task["task_id"] == 0
    assert task["fs"] == 45.0
    assert task["tau"] == 0.4
    assert task["nplanes"] == 1
    assert task["nchannels"] == 2
    assert json.loads(str(task["registration_json"])) == config.REGISTRATION
    assert Path(str(task["run_dir"])) == (
        tmp_path
        / "processed"
        / "sub-02"
        / "ses-a"
        / "chosen-registration-cpsam"
    )


def test_rejects_output_overlapping_raw_data(tmp_path: Path) -> None:
    source = tmp_path / "tasks.csv"
    row = source_row(tmp_path, task_id=0, session="ses-a", candidate="old")
    write_source(source, [row])

    with pytest.raises(ValueError, match="overlaps raw input"):
        prepare_session.build_task(source, 0, Path(str(row["input_path"])))


def test_detection_defaults_and_visual_mask_helpers() -> None:
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
    configured = run_roi_detection.configure_default_cellpose(settings, "cuda")
    assert configured["fs"] == config.FS
    assert configured["tau"] == config.TAU
    assert configured["detection"]["algorithm"] == "cellpose"

    stat = np.asarray(
        [{"ypix": np.array([1, 1, 2]), "xpix": np.array([1, 2, 1])}],
        dtype=object,
    )
    masks = run_roi_detection.masks_from_stat(stat, 4, 4)
    assert masks.max() == 1
    assert run_roi_detection.boundaries(masks).any()


def test_detection_forbids_downstream_outputs() -> None:
    assert set(run_roi_detection.FORBIDDEN_DOWNSTREAM_OUTPUTS) == {
        "F.npy",
        "Fneu.npy",
        "iscell.npy",
        "spks.npy",
    }
