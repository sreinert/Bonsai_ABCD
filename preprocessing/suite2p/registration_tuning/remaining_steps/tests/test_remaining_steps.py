from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile


SCRIPT_DIR = Path(__file__).parents[1]
WORKFLOW_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(WORKFLOW_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

import config  # noqa: E402
import inspect_run  # noqa: E402
import prepare_session  # noqa: E402
import run_detection_experiment  # noqa: E402
import run_roi_detection  # noqa: E402


def source_row(tmp_path: Path, *, task_id: int, session: str, candidate: str) -> dict[str, object]:
    funcimg = tmp_path / "raw" / "sub-02" / session / "funcimg"
    input_path = funcimg / "others"
    input_path.mkdir(parents=True, exist_ok=True)
    for name in ("full_001.tif", "full_002.tiff"):
        tifffile.imwrite(funcimg / name, np.zeros((2, 3), dtype=np.uint16))
    pilot_tiff = input_path / "pilot_subset.tif"
    tifffile.imwrite(pilot_tiff, np.zeros((2, 3), dtype=np.uint16))
    return {
        "task_id": task_id,
        "mouse_id": "sub-02",
        "session_id": session,
        "selection_role": "first",
        "candidate_name": candidate,
        "candidate_id": candidate,
        "suite2p_version": "1.1.0",
        "input_path": str(input_path),
        "tiff_files_json": json.dumps([str(pilot_tiff)]),
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
    assert Path(str(task["input_path"])).name == "funcimg"
    assert {Path(path).name for path in json.loads(str(task["tiff_files_json"]))} == {
        "full_001.tif",
        "full_002.tiff",
    }
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


def test_detection_experiment_parses_cellpose_parameters() -> None:
    assert run_detection_experiment.parse_cellpose_params(
        ["min_size=5", "normalize=false", "tile_overlap=0.2"]
    ) == {"min_size": 5, "normalize": False, "tile_overlap": 0.2}
    with pytest.raises(ValueError, match="dedicated"):
        run_detection_experiment.parse_cellpose_params(["diameter=10"])


def test_detection_experiment_builds_each_supported_input() -> None:
    mean = np.full((3, 4), 2.0)
    maximum = np.full((3, 4), 4.0)
    ratio, ratio_weights = run_detection_experiment.build_cellpose_input(
        mean, maximum, "max_proj / meanImg", [12.0, 12.0], 0
    )
    mean_input, mean_weights = run_detection_experiment.build_cellpose_input(
        mean, maximum, "meanImg", [12.0, 12.0], 0
    )
    max_input, max_weights = run_detection_experiment.build_cellpose_input(
        mean, maximum, "max_proj", [12.0, 12.0], 0
    )
    assert np.allclose(ratio, np.log(2))
    assert np.array_equal(ratio_weights, maximum)
    assert np.array_equal(mean_input, mean)
    assert np.isfinite(mean_weights).all()
    assert np.array_equal(max_input, maximum)
    assert np.array_equal(max_weights, maximum)


def test_registration_mean_image_is_an_explicit_input_choice() -> None:
    assert "registration_meanImg" in run_detection_experiment.INPUT_IMAGE_CHOICES


def test_detection_experiment_rejects_cyto3_under_cellpose4() -> None:
    class Models:
        MODEL_NAMES = ["cpsam", "cpsam_v2"]

    with pytest.raises(RuntimeError, match="requires Cellpose 3"):
        run_detection_experiment.validate_model("4.2.1.1", "cyto3", Models)
    assert run_detection_experiment.validate_model("4.2.1.1", "cpsam", Models) == "cpsam"


def test_detection_experiment_zero_roi_qc_is_saved(tmp_path: Path) -> None:
    outputs = run_detection_experiment.save_qc(
        tmp_path,
        np.zeros((4, 4)),
        np.zeros((6, 6)),
        np.zeros((4, 4), dtype=np.int32),
        np.zeros((6, 6), dtype=np.int32),
        np.asarray([], dtype=object),
        "zero ROI test",
    )
    assert Path(outputs["qc"]).is_file()
    assert Path(outputs["html"]).is_file()
    assert np.load(tmp_path / "roi_areas.npy").size == 0


def test_inspector_reads_standalone_resolved_configuration(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "status.json").write_text(
        json.dumps({"state": "failed", "error": "registration failed"}),
        encoding="utf-8",
    )
    expected = {
        "suite2p_version": "1.1.0",
        "db": {"nchannels": 2},
        "settings": {"registration": config.REGISTRATION},
    }
    (run_dir / "resolved_suite2p_configuration.json").write_text(
        json.dumps(expected), encoding="utf-8"
    )

    status, configuration = inspect_run.load_run(run_dir)

    assert status == {"state": "failed", "error": "registration failed"}
    assert configuration == expected
