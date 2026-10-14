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
import generate_trace_qc  # noqa: E402
import generate_full_session_qc  # noqa: E402
import prepare_full_session_pipeline  # noqa: E402
import run_full_session_pipeline  # noqa: E402
import run_roi_detection  # noqa: E402
import run_trace_extraction  # noqa: E402


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
    assert configured["detection"]["cellpose_settings"] == {
        "cellpose_model": "cpsam",
        "img": "meanImg",
        "flow_threshold": 0.4,
        "cellprob_threshold": 0.0,
    }

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


def test_trace_configuration_is_reproducible_and_skips_registration() -> None:
    settings = {
        "fs": 10.0,
        "tau": 1.0,
        "torch_device": "cpu",
        "run": {
            "do_registration": 2,
            "do_detection": False,
            "do_deconvolution": False,
        },
        "classification": {
            "classifier_path": "/tmp/user.npy",
            "use_builtin_classifier": False,
            "preclassify": 0.5,
        },
    }
    configured = run_trace_extraction.configure_settings(settings, "cuda")
    assert configured["fs"] == config.FS
    assert configured["tau"] == config.TAU
    assert configured["torch_device"] == "cuda"
    assert configured["run"] == {
        "do_registration": 0,
        "do_detection": False,
        "do_deconvolution": True,
    }
    assert configured["classification"] == {
        "classifier_path": None,
        "use_builtin_classifier": True,
        "preclassify": 0.0,
    }


def test_trace_loader_requires_completed_reviewed_detection(tmp_path: Path) -> None:
    detection_dir = tmp_path / "roi_detection" / "cpsam"
    detection_dir.mkdir(parents=True)
    stat = np.asarray(
        [{
            "ypix": np.array([1, 1]),
            "xpix": np.array([1, 2]),
            "lam": np.ones(2),
            "overlap": np.zeros(2, dtype=bool),
        }],
        dtype=object,
    )
    np.save(detection_dir / "stat.npy", stat)
    (detection_dir / "status.json").write_text(
        json.dumps({"state": "complete", "n_rois": 1}), encoding="utf-8"
    )

    loaded, status = run_trace_extraction.load_reviewed_detection(detection_dir)

    assert len(loaded) == 1
    assert status["state"] == "complete"


def test_trace_loader_accepts_plane_stat_path(tmp_path: Path) -> None:
    detection_dir = tmp_path / "suite2p/qc/detection"
    plane_dir = tmp_path / "suite2p/plane0"
    detection_dir.mkdir(parents=True)
    plane_dir.mkdir(parents=True)
    stat = np.asarray(
        [{
            "ypix": np.array([1, 1]),
            "xpix": np.array([1, 2]),
            "lam": np.ones(2),
            "overlap": np.zeros(2, dtype=bool),
        }],
        dtype=object,
    )
    np.save(plane_dir / "stat.npy", stat)
    (detection_dir / "status.json").write_text(
        json.dumps({"state": "complete", "n_rois": 1}), encoding="utf-8"
    )

    loaded, _ = run_trace_extraction.load_reviewed_detection(
        detection_dir, plane_dir / "stat.npy"
    )

    assert len(loaded) == 1


def test_trace_bundle_promotion_rolls_back_replaced_plane_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plane_dir = tmp_path / "suite2p/plane0"
    plane_dir.mkdir(parents=True)
    old_stat = np.asarray([{"old": True}], dtype=object)
    old_ops = {"registration": "old"}
    np.save(plane_dir / "stat.npy", old_stat)
    np.save(plane_dir / "ops.npy", old_ops)
    real_replace = run_trace_extraction.os.replace

    def fail_during_promotion(source: str | Path, destination: str | Path) -> None:
        destination = Path(destination)
        if destination == plane_dir / "Fneu.npy":
            raise OSError("simulated promotion failure")
        real_replace(source, destination)

    monkeypatch.setattr(run_trace_extraction.os, "replace", fail_during_promotion)
    with pytest.raises(OSError, match="simulated"):
        run_trace_extraction.promote_output_bundle(
            plane_dir,
            {
                "stat.npy": np.asarray([{"new": True}], dtype=object),
                "ops.npy": {"registration": "new"},
                "F.npy": np.ones((1, 2)),
                "Fneu.npy": np.ones((1, 2)),
            },
        )

    assert np.load(plane_dir / "stat.npy", allow_pickle=True)[0]["old"]
    assert np.load(plane_dir / "ops.npy", allow_pickle=True).item() == old_ops
    assert not (plane_dir / "F.npy").exists()


def test_full_pipeline_uses_standard_plane0_layout(tmp_path: Path) -> None:
    mouse = "sub-02"
    session = "ses-abab-random-001_date-20260824T120630"
    funcimg = tmp_path / "rawdata/cohort2" / mouse / session / "funcimg"
    funcimg.mkdir(parents=True)
    tifffile.imwrite(funcimg / "full.tif", np.zeros((2, 3), dtype=np.uint16))
    (funcimg / "._full.tif").write_bytes(b"metadata")

    row, paths = prepare_full_session_pipeline.build_pipeline(
        tmp_path, mouse, session
    )

    expected_run = tmp_path / "processed/cohort2/sub-02/abab-random-001"
    assert Path(str(row["run_dir"])) == expected_run
    assert Path(str(paths["plane_dir"])) == expected_run / "suite2p/plane0"
    assert Path(str(paths["qc_dir"])) == expected_run / "suite2p/qc"
    assert json.loads(str(row["tiff_files_json"])) == [str((funcimg / "full.tif").resolve())]

    with pytest.raises(ValueError, match="one directory name"):
        prepare_full_session_pipeline.build_pipeline(tmp_path, "../sub-02", session)


def test_full_pipeline_overwrite_removes_only_suite2p(tmp_path: Path) -> None:
    mouse = "sub-02"
    session = "ses-full002_date-20260818T134310"
    funcimg = tmp_path / "rawdata/cohort2" / mouse / session / "funcimg"
    funcimg.mkdir(parents=True)
    tifffile.imwrite(funcimg / "full.tif", np.zeros((2, 3), dtype=np.uint16))
    run_dir = tmp_path / "processed/cohort2/sub-02/full002"
    suite2p_dir = run_dir / "suite2p"
    suite2p_dir.mkdir(parents=True)
    (suite2p_dir / "partial.txt").write_text("partial", encoding="utf-8")
    sibling = run_dir / "behaviour.csv"
    sibling.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="pass --overwrite"):
        prepare_full_session_pipeline.build_pipeline(tmp_path, mouse, session)

    _, paths = prepare_full_session_pipeline.build_pipeline(
        tmp_path, mouse, session, overwrite=True
    )

    assert not suite2p_dir.exists()
    assert sibling.read_text(encoding="utf-8") == "keep"
    assert paths["overwrote_existing"] is True


def make_resumable_registration(
    tmp_path: Path, mouse: str, session: str
) -> tuple[Path, Path]:
    funcimg = tmp_path / "rawdata/cohort2" / mouse / session / "funcimg"
    funcimg.mkdir(parents=True)
    tiff = funcimg / "full.tif"
    tifffile.imwrite(tiff, np.zeros((2, 3), dtype=np.uint16))
    suite2p_dir = tmp_path / "processed/cohort2" / mouse / "full002" / "suite2p"
    plane_dir = suite2p_dir / "plane0"
    workflow_dir = suite2p_dir / ".workflow"
    plane_dir.mkdir(parents=True)
    workflow_dir.mkdir(parents=True)
    db = {"Ly": 3, "Lx": 4, "nframes": 5}
    np.save(plane_dir / "db.npy", db)
    np.save(
        plane_dir / "reg_outputs.npy",
        {
            "meanImg": np.ones((3, 4)),
            "badframes": np.zeros(5, dtype=bool),
            "yrange": [0, 3],
            "xrange": [0, 4],
        },
    )
    np.save(plane_dir / "ops.npy", {"registration": "complete"})
    for name in ("data.bin", "data_chan2.bin"):
        (plane_dir / name).write_bytes(bytes(3 * 4 * 5 * 2))
    snapshot = prepare_full_session_pipeline.file_snapshot([tiff.resolve()])
    (workflow_dir / "registration_status.json").write_text(
        json.dumps({"state": "complete", "source_snapshot_after": snapshot}),
        encoding="utf-8",
    )
    return suite2p_dir, tiff


def test_full_pipeline_resume_validates_and_preserves_registration(tmp_path: Path) -> None:
    mouse = "sub-02"
    session = "ses-full002_date-20260818T134310"
    suite2p_dir, _ = make_resumable_registration(tmp_path, mouse, session)
    plane_dir = suite2p_dir / "plane0"
    detection_dir = suite2p_dir / "qc" / "detection"
    detection_dir.mkdir(parents=True)
    (detection_dir / "status.json").write_text('{"state":"failed"}', encoding="utf-8")
    np.save(plane_dir / "stat.npy", np.asarray([], dtype=object))

    _, paths = prepare_full_session_pipeline.build_pipeline(
        tmp_path, mouse, session, resume=True
    )

    assert paths["resume"] is True
    assert paths["start_stage"] == "detection"
    # Preparation validates only; cleanup happens after the new Slurm job starts.
    assert (plane_dir / "data.bin").is_file()
    assert (plane_dir / "stat.npy").is_file()

    removed = prepare_full_session_pipeline.clean_downstream_outputs(suite2p_dir)
    assert str(plane_dir / "stat.npy") in removed
    assert not detection_dir.exists()
    assert (plane_dir / "data.bin").is_file()
    assert (plane_dir / "data_chan2.bin").is_file()
    assert (plane_dir / "db.npy").is_file()
    assert (plane_dir / "reg_outputs.npy").is_file()
    assert (plane_dir / "ops.npy").is_file()
    assert (suite2p_dir / ".workflow/registration_status.json").is_file()


def test_full_pipeline_resume_rejects_invalid_registration(tmp_path: Path) -> None:
    mouse = "sub-02"
    session = "ses-full002_date-20260818T134310"
    suite2p_dir, _ = make_resumable_registration(tmp_path, mouse, session)
    (suite2p_dir / "plane0/data.bin").write_bytes(b"short")

    with pytest.raises(RuntimeError, match="expected 120"):
        prepare_full_session_pipeline.build_pipeline(
            tmp_path, mouse, session, resume=True
        )


def test_full_pipeline_resume_requires_existing_registration(tmp_path: Path) -> None:
    mouse = "sub-02"
    session = "ses-full002_date-20260818T134310"
    funcimg = tmp_path / "rawdata/cohort2" / mouse / session / "funcimg"
    funcimg.mkdir(parents=True)
    tifffile.imwrite(funcimg / "full.tif", np.zeros((2, 3), dtype=np.uint16))

    with pytest.raises(FileNotFoundError, match="no existing Suite2p registration"):
        prepare_full_session_pipeline.build_pipeline(
            tmp_path, mouse, session, resume=True
        )


def test_full_pipeline_resume_and_overwrite_are_mutually_exclusive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        prepare_full_session_pipeline.build_pipeline(
            tmp_path, "sub-02", "session", resume=True, overwrite=True
        )


def test_full_pipeline_runner_resume_skips_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mouse = "sub-02"
    session = "ses-full002_date-20260818T134310"
    suite2p_dir, _ = make_resumable_registration(tmp_path, mouse, session)
    row, paths = prepare_full_session_pipeline.build_pipeline(
        tmp_path, mouse, session, resume=True
    )
    manifest = suite2p_dir / ".workflow/manifest.csv"
    prepare_full_session_pipeline.write_csv(manifest, [row])
    commands: list[list[str]] = []

    def fake_run(command: list[str], check: bool) -> None:
        assert check
        commands.append(command)

    monkeypatch.setattr(run_full_session_pipeline.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_full_session_pipeline.py",
            "--manifest",
            str(manifest),
            "--run-dir",
            str(paths["run_dir"]),
            "--script-dir",
            str(SCRIPT_DIR),
            "--python-executable",
            sys.executable,
            "--start-stage",
            "detection",
        ],
    )

    run_full_session_pipeline.main()

    invoked_scripts = [Path(command[1]).name for command in commands]
    assert invoked_scripts == [
        "run_roi_detection.py",
        "run_trace_extraction.py",
        "generate_full_session_qc.py",
    ]
    status = json.loads(
        (suite2p_dir / ".workflow/status.json").read_text(encoding="utf-8")
    )
    assert status["stages"]["registration"]["state"] == "reused"
    assert status["start_stage"] == "detection"


def test_full_pipeline_launcher_submits_one_job() -> None:
    launcher = SCRIPT_DIR / "submit_full_session_pipeline.sh"
    text = launcher.read_text(encoding="utf-8")
    assert text.count("sbatch --parsable") == 1
    assert "gpu-350-02,gpu-350-04" in text
    assert "gpu-580" not in text
    assert "--mail-type=BEGIN,END,FAIL" in text
    assert "--resume" in text
    assert '"${start_stage}"' in text


def test_combined_report_writes_html_png_and_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "processed/session"
    suite2p_dir = run_dir / "suite2p"
    plane_dir = suite2p_dir / "plane0"
    qc_dir = suite2p_dir / "qc"
    detection_dir = qc_dir / "detection"
    workflow_dir = suite2p_dir / ".workflow"
    trace_metadata = workflow_dir / "trace_extraction"
    for path in (plane_dir, detection_dir, trace_metadata):
        path.mkdir(parents=True, exist_ok=True)
    complete = {"state": "complete", "elapsed_seconds": 1.0}
    for path in (
        workflow_dir / "registration_status.json",
        detection_dir / "status.json",
        trace_metadata / "status.json",
    ):
        path.write_text(json.dumps(complete), encoding="utf-8")

    ops = {
        "Ly": 4,
        "Lx": 5,
        "fs": 45.0,
        "meanImg": np.arange(20, dtype=float).reshape(4, 5),
        "refImg": np.arange(20, dtype=float).reshape(4, 5),
        "corrXY": np.linspace(0.5, 0.9, 10),
        "yoff": np.linspace(-1, 1, 10),
        "xoff": np.linspace(1, -1, 10),
        "badframes": np.zeros(10, dtype=bool),
        "yoff1": np.zeros((10, 1)),
        "xoff1": np.zeros((10, 1)),
        "regDX": np.zeros((2, 4)),
        "registration": {"maxregshift": 0.1},
    }
    np.save(plane_dir / "ops.npy", ops)
    np.save(plane_dir / "iscell.npy", np.asarray([[1, 0.9], [0, 0.2]]))
    np.save(detection_dir / "roi_masks.npy", np.zeros((4, 5), dtype=np.int32))
    np.save(detection_dir / "roi_areas.npy", np.asarray([10.0, 12.0]))
    np.save(detection_dir / "roi_diameters.npy", np.asarray([3.5, 3.9]))

    def fake_trace_qc(command: list[str], check: bool) -> None:
        assert check
        output_dir = Path(command[command.index("--output-dir") + 1])
        output_dir.mkdir(parents=True)
        (output_dir / "trace_metrics.csv").write_text(
            "roi,iscell,green_red_correlation,ratio_red_floor_fraction,green_dff_robust_sigma\n"
            "0,1,0.5,0.1,0.2\n1,0,0.1,0.3,0.4\n",
            encoding="utf-8",
        )
        (output_dir / "calculation.json").write_text("{}", encoding="utf-8")
        for name in (
            "summary.png",
            "selected_cells_celltv.png",
            "selected_cells_dff.png",
        ):
            (output_dir / name).write_bytes(b"test")

    monkeypatch.setattr(generate_full_session_qc.subprocess, "run", fake_trace_qc)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_full_session_qc.py",
            "--run-dir",
            str(run_dir),
            "--script-dir",
            str(SCRIPT_DIR),
        ],
    )

    generate_full_session_qc.main()

    assert (qc_dir / "index.html").is_file()
    assert (qc_dir / "summary.png").is_file()
    assert (qc_dir / "session_metrics.csv").is_file()
    assert (qc_dir / "calculation.json").is_file()


def test_trace_output_validation_rejects_wrong_frame_count() -> None:
    stat = np.asarray([{}], dtype=object)
    trace = np.ones((1, 4))
    with pytest.raises(RuntimeError, match="expected.*5"):
        run_trace_extraction.validate_outputs(
            stat, trace, trace, trace, trace, trace, np.ones((1, 2)), 5
        )


def test_trace_binary_size_validation(tmp_path: Path) -> None:
    paths = [tmp_path / "data.bin", tmp_path / "data_chan2.bin"]
    for path in paths:
        path.write_bytes(bytes(3 * 4 * 5 * 2))
    assert run_trace_extraction.validate_registered_binary_sizes(
        {"Ly": 3, "Lx": 4, "nframes": 5}, paths
    ) == 120
    paths[1].write_bytes(b"too short")
    with pytest.raises(RuntimeError, match="expected 120"):
        run_trace_extraction.validate_registered_binary_sizes(
            {"Ly": 3, "Lx": 4, "nframes": 5}, paths
        )


def test_trace_qc_channel_calculation_matches_green_red_method() -> None:
    ops = {"fs": 8.0}
    F = np.asarray([[10.0, 12.0, 14.0, 16.0, 18.0]])
    Fneu = np.ones_like(F)
    F2 = np.asarray([[4.0, 5.0, 6.0, 7.0, 8.0]])
    Fneu2 = np.zeros_like(F2)

    result = generate_trace_qc.channel_signals(F, Fneu, F2, Fneu2, ops, 0.7)

    expected_green = F - 0.7 * Fneu
    expected_red = np.asarray([[4.0, 5.0, 6.0, 7.0, 7.0]])  # medfilt width 3
    expected_floor = max(np.percentile(expected_red, 5), 1e-3)
    assert np.allclose(result["green"], expected_green)
    assert np.allclose(result["red_smoothed"], expected_red)
    assert np.allclose(result["ratio"], expected_green / np.maximum(expected_red, expected_floor))


def test_trace_qc_normalizes_by_baseline() -> None:
    signal = np.asarray([[2.0, 4.0, 6.0]], dtype=np.float32)
    ops = {
        "fs": 1.0,
        "baseline": "constant_prctile",
        "prctile_baseline": 0.0,
    }

    dff, base, clipped = generate_trace_qc.normalized_dff(signal, ops)

    assert np.allclose(base, 2.0)
    assert np.allclose(dff, [[0.0, 1.0, 2.0]])
    assert np.allclose(signal - base, [[0.0, 2.0, 4.0]])
    assert not clipped.any()


def test_trace_qc_reads_nested_suite2p_1_1_settings() -> None:
    signal = np.asarray([[2.0, 4.0, 6.0]], dtype=np.float32)
    ops = {
        "fs": 1.0,
        "dcnv_preprocess": {
            "baseline": "constant_prctile",
            "prctile_baseline": 0.0,
            "win_baseline": 60.0,
            "sig_baseline": 10.0,
        },
    }

    dff, base, _ = generate_trace_qc.normalized_dff(signal, ops)

    assert np.allclose(base, 2.0)
    assert np.allclose(dff, [[0.0, 1.0, 2.0]])


def test_trace_qc_cell_selection_is_deterministic() -> None:
    iscell = np.asarray([[1, 0.8], [1, 0.9], [0, 0.99], [1, 0.9]])
    dff = np.asarray(
        [[0, 1, 0], [0, 1, 0], [0, 9, 0], [0, 2, 0]], dtype=float
    )
    assert np.array_equal(generate_trace_qc.select_rois(iscell, dff, 2), [3, 1])


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


def test_robust_display_limits_clip_outliers_and_handle_constant_images() -> None:
    image = np.concatenate([np.arange(100, dtype=float), np.asarray([10000.0])])
    vmin, vmax = run_detection_experiment.robust_display_limits(image)
    assert vmin >= 0
    assert vmax < 10000
    constant_min, constant_max = run_detection_experiment.robust_display_limits(
        np.ones((3, 3))
    )
    assert constant_min < 1 < constant_max


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
