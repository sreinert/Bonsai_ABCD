from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import pytest


WORKFLOW_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(WORKFLOW_DIR))

import cli  # noqa: E402
import execution  # noqa: E402
import session_inputs  # noqa: E402
import tuning_config  # noqa: E402


def write_tiff_header(path: Path) -> None:
    path.write_bytes(b"II*\x00test-data")


def config_payload() -> dict[str, object]:
    return {
        "suite2p_version": "1.1.0",
        "fixed_registration": {
            "nonrigid": True,
            "two_step_registration": True,
            "smooth_sigma_time": 0,
            "do_bidiphase": False,
            "bidiphase": 0,
        },
        "parameter_grid": {
            "smooth_sigma": [1.0, 2.0],
            "maxregshift": [0.1, 0.2, 0.3],
        },
        "presets": {"baseline": {}, "chosen": {"smooth_sigma": 3.0}},
    }


def make_session(tmp_path: Path, mouse: str, session: str) -> Path:
    result = tmp_path / mouse / session
    (result / "funcimg" / "others").mkdir(parents=True)
    return result


def test_cartesian_grid_uses_config_order_and_selected_keys() -> None:
    candidates = tuning_config.build_sweep_candidates(
        config_payload(), ["maxregshift", "smooth_sigma"]
    )
    assert len(candidates) == 6
    assert [candidate["name"] for candidate in candidates[:3]] == [
        "smooth_sigma=1.0__maxregshift=0.1",
        "smooth_sigma=1.0__maxregshift=0.2",
        "smooth_sigma=1.0__maxregshift=0.3",
    ]
    assert candidates[-1]["registration"]["smooth_sigma"] == 2.0
    assert candidates[-1]["registration"]["maxregshift"] == 0.3


def test_named_preset_merges_fixed_registration() -> None:
    candidate = tuning_config.build_preset_candidate(config_payload(), "chosen")
    assert candidate["name"] == "chosen"
    assert candidate["registration"]["nonrigid"] is True
    assert candidate["registration"]["smooth_sigma"] == 3.0


def test_sweep_rejects_unknown_and_repeated_parameters() -> None:
    with pytest.raises(ValueError, match="absent"):
        tuning_config.build_sweep_candidates(config_payload(), ["unknown"])
    with pytest.raises(ValueError, match="repeated"):
        tuning_config.build_sweep_candidates(
            config_payload(), ["smooth_sigma", "smooth_sigma"]
        )


def test_small_input_selects_newest_visible_tiff_without_counting(
    tmp_path: Path,
) -> None:
    session = make_session(tmp_path, "mouse-a", "session-1")
    others = session / "funcimg" / "others"
    older = others / "older.tif"
    newer = others / "newer.tif"
    hidden = others / "._newest.tif"
    for path in (older, newer, hidden):
        write_tiff_header(path)
    os.utime(older, (1000, 1000))
    os.utime(newer, (2000, 2000))
    os.utime(hidden, (3000, 3000))

    selected = session_inputs.discover_session_input(session, "small")

    assert selected.tiff_files == (newer.resolve(),)
    assert selected.expected_timepoints == 2_000
    assert selected.expected_raw_pages == 4_000
    assert selected.manifest_row()["actual_raw_pages"] == ""


def test_full_input_prefers_all_cropped_tiffs_in_natural_order(tmp_path: Path) -> None:
    session = make_session(tmp_path, "mouse-a", "session-1")
    funcimg = session / "funcimg"
    for name in ("movie.tif", "part10_cropped.tif", "part2_cropped.tif"):
        write_tiff_header(funcimg / name)

    selected = session_inputs.discover_session_input(session, "full")

    assert [path.name for path in selected.tiff_files] == [
        "part2_cropped.tif",
        "part10_cropped.tif",
    ]
    assert selected.input_selection == "cropped"
    assert selected.expected_timepoints == 81_000
    assert selected.expected_raw_pages == 162_000


def test_input_selection_rejects_missing_directory_and_invalid_header(
    tmp_path: Path,
) -> None:
    with pytest.raises(NotADirectoryError, match="does not exist"):
        session_inputs.discover_session_input(tmp_path / "missing", "small")

    session = make_session(tmp_path, "mouse-a", "session-1")
    (session / "funcimg" / "others" / "broken.tif").write_bytes(b"not-a-tiff")
    with pytest.raises(ValueError, match="Unrecognised TIFF header"):
        session_inputs.discover_session_input(session, "small")


def test_prepare_sweep_writes_cartesian_manifest_and_reuses_it(tmp_path: Path) -> None:
    session = make_session(tmp_path, "mouse-a", "session-1")
    write_tiff_header(session / "funcimg" / "others" / "stack.tif")
    config = tmp_path / "config.json"
    config.write_text(json.dumps(config_payload()), encoding="utf-8")
    output = tmp_path / "experiment"
    args = argparse.Namespace(
        mode="sweep",
        session_dir=[session],
        input_kind="small",
        config=config,
        output_root=output,
        execution="direct",
        device="cpu",
        parameters=["smooth_sigma", "maxregshift"],
    )

    manifest, count, _ = cli.prepare_experiment(args)
    second_manifest, second_count, _ = cli.prepare_experiment(args)

    assert manifest == second_manifest
    assert count == second_count == 6
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["input_kind"] == "small"
    assert rows[0]["expected_timepoints"] == "2000"
    assert rows[0]["expected_raw_pages"] == "4000"
    assert rows[0]["actual_raw_pages"] == ""
    assert [row["task_id"] for row in rows] == [str(i) for i in range(6)]


def test_single_mode_requires_exactly_one_session(tmp_path: Path) -> None:
    args = argparse.Namespace(
        mode="single",
        session_dir=[tmp_path / "one", tmp_path / "two"],
    )
    with pytest.raises(ValueError, match="exactly one"):
        cli.prepare_experiment(args)


def test_prepare_single_writes_one_preset_task(tmp_path: Path) -> None:
    session = make_session(tmp_path, "mouse-a", "session-1")
    write_tiff_header(session / "funcimg" / "others" / "stack.tif")
    config = tmp_path / "config.json"
    config.write_text(json.dumps(config_payload()), encoding="utf-8")
    args = argparse.Namespace(
        mode="single",
        session_dir=[session],
        input_kind="small",
        config=config,
        output_root=tmp_path / "experiment",
        execution="direct",
        device="cpu",
        preset="chosen",
    )

    manifest, count, _ = cli.prepare_experiment(args)

    assert count == 1
    with manifest.open(newline="", encoding="utf-8") as handle:
        [row] = list(csv.DictReader(handle))
    assert row["candidate_name"] == "chosen"
    assert json.loads(row["registration_json"])["smooth_sigma"] == 3.0


def test_existing_manifest_must_match(tmp_path: Path) -> None:
    path = tmp_path / "tasks.csv"
    execution.write_or_validate_manifest(path, [{"task_id": 0, "name": "first"}])
    with pytest.raises(RuntimeError, match="does not match"):
        execution.write_or_validate_manifest(
            path, [{"task_id": 0, "name": "changed"}]
        )


def test_output_root_cannot_contain_raw_input(tmp_path: Path) -> None:
    session = make_session(tmp_path, "mouse-a", "session-1")
    write_tiff_header(session / "funcimg" / "others" / "stack.tif")
    config = tmp_path / "config.json"
    config.write_text(json.dumps(config_payload()), encoding="utf-8")
    args = argparse.Namespace(
        mode="single",
        session_dir=[session],
        input_kind="small",
        config=config,
        output_root=session,
        execution="direct",
        device="cpu",
        preset="baseline",
    )

    with pytest.raises(ValueError, match="overlaps raw input"):
        cli.prepare_experiment(args)


def test_slurm_submission_uses_input_specific_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []
    python = tmp_path / "python"
    python.touch()
    monkeypatch.setenv("PYTHON_EXECUTABLE", str(python))
    monkeypatch.delenv("TASK_RANGE", raising=False)
    monkeypatch.delenv("MAX_CONCURRENT", raising=False)
    monkeypatch.delenv("SLURM_TIME", raising=False)
    monkeypatch.setattr(
        execution.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command),
    )

    execution.submit_slurm(
        tmp_path / "tasks.csv",
        9,
        tmp_path,
        tmp_path / "runner.sbatch",
        tmp_path / "repo",
        input_kind="full",
    )

    sbatch = calls[-1]
    assert "--array=0-8%4" in sbatch
    assert "--time=12:00:00" in sbatch


def test_direct_execution_runs_each_task_with_separate_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []
    python = tmp_path / "python"
    python.touch()
    monkeypatch.setenv("PYTHON_EXECUTABLE", str(python))
    monkeypatch.setattr(
        execution.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command),
    )

    execution.run_direct(
        tmp_path / "tasks.csv",
        2,
        tmp_path,
        tmp_path / "runner.py",
        device="cpu",
    )

    assert len(calls) == 2
    assert calls[0][-3:] == ["0", "--device", "cpu"]
    assert calls[1][-3:] == ["1", "--device", "cpu"]
    assert (tmp_path / "logs" / "direct_task_0.out").is_file()
    assert (tmp_path / "logs" / "direct_task_1.err").is_file()
