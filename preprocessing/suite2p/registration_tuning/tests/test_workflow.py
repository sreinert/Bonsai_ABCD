from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile


WORKFLOW_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(WORKFLOW_DIR))

import discover_sessions  # noqa: E402
import evaluate  # noqa: E402
import make_tasks  # noqa: E402


def make_session(mouse: str, session: str, order: int) -> discover_sessions.Session:
    root = Path("/data") / mouse / session
    return discover_sessions.Session(
        mouse_id=mouse,
        session_id=session,
        session_path=root,
        input_path=root / "funcimg" / "others",
        tiff_files=(root / "funcimg" / "others" / "sample.tif",),
        frame_count=2000,
        date=None,
        session_number=order,
    )


def test_tiff_frame_count_uses_metadata(tmp_path: Path) -> None:
    path = tmp_path / "stack.tif"
    tifffile.imwrite(path, np.zeros((7, 8, 9), dtype=np.uint16), photometric="minisblack")
    assert discover_sessions.tiff_frame_count(path) == 7


def test_timestamped_sequence_compression_session_date() -> None:
    parsed = discover_sessions.parse_date(
        "ses-abab-random-001_date-20260824T102425"
    )
    assert parsed is not None
    assert parsed.isoformat() == "2026-08-24T10:24:25"


def test_session_selection_is_seeded_and_keeps_endpoints() -> None:
    sessions = [make_session("m1", f"ses-{index:03d}", index) for index in range(1, 8)]
    first = discover_sessions.choose_sessions(sessions, seed=123)
    second = discover_sessions.choose_sessions(sessions, seed=123)
    assert first == second
    assert [role for role, _ in first] == ["first", "middle", "last"]
    assert first[0][1] == sessions[0]
    assert first[-1][1] == sessions[-1]
    assert first[1][1] in sessions[1:-1]


def test_stable_mouse_seed_differs_by_mouse() -> None:
    assert discover_sessions.stable_mouse_seed(10, "mouse-a") == discover_sessions.stable_mouse_seed(10, "mouse-a")
    assert discover_sessions.stable_mouse_seed(10, "mouse-a") != discover_sessions.stable_mouse_seed(10, "mouse-b")


def test_task_builder_rejects_output_inside_input(tmp_path: Path) -> None:
    input_path = tmp_path / "mouse" / "session" / "funcimg" / "others"
    input_path.mkdir(parents=True)
    sessions = [
        {
            "mouse_id": "mouse",
            "session_id": "session",
            "selection_role": "only",
            "input_path": str(input_path),
            "tiff_files_json": json.dumps([str(input_path / "sample.tif")]),
            "fs": "45",
            "tau": "0.4",
            "nplanes": "1",
            "nchannels": "1",
        }
    ]
    candidates = [{"name": "rigid", "registration": {"nonrigid": False}}]
    with pytest.raises(ValueError, match="overlaps raw input"):
        make_tasks.build_tasks(sessions, candidates, "1.1.0", input_path / "results")


def test_task_builder_is_cartesian_and_deterministic(tmp_path: Path) -> None:
    inputs = [tmp_path / "inputs" / name for name in ("s1", "s2")]
    for path in inputs:
        path.mkdir(parents=True)
    sessions = [
        {
            "mouse_id": "m1",
            "session_id": path.name,
            "selection_role": "first",
            "input_path": str(path),
            "tiff_files_json": json.dumps([str(path / "sample.tif")]),
            "fs": "45",
            "tau": "0.4",
            "nplanes": "1",
            "nchannels": "1",
        }
        for path in inputs
    ]
    candidates = [
        {"name": "rigid", "registration": {"nonrigid": False}},
        {"name": "nonrigid", "registration": {"nonrigid": True}},
    ]
    output = tmp_path / "outputs"
    first = make_tasks.build_tasks(sessions, candidates, "1.1.0", output)
    second = make_tasks.build_tasks(sessions, candidates, "1.1.0", output)
    assert first == second
    assert len(first) == 4
    assert [row["task_id"] for row in first] == [0, 1, 2, 3]


def test_metrics_detect_bad_frames_and_boundary_hits() -> None:
    ops = {"Ly": 100, "Lx": 100, "maxregshift": 0.1}
    outputs = {
        "meanImg": np.eye(100),
        "yoff": np.array([0, 10, 0]),
        "xoff": np.array([0, 0, 10]),
        "corrXY": np.array([0.8, 0.9, 0.7]),
        "badframes": np.array([False, True, False]),
    }
    metrics = evaluate.compute_metrics(ops, outputs)
    assert metrics["boundary_hit_fraction"] == pytest.approx(2 / 3)
    assert metrics["bad_frame_fraction"] == pytest.approx(1 / 3)
