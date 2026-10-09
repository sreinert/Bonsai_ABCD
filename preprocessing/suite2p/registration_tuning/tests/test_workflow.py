from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile


WORKFLOW_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(WORKFLOW_DIR))

import discover_sessions  # noqa: E402
import evaluate  # noqa: E402
import make_tasks  # noqa: E402
import run_candidate  # noqa: E402
from common import remap_mounted_path  # noqa: E402


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


def test_relative_data_root_uses_available_projects_mount(tmp_path: Path) -> None:
    projects_root = tmp_path / "projects"
    relative = Path("project-name/rawdata/cohort2")
    expected = projects_root / relative
    expected.mkdir(parents=True)

    resolved = discover_sessions.resolve_data_root(
        relative,
        project_roots=(tmp_path / "missing", projects_root),
    )

    assert resolved == expected.resolve()


def test_manifest_path_remaps_between_mounts(tmp_path: Path) -> None:
    volumes_root = tmp_path / "Volumes" / "projects"
    ceph_root = tmp_path / "ceph" / "projects"
    destination = ceph_root / "project" / "sub-02" / "sample.tif"
    destination.parent.mkdir(parents=True)
    destination.touch()

    stored_path = volumes_root / "project" / "sub-02" / "sample.tif"
    remapped = remap_mounted_path(
        stored_path,
        must_exist=True,
        project_roots=(ceph_root, volumes_root),
    )

    assert remapped == destination.resolve()


def test_discovery_selects_most_recent_tiff_without_frame_requirement(
    tmp_path: Path,
) -> None:
    mouse = tmp_path / "sub-02"
    others = (
        mouse
        / "ses-abab-random-001_date-20260824T102425"
        / "funcimg"
        / "others"
    )
    others.mkdir(parents=True)
    older = others / "older.tif"
    newer = others / "newer.tif"
    derived = others / "suite2p" / "plane0" / "meanImg.tif"
    derived.parent.mkdir(parents=True)
    tifffile.imwrite(older, np.zeros((5, 8, 9), dtype=np.uint16), photometric="minisblack")
    tifffile.imwrite(newer, np.zeros((7, 8, 9), dtype=np.uint16), photometric="minisblack")
    tifffile.imwrite(derived, np.zeros((8, 9), dtype=np.uint16), photometric="minisblack")
    os.utime(older, (1000, 1000))
    os.utime(newer, (2000, 2000))
    os.utime(derived, (3000, 3000))

    sessions, audit = discover_sessions.discover_mouse(
        mouse,
        "ses-*",
        Path("funcimg/others"),
        expected_frames=None,
        count_frames=True,
    )

    assert len(sessions) == 1
    assert sessions[0].tiff_files == (newer.resolve(),)
    assert sessions[0].frame_count == 7
    assert audit[0]["tiff_file_count"] == 2
    assert audit[0]["selected_tiff"] == str(newer.resolve())


def test_discovery_does_not_need_to_count_network_tiff_pages(tmp_path: Path) -> None:
    mouse = tmp_path / "sub-02"
    others = mouse / "ses-001_date-20260824T102425" / "funcimg" / "others"
    others.mkdir(parents=True)
    stack = others / "stack.tif"
    tifffile.imwrite(stack, np.zeros((7, 8, 9), dtype=np.uint16), photometric="minisblack")

    sessions, audit = discover_sessions.discover_mouse(
        mouse,
        "ses-*",
        Path("funcimg/others"),
        expected_frames=None,
    )

    assert len(sessions) == 1
    assert sessions[0].frame_count is None
    assert audit[0]["eligible"] is True
    assert audit[0]["frame_count_status"] == "not_counted"


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
            "nchannels": "2",
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
            "nchannels": "2",
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


def test_resource_metrics_convert_bytes_to_gib() -> None:
    metrics = evaluate.resource_metrics(
        {
            "resource_usage": {
                "peak_python_rss_bytes": 3 * 1024**3,
                "peak_cuda_allocated_bytes": 2 * 1024**3,
                "peak_cuda_reserved_bytes": 4 * 1024**3,
            }
        }
    )
    assert metrics == {
        "peak_python_rss_gib": 3.0,
        "peak_cuda_allocated_gib": 2.0,
        "peak_cuda_reserved_gib": 4.0,
    }


def test_cleanup_retains_registered_and_removes_only_generated_raw_binaries(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    plane_dir = run_dir / "suite2p" / "plane0"
    plane_dir.mkdir(parents=True)
    frames = np.arange(3 * 4 * 5, dtype=np.int16).reshape(3, 4, 5)
    (plane_dir / "data.bin").write_bytes(frames.tobytes())
    (plane_dir / "data_chan2.bin").write_bytes((frames + 10).tobytes())
    (plane_dir / "data_raw.bin").write_bytes(frames.tobytes())
    (plane_dir / "data_raw_chan2.bin").write_bytes((frames + 10).tobytes())
    np.save(
        plane_dir / "db.npy",
        {"nframes": 3, "Ly": 4, "Lx": 5, "nchannels": 2},
    )
    ops = {
        "meanImg": frames.mean(axis=0),
        "meanImg_chan2": (frames + 10).mean(axis=0),
    }
    ops_path = plane_dir / "ops.npy"
    np.save(ops_path, ops)

    exports = run_candidate.export_registered_montages([ops_path])
    retained = run_candidate.validate_registered_binaries(run_dir, require_chan2=True)
    removed = run_candidate.remove_generated_raw_binaries(run_dir)

    assert {path.name for path in exports} == {
        "registered_frames.png",
        "registered_frames_chan2.png",
    }
    assert all(path.is_file() for path in exports)
    assert {Path(item["path"]).name for item in retained} == {
        "data.bin",
        "data_chan2.bin",
    }
    assert {Path(item["path"]).name for item in removed} == {
        "data_raw.bin",
        "data_raw_chan2.bin",
    }
    assert (plane_dir / "data.bin").is_file()
    assert (plane_dir / "data_chan2.bin").is_file()
    assert not (plane_dir / "data_raw.bin").exists()
    assert not (plane_dir / "data_raw_chan2.bin").exists()
    assert evaluate.make_registered_montage(ops_path, ops, ops) == exports[0]


def test_registration_configuration_inherits_suite2p_defaults(monkeypatch, tmp_path: Path) -> None:
    registration_defaults = {
        "align_by_chan2": False,
        "smooth_sigma": 1.15,
        "smooth_sigma_time": 0,
        "norm_frames": True,
        "nimg_init": 400,
        "do_bidiphase": False,
        "bidiphase": 0.0,
        "nonrigid": True,
        "block_size": (128, 128),
        "two_step_registration": False,
        "maxregshift": 0.1,
        "maxregshiftNR": 5,
        "snr_thresh": 1.2,
        "batch_size": 100,
    }
    defaults = {
        "torch_device": "cpu",
        "fs": 30.0,
        "tau": 1.0,
        "run": {
            "do_registration": 1,
            "do_regmetrics": True,
            "do_detection": True,
            "do_deconvolution": True,
        },
        "io": {"delete_bin": False, "move_bin": False, "save_ops_orig": True},
        "registration": registration_defaults,
    }
    fake_suite2p = SimpleNamespace(default_settings=lambda: deepcopy(defaults))
    monkeypatch.setitem(sys.modules, "suite2p", fake_suite2p)
    input_path = tmp_path / "raw" / "funcimg"
    input_path.mkdir(parents=True)
    tiff = input_path / "movie.tif"
    tiff.touch()
    row = {
        "fs": "45",
        "tau": "0.4",
        "nplanes": "1",
        "nchannels": "2",
        "frame_count": "",
        "registration_json": json.dumps(
            {
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
            }
        ),
    }

    db, settings = run_candidate.build_suite2p_configuration(
        row, input_path, tmp_path / "processed", [tiff], "cuda"
    )

    assert settings["registration"]["align_by_chan2"] is True
    assert settings["registration"]["smooth_sigma"] == 3.0
    assert settings["registration"]["maxregshift"] == 0.1
    assert settings["registration"]["maxregshiftNR"] == 5
    assert settings["registration"]["snr_thresh"] == 1.2
    assert settings["registration"]["batch_size"] == 100
    assert settings["run"]["do_detection"] is False
    assert settings["run"]["do_deconvolution"] is False
    assert db["functional_chan"] == 1
    assert db["keep_movie_raw"] is True
