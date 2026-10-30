from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "Convert_seg_to_stat.py"
SPEC = importlib.util.spec_from_file_location("convert_seg_to_stat", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    processed_root = tmp_path / "processed" / "cohort2"
    suite2p = (
        processed_root
        / "sub-02"
        / "ses-full002_date-20260818T134310"
        / "suite2p"
    )
    plane = suite2p / "plane0"
    registration = suite2p / "qc" / "registration"
    plane.mkdir(parents=True)
    registration.mkdir(parents=True)
    segmentation = registration / "meanImg_chan1_seg.npy"
    np.save(segmentation, {"masks": np.asarray([[0, 1], [2, 2]])})
    return processed_root, segmentation, plane


def test_full_session_name_resolves_to_current_layout(tmp_path: Path) -> None:
    processed_root, segmentation, plane = make_layout(tmp_path)
    assert MODULE.resolve_paths(
        "sub-02",
        "ses-full002_date-20260818T134310",
        project_root=tmp_path,
        processed_root=processed_root,
    ) == (segmentation.resolve(), plane.resolve())


def test_short_session_name_is_rejected(tmp_path: Path) -> None:
    processed_root, _, _ = make_layout(tmp_path)
    with pytest.raises(ValueError, match="full rawdata folder name"):
        MODULE.resolve_paths(
            "sub-02",
            "full002",
            project_root=tmp_path,
            processed_root=processed_root,
        )


def test_load_masks_and_make_raw_stat(tmp_path: Path) -> None:
    _, segmentation, _ = make_layout(tmp_path)
    masks = MODULE.load_masks(segmentation)
    stat = MODULE.raw_stat_from_masks(masks)
    assert len(stat) == 2
    assert stat[0]["npix"] == 1
    assert stat[1]["npix"] == 2
    np.testing.assert_array_equal(stat[1]["ypix"], [1, 1])
    np.testing.assert_array_equal(stat[1]["xpix"], [0, 1])


def test_resolve_paths_requires_manual_segmentation(tmp_path: Path) -> None:
    processed_root, segmentation, _ = make_layout(tmp_path)
    segmentation.unlink()
    with pytest.raises(FileNotFoundError, match="Manual Cellpose segmentation"):
        MODULE.resolve_paths(
            "sub-02",
            "ses-full002_date-20260818T134310",
            project_root=tmp_path,
            processed_root=processed_root,
        )


def test_meanimg_folder_supports_legacy_plane_layout(tmp_path: Path) -> None:
    processed_root, segmentation, plane = make_layout(tmp_path)
    segmentation.unlink()
    legacy_segmentation = plane / "meanImg_seg.npy"
    np.save(legacy_segmentation, {"masks": np.asarray([[0, 1], [2, 2]])})
    assert MODULE.resolve_paths(
        "sub-02",
        "ses-full002_date-20260818T134310",
        project_root=tmp_path,
        processed_root=processed_root,
        meanimg_folder=Path("plane0"),
    ) == (legacy_segmentation.resolve(), plane.resolve())


def test_load_masks_rejects_empty_segmentation(tmp_path: Path) -> None:
    path = tmp_path / "empty_seg.npy"
    np.save(path, {"masks": np.zeros((3, 3), dtype=np.int32)})
    with pytest.raises(ValueError, match="no labelled ROIs"):
        MODULE.load_masks(path)


def test_backup_and_promote_keeps_previous_bundle(tmp_path: Path) -> None:
    plane = tmp_path / "suite2p" / "plane0"
    plane.mkdir(parents=True)
    np.save(plane / "stat.npy", np.asarray(["old"]))
    backup = MODULE.backup_and_promote(
        plane,
        {"stat.npy": np.asarray(["new"])},
        tmp_path / "suite2p" / ".workflow" / "manual_segmentation_backups",
    )
    np.testing.assert_array_equal(np.load(plane / "stat.npy"), ["new"])
    np.testing.assert_array_equal(np.load(backup / "stat.npy"), ["old"])
