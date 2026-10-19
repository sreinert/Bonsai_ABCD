from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).parents[1] / "create_session_inventory.py"
SPEC = importlib.util.spec_from_file_location("create_session_inventory", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
inventory = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = inventory
SPEC.loader.exec_module(inventory)


def make_session(
    root: Path, mouse: str, name: str, *, with_tiff: bool = True
) -> Path:
    session = root / mouse / name
    (session / "funcimg").mkdir(parents=True)
    if with_tiff:
        (session / "funcimg" / "movie.tif").write_bytes(b"test TIFF placeholder")
    return session


def save_complete_suite2p(processed_session: Path) -> None:
    plane = processed_session / "suite2p" / "plane0"
    plane.mkdir(parents=True)
    np.save(plane / "ops.npy", {"nframes": 10})
    (plane / "data.bin").write_bytes(b"registered")
    np.save(plane / "stat.npy", np.asarray([{"id": 0}, {"id": 1}], dtype=object))
    np.save(plane / "F.npy", np.ones((2, 4)))
    np.save(plane / "Fneu.npy", np.ones((2, 4)))
    np.save(plane / "spks.npy", np.ones((2, 4)))
    np.save(plane / "iscell.npy", np.asarray([[1, 0.9], [0, 0.1]]))


def test_inventory_filters_sessions_and_reports_complete_pipeline(tmp_path: Path) -> None:
    raw_root = tmp_path / "rawdata" / "cohort2"
    processed_root = tmp_path / "processed" / "cohort2"
    session = make_session(
        raw_root, "sub-02", "ses-full002_date-20260818T134310"
    )
    save_complete_suite2p(processed_root / "sub-02" / "full002")
    (raw_root / "sub-02" / ".hidden" / "funcimg").mkdir(parents=True)
    (raw_root / "sub-02" / "not-imaging").mkdir()

    rows = inventory.inventory_mouse(raw_root, processed_root, "sub-02")

    assert rows == [
        {
            "mouse_id": "sub-02",
            "session_full_name": "ses-full002_date-20260818T134310",
            "session_short_name": "full002",
            "date": "2026-08-18",
            "tiff_files_present": "yes",
            "suite2p_path_created": "yes",
            "registration": "yes",
            "registration_status": "succeeded",
            "detection": "yes",
            "trace_extraction": "yes",
            "number_of_rois": 2,
            "number_of_cells": 1,
        }
    ]


def test_inventory_reports_partial_pipeline(tmp_path: Path) -> None:
    raw_root = tmp_path / "rawdata" / "cohort2"
    processed_root = tmp_path / "processed" / "cohort2"
    make_session(raw_root, "sub-03", "ses-training_date-20261001")
    plane = processed_root / "sub-03" / "training" / "suite2p" / "plane0"
    plane.mkdir(parents=True)
    np.save(plane / "ops.npy", {})
    (plane / "data.bin").write_bytes(b"registered")
    np.save(plane / "stat.npy", np.asarray([], dtype=object))

    [row] = inventory.inventory_mouse(raw_root, processed_root, "sub-03")

    assert row["registration"] == "yes"
    assert row["suite2p_path_created"] == "yes"
    assert row["registration_status"] == "succeeded"
    assert row["detection"] == "yes"
    assert row["trace_extraction"] == "no"
    assert row["number_of_rois"] == 0
    assert row["number_of_cells"] == ""


def test_write_inventory_writes_headers_even_with_no_rows(tmp_path: Path) -> None:
    output = tmp_path / "inventory.csv"

    inventory.write_inventory(output, [], overwrite=False)

    with output.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(inventory.FIELDNAMES)
        assert list(reader) == []


def test_suite2p_falls_back_to_legacy_funcimg_location(tmp_path: Path) -> None:
    raw_root = tmp_path / "rawdata" / "cohort2"
    processed_root = tmp_path / "processed" / "cohort2"
    session = make_session(raw_root, "sub-04", "ses-legacy_date-20261002")
    save_complete_suite2p(session / "funcimg")

    [row] = inventory.inventory_mouse(raw_root, processed_root, "sub-04")

    assert row["registration"] == "yes"
    assert row["number_of_rois"] == 2


def test_created_suite2p_path_with_incomplete_registration_is_failed(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "rawdata" / "cohort2"
    processed_root = tmp_path / "processed" / "cohort2"
    make_session(raw_root, "sub-05", "ses-failed_date-20261003")
    (processed_root / "sub-05" / "failed" / "suite2p").mkdir(parents=True)

    [row] = inventory.inventory_mouse(raw_root, processed_root, "sub-05")

    assert row["suite2p_path_created"] == "yes"
    assert row["registration"] == "no"
    assert row["registration_status"] == "failed"


def test_missing_suite2p_path_means_registration_not_started(tmp_path: Path) -> None:
    raw_root = tmp_path / "rawdata" / "cohort2"
    processed_root = tmp_path / "processed" / "cohort2"
    make_session(raw_root, "sub-06", "ses-new_date-20261004")

    [row] = inventory.inventory_mouse(raw_root, processed_root, "sub-06")

    assert row["suite2p_path_created"] == "no"
    assert row["registration"] == "no"
    assert row["registration_status"] == "not_started"


def test_empty_funcimg_is_included_and_reports_no_tiff(tmp_path: Path) -> None:
    raw_root = tmp_path / "rawdata" / "cohort2"
    processed_root = tmp_path / "processed" / "cohort2"
    make_session(
        raw_root,
        "sub-07",
        "ses-empty_date-20261005",
        with_tiff=False,
    )

    [row] = inventory.inventory_mouse(raw_root, processed_root, "sub-07")

    assert row["tiff_files_present"] == "no"


def test_default_output_is_inside_processed_mouse_folder(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed" / "cohort2"

    output = inventory.mouse_output_path(processed_root, "sub-02", None)

    assert output == (
        processed_root / "sub-02" / "suite2p_session_inventory.csv"
    )
