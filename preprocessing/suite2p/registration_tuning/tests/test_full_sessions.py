from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tifffile


WORKFLOW_DIR = Path(__file__).parents[1]
if WORKFLOW_DIR.name != "registration_tuning":
    WORKFLOW_DIR = Path(__file__).parent
sys.path.insert(0, str(WORKFLOW_DIR))

import make_full_session_tasks as workflow  # noqa: E402


def write_session(root: Path, mouse: str, session: str) -> Path:
    funcimg = root / "rawdata" / "cohort2" / mouse / session / "funcimg"
    funcimg.mkdir(parents=True)
    tifffile.imwrite(
        funcimg / "full.tif",
        np.zeros((4, 8, 9), dtype=np.uint16),
        photometric="minisblack",
    )
    others = funcimg / "others"
    others.mkdir()
    tifffile.imwrite(
        others / "subset.tif",
        np.zeros((2, 8, 9), dtype=np.uint16),
        photometric="minisblack",
    )
    return funcimg


def test_processed_session_id() -> None:
    assert (
        workflow.processed_session_id(
            "ses-abab-random-001_date-20260824T120630"
        )
        == "abab-random-001"
    )


def test_builds_first_last_full_session_tasks(tmp_path: Path) -> None:
    first = write_session(
        tmp_path, "sub-02", "ses-abab-random-001_date-20260824T120630"
    )
    write_session(tmp_path, "sub-02", "ses-abab-random-002_date-20260825T120630")
    last = write_session(
        tmp_path, "sub-02", "ses-abab-random-003_date-20260826T120630"
    )

    rows = workflow.build_tasks(
        tmp_path / "rawdata" / "cohort2",
        tmp_path / "processed" / "cohort2",
    )

    assert [row["selection_role"] for row in rows] == ["first", "last"]
    assert [row["run_dir"] for row in rows] == [
        str((tmp_path / "processed/cohort2/sub-02/abab-random-001").resolve()),
        str((tmp_path / "processed/cohort2/sub-02/abab-random-003").resolve()),
    ]
    assert json.loads(rows[0]["tiff_files_json"]) == [
        str((first / "full.tif").resolve())
    ]
    assert json.loads(rows[1]["tiff_files_json"]) == [
        str((last / "full.tif").resolve())
    ]
    registration = json.loads(rows[0]["registration_json"])
    assert registration["align_by_chan2"] is True
    assert registration["smooth_sigma"] == 3.0
    assert rows[0]["nchannels"] == 2
