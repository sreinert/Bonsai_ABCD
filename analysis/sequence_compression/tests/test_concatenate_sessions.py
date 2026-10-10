import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from analysis.sequence_compression.concatenate_sessions import (
    CHUNK_STAMP,
    ConcatenationError,
    concatenate_sessions,
)
from analysis.sequence_compression.session_functions.io import (
    load_analog_data,
    load_data,
    load_settings,
)


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _make_session(
    parent: Path,
    timestamp: str,
    *,
    settings_marker: str = "same",
) -> Path:
    session = parent / f"ses-abab-random-001_date-{timestamp}"
    behav = session / "behav"

    _write_csv(
        behav / "analog-data" / f"analog-data_{CHUNK_STAMP}.csv",
        ["Seconds", "BufferIndex"],
        [[0.1, 0], [1.1, 1]],
    )
    analog = np.arange(8, dtype=np.float64).reshape(8, 1)
    analog.tofile(behav / "analog-data" / f"analog-data_{CHUNK_STAMP}.bin")
    _write_csv(
        behav / "current-position" / f"current-position_{CHUNK_STAMP}.csv",
        [
            "Seconds",
            "Value.X",
            "Value.Y",
            "Value.Z",
            "Value.Length",
            "Value.LengthFast",
            "Value.LengthSquared",
        ],
        [[0.2, 0, 0, 0, 0, 0, 0], [1.0, 0, 0, 1, 1, 1, 1]],
    )
    _write_csv(
        behav / "current-landmark" / f"current-landmark_{CHUNK_STAMP}.csv",
        [
            "Seconds",
            "Value.Index",
            "Value.Value.Landmark.Size",
            "Value.Value.Landmark",
            "Value.Value.Landmark",
            "Value.Value.Landmark.RewardSequencePosition",
            "Value.Value.Position",
            "Value.Value.Visited",
            "Value.Value.RewardDelivered",
            "Value.Value.IsGap",
            "Value.Value.IgnoreInBoundaryCalculations",
        ],
        [[0.5, 0, 3, "logs", "odour1", 0, 0.5, False, False, False, False]],
    )
    _write_csv(
        behav / "experiment-events" / f"experiment-events_{CHUNK_STAMP}.csv",
        ["Seconds", "Value"],
        [[0.3, "release: odour1"]],
    )
    _write_csv(
        behav / "licks" / f"licks_{CHUNK_STAMP}.csv",
        ["Seconds", "Value"],
        [[0.1, False], [1.0, True]],
    )
    _write_csv(
        behav / "reward" / f"reward_{CHUNK_STAMP}.csv",
        ["Seconds", "Value"],
        [[0.9, "Water"]],
    )
    _write_csv(
        behav / "treadmill-speed" / f"treadmill-speed_{CHUNK_STAMP}.csv",
        ["Seconds", "Value"],
        [[0.1, 0], [1.0, 2]],
    )

    settings = {
        "value": {
            "marker": settings_marker,
            "velocityThreshold": 1,
            "trial": {"landmarks": [[{"size": 3, "rewardSequencePosition": 0}]]},
        }
    }
    rig = {"value": {"analogInputChannels": [{"alias": "rewards"}]}}
    for directory, filename, document in (
        ("session-settings", "session-settings.json", settings),
        ("rig-settings", "rig-settings.json", rig),
    ):
        folder = behav / directory
        folder.mkdir(parents=True)
        (folder / filename).write_text(json.dumps(document), encoding="utf-8")

    camera = behav / "camera-face"
    camera.mkdir()
    (camera / "clip.avi").write_bytes(b"not a real video")
    return session


def test_concatenates_restarted_sessions_and_existing_loaders_can_read_it(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "cohort2" / "source"
    first = _make_session(source_root, "20260827T140000")
    second = _make_session(source_root, "20260827T150000")
    output_root = tmp_path / "cohort2" / "combined"

    output = concatenate_sessions(
        [second, first],
        output_root=output_root,
        creation_time=datetime(2026, 10, 9, 15, 30, 12),
    )

    assert output.name == "ses-abab-random-001_date-20261009T153012"
    assert (output / "original_sessions" / first.name).is_dir()
    assert (output / "original_sessions" / second.name).is_dir()
    assert (
        output / "original_sessions" / first.name / "behav" / "camera-face" / "clip.avi"
    ).read_bytes() == b"not a real video"
    assert not (output / "behav" / "camera-face").exists()

    session_settings, rig_settings = load_settings(output)
    dataframe = load_data(output)
    analog = load_analog_data(output, rig_settings)
    assert session_settings["marker"] == "same"
    assert not dataframe.empty
    assert len(analog) == 16

    analog_csv = output / "behav" / "analog-data" / f"analog-data_{CHUNK_STAMP}.csv"
    with analog_csv.open(newline="", encoding="utf-8") as handle:
        analog_rows = list(csv.DictReader(handle))
    assert [float(row["BufferIndex"]) for row in analog_rows] == [0, 1, 2, 3]
    assert [float(row["Seconds"]) for row in analog_rows] == pytest.approx(
        [0.1, 1.1, 1.100001, 2.100001]
    )

    position_csv = (
        output
        / "behav"
        / "current-position"
        / f"current-position_{CHUNK_STAMP}.csv"
    )
    with position_csv.open(newline="", encoding="utf-8") as handle:
        position_rows = list(csv.DictReader(handle))
    assert [float(row["Value.Length"]) for row in position_rows] == [0, 1, 1, 2]

    landmark_csv = (
        output
        / "behav"
        / "current-landmark"
        / f"current-landmark_{CHUNK_STAMP}.csv"
    )
    with landmark_csv.open(newline="", encoding="utf-8") as handle:
        landmark_rows = list(csv.reader(handle))
    assert [row[3:5] for row in landmark_rows[1:]] == [
        ["logs", "odour1"],
        ["logs", "odour1"],
    ]
    assert [float(row[6]) for row in landmark_rows[1:]] == [0.5, 1.5]

    manifest = json.loads(
        (output / "concatenation_manifest.json").read_text(encoding="utf-8")
    )
    assert [Path(item["path"]).name for item in manifest["sources"]] == [
        first.name,
        second.name,
    ]
    assert manifest["existing_loader_validation"]["analog_samples"] == 16


def test_rejects_incompatible_settings(tmp_path: Path) -> None:
    source_root = tmp_path / "cohort2" / "source"
    first = _make_session(source_root, "20260827T140000")
    second = _make_session(
        source_root, "20260827T150000", settings_marker="different"
    )

    with pytest.raises(ConcatenationError, match="session-settings differ"):
        concatenate_sessions(
            [first, second],
            output_root=tmp_path / "cohort2" / "combined",
            creation_time=datetime(2026, 10, 9, 15, 30, 12),
        )
