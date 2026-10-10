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
    find_base_path,
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
    landmark_centers: tuple[float, ...] = (1.0, 3.0, 5.0),
) -> Path:
    session = parent / f"ses-abab-random-001_date-{timestamp}"
    behav = session / "behav"

    _write_csv(
        behav / "analog-data" / f"analog-data_{CHUNK_STAMP}.csv",
        ["Seconds", "BufferIndex"],
        [[index / 10, index - 1] for index in range(1, 8)],
    )
    analog = np.arange(28, dtype=np.float64).reshape(28, 1)
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
        [
            [index / 10, 0, 0, position, position, position, position**2]
            for index, position in enumerate(range(7), start=1)
        ],
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
        [
            [
                (center + 1) / 10,
                index,
                1,
                "logs",
                f"odour{index + 1}",
                0 if index == 0 else -1,
                center,
                False,
                False,
                False,
                False,
            ]
            for index, center in enumerate(landmark_centers)
        ],
    )
    _write_csv(
        behav / "experiment-events" / f"experiment-events_{CHUNK_STAMP}.csv",
        ["Seconds", "Value"],
        [[0.2, "release: odour1"], [0.4, "release: odour2"]],
    )
    _write_csv(
        behav / "licks" / f"licks_{CHUNK_STAMP}.csv",
        ["Seconds", "Value"],
        [[index / 10, index == 4] for index in range(1, 8)],
    )
    _write_csv(
        behav / "reward" / f"reward_{CHUNK_STAMP}.csv",
        ["Seconds", "Value"],
        [[0.4, "Water"]],
    )
    _write_csv(
        behav / "treadmill-speed" / f"treadmill-speed_{CHUNK_STAMP}.csv",
        ["Seconds", "Value"],
        [[index / 10, index] for index in range(1, 8)],
    )

    settings = {
        "value": {
            "marker": settings_marker,
            "velocityThreshold": 1,
            "trial": {
                "offsets": [1, 9],
                "landmarks": [
                    [{"size": 1, "rewardSequencePosition": 0}],
                    [{"size": 1, "rewardSequencePosition": -1}],
                ],
            },
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
    assert len(analog) == 44

    analog_csv = output / "behav" / "analog-data" / f"analog-data_{CHUNK_STAMP}.csv"
    with analog_csv.open(newline="", encoding="utf-8") as handle:
        analog_rows = list(csv.DictReader(handle))
    assert [float(row["BufferIndex"]) for row in analog_rows] == list(range(11))
    assert [float(row["Seconds"]) for row in analog_rows] == pytest.approx(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.500001, 0.600001,
         0.700001, 0.800001, 0.900001, 1.000001]
    )

    position_csv = (
        output
        / "behav"
        / "current-position"
        / f"current-position_{CHUNK_STAMP}.csv"
    )
    with position_csv.open(newline="", encoding="utf-8") as handle:
        position_rows = list(csv.DictReader(handle))
    assert [float(row["Value.Length"]) for row in position_rows] == list(range(11))

    landmark_csv = (
        output
        / "behav"
        / "current-landmark"
        / f"current-landmark_{CHUNK_STAMP}.csv"
    )
    with landmark_csv.open(newline="", encoding="utf-8") as handle:
        landmark_rows = list(csv.reader(handle))
    assert [row[3] for row in landmark_rows[1:]] == ["logs"] * 5
    assert [float(row[6]) for row in landmark_rows[1:]] == [1, 3, 5, 7, 9]

    manifest = json.loads(
        (output / "concatenation_manifest.json").read_text(encoding="utf-8")
    )
    assert [Path(item["path"]).name for item in manifest["sources"]] == [
        first.name,
        second.name,
    ]
    assert manifest["existing_loader_validation"]["analog_samples"] == 44
    assert manifest["splices"][0]["gap_source"] == "recorded"
    assert manifest["splices"][0]["preserved_gap"] == 1
    assert manifest["splices"][0]["discarded_logical_landmarks"] == 1


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


def test_complete_first_session_uses_fallback_gap(tmp_path: Path) -> None:
    source_root = tmp_path / "cohort2" / "source"
    first = _make_session(
        source_root,
        "20260827T140000",
        landmark_centers=(1.0, 3.0),
    )
    second = _make_session(source_root, "20260827T150000")

    output = concatenate_sessions(
        [first, second],
        output_root=tmp_path / "cohort2" / "combined",
        creation_time=datetime(2026, 10, 9, 15, 30, 12),
    )

    manifest = json.loads(
        (output / "concatenation_manifest.json").read_text(encoding="utf-8")
    )
    splice = manifest["splices"][0]
    assert splice["gap_source"] == "fallback"
    assert splice["preserved_gap"] == 9
    assert splice["last_complete_landmark_exit"] == 3.5
    assert splice["target_next_landmark_entry"] == 12.5
    assert splice["discarded_logical_landmarks"] == 0

    landmark_csv = (
        output
        / "behav"
        / "current-landmark"
        / f"current-landmark_{CHUNK_STAMP}.csv"
    )
    with landmark_csv.open(newline="", encoding="utf-8") as handle:
        landmark_rows = list(csv.reader(handle))[1:]
    # Session 2's first landmark has centre 13 after shifting, hence entry 12.5.
    assert [float(row[6]) for row in landmark_rows] == [1, 3, 13, 15, 17]


def test_allows_one_trailing_analog_csv_buffer_without_binary_data(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "cohort2" / "source"
    first = _make_session(source_root, "20260827T140000")
    second = _make_session(source_root, "20260827T150000")
    binary = first / "behav" / "analog-data" / f"analog-data_{CHUNK_STAMP}.bin"
    # Four float64 samples make one synthetic analog buffer.
    binary.write_bytes(binary.read_bytes()[: -4 * np.dtype(np.float64).itemsize])

    output = concatenate_sessions(
        [first, second],
        output_root=tmp_path / "cohort2" / "combined",
        creation_time=datetime(2026, 10, 9, 15, 30, 12),
    )

    manifest = json.loads(
        (output / "concatenation_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["existing_loader_validation"]["analog_samples"] == 44


def test_find_base_path_uses_source_date_from_concatenation_manifest(
    tmp_path: Path,
) -> None:
    mouse_root = tmp_path / "cohort2" / "sub-08"
    combined = mouse_root / "ses-full017_date-20261010T090000"
    (combined / "behav").mkdir(parents=True)
    manifest = {
        "sources": [
            {
                "path": "/raw/sub-08/ses-full017_date-20261009T075421",
                "acquired_at": "2026-10-09T07:54:21",
            },
            {
                "path": "/raw/sub-08/ses-full017_date-20261009T082201",
                "acquired_at": "2026-10-09T08:22:01",
            },
        ]
    }
    (combined / "concatenation_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    assert find_base_path("08", "261009", tmp_path / "cohort2") == combined
