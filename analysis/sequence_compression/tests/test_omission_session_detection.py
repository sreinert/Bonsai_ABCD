import json

import pytest

from analysis.sequence_compression.session_functions.io import (
    count_session_trial_types,
    has_multiple_trial_types,
)


def _write_settings(session_folder, trial, extension="json"):
    settings_folder = session_folder / "behav" / "session-settings"
    settings_folder.mkdir(parents=True)
    settings_file = settings_folder / f"session-settings_0.{extension}"
    settings_file.write_text(json.dumps({"seconds": 0, "value": {"trial": trial}}))


def test_detects_multiple_configured_trial_types(tmp_path):
    session_folder = tmp_path / "ses-abab-random-001_date-20260101T120000"
    _write_settings(
        session_folder,
        [
            {"trial": {"landmarks": []}, "probability": 0.8},
            {"trial": {"landmarks": []}, "probability": 0.2},
        ],
    )

    assert count_session_trial_types(session_folder) == 2
    assert has_multiple_trial_types(session_folder)


@pytest.mark.parametrize(
    "trial",
    [
        [{"trial": {"landmarks": []}, "probability": 1.0}],
        {"landmarks": []},
    ],
)
def test_single_trial_is_not_classified_as_probe_session(tmp_path, trial):
    session_folder = tmp_path / "ses-ab-random-001_date-20260101T120000"
    _write_settings(session_folder, trial)

    assert count_session_trial_types(session_folder) == 1
    assert not has_multiple_trial_types(session_folder)


def test_missing_trial_field_is_rejected(tmp_path):
    session_folder = tmp_path / "ses-ab-random-001_date-20260101T120000"
    _write_settings(session_folder, None)

    with pytest.raises(ValueError, match="valid 'trial' field"):
        count_session_trial_types(session_folder)
