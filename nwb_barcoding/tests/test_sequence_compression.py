from pathlib import Path

import pytest

from nwb_barcoding.sequence_compression import (
    resolve_paths,
    resolve_raw_session,
    short_session_name,
)


def _make_session(project, subject, full_name):
    raw = project / "rawdata" / "cohort2" / subject / full_name
    (raw / "behav").mkdir(parents=True)
    suite2p = project / "processed" / "cohort2" / subject / full_name / "suite2p"
    suite2p.mkdir(parents=True)
    return raw, suite2p


def test_resolve_paths_accepts_short_session(tmp_path):
    raw, suite2p = _make_session(
        tmp_path, "sub-02", "ses-full002_date-20260818T134310"
    )

    paths = resolve_paths("sub-02", "full002", tmp_path)

    assert paths.raw_session_path == raw
    assert paths.suite2p_dir == suite2p
    assert paths.output_session == (
        tmp_path
        / "BehaviourAligned_robust"
        / "cohort2"
        / "sub-02"
        / raw.name
    )


def test_resolve_paths_accepts_full_session(tmp_path):
    raw, _ = _make_session(
        tmp_path, "sub-02", "ses-full002_date-20260818T134310"
    )
    assert resolve_paths("sub-02", raw.name, tmp_path).raw_session_path == raw


def test_short_session_rejects_ambiguous_matches(tmp_path):
    subject = tmp_path / "sub-02"
    (subject / "ses-full002_date-20260818T134310").mkdir(parents=True)
    (subject / "ses-full002_date-20260818T150000").mkdir()

    with pytest.raises(ValueError, match="ambiguous"):
        resolve_raw_session(subject, "full002")


def test_missing_suite2p_is_reported(tmp_path):
    raw = (
        tmp_path
        / "rawdata"
        / "cohort2"
        / "sub-02"
        / "ses-full002_date-20260818T134310"
    )
    raw.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="Suite2p directory"):
        resolve_paths("sub-02", "full002", tmp_path)

