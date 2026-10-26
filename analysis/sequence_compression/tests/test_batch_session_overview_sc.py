from pathlib import Path

import pytest

import analysis.sequence_compression.batch_session_overview_sc as batch


def _aligned_session(project, subject, session, complete=True):
    aligned = project / "BehaviourAligned_robust" / "cohort2" / subject / session
    aligned.mkdir(parents=True)
    (aligned / "vrabcd_2p_preprocessed_test.nwb").touch()
    (aligned / "behaviour_data.npz").touch()
    if complete:
        (aligned / "valid_frames.npz").touch()
    return aligned


def test_discovery_only_includes_complete_alignment_products(tmp_path):
    complete = _aligned_session(tmp_path, "sub-07", "ses-full001_date-test")
    _aligned_session(tmp_path, "sub-07", "ses-full002_date-test", complete=False)
    _aligned_session(tmp_path, "sub-02", "ses-full001_date-test")

    found = batch.discover_aligned_sessions(tmp_path, ["07"])

    assert found == [
        batch.AlignedSession("sub-07", complete.name, complete)
    ]


def test_manifest_round_trip(tmp_path):
    aligned = _aligned_session(tmp_path, "sub-07", "ses-full001_date-test")
    sessions = [batch.AlignedSession("sub-07", aligned.name, aligned)]
    manifest = batch.write_manifest(sessions, tmp_path / "overview.csv")

    assert batch.read_manifest_task(manifest, 0) == sessions[0]
    with pytest.raises(ValueError, match="exactly one row"):
        batch.read_manifest_task(manifest, 1)


def test_run_session_only_generates_overview(tmp_path, monkeypatch):
    aligned = _aligned_session(tmp_path, "sub-07", "ses-full001_date-test")
    calls = []

    def fake_overview(**kwargs):
        calls.append(kwargs)
        return {}, {}

    fake_module = type("FakeOverview", (), {"run_overview": staticmethod(fake_overview)})
    monkeypatch.setitem(
        __import__("sys").modules,
        "analysis.sequence_compression.session_overview_sc",
        fake_module(),
    )

    result = batch.run_session(tmp_path, "07", aligned.name)

    expected = tmp_path / "analysis" / "cohort2" / "sub-07" / aligned.name
    assert result == expected
    assert len(calls) == 1
    assert calls[0]["session_id"] == aligned.name
    assert calls[0]["save_dir"] == expected
    assert calls[0]["show"] is False
