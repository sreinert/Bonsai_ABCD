from pathlib import Path

import numpy as np
import pytest

import nwb_barcoding.batch_sequence_compression as batch


def _session_tree(project, subject, session, with_suite2p=True):
    processed = project / "processed" / "cohort2" / subject / session
    processed.mkdir(parents=True)
    if with_suite2p:
        plane0 = processed / "suite2p" / "plane0"
        plane0.mkdir(parents=True)
        for filename in batch.REQUIRED_PLANE0_FILES:
            (plane0 / filename).touch()
    raw = project / "rawdata" / "cohort2" / subject / f"ses-{session}_date-20260101T120000"
    raw.mkdir(parents=True)
    return raw


def test_discover_sessions_only_includes_complete_suite2p_directories(tmp_path):
    _session_tree(tmp_path, "sub-02", "full002")
    _session_tree(tmp_path, "sub-02", "full003", with_suite2p=False)
    _session_tree(tmp_path, "sub-01", "full001")
    _session_tree(tmp_path, "sub-03", "full004")
    (tmp_path / "processed" / "cohort2" / "sub-03" / "full004" / "suite2p"
     / "plane0" / "stat.npy").unlink()

    found = batch.discover_sessions(tmp_path)

    assert [(item.subject, item.session) for item in found] == [
        ("sub-01", "ses-full001_date-20260101T120000"),
        ("sub-02", "ses-full002_date-20260101T120000"),
    ]


def test_discover_sessions_can_be_limited_to_subject(tmp_path):
    _session_tree(tmp_path, "sub-02", "full002")
    sub07_raw = _session_tree(tmp_path, "sub-07", "full007")

    found = batch.discover_sessions(tmp_path, ["07"])

    assert [(item.subject, item.session) for item in found] == [
        ("sub-07", sub07_raw.name)
    ]


def test_discover_sessions_uses_ops_source_to_resolve_restart(tmp_path):
    first_raw = _session_tree(tmp_path, "sub-07", "full007")
    second_raw = first_raw.with_name("ses-full007_date-20260102T120000")
    (second_raw / "funcimg").mkdir(parents=True)
    ops_path = (
        tmp_path / "processed" / "cohort2" / "sub-07" / "full007"
        / "suite2p" / "plane0" / "ops.npy"
    )
    np.save(ops_path, {"data_path": [str(second_raw / "funcimg")]})

    found = batch.discover_sessions(tmp_path, ["sub-07"])

    assert len(found) == 1
    assert found[0].session == second_raw.name


def test_manifest_round_trip(tmp_path):
    _session_tree(tmp_path, "sub-02", "full002")
    sessions = batch.discover_sessions(tmp_path)
    manifest = batch.write_manifest(sessions, tmp_path / "tasks.csv")

    task = batch.read_manifest_task(manifest, 0)

    assert task == sessions[0]
    with pytest.raises(ValueError, match="exactly one row"):
        batch.read_manifest_task(manifest, 1)


def test_run_session_aligns_before_overview_and_uses_cohort_output(tmp_path, monkeypatch):
    raw = _session_tree(tmp_path, "sub-02", "full002")
    calls = []
    merged = tmp_path / "merged.nwb"

    def fake_build(**kwargs):
        calls.append(("build", kwargs))
        return merged, []

    def fake_overview(**kwargs):
        calls.append(("overview", kwargs))
        return {}, {}

    monkeypatch.setattr(batch, "build_sequence_compression_session", fake_build)
    fake_module = type("FakeOverview", (), {"run_overview": staticmethod(fake_overview)})
    monkeypatch.setitem(
        __import__("sys").modules,
        "analysis.sequence_compression.session_overview_sc",
        fake_module(),
    )

    result, overview_dir = batch.run_session(tmp_path, "sub-02", "full002")

    assert result == merged
    assert calls[0][0] == "build"
    assert calls[1][0] == "overview"
    assert calls[1][1]["save_dir"] == (
        tmp_path / "analysis" / "cohort2" / "sub-02" / raw.name
    )
    assert "stage" not in calls[1][1]
    assert overview_dir == calls[1][1]["save_dir"]


def test_run_session_does_not_plot_when_verification_flags(tmp_path, monkeypatch):
    _session_tree(tmp_path, "sub-02", "full002")
    monkeypatch.setattr(
        batch,
        "build_sequence_compression_session",
        lambda **kwargs: (Path("bad.nwb"), ["bad barcode fit"]),
    )

    with pytest.raises(RuntimeError, match="overview was not run"):
        batch.run_session(tmp_path, "sub-02", "full002")
