from pathlib import Path

import numpy as np
import pytest

from nwb_barcoding import staging


def _snapshot(root):
    return {
        path.relative_to(root): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


def test_staging_does_not_modify_suite2p_source(tmp_path, monkeypatch):
    source = tmp_path / "processed" / "suite2p" / "plane0"
    source.mkdir(parents=True)
    (source / "F.npy").write_bytes(b"source-data")
    before = _snapshot(tmp_path / "processed")

    def fake_ensure_dff(funcimg_dir, **_kwargs):
        funcimg_dir = Path(funcimg_dir)
        assert (funcimg_dir / "suite2p").is_symlink()
        dff = funcimg_dir / "dff" / "plane0" / "dFF.npy"
        dff.parent.mkdir(parents=True)
        dff.write_bytes(b"derived-data")

    monkeypatch.setattr(staging.compute_dff, "ensure_dff", fake_ensure_dff)
    output = tmp_path / "aligned"
    with staging.staged_tracking_layout(source.parent, output) as tracking:
        assert (tracking / "funcimg" / "dff" / "plane0" / "dFF.npy").is_file()

    assert _snapshot(tmp_path / "processed") == before
    assert list(output.iterdir()) == []


def test_raw_funcimg_view_excludes_appledouble_sidecar(tmp_path):
    raw_session = tmp_path / "raw" / "session"
    funcimg = raw_session / "funcimg"
    funcimg.mkdir(parents=True)
    tif = funcimg / "recording.tif"
    h5 = funcimg / "recording.h5"
    tif.write_bytes(b"tif")
    h5.write_bytes(b"h5")
    (funcimg / "._recording.tif").write_bytes(b"sidecar")

    view = staging.stage_raw_funcimg(raw_session, tmp_path / "stage", tif)

    assert sorted(path.name for path in (view / "funcimg").iterdir()) == [
        "recording.h5",
        "recording.tif",
    ]
    assert (view / "funcimg" / "recording.tif").is_symlink()


def test_select_scanimage_tif_uses_suite2p_cropped_source(tmp_path):
    raw_session = tmp_path / "raw" / "session"
    funcimg = raw_session / "funcimg"
    funcimg.mkdir(parents=True)
    (funcimg / "recording.tif").write_bytes(b"original")
    cropped = funcimg / "recording_cropped.tif"
    cropped.write_bytes(b"cropped")
    plane = tmp_path / "processed" / "suite2p" / "plane0"
    plane.mkdir(parents=True)
    np.save(plane / "ops.npy", {"filelist": [str(cropped)]})

    assert staging.select_scanimage_tif(raw_session, plane) == cropped


def test_select_scanimage_tif_rejects_source_different_from_suite2p(tmp_path):
    raw_session = tmp_path / "raw" / "session"
    funcimg = raw_session / "funcimg"
    funcimg.mkdir(parents=True)
    (funcimg / "recording.tif").write_bytes(b"original")
    (funcimg / "recording_cropped.tif").write_bytes(b"cropped")
    plane = tmp_path / "processed" / "suite2p" / "plane0"
    plane.mkdir(parents=True)
    np.save(plane / "ops.npy", {"filelist": ["/elsewhere/another.tif"]})

    with pytest.raises(RuntimeError, match="Suite2p says it processed"):
        staging.select_scanimage_tif(raw_session, plane)


def test_cropped_tiff_uses_original_vdaq_and_stages_both(tmp_path):
    raw_session = tmp_path / "raw" / "session"
    funcimg = raw_session / "funcimg"
    funcimg.mkdir(parents=True)
    original = funcimg / "recording.tif"
    original.write_bytes(b"original with ScanImage metadata")
    cropped = funcimg / "recording_cropped.tif"
    cropped.write_bytes(b"cropped")
    original_h5 = funcimg / "recording.h5"
    original_h5.write_bytes(b"h5")

    assert staging.resolve_scanimage_metadata_tif(cropped) == original
    assert staging.resolve_vdaq_h5(cropped) == original_h5
    view = staging.stage_raw_funcimg(raw_session, tmp_path / "stage", original)
    assert sorted(path.name for path in (view / "funcimg").iterdir()) == [
        "recording.h5",
        "recording.tif",
    ]


def test_cropped_tiff_requires_original_for_scanimage_metadata(tmp_path):
    cropped = tmp_path / "recording_cropped.tif"
    cropped.write_bytes(b"cropped")

    with pytest.raises(FileNotFoundError, match="no uncropped TIFF"):
        staging.resolve_scanimage_metadata_tif(cropped)


def test_cropped_tiff_prefers_complete_original_vdaq(tmp_path, monkeypatch):
    cropped = tmp_path / "recording_cropped.tif"
    cropped.write_bytes(b"cropped")
    original = tmp_path / "recording.h5"
    original.write_bytes(b"original")
    recovered = tmp_path / "recording_recovered_syncG1_upto_0534.h5"
    recovered.write_bytes(b"recovered")
    counts = {original: 76_953, recovered: 44_538}
    monkeypatch.setattr(staging, "_vdaq_frame_count", counts.__getitem__)

    assert staging.resolve_vdaq_h5(cropped, expected_frame_count=76_953) == original


def test_cropped_tiff_uses_recovery_when_original_is_unreadable(tmp_path, monkeypatch):
    cropped = tmp_path / "recording_cropped.tif"
    cropped.write_bytes(b"cropped")
    original = tmp_path / "recording.h5"
    original.write_bytes(b"damaged")
    recovered = tmp_path / "recording_recovered_syncG1_upto_0860.h5"
    recovered.write_bytes(b"recovered")

    def inspect(path):
        if path == original:
            raise OSError("corrupt chunk")
        return 76_953

    monkeypatch.setattr(staging, "_vdaq_frame_count", inspect)
    assert staging.resolve_vdaq_h5(cropped, expected_frame_count=76_953) == recovered


def test_cropped_tiff_rejects_truncated_recovery(tmp_path, monkeypatch):
    cropped = tmp_path / "recording_cropped.tif"
    cropped.write_bytes(b"cropped")
    original = tmp_path / "recording.h5"
    original.write_bytes(b"damaged")
    recovered = tmp_path / "recording_recovered_syncG1_upto_0534.h5"
    recovered.write_bytes(b"recovered")
    counts = {original: 76_000, recovered: 44_538}
    monkeypatch.setattr(staging, "_vdaq_frame_count", counts.__getitem__)

    with pytest.raises(RuntimeError, match=r"76953 frames.*44538 FrameTTL onsets"):
        staging.resolve_vdaq_h5(cropped, expected_frame_count=76_953)
