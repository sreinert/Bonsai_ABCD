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
    cropped = funcimg / "recording_cropped.tif"
    cropped.write_bytes(b"cropped")
    original_h5 = funcimg / "recording.h5"
    original_h5.write_bytes(b"h5")

    assert staging.resolve_vdaq_h5(cropped) == original_h5
    view = staging.stage_raw_funcimg(raw_session, tmp_path / "stage", cropped)
    assert sorted(path.name for path in (view / "funcimg").iterdir()) == [
        "recording.h5",
        "recording_cropped.tif",
    ]
