import numpy as np

from nwb_barcoding import compute_dff


def test_compute_dff_writes_only_to_requested_destination(tmp_path, monkeypatch):
    source = tmp_path / "source" / "suite2p" / "plane0"
    source.mkdir(parents=True)
    fluorescence = np.array([[1.0, 2.0, 3.0, 4.0]])
    neuropil = np.array([[0.1, 0.1, 0.1, 0.1]])
    np.save(source / "F.npy", fluorescence)
    np.save(source / "Fneu.npy", neuropil)
    np.save(
        source / "ops.npy",
        {
            "baseline": "constant_prctile",
            "win_baseline": 60.0,
            "sig_baseline": 10.0,
            "prctile_baseline": 0.0,
            "fs": 45.0,
            "neucoeff": 0.35,
        },
    )
    source_files = sorted(path.name for path in source.iterdir())

    destination = tmp_path / "output" / "dff" / "plane0"
    def constant_percentile(values, _baseline, _window, _sigma, _fs, percentile):
        return values - np.percentile(values, percentile, axis=1)[:, None]

    monkeypatch.setattr(compute_dff, "_suite2p_preprocess", constant_percentile)
    result = compute_dff.compute_dff_for_plane(source, destination)

    corrected = fluorescence - 0.35 * neuropil
    expected = corrected - np.percentile(corrected, 0.0, axis=1)[:, None]
    assert np.allclose(np.load(result), expected)
    assert sorted(path.name for path in source.iterdir()) == source_files


def test_neuropil_coefficient_can_be_read_from_nested_ops():
    ops = {"extraction": {"neuropil_coefficient": 0.55}}

    assert compute_dff._get_neuropil_coefficient(ops) == 0.55


def test_missing_neuropil_coefficient_is_not_silently_assumed():
    import pytest

    with pytest.raises(KeyError, match="cannot reproduce Suite2p/CellTV"):
        compute_dff._get_neuropil_coefficient({})


def test_explicit_neuropil_coefficient_override_remains_available():
    assert compute_dff._get_neuropil_coefficient({"neucoeff": 0.35}, 0.8) == 0.8
