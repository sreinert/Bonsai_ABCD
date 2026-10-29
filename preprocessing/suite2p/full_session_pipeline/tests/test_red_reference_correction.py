from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


PIPELINE_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
for import_root in (REPOSITORY_ROOT, PIPELINE_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import red_reference_correction  # noqa: E402


def legacy_ops() -> dict[str, object]:
    return {
        "fs": 10.0,
        "baseline": "maximin",
        "win_baseline": 60.0,
        "sig_baseline": 10.0,
        "prctile_baseline": 8.0,
        "neucoeff": 0.7,
    }


def test_channel_delta_matches_nwb_barcoding(monkeypatch: pytest.MonkeyPatch) -> None:
    from nwb_barcoding import compute_dff

    fluorescence = np.asarray([[5.0, 7.0, 11.0]], dtype=np.float32)
    neuropil = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
    ops = {
        "fs": 2.0,
        "dcnv_preprocess": {
            "baseline": "constant_prctile",
            "win_baseline": 60.0,
            "sig_baseline": 10.0,
            "prctile_baseline": 0.0,
        },
        "extraction": {"neuropil_coefficient": 0.55},
    }

    def fake_preprocess(values, baseline, window, sigma, fs, percentile):
        del baseline, window, sigma, fs
        return values - np.percentile(values, percentile, axis=1, keepdims=True)

    monkeypatch.setattr(red_reference_correction, "_suite2p_preprocess", fake_preprocess)
    monkeypatch.setattr(compute_dff, "_suite2p_preprocess", fake_preprocess)
    observed = red_reference_correction.compute_channel_delta(fluorescence, neuropil, ops)
    corrected = fluorescence - 0.55 * neuropil
    baseline, window, sigma, percentile = compute_dff._get_baseline_params(ops)
    expected = compute_dff._suite2p_preprocess(
        corrected, baseline, window, sigma, ops["fs"], percentile
    )
    assert np.allclose(observed, expected)


def test_channel_delta_supports_legacy_ops_and_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(red_reference_correction, "_suite2p_preprocess", lambda values, *_: values)
    fluorescence = np.asarray([[5.0, 6.0]])
    neuropil = np.asarray([[2.0, 2.0]])
    observed = red_reference_correction.compute_channel_delta(
        fluorescence, neuropil, legacy_ops(), neuropil_coefficient=0.25
    )
    assert np.allclose(observed, fluorescence - 0.25 * neuropil)


def test_channel_delta_validates_shapes() -> None:
    with pytest.raises(ValueError, match="equal shapes"):
        red_reference_correction.compute_channel_delta(
            np.ones((1, 3)), np.ones((1, 2)), legacy_ops()
        )


def test_suite2p_preprocess_explicitly_uses_cpu_when_device_is_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    dcnv = ModuleType("suite2p.extraction.dcnv")

    def fake_preprocess(*arguments, device=None):
        observed["device"] = device
        return arguments[0]

    dcnv.preprocess = fake_preprocess
    extraction = ModuleType("suite2p.extraction")
    extraction.dcnv = dcnv
    suite2p = ModuleType("suite2p")
    suite2p.extraction = extraction
    torch = ModuleType("torch")
    torch.device = lambda name: name
    monkeypatch.setitem(sys.modules, "suite2p", suite2p)
    monkeypatch.setitem(sys.modules, "suite2p.extraction", extraction)
    monkeypatch.setitem(sys.modules, "suite2p.extraction.dcnv", dcnv)
    monkeypatch.setitem(sys.modules, "torch", torch)

    values = np.ones((1, 3), dtype=np.float32)
    result = red_reference_correction._suite2p_preprocess(
        values, "maximin", 60.0, 10.0, 30.0, 8.0
    )

    assert result is values
    assert observed["device"] == "cpu"


def test_ratio_matches_celltv_percentile_floor_without_interpolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        red_reference_correction, "_suite2p_preprocess", lambda values, *_: values
    )
    green = np.full((1, 9), 100.0)
    red = np.full((1, 9), 10.0)
    red[0, 4] = 0.0
    ops = legacy_ops()

    result = red_reference_correction.compute_celltv_ratio_delta(
        green,
        np.zeros_like(green),
        red,
        np.zeros_like(red),
        ops,
        neuropil_coefficient=0.0,
        smoothing_seconds=0.0,
    )

    assert result.denominator_floor[0] == pytest.approx(4.0)
    np.testing.assert_allclose(result.ratio[0, [0, 1, 2, 3, 5, 6, 7, 8]], 10.0)
    assert result.ratio[0, 4] == pytest.approx(25.0)
    assert result.status[0] == "ok"
    assert np.all(np.isfinite(result.delta_ratio))


def test_ratio_uses_celltv_absolute_floor_for_zero_red(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        red_reference_correction, "_suite2p_preprocess", lambda values, *_: values
    )
    green = np.full((1, 9), 100.0)
    red = np.zeros((1, 9))

    result = red_reference_correction.compute_celltv_ratio_delta(
        green,
        np.zeros_like(green),
        red,
        np.zeros_like(red),
        legacy_ops(),
        neuropil_coefficient=0.0,
    )

    assert result.denominator_floor[0] == pytest.approx(1e-3)
    np.testing.assert_allclose(result.delta_ratio, 100_000.0)
    assert result.status[0] == "ok"


def test_ratio_does_not_reject_frequent_low_red_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        red_reference_correction, "_suite2p_preprocess", lambda values, *_: values
    )
    green = np.full((1, 20), 100.0)
    red = np.full((1, 20), 10.0)
    red[0, :3] = 0.0

    result = red_reference_correction.compute_celltv_ratio_delta(
        green,
        np.zeros_like(green),
        red,
        np.zeros_like(red),
        legacy_ops(),
        neuropil_coefficient=0.0,
        smoothing_seconds=0.0,
    )

    assert result.status[0] == "ok"
    np.testing.assert_allclose(result.delta_ratio[0, :3], 100_000.0)
    np.testing.assert_allclose(result.delta_ratio[0, 3:], 10.0)


def test_ratio_uses_celltv_medfilt_zero_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        red_reference_correction, "_suite2p_preprocess", lambda values, *_: values
    )
    green = np.ones((1, 5))
    red = np.array([[100.0, 0.0, 0.0, 0.0, 0.0]])
    ops = legacy_ops()
    ops["fs"] = 12.0

    result = red_reference_correction.compute_celltv_ratio_delta(
        green,
        np.zeros_like(green),
        red,
        np.zeros_like(red),
        ops,
        neuropil_coefficient=0.0,
    )

    assert result.smoothing_kernel_frames == 3
    np.testing.assert_array_equal(result.smoothed_red, np.zeros((1, 5)))


def test_tukey_correction_removes_shared_motion_and_preserves_transient() -> None:
    rng = np.random.default_rng(7)
    red = rng.normal(size=2000)
    green = 2.0 * red + rng.normal(scale=0.05, size=red.size)
    transient = red > np.percentile(red, 97)
    green[transient] += 20.0

    result = red_reference_correction.fit_reference_correction(
        green[None, :], red[None, :], fs=20.0, smoothing_seconds=0.0
    )
    ordinary_slope = np.polyfit(red, green, 1)[0]

    assert result.status[0] == "applied"
    assert abs(result.beta[0] - 2.0) < abs(ordinary_slope - 2.0)
    assert abs(np.corrcoef(result.corrected[0, ~transient], red[~transient])[0, 1]) < 0.1
    assert np.median(result.corrected[0, transient]) > 10.0


def test_correction_uses_nearest_edge_median_filter() -> None:
    red = np.asarray([[100.0, 0.0, 0.0, 0.0, 0.0]])
    result = red_reference_correction.fit_reference_correction(
        red.copy(), red, fs=12.0, smoothing_seconds=0.25
    )
    assert result.smoothing_kernel_frames == 3
    assert result.smoothed_red[0, 0] == 100.0


@pytest.mark.parametrize(
    ("green", "red", "kwargs", "expected_status"),
    [
        (np.arange(20.0), np.ones(20), {}, "flat_red"),
        (-np.arange(20.0), np.arange(20.0), {"smoothing_seconds": 0.0}, "nonpositive_coupling"),
        (np.arange(20.0), np.r_[np.arange(19.0), np.nan], {}, "nonfinite_red"),
        (np.arange(20.0), np.arange(20.0), {"max_iterations": 0}, "nonconvergent"),
    ],
)
def test_correction_safe_fallbacks(
    green: np.ndarray,
    red: np.ndarray,
    kwargs: dict[str, object],
    expected_status: str,
) -> None:
    result = red_reference_correction.fit_reference_correction(
        green[None, :], red[None, :], fs=10.0, **kwargs
    )
    assert result.status[0] == expected_status
    assert result.beta[0] == 0
    assert np.allclose(result.corrected[0], green, equal_nan=True)
