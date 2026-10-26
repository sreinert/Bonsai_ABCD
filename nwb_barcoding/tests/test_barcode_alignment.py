import numpy as np
import pytest

from nwb_barcoding.barcode_alignment import (
    barcode_discontinuities_are_sparse,
    find_dropped_barcodes,
    fit_alignment,
    match_barcodes,
)


def test_fit_alignment_recovers_offset_and_clock_rate():
    nominal_rate = 9000.0
    true_rate = 8999.0
    secondary_samples = np.arange(10, dtype=np.int64) * 9000
    secondary_nominal_times = secondary_samples / nominal_rate
    slope = nominal_rate / true_rate
    offset = 2.75
    main_rate = 5000.0
    main_samples = np.rint(
        (secondary_nominal_times * slope + offset) * main_rate
    ).astype(np.int64)
    values = np.arange(100, 110)

    fit = fit_alignment(
        {"start_sample": main_samples, "values": values, "sample_rate": main_rate},
        {"start_sample": secondary_samples, "values": values},
        secondary_nominal_rate=nominal_rate,
    )

    assert fit["delta_t"] == pytest.approx(offset, abs=1 / main_rate)
    assert fit["corrected_secondary_rate"] == pytest.approx(true_rate, abs=0.2)


def test_match_barcodes_excludes_large_outlier():
    secondary = np.arange(8, dtype=float)
    main = secondary * 1.001 + 0.5
    main[4] += 0.2

    match = match_barcodes(
        main, np.arange(8), secondary, np.arange(8), outlier_threshold_ms=50
    )

    assert match["n_outliers_excluded"] == 1
    assert match["slope"] == pytest.approx(1.001)
    assert match["intercept"] == pytest.approx(0.5)


def test_match_barcodes_requires_two_shared_values():
    with pytest.raises(ValueError, match="at least 2"):
        match_barcodes([0], [1], [0], [1])


def test_find_dropped_barcodes():
    assert find_dropped_barcodes([10, 11, 13, 14]).tolist() == [1]


def test_isolated_dropped_or_terminal_barcode_is_tolerated():
    assert barcode_discontinuities_are_sparse([10, 11, 13, 14])
    assert barcode_discontinuities_are_sparse([10, 11, 12, 4081])


def test_systematic_barcode_discontinuities_are_rejected():
    assert not barcode_discontinuities_are_sparse([10, 12, 14, 16, 18, 20])

