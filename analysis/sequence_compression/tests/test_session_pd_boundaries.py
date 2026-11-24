import numpy as np

from analysis.sequence_compression.session_functions.session import (
    _match_pd_boundaries,
)


def test_internal_missing_pd_boundary_uses_expected_position():
    matched, fallback_visits = _match_pd_boundaries(
        candidates=[10.1, 30.1],
        expected=[10.0, 20.0, 30.0],
        tolerance=0.5,
        max_observed_position=40.0,
    )

    np.testing.assert_allclose(matched, [10.1, 20.0, 30.1])
    np.testing.assert_array_equal(fallback_visits, [1])


def test_unreached_final_boundary_is_not_synthesized():
    matched, fallback_visits = _match_pd_boundaries(
        candidates=[10.1],
        expected=[10.0, 20.0],
        tolerance=0.5,
        max_observed_position=15.0,
    )

    np.testing.assert_allclose(matched, [10.1])
    np.testing.assert_array_equal(fallback_visits, [])


def test_pd_candidate_is_not_reused_for_two_visits():
    matched, fallback_visits = _match_pd_boundaries(
        candidates=[10.1],
        expected=[10.0, 10.2],
        tolerance=0.5,
        max_observed_position=20.0,
    )

    np.testing.assert_allclose(matched, [10.1, 10.2])
    np.testing.assert_array_equal(fallback_visits, [1])
