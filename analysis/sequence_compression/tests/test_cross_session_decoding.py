import numpy as np
import pytest

from analysis.sequence_compression.sc_neural_data.cross_session_decoding import (
    decode_session_pair,
    make_landmark_population,
    make_trial_population,
)


def test_make_trial_population_baseline_subtracts_and_drops_edge_trials():
    dff = np.vstack([np.arange(20), 2 * np.arange(20)]).astype(float)
    result = make_trial_population(
        dff,
        entry_frames=[0, 5, 15],
        labels=["A", "A", "B"],
        cluster_ids=[10, 20],
        frame_rate=1,
        response_window=(0, 2),
        baseline_window=(-2, 0),
        zscore_traces=False,
    )
    assert result["trial_indices"].tolist() == [1, 2]
    np.testing.assert_allclose(result["X"], [[2, 4], [2, 4]])
    assert result["y"].tolist() == [0, 1]


def test_make_trial_population_uses_two_seconds_before_entry_without_baseline():
    dff = np.arange(12, dtype=float)[None, :]
    result = make_trial_population(
        dff,
        entry_frames=[3, 8],
        labels=["A", "B"],
        cluster_ids=[10],
        frame_rate=1,
        response_window=(-2, 0),
        baseline_window=None,
        zscore_traces=False,
    )
    np.testing.assert_allclose(result["X"], [[1.5], [6.5]])
    assert result["trial_indices"].tolist() == [0, 1]
    assert result["y"].tolist() == [0, 1]


def test_cross_session_decoder_matches_reordered_tracked_cells():
    rng = np.random.default_rng(4)
    train_y = np.repeat([0, 1], 30)
    test_y = np.repeat([0, 1], 25)
    train_ids = np.array([11, 22, 33, 44, 55, 66])
    test_ids = np.array([66, 33, 11, 55, 44, 22])
    weights = dict(zip(train_ids, [2.0, -1.5, 1.2, -0.8, 1.7, -1.1]))
    train_X = np.column_stack([
        (2 * train_y - 1) * weights[cell] + rng.normal(0, 0.3, train_y.size)
        for cell in train_ids
    ])
    test_X = np.column_stack([
        (2 * test_y - 1) * weights[cell] + rng.normal(0, 0.3, test_y.size)
        for cell in test_ids
    ])
    train = {"X": train_X, "y": train_y, "cluster_ids": train_ids,
             "classes": ("A", "B")}
    test = {"X": test_X, "y": test_y, "cluster_ids": test_ids,
            "classes": ("A", "B")}
    result = decode_session_pair(
        train, test, minimum_cells=6, n_permutations=20, random_state=7
    )
    assert result["balanced_accuracy"] > 0.95
    assert result["n_cells"] == 6
    assert result["permutation_p"] <= 2 / 21


def test_make_landmark_population_uses_each_entry_to_exit_interval():
    dff = np.vstack([np.arange(12), 2 * np.arange(12)]).astype(float)
    result = make_landmark_population(
        dff,
        entry_frames=[1, 6],
        exit_frames=[3, 9],
        labels=["A", "B"],
        cluster_ids=[10, 20],
        zscore_traces=False,
    )
    np.testing.assert_allclose(result["X"], [[2, 4], [7.5, 15]])
    assert result["landmark_duration_frames"].tolist() == [3, 4]
    assert result["y"].tolist() == [0, 1]


def test_cross_session_decoder_rejects_too_few_shared_cells():
    population = {
        "X": np.array([[0.0], [1.0]]),
        "y": np.array([0, 1]),
        "cluster_ids": np.array([3]),
        "classes": ("A", "B"),
    }
    with pytest.raises(ValueError, match="Only 1 shared cells"):
        decode_session_pair(population, population, minimum_cells=2)
