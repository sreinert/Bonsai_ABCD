import numpy as np

from musicboxes.session_overview import (
    _concat_masks,
    position_tuning_matrix,
    reward_tuning_matrix,
)


def test_concat_masks_labels_each_roi():
    first = np.array([[True, False], [False, False]])
    second = np.array([[False, False], [False, True]])
    np.testing.assert_array_equal(_concat_masks([first, second]), [[1, 0], [0, 2]])


def test_reward_tuning_uses_complete_windows_without_mutating_rewards():
    dff = np.arange(40, dtype=float).reshape(2, 20)
    session = {"rewards": np.array([1, 3, 10])}

    result = reward_tuning_matrix(dff, session, frame_rate=2, window_size=(-1, 2))

    assert result.shape == (2, 1, 6)
    np.testing.assert_array_equal(result[:, 0], dff[:, 8:14])
    np.testing.assert_array_equal(session["rewards"], [1, 3, 10])


def test_position_tuning_bins_each_lap_on_the_aligned_axis():
    dff = np.array([
        [0, 2, 4, 6, 10, 12, 14, 16],
        [1, 3, 5, 7, 11, 13, 15, 17],
    ], dtype=float)
    session = {
        "position": np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=float),
        "lap_idx": np.array([0, 0, 0, 0, 1, 1, 1, 1]),
        "num_laps": 2,
    }

    result, edges = position_tuning_matrix(dff, session, bins=2)

    assert result.shape == (2, 2, 2)
    np.testing.assert_allclose(edges, [0, 1.5, 3])
    np.testing.assert_allclose(result[0], [[1, 5], [11, 15]])
    np.testing.assert_allclose(result[1], [[2, 6], [12, 16]])
