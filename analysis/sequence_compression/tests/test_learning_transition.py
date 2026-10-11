from pathlib import Path

import numpy as np

from analysis.sequence_compression.learning_transition import (
    ROLE_GOAL,
    ROLE_NONGOAL,
    ROLE_TEST,
    build_session_lick_record,
    calculate_lap_binary_rates,
    calculate_lap_category_rates,
    detect_discrimination_onset,
    detect_legacy_phase_boundaries,
    empty_cache,
    get_cached_record,
    load_lick_cache,
    put_cached_record,
    save_lick_cache,
)


def test_rewarded_goals_are_cut_off_at_their_actual_reward() -> None:
    entries = np.arange(0, 60, 10)
    exits = entries + 9
    signal = np.zeros(60)

    # First goal: one reward-triggering lick and two consumption licks.
    signal[[4, 7, 8]] = 1
    # The second goal is missed and therefore retains its full landmark.
    signal[[37, 38]] = 1
    # Non-goal and test licking also retain their full landmarks.
    signal[[18, 28, 48, 58]] = 1

    sess = {
        "num_landmarks": 3,
        "all_lms": np.tile([0, 1, 2], 2),
        "thresholded_licks": signal,
        "reward_idx": np.array([4]),
        "goals_idx": np.array([0, 3]),
        "non_goals_idx": np.array([1, 4]),
        "test_idx": np.array([2, 5]),
    }

    record = build_session_lick_record(sess, entries, exits, bins=5)

    assert record["full_lick_rate"].shape == (2, 3, 5)
    assert record["full_binary_lick"].shape == (2, 3)
    assert record["role_codes"][0].tolist() == [
        ROLE_GOAL,
        ROLE_NONGOAL,
        ROLE_TEST,
    ]
    assert record["rewarded_goal"].tolist() == [
        [True, False, False],
        [False, False, False],
    ]
    assert record["reward_cutoff_idx"][0, 0] == 4

    # The trigger lick is retained, whereas bins after reward are unobserved.
    first_goal = record["pre_reward_lick_rate"][0, 0]
    assert np.nansum(first_goal) > 0
    assert np.isnan(first_goal[-1])
    assert np.nansum(record["full_lick_rate"][0, 0]) > np.nansum(
        first_goal
    )

    # A missed goal is identical in the full and truncated maps.
    np.testing.assert_allclose(
        record["pre_reward_lick_rate"][1, 0],
        record["full_lick_rate"][1, 0],
        equal_nan=True,
    )
    assert record["full_binary_lick"].tolist() == [
        [1, 1, 1],
        [1, 1, 1],
    ]


def test_category_rates_keep_rewarded_and_missed_goals_separate() -> None:
    lick_map = np.array(
        [
            [[1.0, 1.0], [0.0, 0.0], [0.25, 0.25]],
            [[0.5, 0.5], [0.1, 0.1], [0.0, 0.0]],
        ]
    )
    roles = np.tile([ROLE_GOAL, ROLE_NONGOAL, ROLE_TEST], (2, 1))
    rewarded = np.array(
        [[True, False, False], [False, False, False]]
    )

    rates = calculate_lap_category_rates(lick_map, roles, rewarded)

    np.testing.assert_allclose(rates["goal"], [1.0, 0.5])
    np.testing.assert_allclose(rates["non_goal"], [0.0, 0.1])
    np.testing.assert_allclose(rates["test"], [0.25, 0.0])
    assert rates["rewarded_goal"][0] == 1.0
    assert np.isnan(rates["rewarded_goal"][1])
    assert np.isnan(rates["missed_goal"][0])
    assert rates["missed_goal"][1] == 0.5


def test_full_binary_readout_includes_post_reward_licking() -> None:
    signal = np.zeros(10)
    signal[7] = 1
    sess = {
        "num_landmarks": 1,
        "all_lms": np.array([0]),
        "thresholded_licks": signal,
        "reward_idx": np.array([4]),
        "goals_idx": np.array([0]),
        "non_goals_idx": np.array([], dtype=int),
        "test_idx": np.array([], dtype=int),
    }

    record = build_session_lick_record(
        sess, np.array([0]), np.array([9]), bins=5
    )

    assert record["full_binary_lick"][0, 0] == 1
    assert record["pre_reward_binary_lick"][0, 0] == 0


def test_binary_rates_are_fraction_of_landmarks_licked() -> None:
    binary_licks = np.array(
        [
            [1, 0, 1, 0],
            [1, 1, 0, 0],
        ]
    )
    roles = np.array(
        [
            [ROLE_GOAL, ROLE_GOAL, ROLE_NONGOAL, ROLE_TEST],
            [ROLE_GOAL, ROLE_GOAL, ROLE_NONGOAL, ROLE_TEST],
        ]
    )

    rates = calculate_lap_binary_rates(binary_licks, roles)

    np.testing.assert_allclose(rates["goal"], [0.5, 1.0])
    np.testing.assert_allclose(rates["non_goal"], [1.0, 0.0])
    np.testing.assert_allclose(rates["test"], [0.0, 0.0])


def test_discrimination_onset_requires_sustained_selectivity() -> None:
    lap_rates = {
        "goal": np.array([0.4] * 4 + [0.8] * 8),
        "non_goal": np.array([0.4] * 4 + [0.1] * 8),
        "test": np.array([0.4] * 4 + [0.2] * 8),
    }

    result = detect_discrimination_onset(
        lap_rates,
        window=3,
        persistence=2,
        min_effect=0.2,
        min_goal_fraction=0.25,
    )

    assert result["onset_lap"] == 5
    np.testing.assert_allclose(
        result["rolling_non_goal"][2:4], [0.4, 0.4]
    )
    np.testing.assert_allclose(result["rolling_test"][2:4], [0.4, 0.4])
    assert result["criterion"][4]
    assert result["criterion"][5]
    assert result["criterion"][6]


def test_legacy_phase_boundaries_report_missing_crossings() -> None:
    rates = {
        "goal": np.ones(20),
        "non_goal": np.zeros(20),
        "test": np.ones(20),
    }

    result = detect_legacy_phase_boundaries(rates, start_lap=5)

    assert result["exploration_end_lap"] == 6
    assert result["discrimination_onset_lap"] is None


def test_npz_cache_can_be_reloaded_and_extended(tmp_path: Path) -> None:
    source = tmp_path / "session"
    source.mkdir()
    cache_path = tmp_path / "lick_cache.npz"
    cache = empty_cache()

    put_cached_record(
        cache,
        sequence="even",
        session_id="session-1",
        source_path=source,
        record={"bins": 16, "pre_reward_lick_rate": np.ones((1, 2, 16))},
    )
    save_lick_cache(cache_path, cache)

    loaded = load_lick_cache(cache_path)
    record = get_cached_record(
        loaded,
        sequence="even",
        session_id="session-1",
        source_path=source,
        bins=16,
    )
    assert record is not None
    assert record["pre_reward_lick_rate"].shape == (1, 2, 16)

    put_cached_record(
        loaded,
        sequence="odd",
        session_id="session-2",
        source_path=source,
        record={"bins": 16, "pre_reward_lick_rate": np.zeros((1, 2, 16))},
    )
    save_lick_cache(cache_path, loaded)

    reloaded = load_lick_cache(cache_path)
    assert len(reloaded["records"]) == 2
