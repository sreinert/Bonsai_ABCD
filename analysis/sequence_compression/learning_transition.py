"""Utilities for analysing the onset of selective licking.

The functions in this module are deliberately independent of the notebook and
of the raw-data loaders.  A notebook can therefore cache the small derived
arrays and avoid reopening the raw behaviour files on every run.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter1d


CACHE_SCHEMA_VERSION = 3

ROLE_OTHER = 0
ROLE_GOAL = 1
ROLE_NONGOAL = 2
ROLE_TEST = 3


def _as_index_array(values: Any) -> np.ndarray:
    """Return an integer index array, including for ``None`` inputs."""

    if values is None:
        return np.array([], dtype=int)
    array = np.asarray(values)
    if array.ndim == 0:
        array = array.reshape(1)
    return array.astype(int, copy=False)


def _bin_landmark_signal(
    signal: np.ndarray,
    entry: int,
    exit_: int,
    bins: int,
    cutoff: int | None = None,
) -> np.ndarray:
    """Bin a landmark signal using edges defined by the full landmark.

    If ``cutoff`` is supplied, samples after it are omitted.  Consequently,
    bins after a reward are NaN rather than zeros and cannot be mistaken for a
    lack of licking.
    """

    if exit_ < entry:
        raise ValueError(f"Landmark exit {exit_} precedes entry {entry}")

    observed_exit = exit_ if cutoff is None else min(int(cutoff), exit_)
    if observed_exit < entry:
        return np.full(bins, np.nan, dtype=float)

    sample_indices = np.arange(entry, observed_exit + 1)
    bin_edges = np.linspace(entry, exit_ + 1, bins + 1)
    values = np.asarray(signal[sample_indices], dtype=float)
    binned, _, _ = stats.binned_statistic(
        sample_indices,
        values,
        statistic="mean",
        bins=bin_edges,
    )
    return binned


def _bin_full_landmark_signal(
    signal: np.ndarray,
    entry: int,
    exit_: int,
    bins: int,
) -> np.ndarray:
    """Match the original ``Behaviour.get_lm_lick_rate`` calculation."""

    if exit_ < entry:
        raise ValueError(f"Landmark exit {exit_} precedes entry {entry}")
    sample_indices = np.arange(entry, exit_ + 1)
    binned, _, _ = stats.binned_statistic(
        sample_indices,
        np.asarray(signal[sample_indices], dtype=float),
        statistic="mean",
        bins=bins,
    )
    return binned


def build_session_lick_record(
    sess: Mapping[str, Any],
    lm_entry_idx: np.ndarray,
    lm_exit_idx: np.ndarray,
    *,
    bins: int = 16,
) -> dict[str, Any]:
    """Create continuous and binary lick maps for one parsed session.

    Full maps cover each entire landmark. In the optional reward-truncated
    maps, rewarded goals are observed from landmark entry through their own
    first reward sample, inclusive; all other landmarks retain their full
    extent. Only complete laps are returned.
    """

    n_landmarks = int(sess["num_landmarks"])
    if n_landmarks <= 0:
        raise ValueError("num_landmarks must be positive")

    all_lms = np.asarray(sess["all_lms"])
    entries = np.asarray(lm_entry_idx, dtype=int)
    exits = np.asarray(lm_exit_idx, dtype=int)
    n_occurrences = min(len(all_lms), len(entries), len(exits))
    n_complete = (n_occurrences // n_landmarks) * n_landmarks
    if n_complete == 0:
        empty_map = np.empty((0, n_landmarks, bins), dtype=float)
        empty_lap_map = np.empty((0, n_landmarks), dtype=int)
        return {
            "full_lick_rate": empty_map.copy(),
            "pre_reward_lick_rate": empty_map.copy(),
            "full_binary_lick": empty_lap_map.copy(),
            "pre_reward_binary_lick": empty_lap_map.copy(),
            "role_codes": empty_lap_map.copy(),
            "rewarded_goal": empty_lap_map.astype(bool),
            "landmark_ids": empty_lap_map.copy(),
            "reward_cutoff_idx": empty_lap_map.copy(),
            "bins": bins,
        }

    signal = np.asarray(sess["thresholded_licks"], dtype=float)
    reward_idx = _as_index_array(sess.get("reward_idx"))
    goal_idx = set(_as_index_array(sess.get("goals_idx")).tolist())
    nongoal_idx = set(_as_index_array(sess.get("non_goals_idx")).tolist())
    test_idx = set(_as_index_array(sess.get("test_idx")).tolist())

    full_map = np.full((n_complete, bins), np.nan, dtype=float)
    truncated_map = np.full_like(full_map, np.nan)
    full_binary = np.zeros(n_complete, dtype=int)
    pre_reward_binary = np.zeros(n_complete, dtype=int)
    roles = np.full(n_complete, ROLE_OTHER, dtype=int)
    rewarded_goal = np.zeros(n_complete, dtype=bool)
    reward_cutoffs = np.full(n_complete, -1, dtype=int)

    for occurrence in range(n_complete):
        entry = int(entries[occurrence])
        exit_ = int(exits[occurrence])

        if occurrence in goal_idx:
            roles[occurrence] = ROLE_GOAL
        elif occurrence in nongoal_idx:
            roles[occurrence] = ROLE_NONGOAL
        elif occurrence in test_idx:
            roles[occurrence] = ROLE_TEST

        rewards_here = reward_idx[
            (reward_idx >= entry) & (reward_idx <= exit_)
        ]
        cutoff = None
        if roles[occurrence] == ROLE_GOAL and len(rewards_here):
            cutoff = int(rewards_here[0])
            rewarded_goal[occurrence] = True
            reward_cutoffs[occurrence] = cutoff

        # Preserve the exact full-landmark heatmap calculation used by the
        # original notebook's Behaviour.get_lm_lick_rate method.
        full_map[occurrence] = _bin_full_landmark_signal(
            signal, entry, exit_, bins
        )
        truncated_map[occurrence] = _bin_landmark_signal(
            signal, entry, exit_, bins, cutoff=cutoff
        )

        observed_exit = exit_ if cutoff is None else cutoff
        full_binary[occurrence] = int(
            np.any(signal[entry : exit_ + 1] > 0)
        )
        pre_reward_binary[occurrence] = int(
            np.any(signal[entry : observed_exit + 1] > 0)
        )

    n_laps = n_complete // n_landmarks
    shape_3d = (n_laps, n_landmarks, bins)
    shape_2d = (n_laps, n_landmarks)

    return {
        "full_lick_rate": full_map.reshape(shape_3d),
        "pre_reward_lick_rate": truncated_map.reshape(shape_3d),
        "full_binary_lick": full_binary.reshape(shape_2d),
        "pre_reward_binary_lick": pre_reward_binary.reshape(shape_2d),
        "role_codes": roles.reshape(shape_2d),
        "rewarded_goal": rewarded_goal.reshape(shape_2d),
        "landmark_ids": all_lms[:n_complete].reshape(shape_2d),
        "reward_cutoff_idx": reward_cutoffs.reshape(shape_2d),
        "bins": bins,
    }


def _mean_selected(
    lick_map: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Mean across selected landmarks and bins for every lap."""

    values = np.where(mask[..., None], lick_map, np.nan)
    counts = np.sum(np.isfinite(values), axis=(1, 2))
    totals = np.nansum(values, axis=(1, 2))
    result = np.full(lick_map.shape[0], np.nan, dtype=float)
    np.divide(totals, counts, out=result, where=counts > 0)
    return result


def calculate_lap_category_rates(
    lick_map: np.ndarray,
    role_codes: np.ndarray,
    rewarded_goal: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Reduce a lap × landmark × bin map to category rates per lap."""

    lick_map = np.asarray(lick_map, dtype=float)
    roles = np.asarray(role_codes, dtype=int)
    if lick_map.ndim != 3:
        raise ValueError("lick_map must have shape laps × landmarks × bins")
    if roles.shape != lick_map.shape[:2]:
        raise ValueError("role_codes must match the first two lick_map axes")

    result = {
        "goal": _mean_selected(lick_map, roles == ROLE_GOAL),
        "non_goal": _mean_selected(lick_map, roles == ROLE_NONGOAL),
        "test": _mean_selected(lick_map, roles == ROLE_TEST),
    }

    if rewarded_goal is not None:
        rewarded = np.asarray(rewarded_goal, dtype=bool)
        if rewarded.shape != roles.shape:
            raise ValueError("rewarded_goal must match role_codes")
        goal_mask = roles == ROLE_GOAL
        result["rewarded_goal"] = _mean_selected(
            lick_map, goal_mask & rewarded
        )
        result["missed_goal"] = _mean_selected(
            lick_map, goal_mask & ~rewarded
        )

    return result


def calculate_lap_binary_rates(
    binary_lick_map: np.ndarray,
    role_codes: np.ndarray,
    rewarded_goal: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Calculate the fraction of each landmark category licked per lap."""

    binary = np.asarray(binary_lick_map, dtype=float)
    roles = np.asarray(role_codes, dtype=int)
    if binary.ndim != 2:
        raise ValueError("binary_lick_map must have shape laps × landmarks")
    if roles.shape != binary.shape:
        raise ValueError("role_codes must match binary_lick_map")

    # Reuse the category reducer with a singleton measurement axis.
    return calculate_lap_category_rates(
        binary[..., None], roles, rewarded_goal
    )


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.full(len(values), np.nan, dtype=float)
    for end in range(window - 1, len(values)):
        current = values[end - window + 1 : end + 1]
        if np.any(np.isfinite(current)):
            result[end] = np.nanmean(current)
    return result


def detect_discrimination_onset(
    lap_rates: Mapping[str, np.ndarray],
    *,
    window: int = 10,
    persistence: int = 3,
    min_effect: float = 0.0,
    min_goal_fraction: float | None = 0.25,
) -> dict[str, Any]:
    """Find the first sustained goal-selective rolling window.

    Onset is reported as a one-based lap number at the end of the first
    qualifying window. A window qualifies when goal licking exceeds both
    non-goal and test licking by ``min_effect`` and the animal remains engaged.
    """

    if window < 1 or persistence < 1:
        raise ValueError("window and persistence must be positive")

    goal = np.asarray(lap_rates["goal"], dtype=float)
    non_goal = np.asarray(lap_rates["non_goal"], dtype=float)
    test = np.asarray(lap_rates["test"], dtype=float)
    if not (len(goal) == len(non_goal) == len(test)):
        raise ValueError("All lap-rate traces must have equal length")

    rolling_goal = _rolling_mean(goal, window)
    rolling_non_goal = _rolling_mean(non_goal, window)
    rolling_test = _rolling_mean(test, window)
    goal_vs_nongoal = _rolling_mean(goal - non_goal, window)
    goal_vs_test = _rolling_mean(goal - test, window)

    if min_goal_fraction is None:
        engagement_threshold = -np.inf
    else:
        if not 0 <= min_goal_fraction <= 1:
            raise ValueError("min_goal_fraction must lie between 0 and 1")
        maximum = np.nanmax(rolling_goal) if np.any(
            np.isfinite(rolling_goal)
        ) else np.nan
        engagement_threshold = min_goal_fraction * maximum

    criterion = (
        (goal_vs_nongoal > min_effect)
        & (goal_vs_test > min_effect)
        & (rolling_goal >= engagement_threshold)
    )

    onset_index = None
    run_length = 0
    for index, qualifies in enumerate(criterion):
        run_length = run_length + 1 if qualifies else 0
        if run_length >= persistence:
            onset_index = index - persistence + 1
            break

    return {
        "onset_index": onset_index,
        "onset_lap": None if onset_index is None else onset_index + 1,
        "criterion": criterion,
        "rolling_goal": rolling_goal,
        "rolling_non_goal": rolling_non_goal,
        "rolling_test": rolling_test,
        "goal_vs_nongoal": goal_vs_nongoal,
        "goal_vs_test": goal_vs_test,
        "engagement_threshold": engagement_threshold,
        "window": window,
        "persistence": persistence,
        "min_effect": min_effect,
    }


def detect_legacy_phase_boundaries(
    lap_rates: Mapping[str, np.ndarray],
    *,
    sigma: float = 2,
    start_lap: int = 10,
    gap: int = 15,
) -> dict[str, Any]:
    """Reproduce the percentile-based phase rules used in the old notebook.

    The returned lap numbers are one-based. Missing crossings are returned as
    ``None`` rather than being silently assigned to the last recorded lap.
    """

    non_goal = gaussian_filter1d(
        np.asarray(lap_rates["non_goal"], dtype=float), sigma=sigma
    )
    test = gaussian_filter1d(
        np.asarray(lap_rates["test"], dtype=float), sigma=sigma
    )
    if len(non_goal) != len(test):
        raise ValueError("All lap-rate traces must have equal length")

    exploration_threshold = np.nanpercentile(non_goal, 75)
    exploration_candidates = np.flatnonzero(
        non_goal[start_lap:] <= exploration_threshold
    )
    exploration_index = (
        None
        if len(exploration_candidates) == 0
        else int(exploration_candidates[0] + start_lap)
    )

    discrimination_threshold = np.nanmean(non_goal)
    discrimination_candidates = np.flatnonzero(
        test[start_lap:] <= discrimination_threshold
    )
    discrimination_index = None
    if len(discrimination_candidates):
        jumps = np.flatnonzero(np.diff(discrimination_candidates) > gap)
        first_in_last_block = (
            discrimination_candidates[jumps[-1] + 1]
            if len(jumps)
            else discrimination_candidates[0]
        )
        discrimination_index = int(first_in_last_block + start_lap)

    return {
        "exploration_end_index": exploration_index,
        "exploration_end_lap": (
            None if exploration_index is None else exploration_index + 1
        ),
        "discrimination_onset_index": discrimination_index,
        "discrimination_onset_lap": (
            None
            if discrimination_index is None
            else discrimination_index + 1
        ),
        "exploration_threshold": exploration_threshold,
        "discrimination_threshold": discrimination_threshold,
        "non_goal_smoothed": non_goal,
        "test_smoothed": test,
    }


def empty_cache() -> dict[str, Any]:
    return {"schema_version": CACHE_SCHEMA_VERSION, "records": {}}


def load_lick_cache(path: str | Path) -> dict[str, Any]:
    """Load a local NPZ cache, or return an empty cache."""

    path = Path(path)
    if not path.exists():
        return empty_cache()

    with np.load(path, allow_pickle=True) as archive:
        cache = archive["cache"].item()
    if cache.get("schema_version") != CACHE_SCHEMA_VERSION:
        return empty_cache()
    cache.setdefault("records", {})
    return cache


def cache_key(sequence: str, session_id: str) -> str:
    return f"{sequence}::{session_id}"


def get_cached_record(
    cache: Mapping[str, Any],
    *,
    sequence: str,
    session_id: str,
    source_path: str | Path,
    bins: int,
) -> dict[str, Any] | None:
    """Return a record only when its derivation metadata still matches."""

    record = cache.get("records", {}).get(cache_key(sequence, session_id))
    if record is None:
        return None
    source_path = Path(source_path)
    expected_mtime = source_path.stat().st_mtime_ns
    if int(record.get("bins", -1)) != bins:
        return None
    if record.get("source_path") != str(source_path):
        return None
    if int(record.get("source_mtime_ns", -1)) != expected_mtime:
        return None
    return record


def put_cached_record(
    cache: dict[str, Any],
    *,
    sequence: str,
    session_id: str,
    source_path: str | Path,
    record: Mapping[str, Any],
) -> None:
    source_path = Path(source_path)
    stored = dict(record)
    stored.update(
        {
            "sequence": sequence,
            "session_id": session_id,
            "source_path": str(source_path),
            "source_mtime_ns": source_path.stat().st_mtime_ns,
        }
    )
    cache.setdefault("records", {})[
        cache_key(sequence, session_id)
    ] = stored


def save_lick_cache(path: str | Path, cache: Mapping[str, Any]) -> None:
    """Atomically rewrite an NPZ cache after records have been appended."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=".npz", dir=path.parent
    )
    os.close(fd)
    try:
        np.savez_compressed(
            temporary_name,
            cache=np.array(dict(cache), dtype=object),
        )
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
