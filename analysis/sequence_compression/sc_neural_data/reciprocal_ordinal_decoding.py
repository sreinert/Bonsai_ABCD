"""Reciprocal first-versus-second decoding across ABB and AAB protocols.

The functions here operate on populations produced by
``cross_protocol_decoding.build_protocol_population``.  Complete adjacent
first/second pairs are the unit of analysis and of label permutation.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TRIAL_FIELDS = (
    "landmark_duration_frames",
    "visit_was_licked",
    "visit_was_licked_in_search_window",
    "visit_had_any_lick",
    "visit_mean_speed",
)


def repeated_element_labels(protocol):
    """Return ``(singleton, first, second)`` labels for one protocol."""
    if protocol == "ABB":
        return "A", "B1", "B2"
    if protocol == "AAB":
        return "B", "A1", "A2"
    raise ValueError("protocol must be 'ABB' or 'AAB'")


def _complete_pair_rows(fine_labels, source_visit_indices, first, second):
    fine_labels = np.asarray(fine_labels, dtype=str)
    source_visit_indices = np.asarray(source_visit_indices, dtype=int)
    if fine_labels.ndim != 1 or source_visit_indices.shape != fine_labels.shape:
        raise ValueError("fine labels and source visit indices must be aligned")
    if np.unique(source_visit_indices).size != source_visit_indices.size:
        raise ValueError("source visit indices must be unique")

    by_source = {source: row for row, source in enumerate(source_visit_indices)}
    pairs = []
    for first_row in np.flatnonzero(fine_labels == first):
        second_row = by_source.get(int(source_visit_indices[first_row]) + 1)
        if second_row is not None and fine_labels[second_row] == second:
            pairs.append((int(first_row), int(second_row)))
    if not pairs:
        raise ValueError(f"no complete adjacent {first}/{second} pairs")
    return np.asarray(pairs, dtype=int)


def make_repeated_element_population(population, protocol):
    """Select complete first/second pairs and label the second element positive."""
    _, first, second = repeated_element_labels(protocol)
    pairs = _complete_pair_rows(
        population["fine_labels"],
        population["source_visit_indices"],
        first,
        second,
    )
    rows = pairs.reshape(-1)
    X = np.asarray(population["X"], dtype=float)
    cluster_ids = np.asarray(population["cluster_ids"])
    if X.ndim != 2 or X.shape[1] != cluster_ids.size:
        raise ValueError("population X and cluster IDs are incompatible")
    result = {
        "X": X[rows],
        "y": np.tile([0, 1], len(pairs)),
        "labels": np.tile([first, second], len(pairs)),
        "pair_ids": np.repeat(np.arange(len(pairs)), 2),
        "source_visit_indices": np.asarray(
            population["source_visit_indices"], dtype=int
        )[rows],
        "cluster_ids": cluster_ids.copy(),
        "classes": (first, second),
        "n_unpaired_repeated_visits": int(
            np.sum(np.isin(population["fine_labels"], [first, second]))
            - 2 * len(pairs)
        ),
    }
    for field in TRIAL_FIELDS:
        if field in population:
            values = np.asarray(population[field])
            if values.shape != (len(population["X"]),):
                raise ValueError(f"population field {field!r} is misaligned")
            result[field] = values[rows]
    return result


def make_singleton_repeated_population(population, protocol):
    """Select complete triplets and label singleton=0, repeated elements=1."""
    singleton, first, second = repeated_element_labels(protocol)
    labels = np.asarray(population["fine_labels"], dtype=str)
    sources = np.asarray(population["source_visit_indices"], dtype=int)
    if labels.ndim != 1 or sources.shape != labels.shape:
        raise ValueError("fine labels and source visit indices must be aligned")
    by_source = {source: row for row, source in enumerate(sources)}
    triplets = []
    for singleton_row in np.flatnonzero(labels == singleton):
        source = int(sources[singleton_row])
        first_row = by_source.get(source + 1)
        second_row = by_source.get(source + 2)
        if (
            first_row is not None
            and second_row is not None
            and labels[first_row] == first
            and labels[second_row] == second
        ):
            triplets.append((int(singleton_row), int(first_row), int(second_row)))
    if not triplets:
        raise ValueError(
            f"no complete adjacent {singleton}/{first}/{second} triplets"
        )
    triplets = np.asarray(triplets, dtype=int)
    rows = triplets.reshape(-1)
    X = np.asarray(population["X"], dtype=float)
    cluster_ids = np.asarray(population["cluster_ids"])
    if X.ndim != 2 or X.shape[1] != cluster_ids.size:
        raise ValueError("population X and cluster IDs are incompatible")
    result = {
        "X": X[rows],
        "y": np.tile([0, 1, 1], len(triplets)),
        "labels": labels[rows],
        "role_labels": np.tile(
            ["singleton", "repeated", "repeated"], len(triplets)
        ),
        "cycle_ids": np.repeat(np.arange(len(triplets)), 3),
        "source_visit_indices": sources[rows],
        "cluster_ids": cluster_ids.copy(),
        "classes": ("singleton", "repeated"),
    }
    for field in TRIAL_FIELDS:
        if field in population:
            values = np.asarray(population[field])
            if values.shape != (len(population["X"]),):
                raise ValueError(f"population field {field!r} is misaligned")
            result[field] = values[rows]
    return result


def filter_repeated_elements_by_lick(
    population, *, licked, lick_field="visit_was_licked"
):
    """Keep singleton visits and repeated visits with one lick status.

    A cycle is retained only when it contains its singleton and at least one
    repeated visit matching ``licked`` in ``lick_field``.  This keeps every
    retained source cycle informative and permits a cycle-preserving
    role-label permutation.
    """
    if lick_field not in population:
        raise ValueError(f"population lacks {lick_field}")
    y = np.asarray(population["y"], dtype=int)
    cycle_ids = np.asarray(population["cycle_ids"], dtype=int)
    was_licked = np.asarray(population[lick_field], dtype=bool)
    if not (y.shape == cycle_ids.shape == was_licked.shape):
        raise ValueError("role, cycle, and lick fields must be aligned")

    retained_rows = []
    excluded_cycles = 0
    for cycle in np.unique(cycle_ids):
        rows = np.flatnonzero(cycle_ids == cycle)
        singleton_rows = rows[y[rows] == 0]
        repeated_rows = rows[(y[rows] == 1) & (was_licked[rows] == bool(licked))]
        if singleton_rows.size != 1:
            raise ValueError("every cycle must contain exactly one singleton")
        if repeated_rows.size == 0:
            excluded_cycles += 1
            continue
        retained_rows.extend(singleton_rows.tolist())
        retained_rows.extend(repeated_rows.tolist())
    if not retained_rows:
        condition = "licked" if licked else "non-licked"
        raise ValueError(f"no cycles contain a {condition} repeated visit")

    retained_rows = np.asarray(retained_rows, dtype=int)
    retained_old_cycles = cycle_ids[retained_rows]
    old_to_new = {
        old: new for new, old in enumerate(np.unique(retained_old_cycles))
    }
    result = {}
    n_trials = len(y)
    for key, value in population.items():
        array = np.asarray(value) if isinstance(value, (list, tuple, np.ndarray)) else None
        if array is not None and array.ndim >= 1 and array.shape[0] == n_trials:
            result[key] = array[retained_rows]
        else:
            result[key] = value
    result["cycle_ids"] = np.asarray(
        [old_to_new[cycle] for cycle in retained_old_cycles], dtype=int
    )
    result["repeated_lick_mode"] = "licked" if licked else "non-licked"
    result["repeated_lick_field"] = lick_field
    result["n_cycles_excluded_no_matching_repeated"] = excluded_cycles
    return result


def filter_binary_population_by_lick(
    population,
    *,
    licked,
    lick_field="visit_was_licked",
):
    """Filter both classes of a binary population by one lick condition."""
    if lick_field not in population:
        raise ValueError(f"population lacks {lick_field}")
    y = np.asarray(population["y"], dtype=int)
    was_licked = np.asarray(population[lick_field], dtype=bool)
    if y.shape != was_licked.shape:
        raise ValueError("class and lick fields must be aligned")
    keep = was_licked == bool(licked)
    if not np.any(keep):
        condition = "licked" if licked else "non-licked"
        raise ValueError(f"no {condition} visits")

    result = {}
    n_trials = len(y)
    for key, value in population.items():
        array = (
            np.asarray(value)
            if isinstance(value, (list, tuple, np.ndarray))
            else None
        )
        if array is not None and array.ndim >= 1 and array.shape[0] == n_trials:
            result[key] = array[keep]
        else:
            result[key] = value
    if not np.array_equal(np.unique(result["y"]), [0, 1]):
        counts = pd.Series(result["y"]).value_counts().to_dict()
        condition = "licked" if licked else "non-licked"
        raise ValueError(
            f"{condition} filtering removed one decoder class: {counts}"
        )
    result["lick_mode"] = "licked" if licked else "non-licked"
    result["lick_field"] = lick_field
    return result


def _subset_population_rows(population, rows):
    rows = np.asarray(rows, dtype=int)
    n_trials = len(population["y"])
    result = {}
    for key, value in population.items():
        array = (
            np.asarray(value)
            if isinstance(value, (list, tuple, np.ndarray))
            else None
        )
        if array is not None and array.ndim >= 1 and array.shape[0] == n_trials:
            result[key] = array[rows]
        else:
            result[key] = value
    return result


def make_ordinal_binary_population(
    population,
    protocol,
    *,
    lick_mode="all",
    lick_field="visit_was_licked",
):
    """Build a first-versus-second population for ABB or AAB visits."""
    _, first, second = repeated_element_labels(protocol)
    fine = np.asarray(population["fine_labels"], dtype=str)
    keep = np.isin(fine, [first, second])
    if lick_mode != "all":
        if lick_mode not in {"licked", "non-licked"}:
            raise ValueError("lick_mode must be 'all', 'licked', or 'non-licked'")
        if lick_field not in population:
            raise ValueError(f"population lacks {lick_field}")
        was_licked = np.asarray(population[lick_field], dtype=bool)
        keep &= was_licked == (lick_mode == "licked")
    rows = np.flatnonzero(keep)
    labels = fine[rows]
    if not {first, second}.issubset(set(labels)):
        raise ValueError(
            f"{lick_mode} filtering removed {first} or {second}: "
            f"{pd.Series(labels).value_counts().to_dict()}"
        )
    result = _subset_population_rows(population, rows)
    result["labels"] = labels
    result["y"] = (labels == second).astype(int)
    result["classes"] = ("first", "second")
    result["fine_classes"] = (first, second)
    result["lick_mode"] = lick_mode
    return result


def make_structural_binary_population(
    population,
    protocol,
    *,
    abb_b_lick_mode="all",
    lick_field="visit_was_licked",
):
    """Build singleton-versus-doubleton classes across ABB or AAB.

    In ABB, every A visit is retained while B1/B2 can be restricted by lick
    status.  In AAB, all B singleton and A1/A2 doubleton visits are retained.
    """
    fine = np.asarray(population["fine_labels"], dtype=str)
    if protocol == "ABB":
        keep = fine == "A"
        b_visit = np.isin(fine, ["B1", "B2"])
        if abb_b_lick_mode == "all":
            keep |= b_visit
        else:
            if abb_b_lick_mode not in {"licked", "non-licked"}:
                raise ValueError(
                    "abb_b_lick_mode must be 'all', 'licked', or 'non-licked'"
                )
            if lick_field not in population:
                raise ValueError(f"population lacks {lick_field}")
            was_licked = np.asarray(population[lick_field], dtype=bool)
            keep |= b_visit & (
                was_licked == (abb_b_lick_mode == "licked")
            )
        singleton = fine == "A"
    elif protocol == "AAB":
        if abb_b_lick_mode != "all":
            raise ValueError("AAB structural populations are not lick-filtered")
        keep = np.isin(fine, ["B", "A1", "A2"])
        singleton = fine == "B"
    else:
        raise ValueError("protocol must be 'ABB' or 'AAB'")

    rows = np.flatnonzero(keep)
    y = (~singleton[rows]).astype(int)
    if not np.array_equal(np.unique(y), [0, 1]):
        raise ValueError("structural filtering removed singleton or doubleton")
    result = _subset_population_rows(population, rows)
    result["y"] = y
    result["labels"] = np.where(y == 0, "singleton", "doubleton")
    result["classes"] = ("singleton", "doubleton")
    result["abb_b_lick_mode"] = abb_b_lick_mode
    return result


def _shared_columns(source, target, minimum_cells):
    source_ids = np.asarray(source["cluster_ids"])
    target_ids = np.asarray(target["cluster_ids"])
    shared = np.intersect1d(source_ids, target_ids)
    if shared.size < minimum_cells:
        raise ValueError(
            f"Only {shared.size} shared cells; minimum_cells={minimum_cells}"
        )
    source_lookup = {cluster_id: i for i, cluster_id in enumerate(source_ids)}
    target_lookup = {cluster_id: i for i, cluster_id in enumerate(target_ids)}
    return (
        np.asarray(source["X"], dtype=float)[
            :, [source_lookup[cluster_id] for cluster_id in shared]
        ],
        np.asarray(target["X"], dtype=float)[
            :, [target_lookup[cluster_id] for cluster_id in shared]
        ],
        shared,
    )


def _classifier(C, random_state):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=C,
            class_weight="balanced",
            solver="liblinear",
            random_state=random_state,
        ),
    )


def _swap_labels_within_pairs(y, pair_ids, swaps):
    y = np.asarray(y, dtype=int)
    pair_ids = np.asarray(pair_ids, dtype=int)
    swaps = np.asarray(swaps, dtype=bool)
    if swaps.shape != (np.unique(pair_ids).size,):
        raise ValueError("one swap flag is required per pair")
    return np.where(swaps[pair_ids], 1 - y, y)


def reciprocal_transfer(
    source,
    target,
    *,
    minimum_cells=5,
    C=1.0,
    n_permutations=1000,
    random_state=0,
):
    """Fit on one protocol and test first/second separation in the other.

    The target ROC AUC and mean within-pair score difference are directional:
    values above 0.5 and 0 respectively indicate a more second-like score for
    the target protocol's second repeated element.  Permutations independently
    swap the two source labels inside each complete source pair.
    """
    if minimum_cells < 1 or n_permutations < 0:
        raise ValueError("minimum_cells must be positive and permutations non-negative")
    X_source, X_target, shared = _shared_columns(
        source, target, minimum_cells
    )
    y_source = np.asarray(source["y"], dtype=int)
    y_target = np.asarray(target["y"], dtype=int)
    target_pairs = np.asarray(target["pair_ids"], dtype=int)
    if not np.array_equal(np.unique(y_source), [0, 1]):
        raise ValueError("source must contain both repeated-element classes")
    if not np.array_equal(np.unique(y_target), [0, 1]):
        raise ValueError("target must contain both repeated-element classes")

    model = _classifier(C, random_state).fit(X_source, y_source)
    target_score = model.decision_function(X_target)
    target_auc = float(roc_auc_score(y_target, target_score))
    pair_ids = np.unique(target_pairs)
    pair_differences = np.asarray([
        np.mean(target_score[(target_pairs == pair) & (y_target == 1)])
        - np.mean(target_score[(target_pairs == pair) & (y_target == 0)])
        for pair in pair_ids
    ])
    mean_pair_difference = float(np.mean(pair_differences))

    rng = np.random.default_rng(random_state)
    source_pairs = np.asarray(source["pair_ids"], dtype=int)
    n_source_pairs = np.unique(source_pairs).size
    null_auc = np.empty(n_permutations, dtype=float)
    null_pair_difference = np.empty(n_permutations, dtype=float)
    for permutation in range(n_permutations):
        shuffled_y = _swap_labels_within_pairs(
            y_source,
            source_pairs,
            rng.integers(0, 2, size=n_source_pairs).astype(bool),
        )
        null_model = _classifier(C, random_state + permutation + 1).fit(
            X_source, shuffled_y
        )
        null_score = null_model.decision_function(X_target)
        null_auc[permutation] = roc_auc_score(y_target, null_score)
        null_pair_difference[permutation] = np.mean([
            np.mean(null_score[(target_pairs == pair) & (y_target == 1)])
            - np.mean(null_score[(target_pairs == pair) & (y_target == 0)])
            for pair in pair_ids
        ])

    return {
        "model": model,
        "shared_cluster_ids": shared,
        "n_cells": int(shared.size),
        "n_source_pairs": int(n_source_pairs),
        "n_target_pairs": int(pair_ids.size),
        "target_y": y_target,
        "target_labels": np.asarray(target["labels"]),
        "target_pair_ids": target_pairs,
        "target_source_visit_indices": np.asarray(
            target["source_visit_indices"], dtype=int
        ),
        "target_score": target_score,
        "target_auc": target_auc,
        "pair_differences": pair_differences,
        "mean_pair_difference": mean_pair_difference,
        "null_auc": null_auc,
        "null_mean_pair_difference": null_pair_difference,
        "auc_permutation_p": float(
            (1 + np.sum(null_auc >= target_auc)) / (n_permutations + 1)
        ),
        "pair_difference_permutation_p": float(
            (1 + np.sum(null_pair_difference >= mean_pair_difference))
            / (n_permutations + 1)
        ),
    }


def singleton_repeated_transfer(
    source,
    target,
    *,
    minimum_cells=5,
    C=1.0,
    n_permutations=1000,
    random_state=0,
):
    """Transfer a singleton-versus-repeated decoder across protocols.

    Every source permutation randomly assigns the singleton role to one of the
    three visits in each complete triplet.  This preserves the one-versus-two
    class count within every cycle.  ``identity_auc`` evaluates the opposite
    target mapping (B-positive sensory identity) using the same fixed scores.
    """
    if minimum_cells < 1 or n_permutations < 0:
        raise ValueError("minimum_cells must be positive and permutations non-negative")
    X_source, X_target, shared = _shared_columns(
        source, target, minimum_cells
    )
    y_source = np.asarray(source["y"], dtype=int)
    y_target = np.asarray(target["y"], dtype=int)
    source_cycles = np.asarray(source["cycle_ids"], dtype=int)
    target_cycles = np.asarray(target["cycle_ids"], dtype=int)
    if not np.array_equal(np.unique(y_source), [0, 1]):
        raise ValueError("source must contain singleton and repeated visits")
    if not np.array_equal(np.unique(y_target), [0, 1]):
        raise ValueError("target must contain singleton and repeated visits")

    model = _classifier(C, random_state).fit(X_source, y_source)
    target_score = model.decision_function(X_target)
    role_auc = float(roc_auc_score(y_target, target_score))
    identity_auc = float(roc_auc_score(1 - y_target, target_score))
    target_cycle_ids = np.unique(target_cycles)
    cycle_differences = np.asarray([
        np.mean(target_score[(target_cycles == cycle) & (y_target == 1)])
        - target_score[(target_cycles == cycle) & (y_target == 0)][0]
        for cycle in target_cycle_ids
    ])
    mean_cycle_difference = float(np.mean(cycle_differences))

    rng = np.random.default_rng(random_state)
    source_cycle_ids = np.unique(source_cycles)
    null_role_auc = np.empty(n_permutations, dtype=float)
    null_identity_auc = np.empty(n_permutations, dtype=float)
    null_cycle_difference = np.empty(n_permutations, dtype=float)
    for permutation in range(n_permutations):
        shuffled_y = np.ones_like(y_source)
        for cycle in source_cycle_ids:
            rows = np.flatnonzero(source_cycles == cycle)
            if rows.size < 2:
                raise ValueError(
                    "every source cycle must contain a singleton and at least "
                    "one repeated visit"
                )
            shuffled_y[rng.choice(rows)] = 0
        null_model = _classifier(C, random_state + permutation + 1).fit(
            X_source, shuffled_y
        )
        null_score = null_model.decision_function(X_target)
        null_role_auc[permutation] = roc_auc_score(y_target, null_score)
        null_identity_auc[permutation] = roc_auc_score(1 - y_target, null_score)
        null_cycle_difference[permutation] = np.mean([
            np.mean(null_score[(target_cycles == cycle) & (y_target == 1)])
            - null_score[(target_cycles == cycle) & (y_target == 0)][0]
            for cycle in target_cycle_ids
        ])

    return {
        "model": model,
        "shared_cluster_ids": shared,
        "n_cells": int(shared.size),
        "n_source_cycles": int(source_cycle_ids.size),
        "n_target_cycles": int(target_cycle_ids.size),
        "target_y": y_target,
        "target_labels": np.asarray(target["labels"]),
        "target_role_labels": np.asarray(target["role_labels"]),
        "target_cycle_ids": target_cycles,
        "target_score": target_score,
        "role_auc": role_auc,
        "identity_auc": identity_auc,
        "cycle_differences": cycle_differences,
        "mean_cycle_difference": mean_cycle_difference,
        "null_role_auc": null_role_auc,
        "null_identity_auc": null_identity_auc,
        "null_mean_cycle_difference": null_cycle_difference,
        "role_auc_permutation_p": float(
            (1 + np.sum(null_role_auc >= role_auc)) / (n_permutations + 1)
        ),
        "identity_auc_permutation_p": float(
            (1 + np.sum(null_identity_auc >= identity_auc))
            / (n_permutations + 1)
        ),
        "cycle_difference_permutation_p": float(
            (1 + np.sum(null_cycle_difference >= mean_cycle_difference))
            / (n_permutations + 1)
        ),
    }


def holm_adjust(p_values: Sequence[float]):
    """Return Holm-adjusted p-values in the input order."""
    p_values = np.asarray(p_values, dtype=float)
    if p_values.ndim != 1 or np.any(~np.isfinite(p_values)):
        raise ValueError("p_values must be a finite one-dimensional sequence")
    if np.any((p_values < 0) | (p_values > 1)):
        raise ValueError("p-values must lie between zero and one")
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted_ranked = np.maximum.accumulate(
        (len(ranked) - np.arange(len(ranked))) * ranked
    )
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def behavior_summary(population, *, frame_rate):
    """Summarize lick, duration, and speed for each repeated-element label."""
    rows = []
    for code, label in enumerate(population["classes"]):
        selected = np.asarray(population["y"]) == code
        row = {"label": label, "visits": int(np.sum(selected))}
        if "visit_was_licked" in population:
            row["lick fraction"] = float(np.mean(
                np.asarray(population["visit_was_licked"])[selected]
            ))
        if "landmark_duration_frames" in population:
            duration = (
                np.asarray(population["landmark_duration_frames"])[selected]
                / float(frame_rate)
            )
            row["mean duration (s)"] = float(np.mean(duration))
            row["median duration (s)"] = float(np.median(duration))
        if "visit_mean_speed" in population:
            speed = np.asarray(population["visit_mean_speed"])[selected]
            row["mean speed"] = float(np.mean(speed))
            row["median speed"] = float(np.median(speed))
        rows.append(row)
    return pd.DataFrame(rows)


def find_sequence_cycles(prepared):
    """Find complete singleton/first/second triples in original visit order."""
    singleton, first, second = repeated_element_labels(prepared["protocol"])
    labels = np.asarray(prepared["fine_labels"], dtype=str)
    sources = np.asarray(prepared["source_visit_indices"], dtype=int)
    cycles = []
    for row in range(len(labels) - 2):
        if labels[row:row + 3].tolist() != [singleton, first, second]:
            continue
        if not np.array_equal(sources[row:row + 3], sources[row] + np.arange(3)):
            continue
        cycles.append((row, row + 1, row + 2))
    if not cycles:
        raise ValueError("no complete singleton/first/second cycles")
    return np.asarray(cycles, dtype=int)


def score_cycle_trajectories(
    prepared,
    model,
    shared_cluster_ids,
    *,
    frame_rate,
    window_seconds=0.5,
    step_seconds=0.1,
):
    """Project forward sliding windows from singleton exit to second exit."""
    if window_seconds <= 0 or step_seconds <= 0:
        raise ValueError("window_seconds and step_seconds must be positive")
    signal = np.asarray(prepared["selected_dff"], dtype=float)
    cluster_ids = np.asarray(prepared["cluster_ids"])
    shared_cluster_ids = np.asarray(shared_cluster_ids)
    lookup = {cluster_id: row for row, cluster_id in enumerate(cluster_ids)}
    if not set(shared_cluster_ids).issubset(lookup):
        raise ValueError("prepared session is missing shared decoder cells")
    signal = signal[[lookup[cluster_id] for cluster_id in shared_cluster_ids]]
    means = np.nanmean(signal, axis=1, keepdims=True)
    scales = np.nanstd(signal, axis=1, keepdims=True)
    if np.any(~np.isfinite(scales)) or np.any(scales == 0):
        raise ValueError("every decoder cell must have finite, non-zero variance")
    signal = (signal - means) / scales

    entries = np.asarray(prepared["entry_frames"], dtype=int)
    exits = np.asarray(prepared["exit_frames"], dtype=int)
    _, first_label, second_label = repeated_element_labels(prepared["protocol"])
    window_frames = max(1, int(round(window_seconds * frame_rate)))
    step_frames = max(1, int(round(step_seconds * frame_rate)))
    trajectory_rows = []
    event_rows = []
    for cycle_id, (singleton, first, second) in enumerate(
        find_sequence_cycles(prepared)
    ):
        origin = exits[singleton]
        event_frames = {
            f"{first_label} entry": entries[first],
            f"{first_label} exit": exits[first],
            f"{second_label} entry": entries[second],
            f"{second_label} exit": exits[second],
        }
        for event, frame in event_frames.items():
            event_rows.append({
                "cycle": cycle_id,
                "event": event,
                "time_seconds": (frame - origin) / float(frame_rate),
            })
        last_start = exits[second] - window_frames + 1
        for start in range(origin, last_start + 1, step_frames):
            stop = start + window_frames
            centre = start + (window_frames - 1) // 2
            feature = np.mean(signal[:, start:stop], axis=1)
            score = float(model.decision_function(feature[None, :])[0])
            if centre < entries[first]:
                phase = f"corridor to {first_label}"
            elif centre <= exits[first]:
                phase = first_label
            elif centre < entries[second]:
                phase = f"corridor to {second_label}"
            else:
                phase = second_label
            trajectory_rows.append({
                "cycle": cycle_id,
                "time_seconds": (start - origin) / float(frame_rate),
                "score": score,
                "phase": phase,
                "window_start_frame": start,
            })
    return pd.DataFrame(trajectory_rows), pd.DataFrame(event_rows)
