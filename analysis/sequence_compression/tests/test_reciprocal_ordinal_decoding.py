import numpy as np

from analysis.sequence_compression.sc_neural_data.reciprocal_ordinal_decoding import (
    find_sequence_cycles,
    filter_repeated_elements_by_lick,
    filter_binary_population_by_lick,
    holm_adjust,
    make_repeated_element_population,
    make_ordinal_binary_population,
    make_structural_binary_population,
    make_singleton_repeated_population,
    reciprocal_transfer,
    score_cycle_trajectories,
    singleton_repeated_transfer,
)


def _population(labels, source_indices, X, cluster_ids=(10, 20, 30)):
    labels = np.asarray(labels)
    return {
        "X": np.asarray(X, dtype=float),
        "fine_labels": labels,
        "source_visit_indices": np.asarray(source_indices, dtype=int),
        "cluster_ids": np.asarray(cluster_ids),
        "landmark_duration_frames": np.arange(len(labels)) + 5,
        "visit_was_licked": np.arange(len(labels)) % 2 == 0,
        "visit_mean_speed": np.arange(len(labels), dtype=float),
    }


def test_repeated_population_keeps_only_complete_adjacent_pairs():
    population = _population(
        ["A1", "A2", "B", "A1", "B", "A1", "A2"],
        np.arange(7),
        np.arange(21).reshape(7, 3),
    )
    result = make_repeated_element_population(population, "AAB")
    assert result["labels"].tolist() == ["A1", "A2", "A1", "A2"]
    assert result["source_visit_indices"].tolist() == [0, 1, 5, 6]
    assert result["pair_ids"].tolist() == [0, 0, 1, 1]
    assert result["n_unpaired_repeated_visits"] == 1
    assert result["visit_mean_speed"].tolist() == [0, 1, 5, 6]


def test_singleton_repeated_population_uses_complete_cross_boundary_triplets():
    population = _population(
        ["A1", "A2", "B", "A1", "A2", "B", "A1", "A2"],
        np.arange(8),
        np.arange(24).reshape(8, 3),
    )
    result = make_singleton_repeated_population(population, "AAB")
    assert result["labels"].tolist() == ["B", "A1", "A2"] * 2
    assert result["y"].tolist() == [0, 1, 1] * 2
    assert result["cycle_ids"].tolist() == [0, 0, 0, 1, 1, 1]


def test_lick_filter_keeps_singleton_and_matching_repeated_visits_per_cycle():
    population = _population(
        np.tile(["A", "B1", "B2"], 3),
        np.arange(9),
        np.arange(27).reshape(9, 3),
    )
    population["visit_was_licked"] = np.array([
        False, True, False,
        True, False, False,
        False, True, True,
    ])
    roles = make_singleton_repeated_population(population, "ABB")
    roles["visit_was_licked_in_search_window"] = np.logical_not(
        roles["visit_was_licked"]
    )
    licked = filter_repeated_elements_by_lick(roles, licked=True)
    assert licked["labels"].tolist() == ["A", "B1", "A", "B1", "B2"]
    assert licked["y"].tolist() == [0, 1, 0, 1, 1]
    assert licked["cycle_ids"].tolist() == [0, 0, 1, 1, 1]
    assert licked["n_cycles_excluded_no_matching_repeated"] == 1
    non_licked = filter_repeated_elements_by_lick(roles, licked=False)
    assert non_licked["labels"].tolist() == [
        "A", "B2", "A", "B1", "B2"
    ]
    assert non_licked["cycle_ids"].tolist() == [0, 0, 1, 1, 1]
    searched = filter_repeated_elements_by_lick(
        roles,
        licked=True,
        lick_field="visit_was_licked_in_search_window",
    )
    assert searched["labels"].tolist() == non_licked["labels"].tolist()
    assert searched["repeated_lick_field"] == (
        "visit_was_licked_in_search_window"
    )
    transfer = singleton_repeated_transfer(
        licked, roles, minimum_cells=3, n_permutations=3, random_state=8
    )
    assert transfer["n_source_cycles"] == 2
    assert len(transfer["null_role_auc"]) == 3


def test_binary_lick_filter_applies_to_both_decoder_classes():
    population = {
        "X": np.arange(18, dtype=float).reshape(6, 3),
        "y": np.array([0, 0, 0, 1, 1, 1]),
        "labels": np.array(["A", "A", "A", "B", "B", "B"]),
        "fine_labels": np.array(["A1", "A2", "A1", "B", "B", "B"]),
        "cluster_ids": np.array([10, 20, 30]),
        "classes": ("A", "B"),
        "visit_was_licked_in_search_window": np.array(
            [True, False, True, False, True, True]
        ),
    }
    licked = filter_binary_population_by_lick(
        population,
        licked=True,
        lick_field="visit_was_licked_in_search_window",
    )
    assert licked["labels"].tolist() == ["A", "A", "B", "B"]
    assert licked["y"].tolist() == [0, 0, 1, 1]
    non_licked = filter_binary_population_by_lick(
        population,
        licked=False,
        lick_field="visit_was_licked_in_search_window",
    )
    assert non_licked["labels"].tolist() == ["A", "B"]
    assert non_licked["lick_mode"] == "non-licked"


def test_clean_question_variants_pool_and_filter_expected_labels():
    abb = {
        "X": np.arange(18, dtype=float).reshape(6, 3),
        "y": np.array([0, 1, 1, 0, 1, 1]),
        "labels": np.array(["A", "B", "B", "A", "B", "B"]),
        "fine_labels": np.array(["A", "B1", "B2", "A", "B1", "B2"]),
        "cluster_ids": np.array([10, 20, 30]),
        "classes": ("A", "B"),
        "visit_was_licked_in_search_window": np.array(
            [True, True, False, False, False, True]
        ),
    }
    ordinal = make_ordinal_binary_population(
        abb,
        "ABB",
        lick_mode="licked",
        lick_field="visit_was_licked_in_search_window",
    )
    assert ordinal["fine_labels"].tolist() == ["B1", "B2"]
    assert ordinal["y"].tolist() == [0, 1]
    assert ordinal["classes"] == ("first", "second")

    structural = make_structural_binary_population(
        abb,
        "ABB",
        abb_b_lick_mode="non-licked",
        lick_field="visit_was_licked_in_search_window",
    )
    assert structural["fine_labels"].tolist() == ["A", "B2", "A", "B1"]
    assert structural["y"].tolist() == [0, 1, 0, 1]

    aab = dict(abb)
    aab["fine_labels"] = np.array(["A1", "A2", "B", "A1", "A2", "B"])
    aab["labels"] = np.array(["A", "A", "B", "A", "A", "B"])
    aab["y"] = np.array([0, 0, 1, 0, 0, 1])
    aab_ordinal = make_ordinal_binary_population(aab, "AAB")
    assert aab_ordinal["fine_labels"].tolist() == ["A1", "A2", "A1", "A2"]
    aab_structure = make_structural_binary_population(aab, "AAB")
    assert aab_structure["y"].tolist() == [1, 1, 0, 1, 1, 0]


def test_reciprocal_transfer_recovers_shared_ordinal_axis():
    rng = np.random.default_rng(12)
    source_labels = np.tile(["B1", "B2"], 20)
    target_labels = np.tile(["A1", "A2"], 18)
    source_y = np.tile([0, 1], 20)
    target_y = np.tile([0, 1], 18)
    weights = np.array([1.8, -1.2, 0.9])
    source_X = np.outer(2 * source_y - 1, weights) + rng.normal(
        0, 0.25, (source_y.size, weights.size)
    )
    target_X = np.outer(2 * target_y - 1, weights) + rng.normal(
        0, 0.25, (target_y.size, weights.size)
    )
    source = make_repeated_element_population(
        _population(source_labels, np.arange(len(source_labels)), source_X),
        "ABB",
    )
    target = make_repeated_element_population(
        _population(
            target_labels,
            np.arange(len(target_labels)),
            target_X[:, [2, 0, 1]],
            cluster_ids=(30, 10, 20),
        ),
        "AAB",
    )
    result = reciprocal_transfer(
        source,
        target,
        minimum_cells=3,
        n_permutations=49,
        random_state=4,
    )
    assert result["target_auc"] > 0.99
    assert result["mean_pair_difference"] > 0
    # In this deliberately near-one-dimensional example, permuted models can
    # retain the correct sign whenever more source pairs happen to remain
    # unswapped, making the AUC null discrete at zero and one.  The continuous
    # paired-score statistic still detects the much larger observed effect.
    assert result["pair_difference_permutation_p"] <= 2 / 50
    assert 0 < result["auc_permutation_p"] <= 1
    assert result["shared_cluster_ids"].tolist() == [10, 20, 30]


def test_singleton_repeated_transfer_distinguishes_role_from_identity():
    rng = np.random.default_rng(19)
    weights = np.array([1.5, -1.0, 0.7])
    source_y = np.tile([0, 1, 1], 20)
    target_y = np.tile([0, 1, 1], 18)
    source_X = np.outer(2 * source_y - 1, weights) + rng.normal(
        0, 0.3, (len(source_y), len(weights))
    )
    target_X = np.outer(2 * target_y - 1, weights) + rng.normal(
        0, 0.3, (len(target_y), len(weights))
    )
    source_labels = np.tile(["A", "B1", "B2"], 20)
    target_labels = np.tile(["B", "A1", "A2"], 18)
    source = make_singleton_repeated_population(
        _population(source_labels, np.arange(len(source_labels)), source_X),
        "ABB",
    )
    target = make_singleton_repeated_population(
        _population(target_labels, np.arange(len(target_labels)), target_X),
        "AAB",
    )
    result = singleton_repeated_transfer(
        source,
        target,
        minimum_cells=3,
        n_permutations=49,
        random_state=3,
    )
    assert result["role_auc"] > 0.99
    assert result["identity_auc"] < 0.01
    assert result["mean_cycle_difference"] > 0
    assert result["cycle_difference_permutation_p"] <= 2 / 50


def test_holm_adjust_restores_original_order_and_is_monotone():
    adjusted = holm_adjust([0.04, 0.01, 0.03])
    np.testing.assert_allclose(adjusted, [0.06, 0.03, 0.06])


def test_sequence_cycles_require_consecutive_original_visits():
    prepared = {
        "protocol": "ABB",
        "fine_labels": np.array([
            "A", "B1", "B2", "A", "B1", "B2", "A", "B1", "B2"
        ]),
        "source_visit_indices": np.array([0, 1, 2, 3, 4, 6, 7, 8, 9]),
    }
    assert find_sequence_cycles(prepared).tolist() == [[0, 1, 2], [6, 7, 8]]


def test_sliding_trajectory_runs_from_singleton_exit_to_second_exit():
    rng = np.random.default_rng(2)
    dff = rng.normal(size=(2, 80))
    prepared = {
        "protocol": "ABB",
        "selected_dff": dff,
        "cluster_ids": np.array([1, 2]),
        "fine_labels": np.array(["A", "B1", "B2"]),
        "source_visit_indices": np.array([0, 1, 2]),
        "entry_frames": np.array([5, 25, 50]),
        "exit_frames": np.array([10, 35, 65]),
    }
    source = {
        "X": np.array([[-1.0, 0.5], [1.0, -0.5], [-0.8, 0.4], [0.8, -0.4]]),
        "y": np.array([0, 1, 0, 1]),
        "pair_ids": np.array([0, 0, 1, 1]),
        "cluster_ids": np.array([1, 2]),
        "labels": np.array(["B1", "B2", "B1", "B2"]),
        "source_visit_indices": np.arange(4),
    }
    fitted = reciprocal_transfer(
        source, source, minimum_cells=2, n_permutations=0
    )
    trajectories, events = score_cycle_trajectories(
        prepared,
        fitted["model"],
        fitted["shared_cluster_ids"],
        frame_rate=10,
        window_seconds=0.5,
        step_seconds=0.5,
    )
    assert trajectories["time_seconds"].min() == 0
    assert set(trajectories["phase"]) == {
        "corridor to B1", "B1", "corridor to B2", "B2"
    }
    assert events.set_index("event").loc["B2 exit", "time_seconds"] == 5.5
