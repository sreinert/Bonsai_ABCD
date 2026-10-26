import numpy as np
import pandas as pd
import pytest

from analysis.sequence_compression.paired_omission_power import (
    analyse_by_mouse,
    make_paired_table,
    paired_effect_size,
    paired_t_power,
    required_pairs,
    sensitivity_table,
)


def test_paired_effect_size_uses_sd_of_pairwise_differences():
    a = np.array([3.0, 5.0, 4.0, 8.0])
    b = np.array([2.0, 2.0, 3.0, 4.0])
    differences = a - b

    expected = differences.mean() / differences.std(ddof=1)

    assert paired_effect_size(a, b) == pytest.approx(expected)


def test_required_pairs_reaches_target_and_is_minimal():
    n_pairs = required_pairs(0.5, target_power=0.8)

    assert n_pairs == 34
    assert paired_t_power(0.5, n_pairs) >= 0.8
    assert paired_t_power(0.5, n_pairs - 1) < 0.8


def test_make_paired_table_and_analyse_each_mouse():
    data = pd.DataFrame(
        {
            "mouse": ["01"] * 6 + ["02"] * 6,
            "session": [1] * 12,
            "pair_id": [0, 0, 1, 1, 2, 2] * 2,
            "condition": ["omitted_A", "B"] * 6,
            "response": [4, 2, 5, 2, 5, 4, 3, 2, 4, 2, 6, 3],
        }
    )

    paired = make_paired_table(data)
    result = analyse_by_mouse(paired, exclusion_rate=0.2)

    assert len(paired) == 6
    assert result["mouse"].tolist() == ["01", "02"]
    assert result["pilot_pairs"].tolist() == [3, 3]
    assert result.loc[0, "mean_difference_A_minus_B"] == pytest.approx(2.0)
    assert np.all(
        result["planned_probe_trials"] >= result["required_valid_pairs"]
    )


def test_duplicate_condition_within_pair_is_rejected():
    data = pd.DataFrame(
        {
            "mouse": ["01", "01", "01"],
            "session": [1, 1, 1],
            "pair_id": [0, 0, 0],
            "condition": ["omitted_A", "omitted_A", "B"],
            "response": [4, 5, 2],
        }
    )

    with pytest.raises(ValueError, match="at most one observation"):
        make_paired_table(data)


def test_sensitivity_table_matches_standard_reference_points():
    result = sensitivity_table(effect_sizes=[0.5, 0.8], exclusion_rate=0.0)

    assert result["required_valid_pairs"].tolist() == [34, 15]
    assert result["planned_probe_trials"].tolist() == [34, 15]
