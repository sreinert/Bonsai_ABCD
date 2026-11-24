import numpy as np
import pandas as pd

from analysis.sequence_compression.sc_neural_data.cross_protocol_decoding import (
    find_abb_omission_triplet_visits,
    label_landmark_visits,
)


def _events(sequence_positions):
    return pd.DataFrame({
        "LM_Count": np.arange(len(sequence_positions)),
        "LM_Position": 10 * np.arange(len(sequence_positions)),
        "Sequence_Position": sequence_positions,
    })


def test_abab_labels_are_unchanged():
    frame = _events([0, -1, 0, -1])
    fine, pooled = label_landmark_visits(frame, [0, 10, 20, 30], "ABAB")
    assert fine.tolist() == ["A", "B", "A", "B"]
    assert pooled.tolist() == ["A", "B", "A", "B"]


def test_abb_b1_and_b2_are_pooled_as_b():
    frame = _events([0, -1, -1, 0, -1, -1])
    fine, pooled = label_landmark_visits(
        frame, [0, 10, 20, 30, 40, 50], "ABB"
    )
    assert fine.tolist() == ["A", "B1", "B2", "A", "B1", "B2"]
    assert pooled.tolist() == ["A", "B", "B", "A", "B", "B"]


def test_abb_phase_allows_an_unrewarded_a_visit():
    frame = _events([0, -1, -1, -1, -1, -1, 0, -1, -1])
    fine, pooled = label_landmark_visits(
        frame, np.arange(9) * 10, "ABB"
    )
    assert fine.tolist() == ["A", "B1", "B2"] * 3
    assert pooled.tolist() == ["A", "B", "B"] * 3


def test_abb_omission_excludes_a_and_its_following_b1_b2():
    frame = _events([0, -1, -1, -1, -1, -1, 0, -1, -1])
    fine, _, metadata = label_landmark_visits(
        frame, np.arange(9) * 10, "ABB", return_event_metadata=True
    )
    excluded, omitted_a = find_abb_omission_triplet_visits(
        fine, metadata["event_visit"], metadata["sequence_position"]
    )
    assert omitted_a.tolist() == [3]
    assert excluded.tolist() == [3, 4, 5]


def test_abb_rewarded_a_triplets_are_not_excluded():
    frame = _events([0, -1, -1, 0, -1, -1])
    fine, _, metadata = label_landmark_visits(
        frame, np.arange(6) * 10, "ABB", return_event_metadata=True
    )
    excluded, omitted_a = find_abb_omission_triplet_visits(
        fine, metadata["event_visit"], metadata["sequence_position"]
    )
    assert omitted_a.size == 0
    assert excluded.size == 0


def test_aab_a1_and_a2_are_pooled_as_a():
    frame = _events([0, 0, -1, 0, 0, -1])
    fine, pooled = label_landmark_visits(
        frame, [0, 10, 20, 30, 40, 50], "AAB"
    )
    assert fine.tolist() == ["A1", "A2", "B", "A1", "A2", "B"]
    assert pooled.tolist() == ["A", "A", "B", "A", "A", "B"]


def test_aab_phase_allows_an_unrewarded_a_visit():
    frame = _events([0, 0, -1, 0, -1, -1, 0, 0, -1])
    fine, pooled = label_landmark_visits(
        frame, np.arange(9) * 10, "AAB"
    )
    assert fine.tolist() == ["A1", "A2", "B"] * 3
    assert pooled.tolist() == ["A", "A", "B"] * 3
