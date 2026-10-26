import numpy as np
import pandas as pd
import pytest

from analysis.sequence_compression.session_functions.behaviour import (
    Behaviour,
    summarize_aab_omission_lick_counts,
)
from analysis.sequence_compression.session_functions.session import Session


class FakeAABSession:
    sequence = "AAB"
    lm_size = 5.0
    animal = "01"
    session_id = "aab-omissions-001"

    def __init__(self, reward_positions, lick_positions):
        self.reward_positions = np.asarray(reward_positions, dtype=float)
        self.lick_positions = np.asarray(lick_positions, dtype=float)
        self.threshold_seen = None
        self.release_positions = np.arange(0.0, 110.0, 10.0)

    def get_event_parsed(self, threshold="below"):
        self.threshold_seen = threshold
        return (
            self.lick_positions,
            np.arange(len(self.lick_positions)),
            np.arange(len(self.reward_positions)),
            self.reward_positions,
            None,
        )

    def find_targets_distractors(self):
        target_positions = self.release_positions[[0, 2, 3, 5, 6, 8, 9]]
        distractor_positions = self.release_positions[[1, 4, 7, 10]]
        return None, None, target_positions, distractor_positions, None, None

    def get_A_B_splits(self):
        # Cohort-1 AAB parsing begins A2, B, A1, A2, B, ...
        return (
            np.array([2, 5, 8]),
            np.array([0, 3, 6, 9]),
            np.array([1, 4, 7, 10]),
            np.array([]),
            np.array([]),
        )


class FakeBehaviour(Behaviour):
    def __init__(self, session, omitted_landmarks):
        super().__init__(session, performance=None)
        self.omitted_landmarks = np.asarray(omitted_landmarks, dtype=int)

    def get_omissions(self):
        return self.omitted_landmarks, np.array([])


class FakeParsingSession(Session):
    @property
    def sequence(self):
        return "AAB"

    @property
    def reward_seq(self):
        return np.array([0, 0, -1])

    @property
    def lm_size(self):
        return 3.0

    @property
    def sess(self):
        return {
            "position": self.dataframe["Position"].to_numpy(),
            "speed": np.zeros(len(self.dataframe)),
            "licks": self.dataframe["Licks"].to_numpy(),
            "rewards": np.array([]),
            "lick_threshold": 1.0,
        }

    def estimate_lm_events(self):
        return pd.DataFrame(
            {
                "Position": [0.5, 10.0, 20.0, 30.0],
                "Index": [0, 1, 2, 3],
                "Odour": [0, 1, 2, 1],
            }
        )


def test_extracts_b_counts_and_respects_aab_offset_and_boundaries():
    session = FakeAABSession(
        reward_positions=[31.0, 51.0],
        lick_positions=[40.0, 41.0, 44.0, 45.0, 72.0],
    )
    behaviour = FakeBehaviour(session, omitted_landmarks=[2, 6, 8, 9])

    result = behaviour.get_aab_omission_b_lick_counts()

    assert session.threshold_seen == "below"
    assert [(r["omitted_a"], r["b_lick_count"]) for r in result["records"]] == [
        ("A1", 3),
        ("A2", 1),
    ]
    assert result["records"][0]["b_landmark_index"] == 4
    assert result["qc"] == {
        "total_b": 4,
        "included": 2,
        "incomplete_triplet": 1,
        "no_omission": 0,
        "invalid_omission_pattern": 1,
        "other_a_unrewarded": 0,
    }


def test_aab_parser_discards_faulty_initial_cohort1_landmark():
    dataframe = pd.DataFrame(
        {
            "Position": [0.5, 10.0, 20.0, 30.0],
            "Licks": [False, False, False, False],
            "Rewards": [np.nan, np.nan, np.nan, np.nan],
            "LM_Count": [0, 1, 2, 3],
        }
    )
    session = FakeParsingSession({}, dataframe, None, "01", 1, "aab-test")

    *_, release_df = session.get_event_parsed(threshold="all")

    assert release_df["Position"].tolist() == [10.0, 20.0, 30.0]


def test_cohort2_aab_offset_starts_with_a1():
    dataframe = pd.DataFrame(
        {
            "Position": [5.0, 10.0, 20.0, 30.0],
            "Licks": [False, False, False, False],
            "Rewards": [np.nan, np.nan, np.nan, np.nan],
            "LM_Count": [0, 1, 2, 3],
        }
    )
    session = FakeParsingSession(
        {"initialCorridorOffset": 5}, dataframe, None, "01", 2, "aab-test"
    )

    A_landmarks, B_landmarks, _, _ = session.get_A_B_landmarks()
    A1, A2, B1, _, _ = session.get_A_B_splits()
    target_id, distractor_id, _, _, _, lm_id_sequence = (
        session.find_targets_distractors()
    )

    assert A_landmarks.tolist() == [0, 1, 3]
    assert B_landmarks.tolist() == [2]
    assert A1.tolist() == [0, 3]
    assert A2.tolist() == [1]
    assert B1.tolist() == [2]
    assert target_id.tolist() == [0, 1]
    assert distractor_id.tolist() == [2]
    assert lm_id_sequence.tolist() == [0, 1, 2, 0]


def test_excludes_triplet_when_non_omitted_a_did_not_deliver_reward():
    session = FakeAABSession(reward_positions=[], lick_positions=[41.0])
    behaviour = FakeBehaviour(session, omitted_landmarks=[2])

    result = behaviour.get_aab_omission_b_lick_counts()

    assert result["records"] == []
    assert result["qc"]["other_a_unrewarded"] == 1


def test_explicit_omission_indices_override_the_session_detector():
    session = FakeAABSession(reward_positions=[51.0], lick_positions=[72.0])
    behaviour = FakeBehaviour(session, omitted_landmarks=[2])

    result = behaviour.get_aab_omission_b_lick_counts(
        omitted_landmarks=np.array([6])
    )

    assert [(r["omitted_a"], r["b_lick_count"]) for r in result["records"]] == [
        ("A2", 1)
    ]


def test_session_without_omissions_returns_no_records():
    session = FakeAABSession(
        reward_positions=[21.0, 31.0, 51.0, 61.0, 81.0, 91.0],
        lick_positions=[41.0, 71.0],
    )
    behaviour = FakeBehaviour(session, omitted_landmarks=[])

    result = behaviour.get_aab_omission_b_lick_counts()

    assert result["records"] == []
    assert result["qc"]["no_omission"] == 3
    assert result["qc"]["incomplete_triplet"] == 1


def test_rejects_non_aab_sessions():
    session = FakeAABSession(reward_positions=[], lick_positions=[])
    session.sequence = "ABAB"
    behaviour = FakeBehaviour(session, omitted_landmarks=[])

    with pytest.raises(ValueError, match="requires an AAB session"):
        behaviour.get_aab_omission_b_lick_counts()


def test_summary_retains_empty_and_single_group_mice():
    records = [
        {"mouse": "01", "omitted_a": "A1", "b_lick_count": 3},
        {"mouse": "01", "omitted_a": "A2", "b_lick_count": 1},
        {"mouse": "02", "omitted_a": "A1", "b_lick_count": 2},
    ]

    summary = summarize_aab_omission_lick_counts(records, ["01", "02", "03"])

    assert [row["mouse"] for row in summary] == ["01", "02", "03"]
    assert summary[0]["status"] == "ok"
    assert np.isfinite(summary[0]["p_holm"])
    assert summary[1]["status"] == "insufficient data"
    assert summary[2]["status"] == "insufficient data"
    assert np.isnan(summary[1]["p_raw"])
    assert np.isnan(summary[2]["mean_difference_A1_minus_A2"])
