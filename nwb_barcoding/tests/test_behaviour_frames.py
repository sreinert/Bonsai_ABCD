import numpy as np
import pytest

from nwb_barcoding.build_behaviour_data import (
    _atomic_savez,
    frame_bins_for_behaviour,
)


def test_frame_bins_when_behaviour_ends_first():
    frames, edges = frame_bins_for_behaviour(np.arange(0.0, 11.0), 2.2, 6.4)
    assert frames.tolist() == [2, 3, 4, 5, 6]
    assert edges.tolist() == [2, 3, 4, 5, 6, 7]
    assert len(edges) == len(frames) + 1


def test_frame_bins_when_imaging_ends_first():
    frames, edges = frame_bins_for_behaviour(np.arange(5.0), -1.0, 10.0)
    assert frames.tolist() == [0, 1, 2, 3]
    assert edges.tolist() == [0, 1, 2, 3, 4]


def test_frame_bins_reject_no_overlap_and_bad_timestamps():
    with pytest.raises(ValueError, match="does not overlap"):
        frame_bins_for_behaviour(np.arange(5.0), 6.0, 8.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        frame_bins_for_behaviour([0.0, 1.0, 1.0], 0.0, 1.0)


def test_atomic_npz_uses_correct_pd2_key(tmp_path):
    output = tmp_path / "behaviour_data.npz"
    _atomic_savez(output, pd1=np.ones(3), pd2=np.zeros(3))
    with np.load(output) as saved:
        assert set(saved.files) == {"pd1", "pd2"}
    assert not (tmp_path / ".behaviour_data.npz.partial").exists()

