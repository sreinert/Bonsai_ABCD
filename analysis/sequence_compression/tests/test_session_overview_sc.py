from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from analysis.sequence_compression import session_overview_sc as overview


def test_orient_dff_accepts_both_nwb_layouts():
    frames_by_roi = np.arange(15).reshape(5, 3)

    np.testing.assert_array_equal(overview._orient_dff(frames_by_roi, 3), frames_by_roi)
    np.testing.assert_array_equal(
        overview._orient_dff(frames_by_roi.T, 3), frames_by_roi
    )
    with pytest.raises(ValueError, match="Cannot match"):
        overview._orient_dff(np.zeros((4, 5)), 3)


def test_valid_frames_are_checked_before_selection():
    dff = np.arange(15).reshape(5, 3)
    np.testing.assert_array_equal(
        overview._select_valid_frames(dff, [1, 3]), dff[[1, 3]]
    )
    with pytest.raises(IndexError, match="incompatible"):
        overview._select_valid_frames(dff, [0, 5])
    with pytest.raises(ValueError, match="non-empty"):
        overview._select_valid_frames(dff, [])


def test_reference_signals_use_new_api_and_are_saved_aligned(tmp_path, monkeypatch):
    plane = tmp_path / "suite2p" / "plane0"
    plane.mkdir(parents=True)
    F = np.array([[10, 12, 14, 16], [20, 23, 26, 29]], dtype=float)
    Fneu = np.ones_like(F) * 2
    F2 = np.array([[4, 5, 6, 7], [8, 9, 10, 11]], dtype=float)
    Fneu2 = np.ones_like(F2)
    for name, values in (
        ("F.npy", F),
        ("Fneu.npy", Fneu),
        ("F_chan2.npy", F2),
        ("Fneu_chan2.npy", Fneu2),
    ):
        np.save(plane / name, values)
    np.save(
        plane / "ops.npy",
        {
            "fs": 4.0,
            "neucoeff": 0.25,
            "baseline": "constant_prctile",
            "prctile_baseline": 0.0,
        },
    )

    delta_calls = []

    def fake_channel_delta(signal, neuropil, ops):
        delta_calls.append((signal.copy(), neuropil.copy(), ops))
        corrected = signal - ops["neucoeff"] * neuropil
        return corrected - np.min(corrected, axis=1, keepdims=True)

    def fake_reference_correction(green, red, fs):
        corrected = green - 0.5 * (red - np.median(red, axis=1, keepdims=True))
        n_rois = green.shape[0]
        return SimpleNamespace(
            corrected=corrected,
            beta=np.full(n_rois, 0.5),
            raw_standardized_coupling=np.full(n_rois, 0.6),
            applied_standardized_coupling=np.full(n_rois, 0.6),
            green_scale=np.ones(n_rois),
            red_scale=np.ones(n_rois),
            converged=np.ones(n_rois, dtype=bool),
            iterations=np.ones(n_rois, dtype=int),
            status=np.full(n_rois, "applied"),
            smoothing_kernel_frames=1,
            tuning_constant=3.0,
            max_iterations=50,
            tolerance=1e-6,
            scale_epsilon=1e-6,
        )

    monkeypatch.setattr(overview, "compute_channel_delta", fake_channel_delta)
    monkeypatch.setattr(
        overview, "fit_reference_correction", fake_reference_correction
    )
    aligned_funcimg = tmp_path / "aligned" / "funcimg"
    aligned_funcimg.mkdir(parents=True)
    for legacy_name in ("DF_F0.npy", "DF_F02.npy", "DG_R.npy"):
        np.save(aligned_funcimg / legacy_name, np.full((2, 2), 999))
    result = overview._load_or_extract_reference_signals(
        plane, aligned_funcimg, accepted=[1], valid_frames=[1, 3]
    )

    green_full = F - 0.25 * Fneu
    red_full = F2 - 0.25 * Fneu2
    expected_green_full = green_full - np.min(
        green_full, axis=1, keepdims=True
    )
    expected_red_full = red_full - np.min(red_full, axis=1, keepdims=True)
    expected_corrected_full = expected_green_full - 0.5 * (
        expected_red_full - np.median(expected_red_full, axis=1, keepdims=True)
    )
    expected_green = expected_green_full[:, [1, 3]]
    expected_red = expected_red_full[:, [1, 3]]
    expected_corrected = expected_corrected_full[:, [1, 3]]
    np.testing.assert_allclose(result["green_delta"], expected_green[[1]])
    np.testing.assert_allclose(result["red_delta"], expected_red[[1]])
    np.testing.assert_allclose(
        result["corrected_green_delta"], expected_corrected[[1]]
    )
    assert len(delta_calls) == 2
    np.testing.assert_array_equal(delta_calls[0][0], F)
    np.testing.assert_array_equal(delta_calls[1][0], F2)
    assert {path.name for path in aligned_funcimg.glob("*.npy")} == {
        "DF_F0.npy",
        "DF_F02.npy",
        "DG_R.npy",
        "reference_green_delta_v1.npy",
        "reference_red_delta_v1.npy",
        "reference_corrected_green_delta_v1.npy",
    }
    assert (aligned_funcimg / "reference_correction_diagnostics_v1.npz").is_file()


def test_accepted_cells_and_empirical_frame_rate():
    plane_seg = pd.DataFrame({"Accepted": [True, False, True]})
    np.testing.assert_array_equal(
        overview._accepted_roi_indices(plane_seg), [0, 2]
    )
    assert overview._empirical_frame_rate([0.0, 0.02, 0.04, 0.06]) == pytest.approx(50)
    with pytest.raises(ValueError, match="No accepted"):
        overview._accepted_roi_indices(pd.DataFrame({"Accepted": [False, False]}))


def test_load_aligned_behaviour_checks_alignment(tmp_path):
    np.savez(
        tmp_path / "behaviour_data.npz",
        rewards=[0, 1, 0, 2],
        position=[0.0, 1.0, np.nan, 3.0],
    )

    rewards, position = overview._load_aligned_behaviour(tmp_path, 4)
    np.testing.assert_array_equal(rewards, [1, 3])
    np.testing.assert_array_equal(position, [0, 1, 0, 3])
    with pytest.raises(ValueError, match="rewards has 4"):
        overview._load_aligned_behaviour(tmp_path, 3)


def test_reward_tuning_uses_complete_separated_windows_without_mutating_rewards():
    dff = np.arange(80, dtype=float).reshape(2, 40)
    rewards = np.array([1, 10, 11, 25])

    result = overview.reward_tuning_matrix(
        dff, rewards, frame_rate=2, window_size=(-1, 2)
    )

    assert result.shape == (2, 2, 6)
    np.testing.assert_array_equal(result[:, 0], dff[:, 8:14])
    np.testing.assert_array_equal(result[:, 1], dff[:, 23:29])
    np.testing.assert_array_equal(rewards, [1, 10, 11, 25])


def test_reward_psth_uses_largest_common_window_before_another_landmark():
    dff = np.arange(200, dtype=float).reshape(2, 100)

    matrix, usable_rewards, time = overview.reward_psth_matrix(
        dff,
        cell_indices=[1],
        reward_frames=[30, 65],
        landmark_entries=[25, 60, 74],
        landmark_exits=[35, 72, 85],
        frame_rate=10,
        window_size=(-1, 1),
    )

    assert matrix.shape == (1, 2, 19)
    np.testing.assert_array_equal(usable_rewards, [30, 65])
    np.testing.assert_array_equal(matrix[0, 0], dff[1, 20:39])
    np.testing.assert_array_equal(matrix[0, 1], dff[1, 55:74])
    assert time[0] == pytest.approx(-1)
    assert time[-1] == pytest.approx(0.8)


def test_reward_psth_subplots_only_tick_window_limits_and_reward():
    dff = np.arange(200, dtype=float).reshape(2, 100)

    figure = overview.plot_selected_cell_reward_psths(
        dff,
        cell_indices=[0],
        reward_frames=[50],
        landmark_entries=[45],
        landmark_exits=[55],
        frame_rate=10,
        window_size=(-2, 2),
        cell_labels=[7],
    )

    np.testing.assert_allclose(figure.axes[0].get_xticks(), [-2, 0, 1.9])
    assert [tick.get_text() for tick in figure.axes[0].get_xticklabels()] == [
        "-2.00", "0", "1.90"
    ]
    plt.close(figure)


def test_run_overview_adds_four_corrected_plots_and_cleans_legacy(tmp_path, monkeypatch):
    session_path = tmp_path / "aligned" / "sub-07" / "ses-full001_date-test"
    session_path.mkdir(parents=True)
    save_dir = tmp_path / "analysis"
    save_dir.mkdir()
    legacy = save_dir / "7_goal_progress_all.png"
    legacy.touch()
    unrelated = save_dir / "notes.png"
    unrelated.touch()
    calls = []

    monkeypatch.setattr(
        overview,
        "load_session_data",
        lambda *args, **kwargs: {
            "dF": np.arange(120, dtype=float).reshape(3, 40),
            "red_reference_corrected_dF": (
                np.arange(120, dtype=float).reshape(3, 40) / 10
            ),
            "neurons": np.array([2, 5, 9]),
            "mean_img": np.zeros((2, 2)),
            "image_mask": [np.eye(2, dtype=bool)],
            "frame_rate": 2.0,
            "reward_frames": np.array([10, 25]),
            "landmark_entries": np.array([8, 23]),
            "landmark_exits": np.array([14, 29]),
            "session_path": session_path,
            "subject": "sub-07",
        },
    )

    def fake_fov(*args, **kwargs):
        calls.append(("fov", kwargs["save_path"]))
        return plt.figure()

    def fake_traces(*args, **kwargs):
        name = (
            "corrected_traces"
            if kwargs.get("signal_label", "").startswith("Red-reference")
            else "traces"
        )
        calls.append((name, kwargs["save_path"]))
        selected = overview.select_trace_cells(3, 20)
        np.testing.assert_array_equal(kwargs["cell_indices"], selected)
        np.testing.assert_array_equal(kwargs["cell_labels"], np.array([2, 5, 9])[selected])
        return plt.figure()

    def fake_reward(*args, **kwargs):
        prefix = (
            "corrected_"
            if kwargs.get("signal_label", "").startswith("Red-reference")
            else ""
        )
        calls.append(
            (
                prefix + ("reward_cv" if kwargs["cross_validated"] else "reward"),
                kwargs["save_path"],
            )
        )
        return plt.figure()

    def fake_reward_psth(*args, **kwargs):
        prefix = (
            "corrected_"
            if kwargs.get("signal_label", "").startswith("Red-reference")
            else ""
        )
        calls.append((prefix + "reward_psth", kwargs["save_path"]))
        selected = overview.select_trace_cells(3, 20)
        np.testing.assert_array_equal(args[1], selected)
        np.testing.assert_array_equal(kwargs["cell_labels"], np.array([2, 5, 9])[selected])
        return plt.figure()

    monkeypatch.setattr(overview, "plot_fov_masks", fake_fov)
    monkeypatch.setattr(overview, "plot_raw_traces", fake_traces)
    monkeypatch.setattr(overview, "plot_reward_heatmap", fake_reward)
    monkeypatch.setattr(overview, "plot_selected_cell_reward_psths", fake_reward_psth)

    _, figures = overview.run_overview(
        "sub-07", "full001", tmp_path, save_dir=save_dir, show=False
    )

    assert [name for name, _ in calls] == [
        "fov",
        "traces",
        "reward",
        "reward_cv",
        "reward_psth",
        "corrected_traces",
        "corrected_reward",
        "corrected_reward_cv",
        "corrected_reward_psth",
    ]
    assert set(figures) == {
        "fov",
        "raw_traces",
        "reward",
        "reward_cv",
        "reward_psth_selected_cells",
        "reference_corrected_raw_traces",
        "reference_corrected_reward",
        "reference_corrected_reward_cv",
        "reference_corrected_reward_psth_selected_cells",
    }
    assert not legacy.exists()
    assert unrelated.exists()
    for figure in figures.values():
        plt.close(figure)


def test_sparse_reward_plot_is_skipped_without_failing(tmp_path, monkeypatch):
    session_path = tmp_path / "ses-full001_date-test"
    session_path.mkdir()
    monkeypatch.setattr(
        overview,
        "load_session_data",
        lambda *args, **kwargs: {
            "dF": np.ones((2, 20)),
            "neurons": np.array([1, 4]),
            "mean_img": np.zeros((2, 2)),
            "image_mask": [np.eye(2, dtype=bool)],
            "frame_rate": 2.0,
            "reward_frames": np.array([], dtype=int),
            "landmark_entries": np.array([], dtype=int),
            "landmark_exits": np.array([], dtype=int),
            "session_path": session_path,
            "subject": "sub-07",
        },
    )
    monkeypatch.setattr(overview, "plot_fov_masks", lambda *a, **k: plt.figure())
    monkeypatch.setattr(overview, "plot_raw_traces", lambda *a, **k: plt.figure())
    monkeypatch.setattr(
        overview,
        "plot_reward_heatmap",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("not enough rewards")),
    )
    monkeypatch.setattr(
        overview,
        "plot_selected_cell_reward_psths",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("not enough clean rewards")),
    )

    save_dir = tmp_path / "plots"
    save_dir.mkdir()
    stale_reward_plots = [
        save_dir / "3_reward_heatmap.png",
        save_dir / "4_reward_heatmap_cv.png",
        save_dir / "5_reward_psth_selected_cells.png",
        save_dir / "6_reference_corrected_raw_traces.png",
        save_dir / "7_reference_corrected_reward_heatmap.png",
        save_dir / "8_reference_corrected_reward_heatmap_cv.png",
        save_dir / "9_reference_corrected_reward_psth_selected_cells.png",
    ]
    for path in stale_reward_plots:
        path.touch()

    _, figures = overview.run_overview(
        "sub-07", "full001", tmp_path, save_dir=save_dir, show=False
    )

    assert figures["reward"] is None
    assert figures["reward_cv"] is None
    assert figures["reward_psth_selected_cells"] is None
    assert all(not path.exists() for path in stale_reward_plots)
    plt.close(figures["fov"])
    plt.close(figures["raw_traces"])


def test_legacy_cleanup_is_limited_to_known_overview_files(tmp_path):
    legacy_names = [
        "5_position_heatmap.png",
        "5b_position_heatmap_cv.png",
        "7_goal_progress_all.png",
        "13_goal_progress_polar_high_tuning_correct.png",
    ]
    for name in legacy_names:
        (tmp_path / name).touch()
    preserved = ["4_reward_heatmap_cv.png", "14_custom.png", "notes.txt"]
    for name in preserved:
        (tmp_path / name).touch()

    removed = overview.remove_legacy_overview_plots(tmp_path)

    assert {path.name for path in removed} == set(legacy_names)
    assert all(not (tmp_path / name).exists() for name in legacy_names)
    assert all((tmp_path / name).exists() for name in preserved)
