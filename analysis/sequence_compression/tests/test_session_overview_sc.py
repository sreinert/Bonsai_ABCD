from pathlib import Path

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


def test_celltv_signals_use_ops_coefficient_and_are_saved_aligned(tmp_path, monkeypatch):
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

    def constant_percentile(signal, ops):
        percentile = ops["prctile_baseline"]
        return signal - np.percentile(signal, percentile, axis=1, keepdims=True)

    monkeypatch.setattr(overview, "_suite2p_preprocess", constant_percentile)
    aligned_funcimg = tmp_path / "aligned" / "funcimg"
    result = overview._load_or_extract_celltv_signals(
        plane, aligned_funcimg, accepted=[1], valid_frames=[1, 3]
    )

    green = (F - 0.25 * Fneu)[:, [1, 3]]
    red = (F2 - 0.25 * Fneu2)[:, [1, 3]]
    red_floor = np.maximum(np.percentile(red, 5, axis=1, keepdims=True), 1e-3)
    ratio = green / np.maximum(red, red_floor)
    celltv_gr = ratio - np.percentile(ratio, 0, axis=1, keepdims=True)
    expected_green = green - np.percentile(green, 0, axis=1, keepdims=True)
    expected_red = red - np.percentile(red, 0, axis=1, keepdims=True)
    np.testing.assert_allclose(result["dff"], expected_green[[1]], rtol=1e-6)
    np.testing.assert_allclose(result["dff2"], expected_red[[1]], rtol=1e-6)
    np.testing.assert_allclose(result["dg_r"], celltv_gr[[1]], rtol=3e-6)
    assert {path.name for path in aligned_funcimg.glob("*.npy")} == {
        "DF_F0.npy",
        "DF_F02.npy",
        "DG_R.npy",
    }


def test_overview_reads_neuropil_coefficient_without_qc_helper():
    assert overview._neuropil_coefficient({"neucoeff": 0.35}) == 0.35
    assert overview._neuropil_coefficient(
        {"extraction": {"neuropil_coefficient": 0.55}}
    ) == 0.55
    with pytest.raises(KeyError, match="cannot reproduce CellTV"):
        overview._neuropil_coefficient({})


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


def test_run_overview_adds_four_gr_plots_and_cleans_legacy(tmp_path, monkeypatch):
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
            "gr": np.arange(120, dtype=float).reshape(3, 40) / 10,
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
            "gr_traces"
            if kwargs.get("signal_label", "").startswith("CellTV")
            else "traces"
        )
        calls.append((name, kwargs["save_path"]))
        selected = overview.select_trace_cells(3, 20)
        np.testing.assert_array_equal(kwargs["cell_indices"], selected)
        np.testing.assert_array_equal(kwargs["cell_labels"], np.array([2, 5, 9])[selected])
        return plt.figure()

    def fake_reward(*args, **kwargs):
        prefix = (
            "gr_" if kwargs.get("signal_label", "").startswith("CellTV") else ""
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
            "gr_" if kwargs.get("signal_label", "").startswith("CellTV") else ""
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
        "gr_traces",
        "gr_reward",
        "gr_reward_cv",
        "gr_reward_psth",
    ]
    assert set(figures) == {
        "fov",
        "raw_traces",
        "reward",
        "reward_cv",
        "reward_psth_selected_cells",
        "gr_raw_traces",
        "gr_reward",
        "gr_reward_cv",
        "gr_reward_psth_selected_cells",
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
        save_dir / "6_gr_raw_traces.png",
        save_dir / "7_gr_reward_heatmap.png",
        save_dir / "8_gr_reward_heatmap_cv.png",
        save_dir / "9_gr_reward_psth_selected_cells.png",
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
