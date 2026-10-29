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


def test_trace_selection_ranks_all_finite_snr_and_respects_eligibility():
    rng = np.random.default_rng(4)
    traces = rng.normal(scale=1.0, size=(3, 1000))
    traces[1, 400:430] += 20.0
    traces[2] = 0.0

    selected = overview.select_trace_cells(3, 2, traces=traces)
    np.testing.assert_array_equal(selected, [1, 0])

    selected = overview.select_trace_cells(
        3,
        1,
        traces=traces,
        eligible=[True, False, False],
    )
    np.testing.assert_array_equal(selected, [0])

    random_a = overview.select_random_trace_cells(
        3, 2, seed=12, eligible=[True, False, True]
    )
    random_b = overview.select_random_trace_cells(
        3, 2, seed=12, eligible=[True, False, True]
    )
    np.testing.assert_array_equal(random_a, [0, 2])
    np.testing.assert_array_equal(random_a, random_b)


def test_safe_ratio_signals_use_new_api_and_are_saved_aligned(tmp_path, monkeypatch):
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
    np.save(plane / "iscell.npy", np.array([[0, 0.1], [1, 0.9]]))
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

    def fake_safe_ratio(green, green_neuropil, red, red_neuropil, ops):
        del green_neuropil, red_neuropil, ops
        ratio = green / red
        n_rois = green.shape[0]
        return SimpleNamespace(
            delta_ratio=ratio,
            denominator_floor=np.ones(n_rois),
            floor_fraction=np.zeros(n_rois),
            typical_red=np.ones(n_rois),
            status=np.full(n_rois, "ok"),
            smoothing_kernel_frames=1,
            floor_percentile=5.0,
            absolute_floor=1e-3,
        )

    monkeypatch.setattr(overview, "compute_channel_delta", fake_channel_delta)
    monkeypatch.setattr(overview, "compute_celltv_ratio_delta", fake_safe_ratio)
    responsive_calls = []

    def fake_responsive(dff, imaging_path):
        responsive_calls.append((np.asarray(dff).copy(), Path(imaging_path)))
        return np.array([1])

    monkeypatch.setattr(overview, "_celltv_responsive_neurons", fake_responsive)
    aligned_funcimg = tmp_path / "aligned" / "funcimg"
    aligned_funcimg.mkdir(parents=True)
    for legacy_name in ("DF_F0.npy", "DF_F02.npy", "DG_R.npy"):
        np.save(aligned_funcimg / legacy_name, np.full((2, 2), 999))
    result = overview._load_or_extract_ratio_signals(
        plane, aligned_funcimg, accepted=[1], valid_frames=[1, 3]
    )

    green_full = F - 0.25 * Fneu
    red_full = F2 - 0.25 * Fneu2
    expected_green_full = green_full - np.min(
        green_full, axis=1, keepdims=True
    )
    expected_red_full = red_full - np.min(red_full, axis=1, keepdims=True)
    expected_green = expected_green_full[:, [1, 3]]
    expected_red = expected_red_full[:, [1, 3]]
    expected_ratio = (F / F2)[:, [1, 3]]
    np.testing.assert_allclose(result["green_delta"], expected_green[[1]])
    np.testing.assert_allclose(result["red_delta"], expected_red[[1]])
    np.testing.assert_allclose(
        result["ratio_delta"], expected_ratio[[1]]
    )
    np.testing.assert_array_equal(result["valid_neurons"], [1])
    np.testing.assert_allclose(responsive_calls[0][0], expected_green_full[:, [1, 3]])
    assert responsive_calls[0][1] == plane
    assert len(delta_calls) == 2
    np.testing.assert_array_equal(delta_calls[0][0], F)
    np.testing.assert_array_equal(delta_calls[1][0], F2)
    assert {path.name for path in aligned_funcimg.glob("*.npy")} == {
        "DF_F0.npy",
        "DF_F02.npy",
        "DG_R.npy",
        "celltv_green_delta_v4.npy",
        "celltv_red_delta_v4.npy",
        "celltv_green_red_ratio_delta_v4.npy",
        "valid_neurons.npy",
    }
    np.testing.assert_array_equal(np.load(aligned_funcimg / "valid_neurons.npy"), [1])
    assert (aligned_funcimg / "celltv_green_red_ratio_diagnostics_v4.npz").is_file()


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
        speed=[1.0, np.nan, 3.0, 4.0],
        licks=[0, 0, 2, 0],
    )

    aligned = overview._load_aligned_behaviour(tmp_path, 4)
    assert set(aligned) == {"position", "speed", "licks", "rewards"}
    np.testing.assert_array_equal(aligned["position"], [0, 1, 0, 3])
    np.testing.assert_array_equal(aligned["speed"], [1, 0, 3, 4])
    np.testing.assert_array_equal(aligned["licks"], [0, 0, 2, 0])
    np.testing.assert_array_equal(aligned["rewards"], [0, 1, 0, 2])
    with pytest.raises(ValueError, match="has 4"):
        overview._load_aligned_behaviour(tmp_path, 3)


def test_aligned_frame_midpoints_and_raw_clock_mapping():
    midpoints = overview._aligned_frame_midpoints(
        [10.0, 10.2, 10.4, 10.6, 10.8], [1, 2, 3]
    )
    np.testing.assert_allclose(midpoints, [10.3, 10.5, 10.7])

    analog_index = pd.to_datetime([100.0, 101.0, 102.0], unit="s")
    raw_index = pd.to_datetime([100.5, 101.5], unit="s")
    mapped = overview._map_raw_times_to_imaging_clock(
        raw_index, analog_index, [5.0, 7.0, 9.0]
    )
    np.testing.assert_allclose(mapped, [6.0, 8.0])


def test_aligned_task_dataframe_maps_landmarks_to_distinct_frames():
    aligned = {
        "position": np.array([0.0, 10.0, 20.0, 30.0]),
        "speed": np.array([1.0, 2.0, 3.0, 4.0]),
        "licks": np.array([0.0, 1.0, 0.0, 0.0]),
        "rewards": np.array([0.0, 0.0, 1.0, 0.0]),
    }
    raw = pd.DataFrame(
        {
            "LM_Count": [np.nan, 0, 1],
            "LM_Position": [np.nan, 10.0, 30.0],
            "LM_Odour": [np.nan, "odour1", "odour2"],
        },
        index=pd.to_datetime([0.0, 1.0, 3.0], unit="s"),
    )

    dataframe = overview._build_aligned_task_dataframe(
        aligned, [0.0, 1.0, 2.0, 3.0], raw, [0.0, 1.0, 3.0]
    )

    np.testing.assert_array_equal(
        np.flatnonzero(dataframe["LM_Count"].notna()), [1, 3]
    )
    assert dataframe["Licks"].iloc[1] == 1
    assert dataframe["Rewards"].notna().to_numpy().tolist() == [False, False, True, False]

    with pytest.raises(ValueError, match="distinct, strictly ordered"):
        overview._nearest_frame_indices([0.9, 1.1], [0.0, 1.0, 2.0])


def test_paired_behaviour_figures_use_raw_and_aligned_calculations(
    tmp_path, monkeypatch
):
    from analysis.sequence_compression.session_functions import plotting

    calls = []

    def fake_psth(session, behaviour, bins=None, axes=None, **kwargs):
        calls.append(("psth", session, behaviour, bins, len(axes)))
        axes[0].set_title("Speed")
        axes[1].set_title("Lick rate")
        return axes[0].figure

    def fake_transition(session, performance, axes=None):
        calls.append(("transition", session, performance, len(axes)))
        for axis, title in zip(axes, ("Stimulus", "Lick", "Ideal")):
            axis.set_title(title)
        return axes[0].figure

    class FakePerformance:
        def __init__(self, name):
            self.name = name

        def calc_sw_hit_fa(self, **kwargs):
            calls.append(("sliding", self.name, kwargs["window"], kwargs["split_lms"]))
            kwargs["ax"].plot([0, 1], [0.2, 0.8])
            return {}, {}, kwargs["ax"].figure

    monkeypatch.setattr(plotting, "plot_speed_lick_rate_psth", fake_psth)
    monkeypatch.setattr(plotting, "plot_transition_matrix", fake_transition)
    context = {
        "raw_session": "raw-session",
        "aligned_session": "aligned-session",
        "raw_behaviour": "raw-behaviour",
        "aligned_behaviour": "aligned-behaviour",
        "raw_performance": FakePerformance("raw"),
        "aligned_performance": FakePerformance("aligned"),
    }

    psth = overview.plot_behaviour_psth_comparison(
        context, tmp_path / "psth.png"
    )
    transition = overview.plot_behaviour_transition_comparison(
        context, tmp_path / "transition.png"
    )
    sliding = overview.plot_behaviour_sliding_performance(
        context, tmp_path / "sliding.png"
    )

    assert [call[0] for call in calls] == [
        "psth",
        "psth",
        "transition",
        "transition",
        "sliding",
        "sliding",
    ]
    assert calls[0][1:4] == ("raw-session", "raw-behaviour", None)
    assert calls[1][1:4] == ("aligned-session", "aligned-behaviour", None)
    assert calls[-2][2:] == (24, True)
    assert calls[-1][2:] == (24, True)
    assert (tmp_path / "psth.png").is_file()
    assert (tmp_path / "transition.png").is_file()
    assert (tmp_path / "sliding.png").is_file()
    plt.close(psth)
    plt.close(transition)
    plt.close(sliding)


def test_behaviour_alignment_plot_contains_four_comparison_panels(tmp_path):
    raw = pd.DataFrame(
        {
            "Position": [0.0, 5.0, 10.0],
            "Treadmill": [1.0, 2.0, 3.0],
            "Licks": [False, True, False],
            "Rewards": [np.nan, np.nan, "Reward"],
        }
    )
    data = {
        "frame_times": np.array([10.0, 11.0, 12.0]),
        "aligned_behaviour": {
            "position": np.array([0.0, 5.0, 10.0]),
            "speed": np.array([1.0, 2.0, 3.0]),
            "licks": np.array([0.0, 1.0, 0.0]),
            "rewards": np.array([0.0, 0.0, 1.0]),
        },
        "behaviour_analysis": {
            "raw_dataframe": raw,
            "raw_times": np.array([10.0, 11.0, 12.0]),
        },
    }
    output = tmp_path / "behaviour_raw_vs_aligned.png"

    figure = overview.plot_behaviour_alignment(data, output)

    assert len(figure.axes) == 4
    assert [axis.get_ylabel() for axis in figure.axes] == [
        "Position",
        "Speed",
        "Licks",
        "Rewards",
    ]
    assert output.is_file()
    plt.close(figure)


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


def test_run_overview_adds_four_safe_ratio_plots_and_cleans_legacy(tmp_path, monkeypatch):
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
            "green_red_ratio_dF": (
                np.arange(120, dtype=float).reshape(3, 40) / 10
            ),
            "valid_cell_mask": np.array([True, False, True]),
            "neurons": np.array([2, 5, 9]),
            "mean_img": np.zeros((2, 2)),
            "image_mask": [np.eye(2, dtype=bool)],
            "frame_rate": 2.0,
            "reward_frames": np.array([10, 25]),
            "landmark_entries": np.array([8, 23]),
            "landmark_exits": np.array([14, 29]),
            "session_path": session_path,
            "subject": "sub-07",
            "behaviour_analysis": {},
        },
    )

    def fake_fov(*args, **kwargs):
        calls.append(("fov", kwargs["save_path"]))
        return plt.figure()

    def fake_traces(*args, **kwargs):
        name = (
            "ratio_traces"
            if kwargs.get("signal_label", "").startswith("Safe")
            else "traces"
        )
        calls.append((name, kwargs["save_path"]))
        selected = np.array([0, 2])
        np.testing.assert_array_equal(kwargs["cell_indices"], selected)
        np.testing.assert_array_equal(kwargs["cell_labels"], np.array([2, 5, 9])[selected])
        return plt.figure()

    def fake_reward(*args, **kwargs):
        prefix = (
            "ratio_"
            if kwargs.get("signal_label", "").startswith("Safe")
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
            "ratio_"
            if kwargs.get("signal_label", "").startswith("Safe")
            else ""
        )
        calls.append((prefix + "reward_psth", kwargs["save_path"]))
        selected = np.array([0, 2])
        np.testing.assert_array_equal(args[1], selected)
        np.testing.assert_array_equal(kwargs["cell_labels"], np.array([2, 5, 9])[selected])
        return plt.figure()

    monkeypatch.setattr(overview, "plot_fov_masks", fake_fov)
    monkeypatch.setattr(overview, "plot_raw_traces", fake_traces)
    monkeypatch.setattr(overview, "plot_reward_heatmap", fake_reward)
    monkeypatch.setattr(overview, "plot_selected_cell_reward_psths", fake_reward_psth)

    def fake_behaviour(name):
        def plot(*args, **kwargs):
            calls.append((name, kwargs["save_path"]))
            return plt.figure()

        return plot

    monkeypatch.setattr(
        overview, "plot_behaviour_alignment", fake_behaviour("behaviour_alignment")
    )
    monkeypatch.setattr(
        overview,
        "plot_behaviour_psth_comparison",
        fake_behaviour("behaviour_speed_lick_psth"),
    )
    monkeypatch.setattr(
        overview,
        "plot_behaviour_transition_comparison",
        fake_behaviour("behaviour_transition_matrix"),
    )
    monkeypatch.setattr(
        overview,
        "plot_behaviour_sliding_performance",
        fake_behaviour("behaviour_sliding_performance"),
    )

    _, figures = overview.run_overview(
        "sub-07", "full001", tmp_path, save_dir=save_dir, show=False
    )

    assert [name for name, _ in calls] == [
        "fov",
        "traces",
        "traces",
        "reward",
        "reward_cv",
        "reward_psth",
        "reward_psth",
        "ratio_traces",
        "ratio_traces",
        "ratio_reward",
        "ratio_reward_cv",
        "ratio_reward_psth",
        "ratio_reward_psth",
        "behaviour_alignment",
        "behaviour_speed_lick_psth",
        "behaviour_transition_matrix",
        "behaviour_sliding_performance",
    ]
    assert set(figures) == {
        "fov",
        "raw_traces_high_snr",
        "raw_traces_random",
        "reward",
        "reward_cv",
        "reward_psth_high_snr",
        "reward_psth_random",
        "safe_gr_raw_traces_high_snr",
        "safe_gr_raw_traces_random",
        "safe_gr_reward",
        "safe_gr_reward_cv",
        "safe_gr_reward_psth_high_snr",
        "safe_gr_reward_psth_random",
        "behaviour_alignment",
        "behaviour_speed_lick_psth",
        "behaviour_transition_matrix",
        "behaviour_sliding_performance",
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
            "valid_cell_mask": np.array([True, True]),
            "neurons": np.array([1, 4]),
            "mean_img": np.zeros((2, 2)),
            "image_mask": [np.eye(2, dtype=bool)],
            "frame_rate": 2.0,
            "reward_frames": np.array([], dtype=int),
            "landmark_entries": np.array([], dtype=int),
            "landmark_exits": np.array([], dtype=int),
            "session_path": session_path,
            "subject": "sub-07",
            "behaviour_analysis": {},
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
    monkeypatch.setattr(
        overview, "plot_behaviour_alignment", lambda *a, **k: plt.figure()
    )
    monkeypatch.setattr(
        overview, "plot_behaviour_psth_comparison", lambda *a, **k: plt.figure()
    )
    monkeypatch.setattr(
        overview, "plot_behaviour_transition_comparison", lambda *a, **k: plt.figure()
    )
    monkeypatch.setattr(
        overview, "plot_behaviour_sliding_performance", lambda *a, **k: plt.figure()
    )

    save_dir = tmp_path / "plots"
    save_dir.mkdir()
    stale_reward_plots = [
        save_dir / "3_reward_heatmap.png",
        save_dir / "4_reward_heatmap_cv.png",
        save_dir / "5h_high_snr_reward_psth_selected_cells.png",
        save_dir / "5r_random_reward_psth_selected_cells.png",
        save_dir / "6h_high_snr_safe_gr_raw_traces.png",
        save_dir / "6r_random_safe_gr_raw_traces.png",
        save_dir / "7_safe_gr_reward_heatmap.png",
        save_dir / "8_safe_gr_reward_heatmap_cv.png",
        save_dir / "9h_high_snr_safe_gr_reward_psth.png",
        save_dir / "9r_random_safe_gr_reward_psth.png",
    ]
    for path in stale_reward_plots:
        path.touch()

    _, figures = overview.run_overview(
        "sub-07", "full001", tmp_path, save_dir=save_dir, show=False
    )

    assert figures["reward"] is None
    assert figures["reward_cv"] is None
    assert figures["reward_psth_high_snr"] is None
    assert figures["reward_psth_random"] is None
    assert all(not path.exists() for path in stale_reward_plots)
    plt.close(figures["fov"])
    plt.close(figures["raw_traces_high_snr"])
    plt.close(figures["raw_traces_random"])


def test_behaviour_failure_propagates_and_stale_outputs_are_removed(
    tmp_path, monkeypatch
):
    session_path = tmp_path / "ses-full001_date-test"
    session_path.mkdir()
    monkeypatch.setattr(
        overview,
        "load_session_data",
        lambda *args, **kwargs: {
            "dF": np.ones((2, 40)),
            "valid_cell_mask": np.array([True, True]),
            "neurons": np.array([1, 4]),
            "mean_img": np.zeros((2, 2)),
            "image_mask": [np.eye(2, dtype=bool)],
            "frame_rate": 2.0,
            "reward_frames": np.array([10, 25]),
            "landmark_entries": np.array([8, 23]),
            "landmark_exits": np.array([14, 29]),
            "session_path": session_path,
            "subject": "sub-07",
            "behaviour_analysis": {},
        },
    )
    monkeypatch.setattr(overview, "plot_fov_masks", lambda *a, **k: plt.figure())
    monkeypatch.setattr(overview, "plot_raw_traces", lambda *a, **k: plt.figure())
    monkeypatch.setattr(overview, "plot_reward_heatmap", lambda *a, **k: plt.figure())
    monkeypatch.setattr(
        overview, "plot_selected_cell_reward_psths", lambda *a, **k: plt.figure()
    )
    monkeypatch.setattr(
        overview,
        "plot_behaviour_alignment",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("raw mapping failed")),
    )
    save_dir = tmp_path / "plots"
    save_dir.mkdir()
    for filename in overview.BEHAVIOUR_OUTPUT_NAMES:
        (save_dir / filename).touch()

    with pytest.raises(RuntimeError, match="raw mapping failed"):
        overview.run_overview(
            "sub-07", "full001", tmp_path, save_dir=save_dir, show=False
        )

    assert all(
        not (save_dir / filename).exists()
        for filename in overview.BEHAVIOUR_OUTPUT_NAMES
    )
    plt.close("all")


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
