"""Combined T5 monotonic-trend and T3 alternation summary plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import alternation_analysis_helpers_v2 as alternation
import neural_analysis_helpers


XY_ANALYSES = {
    "AB_A": {"condition": "AB", "chosen_lm": "A", "slice": slice(0, None, 2)},
    "AB_B": {"condition": "AB", "chosen_lm": "B", "slice": slice(1, None, 2)},
    "BA_A": {"condition": "BA", "chosen_lm": "A", "slice": slice(1, None, 2)},
    "BA_B": {"condition": "BA", "chosen_lm": "B", "slice": slice(0, None, 2)},
}


def _load_npz_dict(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing analysis result: {path}\n"
            "Run the corresponding T3 analysis section before making the summary plots."
        )

    with np.load(path, allow_pickle=True) as saved:
        return {
            key: value.item()
            if isinstance(value, np.ndarray) and value.shape == ()
            else value
            for key, value in saved.items()
        }


def get_tracked_cell_pairs(idx_original_aligned, t5_cells, t3_index=3, t5_index=5):
    """Return one-to-one ``(T5 cell, T3 cell)`` pairs in ROICaT row order."""
    t5_rows = np.flatnonzero(np.isin(idx_original_aligned[t5_index], t5_cells))
    tracked_rows = t5_rows[np.isfinite(idx_original_aligned[t3_index, t5_rows])]

    pairs = []
    for row in tracked_rows:
        t5_cell = int(idx_original_aligned[t5_index, row])
        t3_cell = int(idx_original_aligned[t3_index, row])
        pairs.append((t5_cell, t3_cell))
    return pairs


def prepare_t3_summary_data(
    neurons,
    dF,
    session,
    stage="t3",
    yy_zscoring=True,
    regression_bins=30,
):
    """Rebuild plotting inputs and load saved results for every T3 analysis.

    This deliberately loads results from ``session['analysis_dir']`` rather than
    using notebook globals, which may contain T4 results after running an earlier
    section.
    """
    stage_id = int(stage.removeprefix("t"))
    analysis_dir = Path(session["analysis_dir"])
    bins_cpa = alternation.get_min_frames_between_lms(session)

    if session["stim_order"] == "random":
        ABB_patches, BAA_patches, _, _ = alternation.get_XYY_patches(
            session, precede_XY=True
        )
    elif session["stim_order"] == "pseudorandom":
        ABB_patches, BAA_patches, _, _ = alternation.get_XYY_patches(
            session, precede_XY=False
        )
    else:
        raise ValueError(f"Unknown stimulus order: {session['stim_order']!r}")

    events_AA = alternation.get_YY_events(session, BAA_patches)
    events_BB = alternation.get_YY_events(session, ABB_patches)
    binned_AA = alternation.get_reward_aligned_temporal_phase_binning_per_lm(
        neurons,
        dF,
        BAA_patches,
        events_AA,
        session,
        bins_cpa,
        condition="AA",
        zscoring=yy_zscoring,
        plot=False,
    )
    binned_BB = alternation.get_reward_aligned_temporal_phase_binning_per_lm(
        neurons,
        dF,
        ABB_patches,
        events_BB,
        session,
        bins_cpa,
        condition="BB",
        zscoring=yy_zscoring,
        plot=False,
    )

    yy_dir = analysis_dir / f"t{stage_id}_linear_regression_YY_diff_rew_aligned_XYrepeats_cpa"
    yy = {
        "AA": {
            "results": _load_npz_dict(yy_dir / "BA_YY_diff_linear_regression_results.npz"),
            "y_data": binned_AA,
            "condition": "BA",
        },
        "BB": {
            "results": _load_npz_dict(yy_dir / "AB_YY_diff_linear_regression_results.npz"),
            "y_data": binned_BB,
            "condition": "AB",
        },
    }

    A2_activity = alternation.get_Y2_activity(neurons, dF, session, BAA_patches)
    B2_activity = alternation.get_Y2_activity(neurons, dF, session, ABB_patches)
    y2_dir = analysis_dir / f"t{stage_id}_linear_regression_Y2_ramp_XYrepeats"
    y2 = {
        "A2": {
            "results": _load_npz_dict(y2_dir / "BA_Y2_ramp_linear_regression_results.npz"),
            "y_data": A2_activity,
            "condition": "BA",
        },
        "B2": {
            "results": _load_npz_dict(y2_dir / "AB_Y2_ramp_linear_regression_results.npz"),
            "y_data": B2_activity,
            "condition": "AB",
        },
    }

    AB_ordered, BA_ordered = alternation.get_XY_repeat_ordering(
        session, min_length=4, return_list=True
    )
    _, AB_patches, BA_patches, _, _, _ = alternation.get_repeating_XY_patches(
        session, min_length=4, return_list=True
    )
    patches = {"AB": AB_patches, "BA": BA_patches}
    ordered_patches = {"AB": AB_ordered, "BA": BA_ordered}

    xy_dir = analysis_dir / f"t{stage_id}_linear_regression_XY_order"
    xy = {}
    for name, params in XY_ANALYSES.items():
        condition = params["condition"]
        lm_slice = params["slice"]
        heatmap_lms = patches[condition][lm_slice]
        xy[name] = {
            "results": _load_npz_dict(
                xy_dir / f"{name}_XY_order_linear_regression_results.npz"
            ),
            "y_data": alternation.get_mean_lm_activity(
                session, neurons, dF, heatmap_lms
            ),
            "x_data": ordered_patches[condition][lm_slice],
            "heatmap_lms": heatmap_lms,
            **params,
        }

    return {
        "yy": yy,
        "y2": y2,
        "xy": xy,
        "bins_cpa": bins_cpa,
        "regression_bins": regression_bins,
        "yy_zscoring": yy_zscoring,
    }


def _add_row_label(fig, axes, label):
    top = max(ax.get_position().y1 for ax in axes)
    left = min(ax.get_position().x0 for ax in axes)
    fig.text(left, top + 0.005, label, ha="left", va="bottom", weight="bold", fontsize=12)


def _plot_unavailable(axes, message):
    for ax in axes:
        ax.set_axis_off()
    axes[0].text(0.02, 0.5, message, transform=axes[0].transAxes, va="center")


def plot_t5_monotonic_with_t3_summary(
    t5_cell,
    t3_cell,
    t5_dF,
    t5_session,
    max_window_activity,
    monotonic_trend_results,
    t3_dF,
    t3_session,
    t3_data,
    bins_goal=90,
    save_path=None,
):
    """Make one figure containing the T5 trend and all T3 alternation results."""
    fig = plt.figure(figsize=(25, 28), layout="constrained")
    outer = fig.add_gridspec(5, 1, height_ratios=[1.0, 1.05, 1.05, 1.05, 1.05])

    goal_gs = outer[0].subgridspec(1, 3, width_ratios=[1.1, 1.0, 0.85])
    goal_axes = [
        fig.add_subplot(goal_gs[0], projection="polar"),
        fig.add_subplot(goal_gs[1]),
        fig.add_subplot(goal_gs[2]),
    ]
    neural_analysis_helpers.plot_progress_with_monotonic_trend(
        dF=t5_dF,
        cell=t5_cell,
        event_frames=t5_session["event_idx"],
        ngoals=5,
        bins=bins_goal,
        stage="t5",
        session=t5_session,
        activity_by_cell=max_window_activity,
        trend_results=monotonic_trend_results,
        show_permutation=True,
        axes=goal_axes,
    )
    _add_row_label(fig, goal_axes, "T5 monotonic goal-progress trend")

    cpa_gs = outer[1].subgridspec(1, 4, width_ratios=[1.6, 1, 1.6, 1])
    cpa_axes = [fig.add_subplot(cpa_gs[i]) for i in range(4)]
    for offset, name in zip((0, 2), ("AA", "BB")):
        item = t3_data["yy"][name]
        axes = cpa_axes[offset : offset + 2]
        if t3_cell not in item["results"]["slopes"]:
            _plot_unavailable(axes, f"{name}: T3 cell {t3_cell} unavailable")
            continue
        alternation.plot_cpa_results(
            item["results"],
            [t3_cell],
            item["y_data"],
            t3_session,
            condition=item["condition"],
            bins=t3_data["bins_cpa"],
            zscored=t3_data["yy_zscoring"],
            axes=axes,
        )
        axes[0].set_title(f"{name} CPA | " + axes[0].get_title())
    _add_row_label(fig, cpa_axes, "T3 YY difference CPA")

    y2_gs = outer[2].subgridspec(1, 6, width_ratios=[1.5, 1, 1, 1.5, 1, 1])
    y2_axes = [fig.add_subplot(y2_gs[i]) for i in range(6)]
    for offset, name in zip((0, 3), ("A2", "B2")):
        item = t3_data["y2"][name]
        axes = y2_axes[offset : offset + 3]
        if t3_cell not in item["results"]["slopes"]:
            _plot_unavailable(axes, f"{name}: T3 cell {t3_cell} unavailable")
            continue
        alternation.plot_linear_regression_results(
            item["results"],
            [t3_cell],
            t3_dF,
            t3_session,
            item["y_data"],
            condition=item["condition"],
            data_type="Y2_ramp",
            bins=t3_data["bins_cpa"],
            sort_heatmap=True,
            cluster_repeats=True,
            zscoring=False,
            axes=axes,
        )
        axes[0].set_title(f"{name} ramp | " + axes[0].get_title())
    _add_row_label(fig, y2_axes, "T3 Y2 ramp")

    for row, names in zip((3, 4), (("AB_A", "AB_B"), ("BA_A", "BA_B"))):
        xy_gs = outer[row].subgridspec(1, 6, width_ratios=[1.5, 1, 1, 1.5, 1, 1])
        xy_axes = [fig.add_subplot(xy_gs[i]) for i in range(6)]
        for offset, name in zip((0, 3), names):
            item = t3_data["xy"][name]
            axes = xy_axes[offset : offset + 3]
            if t3_cell not in item["results"]["slopes"]:
                _plot_unavailable(axes, f"{name}: T3 cell {t3_cell} unavailable")
                continue
            alternation.plot_linear_regression_results(
                item["results"],
                [t3_cell],
                t3_dF,
                t3_session,
                item["y_data"],
                x_data=item["x_data"],
                heatmap_lms=item["heatmap_lms"],
                condition=item["condition"],
                chosen_lm=item["chosen_lm"],
                data_type="XY_order",
                bins=t3_data["regression_bins"],
                sort_heatmap=True,
                cluster_repeats=False,
                zscoring=False,
                axes=axes,
            )
            axes[0].set_title(f"{name} | " + axes[0].get_title())
        _add_row_label(fig, xy_axes, f"T3 XY order: {names[0][:2]} sequence")

    rho = monotonic_trend_results["cell_score"][t5_cell]
    pvalue = monotonic_trend_results["pvalue"][t5_cell]
    fig.suptitle(
        f"T5 cell {t5_cell} → T3 cell {t3_cell} | "
        f"monotonic trend: $\\rho$={rho:.2f}, p={pvalue:.3g}",
        fontsize=16,
        weight="bold",
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")

    return fig
