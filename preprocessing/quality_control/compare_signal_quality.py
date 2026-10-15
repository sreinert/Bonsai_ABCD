"""
Compare raw suite2p signal quality between mice, session by session.

Written to investigate why mouse 002 (GoalProgress cohort) shows no visible
calcium transients and no cross-validated event-aligned responses, in contrast
to mouse 010 (cohort3). Computes quality metrics directly from raw F.npy/Fneu.npy
(see signal_quality_metrics.py) so the comparison does not depend on suite2p
version or ROI-detection pipeline (mouse 002's ROIs were imported from an
anatomical Cellpose segmentation rather than suite2p's own functional detection).

Mice, sessions, output location, highlighted session and figure titles come from
a YAML settings file (see settings/signal_quality_002_vs_010.yaml):

    python compare_signal_quality.py settings/<file>.yaml [--recompute]

Per-session metrics are cached next to each session's suite2p output (or in the
mouse's `cache_dir`) by signal_quality_cache.py and reused on later runs; they are
recomputed automatically when the suite2p files or parameters change, or always
with --recompute. The same cache also holds the top-SNR trace windows used by
plot_high_snr_traces.py.

Produces one summary figure, one channel-2 control figure and one CSV table in
the settings' output_dir.
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from signal_quality_cache import load_settings, iter_sessions, get_session_results

# dataviz reference palette chrome (references/palette.md)
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"


def gather_results(settings, recompute=False):
    rows = []
    for mouse, ses, plane0_dir in iter_sessions(settings):
        metrics, _, source = get_session_results(
            plane0_dir, ses, cache_dir=mouse["cache_dir"], recompute=recompute,
            params=settings["params"], trace_params=settings["traces"])
        metrics = dict(metrics)
        if "error" in metrics:
            print(f"[warn] could not process {mouse['label']} / {ses}: {metrics['error']}")
        else:
            print(f"[{source}] {mouse['label']} / {ses}")
            if "chan2_error" in metrics:
                print(f"[warn] no channel-2 metrics for {mouse['label']} / {ses}: {metrics['chan2_error']}")
        metrics["mouse"] = mouse["label"]
        metrics["session"] = ses
        metrics["color"] = mouse["color"]
        rows.append(metrics)
    df = pd.DataFrame(rows)
    if "mean_img_mean" in df and "fov_chan2_mean" in df:
        df["chan1_chan2_ratio"] = df["mean_img_mean"] / df["fov_chan2_mean"]
    return df


def _strip_panel(ax, df, mice_cfg, metric, title, ylabel, highlight_session=None):
    mice = [m["label"] for m in mice_cfg]
    rng = np.random.default_rng(0)
    for i, mouse in enumerate(mice):
        sub = df[df["mouse"] == mouse].dropna(subset=[metric])
        if sub.empty:
            continue
        color = mice_cfg[i]["color"]
        x = i + rng.uniform(-0.12, 0.12, size=len(sub))
        is_highlight = sub["session"] == highlight_session
        ax.scatter(x[~is_highlight.values], sub.loc[~is_highlight, metric],
                   s=40, color=color, alpha=0.85, edgecolor="none", zorder=2)
        if is_highlight.any():
            ax.scatter(x[is_highlight.values], sub.loc[is_highlight, metric],
                       s=70, facecolor=color, edgecolor=INK_PRIMARY, linewidth=1.3, zorder=3)
            ax.annotate("dim session", (x[is_highlight.values][0], sub.loc[is_highlight, metric].values[0]),
                        xytext=(6, 6), textcoords="offset points", fontsize=8, color=INK_SECONDARY)
        median = np.median(sub[metric])
        ax.plot([i - 0.2, i + 0.2], [median, median], color=INK_PRIMARY, linewidth=2, zorder=4)
        ax.annotate(f"{median:.2g}", (i + 0.22, median), fontsize=8.5, color=INK_PRIMARY,
                    va="center")

    short_labels = [m.split(" (")[0] for m in mice]
    ax.set_xticks(range(len(mice)))
    ax.set_xticklabels(short_labels, fontsize=9.5)
    ax.set_xlim(-0.6, len(mice) - 0.4)
    ax.set_title(title, fontsize=10.5, color=INK_PRIMARY, loc="left")
    ax.set_ylabel(ylabel, fontsize=9, color=INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRIDLINE)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)


def _legend_handles(mice_cfg):
    return [plt.Line2D([0], [0], marker="o", linestyle="", color=m["color"],
                       label=m["label"], markersize=7) for m in mice_cfg]


def make_figure(df, settings):
    plt.rcParams["font.family"] = "sans-serif"
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    fig.patch.set_facecolor(SURFACE)

    mice_cfg = settings["mice"]
    highlight = settings["highlight_session"]
    _strip_panel(axes[0, 0], df, mice_cfg, "baseline_F_median",
                 "Raw baseline fluorescence", "median F (a.u., 8th pct.)")
    _strip_panel(axes[0, 1], df, mice_cfg, "mean_img_mean",
                 "Raw mean-image brightness", "mean pixel intensity (a.u.)")
    _strip_panel(axes[0, 2], df, mice_cfg, "median_snr",
                 "Median per-cell SNR", "peak dF/F (p99) / noise SD")
    _strip_panel(axes[1, 0], df, mice_cfg, "frac_cells_snr_gt_thresh",
                 "Cells with clear transients", f"fraction with SNR > {settings['params']['snr_threshold']}",
                 highlight_session=highlight)
    _strip_panel(axes[1, 1], df, mice_cfg, "frac_cells_skew_gt_thresh",
                 "Cells with skewed (transient-like) dF/F", f"fraction with skew > {settings['params']['skew_threshold']}",
                 highlight_session=highlight)
    _strip_panel(axes[1, 2], df, mice_cfg, "median_event_frac",
                 "Detected calcium events", f"median fraction of frames > {settings['params']['event_sigma']} SD",
                 highlight_session=highlight)

    fig.legend(handles=_legend_handles(mice_cfg), loc="upper center", ncol=len(mice_cfg), frameon=False,
               bbox_to_anchor=(0.5, 0.94), fontsize=10)

    title = settings["title"] or "Raw signal quality: " + " vs. ".join(m["label"] for m in mice_cfg)
    fig.suptitle(title, fontsize=10, color=INK_SECONDARY, y=0.995)

    fig.tight_layout(rect=[0, 0, 1, 0.86])
    return fig


def make_channel2_figure(df, settings):
    """Channel 2 (structural, activity-independent marker) as a control: same
    expression protocol in both mice, so its raw brightness should roughly
    match unless something about the imaging itself (not the functional
    indicator) differed for mouse 002.
    """
    plt.rcParams["font.family"] = "sans-serif"
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.patch.set_facecolor(SURFACE)

    mice_cfg = settings["mice"]
    highlight = settings["highlight_session"]
    _strip_panel(axes[0], df, mice_cfg, "fov_chan2_mean",
                 "Channel 2 (structural) FOV brightness", "mean pixel intensity (a.u.)",
                 highlight_session=highlight)
    _strip_panel(axes[1], df, mice_cfg, "roi_chan2_median_brightness",
                 "Channel 2 brightness within functional ROIs", "median ROI intensity (a.u.)",
                 highlight_session=highlight)
    _strip_panel(axes[2], df, mice_cfg, "chan1_chan2_ratio",
                 "Channel 1 / channel 2 brightness ratio", "ratio (higher = ch.1 relatively brighter)",
                 highlight_session=highlight)

    # mice without a red channel (e.g. cohort2) will not appear in these panels;
    # kept in the legend for consistency with the main figure.
    fig.legend(handles=_legend_handles(mice_cfg), loc="upper center", ncol=len(mice_cfg), frameon=False,
               bbox_to_anchor=(0.5, 0.88), fontsize=10)

    title = settings["chan2_title"] or "Channel 2 (structural, activity-independent) control"
    fig.suptitle(title, fontsize=10, color=INK_SECONDARY, y=0.995)

    fig.tight_layout(rect=[0, 0, 1, 0.72])
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("settings", help="YAML settings file (see settings/)")
    parser.add_argument("--recompute", action="store_true",
                        help="ignore cached per-session results and recompute all sessions")
    args = parser.parse_args()

    settings = load_settings(args.settings)
    out_dir, name = settings["output_dir"], settings["output_name"]
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{name}.csv")
    fig_path = os.path.join(out_dir, f"{name}.png")
    chan2_fig_path = os.path.join(out_dir, f"{name}_chan2.png")

    df = gather_results(settings, recompute=args.recompute)
    df.to_csv(csv_path, index=False)
    print(f"Saved per-session metrics table to {csv_path}")

    fig = make_figure(df, settings)
    fig.savefig(fig_path, dpi=200, facecolor=fig.get_facecolor())
    print(f"Saved summary figure to {fig_path}")

    fig2 = make_channel2_figure(df, settings)
    fig2.savefig(chan2_fig_path, dpi=200, facecolor=fig2.get_facecolor())
    print(f"Saved channel-2 control figure to {chan2_fig_path}")


if __name__ == "__main__":
    main()
