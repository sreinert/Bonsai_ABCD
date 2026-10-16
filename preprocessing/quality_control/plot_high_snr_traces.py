"""
For each session in a settings file (see settings/ and compare_signal_quality.py),
plot the raw dF/F traces of the highest-SNR cells over a window (default 20000
frames) taken from the middle of the recording. This is the direct visual check
behind the summary metrics in compare_signal_quality.py: does the "best" cell in
a session actually look like it has calcium transients, or just noise that
happens to have a high peak-to-noise ratio?

    python plot_high_snr_traces.py settings/<file>.yaml [--recompute]

Cells are ranked by the same per-cell SNR used in signal_quality_metrics.py
(99th-percentile dF/F over a frame-to-frame-noise estimate). Up to n_cells
(settings 'traces' block, default 8) are shown per session; if a session has none
above the SNR threshold, the top cells by SNR are shown anyway, labelled as such,
so a "no good cells" session still produces a plot rather than being silently
skipped. Cell selection and the trace window are done in
signal_quality_cache.select_top_cells and cached per session (shared with
compare_signal_quality.py), so this script only plots unless the cache is stale
or --recompute is given.

One PNG per session is saved to <output_dir>/<output_name>_high_snr_traces/.
"""
import os
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

try:  # Package import used by the full-session pipeline.
    from .signal_quality_cache import get_session_results, iter_sessions, load_settings
except ImportError:  # Direct execution from preprocessing/quality_control.
    from signal_quality_cache import get_session_results, iter_sessions, load_settings

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"


def plot_session(
    mouse_label,
    session,
    top,
    color,
    snr_threshold,
    out_dir=None,
    *,
    out_path=None,
):
    """Plot one session and return its output path, or ``None`` on data error."""
    if "error" in top:
        print(f"[warn] skipping {mouse_label} / {session}: {top['error']}")
        return None

    snr = top["snr"]
    n_frames = top["n_frames"]
    fs = top["fs"] if np.isfinite(top["fs"]) and top["fs"] else 1.0
    n_show = len(snr)
    start, end = top["start"], top["end"]
    passing, n_finite, used_fallback = top["n_passing"], top["n_finite"], top["used_fallback"]
    t = (np.arange(start, end) - start) / fs

    fig, ax = plt.subplots(figsize=(14, max(4, 1.1 * n_show)))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # Plot dF/F z-scored by each cell's own noise SD (same units as the SNR
    # metric itself), not raw dF/F: mouse 002's baseline F0 is so close to the
    # noise floor that raw dF/F spikes are enormous and swamp the plot even
    # for its "best" cells, making traces from very dim vs. bright sessions
    # visually incomparable. z-scoring puts every cell on the same noise-SD
    # scale regardless of its absolute brightness.
    offset_step = 12.0  # noise-SD units between stacked traces
    for i in range(n_show):
        trace = top["dff"][i] / top["noise_sd"][i]
        ax.plot(t, trace + i * offset_step, color=color, linewidth=0.5)
        roi_idx = top["roi_idx"][i]
        ax.text(t[0] - 0.01 * (t[-1] - t[0]), i * offset_step, f"ROI {roi_idx}  SNR={snr[i]:.1f}",
                fontsize=8, color=INK_SECONDARY, ha="right", va="center")

    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-offset_step, n_show * offset_step)
    ax.set_yticks([])
    ax.set_xlabel("time in window (s)  |  amplitude in units of each cell's own noise SD", fontsize=9.5, color=INK_SECONDARY)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRIDLINE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8.5)

    n_total = top["n_total"]
    n_nan = n_total - n_finite
    fallback_note = f"  (no cells passed SNR>{snr_threshold} — showing best available anyway)" if used_fallback else ""
    nan_note = f", {n_nan} cells had non-finite SNR (baseline ≤ 0)" if n_nan > 0 else ""
    ax.set_title(
        f"{mouse_label}  —  {session}\n"
        f"top {n_show} cells by SNR ({passing}/{n_total} cells pass SNR>{snr_threshold}), "
        f"frames {start}–{end} of {n_frames}{fallback_note}{nan_note}",
        fontsize=10, color=INK_PRIMARY, loc="left")

    fig.tight_layout()
    if out_path is None:
        if out_dir is None:
            raise ValueError("either out_dir or out_path is required")
        out_path = os.path.join(out_dir, f"{mouse_label.split(' ')[0]}_{session}.png")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("settings", help="YAML settings file (see settings/)")
    parser.add_argument("--recompute", action="store_true",
                        help="ignore cached per-session results and recompute all sessions")
    args = parser.parse_args()

    settings = load_settings(args.settings)
    out_dir = os.path.join(settings["output_dir"], f"{settings['output_name']}_high_snr_traces")
    os.makedirs(out_dir, exist_ok=True)
    for mouse, session, plane0_dir in iter_sessions(settings):
        _, top, source = get_session_results(
            plane0_dir, session, cache_dir=mouse["cache_dir"], recompute=args.recompute,
            params=settings["params"], trace_params=settings["traces"])
        print(f"[{source}] {mouse['label']} / {session}")
        plot_session(mouse["label"], session, top, mouse["color"],
                     settings["params"]["snr_threshold"], out_dir)


if __name__ == "__main__":
    main()
