"""
Compare two suite2p ROI-detection runs of the same session (e.g. cellpose vs.
functional/sparsery detection) to decide which one to take forward to NWB
processing and alignment.

Sessions and paths come from a YAML settings file
(see settings/detection_cellpose_vs_functional.yaml):

    python compare_detection_pipelines.py settings/<file>.yaml

For each session, both runs are scored on all of their ROIs (iscell kept as a
column, so counts can be read with and without suite2p's classifier):

- Trace quality, using the same definitions as signal_quality_metrics.py (noise SD
  from the median absolute frame-to-frame difference, 99th-percentile peak, skew,
  fraction of frames > event_sigma noise SDs, F/Fneu correlation), computed on the
  neuropil-corrected trace F - neuropil_coeff * Fneu and on raw F. SNR is
  (p99 - p8) / noise SD in F units, which equals signal_quality_metrics' dF/F-based
  SNR when the baseline is positive but stays finite when the neuropil-corrected
  baseline is <= 0.
- Channel-2 positivity, computed identically for both runs from ops['meanImg_chan2']
  (not redcell.npy, which is missing when ROIs were imported from a cellpose
  segmentation): ROI mean / mean of a surrounding annulus that excludes all ROIs.
- ROI matching between the runs: pixel-mask IoU for every pair, one-to-one
  assignment (Hungarian), matched if IoU >= iou_threshold. The runs may have been
  registered separately, so the shift between their mean images is estimated
  (phase correlation) and applied to run B's ROIs before matching.
- For matched pairs (the same cell found by both runs): correlation of the two
  neuropil-corrected traces, and paired SNR/skew/noise/amplitude.

Reads only from the suite2p folders; writes figures and CSVs to
<output_dir>/<session label>/ plus <output_dir>/summary.csv. Sessions whose
F.npy is missing in either run (e.g. a run still in progress) are skipped.
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from scipy import stats, ndimage
from scipy.optimize import linear_sum_assignment
from skimage.registration import phase_cross_correlation

# dataviz reference palette (references/palette.md), categorical slots 1-3 in order
COLOR_A = "#2a78d6"        # run A (e.g. cellpose)
COLOR_B = "#eb6834"        # run B (e.g. functional)
COLOR_MATCHED = "#1baf7a"  # ROIs found by both
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"

DEFAULT_PARAMS = {
    "neuropil_coeff": 0.7,
    "baseline_percentile": 8,
    "event_sigma": 3,
    "snr_threshold": 5,
    "iou_threshold": 0.3,
    "chan2_ratio_threshold": 1.2,
    "annulus_inner": 2,
    "annulus_outer": 8,
    "n_example_traces": 8,
    "trace_window_s": 120,
}


# ---------------------------------------------------------------- settings

def load_settings(path):
    with open(path) as f:
        settings = yaml.safe_load(f)
    if not settings or "sessions" not in settings:
        raise ValueError(f"{path}: settings file needs a 'sessions' list")
    settings["output_dir"] = os.path.expanduser(settings.get("output_dir", "~/Desktop/detection_comparison"))
    settings["params"] = {**DEFAULT_PARAMS, **(settings.get("params") or {})}
    settings.setdefault("run_a", {"label": "cellpose", "subpath": "funcimg/suite2p/plane0"})
    settings.setdefault("run_b", {"label": "functional", "subpath": "funcimg/func_suite2p/plane0"})
    for i, ses in enumerate(settings["sessions"]):
        missing = [k for k in ("label", "path") if k not in ses]
        if missing:
            raise ValueError(f"{path}: session #{i} is missing {missing}")
        ses["path"] = os.path.expanduser(ses["path"])
    return settings


# ---------------------------------------------------------------- loading and per-ROI metrics

def load_run(plane0_dir):
    """Load the suite2p files needed for the comparison from one plane0 folder."""
    run = {
        "plane0_dir": plane0_dir,
        "F": np.load(os.path.join(plane0_dir, "F.npy")).astype(np.float64),
        "Fneu": np.load(os.path.join(plane0_dir, "Fneu.npy")).astype(np.float64),
        "iscell": np.load(os.path.join(plane0_dir, "iscell.npy")),
        "stat": np.load(os.path.join(plane0_dir, "stat.npy"), allow_pickle=True),
        "ops": np.load(os.path.join(plane0_dir, "ops.npy"), allow_pickle=True).item(),
    }
    redcell_path = os.path.join(plane0_dir, "redcell.npy")
    run["redcell"] = np.load(redcell_path) if os.path.exists(redcell_path) else None
    return run


def noise_sd(x):
    """Frame-to-frame noise SD (median absolute deviation of the first difference),
    as in signal_quality_metrics.load_percell_traces."""
    d = np.diff(x, axis=1)
    return 1.4826 * np.median(np.abs(d - np.median(d, axis=1, keepdims=True)), axis=1) / np.sqrt(2)


def trace_metrics(X, Fneu, p, prefix):
    """Per-ROI quality metrics for traces X (n_roi, n_frames), in X's own units."""
    sd = noise_sd(X)
    base = np.percentile(X, p["baseline_percentile"], axis=1)
    peak = np.percentile(X, 99, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = (peak - base) / sd
    event_frac = np.mean((X - base[:, None]) > p["event_sigma"] * sd[:, None], axis=1)
    out = {
        f"{prefix}noise_sd": sd,
        f"{prefix}baseline": base,
        f"{prefix}snr": snr,
        f"{prefix}skew": stats.skew(X, axis=1),
        f"{prefix}event_frac": event_frac,
    }
    if prefix == "raw_":
        out["neu_corr"] = np.array([np.corrcoef(X[i], Fneu[i])[0, 1] for i in range(len(X))])
    return out


def roi_masks(stat, shape, shift=(0, 0)):
    """List of flat pixel-index arrays per ROI, optionally shifted by (dy, dx)."""
    Ly, Lx = shape
    dy, dx = int(round(shift[0])), int(round(shift[1]))
    masks = []
    for s in stat:
        y, x = s["ypix"] + dy, s["xpix"] + dx
        ok = (y >= 0) & (y < Ly) & (x >= 0) & (x < Lx)
        masks.append(np.ravel_multi_index((y[ok], x[ok]), shape))
    return masks


def chan2_ratio(stat, mean_img2, p):
    """ROI mean of the chan2 mean image divided by the mean of an annulus around it
    (annulus_inner..annulus_outer px), excluding every ROI's pixels from the annulus."""
    shape = mean_img2.shape
    all_rois = np.zeros(shape, bool)
    for s in stat:
        all_rois[s["ypix"], s["xpix"]] = True
    ratios = np.full(len(stat), np.nan)
    for i, s in enumerate(stat):
        m = np.zeros(shape, bool)
        m[s["ypix"], s["xpix"]] = True
        inner = ndimage.binary_dilation(m, iterations=p["annulus_inner"])
        outer = ndimage.binary_dilation(m, iterations=p["annulus_outer"])
        ring = outer & ~inner & ~all_rois
        if ring.sum() > 10:
            ratios[i] = mean_img2[m].mean() / mean_img2[ring].mean()
    return ratios


def roi_table(run, p, label):
    F, Fneu, stat = run["F"], run["Fneu"], run["stat"]
    Fc = F - p["neuropil_coeff"] * Fneu
    df = pd.DataFrame({
        "run": label,
        "roi": np.arange(len(stat)),
        "iscell": run["iscell"][:, 0].astype(bool),
        "iscell_prob": run["iscell"][:, 1],
        "npix": [s["npix"] for s in stat],
        "radius": [s.get("radius", np.nan) for s in stat],
        "compact": [s.get("compact", np.nan) for s in stat],
        "aspect_ratio": [s.get("aspect_ratio", np.nan) for s in stat],
        "med_y": [s["med"][0] for s in stat],
        "med_x": [s["med"][1] for s in stat],
        **trace_metrics(Fc, Fneu, p, ""),
        **trace_metrics(F, Fneu, p, "raw_"),
    })
    mean_img2 = run["ops"].get("meanImg_chan2")
    df["chan2_ratio"] = chan2_ratio(stat, mean_img2, p) if mean_img2 is not None else np.nan
    df["chan2_pos"] = df["chan2_ratio"] > p["chan2_ratio_threshold"]
    df["redcell_prob"] = run["redcell"][:, 1] if run["redcell"] is not None else np.nan
    df["good"] = df["snr"] > p["snr_threshold"]
    return df, Fc


# ---------------------------------------------------------------- matching

def match_rois(stat_a, stat_b, shape, shift, iou_threshold):
    """One-to-one ROI matching by pixel-mask IoU. Returns (pairs, iou matrix), where
    pairs is a DataFrame with roi_a, roi_b, iou, centroid distance."""
    ma = roi_masks(stat_a, shape)
    mb = roi_masks(stat_b, shape, shift)
    n_pix = shape[0] * shape[1]
    # sparse ROI x pixel indicator matrices -> intersections in one product
    from scipy.sparse import csr_matrix
    def indicator(masks):
        rows = np.concatenate([np.full(len(m), i) for i, m in enumerate(masks)])
        cols = np.concatenate(masks)
        return csr_matrix((np.ones(len(cols)), (rows, cols)), shape=(len(masks), n_pix))
    A, B = indicator(ma), indicator(mb)
    inter = (A @ B.T).toarray()
    size_a = np.array([len(m) for m in ma])[:, None]
    size_b = np.array([len(m) for m in mb])[None, :]
    iou = inter / (size_a + size_b - inter)

    ra, rb = linear_sum_assignment(-iou)
    keep = iou[ra, rb] >= iou_threshold
    ra, rb = ra[keep], rb[keep]
    ca = np.array([s["med"] for s in stat_a])[ra]
    cb = np.array([s["med"] for s in stat_b])[rb] + np.array(shift)
    pairs = pd.DataFrame({"roi_a": ra, "roi_b": rb, "iou": iou[ra, rb],
                          "centroid_dist": np.linalg.norm(ca - cb, axis=1)})
    return pairs, iou


# ---------------------------------------------------------------- plotting helpers

def style_ax(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, fontsize=10, color=INK_PRIMARY, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=INK_SECONDARY)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRIDLINE)
    ax.set_facecolor(SURFACE)


def new_fig(*args, **kwargs):
    plt.rcParams["font.family"] = "sans-serif"
    fig, axes = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


def draw_contours(ax, stat, shape, color, shift=(0, 0), lw=0.8, ls="-"):
    for s in stat:
        m = np.zeros(shape, float)
        y = np.clip(s["ypix"] + int(round(shift[0])), 0, shape[0] - 1)
        x = np.clip(s["xpix"] + int(round(shift[1])), 0, shape[1] - 1)
        m[y, x] = 1
        ax.contour(m, levels=[0.5], colors=[color], linewidths=lw, linestyles=ls)


def show_img(ax, img, title):
    lo, hi = np.percentile(img, [1, 99.5])
    ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
    ax.set_title(title, fontsize=10, color=INK_PRIMARY, loc="left")
    ax.axis("off")


# ---------------------------------------------------------------- figures

def fig_overlay(run_a, run_b, df_a, df_b, pairs, shift, la, lb, title):
    ops = run_a["ops"]
    shape = ops["meanImg"].shape
    imgs = [("chan1 mean image", ops["meanImg"])]
    if ops.get("meanImg_chan2") is not None:
        imgs.append(("chan2 mean image", ops["meanImg_chan2"]))
    fig, axes = new_fig(1, len(imgs), figsize=(8 * len(imgs), 8.6))
    axes = np.atleast_1d(axes)
    matched_a, matched_b = set(pairs["roi_a"]), set(pairs["roi_b"])
    for ax, (name, img) in zip(axes, imgs):
        show_img(ax, img, name)
        st_a, st_b = run_a["stat"], run_b["stat"]
        draw_contours(ax, [st_a[i] for i in sorted(matched_a)], shape, COLOR_MATCHED, lw=1.0)
        draw_contours(ax, [s for i, s in enumerate(st_a) if i not in matched_a], shape, COLOR_A, lw=1.0)
        draw_contours(ax, [s for i, s in enumerate(st_b) if i not in matched_b], shape, COLOR_B, shift, lw=1.0)
        if name.startswith("chan2"):
            for df, col in ((df_a, COLOR_A), (df_b, COLOR_B)):
                pos = df[df["chan2_pos"]]
                off = shift if col == COLOR_B else (0, 0)
                ax.scatter(pos["med_x"] + off[1], pos["med_y"] + off[0], marker="+", s=30,
                           color="white", linewidths=0.8)
    handles = [plt.Line2D([0], [0], color=COLOR_MATCHED, lw=2, label=f"both ({len(pairs)})"),
               plt.Line2D([0], [0], color=COLOR_A, lw=2, label=f"{la} only ({len(df_a) - len(pairs)})"),
               plt.Line2D([0], [0], color=COLOR_B, lw=2, label=f"{lb} only ({len(df_b) - len(pairs)})"),
               plt.Line2D([0], [0], color="white", marker="+", lw=0, markeredgecolor=INK_PRIMARY,
                          label="chan2+ (white + on chan2 image)")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.97))
    fig.suptitle(title, fontsize=10, color=INK_SECONDARY, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def fig_summary(df_a, df_b, pairs, la, lb, p, title):
    df_a = df_a.assign(group=np.where(df_a["roi"].isin(pairs["roi_a"]), "both", f"{la} only"))
    df_b = df_b.assign(group=np.where(df_b["roi"].isin(pairs["roi_b"]), "both", f"{lb} only"))
    fig, axes = new_fig(2, 3, figsize=(15, 8.5))

    # counts
    ax = axes[0, 0]
    cats = ["all ROIs", "iscell", f"SNR > {p['snr_threshold']}", f"iscell &\nSNR > {p['snr_threshold']}",
            "chan2+", f"chan2+ &\nSNR > {p['snr_threshold']}"]
    def counts(d):
        return [len(d), d["iscell"].sum(), d["good"].sum(), (d["iscell"] & d["good"]).sum(),
                d["chan2_pos"].sum(), (d["chan2_pos"] & d["good"]).sum()]
    x = np.arange(len(cats))
    for off, d, col, lab in ((-0.2, df_a, COLOR_A, la), (0.2, df_b, COLOR_B, lb)):
        c = counts(d)
        ax.bar(x + off, c, width=0.38, color=col, label=lab, zorder=2)
        for xi, ci in zip(x + off, c):
            ax.text(xi, ci, str(int(ci)), ha="center", va="bottom", fontsize=7.5, color=INK_SECONDARY)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=7.5)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", color=GRIDLINE, lw=0.8, zorder=0)
    style_ax(ax, "ROI counts", ylabel="number of ROIs")

    # distributions by run x group
    panels = [("snr", "SNR (neuropil-corrected)", "(p99 − p8) / noise SD", False),
              ("skew", "Skew (neuropil-corrected)", "skew", False),
              ("event_frac", "Event fraction", f"fraction of frames > {p['event_sigma']} SD", False),
              ("neu_corr", "Raw F vs. neuropil correlation", "Pearson r (lower = less contamination)", False),
              ("chan2_ratio", "Chan2 ROI / surround ratio", "ratio", False)]
    groups = [(df_a, "both", COLOR_A, f"{la}\nboth"), (df_a, f"{la} only", COLOR_A, f"{la}\nonly"),
              (df_b, "both", COLOR_B, f"{lb}\nboth"), (df_b, f"{lb} only", COLOR_B, f"{lb}\nonly")]
    rng = np.random.default_rng(0)
    for ax, (col, ttl, ylab, logy) in zip(axes.ravel()[1:], panels):
        for i, (d, g, c, _) in enumerate(groups):
            v = d.loc[d["group"] == g, col].dropna().values
            if logy:
                v = v[v > 0]
            if len(v) == 0:
                continue
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=12, color=c,
                       alpha=0.35 if g == "both" else 0.85, edgecolor="none", zorder=2)
            med = np.median(v)
            ax.plot([i - 0.25, i + 0.25], [med, med], color=INK_PRIMARY, lw=2, zorder=3)
            ax.annotate(f"{med:.2g}", (i + 0.27, med), fontsize=8, color=INK_PRIMARY, va="center")
        if col == "snr":
            ax.axhline(p["snr_threshold"], color=INK_MUTED, lw=0.8, ls="--")
        if col == "chan2_ratio":
            ax.axhline(p["chan2_ratio_threshold"], color=INK_MUTED, lw=0.8, ls="--")
        if logy:
            ax.set_yscale("log")
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([g[3] for g in groups], fontsize=8)
        ax.grid(axis="y", color=GRIDLINE, lw=0.8, zorder=0)
        style_ax(ax, ttl, ylabel=ylab)
    fig.suptitle(title, fontsize=10, color=INK_SECONDARY, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def fig_matched_pairs(pairs, la, lb, p, title):
    fig, axes = new_fig(1, 4, figsize=(17, 4.5))
    ax = axes[0]
    ax.scatter(pairs["snr_a"], pairs["snr_b"], s=18, color=COLOR_MATCHED, edgecolor="none", zorder=2)
    lim = [np.floor(np.nanmin(pairs[["snr_a", "snr_b"]].values)) - 0.5,
           np.ceil(np.nanmax(pairs[["snr_a", "snr_b"]].values)) + 0.5]
    ax.plot(lim, lim, color=INK_MUTED, lw=1, ls="--", zorder=1)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    n_a_better = int((pairs["snr_a"] > pairs["snr_b"]).sum())
    style_ax(ax, f"SNR of the same cell ({n_a_better}/{len(pairs)} higher in {la})", f"SNR {la}", f"SNR {lb}")

    ax = axes[1]
    ax.hist(pairs["trace_corr"], bins=np.linspace(-0.2, 1, 25), color=COLOR_MATCHED, zorder=2)
    ax.axvline(pairs["trace_corr"].median(), color=INK_PRIMARY, lw=1.5)
    style_ax(ax, f"Trace correlation, median r = {pairs['trace_corr'].median():.2f}",
             "Pearson r (neuropil-corrected traces)", "matched pairs")

    ax = axes[2]
    ax.hist(np.log2(pairs["amp_ratio"].clip(1e-3)), bins=25, color=COLOR_MATCHED, zorder=2)
    ax.axvline(0, color=INK_MUTED, lw=1, ls="--")
    style_ax(ax, f"Transient amplitude ({la} / {lb})", "log2 ratio of (p99 − p8)", "matched pairs")

    ax = axes[3]
    ax.hist(pairs["iou"], bins=np.linspace(p["iou_threshold"], 1, 20), color=COLOR_MATCHED, zorder=2)
    style_ax(ax, f"Mask overlap, median IoU = {pairs['iou'].median():.2f}", "IoU", "matched pairs")
    fig.suptitle(title, fontsize=10, color=INK_SECONDARY, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def _window(n_frames, fs, window_s):
    w = int(window_s * fs)
    mid = n_frames // 2
    return max(0, mid - w // 2), min(n_frames, mid + w // 2)


def fig_matched_traces(pairs, Fc_a, Fc_b, df_a, df_b, la, lb, fs, p, title):
    n = p["n_example_traces"]
    order = pairs.assign(mean_snr=(pairs["snr_a"] + pairs["snr_b"]) / 2).sort_values("mean_snr", ascending=False)
    n_top = min(len(order), n - n // 4)
    rest = order.iloc[n_top:]
    pick = pd.concat([order.iloc[:n_top], rest.sample(min(len(rest), n // 4), random_state=0)])
    start, end = _window(Fc_a.shape[1], fs, p["trace_window_s"])
    t = np.arange(end - start) / fs
    fig, ax = new_fig(figsize=(15, 1.3 * len(pick) + 1.5))
    step = 14.0
    for k, (_, r) in enumerate(pick.iterrows()):
        a, b = int(r["roi_a"]), int(r["roi_b"])
        ya = (Fc_a[a, start:end] - df_a.loc[a, "baseline"]) / df_a.loc[a, "noise_sd"]
        yb = (Fc_b[b, start:end] - df_b.loc[b, "baseline"]) / df_b.loc[b, "noise_sd"]
        off = -k * step
        ax.plot(t, ya + off, color=COLOR_A, lw=0.6)
        ax.plot(t, yb + off, color=COLOR_B, lw=0.6, alpha=0.8)
        ax.text(t[-1] * 1.005, off,
                f"{la} {a} (SNR {r['snr_a']:.1f})\n{lb} {b} (SNR {r['snr_b']:.1f})\nr = {r['trace_corr']:.2f}, IoU {r['iou']:.2f}",
                fontsize=7.5, color=INK_SECONDARY, va="center")
    ax.set_yticks([])
    ax.set_xlim(0, t[-1])
    ax.spines["left"].set_visible(False)
    style_ax(ax, f"Same cell in both runs: top {n_top} by SNR + {len(pick) - n_top} random matched pairs "
                 f"(frames {start}–{end}, z-scored by each trace's own noise SD)",
             "time in window (s)")
    ax.legend(handles=[plt.Line2D([0], [0], color=COLOR_A, lw=2, label=la),
                       plt.Line2D([0], [0], color=COLOR_B, lw=2, label=lb)],
              loc="upper right", bbox_to_anchor=(1.0, 1.06), ncol=2, frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=10, color=INK_SECONDARY, y=0.998)
    fig.tight_layout(rect=[0, 0, 0.9, 0.98])
    return fig


def fig_unique_traces(pairs, Fc_a, Fc_b, df_a, df_b, la, lb, fs, p, title):
    n = p["n_example_traces"]
    ua = df_a[~df_a["roi"].isin(pairs["roi_a"])].sort_values("snr", ascending=False).head(n)
    ub = df_b[~df_b["roi"].isin(pairs["roi_b"])].sort_values("snr", ascending=False).head(n)
    start, end = _window(Fc_a.shape[1], fs, p["trace_window_s"])
    t = np.arange(end - start) / fs
    fig, axes = new_fig(1, 2, figsize=(18, 1.1 * max(len(ua), len(ub), 1) + 1.8), sharey=True)
    step = 14.0
    for ax, u, Fc, col, lab, n_unique in ((axes[0], ua, Fc_a, COLOR_A, la, (~df_a["roi"].isin(pairs["roi_a"])).sum()),
                                          (axes[1], ub, Fc_b, COLOR_B, lb, (~df_b["roi"].isin(pairs["roi_b"])).sum())):
        for k, (_, r) in enumerate(u.iterrows()):
            i = int(r["roi"])
            y = (Fc[i, start:end] - r["baseline"]) / r["noise_sd"]
            ax.plot(t, y - k * step, color=col, lw=0.6)
            ax.text(t[-1] * 1.01, -k * step,
                    f"ROI {i}  SNR {r['snr']:.1f}\niscell {int(r['iscell'])}, chan2 {r['chan2_ratio']:.2f}\n"
                    f"npix {int(r['npix'])}, compact {r['compact']:.2f}",
                    fontsize=7, color=INK_SECONDARY, va="center")
        ax.set_yticks([])
        ax.set_xlim(0, t[-1])
        ax.spines["left"].set_visible(False)
        style_ax(ax, f"Only in {lab}: top {len(u)} of {n_unique} by SNR", "time in window (s)")
    fig.suptitle(title + f"  |  frames {start}–{end}, z-scored by each trace's own noise SD",
                 fontsize=10, color=INK_SECONDARY, y=0.998)
    fig.tight_layout(rect=[0, 0, 0.94, 0.97])
    return fig


# ---------------------------------------------------------------- per-session pipeline

def compare_session(ses, settings):
    p = settings["params"]
    ra, rb = settings["run_a"], settings["run_b"]
    la, lb = ra["label"], rb["label"]
    dir_a = os.path.join(ses["path"], ra["subpath"])
    dir_b = os.path.join(ses["path"], rb["subpath"])
    for d in (dir_a, dir_b):
        if not os.path.exists(os.path.join(d, "F.npy")):
            print(f"[skip] {ses['label']}: no F.npy in {d}")
            return None

    print(f"[load] {ses['label']}")
    run_a, run_b = load_run(dir_a), load_run(dir_b)
    shape = run_a["ops"]["meanImg"].shape
    shift = phase_cross_correlation(run_a["ops"]["meanImg"], run_b["ops"]["meanImg"], upsample_factor=10)[0]
    print(f"  mean-image shift {la} vs {lb}: dy={shift[0]:.1f}, dx={shift[1]:.1f} px")

    df_a, Fc_a = roi_table(run_a, p, la)
    df_b, Fc_b = roi_table(run_b, p, lb)
    pairs, _ = match_rois(run_a["stat"], run_b["stat"], shape, shift, p["iou_threshold"])

    a_idx, b_idx = pairs["roi_a"].values, pairs["roi_b"].values
    pairs["trace_corr"] = [np.corrcoef(Fc_a[i], Fc_b[j])[0, 1] for i, j in zip(a_idx, b_idx)]
    for col in ("snr", "skew", "noise_sd", "event_frac", "chan2_ratio", "iscell"):
        pairs[f"{col}_a"] = df_a.loc[a_idx, col].values
        pairs[f"{col}_b"] = df_b.loc[b_idx, col].values
    amp_a = np.percentile(Fc_a[a_idx], 99, axis=1) - df_a.loc[a_idx, "baseline"].values
    amp_b = np.percentile(Fc_b[b_idx], 99, axis=1) - df_b.loc[b_idx, "baseline"].values
    pairs["amp_ratio"] = amp_a / amp_b

    df_a["matched"] = df_a["roi"].isin(a_idx)
    df_b["matched"] = df_b["roi"].isin(b_idx)
    df_a["match_roi"] = df_a["roi"].map(dict(zip(a_idx, b_idx)))
    df_b["match_roi"] = df_b["roi"].map(dict(zip(b_idx, a_idx)))

    out_dir = os.path.join(settings["output_dir"], ses["label"])
    os.makedirs(out_dir, exist_ok=True)
    pd.concat([df_a, df_b]).to_csv(os.path.join(out_dir, "rois.csv"), index=False)
    pairs.to_csv(os.path.join(out_dir, "matched_pairs.csv"), index=False)

    fs = run_a["ops"].get("fs", 45)
    title = f"{ses['label']}  —  {la} vs. {lb}"
    figs = {
        "roi_overlay": fig_overlay(run_a, run_b, df_a, df_b, pairs, shift, la, lb, title),
        "summary": fig_summary(df_a, df_b, pairs, la, lb, p, title),
        "matched_pairs": fig_matched_pairs(pairs, la, lb, p, title),
        "matched_traces": fig_matched_traces(pairs, Fc_a, Fc_b, df_a, df_b, la, lb, fs, p, title),
        "unique_traces": fig_unique_traces(pairs, Fc_a, Fc_b, df_a, df_b, la, lb, fs, p, title),
    }
    for name, fig in figs.items():
        path = os.path.join(out_dir, f"{name}.png")
        fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
    print(f"  saved figures and CSVs to {out_dir}")

    rows = []
    for lab, d in ((la, df_a), (lb, df_b)):
        rows.append({
            "session": ses["label"], "run": lab, "n_roi": len(d), "n_iscell": int(d["iscell"].sum()),
            "n_snr_gt_thresh": int(d["good"].sum()), "n_iscell_and_snr": int((d["iscell"] & d["good"]).sum()),
            "n_chan2_pos": int(d["chan2_pos"].sum()), "n_chan2_pos_and_snr": int((d["chan2_pos"] & d["good"]).sum()),
            "n_matched": len(pairs), "n_unique": int((~d["matched"]).sum()),
            "n_unique_snr_gt_thresh": int((~d["matched"] & d["good"]).sum()),
            "median_snr": d["snr"].median(), "median_snr_iscell": d.loc[d["iscell"], "snr"].median(),
            "median_skew": d["skew"].median(), "median_event_frac": d["event_frac"].median(),
            "median_neu_corr": d["neu_corr"].median(),
            "matched_median_trace_corr": pairs["trace_corr"].median(),
            "matched_frac_higher_snr": float(np.mean(pairs["snr_a"] > pairs["snr_b"])) if lab == la
                                       else float(np.mean(pairs["snr_b"] > pairs["snr_a"])),
            "shift_dy": shift[0], "shift_dx": shift[1],
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("settings", help="YAML settings file (see settings/)")
    args = parser.parse_args()

    settings = load_settings(args.settings)
    os.makedirs(settings["output_dir"], exist_ok=True)
    rows = []
    for ses in settings["sessions"]:
        r = compare_session(ses, settings)
        if r:
            rows.extend(r)
    if rows:
        path = os.path.join(settings["output_dir"], "summary.csv")
        summary = pd.DataFrame(rows)
        summary.to_csv(path, index=False)
        with pd.option_context("display.width", 200, "display.max_columns", 30):
            print(summary.T.to_string(header=False))
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
